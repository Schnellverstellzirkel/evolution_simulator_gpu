//! CPU evaluation engine: 16 creatures with an identical body plan share one
//! SIMD group. Node and bone indices are uniform across the group, so every
//! per-lane access is a contiguous 16-wide load. Each statement is a loop over
//! the lanes with branch-free selects, which LLVM compiles to AVX-512 on this
//! machine (`target-cpu=native`). The equations mirror
//! `shaders/physics_creature.wgsl`; results differ from the GPU by rounding.
use crate::{
    config::Config,
    creature_kernel::GpuResult,
    evolution::Population,
    physics,
};
use crate::simd::F;
use rayon::prelude::*;
use std::collections::HashMap;

const L: usize = 16;
type V = [f32; L];
const ZERO: V = [0.0; L];

const MAX_MUSCLE_FORCE: f32 = 5.0;
const MAX_NODE_SPEED: f32 = 5.0;
const MAX_BONE_ANGULAR_SPEED: f32 = 15.0;
const MAX_BONE_TURN_COS: f32 = 0.992_197_7;
const MAX_BONE_TURN_TAN: f32 = 0.125_655_14;
const PI: f32 = 3.141_592_653_59;

#[derive(Clone, Copy)]
struct Muscle {
    a0: usize,
    a1: usize,
    b0: usize,
    b1: usize,
    /// Distinct endpoint nodes in endpoint order; `usize::MAX` marks a repeat.
    targets: [usize; 4],
}

#[derive(Clone, Copy, Default)]
struct MuscleLanes {
    anchor_a: V,
    anchor_b: V,
    short: V,
    long: V,
    inv_period: V,
    phase: V,
    duty: V,
    stiffness: V,
    inv_duty: V,
    inv_complement: V,
    /// Summed weight of each distinct endpoint, per lane.
    weights: [V; 4],
}

/// Up to 16 creatures with the same nodes, bones, and muscle attachments.
struct Group {
    /// Position of each real lane within the unit.
    slots: Vec<usize>,
    nodes: usize,
    bones: Vec<(usize, usize)>,
    muscles: Vec<Muscle>,
    lanes: Vec<MuscleLanes>,
    pos_x: Vec<V>,
    pos_y: Vec<V>,
    radius: Vec<V>,
    mass: Vec<V>,
    friction: Vec<V>,
    rest: Vec<V>,
    inv_a: Vec<V>,
    inv_b: Vec<V>,
    rad_a: Vec<V>,
    rad_b: Vec<V>,
}

/// The GPU's default polynomial for cos(pi * x) on [0, 1].
#[inline(always)]
fn fast_cos_pi(x: F) -> F {
    let y = (x - 0.5) * PI;
    let z = y * y;
    let mut p = z.mul_add(F::splat(-2.505_210_8e-8), F::splat(2.755_731_9e-6));
    p = z.mul_add(p, F::splat(-1.984_127e-4));
    p = z.mul_add(p, F::splat(8.333_334e-3));
    p = z.mul_add(p, F::splat(-1.666_666_7e-1));
    p = z.mul_add(p, F::splat(1.0));
    -(y * p)
}

#[inline(always)]
fn cos_pi(x: F, exact: bool) -> F {
    if exact {
        let v = x.to_array();
        F::load(&std::array::from_fn(|i| (PI * v[i]).cos()))
    } else {
        fast_cos_pi(x)
    }
}

/// Muscle parameters for one muscle across the 16 lanes.
#[derive(Clone, Copy)]
struct MuscleF {
    anchor_a: F,
    anchor_b: F,
    long: F,
    amplitude: F,
    inv_period: F,
    phase: F,
    duty: F,
    stiffness: F,
    inv_duty: F,
    inv_complement: F,
    weights: [F; 4],
}

#[inline(always)]
fn muscle_length(m: &MuscleF, time: F, exact: bool) -> F {
    let x = time * m.inv_period + m.phase;
    let phase = x - x.floor();
    let rising = cos_pi(phase * m.inv_duty, exact) * 0.5 + 0.5;
    let falling = F::splat(0.5) - cos_pi((phase - m.duty) * m.inv_complement, exact) * 0.5;
    let wave = F::select(phase.lt(m.duty), rising, falling);
    m.long - m.amplitude * (F::splat(1.0) - wave)
}

#[inline(always)]
fn limit_speed(x: &mut F, y: &mut F) {
    let speed = (*x * *x + *y * *y).sqrt();
    let scale = F::select(
        speed.gt(F::splat(MAX_NODE_SPEED)),
        F::splat(MAX_NODE_SPEED) / speed,
        F::splat(1.0),
    );
    *x = *x * scale;
    *y = *y * scale;
}

impl Group {
    fn build(pop: &Population, unit: &[usize], members: &[usize]) -> Self {
        let first = &pop.genomes[unit[members[0]]];
        let nodes = first.node_count;
        let bones: Vec<(usize, usize)> = pop.bones[first.bone_start..first.bone_start + first.bone_count]
            .iter()
            .map(|b| (b.a as usize, b.b as usize))
            .collect();
        let first_bones = &pop.bones[first.bone_start..first.bone_start + first.bone_count];
        let muscles: Vec<Muscle> = pop.muscles
            [first.muscle_start..first.muscle_start + first.muscle_count]
            .iter()
            .map(|m| {
                let ba = first_bones[m.bone_a as usize];
                let bb = first_bones[m.bone_b as usize];
                let ends = [ba.a as usize, ba.b as usize, bb.a as usize, bb.b as usize];
                let mut targets = [usize::MAX; 4];
                for e in 0..4 {
                    if !ends[..e].contains(&ends[e]) {
                        targets[e] = ends[e];
                    }
                }
                Muscle {
                    a0: ends[0],
                    a1: ends[1],
                    b0: ends[2],
                    b1: ends[3],
                    targets,
                }
            })
            .collect();
        let mut group = Group {
            slots: members.to_vec(),
            nodes,
            bones,
            lanes: vec![MuscleLanes::default(); muscles.len()],
            muscles,
            pos_x: vec![ZERO; nodes],
            pos_y: vec![ZERO; nodes],
            radius: vec![ZERO; nodes],
            mass: vec![ZERO; nodes],
            friction: vec![ZERO; nodes],
            rest: vec![ZERO; nodes - 1],
            inv_a: vec![ZERO; nodes - 1],
            inv_b: vec![ZERO; nodes - 1],
            rad_a: vec![ZERO; nodes - 1],
            rad_b: vec![ZERO; nodes - 1],
        };
        // Unused lanes repeat the first creature; their results are discarded.
        for l in 0..L {
            let g = &pop.genomes[unit[members[l.min(members.len() - 1)]]];
            let genes = &pop.nodes[g.node_start..g.node_start + g.node_count];
            let node_state: Vec<_> = genes.iter().map(physics::node).collect();
            for (j, n) in node_state.iter().enumerate() {
                group.pos_x[j][l] = n.pos[0];
                group.pos_y[j][l] = n.pos[1];
                group.radius[j][l] = n.radius;
                group.mass[j][l] = n.mass;
                group.friction[j][l] = n.friction;
            }
            let bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
            for (j, b) in bones.iter().enumerate() {
                let (a, bn) = (b.a as usize, b.b as usize);
                group.rest[j][l] = b.rest_length;
                group.inv_a[j][l] = 1.0 / node_state[a].mass;
                group.inv_b[j][l] = 1.0 / node_state[bn].mass;
                group.rad_a[j][l] = node_state[a].radius;
                group.rad_b[j][l] = node_state[bn].radius;
            }
            let muscles = &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
            for (j, m) in muscles.iter().enumerate() {
                let lanes = &mut group.lanes[j];
                lanes.anchor_a[l] = m.anchor_a;
                lanes.anchor_b[l] = m.anchor_b;
                // Holds the waveform amplitude, as in the GPU packing.
                lanes.short[l] = crate::creature_kernel::muscle_amplitude(m);
                lanes.long[l] = m.long;
                lanes.inv_period[l] = 1.0 / m.period;
                lanes.phase[l] = m.phase;
                lanes.duty[l] = m.duty;
                lanes.stiffness[l] = m.stiffness;
                lanes.inv_duty[l] = 1.0 / m.duty;
                lanes.inv_complement[l] = 1.0 / (1.0 - m.duty);
                let shape = group.muscles[j];
                for e in 0..4 {
                    let node = shape.targets[e];
                    if node == usize::MAX {
                        continue;
                    }
                    // Same sum order as the GPU's per-node gather.
                    let mut weight = 0.0f32;
                    if shape.a0 == node {
                        weight += 1.0 - m.anchor_a;
                    }
                    if shape.a1 == node {
                        weight += m.anchor_a;
                    }
                    if shape.b0 == node {
                        weight -= 1.0 - m.anchor_b;
                    }
                    if shape.b1 == node {
                        weight -= m.anchor_b;
                    }
                    lanes.weights[e][l] = weight;
                }
            }
        }
        group
    }

    /// Runs the full trial and returns one result per real lane.
    fn simulate(&self, cfg: &Config) -> Vec<GpuResult> {
        let n = self.nodes;
        let exact = std::env::var_os("EVOLUTION_EXACT_COS").is_some();
        let total_steps = physics::SETTLE + cfg.steps();
        let air = cfg.air_retention.sqrt();
        let ground_friction = cfg.ground_friction;
        let ground = cfg.ground;
        let load = |v: &Vec<V>| -> Vec<F> { v.iter().map(F::load).collect() };
        let mass = load(&self.mass);
        let radius = load(&self.radius);
        let friction = load(&self.friction);
        let rest = load(&self.rest);
        let inv_a = load(&self.inv_a);
        let inv_b = load(&self.inv_b);
        let share_a: Vec<F> = inv_a.iter().zip(&inv_b).map(|(&a, &b)| a / (a + b)).collect();
        let share_b: Vec<F> = inv_a.iter().zip(&inv_b).map(|(&a, &b)| b / (a + b)).collect();
        let inv_mass: Vec<F> = mass.iter().map(|&m| F::splat(1.0) / m).collect();
        let total_mass = mass.iter().fold(F::splat(0.0), |t, &m| t + m);
        let inv_total_mass = F::splat(1.0) / total_mass;
        let dt = 1.0f32 / 120.0;
        let rad_a = load(&self.rad_a);
        let rad_b = load(&self.rad_b);
        let muscles: Vec<MuscleF> = self
            .lanes
            .iter()
            .map(|m| {
                let duty = F::load(&m.duty);
                let long = F::load(&m.long);
                let inv_period = F::load(&m.inv_period);
                MuscleF {
                    anchor_a: F::load(&m.anchor_a),
                    anchor_b: F::load(&m.anchor_b),
                    long,
                    amplitude: F::load(&m.short),
                    inv_period,
                    phase: F::load(&m.phase),
                    duty,
                    stiffness: F::load(&m.stiffness),
                    inv_duty: F::load(&m.inv_duty),
                    inv_complement: F::load(&m.inv_complement),
                    weights: m.weights.map(|w| F::load(&w)),
                }
            })
            .collect();
        let zero = F::splat(0.0);
        let one = F::splat(1.0);
        let mut px = load(&self.pos_x);
        let mut py = load(&self.pos_y);
        let mut vx = vec![zero; n];
        let mut vy = vec![zero; n];
        let mut ox = vec![zero; n];
        let mut oy = vec![zero; n];
        let mut sx = vec![zero; n];
        let mut sy = vec![zero; n];
        let mut failed = vec![zero; n];
        let mut ground_contact = zero;
        let mut height_sum = zero;
        let mut low_center = F::splat(1e20);
        let mut high_center = F::splat(-1e20);
        let mut previous_center = [0.0f32; L];
        let mut extremum = [0.0f32; L];
        let mut trend = [0.0f32; L];
        let mut turns = [0.0f32; L];

        for tick in 0..total_steps {
            if tick == physics::SETTLE {
                let mut avg = zero;
                let mut low = F::splat(1e20);
                for j in 0..n {
                    avg += px[j] * mass[j];
                    low = low.min(py[j] - radius[j]);
                }
                let shift_x = avg * inv_total_mass;
                for j in 0..n {
                    px[j] -= shift_x;
                    py[j] -= low;
                    vx[j] = zero;
                    vy[j] = zero;
                }
            }
            ox.copy_from_slice(&px);
            oy.copy_from_slice(&py);
            sx.fill(zero);
            sy.fill(zero);

            let time_now = (tick.max(physics::SETTLE) - physics::SETTLE) as f32 * dt;
            let time = F::splat(time_now);
            let previous_time = F::splat((time_now - dt).max(0.0));
            for (shape, m) in self.muscles.iter().zip(&muscles) {
                let (aa, ab) = (m.anchor_a, m.anchor_b);
                let (ra, rb) = (one - aa, one - ab);
                let pax = px[shape.a0] * ra + px[shape.a1] * aa;
                let pay = py[shape.a0] * ra + py[shape.a1] * aa;
                let pbx = px[shape.b0] * rb + px[shape.b1] * ab;
                let pby = py[shape.b0] * rb + py[shape.b1] * ab;
                let vax = vx[shape.a0] * ra + vx[shape.a1] * aa;
                let vay = vy[shape.a0] * ra + vy[shape.a1] * aa;
                let vbx = vx[shape.b0] * rb + vx[shape.b1] * ab;
                let vby = vy[shape.b0] * rb + vy[shape.b1] * ab;
                let dx = pbx - pax;
                let dy = pby - pay;
                let inv_distance = F::splat(1.0) / (dx * dx + dy * dy).sqrt().max(F::splat(1e-6));
                let dir_x = dx * inv_distance;
                let dir_y = dy * inv_distance;
                let relative = (vbx - vax) * dir_x + (vby - vay) * dir_y;
                let target_speed = (muscle_length(m, time, exact)
                    - muscle_length(m, previous_time, exact))
                    * 120.0;
                let magnitude = (-(target_speed * m.stiffness) * 0.25 + relative * 0.15)
                    .max(F::splat(-MAX_MUSCLE_FORCE))
                    .min(F::splat(MAX_MUSCLE_FORCE));
                let push_x = dir_x * magnitude;
                let push_y = dir_y * magnitude;
                for e in 0..4 {
                    let node = shape.targets[e];
                    if node != usize::MAX {
                        sx[node] += push_x * m.weights[e];
                        sy[node] += push_y * m.weights[e];
                    }
                }
            }

            let gravity = if tick >= physics::SETTLE { cfg.gravity } else { 0.0 };
            let colliding = tick >= physics::SETTLE && ground;
            for j in 0..n {
                let mut vel_x = (vx[j] + (sx[j] * inv_mass[j]) * dt) * air;
                let mut vel_y = (vy[j] + (sy[j] * inv_mass[j] - gravity) * dt) * air;
                limit_speed(&mut vel_x, &mut vel_y);
                let pos_x = px[j] + vel_x * dt;
                let mut pos_y = py[j] + vel_y * dt;
                if colliding {
                    let below = pos_y.lt(radius[j]);
                    pos_y = F::select(below, pos_y + (radius[j] - pos_y), pos_y);
                    let vn = vel_y;
                    let hit = below & vn.lt(zero);
                    let slid_y = vel_y - vn;
                    let length = (vel_x * vel_x + slid_y * slid_y).sqrt().max(F::splat(1e-8));
                    let keep = (one - (-vn) * friction[j] * ground_friction / length).max(zero);
                    vel_x = F::select(hit, vel_x * keep, vel_x);
                    vel_y = F::select(hit, slid_y * keep, vel_y);
                }
                let limit = F::splat(1e6);
                let finite = pos_x.abs().lt(limit)
                    & pos_y.abs().lt(limit)
                    & vel_x.abs().lt(limit)
                    & vel_y.abs().lt(limit);
                let alive = failed[j].lt(F::splat(0.5));
                let fails = alive & !finite;
                failed[j] = F::select(fails, one, failed[j]);
                let update = alive & finite;
                px[j] = F::select(update, pos_x, F::select(fails, zero, px[j]));
                py[j] = F::select(update, pos_y, F::select(fails, zero, py[j]));
                vx[j] = F::select(update, vel_x, F::select(fails, zero, vx[j]));
                vy[j] = F::select(update, vel_y, F::select(fails, zero, vy[j]));
            }

            let tiny = F::splat(1e-6);
            for _ in 0..physics::solver_passes().0 {
                for (b, &(a, c)) in self.bones.iter().enumerate() {
                    let dx = px[c] - px[a];
                    let dy = py[c] - py[a];
                    let raw = (dx * dx + dy * dy).sqrt();
                    let distance = raw.max(tiny);
                    let error = distance - rest[b];
                    let valid = raw.gt(tiny);
                    let scale = error / distance;
                    let cx = F::select(valid, dx * scale, error);
                    let cy_ = F::select(valid, dy * scale, zero);
                    px[a] += cx * share_a[b];
                    px[c] -= cx * share_b[b];
                    let mut ay = py[a] + cy_ * share_a[b];
                    let mut cy = py[c] - cy_ * share_b[b];
                    if colliding {
                        ay = ay.max(rad_a[b]);
                        cy = cy.max(rad_b[b]);
                    }
                    py[a] = ay;
                    py[c] = cy;
                }
            }

            let mut target_x = zero;
            let mut target_y = zero;
            sx.copy_from_slice(&px);
            sy.copy_from_slice(&py);
            for j in 0..n {
                target_x += px[j] * mass[j];
                target_y += py[j] * mass[j];
            }
            let turn_cos = F::splat(MAX_BONE_TURN_COS);
            for (b, &(a, c)) in self.bones.iter().enumerate() {
                let dx = sx[c] - sx[a];
                let dy = sy[c] - sy[a];
                let raw = (dx * dx + dy * dy).sqrt();
                let valid = raw.gt(tiny);
                let inv_raw = one / raw.max(tiny);
                let mut dir_x = F::select(valid, dx * inv_raw, one);
                let mut dir_y = F::select(valid, dy * inv_raw, zero);
                let pdx = ox[c] - ox[a];
                let pdy = oy[c] - oy[a];
                let previous_length = (pdx * pdx + pdy * pdy).sqrt();
                let inv_previous = one / previous_length;
                let prev_x = pdx * inv_previous;
                let prev_y = pdy * inv_previous;
                let turn = previous_length.gt(tiny) & (prev_x * dir_x + prev_y * dir_y).lt(turn_cos);
                if turn.any() {
                    let cross = prev_x * dir_y - prev_y * dir_x;
                    let sign = F::select(cross.lt(zero), F::splat(-1.0), one);
                    let tx = prev_x - prev_y * sign * MAX_BONE_TURN_TAN;
                    let ty = prev_y + prev_x * sign * MAX_BONE_TURN_TAN;
                    let length = (tx * tx + ty * ty).sqrt();
                    dir_x = F::select(turn, tx / length, dir_x);
                    dir_y = F::select(turn, ty / length, dir_y);
                }
                px[c] = px[a] + dir_x * rest[b];
                py[c] = py[a] + dir_y * rest[b];
            }
            let mut current_x = zero;
            let mut current_y = zero;
            for j in 0..n {
                current_x += px[j] * mass[j];
                current_y += py[j] * mass[j];
            }
            let shift_x = (target_x - current_x) * inv_total_mass;
            let shift_y = (target_y - current_y) * inv_total_mass;
            let mut lift = zero;
            for j in 0..n {
                px[j] += shift_x;
                py[j] += shift_y;
                lift = lift.max(radius[j] - py[j]);
            }
            if colliding {
                for y in &mut py {
                    *y += lift;
                }
            }
            for _ in 0..physics::solver_passes().1 {
                for (b, &(a, c)) in self.bones.iter().enumerate() {
                    let dx = px[c] - px[a];
                    let dy = py[c] - py[a];
                    let length = (dx * dx + dy * dy).sqrt().max(tiny);
                    let inv_length = one / length;
                    let dir_x = dx * inv_length;
                    let dir_y = dy * inv_length;
                    let (sa, sb) = (share_a[b], share_b[b]);
                    let radial = (vx[c] - vx[a]) * dir_x + (vy[c] - vy[a]) * dir_y;
                    let vax = vx[a] + dir_x * radial * sa;
                    let vay = vy[a] + dir_y * radial * sa;
                    let vcx = vx[c] - dir_x * radial * sb;
                    let vcy = vy[c] - dir_y * radial * sb;
                    let (tx, ty) = (-dir_y, dir_x);
                    let tangent = (vcx - vax) * tx + (vcy - vay) * ty;
                    let limit = length * MAX_BONE_ANGULAR_SPEED;
                    let limited = tangent.max(-limit).min(limit);
                    let angular = tangent - limited;
                    vx[a] = vax + tx * angular * sa;
                    vy[a] = vay + ty * angular * sa;
                    vx[c] = vcx - tx * angular * sb;
                    vy[c] = vcy - ty * angular * sb;
                }
                for j in 0..n {
                    limit_speed(&mut vx[j], &mut vy[j]);
                    if colliding {
                        let resting = py[j].le(radius[j] + 1e-5);
                        vy[j] = F::select(resting, vy[j].max(zero), vy[j]);
                    }
                }
            }

            if tick >= physics::SETTLE {
                let mut center = zero;
                let mut contacts = zero;
                let mut low = F::splat(1e20);
                let mut high = F::splat(-1e20);
                for j in 0..n {
                    let y = py[j];
                    center += y;
                    low = low.min(y - radius[j]);
                    high = high.max(y + radius[j]);
                    if ground {
                        contacts += F::select(y.le(radius[j] + 0.002), one, zero);
                    }
                }
                let center = center * (1.0 / n as f32);
                ground_contact += contacts;
                height_sum += high - low;
                low_center = low_center.min(center);
                high_center = high_center.max(center);
                let sample = tick == physics::SETTLE || (tick - physics::SETTLE) % 4 == 0;
                if sample {
                    let c = center.to_array();
                    for l in 0..L {
                        let c = c[l];
                        if tick == physics::SETTLE {
                            previous_center[l] = c;
                            extremum[l] = c;
                            trend[l] = 0.0;
                            turns[l] = 0.0;
                            continue;
                        }
                        let delta = c - previous_center[l];
                        if trend[l] == 0.0 {
                            if delta.abs() > 0.0005 {
                                trend[l] = if delta > 0.0 { 1.0 } else { -1.0 };
                                extremum[l] = c;
                            }
                        } else if trend[l] > 0.0 {
                            if c > extremum[l] {
                                extremum[l] = c;
                            } else if extremum[l] - c > 0.005 {
                                turns[l] += 1.0;
                                trend[l] = -1.0;
                                extremum[l] = c;
                            }
                        } else if c < extremum[l] {
                            extremum[l] = c;
                        } else if c - extremum[l] > 0.005 {
                            turns[l] += 1.0;
                            trend[l] = 1.0;
                            extremum[l] = c;
                        }
                        previous_center[l] = c;
                    }
                }
            }
        }

        let timed_steps = total_steps.saturating_sub(physics::SETTLE).max(1) as f32;
        let timed = total_steps > physics::SETTLE;
        let px: Vec<[f32; L]> = px.iter().map(|v| v.to_array()).collect();
        let failed: Vec<[f32; L]> = failed.iter().map(|v| v.to_array()).collect();
        let ground_contact = ground_contact.to_array();
        let height_sum = height_sum.to_array();
        let low_center = low_center.to_array();
        let high_center = high_center.to_array();
        (0..self.slots.len())
            .map(|l| {
                let mut score = 0.0f32;
                let mut mass_sum = 0.0f32;
                let mut failures = 0.0f32;
                for j in 0..n {
                    score += px[j][l] * self.mass[j][l];
                    mass_sum += self.mass[j][l];
                    failures += failed[j][l];
                }
                let fitness = if failures > 0.0 {
                    -1e20
                } else {
                    let mean_height = height_sum[l] / timed_steps;
                    let contact_fraction = ground_contact[l] / (timed_steps * n as f32);
                    let posture = ((mean_height - 0.25) / 0.75).clamp(0.0, 1.0);
                    let stepping = ((0.95 - contact_fraction) / 0.20).clamp(0.0, 1.0);
                    score / mass_sum * posture * stepping
                };
                GpuResult {
                    fitness,
                    ground_contact: ground_contact[l],
                    vertical_oscillation: if timed {
                        (high_center[l] - low_center[l]).max(0.0)
                    } else {
                        0.0
                    },
                    gait_frequency: if timed {
                        turns[l] * 0.5 / ((total_steps - physics::SETTLE) as f32 / 120.0)
                    } else {
                        0.0
                    },
                    previous_center_y: previous_center[l],
                    vertical_extremum: extremum[l],
                    vertical_trend: trend[l],
                    gait_turns: turns[l],
                    height_sum: height_sum[l],
                }
            })
            .collect()
    }
}

/// Groups a unit's creatures by exact body plan, 16 per SIMD group.
fn build_groups(pop: &Population, unit: &[usize]) -> Vec<Group> {
    let mut plans: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
    for (slot, &i) in unit.iter().enumerate() {
        let g = &pop.genomes[i];
        let mut key = Vec::with_capacity(1 + 2 * g.bone_count + 2 * g.muscle_count);
        key.push(g.node_count as u32);
        key.extend(
            pop.bones[g.bone_start..g.bone_start + g.bone_count]
                .iter()
                .flat_map(|b| [b.a, b.b]),
        );
        key.push(u32::MAX);
        key.extend(
            pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count]
                .iter()
                .flat_map(|m| [m.bone_a, m.bone_b]),
        );
        plans.entry(key).or_default().push(slot);
    }
    let chunks: Vec<Vec<usize>> = plans
        .into_values()
        .flat_map(|slots| slots.chunks(L).map(<[usize]>::to_vec).collect::<Vec<_>>())
        .collect();
    chunks
        .par_iter()
        .map(|members| Group::build(pop, unit, members))
        .collect()
}

/// Evaluates every creature of `unit` and returns results in unit order.
pub fn evaluate(unit: &Population, cfg: &Config) -> Vec<GpuResult> {
    let indices: Vec<usize> = (0..unit.genomes.len()).collect();
    let groups = build_groups(unit, &indices);
    let per_group: Vec<Vec<GpuResult>> = groups.par_iter().map(|g| g.simulate(cfg)).collect();
    let mut results = vec![GpuResult::default(); indices.len()];
    for (group, values) in groups.iter().zip(per_group) {
        for (&slot, value) in group.slots.iter().zip(values) {
            results[slot] = value;
        }
    }
    results
}
