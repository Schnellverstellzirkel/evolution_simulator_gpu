//! CPU evaluation engine: 16 creatures with an identical body plan share one
//! SIMD group. Node and bone indices are uniform across the group, so every
//! per-lane access is a contiguous 16-wide load. Each statement is a loop over
//! the lanes with branch-free selects, which LLVM compiles to AVX-512 on this
//! machine (`target-cpu=native`). The equations mirror
//! `shaders/physics_creature.wgsl`; results differ from the GPU by rounding.
use crate::simd::F;
use crate::{config::Config, creature_kernel::GpuResult, evolution::Population, physics};
use rayon::prelude::*;
use std::collections::HashMap;

/// Diagnostic ledger of horizontal momentum changes for lane 0 of each group,
/// enabled by `EVOLUTION_LEDGER`: [integration speed cap, ground contact,
/// velocity-pass speed cap, velocity-pass constraints, projection/rebuild
/// center-of-mass shift x mass / dt, muscle forces].
pub static LEDGER: std::sync::Mutex<[f64; 6]> = std::sync::Mutex::new([0.0; 6]);

const L: usize = 16;
/// Muscle energy: stored work (J), recovery per second, and the drive left
/// when exhausted. Mirrors the GPU kernel.
const TIRED_DRIVE: f32 = 0.2;
type V = [f32; L];
const ZERO: V = [0.0; L];

// Same f32 as the kernel's 3.14159265359 literal.
const PI: f32 = std::f32::consts::PI;

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
    /// Touchdown sensor endpoint (0-3) or `NO_SENSOR`, and reset phase.
    sensor: [u32; L],
    reset: V,
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
    /// Joint reference node of each bone (the same across a body plan).
    joint_reference: Vec<Option<usize>>,
    /// Per bone: range center (x, y), half range (cos, sin), child share, and
    /// child and reference mass fractions, as in `physics::Joint`.
    joint: Vec<[V; 7]>,
    inv_a: Vec<V>,
    inv_b: Vec<V>,
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

/// Height and slope of the rough ground across the lanes; mirrors
/// `physics::terrain`.
#[inline(always)]
fn terrain(x: F, amplitude: f32) -> (F, F) {
    let mut height = F::splat(0.0);
    let mut slope = F::splat(0.0);
    for (wavelength, weight, offset) in physics::TERRAIN_WAVES {
        let t = x * (1.0 / wavelength) + offset;
        let u = t - t.floor();
        let w = u * (F::splat(1.0) - u);
        height += w * w * (weight * 16.0);
        slope += w * (F::splat(1.0) - u * 2.0) * (weight * 32.0 * (1.0 / wavelength));
    }
    (height * amplitude, slope * amplitude)
}

/// Clearance a touching node must reach to count as a lifted foot.
const LIFT_CLEARANCE: f32 = 0.01;

/// Change of (x, y) under a small rotation; mirrors the kernel's
/// `rotate_small`.
#[inline(always)]
fn rotate_small(x: F, y: F, angle: F) -> (F, F) {
    let a2 = angle * angle;
    let c = F::splat(1.0) - a2 * (F::splat(0.5) - a2 * (1.0 / 24.0));
    let s = angle * (F::splat(1.0) - a2 * (F::splat(1.0 / 6.0) - a2 * (1.0 / 120.0)));
    (x * c - y * s - x, x * s + y * c - y)
}

#[inline(always)]
fn limit_speed(x: &mut F, y: &mut F, max: F) {
    let speed = (*x * *x + *y * *y).sqrt();
    let scale = F::select(speed.gt(max), max / speed, F::splat(1.0));
    *x = *x * scale;
    *y = *y * scale;
}

impl Group {
    fn build(pop: &Population, unit: &[usize], members: &[usize]) -> Self {
        let first = &pop.genomes[unit[members[0]]];
        let nodes = first.node_count;
        let bones: Vec<(usize, usize)> = pop.bones
            [first.bone_start..first.bone_start + first.bone_count]
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
            joint_reference: vec![None; nodes - 1],
            joint: vec![[ZERO; 7]; nodes - 1],
            inv_a: vec![ZERO; nodes - 1],
            inv_b: vec![ZERO; nodes - 1],
        };
        // Unused lanes repeat the first creature; their results are discarded.
        for l in 0..L {
            let g = &pop.genomes[unit[members[l.min(members.len() - 1)]]];
            let genes = &pop.nodes[g.node_start..g.node_start + g.node_count];
            let bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
            let node_state = physics::body(genes, bones);
            for (j, n) in node_state.iter().enumerate() {
                group.pos_x[j][l] = n.pos[0];
                group.pos_y[j][l] = n.pos[1];
                group.radius[j][l] = n.radius;
                group.mass[j][l] = n.mass;
                group.friction[j][l] = n.friction;
            }
            let joints = physics::joints(genes, bones);
            for (j, b) in bones.iter().enumerate() {
                let (a, bn) = (b.a as usize, b.b as usize);
                group.rest[j][l] = b.rest_length;
                let joint = joints[j];
                group.joint_reference[j] = joint.reference;
                let values = [
                    joint.center[0],
                    joint.center[1],
                    joint.half[0],
                    joint.half[1],
                    joint.child_share,
                    joint.child_mass,
                    joint.reference_mass,
                ];
                for (field, value) in values.into_iter().enumerate() {
                    group.joint[j][field][l] = value;
                }
                group.inv_a[j][l] = 1.0 / node_state[a].mass;
                group.inv_b[j][l] = 1.0 / node_state[bn].mass;
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
                lanes.sensor[l] = m.sensor;
                lanes.reset[l] = m.reset;
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
        self.run(cfg, None)
    }

    /// Runs the trial; `record` receives lane 0's node positions before every
    /// step and after the last one (index = steps completed).
    fn run(&self, cfg: &Config, mut record: Option<&mut Vec<Vec<[f32; 2]>>>) -> Vec<GpuResult> {
        let n = self.nodes;
        let exact = std::env::var_os("EVOLUTION_EXACT_COS").is_some();
        let fidelity = cfg.fidelity();
        let total_steps = fidelity.settle() + cfg.steps();
        let air = fidelity.air_per_step(cfg.air_retention);
        let settle = fidelity.settle();
        let sample_interval = fidelity.sample_interval();
        let (turn_cos_limit, turn_tan) = fidelity.turn_limits();
        let ground_friction = cfg.ground_friction;
        let ground = cfg.ground;
        let limits = physics::limits();
        let max_node_speed = F::splat(limits.node_speed);
        let max_force = F::splat(limits.muscle_force);
        let max_spin = F::splat(limits.bone_spin);
        let amplitude = physics::terrain_amplitude(cfg.terrain);
        let rough = amplitude > 0.0;
        let load = |v: &Vec<V>| -> Vec<F> { v.iter().map(F::load).collect() };
        let mass = load(&self.mass);
        let radius = load(&self.radius);
        let friction = load(&self.friction);
        let rest = load(&self.rest);
        let joint: Vec<[F; 7]> = self.joint.iter().map(|j| j.map(|v| F::load(&v))).collect();
        let inv_a = load(&self.inv_a);
        let inv_b = load(&self.inv_b);
        let share_a: Vec<F> = inv_a
            .iter()
            .zip(&inv_b)
            .map(|(&a, &b)| a / (a + b))
            .collect();
        let share_b: Vec<F> = share_a.iter().map(|&a| F::splat(1.0) - a).collect();
        let inv_mass: Vec<F> = mass.iter().map(|&m| F::splat(1.0) / m).collect();
        let total_mass = mass.iter().fold(F::splat(0.0), |t, &m| t + m);
        let inv_total_mass = F::splat(1.0) / total_mass;
        let dt = fidelity.dt();
        let ledger_on = std::env::var_os("EVOLUTION_LEDGER").is_some();
        let lane0_mass: Vec<f32> = mass.iter().map(|m| m.to_array()[0]).collect();
        let momentum = |v: &[F]| -> f32 {
            v.iter()
                .zip(&lane0_mass)
                .map(|(x, m)| x.to_array()[0] * m)
                .sum()
        };
        let com = |p: &[F]| -> f32 {
            p.iter()
                .zip(&lane0_mass)
                .map(|(x, m)| x.to_array()[0] * m)
                .sum()
        };
        let mut ledger = [0.0f64; 6];
        let rate = fidelity.rate as f32;
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
        // Lowest allowed center height of each node for the current step.
        let mut floor = radius.clone();
        let mut ground_contact = zero;
        let mut height_sum = zero;
        let mut low_center = F::splat(1e20);
        let mut high_center = F::splat(-1e20);
        let mut previous_center = [0.0f32; L];
        let mut extremum = [0.0f32; L];
        let mut trend = [0.0f32; L];
        let mut turns = [0.0f32; L];
        let mut contact_bits = [0u64; L];
        let mut lift_bits = [0u64; L];
        let mut grounded_before = [0u64; L];
        let mut offsets = vec![zero; muscles.len()];
        let mut energies = vec![one; muscles.len()];
        // The head is node 0 and its neck base is bone 0's child. A creature
        // falls when the head drops below the base: its muscles go limp and
        // its fitness keeps the distance at the fall.
        let neck_base = self.bones[0].1;
        let mut fall_time = zero;
        let mut fall_x = zero;

        let snapshot = |px: &[F], py: &[F]| -> Vec<[f32; 2]> {
            px.iter()
                .zip(py)
                .map(|(x, y)| [x.to_array()[0], y.to_array()[0]])
                .collect()
        };
        for tick in 0..total_steps {
            if let Some(frames) = record.as_mut() {
                frames.push(snapshot(&px, &py));
            }
            if tick == settle {
                let mut avg = zero;
                let mut low = F::splat(1e20);
                for j in 0..n {
                    avg += px[j] * mass[j];
                }
                let shift_x = avg * inv_total_mass;
                for j in 0..n {
                    let ground_y = if rough {
                        terrain(px[j] - shift_x, amplitude).0
                    } else {
                        zero
                    };
                    low = low.min(py[j] - radius[j] - ground_y);
                }
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

            let time_now = (tick.max(settle) - settle) as f32 * dt;
            let time = F::splat(time_now);
            let previous_time = F::splat((time_now - dt).max(0.0));
            for (index, (shape, m)) in self.muscles.iter().zip(&muscles).enumerate() {
                if tick == settle {
                    energies[index] = one;
                }
                let mut clocked = *m;
                clocked.phase = m.phase + offsets[index];
                let m = &clocked;
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
                let target_speed =
                    (muscle_length(m, time, exact) - muscle_length(m, previous_time, exact)) * rate;
                // A tired muscle drives weaker and slower.
                let vigor = F::splat(TIRED_DRIVE) + energies[index] * (1.0 - TIRED_DRIVE);
                let magnitude = (-(target_speed * m.stiffness) * 0.25 * vigor + relative * 0.15)
                    .max(-max_force)
                    .min(max_force);
                let magnitude = F::select(fall_time.gt(zero), zero, magnitude);
                if tick >= settle {
                    let work = (magnitude * relative).abs() * dt;
                    energies[index] = (energies[index] - work * (1.0 / limits.muscle_energy)
                        + (one - energies[index]) * (limits.muscle_recovery * dt))
                        .max(zero)
                        .min(one);
                }
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

            let gravity = if tick >= settle { cfg.gravity } else { 0.0 };
            let colliding = tick >= settle && ground;
            // The speed cap must not push the body: spread the momentum it
            // removes back over all nodes.
            let (mut removed_x, mut removed_y) = (zero, zero);
            for j in 0..n {
                let free_x = (vx[j] + (sx[j] * inv_mass[j]) * dt) * air;
                let free_y = (vy[j] + (sy[j] * inv_mass[j] - gravity) * dt) * air;
                if ledger_on && colliding {
                    ledger[5] +=
                        f64::from((sx[j] * inv_mass[j] * dt).to_array()[0] * lane0_mass[j]);
                }
                let (mut cap_x, mut cap_y) = (free_x, free_y);
                limit_speed(&mut cap_x, &mut cap_y, max_node_speed);
                let alive = failed[j].lt(F::splat(0.5));
                removed_x += F::select(alive, (free_x - cap_x) * mass[j], zero);
                removed_y += F::select(alive, (free_y - cap_y) * mass[j], zero);
                sx[j] = cap_x;
                sy[j] = cap_y;
            }
            let (fix_x, fix_y) = (removed_x * inv_total_mass, removed_y * inv_total_mass);
            for j in 0..n {
                let vel_x = sx[j] + fix_x;
                let vel_y = sy[j] + fix_y;
                let pos_x = px[j] + vel_x * dt;
                let pos_y = py[j] + vel_y * dt;
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
                // Velocities are rebuilt from positions after the solve; keep the
                // predicted height to measure how far the ground pushed the node.
                vx[j] = py[j];
                let _ = vel_x;
            }
            if colliding {
                for j in 0..n {
                    if rough {
                        // Push out along the ground normal, so bumps resist sliding.
                        let (height, slope) = terrain(px[j], amplitude);
                        let secant_sq = one + slope * slope;
                        floor[j] = height + radius[j] * secant_sq.sqrt();
                        let depth = (floor[j] - py[j]).max(zero) / secant_sq;
                        px[j] -= slope * depth;
                        py[j] += depth;
                    } else {
                        py[j] = py[j].max(radius[j]);
                    }
                }
            }

            let tiny = F::splat(1e-6);
            let com_before = com(&px);
            for _ in 0..fidelity.bone_passes {
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
                        ay = ay.max(floor[a]);
                        cy = cy.max(floor[c]);
                    }
                    py[a] = ay;
                    py[c] = cy;
                }
                // Joint ranges: a bone may not turn past its evolved limits
                // against its reference bone, so no joint can spin like a wheel.
                for (b, &(pivot, child)) in self.bones.iter().enumerate() {
                    let Some(reference) = self.joint_reference[b] else {
                        continue;
                    };
                    let [
                        center_x,
                        center_y,
                        cos_half,
                        sin_half,
                        share,
                        child_mass,
                        reference_mass,
                    ] = joint[b];
                    let (nx, ny) = (px[pivot], py[pivot]);
                    let (ux, uy) = (px[reference] - nx, py[reference] - ny);
                    let (vx_, vy_) = (px[child] - nx, py[child] - ny);
                    let norm = ((ux * ux + uy * uy) * (vx_ * vx_ + vy_ * vy_)).sqrt();
                    let inv_norm = one / norm.max(F::splat(1e-12));
                    let rx = (ux * vx_ + uy * vy_) * inv_norm;
                    let ry = (ux * vy_ - uy * vx_) * inv_norm;
                    let zx = rx * center_x + ry * center_y;
                    let zy = ry * center_x - rx * center_y;
                    let outside = zx.lt(cos_half) & !norm.lt(F::splat(1e-12));
                    if !outside.any() {
                        continue;
                    }
                    let side = F::select(zy.lt(zero), F::splat(-1.0), one);
                    let abs_zy = zy.abs();
                    let sin_excess = abs_zy * cos_half - zx * sin_half;
                    let cos_excess = zx * cos_half + abs_zy * sin_half;
                    let series =
                        (sin_excess * (one + sin_excess * sin_excess * (1.0 / 6.0))).min(one);
                    let excess = F::select(cos_excess.gt(zero), series, one);
                    let excess = F::select(outside, excess, zero);
                    // A node resting on the ground cannot give way, so the
                    // other side of the joint takes the whole correction.
                    let share = if colliding {
                        let child_down = py[child].le(floor[child] + 1e-4);
                        let reference_down = py[reference].le(floor[reference] + 1e-4);
                        F::select(
                            child_down & !reference_down,
                            zero,
                            F::select(reference_down & !child_down, one, share),
                        )
                    } else {
                        share
                    };
                    let (dvx, dvy) = rotate_small(vx_, vy_, -side * excess * share);
                    let (dux, duy) = rotate_small(ux, uy, side * excess * (one - share));
                    let shift_x = dvx * child_mass + dux * reference_mass;
                    let shift_y = dvy * child_mass + duy * reference_mass;
                    px[pivot] = nx - shift_x;
                    px[child] += dvx - shift_x;
                    px[reference] += dux - shift_x;
                    let mut new_n = ny - shift_y;
                    let mut new_c = py[child] + dvy - shift_y;
                    let mut new_q = py[reference] + duy - shift_y;
                    if colliding {
                        new_n = new_n.max(floor[pivot]);
                        new_c = new_c.max(floor[child]);
                        new_q = new_q.max(floor[reference]);
                    }
                    py[pivot] = new_n;
                    py[child] = new_c;
                    py[reference] = new_q;
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
            let turn_cos = F::splat(turn_cos_limit);
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
                let turn =
                    previous_length.gt(tiny) & (prev_x * dir_x + prev_y * dir_y).lt(turn_cos);
                if turn.any() {
                    let cross = prev_x * dir_y - prev_y * dir_x;
                    let sign = F::select(cross.lt(zero), F::splat(-1.0), one);
                    let tx = prev_x - prev_y * sign * turn_tan;
                    let ty = prev_y + prev_x * sign * turn_tan;
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
                lift = lift.max(floor[j] - py[j]);
            }
            if colliding {
                for y in &mut py {
                    *y += lift;
                }
            }
            if ledger_on && colliding {
                ledger[4] += f64::from((com(&px) - com_before) / dt);
            }
            // Velocity is the actual movement over the step. Ground friction uses
            // the real upward push the node received, so grip needs real pressure.
            for j in 0..n {
                let predicted_y = vx[j];
                let mut vel_x = (px[j] - ox[j]) * rate;
                let vel_y = (py[j] - oy[j]) * rate;
                if colliding {
                    let contact = py[j].le(floor[j] + 1e-4);
                    let push = (py[j] - predicted_y).max(zero);
                    let max_change = friction[j] * ground_friction * push * rate;
                    let reduced = vel_x - vel_x.max(-max_change).min(max_change);
                    if ledger_on {
                        let m = f64::from(lane0_mass[j]);
                        let chosen = F::select(contact, reduced, vel_x);
                        ledger[1] +=
                            (f64::from(chosen.to_array()[0]) - f64::from(vel_x.to_array()[0])) * m;
                    }
                    vel_x = F::select(contact, reduced, vel_x);
                }
                let alive = failed[j].lt(F::splat(0.5));
                vx[j] = F::select(alive, vel_x, zero);
                vy[j] = F::select(alive, vel_y, zero);
            }
            for _ in 0..fidelity.velocity_passes {
                let before = momentum(&vx);
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
                    let limit = length * max_spin;
                    let limited = tangent.max(-limit).min(limit);
                    let angular = tangent - limited;
                    vx[a] = vax + tx * angular * sa;
                    vy[a] = vay + ty * angular * sa;
                    vx[c] = vcx - tx * angular * sb;
                    vy[c] = vcy - ty * angular * sb;
                }
                let middle = momentum(&vx);
                let (mut removed_x, mut removed_y) = (zero, zero);
                for j in 0..n {
                    let (free_x, free_y) = (vx[j], vy[j]);
                    limit_speed(&mut vx[j], &mut vy[j], max_node_speed);
                    removed_x += (free_x - vx[j]) * mass[j];
                    removed_y += (free_y - vy[j]) * mass[j];
                }
                let (fix_x, fix_y) = (removed_x * inv_total_mass, removed_y * inv_total_mass);
                for j in 0..n {
                    vx[j] += fix_x;
                    vy[j] += fix_y;
                    if colliding {
                        let resting = py[j].le(floor[j] + 1e-5);
                        vy[j] = F::select(resting, vy[j].max(zero), vy[j]);
                    }
                }
                if ledger_on && colliding {
                    ledger[3] += f64::from(middle - before);
                    ledger[2] += f64::from(momentum(&vx) - middle);
                }
            }

            let mut grounded_now = [0u64; L];
            if tick >= settle {
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
                        let touching = y.le(floor[j] + 0.002);
                        contacts += F::select(touching, one, zero);
                        let bits = F::select(touching, one, zero).to_array();
                        // A foot must leave the ground after touching it; a
                        // dragged node never does.
                        let lifted =
                            F::select(y.gt(floor[j] + LIFT_CLEARANCE), one, zero).to_array();
                        for l in 0..L {
                            if bits[l] > 0.0 {
                                grounded_now[l] |= 1 << j;
                                contact_bits[l] |= 1 << j;
                            } else if lifted[l] > 0.0 {
                                lift_bits[l] |= contact_bits[l] & (1 << j);
                            }
                        }
                    }
                }
                // Muscles sensing a touchdown restart their rhythm at their reset
                // phase from the next step on.
                let next_time = time_now + dt;
                for l in 0..L {
                    let down = grounded_now[l] & !grounded_before[l];
                    grounded_before[l] = grounded_now[l];
                    if down == 0 || tick == settle {
                        continue;
                    }
                    for (index, shape) in self.muscles.iter().enumerate() {
                        let lanes = &self.lanes[index];
                        let sensor = lanes.sensor[l];
                        if sensor == crate::evolution::NO_SENSOR {
                            continue;
                        }
                        let node = [shape.a0, shape.a1, shape.b0, shape.b1][sensor as usize];
                        if down >> node & 1 == 1 {
                            let clock = next_time * lanes.inv_period[l] + lanes.phase[l];
                            let reset = lanes.reset[l] - clock;
                            let mut values = offsets[index].to_array();
                            values[l] = reset - reset.floor();
                            offsets[index] = F::load(&values);
                        }
                    }
                }
                // A joint forced far past its range breaks, which also ends
                // the trial.
                let mut broken = zero.lt(zero);
                for (b, &(pivot, child)) in self.bones.iter().enumerate() {
                    let Some(reference) = self.joint_reference[b] else {
                        continue;
                    };
                    let [center_x, center_y, cos_half, sin_half, ..] = joint[b];
                    let (ux, uy) = (px[reference] - px[pivot], py[reference] - py[pivot]);
                    let (vx_, vy_) = (px[child] - px[pivot], py[child] - py[pivot]);
                    let norm = ((ux * ux + uy * uy) * (vx_ * vx_ + vy_ * vy_)).sqrt();
                    let inv_norm = one / norm.max(F::splat(1e-12));
                    let rx = (ux * vx_ + uy * vy_) * inv_norm;
                    let ry = (ux * vy_ - uy * vx_) * inv_norm;
                    let (sin_break, cos_break) = physics::JOINT_BREAK.sin_cos();
                    let limit = cos_half * cos_break - sin_half * sin_break;
                    broken = broken
                        | ((rx * center_x + ry * center_y).lt(limit) & norm.gt(F::splat(1e-12)));
                }
                let falls = fall_time.le(zero) & (py[0].lt(py[neck_base]) | broken);
                if falls.any() {
                    let mut x = zero;
                    for j in 0..n {
                        x += px[j] * mass[j];
                    }
                    fall_x = F::select(falls, x * inv_total_mass, fall_x);
                    fall_time = F::select(falls, F::splat(time_now + dt), fall_time);
                }
                let center = center * (1.0 / n as f32);
                ground_contact += contacts;
                height_sum += high - low;
                low_center = low_center.min(center);
                high_center = high_center.max(center);
                let sample = tick == settle || (tick - settle).is_multiple_of(sample_interval);
                if sample {
                    let c = center.to_array();
                    for l in 0..L {
                        let c = c[l];
                        if tick == settle {
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

        if let Some(frames) = record.as_mut() {
            frames.push(snapshot(&px, &py));
        }
        if ledger_on {
            let mut total = LEDGER.lock().unwrap();
            for (t, v) in total.iter_mut().zip(ledger) {
                *t += v;
            }
        }
        let timed = total_steps > settle;
        let px: Vec<[f32; L]> = px.iter().map(|v| v.to_array()).collect();
        let failed: Vec<[f32; L]> = failed.iter().map(|v| v.to_array()).collect();
        let ground_contact = ground_contact.to_array();
        let height_sum = height_sum.to_array();
        let low_center = low_center.to_array();
        let high_center = high_center.to_array();
        let fall_time = fall_time.to_array();
        let fall_x = fall_x.to_array();
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
                // Fitness is distance only; gait style is left to the niches.
                let fitness = if failures > 0.0 {
                    -1e20
                } else if fall_time[l] > 0.0 {
                    fall_x[l]
                } else {
                    score / mass_sum
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
                        turns[l] * 0.5 / ((total_steps - settle) as f32 / rate)
                    } else {
                        0.0
                    },
                    previous_center_y: previous_center[l],
                    vertical_extremum: extremum[l],
                    vertical_trend: trend[l],
                    gait_turns: turns[l],
                    height_sum: height_sum[l],
                    contact_lo: f32::from_bits(contact_bits[l] as u32),
                    contact_hi: f32::from_bits((contact_bits[l] >> 32) as u32),
                    lift_lo: f32::from_bits(lift_bits[l] as u32),
                    lift_hi: f32::from_bits((lift_bits[l] >> 32) as u32),
                    ground_lo: f32::from_bits(grounded_before[l] as u32),
                    ground_hi: f32::from_bits((grounded_before[l] >> 32) as u32),
                    fall_time: fall_time[l],
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

/// Node positions of one creature's full trial with the evaluation physics:
/// entry `t` is the state after `t` steps. The replay shows exactly this.
pub fn trajectory(creature: &crate::evolution::Creature, cfg: &Config) -> Vec<Vec<[f32; 2]>> {
    let mut pop = Population::default();
    pop.push(creature.clone());
    let group = Group::build(&pop, &[0], &[0]);
    let mut frames = Vec::with_capacity((cfg.fidelity().settle() + cfg.steps() + 1) as usize);
    group.run(cfg, Some(&mut frames));
    frames
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
