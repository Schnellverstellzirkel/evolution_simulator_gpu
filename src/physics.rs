use crate::{
    config::Config,
    evolution::{Bone, Creature, FAILED, Muscle, NodeGene},
};
/// Physics steps per second. `EVOLUTION_PHYSICS_RATE` overrides it for
/// experiments; every engine, the replay, and trial lengths follow it.
pub fn rate() -> u32 {
    static RATE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *RATE.get_or_init(|| {
        std::env::var("EVOLUTION_PHYSICS_RATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&r: &u32| (15..=480).contains(&r))
            .unwrap_or(60)
    })
}
/// Seconds per physics step.
pub fn dt() -> f32 {
    1.0 / rate() as f32
}
/// Steps of pose settling (1.67 s) before the timed trial.
pub fn settle() -> u32 {
    (200 * rate()).div_ceil(120)
}
/// Gait sampling interval in steps (30 samples per second).
pub fn sample_interval() -> u32 {
    (rate() / 30).max(1)
}
/// Cosine and tangent of the largest bone turn allowed in one step.
pub fn turn_limits() -> (f32, f32) {
    let angle = MAX_BONE_ANGULAR_SPEED * dt();
    (angle.cos(), angle.tan())
}
/// Velocity kept per step, from `air_retention` per 1/60 s.
pub fn air_per_step(air_retention: f32) -> f32 {
    air_retention.powf(60.0 / rate() as f32)
}
/// Position-projection and velocity-constraint passes per step. The rebuild
/// after projection makes every bone exactly its rest length regardless.
/// `EVOLUTION_BONE_PASSES` / `EVOLUTION_VELOCITY_PASSES` override them for
/// solver experiments; every engine and the replay read the same values.
pub fn solver_passes() -> (usize, usize) {
    static PASSES: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *PASSES.get_or_init(|| {
        let read = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        (
            read("EVOLUTION_BONE_PASSES", 2),
            read("EVOLUTION_VELOCITY_PASSES", 1),
        )
    })
}
const MAX_MUSCLE_LENGTH_SPEED: f32 = 2.0;
const MAX_MUSCLE_FORCE: f32 = 5.0;
const MAX_NODE_SPEED: f32 = 5.0;
const MAX_BONE_ANGULAR_SPEED: f32 = 15.0;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub radius: f32,
    pub friction: f32,
    pub mass: f32,
    pub failed: f32,
}
/// Mass (kg) of a node of the given diameter.
#[inline]
pub fn node_mass(diameter: f32) -> f32 {
    (0.1 * (diameter / 0.08).powi(2)).clamp(0.02, 10.0)
}
/// A node on its own, without the organs its bones carry.
#[inline]
pub fn node(gene: &NodeGene) -> Node {
    Node {
        pos: [gene.x, gene.y],
        vel: [0.0; 2],
        radius: gene.diameter * 0.5,
        friction: gene.friction,
        mass: node_mass(gene.diameter),
        failed: 0.0,
    }
}
/// A body's nodes with its organs' masses included. An organ rides rigidly
/// on its bone, so its mass is shared by the bone's two nodes in proportion
/// to its position: the body's center of mass is exact, and the organ never
/// touches the ground.
pub fn body(genes: &[NodeGene], bones: &[Bone]) -> Vec<Node> {
    let mut nodes: Vec<Node> = genes.iter().map(node).collect();
    for bone in bones {
        if bone.organ_mass > 0.0 {
            let (a, b) = (bone.a as usize, bone.b as usize);
            if a < nodes.len() && b < nodes.len() {
                nodes[a].mass += bone.organ_mass * (1.0 - bone.organ_at);
                nodes[b].mass += bone.organ_mass * bone.organ_at;
            }
        }
    }
    nodes
}
pub fn nodes(c: &Creature) -> Vec<Node> {
    body(&c.nodes, &c.bones)
}
pub fn target(m: &Muscle, time: f32) -> f32 {
    let phase = (time / m.period + m.phase).fract();
    let wave = if phase < m.duty {
        0.5 + 0.5 * (std::f32::consts::PI * phase / m.duty).cos()
    } else {
        0.5 - 0.5 * (std::f32::consts::PI * (phase - m.duty) / (1.0 - m.duty)).cos()
    };
    m.short + (m.long - m.short) * wave
}
fn limited_target(m: &Muscle, time: f32) -> f32 {
    // Bound the slope of the entire waveform. Clamping each frame against the
    // previous *raw* target allowed the target to jump on the next frame.
    let amplitude = (m.long - m.short).min(
        2.0 * MAX_MUSCLE_LENGTH_SPEED * m.period * m.duty.min(1.0 - m.duty) / std::f32::consts::PI,
    );
    let mut limited = *m;
    limited.short = m.long - amplitude;
    target(&limited, time)
}
fn motor_force(m: &Muscle, time: f32, relative: f32) -> f32 {
    let target_speed = (limited_target(m, time) - limited_target(m, (time - dt()).max(0.0))) / dt();
    // A fixed target is a passive constraint, not an inexhaustible motor.
    // The actuator pulls while shortening and pushes while lengthening.
    (-target_speed * m.stiffness * 0.25 + relative * 0.15)
        .clamp(-MAX_MUSCLE_FORCE, MAX_MUSCLE_FORCE)
}
thread_local! {
    /// Diagnostic ledger of horizontal momentum changes by source:
    /// [integration speed cap, ground contact, velocity-pass speed cap,
    ///  velocity-pass constraints, projection/rebuild center-of-mass shift x mass].
    pub static MOMENTUM_LEDGER: std::cell::Cell<[f64; 5]> = const { std::cell::Cell::new([0.0; 5]) };
}
fn ledger_add(slot: usize, amount: f32) {
    MOMENTUM_LEDGER.with(|l| {
        let mut v = l.get();
        v[slot] += f64::from(amount);
        l.set(v);
    });
}
fn momentum_x(nodes: &[Node]) -> f32 {
    nodes.iter().map(|n| n.vel[0] * n.mass).sum()
}
fn limit_speed(velocity: &mut [f32; 2]) {
    let speed = velocity[0].hypot(velocity[1]);
    if speed > MAX_NODE_SPEED {
        let scale = MAX_NODE_SPEED / speed;
        velocity[0] *= scale;
        velocity[1] *= scale;
    }
}
pub fn center(nodes: &mut [Node]) {
    let mass_sum = nodes.iter().map(|n| n.mass).sum::<f32>();
    let x = nodes.iter().map(|n| n.pos[0] * n.mass).sum::<f32>() / mass_sum;
    let low = nodes
        .iter()
        .map(|n| n.pos[1] - n.radius)
        .fold(f32::INFINITY, f32::min);
    for n in nodes {
        n.pos[0] -= x;
        n.pos[1] -= low;
        n.vel = [0.0; 2];
    }
}
fn contact(n: &mut Node, normal: [f32; 2], penetration: f32, friction: f32) {
    n.pos[0] += normal[0] * penetration;
    n.pos[1] += normal[1] * penetration;
    let vn = n.vel[0] * normal[0] + n.vel[1] * normal[1];
    if vn < 0.0 {
        n.vel[0] -= vn * normal[0];
        n.vel[1] -= vn * normal[1];
        let speed = n.vel[0].hypot(n.vel[1]);
        let keep = (1.0 - (-vn) * friction / speed.max(1e-8)).max(0.0);
        n.vel[0] *= keep;
        n.vel[1] *= keep;
    }
}
pub fn collide(n: &mut Node, cfg: &Config) {
    let mu = n.friction * cfg.ground_friction;
    if cfg.ground && n.pos[1] < n.radius {
        contact(n, [0.0, 1.0], n.radius - n.pos[1], mu);
    }
}
fn bone_point(bone: Bone, nodes: &[Node; 64], t: f32, velocity: bool) -> [f32; 2] {
    let a = &nodes[bone.a as usize];
    let b = &nodes[bone.b as usize];
    let av = if velocity { a.vel } else { a.pos };
    let bv = if velocity { b.vel } else { b.pos };
    [av[0] + (bv[0] - av[0]) * t, av[1] + (bv[1] - av[1]) * t]
}

fn project_bones(nodes: &mut [Node], bones: &[Bone], ground: bool, previous: &[Node; 64]) {
    let com_before: f32 = nodes.iter().map(|n| n.pos[0] * n.mass).sum();
    let mut positions = [[0.0; 2]; 64];
    for (i, node) in nodes.iter().enumerate() {
        positions[i] = node.pos;
    }
    for _ in 0..solver_passes().0 {
        for bone in bones {
            let a = bone.a as usize;
            let b = bone.b as usize;
            let delta = [
                positions[b][0] - positions[a][0],
                positions[b][1] - positions[a][1],
            ];
            let raw_distance = delta[0].hypot(delta[1]);
            let distance = raw_distance.max(1.0e-6);
            let error = distance - bone.rest_length;
            let direction = if raw_distance > 1.0e-6 {
                [delta[0] / distance, delta[1] / distance]
            } else {
                [1.0, 0.0]
            };
            let inverse_a = 1.0 / nodes[a].mass;
            let inverse_b = 1.0 / nodes[b].mass;
            let inverse_sum = inverse_a + inverse_b;
            let share_a = inverse_a / inverse_sum;
            let share_b = inverse_b / inverse_sum;
            positions[a][0] += direction[0] * error * share_a;
            positions[a][1] += direction[1] * error * share_a;
            positions[b][0] -= direction[0] * error * share_b;
            positions[b][1] -= direction[1] * error * share_b;
            if ground {
                positions[a][1] = positions[a][1].max(nodes[a].radius);
                positions[b][1] = positions[b][1].max(nodes[b].radius);
            }
        }
    }
    // A final parent-first reconstruction puts every tree edge exactly on its
    // rest length. The iterative projections above choose a stable set of bone
    // directions; this pass removes accumulated chain-compression error.
    let shape = positions;
    let mut target_center = [0.0; 2];
    let mut mass_sum = 0.0;
    for (i, node) in nodes.iter().enumerate() {
        mass_sum += node.mass;
        target_center[0] += shape[i][0] * node.mass;
        target_center[1] += shape[i][1] * node.mass;
    }
    for bone in bones {
        let a = bone.a as usize;
        let b = bone.b as usize;
        let delta = [shape[b][0] - shape[a][0], shape[b][1] - shape[a][1]];
        let length = delta[0].hypot(delta[1]);
        let mut direction = if length > 1.0e-6 {
            [delta[0] / length, delta[1] / length]
        } else {
            [1.0, 0.0]
        };
        let previous_delta = [
            previous[b].pos[0] - previous[a].pos[0],
            previous[b].pos[1] - previous[a].pos[1],
        ];
        let previous_length = previous_delta[0].hypot(previous_delta[1]);
        if previous_length > 1.0e-6 {
            let previous_direction = [
                previous_delta[0] / previous_length,
                previous_delta[1] / previous_length,
            ];
            let dot = previous_direction[0] * direction[0] + previous_direction[1] * direction[1];
            if dot < turn_limits().0 {
                let cross =
                    previous_direction[0] * direction[1] - previous_direction[1] * direction[0];
                let turn_sign = if cross < 0.0 { -1.0 } else { 1.0 };
                let turned = [
                    previous_direction[0] - previous_direction[1] * turn_sign * turn_limits().1,
                    previous_direction[1] + previous_direction[0] * turn_sign * turn_limits().1,
                ];
                let turn_length = turned[0].hypot(turned[1]);
                direction = [turned[0] / turn_length, turned[1] / turn_length];
            }
        }
        positions[b] = [
            positions[a][0] + direction[0] * bone.rest_length,
            positions[a][1] + direction[1] * bone.rest_length,
        ];
    }
    let mut current_center = [0.0; 2];
    for (i, node) in nodes.iter().enumerate() {
        current_center[0] += positions[i][0] * node.mass;
        current_center[1] += positions[i][1] * node.mass;
    }
    let shift = [
        (target_center[0] - current_center[0]) / mass_sum,
        (target_center[1] - current_center[1]) / mass_sum,
    ];
    for position in &mut positions[..nodes.len()] {
        position[0] += shift[0];
        position[1] += shift[1];
    }
    if ground {
        let lift = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| node.radius - positions[i][1])
            .fold(0.0f32, f32::max);
        for position in &mut positions[..nodes.len()] {
            position[1] += lift;
        }
    }
    let com_after: f32 = (0..nodes.len()).map(|i| positions[i][0] * nodes[i].mass).sum();
    ledger_add(4, (com_after - com_before) / dt());
    let before = momentum_x(nodes);
    for (i, node) in nodes.iter_mut().enumerate() {
        node.pos = positions[i];
        limit_speed(&mut node.vel);
        if ground && node.pos[1] <= node.radius + 1e-5 {
            node.vel[1] = node.vel[1].max(0.0);
        }
    }
    ledger_add(2, momentum_x(nodes) - before);
    // Keep each link's rotation bounded and remove only velocity components
    // that would stretch a bone or rotate it beyond the same angular limit.
    for _ in 0..solver_passes().1 {
        let before = momentum_x(nodes);
        for bone in bones {
            let a = bone.a as usize;
            let b = bone.b as usize;
            let delta = [
                nodes[b].pos[0] - nodes[a].pos[0],
                nodes[b].pos[1] - nodes[a].pos[1],
            ];
            let length = delta[0].hypot(delta[1]).max(1.0e-6);
            let direction = [delta[0] / length, delta[1] / length];
            let inverse_a = 1.0 / nodes[a].mass;
            let inverse_b = 1.0 / nodes[b].mass;
            let inverse_sum = inverse_a + inverse_b;
            let relative = (nodes[b].vel[0] - nodes[a].vel[0]) * direction[0]
                + (nodes[b].vel[1] - nodes[a].vel[1]) * direction[1];
            let impulse = relative / inverse_sum;
            nodes[a].vel[0] += direction[0] * impulse * inverse_a;
            nodes[a].vel[1] += direction[1] * impulse * inverse_a;
            nodes[b].vel[0] -= direction[0] * impulse * inverse_b;
            nodes[b].vel[1] -= direction[1] * impulse * inverse_b;

            let tangent = [-direction[1], direction[0]];
            let angular_velocity = (nodes[b].vel[0] - nodes[a].vel[0]) * tangent[0]
                + (nodes[b].vel[1] - nodes[a].vel[1]) * tangent[1];
            let target_angular_velocity = angular_velocity.clamp(
                -MAX_BONE_ANGULAR_SPEED * length,
                MAX_BONE_ANGULAR_SPEED * length,
            );
            let impulse = (angular_velocity - target_angular_velocity) / inverse_sum;
            nodes[a].vel[0] += tangent[0] * impulse * inverse_a;
            nodes[a].vel[1] += tangent[1] * impulse * inverse_a;
            nodes[b].vel[0] -= tangent[0] * impulse * inverse_b;
            nodes[b].vel[1] -= tangent[1] * impulse * inverse_b;
        }
        ledger_add(3, momentum_x(nodes) - before);
        let before = momentum_x(nodes);
        for node in nodes.iter_mut() {
            limit_speed(&mut node.vel);
            if ground && node.pos[1] <= node.radius + 1e-5 {
                node.vel[1] = node.vel[1].max(0.0);
            }
        }
        ledger_add(2, momentum_x(nodes) - before);
    }
}

pub fn step(nodes: &mut [Node], bones: &[Bone], muscles: &[Muscle], cfg: &Config, tick: u32) {
    if tick == settle() {
        center(nodes);
    }
    let mut old = [Node::default(); 64];
    old[..nodes.len()].copy_from_slice(nodes);
    let time = tick.saturating_sub(settle()) as f32 * dt();
    for (i, n) in nodes.iter_mut().enumerate() {
        let mut f = [0.0; 2];
        for m in muscles {
            let bone_a = bones[m.bone_a as usize];
            let bone_b = bones[m.bone_b as usize];
            let endpoint_a = bone_point(bone_a, &old, m.anchor_a, false);
            let endpoint_b = bone_point(bone_b, &old, m.anchor_b, false);
            let d = [endpoint_b[0] - endpoint_a[0], endpoint_b[1] - endpoint_a[1]];
            let distance = d[0].hypot(d[1]).max(1e-6);
            let dir = [d[0] / distance, d[1] / distance];
            let velocity_a = bone_point(bone_a, &old, m.anchor_a, true);
            let velocity_b = bone_point(bone_b, &old, m.anchor_b, true);
            let relative =
                (velocity_b[0] - velocity_a[0]) * dir[0] + (velocity_b[1] - velocity_a[1]) * dir[1];
            let force = motor_force(m, time, relative);
            let mut weight = 0.0;
            if bone_a.a as usize == i {
                weight += 1.0 - m.anchor_a;
            }
            if bone_a.b as usize == i {
                weight += m.anchor_a;
            }
            if bone_b.a as usize == i {
                weight -= 1.0 - m.anchor_b;
            }
            if bone_b.b as usize == i {
                weight -= m.anchor_b;
            }
            f[0] += dir[0] * force * weight;
            f[1] += dir[1] * force * weight;
        }
        n.vel[0] = (n.vel[0] + f[0] / n.mass * dt()) * air_per_step(cfg.air_retention);
        n.vel[1] = (n.vel[1]
            + (f[1] / n.mass - if tick >= settle() { cfg.gravity } else { 0.0 }) * dt())
            * air_per_step(cfg.air_retention);
        let before = n.vel[0] * n.mass;
        limit_speed(&mut n.vel);
        ledger_add(0, n.vel[0] * n.mass - before);
        n.pos[0] += n.vel[0] * dt();
        n.pos[1] += n.vel[1] * dt();
        if tick >= settle() {
            let before = n.vel[0] * n.mass;
            collide(n, cfg);
            ledger_add(1, n.vel[0] * n.mass - before);
        }
        if !n
            .pos
            .iter()
            .chain(n.vel.iter())
            .all(|v| v.is_finite() && v.abs() < 1e6)
        {
            n.failed = 1.0;
            n.pos = [0.0; 2];
            n.vel = [0.0; 2];
        }
    }
    project_bones(nodes, bones, tick >= settle() && cfg.ground, &old);
}
pub fn evaluate(c: &Creature, cfg: &Config) -> f32 {
    let mut canonical = c.clone();
    crate::evolution::canonicalize_bone_order(&mut canonical);
    let mut n = nodes(&canonical);
    for tick in 0..settle() + cfg.steps() {
        step(&mut n, &canonical.bones, &canonical.muscles, cfg, tick);
    }
    fitness(&n)
}
/// Joint range constraint for one bone, precomputed from the genome. The bone
/// turns about its parent node `a` against a reference bone that shares that
/// node: the parent's own bone, or for bones leaving the root, the first root
/// bone. Angles are measured from the reference end to the child end.
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    /// Far node of the reference bone; `None` for the unconstrained first bone.
    pub reference: Option<usize>,
    /// Direction of the middle of the allowed range, relative to the reference.
    pub center: [f32; 2],
    /// Cosine and sine of half the allowed range.
    pub half: [f32; 2],
    /// Share of a correction taken by the child end (by inverse inertia).
    pub child_share: f32,
    /// Masses of the child and reference ends over the three joint masses,
    /// used to keep the joint's center of mass in place.
    pub child_mass: f32,
    pub reference_mass: f32,
}
impl Joint {
    pub const FREE: Joint = Joint {
        reference: None,
        center: [1.0, 0.0],
        half: [-1.0, 0.0],
        child_share: 0.5,
        child_mass: 0.0,
        reference_mass: 0.0,
    };
}
/// Rounds to the GPU's snorm16 storage of the joint range center.
fn snorm16(v: f32) -> f32 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() / 32767.0
}
/// Joint constraints for a canonical (parent-first) skeleton.
pub fn joints(genes: &[NodeGene], bones: &[Bone]) -> Vec<Joint> {
    let state = body(genes, bones);
    let first_root = bones.iter().position(|b| b.a == 0);
    bones
        .iter()
        .enumerate()
        .map(|(index, bone)| {
            let pivot = bone.a as usize;
            let child = bone.b as usize;
            let reference = match bones.iter().find(|p| p.b as usize == pivot) {
                Some(parent) => parent.a as usize,
                None => match first_root {
                    Some(first) if first != index => bones[first].b as usize,
                    _ => return Joint::FREE,
                },
            };
            let at = |i: usize| [genes[i].x - genes[pivot].x, genes[i].y - genes[pivot].y];
            let (u, v) = (at(reference), at(child));
            let rest = (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1]);
            let middle = rest + 0.5 * (bone.min_angle + bone.max_angle);
            let half = 0.5 * (bone.max_angle - bone.min_angle);
            let length = |d: [f32; 2]| d[0].hypot(d[1]).max(0.01);
            let inertia_child = state[child].mass * length(v).powi(2);
            let inertia_reference = state[reference].mass * length(u).powi(2);
            let total = state[pivot].mass + state[child].mass + state[reference].mass;
            Joint {
                reference: Some(reference),
                center: [snorm16(middle.cos()), snorm16(middle.sin())],
                half: [half.cos(), half.sin()],
                child_share: inertia_reference / (inertia_child + inertia_reference),
                child_mass: state[child].mass / total,
                reference_mass: state[reference].mass / total,
            }
        })
        .collect()
}
/// Ground heights (m) of the bumps added by each roughness level.
pub const TERRAIN_AMPLITUDES: [f32; 4] = [0.0, 0.02, 0.04, 0.07];
/// Bump height for `Config::terrain`.
pub fn terrain_amplitude(level: u8) -> f32 {
    TERRAIN_AMPLITUDES[usize::from(level).min(TERRAIN_AMPLITUDES.len() - 1)]
}
/// Wavelengths (m), weights, and phase offsets of the two bump trains. The
/// engines and the UI all evaluate the same ground.
pub const TERRAIN_WAVES: [(f32, f32, f32); 2] = [(1.1, 0.65, 0.0), (0.43, 0.35, 0.3)];
/// Ground height and slope at `x` for bump height `amplitude`. Each bump is
/// 16 u^2 (1 - u)^2 over one wavelength: smooth, cheap, and free of trig.
pub fn terrain(x: f32, amplitude: f32) -> (f32, f32) {
    let mut height = 0.0;
    let mut slope = 0.0;
    for (wavelength, weight, offset) in TERRAIN_WAVES {
        let t = x / wavelength + offset;
        let u = t - t.floor();
        let w = u * (1.0 - u);
        height += weight * 16.0 * w * w;
        slope += weight * 32.0 * w * (1.0 - 2.0 * u) / wavelength;
    }
    (amplitude * height, amplitude * slope)
}
pub fn fitness(n: &[Node]) -> f32 {
    if n.iter().any(|n| n.failed != 0.0) {
        FAILED
    } else {
        let mass_sum = n.iter().map(|node| node.mass).sum::<f32>();
        n.iter().map(|node| node.pos[0] * node.mass).sum::<f32>() / mass_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_node(pos: [f32; 2]) -> Node {
        Node {
            pos,
            radius: 0.03,
            mass: 1.0,
            ..Node::default()
        }
    }

    #[test]
    fn muscle_targets_change_at_a_bounded_rate() {
        let muscle = Muscle {
            bone_a: 0,
            bone_b: 1,
            anchor_a: 0.5,
            anchor_b: 0.5,
            short: 0.01,
            long: 1.0,
            period: 0.1,
            phase: 0.0,
            duty: 0.5,
            stiffness: 20.0,
            sensor: 255,
            reset: 0.0,
        };
        for time in [0.025, 0.075] {
            assert!(
                (limited_target(&muscle, time) - limited_target(&muscle, (time - dt()).max(0.0)))
                    .abs()
                    <= MAX_MUSCLE_LENGTH_SPEED * dt() + 1e-6
            );
        }
        for tick in 1..120 {
            let time = tick as f32 * dt();
            assert!(
                (limited_target(&muscle, time) - limited_target(&muscle, time - dt())).abs()
                    <= MAX_MUSCLE_LENGTH_SPEED * dt() + 1e-6
            );
        }
    }

    #[test]
    fn fixed_target_cannot_supply_motor_force() {
        let muscle = Muscle {
            bone_a: 0,
            bone_b: 1,
            anchor_a: 0.5,
            anchor_b: 0.5,
            short: 0.1,
            long: 0.1,
            period: 0.1,
            phase: 0.25,
            duty: 0.5,
            stiffness: 120.0,
            sensor: 255,
            reset: 0.0,
        };
        for tick in 0..120 {
            assert_eq!(motor_force(&muscle, tick as f32 * dt(), 0.0), 0.0);
        }
        assert!(motor_force(&muscle, 0.025, 0.0) == 0.0);
    }

    #[test]
    fn terrain_slope_matches_its_height() {
        for i in 0..200 {
            let x = i as f32 * 0.037 - 3.0;
            let (_, slope) = terrain(x, 0.05);
            let h = 1e-3;
            let numeric = (terrain(x + h, 0.05).0 - terrain(x - h, 0.05).0) / (2.0 * h);
            assert!((slope - numeric).abs() < 1e-2, "{x}: {slope} vs {numeric}");
            assert!((0.0..=0.05 + 1e-6).contains(&terrain(x, 0.05).0));
        }
        assert_eq!(terrain(0.7, 0.0), (0.0, 0.0));
    }

    #[test]
    fn correcting_a_bone_does_not_create_velocity() {
        let mut nodes = [test_node([0.0, 0.0]), test_node([0.2, 0.0])];
        let mut previous = [Node::default(); 64];
        previous[0] = nodes[0];
        previous[1] = nodes[1];
        let bone = Bone::new(0, 1, 1.0);

        project_bones(&mut nodes, &[bone], false, &previous);

        let length = (nodes[1].pos[0] - nodes[0].pos[0]).hypot(nodes[1].pos[1] - nodes[0].pos[1]);
        assert!((length - bone.rest_length).abs() < 1e-6);
        assert!(nodes.iter().all(|node| node.vel == [0.0; 2]));
    }

    #[test]
    fn fitness_tracks_center_of_mass_instead_of_shape_average() {
        let before = [
            Node {
                pos: [-1.0, 0.0],
                mass: 1.0,
                ..Node::default()
            },
            Node {
                pos: [1.0, 0.0],
                mass: 3.0,
                ..Node::default()
            },
        ];
        let reshaped = [
            Node {
                pos: [-3.0, 0.0],
                mass: 1.0,
                ..Node::default()
            },
            Node {
                pos: [5.0 / 3.0, 0.0],
                mass: 3.0,
                ..Node::default()
            },
        ];
        assert!(
            (before.iter().map(|node| node.pos[0]).sum::<f32>() / 2.0
                - reshaped.iter().map(|node| node.pos[0]).sum::<f32>() / 2.0)
                .abs()
                > 0.1
        );
        assert!((fitness(&before) - fitness(&reshaped)).abs() < 1e-6);
    }

    #[test]
    fn bone_rotation_and_node_speed_are_bounded() {
        let mut previous = [Node::default(); 64];
        previous[0] = test_node([0.0, 0.0]);
        previous[1] = test_node([0.1, 0.0]);
        let mut nodes = [previous[0], previous[1]];
        nodes[1].pos = [0.1, 0.1];
        nodes[1].vel = [0.0, 100.0];
        let bone = Bone::new(0, 1, 0.1);

        project_bones(&mut nodes, &[bone], false, &previous);

        let delta = [
            nodes[1].pos[0] - nodes[0].pos[0],
            nodes[1].pos[1] - nodes[0].pos[1],
        ];
        let angle = delta[1].atan2(delta[0]).abs();
        assert!(angle <= MAX_BONE_ANGULAR_SPEED * dt() + 1e-5);
        assert!(
            nodes
                .iter()
                .all(|node| node.vel[0].hypot(node.vel[1]) <= MAX_NODE_SPEED + 1e-5)
        );
        let length = delta[0].hypot(delta[1]);
        let direction = [delta[0] / length, delta[1] / length];
        let relative_radial = (nodes[1].vel[0] - nodes[0].vel[0]) * direction[0]
            + (nodes[1].vel[1] - nodes[0].vel[1]) * direction[1];
        let tangent = [-direction[1], direction[0]];
        let relative_tangent = (nodes[1].vel[0] - nodes[0].vel[0]) * tangent[0]
            + (nodes[1].vel[1] - nodes[0].vel[1]) * tangent[1];
        assert!(relative_radial.abs() < 1e-5);
        assert!(relative_tangent.abs() <= MAX_BONE_ANGULAR_SPEED * length + 1e-5);
    }
}
