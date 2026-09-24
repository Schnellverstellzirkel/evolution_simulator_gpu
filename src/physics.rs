use crate::{
    config::Config,
    evolution::{Bone, Creature, FAILED, Muscle, NodeGene},
};
pub const DT: f32 = 1.0 / 120.0;
pub const SETTLE: u32 = 200;
const BONE_SOLVE_ITERATIONS: usize = 8;
const BONE_COLLISION_RADIUS: f32 = 0.04;
const MAX_BONE_PROJECTION_SPEED: f32 = 10.0;
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
#[inline]
pub fn node(gene: &NodeGene) -> Node {
    Node {
        pos: [gene.x, gene.y],
        vel: [0.0; 2],
        radius: gene.diameter * 0.5,
        friction: gene.friction,
        mass: (0.1 * (gene.diameter / 0.08).powi(2)).clamp(0.02, 10.0),
        failed: 0.0,
    }
}
pub fn nodes(c: &Creature) -> Vec<Node> {
    c.nodes.iter().map(node).collect()
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
pub fn center(nodes: &mut [Node]) {
    let x = nodes.iter().map(|n| n.pos[0]).sum::<f32>() / nodes.len() as f32;
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

fn closest_segment_points(
    a0: [f32; 2],
    a1: [f32; 2],
    b0: [f32; 2],
    b1: [f32; 2],
) -> (f32, f32, [f32; 2], [f32; 2]) {
    let u = [a1[0] - a0[0], a1[1] - a0[1]];
    let v = [b1[0] - b0[0], b1[1] - b0[1]];
    let w = [a0[0] - b0[0], a0[1] - b0[1]];
    let aa = u[0] * u[0] + u[1] * u[1];
    let bb = u[0] * v[0] + u[1] * v[1];
    let cc = v[0] * v[0] + v[1] * v[1];
    let dd = u[0] * w[0] + u[1] * w[1];
    let ee = v[0] * w[0] + v[1] * w[1];
    let (s, t) = if aa <= 1.0e-12 && cc <= 1.0e-12 {
        (0.0, 0.0)
    } else if aa <= 1.0e-12 {
        (0.0, (ee / cc).clamp(0.0, 1.0))
    } else if cc <= 1.0e-12 {
        ((-dd / aa).clamp(0.0, 1.0), 0.0)
    } else {
        let denominator = aa * cc - bb * bb;
        let mut s_numerator;
        let mut s_denominator;
        let mut t_numerator;
        let mut t_denominator;
        if denominator <= 1.0e-12 {
            s_numerator = 0.0;
            s_denominator = 1.0;
            t_numerator = ee;
            t_denominator = cc;
        } else {
            s_numerator = bb * ee - cc * dd;
            t_numerator = aa * ee - bb * dd;
            s_denominator = denominator;
            t_denominator = denominator;
        }
        if s_numerator < 0.0 {
            s_numerator = 0.0;
            t_numerator = ee;
            t_denominator = cc;
        } else if s_numerator > s_denominator {
            s_numerator = s_denominator;
            t_numerator = ee + bb;
            t_denominator = cc;
        }
        if t_numerator < 0.0 {
            t_numerator = 0.0;
            if -dd < 0.0 {
                s_numerator = 0.0;
                s_denominator = 1.0;
            } else if -dd > aa {
                s_numerator = 1.0;
                s_denominator = 1.0;
            } else {
                s_numerator = -dd;
                s_denominator = aa;
            }
        } else if t_numerator > t_denominator {
            t_numerator = t_denominator;
            let endpoint_projection = -dd + bb;
            if endpoint_projection < 0.0 {
                s_numerator = 0.0;
                s_denominator = 1.0;
            } else if endpoint_projection > aa {
                s_numerator = 1.0;
                s_denominator = 1.0;
            } else {
                s_numerator = endpoint_projection;
                s_denominator = aa;
            }
        }
        (
            if s_numerator.abs() < 1.0e-12 {
                0.0
            } else {
                s_numerator / s_denominator
            },
            if t_numerator.abs() < 1.0e-12 {
                0.0
            } else {
                t_numerator / t_denominator
            },
        )
    };
    let pa = [a0[0] + u[0] * s, a0[1] + u[1] * s];
    let pb = [b0[0] + v[0] * t, b0[1] + v[1] * t];
    (s, t, pa, pb)
}

fn bone_collision_radius(nodes: &[Node], bone: Bone) -> f32 {
    BONE_COLLISION_RADIUS.min(
        nodes[bone.a as usize]
            .radius
            .min(nodes[bone.b as usize].radius),
    )
}

fn shared_bone_joint(a: Bone, b: Bone) -> Option<usize> {
    [a.a, a.b]
        .into_iter()
        .find(|node| *node == b.a || *node == b.b)
        .map(|node| node as usize)
}

#[derive(Clone, Copy)]
struct CollisionSegment {
    start_t: f32,
    end_t: f32,
    start: [f32; 2],
    end: [f32; 2],
    radius: f32,
}

fn collision_segment(
    nodes: &[Node],
    positions: &[[f32; 2]; 64],
    bone: Bone,
    joint: Option<usize>,
) -> Option<CollisionSegment> {
    let a = positions[bone.a as usize];
    let b = positions[bone.b as usize];
    let length = (b[0] - a[0]).hypot(b[1] - a[1]).max(1.0e-6);
    let radius = bone_collision_radius(nodes, bone);
    let mut start = 0.0;
    let mut end = 1.0;
    if let Some(joint) = joint {
        let trim = (nodes[joint].radius + radius) / length;
        if bone.a as usize == joint {
            start = trim;
        } else {
            end = 1.0 - trim;
        }
    }
    if start >= end {
        return None;
    }
    let interpolate = |t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    Some(CollisionSegment {
        start_t: start,
        end_t: end,
        start: interpolate(start),
        end: interpolate(end),
        radius,
    })
}

fn project_bone_collisions(
    nodes: &[Node],
    positions: &mut [[f32; 2]; 64],
    bones: &[Bone],
    ground: bool,
) {
    for i in 0..bones.len() {
        let bone_a = bones[i];
        let a0 = bone_a.a as usize;
        let a1 = bone_a.b as usize;
        for &bone_b in &bones[i + 1..] {
            let b0 = bone_b.a as usize;
            let b1 = bone_b.b as usize;
            let joint = shared_bone_joint(bone_a, bone_b);
            let Some(segment_a) = collision_segment(nodes, positions, bone_a, joint) else {
                continue;
            };
            let Some(segment_b) = collision_segment(nodes, positions, bone_b, joint) else {
                continue;
            };
            let separation = segment_a.radius + segment_b.radius;
            if segment_a.start[0].max(segment_a.end[0]) + separation
                < segment_b.start[0].min(segment_b.end[0])
                || segment_b.start[0].max(segment_b.end[0]) + separation
                    < segment_a.start[0].min(segment_a.end[0])
                || segment_a.start[1].max(segment_a.end[1]) + separation
                    < segment_b.start[1].min(segment_b.end[1])
                || segment_b.start[1].max(segment_b.end[1]) + separation
                    < segment_a.start[1].min(segment_a.end[1])
            {
                continue;
            }
            let (local_s, local_t, pa, pb) = closest_segment_points(
                segment_a.start,
                segment_a.end,
                segment_b.start,
                segment_b.end,
            );
            let mut s = segment_a.start_t + local_s * (segment_a.end_t - segment_a.start_t);
            let mut t = segment_b.start_t + local_t * (segment_b.end_t - segment_b.start_t);
            let delta = [pa[0] - pb[0], pa[1] - pb[1]];
            let distance = delta[0].hypot(delta[1]);
            let edge_a = [
                positions[a1][0] - positions[a0][0],
                positions[a1][1] - positions[a0][1],
            ];
            let edge_b = [
                positions[b1][0] - positions[b0][0],
                positions[b1][1] - positions[b0][1],
            ];
            let length_a = edge_a[0].hypot(edge_a[1]);
            let length_b = edge_b[0].hypot(edge_b[1]);
            if distance <= 1.0e-6 && joint.is_none() {
                // A true crossing has no useful closest-point normal: midpoint
                // corrections only translate both segments. Push one endpoint
                // across the other segment's line to fold the skeleton apart.
                let (moving, fixed, fixed_edge, fixed_length) = if length_a < length_b {
                    ([a0, a1], [b0, b1], edge_b, length_b)
                } else {
                    ([b0, b1], [a0, a1], edge_a, length_a)
                };
                if fixed_length > 1.0e-6 {
                    let line_normal = [-fixed_edge[1] / fixed_length, fixed_edge[0] / fixed_length];
                    let d0 = (positions[moving[0]][0] - positions[fixed[0]][0]) * line_normal[0]
                        + (positions[moving[0]][1] - positions[fixed[0]][1]) * line_normal[1];
                    let d1 = (positions[moving[1]][0] - positions[fixed[0]][0]) * line_normal[0]
                        + (positions[moving[1]][1] - positions[fixed[0]][1]) * line_normal[1];
                    if d0 * d1 < 0.0 {
                        let endpoint = if nodes[moving[0]].mass <= nodes[moving[1]].mass {
                            moving[0]
                        } else {
                            moving[1]
                        };
                        let other_distance = if endpoint == moving[0] { d1 } else { d0 };
                        let current_distance = if endpoint == moving[0] { d0 } else { d1 };
                        let target_distance = other_distance.signum()
                            * (other_distance.abs() + current_distance.abs() + separation);
                        let correction = target_distance - current_distance;
                        positions[endpoint][0] += line_normal[0] * correction;
                        positions[endpoint][1] += line_normal[1] * correction;
                        if ground {
                            positions[endpoint][1] =
                                positions[endpoint][1].max(nodes[endpoint].radius);
                        }
                        continue;
                    }
                }
            }
            let (normal, penetration) = if distance > 1.0e-6 {
                (
                    [delta[0] / distance, delta[1] / distance],
                    separation - distance,
                )
            } else {
                let normal = if length_a < length_b {
                    s = segment_a.start_t + 0.25 * (segment_a.end_t - segment_a.start_t);
                    if length_a > 1.0e-6 {
                        [-edge_a[1] / length_a, edge_a[0] / length_a]
                    } else {
                        [1.0, 0.0]
                    }
                } else {
                    t = segment_b.start_t + 0.25 * (segment_b.end_t - segment_b.start_t);
                    if length_b > 1.0e-6 {
                        [-edge_b[1] / length_b, edge_b[0] / length_b]
                    } else {
                        [1.0, 0.0]
                    }
                };
                (normal, separation)
            };
            if penetration <= 0.0 {
                continue;
            }
            let weights = [1.0 - s, s, 1.0 - t, t];
            let indices = [a0, a1, b0, b1];
            let signs = [1.0, 1.0, -1.0, -1.0];
            let gradients = std::array::from_fn::<_, 4, _>(|j| {
                (0..4)
                    .filter(|&k| indices[k] == indices[j])
                    .map(|k| signs[k] * weights[k])
                    .sum::<f32>()
            });
            let mut denominator = 0.0;
            for j in 0..4 {
                if indices[..j].contains(&indices[j]) {
                    continue;
                }
                let inverse_mass = 1.0 / nodes[indices[j]].mass;
                denominator += inverse_mass * gradients[j] * gradients[j];
            }
            denominator = denominator.max(1.0e-8);
            for j in 0..4 {
                if indices[..j].contains(&indices[j]) {
                    continue;
                }
                let index = indices[j];
                let correction = penetration * gradients[j] / nodes[index].mass / denominator;
                positions[index][0] += normal[0] * correction;
                positions[index][1] += normal[1] * correction;
                if ground {
                    positions[index][1] = positions[index][1].max(nodes[index].radius);
                }
            }
        }
    }
}

fn has_bone_overlap(nodes: &[Node], positions: &[[f32; 2]; 64], bones: &[Bone]) -> bool {
    for (i, bone_a) in bones.iter().enumerate() {
        for bone_b in &bones[i + 1..] {
            let joint = shared_bone_joint(*bone_a, *bone_b);
            let Some(segment_a) = collision_segment(nodes, positions, *bone_a, joint) else {
                continue;
            };
            let Some(segment_b) = collision_segment(nodes, positions, *bone_b, joint) else {
                continue;
            };
            let (_, _, pa, pb) = closest_segment_points(
                segment_a.start,
                segment_a.end,
                segment_b.start,
                segment_b.end,
            );
            if (pa[0] - pb[0]).hypot(pa[1] - pb[1]) < segment_a.radius + segment_b.radius - 0.001 {
                return true;
            }
        }
    }
    false
}

fn project_bones(nodes: &mut [Node], bones: &[Bone], ground: bool) {
    let mut positions = [[0.0; 2]; 64];
    let mut original = [[0.0; 2]; 64];
    let mut original_velocity = [[0.0; 2]; 64];
    for (i, node) in nodes.iter().enumerate() {
        positions[i] = node.pos;
        original[i] = node.pos;
        original_velocity[i] = node.vel;
    }
    for iteration in 0..BONE_SOLVE_ITERATIONS {
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
        project_bone_collisions(nodes, &mut positions, bones, ground);
        if iteration + 1 < BONE_SOLVE_ITERATIONS {
            let shape = positions;
            for bone in bones {
                let a = bone.a as usize;
                let b = bone.b as usize;
                let delta = [shape[b][0] - shape[a][0], shape[b][1] - shape[a][1]];
                let length = delta[0].hypot(delta[1]);
                let direction = if length > 1.0e-6 {
                    [delta[0] / length, delta[1] / length]
                } else {
                    [1.0, 0.0]
                };
                positions[b] = [
                    positions[a][0] + direction[0] * bone.rest_length,
                    positions[a][1] + direction[1] * bone.rest_length,
                ];
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
        let direction = if length > 1.0e-6 {
            [delta[0] / length, delta[1] / length]
        } else {
            [1.0, 0.0]
        };
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
    for (i, node) in nodes.iter_mut().enumerate() {
        let delta = [
            positions[i][0] - original[i][0],
            positions[i][1] - original[i][1],
        ];
        node.pos = positions[i];
        let mut correction_velocity = [delta[0] / DT, delta[1] / DT];
        let speed = correction_velocity[0].hypot(correction_velocity[1]);
        if speed > MAX_BONE_PROJECTION_SPEED {
            correction_velocity[0] *= MAX_BONE_PROJECTION_SPEED / speed;
            correction_velocity[1] *= MAX_BONE_PROJECTION_SPEED / speed;
        }
        node.vel[0] = original_velocity[i][0] + correction_velocity[0];
        node.vel[1] = original_velocity[i][1] + correction_velocity[1];
    }
    if !nodes.is_empty() && has_bone_overlap(nodes, &positions, bones) {
        // The collision solver makes a best effort to untangle initial or
        // newly folded poses. Any residual capsule penetration is invalid for
        // selection, so a folded specimen cannot win on its score.
        for node in nodes {
            node.failed = 1.0;
        }
    }
}

pub fn step(nodes: &mut [Node], bones: &[Bone], muscles: &[Muscle], cfg: &Config, tick: u32) {
    if tick == SETTLE {
        center(nodes);
    }
    let mut old = [Node::default(); 64];
    old[..nodes.len()].copy_from_slice(nodes);
    let time = tick.saturating_sub(SETTLE) as f32 * DT;
    for (i, n) in nodes.iter_mut().enumerate() {
        if n.failed >= 0.5 {
            continue;
        }
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
            let force = ((distance - target(m, time)).clamp(-0.25, 0.25) * m.stiffness
                + relative * 0.15)
                .clamp(-30.0, 30.0);
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
        n.vel[0] = (n.vel[0] + f[0] / n.mass * DT) * cfg.air_retention.sqrt();
        n.vel[1] = (n.vel[1]
            + (f[1] / n.mass - if tick >= SETTLE { cfg.gravity } else { 0.0 }) * DT)
            * cfg.air_retention.sqrt();
        n.pos[0] += n.vel[0] * DT;
        n.pos[1] += n.vel[1] * DT;
        if tick >= SETTLE {
            collide(n, cfg);
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
    project_bones(nodes, bones, tick >= SETTLE && cfg.ground);
}
pub fn evaluate(c: &Creature, cfg: &Config) -> f32 {
    let mut canonical = c.clone();
    crate::evolution::canonicalize_bone_order(&mut canonical);
    let mut n = nodes(&canonical);
    for tick in 0..SETTLE + cfg.steps() {
        step(&mut n, &canonical.bones, &canonical.muscles, cfg, tick);
    }
    fitness(&n)
}
pub fn fitness(n: &[Node]) -> f32 {
    if n.iter().any(|n| n.failed != 0.0) {
        FAILED
    } else {
        n.iter().map(|n| n.pos[0]).sum::<f32>() / n.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment_distance(a: Bone, b: Bone, nodes: &[Node]) -> f32 {
        let (_, _, pa, pb) = closest_segment_points(
            nodes[a.a as usize].pos,
            nodes[a.b as usize].pos,
            nodes[b.a as usize].pos,
            nodes[b.b as usize].pos,
        );
        (pa[0] - pb[0]).hypot(pa[1] - pb[1])
    }

    #[test]
    fn crossing_nonadjacent_bones_cannot_win_and_keep_exact_lengths() {
        let bones = [
            Bone {
                a: 0,
                b: 1,
                rest_length: 2.0,
            },
            Bone {
                a: 1,
                b: 2,
                rest_length: 1.0,
            },
            Bone {
                a: 2,
                b: 3,
                rest_length: 1.0,
            },
            Bone {
                a: 3,
                b: 4,
                rest_length: 2.0,
            },
        ];
        let positions = [[-1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, -1.0]];
        let mut nodes = positions.map(|pos| Node {
            pos,
            radius: 0.03,
            mass: 1.0,
            ..Node::default()
        });

        assert!(segment_distance(bones[0], bones[3], &nodes) < 1.0e-6);
        project_bones(&mut nodes, &bones, false);

        for bone in bones {
            let length = {
                let a = nodes[bone.a as usize].pos;
                let b = nodes[bone.b as usize].pos;
                (b[0] - a[0]).hypot(b[1] - a[1])
            };
            assert!((length - bone.rest_length).abs() < 1.0e-5);
        }
        assert!(nodes.iter().any(|node| node.failed != 0.0));
        assert_eq!(fitness(&nodes), FAILED);
        assert!(segment_distance(bones[0], bones[3], &nodes) < 2.0 * BONE_COLLISION_RADIUS);
    }

    #[test]
    fn bones_may_meet_at_their_shared_joint() {
        let bones = [
            Bone {
                a: 0,
                b: 1,
                rest_length: 1.0,
            },
            Bone {
                a: 1,
                b: 2,
                rest_length: 1.0,
            },
        ];
        let mut nodes = [
            Node {
                pos: [-1.0, 0.0],
                radius: 0.04,
                mass: 1.0,
                ..Node::default()
            },
            Node {
                pos: [0.0, 0.0],
                radius: 0.04,
                mass: 1.0,
                ..Node::default()
            },
            Node {
                pos: [0.0, 1.0],
                radius: 0.04,
                mass: 1.0,
                ..Node::default()
            },
        ];
        project_bones(&mut nodes, &bones, false);
        assert!(nodes.iter().all(|node| node.failed == 0.0));
    }

    #[test]
    fn adjacent_bones_unfold_when_their_segments_overlap() {
        let bones = [
            Bone {
                a: 0,
                b: 1,
                rest_length: 1.0,
            },
            Bone {
                a: 1,
                b: 2,
                rest_length: 1.0,
            },
        ];
        let mut nodes = [
            Node {
                pos: [-1.0, 0.0],
                radius: 0.04,
                mass: 1.0,
                ..Node::default()
            },
            Node {
                pos: [0.0, 0.0],
                radius: 0.04,
                mass: 1.0,
                ..Node::default()
            },
            Node {
                pos: [-1.0, 0.0],
                radius: 0.04,
                mass: 1.0,
                ..Node::default()
            },
        ];
        project_bones(&mut nodes, &bones, false);
        assert!(nodes.iter().all(|node| node.failed == 0.0));
        for bone in bones {
            let a = nodes[bone.a as usize].pos;
            let b = nodes[bone.b as usize].pos;
            assert!(((b[0] - a[0]).hypot(b[1] - a[1]) - bone.rest_length).abs() < 1e-5);
        }
    }
}
