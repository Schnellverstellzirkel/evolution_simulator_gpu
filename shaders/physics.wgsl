struct Node {
    pos: vec2f,
    vel: vec2f,
    radius: f32,
    friction: f32,
    mass: f32,
    failed: f32,
}
struct Muscle {
    a0: u32,
    a1: u32,
    b0: u32,
    b1: u32,
    anchor_a: f32,
    anchor_b: f32,
    short: f32,
    long: f32,
    inv_period: f32,
    phase: f32,
    duty: f32,
    stiffness: f32,
    inv_duty: f32,
    inv_complement: f32,
}
struct Bone {
    a: u32,
    b: u32,
    rest_length: f32,
}
struct Meta {
    nodes: u32,
    bones_start: u32,
    bone_count: u32,
}
struct NodeAdj { start: u32, count: u32 }
struct Params {
    tick: u32,
    steps: u32,
    stride: u32,
    count: u32,
    gravity: f32,
    air: f32,
    friction: f32,
    ground: f32,
    total_steps: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
struct Result {
    fitness: f32,
    ground_contact: f32,
    // Accumulate vertical min/max, then rewrite as range/cadence on the final tick.
    vertical_oscillation: f32,
    gait_frequency: f32,
    previous_center_y: f32,
    vertical_extremum: f32,
    vertical_trend: f32,
    gait_turns: f32,
}
struct ClosestPoints {
    s: f32,
    t: f32,
    pa: vec2f,
    pb: vec2f,
}
@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> muscles: array<Muscle>;
@group(0) @binding(2) var<storage, read> metadata: array<Meta>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read_write> results: array<Result>;
@group(0) @binding(5) var<storage, read> node_adjacencies: array<NodeAdj>;
@group(0) @binding(6) var<storage, read> bones: array<Bone>;
@group(0) @binding(7) var<storage, read> muscle_indices: array<u32>;

var<workgroup> positions: array<vec2f, WORKGROUPX2>;
var<workgroup> velocities: array<vec2f, WORKGROUPX2>;
var<workgroup> radii: array<f32, WORKGROUP>;
var<workgroup> masses: array<f32, WORKGROUP>;
var<workgroup> failures: array<f32, WORKGROUP>;

const BONE_SOLVE_ITERATIONS: u32 = 8u;
const BONE_COLLISION_RADIUS: f32 = 0.04;
const MAX_BONE_PROJECTION_SPEED: f32 = 10.0;

fn closest_segment_points(a0: vec2f, a1: vec2f, b0: vec2f, b1: vec2f) -> ClosestPoints {
    let u = a1 - a0;
    let v = b1 - b0;
    let w = a0 - b0;
    let aa = dot(u, u);
    let bb = dot(u, v);
    let cc = dot(v, v);
    let dd = dot(u, w);
    let ee = dot(v, w);
    var s = 0.0;
    var t = 0.0;
    if aa <= 1e-12 && cc <= 1e-12 {
        s = 0.0;
        t = 0.0;
    } else if aa <= 1e-12 {
        s = 0.0;
        t = clamp(ee / cc, 0.0, 1.0);
    } else if cc <= 1e-12 {
        s = clamp(-dd / aa, 0.0, 1.0);
        t = 0.0;
    } else {
        let denominator = aa * cc - bb * bb;
        var s_numerator = 0.0;
        var s_denominator = 1.0;
        var t_numerator = 0.0;
        var t_denominator = 1.0;
        if denominator <= 1e-12 {
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
        if abs(s_numerator) >= 1e-12 {
            s = s_numerator / s_denominator;
        }
        if abs(t_numerator) >= 1e-12 {
            t = t_numerator / t_denominator;
        }
    }
    return ClosestPoints(s, t, a0 + u * s, b0 + v * t);
}

fn muscle_length(m: Muscle, time: f32) -> f32 {
    let phase = fract(time * m.inv_period + m.phase);
    var wave: f32;
    if phase < m.duty {
        wave = 0.5 + 0.5 * cos(3.14159265359 * phase * m.inv_duty);
    } else {
        wave = 0.5 - 0.5 * cos(3.14159265359 * (phase - m.duty) * m.inv_complement);
    }
    return mix(m.short, m.long, wave);
}
fn contact(node: Node, normal: vec2f, penetration: f32) -> Node {
    var n = node;
    n.pos += normal * penetration;
    let vn = dot(n.vel, normal);
    if vn < 0.0 {
        n.vel -= vn * normal;
        let keep = max(0.0, 1.0 - (-vn) * n.friction * p.friction / max(length(n.vel), 1e-8));
        n.vel *= keep;
    }
    return n;
}
fn collide(node: Node) -> Node {
    var n = node;
    if p.ground > 0.0 && n.pos.y < n.radius {
        n = contact(n, vec2f(0.0, 1.0), n.radius - n.pos.y);
    }
    return n;
}

@compute @workgroup_size(WORKGROUP)
fn advance(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
    let creature = (group.x * WORKGROUPu + lane) / p.stride;
    let local = lane % p.stride;
    let base = lane - local;
    let half = WORKGROUPu;
    var body = Meta(0u, 0u, 0u);
    var n = Node(vec2f(0.0), vec2f(0.0), 0.0, 0.0, 1.0, 0.0);
    if creature < p.count {
        body = metadata[creature];
        if local < body.nodes {
            n = nodes[creature * p.stride + local];
        }
    }
    positions[lane] = n.pos;
    positions[half + lane] = n.pos;
    velocities[lane] = n.vel;
    velocities[half + lane] = n.vel;
    radii[lane] = n.radius;
    masses[lane] = n.mass;
    failures[lane] = n.failed;

    // Only lane zero owns behavior metrics. Keep them in registers for the
    // whole dispatch and exchange them with global memory at chunk boundaries.
    var metrics = Result(0.0, 0.0, 1e20, -1e20, 0.0, 0.0, 0.0, 0.0);
    if creature < p.count && local == 0u && p.tick > 0u {
        metrics = results[creature];
    }
    var muscle_adjacency = NodeAdj(0u, 0u);
    if creature < p.count && local < body.nodes {
        let slot = creature * p.stride + local;
        muscle_adjacency = node_adjacencies[slot];
    }
    workgroupBarrier();

    for (var s = 0u; s < p.steps; s++) {
        let tick = p.tick + s;
        let read_base = u32(s & 1u) * half;
        let write_base = half - read_base;
        if creature < p.count && local < body.nodes {
            n.pos = positions[read_base + lane];
            n.vel = velocities[read_base + lane];
        }
        if tick == 200u {
            if local < body.nodes {
                var avg = 0.0;
                var low = 1e20;
                for (var j = 0u; j < body.nodes; j++) {
                    avg += positions[read_base + base + j].x;
                    low = min(low, positions[read_base + base + j].y - radii[base + j]);
                }
                n.pos -= vec2f(avg / f32(body.nodes), low);
                n.vel = vec2f(0.0);
            }
            workgroupBarrier();
            positions[read_base + lane] = n.pos;
            velocities[read_base + lane] = n.vel;
            workgroupBarrier();
        }
        if local < body.nodes && n.failed < 0.5 {
            let time = f32(max(tick, 200u) - 200u) / 120.0;
            var force = vec2f(0.0);
            for (var j = 0u; j < muscle_adjacency.count; j++) {
                let m = muscles[muscle_indices[muscle_adjacency.start + j]];
                let position_a = mix(
                    positions[read_base + base + m.a0],
                    positions[read_base + base + m.a1],
                    m.anchor_a,
                );
                let position_b = mix(
                    positions[read_base + base + m.b0],
                    positions[read_base + base + m.b1],
                    m.anchor_b,
                );
                let velocity_a = mix(
                    velocities[read_base + base + m.a0],
                    velocities[read_base + base + m.a1],
                    m.anchor_a,
                );
                let velocity_b = mix(
                    velocities[read_base + base + m.b0],
                    velocities[read_base + base + m.b1],
                    m.anchor_b,
                );
                let d = position_b - position_a;
                let distance = max(length(d), 1e-6);
                let dir = d / distance;
                let relative = dot(velocity_b - velocity_a, dir);
                let magnitude = clamp(
                    clamp(distance - muscle_length(m, time), -0.25, 0.25) * m.stiffness
                        + relative * 0.15,
                    -30.0,
                    30.0,
                );
                var weight = 0.0;
                if m.a0 == local { weight += 1.0 - m.anchor_a; }
                if m.a1 == local { weight += m.anchor_a; }
                if m.b0 == local { weight -= 1.0 - m.anchor_b; }
                if m.b1 == local { weight -= m.anchor_b; }
                force += dir * magnitude * weight;
            }
            var gravity = 0.0;
            if tick >= 200u { gravity = p.gravity; }
            n.vel = (n.vel + (force / n.mass - vec2f(0.0, gravity)) / 120.0) * p.air;
            n.pos += n.vel / 120.0;
            if tick >= 200u { n = collide(n); }
            if !all(abs(n.pos) < vec2f(1e6)) || !all(abs(n.vel) < vec2f(1e6)) {
                n.failed = 1.0;
                n.pos = vec2f(0.0);
                n.vel = vec2f(0.0);
            }
        }
        positions[write_base + lane] = n.pos;
        velocities[write_base + lane] = n.vel;
        failures[lane] = n.failed;
        workgroupBarrier();

        // Each creature's first lane solves its small bone list in place.
        // This keeps the repeated constraint iterations local to one lane and
        // avoids a workgroup-wide barrier between every projection pass.
        if creature < p.count && local == 0u {
            for (var j = 0u; j < body.nodes; j++) {
                positions[read_base + base + j] = velocities[write_base + base + j];
            }
            for (var iteration = 0u; iteration < BONE_SOLVE_ITERATIONS; iteration++) {
                for (var j = 0u; j < body.bone_count; j++) {
                    let bone = bones[body.bones_start + j];
                    let index_a = write_base + base + bone.a;
                    let index_b = write_base + base + bone.b;
                    let delta = positions[index_b] - positions[index_a];
                    let raw_distance = length(delta);
                    let distance = max(raw_distance, 1e-6);
                    let error = distance - bone.rest_length;
                    let direction = select(vec2f(1.0, 0.0), delta / distance, raw_distance > 1e-6);
                    let inverse_a = 1.0 / masses[base + bone.a];
                    let inverse_b = 1.0 / masses[base + bone.b];
                    let inverse_sum = inverse_a + inverse_b;
                    let old_a = positions[index_a];
                    let old_b = positions[index_b];
                    var new_a = old_a + direction * error * inverse_a / inverse_sum;
                    var new_b = old_b - direction * error * inverse_b / inverse_sum;
                    if tick >= 200u && p.ground > 0.0 {
                        new_a.y = max(new_a.y, radii[base + bone.a]);
                        new_b.y = max(new_b.y, radii[base + bone.b]);
                    }
                    velocities[index_a] += (new_a - old_a) * 120.0;
                    velocities[index_b] += (new_b - old_b) * 120.0;
                    positions[index_a] = new_a;
                    positions[index_b] = new_b;
                }
                for (var j = 0u; j < body.bone_count; j++) {
                    let bone_a = bones[body.bones_start + j];
                    let a0 = bone_a.a;
                    let a1 = bone_a.b;
                    for (var k = j + 1u; k < body.bone_count; k++) {
                        let bone_b = bones[body.bones_start + k];
                        let b0 = bone_b.a;
                        let b1 = bone_b.b;
                        let radius_a = min(BONE_COLLISION_RADIUS, min(radii[base + a0], radii[base + a1]));
                        let radius_b = min(BONE_COLLISION_RADIUS, min(radii[base + b0], radii[base + b1]));
                        let separation = radius_a + radius_b;
                        var joint_node = 0xffffffffu;
                        if a0 == b0 || a0 == b1 {
                            joint_node = a0;
                        } else if a1 == b0 || a1 == b1 {
                            joint_node = a1;
                        }
                        let index_a0 = write_base + base + a0;
                        let index_a1 = write_base + base + a1;
                        let index_b0 = write_base + base + b0;
                        let index_b1 = write_base + base + b1;
                        let full_a0 = positions[index_a0];
                        let full_a1 = positions[index_a1];
                        let full_b0 = positions[index_b0];
                        let full_b1 = positions[index_b1];
                        var a_start = 0.0;
                        var a_end = 1.0;
                        var b_start = 0.0;
                        var b_end = 1.0;
                        if joint_node != 0xffffffffu {
                            let length_a = length(full_a1 - full_a0);
                            let length_b = length(full_b1 - full_b0);
                            let trim_a = (radii[base + joint_node] + radius_a) / max(length_a, 1e-6);
                            let trim_b = (radii[base + joint_node] + radius_b) / max(length_b, 1e-6);
                            if a0 == joint_node {
                                a_start = trim_a;
                            } else {
                                a_end = 1.0 - trim_a;
                            }
                            if b0 == joint_node {
                                b_start = trim_b;
                            } else {
                                b_end = 1.0 - trim_b;
                            }
                            if a_start >= a_end || b_start >= b_end {
                                continue;
                            }
                        }
                        let a0_pos = mix(full_a0, full_a1, a_start);
                        let a1_pos = mix(full_a0, full_a1, a_end);
                        let b0_pos = mix(full_b0, full_b1, b_start);
                        let b1_pos = mix(full_b0, full_b1, b_end);
                        if max(a0_pos.x, a1_pos.x) + separation < min(b0_pos.x, b1_pos.x)
                            || max(b0_pos.x, b1_pos.x) + separation < min(a0_pos.x, a1_pos.x)
                            || max(a0_pos.y, a1_pos.y) + separation < min(b0_pos.y, b1_pos.y)
                            || max(b0_pos.y, b1_pos.y) + separation < min(a0_pos.y, a1_pos.y) {
                            continue;
                        }
                        let closest = closest_segment_points(a0_pos, a1_pos, b0_pos, b1_pos);
                        let delta = closest.pa - closest.pb;
                        let distance = length(delta);
                        let edge_a = full_a1 - full_a0;
                        let edge_b = full_b1 - full_b0;
                        let length_a = length(edge_a);
                        let length_b = length(edge_b);
                        if distance <= 1e-6 && joint_node == 0xffffffffu {
                            // A crossing needs an asymmetric correction so the
                            // exact-length reconstruction can fold it apart.
                            var moving0 = b0;
                            var moving1 = b1;
                            var fixed0 = a0;
                            var fixed_edge = edge_a;
                            var fixed_length = length_a;
                            if length_a < length_b {
                                moving0 = a0;
                                moving1 = a1;
                                fixed0 = b0;
                                fixed_edge = edge_b;
                                fixed_length = length_b;
                            }
                            if fixed_length > 1e-6 {
                                let line_normal = vec2f(-fixed_edge.y, fixed_edge.x) / fixed_length;
                                let d0 = dot(positions[write_base + base + moving0]
                                    - positions[write_base + base + fixed0], line_normal);
                                let d1 = dot(positions[write_base + base + moving1]
                                    - positions[write_base + base + fixed0], line_normal);
                                if d0 * d1 < 0.0 {
                                    var endpoint = moving0;
                                    if masses[base + moving1] < masses[base + moving0] {
                                        endpoint = moving1;
                                    }
                                    var other_distance = d1;
                                    var current_distance = d0;
                                    if endpoint == moving1 {
                                        other_distance = d0;
                                        current_distance = d1;
                                    }
                                    let target_distance = sign(other_distance)
                                        * (abs(other_distance) + abs(current_distance) + separation);
                                    let correction = (target_distance - current_distance) * line_normal;
                                    let node_index = write_base + base + endpoint;
                                    var new_position = positions[node_index] + correction;
                                    if tick >= 200u && p.ground > 0.0 {
                                        new_position.y = max(new_position.y, radii[base + endpoint]);
                                    }
                                    velocities[node_index] += (new_position - positions[node_index]) * 120.0;
                                    positions[node_index] = new_position;
                                    continue;
                                }
                            }
                        }
                        var contact_s = a_start + closest.s * (a_end - a_start);
                        var contact_t = b_start + closest.t * (b_end - b_start);
                        var normal = vec2f(1.0, 0.0);
                        var penetration = separation - distance;
                        if distance > 1e-6 {
                            normal = delta / distance;
                        } else {
                            if length_a < length_b {
                                contact_s = a_start + 0.25 * (a_end - a_start);
                                normal = vec2f(-edge_a.y, edge_a.x) / max(length_a, 1e-6);
                            } else {
                                contact_t = b_start + 0.25 * (b_end - b_start);
                                normal = vec2f(-edge_b.y, edge_b.x) / max(length_b, 1e-6);
                            }
                            penetration = separation;
                        }
                        if penetration <= 0.0 {
                            continue;
                        }
                        let contact_weights = array<f32, 4>(
                            1.0 - contact_s,
                            contact_s,
                            1.0 - contact_t,
                            contact_t,
                        );
                        let contact_indices = array<u32, 4>(a0, a1, b0, b1);
                        let contact_signs = array<f32, 4>(1.0, 1.0, -1.0, -1.0);
                        var contact_gradients = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
                        var inverse_masses = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
                        var denominator = 0.0;
                        for (var c = 0u; c < 4u; c++) {
                            inverse_masses[c] = 1.0 / masses[base + contact_indices[c]];
                            for (var d = 0u; d < 4u; d++) {
                                if contact_indices[c] == contact_indices[d] {
                                    contact_gradients[c] += contact_signs[d] * contact_weights[d];
                                }
                            }
                            var first = true;
                            for (var d = 0u; d < c; d++) {
                                if contact_indices[c] == contact_indices[d] {
                                    first = false;
                                }
                            }
                            if first {
                                denominator += inverse_masses[c] * contact_gradients[c] * contact_gradients[c];
                            }
                        }
                        denominator = max(denominator, 1e-8);
                        for (var c = 0u; c < 4u; c++) {
                            var first = true;
                            for (var d = 0u; d < c; d++) {
                                if contact_indices[c] == contact_indices[d] {
                                    first = false;
                                }
                            }
                            if !first {
                                continue;
                            }
                            let node_index = write_base + base + contact_indices[c];
                            let correction = normal * penetration * inverse_masses[c]
                                * contact_gradients[c] / denominator;
                            var new_position = positions[node_index] + correction;
                            if tick >= 200u && p.ground > 0.0 {
                                new_position.y = max(new_position.y, radii[base + contact_indices[c]]);
                            }
                            velocities[node_index] += (new_position - positions[node_index]) * 120.0;
                            positions[node_index] = new_position;
                        }
                    }
                }
                if iteration + 1u < BONE_SOLVE_ITERATIONS {
                    for (var j = 0u; j < body.nodes; j++) {
                        velocities[read_base + base + j] = positions[write_base + base + j];
                    }
                    for (var j = 0u; j < body.bone_count; j++) {
                        let bone = bones[body.bones_start + j];
                        let parent_index = write_base + base + bone.a;
                        let child_index = write_base + base + bone.b;
                        let shape_a = velocities[read_base + base + bone.a];
                        let shape_b = velocities[read_base + base + bone.b];
                        let delta = shape_b - shape_a;
                        let raw_distance = length(delta);
                        let direction = select(vec2f(1.0, 0.0), delta / max(raw_distance, 1e-6), raw_distance > 1e-6);
                        let old_child = positions[child_index];
                        let new_child = positions[parent_index] + direction * bone.rest_length;
                        positions[child_index] = new_child;
                        velocities[child_index] += (new_child - old_child) * 120.0;
                    }
                }
            }

            // Preserve the converged joint directions, then reconstruct the
            // tree parent-first so every edge has its exact rest length.
            var target_center = vec2f(0.0);
            var mass_sum = 0.0;
            for (var j = 0u; j < body.nodes; j++) {
                let shape = positions[write_base + base + j];
                velocities[read_base + base + j] = shape;
                target_center += shape * masses[base + j];
                mass_sum += masses[base + j];
            }
            for (var j = 0u; j < body.bone_count; j++) {
                let bone = bones[body.bones_start + j];
                let parent_index = write_base + base + bone.a;
                let child_index = write_base + base + bone.b;
                let shape_a = velocities[read_base + base + bone.a];
                let shape_b = velocities[read_base + base + bone.b];
                let delta = shape_b - shape_a;
                let raw_distance = length(delta);
                let direction = select(vec2f(1.0, 0.0), delta / max(raw_distance, 1e-6), raw_distance > 1e-6);
                let old_child = positions[child_index];
                let new_child = positions[parent_index] + direction * bone.rest_length;
                positions[child_index] = new_child;
                velocities[child_index] += (new_child - old_child) * 120.0;
            }
            var current_center = vec2f(0.0);
            for (var j = 0u; j < body.nodes; j++) {
                current_center += positions[write_base + base + j] * masses[base + j];
            }
            let center_shift = (target_center - current_center) / mass_sum;
            var ground_lift = 0.0;
            for (var j = 0u; j < body.nodes; j++) {
                let node_index = write_base + base + j;
                positions[node_index] += center_shift;
                velocities[node_index] += center_shift * 120.0;
                if tick >= 200u && p.ground > 0.0 {
                    ground_lift = max(ground_lift, radii[base + j] - positions[node_index].y);
                }
            }
            if tick >= 200u && p.ground > 0.0 {
                for (var j = 0u; j < body.nodes; j++) {
                    let node_index = write_base + base + j;
                    positions[node_index].y += ground_lift;
                    velocities[node_index].y += ground_lift * 120.0;
                }
            }
            for (var j = 0u; j < body.nodes; j++) {
                let node_index = write_base + base + j;
                let base_velocity = positions[read_base + base + j];
                var correction_velocity = velocities[node_index] - base_velocity;
                let correction_speed = length(correction_velocity);
                if correction_speed > MAX_BONE_PROJECTION_SPEED {
                    correction_velocity *= MAX_BONE_PROJECTION_SPEED / correction_speed;
                }
                velocities[node_index] = base_velocity + correction_velocity;
            }
            var overlap = false;
            for (var j = 0u; j < body.bone_count; j++) {
                let bone_a = bones[body.bones_start + j];
                for (var k = j + 1u; k < body.bone_count; k++) {
                    let bone_b = bones[body.bones_start + k];
                    let radius_a = min(BONE_COLLISION_RADIUS, min(radii[base + bone_a.a], radii[base + bone_a.b]));
                    let radius_b = min(BONE_COLLISION_RADIUS, min(radii[base + bone_b.a], radii[base + bone_b.b]));
                    let minimum_distance = radius_a + radius_b - 0.001;
                    var joint_node = 0xffffffffu;
                    if bone_a.a == bone_b.a || bone_a.a == bone_b.b {
                        joint_node = bone_a.a;
                    } else if bone_a.b == bone_b.a || bone_a.b == bone_b.b {
                        joint_node = bone_a.b;
                    }
                    var a_start = 0.0;
                    var a_end = 1.0;
                    var b_start = 0.0;
                    var b_end = 1.0;
                    if joint_node != 0xffffffffu {
                        let length_a = length(positions[write_base + base + bone_a.b]
                            - positions[write_base + base + bone_a.a]);
                        let length_b = length(positions[write_base + base + bone_b.b]
                            - positions[write_base + base + bone_b.a]);
                        let trim_a = (radii[base + joint_node] + radius_a) / max(length_a, 1e-6);
                        let trim_b = (radii[base + joint_node] + radius_b) / max(length_b, 1e-6);
                        if bone_a.a == joint_node {
                            a_start = trim_a;
                        } else {
                            a_end = 1.0 - trim_a;
                        }
                        if bone_b.a == joint_node {
                            b_start = trim_b;
                        } else {
                            b_end = 1.0 - trim_b;
                        }
                        if a_start >= a_end || b_start >= b_end {
                            continue;
                        }
                    }
                    let a0 = positions[write_base + base + bone_a.a];
                    let a1 = positions[write_base + base + bone_a.b];
                    let b0 = positions[write_base + base + bone_b.a];
                    let b1 = positions[write_base + base + bone_b.b];
                    let closest = closest_segment_points(
                        mix(a0, a1, a_start),
                        mix(a0, a1, a_end),
                        mix(b0, b1, b_start),
                        mix(b0, b1, b_end),
                    );
                    if length(closest.pa - closest.pb) < minimum_distance {
                        overlap = true;
                    }
                }
            }
            if overlap {
                for (var j = 0u; j < body.nodes; j++) {
                    failures[base + j] = 1.0;
                }
            }
        }
        workgroupBarrier();
        if creature < p.count && local < body.nodes {
            n.pos = positions[write_base + lane];
            n.vel = velocities[write_base + lane];
            n.failed = failures[lane];
        }

        if creature < p.count && local == 0u {
            if tick >= 200u {
                var center_y = 0.0;
                var contacts = 0.0;
                for (var j = 0u; j < body.nodes; j++) {
                    center_y += positions[write_base + base + j].y;
                    if p.ground > 0.0
                        && positions[write_base + base + j].y <= radii[base + j] + 0.002 {
                        contacts += 1.0;
                    }
                }
                center_y /= f32(body.nodes);
                metrics.ground_contact += contacts;
                metrics.vertical_oscillation = min(metrics.vertical_oscillation, center_y);
                metrics.gait_frequency = max(metrics.gait_frequency, center_y);
                if tick == 200u {
                    metrics.previous_center_y = center_y;
                    metrics.vertical_extremum = center_y;
                    metrics.vertical_trend = 0.0;
                    metrics.gait_turns = 0.0;
                } else if (tick - 200u) % 4u == 0u {
                    let delta = center_y - metrics.previous_center_y;
                    if metrics.vertical_trend == 0.0 {
                        if abs(delta) > 0.0005 {
                            metrics.vertical_trend = select(-1.0, 1.0, delta > 0.0);
                            metrics.vertical_extremum = center_y;
                        }
                    } else if metrics.vertical_trend > 0.0 {
                        if center_y > metrics.vertical_extremum {
                            metrics.vertical_extremum = center_y;
                        } else if metrics.vertical_extremum - center_y > 0.005 {
                            metrics.gait_turns += 1.0;
                            metrics.vertical_trend = -1.0;
                            metrics.vertical_extremum = center_y;
                        }
                    } else {
                        if center_y < metrics.vertical_extremum {
                            metrics.vertical_extremum = center_y;
                        } else if center_y - metrics.vertical_extremum > 0.005 {
                            metrics.gait_turns += 1.0;
                            metrics.vertical_trend = 1.0;
                            metrics.vertical_extremum = center_y;
                        }
                    }
                    metrics.previous_center_y = center_y;
                }
            }
            if tick + 1u == p.total_steps {
                var score = 0.0;
                var failed = 0.0;
                for (var j = 0u; j < body.nodes; j++) {
                    score += positions[write_base + base + j].x;
                    failed += failures[base + j];
                }
                if failed > 0.0 {
                    metrics.fitness = -1e20;
                } else {
                    metrics.fitness = score / f32(body.nodes);
                }
                if p.total_steps > 200u {
                    metrics.vertical_oscillation = max(
                        metrics.gait_frequency - metrics.vertical_oscillation,
                        0.0,
                    );
                    metrics.gait_frequency = metrics.gait_turns * 0.5
                        / (f32(p.total_steps - 200u) / 120.0);
                } else {
                    metrics.vertical_oscillation = 0.0;
                    metrics.gait_frequency = 0.0;
                }
            }
        }
    }
    if creature < p.count && local == 0u { results[creature] = metrics; }
    if creature < p.count && local < body.nodes { nodes[creature * p.stride + local] = n; }
}
