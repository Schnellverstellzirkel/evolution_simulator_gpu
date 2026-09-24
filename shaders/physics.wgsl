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
    height_sum: f32,
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
const VELOCITY_SOLVE_ITERATIONS: u32 = 4u;
const MAX_MUSCLE_LENGTH_SPEED: f32 = 2.0;
const MAX_MUSCLE_FORCE: f32 = 5.0;
const MAX_NODE_SPEED: f32 = 5.0;
const MAX_BONE_ANGULAR_SPEED: f32 = 15.0;
const MAX_BONE_TURN_COS: f32 = 0.9921977;
const MAX_BONE_TURN_TAN: f32 = 0.12565514;

fn limited_muscle_length(m: Muscle, time: f32) -> f32 {
    let amplitude = min(
        m.long - m.short,
        2.0 * MAX_MUSCLE_LENGTH_SPEED / m.inv_period * min(m.duty, 1.0 - m.duty) / 3.14159265359,
    );
    let phase = fract(time * m.inv_period + m.phase);
    var wave: f32;
    if phase < m.duty {
        wave = 0.5 + 0.5 * cos(3.14159265359 * phase * m.inv_duty);
    } else {
        wave = 0.5 - 0.5 * cos(3.14159265359 * (phase - m.duty) * m.inv_complement);
    }
    return m.long - amplitude * (1.0 - wave);
}
fn limit_speed(velocity: vec2f) -> vec2f {
    let speed = length(velocity);
    if speed > MAX_NODE_SPEED {
        return velocity * (MAX_NODE_SPEED / speed);
    }
    return velocity;
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
    var metrics = Result(0.0, 0.0, 1e20, -1e20, 0.0, 0.0, 0.0, 0.0, 0.0);
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
                var mass_sum = 0.0;
                var low = 1e20;
                for (var j = 0u; j < body.nodes; j++) {
                    avg += positions[read_base + base + j].x * masses[base + j];
                    mass_sum += masses[base + j];
                    low = min(low, positions[read_base + base + j].y - radii[base + j]);
                }
                n.pos -= vec2f(avg / mass_sum, low);
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
                let target_speed = (limited_muscle_length(m, time)
                    - limited_muscle_length(m, max(time - 1.0 / 120.0, 0.0))) * 120.0;
                let magnitude = clamp(-target_speed * m.stiffness * 0.25
                    + relative * 0.15, -MAX_MUSCLE_FORCE, MAX_MUSCLE_FORCE);
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
            n.vel = limit_speed(n.vel);
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
                    positions[index_a] = new_a;
                    positions[index_b] = new_b;
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
                var direction = select(vec2f(1.0, 0.0), delta / max(raw_distance, 1e-6), raw_distance > 1e-6);
                let previous_delta = positions[read_base + base + bone.b]
                    - positions[read_base + base + bone.a];
                let previous_length = length(previous_delta);
                if previous_length > 1e-6 {
                    let previous_direction = previous_delta / previous_length;
                    if dot(previous_direction, direction) < MAX_BONE_TURN_COS {
                        let cross = previous_direction.x * direction.y
                            - previous_direction.y * direction.x;
                        let turn_sign = select(1.0, -1.0, cross < 0.0);
                        let turned = vec2f(
                            previous_direction.x - previous_direction.y * turn_sign * MAX_BONE_TURN_TAN,
                            previous_direction.y + previous_direction.x * turn_sign * MAX_BONE_TURN_TAN,
                        );
                        direction = turned / length(turned);
                    }
                }
                let new_child = positions[parent_index] + direction * bone.rest_length;
                positions[child_index] = new_child;
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
                if tick >= 200u && p.ground > 0.0 {
                    ground_lift = max(ground_lift, radii[base + j] - positions[node_index].y);
                }
            }
            if tick >= 200u && p.ground > 0.0 {
                for (var j = 0u; j < body.nodes; j++) {
                    let node_index = write_base + base + j;
                    positions[node_index].y += ground_lift;
                }
            }
            for (var iteration = 0u; iteration < VELOCITY_SOLVE_ITERATIONS; iteration++) {
                for (var j = 0u; j < body.bone_count; j++) {
                    let bone = bones[body.bones_start + j];
                    let index_a = write_base + base + bone.a;
                    let index_b = write_base + base + bone.b;
                    let delta = positions[index_b] - positions[index_a];
                    let length_bone = max(length(delta), 1e-6);
                    let direction = delta / length_bone;
                    let inverse_a = 1.0 / masses[base + bone.a];
                    let inverse_b = 1.0 / masses[base + bone.b];
                    let inverse_sum = inverse_a + inverse_b;
                    let relative_radial = dot(velocities[index_b] - velocities[index_a], direction);
                    let radial_impulse = relative_radial / inverse_sum;
                    velocities[index_a] += direction * radial_impulse * inverse_a;
                    velocities[index_b] -= direction * radial_impulse * inverse_b;

                    let tangent = vec2f(-direction.y, direction.x);
                    let relative_tangent = dot(velocities[index_b] - velocities[index_a], tangent);
                    let max_tangent = MAX_BONE_ANGULAR_SPEED * length_bone;
                    let limited_tangent = clamp(relative_tangent, -max_tangent, max_tangent);
                    let angular_impulse = (relative_tangent - limited_tangent) / inverse_sum;
                    velocities[index_a] += tangent * angular_impulse * inverse_a;
                    velocities[index_b] -= tangent * angular_impulse * inverse_b;
                }
                for (var j = 0u; j < body.nodes; j++) {
                    let node_index = write_base + base + j;
                    velocities[node_index] = limit_speed(velocities[node_index]);
                    if tick >= 200u && p.ground > 0.0
                        && positions[node_index].y <= radii[base + j] + 1e-5 {
                        velocities[node_index].y = max(velocities[node_index].y, 0.0);
                    }
                }
            }
        }
        workgroupBarrier();
        if creature < p.count && local < body.nodes {
            n.pos = positions[write_base + lane];
            n.vel = velocities[write_base + lane];
        }

        if creature < p.count && local == 0u {
            if tick >= 200u {
                var center_y = 0.0;
                var contacts = 0.0;
                var low = 1e20;
                var high = -1e20;
                for (var j = 0u; j < body.nodes; j++) {
                    let y = positions[write_base + base + j].y;
                    center_y += y;
                    low = min(low, y - radii[base + j]);
                    high = max(high, y + radii[base + j]);
                    if p.ground > 0.0
                        && y <= radii[base + j] + 0.002 {
                        contacts += 1.0;
                    }
                }
                center_y /= f32(body.nodes);
                metrics.ground_contact += contacts;
                metrics.height_sum += high - low;
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
                var mass_sum = 0.0;
                var failed = 0.0;
                for (var j = 0u; j < body.nodes; j++) {
                    score += positions[write_base + base + j].x * masses[base + j];
                    mass_sum += masses[base + j];
                    failed += failures[base + j];
                }
                if failed > 0.0 {
                    metrics.fitness = -1e20;
                } else {
                    let timed_steps = max(p.total_steps - 200u, 1u);
                    let mean_height = metrics.height_sum / f32(timed_steps);
                    let contact_fraction = metrics.ground_contact
                        / (f32(timed_steps) * f32(body.nodes));
                    let posture = clamp((mean_height - 0.25) / 0.75, 0.0, 1.0);
                    let stepping = clamp((0.95 - contact_fraction) / 0.20, 0.0, 1.0);
                    metrics.fitness = score / mass_sum * posture * stepping;
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
