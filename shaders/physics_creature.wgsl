// One creature per lane. Node state lives in workgroup memory laid out as
// [node][lane], so every per-creature node index is bank-conflict free.
// Node and bone loops run to the bucket's compile-time bound and exit early,
// so per-node and per-bone constants can stay in registers.
// Muscles and bones are packed in 32-creature tiles laid out as
// [item][field][lane], so each warp load is one coalesced line.
// Every expression mirrors shaders/physics.wgsl; only the work distribution
// differs (each muscle is evaluated once and scattered in genome order).
struct Node {
    pos: vec2f,
    vel: vec2f,
    radius: f32,
    friction: f32,
    mass: f32,
    failed: f32,
}
struct Muscle {
    anchor_a: f32,
    anchor_b: f32,
    // Waveform amplitude, min(long - short, speed-limited amplitude), from packing.
    amplitude: f32,
    long: f32,
    inv_period: f32,
    phase: f32,
    duty: f32,
    stiffness: f32,
    inv_duty: f32,
    inv_complement: f32,
}
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
    vertical_oscillation: f32,
    gait_frequency: f32,
    previous_center_y: f32,
    vertical_extremum: f32,
    vertical_trend: f32,
    gait_turns: f32,
    height_sum: f32,
}
@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> muscle_data: array<f32>;
@group(0) @binding(2) var<storage, read> bone_data: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read_write> results: array<Result>;
// (nodes, bones, muscles, unused) per creature.
@group(0) @binding(5) var<storage, read> creature_info: array<vec4u>;
// (muscle base, bone base, unused, unused) per 32-creature tile.
@group(0) @binding(6) var<storage, read> tile_info: array<vec4u>;

const WG: u32 = WGSIZEu;
const MAXN: u32 = MAXNODESu;
const TILE: u32 = 32u;
const MUSCLE_FIELDS: u32 = 11u;
const BONE_FIELDS: u32 = 2u;
const MAXB: u32 = MAXN - 1u;

var<workgroup> pos: array<vec2f, SHAREDLEN>;
var<workgroup> vel: array<vec2f, SHAREDLEN>;
var<workgroup> old: array<vec2f, SHAREDLEN>;
var<workgroup> scr: array<vec2f, SHAREDLEN>;

const BONE_SOLVE_ITERATIONS: u32 = 8u;
const VELOCITY_SOLVE_ITERATIONS: u32 = 4u;
const MAX_MUSCLE_LENGTH_SPEED: f32 = 2.0;
const MAX_MUSCLE_FORCE: f32 = 5.0;
const MAX_NODE_SPEED: f32 = 5.0;
const MAX_BONE_ANGULAR_SPEED: f32 = 15.0;
const MAX_BONE_TURN_COS: f32 = 0.9921977;
const MAX_BONE_TURN_TAN: f32 = 0.12565514;

const DT: f32 = 1.0 / 120.0;

fn limited_muscle_length(m: Muscle, time: f32) -> f32 {
    let amplitude = m.amplitude;
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

@compute @workgroup_size(WG)
fn advance(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
    let creature = group.x * WG + lane;
    if creature >= p.count {
        return;
    }
    let info = creature_info[creature];
    let body_nodes = info.x;
    let bone_count = info.y;
    let muscle_count = info.z;
    let tile = tile_info[creature / TILE];
    let tl = creature % TILE;
    let base = creature * MAXN;

    var radius: array<f32, MAXN>;
    var mass: array<f32, MAXN>;
    var friction: array<f32, MAXN>;
    var failed: array<f32, MAXN>;
    var inv_mass: array<f32, MAXN>;
    var total_mass = 0.0;
    for (var j = 0u; j < MAXN; j++) {
        if j >= body_nodes { break; }
        let n = nodes[base + j];
        let k = j * WG + lane;
        pos[k] = n.pos;
        vel[k] = n.vel;
        radius[j] = n.radius;
        mass[j] = n.mass;
        friction[j] = n.friction;
        failed[j] = n.failed;
        inv_mass[j] = 1.0 / n.mass;
        total_mass += n.mass;
    }
    let inv_total_mass = 1.0 / total_mass;
    let inv_nodes = 1.0 / f32(body_nodes);
    // Bone endpoints as workgroup indices, plus the per-endpoint constants the
    // solver reads on every pass.
    var bone_ka: array<u32, MAXB>;
    var bone_kb: array<u32, MAXB>;
    var bone_rest: array<f32, MAXB>;
    // Mass shares of each bone's correction: inverse mass over their sum.
    var bone_sa: array<f32, MAXB>;
    var bone_sb: array<f32, MAXB>;
    var bone_ra: array<f32, MAXB>;
    var bone_rb: array<f32, MAXB>;
    for (var j = 0u; j < MAXB; j++) {
        if j >= bone_count { break; }
        let field = tile.y + j * BONE_FIELDS * TILE + tl;
        let packed = bitcast<u32>(bone_data[field]);
        let na = packed & 0xffu;
        let nb = packed >> 8u;
        let node_a = nodes[base + na];
        let node_b = nodes[base + nb];
        bone_ka[j] = na * WG + lane;
        bone_kb[j] = nb * WG + lane;
        bone_rest[j] = bone_data[field + TILE];
        let inverse_a = 1.0 / node_a.mass;
        let inverse_b = 1.0 / node_b.mass;
        let inverse_sum = inverse_a + inverse_b;
        bone_sa[j] = inverse_a / inverse_sum;
        bone_sb[j] = inverse_b / inverse_sum;
        bone_ra[j] = node_a.radius;
        bone_rb[j] = node_b.radius;
    }
    var metrics = Result(0.0, 0.0, 1e20, -1e20, 0.0, 0.0, 0.0, 0.0, 0.0);
    if p.tick > 0u {
        metrics = results[creature];
    }

    for (var s = 0u; s < p.steps; s++) {
        let tick = p.tick + s;
        if tick == 200u {
            var avg = 0.0;
            var mass_sum = 0.0;
            var low = 1e20;
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                avg += pos[k].x * mass[j];
                mass_sum += mass[j];
                low = min(low, pos[k].y - radius[j]);
            }
            let shift = vec2f(avg * inv_total_mass, low);
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                let shifted_start = pos[k] - shift;
                pos[k] = shifted_start;
                vel[k] = vec2f(0.0);
            }
        }
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            let k = j * WG + lane;
            old[k] = pos[k];
            scr[k] = vec2f(0.0);
        }

        let time = f32(max(tick, 200u) - 200u) * DT;
        for (var j = 0u; j < muscle_count; j++) {
            let field = tile.x + j * MUSCLE_FIELDS * TILE + tl;
            let packed = bitcast<u32>(muscle_data[field]);
            let m = Muscle(
                muscle_data[field + 1u * TILE],
                muscle_data[field + 2u * TILE],
                muscle_data[field + 3u * TILE],
                muscle_data[field + 4u * TILE],
                muscle_data[field + 5u * TILE],
                muscle_data[field + 6u * TILE],
                muscle_data[field + 7u * TILE],
                muscle_data[field + 8u * TILE],
                muscle_data[field + 9u * TILE],
                muscle_data[field + 10u * TILE],
            );
            let ka0 = (packed & 0xffu) * WG + lane;
            let ka1 = ((packed >> 8u) & 0xffu) * WG + lane;
            let kb0 = ((packed >> 16u) & 0xffu) * WG + lane;
            let kb1 = (packed >> 24u) * WG + lane;
            let position_a = mix(pos[ka0], pos[ka1], m.anchor_a);
            let position_b = mix(pos[kb0], pos[kb1], m.anchor_b);
            let velocity_a = mix(vel[ka0], vel[ka1], m.anchor_a);
            let velocity_b = mix(vel[kb0], vel[kb1], m.anchor_b);
            let d = position_b - position_a;
            let dir = d * (1.0 / max(length(d), 1e-6));
            let relative = dot(velocity_b - velocity_a, dir);
            let target_speed = (limited_muscle_length(m, time)
                - limited_muscle_length(m, max(time - DT, 0.0))) * 120.0;
            let magnitude = clamp(-target_speed * m.stiffness * 0.25
                + relative * 0.15, -MAX_MUSCLE_FORCE, MAX_MUSCLE_FORCE);
            let push = dir * magnitude;
            let a0 = packed & 0xffu;
            let a1 = (packed >> 8u) & 0xffu;
            let b0 = (packed >> 16u) & 0xffu;
            let b1 = packed >> 24u;
            for (var e = 0u; e < 4u; e++) {
                let node = (packed >> (8u * e)) & 0xffu;
                if (e >= 1u && node == a0) || (e >= 2u && node == a1) || (e == 3u && node == b0) {
                    continue;
                }
                var weight = 0.0;
                if a0 == node { weight += 1.0 - m.anchor_a; }
                if a1 == node { weight += m.anchor_a; }
                if b0 == node { weight -= 1.0 - m.anchor_b; }
                if b1 == node { weight -= m.anchor_b; }
                let f = push * weight;
                let k = node * WG + lane;
                scr[k] += f;
            }
        }

        var gravity = 0.0;
        if tick >= 200u { gravity = p.gravity; }
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            let k = j * WG + lane;
            if failed[j] < 0.5 {
                var n = Node(pos[k], vel[k], radius[j], friction[j], mass[j], 0.0);
                let force = scr[k];
                n.vel = (n.vel + (force * inv_mass[j] - vec2f(0.0, gravity)) * DT) * p.air;
                n.vel = limit_speed(n.vel);
                n.pos += n.vel * DT;
                if tick >= 200u { n = collide(n); }
                if !all(abs(n.pos) < vec2f(1e6)) || !all(abs(n.vel) < vec2f(1e6)) {
                    failed[j] = 1.0;
                    n.pos = vec2f(0.0);
                    n.vel = vec2f(0.0);
                }
                pos[k] = n.pos;
                vel[k] = n.vel;
            }
        }

        let grounded = tick >= 200u && p.ground > 0.0;
        for (var iteration = 0u; iteration < BONE_SOLVE_ITERATIONS; iteration++) {
            for (var j = 0u; j < MAXB; j++) {
                if j >= bone_count { break; }
                let ka = bone_ka[j];
                let kb = bone_kb[j];
                let old_a = pos[ka];
                let old_b = pos[kb];
                let delta = old_b - old_a;
                let raw_distance = length(delta);
                let distance = max(raw_distance, 1e-6);
                let error = distance - bone_rest[j];
                // Correction along the bone: delta / distance * error.
                let correction = select(vec2f(error, 0.0), delta * (error / distance), raw_distance > 1e-6);
                var new_a = old_a + correction * bone_sa[j];
                var new_b = old_b - correction * bone_sb[j];
                if grounded {
                    new_a.y = max(new_a.y, bone_ra[j]);
                    new_b.y = max(new_b.y, bone_rb[j]);
                }
                pos[ka] = new_a;
                pos[kb] = new_b;
            }
        }

        // Preserve the converged joint directions, then reconstruct the
        // tree parent-first so every edge has its exact rest length.
        var target_center = vec2f(0.0);
        var mass_sum = 0.0;
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            let k = j * WG + lane;
            let shape = pos[k];
            scr[k] = shape;
            target_center += shape * mass[j];
            mass_sum += mass[j];
        }
        for (var j = 0u; j < MAXB; j++) {
            if j >= bone_count { break; }
            let ka = bone_ka[j];
            let kb = bone_kb[j];
            let delta = scr[kb] - scr[ka];
            let raw_distance = length(delta);
            var direction = select(vec2f(1.0, 0.0), delta * (1.0 / max(raw_distance, 1e-6)), raw_distance > 1e-6);
            let previous_delta = old[kb] - old[ka];
            let previous_length = length(previous_delta);
            if previous_length > 1e-6 {
                let previous_direction = previous_delta * (1.0 / previous_length);
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
            let new_child = pos[ka] + direction * bone_rest[j];
            pos[kb] = new_child;
        }
        var current_center = vec2f(0.0);
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            let k = j * WG + lane;
            current_center += pos[k] * mass[j];
        }
        let center_shift = (target_center - current_center) * inv_total_mass;
        var ground_lift = 0.0;
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            let k = j * WG + lane;
            let shifted = pos[k] + center_shift;
            pos[k] = shifted;
            if grounded {
                ground_lift = max(ground_lift, radius[j] - shifted.y);
            }
        }
        if grounded {
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                pos[k].y += ground_lift;
            }
        }
        for (var iteration = 0u; iteration < VELOCITY_SOLVE_ITERATIONS; iteration++) {
            for (var j = 0u; j < MAXB; j++) {
                if j >= bone_count { break; }
                let ka = bone_ka[j];
                let kb = bone_kb[j];
                let delta = pos[kb] - pos[ka];
                let length_bone = max(length(delta), 1e-6);
                let direction = delta * (1.0 / length_bone);
                let share_a = bone_sa[j];
                let share_b = bone_sb[j];
                var velocity_a = vel[ka];
                var velocity_b = vel[kb];
                // Impulse / inverse-mass sum * inverse mass = relative speed * share.
                let radial = direction * dot(velocity_b - velocity_a, direction);
                velocity_a += radial * share_a;
                velocity_b -= radial * share_b;

                let tangent = vec2f(-direction.y, direction.x);
                let relative_tangent = dot(velocity_b - velocity_a, tangent);
                let max_tangent = MAX_BONE_ANGULAR_SPEED * length_bone;
                let limited_tangent = clamp(relative_tangent, -max_tangent, max_tangent);
                let angular = tangent * (relative_tangent - limited_tangent);
                velocity_a += angular * share_a;
                velocity_b -= angular * share_b;
                vel[ka] = velocity_a;
                vel[kb] = velocity_b;
            }
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                var velocity = limit_speed(vel[k]);
                if grounded && pos[k].y <= radius[j] + 1e-5 {
                    velocity.y = max(velocity.y, 0.0);
                }
                vel[k] = velocity;
            }
        }

        if tick >= 200u {
            var center_y = 0.0;
            var contacts = 0.0;
            var low = 1e20;
            var high = -1e20;
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let y = pos[j * WG + lane].y;
                center_y += y;
                low = min(low, y - radius[j]);
                high = max(high, y + radius[j]);
                if p.ground > 0.0
                    && y <= radius[j] + 0.002 {
                    contacts += 1.0;
                }
            }
            center_y *= inv_nodes;
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
            var failures = 0.0;
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                score += pos[j * WG + lane].x * mass[j];
                mass_sum += mass[j];
                failures += failed[j];
            }
            if failures > 0.0 {
                metrics.fitness = -1e20;
            } else {
                let timed_steps = max(p.total_steps - 200u, 1u);
                let mean_height = metrics.height_sum / f32(timed_steps);
                let contact_fraction = metrics.ground_contact
                    / (f32(timed_steps) * f32(body_nodes));
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
    results[creature] = metrics;
    for (var j = 0u; j < MAXN; j++) {
        if j >= body_nodes { break; }
        let k = j * WG + lane;
        var n = nodes[base + j];
        n.pos = pos[k];
        n.vel = vel[k];
        n.failed = failed[j];
        nodes[base + j] = n;
    }
}
