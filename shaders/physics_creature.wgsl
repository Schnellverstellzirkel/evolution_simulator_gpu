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
    // Bump height of the rough ground; 0 is flat.
    terrain: f32,
    // Environment effect multipliers on the baseline constants: heat wave
    // shrinks the energy store, drought slows recovery. Both 1.0 in the calm
    // world and packed in the same order as creature_kernel::Params.
    muscle_energy: f32,
    muscle_recovery: f32,
    // Ground slope (rise over run), already zeroed when the ground is
    // disabled, and steady horizontal wind acceleration (m/s²).
    slope: f32,
    wind: f32,
    // Mud sink depth (m); 0.0 is dry ground. Contacting nodes sink by up to
    // this depth, which raises the effective normal push and multiplies the
    // friction budget.
    mud: f32,
    // Pit opening width (m); 0.0 is solid ground. Gaps are zeroed when the
    // ground is disabled.
    gaps: f32,
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
    // Bitmasks (as f32 bits) of nodes 0-31 and 32-63 that touched the ground.
    contact_lo: f32,
    contact_hi: f32,
    // Bitmasks of nodes that later lifted clear of the ground again.
    lift_lo: f32,
    lift_hi: f32,
    // Nodes touching the ground at the end of the previous step (f32 bits),
    // for muscle sensor touchdown events.
    ground_lo: f32,
    ground_hi: f32,
    // Seconds into the trial when the head (node 0) tipped below its neck
    // base (bone 0's child), or 0 while upright. After a fall the muscles go
    // limp and the fitness keeps the distance at the fall.
    fall_time: f32,
    // Mean head acceleration (m/s^2) over about HEAD_SHAKE_WINDOW seconds.
    head_shake: f32,
}
@group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
// Muscle genes plus per-muscle state (rhythm offset and energy), which the
// kernel updates.
@group(0) @binding(1) var<storage, read_write> muscle_data: array<f32>;
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
// Fields: packed endpoints, 10 muscle genes, sensor endpoint, reset phase,
// rhythm offset (state), energy (state).
const MUSCLE_FIELDS: u32 = 15u;
const NO_SENSOR: u32 = 255u;
// Muscle energy: stored work (J), recovery per second, and the drive left
// when exhausted.
const MUSCLE_CAPACITY: f32 = 15.0;
const MUSCLE_RECOVERY: f32 = 0.25;
const TIRED_DRIVE: f32 = 0.0;
const BONE_FIELDS: u32 = 9u;
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
// Feet sliding slower than this (m/s) count as planted (physics::PLANTED_SPEED).
const PLANTED_SPEED: f32 = 0.01;
// Extra weight per unit of grip for nodes resting on the ground during the
// constraint passes (physics::STANCE_GRIP).
const STANCE_GRIP: f32 = 10.0;
// Head shaking limit (m/s^2) and averaging window (s); physics::HEAD_SHAKE_*.
const HEAD_SHAKE_LIMIT: f32 = 78.4;
const HEAD_SHAKE_WINDOW: f32 = 0.1;
// Mud and gap geometry (physics::MUD_*, physics::GAP_*).
const MUD_NORMAL: f32 = 2.0;
const MUD_GRIP: f32 = 2.0;
const MUD_DRAG: f32 = 2.0;
const MUD_FULL_DEPTH: f32 = 0.1;
const GAP_DEPTH: f32 = 2.0;
const GAP_RUN: f32 = 0.15;
const MAX_BONE_ANGULAR_SPEED: f32 = 15.0;
const MAX_BONE_TURN_COS: f32 = TURNCOS;
const MAX_BONE_TURN_TAN: f32 = TURNTAN;

const RATE: f32 = PHYSICSRATE;
const DT: f32 = 1.0 / RATE;
const SETTLE: u32 = SETTLESTEPSu;
const SAMPLE: u32 = SAMPLEINTERVALu;
// Clearance a touching node must reach to count as a lifted foot.
const LIFT_CLEARANCE: f32 = 0.01;
// Cosine and sine of physics::JOINT_BREAK, the angle past its range at which
// a joint breaks.
const JOINT_BREAK_COS: f32 = JOINTBREAKCOS;
const JOINT_BREAK_SIN: f32 = JOINTBREAKSIN;

// Height and slope of the rough ground; mirrors physics::terrain plus the
// linear tilt of physics::terrain_with_slope and the periodic pits of
// physics::gaps.
fn terrain(x: f32) -> vec2f {
    let t0 = x * (1.0 / 1.1);
    let u0 = t0 - floor(t0);
    let w0 = u0 * (1.0 - u0);
    let t1 = x * (1.0 / 0.43) + 0.3;
    let u1 = t1 - floor(t1);
    let w1 = u1 * (1.0 - u1);
    var height = 0.65 * 16.0 * w0 * w0 + 0.35 * 16.0 * w1 * w1;
    var slope = 0.65 * 32.0 * w0 * (1.0 - 2.0 * u0) * (1.0 / 1.1)
        + 0.35 * 32.0 * w1 * (1.0 - 2.0 * u1) * (1.0 / 0.43);
    height = p.terrain * height + p.slope * x;
    slope = p.terrain * slope + p.slope;
    if p.gaps > 0.0 {
        let spacing = 2.0 + 4.0 * p.gaps;
        let center = spacing * 0.5;
        let t = x / spacing;
        let r = x - floor(t) * spacing;
        let distance = abs(r - center);
        let half = 0.5 * p.gaps;
        let run = max(min(GAP_RUN, half), 1e-6);
        let ramp = clamp((half - distance) / run, 0.0, 1.0);
        var factor = ramp;
        if distance <= half - run {
            factor = 1.0;
        } else if distance >= half {
            factor = 0.0;
        }
        let on_ramp = distance > half - run && distance < half;
        var side = -1.0;
        if r < center {
            side = 1.0;
        }
        height -= GAP_DEPTH * factor;
        if on_ramp {
            slope -= GAP_DEPTH * (side / run);
        }
    }
    return vec2f(height, slope);
}

// Whether node `n` is set in a 64-node bitmask split into two words.
fn in_mask(n: u32, lo: u32, hi: u32) -> bool {
    return select((hi >> (n - 32u)) & 1u, (lo >> n) & 1u, n < 32u) == 1u;
}
// Bone share of node a after weighting nodes on the ground by their grip.
fn stance_share(share: f32, fa: f32, fb: f32) -> f32 {
    return share * fb / (share * fb + (1.0 - share) * fa);
}
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
// Small-angle rotation shared with the CPU engine; joint corrections stay
// below one radian.
fn rotate_small(v: vec2f, angle: f32) -> vec2f {
    let a2 = angle * angle;
    let c = 1.0 - a2 * (0.5 - a2 * (1.0 / 24.0));
    let s = angle * (1.0 - a2 * ((1.0 / 6.0) - a2 * (1.0 / 120.0)));
    return vec2f(v.x * c - v.y * s, v.x * s + v.y * c);
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
    // The other endpoint's share is 1 - bone_sa.
    var bone_sa: array<f32, MAXB>;
    // Joint range: middle direction as two snorm16 values, and the cosine of
    // half the range (-1 for a free joint). bone_ka carries the reference
    // node's workgroup index in its upper 16 bits.
    var bone_center: array<u32, MAXB>;
    var bone_cos_half: array<f32, MAXB>;
    for (var j = 0u; j < MAXB; j++) {
        if j >= bone_count { break; }
        let field = tile.y + j * BONE_FIELDS * TILE + tl;
        let packed = bitcast<u32>(bone_data[field]);
        let na = packed & 0xffu;
        let nb = (packed >> 8u) & 0xffu;
        let node_a = nodes[base + na];
        let node_b = nodes[base + nb];
        let nq = (packed >> 16u) & 0xffu;
        bone_ka[j] = (na * WG + lane) | ((nq * WG + lane) << 16u);
        bone_kb[j] = nb * WG + lane;
        bone_rest[j] = bone_data[field + TILE];
        let inverse_a = 1.0 / node_a.mass;
        let inverse_b = 1.0 / node_b.mass;
        let inverse_sum = inverse_a + inverse_b;
        bone_sa[j] = inverse_a / inverse_sum;
        bone_center[j] = pack2x16snorm(vec2f(bone_data[field + 2u * TILE], bone_data[field + 3u * TILE]));
        bone_cos_half[j] = bone_data[field + 4u * TILE];
    }
    var metrics = Result(0.0, 0.0, 1e20, -1e20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if p.tick > 0u {
        metrics = results[creature];
    }

    for (var s = 0u; s < p.steps; s++) {
        let tick = p.tick + s;
        // The head's velocity before this step, for the head shaking limit.
        let head_start = vel[lane];
        if tick == SETTLE {
            var avg = 0.0;
            var mass_sum = 0.0;
            var low = 1e20;
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                avg += pos[k].x * mass[j];
                mass_sum += mass[j];
            }
            let shift_x = avg * inv_total_mass;
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                var floor_y = 0.0;
                if p.terrain > 0.0 || p.slope != 0.0 || p.gaps > 0.0 {
                    floor_y = terrain(pos[k].x - shift_x).x;
                }
                low = min(low, pos[k].y - radius[j] - floor_y);
            }
            let shift = vec2f(shift_x, low);
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

        let time = f32(max(tick, SETTLE) - SETTLE) * DT;
        for (var j = 0u; j < muscle_count; j++) {
            let field = tile.x + j * MUSCLE_FIELDS * TILE + tl;
            let packed = bitcast<u32>(muscle_data[field]);
            let offset = muscle_data[field + 13u * TILE];
            var energy = muscle_data[field + 14u * TILE];
            if tick == SETTLE {
                energy = 1.0;
            }
            let m = Muscle(
                muscle_data[field + 1u * TILE],
                muscle_data[field + 2u * TILE],
                muscle_data[field + 3u * TILE],
                muscle_data[field + 4u * TILE],
                muscle_data[field + 5u * TILE],
                muscle_data[field + 6u * TILE] + offset,
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
                - limited_muscle_length(m, max(time - DT, 0.0))) * RATE;
            // A muscle only pulls: it drives while its target shortens and goes
            // slack while the target lengthens. Its drive scales with its stored
            // energy, so an exhausted muscle does no work until it recovers.
            let drive = max(-target_speed * m.stiffness * 0.25, 0.0) * (TIRED_DRIVE + (1.0 - TIRED_DRIVE) * energy);
            var magnitude = clamp(drive + relative * 0.15, -MAX_MUSCLE_FORCE, MAX_MUSCLE_FORCE);
            if metrics.fall_time > 0.0 {
                magnitude = 0.0;
            }
            if tick >= SETTLE {
                let work = abs(magnitude * relative) * DT;
                energy = clamp(
                    energy - work / (MUSCLE_CAPACITY * p.muscle_energy)
                        + MUSCLE_RECOVERY * p.muscle_recovery * DT * (1.0 - energy),
                    0.0,
                    1.0,
                );
            }
            muscle_data[field + 14u * TILE] = energy;
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
        var wind = 0.0;
        if tick >= SETTLE {
            gravity = p.gravity;
            // A steady wind is a horizontal acceleration on every node. It is
            // a force only, never a fitness term.
            wind = p.wind;
        }
        // Integrate velocities first. The per-node speed cap must not push the
        // body: the momentum it removes is spread back over the whole body.
        var capped_momentum = vec2f(0.0);
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            let k = j * WG + lane;
            if failed[j] < 0.5 {
                let free = (vel[k] + (scr[k] * inv_mass[j] - vec2f(0.0, gravity) + vec2f(wind, 0.0)) * DT) * p.air;
                let capped = limit_speed(free);
                capped_momentum += (free - capped) * mass[j];
                scr[k] = capped;
            }
        }
        let cap_correction = capped_momentum * inv_total_mass;
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            let k = j * WG + lane;
            if failed[j] < 0.5 {
                var n = Node(pos[k], vel[k], radius[j], friction[j], mass[j], 0.0);
                n.vel = scr[k] + cap_correction;
                n.pos += n.vel * DT;
                if !all(abs(n.pos) < vec2f(1e6)) || !all(abs(n.vel) < vec2f(1e6)) {
                    failed[j] = 1.0;
                    n.pos = vec2f(0.0);
                    n.vel = vec2f(0.0);
                }
                pos[k] = n.pos;
            }
            // Velocities are rebuilt from positions after the solve; keep the
            // predicted height to measure how far the ground pushed the node,
            // and the lowest allowed center height for this step.
            vel[k] = vec2f(pos[k].y, radius[j]);
        }

        let grounded = tick >= SETTLE && p.ground > 0.0;
        if grounded {
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                if p.terrain > 0.0 || p.slope != 0.0 || p.gaps > 0.0 {
                    // Push out along the ground normal, so bumps and pit walls
                    // resist sliding.
                    let ground = terrain(pos[k].x);
                    let secant = sqrt(1.0 + ground.y * ground.y);
                    let floor_y = ground.x + radius[j] * secant - p.mud;
                    vel[k].y = floor_y;
                    let gap = floor_y - pos[k].y;
                    if gap > 0.0 {
                        let depth = gap / (secant * secant);
                        pos[k] += vec2f(-ground.y, 1.0) * depth;
                    }
                } else {
                    // Mud lowers the dry floor by the sink depth.
                    vel[k].y = radius[j] - p.mud;
                    pos[k].y = max(pos[k].y, vel[k].y);
                }
            }
        }
        // Nodes resting on the ground hold their place like planted feet: they
        // count as heavier, by their grip, when bones pull on them.
        var stance_lo = 0u;
        var stance_hi = 0u;
        if grounded {
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                if pos[k].y <= vel[k].y + 1e-4 {
                    if j < 32u {
                        stance_lo |= 1u << j;
                    } else {
                        stance_hi |= 1u << (j - 32u);
                    }
                }
            }
        }
        var com_x_before = 0.0;
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            com_x_before += pos[j * WG + lane].x * mass[j];
        }
        com_x_before *= inv_total_mass;
        for (var iteration = 0u; iteration < BONE_SOLVE_ITERATIONS; iteration++) {
            for (var j = 0u; j < MAXB; j++) {
                if j >= bone_count { break; }
                let ka = bone_ka[j] & 0xffffu;
                let kb = bone_kb[j];
                let na = ka / WG;
                let nb = kb / WG;
                let fa = select(1.0, 1.0 + STANCE_GRIP * friction[na] * p.friction, in_mask(na, stance_lo, stance_hi));
                let fb = select(1.0, 1.0 + STANCE_GRIP * friction[nb] * p.friction, in_mask(nb, stance_lo, stance_hi));
                let share = stance_share(bone_sa[j], fa, fb);
                let old_a = pos[ka];
                let old_b = pos[kb];
                let delta = old_b - old_a;
                let raw_distance = length(delta);
                let distance = max(raw_distance, 1e-6);
                let error = distance - bone_rest[j];
                // Correction along the bone: delta / distance * error.
                let correction = select(vec2f(error, 0.0), delta * (error / distance), raw_distance > 1e-6);
                var new_a = old_a + correction * share;
                var new_b = old_b - correction * (1.0 - share);
                if grounded {
                    // A clamp is the ground pushing back, so it counts toward the
                    // node's normal push (vel.x holds its height without ground).
                    vel[ka].x -= max(vel[ka].y - new_a.y, 0.0);
                    vel[kb].x -= max(vel[kb].y - new_b.y, 0.0);
                    new_a.y = max(new_a.y, vel[ka].y);
                    new_b.y = max(new_b.y, vel[kb].y);
                }
                pos[ka] = new_a;
                pos[kb] = new_b;
            }
        }

        // Joint ranges: a bone may not turn past its evolved limits
        // against its reference bone, so no joint can spin like a wheel.
        for (var j = 0u; j < MAXB; j++) {
            if j >= bone_count { break; }
            let cos_half = bone_cos_half[j];
            if cos_half <= -1.0 { continue; }
            let kn = bone_ka[j] & 0xffffu;
            let kc = bone_kb[j];
            let kq = bone_ka[j] >> 16u;
            let pivot = pos[kn];
            let u = pos[kq] - pivot;
            let v = pos[kc] - pivot;
            let norm = sqrt(dot(u, u) * dot(v, v));
            if norm < 1e-12 { continue; }
            let relative = vec2f(dot(u, v), u.x * v.y - u.y * v.x) * (1.0 / norm);
            let center = unpack2x16snorm(bone_center[j]);
            // Angle from the middle of the range, as a unit vector.
            let z = vec2f(
                relative.x * center.x + relative.y * center.y,
                relative.y * center.x - relative.x * center.y,
            );
            if z.x >= cos_half { continue; }
            let field = tile.y + j * BONE_FIELDS * TILE + tl;
            let sin_half = bone_data[field + 5u * TILE];
            let side = select(-1.0, 1.0, z.y >= 0.0);
            let sin_excess = abs(z.y) * cos_half - z.x * sin_half;
            let cos_excess = z.x * cos_half + abs(z.y) * sin_half;
            var excess = 1.0;
            if cos_excess > 0.0 {
                excess = min(sin_excess * (1.0 + sin_excess * sin_excess * (1.0 / 6.0)), 1.0);
            }
            var share = bone_data[field + 6u * TILE];
            if grounded {
                // A node resting on the ground cannot give way, so the other
                // side of the joint takes the whole correction.
                let child_down = pos[kc].y <= vel[kc].y + 1e-4;
                let reference_down = pos[kq].y <= vel[kq].y + 1e-4;
                if child_down && !reference_down {
                    share = 0.0;
                } else if reference_down && !child_down {
                    share = 1.0;
                }
            }
            let dv = rotate_small(v, -side * excess * share) - v;
            let du = rotate_small(u, side * excess * (1.0 - share)) - u;
            // Keep the three joint nodes' center of mass in place.
            let shift = dv * bone_data[field + 7u * TILE] + du * bone_data[field + 8u * TILE];
            var new_n = pivot - shift;
            var new_c = pos[kc] + dv - shift;
            var new_q = pos[kq] + du - shift;
            if grounded {
                vel[kn].x -= max(vel[kn].y - new_n.y, 0.0);
                vel[kc].x -= max(vel[kc].y - new_c.y, 0.0);
                vel[kq].x -= max(vel[kq].y - new_q.y, 0.0);
                new_n.y = max(new_n.y, vel[kn].y);
                new_c.y = max(new_c.y, vel[kc].y);
                new_q.y = max(new_q.y, vel[kq].y);
            }
            pos[kn] = new_n;
            pos[kc] = new_c;
            pos[kq] = new_q;
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
            let ka = bone_ka[j] & 0xffffu;
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
                ground_lift = max(ground_lift, vel[k].y - shifted.y);
            }
        }
        if grounded {
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                pos[k].y += ground_lift;
            }
        }
        // Planted feet push the body along through the bone passes. That is
        // ground friction, so it may move the body's center of mass at most mu
        // times the ground's normal push this step (each node's push, plus the
        // whole-body lift for the rest of the body). Only planted feet may push
        // the body forward: while the feet slide, friction can only oppose
        // their slide, so a sliding body cannot propel itself. Beyond that, the
        // excess is taken back as a rigid shift.
        if grounded {
            var held_mass = 0.0;
            var held_grip = 0.0;
            var normal = 0.0;
            var com_x = 0.0;
            var slide = 0.0;
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                if pos[k].y <= vel[k].y + 1e-4 && failed[j] < 0.5 {
                    held_mass += mass[j];
                    held_grip += mass[j] * friction[j];
                    normal += mass[j] * max(pos[k].y - vel[k].x, 0.0);
                    slide += mass[j] * (pos[k].x - old[k].x);
                }
                com_x += pos[k].x * mass[j];
            }
            normal += (total_mass - held_mass) * ground_lift;
            let mu = held_grip / max(held_mass, 1e-6) * p.friction;
            let allowed = mu * normal * inv_total_mass;
            slide /= max(held_mass, 1e-6);
            let planted = PLANTED_SPEED * DT;
            let low = select(-allowed, 0.0, slide < -planted);
            let high = select(allowed, 0.0, slide > planted);
            let shift = com_x * inv_total_mass - com_x_before;
            let excess = shift - clamp(shift, low, high);
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                pos[j * WG + lane].x -= excess;
            }
        }
        var contact_mass = 0.0;
        var contact_momentum = 0.0;
        var contact_grip = 0.0;
        // Velocity is the actual movement over the step. Ground friction uses the
        // real upward push the node received, so grip needs real pressure. The
        // whole-body lift only moves the body out of the ground; it adds no
        // upward speed, or a limb swung into the ground would launch it.
        for (var j = 0u; j < MAXN; j++) {
            if j >= body_nodes { break; }
            let k = j * WG + lane;
            let predicted_y = vel[k].x;
            let floor_y = vel[k].y;
            var velocity = (pos[k] - old[k]) * RATE;
            velocity.y -= ground_lift * RATE;
            // The previous position is no longer needed; keep the floor height
            // for the velocity passes and contact metrics.
            old[k] = vec2f(floor_y, 0.0);
            // How deep the node sits below its dry floor, in meters, capped
            // at the local mud depth. The multipliers scale with it up to
            // MUD_FULL_DEPTH.
            var sink = 0.0;
            var mud_mu = 1.0;
            if p.mud > 0.0 {
                sink = clamp(floor_y + p.mud - pos[k].y, 0.0, p.mud) / MUD_FULL_DEPTH;
                mud_mu = 1.0 + MUD_GRIP * sink;
            }
            if grounded && pos[k].y <= floor_y + 1e-4 {
                let push = max(pos[k].y - predicted_y, 0.0) * (1.0 + MUD_NORMAL * sink);
                let max_change = friction[j] * p.friction * mud_mu * push * RATE;
                velocity.x -= clamp(velocity.x, -max_change, max_change);
                // Moving through mud also loses speed to viscous drag.
                velocity.x *= max(0.0, 1.0 - MUD_DRAG * DT * sink);
                if failed[j] < 0.5 {
                    contact_mass += mass[j];
                    contact_momentum += mass[j] * velocity.x;
                    contact_grip += mass[j] * friction[j] * mud_mu;
                }
            } else if sink > 0.0 {
                velocity.x *= max(0.0, 1.0 - MUD_DRAG * DT * sink);
            }
            if failed[j] >= 0.5 {
                velocity = vec2f(0.0);
            }
            vel[k] = velocity;
        }
        // The whole-body lift is the ground holding the body up: its normal
        // impulse is the body's mass times the lift. Each node's own friction
        // only sees its own push, so the feet on the ground also resist the
        // body's sliding with up to mu times the lift's impulse, applied to
        // the whole body so momentum stays exact.
        if grounded && ground_lift > 0.0 && contact_mass > 0.0 {
            let inv_contact = 1.0 / contact_mass;
            let budget = contact_grip * inv_contact * p.friction * ground_lift * RATE;
            let slide = contact_momentum * inv_contact;
            let change = -clamp(slide, -budget, budget);
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                if failed[j] < 0.5 {
                    vel[j * WG + lane].x += change;
                }
            }
        }
        for (var iteration = 0u; iteration < VELOCITY_SOLVE_ITERATIONS; iteration++) {
            for (var j = 0u; j < MAXB; j++) {
                if j >= bone_count { break; }
                let ka = bone_ka[j] & 0xffffu;
                let kb = bone_kb[j];
                let delta = pos[kb] - pos[ka];
                let length_bone = max(length(delta), 1e-6);
                let direction = delta * (1.0 / length_bone);
                let share_a = bone_sa[j];
                let share_b = 1.0 - share_a;
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
            var removed = vec2f(0.0);
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                let velocity = limit_speed(vel[k]);
                removed += (vel[k] - velocity) * mass[j];
                vel[k] = velocity;
            }
            let correction = removed * inv_total_mass;
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let k = j * WG + lane;
                var velocity = vel[k] + correction;
                if grounded && pos[k].y <= old[k].x + 1e-5 {
                    velocity.y = max(velocity.y, 0.0);
                }
                vel[k] = velocity;
            }
        }

        if tick >= SETTLE {
            var center_y = 0.0;
            var contacts = 0.0;
            var low = 1e20;
            var high = -1e20;
            for (var j = 0u; j < MAXN; j++) {
                if j >= body_nodes { break; }
                let y = pos[j * WG + lane].y;
                let floor_y = old[j * WG + lane].x;
                center_y += y;
                low = min(low, y - radius[j]);
                high = max(high, y + radius[j]);
                if p.ground > 0.0 {
                    if y <= floor_y + 0.002 {
                        contacts += 1.0;
                        if j < 32u {
                            metrics.contact_lo = bitcast<f32>(bitcast<u32>(metrics.contact_lo) | (1u << j));
                        } else {
                            metrics.contact_hi = bitcast<f32>(bitcast<u32>(metrics.contact_hi) | (1u << (j - 32u)));
                        }
                    } else if y > floor_y + LIFT_CLEARANCE {
                        // A foot must leave the ground after touching it; a
                        // dragged node never does.
                        if j < 32u {
                            let touched = bitcast<u32>(metrics.contact_lo) & (1u << j);
                            metrics.lift_lo = bitcast<f32>(bitcast<u32>(metrics.lift_lo) | touched);
                        } else {
                            let touched = bitcast<u32>(metrics.contact_hi) & (1u << (j - 32u));
                            metrics.lift_hi = bitcast<f32>(bitcast<u32>(metrics.lift_hi) | touched);
                        }
                    }
                }
            }
            center_y *= inv_nodes;
            // Touchdown: a node grounded now that was not after the last step.
            var now_lo = 0u;
            var now_hi = 0u;
            if p.ground > 0.0 {
                for (var j = 0u; j < MAXN; j++) {
                    if j >= body_nodes { break; }
                    if pos[j * WG + lane].y <= old[j * WG + lane].x + 0.002 {
                        if j < 32u { now_lo |= 1u << j; } else { now_hi |= 1u << (j - 32u); }
                    }
                }
            }
            let down_lo = now_lo & ~bitcast<u32>(metrics.ground_lo);
            let down_hi = now_hi & ~bitcast<u32>(metrics.ground_hi);
            metrics.ground_lo = bitcast<f32>(now_lo);
            metrics.ground_hi = bitcast<f32>(now_hi);
            if (down_lo | down_hi) != 0u && tick > SETTLE {
                // Muscles sensing a touchdown restart their rhythm at their
                // reset phase from the next step on.
                let next_time = time + DT;
                for (var j = 0u; j < muscle_count; j++) {
                    let field = tile.x + j * MUSCLE_FIELDS * TILE + tl;
                    let sensor = bitcast<u32>(muscle_data[field + 11u * TILE]);
                    if sensor == NO_SENSOR {
                        continue;
                    }
                    let packed = bitcast<u32>(muscle_data[field]);
                    let node = (packed >> (8u * sensor)) & 0xffu;
                    let touched = select((down_hi >> (node - 32u)) & 1u, (down_lo >> node) & 1u, node < 32u);
                    if touched == 1u {
                        let clock = next_time * muscle_data[field + 5u * TILE] + muscle_data[field + 6u * TILE];
                        let reset = muscle_data[field + 12u * TILE];
                        muscle_data[field + 13u * TILE] = fract(reset - clock);
                    }
                }
            }
            // A joint forced far past its range breaks, which also ends the
            // trial.
            var broken = false;
            for (var j = 0u; j < MAXB; j++) {
                if j >= bone_count || metrics.fall_time != 0.0 { break; }
                let cos_half = bone_cos_half[j];
                if cos_half <= -1.0 { continue; }
                let pivot = pos[bone_ka[j] & 0xffffu];
                let u = pos[bone_ka[j] >> 16u] - pivot;
                let v = pos[bone_kb[j]] - pivot;
                let norm = sqrt(dot(u, u) * dot(v, v));
                if norm < 1e-12 { continue; }
                let relative = vec2f(dot(u, v), u.x * v.y - u.y * v.x) * (1.0 / norm);
                let center = unpack2x16snorm(bone_center[j]);
                let sin_half = bone_data[tile.y + j * BONE_FIELDS * TILE + tl + 5u * TILE];
                let limit = cos_half * JOINT_BREAK_COS - sin_half * JOINT_BREAK_SIN;
                if relative.x * center.x + relative.y * center.y < limit {
                    broken = true;
                }
            }
            // A head shaken too hard kills the creature: its acceleration,
            // averaged over about HEAD_SHAKE_WINDOW seconds, may not pass the limit.
            if metrics.fall_time == 0.0 && time >= HEAD_SHAKE_WINDOW {
                let head_accel = length(vel[lane] - head_start) * RATE;
                metrics.head_shake += (head_accel - metrics.head_shake)
                    * min(1.0, 1.0 / (HEAD_SHAKE_WINDOW * RATE));
            }
            if metrics.fall_time == 0.0
                && (pos[lane].y < pos[bone_kb[0]].y || broken || metrics.head_shake > HEAD_SHAKE_LIMIT) {
                var fall_x = 0.0;
                for (var j = 0u; j < MAXN; j++) {
                    if j >= body_nodes { break; }
                    fall_x += pos[j * WG + lane].x * mass[j];
                }
                metrics.fall_time = time + DT;
                metrics.fitness = fall_x * inv_total_mass;
            }
            metrics.ground_contact += contacts;
            metrics.height_sum += high - low;
            metrics.vertical_oscillation = min(metrics.vertical_oscillation, center_y);
            metrics.gait_frequency = max(metrics.gait_frequency, center_y);
            if tick == SETTLE {
                metrics.previous_center_y = center_y;
                metrics.vertical_extremum = center_y;
                metrics.vertical_trend = 0.0;
                metrics.gait_turns = 0.0;
            } else if (tick - SETTLE) % SAMPLE == 0u {
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
            } else if metrics.fall_time == 0.0 {
                // Fitness is distance only; gait style is left to the niches.
                metrics.fitness = score / mass_sum;
            }
            if p.total_steps > SETTLE {
                metrics.vertical_oscillation = max(
                    metrics.gait_frequency - metrics.vertical_oscillation,
                    0.0,
                );
                metrics.gait_frequency = metrics.gait_turns * 0.5
                    / (f32(p.total_steps - SETTLE) / RATE);
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
