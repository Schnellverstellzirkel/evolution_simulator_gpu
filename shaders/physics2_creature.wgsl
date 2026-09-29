// Physics v2 (src/physics2.rs) on the GPU: one creature per lane.
//
// A creature is a tree of point masses on rigid, massless bones. Its state
// is the head's position and velocity, the neck's angle and turning rate, and
// one angle and rate per other bone relative to its parent bone. Nodes are
// packed so that bone j ends at node j + 1 (node 0 is the head), and every
// bone's parent bone has a smaller index, so passes over the bones run
// parents first or children first by index. The step follows
// `physics2::simulate_step_inner`.
//
// Memory. Every private array is indexed only by loop counters, so with the
// loops unrolled it can live in registers. Everything a creature reaches
// through data (a bone's pivot or parent, a muscle's nodes, a contact's
// node) goes through a per-lane table in workgroup memory, laid out
// [field][lane]: every node's position and velocity, then six scratch
// fields per bone whose use changes through the step. A bone's parent
// receives its articulated inertia through selects over the bones before
// it, so that pass stays in registers too.
//
// Per creature and node record (8 floats, `STRIDE` records per creature):
//   record 0: head position, head velocity, the head's contact force of the
//             last step (normal, friction), unused.
//   record r: bone r - 1's angle and rate (the neck's are absolute), node r's
//             contact force of the last step, unused.
// Bone constants (`BONE_FIELDS` per bone, 32-creature tiles [bone][field][lane]):
//   pivot node, length, joint range low and high, child node mass, radius and
//   friction, and on bone 0 the head's mass, radius (fields 7, 8) and
//   friction (field 2; the neck has no range).
// Muscle constants and state (`MUSCLE_FIELDS` per muscle, same tiling):
//   nodes (a0 | a1 << 6 | b0 << 12 | b1 << 18, the sensor's endpoint << 24
//   or 7 for none), anchors, waveform amplitude, Hill factor, 1/period,
//   phase, duty, stiffness, 1/duty, 1/(1 - duty), the step's force
//   (scratch), reset phase, rhythm offset (state), energy (state), longest
//   strength (the muscle's force cap and energy store over the fixed ones),
//   tendon stiffness (N/m), longest length (where the tendon starts to pull).
// Bone 0's joint range high (the neck has none) holds the creature's muscle
// scale: each muscle's force cap and energy store over the fixed ones.
struct Record {
    a: vec2f,
    b: vec2f,
    c: vec2f,
    d: vec2f,
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
    terrain: f32,
    muscle_energy: f32,
    muscle_recovery: f32,
    slope: f32,
    wind: f32,
    mud: f32,
    gaps: f32,
    hurdles: f32,
    quake: f32,
    screen_tick: u32,
    screen_bar: f32,
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
    contact_lo: f32,
    contact_hi: f32,
    lift_lo: f32,
    lift_hi: f32,
    ground_lo: f32,
    ground_hi: f32,
    fall_time: f32,
    head_shake: f32,
    screen_x: f32,
    screened: f32,
}
@group(0) @binding(0) var<storage, read_write> records: array<Record>;
@group(0) @binding(1) var<storage, read_write> muscle_data: array<f32>;
@group(0) @binding(2) var<storage, read> bone_data: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read_write> results: array<Result>;
@group(0) @binding(5) var<storage, read> creature_info: array<vec4u>;
@group(0) @binding(6) var<storage, read> tile_info: array<vec4u>;

const WG: u32 = WGSIZEu;
const MAXN: u32 = MAXNODESu;
const MAXB: u32 = MAXN - 1u;
// Most nodes in one step's contact solve, and its rows (normal, friction).
const MAXC: u32 = MAXCONTACTSu;
const MAXR: u32 = 2u * MAXC;
const TILE: u32 = 32u;
const MUSCLE_FIELDS: u32 = 18u;
const BONE_FIELDS: u32 = 9u;
const NO_SENSOR: u32 = 7u;

const RATE: f32 = PHYSICSRATE;
const DT: f32 = 1.0 / RATE;
const SETTLE: u32 = SETTLESTEPSu;
const SAMPLE: u32 = SAMPLEINTERVALu;
const MUSCLE_CAPACITY: f32 = 120.0;
const MUSCLE_RECOVERY: f32 = 0.5;
const MAX_MUSCLE_FORCE: f32 = 100.0;
const INV_JOINT_DAMPING: f32 = 10.0;
const LIMIT_HARDNESS: f32 = 20.0;
const JOINT_BREAK: f32 = 0.5;
const SPIN_CAP: f32 = 15.0;
const INV_SPIN_CAP: f32 = 0.06666667;
const SPIN_HARDNESS: f32 = 20.0;
const PGS_SWEEPS: u32 = 20u;
const PLANT_SWEEPS: u32 = 20u;
const PLANT_ROUNDS: u32 = 2u;
const AIR_DRAG: f32 = 0.6;
const WARM: bool = false;
const PUSH_OUT: f32 = 0.2;
const MUD_NORMAL: f32 = 2.0;
const MUD_GRIP: f32 = 2.0;
const MUD_DRAG: f32 = 2.0;
const MUD_FULL_DEPTH: f32 = 0.1;
const HEAD_SHAKE_LIMIT: f32 = 78.4;
const HEAD_SHAKE_WINDOW: f32 = 0.1;
const CONTACT_SLACK: f32 = 0.002;
const LIFT_CLEARANCE: f32 = 0.01;
const GAP_DEPTH: f32 = 2.0;
const GAP_RUN: f32 = 0.15;
const HURDLE_SPACING: f32 = 3.0;
const HURDLE_TOP: f32 = 1.2;
const HURDLE_RUN: f32 = 0.2;
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// The per-lane table: 4 fields per node (position, velocity), then 6 per
// bone (two scratch vectors).
const TN: u32 = 0u;
const TB: u32 = 4u * MAXN;
const TABF: u32 = 4u * MAXN + 6u * MAXB;
var<workgroup> tab: array<f32, TABF * WG>;

var<private> lane_id: u32;
var<private> bone_base: u32;
var<private> record_base: u32;
var<private> nn: u32;
var<private> nb: u32;
var<private> mass: array<f32, MAXN>;
var<private> pivot: array<u32, MAXB>;
var<private> len: array<f32, MAXB>;
var<private> total_mass: f32;
var<private> inv_mass: f32;
var<private> phase_q: f32;
var<private> amplitude: f32;
var<private> rough: bool;
// State.
var<private> x0: vec2f;
var<private> v0: vec2f;
var<private> q: array<f32, MAXB>;
var<private> qd: array<f32, MAXB>;
// Absolute turning rates, from the kinematics.
var<private> om: array<f32, MAXB>;
// The articulated-body pass: inertias as (ww, wx, wy) and (xx, xy, yy),
// bias forces, pivot arms (the joint axis is (1, arm.y, -arm.x)), U, 1/D and
// u of each joint, accelerations.
var<private> i0: array<vec3f, MAXB>;
var<private> i1: array<vec3f, MAXB>;
var<private> bias: array<vec3f, MAXB>;
var<private> arm: array<vec2f, MAXB>;
var<private> uvec: array<vec3f, MAXB>;
var<private> dinv: array<f32, MAXB>;
var<private> uu: array<f32, MAXB>;
var<private> acc: array<vec3f, MAXB>;
var<private> qdd: array<f32, MAXB>;
var<private> dq: array<f32, MAXB>;
var<private> root0: vec3f;
var<private> root1: vec3f;
// Contacts of this step, in slots.
var<private> nc: u32;
var<private> c_on: array<bool, MAXC>;
var<private> c_node: array<u32, MAXC>;
var<private> c_dn: array<vec3f, MAXC>;
var<private> c_dt: array<vec3f, MAXC>;
var<private> c_vn: array<f32, MAXC>;
var<private> c_vt: array<f32, MAXC>;
// The node's speed along the ground at the start of the step: friction may
// only push against the mean of this and the speed after the step.
var<private> c_vs: array<f32, MAXC>;
var<private> c_goal: array<f32, MAXC>;
var<private> c_mu: array<f32, MAXC>;
var<private> reach: array<f32, MAXN>;
var<private> kmat: array<f32, MAXR * (MAXR + 1u) / 2u>;
var<private> lambda: array<f32, MAXR>;
var<private> old: array<f32, MAXR>;
var<private> vrow: array<f32, MAXR>;

fn ti(f: u32) -> u32 {
    return f * WG + lane_id;
}
fn node_pos(i: u32) -> vec2f {
    let f = TN + 4u * i;
    return vec2f(tab[ti(f)], tab[ti(f + 1u)]);
}
fn node_vel(i: u32) -> vec2f {
    let f = TN + 4u * i + 2u;
    return vec2f(tab[ti(f)], tab[ti(f + 1u)]);
}
fn set_pos(i: u32, v: vec2f) {
    let f = TN + 4u * i;
    tab[ti(f)] = v.x;
    tab[ti(f + 1u)] = v.y;
}
fn set_vel(i: u32, v: vec2f) {
    let f = TN + 4u * i + 2u;
    tab[ti(f)] = v.x;
    tab[ti(f + 1u)] = v.y;
}
fn body_get(j: u32, s: u32) -> vec3f {
    let f = TB + 6u * j + 3u * s;
    return vec3f(tab[ti(f)], tab[ti(f + 1u)], tab[ti(f + 2u)]);
}
fn body_set(j: u32, s: u32, v: vec3f) {
    let f = TB + 6u * j + 3u * s;
    tab[ti(f)] = v.x;
    tab[ti(f + 1u)] = v.y;
    tab[ti(f + 2u)] = v.z;
}
fn body_add(j: u32, s: u32, v: vec3f) {
    let f = TB + 6u * j + 3u * s;
    tab[ti(f)] += v.x;
    tab[ti(f + 1u)] += v.y;
    tab[ti(f + 2u)] += v.z;
}
fn bone_field(j: u32, f: u32) -> f32 {
    return bone_data[bone_base + (j * BONE_FIELDS + f) * TILE];
}
// How deep node `i` sits in the mud, as a share of the deepest mud.
fn mud_sink(i: u32) -> f32 {
    let pn = node_pos(i);
    let g = terrain(pn.x);
    let secant = sqrt(1.0 + g.y * g.y);
    let dry = (pn.y - g.x) / secant - node_radius(i);
    return clamp(-dry, 0.0, p.mud) * (1.0 / MUD_FULL_DEPTH);
}
fn node_radius(i: u32) -> f32 {
    if i == 0u {
        return bone_field(0u, 8u);
    }
    return bone_field(i - 1u, 5u);
}
fn node_fric(i: u32) -> f32 {
    if i == 0u {
        return bone_field(0u, 2u);
    }
    return bone_field(i - 1u, 6u);
}
fn body_of(i: u32) -> u32 {
    return select(i - 1u, 0u, i == 0u);
}
fn parent_of(j: u32) -> u32 {
    return body_of(pivot[j]);
}

fn quake_phase(seed: u32) -> f32 {
    return f32(seed & 0xffffu) * (1.0 / 65536.0);
}
fn quake_scale(seed: u32) -> f32 {
    return 0.6 + f32((seed >> 16u) & 0xffffu) * (0.8 / 65536.0);
}
// Height and slope of the ground (physics::ground); flat ground is exactly
// (0, 0) there too.
fn terrain(x: f32) -> vec2f {
    if !rough {
        return vec2f(0.0);
    }
    let t0 = x * (1.0 / 1.1) + phase_q;
    let u0 = t0 - floor(t0);
    let w0 = u0 * (1.0 - u0);
    let t1 = x * (1.0 / 0.43) + 0.3 + phase_q;
    let u1 = t1 - floor(t1);
    let w1 = u1 * (1.0 - u1);
    var height = 0.65 * 16.0 * w0 * w0 + 0.35 * 16.0 * w1 * w1;
    var slope = 0.65 * 32.0 * w0 * (1.0 - 2.0 * u0) * (1.0 / 1.1)
        + 0.35 * 32.0 * w1 * (1.0 - 2.0 * u1) * (1.0 / 0.43);
    height = amplitude * height + p.slope * x;
    slope = amplitude * slope + p.slope;
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
    if p.hurdles > 0.0 {
        let spacing = HURDLE_SPACING;
        let center = spacing * 0.5;
        let t = x / spacing;
        let r = x - floor(t) * spacing;
        let distance = abs(r - center);
        let half = 0.5 * HURDLE_TOP;
        let run = HURDLE_RUN;
        let ramp = clamp((half + run - distance) / run, 0.0, 1.0);
        var factor = ramp;
        if distance <= half {
            factor = 1.0;
        } else if distance >= half + run {
            factor = 0.0;
        }
        let on_ramp = distance > half && distance < half + run;
        var side = -1.0;
        if r < center {
            side = 1.0;
        }
        height += p.hurdles * factor;
        if on_ramp {
            slope += p.hurdles * (side / run);
        }
    }
    return vec2f(height, slope);
}

// Planar spatial vectors (angular, x, y) as vec3f.
fn crm(v: vec3f, u: vec3f) -> vec3f {
    return vec3f(0.0, -v.x * u.z + u.x * v.z, v.x * u.y - u.x * v.y);
}
fn crf(v: vec3f, f: vec3f) -> vec3f {
    return vec3f(v.y * f.z - v.z * f.y, -v.x * f.z, v.x * f.y);
}
fn force_at(r: vec2f, f: vec2f) -> vec3f {
    return vec3f(r.x * f.y - r.y * f.x, f.x, f.y);
}
fn sym_mul(s0: vec3f, s1: vec3f, v: vec3f) -> vec3f {
    return vec3f(
        s0.x * v.x + s0.y * v.y + s0.z * v.z,
        s0.y * v.x + s1.x * v.y + s1.y * v.z,
        s0.z * v.x + s1.y * v.y + s1.z * v.z,
    );
}
fn sdot(a: vec3f, b: vec3f) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
fn axis_of(j: u32) -> vec3f {
    return vec3f(1.0, arm[j].y, -arm[j].x);
}
fn wrap(a: f32) -> f32 {
    return a - TAU * floor((a + PI) / TAU);
}
// The waveform's shape at time t: the target length is
// long - amplitude * (1 - w) (physics2::simulate_step_inner).
fn wave(t: f32, inv_period: f32, phase: f32, offset: f32, duty: f32, inv_duty: f32, inv_complement: f32) -> f32 {
    let x = t * inv_period + phase + offset;
    let ph = x - floor(x);
    if ph < duty {
        return 0.5 + 0.5 * cos(PI * (ph * inv_duty));
    }
    return 0.5 - 0.5 * cos(PI * ((ph - duty) * inv_complement));
}
// Body j's velocity-product acceleration, from its pivot's velocity in the
// table (the spatial velocity's linear part is the velocity of the body
// point at the origin).
fn cvel_of(j: u32) -> vec3f {
    let vp = node_vel(pivot[j]);
    let w = om[j];
    let sv = vec3f(w, vp.x + w * arm[j].y, vp.y - w * arm[j].x);
    return crm(sv, axis_of(j)) * qd[j];
}

// Absolute angles and rates, node velocities and, with `positions`, node
// positions, from the state. A bone's parent's angle and rate come from the
// table.
fn kinematics(positions: bool) {
    if positions {
        set_pos(0u, x0);
    }
    set_vel(0u, v0);
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        var t = q[0];
        var w = qd[0];
        if j > 0u {
            let up = body_get(parent_of(j), 0u);
            t = up.x + q[j];
            w = up.y + qd[j];
        }
        om[j] = w;
        body_set(j, 0u, vec3f(t, w, 0.0));
        let sn = sin(t);
        let cs = cos(t);
        let l = len[j];
        let pv0 = node_vel(pivot[j]);
        set_vel(j + 1u, vec2f(pv0.x - l * w * sn, pv0.y + l * w * cs));
        if positions {
            let pp0 = node_pos(pivot[j]);
            set_pos(j + 1u, vec2f(pp0.x + l * cs, pp0.y + l * sn));
        }
    }
}

fn momentum() -> vec2f {
    var m = vec2f(0.0);
    for (var i = 0u; i < MAXN; i++) {
        if i >= nn { break; }
        m += node_vel(i) * mass[i];
    }
    return m;
}

// Semi-implicit Euler on the joint coordinates with the accelerations in
// `acc` and `qdd`, and air drag. Returns the head's acceleration.
fn integrate_state(air: f32) -> vec2f {
    let a0 = acc[0];
    let head = vec2f(a0.y - qd[0] * v0.y, a0.z + qd[0] * v0.x);
    v0 = vec2f((v0.x + head.x * DT) * air, (v0.y + head.y * DT) * air);
    qd[0] = (qd[0] + a0.x * DT) * air;
    x0 = vec2f(x0.x + v0.x * DT, x0.y + v0.y * DT);
    q[0] += qd[0] * DT;
    for (var j = 1u; j < MAXB; j++) {
        if j >= nb { break; }
        qd[j] = (qd[j] + qdd[j] * DT) * air;
        q[j] += qd[j] * DT;
    }
    return head;
}

// The accelerations that the spatial forces in the table's first scratch
// vector of every bone cause, through the articulated inertias of this
// step: children first, then parents first. Leaves them in the second
// scratch vector and the joint accelerations in `dq`.
fn response() {
    for (var i = 1u; i < MAXB; i++) {
        let j = MAXB - i;
        if j >= nb { continue; }
        let pj = body_get(j, 0u);
        let t = -sdot(axis_of(j), pj);
        dq[j] = t;
        body_add(parent_of(j), 0u, pj + uvec[j] * (t * dinv[j]));
    }
    body_set(0u, 1u, -sym_mul(root0, root1, body_get(0u, 0u)));
    dq[0] = 0.0;
    for (var j = 1u; j < MAXB; j++) {
        if j >= nb { break; }
        let a = body_get(parent_of(j), 1u);
        let t = (dq[j] - sdot(uvec[j], a)) * dinv[j];
        dq[j] = t;
        body_set(j, 1u, a + axis_of(j) * t);
    }
}

fn clear_forces() {
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        body_set(j, 0u, vec3f(0.0));
    }
}

// The symmetric contact-space matrix, lower triangle.
fn tri(r: u32, c: u32) -> u32 {
    let hi_ = max(r, c);
    return hi_ * (hi_ + 1u) / 2u + min(r, c);
}

// Adds the accelerations that contact forces `f` (normal, friction per
// contact slot) cause.
fn apply_contacts(f: ptr<private, array<f32, MAXR>>) {
    clear_forces();
    for (var ci = 0u; ci < MAXC; ci++) {
        if !c_on[ci] { continue; }
        body_add(body_of(c_node[ci]), 0u, -(c_dn[ci] * (*f)[2u * ci] + c_dt[ci] * (*f)[2u * ci + 1u]));
    }
    response();
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        acc[j] += body_get(j, 1u);
        qdd[j] += dq[j];
    }
}

// Projected Gauss-Seidel on the contact impulses: a touching node may
// approach the ground only as fast as its goal allows, normal forces only
// push, and friction stays within mu times the normal force and opposes the
// slip. Every row's velocity (`vrow`) is kept current as the forces change.
fn pgs(sweeps: u32) {
    for (var row = 0u; row < MAXR; row++) {
        if !c_on[row / 2u] { continue; }
        var v = c_vn[row / 2u];
        if (row & 1u) == 1u {
            v = c_vt[row / 2u];
        }
        for (var j = 0u; j < MAXR; j++) {
            if !c_on[j / 2u] { continue; }
            v += kmat[tri(row, j)] * lambda[j];
        }
        vrow[row] = v;
    }
    for (var sweep = 0u; sweep < sweeps; sweep++) {
        for (var ci = 0u; ci < MAXC; ci++) {
            if !c_on[ci] { continue; }
            let rn = 2u * ci;
            let rt = rn + 1u;
            let normal = max(lambda[rn] + (c_goal[ci] - vrow[rn]) * (1.0 / kmat[tri(rn, rn)]), 0.0);
            let dn = normal - lambda[rn];
            lambda[rn] = normal;
            for (var j = 0u; j < MAXR; j++) {
                if !c_on[j / 2u] { continue; }
                vrow[j] += kmat[tri(j, rn)] * dn;
            }
            let bound = c_mu[ci] * lambda[rn];
            // Friction may not do positive work: it only opposes
            // a = start speed + end speed without its own force, and only up
            // to |a| / k (see physics2.rs).
            let stiff = kmat[tri(rt, rt)];
            let a = c_vs[ci] + vrow[rt] - stiff * lambda[rt];
            let reach = abs(a) / stiff;
            let cap = min(bound, reach);
            let friction = clamp(lambda[rt] - vrow[rt] * (1.0 / stiff), select(0.0, -cap, a > 0.0), select(cap, 0.0, a > 0.0));
            let dt_ = friction - lambda[rt];
            lambda[rt] = friction;
            for (var j = 0u; j < MAXR; j++) {
                if !c_on[j / 2u] { continue; }
                vrow[j] += kmat[tri(j, rt)] * dt_;
            }
        }
    }
}

// Removes friction that would still do positive work after the solves: each
// contact's friction only opposes the mean of its node's speed before and
// after the step, up to the size that stops the node. Two sweeps.
fn clean_friction() {
    for (var sweep = 0u; sweep < 2u; sweep++) {
        for (var ci = 0u; ci < MAXC; ci++) {
            if !c_on[ci] { continue; }
            let rn = 2u * ci;
            let rt = rn + 1u;
            let stiff = kmat[tri(rt, rt)];
            let a = c_vs[ci] + vrow[rt] - stiff * lambda[rt];
            let cap = min(c_mu[ci] * lambda[rn], abs(a) / stiff);
            let friction = clamp(lambda[rt], select(0.0, -cap, a > 0.0), select(cap, 0.0, a > 0.0));
            let dt_ = friction - lambda[rt];
            lambda[rt] = friction;
            for (var j = 0u; j < MAXR; j++) {
                if !c_on[j / 2u] { continue; }
                vrow[j] += kmat[tri(j, rt)] * dt_;
            }
        }
    }
}

// Ground contacts at velocity level, solved together (physics2's contact
// section): detection, the deepest MAXC nodes into slots, the contact-space
// matrix, Gauss-Seidel, the forces' response, and one pass that plants the
// contacts against the step's end pose. Returns `outside` (the step's
// impulse from gravity and wind) plus the ground's impulse.
fn contacts(origin: vec2f, before: vec2f, outside: vec2f) -> vec2f {
    nc = 0u;
    var candidates = 0u;
    for (var i = 0u; i < MAXN; i++) {
        reach[i] = 1e30;
        if i >= nn { continue; }
        let pn = node_pos(i);
        let g = terrain(pn.x);
        let secant = sqrt(1.0 + g.y * g.y);
        let normal = vec2f(-g.y / secant, 1.0 / secant);
        let mud = select(0.0, p.mud, p.ground > 0.0);
        let gap = (pn.y - g.x) / secant - node_radius(i) + mud;
        let body = body_of(i);
        let r = pn - origin;
        let v = node_vel(i);
        let w = om[body];
        let dn = force_at(r, normal);
        let vn_free = v.x * normal.x + v.y * normal.y
            + DT * (sdot(dn, acc[body]) + w * (-v.y * normal.x + v.x * normal.y));
        let depth = gap + DT * vn_free;
        if depth <= 0.0 {
            reach[i] = depth;
            candidates += 1u;
        }
    }
    for (var ci = 0u; ci < MAXC; ci++) {
        c_on[ci] = false;
    }
    if candidates == 0u {
        return outside;
    }
    // The deepest MAXC nodes take part, in node order.
    var chosen: array<bool, MAXN>;
    for (var i = 0u; i < MAXN; i++) {
        chosen[i] = reach[i] < 1e29;
        if !chosen[i] || candidates <= MAXC { continue; }
        var deeper = 0u;
        for (var j = 0u; j < MAXN; j++) {
            if reach[j] < reach[i] || (reach[j] == reach[i] && j < i) {
                deeper += 1u;
            }
        }
        chosen[i] = deeper < MAXC;
    }
    for (var ci = 0u; ci < MAXC; ci++) {
        if ci >= candidates { break; }
        var node = 0u;
        var seen = 0u;
        for (var i = 0u; i < MAXN; i++) {
            if chosen[i] {
                if seen == ci {
                    node = i;
                }
                seen += 1u;
            }
        }
        let pn = node_pos(node);
        let g = terrain(pn.x);
        let secant = sqrt(1.0 + g.y * g.y);
        let normal = vec2f(-g.y / secant, 1.0 / secant);
        let tangent = vec2f(normal.y, -normal.x);
        let mud = select(0.0, p.mud, p.ground > 0.0);
        let dry = (pn.y - g.x) / secant - node_radius(node);
        let gap = dry + mud;
        let sink = clamp(-dry, 0.0, mud) * (1.0 / MUD_FULL_DEPTH);
        let body = body_of(node);
        let r = pn - origin;
        let v = node_vel(node);
        // The body's acceleration and turning rate, from the table.
        let a = body_get(body, 0u);
        let w = body_get(body, 1u).x;
        let dn = force_at(r, normal);
        let dtan = force_at(r, tangent);
        c_on[ci] = true;
        c_node[ci] = node;
        c_dn[ci] = dn;
        c_dt[ci] = dtan;
        c_vn[ci] = v.x * normal.x + v.y * normal.y
            + DT * (sdot(dn, a) + w * (-v.y * normal.x + v.x * normal.y));
        c_vt[ci] = v.x * tangent.x + v.y * tangent.y
            + DT * (sdot(dtan, a) + w * (-v.y * tangent.x + v.x * tangent.y));
        c_vs[ci] = v.x * tangent.x + v.y * tangent.y;
        c_goal[ci] = select(-gap * PUSH_OUT * RATE, -gap * RATE, gap >= 0.0);
        c_mu[ci] = node_fric(node) * p.friction * (1.0 + MUD_GRIP * sink) * (1.0 + MUD_NORMAL * sink);
        nc += 1u;
    }
    // Contact-space matrix, column by column from each unit force's
    // response; symmetric, the lower triangle kept.
    for (var col = 0u; col < MAXR; col++) {
        let ci = col / 2u;
        if !c_on[ci] { continue; }
        var fdir = c_dn[ci];
        if (col & 1u) == 1u {
            fdir = c_dt[ci];
        }
        clear_forces();
        body_add(body_of(c_node[ci]), 0u, -fdir);
        response();
        for (var row = 0u; row < MAXR; row++) {
            let ri = row / 2u;
            if row < col || !c_on[ri] { continue; }
            var rdir = c_dn[ri];
            if (row & 1u) == 1u {
                rdir = c_dt[ri];
            }
            kmat[tri(row, col)] = DT * sdot(rdir, body_get(body_of(c_node[ri]), 1u));
        }
    }
    for (var ci = 0u; ci < MAXC; ci++) {
        var ln = 0.0;
        var lt = 0.0;
        if WARM && c_on[ci] {
            let node = c_node[ci];
            let rec = records[record_base + node];
            let w = select(rec.b, rec.c, node == 0u);
            ln = w.x;
            lt = clamp(w.y, -c_mu[ci] * ln, c_mu[ci] * ln);
        }
        lambda[2u * ci] = ln;
        lambda[2u * ci + 1u] = lt;
    }
    pgs(PGS_SWEEPS);
    apply_contacts(&lambda);
    // Plant against the end pose: take the step, measure each contact's
    // velocity in the end pose (after the momentum balance), and solve again
    // with the difference.
    for (var round = 0u; round < PLANT_ROUNDS; round++) {
        let sx0 = x0;
        let sv0 = v0;
        var sq: array<f32, MAXB>;
        var sqd: array<f32, MAXB>;
        for (var j = 0u; j < MAXB; j++) {
            sq[j] = q[j];
            sqd[j] = qd[j];
        }
        integrate_state(1.0);
        kinematics(false);
        var ground = vec2f(0.0);
        for (var ci = 0u; ci < MAXC; ci++) {
            if !c_on[ci] { continue; }
            ground += vec2f(
                (lambda[2u * ci] * c_dn[ci].y + lambda[2u * ci + 1u] * c_dt[ci].y) * DT,
                (lambda[2u * ci] * c_dn[ci].z + lambda[2u * ci + 1u] * c_dt[ci].z) * DT,
            );
        }
        let after = momentum();
        let shift = vec2f(
            (before.x + outside.x + ground.x - after.x) * inv_mass,
            (before.y + outside.y + ground.y - after.y) * inv_mass,
        );
        for (var ci = 0u; ci < MAXC; ci++) {
            if !c_on[ci] { continue; }
            let rn = 2u * ci;
            let rt = rn + 1u;
            let v = node_vel(c_node[ci]) + shift;
            let normal = c_dn[ci].yz;
            c_vn[ci] += v.x * normal.x + v.y * normal.y - vrow[rn];
            c_vt[ci] += v.x * normal.y + v.y * -normal.x - vrow[rt];
        }
        x0 = sx0;
        v0 = sv0;
        for (var j = 0u; j < MAXB; j++) {
            q[j] = sq[j];
            qd[j] = sqd[j];
        }
        for (var j = 0u; j < MAXR; j++) {
            old[j] = lambda[j];
        }
        pgs(PLANT_SWEEPS);
        clean_friction();
        for (var j = 0u; j < MAXR; j++) {
            old[j] = lambda[j] - old[j];
        }
        apply_contacts(&old);
    }
    var impulse = outside;
    for (var ci = 0u; ci < MAXC; ci++) {
        if !c_on[ci] { continue; }
        impulse += vec2f(
            (lambda[2u * ci] * c_dn[ci].y + lambda[2u * ci + 1u] * c_dt[ci].y) * DT,
            (lambda[2u * ci] * c_dn[ci].z + lambda[2u * ci + 1u] * c_dt[ci].z) * DT,
        );
    }
    return impulse;
}

@compute @workgroup_size(WG)
fn advance(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3u) {
    let creature = group.x * WG + lane;
    if creature >= p.count {
        return;
    }
    lane_id = lane;
    let info = creature_info[creature];
    nn = info.x;
    nb = info.y;
    let muscle_count = info.z;
    let still = p.quake <= 0.0 || p.ground <= 0.0;
    phase_q = select(quake_phase(info.w), 0.0, still);
    amplitude = p.terrain + select(p.quake * quake_scale(info.w), 0.0, still);
    rough = amplitude != 0.0 || p.slope != 0.0 || p.gaps > 0.0 || p.hurdles > 0.0;
    let tile = tile_info[creature / TILE];
    let tl = creature % TILE;
    bone_base = tile.y + tl;
    record_base = creature * STRIDE;

    let head = records[record_base];
    x0 = head.a;
    v0 = head.b;
    mass[0] = bone_field(0u, 7u);
    total_mass = mass[0];
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        pivot[j] = bitcast<u32>(bone_field(j, 0u));
        len[j] = bone_field(j, 1u);
        mass[j + 1u] = bone_field(j, 4u);
        total_mass += mass[j + 1u];
        let r = records[record_base + j + 1u];
        q[j] = r.a.x;
        qd[j] = r.a.y;
    }
    inv_mass = 1.0 / total_mass;
    let inv_nodes = 1.0 / f32(nn);
    let muscle_scale = bone_field(0u, 3u);
    kinematics(true);

    var metrics = Result(0.0, 0.0, 1e20, -1e20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if p.tick > SETTLE {
        metrics = results[creature];
    }
    var head_shake = metrics.head_shake;
    let grounded = p.ground > 0.0;

    for (var s = 0u; s < p.steps; s++) {
        if metrics.fall_time > 0.0 || metrics.screened > 0.0 {
            break;
        }
        let tick = p.tick + s;
        if tick < SETTLE {
            continue;
        }
        let step = tick - SETTLE;
        let time = f32(step) * DT;
        let head_before = v0;
        let origin = x0;
        let before = momentum();
        // For the first-law check in flight.
        var energy_start = 0.0;
        var energy_scale = 0.0;
        var mass_x_start = 0.0;
        for (var i = 0u; i < MAXN; i++) {
            if i >= nn { break; }
            let pi = node_pos(i);
            let vi = node_vel(i);
            let kinetic = 0.5 * mass[i] * (vi.x * vi.x + vi.y * vi.y);
            let potential = mass[i] * p.gravity * pi.y;
            energy_start += kinetic + potential;
            energy_scale += kinetic + abs(potential);
            mass_x_start += pi.x * mass[i];
        }
        var muscle_start = 0.0;
        // Body inertias (true, for the velocity products), pivot arms, and
        // the velocity-product forces; the forces below go in with the
        // opposite sign.
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            let m = mass[j + 1u];
            let r = node_pos(j + 1u) - origin;
            i0[j] = vec3f(m * (r.x * r.x + r.y * r.y), -m * r.y, m * r.x);
            i1[j] = vec3f(m, 0.0, m);
            arm[j] = node_pos(pivot[j]) - origin;
        }
        i1[0] += vec3f(mass[0], 0.0, mass[0]);
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            let vp = node_vel(pivot[j]);
            let w = om[j];
            let sv = vec3f(w, vp.x + w * arm[j].y, vp.y - w * arm[j].x);
            bias[j] = crf(sv, sym_mul(i0[j], i1[j], sv));
        }
        // Gravity and wind on every node.
        var mud_impulse = 0.0;
        for (var i = 0u; i < MAXN; i++) {
            if i >= nn { break; }
            let m = mass[i];
            var fx = p.wind * m;
            if p.mud > 0.0 && p.ground > 0.0 {
                let drag = -m * MUD_DRAG * mud_sink(i) * node_vel(i).x;
                fx += drag;
                mud_impulse += drag * DT;
            }
            bias[body_of(i)] -= force_at(node_pos(i) - origin, vec2f(fx, -p.gravity * m));
        }
        // Air drag on every bone, at its midpoint, limited so a step of drag
        // never more than halves the speed it acts on.
        var air_impulse = vec2f(0.0);
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            let pv = pivot[j];
            let mid = 0.5 * (node_pos(pv) + node_pos(j + 1u));
            let v = 0.5 * (node_vel(pv) + node_vel(j + 1u));
            let speed = sqrt(v.x * v.x + v.y * v.y);
            let width = node_radius(pv) + node_radius(j + 1u);
            let strength = max(min(AIR_DRAG * len[j] * width * speed, 0.5 * mass[j + 1u] * RATE), 0.0);
            let f = -v * strength;
            bias[j] -= force_at(mid - origin, f);
            air_impulse += f * DT;
        }
        // Muscles pull between points on two bones; the forces collect in
        // the table.
        clear_forces();
        for (var k = 0u; k < muscle_count; k++) {
            let field = tile.x + k * MUSCLE_FIELDS * TILE + tl;
            let packed = bitcast<u32>(muscle_data[field]);
            let a0 = packed & 63u;
            let a1 = (packed >> 6u) & 63u;
            let b0 = (packed >> 12u) & 63u;
            let b1 = (packed >> 18u) & 63u;
            let anchor_a = muscle_data[field + TILE];
            let anchor_b = muscle_data[field + 2u * TILE];
            let amp = muscle_data[field + 3u * TILE];
            let hill = muscle_data[field + 4u * TILE];
            let inv_period = muscle_data[field + 5u * TILE];
            let phase = muscle_data[field + 6u * TILE];
            let duty = muscle_data[field + 7u * TILE];
            let stiffness = muscle_data[field + 8u * TILE];
            let inv_duty = muscle_data[field + 9u * TILE];
            let inv_complement = muscle_data[field + 10u * TILE];
            let offset = muscle_data[field + 13u * TILE];
            let energy = muscle_data[field + 14u * TILE];
            let strength = muscle_data[field + 15u * TILE] * muscle_scale;
            let cap = MAX_MUSCLE_FORCE * strength;
            let inv_capacity = 1.0 / (MUSCLE_CAPACITY * p.muscle_energy * strength);
            let pa0 = node_pos(a0);
            let pa1 = node_pos(a1);
            let pb0 = node_pos(b0);
            let pb1 = node_pos(b1);
            let va0 = node_vel(a0);
            let va1 = node_vel(a1);
            let vb0 = node_vel(b0);
            let vb1 = node_vel(b1);
            let pa = vec2f(pa0.x + (pa1.x - pa0.x) * anchor_a, pa0.y + (pa1.y - pa0.y) * anchor_a);
            let va = vec2f(va0.x + (va1.x - va0.x) * anchor_a, va0.y + (va1.y - va0.y) * anchor_a);
            let pb = vec2f(pb0.x + (pb1.x - pb0.x) * anchor_b, pb0.y + (pb1.y - pb0.y) * anchor_b);
            let vb = vec2f(vb0.x + (vb1.x - vb0.x) * anchor_b, vb0.y + (vb1.y - vb0.y) * anchor_b);
            let d = pb - pa;
            let length_m = max(sqrt(d.x * d.x + d.y * d.y), 1e-6);
            let inverse = 1.0 / length_m;
            let dir = vec2f(d.x * inverse, d.y * inverse);
            let relative = (vb.x - va.x) * dir.x + (vb.y - va.y) * dir.y;
            var target_speed = 0.0;
            if time > 0.0 {
                target_speed = amp * (wave(time, inv_period, phase, offset, duty, inv_duty, inv_complement)
                    - wave(max(time - DT, 0.0), inv_period, phase, offset, duty, inv_duty, inv_complement)) * RATE;
            }
            var drive = max(-target_speed * stiffness * 0.25, 0.0) * energy;
            if hill > 0.0 {
                drive *= clamp(1.0 + relative * hill, 0.0, 1.0);
            }
            let magnitude = clamp(drive + relative * 0.15, -cap, cap);
            let work = min(drive, cap) * max(-relative, 0.0) * DT;
            muscle_data[field + 14u * TILE] = clamp(
                energy - work * inv_capacity
                    + MUSCLE_RECOVERY * p.muscle_recovery * DT * (1.0 - energy),
                0.0,
                1.0,
            );
            muscle_data[field + 11u * TILE] = magnitude;
            muscle_start += magnitude * length_m;
            let stretch_start = max(length_m - muscle_data[field + 17u * TILE], 0.0);
            let stored_start = 0.5 * muscle_data[field + 16u * TILE] * stretch_start * stretch_start;
            energy_start += stored_start;
            energy_scale += stored_start;
            let pull = magnitude + muscle_data[field + 16u * TILE] * max(length_m - muscle_data[field + 17u * TILE], 0.0);
            let f = dir * pull;
            body_add(a1 - 1u, 0u, force_at(pa - origin, f));
            body_add(b1 - 1u, 0u, -force_at(pb - origin, f));
        }
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            bias[j] -= body_get(j, 0u);
        }
        // Spin cap: rotational drag past the cap, implicit, toward rest.
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            let w = om[j];
            if abs(w) > SPIN_CAP {
                let l = len[j];
                let drag = SPIN_HARDNESS * mass[j + 1u] * l * l * (abs(w) * INV_SPIN_CAP - 1.0);
                i0[j].x += drag;
                bias[j].x += drag * RATE * w;
            }
        }
        // Articulated-body pass, children first, with joint damping and the
        // joint limits' inelastic stops implicit in each joint's inertia. A
        // bone hands its articulated inertia to its parent through selects
        // over the bones before it.
        for (var i = 1u; i < MAXB; i++) {
            let j = MAXB - i;
            if j >= nb { continue; }
            let axis = axis_of(j);
            let uv = sym_mul(i0[j], i1[j], axis);
            var d = sdot(axis, uv);
            var tau = 0.0;
            let c = d * INV_JOINT_DAMPING;
            tau -= c * qd[j];
            d += c * DT;
            let qj = q[j];
            let qdj = qd[j];
            let lo = bone_field(j, 2u);
            let hi = bone_field(j, 3u);
            let predicted = qj + DT * qdj;
            let upper = predicted > hi;
            if upper || predicted < lo {
                let room = select(lo - qj, hi - qj, upper);
                let past = (room < 0.0) == upper;
                let goal = select(room, room * PUSH_OUT, past) * RATE;
                if (qdj > goal) == upper {
                    let cl = LIMIT_HARDNESS * d * RATE;
                    tau -= cl * (qdj - goal);
                    d += cl * DT;
                }
            }
            let u = tau - sdot(axis, bias[j]);
            let di = 1.0 / d;
            uvec[j] = uv;
            dinv[j] = di;
            uu[j] = u;
            let k = -di;
            let a0 = i0[j] + vec3f(k * uv.x * uv.x, k * uv.x * uv.y, k * uv.x * uv.z);
            let a1 = i1[j] + vec3f(k * uv.y * uv.y, k * uv.y * uv.z, k * uv.z * uv.z);
            let pa = bias[j] + sym_mul(a0, a1, cvel_of(j)) + uv * (u * di);
            let pr = parent_of(j);
            for (var up = 0u; up < MAXB; up++) {
                if up >= j { break; }
                if pr == up {
                    i0[up] += a0;
                    i1[up] += a1;
                    bias[up] += pa;
                }
            }
        }
        // The neck body floats freely: the root's inverse inertia.
        {
            let r0 = i0[0];
            let r1 = i1[0];
            let c00 = r1.x * r1.z - r1.y * r1.y;
            let c01 = r0.z * r1.y - r0.y * r1.z;
            let c02 = r0.y * r1.y - r0.z * r1.x;
            let inv_det = 1.0 / (r0.x * c00 + r0.y * c01 + r0.z * c02);
            let c11 = r0.x * r1.z - r0.z * r0.z;
            let c12 = r0.y * r0.z - r0.x * r1.y;
            let c22 = r0.x * r1.x - r0.y * r0.y;
            root0 = vec3f(c00 * inv_det, c01 * inv_det, c02 * inv_det);
            root1 = vec3f(c11 * inv_det, c12 * inv_det, c22 * inv_det);
        }
        acc[0] = -sym_mul(root0, root1, bias[0]);
        qdd[0] = 0.0;
        // Parents first; each bone leaves its acceleration and turning rate
        // in the table for its children and the contacts.
        body_set(0u, 0u, acc[0]);
        body_set(0u, 1u, vec3f(om[0], 0.0, 0.0));
        for (var j = 1u; j < MAXB; j++) {
            if j >= nb { break; }
            let a = body_get(parent_of(j), 0u) + cvel_of(j);
            qdd[j] = (uu[j] - sdot(uvec[j], a)) * dinv[j];
            acc[j] = a + axis_of(j) * qdd[j];
            body_set(j, 0u, acc[j]);
            body_set(j, 1u, vec3f(om[j], 0.0, 0.0));
        }

        var impulse = vec2f(p.wind * total_mass * DT + mud_impulse, -p.gravity * total_mass * DT) + air_impulse;
        nc = 0u;
        if grounded {
            impulse = contacts(origin, before, impulse);
        }
        // Each node's contact force starts the next step's solve.
        if WARM && grounded {
            for (var i = 0u; i < MAXN; i++) {
                if i >= nn { break; }
                var w = vec2f(0.0);
                for (var ci = 0u; ci < MAXC; ci++) {
                    if c_on[ci] && c_node[ci] == i {
                        w = vec2f(lambda[2u * ci], lambda[2u * ci + 1u]);
                    }
                }
                if i == 0u {
                    records[record_base].c = w;
                } else {
                    records[record_base + i].b = w;
                }
            }
        }
        integrate_state(p.air);
        kinematics(true);
        // Momentum balance.
        let after = momentum();
        let expected = vec2f((before.x + impulse.x) * p.air, (before.y + impulse.y) * p.air);
        let shift = vec2f((expected.x - after.x) * inv_mass, (expected.y - after.y) * inv_mass);
        v0 += shift;
        for (var i = 0u; i < MAXN; i++) {
            if i >= nn { break; }
            set_vel(i, node_vel(i) + shift);
        }
        // First law in flight.
        if nc == 0u {
            var muscle_end = 0.0;
            var stored_end = 0.0;
            for (var k = 0u; k < muscle_count; k++) {
                let field = tile.x + k * MUSCLE_FIELDS * TILE + tl;
                let packed = bitcast<u32>(muscle_data[field]);
                let anchor_a = muscle_data[field + TILE];
                let anchor_b = muscle_data[field + 2u * TILE];
                let pa0 = node_pos(packed & 63u);
                let pa1 = node_pos((packed >> 6u) & 63u);
                let pb0 = node_pos((packed >> 12u) & 63u);
                let pb1 = node_pos((packed >> 18u) & 63u);
                let pa = vec2f(pa0.x + (pa1.x - pa0.x) * anchor_a, pa0.y + (pa1.y - pa0.y) * anchor_a);
                let pb = vec2f(pb0.x + (pb1.x - pb0.x) * anchor_b, pb0.y + (pb1.y - pb0.y) * anchor_b);
                let d = pb - pa;
                let length_m = sqrt(d.x * d.x + d.y * d.y);
                muscle_end += muscle_data[field + 11u * TILE] * length_m;
                let stretch_end = max(length_m - muscle_data[field + 17u * TILE], 0.0);
                stored_end += 0.5 * muscle_data[field + 16u * TILE] * stretch_end * stretch_end;
            }
            var energy_end = stored_end;
            var mass_x_end = 0.0;
            for (var i = 0u; i < MAXN; i++) {
                if i >= nn { break; }
                let pi = node_pos(i);
                let vi = node_vel(i);
                energy_end += 0.5 * mass[i] * (vi.x * vi.x + vi.y * vi.y) + mass[i] * p.gravity * pi.y;
                mass_x_end += pi.x * mass[i];
            }
            let work = (muscle_start - muscle_end) + p.wind * (mass_x_end - mass_x_start);
            let excess = energy_end - energy_start - work - (1e-4 + 1e-5 * energy_scale);
            if excess > 0.0 {
                let center = vec2f(expected.x * inv_mass, expected.y * inv_mass);
                var internal = 0.0;
                for (var i = 0u; i < MAXN; i++) {
                    if i >= nn { break; }
                    let vi = node_vel(i);
                    let x = vi.x - center.x;
                    let y = vi.y - center.y;
                    internal += 0.5 * mass[i] * (x * x + y * y);
                }
                var keep = 0.0;
                if internal > 0.0 {
                    keep = sqrt(max(1.0 - excess / internal, 0.0));
                }
                qd[0] *= keep;
                for (var j = 1u; j < MAXB; j++) {
                    if j >= nb { break; }
                    qd[j] *= keep;
                }
                v0 = vec2f(center.x + keep * (v0.x - center.x), center.y + keep * (v0.y - center.y));
                for (var i = 0u; i < MAXN; i++) {
                    if i >= nn { break; }
                    let vi = node_vel(i);
                    set_vel(i, vec2f(center.x + keep * (vi.x - center.x), center.y + keep * (vi.y - center.y)));
                }
                // The absolute rates scale with the joint rates.
                for (var j = 0u; j < MAXB; j++) {
                    if j >= nb { break; }
                    om[j] *= keep;
                }
            }
        }
        q[0] = wrap(q[0]);

        // Metrics, falls and the screen (physics2::run).
        var failed = false;
        var center_y = 0.0;
        var touching = 0.0;
        var low = 1e20;
        var high = -1e20;
        var contact_lo = bitcast<u32>(metrics.contact_lo);
        var contact_hi = bitcast<u32>(metrics.contact_hi);
        var lift_lo = bitcast<u32>(metrics.lift_lo);
        var lift_hi = bitcast<u32>(metrics.lift_hi);
        var now_lo = 0u;
        var now_hi = 0u;
        var com_x = 0.0;
        for (var i = 0u; i < MAXN; i++) {
            if i >= nn { break; }
            let pi = node_pos(i);
            let x = pi.x;
            let y = pi.y;
            if !(abs(x) <= 1e6) || !(abs(y) <= 1e6) {
                failed = true;
            }
            let r = node_radius(i);
            center_y += y;
            low = min(low, y - r);
            high = max(high, y + r);
            com_x += x * mass[i];
            if grounded {
                let g = terrain(x);
                let floor_y = g.x + r * sqrt(1.0 + g.y * g.y);
                if y <= floor_y + CONTACT_SLACK {
                    touching += 1.0;
                    if i < 32u {
                        contact_lo |= 1u << (i & 31u);
                        now_lo |= 1u << (i & 31u);
                    } else {
                        contact_hi |= 1u << ((i - 32u) & 31u);
                        now_hi |= 1u << ((i - 32u) & 31u);
                    }
                } else if y > floor_y + LIFT_CLEARANCE {
                    if i < 32u {
                        lift_lo |= contact_lo & (1u << (i & 31u));
                    } else {
                        lift_hi |= contact_hi & (1u << ((i - 32u) & 31u));
                    }
                }
            }
        }
        com_x *= inv_mass;
        metrics.contact_lo = bitcast<f32>(contact_lo);
        metrics.contact_hi = bitcast<f32>(contact_hi);
        metrics.lift_lo = bitcast<f32>(lift_lo);
        metrics.lift_hi = bitcast<f32>(lift_hi);
        center_y *= inv_nodes;
        let down_lo = now_lo & ~bitcast<u32>(metrics.ground_lo);
        let down_hi = now_hi & ~bitcast<u32>(metrics.ground_hi);
        metrics.ground_lo = bitcast<f32>(now_lo);
        metrics.ground_hi = bitcast<f32>(now_hi);
        if (down_lo | down_hi) != 0u && step > 0u {
            let next = time + DT;
            for (var k = 0u; k < muscle_count; k++) {
                let field = tile.x + k * MUSCLE_FIELDS * TILE + tl;
                let packed = bitcast<u32>(muscle_data[field]);
                let end = (packed >> 24u) & 7u;
                if end == NO_SENSOR {
                    continue;
                }
                let sensor = (packed >> (6u * end)) & 63u;
                let touched = select((down_hi >> (sensor - 32u)) & 1u, (down_lo >> sensor) & 1u, sensor < 32u);
                if touched == 1u {
                    let clock = next * muscle_data[field + 5u * TILE] + muscle_data[field + 6u * TILE];
                    let x = muscle_data[field + 12u * TILE] - clock;
                    muscle_data[field + 13u * TILE] = x - floor(x);
                }
            }
        }
        if time >= HEAD_SHAKE_WINDOW {
            let dv = v0 - head_before;
            let accel = sqrt(dv.x * dv.x + dv.y * dv.y) * RATE;
            head_shake += (accel - head_shake) * min(1.0 / (HEAD_SHAKE_WINDOW * RATE), 1.0);
        }
        metrics.head_shake = head_shake;
        var broken = false;
        for (var j = 1u; j < MAXB; j++) {
            if j >= nb { break; }
            if q[j] < bone_field(j, 2u) - JOINT_BREAK || q[j] > bone_field(j, 3u) + JOINT_BREAK {
                broken = true;
            }
        }
        let fell = x0.y < node_pos(1u).y || broken || head_shake > HEAD_SHAKE_LIMIT || failed;
        var ended = false;
        if fell {
            metrics.fall_time = time + DT;
            metrics.fitness = select(com_x, -1e20, failed);
            if tick <= p.screen_tick {
                metrics.screen_x = metrics.fitness;
            }
            ended = true;
        }
        metrics.ground_contact += touching;
        metrics.height_sum += high - low;
        metrics.vertical_oscillation = min(metrics.vertical_oscillation, center_y);
        metrics.gait_frequency = max(metrics.gait_frequency, center_y);
        if step == 0u {
            metrics.previous_center_y = center_y;
            metrics.vertical_extremum = center_y;
            metrics.vertical_trend = 0.0;
            metrics.gait_turns = 0.0;
        } else if step % SAMPLE == 0u {
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
        if tick == p.screen_tick && !ended {
            metrics.screen_x = com_x;
            if com_x < p.screen_bar {
                metrics.screened = time + DT;
                metrics.fitness = com_x;
                ended = true;
            }
        }
        if ended {
            metrics.vertical_oscillation = max(metrics.gait_frequency - metrics.vertical_oscillation, 0.0);
            metrics.gait_frequency = metrics.gait_turns * 0.5 / (time + DT);
        } else if tick + 1u == p.total_steps {
            metrics.fitness = com_x;
            metrics.vertical_oscillation = max(metrics.gait_frequency - metrics.vertical_oscillation, 0.0);
            metrics.gait_frequency = metrics.gait_turns * 0.5 / max(f32(step + 1u) * DT, DT);
        }
    }
    metrics.head_shake = head_shake;
    results[creature] = metrics;
    records[record_base] = Record(x0, v0, records[record_base].c, vec2f(0.0));
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        records[record_base + j + 1u] = Record(vec2f(q[j], qd[j]), records[record_base + j + 1u].b, vec2f(0.0), vec2f(0.0));
    }
}
