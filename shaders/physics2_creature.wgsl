// Physics v2 (src/physics2.rs) on the GPU: one creature per lane.
//
// A creature is a tree of point masses on rigid, massless bones. Its state
// is the head's position and velocity, the neck's angle and turning rate, and
// one angle and rate per other bone relative to its parent bone. Nodes are
// packed so that bone j ends at node j + 1 (node 0 is the head), and every
// bone's parent bone has a smaller index, so passes over the bones run
// parents first or children first by index. The step mirrors
// `physics2::simulate_step_inner` expression by expression, and the CPU keeps
// the same order of operations, so the two drift apart only by rounding.
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
//   bones (a | b << 8 | sensor node << 16), anchors, waveform amplitude, long
//   length, 1/period, phase, duty, stiffness, 1/duty, 1/(1 - duty), the
//   step's force (scratch), reset phase, rhythm offset (state), energy (state).
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
const MUSCLE_FIELDS: u32 = 15u;
const BONE_FIELDS: u32 = 9u;
const NO_SENSOR: u32 = 255u;
// Small bodies keep every per-bone array in registers and reach a parent
// or pivot through a chain of selects; large bodies index memory.
const SELECT: bool = SELECTTREE;
// Per-lane table in workgroup memory for the muscles, which name their
// bones by data: per bone its pivot's and child's positions and velocities
// and the force collected on it. Laid out [field][bone][lane].
const TABLE_FIELDS: u32 = 11u;

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
const WARM: bool = false;
const PUSH_OUT: f32 = 0.2;
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

// This lane's creature: constants.
var<private> nn: u32;
var<private> nb: u32;
var<private> mass: array<f32, MAXN>;
var<private> radius: array<f32, MAXN>;
var<private> fric: array<f32, MAXN>;
var<private> pivot: array<u32, MAXB>;
var<private> parent: array<u32, MAXB>;
var<private> len: array<f32, MAXB>;
var<private> lo: array<f32, MAXB>;
var<private> hi: array<f32, MAXB>;
var<private> total_mass: f32;
var<private> inv_mass: f32;
var<private> phase_q: f32;
var<private> amplitude: f32;
var<private> rough: bool;
// State: head position and velocity, joint angles and rates (bone 0: the
// neck's absolute angle and rate), contact forces of the last step.
var<private> x0: vec2f;
var<private> v0: vec2f;
var<private> q: array<f32, MAXB>;
var<private> qd: array<f32, MAXB>;
var<private> warm: array<vec2f, MAXN>;
// Kinematics: absolute angles and rates, node positions and velocities.
var<private> th: array<f32, MAXB>;
var<private> om: array<f32, MAXB>;
var<private> pos: array<vec2f, MAXN>;
var<private> vel: array<vec2f, MAXN>;
// The articulated-body pass: inertias as (ww, wx, wy) and (xx, xy, yy),
// bias forces, spatial velocities, velocity products, pivot arms (the joint
// axis is (1, arm.y, -arm.x)), U, 1/D and u of each joint, accelerations.
var<private> i0: array<vec3f, MAXB>;
var<private> i1: array<vec3f, MAXB>;
var<private> bias: array<vec3f, MAXB>;
var<private> sv: array<vec3f, MAXB>;
var<private> cvel: array<vec3f, MAXB>;
var<private> arm: array<vec2f, MAXB>;
var<private> uvec: array<vec3f, MAXB>;
var<private> dinv: array<f32, MAXB>;
var<private> uu: array<f32, MAXB>;
var<private> acc: array<vec3f, MAXB>;
var<private> qdd: array<f32, MAXB>;
var<private> root0: vec3f;
var<private> root1: vec3f;
// Scratch for responses to forces: forces passed up the tree, and the
// resulting accelerations.
var<private> pp: array<vec3f, MAXB>;
var<private> da: array<vec3f, MAXB>;
var<private> dq: array<f32, MAXB>;
// Contacts of this step.
var<private> nc: u32;
var<private> c_node: array<u32, MAXC>;
var<private> c_dn: array<vec3f, MAXC>;
var<private> c_dt: array<vec3f, MAXC>;
var<private> c_vn: array<f32, MAXC>;
var<private> c_vt: array<f32, MAXC>;
var<private> c_goal: array<f32, MAXC>;
var<private> c_mu: array<f32, MAXC>;
var<private> reach: array<f32, MAXN>;
var<private> kmat: array<f32, MAXR * (MAXR + 1u) / 2u>;
var<private> c_on: array<bool, MAXC>;
var<private> lambda: array<f32, MAXR>;
var<private> old: array<f32, MAXR>;
var<private> vrow: array<f32, MAXR>;
var<workgroup> table: array<f32, TABLE_FIELDS * MAXB * WG>;
var<private> lane_id: u32;
fn tab(field: u32, bone: u32) -> u32 {
    return (field * MAXB + bone) * WG + lane_id;
}

// Element `i` of a per-bone or per-node array, reached through selects for
// small bodies (the arrays then stay in registers).
fn get_f(a: ptr<private, array<f32, MAXB>>, i: u32) -> f32 {
    if SELECT {
        var r = (*a)[0];
        for (var k = 1u; k < MAXB; k++) {
            if k == i {
                r = (*a)[k];
            }
        }
        return r;
    }
    return (*a)[i];
}
fn get_b3(a: ptr<private, array<vec3f, MAXB>>, i: u32) -> vec3f {
    if SELECT {
        var r = (*a)[0];
        for (var k = 1u; k < MAXB; k++) {
            if k == i {
                r = (*a)[k];
            }
        }
        return r;
    }
    return (*a)[i];
}
fn add_b3(a: ptr<private, array<vec3f, MAXB>>, i: u32, v: vec3f) {
    if SELECT {
        for (var k = 0u; k < MAXB; k++) {
            if k == i {
                (*a)[k] += v;
            }
        }
        return;
    }
    (*a)[i] += v;
}
fn get_n2(a: ptr<private, array<vec2f, MAXN>>, i: u32) -> vec2f {
    if SELECT {
        var r = (*a)[0];
        for (var k = 1u; k < MAXN; k++) {
            if k == i {
                r = (*a)[k];
            }
        }
        return r;
    }
    return (*a)[i];
}
fn set_n2(a: ptr<private, array<vec2f, MAXN>>, i: u32, v: vec2f) {
    if SELECT {
        for (var k = 0u; k < MAXN; k++) {
            if k == i {
                (*a)[k] = v;
            }
        }
        return;
    }
    (*a)[i] = v;
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
fn body_of(i: u32) -> u32 {
    return select(i - 1u, 0u, i == 0u);
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

// Absolute angles and rates and node velocities from the state, for the
// planting pass, which needs no positions.
fn velocities() {
    vel[0] = v0;
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        var t = q[0];
        var w = qd[0];
        if j > 0u {
            t = get_f(&th, parent[j]) + q[j];
            w = get_f(&om, parent[j]) + qd[j];
        }
        th[j] = t;
        om[j] = w;
        let sn = sin(t);
        let cs = cos(t);
        let l = len[j];
        let pv0 = get_n2(&vel, pivot[j]);
        vel[j + 1u] = vec2f(pv0.x - l * w * sn, pv0.y + l * w * cs);
    }
}

// Absolute angles and rates, node positions and velocities, from the state.
fn kinematics() {
    pos[0] = x0;
    vel[0] = v0;
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        var t = q[0];
        var w = qd[0];
        if j > 0u {
            t = get_f(&th, parent[j]) + q[j];
            w = get_f(&om, parent[j]) + qd[j];
        }
        th[j] = t;
        om[j] = w;
        let sn = sin(t);
        let cs = cos(t);
        let l = len[j];
        let pp0 = get_n2(&pos, pivot[j]);
        let pv0 = get_n2(&vel, pivot[j]);
        pos[j + 1u] = vec2f(pp0.x + l * cs, pp0.y + l * sn);
        vel[j + 1u] = vec2f(pv0.x - l * w * sn, pv0.y + l * w * cs);
    }
}

fn momentum() -> vec2f {
    var m = vec2f(0.0);
    for (var i = 0u; i < MAXN; i++) {
        if i >= nn { break; }
        m += vel[i] * mass[i];
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

// The accelerations (`da`, `dq`) that the spatial forces in `pp` cause,
// through the articulated inertias of the last `solve` (children first,
// then parents first). Clears nothing: the caller fills `pp`.
fn response() {
    for (var i = 1u; i < MAXB; i++) {
        let j = MAXB - i;
        if j >= nb { continue; }
        let t = -sdot(axis_of(j), pp[j]);
        dq[j] = t;
        add_b3(&pp, parent[j], pp[j] + uvec[j] * (t * dinv[j]));
    }
    da[0] = -sym_mul(root0, root1, pp[0]);
    dq[0] = 0.0;
    for (var j = 1u; j < MAXB; j++) {
        if j >= nb { break; }
        let a = get_b3(&da, parent[j]);
        let t = (dq[j] - sdot(uvec[j], a)) * dinv[j];
        dq[j] = t;
        da[j] = a + axis_of(j) * t;
    }
}

fn clear_forces() {
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        pp[j] = vec3f(0.0);
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
        let cb = body_of(c_node[ci]);
        add_b3(&pp, cb, -(c_dn[ci] * (*f)[2u * ci] + c_dt[ci] * (*f)[2u * ci + 1u]));
    }
    response();
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        acc[j] += da[j];
        qdd[j] += dq[j];
    }
}

// Projected Gauss-Seidel on the contact impulses: a touching node may
// approach the ground only as fast as its goal allows, normal forces only
// push, and friction stays within mu times the normal force and opposes the
// slip. Every row's velocity (`vrow`) is kept current as the forces change,
// so an update reads one number instead of a sum.
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
            let friction = clamp(lambda[rt] - vrow[rt] * (1.0 / kmat[tri(rt, rt)]), -bound, bound);
            let dt_ = friction - lambda[rt];
            lambda[rt] = friction;
            for (var j = 0u; j < MAXR; j++) {
                if !c_on[j / 2u] { continue; }
                vrow[j] += kmat[tri(j, rt)] * dt_;
            }
        }
    }
}

// Fills contact slot `ci` for node `i`.
fn fill_contact(ci: u32, i: u32, origin: vec2f) {
    let g = terrain(pos[i].x);
    let secant = sqrt(1.0 + g.y * g.y);
    let normal = vec2f(-g.y / secant, 1.0 / secant);
    let tangent = vec2f(normal.y, -normal.x);
    let gap = (pos[i].y - g.x) / secant - radius[i];
    let body = body_of(i);
    let r = pos[i] - origin;
    let v = vel[i];
    let w = om[body];
    let dn = force_at(r, normal);
    let dtan = force_at(r, tangent);
    let a = acc[body];
    c_node[ci] = i;
    c_dn[ci] = dn;
    c_dt[ci] = dtan;
    c_vn[ci] = v.x * normal.x + v.y * normal.y
        + DT * (sdot(dn, a) + w * (-v.y * normal.x + v.x * normal.y));
    c_vt[ci] = v.x * tangent.x + v.y * tangent.y
        + DT * (sdot(dtan, a) + w * (-v.y * tangent.x + v.x * tangent.y));
    c_goal[ci] = select(-gap * PUSH_OUT * RATE, -gap * RATE, gap >= 0.0);
    c_mu[ci] = fric[i] * p.friction;
}

// Ground contacts at velocity level, solved together (physics2's contact
// section): detection of the deepest MAXC nodes, the contact-space matrix,
// Gauss-Seidel, the forces' response, and one pass that plants the contacts
// against the step's end pose. Returns `outside` (the step's impulse from
// gravity and wind) plus the ground's impulse. Small bodies
// give every node its own slot (MAXC is then their node count), so every
// index stays static; large ones pack the chosen nodes in node order.
fn contacts(origin: vec2f, before: vec2f, outside: vec2f) -> vec2f {
    nc = 0u;
    var candidates = 0u;
    for (var i = 0u; i < MAXN; i++) {
        reach[i] = 1e30;
        if i >= nn { continue; }
        let g = terrain(pos[i].x);
        let secant = sqrt(1.0 + g.y * g.y);
        let normal = vec2f(-g.y / secant, 1.0 / secant);
        let gap = (pos[i].y - g.x) / secant - radius[i];
        let body = body_of(i);
        let r = pos[i] - origin;
        let v = vel[i];
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
    if MAXC == MAXN {
        for (var i = 0u; i < MAXN; i++) {
            if reach[i] > 1e29 { continue; }
            c_on[i] = true;
            fill_contact(i, i, origin);
        }
        nc = candidates;
    } else {
        // The deepest MAXC nodes, in node order; small bodies write their
        // slots through selects so the indices stay static.
        for (var i = 0u; i < MAXN; i++) {
            if i >= nn { break; }
            if reach[i] > 1e29 { continue; }
            if candidates > MAXC {
                var rank = 0u;
                for (var j = 0u; j < MAXN; j++) {
                    if j >= nn { break; }
                    if reach[j] < reach[i] || (reach[j] == reach[i] && j < i) {
                        rank += 1u;
                    }
                }
                if rank >= MAXC { continue; }
            }
            if SELECT {
                for (var k = 0u; k < MAXC; k++) {
                    if k == nc {
                        c_on[k] = true;
                        fill_contact(k, i, origin);
                    }
                }
            } else {
                c_on[nc] = true;
                fill_contact(nc, i, origin);
            }
            nc += 1u;
        }
    }
    // Contact-space matrix, column by column from each unit force's
    // response; symmetric, the lower triangle kept.
    for (var col = 0u; col < MAXR; col++) {
        if !c_on[col / 2u] { continue; }
        let ci = col / 2u;
        var fdir = c_dn[ci];
        if (col & 1u) == 1u {
            fdir = c_dt[ci];
        }
        clear_forces();
        add_b3(&pp, body_of(c_node[ci]), -fdir);
        response();
        for (var row = 0u; row < MAXR; row++) {
            if row < col || !c_on[row / 2u] { continue; }
            let ri = row / 2u;
            var rdir = c_dn[ri];
            if (row & 1u) == 1u {
                rdir = c_dt[ri];
            }
            kmat[tri(row, col)] = DT * sdot(rdir, get_b3(&da, body_of(c_node[ri])));
        }
    }
    for (var ci = 0u; ci < MAXC; ci++) {
        var ln = 0.0;
        var lt = 0.0;
        if WARM && c_on[ci] {
            let w = get_n2(&warm, c_node[ci]);
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
    {
        let sx0 = x0;
        let sv0 = v0;
        var sq: array<f32, MAXB>;
        var sqd: array<f32, MAXB>;
        for (var j = 0u; j < MAXB; j++) {
            sq[j] = q[j];
            sqd[j] = qd[j];
        }
        integrate_state(1.0);
        velocities();
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
            let v = get_n2(&vel, c_node[ci]) + shift;
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
        for (var j = 0u; j < MAXR; j++) {
            old[j] = lambda[j] - old[j];
        }
        apply_contacts(&old);
    }
    var impulse = outside;
    for (var ci = 0u; ci < MAXC; ci++) {
        if !c_on[ci] { continue; }
        set_n2(&warm, c_node[ci], vec2f(lambda[2u * ci], lambda[2u * ci + 1u]));
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
    let base = creature * STRIDE;

    let head = records[base];
    x0 = head.a;
    v0 = head.b;
    warm[0] = head.c;
    mass[0] = bone_data[tile.y + tl + 7u * TILE];
    radius[0] = bone_data[tile.y + tl + 8u * TILE];
    fric[0] = bone_data[tile.y + tl + 2u * TILE];
    total_mass = mass[0];
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        let field = tile.y + j * BONE_FIELDS * TILE + tl;
        let pv = bitcast<u32>(bone_data[field]);
        pivot[j] = pv;
        parent[j] = body_of(pv);
        len[j] = bone_data[field + TILE];
        lo[j] = bone_data[field + 2u * TILE];
        hi[j] = bone_data[field + 3u * TILE];
        mass[j + 1u] = bone_data[field + 4u * TILE];
        radius[j + 1u] = bone_data[field + 5u * TILE];
        fric[j + 1u] = bone_data[field + 6u * TILE];
        total_mass += mass[j + 1u];
        let r = records[base + j + 1u];
        q[j] = r.a.x;
        qd[j] = r.a.y;
        warm[j + 1u] = r.b;
    }
    inv_mass = 1.0 / total_mass;
    let inv_nodes = 1.0 / f32(nn);
    let inv_capacity = 1.0 / (MUSCLE_CAPACITY * p.muscle_energy);
    kinematics();

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
        let head_before = vel[0];
        let origin = pos[0];
        let before = momentum();
        // For the first-law check in flight.
        var energy_start = 0.0;
        var energy_scale = 0.0;
        var mass_x_start = 0.0;
        for (var i = 0u; i < MAXN; i++) {
            if i >= nn { break; }
            let kinetic = 0.5 * mass[i] * (vel[i].x * vel[i].x + vel[i].y * vel[i].y);
            let potential = mass[i] * p.gravity * pos[i].y;
            energy_start += kinetic + potential;
            energy_scale += kinetic + abs(potential);
            mass_x_start += pos[i].x * mass[i];
        }
        var muscle_start = 0.0;
        // Body inertias (true, for the velocity products) and forces, which
        // `bias` collects with the opposite sign at the end.
        var force: array<vec3f, MAXB>;
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            let m = mass[j + 1u];
            let r = pos[j + 1u] - origin;
            i0[j] = vec3f(m * (r.x * r.x + r.y * r.y), -m * r.y, m * r.x);
            i1[j] = vec3f(m, 0.0, m);
            force[j] = vec3f(0.0);
        }
        i1[0] += vec3f(mass[0], 0.0, mass[0]);
        sv[0] = vec3f(qd[0], v0.x, v0.y);
        for (var j = 1u; j < MAXB; j++) {
            if j >= nb { break; }
            arm[j] = get_n2(&pos, pivot[j]) - origin;
            let axis = axis_of(j);
            sv[j] = get_b3(&sv, parent[j]) + axis * qd[j];
            cvel[j] = crm(sv[j], axis) * qd[j];
        }
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            bias[j] = crf(sv[j], sym_mul(i0[j], i1[j], sv[j]));
        }
        // Gravity and wind on every node.
        for (var i = 0u; i < MAXN; i++) {
            if i >= nn { break; }
            let m = mass[i];
            let f = vec2f(p.wind * m, -p.gravity * m);
            force[body_of(i)] += force_at(pos[i] - origin, f);
        }
        // Muscles, through the per-lane table: each bone's end positions and
        // velocities, and the forces so far.
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            let pp0 = get_n2(&pos, pivot[j]);
            let pv0 = get_n2(&vel, pivot[j]);
            table[tab(0u, j)] = pp0.x;
            table[tab(1u, j)] = pp0.y;
            table[tab(2u, j)] = pv0.x;
            table[tab(3u, j)] = pv0.y;
            table[tab(4u, j)] = pos[j + 1u].x;
            table[tab(5u, j)] = pos[j + 1u].y;
            table[tab(6u, j)] = vel[j + 1u].x;
            table[tab(7u, j)] = vel[j + 1u].y;
            table[tab(8u, j)] = force[j].x;
            table[tab(9u, j)] = force[j].y;
            table[tab(10u, j)] = force[j].z;
        }
        for (var k = 0u; k < muscle_count; k++) {
            let field = tile.x + k * MUSCLE_FIELDS * TILE + tl;
            let packed = bitcast<u32>(muscle_data[field]);
            let ba = packed & 0xffu;
            let bb = (packed >> 8u) & 0xffu;
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
            let pa0 = vec2f(table[tab(0u, ba)], table[tab(1u, ba)]);
            let va0 = vec2f(table[tab(2u, ba)], table[tab(3u, ba)]);
            let pa1 = vec2f(table[tab(4u, ba)], table[tab(5u, ba)]);
            let va1 = vec2f(table[tab(6u, ba)], table[tab(7u, ba)]);
            let pb0 = vec2f(table[tab(0u, bb)], table[tab(1u, bb)]);
            let vb0 = vec2f(table[tab(2u, bb)], table[tab(3u, bb)]);
            let pb1 = vec2f(table[tab(4u, bb)], table[tab(5u, bb)]);
            let vb1 = vec2f(table[tab(6u, bb)], table[tab(7u, bb)]);
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
            let magnitude = clamp(drive + relative * 0.15, -MAX_MUSCLE_FORCE, MAX_MUSCLE_FORCE);
            let work = abs(magnitude * relative) * DT;
            muscle_data[field + 14u * TILE] = clamp(
                energy - work * inv_capacity
                    + MUSCLE_RECOVERY * p.muscle_recovery * DT * (1.0 - energy),
                0.0,
                1.0,
            );
            muscle_data[field + 11u * TILE] = magnitude;
            muscle_start += magnitude * length_m;
            let f = dir * magnitude;
            let fa = force_at(pa - origin, f);
            table[tab(8u, ba)] += fa.x;
            table[tab(9u, ba)] += fa.y;
            table[tab(10u, ba)] += fa.z;
            let fb = force_at(pb - origin, f);
            table[tab(8u, bb)] -= fb.x;
            table[tab(9u, bb)] -= fb.y;
            table[tab(10u, bb)] -= fb.z;
        }
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            force[j] = vec3f(table[tab(8u, j)], table[tab(9u, j)], table[tab(10u, j)]);
        }
        // Spin cap: rotational drag past the cap, implicit, toward rest.
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            let w = om[j];
            if abs(w) > SPIN_CAP {
                let l = len[j];
                let drag = SPIN_HARDNESS * mass[j + 1u] * l * l * (abs(w) * INV_SPIN_CAP - 1.0);
                i0[j].x += drag;
                force[j].x += -drag * RATE * w;
            }
        }
        for (var j = 0u; j < MAXB; j++) {
            if j >= nb { break; }
            bias[j] -= force[j];
        }
        // Articulated-body pass, children first, with joint damping and the
        // joint limits' inelastic stops implicit in each joint's inertia.
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
            let predicted = qj + DT * qdj;
            let upper = predicted > hi[j];
            if upper || predicted < lo[j] {
                let room = select(lo[j] - qj, hi[j] - qj, upper);
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
            let pa = bias[j] + sym_mul(a0, a1, cvel[j]) + uv * (u * di);
            let pr = parent[j];
            add_b3(&i0, pr, a0);
            add_b3(&i1, pr, a1);
            add_b3(&bias, pr, pa);
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
        for (var j = 1u; j < MAXB; j++) {
            if j >= nb { break; }
            let a = get_b3(&acc, parent[j]) + cvel[j];
            qdd[j] = (uu[j] - sdot(uvec[j], a)) * dinv[j];
            acc[j] = a + axis_of(j) * qdd[j];
        }

        var impulse = vec2f(p.wind * total_mass * DT, -p.gravity * total_mass * DT);
        nc = 0u;
        if grounded {
            impulse = contacts(origin, before, impulse);
        }
        if nc == 0u {
            for (var i = 0u; i < MAXN; i++) {
                if i >= nn { break; }
                warm[i] = vec2f(0.0);
            }
        } else {
            for (var i = 0u; i < MAXN; i++) {
                if i >= nn { break; }
                var used = false;
                for (var ci = 0u; ci < MAXC; ci++) {
                    if c_on[ci] && c_node[ci] == i {
                        used = true;
                    }
                }
                if !used {
                    warm[i] = vec2f(0.0);
                }
            }
        }
        integrate_state(p.air);
        kinematics();
        // Momentum balance.
        let after = momentum();
        let expected = vec2f((before.x + impulse.x) * p.air, (before.y + impulse.y) * p.air);
        let shift = vec2f((expected.x - after.x) * inv_mass, (expected.y - after.y) * inv_mass);
        v0 += shift;
        for (var i = 0u; i < MAXN; i++) {
            if i >= nn { break; }
            vel[i] += shift;
        }
        // First law in flight.
        if nc == 0u {
            var muscle_end = 0.0;
            for (var j = 0u; j < MAXB; j++) {
                if j >= nb { break; }
                let pp0 = get_n2(&pos, pivot[j]);
                table[tab(0u, j)] = pp0.x;
                table[tab(1u, j)] = pp0.y;
                table[tab(4u, j)] = pos[j + 1u].x;
                table[tab(5u, j)] = pos[j + 1u].y;
            }
            for (var k = 0u; k < muscle_count; k++) {
                let field = tile.x + k * MUSCLE_FIELDS * TILE + tl;
                let packed = bitcast<u32>(muscle_data[field]);
                let ba = packed & 0xffu;
                let bb = (packed >> 8u) & 0xffu;
                let anchor_a = muscle_data[field + TILE];
                let anchor_b = muscle_data[field + 2u * TILE];
                let pa0 = vec2f(table[tab(0u, ba)], table[tab(1u, ba)]);
                let pa1 = vec2f(table[tab(4u, ba)], table[tab(5u, ba)]);
                let pb0 = vec2f(table[tab(0u, bb)], table[tab(1u, bb)]);
                let pb1 = vec2f(table[tab(4u, bb)], table[tab(5u, bb)]);
                let pa = vec2f(pa0.x + (pa1.x - pa0.x) * anchor_a, pa0.y + (pa1.y - pa0.y) * anchor_a);
                let pb = vec2f(pb0.x + (pb1.x - pb0.x) * anchor_b, pb0.y + (pb1.y - pb0.y) * anchor_b);
                let d = pb - pa;
                muscle_end += muscle_data[field + 11u * TILE] * sqrt(d.x * d.x + d.y * d.y);
            }
            var energy_end = 0.0;
            var mass_x_end = 0.0;
            for (var i = 0u; i < MAXN; i++) {
                if i >= nn { break; }
                let kinetic = 0.5 * mass[i] * (vel[i].x * vel[i].x + vel[i].y * vel[i].y);
                energy_end += kinetic + mass[i] * p.gravity * pos[i].y;
                mass_x_end += pos[i].x * mass[i];
            }
            let work = (muscle_start - muscle_end) + p.wind * (mass_x_end - mass_x_start);
            let excess = energy_end - energy_start - work - (1e-4 + 1e-5 * energy_scale);
            if excess > 0.0 {
                let center = vec2f(expected.x * inv_mass, expected.y * inv_mass);
                var internal = 0.0;
                for (var i = 0u; i < MAXN; i++) {
                    if i >= nn { break; }
                    let x = vel[i].x - center.x;
                    let y = vel[i].y - center.y;
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
                    vel[i] = vec2f(center.x + keep * (vel[i].x - center.x), center.y + keep * (vel[i].y - center.y));
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
            let x = pos[i].x;
            let y = pos[i].y;
            if !(abs(x) <= 1e6) || !(abs(y) <= 1e6) {
                failed = true;
            }
            let r = radius[i];
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
                        contact_lo |= 1u << i;
                        now_lo |= 1u << i;
                    } else {
                        contact_hi |= 1u << (i - 32u);
                        now_hi |= 1u << (i - 32u);
                    }
                } else if y > floor_y + LIFT_CLEARANCE {
                    if i < 32u {
                        lift_lo |= contact_lo & (1u << i);
                    } else {
                        lift_hi |= contact_hi & (1u << (i - 32u));
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
                let sensor = (bitcast<u32>(muscle_data[field]) >> 16u) & 0xffu;
                if sensor == NO_SENSOR {
                    continue;
                }
                let touched = select((down_hi >> (sensor - 32u)) & 1u, (down_lo >> sensor) & 1u, sensor < 32u);
                if touched == 1u {
                    let clock = next * muscle_data[field + 5u * TILE] + muscle_data[field + 6u * TILE];
                    let x = muscle_data[field + 12u * TILE] - clock;
                    muscle_data[field + 13u * TILE] = x - floor(x);
                }
            }
        }
        if time >= HEAD_SHAKE_WINDOW {
            let dv = vel[0] - head_before;
            let accel = sqrt(dv.x * dv.x + dv.y * dv.y) * RATE;
            head_shake += (accel - head_shake) * min(1.0 / (HEAD_SHAKE_WINDOW * RATE), 1.0);
        }
        metrics.head_shake = head_shake;
        var broken = false;
        for (var j = 1u; j < MAXB; j++) {
            if j >= nb { break; }
            if q[j] < lo[j] - JOINT_BREAK || q[j] > hi[j] + JOINT_BREAK {
                broken = true;
            }
        }
        let fell = pos[0].y < pos[1].y || broken || head_shake > HEAD_SHAKE_LIMIT || failed;
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
    records[base] = Record(x0, v0, warm[0], vec2f(0.0));
    for (var j = 0u; j < MAXB; j++) {
        if j >= nb { break; }
        records[base + j + 1u] = Record(vec2f(q[j], qd[j]), warm[j + 1u], vec2f(0.0), vec2f(0.0));
    }
}
