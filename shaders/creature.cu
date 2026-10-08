// The creature physics: one GPU thread simulates one creature from its start
// pose to the end of its trial. src/kernel.rs packs the creatures and writes
// the #defines in front of this text.
//
// A creature is a tree of point masses (nodes) joined by bones. The kernel
// uses position-based dynamics with small substeps (Mueller et al., "Small
// Steps in Physics Simulation", 2019). Each substep:
//
//   1. forces: gravity, wind, drag, water, mud, brambles and the muscles
//      change the node velocities;
//   2. prediction: every node moves by its velocity;
//   3. constraints, once each in this order: every bone keeps its length,
//      every joint stays inside its range, and every node inside the ground
//      is moved out of it, with Coulomb friction on the move;
//   4. velocities: each node's velocity is its move over the substep, then
//      joint damping takes a share of every joint's turning speed.
//
// Constraints only move nodes toward a valid pose and friction only takes
// back sliding, so no step adds energy. The muscles are the only source of
// work. Every node is its own contact, with no limit on their number.
//
// Where the state lives. The node positions, velocities and muscle forces of
// a body are in shared memory, one column per thread (two for a body of more
// than NS nodes, see `s_pos`). A thread reads them by the node numbers of its
// bones, joints and muscles, and a warp reads a row of shared memory without
// bank conflicts whatever those numbers are. In the thread's own memory the
// same reads cost a cache line per lane, and the lines of all resident warps
// do not fit in the L1 cache. The constants of a body and the energy of its
// muscles stay in the thread's own memory, where a warp reads them by one
// index. A warp runs bodies of one of the two classes at a time (see
// `advance`).
//
// Defines: RATE (steps per second), SUBSTEPS, SETTLE, SAMPLE, MAXN, MAXM,
// BLOCK, MIN_BLOCKS, RECORD, the world flags (GROUND, TERRAIN, SLOPE, GAPS,
// HURDLES, QUAKE, MUD, WATER, ICE, WIND, AIR, BRAMBLES) and the physics
// constants.

// A step lasts `DT` seconds and a substep `H` seconds. `INV_H` is the number
// of substeps per second.
#define DT (1.0f / RATE)
#define H (1.0f / (RATE * SUBSTEPS))
#define INV_H (RATE * SUBSTEPS)
#define PI_F 3.14159265359f
#define TAU_F 6.28318530718f
// The value in the records for no parent bone and no sensor node.
#define NONE 0xffffffffu
// Steps per 1/60 s, the clock of the head shake measure.
#define HEAD_SAMPLE ((unsigned)(RATE / 60.0f + 0.5f))
// Words of a node record and of a bone record, and floats of a muscle record.
// `kernel::NODE_WORDS`, `kernel::BONE_WORDS` and `kernel::MUSCLE_FIELDS` have
// the same values.
#define NODE_WORDS 6u
#define BONE_WORDS 6u
#define MUSCLE_WORDS 20u

// One early rung's rule, with the layout of `rungs::Rung`. The creature stops
// when the chain of fused multiply-adds of `w` and the six features, from zero,
// is below `bias`. The features are the distance, the speed over the last half
// second, the share of nodes that touched the ground, the head shake, the mean
// muscle energy store and the rhythm period. Bit b of `off` is set when
// cadence band b is off.
struct RungParams {
    float w[6];
    float bias;
    unsigned off;
};
// The launch parameters, with the layout of `kernel::Params`. That struct says
// what each field means. This kernel does not read `air` and `muscle_energy`
// and reads `air_sub` and `inv_muscle_energy` in their place. Only a
// recording kernel reads `stride`.
struct Params {
    unsigned count;
    unsigned base;
    unsigned steps;
    unsigned screen_step;
    unsigned stride;
    float gravity;
    float air;
    float friction;
    float terrain;
    float muscle_energy;
    float muscle_recovery;
    float slope;
    float wind;
    float mud;
    float gaps;
    float hurdles;
    float quake;
    float screen_bar;
    float screen_bar_young;
    float screen_bar_reshaped;
    float water;
    float patches;
    float air_sub;
    float inv_muscle_energy;
    float brambles;
    RungParams r1;
    RungParams r2;
};
// The result of a trial, with the layout of `creature_kernel::GpuResult`. The
// field docs there say what each field holds when the trial has ended. While
// it runs, some fields count something else, and `run_step` rewrites them at
// the end.
struct Result {
    float fitness;
    float ground_contact;
    float vertical_oscillation;
    float gait_frequency;
    float previous_center_y;
    float vertical_extremum;
    float vertical_trend;
    float gait_turns;
    float height_sum;
    float contact_lo;
    float contact_hi;
    float lift_lo;
    float lift_hi;
    float ground_lo;
    float ground_hi;
    float fall_time;
    float head_shake;
    float screen_x;
    float screened;
};

__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
// `a` plus a whole number of turns, in [-PI_F, PI_F).
__device__ __forceinline__ float wrapf(float a) { return a - TAU_F * floorf((a + PI_F) * (1.0f / TAU_F)); }
// The part of `x` above its floor. It is in [0, 1) for a negative `x` too.
__device__ __forceinline__ float fracf(float x) { return x - floorf(x); }

// Half precision, as the rung trace carries values. `f2h` rounds a float to
// the nearest half and gives its 16 bits. `h2f` turns such bits into a float.
__device__ __forceinline__ unsigned f2h(float v) {
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(v));
    return (unsigned)r;
}
__device__ __forceinline__ float h2f(unsigned h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"((unsigned short)h));
    return r;
}

// A muscle's rhythm at time `t`: 0 at its longest, 1 at its shortest. It rises
// over `duty` of the cycle and falls over the rest, each as half a cosine.
// `phase` and `offset` shift the cycle, and a touchdown sets `offset`.
// `inv_duty` is 1 / `duty` and `inv_complement` is 1 / (1 - `duty`).
__device__ float rhythm(float t, float inv_period, float phase, float offset, float duty, float inv_duty, float inv_complement) {
    const float ph = fracf(t * inv_period + phase + offset);
    if (ph < duty) {
        return 0.5f - 0.5f * __cosf(PI_F * ph * inv_duty);
    }
    return 0.5f + 0.5f * __cosf(PI_F * (ph - duty) * inv_complement);
}

// Height and slope of the ground under x for a creature with terrain
// amplitude `amp` and quake phase `qphase`. The bumps use the numbers of
// `physics::TERRAIN_WAVES` as literals, and the pits and the hurdles follow
// `physics::gaps` and `physics::hurdles`. A change to the ground here needs the
// same change on the host.
__device__ float2 ground(float x, float amp, float qphase, const Params& p) {
    float height = 0.0f, slope = 0.0f;
#if TERRAIN
    // Two trains of bumps with wavelengths 1.1 m and 0.43 m and weights 0.65
    // and 0.35. Each bump is 16 u^2 (1 - u)^2 over one wavelength, and `qphase`
    // shifts both trains.
    {
        const float t0 = x * (1.0f / 1.1f) + qphase;
        const float u0 = fracf(t0);
        const float w0 = u0 * (1.0f - u0);
        const float t1 = x * (1.0f / 0.43f) + 0.3f + qphase;
        const float u1 = fracf(t1);
        const float w1 = u1 * (1.0f - u1);
        height = amp * (0.65f * 16.0f * w0 * w0 + 0.35f * 16.0f * w1 * w1);
        slope = amp * (0.65f * 32.0f * w0 * (1.0f - 2.0f * u0) * (1.0f / 1.1f)
            + 0.35f * 32.0f * w1 * (1.0f - 2.0f * u1) * (1.0f / 0.43f));
    }
#endif
#if SLOPE
    // The ground rises by `p.slope` per meter.
    height += p.slope * x;
    slope += p.slope;
#endif
#if GAPS
    // Pits `p.gaps` wide and `GAP_DEPTH` deep, with walls that are ramps of at
    // most `GAP_RUN`. Their centers are 2 + 4 * `p.gaps` meters apart.
    {
        const float spacing = 2.0f + 4.0f * p.gaps;
        const float center = spacing * 0.5f;
        const float r = x - floorf(x / spacing) * spacing;
        const float distance = fabsf(r - center);
        const float half_w = 0.5f * p.gaps;
        const float run = fmaxf(fminf(GAP_RUN, half_w), 1e-6f);
        float factor = clampf((half_w - distance) / run, 0.0f, 1.0f);
        if (distance <= half_w - run) { factor = 1.0f; } else if (distance >= half_w) { factor = 0.0f; }
        height -= GAP_DEPTH * factor;
        if (distance > half_w - run && distance < half_w) {
            slope -= GAP_DEPTH * ((r < center ? 1.0f : -1.0f) / run);
        }
    }
#endif
#if HURDLES
    // Steps `p.hurdles` high every `HURDLE_SPACING` meters. Each has a flat
    // top of `HURDLE_TOP` and ramps of `HURDLE_RUN`.
    {
        const float spacing = HURDLE_SPACING;
        const float center = spacing * 0.5f;
        const float r = x - floorf(x / spacing) * spacing;
        const float distance = fabsf(r - center);
        const float half_w = 0.5f * HURDLE_TOP;
        const float run = HURDLE_RUN;
        float factor = clampf((half_w + run - distance) / run, 0.0f, 1.0f);
        if (distance <= half_w) { factor = 1.0f; } else if (distance >= half_w + run) { factor = 0.0f; }
        height += p.hurdles * factor;
        if (distance > half_w && distance < half_w + run) {
            slope += p.hurdles * ((r < center ? 1.0f : -1.0f) / run);
        }
    }
#endif
    return make_float2(height, slope);
}

// Ice at x: 1 on a patch, 0 between patches, with smooth edges. It is the
// formula of `physics::ice`.
__device__ float ice_at(float x) {
    const float w = fracf(x * ICE_INV);
    const float t = fabsf(w - 0.5f) * 2.0f;
    const float s = clampf((0.7f - t) * 2.5f, 0.0f, 1.0f);
    return s * s * (3.0f - 2.0f * s);
}

// The moving state of the nodes is in shared memory: three tables with a pair
// per node (position, velocity, and the force of the muscles, which the node
// loop then replaces with the position at the start of the substep), NS rows
// by one column per thread. A warp reads a row without bank conflicts
// whatever nodes its lanes ask for.
//   MODE 0, bodies of at most NS nodes: node i is in row i of the thread's
//     column. All 32 lanes of a warp run.
//   MODE 1, bodies of up to 2 * NS nodes: node i is in row i / 2, in the
//     thread's column when i is even and in the column 16 to its right when i
//     is odd. Only the first 16 lanes of a warp run.
#define NS 16u
__shared__ float2 s_pos[NS * BLOCK];
__shared__ float2 s_vel[NS * BLOCK];
__shared__ float2 s_fp[NS * BLOCK];

// One creature's constants and the state of its muscles, in the thread's own
// memory. The positions, velocities and forces of its nodes are in the tables
// above.
struct Body {
    unsigned nodes, bones, muscles;
    // `total_mass` is loaded and not used. `inv_mass` is one over it. `amp` is
    // the height of this creature's ground bumps and `qphase` their phase.
    float total_mass, inv_mass, amp, qphase;
    // Nodes: mass, inverse mass, radius and friction coefficient. Bit i of
    // `feet` is set when node i is a foot.
    float mass[MAXN], inv_m[MAXN], radius[MAXN], fric[MAXN];
    unsigned feet;
    // Bones: bone j joins node pivot[j] to node j + 1. `parent` is the bone
    // that bone j turns against, or 0xff for none. `lo` and `hi` are the range
    // of its joint in radians.
    unsigned char pivot[MAXN], parent[MAXN];
    float length[MAXN], lo[MAXN], hi[MAXN];
    // Muscles: the energy store (1 when rested), the rhythm offset that a
    // touchdown sets, the demand of the step before the store and Hill's
    // relation scale it, and the pull of the last substep, which only a
    // recording sets.
    float energy[MAXM], offset[MAXM], demand[MAXM], pull[MAXM];
    // The contact forces per node, the push of the ground and the friction,
    // averaged over the last step. Only a recording fills them.
    float normal_force[MAXN], friction_force[MAXN];
};

// Where node i of this thread is in a table.
template <int MODE>
__device__ __forceinline__ unsigned node_slot(unsigned i) {
    if constexpr (MODE == 0) { return i * BLOCK + threadIdx.x; }
    else { return (i >> 1) * BLOCK + threadIdx.x + ((i & 1u) << 4); }
}
// Node i's position, velocity, and force-then-previous-position pair.
template <int MODE>
__device__ __forceinline__ float2& pos_at(Body&, unsigned i) { return s_pos[node_slot<MODE>(i)]; }
template <int MODE>
__device__ __forceinline__ float2& vel_at(Body&, unsigned i) { return s_vel[node_slot<MODE>(i)]; }
template <int MODE>
__device__ __forceinline__ float2& fp_at(Body&, unsigned i) { return s_fp[node_slot<MODE>(i)]; }

// A joint: the relative angle of a bone to its parent bone and its gradient.
// A joint moves three nodes: the tip of the bone (`tip`), the parent bone's
// far end (`far`) and the node they share (`hub`). The gradient of `hub` is
// minus the sum of the other two, because moving all three nodes by the same
// amount does not change the angle.
struct Joint {
    unsigned tip, far, hub;
    float2 g_tip, g_far, g_hub;
    float angle;
};
// Joint j of the body. Its angle is taken within half a turn of the middle of
// the joint's range. With `with_angle` false the `atan2` is skipped and the
// angle is 0, for the callers that need only the gradients.
template <int MODE>
__device__ Joint joint(Body& b, unsigned j, bool with_angle = true) {
    Joint k;
    const unsigned pj = b.parent[j];
    k.tip = j + 1u;
    k.hub = b.pivot[j];
    // A bone at the head turns against the neck, which also starts there.
    const bool at_head = k.hub == b.pivot[pj];
    k.far = at_head ? pj + 1u : b.pivot[pj];
    // u is the parent bone, v this bone, each from its pivot to its tip.
    const float2 u = make_float2(pos_at<MODE>(b, pj + 1u).x - pos_at<MODE>(b, b.pivot[pj]).x, pos_at<MODE>(b, pj + 1u).y - pos_at<MODE>(b, b.pivot[pj]).y);
    const float2 v = make_float2(pos_at<MODE>(b, k.tip).x - pos_at<MODE>(b, k.hub).x, pos_at<MODE>(b, k.tip).y - pos_at<MODE>(b, k.hub).y);
    const float uu = fmaxf(u.x * u.x + u.y * u.y, 1e-12f), vv = fmaxf(v.x * v.x + v.y * v.y, 1e-12f);
    // The angle from u to v, one atan2 of their cross and dot products.
    k.angle = 0.0f;
    if (with_angle) {
        const float turn = atan2f(u.x * v.y - u.y * v.x, u.x * v.x + u.y * v.y);
        const float mid = 0.5f * (b.lo[j] + b.hi[j]);
        k.angle = mid + wrapf(turn - mid);
    }
    // d(atan2(w))/dw = perp(w) / |w|^2, perp(w) = (-w.y, w.x). The angle is
    // atan2(v) - atan2(u): the tip of v counts +, the tip of u counts -, and
    // the pivot of u counts +. At the head the far node is the tip of u, so
    // `g_far` has the opposite sign there.
    k.g_tip = make_float2(-v.y / vv, v.x / vv);
    const float2 g_u = make_float2(-u.y / uu, u.x / uu);
    k.g_far = at_head ? make_float2(-g_u.x, -g_u.y) : g_u;
    k.g_hub = make_float2(-(k.g_tip.x + k.g_far.x), -(k.g_tip.y + k.g_far.y));
    return k;
}

// The sum over the joint's three nodes of the inverse mass times the squared
// gradient. A push of `lambda` along the gradients moves the angle by `lambda`
// times this weight.
__device__ float joint_weight(const Body& b, const Joint& k) {
    return b.inv_m[k.tip] * (k.g_tip.x * k.g_tip.x + k.g_tip.y * k.g_tip.y)
         + b.inv_m[k.far] * (k.g_far.x * k.g_far.x + k.g_far.y * k.g_far.y)
         + b.inv_m[k.hub] * (k.g_hub.x * k.g_hub.x + k.g_hub.y * k.g_hub.y);
}

// Adds `lambda` times each node's inverse mass times its gradient to the
// positions or, with VEL, the velocities.
template <int MODE, bool VEL>
__device__ void joint_push(Body& b, const Joint& k, float lambda) {
    float2& tip = VEL ? vel_at<MODE>(b, k.tip) : pos_at<MODE>(b, k.tip);
    float2& far = VEL ? vel_at<MODE>(b, k.far) : pos_at<MODE>(b, k.far);
    float2& hub = VEL ? vel_at<MODE>(b, k.hub) : pos_at<MODE>(b, k.hub);
    tip.x += b.inv_m[k.tip] * lambda * k.g_tip.x;
    tip.y += b.inv_m[k.tip] * lambda * k.g_tip.y;
    far.x += b.inv_m[k.far] * lambda * k.g_far.x;
    far.y += b.inv_m[k.far] * lambda * k.g_far.y;
    hub.x += b.inv_m[k.hub] * lambda * k.g_hub.x;
    hub.y += b.inv_m[k.hub] * lambda * k.g_hub.y;
}

// A muscle record is five 16-byte lines (`kernel::fill_creature`). The
// first two hold what every substep reads, the rest what only the rhythm
// reads once a step.
struct MusclePull {
    // The two bones' pivot and tip nodes.
    unsigned a0, a1, b0, b1;
    // `anchor_a` and `anchor_b` are the anchors, as shares of the bones'
    // lengths from their pivots. `hill` is Hill's factor on the lengthening
    // speed. `cap` is the force cap (N). `inv_capacity` is one over the
    // capacity of the energy store (1/J). `tendon_k` is the stiffness of the
    // tendon (N/m) and `slack` its slack length (m).
    float anchor_a, anchor_b, hill, cap, inv_capacity, tendon_k, slack;
};
// The first two lines of the muscle record at `m`.
__device__ MusclePull load_pull(const float* __restrict__ m) {
    const float4* q = reinterpret_cast<const float4*>(m);
    const float4 r0 = q[0], r1 = q[1];
    const unsigned nodes = __float_as_uint(r0.x);
    MusclePull u;
    u.a0 = nodes & 255u;
    u.a1 = (nodes >> 8u) & 255u;
    u.b0 = (nodes >> 16u) & 255u;
    u.b1 = nodes >> 24u;
    u.anchor_a = r0.y;
    u.anchor_b = r0.z;
    u.hill = r0.w;
    u.cap = r1.x;
    u.inv_capacity = r1.y;
    u.tendon_k = r1.z;
    u.slack = r1.w;
    return u;
}
struct MuscleRhythm {
    // `amplitude` is the stroke of one cycle in meters and `inv_period` is one
    // over the period in seconds. `stiffness` is a factor on the muscle's
    // demand. A touchdown of node `sensor` restarts the rhythm at the phase
    // `reset`.
    float amplitude, inv_period, phase, duty, inv_duty, inv_complement, stiffness, reset;
    // The sensor node, or `NONE` for a muscle that has no sensor.
    unsigned sensor;
};
// The last three lines of the muscle record at `m`.
__device__ MuscleRhythm load_rhythm(const float* __restrict__ m) {
    const float4* q = reinterpret_cast<const float4*>(m);
    const float4 r2 = q[2], r3 = q[3], r4 = q[4];
    MuscleRhythm u;
    u.amplitude = r2.x;
    u.inv_period = r2.y;
    u.phase = r2.z;
    u.duty = r2.w;
    u.inv_duty = r3.x;
    u.inv_complement = r3.y;
    u.stiffness = r3.z;
    u.reset = r3.w;
    u.sensor = __float_as_uint(r4.x);
    return u;
}

// Each muscle's demand for the step that starts at time `t`: the speed at
// which its rhythm shortens it over the step, times a quarter of its
// stiffness, and never below 0. A limp muscle has no demand.
__device__ void muscle_demands(Body& b, const float* __restrict__ muscles, float t, bool limp) {
    for (unsigned k = 0u; k < b.muscles; k++) {
        const MuscleRhythm u = load_rhythm(muscles + k * MUSCLE_WORDS);
        const float before = rhythm(t, u.inv_period, u.phase, b.offset[k], u.duty, u.inv_duty, u.inv_complement);
        const float after = rhythm(t + DT, u.inv_period, u.phase, b.offset[k], u.duty, u.inv_duty, u.inv_complement);
        const float shortening = u.amplitude * (after - before) * RATE;
        b.demand[k] = limp ? 0.0f : fmaxf(shortening * u.stiffness * 0.25f, 0.0f);
    }
}

// Fills the force table with the pull of every muscle for one substep and
// updates the energy stores. A muscle pulls its two anchors together.
template <int MODE>
__device__ void muscle_forces(Body& b, const float* __restrict__ muscles, const Params& p) {
    for (unsigned i = 0u; i < b.nodes; i++) { fp_at<MODE>(b, i) = make_float2(0.0f, 0.0f); }
    for (unsigned k = 0u; k < b.muscles; k++) {
        const MusclePull u = load_pull(muscles + k * MUSCLE_WORDS);
        // The anchors: a share of each bone's length from its pivot.
        const unsigned a0 = u.a0, a1 = u.a1, b0 = u.b0, b1 = u.b1;
        const float ax = pos_at<MODE>(b, a0).x + (pos_at<MODE>(b, a1).x - pos_at<MODE>(b, a0).x) * u.anchor_a;
        const float ay = pos_at<MODE>(b, a0).y + (pos_at<MODE>(b, a1).y - pos_at<MODE>(b, a0).y) * u.anchor_a;
        const float avx = vel_at<MODE>(b, a0).x + (vel_at<MODE>(b, a1).x - vel_at<MODE>(b, a0).x) * u.anchor_a;
        const float avy = vel_at<MODE>(b, a0).y + (vel_at<MODE>(b, a1).y - vel_at<MODE>(b, a0).y) * u.anchor_a;
        const float bx = pos_at<MODE>(b, b0).x + (pos_at<MODE>(b, b1).x - pos_at<MODE>(b, b0).x) * u.anchor_b;
        const float by = pos_at<MODE>(b, b0).y + (pos_at<MODE>(b, b1).y - pos_at<MODE>(b, b0).y) * u.anchor_b;
        const float bvx = vel_at<MODE>(b, b0).x + (vel_at<MODE>(b, b1).x - vel_at<MODE>(b, b0).x) * u.anchor_b;
        const float bvy = vel_at<MODE>(b, b0).y + (vel_at<MODE>(b, b1).y - vel_at<MODE>(b, b0).y) * u.anchor_b;
        const float dx = bx - ax, dy = by - ay;
        const float len = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f);
        const float ex = dx / len, ey = dy / len;
        // Lengthening speed of the muscle.
        const float lengthening = (bvx - avx) * ex + (bvy - avy) * ey;
        // The step's demand, scaled by the energy store.
        float drive = b.demand[k] * b.energy[k];
        // Hill: the active pull falls with the shortening speed.
        if (u.hill > 0.0f) {
            drive *= clampf(1.0f + lengthening * u.hill, 0.0f, 1.0f);
        }
        // A light damper on the change of length adds to the drive. The sum
        // stays within the force cap.
        const float active = clampf(drive + lengthening * 0.15f, -u.cap, u.cap);
        // Only active shortening is charged to the store, which recovers.
        const float work = fminf(drive, u.cap) * fmaxf(-lengthening, 0.0f) * H;
        b.energy[k] = clampf(b.energy[k] - work * u.inv_capacity * p.inv_muscle_energy
            + MUSCLE_RECOVERY * p.muscle_recovery * H * (1.0f - b.energy[k]), 0.0f, 1.0f);
        // The tendon pulls back once the muscle is stretched past its slack.
        const float tendon = u.tendon_k * fmaxf(len - u.slack, 0.0f);
        const float pull = active + tendon;
#if RECORD
        b.pull[k] = pull;
#endif
        // The pull acts along the muscle. Each anchor's share goes to the two
        // nodes of its bone by where the anchor sits.
        const float fx = ex * pull, fy = ey * pull;
        fp_at<MODE>(b, a0).x += fx * (1.0f - u.anchor_a);
        fp_at<MODE>(b, a0).y += fy * (1.0f - u.anchor_a);
        fp_at<MODE>(b, a1).x += fx * u.anchor_a;
        fp_at<MODE>(b, a1).y += fy * u.anchor_a;
        fp_at<MODE>(b, b0).x -= fx * (1.0f - u.anchor_b);
        fp_at<MODE>(b, b0).y -= fy * (1.0f - u.anchor_b);
        fp_at<MODE>(b, b1).x -= fx * u.anchor_b;
        fp_at<MODE>(b, b1).y -= fy * u.anchor_b;
    }
}

// The world's force on node i: gravity, wind, mud, brambles and buoyancy.
template <int MODE>
__device__ float2 node_force(Body& b, unsigned i, const Params& p) {
    const float m = b.mass[i];
    float fx = 0.0f, fy = -p.gravity * m;
#if WIND
    fx += p.wind * m;
#endif
#if MUD || BRAMBLES
    // How far the node's surface is above the ground, along the ground's
    // normal. It is negative when the node is sunk in.
    const float2 g = ground(pos_at<MODE>(b, i).x, b.amp, b.qphase, p);
    const float dry = (pos_at<MODE>(b, i).y - g.x) / sqrtf(1.0f + g.y * g.y) - b.radius[i];
#endif
#if MUD
    // Mud drags the horizontal velocity of a sunk node. The drag grows with
    // the sink depth, up to `p.mud`, and is `MUD_DRAG` per second at
    // `MUD_FULL_DEPTH`.
    fx -= m * MUD_DRAG * (clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH)) * vel_at<MODE>(b, i).x;
#endif
#if BRAMBLES
    // Brambles hold back every node but the feet while its surface is within
    // `BRAMBLE_REACH` of the ground: a drag against its horizontal velocity.
    if (((b.feet >> i) & 1u) == 0u && dry < BRAMBLE_REACH) {
        fx -= m * p.brambles * vel_at<MODE>(b, i).x;
    }
#endif
#if WATER
    {
        // Buoyancy is `WATER_BUOYANCY` of the node's weight times the share of
        // its diameter under the water line `p.water`.
        const float wet = clampf((p.water - (pos_at<MODE>(b, i).y - b.radius[i])) / (2.0f * b.radius[i]), 0.0f, 1.0f);
        fy += WATER_BUOYANCY * m * p.gravity * wet;
    }
#endif
    return make_float2(fx, fy);
}

// Air and water drag on each bone at its midpoint, shared by its two nodes,
// never more than half the bone's speed in one substep. The air drag grows
// with the bone's length, width and speed. The water drag also grows with the
// share of the bone under water. It acts across the bone in full and along it
// at `WATER_ALONG` of that.
template <int MODE>
__device__ void bone_drag(Body& b, const Params& p) {
    for (unsigned j = 0u; j < b.bones; j++) {
        const unsigned i0 = b.pivot[j], i1 = j + 1u;
        const float wx = 0.5f * (vel_at<MODE>(b, i0).x + vel_at<MODE>(b, i1).x), wy = 0.5f * (vel_at<MODE>(b, i0).y + vel_at<MODE>(b, i1).y);
        const float speed = sqrtf(wx * wx + wy * wy);
        const float width = b.radius[i0] + b.radius[i1];
        const float bone_mass = b.mass[i0] + b.mass[i1];
        const float limit = 0.5f * bone_mass * INV_H;
        const float k_air = fminf(AIR_DRAG * b.length[j] * width * speed, limit);
        float dfx = -wx * k_air, dfy = -wy * k_air;
#if WATER
        {
            // The wet share is the mean of the two nodes'. The vector (ax, ay)
            // points along the bone. The midpoint velocity splits into the part
            // along it (lx, ly) and the part across it (sx, sy).
            const float wet0 = clampf((p.water - (pos_at<MODE>(b, i0).y - b.radius[i0])) / (2.0f * b.radius[i0]), 0.0f, 1.0f);
            const float wet1 = clampf((p.water - (pos_at<MODE>(b, i1).y - b.radius[i1])) / (2.0f * b.radius[i1]), 0.0f, 1.0f);
            const float wet = 0.5f * (wet0 + wet1);
            const float ax = (pos_at<MODE>(b, i1).x - pos_at<MODE>(b, i0).x) / b.length[j], ay = (pos_at<MODE>(b, i1).y - pos_at<MODE>(b, i0).y) / b.length[j];
            const float along = wx * ax + wy * ay;
            const float lx = ax * along, ly = ay * along;
            const float sx = wx - lx, sy = wy - ly;
            const float k_water = fminf(WATER_DRAG * wet * b.length[j] * width * speed, limit);
            dfx -= sx * k_water + lx * k_water * WATER_ALONG;
            dfy -= sy * k_water + ly * k_water * WATER_ALONG;
        }
#endif
        const float s0 = 0.5f * H * b.inv_m[i0], s1 = 0.5f * H * b.inv_m[i1];
        vel_at<MODE>(b, i0).x += s0 * dfx;
        vel_at<MODE>(b, i0).y += s0 * dfy;
        vel_at<MODE>(b, i1).x += s1 * dfx;
        vel_at<MODE>(b, i1).y += s1 * dfy;
    }
}

// Moves the two nodes of every bone back to the bone's length. The move is
// shared by the inverse masses, so the lighter node moves more.
template <int MODE>
__device__ void solve_bones(Body& b) {
    for (unsigned j = 0u; j < b.bones; j++) {
        const unsigned i0 = b.pivot[j], i1 = j + 1u;
        const float dx = pos_at<MODE>(b, i1).x - pos_at<MODE>(b, i0).x, dy = pos_at<MODE>(b, i1).y - pos_at<MODE>(b, i0).y;
        const float len = fmaxf(sqrtf(dx * dx + dy * dy), 1e-9f);
        const float w0 = b.inv_m[i0], w1 = b.inv_m[i1];
        const float s = (len - b.length[j]) / ((w0 + w1) * len);
        pos_at<MODE>(b, i0).x += w0 * s * dx;
        pos_at<MODE>(b, i0).y += w0 * s * dy;
        pos_at<MODE>(b, i1).x -= w1 * s * dx;
        pos_at<MODE>(b, i1).y -= w1 * s * dy;
    }
}

// Every joint back inside its range. A joint outside its range gets one push
// along its gradients that takes back the whole error. Bone 0 and a bone with
// no parent have no joint.
template <int MODE>
__device__ void solve_joints(Body& b) {
    for (unsigned j = 1u; j < b.bones; j++) {
        if (b.parent[j] == 0xffu) { continue; }
        const Joint k = joint<MODE>(b, j);
        float error = 0.0f;
        if (k.angle < b.lo[j]) { error = k.angle - b.lo[j]; }
        if (k.angle > b.hi[j]) { error = k.angle - b.hi[j]; }
        if (error == 0.0f) { continue; }
        const float w = joint_weight(b, k);
        if (w > 0.0f) { joint_push<MODE, false>(b, k, -error / w); }
    }
}

// Node i, if inside the ground, moves out along the ground's normal, and
// friction takes back up to mu times that move of its slide over the
// substep. The moves give the contact forces of a recording. Without `GROUND`
// the world has no floor and this does nothing.
template <int MODE>
__device__ void solve_ground(Body& b, unsigned i, const Params& p) {
#if GROUND
    // The ground's normal is (nx, ny), and `dry` is the clearance of the node's
    // surface above the ground. `depth` is how far the node is inside it.
    const float2 g = ground(pos_at<MODE>(b, i).x, b.amp, b.qphase, p);
    const float secant = sqrtf(1.0f + g.y * g.y);
    const float nx = -g.y / secant, ny = 1.0f / secant;
    const float dry = (pos_at<MODE>(b, i).y - g.x) / secant - b.radius[i];
#if MUD
    // In mud the floor is `p.mud` deeper, and a node that has sunk in has a
    // larger friction budget.
    const float depth = -(dry + p.mud);
    const float sink = clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH);
    float mu = b.fric[i] * p.friction * (1.0f + MUD_GRIP * sink) * (1.0f + MUD_NORMAL * sink);
#else
    const float depth = -dry;
    float mu = b.fric[i] * p.friction;
#endif
    if (depth <= 0.0f) { return; }
#if ICE
    // Ice patches take a share of the friction away.
    mu *= 1.0f - p.patches * ice_at(pos_at<MODE>(b, i).x);
#endif
    pos_at<MODE>(b, i).x += nx * depth;
    pos_at<MODE>(b, i).y += ny * depth;
    // The slide along the ground since the substep began. Friction takes back
    // the slide, but no more than the budget.
    const float tx = ny, ty = -nx;
    const float slide = (pos_at<MODE>(b, i).x - fp_at<MODE>(b, i).x) * tx + (pos_at<MODE>(b, i).y - fp_at<MODE>(b, i).y) * ty;
    const float budget = mu * depth;
    const float back = fabsf(slide) <= budget ? slide : copysignf(budget, slide);
    pos_at<MODE>(b, i).x -= tx * back;
    pos_at<MODE>(b, i).y -= ty * back;
#if RECORD
    b.normal_force[i] += b.mass[i] * depth * INV_H * INV_H;
    b.friction_force[i] += b.mass[i] * back * INV_H * INV_H;
#endif
#endif
}

// Joint damping: each joint loses a share of its turning speed every
// substep, with equal and opposite pushes that keep the body's momentum.
template <int MODE>
__device__ void damp_joints(Body& b) {
    // The share of the turning speed that one substep takes: the substep over
    // the damping time constant, at most 1.
    const float share = fminf(H * INV_JOINT_DAMPING, 1.0f);
    for (unsigned j = 1u; j < b.bones; j++) {
        if (b.parent[j] == 0xffu) { continue; }
        const Joint k = joint<MODE>(b, j, false);
        // The joint's turning speed: the gradients times the node velocities.
        const float rate = k.g_tip.x * vel_at<MODE>(b, k.tip).x + k.g_tip.y * vel_at<MODE>(b, k.tip).y
                         + k.g_far.x * vel_at<MODE>(b, k.far).x + k.g_far.y * vel_at<MODE>(b, k.far).y
                         + k.g_hub.x * vel_at<MODE>(b, k.hub).x + k.g_hub.y * vel_at<MODE>(b, k.hub).y;
        const float w = joint_weight(b, k);
        if (w > 0.0f) { joint_push<MODE, true>(b, k, -share * rate / w); }
    }
}

// One substep: forces, prediction, constraints, velocities.
template <int MODE>
__device__ void substep(Body& b, const float* __restrict__ muscles, const Params& p) {
    // The muscles' pulls go to the force table. The drag on the bones changes
    // the velocities.
    muscle_forces<MODE>(b, muscles, p);
    bone_drag<MODE>(b, p);
    // Per node: the world's force and the muscles' pull change the velocity,
    // and `AIR` keeps a share of it. The force table takes the position at the
    // start of the substep, and then the node moves.
    for (unsigned i = 0u; i < b.nodes; i++) {
        const float2 f = node_force<MODE>(b, i, p);
        vel_at<MODE>(b, i).x += H * (f.x + fp_at<MODE>(b, i).x) * b.inv_m[i];
        vel_at<MODE>(b, i).y += H * (f.y + fp_at<MODE>(b, i).y) * b.inv_m[i];
#if AIR
        vel_at<MODE>(b, i).x *= p.air_sub;
        vel_at<MODE>(b, i).y *= p.air_sub;
#endif
        fp_at<MODE>(b, i) = pos_at<MODE>(b, i);
        pos_at<MODE>(b, i).x += H * vel_at<MODE>(b, i).x;
        pos_at<MODE>(b, i).y += H * vel_at<MODE>(b, i).y;
    }
    solve_bones<MODE>(b);
    solve_joints<MODE>(b);
    // Per node: the ground moves it out and friction takes back its slide.
    // Then its velocity is its move over the substep.
    for (unsigned i = 0u; i < b.nodes; i++) {
        solve_ground<MODE>(b, i, p);
        vel_at<MODE>(b, i).x = (pos_at<MODE>(b, i).x - fp_at<MODE>(b, i).x) * INV_H;
        vel_at<MODE>(b, i).y = (pos_at<MODE>(b, i).y - fp_at<MODE>(b, i).y) * INV_H;
    }
    damp_joints<MODE>(b);
}

// The joints forced past their range by more than `JOINT_BREAK`, as bits. Bit
// j is the joint of bone j.
template <int MODE>
__device__ unsigned long long broken_joints(Body& b) {
    unsigned long long bits = 0ull;
    for (unsigned j = 1u; j < b.bones; j++) {
        if (b.parent[j] == 0xffu) { continue; }
        const float angle = joint<MODE>(b, j).angle;
        if (angle < b.lo[j] - JOINT_BREAK || angle > b.hi[j] + JOINT_BREAK) { bits |= 1ull << j; }
    }
    return bits;
}

#if RECORD
// Writes frame `t` of a recording, `p.stride` slots of two floats. First come
// the `MAXN` node positions. Then come one pair per muscle (energy store and
// pull) and one pair per node (push of the ground and friction). The last slot
// holds the broken joints as two words of bits. See
// `creature_kernel::frame_stride`.
template <int MODE>
__device__ void record_frame(float2* __restrict__ frames, unsigned t, Body& b, const Params& p) {
    const unsigned fb = t * p.stride;
    for (unsigned i = 0u; i < b.nodes; i++) {
        frames[fb + i] = make_float2(pos_at<MODE>(b, i).x, pos_at<MODE>(b, i).y);
        frames[fb + MAXN + b.muscles + i] = make_float2(b.normal_force[i], b.friction_force[i]);
    }
    for (unsigned k = 0u; k < b.muscles; k++) {
        frames[fb + MAXN + k] = make_float2(b.energy[k], b.pull[k]);
    }
    const unsigned long long broken = broken_joints<MODE>(b);
    frames[fb + p.stride - 1u] = make_float2(__uint_as_float((unsigned)broken), __uint_as_float((unsigned)(broken >> 32)));
}
#endif

// The behavior totals of a trial, kept between steps.
struct Tally {
    // Node bits, node i in bit i: the nodes that have touched the ground, those
    // of them that later lifted clear of it, and the nodes on the ground after
    // the last step.
    unsigned contact_bits, lift_bits, ground_bits;
    // The running average of the head's acceleration (m/s^2).
    float head_shake;
    // The head's position and velocity at the last 60 Hz sample.
    float2 head_at, head_vel;
    // The rung trace while the trial runs. `rung_x` is the distance half a
    // second before a rung, as float bits. `rung_speed` holds the speeds at R1
    // and R2 as two halves. `rung_early` and `rung_late` each hold the share of
    // nodes that touched the ground and the mean muscle energy store, at R1 and
    // at R2.
    unsigned rung_x, rung_speed, rung_early, rung_late;
    // The bits of the end code that the rungs set: the rung that stopped the
    // trial and the cadence band at each rung.
    unsigned rung_bits;
};


// What a lane keeps of the creature it runs, between steps.
struct Lane {
    // Whether the lane has a creature.
    bool live;
    // The creature's index in the batch, its next step, its flags and the float
    // offset of its muscle records. The flags are those of `rungs`: 1 audit, 2
    // exempt, 4 and 8 exempt from R1 and R2, 16 young and 32 reshaped.
    unsigned cidx, step, flags, mus_off;
    // The rhythm period of the first muscle, rounded to a half. It is the sixth
    // rung feature.
    float period_f;
    // The muscles are limp, because the score of a recording is final.
    bool limp;
    // The result and the totals so far.
    Result mt;
    Tally tl;
#if RECORD
    // A recording scores until its score is final. Then `kept` holds the result
    // and the recording plays on.
    bool scoring;
    Result kept;
#endif
};

// Creatures a warp claims from the wave's counter at a time. It may not be
// more than 32, because `advance` tests one creature of a chunk per lane.
#define CHUNK 32u

// Whether creature `cidx` has more than NS nodes, so that it runs in mode 1.
__device__ __forceinline__ bool is_big(const uint4* __restrict__ heads, unsigned cidx) {
    return (heads[2u * cidx].x & 255u) > NS;
}

// Starts creature `cidx` on this lane: its constants, its start pose and its
// totals.
template <int MODE>
__device__ __forceinline__ void begin_creature(
    Body& b, Lane& ln, unsigned cidx,
    const unsigned* __restrict__ records,
    const uint4* __restrict__ heads,
    const Params& p
#if RECORD
    , float2* __restrict__ frames
#endif
    ) {
    ln.cidx = cidx;
    // The two head words of the creature (`kernel::WavePack::heads`). The first
    // holds the node count, the muscle count, the quake hash and the word
    // offset of its records. The second holds the float offset of its muscle
    // records, its total mass, one over it, and the rhythm period of its first
    // muscle as a half with the creature flags in the high 16 bits.
    const uint4 h0 = heads[2u * cidx];
    const uint4 h1 = heads[2u * cidx + 1u];
    b.nodes = h0.x & 255u;
    b.bones = b.nodes - 1u;
    b.muscles = h0.y;
    const unsigned* rec = records + h0.w;
    ln.mus_off = h1.x;
    b.total_mass = __uint_as_float(h1.y);
    b.inv_mass = __uint_as_float(h1.z);
    ln.flags = h1.w >> 16u;
    ln.period_f = h2f(h1.w & 0xffffu);
#if QUAKE
    // The quake hash gives the creature its own bumps. Its low 16 bits are the
    // phase in wave turns, and its high 16 bits scale the height by 0.6 to 1.4
    // (`physics::quake_phase` and `physics::quake_scale`).
    b.qphase = (float)(h0.z & 0xffffu) * (1.0f / 65536.0f);
    b.amp = p.terrain + p.quake * (0.6f + (float)((h0.z >> 16u) & 0xffffu) * (0.8f / 65536.0f));
#else
    b.qphase = 0.0f;
    b.amp = p.terrain;
#endif
    // A node record is the mass, the radius, the friction coefficient, the
    // start position and the foot flag. The nodes start at rest.
    b.feet = 0u;
    for (unsigned i = 0u; i < b.nodes; i++) {
        const unsigned* r = rec + i * NODE_WORDS;
        b.mass[i] = __uint_as_float(r[0]);
        b.inv_m[i] = 1.0f / b.mass[i];
        b.radius[i] = __uint_as_float(r[1]);
        b.fric[i] = __uint_as_float(r[2]);
        pos_at<MODE>(b, i).x = __uint_as_float(r[3]);
        pos_at<MODE>(b, i).y = __uint_as_float(r[4]);
        b.feet |= (r[5] & 1u) << i;
        vel_at<MODE>(b, i).x = 0.0f; vel_at<MODE>(b, i).y = 0.0f;
        b.normal_force[i] = 0.0f; b.friction_force[i] = 0.0f;
    }
    // A bone record is the pivot node, the length, the parent bone and the
    // joint range, low and high. Its sixth word is spare.
    const unsigned* bone_rec = rec + b.nodes * NODE_WORDS;
    for (unsigned j = 0u; j < b.bones; j++) {
        const unsigned* r = bone_rec + j * BONE_WORDS;
        b.pivot[j] = (unsigned char)r[0];
        b.length[j] = __uint_as_float(r[1]);
        b.parent[j] = (unsigned char)(r[2] == NONE ? 0xffu : r[2]);
        b.lo[j] = __uint_as_float(r[3]);
        b.hi[j] = __uint_as_float(r[4]);
    }
    // Every muscle starts rested, with no offset and no demand.
    for (unsigned k = 0u; k < b.muscles; k++) {
        b.energy[k] = 1.0f; b.offset[k] = 0.0f; b.demand[k] = 0.0f; b.pull[k] = 0.0f;
    }
#if RECORD
    // The settling frames show the start pose.
    for (unsigned t = 0u; t <= SETTLE; t++) { record_frame<MODE>(frames, t, b, p); }
    ln.scoring = true;
#endif
    // The result starts at 0, except that `vertical_oscillation` and
    // `gait_frequency` start at 1e20 and -1e20. They hold the lowest and the
    // highest mean node height, so the first step sets both.
    Result& mt = ln.mt;
    mt.fitness = 0.0f; mt.ground_contact = 0.0f; mt.vertical_oscillation = 1e20f; mt.gait_frequency = -1e20f;
    mt.previous_center_y = 0.0f; mt.vertical_extremum = 0.0f; mt.vertical_trend = 0.0f; mt.gait_turns = 0.0f;
    mt.height_sum = 0.0f; mt.contact_lo = 0.0f; mt.contact_hi = 0.0f; mt.lift_lo = 0.0f; mt.lift_hi = 0.0f;
    mt.ground_lo = 0.0f; mt.ground_hi = 0.0f; mt.fall_time = 0.0f; mt.head_shake = 0.0f; mt.screen_x = 0.0f;
    mt.screened = 0.0f;
    Tally& tl = ln.tl;
    tl.contact_bits = 0u; tl.lift_bits = 0u; tl.ground_bits = 0u; tl.head_shake = 0.0f;
    tl.head_at = pos_at<MODE>(b, 0u); tl.head_vel = make_float2(0.0f, 0.0f);
    tl.rung_x = 0u; tl.rung_speed = 0u; tl.rung_early = 0u; tl.rung_late = 0u; tl.rung_bits = 0u;
    ln.limp = false;
    ln.step = 0u;
}

// One step of the creature on this lane: its muscles' demand, the substeps,
// and what the step leaves (distance, height, contacts, the rungs). When the
// trial ends, its result is stored and the lane is free. A recording plays on
// to its last step with limp muscles and stores the result then.
template <int MODE>
__device__ __forceinline__ void run_step(
    Body& b, Lane& ln,
    const float* __restrict__ muscles,
    Result* __restrict__ results,
    const Params& p
#if RECORD
    , float2* __restrict__ frames
#endif
    ) {
    const unsigned step = ln.step;
    const float* mus = muscles + ln.mus_off;
    Result& mt = ln.mt;
    Tally& tl = ln.tl;
    const unsigned flags = ln.flags;
    // The steps that end at 1, 2.5, 5 and 10 s, the rungs R1 to R4. `half_s` is
    // half a second in steps.
    const unsigned rung1 = (unsigned)(RATE) - 1u, rung2 = (unsigned)(2.5f * RATE) - 1u;
    const unsigned rung3 = (unsigned)(5.0f * RATE) - 1u, rung4 = (unsigned)(10.0f * RATE) - 1u;
    const unsigned half_s = (unsigned)(0.5f * RATE);
    const float t_now = (float)step * DT;
#if RECORD
    for (unsigned i = 0u; i < b.nodes; i++) { b.normal_force[i] = 0.0f; b.friction_force[i] = 0.0f; }
#endif
    muscle_demands(b, mus, t_now, ln.limp);
    for (unsigned s = 0u; s < SUBSTEPS; s++) {
        substep<MODE>(b, mus, p);
    }
#if RECORD
    for (unsigned i = 0u; i < b.nodes; i++) {
        b.normal_force[i] *= 1.0f / SUBSTEPS;
        b.friction_force[i] *= 1.0f / SUBSTEPS;
    }
#endif

    // What the step leaves. `failed` is a position that is not finite or lies
    // beyond 1e6 m. `center_y` is the mean node height and `com_x` the distance
    // of the center of mass, which becomes the fitness. `low` and `high` are
    // the bottom of the lowest node and the top of the highest. `touching` has
    // the nodes at most `CONTACT_SLACK` above their resting height on the
    // ground, and `lifted` the nodes more than `LIFT_CLEARANCE` above it.
    bool failed = false;
    float center_y = 0.0f, com_x = 0.0f, low = 1e20f, high = -1e20f;
    unsigned touching = 0u, lifted = 0u;
    for (unsigned i = 0u; i < b.nodes; i++) {
        failed = failed || !(fabsf(pos_at<MODE>(b, i).x) <= 1e6f && fabsf(pos_at<MODE>(b, i).y) <= 1e6f);
        center_y += pos_at<MODE>(b, i).y;
        com_x += b.mass[i] * pos_at<MODE>(b, i).x;
        low = fminf(low, pos_at<MODE>(b, i).y - b.radius[i]);
        high = fmaxf(high, pos_at<MODE>(b, i).y + b.radius[i]);
#if GROUND
        const float2 g = ground(pos_at<MODE>(b, i).x, b.amp, b.qphase, p);
        const float floor_y = g.x + b.radius[i] * sqrtf(1.0f + g.y * g.y);
        if (pos_at<MODE>(b, i).y <= floor_y + CONTACT_SLACK) { touching |= 1u << i; }
        if (pos_at<MODE>(b, i).y > floor_y + LIFT_CLEARANCE) { lifted |= 1u << i; }
#endif
    }
    center_y /= (float)b.nodes;
    com_x *= b.inv_mass;
    tl.contact_bits |= touching;
    tl.lift_bits |= tl.contact_bits & lifted;
    // The nodes that touch the ground now and did not after the last step.
    const unsigned down = touching & ~tl.ground_bits;
    tl.ground_bits = touching;
    // Touchdowns restart the rhythm of the muscles that sense them.
    if (down != 0u && step > 0u) {
        const float next = t_now + DT;
        for (unsigned k = 0u; k < b.muscles; k++) {
            const MuscleRhythm u = load_rhythm(mus + k * MUSCLE_WORDS);
            if (u.sensor != NONE && ((down >> u.sensor) & 1u)) {
                b.offset[k] = fracf(u.reset - (next * u.inv_period + u.phase));
            }
        }
    }
    // Head shake: the head's acceleration averaged over about
    // HEAD_SHAKE_WINDOW. The head's velocity is its move over each
    // 1/60 s, so the measure is the same at every step rate. The average
    // starts when the trial is HEAD_SHAKE_WINDOW old.
    if ((step + 1u) % HEAD_SAMPLE == 0u) {
        const float2 v = make_float2((pos_at<MODE>(b, 0u).x - tl.head_at.x) * 60.0f, (pos_at<MODE>(b, 0u).y - tl.head_at.y) * 60.0f);
        const float ax = (v.x - tl.head_vel.x) * 60.0f, ay = (v.y - tl.head_vel.y) * 60.0f;
        if (t_now >= HEAD_SHAKE_WINDOW) {
            const float accel = sqrtf(ax * ax + ay * ay);
            tl.head_shake += (accel - tl.head_shake) * fminf(1.0f / (HEAD_SHAKE_WINDOW * 60.0f), 1.0f);
        }
        tl.head_at = pos_at<MODE>(b, 0u);
        tl.head_vel = v;
    }
    const bool broken = broken_joints<MODE>(b) != 0ull;
    // In a recording the scoring below stops when the score is final, and the
    // frames go on.
#if RECORD
    if (ln.scoring) {
#endif
    mt.contact_lo = __uint_as_float(tl.contact_bits);
    mt.lift_lo = __uint_as_float(tl.lift_bits);
    mt.ground_lo = __uint_as_float(tl.ground_bits);
    mt.head_shake = tl.head_shake;
    // A fall: the head is below the other end of the neck, a joint is broken,
    // the head shakes past HEAD_SHAKE_LIMIT, or the trial failed. The distance
    // at that moment is the fitness, or -1e20 for a failed trial.
    const bool fell = pos_at<MODE>(b, 0u).y < pos_at<MODE>(b, 1u).y || broken || tl.head_shake > HEAD_SHAKE_LIMIT || failed;
    bool ended = false;
    if (fell) {
        mt.fall_time = t_now + DT;
        mt.fitness = failed ? -1e20f : com_x;
        // A fall at or before the screen step gives its distance as the
        // distance at the screen.
        if (step <= p.screen_step) { mt.screen_x = mt.fitness; }
        ended = true;
    }
    mt.ground_contact += (float)__popc(touching);
    mt.height_sum += high - low;
    mt.vertical_oscillation = fminf(mt.vertical_oscillation, center_y);
    mt.gait_frequency = fmaxf(mt.gait_frequency, center_y);
    // Gait turns, counted on every SAMPLE-th step. `vertical_trend` is 0 until
    // the mean node height has moved by more than 0.5 mm between two samples.
    // Then it is 1 while the height rises and -1 while it falls, and
    // `vertical_extremum` is the highest or lowest height since the last turn.
    // A height that turns back from it by more than 5 mm counts one turn.
    if (step == 0u) {
        mt.previous_center_y = center_y;
        mt.vertical_extremum = center_y;
        mt.vertical_trend = 0.0f;
        mt.gait_turns = 0.0f;
    } else if (step % SAMPLE == 0u) {
        const float delta = center_y - mt.previous_center_y;
        if (mt.vertical_trend == 0.0f) {
            if (fabsf(delta) > 0.0005f) {
                mt.vertical_trend = delta > 0.0f ? 1.0f : -1.0f;
                mt.vertical_extremum = center_y;
            }
        } else if (mt.vertical_trend > 0.0f) {
            if (center_y > mt.vertical_extremum) {
                mt.vertical_extremum = center_y;
            } else if (mt.vertical_extremum - center_y > 0.005f) {
                mt.gait_turns += 1.0f;
                mt.vertical_trend = -1.0f;
                mt.vertical_extremum = center_y;
            }
        } else {
            if (center_y < mt.vertical_extremum) {
                mt.vertical_extremum = center_y;
            } else if (center_y - mt.vertical_extremum > 0.005f) {
                mt.gait_turns += 1.0f;
                mt.vertical_trend = 1.0f;
                mt.vertical_extremum = center_y;
            }
        }
        mt.previous_center_y = center_y;
    }
    // The rung trace (`creature_kernel::RungTrace`). At 1 and 2.5 s (R1 and R2)
    // it takes the distance, the speed over the last half second, the share of
    // nodes that touched the ground, the head shake and the mean muscle energy
    // store. At 5 and 10 s (R3 and R4) it takes the distance. Each value is an
    // fp16, two to a word. Some words wait in result fields that the host reads
    // for nothing else, and the end of the trial moves the rest there.
    // Half a second before R1 and R2 the distance is noted for the speed.
    if (step + half_s == rung1 || step + half_s == rung2) { tl.rung_x = __float_as_uint(com_x); }
    if (step == rung1 || step == rung2) {
        const bool second = step == rung2;
        const unsigned sh = second ? 16u : 0u;
        const unsigned keep = second ? 0x0000ffffu : 0xffff0000u;
        const float speed = (com_x - __uint_as_float(tl.rung_x)) * (RATE / (float)half_s);
        const float touched = (float)__popc(tl.contact_bits) / (float)b.nodes;
        float en_mean = 0.0f;
        for (unsigned k = 0u; k < b.muscles; k++) { en_mean += b.energy[k]; }
        en_mean /= (float)max(b.muscles, 1u);
        mt.contact_hi = __uint_as_float((__float_as_uint(mt.contact_hi) & keep) | (f2h(com_x) << sh));
        mt.ground_hi = __uint_as_float((__float_as_uint(mt.ground_hi) & keep) | (f2h(tl.head_shake) << sh));
        tl.rung_speed = (tl.rung_speed & keep) | (f2h(speed) << sh);
        const unsigned pair = f2h(touched) | (f2h(en_mean) << 16u);
        if (second) { tl.rung_late = pair; } else { tl.rung_early = pair; }
        // The cadence band: the gait frequency so far in 8 bands of 0.75 Hz
        // (`RungTrace::BAND_COUNT`), the last one open. The audit lane files
        // this trial under it.
        const unsigned r = second ? 1u : 0u;
        const float gait_now = mt.gait_turns * (0.5f / (t_now + DT));
        const unsigned band = min((unsigned)(gait_now * (8.0f / 6.0f)), 7u);
        tl.rung_bits |= band << (8u + 3u * r);
        // The rule does not apply to an audit creature (flag 1), an exempt one
        // (flag 2) or one that is exempt from this rung (flag 4 for R1, flag 8
        // for R2). A creature that has already ended is not stopped again.
        if (!ended && (flags & (second ? 0xbu : 0x7u)) == 0u) {
            const RungParams rp = second ? p.r2 : p.r1;
            // The features, rounded to half precision as the host rounds them
            // (`rungs::features`).
            const float fv[6] = {h2f(f2h(com_x)), h2f(f2h(speed)), h2f(f2h(touched)),
                                 h2f(f2h(tl.head_shake)), h2f(f2h(en_mean)), ln.period_f};
            float score = 0.0f;
            bool finite = true;
            for (int i = 0; i < 6; i++) {
                score = fmaf(rp.w[i], fv[i], score);
                finite = finite && isfinite(fv[i]);
            }
            // The rule stops the creature when all features are finite, the
            // score is below the bias and its band is on. The creature keeps
            // its distance, and `rung_bits` notes the rung, 1 for R1 and 2 for
            // R2.
            if (finite && score < rp.bias && ((rp.off >> band) & 1u) == 0u) {
                mt.screened = t_now + DT;
                mt.screen_x = com_x;
                mt.fitness = com_x;
                tl.rung_bits |= (r + 1u) << 6u;
                ended = true;
            }
        }
    }
    // R3 and R4 note the distance and stop nothing.
    if (step == rung3 || step == rung4) {
        const bool second = step == rung4;
        const unsigned keep = second ? 0x0000ffffu : 0xffff0000u;
        mt.lift_hi = __uint_as_float((__float_as_uint(mt.lift_hi) & keep) | (f2h(com_x) << (second ? 16u : 0u)));
    }
    // The early screen (`physics::Screen`): below the bar at the screen step,
    // the trial stops. The bar is `p.screen_bar_reshaped` for flag 32,
    // `p.screen_bar_young` for flag 16 and `p.screen_bar` for the rest. An
    // audit creature is never stopped.
    if (step == p.screen_step && !ended) {
        mt.screen_x = com_x;
        const float bar = (flags & 32u) != 0u ? p.screen_bar_reshaped
                        : (flags & 16u) != 0u ? p.screen_bar_young : p.screen_bar;
        if (com_x < bar && (flags & 1u) == 0u) {
            mt.screened = t_now + DT;
            mt.fitness = com_x;
            ended = true;
        }
    }
    // The trial ends at a stop, at a fall or at its last step. Then
    // `vertical_oscillation` becomes the range of the mean node height and
    // `gait_frequency` half the turns per second.
    bool last = false;
    if (ended) {
        mt.vertical_oscillation = fmaxf(mt.gait_frequency - mt.vertical_oscillation, 0.0f);
        mt.gait_frequency = mt.gait_turns * 0.5f / (t_now + DT);
    } else if (step + 1u == p.steps) {
        mt.fitness = com_x;
        mt.vertical_oscillation = fmaxf(mt.gait_frequency - mt.vertical_oscillation, 0.0f);
        mt.gait_frequency = mt.gait_turns * 0.5f / fmaxf((float)(step + 1u) * DT, DT);
        last = true;
    }
    if (ended || last) {
        // A rung the trial did not reach holds the final distance.
        const unsigned fin = f2h(mt.fitness);
        unsigned d = __float_as_uint(mt.contact_hi);
        if (step < rung1) { d = (d & 0xffff0000u) | fin; }
        if (step < rung2) { d = (d & 0x0000ffffu) | (fin << 16u); }
        mt.contact_hi = __uint_as_float(d);
        d = __float_as_uint(mt.lift_hi);
        if (step < rung3) { d = (d & 0xffff0000u) | fin; }
        if (step < rung4) { d = (d & 0x0000ffffu) | (fin << 16u); }
        mt.lift_hi = __uint_as_float(d);
        // The end code (`RungTrace::code`): bits 0 and 1 for a stop by the
        // screen or a rung, bit 4 for a fall, bit 5 for a failed trial, bits 6
        // to 13 from `rung_bits` and bit 14 for an audit creature.
        const unsigned code = (fell ? 16u : 0u) | (failed ? 32u : 0u) | (mt.screened > 0.0f ? 3u : 0u)
            | (tl.rung_bits & 0x3fc0u) | ((flags & 1u) << 14u);
        // The gait counters are finished, so their words carry the rest of the
        // trace now: the speeds, the features at R1 and at R2, and the head
        // shakes. `ground_hi` takes the end code and the steps run.
        mt.previous_center_y = __uint_as_float(tl.rung_speed);
        mt.vertical_extremum = __uint_as_float(tl.rung_early);
        mt.vertical_trend = __uint_as_float(tl.rung_late);
        mt.gait_turns = mt.ground_hi;
        mt.ground_hi = __uint_as_float(code | (min(step + 1u, 65535u) << 16u));
#if RECORD
        // A recording plays the whole trial; after the score is
        // final the muscles go limp.
        ln.kept = mt;
        ln.scoring = false;
        ln.limp = true;
#else
        results[ln.cidx] = mt;
        ln.live = false;
        return;
#endif
    }
#if RECORD
    }
    // The frame after this step comes after the SETTLE + 1 frames of the start
    // pose. The recording ends at the last step and stores the result it kept.
    record_frame<MODE>(frames, SETTLE + step + 1u, b, p);
    if (step + 1u >= p.steps) {
        results[ln.cidx] = ln.kept;
        ln.live = false;
        return;
    }
#endif
    ln.step = step + 1u;
}

// One thread per creature: runs its trial and tallies its behavior. A warp
// claims runs of CHUNK creatures from the wave's counter, and its lanes take
// creatures from the claim as they free up. The host sorts creatures by
// their muscles and then their nodes, so a claim holds alike bodies. A warp
// runs bodies of one class at a time: mode 0 for those of at most NS nodes
// and mode 1, with half its lanes, for the others.
//
// The arguments are those of the host's launch (`cuda_engine`). `records`,
// `muscles` and `heads` hold the node and bone records, the muscle records and
// the two head words of the batch (`kernel::WavePack`). `unused` is a word
// that the kernel does not read. `results` has one `Result` for each creature
// of the batch. The wave is the `p.count` creatures from creature `p.base`,
// and `counter` counts from 0 the creatures that the warps have claimed. A
// recording kernel (RECORD) also gets `frames`, where it writes every frame of
// its one creature. The launch bounds ask the compiler to fit `MIN_BLOCKS`
// blocks of `BLOCK` threads on a multiprocessor.
extern "C" __global__ void __launch_bounds__(BLOCK, MIN_BLOCKS) advance(
    const unsigned* __restrict__ records,
    const float* __restrict__ muscles,
    const unsigned* __restrict__ unused,
    const uint4* __restrict__ heads,
    Result* __restrict__ results,
    unsigned* __restrict__ counter,
    const Params p
#if RECORD
    , float2* __restrict__ frames
#endif
    ) {
    // All 32 lanes of a warp, for the warp votes. `lane` is this thread's place
    // in its warp, and `below` is the mask of the lanes before it.
    const unsigned FULL = 0xffffffffu;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned below = (1u << lane) - 1u;
    Body b;
    Lane ln;
    ln.live = false;
    // The warp's claim: `win` holds creatures of one class that no lane has
    // taken yet, `chunk` creatures that it has claimed and not yet sorted into
    // classes. `mode` is the class of the window and of every running lane.
    unsigned win_cur = 0u, win_end = 0u, chunk_cur = 0u, chunk_end = 0u;
    bool claimed_all = false;
    int mode = 0;
    for (;;) {
        __syncwarp();
        // Free lanes take creatures from the window. In mode 0 all 32 lanes are
        // usable. In mode 1 only the first 16 are, because a creature there
        // also uses the columns of the other 16 lanes, for its odd nodes.
        unsigned usable = mode == 0 ? FULL : 0xffffu;
        unsigned need = ~__ballot_sync(FULL, ln.live) & usable;
        unsigned fresh = 0u;
        unsigned mine_at = 0u;
        while (need != 0u) {
            if (win_cur < win_end) {
                // The free lanes, in lane order, take the next creatures of
                // the window.
                const unsigned rank = (unsigned)__popc(need & below);
                const unsigned take = min(win_end - win_cur, (unsigned)__popc(need));
                const bool mine = ((need >> lane) & 1u) != 0u && rank < take;
                if (mine) { mine_at = win_cur + rank; }
                const unsigned served = __ballot_sync(FULL, mine);
                fresh |= served;
                need &= ~served;
                win_cur += take;
                continue;
            }
            if (chunk_cur >= chunk_end) {
                // The chunk is used up. Lane 0 claims the next CHUNK creatures
                // of the wave, and the claims end when the counter is past it.
                if (claimed_all) { break; }
                unsigned c0 = 0u;
                if (lane == 0u) { c0 = atomicAdd(counter, CHUNK); }
                c0 = __shfl_sync(FULL, c0, 0);
                if (c0 >= p.count) { claimed_all = true; break; }
                chunk_cur = c0;
                chunk_end = min(c0 + CHUNK, p.count);
            }
            // The run of one class that the chunk starts with.
            const bool big = is_big(heads, p.base + chunk_cur);
            const unsigned at = chunk_cur + lane;
            const unsigned other = __ballot_sync(FULL, at < chunk_end && is_big(heads, p.base + at) != big);
            const unsigned run_end = other != 0u ? chunk_cur + (unsigned)(__ffs(other) - 1) : chunk_end;
            // Running lanes must finish before the warp changes class.
            const bool running = (__ballot_sync(FULL, ln.live) | fresh) != 0u;
            if ((mode == 1) != big) {
                if (running) { break; }
                mode = big ? 1 : 0;
                usable = mode == 0 ? FULL : 0xffffu;
                need = usable;
            }
            win_cur = chunk_cur;
            win_end = run_end;
            chunk_cur = run_end;
        }
        // A lane that got a creature starts it.
        if (((fresh >> lane) & 1u) != 0u) {
            ln.live = true;
            if (mode == 0) {
                begin_creature<0>(b, ln, p.base + mine_at, records, heads, p
#if RECORD
                    , frames
#endif
                    );
            } else {
                begin_creature<1>(b, ln, p.base + mine_at, records, heads, p
#if RECORD
                    , frames
#endif
                    );
            }
        }
        // No lane has a creature and none is left to claim: the warp is done.
        if (__ballot_sync(FULL, ln.live) == 0u) { break; }
        // Every live lane runs one step.
        if (ln.live) {
            if (mode == 0) {
                run_step<0>(b, ln, muscles, results, p
#if RECORD
                    , frames
#endif
                    );
            } else {
                run_step<1>(b, ln, muscles, results, p
#if RECORD
                    , frames
#endif
                    );
            }
        }
    }
}
