// The creature physics: one GPU thread simulates one creature from its start
// pose to the end of its trial. src/warp_kernel.rs packs the creatures and
// writes the #defines in front of this text.
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
// Defines: RATE (steps per second), SUBSTEPS, SETTLE, SAMPLE, MAXN, MAXM,
// BLOCK, MIN_BLOCKS, RECORD, the world flags (GROUND, TERRAIN, SLOPE, GAPS,
// HURDLES, QUAKE, MUD, WATER, ICE, WIND, AIR, BRAMBLES) and the physics
// constants.

#define DT (1.0f / RATE)
#define H (1.0f / (RATE * SUBSTEPS))
#define INV_H (RATE * SUBSTEPS)
#define PI_F 3.14159265359f
#define TAU_F 6.28318530718f
#define NONE 0xffffffffu
// Steps per 1/60 s, the clock of the head shake measure.
#define HEAD_SAMPLE ((unsigned)(RATE / 60.0f + 0.5f))
// Words of a node record and a bone record, floats of a muscle record.
#define NODE_WORDS 6u
#define BONE_WORDS 6u
#define MUSCLE_WORDS 20u

// One early rung's rule (rungs::Rung): stop when the chain of fused
// multiply-adds of the weights and the six features, from zero, is below the
// bias; `off` has bit b set when cadence band b is off.
struct RungParams {
    float w[6];
    float bias;
    unsigned off;
};
// Same layout as warp_kernel::Params.
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
// Same layout as creature_kernel::GpuResult.
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
__device__ __forceinline__ float wrapf(float a) { return a - TAU_F * floorf((a + PI_F) * (1.0f / TAU_F)); }
__device__ __forceinline__ float fracf(float x) { return x - floorf(x); }

// Half precision, as the rung trace carries values.
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

// A muscle's rhythm: 0 at its longest, 1 at its shortest. It rises over
// `duty` of the cycle and falls over the rest.
__device__ float rhythm(float t, float inv_period, float phase, float offset, float duty, float inv_duty, float inv_complement) {
    const float ph = fracf(t * inv_period + phase + offset);
    if (ph < duty) {
        return 0.5f - 0.5f * __cosf(PI_F * ph * inv_duty);
    }
    return 0.5f + 0.5f * __cosf(PI_F * (ph - duty) * inv_complement);
}

// Height and slope of the ground under x for a creature with terrain
// amplitude `amp` and quake phase `qphase`.
__device__ float2 ground(float x, float amp, float qphase, const Params& p) {
    float height = 0.0f, slope = 0.0f;
#if TERRAIN
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
    height += p.slope * x;
    slope += p.slope;
#endif
#if GAPS
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

// Ice patches: 1 on a patch, 0 between patches.
__device__ float ice_at(float x) {
    const float w = fracf(x * ICE_INV);
    const float t = fabsf(w - 0.5f) * 2.0f;
    const float s = clampf((0.7f - t) * 2.5f, 0.0f, 1.0f);
    return s * s * (3.0f - 2.0f * s);
}

// One creature's constants and state, in the thread's own memory.
struct Body {
    unsigned nodes, bones, muscles;
    float total_mass, inv_mass, amp, qphase;
    // Nodes.
    float mass[MAXN], inv_m[MAXN], radius[MAXN], fric[MAXN];
    unsigned feet;
    // Position, velocity, position at the start of the substep, force.
    float2 pos[MAXN], vel[MAXN], prev[MAXN];
    // The muscles' forces of the substep.
    float2 muscle_force[MAXN];
    // Bones: bone j joins node pivot[j] to node j + 1.
    unsigned char pivot[MAXN], parent[MAXN];
    float length[MAXN], lo[MAXN], hi[MAXN];
    // Muscles: energy store (1 rested), rhythm offset, drive of the step
    // before the store and Hill's relation, force of the last substep.
    float energy[MAXM], offset[MAXM], demand[MAXM], pull[MAXM];
    // Contact forces of the last step per node, for a recording.
    float normal_force[MAXN], friction_force[MAXN];
};

// Joint j: the relative angle of bone j to its parent bone, near the middle
// of its range, and its gradient. A joint moves three nodes: the tip of
// bone j (`tip`), the parent bone's far end (`far`) and the node they share
// (`hub`), whose gradient is minus the sum of the other two because the
// angle does not change when the body moves.
struct Joint {
    unsigned tip, far, hub;
    float2 g_tip, g_far, g_hub;
    float angle;
};
__device__ Joint joint(const Body& b, unsigned j, bool with_angle = true) {
    Joint k;
    const unsigned pj = b.parent[j];
    k.tip = j + 1u;
    k.hub = b.pivot[j];
    // A bone at the head turns against the neck, which also starts there.
    const bool at_head = k.hub == b.pivot[pj];
    k.far = at_head ? pj + 1u : b.pivot[pj];
    // u is the parent bone, v this bone, each from its pivot to its tip.
    const float2 u = make_float2(b.pos[pj + 1u].x - b.pos[b.pivot[pj]].x, b.pos[pj + 1u].y - b.pos[b.pivot[pj]].y);
    const float2 v = make_float2(b.pos[k.tip].x - b.pos[k.hub].x, b.pos[k.tip].y - b.pos[k.hub].y);
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
    // the pivot of u counts +.
    k.g_tip = make_float2(-v.y / vv, v.x / vv);
    const float2 g_u = make_float2(-u.y / uu, u.x / uu);
    k.g_far = at_head ? make_float2(-g_u.x, -g_u.y) : g_u;
    k.g_hub = make_float2(-(k.g_tip.x + k.g_far.x), -(k.g_tip.y + k.g_far.y));
    return k;
}

// The joint's stiffness denominator: the inverse masses times the squared
// gradients.
__device__ float joint_weight(const Body& b, const Joint& k) {
    return b.inv_m[k.tip] * (k.g_tip.x * k.g_tip.x + k.g_tip.y * k.g_tip.y)
         + b.inv_m[k.far] * (k.g_far.x * k.g_far.x + k.g_far.y * k.g_far.y)
         + b.inv_m[k.hub] * (k.g_hub.x * k.g_hub.x + k.g_hub.y * k.g_hub.y);
}

// Adds `lambda` times each node's inverse mass times its gradient to `field`
// (positions or velocities).
__device__ void joint_push(const Body& b, const Joint& k, float lambda, float2* field) {
    field[k.tip].x += b.inv_m[k.tip] * lambda * k.g_tip.x;
    field[k.tip].y += b.inv_m[k.tip] * lambda * k.g_tip.y;
    field[k.far].x += b.inv_m[k.far] * lambda * k.g_far.x;
    field[k.far].y += b.inv_m[k.far] * lambda * k.g_far.y;
    field[k.hub].x += b.inv_m[k.hub] * lambda * k.g_hub.x;
    field[k.hub].y += b.inv_m[k.hub] * lambda * k.g_hub.y;
}

// A muscle record is five 16-byte lines (`kernel::fill_creature`). The
// first two hold what every substep reads, the rest what only the rhythm
// reads once a step.
struct MusclePull {
    // The two bones' pivot and tip nodes.
    unsigned a0, a1, b0, b1;
    float anchor_a, anchor_b, hill, cap, inv_capacity, tendon_k, slack;
};
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
    float amplitude, inv_period, phase, duty, inv_duty, inv_complement, stiffness, reset;
    unsigned sensor;
};
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

// Each muscle's drive for the step starting at `t`: it follows the rhythm's
// shortening speed over the step, and a limp muscle has none.
__device__ void muscle_demands(Body& b, const float* __restrict__ muscles, float t, bool limp) {
    for (unsigned k = 0u; k < b.muscles; k++) {
        const MuscleRhythm u = load_rhythm(muscles + k * MUSCLE_WORDS);
        const float before = rhythm(t, u.inv_period, u.phase, b.offset[k], u.duty, u.inv_duty, u.inv_complement);
        const float after = rhythm(t + DT, u.inv_period, u.phase, b.offset[k], u.duty, u.inv_duty, u.inv_complement);
        const float shortening = u.amplitude * (after - before) * RATE;
        b.demand[k] = limp ? 0.0f : fmaxf(shortening * u.stiffness * 0.25f, 0.0f);
    }
}

// Sets the muscles' forces for one substep.
__device__ void muscle_forces(Body& b, const float* __restrict__ muscles, const Params& p) {
    for (unsigned i = 0u; i < b.nodes; i++) { b.muscle_force[i] = make_float2(0.0f, 0.0f); }
    for (unsigned k = 0u; k < b.muscles; k++) {
        const MusclePull u = load_pull(muscles + k * MUSCLE_WORDS);
        // The anchors: a share of each bone's length from its pivot.
        const unsigned a0 = u.a0, a1 = u.a1, b0 = u.b0, b1 = u.b1;
        const float ax = b.pos[a0].x + (b.pos[a1].x - b.pos[a0].x) * u.anchor_a;
        const float ay = b.pos[a0].y + (b.pos[a1].y - b.pos[a0].y) * u.anchor_a;
        const float avx = b.vel[a0].x + (b.vel[a1].x - b.vel[a0].x) * u.anchor_a;
        const float avy = b.vel[a0].y + (b.vel[a1].y - b.vel[a0].y) * u.anchor_a;
        const float bx = b.pos[b0].x + (b.pos[b1].x - b.pos[b0].x) * u.anchor_b;
        const float by = b.pos[b0].y + (b.pos[b1].y - b.pos[b0].y) * u.anchor_b;
        const float bvx = b.vel[b0].x + (b.vel[b1].x - b.vel[b0].x) * u.anchor_b;
        const float bvy = b.vel[b0].y + (b.vel[b1].y - b.vel[b0].y) * u.anchor_b;
        const float dx = bx - ax, dy = by - ay;
        const float len = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f);
        const float ex = dx / len, ey = dy / len;
        // Lengthening speed of the muscle.
        const float lengthening = (bvx - avx) * ex + (bvy - avy) * ey;
        float drive = b.demand[k] * b.energy[k];
        // Hill: the active pull falls with the shortening speed.
        if (u.hill > 0.0f) {
            drive *= clampf(1.0f + lengthening * u.hill, 0.0f, 1.0f);
        }
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
        const float fx = ex * pull, fy = ey * pull;
        b.muscle_force[a0].x += fx * (1.0f - u.anchor_a);
        b.muscle_force[a0].y += fy * (1.0f - u.anchor_a);
        b.muscle_force[a1].x += fx * u.anchor_a;
        b.muscle_force[a1].y += fy * u.anchor_a;
        b.muscle_force[b0].x -= fx * (1.0f - u.anchor_b);
        b.muscle_force[b0].y -= fy * (1.0f - u.anchor_b);
        b.muscle_force[b1].x -= fx * u.anchor_b;
        b.muscle_force[b1].y -= fy * u.anchor_b;
    }
}

// The world's force on node i: gravity, wind, mud, brambles and buoyancy.
__device__ float2 node_force(const Body& b, unsigned i, const Params& p) {
    const float m = b.mass[i];
    float fx = 0.0f, fy = -p.gravity * m;
#if WIND
    fx += p.wind * m;
#endif
#if MUD || BRAMBLES
    const float2 g = ground(b.pos[i].x, b.amp, b.qphase, p);
    const float dry = (b.pos[i].y - g.x) / sqrtf(1.0f + g.y * g.y) - b.radius[i];
#endif
#if MUD
    fx -= m * MUD_DRAG * (clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH)) * b.vel[i].x;
#endif
#if BRAMBLES
    // Brambles hold back every node but the feet while it touches the
    // ground: a drag against its velocity.
    if (((b.feet >> i) & 1u) == 0u && dry < BRAMBLE_REACH) {
        fx -= m * p.brambles * b.vel[i].x;
    }
#endif
#if WATER
    {
        const float wet = clampf((p.water - (b.pos[i].y - b.radius[i])) / (2.0f * b.radius[i]), 0.0f, 1.0f);
        fy += WATER_BUOYANCY * m * p.gravity * wet;
    }
#endif
    return make_float2(fx, fy);
}

// Air and water drag on each bone at its midpoint, shared by its two nodes,
// never more than half the bone's speed in one substep.
__device__ void bone_drag(Body& b, const Params& p) {
    for (unsigned j = 0u; j < b.bones; j++) {
        const unsigned i0 = b.pivot[j], i1 = j + 1u;
        const float wx = 0.5f * (b.vel[i0].x + b.vel[i1].x), wy = 0.5f * (b.vel[i0].y + b.vel[i1].y);
        const float speed = sqrtf(wx * wx + wy * wy);
        const float width = b.radius[i0] + b.radius[i1];
        const float bone_mass = b.mass[i0] + b.mass[i1];
        const float limit = 0.5f * bone_mass * INV_H;
        const float k_air = fminf(AIR_DRAG * b.length[j] * width * speed, limit);
        float dfx = -wx * k_air, dfy = -wy * k_air;
#if WATER
        {
            const float wet0 = clampf((p.water - (b.pos[i0].y - b.radius[i0])) / (2.0f * b.radius[i0]), 0.0f, 1.0f);
            const float wet1 = clampf((p.water - (b.pos[i1].y - b.radius[i1])) / (2.0f * b.radius[i1]), 0.0f, 1.0f);
            const float wet = 0.5f * (wet0 + wet1);
            const float ax = (b.pos[i1].x - b.pos[i0].x) / b.length[j], ay = (b.pos[i1].y - b.pos[i0].y) / b.length[j];
            const float along = wx * ax + wy * ay;
            const float lx = ax * along, ly = ay * along;
            const float sx = wx - lx, sy = wy - ly;
            const float k_water = fminf(WATER_DRAG * wet * b.length[j] * width * speed, limit);
            dfx -= sx * k_water + lx * k_water * WATER_ALONG;
            dfy -= sy * k_water + ly * k_water * WATER_ALONG;
        }
#endif
        const float s0 = 0.5f * H * b.inv_m[i0], s1 = 0.5f * H * b.inv_m[i1];
        b.vel[i0].x += s0 * dfx;
        b.vel[i0].y += s0 * dfy;
        b.vel[i1].x += s1 * dfx;
        b.vel[i1].y += s1 * dfy;
    }
}

// Every bone back to its length, the move shared by the inverse masses.
__device__ void solve_bones(Body& b) {
    for (unsigned j = 0u; j < b.bones; j++) {
        const unsigned i0 = b.pivot[j], i1 = j + 1u;
        const float dx = b.pos[i1].x - b.pos[i0].x, dy = b.pos[i1].y - b.pos[i0].y;
        const float len = fmaxf(sqrtf(dx * dx + dy * dy), 1e-9f);
        const float w0 = b.inv_m[i0], w1 = b.inv_m[i1];
        const float s = (len - b.length[j]) / ((w0 + w1) * len);
        b.pos[i0].x += w0 * s * dx;
        b.pos[i0].y += w0 * s * dy;
        b.pos[i1].x -= w1 * s * dx;
        b.pos[i1].y -= w1 * s * dy;
    }
}

// Every joint back inside its range.
__device__ void solve_joints(Body& b) {
    for (unsigned j = 1u; j < b.bones; j++) {
        if (b.parent[j] == 0xffu) { continue; }
        const Joint k = joint(b, j);
        float error = 0.0f;
        if (k.angle < b.lo[j]) { error = k.angle - b.lo[j]; }
        if (k.angle > b.hi[j]) { error = k.angle - b.hi[j]; }
        if (error == 0.0f) { continue; }
        const float w = joint_weight(b, k);
        if (w > 0.0f) { joint_push(b, k, -error / w, b.pos); }
    }
}

// Node i, if inside the ground, moves out along the ground's normal, and
// friction takes back up to mu times that move of its slide over the
// substep. The moves give the contact forces of a recording.
__device__ void solve_ground(Body& b, unsigned i, const Params& p) {
#if GROUND
    const float2 g = ground(b.pos[i].x, b.amp, b.qphase, p);
    const float secant = sqrtf(1.0f + g.y * g.y);
    const float nx = -g.y / secant, ny = 1.0f / secant;
    const float dry = (b.pos[i].y - g.x) / secant - b.radius[i];
#if MUD
    const float depth = -(dry + p.mud);
    const float sink = clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH);
    float mu = b.fric[i] * p.friction * (1.0f + MUD_GRIP * sink) * (1.0f + MUD_NORMAL * sink);
#else
    const float depth = -dry;
    float mu = b.fric[i] * p.friction;
#endif
    if (depth <= 0.0f) { return; }
#if ICE
    mu *= 1.0f - p.patches * ice_at(b.pos[i].x);
#endif
    b.pos[i].x += nx * depth;
    b.pos[i].y += ny * depth;
    // The slide along the ground since the substep began.
    const float tx = ny, ty = -nx;
    const float slide = (b.pos[i].x - b.prev[i].x) * tx + (b.pos[i].y - b.prev[i].y) * ty;
    const float budget = mu * depth;
    const float back = fabsf(slide) <= budget ? slide : copysignf(budget, slide);
    b.pos[i].x -= tx * back;
    b.pos[i].y -= ty * back;
#if RECORD
    b.normal_force[i] += b.mass[i] * depth * INV_H * INV_H;
    b.friction_force[i] += b.mass[i] * back * INV_H * INV_H;
#endif
#endif
}

// Joint damping: each joint loses a share of its turning speed every
// substep, with equal and opposite pushes that keep the body's momentum.
__device__ void damp_joints(Body& b) {
    const float share = fminf(H * INV_JOINT_DAMPING, 1.0f);
    for (unsigned j = 1u; j < b.bones; j++) {
        if (b.parent[j] == 0xffu) { continue; }
        const Joint k = joint(b, j, false);
        const float rate = k.g_tip.x * b.vel[k.tip].x + k.g_tip.y * b.vel[k.tip].y
                         + k.g_far.x * b.vel[k.far].x + k.g_far.y * b.vel[k.far].y
                         + k.g_hub.x * b.vel[k.hub].x + k.g_hub.y * b.vel[k.hub].y;
        const float w = joint_weight(b, k);
        if (w > 0.0f) { joint_push(b, k, -share * rate / w, b.vel); }
    }
}

// One substep: forces, prediction, constraints, velocities.
__device__ void substep(Body& b, const float* __restrict__ muscles, const Params& p) {
    muscle_forces(b, muscles, p);
    bone_drag(b, p);
    for (unsigned i = 0u; i < b.nodes; i++) {
        const float2 f = node_force(b, i, p);
        b.vel[i].x += H * (f.x + b.muscle_force[i].x) * b.inv_m[i];
        b.vel[i].y += H * (f.y + b.muscle_force[i].y) * b.inv_m[i];
#if AIR
        b.vel[i].x *= p.air_sub;
        b.vel[i].y *= p.air_sub;
#endif
        b.prev[i] = b.pos[i];
        b.pos[i].x += H * b.vel[i].x;
        b.pos[i].y += H * b.vel[i].y;
    }
    solve_bones(b);
    solve_joints(b);
    for (unsigned i = 0u; i < b.nodes; i++) {
        solve_ground(b, i, p);
        b.vel[i].x = (b.pos[i].x - b.prev[i].x) * INV_H;
        b.vel[i].y = (b.pos[i].y - b.prev[i].y) * INV_H;
    }
    damp_joints(b);
}

// Whether a joint is forced past its range by more than JOINT_BREAK.
__device__ unsigned long long broken_joints(const Body& b) {
    unsigned long long bits = 0ull;
    for (unsigned j = 1u; j < b.bones; j++) {
        if (b.parent[j] == 0xffu) { continue; }
        const float angle = joint(b, j).angle;
        if (angle < b.lo[j] - JOINT_BREAK || angle > b.hi[j] + JOINT_BREAK) { bits |= 1ull << j; }
    }
    return bits;
}

#if RECORD
// Frame t of a recording: node positions, then per muscle its energy and
// pull, then per node its contact forces, then the broken joints.
__device__ void record_frame(float2* __restrict__ frames, unsigned t, const Body& b, const Params& p) {
    const unsigned fb = t * p.stride;
    for (unsigned i = 0u; i < b.nodes; i++) {
        frames[fb + i] = make_float2(b.pos[i].x, b.pos[i].y);
        frames[fb + MAXN + b.muscles + i] = make_float2(b.normal_force[i], b.friction_force[i]);
    }
    for (unsigned k = 0u; k < b.muscles; k++) {
        frames[fb + MAXN + k] = make_float2(b.energy[k], b.pull[k]);
    }
    const unsigned long long broken = broken_joints(b);
    frames[fb + p.stride - 1u] = make_float2(__uint_as_float((unsigned)broken), __uint_as_float((unsigned)(broken >> 32)));
}
#endif

// The behavior totals of a trial, kept between steps.
struct Tally {
    unsigned contact_bits, lift_bits, ground_bits;
    float head_shake;
    // The head's position and velocity at the last 60 Hz sample.
    float2 head_at, head_vel;
    // The rung trace while the trial runs: the distance half a second
    // before a rung, the speed pair, and the contact and energy pairs.
    unsigned rung_x, rung_speed, rung_early, rung_late;
    unsigned rung_bits;
};

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
    Body b;
    for (;;) {
        // The next creature of the wave.
        const unsigned got = atomicAdd(counter, 1u);
        if (got >= p.count) { return; }
        const unsigned cidx = p.base + got;
        const uint4 h0 = heads[2u * cidx];
        const uint4 h1 = heads[2u * cidx + 1u];
        b.nodes = h0.x & 255u;
        b.bones = b.nodes - 1u;
        b.muscles = h0.y;
        const unsigned* rec = records + h0.w;
        const float* mus = muscles + h1.x;
        b.total_mass = __uint_as_float(h1.y);
        b.inv_mass = __uint_as_float(h1.z);
        const unsigned flags = h1.w >> 16u;
        const float period_f = h2f(h1.w & 0xffffu);
#if QUAKE
        b.qphase = (float)(h0.z & 0xffffu) * (1.0f / 65536.0f);
        b.amp = p.terrain + p.quake * (0.6f + (float)((h0.z >> 16u) & 0xffffu) * (0.8f / 65536.0f));
#else
        b.qphase = 0.0f;
        b.amp = p.terrain;
#endif
        b.feet = 0u;
        for (unsigned i = 0u; i < b.nodes; i++) {
            const unsigned* r = rec + i * NODE_WORDS;
            b.mass[i] = __uint_as_float(r[0]);
            b.inv_m[i] = 1.0f / b.mass[i];
            b.radius[i] = __uint_as_float(r[1]);
            b.fric[i] = __uint_as_float(r[2]);
            b.pos[i].x = __uint_as_float(r[3]);
            b.pos[i].y = __uint_as_float(r[4]);
            b.feet |= (r[5] & 1u) << i;
            b.vel[i].x = 0.0f; b.vel[i].y = 0.0f;
            b.normal_force[i] = 0.0f; b.friction_force[i] = 0.0f;
        }
        const unsigned* bone_rec = rec + b.nodes * NODE_WORDS;
        for (unsigned j = 0u; j < b.bones; j++) {
            const unsigned* r = bone_rec + j * BONE_WORDS;
            b.pivot[j] = (unsigned char)r[0];
            b.length[j] = __uint_as_float(r[1]);
            b.parent[j] = (unsigned char)(r[2] == NONE ? 0xffu : r[2]);
            b.lo[j] = __uint_as_float(r[3]);
            b.hi[j] = __uint_as_float(r[4]);
        }
        for (unsigned k = 0u; k < b.muscles; k++) {
            b.energy[k] = 1.0f; b.offset[k] = 0.0f; b.demand[k] = 0.0f; b.pull[k] = 0.0f;
        }
#if RECORD
        // The settling frames show the start pose.
        for (unsigned t = 0u; t <= SETTLE; t++) { record_frame(frames, t, b, p); }
        bool scoring = true;
        Result kept;
#endif
        Result mt;
        mt.fitness = 0.0f; mt.ground_contact = 0.0f; mt.vertical_oscillation = 1e20f; mt.gait_frequency = -1e20f;
        mt.previous_center_y = 0.0f; mt.vertical_extremum = 0.0f; mt.vertical_trend = 0.0f; mt.gait_turns = 0.0f;
        mt.height_sum = 0.0f; mt.contact_lo = 0.0f; mt.contact_hi = 0.0f; mt.lift_lo = 0.0f; mt.lift_hi = 0.0f;
        mt.ground_lo = 0.0f; mt.ground_hi = 0.0f; mt.fall_time = 0.0f; mt.head_shake = 0.0f; mt.screen_x = 0.0f;
        mt.screened = 0.0f;
        Tally tl;
        tl.contact_bits = 0u; tl.lift_bits = 0u; tl.ground_bits = 0u; tl.head_shake = 0.0f;
        tl.head_at = b.pos[0]; tl.head_vel = make_float2(0.0f, 0.0f);
        tl.rung_x = 0u; tl.rung_speed = 0u; tl.rung_early = 0u; tl.rung_late = 0u; tl.rung_bits = 0u;
        bool limp = false;

        const unsigned rung1 = (unsigned)(RATE) - 1u, rung2 = (unsigned)(2.5f * RATE) - 1u;
        const unsigned rung3 = (unsigned)(5.0f * RATE) - 1u, rung4 = (unsigned)(10.0f * RATE) - 1u;
        const unsigned half_s = (unsigned)(0.5f * RATE);
        for (unsigned step = 0u; step < p.steps; step++) {
            const float t_now = (float)step * DT;
#if RECORD
            for (unsigned i = 0u; i < b.nodes; i++) { b.normal_force[i] = 0.0f; b.friction_force[i] = 0.0f; }
#endif
            muscle_demands(b, mus, t_now, limp);
            for (unsigned s = 0u; s < SUBSTEPS; s++) {
                substep(b, mus, p);
            }
#if RECORD
            for (unsigned i = 0u; i < b.nodes; i++) {
                b.normal_force[i] *= 1.0f / SUBSTEPS;
                b.friction_force[i] *= 1.0f / SUBSTEPS;
            }
#endif

            // What the step leaves: distance, height, contacts.
            bool failed = false;
            float center_y = 0.0f, com_x = 0.0f, low = 1e20f, high = -1e20f;
            unsigned touching = 0u, lifted = 0u;
            for (unsigned i = 0u; i < b.nodes; i++) {
                failed = failed || !(fabsf(b.pos[i].x) <= 1e6f && fabsf(b.pos[i].y) <= 1e6f);
                center_y += b.pos[i].y;
                com_x += b.mass[i] * b.pos[i].x;
                low = fminf(low, b.pos[i].y - b.radius[i]);
                high = fmaxf(high, b.pos[i].y + b.radius[i]);
#if GROUND
                const float2 g = ground(b.pos[i].x, b.amp, b.qphase, p);
                const float floor_y = g.x + b.radius[i] * sqrtf(1.0f + g.y * g.y);
                if (b.pos[i].y <= floor_y + CONTACT_SLACK) { touching |= 1u << i; }
                if (b.pos[i].y > floor_y + LIFT_CLEARANCE) { lifted |= 1u << i; }
#endif
            }
            center_y /= (float)b.nodes;
            com_x *= b.inv_mass;
            tl.contact_bits |= touching;
            tl.lift_bits |= tl.contact_bits & lifted;
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
            // 1/60 s, so the measure is the same at every step rate.
            if ((step + 1u) % HEAD_SAMPLE == 0u) {
                const float2 v = make_float2((b.pos[0].x - tl.head_at.x) * 60.0f, (b.pos[0].y - tl.head_at.y) * 60.0f);
                const float ax = (v.x - tl.head_vel.x) * 60.0f, ay = (v.y - tl.head_vel.y) * 60.0f;
                if (t_now >= HEAD_SHAKE_WINDOW) {
                    const float accel = sqrtf(ax * ax + ay * ay);
                    tl.head_shake += (accel - tl.head_shake) * fminf(1.0f / (HEAD_SHAKE_WINDOW * 60.0f), 1.0f);
                }
                tl.head_at = b.pos[0];
                tl.head_vel = v;
            }
            const bool broken = broken_joints(b) != 0ull;
#if RECORD
            if (scoring) {
#endif
            mt.contact_lo = __uint_as_float(tl.contact_bits);
            mt.lift_lo = __uint_as_float(tl.lift_bits);
            mt.ground_lo = __uint_as_float(tl.ground_bits);
            mt.head_shake = tl.head_shake;
            const bool fell = b.pos[0].y < b.pos[1].y || broken || tl.head_shake > HEAD_SHAKE_LIMIT || failed;
            bool ended = false;
            if (fell) {
                mt.fall_time = t_now + DT;
                mt.fitness = failed ? -1e20f : com_x;
                if (step <= p.screen_step) { mt.screen_x = mt.fitness; }
                ended = true;
            }
            mt.ground_contact += (float)__popc(touching);
            mt.height_sum += high - low;
            mt.vertical_oscillation = fminf(mt.vertical_oscillation, center_y);
            mt.gait_frequency = fmaxf(mt.gait_frequency, center_y);
            // Gait turns: the center's height changes direction.
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
            // The rung trace (creature_kernel::RungTrace): the distance at 1,
            // 2.5, 5 and 10 s and the early features at 1 and 2.5 s, as fp16
            // pairs in result words the host reads for nothing else.
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
                // The cadence band the audit lane files this trial under.
                const unsigned r = second ? 1u : 0u;
                const float gait_now = mt.gait_turns * (0.5f / (t_now + DT));
                const unsigned band = min((unsigned)(gait_now * (8.0f / 6.0f)), 7u);
                tl.rung_bits |= band << (8u + 3u * r);
                // Flags: 1 audit, 2 exempt, 4 and 8 exempt from R1 and R2.
                if (!ended && (flags & (second ? 0xbu : 0x7u)) == 0u) {
                    const RungParams rp = second ? p.r2 : p.r1;
                    const float fv[6] = {h2f(f2h(com_x)), h2f(f2h(speed)), h2f(f2h(touched)),
                                         h2f(f2h(tl.head_shake)), h2f(f2h(en_mean)), period_f};
                    float score = 0.0f;
                    bool finite = true;
                    for (int i = 0; i < 6; i++) {
                        score = fmaf(rp.w[i], fv[i], score);
                        finite = finite && isfinite(fv[i]);
                    }
                    if (finite && score < rp.bias && ((rp.off >> band) & 1u) == 0u) {
                        mt.screened = t_now + DT;
                        mt.screen_x = com_x;
                        mt.fitness = com_x;
                        tl.rung_bits |= (r + 1u) << 6u;
                        ended = true;
                    }
                }
            }
            if (step == rung3 || step == rung4) {
                const bool second = step == rung4;
                const unsigned keep = second ? 0x0000ffffu : 0xffff0000u;
                mt.lift_hi = __uint_as_float((__float_as_uint(mt.lift_hi) & keep) | (f2h(com_x) << (second ? 16u : 0u)));
            }
            // The early screen: below the bar at the screen time, the trial stops.
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
                const unsigned code = (fell ? 16u : 0u) | (failed ? 32u : 0u) | (mt.screened > 0.0f ? 3u : 0u)
                    | (tl.rung_bits & 0x3fc0u) | ((flags & 1u) << 14u);
                mt.previous_center_y = __uint_as_float(tl.rung_speed);
                mt.vertical_extremum = __uint_as_float(tl.rung_early);
                mt.vertical_trend = __uint_as_float(tl.rung_late);
                mt.gait_turns = mt.ground_hi;
                mt.ground_hi = __uint_as_float(code | (min(step + 1u, 65535u) << 16u));
#if RECORD
                // A recording plays the whole trial; after the score is
                // final the muscles go limp.
                kept = mt;
                scoring = false;
                limp = true;
#else
                results[cidx] = mt;
                break;
#endif
            }
#if RECORD
            }
            record_frame(frames, SETTLE + step + 1u, b, p);
#endif
        }
#if RECORD
        results[cidx] = kept;
#endif
    }
}
