// Physics v3 on NVIDIA GPUs: one creature per lane group (src/warp_kernel.rs
// packs the creatures and writes the #defines in front of this text).
//
// A warp holds 32 / W groups of W lanes (W = 4, 8, 16 or 32). Lane i of a group
// owns node i of its creature and the bone that ends there, so lane 0 is the
// head and lane 1 the neck bone. Lanes are numbered breadth first over the
// bone tree: the bones of one tree level are neighbours, and the children of
// one bone are consecutive lanes. The tree passes (kinematics, the
// articulated-body pass, the responses) run level by level and exchange data
// by warp shuffles. All state stays in registers for the whole trial.
//
// A group runs one creature from its start to its end (a fall, the screen or
// the last step), then takes the next creature of the wave from an atomic
// counter. The wave is sorted by muscle rounds and has one counter per
// bucket of NB (1, 2, 3, or 4 and more rounds); every warp starts on one
// bucket, so its groups run about the same round count, and moves to the next
// bucket up when its own runs dry. Each 1/RATE step is SUBSTEPS substeps. A substep is one
// articulated-body pass with the muscles, gravity, wind, drag and water, a
// contact solve of at most MAXC contacts by projected Gauss-Seidel on the
// exact contact-space matrix, the contact response, semi-implicit Euler, a
// momentum balance and, without contact, the first-law check.
//
// Defines from warp_kernel::cuda_source: W, MAXR (the most muscle rounds of
// the class), BLOCK, RATE, SETTLE, SAMPLE, SUBSTEPS, PGS_SWEEPS,
// CLEAN_SWEEPS, RECORD, the world flags (GROUND,
// TERRAIN, SLOPE, GAPS, HURDLES, QUAKE, MUD, WATER, ICE, WIND, AIR) and the
// physics constants.

#define FULL 0xffffffffu
#define GM ((W == 32) ? 0xffffffffu : ((1u << W) - 1u))
#define DT (1.0f / RATE)
#define HS (1.0f / (RATE * SUBSTEPS))
#define INV_HS (RATE * SUBSTEPS)
#define LF 12u
#define MF 16u
#define NB 4
#define MAXC 4
#define PI_F 3.14159265359f
#define TAU_F 6.28318530718f

// One early rung's rule (rungs::Rung): stop when the chain of fused
// multiply-adds of the weights and the six features, from zero, is below the
// bias; `off` has bit b set when cadence band b is off.
struct RungParams {
    float w[6];
    float bias;
    unsigned off;
};
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
// The take-up buckets of a wave (cuda_engine::Takeup): bucket b holds the
// wave's creatures from start[b] to end[b], sorted by muscle rounds, and
// warps from warp[b] on (in the order warp in block, then block) start on it.
struct Takeup {
    unsigned start[NB];
    unsigned end[NB];
    unsigned warp[NB];
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

// Half precision, as the rung trace and the rung features carry values.
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

typedef float3 vec3;
__device__ __forceinline__ vec3 v3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ vec3 operator+(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ vec3 operator-(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ vec3 operator-(vec3 a) { return v3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ vec3 operator*(vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ void operator+=(vec3& a, vec3 b) { a = a + b; }
__device__ __forceinline__ void operator-=(vec3& a, vec3 b) { a = a - b; }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ __forceinline__ float sdot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
// Planar spatial vectors (angular, x, y) about the substep's origin.
__device__ __forceinline__ vec3 crm(vec3 v, vec3 u) {
    return v3(0.0f, -v.x * u.z + u.x * v.z, v.x * u.y - u.x * v.y);
}
__device__ __forceinline__ vec3 crf(vec3 v, vec3 f) {
    return v3(v.y * f.z - v.z * f.y, -v.x * f.z, v.x * f.y);
}
__device__ __forceinline__ vec3 force_at(float rx, float ry, float fx, float fy) {
    return v3(rx * fy - ry * fx, fx, fy);
}
__device__ __forceinline__ vec3 sym_mul(vec3 s0, vec3 s1, vec3 v) {
    return v3(
        s0.x * v.x + s0.y * v.y + s0.z * v.z,
        s0.y * v.x + s1.x * v.y + s1.y * v.z,
        s0.z * v.x + s1.y * v.y + s1.z * v.z);
}
__device__ __forceinline__ float shf(float v, unsigned src) { return __shfl_sync(FULL, v, src, W); }
__device__ __forceinline__ unsigned shu(unsigned v, unsigned src) { return __shfl_sync(FULL, v, src, W); }
__device__ __forceinline__ vec3 shv(vec3 v, unsigned src) { return v3(shf(v.x, src), shf(v.y, src), shf(v.z, src)); }
// Sums, minima and maxima over the lane group; every lane gets the same bits.
__device__ __forceinline__ float gsum(float v) {
#pragma unroll
    for (unsigned o = W / 2u; o > 0u; o >>= 1u) { v += __shfl_xor_sync(FULL, v, o, W); }
    return v;
}
__device__ __forceinline__ float gmin(float v) {
#pragma unroll
    for (unsigned o = W / 2u; o > 0u; o >>= 1u) { v = fminf(v, __shfl_xor_sync(FULL, v, o, W)); }
    return v;
}
__device__ __forceinline__ float gmax(float v) {
#pragma unroll
    for (unsigned o = W / 2u; o > 0u; o >>= 1u) { v = fmaxf(v, __shfl_xor_sync(FULL, v, o, W)); }
    return v;
}
__device__ __forceinline__ unsigned gmaxu(unsigned v) {
#pragma unroll
    for (unsigned o = W / 2u; o > 0u; o >>= 1u) { v = max(v, __shfl_xor_sync(FULL, v, o, W)); }
    return v;
}
__device__ __forceinline__ float wave(float t, float inv_period, float phase, float offset, float duty, float inv_duty, float inv_complement) {
    const float x = t * inv_period + phase + offset;
    const float ph = x - floorf(x);
    const bool rise = ph < duty;
    const float arg = rise ? ph * inv_duty : (ph - duty) * inv_complement;
    const float half = rise ? 0.5f : -0.5f;
    return 0.5f + half * __cosf(PI_F * arg);
}
__device__ __forceinline__ float ice_at(float x) {
    float u = x * ICE_INV;
    float w = u - floorf(u);
    float t = fabsf(w - 0.5f) * 2.0f;
    float s = clampf((0.7f - t) * 2.5f, 0.0f, 1.0f);
    return s * s * (3.0f - 2.0f * s);
}

#if PROFILE
// Cycles per kernel section of each warp, for a developer's breakdown.
#define PROF(k) { const long long now_ = clock64(); if (lane == 0u) { s_prof[threadIdx.x >> 5][prof_at] += now_ - prof_t; } prof_t = now_; prof_at = (k); }
#else
#define PROF(k)
#endif

extern "C" __global__ void __launch_bounds__(BLOCK, MIN_BLOCKS) advance(
    const unsigned* __restrict__ lanes,
    const float* __restrict__ muscles,
    const unsigned* __restrict__ ends,
    const uint4* __restrict__ heads,
    Result* __restrict__ results,
    unsigned* __restrict__ counter,
    const Params p,
    const Takeup tk
#if RECORD
    , float2* __restrict__ frames
#endif
    ) {
    // 96 float4 per warp, reused by phase: the muscle forces (two per muscle
    // lane), the children's articulated inertias and biases (three per
    // lane), the walkers' torques at each joint (two per lane).
    __shared__ float4 scatter[BLOCK / 32][96];
    // Per-muscle state (energy store, rhythm offset, force of the substep) of
    // each lane's muscles, and each group's behavior totals: touched once per
    // round or per step, so they wait in shared memory instead of registers.
    __shared__ float s_en[MAXR][BLOCK];
    __shared__ float s_off[MAXR][BLOCK];
    __shared__ float s_mag[MAXR][BLOCK];
    // The waveform at the last substep, or -1 when it must be computed.
    __shared__ float s_wp[MAXR][BLOCK];
    __shared__ Result s_mt[BLOCK / W];
    __shared__ uint4 s_bits[BLOCK / W];
    // Each lane's joint after the articulated-body pass (pivot arm, 1 / d,
    // topology; the u vector), the root's inverse inertia per group, and a
    // walker's contact-matrix entries per contact c: its normal row at
    // columns 2c and 2c + 1, then its friction row at the same columns.
    // Two float4 per lane, reused by phase: the node table (position and
    // pivot position, velocity and pivot velocity) for the muscles, then the
    // joint table (pivot arm, 1 / d, topology; u) for the contact walkers.
    __shared__ float4 s_t0[BLOCK];
    __shared__ float4 s_t1[BLOCK];
    __shared__ float4 s_root[BLOCK / W][2];
    // Each lane's bone and its ancestors, as a lane mask.
    __shared__ unsigned s_anc[BLOCK];
    const unsigned tid = threadIdx.x;
    const unsigned gtid = tid - (threadIdx.x & 31u) % W;
#if PROFILE
    __shared__ long long s_prof[BLOCK / 32][16];
    for (int k = 0; k < 16; k++) { if ((threadIdx.x & 31u) == 0u) { s_prof[threadIdx.x >> 5][k] = 0; } }
    long long prof_t = clock64();
    int prof_at = 0;
#endif
    const unsigned lane = threadIdx.x & 31u;
    const unsigned lg = lane % W;
    const unsigned gbase = lane - lg;
    const unsigned gshift = gbase;
    // Every phase keeps to its own lanes' slice of the warp's scratch (three
    // float4 per lane), so groups of one warp that run apart never touch the
    // same memory: the muscle forces sit at the start of the group's slice.
    float4* const mf = &scatter[threadIdx.x >> 5][gbase * 3u];
    float4* const region = scatter[threadIdx.x >> 5];
    // The scratch starts at zero: shared memory holds what the last kernel
    // left there.
    for (unsigned k = 0u; k < 3u; k++) { region[3u * lane + k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
    __syncwarp();
    const unsigned below = (1u << lg) - 1u;

    // The group's creature (the same in every lane of the group).
    bool live = false;
    bool exhausted = false;
    bool fresh = false;
    bool limp = false;
    unsigned cidx = 0u, nn = 0u, depth = 0u, rounds = 0u, ew = 0u, nmus = 0u, mbase = 0u, ebase = 0u;
    unsigned step = 0u;
    float amp = 0.0f, qphase = 0.0f, inv_mass = 0.0f, inv_nodes = 0.0f;
    // This lane's node and bone.
    float m = 0.0f, rad = 0.0f, fric = 0.0f, len = 0.0f, lo = 0.0f, hi = 0.0f, prad = 0.0f, hm = 0.0f;
    unsigned topo = 0u, mnode = 0u;
    // State: the joint angle and rate of the lane's bone (the neck's absolute
    // angle on lane 1); the head's position and velocity are lane 0's node
    // position and velocity.
    float q = 0.0f, qd = 0.0f;
    float tpull[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; r++) { s_en[r][tid] = 1.0f; s_off[r][tid] = 0.0f; s_mag[r][tid] = 0.0f; s_wp[r][tid] = -1.0f; tpull[r] = 0.0f; }
    // Kinematics: absolute angle and rate, node and pivot position and velocity.
    float th = 0.0f, om = 0.0f, px = 0.0f, py = 0.0f, vx = 0.0f, vy = 0.0f;
    float ppx = 0.0f, ppy = 0.0f, pvx = 0.0f, pvy = 0.0f;
    // Behavior totals.
    Result& mt = s_mt[tid / W];
    unsigned& contact_bits = s_bits[tid / W].x;
    unsigned& lift_bits = s_bits[tid / W].y;
    unsigned& ground_bits = s_bits[tid / W].z;
    float& head_shake = *reinterpret_cast<float*>(&s_bits[tid / W].w);
    // Contact forces of the last step, per node, for a recording.
    float rec_n = 0.0f, rec_t = 0.0f;
#if RECORD
    Result kept;
    bool done_scoring = false;
#endif

    auto reset_metrics = [&]() {
        mt.fitness = 0.0f; mt.ground_contact = 0.0f; mt.vertical_oscillation = 1e20f; mt.gait_frequency = -1e20f;
        mt.previous_center_y = 0.0f; mt.vertical_extremum = 0.0f; mt.vertical_trend = 0.0f; mt.gait_turns = 0.0f;
        mt.height_sum = 0.0f; mt.contact_lo = 0.0f; mt.contact_hi = 0.0f; mt.lift_lo = 0.0f; mt.lift_hi = 0.0f;
        mt.ground_lo = 0.0f; mt.ground_hi = 0.0f; mt.fall_time = 0.0f; mt.head_shake = 0.0f; mt.screen_x = 0.0f;
        mt.screened = 0.0f;
        contact_bits = 0u; lift_bits = 0u; ground_bits = 0u; head_shake = 0.0f;
    };
    reset_metrics();

    // Height and slope of the ground under x; flat ground compiles to (0, 0).
    auto terrain = [&](float x) -> float2 {
        float height = 0.0f, slope = 0.0f;
#if TERRAIN
        {
            float t0 = x * (1.0f / 1.1f) + qphase;
            float u0 = t0 - floorf(t0);
            float w0 = u0 * (1.0f - u0);
            float t1 = x * (1.0f / 0.43f) + 0.3f + qphase;
            float u1 = t1 - floorf(t1);
            float w1 = u1 * (1.0f - u1);
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
            float spacing = 2.0f + 4.0f * p.gaps;
            float center = spacing * 0.5f;
            float r = x - floorf(x / spacing) * spacing;
            float distance = fabsf(r - center);
            float half_w = 0.5f * p.gaps;
            float run = fmaxf(fminf(GAP_RUN, half_w), 1e-6f);
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
            float spacing = HURDLE_SPACING;
            float center = spacing * 0.5f;
            float r = x - floorf(x / spacing) * spacing;
            float distance = fabsf(r - center);
            float half_w = 0.5f * HURDLE_TOP;
            float run = HURDLE_RUN;
            float factor = clampf((half_w + run - distance) / run, 0.0f, 1.0f);
            if (distance <= half_w) { factor = 1.0f; } else if (distance >= half_w + run) { factor = 0.0f; }
            height += p.hurdles * factor;
            if (distance > half_w && distance < half_w + run) {
                slope += p.hurdles * ((r < center ? 1.0f : -1.0f) / run);
            }
        }
#endif
        return make_float2(height, slope);
    };

    // This warp's bucket: the last one whose first warp is at or before it.
    unsigned bucket = 0u;
    {
        const unsigned order = (threadIdx.x >> 5) * gridDim.x + blockIdx.x;
#pragma unroll
        for (unsigned b = 1u; b < NB; b++) { if (tk.warp[b] <= order) { bucket = b; } }
    }
    const unsigned* ltab = lanes;
    for (;;) {
        // A group without a creature takes the next one of the wave.
        PROF(0);
        if (!live && !exhausted) {
            // The next creature of this group's bucket, or of the next bucket
            // up with creatures left.
            unsigned got = 0xffffffffu;
            if (lg == 0u) {
                for (unsigned k = 0u; k < NB; k++) {
                    const unsigned b = (bucket + k) % NB;
                    const unsigned size = tk.end[b] - tk.start[b];
                    if (size == 0u || *(volatile unsigned*)&counter[b] >= size) { continue; }
                    const unsigned i = atomicAdd(&counter[b], 1u);
                    if (i < size) { got = tk.start[b] + i; bucket = b; break; }
                }
            }
            got = __shfl_sync(GM << gshift, got, gbase);
            bucket = __shfl_sync(GM << gshift, bucket, gbase);
            if (got < p.count) {
                cidx = p.base + got;
                const uint4 h0 = heads[2u * cidx];
                const uint4 h1 = heads[2u * cidx + 1u];
                nn = h0.x & 255u;
                depth = (h0.x >> 8u) & 255u;
                rounds = (h0.x >> 16u) & 255u;
                ew = h0.x >> 24u;
                nmus = h0.y;
                mbase = h0.w;
                ebase = h1.x;
                inv_mass = __uint_as_float(h1.z);
                inv_nodes = 1.0f / (float)nn;
#if QUAKE
                qphase = (float)(h0.z & 0xffffu) * (1.0f / 65536.0f);
                amp = p.terrain + p.quake * (0.6f + (float)((h0.z >> 16u) & 0xffffu) * (0.8f / 65536.0f));
#else
                qphase = 0.0f;
                amp = p.terrain;
#endif
                const unsigned* rec = ltab + cidx * (LF * W) + lg;
                m = __uint_as_float(rec[0u * W]);
                rad = __uint_as_float(rec[1u * W]);
                fric = __uint_as_float(rec[2u * W]);
                len = __uint_as_float(rec[3u * W]);
                lo = __uint_as_float(rec[4u * W]);
                hi = __uint_as_float(rec[5u * W]);
                const float s6 = __uint_as_float(rec[6u * W]);
                const float s7 = __uint_as_float(rec[7u * W]);
                prad = __uint_as_float(rec[8u * W]);
                topo = rec[9u * W];
                s_anc[tid] = rec[10u * W];
                mnode = rec[11u * W];
                hm = lg == 1u ? s7 : 0.0f;
                if (lg == 0u) {
                    px = s6; py = s7; vx = 0.0f; vy = 0.0f; q = 0.0f;
                } else {
                    q = s6;
                }
                qd = 0.0f;
                th = 0.0f; om = 0.0f;
#pragma unroll
                for (int r = 0; r < MAXR; r++) { s_en[r][tid] = 1.0f; s_off[r][tid] = 0.0f; s_mag[r][tid] = 0.0f; s_wp[r][tid] = -1.0f; tpull[r] = 0.0f; }
                reset_metrics();
                rec_n = 0.0f; rec_t = 0.0f;
                step = 0u;
                live = true;
                fresh = true;
                limp = false;
#if RECORD
                done_scoring = false;
#endif
            } else {
                exhausted = true;
                nn = 0u; depth = 0u; rounds = 0u; ew = 0u; nmus = 0u;
                m = 0.0f; rad = 0.0f; fric = 0.0f; len = 0.0f; hm = 0.0f; topo = 0u;
                inv_mass = 0.0f; inv_nodes = 0.0f;
                q = 0.0f; qd = 0.0f;
            }
        }
        if (!__any_sync(FULL, live)) {
            break;
        }
        const bool valid = lg < nn;
        const bool body = valid && lg >= 1u;
        const unsigned piv = topo & 31u;
        const unsigned pb = (topo >> 5u) & 31u;
        const unsigned lvl = (topo >> 10u) & 31u;
        const unsigned fc = (topo >> 15u) & 31u;
        const unsigned nch = (topo >> 20u) & 63u;
        const unsigned maxlev = __reduce_max_sync(FULL, depth);
        const unsigned maxrounds = __reduce_max_sync(FULL, rounds);
        const unsigned maxew = __reduce_max_sync(FULL, ew);
        const unsigned maxsum = maxew > 0u ? maxew - 1u : 0u;

        // Node positions and velocities from the state, parents first.
        // `upd` false leaves the group's values as they are, so a neighbour
        // group taking a new creature never changes this one's numbers.
        auto kinematics = [&](bool upd) {
            for (unsigned L = 1u; L <= maxlev; L++) {
                const float a = shf(th, pb);
                const float w = shf(om, pb);
                if (lvl == L && upd) {
                    th = lg == 1u ? q : a + q;
                    om = lg == 1u ? qd : w + qd;
                }
            }
            float sn, cs;
            __sincosf(th - TAU_F * rintf(th * (1.0f / TAU_F)), &sn, &cs);
            for (unsigned L = 1u; L <= maxlev; L++) {
                const float ax = shf(px, piv);
                const float ay = shf(py, piv);
                const float bx = shf(vx, piv);
                const float by = shf(vy, piv);
                if (lvl == L && upd) {
                    ppx = ax; ppy = ay; pvx = bx; pvy = by;
                    px = ax + len * cs;
                    py = ay + len * sn;
                    vx = bx - len * om * sn;
                    vy = by + len * om * cs;
                }
            }
        };
#if RECORD
        // Called by every lane; only a group with `write` stores the frame.
        auto record_frame = [&](unsigned t, bool write) {
            const unsigned fb = t * p.stride;
            if (valid && write) {
                frames[fb + mnode] = make_float2(px, py);
                frames[fb + W + nmus + mnode] = make_float2(rec_n, rec_t);
            }
#pragma unroll
            for (int r = 0; r < MAXR; r++) {
                const unsigned mi = (unsigned)r * W + lg;
                if (write && (unsigned)r < rounds && mi < nmus) {
                    frames[fb + W + mi] = make_float2(s_en[r][tid], s_mag[r][tid] + tpull[r]);
                }
            }
            const bool broken = body && lg >= 2u && (q < lo - JOINT_BREAK || q > hi + JOINT_BREAK);
            const unsigned bits = (__ballot_sync(FULL, broken) >> gshift) & GM;
            // Broken joints by the creature's bone number (bone j ends at node j + 1).
            unsigned lo_bits = 0u, hi_bits = 0u;
            for (unsigned k = 0u; k < W; k++) {
                const unsigned node = shu(mnode, k);
                if ((bits >> k) & 1u) {
                    const unsigned b = node - 1u;
                    if (b < 32u) { lo_bits |= 1u << b; } else { hi_bits |= 1u << (b - 32u); }
                }
            }
            if (lg == 0u && write) {
                frames[fb + p.stride - 1u] = make_float2(__uint_as_float(lo_bits), __uint_as_float(hi_bits));
            }
        };
#endif
        const float t_now = (float)step * DT;
        float head_vx0 = 0.0f, head_vy0 = 0.0f;
        float step_n = 0.0f, step_t = 0.0f;
        // Carried from a substep's body to the balance at the top of the next.
        float mx = 0.0f, my = 0.0f, ledger = 0.0f, scale = 0.0f, y_start = 0.0f, buoy = 0.0f;
        unsigned nc = 0u;
        for (unsigned sub = 0u;; sub++) {
            // Node positions and velocities from the state: for a new
            // creature before its first substep, and after every substep.
            PROF(1);
            if (sub > 0u || __any_sync(FULL, fresh)) {
                kinematics(sub > 0u || fresh);
            }
            PROF(2);
            s_t0[tid] = make_float4(px, py, ppx, ppy);
            __syncwarp();
            if (sub == 0u) {
#if RECORD
                if (__any_sync(FULL, fresh)) {
                    for (unsigned t = 0u; t <= SETTLE; t++) { record_frame(t, fresh); }
                }
#endif
                fresh = false;
                head_vx0 = shf(vx, 0u);
                head_vy0 = shf(vy, 0u);
            } else {
                // Momentum balance: the body's momentum is its old momentum plus
                // the external impulses; the rest of first-order integration's
                // error goes as one uniform velocity.
                {
                    const float ex = gsum(mx), ey = gsum(my);
                    const float ax_ = gsum(valid ? m * vx : 0.0f), ay_ = gsum(valid ? m * vy : 0.0f);
    #if AIR
                    const float wantx = ex * p.air_sub, wanty = ey * p.air_sub;
    #else
                    const float wantx = ex, wanty = ey;
    #endif
                    const float sx = (wantx - ax_) * inv_mass, sy = (wanty - ay_) * inv_mass;
                    vx += sx;
                    vy += sy;
                    pvx += sx;
                    pvy += sy;
                    // First law in flight: a substep without ground contact gains no
                    // more energy than the muscles, the wind, the buoyancy and the
                    // tendons put in.
                    if (__any_sync(FULL, live && nc == 0u)) {
    #pragma unroll
                        for (int r = 0; r < MAXR; r++) {
                            if ((unsigned)r >= maxrounds) { break; }
                            const unsigned mi = (unsigned)r * W + lg;
                            const bool mon = (unsigned)r < rounds && mi < nmus;
                            const float4* mrec = reinterpret_cast<const float4*>(muscles + mbase) + mi * 4u;
                            const float4 f0 = mon ? mrec[0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                            const unsigned packed = __float_as_uint(f0.x);
                            const unsigned la = packed & 31u;
                            const unsigned lb = (packed >> 5u) & 31u;
                            const float4 ea = s_t0[gtid + la], eb = s_t0[gtid + lb];
                            const float a_px = ea.x, a_py = ea.y, a_qx = ea.z, a_qy = ea.w;
                            const float b_px = eb.x, b_py = eb.y, b_qx = eb.z, b_qy = eb.w;
                            if (mon) {
                                const float4 f3 = mrec[3];
                                const float anchor_a = f0.y, anchor_b = f0.z;
                                const float tendon_k = f3.z, slack = f3.w;
                                const float pax = a_qx + (a_px - a_qx) * anchor_a, pay = a_qy + (a_py - a_qy) * anchor_a;
                                const float pbx = b_qx + (b_px - b_qx) * anchor_b, pby = b_qy + (b_py - b_qy) * anchor_b;
                                const float dx = pbx - pax, dy = pby - pay;
                                const float length_m = sqrtf(dx * dx + dy * dy);
                                const float stretch = fmaxf(length_m - slack, 0.0f);
                                ledger += 0.5f * tendon_k * stretch * stretch + s_mag[r][tid] * length_m;
                            }
                        }
                        if (valid) {
                            ledger += 0.5f * m * (vx * vx + vy * vy) + m * p.gravity * py;
    #if WIND
                            ledger -= p.wind * m * px;
    #endif
    #if WATER
                            ledger -= buoy * (py - y_start);
    #endif
                        }
                        const float excess = gsum(ledger) - (1e-4f + 1e-5f * gsum(scale));
                        const float cx = wantx * inv_mass, cy = wanty * inv_mass;
                        const float dvx = vx - cx, dvy = vy - cy;
                        const float internal = gsum(valid ? 0.5f * m * (dvx * dvx + dvy * dvy) : 0.0f);
                        if (live && nc == 0u && excess > 0.0f) {
                            const float keep = internal > 0.0f ? sqrtf(fmaxf(1.0f - excess / internal, 0.0f)) : 0.0f;
                            qd *= keep;
                            om *= keep;
                            vx = cx + keep * dvx;
                            vy = cy + keep * dvy;
                            pvx = cx + keep * (pvx - cx);
                            pvy = cy + keep * (pvy - cy);
                        }
                    }
                }
            }
            PROF(3);
            if (sub == SUBSTEPS) {
                break;
            }
            s_t1[tid] = make_float4(vx, vy, pvx, pvy);
            // The first muscle round's first record, fetched early.
            float4 pf0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (0u < rounds && lg < nmus) { pf0 = reinterpret_cast<const float4*>(muscles + mbase)[lg * 4u]; }
            __syncwarp();
            const float ts = t_now + (float)sub * HS;
            const float ox = shf(px, 0u);
            const float oy = shf(py, 0u);
            // Momentum before the substep plus the external impulses, and the
            // first-law ledger, per lane.
            mx = valid ? m * vx : 0.0f;
            my = valid ? m * vy : 0.0f;
            ledger = 0.0f;
            scale = 0.0f;
            if (valid) {
                const float kinetic = 0.5f * m * (vx * vx + vy * vy);
                const float potential = m * p.gravity * py;
                ledger -= kinetic + potential;
                scale += kinetic + fabsf(potential);
#if WIND
                ledger += p.wind * m * px;
#endif
            }
            y_start = py;
            // Body inertia (the neck also carries the head), pivot arm and the
            // velocity-product force.
            vec3 i0 = v3(0.0f, 0.0f, 0.0f), i1 = v3(0.0f, 0.0f, 0.0f), bs = v3(0.0f, 0.0f, 0.0f);
            const float armx = ppx - ox;
            const float army = ppy - oy;
            const vec3 axis = v3(1.0f, army, -armx);
            if (body) {
                const float rx = px - ox, ry = py - oy;
                i0 = v3(m * (rx * rx + ry * ry), -m * ry, m * rx);
                i1 = v3(m + hm, 0.0f, m + hm);
                const vec3 sv = v3(om, pvx + om * army, pvy - om * armx);
                bs = crf(sv, sym_mul(i0, i1, sv));
            }
            // Gravity, wind, mud drag and buoyancy on the lane's node; the
            // head's force goes to the neck.
            buoy = 0.0f;
            vec3 nf = v3(0.0f, 0.0f, 0.0f);
            if (valid) {
                float fx = 0.0f;
                float fy = -p.gravity * m;
#if WIND
                fx += p.wind * m;
#endif
#if MUD
                {
                    const float2 g = terrain(px);
                    const float dry = (py - g.x) / sqrtf(1.0f + g.y * g.y) - rad;
                    fx -= m * MUD_DRAG * (clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH)) * vx;
                }
#endif
#if BRAMBLES
                // Brambles hold back every node but the feet (the ends of
                // bones with no child) while it touches the ground: a drag
                // against its velocity, which only takes energy.
                if (!(body && nch == 0u)) {
                    const float2 g = terrain(px);
                    const float dry = (py - g.x) / sqrtf(1.0f + g.y * g.y) - rad;
                    if (dry < BRAMBLE_REACH) {
                        fx -= m * p.brambles * vx;
                    }
                }
#endif
#if WATER
                {
                    const float sub_i = clampf((p.water - (py - rad)) / (2.0f * rad), 0.0f, 1.0f);
                    buoy = WATER_BUOYANCY * m * p.gravity * sub_i;
                    fy += buoy;
                }
#endif
                mx += fx * HS;
                my += fy * HS;
                nf = force_at(px - ox, py - oy, fx, fy);
            }
            {
                const vec3 head = shv(nf, 0u);
                if (lg == 1u) { bs -= head; }
                if (body) { bs -= nf; }
            }
            if (body) {
                // Air drag on the bone at its midpoint, limited so a substep of
                // drag never more than halves the speed.
                const float midx = 0.5f * (ppx + px), midy = 0.5f * (ppy + py);
                const float wx = 0.5f * (pvx + vx), wy = 0.5f * (pvy + vy);
                const float speed = sqrtf(wx * wx + wy * wy);
                const float width = prad + rad;
                const float strength = fmaxf(fminf(AIR_DRAG * len * width * speed, 0.5f * m * INV_HS), 0.0f);
                bs -= force_at(midx - ox, midy - oy, -wx * strength, -wy * strength);
                mx -= wx * strength * HS;
                my -= wy * strength * HS;
#if WATER
                {
                    const float sub_i = clampf((p.water - (py - rad)) / (2.0f * rad), 0.0f, 1.0f);
                    const float sub_p = clampf((p.water - (ppy - prad)) / (2.0f * prad), 0.0f, 1.0f);
                    const float wet = 0.5f * (sub_p + sub_i);
                    const float inverse = 1.0f / len;
                    const float ax_ = (px - ppx) * inverse, ay_ = (py - ppy) * inverse;
                    const float along = wx * ax_ + wy * ay_;
                    const float lx = ax_ * along, ly = ay_ * along;
                    const float sx = wx - lx, sy = wy - ly;
                    const float ws = fmaxf(fminf(WATER_DRAG * wet * len * width * speed, 0.5f * m * INV_HS), 0.0f);
                    const float weak = ws * WATER_ALONG;
                    const float fx = -(sx * ws + lx * weak), fy = -(sy * ws + ly * weak);
                    bs -= force_at(midx - ox, midy - oy, fx, fy);
                    mx += fx * HS;
                    my += fy * HS;
                }
#endif
                // Spin cap: rotational drag past the cap, implicit, toward rest.
                if (fabsf(om) > SPIN_CAP) {
                    const float drag = SPIN_HARDNESS * m * len * len * (fabsf(om) * INV_SPIN_CAP - 1.0f);
                    i0.x += drag;
                    bs.x += drag * INV_HS * om;
                }
            }
            PROF(4);
            // Muscles, W at a time: each muscle lane pulls its two bones' ends,
            // writes its forces on both bones to shared memory, and each bone
            // lane gathers the forces of the muscle ends it carries.
            vec3 fm = v3(0.0f, 0.0f, 0.0f);
#pragma unroll 1
            for (int r = 0; r < MAXR; r++) {
                if ((unsigned)r >= maxrounds) { break; }
                const unsigned mi = (unsigned)r * W + lg;
                const bool mon = (unsigned)r < rounds && mi < nmus;
                const float4* mrec = reinterpret_cast<const float4*>(muscles + mbase) + mi * 4u;
                const float4 f0 = pf0;
                {
                    const unsigned nx = (unsigned)(r + 1) * W + lg;
                    pf0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if ((unsigned)(r + 1) < rounds && nx < nmus) { pf0 = mrec[4u * W]; }
                }
                const unsigned packed = __float_as_uint(f0.x);
                const unsigned la = packed & 31u;
                const unsigned lb = (packed >> 5u) & 31u;
                const float4 ea = s_t0[gtid + la], eb = s_t0[gtid + lb];
                const float4 fa_ = s_t1[gtid + la], fb_ = s_t1[gtid + lb];
                const float a_px = ea.x, a_py = ea.y, a_qx = ea.z, a_qy = ea.w;
                const float a_vx = fa_.x, a_vy = fa_.y, a_wx = fa_.z, a_wy = fa_.w;
                const float b_px = eb.x, b_py = eb.y, b_qx = eb.z, b_qy = eb.w;
                const float b_vx = fb_.x, b_vy = fb_.y, b_wx = fb_.z, b_wy = fb_.w;
                float4 fa = make_float4(0.0f, 0.0f, 0.0f, 0.0f), fb = fa;
                if (mon) {
                    const float4 f1 = mrec[1], f2 = mrec[2], f3 = mrec[3];
                    const float anchor_a = f0.y, anchor_b = f0.z, mamp = f0.w;
                    const float hill = f1.x, inv_period = f1.y, phase = f1.z, duty = f1.w;
                    const float stiffness = f2.x, inv_duty = f2.y, inv_complement = f2.z;
                    const float cap = f3.x;
                    const float inv_capacity = f3.y * p.inv_muscle_energy;
                    const float tendon_k = f3.z, slack = f3.w;
                    const float pax = a_qx + (a_px - a_qx) * anchor_a, pay = a_qy + (a_py - a_qy) * anchor_a;
                    const float vax = a_wx + (a_vx - a_wx) * anchor_a, vay = a_wy + (a_vy - a_wy) * anchor_a;
                    const float pbx = b_qx + (b_px - b_qx) * anchor_b, pby = b_qy + (b_py - b_qy) * anchor_b;
                    const float vbx = b_wx + (b_vx - b_wx) * anchor_b, vby = b_wy + (b_vy - b_wy) * anchor_b;
                    const float dx = pbx - pax, dy = pby - pay;
                    const float length_m = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f);
                    const float inverse = 1.0f / length_m;
                    const float dirx = dx * inverse, diry = dy * inverse;
                    const float relative = (vbx - vax) * dirx + (vby - vay) * diry;
                    // The waveform's shortening speed over the substep; the
                    // waveform at the substep's start is kept from the last one.
                    const float w_now = wave(ts, inv_period, phase, s_off[r][tid], duty, inv_duty, inv_complement);
                    float w_prev = s_wp[r][tid];
                    if (w_prev < 0.0f) {
                        w_prev = wave(fmaxf(ts - HS, 0.0f), inv_period, phase, s_off[r][tid], duty, inv_duty, inv_complement);
                    }
                    s_wp[r][tid] = w_now;
                    const float target_speed = ts > 0.0f ? mamp * (w_now - w_prev) * INV_HS : 0.0f;
                    float drive = limp ? 0.0f : fmaxf(-target_speed * stiffness * 0.25f, 0.0f) * s_en[r][tid];
                    if (hill > 0.0f) {
                        drive *= clampf(1.0f + relative * hill, 0.0f, 1.0f);
                    }
                    const float magnitude = clampf(drive + relative * 0.15f, -cap, cap);
                    const float work = fminf(drive, cap) * fmaxf(-relative, 0.0f) * HS;
                    s_en[r][tid] = clampf(s_en[r][tid] - work * inv_capacity
                        + MUSCLE_RECOVERY * p.muscle_recovery * HS * (1.0f - s_en[r][tid]), 0.0f, 1.0f);
                    s_mag[r][tid] = magnitude;
                    const float stretch = fmaxf(length_m - slack, 0.0f);
                    const float tendon = tendon_k * stretch;
                    tpull[r] = tendon;
                    // First law: the muscle's work is its force times its
                    // shortening; the tendon's store counts as energy.
                    ledger -= magnitude * length_m + 0.5f * tendon * stretch;
                    scale += 0.5f * tendon * stretch;
                    const float pull = magnitude + tendon;
                    const float fx = dirx * pull, fy = diry * pull;
                    const vec3 ga = force_at(pax - ox, pay - oy, fx, fy);
                    const vec3 gb = force_at(pbx - ox, pby - oy, fx, fy);
                    fa = make_float4(ga.x, ga.y, ga.z, 0.0f);
                    fb = make_float4(-gb.x, -gb.y, -gb.z, 0.0f);
                }
                mf[2u * lg] = fa;
                mf[2u * lg + 1u] = fb;
                __syncwarp();
                // Each lane adds up the forces on a few ends (a chunk of its
                // own bone's ends, or of a bone with more ends than a lane
                // takes) from the words of ends it holds; the lane of a bone
                // with more adds the sums of the lanes that took the rest,
                // with shuffles (warp_kernel::plan_round).
                const unsigned nsum = ew > 0u ? ew - 1u : 0u;
                const unsigned* elist = ends + ebase + (unsigned)r * ew * W + lg;
                vec3 sum = v3(0.0f, 0.0f, 0.0f);
                for (unsigned e = 0u; e < maxsum; e++) {
                    const unsigned word = (e < nsum && (unsigned)r < rounds) ? elist[e * W] : 0xffffffffu;
#pragma unroll
                    for (unsigned b = 0u; b < 4u; b++) {
                        const unsigned slot = (word >> (8u * b)) & 255u;
                        const float4 f = mf[slot & (2u * W - 1u)];
                        // A select, not a product: a slot nobody wrote may hold
                        // NaN bits, and NaN times zero is NaN.
                        if (slot != 255u) { sum += v3(f.x, f.y, f.z); }
                    }
                }
                __syncwarp();
                const unsigned srcs = ((unsigned)r < rounds && nsum > 0u) ? elist[nsum * W] : 0u;
                const unsigned nsrc = (srcs >> 25u) & 7u;
                const unsigned maxsrc = __reduce_max_sync(FULL, nsrc);
                vec3 mine = ((srcs >> 28u) & 1u) != 0u ? sum : v3(0.0f, 0.0f, 0.0f);
                for (unsigned k = 0u; k < maxsrc; k++) {
                    const vec3 other = shv(sum, (srcs >> (5u * k)) & 31u);
                    if (k < nsrc) { mine += other; }
                }
                fm += mine;
            }
            bs -= fm;

            PROF(5);
            // Articulated-body pass, children first. Joint damping and the joint
            // limits' inelastic stops are implicit in each joint's inertia.
            vec3 uvs = v3(0.0f, 0.0f, 0.0f);
            // The joint's velocity-product acceleration.
            const vec3 cv = crm(v3(om, pvx + om * army, pvy - om * armx), axis) * qd;
            float dis = 0.0f, uus = 0.0f;
            // Damping and a limit stop that engages are implicit terms that
            // scale with the joint's inertia d: the joint's inertia becomes
            // d * jf and its torque -d * jg.
            float jf = 1.0f + INV_JOINT_DAMPING * HS;
            float jg = INV_JOINT_DAMPING * qd;
            {
                const float predicted = q + HS * qd;
                const bool upper = predicted > hi;
                if (upper || predicted < lo) {
                    const float room = upper ? hi - q : lo - q;
                    const bool past = (room < 0.0f) == upper;
                    const float goal = (past ? room * PUSH_OUT : room) * INV_HS;
                    if ((qd > goal) == upper) {
                        jg += LIMIT_HARDNESS * jf * INV_HS * (qd - goal);
                        jf *= 1.0f + LIMIT_HARDNESS;
                    }
                }
            }
            for (unsigned L = maxlev; L >= 2u; L--) {
                if (lvl == L) {
                    const vec3 uv = sym_mul(i0, i1, axis);
                    const float d = sdot(axis, uv);
                    const float u = -d * jg - sdot(axis, bs);
                    const float di = 1.0f / (d * jf);
                    uvs = uv; dis = di; uus = u;
                    const float kx = -di * uv.x, ky = -di * uv.y, kz = -di * uv.z;
                    const vec3 c0 = i0 + v3(kx * uv.x, kx * uv.y, kx * uv.z);
                    const vec3 c1 = i1 + v3(ky * uv.y, ky * uv.z, kz * uv.z);
                    const vec3 cp = bs + sym_mul(c0, c1, cv) + uv * (u * di);
                    region[3u * lane] = make_float4(c0.x, c0.y, c0.z, c1.x);
                    region[3u * lane + 1u] = make_float4(c1.y, c1.z, cp.x, cp.y);
                    reinterpret_cast<float*>(region + 3u * lane + 2u)[0] = cp.z;
                }
                __syncwarp();
                const unsigned cm = __reduce_max_sync(FULL, lvl == L - 1u ? nch : 0u);
                for (unsigned k = 0u; k < cm; k++) {
                    const bool take = lvl == L - 1u && k < nch;
                    const unsigned src = gbase + ((fc + k) & (W - 1u));
                    const float4 e0 = region[3u * src], e1 = region[3u * src + 1u];
                    const float e2 = reinterpret_cast<const float*>(region + 3u * src + 2u)[0];
                    if (take) {
                        i0 += v3(e0.x, e0.y, e0.z);
                        i1 += v3(e0.w, e1.x, e1.y);
                        bs += v3(e1.z, e1.w, e2);
                    }
                }
            }
            PROF(6);
            // The neck floats freely: the root's inverse articulated inertia.
            vec3 r0 = v3(0.0f, 0.0f, 0.0f), r1 = r0;
            vec3 acc = v3(0.0f, 0.0f, 0.0f);
            float qdd = 0.0f;
            if (lg == 1u && valid) {
                const float c00 = i1.x * i1.z - i1.y * i1.y;
                const float c01 = i0.z * i1.y - i0.y * i1.z;
                const float c02 = i0.y * i1.y - i0.z * i1.x;
                const float inv_det = 1.0f / (i0.x * c00 + i0.y * c01 + i0.z * c02);
                const float c11 = i0.x * i1.z - i0.z * i0.z;
                const float c12 = i0.y * i0.z - i0.x * i1.y;
                const float c22 = i0.x * i1.x - i0.y * i0.y;
                r0 = v3(c00 * inv_det, c01 * inv_det, c02 * inv_det);
                r1 = v3(c11 * inv_det, c12 * inv_det, c22 * inv_det);
                acc = -sym_mul(r0, r1, bs);
            }
            for (unsigned L = 2u; L <= maxlev; L++) {
                const vec3 pa = shv(acc, pb);
                if (lvl == L) {
                    const vec3 a = pa + cv;
                    qdd = (uus - sdot(uvs, a)) * dis;
                    acc = a + axis * qdd;
                }
            }

            PROF(7);
            // Ground contacts at velocity level: the deepest MAXC nodes that
            // would reach the ground within the substep, solved together.
            nc = 0u;
            float ln = 0.0f, lt = 0.0f;
            vec3 dn = v3(0.0f, 0.0f, 0.0f), dtg = dn;
#if GROUND
            {
                const vec3 nacc = shv(acc, 1u);
                const float nom = shf(om, 1u);
                const vec3 ba = lg == 0u ? nacc : acc;
                const float bw = lg == 0u ? nom : om;
                float nx = 0.0f, ny = 1.0f, gap = 0.0f, dry = 0.0f, vnf = 0.0f, depth_c = 0.0f;
                bool cand = false;
                if (valid) {
                    const float2 g = terrain(px);
                    const float secant = sqrtf(1.0f + g.y * g.y);
                    nx = -g.y / secant;
                    ny = 1.0f / secant;
                    dry = (py - g.x) / secant - rad;
#if MUD
                    gap = dry + p.mud;
#else
                    gap = dry;
#endif
                    dn = force_at(px - ox, py - oy, nx, ny);
                    vnf = vx * nx + vy * ny + HS * (sdot(dn, ba) + bw * (-vy * nx + vx * ny));
                    depth_c = gap + HS * vnf;
                    cand = depth_c <= 0.0f;
                }
                const unsigned cb = (__ballot_sync(FULL, cand) >> gshift) & GM;
                const unsigned ncand = __popc(cb);
                int slot = cand ? (int)__popc(cb & below) : -1;
                if (__any_sync(FULL, ncand > MAXC)) {
                    // Keep the deepest: MAXC rounds of a group maximum.
                    unsigned key = cand ? ((__float_as_uint(fmaxf(-depth_c, 0.0f)) & ~63u) | 32u | (31u - lg)) : 0u;
                    int rank = -1;
#pragma unroll
                    for (int r = 0; r < MAXC; r++) {
                        const unsigned top = gmaxu(key);
                        if (key != 0u && key == top) { rank = r; key = 0u; }
                    }
                    if (ncand > MAXC) { slot = rank; }
                }
                nc = min(ncand, (unsigned)MAXC);
                // Each slot's lane, five bits per slot.
                unsigned slot_lanes = 0u;
#pragma unroll
                for (int s = 0; s < MAXC; s++) {
                    const unsigned b = (__ballot_sync(FULL, slot == s) >> gshift) & GM;
                    slot_lanes |= (b ? (unsigned)(__ffs(b) - 1) : 0u) << (5 * s);
                }
                const bool walker = slot >= 0;
                const unsigned ncmax = __reduce_max_sync(FULL, nc);
                if (ncmax > 0u) {
                    float vn = 0.0f, vt = 0.0f, vs = 0.0f, goal = 0.0f, mu = 0.0f;
                    if (walker) {
                        const float tx = ny, ty = -nx;
                        dtg = force_at(px - ox, py - oy, tx, ty);
                        vn = vnf;
                        vt = vx * tx + vy * ty + HS * (sdot(dtg, ba) + bw * (-vy * tx + vx * ty));
                        vs = vx * tx + vy * ty;
                        goal = gap >= 0.0f ? -gap * INV_HS : -gap * PUSH_OUT * INV_HS;
#if MUD
                        const float sink = clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH);
                        mu = fric * p.friction * (1.0f + MUD_GRIP * sink) * (1.0f + MUD_NORMAL * sink);
#else
                        mu = fric * p.friction;
#endif
#if ICE
                        mu *= 1.0f - p.patches * ice_at(px);
#endif
                    } else {
                        dn = v3(0.0f, 0.0f, 0.0f);
                    }
                    PROF(8);
                    // Contact-space matrix: each walker climbs from its body to
                    // the root. At every joint it passes, the unit forces of its
                    // two rows reach the joint as t = -s.p; two walkers at the
                    // same joint add dinv t_r t_c, and at the root F_r' Phi F_c.
                    // A walker keeps its two rows in registers.
                    __syncwarp();
                    s_t0[tid] = make_float4(armx, army, dis, __uint_as_float(topo));
                    s_t1[tid] = make_float4(uvs.x, uvs.y, uvs.z, 0.0f);
                    if (lg == 1u) {
                        s_root[tid / W][0] = make_float4(r0.x, r0.y, r0.z, 0.0f);
                        s_root[tid / W][1] = make_float4(r1.x, r1.y, r1.z, 0.0f);
                    }
                    __syncwarp();
                    unsigned cur = walker ? (lg == 0u ? 1u : lg) : lg;
                    // The bones on each walker's path to the root.
                    unsigned pc[MAXC];
                    float4 kr[MAXC];
                    // The torques at this lane's joint of the walkers that pass it.
                    float tjn[MAXC], tjt[MAXC];
                    {
                        const unsigned path = walker ? s_anc[gtid + cur] : 0u;
#pragma unroll
                        for (int c = 0; c < MAXC; c++) {
                            const unsigned v = shu(path, (slot_lanes >> (5 * c)) & 31u);
                            pc[c] = (unsigned)c < nc ? v : 0u;
                            kr[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                            tjn[c] = 0.0f;
                            tjt[c] = 0.0f;
                        }
                    }
                    vec3 pn = -dn, pt = -dtg;
                    // Levels above every walker's body hold no work.
                    const unsigned wtop = __reduce_max_sync(FULL, walker ? ((__float_as_uint(s_t0[tid].w) >> 10u) & 31u) : 0u);
                    for (unsigned L = wtop; L >= 2u; L--) {
                        const float4 ja = s_t0[gtid + cur];
                        const float4 ju = s_t1[gtid + cur];
                        const unsigned jt = __float_as_uint(ja.w);
                        const bool act = walker && ((jt >> 10u) & 31u) == L;
                        float tn = 0.0f, tt = 0.0f;
                        unsigned jj = 0u;
                        if (act) {
                            const vec3 axj = v3(1.0f, ja.y, -ja.x);
                            tn = -sdot(axj, pn);
                            tt = -sdot(axj, pt);
                            const vec3 u = v3(ju.x, ju.y, ju.z);
                            pn += u * (tn * ja.z);
                            pt += u * (tt * ja.z);
                            jj = cur;
                            cur = (jt >> 5u) & 31u;
                        }
                        const float jd = ja.z;
                        const bool here = lvl == L;
#pragma unroll
                        for (int c = 0; c < MAXC; c++) {
                            if ((unsigned)c >= ncmax) { break; }
                            const unsigned src = (slot_lanes >> (5 * c)) & 31u;
                            const float tnc = shf(tn, src), ttc = shf(tt, src);
                            const float wd = (act && ((pc[c] >> jj) & 1u)) ? jd : 0.0f;
                            const float wn = wd * tn, wt = wd * tt;
                            kr[c].x += wn * tnc;
                            kr[c].y += wn * ttc;
                            kr[c].z += wt * tnc;
                            kr[c].w += wt * ttc;
                            if (here && ((pc[c] >> lg) & 1u)) { tjn[c] = tnc; tjt[c] = ttc; }
                        }
                    }
                    float knn = 1.0f, ktt = 1.0f;
                    {
                        const float4 q0 = s_root[tid / W][0], q1 = s_root[tid / W][1];
                        const vec3 f0 = v3(q0.x, q0.y, q0.z), f1 = v3(q1.x, q1.y, q1.z);
                        const vec3 phn = sym_mul(f0, f1, pn), pht = sym_mul(f0, f1, pt);
#pragma unroll
                        for (int c = 0; c < MAXC; c++) {
                            if ((unsigned)c >= ncmax) { break; }
                            const unsigned src = (slot_lanes >> (5 * c)) & 31u;
                            const vec3 qn = shv(pn, src), qt = shv(pt, src);
                            kr[c].x = (kr[c].x + sdot(phn, qn)) * HS;
                            kr[c].y = (kr[c].y + sdot(phn, qt)) * HS;
                            kr[c].z = (kr[c].z + sdot(pht, qn)) * HS;
                            kr[c].w = (kr[c].w + sdot(pht, qt)) * HS;
                            if (slot == c) { knn = kr[c].x; ktt = kr[c].w; }
                        }
                    }
                    const float inv_knn = 1.0f / knn, inv_ktt = 1.0f / ktt;
                    // The joint torques wait in shared memory during the solve.
                    region[3u * lane] = make_float4(tjn[0], tjn[1], tjn[2], tjn[3]);
                    region[3u * lane + 1u] = make_float4(tjt[0], tjt[1], tjt[2], tjt[3]);
                    PROF(9);
                    // Projected Gauss-Seidel. Walker c holds rows 2c and 2c + 1;
                    // every impulse change is broadcast to the other walkers.
                    // Every lane computes the owner's update; only the owner
                    // keeps it.
#pragma unroll
                    for (unsigned sweep = 0u; sweep < PGS_SWEEPS + CLEAN_SWEEPS; sweep++) {
                        const bool clean = sweep >= PGS_SWEEPS;
#pragma unroll
                        for (int c = 0; c < MAXC; c++) {
                            if ((unsigned)c >= ncmax) { break; }
                            const unsigned src = (slot_lanes >> (5 * c)) & 31u;
                            const bool own = slot == c;
                            const float4 k = kr[c];
                            // The owner updates its normal, then its friction
                            // against its own new speed, and broadcasts both.
                            const float normal = clean ? ln : fmaxf(fmaf(goal - vn, inv_knn, ln), 0.0f);
                            float dn_ = own ? normal - ln : 0.0f;
                            ln += dn_;
                            const float vt1 = vt + k.z * dn_;
                            // Friction may not do positive work: it only
                            // opposes a = start speed + end speed without
                            // its own force, and only up to |a| / k.
                            const float a = vs + vt1 - ktt * lt;
                            const float cap = fminf(mu * ln, fabsf(a) * inv_ktt);
                            const float tgt = clean ? lt : lt - vt1 * inv_ktt;
                            const float friction = clampf(tgt, a > 0.0f ? -cap : 0.0f, a > 0.0f ? 0.0f : cap);
                            float dt_ = own ? friction - lt : 0.0f;
                            lt += dt_;
                            dn_ = shf(dn_, src);
                            dt_ = shf(dt_, src);
                            vn += k.x * dn_ + k.y * dt_;
                            vt += k.z * dn_ + k.w * dt_;
                        }
                    }
                    if (!walker) { ln = 0.0f; lt = 0.0f; }
                    PROF(10);
                    // The response to the contact impulses through the same
                    // articulated inertias: the root takes the walkers' unit
                    // forces there, and every joint the torques of the walkers
                    // that pass it.
                    {
                        const float4 en = region[3u * lane], et = region[3u * lane + 1u];
                        const float tn4[MAXC] = {en.x, en.y, en.z, en.w};
                        const float tt4[MAXC] = {et.x, et.y, et.z, et.w};
                        float dqr = 0.0f;
#pragma unroll
                        for (int c = 0; c < MAXC; c++) {
                            if ((unsigned)c >= ncmax) { break; }
                            const unsigned src = (slot_lanes >> (5 * c)) & 31u;
                            const float lnc = shf(ln, src), ltc = shf(lt, src);
                            dqr += lnc * tn4[c] + ltc * tt4[c];
                        }
                        vec3 pr = walker ? pn * ln + pt * lt : v3(0.0f, 0.0f, 0.0f);
                        pr = v3(gsum(pr.x), gsum(pr.y), gsum(pr.z));
                        vec3 ar = v3(0.0f, 0.0f, 0.0f);
                        if (lg == 1u && valid) {
                            const float4 q0 = s_root[tid / W][0], q1 = s_root[tid / W][1];
                            ar = -sym_mul(v3(q0.x, q0.y, q0.z), v3(q1.x, q1.y, q1.z), pr);
                            acc += ar;
                        }
                        for (unsigned L = 2u; L <= maxlev; L++) {
                            const vec3 pa = shv(ar, pb);
                            if (lvl == L) {
                                const float tq = (dqr - sdot(uvs, pa)) * dis;
                                ar = pa + axis * tq;
                                qdd += tq;
                            }
                        }
                    }
                    if (walker) {
                        mx += (ln * dn.y + lt * dtg.y) * HS;
                        my += (ln * dn.z + lt * dtg.z) * HS;
                    }
                }
            }
#endif
            step_n += ln;
            step_t += lt;
            PROF(11);
            // Semi-implicit Euler on the joint coordinates.
            {
                const vec3 ra = shv(acc, 1u);
                const float rqd = shf(qd, 1u);
                if (lg == 0u) {
                    const float hax = ra.y - rqd * vy;
                    const float hay = ra.z + rqd * vx;
#if AIR
                    vx = (vx + hax * HS) * p.air_sub;
                    vy = (vy + hay * HS) * p.air_sub;
#else
                    vx = vx + hax * HS;
                    vy = vy + hay * HS;
#endif
                    px += vx * HS;
                    py += vy * HS;
                } else if (body) {
                    const float a = lg == 1u ? acc.x : qdd;
#if AIR
                    qd = (qd + a * HS) * p.air_sub;
#else
                    qd = qd + a * HS;
#endif
                    q += qd * HS;
                }
            }
        }
        if (lg == 0u) { q = 0.0f; }
        if (lg == 1u) { q = q - TAU_F * floorf((q + PI_F) / TAU_F); }
        rec_n = step_n * (1.0f / SUBSTEPS);
        rec_t = step_t * (1.0f / SUBSTEPS);

        PROF(12);
        // Metrics, falls and the screen, once per step. Every lane works on its
        // own copy of the group's totals and lane 0 writes them back, so lanes
        // of one group that run apart never apply a change twice.
        {
            __shared__ uint4 s_rung[BLOCK / W];
            __shared__ unsigned s_rb[BLOCK / W];
            __syncwarp();
            Result mt = s_mt[tid / W];
            const uint4 bits0 = s_bits[tid / W];
            unsigned contact_bits = bits0.x, lift_bits = bits0.y, ground_bits = bits0.z;
            float head_shake = __uint_as_float(bits0.w);
            uint4 rung = s_rung[tid / W];
            unsigned rbits = s_rb[tid / W];
            const bool bad = valid && !(fabsf(px) <= 1e6f && fabsf(py) <= 1e6f);
            const bool failed = ((__ballot_sync(FULL, bad) >> gshift) & GM) != 0u;
            const float center_y = gsum(valid ? py : 0.0f) * inv_nodes;
            const float com_x = gsum(valid ? m * px : 0.0f) * inv_mass;
            const float low = gmin(valid ? py - rad : 1e20f);
            const float high = gmax(valid ? py + rad : -1e20f);
            bool touch = false, clear = false;
#if GROUND
            if (valid) {
                const float2 g = terrain(px);
                const float floor_y = g.x + rad * sqrtf(1.0f + g.y * g.y);
                touch = py <= floor_y + CONTACT_SLACK;
                clear = py > floor_y + LIFT_CLEARANCE;
            }
#endif
            const unsigned now = (__ballot_sync(FULL, touch) >> gshift) & GM;
            const unsigned lifted = (__ballot_sync(FULL, clear) >> gshift) & GM;
            contact_bits |= now;
            lift_bits |= contact_bits & lifted;
            const unsigned down = now & ~ground_bits;
            ground_bits = now;
            // Touchdowns restart the rhythm of the muscles that sense them.
            if (down != 0u && step > 0u) {
                const float next = t_now + DT;
#pragma unroll
                for (int r = 0; r < MAXR; r++) {
                    const unsigned mi = (unsigned)r * W + lg;
                    if ((unsigned)r < rounds && mi < nmus) {
                        const float4* mrec = reinterpret_cast<const float4*>(muscles + mbase) + mi * 4u;
                        const unsigned packed = __float_as_uint(mrec[0].x);
                        if ((packed >> 15u) & 1u) {
                            const unsigned sensor = (packed >> 10u) & 31u;
                            if ((down >> sensor) & 1u) {
                                const float4 f1 = mrec[1];
                                const float clock = next * f1.y + f1.z;
                                const float x = mrec[2].w - clock;
                                s_off[r][tid] = x - floorf(x);
                                s_wp[r][tid] = -1.0f;
                            }
                        }
                    }
                }
            }
            {
                const float dvx = shf(vx, 0u) - head_vx0, dvy = shf(vy, 0u) - head_vy0;
                if (t_now >= HEAD_SHAKE_WINDOW) {
                    const float accel = sqrtf(dvx * dvx + dvy * dvy) * RATE;
                    head_shake += (accel - head_shake) * fminf(1.0f / (HEAD_SHAKE_WINDOW * RATE), 1.0f);
                }
            }
            const bool broken_lane = body && lg >= 2u && (q < lo - JOINT_BREAK || q > hi + JOINT_BREAK);
            const bool broken = ((__ballot_sync(FULL, broken_lane) >> gshift) & GM) != 0u;
            const float head_y = shf(py, 0u), neck_y = shf(py, 1u);
            // The rung trace (creature_kernel::RungTrace): the distance at 1,
            // 2.5, 5 and 10 s and the early features at 1 and 2.5 s, as fp16
            // pairs in the seven result words the host reads for nothing
            // else. It records; it stops no trial. During the trial the
            // speeds and the contact and energy pairs wait in `s_rung` (.y,
            // .z, .w; .x is the distance half a second before a rung) and the
            // head shake pair in `ground_hi`.
            // The early rungs' record of the trial, in the end code's bit
            // positions: the rung that stopped it (bits 6 and 7) and the
            // cadence band at 1 and 2.5 s (bits 8 to 13).
            // The creature's last head word, read when a rule needs it: its
            // rhythm period (a rung feature) in the low half and its flags in
            // the high half (an audit creature runs every rule off, an exempt
            // one, a nursery's or an immigrant's, skips the early rungs).
            auto head_flags = [&]() -> unsigned { return heads[2u * cidx + 1u].w; };
            auto h16 = [](float v) -> unsigned {
                unsigned short r;
                asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(v));
                return (unsigned)r;
            };
            const unsigned rung1 = (unsigned)(RATE) - 1u, rung2 = (unsigned)(2.5f * RATE) - 1u;
            const unsigned rung3 = (unsigned)(5.0f * RATE) - 1u, rung4 = (unsigned)(10.0f * RATE) - 1u;
            const unsigned half_s = (unsigned)(0.5f * RATE);
            const bool early = step == rung1 || step == rung2;
            float en_mean = 0.0f;
            if (__any_sync(FULL, early)) {
                float en = 0.0f;
#pragma unroll
                for (int r = 0; r < MAXR; r++) {
                    const unsigned mi = (unsigned)r * W + lg;
                    if ((unsigned)r < rounds && mi < nmus) { en += s_en[r][tid]; }
                }
                en_mean = gsum(en) / (float)max(nmus, 1u);
            }
            if (live
#if RECORD
                && !done_scoring
#endif
            ) {
                mt.contact_lo = __uint_as_float(contact_bits);
                mt.lift_lo = __uint_as_float(lift_bits);
                mt.ground_lo = __uint_as_float(ground_bits);
                mt.head_shake = head_shake;
                const bool fell = head_y < neck_y || broken || head_shake > HEAD_SHAKE_LIMIT || failed;
                bool ended = false;
                if (fell) {
                    mt.fall_time = t_now + DT;
                    mt.fitness = failed ? -1e20f : com_x;
                    if (step <= p.screen_step) { mt.screen_x = mt.fitness; }
                    ended = true;
                }
                mt.ground_contact += (float)__popc(now);
                mt.height_sum += high - low;
                mt.vertical_oscillation = fminf(mt.vertical_oscillation, center_y);
                mt.gait_frequency = fmaxf(mt.gait_frequency, center_y);
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
                if (step == 0u) { rung = make_uint4(0u, 0u, 0u, 0u); rbits = 0u; }
                if (step + half_s == rung1 || step + half_s == rung2) { rung.x = __float_as_uint(com_x); }
                if (early) {
                    const bool second = step == rung2;
                    const unsigned sh = second ? 16u : 0u;
                    const unsigned keep = second ? 0x0000ffffu : 0xffff0000u;
                    const float speed = (com_x - __uint_as_float(rung.x)) * (RATE / (float)half_s);
                    const float touched = (float)__popc(contact_bits) * inv_nodes;
                    mt.contact_hi = __uint_as_float((__float_as_uint(mt.contact_hi) & keep) | (h16(com_x) << sh));
                    mt.ground_hi = __uint_as_float((__float_as_uint(mt.ground_hi) & keep) | (h16(head_shake) << sh));
                    rung.y = (rung.y & keep) | (h16(speed) << sh);
                    const unsigned pair = h16(touched) | (h16(en_mean) << 16u);
                    if (second) { rung.w = pair; } else { rung.z = pair; }
                    // The cadence band the audit lane files this trial under:
                    // the live gait frequency in eight bins of 0 to 6 Hz, as
                    // the archive's cadence axis bins it.
                    const unsigned r = second ? 1u : 0u;
                    const float gait_now = mt.gait_turns * (0.5f / (t_now + DT));
                    const unsigned band = min((unsigned)(gait_now * (8.0f / 6.0f)), 7u);
                    rbits |= band << (8u + 3u * r);
                    const unsigned hw = head_flags();
                    // Flags: 1 audit, 2 exempt, 4 and 8 exempt from R1 and R2.
                    if (!ended && ((hw >> 16u) & (second ? 0xbu : 0x7u)) == 0u) {
                        const RungParams rp = second ? p.r2 : p.r1;
                        const float period_f = h2f(hw & 0xffffu);
                        // The decision reads the features as the trace stores
                        // them, so the host can replay it.
                        const float fv[6] = {h2f(h16(com_x)), h2f(h16(speed)), h2f(h16(touched)),
                                             h2f(h16(head_shake)), h2f(h16(en_mean)), period_f};
                        float score = 0.0f;
                        bool finite = true;
#pragma unroll
                        for (int i = 0; i < 6; i++) {
                            score = fmaf(rp.w[i], fv[i], score);
                            finite = finite && isfinite(fv[i]);
                        }
                        if (finite && score < rp.bias && ((rp.off >> band) & 1u) == 0u) {
                            mt.screened = t_now + DT;
                            mt.screen_x = com_x;
                            mt.fitness = com_x;
                            rbits |= (r + 1u) << 6u;
                            ended = true;
                        }
                    }
                }
                if (step == rung3 || step == rung4) {
                    const bool second = step == rung4;
                    const unsigned keep = second ? 0x0000ffffu : 0xffff0000u;
                    mt.lift_hi = __uint_as_float((__float_as_uint(mt.lift_hi) & keep) | (h16(com_x) << (second ? 16u : 0u)));
                }
                if (step == p.screen_step && !ended) {
                    mt.screen_x = com_x;
                    // Flag 16 holds a young creature to the young bar, flag 32
                    // a reshaped one to its own.
                    const unsigned flags = head_flags() >> 16u;
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
                    {
                        const unsigned fin = h16(mt.fitness);
                        unsigned d = __float_as_uint(mt.contact_hi);
                        if (step < rung1) { d = (d & 0xffff0000u) | fin; }
                        if (step < rung2) { d = (d & 0x0000ffffu) | (fin << 16u); }
                        mt.contact_hi = __uint_as_float(d);
                        d = __float_as_uint(mt.lift_hi);
                        if (step < rung3) { d = (d & 0xffff0000u) | fin; }
                        if (step < rung4) { d = (d & 0x0000ffffu) | (fin << 16u); }
                        mt.lift_hi = __uint_as_float(d);
                        const unsigned code = (fell ? 16u : 0u) | (failed ? 32u : 0u) | (mt.screened > 0.0f ? 3u : 0u)
                            | (rbits & 0x3fc0u) | (((head_flags() >> 16u) & 1u) << 14u);
                        mt.previous_center_y = __uint_as_float(rung.y);
                        mt.vertical_extremum = __uint_as_float(rung.z);
                        mt.vertical_trend = __uint_as_float(rung.w);
                        mt.gait_turns = mt.ground_hi;
                        mt.ground_hi = __uint_as_float(code | (min(step + 1u, 65535u) << 16u));
                    }
#if RECORD
                    kept = mt;
                    done_scoring = true;
                    limp = true;
#else
                    if (lg == 0u) { results[cidx] = mt; }
                    live = false;
#endif
                }
            }
            __syncwarp();
            if (lg == 0u) {
                s_mt[tid / W] = mt;
                s_bits[tid / W] = make_uint4(contact_bits, lift_bits, ground_bits, __float_as_uint(head_shake));
                s_rung[tid / W] = rung;
                s_rb[tid / W] = rbits;
            }
            __syncwarp();
        }
#if RECORD
        record_frame(SETTLE + step + 1u, live);
        if (live) {
            if (step + 1u >= p.steps) {
                if (lg == 0u) { results[cidx] = kept; }
                live = false;
            }
        }
#endif
        step += 1u;
    }
#if PROFILE
    PROF(0);
    if (blockIdx.x == 0u && (threadIdx.x >> 5) == 0u && lane == 0u) {
        const long long* c = s_prof[0];
        long long total = 0;
        for (int k = 0; k < 13; k++) { total += c[k]; }
        printf("profile W=%u cycles %lld: fetch %.1f%% kin %.1f%% balance %.1f%% setup %.1f%% muscles %.1f%% aba %.1f%% fwd %.1f%% detect %.1f%% kwalk %.1f%% pgs %.1f%% response %.1f%% integrate %.1f%% metrics %.1f%%\n",
            (unsigned)W, total, 100.0 * c[0] / total, 100.0 * c[1] / total, 100.0 * c[2] / total, 100.0 * c[3] / total,
            100.0 * c[4] / total, 100.0 * c[5] / total, 100.0 * c[6] / total, 100.0 * c[7] / total, 100.0 * c[8] / total,
            100.0 * c[9] / total, 100.0 * c[10] / total, 100.0 * c[11] / total, 100.0 * c[12] / total);
    }
#endif
}
