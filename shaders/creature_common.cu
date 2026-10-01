// Shared by shaders/warp_creature.cu and shaders/lane_creature.cu: constants,
// the parameter and result layouts, vector helpers and the ground.

#define FULL 0xffffffffu
#define GM ((W == 32) ? 0xffffffffu : ((1u << W) - 1u))
#define DT (1.0f / RATE)
#define HS (1.0f / (RATE * SUBSTEPS))
#define INV_HS (RATE * SUBSTEPS)
#define LF 12u
#define MUSCLE_DAMPER 0.15f
#define RMAX 4
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
    float water;
    float patches;
    float air_sub;
    float inv_muscle_energy;
    RungParams r1;
    RungParams r2;
};
// The take-up buckets of a wave (cuda_engine::Takeup): bucket b holds the
// wave's creatures from start[b] to end[b], sorted by muscle rounds, and
// warps from warp[b] on (in the order warp in block, then block) start on it.
struct Takeup {
    unsigned start[RMAX];
    unsigned end[RMAX];
    unsigned warp[RMAX];
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
// Height and slope of the ground under x for a creature with terrain amplitude
// `amp` and quake phase `qphase`; flat ground compiles to (0, 0).
__device__ __forceinline__ float2 ground_at(float x, const Params& p, float amp, float qphase) {
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
}
