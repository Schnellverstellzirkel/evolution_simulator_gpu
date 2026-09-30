// Stub of the per-lane maximal-coordinate kernel (docs/plan-2m.md, section 6
// item 4). It measures registers, spills, issue rate and speed of the design
// before anyone writes the physics; its numbers are not a physics.
//
// One creature per W = 2 lanes. Lane l owns nodes 4l .. 4l + 3 and the rods
// that end at them (rod j ends at node j + 1; node 0, the head, has none), so
// lane 0 holds the low rods and lane 1 the high ones. Rods are numbered
// breadth first: a rod's parent and its lower siblings have lower numbers,
// and siblings are consecutive.
//
// A substep: node table to shared memory; gravity, air drag and joint
// damping; the muscles (up to MPL per lane, split records read from L2,
// forces gathered through a private per-lane shared table in a fixed order);
// the free velocities; the rod matrix A = J M^-1 J^T factored as LDL^T in
// reverse rod order (no fill: the rods at a node form a clique), lane 1's
// rods first, their Schur updates sent to lane 0 by shuffles; the deepest 4
// nodes as contacts, 2 rows each; one batched tree solve of the rods' own
// row and the 8 contact rows, pipelined so lane 1 runs row r while lane 0
// runs row r - 1 (forward) and the reverse (backward), with 3 shuffles per
// row and pass; the 8 x 8 Delassus matrix W = W0 - R^T A^-1 R in shared
// memory; a direct active-set solve (LDL^T of the active rows, at most
// MAX_ROUNDS rounds, warm started), run redundantly in both lanes; one more
// tree solve for the rod impulses of the contact impulses; the momentum and
// angular-momentum ledgers; semi-implicit Euler. Once per step: the
// waveform and drive target of every muscle, a drift projection with contact
// nodes at infinite mass (a second factor and a tree solve), metrics.
//
// Defines (all have defaults): W, MPL, NB (rod neighbours kept per rod: the
// parent and NB - 1 lower siblings), BLOCK, MIN_BLOCKS, SUBSTEPS,
// MAX_ROUNDS, MSTATE_REGS, MUSCLE_UNROLL. Developer switches: NO_MUSCLES,
// NO_CONTACT, NO_DAMP, NO_FRICTION, NO_LEDGER, USE_STASH, DIAG (solve
// residuals in the result), TRACE, DUMP, DUMPW (see examples/lane_stub.rs).
//
// The synthetic bodies and the explicit forces are not tuned; a speed clamp
// (SPEED_CLAMP) keeps trials finite so every creature runs its full length.

#ifndef W
#define W 2
#endif
#ifndef MPL
#define MPL 16
#endif
#ifndef NB
#define NB 3
#endif
#ifndef BLOCK
#define BLOCK 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef SUBSTEPS
#define SUBSTEPS 2
#endif
#ifndef MAX_ROUNDS
#define MAX_ROUNDS 3
#endif
// 1: the muscle state words live in registers; 0: in an L2 scratch, read
// and written once per muscle and substep.
#ifndef MSTATE_REGS
#define MSTATE_REGS 0
#endif
// Unrolling of the loops over a lane's muscles (MPL with the state in
// registers, which index it).
#ifndef MUSCLE_UNROLL
#if MSTATE_REGS
#define MUSCLE_UNROLL MPL
#else
#define MUSCLE_UNROLL 1
#endif
#endif
// Developer switches for finding instabilities.
#ifndef NO_MUSCLES
#define NO_MUSCLES 0
#endif
#ifndef NO_CONTACT
#define NO_CONTACT 0
#endif
#ifndef NO_DAMP
#define NO_DAMP 0
#endif
#ifndef USE_STASH
#define USE_STASH 1
#endif
#ifndef NO_LEDGER
#define NO_LEDGER 0
#endif
#ifndef NO_FRICTION
#define NO_FRICTION 0
#endif
#if W != 2
#error "the stub implements the W = 2 exchange only"
#endif

#define NPL 4
#define RATE 60.0f
#define DT (1.0f / RATE)
#define HS (DT / SUBSTEPS)
#define INV_HS (RATE * SUBSTEPS)
#define NC 4
#define NE (2 * NC)
#define NR (NE + 1)
#define NS (NR + 1)
#define FULL 0xffffffffu
#define PI_F 3.14159265359f
#define PUSH_OUT 0.2f
#define AIR_DRAG 0.6f
#define INV_JOINT_DAMPING 10.0f
#define MUSCLE_RECOVERY 0.5f
#define COMPLIANCE 0.1f
#define RF 32
#define SPEED_CLAMP 20.0f

struct Params {
    unsigned count;
    unsigned steps;
    unsigned screen_step;
    unsigned spare;
    float gravity;
    float friction;
    float recovery;
    float screen_bar;
};

__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ __forceinline__ float xch(float v) { return __shfl_xor_sync(FULL, v, 1); }
__device__ __forceinline__ unsigned xchu(unsigned v) { return __shfl_xor_sync(FULL, v, 1); }
__device__ __forceinline__ float wave(float t, float inv_period, float phase, float offset, float duty, float inv_duty, float inv_complement) {
    const float x = t * inv_period + phase + offset;
    const float ph = x - floorf(x);
    const bool rise = ph < duty;
    const float arg = rise ? ph * inv_duty : (ph - duty) * inv_complement;
    const float half = rise ? 0.5f : -0.5f;
    return 0.5f + half * __cosf(PI_F * arg);
}
// Muscle state as one word per muscle: energy as unorm16 in the low half,
// the step's drive target as the high half of a float (bfloat16).
__device__ __forceinline__ unsigned pack_ed(float energy, float drive) {
    return (unsigned)__float2uint_rn(energy * 65535.0f) | ((__float_as_uint(drive) + 0x8000u) & 0xffff0000u);
}
__device__ __forceinline__ float2 unpack_ed(unsigned u) {
    return make_float2((float)(u & 0xffffu) * (1.0f / 65535.0f), __uint_as_float(u & 0xffff0000u));
}
// Two unorm16 values in [0, 1] (muscle anchors).
__device__ __forceinline__ float2 unpack_un(unsigned u) {
    return make_float2((float)(u & 0xffffu) * (1.0f / 65535.0f), (float)(u >> 16u) * (1.0f / 65535.0f));
}
// Two bfloat16 values (Hill factor, inverse capacity).
__device__ __forceinline__ float2 unpack_bf(unsigned u) {
    return make_float2(__uint_as_float(u << 16u), __uint_as_float(u & 0xffff0000u));
}

// Rod topology word: pivot node (5 bits), parent rod (5, 31 none), lower
// siblings kept (2), grandparent node (5, the parent's pivot, 31 none),
// valid (1).
#define T_PIVOT(t) ((t) & 31u)
#define T_PARENT(t) (((t) >> 5u) & 31u)
#define T_SR(t) (((t) >> 10u) & 3u)
#define T_GP(t) (((t) >> 12u) & 31u)
#define T_VALID(t) (((t) >> 17u) & 1u)

// One-hot target of a rod's parent: bit q for local rod q, bit 4 + q for
// lane 0's rod q seen from lane 1, 0 without a parent. Bit tests instead of
// index compares keep the compiler from turning register arrays into
// indexed local memory.
__device__ __forceinline__ unsigned parent_mask(unsigned t, unsigned lg) {
    const unsigned p = T_PARENT(t);
    const unsigned tn = p + 1u;
    const bool hasp = T_VALID(t) && p != 31u;
    return hasp ? (((tn >> 2u) != lg) ? (0x10u << (tn & 3u)) : (1u << (tn & 3u))) : 0u;
}

// The LDL^T factor of A = J M^-1 J^T over the creature's rods. `ipv` is the
// inverse mass of each local rod's pivot, `icm` of its child node; `rdx`,
// `rdy` hold lane 0's rod directions on lane 1.
__device__ __forceinline__ void factor(
    const unsigned (&topo)[NPL], const float (&dx)[NPL], const float (&dy)[NPL],
    const float (&rdx)[NPL], const float (&rdy)[NPL], const float (&ipv)[NPL], const float (&icm)[NPL],
    unsigned lg, float (&invD)[NPL], float (&Lf)[NPL][NB]) {
    float Dacc[NPL], La[NPL][NB];
#pragma unroll
    for (int k = 0; k < NPL; k++) {
        const unsigned t = topo[k];
        const bool valid = T_VALID(t);
        Dacc[k] = valid ? ipv[k] + icm[k] : 1.0f;
        const unsigned pm = parent_mask(t, lg);
        float pdx = 0.0f, pdy = 0.0f;
#pragma unroll
        for (int q = 0; q < k; q++) {
            if (pm & (1u << q)) { pdx = dx[q]; pdy = dy[q]; }
        }
#pragma unroll
        for (int q = 1; q < NPL; q++) {
            if (pm & (0x10u << q)) { pdx = rdx[q]; pdy = rdy[q]; }
        }
        La[k][0] = -(dx[k] * pdx + dy[k] * pdy) * ipv[k];
        const unsigned sr = valid ? T_SR(t) : 0u;
#pragma unroll
        for (int a = 1; a < NB; a++) {
            const float sdx = (k - a >= 0) ? dx[(k - a) & 3] : rdx[(4 + k - a) & 3];
            const float sdy = (k - a >= 0) ? dy[(k - a) & 3] : rdy[(4 + k - a) & 3];
            La[k][a] = (unsigned)a <= sr ? (dx[k] * sdx + dy[k] * sdy) * ipv[k] : 0.0f;
        }
    }
    float oD[NPL], oL[NPL][NB];
#pragma unroll
    for (int q = 0; q < NPL; q++) {
        oD[q] = 0.0f;
#pragma unroll
        for (int a = 0; a < NB; a++) { oL[q][a] = 0.0f; }
    }
#pragma unroll
    for (int phase = 0; phase < 2; phase++) {
        if (phase == 1) {
            // Lane 1's Schur updates of lane 0's rods.
#pragma unroll
            for (int q = 1; q < NPL; q++) {
                Dacc[q] += xch(oD[q]);
#pragma unroll
                for (int a = 0; a < NB; a++) { La[q][a] += xch(oL[q][a]); }
            }
        }
        if (lg == 1u - (unsigned)phase) {
#pragma unroll
            for (int k = NPL - 1; k >= 0; k--) {
                const unsigned t = topo[k];
                const bool valid = T_VALID(t);
                // A rod whose two nodes are both held (projection) has no
                // pivot: it is left as it is.
                const float id = (valid && Dacc[k] > 1e-6f) ? 1.0f / Dacc[k] : 0.0f;
                invD[k] = id;
#pragma unroll
                for (int a = 0; a < NB; a++) { Lf[k][a] = La[k][a] * id; }
                const unsigned pm = parent_mask(t, lg);
                const float vp = La[k][0] * Lf[k][0];
#pragma unroll
                for (int q = 0; q < k; q++) {
                    if (pm & (1u << q)) { Dacc[q] -= vp; }
                }
#pragma unroll
                for (int q = 1; q < NPL; q++) {
                    if (pm & (0x10u << q)) { oD[q] -= vp; }
                }
#pragma unroll
                for (int a = 1; a < NB; a++) {
                    const float va = La[k][a] * Lf[k][a];
                    const float vpa = La[k][a] * Lf[k][0];
                    if (k - a >= 0) {
                        Dacc[(k - a) & 3] -= va;
                        La[(k - a) & 3][0] -= vpa;
                    } else {
                        oD[(4 + k - a) & 3] -= va;
                        oL[(4 + k - a) & 3][0] -= vpa;
                    }
#pragma unroll
                    for (int b = 1; b < a; b++) {
                        const float vab = La[k][a] * Lf[k][b];
                        if (k - b >= 0) {
                            La[(k - b) & 3][a - b] -= vab;
                        } else {
                            oL[(4 + k - b) & 3][a - b] -= vab;
                        }
                    }
                }
            }
        }
    }
}

// Solves A x = z in place for S - 1 rows at once. Row r sits in slot r on
// lane 1 and in slot r + 1 on lane 0, so the forward pass (lane 1 ahead) and
// the backward pass (lane 0 ahead, slots in reverse) both run one slot per
// step in both lanes, with no selects and 3 shuffles per slot and pass.
template <int S>
__device__ __forceinline__ void tree_solve(float (&z)[S][NPL], const unsigned (&topo)[NPL],
    const float (&invD)[NPL], const float (&Lf)[NPL][NB], unsigned lg) {
    float inb[NPL] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < S; s++) {
        float b[NPL];
#pragma unroll
        for (int q = 0; q < NPL; q++) { b[q] = z[s][q] + inb[q]; }
        float out[NPL] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = NPL - 1; k >= 0; k--) {
            const unsigned pm = parent_mask(topo[k], lg);
            const float y = b[k];
            const float ly = Lf[k][0] * y;
#pragma unroll
            for (int q = 0; q < k; q++) {
                if (pm & (1u << q)) { b[q] -= ly; }
            }
#pragma unroll
            for (int q = 1; q < NPL; q++) {
                if (pm & (0x10u << q)) { out[q] -= ly; }
            }
#pragma unroll
            for (int a = 1; a < NB; a++) {
                const float v = Lf[k][a] * y;
                if (k - a >= 0) { b[(k - a) & 3] -= v; } else { out[(4 + k - a) & 3] -= v; }
            }
            z[s][k] = y * invD[k];
        }
#pragma unroll
        for (int q = 1; q < NPL; q++) { inb[q] = xch(out[q]); }
    }
    float xin[NPL] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int it = 0; it < S; it++) {
        const int s = S - 1 - it;
        float x[NPL];
#pragma unroll
        for (int k = 0; k < NPL; k++) {
            const unsigned pm = parent_mask(topo[k], lg);
            float xp = 0.0f;
#pragma unroll
            for (int q = 0; q < k; q++) {
                if (pm & (1u << q)) { xp = x[q]; }
            }
#pragma unroll
            for (int q = 1; q < NPL; q++) {
                if (pm & (0x10u << q)) { xp = xin[q]; }
            }
            float acc = z[s][k] - Lf[k][0] * xp;
#pragma unroll
            for (int a = 1; a < NB; a++) {
                acc -= Lf[k][a] * ((k - a >= 0) ? x[(k - a) & 3] : xin[(4 + k - a) & 3]);
            }
            x[k] = acc;
            z[s][k] = acc;
        }
#pragma unroll
        for (int q = 1; q < NPL; q++) { xin[q] = xch(x[q]); }
    }
}

extern "C" __global__ void __launch_bounds__(BLOCK, MIN_BLOCKS) lane_stub(
    const uint4* __restrict__ heads,
    const float* __restrict__ lanes,
    const float4* __restrict__ msa,
    const float2* __restrict__ msb,
    const float4* __restrict__ mss,
    float* __restrict__ roff,
    unsigned* __restrict__ mstate,
    float* __restrict__ anch,
    float4* __restrict__ results,
    unsigned* __restrict__ counter,
    const Params p
#if defined(TRACE) || defined(DUMP) || defined(DUMPW)
    , volatile unsigned* trace
#endif
    ) {
    // 176 B per lane. NODE: each lane's nodes (position, velocity). FRC:
    // each lane's private force table over the creature's 8 nodes, then the
    // node statics for the contact rows, then the rod impulses. During the
    // contact solve the same bytes, as words SCR(q, lane column) with
    // q < 44, hold the creature's Delassus matrix (words 0 to 17 of both
    // lane columns) and each lane's rod factor, rod impulses and directions
    // (words 18 to 41 of its own column), so they leave the registers.
    // Every warp owns 44 words per lane; the three views never leave it, so
    // warps at different phases never touch each other's bytes.
    __shared__ float4 s_mem[(BLOCK / 32) * 11 * 32];
    float* const s_warp = reinterpret_cast<float*>(s_mem) + (threadIdx.x >> 5) * (44 * 32);
    float4* const s_node4 = reinterpret_cast<float4*>(s_warp);
    float2* const s_frc2 = reinterpret_cast<float2*>(s_warp + 16 * 32);
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned lg = tid & 1u;
    const unsigned gl = lane & 30u;
    const unsigned gmask = 3u << gl;
#define NODE(k, col) s_node4[(k) * 32 + (col)]
#define FRC(g, col) s_frc2[(g) * 32 + (col)]
#define SCR(q, col) s_warp[(q) * 32 + (col)]
#define WAT(idx) SCR((idx) >> 1, gl + ((idx) & 1))
#define WIJ(i, j) WAT((i) >= (j) ? (i) * ((i) + 1) / 2 + (j) : (j) * ((j) + 1) / 2 + (i))

    bool live = false, exhausted = false;
    unsigned cidx = 0u, mc = 0u, step = 0u, warm = 0u, prevc = 0u;
    // Registers hold the state; the statics (radius, friction, rest length,
    // pivot inverse mass) are read from the lane record in L2 when used.
    float invm[NPL];
    unsigned topo[NPL];
    float px[NPL], py[NPL], vx[NPL], vy[NPL];
#if MSTATE_REGS
    unsigned ms[MPL];
#define MS_GET(k) ms[k]
#define MS_SET(k, v) (ms[k] = (v))
#else
#define MS_GET(k) mstate[mb + (size_t)(k) * W]
#define MS_SET(k, v) (mstate[mb + (size_t)(k) * W] = (v))
#endif
    float msum = 0.0f, rounds_sum = 0.0f, contact_sum = 0.0f, drift_max = 0.0f;
#pragma unroll
    for (int k = 0; k < NPL; k++) {
        invm[k] = 0.0f; topo[k] = 0u; px[k] = 0.0f; py[k] = 0.0f; vx[k] = 0.0f; vy[k] = 0.0f;
    }
#if MSTATE_REGS
#pragma unroll
    for (int k = 0; k < MPL; k++) { ms[k] = 0u; }
#endif
#define REC(f, k) lanes[((size_t)cidx * RF + (f) + (k)) * W + lg]
    // Friction anchors per node in L2, NaN when the node has none.
#define ANC(k) anch[((size_t)cidx * NPL + (k)) * W + lg]
#define STASH(q) (reinterpret_cast<volatile float*>(s_warp)[(18 + (q)) * 32 + lane])

    for (;;) {
        if (!live && !exhausted) {
            unsigned got = 0u;
            if (lg == 0u) { got = atomicAdd(counter, 1u); }
            got = __shfl_sync(gmask, got, tid & 30u);
            if (got < p.count) {
                cidx = got;
                const uint4 h = heads[cidx];
                mc = lg == 0u ? (h.x >> 16u) & 255u : h.x >> 24u;
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    invm[k] = REC(0, k);
                    px[k] = REC(12, k);
                    py[k] = REC(16, k);
                    topo[k] = __float_as_uint(REC(24, k));
                    vx[k] = 0.0f; vy[k] = 0.0f;
                    ANC(k) = __uint_as_float(0x7fc00000u);
                }
#pragma unroll MUSCLE_UNROLL
                for (int k = 0; k < MPL; k++) {
                    const size_t mb = (size_t)cidx * MPL * W + lg;
                    MS_SET(k, pack_ed(1.0f, 0.0f));
                    roff[mb + (size_t)k * W] = 0.0f;
                }
                step = 0u; warm = 0u; prevc = 0u;
                msum = 0.0f; rounds_sum = 0.0f; contact_sum = 0.0f; drift_max = 0.0f;
                live = true;
            } else {
                exhausted = true;
                mc = 0u;
                cidx = 0u;
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    invm[k] = 0.0f; topo[k] = 0u; px[k] = 0.0f; py[k] = 0.0f; vx[k] = 0.0f; vy[k] = 0.0f;
                }
            }
        }
        if (!__any_sync(FULL, live)) { break; }
        const unsigned wm = __reduce_max_sync(FULL, mc);
        const float t0 = (float)step * DT;
        const size_t mb = (size_t)cidx * MPL * W + lg;
        // The waveform and drive target, once per step.
#pragma unroll MUSCLE_UNROLL
        for (int k = 0; k < MPL; k++) {
            if ((unsigned)k >= wm) { break; }
            if ((unsigned)k < mc) {
                const float4 s0 = mss[(size_t)cidx * (2 * MPL * W) + (2 * k) * W + lg];
                const float4 s1 = mss[(size_t)cidx * (2 * MPL * W) + (2 * k + 1) * W + lg];
                const float off = roff[mb + (size_t)k * W];
                const float w1 = wave(t0 + DT, s0.x, s0.y, off, s0.z, s0.w, s1.x);
                const float w0 = wave(t0, s0.x, s0.y, off, s0.z, s0.w, s1.x);
                const float target = s1.y * (w1 - w0) * RATE;
                const float drive = fmaxf(-target * s1.z * 0.25f, 0.0f);
                MS_SET(k, pack_ed(unpack_ed(MS_GET(k)).x, drive));
            }
        }
        unsigned cw = 0u;
#pragma unroll 1
        for (int sub = 0; sub < SUBSTEPS; sub++) {
            // Node table, and a clear force table.
#pragma unroll
            for (int k = 0; k < NPL; k++) { NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]); }
#pragma unroll
            for (int g = 0; g < W * NPL; g++) { FRC(g, lane) = make_float2(0.0f, 0.0f); }
            __syncwarp();
            float fx[NPL], fy[NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) { fx[k] = 0.0f; fy[k] = 0.0f; }
            // Air drag at each rod's midpoint, half to each node, and joint
            // damping as a couple on the rod and the opposite on its parent.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned t = topo[k];
                if (T_VALID(t) && !NO_DAMP) {
                    const unsigned a = T_PIVOT(t);
                    const float4 na = NODE(a & 3u, gl + (a >> 2u));
                    const float ddx = px[k] - na.x, ddy = py[k] - na.y;
                    const float il = rsqrtf(fmaxf(ddx * ddx + ddy * ddy, 1e-12f));
                    const float l = (ddx * ddx + ddy * ddy) * il;
                    const float wx = 0.5f * (vx[k] + na.z), wy = 0.5f * (vy[k] + na.w);
                    const float speed = sqrtf(wx * wx + wy * wy);
                    const float m = 1.0f / fmaxf(fmaxf(invm[k], REC(28, k)), 1e-6f);
                    const float strength = fminf(AIR_DRAG * l * (REC(4, k) * 2.0f) * speed, 0.5f * m * INV_HS);
                    const float hx = -0.5f * wx * strength, hy = -0.5f * wy * strength;
                    fx[k] += hx; fy[k] += hy;
                    float2 fa = FRC(a, lane);
                    fa.x += hx; fa.y += hy;
                    const float rvx = vx[k] - na.z, rvy = vy[k] - na.w;
                    const float w_rod = (ddx * rvy - ddy * rvx) * il * il;
                    const unsigned g = T_GP(t);
                    float w_par = 0.0f, pdx = 0.0f, pdy = 0.0f, pil = 0.0f;
                    if (g != 31u) {
                        const float4 ng = NODE(g & 3u, gl + (g >> 2u));
                        pdx = na.x - ng.x; pdy = na.y - ng.y;
                        pil = rsqrtf(fmaxf(pdx * pdx + pdy * pdy, 1e-12f));
                        w_par = (pdx * (na.w - ng.w) - pdy * (na.z - ng.z)) * pil * pil;
                    }
                    const float tau = -m * l * l * (w_rod - w_par) * INV_JOINT_DAMPING;
                    const float cf = tau * il * il;
                    fx[k] += -ddy * cf; fy[k] += ddx * cf;
                    fa.x -= -ddy * cf; fa.y -= ddx * cf;
                    if (g != 31u) {
                        // The parent rod takes -tau.
                        const float pc = tau * pil * pil;
                        fa.x -= -pdy * pc; fa.y -= pdx * pc;
                        float2 fg = FRC(g, lane);
                        fg.x += -pdy * pc; fg.y += pdx * pc;
                        FRC(a, lane) = fa;
                        FRC(g, lane) = fg;
                    } else {
                        FRC(a, lane) = fa;
                    }
                }
            }
            // Muscles.
#pragma unroll MUSCLE_UNROLL
            for (int k = 0; k < MPL; k++) {
                if ((unsigned)k >= wm) { break; }
                if ((unsigned)k < mc && !NO_MUSCLES) {
                    const float4 A = msa[mb + (size_t)k * W];
                    const float2 B = msb[mb + (size_t)k * W];
                    const unsigned pk = __float_as_uint(A.x);
                    const unsigned na_ = pk & 31u, nb_ = (pk >> 5u) & 31u, nc_ = (pk >> 10u) & 31u, nd_ = (pk >> 15u) & 31u;
                    const float4 e0 = NODE(na_ & 3u, gl + (na_ >> 2u));
                    const float4 e1 = NODE(nb_ & 3u, gl + (nb_ >> 2u));
                    const float4 e2 = NODE(nc_ & 3u, gl + (nc_ >> 2u));
                    const float4 e3 = NODE(nd_ & 3u, gl + (nd_ >> 2u));
                    const float2 an = unpack_un(__float_as_uint(A.y));
                    const float2 hc = unpack_bf(__float_as_uint(B.y));
                    const float cap = A.z, tendon_k = A.w, slack = B.x, hill = hc.x, inv_capacity = hc.y;
                    const float pax = e0.x + (e1.x - e0.x) * an.x, pay = e0.y + (e1.y - e0.y) * an.x;
                    const float vax = e0.z + (e1.z - e0.z) * an.x, vay = e0.w + (e1.w - e0.w) * an.x;
                    const float pbx = e2.x + (e3.x - e2.x) * an.y, pby = e2.y + (e3.y - e2.y) * an.y;
                    const float vbx = e2.z + (e3.z - e2.z) * an.y, vby = e2.w + (e3.w - e2.w) * an.y;
                    const float ddx = pbx - pax, ddy = pby - pay;
                    const float length = fmaxf(sqrtf(ddx * ddx + ddy * ddy), 1e-6f);
                    const float inverse = 1.0f / length;
                    const float dirx = ddx * inverse, diry = ddy * inverse;
                    const float relative = (vbx - vax) * dirx + (vby - vay) * diry;
                    const float2 st = unpack_ed(MS_GET(k));
                    float energy = st.x;
                    float drive = st.y * energy;
                    if (hill > 0.0f) { drive *= clampf(1.0f + relative * hill, 0.0f, 1.0f); }
                    const float magnitude = clampf(drive + relative * 0.15f, -cap, cap);
                    const float work = fminf(drive, cap) * fmaxf(-relative, 0.0f) * HS;
                    energy = clampf(energy - work * inv_capacity + MUSCLE_RECOVERY * p.recovery * HS * (1.0f - energy), 0.0f, 1.0f);
                    MS_SET(k, pack_ed(energy, st.y));
                    const float stretch = fmaxf(length - slack, 0.0f);
                    const float pull = magnitude + tendon_k * stretch;
                    const float gx = dirx * pull, gy = diry * pull;
                    msum += magnitude * relative;
                    float2 f;
                    f = FRC(na_, lane); f.x += (1.0f - an.x) * gx; f.y += (1.0f - an.x) * gy; FRC(na_, lane) = f;
                    f = FRC(nb_, lane); f.x += an.x * gx; f.y += an.x * gy; FRC(nb_, lane) = f;
                    f = FRC(nc_, lane); f.x -= (1.0f - an.y) * gx; f.y -= (1.0f - an.y) * gy; FRC(nc_, lane) = f;
                    f = FRC(nd_, lane); f.x -= an.y * gx; f.y -= an.y * gy; FRC(nd_, lane) = f;
                }
            }
            __syncwarp();
            // Free velocities.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned g = lg * NPL + k;
                const float2 f0 = FRC(g, lane), f1 = FRC(g, lane ^ 1u);
                const float gx = fx[k] + f0.x + f1.x, gy = fy[k] + f0.y + f1.y;
                const bool on = invm[k] > 0.0f;
                vx[k] += HS * invm[k] * gx;
                vy[k] += on ? HS * (invm[k] * gy - p.gravity) : 0.0f;
            }
            __syncwarp();
            // Node table with the free velocities; node statics for the rows.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]);
                FRC(2 * k, lane) = make_float2(invm[k], REC(8, k) * p.friction);
                FRC(2 * k + 1, lane) = make_float2(REC(4, k), ANC(k));
            }
            __syncwarp();
            // Rod directions and the rods' own rows.
            float dx[NPL], dy[NPL], rdx[NPL], rdy[NPL], rhs0[NPL];
#ifdef DIAG
            float ctg[NPL];
#endif
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned t = topo[k];
                dx[k] = 0.0f; dy[k] = 0.0f; rhs0[k] = 0.0f;
#ifdef DIAG
                ctg[k] = 0.0f;
#endif
                if (T_VALID(t)) {
                    const unsigned a = T_PIVOT(t);
                    const float4 na = NODE(a & 3u, gl + (a >> 2u));
                    const float ddx = px[k] - na.x, ddy = py[k] - na.y;
                    const float il = rsqrtf(fmaxf(ddx * ddx + ddy * ddy, 1e-12f));
                    dx[k] = ddx * il; dy[k] = ddy * il;
                    const float rvx = vx[k] - na.z, rvy = vy[k] - na.w;
                    const float rn = rvx * dx[k] + rvy * dy[k];
                    const float perp2 = fmaxf(rvx * rvx + rvy * rvy - rn * rn, 0.0f);
                    rhs0[k] = -rn - perp2 * HS * il;
#ifdef DIAG
                    ctg[k] = -perp2 * HS * il;
#endif
                }
            }
#pragma unroll
            for (int q = 0; q < NPL; q++) { rdx[q] = xch(dx[q]); rdy[q] = xch(dy[q]); }
            float invD[NPL], Lf[NPL][NB];
            {
                float ipv[NPL];
#pragma unroll
                for (int k = 0; k < NPL; k++) { ipv[k] = REC(28, k); }
                factor(topo, dx, dy, rdx, rdy, ipv, invm, lg, invD, Lf);
            }
            // Contacts: the deepest 4 nodes that would reach the ground.
            unsigned word = 0u;
            {
                float dep[NPL], rdep[NPL];
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const float d = py[k] - REC(4, k) + HS * vy[k];
                    dep[k] = (invm[k] > 0.0f && d <= 0.0f && !NO_CONTACT) ? d : 1e30f;
                }
#pragma unroll
                for (int q = 0; q < NPL; q++) { rdep[q] = xch(dep[q]); }
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    unsigned r = 0u;
#pragma unroll
                    for (int q = 0; q < NPL; q++) {
                        if (q != k) { r += (dep[q] < dep[k] || (dep[q] == dep[k] && q < k)) ? 1u : 0u; }
                        // The partner's nodes come first when this is lane 1.
                        r += (rdep[q] < dep[k] || (rdep[q] == dep[k] && lg == 1u)) ? 1u : 0u;
                    }
                    if (dep[k] < 1e29f && r < (unsigned)NC) { word |= ((lg * NPL + k) | 32u) << (6u * r); }
                }
                word |= xchu(word);
            }
            cw = word;
            // Each slot's values, for both lanes: inverse mass (0 when the
            // slot is empty), friction, targets and free velocities.
            float cim[NC], cmu[NC], b0[NE];
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const unsigned e = (word >> (6u * c)) & 63u;
                const unsigned g = e & 31u;
                const float4 nd = NODE(g & 3u, gl + (g >> 2u));
                const float2 s0 = FRC(2u * (g & 3u), gl + (g >> 2u));
                const float2 s1 = FRC(2u * (g & 3u) + 1u, gl + (g >> 2u));
                cim[c] = (e & 32u) ? s0.x : 0.0f;
                cmu[c] = NO_FRICTION ? 0.0f : s0.y;
                const float gap = nd.y - s1.x;
                b0[2 * c] = (gap >= 0.0f ? -gap * INV_HS : -gap * PUSH_OUT * INV_HS) - nd.w;
                b0[2 * c + 1] = (isnan(s1.y) ? 0.0f : (s1.y - nd.x) * INV_HS) - nd.z;
            }
#define CNODE(c) ((word >> (6u * (c))) & 31u)
            // Incidence of each local rod at each contact node, two bits per
            // pair: the rod's child (+) or its pivot (-). R's entry is this
            // times the node's inverse mass times d.y (normal) or d.x.
            unsigned sgw = 0u;
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned t = topo[k];
                if (T_VALID(t)) {
#pragma unroll
                    for (int c = 0; c < NC; c++) {
                        const unsigned cn = CNODE(c);
                        if (cn == lg * NPL + (unsigned)k) { sgw |= 1u << (2 * (4 * k + c)); }
                        if (cn == T_PIVOT(t)) { sgw |= 2u << (2 * (4 * k + c)); }
                    }
                }
            }
#define RCO(k, c) (((sgw >> (2 * (4 * (k) + (c)))) & 1u) ? cim[c] : (((sgw >> (2 * (4 * (k) + (c)))) & 2u) ? -cim[c] : 0.0f))
            // Batch A: row 0 the rods' own, rows 1 + 2c and 2 + 2c contact
            // c's normal and friction (c = 0, 1); lane 0 one slot behind.
            float mu0[NPL], me[4][NPL];
            {
                float z[6][NPL];
#pragma unroll
                for (int s = 0; s < 6; s++) {
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        float v1 = 0.0f, v0 = 0.0f;
                        if (s == 0) { v1 = rhs0[k]; }
                        else if (s < 5) { v1 = RCO(k, (s - 1) >> 1) * (((s - 1) & 1) ? dx[k] : dy[k]); }
                        if (s == 1) { v0 = rhs0[k]; }
                        else if (s >= 2) { v0 = RCO(k, (s - 2) >> 1) * (((s - 2) & 1) ? dx[k] : dy[k]); }
                        z[s][k] = lg == 1u ? v1 : v0;
                    }
                }
                tree_solve<6>(z, topo, invD, Lf, lg);
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    mu0[k] = lg == 1u ? z[0][k] : z[1][k];
#ifdef DUMP
                    if (cidx == 0u && step == 0u && sub == 0 && live) {
                        volatile float* d = reinterpret_cast<volatile float*>(trace) + lg * 32;
                        d[k] = dx[k]; d[4 + k] = dy[k]; d[8 + k] = rhs0[k]; d[12 + k] = mu0[k];
                        d[16 + k] = invm[k]; d[20 + k] = __uint_as_float(topo[k]); d[24 + k] = invD[k];
                        d[28 + k] = Lf[k][0];
                    }
#endif
#pragma unroll
                    for (int e = 0; e < 4; e++) { me[e][k] = lg == 1u ? z[1 + e][k] : z[2 + e][k]; }
                }
            }
            // Free contact velocities w = J v* + R^T mu0 go into b0.
#pragma unroll
            for (int e = 0; e < NE; e++) {
                float part = 0.0f;
#pragma unroll
                for (int k = 0; k < NPL; k++) { part += RCO(k, e >> 1) * ((e & 1) ? dx[k] : dy[k]) * mu0[k]; }
                b0[e] -= part + xch(part);
            }
            __syncwarp();
            // W = W0 - R^T A^-1 R, rows 0 to 3 now, 4 to 7 after batch B.
#pragma unroll
            for (int f = 0; f < NE; f++) {
                float rf[NPL];
#pragma unroll
                for (int k = 0; k < NPL; k++) { rf[k] = RCO(k, f >> 1) * ((f & 1) ? dx[k] : dy[k]); }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    if (e <= f) {
                        float part = 0.0f;
#pragma unroll
                        for (int k = 0; k < NPL; k++) { part += rf[k] * me[e][k]; }
                        float v = -(part + xch(part));
                        if (f == e) { v += cim[e >> 1] * (1.0f + COMPLIANCE); }
                        const int idx = f * (f + 1) / 2 + e;
                        if (lg == (unsigned)(idx & 1)) { SCR(idx >> 1, lane) = v; }
                    }
                }
            }
            {
                float z[5][NPL];
#pragma unroll
                for (int s = 0; s < 5; s++) {
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        float v1 = 0.0f, v0 = 0.0f;
                        if (s < 4) { v1 = RCO(k, (4 + s) >> 1) * (((4 + s) & 1) ? dx[k] : dy[k]); }
                        if (s >= 1) { v0 = RCO(k, (3 + s) >> 1) * (((3 + s) & 1) ? dx[k] : dy[k]); }
                        z[s][k] = lg == 1u ? v1 : v0;
                    }
                }
                tree_solve<5>(z, topo, invD, Lf, lg);
#pragma unroll
                for (int k = 0; k < NPL; k++) {
#pragma unroll
                    for (int e = 0; e < 4; e++) { me[e][k] = lg == 1u ? z[e][k] : z[1 + e][k]; }
                }
            }
#pragma unroll
            for (int f = 4; f < NE; f++) {
                float rf[NPL];
#pragma unroll
                for (int k = 0; k < NPL; k++) { rf[k] = RCO(k, f >> 1) * ((f & 1) ? dx[k] : dy[k]); }
#pragma unroll
                for (int e = 4; e < NE; e++) {
                    if (e <= f) {
                        float part = 0.0f;
#pragma unroll
                        for (int k = 0; k < NPL; k++) { part += rf[k] * me[e - 4][k]; }
                        float v = -(part + xch(part));
                        if (f == e) { v += cim[e >> 1] * (1.0f + COMPLIANCE); }
                        const int idx = f * (f + 1) / 2 + e;
                        if (lg == (unsigned)(idx & 1)) { SCR(idx >> 1, lane) = v; }
                    }
                }
            }
            __syncwarp();
#ifdef DUMPW
            if (cidx == 0u && step == DUMPW && sub == 0 && live) {
                volatile float* d = reinterpret_cast<volatile float*>(trace);
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    d[lg * 32 + k] = px[k]; d[lg * 32 + 4 + k] = py[k]; d[lg * 32 + 8 + k] = vx[k]; d[lg * 32 + 12 + k] = vy[k];
                    d[lg * 32 + 16 + k] = invm[k]; d[lg * 32 + 20 + k] = __uint_as_float(topo[k]);
                    d[lg * 32 + 24 + k] = dx[k]; d[lg * 32 + 28 + k] = dy[k];
                }
                if (lg == 0u) {
                    d[64] = __uint_as_float(word);
                    for (int i = 0; i < 36; i++) { d[65 + i] = WAT(i); }
                    for (int e = 0; e < NE; e++) { d[101 + e] = b0[e]; }
                    for (int c = 0; c < NC; c++) { d[109 + c] = cim[c]; }
                }
            }
#endif
            // The rod factor, rod impulses and directions wait in shared
            // memory while the active set runs.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
#pragma unroll
                for (int a = 0; a < NB; a++) { STASH(k * NB + a) = Lf[k][a]; }
                STASH(NPL * NB + k) = mu0[k];
                STASH(NPL * NB + NPL + k) = dx[k];
                STASH(NPL * NB + 2 * NPL + k) = dy[k];
            }
            // The direct active-set solve, the same in both lanes. Bit 2c:
            // normal c on; bit 2c + 1: friction c free (off: at its bound,
            // sign in `neg`). Warm started from the last substep.
            unsigned on = (warm & 0x10000u) ? (warm & 0xffu) : 0xffu, neg = (warm >> 8u) & 0xffu;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                if (cim[c] == 0.0f) { on &= ~(3u << (2 * c)); }
                else if (!((on >> (2 * c)) & 1u)) { on |= 3u << (2 * c); }
            }
            float lam[NE];
#pragma unroll
            for (int e = 0; e < NE; e++) { lam[e] = 0.0f; }
            unsigned rounds = 0u;
            bool again = true;
#pragma unroll 1
            for (int round = 0; round < MAX_ROUNDS; round++) {
                if (!__any_sync(FULL, again)) { break; }
                rounds += 1u;
                float fixv[NE];
#pragma unroll
                for (int e = 0; e < NE; e++) {
                    const int c = e >> 1;
                    const bool fixed = (e & 1) && ((on >> (2 * c)) & 1u) && !((on >> e) & 1u);
                    fixv[e] = fixed ? (((neg >> e) & 1u) ? -1.0f : 1.0f) * cmu[c] * lam[e & ~1] : 0.0f;
                }
                float b[NE], iD[NE];
#pragma unroll
                for (int i = 0; i < NE; i++) {
                    float r = b0[i];
#pragma unroll
                    for (int j = 1; j < NE; j += 2) { r -= WIJ(i, j) * fixv[j]; }
                    b[i] = ((on >> i) & 1u) ? r : 0.0f;
                }
                float L[NE][NE];
#pragma unroll
                for (int i = 0; i < NE; i++) {
                    const bool ai = (on >> i) & 1u;
                    float d = ai ? WIJ(i, i) : 1.0f;
#pragma unroll
                    for (int j = 0; j < i; j++) {
                        const bool aj = (on >> j) & 1u;
                        float v = (ai && aj) ? WIJ(i, j) : 0.0f;
#pragma unroll
                        for (int k = 0; k < j; k++) { v -= L[i][k] * L[j][k] * (1.0f / iD[k]); }
                        L[i][j] = v * iD[j];
                        d -= v * L[i][j];
                    }
                    iD[i] = 1.0f / d;
                }
#pragma unroll
                for (int i = 0; i < NE; i++) {
#pragma unroll
                    for (int k = 0; k < i; k++) { b[i] -= L[i][k] * b[k]; }
                }
#pragma unroll
                for (int i = 0; i < NE; i++) { b[i] *= iD[i]; }
#pragma unroll
                for (int i = NE - 1; i >= 0; i--) {
#pragma unroll
                    for (int k = i + 1; k < NE; k++) { b[i] -= L[k][i] * b[k]; }
                }
                // Leave the set: a pulling normal; friction past its cone.
                unsigned next = on;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const float ln = b[2 * c];
                    const float lt = ((on >> (2 * c + 1)) & 1u) ? b[2 * c + 1] : fixv[2 * c + 1];
                    lam[2 * c] = ln;
                    lam[2 * c + 1] = lt;
                    if (((on >> (2 * c)) & 1u) && ln < 0.0f) {
                        next &= ~(3u << (2 * c)); lam[2 * c] = 0.0f; lam[2 * c + 1] = 0.0f;
                    } else if (((on >> (2 * c + 1)) & 1u) && fabsf(lt) > cmu[c] * fmaxf(ln, 0.0f)) {
                        next &= ~(1u << (2 * c + 1));
                        if (lt < 0.0f) { neg |= 1u << (2 * c + 1); } else { neg &= ~(1u << (2 * c + 1)); }
                    }
                }
                again = next != on;
                on = next;
            }
            rounds_sum += (float)rounds;
            warm = on | (neg << 8u) | 0x10000u;
            float Lg[NPL][NB], mu0g[NPL], dxg[NPL], dyg[NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) {
#pragma unroll
                for (int a = 0; a < NB; a++) { Lg[k][a] = USE_STASH ? STASH(k * NB + a) : Lf[k][a]; }
                mu0g[k] = USE_STASH ? STASH(NPL * NB + k) : mu0[k];
                dxg[k] = USE_STASH ? STASH(NPL * NB + NPL + k) : dx[k];
                dyg[k] = USE_STASH ? STASH(NPL * NB + 2 * NPL + k) : dy[k];
            }
            // The rods' answer to the contact impulses, one more tree solve.
            float z1[2][NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                float r = 0.0f;
#pragma unroll
                for (int c = 0; c < NC; c++) { r += RCO(k, c) * (lam[2 * c] * dyg[k] + lam[2 * c + 1] * dxg[k]); }
                z1[0][k] = lg == 1u ? r : 0.0f;
                z1[1][k] = lg == 1u ? 0.0f : r;
            }
            tree_solve<2>(z1, topo, invD, Lg, lg);
            __syncwarp();
            // Rod impulses mu = mu0 - A^-1 R lambda to the rods' two nodes;
            // contact impulses to theirs.
#pragma unroll
            for (int g = 0; g < W * NPL; g++) { FRC(g, lane) = make_float2(0.0f, 0.0f); }
            __syncwarp();
            float ix[NPL], iy[NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const float mu = mu0g[k] - (lg == 1u ? z1[0][k] : z1[1][k]);
                ix[k] = dxg[k] * mu; iy[k] = dyg[k] * mu;
                if (T_VALID(topo[k])) {
                    const unsigned a = T_PIVOT(topo[k]);
                    float2 f = FRC(a, lane);
                    f.x -= ix[k]; f.y -= iy[k];
                    FRC(a, lane) = f;
                }
            }
            unsigned own = 0u;
            float ext_x = 0.0f, ext_y = 0.0f;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const unsigned g = CNODE(c);
                if (cim[c] != 0.0f && (g >> 2u) == lg) {
                    const unsigned om = 1u << (g & 3u);
                    own |= om;
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        if (om & (1u << k)) { ix[k] += lam[2 * c + 1]; iy[k] += lam[2 * c]; }
                    }
                    ext_x += lam[2 * c + 1]; ext_y += lam[2 * c];
                }
            }
            __syncwarp();
            // New velocities. Ledgers: the impulses of the substep must move
            // the body's momentum and its angular momentum about the centre
            // of mass by exactly the contact impulses and their torques; the
            // rounding rest goes as one uniform velocity and one rotation.
            {
                float mm = 0.0f, mx = 0.0f, my = 0.0f, gx = 0.0f, gy = 0.0f;
                float jx[NPL], jy[NPL];
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const unsigned g = lg * NPL + k;
                    const float2 f0 = FRC(g, lane), f1 = FRC(g, lane ^ 1u);
                    jx[k] = ix[k] + f0.x + f1.x;
                    jy[k] = iy[k] + f0.y + f1.y;
                    vx[k] += invm[k] * jx[k];
                    vy[k] += invm[k] * jy[k];
                    const float m = invm[k] > 0.0f ? 1.0f / invm[k] : 0.0f;
                    mm += m; mx += m * px[k]; my += m * py[k];
                    gx += invm[k] > 0.0f ? jx[k] : 0.0f;
                    gy += invm[k] > 0.0f ? jy[k] : 0.0f;
                }
#ifdef DIAG
                {
                    // Residuals after the impulses: active normal rows against
                    // their targets, rods against their centripetal targets.
                    float rn = 0.0f, rr = 0.0f;
#pragma unroll
                    for (int c = 0; c < NC; c++) {
                        const unsigned g = CNODE(c);
                        if (cim[c] != 0.0f && (g >> 2u) == lg && ((on >> (2 * c)) & 1u)) {
#pragma unroll
                            for (int k = 0; k < NPL; k++) {
                                if ((1u << (g & 3u)) & (1u << k)) {
                                    const float gap = py[k] - REC(4, k);
                                    const float tgt = gap >= 0.0f ? -gap * INV_HS : -gap * PUSH_OUT * INV_HS;
                                    rn = fmaxf(rn, fabsf(vy[k] - tgt));
                                }
                            }
                        }
                    }
                    __syncwarp();
#pragma unroll
                    for (int k = 0; k < NPL; k++) { NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]); }
                    __syncwarp();
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        if (T_VALID(topo[k])) {
                            const unsigned a = T_PIVOT(topo[k]);
                            const float4 na = NODE(a & 3u, gl + (a >> 2u));
                            rr = fmaxf(rr, fabsf((vx[k] - na.z) * dxg[k] + (vy[k] - na.w) * dyg[k] - ctg[k]));
                        }
                    }
                    __syncwarp();
                    drift_max = fmaxf(drift_max, rn);
                    msum = fmaxf(msum, rr);
                }
#endif
                const float itm = 1.0f / fmaxf(mm + xch(mm), 1e-6f);
                const float ccx = (mx + xch(mx)) * itm, ccy = (my + xch(my)) * itm;
                const float sx = NO_LEDGER ? 0.0f : (ext_x + xch(ext_x) - gx - xch(gx)) * itm;
                const float sy = NO_LEDGER ? 0.0f : (ext_y + xch(ext_y) - gy - xch(gy)) * itm;
                float lz = 0.0f, lc = 0.0f, iz = 0.0f;
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const bool on_ = invm[k] > 0.0f;
                    const float m = on_ ? 1.0f / invm[k] : 0.0f;
                    const float rx = px[k] - ccx, ry = py[k] - ccy;
                    lz += on_ ? rx * jy[k] - ry * jx[k] : 0.0f;
                    iz += m * (rx * rx + ry * ry);
                    vx[k] += on_ ? sx : 0.0f;
                    vy[k] += on_ ? sy : 0.0f;
                }
                // The contact impulses' torque about the centre of mass.
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const unsigned g = CNODE(c);
                    if (cim[c] != 0.0f && (g >> 2u) == lg) {
                        const unsigned om = 1u << (g & 3u);
#pragma unroll
                        for (int k = 0; k < NPL; k++) {
                            if (om & (1u << k)) { lc += (px[k] - ccx) * lam[2 * c] - (py[k] - ccy) * lam[2 * c + 1]; }
                        }
                    }
                }
                const float itot = iz + xch(iz);
                const float lsum = lc + xch(lc) - lz - xch(lz);
                const float wc = (itot > 0.0f && !NO_LEDGER) ? lsum / itot : 0.0f;
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const bool on_ = invm[k] > 0.0f;
                    vx[k] -= on_ ? wc * (py[k] - ccy) : 0.0f;
                    vy[k] += on_ ? wc * (px[k] - ccx) : 0.0f;
                }
            }
            // Semi-implicit Euler; anchors set on touching, moved by sliding.
            // The stub's synthetic bodies and explicit forces are not a
            // tuned physics: a speed clamp keeps every trial finite, so all
            // creatures run their full length.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                vx[k] = clampf(vx[k], -SPEED_CLAMP, SPEED_CLAMP);
                vy[k] = clampf(vy[k], -SPEED_CLAMP, SPEED_CLAMP);
                px[k] += HS * vx[k];
                py[k] += HS * vy[k];
            }
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const unsigned g = CNODE(c);
                if (cim[c] != 0.0f && (g >> 2u) == lg) {
                    const bool slid = !((on >> (2 * c + 1)) & 1u);
                    const unsigned om = 1u << (g & 3u);
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        if (om & (1u << k)) {
                            if (slid || isnan(ANC(k))) { ANC(k) = px[k]; }
                        }
                    }
                }
            }
            contact_sum += (float)__popc(word & 0x820820u);
#undef RCO
#undef CNODE
        }
        // Drift projection: rod lengths back to rest with the contact nodes
        // held (infinite mass): one factor and one tree solve.
        unsigned held = 0u;
        {
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const unsigned e = (cw >> (6u * c)) & 63u;
                if (e & 32u) { held |= 1u << (e & 31u); }
            }
            // A node out of contact for the step loses its anchor.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                if (!((held >> (lg * NPL + k)) & 1u) && ((prevc >> (lg * NPL + k)) & 1u)) { ANC(k) = __uint_as_float(0x7fc00000u); }
            }
            __syncwarp();
#pragma unroll
            for (int k = 0; k < NPL; k++) { NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]); }
            __syncwarp();
            float dx[NPL], dy[NPL], rdx[NPL], rdy[NPL], ipe[NPL], icm[NPL];
            float z1[2][NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned t = topo[k];
                dx[k] = 0.0f; dy[k] = 0.0f;
                float c = 0.0f;
                if (T_VALID(t)) {
                    const unsigned a = T_PIVOT(t);
                    const float4 na = NODE(a & 3u, gl + (a >> 2u));
                    const float ddx = px[k] - na.x, ddy = py[k] - na.y;
                    const float l = sqrtf(ddx * ddx + ddy * ddy);
                    const float il = 1.0f / fmaxf(l, 1e-6f);
                    dx[k] = ddx * il; dy[k] = ddy * il;
                    c = l - REC(20, k);
                }
                drift_max = fmaxf(drift_max, fabsf(c));
                ipe[k] = ((held >> T_PIVOT(t)) & 1u) ? 0.0f : REC(28, k);
                icm[k] = ((held >> (lg * NPL + k)) & 1u) ? 0.0f : invm[k];
                z1[0][k] = lg == 1u ? -c : 0.0f;
                z1[1][k] = lg == 1u ? 0.0f : -c;
            }
#pragma unroll
            for (int q = 0; q < NPL; q++) { rdx[q] = xch(dx[q]); rdy[q] = xch(dy[q]); }
            float invD[NPL], Lf[NPL][NB];
            factor(topo, dx, dy, rdx, rdy, ipe, icm, lg, invD, Lf);
            tree_solve<2>(z1, topo, invD, Lf, lg);
#pragma unroll
            for (int g = 0; g < W * NPL; g++) { FRC(g, lane) = make_float2(0.0f, 0.0f); }
            __syncwarp();
            float ix[NPL], iy[NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const float mu = lg == 1u ? z1[0][k] : z1[1][k];
                ix[k] = dx[k] * mu; iy[k] = dy[k] * mu;
                if (T_VALID(topo[k])) {
                    const unsigned a = T_PIVOT(topo[k]);
                    float2 f = FRC(a, lane);
                    f.x -= ix[k]; f.y -= iy[k];
                    FRC(a, lane) = f;
                }
            }
            __syncwarp();
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned g = lg * NPL + k;
                const float2 f0 = FRC(g, lane), f1 = FRC(g, lane ^ 1u);
                px[k] += icm[k] * (ix[k] + f0.x + f1.x);
                py[k] += icm[k] * (iy[k] + f0.y + f1.y);
            }
            __syncwarp();
        }
        // Metrics, the fall rule and the end of the trial, once per step.
        {
            float mm = 0.0f, mx = 0.0f, low = 1e20f;
            bool bad = false;
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const float m = invm[k] > 0.0f ? 1.0f / invm[k] : 0.0f;
                mm += m; mx += m * px[k];
                if (invm[k] > 0.0f) { low = fminf(low, py[k] - REC(4, k)); }
                bad |= !(fabsf(px[k]) <= 1e6f && fabsf(py[k]) <= 1e6f);
            }
            const float com_x = (mx + xch(mx)) / fmaxf(mm + xch(mm), 1e-6f);
            low = fminf(low, xch(low));
            bad = ((__ballot_sync(FULL, bad) >> gl) & 3u) != 0u;
            const float ms_all = msum + xch(msum);
            const float head_y = __shfl_sync(FULL, py[0], tid & 30u);
            const float neck_y = __shfl_sync(FULL, py[1], tid & 30u);
            // Touchdowns restart the rhythm of the muscles that sense them.
            const unsigned down = held & ~prevc;
            if (down != 0u && live) {
#pragma unroll MUSCLE_UNROLL
                for (int k = 0; k < MPL; k++) {
                    if ((unsigned)k < mc) {
                        const unsigned pk = __float_as_uint(msa[mb + (size_t)k * W].x);
                        if (((pk >> 25u) & 1u) && ((down >> ((pk >> 20u) & 31u)) & 1u)) {
                            const float4 s0 = mss[(size_t)cidx * (2 * MPL * W) + (2 * k) * W + lg];
                            const float x = -((t0 + DT) * s0.x + s0.y);
                            roff[mb + (size_t)k * W] = x - floorf(x);
                        }
                    }
                }
            }
            prevc = held;
            if (live) {
                step += 1u;
                if (step >= p.steps || bad) {
                    if (lg == 0u) {
                        results[2u * cidx] = make_float4(bad ? -1e20f : com_x, head_y < neck_y ? 1.0f : 0.0f, low, (float)step);
                        results[2u * cidx + 1u] = make_float4(rounds_sum, contact_sum, drift_max, ms_all);
                    }
                    live = false;
                }
            }
        }
    }
}
