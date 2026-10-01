// Stub of the per-lane maximal-coordinate kernel with the physics-lean
// variants (docs/plan-2m.md, section 2 and phase 2, round 7). It measures
// registers, spills, issue rate and speed of the design before anyone
// writes the physics; its numbers are not a physics.
//
// W = 2: one creature per two lanes. Lane l owns nodes 4l .. 4l + 3 and the
// rods that end at them (rod j ends at node j + 1; node 0, the head, has
// none), so lane 0 holds the low rods and lane 1 the high ones. W = 1: one
// creature per lane, nodes 0 to 7 in registers, no exchange. Rods are
// numbered breadth first: a rod's parent and its lower siblings have lower
// numbers, and siblings are consecutive.
//
// The default build is the phase-0 stub (today's rules). A substep: node
// table to shared memory; gravity, air drag and joint damping; the muscles
// (split records read from L2, forces gathered through a private per-lane
// shared table); the free velocities; the rod matrix A = J M^-1 J^T factored
// as LDL^T in reverse rod order (no fill: the rods at a node form a clique);
// the deepest 4 nodes as contacts, 2 rows each, with the 8 x 8 Delassus
// matrix, an active-set solve, the momentum and angular-momentum ledgers and
// semi-implicit Euler. Once per step: the waveform and drive target of every
// muscle, a drift projection with contact nodes at infinite mass, metrics.
//
// The physics-lean variants (each a define, they compose):
//   MUSCLE_MODEL=1       a muscle joins two nodes; force = cap x strength x
//                        a(t) x E x Hill, pull only, with the damper; a(t) a
//                        trapezoid of period, phase and duty with a ramp of
//                        two substeps on each edge; E one stamina store per
//                        creature, charged with the summed work; no tendon,
//                        no per-muscle state; limb touchdown clocks. With
//                        MUSCLE_ANCHORS=1 the ends are points along two
//                        bones again (the price of the anchor fractions).
//   CONTACT_MODEL=1      per-node impulses against each node's own mass for
//   NPASS=2              every node (normal, then friction clamped to the
//                        cone, the clean rule and the anchor), then the exact
//                        rod solve as the coupling, NPASS times; momentum
//                        and angular ledgers on the contact impulses; the
//                        8 x 8 Delassus matrix, the active set and the stash
//                        are gone. Penetration recovery is a position nudge.
//   SUBSTEPS=4           with LAGGED_FACTOR=1 the rod directions and the LDL
//   LAGGED_FACTOR=1      factor are built once per step, and the explicit
//                        drag and damping forces once per step; the solves,
//                        contacts and muscles run every substep.
//   LIMITS_AS_IMPULSES=1 joint limits and the spin cap as one angular
//   LIGAMENT=<c>         impulse per joint and step, a couple on the rod and
//                        the opposite on its parent, with a compliance c
//                        folded into the joint's effective inertia.
//   BAKED=1 (W = 1)      the bone tree is compiled in as constants BT0..BT7
//                        (the topology words), so every parent and pivot
//                        index is a register name; muscle ends stay runtime.
//
// Other defines (all have defaults): W, MPL, NB (rod neighbours kept per rod:
// the parent and NB - 1 lower siblings), BLOCK, MIN_BLOCKS, SUBSTEPS,
// MAX_ROUNDS, MSTATE_REGS, MUSCLE_UNROLL. Developer switches: NO_MUSCLES,
// NO_CONTACT, NO_DAMP, NO_FRICTION, NO_LEDGER, LEDGER=0 (no enforced ledgers),
// USE_STASH, DIAG (residuals in
// the result).
//
// The synthetic bodies and the explicit forces are not tuned; a speed clamp
// (SPEED_CLAMP) keeps trials finite so every creature runs its full length.

#ifndef W
#define W 2
#endif
#ifndef MPL
#if W == 2
#define MPL 16
#else
#define MPL 24
#endif
#endif
#ifndef NB
#define NB 3
#endif
#ifndef BLOCK
#define BLOCK 128
#endif
#ifndef MIN_BLOCKS
#if W == 2
#define MIN_BLOCKS 4
#else
#define MIN_BLOCKS 3
#endif
#endif
#ifndef SUBSTEPS
#define SUBSTEPS 2
#endif
#ifndef MAX_ROUNDS
#define MAX_ROUNDS 3
#endif
#ifndef MUSCLE_MODEL
#define MUSCLE_MODEL 0
#endif
#ifndef MUSCLE_ANCHORS
#define MUSCLE_ANCHORS 0
#endif
#ifndef CONTACT_MODEL
#define CONTACT_MODEL 0
#endif
#ifndef NPASS
#define NPASS 2
#endif
#ifndef LAGGED_FACTOR
#define LAGGED_FACTOR 0
#endif
#ifndef LIMITS_AS_IMPULSES
#define LIMITS_AS_IMPULSES 0
#endif
#ifndef LIGAMENT
#define LIGAMENT 10.0f
#endif
#ifndef BAKED
#define BAKED 0
#endif
#if BAKED
#ifndef BT0
#error "BAKED needs the topology words BT0 .. BT7"
#endif
#define BTOP(k) ((k) == 0 ? (unsigned)(BT0) : (k) == 1 ? (unsigned)(BT1) : (k) == 2 ? (unsigned)(BT2) : (k) == 3 ? (unsigned)(BT3) : (k) == 4 ? (unsigned)(BT4) : (k) == 5 ? (unsigned)(BT5) : (k) == 6 ? (unsigned)(BT6) : (unsigned)(BT7))
#define TOPOK(k) BTOP(k)
#else
#define TOPOK(k) topo[k]
#endif
// 1: the muscle state words live in registers; 0: in an L2 scratch, read
// and written once per muscle and substep (today's muscle model).
#ifndef MSTATE_REGS
#define MSTATE_REGS 0
#endif
// Unrolling of the loops over a lane's muscles (MPL with the state in
// registers, which index it).
// MC: every creature has exactly MC muscles (a baked plan), so the loop is
// fully unrolled with no counts to test.
#ifdef MC
#define MUSCLE_LOOP MC
#else
#define MUSCLE_LOOP MPL
#endif
#ifndef MUSCLE_UNROLL
#if MSTATE_REGS && !MUSCLE_MODEL
#define MUSCLE_UNROLL MPL
#elif defined(MC)
#define MUSCLE_UNROLL MC
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
// 1: the momentum and angular ledgers of the contact impulses are enforced
// every substep (the stub's rule); 0: audit only (the host's audit tools).
#ifndef LEDGER
#define LEDGER 1
#endif
#if W != 1 && W != 2
#error "W is 1 or 2"
#endif
#if W == 1 && !(CONTACT_MODEL && MUSCLE_MODEL)
#error "W = 1 is the lean physics: it needs MUSCLE_MODEL=1 and CONTACT_MODEL=1"
#endif
#if BAKED && W != 1
#error "BAKED is implemented for W = 1"
#endif

#if W == 2
#define NPL 4
#define LOGNPL 2
#define SW 44
#define FRCW 16
#else
#define NPL 8
#define LOGNPL 3
#define SW 64
#define FRCW 32
#endif
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
#define MUSCLE_DAMPER 0.15f
#define COMPLIANCE 0.1f
#define JOINT_LIMIT 2.2f
#define SPIN_CAP 15.0f
#define SPEED_CLAMP 20.0f
// The lane record, in fields of NPL floats per lane: inverse mass, radius,
// friction, x, y, rest length, topology word, pivot inverse mass, the rest
// angle of the joint to the parent (cos, sin), and the rod's drag coefficient,
// drag limit and damping gain (static per rod, folded at pack time).
#define F_INVM 0
#define F_RAD (1 * NPL)
#define F_MU (2 * NPL)
#define F_X (3 * NPL)
#define F_Y (4 * NPL)
#define F_LEN (5 * NPL)
#define F_TOPO (6 * NPL)
#define F_PINV (7 * NPL)
#define F_C0 (8 * NPL)
#define F_S0 (9 * NPL)
#define F_DRAG (10 * NPL)
#define F_MLIM (11 * NPL)
#define F_MROD (12 * NPL)
#define RF (13 * NPL)
#define RESULT_STRIDE 3
// Result words, in float4s: 0: com x, head below neck, lowest point, steps;
// 1: rounds or normal residual, contact nodes, drift, power; 2: rod
// residual, angular ledger residual, stamina, rounds.

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
// Reciprocal by the hardware approximation: one MUFU, no refinement.
__device__ __forceinline__ float frcp(float x) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
// Two floats as a half pair (low, high), and back.
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ float2 unpack_h2(unsigned v) {
    float x, y;
    asm("{\n.reg .b16 a, b;\nmov.b32 {a, b}, %2;\ncvt.f32.f16 %0, a;\ncvt.f32.f16 %1, b;\n}" : "=f"(x), "=f"(y) : "r"(v));
    return make_float2(x, y);
}
__device__ __forceinline__ float xch(float v) { return __shfl_xor_sync(FULL, v, 1); }
__device__ __forceinline__ unsigned xchu(unsigned v) { return __shfl_xor_sync(FULL, v, 1); }
#if W == 2
#define XSUM(v) ((v) + xch(v))
#else
#define XSUM(v) (v)
#endif
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

// A record field for all NPL nodes of the lane: two vector loads at W = 1.
#if W == 1
__device__ __forceinline__ void load_field(const float* __restrict__ lanes, size_t cidx, int fld, float (&o)[NPL]) {
    const float4* p4 = reinterpret_cast<const float4*>(lanes + cidx * RF + fld);
    const float4 a = __ldg(p4), b = __ldg(p4 + 1);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
#define LOADF(o, fld) load_field(lanes, cidx, (fld), o)
#else
#define LOADF(o, fld) do { _Pragma("unroll") for (int k_ = 0; k_ < NPL; k_++) { (o)[k_] = REC(fld, k_); } } while (0)
#endif

// Rod topology word: pivot node (5 bits), parent rod (5, 31 none), lower
// siblings kept (2), grandparent node (5, the parent's pivot, 31 none),
// valid (1).
#define T_PIVOT(t) ((t) & 31u)
#define T_PARENT(t) (((t) >> 5u) & 31u)
#define T_SR(t) (((t) >> 10u) & 3u)
#define T_GP(t) (((t) >> 12u) & 31u)
#define T_VALID(t) (((t) >> 17u) & 1u)

// One-hot target of a rod's parent. W = 2: bit q for local rod q, bit 4 + q
// for lane 0's rod q seen from lane 1, 0 without a parent. W = 1: bit q for
// rod q. Bit tests instead of index compares keep the compiler from turning
// register arrays into indexed local memory.
__device__ __forceinline__ unsigned parent_mask(unsigned t, unsigned lg) {
    const unsigned p = T_PARENT(t);
    const unsigned tn = p + 1u;
    const bool hasp = T_VALID(t) && p != 31u;
#if W == 2
    return hasp ? (((tn >> 2u) != lg) ? (0x10u << (tn & 3u)) : (1u << (tn & 3u))) : 0u;
#else
    (void)lg;
    return hasp ? (1u << tn) : 0u;
#endif
}

// The LDL^T factor of A = J M^-1 J^T over the creature's rods. `ipv` is the
// inverse mass of each local rod's pivot, `icm` of its child node; `rdx`,
// `rdy` hold lane 0's rod directions on lane 1 (W = 2 only).
__device__ __forceinline__ void factor(
    const unsigned (&topo)[NPL], const float (&dx)[NPL], const float (&dy)[NPL],
    const float (&rdx)[NPL], const float (&rdy)[NPL], const float (&ipv)[NPL], const float (&icm)[NPL],
    unsigned lg, float (&invD)[NPL], float (&Lf)[NPL][NB]) {
    float Dacc[NPL], La[NPL][NB];
#pragma unroll
    for (int k = 0; k < NPL; k++) {
        const unsigned t = TOPOK(k);
        const bool valid = T_VALID(t);
        Dacc[k] = valid ? ipv[k] + icm[k] : 1.0f;
        const unsigned pm = parent_mask(t, lg);
        float pdx = 0.0f, pdy = 0.0f;
#pragma unroll
        for (int q = 0; q < k; q++) {
            if (pm & (1u << q)) { pdx = dx[q]; pdy = dy[q]; }
        }
#if W == 2
#pragma unroll
        for (int q = 1; q < NPL; q++) {
            if (pm & (0x10u << q)) { pdx = rdx[q]; pdy = rdy[q]; }
        }
#endif
        La[k][0] = -(dx[k] * pdx + dy[k] * pdy) * ipv[k];
        const unsigned sr = valid ? T_SR(t) : 0u;
#pragma unroll
        for (int a = 1; a < NB; a++) {
#if W == 2
            const float sdx = (k - a >= 0) ? dx[(k - a) & (NPL - 1)] : rdx[(NPL + k - a) & (NPL - 1)];
            const float sdy = (k - a >= 0) ? dy[(k - a) & (NPL - 1)] : rdy[(NPL + k - a) & (NPL - 1)];
            La[k][a] = (unsigned)a <= sr ? (dx[k] * sdx + dy[k] * sdy) * ipv[k] : 0.0f;
#else
            if (k - a >= 0) {
                La[k][a] = (unsigned)a <= sr ? (dx[k] * dx[k - a] + dy[k] * dy[k - a]) * ipv[k] : 0.0f;
            } else {
                La[k][a] = 0.0f;
            }
#endif
        }
    }
#if W == 2
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
#else
    {
        {
#endif
#pragma unroll
            for (int k = NPL - 1; k >= 0; k--) {
                const unsigned t = TOPOK(k);
                const bool valid = T_VALID(t);
                // A rod whose two nodes are both held (projection) has no
                // pivot: it is left as it is.
                const float id = (valid && Dacc[k] > 1e-6f) ? frcp(Dacc[k]) : 0.0f;
                invD[k] = id;
#pragma unroll
                for (int a = 0; a < NB; a++) { Lf[k][a] = La[k][a] * id; }
                const unsigned pm = parent_mask(t, lg);
                const float vp = La[k][0] * Lf[k][0];
#pragma unroll
                for (int q = 0; q < k; q++) {
                    if (pm & (1u << q)) { Dacc[q] -= vp; }
                }
#if W == 2
#pragma unroll
                for (int q = 1; q < NPL; q++) {
                    if (pm & (0x10u << q)) { oD[q] -= vp; }
                }
#endif
#pragma unroll
                for (int a = 1; a < NB; a++) {
                    const float va = La[k][a] * Lf[k][a];
                    const float vpa = La[k][a] * Lf[k][0];
                    if (k - a >= 0) {
                        Dacc[(k - a) & (NPL - 1)] -= va;
                        La[(k - a) & (NPL - 1)][0] -= vpa;
                    }
#if W == 2
                    else {
                        oD[(NPL + k - a) & (NPL - 1)] -= va;
                        oL[(NPL + k - a) & (NPL - 1)][0] -= vpa;
                    }
#endif
#pragma unroll
                    for (int b = 1; b < a; b++) {
                        const float vab = La[k][a] * Lf[k][b];
                        if (k - b >= 0) {
                            La[(k - b) & (NPL - 1)][a - b] -= vab;
                        }
#if W == 2
                        else {
                            oL[(NPL + k - b) & (NPL - 1)][a - b] -= vab;
                        }
#endif
                    }
                }
            }
        }
    }
}

#if W == 2
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
            const unsigned pm = parent_mask(TOPOK(k), lg);
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
            const unsigned pm = parent_mask(TOPOK(k), lg);
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
#define SOLVE_ROWS 2
#define ZSET(z, k, v) do { (z)[0][k] = lg == 1u ? (v) : 0.0f; (z)[1][k] = lg == 1u ? 0.0f : (v); } while (0)
#define ZGET(z, k) (lg == 1u ? (z)[0][k] : (z)[1][k])
#else
// One creature per lane: S independent rows, forward in reverse rod order,
// backward in rod order, no exchange.
template <int S>
__device__ __forceinline__ void tree_solve(float (&z)[S][NPL], const unsigned (&topo)[NPL],
    const float (&invD)[NPL], const float (&Lf)[NPL][NB], unsigned lg) {
#pragma unroll
    for (int s = 0; s < S; s++) {
        float b[NPL];
#pragma unroll
        for (int q = 0; q < NPL; q++) { b[q] = z[s][q]; }
#pragma unroll
        for (int k = NPL - 1; k >= 0; k--) {
            const unsigned pm = parent_mask(TOPOK(k), lg);
            const float y = b[k];
            const float ly = Lf[k][0] * y;
#pragma unroll
            for (int q = 0; q < k; q++) {
                if (pm & (1u << q)) { b[q] -= ly; }
            }
#pragma unroll
            for (int a = 1; a < NB; a++) {
                if (k - a >= 0) { b[k - a] -= Lf[k][a] * y; }
            }
            z[s][k] = y * invD[k];
        }
        float x[NPL];
#pragma unroll
        for (int k = 0; k < NPL; k++) {
            const unsigned pm = parent_mask(TOPOK(k), lg);
            float xp = 0.0f;
#pragma unroll
            for (int q = 0; q < k; q++) {
                if (pm & (1u << q)) { xp = x[q]; }
            }
            float acc = z[s][k] - Lf[k][0] * xp;
#pragma unroll
            for (int a = 1; a < NB; a++) {
                if (k - a >= 0) { acc -= Lf[k][a] * x[k - a]; }
            }
            x[k] = acc;
            z[s][k] = acc;
        }
    }
}
#define SOLVE_ROWS 1
#define ZSET(z, k, v) ((z)[0][k] = (v))
#define ZGET(z, k) ((z)[0][k])
#endif

extern "C" __global__ void __launch_bounds__(BLOCK, MIN_BLOCKS) lane_lean(
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
    const Params p) {
    // SW words per lane. NODE: each lane's nodes (position, velocity). FRC:
    // each lane's private force table over the creature's 8 nodes. Today's
    // contact solve (W = 2) also uses the same bytes, as words SCR(q, lane
    // column) with q < 44, for the creature's Delassus matrix (words 0 to 17
    // of both lane columns) and each lane's rod factor, rod impulses and
    // directions (words 18 to 41 of its own column), so they leave the
    // registers. Every warp owns its words; the views never leave it, so
    // warps at different phases never touch each other's bytes.
    __shared__ float4 s_mem[(BLOCK / 32) * (SW / 4) * 32];
    float* const s_warp = reinterpret_cast<float*>(s_mem) + (threadIdx.x >> 5) * (SW * 32);
#if W == 2
    float4* const s_node4 = reinterpret_cast<float4*>(s_warp);
    float2* const s_frc2 = reinterpret_cast<float2*>(s_warp + FRCW * 32);
#endif
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
#if W == 2
    const unsigned lg = tid & 1u;
    const unsigned gl = lane & 30u;
    const unsigned gmask = 3u << gl;
#else
    const unsigned lg = 0u;
    const unsigned gl = lane;
#endif
#if W == 2
#define NODE(k, col) s_node4[(k) * 32 + (col)]
#define FRC(g, col) s_frc2[(g) * 32 + (col)]
#else
    // W = 1: one 32 B element per node and lane, the node (x, y, vx, vy) and
    // its force (x, y), so a node is one offset from the lane's base.
#define NODE(k, col) (*reinterpret_cast<float4*>(s_warp + ((k) * 32 + (col)) * 8))
#define FRC(g, col) (*reinterpret_cast<float2*>(s_warp + ((g) * 32 + (col)) * 8 + 4))
#endif
#define SCR(q, col) s_warp[(q) * 32 + (col)]
#define WAT(idx) SCR((idx) >> 1, gl + ((idx) & 1))
#define WIJ(i, j) WAT((i) >= (j) ? (i) * ((i) + 1) / 2 + (j) : (j) * ((j) + 1) / 2 + (i))
    // A node by its creature index: from the shared table, or from the
    // registers when the tree is compiled in.
#define NODEAT(a) NODE((a) & (NPL - 1), gl + ((a) >> LOGNPL))
#if BAKED
#define NODEV(a) make_float4(px[(a)], py[(a)], vx[(a)], vy[(a)])
#define SCAT(a, X, Y) do { fx[(a)] += (X); fy[(a)] += (Y); } while (0)
#else
#define NODEV(a) NODEAT(a)
#define SCAT(a, X, Y) do { float2 f_ = FRC((a), lane); f_.x += (X); f_.y += (Y); FRC((a), lane) = f_; } while (0)
#endif

    bool live = false, exhausted = false;
    unsigned cidx = 0u, mc = 0u, step = 0u, warm = 0u, prevc = 0u;
    // Registers hold the state; the statics (radius, friction, rest length,
    // pivot inverse mass) are read from the lane record in L2 when used.
    float invm[NPL], mass[NPL];
    unsigned topo[NPL];
    float px[NPL], py[NPL], vx[NPL], vy[NPL], ax[NPL];
#if MSTATE_REGS && !MUSCLE_MODEL
    unsigned ms[MPL];
#define MS_GET(k) ms[k]
#define MS_SET(k, v) (ms[k] = (v))
#else
#define MS_GET(k) mstate[mb + (size_t)(k) * W]
#define MS_SET(k, v) (mstate[mb + (size_t)(k) * W] = (v))
#endif
    float msum = 0.0f, rounds_sum = 0.0f, contact_sum = 0.0f, drift_max = 0.0f;
    float rn_max = 0.0f, rr_max = 0.0f, ang_max = 0.0f, pen_max = 0.0f;
    float st = 1.0f, icap = 0.0f, mtot = 1.0f;
#pragma unroll
    for (int k = 0; k < NPL; k++) {
        invm[k] = 0.0f; mass[k] = 0.0f; topo[k] = 0u; px[k] = 0.0f; py[k] = 0.0f; vx[k] = 0.0f; vy[k] = 0.0f; ax[k] = 0.0f;
    }
#if MSTATE_REGS && !MUSCLE_MODEL
#pragma unroll
    for (int k = 0; k < MPL; k++) { ms[k] = 0u; }
#endif
#define REC(f, k) lanes[((size_t)cidx * RF + (f) + (k)) * W + lg]
    // Friction anchors per node in L2, NaN when the node has none (today's
    // contact model; the lean one keeps them in registers).
#define ANC(k) anch[((size_t)cidx * NPL + (k)) * W + lg]
#define STASH(q) (reinterpret_cast<volatile float*>(s_warp)[(18 + (q)) * 32 + lane])

    for (;;) {
        if (!live && !exhausted) {
            unsigned got = 0u;
#if W == 2
            if (lg == 0u) { got = atomicAdd(counter, 1u); }
            got = __shfl_sync(gmask, got, tid & 30u);
#else
            got = atomicAdd(counter, 1u);
#endif
            if (got < p.count) {
                cidx = got;
                const uint4 h = heads[cidx];
#if W == 2
                mc = lg == 0u ? (h.x >> 16u) & 255u : h.x >> 24u;
#else
                mc = (h.x >> 8u) & 255u;
#endif
                icap = __uint_as_float(h.y);
                float msl = 0.0f;
                LOADF(invm, F_INVM);
                LOADF(px, F_X);
                LOADF(py, F_Y);
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    topo[k] = BAKED ? TOPOK(k) : __float_as_uint(REC(F_TOPO, k));
                    vx[k] = 0.0f; vy[k] = 0.0f;
                    ax[k] = px[k];
                    mass[k] = invm[k] > 0.0f ? 1.0f / invm[k] : 0.0f;
                    msl += mass[k];
#if !CONTACT_MODEL
                    ANC(k) = __uint_as_float(0x7fc00000u);
#endif
                }
#if W == 2
                mtot = msl + __shfl_xor_sync(gmask, msl, 1);
#else
                mtot = msl;
#endif
#if MUSCLE_MODEL
#pragma unroll
                for (int k = 0; k < NPL; k++) { roff[(size_t)cidx * 16u + lg * NPL + k] = 0.0f; }
#else
                {
                    const size_t mb0 = (size_t)cidx * MPL * W + lg;
#pragma unroll MUSCLE_UNROLL
                    for (int k = 0; k < MPL; k++) {
#if MSTATE_REGS
                        ms[k] = pack_ed(1.0f, 0.0f);
#else
                        mstate[mb0 + (size_t)k * W] = pack_ed(1.0f, 0.0f);
#endif
                        roff[mb0 + (size_t)k * W] = 0.0f;
                    }
                }
#endif
                step = 0u; warm = 0u; prevc = 0u; st = 1.0f;
                msum = 0.0f; rounds_sum = 0.0f; contact_sum = 0.0f; drift_max = 0.0f;
                rn_max = 0.0f; rr_max = 0.0f; ang_max = 0.0f; pen_max = 0.0f;
                live = true;
            } else {
                exhausted = true;
                mc = 0u;
                cidx = 0u;
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    invm[k] = 0.0f; mass[k] = 0.0f; topo[k] = 0u; px[k] = 0.0f; py[k] = 0.0f; vx[k] = 0.0f; vy[k] = 0.0f; ax[k] = 0.0f;
                }
            }
        }
        if (!__any_sync(FULL, live)) { break; }
        const unsigned wm = __reduce_max_sync(FULL, mc);
        const float t0 = (float)step * DT;
        const size_t mb = (size_t)cidx * MPL * W + lg;
#if !MUSCLE_MODEL
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
#endif
        unsigned cw = 0u;
        unsigned touch = 0u;
        // Per-step state: forces held over the substeps, the rod geometry and
        // its factor (kept across the substeps when lagged), the centre of
        // mass and inertia of the ledger.
        float fx[NPL], fy[NPL];
        float dx[NPL], dy[NPL], rdx[NPL], rdy[NPL], ilr[NPL], rhs0[NPL];
        float invD[NPL], Lf[NPL][NB];
        float cxr = 0.0f, cyr = 0.0f, iic = 0.0f;
#pragma unroll
        for (int k = 0; k < NPL; k++) { fx[k] = 0.0f; fy[k] = 0.0f; rdx[k] = 0.0f; rdy[k] = 0.0f; }
#pragma unroll 1
        for (int sub = 0; sub < SUBSTEPS; sub++) {
            const bool geom_now = !LAGGED_FACTOR || sub == 0;
            // @S table
            // Node table, and a clear force table.
#pragma unroll
            for (int k = 0; k < NPL; k++) { NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]); }
#pragma unroll
            for (int g = 0; g < W * NPL; g++) { FRC(g, lane) = make_float2(0.0f, 0.0f); }
            __syncwarp();
            // @S geometry
            // Rod directions and lengths every substep (the impulses and the
            // right-hand sides want the current direction); the factor and
            // the explicit forces once per step when lagged.
            float lnr[NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned t = TOPOK(k);
                dx[k] = 0.0f; dy[k] = 0.0f; ilr[k] = 0.0f; lnr[k] = 0.0f;
                if (T_VALID(t)) {
                    const unsigned a = T_PIVOT(t);
                    const float4 na = NODEV(a);
                    const float ddx = px[k] - na.x, ddy = py[k] - na.y;
                    const float d2 = ddx * ddx + ddy * ddy;
                    const float il = rsqrtf(fmaxf(d2, 1e-12f));
                    dx[k] = ddx * il; dy[k] = ddy * il; ilr[k] = il; lnr[k] = d2 * il;
                }
            }
#if W == 2
#pragma unroll
            for (int q = 0; q < NPL; q++) { rdx[q] = xch(dx[q]); rdy[q] = xch(dy[q]); }
#endif
            if (geom_now) {
                // @S factor
                float ipv[NPL];
                LOADF(ipv, F_PINV);
                factor(topo, dx, dy, rdx, rdy, ipv, invm, lg, invD, Lf);
                // @S forces
                // Air drag at each rod's midpoint, half to each node, and joint
                // damping as a couple on the rod and the opposite on its parent;
                // with the lagged factor once per step and held.
#pragma unroll
                for (int k = 0; k < NPL; k++) { fx[k] = 0.0f; fy[k] = 0.0f; }
                float drg[NPL], mlm[NPL], mrd[NPL], rcr[NPL], rsr[NPL];
                LOADF(drg, F_DRAG); LOADF(mlm, F_MLIM); LOADF(mrd, F_MROD);
#if LIMITS_AS_IMPULSES
                if (sub == 0) { LOADF(rcr, F_C0); LOADF(rsr, F_S0); }
#endif
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const unsigned t = TOPOK(k);
                    if (T_VALID(t) && !NO_DAMP) {
                        const unsigned a = T_PIVOT(t);
                        const float4 na = NODEV(a);
                        const float il = ilr[k], l = lnr[k];
                        const float wx = 0.5f * (vx[k] + na.z), wy = 0.5f * (vy[k] + na.w);
                        const float speed = sqrtf(wx * wx + wy * wy);
                        const float strength = fminf(drg[k] * l * speed, mlm[k]);
                        const float hx = -0.5f * wx * strength, hy = -0.5f * wy * strength;
                        fx[k] += hx; fy[k] += hy;
                        float fax = hx, fay = hy;
                        const float rvx = vx[k] - na.z, rvy = vy[k] - na.w;
                        const float w_rod = (dx[k] * rvy - dy[k] * rvx) * il;
                        const unsigned g = T_GP(t);
                        float w_par = 0.0f, dxp = 0.0f, dyp = 0.0f, ilp = 0.0f;
                        if (g != 31u) {
#if BAKED
                            const unsigned pr = T_PARENT(t);
                            dxp = dx[pr]; dyp = dy[pr]; ilp = ilr[pr];
#else
                            const float4 ng0 = NODEV(g);
                            const float pdx = na.x - ng0.x, pdy = na.y - ng0.y;
                            ilp = rsqrtf(fmaxf(pdx * pdx + pdy * pdy, 1e-12f));
                            dxp = pdx * ilp; dyp = pdy * ilp;
#endif
                            const float4 ng = NODEV(g);
                            w_par = (dxp * (na.w - ng.w) - dyp * (na.z - ng.z)) * ilp;
                        }
                        const float tau = mrd[k] * l * l * (w_rod - w_par);
                        const float tl = tau * il;
                        fx[k] += -dy[k] * tl; fy[k] += dx[k] * tl;
                        fax -= -dy[k] * tl; fay -= dx[k] * tl;
                        float fgx = 0.0f, fgy = 0.0f;
                        if (g != 31u) {
                            // The parent rod takes -tau.
                            const float tp = tau * ilp;
                            fax -= -dyp * tp; fay -= dxp * tp;
                            fgx = -dyp * tp; fgy = dxp * tp;
#if LIMITS_AS_IMPULSES
                            // The joint's limit and the spin cap, once per
                            // step: the relative angular velocity is held in
                            // [lo, hi], where hi is the rate that reaches the
                            // limit within the step (or recovers from past
                            // it) and the spin cap bounds both. The angular
                            // impulse is a couple on the rod and the
                            // opposite on its parent, softened by the
                            // ligament's compliance.
                            if (sub == 0) {
                                const float cs = dxp * dx[k] + dyp * dy[k], sn = dxp * dy[k] - dyp * dx[k];
                                const float rc = cs * rcr[k] + sn * rsr[k];
                                const float rs = sn * rcr[k] - cs * rsr[k];
                                const float th = 2.0f * rs * frcp(1.0f + rc + 1e-3f);
                                const float wrel = w_rod - w_par;
                                const float hi = clampf((JOINT_LIMIT - th) * RATE, -SPIN_CAP, SPIN_CAP);
                                const float lo = clampf((-JOINT_LIMIT - th) * RATE, -SPIN_CAP, SPIN_CAP);
                                const float dw = clampf(wrel, lo, hi) - wrel;
                                const float ipp = ipv[k];
                                const float iinv = (invm[k] + ipp) * il * il + (ipp + ipp) * ilp * ilp;
                                // With the held forces the impulse is spread over the step's substeps.
                                const float jf = dw * frcp(iinv + LIGAMENT) * (LAGGED_FACTOR ? RATE : INV_HS);
                                const float cx = -dy[k] * jf * il, cy = dx[k] * jf * il;
                                const float qx = -dyp * jf * ilp, qy = dxp * jf * ilp;
                                fx[k] += cx; fy[k] += cy;
                                fax += -cx - qx; fay += -cy - qy;
                                fgx += qx; fgy += qy;
                            }
#endif
                            SCAT(g, fgx, fgy);
                        }
                        SCAT(a, fax, fay);
                    }
                }
#if LAGGED_FACTOR && !BAKED
                // Fold the shared force table into the held forces.
                __syncwarp();
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const unsigned g = lg * NPL + k;
                    const float2 f0 = FRC(g, lane);
#if W == 2
                    const float2 f1 = FRC(g, lane ^ 1u);
                    fx[k] += f0.x + f1.x; fy[k] += f0.y + f1.y;
#else
                    fx[k] += f0.x; fy[k] += f0.y;
#endif
                }
                __syncwarp();
#pragma unroll
                for (int g = 0; g < W * NPL; g++) { FRC(g, lane) = make_float2(0.0f, 0.0f); }
                __syncwarp();
#endif
            }
#if MUSCLE_MODEL && LAGGED_FACTOR && !MUSCLE_ANCHORS && !defined(NO_HOLD)
#define MUSCLE_HELD 1
#else
#define MUSCLE_HELD 0
#endif
#if MUSCLE_HELD
            // @S muscles
            // Muscles: two nodes, no state. A trapezoid of period, phase and
            // duty with a ramp of two substeps on each edge, since the step's
            // start or the limb's last touchdown; force = cap x strength x
            // a(t) x stamina x Hill, pull only, with the damper. The record's
            // end fields are shared-table offsets, so a node is one add away.
            float pw = 0.0f;
            const float tsub = t0 + (float)sub * HS;
#pragma unroll MUSCLE_UNROLL
            for (int k = 0; k < MUSCLE_LOOP; k++) {
#ifndef MC
                if ((unsigned)k >= wm) { break; }
                if ((unsigned)k < mc && !NO_MUSCLES) {
#else
                {
#endif
                    const float4 A = msa[2u * (mb + (size_t)k * W)];
                    const unsigned pk = __float_as_uint(A.x);
#if W == 1
                    const unsigned oa = pk & 0x1fffu, ob = (pk >> 13u) & 0x1fffu;
                    char* const col = reinterpret_cast<char*>(s_warp) + lane * 32u;
                    const float4 e0 = *reinterpret_cast<float4*>(col + oa), e1 = *reinterpret_cast<float4*>(col + ob);
                    const unsigned limb = (pk >> 26u) & 15u;
#else
                    const unsigned ea = (pk >> 5u) & 31u, eb = (pk >> 15u) & 31u;
                    const float4 e0 = NODEAT(ea), e1 = NODEAT(eb);
                    const unsigned limb = ((pk >> 25u) & 1u) ? ((pk >> 20u) & 31u) : 8u;
#endif
                    uint2* const hp = reinterpret_cast<uint2*>(mstate) + (mb + (size_t)k * W);
                    float dirx, diry, a0, a1;
                    if (sub == 0) {
                        // The step's geometry and activation, held for the
                        // substeps: the direction and the activation at the
                        // step's two ends, as half pairs.
                        const float4 R = msa[2u * (mb + (size_t)k * W) + 1u];
                        const float td = __ldcg(&roff[(size_t)cidx * 16u + limb]);
                        const float x0 = (t0 - td) * R.x + R.y;
                        const float p0 = x0 - floorf(x0);
                        const float x1 = x0 + DT * R.x;
                        const float p1 = x1 - floorf(x1);
                        a0 = clampf((R.z - fabsf(p0 - R.z)) * R.w, 0.0f, 1.0f);
                        a1 = clampf((R.z - fabsf(p1 - R.z)) * R.w, 0.0f, 1.0f);
                        const float ddx = e1.x - e0.x, ddy = e1.y - e0.y;
                        const float inv = rsqrtf(ddx * ddx + ddy * ddy + 1e-12f);
                        dirx = ddx * inv; diry = ddy * inv;
                        *hp = make_uint2(pack_h2(dirx, diry), pack_h2(a0, a1));
                    } else {
                        const uint2 hv = *hp;
                        const float2 fd = unpack_h2(hv.x);
                        const float2 fa = unpack_h2(hv.y);
                        dirx = fd.x; diry = fd.y; a0 = fa.x; a1 = fa.y;
                    }
                    const float act = a0 + (a1 - a0) * (((float)sub + 0.5f) * (1.0f / SUBSTEPS));
                    const float rel = (e1.z - e0.z) * dirx + (e1.w - e0.w) * diry;
                    const float cap = A.y;
                    const float drive = cap * act * st * clampf(1.0f + rel * A.z, 0.0f, 1.0f);
                    const float mag = clampf(drive + rel * MUSCLE_DAMPER, -cap, cap);
                    pw += drive * fmaxf(-rel, 0.0f);
#ifdef DIAG
                    msum += mag * rel;
#endif
                    const float gx = dirx * mag, gy = diry * mag;
                    float2 f;
#if W == 1
                    float2* const fa_ = reinterpret_cast<float2*>(col + oa + 16u);
                    float2* const fb_ = reinterpret_cast<float2*>(col + ob + 16u);
                    f = *fa_; f.x += gx; f.y += gy; *fa_ = f;
                    f = *fb_; f.x -= gx; f.y -= gy; *fb_ = f;
#else
                    f = FRC(ea, lane); f.x += gx; f.y += gy; FRC(ea, lane) = f;
                    f = FRC(eb, lane); f.x -= gx; f.y -= gy; FRC(eb, lane) = f;
#endif
                }
            }
            pw = XSUM(pw);
            st = clampf(st - pw * HS * icap + MUSCLE_RECOVERY * p.recovery * HS * (1.0f - st), 0.0f, 1.0f);
#else
            // @S muscles
#if MUSCLE_MODEL
            // Muscles: two nodes, no state. A trapezoid of period, phase and
            // duty with a ramp of two substeps on each edge, since the step's
            // start or the limb's last touchdown; force = cap x strength x
            // a(t) x stamina x Hill, pull only, with the damper. The record's
            // end fields are shared-table offsets, so a node is one add away.
            float pw = 0.0f;
            const float tsub = t0 + (float)sub * HS;
#pragma unroll MUSCLE_UNROLL
            for (int k = 0; k < MUSCLE_LOOP; k++) {
#ifndef MC
                if ((unsigned)k >= wm) { break; }
                if ((unsigned)k < mc && !NO_MUSCLES) {
#else
                {
#endif
                    const float4 A = msa[2u * (mb + (size_t)k * W)];
                    const float4 R = msa[2u * (mb + (size_t)k * W) + 1u];
                    const unsigned pk = __float_as_uint(A.x);
#if MUSCLE_ANCHORS
                    const unsigned na_ = pk & 31u, nb_ = (pk >> 5u) & 31u, nc_ = (pk >> 10u) & 31u, nd_ = (pk >> 15u) & 31u;
                    const float4 f0 = NODEAT(na_), f1 = NODEAT(nb_), f2 = NODEAT(nc_), f3 = NODEAT(nd_);
                    const float2 an = unpack_un(__float_as_uint(A.w));
                    const float4 e0 = make_float4(f0.x + (f1.x - f0.x) * an.x, f0.y + (f1.y - f0.y) * an.x, f0.z + (f1.z - f0.z) * an.x, f0.w + (f1.w - f0.w) * an.x);
                    const float4 e1 = make_float4(f2.x + (f3.x - f2.x) * an.y, f2.y + (f3.y - f2.y) * an.y, f2.z + (f3.z - f2.z) * an.y, f2.w + (f3.w - f2.w) * an.y);
                    const float td = ((pk >> 25u) & 1u) ? __ldcg(&roff[(size_t)cidx * 16u + ((pk >> 20u) & 31u)]) : 0.0f;
#elif W == 1
                    // Bits 0 to 12: end a's byte offset in the lane's column
                    // (node x 1024), 13 to 25: end b's, 26 to 29: the limb
                    // whose clock sets the phase (8: none, its clock is 0).
                    const unsigned oa = pk & 0x1fffu, ob = (pk >> 13u) & 0x1fffu;
                    char* const col = reinterpret_cast<char*>(s_warp) + lane * 32u;
                    const float4 e0 = *reinterpret_cast<float4*>(col + oa), e1 = *reinterpret_cast<float4*>(col + ob);
                    const float td = __ldcg(&roff[(size_t)cidx * 16u + ((pk >> 26u) & 15u)]);
#else
                    const unsigned ea = (pk >> 5u) & 31u, eb = (pk >> 15u) & 31u;
                    const float4 e0 = NODEAT(ea), e1 = NODEAT(eb);
                    const float td = ((pk >> 25u) & 1u) ? __ldcg(&roff[(size_t)cidx * 16u + ((pk >> 20u) & 31u)]) : 0.0f;
#endif
                    const float x = (tsub - td) * R.x + R.y;
                    const float ph = x - floorf(x);
                    const float act = clampf((R.z - fabsf(ph - R.z)) * R.w, 0.0f, 1.0f);
                    const float ddx = e1.x - e0.x, ddy = e1.y - e0.y;
                    const float inv = rsqrtf(ddx * ddx + ddy * ddy + 1e-12f);
                    const float dirx = ddx * inv, diry = ddy * inv;
                    const float rel = (e1.z - e0.z) * dirx + (e1.w - e0.w) * diry;
                    const float cap = A.y;
                    const float drive = cap * act * st * clampf(1.0f + rel * A.z, 0.0f, 1.0f);
                    const float mag = clampf(drive + rel * MUSCLE_DAMPER, -cap, cap);
                    pw += drive * fmaxf(-rel, 0.0f);
#ifdef DIAG
                    msum += mag * rel;
#endif
                    const float gx = dirx * mag, gy = diry * mag;
                    float2 f;
#if MUSCLE_ANCHORS
                    f = FRC(na_, lane); f.x += (1.0f - an.x) * gx; f.y += (1.0f - an.x) * gy; FRC(na_, lane) = f;
                    f = FRC(nb_, lane); f.x += an.x * gx; f.y += an.x * gy; FRC(nb_, lane) = f;
                    f = FRC(nc_, lane); f.x -= (1.0f - an.y) * gx; f.y -= (1.0f - an.y) * gy; FRC(nc_, lane) = f;
                    f = FRC(nd_, lane); f.x -= an.y * gx; f.y -= an.y * gy; FRC(nd_, lane) = f;
#elif W == 1
                    float2* const fa = reinterpret_cast<float2*>(col + oa + 16u);
                    float2* const fb = reinterpret_cast<float2*>(col + ob + 16u);
                    f = *fa; f.x += gx; f.y += gy; *fa = f;
                    f = *fb; f.x -= gx; f.y -= gy; *fb = f;
#else
                    f = FRC(ea, lane); f.x += gx; f.y += gy; FRC(ea, lane) = f;
                    f = FRC(eb, lane); f.x -= gx; f.y -= gy; FRC(eb, lane) = f;
#endif
                }
            }
            pw = XSUM(pw);
            st = clampf(st - pw * HS * icap + MUSCLE_RECOVERY * p.recovery * HS * (1.0f - st), 0.0f, 1.0f);
#else
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
#endif
#endif
            __syncwarp();
            // @S free
            // Free velocities.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned g = lg * NPL + k;
                const float2 f0 = FRC(g, lane);
#if W == 2
                const float2 f1 = FRC(g, lane ^ 1u);
                const float gx = fx[k] + f0.x + f1.x, gy = fy[k] + f0.y + f1.y;
#else
                const float gx = fx[k] + f0.x, gy = fy[k] + f0.y;
#endif
                const bool on = invm[k] > 0.0f;
                vx[k] += HS * invm[k] * gx;
                vy[k] += on ? HS * (invm[k] * gy - p.gravity) : 0.0f;
            }
            __syncwarp();
#if !CONTACT_MODEL
            // Node table with the free velocities; node statics for the rows.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]);
                FRC(2 * k, lane) = make_float2(invm[k], REC(8, k) * p.friction);
                FRC(2 * k + 1, lane) = make_float2(REC(4, k), ANC(k));
            }
            __syncwarp();
#elif !BAKED
#pragma unroll
            for (int k = 0; k < NPL; k++) { NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]); }
            __syncwarp();
#endif
#if !CONTACT_MODEL
#ifdef DIAG
            float ctg[NPL];
#endif
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned t = topo[k];
                rhs0[k] = 0.0f;
#ifdef DIAG
                ctg[k] = 0.0f;
#endif
                if (T_VALID(t)) {
                    const unsigned a = T_PIVOT(t);
                    const float4 na = NODE(a & 3u, gl + (a >> 2u));
                    const float rvx = vx[k] - na.z, rvy = vy[k] - na.w;
                    const float rn = rvx * dx[k] + rvy * dy[k];
                    const float perp2 = fmaxf(rvx * rvx + rvy * rvy - rn * rn, 0.0f);
                    rhs0[k] = -rn - perp2 * HS * ilr[k];
#ifdef DIAG
                    ctg[k] = -perp2 * HS * ilr[k];
#endif
                }
            }
            // @S contact
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
#else
            // @S contact
            // Per-node impulses against each node's own mass, then the exact
            // rod solve as the coupling, NPASS times.
            float rad[NPL], mu[NPL];
            LOADF(rad, F_RAD);
            LOADF(mu, F_MU);
#pragma unroll
            for (int k = 0; k < NPL; k++) { mu[k] = NO_FRICTION ? 0.0f : mu[k] * p.friction; }
            unsigned slid = 0u;
            touch = 0u;
#if LEDGER
            // @S ledgerA
            // The ledger's first stage: momentum and angular momentum about
            // the origin before the impulses, and once per step the centre
            // of mass and the inertia about it.
            float pbx = 0.0f, pby = 0.0f, lbz = 0.0f;
            float smx = 0.0f, smy = 0.0f, sq = 0.0f;
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const float mvx = mass[k] * vx[k], mvy = mass[k] * vy[k];
                pbx += mvx; pby += mvy;
                lbz = fmaf(px[k], mvy, lbz); lbz = fmaf(-py[k], mvx, lbz);
                if (sub == 0) {
                    const float mxk = mass[k] * px[k], myk = mass[k] * py[k];
                    smx += mxk; smy += myk;
                    sq = fmaf(mxk, px[k], sq); sq = fmaf(myk, py[k], sq);
                }
            }
            if (sub == 0) {
                const float im = frcp(fmaxf(mtot, 1e-6f));
                smx = XSUM(smx); smy = XSUM(smy); sq = XSUM(sq);
                cxr = smx * im; cyr = smy * im;
                iic = frcp(fmaxf(sq - mtot * (cxr * cxr + cyr * cyr), 1e-6f));
            }
#endif
#if LEDGER
            float ex = 0.0f, ey = 0.0f, tz = 0.0f;
#endif
#pragma unroll
            for (int pass = 0; pass < NPASS; pass++) {
                // @S nodes
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const float gap = py[k] - rad[k];
                    if (pass == 0) { touch |= ((gap + HS * vy[k] <= 0.0f && invm[k] > 0.0f && !NO_CONTACT) ? 1u : 0u) << k; }
                    const float tn = -fmaxf(gap, 0.0f) * INV_HS;
                    const float dvn = fmaxf(tn - vy[k], 0.0f);
                    vy[k] += dvn;
                    const float cone = mu[k] * dvn;
                    const float tvx = (ax[k] - px[k]) * INV_HS;
                    const float want = tvx - vx[k];
                    float dvt = clampf(want, -cone, cone);
                    // The clean rule: friction never speeds up a slide the
                    // anchor does not ask for.
                    dvt = (dvt * vx[k] > 0.0f && fabsf(tvx) <= fabsf(vx[k])) ? 0.0f : dvt;
                    vx[k] += dvt;
                    if (pass == 0) { slid |= (fabsf(want) > cone ? 1u : 0u) << k; }
#if LEDGER
                    const float ix_ = mass[k] * dvt, iy_ = mass[k] * dvn;
                    ex += ix_; ey += iy_;
                    tz = fmaf(px[k], iy_, tz); tz = fmaf(-py[k], ix_, tz);
#endif
                }
                // @S rod
#if !BAKED
#pragma unroll
                for (int k = 0; k < NPL; k++) { NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]); }
                __syncwarp();
#endif
                {
                    float z[SOLVE_ROWS][NPL];
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        const unsigned t = TOPOK(k);
                        float r = 0.0f;
                        if (T_VALID(t)) {
                            const unsigned a = T_PIVOT(t);
                            const float4 na = NODEV(a);
                            const float rvx = vx[k] - na.z, rvy = vy[k] - na.w;
                            const float rn = rvx * dx[k] + rvy * dy[k];
                            const float perp2 = fmaxf(rvx * rvx + rvy * rvy - rn * rn, 0.0f);
                            r = -rn - perp2 * HS * ilr[k];
                        }
                        ZSET(z, k, r);
                    }
                    tree_solve<SOLVE_ROWS>(z, topo, invD, Lf, lg);
#if BAKED
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        const unsigned t = TOPOK(k);
                        if (T_VALID(t)) {
                            const unsigned a = T_PIVOT(t);
                            const float mu_k = ZGET(z, k);
                            const float ix = dx[k] * mu_k, iy = dy[k] * mu_k;
                            vx[k] += invm[k] * ix; vy[k] += invm[k] * iy;
                            vx[a] -= invm[a] * ix; vy[a] -= invm[a] * iy;
                        }
                    }
#else
#pragma unroll
                    for (int g = 0; g < W * NPL; g++) { FRC(g, lane) = make_float2(0.0f, 0.0f); }
                    __syncwarp();
                    float ix[NPL], iy[NPL];
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        const float mu_k = ZGET(z, k);
                        ix[k] = dx[k] * mu_k; iy[k] = dy[k] * mu_k;
                        if (T_VALID(TOPOK(k))) {
                            const unsigned a = T_PIVOT(TOPOK(k));
                            float2 f = FRC(a, lane);
                            f.x -= ix[k]; f.y -= iy[k];
                            FRC(a, lane) = f;
                        }
                    }
                    __syncwarp();
#pragma unroll
                    for (int k = 0; k < NPL; k++) {
                        const unsigned g = lg * NPL + k;
                        const float2 f0 = FRC(g, lane);
#if W == 2
                        const float2 f1 = FRC(g, lane ^ 1u);
                        vx[k] += invm[k] * (ix[k] + f0.x + f1.x);
                        vy[k] += invm[k] * (iy[k] + f0.y + f1.y);
#else
                        vx[k] += invm[k] * (ix[k] + f0.x);
                        vy[k] += invm[k] * (iy[k] + f0.y);
#endif
                    }
                    __syncwarp();
#endif
                }
            }
#if LEDGER
            // @S ledgerB
            // The ledger's second stage: the impulses of the substep must
            // have moved the momentum by exactly the contact impulses and
            // the angular momentum by their torque; the rounding rest goes
            // back as one uniform velocity and one rotation about the
            // centre of mass.
            {
                float pax = 0.0f, pay = 0.0f, laz = 0.0f;
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const float mvx = mass[k] * vx[k], mvy = mass[k] * vy[k];
                    pax += mvx; pay += mvy;
                    laz = fmaf(px[k], mvy, laz); laz = fmaf(-py[k], mvx, laz);
                }
                const float im = frcp(fmaxf(mtot, 1e-6f));
                const float rpx = NO_LEDGER ? 0.0f : XSUM(pax - pbx - ex);
                const float rpy = NO_LEDGER ? 0.0f : XSUM(pay - pby - ey);
                const float rl = NO_LEDGER ? 0.0f : XSUM(laz - lbz - tz);
                const float sx = -rpx * im, sy = -rpy * im;
                const float wc = (-rl + (cxr * rpy - cyr * rpx)) * iic;
#ifdef DIAG
                ang_max = fmaxf(ang_max, fabsf(rl));
#endif
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const bool on_ = invm[k] > 0.0f;
                    vx[k] += on_ ? sx - wc * (py[k] - cyr) : 0.0f;
                    vy[k] += on_ ? sy + wc * (px[k] - cxr) : 0.0f;
                }
            }
#endif
#ifdef DIAG
            {
                // Residuals after the impulses: normal rows against what the
                // gap allows, rods against their centripetal targets.
                float rn = 0.0f, rr = 0.0f;
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const float tn = -fmaxf(py[k] - rad[k], 0.0f) * INV_HS;
                    if ((touch >> k) & 1u) { rn = fmaxf(rn, fmaxf(tn - vy[k], 0.0f)); }
                }
#if !BAKED
                __syncwarp();
#pragma unroll
                for (int k = 0; k < NPL; k++) { NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]); }
                __syncwarp();
#endif
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    const unsigned t = TOPOK(k);
                    if (T_VALID(t)) {
                        const unsigned a = T_PIVOT(t);
                        const float4 na = NODEV(a);
                        const float rvx = vx[k] - na.z, rvy = vy[k] - na.w;
                        const float rnn = rvx * dx[k] + rvy * dy[k];
                        const float perp2 = fmaxf(rvx * rvx + rvy * rvy - rnn * rnn, 0.0f);
                        rr = fmaxf(rr, fabsf(rnn + perp2 * HS * ilr[k]));
                    }
                }
                __syncwarp();
                rn_max = fmaxf(rn_max, rn);
                rr_max = fmaxf(rr_max, rr);
            }
#endif
            // @S euler
            // Semi-implicit Euler; the penetration left is recovered by a
            // position nudge (a split impulse); anchors are set on touching
            // and moved by sliding. The stub's synthetic bodies and explicit
            // forces are not a tuned physics: a speed clamp keeps every trial
            // finite, so all creatures run their full length.
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                vx[k] = clampf(vx[k], -SPEED_CLAMP, SPEED_CLAMP);
                vy[k] = clampf(vy[k], -SPEED_CLAMP, SPEED_CLAMP);
                px[k] += HS * vx[k];
                py[k] += HS * vy[k];
#ifdef DIAG
                pen_max = fmaxf(pen_max, rad[k] - py[k]);
#endif
                py[k] += PUSH_OUT * fmaxf(rad[k] - py[k], 0.0f);
                ax[k] = (((touch >> k) & 1u) && !((slid >> k) & 1u)) ? ax[k] : px[k];
            }
            contact_sum += (float)__popc(touch);
#endif
        }
        // @S projection
        // Drift projection: rod lengths back to rest with the contact nodes
        // held (infinite mass): one factor and one tree solve.
        unsigned held = 0u;
        {
#if CONTACT_MODEL
            held = touch << (lg * NPL);
#if W == 2
            held |= xchu(held);
#endif
#else
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
#endif
            __syncwarp();
#if !BAKED
#pragma unroll
            for (int k = 0; k < NPL; k++) { NODE(k, lane) = make_float4(px[k], py[k], vx[k], vy[k]); }
            __syncwarp();
#endif
            float pdx_[NPL], pdy_[NPL], ipe[NPL], icm[NPL], lenr[NPL], pinr[NPL];
            LOADF(lenr, F_LEN);
            LOADF(pinr, F_PINV);
            float z1[SOLVE_ROWS][NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned t = TOPOK(k);
                pdx_[k] = 0.0f; pdy_[k] = 0.0f;
                float c = 0.0f;
                if (T_VALID(t)) {
                    const unsigned a = T_PIVOT(t);
                    const float4 na = NODEV(a);
                    const float ddx = px[k] - na.x, ddy = py[k] - na.y;
                    const float l = sqrtf(ddx * ddx + ddy * ddy);
                    const float il = frcp(fmaxf(l, 1e-6f));
                    pdx_[k] = ddx * il; pdy_[k] = ddy * il;
                    c = l - lenr[k];
                }
                drift_max = fmaxf(drift_max, fabsf(c));
                ipe[k] = ((held >> T_PIVOT(t)) & 1u) ? 0.0f : pinr[k];
                icm[k] = ((held >> (lg * NPL + k)) & 1u) ? 0.0f : invm[k];
                ZSET(z1, k, -c);
            }
#if W == 2
#pragma unroll
            for (int q = 0; q < NPL; q++) { rdx[q] = xch(pdx_[q]); rdy[q] = xch(pdy_[q]); }
#endif
            factor(topo, pdx_, pdy_, rdx, rdy, ipe, icm, lg, invD, Lf);
            tree_solve<SOLVE_ROWS>(z1, topo, invD, Lf, lg);
#if BAKED
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const unsigned t = TOPOK(k);
                if (T_VALID(t)) {
                    const unsigned a = T_PIVOT(t);
                    const float mu_k = ZGET(z1, k);
                    const float ix = pdx_[k] * mu_k, iy = pdy_[k] * mu_k;
                    px[k] += icm[k] * ix; py[k] += icm[k] * iy;
                    px[a] -= icm[a] * ix; py[a] -= icm[a] * iy;
                }
            }
#else
#pragma unroll
            for (int g = 0; g < W * NPL; g++) { FRC(g, lane) = make_float2(0.0f, 0.0f); }
            __syncwarp();
            float ix[NPL], iy[NPL];
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const float mu_k = ZGET(z1, k);
                ix[k] = pdx_[k] * mu_k; iy[k] = pdy_[k] * mu_k;
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
                const float2 f0 = FRC(g, lane);
#if W == 2
                const float2 f1 = FRC(g, lane ^ 1u);
                px[k] += icm[k] * (ix[k] + f0.x + f1.x);
                py[k] += icm[k] * (iy[k] + f0.y + f1.y);
#else
                px[k] += icm[k] * (ix[k] + f0.x);
                py[k] += icm[k] * (iy[k] + f0.y);
#endif
            }
            __syncwarp();
#endif
        }
        // @S metrics
        // Metrics, the fall rule and the end of the trial, once per step.
        {
            float mm = 0.0f, mx = 0.0f, low = 1e20f;
            bool bad = false;
            float radm[NPL];
            LOADF(radm, F_RAD);
#pragma unroll
            for (int k = 0; k < NPL; k++) {
                const float m = mass[k];
                mm += m; mx += m * px[k];
                if (invm[k] > 0.0f) { low = fminf(low, py[k] - radm[k]); }
                bad |= !(fabsf(px[k]) <= 1e6f && fabsf(py[k]) <= 1e6f);
            }
#if W == 2
            const float com_x = (mx + xch(mx)) / fmaxf(mm + xch(mm), 1e-6f);
            low = fminf(low, xch(low));
            bad = ((__ballot_sync(FULL, bad) >> gl) & 3u) != 0u;
            const float ms_all = msum + xch(msum);
            const float head_y = __shfl_sync(FULL, py[0], tid & 30u);
            const float neck_y = __shfl_sync(FULL, py[1], tid & 30u);
#else
            const float com_x = mx / fmaxf(mm, 1e-6f);
            const float ms_all = msum;
            const float head_y = py[0];
            const float neck_y = py[1];
#endif
            // Touchdowns restart the rhythm of the muscles that sense them.
            const unsigned down = held & ~prevc;
            if (down != 0u && live) {
#if MUSCLE_MODEL
#pragma unroll
                for (int k = 0; k < NPL; k++) {
                    if ((down >> (lg * NPL + k)) & 1u) { roff[(size_t)cidx * 16u + lg * NPL + k] = t0 + DT; }
                }
#else
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
#endif
            }
            prevc = held;
            if (live) {
                step += 1u;
                if (step >= p.steps || bad) {
                    if (lg == 0u) {
                        results[RESULT_STRIDE * cidx] = make_float4(bad ? -1e20f : com_x, head_y < neck_y ? 1.0f : 0.0f, low, (float)step);
#if CONTACT_MODEL
                        results[RESULT_STRIDE * cidx + 1u] = make_float4(rn_max, contact_sum, drift_max, ms_all);
#else
                        results[RESULT_STRIDE * cidx + 1u] = make_float4(rounds_sum, contact_sum, drift_max, ms_all);
#endif
                        results[RESULT_STRIDE * cidx + 2u] = make_float4(rr_max, ang_max, st, CONTACT_MODEL ? pen_max : rounds_sum);
                    }
                    live = false;
                }
            }
        }
    }
}
