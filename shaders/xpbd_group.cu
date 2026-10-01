// Position-based physics (XPBD, Macklin et al. 2016; small steps with one
// iteration, Macklin et al. 2019; positional friction, Mueller et al. 2020)
// on a group of W lanes per creature (src/warp_kernel.rs packs the creatures
// and writes the #defines and shaders/creature_common.cu in front of this
// text).
//
// A warp holds 32 / W groups of W lanes (W = 8, 16 or 32). Lane i of a group
// owns node i of its creature and the bone that ends there, so lane 0 is the
// head and lane 1 the neck bone. Lanes are numbered breadth first over the
// bone tree, so a tree level is a run of neighbouring lanes. Nodes are
// particles with the masses physics2::Model gives them; a bone is a rod
// between its pivot and its node; a joint keeps the angle between its bone
// and its parent bone inside its range, a hard stop or, with a ligament, a
// compliant one. Muscles pull two nodes together with the lean muscle model
// (cap x strength x trapezoid activation x stamina x Hill), W at a time.
//
// Each 1/RATE step is PSUB substeps: external, drag and muscle forces change
// the velocities, the nodes move, the rods and then the joint limits are
// projected one tree level at a time (rods that share a pivot move it
// together), the ground pushes nodes out and its friction takes back their
// motion along it, and the velocities follow from the moves. A lane hands a
// correction to another lane's node through a fixed-point sum in shared
// memory, so the result does not depend on the order of the additions.
// Friction only removes motion along the ground, so it never does positive
// work, and a node it holds stays where it is.
//
// A group runs one creature from its start to its end (a fall, the screen or
// the last step), then takes the next creature of the wave from an atomic
// counter (see the take-up buckets below).

#define PSUB 4
#define PH (1.0f / (RATE * PSUB))
#define INV_PH (RATE * PSUB)
// Fixed-point scales of the shared sums: velocity changes and moves.
#define VEL_SCALE 1048576.0f
#define POS_SCALE 16777216.0f
// A node faster than this (m/s) means the solver blew up.
#define BLOWUP_SPEED 100.0f

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
    // The force of the substep of each lane's muscles, and each group's
    // behavior totals: touched once per round or per step, so they wait in
    // shared memory instead of registers.
    __shared__ float s_mag[RMAX][BLOCK];
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
    // Corrections one lane hands to another lane's node, in fixed point.
    __shared__ int s_acc[3][BLOCK];
    s_acc[0][threadIdx.x] = 0;
    s_acc[1][threadIdx.x] = 0;
    s_acc[2][threadIdx.x] = 0;
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
    float4* const mf = &scatter[threadIdx.x >> 5][gbase * 2u];
    float4* const region = scatter[threadIdx.x >> 5];
    const unsigned below = (1u << lg) - 1u;

    // The group's creature (the same in every lane of the group).
    bool live = false;
    bool exhausted = false;
    bool fresh = false;
    bool limp = false;
    unsigned cidx = 0u, nn = 0u, depth = 0u, rounds = 0u, ew = 0u, nmus = 0u, mbase = 0u, ebase = 0u;
    unsigned step = 0u;
    float amp = 0.0f, qphase = 0.0f, inv_mass = 0.0f, inv_nodes = 0.0f;
    // The creature's stamina store (the same in every lane of the group), the
    // inverse of its capacity in joules, the time of this lane's node's last
    // touchdown (-1: none yet), and the square of the rate of this lane's
    // joint ligament (0: an inelastic stop) with the spring's stiffness it
    // has in the substep (the joint's inertia times that).
    float stam = 1.0f, inv_cap = 0.0f, tdn = -1.0f, lig2 = 0.0f, lig_k = 0.0f;
    // This lane's node and bone.
    float m = 0.0f, rad = 0.0f, fric = 0.0f, len = 0.0f, lo = 0.0f, hi = 0.0f, prad = 0.0f, hm = 0.0f;
    unsigned topo = 0u, mnode = 0u;
    // State: the joint angle and rate of the lane's bone (the neck's absolute
    // angle on lane 1); the head's position and velocity are lane 0's node
    // position and velocity.
    float q = 0.0f, qd = 0.0f;
    // The joint angle measured from the node positions.
    float qa = 0.0f;
#if AIR
    // The air's retention per substep (p.air_sub is per lane-group substep).
    const float air_ps = powf(p.air_sub, (float)SUBSTEPS / (float)PSUB);
#endif
#pragma unroll
    for (int r = 0; r < RMAX; r++) { s_mag[r][tid] = 0.0f; }
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

    // Height and slope of the ground under x.
    auto terrain = [&](float x) -> float2 { return ground_at(x, p, amp, qphase); };

    // This warp's bucket: the last one whose first warp is at or before it.
    unsigned bucket = 0u;
    {
        const unsigned order = (threadIdx.x >> 5) * gridDim.x + blockIdx.x;
#pragma unroll
        for (unsigned b = 1u; b < RMAX; b++) { if (tk.warp[b] <= order) { bucket = b; } }
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
                for (unsigned k = 0u; k < RMAX; k++) {
                    const unsigned b = (bucket + k) % RMAX;
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
                nmus = h0.y & 255u;
                mbase = h0.w;
                ebase = h1.x;
                inv_mass = __uint_as_float(h1.z);
                inv_cap = __uint_as_float(h1.w) * p.inv_muscle_energy;
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
                lig2 = lg >= 2u ? s7 : 0.0f;
                if (lg == 0u) {
                    px = s6; py = s7; vx = 0.0f; vy = 0.0f; q = 0.0f;
                } else {
                    q = s6;
                }
                qd = 0.0f;
                th = 0.0f; om = 0.0f;
                stam = 1.0f; tdn = -1.0f; lig_k = 0.0f;
#pragma unroll
                for (int r = 0; r < RMAX; r++) { s_mag[r][tid] = 0.0f; }
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
                m = 0.0f; rad = 0.0f; fric = 0.0f; len = 0.0f; hm = 0.0f; topo = 0u; lig2 = 0.0f;
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
            for (int r = 0; r < RMAX; r++) {
                const unsigned mi = (unsigned)r * W + lg;
                if (write && (unsigned)r < rounds && mi < nmus) {
                    frames[fb + W + mi] = make_float2(stam, s_mag[r][tid]);
                }
            }
            const bool broken = body && lg >= 2u && (qa < lo - JOINT_BREAK || qa > hi + JOINT_BREAK);
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
        // A new creature: its start pose from the joint angles, parents first.
        if (__any_sync(FULL, fresh)) {
            kinematics(fresh);
#if RECORD
            if (__any_sync(FULL, fresh)) {
                if (fresh) { qa = q; }
                for (unsigned t = 0u; t <= SETTLE; t++) { record_frame(t, fresh); }
            }
#endif
            fresh = false;
        }
        const float head_vx0 = shf(vx, 0u), head_vy0 = shf(vy, 0u);
        float step_n = 0.0f, step_t = 0.0f;
        const float imass = valid ? 1.0f / m : 0.0f;
        const float ipm = shf(imass, piv);
        // The pivot of the parent bone, for the joint limits.
        const unsigned gp = shu(piv, pb);
        int* const acc_x = s_acc[0] + gtid;
        int* const acc_y = s_acc[1] + gtid;
        int* const acc_n = s_acc[2] + gtid;
        // Adds what the other lanes left for this lane's node, in fixed point
        // so the sum is the same in any order.
        // Constraint moves that several lanes hand one node in the same pass
        // are averaged (Macklin et al. 2014), so siblings pulling one pivot
        // together do not overshoot; forces simply add.
        auto take = [&](float& x, float& y, float inv_scale) {
            __syncwarp();
            const float share = inv_scale / (float)max(acc_n[lg], 1);
            x += (float)acc_x[lg] * share;
            y += (float)acc_y[lg] * share;
            acc_x[lg] = 0;
            acc_y[lg] = 0;
            acc_n[lg] = 0;
            __syncwarp();
        };
        auto give = [&](unsigned node, float x, float y, float scale) {
            atomicAdd(acc_x + node, __float2int_rn(clampf(x * scale, -2.0e9f, 2.0e9f)));
            atomicAdd(acc_y + node, __float2int_rn(clampf(y * scale, -2.0e9f, 2.0e9f)));
        };
        auto give_move = [&](unsigned node, float x, float y) {
            if (x != 0.0f || y != 0.0f) {
                give(node, x, y, POS_SCALE);
                atomicAdd(acc_n + node, 1);
            }
        };
        float sx = px, sy = py;
        for (unsigned sub = 0u; sub < PSUB; sub++) {
            const float ts = t_now + (float)sub * PH;
            PROF(1);
            // External forces: gravity, wind, mud drag and buoyancy.
            if (valid) {
                float ax = 0.0f, ay = -p.gravity;
#if WIND
                ax += p.wind;
#endif
#if MUD
                {
                    const float2 g = terrain(px);
                    const float dry = (py - g.x) / sqrtf(1.0f + g.y * g.y) - rad;
                    ax -= MUD_DRAG * (clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH)) * vx;
                }
#endif
#if WATER
                ay += WATER_BUOYANCY * p.gravity * clampf((p.water - (py - rad)) / (2.0f * rad), 0.0f, 1.0f);
#endif
                vx += ax * PH;
                vy += ay * PH;
            }
            // Air and water drag on each bone at its midpoint, half to each
            // end, limited so a substep never more than halves the speed.
            {
                const float bx = shf(px, piv), by = shf(py, piv), bvx = shf(vx, piv), bvy = shf(vy, piv);
                if (body) {
                    const float wx = 0.5f * (bvx + vx), wy = 0.5f * (bvy + vy);
                    const float speed = sqrtf(wx * wx + wy * wy);
                    const float width = prad + rad;
                    const float lim = 0.5f * fminf(m, 1.0f / ipm) * INV_PH;
                    const float air = fmaxf(fminf(AIR_DRAG * len * width * speed, lim), 0.0f);
                    float fx = -wx * air, fy = -wy * air;
#if WATER
                    {
                        const float sub_i = clampf((p.water - (py - rad)) / (2.0f * rad), 0.0f, 1.0f);
                        const float sub_p = clampf((p.water - (by - prad)) / (2.0f * prad), 0.0f, 1.0f);
                        const float wet = 0.5f * (sub_p + sub_i);
                        const float inverse = 1.0f / len;
                        const float ax_ = (px - bx) * inverse, ay_ = (py - by) * inverse;
                        const float along = wx * ax_ + wy * ay_;
                        const float lx = ax_ * along, ly = ay_ * along;
                        const float ws = fmaxf(fminf(WATER_DRAG * wet * len * width * speed, lim), 0.0f);
                        const float weak = ws * WATER_ALONG;
                        fx -= (wx - lx) * ws + lx * weak;
                        fy -= (wy - ly) * ws + ly * weak;
                    }
#endif
                    vx += 0.5f * fx * PH * imass;
                    vy += 0.5f * fy * PH * imass;
                    give(piv, 0.5f * fx * PH * ipm, 0.5f * fy * PH * ipm, VEL_SCALE);
                }
                (void)bx; (void)by;
            }
            PROF(4);
            // Muscles, W at a time: a muscle has no state; its activation is a
            // trapezoid of the time (since the last touchdown of its sensor),
            // and the creature's stamina scales every muscle's force. The work
            // of all the muscles drains the stamina.
            float pw = 0.0f;
            for (unsigned r = 0u; r < maxrounds; r++) {
                const unsigned mi = r * W + lg;
                const bool mon = r < rounds && mi < nmus;
                float4 f0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), f1 = f0;
                if (mon) {
                    const float4* mrec = reinterpret_cast<const float4*>(muscles + mbase) + mi * 2u;
                    f0 = mrec[0];
                    f1 = mrec[1];
                }
                const unsigned packed = __float_as_uint(f0.x);
                const unsigned la = mon ? packed & 31u : lg, lb = mon ? (packed >> 5u) & 31u : lg;
                const unsigned ls = mon ? (packed >> 10u) & 31u : lg;
                const float ax_ = shf(px, la), ay_ = shf(py, la), avx = shf(vx, la), avy = shf(vy, la), wa = shf(imass, la);
                const float bx_ = shf(px, lb), by_ = shf(py, lb), bvx = shf(vx, lb), bvy = shf(vy, lb), wb = shf(imass, lb);
                const float td = shf(tdn, ls);
                float magnitude = 0.0f;
                if (mon) {
                    const float cap = f0.y, hill = f0.z, inv_period = f0.w;
                    const float phase = f1.x, half = f1.y, inv_ramp = f1.z, reset = f1.w;
                    const float dx = bx_ - ax_, dy = by_ - ay_;
                    const float length_m = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f);
                    const float inverse = 1.0f / length_m;
                    const float dirx = dx * inverse, diry = dy * inverse;
                    const float relative = (bvx - avx) * dirx + (bvy - avy) * diry;
                    const bool sensed = ((packed >> 15u) & 1u) != 0u && td >= 0.0f;
                    const float x = sensed ? (ts - td) * inv_period + reset : ts * inv_period + phase;
                    const float ph = x - floorf(x);
                    const float act = limp ? 0.0f : clampf((half - fabsf(ph - half)) * inv_ramp, 0.0f, 1.0f);
                    const float drive = cap * act * stam * clampf(1.0f + relative * hill, 0.0f, 1.0f);
                    magnitude = clampf(drive + relative * MUSCLE_DAMPER, -cap, cap);
                    pw += drive * fmaxf(-relative, 0.0f);
                    // Implicit in Hill's law: the impulse never drives the
                    // ends together faster than the muscle's top shortening
                    // speed 1 / hill within the substep.
                    const float imp = fminf(magnitude * PH, fmaxf(relative + 1.0f / hill, 0.0f) / (wa + wb));
                    give(la, dirx * imp * wa, diry * imp * wa, VEL_SCALE);
                    give(lb, -dirx * imp * wb, -diry * imp * wb, VEL_SCALE);
                }
                s_mag[r][tid] = magnitude;
            }
            stam = clampf(stam - gsum(pw) * PH * inv_cap
                + MUSCLE_RECOVERY * p.muscle_recovery * PH * (1.0f - stam), 0.0f, 1.0f);
            take(vx, vy, 1.0f / VEL_SCALE);
            PROF(5);
            // Move.
            sx = px; sy = py;
            if (valid) { px += vx * PH; py += vy * PH; }
            // Rods, one tree level at a time: a rod moves its node and its
            // pivot; rods that share a pivot move it together.
            for (unsigned L = 1u; L <= maxlev; L++) {
                const float bx = shf(px, piv), by = shf(py, piv);
                if (body && lvl == L) {
                    const float dx = px - bx, dy = py - by;
                    const float d = fmaxf(sqrtf(dx * dx + dy * dy), 1e-9f);
                    const float c = (d - len) / (d * (ipm + imass));
                    px -= imass * c * dx; py -= imass * c * dy;
                    give_move(piv, ipm * c * dx, ipm * c * dy);
                }
                take(px, py, 1.0f / POS_SCALE);
            }
            PROF(6);
            // Joint limits, one tree level at a time: the angle between a
            // bone and its parent bone stays in the joint's range, a ligament
            // making the stop compliant.
            for (unsigned L = 2u; L <= maxlev; L++) {
                const float bx = shf(px, piv), by = shf(py, piv);
                const float cx = shf(px, pb), cy = shf(py, pb);
                const float dxp = shf(px, gp), dyp = shf(py, gp);
                const float wc = shf(imass, pb), wd = shf(imass, gp);
                if (body && lg >= 2u && lvl == L) {
                    const float vx_ = px - bx, vy_ = py - by;
                    const float ux = cx - dxp, uy = cy - dyp;
                    // Ranges may reach past half a turn, so the angle is
                    // taken within half a turn of the range's middle.
                    const float mid = 0.5f * (lo + hi);
                    const float rel = atan2f(ux * vy_ - uy * vx_, ux * vx_ + uy * vy_) - mid;
                    const float qn = mid + rel - TAU_F * rintf(rel * (1.0f / TAU_F));
                    const float cval = qn > hi ? qn - hi : (qn < lo ? qn - lo : 0.0f);
                    const float iv = 1.0f / fmaxf(vx_ * vx_ + vy_ * vy_, 1e-12f);
                    const float iu = 1.0f / fmaxf(ux * ux + uy * uy, 1e-12f);
                    // d angle / d end of each bone; the child's pivot is the
                    // parent's end or, for a bone at the head, the parent's
                    // pivot.
                    const float gax = -vy_ * iv, gay = vx_ * iv;
                    const float gcx = uy * iu, gcy = -ux * iu;
                    const bool bc = piv == pb, bd = !bc && piv == gp;
                    const float gbx = -gax + (bc ? gcx : 0.0f) - (bd ? gcx : 0.0f);
                    const float gby = -gay + (bc ? gcy : 0.0f) - (bd ? gcy : 0.0f);
                    const float gcx2 = bc ? 0.0f : gcx, gcy2 = bc ? 0.0f : gcy;
                    const float gdx = bd ? 0.0f : -gcx, gdy = bd ? 0.0f : -gcy;
                    // A ligament's stop is a spring of stiffness lig2 times
                    // the joint's inertia.
                    const float alpha = lig2 > 0.0f ? INV_PH * INV_PH / (lig2 * m * len * len) : 0.0f;
                    const float den = imass * (gax * gax + gay * gay) + ipm * (gbx * gbx + gby * gby)
                        + wc * (gcx2 * gcx2 + gcy2 * gcy2) + wd * (gdx * gdx + gdy * gdy) + alpha;
                    const float lam = -cval / den;
                    px += imass * gax * lam; py += imass * gay * lam;
                    give_move(piv, ipm * gbx * lam, ipm * gby * lam);
                    give_move(pb, wc * gcx2 * lam, wc * gcy2 * lam);
                    give_move(gp, wd * gdx * lam, wd * gdy * lam);
                }
                take(px, py, 1.0f / POS_SCALE);
            }
            PROF(7);
            // Ground contacts: a node below the ground goes back out along
            // the normal; friction takes back its motion along the ground,
            // all of it while the normal push can hold it.
#if GROUND
            if (valid) {
                const float2 g = terrain(px);
                const float secant = sqrtf(1.0f + g.y * g.y);
                const float nx = -g.y / secant, ny = 1.0f / secant;
                const float dry = (py - g.x) / secant - rad;
#if MUD
                const float gap = dry + p.mud;
                const float sink = clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH);
                float mu = fric * p.friction * (1.0f + MUD_GRIP * sink) * (1.0f + MUD_NORMAL * sink);
#else
                const float gap = dry;
                float mu = fric * p.friction;
#endif
#if ICE
                mu *= 1.0f - p.patches * ice_at(px);
#endif
                const float push = fmaxf(-gap, 0.0f);
                px += push * nx;
                py += push * ny;
                const float tx = ny, ty = -nx;
                const float slide = (px - sx) * tx + (py - sy) * ty;
                const float back = clampf(slide, -mu * push, mu * push);
                px -= back * tx;
                py -= back * ty;
                step_n += push * m * INV_PH * INV_PH;
                step_t -= back * m * INV_PH * INV_PH;
            }
#endif
            PROF(8);
            // Velocities from the moves, then joint damping and the spin cap
            // on each bone's turn rate against its parent.
            if (valid) {
                vx = (px - sx) * INV_PH;
                vy = (py - sy) * INV_PH;
            }
            {
                const float bx = shf(px, piv), by = shf(py, piv), bvx = shf(vx, piv), bvy = shf(vy, piv);
                const float dx = px - bx, dy = py - by;
                const float il = 1.0f / fmaxf(dx * dx + dy * dy, 1e-12f);
                const float w = body ? (dx * (vy - bvy) - dy * (vx - bvx)) * il : 0.0f;
                const float wp = shf(w, pb);
                if (body) {
                    float target = lg >= 2u ? wp + (w - wp) * __expf(-INV_JOINT_DAMPING * PH) : w;
                    target = clampf(target, -SPIN_CAP, SPIN_CAP);
                    // Turn the bone's end about the pair's centre of mass.
                    const float dw = target - w;
                    const float sjx = -dy * dw, sjy = dx * dw;
                    const float fj = imass / (ipm + imass), fa = ipm / (ipm + imass);
                    vx += sjx * fj; vy += sjy * fj;
                    give(piv, -sjx * fa, -sjy * fa, VEL_SCALE);
                }
                take(vx, vy, 1.0f / VEL_SCALE);
            }
#if AIR
            if (valid) { vx *= air_ps; vy *= air_ps; }
#endif
            PROF(9);
        }
        // The joint angle of this lane's bone, for the break test and a
        // recording.
        {
            const float bx = shf(px, piv), by = shf(py, piv);
            const float cx = shf(px, pb), cy = shf(py, pb);
            const float dxp = shf(px, gp), dyp = shf(py, gp);
            if (body && lg >= 2u) {
                const float vx_ = px - bx, vy_ = py - by, ux = cx - dxp, uy = cy - dyp;
                const float mid = 0.5f * (lo + hi);
                const float rel = atan2f(ux * vy_ - uy * vx_, ux * vx_ + uy * vy_) - mid;
                qa = mid + rel - TAU_F * rintf(rel * (1.0f / TAU_F));
            }
        }
        rec_n = step_n * (1.0f / PSUB);
        rec_t = step_t * (1.0f / PSUB);

        PROF(12);
        // Metrics, falls and the screen, once per step.
        {
            // A blow-up of the solver fails the trial like a non-finite
            // position does.
            const bool bad = valid && (!(fabsf(px) <= 1e6f && fabsf(py) <= 1e6f)
                || !(vx * vx + vy * vy <= BLOWUP_SPEED * BLOWUP_SPEED));
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
            // A touchdown restarts the rhythm of the muscles that sense that
            // node, from the next step.
            if (((down >> lg) & 1u) != 0u && step > 0u) {
                tdn = t_now + DT;
            }
            {
                const float dvx = shf(vx, 0u) - head_vx0, dvy = shf(vy, 0u) - head_vy0;
                if (t_now >= HEAD_SHAKE_WINDOW) {
                    const float accel = sqrtf(dvx * dvx + dvy * dvy) * RATE;
                    head_shake += (accel - head_shake) * fminf(1.0f / (HEAD_SHAKE_WINDOW * RATE), 1.0f);
                }
            }
            const bool broken_lane = body && lg >= 2u && (qa < lo - JOINT_BREAK || qa > hi + JOINT_BREAK);
            const bool broken = ((__ballot_sync(FULL, broken_lane) >> gshift) & GM) != 0u;
            const float head_y = shf(py, 0u), neck_y = shf(py, 1u);
            // The rung trace (creature_kernel::RungTrace): the distance at 1,
            // 2.5, 5 and 10 s and the early features at 1 and 2.5 s, as fp16
            // pairs in the seven result words the host reads for nothing
            // else. It records; it stops no trial. During the trial the
            // speeds and the contact and energy pairs wait in `s_rung` (.y,
            // .z, .w; .x is the distance half a second before a rung) and the
            // head shake pair in `ground_hi`.
            __shared__ uint4 s_rung[BLOCK / W];
            uint4& rung = s_rung[tid / W];
            // The early rungs' record of the trial, in the end code's bit
            // positions: the rung that stopped it (bits 6 and 7) and the
            // cadence band at 1 and 2.5 s (bits 8 to 13).
            __shared__ unsigned s_rb[BLOCK / W];
            unsigned& rbits = s_rb[tid / W];
            // The creature's rhythm period (a rung feature) in the low half
            // and its flags in the high half, read from its second head word
            // when a rule needs them (an audit creature runs every rule off,
            // an exempt one, a nursery's or an immigrant's, skips the early
            // rungs).
            auto head_flags = [&]() -> unsigned {
                const unsigned y = heads[2u * cidx].y;
                return (y >> 16u) | ((y >> 8u) & 255u) << 16u;
            };
            auto h16 = [](float v) -> unsigned {
                unsigned short r;
                asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(v));
                return (unsigned)r;
            };
            const unsigned rung1 = (unsigned)(RATE) - 1u, rung2 = (unsigned)(2.5f * RATE) - 1u;
            const unsigned rung3 = (unsigned)(5.0f * RATE) - 1u, rung4 = (unsigned)(10.0f * RATE) - 1u;
            const unsigned half_s = (unsigned)(0.5f * RATE);
            const bool early = step == rung1 || step == rung2;
            // The creature's stamina stands for the mean muscle energy.
            const float en_mean = stam;
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
                    if (com_x < p.screen_bar && ((head_flags() >> 16u) & 1u) == 0u) {
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
