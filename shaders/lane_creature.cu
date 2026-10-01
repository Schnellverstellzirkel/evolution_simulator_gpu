// The physics of shaders/warp_creature.cu with one creature per thread, for
// bodies of at most W = 8 nodes (src/warp_kernel.rs packs them as the 8-lane
// class and writes the #defines and shaders/creature_common.cu in front of
// this text).
//
// A thread holds its creature's nodes in arrays indexed by the lane the
// 8-lane packing gives them: breadth first over the bone tree, so a parent
// always comes before its children. The tree passes are loops in that order
// (parents first) or in reverse (children first) where the lane-group kernel
// steps level by level with shuffles, and the muscles add their forces
// straight to the bodies that carry their ends. A thread runs one creature
// from its start to its end, then takes the next one of the wave.

#define N W

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
    // This thread's bucket: the last one whose first warp is at or before
    // its warp.
    unsigned bucket = 0u;
    {
        const unsigned order = (threadIdx.x >> 5) * gridDim.x + blockIdx.x;
#pragma unroll
        for (unsigned b = 1u; b < RMAX; b++) { if (tk.warp[b] <= order) { bucket = b; } }
    }
    // The creature's nodes and bones, by lane.
    float m[N], rad[N], fric[N], len[N], lo[N], hi[N], prad[N], lig2[N];
    unsigned topo[N], anc[N], mnode[N];
    // State: joint angle and rate (the neck's absolute angle on lane 1), the
    // head's position and velocity on lane 0, and the kinematics.
    float q[N], qd[N], th[N], om[N], px[N], py[N], vx[N], vy[N], ppx[N], ppy[N], pvx[N], pvy[N];
    float tdn[N], lig_k[N], rec_n[N], rec_t[N];
    // The force of each muscle in the last substep.
    float mag[RMAX * N];
    for (;;) {
        unsigned got = 0xffffffffu;
        for (unsigned k = 0u; k < RMAX; k++) {
            const unsigned b = (bucket + k) % RMAX;
            const unsigned size = tk.end[b] - tk.start[b];
            if (size == 0u || *(volatile unsigned*)&counter[b] >= size) { continue; }
            const unsigned i = atomicAdd(&counter[b], 1u);
            if (i < size) { got = tk.start[b] + i; bucket = b; break; }
        }
        if (got >= p.count) { break; }
        const unsigned cidx = p.base + got;
        const uint4 h0 = heads[2u * cidx];
        const uint4 h1 = heads[2u * cidx + 1u];
        const unsigned nn = h0.x & 255u;
        const unsigned nmus = h0.y & 255u;
        const float* const mrec = muscles + h0.w;
        const float inv_mass = __uint_as_float(h1.z);
        const float inv_cap = __uint_as_float(h1.w) * p.inv_muscle_energy;
        const float inv_nodes = 1.0f / (float)nn;
        // The rhythm period (a rung feature) in the low half and the flags in
        // the high half.
        const unsigned head_flags = (h0.y >> 16u) | ((h0.y >> 8u) & 255u) << 16u;
#if QUAKE
        const float qphase = (float)(h0.z & 0xffffu) * (1.0f / 65536.0f);
        const float amp = p.terrain + p.quake * (0.6f + (float)((h0.z >> 16u) & 0xffffu) * (0.8f / 65536.0f));
#else
        const float qphase = 0.0f;
        const float amp = p.terrain;
#endif
        auto terrain = [&](float x) -> float2 { return ground_at(x, p, amp, qphase); };
        float hm = 0.0f;
        const unsigned* rec = lanes + cidx * (LF * W);
#pragma unroll
        for (unsigned j = 0u; j < N; j++) {
            m[j] = __uint_as_float(rec[0u * W + j]);
            rad[j] = __uint_as_float(rec[1u * W + j]);
            fric[j] = __uint_as_float(rec[2u * W + j]);
            len[j] = __uint_as_float(rec[3u * W + j]);
            lo[j] = __uint_as_float(rec[4u * W + j]);
            hi[j] = __uint_as_float(rec[5u * W + j]);
            const float s6 = __uint_as_float(rec[6u * W + j]);
            const float s7 = __uint_as_float(rec[7u * W + j]);
            prad[j] = __uint_as_float(rec[8u * W + j]);
            topo[j] = rec[9u * W + j];
            anc[j] = rec[10u * W + j];
            mnode[j] = rec[11u * W + j];
            if (j == 1u) { hm = s7; }
            lig2[j] = j >= 2u ? s7 : 0.0f;
            q[j] = j == 0u ? 0.0f : s6;
            px[j] = j == 0u ? s6 : 0.0f;
            py[j] = j == 0u ? s7 : 0.0f;
            qd[j] = 0.0f; th[j] = 0.0f; om[j] = 0.0f; vx[j] = 0.0f; vy[j] = 0.0f;
            ppx[j] = 0.0f; ppy[j] = 0.0f; pvx[j] = 0.0f; pvy[j] = 0.0f;
            tdn[j] = -1.0f; lig_k[j] = 0.0f; rec_n[j] = 0.0f; rec_t[j] = 0.0f;
        }
        for (unsigned k = 0u; k < RMAX * N; k++) { mag[k] = 0.0f; }
        float stam = 1.0f;
        bool limp = false;
        Result mt;
        mt.fitness = 0.0f; mt.ground_contact = 0.0f; mt.vertical_oscillation = 1e20f; mt.gait_frequency = -1e20f;
        mt.previous_center_y = 0.0f; mt.vertical_extremum = 0.0f; mt.vertical_trend = 0.0f; mt.gait_turns = 0.0f;
        mt.height_sum = 0.0f; mt.contact_lo = 0.0f; mt.contact_hi = 0.0f; mt.lift_lo = 0.0f; mt.lift_hi = 0.0f;
        mt.ground_lo = 0.0f; mt.ground_hi = 0.0f; mt.fall_time = 0.0f; mt.head_shake = 0.0f; mt.screen_x = 0.0f;
        mt.screened = 0.0f;
        unsigned contact_bits = 0u, lift_bits = 0u, ground_bits = 0u;
        float head_shake = 0.0f;
        uint4 rung = make_uint4(0u, 0u, 0u, 0u);
        unsigned rbits = 0u;
#if RECORD
        Result kept = mt;
        bool done_scoring = false;
#endif
        auto pivot = [&](unsigned j) { return topo[j] & 31u; };
        auto parent = [&](unsigned j) { return (topo[j] >> 5u) & 31u; };
        auto level = [&](unsigned j) { return (topo[j] >> 10u) & 31u; };
        // Node positions and velocities from the state, parents first.
        auto kinematics = [&]() {
            for (unsigned j = 1u; j < nn; j++) {
                th[j] = j == 1u ? q[j] : th[parent(j)] + q[j];
                om[j] = j == 1u ? qd[j] : om[parent(j)] + qd[j];
                float sn, cs;
                __sincosf(th[j] - TAU_F * rintf(th[j] * (1.0f / TAU_F)), &sn, &cs);
                const unsigned v = pivot(j);
                ppx[j] = px[v]; ppy[j] = py[v]; pvx[j] = vx[v]; pvy[j] = vy[v];
                px[j] = ppx[j] + len[j] * cs;
                py[j] = ppy[j] + len[j] * sn;
                vx[j] = pvx[j] - len[j] * om[j] * sn;
                vy[j] = pvy[j] + len[j] * om[j] * cs;
            }
        };
        auto broken_at = [&](unsigned j) { return j >= 2u && (q[j] < lo[j] - JOINT_BREAK || q[j] > hi[j] + JOINT_BREAK); };
#if RECORD
        auto record_frame = [&](unsigned t) {
            const unsigned fb = t * p.stride;
            for (unsigned j = 0u; j < nn; j++) {
                frames[fb + mnode[j]] = make_float2(px[j], py[j]);
                frames[fb + W + nmus + mnode[j]] = make_float2(rec_n[j], rec_t[j]);
            }
            for (unsigned k = 0u; k < nmus; k++) { frames[fb + W + k] = make_float2(stam, mag[k]); }
            // Broken joints by the creature's bone number (bone j ends at node j + 1).
            unsigned lo_bits = 0u, hi_bits = 0u;
            for (unsigned j = 2u; j < nn; j++) {
                if (broken_at(j)) {
                    const unsigned b = mnode[j] - 1u;
                    if (b < 32u) { lo_bits |= 1u << b; } else { hi_bits |= 1u << (b - 32u); }
                }
            }
            frames[fb + p.stride - 1u] = make_float2(__uint_as_float(lo_bits), __uint_as_float(hi_bits));
        };
        kinematics();
        for (unsigned t = 0u; t <= SETTLE; t++) { record_frame(t); }
#else
        kinematics();
#endif
        for (unsigned step = 0u;; step++) {
            const float t_now = (float)step * DT;
            const float head_vx0 = vx[0], head_vy0 = vy[0];
            float step_n[N], step_t[N];
            for (unsigned j = 0u; j < N; j++) { step_n[j] = 0.0f; step_t[j] = 0.0f; }
            // Carried from a substep's body to the balance at the top of the next.
            float mx = 0.0f, my = 0.0f, ledger = 0.0f, scale = 0.0f;
            float y_start[N], buoy[N];
            unsigned nc = 0u;
            for (unsigned sub = 0u;; sub++) {
                if (sub > 0u) {
                    kinematics();
                    // Momentum balance: the body's momentum is its old momentum
                    // plus the external impulses; the rest of first-order
                    // integration's error goes as one uniform velocity.
                    float ax_ = 0.0f, ay_ = 0.0f;
                    for (unsigned j = 0u; j < nn; j++) { ax_ += m[j] * vx[j]; ay_ += m[j] * vy[j]; }
#if AIR
                    const float wantx = mx * p.air_sub, wanty = my * p.air_sub;
#else
                    const float wantx = mx, wanty = my;
#endif
                    const float sx = (wantx - ax_) * inv_mass, sy = (wanty - ay_) * inv_mass;
                    for (unsigned j = 0u; j < nn; j++) { vx[j] += sx; vy[j] += sy; pvx[j] += sx; pvy[j] += sy; }
                    // First law in flight: a substep without ground contact
                    // gains no more energy than the muscles, the wind, the
                    // buoyancy and the ligaments put in.
                    if (nc == 0u) {
                        for (unsigned k = 0u; k < nmus; k++) {
                            const unsigned packed = __float_as_uint(mrec[k * 8u]);
                            const unsigned la = packed & 31u, lb = (packed >> 5u) & 31u;
                            const float dx = px[lb] - px[la], dy = py[lb] - py[la];
                            ledger += mag[k] * sqrtf(dx * dx + dy * dy);
                        }
                        float internal = 0.0f;
                        const float cx = wantx * inv_mass, cy = wanty * inv_mass;
                        for (unsigned j = 0u; j < nn; j++) {
                            if (lig_k[j] > 0.0f) {
                                // The ligament's store at the end of the substep.
                                const float pen = fmaxf(q[j] - hi[j], 0.0f) + fminf(q[j] - lo[j], 0.0f);
                                const float store = 0.5f * lig_k[j] * pen * pen;
                                ledger += store;
                                scale += store;
                            }
                            ledger += 0.5f * m[j] * (vx[j] * vx[j] + vy[j] * vy[j]) + m[j] * p.gravity * py[j];
#if WIND
                            ledger -= p.wind * m[j] * px[j];
#endif
#if WATER
                            ledger -= buoy[j] * (py[j] - y_start[j]);
#endif
                            const float dvx = vx[j] - cx, dvy = vy[j] - cy;
                            internal += 0.5f * m[j] * (dvx * dvx + dvy * dvy);
                        }
                        const float excess = ledger - (1e-4f + 1e-5f * scale);
                        if (excess > 0.0f) {
                            const float keep = internal > 0.0f ? sqrtf(fmaxf(1.0f - excess / internal, 0.0f)) : 0.0f;
                            for (unsigned j = 0u; j < nn; j++) {
                                qd[j] *= keep;
                                om[j] *= keep;
                                vx[j] = cx + keep * (vx[j] - cx);
                                vy[j] = cy + keep * (vy[j] - cy);
                                pvx[j] = cx + keep * (pvx[j] - cx);
                                pvy[j] = cy + keep * (pvy[j] - cy);
                            }
                        }
                    }
                }
                if (sub == SUBSTEPS) {
                    break;
                }
                const float ts = t_now + (float)sub * HS;
                const float ox = px[0], oy = py[0];
                // Momentum before the substep plus the external impulses, and
                // the first-law ledger.
                mx = 0.0f; my = 0.0f; ledger = 0.0f; scale = 0.0f;
                vec3 i0[N], i1[N], bs[N], axis[N];
                for (unsigned j = 0u; j < nn; j++) {
                    const float kinetic = 0.5f * m[j] * (vx[j] * vx[j] + vy[j] * vy[j]);
                    const float potential = m[j] * p.gravity * py[j];
                    mx += m[j] * vx[j];
                    my += m[j] * vy[j];
                    ledger -= kinetic + potential;
                    scale += kinetic + fabsf(potential);
#if WIND
                    ledger += p.wind * m[j] * px[j];
#endif
                    y_start[j] = py[j];
                    // Body inertia (the neck also carries the head), pivot arm
                    // and the velocity-product force.
                    const float armx = ppx[j] - ox, army = ppy[j] - oy;
                    axis[j] = v3(1.0f, army, -armx);
                    i0[j] = v3(0.0f, 0.0f, 0.0f); i1[j] = i0[j]; bs[j] = i0[j];
                    if (j >= 1u) {
                        const float rx = px[j] - ox, ry = py[j] - oy;
                        const float mj = m[j] + (j == 1u ? hm : 0.0f);
                        i0[j] = v3(m[j] * (rx * rx + ry * ry), -m[j] * ry, m[j] * rx);
                        i1[j] = v3(mj, 0.0f, mj);
                        const vec3 sv = v3(om[j], pvx[j] + om[j] * army, pvy[j] - om[j] * armx);
                        bs[j] = crf(sv, sym_mul(i0[j], i1[j], sv));
                    }
                }
                // Gravity, wind, mud drag and buoyancy on each node; the head's
                // force goes to the neck.
                for (unsigned j = 0u; j < nn; j++) {
                    float fx = 0.0f;
                    float fy = -p.gravity * m[j];
#if WIND
                    fx += p.wind * m[j];
#endif
#if MUD
                    {
                        const float2 g = terrain(px[j]);
                        const float dry = (py[j] - g.x) / sqrtf(1.0f + g.y * g.y) - rad[j];
                        fx -= m[j] * MUD_DRAG * (clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH)) * vx[j];
                    }
#endif
                    buoy[j] = 0.0f;
#if WATER
                    {
                        const float sub_i = clampf((p.water - (py[j] - rad[j])) / (2.0f * rad[j]), 0.0f, 1.0f);
                        buoy[j] = WATER_BUOYANCY * m[j] * p.gravity * sub_i;
                        fy += buoy[j];
                    }
#endif
                    mx += fx * HS;
                    my += fy * HS;
                    bs[j == 0u ? 1u : j] -= force_at(px[j] - ox, py[j] - oy, fx, fy);
                }
                for (unsigned j = 1u; j < nn; j++) {
                    // Air drag on the bone at its midpoint, limited so a
                    // substep of drag never more than halves the speed.
                    const float midx = 0.5f * (ppx[j] + px[j]), midy = 0.5f * (ppy[j] + py[j]);
                    const float wx = 0.5f * (pvx[j] + vx[j]), wy = 0.5f * (pvy[j] + vy[j]);
                    const float speed = sqrtf(wx * wx + wy * wy);
                    const float width = prad[j] + rad[j];
                    const float strength = fmaxf(fminf(AIR_DRAG * len[j] * width * speed, 0.5f * m[j] * INV_HS), 0.0f);
                    bs[j] -= force_at(midx - ox, midy - oy, -wx * strength, -wy * strength);
                    mx -= wx * strength * HS;
                    my -= wy * strength * HS;
#if WATER
                    {
                        const float sub_i = clampf((p.water - (py[j] - rad[j])) / (2.0f * rad[j]), 0.0f, 1.0f);
                        const float sub_p = clampf((p.water - (ppy[j] - prad[j])) / (2.0f * prad[j]), 0.0f, 1.0f);
                        const float wet = 0.5f * (sub_p + sub_i);
                        const float inverse = 1.0f / len[j];
                        const float ax_ = (px[j] - ppx[j]) * inverse, ay_ = (py[j] - ppy[j]) * inverse;
                        const float along = wx * ax_ + wy * ay_;
                        const float lx = ax_ * along, ly = ay_ * along;
                        const float sx = wx - lx, sy = wy - ly;
                        const float ws = fmaxf(fminf(WATER_DRAG * wet * len[j] * width * speed, 0.5f * m[j] * INV_HS), 0.0f);
                        const float weak = ws * WATER_ALONG;
                        const float fx = -(sx * ws + lx * weak), fy = -(sy * ws + ly * weak);
                        bs[j] -= force_at(midx - ox, midy - oy, fx, fy);
                        mx += fx * HS;
                        my += fy * HS;
                    }
#endif
                    // Spin cap: rotational drag past the cap, implicit, toward rest.
                    if (fabsf(om[j]) > SPIN_CAP) {
                        const float drag = SPIN_HARDNESS * m[j] * len[j] * len[j] * (fabsf(om[j]) * INV_SPIN_CAP - 1.0f);
                        i0[j].x += drag;
                        bs[j].x += drag * INV_HS * om[j];
                    }
                }
                // Muscles: each pulls its two nodes; its forces go to the
                // bodies that carry them (the head's to the neck). A muscle
                // has no state: its activation is a trapezoid of the time
                // (since the last touchdown of its sensor), and the creature's
                // stamina scales every muscle's force. The work of all the
                // muscles drains the stamina.
                float pw = 0.0f;
                for (unsigned k = 0u; k < nmus; k++) {
                    const float4 f0 = reinterpret_cast<const float4*>(mrec)[2u * k];
                    const float4 f1 = reinterpret_cast<const float4*>(mrec)[2u * k + 1u];
                    const unsigned packed = __float_as_uint(f0.x);
                    const unsigned la = packed & 31u, lb = (packed >> 5u) & 31u, ls = (packed >> 10u) & 31u;
                    const float cap = f0.y, hill = f0.z, inv_period = f0.w;
                    const float phase = f1.x, half = f1.y, inv_ramp = f1.z, reset = f1.w;
                    const float dx = px[lb] - px[la], dy = py[lb] - py[la];
                    const float length_m = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f);
                    const float inverse = 1.0f / length_m;
                    const float dirx = dx * inverse, diry = dy * inverse;
                    const float relative = (vx[lb] - vx[la]) * dirx + (vy[lb] - vy[la]) * diry;
                    const float td = tdn[ls];
                    const bool sensed = ((packed >> 15u) & 1u) != 0u && td >= 0.0f;
                    const float x = sensed ? (ts - td) * inv_period + reset : ts * inv_period + phase;
                    const float ph = x - floorf(x);
                    const float act = limp ? 0.0f : clampf((half - fabsf(ph - half)) * inv_ramp, 0.0f, 1.0f);
                    const float drive = cap * act * stam * clampf(1.0f + relative * hill, 0.0f, 1.0f);
                    const float magnitude = clampf(drive + relative * MUSCLE_DAMPER, -cap, cap);
                    pw += drive * fmaxf(-relative, 0.0f);
                    mag[k] = magnitude;
                    // First law: the muscle's work is its force times its
                    // shortening.
                    ledger -= magnitude * length_m;
                    const float fx = dirx * magnitude, fy = diry * magnitude;
                    bs[la == 0u ? 1u : la] -= force_at(px[la] - ox, py[la] - oy, fx, fy);
                    bs[lb == 0u ? 1u : lb] += force_at(px[lb] - ox, py[lb] - oy, fx, fy);
                }
                // The stamina pays the work of every muscle and recovers a
                // share of what is missing.
                stam = clampf(stam - pw * HS * inv_cap
                    + MUSCLE_RECOVERY * p.muscle_recovery * HS * (1.0f - stam), 0.0f, 1.0f);

                // Articulated-body pass, children first. Joint damping and the
                // joint limits' inelastic stops are implicit in each joint's
                // inertia.
                vec3 uvs[N], cv[N], acc[N];
                float dis[N], uus[N], qdd[N];
                for (unsigned j = 0u; j < N; j++) {
                    uvs[j] = v3(0.0f, 0.0f, 0.0f); cv[j] = uvs[j]; acc[j] = uvs[j];
                    dis[j] = 0.0f; uus[j] = 0.0f; qdd[j] = 0.0f; lig_k[j] = 0.0f;
                }
                for (unsigned j = nn - 1u; j >= 2u; j--) {
                    const float armx = -axis[j].z, army = axis[j].y;
                    // The joint's velocity-product acceleration.
                    cv[j] = crm(v3(om[j], pvx[j] + om[j] * army, pvy[j] - om[j] * armx), axis[j]) * qd[j];
                    // Damping and a limit stop that engages are implicit terms
                    // that scale with the joint's inertia d: the joint's
                    // inertia becomes d * jf and its torque -d * jg.
                    float jf = 1.0f + INV_JOINT_DAMPING * HS;
                    float jg = INV_JOINT_DAMPING * qd[j];
                    const float predicted = q[j] + HS * qd[j];
                    const bool upper = predicted > hi[j];
                    if (upper || predicted < lo[j]) {
                        if (lig2[j] > 0.0f) {
                            // A ligament: the stop is an implicit spring and
                            // damper on the joint's own inertia.
                            const float pen = predicted - (upper ? hi[j] : lo[j]);
                            const float cd = (2.0f * LIGAMENT_DAMPING) * sqrtf(lig2[j]);
                            jg += cd * qd[j] + lig2[j] * pen;
                            jf += cd * HS + lig2[j] * HS * HS;
                        } else {
                            const float room = upper ? hi[j] - q[j] : lo[j] - q[j];
                            const bool past = (room < 0.0f) == upper;
                            const float goal = (past ? room * PUSH_OUT : room) * INV_HS;
                            if ((qd[j] > goal) == upper) {
                                jg += LIMIT_HARDNESS * jf * INV_HS * (qd[j] - goal);
                                jf *= 1.0f + LIMIT_HARDNESS;
                            }
                        }
                    }
                    const vec3 uv = sym_mul(i0[j], i1[j], axis[j]);
                    const float d = sdot(axis[j], uv);
                    if (lig2[j] > 0.0f) {
                        // The store of the ligament at the substep's start.
                        const float pen = fmaxf(q[j] - hi[j], 0.0f) + fminf(q[j] - lo[j], 0.0f);
                        lig_k[j] = d * lig2[j];
                        ledger -= 0.5f * lig_k[j] * pen * pen;
                    }
                    const float u = -d * jg - sdot(axis[j], bs[j]);
                    const float di = 1.0f / (d * jf);
                    uvs[j] = uv; dis[j] = di; uus[j] = u;
                    const float kx = -di * uv.x, ky = -di * uv.y, kz = -di * uv.z;
                    const unsigned pa = parent(j);
                    i0[pa] += i0[j] + v3(kx * uv.x, kx * uv.y, kx * uv.z);
                    i1[pa] += i1[j] + v3(ky * uv.y, ky * uv.z, kz * uv.z);
                    bs[pa] += bs[j] + sym_mul(i0[j] + v3(kx * uv.x, kx * uv.y, kx * uv.z),
                        i1[j] + v3(ky * uv.y, ky * uv.z, kz * uv.z), cv[j]) + uv * (u * di);
                }
                // The neck floats freely: the root's inverse articulated inertia.
                vec3 r0, r1;
                {
                    const vec3 a0 = i0[1], a1 = i1[1];
                    const float c00 = a1.x * a1.z - a1.y * a1.y;
                    const float c01 = a0.z * a1.y - a0.y * a1.z;
                    const float c02 = a0.y * a1.y - a0.z * a1.x;
                    const float inv_det = 1.0f / (a0.x * c00 + a0.y * c01 + a0.z * c02);
                    const float c11 = a0.x * a1.z - a0.z * a0.z;
                    const float c12 = a0.y * a0.z - a0.x * a1.y;
                    const float c22 = a0.x * a1.x - a0.y * a0.y;
                    r0 = v3(c00 * inv_det, c01 * inv_det, c02 * inv_det);
                    r1 = v3(c11 * inv_det, c12 * inv_det, c22 * inv_det);
                    acc[1] = -sym_mul(r0, r1, bs[1]);
                }
                for (unsigned j = 2u; j < nn; j++) {
                    const vec3 a = acc[parent(j)] + cv[j];
                    qdd[j] = (uus[j] - sdot(uvs[j], a)) * dis[j];
                    acc[j] = a + axis[j] * qdd[j];
                }

                // Ground contacts at velocity level: the deepest MAXC nodes
                // that would reach the ground within the substep, solved
                // together.
                nc = 0u;
#if GROUND
                {
                    float gap[N], dry[N], vnf[N], depth_c[N], cnx[N], cny[N];
                    unsigned cb = 0u;
                    for (unsigned j = 0u; j < nn; j++) {
                        const vec3 ba = acc[j == 0u ? 1u : j];
                        const float bw = om[j == 0u ? 1u : j];
                        const float2 g = terrain(px[j]);
                        const float secant = sqrtf(1.0f + g.y * g.y);
                        cnx[j] = -g.y / secant;
                        cny[j] = 1.0f / secant;
                        dry[j] = (py[j] - g.x) / secant - rad[j];
#if MUD
                        gap[j] = dry[j] + p.mud;
#else
                        gap[j] = dry[j];
#endif
                        const vec3 dn = force_at(px[j] - ox, py[j] - oy, cnx[j], cny[j]);
                        vnf[j] = vx[j] * cnx[j] + vy[j] * cny[j] + HS * (sdot(dn, ba) + bw * (-vy[j] * cnx[j] + vx[j] * cny[j]));
                        depth_c[j] = gap[j] + HS * vnf[j];
                        if (depth_c[j] <= 0.0f) { cb |= 1u << j; }
                    }
                    const unsigned ncand = __popc(cb);
                    // Each slot's lane: in lane order, or the deepest MAXC
                    // (ties to the lower lane) when more would touch.
                    unsigned sl[MAXC] = {0u, 0u, 0u, 0u};
                    if (ncand <= MAXC) {
                        unsigned rest = cb;
                        for (unsigned s = 0u; s < ncand; s++) { sl[s] = __ffs(rest) - 1u; rest &= rest - 1u; }
                    } else {
                        unsigned taken = 0u;
                        for (unsigned s = 0u; s < MAXC; s++) {
                            unsigned best = 0u, at = 0u;
                            for (unsigned j = 0u; j < nn; j++) {
                                if (((cb & ~taken) >> j) & 1u) {
                                    const unsigned key = (__float_as_uint(fmaxf(-depth_c[j], 0.0f)) & ~63u) | 32u | (31u - j);
                                    if (key > best) { best = key; at = j; }
                                }
                            }
                            sl[s] = at;
                            taken |= 1u << at;
                        }
                    }
                    nc = min(ncand, (unsigned)MAXC);
                    if (nc > 0u) {
                        // Per contact: its rows' unit forces (climbed to the
                        // root), its torques at every joint on its path, its
                        // speeds and its friction.
                        vec3 pn[MAXC], pt[MAXC], dnc[MAXC], dtc[MAXC];
                        float tjn[MAXC][N], tjt[MAXC][N];
                        float vn[MAXC], vt[MAXC], vs[MAXC], goal[MAXC], mu[MAXC], ln[MAXC], lt[MAXC];
                        for (unsigned c = 0u; c < nc; c++) {
                            const unsigned j = sl[c];
                            const float tx = cny[j], ty = -cnx[j];
                            const vec3 ba = acc[j == 0u ? 1u : j];
                            const float bw = om[j == 0u ? 1u : j];
                            dnc[c] = force_at(px[j] - ox, py[j] - oy, cnx[j], cny[j]);
                            dtc[c] = force_at(px[j] - ox, py[j] - oy, tx, ty);
                            vn[c] = vnf[j];
                            vt[c] = vx[j] * tx + vy[j] * ty + HS * (sdot(dtc[c], ba) + bw * (-vy[j] * tx + vx[j] * ty));
                            vs[c] = vx[j] * tx + vy[j] * ty;
                            goal[c] = gap[j] >= 0.0f ? -gap[j] * INV_HS : -gap[j] * PUSH_OUT * INV_HS;
#if MUD
                            const float sink = clampf(-dry[j], 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH);
                            mu[c] = fric[j] * p.friction * (1.0f + MUD_GRIP * sink) * (1.0f + MUD_NORMAL * sink);
#else
                            mu[c] = fric[j] * p.friction;
#endif
#if ICE
                            mu[c] *= 1.0f - p.patches * ice_at(px[j]);
#endif
                            ln[c] = 0.0f; lt[c] = 0.0f;
                            // Climb from the contact's body to the root: at
                            // every joint it passes, its unit forces reach the
                            // joint as t = -s.p.
                            pn[c] = -dnc[c]; pt[c] = -dtc[c];
                            for (unsigned k = 0u; k < N; k++) { tjn[c][k] = 0.0f; tjt[c][k] = 0.0f; }
                            for (unsigned b = j == 0u ? 1u : j; level(b) >= 2u; b = parent(b)) {
                                const float tn = -sdot(axis[b], pn[c]);
                                const float tt = -sdot(axis[b], pt[c]);
                                pn[c] += uvs[b] * (tn * dis[b]);
                                pt[c] += uvs[b] * (tt * dis[b]);
                                tjn[c][b] = tn;
                                tjt[c][b] = tt;
                            }
                        }
                        // Contact-space matrix: two contacts at the same joint
                        // add dinv t_r t_c, and at the root F_r' Phi F_c.
                        float4 kr[MAXC][MAXC];
                        for (unsigned r = 0u; r < nc; r++) {
                            const vec3 phn = sym_mul(r0, r1, pn[r]), pht = sym_mul(r0, r1, pt[r]);
                            for (unsigned c = 0u; c < nc; c++) {
                                float4 k = make_float4(sdot(phn, pn[c]), sdot(phn, pt[c]), sdot(pht, pn[c]), sdot(pht, pt[c]));
                                for (unsigned b = 2u; b < nn; b++) {
                                    const float wn = dis[b] * tjn[r][b], wt = dis[b] * tjt[r][b];
                                    k.x += wn * tjn[c][b];
                                    k.y += wn * tjt[c][b];
                                    k.z += wt * tjn[c][b];
                                    k.w += wt * tjt[c][b];
                                }
                                kr[r][c] = make_float4(k.x * HS, k.y * HS, k.z * HS, k.w * HS);
                            }
                        }
                        // Projected Gauss-Seidel: each contact updates its
                        // normal, then its friction against its own new speed,
                        // and the change reaches every contact's speeds.
                        for (unsigned sweep = 0u; sweep < PGS_SWEEPS + CLEAN_SWEEPS; sweep++) {
                            const bool clean = sweep >= PGS_SWEEPS;
                            for (unsigned c = 0u; c < nc; c++) {
                                const float4 k = kr[c][c];
                                const float normal = clean ? ln[c] : fmaxf(fmaf(goal[c] - vn[c], 1.0f / k.x, ln[c]), 0.0f);
                                const float dn_ = normal - ln[c];
                                ln[c] += dn_;
                                const float vt1 = vt[c] + k.z * dn_;
                                // Friction may not do positive work: it only
                                // opposes a = start speed + end speed without
                                // its own force, and only up to |a| / k.
                                const float a = vs[c] + vt1 - k.w * lt[c];
                                const float cap = fminf(mu[c] * ln[c], fabsf(a) / k.w);
                                const float tgt = clean ? lt[c] : lt[c] - vt1 / k.w;
                                const float friction = clampf(tgt, a > 0.0f ? -cap : 0.0f, a > 0.0f ? 0.0f : cap);
                                const float dt_ = friction - lt[c];
                                lt[c] += dt_;
                                for (unsigned r = 0u; r < nc; r++) {
                                    const float4 kc = kr[r][c];
                                    vn[r] += kc.x * dn_ + kc.y * dt_;
                                    vt[r] += kc.z * dn_ + kc.w * dt_;
                                }
                            }
                        }
                        // The response to the contact impulses through the
                        // same articulated inertias: the root takes the
                        // contacts' forces, and every joint their torques.
                        vec3 pr = v3(0.0f, 0.0f, 0.0f);
                        for (unsigned c = 0u; c < nc; c++) {
                            pr += pn[c] * ln[c] + pt[c] * lt[c];
                            mx += (ln[c] * dnc[c].y + lt[c] * dtc[c].y) * HS;
                            my += (ln[c] * dnc[c].z + lt[c] * dtc[c].z) * HS;
                            step_n[sl[c]] += ln[c];
                            step_t[sl[c]] += lt[c];
                        }
                        vec3 ar[N];
                        ar[0] = v3(0.0f, 0.0f, 0.0f);
                        ar[1] = -sym_mul(r0, r1, pr);
                        acc[1] += ar[1];
                        for (unsigned j = 2u; j < nn; j++) {
                            float dqr = 0.0f;
                            for (unsigned c = 0u; c < nc; c++) { dqr += ln[c] * tjn[c][j] + lt[c] * tjt[c][j]; }
                            const vec3 pa = ar[parent(j)];
                            const float tq = (dqr - sdot(uvs[j], pa)) * dis[j];
                            ar[j] = pa + axis[j] * tq;
                            qdd[j] += tq;
                        }
                    }
                }
#endif
                // Semi-implicit Euler on the joint coordinates.
                {
                    const float hax = acc[1].y - qd[1] * vy[0];
                    const float hay = acc[1].z + qd[1] * vx[0];
#if AIR
                    vx[0] = (vx[0] + hax * HS) * p.air_sub;
                    vy[0] = (vy[0] + hay * HS) * p.air_sub;
#else
                    vx[0] = vx[0] + hax * HS;
                    vy[0] = vy[0] + hay * HS;
#endif
                    px[0] += vx[0] * HS;
                    py[0] += vy[0] * HS;
                    for (unsigned j = 1u; j < nn; j++) {
                        const float a = j == 1u ? acc[1].x : qdd[j];
#if AIR
                        qd[j] = (qd[j] + a * HS) * p.air_sub;
#else
                        qd[j] = qd[j] + a * HS;
#endif
                        q[j] += qd[j] * HS;
                    }
                }
            }
            q[1] = q[1] - TAU_F * floorf((q[1] + PI_F) / TAU_F);
            for (unsigned j = 0u; j < nn; j++) {
                rec_n[j] = step_n[j] * (1.0f / SUBSTEPS);
                rec_t[j] = step_t[j] * (1.0f / SUBSTEPS);
            }

            // Metrics, falls and the screen, once per step.
            bool failed = false, broken = false;
            float center_y = 0.0f, com_x = 0.0f, low = 1e20f, high = -1e20f;
            unsigned now = 0u, lifted = 0u;
            for (unsigned j = 0u; j < nn; j++) {
                failed = failed || !(fabsf(px[j]) <= 1e6f && fabsf(py[j]) <= 1e6f);
                broken = broken || broken_at(j);
                center_y += py[j];
                com_x += m[j] * px[j];
                low = fminf(low, py[j] - rad[j]);
                high = fmaxf(high, py[j] + rad[j]);
#if GROUND
                const float2 g = terrain(px[j]);
                const float floor_y = g.x + rad[j] * sqrtf(1.0f + g.y * g.y);
                if (py[j] <= floor_y + CONTACT_SLACK) { now |= 1u << j; }
                if (py[j] > floor_y + LIFT_CLEARANCE) { lifted |= 1u << j; }
#endif
            }
            center_y *= inv_nodes;
            com_x *= inv_mass;
            contact_bits |= now;
            lift_bits |= contact_bits & lifted;
            const unsigned down = now & ~ground_bits;
            ground_bits = now;
            // A touchdown restarts the rhythm of the muscles that sense that
            // node, from the next step.
            if (step > 0u) {
                for (unsigned j = 0u; j < nn; j++) { if ((down >> j) & 1u) { tdn[j] = t_now + DT; } }
            }
            if (t_now >= HEAD_SHAKE_WINDOW) {
                const float dvx = vx[0] - head_vx0, dvy = vy[0] - head_vy0;
                const float accel = sqrtf(dvx * dvx + dvy * dvy) * RATE;
                head_shake += (accel - head_shake) * fminf(1.0f / (HEAD_SHAKE_WINDOW * RATE), 1.0f);
            }
            const float head_y = py[0], neck_y = py[1];
            // The rung trace (creature_kernel::RungTrace), as in
            // shaders/warp_creature.cu.
            const unsigned rung1 = (unsigned)(RATE) - 1u, rung2 = (unsigned)(2.5f * RATE) - 1u;
            const unsigned rung3 = (unsigned)(5.0f * RATE) - 1u, rung4 = (unsigned)(10.0f * RATE) - 1u;
            const unsigned half_s = (unsigned)(0.5f * RATE);
            const bool early = step == rung1 || step == rung2;
            // The creature's stamina stands for the mean muscle energy.
            const float en_mean = stam;
            bool live = true;
#if RECORD
            if (!done_scoring)
#endif
            {
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
                if (step + half_s == rung1 || step + half_s == rung2) { rung.x = __float_as_uint(com_x); }
                if (early) {
                    const bool second = step == rung2;
                    const unsigned sh = second ? 16u : 0u;
                    const unsigned keep = second ? 0x0000ffffu : 0xffff0000u;
                    const float speed = (com_x - __uint_as_float(rung.x)) * (RATE / (float)half_s);
                    const float touched = (float)__popc(contact_bits) * inv_nodes;
                    mt.contact_hi = __uint_as_float((__float_as_uint(mt.contact_hi) & keep) | (f2h(com_x) << sh));
                    mt.ground_hi = __uint_as_float((__float_as_uint(mt.ground_hi) & keep) | (f2h(head_shake) << sh));
                    rung.y = (rung.y & keep) | (f2h(speed) << sh);
                    const unsigned pair = f2h(touched) | (f2h(en_mean) << 16u);
                    if (second) { rung.w = pair; } else { rung.z = pair; }
                    // The cadence band the audit lane files this trial under.
                    const unsigned r = second ? 1u : 0u;
                    const float gait_now = mt.gait_turns * (0.5f / (t_now + DT));
                    const unsigned band = min((unsigned)(gait_now * (8.0f / 6.0f)), 7u);
                    rbits |= band << (8u + 3u * r);
                    // Flags: 1 audit, 2 exempt, 4 and 8 exempt from R1 and R2.
                    if (!ended && ((head_flags >> 16u) & (second ? 0xbu : 0x7u)) == 0u) {
                        const RungParams rp = second ? p.r2 : p.r1;
                        const float period_f = h2f(head_flags & 0xffffu);
                        const float fv[6] = {h2f(f2h(com_x)), h2f(f2h(speed)), h2f(f2h(touched)),
                                             h2f(f2h(head_shake)), h2f(f2h(en_mean)), period_f};
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
                    mt.lift_hi = __uint_as_float((__float_as_uint(mt.lift_hi) & keep) | (f2h(com_x) << (second ? 16u : 0u)));
                }
                if (step == p.screen_step && !ended) {
                    mt.screen_x = com_x;
                    if (com_x < p.screen_bar && ((head_flags >> 16u) & 1u) == 0u) {
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
                        | (rbits & 0x3fc0u) | (((head_flags >> 16u) & 1u) << 14u);
                    mt.previous_center_y = __uint_as_float(rung.y);
                    mt.vertical_extremum = __uint_as_float(rung.z);
                    mt.vertical_trend = __uint_as_float(rung.w);
                    mt.gait_turns = mt.ground_hi;
                    mt.ground_hi = __uint_as_float(code | (min(step + 1u, 65535u) << 16u));
#if RECORD
                    kept = mt;
                    done_scoring = true;
                    limp = true;
#else
                    results[cidx] = mt;
                    live = false;
#endif
                }
            }
#if RECORD
            record_frame(SETTLE + step + 1u);
            if (step + 1u >= p.steps) {
                results[cidx] = kept;
                live = false;
            }
#endif
            if (!live) { break; }
        }
    }
}
