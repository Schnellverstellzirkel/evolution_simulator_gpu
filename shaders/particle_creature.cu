// Position-based physics (XPBD, Macklin et al. 2016; small steps with one
// iteration, Macklin et al. 2019; positional friction, Mueller et al. 2020),
// one creature per thread. src/warp_kernel.rs packs the creatures by lane
// class (W = 8, 16 or 32 node slots) and writes the #defines and
// shaders/creature_common.cu in front of this text.
//
// Nodes are particles with the masses physics2::Model gives them. A bone is
// a rod between its pivot and its node. A joint keeps the angle between its
// bone and its parent bone inside the joint's range: a hard stop, or with a
// ligament a compliant one. Muscles pull two nodes together with the lean
// muscle model (cap x strength x trapezoid activation x stamina x Hill).
// Each 1/RATE step is PSUB substeps: external and muscle forces change the
// velocities, the positions move, then one pass over the rods, the joint
// limits and the ground contacts projects them, and the velocities follow
// from the moves. Friction only removes motion along the ground, so it never
// does positive work, and a node that holds stays exactly where it is.
//
// The node slots follow the lane packing: breadth first over the bone tree,
// slot 0 the head, slot 1 the neck's node; bone j ends at slot j.

#define N W
// The 8-slot class keeps its moving node state (positions, velocities,
// substep starts) in shared memory, one slot per node and thread, so an
// index known only at run time costs one load; larger classes keep it in
// thread-local memory.
#define UNROLL _Pragma("unroll")
struct Slots {
    float* base;
    __device__ __forceinline__ float& operator[](unsigned j) const { return base[j * BLOCK]; }
};
template <typename A>
__device__ __forceinline__ float pick(const A& a, unsigned i) { return a[i]; }
template <typename A>
__device__ __forceinline__ void add_at(A& a, unsigned i, float v) { a[i] += v; }
template <typename A>
__device__ __forceinline__ void add_at(const A& a, unsigned i, float v) { a[i] += v; }
// Node loops: every slot below the creature's node count.
#define NODES(j, from) UNROLL for (unsigned j = (from); j < N; j++) if (j < nn)
#define PSUB 4
#define PH (1.0f / (RATE * PSUB))
#define INV_PH (RATE * PSUB)
// A node faster than this (m/s) means the solver blew up.
#define BLOWUP_SPEED 100.0f

extern "C" __global__ void __launch_bounds__(BLOCK, 4) advance(
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
    unsigned bucket = 0u;
    {
        const unsigned order = (threadIdx.x >> 5) * gridDim.x + blockIdx.x;
#pragma unroll
        for (unsigned b = 1u; b < RMAX; b++) { if (tk.warp[b] <= order) { bucket = b; } }
    }
    // Per node: position, velocity, the position at the substep's start, and
    // the last touchdown time; per muscle its force in the last substep.
#if W <= 16
    // Positions and velocities in shared memory, one slot per node and
    // thread; the 8-slot class also keeps its inverse masses and touchdown
    // times there.
    __shared__ float s_state[(W == 8 ? 6 : 4) * N * BLOCK];
    const Slots px{s_state + 0 * N * BLOCK + threadIdx.x}, py{s_state + 1 * N * BLOCK + threadIdx.x};
    const Slots vx{s_state + 2 * N * BLOCK + threadIdx.x}, vy{s_state + 3 * N * BLOCK + threadIdx.x};
#if W == 8
    const Slots imass{s_state + 4 * N * BLOCK + threadIdx.x}, tdn{s_state + 5 * N * BLOCK + threadIdx.x};
#else
    float imass[N], tdn[N];
#endif
    float sx[N], sy[N];
#else
    float px[N], py[N], vx[N], vy[N], sx[N], sy[N], imass[N], tdn[N];
#endif
    // Per node, copied from the lane record once per creature so the
    // substeps read them from this thread's own memory: inverse mass,
    // radius, friction, and the node's bone (length, range, ligament, pivot
    // radius, pivot and parent slots).
    float rad[N], fric[N], blen[N], blo[N], bhi[N], blig[N], bprad[N];
    unsigned char piv[N], par[N];
    float mag[RMAX * N];
#if RECORD
    float rec_n[N], rec_t[N];
#endif
    // The thread's creature; `live` false makes the next pass take one.
    unsigned cidx = 0u, nn = 0u, nmus = 0u, head_flags = 0u, step = 0u;
    const float* mrec = muscles;
    const unsigned* rec = lanes;
    float inv_mass = 0.0f, inv_cap = 0.0f, inv_nodes = 0.0f, qphase = 0.0f, amp = 0.0f, stam = 1.0f;
    bool live = false, limp = false;
    Result mt;
    unsigned contact_bits = 0u, lift_bits = 0u, ground_bits = 0u, rbits = 0u;
    float head_shake = 0.0f;
    uint4 rung = make_uint4(0u, 0u, 0u, 0u);
#if RECORD
    Result kept;
    bool done_scoring = false;
#endif
#if AIR
    // The air's retention per substep (p.air_sub is per lane-group substep).
    const float air_ps = powf(p.air_sub, (float)SUBSTEPS / (float)PSUB);
#endif
    auto terrain = [&](float x) -> float2 { return ground_at(x, p, amp, qphase); };
    auto field = [&](unsigned f, unsigned j) { return __uint_as_float(__ldg(rec + f * W + j)); };
    auto mass = [&](unsigned j) { return 1.0f / imass[j]; };
    auto radius = [&](unsigned j) { return rad[j]; };
    auto pivot = [&](unsigned j) { return (unsigned)piv[j]; };
    auto parent = [&](unsigned j) { return (unsigned)par[j]; };
    // The angle of bone j against its parent bone (the neck's against the x
    // axis), as the joint coordinate q of the reduced model.
    auto joint_angle = [&](unsigned j) {
        const unsigned a = pivot(j);
        const float vx_ = px[j] - pick(px, a), vy_ = py[j] - pick(py, a);
        if (j == 1u) { return atan2f(vy_, vx_); }
        const unsigned c = parent(j), d = pivot(c);
        const float ux = pick(px, c) - pick(px, d), uy = pick(py, c) - pick(py, d);
        // Ranges may reach past half a turn, so the angle is taken within
        // half a turn of the range's middle.
        const float mid = 0.5f * (blo[j] + bhi[j]);
        const float rel = atan2f(ux * vy_ - uy * vx_, ux * vx_ + uy * vy_) - mid;
        return mid + rel - TAU_F * rintf(rel * (1.0f / TAU_F));
    };
    auto broken_at = [&](unsigned j) {
        if (j < 2u) { return false; }
        const float q = joint_angle(j);
        return q < blo[j] - JOINT_BREAK || q > bhi[j] + JOINT_BREAK;
    };
#if RECORD
    auto record_frame = [&](unsigned t) {
        const unsigned fb = t * p.stride;
        unsigned lo_bits = 0u, hi_bits = 0u;
        NODES(j, 0u) {
            const unsigned node = __ldg(rec + 11u * W + j);
            frames[fb + node] = make_float2(px[j], py[j]);
            frames[fb + W + nmus + node] = make_float2(rec_n[j], rec_t[j]);
            // Broken joints by the creature's bone number (bone j ends at node j + 1).
            if (broken_at(j)) {
                const unsigned b = node - 1u;
                if (b < 32u) { lo_bits |= 1u << b; } else { hi_bits |= 1u << (b - 32u); }
            }
        }
        for (unsigned k = 0u; k < nmus; k++) { frames[fb + W + k] = make_float2(stam, mag[k]); }
        frames[fb + p.stride - 1u] = make_float2(__uint_as_float(lo_bits), __uint_as_float(hi_bits));
    };
#endif
    // One pass is one step of every thread's creature, so the threads of a
    // warp run the same step code together; a thread whose creature ended
    // takes the next one first.
    for (;;) {
        if (!live) {
            unsigned got = 0xffffffffu;
            for (unsigned k = 0u; k < RMAX; k++) {
                const unsigned b = (bucket + k) % RMAX;
                const unsigned size = tk.end[b] - tk.start[b];
                if (size == 0u || *(volatile unsigned*)&counter[b] >= size) { continue; }
                const unsigned i = atomicAdd(&counter[b], 1u);
                if (i < size) { got = tk.start[b] + i; bucket = b; break; }
            }
            if (got >= p.count) { break; }
            cidx = p.base + got;
            const uint4 h0 = heads[2u * cidx];
            const uint4 h1 = heads[2u * cidx + 1u];
            nn = h0.x & 255u;
            nmus = h0.y & 255u;
            mrec = muscles + h0.w;
            inv_mass = __uint_as_float(h1.z);
            inv_cap = __uint_as_float(h1.w) * p.inv_muscle_energy;
            inv_nodes = 1.0f / (float)nn;
            // The rhythm period (a rung feature) in the low half and the
            // flags in the high half.
            head_flags = (h0.y >> 16u) | ((h0.y >> 8u) & 255u) << 16u;
#if QUAKE
            qphase = (float)(h0.z & 0xffffu) * (1.0f / 65536.0f);
            amp = p.terrain + p.quake * (0.6f + (float)((h0.z >> 16u) & 0xffffu) * (0.8f / 65536.0f));
#else
            qphase = 0.0f;
            amp = p.terrain;
#endif
            rec = lanes + cidx * (LF * W);
            NODES(j, 0u) {
                imass[j] = 1.0f / field(0u, j);
                rad[j] = field(1u, j);
                fric[j] = field(2u, j);
                blen[j] = field(3u, j);
                blo[j] = field(4u, j);
                bhi[j] = field(5u, j);
                blig[j] = field(7u, j);
                bprad[j] = field(8u, j);
                const unsigned t = __ldg(rec + 9u * W + j);
                piv[j] = (unsigned char)(t & 31u);
                par[j] = (unsigned char)((t >> 5u) & 31u);
            }
            // The start pose: the head where the record puts it and every
            // bone at its start angle, parents first.
            {
                float th[N];
                px[0] = field(6u, 0u);
                py[0] = field(7u, 0u);
                th[0] = 0.0f;
                NODES(j, 1u) {
                    const float q = field(6u, j);
                    th[j] = j == 1u ? q : pick(th, parent(j)) + q;
                    const unsigned a = pivot(j);
                    px[j] = pick(px, a) + blen[j] * cosf(th[j]);
                    py[j] = pick(py, a) + blen[j] * sinf(th[j]);
                }
                NODES(j, 0u) {
                    vx[j] = 0.0f; vy[j] = 0.0f; tdn[j] = -1.0f;
#if RECORD
                    rec_n[j] = 0.0f; rec_t[j] = 0.0f;
#endif
                }
            }
            for (unsigned k = 0u; k < RMAX * N; k++) { mag[k] = 0.0f; }
            stam = 1.0f;
            limp = false;
            mt.fitness = 0.0f; mt.ground_contact = 0.0f; mt.vertical_oscillation = 1e20f; mt.gait_frequency = -1e20f;
            mt.previous_center_y = 0.0f; mt.vertical_extremum = 0.0f; mt.vertical_trend = 0.0f; mt.gait_turns = 0.0f;
            mt.height_sum = 0.0f; mt.contact_lo = 0.0f; mt.contact_hi = 0.0f; mt.lift_lo = 0.0f; mt.lift_hi = 0.0f;
            mt.ground_lo = 0.0f; mt.ground_hi = 0.0f; mt.fall_time = 0.0f; mt.head_shake = 0.0f; mt.screen_x = 0.0f;
            mt.screened = 0.0f;
            contact_bits = 0u; lift_bits = 0u; ground_bits = 0u; rbits = 0u;
            head_shake = 0.0f;
            rung = make_uint4(0u, 0u, 0u, 0u);
            step = 0u;
            live = true;
#if RECORD
            kept = mt;
            done_scoring = false;
            for (unsigned t = 0u; t <= SETTLE; t++) { record_frame(t); }
#endif
        }
        {
            const float t_now = (float)step * DT;
            const float head_vx0 = vx[0], head_vy0 = vy[0];
#if RECORD
            NODES(j, 0u) { rec_n[j] = 0.0f; rec_t[j] = 0.0f; }
#endif
            for (unsigned sub = 0u; sub < PSUB; sub++) {
                const float ts = t_now + (float)sub * PH;
                // External forces: gravity, wind, mud drag and buoyancy on
                // each node.
                NODES(j, 0u) {
                    const float m = mass(j);
                    float ax = 0.0f, ay = -p.gravity;
#if WIND
                    ax += p.wind;
#endif
#if MUD
                    {
                        const float2 g = terrain(px[j]);
                        const float dry = (py[j] - g.x) / sqrtf(1.0f + g.y * g.y) - radius(j);
                        ax -= MUD_DRAG * (clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH)) * vx[j];
                    }
#endif
#if WATER
                    {
                        const float r = radius(j);
                        ay += WATER_BUOYANCY * p.gravity * clampf((p.water - (py[j] - r)) / (2.0f * r), 0.0f, 1.0f);
                    }
#endif
                    vx[j] += ax * PH;
                    vy[j] += ay * PH;
                    (void)m;
                }
                // Air and water drag on each bone at its midpoint, half to
                // each end, limited so a substep never more than halves the
                // speed.
                NODES(j, 1u) {
                    const unsigned a = pivot(j);
                    const float len = blen[j];
                    const float wx = 0.5f * (pick(vx, a) + vx[j]), wy = 0.5f * (pick(vy, a) + vy[j]);
                    const float speed = sqrtf(wx * wx + wy * wy);
                    const float width = bprad[j] + radius(j);
                    const float ma = mass(a), mj = mass(j);
                    const float lim = 0.5f * fminf(ma, mj) * INV_PH;
                    float fx = -wx * fmaxf(fminf(AIR_DRAG * len * width * speed, lim), 0.0f);
                    float fy = -wy * fmaxf(fminf(AIR_DRAG * len * width * speed, lim), 0.0f);
#if WATER
                    {
                        const float ri = radius(j), rp = bprad[j];
                        const float sub_i = clampf((p.water - (py[j] - ri)) / (2.0f * ri), 0.0f, 1.0f);
                        const float sub_p = clampf((p.water - (pick(py, a) - rp)) / (2.0f * rp), 0.0f, 1.0f);
                        const float wet = 0.5f * (sub_p + sub_i);
                        const float inverse = 1.0f / len;
                        const float ax_ = (px[j] - pick(px, a)) * inverse, ay_ = (py[j] - pick(py, a)) * inverse;
                        const float along = wx * ax_ + wy * ay_;
                        const float lx = ax_ * along, ly = ay_ * along;
                        const float ws = fmaxf(fminf(WATER_DRAG * wet * len * width * speed, lim), 0.0f);
                        const float weak = ws * WATER_ALONG;
                        fx -= (wx - lx) * ws + lx * weak;
                        fy -= (wy - ly) * ws + ly * weak;
                    }
#endif
                    add_at(vx, a, (0.5f * fx * PH / ma)); add_at(vy, a, (0.5f * fy * PH / ma));
                    vx[j] += 0.5f * fx * PH / mj; vy[j] += 0.5f * fy * PH / mj;
                }
                // Muscles: a muscle has no state; its activation is a
                // trapezoid of the time (since the last touchdown of its
                // sensor), and the creature's stamina scales every muscle's
                // force. The work of all the muscles drains the stamina.
                float pw = 0.0f;
                for (unsigned k = 0u; k < nmus; k++) {
                    const float4 f0 = __ldg(reinterpret_cast<const float4*>(mrec) + 2u * k);
                    const float4 f1 = __ldg(reinterpret_cast<const float4*>(mrec) + 2u * k + 1u);
                    const unsigned packed = __float_as_uint(f0.x);
                    const unsigned la = packed & 31u, lb = (packed >> 5u) & 31u, ls = (packed >> 10u) & 31u;
                    const float cap = f0.y, hill = f0.z, inv_period = f0.w;
                    const float phase = f1.x, half = f1.y, inv_ramp = f1.z, reset = f1.w;
                    const float dx = pick(px, lb) - pick(px, la), dy = pick(py, lb) - pick(py, la);
                    const float length_m = fmaxf(sqrtf(dx * dx + dy * dy), 1e-6f);
                    const float inverse = 1.0f / length_m;
                    const float dirx = dx * inverse, diry = dy * inverse;
                    const float relative = (pick(vx, lb) - pick(vx, la)) * dirx + (pick(vy, lb) - pick(vy, la)) * diry;
                    const float td = pick(tdn, ls);
                    const bool sensed = ((packed >> 15u) & 1u) != 0u && td >= 0.0f;
                    const float x = sensed ? (ts - td) * inv_period + reset : ts * inv_period + phase;
                    const float ph = x - floorf(x);
                    const float act = limp ? 0.0f : clampf((half - fabsf(ph - half)) * inv_ramp, 0.0f, 1.0f);
                    const float drive = cap * act * stam * clampf(1.0f + relative * hill, 0.0f, 1.0f);
                    const float magnitude = clampf(drive + relative * MUSCLE_DAMPER, -cap, cap);
                    pw += drive * fmaxf(-relative, 0.0f);
                    mag[k] = magnitude;
                    const float wa = pick(imass, la), wb = pick(imass, lb);
                    // Implicit in Hill's law: the impulse never drives the
                    // ends together faster than the muscle's top shortening
                    // speed 1 / hill within the substep.
                    const float imp = fminf(magnitude * PH, fmaxf(relative + 1.0f / hill, 0.0f) / (wa + wb));
                    add_at(vx, la, (dirx * imp * wa)); add_at(vy, la, (diry * imp * wa));
                    add_at(vx, lb, -(dirx * imp * wb)); add_at(vy, lb, -(diry * imp * wb));
                }
                stam = clampf(stam - pw * PH * inv_cap
                    + MUSCLE_RECOVERY * p.muscle_recovery * PH * (1.0f - stam), 0.0f, 1.0f);
                // Move.
                NODES(j, 0u) {
                    sx[j] = px[j]; sy[j] = py[j];
                    px[j] += vx[j] * PH;
                    py[j] += vy[j] * PH;
                }
                // Rods, parents first.
                NODES(j, 1u) {
                    const unsigned a = pivot(j);
                    const float wa = pick(imass, a), wj = imass[j];
                    const float dx = px[j] - pick(px, a), dy = py[j] - pick(py, a);
                    const float d = fmaxf(sqrtf(dx * dx + dy * dy), 1e-9f);
                    const float c = (d - blen[j]) / (d * (wa + wj));
                    add_at(px, a, (wa * c * dx)); add_at(py, a, (wa * c * dy));
                    px[j] -= wj * c * dx; py[j] -= wj * c * dy;
                }
                // Joint limits: the angle between a bone and its parent bone
                // stays in the joint's range, a ligament making the stop
                // compliant.
                NODES(j, 2u) {
                    const float q = joint_angle(j);
                    const float lo = blo[j], hi = bhi[j];
                    // Inside the range the correction is zero; every thread
                    // runs the same code either way.
                    const float cval = q > hi ? q - hi : (q < lo ? q - lo : 0.0f);
                    const unsigned b = pivot(j), c = parent(j), d = pivot(c);
                    const float vx_ = px[j] - pick(px, b), vy_ = py[j] - pick(py, b);
                    const float ux = pick(px, c) - pick(px, d), uy = pick(py, c) - pick(py, d);
                    const float iv = 1.0f / fmaxf(vx_ * vx_ + vy_ * vy_, 1e-12f);
                    const float iu = 1.0f / fmaxf(ux * ux + uy * uy, 1e-12f);
                    // d angle / d end of each bone.
                    const float gax = -vy_ * iv, gay = vx_ * iv;
                    const float gcx = uy * iu, gcy = -ux * iu;
                    // Gradients per point; the child's pivot is the parent's
                    // end or, for a bone at the head, the parent's pivot.
                    const bool bc = b == c, bd = !bc && b == d;
                    const float gbx = -gax + (bc ? gcx : 0.0f) - (bd ? gcx : 0.0f);
                    const float gby = -gay + (bc ? gcy : 0.0f) - (bd ? gcy : 0.0f);
                    const float gcx2 = bc ? 0.0f : gcx, gcy2 = bc ? 0.0f : gcy;
                    const float gdx = bd ? 0.0f : -gcx, gdy = bd ? 0.0f : -gcy;
                    const float wj = imass[j], wb = pick(imass, b), wc = pick(imass, c), wd = pick(imass, d);
                    const float lig2 = blig[j];
                    const float len = blen[j];
                    // Compliance of a ligament: its stop is a spring of
                    // stiffness lig2 times the joint's inertia.
                    const float alpha = lig2 > 0.0f ? INV_PH * INV_PH / (lig2 * mass(j) * len * len) : 0.0f;
                    const float den = wj * (gax * gax + gay * gay) + wb * (gbx * gbx + gby * gby)
                        + wc * (gcx2 * gcx2 + gcy2 * gcy2) + wd * (gdx * gdx + gdy * gdy) + alpha;
                    const float lam = -cval / den;
                    px[j] += wj * gax * lam; py[j] += wj * gay * lam;
                    add_at(px, b, (wb * gbx * lam)); add_at(py, b, (wb * gby * lam));
                    add_at(px, c, (wc * gcx2 * lam)); add_at(py, c, (wc * gcy2 * lam));
                    add_at(px, d, (wd * gdx * lam)); add_at(py, d, (wd * gdy * lam));
                }
                // Ground contacts: a node below the ground goes back out
                // along the normal; friction takes back its motion along the
                // ground, all of it while the normal push can hold it.
#if GROUND
                NODES(j, 0u) {
                    const float2 g = terrain(px[j]);
                    const float secant = sqrtf(1.0f + g.y * g.y);
                    const float nx = -g.y / secant, ny = 1.0f / secant;
                    const float dry = (py[j] - g.x) / secant - radius(j);
#if MUD
                    const float gap = dry + p.mud;
#else
                    const float gap = dry;
#endif
                    // Above the ground the push and the friction are zero.
                    const float push = fmaxf(-gap, 0.0f);
                    px[j] += push * nx;
                    py[j] += push * ny;
#if MUD
                    const float sink = clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH);
                    float mu = fric[j] * p.friction * (1.0f + MUD_GRIP * sink) * (1.0f + MUD_NORMAL * sink);
#else
                    float mu = fric[j] * p.friction;
#endif
#if ICE
                    mu *= 1.0f - p.patches * ice_at(px[j]);
#endif
                    const float tx = ny, ty = -nx;
                    const float slide = (px[j] - sx[j]) * tx + (py[j] - sy[j]) * ty;
                    const float hold = mu * push;
                    const float back = clampf(slide, -hold, hold);
                    px[j] -= back * tx;
                    py[j] -= back * ty;
#if RECORD
                    const float m = mass(j);
                    rec_n[j] += push * m * INV_PH * INV_PH * (1.0f / PSUB);
                    rec_t[j] += -back * m * INV_PH * INV_PH * (1.0f / PSUB);
#endif
                }
#endif
                // Velocities from the moves, then joint damping and the spin
                // cap on each bone's turn rate against its parent.
                NODES(j, 0u) {
                    vx[j] = (px[j] - sx[j]) * INV_PH;
                    vy[j] = (py[j] - sy[j]) * INV_PH;
                }
                NODES(j, 1u) {
                    const unsigned a = pivot(j);
                    const float dx = px[j] - pick(px, a), dy = py[j] - pick(py, a);
                    const float il = 1.0f / fmaxf(dx * dx + dy * dy, 1e-12f);
                    // The bone's turn rate, and its parent's.
                    const float w = (dx * (vy[j] - pick(vy, a)) - dy * (vx[j] - pick(vx, a))) * il;
                    float wp = 0.0f;
                    if (j >= 2u) {
                        const unsigned c = parent(j), d = pivot(c);
                        const float ux = pick(px, c) - pick(px, d), uy = pick(py, c) - pick(py, d);
                        wp = (ux * (pick(vy, c) - pick(vy, d)) - uy * (pick(vx, c) - pick(vx, d))) / fmaxf(ux * ux + uy * uy, 1e-12f);
                    }
                    float target = w;
                    if (j >= 2u) { target = wp + (w - wp) * __expf(-INV_JOINT_DAMPING * PH); }
                    target = clampf(target, -SPIN_CAP, SPIN_CAP);
                    {
                        // Turn the bone's end about the pair's centre of mass.
                        const float wa = pick(imass, a), wj = imass[j];
                        const float dw = (target - w);
                        const float sjx = -dy * dw, sjy = dx * dw;
                        const float fj = wj / (wa + wj), fa = wa / (wa + wj);
                        vx[j] += sjx * fj; vy[j] += sjy * fj;
                        add_at(vx, a, -(sjx * fa)); add_at(vy, a, -(sjy * fa));
                    }
                }
#if AIR
                NODES(j, 0u) { vx[j] *= air_ps; vy[j] *= air_ps; }
#endif
            }

            // Metrics, falls and the screen, once per step.
            bool failed = false, broken = false;
            float center_y = 0.0f, com_x = 0.0f, low = 1e20f, high = -1e20f;
            unsigned now = 0u, lifted = 0u;
            NODES(j, 0u) {
                // A blow-up of the solver fails the trial like a non-finite
                // position does.
                failed = failed || !(fabsf(px[j]) <= 1e6f && fabsf(py[j]) <= 1e6f)
                    || !(vx[j] * vx[j] + vy[j] * vy[j] <= BLOWUP_SPEED * BLOWUP_SPEED);
                broken = broken || broken_at(j);
                center_y += py[j];
                com_x += mass(j) * px[j];
                low = fminf(low, py[j] - radius(j));
                high = fmaxf(high, py[j] + radius(j));
#if GROUND
                const float2 g = terrain(px[j]);
                const float floor_y = g.x + radius(j) * sqrtf(1.0f + g.y * g.y);
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
                NODES(j, 0u) { if ((down >> j) & 1u) { tdn[j] = t_now + DT; } }
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
        }
        step += 1u;
    }
}
