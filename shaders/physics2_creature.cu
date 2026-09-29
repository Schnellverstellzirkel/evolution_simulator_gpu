// CUDA port of shaders/physics2_creature.wgsl (physics v2, src/physics2.rs)
// for the NVIDIA backend (src/cuda_engine.rs). It computes the same physics in
// the same order as the WGSL kernel, section for section, so the two stay easy
// to compare; read the WGSL file for the reasons behind each step.
//
// creature_kernel::cuda_source2 prepends the constants that
// physics2::shader_source writes into the WGSL text as #define lines: WG,
// MAXN, STRIDE, MAXC (contacts per solve), the sweeps, the physics limits and
// the fidelity, plus UNROLL (the pragma that gives the node and bone loops
// fixed indices for bodies of up to 16 nodes, so the private arrays live in
// registers) and TAB_LOCAL (bodies above 32 nodes keep the per-lane table in
// local memory, as the WGSL kernel keeps it in private memory).
//
// Differences from the WGSL text. None of them changes the physics:
// - Params arrives as a kernel argument instead of a uniform buffer.
// - The WGSL module-level private variables are members of `Lane`, and the
//   functions that use them are its member functions.
// - The table is [field][lane] in shared memory, as in the WGSL kernel.
// - Every float literal carries an f suffix, so no expression runs in double
//   precision.

struct Record {
    float2 a;
    float2 b;
    float2 c;
    float2 d;
};
// Same layout as creature_kernel::Params and the WGSL uniform.
struct Params {
    unsigned tick;
    unsigned steps;
    unsigned stride;
    unsigned count;
    float gravity;
    float air;
    float friction;
    float ground;
    unsigned total_steps;
    float terrain;
    float muscle_energy;
    float muscle_recovery;
    float slope;
    float wind;
    float mud;
    float gaps;
    float hurdles;
    float quake;
    unsigned screen_tick;
    float screen_bar;
};
// Same layout as creature_kernel::GpuResult and the WGSL Result.
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

#define TILE 32u
#define MUSCLE_FIELDS 16u
#define BONE_FIELDS 9u
#define NO_SENSOR 7u
#define MAXB (MAXN - 1u)
#define MAXR (2u * MAXC)
#define DT (1.0f / RATE)
#define PI_F 3.14159265359f
#define TAU_F 6.28318530718f
// Sweeps and the like arrive as #defines; the per-lane table's size.
#define TN 0u
#define TB (4u * MAXN)
#define TABF (4u * MAXN + 6u * MAXB)
// Loops over contacts and rows always unroll: at most a few of each.
#define UNROLL_C _Pragma("unroll")

typedef float3 vec3;

__device__ __forceinline__ float2 v2(float x, float y) { return make_float2(x, y); }
__device__ __forceinline__ vec3 v3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return v2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return v2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 operator*(float2 a, float s) { return v2(a.x * s, a.y * s); }
__device__ __forceinline__ float2 operator*(float s, float2 a) { return v2(s * a.x, s * a.y); }
__device__ __forceinline__ void operator+=(float2& a, float2 b) { a = a + b; }
__device__ __forceinline__ vec3 operator+(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ vec3 operator-(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ vec3 operator-(vec3 a) { return v3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ vec3 operator*(vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ void operator+=(vec3& a, vec3 b) { a = a + b; }
__device__ __forceinline__ void operator-=(vec3& a, vec3 b) { a = a - b; }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

__device__ __forceinline__ float quake_phase(unsigned seed) {
    return (float)(seed & 0xffffu) * (1.0f / 65536.0f);
}
__device__ __forceinline__ float quake_scale(unsigned seed) {
    return 0.6f + (float)((seed >> 16u) & 0xffffu) * (0.8f / 65536.0f);
}

// Planar spatial vectors (angular, x, y).
__device__ __forceinline__ vec3 crm(vec3 v, vec3 u) {
    return v3(0.0f, -v.x * u.z + u.x * v.z, v.x * u.y - u.x * v.y);
}
__device__ __forceinline__ vec3 crf(vec3 v, vec3 f) {
    return v3(v.y * f.z - v.z * f.y, -v.x * f.z, v.x * f.y);
}
__device__ __forceinline__ vec3 force_at(float2 r, float2 f) {
    return v3(r.x * f.y - r.y * f.x, f.x, f.y);
}
__device__ __forceinline__ vec3 sym_mul(vec3 s0, vec3 s1, vec3 v) {
    return v3(
        s0.x * v.x + s0.y * v.y + s0.z * v.z,
        s0.y * v.x + s1.x * v.y + s1.y * v.z,
        s0.z * v.x + s1.y * v.y + s1.z * v.z);
}
__device__ __forceinline__ float sdot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float wrap_angle(float a) {
    return a - TAU_F * floorf((a + PI_F) / TAU_F);
}
// The waveform's shape at time t.
__device__ __forceinline__ float wave(float t, float inv_period, float phase, float offset, float duty, float inv_duty, float inv_complement) {
    float x = t * inv_period + phase + offset;
    float ph = x - floorf(x);
    if (ph < duty) {
        return 0.5f + 0.5f * cosf(PI_F * (ph * inv_duty));
    }
    return 0.5f - 0.5f * cosf(PI_F * ((ph - duty) * inv_complement));
}
// The symmetric contact-space matrix, lower triangle.
__device__ __forceinline__ unsigned tri(unsigned r, unsigned c) {
    unsigned hi_ = max(r, c);
    return hi_ * (hi_ + 1u) / 2u + min(r, c);
}

struct Lane {
    const Params& p;
    const Record* records_in;
    const float* bone_data;
#if TAB_LOCAL
    float tab[TABF];
#else
    float* tab;
#endif
    unsigned lane_id;
    unsigned bone_base;
    unsigned record_base;
    unsigned nn;
    unsigned nb;
    float total_mass;
    float inv_mass;
    float phase_q;
    float amplitude;
    bool rough;
    float mass[MAXN];
    unsigned pivot[MAXB];
    float len[MAXB];
    // State.
    float2 x0;
    float2 v0;
    float q[MAXB];
    float qd[MAXB];
    float om[MAXB];
    // The articulated-body pass.
    vec3 i0[MAXB];
    vec3 i1[MAXB];
    vec3 bias[MAXB];
    float2 arm[MAXB];
    vec3 uvec[MAXB];
    float dinv[MAXB];
    float uu[MAXB];
    vec3 acc[MAXB];
    float qdd[MAXB];
    float dq[MAXB];
    vec3 root0;
    vec3 root1;
    // Contacts of this step, in slots.
    unsigned nc;
    bool c_on[MAXC];
    unsigned c_node[MAXC];
    vec3 c_dn[MAXC];
    vec3 c_dt[MAXC];
    float c_vn[MAXC];
    float c_vt[MAXC];
    // The node's speed along the ground at the start of the step: friction
    // may only push against the mean of this and the speed after the step.
    float c_vs[MAXC];
    float c_goal[MAXC];
    float c_mu[MAXC];
    float reach[MAXN];
    float kmat[MAXR * (MAXR + 1u) / 2u];
    float lambda[MAXR];
    float old[MAXR];
    float vrow[MAXR];

    __device__ __forceinline__ Lane(const Params& params) : p(params) {}

    __device__ __forceinline__ unsigned ti(unsigned f) const {
#if TAB_LOCAL
        return f;
#else
        return f * WG + lane_id;
#endif
    }
    __device__ __forceinline__ float2 node_pos(unsigned i) const {
        unsigned f = TN + 4u * i;
        return v2(tab[ti(f)], tab[ti(f + 1u)]);
    }
    __device__ __forceinline__ float2 node_vel(unsigned i) const {
        unsigned f = TN + 4u * i + 2u;
        return v2(tab[ti(f)], tab[ti(f + 1u)]);
    }
    __device__ __forceinline__ void set_pos(unsigned i, float2 v) {
        unsigned f = TN + 4u * i;
        tab[ti(f)] = v.x;
        tab[ti(f + 1u)] = v.y;
    }
    __device__ __forceinline__ void set_vel(unsigned i, float2 v) {
        unsigned f = TN + 4u * i + 2u;
        tab[ti(f)] = v.x;
        tab[ti(f + 1u)] = v.y;
    }
    __device__ __forceinline__ vec3 body_get(unsigned j, unsigned s) const {
        unsigned f = TB + 6u * j + 3u * s;
        return v3(tab[ti(f)], tab[ti(f + 1u)], tab[ti(f + 2u)]);
    }
    __device__ __forceinline__ void body_set(unsigned j, unsigned s, vec3 v) {
        unsigned f = TB + 6u * j + 3u * s;
        tab[ti(f)] = v.x;
        tab[ti(f + 1u)] = v.y;
        tab[ti(f + 2u)] = v.z;
    }
    __device__ __forceinline__ void body_add(unsigned j, unsigned s, vec3 v) {
        unsigned f = TB + 6u * j + 3u * s;
        tab[ti(f)] += v.x;
        tab[ti(f + 1u)] += v.y;
        tab[ti(f + 2u)] += v.z;
    }
    __device__ __forceinline__ float bone_field(unsigned j, unsigned f) const {
        return bone_data[bone_base + (j * BONE_FIELDS + f) * TILE];
    }
    // How deep node `i` sits in the mud, as a share of the deepest mud.
    __device__ __forceinline__ float mud_sink(unsigned i) const {
        float2 pn = node_pos(i);
        float2 g = terrain(pn.x);
        float secant = sqrtf(1.0f + g.y * g.y);
        float dry = (pn.y - g.x) / secant - node_radius(i);
        return clampf(-dry, 0.0f, p.mud) * (1.0f / MUD_FULL_DEPTH);
    }
    __device__ __forceinline__ float node_radius(unsigned i) const {
        if (i == 0u) {
            return bone_field(0u, 8u);
        }
        return bone_field(i - 1u, 5u);
    }
    __device__ __forceinline__ float node_fric(unsigned i) const {
        if (i == 0u) {
            return bone_field(0u, 2u);
        }
        return bone_field(i - 1u, 6u);
    }
    __device__ __forceinline__ unsigned body_of(unsigned i) const { return i == 0u ? 0u : i - 1u; }
    __device__ __forceinline__ unsigned parent_of(unsigned j) const { return body_of(pivot[j]); }
    __device__ __forceinline__ vec3 axis_of(unsigned j) const { return v3(1.0f, arm[j].y, -arm[j].x); }

    // Height and slope of the ground; flat ground is exactly (0, 0).
    __device__ float2 terrain(float x) const {
        if (!rough) {
            return v2(0.0f, 0.0f);
        }
        float t0 = x * (1.0f / 1.1f) + phase_q;
        float u0 = t0 - floorf(t0);
        float w0 = u0 * (1.0f - u0);
        float t1 = x * (1.0f / 0.43f) + 0.3f + phase_q;
        float u1 = t1 - floorf(t1);
        float w1 = u1 * (1.0f - u1);
        float height = 0.65f * 16.0f * w0 * w0 + 0.35f * 16.0f * w1 * w1;
        float slope = 0.65f * 32.0f * w0 * (1.0f - 2.0f * u0) * (1.0f / 1.1f)
            + 0.35f * 32.0f * w1 * (1.0f - 2.0f * u1) * (1.0f / 0.43f);
        height = amplitude * height + p.slope * x;
        slope = amplitude * slope + p.slope;
        if (p.gaps > 0.0f) {
            float spacing = 2.0f + 4.0f * p.gaps;
            float center = spacing * 0.5f;
            float t = x / spacing;
            float r = x - floorf(t) * spacing;
            float distance = fabsf(r - center);
            float half_w = 0.5f * p.gaps;
            float run = fmaxf(fminf(GAP_RUN, half_w), 1e-6f);
            float ramp = clampf((half_w - distance) / run, 0.0f, 1.0f);
            float factor = ramp;
            if (distance <= half_w - run) {
                factor = 1.0f;
            } else if (distance >= half_w) {
                factor = 0.0f;
            }
            bool on_ramp = distance > half_w - run && distance < half_w;
            float side = -1.0f;
            if (r < center) {
                side = 1.0f;
            }
            height -= GAP_DEPTH * factor;
            if (on_ramp) {
                slope -= GAP_DEPTH * (side / run);
            }
        }
        if (p.hurdles > 0.0f) {
            float spacing = HURDLE_SPACING;
            float center = spacing * 0.5f;
            float t = x / spacing;
            float r = x - floorf(t) * spacing;
            float distance = fabsf(r - center);
            float half_w = 0.5f * HURDLE_TOP;
            float run = HURDLE_RUN;
            float ramp = clampf((half_w + run - distance) / run, 0.0f, 1.0f);
            float factor = ramp;
            if (distance <= half_w) {
                factor = 1.0f;
            } else if (distance >= half_w + run) {
                factor = 0.0f;
            }
            bool on_ramp = distance > half_w && distance < half_w + run;
            float side = -1.0f;
            if (r < center) {
                side = 1.0f;
            }
            height += p.hurdles * factor;
            if (on_ramp) {
                slope += p.hurdles * (side / run);
            }
        }
        return v2(height, slope);
    }

    // Body j's velocity-product acceleration, from its pivot's velocity in
    // the table.
    __device__ __forceinline__ vec3 cvel_of(unsigned j) const {
        float2 vp = node_vel(pivot[j]);
        float w = om[j];
        vec3 sv = v3(w, vp.x + w * arm[j].y, vp.y - w * arm[j].x);
        return crm(sv, axis_of(j)) * qd[j];
    }

    // Absolute angles and rates, node velocities and, with `positions`, node
    // positions, from the state.
    __device__ __forceinline__ void kinematics(bool positions) {
        if (positions) {
            set_pos(0u, x0);
        }
        set_vel(0u, v0);
        UNROLL
        for (unsigned j = 0u; j < MAXB; j++) {
            if (j >= nb) { break; }
            float t = q[0];
            float w = qd[0];
            if (j > 0u) {
                vec3 up = body_get(parent_of(j), 0u);
                t = up.x + q[j];
                w = up.y + qd[j];
            }
            om[j] = w;
            body_set(j, 0u, v3(t, w, 0.0f));
            float sn = sinf(t);
            float cs = cosf(t);
            float l = len[j];
            float2 pv0 = node_vel(pivot[j]);
            set_vel(j + 1u, v2(pv0.x - l * w * sn, pv0.y + l * w * cs));
            if (positions) {
                float2 pp0 = node_pos(pivot[j]);
                set_pos(j + 1u, v2(pp0.x + l * cs, pp0.y + l * sn));
            }
        }
    }

    __device__ __forceinline__ float2 momentum() const {
        float2 m = v2(0.0f, 0.0f);
        UNROLL
        for (unsigned i = 0u; i < MAXN; i++) {
            if (i >= nn) { break; }
            m += node_vel(i) * mass[i];
        }
        return m;
    }

    // Semi-implicit Euler on the joint coordinates with the accelerations in
    // `acc` and `qdd`, and air drag.
    __device__ __forceinline__ void integrate_state(float air) {
        vec3 a0 = acc[0];
        float2 head = v2(a0.y - qd[0] * v0.y, a0.z + qd[0] * v0.x);
        v0 = v2((v0.x + head.x * DT) * air, (v0.y + head.y * DT) * air);
        qd[0] = (qd[0] + a0.x * DT) * air;
        x0 = v2(x0.x + v0.x * DT, x0.y + v0.y * DT);
        q[0] += qd[0] * DT;
        UNROLL
        for (unsigned j = 1u; j < MAXB; j++) {
            if (j >= nb) { break; }
            qd[j] = (qd[j] + qdd[j] * DT) * air;
            q[j] += qd[j] * DT;
        }
    }

    // The accelerations that the spatial forces in the table's first scratch
    // vector of every bone cause, through the articulated inertias of this
    // step. Leaves them in the second scratch vector and the joint
    // accelerations in `dq`.
    __device__ __forceinline__ void response() {
        UNROLL
        for (unsigned i = 1u; i < MAXB; i++) {
            unsigned j = MAXB - i;
            if (j >= nb) { continue; }
            vec3 pj = body_get(j, 0u);
            float t = -sdot(axis_of(j), pj);
            dq[j] = t;
            body_add(parent_of(j), 0u, pj + uvec[j] * (t * dinv[j]));
        }
        body_set(0u, 1u, -sym_mul(root0, root1, body_get(0u, 0u)));
        dq[0] = 0.0f;
        UNROLL
        for (unsigned j = 1u; j < MAXB; j++) {
            if (j >= nb) { break; }
            vec3 a = body_get(parent_of(j), 1u);
            float t = (dq[j] - sdot(uvec[j], a)) * dinv[j];
            dq[j] = t;
            body_set(j, 1u, a + axis_of(j) * t);
        }
    }

    __device__ __forceinline__ void clear_forces() {
        UNROLL
        for (unsigned j = 0u; j < MAXB; j++) {
            if (j >= nb) { break; }
            body_set(j, 0u, v3(0.0f, 0.0f, 0.0f));
        }
    }

    // Adds the accelerations that contact forces `f` (normal, friction per
    // contact slot) cause.
    __device__ __forceinline__ void apply_contacts(const float* f) {
        clear_forces();
        UNROLL_C
        for (unsigned ci = 0u; ci < MAXC; ci++) {
            if (!c_on[ci]) { continue; }
            body_add(body_of(c_node[ci]), 0u, -(c_dn[ci] * f[2u * ci] + c_dt[ci] * f[2u * ci + 1u]));
        }
        response();
        UNROLL
        for (unsigned j = 0u; j < MAXB; j++) {
            if (j >= nb) { break; }
            acc[j] += body_get(j, 1u);
            qdd[j] += dq[j];
        }
    }

    // Projected Gauss-Seidel on the contact impulses.
    __device__ __forceinline__ void pgs(unsigned sweeps) {
        UNROLL_C
        for (unsigned row = 0u; row < MAXR; row++) {
            if (!c_on[row / 2u]) { continue; }
            float v = c_vn[row / 2u];
            if ((row & 1u) == 1u) {
                v = c_vt[row / 2u];
            }
            UNROLL_C
            for (unsigned j = 0u; j < MAXR; j++) {
                if (!c_on[j / 2u]) { continue; }
                v += kmat[tri(row, j)] * lambda[j];
            }
            vrow[row] = v;
        }
        for (unsigned sweep = 0u; sweep < sweeps; sweep++) {
            UNROLL_C
            for (unsigned ci = 0u; ci < MAXC; ci++) {
                if (!c_on[ci]) { continue; }
                unsigned rn = 2u * ci;
                unsigned rt = rn + 1u;
                float normal = fmaxf(lambda[rn] + (c_goal[ci] - vrow[rn]) * (1.0f / kmat[tri(rn, rn)]), 0.0f);
                float dn = normal - lambda[rn];
                lambda[rn] = normal;
                UNROLL_C
                for (unsigned j = 0u; j < MAXR; j++) {
                    if (!c_on[j / 2u]) { continue; }
                    vrow[j] += kmat[tri(j, rn)] * dn;
                }
                float bound = c_mu[ci] * lambda[rn];
                // Friction may not do positive work: it only opposes
                // a = start speed + end speed without its own force, and only
                // up to |a| / k.
                float stiff = kmat[tri(rt, rt)];
                float a = c_vs[ci] + vrow[rt] - stiff * lambda[rt];
                float stop = fabsf(a) / stiff;
                float cap = fminf(bound, stop);
                float friction = clampf(lambda[rt] - vrow[rt] * (1.0f / stiff), a > 0.0f ? -cap : 0.0f, a > 0.0f ? 0.0f : cap);
                float dt_ = friction - lambda[rt];
                lambda[rt] = friction;
                UNROLL_C
                for (unsigned j = 0u; j < MAXR; j++) {
                    if (!c_on[j / 2u]) { continue; }
                    vrow[j] += kmat[tri(j, rt)] * dt_;
                }
            }
        }
    }

    // Removes friction that would still do positive work after the solves:
    // each contact's friction only opposes the mean of its node's speed before
    // and after the step, up to the size that stops the node. Two sweeps.
    __device__ __forceinline__ void clean_friction() {
        for (unsigned sweep = 0u; sweep < 2u; sweep++) {
            UNROLL_C
            for (unsigned ci = 0u; ci < MAXC; ci++) {
                if (!c_on[ci]) { continue; }
                unsigned rn = 2u * ci;
                unsigned rt = rn + 1u;
                float stiff = kmat[tri(rt, rt)];
                float a = c_vs[ci] + vrow[rt] - stiff * lambda[rt];
                float cap = fminf(c_mu[ci] * lambda[rn], fabsf(a) / stiff);
                float friction = clampf(lambda[rt], a > 0.0f ? -cap : 0.0f, a > 0.0f ? 0.0f : cap);
                float dt_ = friction - lambda[rt];
                lambda[rt] = friction;
                UNROLL_C
                for (unsigned j = 0u; j < MAXR; j++) {
                    if (!c_on[j / 2u]) { continue; }
                    vrow[j] += kmat[tri(j, rt)] * dt_;
                }
            }
        }
    }

    // Ground contacts at velocity level, solved together (physics2's contact
    // section). Returns `outside` (the step's impulse from gravity and wind)
    // plus the ground's impulse.
    __device__ float2 contacts(float2 origin, float2 before, float2 outside) {
        nc = 0u;
        unsigned candidates = 0u;
        UNROLL
        for (unsigned i = 0u; i < MAXN; i++) {
            reach[i] = 1e30f;
            if (i >= nn) { continue; }
            float2 pn = node_pos(i);
            float2 g = terrain(pn.x);
            float secant = sqrtf(1.0f + g.y * g.y);
            float2 normal = v2(-g.y / secant, 1.0f / secant);
            float mud = p.ground > 0.0f ? p.mud : 0.0f;
            float gap = (pn.y - g.x) / secant - node_radius(i) + mud;
            unsigned body = body_of(i);
            float2 r = pn - origin;
            float2 v = node_vel(i);
            float w = om[body];
            vec3 dn = force_at(r, normal);
            float vn_free = v.x * normal.x + v.y * normal.y
                + DT * (sdot(dn, acc[body]) + w * (-v.y * normal.x + v.x * normal.y));
            float depth = gap + DT * vn_free;
            if (depth <= 0.0f) {
                reach[i] = depth;
                candidates += 1u;
            }
        }
        UNROLL_C
        for (unsigned ci = 0u; ci < MAXC; ci++) {
            c_on[ci] = false;
        }
        if (candidates == 0u) {
            return outside;
        }
        // The deepest MAXC nodes take part, in node order.
        bool chosen[MAXN];
        UNROLL
        for (unsigned i = 0u; i < MAXN; i++) {
            chosen[i] = reach[i] < 1e29f;
            if (!chosen[i] || candidates <= MAXC) { continue; }
            unsigned deeper = 0u;
            UNROLL
            for (unsigned j = 0u; j < MAXN; j++) {
                if (reach[j] < reach[i] || (reach[j] == reach[i] && j < i)) {
                    deeper += 1u;
                }
            }
            chosen[i] = deeper < MAXC;
        }
        UNROLL_C
        for (unsigned ci = 0u; ci < MAXC; ci++) {
            if (ci >= candidates) { break; }
            unsigned node = 0u;
            unsigned seen = 0u;
            UNROLL
            for (unsigned i = 0u; i < MAXN; i++) {
                if (chosen[i]) {
                    if (seen == ci) {
                        node = i;
                    }
                    seen += 1u;
                }
            }
            float2 pn = node_pos(node);
            float2 g = terrain(pn.x);
            float secant = sqrtf(1.0f + g.y * g.y);
            float2 normal = v2(-g.y / secant, 1.0f / secant);
            float2 tangent = v2(normal.y, -normal.x);
            float mud = p.ground > 0.0f ? p.mud : 0.0f;
            float dry = (pn.y - g.x) / secant - node_radius(node);
            float gap = dry + mud;
            float sink = clampf(-dry, 0.0f, mud) * (1.0f / MUD_FULL_DEPTH);
            unsigned body = body_of(node);
            float2 r = pn - origin;
            float2 v = node_vel(node);
            // The body's acceleration and turning rate, from the table.
            vec3 a = body_get(body, 0u);
            float w = body_get(body, 1u).x;
            vec3 dn = force_at(r, normal);
            vec3 dtan = force_at(r, tangent);
            c_on[ci] = true;
            c_node[ci] = node;
            c_dn[ci] = dn;
            c_dt[ci] = dtan;
            c_vn[ci] = v.x * normal.x + v.y * normal.y
                + DT * (sdot(dn, a) + w * (-v.y * normal.x + v.x * normal.y));
            c_vt[ci] = v.x * tangent.x + v.y * tangent.y
                + DT * (sdot(dtan, a) + w * (-v.y * tangent.x + v.x * tangent.y));
            c_vs[ci] = v.x * tangent.x + v.y * tangent.y;
            c_goal[ci] = gap >= 0.0f ? -gap * RATE : -gap * PUSH_OUT * RATE;
            c_mu[ci] = node_fric(node) * p.friction * (1.0f + MUD_GRIP * sink) * (1.0f + MUD_NORMAL * sink);
            nc += 1u;
        }
        // Contact-space matrix, column by column from each unit force's
        // response; symmetric, the lower triangle kept.
        UNROLL_C
        for (unsigned col = 0u; col < MAXR; col++) {
            unsigned ci = col / 2u;
            if (!c_on[ci]) { continue; }
            vec3 fdir = c_dn[ci];
            if ((col & 1u) == 1u) {
                fdir = c_dt[ci];
            }
            clear_forces();
            body_add(body_of(c_node[ci]), 0u, -fdir);
            response();
            UNROLL_C
            for (unsigned row = 0u; row < MAXR; row++) {
                unsigned ri = row / 2u;
                if (row < col || !c_on[ri]) { continue; }
                vec3 rdir = c_dn[ri];
                if ((row & 1u) == 1u) {
                    rdir = c_dt[ri];
                }
                kmat[tri(row, col)] = DT * sdot(rdir, body_get(body_of(c_node[ri]), 1u));
            }
        }
        UNROLL_C
        for (unsigned ci = 0u; ci < MAXC; ci++) {
            float ln = 0.0f;
            float lt = 0.0f;
            if (WARM && c_on[ci]) {
                unsigned node = c_node[ci];
                Record rec = records_in[record_base + node];
                float2 w = node == 0u ? rec.c : rec.b;
                ln = w.x;
                lt = clampf(w.y, -c_mu[ci] * ln, c_mu[ci] * ln);
            }
            lambda[2u * ci] = ln;
            lambda[2u * ci + 1u] = lt;
        }
        pgs(PGS_SWEEPS);
        apply_contacts(lambda);
        // Plant against the end pose: take the step, measure each contact's
        // velocity in the end pose (after the momentum balance), and solve
        // again with the difference.
        for (unsigned round = 0u; round < PLANT_ROUNDS; round++) {
            float2 sx0 = x0;
            float2 sv0 = v0;
            float sq[MAXB];
            float sqd[MAXB];
            UNROLL
            for (unsigned j = 0u; j < MAXB; j++) {
                sq[j] = q[j];
                sqd[j] = qd[j];
            }
            integrate_state(1.0f);
            kinematics(false);
            float2 ground = v2(0.0f, 0.0f);
            UNROLL_C
            for (unsigned ci = 0u; ci < MAXC; ci++) {
                if (!c_on[ci]) { continue; }
                ground += v2(
                    (lambda[2u * ci] * c_dn[ci].y + lambda[2u * ci + 1u] * c_dt[ci].y) * DT,
                    (lambda[2u * ci] * c_dn[ci].z + lambda[2u * ci + 1u] * c_dt[ci].z) * DT);
            }
            float2 after = momentum();
            float2 shift = v2(
                (before.x + outside.x + ground.x - after.x) * inv_mass,
                (before.y + outside.y + ground.y - after.y) * inv_mass);
            UNROLL_C
            for (unsigned ci = 0u; ci < MAXC; ci++) {
                if (!c_on[ci]) { continue; }
                unsigned rn = 2u * ci;
                unsigned rt = rn + 1u;
                float2 v = node_vel(c_node[ci]) + shift;
                float2 normal = v2(c_dn[ci].y, c_dn[ci].z);
                c_vn[ci] += v.x * normal.x + v.y * normal.y - vrow[rn];
                c_vt[ci] += v.x * normal.y + v.y * -normal.x - vrow[rt];
            }
            x0 = sx0;
            v0 = sv0;
            UNROLL
            for (unsigned j = 0u; j < MAXB; j++) {
                q[j] = sq[j];
                qd[j] = sqd[j];
            }
            UNROLL_C
            for (unsigned j = 0u; j < MAXR; j++) {
                old[j] = lambda[j];
            }
            pgs(PLANT_SWEEPS);
            clean_friction();
            UNROLL_C
            for (unsigned j = 0u; j < MAXR; j++) {
                old[j] = lambda[j] - old[j];
            }
            apply_contacts(old);
        }
        float2 impulse = outside;
        UNROLL_C
        for (unsigned ci = 0u; ci < MAXC; ci++) {
            if (!c_on[ci]) { continue; }
            impulse += v2(
                (lambda[2u * ci] * c_dn[ci].y + lambda[2u * ci + 1u] * c_dt[ci].y) * DT,
                (lambda[2u * ci] * c_dn[ci].z + lambda[2u * ci + 1u] * c_dt[ci].z) * DT);
        }
        return impulse;
    }
};

#if RECORD
// The recorded frame is `p.stride` float2 long: the node positions (STRIDE of
// them), then one (energy, force) pair per muscle, then one (normal, friction)
// contact force per node. The forces of a frame are those of the step that led
// to it; the energy is the state it starts from.
__device__ __forceinline__ void record_extras(
    float2* __restrict__ frames, unsigned base, unsigned muscle_count, unsigned tile_x, unsigned tl,
    const float* __restrict__ muscle_data, const Record* __restrict__ records, unsigned record_base, unsigned nn) {
    for (unsigned k = 0u; k < muscle_count; k++) {
        const unsigned field = tile_x + k * MUSCLE_FIELDS * TILE + tl;
        frames[base + STRIDE + k] = v2(muscle_data[field + 14u * TILE], muscle_data[field + 11u * TILE]);
    }
    UNROLL
    for (unsigned i = 0u; i < MAXN; i++) {
        if (i >= nn) { break; }
        float2 w = i == 0u ? records[record_base].c : records[record_base + i].b;
        frames[base + STRIDE + muscle_count + i] = w;
    }
}
#endif

extern "C" __global__ void LAUNCH_BOUNDS advance(
    Record* __restrict__ records,
    float* __restrict__ muscle_data,
    const float* __restrict__ bone_data,
    const Params p,
    Result* __restrict__ results,
    // (nodes, bones, muscles, quake seed) per creature.
    const uint4* __restrict__ creature_info,
    // (muscle base, bone base, unused, unused) per 32-creature tile.
    const uint4* __restrict__ tile_info
#if RECORD
    ,
    // Node positions as [creature][frame][node], with the node stride.
    float2* __restrict__ frames
#endif
    ) {
#if !TAB_LOCAL
    __shared__ float tab_shared[TABF * WG];
#endif
    const unsigned lane = threadIdx.x;
    const unsigned creature = blockIdx.x * WG + lane;
    if (creature >= p.count) {
        return;
    }
    Lane L(p);
    L.records_in = records;
    L.bone_data = bone_data;
#if !TAB_LOCAL
    L.tab = tab_shared;
#endif
    L.lane_id = lane;
    const uint4 info = creature_info[creature];
    L.nn = info.x;
    L.nb = info.y;
    const unsigned nn = L.nn;
    const unsigned nb = L.nb;
    const unsigned muscle_count = info.z;
    const bool still = p.quake <= 0.0f || p.ground <= 0.0f;
    L.phase_q = still ? 0.0f : quake_phase(info.w);
    L.amplitude = p.terrain + (still ? 0.0f : p.quake * quake_scale(info.w));
    L.rough = L.amplitude != 0.0f || p.slope != 0.0f || p.gaps > 0.0f || p.hurdles > 0.0f;
    const uint4 tile = tile_info[creature / TILE];
    const unsigned tl = creature % TILE;
    L.bone_base = tile.y + tl;
    L.record_base = creature * STRIDE;
    const unsigned record_base = L.record_base;

    Record head = records[record_base];
    L.x0 = head.a;
    L.v0 = head.b;
    L.mass[0] = L.bone_field(0u, 7u);
    L.total_mass = L.mass[0];
    UNROLL
    for (unsigned j = 0u; j < MAXB; j++) {
        if (j >= nb) { break; }
        L.pivot[j] = __float_as_uint(L.bone_field(j, 0u));
        L.len[j] = L.bone_field(j, 1u);
        L.mass[j + 1u] = L.bone_field(j, 4u);
        L.total_mass += L.mass[j + 1u];
        Record r = records[record_base + j + 1u];
        L.q[j] = r.a.x;
        L.qd[j] = r.a.y;
    }
    const float total_mass = L.total_mass;
    L.inv_mass = 1.0f / total_mass;
    const float inv_mass = L.inv_mass;
    const float inv_nodes = 1.0f / (float)nn;
    const float muscle_scale = L.bone_field(0u, 3u);
    L.kinematics(true);

    Result metrics = {0.0f, 0.0f, 1e20f, -1e20f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (p.tick > SETTLE) {
        metrics = results[creature];
    }
    float head_shake = metrics.head_shake;
    const bool grounded = p.ground > 0.0f;

#if RECORD
    // The result at the end of the trial; the body keeps moving after it.
    Result kept = metrics;
    bool done = metrics.fall_time > 0.0f || metrics.screened > 0.0f;
#endif
    for (unsigned s = 0u; s < p.steps; s++) {
#if !RECORD
        if (metrics.fall_time > 0.0f || metrics.screened > 0.0f) {
            break;
        }
#endif
        const unsigned tick = p.tick + s;
#if RECORD
        if (!done && (metrics.fall_time > 0.0f || metrics.screened > 0.0f)) {
            kept = metrics;
            kept.head_shake = head_shake;
            done = true;
        }
        {
            const unsigned frame = (creature * (p.total_steps + 1u) + tick) * p.stride;
            UNROLL
            for (unsigned j = 0u; j < MAXN; j++) {
                if (j >= nn) { break; }
                frames[frame + j] = L.node_pos(j);
            }
            record_extras(frames, frame, muscle_count, tile.x, tl, muscle_data, records, record_base, nn);
            if (s == 0u && p.tick == SETTLE) {
                for (unsigned t = 0u; t < SETTLE; t++) {
                    const unsigned before_frame = (creature * (p.total_steps + 1u) + t) * p.stride;
                    UNROLL
                    for (unsigned j = 0u; j < MAXN; j++) {
                        if (j >= nn) { break; }
                        frames[before_frame + j] = L.node_pos(j);
                    }
                    record_extras(frames, before_frame, muscle_count, tile.x, tl, muscle_data, records, record_base, nn);
                }
            }
        }
#endif
        if (tick < SETTLE) {
            continue;
        }
        const unsigned step = tick - SETTLE;
        const float t_now = (float)step * DT;
        const float2 head_before = L.v0;
        const float2 origin = L.x0;
        const float2 before = L.momentum();
        // For the first-law check in flight.
        float energy_start = 0.0f;
        float energy_scale = 0.0f;
        float mass_x_start = 0.0f;
        UNROLL
        for (unsigned i = 0u; i < MAXN; i++) {
            if (i >= nn) { break; }
            float2 pi_ = L.node_pos(i);
            float2 vi = L.node_vel(i);
            float kinetic = 0.5f * L.mass[i] * (vi.x * vi.x + vi.y * vi.y);
            float potential = L.mass[i] * p.gravity * pi_.y;
            energy_start += kinetic + potential;
            energy_scale += kinetic + fabsf(potential);
            mass_x_start += pi_.x * L.mass[i];
        }
        float muscle_start = 0.0f;
        // Body inertias (true, for the velocity products), pivot arms, and
        // the velocity-product forces.
        UNROLL
        for (unsigned j = 0u; j < MAXB; j++) {
            if (j >= nb) { break; }
            float m = L.mass[j + 1u];
            float2 r = L.node_pos(j + 1u) - origin;
            L.i0[j] = v3(m * (r.x * r.x + r.y * r.y), -m * r.y, m * r.x);
            L.i1[j] = v3(m, 0.0f, m);
            L.arm[j] = L.node_pos(L.pivot[j]) - origin;
        }
        L.i1[0] += v3(L.mass[0], 0.0f, L.mass[0]);
        UNROLL
        for (unsigned j = 0u; j < MAXB; j++) {
            if (j >= nb) { break; }
            float2 vp = L.node_vel(L.pivot[j]);
            float w = L.om[j];
            vec3 sv = v3(w, vp.x + w * L.arm[j].y, vp.y - w * L.arm[j].x);
            L.bias[j] = crf(sv, sym_mul(L.i0[j], L.i1[j], sv));
        }
        // Gravity, wind and mud drag on every node.
        float mud_impulse = 0.0f;
        UNROLL
        for (unsigned i = 0u; i < MAXN; i++) {
            if (i >= nn) { break; }
            float m = L.mass[i];
            float fx = p.wind * m;
            if (p.mud > 0.0f && p.ground > 0.0f) {
                float drag = -m * MUD_DRAG * L.mud_sink(i) * L.node_vel(i).x;
                fx += drag;
                mud_impulse += drag * DT;
            }
            L.bias[L.body_of(i)] -= force_at(L.node_pos(i) - origin, v2(fx, -p.gravity * m));
        }
        // Air drag on every bone, at its midpoint, limited so a step of drag
        // never more than halves the speed it acts on.
        float2 air_impulse = v2(0.0f, 0.0f);
        UNROLL
        for (unsigned j = 0u; j < MAXB; j++) {
            if (j >= nb) { break; }
            const unsigned pv = L.pivot[j];
            const float2 mid = (L.node_pos(pv) + L.node_pos(j + 1u)) * 0.5f;
            const float2 v = (L.node_vel(pv) + L.node_vel(j + 1u)) * 0.5f;
            const float speed = sqrtf(v.x * v.x + v.y * v.y);
            const float width = L.node_radius(pv) + L.node_radius(j + 1u);
            const float strength = fmaxf(fminf(AIR_DRAG * L.len[j] * width * speed, 0.5f * L.mass[j + 1u] * RATE), 0.0f);
            const float2 f = v * -strength;
            L.bias[j] -= force_at(mid - origin, f);
            air_impulse = air_impulse + f * DT;
        }
        // Muscles pull between points on two bones; the forces collect in
        // the table.
        L.clear_forces();
        for (unsigned k = 0u; k < muscle_count; k++) {
            const unsigned field = tile.x + k * MUSCLE_FIELDS * TILE + tl;
            const unsigned packed = __float_as_uint(muscle_data[field]);
            const unsigned a0 = packed & 63u;
            const unsigned a1 = (packed >> 6u) & 63u;
            const unsigned b0 = (packed >> 12u) & 63u;
            const unsigned b1 = (packed >> 18u) & 63u;
            const float anchor_a = muscle_data[field + TILE];
            const float anchor_b = muscle_data[field + 2u * TILE];
            const float amp = muscle_data[field + 3u * TILE];
            const float hill = muscle_data[field + 4u * TILE];
            const float inv_period = muscle_data[field + 5u * TILE];
            const float phase = muscle_data[field + 6u * TILE];
            const float duty = muscle_data[field + 7u * TILE];
            const float stiffness = muscle_data[field + 8u * TILE];
            const float inv_duty = muscle_data[field + 9u * TILE];
            const float inv_complement = muscle_data[field + 10u * TILE];
            const float offset = muscle_data[field + 13u * TILE];
            const float energy = muscle_data[field + 14u * TILE];
            const float strength = muscle_data[field + 15u * TILE] * muscle_scale;
            const float cap = MAX_MUSCLE_FORCE * strength;
            const float inv_capacity = 1.0f / (MUSCLE_CAPACITY * p.muscle_energy * strength);
            const float2 pa0 = L.node_pos(a0);
            const float2 pa1 = L.node_pos(a1);
            const float2 pb0 = L.node_pos(b0);
            const float2 pb1 = L.node_pos(b1);
            const float2 va0 = L.node_vel(a0);
            const float2 va1 = L.node_vel(a1);
            const float2 vb0 = L.node_vel(b0);
            const float2 vb1 = L.node_vel(b1);
            const float2 pa = v2(pa0.x + (pa1.x - pa0.x) * anchor_a, pa0.y + (pa1.y - pa0.y) * anchor_a);
            const float2 va = v2(va0.x + (va1.x - va0.x) * anchor_a, va0.y + (va1.y - va0.y) * anchor_a);
            const float2 pb = v2(pb0.x + (pb1.x - pb0.x) * anchor_b, pb0.y + (pb1.y - pb0.y) * anchor_b);
            const float2 vb = v2(vb0.x + (vb1.x - vb0.x) * anchor_b, vb0.y + (vb1.y - vb0.y) * anchor_b);
            const float2 d = pb - pa;
            const float length_m = fmaxf(sqrtf(d.x * d.x + d.y * d.y), 1e-6f);
            const float inverse = 1.0f / length_m;
            const float2 dir = v2(d.x * inverse, d.y * inverse);
            const float relative = (vb.x - va.x) * dir.x + (vb.y - va.y) * dir.y;
            float target_speed = 0.0f;
            if (t_now > 0.0f) {
                target_speed = amp * (wave(t_now, inv_period, phase, offset, duty, inv_duty, inv_complement)
                    - wave(fmaxf(t_now - DT, 0.0f), inv_period, phase, offset, duty, inv_duty, inv_complement)) * RATE;
            }
            float drive = fmaxf(-target_speed * stiffness * 0.25f, 0.0f) * energy;
            if (hill > 0.0f) {
                drive *= clampf(1.0f + relative * hill, 0.0f, 1.0f);
            }
            const float magnitude = clampf(drive + relative * 0.15f, -cap, cap);
            const float work = fminf(drive, cap) * fmaxf(-relative, 0.0f) * DT;
            muscle_data[field + 14u * TILE] = clampf(
                energy - work * inv_capacity
                    + MUSCLE_RECOVERY * p.muscle_recovery * DT * (1.0f - energy),
                0.0f,
                1.0f);
            muscle_data[field + 11u * TILE] = magnitude;
            muscle_start += magnitude * length_m;
            const float pull = magnitude;
            const float2 f = dir * pull;
            L.body_add(a1 - 1u, 0u, force_at(pa - origin, f));
            L.body_add(b1 - 1u, 0u, -force_at(pb - origin, f));
        }
        UNROLL
        for (unsigned j = 0u; j < MAXB; j++) {
            if (j >= nb) { break; }
            L.bias[j] -= L.body_get(j, 0u);
        }
        // Spin cap: rotational drag past the cap, implicit, toward rest.
        UNROLL
        for (unsigned j = 0u; j < MAXB; j++) {
            if (j >= nb) { break; }
            float w = L.om[j];
            if (fabsf(w) > SPIN_CAP) {
                float l = L.len[j];
                float drag = SPIN_HARDNESS * L.mass[j + 1u] * l * l * (fabsf(w) * INV_SPIN_CAP - 1.0f);
                L.i0[j].x += drag;
                L.bias[j].x += drag * RATE * w;
            }
        }
        // Articulated-body pass, children first, with joint damping and the
        // joint limits' inelastic stops implicit in each joint's inertia. A
        // bone hands its articulated inertia to its parent through selects
        // over the bones before it.
        UNROLL
        for (unsigned i = 1u; i < MAXB; i++) {
            const unsigned j = MAXB - i;
            if (j >= nb) { continue; }
            const vec3 axis = L.axis_of(j);
            const vec3 uv = sym_mul(L.i0[j], L.i1[j], axis);
            float d = sdot(axis, uv);
            float tau = 0.0f;
            const float c = d * INV_JOINT_DAMPING;
            tau -= c * L.qd[j];
            d += c * DT;
            const float qj = L.q[j];
            const float qdj = L.qd[j];
            const float lo = L.bone_field(j, 2u);
            const float hi = L.bone_field(j, 3u);
            const float predicted = qj + DT * qdj;
            const bool upper = predicted > hi;
            if (upper || predicted < lo) {
                const float room = upper ? hi - qj : lo - qj;
                const bool past = (room < 0.0f) == upper;
                const float goal = (past ? room * PUSH_OUT : room) * RATE;
                if ((qdj > goal) == upper) {
                    const float cl = LIMIT_HARDNESS * d * RATE;
                    tau -= cl * (qdj - goal);
                    d += cl * DT;
                }
            }
            const float u = tau - sdot(axis, L.bias[j]);
            const float di = 1.0f / d;
            L.uvec[j] = uv;
            L.dinv[j] = di;
            L.uu[j] = u;
            const float k = -di;
            const vec3 a0 = L.i0[j] + v3(k * uv.x * uv.x, k * uv.x * uv.y, k * uv.x * uv.z);
            const vec3 a1 = L.i1[j] + v3(k * uv.y * uv.y, k * uv.y * uv.z, k * uv.z * uv.z);
            const vec3 pa = L.bias[j] + sym_mul(a0, a1, L.cvel_of(j)) + uv * (u * di);
            const unsigned pr = L.parent_of(j);
            UNROLL
            for (unsigned up = 0u; up < MAXB; up++) {
                if (up >= j) { break; }
                if (pr == up) {
                    L.i0[up] += a0;
                    L.i1[up] += a1;
                    L.bias[up] += pa;
                }
            }
        }
        // The neck body floats freely: the root's inverse inertia.
        {
            const vec3 r0 = L.i0[0];
            const vec3 r1 = L.i1[0];
            const float c00 = r1.x * r1.z - r1.y * r1.y;
            const float c01 = r0.z * r1.y - r0.y * r1.z;
            const float c02 = r0.y * r1.y - r0.z * r1.x;
            const float inv_det = 1.0f / (r0.x * c00 + r0.y * c01 + r0.z * c02);
            const float c11 = r0.x * r1.z - r0.z * r0.z;
            const float c12 = r0.y * r0.z - r0.x * r1.y;
            const float c22 = r0.x * r1.x - r0.y * r0.y;
            L.root0 = v3(c00 * inv_det, c01 * inv_det, c02 * inv_det);
            L.root1 = v3(c11 * inv_det, c12 * inv_det, c22 * inv_det);
        }
        L.acc[0] = -sym_mul(L.root0, L.root1, L.bias[0]);
        L.qdd[0] = 0.0f;
        // Parents first; each bone leaves its acceleration and turning rate
        // in the table for its children and the contacts.
        L.body_set(0u, 0u, L.acc[0]);
        L.body_set(0u, 1u, v3(L.om[0], 0.0f, 0.0f));
        UNROLL
        for (unsigned j = 1u; j < MAXB; j++) {
            if (j >= nb) { break; }
            const vec3 a = L.body_get(L.parent_of(j), 0u) + L.cvel_of(j);
            L.qdd[j] = (L.uu[j] - sdot(L.uvec[j], a)) * L.dinv[j];
            L.acc[j] = a + L.axis_of(j) * L.qdd[j];
            L.body_set(j, 0u, L.acc[j]);
            L.body_set(j, 1u, v3(L.om[j], 0.0f, 0.0f));
        }

        float2 impulse = v2(p.wind * total_mass * DT + mud_impulse, -p.gravity * total_mass * DT) + air_impulse;
        L.nc = 0u;
        if (grounded) {
            impulse = L.contacts(origin, before, impulse);
        }
        // Each node's contact force starts the next step's solve.
        if (WARM && grounded) {
            UNROLL
            for (unsigned i = 0u; i < MAXN; i++) {
                if (i >= nn) { break; }
                float2 w = v2(0.0f, 0.0f);
                UNROLL_C
                for (unsigned ci = 0u; ci < MAXC; ci++) {
                    if (L.c_on[ci] && L.c_node[ci] == i) {
                        w = v2(L.lambda[2u * ci], L.lambda[2u * ci + 1u]);
                    }
                }
                if (i == 0u) {
                    records[record_base].c = w;
                } else {
                    records[record_base + i].b = w;
                }
            }
        }
        L.integrate_state(p.air);
        L.kinematics(true);
        // Momentum balance.
        const float2 after = L.momentum();
        const float2 expected = v2((before.x + impulse.x) * p.air, (before.y + impulse.y) * p.air);
        const float2 shift = v2((expected.x - after.x) * inv_mass, (expected.y - after.y) * inv_mass);
        L.v0 += shift;
        UNROLL
        for (unsigned i = 0u; i < MAXN; i++) {
            if (i >= nn) { break; }
            L.set_vel(i, L.node_vel(i) + shift);
        }
        // First law in flight.
        if (L.nc == 0u) {
            float muscle_end = 0.0f;
                        for (unsigned k = 0u; k < muscle_count; k++) {
                const unsigned field = tile.x + k * MUSCLE_FIELDS * TILE + tl;
                const unsigned packed = __float_as_uint(muscle_data[field]);
                const float anchor_a = muscle_data[field + TILE];
                const float anchor_b = muscle_data[field + 2u * TILE];
                const float2 pa0 = L.node_pos(packed & 63u);
                const float2 pa1 = L.node_pos((packed >> 6u) & 63u);
                const float2 pb0 = L.node_pos((packed >> 12u) & 63u);
                const float2 pb1 = L.node_pos((packed >> 18u) & 63u);
                const float2 pa = v2(pa0.x + (pa1.x - pa0.x) * anchor_a, pa0.y + (pa1.y - pa0.y) * anchor_a);
                const float2 pb = v2(pb0.x + (pb1.x - pb0.x) * anchor_b, pb0.y + (pb1.y - pb0.y) * anchor_b);
                const float2 d = pb - pa;
                const float length_m = sqrtf(d.x * d.x + d.y * d.y);
                muscle_end += muscle_data[field + 11u * TILE] * length_m;
            }
            float energy_end = 0.0f;
            float mass_x_end = 0.0f;
            UNROLL
            for (unsigned i = 0u; i < MAXN; i++) {
                if (i >= nn) { break; }
                float2 pi_ = L.node_pos(i);
                float2 vi = L.node_vel(i);
                energy_end += 0.5f * L.mass[i] * (vi.x * vi.x + vi.y * vi.y) + L.mass[i] * p.gravity * pi_.y;
                mass_x_end += pi_.x * L.mass[i];
            }
            const float work = (muscle_start - muscle_end) + p.wind * (mass_x_end - mass_x_start);
            const float excess = energy_end - energy_start - work - (1e-4f + 1e-5f * energy_scale);
            if (excess > 0.0f) {
                const float2 center = v2(expected.x * inv_mass, expected.y * inv_mass);
                float internal = 0.0f;
                UNROLL
                for (unsigned i = 0u; i < MAXN; i++) {
                    if (i >= nn) { break; }
                    float2 vi = L.node_vel(i);
                    float x = vi.x - center.x;
                    float y = vi.y - center.y;
                    internal += 0.5f * L.mass[i] * (x * x + y * y);
                }
                float keep = 0.0f;
                if (internal > 0.0f) {
                    keep = sqrtf(fmaxf(1.0f - excess / internal, 0.0f));
                }
                L.qd[0] *= keep;
                UNROLL
                for (unsigned j = 1u; j < MAXB; j++) {
                    if (j >= nb) { break; }
                    L.qd[j] *= keep;
                }
                L.v0 = v2(center.x + keep * (L.v0.x - center.x), center.y + keep * (L.v0.y - center.y));
                UNROLL
                for (unsigned i = 0u; i < MAXN; i++) {
                    if (i >= nn) { break; }
                    float2 vi = L.node_vel(i);
                    L.set_vel(i, v2(center.x + keep * (vi.x - center.x), center.y + keep * (vi.y - center.y)));
                }
                // The absolute rates scale with the joint rates.
                UNROLL
                for (unsigned j = 0u; j < MAXB; j++) {
                    if (j >= nb) { break; }
                    L.om[j] *= keep;
                }
            }
        }
        L.q[0] = wrap_angle(L.q[0]);

        // Metrics, falls and the screen (physics2::run).
        bool failed = false;
        float center_y = 0.0f;
        float touching = 0.0f;
        float low = 1e20f;
        float high = -1e20f;
        unsigned contact_lo = __float_as_uint(metrics.contact_lo);
        unsigned contact_hi = __float_as_uint(metrics.contact_hi);
        unsigned lift_lo = __float_as_uint(metrics.lift_lo);
        unsigned lift_hi = __float_as_uint(metrics.lift_hi);
        unsigned now_lo = 0u;
        unsigned now_hi = 0u;
        float com_x = 0.0f;
        UNROLL
        for (unsigned i = 0u; i < MAXN; i++) {
            if (i >= nn) { break; }
            const float2 pi_ = L.node_pos(i);
            const float x = pi_.x;
            const float y = pi_.y;
            if (!(fabsf(x) <= 1e6f) || !(fabsf(y) <= 1e6f)) {
                failed = true;
            }
            const float r = L.node_radius(i);
            center_y += y;
            low = fminf(low, y - r);
            high = fmaxf(high, y + r);
            com_x += x * L.mass[i];
            if (grounded) {
                const float2 g = L.terrain(x);
                const float floor_y = g.x + r * sqrtf(1.0f + g.y * g.y);
                if (y <= floor_y + CONTACT_SLACK) {
                    touching += 1.0f;
                    if (i < 32u) {
                        contact_lo |= 1u << (i & 31u);
                        now_lo |= 1u << (i & 31u);
                    } else {
                        contact_hi |= 1u << ((i - 32u) & 31u);
                        now_hi |= 1u << ((i - 32u) & 31u);
                    }
                } else if (y > floor_y + LIFT_CLEARANCE) {
                    if (i < 32u) {
                        lift_lo |= contact_lo & (1u << (i & 31u));
                    } else {
                        lift_hi |= contact_hi & (1u << ((i - 32u) & 31u));
                    }
                }
            }
        }
        com_x *= inv_mass;
        metrics.contact_lo = __uint_as_float(contact_lo);
        metrics.contact_hi = __uint_as_float(contact_hi);
        metrics.lift_lo = __uint_as_float(lift_lo);
        metrics.lift_hi = __uint_as_float(lift_hi);
        center_y *= inv_nodes;
        const unsigned down_lo = now_lo & ~__float_as_uint(metrics.ground_lo);
        const unsigned down_hi = now_hi & ~__float_as_uint(metrics.ground_hi);
        metrics.ground_lo = __uint_as_float(now_lo);
        metrics.ground_hi = __uint_as_float(now_hi);
        if ((down_lo | down_hi) != 0u && step > 0u) {
            const float next = t_now + DT;
            for (unsigned k = 0u; k < muscle_count; k++) {
                const unsigned field = tile.x + k * MUSCLE_FIELDS * TILE + tl;
                const unsigned packed = __float_as_uint(muscle_data[field]);
                const unsigned end = (packed >> 24u) & 7u;
                if (end == NO_SENSOR) {
                    continue;
                }
                const unsigned sensor = (packed >> (6u * end)) & 63u;
                const unsigned touched = sensor < 32u ? (down_lo >> sensor) & 1u : (down_hi >> (sensor - 32u)) & 1u;
                if (touched == 1u) {
                    const float clock = next * muscle_data[field + 5u * TILE] + muscle_data[field + 6u * TILE];
                    const float x = muscle_data[field + 12u * TILE] - clock;
                    muscle_data[field + 13u * TILE] = x - floorf(x);
                }
            }
        }
        if (t_now >= HEAD_SHAKE_WINDOW) {
            const float2 dv = L.v0 - head_before;
            const float accel = sqrtf(dv.x * dv.x + dv.y * dv.y) * RATE;
            head_shake += (accel - head_shake) * fminf(1.0f / (HEAD_SHAKE_WINDOW * RATE), 1.0f);
        }
        metrics.head_shake = head_shake;
        bool broken = false;
        UNROLL
        for (unsigned j = 1u; j < MAXB; j++) {
            if (j >= nb) { break; }
            if (L.q[j] < L.bone_field(j, 2u) - JOINT_BREAK || L.q[j] > L.bone_field(j, 3u) + JOINT_BREAK) {
                broken = true;
            }
        }
        const bool fell = L.x0.y < L.node_pos(1u).y || broken || head_shake > HEAD_SHAKE_LIMIT || failed;
        bool ended = false;
        if (fell) {
            metrics.fall_time = t_now + DT;
            metrics.fitness = failed ? -1e20f : com_x;
            if (tick <= p.screen_tick) {
                metrics.screen_x = metrics.fitness;
            }
            ended = true;
        }
        metrics.ground_contact += touching;
        metrics.height_sum += high - low;
        metrics.vertical_oscillation = fminf(metrics.vertical_oscillation, center_y);
        metrics.gait_frequency = fmaxf(metrics.gait_frequency, center_y);
        if (step == 0u) {
            metrics.previous_center_y = center_y;
            metrics.vertical_extremum = center_y;
            metrics.vertical_trend = 0.0f;
            metrics.gait_turns = 0.0f;
        } else if (step % SAMPLE == 0u) {
            const float delta = center_y - metrics.previous_center_y;
            if (metrics.vertical_trend == 0.0f) {
                if (fabsf(delta) > 0.0005f) {
                    metrics.vertical_trend = delta > 0.0f ? 1.0f : -1.0f;
                    metrics.vertical_extremum = center_y;
                }
            } else if (metrics.vertical_trend > 0.0f) {
                if (center_y > metrics.vertical_extremum) {
                    metrics.vertical_extremum = center_y;
                } else if (metrics.vertical_extremum - center_y > 0.005f) {
                    metrics.gait_turns += 1.0f;
                    metrics.vertical_trend = -1.0f;
                    metrics.vertical_extremum = center_y;
                }
            } else {
                if (center_y < metrics.vertical_extremum) {
                    metrics.vertical_extremum = center_y;
                } else if (center_y - metrics.vertical_extremum > 0.005f) {
                    metrics.gait_turns += 1.0f;
                    metrics.vertical_trend = 1.0f;
                    metrics.vertical_extremum = center_y;
                }
            }
            metrics.previous_center_y = center_y;
        }
        if (tick == p.screen_tick && !ended) {
            metrics.screen_x = com_x;
            if (com_x < p.screen_bar) {
                metrics.screened = t_now + DT;
                metrics.fitness = com_x;
                ended = true;
            }
        }
        if (ended) {
            metrics.vertical_oscillation = fmaxf(metrics.gait_frequency - metrics.vertical_oscillation, 0.0f);
            metrics.gait_frequency = metrics.gait_turns * 0.5f / (t_now + DT);
        } else if (tick + 1u == p.total_steps) {
            metrics.fitness = com_x;
            metrics.vertical_oscillation = fmaxf(metrics.gait_frequency - metrics.vertical_oscillation, 0.0f);
            metrics.gait_frequency = metrics.gait_turns * 0.5f / fmaxf((float)(step + 1u) * DT, DT);
        }
    }
    metrics.head_shake = head_shake;
#if RECORD
    if (!done) {
        kept = metrics;
    }
    results[creature] = kept;
    if (p.tick + p.steps >= p.total_steps) {
        const unsigned frame = (creature * (p.total_steps + 1u) + p.total_steps) * p.stride;
        UNROLL
        for (unsigned j = 0u; j < MAXN; j++) {
            if (j >= nn) { break; }
            frames[frame + j] = L.node_pos(j);
        }
        record_extras(frames, frame, muscle_count, tile.x, tl, muscle_data, records, record_base, nn);
    }
#else
    results[creature] = metrics;
#endif
    {
        Record h = records[record_base];
        h.a = L.x0;
        h.b = L.v0;
        h.d = v2(0.0f, 0.0f);
        records[record_base] = h;
    }
    UNROLL
    for (unsigned j = 0u; j < MAXB; j++) {
        if (j >= nb) { break; }
        Record r = records[record_base + j + 1u];
        r.a = v2(L.q[j], L.qd[j]);
        r.c = v2(0.0f, 0.0f);
        r.d = v2(0.0f, 0.0f);
        records[record_base + j + 1u] = r;
    }
}
