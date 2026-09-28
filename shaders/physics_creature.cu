// CUDA port of shaders/physics_creature.wgsl for the NVIDIA-only backend
// (src/cuda_engine.rs, EVOLUTION_CUDA=1). It computes the same physics in the
// same order as the WGSL kernel, section for section, so the two stay easy to
// compare; read the WGSL file for the reasons behind each step.
//
// creature_kernel::cuda_source prepends the constants that
// creature_kernel::base_source writes into the WGSL text (WG, MAXN, STRIDE,
// SHAREDLEN, the physics limits, the fidelity) as #define lines, plus
// NODE_BOUND, BONE_BOUND and UNROLL, which give the node and bone loops the
// same bounds the WGSL kernel gets for this capacity.
//
// Differences from the WGSL text. None of them changes the physics:
// - Params arrives as a kernel argument instead of a uniform buffer, and
//   terrain() takes the ground settings it reads as arguments.
// - The WGSL contact() and collide() helpers are never called; they are left out.
// - The rebuild's children-first pass runs j downward over the compile-time
//   bone bound and skips j >= bone_count, in place of i upward with
//   j = bone_count - 1 - i. It visits the same bones in the same order, and
//   every bone index is a constant after unrolling, so the per-bone arrays can
//   stay in registers.
// - Every float literal carries an f suffix, so no expression runs in double
//   precision.

struct Node {
    float2 pos;
    float2 vel;
    float radius;
    float friction;
    float mass;
    float failed;
};
struct Muscle {
    float anchor_a;
    float anchor_b;
    // Waveform amplitude, min(long - short, speed-limited amplitude), from packing.
    float amplitude;
    float long_length;
    float inv_period;
    float phase;
    float duty;
    float stiffness;
    float inv_duty;
    float inv_complement;
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
// Fields: packed endpoints, 10 muscle genes, sensor endpoint, reset phase,
// rhythm offset (state), energy (state).
#define MUSCLE_FIELDS 15u
#define NO_SENSOR 255u
#define TIRED_DRIVE 0.0f
#define BONE_FIELDS 9u
#define MAXB (MAXN - 1u)
#define DT (1.0f / RATE)

// WGSL vector operations on float2.
__device__ __forceinline__ float2 v2(float x, float y) { return make_float2(x, y); }
__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return v2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return v2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 operator*(float2 a, float s) { return v2(a.x * s, a.y * s); }
__device__ __forceinline__ float2 operator*(float s, float2 a) { return v2(s * a.x, s * a.y); }
__device__ __forceinline__ float2 operator/(float2 a, float s) { return v2(a.x / s, a.y / s); }
__device__ __forceinline__ void operator+=(float2& a, float2 b) { a = a + b; }
__device__ __forceinline__ void operator-=(float2& a, float2 b) { a = a - b; }
__device__ __forceinline__ void operator*=(float2& a, float s) { a = a * s; }
__device__ __forceinline__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
__device__ __forceinline__ float length(float2 a) { return sqrtf(dot(a, a)); }
// WGSL mix(a, b, t) = a * (1 - t) + b * t.
__device__ __forceinline__ float2 mix(float2 a, float2 b, float t) { return a * (1.0f - t) + b * t; }
__device__ __forceinline__ float fract(float x) { return x - floorf(x); }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
// WGSL pack2x16snorm and unpack2x16snorm, by the formulas of the WGSL spec.
__device__ __forceinline__ unsigned pack2x16snorm(float2 v) {
    int x = (int)floorf(0.5f + 32767.0f * fminf(1.0f, fmaxf(-1.0f, v.x)));
    int y = (int)floorf(0.5f + 32767.0f * fminf(1.0f, fmaxf(-1.0f, v.y)));
    return ((unsigned)x & 0xffffu) | (((unsigned)y & 0xffffu) << 16u);
}
__device__ __forceinline__ float2 unpack2x16snorm(unsigned v) {
    float x = (float)(short)(v & 0xffffu);
    float y = (float)(short)(v >> 16u);
    return v2(fmaxf(x / 32767.0f, -1.0f), fmaxf(y / 32767.0f, -1.0f));
}

// Index helpers, as in the WGSL kernel.
// Node state index of node `j` of this lane's creature.
__device__ __forceinline__ unsigned node_k(unsigned j, unsigned lane) { return j * WG + lane; }
// Node number of node state index `k`.
__device__ __forceinline__ unsigned node_of(unsigned k) { return k / WG; }
// Bone `j`'s endpoints and joint reference node as node state indices, from
// the per-lane words packed at load time.
__device__ __forceinline__ unsigned bone_a(unsigned packed_a) { return packed_a & 0xffffu; }
__device__ __forceinline__ unsigned bone_q(unsigned packed_a) { return packed_a >> 16u; }
__device__ __forceinline__ unsigned bone_b(unsigned packed_b) { return packed_b; }
// Endpoint `e` (0..3: a0, a1, b0, b1) of muscle `j`, as a node number.
__device__ __forceinline__ unsigned muscle_node(unsigned packed, unsigned e) {
    return (packed >> (8u * e)) & 0xffu;
}

// Earthquake per-creature bump phase and amplitude scale (physics::quake_*).
__device__ __forceinline__ float quake_phase(unsigned seed) {
    return (float)(seed & 0xffffu) * (1.0f / 65536.0f);
}
__device__ __forceinline__ float quake_scale(unsigned seed) {
    return 0.6f + (float)((seed >> 16u) & 0xffffu) * (0.8f / 65536.0f);
}
// Height and slope of the rough ground (physics::terrain plus the earthquake
// phase, slope, gaps and hurdles). The WGSL function reads the slope, gap and
// hurdle settings from the uniform; here the caller passes them.
__device__ float2 terrain(float x, float phase, float amplitude, float p_slope, float p_gaps, float p_hurdles) {
    float t0 = x * (1.0f / 1.1f) + phase;
    float u0 = t0 - floorf(t0);
    float w0 = u0 * (1.0f - u0);
    float t1 = x * (1.0f / 0.43f) + 0.3f + phase;
    float u1 = t1 - floorf(t1);
    float w1 = u1 * (1.0f - u1);
    float height = 0.65f * 16.0f * w0 * w0 + 0.35f * 16.0f * w1 * w1;
    float slope = 0.65f * 32.0f * w0 * (1.0f - 2.0f * u0) * (1.0f / 1.1f)
        + 0.35f * 32.0f * w1 * (1.0f - 2.0f * u1) * (1.0f / 0.43f);
    height = amplitude * height + p_slope * x;
    slope = amplitude * slope + p_slope;
    if (p_gaps > 0.0f) {
        float spacing = 2.0f + 4.0f * p_gaps;
        float center = spacing * 0.5f;
        float t = x / spacing;
        float r = x - floorf(t) * spacing;
        float distance = fabsf(r - center);
        float half_width = 0.5f * p_gaps;
        float run = fmaxf(fminf(GAP_RUN, half_width), 1e-6f);
        float ramp = clampf((half_width - distance) / run, 0.0f, 1.0f);
        float factor = ramp;
        if (distance <= half_width - run) {
            factor = 1.0f;
        } else if (distance >= half_width) {
            factor = 0.0f;
        }
        bool on_ramp = distance > half_width - run && distance < half_width;
        float side = -1.0f;
        if (r < center) {
            side = 1.0f;
        }
        height -= GAP_DEPTH * factor;
        if (on_ramp) {
            slope -= GAP_DEPTH * (side / run);
        }
    }
    if (p_hurdles > 0.0f) {
        float spacing = HURDLE_SPACING;
        float center = spacing * 0.5f;
        float t = x / spacing;
        float r = x - floorf(t) * spacing;
        float distance = fabsf(r - center);
        float half_width = 0.5f * HURDLE_TOP;
        float run = HURDLE_RUN;
        float ramp = clampf((half_width + run - distance) / run, 0.0f, 1.0f);
        float factor = ramp;
        if (distance <= half_width) {
            factor = 1.0f;
        } else if (distance >= half_width + run) {
            factor = 0.0f;
        }
        bool on_ramp = distance > half_width && distance < half_width + run;
        float side = -1.0f;
        if (r < center) {
            side = 1.0f;
        }
        height += p_hurdles * factor;
        if (on_ramp) {
            slope += p_hurdles * (side / run);
        }
    }
    return v2(height, slope);
}

// Whether node `n` is set in a 64-node bitmask split into two words.
__device__ __forceinline__ bool in_mask(unsigned n, unsigned lo, unsigned hi) {
    return (n < 32u ? (lo >> n) & 1u : (hi >> (n - 32u)) & 1u) == 1u;
}
// Bone share of node a after weighting nodes on the ground by their grip.
__device__ __forceinline__ float stance_share(float share, float fa, float fb) {
    return share * fb / (share * fb + (1.0f - share) * fa);
}
// The default polynomial cosine of pi * x (creature_kernel::apply_fast_cos).
__device__ __forceinline__ float fast_cos_pi(float x) {
    float y = (x - 0.5f) * 3.14159265359f;
    float z = y * y;
    float p = fmaf(z, -2.50521084e-8f, 2.75573192e-6f);
    p = fmaf(z, p, -1.98412698e-4f);
    p = fmaf(z, p, 8.33333377e-3f);
    p = fmaf(z, p, -1.66666672e-1f);
    p = fmaf(z, p, 1.0f);
    return -y * p;
}
__device__ __forceinline__ float limited_muscle_length(const Muscle& m, float time) {
    float amplitude = m.amplitude;
    float phase = fract(time * m.inv_period + m.phase);
    float wave;
    if (phase < m.duty) {
#if EXACT_COS
        wave = 0.5f + 0.5f * cosf(3.14159265359f * phase * m.inv_duty);
#else
        wave = 0.5f + 0.5f * fast_cos_pi(phase * m.inv_duty);
#endif
    } else {
#if EXACT_COS
        wave = 0.5f - 0.5f * cosf(3.14159265359f * (phase - m.duty) * m.inv_complement);
#else
        wave = 0.5f - 0.5f * fast_cos_pi((phase - m.duty) * m.inv_complement);
#endif
    }
    return m.long_length - amplitude * (1.0f - wave);
}
// Small-angle rotation shared with the CPU engine.
__device__ __forceinline__ float2 rotate_small(float2 v, float angle) {
    float a2 = angle * angle;
    float c = 1.0f - a2 * (0.5f - a2 * (1.0f / 24.0f));
    float s = angle * (1.0f - a2 * ((1.0f / 6.0f) - a2 * (1.0f / 120.0f)));
    return v2(v.x * c - v.y * s, v.x * s + v.y * c);
}
__device__ __forceinline__ float2 limit_speed(float2 velocity) {
    float speed = length(velocity);
    if (speed > MAX_NODE_SPEED) {
        return velocity * (MAX_NODE_SPEED / speed);
    }
    return velocity;
}

extern "C" __global__ void LAUNCH_BOUNDS advance(
    Node* __restrict__ nodes,
    // Muscle genes plus per-muscle state (rhythm offset and energy).
    float* __restrict__ muscle_data,
    const float* __restrict__ bone_data,
    const Params p,
    Result* __restrict__ results,
    // (nodes, bones, muscles, quake seed) per creature.
    const uint4* __restrict__ creature_info,
    // (muscle base, bone base, unused, unused) per 32-creature tile.
    const uint4* __restrict__ tile_info) {
    __shared__ float2 pos[SHAREDLEN];
    __shared__ float2 vel[SHAREDLEN];
    __shared__ float2 old[SHAREDLEN];
    const unsigned lane = threadIdx.x;
    const unsigned creature = blockIdx.x * WG + lane;
    if (creature >= p.count) {
        return;
    }
    const uint4 info = creature_info[creature];
    const unsigned body_nodes = info.x;
    const unsigned bone_count = info.y;
    const unsigned muscle_count = info.z;
    const unsigned quake_seed = info.w;
    const float quake_phase_value = p.quake <= 0.0f ? 0.0f : quake_phase(quake_seed);
    const float terrain_amplitude = p.terrain + p.quake * quake_scale(quake_seed);
    const bool rough = terrain_amplitude > 0.0f || p.slope != 0.0f || p.gaps > 0.0f || p.hurdles > 0.0f;
    const uint4 tile = tile_info[creature / TILE];
    const unsigned tl = creature % TILE;
    const unsigned base = creature * STRIDE;

    float radius[MAXN];
    float mass[MAXN];
    float friction[MAXN];
    float failed[MAXN];
    float inv_mass[MAXN];
    float total_mass = 0.0f;
    UNROLL
    for (unsigned j = 0u; j < NODE_BOUND; j++) {
        if (j >= body_nodes) { break; }
        const Node n = nodes[base + j];
        const unsigned k = node_k(j, lane);
        pos[k] = n.pos;
        vel[k] = n.vel;
        radius[j] = n.radius;
        mass[j] = n.mass;
        friction[j] = n.friction;
        failed[j] = n.failed;
        inv_mass[j] = 1.0f / n.mass;
        total_mass += n.mass;
    }
    const float inv_total_mass = 1.0f / total_mass;
    const float inv_nodes = 1.0f / (float)body_nodes;
    // Bone endpoints as node state indices, plus the per-endpoint constants the
    // solver reads on every pass.
    unsigned bone_ka[MAXB];
    unsigned bone_kb[MAXB];
    float bone_rest[MAXB];
    float bone_sa[MAXB];
    unsigned bone_center[MAXB];
    float bone_cos_half[MAXB];
    float bone_break[MAXB];
    UNROLL
    for (unsigned j = 0u; j < BONE_BOUND; j++) {
        if (j >= bone_count) { break; }
        const unsigned field = tile.y + j * BONE_FIELDS * TILE + tl;
        const unsigned packed = __float_as_uint(bone_data[field]);
        const unsigned na = packed & 0xffu;
        const unsigned nb = (packed >> 8u) & 0xffu;
        const Node node_a = nodes[base + na];
        const Node node_b = nodes[base + nb];
        const unsigned nq = (packed >> 16u) & 0xffu;
        bone_ka[j] = (na * WG + lane) | ((nq * WG + lane) << 16u);
        bone_kb[j] = nb * WG + lane;
        bone_rest[j] = bone_data[field + TILE];
        const float inverse_a = 1.0f / node_a.mass;
        const float inverse_b = 1.0f / node_b.mass;
        const float inverse_sum = inverse_a + inverse_b;
        bone_sa[j] = inverse_a / inverse_sum;
        bone_center[j] = pack2x16snorm(v2(bone_data[field + 2u * TILE], bone_data[field + 3u * TILE]));
        bone_cos_half[j] = bone_data[field + 4u * TILE];
        bone_break[j] = bone_cos_half[j] * JOINT_BREAK_COS - bone_data[field + 5u * TILE] * JOINT_BREAK_SIN;
    }
    Result metrics = {0.0f, 0.0f, 1e20f, -1e20f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (p.tick > 0u) {
        metrics = results[creature];
    }

    for (unsigned s = 0u; s < p.steps; s++) {
        // A fall or the screen ends the trial.
        if (metrics.fall_time > 0.0f || metrics.screened > 0.0f) {
            break;
        }
        const unsigned tick = p.tick + s;
        // The head's velocity before this step, for the head shaking limit.
        const float2 head_start = vel[node_k(0u, lane)];
        if (tick == SETTLE) {
            float avg = 0.0f;
            float mass_sum = 0.0f;
            float low = 1e20f;
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                const unsigned k = node_k(j, lane);
                avg += pos[k].x * mass[j];
                mass_sum += mass[j];
            }
            const float shift_x = avg * inv_total_mass;
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                const unsigned k = node_k(j, lane);
                float floor_y = 0.0f;
                if (rough) {
                    floor_y = terrain(pos[k].x - shift_x, quake_phase_value, terrain_amplitude, p.slope, p.gaps, p.hurdles).x;
                }
                low = fminf(low, pos[k].y - radius[j] - floor_y);
            }
            const float2 shift = v2(shift_x, low);
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                const unsigned k = node_k(j, lane);
                const float2 shifted_start = pos[k] - shift;
                pos[k] = shifted_start;
                vel[k] = v2(0.0f, 0.0f);
            }
        }
        UNROLL
        for (unsigned j = 0u; j < NODE_BOUND; j++) {
            if (j >= body_nodes) { break; }
            const unsigned k = node_k(j, lane);
            // Muscle forces add up here.
            old[k] = v2(0.0f, 0.0f);
        }

        const float time = (float)(max(tick, SETTLE) - SETTLE) * DT;
        for (unsigned j = 0u; j < muscle_count; j++) {
            const unsigned field = tile.x + j * MUSCLE_FIELDS * TILE + tl;
            const unsigned packed = __float_as_uint(muscle_data[field]);
            const float offset = muscle_data[field + 13u * TILE];
            float energy = muscle_data[field + 14u * TILE];
            if (tick == SETTLE) {
                energy = 1.0f;
            }
            Muscle m;
            m.anchor_a = muscle_data[field + 1u * TILE];
            m.anchor_b = muscle_data[field + 2u * TILE];
            m.amplitude = muscle_data[field + 3u * TILE];
            m.long_length = muscle_data[field + 4u * TILE];
            m.inv_period = muscle_data[field + 5u * TILE];
            m.phase = muscle_data[field + 6u * TILE] + offset;
            m.duty = muscle_data[field + 7u * TILE];
            m.stiffness = muscle_data[field + 8u * TILE];
            m.inv_duty = muscle_data[field + 9u * TILE];
            m.inv_complement = muscle_data[field + 10u * TILE];
            const unsigned ka0 = node_k(muscle_node(packed, 0u), lane);
            const unsigned ka1 = node_k(muscle_node(packed, 1u), lane);
            const unsigned kb0 = node_k(muscle_node(packed, 2u), lane);
            const unsigned kb1 = node_k(muscle_node(packed, 3u), lane);
            const float2 position_a = mix(pos[ka0], pos[ka1], m.anchor_a);
            const float2 position_b = mix(pos[kb0], pos[kb1], m.anchor_b);
            const float2 velocity_a = mix(vel[ka0], vel[ka1], m.anchor_a);
            const float2 velocity_b = mix(vel[kb0], vel[kb1], m.anchor_b);
            const float2 d = position_b - position_a;
            const float2 dir = d * (1.0f / fmaxf(length(d), 1e-6f));
            const float relative = dot(velocity_b - velocity_a, dir);
            float target_speed = 0.0f;
            if (tick > SETTLE) {
                target_speed = (limited_muscle_length(m, time)
                    - limited_muscle_length(m, fmaxf(time - DT, 0.0f))) * RATE;
            }
            // A muscle only pulls, and its drive scales with its stored energy.
            const float drive = fmaxf(-target_speed * m.stiffness * 0.25f, 0.0f) * (TIRED_DRIVE + (1.0f - TIRED_DRIVE) * energy);
            float magnitude = clampf(drive + relative * 0.15f, -MAX_MUSCLE_FORCE, MAX_MUSCLE_FORCE);
            if (metrics.fall_time > 0.0f) {
                magnitude = 0.0f;
            }
            if (tick >= SETTLE) {
                const float work = fabsf(magnitude * relative) * DT;
                energy = clampf(
                    energy - work / (MUSCLE_CAPACITY * p.muscle_energy)
                        + MUSCLE_RECOVERY * p.muscle_recovery * DT * (1.0f - energy),
                    0.0f,
                    1.0f);
            }
            muscle_data[field + 14u * TILE] = energy;
            const float2 push = dir * magnitude;
            const unsigned a0 = muscle_node(packed, 0u);
            const unsigned a1 = muscle_node(packed, 1u);
            const unsigned b0 = muscle_node(packed, 2u);
            const unsigned b1 = muscle_node(packed, 3u);
            for (unsigned e = 0u; e < 4u; e++) {
                const unsigned node = muscle_node(packed, e);
                if ((e >= 1u && node == a0) || (e >= 2u && node == a1) || (e == 3u && node == b0)) {
                    continue;
                }
                float weight = 0.0f;
                if (a0 == node) { weight += 1.0f - m.anchor_a; }
                if (a1 == node) { weight += m.anchor_a; }
                if (b0 == node) { weight -= 1.0f - m.anchor_b; }
                if (b1 == node) { weight -= m.anchor_b; }
                const float2 f = push * weight;
                const unsigned k = node_k(node, lane);
                old[k] += f;
            }
        }

        float gravity = 0.0f;
        float wind = 0.0f;
        if (tick >= SETTLE) {
            gravity = p.gravity;
            wind = p.wind;
        }
        // Integrate velocities first; the speed cap's removed momentum is
        // spread back over the whole body.
        float2 capped_momentum = v2(0.0f, 0.0f);
        UNROLL
        for (unsigned j = 0u; j < NODE_BOUND; j++) {
            if (j >= body_nodes) { break; }
            const unsigned k = node_k(j, lane);
            if (failed[j] < 0.5f) {
                const float2 free = (vel[k] + (old[k] * inv_mass[j] - v2(0.0f, gravity) + v2(wind, 0.0f)) * DT) * p.air;
                const float2 capped = limit_speed(free);
                capped_momentum += (free - capped) * mass[j];
                old[k] = capped;
            }
        }
        const float2 cap_correction = capped_momentum * inv_total_mass;
        UNROLL
        for (unsigned j = 0u; j < NODE_BOUND; j++) {
            if (j >= body_nodes) { break; }
            const unsigned k = node_k(j, lane);
            const float2 start = pos[k];
            if (failed[j] < 0.5f) {
                Node n;
                n.pos = pos[k];
                n.vel = old[k] + cap_correction;
                n.pos += n.vel * DT;
                if (!(fabsf(n.pos.x) < 1e6f && fabsf(n.pos.y) < 1e6f)
                    || !(fabsf(n.vel.x) < 1e6f && fabsf(n.vel.y) < 1e6f)) {
                    failed[j] = 1.0f;
                    n.pos = v2(0.0f, 0.0f);
                    n.vel = v2(0.0f, 0.0f);
                }
                pos[k] = n.pos;
            }
            old[k] = start;
            // Predicted height and lowest allowed center height for this step.
            vel[k] = v2(pos[k].y, radius[j]);
        }

        const bool grounded = tick >= SETTLE && p.ground > 0.0f;
        // Planted feet: one pass pushes each node out of the ground, marks
        // its stance, and adds it to the center of mass.
        unsigned stance_lo = 0u;
        unsigned stance_hi = 0u;
        float com_x_before = 0.0f;
        UNROLL
        for (unsigned j = 0u; j < NODE_BOUND; j++) {
            if (j >= body_nodes) { break; }
            const unsigned k = node_k(j, lane);
            if (grounded) {
                if (rough) {
                    const float2 ground = terrain(pos[k].x, quake_phase_value, terrain_amplitude, p.slope, p.gaps, p.hurdles);
                    const float secant = sqrtf(1.0f + ground.y * ground.y);
                    const float floor_y = ground.x + radius[j] * secant - p.mud;
                    vel[k].y = floor_y;
                    const float gap = floor_y - pos[k].y;
                    if (gap > 0.0f) {
                        const float depth = gap / (secant * secant);
                        pos[k] += v2(-ground.y, 1.0f) * depth;
                    }
                } else {
                    vel[k].y = radius[j] - p.mud;
                    pos[k].y = fmaxf(pos[k].y, vel[k].y);
                }
                if (pos[k].y <= vel[k].y + 1e-4f) {
                    if (j < 32u) {
                        stance_lo |= 1u << j;
                    } else {
                        stance_hi |= 1u << (j - 32u);
                    }
                }
            }
            com_x_before += pos[k].x * mass[j];
        }
        com_x_before *= inv_total_mass;
        for (unsigned iteration = 0u; iteration < BONE_SOLVE_ITERATIONS; iteration++) {
            UNROLL
            for (unsigned j = 0u; j < BONE_BOUND; j++) {
                if (j >= bone_count) { break; }
                const unsigned ka = bone_a(bone_ka[j]);
                const unsigned kb = bone_b(bone_kb[j]);
                const unsigned na = node_of(ka);
                const unsigned nb = node_of(kb);
                const float fa = in_mask(na, stance_lo, stance_hi) ? 1.0f + STANCE_GRIP * friction[na] * p.friction : 1.0f;
                const float fb = in_mask(nb, stance_lo, stance_hi) ? 1.0f + STANCE_GRIP * friction[nb] * p.friction : 1.0f;
                const float share = stance_share(bone_sa[j], fa, fb);
                const float2 old_a = pos[ka];
                const float2 old_b = pos[kb];
                const float2 delta = old_b - old_a;
                const float raw_distance = length(delta);
                const float distance = fmaxf(raw_distance, 1e-6f);
                const float error = distance - bone_rest[j];
                const float2 correction = raw_distance > 1e-6f ? delta * (error / distance) : v2(error, 0.0f);
                float2 new_a = old_a + correction * share;
                float2 new_b = old_b - correction * (1.0f - share);
                if (grounded) {
                    // A clamp is the ground pushing back (vel.x holds the
                    // node's height without ground).
                    vel[ka].x -= fmaxf(vel[ka].y - new_a.y, 0.0f);
                    vel[kb].x -= fmaxf(vel[kb].y - new_b.y, 0.0f);
                    new_a.y = fmaxf(new_a.y, vel[ka].y);
                    new_b.y = fmaxf(new_b.y, vel[kb].y);
                }
                pos[ka] = new_a;
                pos[kb] = new_b;
            }
        }

        // Joint ranges.
        UNROLL
        for (unsigned j = 0u; j < BONE_BOUND; j++) {
            if (j >= bone_count) { break; }
            const float cos_half = bone_cos_half[j];
            if (cos_half <= -1.0f) { continue; }
            const unsigned kn = bone_a(bone_ka[j]);
            const unsigned kc = bone_b(bone_kb[j]);
            const unsigned kq = bone_q(bone_ka[j]);
            const float2 pivot = pos[kn];
            const float2 u = pos[kq] - pivot;
            const float2 v = pos[kc] - pivot;
            const float norm = sqrtf(dot(u, u) * dot(v, v));
            if (norm < 1e-12f) { continue; }
            const float2 relative = v2(dot(u, v), u.x * v.y - u.y * v.x) * (1.0f / norm);
            const float2 center = unpack2x16snorm(bone_center[j]);
            // Angle from the middle of the range, as a unit vector.
            const float2 z = v2(
                relative.x * center.x + relative.y * center.y,
                relative.y * center.x - relative.x * center.y);
            if (z.x >= cos_half) { continue; }
            const unsigned field = tile.y + j * BONE_FIELDS * TILE + tl;
            const float sin_half = bone_data[field + 5u * TILE];
            const float side = z.y >= 0.0f ? 1.0f : -1.0f;
            const float sin_excess = fabsf(z.y) * cos_half - z.x * sin_half;
            const float cos_excess = z.x * cos_half + fabsf(z.y) * sin_half;
            float excess = 1.0f;
            if (cos_excess > 0.0f) {
                excess = fminf(sin_excess * (1.0f + sin_excess * sin_excess * (1.0f / 6.0f)), 1.0f);
            }
            float share = bone_data[field + 6u * TILE];
            if (grounded) {
                // A node resting on the ground cannot give way.
                const bool child_down = pos[kc].y <= vel[kc].y + 1e-4f;
                const bool reference_down = pos[kq].y <= vel[kq].y + 1e-4f;
                if (child_down && !reference_down) {
                    share = 0.0f;
                } else if (reference_down && !child_down) {
                    share = 1.0f;
                }
            }
            const float2 dv = rotate_small(v, -side * excess * share) - v;
            const float2 du = rotate_small(u, side * excess * (1.0f - share)) - u;
            // Keep the three joint nodes' center of mass in place.
            const float2 shift = dv * bone_data[field + 7u * TILE] + du * bone_data[field + 8u * TILE];
            float2 new_n = pivot - shift;
            float2 new_c = pos[kc] + dv - shift;
            float2 new_q = pos[kq] + du - shift;
            if (grounded) {
                vel[kn].x -= fmaxf(vel[kn].y - new_n.y, 0.0f);
                vel[kc].x -= fmaxf(vel[kc].y - new_c.y, 0.0f);
                vel[kq].x -= fmaxf(vel[kq].y - new_q.y, 0.0f);
                new_n.y = fmaxf(new_n.y, vel[kn].y);
                new_c.y = fmaxf(new_c.y, vel[kc].y);
                new_q.y = fmaxf(new_q.y, vel[kq].y);
            }
            pos[kn] = new_n;
            pos[kc] = new_c;
            pos[kq] = new_q;
        }

        // Preserve the converged joint directions, then reconstruct the tree
        // parent-first so every edge has its exact rest length.
        float2 target_center = v2(0.0f, 0.0f);
        float mass_sum = 0.0f;
        UNROLL
        for (unsigned j = 0u; j < NODE_BOUND; j++) {
            if (j >= body_nodes) { break; }
            const unsigned k = node_k(j, lane);
            target_center += pos[k] * mass[j];
            mass_sum += mass[j];
        }
        // Each child keeps its offset from its parent, children first.
        UNROLL
        for (int j = (int)BONE_BOUND - 1; j >= 0; j--) {
            if ((unsigned)j >= bone_count) { continue; }
            const unsigned kb = bone_b(bone_kb[j]);
            pos[kb] = pos[kb] - pos[bone_a(bone_ka[j])];
        }
        UNROLL
        for (unsigned j = 0u; j < BONE_BOUND; j++) {
            if (j >= bone_count) { break; }
            const unsigned ka = bone_a(bone_ka[j]);
            const unsigned kb = bone_b(bone_kb[j]);
            const float2 delta = pos[kb];
            const float raw_distance = length(delta);
            float2 direction = raw_distance > 1e-6f ? delta * (1.0f / fmaxf(raw_distance, 1e-6f)) : v2(1.0f, 0.0f);
            const float2 previous_delta = old[kb] - old[ka];
            const float previous_length = length(previous_delta);
            if (previous_length > 1e-6f) {
                const float2 previous_direction = previous_delta * (1.0f / previous_length);
                if (dot(previous_direction, direction) < MAX_BONE_TURN_COS) {
                    const float cross = previous_direction.x * direction.y
                        - previous_direction.y * direction.x;
                    const float turn_sign = cross < 0.0f ? -1.0f : 1.0f;
                    const float2 turned = v2(
                        previous_direction.x - previous_direction.y * turn_sign * MAX_BONE_TURN_TAN,
                        previous_direction.y + previous_direction.x * turn_sign * MAX_BONE_TURN_TAN);
                    direction = turned / length(turned);
                }
            }
            const float2 new_child = pos[ka] + direction * bone_rest[j];
            pos[kb] = new_child;
        }
        float2 current_center = v2(0.0f, 0.0f);
        UNROLL
        for (unsigned j = 0u; j < NODE_BOUND; j++) {
            if (j >= body_nodes) { break; }
            const unsigned k = node_k(j, lane);
            current_center += pos[k] * mass[j];
        }
        const float2 center_shift = (target_center - current_center) * inv_total_mass;
        float ground_lift = 0.0f;
        UNROLL
        for (unsigned j = 0u; j < NODE_BOUND; j++) {
            if (j >= body_nodes) { break; }
            const unsigned k = node_k(j, lane);
            const float2 shifted = pos[k] + center_shift;
            pos[k] = shifted;
            if (grounded) {
                ground_lift = fmaxf(ground_lift, vel[k].y - shifted.y);
            }
        }
        // Planted feet may move the center of mass at most mu times the
        // ground's normal push; only planted feet may push it forward.
        float excess = 0.0f;
        if (grounded) {
            float held_mass = 0.0f;
            float held_grip = 0.0f;
            float normal = 0.0f;
            float com_x = 0.0f;
            float slide = 0.0f;
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                const unsigned k = node_k(j, lane);
                pos[k].y += ground_lift;
                // A node the ground pushed this step touched it, even if the
                // rebuild and the lift left it above the floor.
                const float push = fmaxf(fminf(pos[k].y, vel[k].y) - vel[k].x, 0.0f);
                if ((pos[k].y <= vel[k].y + 1e-4f || push > 0.0f) && failed[j] < 0.5f) {
                    held_mass += mass[j];
                    held_grip += mass[j] * friction[j];
                    normal += mass[j] * push;
                    slide += mass[j] * (pos[k].x - old[k].x);
                }
                com_x += pos[k].x * mass[j];
            }
            normal += (total_mass - held_mass) * ground_lift;
            const float mu = held_grip / fmaxf(held_mass, 1e-6f) * p.friction;
            const float allowed = mu * normal * inv_total_mass;
            slide /= fmaxf(held_mass, 1e-6f);
            const float planted = PLANTED_SPEED * DT;
            const float low = slide < -planted ? 0.0f : -allowed;
            const float high = slide > planted ? 0.0f : allowed;
            const float shift = com_x * inv_total_mass - com_x_before;
            excess = shift - clampf(shift, low, high);
        }
        float contact_mass = 0.0f;
        float contact_momentum = 0.0f;
        float contact_grip = 0.0f;
        float body_momentum = 0.0f;
        // Velocity is the actual movement over the step. Every node the
        // ground pushed feels friction, up to the floor.
        UNROLL
        for (unsigned j = 0u; j < NODE_BOUND; j++) {
            if (j >= body_nodes) { break; }
            const unsigned k = node_k(j, lane);
            if (grounded) {
                pos[k].x -= excess;
            }
            const float predicted_y = vel[k].x;
            const float floor_y = vel[k].y;
            float2 velocity = (pos[k] - old[k]) * RATE;
            velocity.y -= ground_lift * RATE;
            old[k] = v2(floor_y, 0.0f);
            float sink = 0.0f;
            float mud_mu = 1.0f;
            if (p.mud > 0.0f) {
                sink = clampf(floor_y + p.mud - pos[k].y, 0.0f, p.mud) / MUD_FULL_DEPTH;
                mud_mu = 1.0f + MUD_GRIP * sink;
            }
            const float pushed = fmaxf(fminf(pos[k].y, floor_y) - predicted_y, 0.0f);
            if (grounded && (pos[k].y <= floor_y + 1e-4f || pushed > 0.0f)) {
                const float push = pushed * (1.0f + MUD_NORMAL * sink);
                const float max_change = friction[j] * p.friction * mud_mu * push * RATE;
                velocity.x -= clampf(velocity.x, -max_change, max_change);
                velocity.x *= fmaxf(0.0f, 1.0f - MUD_DRAG * DT * sink);
                if (failed[j] < 0.5f) {
                    contact_mass += mass[j];
                    contact_momentum += mass[j] * velocity.x;
                    contact_grip += mass[j] * friction[j] * mud_mu;
                }
            } else if (sink > 0.0f) {
                velocity.x *= fmaxf(0.0f, 1.0f - MUD_DRAG * DT * sink);
            }
            if (failed[j] >= 0.5f) {
                velocity = v2(0.0f, 0.0f);
            }
            vel[k] = velocity;
            body_momentum += velocity.x * mass[j];
        }
        // The whole-body lift's friction, applied to the whole body. It may
        // only slow the body, never speed it up.
        if (grounded && ground_lift > 0.0f && contact_mass > 0.0f) {
            const float inv_contact = 1.0f / contact_mass;
            const float budget = contact_grip * inv_contact * p.friction * ground_lift * RATE;
            const float slide = contact_momentum * inv_contact;
            const float stop = -(body_momentum * inv_total_mass);
            const float change = clampf(-clampf(slide, -budget, budget), fminf(stop, 0.0f), fmaxf(stop, 0.0f));
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                if (failed[j] < 0.5f) {
                    vel[node_k(j, lane)].x += change;
                }
            }
        }
        for (unsigned iteration = 0u; iteration < VELOCITY_SOLVE_ITERATIONS; iteration++) {
            UNROLL
            for (unsigned j = 0u; j < BONE_BOUND; j++) {
                if (j >= bone_count) { break; }
                const unsigned ka = bone_a(bone_ka[j]);
                const unsigned kb = bone_b(bone_kb[j]);
                const float2 delta = pos[kb] - pos[ka];
                const float length_bone = fmaxf(length(delta), 1e-6f);
                const float2 direction = delta * (1.0f / length_bone);
                const float share_a = bone_sa[j];
                const float share_b = 1.0f - share_a;
                float2 velocity_a = vel[ka];
                float2 velocity_b = vel[kb];
                const float2 radial = direction * dot(velocity_b - velocity_a, direction);
                velocity_a += radial * share_a;
                velocity_b -= radial * share_b;

                const float2 tangent = v2(-direction.y, direction.x);
                const float relative_tangent = dot(velocity_b - velocity_a, tangent);
                const float max_tangent = MAX_BONE_ANGULAR_SPEED * length_bone;
                const float limited_tangent = clampf(relative_tangent, -max_tangent, max_tangent);
                const float2 angular = tangent * (relative_tangent - limited_tangent);
                velocity_a += angular * share_a;
                velocity_b -= angular * share_b;
                vel[ka] = velocity_a;
                vel[kb] = velocity_b;
            }
            float2 removed = v2(0.0f, 0.0f);
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                const unsigned k = node_k(j, lane);
                const float2 velocity = limit_speed(vel[k]);
                removed += (vel[k] - velocity) * mass[j];
                vel[k] = velocity;
            }
            const float2 correction = removed * inv_total_mass;
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                const unsigned k = node_k(j, lane);
                float2 velocity = vel[k] + correction;
                if (grounded && pos[k].y <= old[k].x + 1e-5f) {
                    velocity.y = fmaxf(velocity.y, 0.0f);
                }
                vel[k] = velocity;
            }
        }

        bool fell_now = false;
        if (tick >= SETTLE) {
            float center_y = 0.0f;
            float contacts = 0.0f;
            float low = 1e20f;
            float high = -1e20f;
            unsigned contact_lo = __float_as_uint(metrics.contact_lo);
            unsigned contact_hi = __float_as_uint(metrics.contact_hi);
            unsigned lift_lo = __float_as_uint(metrics.lift_lo);
            unsigned lift_hi = __float_as_uint(metrics.lift_hi);
            unsigned now_lo = 0u;
            unsigned now_hi = 0u;
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                const float y = pos[node_k(j, lane)].y;
                const float floor_y = old[node_k(j, lane)].x;
                center_y += y;
                low = fminf(low, y - radius[j]);
                high = fmaxf(high, y + radius[j]);
                if (p.ground > 0.0f) {
                    if (y <= floor_y + 0.002f) {
                        contacts += 1.0f;
                        if (j < 32u) {
                            contact_lo |= 1u << j;
                            now_lo |= 1u << j;
                        } else {
                            contact_hi |= 1u << (j - 32u);
                            now_hi |= 1u << (j - 32u);
                        }
                    } else if (y > floor_y + LIFT_CLEARANCE) {
                        // A foot must leave the ground after touching it.
                        if (j < 32u) {
                            lift_lo |= contact_lo & (1u << j);
                        } else {
                            lift_hi |= contact_hi & (1u << (j - 32u));
                        }
                    }
                }
            }
            metrics.contact_lo = __uint_as_float(contact_lo);
            metrics.contact_hi = __uint_as_float(contact_hi);
            metrics.lift_lo = __uint_as_float(lift_lo);
            metrics.lift_hi = __uint_as_float(lift_hi);
            center_y *= inv_nodes;
            const unsigned down_lo = now_lo & ~__float_as_uint(metrics.ground_lo);
            const unsigned down_hi = now_hi & ~__float_as_uint(metrics.ground_hi);
            metrics.ground_lo = __uint_as_float(now_lo);
            metrics.ground_hi = __uint_as_float(now_hi);
            if ((down_lo | down_hi) != 0u && tick > SETTLE) {
                // Muscles sensing a touchdown restart their rhythm at their
                // reset phase from the next step on.
                const float next_time = time + DT;
                for (unsigned j = 0u; j < muscle_count; j++) {
                    const unsigned field = tile.x + j * MUSCLE_FIELDS * TILE + tl;
                    const unsigned sensor = __float_as_uint(muscle_data[field + 11u * TILE]);
                    if (sensor == NO_SENSOR) {
                        continue;
                    }
                    const unsigned packed = __float_as_uint(muscle_data[field]);
                    const unsigned node = (packed >> (8u * sensor)) & 0xffu;
                    const unsigned touched = node < 32u ? (down_lo >> node) & 1u : (down_hi >> (node - 32u)) & 1u;
                    if (touched == 1u) {
                        const float clock = next_time * muscle_data[field + 5u * TILE] + muscle_data[field + 6u * TILE];
                        const float reset = muscle_data[field + 12u * TILE];
                        muscle_data[field + 13u * TILE] = fract(reset - clock);
                    }
                }
            }
            // A joint forced far past its range breaks.
            bool broken = false;
            UNROLL
            for (unsigned j = 0u; j < BONE_BOUND; j++) {
                if (j >= bone_count || metrics.fall_time != 0.0f) { break; }
                const float cos_half = bone_cos_half[j];
                if (cos_half <= -1.0f) { continue; }
                const float2 pivot = pos[bone_a(bone_ka[j])];
                const float2 u = pos[bone_q(bone_ka[j])] - pivot;
                const float2 v = pos[bone_b(bone_kb[j])] - pivot;
                const float norm = sqrtf(dot(u, u) * dot(v, v));
                if (norm < 1e-12f) { continue; }
                const float2 relative = v2(dot(u, v), u.x * v.y - u.y * v.x) * (1.0f / norm);
                const float2 center = unpack2x16snorm(bone_center[j]);
                if (relative.x * center.x + relative.y * center.y < bone_break[j]) {
                    broken = true;
                }
            }
            // A head shaken too hard kills the creature.
            if (metrics.fall_time == 0.0f && time >= HEAD_SHAKE_WINDOW) {
                const float head_accel = length(vel[node_k(0u, lane)] - head_start) * RATE;
                metrics.head_shake += (head_accel - metrics.head_shake)
                    * fminf(1.0f, 1.0f / (HEAD_SHAKE_WINDOW * RATE));
            }
            if (metrics.fall_time == 0.0f
                && (pos[node_k(0u, lane)].y < pos[bone_b(bone_kb[0])].y || broken || metrics.head_shake > HEAD_SHAKE_LIMIT)) {
                float fall_x = 0.0f;
                float failures = 0.0f;
                UNROLL
                for (unsigned j = 0u; j < NODE_BOUND; j++) {
                    if (j >= body_nodes) { break; }
                    fall_x += pos[node_k(j, lane)].x * mass[j];
                    failures += failed[j];
                }
                metrics.fall_time = time + DT;
                // Only failures up to the fall count: it ends the trial.
                metrics.fitness = failures > 0.0f ? -1e20f : fall_x * inv_total_mass;
                if (tick <= p.screen_tick) {
                    metrics.screen_x = metrics.fitness;
                }
                fell_now = true;
            }
            metrics.ground_contact += contacts;
            metrics.height_sum += high - low;
            metrics.vertical_oscillation = fminf(metrics.vertical_oscillation, center_y);
            metrics.gait_frequency = fmaxf(metrics.gait_frequency, center_y);
            if (tick == SETTLE) {
                metrics.previous_center_y = center_y;
                metrics.vertical_extremum = center_y;
                metrics.vertical_trend = 0.0f;
                metrics.gait_turns = 0.0f;
            } else if ((tick - SETTLE) % SAMPLE == 0u) {
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
            // Early screen.
            bool screened_now = false;
            if (tick == p.screen_tick && !fell_now) {
                float screen_x = 0.0f;
                float failures = 0.0f;
                UNROLL
                for (unsigned j = 0u; j < NODE_BOUND; j++) {
                    if (j >= body_nodes) { break; }
                    screen_x += pos[node_k(j, lane)].x * mass[j];
                    failures += failed[j];
                }
                metrics.screen_x = screen_x * inv_total_mass;
                if (metrics.screen_x < p.screen_bar) {
                    metrics.screened = time + DT;
                    metrics.fitness = failures > 0.0f ? -1e20f : metrics.screen_x;
                    screened_now = true;
                }
            }
            if (fell_now || screened_now) {
                metrics.vertical_oscillation = fmaxf(
                    metrics.gait_frequency - metrics.vertical_oscillation,
                    0.0f);
                metrics.gait_frequency = metrics.gait_turns * 0.5f / (time + DT);
                fell_now = true;
            }
        }
        if (tick + 1u == p.total_steps && !fell_now) {
            float score = 0.0f;
            float mass_sum = 0.0f;
            float failures = 0.0f;
            UNROLL
            for (unsigned j = 0u; j < NODE_BOUND; j++) {
                if (j >= body_nodes) { break; }
                score += pos[node_k(j, lane)].x * mass[j];
                mass_sum += mass[j];
                failures += failed[j];
            }
            if (failures > 0.0f) {
                metrics.fitness = -1e20f;
            } else if (metrics.fall_time == 0.0f) {
                // Fitness is distance only.
                metrics.fitness = score / mass_sum;
            }
            if (p.total_steps > SETTLE) {
                metrics.vertical_oscillation = fmaxf(
                    metrics.gait_frequency - metrics.vertical_oscillation,
                    0.0f);
                metrics.gait_frequency = metrics.gait_turns * 0.5f
                    / ((float)(p.total_steps - SETTLE) / RATE);
            } else {
                metrics.vertical_oscillation = 0.0f;
                metrics.gait_frequency = 0.0f;
            }
        }
    }
    results[creature] = metrics;
    UNROLL
    for (unsigned j = 0u; j < NODE_BOUND; j++) {
        if (j >= body_nodes) { break; }
        const unsigned k = node_k(j, lane);
        Node n = nodes[base + j];
        n.pos = pos[k];
        n.vel = vel[k];
        n.failed = failed[j];
        nodes[base + j] = n;
    }
}
