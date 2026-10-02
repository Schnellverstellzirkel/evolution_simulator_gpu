use crate::evolution::{Bone, Creature, FAILED, Muscle, NodeGene};
/// Physics steps per second. Every engine, the replay, and trial lengths
/// follow it.
pub fn rate() -> u32 {
    60
}
/// Seconds per physics step.
pub fn dt() -> f32 {
    Fidelity::standard().dt()
}
/// Steps of pose settling (1.67 s) before the timed trial.
pub fn settle() -> u32 {
    Fidelity::standard().settle()
}
/// Gait sampling interval in steps (30 samples per second).
pub fn sample_interval() -> u32 {
    Fidelity::standard().sample_interval()
}
/// Cosine and tangent of the largest bone turn allowed in one step.
pub fn turn_limits() -> (f32, f32) {
    Fidelity::standard().turn_limits()
}
/// Velocity kept per step, from `air_retention` per 1/60 s.
pub fn air_per_step(air_retention: f32) -> f32 {
    Fidelity::standard().air_per_step(air_retention)
}
/// How finely one evaluation resolves the physics: steps per second and
/// solver passes per step. Evaluations normally use `Fidelity::standard()`;
/// a finer one checks that a gait does not depend on the coarse steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fidelity {
    pub rate: u32,
    pub bone_passes: usize,
    pub velocity_passes: usize,
}
impl Fidelity {
    /// The standard physics: 60 steps per second and `solver_passes`.
    pub fn standard() -> Self {
        let (bone_passes, velocity_passes) = solver_passes();
        Self {
            rate: rate(),
            bone_passes,
            velocity_passes,
        }
    }
    /// Twice the standard rate and solver passes, for the confirmation trial
    /// of a creature that would set an island record.
    pub fn fine() -> Self {
        let standard = Self::standard();
        Self {
            // Only new records run a confirmation trial, so it can afford 4x:
            // a 2x check once let integrator exploits through
            // (docs/design-decisions.md).
            rate: (standard.rate * 4).min(960),
            bone_passes: standard.bone_passes * 4,
            velocity_passes: standard.velocity_passes * 4,
        }
    }
    pub fn dt(self) -> f32 {
        1.0 / self.rate as f32
    }
    /// Steps of pose settling (1.67 s) before the timed trial.
    pub fn settle(self) -> u32 {
        (200 * self.rate).div_ceil(120)
    }
    /// Gait sampling interval in steps (30 samples per second).
    pub fn sample_interval(self) -> u32 {
        (self.rate / 30).max(1)
    }
    /// Cosine and tangent of the largest bone turn allowed in one step.
    pub fn turn_limits(self) -> (f32, f32) {
        let angle = limits().bone_spin * self.dt();
        (angle.cos(), angle.tan())
    }
    /// Velocity kept per step, from `air_retention` per 1/60 s.
    pub fn air_per_step(self, air_retention: f32) -> f32 {
        air_retention.powf(60.0 / self.rate as f32)
    }
}
/// Position-projection and velocity-constraint passes per step. The rebuild
/// after projection makes every bone exactly its rest length regardless.
pub fn solver_passes() -> (usize, usize) {
    (2, 1)
}
/// Actuator and safety limits of the physics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// Fastest a muscle's target length may change (m/s).
    pub muscle_speed: f32,
    /// Largest muscle force (N).
    pub muscle_force: f32,
    /// Fastest any node may move (m/s).
    pub node_speed: f32,
    /// Fastest a bone may turn (rad/s).
    pub bone_spin: f32,
    /// Shortest muscle rhythm period (s).
    pub min_period: f32,
    /// Work a rested muscle can do before it tires (J).
    pub muscle_energy: f32,
    /// Share of a muscle's missing energy restored per second.
    pub muscle_recovery: f32,
    /// Longest bone (m).
    pub max_bone: f32,
    /// Longest muscle length (m); the shortest contracted length is 80% of it.
    pub max_stroke: f32,
    /// Bone mass per squared meter of bone length (kg/m^2). Longer bones are
    /// proportionally thicker, so mass grows with the square of length while
    /// muscle force stays capped: large bodies are heavy and slow, and weight
    /// gives feet the ground pressure they need to grip.
    pub bone_density: f32,
}
impl Limits {
    pub const DEFAULT: Limits = Limits {
        muscle_speed: 24.0,
        muscle_force: 100.0,
        node_speed: 60.0,
        bone_spin: 40.0,
        min_period: 0.2,
        muscle_energy: 120.0,
        muscle_recovery: 0.5,
        max_bone: 2.0,
        max_stroke: 2.0,
        bone_density: 4.0,
    };
}
/// The physics limits. Every engine, the replay, and mutation read the same
/// values.
pub fn limits() -> Limits {
    Limits::DEFAULT
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub radius: f32,
    pub friction: f32,
    pub mass: f32,
    pub failed: f32,
}
/// Mass (kg) of a node of the given diameter.
#[inline]
pub fn node_mass(diameter: f32) -> f32 {
    (0.1 * (diameter / 0.08).powi(2)).clamp(0.02, 10.0)
}
/// A node on its own, without the organs its bones carry.
#[inline]
pub fn node(gene: &NodeGene) -> Node {
    Node {
        pos: [gene.x, gene.y],
        vel: [0.0; 2],
        radius: gene.diameter * 0.5,
        friction: gene.friction,
        mass: node_mass(gene.diameter),
        failed: 0.0,
    }
}
/// A body's nodes with its bones' and organs' masses included. A bone's own
/// mass (`Limits::bone_density` times its length squared) is split evenly
/// between its two nodes. An organ rides rigidly on its bone, so its mass is
/// shared by the bone's two nodes in proportion to its position: the body's
/// center of mass is exact, and the organ never touches the ground.
pub fn body(genes: &[NodeGene], bones: &[Bone]) -> Vec<Node> {
    let mut nodes: Vec<Node> = genes.iter().map(node).collect();
    add_bone_masses(bones, &mut nodes);
    nodes
}

/// Writes a body's initial node state into caller-owned storage. GPU packing
/// processes millions of small bodies, so this avoids one heap allocation per
/// creature while preserving the same mass calculation as `body`.
pub fn body_into(genes: &[NodeGene], bones: &[Bone], nodes: &mut [Node]) {
    assert_eq!(
        nodes.len(),
        genes.len(),
        "body output length must match genes"
    );
    for (dst, gene) in nodes.iter_mut().zip(genes) {
        *dst = node(gene);
    }
    add_bone_masses(bones, nodes);
}

fn add_bone_masses(bones: &[Bone], nodes: &mut [Node]) {
    let density = limits().bone_density;
    for bone in bones {
        let (a, b) = (bone.a as usize, bone.b as usize);
        if a >= nodes.len() || b >= nodes.len() {
            continue;
        }
        let half = 0.5 * density * bone.rest_length * bone.rest_length;
        nodes[a].mass += half;
        nodes[b].mass += half;
        if bone.organ_mass > 0.0 {
            nodes[a].mass += bone.organ_mass * (1.0 - bone.organ_at);
            nodes[b].mass += bone.organ_mass * bone.organ_at;
        }
    }
}
/// A muscle's mass: a fixed part plus a part per metre of its length in the
/// start pose.
pub const MUSCLE_MASS_BASE: f32 = 0.05;
pub const MUSCLE_MASS_PER_M: f32 = 1.0;
/// Distance between a muscle's two nodes on `nodes` (the start pose).
pub fn muscle_span(nodes: &[Node], m: &Muscle) -> f32 {
    match (nodes.get(m.node_a as usize), nodes.get(m.node_b as usize)) {
        (Some(p), Some(q)) => ((p.pos[0] - q.pos[0]).powi(2) + (p.pos[1] - q.pos[1]).powi(2)).sqrt(),
        _ => 0.0,
    }
}
/// Adds each muscle's mass to the nodes (in the start pose): half at each
/// end.
pub fn add_muscle_masses(muscles: &[Muscle], nodes: &mut [Node]) {
    for m in muscles {
        let half = 0.5 * (MUSCLE_MASS_BASE + MUSCLE_MASS_PER_M * muscle_span(nodes, m));
        for node in [m.node_a, m.node_b] {
            if let Some(node) = nodes.get_mut(node as usize) {
                node.mass += half;
            }
        }
    }
}
/// A creature's nodes with the masses of its bones, organs and muscles.
pub fn nodes(c: &Creature) -> Vec<Node> {
    let mut nodes = body(&c.nodes, &c.bones);
    add_muscle_masses(&c.muscles, &mut nodes);
    nodes
}
/// The ramp on each edge of a muscle's activation (s): one step, which is
/// two substeps at the shipped count. A fixed time, so a finer substep
/// retests the same muscle.
pub fn muscle_ramp() -> f32 {
    dt()
}
/// A muscle's activation at `time` (0 to 1): a trapezoid of its period,
/// phase and duty with a ramp on each edge. Its clock restarts at `since`
/// (the time of its sensor's last touchdown) with the `reset` phase; without
/// one, `since` is 0 and the clock starts at `phase`. The kernel
/// computes the same thing.
pub fn activation(m: &Muscle, time: f32, since: Option<f32>) -> f32 {
    let (origin, phase) = match since {
        Some(t) => (t, m.reset),
        None => (0.0, m.phase),
    };
    let x = (time - origin) / m.period + phase;
    let ph = x - x.floor();
    let rp = muscle_ramp() / m.period;
    let c = (m.duty + rp).min(1.0) * 0.5;
    ((c - (ph - c).abs()) / rp).clamp(0.0, 1.0)
}
/// Joint range constraint for one bone, precomputed from the genome. The bone
/// turns about its parent node `a` against a reference bone that shares that
/// node: the parent's own bone, or for bones leaving the root, the first root
/// bone. Angles are measured from the reference end to the child end.
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    /// Far node of the reference bone; `None` for the unconstrained first bone.
    pub reference: Option<usize>,
    /// Direction of the middle of the allowed range, relative to the reference.
    pub center: [f32; 2],
    /// Cosine and sine of half the allowed range.
    pub half: [f32; 2],
    /// Share of a correction taken by the child end (by inverse inertia).
    pub child_share: f32,
    /// Masses of the child and reference ends over the three joint masses,
    /// used to keep the joint's center of mass in place.
    pub child_mass: f32,
    pub reference_mass: f32,
}
impl Joint {
    pub const FREE: Joint = Joint {
        reference: None,
        center: [1.0, 0.0],
        half: [-1.0, 0.0],
        child_share: 0.5,
        child_mass: 0.0,
        reference_mass: 0.0,
    };
}
/// How much heavier a node resting on the ground counts, per unit of grip
/// (node friction times ground friction), when bones pull on it during the
/// constraint passes. The solver splits every bone correction by mass, so a
/// light foot would otherwise be dragged along by its heavy body instead of
/// holding its place; with this, a body pivots over planted feet. On ice the
/// grip is small, so feet still slide.
pub const STANCE_GRIP: f32 = 10.0;
/// Feet on the ground sliding slower than this (m/s, mass-weighted mean) count
/// as planted, so friction may push the body forward from them. Faster, the
/// feet slide and friction can only oppose the slide.
pub const PLANTED_SPEED: f32 = 0.01;
/// Static friction: a foot that barely slides can take `1 + STATIC_EXTRA`
/// times the kinetic friction bound. The extra fades linearly to nothing
/// between 1 cm/s and 2 cm/s of slide, so no step chatters across a switch.
/// The kernels write the same numbers as literals (0.25, 0.02, 100.0).
pub const STATIC_EXTRA: f32 = 0.25;
/// Slide speed (m/s) at which the static extra is gone.
pub const STATIC_FADE_END: f32 = 0.02;
/// The static factor, `1 + STATIC_EXTRA * clamp((STATIC_FADE_END - |v|) * 100, 0, 1)`.
pub fn static_factor(slide: f32) -> f32 {
    1.0 + 0.25 * ((0.02 - slide.abs()) * 100.0).clamp(0.0, 1.0)
}
/// Head shaking limit: the head's acceleration, averaged over about
/// `HEAD_SHAKE_WINDOW` seconds, may not pass 8 g (m/s^2). A creature that
/// shakes its head harder dies like a fall. Single impacts average out, but a
/// body jiggling at the physics step rate does not, so solver jitter cannot
/// carry a creature forward.
pub const HEAD_SHAKE_LIMIT: f32 = 8.0 * 9.8;
pub const HEAD_SHAKE_WINDOW: f32 = 0.1;
/// Early screening of a standard trial: at `seconds` after settling, a
/// creature whose distance is below `bar` stops, like a fall, keeping that
/// distance. Survivors run the full trial. A screened creature never enters
/// an archive, so every elite has a full trial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Screen {
    pub seconds: f32,
    pub bar: f32,
}
impl Screen {
    /// The step at whose end the screen applies.
    pub fn tick(self, fidelity: Fidelity) -> u32 {
        fidelity.settle() + ((self.seconds * fidelity.rate as f32).round() as u32).max(1) - 1
    }
}
/// Seconds after settling at which trials are screened.
pub fn screen_seconds() -> Option<f32> {
    Some(5.0)
}
/// Share of creatures, by distance at the screen, that runs the full trial
/// (the owner's choice of 2026-10-02, for speed; it was 20%, which held
/// every creature of the final top 1% and 96% of the final top 10% on an
/// evolved 3M population).
pub fn screen_keep() -> f32 {
    0.1
}
/// The distance that the best `keep` share of `distances` reached (NaN
/// entries are ignored), or no bar when fewer than 64 distances are known.
pub fn screen_bar(distances: impl Iterator<Item = f32>, keep: f32) -> f32 {
    let mut distances: Vec<f32> = distances.filter(|d| !d.is_nan()).collect();
    if distances.len() < 64 || keep >= 1.0 {
        return f32::NEG_INFINITY;
    }
    let rank = ((distances.len() as f32 * (1.0 - keep)) as usize).min(distances.len() - 1);
    *distances.select_nth_unstable_by(rank, f32::total_cmp).1
}
/// How far (rad) a joint may be forced past its range before it breaks. A
/// broken joint ends the trial like a fall, so no gait can profit from
/// muscles forcing joints round like wheels.
pub const JOINT_BREAK: f32 = 0.5;
/// Joint constraints for a canonical (parent-first) skeleton.
/// Reference node of bone `index`'s joint: its parent bone's pivot, or for a
/// bone at the root the first root bone's child. `None` for a free joint. It
/// depends only on the skeleton, so every creature of a body plan shares it.
pub fn joint_reference(bones: &[Bone], index: usize) -> Option<usize> {
    let pivot = bones[index].a as usize;
    match bones.iter().find(|p| p.b as usize == pivot) {
        Some(parent) => Some(parent.a as usize),
        None => match bones.iter().position(|b| b.a == 0) {
            Some(first) if first != index => Some(bones[first].b as usize),
            _ => None,
        },
    }
}
/// Rounds to the GPU's snorm16 storage of the joint range center.
fn snorm16(v: f32) -> f32 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() / 32767.0
}

/// Writes joint constants using an already computed body state. Callers that
/// pack many bodies can reuse the output slice and avoid allocating both the
/// body nodes and the joint vector for every creature.
pub fn joints_from_body(genes: &[NodeGene], bones: &[Bone], state: &[Node], out: &mut [Joint]) {
    assert_eq!(
        state.len(),
        genes.len(),
        "joint state length must match genes"
    );
    assert_eq!(
        out.len(),
        bones.len(),
        "joint output length must match bones"
    );
    for (index, bone) in bones.iter().enumerate() {
        let pivot = bone.a as usize;
        let child = bone.b as usize;
        let Some(reference) = joint_reference(bones, index) else {
            out[index] = Joint::FREE;
            continue;
        };
        let at = |i: usize| [genes[i].x - genes[pivot].x, genes[i].y - genes[pivot].y];
        let (u, v) = (at(reference), at(child));
        let rest = (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1]);
        let middle = rest + 0.5 * (bone.min_angle + bone.max_angle);
        let half = 0.5 * (bone.max_angle - bone.min_angle);
        let length = |d: [f32; 2]| d[0].hypot(d[1]).max(0.01);
        let inertia_child = state[child].mass * length(v).powi(2);
        let inertia_reference = state[reference].mass * length(u).powi(2);
        let total = state[pivot].mass + state[child].mass + state[reference].mass;
        out[index] = Joint {
            reference: Some(reference),
            center: [snorm16(middle.cos()), snorm16(middle.sin())],
            half: [half.cos(), half.sin()],
            child_share: inertia_reference / (inertia_child + inertia_reference),
            child_mass: state[child].mass / total,
            reference_mass: state[reference].mass / total,
        };
    }
}

pub fn joints(genes: &[NodeGene], bones: &[Bone]) -> Vec<Joint> {
    let state = body(genes, bones);
    let mut out = vec![Joint::FREE; bones.len()];
    joints_from_body(genes, bones, &state, &mut out);
    out
}

/// Ground heights (m) of the bumps added by each roughness level.
pub const TERRAIN_AMPLITUDES: [f32; 5] = [0.0, 0.03, 0.08, 0.15, 0.25];
/// Bump height for `Config::terrain`.
pub fn terrain_amplitude(level: u8) -> f32 {
    TERRAIN_AMPLITUDES[usize::from(level).min(TERRAIN_AMPLITUDES.len() - 1)]
}
/// Wavelengths (m), weights, and phase offsets of the two bump trains. The
/// engines and the UI all evaluate the same ground.
pub const TERRAIN_WAVES: [(f32, f32, f32); 2] = [(1.1, 0.65, 0.0), (0.43, 0.35, 0.3)];
/// Ground height and slope at `x` for bump height `amplitude`. Each bump is
/// 16 u^2 (1 - u)^2 over one wavelength: smooth, cheap, and free of trig.
pub fn terrain(x: f32, amplitude: f32) -> (f32, f32) {
    terrain_phase(x, amplitude, 0.0)
}
/// Ground height and slope of the bump trains with a phase in wave turns:
/// `phase` shifts both trains by the same fraction of their wavelength. The
/// earthquake effect passes a per-creature phase here; a 16-bit fraction is
/// exactly representable, so both engines apply it bit for bit.
pub fn terrain_phase(x: f32, amplitude: f32, phase: f32) -> (f32, f32) {
    let mut height = 0.0;
    let mut slope = 0.0;
    for (wavelength, weight, offset) in TERRAIN_WAVES {
        let t = x / wavelength + offset + phase;
        let u = t - t.floor();
        let w = u * (1.0 - u);
        height += weight * 16.0 * w * w;
        slope += weight * 32.0 * w * (1.0 - 2.0 * u) / wavelength;
    }
    (amplitude * height, amplitude * slope)
}
/// Ground height and slope at `x` for bump height `amplitude` plus a linear
/// `tilt` (rise over run) that raises the ground in the +x direction. The two
/// engines sample this same ground; `tilt` is 0 when the slope effect is calm
/// or the ground is disabled.
pub fn terrain_with_slope(x: f32, amplitude: f32, tilt: f32) -> (f32, f32) {
    let (height, slope) = terrain(x, amplitude);
    (height + tilt * x, slope + tilt)
}
/// Depth (m) of every pit below the surrounding ground.
pub const GAP_DEPTH: f32 = 2.0;
/// Horizontal run (m) of each pit's wall. Walls are short steep ramps, not
/// vertical steps, so a node never teleports when the sampled floor jumps.
pub const GAP_RUN: f32 = 0.15;
/// Distance (m) between pit centers for a pit opening `width` wide. Wider
/// pits are also farther apart, so the solid stretches stay walkable.
pub fn gap_spacing(width: f32) -> f32 {
    2.0 + 4.0 * width
}
/// Ground height (m, negative) and slope of the periodic pits for pit width
/// `width`; both are 0 on solid ground. Pit centers sit at odd multiples of
/// `gap_spacing / 2`, so x = 0 is solid ground. Every engine and the UI carve
/// the same pits.
pub fn gaps(x: f32, width: f32) -> (f32, f32) {
    if width <= 0.0 {
        return (0.0, 0.0);
    }
    let spacing = gap_spacing(width);
    let r = x - (x / spacing).floor() * spacing;
    let center = 0.5 * spacing;
    let distance = (r - center).abs();
    let half = 0.5 * width;
    let run = GAP_RUN.min(half);
    let factor = if distance >= half {
        0.0
    } else if distance <= half - run {
        1.0
    } else {
        (half - distance) / run
    };
    let factor_slope = if distance > half - run && distance < half {
        if r < center { 1.0 / run } else { -1.0 / run }
    } else {
        0.0
    };
    (-GAP_DEPTH * factor, -GAP_DEPTH * factor_slope)
}
/// Distance (m) between the centers of two raised hurdles.
pub const HURDLE_SPACING: f32 = 3.0;
/// Width (m) of each hurdle's flat top.
pub const HURDLE_TOP: f32 = 1.2;
/// Horizontal run (m) of each hurdle's ramp. Ramps are short and steep, not
/// vertical steps, so a node never teleports when the sampled floor jumps.
pub const HURDLE_RUN: f32 = 0.2;
/// Ground height (m, positive) and slope of the periodic raised steps for step
/// height `height`; both are 0 on clear ground. Each step is a ramp up, a flat
/// top of `HURDLE_TOP` meters, and a ramp down, centered at odd multiples of
/// half the spacing, so x = 0 starts on clear ground. Every engine and the UI
/// raise the same steps.
pub fn hurdles(x: f32, height: f32) -> (f32, f32) {
    if height <= 0.0 {
        return (0.0, 0.0);
    }
    let spacing = HURDLE_SPACING;
    let r = x - (x / spacing).floor() * spacing;
    let center = 0.5 * spacing;
    let distance = (r - center).abs();
    let half = 0.5 * HURDLE_TOP;
    let run = HURDLE_RUN;
    let factor = if distance <= half {
        1.0
    } else if distance >= half + run {
        0.0
    } else {
        (half + run - distance) / run
    };
    let factor_slope = if distance > half && distance < half + run {
        if r < center { 1.0 / run } else { -1.0 / run }
    } else {
        0.0
    };
    (height * factor, height * factor_slope)
}
/// Deterministic earthquake stream of a creature id. Both engines derive a
/// creature's terrain from this same unsigned word, so a replay and both
/// engines give one creature one ground. Only the low 32 bits of the id are
/// mixed; the GPU receives the mixed word directly.
#[inline]
pub fn quake_hash(id: u64) -> u32 {
    let mut x = id as u32;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}
/// Phase (wave turns, 0 to 1) of a creature's earthquake bumps, from its
/// hash. A 16-bit fraction is exactly representable in f32, so every engine
/// gets a stable phase value without requiring matching trajectories.
pub fn quake_phase(hash: u32) -> f32 {
    (hash & 0xffff) as f32 * (1.0 / 65536.0)
}
/// Amplitude multiplier of a creature's earthquake bumps, 0.6 to 1.4, from
/// its hash. Applied to `Config::quake` on top of the shared roughness.
pub fn quake_scale(hash: u32) -> f32 {
    0.6 + ((hash >> 16) & 0xffff) as f32 * (0.8 / 65536.0)
}
/// Ground height and slope at `x` for bump height `amplitude`, linear `tilt`,
/// periodic pits of opening `width`, and raised steps of height `hurdle`.
/// `phase` is the earthquake phase in wave turns applied to the bumps alone.
/// This is the one place the effects join the ground; `width` and `hurdle`
/// are 0 when the gaps or hurdles effects are calm, the phase is 0 when the
/// quake is still, and the tilt, pits, and steps are zeroed when the ground
/// is disabled.
pub fn ground(
    x: f32,
    amplitude: f32,
    tilt: f32,
    width: f32,
    hurdle: f32,
    phase: f32,
) -> (f32, f32) {
    let (bump_height, bump_slope) = terrain_phase(x, amplitude, phase);
    let (mut height, mut slope) = (bump_height + tilt * x, bump_slope + tilt);
    let (pit_height, pit_slope) = gaps(x, width);
    height += pit_height;
    slope += pit_slope;
    let (step_height, step_slope) = hurdles(x, hurdle);
    (height + step_height, slope + step_slope)
}
/// Effective ground push multiplier while a node is sunk in mud. The push a
/// contacting node receives counts this much higher at `MUD_FULL_DEPTH` of
/// sink, which multiplies the friction budget over it.
pub const MUD_NORMAL: f32 = 2.0;
/// Friction multiplier at `MUD_FULL_DEPTH` of sink. Together with
/// `MUD_NORMAL` a node sunk that far feels Coulomb friction scaled by
/// `(1 + MUD_GRIP) * (1 + MUD_NORMAL)`.
pub const MUD_GRIP: f32 = 2.0;
/// Horizontal velocity retention lost per second at `MUD_FULL_DEPTH` of
/// sink: the viscous drag of moving a foot through mud. Zero effect on a
/// lifted foot.
pub const MUD_DRAG: f32 = 2.0;
/// Sink depth (m) at which the mud multipliers reach their full value, i.e.
/// the deepest mud level. Shallower mud drags proportionally less.
pub const MUD_FULL_DEPTH: f32 = 0.10;
/// Distance (m) between the starts of two ice patches.
pub const ICE_SPACING: f32 = 6.0;
/// How icy the ground is at `x`, from 0 (dry) to 1 (ice): bands about 2.4 m
/// wide in the middle of every `ICE_SPACING`, with smooth 0.8 m edges. Only
/// IEEE arithmetic, so the kernels compute it bit for bit the same.
pub fn ice(x: f32) -> f32 {
    let u = x * (1.0 / ICE_SPACING);
    let w = u - u.floor();
    let t = (w - 0.5).abs() * 2.0;
    let s = ((0.7 - t) * 2.5).clamp(0.0, 1.0);
    s * s * (3.0 - 2.0 * s)
}
pub fn fitness(n: &[Node]) -> f32 {
    if n.iter().any(|n| n.failed != 0.0) {
        FAILED
    } else {
        let mass_sum = n.iter().map(|node| node.mass).sum::<f32>();
        n.iter().map(|node| node.pos[0] * node.mass).sum::<f32>() / mass_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_slope_matches_its_height() {
        for i in 0..200 {
            let x = i as f32 * 0.037 - 3.0;
            let (_, slope) = terrain(x, 0.05);
            let h = 1e-3;
            let numeric = (terrain(x + h, 0.05).0 - terrain(x - h, 0.05).0) / (2.0 * h);
            assert!((slope - numeric).abs() < 1e-2, "{x}: {slope} vs {numeric}");
            assert!((0.0..=0.05 + 1e-6).contains(&terrain(x, 0.05).0));
        }
        assert_eq!(terrain(0.7, 0.0), (0.0, 0.0));
        // The linear tilt raises the ground and adds a constant to the slope,
        // so the same numeric check holds on a hill.
        let tilt = 0.15;
        for i in 0..200 {
            let x = i as f32 * 0.037 - 3.0;
            let (height, slope) = terrain_with_slope(x, 0.05, tilt);
            let h = 1e-3;
            let numeric = (terrain_with_slope(x + h, 0.05, tilt).0
                - terrain_with_slope(x - h, 0.05, tilt).0)
                / (2.0 * h);
            assert!((slope - numeric).abs() < 1e-2, "{x}: {slope} vs {numeric}");
            assert!((height - (terrain(x, 0.05).0 + tilt * x)).abs() < 1e-6);
        }
        assert_eq!(terrain_with_slope(0.7, 0.0, 0.0), (0.0, 0.0));
        assert_eq!(terrain_with_slope(2.0, 0.0, 0.25), (0.5, 0.25));
    }

    #[test]
    fn gaps_carve_bounded_pits_every_spacing() {
        let width = 1.0;
        let spacing = gap_spacing(width);
        // The start is solid ground: a trial never begins over a pit.
        assert_eq!(gaps(0.0, width), (0.0, 0.0));
        assert!(gaps(0.6, width).0 > -1e-6);
        // The middle of the first pit is at the full depth with a flat floor.
        let center = 0.5 * spacing;
        assert_eq!(gaps(center, width), (-GAP_DEPTH, 0.0));
        let floor = -GAP_DEPTH;
        for x in [0.01, 0.5, 1.4, 7.3, 15.9] {
            let (height, _) = gaps(x, width);
            assert!(
                (floor - 1e-6..=0.0).contains(&height),
                "pit at {x} has height {height}"
            );
        }
        // Outside the pits the ground is untouched, and the profile is
        // continuous across every pit wall.
        let mut previous = gaps(-spacing, width).0;
        let step = 0.001;
        let mut x = -spacing + step;
        while x <= 3.0 * spacing {
            let height = gaps(x, width).0;
            assert!(
                (height - previous).abs() < 0.5,
                "pit profile jumps at {x}: {previous} to {height}"
            );
            previous = height;
            x += step;
        }
        assert_eq!(gaps(0.5, 0.0), (0.0, 0.0));
    }

    #[test]
    fn ground_slope_matches_its_height_with_pits() {
        for i in 0..400 {
            let x = i as f32 * 0.037 - 4.0;
            let (_, slope) = ground(x, 0.05, 0.15, 1.0, 0.0, 0.0);
            let h = 1e-3;
            let numeric = (ground(x + h, 0.05, 0.15, 1.0, 0.0, 0.0).0
                - ground(x - h, 0.05, 0.15, 1.0, 0.0, 0.0).0)
                / (2.0 * h);
            assert!((slope - numeric).abs() < 1e-2, "{x}: {slope} vs {numeric}");
        }
        // With no gaps, hurdles, or phase the combined ground is exactly
        // terrain plus slope.
        for x in [0.0, 0.7, -2.4, 13.1] {
            assert_eq!(
                ground(x, 0.05, 0.15, 0.0, 0.0, 0.0),
                terrain_with_slope(x, 0.05, 0.15)
            );
        }
    }

    #[test]
    fn hurdles_rise_and_stay_continuous() {
        let height = 0.3;
        // The start is clear ground: a trial never begins on a step.
        assert_eq!(hurdles(0.0, height), (0.0, 0.0));
        assert_eq!(hurdles(1.5, 0.0), (0.0, 0.0));
        // The flat top holds the full height with no slope.
        let top = 0.5 * HURDLE_SPACING;
        assert_eq!(hurdles(top, height), (height, 0.0));
        // The profile is continuous across both ramps, and the height never
        // leaves [0, height].
        let step = 0.0005;
        let mut previous = hurdles(-HURDLE_SPACING, height).0;
        let mut x = -HURDLE_SPACING + step;
        while x <= 2.0 * HURDLE_SPACING {
            let current = hurdles(x, height).0;
            assert!(
                (current - previous).abs() < 0.5,
                "hurdle profile jumps at {x}: {previous} to {current}"
            );
            assert!((0.0..=height + 1e-6).contains(&current));
            previous = current;
            x += step;
        }
        // A taller level raises the same profile proportionally.
        for x in [0.2, 0.8, 1.5, 2.6] {
            let (low, low_slope) = hurdles(x, 0.1);
            let (high, high_slope) = hurdles(x, 0.2);
            assert!((high - 2.0 * low).abs() < 1e-6);
            assert!((high_slope - 2.0 * low_slope).abs() < 1e-6);
        }
    }

    #[test]
    fn quake_streams_are_deterministic_and_spread() {
        assert_eq!(quake_hash(47300), quake_hash(47300));
        let mut phases = std::collections::HashSet::new();
        let mut scales = std::collections::HashSet::new();
        for id in 1..=1000u64 {
            let hash = quake_hash(id);
            let (phase, scale) = (quake_phase(hash), quake_scale(hash));
            assert!((0.0..1.0).contains(&phase), "phase {phase} for id {id}");
            assert!((0.6..=1.4).contains(&scale), "scale {scale} for id {id}");
            phases.insert(phase.to_bits());
            scales.insert(scale.to_bits());
        }
        assert!(
            phases.len() > 900,
            "the phase must vary across creature ids: {} distinct",
            phases.len()
        );
        assert!(
            scales.len() > 900,
            "the amplitude must vary across creature ids: {} distinct",
            scales.len()
        );
    }

    #[test]
    fn ground_slope_matches_its_height_with_hurdles() {
        for i in 0..400 {
            let x = i as f32 * 0.0371 - 4.0;
            let (_, slope) = ground(x, 0.03, 0.05, 0.0, 0.25, 0.37);
            let h = 1e-3;
            let numeric = (ground(x + h, 0.03, 0.05, 0.0, 0.25, 0.37).0
                - ground(x - h, 0.03, 0.05, 0.0, 0.25, 0.37).0)
                / (2.0 * h);
            assert!((slope - numeric).abs() < 1e-2, "{x}: {slope} vs {numeric}");
        }
    }

    #[test]
    fn fitness_tracks_center_of_mass_instead_of_shape_average() {
        let before = [
            Node {
                pos: [-1.0, 0.0],
                mass: 1.0,
                ..Node::default()
            },
            Node {
                pos: [1.0, 0.0],
                mass: 3.0,
                ..Node::default()
            },
        ];
        let reshaped = [
            Node {
                pos: [-3.0, 0.0],
                mass: 1.0,
                ..Node::default()
            },
            Node {
                pos: [5.0 / 3.0, 0.0],
                mass: 3.0,
                ..Node::default()
            },
        ];
        assert!(
            (before.iter().map(|node| node.pos[0]).sum::<f32>() / 2.0
                - reshaped.iter().map(|node| node.pos[0]).sum::<f32>() / 2.0)
                .abs()
                > 0.1
        );
        assert!((fitness(&before) - fitness(&reshaped)).abs() < 1e-6);
    }
}
