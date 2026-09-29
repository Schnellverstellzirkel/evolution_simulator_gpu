//! One-creature-per-lane GPU kernel (`shaders/physics_creature.wgsl`).
//!
//! Creatures are grouped by padded node capacity. Inside a group they are
//! sorted by body size so the 32 creatures that share a warp run similar loop
//! counts. Muscles and bones are packed per 32-creature tile as
//! `[item][field][lane]`, which makes every warp load one coalesced line.
use crate::{
    evolution::Population,
    physics::{self, Node},
};
use anyhow::{Result, ensure};
use rayon::prelude::*;

pub const TILE: usize = 32;
pub const MUSCLE_FIELDS: usize = 15;
/// Per bone: endpoints and joint reference node, rest length, and the joint
/// range constants of `physics::Joint`.
pub const BONE_FIELDS: usize = 9;
pub const CAPACITIES: [usize; 12] = [3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub tick: u32,
    pub steps: u32,
    pub stride: u32,
    pub count: u32,
    pub gravity: f32,
    pub air: f32,
    pub friction: f32,
    pub ground: f32,
    pub total_steps: u32,
    /// Bump height of the rough ground (m); 0 is flat.
    pub terrain: f32,
    /// Multiplier on the muscle energy store (heat wave); 1.0 is calm.
    pub muscle_energy: f32,
    /// Multiplier on muscle energy recovery (drought); 1.0 is calm.
    pub muscle_recovery: f32,
    /// Ground slope (rise over run), zeroed when the ground is disabled.
    pub slope: f32,
    /// Steady horizontal wind acceleration (m/s²); positive pushes +x.
    pub wind: f32,
    /// Mud sink depth (m); 0.0 is dry ground.
    pub mud: f32,
    /// Pit opening width (m); 0.0 is solid ground.
    pub gaps: f32,
    /// Raised step height (m); 0.0 is clear ground.
    pub hurdles: f32,
    /// Earthquake base bump height (m); each creature jitters it from the
    /// hash of its id, packed in the last word of `creature_info`.
    pub quake: f32,
    /// Step at whose end trials are screened (`physics::Screen::tick`), or 0
    /// for no screen, and the distance a creature needs there to continue.
    pub screen_tick: u32,
    pub screen_bar: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuResult {
    pub fitness: f32,
    pub ground_contact: f32,
    pub vertical_oscillation: f32,
    pub gait_frequency: f32,
    pub previous_center_y: f32,
    pub vertical_extremum: f32,
    pub vertical_trend: f32,
    pub gait_turns: f32,
    pub height_sum: f32,
    /// Bits of nodes 0-31 / 32-63 that touched the ground (stored as f32 bits).
    pub contact_lo: f32,
    pub contact_hi: f32,
    /// Bits of touching nodes that later lifted clear of the ground again.
    pub lift_lo: f32,
    pub lift_hi: f32,
    /// Nodes grounded after the last step (f32 bits), for sensor touchdowns.
    pub ground_lo: f32,
    pub ground_hi: f32,
    /// Seconds into the trial when the head tipped below its neck base, or 0
    /// if the creature stayed upright. Fitness is the distance at the fall.
    pub fall_time: f32,
    /// Mean head acceleration (m/s^2) over about `physics::HEAD_SHAKE_WINDOW`
    /// seconds, for the head shaking limit.
    pub head_shake: f32,
    /// Distance at the screen, or at an earlier fall; 0 until then. The
    /// experiment sets the next generation's screen bar from these.
    pub screen_x: f32,
    /// Seconds into the trial when the screen stopped the creature, or 0.
    /// Its fitness is the distance there and its behavior totals end there.
    pub screened: f32,
}
impl GpuResult {
    /// Number of feet: nodes that touched the ground and lifted off again.
    /// A node dragged along the ground never lifts, so it is not a foot.
    pub fn feet(&self) -> u32 {
        self.lift_lo.to_bits().count_ones() + self.lift_hi.to_bits().count_ones()
    }
}

/// One node-capacity group, ready for upload.
pub struct LaneBatch {
    pub capacity: usize,
    /// Set when every creature in the batch shares this body plan, so the
    /// batch can run the plan's specialized kernel.
    pub plan: Option<Plan>,
    /// Position of each packed creature within the caller's index slice.
    pub slots: Vec<usize>,
    /// Population index of each packed creature.
    pub creatures: Vec<usize>,
    pub nodes: Vec<Node>,
    pub info: Vec<[u32; 4]>,
    pub tiles: Vec<[u32; 4]>,
    pub muscles: Vec<f32>,
    /// Fields per muscle in `muscles` (`MUSCLE_FIELDS` for this kernel).
    pub muscle_fields: usize,
    pub bones: Vec<f32>,
    /// Behavior totals to resume from, for a batch that continues a trial
    /// (`repack`); a fresh batch starts from zero.
    pub results: Option<Vec<GpuResult>>,
}

impl LaneBatch {
    /// Frees the node state, muscle buffer and resumed totals once the GPU
    /// holds them: a later segment repacks from the state it reads back, and
    /// the bodies (`bones`, `info`, `tiles`) stay for that.
    pub fn release_uploaded(&mut self) {
        self.nodes = Vec::new();
        self.muscles = Vec::new();
        self.results = None;
    }
    /// The creatures at positions `keep` (ascending) of this batch, with the
    /// node state, muscle buffer (rhythm offsets and energy included) and
    /// behavior totals read back after a trial segment, so a later segment
    /// continues their trials exactly. Tiles are rebuilt for the survivors.
    pub fn repack(
        &self,
        keep: &[usize],
        nodes: &[Node],
        muscles: &[f32],
        results: &[GpuResult],
    ) -> LaneBatch {
        let capacity = self.capacity;
        let count = keep.len();
        let mut tiles = Vec::with_capacity(count.div_ceil(TILE));
        let mut muscle_len = 0usize;
        let mut bone_len = 0usize;
        for tile in keep.chunks(TILE) {
            let max_muscles = tile.iter().map(|&j| self.info[j][2]).max().unwrap_or(0) as usize;
            let max_bones = tile.iter().map(|&j| self.info[j][1]).max().unwrap_or(0) as usize;
            tiles.push([
                muscle_len as u32,
                bone_len as u32,
                max_muscles as u32,
                max_bones as u32,
            ]);
            muscle_len += max_muscles * self.muscle_fields * TILE;
            bone_len += max_bones * BONE_FIELDS * TILE;
        }
        let mut new_nodes = vec![Node::default(); count * capacity];
        let mut new_muscles = vec![0f32; muscle_len.max(1)];
        let mut new_bones = vec![0f32; bone_len.max(1)];
        for (to, &from) in keep.iter().enumerate() {
            new_nodes[to * capacity..(to + 1) * capacity]
                .copy_from_slice(&nodes[from * capacity..(from + 1) * capacity]);
            let (old_tile, old_lane) = (self.tiles[from / TILE], from % TILE);
            let (new_tile, new_lane) = (tiles[to / TILE], to % TILE);
            for m in 0..self.info[from][2] as usize {
                for f in 0..self.muscle_fields {
                    let at = (m * self.muscle_fields + f) * TILE;
                    new_muscles[new_tile[0] as usize + at + new_lane] =
                        muscles[old_tile[0] as usize + at + old_lane];
                }
            }
            for b in 0..self.info[from][1] as usize {
                for f in 0..BONE_FIELDS {
                    let at = (b * BONE_FIELDS + f) * TILE;
                    new_bones[new_tile[1] as usize + at + new_lane] =
                        self.bones[old_tile[1] as usize + at + old_lane];
                }
            }
        }
        LaneBatch {
            capacity,
            plan: self.plan.clone(),
            slots: keep.iter().map(|&j| self.slots[j]).collect(),
            creatures: keep.iter().map(|&j| self.creatures[j]).collect(),
            nodes: new_nodes,
            info: keep.iter().map(|&j| self.info[j]).collect(),
            tiles,
            muscles: new_muscles,
            muscle_fields: self.muscle_fields,
            bones: new_bones,
            results: Some(keep.iter().map(|&j| results[j]).collect()),
        }
    }
}

pub fn capacity_index(nodes: usize) -> usize {
    CAPACITIES
        .iter()
        .position(|&c| nodes <= c)
        .unwrap_or(CAPACITIES.len() - 1)
}

/// Waveform amplitude: the full stroke, limited so the target length changes
/// at most `physics::limits().muscle_speed`.
pub fn muscle_amplitude(m: &crate::evolution::Muscle) -> f32 {
    (m.long - m.short).min(
        2.0 * crate::physics::limits().muscle_speed * m.period * m.duty.min(1.0 - m.duty)
            / std::f32::consts::PI,
    )
}

/// Hash of a creature's bone and muscle attachment layout.
pub fn plan_hash(pop: &Population, index: usize) -> u64 {
    let g = &pop.genomes[index];
    let mix = |h: u64, v: u32| (h ^ u64::from(v)).wrapping_mul(0x100_0000_01b3);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in &pop.bones[g.bone_start..g.bone_start + g.bone_count] {
        h = mix(mix(h, b.a), b.b);
    }
    for m in &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count] {
        h = mix(mix(h, m.bone_a), m.bone_b);
    }
    h
}

pub fn pack(pop: &Population, indices: &[usize]) -> Result<Vec<LaneBatch>> {
    let mut groups: [Vec<(usize, usize)>; CAPACITIES.len()] = Default::default();
    for (slot, &i) in indices.iter().enumerate() {
        let n = pop.genomes[i].node_count;
        ensure!((1..=64).contains(&n), "Unsupported body size");
        groups[capacity_index(n)].push((slot, i));
    }
    let plan_batch = plan_batch_size();
    Ok(groups
        .into_par_iter()
        .enumerate()
        .filter(|(_, g)| !g.is_empty())
        .flat_map_iter(|(group, mut members)| {
            let capacity = CAPACITIES[group];
            // Identical body plans share a warp: same loop counts and index patterns.
            members.par_sort_by_cached_key(|&(_, i)| {
                let g = &pop.genomes[i];
                (g.node_count, g.muscle_count, plan_hash(pop, i), i)
            });
            // Long runs of one body plan get their own batch and kernel.
            let mut parts: Vec<(Option<Plan>, Members)> = Vec::new();
            let mut rest = Vec::new();
            let mut start = 0;
            if plan_batch == 0 || capacity >= 24 {
                // No run can get its own batch: skip comparing plans.
                rest = std::mem::take(&mut members);
            }
            while start < members.len() {
                let first = members[start].1;
                let end = start
                    + members[start..]
                        .iter()
                        .take_while(|&&(_, i)| same_plan(pop, first, i))
                        .count();
                if plan_batch > 0 && end - start >= plan_batch && capacity < 24 {
                    parts.push((Some(plan_of(pop, first)), members[start..end].to_vec()));
                } else {
                    rest.extend_from_slice(&members[start..end]);
                }
                start = end;
            }
            if !rest.is_empty() {
                parts.push((None, rest));
            }
            parts
                .into_iter()
                .map(move |(plan, members)| build_batch(pop, capacity, plan, &members))
        })
        .collect())
}

/// (unit position, population index) of the creatures in one batch.
type Members = Vec<(usize, usize)>;

/// Smallest run of one body plan in a unit that gets its own batch and
/// specialized kernel (`EVOLUTION_PLAN_BATCH`; 0, the default, turns it off).
/// Off because it measured no gain on mixed units: on 500k evolved 3M bodies
/// generic kernels ran 82k creatures/s, plan batches of at least 2,048
/// creatures 60k, 8,192 80k and 20,000 82k. Splitting a unit into smaller
/// dispatches costs what the specialized kernels save (+10 to 20% on a
/// single-plan population).
fn plan_batch_size() -> usize {
    std::env::var("EVOLUTION_PLAN_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Whether creatures `a` and `b` have the same skeleton and muscle
/// attachments, compared directly so a hash collision cannot mix plans.
fn same_plan(pop: &Population, a: usize, b: usize) -> bool {
    let (ga, gb) = (&pop.genomes[a], &pop.genomes[b]);
    if ga.node_count != gb.node_count
        || ga.bone_count != gb.bone_count
        || ga.muscle_count != gb.muscle_count
    {
        return false;
    }
    let bones =
        |g: &crate::evolution::Genome| &pop.bones[g.bone_start..g.bone_start + g.bone_count];
    let muscles = |g: &crate::evolution::Genome| {
        &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count]
    };
    bones(ga)
        .iter()
        .zip(bones(gb))
        .all(|(x, y)| x.a == y.a && x.b == y.b)
        && muscles(ga)
            .iter()
            .zip(muscles(gb))
            .all(|(x, y)| x.bone_a == y.bone_a && x.bone_b == y.bone_b)
}

/// Splits `all` into consecutive parts of the given sizes.
fn split_parts<T>(mut all: &mut [T], sizes: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    let mut out = Vec::new();
    for size in sizes {
        let (head, tail) = std::mem::take(&mut all).split_at_mut(size);
        out.push(head);
        all = tail;
    }
    out
}

/// Packs `members` into one batch. Each tile of `TILE` creatures owns its own
/// stretch of every buffer, so the tiles fill in parallel.
fn build_batch(
    pop: &Population,
    capacity: usize,
    plan: Option<Plan>,
    members: &[(usize, usize)],
) -> LaneBatch {
    let count = members.len();
    let tile_count = count.div_ceil(TILE);
    let mut tiles = Vec::with_capacity(tile_count);
    let mut muscle_len = 0usize;
    let mut bone_len = 0usize;
    for tile in members.chunks(TILE) {
        let max_muscles = tile
            .iter()
            .map(|&(_, i)| pop.genomes[i].muscle_count)
            .max()
            .unwrap_or(0);
        let max_bones = tile
            .iter()
            .map(|&(_, i)| pop.genomes[i].bone_count)
            .max()
            .unwrap_or(0);
        tiles.push([
            muscle_len as u32,
            bone_len as u32,
            max_muscles as u32,
            max_bones as u32,
        ]);
        muscle_len += max_muscles * MUSCLE_FIELDS * TILE;
        bone_len += max_bones * BONE_FIELDS * TILE;
    }
    let mut nodes = vec![Node::default(); count * capacity];
    let mut info = vec![[0u32; 4]; count];
    let mut muscles = vec![0f32; muscle_len.max(1)];
    let mut bones = vec![0f32; bone_len.max(1)];
    let node_parts: Vec<&mut [Node]> = nodes.chunks_mut((TILE * capacity).max(1)).collect();
    let info_parts: Vec<&mut [[u32; 4]]> = info.chunks_mut(TILE).collect();
    let muscle_parts = split_parts(
        &mut muscles[..muscle_len],
        tiles.iter().map(|t| t[2] as usize * MUSCLE_FIELDS * TILE),
    );
    let bone_parts = split_parts(
        &mut bones[..bone_len],
        tiles.iter().map(|t| t[3] as usize * BONE_FIELDS * TILE),
    );
    members
        .par_chunks(TILE)
        .zip(node_parts)
        .zip(info_parts)
        .zip(muscle_parts)
        .zip(bone_parts)
        .for_each(|((((tile, nodes), info), muscles), bones)| {
            // Reuse one scratch vector for every creature in this tile.
            let mut joints = vec![physics::Joint::FREE; capacity.saturating_sub(1)];
            for (lane, &(_, i)) in tile.iter().enumerate() {
                let g = &pop.genomes[i];
                let genes = &pop.nodes[g.node_start..g.node_start + g.node_count];
                let source_bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
                let node_start = lane * capacity;
                let body_state = &mut nodes[node_start..node_start + genes.len()];
                physics::body_into(genes, source_bones, body_state);
                info[lane] = [
                    g.node_count as u32,
                    g.bone_count as u32,
                    g.muscle_count as u32,
                    // Earthquake seed: the kernel derives this creature's bump
                    // phase and height jitter from it, so it never needs to
                    // compute the hash itself.
                    crate::physics::quake_hash(g.id),
                ];
                let body_state = &nodes[node_start..node_start + genes.len()];
                let joints = &mut joints[..source_bones.len()];
                physics::joints_from_body(genes, source_bones, body_state, joints);
                for (b, (bone, joint)) in source_bones.iter().zip(joints.iter()).enumerate() {
                    let field = b * BONE_FIELDS * TILE + lane;
                    // A free joint points its reference at its own pivot; the
                    // kernel skips it because its half range cosine is -1.
                    let reference = joint.reference.unwrap_or(bone.a as usize) as u32;
                    let values = [
                        f32::from_bits(bone.a | (bone.b << 8) | (reference << 16)),
                        bone.rest_length,
                        joint.center[0],
                        joint.center[1],
                        joint.half[0],
                        joint.half[1],
                        joint.child_share,
                        joint.child_mass,
                        joint.reference_mass,
                    ];
                    for (f, value) in values.into_iter().enumerate() {
                        bones[field + f * TILE] = value;
                    }
                }
                let source = &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
                for (m, muscle) in source.iter().enumerate() {
                    let bone_a = source_bones[muscle.bone_a as usize];
                    let bone_b = source_bones[muscle.bone_b as usize];
                    let field = m * MUSCLE_FIELDS * TILE + lane;
                    let values = [
                        f32::from_bits(
                            bone_a.a | (bone_a.b << 8) | (bone_b.a << 16) | (bone_b.b << 24),
                        ),
                        muscle.anchor_a,
                        muscle.anchor_b,
                        muscle_amplitude(muscle),
                        muscle.long,
                        1.0 / muscle.period,
                        muscle.phase,
                        muscle.duty,
                        muscle.stiffness,
                        1.0 / muscle.duty,
                        1.0 / (1.0 - muscle.duty),
                        f32::from_bits(muscle.sensor),
                        muscle.reset,
                        0.0,
                        1.0,
                    ];
                    for (f, value) in values.into_iter().enumerate() {
                        muscles[field + f * TILE] = value;
                    }
                }
            }
        });
    LaneBatch {
        capacity,
        plan,
        slots: members.iter().map(|&(slot, _)| slot).collect(),
        creatures: members.iter().map(|&(_, i)| i).collect(),
        nodes,
        info,
        tiles,
        muscles,
        muscle_fields: MUSCLE_FIELDS,
        bones,
        results: None,
    }
}

/// A body plan: the skeleton and muscle attachments as node numbers. Every
/// node index the kernel uses follows from it, so creatures of one plan can
/// run a kernel with those indices compiled in (`specialized_source`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Plan {
    pub nodes: u8,
    /// Per bone: pivot, child, and joint reference node (the pivot for a free
    /// joint, which the kernel skips).
    pub bones: Vec<[u8; 3]>,
    /// Per muscle: the endpoints of its two bones (a0, a1, b0, b1).
    pub muscles: Vec<[u8; 4]>,
}

/// Body plan of creature `index`.
pub fn plan_of(pop: &Population, index: usize) -> Plan {
    let g = &pop.genomes[index];
    let bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
    let muscles = &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
    Plan {
        nodes: g.node_count as u8,
        bones: bones
            .iter()
            .enumerate()
            .map(|(b, bone)| {
                let reference = physics::joint_reference(bones, b).unwrap_or(bone.a as usize);
                [bone.a as u8, bone.b as u8, reference as u8]
            })
            .collect(),
        muscles: muscles
            .iter()
            .map(|m| {
                let a = bones[m.bone_a as usize];
                let b = bones[m.bone_b as usize];
                [a.a as u8, a.b as u8, b.a as u8, b.b as u8]
            })
            .collect(),
    }
}

/// Kernel parameters for ticks `tick..tick + steps` of a `total`-tick trial
/// of `count` creatures in `capacity`-node buckets. Both GPU backends use it.
/// Length in `[f32; 2]` slots of one recorded frame of `batch`. Physics v1
/// records only the node positions (`capacity` slots). Physics v2 adds an
/// (energy, force) pair per muscle and a (normal, friction) contact force per
/// node after them: `2 * capacity + muscles` slots, the muscle count being the
/// batch's largest (a replay batch holds one creature).
pub fn frame_stride(batch: &LaneBatch) -> usize {
    if crate::physics2::enabled() {
        let muscles = batch.info.iter().map(|i| i[2] as usize).max().unwrap_or(0);
        2 * batch.capacity + muscles
    } else {
        batch.capacity
    }
}

pub fn launch_params(
    cfg: &crate::config::Config,
    capacity: usize,
    count: usize,
    tick: u32,
    steps: u32,
    total: u32,
) -> Params {
    let fidelity = cfg.fidelity();
    Params {
        tick,
        steps,
        stride: capacity as u32,
        count: count as u32,
        gravity: cfg.gravity,
        air: fidelity.air_per_step(cfg.air_retention),
        friction: cfg.ground_friction,
        ground: if cfg.ground { 1.0 } else { 0.0 },
        total_steps: total,
        terrain: crate::physics::terrain_amplitude(cfg.terrain),
        muscle_energy: cfg.muscle_energy,
        muscle_recovery: cfg.muscle_recovery,
        // A disabled ground ignores the slope effect, as on the CPU.
        slope: if cfg.ground { cfg.slope } else { 0.0 },
        wind: cfg.wind,
        // A disabled ground also ignores mud, gaps, hurdles, and the earthquake.
        mud: if cfg.ground { cfg.mud } else { 0.0 },
        gaps: if cfg.ground { cfg.gaps } else { 0.0 },
        hurdles: if cfg.ground { cfg.hurdles } else { 0.0 },
        quake: if cfg.ground { cfg.quake } else { 0.0 },
        screen_tick: cfg.screen.map_or(0, |screen| screen.tick(fidelity)),
        screen_bar: cfg.screen.map_or(f32::NEG_INFINITY, |screen| screen.bar),
    }
}

/// The CUDA C++ creature kernel (`shaders/physics_creature.cu`) for
/// `capacity`-node buckets. It prepends, as `#define` lines, the same
/// constants `base_source` writes into the WGSL kernel. With
/// `launch_bounds`, the kernel declares its block size and the compiler may
/// use up to 255 registers per thread. Without, NVRTC's `--maxrregcount`
/// caps them (launch bounds would override it).
pub fn cuda_source(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
    launch_bounds: bool,
) -> String {
    cuda_source_variant(capacity, workgroup, fidelity, launch_bounds, false)
}

/// The CUDA recording kernel, the counterpart of `record_source`: the
/// scoring kernel of `cuda_source` plus the frame output in an eighth
/// argument. It computes exactly what the scoring kernel computes.
pub fn cuda_record_source(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
    launch_bounds: bool,
) -> String {
    cuda_source_variant(capacity, workgroup, fidelity, launch_bounds, true)
}

fn cuda_source_variant(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
    launch_bounds: bool,
    record: bool,
) -> String {
    let limits = crate::physics::limits();
    // Large bodies loop to their runtime size, as in base_source.
    let large = capacity >= 24;
    let float = |value: f32| format!("{value:?}f");
    let (turn_cos, turn_tan) = fidelity.turn_limits();
    let defines = [
        ("WG", format!("{workgroup}u")),
        ("MAXN", format!("{capacity}u")),
        ("STRIDE", format!("{capacity}u")),
        ("SHAREDLEN", (capacity * workgroup as usize).to_string()),
        (
            "NODE_BOUND",
            (if large { "body_nodes" } else { "MAXN" }).into(),
        ),
        (
            "BONE_BOUND",
            (if large { "bone_count" } else { "MAXB" }).into(),
        ),
        (
            "UNROLL",
            (if large { "" } else { "_Pragma(\"unroll\")" }).into(),
        ),
        (
            "LAUNCH_BOUNDS",
            if launch_bounds {
                format!("__launch_bounds__({workgroup})")
            } else {
                String::new()
            },
        ),
        (
            "EXACT_COS",
            (if std::env::var_os("EVOLUTION_EXACT_COS").is_some() {
                "1"
            } else {
                "0"
            })
            .into(),
        ),
        ("MUSCLE_CAPACITY", float(limits.muscle_energy)),
        ("MUSCLE_RECOVERY", float(limits.muscle_recovery)),
        ("MAX_MUSCLE_FORCE", float(limits.muscle_force)),
        ("MAX_NODE_SPEED", float(limits.node_speed)),
        ("MAX_BONE_ANGULAR_SPEED", float(limits.bone_spin)),
        ("PLANTED_SPEED", float(physics::PLANTED_SPEED)),
        ("STANCE_GRIP", float(physics::stance_grip())),
        ("HEAD_SHAKE_LIMIT", float(physics::HEAD_SHAKE_LIMIT)),
        ("HEAD_SHAKE_WINDOW", float(physics::HEAD_SHAKE_WINDOW)),
        ("MUD_NORMAL", float(physics::MUD_NORMAL)),
        ("MUD_GRIP", float(physics::MUD_GRIP)),
        ("MUD_DRAG", float(physics::MUD_DRAG)),
        ("MUD_FULL_DEPTH", float(physics::MUD_FULL_DEPTH)),
        ("GAP_DEPTH", float(physics::GAP_DEPTH)),
        ("GAP_RUN", float(physics::GAP_RUN)),
        ("HURDLE_SPACING", float(physics::HURDLE_SPACING)),
        ("HURDLE_TOP", float(physics::HURDLE_TOP)),
        ("HURDLE_RUN", float(physics::HURDLE_RUN)),
        ("LIFT_CLEARANCE", "0.01f".into()),
        ("RECORD", (if record { "1" } else { "0" }).into()),
        (
            "BONE_SOLVE_ITERATIONS",
            format!("{}u", fidelity.bone_passes),
        ),
        (
            "VELOCITY_SOLVE_ITERATIONS",
            format!("{}u", fidelity.velocity_passes),
        ),
        ("RATE", format!("{:.1}f", fidelity.rate as f32)),
        ("SETTLE", format!("{}u", fidelity.settle())),
        ("SAMPLE", format!("{}u", fidelity.sample_interval())),
        (
            "JOINT_BREAK_COS",
            format!("{:.9}f", physics::JOINT_BREAK.cos()),
        ),
        (
            "JOINT_BREAK_SIN",
            format!("{:.9}f", physics::JOINT_BREAK.sin()),
        ),
        ("MAX_BONE_TURN_COS", format!("{turn_cos:.9}f")),
        ("MAX_BONE_TURN_TAN", format!("{turn_tan:.9}f")),
    ];
    let mut source = String::new();
    for (name, value) in defines {
        source.push_str(&format!("#define {name} {value}\n"));
    }
    source.push_str(include_str!("../shaders/physics_creature.cu"));
    source
}

/// Whether the v2 CUDA kernel of `capacity` nodes keeps its per-lane table in
/// local memory (through L1 and L2) instead of shared memory. Above 32 nodes
/// it does, because the table would not fit. `EVOLUTION_CUDA_TABLE_LOCAL=N`
/// (a developer diagnostic) moves the limit down to bodies above N nodes.
pub fn cuda_table_local(capacity: usize) -> bool {
    let above = std::env::var("EVOLUTION_CUDA_TABLE_LOCAL")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(32)
        .min(32);
    capacity > above
}

/// The physics v2 CUDA kernel (`shaders/physics2_creature.cu`) for
/// `capacity`-node buckets: the counterpart of `physics2::shader_source`,
/// with the same constants as `#define` lines. Bodies of up to 16 nodes
/// unroll their node and bone loops (private arrays in registers); bodies
/// above 32 nodes keep the per-lane table in local memory.
pub fn cuda_source2(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
    launch_bounds: bool,
) -> String {
    cuda_source2_variant(capacity, workgroup, fidelity, launch_bounds, false)
}

/// The physics v2 CUDA recording kernel: the scoring kernel of `cuda_source2`
/// plus the frame output in an eighth argument (the counterpart of
/// `physics2::record_source`).
pub fn cuda_record_source2(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
    launch_bounds: bool,
) -> String {
    cuda_source2_variant(capacity, workgroup, fidelity, launch_bounds, true)
}

fn cuda_source2_variant(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
    launch_bounds: bool,
    record: bool,
) -> String {
    use crate::physics2 as p2;
    let limits = crate::physics::limits();
    let float = |value: f32| format!("{value:?}f");
    let unroll = capacity <= 16;
    let defines = [
        ("WG", format!("{workgroup}u")),
        ("MAXN", format!("{capacity}u")),
        ("STRIDE", format!("{capacity}u")),
        ("MAXC", format!("{}u", capacity.min(p2::MAX_CONTACTS))),
        (
            "UNROLL",
            (if unroll { "_Pragma(\"unroll\")" } else { "" }).into(),
        ),
        (
            "TAB_LOCAL",
            (if cuda_table_local(capacity) { "1" } else { "0" }).into(),
        ),
        (
            "LAUNCH_BOUNDS",
            if launch_bounds {
                format!("__launch_bounds__({workgroup})")
            } else {
                String::new()
            },
        ),
        ("RECORD", (if record { "1" } else { "0" }).into()),
        ("MUSCLE_CAPACITY", float(limits.muscle_energy)),
        ("MUSCLE_RECOVERY", float(limits.muscle_recovery)),
        ("MAX_MUSCLE_FORCE", float(limits.muscle_force)),
        (
            "INV_JOINT_DAMPING",
            float(if p2::joint_damping() > 0.0 {
                1.0 / p2::joint_damping()
            } else {
                0.0
            }),
        ),
        ("LIMIT_HARDNESS", float(p2::LIMIT_HARDNESS)),
        ("JOINT_BREAK", float(physics::JOINT_BREAK)),
        ("MUD_NORMAL", float(physics::MUD_NORMAL)),
        ("MUD_GRIP", float(physics::MUD_GRIP)),
        ("MUD_DRAG", float(physics::MUD_DRAG)),
        ("MUD_FULL_DEPTH", float(physics::MUD_FULL_DEPTH)),
        ("SPIN_CAP", float(p2::SPIN_CAP)),
        ("INV_SPIN_CAP", float(1.0 / p2::SPIN_CAP)),
        ("SPIN_HARDNESS", float(p2::SPIN_HARDNESS)),
        ("PGS_SWEEPS", format!("{}u", p2::PGS_ITERATIONS)),
        ("PLANT_SWEEPS", format!("{}u", p2::PLANT_SWEEPS)),
        ("PLANT_ROUNDS", format!("{}u", p2::PLANT_ROUNDS)),
        ("WARM", "1".into()),
        ("PUSH_OUT", float(p2::PUSH_OUT)),
        ("HEAD_SHAKE_LIMIT", float(physics::HEAD_SHAKE_LIMIT)),
        ("HEAD_SHAKE_WINDOW", float(physics::HEAD_SHAKE_WINDOW)),
        ("CONTACT_SLACK", float(p2::CONTACT_SLACK)),
        ("LIFT_CLEARANCE", float(p2::LIFT_CLEARANCE)),
        ("GAP_DEPTH", float(physics::GAP_DEPTH)),
        ("GAP_RUN", float(physics::GAP_RUN)),
        ("HURDLE_SPACING", float(physics::HURDLE_SPACING)),
        ("HURDLE_TOP", float(physics::HURDLE_TOP)),
        ("HURDLE_RUN", float(physics::HURDLE_RUN)),
        ("RATE", format!("{:.1}f", fidelity.rate as f32)),
        ("SETTLE", format!("{}u", fidelity.settle())),
        ("SAMPLE", format!("{}u", fidelity.sample_interval())),
    ];
    let mut source = String::new();
    for (name, value) in defines {
        source.push_str(&format!("#define {name} {value}\n"));
    }
    source.push_str(include_str!("../shaders/physics2_creature.cu"));
    source
}

/// The creature kernel for `capacity`-node buckets.
pub fn shader_source(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
) -> String {
    base_source(capacity, capacity, workgroup, fidelity).replace("MUSCLE_LOOP", "muscle_count")
}

/// The creature kernel that also records every creature's trial for a
/// replay: node positions before each step and after the last, in binding 7
/// as `[creature][frame][node]` with the bucket's node stride. It computes
/// exactly what `shader_source` computes. Its only other change: a creature
/// keeps moving after its trial ends (limp after a fall), as the CPU replay
/// shows it, while its result stays the one at the end of the trial.
pub fn record_source(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
) -> String {
    base_source_from(recording_text(), capacity, capacity, workgroup, fidelity)
        .replace("MUSCLE_LOOP", "muscle_count")
}

/// The changes `record_source` makes to the kernel text, applied before the
/// constants are filled in. Each must match exactly once.
const RECORD_EDITS: [(&str, &str); 4] = [
    (
        "@group(0) @binding(6) var<storage, read> tile_info: array<vec4u>;\n",
        "@group(0) @binding(6) var<storage, read> tile_info: array<vec4u>;\n\
         @group(0) @binding(7) var<storage, read_write> frames: array<vec2f>;\n",
    ),
    (
        "    for (var s = 0u; s < p.steps; s++) {\n",
        "    // The result at the end of the trial; the body keeps moving after it.\n\
         var kept = metrics;\n\
         var ended = metrics.fall_time > 0.0 || metrics.screened > 0.0;\n\
         for (var s = 0u; s < p.steps; s++) {\n",
    ),
    (
        "        if metrics.fall_time > 0.0 || metrics.screened > 0.0 {\n            break;\n        }\n        let tick = p.tick + s;\n",
        "        let tick = p.tick + s;\n\
         if !ended && (metrics.fall_time > 0.0 || metrics.screened > 0.0) {\n\
             kept = metrics;\n\
             ended = true;\n\
         }\n\
         let frame = (creature * (p.total_steps + 1u) + tick) * STRIDE;\n\
         for (var j = 0u; j < MAXN; j++) {\n\
             if j >= body_nodes { break; }\n\
             frames[frame + j] = pos[node_k(j, lane)];\n\
         }\n",
    ),
    (
        "    results[creature] = metrics;\n",
        "    if !ended {\n\
             kept = metrics;\n\
         }\n\
         results[creature] = kept;\n\
         if p.tick + p.steps >= p.total_steps {\n\
             let frame = (creature * (p.total_steps + 1u) + p.total_steps) * STRIDE;\n\
             for (var j = 0u; j < MAXN; j++) {\n\
                 if j >= body_nodes { break; }\n\
                 frames[frame + j] = pos[node_k(j, lane)];\n\
             }\n\
         }\n",
    ),
];

/// The creature kernel for one body plan, in a bucket of node stride
/// `capacity`. Node, bone and muscle endpoints are constants, so every node
/// and bone loop unrolls with fixed indices and node state lives in
/// function-local arrays the compiler can keep in registers. It computes the
/// same expressions in the same order as `shader_source`.
pub fn specialized_source(
    plan: &Plan,
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
) -> String {
    let nodes = usize::from(plan.nodes);
    let mut source = base_source(nodes, capacity, workgroup, fidelity);
    let cut = |source: &mut String, begin: &str, end: &str, with: &str| {
        let a = source.find(begin).expect("kernel marker");
        let b = source.find(end).expect("kernel marker") + end.len();
        source.replace_range(a..b, with);
    };
    cut(&mut source, "// NODE-STATE-BEGIN", "// NODE-STATE-END", "");
    let mut helpers = String::from(
        "fn node_k(j: u32, lane: u32) -> u32 {\n    return j;\n}\n\
         fn node_of(k: u32) -> u32 {\n    return k;\n}\n",
    );
    let mut bone = String::from("fn plan_bone(j: u32) -> u32 {\n    switch j {\n");
    for (j, [a, b, q]) in plan.bones.iter().enumerate() {
        let word = u32::from(*a) | u32::from(*b) << 8 | u32::from(*q) << 16;
        bone.push_str(&format!("        case {j}u: {{ return {word}u; }}\n"));
    }
    bone.push_str("        default: { return 0u; }\n    }\n}\n");
    let mut muscle = String::from("fn plan_muscle(j: u32) -> u32 {\n    switch j {\n");
    for (j, ends) in plan.muscles.iter().enumerate() {
        let word = u32::from_le_bytes(*ends);
        muscle.push_str(&format!("        case {j}u: {{ return {word}u; }}\n"));
    }
    muscle.push_str("        default: { return 0u; }\n    }\n}\n");
    helpers.push_str(&bone);
    helpers.push_str(&muscle);
    helpers.push_str(
        "fn bone_a(packed_a: u32, j: u32) -> u32 {\n    return plan_bone(j) & 0xffu;\n}\n\
         fn bone_q(packed_a: u32, j: u32) -> u32 {\n    return (plan_bone(j) >> 16u) & 0xffu;\n}\n\
         fn bone_b(packed_b: u32, j: u32) -> u32 {\n    return (plan_bone(j) >> 8u) & 0xffu;\n}\n\
         fn muscle_node(packed: u32, j: u32, e: u32) -> u32 {\n    return (plan_muscle(j) >> (8u * e)) & 0xffu;\n}\n",
    );
    cut(&mut source, "// HELPERS-BEGIN", "// HELPERS-END", &helpers);
    let locals = "var pos: array<vec2f, MAXN>;\n    var vel: array<vec2f, MAXN>;\n    \
                  var old: array<vec2f, MAXN>;";
    let replace = |source: String, from: &str, to: &str| {
        assert!(source.contains(from), "kernel text {from:?} missing");
        source.replace(from, to)
    };
    let source = replace(source, "// LOCAL-NODE-STATE", locals);
    let source = replace(source, "let body_nodes = info.x;", "let body_nodes = MAXN;");
    let source = replace(source, "let bone_count = info.y;", "let bone_count = MAXB;");
    let source = replace(
        source,
        "let muscle_count = info.z;",
        &format!("let muscle_count = {}u;", plan.muscles.len()),
    );
    // `MAXN` is the plan's node count here, and `MAXB = MAXN - 1u` its bone
    // count: skeletons are trees. The driver unrolls the loops it chooses;
    // unrolling every loop in the text measured slower (register spills).
    source.replace("MUSCLE_LOOP", &format!("{}u", plan.muscles.len()))
}

fn base_source(
    capacity: usize,
    stride: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
) -> String {
    base_source_from(
        include_str!("../shaders/physics_creature.wgsl").to_owned(),
        capacity,
        stride,
        workgroup,
        fidelity,
    )
}

/// The kernel text with `RECORD_EDITS` applied.
fn recording_text() -> String {
    let mut source = include_str!("../shaders/physics_creature.wgsl").to_owned();
    for (from, to) in RECORD_EDITS {
        assert_eq!(source.matches(from).count(), 1, "record edit {from:?}");
        source = source.replace(from, to);
    }
    source
}

fn base_source_from(
    mut source: String,
    capacity: usize,
    stride: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
) -> String {
    if capacity >= 24 {
        // Constant bounds let compilers unroll every node and bone loop. That
        // keeps small bodies in registers, but RADV's compile time explodes for
        // large bodies, so those loops use the runtime size instead.
        source = source
            .replace("j < MAXN; j++)", "j < body_nodes; j++)")
            .replace("j < MAXB; j++)", "j < bone_count; j++)");
    }
    let (bone_passes, velocity_passes) = (fidelity.bone_passes, fidelity.velocity_passes);
    let limits = crate::physics::limits();
    let source = source
        .replace(
            "const MAX_MUSCLE_LENGTH_SPEED: f32 = 2.0;",
            &format!(
                "const MAX_MUSCLE_LENGTH_SPEED: f32 = {:?};",
                limits.muscle_speed
            ),
        )
        .replace(
            "const MUSCLE_CAPACITY: f32 = 15.0;",
            &format!("const MUSCLE_CAPACITY: f32 = {:?};", limits.muscle_energy),
        )
        .replace(
            "const MUSCLE_RECOVERY: f32 = 0.25;",
            &format!("const MUSCLE_RECOVERY: f32 = {:?};", limits.muscle_recovery),
        )
        .replace(
            "const MAX_MUSCLE_FORCE: f32 = 5.0;",
            &format!("const MAX_MUSCLE_FORCE: f32 = {:?};", limits.muscle_force),
        )
        .replace(
            "const PLANTED_SPEED: f32 = 0.01;",
            &format!(
                "const PLANTED_SPEED: f32 = {:?};",
                crate::physics::PLANTED_SPEED
            ),
        )
        .replace(
            "const STANCE_GRIP: f32 = 10.0;",
            &format!(
                "const STANCE_GRIP: f32 = {:?};",
                crate::physics::stance_grip()
            ),
        )
        .replace(
            "const MUD_NORMAL: f32 = 2.0;",
            &format!("const MUD_NORMAL: f32 = {:?};", crate::physics::MUD_NORMAL),
        )
        .replace(
            "const MUD_GRIP: f32 = 2.0;",
            &format!("const MUD_GRIP: f32 = {:?};", crate::physics::MUD_GRIP),
        )
        .replace(
            "const MUD_DRAG: f32 = 2.0;",
            &format!("const MUD_DRAG: f32 = {:?};", crate::physics::MUD_DRAG),
        )
        .replace(
            "const MUD_FULL_DEPTH: f32 = 0.1;",
            &format!(
                "const MUD_FULL_DEPTH: f32 = {:?};",
                crate::physics::MUD_FULL_DEPTH
            ),
        )
        .replace(
            "const GAP_DEPTH: f32 = 2.0;",
            &format!("const GAP_DEPTH: f32 = {:?};", crate::physics::GAP_DEPTH),
        )
        .replace(
            "const GAP_RUN: f32 = 0.15;",
            &format!("const GAP_RUN: f32 = {:?};", crate::physics::GAP_RUN),
        )
        .replace(
            "const HURDLE_SPACING: f32 = 3.0;",
            &format!(
                "const HURDLE_SPACING: f32 = {:?};",
                crate::physics::HURDLE_SPACING
            ),
        )
        .replace(
            "const HURDLE_TOP: f32 = 1.2;",
            &format!("const HURDLE_TOP: f32 = {:?};", crate::physics::HURDLE_TOP),
        )
        .replace(
            "const HURDLE_RUN: f32 = 0.2;",
            &format!("const HURDLE_RUN: f32 = {:?};", crate::physics::HURDLE_RUN),
        )
        .replace(
            "const HEAD_SHAKE_LIMIT: f32 = 78.4;",
            &format!(
                "const HEAD_SHAKE_LIMIT: f32 = {:?};",
                crate::physics::HEAD_SHAKE_LIMIT
            ),
        )
        .replace(
            "const HEAD_SHAKE_WINDOW: f32 = 0.1;",
            &format!(
                "const HEAD_SHAKE_WINDOW: f32 = {:?};",
                crate::physics::HEAD_SHAKE_WINDOW
            ),
        )
        .replace(
            "const MAX_NODE_SPEED: f32 = 5.0;",
            &format!("const MAX_NODE_SPEED: f32 = {:?};", limits.node_speed),
        )
        .replace(
            "const MAX_BONE_ANGULAR_SPEED: f32 = 15.0;",
            &format!(
                "const MAX_BONE_ANGULAR_SPEED: f32 = {:?};",
                limits.bone_spin
            ),
        )
        .replace(
            "const BONE_SOLVE_ITERATIONS: u32 = 8u;",
            &format!("const BONE_SOLVE_ITERATIONS: u32 = {bone_passes}u;"),
        )
        .replace(
            "const VELOCITY_SOLVE_ITERATIONS: u32 = 4u;",
            &format!("const VELOCITY_SOLVE_ITERATIONS: u32 = {velocity_passes}u;"),
        )
        .replace("PHYSICSRATE", &format!("{:.1}", fidelity.rate as f32))
        .replace("SETTLESTEPSu", &format!("{}u", fidelity.settle()))
        .replace(
            "SAMPLEINTERVALu",
            &format!("{}u", fidelity.sample_interval()),
        )
        .replace(
            "JOINTBREAKCOS",
            &format!("{:.9}", physics::JOINT_BREAK.cos()),
        )
        .replace(
            "JOINTBREAKSIN",
            &format!("{:.9}", physics::JOINT_BREAK.sin()),
        )
        .replace("TURNCOS", &format!("{:.9}", fidelity.turn_limits().0))
        .replace("TURNTAN", &format!("{:.9}", fidelity.turn_limits().1))
        .replace("SHAREDLEN", &(capacity * workgroup as usize).to_string())
        .replace("WGSIZEu", &format!("{workgroup}u"))
        .replace("WGSIZE", &workgroup.to_string())
        .replace("STRIDE", &format!("{stride}u"))
        .replace("MAXNODESu", &format!("{capacity}u"));
    crate::gpu::apply_fast_cos(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::Fidelity;

    /// The serial packer the parallel `build_batch` replaced.
    fn build_batch_reference(
        pop: &Population,
        capacity: usize,
        plan: Option<Plan>,
        members: &[(usize, usize)],
    ) -> LaneBatch {
        {
            {
                let count = members.len();
                let tile_count = count.div_ceil(TILE);
                let mut tiles = Vec::with_capacity(tile_count);
                let mut muscle_len = 0usize;
                let mut bone_len = 0usize;
                for tile in members.chunks(TILE) {
                    let max_muscles = tile
                        .iter()
                        .map(|&(_, i)| pop.genomes[i].muscle_count)
                        .max()
                        .unwrap_or(0);
                    let max_bones = tile
                        .iter()
                        .map(|&(_, i)| pop.genomes[i].bone_count)
                        .max()
                        .unwrap_or(0);
                    tiles.push([
                        muscle_len as u32,
                        bone_len as u32,
                        max_muscles as u32,
                        max_bones as u32,
                    ]);
                    muscle_len += max_muscles * MUSCLE_FIELDS * TILE;
                    bone_len += max_bones * BONE_FIELDS * TILE;
                }
                let mut nodes = vec![Node::default(); count * capacity];
                let mut info = Vec::with_capacity(count);
                let mut muscles = vec![0f32; muscle_len.max(1)];
                let mut bones = vec![0f32; bone_len.max(1)];
                // Reuse one scratch vector for every creature in this batch. The
                // old path allocated body nodes and joint constants per creature,
                // producing millions of tiny allocations during a generation.
                let mut joints = vec![physics::Joint::FREE; capacity.saturating_sub(1)];
                for (j, &(_, i)) in members.iter().enumerate() {
                    let g = &pop.genomes[i];
                    let genes = &pop.nodes[g.node_start..g.node_start + g.node_count];
                    let body_bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
                    let node_start = j * capacity;
                    let body_state = &mut nodes[node_start..node_start + genes.len()];
                    physics::body_into(genes, body_bones, body_state);
                    info.push([
                        g.node_count as u32,
                        g.bone_count as u32,
                        g.muscle_count as u32,
                        // Earthquake seed: the kernel derives this creature's bump
                        // phase and height jitter from it, so it never needs to
                        // compute the hash itself.
                        crate::physics::quake_hash(g.id),
                    ]);
                    let tile = tiles[j / TILE];
                    let lane = j % TILE;
                    let source_bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
                    let body_state = &nodes[node_start..node_start + genes.len()];
                    let joints = &mut joints[..source_bones.len()];
                    physics::joints_from_body(genes, source_bones, body_state, joints);
                    for (b, (bone, joint)) in source_bones.iter().zip(joints.iter()).enumerate() {
                        let field = tile[1] as usize + b * BONE_FIELDS * TILE + lane;
                        // A free joint points its reference at its own pivot; the
                        // kernel skips it because its half range cosine is -1.
                        let reference = joint.reference.unwrap_or(bone.a as usize) as u32;
                        let values = [
                            f32::from_bits(bone.a | (bone.b << 8) | (reference << 16)),
                            bone.rest_length,
                            joint.center[0],
                            joint.center[1],
                            joint.half[0],
                            joint.half[1],
                            joint.child_share,
                            joint.child_mass,
                            joint.reference_mass,
                        ];
                        for (f, value) in values.into_iter().enumerate() {
                            bones[field + f * TILE] = value;
                        }
                    }
                    let source = &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
                    for (m, muscle) in source.iter().enumerate() {
                        let bone_a = source_bones[muscle.bone_a as usize];
                        let bone_b = source_bones[muscle.bone_b as usize];
                        let field = tile[0] as usize + m * MUSCLE_FIELDS * TILE + lane;
                        let values = [
                            f32::from_bits(
                                bone_a.a | (bone_a.b << 8) | (bone_b.a << 16) | (bone_b.b << 24),
                            ),
                            muscle.anchor_a,
                            muscle.anchor_b,
                            muscle_amplitude(muscle),
                            muscle.long,
                            1.0 / muscle.period,
                            muscle.phase,
                            muscle.duty,
                            muscle.stiffness,
                            1.0 / muscle.duty,
                            1.0 / (1.0 - muscle.duty),
                            f32::from_bits(muscle.sensor),
                            muscle.reset,
                            0.0,
                            1.0,
                        ];
                        for (f, value) in values.into_iter().enumerate() {
                            muscles[field + f * TILE] = value;
                        }
                    }
                }
                LaneBatch {
                    capacity,
                    plan,
                    slots: members.iter().map(|&(slot, _)| slot).collect(),
                    creatures: members.iter().map(|&(_, i)| i).collect(),
                    nodes,
                    info,
                    tiles,
                    muscles,
                    muscle_fields: MUSCLE_FIELDS,
                    bones,
                    results: None,
                }
            }
        }
    }

    #[test]
    fn parallel_packing_matches_the_serial_packer() {
        let cfg = crate::config::Config {
            population: 3_000,
            random_seed: false,
            seed: 4,
            ..Default::default()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        for capacity in [8usize, 16] {
            let members: Vec<(usize, usize)> = (0..2_500)
                .map(|k| (k, (k * 31 + 7) % 3_000))
                .filter(|&(_, i)| pop.genomes[i].node_count <= capacity)
                .collect();
            assert!(members.len() > 2 * TILE);
            let a = build_batch(&pop, capacity, None, &members);
            let b = build_batch_reference(&pop, capacity, None, &members);
            assert_eq!(a.slots, b.slots);
            assert_eq!(a.creatures, b.creatures);
            assert_eq!(a.info, b.info);
            assert_eq!(a.tiles, b.tiles);
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&a.nodes),
                bytemuck::cast_slice::<_, u8>(&b.nodes)
            );
            assert_eq!(
                a.muscles.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.muscles.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                a.bones.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.bones.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn params_layout_matches_the_kernel_uniform() {
        // The WGSL `Params` struct must read every field at these offsets.
        // Adding or reordering a scalar can silently shift later fields.
        assert_eq!(std::mem::offset_of!(Params, gravity), 16);
        assert_eq!(std::mem::offset_of!(Params, total_steps), 32);
        assert_eq!(std::mem::offset_of!(Params, terrain), 36);
        assert_eq!(std::mem::offset_of!(Params, muscle_energy), 40);
        assert_eq!(std::mem::offset_of!(Params, muscle_recovery), 44);
        assert_eq!(std::mem::offset_of!(Params, slope), 48);
        assert_eq!(std::mem::offset_of!(Params, wind), 52);
        assert_eq!(std::mem::offset_of!(Params, mud), 56);
        assert_eq!(std::mem::offset_of!(Params, gaps), 60);
        assert_eq!(std::mem::offset_of!(Params, hurdles), 64);
        assert_eq!(std::mem::offset_of!(Params, quake), 68);
        assert_eq!(std::mem::offset_of!(Params, screen_tick), 72);
        assert_eq!(std::mem::offset_of!(Params, screen_bar), 76);
        assert_eq!(std::mem::size_of::<Params>(), 80);
    }

    #[test]
    fn packed_info_carries_the_quake_seed() {
        let cfg = crate::config::Config {
            population: 16,
            ..Default::default()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let indices: Vec<usize> = (0..pop.genomes.len()).collect();
        let batches = pack(&pop, &indices).unwrap();
        for batch in &batches {
            assert_eq!(batch.info.len(), batch.creatures.len());
            for (j, &i) in batch.creatures.iter().enumerate() {
                assert_eq!(
                    batch.info[j][3],
                    crate::physics::quake_hash(pop.genomes[i].id),
                    "packed creature {i} must carry its own quake seed"
                );
            }
        }
    }

    #[test]
    fn cuda_recording_kernel_differs_only_in_its_switch() {
        let fidelity = Fidelity::standard();
        let scoring = cuda_source(6, 128, fidelity, false);
        let recording = cuda_record_source(6, 128, fidelity, false);
        assert!(scoring.contains("#define RECORD 0\n"));
        assert_eq!(
            recording,
            scoring.replace("#define RECORD 0\n", "#define RECORD 1\n")
        );
    }

    #[test]
    fn recording_kernels_compile() {
        for fidelity in [Fidelity::standard(), Fidelity::fine()] {
            for capacity in [3, 8, 24, 64] {
                let source = record_source(capacity, 32, fidelity);
                assert!(source.contains("frames[frame + j]"));
                crate::vk_engine::spirv(&source).unwrap_or_else(|e| {
                    panic!("{capacity}-node recording kernel at {fidelity:?}: {e:#}")
                });
            }
        }
    }

    #[test]
    fn kernels_compile_for_standard_and_fine_physics() {
        for fidelity in [Fidelity::standard(), Fidelity::fine()] {
            for capacity in CAPACITIES {
                let source = shader_source(capacity, 32, fidelity);
                for constant in [
                    "PHYSICSRATE",
                    "SETTLESTEPSu",
                    "TURNCOS",
                    "TURNTAN",
                    "MAXNODESu",
                ] {
                    assert!(!source.contains(constant), "{constant} left unreplaced");
                }
                crate::vk_engine::spirv(&source)
                    .unwrap_or_else(|e| panic!("{capacity}-node kernel at {fidelity:?}: {e:#}"));
            }
        }
    }
}
