//! Data layout of the one-creature-per-lane GPU kernels of physics v2
//! (`shaders/physics2_creature.wgsl` and `.cu`, packed by `physics2::pack`).
//!
//! Creatures are grouped by padded node capacity. Inside a group they are
//! sorted by body size so the 32 creatures that share a warp run similar loop
//! counts. Muscles and bones are packed per 32-creature tile as
//! `[item][field][lane]`, which makes every warp load one coalesced line.
use crate::physics::Node;

pub const TILE: usize = 32;
/// Per bone: pivot node, length, joint range, the child node's mass, radius
/// and friction (see `shaders/physics2_creature.wgsl`).
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
    /// Water line height (m) above the flat ground; 0.0 is dry.
    pub water: f32,
    /// Friction lost on ice patches (share of the calm friction); 0.0 is none.
    pub patches: f32,
    /// Room for the next effects; keeps the struct a multiple of 16 bytes.
    pub spare: [f32; 2],
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
    /// Position of each packed creature within the caller's index slice.
    pub slots: Vec<usize>,
    /// Population index of each packed creature.
    pub creatures: Vec<usize>,
    pub nodes: Vec<Node>,
    pub info: Vec<[u32; 4]>,
    pub tiles: Vec<[u32; 4]>,
    pub muscles: Vec<f32>,
    /// Fields per muscle in `muscles` (`physics2::MUSCLE_FIELDS`).
    pub muscle_fields: usize,
    pub bones: Vec<f32>,
    /// Behavior totals to resume from, for a batch that continues a trial
    /// (`repack`); a fresh batch starts from zero.
    pub results: Option<Vec<GpuResult>>,
    /// The lane-group CUDA kernel's records (`warp_kernel::pack`); the
    /// per-lane fields above are then empty.
    pub wave: Option<crate::warp_kernel::WavePack>,
}

impl LaneBatch {
    /// Frees the node state, muscle buffer and resumed totals once the GPU
    /// holds them: a later segment repacks from the state it reads back, and
    /// the bodies (`bones`, `info`, `tiles`) stay for that.
    pub fn release_uploaded(&mut self) {
        self.nodes = Vec::new();
        self.muscles = Vec::new();
        self.results = None;
        self.wave = None;
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
            slots: keep.iter().map(|&j| self.slots[j]).collect(),
            creatures: keep.iter().map(|&j| self.creatures[j]).collect(),
            nodes: new_nodes,
            info: keep.iter().map(|&j| self.info[j]).collect(),
            tiles,
            muscles: new_muscles,
            muscle_fields: self.muscle_fields,
            bones: new_bones,
            results: Some(keep.iter().map(|&j| results[j]).collect()),
            wave: None,
        }
    }
}

pub fn capacity_index(nodes: usize) -> usize {
    CAPACITIES
        .iter()
        .position(|&c| nodes <= c)
        .unwrap_or(CAPACITIES.len() - 1)
}

/// Length in `[f32; 2]` slots of one recorded frame of `batch`: the node
/// positions (`capacity` slots), then an (energy, force) pair per muscle and a
/// (normal, friction) contact force per node, and last the broken joints:
/// `2 * capacity + muscles + 1` slots, the muscle count being the batch's
/// largest (a replay batch holds one creature). The last slot holds the bits
/// of the bones whose joint is past its break angle (the kernel's rule), bones
/// 0 to 31 in the first word and 32 to 63 in the second, as `f32` bits.
pub fn frame_stride(batch: &LaneBatch) -> usize {
    let muscles = batch.info.iter().map(|i| i[2] as usize).max().unwrap_or(0);
    2 * batch.capacity + muscles + 1
}

/// Kernel parameters for ticks `tick..tick + steps` of a `total`-tick trial
/// of `count` creatures in `capacity`-node buckets. Both GPU backends use it.
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
        water: cfg.water,
        // A disabled ground has no ice.
        patches: if cfg.ground { cfg.patches } else { 0.0 },
        spare: [0.0; 2],
    }
}
