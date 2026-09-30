//! What the GPU kernel returns per creature (`GpuResult`), the packed batch
//! the engine uploads (`LaneBatch`, filled by `warp_kernel::pack`), and the
//! layout of a recorded frame.
use crate::physics::Node;

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

/// One group of creatures, ready for upload.
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
    /// Fields per muscle in `muscles` (`warp_kernel::MUSCLE_FIELDS`).
    pub muscle_fields: usize,
    pub bones: Vec<f32>,
    /// Behavior totals to resume from; `None` starts from zero.
    pub results: Option<Vec<GpuResult>>,
    /// The lane-group CUDA kernel's records (`warp_kernel::pack`); the
    /// per-lane fields above are then empty.
    pub wave: Option<crate::warp_kernel::WavePack>,
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
