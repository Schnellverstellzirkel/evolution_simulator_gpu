//! The data the host shares with the CUDA kernel in `shaders/creature.cu`.
//! `GpuResult` is what the kernel returns for each creature, and `RungTrace`
//! decodes the trace the kernel stores in seven of its words. `LaneBatch` is
//! the packed batch that `kernel::pack` fills and the engine uploads, and
//! `frame_stride` gives the size of a recorded frame. The half precision
//! conversions that the trace and the rungs use are here too.
use crate::physics::Node;

/// One creature's trial as the CUDA kernel reports it. The layout is that of
/// `Result` in `shaders/creature.cu`, and every field is an `f32`. The kernel
/// stores bit sets and packed words as `f32` bits, so read those with
/// `to_bits`. Seven fields hold the rung trace once the trial has ended
/// (`rung_trace`).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuResult {
    /// Horizontal distance (m) of the center of mass when the trial ended,
    /// whether in a fall, at a screen or rung stop, or at the last step.
    /// `evolution::FAILED` for a failed trial.
    pub fitness: f32,
    /// Node-steps on the ground: the sum over the steps run of the number of
    /// nodes touching the ground. `scheduler::to_metrics` turns it into a
    /// share.
    pub ground_contact: f32,
    /// Height range (m) of the mean node height over the steps run. Until the
    /// trial ends the kernel keeps the lowest height here.
    pub vertical_oscillation: f32,
    /// Gait frequency (Hz): half the turns of the mean node height per second
    /// the trial ran. Until the trial ends the kernel keeps the highest mean
    /// node height here.
    pub gait_frequency: f32,
    /// Gait counter: the mean node height at the last sample. When the trial
    /// ends the kernel puts rung trace word 3 here.
    pub previous_center_y: f32,
    /// Gait counter: the highest mean node height while the body rises, the
    /// lowest while it falls, since the last turn. Rung trace word 4 at the
    /// end.
    pub vertical_extremum: f32,
    /// Gait counter: 1 while the body rises, -1 while it falls, 0 until the
    /// height has changed by more than 0.5 mm between two samples. Rung trace
    /// word 5 at the end.
    pub vertical_trend: f32,
    /// Gait counter: the turns so far, which are changes of direction of
    /// more than 5 mm in the mean node height. Rung trace word 6 at the end.
    pub gait_turns: f32,
    /// Sum over the steps run of the body's height (m): the top of its
    /// highest node minus the bottom of its lowest. `scheduler::to_metrics`
    /// divides it by the steps for the mean height.
    pub height_sum: f32,
    /// Bits of the nodes that touched the ground, node `i` in bit `i`.
    pub contact_lo: f32,
    /// Rung trace word 0: the distances at 1 s and 2.5 s. Bodies have at most
    /// 32 nodes, so no contact bit lives here.
    pub contact_hi: f32,
    /// Bits of touching nodes that later lifted clear of the ground again.
    pub lift_lo: f32,
    /// Rung trace word 1: the distances at 5 s and 10 s.
    pub lift_hi: f32,
    /// Bits of the nodes on the ground after the last step, the set the
    /// kernel compares the next step with to find touchdowns for sensor
    /// muscles.
    pub ground_lo: f32,
    /// Rung trace word 2: the end code and the steps run.
    pub ground_hi: f32,
    /// Seconds into the trial when it ended in a fall, or 0 if it did not.
    /// The kernel ends a trial like a fall when
    ///
    /// - the head tips below its neck base,
    /// - a joint breaks (`physics::JOINT_BREAK`),
    /// - the head shakes past `physics::HEAD_SHAKE_LIMIT`, or
    /// - the trial fails, because a node position is not finite or lies
    ///   beyond 1e6 m.
    ///
    /// Fitness is the distance at that moment, or `evolution::FAILED` for a
    /// failed trial.
    pub fall_time: f32,
    /// Running mean of the head's acceleration (m/s^2) over about
    /// `physics::HEAD_SHAKE_WINDOW` seconds. It stops changing when the trial
    /// ends. A trial ends in a fall when it passes `physics::HEAD_SHAKE_LIMIT`.
    pub head_shake: f32,
    /// Distance (m) at the screen step. A trial that ended earlier, in a fall
    /// or at an early rung, gives its distance there. It is 0 for a trial
    /// without a screen that did not fall. The experiment sets the next
    /// generation's screen bar from these.
    pub screen_x: f32,
    /// Seconds into the trial when the screen or an early rung (`rungs`)
    /// stopped the creature, or 0. Its fitness is the distance there and its
    /// behavior totals end there.
    pub screened: f32,
}
impl GpuResult {
    /// Number of feet: nodes that touched the ground and lifted off again.
    /// A node dragged along the ground never lifts, so it is not a foot.
    /// Bodies have at most 32 nodes, so `lift_lo` holds them all. The CUDA
    /// kernel uses `lift_hi` for the rung trace.
    pub fn feet(&self) -> u32 {
        self.lift_lo.to_bits().count_ones()
    }
    /// The rung trace the kernel leaves in seven result words: `contact_hi`,
    /// `lift_hi`, `ground_hi`, `previous_center_y`, `vertical_extremum`,
    /// `vertical_trend` and `gait_turns`, in that order. The host reads these
    /// words for nothing else. The trace also carries this result's fitness.
    pub fn rung_trace(&self) -> RungTrace {
        RungTrace {
            words: [
                self.contact_hi,
                self.lift_hi,
                self.ground_hi,
                self.previous_center_y,
                self.vertical_extremum,
                self.vertical_trend,
                self.gait_turns,
            ]
            .map(f32::to_bits),
            fitness: self.fitness,
        }
    }
}

/// What a trial looked like on its way, for the steps ladder: the distance
/// at 1, 2.5, 5 and 10 s (the rungs R1 to R4) and the early features at 1 and
/// 2.5 s. The kernel stores it in seven words. Most hold two fp16 values, the
/// low half first:
///
/// - word 0: d60 and d150, the distances (m) at 1 s and 2.5 s
/// - word 1: d300 and d600, the distances at 5 s and 10 s
/// - word 2: the end code (u16), then the steps run (u16)
/// - word 3: the speed (m/s) over the half second before 1 s, then before
///   2.5 s
/// - word 4: at 1 s, the share of nodes that touched the ground, then the
///   mean muscle energy store
/// - word 5: the same two at 2.5 s
/// - word 6: the head shake (m/s^2) at 1 s, then at 2.5 s
///
/// A distance for a rung that the trial did not reach holds the final
/// distance. The bits of the end code are:
///
/// - bits 0 and 1: both set when a screen or an early rung stopped the trial
/// - bit 4: the trial ended in a fall (see `GpuResult::fall_time`), which
///   includes a failed trial
/// - bit 5: the trial failed
/// - bits 6 and 7: the early rung that stopped it, 0 for none, 1 for R1 and
///   2 for R2 (a stop by the 5 s screen sets neither)
/// - bits 8 to 10 and 11 to 13: the cadence band at 1 s and at 2.5 s
///   (`BAND_COUNT` bands of the live gait frequency)
/// - bit 14: an audit creature, which runs with every rule off
///
/// Only the CUDA kernel writes it, so `steps()` is 0 for a result that did
/// not come from the kernel. fp16 holds a distance under 256 m to within
/// 0.125 m, which is enough to fit the rungs. The fitness stays f32.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RungTrace {
    /// The seven words, laid out as above.
    pub words: [u32; 7],
    /// The standard trial's fitness (a confirmation may lower the score).
    pub fitness: f32,
}

impl RungTrace {
    /// Number of cadence bands the kernel files a trial under at each early
    /// rung: bins of the live gait frequency over 0 to 6 Hz, as the archive's
    /// cadence axis bins it. The top band also takes anything above 6 Hz.
    pub const BAND_COUNT: usize = 8;
    /// Steps of the kernel's rungs: 1, 2.5, 5 and 10 s at 60 Hz.
    pub const STEPS: [u32; 4] = [60, 150, 300, 600];
    /// The fp16 value in the low or the high half of word `word`.
    fn half(&self, word: usize, high: bool) -> f32 {
        f16_to_f32((self.words[word] >> if high { 16 } else { 0 }) as u16)
    }
    /// Distance (m) at rung `r` (0 to 3: 1, 2.5, 5, 10 s).
    pub fn distance(&self, r: usize) -> f32 {
        self.half(r / 2, r % 2 == 1)
    }
    /// The end code, the low half of word 2. The type's doc lists its bits.
    pub fn code(&self) -> u16 {
        self.words[2] as u16
    }
    /// Steps the trial ran, or 0 when the kernel wrote no trace.
    pub fn steps(&self) -> u32 {
        self.words[2] >> 16
    }
    /// Whether the trial ended in a fall. A broken joint, a head shake past
    /// its limit and a failed trial count as falls (`GpuResult::fall_time`).
    pub fn fell(&self) -> bool {
        self.code() & 16 != 0
    }
    /// The early rung that stopped the trial: 1 (R1), 2 (R2), or 0 for none.
    /// A stop by the 5 s screen also gives 0.
    pub fn stopped_by(&self) -> u8 {
        ((self.code() >> 6) & 3) as u8
    }
    /// Cadence band at rung `r` (0 or 1), from the live gait frequency.
    pub fn band(&self, r: usize) -> usize {
        ((self.code() >> (8 + 3 * r)) & 7) as usize
    }
    /// Whether the creature was an audit creature (`rungs::AUDIT`), which
    /// runs with every rule off.
    pub fn audit(&self) -> bool {
        self.code() & (1 << 14) != 0
    }
    /// Speed (m/s) over the half second before rung `r` (0 or 1).
    pub fn speed(&self, r: usize) -> f32 {
        self.half(3, r == 1)
    }
    /// Share of nodes that touched the ground by rung `r` (0 or 1).
    pub fn touched(&self, r: usize) -> f32 {
        self.half(4 + r, false)
    }
    /// Mean muscle energy store at rung `r` (0 or 1).
    pub fn energy(&self, r: usize) -> f32 {
        self.half(4 + r, true)
    }
    /// Head shake (m/s^2) at rung `r` (0 or 1).
    pub fn head_shake(&self, r: usize) -> f32 {
        self.half(6, r == 1)
    }
}

/// IEEE half to single precision.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exponent = ((h >> 10) & 0x1f) as u32;
    let mantissa = (h & 0x3ff) as u32;
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalize.
            let shift = mantissa.leading_zeros() - 21;
            let mantissa = (mantissa << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (mantissa << 13)
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, _) => sign | 0x7fc0_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}
/// Single to IEEE half precision, rounded to nearest even, for the dump files.
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut out = m >> shift;
        if rest > half || (rest == half && out & 1 == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    let rest = mantissa & 0x1fff;
    let mut out = ((e as u32) << 10) | (mantissa >> 13);
    if rest > 0x1000 || (rest == 0x1000 && out & 1 == 1) {
        out += 1;
    }
    sign | out as u16
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
    /// Fields per muscle in `muscles` (`kernel::MUSCLE_FIELDS`).
    pub muscle_fields: usize,
    pub bones: Vec<f32>,
    /// Behavior totals to resume from; `None` starts from zero.
    pub results: Option<Vec<GpuResult>>,
    /// The CUDA kernel's records (`kernel::pack`); the
    /// per-lane fields above are then empty.
    pub wave: Option<crate::kernel::WavePack>,
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

#[cfg(test)]
mod half_tests {
    use super::*;
    #[test]
    fn halves_round_trip() {
        for h in 0..=u16::MAX {
            let v = f16_to_f32(h);
            if v.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(v), h, "{h:#06x} {v}");
        }
        assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(-1e20)), f32::NEG_INFINITY);
        assert_eq!(f16_to_f32(f32_to_f16(12.3)), 12.296875);
    }
}
