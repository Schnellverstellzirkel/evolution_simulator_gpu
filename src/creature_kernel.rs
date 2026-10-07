//! What the GPU kernel returns per creature (`GpuResult`), the packed batch
//! the engine uploads (`LaneBatch`, filled by `kernel::pack`), and the
//! layout of a recorded frame.
use crate::physics::Node;

/// One creature's trial as the kernel reports it: its fitness and behavior scores.
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
    /// Bodies have at most 32 nodes, so `lift_lo` holds them all; the CUDA
    /// kernel uses `lift_hi` for the rung trace.
    pub fn feet(&self) -> u32 {
        self.lift_lo.to_bits().count_ones()
    }
    /// The rung trace the CUDA kernel writes into the seven words
    /// the host reads for nothing else (`contact_hi`, `lift_hi`, `ground_hi`
    /// and the four gait working words), with the standard fitness.
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

/// What a trial looked like on its way, for the steps ladder
/// (R1 to R4): the distance at 1, 2.5, 5 and 10 s and the
/// early features at 1 and 2.5 s, as fp16 pairs (low half first):
///
/// - word 0: d60, d150 (m; a rung the trial did not reach holds the final
///   distance)
/// - word 1: d300, d600
/// - word 2: end code (u16), steps run (u16). The end code: bits 0 and 1 are
///   set when a screen or a rung stopped it, bit 4 it fell, bit 5 it failed,
///   bits 6 and 7 the early rung that stopped it (0 none, 1 R1, 2 R2; a stop
///   by the 5 s screen has neither), bits 8 to 10 and 11 to 13 the cadence
///   band (`BAND_COUNT` bands of the live gait frequency) at 1 and 2.5 s,
///   bit 14 an audit creature (every rule off)
/// - word 3: speed over the half second before 1 s and before 2.5 s (m/s)
/// - word 4: share of nodes that touched the ground by 1 s, mean muscle
///   energy store at 1 s
/// - word 5: the same two at 2.5 s
/// - word 6: head shake at 1 s and at 2.5 s (m/s^2)
///
/// Only the CUDA kernel writes it; the other engines leave working state in
/// these words, so `steps()` is 0 when the trace is absent. fp16 holds a
/// distance under 256 m to 0.125 m, enough for calibration; the fitness
/// stays f32.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RungTrace {
    pub words: [u32; 7],
    /// The standard trial's fitness (a confirmation may lower the score).
    pub fitness: f32,
}

impl RungTrace {
    /// Cadence bands of the early rungs: the live gait frequency in
    /// `BAND_COUNT` bins of 0 to 6 Hz, as the archive's cadence axis bins it.
    pub const BAND_COUNT: usize = 8;
    /// Steps of the kernel's rungs: 1, 2.5, 5 and 10 s at 60 Hz.
    pub const STEPS: [u32; 4] = [60, 150, 300, 600];
    fn half(&self, word: usize, high: bool) -> f32 {
        f16_to_f32((self.words[word] >> if high { 16 } else { 0 }) as u16)
    }
    /// Distance at rung `r` (0 to 3: 1, 2.5, 5, 10 s).
    pub fn distance(&self, r: usize) -> f32 {
        self.half(r / 2, r % 2 == 1)
    }
    /// End code: 3 when the screen stopped the trial, plus 16 for a fall and
    /// 32 for a failed trial.
    pub fn code(&self) -> u16 {
        self.words[2] as u16
    }
    /// Steps the trial ran, or 0 when no kernel wrote a trace.
    pub fn steps(&self) -> u32 {
        self.words[2] >> 16
    }
    /// Whether the creature fell during the trial.
    pub fn fell(&self) -> bool {
        self.code() & 16 != 0
    }
    /// The early rung that stopped the trial: 1 (R1), 2 (R2), or 0.
    pub fn stopped_by(&self) -> u8 {
        ((self.code() >> 6) & 3) as u8
    }
    /// Cadence band at rung `r` (0 or 1), from the live gait frequency.
    pub fn band(&self, r: usize) -> usize {
        ((self.code() >> (8 + 3 * r)) & 7) as usize
    }
    /// The creature was an audit creature.
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
    /// Head shake at rung `r` (0 or 1).
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
