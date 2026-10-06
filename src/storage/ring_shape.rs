use super::*;

/// The shape of the ring: creatures per block and blocks in flight. It is
/// chosen when an experiment starts, from the engine's rate and the host's
/// time per block (`RingShape::size`), saved with the experiment and written
/// into every generation's statistics. It never follows the rate while the
/// experiment runs: how many blocks were absorbed before a child is bred
/// decides its parents, so a shape that followed the rate would give one
/// seed a different search on every run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingShape {
    /// Creatures per block.
    pub block: usize,
    /// Blocks in the ring.
    pub blocks: usize,
}

/// What the ring is sized from.
#[derive(Clone, Copy, Debug)]
pub struct RingTimes {
    /// Standard creatures per second the engines finish.
    pub rate: f64,
    /// 95th percentile of the host's time per block: from the moment a block
    /// could be absorbed until it is bred again and queued.
    pub chain: f64,
    /// The longest such time of a block that ended a generation, over the
    /// last three generations.
    pub boundary: f64,
}

impl Default for RingTimes {
    /// Before anything is measured: the scheduler's first rate estimate for
    /// the RTX 4060, and a host time per block above `CONFIRM_LIMIT`, so a
    /// new game gets the ring of `RingShape::LEGACY`. A block that sets
    /// island records waits for its confirmation trials, so its host time is
    /// one or more round trips through the GPU: the p95 was 1.4 to 3.6 s and
    /// the boundary 0.14 to 1.9 s over generations 0 to 7 on a GPU shared
    /// with other runs (worker_rate, 2026-09-30).
    fn default() -> Self {
        Self {
            rate: 180_000.0,
            chain: 1.0,
            boundary: 0.35,
        }
    }
}

impl RingShape {
    /// The ring of 8 blocks of 196,608 creatures, 1.6M in flight. Blocks
    /// wait on confirmation round trips of 1 to 6 s, and more blocks in
    /// flight keep the GPU fed meanwhile: at 3M per generation 8 blocks ran
    /// 336k to 507k creatures/s in generations 1 to 10, 6 blocks 253k to
    /// 550k and 4 blocks 155k to 424k (worker_rate, seed 38, 2026-10-02),
    /// at a peak RSS of 6.2 GB.
    pub const LEGACY: RingShape = RingShape {
        block: 196_608,
        blocks: 8,
    };
    /// Host time per block (seconds) at or below which the sized ring is
    /// used. A block asks for about 40 confirmation trials whatever its
    /// size, so the shorter the blocks the more confirmation round trips a
    /// generation waits for, and a ring of 1 s or less starves the GPU while
    /// a round trip takes longer than the ring holds. Until confirmations
    /// have slots of their own and the measured host time falls below this,
    /// the ring is `LEGACY`.
    pub const CONFIRM_LIMIT: f64 = 0.2;
    /// GPU seconds of work in one block.
    pub const BLOCK_SECONDS: f64 = 0.05;
    /// Smallest and largest block.
    pub const MIN_BLOCK: usize = 32_768;
    pub const MAX_BLOCK: usize = 262_144;
    /// Host times per block the ring holds, so a slow block does not leave
    /// the GPU without work.
    pub const CHAIN_BLOCKS: f64 = 5.0;
    /// Shortest and longest ring in seconds of GPU work. A world change
    /// throws away at most the longest.
    pub const MIN_SECONDS: f64 = 0.3;
    pub const MAX_SECONDS: f64 = 1.0;

    /// While the p95 host time per block is above `CONFIRM_LIMIT` the ring is
    /// `LEGACY`. Otherwise block = 50 ms of GPU work between 32k and 256k
    /// creatures (a multiple of 4,096). Ring = 5 host times per block, or the generation boundary
    /// plus 2 blocks when that is longer, kept between 0.3 s and 1 s of GPU
    /// work, and at least 2 blocks so the GPU runs one while the host
    /// absorbs another.
    pub fn size(times: &RingTimes) -> Self {
        if times.chain.is_nan() || times.chain > Self::CONFIRM_LIMIT {
            return Self::LEGACY;
        }
        let rate = if times.rate.is_finite() && times.rate > 0.0 {
            times.rate
        } else {
            RingTimes::default().rate
        };
        let finite = |x: f64| if x.is_finite() { x.max(0.0) } else { 0.0 };
        let block = ((rate * Self::BLOCK_SECONDS) as usize)
            .next_multiple_of(4096)
            .clamp(Self::MIN_BLOCK, Self::MAX_BLOCK);
        let block_seconds = block as f64 / rate;
        let seconds = (Self::CHAIN_BLOCKS * finite(times.chain))
            .max(finite(times.boundary) + 2.0 * block_seconds)
            .clamp(Self::MIN_SECONDS, Self::MAX_SECONDS);
        // Whole blocks that cover the ring's seconds, one fewer when that
        // would pass the longest ring.
        let mut blocks = (seconds / block_seconds - 1e-9).ceil().max(1.0) as usize;
        if blocks as f64 * block_seconds > Self::MAX_SECONDS + 1e-9 {
            blocks -= 1;
        }
        let blocks = blocks.max(2);
        Self { block, blocks }
    }
    /// Ring slots for a generation of `population` evaluations.
    pub fn len(&self, population: usize) -> usize {
        population.clamp(1, self.block * self.blocks)
    }
    /// First slot and length of each block of a ring of `len` slots: the
    /// ring's blocks, fewer and smaller for a small population.
    pub(super) fn ranges(&self, len: usize) -> Vec<(usize, usize)> {
        let size = len.div_ceil(self.blocks).max(1);
        (0..len)
            .step_by(size)
            .map(|first| (first, size.min(len - first)))
            .collect()
    }
    /// Seconds of GPU work the ring holds at `rate` creatures per second.
    pub fn seconds(&self, population: usize, rate: f64) -> f64 {
        self.len(population) as f64 / rate.max(1.0)
    }
}

impl Default for RingShape {
    fn default() -> Self {
        Self::size(&RingTimes::default())
    }
}

#[cfg(test)]
mod ring_shape_tests {
    use super::*;

    fn times(rate: f64, chain: f64, boundary: f64) -> RingTimes {
        RingTimes {
            rate,
            chain,
            boundary,
        }
    }

    #[test]
    fn a_block_is_50_ms_of_gpu_work_between_32k_and_256k() {
        assert_eq!(RingShape::size(&times(167_000.0, 0.0, 0.0)).block, 32_768);
        assert_eq!(RingShape::size(&times(2e6, 0.0, 0.0)).block, 102_400);
        assert_eq!(RingShape::size(&times(1e8, 0.0, 0.0)).block, 262_144);
    }

    #[test]
    fn the_ring_holds_5_host_times_or_the_boundary_within_03_to_1_s() {
        let seconds = |t: RingTimes| {
            let r = RingShape::size(&t);
            (r.block * r.blocks) as f64 / t.rate
        };
        // Fast host: the shortest ring, rounded up to whole blocks.
        let s = seconds(times(2e6, 0.01, 0.0));
        assert!((0.3..0.36).contains(&s), "{s}");
        // Five host times.
        let s = seconds(times(2e6, 0.12, 0.0));
        assert!((0.6..0.66).contains(&s), "{s}");
        // The boundary plus two blocks.
        let s = seconds(times(2e6, 0.01, 0.5));
        assert!((0.6..0.66).contains(&s), "{s}");
        // Never past 1 s, and at least 2 blocks.
        let s = seconds(times(2e6, 0.2, 5.0));
        assert!((0.95..=1.0).contains(&s), "{s}");
        assert_eq!(RingShape::size(&times(40_000.0, 0.2, 5.0)).blocks, 2);
        // Today's rate with a fast host: 5 blocks of 32k, just under 1 s.
        let today = RingShape::size(&times(167_000.0, 0.2, 0.3));
        assert_eq!((today.block, today.blocks), (32_768, 5));
    }

    #[test]
    fn a_slow_confirmation_round_trip_keeps_the_legacy_ring() {
        for rate in [40_000.0, 167_000.0, 2e6] {
            for chain in [0.21, 1.0, 5.0, f64::NAN] {
                assert_eq!(RingShape::size(&times(rate, chain, 0.3)), RingShape::LEGACY);
            }
        }
        // Nothing measured.
        assert_eq!(RingShape::default(), RingShape::LEGACY);
    }

    #[test]
    fn a_small_population_splits_into_the_rings_blocks() {
        let ring = RingShape {
            block: 32_768,
            blocks: 5,
        };
        assert_eq!(ring.len(100), 100);
        assert_eq!(ring.len(3_000_000), 163_840);
        let ranges = ring.ranges(100);
        assert_eq!(ranges.len(), 5);
        assert_eq!(ranges.iter().map(|r| r.1).sum::<usize>(), 100);
    }
}
