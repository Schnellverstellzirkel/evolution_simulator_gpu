//! The stage log (`EVOLUTION_STAGE_LOG`) and the host time per ring block
//! that sizes the next ring.

use std::{io::Write, time::Instant};

pub(super) struct StageLog {
    file: std::fs::File,
    started: Instant,
    seconds: [f64; 3],
    /// Scheduler totals at the last row: confirmation trials submitted and
    /// their busy seconds, device busy seconds, device idle seconds.
    totals: [f64; 4],
    /// Seconds engine threads waited for kernels, at the last row.
    kernel_wait: f64,
    /// Lane-steps per lane class at the last row.
    lane_steps: [u64; 4],
    /// Device idle seconds at the last absorbed block, and the most that
    /// passed between two absorbed blocks this generation.
    idle_at_block: f64,
    starved_block: f64,
    /// Creatures a world change threw away this generation.
    pub(super) discarded: usize,
}
impl StageLog {
    /// Opens the stage log file from the `EVOLUTION_STAGE_LOG` environment variable.
    pub(super) fn open() -> Option<Self> {
        let path = std::env::var_os("EVOLUTION_STAGE_LOG")?;
        let mut file = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => file,
            Err(err) => {
                eprintln!("Stage log {} unavailable: {err:#}", path.to_string_lossy());
                return None;
            }
        };
        if file.metadata().map(|m| m.len()).unwrap_or(1) == 0 {
            let _ = writeln!(
                file,
                "generation,evaluation_seconds,archive_seconds,breeding_seconds,end_to_end_creatures_per_second,confirmations,confirmation_busy_seconds,device_busy_seconds,device_idle_seconds,mean_nodes,share_over_8_nodes,ring_block,ring_blocks,chain_p95_seconds,boundary_seconds,starved_block_max_seconds,lane_steps_4,lane_steps_8,lane_steps_16,lane_steps_32,world_change_discarded,kernel_wait_seconds,steps_per_creature,audit_rows,rung1_stop_share,rung2_stop_share,rung3_stop_share,rung1_entrant_misses_per_10k,rung2_entrant_misses_per_10k,rung1_extra_misses_per_10k,rung2_extra_misses_per_10k,audit_top1_kept,audit_top10_kept,screen_top1_kept,screen_top10_kept,rungs_armed,bands_off"
            );
        }
        Some(Self {
            file,
            started: Instant::now(),
            seconds: [0.0; 3],
            totals: [0.0; 4],
            kernel_wait: 0.0,
            lane_steps: [0; 4],
            idle_at_block: 0.0,
            starved_block: 0.0,
            discarded: 0,
        })
    }
    pub(super) fn add(&mut self, stage: usize, seconds: f64) {
        self.seconds[stage] += seconds;
    }
    pub(super) fn reset(&mut self) {
        self.started = Instant::now();
        self.seconds = [0.0; 3];
        self.starved_block = 0.0;
        self.discarded = 0;
    }
    /// A block was absorbed: the GPU idle time since the last one.
    pub(super) fn block(&mut self, idle: f64) {
        self.starved_block = self.starved_block.max(idle - self.idle_at_block);
        self.idle_at_block = idle;
    }
    /// A generation of `e` ended: its row, with the node counts of the
    /// genomes in the ring.
    pub(super) fn write_generation(
        &mut self,
        e: &crate::storage::Experiment,
        sched: &crate::scheduler::Scheduler,
        ring_meter: &RingMeter,
    ) {
        let genomes = e.blocks.iter().flat_map(|b| &b.population.genomes);
        let count = e.ring_len().max(1) as f64;
        let nodes = [
            genomes.clone().map(|g| g.node_count as f64).sum::<f64>() / count,
            genomes.filter(|g| g.node_count > 8).count() as f64 / count,
        ];
        self.write_row(
            e.generation.saturating_sub(1),
            e.config.population,
            Some(sched),
            nodes,
            e.ring,
            ring_meter,
            e.rungs.last(),
        );
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_row(
        &mut self,
        generation: u32,
        population: usize,
        sched: Option<&crate::scheduler::Scheduler>,
        nodes: [f64; 2],
        ring: crate::storage::RingShape,
        meter: &RingMeter,
        rungs: &crate::rungs::Report,
    ) {
        let seconds = self.started.elapsed().as_secs_f64().max(1e-9);
        let totals = sched.map_or([0.0; 4], |s| {
            [
                s.confirms_submitted as f64,
                s.confirm_busy_seconds,
                s.devices.iter().map(|d| d.busy_seconds).sum(),
                s.devices.iter().map(|d| d.idle_seconds).sum(),
            ]
        });
        let delta: [f64; 4] = std::array::from_fn(|k| totals[k] - self.totals[k]);
        self.totals = totals;
        let kernel_wait = crate::cuda_engine::kernel_wait_seconds();
        let kernel_wait_delta = kernel_wait - self.kernel_wait;
        self.kernel_wait = kernel_wait;
        let lane_totals = sched.map_or([0; 4], |s| s.lane_steps);
        let lanes: [u64; 4] =
            std::array::from_fn(|k| lane_totals[k].saturating_sub(self.lane_steps[k]));
        self.lane_steps = lane_totals;
        if std::env::var_os("EVOLUTION_PROFILE_BREED").is_some() {
            let [plan, emit, write] = crate::storage::take_breed_nanos();
            eprintln!(
                "Breeding: generation {generation}, plan {:.3} s, emit {:.3} s, write {:.3} s",
                plan as f64 * 1e-9,
                emit as f64 * 1e-9,
                write as f64 * 1e-9
            );
        }
        let share = |stops: u64| stops as f64 / rungs.creatures.max(1) as f64;
        let _ = writeln!(
            self.file,
            "{generation},{:.6},{:.6},{:.6},{:.3},{:.0},{:.3},{:.3},{:.3},{:.3},{:.4},{},{},{:.4},{:.4},{:.4},{},{},{},{},{},{:.3},{:.1},{},{:.4},{:.4},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}",
            self.seconds[0],
            self.seconds[1],
            self.seconds[2],
            population as f64 / seconds,
            delta[0],
            delta[1],
            delta[2],
            delta[3],
            nodes[0],
            nodes[1],
            ring.block,
            ring.blocks,
            meter.chain_p95_of(generation),
            meter.boundary_of(generation),
            self.starved_block,
            lanes[0],
            lanes[1],
            lanes[2],
            lanes[3],
            self.discarded,
            kernel_wait_delta,
            rungs.steps_per_creature(),
            rungs.audit_rows,
            share(rungs.stops[0]),
            share(rungs.stops[1]),
            share(rungs.stops[2]),
            rungs.misses_per_10k(0),
            rungs.misses_per_10k(1),
            rungs.extra_misses_per_10k(0),
            rungs.extra_misses_per_10k(1),
            rungs.top1_kept,
            rungs.top10_kept,
            rungs.top1_screen,
            rungs.top10_screen,
            rungs.armed.iter().filter(|&&a| a).count(),
            rungs.bands_off.iter().map(|&b| u32::from(b)).sum::<u32>(),
        );
        let _ = self.file.flush();
        self.reset();
    }
}
/// The host's time per ring block (`ring::Step::chain`) over the last three
/// generations, which sizes the ring of the next new game.
#[derive(Default)]
pub(super) struct RingMeter {
    /// (generation, seconds) of blocks that did not end a generation.
    chains: std::collections::VecDeque<(u32, f64)>,
    /// (generation, seconds) of the blocks that ended one.
    boundaries: std::collections::VecDeque<(u32, f64)>,
}
impl RingMeter {
    const GENERATIONS: u32 = 3;
    pub(super) fn clear(&mut self) {
        self.chains.clear();
        self.boundaries.clear();
    }
    /// A block of `generation` took `seconds`; `boundary` when it ended it.
    pub(super) fn add(&mut self, generation: u32, seconds: f64, boundary: bool) {
        let list = if boundary {
            &mut self.boundaries
        } else {
            &mut self.chains
        };
        list.push_back((generation, seconds));
        let oldest = generation.saturating_sub(Self::GENERATIONS - 1);
        for list in [&mut self.chains, &mut self.boundaries] {
            while list.front().is_some_and(|&(g, _)| g < oldest) {
                list.pop_front();
            }
        }
    }
    fn p95(values: impl Iterator<Item = f64>) -> Option<f64> {
        let mut v: Vec<f64> = values.collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        Some(v[((v.len() - 1) as f64 * 0.95).round() as usize])
    }
    /// For the stage log: the p95 time of this generation's blocks that did
    /// not end it, 0 if none.
    fn chain_p95_of(&self, generation: u32) -> f64 {
        Self::p95(
            self.chains
                .iter()
                .filter(|c| c.0 == generation)
                .map(|c| c.1),
        )
        .unwrap_or(0.0)
    }
    /// For the stage log: the time of the block that ended this generation,
    /// 0 if none.
    fn boundary_of(&self, generation: u32) -> f64 {
        self.boundaries
            .iter()
            .filter(|c| c.0 == generation)
            .map(|c| c.1)
            .fold(0.0, f64::max)
    }
    /// What the next ring is sized from: the scheduler's rate, and the
    /// measured host times when this session has run a generation.
    pub(super) fn times(
        &self,
        sched: Option<&crate::scheduler::Scheduler>,
    ) -> crate::storage::RingTimes {
        let prior = crate::storage::RingTimes::default();
        let rate = sched.map_or(prior.rate, |s| {
            s.devices.iter().map(|d| d.rate).sum::<f64>()
        });
        if self.boundaries.is_empty() {
            return crate::storage::RingTimes { rate, ..prior };
        }
        crate::storage::RingTimes {
            rate,
            chain: Self::p95(self.chains.iter().map(|c| c.1)).unwrap_or(prior.chain),
            boundary: self.boundaries.iter().map(|c| c.1).fold(0.0, f64::max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_log_writes_one_csv_row_per_generation() {
        let path =
            std::env::temp_dir().join(format!("evolution-stage-log-{}.csv", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut log = StageLog {
            file: std::fs::File::create(&path).unwrap(),
            started: Instant::now(),
            seconds: [1.0, 2.0, 3.0],
            totals: [0.0; 4],
            kernel_wait: 0.0,
            lane_steps: [0; 4],
            idle_at_block: 0.0,
            starved_block: 0.0,
            discarded: 0,
        };
        log.write_row(
            5,
            1000,
            None,
            [4.0, 0.0],
            Default::default(),
            &RingMeter::default(),
            &Default::default(),
        );
        log.add(0, 4.0);
        log.write_row(
            6,
            1000,
            None,
            [4.0, 0.0],
            Default::default(),
            &RingMeter::default(),
            &Default::default(),
        );
        drop(log);
        let text = std::fs::read_to_string(&path).unwrap();
        let rows: Vec<&str> = text.lines().collect();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].starts_with("5,1.000000,2.000000,3.000000,"));
        assert!(rows[1].starts_with("6,4.000000,0.000000,0.000000,"));
        let _ = std::fs::remove_file(&path);
    }
}
