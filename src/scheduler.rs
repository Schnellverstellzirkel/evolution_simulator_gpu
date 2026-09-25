//! Spreads a generation's evaluations across every evaluation engine: the
//! discrete GPU, the integrated GPU, and CPU SIMD cores.
//!
//! Work is handed out in body-size order so GPU units fill few, large buckets.
//! Each engine keeps up to two units queued; unit sizes follow each engine's
//! measured rate so that all engines finish a generation together. Bodies
//! larger than an engine supports go to one that supports them. Results differ
//! slightly by engine (floating-point rounding), so `EVOLUTION_DEVICES=primary`
//! and `EVOLUTION_CPU_THREADS=0` give a single-device, bit-reproducible setup.
use crate::{
    config::Config,
    creature_kernel::{self, GpuResult},
    engine::{self, Engine},
    evolution::Population,
    qd::{EvaluationMetrics, TrialMetrics},
};
use anyhow::{Context, Result};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub struct Device {
    pub engine: Box<dyn Engine>,
    /// Queued units: ticket and population indices.
    queued: VecDeque<(u64, Vec<usize>)>,
    /// Measured creatures per second of device time (exponential average).
    pub rate: f64,
    pub creatures: u64,
    pub busy_seconds: f64,
    /// Smallest unit worth submitting to this engine.
    min_unit: usize,
    /// Seconds of work per unit; short on a GPU that also drives the display.
    unit_seconds: f64,
}

impl Device {
    fn new(engine: Box<dyn Engine>, rate: f64, min_unit: usize, unit_seconds: f64) -> Self {
        Self {
            engine,
            queued: VecDeque::new(),
            rate,
            creatures: 0,
            busy_seconds: 0.0,
            min_unit,
            unit_seconds,
        }
    }
    fn queued_creatures(&self) -> usize {
        self.queued.iter().map(|(_, unit)| unit.len()).sum()
    }
}

struct Round {
    order: Vec<usize>,
    cursor: usize,
    /// Creatures skipped by an engine that cannot hold them.
    oversize: Vec<usize>,
}

pub struct Scheduler {
    pub devices: Vec<Device>,
    round: Option<Round>,
    pub packing_seconds: f64,
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Scheduler {
    /// Opens the named primary GPU, the other GPUs listed in `EVOLUTION_DEVICES`
    /// (default `radeon`; `primary` for none), and a CPU engine with
    /// `EVOLUTION_CPU_THREADS` threads (default: all but four logical CPUs,
    /// which stay free for the display, the UI, and breeding).
    pub fn new(primary: &str) -> Result<Self> {
        let step_range = env_or("EVOLUTION_GPU_CHUNK", crate::gpu::DEFAULT_STEP_RANGE);
        let mut devices = vec![Device::new(
            Box::new(engine::gpu_engine(primary, 64, step_range)?),
            180_000.0,
            8192,
            env_or("EVOLUTION_UNIT_SECONDS", 1.0),
        )];
        let extra = std::env::var("EVOLUTION_DEVICES").unwrap_or_else(|_| "radeon".into());
        for name in extra.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            if name == "primary" || primary.to_lowercase().contains(&name.to_lowercase()) {
                continue;
            }
            // RADV compile time explodes for the largest bodies; those stay on the
            // primary GPU. Short dispatches let the desktop interleave its frames.
            match engine::gpu_engine(name, 16, env_or("EVOLUTION_SECONDARY_CHUNK", 16)) {
                Ok(engine) => devices.push(Device::new(
                    Box::new(engine),
                    40_000.0,
                    2048,
                    env_or("EVOLUTION_SECONDARY_UNIT_SECONDS", 0.05),
                )),
                Err(err) => eprintln!("Evaluation device {name:?} unavailable: {err:#}"),
            }
        }
        let logical = std::thread::available_parallelism().map_or(4, usize::from);
        let threads = env_or("EVOLUTION_CPU_THREADS", logical.saturating_sub(4));
        if threads > 0 {
            devices.push(Device::new(
                Box::new(engine::cpu_engine(threads)?),
                30_000.0,
                1024,
                env_or("EVOLUTION_UNIT_SECONDS", 1.0),
            ));
        }
        Ok(Self {
            devices,
            round: None,
            packing_seconds: 0.0,
        })
    }

    pub fn names(&self) -> String {
        self.devices
            .iter()
            .map(|d| d.engine.name())
            .collect::<Vec<_>>()
            .join(" + ")
    }

    pub fn in_flight(&self) -> usize {
        self.devices.iter().map(|d| d.queued.len()).sum()
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.devices
            .iter()
            .map(|d| d.engine.allocated_bytes())
            .sum()
    }

    /// Starts handing out `indices` unless a round is active. Creatures are
    /// ordered by kernel capacity so each GPU unit fills few, large buckets.
    pub fn begin(&mut self, pop: &Population, indices: impl IntoIterator<Item = usize>) {
        if self.round.is_some() {
            return;
        }
        let mut groups: [Vec<usize>; creature_kernel::CAPACITIES.len()] = Default::default();
        for i in indices {
            groups[creature_kernel::capacity_index(pop.genomes[i].node_count)].push(i);
        }
        let order = groups.concat();
        if !order.is_empty() {
            self.round = Some(Round {
                order,
                cursor: 0,
                oversize: Vec::new(),
            });
        }
    }

    /// Appends `indices` to the active round (or starts one). Used while the
    /// next generation is still being bred.
    pub fn extend(&mut self, pop: &Population, indices: impl IntoIterator<Item = usize>) {
        let mut groups: [Vec<usize>; creature_kernel::CAPACITIES.len()] = Default::default();
        for i in indices {
            groups[creature_kernel::capacity_index(pop.genomes[i].node_count)].push(i);
        }
        let order = groups.concat();
        match self.round.as_mut() {
            Some(round) => round.order.extend(order),
            None if !order.is_empty() => {
                self.round = Some(Round {
                    order,
                    cursor: 0,
                    oversize: Vec::new(),
                })
            }
            None => {}
        }
    }

    /// Stops handing out new work; queued work still completes.
    pub fn stop(&mut self) {
        self.round = None;
    }

    /// Queues work on every engine with a free slot. Creatures marked in
    /// `done` are skipped.
    pub fn pump(&mut self, pop: &Population, cfg: &Config, done: &[bool]) -> Result<()> {
        let Some(round) = self.round.as_mut() else {
            return Ok(());
        };
        let total_rate: f64 = self.devices.iter().map(|d| d.rate).sum();
        let queued: usize = self.devices.iter().map(Device::queued_creatures).sum();
        for device in &mut self.devices {
            while device.engine.free_slots() > 0 {
                let remaining = round.order.len() - round.cursor + round.oversize.len();
                if remaining == 0 {
                    break;
                }
                // Seconds until every engine has drained the work that is left,
                // if each works at its measured rate.
                let horizon = (remaining + queued) as f64 / total_rate;
                let fair = (device.rate * horizon) as usize;
                let want = fair.saturating_sub(device.queued_creatures());
                // Near the end, a slow engine only takes work it can finish in time.
                if want < device.min_unit / 2 && device.queued_creatures() > 0 {
                    break;
                }
                let size = ((device.rate * device.unit_seconds) as usize)
                    .min(want)
                    .max(device.min_unit)
                    .min(remaining);
                let capacity = device.engine.max_nodes();
                let mut indices = Vec::with_capacity(size);
                if capacity >= 64 {
                    let take = round.oversize.len().min(size);
                    indices.extend(round.oversize.drain(..take));
                }
                while indices.len() < size && round.cursor < round.order.len() {
                    let i = round.order[round.cursor];
                    round.cursor += 1;
                    if done.get(i).copied().unwrap_or(false) {
                        continue;
                    }
                    if pop.genomes[i].node_count > capacity {
                        round.oversize.push(i);
                    } else {
                        indices.push(i);
                    }
                }
                if indices.is_empty() {
                    break;
                }
                let started = Instant::now();
                let ticket = device.engine.submit(pop.subset(&indices), cfg)?;
                self.packing_seconds += started.elapsed().as_secs_f64();
                device.queued.push_back((ticket, indices));
            }
        }
        if round.cursor == round.order.len() && round.oversize.is_empty() {
            self.round = None;
        }
        Ok(())
    }

    /// Returns every finished unit as (population indices, metrics), waiting up
    /// to `timeout` when nothing is ready.
    pub fn collect(
        &mut self,
        pop: &Population,
        cfg: &Config,
        timeout: Duration,
    ) -> Result<Vec<(Vec<usize>, Vec<EvaluationMetrics>)>> {
        let mut out = Vec::new();
        let deadline = Instant::now() + timeout;
        loop {
            for device in &mut self.devices {
                while let Some(done) = device.engine.poll()? {
                    let (ticket, indices) = device
                        .queued
                        .pop_front()
                        .context("Unexpected evaluation result")?;
                    anyhow::ensure!(ticket == done.ticket, "Evaluation results out of order");
                    device.busy_seconds += done.busy_seconds;
                    device.creatures += indices.len() as u64;
                    if done.busy_seconds > 0.0 && indices.len() >= device.min_unit / 2 {
                        let rate = indices.len() as f64 / done.busy_seconds;
                        device.rate = 0.7 * device.rate + 0.3 * rate;
                    }
                    let metrics = indices
                        .iter()
                        .zip(&done.results)
                        .map(|(&i, r)| to_metrics(pop, i, r, cfg))
                        .collect();
                    out.push((indices, metrics));
                }
            }
            if !out.is_empty() || self.in_flight() == 0 || Instant::now() >= deadline {
                return Ok(out);
            }
            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(2));
            if let Some(device) = self.devices.iter_mut().find(|d| !d.queued.is_empty()) {
                device.engine.wait(wait);
            }
        }
    }

    /// Evaluates `indices` on every engine and returns metrics in the same order.
    pub fn evaluate(
        &mut self,
        pop: &Population,
        indices: &[usize],
        cfg: &Config,
    ) -> Result<Vec<EvaluationMetrics>> {
        // Finish anything left from an interrupted round first.
        while self.in_flight() > 0 {
            self.collect(pop, cfg, Duration::from_secs(1))?;
        }
        self.stop();
        let mut position = std::collections::HashMap::with_capacity(indices.len());
        for (slot, &i) in indices.iter().enumerate() {
            position.insert(i, slot);
        }
        let mut out = vec![EvaluationMetrics::default(); indices.len()];
        let mut remaining = indices.len();
        self.begin(pop, indices.iter().copied());
        while remaining > 0 {
            self.pump(pop, cfg, &[])?;
            for (unit, metrics) in self.collect(pop, cfg, Duration::from_millis(50))? {
                for (i, metric) in unit.into_iter().zip(metrics) {
                    out[position[&i]] = metric;
                    remaining -= 1;
                }
            }
        }
        Ok(out)
    }
}

/// Converts a raw kernel result to the archive's normalized metrics.
pub fn to_metrics(
    pop: &Population,
    index: usize,
    r: &GpuResult,
    cfg: &Config,
) -> EvaluationMetrics {
    let contact_denominator = (cfg.steps().max(1) * pop.genomes[index].node_count as u32) as f32;
    EvaluationMetrics {
        fitness: r.fitness,
        behavior: TrialMetrics {
            ground_contact: (r.ground_contact / contact_denominator).clamp(0.0, 1.0),
            vertical_oscillation: if r.vertical_oscillation.is_finite() {
                r.vertical_oscillation.max(0.0)
            } else {
                0.0
            },
            gait_frequency: if r.gait_frequency.is_finite() {
                r.gait_frequency.max(0.0)
            } else {
                0.0
            },
        },
    }
}
