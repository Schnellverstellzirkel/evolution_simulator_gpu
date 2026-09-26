//! Spreads a generation's evaluations across the primary GPU and CPU SIMD cores.
//! Additional GPUs require an explicit `EVOLUTION_DEVICES` selection.
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
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

/// What a queued unit evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trial {
    /// Each creature once, at the standard physics.
    Standard,
    /// Perturbed copies of contenders, at the fine physics.
    Check,
}

struct QueuedUnit {
    ticket: u64,
    indices: Vec<usize>,
    trial: Trial,
    population: Arc<Population>,
    config: Config,
}

pub struct Device {
    pub engine: Box<dyn Engine>,
    /// Exact submitted input remains available until its result is accepted.
    queued: VecDeque<QueuedUnit>,
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
        self.queued.iter().map(|unit| unit.indices.len()).sum()
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
    /// Trials per creature (`EVOLUTION_ROBUST_TRIALS`, 1 or 2). With 2, a
    /// creature that could enter the archive also runs a slightly perturbed
    /// copy at four times the physics resolution, and its fitness is the
    /// lower of the two. Gaits that only work at the coarse standard physics,
    /// or only from one exact pose, lose that way.
    robust_trials: usize,
    /// Contenders waiting for their check trial, and when the oldest arrived.
    checks: Vec<usize>,
    checks_since: Option<Instant>,
    /// Standard-trial results of contenders whose check is pending.
    held: HashMap<usize, EvaluationMetrics>,
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Returns explicitly requested secondary GPU names. The safe default is to
/// use only the primary GPU; `primary` and `off` both keep secondary GPUs off.
fn secondary_device_names(selection: Option<&str>) -> Vec<&str> {
    let Some(selection) = selection else {
        return Vec::new();
    };
    if matches!(
        selection.trim().to_ascii_lowercase().as_str(),
        "primary" | "off"
    ) {
        return Vec::new();
    }
    selection
        .split(',')
        .map(str::trim)
        .filter(|name| {
            !name.is_empty() && !matches!(name.to_ascii_lowercase().as_str(), "primary" | "off")
        })
        .collect()
}

impl Scheduler {
    /// Opens the named primary GPU, the other GPUs listed in `EVOLUTION_DEVICES`
    /// (off by default; `primary` or `off` for none), and a CPU engine with
    /// `EVOLUTION_CPU_THREADS` threads (default six). Evaluation and general
    /// workers share half the logical CPUs, at most eight, with at least one
    /// general worker. Zero or a one-worker budget disables the CPU engine.
    pub fn new(primary: &str) -> Result<Self> {
        let step_range = env_or("EVOLUTION_GPU_CHUNK", crate::gpu::DEFAULT_STEP_RANGE);
        let mut devices = vec![Device::new(
            Box::new(engine::gpu_engine(primary, 64, step_range)?),
            180_000.0,
            8192,
            env_or("EVOLUTION_UNIT_SECONDS", 1.0),
        )];
        let extra = std::env::var("EVOLUTION_DEVICES").ok();
        for name in secondary_device_names(extra.as_deref()) {
            if primary.to_lowercase().contains(&name.to_lowercase()) {
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
        let threads = engine::cpu_threads();
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
            robust_trials: env_or("EVOLUTION_ROBUST_TRIALS", 2usize).clamp(1, 2),
            checks: Vec::new(),
            checks_since: None,
            held: HashMap::new(),
        })
    }

    /// A scheduler with only the CPU engine, for machines without a GPU and
    /// for tests.
    pub fn cpu_only(threads: usize) -> Result<Self> {
        Ok(Self {
            devices: vec![Device::new(
                Box::new(engine::cpu_engine(threads.max(1))?),
                30_000.0,
                64,
                1.0,
            )],
            round: None,
            packing_seconds: 0.0,
            robust_trials: 2,
            checks: Vec::new(),
            checks_since: None,
            held: HashMap::new(),
        })
    }

    pub fn names(&self) -> String {
        self.devices
            .iter()
            .map(|d| d.engine.name())
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// Queued units, plus one while contenders still wait for their checks.
    pub fn in_flight(&self) -> usize {
        self.devices.iter().map(|d| d.queued.len()).sum::<usize>()
            + usize::from(!self.held.is_empty())
    }

    /// Queues check units for waiting contenders. Checks go first, but wait
    /// to fill a reasonable unit while standard work remains.
    pub fn pump_checks(&mut self, pop: &Population, cfg: &Config) -> Result<()> {
        if self.checks.is_empty() {
            return Ok(());
        }
        let waited = self
            .checks_since
            .is_some_and(|since| since.elapsed() > Duration::from_millis(500));
        let fine = Config {
            fidelity: Some(crate::physics::Fidelity::fine()),
            ..cfg.clone()
        };
        for device in &mut self.devices {
            while device.engine.free_slots() > 0 && !self.checks.is_empty() {
                if self.checks.len() < device.min_unit / 2 && self.round.is_some() && !waited {
                    break;
                }
                // A check costs several standard trials; keep units about as long.
                let size =
                    ((device.rate * device.unit_seconds / 6.0) as usize).max(device.min_unit / 2);
                let capacity = device.engine.max_nodes();
                let mut indices = Vec::with_capacity(size.min(self.checks.len()));
                let mut rest = Vec::new();
                for i in self.checks.drain(..) {
                    if indices.len() < size && pop.genomes[i].node_count <= capacity {
                        indices.push(i);
                    } else {
                        rest.push(i);
                    }
                }
                self.checks = rest;
                if indices.is_empty() {
                    break;
                }
                let started = Instant::now();
                let mut unit = Population::default();
                for &i in &indices {
                    let mut creature = pop.creature(i);
                    perturb(&mut creature);
                    unit.push(creature);
                }
                let population = Arc::new(unit);
                let ticket = device
                    .engine
                    .submit_shared(Arc::clone(&population), &fine)?;
                self.packing_seconds += started.elapsed().as_secs_f64();
                device.queued.push_back(QueuedUnit {
                    ticket,
                    indices,
                    trial: Trial::Check,
                    population,
                    config: fine.clone(),
                });
            }
        }
        if self.checks.is_empty() {
            self.checks_since = None;
        }
        Ok(())
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
        self.pump_checks(pop, cfg)?;
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
                let population = Arc::new(pop.subset(&indices));
                let ticket = device.engine.submit_shared(Arc::clone(&population), cfg)?;
                self.packing_seconds += started.elapsed().as_secs_f64();
                device.queued.push_back(QueuedUnit {
                    ticket,
                    indices,
                    trial: Trial::Standard,
                    population,
                    config: cfg.clone(),
                });
            }
        }
        if round.cursor == round.order.len() && round.oversize.is_empty() {
            self.round = None;
        }
        Ok(())
    }

    /// Returns creatures whose evaluation is final as (population indices,
    /// metrics), waiting up to `timeout` when nothing is ready. `contender`
    /// says whether a standard-trial result could enter the archive; those
    /// creatures are held for a check trial and returned once it finishes.
    pub fn collect(
        &mut self,
        _pop: &Population,
        _cfg: &Config,
        timeout: Duration,
        mut contender: impl FnMut(usize, &EvaluationMetrics) -> bool,
    ) -> Result<Vec<(Vec<usize>, Vec<EvaluationMetrics>)>> {
        let mut out = Vec::new();
        let deadline = Instant::now() + timeout;
        loop {
            for device in &mut self.devices {
                while let Some(done) = device.engine.poll()? {
                    let queued = device
                        .queued
                        .front()
                        .context("Unexpected evaluation result")?;
                    anyhow::ensure!(
                        queued.ticket == done.ticket,
                        "Evaluation results out of order"
                    );
                    anyhow::ensure!(
                        done.results.len() == queued.indices.len(),
                        "Evaluation result count mismatch: expected {}, received {}",
                        queued.indices.len(),
                        done.results.len()
                    );
                    let QueuedUnit {
                        indices,
                        trial,
                        population,
                        config,
                        ..
                    } = device.queued.pop_front().expect("validated queued unit");
                    device.busy_seconds += done.busy_seconds;
                    let mut finals = Vec::with_capacity(indices.len());
                    let mut metrics = Vec::with_capacity(indices.len());
                    match trial {
                        Trial::Standard => {
                            device.creatures += indices.len() as u64;
                            if done.busy_seconds > 0.0 && indices.len() >= device.min_unit / 2 {
                                let rate = indices.len() as f64 / done.busy_seconds;
                                device.rate = 0.7 * device.rate + 0.3 * rate;
                            }
                            for (k, &i) in indices.iter().enumerate() {
                                let metric = to_metrics(&population, k, &done.results[k], &config);
                                if self.robust_trials > 1 && contender(i, &metric) {
                                    self.held.insert(i, metric);
                                    self.checks.push(i);
                                    self.checks_since.get_or_insert_with(Instant::now);
                                } else {
                                    finals.push(i);
                                    metrics.push(metric);
                                }
                            }
                        }
                        Trial::Check => {
                            for (k, &i) in indices.iter().enumerate() {
                                // A creature re-queued meanwhile may have been settled already.
                                if let Some(mut metric) = self.held.remove(&i) {
                                    // Reliable motion only: keep the worse of both trials.
                                    metric.fitness = metric.fitness.min(done.results[k].fitness);
                                    finals.push(i);
                                    metrics.push(metric);
                                }
                            }
                        }
                    }
                    if !finals.is_empty() {
                        out.push((finals, metrics));
                    }
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
        // Without an archive to compare against, every creature is checked.
        self.evaluate_checked(pop, indices, cfg, true)
    }
    /// One trial per creature at `cfg`'s fidelity, without the contender check.
    pub fn evaluate_single(
        &mut self,
        pop: &Population,
        indices: &[usize],
        cfg: &Config,
    ) -> Result<Vec<EvaluationMetrics>> {
        self.evaluate_checked(pop, indices, cfg, false)
    }
    fn evaluate_checked(
        &mut self,
        pop: &Population,
        indices: &[usize],
        cfg: &Config,
        check: bool,
    ) -> Result<Vec<EvaluationMetrics>> {
        // Finish anything left from an interrupted round first.
        self.stop();
        while self.in_flight() > 0 {
            self.pump_checks(pop, cfg)?;
            self.collect(pop, cfg, Duration::from_secs(1), |_, _| false)?;
        }
        let mut position = std::collections::HashMap::with_capacity(indices.len());
        for (slot, &i) in indices.iter().enumerate() {
            position.insert(i, slot);
        }
        let mut out = vec![EvaluationMetrics::default(); indices.len()];
        let mut remaining = indices.len();
        self.begin(pop, indices.iter().copied());
        while remaining > 0 {
            self.pump(pop, cfg, &[])?;
            for (unit, metrics) in
                self.collect(pop, cfg, Duration::from_millis(50), |_, _| check)?
            {
                for (i, metric) in unit.into_iter().zip(metrics) {
                    out[position[&i]] = metric;
                    remaining -= 1;
                }
            }
        }
        Ok(out)
    }
}

/// Small deterministic change to a creature's starting pose and grip, for the
/// robustness trial.
fn perturb(creature: &mut crate::evolution::Creature) {
    let mut rng = crate::evolution::Rng::new(creature.id ^ 0x5eed_7a11, 0, 0);
    for node in &mut creature.nodes {
        node.x += rng.range(-0.02, 0.02);
        node.y += rng.range(0.0, 0.02);
        node.friction = (node.friction * rng.range(0.9, 1.1)).clamp(0.0, 1.0);
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
            mean_height: (r.height_sum / cfg.steps().max(1) as f32).max(0.0),
            feet: r.feet() as f32,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Finished;
    use std::sync::{Arc, Mutex};

    struct Submission {
        ticket: u64,
        population: Arc<Population>,
        config: Config,
    }

    #[derive(Default)]
    struct FakeState {
        submissions: Vec<Submission>,
        results: VecDeque<Finished>,
        pending: bool,
    }

    struct FakeEngine(Arc<Mutex<FakeState>>);

    impl Engine for FakeEngine {
        fn name(&self) -> String {
            "fake evaluator".into()
        }

        fn max_nodes(&self) -> usize {
            64
        }

        fn free_slots(&self) -> usize {
            usize::from(!self.0.lock().unwrap().pending)
        }

        fn submit_shared(&mut self, population: Arc<Population>, config: &Config) -> Result<u64> {
            let mut state = self.0.lock().unwrap();
            let ticket = state.submissions.len() as u64 + 1;
            state.submissions.push(Submission {
                ticket,
                population,
                config: config.clone(),
            });
            state.pending = true;
            Ok(ticket)
        }

        fn poll(&mut self) -> Result<Option<Finished>> {
            let mut state = self.0.lock().unwrap();
            let done = state.results.pop_front();
            if done.is_some() {
                state.pending = false;
            }
            Ok(done)
        }

        fn wait(&mut self, _timeout: Duration) {}
    }

    fn fake_scheduler() -> (Scheduler, Arc<Mutex<FakeState>>) {
        let state = Arc::new(Mutex::new(FakeState::default()));
        let scheduler = Scheduler {
            devices: vec![Device::new(
                Box::new(FakeEngine(Arc::clone(&state))),
                100.0,
                2,
                1.0,
            )],
            round: None,
            packing_seconds: 0.0,
            robust_trials: 1,
            checks: Vec::new(),
            checks_since: None,
            held: HashMap::new(),
        };
        (scheduler, state)
    }

    fn submission_config() -> Config {
        Config {
            population: 4,
            duration: 1.0,
            random_seed: false,
            ..Config::default()
        }
    }

    #[test]
    fn retained_standard_results_use_the_submitted_population_and_config() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, state) = fake_scheduler();
        scheduler.begin(&pop, [3, 1]);
        scheduler.pump(&pop, &cfg, &[]).unwrap();
        let (ticket, indices, results) = {
            let state = state.lock().unwrap();
            let submission = &state.submissions[0];
            assert_eq!(submission.config, cfg);
            let indices: Vec<_> = submission
                .population
                .genomes
                .iter()
                .map(|genome| {
                    pop.genomes
                        .iter()
                        .position(|original| original.id == genome.id)
                        .unwrap()
                })
                .collect();
            let results = submission
                .population
                .genomes
                .iter()
                .map(|genome| GpuResult {
                    fitness: genome.id as f32,
                    ground_contact: cfg.steps() as f32 * genome.node_count as f32 * 0.25,
                    height_sum: cfg.steps() as f32 * 1.5,
                    ..GpuResult::default()
                })
                .collect();
            (submission.ticket, indices, results)
        };
        state.lock().unwrap().results.push_back(Finished {
            ticket,
            results,
            busy_seconds: 0.5,
        });
        // The caller has moved to a new population and different trial settings
        // while the original unit was in flight. Its results still use its own
        // node counts, duration and fidelity for normalization.
        let changed = Config {
            duration: 2.0,
            fidelity: Some(crate::physics::Fidelity::fine()),
            ..cfg
        };
        let output = scheduler
            .collect(&Population::default(), &changed, Duration::ZERO, |_, _| {
                false
            })
            .unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].0, indices);
        for (&index, metric) in indices.iter().zip(&output[0].1) {
            assert_eq!(metric.fitness, pop.genomes[index].id as f32);
            assert_eq!(metric.behavior.ground_contact, 0.25);
            assert_eq!(metric.behavior.mean_height, 1.5);
        }
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn retained_fine_checks_keep_the_exact_perturbed_submission() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, state) = fake_scheduler();
        let indices = vec![2, 0];
        scheduler.checks = indices.clone();
        for &index in &indices {
            scheduler.held.insert(
                index,
                EvaluationMetrics {
                    fitness: 10.0,
                    ..EvaluationMetrics::default()
                },
            );
        }
        scheduler.pump_checks(&pop, &cfg).unwrap();
        let ticket = {
            let state = state.lock().unwrap();
            let submission = &state.submissions[0];
            assert_eq!(
                submission.config.fidelity,
                Some(crate::physics::Fidelity::fine())
            );
            assert!(
                Arc::strong_count(&submission.population) >= 2,
                "scheduler must retain the same shared population as the engine"
            );
            for (slot, &index) in indices.iter().enumerate() {
                let mut expected = pop.creature(index);
                perturb(&mut expected);
                let actual = submission.population.creature(slot);
                assert_eq!(actual.id, expected.id);
                assert_eq!(actual.nodes, expected.nodes);
                assert_eq!(actual.bones, expected.bones);
                assert_eq!(actual.muscles, expected.muscles);
            }
            submission.ticket
        };
        state.lock().unwrap().results.push_back(Finished {
            ticket,
            results: vec![
                GpuResult {
                    fitness: 4.0,
                    ..GpuResult::default()
                };
                indices.len()
            ],
            busy_seconds: 0.5,
        });
        let output = scheduler
            .collect(
                &Population::default(),
                &Config::default(),
                Duration::ZERO,
                |_, _| false,
            )
            .unwrap();
        assert_eq!(output[0].0, indices);
        assert!(output[0].1.iter().all(|metric| metric.fitness == 4.0));
        assert!(scheduler.held.is_empty());
        assert_eq!(scheduler.in_flight(), 0);
        assert_eq!(
            Arc::strong_count(&state.lock().unwrap().submissions[0].population),
            1
        );
    }

    #[test]
    fn retained_units_reject_malformed_results_without_consuming_work() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        for trial in [Trial::Standard, Trial::Check] {
            for (wrong_ticket, result_count) in [(true, 2), (false, 1), (false, 3)] {
                let (mut scheduler, state) = fake_scheduler();
                if trial == Trial::Check {
                    scheduler.checks = vec![0, 1];
                    for index in 0..2 {
                        scheduler.held.insert(
                            index,
                            EvaluationMetrics {
                                fitness: 10.0,
                                ..EvaluationMetrics::default()
                            },
                        );
                    }
                    scheduler.pump_checks(&pop, &cfg).unwrap();
                } else {
                    scheduler.begin(&pop, 0..2);
                    scheduler.pump(&pop, &cfg, &[]).unwrap();
                }
                let ticket = state.lock().unwrap().submissions[0].ticket;
                let before = scheduler.in_flight();
                let rate = scheduler.devices[0].rate;
                state.lock().unwrap().results.push_back(Finished {
                    ticket: ticket + u64::from(wrong_ticket),
                    results: vec![GpuResult::default(); result_count],
                    busy_seconds: 0.5,
                });
                assert!(
                    scheduler
                        .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
                        .is_err()
                );
                assert_eq!(
                    scheduler.in_flight(),
                    before,
                    "{trial:?} pending work was consumed"
                );
                assert_eq!(scheduler.devices[0].queued.len(), 1);
                assert_eq!(scheduler.devices[0].creatures, 0);
                assert_eq!(scheduler.devices[0].busy_seconds, 0.0);
                assert_eq!(scheduler.devices[0].rate, rate);
                state.lock().unwrap().results.push_back(Finished {
                    ticket,
                    results: vec![GpuResult::default(); 2],
                    busy_seconds: 0.5,
                });
                let output = scheduler
                    .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
                    .unwrap();
                assert_eq!(
                    output
                        .iter()
                        .map(|(indices, _)| indices.len())
                        .sum::<usize>(),
                    2
                );
                assert_eq!(scheduler.in_flight(), 0);
            }
        }
    }

    #[test]
    fn secondary_devices_are_opt_in_and_selection_sentinels_disable_them() {
        assert_eq!(secondary_device_names(None), Vec::<&str>::new());
        assert_eq!(secondary_device_names(Some("primary")), Vec::<&str>::new());
        assert_eq!(secondary_device_names(Some("off")), Vec::<&str>::new());
        assert_eq!(secondary_device_names(Some(" OFF ")), Vec::<&str>::new());
    }

    #[test]
    fn secondary_device_names_preserve_explicit_comma_separated_selection() {
        assert_eq!(
            secondary_device_names(Some("radeon, RTX 4060")),
            vec!["radeon", "RTX 4060"]
        );
        assert_eq!(
            secondary_device_names(Some("primary, radeon, off")),
            vec!["radeon"]
        );
        assert_eq!(secondary_device_names(Some(" , ")), Vec::<&str>::new());
    }

    #[test]
    fn contenders_get_the_worse_of_a_fine_perturbed_check() {
        let cfg = Config {
            population: 48,
            duration: 3.0,
            random_seed: false,
            ..Config::default()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let mut sched = Scheduler::cpu_only(2).unwrap();
        sched.begin(&pop, 0..cfg.population);
        let mut got = vec![None; cfg.population];
        let deadline = Instant::now() + Duration::from_secs(120);
        while got.iter().any(Option::is_none) {
            assert!(Instant::now() < deadline, "evaluation stalled");
            sched.pump(&pop, &cfg, &[]).unwrap();
            for (indices, metrics) in sched
                .collect(&pop, &cfg, Duration::from_millis(20), |i, _| i % 2 == 0)
                .unwrap()
            {
                for (i, m) in indices.into_iter().zip(metrics) {
                    assert!(got[i].is_none(), "creature {i} returned twice");
                    got[i] = Some(m.fitness);
                }
            }
        }
        assert_eq!(sched.in_flight(), 0);
        let standard = crate::cpu_engine::evaluate(&pop, &cfg);
        let mut perturbed = Population::default();
        for i in 0..cfg.population {
            let mut c = pop.creature(i);
            perturb(&mut c);
            perturbed.push(c);
        }
        let fine = Config {
            fidelity: Some(crate::physics::Fidelity::fine()),
            ..cfg.clone()
        };
        let check = crate::cpu_engine::evaluate(&perturbed, &fine);
        for i in 0..cfg.population {
            let expected = if i % 2 == 0 {
                standard[i].fitness.min(check[i].fitness)
            } else {
                standard[i].fitness
            };
            let actual = got[i].unwrap();
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "creature {i}: {actual} vs {expected}"
            );
        }
        // The check really ran at the fine physics: some differ from a
        // perturbed trial at the standard physics.
        let coarse = crate::cpu_engine::evaluate(&perturbed, &cfg);
        assert!((0..cfg.population).any(|i| (coarse[i].fitness - check[i].fitness).abs() > 1e-3));
    }
}
