//! Routes evaluation work to the selected GPU devices. Additional GPUs require
//! an explicit `EVOLUTION_DEVICES` selection. The GPU is the only engine:
//! there is no CPU evaluation.
//!
//! Callers queue work: a population, the creatures of it to evaluate, and the
//! trial settings. Each piece of work goes to an engine as one unit. A piece
//! that covers its whole population is handed over without a copy.
//! Confirmation work goes before standard work. Results come back tagged with
//! the caller's tag, in whatever order the engines finish. The GPU's score
//! is final.
use crate::{
    config::Config,
    creature_kernel::GpuResult,
    engine::{self, Engine},
    evolution::Population,
    qd::{EvaluationMetrics, TrialMetrics},
};
use anyhow::{Context, Result};
use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

mod suspend;

/// What a piece of work evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trial {
    /// A standard trial.
    Standard,
    /// The confirmation trial of a creature that would set an island record
    /// (`confirm_config`). It goes before standard work.
    Confirm,
}

/// Cost of one confirmation trial relative to standard trials (twice the
/// steps at twice the solver passes).
const CONFIRM_COST: f64 = 4.0;

/// Largest piece of work `evaluate` queues at once: about one second of the
/// RTX 4060, the unit size that measured fastest end to end.
pub const WORK_UNIT: usize = 196_608;

/// The settings of a confirmation trial: the same world and screen as the
/// standard trial, at the fine physics (`physics::Fidelity::fine()`), from
/// the same pose. It runs no early rung: a rung is a prediction about the
/// standard trial.
pub fn confirm_config(cfg: &Config) -> Config {
    Config {
        fidelity: Some(crate::physics::Fidelity::fine()),
        rungs: None,
        ..cfg.clone()
    }
}

/// Creatures waiting for an engine.
struct Work {
    /// Caller's work tag to return in results.
    tag: u64,
    /// A standard or a confirmation trial.
    trial: Trial,
    population: Arc<Population>,
    /// Creatures of `population` to evaluate, in order; `None` for all.
    members: Option<Vec<usize>>,
    config: Arc<Config>,
    /// Largest body among the members, in nodes.
    max_nodes: usize,
}

#[derive(Clone)]
struct QueuedUnit {
    /// Engine's work ticket.
    ticket: u64,
    /// Caller's work tag to return in results.
    tag: u64,
    /// A standard or a confirmation trial.
    trial: Trial,
    /// Creature indices of the queued work, in unit order.
    members: Vec<usize>,
    population: Arc<Population>,
    config: Config,
    /// The scheduler session the unit belongs to. Units of an older session
    /// finish on their engine and their results are dropped.
    session: u64,
    /// How many times this unit has moved to another engine.
    retries: u8,
}

/// Finished creatures of one piece of work.
#[derive(Clone, Debug)]
pub struct Done {
    /// Caller's work tag.
    pub tag: u64,
    pub trial: Trial,
    /// Creature indices in the queued population.
    pub members: Vec<usize>,
    /// Evaluation metrics in member order.
    pub metrics: Vec<EvaluationMetrics>,
}

/// How to open a GPU engine again after it failed, and how long to wait
/// before each attempt.
pub struct Reopen {
    open: Box<dyn FnMut() -> Result<Box<dyn Engine>> + Send>,
    /// Delay before each recovery attempt.
    backoff: Vec<Duration>,
}

/// Stands in for a failed engine while its replacement opens. It holds no
/// device, so the driver can release the old one.
struct RetiredEngine;

impl Engine for RetiredEngine {
    fn name(&self) -> String {
        "retired GPU".into()
    }
    fn max_nodes(&self) -> usize {
        0
    }
    fn free_slots(&self) -> usize {
        0
    }
    fn submit_shared(&mut self, _: Arc<Population>, _: &Config) -> Result<u64> {
        anyhow::bail!("The GPU engine was retired")
    }
    fn poll(&mut self) -> Result<Option<crate::engine::Finished>> {
        Ok(None)
    }
    fn wait(&mut self, _: Duration) {}
}

pub struct Device {
    pub engine: Box<dyn Engine>,
    /// Set for a GPU that can be opened again after a failure.
    reopen: Option<Reopen>,
    /// Reopens since the last unit finished on this device.
    recoveries: usize,
    /// Set when the engine reported a failure. A failed GPU is reopened, and
    /// one that does not open again stops the scheduler.
    failure: Option<String>,
    /// Exact submitted input remains available until its result is accepted.
    queued: VecDeque<QueuedUnit>,
    /// Measured standard creatures per wall second while the device has
    /// work queued. Units on separate queues overlap, so their own busy
    /// times would overstate the device's time.
    pub rate: f64,
    /// Exponentially decayed standard creatures finished and busy wall
    /// seconds behind `rate`, and when they were last updated.
    rate_work: f64,
    rate_time: f64,
    rate_at: Option<Instant>,
    /// Wall seconds with nothing queued, and since when the device has been
    /// idle (sampled whenever the scheduler pumps or collects).
    pub idle_seconds: f64,
    idle_since: Option<Instant>,
    pub creatures: u64,
    pub busy_seconds: f64,
}

impl Device {
    fn new(engine: Box<dyn Engine>, rate: f64) -> Self {
        Self {
            engine,
            reopen: None,
            recoveries: 0,
            failure: None,
            queued: VecDeque::new(),
            rate,
            creatures: 0,
            busy_seconds: 0.0,
            rate_work: 0.0,
            rate_time: 0.0,
            rate_at: None,
            idle_seconds: 0.0,
            idle_since: None,
        }
    }
    /// Advances the rate average to `now`, counting the time since the last
    /// update as busy when work was queued, plus `done` finished standard
    /// creatures. Averages decay over about 20 s of wall time.
    fn update_rate(&mut self, now: Instant, busy: bool, done: usize) {
        if let Some(at) = self.rate_at {
            let dt = now.duration_since(at).as_secs_f64();
            let decay = (-dt / 20.0).exp();
            self.rate_work *= decay;
            self.rate_time *= decay;
            if busy {
                self.rate_time += dt;
            }
        }
        self.rate_at = Some(now);
        self.rate_work += done as f64;
        if self.rate_time >= 2.0 && self.rate_work >= 1024.0 {
            self.rate = self.rate_work / self.rate_time;
        }
    }
    /// Updates the idle time and rate average at the given instant.
    fn sample_idle(&mut self, now: Instant) {
        self.update_rate(now, self.idle_since.is_none(), 0);
        match (self.queued.is_empty(), self.idle_since) {
            (true, None) => self.idle_since = Some(now),
            (false, Some(since)) => {
                self.idle_seconds += now.duration_since(since).as_secs_f64();
                self.idle_since = None;
            }
            _ => {}
        }
    }
}

pub struct Scheduler {
    pub devices: Vec<Device>,
    /// Work waiting for an engine: confirmations first, then standard work.
    confirms: VecDeque<Work>,
    work: VecDeque<Work>,
    /// Current session ID for dropping old in-flight results.
    session: u64,
    /// Seconds spent packing work for the GPU.
    pub packing_seconds: f64,
    /// Totals since start: confirmation trials submitted, and the device
    /// busy seconds they took.
    pub confirms_submitted: u64,
    pub confirm_busy_seconds: f64,
    /// Lane-steps of standard trials since start, per lane class
    /// (`kernel::CLASSES`): steps a creature ran times its lanes.
    pub lane_steps: [u64; 4],
    /// Messages for the player (a GPU lost and reopened), taken by the worker.
    notices: Vec<String>,
    /// Developer hook (`EVOLUTION_SIMULATE_GPU_LOSS=N`): the GPU fails once,
    /// after N units of its results were collected.
    simulate_loss_after: Option<u64>,
    collected_units: u64,
    /// Evaluation held for a developer measurement (see `suspend.rs`).
    suspension: suspend::Suspension,
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
    fn with_devices(devices: Vec<Device>) -> Self {
        Self {
            devices,
            confirms: VecDeque::new(),
            work: VecDeque::new(),
            session: 0,
            packing_seconds: 0.0,
            confirms_submitted: 0,
            confirm_busy_seconds: 0.0,
            lane_steps: [0; 4],
            notices: Vec::new(),
            simulate_loss_after: None,
            collected_units: 0,
            suspension: Default::default(),
        }
    }

    /// Opens the named primary GPU and the other GPUs listed in
    /// `EVOLUTION_DEVICES` (off by default; `primary` or `off` for none). The
    /// game needs the primary GPU: without it this fails.
    pub fn new(primary: &str) -> Result<Self> {
        let gpu = engine::gpu_engine(primary, 64).with_context(|| {
            format!(
                "The game needs an NVIDIA GPU with the CUDA driver and NVRTC, and GPU {primary:?} did not open"
            )
        })?;
        // The primary GPU scores the archive, so it records replays.
        gpu.publish_replays();
        let mut device = Device::new(Box::new(gpu), 180_000.0);
        let name = primary.to_owned();
        device.reopen = Some(Reopen {
            open: Box::new(move || {
                let gpu = engine::gpu_engine(&name, 64)?;
                gpu.publish_replays();
                Ok(Box::new(gpu) as Box<dyn Engine>)
            }),
            backoff: [1u64, 4, 10].map(Duration::from_secs).to_vec(),
        });
        let mut devices = vec![device];
        let extra = std::env::var("EVOLUTION_DEVICES").ok();
        for name in secondary_device_names(extra.as_deref()) {
            if primary.to_lowercase().contains(&name.to_lowercase()) {
                continue;
            }
            match engine::gpu_engine(name, 64) {
                Ok(engine) => devices.push(Device::new(Box::new(engine), 40_000.0)),
                Err(err) => eprintln!("Evaluation device {name:?} unavailable: {err:#}"),
            }
        }
        let mut scheduler = Self::with_devices(devices);
        scheduler.simulate_loss_after = std::env::var("EVOLUTION_SIMULATE_GPU_LOSS")
            .ok()
            .and_then(|v| v.parse().ok());
        Ok(scheduler)
    }

    /// Returns the names of all GPU devices, joined with " + ".
    pub fn names(&self) -> String {
        self.devices
            .iter()
            .map(|d| d.engine.name())
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// Units on engines plus work waiting for one. A failed engine that has
    /// not been retired also counts as one, so drain loops keep asking for
    /// results and the failure surfaces instead of looking like a finished
    /// round.
    pub fn in_flight(&self) -> usize {
        self.devices.iter().map(|d| d.queued.len()).sum::<usize>()
            + self.confirms.len()
            + self.work.len()
            + self.devices.iter().filter(|d| d.failure.is_some()).count()
    }

    /// Units on engines, plus failed engines not yet retired.
    pub fn on_engines(&self) -> usize {
        self.devices.iter().map(|d| d.queued.len()).sum::<usize>()
            + self.devices.iter().filter(|d| d.failure.is_some()).count()
    }

    /// Queues creatures `members` of `population` (all of them for `None`)
    /// for one unit with `config`. Their results come back with `tag`.
    pub fn queue(
        &mut self,
        tag: u64,
        trial: Trial,
        population: Arc<Population>,
        members: Option<Vec<usize>>,
        config: Arc<Config>,
    ) {
        let max_nodes = match &members {
            Some(members) => members
                .iter()
                .map(|&i| population.genomes[i].node_count)
                .max(),
            None => population.genomes.iter().map(|g| g.node_count).max(),
        };
        let Some(max_nodes) = max_nodes else {
            return;
        };
        let work = Work {
            tag,
            trial,
            population,
            members,
            config,
            max_nodes,
        };
        match trial {
            Trial::Confirm => self.confirms.push_back(work),
            Trial::Standard => self.work.push_back(work),
        }
    }

    /// Gives `config` to queued standard work that has not reached an engine
    /// and whose physics differs from it (a world change), and returns the
    /// tags of that work: it now runs in the new world.
    pub fn retarget(&mut self, config: &Arc<Config>) -> Vec<u64> {
        let mut tags = Vec::new();
        for work in &mut self.work {
            if work.tag & crate::ring::WILD == 0 && work.config.physics_differs(config) {
                work.config = Arc::clone(config);
                tags.push(work.tag);
            }
        }
        tags
    }

    /// Drops all waiting work and starts a new session: units still on an
    /// engine finish there and their results are dropped.
    pub fn reset(&mut self) {
        self.confirms.clear();
        self.work.clear();
        self.session += 1;
    }

    /// Total GPU memory allocated across all devices.
    pub fn allocated_bytes(&self) -> u64 {
        self.devices
            .iter()
            .map(|d| d.engine.allocated_bytes())
            .sum()
    }

    /// Hands waiting work to every engine with a free slot, confirmations
    /// first.
    pub fn pump(&mut self) -> Result<()> {
        let now = Instant::now();
        if self.may_submit() {
            for index in 0..self.devices.len() {
                loop {
                    let device = &self.devices[index];
                    if device.failure.is_some() {
                        break;
                    }
                    let capacity = device.engine.max_nodes();
                    // The first waiting work this engine can hold. A GPU has
                    // a slot of its own for confirmation trials, so they go
                    // whenever it is free, and standard work when a standard
                    // slot is.
                    let fits = |w: &Work| w.max_nodes <= capacity;
                    let confirm_open = device.engine.free_confirm_slots() > 0;
                    let standard_open = device.engine.free_slots() > 0;
                    let work = if confirm_open && let Some(at) = self.confirms.iter().position(fits)
                    {
                        self.confirms.remove(at)
                    } else if standard_open && let Some(at) = self.work.iter().position(fits) {
                        self.work.remove(at)
                    } else {
                        None
                    };
                    let Some(work) = work else {
                        break;
                    };
                    if let Err(work) = self.submit(index, work) {
                        // The work waits for another engine.
                        match work.trial {
                            Trial::Confirm => self.confirms.push_front(*work),
                            Trial::Standard => self.work.push_front(*work),
                        }
                        break;
                    }
                }
            }
        }
        for device in &mut self.devices {
            device.sample_idle(now);
        }
        Ok(())
    }

    /// Submits `work` to device `index` as one unit. On failure the device
    /// is marked failed and the work comes back.
    fn submit(&mut self, index: usize, work: Work) -> std::result::Result<(), Box<Work>> {
        let started = Instant::now();
        let (population, members) = match &work.members {
            None => (
                Arc::clone(&work.population),
                (0..work.population.genomes.len()).collect(),
            ),
            Some(members) => (Arc::new(work.population.subset(members)), members.clone()),
        };
        let device = &mut self.devices[index];
        match device
            .engine
            .submit_shared(Arc::clone(&population), &work.config)
        {
            Ok(ticket) => {
                self.packing_seconds += started.elapsed().as_secs_f64();
                if work.trial == Trial::Confirm {
                    self.confirms_submitted += members.len() as u64;
                }
                device.queued.push_back(QueuedUnit {
                    ticket,
                    tag: work.tag,
                    trial: work.trial,
                    members,
                    population,
                    config: (*work.config).clone(),
                    session: self.session,
                    retries: 0,
                });
                Ok(())
            }
            Err(error) => {
                device.failure = Some(format!("{} failed: {error:#}", device.engine.name()));
                Err(Box::new(work))
            }
        }
    }

    /// Returns finished work, waiting up to `timeout` when nothing is ready.
    pub fn collect(&mut self, timeout: Duration) -> Result<Vec<Done>> {
        let mut out = Vec::new();
        let deadline = Instant::now() + timeout;
        loop {
            for device in &mut self.devices {
                if device.failure.is_some() {
                    continue;
                }
                loop {
                    if self.simulate_loss_after == Some(self.collected_units) {
                        self.simulate_loss_after = None;
                        device.failure = Some("simulated device loss".into());
                        break;
                    }
                    match device.engine.poll() {
                        Ok(None) => break,
                        Ok(Some(done)) => {
                            self.collected_units += 1;
                            device.recoveries = 0;
                            // Engines with several queues finish units in any order.
                            let position = device
                                .queued
                                .iter()
                                .position(|unit| unit.ticket == done.ticket)
                                .context("Unexpected evaluation result")?;
                            let queued = &device.queued[position];
                            anyhow::ensure!(
                                done.results.len() == queued.members.len(),
                                "Evaluation result count mismatch: expected {}, received {}",
                                queued.members.len(),
                                done.results.len()
                            );
                            let unit = device
                                .queued
                                .remove(position)
                                .expect("validated queued unit");
                            device.busy_seconds += done.busy_seconds;
                            let count = unit.members.len();
                            match unit.trial {
                                Trial::Standard => {
                                    device.creatures += count as u64;
                                    device.update_rate(Instant::now(), true, count);
                                    count_lane_steps(
                                        &mut self.lane_steps,
                                        &unit.population,
                                        &done.results,
                                        &unit.config,
                                    );
                                }
                                Trial::Confirm => {
                                    // The rate counts standard-trial
                                    // equivalents, so a device busy with
                                    // confirmations still shows its capacity.
                                    self.confirm_busy_seconds += done.busy_seconds;
                                    device.update_rate(
                                        Instant::now(),
                                        true,
                                        (count as f64 * CONFIRM_COST) as usize,
                                    );
                                }
                            }
                            if unit.session != self.session {
                                continue;
                            }
                            let metrics = done
                                .results
                                .iter()
                                .enumerate()
                                .map(|(k, r)| to_metrics(&unit.population, k, r, &unit.config))
                                .collect();
                            out.push(Done {
                                tag: unit.tag,
                                trial: unit.trial,
                                members: unit.members,
                                metrics,
                            });
                        }
                        Err(error) => {
                            let name = device.engine.name();
                            device.failure = Some(format!("{name} failed: {error:#}"));
                            break;
                        }
                    }
                }
            }
            let now = Instant::now();
            for device in &mut self.devices {
                device.sample_idle(now);
            }
            if let Err(error) = self.retire_failed() {
                // Completed output is delivered before a terminal failure.
                if out.is_empty() {
                    return Err(error);
                }
                return Ok(out);
            }
            if !out.is_empty() || Instant::now() >= deadline {
                return Ok(out);
            }
            let wait = deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(2));
            let Some(device) = self
                .devices
                .iter_mut()
                .find(|d| d.failure.is_none() && !d.queued.is_empty())
            else {
                return Ok(out);
            };
            device.engine.wait(wait);
        }
    }

    /// Handles every engine that reported a failure since the last pass. A
    /// failed GPU is reopened and gets its unfinished units again. A GPU that
    /// does not open again is terminal.
    fn retire_failed(&mut self) -> Result<()> {
        for index in 0..self.devices.len() {
            if self.devices[index].failure.is_some() && !self.retire(index) {
                let error = self.devices[index]
                    .failure
                    .clone()
                    .unwrap_or_else(|| "Evaluation device failed".into());
                anyhow::bail!("{error}");
            }
        }
        Ok(())
    }

    /// Reopens one failed device. Returns false when the failure is terminal.
    fn retire(&mut self, index: usize) -> bool {
        let reason = self.devices[index]
            .failure
            .clone()
            .unwrap_or_else(|| "evaluation failed".into());
        // Units of an older session need no second run.
        let session = self.session;
        self.devices[index]
            .queued
            .retain(|unit| unit.session == session);
        if self.devices[index].reopen.is_some() {
            if self.recover(index, &reason) {
                return true;
            }
            self.devices[index].failure = Some(format!("{reason}; the GPU could not be reopened"));
        }
        false
    }

    /// Developer hook: the GPU fails once, after `units` units of results were
    /// collected (`EVOLUTION_SIMULATE_GPU_LOSS` sets it at start).
    pub fn simulate_gpu_loss_after(&mut self, units: u64) {
        self.simulate_loss_after = Some(self.collected_units + units);
    }

    /// Messages for the player since the last call.
    pub fn take_notices(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notices)
    }

    /// Opens a failed GPU again: the broken engine is dropped, the scheduler
    /// waits (a little longer after each failed attempt), opens a new engine
    /// and submits the unfinished units again with their exact inputs, so
    /// they give the same results. Returns false when every attempt failed;
    /// the units stay queued.
    fn recover(&mut self, index: usize, reason: &str) -> bool {
        let Some(mut reopen) = self.devices[index].reopen.take() else {
            return false;
        };
        let units: Vec<QueuedUnit> = self.devices[index].queued.drain(..).collect();
        self.devices[index].engine = Box::new(RetiredEngine);
        let total = reopen.backoff.len();
        let first = self.devices[index].recoveries.min(total);
        for attempt in first..total {
            self.notices.push(format!(
                "The GPU failed ({reason}). Reopening it, attempt {} of {total}.",
                attempt + 1
            ));
            std::thread::sleep(reopen.backoff[attempt]);
            let mut engine = match (reopen.open)() {
                Ok(engine) => engine,
                Err(error) => {
                    self.notices
                        .push(format!("The GPU did not open again: {error:#}"));
                    continue;
                }
            };
            let mut queued = VecDeque::with_capacity(units.len());
            let mut failed = None;
            for unit in &units {
                match engine.submit_shared(Arc::clone(&unit.population), &unit.config) {
                    Ok(ticket) => queued.push_back(QueuedUnit {
                        ticket,
                        retries: unit.retries.saturating_add(1),
                        ..unit.clone()
                    }),
                    Err(error) => {
                        failed = Some(error);
                        break;
                    }
                }
            }
            if let Some(error) = failed {
                self.notices
                    .push(format!("The reopened GPU rejected its work: {error:#}"));
                continue;
            }
            let device = &mut self.devices[index];
            device.engine = engine;
            device.queued = queued;
            device.failure = None;
            device.recoveries = attempt + 1;
            device.reopen = Some(reopen);
            self.notices.push(format!(
                "The GPU is back. {} units started again.",
                units.len()
            ));
            eprintln!("{reason}; GPU reopened, {} units resubmitted", units.len());
            return true;
        }
        self.notices
            .push("The GPU could not be reopened. Evolution stops.".into());
        self.devices[index].queued = units.into_iter().collect();
        false
    }

    /// Evaluates `indices` of `pop` with one standard trial each at `cfg`
    /// and returns the metrics in the same order. Results of other work
    /// that finishes meanwhile are dropped, so call this on an idle
    /// scheduler.
    pub fn evaluate(
        &mut self,
        pop: &Population,
        indices: &[usize],
        cfg: &Config,
    ) -> Result<Vec<EvaluationMetrics>> {
        // Tags above every caller's: the ring tags with block sequences.
        const TAG: u64 = u64::MAX / 2;
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let whole = indices.len() == pop.genomes.len()
            && indices.iter().enumerate().all(|(a, &b)| a == b)
            && indices.len() <= WORK_UNIT;
        let population = Arc::new(pop.clone());
        let config = Arc::new(cfg.clone());
        let chunks: Vec<&[usize]> = indices.chunks(WORK_UNIT).collect();
        for (k, chunk) in chunks.iter().enumerate() {
            self.queue(
                TAG + k as u64,
                Trial::Standard,
                Arc::clone(&population),
                (!whole).then(|| chunk.to_vec()),
                Arc::clone(&config),
            );
        }
        drop(population);
        let mut out = vec![EvaluationMetrics::default(); indices.len()];
        let mut remaining = chunks.len();
        while remaining > 0 {
            self.pump()?;
            for done in self.collect(Duration::from_millis(50))? {
                let Some(k) = done
                    .tag
                    .checked_sub(TAG)
                    .filter(|&k| k < chunks.len() as u64)
                else {
                    continue;
                };
                let start = k as usize * WORK_UNIT;
                for (at, metric) in done.metrics.into_iter().enumerate() {
                    out[start + at] = metric;
                }
                remaining -= 1;
            }
        }
        Ok(out)
    }
}

/// Steps a trial ran: to its fall, its screen, or its end.
fn trial_steps(r: &GpuResult, cfg: &Config) -> u32 {
    let ended = if r.fall_time > 0.0 {
        r.fall_time
    } else {
        r.screened
    };
    if ended > 0.0 {
        ((ended * cfg.fidelity().rate as f32).round() as u32).clamp(1, cfg.steps().max(1))
    } else {
        cfg.steps().max(1)
    }
}

/// Adds each creature's steps times the lanes of its class to `totals`.
fn count_lane_steps(totals: &mut [u64; 4], pop: &Population, results: &[GpuResult], cfg: &Config) {
    let classes = crate::kernel::CLASSES;
    for (genome, r) in pop.genomes.iter().zip(results) {
        let Some(lanes) = crate::kernel::class_of(genome.node_count, genome.muscle_count) else {
            continue;
        };
        let class = classes.iter().position(|&w| w == lanes).unwrap_or(0);
        totals[class] += u64::from(trial_steps(r, cfg)) * lanes as u64;
    }
}

/// Converts a raw kernel result to the archive's normalized metrics.
pub fn to_metrics(
    pop: &Population,
    index: usize,
    r: &GpuResult,
    cfg: &Config,
) -> EvaluationMetrics {
    // Behavior totals end at a fall or the screen, so they average over the
    // steps walked.
    let steps = trial_steps(r, cfg);
    let contact_denominator = (steps * pop.genomes[index].node_count as u32) as f32;
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
            mean_height: (r.height_sum / steps as f32).max(0.0),
            feet: r.feet() as f32,
        },
        excluded: false,
        screened: r.screened > 0.0,
        screen_x: r.screen_x,
        fine: false,
        trace: r.rung_trace(),
    }
}

#[cfg(test)]
mod tests;
