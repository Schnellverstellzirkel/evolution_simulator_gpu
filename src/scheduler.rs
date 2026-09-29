//! Routes a generation to the selected GPU devices. Additional GPUs require
//! an explicit `EVOLUTION_DEVICES` selection; CPU evaluation is reserved for
//! CPU-only runs or GPU failover.
//!
//! Work is handed out in body-size order so GPU units fill few, large buckets.
//! Each engine keeps up to two units queued; unit sizes follow each engine's
//! measured rate so that all engines finish a generation together. The
//! selected evaluation engine defines fitness; CPU/GPU comparisons are only
//! diagnostics, and CPU output never validates or lowers a GPU score.
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

/// Standard trials one fine check costs (four times the steps at four
/// times the solver passes; 7.7 measured on the RTX 4060 for evolved bodies).
const CHECK_COST: f64 = 8.0;

/// The configuration of a contender's check trial: fine physics
/// (`physics::Fidelity::fine()`) and the same screen as the standard trial.
pub fn check_config(cfg: &Config) -> Config {
    Config {
        fidelity: Some(crate::physics::Fidelity::fine()),
        ..cfg.clone()
    }
}

/// The score a contender keeps after its check trial, and whether it must
/// stay out of every archive. The worse trial counts, and a check that the
/// screen stopped is not robust: the creature enters no archive.
pub fn check_verdict(standard: &EvaluationMetrics, check: &GpuResult) -> (f32, bool) {
    (standard.fitness.min(check.fitness), check.screened > 0.0)
}

/// What a held contender needs, decided again each time checks are sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckNeed {
    /// It can no longer enter an archive: its standard result is final.
    Release,
    /// A check trial. Contenders for the same archive `cell` share it: only
    /// the best waiting one is checked while no check for that cell is in
    /// flight, and the rest are decided again once it lands. `None` never
    /// shares.
    Check { cell: Option<u64> },
}

/// Which engine backs a device. A failed GPU can hand its work to the CPU;
/// a failed CPU is terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceKind {
    Gpu,
    Cpu,
}

#[derive(Clone)]
struct QueuedUnit {
    ticket: u64,
    indices: Vec<usize>,
    trial: Trial,
    population: Arc<Population>,
    config: Config,
    /// How many times this unit has moved to another engine.
    retries: u8,
}

/// How to open a GPU engine again after it failed, and how long to wait
/// before each attempt.
pub struct Reopen {
    open: Box<dyn FnMut() -> Result<Box<dyn Engine>> + Send>,
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
    kind: DeviceKind,
    /// Set when the engine reported a failure. A failed GPU is retired and its
    /// unfinished units move to the CPU; a failed CPU stops the scheduler.
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
    /// Smallest unit worth submitting to this engine.
    min_unit: usize,
    /// Seconds of work per unit; short on a GPU that also drives the display.
    unit_seconds: f64,
    /// A reserve engine takes over evaluation work only when no primary
    /// engine remains healthy.
    reserve: bool,
}

impl Device {
    fn new(
        engine: Box<dyn Engine>,
        kind: DeviceKind,
        rate: f64,
        min_unit: usize,
        unit_seconds: f64,
    ) -> Self {
        Self {
            engine,
            reopen: None,
            recoveries: 0,
            kind,
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
            min_unit,
            unit_seconds,
            reserve: false,
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
        if self.rate_time >= 2.0 && self.rate_work >= self.min_unit as f64 {
            self.rate = self.rate_work / self.rate_time;
        }
    }
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
    fn queued_creatures(&self) -> usize {
        self.queued.iter().map(|unit| unit.indices.len()).sum()
    }
}

struct Round {
    order: Vec<usize>,
    cursor: usize,
    /// Creatures skipped by an engine that cannot hold them.
    oversize: Vec<usize>,
    /// Creatures whose submission failed; offered to another engine first.
    retry: Vec<usize>,
}

pub struct Scheduler {
    pub devices: Vec<Device>,
    round: Option<Round>,
    pub packing_seconds: f64,
    /// Trials per creature (1 or 2, 2 in the game). With 2, a
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
    /// Archive cell claimed by each contender being checked, and the claimed
    /// cells.
    checking: HashMap<usize, u64>,
    busy_cells: std::collections::HashSet<u64>,
    /// Decided contenders waiting for a check unit.
    ready: Vec<usize>,

    /// Standard work has stopped (a pause, a save, the end of a run): every
    /// waiting contender goes out for its check at once instead of one per
    /// cell, so draining takes one check round, not a chain of them.
    draining: bool,
    /// Waiting contenders that no longer need a check, with their final result.
    released: Vec<(usize, EvaluationMetrics)>,
    /// Totals since start: check trials submitted, and contenders released
    /// without one.
    pub checks_submitted: u64,
    pub checks_released: u64,
    /// Totals since start: check units submitted, and contenders dropped
    /// because another contender held their cell's check.
    pub check_units: u64,
    pub checks_dropped: u64,
    /// Device busy seconds spent on check units (totals since start).
    pub check_busy_seconds: f64,
    /// Why the primary GPU was not used, reported once at startup.
    startup_failure: Option<String>,
    /// Messages for the player (a GPU lost and reopened), taken by the worker.
    notices: Vec<String>,
    /// Developer hook (`EVOLUTION_SIMULATE_GPU_LOSS=N`): the GPU fails once,
    /// after N units of its results were collected.
    simulate_loss_after: Option<u64>,
    collected_units: u64,
    /// Trial settings of creatures queued through `extend_with`: every unit
    /// holds creatures of one setting only. Creatures without an entry use
    /// the settings passed to `pump`.
    cfg_of: HashMap<usize, Arc<Config>>,
    /// In-order absorption of results (see `ordered.rs`).
    ordered: ordered::Ordered,
    /// Evaluation held for a developer measurement (see `suspend.rs`).
    suspension: suspend::Suspension,
}

mod ordered;
mod suspend;

fn same_cfg(a: &Option<Arc<Config>>, b: &Option<Arc<Config>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn reserve_cpu_when_gpu_available(devices: &mut [Device]) {
    if devices.iter().any(|device| device.kind == DeviceKind::Gpu) {
        for device in devices {
            if device.kind == DeviceKind::Cpu {
                device.reserve = true;
            }
        }
    }
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
    /// general worker. Zero or a one-worker budget disables the separate CPU
    /// engine. If the primary GPU cannot open, evaluation falls back to the
    /// CPU; with the separate pool disabled, it shares the general Rayon pool.
    pub fn new(primary: &str) -> Result<Self> {
        let step_range = env_or("EVOLUTION_GPU_CHUNK", crate::gpu::DEFAULT_STEP_RANGE);
        let mut devices = Vec::new();
        let mut startup_failure = None;
        match engine::gpu_engine(primary, 64, step_range) {
            // Units long enough to fill the GPU between the step-range
            // barriers, short enough that the in-flight copies stay small.
            // With screening and segments at 3M: 0.5 s units measured 165k
            // creatures/s end to end, 1 s 187k (12.5 GB peak RSS), 2 s 177k,
            // 3 s 169k (15.0 GB).
            Ok(gpu) => {
                // The primary GPU scores the archive, so it records replays.
                gpu.publish_replays();
                let mut device = Device::new(
                    Box::new(gpu),
                    DeviceKind::Gpu,
                    180_000.0,
                    8192,
                    env_or("EVOLUTION_UNIT_SECONDS", 1.0),
                );
                let name = primary.to_owned();
                device.reopen = Some(Reopen {
                    open: Box::new(move || {
                        let gpu = engine::gpu_engine(&name, 64, step_range)?;
                        gpu.publish_replays();
                        Ok(Box::new(gpu) as Box<dyn Engine>)
                    }),
                    backoff: [1u64, 4, 10].map(Duration::from_secs).to_vec(),
                });
                devices.push(device)
            }
            Err(error) => {
                let message = format!("Primary GPU {primary:?} unavailable: {error:#}");
                eprintln!("{message}; evaluating on the CPU");
                startup_failure = Some(message);
            }
        }
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
                    DeviceKind::Gpu,
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
                DeviceKind::Cpu,
                30_000.0,
                1024,
                env_or("EVOLUTION_CPU_UNIT_SECONDS", 1.0),
            ));
        } else {
            // Without a separate CPU pool the CPU engine shares the general
            // pool. When a GPU is available it stands by for GPU failure;
            // without a GPU it evaluates the whole run.
            let mut shared = Device::new(
                Box::new(engine::cpu_engine_shared()?),
                DeviceKind::Cpu,
                30_000.0,
                64,
                env_or("EVOLUTION_CPU_UNIT_SECONDS", 1.0),
            );
            shared.reserve = !devices.is_empty();
            devices.push(shared);
        }
        // A configured CPU pool also stays idle while any GPU is healthy.
        // This keeps every score in a GPU run on the GPU path; CPU results
        // enter only a CPU-only run or an explicit failover after GPU loss.
        reserve_cpu_when_gpu_available(&mut devices);
        Ok(Self {
            devices,
            round: None,
            packing_seconds: 0.0,
            robust_trials: 2,
            checks: Vec::new(),
            checks_since: None,
            held: HashMap::new(),
            checking: HashMap::new(),
            busy_cells: Default::default(),
            ready: Vec::new(),
            draining: false,
            released: Vec::new(),
            checks_submitted: 0,
            check_busy_seconds: 0.0,
            checks_released: 0,
            check_units: 0,
            checks_dropped: 0,
            startup_failure,
            notices: Vec::new(),
            simulate_loss_after: std::env::var("EVOLUTION_SIMULATE_GPU_LOSS")
                .ok()
                .and_then(|v| v.parse().ok()),
            collected_units: 0,
            cfg_of: HashMap::new(),
            ordered: Default::default(),
            suspension: Default::default(),
        })
    }

    /// A scheduler with only the CPU engine, for machines without a GPU and
    /// for tests.
    pub fn cpu_only(threads: usize) -> Result<Self> {
        Ok(Self {
            devices: vec![Device::new(
                Box::new(engine::cpu_engine(threads.max(1))?),
                DeviceKind::Cpu,
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
            checking: HashMap::new(),
            busy_cells: Default::default(),
            ready: Vec::new(),
            draining: false,
            released: Vec::new(),
            checks_submitted: 0,
            check_busy_seconds: 0.0,
            checks_released: 0,
            check_units: 0,
            checks_dropped: 0,
            startup_failure: None,
            notices: Vec::new(),
            simulate_loss_after: None,
            collected_units: 0,
            cfg_of: HashMap::new(),
            ordered: Default::default(),
            suspension: Default::default(),
        })
    }

    pub fn names(&self) -> String {
        self.devices
            .iter()
            .map(|d| d.engine.name())
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// Why the primary GPU was not used, for reporting once at startup.
    pub fn startup_failure(&self) -> Option<&str> {
        self.startup_failure.as_deref()
    }

    /// Queued units, plus one while contenders still wait for their checks.
    /// A failed engine that has not been retired also counts as one, so drain
    /// loops keep asking for results and the failure surfaces instead of
    /// looking like a finished round.
    pub fn in_flight(&self) -> usize {
        self.devices.iter().map(|d| d.queued.len()).sum::<usize>()
            + usize::from(!self.held.is_empty() || !self.released.is_empty())
            + self.devices.iter().filter(|d| d.failure.is_some()).count()
    }

    /// Queues check units for waiting contenders. Checks go first, but wait
    /// to fill a reasonable unit while standard work remains.
    pub fn pump_checks(
        &mut self,
        pop: &Population,
        cfg: &Config,
        mut need: impl FnMut(usize, &EvaluationMetrics) -> CheckNeed,
    ) -> Result<()> {
        // Decide new contenders against the live archives. Each archive cell
        // gets one check at a time: the best new contender for a free cell
        // claims it, and the other contenders for that cell, or for a cell
        // whose check is running, are dropped: they count as evaluated but
        // enter no archive. Checks cost several standard trials, so their
        // volume is bounded by the archive's cells, not by the population:
        // on ground where fine checks fail for most gaits, nearly every
        // creature beats the low checked elites, and checking each one held
        // the whole 3M population for minutes.
        let undecided = std::mem::take(&mut self.checks);
        let mut decided = Vec::with_capacity(undecided.len());
        let mut champions: HashMap<u64, (f32, usize)> = HashMap::new();
        for i in undecided {
            let Some(metric) = self.held.get(&i) else {
                continue;
            };
            match need(i, metric) {
                CheckNeed::Release => {
                    let metric = self.held.remove(&i).expect("held contender");
                    self.released.push((i, metric));
                    self.checks_released += 1;
                }
                CheckNeed::Check { cell } => {
                    // While draining, every contender still held is checked.
                    let cell = cell.filter(|_| !self.draining);
                    if let Some(cell) = cell
                        && !self.busy_cells.contains(&cell)
                    {
                        let entry = champions.entry(cell).or_insert((metric.fitness, i));
                        if metric.fitness > entry.0 {
                            *entry = (metric.fitness, i);
                        }
                    }
                    decided.push((i, cell));
                }
            }
        }
        for (i, cell) in decided {
            match cell {
                None => self.ready.push(i),
                Some(cell) if champions.get(&cell).is_some_and(|&(_, c)| c == i) => {
                    self.busy_cells.insert(cell);
                    self.checking.insert(i, cell);
                    self.ready.push(i);
                }
                Some(_) => {
                    let mut metric = self.held.remove(&i).expect("held contender");
                    metric.unchecked = true;
                    self.released.push((i, metric));
                    self.checks_dropped += 1;
                }
            }
        }
        if self.ready.is_empty() {
            self.checks_since = None;
            return Ok(());
        }
        let waited = *self.checks_since.get_or_insert_with(Instant::now);
        let waited = waited.elapsed() > Duration::from_millis(500);
        // A check faces the same screen as the standard trial it checks: a
        // contender whose fine trial falls below the bar at the screen is
        // not robust, and its check stops there instead of running the full
        // trial at four times the steps.
        if !self.may_submit() {
            return Ok(());
        }
        let standby = self.reserves_standing_by();
        let cfg_of = &self.cfg_of;
        let max_check_units = env_or("EVOLUTION_CHECK_UNITS", 2usize).max(1);
        for device in &mut self.devices {
            if standby && device.reserve {
                continue;
            }
            // A check costs several standard trials; keep units about as long.
            let size =
                ((device.rate * device.unit_seconds / 6.0) as usize).max(device.min_unit / 2);
            // A check unit runs four times as many steps as a standard one and
            // holds its queue slot that long, so a device runs one at a time,
            // or a second one while more than a unit's worth is waiting.
            let check_units = if self.draining {
                usize::MAX
            } else if self.ready.len() > size {
                max_check_units
            } else {
                1
            };
            while device.failure.is_none()
                && device.engine.free_slots() > 0
                && !self.ready.is_empty()
                && device
                    .queued
                    .iter()
                    .filter(|unit| unit.trial == Trial::Check)
                    .count()
                    < check_units
            {
                if self.ready.len() < device.min_unit / 2 && self.round.is_some() && !waited {
                    break;
                }
                let capacity = device.engine.max_nodes();
                let mut indices = Vec::with_capacity(size.min(self.ready.len()));
                let mut rest = Vec::new();
                let mut unit_cfg: Option<Option<Arc<Config>>> = None;
                for i in self.ready.drain(..) {
                    let c = cfg_of.get(&i).cloned();
                    let same = match &unit_cfg {
                        None => true,
                        Some(u) => same_cfg(u, &c),
                    };
                    if indices.len() < size && pop.genomes[i].node_count <= capacity && same {
                        if unit_cfg.is_none() {
                            unit_cfg = Some(c);
                        }
                        indices.push(i);
                    } else {
                        rest.push(i);
                    }
                }
                self.ready = rest;
                if indices.is_empty() {
                    break;
                }
                let check_config = match unit_cfg {
                    Some(Some(own)) => check_config(&own),
                    _ => check_config(cfg),
                };
                let started = Instant::now();
                let mut unit = Population::default();
                for &i in &indices {
                    let mut creature = pop.creature(i);
                    perturb(&mut creature);
                    unit.push(creature);
                }
                let population = Arc::new(unit);
                match device
                    .engine
                    .submit_shared(Arc::clone(&population), &check_config)
                {
                    Ok(ticket) => {
                        self.packing_seconds += started.elapsed().as_secs_f64();
                        self.checks_submitted += indices.len() as u64;
                        self.check_units += 1;
                        self.checks_since = Some(Instant::now());
                        device.queued.push_back(QueuedUnit {
                            ticket,
                            indices,
                            trial: Trial::Check,
                            population,
                            config: check_config,
                            retries: 0,
                        });
                    }
                    Err(error) => {
                        // The checks stay ready for another engine.
                        self.ready.splice(0..0, indices);
                        device.failure =
                            Some(format!("{} failed: {error:#}", device.engine.name()));
                        break;
                    }
                }
            }
        }
        if self.ready.is_empty() {
            self.checks_since = None;
        }
        Ok(())
    }

    /// Contenders held for their check (waiting, ready or running).
    pub fn holding(&self) -> usize {
        self.held.len()
    }

    /// Whether reserve engines stand by: another engine is healthy.
    fn reserves_standing_by(&self) -> bool {
        self.devices
            .iter()
            .any(|d| !d.reserve && d.failure.is_none())
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
        self.draining = false;
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
                retry: Vec::new(),
            });
        }
    }

    /// Appends `indices` to the active round (or starts one). Used while the
    /// next generation is still being bred.
    pub fn extend(&mut self, pop: &Population, indices: impl IntoIterator<Item = usize>) {
        self.draining = false;
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
                    retry: Vec::new(),
                })
            }
            None => {}
        }
    }

    /// Like `extend`, but the creatures run (and are checked) with `cfg`,
    /// whatever settings later calls pass. Units never mix settings.
    pub(crate) fn extend_with(&mut self, pop: &Population, indices: Vec<usize>, cfg: Arc<Config>) {
        for &i in &indices {
            self.cfg_of.insert(i, Arc::clone(&cfg));
        }
        self.extend(pop, indices);
    }

    /// Holds creature `i`'s standard result for a check trial, which the next
    /// `pump_checks` sends (with `CheckNeed::Check { cell: None }`).
    pub(crate) fn submit_check(&mut self, i: usize, metric: EvaluationMetrics) {
        self.held.insert(i, metric);
        self.checks.push(i);
        self.checks_since.get_or_insert_with(Instant::now);
    }

    /// Stops handing out new work; queued work still completes.
    pub fn stop(&mut self) {
        self.round = None;
        self.draining = true;
    }

    /// Queues work on every engine with a free slot. Creatures marked in
    /// `done` are skipped.
    pub fn pump(
        &mut self,
        pop: &Population,
        cfg: &Config,
        done: &[bool],
        need: impl FnMut(usize, &EvaluationMetrics) -> CheckNeed,
    ) -> Result<()> {
        self.pump_checks(pop, cfg, need)?;
        let result = self.pump_standard(pop, cfg, done);
        let now = Instant::now();
        for device in &mut self.devices {
            device.sample_idle(now);
        }
        result
    }

    /// Queues standard work only; waiting contenders keep waiting. For
    /// callers that cannot decide contenders at the moment (while breeding).
    pub fn pump_standard(&mut self, pop: &Population, cfg: &Config, done: &[bool]) -> Result<()> {
        if !self.may_submit() {
            return Ok(());
        }
        let standby = self.reserves_standing_by();
        let Some(round) = self.round.as_mut() else {
            return Ok(());
        };
        let cfg_of = &self.cfg_of;
        let working = |d: &Device| !(standby && d.reserve);
        let total_rate: f64 = self
            .devices
            .iter()
            .filter(|d| working(d))
            .map(|d| d.rate)
            .sum();
        let queued: usize = self.devices.iter().map(Device::queued_creatures).sum();
        for device in &mut self.devices {
            if standby && device.reserve {
                continue;
            }
            while device.failure.is_none() && device.engine.free_slots() > 0 {
                let remaining =
                    round.order.len() - round.cursor + round.oversize.len() + round.retry.len();
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
                // A unit holds creatures of one trial setting only.
                let mut unit_cfg: Option<Option<Arc<Config>>> = None;
                let fits = |i: usize, unit_cfg: &mut Option<Option<Arc<Config>>>| {
                    let c = cfg_of.get(&i).cloned();
                    match unit_cfg {
                        None => {
                            *unit_cfg = Some(c);
                            true
                        }
                        Some(u) => same_cfg(u, &c),
                    }
                };
                let mut kept = Vec::new();
                let mut moved = 0;
                for i in std::mem::take(&mut round.retry) {
                    if moved < size && fits(i, &mut unit_cfg) {
                        indices.push(i);
                        moved += 1;
                    } else {
                        kept.push(i);
                    }
                }
                round.retry = kept;
                if capacity >= 64 {
                    let mut kept = Vec::new();
                    for i in std::mem::take(&mut round.oversize) {
                        if indices.len() < size && fits(i, &mut unit_cfg) {
                            indices.push(i);
                        } else {
                            kept.push(i);
                        }
                    }
                    round.oversize = kept;
                }
                while indices.len() < size && round.cursor < round.order.len() {
                    let i = round.order[round.cursor];
                    if done.get(i).copied().unwrap_or(false) {
                        round.cursor += 1;
                        continue;
                    }
                    if pop.genomes[i].node_count > capacity {
                        round.cursor += 1;
                        round.oversize.push(i);
                    } else if fits(i, &mut unit_cfg) {
                        round.cursor += 1;
                        indices.push(i);
                    } else {
                        break;
                    }
                }
                if indices.is_empty() {
                    break;
                }
                let unit_config = match unit_cfg {
                    Some(Some(own)) => (*own).clone(),
                    _ => cfg.clone(),
                };
                let started = Instant::now();
                let population = Arc::new(pop.subset(&indices));
                match device
                    .engine
                    .submit_shared(Arc::clone(&population), &unit_config)
                {
                    Ok(ticket) => {
                        self.packing_seconds += started.elapsed().as_secs_f64();
                        device.queued.push_back(QueuedUnit {
                            ticket,
                            indices,
                            trial: Trial::Standard,
                            population,
                            config: unit_config,
                            retries: 0,
                        });
                    }
                    Err(error) => {
                        // Keep the creatures: the next engine offers them again.
                        round.retry.extend(indices);
                        device.failure =
                            Some(format!("{} failed: {error:#}", device.engine.name()));
                        break;
                    }
                }
            }
        }
        if round.cursor == round.order.len() && round.oversize.is_empty() && round.retry.is_empty()
        {
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
        pop: &Population,
        cfg: &Config,
        timeout: Duration,
        contender: impl FnMut(usize, &EvaluationMetrics) -> bool,
    ) -> Result<Vec<(Vec<usize>, Vec<EvaluationMetrics>)>> {
        self.collect_up_to(pop, cfg, timeout, usize::MAX, contender)
    }
    fn collect_up_to(
        &mut self,
        _pop: &Population,
        cfg: &Config,
        timeout: Duration,
        limit: usize,
        mut contender: impl FnMut(usize, &EvaluationMetrics) -> bool,
    ) -> Result<Vec<(Vec<usize>, Vec<EvaluationMetrics>)>> {
        let mut out = Vec::new();
        if !self.released.is_empty() {
            let (indices, metrics) = std::mem::take(&mut self.released).into_iter().unzip();
            out.push((indices, metrics));
        }
        let deadline = Instant::now() + timeout;
        loop {
            'devices: for device in &mut self.devices {
                if device.failure.is_some() {
                    continue;
                }
                loop {
                    if out.len() >= limit {
                        break 'devices;
                    }
                    if device.kind == DeviceKind::Gpu
                        && self.simulate_loss_after == Some(self.collected_units)
                    {
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
                            } = device
                                .queued
                                .remove(position)
                                .expect("validated queued unit");
                            // A world change can leave units from the old
                            // physics in flight. Their results are not
                            // comparable with the live world, so they enter no
                            // archive and their slots are re-bred from the new
                            // world's archive.
                            let stale = config.physics_differs(cfg);
                            device.busy_seconds += done.busy_seconds;
                            if trial == Trial::Check {
                                self.check_busy_seconds += done.busy_seconds;
                            }
                            let mut finals = Vec::with_capacity(indices.len());
                            let mut metrics = Vec::with_capacity(indices.len());
                            match trial {
                                Trial::Standard => {
                                    device.creatures += indices.len() as u64;
                                    device.update_rate(Instant::now(), true, indices.len());
                                    for (k, &i) in indices.iter().enumerate() {
                                        let mut metric =
                                            to_metrics(&population, k, &done.results[k], &config);
                                        if stale {
                                            metric.unchecked = true;
                                        }
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
                                    // The rate counts standard-trial equivalents,
                                    // so a device busy with checks still shows
                                    // its capacity.
                                    device.update_rate(
                                        Instant::now(),
                                        true,
                                        (indices.len() as f64 * CHECK_COST) as usize,
                                    );
                                    for (k, &i) in indices.iter().enumerate() {
                                        if let Some(cell) = self.checking.remove(&i) {
                                            // The cell is free: its waiters are
                                            // decided again on the next pump.
                                            self.busy_cells.remove(&cell);
                                        }
                                        // A creature re-queued meanwhile may have been settled already.
                                        let Some(metric) = self.held.get_mut(&i) else {
                                            continue;
                                        };
                                        // Reliable motion only: keep the worse of all trials.
                                        let (fitness, unchecked) =
                                            check_verdict(metric, &done.results[k]);
                                        metric.fitness = fitness;
                                        // A check stopped by the screen:
                                        // the creature enters no archive.
                                        metric.unchecked |= unchecked;
                                        if stale {
                                            // The check ran under the old physics:
                                            // the whole evaluation is stale.
                                            metric.unchecked = true;
                                        }
                                        let metric = self.held.remove(&i).expect("held contender");
                                        finals.push(i);
                                        metrics.push(metric);
                                    }
                                }
                            }
                            if !finals.is_empty() {
                                out.push((finals, metrics));
                            }
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
            if !out.is_empty() || self.in_flight() == 0 || Instant::now() >= deadline {
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

    /// Retires every engine that reported a failure since the last pass. A
    /// failed GPU hands its unfinished units to a healthy CPU engine, which
    /// preserves their inputs and the order of their tickets. A failed CPU,
    /// or a GPU with no CPU to fall back on, is terminal.
    fn retire_failed(&mut self) -> Result<()> {
        let mut index = 0;
        while index < self.devices.len() {
            if self.devices[index].failure.is_some() {
                if self.retire(index) {
                    continue;
                }
                let error = self.devices[index]
                    .failure
                    .clone()
                    .unwrap_or_else(|| "Evaluation device failed".into());
                anyhow::bail!("{error}");
            }
            index += 1;
        }
        Ok(())
    }

    /// Removes one failed device, re-submitting a GPU's queued units to the
    /// CPU. Returns false when the failure is terminal.
    fn retire(&mut self, index: usize) -> bool {
        let reason = self.devices[index]
            .failure
            .clone()
            .unwrap_or_else(|| "evaluation failed".into());
        if self.devices[index].kind == DeviceKind::Cpu {
            return false;
        }
        if self.devices[index].reopen.is_some() && self.recover(index, &reason) {
            return true;
        }
        let Some(cpu) = self
            .devices
            .iter()
            .position(|d| d.kind == DeviceKind::Cpu && d.failure.is_none())
        else {
            self.devices[index].failure =
                Some(format!("{reason}; no CPU engine can retry its work"));
            return false;
        };
        let units: Vec<QueuedUnit> = self.devices[index].queued.drain(..).collect();
        let count = units.len();
        for mut unit in units {
            unit.retries = unit.retries.saturating_add(1);
            match self.devices[cpu]
                .engine
                .submit_shared(Arc::clone(&unit.population), &unit.config)
            {
                Ok(ticket) => {
                    self.devices[cpu]
                        .queued
                        .push_back(QueuedUnit { ticket, ..unit });
                }
                Err(error) => {
                    let message = format!("{reason}; CPU retry failed: {error:#}");
                    self.devices[cpu].failure = Some(message.clone());
                    self.devices[index].failure = Some(message);
                    return false;
                }
            }
        }
        self.devices.remove(index);
        eprintln!("{reason}; retried {count} units on the CPU");
        true
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
    /// the units stay queued for the CPU.
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
            .push("The GPU could not be reopened. Continuing on the CPU.".into());
        self.devices[index].queued = units.into_iter().collect();
        false
    }

    /// Evaluates `indices` on the active engine path and returns metrics in
    /// the same order. Healthy GPU devices own all scores in a GPU run.
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
            self.pump_checks(pop, cfg, |_, _| CheckNeed::Check { cell: None })?;
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
            self.pump(pop, cfg, &[], |_, _| CheckNeed::Check { cell: None })?;
            for (unit, metrics) in self.collect(pop, cfg, Duration::from_millis(50), |_, m| {
                check && !m.screened
            })? {
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
pub fn perturb(creature: &mut crate::evolution::Creature) {
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
    // Behavior totals end at a fall or the screen, so they average over the
    // steps walked.
    let ended = if r.fall_time > 0.0 {
        r.fall_time
    } else {
        r.screened
    };
    let steps = if ended > 0.0 {
        ((ended * cfg.fidelity().rate as f32).round() as u32).clamp(1, cfg.steps().max(1))
    } else {
        cfg.steps().max(1)
    };
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
        unchecked: false,
        screened: r.screened > 0.0,
        screen_x: r.screen_x,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unshared(_: usize, _: &EvaluationMetrics) -> CheckNeed {
        CheckNeed::Check { cell: None }
    }
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
        /// Returned from the next poll.
        poll_failure: Option<String>,
        /// Returned from the next submission.
        submit_failure: Option<String>,
    }

    struct FakeEngine {
        name: &'static str,
        state: Arc<Mutex<FakeState>>,
    }

    impl Engine for FakeEngine {
        fn name(&self) -> String {
            self.name.into()
        }

        fn max_nodes(&self) -> usize {
            64
        }

        fn free_slots(&self) -> usize {
            let state = self.state.lock().unwrap();
            // A pending submission failure still lets the scheduler try once.
            usize::from(!state.pending && state.poll_failure.is_none())
        }

        fn submit_shared(&mut self, population: Arc<Population>, config: &Config) -> Result<u64> {
            let mut state = self.state.lock().unwrap();
            if let Some(error) = state.submit_failure.take() {
                anyhow::bail!("{error}");
            }
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
            let mut state = self.state.lock().unwrap();
            if let Some(error) = state.poll_failure.take() {
                return Err(anyhow::anyhow!("{error}"));
            }
            let done = state.results.pop_front();
            if done.is_some() {
                state.pending = false;
            }
            Ok(done)
        }

        fn wait(&mut self, _timeout: Duration) {}
    }

    fn fake_device(name: &'static str, kind: DeviceKind, state: &Arc<Mutex<FakeState>>) -> Device {
        Device::new(
            Box::new(FakeEngine {
                name,
                state: Arc::clone(state),
            }),
            kind,
            100.0,
            2,
            1.0,
        )
    }

    fn fake_scheduler() -> (Scheduler, Arc<Mutex<FakeState>>) {
        let state = Arc::new(Mutex::new(FakeState::default()));
        let scheduler = Scheduler {
            devices: vec![fake_device("fake evaluator", DeviceKind::Cpu, &state)],
            round: None,
            packing_seconds: 0.0,
            robust_trials: 1,
            checks: Vec::new(),
            checks_since: None,
            held: HashMap::new(),
            checking: HashMap::new(),
            busy_cells: Default::default(),
            ready: Vec::new(),
            draining: false,
            released: Vec::new(),
            checks_submitted: 0,
            check_busy_seconds: 0.0,
            checks_released: 0,
            check_units: 0,
            checks_dropped: 0,
            startup_failure: None,
            notices: Vec::new(),
            simulate_loss_after: None,
            collected_units: 0,
            cfg_of: HashMap::new(),
            ordered: Default::default(),
            suspension: Default::default(),
        };
        (scheduler, state)
    }

    /// One fake GPU and one fake CPU, for recovery tests that need both.
    fn mixed_scheduler() -> (Scheduler, Arc<Mutex<FakeState>>, Arc<Mutex<FakeState>>) {
        let gpu = Arc::new(Mutex::new(FakeState::default()));
        let cpu = Arc::new(Mutex::new(FakeState::default()));
        let scheduler = Scheduler {
            devices: vec![
                fake_device("fake gpu", DeviceKind::Gpu, &gpu),
                fake_device("fake cpu", DeviceKind::Cpu, &cpu),
            ],
            round: None,
            packing_seconds: 0.0,
            robust_trials: 1,
            checks: Vec::new(),
            checks_since: None,
            held: HashMap::new(),
            checking: HashMap::new(),
            busy_cells: Default::default(),
            ready: Vec::new(),
            draining: false,
            released: Vec::new(),
            checks_submitted: 0,
            check_busy_seconds: 0.0,
            checks_released: 0,
            check_units: 0,
            checks_dropped: 0,
            startup_failure: None,
            notices: Vec::new(),
            simulate_loss_after: None,
            collected_units: 0,
            cfg_of: HashMap::new(),
            ordered: Default::default(),
            suspension: Default::default(),
        };
        (scheduler, gpu, cpu)
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
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
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
        scheduler.pump_checks(&pop, &cfg, unshared).unwrap();
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
                    scheduler.pump_checks(&pop, &cfg, unshared).unwrap();
                } else {
                    scheduler.begin(&pop, 0..2);
                    scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
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

    /// Queues one valid result per submission from `from` onward, in order.
    fn complete_submissions(state: &Arc<Mutex<FakeState>>, from: usize, fitness: f32) {
        let mut state = state.lock().unwrap();
        let submissions: Vec<(u64, usize)> = state
            .submissions
            .iter()
            .skip(from)
            .map(|s| (s.ticket, s.population.genomes.len()))
            .collect();
        for (ticket, count) in submissions {
            state.results.push_back(Finished {
                ticket,
                results: vec![
                    GpuResult {
                        fitness,
                        ..GpuResult::default()
                    };
                    count
                ],
                busy_seconds: 0.5,
            });
        }
    }

    /// Collects until the scheduler is idle; returns every index once, sorted.
    fn drain_all(scheduler: &mut Scheduler, pop: &Population, cfg: &Config) -> Vec<usize> {
        let mut seen = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while scheduler.in_flight() > 0 {
            assert!(Instant::now() < deadline, "evaluation stalled");
            for (indices, _) in scheduler
                .collect(pop, cfg, Duration::from_millis(5), |_, _| false)
                .unwrap()
            {
                seen.extend(indices);
            }
        }
        seen.sort_unstable();
        seen
    }

    #[test]
    fn a_lost_gpu_is_reopened_and_gets_its_units_again() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, gpu, cpu) = mixed_scheduler();
        let reopened = Arc::new(Mutex::new(FakeState::default()));
        let engine_state = Arc::clone(&reopened);
        let mut attempts = 0;
        scheduler.devices[0].reopen = Some(Reopen {
            open: Box::new(move || {
                attempts += 1;
                // The first attempt finds no device, the second one opens.
                if attempts == 1 {
                    anyhow::bail!("device not ready");
                }
                Ok(Box::new(FakeEngine {
                    name: "reopened gpu",
                    state: Arc::clone(&engine_state),
                }) as Box<dyn Engine>)
            }),
            backoff: vec![Duration::ZERO; 3],
        });
        scheduler.begin(&pop, 0..cfg.population);
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
        let (gpu_units, gpu_populations, cpu_units) = {
            let gpu = gpu.lock().unwrap();
            assert!(!gpu.submissions.is_empty(), "the GPU took no work");
            (
                gpu.submissions.len(),
                gpu.submissions
                    .iter()
                    .map(|s| Arc::clone(&s.population))
                    .collect::<Vec<_>>(),
                cpu.lock().unwrap().submissions.len(),
            )
        };
        gpu.lock().unwrap().poll_failure = Some("device lost".into());
        let first = scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap();
        assert!(first.is_empty());
        assert!(
            scheduler.devices.iter().any(|d| d.kind == DeviceKind::Gpu),
            "the GPU must stay"
        );
        assert_eq!(scheduler.devices[0].engine.name(), "reopened gpu");
        {
            let state = reopened.lock().unwrap();
            assert_eq!(state.submissions.len(), gpu_units);
            for (again, original) in state.submissions.iter().zip(&gpu_populations) {
                assert!(
                    Arc::ptr_eq(&again.population, original),
                    "the resubmitted unit must keep the exact creatures"
                );
                assert_eq!(again.config, cfg);
            }
        }
        assert_eq!(
            cpu.lock().unwrap().submissions.len(),
            cpu_units,
            "no work moves to the CPU"
        );
        let notices = scheduler.take_notices();
        assert!(
            notices.iter().any(|n| n.contains("GPU is back")),
            "{notices:?}"
        );
        assert!(scheduler.devices[0].queued.iter().all(|u| u.retries == 1));
    }

    #[test]
    fn a_gpu_that_never_reopens_falls_back_to_the_cpu() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, gpu, cpu) = mixed_scheduler();
        scheduler.devices[0].reopen = Some(Reopen {
            open: Box::new(|| anyhow::bail!("no device")),
            backoff: vec![Duration::ZERO; 3],
        });
        scheduler.begin(&pop, 0..cfg.population);
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
        let (gpu_units, cpu_units) = (
            gpu.lock().unwrap().submissions.len(),
            cpu.lock().unwrap().submissions.len(),
        );
        gpu.lock().unwrap().poll_failure = Some("device lost".into());
        scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap();
        assert!(scheduler.devices.iter().all(|d| d.kind != DeviceKind::Gpu));
        assert_eq!(cpu.lock().unwrap().submissions.len(), cpu_units + gpu_units);
        let notices = scheduler.take_notices();
        assert!(notices.iter().any(|n| n.contains("Continuing on the CPU")));
    }

    #[test]
    fn a_failed_gpu_retries_its_unfinished_units_on_the_cpu() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, gpu, cpu) = mixed_scheduler();
        scheduler.begin(&pop, 0..cfg.population);
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
        let (gpu_units, cpu_units, gpu_populations) = {
            let gpu = gpu.lock().unwrap();
            let cpu = cpu.lock().unwrap();
            assert!(!gpu.submissions.is_empty(), "the GPU took no work");
            assert!(!cpu.submissions.is_empty(), "the CPU took no work");
            (
                gpu.submissions.len(),
                cpu.submissions.len(),
                gpu.submissions
                    .iter()
                    .map(|s| Arc::clone(&s.population))
                    .collect::<Vec<_>>(),
            )
        };
        // The GPU dies with every one of its units still unfinished.
        gpu.lock().unwrap().poll_failure = Some("device lost".into());
        let first = scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap();
        assert!(first.is_empty(), "the GPU returned a result");
        assert!(
            scheduler.devices.iter().all(|d| d.kind != DeviceKind::Gpu),
            "the failed GPU must be retired"
        );
        {
            let cpu = cpu.lock().unwrap();
            assert_eq!(cpu.submissions.len(), cpu_units + gpu_units);
            for (retried, original) in cpu.submissions[cpu_units..].iter().zip(&gpu_populations) {
                assert!(
                    Arc::ptr_eq(&retried.population, original),
                    "the retried unit must keep the exact submitted creatures"
                );
                assert_eq!(retried.config, cfg);
            }
        }
        assert!(
            scheduler.devices[0]
                .queued
                .iter()
                .skip(cpu_units)
                .all(|unit| unit.retries == 1),
            "moved units must carry their retry state"
        );
        complete_submissions(&cpu, 0, 3.0);
        let seen = drain_all(&mut scheduler, &pop, &cfg);
        assert_eq!(
            seen,
            (0..cfg.population).collect::<Vec<_>>(),
            "every creature must finish exactly once"
        );
    }

    #[test]
    fn a_reserve_cpu_stands_by_while_the_gpu_works_and_takes_over_after() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, gpu, cpu) = mixed_scheduler();
        reserve_cpu_when_gpu_available(&mut scheduler.devices);
        assert!(scheduler.devices[1].reserve);
        scheduler.begin(&pop, 0..cfg.population);
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
        let gpu_units = gpu.lock().unwrap().submissions.len();
        assert!(gpu_units > 0, "the GPU took no work");
        assert!(
            cpu.lock().unwrap().submissions.is_empty(),
            "a reserve CPU must not take standard work beside a healthy GPU"
        );
        gpu.lock().unwrap().poll_failure = Some("device lost".into());
        let first = scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap();
        assert!(first.is_empty(), "the GPU returned a result");
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
        assert!(
            cpu.lock().unwrap().submissions.len() >= gpu_units,
            "the reserve must take over the failed GPU's work"
        );
        complete_submissions(&cpu, 0, 3.0);
        let seen = drain_all(&mut scheduler, &pop, &cfg);
        assert_eq!(seen, (0..cfg.population).collect::<Vec<_>>());
    }

    #[test]
    fn a_failed_gpu_submission_keeps_its_creatures_for_the_cpu() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, gpu, cpu) = mixed_scheduler();
        gpu.lock().unwrap().submit_failure = Some("device lost".into());
        scheduler.begin(&pop, 0..cfg.population);
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
        assert_eq!(gpu.lock().unwrap().submissions.len(), 0);
        assert_eq!(
            cpu.lock().unwrap().submissions.len(),
            1,
            "the CPU must take the rejected unit"
        );
        scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap();
        assert!(scheduler.devices.iter().all(|d| d.kind != DeviceKind::Gpu));
        complete_submissions(&cpu, 0, 2.5);
        let mut seen: Vec<usize> = scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap()
            .into_iter()
            .flat_map(|(indices, _)| indices)
            .collect();
        // The remaining rejected creatures are still in the round.
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
        assert_eq!(cpu.lock().unwrap().submissions.len(), 2);
        complete_submissions(&cpu, 1, 2.5);
        seen.extend(drain_all(&mut scheduler, &pop, &cfg));
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..cfg.population).collect::<Vec<_>>(),
            "every creature must finish exactly once"
        );
    }

    #[test]
    fn a_failed_cpu_is_terminal_and_delivers_completed_output_first() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, gpu, cpu) = mixed_scheduler();
        scheduler.begin(&pop, 0..cfg.population);
        scheduler.pump(&pop, &cfg, &[], unshared).unwrap();
        let (ticket, count) = {
            let gpu = gpu.lock().unwrap();
            let submission = &gpu.submissions[0];
            (submission.ticket, submission.population.genomes.len())
        };
        // One GPU unit completes; the CPU dies before returning anything.
        gpu.lock().unwrap().results.push_back(Finished {
            ticket,
            results: vec![
                GpuResult {
                    fitness: 7.0,
                    ..GpuResult::default()
                };
                count
            ],
            busy_seconds: 0.5,
        });
        cpu.lock().unwrap().poll_failure = Some("cpu lost".into());
        let completed = scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap();
        assert_eq!(
            completed
                .iter()
                .map(|(indices, _)| indices.len())
                .sum::<usize>(),
            count,
            "completed output must be delivered before the CPU error"
        );
        let submissions = cpu.lock().unwrap().submissions.len();
        for _ in 0..3 {
            let error = scheduler
                .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
                .expect_err("the CPU error must persist");
            assert!(error.to_string().contains("cpu lost"));
        }
        assert_eq!(
            cpu.lock().unwrap().submissions.len(),
            submissions,
            "a failed CPU must not be retried"
        );
        assert!(scheduler.in_flight() > 0, "the failure must stay visible");
    }

    #[test]
    fn a_failed_gpu_retries_pending_checks_with_their_perturbed_inputs() {
        let cfg = submission_config();
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, gpu, cpu) = mixed_scheduler();
        scheduler.robust_trials = 2;
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
        scheduler.pump_checks(&pop, &cfg, unshared).unwrap();
        let fine = Config {
            fidelity: Some(crate::physics::Fidelity::fine()),
            ..cfg.clone()
        };
        {
            let gpu = gpu.lock().unwrap();
            assert_eq!(gpu.submissions.len(), 1);
            let submission = &gpu.submissions[0];
            assert_eq!(submission.config, fine);
            for (slot, &index) in indices.iter().enumerate() {
                let mut expected = pop.creature(index);
                perturb(&mut expected);
                let actual = submission.population.creature(slot);
                assert_eq!(actual.id, expected.id);
                assert_eq!(actual.nodes, expected.nodes);
                assert_eq!(actual.bones, expected.bones);
                assert_eq!(actual.muscles, expected.muscles);
            }
        }
        gpu.lock().unwrap().poll_failure = Some("device lost".into());
        scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap();
        {
            let cpu = cpu.lock().unwrap();
            assert_eq!(cpu.submissions.len(), 1, "the check must move to the CPU");
            let submission = &cpu.submissions[0];
            assert_eq!(submission.config, fine);
            for (slot, &index) in indices.iter().enumerate() {
                let mut expected = pop.creature(index);
                perturb(&mut expected);
                let actual = submission.population.creature(slot);
                assert_eq!(actual.id, expected.id);
                assert_eq!(actual.nodes, expected.nodes);
                assert_eq!(actual.bones, expected.bones);
                assert_eq!(actual.muscles, expected.muscles);
            }
        }
        complete_submissions(&cpu, 0, 4.0);
        let output = scheduler
            .collect(&pop, &cfg, Duration::ZERO, |_, _| false)
            .unwrap();
        let mut finished: Vec<(usize, f32)> = output
            .iter()
            .flat_map(|(indices, metrics)| {
                indices
                    .iter()
                    .copied()
                    .zip(metrics.iter().map(|m| m.fitness))
            })
            .collect();
        finished.sort_by_key(|&(index, _)| index);
        assert_eq!(finished, vec![(0, 4.0), (2, 4.0)]);
        assert_eq!(scheduler.in_flight(), 0);
    }

    #[test]
    fn a_missing_primary_gpu_falls_back_to_the_cpu() {
        // An invalid adapter name cannot open; the CPU keeps the session alive
        // and reports why the GPU was skipped.
        let scheduler = Scheduler::new("definitely not a vulkan adapter").unwrap();
        assert!(scheduler.startup_failure().is_some());
        assert!(
            scheduler.devices.iter().any(|d| d.kind == DeviceKind::Cpu),
            "the fallback must include a CPU engine"
        );
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
            sched.pump(&pop, &cfg, &[], unshared).unwrap();
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

    /// Evaluates a small population where every creature is a contender for
    /// one shared cell; `release_after_landing` releases the waiting ones once
    /// the first check is back. Returns final fitness per creature.
    /// Evaluates a small population in which every creature is a contender
    /// for one shared cell. Returns the final metrics per creature.
    fn shared_cell_run() -> (Scheduler, Vec<EvaluationMetrics>, Population, Config) {
        let cfg = Config {
            population: 6,
            duration: 1.0,
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
            sched
                .pump(&pop, &cfg, &[], |_, _| CheckNeed::Check { cell: Some(7) })
                .unwrap();
            for (indices, metrics) in sched
                .collect(&pop, &cfg, Duration::from_millis(20), |_, _| true)
                .unwrap()
            {
                for (i, m) in indices.into_iter().zip(metrics) {
                    assert!(got[i].is_none(), "creature {i} returned twice");
                    got[i] = Some(m);
                }
            }
        }
        assert_eq!(sched.in_flight(), 0);
        (
            sched,
            got.into_iter().map(Option::unwrap).collect(),
            pop,
            cfg,
        )
    }

    fn standard_and_check(pop: &Population, cfg: &Config) -> (Vec<f32>, Vec<f32>) {
        let standard = crate::cpu_engine::evaluate(pop, cfg);
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
        (
            standard.iter().map(|r| r.fitness).collect(),
            check.iter().map(|r| r.fitness).collect(),
        )
    }

    #[test]
    fn contender_fitness_comes_from_the_selected_engine_and_its_check() {
        let cfg = Config {
            population: 4,
            duration: 1.0,
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
            sched
                .pump(&pop, &cfg, &[], |_, _| CheckNeed::Check { cell: None })
                .unwrap();
            for (indices, metrics) in sched
                .collect(&pop, &cfg, Duration::from_millis(20), |_, _| true)
                .unwrap()
            {
                for (i, m) in indices.into_iter().zip(metrics) {
                    assert!(got[i].is_none(), "creature {i} returned twice");
                    got[i] = Some(m);
                }
            }
        }
        assert_eq!(sched.in_flight(), 0);
        let (standard, check) = standard_and_check(&pop, &cfg);
        for (i, metric) in got.into_iter().enumerate() {
            let metric = metric.unwrap();
            let expected = standard[i].min(check[i]);
            assert!(
                (metric.fitness - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "creature {i}: {} vs {expected}",
                metric.fitness
            );
        }
    }

    #[test]
    fn a_world_change_stops_stale_results_from_entering_the_archive() {
        let calm = Config {
            population: 4,
            duration: 1.0,
            random_seed: false,
            ..Config::default()
        };
        let rough = Config {
            terrain: 3,
            ..calm.clone()
        };
        let pop = crate::evolution::create(&calm).unwrap();
        let mut sched = Scheduler::cpu_only(2).unwrap();
        sched.begin(&pop, 0..calm.population);
        let global = |_: usize, _: &EvaluationMetrics| CheckNeed::Check { cell: None };
        let deadline = Instant::now() + Duration::from_secs(120);
        // Calm-world standards finish, then at least one check is queued.
        while sched.check_units == 0 {
            assert!(Instant::now() < deadline, "evaluation stalled");
            sched.pump(&pop, &calm, &[], global).unwrap();
            if sched.check_units > 0 {
                break;
            }
            sched
                .collect(&pop, &calm, Duration::from_millis(20), |_, _| true)
                .unwrap();
        }
        // The world changes while that check is in flight. Its results must
        // not claim the new world.
        let mut finals = 0;
        while sched.in_flight() > 0 {
            assert!(Instant::now() < deadline, "evaluation stalled");
            for (indices, metrics) in sched
                .collect(&pop, &rough, Duration::from_millis(20), |_, _| false)
                .unwrap()
            {
                for (i, metric) in indices.into_iter().zip(metrics) {
                    assert!(
                        metric.unchecked,
                        "old-world result {i} would enter an archive"
                    );
                    finals += 1;
                }
            }
        }
        assert!(finals > 0, "the old-world results were never finalized");
    }

    #[test]
    fn a_check_in_the_same_world_is_not_an_old_world_result() {
        let cfg = Config {
            terrain: 2,
            ..submission_config()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, state) = fake_scheduler();
        let indices = vec![3, 1];
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
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut metrics = Vec::new();
        let mut completed = 0usize;
        while scheduler.in_flight() > 0 {
            assert!(Instant::now() < deadline, "check stalled");
            scheduler.pump_checks(&pop, &cfg, unshared).unwrap();
            {
                let mut state = state.lock().unwrap();
                while completed < state.submissions.len() {
                    let count = state.submissions[completed].population.genomes.len();
                    state.results.push_back(Finished {
                        ticket: completed as u64 + 1,
                        results: vec![
                            GpuResult {
                                fitness: 0.0,
                                ..GpuResult::default()
                            };
                            count
                        ],
                        busy_seconds: 0.0,
                    });
                    completed += 1;
                }
            }
            let out = scheduler
                .collect_up_to(&pop, &cfg, Duration::ZERO, usize::MAX, |_, _| false)
                .unwrap();
            metrics.extend(out.into_iter().flat_map(|(_, batch)| batch));
        }
        assert_eq!(metrics.len(), 2);
        assert!(
            metrics.iter().all(|metric| !metric.unchecked),
            "a same-world check was treated as an old-world result"
        );
    }

    #[test]
    fn a_stopped_scheduler_checks_every_waiting_contender_at_once() {
        let cfg = Config {
            population: 6,
            duration: 1.0,
            random_seed: false,
            ..Config::default()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let mut sched = Scheduler::cpu_only(2).unwrap();
        let shared = |_: usize, _: &EvaluationMetrics| CheckNeed::Check { cell: Some(7) };
        sched.begin(&pop, 0..cfg.population);
        let deadline = Instant::now() + Duration::from_secs(120);
        // Run until every standard result is held.
        while sched.held.len() < cfg.population {
            assert!(Instant::now() < deadline, "evaluation stalled");
            sched.pump_standard(&pop, &cfg, &[]).unwrap();
            let out = sched
                .collect(&pop, &cfg, Duration::from_millis(20), |_, _| true)
                .unwrap();
            assert!(out.is_empty(), "no check has run yet");
        }
        sched.stop();
        sched.pump(&pop, &cfg, &[], shared).unwrap();
        assert_eq!(
            sched.checks_submitted, cfg.population as u64,
            "draining sends every contender at once"
        );
        let mut done = 0;
        while done < cfg.population {
            assert!(Instant::now() < deadline, "checks stalled");
            for (indices, _) in sched
                .collect(&pop, &cfg, Duration::from_millis(20), |_, _| true)
                .unwrap()
            {
                done += indices.len();
            }
        }
        assert_eq!(sched.in_flight(), 0);
    }

    #[test]
    fn check_verdicts_keep_the_worse_trial() {
        let standard = EvaluationMetrics {
            fitness: 30.0,
            ..EvaluationMetrics::default()
        };
        let check = |fitness: f32, screened: f32| GpuResult {
            fitness,
            screened,
            ..GpuResult::default()
        };
        assert_eq!(check_verdict(&standard, &check(25.0, 0.0)), (25.0, false));
        assert_eq!(check_verdict(&standard, &check(35.0, 0.0)), (30.0, false));
        assert!(check_verdict(&standard, &check(1.0, 5.0)).1);
    }

    #[test]
    fn a_check_that_fails_the_screen_keeps_its_creature_out_of_the_archive() {
        let cfg = Config {
            population: 4,
            duration: 3.0,
            random_seed: false,
            screen: Some(crate::physics::Screen {
                seconds: 1.0,
                bar: f32::INFINITY,
            }),
            ..Config::default()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let mut sched = Scheduler::cpu_only(2).unwrap();
        sched.begin(&pop, 0..cfg.population);
        let mut got = vec![None; cfg.population];
        let deadline = Instant::now() + Duration::from_secs(120);
        while got.iter().any(Option::is_none) {
            assert!(Instant::now() < deadline, "evaluation stalled");
            sched.pump(&pop, &cfg, &[], unshared).unwrap();
            for (indices, metrics) in sched
                .collect(&pop, &cfg, Duration::from_millis(20), |_, _| true)
                .unwrap()
            {
                for (i, m) in indices.into_iter().zip(metrics) {
                    got[i] = Some(m);
                }
            }
        }
        assert_eq!(sched.checks_submitted, cfg.population as u64);
        for (i, metric) in got.into_iter().enumerate() {
            assert!(
                metric.unwrap().unchecked,
                "creature {i}: its check stopped at the screen"
            );
        }
    }

    #[test]
    fn only_the_best_contender_for_a_cell_is_checked() {
        let (sched, got, pop, cfg) = shared_cell_run();
        // One standard unit returns all six together: the best claims the
        // cell's check and the other five are dropped unchecked.
        assert_eq!(sched.checks_submitted, 1);
        assert_eq!(sched.checks_dropped, cfg.population as u64 - 1);
        let (standard, check) = standard_and_check(&pop, &cfg);
        let best = (0..cfg.population)
            .max_by(|&a, &b| standard[a].total_cmp(&standard[b]))
            .unwrap();
        for (i, metric) in got.iter().enumerate() {
            let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * b.abs().max(1.0);
            if i == best {
                assert!(!metric.unchecked);
                assert!(
                    close(metric.fitness, standard[i].min(check[i])),
                    "creature {i}"
                );
            } else {
                assert!(metric.unchecked, "creature {i} must be dropped");
                assert!(close(metric.fitness, standard[i]), "creature {i}");
            }
        }
    }

    #[test]
    fn checks_keep_the_standard_terrain_and_perturb_the_pose() {
        let cfg = Config {
            terrain: 2,
            ..submission_config()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let (mut scheduler, state) = fake_scheduler();
        let indices = vec![3, 1];
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
        scheduler.pump_checks(&pop, &cfg, unshared).unwrap();
        let state = state.lock().unwrap();
        assert_eq!(state.submissions.len(), 1);
        let submission = &state.submissions[0];
        assert_eq!(
            submission.config,
            Config {
                fidelity: Some(crate::physics::Fidelity::fine()),
                ..cfg
            }
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
    }
}
