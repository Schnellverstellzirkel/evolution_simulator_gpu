//! Records the native benchmark (`EVOLUTION_BENCH_GENERATIONS`) while the game
//! runs and prints its report at the end. `Loop` calls the hooks of `Benchmark`
//! as commands, search passes and generations happen. The UI reads the flag and
//! the breeding spans in `Bench` and prints the frame part of the report when
//! it exits.

use crate::storage::{self, Experiment};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

/// What the worker shares with the UI for the native benchmark.
pub(super) struct Bench {
    /// True while the benchmark measures, after the warm-up and before its
    /// last generation. The UI records frames only then.
    pub(super) measuring: Arc<AtomicBool>,
    /// Start and end of every ring step that absorbed and bred a block while
    /// measuring, and whether it ended a generation.
    pub(super) breeding: Arc<Mutex<Vec<(Instant, Instant, bool)>>>,
}

/// What the benchmark records over a run. It measures and reports only when
/// `EVOLUTION_BENCH_GENERATIONS` is set, and it measures only between the
/// warm-up and its last generation. Snapshot build times are kept in every run.
pub(super) struct Benchmark {
    /// The flag and the spans that the UI reads too.
    shared: Bench,
    /// Generations to measure after the warm-up, from
    /// `EVOLUTION_BENCH_GENERATIONS`. `None` when no benchmark was asked for.
    generations: Option<u32>,
    /// Generations to run before measuring starts, from
    /// `EVOLUTION_BENCH_WARMUP`.
    warmup: u32,
    /// Generation at which the benchmark run was started. Measurement begins
    /// after the warm-up.
    run_generation: Option<u32>,
    /// Generation and time when measurement began.
    start: Option<(u32, Instant)>,
    /// Total evaluation, archive and breeding seconds, in that order.
    stage_seconds: [f64; 3],
    /// Seconds each measured generation took.
    generation_seconds: Vec<f64>,
    /// When the generation now running began.
    generation_started: Instant,
    /// Send and read time of every ping.
    pings: Vec<(Instant, Instant)>,
    /// Milliseconds each settings probe waited before it was applied.
    configure_ms: Vec<f64>,
    /// Milliseconds each snapshot took to build.
    snapshot_build_ms: Vec<f64>,
}

impl Benchmark {
    /// A benchmark that is off unless `EVOLUTION_BENCH_GENERATIONS` holds a
    /// number above 0. `EVOLUTION_BENCH_WARMUP` sets the warm-up generations
    /// and is 1 when unset or not a number.
    pub(super) fn new(shared: Bench) -> Self {
        Self {
            shared,
            generations: std::env::var("EVOLUTION_BENCH_GENERATIONS")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .filter(|&value| value > 0),
            warmup: std::env::var("EVOLUTION_BENCH_WARMUP")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(1),
            run_generation: None,
            start: None,
            stage_seconds: [0.0f64; 3],
            generation_seconds: Vec::new(),
            generation_started: Instant::now(),
            pings: Vec::new(),
            configure_ms: Vec::new(),
            snapshot_build_ms: Vec::new(),
        }
    }
    /// A run was started (`Command::Run`) at `generation`, the game's
    /// generation if there is a game. The first run of a benchmark is the one
    /// the warm-up counts from. With no warm-up, measuring starts now.
    pub(super) fn run_started(&mut self, generation: Option<u32>) {
        if self.generations.is_some() && self.run_generation.is_none() {
            self.run_generation = generation;
            if self.warmup == 0 {
                self.start = generation.map(|generation| (generation, Instant::now()));
                self.generation_started = Instant::now();
                self.shared.measuring.store(true, Ordering::Relaxed);
            }
        }
    }
    /// A ping sent at `sent` was read now. It is kept while measuring. Pings
    /// read during a search pass come in with `pass_done` instead.
    pub(super) fn ping(&mut self, sent: Instant) {
        if self.shared.measuring.load(Ordering::Relaxed) {
            self.pings.push((sent, Instant::now()));
        }
    }
    /// A settings probe sent at `sent` has been applied. Its wait is kept
    /// while measuring.
    pub(super) fn configure_probe(&mut self, sent: Instant) {
        if self.shared.measuring.load(Ordering::Relaxed) {
            self.configure_ms.push(sent.elapsed().as_secs_f64() * 1e3);
        }
    }
    /// A search pass is over: the pings it read and its breeding steps. While
    /// measuring, this keeps the pings and adds the breeding steps to the list
    /// the UI reads.
    pub(super) fn pass_done(
        &mut self,
        pings: &[(Instant, Instant)],
        breeding: &[(Instant, Instant, bool)],
    ) {
        if self.shared.measuring.load(Ordering::Relaxed) {
            self.pings.extend(pings);
            self.shared.breeding.lock().unwrap().extend(breeding);
        }
    }
    /// The seconds a pass spent on evaluation, archive and breeding. They
    /// count once measuring has started.
    pub(super) fn add_stage_seconds(&mut self, seconds: [f64; 3]) {
        if self.start.is_some() {
            for (total, seconds) in self.stage_seconds.iter_mut().zip(seconds) {
                *total += seconds;
            }
        }
    }
    /// A snapshot was built in `ms` milliseconds. This is kept in every run,
    /// measured or not.
    pub(super) fn snapshot_built(&mut self, ms: f64) {
        self.snapshot_build_ms.push(ms);
    }
    /// A generation of `e` ended. Starts measuring once the warm-up
    /// generations are done. After the last generation it prints the report,
    /// saves the game if `EVOLUTION_BENCH_SAVE` is set and closes the window.
    /// True once the benchmark is over.
    pub(super) fn generation_done(
        &mut self,
        e: &Experiment,
        sched: &crate::scheduler::Scheduler,
        ctx: &eframe::egui::Context,
    ) -> bool {
        if self.start.is_some() {
            self.generation_seconds
                .push(self.generation_started.elapsed().as_secs_f64());
        }
        self.generation_started = Instant::now();
        if self.start.is_none()
            && let Some(first) = self.run_generation
            && e.generation.saturating_sub(first) >= self.warmup
        {
            self.start = Some((e.generation, Instant::now()));
            self.shared.measuring.store(true, Ordering::Relaxed);
        }
        if let (Some(target), Some((first, started))) = (self.generations, self.start)
            && e.generation.saturating_sub(first) >= target
        {
            self.shared.measuring.store(false, Ordering::Relaxed);
            report_benchmark(
                e,
                sched,
                e.generation - first,
                started.elapsed().as_secs_f64(),
                self.warmup,
                self.stage_seconds,
                &self.generation_seconds,
                &self.snapshot_build_ms,
                &self.configure_ms,
                &self.pings,
                &self.shared.breeding.lock().unwrap(),
            );
            // `EVOLUTION_BENCH_SAVE` keeps the evolved game for later runs.
            if let Some(path) = std::env::var_os("EVOLUTION_BENCH_SAVE") {
                let path = PathBuf::from(path);
                match storage::save(&path, e) {
                    Ok(()) => eprintln!("Benchmark saved {}", path.display()),
                    Err(err) => {
                        eprintln!("Benchmark save {} failed: {err:#}", path.display())
                    }
                }
            }
            ctx.send_viewport_cmd(eframe::egui::ViewportCommand::Close);
            ctx.request_repaint();
            return true;
        }
        false
    }
}
/// Prints the benchmark report on stderr. It gives the run's rate, the stage
/// seconds, the generation and snapshot build times, the settings and control
/// latency, the worker thread's major faults, and the totals of each device
/// and of packing. `generations` and `seconds` cover the measured window.
#[allow(clippy::too_many_arguments)]
fn report_benchmark(
    e: &Experiment,
    sched: &crate::scheduler::Scheduler,
    generations: u32,
    seconds: f64,
    warmup: u32,
    stage_seconds: [f64; 3],
    generation_seconds: &[f64],
    snapshot_build_ms: &[f64],
    configure_ms: &[f64],
    pings: &[(Instant, Instant)],
    breeding: &[(Instant, Instant, bool)],
) {
    let creatures = f64::from(generations) * e.config.population as f64;
    eprintln!(
        "Native generation benchmark: {} generations in {:.6} s ({:.3} generations/s), population {}, duration {} s, throughput {}, warm-up {} generations",
        generations,
        seconds,
        f64::from(generations) / seconds,
        e.config.population,
        e.config.duration,
        e.config.throughput,
        warmup
    );
    eprintln!(
        "Native benchmark stages: evaluation {:.6} s, archive {:.6} s, breeding {:.6} s",
        stage_seconds[0], stage_seconds[1], stage_seconds[2]
    );
    let sorted = |values: &[f64]| {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        values
    };
    let per_generation = sorted(generation_seconds);
    eprintln!(
        "Native benchmark throughput: end-to-end {:.0} creatures/s; generation seconds min {:.3} median {:.3} max {:.3}",
        creatures / seconds,
        per_generation.first().copied().unwrap_or(0.0),
        per_generation
            .get(per_generation.len() / 2)
            .copied()
            .unwrap_or(0.0),
        per_generation.last().copied().unwrap_or(0.0)
    );
    let builds = sorted(snapshot_build_ms);
    if !builds.is_empty() {
        eprintln!(
            "Native benchmark snapshot build: {} snapshots, median {:.3} ms, p95 {:.3} ms, max {:.3} ms",
            builds.len(),
            builds[builds.len() / 2],
            builds[builds.len() * 95 / 100],
            builds.last().copied().unwrap_or(0.0)
        );
    }
    let configures = sorted(configure_ms);
    if !configures.is_empty() {
        eprintln!(
            "Native benchmark settings latency: {} probes, median {:.1} ms, p99 {:.1} ms, max {:.1} ms",
            configures.len(),
            configures[configures.len() / 2],
            configures[(configures.len() * 99 / 100).min(configures.len() - 1)],
            configures.last().copied().unwrap_or(0.0)
        );
    }
    // Control latency over all probes, over the probes that waited while a
    // block was absorbed and bred, and over those that waited while a
    // generation ended.
    let waited = |boundary: bool| -> Vec<f64> {
        pings
            .iter()
            .filter(|&&(sent, read)| {
                breeding
                    .iter()
                    .any(|&(a, b, ended)| (ended || !boundary) && sent < b && read > a)
            })
            .map(|(sent, read)| (*read - *sent).as_secs_f64() * 1e3)
            .collect()
    };
    let all: Vec<f64> = pings
        .iter()
        .map(|(sent, read)| (*read - *sent).as_secs_f64() * 1e3)
        .collect();
    for (pings, when) in [all, waited(false), waited(true)].iter().zip([
        "",
        " during breeding",
        " across boundaries",
    ]) {
        let pings = sorted(pings);
        let pct = |q: usize| {
            pings
                .get((pings.len() * q / 100).min(pings.len().saturating_sub(1)))
                .copied()
                .unwrap_or(0.0)
        };
        eprintln!(
            "Native benchmark control latency{when}: {} probes, p50 {:.1} ms, p95 {:.1} ms, p99 {:.1} ms, max {:.1} ms",
            pings.len(),
            pct(50),
            pct(95),
            pct(99),
            pings.last().copied().unwrap_or(0.0)
        );
    }
    if let Some(faults) = crate::threads::major_faults() {
        eprintln!("Native benchmark worker thread: {faults} major faults since start");
    }
    for device in &sched.devices {
        eprintln!(
            "Native benchmark device {}: {} creatures, busy {:.3} s, idle {:.3} s, rate {:.0}/s (totals since start)",
            device.engine.name(),
            device.creatures,
            device.busy_seconds,
            device.idle_seconds,
            device.rate
        );
    }
    eprintln!(
        "Native benchmark packing {:.3} s, {} confirmation trials (busy {:.3} s) (totals since start)",
        sched.packing_seconds, sched.confirms_submitted, sched.confirm_busy_seconds,
    );
}
