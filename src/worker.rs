use crate::{
    config::Config,
    evolution::Creature,
    gpu::Gpu,
    qd::{self, Descriptor, Emitter, EmitterStats},
    storage::{self, Experiment, Stage, Stats},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};
pub enum Command {
    New(Config),
    Run {
        continuous: bool,
        guided: bool,
    },
    Next,
    Pause,
    Configure(Config),
    Save(PathBuf),
    Load(PathBuf),
    Export(PathBuf),
    Page(usize),
    Preview(usize),
    /// Benchmark probe: the UI send time, used to measure how long queued controls wait.
    Ping(Instant),
    Shutdown,
}
#[derive(Clone)]
pub struct Card {
    pub index: usize,
    pub rank: usize,
    pub score: f32,
    pub parent_score: f32,
    pub survivor: bool,
    pub descriptor: Option<Descriptor>,
    pub emitter: Option<Emitter>,
    pub visits: u64,
    pub innovation_reserve: bool,
    pub creature: Creature,
}
/// One ancestor of a selected creature.
#[derive(Clone)]
pub struct LineageStep {
    pub generation: u32,
    pub fitness: f32,
    /// Fitness gained over this ancestor's own parent.
    pub gain: f32,
    pub change: String,
    pub creature: Creature,
}
#[derive(Clone)]
pub struct Snapshot {
    pub epoch: u64,
    pub config: Config,
    pub generation: u32,
    pub evaluated: usize,
    /// Creatures of the current generation with stored results. Engines finish
    /// out of order, so this runs ahead of the contiguous `evaluated` prefix.
    pub completed: usize,
    pub stage: Stage,
    pub running: bool,
    pub history: Arc<Vec<Stats>>,
    pub page: Vec<Card>,
    pub page_start: usize,
    pub preview: Option<(Creature, Config)>,
    /// Ancestor chain of the selected elite, newest first; sent once per selection.
    pub lineage: Option<Vec<LineageStep>>,
    pub gpu: String,
    /// Evaluation engines: name, measured creatures/s, creatures evaluated.
    pub engines: Vec<(String, f64, u64)>,
    /// Creatures per second over complete generations in the last ~10 s,
    /// including archive updates, breeding, and transfers.
    pub end_to_end: f64,
    pub gpu_bytes: u64,
    pub ram_bytes: usize,
    pub elapsed: f64,
    pub archive_cells: usize,
    pub archive_size: usize,
    pub innovation_reserve_count: usize,
    pub qd_score: f64,
    pub emitters: [EmitterStats; 4],
    pub emitter_weights: [f64; 4],
    pub status: String,
    pub error: Option<String>,
}
pub struct Worker {
    pub tx: Sender<Command>,
    pub view: Arc<Mutex<Option<Snapshot>>>,
    pub pause: Arc<AtomicBool>,
    /// True while a native benchmark is inside its measured window (after warm-up).
    pub measuring: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}
impl Worker {
    pub fn spawn(gpu: Gpu, ctx: eframe::egui::Context) -> Self {
        let (tx, rx) = mpsc::channel();
        let view = Arc::new(Mutex::new(None));
        let output = view.clone();
        let pause = Arc::new(AtomicBool::new(false));
        let paused = pause.clone();
        let measuring = Arc::new(AtomicBool::new(false));
        let bench_measuring = measuring.clone();
        let join = std::thread::Builder::new()
            .name("evolution".into())
            .spawn(move || run(gpu, rx, output, paused, bench_measuring, ctx))
            .expect("Start simulation worker");
        Self {
            tx,
            view,
            pause,
            measuring,
            join: Some(join),
        }
    }
    pub fn send(&self, c: Command) {
        let _ = self.tx.send(c);
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.pause.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Command::Shutdown);
        // Finish compute before eframe destroys the Vulkan surface/device resources.
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
fn run(
    mut gpu: Gpu,
    rx: Receiver<Command>,
    output: Arc<Mutex<Option<Snapshot>>>,
    pause: Arc<AtomicBool>,
    measuring: Arc<AtomicBool>,
    ctx: eframe::egui::Context,
) {
    let mut exp: Option<Experiment> = None;
    let mut running = false;
    let mut continuous = false;
    let mut guided = false;
    let mut page = 0usize;
    let mut preview = None;
    let mut lineage: Option<Vec<LineageStep>> = None;
    let mut status = "Create a population to begin".to_owned();
    let mut error = None;
    let mut last_publish = Instant::now() - Duration::from_secs(1);
    let mut changed = true;
    let mut epoch = 0u64;
    let mut history = Arc::new(Vec::new());
    let mut checkpoint_thread: Option<std::thread::JoinHandle<()>> = None;
    let benchmark_generations = std::env::var("EVOLUTION_BENCH_GENERATIONS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0);
    let benchmark_warmup = std::env::var("EVOLUTION_BENCH_WARMUP")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(1);
    // Generation at which the benchmark run was started; measurement begins after warm-up.
    let mut benchmark_run_generation: Option<u32> = None;
    let mut benchmark_start: Option<(u32, Instant)> = None;
    let mut benchmark_stage_seconds = [0.0f64; 3];
    let mut benchmark_generation_seconds: Vec<f64> = Vec::new();
    let mut benchmark_generation_started = Instant::now();
    let mut benchmark_ping_ms: Vec<f64> = Vec::new();
    // Creatures of the current generation whose results are stored, and the
    // (experiment, generation) the bitmap belongs to.
    let mut done: Vec<bool> = Vec::new();
    let mut done_key = (u64::MAX, u32::MAX);
    // Steady-state evolution (continuous runs): slots cycle through the engines.
    let mut steady = Steady::default();
    // Completion time and population of recent generations.
    let mut generation_marks: std::collections::VecDeque<(Instant, usize)> = Default::default();
    'worker: loop {
        let first = if running || gpu.async_in_flight() > 0 {
            rx.try_recv().ok()
        } else {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(c) => Some(c),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(_) => None,
            }
        };
        // Handle every queued command now; controls never wait behind a GPU batch.
        let mut commands: Vec<Command> = first.into_iter().collect();
        commands.extend(rx.try_iter());
        for command in commands {
            if matches!(command, Command::Shutdown) {
                break 'worker;
            }
            // Commands that change or persist the experiment first collect the
            // results of queued GPU work, so the completed prefix stays exact.
            if !matches!(
                command,
                Command::Ping(_)
                    | Command::Page(_)
                    | Command::Preview(_)
                    | Command::Pause
                    | Command::Run { .. }
                    | Command::Next
            ) && let Some(e) = &mut exp
                && let Err(err) = finish_queued(&mut gpu, e, &mut done, &mut steady)
            {
                error = Some(format!("{err:#}"));
                running = false;
                changed = true;
                continue;
            }
            if matches!(command, Command::New(_) | Command::Load(_))
                && let Some(handle) = checkpoint_thread.take()
            {
                let _ = handle.join();
            }
            changed = true;
            error = None;
            let result: anyhow::Result<()> = (|| {
                match command {
                    Command::Shutdown => return Ok(()),
                    Command::New(cfg) => {
                        running = false;
                        steady = Steady::default();
                        status = "Creating population…".into();
                        let next = Experiment::new(cfg)?;
                        preview = Some((next.population.creature(0), next.config.clone()));
                        exp = Some(next);
                        epoch += 1;
                        history = Arc::new(Vec::new());
                        page = 0;
                        status = "Population ready".into();
                    }
                    Command::Run {
                        continuous: c,
                        guided: g,
                    } => {
                        if benchmark_generations.is_some() && benchmark_run_generation.is_none() {
                            benchmark_run_generation = exp.as_ref().map(|e| e.generation);
                            if benchmark_warmup == 0 {
                                benchmark_start =
                                    exp.as_ref().map(|e| (e.generation, Instant::now()));
                                benchmark_generation_started = Instant::now();
                                measuring.store(true, Ordering::Relaxed);
                            }
                        }
                        continuous = c;
                        guided = g;
                        pause.store(false, Ordering::Relaxed);
                        running = true;
                    }
                    Command::Pause => {
                        running = false;
                        status = "Paused at a completed batch".into();
                    }
                    Command::Next => {
                        guided = true;
                        continuous = false;
                        pause.store(false, Ordering::Relaxed);
                        running = true;
                    }
                    Command::Configure(cfg) => {
                        if let Some(e) = &mut exp {
                            e.update_config(cfg)?;
                            status = "Settings applied or queued for the next generation".into();
                        }
                    }
                    Command::Save(path) => {
                        if let Some(e) = &exp {
                            storage::save(&path, e)?;
                            status = format!("Saved {}", path.display());
                        }
                    }
                    Command::Load(path) => {
                        steady = Steady::default();
                        let next = storage::load(&path)?;
                        let creature = next
                            .archive
                            .entries
                            .iter()
                            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
                            .map_or_else(
                                || next.population.creature(0),
                                |elite| elite.creature.clone(),
                            );
                        preview = Some((creature, next.config.clone()));
                        exp = Some(next);
                        epoch += 1;
                        history = Arc::new(Vec::new());
                        running = false;
                        page = 0;
                        status = format!("Loaded {}", path.display());
                    }
                    Command::Export(path) => {
                        if let Some(e) = &exp {
                            storage::export_csv(&path, &e.history)?;
                            status = format!("Exported {}", path.display());
                        }
                    }
                    Command::Ping(sent) => {
                        if measuring.load(Ordering::Relaxed) {
                            benchmark_ping_ms.push(sent.elapsed().as_secs_f64() * 1e3);
                        }
                        changed = false;
                    }
                    Command::Page(start) => {
                        page = start;
                    }
                    Command::Preview(i) => {
                        if let Some(e) = &exp {
                            if let Some(elite) = e.archive.entries.get(i) {
                                preview = Some((elite.creature.clone(), e.config.clone()));
                                let chain = e.ancestry(elite.creature.id, 400);
                                lineage = Some(
                                    chain
                                        .iter()
                                        .enumerate()
                                        .map(|(k, a)| LineageStep {
                                            generation: a.generation,
                                            fitness: a.fitness,
                                            gain: chain
                                                .get(k + 1)
                                                .map_or(0.0, |parent| a.fitness - parent.fitness),
                                            change: a.change.clone(),
                                            creature: a.creature.clone(),
                                        })
                                        .collect(),
                                );
                            } else if e.archive.entries.is_empty() && i < e.config.population {
                                preview = Some((e.population.creature(i), e.config.clone()));
                            }
                        }
                    }
                }
                Ok(())
            })();
            if let Err(e) = result {
                error = Some(format!("{e:#}"));
                running = false;
            }
            // Disconnection is handled separately; shutdown closes the receiver after this iteration.
        }
        if pause.load(Ordering::Relaxed) {
            running = false;
        }
        if running {
            if let Some(e) = &mut exp {
                let result: anyhow::Result<()> = (|| {
                    match e.stage {
                        Stage::Ready | Stage::Evaluating
                            if gpu.async_capable() && continuous && !guided =>
                        {
                            let stage_start = Instant::now();
                            e.stage = Stage::Evaluating;
                            let sched = gpu.sched.as_mut().unwrap();
                            if !steady.active {
                                if sched.in_flight() > 0 {
                                    // Results from a generational run: keep them; the
                                    // next pass offers them to the archive.
                                    sched.pump_checks(&e.population, &e.config)?;
                                    for (indices, metrics) in sched.collect(
                                        &e.population,
                                        &e.config,
                                        Duration::from_millis(4),
                                        |i, m| e.contender(i, m),
                                    )? {
                                        for (&i, m) in indices.iter().zip(&metrics) {
                                            e.scores[i] = m.fitness;
                                            e.trial_metrics[i] = m.behavior;
                                        }
                                    }
                                    return Ok(());
                                }
                                // Offer creatures that already have results, breed their
                                // replacements, then keep every slot cycling.
                                let evaluated: Vec<usize> = (0..e.config.population)
                                    .filter(|&i| !e.scores[i].is_nan())
                                    .collect();
                                if !evaluated.is_empty() {
                                    steady.failed += e.archive_slots(&evaluated);
                                    e.breed_slots(&evaluated)?;
                                }
                                sched.stop();
                                sched.begin(&e.population, 0..e.config.population);
                                steady.active = true;
                            }
                            sched.pump(&e.population, &e.config, &[])?;
                            for (indices, metrics) in sched.collect(
                                &e.population,
                                &e.config,
                                Duration::from_millis(4),
                                |i, m| e.contender(i, m),
                            )? {
                                steady_absorb(e, &mut steady, sched, &indices, &metrics, true)?;
                            }
                            status = format!("Evolving · generation {}", e.generation);
                            let seconds = stage_start.elapsed().as_secs_f64();
                            e.evaluation_seconds += seconds;
                            if benchmark_start.is_some() {
                                benchmark_stage_seconds[0] += seconds;
                            }
                        }
                        Stage::Ready | Stage::Evaluating if gpu.async_capable() => {
                            let stage_start = Instant::now();
                            if done.len() != e.config.population || done_key != (epoch, e.generation) {
                                done = vec![false; e.config.population];
                                done[..e.evaluated].fill(true);
                                done_key = (epoch, e.generation);
                            }
                            e.stage = Stage::Evaluating;
                            let sched = gpu.sched.as_mut().unwrap();
                            if sched.in_flight() == 0 {
                                sched.begin(&e.population, e.evaluated..e.config.population);
                            }
                            sched.pump(&e.population, &e.config, &done)?;
                            for (indices, metrics) in sched.collect(
                                &e.population,
                                &e.config,
                                Duration::from_millis(4),
                                |i, m| e.contender(i, m),
                            )? {
                                store_results(e, &mut done, &indices, &metrics);
                            }
                            status = format!("Evaluating generation {}", e.generation);
                            if e.stage == Stage::Evaluated && guided {
                                running = false;
                            }
                            let seconds = stage_start.elapsed().as_secs_f64();
                            e.evaluation_seconds += seconds;
                            if benchmark_start.is_some() {
                                benchmark_stage_seconds[0] += seconds;
                            }
                        }
                        Stage::Ready | Stage::Evaluating => {
                            let stage_start = Instant::now();
                            e.stage = Stage::Evaluating;
                            let batch = e.config.batch_size();
                            let end = (e.evaluated + batch).min(e.config.population);
                            let indices: Vec<_> = (e.evaluated..end).collect();
                            let start = Instant::now();
                            let metrics =
                                gpu.evaluate_with_metrics(&e.population, &indices, &e.config)?;
                            e.evaluation_seconds += start.elapsed().as_secs_f64();
                            for (offset, metric) in metrics.iter().enumerate() {
                                e.scores[e.evaluated + offset] = metric.fitness;
                                e.trial_metrics[e.evaluated + offset] = metric.behavior;
                            }
                            e.evaluated = end;
                            status = format!("Evaluating generation {}", e.generation);
                            if end == e.config.population {
                                e.stage = Stage::Evaluated;
                                if guided {
                                    running = false;
                                }
                            }
                            if benchmark_start.is_some() {
                                benchmark_stage_seconds[0] += stage_start.elapsed().as_secs_f64();
                            }
                        }
                        Stage::Evaluated | Stage::Ranked | Stage::Selected => {
                            let stage_start = Instant::now();
                            e.archive_batch()?;
                            if benchmark_start.is_some() {
                                benchmark_stage_seconds[1] += stage_start.elapsed().as_secs_f64();
                            }
                            status = format!(
                                "Archive: {} niches · QD score {:.2}",
                                e.archive.entries.len(),
                                e.archive.qd_score
                            );
                            if guided {
                                running = false;
                            }
                        }
                        Stage::Archived => {
                            let stage_start = Instant::now();
                            if steady.boundary {
                                steady.boundary = false;
                                let failed = std::mem::take(&mut steady.failed);
                                steady.count = steady.count.saturating_sub(e.config.population);
                                e.finish_steady_generation(failed)?;
                                e.stage = Stage::Evaluating;
                                e.evaluated = steady.count.min(e.config.population);
                            } else if continuous
                                && !guided
                                && let Some(sched) = gpu.sched.as_mut()
                            {
                                // Offspring go to the evaluation engines slice by slice
                                // while the rest of the generation is bred.
                                let slice = (e.config.population / 8).max(4096);
                                e.prepare_next_batch_streaming(slice, |pop, range, cfg| {
                                    sched.extend(pop, range);
                                    sched.pump(pop, cfg, &[])
                                })?;
                                done = vec![false; e.config.population];
                                done_key = (epoch, e.generation);
                            } else {
                                e.prepare_next_batch()?;
                            }
                            if benchmark_start.is_some() {
                                benchmark_stage_seconds[2] += stage_start.elapsed().as_secs_f64();
                            }
                            generation_marks.push_back((Instant::now(), e.config.population));
                            while generation_marks.len() > 2
                                && generation_marks[0].0.elapsed() > Duration::from_secs(10)
                            {
                                generation_marks.pop_front();
                            }
                            if benchmark_start.is_some() {
                                benchmark_generation_seconds
                                    .push(benchmark_generation_started.elapsed().as_secs_f64());
                            }
                            benchmark_generation_started = Instant::now();
                            if benchmark_start.is_none()
                                && let Some(first) = benchmark_run_generation
                                && e.generation.saturating_sub(first) >= benchmark_warmup
                            {
                                benchmark_start = Some((e.generation, Instant::now()));
                                measuring.store(true, Ordering::Relaxed);
                            }
                            if let (Some(target), Some((first, started))) =
                                (benchmark_generations, benchmark_start)
                                && e.generation.saturating_sub(first) >= target
                            {
                                measuring.store(false, Ordering::Relaxed);
                                let seconds = started.elapsed().as_secs_f64();
                                let generations = e.generation - first;
                                let creatures = f64::from(generations) * e.config.population as f64;
                                eprintln!(
                                    "Native generation benchmark: {} generations in {:.6} s ({:.3} generations/s), population {}, duration {} s, throughput {}, warm-up {} generations",
                                    generations,
                                    seconds,
                                    f64::from(generations) / seconds,
                                    e.config.population,
                                    e.config.duration,
                                    e.config.throughput,
                                    benchmark_warmup
                                );
                                eprintln!(
                                    "Native benchmark stages: evaluation {:.6} s, archive {:.6} s, breeding {:.6} s",
                                    benchmark_stage_seconds[0],
                                    benchmark_stage_seconds[1],
                                    benchmark_stage_seconds[2]
                                );
                                let mut per_generation = benchmark_generation_seconds.clone();
                                per_generation.sort_by(f64::total_cmp);
                                eprintln!(
                                    "Native benchmark throughput: evaluation {:.0} creatures/s, end-to-end {:.0} creatures/s; generation seconds min {:.3} median {:.3} max {:.3}",
                                    creatures / benchmark_stage_seconds[0].max(1e-9),
                                    creatures / seconds,
                                    per_generation.first().copied().unwrap_or(0.0),
                                    per_generation
                                        .get(per_generation.len() / 2)
                                        .copied()
                                        .unwrap_or(0.0),
                                    per_generation.last().copied().unwrap_or(0.0)
                                );
                                let mut pings = benchmark_ping_ms.clone();
                                pings.sort_by(f64::total_cmp);
                                let pct = |q: usize| {
                                    pings
                                        .get(
                                            (pings.len() * q / 100)
                                                .min(pings.len().saturating_sub(1)),
                                        )
                                        .copied()
                                        .unwrap_or(0.0)
                                };
                                eprintln!(
                                    "Native benchmark control latency: {} probes, p50 {:.1} ms, p95 {:.1} ms, p99 {:.1} ms, max {:.1} ms",
                                    pings.len(),
                                    pct(50),
                                    pct(95),
                                    pct(99),
                                    pings.last().copied().unwrap_or(0.0)
                                );
                                if let Some(sched) = &gpu.sched {
                                    for device in &sched.devices {
                                        eprintln!(
                                            "Native benchmark device {}: {} creatures, busy {:.3} s, rate {:.0}/s (totals since start)",
                                            device.engine.name(),
                                            device.creatures,
                                            device.busy_seconds,
                                            device.rate
                                        );
                                    }
                                    eprintln!(
                                        "Native benchmark packing {:.3} s",
                                        sched.packing_seconds
                                    );
                                }
                                running = false;
                                ctx.send_viewport_cmd(eframe::egui::ViewportCommand::Close);
                                ctx.request_repaint();
                            }
                            status = "Breeding from diverse archive elites".into();
                            if e.config.checkpoint_interval > 0
                                && e.generation.is_multiple_of(e.config.checkpoint_interval)
                                && checkpoint_thread
                                    .as_ref()
                                    .is_none_or(|handle| handle.is_finished())
                            {
                                if let Some(handle) = checkpoint_thread.take() {
                                    let _ = handle.join();
                                }
                                let path =
                                    PathBuf::from(format!("runs/seed-{}-auto.evo", e.config.seed));
                                let snapshot = e.clone();
                                checkpoint_thread = Some(std::thread::spawn(move || {
                                    if let Err(err) = storage::save(&path, &snapshot) {
                                        eprintln!("Background checkpoint failed: {err:#}");
                                    }
                                }));
                            }
                            if !continuous || guided {
                                running = false;
                            }
                        }
                    }
                    Ok(())
                })();
                if let Err(err) = result {
                    error = Some(format!("{err:#}"));
                    running = false;
                }
                changed = true;
            } else {
                running = false;
            }
        }
        // A pause stops new submissions; queued GPU work still completes and is kept.
        if !running && let Some(sched) = gpu.sched.as_mut() {
            sched.stop();
            if sched.in_flight() > 0
                && let Some(e) = &mut exp
            {
                let absorbed = sched
                    .pump_checks(&e.population, &e.config)
                    .and_then(|()| {
                        sched.collect(
                            &e.population,
                            &e.config,
                            Duration::from_millis(4),
                            |i, m| e.contender(i, m),
                        )
                    })
                    .and_then(|units| {
                        for (indices, metrics) in units {
                            if steady.active {
                                steady_absorb(e, &mut steady, sched, &indices, &metrics, false)?;
                            } else {
                                store_results(e, &mut done, &indices, &metrics);
                            }
                        }
                        Ok(())
                    });
                if let Err(err) = absorbed {
                    error = Some(format!("{err:#}"));
                }
                changed = true;
            }
            if sched.in_flight() == 0 {
                steady.active = false;
            }
        }
        if changed && (last_publish.elapsed() > Duration::from_millis(200) || !running) {
            let snapshot = if let Some(e) = &exp {
                if history.len() != e.history.len() {
                    history = Arc::new(e.history.clone());
                }
                let archive_count = e.archive.entries.len();
                let item_count = if archive_count > 0 {
                    archive_count
                } else {
                    e.config.population
                };
                if page >= item_count {
                    page = 0;
                }
                let end = (page + 120).min(item_count);
                let cards = if archive_count > 0 {
                    let mut order: Vec<_> = (0..archive_count).collect();
                    order.sort_unstable_by(|&a, &b| {
                        e.archive.entries[b]
                            .fitness
                            .total_cmp(&e.archive.entries[a].fitness)
                    });
                    order
                        .into_iter()
                        .enumerate()
                        .skip(page)
                        .take(end.saturating_sub(page))
                        .map(|(rank, i)| {
                            let elite = &e.archive.entries[i];
                            Card {
                                index: i,
                                rank,
                                score: elite.fitness,
                                parent_score: f32::NAN,
                                survivor: false,
                                descriptor: Some(elite.descriptor),
                                emitter: Some(elite.emitter),
                                visits: elite.visits,
                                innovation_reserve: qd::is_morphology_niche(&elite.niche),
                                creature: elite.creature.clone(),
                            }
                        })
                        .collect()
                } else {
                    (page.min(end)..end)
                        .map(|i| Card {
                            index: i,
                            rank: i,
                            score: e.scores[i],
                            parent_score: e.parent_scores.get(i).copied().unwrap_or(f32::NAN),
                            survivor: false,
                            descriptor: None,
                            emitter: None,
                            visits: 0,
                            innovation_reserve: false,
                            creature: e.population.creature(i),
                        })
                        .collect()
                };
                Snapshot {
                    epoch,
                    config: e.config.clone(),
                    generation: e.generation,
                    evaluated: e.evaluated,
                    completed: if done.len() == e.config.population
                        && done_key == (epoch, e.generation)
                    {
                        done.iter().filter(|&&d| d).count().max(e.evaluated)
                    } else {
                        e.evaluated
                    },
                    stage: e.stage,
                    running,
                    history: history.clone(),
                    page: cards,
                    page_start: page,
                    preview: preview.take(),
                    lineage: lineage.take(),
                    gpu: gpu.name.clone(),
                    engines: engine_rows(&gpu),
                    end_to_end: end_to_end_rate(&generation_marks),
                    gpu_bytes: gpu.allocated_bytes,
                    ram_bytes: e.population.bytes()
                        + e.scores.capacity() * 4
                        + e.trial_metrics.capacity()
                            * std::mem::size_of::<crate::qd::TrialMetrics>()
                        + e.ranks.capacity() * 8
                        + e.parents.capacity() * 8
                        + e.archive
                            .entries
                            .iter()
                            .map(|elite| {
                                elite.creature.nodes.len()
                                    * std::mem::size_of::<crate::evolution::NodeGene>()
                                    + elite.creature.bones.len()
                                        * std::mem::size_of::<crate::evolution::Bone>()
                                    + elite.creature.muscles.len()
                                        * std::mem::size_of::<crate::evolution::Muscle>()
                            })
                            .sum::<usize>(),
                    elapsed: e.evaluation_seconds,
                    archive_cells: e.archive.behavior_count(),
                    archive_size: archive_count,
                    innovation_reserve_count: e.archive.morphology_count(),
                    qd_score: e.archive.qd_score,
                    emitters: e.emitter_stats,
                    emitter_weights: qd::emitter_weights(&e.emitter_stats),
                    status: status.clone(),
                    error: error.clone(),
                }
            } else {
                Snapshot {
                    epoch,
                    config: Config::default(),
                    generation: 0,
                    evaluated: 0,
                    completed: 0,
                    stage: Stage::Ready,
                    running: false,
                    history: history.clone(),
                    page: vec![],
                    page_start: 0,
                    preview: None,
                    lineage: None,
                    gpu: gpu.name.clone(),
                    engines: engine_rows(&gpu),
                    end_to_end: end_to_end_rate(&generation_marks),
                    gpu_bytes: gpu.allocated_bytes,
                    ram_bytes: 0,
                    elapsed: 0.,
                    archive_cells: 0,
                    archive_size: 0,
                    innovation_reserve_count: 0,
                    qd_score: 0.0,
                    emitters: [EmitterStats::default(); 4],
                    emitter_weights: qd::emitter_weights(&[EmitterStats::default(); 4]),
                    status: status.clone(),
                    error: error.clone(),
                }
            };
            *output.lock().unwrap() = Some(snapshot);
            ctx.request_repaint();
            changed = false;
            last_publish = Instant::now();
        }
    }
    if let Some(handle) = checkpoint_thread.take() {
        let _ = handle.join();
    }
}

fn engine_rows(gpu: &Gpu) -> Vec<(String, f64, u64)> {
    gpu.sched.as_ref().map_or_else(Vec::new, |sched| {
        sched
            .devices
            .iter()
            .map(|d| (d.engine.name(), d.rate, d.creatures))
            .collect()
    })
}
/// Creatures per second between the oldest and newest recent generation ends.
fn end_to_end_rate(marks: &std::collections::VecDeque<(Instant, usize)>) -> f64 {
    match (marks.front(), marks.back()) {
        (Some(first), Some(last)) if marks.len() > 1 => {
            let creatures: usize = marks.iter().skip(1).map(|&(_, n)| n).sum();
            creatures as f64 / (last.0 - first.0).as_secs_f64().max(1e-3)
        }
        _ => 0.0,
    }
}
/// Stores a finished unit and advances the contiguous evaluated prefix that
/// checkpoints record. Devices can finish units out of order.
fn store_results(
    e: &mut Experiment,
    done: &mut Vec<bool>,
    indices: &[usize],
    metrics: &[crate::qd::EvaluationMetrics],
) {
    if done.len() != e.config.population {
        *done = vec![false; e.config.population];
        done[..e.evaluated].fill(true);
    }
    for (&i, metric) in indices.iter().zip(metrics) {
        e.scores[i] = metric.fitness;
        e.trial_metrics[i] = metric.behavior;
        done[i] = true;
    }
    while e.evaluated < e.config.population && done[e.evaluated] {
        e.evaluated += 1;
    }
    if e.evaluated == e.config.population {
        e.stage = Stage::Evaluated;
    }
}
/// Waits for all queued GPU work and stores its results.
fn finish_queued(
    gpu: &mut Gpu,
    e: &mut Experiment,
    done: &mut Vec<bool>,
    steady: &mut Steady,
) -> anyhow::Result<()> {
    if let Some(sched) = gpu.sched.as_mut() {
        sched.stop();
        while sched.in_flight() > 0 {
            // With no round left, waiting contenders go out for their checks now.
            sched.pump_checks(&e.population, &e.config)?;
            for (indices, metrics) in sched.collect(
                &e.population,
                &e.config,
                Duration::from_millis(100),
                |i, m| e.contender(i, m),
            )? {
                if steady.active {
                    steady_absorb(e, steady, sched, &indices, &metrics, false)?;
                } else {
                    store_results(e, done, &indices, &metrics);
                }
            }
        }
        steady.active = false;
    }
    Ok(())
}

/// Steady-state evolution bookkeeping.
#[derive(Default)]
struct Steady {
    /// Slots are cycling through the engines.
    active: bool,
    /// Evaluations toward the current generation.
    count: usize,
    /// Failed trials in the current generation.
    failed: usize,
    /// A generation's worth of evaluations finished; record it next pass.
    boundary: bool,
}

/// Stores a finished unit, offers it to the archive, breeds replacements into
/// the same slots from the updated archive, and queues them when `resubmit`.
fn steady_absorb(
    e: &mut Experiment,
    steady: &mut Steady,
    sched: &mut crate::scheduler::Scheduler,
    indices: &[usize],
    metrics: &[crate::qd::EvaluationMetrics],
    resubmit: bool,
) -> anyhow::Result<()> {
    for (&i, m) in indices.iter().zip(metrics) {
        e.scores[i] = m.fitness;
        e.trial_metrics[i] = m.behavior;
    }
    steady.failed += e.archive_slots(indices);
    e.breed_slots(indices)?;
    if resubmit {
        sched.extend(&e.population, indices.iter().copied());
    }
    steady.count += indices.len();
    e.evaluated = steady.count.min(e.config.population);
    if steady.count >= e.config.population {
        steady.boundary = true;
        e.stage = Stage::Archived;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn continuous_run_advances_generations() {
        let gpu = Gpu::new("RTX 4060").unwrap();
        let worker = Worker::spawn(gpu, eframe::egui::Context::default());
        let cfg = Config {
            population: 32,
            duration: 0.1,
            random_seed: false,
            checkpoint_interval: 0,
            ..Config::default()
        };
        worker.send(Command::New(cfg));
        worker.send(Command::Run {
            continuous: true,
            guided: false,
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(snapshot) = worker.view.lock().unwrap().take() {
                assert!(snapshot.error.is_none(), "{:?}", snapshot.error);
                if snapshot.generation >= 2 {
                    break;
                }
            }
            assert!(Instant::now() < deadline, "continuous run did not advance");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}
