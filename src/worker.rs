use crate::{
    config::Config,
    evolution::Creature,
    gpu::Gpu,
    qd::{self, Descriptor, Emitter, EmitterStats},
    storage::{self, Experiment, Stage, Stats},
};
use std::{
    io::Write,
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
    /// Wipe out half of every archive's elites (kept as fossils for undo).
    Meteor,
    /// Wipe out the weakest island (kept as fossils for undo).
    Extinction,
    /// Return the fossils of earlier catastrophes to their archives.
    UndoMeteor,
    Save(PathBuf),
    Load(PathBuf),
    Export(PathBuf),
    /// Send the whole archive, ranked by distance, once (`Snapshot::cards`).
    Cards,
    /// Ancestors of the creature with this id, answered in `Snapshot::lineage`.
    Lineage(u64),
    /// Start or stop sending the archive map table with snapshots.
    MapTable(bool),
    /// The archive creature with this id, answered in `Snapshot::selected`.
    Select(u64),
    /// Benchmark probe: the UI send time, used to measure how long queued controls wait.
    Ping(Instant),
    /// Benchmark probe: re-applies the current settings like an environment
    /// button, to measure how long such a change waits.
    ConfigureProbe(Instant),
    Shutdown,
}
/// Body size class by node count: 0 small (up to 5 nodes), 1 medium (6 to
/// 9), 2 large (10 or more).
pub fn size_class(nodes: usize) -> u8 {
    match nodes {
        0..=5 => 0,
        6..=9 => 1,
        _ => 2,
    }
}
/// A key for a creature's body plan: its counts of nodes, bones and muscles
/// and which parts connect to which. Lengths, masses and rhythms stay out,
/// so a small mutation keeps the plan. The sums do not depend on part order.
pub fn body_plan(creature: &Creature) -> u64 {
    let (nodes, bones, muscles) = (
        creature.nodes.len() as u64,
        creature.bones.len() as u64,
        creature.muscles.len() as u64,
    );
    let mut plan = (nodes << 42) ^ (bones << 21) ^ muscles;
    for bone in &creature.bones {
        let (a, b) = (u64::from(bone.a.min(bone.b)), u64::from(bone.a.max(bone.b)));
        plan = plan.wrapping_add(
            a.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(b.wrapping_mul(0xbf58_476d_1ce4_e5b9))
                .rotate_left(17),
        );
    }
    for muscle in &creature.muscles {
        let (a, b) = (
            u64::from(muscle.bone_a.min(muscle.bone_b)),
            u64::from(muscle.bone_a.max(muscle.bone_b)),
        );
        plan = plan.wrapping_add(
            a.wrapping_mul(0x94d0_49bb_1331_11eb)
                .wrapping_add(b.wrapping_mul(0x2545_f491_4f6c_dd1d))
                .rotate_left(29),
        );
    }
    plan
}
/// The whole archive at one moment, ranked by distance: the UI keeps it on
/// screen unchanged until the player asks for a newer one.
#[derive(Clone)]
pub struct CardList {
    /// The generation running when the list was made.
    pub generation: u32,
    /// The world the scores were measured in.
    pub config: Config,
    pub cards: Arc<Vec<Card>>,
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
/// Something that happened to the experiment, for the UI's event feed.
#[derive(Clone)]
pub struct Event {
    /// The generation running when it happened.
    pub generation: u32,
    pub kind: EventKind,
    pub text: String,
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EventKind {
    /// A new experiment began.
    Started,
    /// A saved experiment was opened.
    Opened,
    /// The player changed the world.
    World,
    /// The seasons changed the world.
    Season,
    /// A meteor strike or an extinction.
    Catastrophe,
    /// An undo brought creatures back.
    Undo,
    /// The experiment was saved.
    Saved,
}
/// Events kept per experiment; older ones drop off.
const EVENT_LOG: usize = 200;
fn log_event(events: &mut Arc<Vec<Event>>, generation: u32, kind: EventKind, text: String) {
    let log = Arc::make_mut(events);
    if log.len() >= EVENT_LOG {
        log.remove(0);
    }
    log.push(Event {
        generation,
        kind,
        text,
    });
}
/// The effects that differ between two worlds, as "Ground Flat to Rough, 8 cm".
pub fn world_change_text(before: &Config, after: &Config) -> Option<String> {
    let parts: Vec<String> = crate::environment::EFFECTS
        .iter()
        .filter(|effect| effect.name != "Seasons")
        .filter(|effect| effect.level(before) != effect.level(after))
        .map(|effect| {
            format!(
                "{} {} to {}",
                effect.name,
                effect.levels[effect.level(before)],
                effect.levels[effect.level(after)]
            )
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join(", "))
}
/// Logs a world change between two configs, if the physics changed.
fn log_world_change(
    events: &mut Arc<Vec<Event>>,
    before: &Config,
    after: &Config,
    generation: u32,
    retesting: usize,
) {
    if !before.physics_differs(after) {
        return;
    }
    let season = after.seasons > 0 && after.season_step != before.season_step;
    let change = world_change_text(before, after).unwrap_or_else(|| "The world changed".into());
    let text = if retesting > 0 {
        format!("{change}. Re-testing {retesting} kept creatures in the new world.")
    } else {
        format!("{change}.")
    };
    let kind = if season {
        EventKind::Season
    } else {
        EventKind::World
    };
    log_event(events, generation, kind, text);
}
/// Logs the world changes a history records, so a loaded game's feed still
/// shows them. Saves keep each generation's world in its statistics, which is
/// all this needs.
fn log_history_world_changes(events: &mut Arc<Vec<Event>>, history: &[Stats]) {
    for pair in history.windows(2) {
        let (before, after) = (&pair[0].config, &pair[1].config);
        if before.physics_differs(after) {
            log_world_change(events, before, after, pair[1].generation, 0);
        }
    }
}
/// One occupied behavior cell of the global archive, for the map. The table
/// is small (one row per elite) and carries no body, so it can follow every
/// archive change; a click asks for the body with `Command::Select`.
#[derive(Clone, Copy)]
pub struct MapCell {
    /// Behavior bins: ground contact, cadence, bounce, height, feet.
    pub niche: [u8; 6],
    pub score: f32,
    /// Place in the archive ranking by distance.
    pub rank: usize,
    pub id: u64,
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
/// One island archive at a glance, for the Islands view and the "How
/// evolution works" schematic.
#[derive(Clone)]
pub struct IslandSummary {
    /// Its fastest behavior elite's distance (NaN while it is empty).
    pub best: f32,
    /// Filled behavior niches.
    pub cells: usize,
    /// Its fastest behavior elite.
    pub leader: Option<Creature>,
    /// Its fastest behavior elites with their distances, best first (the
    /// leader is the first).
    pub top: Vec<(f32, Creature)>,
    /// How many of its behavior elites each emitter bred, in `Emitter::ALL`
    /// order.
    pub origins: [usize; qd::EMITTER_COUNT],
}
/// How many top elites an island summary carries.
pub const ISLAND_TOP: usize = 3;
impl IslandSummary {
    pub fn of(island: &qd::QdArchive) -> Self {
        let mut origins = [0; qd::EMITTER_COUNT];
        let mut ranked: Vec<&qd::Elite> = island
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .inspect(|elite| origins[elite.emitter.index()] += 1)
            .collect();
        ranked.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        ranked.truncate(ISLAND_TOP);
        let top: Vec<(f32, Creature)> = ranked
            .iter()
            .map(|elite| (elite.fitness, elite.creature.clone()))
            .collect();
        Self {
            best: top.first().map_or(f32::NAN, |t| t.0),
            cells: island.behavior_count(),
            leader: top.first().map(|t| t.1.clone()),
            top,
            origins,
        }
    }
}
/// The last island migration of this session: the generation it happened at
/// and, per island, the elites it sent and how many its neighbor kept.
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationSummary {
    pub generation: u32,
    pub exchange: Vec<(usize, usize)>,
}
impl MigrationSummary {
    /// Elites island `island` received from the previous island in the ring,
    /// and how many it kept.
    pub fn received(&self, island: usize) -> Option<(usize, usize)> {
        let count = self.exchange.len();
        (count > 0)
            .then(|| self.exchange.get((island + count - 1) % count).copied())
            .flatten()
    }
}
#[derive(Clone)]
pub struct Snapshot {
    pub epoch: u64,
    pub config: Config,
    /// Settings the player asked for that take effect when the next
    /// generation starts.
    pub pending: Option<Config>,
    /// Elites lost to meteor strikes that an undo could bring back.
    pub fossils: usize,
    pub generation: u32,
    pub evaluated: usize,
    /// Creatures of the current generation with stored results. Engines finish
    /// out of order, so this runs ahead of the contiguous `evaluated` prefix.
    pub completed: usize,
    /// Creatures whose standard trial could enter an archive, waiting for
    /// their fine check before their result counts.
    pub checking: usize,
    pub stage: Stage,
    pub running: bool,
    pub history: Arc<Vec<Stats>>,
    /// The archive ranked by distance, sent once per `Command::Cards`.
    pub cards: Option<CardList>,
    pub preview: Option<(Creature, Config)>,
    /// What happened to this experiment, oldest first.
    pub events: Arc<Vec<Event>>,
    /// The archive map table while the UI asks for it.
    pub map: Option<Arc<Vec<MapCell>>>,
    /// A creature the UI asked for with `Command::Select`, and the world it
    /// is scored in; sent once.
    pub selected: Option<(Creature, Config)>,
    /// Ancestor chain of a requested creature (its id first), newest first;
    /// sent once per request.
    pub lineage: Option<(u64, Vec<LineageStep>)>,
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
    /// Each island archive, in island order.
    pub islands: Vec<IslandSummary>,
    /// The last island migration this session, if one happened.
    pub migration: Option<MigrationSummary>,
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
struct StageLog {
    file: std::fs::File,
    started: Instant,
    seconds: [f64; 3],
}
impl StageLog {
    fn open() -> Option<Self> {
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
                "generation,evaluation_seconds,archive_seconds,breeding_seconds,end_to_end_creatures_per_second"
            );
        }
        Some(Self {
            file,
            started: Instant::now(),
            seconds: [0.0; 3],
        })
    }
    fn add(&mut self, stage: usize, seconds: f64) {
        self.seconds[stage] += seconds;
    }
    fn reset(&mut self) {
        self.started = Instant::now();
        self.seconds = [0.0; 3];
    }
    fn write_row(&mut self, generation: u32, population: usize) {
        let seconds = self.started.elapsed().as_secs_f64().max(1e-9);
        if std::env::var_os("EVOLUTION_PROFILE_BREED").is_some() {
            let [rejected, optimizer, global, island, reserve] =
                crate::storage::take_contender_counts();
            eprintln!(
                "Contenders: generation {generation}, optimizer {optimizer}, global {global}, island {island}, reserve {reserve}, not checked {rejected}"
            );
            let [plan, emit, write] = crate::storage::take_breed_nanos();
            eprintln!(
                "Breeding: generation {generation}, plan {:.3} s, emit {:.3} s, write {:.3} s",
                plan as f64 * 1e-9,
                emit as f64 * 1e-9,
                write as f64 * 1e-9
            );
        }
        let _ = writeln!(
            self.file,
            "{generation},{:.6},{:.6},{:.6},{:.3}",
            self.seconds[0],
            self.seconds[1],
            self.seconds[2],
            population as f64 / seconds
        );
        let _ = self.file.flush();
        self.reset();
    }
}
/// A save loading on its own thread.
struct Loading {
    path: std::path::PathBuf,
    progress: Arc<storage::Progress>,
    handle: std::thread::JoinHandle<anyhow::Result<Experiment>>,
    started: Instant,
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
    // The UI asked for the ranked archive (`Command::Cards`).
    let mut send_cards = false;
    let mut preview = None;
    let mut lineage: Option<(u64, Vec<LineageStep>)> = None;
    let mut status = gpu
        .startup_warning
        .clone()
        .unwrap_or_else(|| "Create a population to begin".to_owned());
    let mut error = None;
    let mut last_publish = Instant::now() - Duration::from_secs(1);
    let mut changed = true;
    let mut epoch = 0u64;
    let mut history = Arc::new(Vec::new());
    // A save being loaded on its own thread, and a save waiting until the
    // "Saving" status has reached the window.
    let mut loading: Option<Loading> = None;
    let mut pending_save: Option<std::path::PathBuf> = None;
    let mut deferred: Vec<Command> = Vec::new();
    let mut last_progress = Instant::now();
    let mut events: Arc<Vec<Event>> = Arc::new(Vec::new());
    // Archive map table: whether the UI wants it, the last one built, and the
    // archive state it was built from.
    let mut want_map = false;
    let mut map: Option<Arc<Vec<MapCell>>> = None;
    let mut map_key = (u64::MAX, usize::MAX, 0u64);
    let mut selected: Option<(Creature, Config)> = None;
    // A background autosave, which reports the file and generation it wrote.
    let mut checkpoint_thread: Option<std::thread::JoinHandle<Option<(PathBuf, u32)>>> = None;
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
    let mut benchmark_configure_ms: Vec<f64> = Vec::new();
    let mut stage_log = StageLog::open();
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
        // Commands held back during a load run first once it is done.
        let mut commands: Vec<Command> = if loading.is_none() {
            std::mem::take(&mut deferred)
        } else {
            Vec::new()
        };
        commands.extend(first);
        commands.extend(rx.try_iter());
        for command in commands {
            if matches!(command, Command::Shutdown) {
                break 'worker;
            }
            // While a save loads there is no game to act on: hold every other
            // command, in order, until the load is done.
            if loading.is_some()
                && !matches!(
                    command,
                    Command::New(_) | Command::Load(_) | Command::Ping(_)
                )
            {
                deferred.push(command);
                continue;
            }
            // Commands that change or persist the experiment first collect the
            // results of queued GPU work, so the completed prefix stays exact.
            // A steady run needs no such prefix: settings wait for the next
            // generation boundary, where work in flight carries over anyway,
            // and catastrophes only thin the archives. Draining there froze
            // the worker and idled the GPU for 3 to 9 s per button.
            let steady_safe = steady.active
                && matches!(
                    command,
                    Command::Configure(_)
                        | Command::ConfigureProbe(_)
                        | Command::Meteor
                        | Command::Extinction
                        | Command::UndoMeteor
                        | Command::Export(_)
                );
            if !steady_safe
                && !matches!(
                    command,
                    Command::Ping(_)
                        | Command::Cards
                        | Command::Lineage(_)
                        | Command::MapTable(_)
                        | Command::Select(_)
                        | Command::Pause
                        | Command::Run { .. }
                        | Command::Next
                )
                && let Some(e) = &mut exp
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
                        if let Some(old) = loading.take() {
                            old.progress.cancel.store(true, Ordering::Relaxed);
                        }
                        running = false;
                        steady = Steady::default();
                        status = "Creating population…".into();
                        let next = Experiment::new(cfg)?;
                        preview = Some((next.population.creature(0), next.config.clone()));
                        events = Arc::new(Vec::new());
                        log_event(
                            &mut events,
                            next.generation,
                            EventKind::Started,
                            format!(
                                "New experiment: {} creatures, seed {}",
                                next.config.population, next.config.seed
                            ),
                        );
                        exp = Some(next);
                        epoch += 1;
                        history = Arc::new(Vec::new());
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
                        if let Some(log) = &mut stage_log {
                            log.reset();
                        }
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
                            let before = e.config.clone();
                            e.update_config(cfg)?;
                            // Between generations a change applies at once;
                            // otherwise it waits in `pending` and is logged
                            // when the next generation starts.
                            log_world_change(
                                &mut events,
                                &before,
                                &e.config,
                                e.generation,
                                e.reseed.len(),
                            );
                            status = if e.pending.is_some() {
                                "The world changes when the next generation starts".into()
                            } else {
                                "Settings applied".into()
                            };
                        }
                    }
                    Command::ConfigureProbe(sent) => {
                        if let Some(e) = &mut exp {
                            let cfg = e.config.clone();
                            e.update_config(cfg)?;
                            if measuring.load(Ordering::Relaxed) {
                                benchmark_configure_ms.push(sent.elapsed().as_secs_f64() * 1e3);
                            }
                        }
                    }
                    Command::Meteor => {
                        if let Some(e) = &mut exp {
                            let lost = e.meteor(0.5);
                            status = format!("A meteor wiped out {lost} creatures");
                            log_event(
                                &mut events,
                                e.generation,
                                EventKind::Catastrophe,
                                format!("Meteor strike: {lost} kept creatures wiped out."),
                            );
                        }
                    }
                    Command::Extinction => {
                        if let Some(e) = &mut exp {
                            let lost = e.extinction();
                            status = format!("The slowest group lost all {lost} creatures");
                            log_event(
                                &mut events,
                                e.generation,
                                EventKind::Catastrophe,
                                format!("Extinction: the slowest group lost all {lost} creatures."),
                            );
                        }
                    }
                    Command::UndoMeteor => {
                        if let Some(e) = &mut exp {
                            let back = e.undo_meteor();
                            status = format!("{back} creatures came back");
                            log_event(
                                &mut events,
                                e.generation,
                                EventKind::Undo,
                                format!("Undo: {back} creatures came back."),
                            );
                        }
                    }
                    Command::Save(path) => {
                        if exp.is_some() {
                            // Saved after this status reaches the window.
                            status = format!("Saving {}…", path.display());
                            pending_save = Some(path);
                        }
                    }
                    Command::Load(path) => {
                        // An incompatible save is turned down from its header,
                        // before gigabytes are read.
                        let header = storage::check(&path)?;
                        if let Some(old) = loading.take() {
                            old.progress.cancel.store(true, Ordering::Relaxed);
                        }
                        steady = Steady::default();
                        running = false;
                        // Holding the current game while a 3M save loads
                        // doubles the memory and can push the machine into
                        // swap: let it go first.
                        exp = None;
                        preview = None;
                        lineage = None;
                        epoch += 1;
                        history = Arc::new(Vec::new());
                        let progress = Arc::new(storage::Progress::default());
                        let thread_progress = progress.clone();
                        let thread_path = path.clone();
                        let handle =
                            std::thread::Builder::new()
                                .name("load".into())
                                .spawn(move || {
                                    storage::load_with_progress(
                                        &thread_path,
                                        Some(&thread_progress),
                                    )
                                })?;
                        status = format!(
                            "Loading {} (generation {}, {} creatures)…",
                            path.display(),
                            header.generation,
                            header.population
                        );
                        loading = Some(Loading {
                            path,
                            progress,
                            handle,
                            started: Instant::now(),
                        });
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
                    Command::Cards => {
                        send_cards = true;
                    }
                    Command::MapTable(on) => {
                        want_map = on;
                        if !on {
                            map = None;
                            map_key = (u64::MAX, usize::MAX, 0);
                        }
                    }
                    Command::Select(id) => {
                        if let Some(e) = &exp
                            && let Some(elite) = e
                                .archive
                                .entries
                                .iter()
                                .find(|elite| elite.creature.id == id)
                        {
                            selected = Some((elite.creature.clone(), e.config.clone()));
                        }
                    }
                    Command::Lineage(id) => {
                        if let Some(e) = &exp {
                            let chain = e.ancestry(id, 400);
                            lineage = Some((
                                id,
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
                            ));
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
        if let Some(load) = &loading {
            if load.handle.is_finished() {
                let load = loading.take().expect("a load in progress");
                match load.handle.join() {
                    Ok(Ok(mut next)) => {
                        // A loaded game starts with autosave off, like a new
                        // one, whatever interval the checkpoint carried.
                        next.config.checkpoint_interval = 0;
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
                        events = Arc::new(Vec::new());
                        log_history_world_changes(&mut events, &next.history);
                        log_event(
                            &mut events,
                            next.generation,
                            EventKind::Opened,
                            format!(
                                "Opened {} at generation {}.",
                                load.path.file_name().map_or_else(
                                    || load.path.display().to_string(),
                                    |name| name.to_string_lossy().into_owned()
                                ),
                                next.generation
                            ),
                        );
                        exp = Some(next);
                        epoch += 1;
                        history = Arc::new(Vec::new());
                        status = format!(
                            "Loaded {} in {:.1} s",
                            load.path.display(),
                            load.started.elapsed().as_secs_f64()
                        );
                    }
                    Ok(Err(err)) => {
                        error = Some(format!("{err:#}"));
                        status = format!("Could not load {}", load.path.display());
                    }
                    Err(_) => {
                        error = Some("The loading thread stopped unexpectedly".into());
                        status = format!("Could not load {}", load.path.display());
                    }
                }
                changed = true;
            } else if last_progress.elapsed() > Duration::from_millis(250) {
                let done = load.progress.done.load(Ordering::Relaxed) as f64;
                let total = load.progress.total.load(Ordering::Relaxed).max(1) as f64;
                status = if done < total {
                    format!(
                        "Loading {}: {:.0}% of {:.0} MB, {:.0} s",
                        load.path.display(),
                        100.0 * done / total,
                        total / 1e6,
                        load.started.elapsed().as_secs_f64()
                    )
                } else {
                    // A save holds the archives; the population is bred again.
                    format!(
                        "Loading {}: breeding the population from the archives, {:.0} s",
                        load.path.display(),
                        load.started.elapsed().as_secs_f64()
                    )
                };
                last_progress = Instant::now();
                changed = true;
            }
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
                                    sched.pump_checks(&e.population, &e.config, |i, m| {
                                        e.check_need(i, m)
                                    })?;
                                    for (indices, metrics) in sched.collect(
                                        &e.population,
                                        &e.config,
                                        Duration::from_millis(4),
                                        |i, m| e.contender(i, m),
                                    )? {
                                        for (&i, m) in indices.iter().zip(&metrics) {
                                            e.record_result(i, m);
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
                            sched.pump(&e.population, &e.config, &[], |i, m| e.check_need(i, m))?;
                            // One unit per pass: archiving and breeding a unit
                            // takes a few tenths of a second, and controls are
                            // read between passes.
                            for (indices, metrics) in sched.collect_one(
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
                            if let Some(log) = &mut stage_log {
                                let archive = std::mem::take(&mut steady.stage_seconds[1]);
                                let breeding = std::mem::take(&mut steady.stage_seconds[2]);
                                log.add(0, (seconds - archive - breeding).max(0.0));
                                log.add(1, archive);
                                log.add(2, breeding);
                            }
                        }
                        Stage::Ready | Stage::Evaluating if gpu.async_capable() => {
                            let stage_start = Instant::now();
                            if done.len() != e.config.population
                                || done_key != (epoch, e.generation)
                            {
                                done = vec![false; e.config.population];
                                done[..e.evaluated].fill(true);
                                done_key = (epoch, e.generation);
                            }
                            e.stage = Stage::Evaluating;
                            let sched = gpu.sched.as_mut().unwrap();
                            if sched.in_flight() == 0 {
                                sched.begin(&e.population, e.evaluated..e.config.population);
                            }
                            sched
                                .pump(&e.population, &e.config, &done, |i, m| e.check_need(i, m))?;
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
                            if let Some(log) = &mut stage_log {
                                log.add(0, seconds);
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
                                e.record_result(e.evaluated + offset, metric);
                            }
                            e.evaluated = end;
                            status = format!("Evaluating generation {}", e.generation);
                            if end == e.config.population {
                                e.stage = Stage::Evaluated;
                                if guided {
                                    running = false;
                                }
                            }
                            let evaluation_seconds = stage_start.elapsed().as_secs_f64();
                            if benchmark_start.is_some() {
                                benchmark_stage_seconds[0] += evaluation_seconds;
                            }
                            if let Some(log) = &mut stage_log {
                                log.add(0, evaluation_seconds);
                            }
                        }
                        Stage::Evaluated | Stage::Ranked | Stage::Selected => {
                            let stage_start = Instant::now();
                            e.archive_batch()?;
                            let archive_seconds = stage_start.elapsed().as_secs_f64();
                            if benchmark_start.is_some() {
                                benchmark_stage_seconds[1] += archive_seconds;
                            }
                            if let Some(log) = &mut stage_log {
                                log.add(1, archive_seconds);
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
                            let world_before = e.config.clone();
                            let kept_before = e.archive.entries.len();
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
                                    sched.pump_standard(pop, cfg, &[])
                                })?;
                                done = vec![false; e.config.population];
                                done_key = (epoch, e.generation);
                            } else {
                                e.prepare_next_batch()?;
                            }
                            if e.config.physics_differs(&world_before) {
                                // The generation that just began runs in the
                                // new world, and its first creatures are the
                                // kept ones being re-tested.
                                let retesting = kept_before.min(e.config.population);
                                log_world_change(
                                    &mut events,
                                    &world_before,
                                    &e.config,
                                    e.generation,
                                    retesting,
                                );
                            }
                            let breeding_seconds = stage_start.elapsed().as_secs_f64();
                            if benchmark_start.is_some() {
                                benchmark_stage_seconds[2] += breeding_seconds;
                            }
                            if let Some(log) = &mut stage_log {
                                log.add(2, breeding_seconds);
                                log.write_row(e.generation.saturating_sub(1), e.config.population);
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
                                let mut configures = benchmark_configure_ms.clone();
                                configures.sort_by(f64::total_cmp);
                                if !configures.is_empty() {
                                    eprintln!(
                                        "Native benchmark settings latency: {} probes, median {:.1} ms, max {:.1} ms",
                                        configures.len(),
                                        configures[configures.len() / 2],
                                        configures.last().copied().unwrap_or(0.0)
                                    );
                                }
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
                                            "Native benchmark device {}: {} creatures, busy {:.3} s, idle {:.3} s, rate {:.0}/s (totals since start)",
                                            device.engine.name(),
                                            device.creatures,
                                            device.busy_seconds,
                                            device.idle_seconds,
                                            device.rate
                                        );
                                    }
                                    eprintln!(
                                        "Native benchmark packing {:.3} s, checks {} submitted in {} units (busy {:.3} s), {} released, {} dropped for a shared cell (totals since start)",
                                        sched.packing_seconds,
                                        sched.checks_submitted,
                                        sched.check_units,
                                        sched.check_busy_seconds,
                                        sched.checks_released,
                                        sched.checks_dropped
                                    );
                                }
                                running = false;
                                ctx.send_viewport_cmd(eframe::egui::ViewportCommand::Close);
                                ctx.request_repaint();
                            }
                            status = "Breeding from diverse archive elites".into();
                            // Benchmarks keep autosaves off even for a loaded
                            // checkpoint, which brings its own interval.
                            if e.config.checkpoint_interval > 0
                                && std::env::var_os("EVOLUTION_BENCH_NO_AUTOSAVE").is_none()
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
                                        return None;
                                    }
                                    if let Some(dir) = path.parent() {
                                        // One autosave per experiment piles up: keep the
                                        // three most recent experiments' autosaves.
                                        storage::rotate_autosaves(dir, 3);
                                    }
                                    Some((path, snapshot.generation))
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
                    .pump_checks(&e.population, &e.config, |i, m| e.check_need(i, m))
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
        // A finished autosave goes into the event log, so the UI can say
        // when the experiment was last saved.
        if checkpoint_thread
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = checkpoint_thread.take()
            && let Ok(Some((path, generation))) = handle.join()
        {
            log_event(
                &mut events,
                generation,
                EventKind::Saved,
                format!("Autosaved {}.", path.display()),
            );
            changed = true;
        }
        if changed && (last_publish.elapsed() > Duration::from_millis(200) || !running) {
            let snapshot = if let Some(e) = &exp {
                if history.len() != e.history.len() {
                    history = Arc::new(e.history.clone());
                }
                if want_map {
                    let key = (epoch, e.archive.entries.len(), e.archive.qd_score.to_bits());
                    if map.is_none() || key != map_key {
                        map_key = key;
                        let mut order: Vec<usize> = (0..e.archive.entries.len()).collect();
                        order.sort_unstable_by(|&a, &b| {
                            e.archive.entries[b]
                                .fitness
                                .total_cmp(&e.archive.entries[a].fitness)
                        });
                        map = Some(Arc::new(
                            order
                                .into_iter()
                                .enumerate()
                                .filter(|&(_, i)| {
                                    !qd::is_morphology_niche(&e.archive.entries[i].niche)
                                        && e.archive.entries[i].fitness.is_finite()
                                })
                                .map(|(rank, i)| {
                                    let elite = &e.archive.entries[i];
                                    MapCell {
                                        niche: elite.descriptor.niche().0,
                                        score: elite.fitness,
                                        rank,
                                        id: elite.creature.id,
                                    }
                                })
                                .collect(),
                        ));
                    }
                }
                let archive_count = e.archive.entries.len();
                // The ranked archive, built only when the UI asks: one sort
                // and one copy of each kept creature (about 1,500 at 3M).
                let cards = if std::mem::take(&mut send_cards) {
                    let mut order: Vec<_> = (0..archive_count).collect();
                    order.sort_unstable_by(|&a, &b| {
                        e.archive.entries[b]
                            .fitness
                            .total_cmp(&e.archive.entries[a].fitness)
                            .then(a.cmp(&b))
                    });
                    Some(CardList {
                        generation: e.generation,
                        config: e.config.clone(),
                        cards: Arc::new(
                            order
                                .into_iter()
                                .enumerate()
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
                                .collect(),
                        ),
                    })
                } else {
                    None
                };
                Snapshot {
                    epoch,
                    config: e.config.clone(),
                    pending: e.pending.clone(),
                    fossils: e.fossils.len(),
                    generation: e.generation,
                    evaluated: e.evaluated,
                    completed: if done.len() == e.config.population
                        && done_key == (epoch, e.generation)
                    {
                        done.iter().filter(|&&d| d).count().max(e.evaluated)
                    } else {
                        e.evaluated
                    },
                    checking: gpu.sched.as_ref().map_or(0, |sched| sched.holding()),
                    stage: e.stage,
                    running,
                    history: history.clone(),
                    events: events.clone(),
                    map: map.clone(),
                    selected: selected.take(),
                    cards,
                    preview: preview.take(),
                    lineage: lineage.take(),
                    gpu: gpu.names(),
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
                    islands: e.islands.iter().map(IslandSummary::of).collect(),
                    migration: e.last_migration.clone().map(|(generation, exchange)| {
                        MigrationSummary {
                            generation,
                            exchange,
                        }
                    }),
                    status: status.clone(),
                    error: error.clone(),
                }
            } else {
                Snapshot {
                    epoch,
                    config: Config::default(),
                    pending: None,
                    fossils: 0,
                    generation: 0,
                    evaluated: 0,
                    completed: 0,
                    checking: 0,
                    stage: Stage::Ready,
                    running: false,
                    history: history.clone(),
                    events: events.clone(),
                    map: None,
                    selected: selected.take(),
                    cards: None,
                    preview: None,
                    lineage: None,
                    gpu: gpu.names(),
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
                    islands: Vec::new(),
                    migration: None,
                    status: status.clone(),
                    error: error.clone(),
                }
            };
            *output.lock().unwrap() = Some(snapshot);
            ctx.request_repaint();
            changed = false;
            last_publish = Instant::now();
        }
        // A save runs once its "Saving" status is on screen.
        if let Some(path) = pending_save.take()
            && let Some(e) = &exp
        {
            let started = Instant::now();
            match storage::save(&path, e) {
                Ok(()) => {
                    let seconds = started.elapsed().as_secs_f64();
                    let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
                    status = format!("Saved {} in {:.1} s", path.display(), seconds);
                    log_event(
                        &mut events,
                        e.generation,
                        EventKind::Saved,
                        format!(
                            "Saved {} ({:.1} MB in {:.1} s).",
                            path.display(),
                            bytes as f64 / 1e6,
                            seconds
                        ),
                    );
                }
                Err(err) => {
                    error = Some(format!("{err:#}"));
                    status = format!("Could not save {}", path.display());
                }
            }
            changed = true;
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
        e.record_result(i, metric);
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
            sched.pump_checks(&e.population, &e.config, |i, m| e.check_need(i, m))?;
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
    /// Archive and breeding seconds inside the current generation.
    stage_seconds: [f64; 3],
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
        e.record_result(i, m);
    }
    e.arm_screen_early();
    let archive_started = Instant::now();
    steady.failed += e.archive_slots(indices);
    steady.stage_seconds[1] += archive_started.elapsed().as_secs_f64();
    let breeding_started = Instant::now();
    e.breed_slots(indices)?;
    steady.stage_seconds[2] += breeding_started.elapsed().as_secs_f64();
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
    fn a_loaded_history_rebuilds_its_world_changes() {
        let stats = |generation: u32, wind: f32| {
            let config = Config {
                wind,
                ..Config::default()
            };
            Stats {
                generation,
                best: 0.0,
                median: 0.0,
                worst: 0.0,
                mean: 0.0,
                failed: 0,
                seconds: 0.0,
                population: 1,
                percentiles: vec![0.0; 29],
                histogram: vec![],
                species: vec![],
                representatives: vec![],
                config,
                archive_cells: 0,
                qd_score: 0.0,
                archive_coverage: 0.0,
                emitters: Default::default(),
            }
        };
        let history = vec![stats(0, 0.0), stats(1, -3.0), stats(2, -3.0)];
        let mut events = Arc::new(Vec::new());
        log_history_world_changes(&mut events, &history);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].generation, 1);
        assert!(
            events[0].text.starts_with("Wind Calm to Strong"),
            "{}",
            events[0].text
        );
    }

    #[test]
    fn stage_log_writes_one_csv_row_per_generation() {
        let path =
            std::env::temp_dir().join(format!("evolution-stage-log-{}.csv", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut log = StageLog {
            file: std::fs::File::create(&path).unwrap(),
            started: Instant::now(),
            seconds: [1.0, 2.0, 3.0],
        };
        log.write_row(5, 1000);
        log.add(0, 4.0);
        log.write_row(6, 1000);
        drop(log);
        let text = std::fs::read_to_string(&path).unwrap();
        let rows: Vec<&str> = text.lines().collect();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].starts_with("5,1.000000,2.000000,3.000000,"));
        assert!(rows[1].starts_with("6,4.000000,0.000000,0.000000,"));
        let _ = std::fs::remove_file(&path);
    }

    /// A world change mid-generation empties the archive at the next
    /// boundary, so nearly every result becomes a contender. The run must
    /// keep advancing generations instead of holding every slot for checks.
    #[test]
    #[ignore = "requires a Vulkan GPU"]
    fn a_world_change_mid_generation_keeps_generations_advancing() {
        let gpu = Gpu::new("RTX 4060").unwrap();
        let worker = Worker::spawn(gpu, eframe::egui::Context::default());
        let population = std::env::var("EVOLUTION_TEST_POPULATION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);
        let cfg = Config {
            population,
            duration: 20.0,
            random_seed: false,
            checkpoint_interval: 0,
            throughput: true,
            ..Config::default()
        };
        match std::env::var_os("EVOLUTION_TEST_CHECKPOINT") {
            Some(path) => worker.send(Command::Load(std::path::PathBuf::from(path))),
            None => worker.send(Command::New(cfg)),
        }
        worker.send(Command::Run {
            continuous: true,
            guided: false,
        });
        let started = Instant::now();
        let mut first_generation: Option<u32> = None;
        let mut changed_at: Option<u32> = None;
        let mut last_print = Instant::now();
        loop {
            if let Some(snapshot) = worker.view.lock().unwrap().take() {
                assert!(snapshot.error.is_none(), "{:?}", snapshot.error);
                if last_print.elapsed() > Duration::from_secs(2) {
                    eprintln!(
                        "{:6.1} s: generation {}, evaluated {}, in checks {}, cells {}",
                        started.elapsed().as_secs_f64(),
                        snapshot.generation,
                        snapshot.evaluated,
                        snapshot.checking,
                        snapshot.archive_cells
                    );
                    last_print = Instant::now();
                }
                let first = *first_generation.get_or_insert(snapshot.generation);
                if changed_at.is_none()
                    && snapshot.generation >= first + 2
                    && snapshot.evaluated > snapshot.config.population / 2
                {
                    let mut rough = snapshot.config.clone();
                    rough.terrain = 3;
                    worker.send(Command::Configure(rough));
                    changed_at = Some(snapshot.generation);
                    eprintln!("roughened at generation {}", snapshot.generation);
                }
                if let Some(at) = changed_at
                    && snapshot.generation >= at + 3
                {
                    break;
                }
            }
            assert!(
                started.elapsed() < Duration::from_secs(900),
                "the run stopped advancing after the world change"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

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
