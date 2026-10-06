use crate::{
    config::Config,
    evolution::Creature,
    gpu::Gpu,
    qd::{self, Descriptor, Emitter, EmitterStats},
    storage::{Experiment, Stats},
};
use std::{
    ops::ControlFlow,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};
mod autosave;
mod benchmark;
mod commands;
mod files;
mod stage_log;
use autosave::Autosave;
use benchmark::{Bench, Benchmark};
use files::Loading;
use stage_log::{RingMeter, StageLog};
pub enum Command {
    New(Config),
    /// Evolve; a run that is not `continuous`, or is `guided`, stops after
    /// one generation.
    Run {
        continuous: bool,
        guided: bool,
    },
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
    /// The score is the confirmation trial's: replay at fine fidelity.
    pub fine: bool,
}
impl Card {
    /// The creature and world of the trial this card's score came from.
    pub fn replay_of(&self, cfg: &Config) -> (Creature, Config) {
        qd::replay_of(&self.creature, self.fine, cfg)
    }
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
    /// Autochange changed the world.
    Autochange,
    /// A meteor strike or an extinction.
    Catastrophe,
    /// An undo brought creatures back.
    Undo,
    /// The experiment was saved.
    Saved,
    /// The GPU failed and was opened again, or could not be.
    Gpu,
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
        .filter(|effect| effect.name != "Autochange environment")
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
    let autochange = after.autochange > 0 && after.autochange_step != before.autochange_step;
    let change = world_change_text(before, after).unwrap_or_else(|| "The world changed".into());
    let text = if retesting > 0 {
        format!("{change}. Re-testing {retesting} kept creatures in the new world.")
    } else {
        format!("{change}.")
    };
    let kind = if autochange {
        EventKind::Autochange
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
    /// Filled behavior niches (a way of moving for each body class).
    pub cells: usize,
    /// Ways of moving the island covers, counting its cells without their
    /// body classes.
    pub moves: usize,
    /// Its fastest behavior elite.
    pub leader: Option<Creature>,
    /// Its fastest behavior elites with their distances, best first (the
    /// leader is the first).
    pub top: Vec<(f32, Creature)>,
    /// How many of its behavior elites each emitter bred, in `Emitter::ALL`
    /// order. Elites that grew up in one of the island's nurseries count as
    /// new bodies, whatever emitter bred them last.
    pub origins: [usize; qd::EMITTER_COUNT],
    /// Bodies in the island's nurseries now (new random bodies and reshaped
    /// bodies).
    pub nursery: usize,
    /// The nurseries' fastest distance (NaN while both are empty).
    pub nursery_best: f32,
    /// What the nurseries graduated this session, the two together.
    pub graduation: crate::storage::Graduation,
}
/// How many top elites an island summary carries.
pub const ISLAND_TOP: usize = 3;
impl IslandSummary {
    pub fn of(
        island: &qd::QdArchive,
        nurseries: [&qd::QdArchive; 2],
        graduation: crate::storage::Graduation,
    ) -> Self {
        let mut origins = [0; qd::EMITTER_COUNT];
        let mut ranked: Vec<&qd::Elite> = island
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .inspect(|elite| {
                let origin = if elite.graduate {
                    qd::Emitter::Restart
                } else {
                    elite.emitter
                };
                origins[origin.index()] += 1;
            })
            .collect();
        ranked.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        ranked.truncate(ISLAND_TOP);
        let top: Vec<(f32, Creature)> = ranked
            .iter()
            .map(|elite| (elite.fitness, elite.creature.unpack()))
            .collect();
        Self {
            best: top.first().map_or(f32::NAN, |t| t.0),
            cells: island.behavior_count(),
            moves: island.movement_count(),
            leader: top.first().map(|t| t.1.clone()),
            top,
            origins,
            nursery: nurseries.iter().map(|n| n.behavior_count()).sum(),
            nursery_best: nurseries
                .iter()
                .filter(|n| !n.entries.is_empty())
                .map(|n| n.best_fitness())
                .reduce(f32::max)
                .unwrap_or(f32::NAN),
            graduation,
        }
    }
}
/// The last migration to the hub this session: the generation it happened
/// at and, per island, the elites it sent and how many the hub kept (zero for
/// the hub itself).
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationSummary {
    pub generation: u32,
    pub exchange: Vec<(usize, usize)>,
}
impl MigrationSummary {
    /// Elites the hub received from all isolated islands, and how many it
    /// kept.
    pub fn hub_received(&self) -> (usize, usize) {
        self.exchange
            .iter()
            .fold((0, 0), |(sent, kept), &(s, k)| (sent + s, kept + k))
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
    /// Evaluations absorbed toward the current generation.
    pub completed: usize,
    /// Confirmation trials running for creatures that would set an island
    /// record.
    pub checking: usize,
    pub running: bool,
    pub history: Arc<Vec<Stats>>,
    /// The archive ranked by distance, sent once per `Command::Cards`.
    pub cards: Option<CardList>,
    pub preview: Option<(Creature, Config)>,
    /// The best elite of the global archive now, and the world it is scored
    /// in. It changes as soon as a new record is absorbed, mid-generation
    /// too, so the world view can switch to it at once.
    pub champion: Option<Arc<(Creature, Config)>>,
    /// The best distance in the archive now and the median of its behavior
    /// elites (NaN before any elite), the same numbers a history row keeps at
    /// the end of a generation.
    pub live_best: f32,
    pub live_median: f32,
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
    /// Ways of moving the global archive covers, counting its cells
    /// without their body classes.
    pub movement_cells: usize,
    pub archive_size: usize,
    pub innovation_reserve_count: usize,
    pub qd_score: f64,
    pub emitters: [EmitterStats; 4],
    pub emitter_weights: [f64; 4],
    /// Each island archive, in island order.
    pub islands: Vec<IslandSummary>,
    /// The last island migration this session, if one happened.
    pub migration: Option<MigrationSummary>,
    /// Per island, how many of its wild migrants took a hub cell.
    pub wild_wins: Vec<u32>,
    /// The main islands' elite with the body farthest from the others.
    pub strangest: Option<Creature>,
    pub status: String,
    pub error: Option<String>,
}
pub struct Worker {
    pub tx: Sender<Command>,
    pub view: Arc<Mutex<Option<Snapshot>>>,
    pub pause: Arc<AtomicBool>,
    /// True while a native benchmark is inside its measured window (after warm-up).
    pub measuring: Arc<AtomicBool>,
    /// Start and end of every ring step that absorbed and bred a block while
    /// measuring, and whether it ended a generation, for the benchmark's
    /// frame times and control latency during breeding.
    pub breeding: Arc<Mutex<Vec<(Instant, Instant, bool)>>>,
    /// A pause developers asked for from outside the game (`dev_pause`).
    pub dev_pause: Arc<crate::dev_pause::Shared>,
    /// Set once the evaluation devices have opened (or failed to).
    pub opened: Arc<AtomicBool>,
    /// Why the devices could not open, when they could not.
    pub failed: Arc<Mutex<Option<String>>>,
    join: Option<std::thread::JoinHandle<()>>,
}
impl Worker {
    pub fn spawn(gpu: Gpu, ctx: eframe::egui::Context) -> Self {
        Self::spawn_with_pause_dir(gpu, ctx, crate::dev_pause::dir())
    }
    /// `spawn`, watching `pause_dir` for developer pause requests.
    pub fn spawn_with_pause_dir(gpu: Gpu, ctx: eframe::egui::Context, pause_dir: PathBuf) -> Self {
        Self::start(ctx, pause_dir, move || Ok(gpu))
    }
    /// Opens the evaluation devices on the worker's own thread, so the window
    /// can draw its loading screen while they open. Commands sent before
    /// then wait in the channel.
    pub fn open(primary: String, ctx: eframe::egui::Context) -> Self {
        Self::start(ctx, crate::dev_pause::dir(), move || Gpu::new(&primary))
    }
    fn start(
        ctx: eframe::egui::Context,
        pause_dir: PathBuf,
        open: impl FnOnce() -> anyhow::Result<Gpu> + Send + 'static,
    ) -> Self {
        let dev_pause = Arc::new(crate::dev_pause::Shared::default());
        let dev = crate::dev_pause::DevPause::new(pause_dir, dev_pause.clone());
        let (tx, rx) = mpsc::channel();
        let view = Arc::new(Mutex::new(None));
        let output = view.clone();
        let pause = Arc::new(AtomicBool::new(false));
        let paused = pause.clone();
        let measuring = Arc::new(AtomicBool::new(false));
        let breeding = Arc::new(Mutex::new(Vec::new()));
        let bench = Bench {
            measuring: measuring.clone(),
            breeding: breeding.clone(),
        };
        let opened = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(Mutex::new(None));
        let (open_flag, open_error) = (opened.clone(), failed.clone());
        let join = std::thread::Builder::new()
            .name("evolution".into())
            .spawn(move || {
                let gpu = open();
                open_flag.store(true, Ordering::Relaxed);
                match gpu {
                    Ok(gpu) => run(gpu, rx, output, paused, bench, ctx, dev),
                    Err(error) => {
                        *open_error.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some(format!("{error:#}"));
                        ctx.request_repaint();
                    }
                }
            })
            .expect("Start simulation worker");
        Self {
            tx,
            view,
            pause,
            measuring,
            breeding,
            dev_pause,
            opened,
            failed,
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
/// Ring steps of one search pass, run on a helper thread.
#[derive(Default)]
struct Pass {
    step: crate::ring::Step,
    /// Send and read times of the pings read during the pass.
    pings: Vec<(Instant, Instant)>,
    /// Steps that absorbed a block: start, end, and whether it ended a
    /// generation.
    breeding: Vec<(Instant, Instant, bool)>,
    /// Each absorbed block: its generation, its host time (`ring::Step::chain`),
    /// whether it ended the generation, and the engines' idle seconds so far.
    blocks: Vec<BlockNote>,
}
/// What one absorbed block tells the ring meter and the stage log.
struct BlockNote {
    generation: u32,
    chain: f64,
    boundary: bool,
    idle: f64,
}
/// Longest search pass: snapshots are built between passes.
const PASS_LIMIT: Duration = Duration::from_millis(100);
/// Runs ring steps on `helper`, a thread on the pool's CPUs, until a
/// generation ends, a command arrives or `PASS_LIMIT` passes, while this
/// thread reads commands every millisecond. Absorbing and breeding a block
/// takes a few tenths of a second at 3M; the worker keeps reading commands
/// through it. Pings are answered at once. Other commands act on the
/// experiment, so they wait in `deferred`, in order, for the step in
/// progress to end.
fn search_pass(
    helper: &crate::threads::Helper,
    e: &mut Experiment,
    sched: &mut crate::scheduler::Scheduler,
    ring: &mut crate::ring::Ring,
    rx: &Receiver<Command>,
    deferred: &mut Vec<Command>,
) -> anyhow::Result<Pass> {
    let stop = AtomicBool::new(false);
    let mut pings = Vec::new();
    let mut result = helper.run(
        || -> anyhow::Result<Pass> {
            let started = Instant::now();
            let mut pass = Pass::default();
            loop {
                let step_started = Instant::now();
                let generation = e.generation;
                let step = ring.step(e, sched, Duration::from_millis(4), 1)?;
                if step.absorbed > 0 {
                    pass.blocks.push(BlockNote {
                        generation,
                        chain: step.chain,
                        boundary: step.generations > 0,
                        idle: sched.devices.iter().map(|d| d.idle_seconds).sum(),
                    });
                    pass.breeding
                        .push((step_started, Instant::now(), step.generations > 0));
                }
                pass.step.absorbed += step.absorbed;
                pass.step.generations += step.generations;
                if pass.step.generations > 0
                    || stop.load(Ordering::Relaxed)
                    || started.elapsed() >= PASS_LIMIT
                {
                    return Ok(pass);
                }
            }
        },
        || match rx.recv_timeout(Duration::from_millis(1)) {
            Ok(Command::Ping(sent)) => pings.push((sent, Instant::now())),
            Ok(command) => {
                deferred.push(command);
                stop.store(true, Ordering::Relaxed);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The UI is gone: the loop ends after this pass.
                stop.store(true, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(1));
            }
        },
    );
    if let Ok(pass) = &mut result {
        pass.pings = pings;
    }
    result
}
/// The worker thread's state: what `run` kept in local variables. The
/// fields drop in the order the locals did: `gpu` last, after the game, the
/// helper thread and the developer pause.
struct Loop {
    /// Completion time and population of recent generations.
    generation_marks: std::collections::VecDeque<(Instant, usize)>,
    /// The generation a run of one generation stops at.
    run_until: Option<u32>,
    ring_meter: RingMeter,
    /// The creatures in flight.
    ring: crate::ring::Ring,
    stage_log: Option<StageLog>,
    benchmark: Benchmark,
    autosave: Autosave,
    selected: Option<(Creature, Config)>,
    /// The archive state the map table was built from.
    map_key: (u64, usize, u64),
    /// The last archive map table built.
    map: Option<Arc<Vec<MapCell>>>,
    /// Whether the UI wants the archive map table.
    want_map: bool,
    events: Arc<Vec<Event>>,
    last_progress: Instant,
    deferred: Vec<Command>,
    /// A save waiting until the "Saving" status has reached the window.
    pending_save: Option<std::path::PathBuf>,
    /// A save being loaded on its own thread.
    loading: Option<Loading>,
    history: Arc<Vec<Stats>>,
    epoch: u64,
    changed: bool,
    last_publish: Instant,
    error: Option<String>,
    status: String,
    lineage: Option<(u64, Vec<LineageStep>)>,
    /// The (epoch, id) the live champion is for.
    champion_key: Option<(u64, u64)>,
    /// The live champion sent with snapshots.
    champion: Option<Arc<(Creature, Config)>>,
    preview: Option<(Creature, Config)>,
    /// The UI asked for the ranked archive (`Command::Cards`).
    send_cards: bool,
    running: bool,
    exp: Option<Experiment>,
    helper: crate::threads::Helper,
    dev: crate::dev_pause::DevPause,
    ctx: eframe::egui::Context,
    pause: Arc<AtomicBool>,
    output: Arc<Mutex<Option<Snapshot>>>,
    rx: Receiver<Command>,
    gpu: Gpu,
}
impl Loop {
    fn new(
        gpu: Gpu,
        rx: Receiver<Command>,
        output: Arc<Mutex<Option<Snapshot>>>,
        pause: Arc<AtomicBool>,
        bench: Bench,
        ctx: eframe::egui::Context,
        dev: crate::dev_pause::DevPause,
    ) -> Self {
        let helper = crate::threads::Helper::new("search");
        Self {
            generation_marks: Default::default(),
            run_until: None,
            ring_meter: RingMeter::default(),
            ring: crate::ring::Ring::default(),
            stage_log: StageLog::open(),
            benchmark: Benchmark::new(bench),
            autosave: Autosave::default(),
            selected: None,
            map_key: (u64::MAX, usize::MAX, 0u64),
            map: None,
            want_map: false,
            events: Arc::new(Vec::new()),
            last_progress: Instant::now(),
            deferred: Vec::new(),
            pending_save: None,
            loading: None,
            history: Arc::new(Vec::new()),
            epoch: 0,
            changed: true,
            last_publish: Instant::now() - Duration::from_secs(1),
            error: None,
            status: "Create a population to begin".to_owned(),
            lineage: None,
            champion_key: None,
            champion: None,
            preview: None,
            send_cards: false,
            running: false,
            exp: None,
            helper,
            dev,
            ctx,
            pause,
            output,
            rx,
            gpu,
        }
    }
    /// One pass of the worker's loop: commands, a load in progress, one
    /// step of evolution, the snapshot and a requested save. `Break` ends the
    /// loop.
    fn pass(&mut self) -> ControlFlow<()> {
        let first = self.next_command()?;
        self.handle_commands(first)?;
        self.poll_load();
        if self.pause.load(Ordering::Relaxed) {
            self.running = false;
        }
        if let Some(text) = self.dev.tick(self.gpu.sched.as_mut()) {
            let generation = self.exp.as_ref().map_or(0, |e| e.generation);
            log_event(&mut self.events, generation, EventKind::Gpu, text);
            self.changed = true;
        }
        if let Some(e) = &mut self.exp
            && let Some(sched) = self.gpu.sched.as_mut()
            && (self.running || self.ring.active())
        {
            let result: anyhow::Result<()> = (|| {
                if !self.running {
                    // A pause stops new submissions and absorption; work on
                    // the engines still completes and waits in the ring.
                    self.ring.step(e, sched, Duration::from_millis(4), 0)?;
                    return Ok(());
                }
                let pass_started = Instant::now();
                let world_before = e.config.clone();
                let kept_before = e.archive.entries.len();
                if !self.ring.active() {
                    self.ring.start(e, sched);
                }
                sched.pump()?;
                // Blocks are absorbed in ring order, so the run does not
                // depend on which unit finished first, nor on when commands
                // are read.
                let pass = search_pass(
                    &self.helper,
                    e,
                    sched,
                    &mut self.ring,
                    &self.rx,
                    &mut self.deferred,
                )?;
                let step = pass.step;
                for note in &pass.blocks {
                    self.ring_meter
                        .add(note.generation, note.chain, note.boundary);
                    if let Some(log) = &mut self.stage_log {
                        log.block(note.idle);
                    }
                }
                self.benchmark.pass_done(&pass.pings, &pass.breeding);
                let seconds = pass_started.elapsed().as_secs_f64();
                e.evaluation_seconds += seconds;
                let [archive, breeding] = std::mem::take(&mut e.stage_seconds);
                let evaluation = (seconds - archive - breeding).max(0.0);
                self.benchmark
                    .add_stage_seconds([evaluation, archive, breeding]);
                if let Some(log) = &mut self.stage_log {
                    log.add(0, evaluation);
                    log.add(1, archive);
                    log.add(2, breeding);
                }
                self.status = format!("Evolving · generation {}", e.generation);
                // A developer's generation dump (EVOLUTION_DUMP_GENERATION)
                // says where it went.
                if let Some(text) = e.dump_notice.take() {
                    log_event(&mut self.events, e.generation, EventKind::Saved, text);
                }
                if step.generations == 0 {
                    return Ok(());
                }
                if e.config.physics_differs(&world_before) {
                    // Autochange changed the world at the boundary: blocks
                    // not yet on an engine run in the new world, and the
                    // kept elites are tested again in it.
                    let lost = self.ring.world_changed(e, sched);
                    if let Some(log) = &mut self.stage_log {
                        log.discarded += lost;
                    }
                    log_world_change(
                        &mut self.events,
                        &world_before,
                        &e.config,
                        e.generation,
                        kept_before.min(e.config.population),
                    );
                }
                if let Some(log) = &mut self.stage_log {
                    let genomes = e.blocks.iter().flat_map(|b| &b.population.genomes);
                    let count = e.ring_len().max(1) as f64;
                    let nodes = [
                        genomes.clone().map(|g| g.node_count as f64).sum::<f64>() / count,
                        genomes.filter(|g| g.node_count > 8).count() as f64 / count,
                    ];
                    log.write_row(
                        e.generation.saturating_sub(1),
                        e.config.population,
                        Some(&*sched),
                        nodes,
                        e.ring,
                        &self.ring_meter,
                        e.rungs.last(),
                    );
                }
                self.generation_marks
                    .push_back((Instant::now(), e.config.population));
                while self.generation_marks.len() > 2
                    && self.generation_marks[0].0.elapsed() > Duration::from_secs(10)
                {
                    self.generation_marks.pop_front();
                }
                if self.benchmark.generation_done(e, sched, &self.ctx) {
                    self.running = false;
                }
                self.autosave.start_if_due(e);
                if self.run_until.is_some_and(|until| e.generation >= until) {
                    self.running = false;
                    self.status = format!("Paused after generation {}", e.generation - 1);
                }
                Ok(())
            })();
            if let Err(err) = result {
                self.error = Some(format!("{err:#}"));
                self.running = false;
            }
            self.changed = true;
        } else if self.running && self.exp.is_none() {
            self.running = false;
        }
        // A GPU that was lost and reopened is told to the player.
        if let Some(sched) = self.gpu.sched.as_mut() {
            for notice in sched.take_notices() {
                let generation = self.exp.as_ref().map_or(0, |e| e.generation);
                log_event(&mut self.events, generation, EventKind::Gpu, notice);
                self.changed = true;
            }
        }
        // A finished autosave goes into the event log, so the UI can say
        // when the experiment was last saved.
        if let Some((path, generation)) = self.autosave.finished() {
            log_event(
                &mut self.events,
                generation,
                EventKind::Saved,
                format!("Autosaved {}.", path.display()),
            );
            self.changed = true;
        }
        if self.changed
            && (self.last_publish.elapsed() > Duration::from_millis(200) || !self.running)
        {
            let build_started = Instant::now();
            let snapshot = if let Some(e) = &self.exp {
                if self.history.len() != e.history.len() {
                    self.history = Arc::new(e.history.clone());
                }
                if self.want_map {
                    let key = (
                        self.epoch,
                        e.archive.entries.len(),
                        e.archive.qd_score.to_bits(),
                    );
                    if self.map.is_none() || key != self.map_key {
                        self.map_key = key;
                        let mut order: Vec<usize> = (0..e.archive.entries.len()).collect();
                        order.sort_unstable_by(|&a, &b| {
                            e.archive.entries[b]
                                .fitness
                                .total_cmp(&e.archive.entries[a].fitness)
                        });
                        self.map = Some(Arc::new(
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
                // The best elite by distance, the first one on a tie. Only a
                // new record clones a creature.
                let best = e
                    .archive
                    .entries
                    .iter()
                    .filter(|elite| elite.fitness.is_finite())
                    .reduce(|a, b| if b.fitness > a.fitness { b } else { a });
                let key = best.map(|elite| (self.epoch, elite.creature.id));
                if key != self.champion_key {
                    self.champion_key = key;
                    self.champion = best.map(|elite| Arc::new(elite.replay_of(&e.config)));
                }
                let live_best = best.map_or(f32::NAN, |elite| elite.fitness.max(0.0));
                let live_median = {
                    // The best elite of each way of moving, like a history row.
                    let mut kept: Vec<f32> = e
                        .archive
                        .best_per_way_of_moving()
                        .into_iter()
                        .map(|slot| e.archive.entries[slot].fitness)
                        .collect();
                    if kept.is_empty() {
                        f32::NAN
                    } else {
                        // The same rank a history row's median uses.
                        let rank = ((kept.len() - 1) as f32 * 0.5).round() as usize;
                        kept.select_nth_unstable_by(rank, |a, b| b.total_cmp(a));
                        kept[rank]
                    }
                };
                let archive_count = e.archive.entries.len();
                // The ranked archive, built only when the UI asks: one sort
                // and one copy of each kept creature (about 1,500 at 3M).
                let cards = if std::mem::take(&mut self.send_cards) {
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
                                        creature: elite.creature.unpack(),
                                        fine: elite.fine,
                                    }
                                })
                                .collect(),
                        ),
                    })
                } else {
                    None
                };
                Snapshot {
                    epoch: self.epoch,
                    config: e.config.clone(),
                    pending: e.pending.clone(),
                    fossils: e.fossils.len(),
                    generation: e.generation,
                    evaluated: e.evaluated,
                    completed: e.evaluated,
                    checking: self.ring.confirming(),
                    running: self.running,
                    history: self.history.clone(),
                    events: self.events.clone(),
                    map: self.map.clone(),
                    selected: self.selected.take(),
                    cards,
                    preview: self.preview.take(),
                    champion: self.champion.clone(),
                    live_best,
                    live_median,
                    lineage: self.lineage.take(),
                    gpu: self.gpu.names(),
                    engines: engine_rows(&self.gpu),
                    end_to_end: end_to_end_rate(&self.generation_marks),
                    gpu_bytes: self.gpu.allocated_bytes,
                    ram_bytes: e.ring_bytes()
                        + e.archive
                            .entries
                            .iter()
                            .map(|elite| {
                                elite.creature.node_count()
                                    * std::mem::size_of::<crate::evolution::NodeGene>()
                                    + elite.creature.bone_count()
                                        * std::mem::size_of::<crate::evolution::Bone>()
                                    + elite.creature.muscle_count()
                                        * std::mem::size_of::<crate::evolution::Muscle>()
                            })
                            .sum::<usize>(),
                    elapsed: e.evaluation_seconds,
                    archive_cells: e.archive.behavior_count(),
                    movement_cells: e.archive.movement_count(),
                    archive_size: archive_count,
                    innovation_reserve_count: e.archive.morphology_count(),
                    qd_score: e.archive.qd_score,
                    emitters: e.emitter_stats,
                    emitter_weights: qd::emitter_weights(&e.emitter_stats),
                    islands: (0..crate::storage::island_count())
                        .filter_map(|i| {
                            let log = |logs: &[crate::storage::Graduation]| {
                                logs.get(i).copied().unwrap_or_default()
                            };
                            let (random, reshaped) =
                                (log(&e.graduations), log(&e.reshaped_graduations));
                            Some(IslandSummary::of(
                                e.islands.get(i)?,
                                [
                                    e.islands.get(crate::storage::nursery_of(i))?,
                                    e.islands.get(crate::storage::reshaped_of(i))?,
                                ],
                                crate::storage::Graduation {
                                    generation: random.generation.max(reshaped.generation),
                                    sent: random.sent + reshaped.sent,
                                    kept: random.kept + reshaped.kept,
                                    kept_total: random.kept_total + reshaped.kept_total,
                                },
                            ))
                        })
                        .collect(),
                    wild_wins: e.wild_wins.clone(),
                    strangest: e
                        .islands
                        .iter()
                        .take(qd::MAIN_ISLANDS)
                        .filter_map(|island| island.strangest())
                        .max_by(|a, b| a.0.total_cmp(&b.0))
                        .map(|(_, elite)| elite.creature.unpack()),
                    migration: e.last_migration.clone().map(|(generation, exchange)| {
                        MigrationSummary {
                            generation,
                            exchange,
                        }
                    }),
                    status: self.status.clone(),
                    error: self.error.clone(),
                }
            } else {
                Snapshot {
                    epoch: self.epoch,
                    config: Config::default(),
                    pending: None,
                    fossils: 0,
                    generation: 0,
                    evaluated: 0,
                    completed: 0,
                    checking: 0,
                    running: false,
                    history: self.history.clone(),
                    events: self.events.clone(),
                    map: None,
                    selected: self.selected.take(),
                    cards: None,
                    preview: None,
                    champion: None,
                    live_best: f32::NAN,
                    live_median: f32::NAN,
                    lineage: None,
                    gpu: self.gpu.names(),
                    engines: engine_rows(&self.gpu),
                    end_to_end: end_to_end_rate(&self.generation_marks),
                    gpu_bytes: self.gpu.allocated_bytes,
                    ram_bytes: 0,
                    elapsed: 0.,
                    archive_cells: 0,
                    movement_cells: 0,
                    archive_size: 0,
                    innovation_reserve_count: 0,
                    qd_score: 0.0,
                    emitters: [EmitterStats::default(); 4],
                    emitter_weights: qd::emitter_weights(&[EmitterStats::default(); 4]),
                    islands: Vec::new(),
                    wild_wins: Vec::new(),
                    strangest: None,
                    migration: None,
                    status: self.status.clone(),
                    error: self.error.clone(),
                }
            };
            self.benchmark
                .snapshot_built(build_started.elapsed().as_secs_f64() * 1e3);
            *self.output.lock().unwrap() = Some(snapshot);
            self.ctx.request_repaint();
            self.changed = false;
            self.last_publish = Instant::now();
        }
        self.save_pending();
        ControlFlow::Continue(())
    }
    /// The loop has ended: waits for a running autosave.
    fn finish(mut self) {
        self.autosave.join();
    }
}
fn run(
    gpu: Gpu,
    rx: Receiver<Command>,
    output: Arc<Mutex<Option<Snapshot>>>,
    pause: Arc<AtomicBool>,
    bench: Bench,
    ctx: eframe::egui::Context,
    dev: crate::dev_pause::DevPause,
) {
    // The Rayon pool must exist before this thread is pinned: a pool built
    // lazily by a pinned thread inherits its single CPU, and all of breeding
    // and archiving then runs on one core. The game builds its pool in `main`
    // (`threads::pool_threads`), so this is a no-op there. Examples and tests
    // that spawn a worker without one get the default pool, built on this
    // thread's full CPU set.
    let _ = rayon::current_num_threads();
    crate::threads::pin_worker();
    let mut state = Loop::new(gpu, rx, output, pause, bench, ctx, dev);
    while state.pass().is_continue() {}
    state.finish();
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
                ring: Default::default(),
                plans: 0,
                clades: 0.0,
                plan_age: 0.0,
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

    /// A world change mid-generation empties the archives, so nearly every
    /// block sets island records again. The run must keep advancing
    /// generations.
    #[test]
    #[ignore = "requires the GPU"]
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
    #[ignore = "requires the GPU"]
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

    /// The ring absorbs blocks in a fixed order, so two runs of one
    /// seed on one GPU agree in every generation's statistics.
    #[test]
    #[ignore = "requires the GPU"]
    fn two_continuous_runs_of_one_seed_agree() {
        let run = || {
            let gpu = Gpu::new("RTX 4060").unwrap();
            let worker = Worker::spawn(gpu, eframe::egui::Context::default());
            let cfg = Config {
                population: 200_000,
                duration: 10.0,
                seed: 38,
                random_seed: false,
                checkpoint_interval: 0,
                ..Config::default()
            };
            worker.send(Command::New(cfg));
            worker.send(Command::Run {
                continuous: true,
                guided: false,
            });
            let deadline = Instant::now() + Duration::from_secs(600);
            let history = loop {
                if let Some(snapshot) = worker.view.lock().unwrap().take() {
                    assert!(snapshot.error.is_none(), "{:?}", snapshot.error);
                    if snapshot.history.len() >= 12 {
                        break snapshot.history.clone();
                    }
                }
                assert!(Instant::now() < deadline, "the run did not advance");
                std::thread::sleep(Duration::from_millis(20));
            };
            worker.send(Command::Shutdown);
            history[..12]
                .iter()
                .map(|s| {
                    (
                        s.generation,
                        s.best.to_bits(),
                        s.median.to_bits(),
                        s.mean.to_bits(),
                        s.failed,
                        s.archive_cells,
                        s.qd_score.to_bits(),
                        s.percentiles
                            .iter()
                            .map(|p| p.to_bits())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let first = run();
        let second = run();
        assert_eq!(first, second, "two runs of one seed diverged");
    }
}
