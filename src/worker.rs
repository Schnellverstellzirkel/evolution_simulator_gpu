use crate::{
    config::Config,
    evolution::Creature,
    gpu::Gpu,
    qd::{self, Descriptor, Emitter, EmitterStats},
    storage::{self, Experiment, Stats},
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
    /// order. Elites that grew up in the island's nursery count as new
    /// random bodies, whatever emitter bred them last.
    pub origins: [usize; qd::EMITTER_COUNT],
    /// Bodies in the island's nursery now.
    pub nursery: usize,
    /// The nursery's fastest distance (NaN while it is empty).
    pub nursery_best: f32,
    /// What the nursery graduated this session.
    pub graduation: crate::storage::Graduation,
}
/// How many top elites an island summary carries.
pub const ISLAND_TOP: usize = 3;
impl IslandSummary {
    pub fn of(
        island: &qd::QdArchive,
        nursery: &qd::QdArchive,
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
            .map(|elite| (elite.fitness, elite.creature.clone()))
            .collect();
        Self {
            best: top.first().map_or(f32::NAN, |t| t.0),
            cells: island.behavior_count(),
            moves: island.movement_count(),
            leader: top.first().map(|t| t.1.clone()),
            top,
            origins,
            nursery: nursery.behavior_count(),
            nursery_best: if nursery.entries.is_empty() {
                f32::NAN
            } else {
                nursery.best_fitness()
            },
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
    join: Option<std::thread::JoinHandle<()>>,
}
impl Worker {
    pub fn spawn(gpu: Gpu, ctx: eframe::egui::Context) -> Self {
        Self::spawn_with_pause_dir(gpu, ctx, crate::dev_pause::dir())
    }
    /// `spawn`, watching `pause_dir` for developer pause requests.
    pub fn spawn_with_pause_dir(gpu: Gpu, ctx: eframe::egui::Context, pause_dir: PathBuf) -> Self {
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
        let join = std::thread::Builder::new()
            .name("evolution".into())
            .spawn(move || run(gpu, rx, output, paused, bench, ctx, dev))
            .expect("Start simulation worker");
        Self {
            tx,
            view,
            pause,
            measuring,
            breeding,
            dev_pause,
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
    discarded: usize,
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
    fn add(&mut self, stage: usize, seconds: f64) {
        self.seconds[stage] += seconds;
    }
    fn reset(&mut self) {
        self.started = Instant::now();
        self.seconds = [0.0; 3];
        self.starved_block = 0.0;
        self.discarded = 0;
    }
    /// A block was absorbed: the GPU idle time since the last one.
    fn block(&mut self, idle: f64) {
        self.starved_block = self.starved_block.max(idle - self.idle_at_block);
        self.idle_at_block = idle;
    }
    fn write_row(
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
struct RingMeter {
    /// (generation, seconds) of blocks that did not end a generation.
    chains: std::collections::VecDeque<(u32, f64)>,
    /// (generation, seconds) of the blocks that ended one.
    boundaries: std::collections::VecDeque<(u32, f64)>,
}
impl RingMeter {
    const GENERATIONS: u32 = 3;
    fn clear(&mut self) {
        self.chains.clear();
        self.boundaries.clear();
    }
    /// A block of `generation` took `seconds`; `boundary` when it ended it.
    fn add(&mut self, generation: u32, seconds: f64, boundary: bool) {
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
    /// For the stage log: this generation's p95 and boundary, 0 if none.
    fn chain_p95_of(&self, generation: u32) -> f64 {
        Self::p95(
            self.chains
                .iter()
                .filter(|c| c.0 == generation)
                .map(|c| c.1),
        )
        .unwrap_or(0.0)
    }
    fn boundary_of(&self, generation: u32) -> f64 {
        self.boundaries
            .iter()
            .filter(|c| c.0 == generation)
            .map(|c| c.1)
            .fold(0.0, f64::max)
    }
    /// What the next ring is sized from: the scheduler's rate, and the
    /// measured host times when this session has run a generation.
    fn times(&self, sched: Option<&crate::scheduler::Scheduler>) -> crate::storage::RingTimes {
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
/// A save loading on its own thread.
struct Loading {
    path: std::path::PathBuf,
    progress: Arc<storage::Progress>,
    handle: std::thread::JoinHandle<anyhow::Result<Experiment>>,
    started: Instant,
}

/// What the worker shares with the UI for the native benchmark.
struct Bench {
    measuring: Arc<AtomicBool>,
    breeding: Arc<Mutex<Vec<(Instant, Instant, bool)>>>,
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
fn run(
    mut gpu: Gpu,
    rx: Receiver<Command>,
    output: Arc<Mutex<Option<Snapshot>>>,
    pause: Arc<AtomicBool>,
    bench: Bench,
    ctx: eframe::egui::Context,
    mut dev: crate::dev_pause::DevPause,
) {
    // The Rayon pool must exist before this thread is pinned: a pool built
    // lazily by a pinned thread inherits its single CPU, and all of breeding
    // and archiving then runs on one core. The game builds its pool in `main`
    // (`threads::pool_threads`), so this is a no-op there. Examples and tests
    // that spawn a worker without one get the default pool, built on this
    // thread's full CPU set.
    let _ = rayon::current_num_threads();
    crate::threads::pin_worker();
    let helper = crate::threads::Helper::new("search");
    let measuring = bench.measuring.clone();
    let mut exp: Option<Experiment> = None;
    let mut running = false;
    // The UI asked for the ranked archive (`Command::Cards`).
    let mut send_cards = false;
    let mut preview = None;
    // The live champion sent with snapshots, and the (epoch, id) it is for.
    let mut champion: Option<Arc<(Creature, Config)>> = None;
    let mut champion_key: Option<(u64, u64)> = None;
    let mut lineage: Option<(u64, Vec<LineageStep>)> = None;
    let mut status = "Create a population to begin".to_owned();
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
    // Milliseconds each snapshot took to build, for the benchmark report.
    let mut snapshot_build_ms: Vec<f64> = Vec::new();
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
    // Send and read time of every ping.
    let mut benchmark_pings: Vec<(Instant, Instant)> = Vec::new();
    let mut benchmark_configure_ms: Vec<f64> = Vec::new();
    let mut stage_log = StageLog::open();
    // The creatures in flight, and the generation a run of one generation
    // stops at.
    let mut ring = crate::ring::Ring::default();
    let mut ring_meter = RingMeter::default();
    let mut run_until: Option<u32> = None;
    // Completion time and population of recent generations.
    let mut generation_marks: std::collections::VecDeque<(Instant, usize)> = Default::default();
    'worker: loop {
        // A developer pause with the engines closed leaves nothing to
        // collect: wait for commands instead of spinning.
        let first = if (running || gpu.on_engines() > 0) && !dev.idle() {
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
            // No command waits for the engines: a save holds only the
            // archives, settings apply to the blocks bred after them, and a
            // new or loaded game drops the ring.
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
                        if let Some(sched) = gpu.sched.as_mut() {
                            ring.stop(sched);
                        }
                        status = "Creating population…".into();
                        // The ring is sized once, for the whole game, from
                        // what this session measured.
                        let shape =
                            crate::storage::RingShape::size(&ring_meter.times(gpu.sched.as_ref()));
                        let next = Experiment::with_ring(cfg, shape)?;
                        ring_meter.clear();
                        preview =
                            Some((next.blocks[0].population.creature(0), next.config.clone()));
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
                        // A run of one generation stops at its end.
                        run_until = exp.as_ref().filter(|_| !c || g).map(|e| e.generation + 1);
                        pause.store(false, Ordering::Relaxed);
                        running = true;
                        if let Some(log) = &mut stage_log {
                            log.reset();
                        }
                    }
                    Command::Pause => {
                        running = false;
                        status = "Paused".into();
                    }
                    Command::Configure(cfg) => {
                        if let Some(e) = &mut exp {
                            let before = e.config.clone();
                            // The change applies now. Blocks already run or
                            // running in the old world enter no archive;
                            // blocks not yet on an engine run in the new one.
                            e.update_config_now(cfg)?;
                            if before.physics_differs(&e.config)
                                && let Some(sched) = gpu.sched.as_mut()
                            {
                                let lost = ring.world_changed(e, sched);
                                if let Some(log) = &mut stage_log {
                                    log.discarded += lost;
                                }
                            }
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
                            e.update_config_now(cfg)?;
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
                        if let Some(sched) = gpu.sched.as_mut() {
                            ring.stop(sched);
                        }
                        running = false;
                        ring_meter.clear();
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
                                    crate::threads::pin_pool();
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
                            benchmark_pings.push((sent, Instant::now()));
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
                            selected = Some(elite.replay_of(&e.config));
                        }
                    }
                    Command::Lineage(id) => {
                        if let Some(e) = &exp {
                            let chain = e.ancestry(id, crate::storage::ANCESTRY_DEPTH);
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
                        // one, whatever interval the checkpoint carried. An
                        // unattended run (`EVOLUTION_AUTOSTART`) autosaves,
                        // as an unattended new game does.
                        next.config.checkpoint_interval =
                            if std::env::var_os("EVOLUTION_AUTOSTART").is_some() {
                                crate::ui::AUTOSAVE_INTERVAL
                            } else {
                                0
                            };
                        let creature = next
                            .archive
                            .entries
                            .iter()
                            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
                            .map_or_else(
                                || next.blocks[0].population.creature(0),
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
        if let Some(text) = dev.tick(gpu.sched.as_mut()) {
            let generation = exp.as_ref().map_or(0, |e| e.generation);
            log_event(&mut events, generation, EventKind::Gpu, text);
            changed = true;
        }
        if let Some(e) = &mut exp
            && let Some(sched) = gpu.sched.as_mut()
            && (running || ring.active())
        {
            let result: anyhow::Result<()> = (|| {
                if !running {
                    // A pause stops new submissions and absorption; work on
                    // the engines still completes and waits in the ring.
                    ring.step(e, sched, Duration::from_millis(4), 0)?;
                    return Ok(());
                }
                let pass_started = Instant::now();
                let world_before = e.config.clone();
                let kept_before = e.archive.entries.len();
                if !ring.active() {
                    ring.start(e, sched);
                }
                sched.pump()?;
                // Blocks are absorbed in ring order, so the run does not
                // depend on which unit finished first, nor on when commands
                // are read.
                let pass = search_pass(&helper, e, sched, &mut ring, &rx, &mut deferred)?;
                let step = pass.step;
                for note in &pass.blocks {
                    ring_meter.add(note.generation, note.chain, note.boundary);
                    if let Some(log) = &mut stage_log {
                        log.block(note.idle);
                    }
                }
                if measuring.load(Ordering::Relaxed) {
                    benchmark_pings.extend(&pass.pings);
                    bench.breeding.lock().unwrap().extend(&pass.breeding);
                }
                let seconds = pass_started.elapsed().as_secs_f64();
                e.evaluation_seconds += seconds;
                let [archive, breeding] = std::mem::take(&mut e.stage_seconds);
                if benchmark_start.is_some() {
                    benchmark_stage_seconds[0] += (seconds - archive - breeding).max(0.0);
                    benchmark_stage_seconds[1] += archive;
                    benchmark_stage_seconds[2] += breeding;
                }
                if let Some(log) = &mut stage_log {
                    log.add(0, (seconds - archive - breeding).max(0.0));
                    log.add(1, archive);
                    log.add(2, breeding);
                }
                status = format!("Evolving · generation {}", e.generation);
                // A developer's generation dump (EVOLUTION_DUMP_GENERATION)
                // says where it went.
                if let Some(text) = e.dump_notice.take() {
                    log_event(&mut events, e.generation, EventKind::Saved, text);
                }
                if step.generations == 0 {
                    return Ok(());
                }
                if e.config.physics_differs(&world_before) {
                    // Autochange changed the world at the boundary: blocks
                    // not yet on an engine run in the new world, and the
                    // kept elites are tested again in it.
                    let lost = ring.world_changed(e, sched);
                    if let Some(log) = &mut stage_log {
                        log.discarded += lost;
                    }
                    log_world_change(
                        &mut events,
                        &world_before,
                        &e.config,
                        e.generation,
                        kept_before.min(e.config.population),
                    );
                }
                if let Some(log) = &mut stage_log {
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
                        &ring_meter,
                        e.rungs.last(),
                    );
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
                    report_benchmark(
                        e,
                        sched,
                        e.generation - first,
                        started.elapsed().as_secs_f64(),
                        benchmark_warmup,
                        benchmark_stage_seconds,
                        &benchmark_generation_seconds,
                        &snapshot_build_ms,
                        &benchmark_configure_ms,
                        &benchmark_pings,
                        &bench.breeding.lock().unwrap(),
                    );
                    running = false;
                    // Developer benchmarks: EVOLUTION_BENCH_SAVE keeps the
                    // evolved game for later runs.
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
                }
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
                    let path = PathBuf::from(format!("runs/seed-{}-auto.evo", e.config.seed));
                    // The ring is shared, not copied: the save holds only
                    // the archives and the search state.
                    let snapshot = e.clone();
                    checkpoint_thread = Some(std::thread::spawn(move || {
                        crate::threads::pin_pool();
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
                if run_until.is_some_and(|until| e.generation >= until) {
                    running = false;
                    status = format!("Paused after generation {}", e.generation - 1);
                }
                Ok(())
            })();
            if let Err(err) = result {
                error = Some(format!("{err:#}"));
                running = false;
            }
            changed = true;
        } else if running && exp.is_none() {
            running = false;
        }
        // A GPU that was lost and reopened is told to the player.
        if let Some(sched) = gpu.sched.as_mut() {
            for notice in sched.take_notices() {
                let generation = exp.as_ref().map_or(0, |e| e.generation);
                log_event(&mut events, generation, EventKind::Gpu, notice);
                changed = true;
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
            let build_started = Instant::now();
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
                // The best elite by distance, the first one on a tie. Only a
                // new record clones a creature.
                let best = e
                    .archive
                    .entries
                    .iter()
                    .filter(|elite| elite.fitness.is_finite())
                    .reduce(|a, b| if b.fitness > a.fitness { b } else { a });
                let key = best.map(|elite| (epoch, elite.creature.id));
                if key != champion_key {
                    champion_key = key;
                    champion = best.map(|elite| Arc::new(elite.replay_of(&e.config)));
                }
                let live_best = best.map_or(f32::NAN, |elite| elite.fitness.max(0.0));
                let live_median = {
                    let mut kept: Vec<f32> = e
                        .archive
                        .entries
                        .iter()
                        .filter(|elite| !qd::is_morphology_niche(&elite.niche))
                        .map(|elite| elite.fitness)
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
                    epoch,
                    config: e.config.clone(),
                    pending: e.pending.clone(),
                    fossils: e.fossils.len(),
                    generation: e.generation,
                    evaluated: e.evaluated,
                    completed: e.evaluated,
                    checking: ring.confirming(),
                    running,
                    history: history.clone(),
                    events: events.clone(),
                    map: map.clone(),
                    selected: selected.take(),
                    cards,
                    preview: preview.take(),
                    champion: champion.clone(),
                    live_best,
                    live_median,
                    lineage: lineage.take(),
                    gpu: gpu.names(),
                    engines: engine_rows(&gpu),
                    end_to_end: end_to_end_rate(&generation_marks),
                    gpu_bytes: gpu.allocated_bytes,
                    ram_bytes: e.ring_bytes()
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
                    movement_cells: e.archive.movement_count(),
                    archive_size: archive_count,
                    innovation_reserve_count: e.archive.morphology_count(),
                    qd_score: e.archive.qd_score,
                    emitters: e.emitter_stats,
                    emitter_weights: qd::emitter_weights(&e.emitter_stats),
                    islands: (0..crate::storage::island_count())
                        .filter_map(|i| {
                            Some(IslandSummary::of(
                                e.islands.get(i)?,
                                e.islands.get(crate::storage::nursery_of(i))?,
                                e.graduations.get(i).copied().unwrap_or_default(),
                            ))
                        })
                        .collect(),
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
                    running: false,
                    history: history.clone(),
                    events: events.clone(),
                    map: None,
                    selected: selected.take(),
                    cards: None,
                    preview: None,
                    champion: None,
                    live_best: f32::NAN,
                    live_median: f32::NAN,
                    lineage: None,
                    gpu: gpu.names(),
                    engines: engine_rows(&gpu),
                    end_to_end: end_to_end_rate(&generation_marks),
                    gpu_bytes: gpu.allocated_bytes,
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
                    migration: None,
                    status: status.clone(),
                    error: error.clone(),
                }
            };
            snapshot_build_ms.push(build_started.elapsed().as_secs_f64() * 1e3);
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
/// Prints the native generation benchmark (`EVOLUTION_BENCH_GENERATIONS`).
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
