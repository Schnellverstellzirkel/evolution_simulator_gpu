//! The worker thread owns the game and the evaluation engines. It reads
//! `Command`s from the UI, runs the search and publishes a `Snapshot` for the
//! UI to draw. This file holds those two messages, the summaries a snapshot
//! carries and the state of the worker's loop. The submodules hold the parts
//! of the loop: commands, evolution steps, loading and saving, autosave,
//! snapshots, the benchmark and the stage log.

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
mod evolve;
mod files;
mod publish;
mod stage_log;
use autosave::Autosave;
use benchmark::{Bench, Benchmark};
use files::Loading;
use stage_log::{RingMeter, StageLog};
/// A request from the UI to the worker thread, sent with `Worker::send`. While
/// a save loads, the worker holds every command back until the load is done,
/// except `New`, `Load`, `Ping` and `Shutdown`.
pub enum Command {
    /// Start a new game with these settings. It replaces the current game and
    /// stops the run.
    New(Config),
    /// Evolve. A run that is not `continuous`, or is `guided`, stops after
    /// one generation.
    Run {
        /// Keep evolving generation after generation.
        continuous: bool,
        /// Stop after one generation, whatever `continuous` says.
        guided: bool,
    },
    /// Stop evolving. Work already on the engines finishes and waits in the
    /// ring.
    Pause,
    /// Apply these settings now. A change of the world empties the global
    /// archive and the main islands. The elites of the main islands are tested
    /// again in the new world.
    Configure(Config),
    /// Wipe out about half of every archive's elites at random, but spare the
    /// fastest elite of each of the best body plans. The lost elites are kept
    /// as fossils for undo.
    Meteor,
    /// Wipe out the island whose best creature is slowest. Its elites are kept
    /// as fossils for undo.
    Extinction,
    /// Return the fossils of earlier meteors and extinctions to their
    /// archives, where their cells are empty or hold slower elites.
    UndoMeteor,
    /// Save the game to this file.
    Save(PathBuf),
    /// Open the save in this file in place of the current game.
    Load(PathBuf),
    /// Write the history to this file as CSV.
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
    /// End the worker thread. Dropping a `Worker` sends it.
    Shutdown,
}
/// A key for a creature's body plan: its counts of nodes, bones and muscles
/// and which parts connect to which. Lengths, masses and rhythms stay out,
/// so a small mutation keeps the plan. The sums do not depend on part order.
/// The UI names species from this key. The archives use another one,
/// `qd::Topology::plan_key`.
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
    /// The elites, best first.
    pub cards: Arc<Vec<Card>>,
}
/// One ranked elite from the archive, with its metadata.
#[derive(Clone)]
pub struct Card {
    /// Where the elite sat in the archive's entries when the list was made.
    pub index: usize,
    /// Its place in the ranking by distance, 0 for the best.
    pub rank: usize,
    /// Its distance.
    pub score: f32,
    /// Its parent's distance, which the card shows when the creature has no
    /// score of its own. The archive list leaves it NaN.
    pub parent_score: f32,
    /// Marks a survivor of selection, whose score the card draws in the accent
    /// color. The archive list sets it to false.
    pub survivor: bool,
    /// Its behavior descriptor, which says how it moves. A card without one
    /// shows the creature's id in place of its rank.
    pub descriptor: Option<Descriptor>,
    /// The emitter that bred it.
    pub emitter: Option<Emitter>,
    /// How many times the search chose it as a parent.
    pub visits: u64,
    /// It sits in the archive's morphology reserve, which the card marks as a
    /// new body.
    pub innovation_reserve: bool,
    /// The creature itself.
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
    /// What sort of event it is.
    pub kind: EventKind,
    /// What the feed says about it.
    pub text: String,
}
/// What sort of thing an `Event` records.
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
    /// The experiment was saved, by hand or by autosave. The report of a
    /// developer's generation dump (`EVOLUTION_DUMP_GENERATION`) has this kind
    /// too.
    Saved,
    /// The GPU failed and was opened again, or could not be. A developer pause
    /// that closes or reopens the engines is logged with this kind too.
    Gpu,
}
/// Events kept per experiment. Older ones drop off.
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
/// Several effects are joined with commas. The autochange level is left out,
/// and the result is `None` when no effect differs.
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
/// `retesting` is how many kept creatures run again in the new world. The event
/// is an autochange event when autochange is on and its step moved, and a world
/// event otherwise.
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
/// archive change. A click asks for the body with `Command::Select`.
#[derive(Clone, Copy)]
pub struct MapCell {
    /// The six bins of the cell, in this order: ground contact, cadence, body
    /// shape, height, feet and body size.
    pub niche: [u8; 6],
    /// The elite's distance.
    pub score: f32,
    /// Place in the archive ranking by distance.
    pub rank: usize,
    /// The elite's creature id, which `Command::Select` takes.
    pub id: u64,
}
/// One ancestor of a selected creature.
#[derive(Clone)]
pub struct LineageStep {
    /// The generation in which it entered an archive.
    pub generation: u32,
    /// Its distance.
    pub fitness: f32,
    /// Fitness gained over this ancestor's own parent. It is 0 for the oldest
    /// ancestor in the chain.
    pub gain: f32,
    /// How it differs from its parent, in words.
    pub change: String,
    /// The ancestor itself.
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
    /// Summarizes `island` and its two nurseries. `graduation` is what the
    /// nurseries graduated this session, the two together.
    pub fn of(
        island: &qd::QdArchive,
        nurseries: [&qd::QdArchive; 2],
        graduation: crate::storage::Graduation,
    ) -> Self {
        let mut origins = [0; qd::EMITTER_COUNT];
        // The fastest `ISLAND_TOP` in one pass, the earlier elite first on a
        // tie, as a stable sort by distance would give them.
        let mut ranked: Vec<&qd::Elite> = Vec::with_capacity(ISLAND_TOP + 1);
        for elite in island
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
        {
            let origin = if elite.graduate {
                qd::Emitter::Restart
            } else {
                elite.emitter
            };
            origins[origin.index()] += 1;
            let at = ranked
                .iter()
                .position(|kept| elite.fitness.total_cmp(&kept.fitness).is_gt())
                .unwrap_or(ranked.len());
            if at < ISLAND_TOP {
                ranked.insert(at, elite);
                ranked.truncate(ISLAND_TOP);
            }
        }
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
/// at and, per island, the elites it sent and how many the hub kept. The hub's
/// own entry is zero. A wild island's copies count as sent and never as kept,
/// because they run in the hub's world first. `Snapshot::wild_wins` counts the
/// ones that take a hub cell.
#[derive(Clone, Debug, PartialEq)]
pub struct MigrationSummary {
    /// The generation the migration happened at.
    pub generation: u32,
    /// One `(sent, kept)` pair per island, in island order.
    pub exchange: Vec<(usize, usize)>,
}
impl MigrationSummary {
    /// Elites the hub received from the other islands, and how many it kept.
    /// These are the sums over `exchange`.
    pub fn hub_received(&self) -> (usize, usize) {
        self.exchange
            .iter()
            .fold((0, 0), |(sent, kept), &(s, k)| (sent + s, kept + k))
    }
}
/// What the worker publishes for the UI to draw. It holds the game's numbers
/// and a summary of each island, and it carries once what the UI asked for
/// with a command. The worker builds a new one when something changed, at most
/// five times a second while the game runs, and leaves it in `Worker::view`.
#[derive(Clone)]
pub struct Snapshot {
    /// Counts the games this worker has started or loaded. A change tells the
    /// UI that the history and the event log began again.
    pub epoch: u64,
    /// The settings of the game now, with its world. They are the defaults
    /// while there is no game.
    pub config: Config,
    /// Settings that wait for the next generation to start.
    /// `Command::Configure` applies at once, so only a loaded save can carry
    /// some.
    pub pending: Option<Config>,
    /// Elites lost to a meteor strike or an extinction that an undo could
    /// bring back.
    pub fossils: usize,
    /// The generation running now.
    pub generation: u32,
    /// Evaluations absorbed toward the current generation.
    pub evaluated: usize,
    /// The same count as `evaluated`.
    pub completed: usize,
    /// Confirmation trials running now, for creatures that would set a record
    /// of an island or a nursery.
    pub checking: usize,
    /// Whether the game is evolving.
    pub running: bool,
    /// One row of statistics per finished generation, oldest first.
    pub history: Arc<Vec<Stats>>,
    /// The archive ranked by distance, sent once per `Command::Cards`.
    pub cards: Option<CardList>,
    /// A creature to show when a game starts or a save opens, with its world.
    /// Sent once.
    pub preview: Option<(Creature, Config)>,
    /// The best elite of the global archive now, and the world it is scored
    /// in. It changes as soon as a new record is absorbed, mid-generation
    /// too, so the world view can switch to it at once.
    pub champion: Option<Arc<(Creature, Config)>>,
    /// The best distance in the global archive now (NaN before any elite). A
    /// history row keeps the same number at the end of a generation.
    pub live_best: f32,
    /// The median distance of the best elite of each way of moving (NaN before
    /// any elite). A history row keeps the same number at the end of a
    /// generation.
    pub live_median: f32,
    /// What happened to this experiment, oldest first.
    pub events: Arc<Vec<Event>>,
    /// The archive map table while the UI asks for it.
    pub map: Option<Arc<Vec<MapCell>>>,
    /// A creature the UI asked for with `Command::Select`, and the world it
    /// is scored in. Sent once.
    pub selected: Option<(Creature, Config)>,
    /// Ancestor chain of a requested creature (its id first), newest first.
    /// Sent once per request.
    pub lineage: Option<(u64, Vec<LineageStep>)>,
    /// The names of the evaluation devices, joined with " + ".
    pub gpu: String,
    /// Evaluation engines: name, measured creatures/s, creatures evaluated.
    pub engines: Vec<(String, f64, u64)>,
    /// Creatures per second over complete generations in the last ~10 s,
    /// including archive updates, breeding, and transfers.
    pub end_to_end: f64,
    /// Bytes allocated on the GPUs, as `Gpu::allocated_bytes` has them. The
    /// worker refreshes that value on every pass (`Gpu::refresh_allocated_bytes`).
    /// Each engine updates its share after a submission, and a closed engine
    /// holds none.
    pub gpu_bytes: u64,
    /// Bytes the ring and the creatures of the global archive hold in memory.
    pub ram_bytes: usize,
    /// Seconds the current generation has been evolving.
    pub elapsed: f64,
    /// Behavior elites in the global archive, one for each filled cell.
    pub archive_cells: usize,
    /// Ways of moving the global archive covers, counting its cells
    /// without their body classes.
    pub movement_cells: usize,
    /// Elites in the global archive, the morphology reserve included.
    pub archive_size: usize,
    /// Elites in the global archive's morphology reserve.
    pub innovation_reserve_count: usize,
    /// The global archive's QD score, the sum of its elites' distances.
    pub qd_score: f64,
    /// What each emitter has done so far, in `Emitter::ALL` order.
    pub emitters: [EmitterStats; 4],
    /// The mix that picks the emitter of each child, from the emitters'
    /// results so far, in `Emitter::ALL` order. The shares add up to 1.
    pub emitter_weights: [f64; 4],
    /// Each island archive, in island order.
    pub islands: Vec<IslandSummary>,
    /// The last island migration this session, if one happened.
    pub migration: Option<MigrationSummary>,
    /// Per island, how many of its wild migrants took a hub cell. It stays
    /// empty until the first one does.
    pub wild_wins: Vec<u32>,
    /// The main islands' elite with the body farthest from the others.
    pub strangest: Option<Creature>,
    /// What the worker is doing, in words for the status line.
    pub status: String,
    /// The last error, shown to the player: a command, a step of evolution, a
    /// load or a save that failed. The next command clears it.
    pub error: Option<String>,
}
/// The UI's handle on the worker thread, which evolves the search and
/// publishes snapshots. Dropping it stops the thread and waits for it to end.
pub struct Worker {
    /// The channel to the worker thread. `send` writes to it.
    pub tx: Sender<Command>,
    /// The newest snapshot. The worker replaces it each time it publishes, and
    /// the UI takes it, so it holds `None` until the next one.
    pub view: Arc<Mutex<Option<Snapshot>>>,
    /// True while the UI wants the game paused. The worker stops running while
    /// it is set, even before it has read `Command::Pause`. `Command::Run`
    /// clears it, and dropping the `Worker` sets it.
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
    /// Starts the worker thread with `gpu` open already. `ctx` is the window's
    /// context, which the worker asks to repaint after each snapshot. The
    /// thread watches the game's developer pause directory (`dev_pause::dir`).
    pub fn spawn(gpu: Gpu, ctx: eframe::egui::Context) -> Self {
        Self::spawn_with_pause_dir(gpu, ctx, crate::dev_pause::dir())
    }
    /// `spawn`, watching `pause_dir` for developer pause requests.
    pub fn spawn_with_pause_dir(gpu: Gpu, ctx: eframe::egui::Context, pause_dir: PathBuf) -> Self {
        Self::start(ctx, pause_dir, move || Ok(gpu))
    }
    /// Opens the evaluation devices on the worker's own thread, so the window
    /// can draw its loading screen while they open. `primary` names the
    /// primary GPU (`Gpu::new`). Commands sent before then wait in the
    /// channel.
    pub fn open(primary: String, ctx: eframe::egui::Context) -> Self {
        Self::start(ctx, crate::dev_pause::dir(), move || Gpu::new(&primary))
    }
    /// Starts the worker thread, which calls `open` to get its `Gpu`. It sets
    /// `opened` when `open` returns. Then it runs the loop, or leaves the
    /// reason in `failed` when the devices did not open.
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
    /// Sends a command to the worker. It does nothing once the worker has
    /// ended.
    pub fn send(&self, c: Command) {
        let _ = self.tx.send(c);
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.pause.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Command::Shutdown);
        // Wait for the thread, so nothing is still using the GPU or writing a
        // save when the window closes.
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
/// The worker thread's state, kept in one place for the methods of the
/// submodules. Fields drop in declaration order. The game and the helper
/// thread go before the developer pause, and `gpu`, which closes the engines,
/// goes last.
struct Loop {
    // What the loop measures.
    /// Completion time and population of recent generations.
    generation_marks: std::collections::VecDeque<(Instant, usize)>,
    ring_meter: RingMeter,
    stage_log: Option<StageLog>,
    benchmark: Benchmark,
    // The run.
    /// The generation a run of one generation stops at.
    run_until: Option<u32>,
    /// The creatures in flight.
    ring: crate::ring::Ring,
    autosave: Autosave,
    // Loading and saving.
    /// When the status last showed the progress of a load.
    last_progress: Instant,
    /// Commands held back until a load is done.
    deferred: Vec<Command>,
    /// A save waiting until the "Saving" status has reached the window.
    pending_save: Option<std::path::PathBuf>,
    /// A save being loaded on its own thread.
    loading: Option<Loading>,
    // What the next snapshot shows.
    /// The creature `Command::Select` found, until a snapshot carries it.
    selected: Option<(Creature, Config)>,
    /// The archive state the map table was built from.
    map_key: (u64, usize, u64),
    /// The last archive map table built.
    map: Option<Arc<Vec<MapCell>>>,
    /// Whether the UI wants the archive map table.
    want_map: bool,
    events: Arc<Vec<Event>>,
    /// The game's history, shared with the snapshots and copied again when its
    /// length changes.
    history: Arc<Vec<Stats>>,
    /// Counts the games started or loaded (`Snapshot::epoch`).
    epoch: u64,
    /// Something changed since the last snapshot.
    changed: bool,
    last_publish: Instant,
    error: Option<String>,
    status: String,
    /// The ancestors `Command::Lineage` traced, until a snapshot carries them.
    lineage: Option<(u64, Vec<LineageStep>)>,
    /// The (epoch, id) the live champion is for.
    champion_key: Option<(u64, u64)>,
    /// The live champion sent with snapshots.
    champion: Option<Arc<(Creature, Config)>>,
    /// The creature to show for a new or loaded game, until a snapshot
    /// carries it.
    preview: Option<(Creature, Config)>,
    /// The UI asked for the ranked archive (`Command::Cards`).
    send_cards: bool,
    // The game.
    running: bool,
    exp: Option<Experiment>,
    // The thread's helper, its link to the window and the engines.
    helper: crate::threads::Helper,
    dev: crate::dev_pause::DevPause,
    ctx: eframe::egui::Context,
    /// The UI's pause flag (`Worker::pause`).
    pause: Arc<AtomicBool>,
    /// Where snapshots go (`Worker::view`).
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
            ring_meter: RingMeter::default(),
            stage_log: StageLog::open(),
            benchmark: Benchmark::new(bench),
            run_until: None,
            ring: crate::ring::Ring::default(),
            autosave: Autosave::default(),
            last_progress: Instant::now(),
            deferred: Vec::new(),
            pending_save: None,
            loading: None,
            selected: None,
            map_key: (u64::MAX, usize::MAX, 0u64),
            map: None,
            want_map: false,
            events: Arc::new(Vec::new()),
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
    /// One pass of the worker's loop. It reads the commands, checks a load in
    /// progress and the pause flags, takes one step of evolution, logs GPU and
    /// autosave notices, refreshes the GPU byte count, publishes a snapshot
    /// when one is due and runs a requested save. `Break` ends the loop.
    fn pass(&mut self) -> ControlFlow<()> {
        let first = self.next_command()?;
        self.handle_commands(first)?;
        self.poll_load();
        if self.pause.load(Ordering::Relaxed) {
            self.running = false;
        }
        self.tick_dev_pause();
        self.step();
        self.log_gpu_notices();
        self.log_autosave();
        self.gpu.refresh_allocated_bytes();
        self.publish_if_due();
        self.save_pending();
        ControlFlow::Continue(())
    }
    /// The loop has ended: waits for a running autosave.
    fn finish(mut self) {
        self.autosave.join();
    }
}
/// The body of the worker thread. It sets the thread up, runs `Loop::pass`
/// until it breaks, and then lets `Loop::finish` wait for a running autosave.
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

/// Records a generation's end for `end_to_end_rate`, and forgets the ends
/// older than 10 s (keeping at least two).
fn note_generation(marks: &mut std::collections::VecDeque<(Instant, usize)>, population: usize) {
    marks.push_back((Instant::now(), population));
    while marks.len() > 2 && marks[0].0.elapsed() > Duration::from_secs(10) {
        marks.pop_front();
    }
}
/// Creatures per second between the oldest and newest recent generation ends:
/// the population of every generation after the oldest, over the time from the
/// oldest end to the newest. It is 0 with fewer than two ends.
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

    /// An island summary's fastest elites are the first of a stable sort by
    /// distance, ties included.
    #[test]
    fn an_island_summary_ranks_its_fastest_elites_as_a_stable_sort() {
        let config = Config {
            population: 200,
            random_seed: false,
            seed: 9,
            ..Config::default()
        };
        let population = crate::evolution::create(&config).unwrap();
        let mut island = qd::QdArchive::default();
        for index in 0..200usize {
            let descriptor = qd::Descriptor {
                ground_contact: (index % 10) as f32 / 10.0,
                gait_frequency: (index / 10 % 6) as f32,
                vertical_oscillation: 0.1,
                mean_height: 0.4,
                feet: 2.0,
                ..Default::default()
            };
            // Few distinct distances, so the top holds ties.
            let fitness = 1.0 + (index * 7 % 5) as f32;
            island.offer(
                &population,
                index,
                descriptor,
                fitness,
                false,
                qd::Emitter::Cma,
                0,
                0,
            );
        }
        let empty = qd::QdArchive::default();
        let summary = IslandSummary::of(&island, [&empty, &empty], Default::default());
        let mut sorted: Vec<&qd::Elite> = island
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .collect();
        sorted.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let expected: Vec<(f32, u64)> = sorted
            .iter()
            .take(ISLAND_TOP)
            .map(|elite| (elite.fitness, elite.creature.id))
            .collect();
        let got: Vec<(f32, u64)> = summary.top.iter().map(|(f, c)| (*f, c.id)).collect();
        assert!(island.behavior_count() > ISLAND_TOP);
        assert_eq!(got, expected);
    }

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
