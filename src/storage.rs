use crate::{
    config::Config,
    evolution::{self, CandidatePlan, Creature, FAILED, Population, Rng, StoredCreature},
    qd::{self, CmaEmitter, Emitter, EmitterStats, EvaluationMetrics, QdArchive},
};
use anyhow::{Context, Result, ensure};
use bincode::Options;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::Arc,
};

mod confirm;
mod lineage;
mod ring_shape;

use confirm::{ConfirmHint, ScreenWindow};
use lineage::ISLAND_LEADERS;
pub use lineage::{ANCESTRY_DEPTH, Ancestor};
pub use ring_shape::{RingShape, RingTimes};

/// Nanoseconds of breeding spent planning, emitting offspring, and writing
/// them into their block, since the last `take_breed_nanos`.
pub static BREED_NANOS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Returns and clears the breeding timers.
pub fn take_breed_nanos() -> [u64; 3] {
    std::array::from_fn(|i| BREED_NANOS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}
/// Children written after their part of the block's arena (low 32 bits)
/// and blocks bred into a new arena because the old one was still shared
/// (high 32 bits), since the last `take_breed_late`.
pub static BREED_LATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Returns and clears `BREED_LATE` as (late children, new arenas).
pub fn take_breed_late() -> (u64, u64) {
    let v = BREED_LATE.swap(0, std::sync::atomic::Ordering::Relaxed);
    (v & 0xffff_ffff, v >> 32)
}

pub const PERCENTILES: [f32; 29] = [
    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 20., 30., 40., 50., 60., 70., 80., 90., 91., 92.,
    93., 94., 95., 96., 97., 98., 99., 100.,
];
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stats {
    pub generation: u32,
    pub best: f32,
    pub median: f32,
    pub worst: f32,
    pub mean: f32,
    pub failed: usize,
    pub seconds: f64,
    pub population: usize,
    pub percentiles: Vec<f32>,
    /// Sparse centimeter bins preserve adjustable historical histograms without storing all scores.
    pub histogram: Vec<(i32, u32)>,
    pub species: Vec<(usize, usize, u32)>,
    pub representatives: Vec<Creature>,
    pub config: Config,
    #[serde(default)]
    pub archive_cells: usize,
    #[serde(default)]
    pub qd_score: f64,
    /// The share of the ways of moving (cells without their body classes)
    /// that an elite covers: `Stats::moves` counts them.
    #[serde(default)]
    pub archive_coverage: f32,
    #[serde(default)]
    pub emitters: [EmitterStats; qd::EMITTER_COUNT],
    /// The ring the generation ran with.
    pub ring: RingShape,
    /// Body plans among the global archive's elites, and their effective
    /// number of clades (Hill number of order 1: exp of the Shannon entropy
    /// of clade sizes; Hill 1973, Jost 2006).
    #[serde(default)]
    pub plans: usize,
    #[serde(default)]
    pub clades: f32,
    /// The median age, in generations, of the global archive's body plans:
    /// how long plans keep their place (their half life).
    #[serde(default)]
    pub plan_age: f32,
}
impl Stats {
    /// The ways of moving the archive covered: its cells, counted without
    /// the body classes. A history saved before the classes held one elite
    /// per way of moving, so its cells are the same count.
    pub fn moves(&self) -> usize {
        (self.archive_coverage * qd::MOVEMENT_CELLS as f32).round() as usize
    }
}
/// How a creature in the ring was bred.
#[derive(Clone, Copy, Debug)]
pub struct Birth {
    pub emitter: Emitter,
    /// The CMA emitter that sampled it.
    pub cma: Option<usize>,
    pub parent_id: Option<u64>,
    /// It came from crossover.
    pub mate: bool,
    /// Generation until which its niche is protected from other body plans.
    pub protection: u32,
}
impl Birth {
    /// A new random body, or an elite queued again after a world change.
    pub const RANDOM: Self = Self {
        emitter: Emitter::Restart,
        cma: None,
        parent_id: None,
        mate: false,
        protection: 0,
    };
}

/// One block of the ring: creatures bred together, evaluated as one unit and
/// absorbed together.
#[derive(Clone)]
pub struct Block {
    /// Ring slot of the first creature: creature `k` holds slot `first + k`,
    /// which picks its island and its random stream.
    pub first: usize,
    pub population: Arc<Population>,
    pub births: Vec<Birth>,
    /// The trial settings the block runs with, fixed when it is bred.
    pub config: Arc<Config>,
}
impl Block {
    /// The screen bar of creature `j` among `screen`'s: a nursery body is
    /// held to the bar of its own kind.
    pub fn screen_bar(&self, screen: &crate::physics::Screen, j: usize) -> f32 {
        let flags = self.population.flags.get(j).copied().unwrap_or(0);
        if flags & crate::rungs::RESHAPED != 0 {
            screen.reshaped_bar
        } else if flags & crate::rungs::YOUNG != 0 {
            screen.young_bar
        } else {
            screen.bar
        }
    }
    /// Which window creature `j`'s distance at the screen belongs to: 0 for
    /// the evolved creatures, 1 for the nursery's new bodies, 2 for its
    /// reshaped bodies.
    fn screen_class(&self, j: usize) -> usize {
        if qd::is_wild(qd::island_of_slot(self.first + j, island_count())) {
            return 3;
        }
        let flags = self.population.flags.get(j).copied().unwrap_or(0);
        if flags & crate::rungs::RESHAPED != 0 {
            2
        } else {
            usize::from(flags & crate::rungs::YOUNG != 0)
        }
    }
    pub fn len(&self) -> usize {
        self.population.genomes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.population.genomes.is_empty()
    }
}

/// What a block needs before it can be absorbed.
#[derive(Clone, Debug)]
pub enum Verdict {
    /// Confirmation trials for these members (positions in the block).
    Confirm(Vec<usize>),
    /// The final result of every member, in block order.
    Final(Vec<EvaluationMetrics>),
}

/// Scores a population with the given trial settings, one result per
/// creature in order. The synchronous drivers (`Experiment::step`) take one.
pub type Evaluate<'a> = dyn FnMut(&Population, &Config) -> Result<Vec<EvaluationMetrics>> + 'a;

/// The game: archives and search state, and the ring of creatures in
/// flight. A generation is a count of `config.population` evaluations; the
/// ring holds at most `ring.block * ring.blocks` creatures, and each block
/// is bred again as soon as it is absorbed.
#[derive(Clone)]
pub struct Experiment {
    pub config: Config,
    /// Settings that take effect when the next generation starts.
    pub pending: Option<Config>,
    pub generation: u32,
    /// Evaluations absorbed toward the current generation.
    pub evaluated: usize,
    pub history: Vec<Stats>,
    pub evaluation_seconds: f64,
    pub archive: QdArchive,
    pub emitter_stats: [EmitterStats; qd::EMITTER_COUNT],
    pub cma_emitters: Vec<CmaEmitter>,
    pub qd_version: u32,
    /// Breeding rounds so far; salts offspring random streams and ids.
    pub breed_round: u64,
    /// Island archives, then one nursery of new random bodies per island,
    /// then one nursery of reshaped bodies per island. Ring slot `i` breeds
    /// for `qd::arena_of_slot(i, arena_count())`; the global `archive`
    /// collects every island's elites for display and statistics and is
    /// never a parent source.
    pub islands: Vec<QdArchive>,
    /// Every creature that entered an archive, keyed by creature id, with its
    /// parent and the change that produced it. Pruned to living elites' ancestors.
    pub lineage: HashMap<u64, Ancestor>,
    /// Each island's best distance so far and the generation it was set.
    pub island_progress: Vec<(f32, u32)>,
    /// Per island, what its nursery of new random bodies graduated this
    /// session, and what its nursery of reshaped bodies did.
    pub graduations: Vec<Graduation>,
    pub reshaped_graduations: Vec<Graduation>,
    /// The last migration to the hub this session: its generation, and per
    /// island how many elites it sent and how many of those the hub kept
    /// (the hub's own entry is zero). Saved after the body of a small save.
    pub last_migration: Option<(u32, Vec<(usize, usize)>)>,
    /// Elites from before an environment change, waiting to be evaluated again
    /// in the new world, each queued for its own island. Breeding hands them
    /// out before new offspring.
    pub reseed: Reseed,
    /// Each island's champions from before the last environment change, and
    /// the generation until which they keep breeding. Not saved.
    pub refuge: Refuge,
    /// Elites a meteor wiped out, with their island (None for the global
    /// archive), kept so the strike can be undone. Not saved.
    pub fossils: Vec<(Option<usize>, qd::Elite)>,
    /// The ring's shape, fixed for the experiment.
    pub ring: RingShape,
    /// The ring. Blocks are absorbed in ring order, starting at `cursor`.
    pub blocks: Vec<Block>,
    pub cursor: usize,
    /// Failed trials in the current generation.
    failed: usize,
    /// Distances at the screen of the newest absorbed blocks, for the bar of
    /// the evolved creatures and, apart, for the bars of the young ones
    /// (`rungs::YOUNG`) and the reshaped ones (`rungs::RESHAPED`).
    screen_window: ScreenWindow,
    young_window: ScreenWindow,
    reshaped_window: ScreenWindow,
    /// How rare the clade of each island elite is (`clade_rarity_of`),
    /// computed once per generation, for the generation it names. Not saved.
    clade_rarity: (u32, Vec<Vec<f32>>),
    /// Wild migrants waiting for their hub trial, by creature id, with their
    /// wild island; and per island, how many of its migrants took a hub
    /// cell this session. Not saved.
    wild_exports: HashMap<u64, usize>,
    pub wild_wins: Vec<u32>,
    /// Wild champions sent to the hub, each with the generation until which
    /// it breeds in the hub's slots. Not saved.
    pub pen: Vec<(Creature, u32)>,
    /// The first elite of each new body plan of the main islands, newest
    /// last, up to `FOUNDERS` (stepping stones, Stanley and Lehman 2015), and
    /// every plan seen so far. Not saved.
    founders: std::collections::VecDeque<Creature>,
    founder_plans: std::collections::HashSet<u64>,
    /// The fastest elite of each body plan of the main islands, rebuilt each
    /// generation; the hub breeds from it (Lehman and Stanley, 2011, an
    /// archive of stepping stones). Not saved.
    hall: Vec<Creature>,
    /// The generation each body plan of the global archive first appeared
    /// in, for `Stats::plan_age`. Not saved.
    plan_born: HashMap<u64, u32>,
    /// The generation each island's archive started its climb: a new game,
    /// or an island that started over. An island is refined once it is
    /// `qd::REFINE_AFTER` generations old. Not saved: a loaded game's
    /// islands count from generation 0, so they are old enough.
    island_epoch: Vec<u32>,
    /// The audit lane and the early rungs it calibrates (`rungs`).
    pub rungs: crate::rungs::Audit,
    /// The last absorbed block's screen bar and the share of its results at
    /// or above it, for the stage log. None when that block ran without a
    /// bar or came from a world that has since changed.
    pub last_screen: Option<(f32, f32)>,
    /// Seconds spent absorbing results into the archives and breeding
    /// blocks again, since the caller last took them.
    pub stage_seconds: [f64; 2],
    /// The generation dump in progress (`EVOLUTION_DUMP_GENERATION`), a
    /// developer diagnostic. Not saved; a clone shares it.
    dump: Option<Arc<std::sync::Mutex<dump::Dump>>>,
    /// What the last finished dump wrote, for the worker's event log.
    pub dump_notice: Option<String>,
    /// How many confirmation trials each archive's verdicts used lately.
    confirm_hint: ConfirmHint,
}

/// Elites waiting to be evaluated again after a world change, one queue per
/// island. Each returns in a slot of its own island, so a world change mixes
/// no island's creatures into another.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reseed {
    queues: Vec<Vec<Creature>>,
}

impl Reseed {
    pub fn len(&self) -> usize {
        self.queues.iter().map(Vec::len).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(Vec::is_empty)
    }
    pub fn clear(&mut self) {
        self.queues.clear();
    }
    /// Queues `creature` for a slot of `island`.
    pub fn push(&mut self, island: usize, creature: Creature) {
        if self.queues.len() <= island {
            self.queues.resize_with(island + 1, Vec::new);
        }
        self.queues[island].push(creature);
    }
    /// The next creature queued for `island`.
    pub fn pop(&mut self, island: usize) -> Option<Creature> {
        self.queues.get_mut(island)?.pop()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Creature> {
        self.queues.iter().flatten()
    }
    /// Whether every queue belongs to an existing island.
    fn fits(&self, islands: usize) -> bool {
        self.queues.len() <= islands
    }
}

/// The champions of each island from before an environment change. A world
/// change can kill every old design at once when the first re-test finds
/// them slow, and random bodies take the islands. For `REFUGE_GENERATIONS`
/// generations a share of each island's slots breed children of its old
/// champions, so an old design gets time to retune its gait to the new world
/// (a refugium, as in island models with migration from a reservoir).
#[derive(Clone, Debug, Default)]
pub struct Refuge {
    champions: Vec<Vec<Creature>>,
    /// The generation the refuge opened, and until which it lasts.
    since: u32,
    until: u32,
    /// Per island: until when its refuge lasts (it ends early when the
    /// island recovered, and lasts longer after a heavy loss), and the
    /// island's best distance before the change.
    island_until: Vec<u32>,
    before: Vec<f32>,
}
/// Generations the refuge breeds only gait retunes, before structural
/// changes join (Cheney et al., 2018: a changed body readapts its control
/// first).
const REFUGE_TUNING: u32 = 3;
/// The refuge of an island that kept this share of its best distance ends
/// after `REFUGE_MIN` generations; one that kept under `REFUGE_HEAVY` lasts
/// `REFUGE_LONG` (Branke, 1999: the memory a changed world needs grows with
/// the loss).
const REFUGE_MIN: u32 = 2;
const REFUGE_RECOVERED: f32 = 0.9;
const REFUGE_HEAVY: f32 = 0.5;
const REFUGE_LONG: u32 = 10;

/// The champions of an archive for the refuge: the fastest elite of each
/// body plan, fastest first, up to `REFUGE_CHAMPIONS` plans (Schluter, 2000:
/// a radiation grows from many founders, not from many copies of one).
fn plan_champions(archive: &QdArchive) -> Vec<Creature> {
    let mut order: Vec<usize> = (0..archive.entries.len())
        .filter(|&i| !qd::is_morphology_niche(&archive.entries[i].niche))
        .collect();
    order.sort_by(|&a, &b| {
        archive.entries[b]
            .fitness
            .total_cmp(&archive.entries[a].fitness)
            .then(a.cmp(&b))
    });
    let mut seen = std::collections::HashSet::new();
    order
        .into_iter()
        .filter(|&i| seen.insert(archive.plan_key(i)))
        .take(REFUGE_CHAMPIONS)
        .map(|i| archive.entries[i].creature.unpack())
        .collect()
}
/// Generations the old champions keep breeding after a world change.
const REFUGE_GENERATIONS: u32 = 5;
/// The best elites of each island that go into the refuge.
const REFUGE_CHAMPIONS: usize = 64;
/// Share of an island's own slots that breed from its refuge.
const REFUGE_SHARE: f32 = 0.15;

impl Refuge {
    /// Opens a refuge at `generation` with these champions, and the islands'
    /// best distances before the change.
    fn open(champions: Vec<Vec<Creature>>, before: Vec<f32>, generation: u32) -> Self {
        let until = generation + REFUGE_GENERATIONS;
        Self {
            island_until: vec![until; champions.len()],
            champions,
            since: generation,
            until: generation + REFUGE_LONG,
            before,
        }
    }
    /// At a generation boundary: an island that has won back most of its
    /// best distance closes its refuge, and one still far below keeps it up
    /// to `REFUGE_LONG` generations.
    fn review(&mut self, islands: &[QdArchive], generation: u32) {
        if generation >= self.until {
            return;
        }
        for (island, until) in self.island_until.iter_mut().enumerate() {
            let before = self.before.get(island).copied().unwrap_or(f32::NAN);
            let Some(archive) = islands.get(island) else {
                continue;
            };
            if !before.is_finite() || before <= 0.0 {
                continue;
            }
            let kept = archive.best_fitness() / before;
            if generation >= self.since + REFUGE_MIN && kept >= REFUGE_RECOVERED {
                *until = (*until).min(generation);
            } else if kept < REFUGE_HEAVY {
                *until = (*until).max(self.since + REFUGE_LONG);
            }
        }
    }
    /// A child of one of `island`'s champions for `slot`, while the refuge
    /// lasts and the draw picks this slot.
    fn child(
        &self,
        island: usize,
        slot: usize,
        cfg: &Config,
        generation: u32,
        round: u64,
    ) -> Option<Creature> {
        if generation >= self.until
            || self.island_until.get(island).is_some_and(|&u| generation >= u)
        {
            return None;
        }
        let champions = self.champions.get(island).filter(|c| !c.is_empty())?;
        let mut rng = evolution::Rng::stream(cfg.seed ^ 0x7265_6675_6765, generation, round, slot);
        if rng.unit() >= REFUGE_SHARE {
            return None;
        }
        let parent = champions[rng.index(champions.len())].clone();
        // The first generations only retune the gait. Then half the children
        // also take a structural mutation.
        let scale = if rng.unit() < 0.1 { 2.0 } else { 0.75 };
        let mut child = evolution::mutate_locally(parent, cfg, &mut rng, scale);
        if generation >= self.since + REFUGE_TUNING && rng.unit() < 0.5 {
            evolution::structural_mutation_any(&mut child, cfg, &mut rng);
        }
        child.id = evolution::bred_id(round, slot);
        Some(child)
    }
}

/// Generations between the stepping stones of the island ring.
const STONE_INTERVAL: u32 = 50;
/// Generations a graduate is protected against bodies of other plans.
const GRADUATE_GRACE: u32 = 3;
/// Share of the structural and novelty children of a reshaped nursery that
/// take a limb from a body of another plan (a hybrid; Arnold, 1997).
const RESHAPED_CROSS_SHARE: f32 = 0.3;

/// Places in the founder bank, and the share of the main islands' own slots
/// that breed from it.
const FOUNDERS: usize = 1024;
const FOUNDER_SHARE: f32 = 0.01;
/// The share of the hub's own slots that breed from the hall of fame, and
/// the most plans it holds.
const HALL_SHARE: f32 = 0.02;
const HALL_PLANS: usize = 2048;

/// Generations a wild champion breeds in the hub's pen, and the share of the
/// hub's own slots that breed from the pen.
const PEN_GENERATIONS: u32 = 30;
const PEN_SHARE: f32 = 0.1;

/// The emitter mix of `island`: each isolated island leans a few points
/// toward one emitter, so the islands develop different habits (Whitley,
/// 1999). The hub, the wild islands and the nurseries keep the mix.
fn island_weights(weights: &[f64; qd::EMITTER_COUNT], island: usize) -> [f64; qd::EMITTER_COUNT] {
    // (from, to): five points move from one emitter to another.
    let lean = match island {
        0 => Some((Emitter::Structural, Emitter::Cma)),
        1 => Some((Emitter::Novelty, Emitter::Structural)),
        2 => Some((Emitter::Cma, Emitter::Novelty)),
        3 => Some((Emitter::Cma, Emitter::Structural)),
        _ => None,
    };
    let mut out = *weights;
    if let Some((from, to)) = lean {
        let moved = out[from.index()].min(0.05);
        out[from.index()] -= moved;
        out[to.index()] += moved;
    }
    out
}

/// What one island's nursery graduated this session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Graduation {
    /// Generation of the last graduation (0 before the first).
    pub generation: u32,
    /// Bodies in the last graduating cohort, and how many the island kept.
    pub sent: usize,
    pub kept: usize,
    /// Bodies the island kept over all graduations this session.
    pub kept_total: usize,
}

/// Island archives: `ISOLATED_ISLANDS` isolated islands, then the hub.
/// Slot `i` breeds from and competes in island `qd::island_of_slot`.
pub fn island_count() -> usize {
    qd::MAIN_ISLANDS + qd::WILD_ISLANDS
}
/// Archives that creatures breed for and compete in: the islands, then one
/// nursery of new random bodies per island, then one nursery of reshaped
/// bodies per island. `Experiment::islands` holds them in this order.
pub fn arena_count() -> usize {
    island_count() * qd::ARENA_KINDS
}
/// The nursery archive of new random bodies of `island` in
/// `Experiment::islands`.
pub fn nursery_of(island: usize) -> usize {
    island_count() + island
}
/// The nursery archive of reshaped bodies of `island`.
pub fn reshaped_of(island: usize) -> usize {
    2 * island_count() + island
}
/// An empty nursery of reshaped bodies of a main island. It starts in the refined layout, so
/// a new body plan has a cell of its own against the bodies of other shapes
/// and sizes, and the nursery holds four times as many bodies as one that
/// keeps one elite per way of moving.
fn new_reshaped_nursery() -> QdArchive {
    let mut archive = QdArchive::default();
    archive.set_refined(true);
    archive
}
/// Empty archives for the islands and their two nurseries each. `refined`
/// says which islands start in the refined layout. The others, and the
/// nursery of new random bodies, keep one elite per way of moving.
fn new_islands(refined: &[bool]) -> Vec<QdArchive> {
    (0..arena_count())
        .map(|arena| {
            if arena >= reshaped_of(0) && !qd::is_wild(arena - reshaped_of(0)) {
                return new_reshaped_nursery();
            }
            let mut archive = QdArchive::default();
            archive.set_refined(refined.get(arena).copied().unwrap_or(false));
            archive
        })
        .collect()
}
/// Islands that never receive immigrants and breed only from their own
/// elites, so each one evolves its own designs.
pub const ISOLATED_ISLANDS: usize = 4;
/// The hub island: every `MIGRATION_INTERVAL` generations it receives copies
/// of each isolated island's best elites, and it breeds from its own archive
/// like any island. Nothing flows from the hub back.
pub fn hub_island() -> usize {
    ISOLATED_ISLANDS
}
/// Share of CMA offspring whose parent is one of its island's fastest 1% of
/// elites; the rest sample by local competition. Spending more on the best
/// elites raised the best distance by about half in fixed-seed tests.
const TOP_PARENT_SHARE: f32 = 0.5;
/// Share of structural and novelty children that graft a limb from an elite
/// with a different body plan.
const CROSS_PLAN_MATE_SHARE: f32 = 0.15;
/// Share of those top-elite CMA offspring bred by an island optimizer
/// (separable CMA-ES in physical units) on one of its fastest designs.
const OPTIMIZER_SHARE: f32 = 0.5;
/// How many of an island's fastest elites the breeding plan ranks.
const FASTEST_ELITES: usize = 512;
/// Generations between migrations to the hub, and the share of each
/// isolated island's elites copied to it.
pub const MIGRATION_INTERVAL: u32 = 25;
/// Generations without a new island record before the island's optimizer
/// turns to its next fastest design.
const OPTIMIZER_STALL: u32 = 30;
pub const MIGRATION_SHARE: f32 = 0.1;
struct OffspringPlan {
    plan: CandidatePlan,
    parent_id: Option<u64>,
    protection: u32,
}

impl Experiment {
    /// A new game with the ring sized before anything is measured
    /// (`RingShape::default`).
    pub fn new(config: Config) -> Result<Self> {
        Self::with_ring(config, RingShape::default())
    }
    /// A new game with a ring of `ring`: the ring holds new random bodies,
    /// and the first generation has no screen bar yet, so every trial runs
    /// in full and records its distance at the screen.
    pub fn with_ring(config: Config, ring: RingShape) -> Result<Self> {
        ensure!(
            ring.block > 0 && ring.blocks > 0,
            "A ring needs at least one block"
        );
        let mut config = config.resolved();
        config.validate()?;
        config.screen = crate::physics::screen_seconds()
            .filter(|&seconds| seconds < config.duration)
            .map(|seconds| crate::physics::Screen {
                seconds,
                bar: f32::NEG_INFINITY,
                young_bar: f32::NEG_INFINITY,
                reshaped_bar: f32::NEG_INFINITY,
            });
        let mut e = Self::empty(config);
        e.ring = ring;
        let shared = Arc::new(e.config.clone());
        e.blocks = e
            .ring
            .ranges(e.ring.len(e.config.population))
            .into_iter()
            .map(|(first, count)| Block {
                first,
                population: Arc::new(evolution::random_block(&e.config, first, count)),
                births: vec![Birth::RANDOM; count],
                config: Arc::clone(&shared),
            })
            .collect();
        Ok(e)
    }
    /// An experiment with empty archives and no ring.
    fn empty(config: Config) -> Self {
        Self {
            config,
            pending: None,
            generation: 0,
            evaluated: 0,
            history: vec![],
            evaluation_seconds: 0.0,
            archive: QdArchive::starting_global(),
            emitter_stats: [EmitterStats::default(); qd::EMITTER_COUNT],
            cma_emitters: vec![],
            qd_version: qd::VERSION,
            breed_round: 0,
            islands: Vec::new(),
            lineage: HashMap::new(),
            island_progress: Vec::new(),
            graduations: Vec::new(),
            reshaped_graduations: Vec::new(),
            last_migration: None,
            reseed: Reseed::default(),
            refuge: Refuge::default(),
            fossils: Vec::new(),
            ring: RingShape::default(),
            blocks: Vec::new(),
            cursor: 0,
            failed: 0,
            screen_window: ScreenWindow::default(),
            young_window: ScreenWindow::default(),
            reshaped_window: ScreenWindow::default(),
            clade_rarity: (u32::MAX, Vec::new()),
            wild_exports: HashMap::new(),
            wild_wins: Vec::new(),
            pen: Vec::new(),
            founders: std::collections::VecDeque::new(),
            founder_plans: std::collections::HashSet::new(),
            hall: Vec::new(),
            plan_born: HashMap::new(),
            island_epoch: Vec::new(),
            rungs: crate::rungs::Audit::default(),
            last_screen: None,
            stage_seconds: [0.0; 2],
            dump: None,
            dump_notice: None,
            confirm_hint: ConfirmHint::default(),
        }
    }
    /// Creatures in the ring.
    pub fn ring_len(&self) -> usize {
        self.blocks.iter().map(Block::len).sum()
    }
    /// The creature in ring slot `slot`.
    pub fn creature(&self, slot: usize) -> Creature {
        let block = self
            .blocks
            .iter()
            .rfind(|b| b.first <= slot)
            .expect("a ring slot");
        block.population.creature(slot - block.first)
    }
    /// Genes the ring holds, in bytes.
    pub fn ring_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.population.bytes()).sum()
    }
    /// Gives block `k` the current trial settings when its world is not the
    /// current one. Only for a block that has not been evaluated yet.
    pub fn retarget_block(&mut self, k: usize, config: &Arc<Config>) {
        if self.blocks[k].config.physics_differs(config) {
            self.blocks[k].config = Arc::clone(config);
        }
    }
    /// Absorbs block `k`, the block at the cursor, with its final results:
    /// offers each creature to its archives in block order, counts the
    /// evaluations, ends the generation once a generation's worth is in, and
    /// breeds the block again from the archives. Returns whether a
    /// generation ended.
    pub fn absorb(&mut self, k: usize, finals: &[EvaluationMetrics]) -> Result<bool> {
        ensure!(
            k == self.cursor && finals.len() == self.blocks[k].len(),
            "Blocks are absorbed whole and in ring order"
        );
        let stale = self.blocks[k].config.physics_differs(&self.config);
        let class: Vec<usize> = (0..finals.len())
            .map(|j| self.blocks[k].screen_class(j))
            .collect();
        self.last_screen = self.blocks[k]
            .config
            .screen
            .map(|s| s.bar)
            .filter(|bar| !stale && bar.is_finite())
            .map(|bar| {
                let kept = finals.iter().filter(|m| m.screen_x >= bar).count();
                (bar, kept as f32 / finals.len().max(1) as f32)
            });
        if !stale {
            // A result from a world that has since changed carries no
            // distance. Each kind of creature sets the bar of its own kind.
            // The bar of the evolved creatures is the one that the top 10% of
            // every result would have reached if the nursery bodies fell
            // short of it, as they do: a tenth of the results are nursery
            // bodies, so about 11% of the evolved creatures pass it.
            let distances = |wanted: usize| -> Vec<f32> {
                finals
                    .iter()
                    .zip(&class)
                    .filter(|&(_, &c)| c == wanted)
                    .map(|(m, _)| m.screen_x)
                    .filter(|x| x.is_finite())
                    .collect()
            };
            let nursery = finals
                .iter()
                .zip(&class)
                .filter(|&(m, &c)| c != 0 && m.screen_x.is_finite())
                .count();
            let ring = self.blocks.len();
            self.screen_window.push(distances(0), nursery, ring);
            self.young_window.push(distances(1), 0, ring);
            self.reshaped_window.push(distances(2), 0, ring);
            self.config.screen = self.next_screen(self.config.duration);
        }
        let started = std::time::Instant::now();
        // A block of the generation dump: its elite re-runs enter no archive,
        // and every creature's row is written with what it entered.
        let dump = self.dump.clone();
        let head = dump
            .as_ref()
            .and_then(|d| d.lock().unwrap_or_else(|e| e.into_inner()).take_head(k));
        // What each creature entered, for the dump's rows and the audit lane.
        let mut kinds = Some(vec![0u8; finals.len()]);
        let marked: Vec<EvaluationMetrics>;
        let finals = match &head {
            Some(head) => {
                marked = finals
                    .iter()
                    .zip(head)
                    .map(|(m, h)| EvaluationMetrics {
                        excluded: m.excluded || h.flags & dump::RERUN != 0,
                        ..*m
                    })
                    .collect();
                &marked[..]
            }
            None => finals,
        };
        self.failed += self.archive_block(k, finals, stale, kinds.as_deref_mut());
        if !stale && let Some(kinds) = &kinds {
            self.record_rungs(k, finals, kinds);
        }
        if let (Some(dump), Some(head), Some(kinds)) = (&dump, &head, &kinds) {
            let block = &self.blocks[k];
            let mut d = dump.lock().unwrap_or_else(|e| e.into_inner());
            // A diagnostic that cannot write gives up; the game goes on.
            let done = d
                .write_rows(&block.population, head, finals, kinds, stale)
                .and_then(|()| d.finished().then(|| d.finish()).transpose());
            drop(d);
            match done {
                Ok(None) => {}
                Ok(Some(notice)) => {
                    eprintln!("{notice}");
                    self.dump_notice = Some(notice);
                    self.dump = None;
                    evolution::record_operators(false);
                }
                Err(err) => {
                    eprintln!("Generation dump stopped: {err:#}");
                    self.dump = None;
                    evolution::record_operators(false);
                }
            }
        }
        self.evaluated += finals.len();
        let ended = self.evaluated >= self.config.population;
        if ended {
            self.evaluated -= self.config.population;
            self.end_generation()?;
        }
        let archived = std::time::Instant::now();
        self.stage_seconds[0] += archived.duration_since(started).as_secs_f64();
        let (first, count) = (self.blocks[k].first, self.blocks[k].len());
        let arena = std::mem::take(&mut self.blocks[k].population);
        self.blocks[k] = self.breed_block(first, count, arena);
        self.note_dump_block(k);
        self.cursor = (k + 1) % self.blocks.len();
        self.stage_seconds[1] += archived.elapsed().as_secs_f64();
        Ok(ended)
    }
    /// Counts block `k`'s trials for the stage log and files its audit
    /// creatures' rows for the fit of the early rungs (`rungs`). `kinds`
    /// says what each creature entered.
    fn record_rungs(&mut self, k: usize, finals: &[EvaluationMetrics], kinds: &[u8]) {
        let block = &self.blocks[k];
        let population = &*block.population;
        for (j, m) in finals.iter().enumerate() {
            let bar = block
                .config
                .screen
                .map(|s| block.screen_bar(&s, j))
                .filter(|bar| bar.is_finite());
            self.rungs.note(&m.trace, m.screened);
            let flags = population.flags.get(j).copied().unwrap_or(0);
            if flags & crate::rungs::AUDIT == 0 || m.trace.steps() == 0 {
                continue;
            }
            let g = &population.genomes[j];
            let period = if g.muscle_count > 0 {
                population.muscles[g.muscle_start].period
            } else {
                0.0
            };
            self.rungs.record(crate::rungs::AuditRow {
                trace: m.trace,
                period,
                exempt: flags & crate::rungs::EXEMPT != 0,
                parent_exempt: flags & (crate::rungs::EXEMPT_R1 | crate::rungs::EXEMPT_R2),
                bar_known: bar.is_some(),
                pass3: bar.is_some_and(|bar| m.screen_x >= bar),
                below_bar: bar.is_some_and(|bar| {
                    m.trace.steps() > crate::rungs::SCREEN_STEPS && m.screen_x < bar
                }),
                entrant: kinds[j] != 0,
            });
        }
    }
    /// Evaluates and absorbs the block at the cursor with `evaluate`, its
    /// confirmation trials included. Returns whether a generation ended.
    pub fn step(&mut self, evaluate: &mut Evaluate) -> Result<bool> {
        let k = self.cursor;
        let current = Arc::new(self.config.clone());
        self.retarget_block(k, &current);
        let block = self.blocks[k].clone();
        let standard = evaluate(&block.population, &block.config)?;
        ensure!(
            standard.len() == block.len(),
            "The evaluator returned {} results for {} creatures",
            standard.len(),
            block.len()
        );
        let mut confirmed = HashMap::new();
        loop {
            match self.verdict(k, &standard, &confirmed) {
                Verdict::Final(finals) => {
                    // The block's arena is bred again in place.
                    drop(block);
                    return self.absorb(k, &finals);
                }
                Verdict::Confirm(need) => {
                    let subset = block.population.subset(&need);
                    let cfg = crate::scheduler::confirm_config(&block.config);
                    let results = evaluate(&subset, &cfg)?;
                    ensure!(results.len() == need.len(), "Missing confirmation results");
                    confirmed.extend(need.into_iter().zip(results));
                }
            }
        }
    }
    /// Steps until the current generation ends.
    pub fn run_generation(&mut self, evaluate: &mut Evaluate) -> Result<()> {
        while !self.step(evaluate)? {}
        Ok(())
    }
    /// Offers block `k`'s creatures to the archives in block order, updates
    /// CMA emitters and emitter statistics, and returns how many trials
    /// failed. Screened and excluded results, and every result of a
    /// `stale` block, enter no archive.
    /// With `kinds` it also marks, per position, what the creature entered
    /// (`dump::ISLAND`, `NURSERY`, `RESERVE`, `GLOBAL`).
    fn archive_block(
        &mut self,
        k: usize,
        finals: &[EvaluationMetrics],
        stale: bool,
        mut kinds: Option<&mut [u8]>,
    ) -> usize {
        let profile = std::env::var_os("EVOLUTION_PROFILE_BREED").is_some();
        let mut timings = [0.0f64; 7];
        let mut section = std::time::Instant::now();
        self.ensure_islands();
        let block = self.blocks[k].clone();
        let population = &*block.population;
        let births = &block.births;
        let first = block.first;
        let mut entered: Vec<usize> = Vec::new();
        let previous_parent_ids: [Option<u64>; qd::EMITTER_COUNT] = std::array::from_fn(|i| {
            self.emitter_stats[i]
                .last_parent
                .and_then(|index| self.archive.entries.get(index))
                .map(|elite| elite.creature.id)
        });
        let mut discoveries = [0u64; qd::EMITTER_COUNT];
        let mut improvements = [0u64; qd::EMITTER_COUNT];
        let mut rewards = [0.0f64; qd::EMITTER_COUNT];
        let mut cma_samples = vec![Vec::<(usize, f32)>::new(); self.cma_emitters.len()];
        let optimizers: Vec<bool> = self.cma_emitters.iter().map(|c| c.optimizing()).collect();
        // Parallel prefilter: descriptors and behavior-offer eligibility against
        // the start-of-block global archive. Occupant fitness only ever rises,
        // so a snapshot reject stays a live reject. Inserts still commit
        // sequentially in block order.
        struct Prep {
            descriptor: qd::Descriptor,
            emitter: Emitter,
            score: f32,
            fine: bool,
            protection: u32,
            behavior_candidate: bool,
            /// A structural or novelty child may enter its island's
            /// morphology reserve, and has a body plan key.
            structural: bool,
            plan: u64,
            /// Offered to no archive.
            screened: bool,
            /// Fitness of the global archive's elite in this creature's
            /// cell at the start of the block, for a CMA sample.
            elite_before: Option<f32>,
        }
        let arenas = self.islands.len().max(arena_count());
        let positions: Vec<usize> = (0..block.len()).collect();
        // The archive each creature breeds for and competes in.
        let arena_of: Vec<u8> = positions
            .par_iter()
            .map(|&j| qd::arena_of_slot(first + j, arenas) as u8)
            .collect();
        let prep: Vec<Prep> = positions
            .par_iter()
            .map(|&j| {
                let m = &finals[j];
                let score = m.fitness;
                let birth = births[j];
                let emitter = birth.emitter;
                let genome = &population.genomes[j];
                let nodes =
                    &population.nodes[genome.node_start..genome.node_start + genome.node_count];
                let muscles = &population.muscles
                    [genome.muscle_start..genome.muscle_start + genome.muscle_count];
                let descriptor = qd::descriptor(nodes, muscles, m.behavior);
                let screened = stale || m.screened || m.excluded;
                // A nursery creature is offered to its nursery only.
                let nursery = arena_of[j] as usize >= island_count()
                    || qd::is_wild(qd::island_of_slot(first + j, island_count()));
                let valid = score.is_finite() && score > FAILED && !screened;
                let behavior_candidate = if valid && !nursery {
                    let niche = self.archive.cell_of(descriptor);
                    match self.archive.slot_for(&niche) {
                        Some(slot) => score > self.archive.entries[slot].fitness,
                        None => self.archive.behavior_count() < self.archive.limit(),
                    }
                } else {
                    false
                };
                let structural = valid && matches!(emitter, Emitter::Structural | Emitter::Novelty);
                let plan = if structural {
                    qd::plan_key_of_population(population, j)
                } else {
                    0
                };
                let elite_before = (emitter == Emitter::Cma)
                    .then_some(birth.cma)
                    .flatten()
                    .filter(|_| score.is_finite() && score > FAILED)
                    .and_then(|_| self.archive.slot_for(&self.archive.cell_of(descriptor)))
                    .map(|slot| self.archive.entries[slot].fitness);
                Prep {
                    descriptor,
                    emitter,
                    score,
                    fine: m.fine,
                    protection: birth.protection,
                    behavior_candidate,
                    structural,
                    plan,
                    screened,
                    elite_before,
                }
            })
            .collect();
        timings[2] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // Every creature also competes in its own island's archive, and a
        // new body plan that does not take a behavior cell may enter the
        // island's morphology reserve. Each island's offers resolve in block
        // order and the islands are independent, so the streams run in
        // parallel.
        let generation = self.generation;
        // Per island: the positions that entered, the emitter and offer of
        // each reserve entry, the positions that entered the reserve, and
        // the new body plans that took neither a cell nor a reserve place.
        type IslandResult = (Vec<usize>, Vec<(usize, qd::Offer)>, Vec<usize>, Vec<usize>);
        let island_results: Vec<IslandResult> = self
            .islands
            .par_iter_mut()
            .enumerate()
            .map(|(island, archive)| {
                let mut entered = Vec::new();
                let mut reserve_offers = Vec::new();
                let mut reserve_entered = Vec::new();
                let mut routed = Vec::new();
                // Reserve admission needs a score above the island's best
                // behavior elite and reserve entry of the same body plan, or
                // above the reserve's floor once it is full
                // (`QdArchive::offer_morphology` makes the final check).
                let mut parents: HashMap<u64, (u64, bool)> = HashMap::new();
                let mut bars: HashMap<u64, (f32, f32)> = HashMap::new();
                for (slot, elite) in archive.entries.iter().enumerate() {
                    let morphology = qd::is_morphology_niche(&elite.niche);
                    parents.insert(elite.creature.id, (archive.plan_key(slot), morphology));
                    let bar = bars
                        .entry(archive.plan_key(slot))
                        .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                    if morphology {
                        bar.1 = bar.1.max(elite.fitness);
                    } else {
                        bar.0 = bar.0.max(elite.fitness);
                    }
                }
                for (j, p) in prep.iter().enumerate() {
                    if arena_of[j] as usize != island {
                        continue;
                    }
                    if !p.score.is_finite() || p.score <= FAILED || p.screened {
                        continue;
                    }
                    let behavior = archive.offer(
                        population,
                        j,
                        p.descriptor,
                        p.score,
                        p.fine,
                        p.emitter,
                        generation,
                        p.protection,
                    );
                    if behavior.inserted {
                        entered.push(j);
                        if p.structural {
                            let bar = bars
                                .entry(p.plan)
                                .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                            bar.0 = bar.0.max(p.score);
                        }
                        continue;
                    }
                    if !p.structural {
                        continue;
                    }
                    // A reserve place goes to a new body plan, or to a better
                    // child of a reserve entry with the same plan.
                    let parent = births[j].parent_id.and_then(|id| parents.get(&id));
                    let changed = parent.is_some_and(|&(plan, _)| plan != p.plan);
                    let from_reserve =
                        parent.is_some_and(|&(plan, morphology)| morphology && plan == p.plan);
                    if !(changed || from_reserve) {
                        continue;
                    }
                    let floor = archive.morphology_floor();
                    let admits = match bars.get(&p.plan) {
                        Some(&(behavior, reserve)) if reserve > f32::NEG_INFINITY => {
                            p.score > behavior && p.score > reserve
                        }
                        Some(&(behavior, _)) => {
                            p.score > behavior && floor.is_none_or(|floor| p.score > floor)
                        }
                        None => floor.is_none_or(|floor| p.score > floor),
                    };
                    // A new body plan that an island turns away goes to the
                    // island's nursery of reshaped bodies.
                    let routes = changed && island < island_count();
                    if !admits {
                        if routes {
                            routed.push(j);
                        }
                        continue;
                    }
                    let offer = archive.offer_morphology(
                        population,
                        j,
                        p.descriptor,
                        qd::topology_of_population(population, j),
                        p.score,
                        p.fine,
                        p.emitter,
                        generation,
                        p.protection,
                    );
                    if offer.inserted {
                        entered.push(j);
                        reserve_entered.push(j);
                        let bar = bars
                            .entry(p.plan)
                            .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                        bar.1 = bar.1.max(p.score);
                        reserve_offers.push((p.emitter.index(), offer));
                    } else if routes {
                        routed.push(j);
                    }
                }
                (entered, reserve_offers, reserve_entered, routed)
            })
            .collect();
        // The turned away body plans compete for cells of their island's
        // nursery of reshaped bodies, where only such bodies compete.
        let routed: Vec<Vec<usize>> = island_results
            .iter()
            .take(island_count())
            .map(|result| result.3.clone())
            .collect();
        let routed_entered: Vec<Vec<usize>> = self.islands[reshaped_of(0)..]
            .par_iter_mut()
            .zip(routed)
            .map(|(archive, routed)| {
                routed
                    .into_iter()
                    .filter(|&j| {
                        let p = &prep[j];
                        archive
                            .offer(
                                population,
                                j,
                                p.descriptor,
                                p.score,
                                p.fine,
                                p.emitter,
                                generation,
                                p.protection,
                            )
                            .inserted
                    })
                    .collect()
            })
            .collect();
        let mut island_changed: Vec<bool> = island_results
            .iter()
            .map(|(group, _, _, _)| !group.is_empty())
            .collect();
        for (island, entered) in routed_entered.iter().enumerate() {
            island_changed[reshaped_of(island)] |= !entered.is_empty();
        }
        // A wild migrant that took a hub cell counts for its wild island.
        if !self.wild_exports.is_empty()
            && let Some((group, _, _, _)) = island_results.get(hub_island())
        {
            for &j in group {
                let id = population.genomes[j].id;
                if let Some(from) = self.wild_exports.remove(&id) {
                    if self.wild_wins.len() < island_count() {
                        self.wild_wins.resize(island_count(), 0);
                    }
                    self.wild_wins[from] += 1;
                }
            }
        }
        // The first elite of a new body plan in a main island joins the
        // founder bank.
        {
            for (group, _, _, _) in island_results.iter().take(qd::MAIN_ISLANDS) {
                for &j in group {
                    let plan = qd::plan_key_of_population(population, j);
                    if self.founder_plans.insert(plan) {
                        if self.founders.len() >= FOUNDERS {
                            self.founders.pop_front();
                        }
                        self.founders.push_back(population.creature(j));
                    }
                }
            }
        }
        let mut reserve_offers = Vec::new();
        for (arena, (group, offers, reserves, _)) in island_results.into_iter().enumerate() {
            if let Some(kinds) = kinds.as_deref_mut() {
                let kind = if arena < island_count() {
                    dump::ISLAND
                } else {
                    dump::NURSERY
                };
                for &j in &group {
                    kinds[j] |= kind;
                }
                for &j in &reserves {
                    kinds[j] = (kinds[j] & !kind) | dump::RESERVE;
                }
            }
            entered.extend(group);
            // Nursery entries count for no emitter: the emitter statistics
            // describe the islands' search.
            if arena < island_count() {
                reserve_offers.extend(offers);
            }
        }
        for group in routed_entered {
            if let Some(kinds) = kinds.as_deref_mut() {
                for &j in &group {
                    kinds[j] |= dump::NURSERY;
                }
            }
            entered.extend(group);
        }
        timings[0] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // The scores depend only on the archive's elites, so an island that
        // took no offer keeps the ones it has. A reshaped nursery takes
        // offers in every block, so it refreshes once a generation instead
        // (`end_generation`).
        for (island, changed) in self
            .islands
            .iter_mut()
            .zip(island_changed)
            .take(reshaped_of(0))
        {
            if changed || !island.scores_current() {
                island.refresh_behavior_scores();
            }
        }
        timings[1] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // The selected evaluation engine owns the score and behavior. CPU
        // playback and cross-engine comparisons are diagnostics only; they do
        // not edit archive fitness or descriptors.
        let mut prep = prep;
        let mut best_by_niche: HashMap<qd::Niche, usize> = HashMap::new();
        for (j, p) in prep.iter().enumerate() {
            if p.behavior_candidate {
                let best = best_by_niche
                    .entry(self.archive.cell_of(p.descriptor))
                    .or_insert(j);
                if prep[*best].score < p.score {
                    *best = j;
                }
            }
        }
        let behavior_best: std::collections::HashSet<usize> =
            best_by_niche.values().copied().collect();
        for (j, p) in prep.iter_mut().enumerate() {
            p.behavior_candidate &= behavior_best.contains(&j);
        }
        timings[2] += section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        let mut attempts = [0u64; qd::EMITTER_COUNT];
        let mut failed = 0usize;
        let mut global_changed = false;
        let mut behavior_inserted = false;
        for (j, prep) in prep.into_iter().enumerate() {
            if !prep.score.is_finite() || prep.score <= FAILED {
                failed += 1;
            }
            let cma = births[j].cma;
            if arena_of[j] as usize >= island_count() {
                // Nursery samples rank by distance alone.
                if prep.emitter == Emitter::Cma
                    && let Some(cma) = cma
                    && let Some(samples) = cma_samples.get_mut(cma)
                    && prep.score.is_finite()
                    && prep.score > FAILED
                {
                    samples.push((j, prep.score));
                }
                continue;
            }
            let emitter_index = prep.emitter.index();
            attempts[emitter_index] += 1;
            // The CMA improvement key needs the cell's fitness before the
            // offers; only CMA samples use it. The prefilter read the cell's
            // elite before any offer of this block; after the first
            // insertion a new read keeps the order.
            let elite_before = if !behavior_inserted {
                prep.elite_before
            } else {
                (prep.emitter == Emitter::Cma)
                    .then_some(cma)
                    .flatten()
                    .filter(|_| prep.score.is_finite() && prep.score > FAILED)
                    .and_then(|_| {
                        self.archive
                            .slot_for(&self.archive.cell_of(prep.descriptor))
                    })
                    .map(|slot| self.archive.entries[slot].fitness)
            };
            let offer = if prep.behavior_candidate {
                self.archive.offer(
                    population,
                    j,
                    prep.descriptor,
                    prep.score,
                    prep.fine,
                    prep.emitter,
                    self.generation,
                    prep.protection,
                )
            } else {
                qd::Offer::default()
            };
            behavior_inserted |= offer.inserted;
            // CMA-ME improvement ranking: new niches first, then improvement over
            // the niche's elite, then how far short of it a sample fell.
            if prep.emitter == Emitter::Cma
                && let Some(cma) = cma
                && let Some(samples) = cma_samples.get_mut(cma)
                && prep.score.is_finite()
                && prep.score > FAILED
            {
                let key = match elite_before {
                    _ if optimizers[cma] => prep.score,
                    None if offer.inserted => 1.0e6 + prep.score,
                    Some(before) if offer.inserted => 1.0e3 + (prep.score - before),
                    Some(before) => prep.score - before,
                    None => prep.score - 1.0e3,
                };
                samples.push((j, key));
            }
            if offer.inserted {
                global_changed = true;
                entered.push(j);
                if let Some(kinds) = kinds.as_deref_mut() {
                    kinds[j] |= dump::GLOBAL;
                }
                rewards[emitter_index] += offer.reward;
                if offer.new_niche {
                    discoveries[emitter_index] += 1;
                } else {
                    improvements[emitter_index] += 1;
                }
            }
        }
        // Island reserve entries count for their emitters like archive entries.
        for (emitter_index, offer) in reserve_offers {
            rewards[emitter_index] += offer.reward;
            if offer.new_niche {
                discoveries[emitter_index] += 1;
            } else {
                improvements[emitter_index] += 1;
            }
        }
        timings[3] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        for (emitter, samples) in self.cma_emitters.iter_mut().zip(&mut cma_samples) {
            emitter.tell(population, samples);
        }
        qd::record_emitter_batch(
            &mut self.emitter_stats,
            &attempts,
            &discoveries,
            &improvements,
            &rewards,
        );
        let parent_index_by_id: HashMap<_, _> = self
            .archive
            .entries
            .iter()
            .enumerate()
            .map(|(index, elite)| (elite.creature.id, index))
            .collect();
        for (stats, parent_id) in self.emitter_stats.iter_mut().zip(previous_parent_ids) {
            stats.last_parent = parent_id.and_then(|id| parent_index_by_id.get(&id).copied());
        }
        timings[4] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        if global_changed || !self.archive.scores_current() {
            self.archive.refresh_behavior_scores();
        }
        timings[5] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        entered.sort_unstable();
        entered.dedup();
        let entered_count = entered.len();
        let records: Vec<(u64, Ancestor)> = entered
            .par_iter()
            .filter_map(|&j| self.ancestor_of(population, births[j], j, &finals[j]))
            .collect();
        self.lineage.extend(records);
        timings[6] = section.elapsed().as_secs_f64();
        if profile {
            eprintln!(
                "Archive profile: generation {}, block {}, island offers {:.6} s, island refresh {:.6} s, prefilter {:.6} s, global offers {:.6} s, cma tell {:.6} s, archive refresh {:.6} s, lineage {:.6} s, entered {}",
                self.generation,
                block.len(),
                timings[0],
                timings[1],
                timings[2],
                timings[3],
                timings[4],
                timings[5],
                timings[6],
                entered_count
            );
        }
        failed
    }
    fn push_archive_stats(&mut self, failed: usize) {
        if self.history.len() > self.generation as usize {
            return;
        }
        let elites: Vec<_> = self
            .archive
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .collect();
        // The median, the worst, the mean and the percentiles read the best
        // elite of each way of moving, so they mean what they meant before the
        // archive had body classes. The histogram and the body types count
        // every elite.
        let mut ways: Vec<_> = self
            .archive
            .best_per_way_of_moving()
            .into_iter()
            .map(|slot| &self.archive.entries[slot])
            .collect();
        ways.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let count = ways.len();
        let archive_best = self.archive.best_fitness();
        let quantile = |p: f32| {
            if count == 0 {
                0.0
            } else {
                ways[((1.0 - p / 100.0) * (count - 1) as f32).round() as usize].fitness
            }
        };
        let mut histogram = BTreeMap::<i32, u32>::new();
        let mut species = BTreeMap::<(usize, usize), u32>::new();
        let mut sum = 0.0f64;
        for elite in &ways {
            sum += elite.fitness as f64;
        }
        for elite in &elites {
            *histogram
                .entry((elite.fitness * 100.0).floor() as i32)
                .or_default() += 1;
            *species
                .entry((elite.creature.node_count(), elite.creature.muscle_count()))
                .or_default() += 1;
        }
        let mut percentiles: Vec<_> = PERCENTILES.iter().map(|&p| quantile(p)).collect();
        if let Some(best_percentile) = percentiles.last_mut() {
            *best_percentile = archive_best.max(0.0);
        }
        let mut all_elites: Vec<_> = self.archive.entries.iter().collect();
        all_elites.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let representatives = if all_elites.is_empty() {
            vec![self.blocks[0].population.creature(0); 3]
        } else {
            [all_elites.len() - 1, (all_elites.len() - 1) / 2, 0]
                .map(|i| all_elites[i].creature.unpack())
                .to_vec()
        };
        self.history.push(Stats {
            generation: self.generation,
            best: archive_best.max(0.0),
            median: quantile(50.0),
            worst: quantile(0.0),
            mean: if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            },
            failed,
            seconds: self.evaluation_seconds,
            population: self.config.population,
            percentiles,
            histogram: histogram.into_iter().collect(),
            species: species.into_iter().map(|((n, m), c)| (n, m, c)).collect(),
            representatives,
            config: self.config.clone(),
            archive_cells: elites.len(),
            qd_score: self.archive.qd_score,
            archive_coverage: self.archive.coverage(),
            emitters: self.emitter_stats,
            ring: self.ring,
            plans: elites
                .iter()
                .map(|e| e.topology.plan_key())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            plan_age: {
                let present: std::collections::HashSet<u64> =
                    elites.iter().map(|e| e.topology.plan_key()).collect();
                let generation = self.generation;
                for &plan in &present {
                    self.plan_born.entry(plan).or_insert(generation);
                }
                self.plan_born.retain(|plan, _| present.contains(plan));
                let mut ages: Vec<u32> =
                    self.plan_born.values().map(|&b| generation - b).collect();
                if ages.is_empty() {
                    0.0
                } else {
                    let mid = ages.len() / 2;
                    *ages.select_nth_unstable(mid).1 as f32
                }
            },
            clades: self.effective_clades(elites.iter().map(|e| e.creature.id)),
        });
    }
    /// The next elite queued for the island of `slot`. A nursery slot takes
    /// none, so a re-tested elite competes in its island's archive.
    fn reseed_for_slot(&mut self, slot: usize) -> Option<Creature> {
        let islands = island_count();
        if qd::is_nursery_slot(slot, islands) {
            return None;
        }
        self.reseed.pop(qd::island_of_slot(slot, islands))
    }
    /// Creates empty island archives if they are missing. They fill from
    /// their own slots' offspring (and queued reseeds). The global archive
    /// is never split among them, because that would mix the islands.
    fn ensure_islands(&mut self) {
        if self.islands.len() == arena_count() {
            return;
        }
        self.islands = new_islands(&[]);
        self.island_progress.clear();
        self.graduations.clear();
        self.reshaped_graduations.clear();
        self.last_migration = None;
    }
    /// Whether the global archive's best has stood through the last
    /// `OPTIMIZER_STALL` generations of this world: the archives have stopped
    /// climbing.
    fn global_stalled(&self) -> bool {
        let stall = OPTIMIZER_STALL as usize;
        self.history.len() > stall && {
            let now = &self.history[self.history.len() - 1];
            let then = &self.history[self.history.len() - 1 - stall];
            !then.config.physics_differs(&self.config) && now.best <= then.best
        }
    }
    /// Refines each island whose archive is `qd::REFINE_AFTER` generations
    /// old: its elites move to the cells of their body classes, and from then
    /// on a body of another shape or size has a cell of its own. Until then
    /// an island keeps one elite per way of moving, so the climb of a new game
    /// pools its lineages as the old archive did. A refined archive stays
    /// refined through a world change (`reset_search_context`), and the
    /// nurseries of new random bodies never refine.
    fn refine_archives(&mut self) {
        for island in 0..island_count().min(self.islands.len()) {
            // A wild island keeps one elite per way of moving: a hundred
            // refined islands with two refined nurseries each held 20 GB.
            if qd::is_wild(island) {
                continue;
            }
            let epoch = self.island_epoch.get(island).copied().unwrap_or(0);
            let archive = &mut self.islands[island];
            // The isolated islands refine 10 generations apart, so climbing
            // and refined islands exist side by side (Hornby, 2006).
            let after = qd::REFINE_AFTER + if island < ISOLATED_ISLANDS { 10 * island as u32 } else { 0 };
            if !archive.refined() && self.generation.saturating_sub(epoch) >= after {
                archive.set_refined(true);
                archive.rebin();
            }
        }
    }
    /// Every `NURSERY_GENERATIONS` generations each island takes the bodies
    /// of its nurseries that beat its elites, and the global archive takes
    /// those the island kept. The nursery of new random bodies starts over
    /// with new random bodies. The nursery of reshaped bodies keeps every
    /// body and goes on tuning it.
    fn graduate_nurseries(&mut self) {
        if self.islands.len() != arena_count()
            || self.generation == 0
            || !self.generation.is_multiple_of(qd::NURSERY_GENERATIONS)
        {
            return;
        }
        self.graduations
            .resize(island_count(), Graduation::default());
        self.reshaped_graduations
            .resize(island_count(), Graduation::default());
        for island in 0..island_count() {
            let nursery = nursery_of(island);
            let mut cohort: Vec<qd::Elite> = std::mem::take(&mut self.islands[nursery].entries)
                .into_iter()
                .filter(|e| !qd::is_morphology_niche(&e.niche))
                .collect();
            self.islands[nursery].rebuild_indices();
            let sent = cohort.len();
            let kept = self.offer_to_island(island, &mut cohort);
            let log = &mut self.graduations[island];
            *log = Graduation {
                generation: self.generation,
                sent,
                kept,
                kept_total: log.kept_total + kept,
            };
            if let Some(progress) = self.island_progress.get_mut(nursery) {
                *progress = (f32::NEG_INFINITY, self.generation);
            }
            // Only the reshaped bodies the island would take are copied.
            let home = &self.islands[island];
            let reshaped = &self.islands[reshaped_of(island)];
            let sent = reshaped.behavior_count();
            let mut winners: Vec<qd::Elite> = reshaped
                .entries
                .iter()
                .filter(|e| home.would_take(e))
                .cloned()
                .collect();
            let kept = self.offer_to_island(island, &mut winners);
            let log = &mut self.reshaped_graduations[island];
            *log = Graduation {
                generation: self.generation,
                sent,
                kept,
                kept_total: log.kept_total + kept,
            };
        }
        // Each island that took bodies and the global archive refresh once.
        for island in &mut self.islands[..island_count()] {
            if !island.scores_current() {
                island.refresh_behavior_scores();
            }
        }
        if !self.archive.scores_current() {
            self.archive.refresh_behavior_scores();
        }
    }
    /// The behavior scores of the nurseries of reshaped bodies, which take
    /// offers in every block, are refreshed once a generation.
    fn refresh_reshaped_scores(&mut self) {
        if self.islands.len() == arena_count() {
            for island in 0..island_count() {
                self.islands[reshaped_of(island)].refresh_behavior_scores();
            }
        }
    }
    /// Offers `elites` to `island`, fastest first, marked as graduates: each
    /// takes the cell of an island elite it beats, or an empty cell, and the
    /// global archive takes those the island kept. Returns how many it kept.
    fn offer_to_island(&mut self, island: usize, elites: &mut [qd::Elite]) -> usize {
        elites.sort_unstable_by(|a, b| {
            b.fitness
                .total_cmp(&a.fitness)
                .then_with(|| a.niche.cmp(&b.niche))
        });
        let mut kept = 0;
        // A graduate keeps its cell against bodies of other plans for a few
        // generations, as a young species is protected in NEAT (Stanley and
        // Miikkulainen, 2002). The global archive's copy has no grace.
        let grace = self.generation + GRADUATE_GRACE;
        for elite in elites.iter_mut() {
            elite.graduate = true;
            let mut copy = elite.clone();
            if !qd::bio_off(16) {
                copy.protected_until = copy.protected_until.max(grace);
            }
            if self.islands[island].absorb(&copy) {
                kept += 1;
                self.archive.absorb(elite);
            }
        }
        kept
    }
    /// Every `MIGRATION_INTERVAL` generations the hub receives copies of the
    /// best share of each isolated island's elites. The isolated islands
    /// never receive any.
    fn migrate_islands(&mut self) {
        if self.islands.len() != arena_count()
            || !self.generation.is_multiple_of(MIGRATION_INTERVAL)
        {
            return;
        }
        let hub = hub_island();
        let migrants: Vec<Vec<qd::Elite>> = self.islands[..ISOLATED_ISLANDS]
            .iter()
            .map(|island| {
                let mut elites: Vec<&qd::Elite> = island
                    .entries
                    .iter()
                    .filter(|e| !qd::is_morphology_niche(&e.niche))
                    .collect();
                elites.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
                let take =
                    ((elites.len() as f32 * MIGRATION_SHARE).ceil() as usize).min(elites.len());
                elites[..take].iter().map(|e| (*e).clone()).collect()
            })
            .collect();
        let mut exchange = vec![(0, 0); island_count()];
        for (from, group) in migrants.into_iter().enumerate() {
            let to = &mut self.islands[hub];
            let kept = group.iter().filter(|elite| to.absorb(elite)).count();
            exchange[from] = (group.len(), kept);
        }
        // A wild island sends its best tenth by its own world's distance. Each
        // runs once as it is in the hub's world and takes a cell if it is
        // fast enough there; and it enters the hub's pen, where it breeds in
        // the hub's slots for `PEN_GENERATIONS` generations so its line can
        // adapt to the hub's world before it is dropped (owner).
        let until = self.generation + PEN_GENERATIONS;
        for from in qd::MAIN_ISLANDS..island_count() {
            let island = &self.islands[from];
            let mut elites: Vec<&qd::Elite> = island
                .entries
                .iter()
                .filter(|e| !qd::is_morphology_niche(&e.niche))
                .collect();
            elites.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
            let take = ((elites.len() as f32 * MIGRATION_SHARE).ceil() as usize).min(elites.len());
            let sent: Vec<Creature> = elites[..take].iter().map(|e| e.creature.unpack()).collect();
            self.pen
                .extend(sent.iter().map(|c| (c.clone(), until)));
            exchange[from] = (sent.len(), 0);
            for creature in sent {
                self.wild_exports.insert(creature.id, from);
                self.reseed.push(hub, creature);
            }
        }
        self.last_migration = Some((self.generation, exchange));
        self.islands[hub].refresh_behavior_scores();
    }
    /// Every `STONE_INTERVAL` generations each isolated island sends one
    /// elite to the next island in a ring: the fastest elite of its rarest
    /// body plan, so isolation is almost kept and a rare design gets a second
    /// home (Cantu-Paz, 2000, migration topologies).
    fn step_stones(&mut self) {
        if qd::bio_off(128)
            || self.islands.len() != arena_count()
            || self.generation == 0
            || !self.generation.is_multiple_of(STONE_INTERVAL)
        {
            return;
        }
        let stones: Vec<Option<qd::Elite>> = (0..ISOLATED_ISLANDS)
            .map(|island| {
                let archive = &self.islands[island];
                let mut counts: HashMap<u64, usize> = HashMap::new();
                for i in 0..archive.entries.len() {
                    if !qd::is_morphology_niche(&archive.entries[i].niche) {
                        *counts.entry(archive.plan_key(i)).or_default() += 1;
                    }
                }
                (0..archive.entries.len())
                    .filter(|&i| !qd::is_morphology_niche(&archive.entries[i].niche))
                    .min_by(|&a, &b| {
                        counts[&archive.plan_key(a)]
                            .cmp(&counts[&archive.plan_key(b)])
                            .then(archive.entries[b].fitness.total_cmp(&archive.entries[a].fitness))
                            .then(a.cmp(&b))
                    })
                    .map(|i| archive.entries[i].clone())
            })
            .collect();
        for (island, stone) in stones.into_iter().enumerate() {
            if let Some(elite) = stone {
                self.islands[(island + 1) % ISOLATED_ISLANDS].absorb(&elite);
            }
        }
    }
    /// Chooses emitters, parents, and CMA slots for offspring in `slots`.
    /// The breeding `round` salts the random streams, so no two blocks
    /// repeat a draw.
    fn plan_offspring(
        &mut self,
        cfg: &Config,
        generation: u32,
        round: u64,
        slots: &[usize],
    ) -> Vec<OffspringPlan> {
        let profile = std::env::var_os("EVOLUTION_PROFILE_BREED").is_some();
        let mut plan_times = [0.0f64; 4];
        let mut section = std::time::Instant::now();
        self.ensure_islands();
        for island in &mut self.islands {
            island.ensure_least_visited();
        }
        let weights = qd::emitter_weights(&self.emitter_stats);
        let mut reset_cma = HashMap::<(usize, qd::Niche, u64), usize>::new();
        // CMA slot lookup keyed by (island, niche, body plan): an emitter
        // samples around one island's elite, so it serves only that island.
        // The bucket stores the full key, so the per-offspring probe hashes
        // and compares without cloning the topology vector; clones are only
        // paid when a slot is created or replaced.
        type CmaKey = (usize, qd::Niche, u64);
        struct CmaLookup {
            buckets: HashMap<u64, Vec<(CmaKey, usize)>>,
        }
        impl CmaLookup {
            fn hash(island: usize, niche: &qd::Niche, plan: u64) -> u64 {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                island.hash(&mut hasher);
                niche.hash(&mut hasher);
                plan.hash(&mut hasher);
                hasher.finish()
            }
            fn get(&self, island: usize, niche: &qd::Niche, plan: u64) -> Option<usize> {
                self.buckets
                    .get(&Self::hash(island, niche, plan))?
                    .iter()
                    .find(|((i, n, t), _)| *i == island && n == niche && *t == plan)
                    .map(|(_, index)| *index)
            }
            fn insert(&mut self, key: CmaKey, index: usize) {
                let bucket = self
                    .buckets
                    .entry(Self::hash(key.0, &key.1, key.2))
                    .or_default();
                if let Some(entry) = bucket.iter_mut().find(|(stored, _)| *stored == key) {
                    entry.1 = index;
                } else {
                    bucket.push((key, index));
                }
            }
            fn remove(&mut self, island: usize, niche: &qd::Niche, plan: u64, index: usize) {
                let hash = Self::hash(island, niche, plan);
                let Some(bucket) = self.buckets.get_mut(&hash) else {
                    return;
                };
                bucket.retain(|((i, n, t), stored_index)| {
                    !(*i == island && n == niche && *t == plan && *stored_index == index)
                });
                if bucket.is_empty() {
                    self.buckets.remove(&hash);
                }
            }
        }
        let mut cma_lookup = CmaLookup {
            buckets: HashMap::new(),
        };
        for (index, cma) in self.cma_emitters.iter().enumerate() {
            cma_lookup.insert(
                (cma.island, cma.niche.clone(), cma.topology.plan_key()),
                index,
            );
        }
        let mut used_cma = vec![false; self.cma_emitters.len()];
        let mut out = Vec::with_capacity(slots.len());
        // Phase A: emitter choice and parent sampling against the start-of-batch
        // archive. Each creature has its own deterministic RNG, so parallel order
        // does not change the draws. last_parent is snapshotted instead of updating
        // mid-loop; visit() and CMA slot allocation stay sequential below.
        struct PlanPrep {
            emitter: Emitter,
            parent: Option<usize>,
            parent_id: Option<u64>,
            protection: u32,
            emitter_stale: bool,
            mate: Option<usize>,
            island: usize,
            /// A fast elite whose design's optimizer breeds this offspring.
            optimize: bool,
            /// A reshaped child of an island elite for a reshaped nursery.
            seeded: bool,
        }
        // Each island's elites by body plan key, grouped for crossover
        // partners: (key, slot) sorted, so a plan's elites are one run.
        let by_plan: Vec<Vec<(u64, u32)>> = self
            .islands
            .par_iter()
            .map(|island| {
                let mut keyed: Vec<(u64, u32)> = (0..island.entries.len())
                    .map(|slot| (island.plan_key(slot), slot as u32))
                    .collect();
                keyed.sort_unstable();
                keyed
            })
            .collect();
        // The elites of `island` that share the body plan `key`.
        let same_plan = |island: usize, key: u64| -> &[(u64, u32)] {
            let keyed = &by_plan[island];
            let first = keyed.partition_point(|&(k, _)| k < key);
            let len = keyed[first..].partition_point(|&(k, _)| k == key);
            &keyed[first..first + len]
        };
        plan_times[0] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        let seed = cfg.seed ^ round.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        // Behavior elites per island, fastest first. Both the exploitation
        // pool and the optimizer targets read this order.
        let orders: Vec<Vec<usize>> = self
            .islands
            .iter()
            .map(|island| {
                let mut order: Vec<usize> = (0..island.entries.len())
                    .filter(|&i| !qd::is_morphology_niche(&island.entries[i].niche))
                    .collect();
                // The exploitation pool and the optimizer targets only read
                // the fastest few hundred.
                let faster = |&a: &usize, &b: &usize| {
                    island.entries[b]
                        .fitness
                        .total_cmp(&island.entries[a].fitness)
                        .then(a.cmp(&b))
                };
                if order.len() > FASTEST_ELITES {
                    order.select_nth_unstable_by(FASTEST_ELITES - 1, faster);
                    order.truncate(FASTEST_ELITES);
                }
                order.sort_unstable_by(faster);
                order
            })
            .collect();
        // Each island's fastest elites, for exploitation: 1% of the movement
        // grid (at least 4), however many body classes the archive holds.
        let top_parents: Vec<Vec<usize>> = orders
            .iter()
            .map(|order| {
                let count = (order.len().min(qd::MOVEMENT_CELLS) / 100).max(4);
                order[..count.min(order.len())].to_vec()
            })
            .collect();
        // How rare each elite's clade is in its island, from 0 (the whole
        // archive) to 1 (one elite).
        if qd::RARITY_WEIGHT > 0.0
            && (self.clade_rarity.0 != generation
                || self.clade_rarity.1.len() != self.islands.len())
        {
            let rarities = self
                .islands
                .iter()
                .map(|island| self.clade_rarity_of(island))
                .collect();
            self.clade_rarity = (generation, rarities);
        }
        let rarities = &self.clade_rarity.1;
        let no_rarity = Vec::new();
        // An island's optimizer works on its fastest design: a body plan with
        // a gait cadence band. When the island has not set a record for a
        // while, it turns to its next fastest designs in turn, so one stuck
        // design does not take all local search.
        self.island_progress
            .resize(self.islands.len(), (f32::NEG_INFINITY, generation));
        let optimizer_targets: Vec<Option<usize>> = self
            .islands
            .iter()
            .zip(&top_parents)
            .zip(&orders)
            .zip(&mut self.island_progress)
            .map(|(((island, top), order), progress)| {
                let best = *top.first()?;
                let fitness = island.entries[best].fitness;
                if fitness > progress.0 {
                    *progress = (fitness, generation);
                }
                let mut plans: Vec<usize> = Vec::new();
                // A design is a body plan with a gait cadence band.
                let design = |i: usize| (&island.entries[i].topology, island.entries[i].niche.0[1]);
                for &i in order {
                    if plans.len() >= 4 {
                        break;
                    }
                    if !plans.iter().any(|&p| design(p) == design(i)) {
                        plans.push(i);
                    }
                }
                let turn = (generation.saturating_sub(progress.1) / OPTIMIZER_STALL) as usize;
                Some(plans[turn % plans.len()])
            })
            .collect();
        // A second optimizer target per island: the fastest elite of its
        // rarest clade, so local search also climbs a design the island is
        // about to lose (Fontaine et al., 2020, CMA-ME on several targets).
        let rare_targets: Vec<Option<usize>> = self
            .islands
            .iter()
            .enumerate()
            .map(|(island, archive)| {
                if qd::bio_off(64) {
                    return None;
                }
                let rarity = rarities.get(island)?;
                let top = rarity.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if !top.is_finite() || top <= 0.0 {
                    return None;
                }
                (0..archive.entries.len().min(rarity.len()))
                    .filter(|&i| rarity[i] >= top && !qd::is_morphology_niche(&archive.entries[i].niche))
                    .max_by(|&a, &b| {
                        archive.entries[a]
                            .fitness
                            .total_cmp(&archive.entries[b].fitness)
                            .then(b.cmp(&a))
                    })
            })
            .collect();
        plan_times[1] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        let plan_prep: Vec<PlanPrep> = slots
            .par_iter()
            .map(|&i| {
                let mut rng = Rng::new(seed, generation, i);
                let arena = qd::arena_of_slot(i, self.islands.len());
                let reshaped = qd::is_reshaped_arena(arena, self.islands.len());
                // Bodies the island turned away fill a reshaped nursery.
                // While it is empty its slots breed structural children of
                // the island's elites instead.
                let home = qd::island_of_slot(i, island_count());
                let seeded = reshaped
                    && !self.islands[home].entries.is_empty()
                    && self.islands[arena].entries.is_empty();
                // The archive the child's parent comes from.
                let island = if seeded { home } else { arena };
                let archive = &self.islands[island];
                let archive_empty = archive.entries.is_empty();
                let emitter = if seeded {
                    Emitter::Structural
                } else if archive_empty
                    || (arena >= island_count()
                        && !reshaped
                        && rng.unit() < qd::NURSERY_FRESH_SHARE)
                {
                    Emitter::Restart
                } else {
                    qd::choose_emitter(&mut rng, &island_weights(&weights, island))
                };
                let emitter_stale = self.emitter_stats[emitter.index()].stale();
                let avoid = None;
                let mut optimize = false;
                let mut from_reserve = false;
                let parent = if emitter == Emitter::Restart || archive_empty {
                    None
                } else if seeded {
                    archive.sample_local_competitive(
                        &mut rng,
                        avoid,
                        rarities.get(island).unwrap_or(&no_rarity),
                    )
                } else if emitter == Emitter::Structural
                    && rng.unit() < qd::MORPHOLOGY_PARENT_FRACTION
                {
                    // Each island keeps its own morphology reserve.
                    let drawn = archive.sample_morphology(&mut rng, avoid);
                    from_reserve = drawn.is_some();
                    drawn.or_else(|| {
                        let rarity = rarities.get(island).unwrap_or(&no_rarity);
                        archive.sample_local_competitive(&mut rng, avoid, rarity)
                    })
                } else if emitter == Emitter::Novelty && rng.unit() < 0.5 {
                    // Half the novelty parents are far from the others in
                    // body, not in behavior.
                    archive
                        .sample_body_novel(&mut rng)
                        .or_else(|| archive.sample_novel(&mut rng, avoid))
                } else if emitter == Emitter::Novelty || emitter_stale {
                    archive.sample_novel(&mut rng, avoid)
                } else if emitter == Emitter::Cma
                    && !top_parents[island].is_empty()
                    && rng.unit() < TOP_PARENT_SHARE
                {
                    // Half of these come from the island's optimizer for one
                    // of its fastest designs; the rest explore around the top
                    // elites.
                    optimize = rng.unit() < OPTIMIZER_SHARE;
                    let second = rare_targets.get(island).copied().flatten();
                    Some(if optimize && second.is_some() && rng.unit() < 0.5 {
                        second.expect("a rare target")
                    } else if optimize {
                        optimizer_targets[island].unwrap_or(top_parents[island][0])
                    } else {
                        top_parents[island][rng.index(top_parents[island].len())]
                    })
                } else {
                    let rarity = rarities.get(island).unwrap_or(&no_rarity);
                    archive.sample_local_competitive(&mut rng, avoid, rarity)
                };
                let parent_id = parent.map(|index| archive.entries[index].creature.id);
                let protection = if matches!(emitter, Emitter::Structural | Emitter::Novelty) {
                    generation.saturating_add(qd::PROTECTION_GENERATIONS)
                } else {
                    parent
                        .map(|index| archive.entries[index].protected_until)
                        .unwrap_or(0)
                };
                let mate = match (emitter, parent) {
                    (Emitter::Structural | Emitter::Novelty, Some(p))
                        if !from_reserve && rng.unit() < 0.2 =>
                    {
                        Some(same_plan(island, archive.plan_key(p)))
                            .filter(|group| group.len() > 1)
                            .map(|group| group[rng.index(group.len())].1 as usize)
                            .filter(|&m| m != p)
                    }
                    _ => None,
                };
                // Sometimes the mate has another body plan: the child gets one of
                // its limbs grafted on (see `evolution::mated`).
                let mate = mate.or_else(|| match (emitter, parent) {
                    (Emitter::Structural | Emitter::Novelty, Some(p))
                        if !from_reserve
                            && rng.unit()
                                < if reshaped && !qd::bio_off(32) {
                                    RESHAPED_CROSS_SHARE
                                } else {
                                    CROSS_PLAN_MATE_SHARE
                                } =>
                    {
                        let other = rng.index(archive.entries.len());
                        (other != p && archive.plan_key(other) != archive.plan_key(p))
                            .then_some(other)
                    }
                    _ => None,
                });
                PlanPrep {
                    emitter,
                    parent,
                    parent_id,
                    protection,
                    emitter_stale,
                    mate,
                    island,
                    optimize,
                    seeded,
                }
            })
            .collect();
        plan_times[2] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        for prep in plan_prep {
            let PlanPrep {
                emitter,
                parent,
                parent_id,
                protection,
                emitter_stale,
                mate,
                island,
                optimize,
                seeded,
            } = prep;
            let cma_index = if emitter == Emitter::Cma {
                if let Some(parent_index) = parent {
                    let elite = &self.islands[island].entries[parent_index];
                    let template = elite.creature.unpack();
                    let plan = self.islands[island].plan_key(parent_index);
                    // Each island runs one optimizer per design. It starts from
                    // the design's fastest elite and then follows its own mean,
                    // so recentering on every lucky new best does not throw
                    // away its progress. A converged one restarts.
                    let lookup_niche = if optimize {
                        qd::optimizer_niche(island, elite.niche.0[1])
                    } else {
                        elite.niche.clone()
                    };
                    let converged = |i: &usize| self.cma_emitters[*i].converged() && !used_cma[*i];
                    let mut index = if optimize {
                        cma_lookup
                            .get(island, &lookup_niche, plan)
                            .filter(|i| !converged(i))
                    } else if emitter_stale {
                        reset_cma
                            .get(&(island, lookup_niche.clone(), plan))
                            .copied()
                    } else {
                        cma_lookup.get(island, &lookup_niche, plan)
                    };
                    if index.is_none() {
                        let restart = cma_lookup
                            .get(island, &lookup_niche, plan)
                            .filter(|_| optimize);
                        let replacement = if restart.is_some() {
                            restart
                        } else if self.cma_emitters.len() < qd::CMA_LIMIT {
                            Some(self.cma_emitters.len())
                        } else {
                            self.cma_emitters
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| !used_cma[*i])
                                .min_by_key(|(_, cma)| cma.last_used_generation)
                                .map(|(i, _)| i)
                        };
                        if let Some(slot) = replacement {
                            let mut new = if optimize {
                                // Another optimizer of this island for the
                                // same plan lends its learned step sizes,
                                // unless this is a restart after converging.
                                // Other islands never lend: their step sizes
                                // carry what their search learned.
                                self.cma_emitters
                                    .iter()
                                    .filter(|c| {
                                        c.optimizing()
                                            && c.island == island
                                            && c.topology.plan_key() == plan
                                            && !c.converged()
                                    })
                                    .max_by_key(|c| c.last_used_generation)
                                    .map_or_else(
                                        || {
                                            CmaEmitter::optimizer(
                                                template.clone(),
                                                lookup_niche.clone(),
                                                generation,
                                            )
                                        },
                                        |c| {
                                            c.recentered(
                                                template.clone(),
                                                lookup_niche.clone(),
                                                generation,
                                            )
                                        },
                                    )
                            } else {
                                CmaEmitter::new(template.clone(), elite.niche.clone(), generation)
                            };
                            new.island = island;
                            if slot == self.cma_emitters.len() {
                                self.cma_emitters.push(new);
                                used_cma.push(false);
                            } else {
                                let old = &self.cma_emitters[slot];
                                let (old_island, old_niche, old_plan) =
                                    (old.island, old.niche.clone(), old.topology.plan_key());
                                if cma_lookup.get(old_island, &old_niche, old_plan) == Some(slot) {
                                    cma_lookup.remove(old_island, &old_niche, old_plan, slot);
                                }
                                reset_cma.retain(|_, index| *index != slot);
                                self.cma_emitters[slot] = new;
                            }
                            let new_niche = self.cma_emitters[slot].niche.clone();
                            let new_plan = self.cma_emitters[slot].topology.plan_key();
                            cma_lookup.insert((island, new_niche, new_plan), slot);
                            if emitter_stale && !optimize {
                                reset_cma.insert((island, lookup_niche.clone(), plan), slot);
                            }
                            index = Some(slot);
                        }
                    }
                    if let Some(index) = index {
                        used_cma[index] = true;
                        self.cma_emitters[index].last_used_generation = generation;
                    }
                    index
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(parent_index) = parent {
                self.islands[island].visit(parent_index);
            }
            out.push(OffspringPlan {
                plan: CandidatePlan {
                    emitter,
                    parent,
                    cma: cma_index,
                    mate,
                    seed: seeded,
                },
                parent_id,
                protection,
            });
        }
        plan_times[3] = section.elapsed().as_secs_f64();
        if profile {
            eprintln!(
                "Plan profile: generation {generation}, by_plan {:.6} s, order/optimizer {:.6} s, sampling {:.6} s, cma/visit {:.6} s",
                plan_times[0], plan_times[1], plan_times[2], plan_times[3]
            );
        }
        out
    }
    /// Plans offspring for `slots` in the next breeding round, as a block's
    /// breeding does, and returns the plans with that round. For
    /// `examples/breed_bench.rs`.
    #[doc(hidden)]
    pub fn plan_for_bench(&mut self, slots: &[usize]) -> (Vec<CandidatePlan>, u64) {
        let cfg = self.config.clone();
        self.breed_round += 1;
        let plans = self.plan_offspring(&cfg, self.generation, self.breed_round, slots);
        (
            plans.into_iter().map(|p| p.plan).collect(),
            self.breed_round,
        )
    }
    /// Breeds a block for ring slots `first..first + count` from the current
    /// archives with the current settings, into `arena`: the genes of the
    /// block bred for these slots last time, whose memory the new block
    /// reuses when nothing else holds it. Elites queued by a world change
    /// take the slots of their own islands first. An island without elites
    /// breeds new random bodies.
    fn breed_block(&mut self, first: usize, count: usize, arena: Arc<Population>) -> Block {
        let slots: Vec<usize> = (first..first + count).collect();
        let cfg = self.config.clone();
        self.breed_round += 1;
        let started = std::time::Instant::now();
        let planned = self.plan_offspring(&cfg, self.generation, self.breed_round, &slots);
        let planned_at = started.elapsed();
        // The parents as they are now, for the generation dump's rows.
        let dump_parents: Option<Vec<dump::Parent>> = self.dump_breeding().then(|| {
            planned
                .iter()
                .zip(&slots)
                .map(|(p, &slot)| {
                    let arena = if p.plan.seed {
                        qd::island_of_slot(slot, island_count())
                    } else {
                        qd::arena_of_slot(slot, self.islands.len())
                    };
                    let elite = p
                        .plan
                        .parent
                        .and_then(|i| self.islands.get(arena)?.entries.get(i));
                    dump::Parent::of(
                        elite,
                        p.plan
                            .cma
                            .and_then(|c| self.cma_emitters.get(c))
                            .is_some_and(CmaEmitter::optimizing),
                    )
                })
                .collect()
        });
        let mut births: Vec<Birth> = planned
            .iter()
            .map(|p| Birth {
                emitter: p.plan.emitter,
                cma: p.plan.cma,
                parent_id: p.parent_id,
                mate: p.plan.mate.is_some(),
                protection: p.protection,
            })
            .collect();
        // Reseeded elites first, then the children.
        let mut lead: Vec<(usize, Creature)> = Vec::new();
        if !self.reseed.is_empty() {
            for (k, &slot) in slots.iter().enumerate() {
                if let Some(elite) = self.reseed_for_slot(slot) {
                    lead.push((k, elite));
                    births[k] = Birth::RANDOM;
                }
            }
        }
        // After a world change the old champions keep breeding in their
        // island's own slots for a few generations.
        if self.generation < self.refuge.until {
            let islands = island_count();
            let reseeded: std::collections::HashSet<usize> = lead.iter().map(|&(k, _)| k).collect();
            for (k, &slot) in slots.iter().enumerate() {
                if reseeded.contains(&k) || qd::is_nursery_slot(slot, islands) {
                    continue;
                }
                let island = qd::island_of_slot(slot, islands);
                if let Some(child) =
                    self.refuge
                        .child(island, slot, &cfg, self.generation, self.breed_round)
                {
                    lead.push((k, child));
                    births[k] = Birth::RANDOM;
                }
            }
            lead.sort_by_key(|&(k, _)| k);
        }
        // The founder bank breeds in the main islands' own slots, and the hall
        // of fame in the hub's.
        if !self.founders.is_empty() || !self.hall.is_empty() {
            let islands = island_count();
            let hub = hub_island();
            let taken: std::collections::HashSet<usize> = lead.iter().map(|&(k, _)| k).collect();
            for (k, &slot) in slots.iter().enumerate() {
                let island = qd::island_of_slot(slot, islands);
                if taken.contains(&k) || qd::is_nursery_slot(slot, islands) || qd::is_wild(island) {
                    continue;
                }
                let mut rng = evolution::Rng::stream(
                    cfg.seed ^ 0x666f_756e_64,
                    self.generation,
                    self.breed_round,
                    slot,
                );
                let draw = rng.unit();
                let parent = if !self.founders.is_empty() && draw < FOUNDER_SHARE {
                    self.founders[rng.index(self.founders.len())].clone()
                } else if island == hub
                    && !self.hall.is_empty()
                    && draw < FOUNDER_SHARE + HALL_SHARE
                {
                    self.hall[rng.index(self.hall.len())].clone()
                } else {
                    continue;
                };
                let scale = if rng.unit() < 0.1 { 2.0 } else { 0.75 };
                let mut child = evolution::mutate_locally(parent, &cfg, &mut rng, scale);
                if rng.unit() < 0.3 {
                    evolution::structural_mutation_any(&mut child, &cfg, &mut rng);
                }
                child.id = evolution::bred_id(self.breed_round, slot);
                lead.push((k, child));
                births[k] = Birth::RANDOM;
            }
            lead.sort_by_key(|&(k, _)| k);
        }
        // Wild champions in the hub's pen breed in the hub's own slots.
        if !self.pen.is_empty() {
            let islands = island_count();
            let hub = hub_island();
            let taken: std::collections::HashSet<usize> = lead.iter().map(|&(k, _)| k).collect();
            for (k, &slot) in slots.iter().enumerate() {
                if taken.contains(&k)
                    || qd::is_nursery_slot(slot, islands)
                    || qd::island_of_slot(slot, islands) != hub
                {
                    continue;
                }
                let mut rng =
                    evolution::Rng::stream(cfg.seed ^ 0x7065_6e, self.generation, self.breed_round, slot);
                if rng.unit() >= PEN_SHARE {
                    continue;
                }
                let parent = self.pen[rng.index(self.pen.len())].0.clone();
                let scale = if rng.unit() < 0.1 { 2.0 } else { 0.75 };
                let mut child = evolution::mutate_locally(parent, &cfg, &mut rng, scale);
                if rng.unit() < 0.3 {
                    evolution::structural_mutation_any(&mut child, &cfg, &mut rng);
                }
                child.id = evolution::bred_id(self.breed_round, slot);
                lead.push((k, child));
                births[k] = Birth::RANDOM;
            }
            lead.sort_by_key(|&(k, _)| k);
        }
        let mut taken = vec![false; count];
        for &(k, _) in &lead {
            taken[k] = true;
        }
        let positions: Vec<usize> = (0..count).filter(|&k| !taken[k]).collect();
        let bred_slots: Vec<usize> = positions.iter().map(|&k| slots[k]).collect();
        let bred_plans: Vec<CandidatePlan> = positions.iter().map(|&k| planned[k].plan).collect();
        // The arena's memory is reused when no unit or save still holds it;
        // otherwise its sizes guide a new one.
        let (mut population, hint) = match Arc::try_unwrap(arena) {
            Ok(population) => (population, None),
            Err(shared) => (Population::default(), Some(shared)),
        };
        let late = population.breed(
            count,
            hint.as_deref(),
            &mut lead,
            &self.islands,
            &self.cma_emitters,
            &bred_plans,
            &bred_slots,
            &positions,
            &cfg,
            self.generation,
            self.breed_round,
        );
        // Each creature's flags for its trial: the audit lane, and the
        // exemption of nurseries and immigrants from the early rungs.
        // The median fitness of each arena's behavior elites: a parent
        // above it is a strong one.
        let medians: Vec<f32> = if cfg.rungs.is_some() {
            self.islands
                .iter()
                .map(|archive| {
                    let mut v: Vec<f32> = archive
                        .entries
                        .iter()
                        .filter(|e| !qd::is_morphology_niche(&e.niche))
                        .map(|e| e.fitness)
                        .collect();
                    if v.is_empty() {
                        f32::NEG_INFINITY
                    } else {
                        let mid = v.len() / 2;
                        *v.select_nth_unstable_by(mid, f32::total_cmp).1
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        population.flags.clear();
        population.flags.extend((0..count).map(|k| {
            let slot = first + k;
            let mut flags = 0u8;
            if crate::rungs::is_audit(cfg.seed, self.breed_round, slot)
                && !qd::is_wild(qd::island_of_slot(slot, island_count()))
            {
                flags |= crate::rungs::AUDIT;
            }
            let arenas = self.islands.len().max(arena_count());
            let arena = qd::arena_of_slot(slot, arenas);
            // A child whose parent is a strong elite that the rules would
            // stop skips those rungs.
            if let Some(rules) = &cfg.rungs {
                let parent = births[k].parent_id.and_then(|id| self.lineage.get(&id));
                let strong = parent
                    .is_some_and(|a| medians.get(arena).is_none_or(|&median| a.fitness >= median));
                flags |= crate::rungs::parent_exemptions(rules, parent.map(|a| &a.rung), strong);
            }
            if births[k].emitter == Emitter::Restart {
                flags |= crate::rungs::EXEMPT;
            }
            // A nursery body is exempt from the early rungs and held to the
            // screen bar of its own kind.
            if qd::is_reshaped_arena(arena, arenas) {
                flags |= crate::rungs::EXEMPT | crate::rungs::RESHAPED;
            } else if arena >= island_count() {
                flags |= crate::rungs::EXEMPT | crate::rungs::YOUNG;
            }
            flags
        }));
        if let (Some(parents), Some(dump)) = (dump_parents, &self.dump) {
            let reseeded: Vec<usize> = lead.iter().map(|&(k, _)| k).collect();
            dump.lock().unwrap_or_else(|e| e.into_inner()).bred(
                first,
                &population,
                &births,
                parents,
                &reseeded,
            );
        }
        let total = started.elapsed();
        let add = |k: usize, d: std::time::Duration| {
            BREED_NANOS[k].fetch_add(d.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        };
        // Children are written into the arena as they are bred, so the
        // write stage is part of emitting.
        add(0, planned_at);
        add(1, total.saturating_sub(planned_at));
        BREED_LATE.fetch_add(late as u64, std::sync::atomic::Ordering::Relaxed);
        if hint.is_some() {
            BREED_LATE.fetch_add(1 << 32, std::sync::atomic::Ordering::Relaxed);
        }
        Block {
            first,
            population: Arc::new(population),
            births,
            config: Arc::new(cfg),
        }
    }
    /// The generation boundary (every `population` evaluations): records
    /// history, graduates the nurseries, migrates to the hub, and applies
    /// queued settings and the autochange ladder.
    fn end_generation(&mut self) -> Result<()> {
        let started = std::time::Instant::now();
        // The audit lane judges the rules this generation ran with and fits
        // the next generation's.
        let rules = self
            .rungs
            .boundary(self.config.rungs, self.global_stalled());
        let failed = std::mem::take(&mut self.failed);
        self.push_archive_stats(failed);
        self.prune_lineage();
        self.generation += 1;
        self.refuge.review(&self.islands, self.generation);
        let generation = self.generation;
        self.pen.retain(|&(_, until)| until > generation);
        if self.islands.len() == arena_count() {
            let mut best: HashMap<u64, (f32, &StoredCreature)> = HashMap::new();
            for archive in self.islands.iter().take(qd::MAIN_ISLANDS) {
                for (i, e) in archive.entries.iter().enumerate() {
                    if qd::is_morphology_niche(&e.niche) {
                        continue;
                    }
                    let slot = best.entry(archive.plan_key(i)).or_insert((e.fitness, &e.creature));
                    if e.fitness > slot.0 {
                        *slot = (e.fitness, &e.creature);
                    }
                }
            }
            let mut hall: Vec<(u64, f32, &StoredCreature)> =
                best.into_iter().map(|(k, (f, c))| (k, f, c)).collect();
            hall.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            hall.truncate(HALL_PLANS);
            self.hall = hall.into_iter().map(|(_, _, c)| c.unpack()).collect();
        }
        // Migrants that never took a hub cell are forgotten after a while.
        if self.wild_exports.len() > 200_000 {
            self.wild_exports.clear();
        }
        // Body novelty, once a generation.
        self.islands
            .par_iter_mut()
            .for_each(QdArchive::refresh_traits);
        self.refine_archives();
        self.graduate_nurseries();
        self.refresh_reshaped_scores();
        self.migrate_islands();
        self.step_stones();
        let mut cfg = self.pending.take().unwrap_or_else(|| self.config.clone());
        cfg.validate()?;
        ensure!(
            cfg.population == self.config.population,
            "Population changes need a new experiment"
        );
        crate::environment::advance_autochange(&mut cfg, self.generation);
        let world_changed = fitness_context_changed(&self.config, &cfg);
        if world_changed {
            self.reset_search_context();
        }
        cfg.screen = self.next_screen(cfg.duration);
        cfg.rungs = if world_changed { None } else { rules };
        self.config = cfg;
        self.start_dump()?;
        if self.dump_breeding() {
            // The dump generation runs every trial in full.
            if let Some(screen) = &mut self.config.screen {
                screen.bar = f32::NEG_INFINITY;
            }
            self.config.rungs = None;
        }
        self.evaluation_seconds = 0.0;
        if std::env::var_os("EVOLUTION_PROFILE_BREED").is_some() {
            eprintln!(
                "Generation boundary: {:.3} s, ring {:.0} MiB",
                started.elapsed().as_secs_f64(),
                self.ring_bytes() as f64 / 1048576.0
            );
        }
        Ok(())
    }
    /// Applies settings at the next generation boundary.
    pub fn update_config(&mut self, cfg: Config) -> Result<()> {
        self.update_config_at(cfg, false)
    }
    /// Applies settings now. A world change resets the search context at
    /// once. Blocks in flight from the old world are recognized by their own
    /// settings and enter no archive.
    pub fn update_config_now(&mut self, cfg: Config) -> Result<()> {
        self.update_config_at(cfg, true)
    }
    fn update_config_at(&mut self, mut cfg: Config, now: bool) -> Result<()> {
        cfg.validate()?;
        // The autochange step advances in the worker, so a settings update must
        // never rewind a checkpoint-carrying counter to its stale copy.
        cfg.autochange_step = cfg.autochange_step.max(self.config.autochange_step);
        ensure!(
            cfg.population == self.config.population
                && cfg.seed == self.config.seed
                && cfg.random_seed == self.config.random_seed,
            "Population or seed changes require a new experiment"
        );
        ensure!(
            self.bodies_fit(&cfg),
            "Existing bodies exceed these limits; start a new experiment"
        );
        if now {
            self.pending = None;
            let world_changed = fitness_context_changed(&self.config, &cfg);
            if world_changed {
                self.reset_search_context();
            }
            cfg.screen = self.next_screen(cfg.duration);
            cfg.rungs = if world_changed {
                None
            } else {
                self.config.rungs
            };
            self.config = cfg;
        } else {
            self.pending = Some(cfg);
        }
        Ok(())
    }
    /// A meteor strike wipes out `share` of the elites in the global archive
    /// and in every island, chosen at random. Survivors and new offspring
    /// refill the emptied cells, which opens room for new kinds of movement.
    /// The lost elites become fossils so the strike can be undone. Returns how
    /// many elites were lost.
    pub fn meteor(&mut self, share: f32) -> usize {
        let mut rng = evolution::Rng::new(
            self.config.seed ^ 0x6d65_7465_6f72,
            self.generation,
            self.fossils.len(),
        );
        // The strike spares the fastest elite of each body plan, so the
        // survivors are the rare plans and the common ones thin out (Raup,
        // 1986, selective extinction).
        let mut strike = |archive: &mut QdArchive, island: Option<usize>| {
            let spared: std::collections::HashSet<u64> =
                plan_champions(archive).iter().map(|c| c.id).collect();
            let (kept, lost): (Vec<_>, Vec<_>) = std::mem::take(&mut archive.entries)
                .into_iter()
                .partition(|e| spared.contains(&e.creature.id) || rng.unit() >= share);
            archive.entries = kept;
            archive.rebuild_indices();
            lost.into_iter().map(move |elite| (island, elite))
        };
        let mut fossils: Vec<_> = strike(&mut self.archive, None).collect();
        for (index, island) in self.islands.iter_mut().enumerate() {
            fossils.extend(strike(island, Some(index)));
        }
        let lost = fossils.len();
        self.fossils.extend(fossils);
        self.radiate();
        lost
    }
    /// After a meteor or an extinction the survivors of each main island
    /// breed from the refuge for a few generations, as a radiation into the
    /// emptied cells (Lehman and Miikkulainen, 2015).
    fn radiate(&mut self) {
        let champions: Vec<Vec<Creature>> = self
            .islands
            .iter()
            .take(qd::MAIN_ISLANDS)
            .map(plan_champions)
            .collect();
        let before = vec![f32::NAN; champions.len()];
        self.refuge = Refuge::open(champions, before, self.generation);
    }
    /// An extinction wipes out the island whose best creature is slowest. An
    /// isolated island starts over from new random bodies, and the hub from
    /// its next copies, so a stalled island starts over from new designs (Lehman and Miikkulainen,
    /// 2015). The lost elites become fossils, so it can be undone. Returns how
    /// many elites were lost.
    pub fn extinction(&mut self) -> usize {
        let weakest = self
            .islands
            .iter()
            .enumerate()
            .take(island_count())
            .filter(|(_, island)| !island.entries.is_empty())
            .min_by(|a, b| a.1.best_fitness().total_cmp(&b.1.best_fitness()))
            .map(|(index, _)| index);
        let Some(index) = weakest else {
            return 0;
        };
        let lost = std::mem::take(&mut self.islands[index].entries);
        // The island starts over from new bodies, and climbs without classes
        // until it is old enough to be refined again.
        self.islands[index].set_refined(false);
        self.islands[index].rebuild_indices();
        if self.island_epoch.len() <= index {
            self.island_epoch.resize(index + 1, 0);
        }
        self.island_epoch[index] = self.generation;
        let count = lost.len();
        self.fossils
            .extend(lost.into_iter().map(|elite| (Some(index), elite)));
        self.radiate();
        count
    }
    /// Undoes meteor strikes: every fossil returns to its archive if its cell
    /// is empty or holds a slower elite. Returns how many came back.
    pub fn undo_meteor(&mut self) -> usize {
        let mut restored = 0;
        let mut touched = std::collections::BTreeSet::new();
        for (island, elite) in std::mem::take(&mut self.fossils) {
            let archive = match island {
                None => &mut self.archive,
                Some(index) => match self.islands.get_mut(index) {
                    Some(archive) => archive,
                    None => continue,
                },
            };
            let mut elite = elite;
            if !qd::is_morphology_niche(&elite.niche) {
                // The cell in the layout the archive has now.
                elite.niche = archive.cell_of(elite.descriptor);
            }
            match archive.slot_for(&elite.niche) {
                Some(slot) if archive.entries[slot].fitness < elite.fitness => {
                    archive.entries[slot] = elite;
                }
                Some(_) => continue,
                // Later fossils must see this cell as taken.
                None => archive.push_unscored(elite),
            }
            touched.insert(island);
            restored += 1;
        }
        for island in touched {
            match island {
                None => self.archive.rebuild_indices(),
                Some(index) => self.islands[index].rebuild_indices(),
            }
        }
        restored
    }
    /// Clears the archives after the world changed. Their scores no longer
    /// hold, but each island's creatures are queued to compete again under
    /// the new physics in that island's own slots. An archive that was
    /// refined starts again refined, so the re-tested elites keep the cells of
    /// their body classes: the archive refills with evolved bodies, not
    /// random ones, and a climb that spreads over many classes is not at stake.
    /// Only a save written before the first elite re-enters forgets this,
    /// because a save tells a layout by the cells its elites hold.
    fn reset_search_context(&mut self) {
        self.reseed.clear();
        self.wild_exports.clear();
        // The nurseries start over; only the islands' creatures are re-tested.
        let mut refined = Vec::new();
        // The refuge takes each island's best. A second change while it lasts
        // keeps the older champions where the islands have none left.
        let mut champions = std::mem::take(&mut self.refuge.champions);
        champions.resize_with(qd::MAIN_ISLANDS, Vec::new);
        // The wild islands live in worlds of their own, which the player's
        // change leaves alone: they keep their archives and nurseries.
        let wild: Vec<(usize, QdArchive)> = if self.islands.len() == arena_count() {
            (0..arena_count())
                .filter(|&a| qd::is_wild(a % island_count()))
                .map(|a| (a, std::mem::take(&mut self.islands[a])))
                .collect()
        } else {
            Vec::new()
        };
        let mut before = vec![f32::NAN; qd::MAIN_ISLANDS];
        for (index, island) in self.islands.iter_mut().take(qd::MAIN_ISLANDS).enumerate() {
            if !island.entries.is_empty() {
                champions[index] = plan_champions(island);
                before[index] = island.best_fitness();
            }
            refined.push(island.refined());
            for elite in std::mem::take(&mut island.entries) {
                self.reseed.push(index, elite.creature.unpack());
            }
        }
        self.archive = QdArchive::starting_global();
        // Fossils are old-world elites: undoing a meteor must not bring them
        // back into the new world's archives.
        self.fossils.clear();
        self.islands = if refined.contains(&true) || !wild.is_empty() {
            new_islands(&refined)
        } else {
            Vec::new()
        };
        for (a, archive) in wild {
            self.islands[a] = archive;
        }
        self.island_progress.clear();
        self.graduations.clear();
        self.reshaped_graduations.clear();
        self.last_migration = None;
        self.emitter_stats = [EmitterStats::default(); qd::EMITTER_COUNT];
        self.cma_emitters.clear();
        // Distances measured in the old world say nothing about the new one,
        // and neither do the audit rows: the rungs disarm and refit.
        self.screen_window.clear();
        self.young_window.clear();
        self.reshaped_window.clear();
        self.clade_rarity = (u32::MAX, Vec::new());
        self.rungs.clear();
        self.config.rungs = None;
        self.refuge = Refuge::open(champions, before, self.generation);
    }
    /// Starts the generation dump (`EVOLUTION_DUMP_GENERATION`) at the
    /// boundary it names: every island elite is queued for a re-run, the
    /// screen bar is off for a generation's worth of blocks, and the rows go
    /// out as those blocks are absorbed.
    fn start_dump(&mut self) -> Result<()> {
        let Some((generation, path)) = dump::target() else {
            return Ok(());
        };
        if self.dump.is_some()
            || generation.is_some_and(|g| g != self.generation)
            || dump::STARTED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(());
        }
        let bar = self.config.screen.map_or(f32::NEG_INFINITY, |s| s.bar);
        let mut elites = Vec::new();
        for (arena, island) in self.islands.iter().enumerate() {
            elites.extend(
                island
                    .entries
                    .iter()
                    .map(|e| dump::Elite::of(arena as u8, e)),
            );
        }
        elites.extend(
            self.archive
                .entries
                .iter()
                .map(|e| dump::Elite::of(u8::MAX, e)),
        );
        // The island elites run again in this generation (their own island's
        // slots, as after a world change), so their rung distances are known.
        let mut reruns = std::collections::HashSet::new();
        for (island, archive) in self.islands.iter().enumerate().take(island_count()) {
            for elite in &archive.entries {
                reruns.insert(elite.creature.id);
                self.reseed.push(island, elite.creature.unpack());
            }
        }
        let d = dump::Dump::create(&path, self, bar, elites, reruns, self.blocks.len())?;
        eprintln!(
            "Generation dump: generation {} to {}, {} elites queued for a re-run",
            self.generation,
            path.display(),
            d.rerun_count()
        );
        evolution::record_operators(true);
        self.dump = Some(Arc::new(std::sync::Mutex::new(d)));
        Ok(())
    }
    /// Whether blocks bred now belong to the generation dump.
    fn dump_breeding(&self) -> bool {
        self.dump
            .as_ref()
            .is_some_and(|d| d.lock().unwrap_or_else(|e| e.into_inner()).breeding())
    }
    /// Block `k` was just bred: files it under the dump if it belongs to it.
    fn note_dump_block(&mut self, k: usize) {
        if let Some(dump) = &self.dump {
            let mut d = dump.lock().unwrap_or_else(|e| e.into_inner());
            d.assign(k);
            if !d.breeding() {
                evolution::record_operators(false);
            }
        }
    }
    /// Whether every body in the ring and the archive fits `cfg`'s limits.
    fn bodies_fit(&self, cfg: &Config) -> bool {
        self.blocks
            .iter()
            .flat_map(|b| &b.population.genomes)
            .all(|g| g.node_count <= cfg.max_nodes && g.muscle_count <= cfg.max_muscles)
            && self.archive.entries.iter().all(|elite| {
                elite.creature.node_count() <= cfg.max_nodes
                    && elite.creature.muscle_count() <= cfg.max_muscles
            })
    }
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        ensure!(
            !self.blocks.is_empty() && self.cursor < self.blocks.len(),
            "Invalid ring"
        );
        let mut next = 0;
        for block in &self.blocks {
            ensure!(
                block.first == next && block.births.len() == block.len() && !block.is_empty(),
                "Invalid ring block"
            );
            next += block.len();
            block.population.validate(&Config {
                population: block.len(),
                ..(*block.config).clone()
            })?;
        }
        ensure!(
            next == self.ring.len(self.config.population),
            "Invalid ring size"
        );
        ensure!(
            self.evaluated < self.config.population.max(1),
            "Invalid evaluation progress"
        );
        if let Some(cfg) = &self.pending {
            cfg.validate()?;
            ensure!(
                cfg.population == self.config.population
                    && cfg.seed == self.config.seed
                    && cfg.random_seed == self.config.random_seed
                    && self.bodies_fit(cfg),
                "Invalid pending settings"
            );
        }
        ensure!(
            self.evaluation_seconds.is_finite() && self.evaluation_seconds >= 0.0,
            "Invalid evaluation time"
        );
        ensure!(
            self.history.len() <= self.generation as usize + 1,
            "Invalid history length"
        );
        ensure!(
            self.qd_version == qd::VERSION
                && self.archive.entries.len() <= self.archive.capacity()
                && self.archive.behavior_count() <= self.archive.limit()
                && self.archive.morphology_count() <= qd::MORPHOLOGY_LIMIT
                && self.cma_emitters.len() <= qd::CMA_LIMIT
                && self
                    .archive
                    .entries
                    .iter()
                    .all(|elite| elite.fitness.is_finite() && elite.fitness > FAILED),
            "Invalid QD archive state"
        );
        ensure!(
            (self.islands.is_empty() || self.islands.len() == arena_count())
                && self.reseed.fits(island_count())
                && self.cma_emitters.iter().all(|c| c.island < arena_count()),
            "Invalid island state"
        );
        for (index, stats) in self.history.iter().enumerate() {
            stats.config.validate()?;
            ensure!(
                stats.generation as usize == index
                    && stats.population == stats.config.population
                    && stats.failed <= stats.population
                    && stats.archive_cells <= qd::HISTORICAL_ARCHIVE_LIMIT
                    && stats.qd_score.is_finite()
                    && stats.qd_score >= 0.0
                    && stats.archive_coverage.is_finite()
                    && (0.0..=1.0).contains(&stats.archive_coverage),
                "Invalid historical generation"
            );
            ensure!(
                stats.percentiles.len() == PERCENTILES.len()
                    && stats.percentiles.iter().all(|v| v.is_finite())
                    && [stats.best, stats.median, stats.worst, stats.mean]
                        .iter()
                        .all(|v| v.is_finite())
                    && stats.seconds.is_finite()
                    && stats.seconds >= 0.0,
                "Invalid historical statistics"
            );
            ensure!(
                stats.representatives.len() == 3,
                "Missing historical representatives"
            );
            if stats.archive_cells == 0 {
                ensure!(
                    stats.histogram.iter().map(|(_, n)| *n as u64).sum::<u64>()
                        + stats.failed as u64
                        == stats.population as u64,
                    "Invalid histogram totals"
                );
                ensure!(
                    stats.species.iter().map(|(_, _, n)| *n as u64).sum::<u64>()
                        == stats.population as u64,
                    "Invalid body-type totals"
                );
            } else {
                ensure!(
                    stats.histogram.iter().map(|(_, n)| *n as u64).sum::<u64>()
                        == stats.archive_cells as u64
                        && stats.species.iter().map(|(_, _, n)| *n as u64).sum::<u64>()
                            == stats.archive_cells as u64,
                    "Invalid archive statistics totals"
                );
            }
            let mut representatives = Population::default();
            for creature in &stats.representatives {
                representatives.push(creature.clone());
            }
            let representative_config = Config {
                population: 3,
                ..stats.config.clone()
            };
            // Historical snapshots retain the bone-length limits in effect
            // when they were recorded; they are never evaluated as candidates.
            representatives.validate_with_max_bone(&representative_config, 12.0, true)?;
        }
        Ok(())
    }
}
// The file starts with the magic, then a small uncompressed header
// (`SaveHeader`), so the game can turn down a save it cannot use before it
// reads gigabytes. The body keeps only the archives and the search state
// (`SmallSave`). A loaded game breeds its population from the archives again
// Any other magic is an older format
// and is turned down.
pub(crate) mod dump;

const MAGIC: &[u8; 8] = b"EVORUST8";

/// What a save holds: the archives and the search state, without the
/// population, its scores, or anything bred for the generation in progress.
/// It holds the islands and the nurseries of new bodies, and none of the
/// nurseries of reshaped bodies: they refill from the bodies the islands
/// turn away, and a save stays as small as it was.
#[derive(Serialize)]
struct SmallSave<'a> {
    config: &'a Config,
    pending: &'a Option<Config>,
    generation: u32,
    history: &'a [Stats],
    archive: &'a QdArchive,
    emitter_stats: &'a [EmitterStats; qd::EMITTER_COUNT],
    cma_emitters: Vec<&'a CmaEmitter>,
    qd_version: u32,
    breed_round: u64,
    islands: &'a [QdArchive],
    lineage: SavedLineage<'a>,
    island_progress: &'a [(f32, u32)],
    reseed: &'a Reseed,
    ring: RingShape,
    audit: &'a crate::rungs::Audit,
}
#[derive(Deserialize)]
struct SmallLoad {
    config: Config,
    pending: Option<Config>,
    generation: u32,
    history: Vec<Stats>,
    archive: QdArchive,
    emitter_stats: [EmitterStats; qd::EMITTER_COUNT],
    cma_emitters: Vec<CmaEmitter>,
    qd_version: u32,
    breed_round: u64,
    islands: Vec<QdArchive>,
    lineage: HashMap<u64, Ancestor>,
    island_progress: Vec<(f32, u32)>,
    reseed: Reseed,
    ring: RingShape,
    audit: crate::rungs::Audit,
}

/// The lineage a save keeps, written in the layout of `HashMap<u64,
/// Ancestor>`. Every living elite keeps its own record, because breeding
/// reads its distance and early-rung features, but not its creature: the
/// archive holds that, and loading puts it back. The ancestors of the global
/// archive's elites and of each island's fastest few keep their creatures
/// back to `ANCESTRY_DEPTH`, which is as far as anything shows them. The
/// ancestors of the other island elites are left out.
struct SavedLineage<'a> {
    lineage: &'a HashMap<u64, Ancestor>,
    /// Which records to write, and whether each writes its creature.
    keep: HashMap<u64, bool>,
}
impl<'a> SavedLineage<'a> {
    /// The lineage of the global archive and of the first `held` archives
    /// of the experiment, the ones a save holds.
    fn of(e: &'a Experiment, held: usize) -> Self {
        let archives = || std::iter::once(&e.archive).chain(e.islands[..held].iter());
        let mut keep: HashMap<u64, bool> = HashMap::new();
        for elite in archives().flat_map(|archive| &archive.entries) {
            keep.insert(elite.creature.id, false);
        }
        let mut shown: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
        for island in e.islands.iter().take(island_count()) {
            let mut fastest: Vec<&qd::Elite> = island.entries.iter().collect();
            fastest.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            shown.extend(fastest.iter().take(ISLAND_LEADERS).map(|x| x.creature.id));
        }
        for start in shown {
            let mut current = e.lineage.get(&start).and_then(|a| a.parent);
            for _ in 0..ANCESTRY_DEPTH {
                let Some(id) = current else { break };
                let Some(ancestor) = e.lineage.get(&id) else {
                    break;
                };
                if keep.get(&id) == Some(&true) {
                    // The rest of this chain is already kept.
                    break;
                }
                // A living elite keeps its record without its creature.
                keep.entry(id).or_insert(true);
                current = ancestor.parent;
            }
        }
        Self {
            lineage: &e.lineage,
            keep,
        }
    }
}
impl Serialize for SavedLineage<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        /// `Ancestor`, field for field.
        #[derive(Serialize)]
        struct Record<'b> {
            parent: Option<u64>,
            creature: &'b StoredCreature,
            fitness: f32,
            generation: u32,
            change: &'b str,
            rung: &'b [u16; 2 * crate::rungs::FEATURES],
        }
        let none = StoredCreature::default();
        let records: Vec<(&u64, Record)> = self
            .keep
            .iter()
            .filter_map(|(id, &with_creature)| {
                let a = self.lineage.get(id)?;
                Some((
                    id,
                    Record {
                        parent: a.parent,
                        creature: if with_creature { &a.creature } else { &none },
                        fitness: a.fitness,
                        generation: a.generation,
                        change: &a.change,
                        rung: &a.rung,
                    },
                ))
            })
            .collect();
        serializer.collect_map(records)
    }
}
impl<'a> SmallSave<'a> {
    fn of(e: &'a Experiment) -> Self {
        // The archives the save holds: each island and its nursery of new
        // bodies.
        let held = e.islands.len().min(island_count() * 2);
        let islands = &e.islands[..held];
        Self {
            config: &e.config,
            pending: &e.pending,
            generation: e.generation,
            history: &e.history,
            archive: &e.archive,
            emitter_stats: &e.emitter_stats,
            cma_emitters: e.cma_emitters.iter().filter(|c| c.island < held).collect(),
            qd_version: e.qd_version,
            breed_round: e.breed_round,
            islands,
            lineage: SavedLineage::of(e, held),
            island_progress: &e.island_progress,
            reseed: &e.reseed,
            ring: e.ring,
            audit: &e.rungs,
        }
    }
}
impl SmallLoad {
    /// The game the save describes, at the start of its saved generation.
    /// Its ring is bred from the archives, as the game would have bred it;
    /// without elites it starts with new random bodies.
    /// `saved_version` is the version the file's header names: an older one
    /// that still loads (`qd::loadable`) gets its archives re-binned.
    fn into_experiment(self, saved_version: u32) -> Result<Experiment> {
        let mut e = Experiment::empty(self.config);
        e.pending = self.pending;
        e.generation = self.generation;
        e.history = repair_history(self.history, self.generation);
        e.archive = self.archive;
        e.archive.set_global(true);
        e.emitter_stats = self.emitter_stats;
        e.cma_emitters = self.cma_emitters;
        e.qd_version = self.qd_version;
        e.breed_round = self.breed_round;
        e.islands = self.islands;
        e.lineage = self.lineage;
        e.island_progress = self.island_progress;
        // A save holds each island and its nursery of new bodies, and the
        // nurseries of reshaped bodies start empty (`SmallSave`).
        if e.islands.len() == island_count() * 2 {
            e.islands.resize_with(arena_count(), new_reshaped_nursery);
            if e.island_progress.len() == island_count() * 2 {
                e.island_progress
                    .resize(arena_count(), (f32::NEG_INFINITY, e.generation));
            }
        }
        e.reseed = self.reseed;
        e.rungs = self.audit;
        ensure!(
            self.ring.block > 0 && self.ring.blocks > 0,
            "Invalid ring shape"
        );
        // The ring keeps the shape it was saved with, so a resumed game
        // continues the search the uninterrupted one would have run.
        e.ring = self.ring;
        ensure!(
            e.island_progress.len() <= arena_count()
                && (e.island_progress.is_empty() || e.island_progress.len() == e.islands.len())
                && e.island_progress.iter().all(|&(fitness, generation)| {
                    (fitness.is_finite() || fitness == f32::NEG_INFINITY)
                        && generation <= e.generation
                }),
            "Invalid checkpoint optimizer progress"
        );
        ensure!(
            (e.islands.is_empty() || e.islands.len() == arena_count())
                && e.reseed.fits(island_count())
                && e.cma_emitters.iter().all(|c| c.island < arena_count()),
            "Invalid island state"
        );
        let refinable =
            std::iter::once(&mut e.archive).chain(e.islands.iter_mut().take(island_count()));
        if saved_version == qd::VERSION {
            for archive in refinable {
                archive.rebuild_indices();
                archive.derive_refined();
            }
            for nursery in e.islands.iter_mut().skip(island_count()) {
                nursery.rebuild_indices();
            }
        } else {
            // The archives were saved under another layout: each elite moves
            // to its cell now. A version 54 archive with elites in body
            // classes was refined. Optimizers keep their body plan's state;
            // the CMA emitters of single cells start over.
            for archive in refinable {
                if saved_version >= 54 {
                    archive.rebuild_indices();
                    archive.derive_refined();
                }
                archive.rebin();
            }
            for nursery in e.islands.iter_mut().skip(island_count()) {
                nursery.rebin();
            }
            e.cma_emitters.retain(CmaEmitter::optimizing);
        }
        // The global archive is always refined: it never breeds, so it has no
        // climb to protect. The islands are refined at the next generation
        // boundary if they are old enough.
        if !e.archive.refined() {
            e.archive.set_refined(true);
            e.archive.rebin();
        }
        // The lineage records of living elites were saved without a creature.
        for elite in std::iter::once(&e.archive)
            .chain(&e.islands)
            .flat_map(|archive| &archive.entries)
        {
            if let Some(record) = e.lineage.get_mut(&elite.creature.id)
                && record.creature.is_empty()
            {
                record.creature = elite.creature.clone();
            }
        }
        // The screen bar is not saved: the resumed generation runs every
        // trial in full until it has set a new one.
        e.config.screen = e.next_screen(e.config.duration);
        // The rungs' rules are not saved either: they are the window's fit.
        e.config.rungs = e.rungs.fit(e.global_stalled());
        let shared = Arc::new(e.config.clone());
        let elites =
            e.archive.entries.len() + e.islands.iter().map(|i| i.entries.len()).sum::<usize>();
        for (first, count) in e.ring.ranges(e.ring.len(e.config.population)) {
            let block = if elites == 0 && e.reseed.is_empty() {
                // Nothing to breed from: the new bodies of a new game.
                Block {
                    first,
                    population: Arc::new(evolution::random_block(&e.config, first, count)),
                    births: vec![Birth::RANDOM; count],
                    config: Arc::clone(&shared),
                }
            } else {
                // Plans sample every island's archive; an empty one breeds
                // new random bodies.
                Block {
                    config: Arc::clone(&shared),
                    ..e.breed_block(first, count, Arc::default())
                }
            };
            e.blocks.push(block);
        }
        e.validate()?;
        Ok(e)
    }
}

fn fitness_context_changed(old: &Config, new: &Config) -> bool {
    old.physics_differs(new)
}
/// Autosaves kept in `dir`: the newest `keep` `seed-*-auto.evo` files stay,
/// older ones are deleted, and so are `.evo.tmp` files that an interrupted
/// save left behind more than ten minutes ago. Files the player saved under
/// other names are never touched. Returns how many files were removed.
pub fn rotate_autosaves(dir: &Path, keep: usize) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let now = std::time::SystemTime::now();
    let mut autosaves = Vec::new();
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if name.ends_with(".evo.tmp") {
            let stale = now
                .duration_since(modified)
                .is_ok_and(|age| age.as_secs() > 600);
            if stale && std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        } else if name.starts_with("seed-") && name.ends_with("-auto.evo") {
            autosaves.push((modified, path));
        }
    }
    autosaves.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    for (_, path) in autosaves.into_iter().skip(keep) {
        if std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}
/// What a save's header says, read without decoding the save.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveHeader {
    pub qd_version: u32,
    pub generation: u32,
    pub population: u64,
}
impl SaveHeader {
    const BYTES: usize = 16;
    fn of(experiment: &Experiment) -> Self {
        Self {
            qd_version: experiment.qd_version,
            generation: experiment.generation,
            population: experiment.config.population as u64,
        }
    }
    fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut bytes = [0; Self::BYTES];
        bytes[..4].copy_from_slice(&self.qd_version.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.generation.to_le_bytes());
        bytes[8..].copy_from_slice(&self.population.to_le_bytes());
        bytes
    }
    fn from_bytes(bytes: [u8; Self::BYTES]) -> Self {
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        Self {
            qd_version: word(0),
            generation: word(4),
            population: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
        }
    }
}

/// Progress of a load or save that runs on another thread: file bytes read
/// or written so far, the file size when known, and a flag that stops it.
#[derive(Default)]
pub struct Progress {
    pub done: std::sync::atomic::AtomicU64,
    pub total: std::sync::atomic::AtomicU64,
    pub cancel: std::sync::atomic::AtomicBool,
}

/// A file that counts its bytes into a `Progress` and fails once cancelled.
struct Counted<'a, T> {
    inner: T,
    progress: Option<&'a Progress>,
}
impl<T> Counted<'_, T> {
    fn count(&self, bytes: usize) -> std::io::Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(progress) = self.progress {
            if progress.cancel.load(Relaxed) {
                return Err(std::io::Error::other("cancelled"));
            }
            progress.done.fetch_add(bytes as u64, Relaxed);
        }
        Ok(())
    }
}
impl<T: Read> Read for Counted<'_, T> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count(n)?;
        Ok(n)
    }
}
impl<T: Write> Write for Counted<'_, T> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count(n)?;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Reads a save's header. Saves from before the header are from older game
/// versions, whose creatures were scored under other physics.
pub fn peek(path: &Path) -> Result<SaveHeader> {
    let mut file = File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic)
        .with_context(|| format!("{} is not a save of this game", path.display()))?;
    reject_other_formats(path, &magic)?;
    let mut header = [0; SaveHeader::BYTES];
    file.read_exact(&mut header)
        .with_context(|| format!("{} is cut short", path.display()))?;
    Ok(SaveHeader::from_bytes(header))
}

/// Turns down a file that is not a save in the current format.
fn reject_other_formats(path: &Path, magic: &[u8; 8]) -> Result<()> {
    if magic == MAGIC {
        return Ok(());
    }
    ensure!(
        magic.starts_with(b"EVORUST"),
        "{} is not a save of this game",
        path.display()
    );
    anyhow::bail!(
        "{} was saved by an older version of the game, under older physics. It cannot be loaded; start a new population instead.",
        path.display()
    )
}

/// The header of a save the game can load now, or a message saying why not.
pub fn check(path: &Path) -> Result<SaveHeader> {
    let header = peek(path)?;
    ensure_current_version(path, &header)?;
    Ok(header)
}

fn ensure_current_version(path: &Path, header: &SaveHeader) -> Result<()> {
    ensure!(
        qd::loadable(header.qd_version),
        "{} was saved under physics version {}, and this game uses version {}. Its scores no longer hold, so it cannot be loaded; start a new population instead.",
        path.display(),
        header.qd_version,
        qd::VERSION
    );
    Ok(())
}

pub fn save(path: &Path, experiment: &Experiment) -> Result<()> {
    save_with_progress(path, experiment, None)
}

/// Saves through a temporary file that is renamed only once complete; a
/// failed or cancelled save removes it.
pub fn save_with_progress(
    path: &Path,
    experiment: &Experiment,
    progress: Option<&Progress>,
) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("evo.tmp");
    let written = (|| -> Result<()> {
        let file = File::create(&tmp)?;
        let mut out = BufWriter::new(Counted {
            inner: file,
            progress,
        });
        out.write_all(MAGIC)?;
        out.write_all(&SaveHeader::of(experiment).to_bytes())?;
        let mut encoder = zstd::stream::write::Encoder::new(out, 3)?;
        encoder.include_checksum(true)?;
        // The same creature sits in the global archive and in an island, far
        // apart in the stream: matching over 128 MB made a save 28% smaller.
        encoder.long_distance_matching(true)?;
        encoder.window_log(27)?;
        // bincode writes field by field; a buffer turns each write into a
        // copy instead of a call into the compressor.
        let mut buffered = BufWriter::with_capacity(1 << 20, encoder);
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize_into(&mut buffered, &SmallSave::of(experiment))?;
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize_into(&mut buffered, &experiment.last_migration)?;
        let encoder = buffered.into_inner().map_err(|error| error.into_error())?;
        let mut out = encoder.finish()?;
        out.flush()?;
        out.get_ref().inner.sync_all()?;
        Ok(())
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    std::fs::rename(&tmp, path)?;
    // Unix permits opening directories to persist the rename. Windows rejects
    // File::open on a directory; the checkpoint file itself was synced above.
    #[cfg(unix)]
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}
/// What the start of a checkpoint says about it: enough to list a save
/// without loading its population.
pub struct SaveSummary {
    pub generation: u32,
    pub config: Config,
}
/// Reads the settings and generation at the start of a checkpoint in the
/// current format. The payload begins with them, so only a few kilobytes are
/// decompressed. Other formats, other physics versions and unreadable files
/// give None.
pub fn summary(path: &Path) -> Option<SaveSummary> {
    #[derive(Deserialize)]
    struct Head {
        config: Config,
        _pending: Option<Config>,
        generation: u32,
    }
    let mut file = BufReader::new(File::open(path).ok()?);
    let mut magic = [0; 8];
    file.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }
    let mut header = [0; SaveHeader::BYTES];
    file.read_exact(&mut header).ok()?;
    if !qd::loadable(SaveHeader::from_bytes(header).qd_version) {
        return None;
    }
    let decoder = zstd::stream::read::Decoder::new(file).ok()?;
    let head: Head = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(1 << 20)
        .deserialize_from(decoder)
        .ok()?;
    Some(SaveSummary {
        generation: head.generation,
        config: head.config,
    })
}
pub fn load(path: &Path) -> Result<Experiment> {
    load_with_progress(path, None)
}

/// `load`, counting the file bytes read into `progress`.
/// One row per generation, in order, up to `generation`: a skipped
/// generation gets a copy of the row before it, and a repeated one is dropped.
fn repair_history(history: Vec<Stats>, generation: u32) -> Vec<Stats> {
    let mut out: Vec<Stats> = Vec::with_capacity(history.len());
    for stats in history {
        if stats.generation < out.len() as u32 || stats.generation > generation {
            continue;
        }
        while (out.len() as u32) < stats.generation {
            let mut fill = out.last().unwrap_or(&stats).clone();
            fill.generation = out.len() as u32;
            out.push(fill);
        }
        out.push(stats);
    }
    while !out.is_empty() && (out.len() as u32) < generation {
        let mut fill = out[out.len() - 1].clone();
        fill.generation = out.len() as u32;
        out.push(fill);
    }
    out
}

pub fn load_with_progress(path: &Path, progress: Option<&Progress>) -> Result<Experiment> {
    load_from(path, progress, false, None)
}

/// `load` for a diagnostic that continues a big save on a small machine
/// (`examples/search_ab.rs --load`): the game runs `population` creatures per
/// generation, so its ring is bred for that and holds no more.
#[doc(hidden)]
pub fn load_for_population(path: &Path, population: usize) -> Result<Experiment> {
    load_from(path, None, false, Some(population))
}

/// `load` for diagnostics that only breed from the archives
/// (`examples/breed_bench.rs`): a save of an older physics version loads
/// too, with the scores it measured then.
#[doc(hidden)]
pub fn load_any_version(path: &Path) -> Result<Experiment> {
    load_from(path, None, true, None)
}

/// `load` for diagnostics that only read the archives
/// (`examples/island_report.rs`): the ring is one block of 64 creatures, not
/// the saved ring, so a 3M save loads in a few hundred MB.
#[doc(hidden)]
pub fn load_archives(path: &Path) -> Result<Experiment> {
    load_from(path, None, false, Some(64))
}

fn load_from(
    path: &Path,
    progress: Option<&Progress>,
    any_version: bool,
    population: Option<usize>,
) -> Result<Experiment> {
    let file = File::open(path).context("Cannot open checkpoint")?;
    if let Some(progress) = progress {
        progress.total.store(
            file.metadata().map_or(0, |m| m.len()),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let mut file = BufReader::new(Counted {
        inner: file,
        progress,
    });
    let mut magic = [0; 8];
    file.read_exact(&mut magic)
        .with_context(|| format!("{} is not a save of this game", path.display()))?;
    reject_other_formats(path, &magic)?;
    let mut header = [0; SaveHeader::BYTES];
    file.read_exact(&mut header)
        .with_context(|| format!("{} is cut short", path.display()))?;
    let saved_version = SaveHeader::from_bytes(header).qd_version;
    if !any_version {
        ensure_current_version(path, &SaveHeader::from_bytes(header))?;
    }
    // bincode reads field by field; a buffer turns each read into a copy
    // instead of a call into the decompressor (18 s to 5 s at 3M).
    let mut decoder =
        BufReader::with_capacity(1 << 20, zstd::stream::read::Decoder::with_buffer(file)?);
    let mut small: SmallLoad = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(24 * 1024 * 1024 * 1024)
        .deserialize_from(&mut decoder)?;
    let migration: Option<(u32, Vec<(usize, usize)>)> = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(16 + 16 * island_count() as u64)
        .deserialize_from(&mut decoder)?;
    let mut trailing = [0u8; 1];
    ensure!(
        decoder.read(&mut trailing)? == 0,
        "Unexpected trailing checkpoint data"
    );
    small.qd_version = qd::VERSION;
    if let Some(population) = population {
        small.config.population = population;
    }
    let mut experiment = small.into_experiment(saved_version)?;
    experiment.last_migration = migration.filter(|(generation, exchange)| {
        *generation <= experiment.generation && exchange.len() == island_count()
    });
    Ok(experiment)
}

pub fn export_csv(path: &Path, history: &[Stats]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "generation",
        "population",
        "best_m",
        "median_m",
        "worst_m",
        "mean_m",
        "failed",
        "evaluation_seconds",
        "seed",
        "archive_cells",
        "qd_score",
        "archive_coverage",
    ])?;
    for s in history {
        w.serialize((
            s.generation,
            s.population,
            s.best,
            s.median,
            s.worst,
            s.mean,
            s.failed,
            s.seconds,
            s.config.seed,
            s.archive_cells,
            s.qd_score,
            s.archive_coverage,
        ))?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod peek_tests {
    use super::*;

    #[test]
    fn peek_reads_the_generation_and_world_of_a_save() {
        let config = Config {
            population: 64,
            random_seed: false,
            terrain: 2,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.generation = 7;
        let dir = std::env::temp_dir().join(format!("evo-peek-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("peek.evo");
        save(&path, &experiment).unwrap();
        let found = summary(&path).expect("a current checkpoint can be peeked");
        assert_eq!(found.generation, 7);
        assert_eq!(found.config.terrain, 2);
        assert_eq!(found.config.population, 64);
        std::fs::write(&path, b"not a checkpoint").unwrap();
        assert!(summary(&path).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
#[cfg(test)]
mod migration_tests {
    use super::*;

    /// Deterministic made-up results: a distance and a behavior from each
    /// creature's id.
    fn synthetic(pop: &Population, _: &Config) -> Result<Vec<EvaluationMetrics>> {
        Ok(pop
            .genomes
            .iter()
            .map(|g| {
                let h = g.id.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 20;
                EvaluationMetrics {
                    fitness: 1.0 + (h % 1000) as f32 * 0.02,
                    behavior: qd::TrialMetrics {
                        ground_contact: ((h >> 10) % 6) as f32 / 6.0 + 0.05,
                        gait_frequency: ((h >> 13) % 8) as f32 * 0.75 + 0.1,
                        mean_height: ((h >> 16) % 6) as f32 * 0.3 + 0.05,
                        feet: ((h >> 19) % 5) as f32,
                        ..Default::default()
                    },
                    ..Default::default()
                }
            })
            .collect())
    }

    #[test]
    fn the_header_turns_down_old_saves_before_reading_them() {
        let config = Config {
            population: 8,
            random_seed: false,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        let dir = std::env::temp_dir();
        let current = dir.join(format!("evolution-header-{}.evo", std::process::id()));
        save(&current, &experiment).unwrap();
        let header = check(&current).unwrap();
        assert_eq!(
            header,
            SaveHeader {
                qd_version: qd::VERSION,
                generation: experiment.generation,
                population: 8,
            }
        );
        assert_eq!(load(&current).unwrap().ring_len(), 8);

        // Saved under other physics: the header says so.
        experiment.qd_version = qd::OLDEST_LOADABLE - 1;
        save(&current, &experiment).unwrap();
        let error = check(&current).unwrap_err().to_string();
        assert!(error.contains("physics version"), "{error}");

        // Before the header: only the magic is read.
        let old = dir.join(format!("evolution-header-v6-{}.evo", std::process::id()));
        let mut bytes = b"EVORUST6".to_vec();
        bytes.extend([0u8; 64]);
        std::fs::write(&old, bytes).unwrap();
        let error = check(&old).unwrap_err().to_string();
        assert!(error.contains("older version"), "{error}");
        std::fs::write(&old, b"not a save").unwrap();
        assert!(check(&old).is_err());
        let _ = std::fs::remove_file(current);
        let _ = std::fs::remove_file(old);
    }

    #[test]
    fn a_cancelled_save_leaves_no_file_behind() {
        let config = Config {
            population: 8,
            random_seed: false,
            ..Config::default()
        };
        let experiment = Experiment::new(config).unwrap();
        let path =
            std::env::temp_dir().join(format!("evolution-cancel-{}.evo", std::process::id()));
        let progress = Progress::default();
        progress
            .cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(save_with_progress(&path, &experiment, Some(&progress)).is_err());
        assert!(!path.exists());
        assert!(!path.with_extension("evo.tmp").exists());
    }

    #[test]
    fn a_checkpoint_mid_generation_resumes_that_generation() {
        let config = Config {
            population: 40,
            random_seed: false,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.step(&mut synthetic).unwrap();
        assert!(experiment.evaluated > 0);
        let checkpoint =
            std::env::temp_dir().join(format!("evolution-steady-{}.evo", std::process::id()));
        // A save keeps no ring: the loaded game starts its saved generation
        // again with a ring bred from its archives.
        save(&checkpoint, &experiment).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);
        assert_eq!(loaded.evaluated, 0);
        assert_eq!(loaded.generation, experiment.generation);
        assert_eq!(loaded.ring_len(), 40);
        assert!(
            loaded
                .blocks
                .iter()
                .flat_map(|b| &b.births)
                .any(|b| b.emitter != Emitter::Restart)
        );
    }

    #[test]
    fn checkpoint_round_trip_keeps_the_autochange_step() {
        let config = Config {
            population: 2,
            random_seed: false,
            autochange: 2,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.config.autochange_step = 7;
        experiment.config.wind = crate::environment::WIND[2];
        let checkpoint = std::env::temp_dir().join(format!(
            "evolution-autochange-step-{}.evo",
            std::process::id()
        ));
        save(&checkpoint, &experiment).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);
        assert_eq!(loaded.config.autochange, 2);
        assert_eq!(loaded.config.autochange_step, 7);
        assert_eq!(loaded.config.wind, experiment.config.wind);
    }

    #[test]
    fn a_saved_game_keeps_its_last_migration() {
        let config = Config {
            population: 64,
            random_seed: false,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.run_generation(&mut synthetic).unwrap();
        let exchange: Vec<(usize, usize)> = (0..island_count()).map(|i| (i + 2, i + 1)).collect();
        assert!(!exchange.is_empty());
        experiment.last_migration = Some((experiment.generation, exchange.clone()));
        let checkpoint =
            std::env::temp_dir().join(format!("evolution-migration-{}.evo", std::process::id()));
        save(&checkpoint, &experiment).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);
        assert_eq!(
            loaded.last_migration,
            Some((experiment.generation, exchange))
        );
    }

    /// The autochange button at each speed: set while a generation runs, the
    /// level must survive the generation boundary and keep stepping.
    #[test]
    fn an_autochange_level_set_mid_generation_stays_and_steps() {
        for level in 1u8..=3 {
            let interval = crate::environment::AUTOCHANGE_INTERVALS[usize::from(level)];
            let config = Config {
                population: 4,
                random_seed: false,
                ..Config::default()
            };
            let mut experiment = Experiment::new(config).unwrap();
            for generation in 1..=interval * 2 {
                if generation == 2 {
                    // The panel sends its whole config, as the game does.
                    let mut cfg = experiment.config.clone();
                    cfg.autochange = level;
                    experiment.update_config(cfg).unwrap();
                }
                experiment.run_generation(&mut synthetic).unwrap();
                if generation >= 2 {
                    assert_eq!(
                        experiment.config.autochange, level,
                        "generation {generation}"
                    );
                }
                assert!(experiment.pending.is_none());
            }
            assert!(experiment.config.autochange_step >= 1, "level {level}");
        }
    }

    #[test]
    fn generation_boundaries_advance_the_autochange() {
        let interval = crate::environment::AUTOCHANGE_INTERVALS[2];
        let config = Config {
            population: 4,
            random_seed: false,
            autochange: 2,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        for generation in 1..=interval {
            experiment.run_generation(&mut synthetic).unwrap();
            assert_eq!(experiment.generation, generation);
            if generation < interval {
                assert_eq!(experiment.config.autochange_step, 0);
            }
        }
        assert_eq!(experiment.config.autochange_step, 1);
        // The first rung of the ladder is on.
        let (effect, level) = crate::environment::autochange_ladder()[0];
        assert_eq!(
            crate::environment::EFFECTS[effect].level(&experiment.config),
            level
        );
    }
}
