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

mod archive;
mod breeding;
mod confirm;
pub(crate) mod dump;
mod islands;
mod lineage;
mod ring_shape;
mod save_format;
mod world;

pub use breeding::{BREED_LATE, BREED_NANOS, take_breed_late, take_breed_nanos};
use confirm::{ConfirmHint, ScreenWindow};
pub use islands::{
    Graduation, ISOLATED_ISLANDS, MIGRATION_INTERVAL, MIGRATION_SHARE, arena_count, hub_island,
    island_count, nursery_of, reshaped_of,
};
use islands::{new_islands, new_reshaped_nursery};
use lineage::ISLAND_LEADERS;
pub use lineage::{ANCESTRY_DEPTH, Ancestor};
pub use ring_shape::{RingShape, RingTimes};
pub use save_format::{
    Progress, SaveHeader, SaveSummary, check, export_csv, load, load_any_version, load_archives,
    load_for_population, load_with_progress, peek, rotate_autosaves, save, save_with_progress,
    summary,
};
pub use world::{Refuge, Reseed};

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
    /// The screen bar of each wild island's creatures, in its own world,
    /// fixed when the block is bred; empty runs them in full.
    pub wild_bars: Arc<Vec<f32>>,
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
    /// Each wild island's distances at the screen, in its own world.
    wild_windows: Vec<ScreenWindow>,
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

/// Generations without a new island record before the island's optimizer
/// turns to its next fastest design.
const OPTIMIZER_STALL: u32 = 30;

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
                wild_bars: Arc::default(),
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
            wild_windows: vec![ScreenWindow::default(); qd::WILD_ISLANDS],
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
            // Each wild island sets its own bar from its evolved creatures.
            let mut wild: Vec<Vec<f32>> = vec![Vec::new(); qd::WILD_ISLANDS];
            let first = self.blocks[k].first;
            for (j, m) in finals.iter().enumerate() {
                let island = qd::island_of_slot(first + j, island_count());
                let young = self.blocks[k].population.flags.get(j).copied().unwrap_or(0)
                    & (crate::rungs::YOUNG | crate::rungs::RESHAPED)
                    != 0;
                if qd::is_wild(island) && !young && m.screen_x.is_finite() {
                    wild[island - qd::MAIN_ISLANDS].push(m.screen_x);
                }
            }
            for (window, distances) in self.wild_windows.iter_mut().zip(wild) {
                window.push(distances, 0, ring);
            }
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
                    let slot = best
                        .entry(archive.plan_key(i))
                        .or_insert((e.fitness, &e.creature));
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
                    .map(|e| dump::Elite::of(arena as u16, e)),
            );
        }
        elites.extend(
            self.archive
                .entries
                .iter()
                .map(|e| dump::Elite::of(u16::MAX, e)),
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

fn fitness_context_changed(old: &Config, new: &Config) -> bool {
    old.physics_differs(new)
}
