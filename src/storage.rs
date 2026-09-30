use crate::{
    config::Config,
    evolution::{self, CandidatePlan, Creature, FAILED, Genome, Population, Rng},
    qd::{self, CmaEmitter, Emitter, EmitterStats, EvaluationMetrics, QdArchive},
};
use anyhow::{Context, Result, ensure};
use bincode::Options;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::Arc,
};

/// Nanoseconds of breeding spent planning, emitting offspring, and writing
/// them into their block, since the last `take_breed_nanos`.
pub static BREED_NANOS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Returns and clears the breeding timers.
pub fn take_breed_nanos() -> [u64; 3] {
    std::array::from_fn(|i| BREED_NANOS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Creatures in flight: the ring holds this many at most, whatever the
/// generation size. It is enough to keep the GPU busy while the host
/// absorbs one block and breeds it again.
pub const RING_SLOTS: usize = 786_432;
/// Blocks in the ring. A block is bred, evaluated as one unit and absorbed
/// as a whole.
pub const RING_BLOCKS: usize = 4;
/// Confirmation trials a block asks for per archive at once while it waits
/// for the ones it needs.
const SPECULATIVE_CONFIRMS: usize = 8;

/// Ring slots for a generation of `population` evaluations.
pub fn ring_len(population: usize) -> usize {
    population.clamp(1, RING_SLOTS)
}
/// First slot and length of each block of a ring of `len` slots.
fn block_ranges(len: usize) -> Vec<(usize, usize)> {
    let size = len.div_ceil(RING_BLOCKS).max(1);
    (0..len)
        .step_by(size)
        .map(|first| (first, size.min(len - first)))
        .collect()
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
    #[serde(default)]
    pub archive_coverage: f32,
    #[serde(default)]
    pub emitters: [EmitterStats; qd::EMITTER_COUNT],
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
/// ring holds at most `RING_SLOTS` creatures, and each block is bred again
/// as soon as it is absorbed.
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
    /// Island archives, then one nursery per island. Ring slot `i` breeds
    /// for `qd::arena_of_slot(i, arena_count())`; the global `archive`
    /// collects every island's elites for display and statistics and is
    /// never a parent source.
    pub islands: Vec<QdArchive>,
    /// Every creature that entered an archive, keyed by creature id, with its
    /// parent and the change that produced it. Pruned to living elites' ancestors.
    pub lineage: HashMap<u64, Ancestor>,
    /// Each island's best distance so far and the generation it was set.
    pub island_progress: Vec<(f32, u32)>,
    /// Per island, what its nursery graduated this session.
    pub graduations: Vec<Graduation>,
    /// The last migration to the hub this session: its generation, and per
    /// island how many elites it sent and how many of those the hub kept
    /// (the hub's own entry is zero). Saved after the body of a small save.
    pub last_migration: Option<(u32, Vec<(usize, usize)>)>,
    /// Elites from before an environment change, waiting to be evaluated again
    /// in the new world, each queued for its own island. Breeding hands them
    /// out before new offspring.
    pub reseed: Reseed,
    /// Elites a meteor wiped out, with their island (None for the global
    /// archive), kept so the strike can be undone. Not saved.
    pub fossils: Vec<(Option<usize>, qd::Elite)>,
    /// The ring. Blocks are absorbed in ring order, starting at `cursor`.
    pub blocks: Vec<Block>,
    pub cursor: usize,
    /// Failed trials in the current generation.
    failed: usize,
    /// Distances at the screen of the current generation's results, for the
    /// next generation's bar.
    screen_log: Vec<f32>,
    /// Seconds spent absorbing results into the archives and breeding
    /// blocks again, since the caller last took them.
    pub stage_seconds: [f64; 2],
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

/// One recorded creature in an elite's ancestry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ancestor {
    pub parent: Option<u64>,
    pub creature: Creature,
    pub fitness: f32,
    pub generation: u32,
    /// What changed from the parent, for display.
    pub change: String,
}

/// Short description of how a child differs from its parent.
fn describe_change(
    parent: Option<&Creature>,
    child: &Creature,
    emitter: Emitter,
    crossed: bool,
) -> String {
    let mut parts: Vec<String> = vec![match emitter {
        Emitter::Cma => "fine-tuned".into(),
        Emitter::Structural => "reshaped".into(),
        Emitter::Novelty => "explored".into(),
        Emitter::Restart => "new random body".into(),
    }];
    if crossed {
        parts.push("crossed with a relative".into());
    }
    if let Some(parent) = parent {
        let nodes = child.nodes.len() as i64 - parent.nodes.len() as i64;
        let muscles = child.muscles.len() as i64 - parent.muscles.len() as i64;
        if nodes != 0 {
            parts.push(format!(
                "{nodes:+} node{}",
                if nodes.abs() == 1 { "" } else { "s" }
            ));
        }
        if muscles != 0 {
            parts.push(format!(
                "{muscles:+} muscle{}",
                if muscles.abs() == 1 { "" } else { "s" }
            ));
        }
        let organs = |c: &Creature| c.bones.iter().filter(|b| b.organ_mass > 0.0).count() as i64;
        let organ_change = organs(child) - organs(parent);
        if organ_change != 0 {
            parts.push(format!(
                "{organ_change:+} organ{}",
                if organ_change.abs() == 1 { "" } else { "s" }
            ));
        }
        let synced = |c: &Creature| {
            c.muscles.len() > 1 && c.muscles.iter().all(|m| m.period == c.muscles[0].period)
        };
        if synced(child) && !synced(parent) {
            parts.push("synced rhythm".into());
        }
    }
    parts.join(", ")
}

/// Island archives: `ISOLATED_ISLANDS` isolated islands, then the hub.
/// Slot `i` breeds from and competes in island `qd::island_of_slot`.
pub fn island_count() -> usize {
    ISOLATED_ISLANDS + 1
}
/// Archives that creatures breed for and compete in: the islands, then one
/// nursery per island. `Experiment::islands` holds them in this order.
pub fn arena_count() -> usize {
    island_count() * 2
}
/// The nursery archive of `island` in `Experiment::islands`.
pub fn nursery_of(island: usize) -> usize {
    island_count() + island
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
    /// A new game: the ring holds new random bodies, and the first
    /// generation has no screen bar yet, so every trial runs in full and
    /// records its distance at the screen.
    pub fn new(config: Config) -> Result<Self> {
        let mut config = config.resolved();
        config.validate()?;
        config.screen = crate::physics::screen_seconds()
            .filter(|&seconds| seconds < config.duration)
            .map(|seconds| crate::physics::Screen {
                seconds,
                bar: f32::NEG_INFINITY,
            });
        let mut e = Self::empty(config);
        let shared = Arc::new(e.config.clone());
        e.blocks = block_ranges(ring_len(e.config.population))
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
            archive: QdArchive::default(),
            emitter_stats: [EmitterStats::default(); qd::EMITTER_COUNT],
            cma_emitters: vec![],
            qd_version: qd::VERSION,
            breed_round: 0,
            islands: Vec::new(),
            lineage: HashMap::new(),
            island_progress: Vec::new(),
            graduations: Vec::new(),
            last_migration: None,
            reseed: Reseed::default(),
            fossils: Vec::new(),
            blocks: Vec::new(),
            cursor: 0,
            failed: 0,
            screen_log: Vec::new(),
            stage_seconds: [0.0; 2],
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
    /// Whether member `m`'s result may enter an archive at all.
    fn eligible(m: &EvaluationMetrics) -> bool {
        m.fitness.is_finite() && m.fitness > FAILED && !m.screened && !m.excluded
    }
    /// Decides block `k`'s results against the archives as they stand now.
    /// A creature that would set a new record of its island (or nursery)
    /// needs a confirmation trial at the fine physics, and its score is the
    /// lower of the two. The record-setters of each archive are taken
    /// fastest first, each against the record the ones before it set, so no
    /// unconfirmed score becomes a record. `confirmed` holds the
    /// confirmations that came back, by position. A block from a world that
    /// has since changed enters no archive.
    pub fn verdict(
        &self,
        k: usize,
        standard: &[EvaluationMetrics],
        confirmed: &HashMap<usize, EvaluationMetrics>,
    ) -> Verdict {
        let block = &self.blocks[k];
        let mut out = standard.to_vec();
        if block.config.physics_differs(&self.config) {
            for m in &mut out {
                m.excluded = true;
                m.screen_x = f32::NAN;
            }
            return Verdict::Final(out);
        }
        let arenas = arena_count();
        let bar = |arena: usize| {
            self.islands
                .get(arena)
                .map_or(f32::NEG_INFINITY, QdArchive::best_fitness)
        };
        let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); arenas];
        for (j, m) in standard.iter().enumerate() {
            let arena = qd::arena_of_slot(block.first + j, arenas);
            if Self::eligible(m) && m.fitness > bar(arena) {
                candidates[arena].push(j);
            }
        }
        let mut need = Vec::new();
        for (arena, mut members) in candidates.into_iter().enumerate() {
            members.sort_by(|&a, &b| {
                standard[b]
                    .fitness
                    .total_cmp(&standard[a].fitness)
                    .then(a.cmp(&b))
            });
            let mut record = bar(arena);
            let mut asked = 0;
            for j in members {
                if standard[j].fitness <= record {
                    break;
                }
                let Some(check) = confirmed.get(&j) else {
                    need.push(j);
                    asked += 1;
                    if asked == SPECULATIVE_CONFIRMS {
                        break;
                    }
                    continue;
                };
                let m = &mut out[j];
                // The replay must show the trial the score came from.
                m.fine = check.fitness < m.fitness;
                m.fitness = m.fitness.min(check.fitness);
                // A confirmation stopped by the screen is not robust.
                m.excluded |= check.screened || !check.fitness.is_finite();
                if Self::eligible(m) {
                    record = record.max(m.fitness);
                }
            }
        }
        if need.is_empty() {
            Verdict::Final(out)
        } else {
            need.sort_unstable();
            Verdict::Confirm(need)
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
        if !stale {
            // A result from a world that has since changed carries no distance.
            self.screen_log
                .extend(finals.iter().map(|m| m.screen_x).filter(|x| x.is_finite()));
            self.arm_screen_early();
        }
        let started = std::time::Instant::now();
        self.failed += self.archive_block(k, finals, stale);
        self.evaluated += finals.len();
        let ended = self.evaluated >= self.config.population;
        if ended {
            self.evaluated -= self.config.population;
            self.end_generation()?;
        }
        let archived = std::time::Instant::now();
        self.stage_seconds[0] += archived.duration_since(started).as_secs_f64();
        let (first, count) = (self.blocks[k].first, self.blocks[k].len());
        self.blocks[k] = self.breed_block(first, count);
        self.cursor = (k + 1) % self.blocks.len();
        self.stage_seconds[1] += archived.elapsed().as_secs_f64();
        Ok(ended)
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
                Verdict::Final(finals) => return self.absorb(k, &finals),
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
    /// Sets the screen bar inside a generation that started without one (a
    /// new game, the first generation after a load or a world change) once a
    /// quarter of the generation has recorded its distance at the screen, so
    /// only that first quarter runs every trial in full. Blocks bred after
    /// that take the bar; later generations take theirs at the boundary.
    fn arm_screen_early(&mut self) {
        let Some(screen) = self.config.screen else {
            return;
        };
        if screen.bar != f32::NEG_INFINITY
            || self.screen_log.len() < (self.config.population / 4).max(64)
        {
            return;
        }
        self.config.screen = self.next_screen(false, self.config.duration);
    }
    /// The early screen for the next generation: its bar is the distance at
    /// the screen that the best `physics::screen_keep()` share of this
    /// generation reached. After a world change distances are not comparable,
    /// so the next generation runs unscreened and sets a new bar.
    fn next_screen(&self, world_changed: bool, duration: f32) -> Option<crate::physics::Screen> {
        // A trial no longer than the screen time has nothing to screen.
        let seconds = crate::physics::screen_seconds().filter(|&s| s < duration)?;
        let bar = if world_changed {
            f32::NEG_INFINITY
        } else {
            crate::physics::screen_bar(
                self.screen_log.iter().copied(),
                crate::physics::screen_keep(),
            )
        };
        Some(crate::physics::Screen { seconds, bar })
    }
    /// Offers block `k`'s creatures to the archives in block order, updates
    /// CMA emitters and emitter statistics, and returns how many trials
    /// failed. Screened and excluded results, and every result of a
    /// `stale` block, enter no archive.
    fn archive_block(&mut self, k: usize, finals: &[EvaluationMetrics], stale: bool) -> usize {
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
            /// Body plan of a structural or novelty child, for its island's
            /// morphology reserve.
            topology: Option<qd::Topology>,
            /// Offered to no archive.
            screened: bool,
            /// Fitness of the global archive's elite in this creature's
            /// cell at the start of the block, for a CMA sample.
            elite_before: Option<f32>,
        }
        let arenas = self.islands.len().max(arena_count());
        let positions: Vec<usize> = (0..block.len()).collect();
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
                let nursery = qd::arena_of_slot(first + j, arenas) >= island_count();
                let valid = score.is_finite() && score > FAILED && !screened;
                let behavior_candidate = if valid && !nursery {
                    let niche = descriptor.niche();
                    match self.archive.slot_for(&niche) {
                        Some(slot) => score > self.archive.entries[slot].fitness,
                        None => self.archive.behavior_count() < qd::ARCHIVE_LIMIT,
                    }
                } else {
                    false
                };
                let topology = (valid && matches!(emitter, Emitter::Structural | Emitter::Novelty))
                    .then(|| qd::topology_of_population(population, j));
                let elite_before = (emitter == Emitter::Cma)
                    .then_some(birth.cma)
                    .flatten()
                    .filter(|_| score.is_finite() && score > FAILED)
                    .and_then(|_| self.archive.slot_for(&descriptor.niche()))
                    .map(|slot| self.archive.entries[slot].fitness);
                Prep {
                    descriptor,
                    emitter,
                    score,
                    fine: m.fine,
                    protection: birth.protection,
                    behavior_candidate,
                    topology,
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
        // Per island: the positions that entered, and the emitter and offer
        // of each reserve entry.
        type IslandResult = (Vec<usize>, Vec<(usize, qd::Offer)>);
        let island_results: Vec<IslandResult> = self
            .islands
            .par_iter_mut()
            .enumerate()
            .map(|(island, archive)| {
                let mut entered = Vec::new();
                let mut reserve_offers = Vec::new();
                // Reserve admission needs a score above the island's best
                // behavior elite and reserve entry of the same body plan, or
                // above the reserve's floor once it is full
                // (`QdArchive::offer_morphology` makes the final check).
                let mut parents: HashMap<u64, (qd::Topology, bool)> = HashMap::new();
                let mut bars: HashMap<qd::Topology, (f32, f32)> = HashMap::new();
                for elite in &archive.entries {
                    let morphology = qd::is_morphology_niche(&elite.niche);
                    parents.insert(elite.creature.id, (elite.topology.clone(), morphology));
                    let bar = bars
                        .entry(elite.topology.clone())
                        .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                    if morphology {
                        bar.1 = bar.1.max(elite.fitness);
                    } else {
                        bar.0 = bar.0.max(elite.fitness);
                    }
                }
                for (j, p) in prep.iter().enumerate() {
                    if qd::arena_of_slot(first + j, arenas) != island {
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
                        if let Some(topology) = &p.topology {
                            let bar = bars
                                .entry(topology.clone())
                                .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                            bar.0 = bar.0.max(p.score);
                        }
                        continue;
                    }
                    let Some(topology) = &p.topology else {
                        continue;
                    };
                    // A reserve place goes to a new body plan, or to a better
                    // child of a reserve entry with the same plan.
                    let parent = births[j].parent_id.and_then(|id| parents.get(&id));
                    let changed = parent.is_some_and(|(plan, _)| {
                        !qd::topology_equivalent_for_archive(topology, plan)
                    });
                    let from_reserve = parent.is_some_and(|(plan, morphology)| {
                        *morphology && qd::topology_equivalent_for_archive(topology, plan)
                    });
                    if !(changed || from_reserve) {
                        continue;
                    }
                    let floor = archive.morphology_floor();
                    let admits = match bars.get(topology) {
                        Some(&(behavior, reserve)) if reserve > f32::NEG_INFINITY => {
                            p.score > behavior && p.score > reserve
                        }
                        Some(&(behavior, _)) => {
                            p.score > behavior && floor.is_none_or(|floor| p.score > floor)
                        }
                        None => floor.is_none_or(|floor| p.score > floor),
                    };
                    if !admits {
                        continue;
                    }
                    let offer = archive.offer_morphology(
                        population,
                        j,
                        p.descriptor,
                        topology.clone(),
                        p.score,
                        p.fine,
                        p.emitter,
                        generation,
                        p.protection,
                    );
                    if offer.inserted {
                        entered.push(j);
                        let bar = bars
                            .entry(topology.clone())
                            .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                        bar.1 = bar.1.max(p.score);
                        reserve_offers.push((p.emitter.index(), offer));
                    }
                }
                (entered, reserve_offers)
            })
            .collect();
        let island_changed: Vec<bool> = island_results
            .iter()
            .map(|(group, _)| !group.is_empty())
            .collect();
        let mut reserve_offers = Vec::new();
        for (arena, (group, offers)) in island_results.into_iter().enumerate() {
            entered.extend(group);
            // Nursery entries count for no emitter: the emitter statistics
            // describe the islands' search.
            if arena < island_count() {
                reserve_offers.extend(offers);
            }
        }
        timings[0] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // The scores depend only on the archive's elites, so an island that
        // took no offer keeps the ones it has.
        for (island, changed) in self.islands.iter_mut().zip(island_changed) {
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
                let best = best_by_niche.entry(p.descriptor.niche()).or_insert(j);
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
            if qd::arena_of_slot(first + j, arenas) >= island_count() {
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
                    .and_then(|_| self.archive.slot_for(&prep.descriptor.niche()))
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
        for j in entered {
            self.record_ancestor(population, births[j], j, finals[j].fitness);
        }
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
    /// Records creature `index` of `population` (which just entered an
    /// archive with `fitness`).
    fn record_ancestor(
        &mut self,
        population: &Population,
        birth: Birth,
        index: usize,
        fitness: f32,
    ) {
        let creature = population.creature(index);
        if self.lineage.contains_key(&creature.id) {
            return;
        }
        let change = describe_change(
            birth
                .parent_id
                .and_then(|id| self.lineage.get(&id))
                .map(|a| &a.creature),
            &creature,
            birth.emitter,
            birth.mate,
        );
        self.lineage.insert(
            creature.id,
            Ancestor {
                parent: birth.parent_id,
                fitness,
                generation: self.generation,
                change,
                creature,
            },
        );
    }
    /// Drops lineage records that no living elite descends from.
    fn prune_lineage(&mut self) {
        let mut keep: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let elites = self
            .archive
            .entries
            .iter()
            .chain(self.islands.iter().flat_map(|island| island.entries.iter()));
        for elite in elites {
            let mut id = Some(elite.creature.id);
            while let Some(current) = id {
                if !keep.insert(current) {
                    break;
                }
                id = self.lineage.get(&current).and_then(|a| a.parent);
            }
        }
        self.lineage.retain(|id, _| keep.contains(id));
    }
    /// Ancestor chain of a creature, newest first (at most `limit` steps).
    pub fn ancestry(&self, id: u64, limit: usize) -> Vec<&Ancestor> {
        let mut chain = Vec::new();
        let mut current = Some(id);
        while let Some(id) = current {
            let Some(ancestor) = self.lineage.get(&id) else {
                break;
            };
            chain.push(ancestor);
            if chain.len() >= limit {
                break;
            }
            current = ancestor.parent;
        }
        chain
    }
    fn push_archive_stats(&mut self, failed: usize) {
        if self.history.len() > self.generation as usize {
            return;
        }
        let mut elites: Vec<_> = self
            .archive
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .collect();
        elites.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let count = elites.len();
        let archive_best = self.archive.best_fitness();
        let quantile = |p: f32| {
            if count == 0 {
                0.0
            } else {
                elites[((1.0 - p / 100.0) * (count - 1) as f32).round() as usize].fitness
            }
        };
        let mut histogram = BTreeMap::<i32, u32>::new();
        let mut species = BTreeMap::<(usize, usize), u32>::new();
        let mut sum = 0.0f64;
        for elite in &elites {
            sum += elite.fitness as f64;
            *histogram
                .entry((elite.fitness * 100.0).floor() as i32)
                .or_default() += 1;
            *species
                .entry((elite.creature.nodes.len(), elite.creature.muscles.len()))
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
                .map(|i| all_elites[i].creature.clone())
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
            archive_cells: count,
            qd_score: self.archive.qd_score,
            archive_coverage: self.archive.coverage(),
            emitters: self.emitter_stats,
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
        self.islands = vec![QdArchive::default(); arena_count()];
        self.island_progress.clear();
        self.graduations.clear();
        self.last_migration = None;
    }
    /// Every `NURSERY_GENERATIONS` generations each nursery's survivors
    /// compete with their island's elites on distance alone, and the global
    /// archive takes those the island kept. Then the nursery starts over
    /// with new random bodies.
    fn graduate_nurseries(&mut self) {
        if self.islands.len() != arena_count()
            || self.generation == 0
            || !self.generation.is_multiple_of(qd::NURSERY_GENERATIONS)
        {
            return;
        }
        self.graduations
            .resize(island_count(), Graduation::default());
        for island in 0..island_count() {
            let nursery = nursery_of(island);
            let mut cohort: Vec<qd::Elite> = std::mem::take(&mut self.islands[nursery].entries)
                .into_iter()
                .filter(|e| !qd::is_morphology_niche(&e.niche))
                .collect();
            self.islands[nursery].rebuild_indices();
            cohort.sort_unstable_by(|a, b| {
                b.fitness
                    .total_cmp(&a.fitness)
                    .then_with(|| a.niche.cmp(&b.niche))
            });
            let mut kept = 0;
            for mut elite in cohort.iter().cloned() {
                elite.graduate = true;
                if self.islands[island].absorb(&elite) {
                    kept += 1;
                    self.archive.absorb(&elite);
                }
            }
            self.islands[island].refresh_behavior_scores();
            if kept > 0 {
                self.archive.refresh_behavior_scores();
            }
            let log = &mut self.graduations[island];
            *log = Graduation {
                generation: self.generation,
                sent: cohort.len(),
                kept,
                kept_total: log.kept_total + kept,
            };
            if let Some(progress) = self.island_progress.get_mut(nursery) {
                *progress = (f32::NEG_INFINITY, self.generation);
            }
        }
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
        self.last_migration = Some((self.generation, exchange));
        self.islands[hub].refresh_behavior_scores();
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
        let mut reset_cma = HashMap::<(usize, qd::Niche, qd::Topology), usize>::new();
        // CMA slot lookup keyed by (island, niche, body plan): an emitter
        // samples around one island's elite, so it serves only that island.
        // The bucket stores the full key, so the per-offspring probe hashes
        // and compares without cloning the topology vector; clones are only
        // paid when a slot is created or replaced.
        type CmaKey = (usize, qd::Niche, qd::Topology);
        struct CmaLookup {
            buckets: HashMap<u64, Vec<(CmaKey, usize)>>,
        }
        impl CmaLookup {
            fn hash(island: usize, niche: &qd::Niche, topology: &qd::Topology) -> u64 {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                island.hash(&mut hasher);
                niche.hash(&mut hasher);
                topology.hash(&mut hasher);
                hasher.finish()
            }
            fn get(
                &self,
                island: usize,
                niche: &qd::Niche,
                topology: &qd::Topology,
            ) -> Option<usize> {
                self.buckets
                    .get(&Self::hash(island, niche, topology))?
                    .iter()
                    .find(|((i, n, t), _)| *i == island && n == niche && t == topology)
                    .map(|(_, index)| *index)
            }
            fn insert(&mut self, key: CmaKey, index: usize) {
                let bucket = self
                    .buckets
                    .entry(Self::hash(key.0, &key.1, &key.2))
                    .or_default();
                if let Some(entry) = bucket.iter_mut().find(|(stored, _)| *stored == key) {
                    entry.1 = index;
                } else {
                    bucket.push((key, index));
                }
            }
            fn remove(
                &mut self,
                island: usize,
                niche: &qd::Niche,
                topology: &qd::Topology,
                index: usize,
            ) {
                let hash = Self::hash(island, niche, topology);
                let Some(bucket) = self.buckets.get_mut(&hash) else {
                    return;
                };
                bucket.retain(|((i, n, t), stored_index)| {
                    !(*i == island && n == niche && t == topology && *stored_index == index)
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
            cma_lookup.insert((cma.island, cma.niche.clone(), cma.topology.clone()), index);
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
        }
        // Each island's elites grouped by body plan, for crossover partners.
        let by_plan: Vec<HashMap<&qd::Topology, Vec<usize>>> = self
            .islands
            .iter()
            .map(|island| {
                let mut groups: HashMap<&qd::Topology, Vec<usize>> = HashMap::new();
                for (index, elite) in island.entries.iter().enumerate() {
                    groups.entry(&elite.topology).or_default().push(index);
                }
                groups
            })
            .collect();
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
                order.sort_by(|&a, &b| {
                    island.entries[b]
                        .fitness
                        .total_cmp(&island.entries[a].fitness)
                });
                order
            })
            .collect();
        // Each island's fastest 1% of elites (at least 4), for exploitation.
        let top_parents: Vec<Vec<usize>> = orders
            .iter()
            .map(|order| order[..(order.len() / 100).max(4).min(order.len())].to_vec())
            .collect();
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
        plan_times[1] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        let plan_prep: Vec<PlanPrep> = slots
            .par_iter()
            .map(|&i| {
                let mut rng = Rng::new(seed, generation, i);
                let island = qd::arena_of_slot(i, self.islands.len());
                let archive = &self.islands[island];
                let archive_empty = archive.entries.is_empty();
                let emitter = if archive_empty
                    || (island >= island_count() && rng.unit() < qd::NURSERY_FRESH_SHARE)
                {
                    Emitter::Restart
                } else {
                    qd::choose_emitter(&mut rng, &weights)
                };
                let emitter_stale = self.emitter_stats[emitter.index()].stale();
                let avoid = None;
                let mut optimize = false;
                let mut from_reserve = false;
                let parent = if emitter == Emitter::Restart || archive_empty {
                    None
                } else if emitter == Emitter::Structural
                    && rng.unit() < qd::MORPHOLOGY_PARENT_FRACTION
                {
                    // Each island keeps its own morphology reserve.
                    let drawn = archive.sample_morphology(&mut rng, avoid);
                    from_reserve = drawn.is_some();
                    drawn.or_else(|| archive.sample_local_competitive(&mut rng, avoid))
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
                    Some(if optimize {
                        optimizer_targets[island].unwrap_or(top_parents[island][0])
                    } else {
                        top_parents[island][rng.index(top_parents[island].len())]
                    })
                } else {
                    archive.sample_local_competitive(&mut rng, avoid)
                };
                let parent_id = parent.map(|index| archive.entries[index].creature.id);
                let protection = if matches!(emitter, Emitter::Structural | Emitter::Novelty) {
                    generation.saturating_add(3)
                } else {
                    parent
                        .map(|index| archive.entries[index].protected_until)
                        .unwrap_or(0)
                };
                let mate = match (emitter, parent) {
                    (Emitter::Structural | Emitter::Novelty, Some(p))
                        if !from_reserve && rng.unit() < 0.2 =>
                    {
                        by_plan[island]
                            .get(&archive.entries[p].topology)
                            .filter(|group| group.len() > 1)
                            .map(|group| group[rng.index(group.len())])
                            .filter(|&m| m != p)
                    }
                    _ => None,
                };
                // Sometimes the mate has another body plan: the child gets one of
                // its limbs grafted on (see `evolution::mated`).
                let mate = mate.or_else(|| match (emitter, parent) {
                    (Emitter::Structural | Emitter::Novelty, Some(p))
                        if !from_reserve && rng.unit() < CROSS_PLAN_MATE_SHARE =>
                    {
                        let other = rng.index(archive.entries.len());
                        (other != p
                            && archive.entries[other].topology != archive.entries[p].topology)
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
            } = prep;
            let cma_index = if emitter == Emitter::Cma {
                if let Some(parent_index) = parent {
                    let elite = &self.islands[island].entries[parent_index];
                    let template = &elite.creature;
                    let topology = &elite.topology;
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
                            .get(island, &lookup_niche, topology)
                            .filter(|i| !converged(i))
                    } else if emitter_stale {
                        reset_cma
                            .get(&(island, lookup_niche.clone(), topology.clone()))
                            .copied()
                    } else {
                        cma_lookup.get(island, &lookup_niche, topology)
                    };
                    if index.is_none() {
                        let restart = cma_lookup
                            .get(island, &lookup_niche, topology)
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
                                            && c.topology == *topology
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
                                let (old_island, old_niche, old_topology) =
                                    (old.island, old.niche.clone(), old.topology.clone());
                                if cma_lookup.get(old_island, &old_niche, &old_topology)
                                    == Some(slot)
                                {
                                    cma_lookup.remove(old_island, &old_niche, &old_topology, slot);
                                }
                                reset_cma.retain(|_, index| *index != slot);
                                self.cma_emitters[slot] = new;
                            }
                            let new_niche = self.cma_emitters[slot].niche.clone();
                            let new_topology = self.cma_emitters[slot].topology.clone();
                            cma_lookup.insert((island, new_niche, new_topology), slot);
                            if emitter_stale && !optimize {
                                reset_cma
                                    .insert((island, lookup_niche.clone(), topology.clone()), slot);
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
        (plans.into_iter().map(|p| p.plan).collect(), self.breed_round)
    }
    /// Breeds a block for ring slots `first..first + count` from the current
    /// archives with the current settings. Elites queued by a world change
    /// take the slots of their own islands first. An island without elites
    /// breeds new random bodies.
    fn breed_block(&mut self, first: usize, count: usize) -> Block {
        let slots: Vec<usize> = (first..first + count).collect();
        let cfg = self.config.clone();
        self.breed_round += 1;
        let started = std::time::Instant::now();
        let planned = self.plan_offspring(&cfg, self.generation, self.breed_round, &slots);
        let planned_at = started.elapsed();
        // Reseeded elites first, then the children, emitted straight into
        // batches so no child is alive after it is copied.
        let mut lead = evolution::ChildBatch::default();
        let mut order: Vec<usize> = Vec::with_capacity(count);
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
        if !self.reseed.is_empty() {
            for (k, &slot) in slots.iter().enumerate() {
                if let Some(elite) = self.reseed_for_slot(slot) {
                    lead.push(elite);
                    order.push(k);
                    births[k] = Birth::RANDOM;
                }
            }
        }
        let reseeded = order.len();
        let mut taken = vec![false; count];
        for &k in &order {
            taken[k] = true;
        }
        order.extend((0..count).filter(|&k| !taken[k]));
        let bred_slots: Vec<usize> = order[reseeded..].iter().map(|&k| slots[k]).collect();
        let bred_plans: Vec<CandidatePlan> =
            order[reseeded..].iter().map(|&k| planned[k].plan).collect();
        let mut batches = Vec::new();
        if reseeded > 0 {
            batches.push(lead);
        }
        batches.extend(evolution::emit_offspring_batches(
            &self.islands,
            &self.cma_emitters,
            &bred_plans,
            &bred_slots,
            &cfg,
            self.generation,
            self.breed_round,
        ));
        let emitted_at = started.elapsed();
        let mut population = Population {
            genomes: vec![Genome::default(); count],
            ..Population::default()
        };
        population.append_batches(&order, batches);
        let total = started.elapsed();
        let add = |k: usize, d: std::time::Duration| {
            BREED_NANOS[k].fetch_add(d.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        };
        add(0, planned_at);
        add(1, emitted_at.saturating_sub(planned_at));
        add(2, total.saturating_sub(emitted_at));
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
        let failed = std::mem::take(&mut self.failed);
        self.push_archive_stats(failed);
        self.prune_lineage();
        self.generation += 1;
        self.graduate_nurseries();
        self.migrate_islands();
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
        cfg.screen = self.next_screen(world_changed, cfg.duration);
        self.config = cfg;
        self.screen_log.clear();
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
            cfg.screen = if world_changed {
                self.next_screen(true, cfg.duration)
            } else {
                self.config.screen
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
        let mut strike = |archive: &mut QdArchive, island: Option<usize>| {
            let (kept, lost): (Vec<_>, Vec<_>) = std::mem::take(&mut archive.entries)
                .into_iter()
                .partition(|_| rng.unit() >= share);
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
        lost
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
        self.islands[index].rebuild_indices();
        let count = lost.len();
        self.fossils
            .extend(lost.into_iter().map(|elite| (Some(index), elite)));
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
            match archive.slot_for(&elite.niche) {
                Some(slot) if archive.entries[slot].fitness < elite.fitness => {
                    archive.entries[slot] = elite;
                }
                Some(_) => continue,
                None => {
                    archive.entries.push(elite);
                    // Later fossils must see this cell as taken.
                    archive.rebuild_indices();
                }
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
    /// the new physics in that island's own slots.
    fn reset_search_context(&mut self) {
        self.reseed.clear();
        // The nurseries start over; only the islands' creatures are re-tested.
        for (index, island) in self.islands.iter_mut().take(island_count()).enumerate() {
            for elite in std::mem::take(&mut island.entries) {
                self.reseed.push(index, elite.creature);
            }
        }
        self.archive = QdArchive::default();
        // Fossils are old-world elites: undoing a meteor must not bring them
        // back into the new world's archives.
        self.fossils.clear();
        self.islands.clear();
        self.island_progress.clear();
        self.graduations.clear();
        self.last_migration = None;
        self.emitter_stats = [EmitterStats::default(); qd::EMITTER_COUNT];
        self.cma_emitters.clear();
        // Distances measured in the old world say nothing about the new one.
        self.screen_log.clear();
    }
    /// Whether every body in the ring and the archive fits `cfg`'s limits.
    fn bodies_fit(&self, cfg: &Config) -> bool {
        self.blocks
            .iter()
            .flat_map(|b| &b.population.genomes)
            .all(|g| g.node_count <= cfg.max_nodes && g.muscle_count <= cfg.max_muscles)
            && self.archive.entries.iter().all(|elite| {
                elite.creature.nodes.len() <= cfg.max_nodes
                    && elite.creature.muscles.len() <= cfg.max_muscles
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
            next == ring_len(self.config.population),
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
                && self.archive.entries.len() <= qd::ARCHIVE_CAPACITY
                && self.archive.behavior_count() <= qd::ARCHIVE_LIMIT
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
const MAGIC: &[u8; 8] = b"EVORUST8";

/// What a save holds: the archives and the search state, without the
/// population, its scores, or anything bred for the generation in progress.
#[derive(Serialize)]
struct SmallSave<'a> {
    config: &'a Config,
    pending: &'a Option<Config>,
    generation: u32,
    history: &'a [Stats],
    archive: &'a QdArchive,
    emitter_stats: &'a [EmitterStats; qd::EMITTER_COUNT],
    cma_emitters: &'a [CmaEmitter],
    qd_version: u32,
    breed_round: u64,
    islands: &'a [QdArchive],
    lineage: &'a HashMap<u64, Ancestor>,
    island_progress: &'a [(f32, u32)],
    reseed: &'a Reseed,
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
}
impl<'a> SmallSave<'a> {
    fn of(e: &'a Experiment) -> Self {
        Self {
            config: &e.config,
            pending: &e.pending,
            generation: e.generation,
            history: &e.history,
            archive: &e.archive,
            emitter_stats: &e.emitter_stats,
            cma_emitters: &e.cma_emitters,
            qd_version: e.qd_version,
            breed_round: e.breed_round,
            islands: &e.islands,
            lineage: &e.lineage,
            island_progress: &e.island_progress,
            reseed: &e.reseed,
        }
    }
}
impl SmallLoad {
    /// The game the save describes, at the start of its saved generation.
    /// Its ring is bred from the archives, as the game would have bred it;
    /// without elites it starts with new random bodies.
    fn into_experiment(self) -> Result<Experiment> {
        let mut e = Experiment::empty(self.config);
        e.pending = self.pending;
        e.generation = self.generation;
        e.history = repair_history(self.history, self.generation);
        e.archive = self.archive;
        e.emitter_stats = self.emitter_stats;
        e.cma_emitters = self.cma_emitters;
        e.qd_version = self.qd_version;
        e.breed_round = self.breed_round;
        e.islands = self.islands;
        e.lineage = self.lineage;
        e.island_progress = self.island_progress;
        e.reseed = self.reseed;
        ensure!(
            e.island_progress.len() <= 64
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
        e.archive.rebuild_indices();
        for island in &mut e.islands {
            island.rebuild_indices();
        }
        // The screen bar is not saved: the resumed generation runs every
        // trial in full until it has set a new one.
        e.config.screen = e.next_screen(true, e.config.duration);
        let shared = Arc::new(e.config.clone());
        let elites =
            e.archive.entries.len() + e.islands.iter().map(|i| i.entries.len()).sum::<usize>();
        for (first, count) in block_ranges(ring_len(e.config.population)) {
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
                    ..e.breed_block(first, count)
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
        header.qd_version == qd::VERSION,
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
    if SaveHeader::from_bytes(header).qd_version != qd::VERSION {
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
    load_from(path, progress, false)
}

/// `load` for diagnostics that only breed from the archives
/// (`examples/breed_bench.rs`): a save of an older physics version loads
/// too, with the scores it measured then.
#[doc(hidden)]
pub fn load_any_version(path: &Path) -> Result<Experiment> {
    load_from(path, None, true)
}

fn load_from(path: &Path, progress: Option<&Progress>, any_version: bool) -> Result<Experiment> {
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
        .with_limit(8 + 64 * 16)
        .deserialize_from(&mut decoder)?;
    let mut trailing = [0u8; 1];
    ensure!(
        decoder.read(&mut trailing)? == 0,
        "Unexpected trailing checkpoint data"
    );
    if any_version {
        small.qd_version = qd::VERSION;
    }
    let mut experiment = small.into_experiment()?;
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
        experiment.qd_version = qd::VERSION - 1;
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
