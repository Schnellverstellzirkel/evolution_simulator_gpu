use crate::{
    config::Config,
    evolution::{self, CandidatePlan, Creature, FAILED, LegacyMuscle, Population, Rng},
    qd::{self, CmaEmitter, Emitter, EmitterStats, QdArchive, TrialMetrics},
};
use anyhow::{Context, Result, ensure};
use bincode::Options;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Stage {
    Ready,
    Evaluating,
    Evaluated,
    Ranked,
    Selected,
    Archived,
}
impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::Evaluating => "Evaluating",
            Self::Evaluated => "Evaluation complete",
            Self::Ranked => "Sorted by fitness",
            Self::Selected => "Survivors selected",
            Self::Archived => "Archive updated",
        }
    }
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
#[derive(Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub config: Config,
    pub pending: Option<Config>,
    pub generation: u32,
    pub population: Population,
    pub scores: Vec<f32>,
    /// Parent generation results for newly created creatures; never used as current fitness.
    #[serde(skip)]
    pub parent_scores: Vec<f32>,
    pub evaluated: usize,
    pub stage: Stage,
    pub ranks: Vec<usize>,
    pub parents: Vec<usize>,
    pub history: Vec<Stats>,
    pub evaluation_seconds: f64,
    #[serde(default)]
    pub archive: QdArchive,
    #[serde(default)]
    pub emitter_stats: [EmitterStats; qd::EMITTER_COUNT],
    #[serde(default)]
    pub cma_emitters: Vec<CmaEmitter>,
    #[serde(default)]
    pub candidate_emitters: Vec<Emitter>,
    #[serde(default)]
    pub candidate_cma: Vec<Option<usize>>,
    /// Parent IDs for the current batch. Kept in memory for benchmark genealogy
    /// analysis; lineage is intentionally not part of checkpoint state.
    #[serde(skip)]
    pub candidate_parent_ids: Vec<Option<u64>>,
    /// Optional per-process override used by paired benchmark runs; checkpoints keep the default.
    #[serde(skip)]
    pub morphology_reserve_override: Option<bool>,
    #[serde(default)]
    pub protected_until: Vec<u32>,
    #[serde(default)]
    pub trial_metrics: Vec<TrialMetrics>,
    #[serde(default)]
    pub qd_version: u32,
    /// Steady-state breeding rounds so far; salts offspring random streams.
    #[serde(default)]
    pub breed_round: u64,
    /// Island archives. Slot `i` breeds from island `i % island_count()`; the global
    /// `archive` collects every island's elites for display and statistics.
    #[serde(default)]
    pub islands: Vec<QdArchive>,
    /// Every creature that entered an archive, keyed by creature id, with its
    /// parent and the change that produced it. Pruned to living elites' ancestors.
    #[serde(default)]
    pub lineage: HashMap<u64, Ancestor>,
    /// Whether each slot's current creature came from crossover.
    #[serde(skip)]
    pub candidate_mates: Vec<bool>,
    /// Each island's best distance so far and the generation it was set.
    /// Stored separately in V4 checkpoints to keep the V3 payload readable.
    #[serde(skip)]
    pub island_progress: Vec<(f32, u32)>,
    /// Elites from before an environment change, waiting to be evaluated again
    /// in the new world. Breeding hands them out before new offspring.
    #[serde(default)]
    pub reseed: Vec<evolution::Creature>,
    /// Elites a meteor wiped out, with their island (None for the global
    /// archive), kept so the strike can be undone. Not saved in checkpoints.
    #[serde(skip)]
    pub fossils: Vec<(Option<usize>, qd::Elite)>,
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

/// Independent parent pools; elites migrate between neighbors periodically.
/// `EVOLUTION_ISLANDS` overrides the count (1 disables islands).
pub fn island_count() -> usize {
    static COUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("EVOLUTION_ISLANDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (1..=64).contains(&n))
            .unwrap_or(4)
    })
}
/// Generations between bounded archive-elite refreshes, read from
/// `EVOLUTION_ELITE_REFRESH`. Unset, zero, or unparsable means off, which is
/// the default. Read on every call so a running game and the measurement
/// harness agree without a restart.
pub fn elite_refresh_interval() -> u64 {
    std::env::var("EVOLUTION_ELITE_REFRESH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}
/// Most elites one refresh cycle re-tests: a small, bounded cost next to the
/// generation's own evaluation. The subset rotates, so every elite is reached
/// after enough cycles.
const ELITE_REFRESH_BATCH: usize = 4;
/// Deterministic pose and grip perturbation for the fresh-perturbation elite
/// refresh. It mirrors the contender robustness check in `scheduler::perturb`:
/// node x and y move by up to 2 cm and grip varies by ±10%, seeded from the
/// creature id alone so the same elite always gets the same fresh trial. The
/// stored score already folded in the unperturbed standard trial, so only a
/// different nearby pose can disprove it.
pub fn perturb_elite(creature: &mut Creature) {
    let mut rng = Rng::new(creature.id ^ 0x5eed_7a11, 0, 0);
    for node in &mut creature.nodes {
        node.x += rng.range(-0.02, 0.02);
        node.y += rng.range(0.0, 0.02);
        node.friction = (node.friction * rng.range(0.9, 1.1)).clamp(0.0, 1.0);
    }
}
/// Share of CMA offspring whose parent is one of its island's fastest 1% of
/// elites; the rest sample by local competition. Spending more on the best
/// elites raised the best distance by about half in fixed-seed tests.
const TOP_PARENT_SHARE: f32 = 0.5;
/// Share of those top-elite CMA offspring bred by an island optimizer
/// (separable CMA-ES in physical units) on one of its fastest designs.
const OPTIMIZER_SHARE: f32 = 0.5;
/// Generations between migrations, and the share of elites that migrate.
/// Rare migration lets each island settle on and refine its own design
/// instead of all islands polishing the same one.
const MIGRATION_INTERVAL: u32 = 25;
/// Generations without a new island record before the island's optimizer
/// turns to its next fastest design.
const OPTIMIZER_STALL: u32 = 30;
const MIGRATION_SHARE: f32 = 0.1;
struct OffspringPlan {
    plan: CandidatePlan,
    parent_id: Option<u64>,
    protection: u32,
}

impl Experiment {
    pub fn new(config: Config) -> Result<Self> {
        let config = config.resolved();
        let population = evolution::create(&config)?;
        let population_count = config.population;
        let scores = vec![f32::NAN; population_count];
        let parent_scores = vec![f32::NAN; population_count];
        Ok(Self {
            config,
            pending: None,
            generation: 0,
            population,
            scores,
            parent_scores,
            evaluated: 0,
            stage: Stage::Ready,
            ranks: vec![],
            parents: vec![],
            history: vec![],
            evaluation_seconds: 0.0,
            archive: QdArchive::default(),
            emitter_stats: [EmitterStats::default(); qd::EMITTER_COUNT],
            cma_emitters: vec![],
            candidate_emitters: vec![Emitter::Restart; population_count],
            candidate_cma: vec![None; population_count],
            candidate_parent_ids: vec![None; population_count],
            morphology_reserve_override: None,
            protected_until: vec![0; population_count],
            trial_metrics: vec![TrialMetrics::default(); population_count],
            qd_version: qd::VERSION,
            breed_round: 0,
            islands: Vec::new(),
            lineage: HashMap::new(),
            candidate_mates: Vec::new(),
            island_progress: Vec::new(),
            reseed: Vec::new(),
            fossils: Vec::new(),
        })
    }
    pub fn rank(&mut self) {
        self.ranks = evolution::ranking(&self.scores);
        let mut histogram = BTreeMap::<i32, u32>::new();
        let mut species = BTreeMap::<(usize, usize), u32>::new();
        let mut sum = 0.0f64;
        let mut failed = 0;
        for (&s, g) in self.scores.iter().zip(&self.population.genomes) {
            if s > FAILED && s.is_finite() {
                sum += s as f64;
                *histogram.entry((s * 100.0).floor() as i32).or_default() += 1;
            } else {
                failed += 1;
            }
            *species.entry((g.node_count, g.muscle_count)).or_default() += 1;
        }
        let count = self.scores.len();
        let valid = count - failed;
        let quantile = |p: f32| {
            if valid == 0 {
                0.0
            } else {
                self.scores[self.ranks[((1.0 - p / 100.0) * (valid - 1) as f32).round() as usize]]
            }
        };
        let representatives = [count - 1, (count - 1) / 2, 0]
            .map(|r| self.population.creature(self.ranks[r]))
            .to_vec();
        self.history.push(Stats {
            generation: self.generation,
            best: quantile(100.),
            median: quantile(50.),
            worst: quantile(0.),
            mean: if valid > 0 {
                (sum / valid as f64) as f32
            } else {
                0.
            },
            failed,
            seconds: self.evaluation_seconds,
            population: count,
            percentiles: PERCENTILES.iter().map(|&p| quantile(p)).collect(),
            histogram: histogram.into_iter().collect(),
            species: species.into_iter().map(|((n, m), c)| (n, m, c)).collect(),
            representatives,
            config: self.config.clone(),
            archive_cells: 0,
            qd_score: 0.0,
            archive_coverage: 0.0,
            emitters: self.emitter_stats,
        });
        self.stage = Stage::Ranked;
    }
    pub fn archive_batch(&mut self) -> Result<()> {
        ensure!(
            self.evaluated == self.config.population,
            "Cannot archive an incomplete batch"
        );
        ensure!(
            self.trial_metrics.len() == self.config.population,
            "Invalid behavior metric count"
        );
        let all: Vec<usize> = (0..self.config.population).collect();
        let failed = self.archive_slots(&all);
        self.push_archive_stats(failed);
        self.prune_lineage();
        self.stage = Stage::Archived;
        Ok(())
    }
    /// Whether creature `i`'s standard-trial result could enter an archive.
    /// Only those creatures need the check trial: their final score is the
    /// lower of both trials, so every other creature is rejected either way.
    pub fn contender(&self, i: usize, metric: &qd::EvaluationMetrics) -> bool {
        if !metric.fitness.is_finite() || metric.fitness <= FAILED {
            return false;
        }
        if self.from_optimizer(i) {
            return true;
        }
        let Some(genome) = self.population.genomes.get(i) else {
            return false;
        };
        let nodes =
            &self.population.nodes[genome.node_start..genome.node_start + genome.node_count];
        let muscles = &self.population.muscles
            [genome.muscle_start..genome.muscle_start + genome.muscle_count];
        let niche = qd::descriptor(nodes, muscles, metric.behavior).niche();
        let beats = |archive: &QdArchive| match archive.slot_for(&niche) {
            Some(slot) => metric.fitness > archive.entries[slot].fitness,
            None => archive.behavior_count() < qd::ARCHIVE_LIMIT,
        };
        if beats(&self.archive) || self.islands.get(i % island_count()).is_some_and(beats) {
            return true;
        }
        let reserve_candidate = self.morphology_reserve_override != Some(false)
            && matches!(
                self.candidate_emitters.get(i),
                Some(Emitter::Structural | Emitter::Novelty)
            );
        reserve_candidate
            && self
                .archive
                .morphology_floor()
                .is_none_or(|floor| metric.fitness > floor)
    }
    /// Whether creature `i` was sampled by an island optimizer. Optimizers
    /// rank all their samples, so all of them get the same check: ranking
    /// checked samples by the check and the rest by their first trial alone
    /// would steer the search away from its most promising samples.
    pub fn from_optimizer(&self, i: usize) -> bool {
        self.candidate_cma
            .get(i)
            .copied()
            .flatten()
            .and_then(|c| self.cma_emitters.get(c))
            .is_some_and(|c| c.optimizing())
    }
    /// Offers the evaluated creatures in `slots` to the archive (in slot-list
    /// order), updates CMA emitters and emitter statistics, and returns how
    /// many trials failed.
    pub fn archive_slots(&mut self, slots: &[usize]) -> usize {
        self.ensure_islands();
        // Every creature also competes in its own island's archive.
        let mut entered: Vec<usize> = Vec::new();
        for &i in slots {
            let score = self.scores[i];
            if !score.is_finite() || score <= FAILED {
                continue;
            }
            let genome = &self.population.genomes[i];
            let nodes =
                &self.population.nodes[genome.node_start..genome.node_start + genome.node_count];
            let muscles = &self.population.muscles
                [genome.muscle_start..genome.muscle_start + genome.muscle_count];
            let descriptor = qd::descriptor(nodes, muscles, self.trial_metrics[i]);
            let emitter = self
                .candidate_emitters
                .get(i)
                .copied()
                .unwrap_or(Emitter::Restart);
            let protection = self.protected_until.get(i).copied().unwrap_or(0);
            if self.islands[i % island_count()]
                .offer(
                    &self.population,
                    i,
                    descriptor,
                    score,
                    emitter,
                    self.generation,
                    protection,
                )
                .inserted
            {
                entered.push(i);
            }
        }
        for island in &mut self.islands {
            island.refresh_behavior_scores();
        }
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
        let parent_morphologies: HashMap<_, _> = self
            .archive
            .entries
            .iter()
            .map(|elite| {
                (
                    elite.creature.id,
                    (
                        elite.topology.clone(),
                        qd::is_morphology_niche(&elite.niche),
                    ),
                )
            })
            .collect();
        // Parallel prefilter: descriptors, behavior-offer eligibility against the
        // start-of-batch archive, and static morphology-offer eligibility. Occupant
        // fitness only ever rises, so a snapshot reject stays a live reject. Inserts
        // still commit sequentially in index order so niche races resolve exactly
        // like the old single loop.
        struct Prep {
            descriptor: qd::Descriptor,
            emitter: Emitter,
            score: f32,
            protection: u32,
            behavior_candidate: bool,
            morphology_topology: Option<qd::Topology>,
        }
        let reserve_enabled = self.morphology_reserve_override != Some(false);
        let prep: Vec<Prep> = slots
            .par_iter()
            .map(|&i| {
                let score = self.scores[i];
                let emitter = self
                    .candidate_emitters
                    .get(i)
                    .copied()
                    .unwrap_or(Emitter::Restart);
                let genome = &self.population.genomes[i];
                let nodes = &self.population.nodes
                    [genome.node_start..genome.node_start + genome.node_count];
                let muscles = &self.population.muscles
                    [genome.muscle_start..genome.muscle_start + genome.muscle_count];
                let descriptor = qd::descriptor(nodes, muscles, self.trial_metrics[i]);
                let protection = self.protected_until.get(i).copied().unwrap_or(0);
                let behavior_candidate = if score.is_finite() && score > FAILED {
                    let niche = descriptor.niche();
                    match self.archive.slot_for(&niche) {
                        Some(slot) => score > self.archive.entries[slot].fitness,
                        None => self.archive.behavior_count() < qd::ARCHIVE_LIMIT,
                    }
                } else {
                    false
                };
                let morphology_topology = if reserve_enabled
                    && matches!(emitter, Emitter::Structural | Emitter::Novelty)
                {
                    let topology = qd::topology_of_population(&self.population, i);
                    let parent = self
                        .candidate_parent_ids
                        .get(i)
                        .copied()
                        .flatten()
                        .and_then(|id| parent_morphologies.get(&id));
                    let topology_changed = parent.is_some_and(|(parent_topology, _)| {
                        !qd::topology_equivalent_for_archive(&topology, parent_topology)
                    });
                    let descended_from_reserve =
                        parent.is_some_and(|(parent_topology, morphology)| {
                            *morphology
                                && qd::topology_equivalent_for_archive(&topology, parent_topology)
                        });
                    (descended_from_reserve || topology_changed).then_some(topology)
                } else {
                    None
                };
                Prep {
                    descriptor,
                    emitter,
                    score,
                    protection,
                    behavior_candidate,
                    morphology_topology,
                }
            })
            .collect();
        // Players browse this archive and replay its creatures with the CPU
        // engine, so it only admits scores the replay reproduces. The best
        // candidate for each behavior cell and each new body plan in this batch
        // (only it can end up as the elite) runs its standard trial again on
        // the CPU engine; its score becomes the worse of all its trials, and
        // its cell comes from the replayed behavior. A gait that only works
        // through one engine's rounding loses its advantage here.
        let mut prep = prep;
        let mut best_by_niche: HashMap<qd::Niche, usize> = HashMap::new();
        let mut best_by_topology: HashMap<qd::Topology, usize> = HashMap::new();
        for (k, p) in prep.iter().enumerate() {
            if p.behavior_candidate {
                let best = best_by_niche.entry(p.descriptor.niche()).or_insert(k);
                if prep[*best].score < p.score {
                    *best = k;
                }
            }
            if let Some(topology) = &p.morphology_topology {
                let best = best_by_topology.entry(topology.clone()).or_insert(k);
                if prep[*best].score < p.score {
                    *best = k;
                }
            }
        }
        let behavior_best: std::collections::HashSet<usize> =
            best_by_niche.values().copied().collect();
        let topology_best: std::collections::HashSet<usize> =
            best_by_topology.values().copied().collect();
        let mut verify: Vec<usize> = behavior_best.union(&topology_best).copied().collect();
        verify.sort_unstable();
        for (k, p) in prep.iter_mut().enumerate() {
            p.behavior_candidate &= behavior_best.contains(&k);
            if !topology_best.contains(&k) {
                p.morphology_topology = None;
            }
        }
        if !verify.is_empty() {
            let indices: Vec<usize> = verify.iter().map(|&k| slots[k]).collect();
            let subset = self.population.subset(&indices);
            let replay_cfg = Config {
                fidelity: None,
                ..self.config.clone()
            };
            let results = crate::cpu_engine::evaluate(&subset, &replay_cfg);
            for (n, &k) in verify.iter().enumerate() {
                let i = slots[k];
                let replayed = crate::scheduler::to_metrics(&subset, n, &results[n], &replay_cfg);
                let score = prep[k].score.min(replayed.fitness);
                let genome = &self.population.genomes[i];
                let nodes = &self.population.nodes
                    [genome.node_start..genome.node_start + genome.node_count];
                let muscles = &self.population.muscles
                    [genome.muscle_start..genome.muscle_start + genome.muscle_count];
                let descriptor = qd::descriptor(nodes, muscles, replayed.behavior);
                self.scores[i] = score;
                self.trial_metrics[i] = replayed.behavior;
                let p = &mut prep[k];
                p.score = score;
                p.descriptor = descriptor;
                if p.behavior_candidate {
                    p.behavior_candidate = score.is_finite()
                        && score > FAILED
                        && match self.archive.slot_for(&descriptor.niche()) {
                            Some(slot) => score > self.archive.entries[slot].fitness,
                            None => self.archive.behavior_count() < qd::ARCHIVE_LIMIT,
                        };
                }
            }
        }
        let mut attempts = [0u64; qd::EMITTER_COUNT];
        let mut failed = 0usize;
        for (&i, prep) in slots.iter().zip(&prep) {
            if !prep.score.is_finite() || prep.score <= FAILED {
                failed += 1;
            }
            let emitter_index = prep.emitter.index();
            attempts[emitter_index] += 1;
            let elite_before = self
                .archive
                .slot_for(&prep.descriptor.niche())
                .map(|slot| self.archive.entries[slot].fitness);
            let behavior_offer = if prep.behavior_candidate {
                self.archive.offer(
                    &self.population,
                    i,
                    prep.descriptor,
                    prep.score,
                    prep.emitter,
                    self.generation,
                    prep.protection,
                )
            } else {
                qd::Offer::default()
            };
            let morphology_offer = if !behavior_offer.inserted
                && let Some(topology) = prep.morphology_topology.clone()
            {
                self.archive.offer_morphology(
                    &self.population,
                    i,
                    prep.descriptor,
                    topology,
                    prep.score,
                    prep.emitter,
                    self.generation,
                    prep.protection,
                )
            } else {
                qd::Offer::default()
            };
            let offer = if behavior_offer.inserted {
                behavior_offer
            } else {
                morphology_offer
            };
            // CMA-ME improvement ranking: new niches first, then improvement over
            // the niche's elite, then how far short of it a sample fell.
            if prep.emitter == Emitter::Cma
                && let Some(cma) = self.candidate_cma.get(i).copied().flatten()
                && let Some(samples) = cma_samples.get_mut(cma)
                && prep.score.is_finite()
                && prep.score > FAILED
            {
                let key = match elite_before {
                    _ if optimizers[cma] => prep.score,
                    None if behavior_offer.inserted => 1.0e6 + prep.score,
                    Some(before) if behavior_offer.inserted => 1.0e3 + (prep.score - before),
                    Some(before) => prep.score - before,
                    None => prep.score - 1.0e3,
                };
                samples.push((i, key));
            }
            if offer.inserted {
                entered.push(i);
                rewards[emitter_index] += offer.reward;
                if offer.new_niche {
                    discoveries[emitter_index] += 1;
                } else {
                    improvements[emitter_index] += 1;
                }
            }
        }
        for (emitter, samples) in self.cma_emitters.iter_mut().zip(&mut cma_samples) {
            emitter.tell(&self.population, samples);
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
        self.archive.refresh_behavior_scores();
        entered.sort_unstable();
        entered.dedup();
        for i in entered {
            self.record_ancestor(i);
        }
        failed
    }
    /// Records the creature in `slot` (which just entered an archive).
    fn record_ancestor(&mut self, slot: usize) {
        let creature = self.population.creature(slot);
        if self.lineage.contains_key(&creature.id) {
            return;
        }
        let parent = self.candidate_parent_ids.get(slot).copied().flatten();
        let emitter = self
            .candidate_emitters
            .get(slot)
            .copied()
            .unwrap_or(Emitter::Restart);
        let crossed = self.candidate_mates.get(slot).copied().unwrap_or(false);
        let change = describe_change(
            parent
                .and_then(|id| self.lineage.get(&id))
                .map(|a| &a.creature),
            &creature,
            emitter,
            crossed,
        );
        self.lineage.insert(
            creature.id,
            Ancestor {
                parent,
                fitness: self.scores[slot],
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
        self.refresh_elites_from_env();
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
            [0, 0, 0].map(|i| self.population.creature(i)).to_vec()
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
    /// Periodic bounded refresh of archive elites. Every `interval`
    /// generations a rotating, deterministic subset of at most
    /// `ELITE_REFRESH_BATCH` elites re-runs its standard trial on the CPU
    /// engine from a fresh deterministic perturbation (`perturb_elite`, the
    /// contender check's pose and grip shift), and the archive keeps the lower
    /// of the stored and re-evaluated fitness. The stored score already folded
    /// in the unperturbed standard trial, so the perturbation is what can
    /// catch a fragile elite that got lucky on its own exact pose. The cell,
    /// creature, and descriptor never change. `interval == 0` disables it.
    /// Returns how many elites were lowered.
    pub fn refresh_elites(&mut self, interval: u64) -> usize {
        let completed = self.generation as u64 + 1;
        if interval == 0 || !completed.is_multiple_of(interval) {
            return 0;
        }
        let cycle = completed / interval;
        self.refresh_elite_batch(cycle)
    }
    /// `refresh_elites` with the interval read from `EVOLUTION_ELITE_REFRESH`
    /// (generations; unset or 0 disables it, the default).
    pub fn refresh_elites_from_env(&mut self) -> usize {
        self.refresh_elites(elite_refresh_interval())
    }
    fn refresh_elite_batch(&mut self, cycle: u64) -> usize {
        let batch = ELITE_REFRESH_BATCH.min(self.archive.entries.len());
        if batch == 0 {
            return 0;
        }
        // Sorting by creature id makes the rotating window independent of the
        // archive's internal entry order, so the selection is deterministic
        // even after insertions and removals reorder the arena.
        let mut ids: Vec<u64> = self
            .archive
            .entries
            .iter()
            .map(|elite| elite.creature.id)
            .collect();
        ids.sort_unstable();
        let start = ((cycle - 1).wrapping_mul(batch as u64) % ids.len() as u64) as usize;
        let mut unit = Population::default();
        for k in 0..batch {
            let id = ids[(start + k) % ids.len()];
            if let Some(elite) = self.archive.entries.iter().find(|e| e.creature.id == id) {
                let mut creature = elite.creature.clone();
                perturb_elite(&mut creature);
                unit.push(creature);
            }
        }
        if unit.genomes.is_empty() {
            return 0;
        }
        // The standard configuration, exactly as the archive-admission check
        // runs it (no fine-fidelity override).
        let cfg = Config {
            fidelity: None,
            ..self.config.clone()
        };
        let results = crate::cpu_engine::evaluate(&unit, &cfg);
        let mut lowered = 0;
        for (index, result) in results.iter().enumerate() {
            let metrics = crate::scheduler::to_metrics(&unit, index, result, &cfg);
            if !metrics.fitness.is_finite() || metrics.fitness <= FAILED {
                // A failed re-test says nothing about the stored score.
                continue;
            }
            let id = unit.genomes[index].id;
            for archive in std::iter::once(&mut self.archive).chain(self.islands.iter_mut()) {
                let slot = archive
                    .entries
                    .iter()
                    .position(|elite| elite.creature.id == id);
                if let Some(slot) = slot
                    && archive.lower_fitness(slot, metrics.fitness)
                {
                    lowered += 1;
                }
            }
        }
        if lowered > 0 {
            self.archive.refresh_behavior_scores();
            for island in &mut self.islands {
                island.refresh_behavior_scores();
            }
        }
        lowered
    }
    pub fn prepare_next_batch(&mut self) -> Result<()> {
        self.prepare_next_batch_streaming(usize::MAX, |_, _, _| Ok(()))
    }
    /// Breeds the next generation, handing each finished slice of `slice`
    /// offspring to `on_slice` (with the generation's settings) so evaluation
    /// can start early. The result is identical for every slice size.
    pub fn prepare_next_batch_streaming(
        &mut self,
        slice: usize,
        mut on_slice: impl FnMut(&Population, std::ops::Range<usize>, &Config) -> Result<()>,
    ) -> Result<()> {
        let preparation_started = std::time::Instant::now();
        ensure!(
            self.stage == Stage::Archived,
            "The archive must be updated before breeding"
        );
        let cfg = self.pending.clone().unwrap_or_else(|| self.config.clone());
        cfg.validate()?;
        if fitness_context_changed(&self.config, &cfg) {
            self.reset_search_context();
        }
        evolution::ensure_archive_batch_memory(&self.population, &self.archive, &cfg)?;
        let generation = self.generation + 1;
        let setup_seconds = preparation_started.elapsed().as_secs_f64();
        let plan_started = std::time::Instant::now();
        let all: Vec<usize> = (0..cfg.population).collect();
        let planned = self.plan_offspring(&cfg, generation, 0, &all);
        let plans: Vec<CandidatePlan> = planned.iter().map(|p| p.plan).collect();
        let emitters: Vec<Emitter> = planned.iter().map(|p| p.plan.emitter).collect();
        let cma_indices: Vec<Option<usize>> = planned.iter().map(|p| p.plan.cma).collect();
        let parent_ids: Vec<Option<u64>> = planned.iter().map(|p| p.parent_id).collect();
        let protections: Vec<u32> = planned.iter().map(|p| p.protection).collect();
        let plan_seconds = plan_started.elapsed().as_secs_f64();
        let emission_started = std::time::Instant::now();
        // Elites queued by a world change take the first slots. No slice is
        // handed over until they are placed, so every device sees them.
        let reseeding = !self.reseed.is_empty();
        let next = evolution::emit_archive_batch_streaming(
            &self.population,
            &self.islands,
            &self.cma_emitters,
            &plans,
            &cfg,
            generation,
            slice.min(cfg.population).max(1),
            |population, range| {
                if reseeding {
                    Ok(())
                } else {
                    on_slice(population, range, &cfg)
                }
            },
        )?;
        let emission_seconds = emission_started.elapsed().as_secs_f64();
        self.config = cfg;
        self.pending = None;
        self.population = next;
        self.candidate_emitters = emitters;
        self.candidate_cma = cma_indices;
        self.candidate_parent_ids = parent_ids;
        self.candidate_mates = planned.iter().map(|p| p.plan.mate.is_some()).collect();
        self.protected_until = protections;
        if reseeding {
            for slot in 0..self.config.population {
                let Some(elite) = self.reseed.pop() else {
                    break;
                };
                self.population.replace(slot, elite);
                self.candidate_emitters[slot] = Emitter::Restart;
                self.candidate_cma[slot] = None;
                self.candidate_parent_ids[slot] = None;
                self.candidate_mates[slot] = false;
                self.protected_until[slot] = 0;
            }
            on_slice(&self.population, 0..self.config.population, &self.config)?;
        }
        self.parent_scores.fill(f32::NAN);
        self.generation = generation;
        self.migrate_islands();
        self.stage = Stage::Ready;
        self.evaluated = 0;
        self.scores.fill(f32::NAN);
        self.trial_metrics.fill(TrialMetrics::default());
        self.ranks.clear();
        self.parents.clear();
        self.evaluation_seconds = 0.0;
        if std::env::var_os("EVOLUTION_PROFILE_BREED").is_some() {
            eprintln!(
                "Breeding profile: generation {generation}, setup {setup_seconds:.6} s, parent plans {plan_seconds:.6} s, candidate emission {emission_seconds:.6} s, finalization {:.6} s, total {:.6} s",
                preparation_started.elapsed().as_secs_f64()
                    - setup_seconds
                    - plan_seconds
                    - emission_seconds,
                preparation_started.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }
    /// Creates the island archives if missing, seeding them from the global
    /// archive's elites.
    fn ensure_islands(&mut self) {
        if self.islands.len() == island_count() {
            return;
        }
        self.islands = vec![QdArchive::default(); island_count()];
        self.island_progress.clear();
        for (index, elite) in self.archive.entries.iter().enumerate() {
            self.islands[index % island_count()].absorb(elite);
        }
        for island in &mut self.islands {
            island.refresh_behavior_scores();
        }
    }
    /// Every few generations each island receives the best share of its
    /// neighbor's elites.
    fn migrate_islands(&mut self) {
        if self.islands.len() != island_count()
            || !self.generation.is_multiple_of(MIGRATION_INTERVAL)
        {
            return;
        }
        let migrants: Vec<Vec<qd::Elite>> = self
            .islands
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
        for (from, group) in migrants.into_iter().enumerate() {
            let to = &mut self.islands[(from + 1) % island_count()];
            for elite in &group {
                to.absorb(elite);
            }
        }
        for island in &mut self.islands {
            island.refresh_behavior_scores();
        }
    }
    /// Chooses emitters, parents, and CMA slots for offspring in `slots`.
    /// Round 0 reproduces the generational random streams; other rounds salt
    /// them so steady-state breeding never repeats a draw.
    fn plan_offspring(
        &mut self,
        cfg: &Config,
        generation: u32,
        round: u64,
        slots: &[usize],
    ) -> Vec<OffspringPlan> {
        self.ensure_islands();
        let weights = qd::emitter_weights(&self.emitter_stats);
        let mut reset_cma = HashMap::<(qd::Niche, qd::Topology), usize>::new();
        let mut cma_lookup: HashMap<(qd::Niche, qd::Topology), usize> = self
            .cma_emitters
            .iter()
            .enumerate()
            .map(|(i, cma)| ((cma.niche.clone(), cma.topology.clone()), i))
            .collect();
        let mut used_cma = vec![false; self.cma_emitters.len()];
        let mut out = Vec::with_capacity(slots.len());
        // Phase A: emitter choice and parent sampling against the start-of-batch
        // archive. Each creature has its own deterministic RNG, so parallel order
        // does not change the draws. last_parent is snapshotted instead of updating
        // mid-loop; visit() and CMA slot allocation stay sequential below.
        let reserve_enabled = self.morphology_reserve_override != Some(false);
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
        let seed = cfg.seed ^ round.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        // Each island's fastest 1% of elites (at least 4), for exploitation.
        let top_parents: Vec<Vec<usize>> = self
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
                order.truncate((order.len() / 100).max(4));
                order
            })
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
            .zip(&mut self.island_progress)
            .map(|((island, top), progress)| {
                let best = *top.first()?;
                let fitness = island.entries[best].fitness;
                if fitness > progress.0 {
                    *progress = (fitness, generation);
                }
                let mut plans: Vec<usize> = Vec::new();
                let mut order: Vec<usize> = (0..island.entries.len())
                    .filter(|&i| !qd::is_morphology_niche(&island.entries[i].niche))
                    .collect();
                order.sort_by(|&a, &b| {
                    island.entries[b]
                        .fitness
                        .total_cmp(&island.entries[a].fitness)
                });
                // A design is a body plan with a gait cadence band.
                let design = |i: usize| (&island.entries[i].topology, island.entries[i].niche.0[1]);
                for i in order {
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
        let plan_prep: Vec<PlanPrep> = slots
            .par_iter()
            .map(|&i| {
                let mut rng = Rng::new(seed, generation, i);
                let island = i % island_count();
                let archive = &self.islands[island];
                let archive_empty = archive.entries.is_empty();
                let emitter = if archive_empty {
                    Emitter::Restart
                } else {
                    qd::choose_emitter(&mut rng, &weights)
                };
                let emitter_stale = self.emitter_stats[emitter.index()].stale();
                let avoid = None;
                let mut optimize = false;
                let parent = if emitter == Emitter::Restart || archive_empty {
                    None
                } else if reserve_enabled
                    && emitter == Emitter::Structural
                    && rng.unit() < qd::MORPHOLOGY_PARENT_FRACTION
                {
                    archive
                        .sample_morphology(&mut rng, avoid)
                        .or_else(|| archive.sample_local_competitive(&mut rng, avoid))
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
                    (Emitter::Structural | Emitter::Novelty, Some(p)) if rng.unit() < 0.2 => {
                        by_plan[island]
                            .get(&archive.entries[p].topology)
                            .filter(|group| group.len() > 1)
                            .map(|group| group[rng.index(group.len())])
                            .filter(|&m| m != p)
                    }
                    _ => None,
                };
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
                    let niche_key = if optimize {
                        (
                            qd::optimizer_niche(island, elite.niche.0[1]),
                            topology.clone(),
                        )
                    } else {
                        (elite.niche.clone(), topology.clone())
                    };
                    let converged = |i: &usize| self.cma_emitters[*i].converged() && !used_cma[*i];
                    let mut index = if optimize {
                        cma_lookup
                            .get(&niche_key)
                            .copied()
                            .filter(|i| !converged(i))
                    } else if emitter_stale {
                        reset_cma.get(&niche_key).copied()
                    } else {
                        cma_lookup.get(&niche_key).copied()
                    };
                    if index.is_none() {
                        let restart = cma_lookup.get(&niche_key).copied().filter(|_| optimize);
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
                            let new = if optimize {
                                // Another island's optimizer for the same plan
                                // lends its learned step sizes, unless this is
                                // a restart after converging.
                                self.cma_emitters
                                    .iter()
                                    .filter(|c| {
                                        c.optimizing() && c.topology == *topology && !c.converged()
                                    })
                                    .max_by_key(|c| c.last_used_generation)
                                    .map_or_else(
                                        || {
                                            CmaEmitter::optimizer(
                                                template.clone(),
                                                niche_key.0.clone(),
                                                generation,
                                            )
                                        },
                                        |c| {
                                            c.recentered(
                                                template.clone(),
                                                niche_key.0.clone(),
                                                generation,
                                            )
                                        },
                                    )
                            } else {
                                CmaEmitter::new(template.clone(), elite.niche.clone(), generation)
                            };
                            let new_key = (new.niche.clone(), new.topology.clone());
                            if slot == self.cma_emitters.len() {
                                self.cma_emitters.push(new);
                                used_cma.push(false);
                            } else {
                                let old_key = (
                                    self.cma_emitters[slot].niche.clone(),
                                    self.cma_emitters[slot].topology.clone(),
                                );
                                if cma_lookup.get(&old_key) == Some(&slot) {
                                    cma_lookup.remove(&old_key);
                                }
                                reset_cma.retain(|_, index| *index != slot);
                                self.cma_emitters[slot] = new;
                            }
                            cma_lookup.insert(new_key, slot);
                            if emitter_stale && !optimize {
                                reset_cma.insert(niche_key, slot);
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
        out
    }
    /// Steady-state breeding: replaces the creatures in `slots` (already offered
    /// to the archive) with offspring bred from the current archive.
    pub fn breed_slots(&mut self, slots: &[usize]) -> Result<()> {
        if slots.is_empty() {
            return Ok(());
        }
        let count = self.config.population;
        self.candidate_parent_ids.resize(count, None);
        self.candidate_mates.resize(count, false);
        self.candidate_emitters.resize(count, Emitter::Restart);
        self.candidate_cma.resize(count, None);
        self.protected_until.resize(count, 0);
        self.parent_scores.resize(count, f32::NAN);
        let cfg = self.config.clone();
        self.breed_round += 1;
        let planned = self.plan_offspring(&cfg, self.generation, self.breed_round, slots);
        let plans: Vec<CandidatePlan> = planned.iter().map(|p| p.plan).collect();
        let children = evolution::emit_offspring(
            &self.islands,
            &self.cma_emitters,
            &plans,
            slots,
            &cfg,
            self.generation,
            self.breed_round,
        );
        for ((&slot, child), plan) in slots.iter().zip(children).zip(&planned) {
            if let Some(elite) = self.reseed.pop() {
                self.population.replace(slot, elite);
                self.candidate_emitters[slot] = Emitter::Restart;
                self.candidate_cma[slot] = None;
                self.candidate_parent_ids[slot] = None;
                self.candidate_mates[slot] = false;
                self.protected_until[slot] = 0;
            } else {
                self.population.replace(slot, child);
                self.candidate_emitters[slot] = plan.plan.emitter;
                self.candidate_cma[slot] = plan.plan.cma;
                self.candidate_parent_ids[slot] = plan.parent_id;
                self.candidate_mates[slot] = plan.plan.mate.is_some();
                self.protected_until[slot] = plan.protection;
            }
            self.parent_scores[slot] = f32::NAN;
            self.scores[slot] = f32::NAN;
            self.trial_metrics[slot] = TrialMetrics::default();
        }
        Ok(())
    }
    /// Steady-state generation boundary (every `population` evaluations):
    /// records history, applies queued settings, and compacts the arenas.
    pub fn finish_steady_generation(&mut self, failed: usize) -> Result<()> {
        self.push_archive_stats(failed);
        self.prune_lineage();
        self.generation += 1;
        self.migrate_islands();
        if let Some(cfg) = self.pending.take() {
            cfg.validate()?;
            ensure!(
                cfg.population == self.config.population,
                "Population changes need a new experiment"
            );
            if fitness_context_changed(&self.config, &cfg) {
                self.reset_search_context();
            }
            self.config = cfg;
        }
        self.population.compact();
        self.evaluation_seconds = 0.0;
        self.evaluated = 0;
        Ok(())
    }
    pub fn update_config(&mut self, cfg: Config) -> Result<()> {
        cfg.validate()?;
        ensure!(
            cfg.population == self.config.population
                && cfg.seed == self.config.seed
                && cfg.random_seed == self.config.random_seed,
            "Population or seed changes require a new experiment"
        );
        ensure!(
            self.population
                .genomes
                .iter()
                .all(|g| g.node_count <= cfg.max_nodes && g.muscle_count <= cfg.max_muscles),
            "Existing bodies exceed these limits; start a new experiment"
        );
        ensure!(
            self.archive.entries.iter().all(|elite| {
                elite.creature.nodes.len() <= cfg.max_nodes
                    && elite.creature.muscles.len() <= cfg.max_muscles
            }),
            "Archived bodies exceed these limits; start a new experiment"
        );
        if self.stage == Stage::Ready {
            if fitness_context_changed(&self.config, &cfg) {
                self.reset_search_context();
            }
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
    /// An extinction wipes out the island whose best creature is slowest. Its
    /// cells refill from its own survivors' offspring and from migrants, so a
    /// stalled island starts over from new designs (Lehman and Miikkulainen,
    /// 2015). The lost elites become fossils, so it can be undone. Returns how
    /// many elites were lost.
    pub fn extinction(&mut self) -> usize {
        let weakest = self
            .islands
            .iter()
            .enumerate()
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
    /// Clears the archive after the world changed. Its scores no longer hold,
    /// but its creatures are queued to compete again under the new physics.
    fn reset_search_context(&mut self) {
        self.reseed = std::mem::take(&mut self.archive.entries)
            .into_iter()
            .map(|elite| elite.creature)
            .collect();
        self.archive = QdArchive::default();
        self.islands.clear();
        self.island_progress.clear();
        self.emitter_stats = [EmitterStats::default(); qd::EMITTER_COUNT];
        self.cma_emitters.clear();
    }
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        self.population.validate(&self.config)?;
        ensure!(
            self.scores.len() == self.config.population
                && self.evaluated <= self.scores.len()
                && self.trial_metrics.len() == self.config.population,
            "Invalid evaluation progress"
        );
        ensure!(
            self.candidate_emitters.len() == self.config.population
                && self.candidate_cma.len() == self.config.population
                && self.protected_until.len() == self.config.population,
            "Invalid QD candidate state"
        );
        ensure!(
            self.scores[..self.evaluated].iter().all(|s| s.is_finite()),
            "Invalid completed fitness values"
        );
        ensure!(
            self.scores[self.evaluated..].iter().all(|s| s.is_nan()),
            "Invalid pending fitness values"
        );
        if matches!(
            self.stage,
            Stage::Evaluated | Stage::Ranked | Stage::Selected | Stage::Archived
        ) {
            ensure!(
                self.evaluated == self.scores.len(),
                "Incomplete evaluated generation"
            );
        }
        if matches!(self.stage, Stage::Ranked | Stage::Selected) {
            ensure!(
                self.ranks.len() == self.config.population,
                "Invalid ranking length"
            );
            let mut seen = vec![false; self.ranks.len()];
            for &r in &self.ranks {
                ensure!(r < seen.len() && !seen[r], "Invalid ranking index");
                seen[r] = true;
            }
        }
        if self.stage == Stage::Selected {
            ensure!(
                self.parents.len() == self.config.population / 2
                    && self.parents.iter().all(|&i| i < self.config.population),
                "Invalid parents"
            );
        }
        if let Some(cfg) = &self.pending {
            cfg.validate()?;
            ensure!(
                cfg.population == self.config.population
                    && cfg.seed == self.config.seed
                    && cfg.random_seed == self.config.random_seed
                    && self
                        .population
                        .genomes
                        .iter()
                        .all(|g| g.node_count <= cfg.max_nodes && g.muscle_count <= cfg.max_muscles)
                    && self
                        .archive
                        .entries
                        .iter()
                        .all(|elite| elite.creature.nodes.len() <= cfg.max_nodes
                            && elite.creature.muscles.len() <= cfg.max_muscles),
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
// V5 dropped the unused obstacle slot and V6 added the environment-effect
// multipliers to the binary configuration. Older files cannot decode the new
// layout and are rejected cleanly instead of failing mid-stream.
const MAGIC: &[u8; 8] = b"EVORUST6";
const V3_MAGIC: &[u8; 8] = b"EVORUST3";
const V2_MAGIC: &[u8; 8] = b"EVORUST2";
const LEGACY_MAGIC: &[u8; 8] = b"EVORUST1";

/// Search state omitted by Experiment's original serialized representation.
/// V4 and later append it inside the same checksummed stream, preserving V3
/// decoding for the legacy conversion.
#[derive(Serialize, Deserialize)]
struct CheckpointResume {
    island_progress: Vec<(f32, u32)>,
}

fn fitness_context_changed(old: &Config, new: &Config) -> bool {
    old.duration != new.duration
        || old.gravity != new.gravity
        || old.air_retention != new.air_retention
        || old.ground_friction != new.ground_friction
        || old.ground != new.ground
        || old.terrain != new.terrain
        || old.muscle_energy != new.muscle_energy
        || old.muscle_recovery != new.muscle_recovery
        || old.slope != new.slope
        || old.wind != new.wind
        || old.mud != new.mud
        || old.gaps != new.gaps
        || old.hurdles != new.hurdles
        || old.quake != new.quake
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
pub fn save(path: &Path, experiment: &Experiment) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("evo.tmp");
    let file = File::create(&tmp)?;
    let mut out = BufWriter::new(file);
    out.write_all(MAGIC)?;
    let mut encoder = zstd::stream::write::Encoder::new(out, 3)?;
    encoder.include_checksum(true)?;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize_into(&mut encoder, experiment)?;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize_into(
            &mut encoder,
            &CheckpointResume {
                island_progress: experiment.island_progress.clone(),
            },
        )?;
    let mut out = encoder.finish()?;
    out.flush()?;
    out.get_ref().sync_all()?;
    drop(out);
    std::fs::rename(&tmp, path)?;
    // Unix permits opening directories to persist the rename. Windows rejects
    // File::open on a directory; the checkpoint file itself was synced above.
    #[cfg(unix)]
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}
pub fn load(path: &Path) -> Result<Experiment> {
    let mut file = BufReader::new(File::open(path).context("Cannot open checkpoint")?);
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    ensure!(
        &magic == MAGIC || &magic == V3_MAGIC || &magic == V2_MAGIC || &magic == LEGACY_MAGIC,
        "Unsupported checkpoint format/version"
    );
    let mut decoder = zstd::stream::read::Decoder::new(file)?;
    let mut experiment: Experiment = if &magic == LEGACY_MAGIC {
        let legacy: LegacyExperiment = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(24 * 1024 * 1024 * 1024)
            .deserialize_from(&mut decoder)?;
        legacy.into()
    } else if &magic == V2_MAGIC {
        let previous: V2Experiment = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(24 * 1024 * 1024 * 1024)
            .deserialize_from(&mut decoder)?;
        previous.into()
    } else {
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(24 * 1024 * 1024 * 1024)
            .deserialize_from(&mut decoder)?
    };
    if &magic == MAGIC {
        let resume: CheckpointResume = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            // A fixed-width vector length plus at most 64 (fitness, generation)
            // pairs. Bound corrupt metadata independently of the large body data.
            .with_limit(8 + 64 * 8)
            .deserialize_from(&mut decoder)?;
        ensure!(
            resume.island_progress.len() <= 64
                && (resume.island_progress.is_empty()
                    || resume.island_progress.len() == experiment.islands.len())
                && resume.island_progress.iter().all(|&(fitness, generation)| {
                    (fitness.is_finite() || fitness == f32::NEG_INFINITY)
                        && generation <= experiment.generation
                }),
            "Invalid checkpoint optimizer progress"
        );
        experiment.island_progress = resume.island_progress;
    }
    let mut trailing = [0u8; 1];
    ensure!(
        decoder.read(&mut trailing)? == 0,
        "Unexpected trailing checkpoint data"
    );
    // Bone projection now walks the skeleton parent-first. Normalize trees in
    // checkpoints written before that invariant was introduced.
    experiment.population.canonicalize_bones()?;
    if experiment.qd_version < qd::VERSION {
        // Older archives used prior descriptors, obstacle physics, bone
        // contact rules, or positional corrections as velocity. Reevaluate
        // their current populations under the current fitness criteria
        // instead of retaining incomparable elites.
        if experiment
            .history
            .last()
            .is_some_and(|s| s.generation == experiment.generation)
        {
            experiment.history.pop();
        }
        experiment
            .population
            .migrate_actuator_geometry(&experiment.config);
        experiment.qd_version = qd::VERSION;
        experiment.archive = QdArchive::default();
        experiment.islands.clear();
        experiment.island_progress.clear();
        experiment.reseed.clear();
        experiment.emitter_stats = [EmitterStats::default(); qd::EMITTER_COUNT];
        experiment.cma_emitters.clear();
        experiment.candidate_emitters = vec![Emitter::Restart; experiment.config.population];
        experiment.candidate_cma = vec![None; experiment.config.population];
        experiment.protected_until = vec![0; experiment.config.population];
        experiment.trial_metrics = vec![TrialMetrics::default(); experiment.config.population];
        experiment.scores.fill(f32::NAN);
        experiment.evaluated = 0;
        experiment.stage = Stage::Ready;
        experiment.ranks.clear();
        experiment.parents.clear();
        experiment.evaluation_seconds = 0.0;
    }
    experiment.parent_scores = vec![f32::NAN; experiment.config.population];
    experiment.archive.rebuild_indices();
    for island in &mut experiment.islands {
        island.rebuild_indices();
    }
    experiment.validate()?;
    Ok(experiment)
}

#[derive(Serialize, Deserialize)]
struct V2Genome {
    node_start: usize,
    node_count: usize,
    muscle_start: usize,
    muscle_count: usize,
    id: u64,
    mutability: f32,
}
#[derive(Serialize, Deserialize)]
struct V2Creature {
    nodes: Vec<crate::evolution::NodeGene>,
    muscles: Vec<LegacyMuscle>,
    id: u64,
    mutability: f32,
}
#[derive(Serialize, Deserialize)]
struct V2Population {
    genomes: Vec<V2Genome>,
    nodes: Vec<crate::evolution::NodeGene>,
    muscles: Vec<LegacyMuscle>,
}
#[derive(Serialize, Deserialize)]
struct V2Stats {
    generation: u32,
    best: f32,
    median: f32,
    worst: f32,
    mean: f32,
    failed: usize,
    seconds: f64,
    population: usize,
    percentiles: Vec<f32>,
    histogram: Vec<(i32, u32)>,
    species: Vec<(usize, usize, u32)>,
    representatives: Vec<V2Creature>,
    config: Config,
    archive_cells: usize,
    qd_score: f64,
    archive_coverage: f32,
    emitters: [EmitterStats; qd::EMITTER_COUNT],
}
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
struct V2Elite {
    niche: qd::Niche,
    descriptor: qd::Descriptor,
    creature: V2Creature,
    fitness: f32,
    emitter: Emitter,
    improved_generation: u32,
    protected_until: u32,
    visits: u64,
    topology: qd::Topology,
}
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
struct V2QdArchive {
    entries: Vec<V2Elite>,
    qd_score: f64,
}
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
struct V2CmaEmitter {
    niche: qd::Niche,
    topology: qd::Topology,
    template: V2Creature,
    mean: Vec<f32>,
    covariance: Vec<f32>,
    path_c: Vec<f32>,
    path_sigma: Vec<f32>,
    sigma: f32,
    last_used_generation: u32,
}
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
struct V2Experiment {
    config: Config,
    pending: Option<Config>,
    generation: u32,
    population: V2Population,
    scores: Vec<f32>,
    evaluated: usize,
    stage: Stage,
    ranks: Vec<usize>,
    parents: Vec<usize>,
    history: Vec<V2Stats>,
    evaluation_seconds: f64,
    archive: V2QdArchive,
    emitter_stats: [EmitterStats; qd::EMITTER_COUNT],
    cma_emitters: Vec<V2CmaEmitter>,
    candidate_emitters: Vec<Emitter>,
    candidate_cma: Vec<Option<usize>>,
    protected_until: Vec<u32>,
    trial_metrics: Vec<TrialMetrics>,
    qd_version: u32,
}
#[derive(Serialize, Deserialize)]
struct LegacyStats {
    generation: u32,
    best: f32,
    median: f32,
    worst: f32,
    mean: f32,
    failed: usize,
    seconds: f64,
    population: usize,
    percentiles: Vec<f32>,
    histogram: Vec<(i32, u32)>,
    species: Vec<(usize, usize, u32)>,
    representatives: Vec<V2Creature>,
    config: Config,
}
#[derive(Serialize, Deserialize)]
struct LegacyExperiment {
    config: Config,
    pending: Option<Config>,
    generation: u32,
    population: V2Population,
    scores: Vec<f32>,
    evaluated: usize,
    stage: Stage,
    ranks: Vec<usize>,
    parents: Vec<usize>,
    history: Vec<LegacyStats>,
    evaluation_seconds: f64,
}
fn migrate_legacy_population(old: V2Population, cfg: &Config) -> Population {
    let mut population = Population::default();
    for genome in old.genomes {
        let nodes = old.nodes[genome.node_start..genome.node_start + genome.node_count].to_vec();
        let muscles = &old.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count];
        population.push(evolution::migrate_legacy_creature(
            nodes,
            muscles,
            genome.id,
            genome.mutability,
            cfg,
        ));
    }
    population
}
fn migrate_legacy_creature(old: V2Creature, cfg: &Config) -> Creature {
    evolution::migrate_legacy_creature(old.nodes, &old.muscles, old.id, old.mutability, cfg)
}
fn migrate_legacy_stats(old: LegacyStats) -> Stats {
    let cfg = old.config.clone();
    Stats {
        generation: old.generation,
        best: old.best,
        median: old.median,
        worst: old.worst,
        mean: old.mean,
        failed: old.failed,
        seconds: old.seconds,
        population: old.population,
        percentiles: old.percentiles,
        histogram: old.histogram,
        species: old.species,
        representatives: old
            .representatives
            .into_iter()
            .map(|creature| migrate_legacy_creature(creature, &cfg))
            .collect(),
        config: old.config,
        archive_cells: 0,
        qd_score: 0.0,
        archive_coverage: 0.0,
        emitters: [EmitterStats::default(); qd::EMITTER_COUNT],
    }
}
fn migrate_v2_stats(old: V2Stats) -> Stats {
    let cfg = old.config.clone();
    Stats {
        generation: old.generation,
        best: old.best,
        median: old.median,
        worst: old.worst,
        mean: old.mean,
        failed: old.failed,
        seconds: old.seconds,
        population: old.population,
        percentiles: old.percentiles,
        histogram: old.histogram,
        species: old.species,
        representatives: old
            .representatives
            .into_iter()
            .map(|creature| migrate_legacy_creature(creature, &cfg))
            .collect(),
        config: old.config,
        archive_cells: old.archive_cells,
        qd_score: old.qd_score,
        archive_coverage: old.archive_coverage,
        emitters: old.emitters,
    }
}
impl From<V2Experiment> for Experiment {
    fn from(old: V2Experiment) -> Self {
        let population = old.config.population;
        let config = old.config;
        Self {
            population: migrate_legacy_population(old.population, &config),
            history: old.history.into_iter().map(migrate_v2_stats).collect(),
            config,
            pending: old.pending,
            generation: old.generation,
            scores: old.scores,
            parent_scores: vec![f32::NAN; population],
            evaluated: old.evaluated,
            stage: old.stage,
            ranks: old.ranks,
            parents: old.parents,
            evaluation_seconds: old.evaluation_seconds,
            archive: QdArchive::default(),
            emitter_stats: [EmitterStats::default(); qd::EMITTER_COUNT],
            cma_emitters: vec![],
            candidate_emitters: vec![Emitter::Restart; population],
            candidate_cma: vec![None; population],
            candidate_parent_ids: vec![None; population],
            morphology_reserve_override: None,
            protected_until: vec![0; population],
            trial_metrics: vec![TrialMetrics::default(); population],
            qd_version: 0,
            breed_round: 0,
            islands: Vec::new(),
            lineage: HashMap::new(),
            candidate_mates: Vec::new(),
            island_progress: Vec::new(),
            reseed: Vec::new(),
            fossils: Vec::new(),
        }
    }
}
impl From<LegacyExperiment> for Experiment {
    fn from(legacy: LegacyExperiment) -> Self {
        let population = legacy.config.population;
        let config = legacy.config;
        Self {
            config: config.clone(),
            pending: legacy.pending,
            generation: legacy.generation,
            population: migrate_legacy_population(legacy.population, &config),
            scores: legacy.scores,
            parent_scores: vec![f32::NAN; population],
            evaluated: legacy.evaluated,
            stage: legacy.stage,
            ranks: legacy.ranks,
            parents: legacy.parents,
            history: legacy
                .history
                .into_iter()
                .map(migrate_legacy_stats)
                .collect(),
            evaluation_seconds: legacy.evaluation_seconds,
            archive: QdArchive::default(),
            emitter_stats: [EmitterStats::default(); qd::EMITTER_COUNT],
            cma_emitters: vec![],
            candidate_emitters: vec![Emitter::Restart; population],
            candidate_cma: vec![None; population],
            candidate_parent_ids: vec![None; population],
            morphology_reserve_override: None,
            protected_until: vec![0; population],
            trial_metrics: vec![TrialMetrics::default(); population],
            qd_version: 0,
            breed_round: 0,
            islands: Vec::new(),
            lineage: HashMap::new(),
            candidate_mates: Vec::new(),
            island_progress: Vec::new(),
            reseed: Vec::new(),
            fossils: Vec::new(),
        }
    }
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
mod migration_tests {
    use super::*;

    #[test]
    fn v2_checkpoint_migrates_node_muscles_to_bones() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let nodes = vec![
            crate::evolution::NodeGene {
                x: 0.0,
                y: 0.0,
                diameter: 0.08,
                friction: 0.5,
            },
            crate::evolution::NodeGene {
                x: 0.3,
                y: 0.0,
                diameter: 0.08,
                friction: 0.5,
            },
            crate::evolution::NodeGene {
                x: 0.15,
                y: 0.25,
                diameter: 0.08,
                friction: 0.5,
            },
        ];
        let muscles = vec![
            LegacyMuscle {
                a: 0,
                b: 1,
                short: 0.1,
                long: 0.2,
                period: 1.0,
                phase: 0.0,
                duty: 0.5,
                stiffness: 40.0,
            },
            LegacyMuscle {
                a: 1,
                b: 2,
                short: 0.1,
                long: 0.2,
                period: 1.0,
                phase: 0.2,
                duty: 0.5,
                stiffness: 40.0,
            },
            LegacyMuscle {
                a: 2,
                b: 0,
                short: 0.1,
                long: 0.2,
                period: 1.0,
                phase: 0.4,
                duty: 0.5,
                stiffness: 40.0,
            },
        ];
        let mut old_population = V2Population {
            genomes: Vec::new(),
            nodes: Vec::new(),
            muscles: Vec::new(),
        };
        for id in 1..=2 {
            let node_start = old_population.nodes.len();
            let muscle_start = old_population.muscles.len();
            old_population.nodes.extend_from_slice(&nodes);
            old_population.muscles.extend_from_slice(&muscles);
            old_population.genomes.push(V2Genome {
                node_start,
                node_count: nodes.len(),
                muscle_start,
                muscle_count: muscles.len(),
                id,
                mutability: 1.0,
            });
        }
        let old = V2Experiment {
            config: config.clone(),
            pending: None,
            generation: 5,
            population: old_population,
            scores: vec![1.0, 1.0],
            evaluated: 2,
            stage: Stage::Evaluated,
            ranks: vec![],
            parents: vec![],
            history: vec![],
            evaluation_seconds: 1.0,
            archive: V2QdArchive {
                entries: vec![],
                qd_score: 0.0,
            },
            emitter_stats: [EmitterStats::default(); qd::EMITTER_COUNT],
            cma_emitters: vec![],
            candidate_emitters: vec![Emitter::Cma; 2],
            candidate_cma: vec![None; 2],
            protected_until: vec![0; 2],
            trial_metrics: vec![TrialMetrics::default(); 2],
            qd_version: qd::VERSION - 1,
        };
        let payload = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize(&old)
            .unwrap();
        let compressed = zstd::stream::encode_all(payload.as_slice(), 3).unwrap();
        let mut bytes = V2_MAGIC.to_vec();
        bytes.extend(compressed);
        let checkpoint =
            std::env::temp_dir().join(format!("evolution-v2-migration-{}.evo", std::process::id()));
        std::fs::write(&checkpoint, bytes).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);

        assert_eq!(loaded.qd_version, qd::VERSION);
        assert_eq!(loaded.stage, Stage::Ready);
        assert!(loaded.scores.iter().all(|score| score.is_nan()));
        assert_eq!(loaded.population.genomes.len(), 2);
        loaded.population.validate(&config).unwrap();
        for genome in &loaded.population.genomes {
            assert_eq!(genome.bone_count, genome.node_count - 1);
        }
    }

    #[test]
    fn old_checkpoint_keeps_historical_representatives_with_missing_muscles() {
        let config = Config {
            population: 4,
            random_seed: false,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.scores.fill(1.0);
        experiment.evaluated = experiment.config.population;
        experiment.stage = Stage::Evaluated;
        experiment.archive_batch().unwrap();
        experiment.prepare_next_batch().unwrap();
        experiment.history[0].representatives[0].muscles.clear();
        experiment.qd_version = qd::VERSION - 1;
        let checkpoint = std::env::temp_dir().join(format!(
            "evolution-disconnected-history-{}.evo",
            std::process::id()
        ));
        save(&checkpoint, &experiment).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);
        assert_eq!(loaded.generation, 1);
        assert_eq!(loaded.history.len(), 1);
        assert!(loaded.history[0].representatives[0].muscles.is_empty());
        loaded.validate().unwrap();
    }

    #[test]
    fn current_checkpoint_normalizes_bone_order_and_keeps_attachments_in_place() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config.clone()).unwrap();
        let genome = experiment.population.genomes[0].clone();
        let bone_range = genome.bone_start..genome.bone_start + genome.bone_count;
        experiment.population.bones[bone_range.clone()].reverse();
        for bone in &mut experiment.population.bones[bone_range] {
            std::mem::swap(&mut bone.a, &mut bone.b);
        }
        let old_creature = experiment.population.creature(0);
        let point = |creature: &crate::evolution::Creature, bone_id: u32, t: f32| {
            let bone = creature.bones[bone_id as usize];
            let a = creature.nodes[bone.a as usize];
            let b = creature.nodes[bone.b as usize];
            [a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t]
        };
        let old_points: Vec<_> = old_creature
            .muscles
            .iter()
            .map(|muscle| {
                [
                    point(&old_creature, muscle.bone_a, muscle.anchor_a),
                    point(&old_creature, muscle.bone_b, muscle.anchor_b),
                ]
            })
            .collect();
        experiment.qd_version = qd::VERSION - 1;
        let checkpoint = std::env::temp_dir().join(format!(
            "evolution-v3-bone-order-{}.evo",
            std::process::id()
        ));
        save(&checkpoint, &experiment).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);

        assert_eq!(loaded.qd_version, qd::VERSION);
        assert!(loaded.scores.iter().all(|score| score.is_nan()));
        loaded.population.validate(&config).unwrap();
        let new_creature = loaded.population.creature(0);
        for (muscle, points) in new_creature.muscles.iter().zip(old_points) {
            let actual = [
                point(&new_creature, muscle.bone_a, muscle.anchor_a),
                point(&new_creature, muscle.bone_b, muscle.anchor_b),
            ];
            for side in 0..2 {
                assert!((actual[side][0] - points[side][0]).abs() < 1e-6);
                assert!((actual[side][1] - points[side][1]).abs() < 1e-6);
            }
        }
    }
}
