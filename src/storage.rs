use crate::{
    config::Config,
    evolution::{self, CandidatePlan, Creature, FAILED, LegacyMuscle, Population, Rng},
    qd::{self, CmaEmitter, Emitter, EmitterStats, QdArchive, TrialMetrics},
};
use anyhow::{Context, Result, ensure};
use bincode::Options;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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
        let population_count = self.config.population;
        let reserve_enabled = self.morphology_reserve_override != Some(false);
        let prep: Vec<Prep> = (0..population_count)
            .into_par_iter()
            .map(|i| {
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
        let mut attempts = [0u64; qd::EMITTER_COUNT];
        let mut failed = 0usize;
        for (i, prep) in prep.iter().enumerate() {
            if !prep.score.is_finite() || prep.score <= FAILED {
                failed += 1;
            }
            let emitter_index = prep.emitter.index();
            attempts[emitter_index] += 1;
            if prep.emitter == Emitter::Cma
                && let Some(cma) = self.candidate_cma.get(i).copied().flatten()
                && let Some(samples) = cma_samples.get_mut(cma)
            {
                samples.push((i, prep.score));
            }
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
            if offer.inserted {
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
        self.push_archive_stats(failed);
        self.stage = Stage::Archived;
        Ok(())
    }
    fn push_archive_stats(&mut self, failed: usize) {
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
    pub fn prepare_next_batch(&mut self) -> Result<()> {
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
        let weights = qd::emitter_weights(&self.emitter_stats);
        let mut reset_cma = HashMap::<(qd::Niche, qd::Topology), usize>::new();
        let mut cma_lookup: HashMap<(qd::Niche, qd::Topology), usize> = self
            .cma_emitters
            .iter()
            .enumerate()
            .map(|(i, cma)| ((cma.niche.clone(), cma.topology.clone()), i))
            .collect();
        let mut used_cma = vec![false; self.cma_emitters.len()];
        let mut plans = Vec::with_capacity(cfg.population);
        let mut emitters = Vec::with_capacity(cfg.population);
        let mut cma_indices = Vec::with_capacity(cfg.population);
        let mut parent_ids = Vec::with_capacity(cfg.population);
        let mut protections = Vec::with_capacity(cfg.population);
        let setup_seconds = preparation_started.elapsed().as_secs_f64();
        let plan_started = std::time::Instant::now();
        // Phase A: emitter choice and parent sampling against the start-of-batch
        // archive. Each creature has its own deterministic RNG, so parallel order
        // does not change the draws. last_parent is snapshotted instead of updating
        // mid-loop; visit() and CMA slot allocation stay sequential below.
        let archive_empty = self.archive.entries.is_empty();
        let reserve_enabled = self.morphology_reserve_override != Some(false);
        let last_parents: [Option<usize>; qd::EMITTER_COUNT] =
            std::array::from_fn(|i| self.emitter_stats[i].last_parent);
        struct PlanPrep {
            emitter: Emitter,
            parent: Option<usize>,
            parent_id: Option<u64>,
            protection: u32,
            emitter_stale: bool,
        }
        let plan_prep: Vec<PlanPrep> = (0..cfg.population)
            .into_par_iter()
            .map(|i| {
                let mut rng = Rng::new(cfg.seed, generation, i);
                let emitter = if archive_empty {
                    Emitter::Restart
                } else {
                    qd::choose_emitter(&mut rng, &weights)
                };
                let emitter_stale = self.emitter_stats[emitter.index()].stale();
                let avoid = last_parents[emitter.index()];
                let parent = if emitter == Emitter::Restart || archive_empty {
                    None
                } else if reserve_enabled
                    && emitter == Emitter::Structural
                    && rng.unit() < qd::MORPHOLOGY_PARENT_FRACTION
                {
                    self.archive
                        .sample_morphology(&mut rng, avoid)
                        .or_else(|| self.archive.sample_local_competitive(&mut rng, avoid))
                } else if emitter == Emitter::Novelty || emitter_stale {
                    self.archive.sample_novel(&mut rng, avoid)
                } else {
                    self.archive.sample_local_competitive(&mut rng, avoid)
                };
                let parent_id = parent.map(|index| self.archive.entries[index].creature.id);
                let protection = if matches!(emitter, Emitter::Structural | Emitter::Novelty) {
                    generation.saturating_add(3)
                } else {
                    parent
                        .map(|index| self.archive.entries[index].protected_until)
                        .unwrap_or(0)
                };
                PlanPrep {
                    emitter,
                    parent,
                    parent_id,
                    protection,
                    emitter_stale,
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
            } = prep;
            parent_ids.push(parent_id);
            let cma_index = if emitter == Emitter::Cma {
                if let Some(parent_index) = parent {
                    let elite = &self.archive.entries[parent_index];
                    let template = &elite.creature;
                    let topology = &elite.topology;
                    let niche_key = (elite.niche.clone(), topology.clone());
                    let mut index = if emitter_stale {
                        reset_cma.get(&niche_key).copied()
                    } else {
                        cma_lookup.get(&niche_key).copied()
                    };
                    if index.is_none() {
                        let replacement = if self.cma_emitters.len() < qd::CMA_LIMIT {
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
                            let new =
                                CmaEmitter::new(template.clone(), elite.niche.clone(), generation);
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
                            if emitter_stale {
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
                self.archive.visit(parent_index);
                self.emitter_stats[emitter.index()].last_parent = Some(parent_index);
            }
            plans.push(CandidatePlan {
                emitter,
                parent,
                cma: cma_index,
            });
            emitters.push(emitter);
            cma_indices.push(cma_index);
            protections.push(protection);
        }
        let plan_seconds = plan_started.elapsed().as_secs_f64();
        let emission_started = std::time::Instant::now();
        let next = evolution::emit_archive_batch(
            &self.population,
            &self.archive,
            &self.cma_emitters,
            &plans,
            &cfg,
            generation,
        )?;
        let emission_seconds = emission_started.elapsed().as_secs_f64();
        self.config = cfg;
        self.pending = None;
        self.population = next;
        self.candidate_emitters = emitters;
        self.candidate_cma = cma_indices;
        self.candidate_parent_ids = parent_ids;
        self.protected_until = protections;
        self.parent_scores.fill(f32::NAN);
        self.generation = generation;
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
    pub fn select(&mut self) {
        self.parents = evolution::survivors(&self.config, self.generation, &self.ranks);
        self.stage = Stage::Selected;
    }
    pub fn reproduce(&mut self) -> Result<()> {
        let cfg = self.pending.as_ref().unwrap_or(&self.config);
        let next = evolution::reproduce(&self.population, cfg, self.generation, &self.parents)?;
        self.parent_scores = self
            .parents
            .iter()
            .flat_map(|&parent| [self.scores[parent]; 2])
            .collect();
        self.population = next;
        if let Some(c) = self.pending.take() {
            self.config = c;
        }
        self.generation += 1;
        self.stage = Stage::Ready;
        self.evaluated = 0;
        self.scores.fill(f32::NAN);
        self.ranks.clear();
        self.parents.clear();
        self.evaluation_seconds = 0.0;
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
    fn reset_search_context(&mut self) {
        self.archive = QdArchive::default();
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
            representatives.validate_with_max_bone(&representative_config, 12.0)?;
        }
        Ok(())
    }
}
const MAGIC: &[u8; 8] = b"EVORUST3";
const V2_MAGIC: &[u8; 8] = b"EVORUST2";
const LEGACY_MAGIC: &[u8; 8] = b"EVORUST1";
fn fitness_context_changed(old: &Config, new: &Config) -> bool {
    old.duration != new.duration
        || old.gravity != new.gravity
        || old.air_retention != new.air_retention
        || old.ground_friction != new.ground_friction
        || old.ground != new.ground
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
    let mut out = encoder.finish()?;
    out.flush()?;
    out.get_ref().sync_all()?;
    drop(out);
    std::fs::rename(&tmp, path)?;
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
        &magic == MAGIC || &magic == V2_MAGIC || &magic == LEGACY_MAGIC,
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
