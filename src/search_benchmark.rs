//! Reproducible, headless measurements of evolutionary search quality.

use crate::{
    config::Config,
    evolution::{Creature, FAILED, Population},
    gpu::Gpu,
    qd::{self, Elite, Emitter, EmitterStats, Topology},
    storage::{self, Experiment},
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const EMITTERS: [Emitter; qd::EMITTER_COUNT] = Emitter::ALL;

#[derive(Clone, Debug, Serialize)]
pub struct MorphologySnapshot {
    pub count: usize,
    /// Distinct node-indexed undirected graph topologies; edge order is ignored.
    pub unique_topologies: usize,
    pub topology_entropy_bits: f64,
    pub effective_topology_count: f64,
    pub triangle_like_count: usize,
    pub triangle_like_fraction: f64,
    pub mean_nodes: f64,
    pub mean_muscles: f64,
    pub node_counts: BTreeMap<String, usize>,
    pub muscle_counts: BTreeMap<String, usize>,
}

#[derive(Default)]
struct MorphologyAccumulator {
    count: usize,
    nodes: BTreeMap<String, usize>,
    muscles: BTreeMap<String, usize>,
    topology_counts: HashMap<String, usize>,
    triangle_like_count: usize,
    node_sum: u64,
    muscle_sum: u64,
}

impl MorphologyAccumulator {
    fn add(&mut self, nodes: usize, muscles: usize, topology: &Topology) {
        self.count += 1;
        self.node_sum += nodes as u64;
        self.muscle_sum += muscles as u64;
        *self.nodes.entry(nodes.to_string()).or_default() += 1;
        *self.muscles.entry(muscles.to_string()).or_default() += 1;
        *self
            .topology_counts
            .entry(topology_key(topology))
            .or_default() += 1;
        if triangle_like(topology) {
            self.triangle_like_count += 1;
        }
    }

    fn finish(self) -> MorphologySnapshot {
        let entropy = shannon_entropy(self.topology_counts.values().copied());
        MorphologySnapshot {
            count: self.count,
            unique_topologies: self.topology_counts.len(),
            topology_entropy_bits: entropy,
            effective_topology_count: entropy.exp2(),
            triangle_like_count: self.triangle_like_count,
            triangle_like_fraction: ratio(self.triangle_like_count, self.count),
            mean_nodes: self.node_sum as f64 / self.count.max(1) as f64,
            mean_muscles: self.muscle_sum as f64 / self.count.max(1) as f64,
            node_counts: self.nodes,
            muscle_counts: self.muscles,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Milestone {
    distance_m: f32,
    generation: u32,
    evaluations: u64,
    search_wall_seconds: f64,
    benchmark_wall_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
struct EmitterSnapshot {
    attempts: u64,
    discoveries: u64,
    improvements: u64,
    insertions: u64,
    insertion_rate: f64,
}

#[derive(Serialize)]
struct GenerationRow {
    seed: u64,
    generation: u32,
    evaluations: u64,
    search_wall_seconds: f64,
    generation_search_seconds: f64,
    gpu_evaluation_seconds: f64,
    archive_seconds: f64,
    offspring_seconds: f64,
    benchmark_wall_seconds: f64,
    best_distance_m: f32,
    median_elite_distance_m: f32,
    qd_score: f64,
    archive_cells: usize,
    archive_coverage: f32,
    archive_unique_topologies: usize,
    archive_topology_entropy_bits: f64,
    archive_triangle_like_count: usize,
    archive_triangle_like_fraction: f64,
    archive_node_counts_json: String,
    archive_muscle_counts_json: String,
    innovation_reserve_count: usize,
    innovation_reserve_morphology_json: String,
    population_unique_topologies: usize,
    population_topology_entropy_bits: f64,
    population_triangle_like_count: usize,
    population_triangle_like_fraction: f64,
    population_node_counts_json: String,
    population_muscle_counts_json: String,
    emitter_statistics_json: String,
    topology_change_candidates_json: String,
    archived_innovations_this_generation: u64,
    innovation_survival_3gen_fraction: f64,
    new_all_time_record: bool,
    evaluations_since_previous_record: u64,
}

#[derive(Clone, Debug, Serialize)]
struct InnovationRecord {
    id: u64,
    parent_innovation_id: Option<u64>,
    generation: u32,
    emitter: String,
    topology: String,
    nodes: usize,
    muscles: usize,
    descendant_evaluations: u64,
    viable_descendant_evaluations: u64,
    archive_generations: u32,
    survived_three_generations: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ParentTopologyOutcome {
    topology: String,
    nodes: usize,
    muscles: usize,
    evaluations: u64,
    viable: u64,
    improved_on_parent: u64,
    best_parent_distance_m: f32,
    best_offspring_distance_m: f32,
}

#[derive(Clone, Debug, Serialize)]
struct SeedSummary {
    seed: u64,
    population: usize,
    generations: u32,
    evaluations: u64,
    population_creation_seconds: f64,
    gpu_warmup_seconds: f64,
    search_wall_seconds: f64,
    benchmark_wall_seconds: f64,
    best_distance_m: f32,
    median_elite_distance_m: f32,
    qd_score: f64,
    archive_coverage: f32,
    record_count: u64,
    evaluations_since_last_record: u64,
    time_to_milestones: BTreeMap<String, Option<Milestone>>,
    final_archive_morphology: MorphologySnapshot,
    final_innovation_reserve_morphology: MorphologySnapshot,
    final_population_morphology: MorphologySnapshot,
    structural_topology_candidates: u64,
    /// First-time, distinct topologies admitted from structural offspring.
    archived_structural_innovations: u64,
    /// Previously observed topologies that re-entered after their innovation lineage left the archive.
    reintroduced_structural_lineages: u64,
    innovations_eligible_for_three_generations: u64,
    innovations_surviving_three_generations: u64,
    innovation_survival_fraction: f64,
    final_emitter_statistics: BTreeMap<String, EmitterSnapshot>,
}

#[derive(Clone, Debug, Serialize)]
struct RunMetadata {
    algorithm_variant: &'static str,
    morphology_parent_fraction: f32,
    minimum_morphology_descendants_before_eviction: u64,
    created_unix_seconds: u64,
    gpu: String,
    logical_cpus: usize,
    seeds: Vec<u64>,
    generations: u32,
    evaluation_budget_per_seed: u64,
    config: Config,
    notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CheckpointAnalysis {
    pub seed: u64,
    pub generation: u32,
    pub population: usize,
    pub duration_seconds: f32,
    pub archive_best_distance_m: f32,
    pub archive_median_distance_m: f32,
    pub qd_score: f64,
    pub archive_coverage: f32,
    pub archive_morphology: MorphologySnapshot,
    pub innovation_reserve_morphology: MorphologySnapshot,
    pub champion: ChampionSummary,
    pub record_count: usize,
    pub records: Vec<(u32, u64, f32)>,
    pub time_to_milestones: BTreeMap<String, Option<Milestone>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChampionSummary {
    pub id: u64,
    pub distance_m: f32,
    pub nodes: usize,
    pub muscles: usize,
    pub topology: String,
}

pub fn run(
    gpu_name: &str,
    mut config: Config,
    seeds: &[u64],
    generations: u32,
    milestones: &[f32],
    output_dir: &Path,
    morphology_reserve_enabled: bool,
) -> Result<()> {
    ensure!(!seeds.is_empty(), "At least one fixed seed is required");
    ensure!(generations > 0, "Generation count must be positive");
    ensure!(
        seeds.iter().copied().collect::<HashSet<_>>().len() == seeds.len(),
        "Seeds must be unique"
    );
    ensure!(
        milestones.iter().all(|m| m.is_finite() && *m >= 0.0),
        "Milestones must be finite and nonnegative"
    );
    config.random_seed = false;
    config.throughput = true;
    config.validate()?;
    ensure!(
        config.population <= u32::MAX as usize,
        "Population is too large for deterministic candidate IDs"
    );
    if output_dir.exists() {
        ensure!(
            fs::read_dir(output_dir)?.next().is_none(),
            "Benchmark output directory must be empty: {}",
            output_dir.display()
        );
    } else {
        fs::create_dir_all(output_dir)?;
    }

    let mut gpu = Gpu::new(gpu_name)?;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let metadata = RunMetadata {
        algorithm_variant: if morphology_reserve_enabled {
            "morphology-reserve"
        } else {
            "behavior-only"
        },
        morphology_parent_fraction: if morphology_reserve_enabled {
            qd::MORPHOLOGY_PARENT_FRACTION
        } else {
            0.0
        },
        minimum_morphology_descendants_before_eviction: if morphology_reserve_enabled {
            qd::MIN_MORPHOLOGY_DESCENDANTS
        } else {
            0
        },
        created_unix_seconds: created,
        gpu: gpu.name.clone(),
        logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
        seeds: seeds.to_vec(),
        generations,
        evaluation_budget_per_seed: config.population as u64 * generations as u64,
        config: config.clone(),
        notes: "GPU evaluation uses throughput mode; physics and fitness are unchanged. Search wall time includes evaluation, archive insertion, and offspring creation. Benchmark wall time also includes metrics and file output.",
    };
    write_json(&output_dir.join("metadata.json"), &metadata)?;

    let mut summaries = Vec::with_capacity(seeds.len());
    for (seed_index, &seed) in seeds.iter().enumerate() {
        let mut seed_config = config.clone();
        seed_config.seed = seed;
        seed_config.random_seed = false;
        let summary = run_seed(
            &mut gpu,
            seed_config,
            generations,
            milestones,
            output_dir,
            morphology_reserve_enabled,
            seed_index == 0,
        )?;
        println!(
            "seed={} best={:.3}m median={:.3}m qd={:.1} coverage={:.1}% records={} search={:.2}s",
            summary.seed,
            summary.best_distance_m,
            summary.median_elite_distance_m,
            summary.qd_score,
            summary.archive_coverage * 100.0,
            summary.record_count,
            summary.search_wall_seconds
        );
        summaries.push(summary);
    }
    write_json(
        &output_dir.join("summary.json"),
        &aggregate_summaries(&summaries),
    )?;
    Ok(())
}

fn run_seed(
    gpu: &mut Gpu,
    config: Config,
    generations: u32,
    milestones: &[f32],
    output_dir: &Path,
    morphology_reserve_enabled: bool,
    warm_up: bool,
) -> Result<SeedSummary> {
    let seed = config.seed;
    let seed_dir = output_dir.join(format!("seed-{seed}"));
    fs::create_dir_all(&seed_dir)?;
    let benchmark_started = Instant::now();
    let creation_started = Instant::now();
    let mut experiment = Experiment::new(config.clone())?;
    experiment.morphology_reserve_override = Some(morphology_reserve_enabled);
    let creation_seconds = creation_started.elapsed().as_secs_f64();
    let warmup_started = Instant::now();
    if warm_up {
        gpu.evaluate(
            &experiment.population,
            &(0..experiment.config.population.min(256)).collect::<Vec<_>>(),
            &experiment.config,
        )?;
    }
    let warmup_seconds = warmup_started.elapsed().as_secs_f64();

    let curve_path = seed_dir.join("curve.csv");
    let mut curve = csv::WriterBuilder::new()
        .has_headers(false)
        .from_path(&curve_path)
        .with_context(|| format!("Creating {}", curve_path.display()))?;
    curve.write_record([
        "seed",
        "generation",
        "evaluations",
        "search_wall_seconds",
        "generation_search_seconds",
        "gpu_evaluation_seconds",
        "archive_seconds",
        "offspring_seconds",
        "benchmark_wall_seconds",
        "best_distance_m",
        "median_elite_distance_m",
        "qd_score",
        "archive_cells",
        "archive_coverage",
        "archive_unique_topologies",
        "archive_topology_entropy_bits",
        "archive_triangle_like_count",
        "archive_triangle_like_fraction",
        "archive_node_counts_json",
        "archive_muscle_counts_json",
        "innovation_reserve_count",
        "innovation_reserve_morphology_json",
        "population_unique_topologies",
        "population_topology_entropy_bits",
        "population_triangle_like_count",
        "population_triangle_like_fraction",
        "population_node_counts_json",
        "population_muscle_counts_json",
        "emitter_statistics_json",
        "topology_change_candidates_json",
        "archived_innovations_this_generation",
        "innovation_survival_3gen_fraction",
        "new_all_time_record",
        "evaluations_since_previous_record",
    ])?;

    let mut genealogy = BufWriter::new(File::create(seed_dir.join("genealogy.jsonl"))?);
    let mut parent_topology_outcomes: BTreeMap<String, ParentTopologyOutcome> = BTreeMap::new();
    let mut innovation_records: BTreeMap<u64, InnovationRecord> = BTreeMap::new();
    let mut lineage_by_elite = HashMap::<u64, u64>::new();
    let mut known_archive_ids = HashSet::<u64>::new();
    let mut seen_topology_keys = HashSet::<String>::new();
    let mut search_wall_seconds = 0.0f64;
    let mut previous_best = f32::NEG_INFINITY;
    let mut previous_record_evaluation = 0u64;
    let mut record_count = 0u64;
    let mut evaluations_since_last_record = 0u64;
    let mut reached: BTreeMap<String, Option<Milestone>> = milestones
        .iter()
        .map(|m| (milestone_key(*m), None))
        .collect();
    let mut structural_topology_candidates = 0u64;
    let mut archived_structural_innovations = 0u64;
    let mut reintroduced_structural_lineages = 0u64;
    let mut innovations_eligible_for_three_generations = 0u64;
    let mut innovations_surviving_three_generations = 0u64;
    let mut final_population_morphology = empty_morphology();
    let mut final_archive_morphology = empty_morphology();
    let mut final_innovation_reserve_morphology = empty_morphology();
    let mut final_median = 0.0;
    let mut final_best = 0.0;
    let mut final_qd = 0.0;
    let mut final_coverage = 0.0;
    let mut final_emitters = BTreeMap::new();
    let mut measured_emitter_totals = [EmitterStats::default(); qd::EMITTER_COUNT];

    for _ in 0..generations {
        let generation = experiment.generation;
        let evaluation_started = Instant::now();
        let batch_size = experiment.config.batch_size();
        for begin in (0..experiment.config.population).step_by(batch_size) {
            let end = (begin + batch_size).min(experiment.config.population);
            let metrics = gpu.evaluate_with_metrics(
                &experiment.population,
                &(begin..end).collect::<Vec<_>>(),
                &experiment.config,
            )?;
            for (offset, metric) in metrics.iter().enumerate() {
                experiment.scores[begin + offset] = metric.fitness;
                experiment.trial_metrics[begin + offset] = metric.behavior;
            }
        }
        let gpu_seconds = evaluation_started.elapsed().as_secs_f64();
        experiment.evaluated = experiment.config.population;

        let old_topology_by_id: HashMap<u64, String> = experiment
            .archive
            .entries
            .iter()
            .map(|elite| (elite.creature.id, topology_key(&elite.topology)))
            .collect();
        let old_elite_by_id: HashMap<u64, (&Topology, f32)> = experiment
            .archive
            .entries
            .iter()
            .map(|elite| (elite.creature.id, (&elite.topology, elite.fitness)))
            .collect();
        let old_lineage_by_elite = lineage_by_elite.clone();
        let mut current_root_by_topology: HashMap<String, u64> = old_lineage_by_elite
            .iter()
            .filter_map(|(id, root)| {
                old_topology_by_id
                    .get(id)
                    .map(|topology| (topology.clone(), *root))
            })
            .collect();
        let (candidate_topologies, topology_change_candidates, candidate_viability) =
            measure_candidate_morphologies(&experiment, &old_topology_by_id, &old_elite_by_id);
        for (id, topology, valid, improved, parent_fitness, offspring_fitness) in
            candidate_viability
        {
            let entry = parent_topology_outcomes
                .entry(topology.clone())
                .or_insert_with(|| {
                    let (nodes, muscles) = topology_dimensions(&topology);
                    ParentTopologyOutcome {
                        topology,
                        nodes,
                        muscles,
                        evaluations: 0,
                        viable: 0,
                        improved_on_parent: 0,
                        best_parent_distance_m: f32::NEG_INFINITY,
                        best_offspring_distance_m: f32::NEG_INFINITY,
                    }
                });
            let _ = id;
            entry.evaluations += 1;
            entry.viable += u64::from(valid);
            entry.improved_on_parent += u64::from(improved);
            entry.best_parent_distance_m = entry.best_parent_distance_m.max(parent_fitness);
            entry.best_offspring_distance_m =
                entry.best_offspring_distance_m.max(offspring_fitness);
        }
        structural_topology_candidates += topology_change_candidates.values().sum::<u64>();

        let emitter_stats_before = experiment.emitter_stats;
        let archive_started = Instant::now();
        experiment.archive_batch()?;
        let archive_seconds = archive_started.elapsed().as_secs_f64();
        let stats = experiment
            .history
            .last()
            .context("Missing generation statistics")?
            .clone();

        // Parent IDs are archive elite IDs. Track which structural changes
        // actually entered the archive, then follow same-topology descendants.
        for (index, parent_id) in experiment.candidate_parent_ids.iter().enumerate() {
            let Some(parent_id) = parent_id else { continue };
            let Some(root_id) = old_lineage_by_elite.get(parent_id).copied() else {
                continue;
            };
            let child_id = experiment.population.genomes[index].id;
            let child_topology = candidate_topologies.get(index).map(String::as_str);
            if old_topology_by_id.get(parent_id).map(String::as_str) != child_topology {
                continue;
            }
            if let Some(innovation) = innovation_records.get_mut(&root_id) {
                innovation.descendant_evaluations += 1;
                let score = experiment.scores[index];
                if score.is_finite() && score > FAILED {
                    innovation.viable_descendant_evaluations += 1;
                }
            }
            let _ = child_id;
        }

        let generation_emitter_stats =
            emitter_delta(&experiment.emitter_stats, &emitter_stats_before);
        let emitter_rows = if generation == 0 {
            let initial = generation_emitter_stats
                .get("Immigrant")
                .cloned()
                .unwrap_or(EmitterSnapshot {
                    attempts: experiment.config.population as u64,
                    discoveries: 0,
                    improvements: 0,
                    insertions: 0,
                    insertion_rate: 0.0,
                });
            BTreeMap::from([("Initial population".to_string(), initial)])
        } else {
            accumulate_emitter_counts(
                &mut measured_emitter_totals,
                &experiment.emitter_stats,
                &emitter_stats_before,
            );
            generation_emitter_stats
        };

        let mut current_lineage_by_elite = HashMap::<u64, u64>::new();
        let population_index_by_id: HashMap<u64, usize> = experiment
            .population
            .genomes
            .iter()
            .enumerate()
            .map(|(index, genome)| (genome.id, index))
            .collect();
        let mut archived_innovations_this_generation = 0u64;
        for elite in &experiment.archive.entries {
            if let Some(root) = old_lineage_by_elite.get(&elite.creature.id) {
                current_lineage_by_elite.insert(elite.creature.id, *root);
                continue;
            }
            if known_archive_ids.contains(&elite.creature.id) {
                if let Some(root) = old_lineage_by_elite.get(&elite.creature.id) {
                    current_lineage_by_elite.insert(elite.creature.id, *root);
                }
                continue;
            }
            let Some(&index) = population_index_by_id.get(&elite.creature.id) else {
                continue;
            };
            let parent_id = experiment
                .candidate_parent_ids
                .get(index)
                .copied()
                .flatten();
            let emitter = experiment
                .candidate_emitters
                .get(index)
                .copied()
                .unwrap_or(elite.emitter);
            let child_topology = topology_key(&elite.topology);
            let parent_topology = parent_id.and_then(|id| old_topology_by_id.get(&id));
            let parent_innovation_id =
                parent_id.and_then(|id| old_lineage_by_elite.get(&id).copied());
            let structural_change = parent_topology.is_some_and(|parent| parent != &child_topology)
                && matches!(emitter, Emitter::Structural | Emitter::Novelty);
            let existing_root = current_root_by_topology.get(&child_topology).copied();
            let structural_innovation = structural_change && existing_root.is_none();
            let root_id = if structural_innovation {
                if seen_topology_keys.contains(&child_topology) {
                    reintroduced_structural_lineages += 1;
                } else {
                    archived_structural_innovations += 1;
                }
                archived_innovations_this_generation += 1;
                let root = elite.creature.id;
                current_root_by_topology.insert(child_topology.clone(), root);
                innovation_records.insert(
                    root,
                    InnovationRecord {
                        id: root,
                        parent_innovation_id,
                        generation,
                        emitter: emitter.label().to_string(),
                        topology: child_topology.clone(),
                        nodes: elite.creature.nodes.len(),
                        muscles: elite.creature.muscles.len(),
                        descendant_evaluations: 0,
                        viable_descendant_evaluations: 0,
                        archive_generations: 0,
                        survived_three_generations: false,
                    },
                );
                Some(root)
            } else if structural_change {
                existing_root.or(parent_innovation_id)
            } else if parent_topology.is_some_and(|parent| parent == &child_topology) {
                parent_innovation_id
            } else {
                None
            };
            seen_topology_keys.insert(child_topology.clone());
            if let Some(root) = root_id {
                current_lineage_by_elite.insert(elite.creature.id, root);
            }
            if known_archive_ids.insert(elite.creature.id) {
                let event = serde_json::json!({
                    "seed": seed,
                    "generation": generation,
                    "creature_id": elite.creature.id,
                    "parent_id": parent_id,
                    "parent_innovation_id": parent_innovation_id,
                    "innovation_id": root_id,
                    "emitter": emitter.label(),
                    "fitness_m": elite.fitness,
                    "nodes": elite.creature.nodes.len(),
                    "muscles": elite.creature.muscles.len(),
                    "topology": child_topology,
                    "structural_innovation": structural_innovation,
                    "protected_until": elite.protected_until,
                });
                serde_json::to_writer(&mut genealogy, &event)?;
                genealogy.write_all(b"\n")?;
            }
        }
        lineage_by_elite = current_lineage_by_elite;
        for root_id in lineage_by_elite.values().copied().collect::<HashSet<_>>() {
            if let Some(innovation) = innovation_records.get_mut(&root_id) {
                innovation.archive_generations += 1;
                if generation >= innovation.generation.saturating_add(3)
                    && !innovation.survived_three_generations
                {
                    innovation.survived_three_generations = true;
                    innovations_surviving_three_generations += 1;
                }
            }
        }
        innovations_eligible_for_three_generations = innovation_records
            .values()
            .filter(|item| generation >= item.generation.saturating_add(3))
            .count() as u64;

        let archive_morphology = archive_morphology(&experiment.archive.entries);
        let innovation_reserve_morphology =
            innovation_reserve_morphology(&experiment.archive.entries);
        let population_morphology = population_morphology(&experiment.population);
        let best = stats.best;
        let is_record = best > previous_best;
        let evaluations = (generation as u64 + 1) * experiment.config.population as u64;
        if is_record {
            record_count += 1;
            evaluations_since_last_record = evaluations.saturating_sub(previous_record_evaluation);
            previous_record_evaluation = evaluations;
            previous_best = best;
            let champion = experiment
                .archive
                .entries
                .iter()
                .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
                .context("The archive has no record holder")?;
            write_json(
                &seed_dir.join(format!("record-gen-{generation:06}.json")),
                &serde_json::json!({
                    "seed": seed,
                    "generation": generation,
                    "evaluations": evaluations,
                    "distance_m": champion.fitness,
                    "morphology": archive_morphology,
                    "innovation_reserve_morphology": innovation_reserve_morphology,
                    "creature": champion.creature,
                }),
            )?;
        } else {
            evaluations_since_last_record = evaluations.saturating_sub(previous_record_evaluation);
        }

        let offspring_started = Instant::now();
        experiment.prepare_next_batch()?;
        let offspring_seconds = offspring_started.elapsed().as_secs_f64();
        let generation_search_seconds = gpu_seconds + archive_seconds + offspring_seconds;
        search_wall_seconds += generation_search_seconds;
        let benchmark_wall_seconds = benchmark_started.elapsed().as_secs_f64();
        for (key, milestone) in reached.iter_mut() {
            if milestone.is_none() {
                let threshold = key.parse::<f32>().unwrap_or(f32::INFINITY);
                if best >= threshold {
                    *milestone = Some(Milestone {
                        distance_m: threshold,
                        generation,
                        evaluations,
                        search_wall_seconds,
                        benchmark_wall_seconds,
                    });
                }
            }
        }

        let topology_change_json = serde_json::to_string(&topology_change_candidates)?;
        curve.serialize(GenerationRow {
            seed,
            generation,
            evaluations,
            search_wall_seconds,
            generation_search_seconds,
            gpu_evaluation_seconds: gpu_seconds,
            archive_seconds,
            offspring_seconds,
            benchmark_wall_seconds,
            best_distance_m: best,
            median_elite_distance_m: stats.median,
            qd_score: stats.qd_score,
            archive_cells: stats.archive_cells,
            archive_coverage: stats.archive_coverage,
            archive_unique_topologies: archive_morphology.unique_topologies,
            archive_topology_entropy_bits: archive_morphology.topology_entropy_bits,
            archive_triangle_like_count: archive_morphology.triangle_like_count,
            archive_triangle_like_fraction: archive_morphology.triangle_like_fraction,
            archive_node_counts_json: serde_json::to_string(&archive_morphology.node_counts)?,
            archive_muscle_counts_json: serde_json::to_string(&archive_morphology.muscle_counts)?,
            innovation_reserve_count: innovation_reserve_morphology.count,
            innovation_reserve_morphology_json: serde_json::to_string(
                &innovation_reserve_morphology,
            )?,
            population_unique_topologies: population_morphology.unique_topologies,
            population_topology_entropy_bits: population_morphology.topology_entropy_bits,
            population_triangle_like_count: population_morphology.triangle_like_count,
            population_triangle_like_fraction: population_morphology.triangle_like_fraction,
            population_node_counts_json: serde_json::to_string(&population_morphology.node_counts)?,
            population_muscle_counts_json: serde_json::to_string(
                &population_morphology.muscle_counts,
            )?,
            emitter_statistics_json: serde_json::to_string(&emitter_rows)?,
            topology_change_candidates_json: topology_change_json,
            archived_innovations_this_generation,
            innovation_survival_3gen_fraction: innovation_survival_fraction(
                innovations_surviving_three_generations,
                innovations_eligible_for_three_generations,
            ),
            new_all_time_record: is_record,
            evaluations_since_previous_record: evaluations_since_last_record,
        })?;
        curve.flush()?;

        final_population_morphology = population_morphology;
        final_archive_morphology = archive_morphology;
        final_innovation_reserve_morphology = innovation_reserve_morphology;
        final_median = stats.median;
        final_best = best;
        final_qd = stats.qd_score;
        final_coverage = stats.archive_coverage;
        final_emitters = emitter_snapshot(&measured_emitter_totals);
    }
    genealogy.flush()?;
    curve.flush()?;
    write_json(
        &seed_dir.join("innovations.json"),
        &innovation_records.values().collect::<Vec<_>>(),
    )?;
    write_json(
        &seed_dir.join("parent-topology-outcomes.json"),
        &parent_topology_outcomes.into_values().collect::<Vec<_>>(),
    )?;

    let evaluations = config.population as u64 * generations as u64;
    let summary = SeedSummary {
        seed,
        population: config.population,
        generations,
        evaluations,
        population_creation_seconds: creation_seconds,
        gpu_warmup_seconds: warmup_seconds,
        search_wall_seconds,
        benchmark_wall_seconds: benchmark_started.elapsed().as_secs_f64(),
        best_distance_m: final_best,
        median_elite_distance_m: final_median,
        qd_score: final_qd,
        archive_coverage: final_coverage,
        record_count,
        evaluations_since_last_record,
        time_to_milestones: reached,
        final_archive_morphology,
        final_innovation_reserve_morphology,
        final_population_morphology,
        structural_topology_candidates,
        archived_structural_innovations,
        reintroduced_structural_lineages,
        innovations_eligible_for_three_generations,
        innovations_surviving_three_generations,
        innovation_survival_fraction: innovation_survival_fraction(
            innovations_surviving_three_generations,
            innovations_eligible_for_three_generations,
        ),
        final_emitter_statistics: final_emitters,
    };
    write_json(&seed_dir.join("summary.json"), &summary)?;
    Ok(summary)
}

/// Load an existing checkpoint and report its measured history and current archive.
pub fn analyze_checkpoint(path: &Path) -> Result<(CheckpointAnalysis, Option<Creature>)> {
    let experiment = storage::load(path)?;
    let mut records = Vec::new();
    let mut previous = f32::NEG_INFINITY;
    let mut reached: BTreeMap<String, Option<Milestone>> =
        [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0]
            .into_iter()
            .map(|m| (milestone_key(m), None))
            .collect();
    let mut evaluation_seconds = 0.0;
    for stats in &experiment.history {
        evaluation_seconds += stats.seconds;
        let evaluations = (stats.generation as u64 + 1) * stats.population as u64;
        if stats.best > previous {
            records.push((stats.generation, evaluations, stats.best));
            previous = stats.best;
        }
        for (key, milestone) in reached.iter_mut() {
            if milestone.is_none() {
                let threshold = key.parse::<f32>().unwrap_or(f32::INFINITY);
                if stats.best >= threshold {
                    *milestone = Some(Milestone {
                        distance_m: threshold,
                        generation: stats.generation,
                        evaluations,
                        search_wall_seconds: evaluation_seconds,
                        benchmark_wall_seconds: 0.0,
                    });
                }
            }
        }
    }
    let mut elites: Vec<&Elite> = experiment.archive.entries.iter().collect();
    elites.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let champion = elites.first().copied();
    let current = experiment.history.last();
    let analysis = CheckpointAnalysis {
        seed: experiment.config.seed,
        generation: experiment.generation,
        population: experiment.config.population,
        duration_seconds: experiment.config.duration,
        archive_best_distance_m: experiment.archive.best_fitness(),
        archive_median_distance_m: current.map_or(0.0, |stats| stats.median),
        qd_score: experiment.archive.qd_score,
        archive_coverage: experiment.archive.coverage(),
        archive_morphology: archive_morphology(&experiment.archive.entries),
        innovation_reserve_morphology: innovation_reserve_morphology(&experiment.archive.entries),
        champion: champion.map_or(
            ChampionSummary {
                id: 0,
                distance_m: 0.0,
                nodes: 0,
                muscles: 0,
                topology: String::new(),
            },
            |elite| ChampionSummary {
                id: elite.creature.id,
                distance_m: elite.fitness,
                nodes: elite.creature.nodes.len(),
                muscles: elite.creature.muscles.len(),
                topology: topology_key(&elite.topology),
            },
        ),
        record_count: records.len(),
        records,
        time_to_milestones: reached,
    };
    Ok((analysis, champion.map(|elite| elite.creature.clone())))
}

#[allow(clippy::type_complexity)]
fn measure_candidate_morphologies(
    experiment: &Experiment,
    old_topology_by_id: &HashMap<u64, String>,
    old_elite_by_id: &HashMap<u64, (&Topology, f32)>,
) -> (
    Vec<String>,
    BTreeMap<String, u64>,
    Vec<(u64, String, bool, bool, f32, f32)>,
) {
    let mut topologies = Vec::with_capacity(experiment.config.population);
    let mut changed = BTreeMap::<String, u64>::new();
    let mut parent_outcomes = Vec::new();
    for index in 0..experiment.config.population {
        let topology = qd::topology_of_population(&experiment.population, index);
        let key = topology_key(&topology);
        topologies.push(key.clone());
        let parent_id = experiment
            .candidate_parent_ids
            .get(index)
            .copied()
            .flatten();
        if let Some(parent_id) = parent_id {
            if let Some(parent_topology) = old_topology_by_id.get(&parent_id)
                && parent_topology != &key
                && matches!(
                    experiment.candidate_emitters[index],
                    Emitter::Structural | Emitter::Novelty
                )
            {
                *changed
                    .entry(experiment.candidate_emitters[index].label().to_string())
                    .or_default() += 1;
            }
            if let Some((parent_topology, parent_fitness)) = old_elite_by_id.get(&parent_id) {
                let parent_key = topology_key(parent_topology);
                parent_outcomes.push((
                    experiment.population.genomes[index].id,
                    parent_key,
                    experiment.scores[index].is_finite() && experiment.scores[index] > FAILED,
                    experiment.scores[index].is_finite()
                        && experiment.scores[index] > FAILED
                        && experiment.scores[index] > *parent_fitness,
                    *parent_fitness,
                    experiment.scores[index],
                ));
            }
        }
    }
    (topologies, changed, parent_outcomes)
}

fn archive_morphology(entries: &[Elite]) -> MorphologySnapshot {
    let mut accumulator = MorphologyAccumulator::default();
    for elite in entries {
        if qd::is_morphology_niche(&elite.niche) {
            continue;
        }
        accumulator.add(
            elite.creature.nodes.len(),
            elite.creature.muscles.len(),
            &elite.topology,
        );
    }
    accumulator.finish()
}

fn innovation_reserve_morphology(entries: &[Elite]) -> MorphologySnapshot {
    let mut accumulator = MorphologyAccumulator::default();
    for elite in entries {
        if !qd::is_morphology_niche(&elite.niche) {
            continue;
        }
        accumulator.add(
            elite.creature.nodes.len(),
            elite.creature.muscles.len(),
            &elite.topology,
        );
    }
    accumulator.finish()
}

fn population_morphology(population: &Population) -> MorphologySnapshot {
    let mut accumulator = MorphologyAccumulator::default();
    for index in 0..population.genomes.len() {
        let genome = &population.genomes[index];
        accumulator.add(
            genome.node_count,
            genome.muscle_count,
            &qd::topology_of_population(population, index),
        );
    }
    accumulator.finish()
}

fn topology_key(topology: &Topology) -> String {
    let mut edges: Vec<(u32, u32)> = topology
        .edges
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges.sort_unstable();
    format!("{}:{edges:?}", topology.nodes)
}

fn topology_dimensions(key: &str) -> (usize, usize) {
    let (nodes, edges) = key.split_once(':').unwrap_or(("0", ""));
    let node_count = nodes.parse().unwrap_or(0);
    let edge_count = edges.matches('(').count();
    (node_count, edge_count)
}

fn triangle_like(topology: &Topology) -> bool {
    if topology.nodes != 3 || topology.edges.len() != 3 {
        return false;
    }
    let edges: BTreeSet<(u32, u32)> = topology
        .edges
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges == BTreeSet::from([(0, 1), (0, 2), (1, 2)])
}

fn shannon_entropy(counts: impl Iterator<Item = usize>) -> f64 {
    let counts: Vec<usize> = counts.collect();
    let total = counts.iter().sum::<usize>() as f64;
    if total == 0.0 {
        return 0.0;
    }
    -counts
        .into_iter()
        .map(|count| count as f64 / total)
        .filter(|p| *p > 0.0)
        .map(|p| p * p.log2())
        .sum::<f64>()
}

fn empty_morphology() -> MorphologySnapshot {
    MorphologyAccumulator::default().finish()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}

fn milestone_key(distance: f32) -> String {
    format!("{distance:.3}")
}

fn emitter_snapshot(
    stats: &[EmitterStats; qd::EMITTER_COUNT],
) -> BTreeMap<String, EmitterSnapshot> {
    let mut output = BTreeMap::new();
    for emitter in EMITTERS {
        let item = stats[emitter.index()];
        let insertions = item.discoveries + item.improvements;
        output.insert(
            emitter.label().to_string(),
            EmitterSnapshot {
                attempts: item.attempts,
                discoveries: item.discoveries,
                improvements: item.improvements,
                insertions,
                insertion_rate: ratio(insertions as usize, item.attempts as usize),
            },
        );
    }
    output
}

fn emitter_delta(
    current: &[EmitterStats; qd::EMITTER_COUNT],
    previous: &[EmitterStats; qd::EMITTER_COUNT],
) -> BTreeMap<String, EmitterSnapshot> {
    let mut output = BTreeMap::new();
    for emitter in EMITTERS {
        let now = current[emitter.index()];
        let before = previous[emitter.index()];
        let attempts = now.attempts.saturating_sub(before.attempts);
        let discoveries = now.discoveries.saturating_sub(before.discoveries);
        let improvements = now.improvements.saturating_sub(before.improvements);
        let insertions = discoveries + improvements;
        output.insert(
            emitter.label().to_string(),
            EmitterSnapshot {
                attempts,
                discoveries,
                improvements,
                insertions,
                insertion_rate: insertions as f64 / attempts.max(1) as f64,
            },
        );
    }
    output
}

fn accumulate_emitter_counts(
    totals: &mut [EmitterStats; qd::EMITTER_COUNT],
    current: &[EmitterStats; qd::EMITTER_COUNT],
    previous: &[EmitterStats; qd::EMITTER_COUNT],
) {
    for index in 0..qd::EMITTER_COUNT {
        totals[index].attempts += current[index]
            .attempts
            .saturating_sub(previous[index].attempts);
        totals[index].discoveries += current[index]
            .discoveries
            .saturating_sub(previous[index].discoveries);
        totals[index].improvements += current[index]
            .improvements
            .saturating_sub(previous[index].improvements);
    }
}

fn innovation_survival_fraction(survived: u64, eligible: u64) -> f64 {
    survived as f64 / eligible.max(1) as f64
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path).with_context(|| format!("Creating {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), value)
        .with_context(|| format!("Writing {}", path.display()))?;
    Ok(())
}

fn aggregate_summaries(summaries: &[SeedSummary]) -> serde_json::Value {
    let mut best: Vec<f32> = summaries.iter().map(|item| item.best_distance_m).collect();
    best.sort_by(f32::total_cmp);
    let mean = |f: fn(&SeedSummary) -> f64| {
        summaries.iter().map(f).sum::<f64>() / summaries.len().max(1) as f64
    };
    serde_json::json!({
        "seed_count": summaries.len(),
        "seeds": summaries,
        "mean_best_distance_m": mean(|item| item.best_distance_m as f64),
        "median_best_distance_m": best.get(best.len() / 2).copied().unwrap_or(0.0),
        "mean_median_elite_distance_m": mean(|item| item.median_elite_distance_m as f64),
        "mean_qd_score": mean(|item| item.qd_score),
        "mean_archive_coverage": mean(|item| item.archive_coverage as f64),
        "mean_evaluations_since_last_record": mean(|item| item.evaluations_since_last_record as f64),
        "mean_innovation_survival_fraction": mean(|item| item.innovation_survival_fraction),
    })
}

pub fn write_checkpoint_analysis(
    checkpoint: &Path,
    output: Option<&Path>,
    champion_output: Option<&Path>,
) -> Result<()> {
    let (analysis, champion) = analyze_checkpoint(checkpoint)?;
    if let Some(path) = champion_output
        && let Some(creature) = champion
    {
        write_json(path, &creature)?;
    }
    if let Some(path) = output {
        write_json(path, &analysis)
    } else {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &analysis)?;
        println!();
        Ok(())
    }
}
