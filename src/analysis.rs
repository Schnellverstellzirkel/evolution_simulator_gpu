//! Search-quality summaries of a saved game: records, milestones and the
//! morphology of its archives.

use crate::{
    evolution::Creature,
    qd::{self, Elite, Topology},
    storage::{self, Experiment},
};
use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::BufWriter,
    path::Path,
};

#[derive(Clone, Debug, Serialize)]
pub struct MorphologySnapshot {
    pub count: usize,
    /// Distinct skeleton and actuator graph topologies; edge order is ignored.
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
        innovation_reserve_morphology: innovation_reserve_morphology(&island_entries(&experiment)),
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

/// Every island's elites: the morphology reserves live in the islands.
fn island_entries(experiment: &Experiment) -> Vec<Elite> {
    experiment
        .islands
        .iter()
        .take(crate::storage::island_count())
        .flat_map(|island| island.entries.iter().cloned())
        .collect()
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

fn topology_key(topology: &Topology) -> String {
    let mut edges: Vec<(u32, u32)> = topology
        .edges
        .iter()
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    edges.sort_unstable();
    format!("{}:{edges:?}", topology.nodes)
}

fn triangle_like(topology: &Topology) -> bool {
    if topology.nodes != 3 || topology.edges.len() != 3 {
        return false;
    }
    let skeleton: BTreeSet<(u32, u32)> = topology
        .edges
        .iter()
        .filter(|(a, b)| *a < 3 && *b < 3)
        .map(|&(a, b)| (a.min(b), a.max(b)))
        .collect();
    let actuators = topology
        .edges
        .iter()
        .filter(|(a, b)| *a >= 3 && *b >= 3)
        .count();
    skeleton.len() == 2 && actuators == 1
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

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}

fn milestone_key(distance: f32) -> String {
    format!("{distance:.3}")
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
