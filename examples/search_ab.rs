//! Deterministic, CPU-only A/B harness for the evolutionary search.
//!
//! Runs the production generational loop (evaluate every creature with
//! `cpu_engine::evaluate`, then `archive_batch` and `prepare_next_batch`) on
//! fixed seeds and prints comparable metrics, so a search change guarded by an
//! environment flag can be measured with the same command before and after.
//!
//! Usage:
//!   cargo run --release --example search_ab -- [generations] [population] [duration] [seed,seed,...] [--tag NAME]
//!   cargo run --release --example search_ab -- <tag> [generations] [population] [duration] [seed,seed,...]
//! Defaults: 2 generations, 64 creatures, 1.0 s trials, seeds 38,39.
//! Wall time goes to stderr so stdout is deterministic and diffable.
use anyhow::{Context, Result};
use evolution_simulator::{
    config::Config, cpu_engine, engine, physics, scheduler, storage::Experiment,
};
use std::{collections::BTreeMap, time::Instant};

const DEFAULT_GENERATIONS: u32 = 2;
const DEFAULT_POPULATION: usize = 64;
const DEFAULT_DURATION: f32 = 1.0;
const DEFAULT_SEEDS: &[u64] = &[38, 39];
const TOP_BODIES: usize = 50;

struct Options {
    generations: u32,
    population: usize,
    duration: f32,
    seeds: Vec<u64>,
    tag: Option<String>,
}

fn usage() -> &'static str {
    "usage: search_ab [tag] [generations] [population] [duration_seconds] [seed,seed,...] [--tag NAME]"
}

fn options() -> Result<Options> {
    let mut positionals: Vec<String> = Vec::new();
    let mut tag = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--tag" {
            tag = Some(args.next().context("--tag needs a name")?);
        } else if arg == "--help" || arg == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        } else if arg.starts_with("--") {
            anyhow::bail!("unknown option {arg}\n{}", usage());
        } else {
            positionals.push(arg);
        }
    }
    if tag.is_none()
        && positionals
            .first()
            .is_some_and(|arg| arg.parse::<u32>().is_err())
    {
        tag = Some(positionals.remove(0));
    }
    anyhow::ensure!(positionals.len() <= 4, "too many arguments\n{}", usage());
    let generations = positionals
        .first()
        .map(|arg| arg.parse().context("generations"))
        .transpose()?
        .unwrap_or(DEFAULT_GENERATIONS);
    let population = positionals
        .get(1)
        .map(|arg| arg.parse().context("population"))
        .transpose()?
        .unwrap_or(DEFAULT_POPULATION);
    let duration = positionals
        .get(2)
        .map(|arg| arg.parse().context("duration"))
        .transpose()?
        .unwrap_or(DEFAULT_DURATION);
    let seeds = match positionals.get(3) {
        Some(list) => list
            .split(',')
            .map(|seed| seed.trim().parse().context("seed"))
            .collect::<Result<Vec<u64>>>()?,
        None => DEFAULT_SEEDS.to_vec(),
    };
    anyhow::ensure!(!seeds.is_empty(), "need at least one seed");
    Ok(Options {
        generations,
        population,
        duration,
        seeds,
        tag,
    })
}

fn main() -> Result<()> {
    // Low priority and the shared half-machine budget, like every other run.
    engine::lower_thread_priority();
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(engine::rayon_threads())
        .start_handler(|_| engine::lower_thread_priority())
        .build_global();
    let options = options()?;
    let scope = options.tag.as_deref().unwrap_or("untagged");
    println!(
        "search_ab {scope}: {} generations, population {}, {:.2} s trials, seeds {}",
        options.generations,
        options.population,
        options.duration,
        options
            .seeds
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("{scope} seed generation best_m qd_score cells");
    let started = Instant::now();
    let (mut distances, mut scores) = (Vec::new(), Vec::new());
    for &seed in &options.seeds {
        let (best, qd) = run_seed(seed, &options, scope)?;
        distances.push(best);
        scores.push(qd as f32);
    }
    paired_summary(&mut distances, &mut scores);
    eprintln!(
        "search_ab: {} seeds in {:.1} s wall",
        options.seeds.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Runs the game's loop for one seed and returns its archive best and QD score.
fn run_seed(seed: u64, options: &Options, scope: &str) -> Result<(f32, f64)> {
    let cfg = Config {
        population: options.population,
        duration: options.duration,
        random_seed: false,
        seed,
        ..Config::default()
    };
    cfg.validate()
        .with_context(|| format!("seed {seed} configuration"))?;
    let mut experiment = Experiment::new(cfg).with_context(|| format!("seed {seed} experiment"))?;
    let mut best = f32::NAN;
    let mut top = Vec::new();
    for generation in 0..options.generations {
        let results = cpu_engine::evaluate(&experiment.population, &experiment.config);
        for (index, result) in results.iter().enumerate() {
            let metrics =
                scheduler::to_metrics(&experiment.population, index, result, &experiment.config);
            experiment.scores[index] = metrics.fitness;
            experiment.trial_metrics[index] = metrics.behavior;
        }
        experiment.evaluated = experiment.config.population;
        experiment
            .archive_batch()
            .with_context(|| format!("seed {seed} generation {generation} archive"))?;
        let generation_best = experiment
            .scores
            .iter()
            .copied()
            .filter(|score| score.is_finite())
            .fold(f32::MIN, f32::max);
        println!(
            "{scope} {seed} {generation} {generation_best:.2} {:.2} {}",
            experiment.archive.qd_score,
            experiment.archive.behavior_count()
        );
        best = experiment
            .archive
            .entries
            .iter()
            .map(|elite| elite.fitness)
            .fold(best, f32::max);
        top = top_bodies(&experiment, TOP_BODIES);
        experiment
            .prepare_next_batch()
            .with_context(|| format!("seed {seed} generation {generation} breeding"))?;
    }
    let qd = experiment.archive.qd_score;
    let cells = experiment.archive.behavior_count();
    if best.is_finite() {
        println!("{scope} seed {seed} summary: best {best:.2} m, qd {qd:.2}, cells {cells}");
    } else {
        println!("{scope} seed {seed} summary: archive empty, qd {qd:.2}, cells {cells}");
    }
    print_body_mix(scope, seed, &top);
    Ok((best, qd))
}

struct BodySize {
    nodes: usize,
    length: f32,
    longest_bone: f32,
    mass: f32,
}

/// The `count` best-scoring creatures of the current evaluated generation.
fn top_bodies(experiment: &Experiment, count: usize) -> Vec<BodySize> {
    let mut order: Vec<usize> = (0..experiment.scores.len())
        .filter(|&index| experiment.scores[index].is_finite())
        .collect();
    order.sort_by(|&a, &b| experiment.scores[b].total_cmp(&experiment.scores[a]));
    order.truncate(count);
    order
        .into_iter()
        .map(|index| {
            let creature = experiment.population.creature(index);
            let mass: f32 = physics::body(&creature.nodes, &creature.bones)
                .iter()
                .map(|node| node.mass)
                .sum();
            let length: f32 = creature.bones.iter().map(|bone| bone.rest_length).sum();
            let longest_bone = creature
                .bones
                .iter()
                .map(|bone| bone.rest_length)
                .fold(0.0, f32::max);
            BodySize {
                nodes: creature.nodes.len(),
                length,
                longest_bone,
                mass,
            }
        })
        .collect()
}

fn print_body_mix(scope: &str, seed: u64, bodies: &[BodySize]) {
    if bodies.is_empty() {
        println!("{scope} seed {seed} top-{TOP_BODIES}: no scored creatures");
        return;
    }
    let mut nodes: BTreeMap<usize, usize> = BTreeMap::new();
    for body in bodies {
        *nodes.entry(body.nodes).or_default() += 1;
    }
    let mix = nodes
        .iter()
        .map(|(count, bodies)| format!("{count}x{bodies}"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut lengths: Vec<f32> = bodies.iter().map(|body| body.length).collect();
    let mut masses: Vec<f32> = bodies.iter().map(|body| body.mass).collect();
    let longest = bodies
        .iter()
        .map(|body| body.longest_bone)
        .fold(0.0, f32::max);
    println!("{scope} seed {seed} top-{TOP_BODIES} node mix: {mix}");
    println!(
        "{scope} seed {seed} top-{TOP_BODIES}: median length {:.2} m, median mass {:.2} kg, longest bone {:.2} m",
        median(&mut lengths),
        median(&mut masses),
        longest
    );
}

fn paired_summary(distances: &mut Vec<f32>, scores: &mut Vec<f32>) {
    let seeds = distances.len();
    distances.retain(|distance| distance.is_finite());
    scores.retain(|score| score.is_finite());
    if distances.is_empty() || scores.is_empty() {
        println!("paired across {seeds} seeds: no archive entries");
        return;
    }
    let distance_mean = mean(distances);
    let distance_median = median(distances);
    let score_mean = mean(scores);
    let score_median = median(scores);
    println!(
        "paired across {seeds} seeds: best distance mean {distance_mean:.2} m, median {distance_median:.2} m; qd score mean {score_mean:.2}, median {score_median:.2}"
    );
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

/// Upper middle entry of the sorted values, like `size_report`.
fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}
