//! Deterministic, CPU-only A/B harness for the evolutionary search.
//!
//! Runs the production generational loop (evaluate every creature with
//! `cpu_engine::evaluate`, then `archive_batch` and `prepare_next_batch`) on
//! fixed seeds and prints comparable metrics, so a search change guarded by an
//! environment flag can be measured with the same command before and after.
//!
//! Usage:
//!   cargo run --release --example search_ab -- [generations] [population] [duration] [seed,seed,...] [--tag NAME] [--checks]
//!   cargo run --release --example search_ab -- <tag> [generations] [population] [duration] [seed,seed,...]
//! Defaults: 2 generations, 64 creatures, 1.0 s trials, seeds 38,39.
//! Wall time goes to stderr so stdout is deterministic and diffable.
//!
//! `--checks` adds the game's contender check: creatures that could enter an
//! archive run a second trial from a perturbed pose at
//! `scheduler::check_config` physics on the CPU, one per archive cell at a
//! time as in the scheduler, and `scheduler::check_verdict` decides their
//! score. Without it no creature is checked, as in the earlier search
//! measurements.
use anyhow::{Context, Result};
use evolution_simulator::{
    config::Config, cpu_engine, engine, evolution::Population, physics, qd::EvaluationMetrics,
    scheduler, storage::Experiment,
};
use std::{
    collections::{BTreeMap, HashMap},
    time::Instant,
};

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
    checks: bool,
    gpu: bool,
    seed_offset: u64,
    save: Option<String>,
}

fn usage() -> &'static str {
    "usage: search_ab [tag] [generations] [population] [duration_seconds] [seed,seed,...] [--tag NAME] [--checks] [--gpu] [--seed-offset N]"
}

fn options() -> Result<Options> {
    let mut positionals: Vec<String> = Vec::new();
    let mut tag = None;
    let mut checks = false;
    let mut gpu = false;
    let mut seed_offset = 0u64;
    let mut save = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--checks" {
            checks = true;
        } else if arg == "--gpu" {
            gpu = true;
        } else if arg == "--seed-offset" {
            seed_offset = args
                .next()
                .context("--seed-offset needs a number")?
                .parse()
                .context("seed offset")?;
        } else if arg == "--save" {
            save = Some(args.next().context("--save needs a path")?);
        } else if arg == "--tag" {
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
        checks,
        gpu,
        seed_offset,
        save,
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
    println!("{scope} seed generation best_m qd_score cells mean_nodes mean_muscles");
    let started = Instant::now();
    let (mut distances, mut scores) = (Vec::new(), Vec::new());
    let mut checks = CheckCounts::default();
    for &seed in &options.seeds {
        let seed_started = Instant::now();
        let (best, qd) = run_seed(seed, &options, scope, &mut checks)?;
        eprintln!(
            "search_ab: seed {seed} in {:.1} s wall, {:.1} s CPU so far",
            seed_started.elapsed().as_secs_f64(),
            cpu_seconds()
        );
        distances.push(best);
        scores.push(qd as f32);
    }
    paired_summary(&mut distances, &mut scores);
    if options.checks {
        println!(
            "{scope} checks: {} checked, {} lowered by more than 1 cm, {} kept out by the screen, {} dropped (their cell's best verdict beat them)",
            checks.checked, checks.lowered, checks.kept_out, checks.unchecked
        );
    }
    eprintln!(
        "search_ab: {} seeds in {:.1} s wall, {:.1} s CPU, checks {:.1} s wall",
        options.seeds.len(),
        started.elapsed().as_secs_f64(),
        cpu_seconds(),
        checks.seconds
    );
    Ok(())
}

/// Processor time of this process (user and system), which other work on a
/// shared machine disturbs less than the wall clock.
fn cpu_seconds() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage fills the struct it is given.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

#[derive(Default)]
struct CheckCounts {
    checked: u64,
    lowered: u64,
    kept_out: u64,
    unchecked: u64,
    seconds: f64,
}

/// The game's contender check on one evaluated generation. Creatures that
/// could enter an archive (against the start-of-generation archives) wait
/// by archive cell, fastest first. Each round checks the fastest waiter of
/// every cell, plus every contender that shares no cell, with a perturbed
/// trial and `scheduler::check_verdict`. As in the scheduler, a cell's
/// remaining waiters are then decided again: those whose standard score no
/// longer beats the best verdict of that cell enter no archive this
/// generation, and the rest wait for the next round.
fn check_contenders(
    experiment: &Experiment,
    metrics: &mut [EvaluationMetrics],
    counts: &mut CheckCounts,
) {
    let started = Instant::now();
    let mut round: Vec<usize> = Vec::new();
    let mut waiting: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, metric) in metrics.iter().enumerate() {
        match experiment.check_need(i, metric) {
            scheduler::CheckNeed::Release => {}
            scheduler::CheckNeed::Check { cell: None, .. } => round.push(i),
            scheduler::CheckNeed::Check {
                cell: Some(cell), ..
            } => waiting.entry(cell).or_default().push(i),
        }
    }
    // Fastest last, so each round pops a cell's fastest waiter.
    for list in waiting.values_mut() {
        list.sort_by(|&a, &b| {
            metrics[a]
                .fitness
                .total_cmp(&metrics[b].fitness)
                .then(b.cmp(&a))
        });
    }
    let mut cells: Vec<u64> = waiting.keys().copied().collect();
    cells.sort_unstable();
    let mut bar: HashMap<u64, f32> = HashMap::new();
    let check_cfg = scheduler::check_config(&experiment.config);
    loop {
        let mut cell_of: HashMap<usize, u64> = HashMap::new();
        for &cell in &cells {
            let list = waiting.get_mut(&cell).expect("waiting cell");
            if let Some(i) = list.pop() {
                cell_of.insert(i, cell);
                round.push(i);
            }
        }
        if round.is_empty() {
            break;
        }
        round.sort_unstable();
        let mut unit = Population::default();
        for &i in &round {
            let mut creature = experiment.population.creature(i);
            scheduler::perturb(&mut creature);
            unit.push(creature);
        }
        let results = cpu_engine::evaluate(&unit, &check_cfg);
        for (&i, result) in round.iter().zip(&results) {
            let before = metrics[i].fitness;
            let (fitness, unchecked) = scheduler::check_verdict(&metrics[i], result);
            metrics[i].fitness = fitness;
            metrics[i].unchecked |= unchecked;
            counts.checked += 1;
            counts.lowered += u64::from(fitness < before - 0.01);
            counts.kept_out += u64::from(unchecked);
            if let Some(&cell) = cell_of.get(&i)
                && !unchecked
            {
                let best = bar.entry(cell).or_insert(f32::NEG_INFINITY);
                *best = best.max(fitness);
            }
        }
        round.clear();
        for &cell in &cells {
            let Some(&best) = bar.get(&cell) else {
                continue;
            };
            let list = waiting.get_mut(&cell).expect("waiting cell");
            while list.first().is_some_and(|&i| metrics[i].fitness <= best) {
                let i = list.remove(0);
                metrics[i].unchecked = true;
                counts.unchecked += 1;
            }
        }
    }
    counts.seconds += started.elapsed().as_secs_f64();
}

/// Runs the game's loop for one seed and returns its archive best and QD score.
fn run_seed(
    seed: u64,
    options: &Options,
    scope: &str,
    counts: &mut CheckCounts,
) -> Result<(f32, f64)> {
    let cfg = Config {
        population: options.population,
        duration: options.duration,
        random_seed: false,
        seed: seed + options.seed_offset,
        ..Config::default()
    };
    cfg.validate()
        .with_context(|| format!("seed {seed} configuration"))?;
    let mut gpu = if options.gpu {
        Some(evolution_simulator::gpu::Gpu::new("RTX 4060")?)
    } else {
        None
    };
    let mut experiment = Experiment::new(cfg).with_context(|| format!("seed {seed} experiment"))?;
    let mut best = f32::NAN;
    let mut top = Vec::new();
    for generation in 0..options.generations {
        if let Some(gpu) = gpu.as_mut() {
            // The game's generational path: the scheduler runs the standard
            // trials with the early screen and the contender checks, and the
            // GPU score is final.
            let sched = gpu.sched.as_mut().expect("scheduler");
            let population = experiment.config.population;
            let mut done = vec![false; population];
            sched.begin(&experiment.population, 0..population);
            let mut stored = 0;
            while stored < population {
                sched.pump(&experiment.population, &experiment.config, &done, |i, m| {
                    experiment.check_need(i, m)
                })?;
                for (indices, metrics) in sched.collect(
                    &experiment.population,
                    &experiment.config,
                    std::time::Duration::from_millis(4),
                    |i, m| experiment.contender(i, m),
                )? {
                    for (&i, m) in indices.iter().zip(&metrics) {
                        experiment.record_result(i, m);
                        done[i] = true;
                        stored += 1;
                    }
                }
            }
        } else {
            let results = cpu_engine::evaluate(&experiment.population, &experiment.config);
            let mut metrics: Vec<EvaluationMetrics> = results
                .iter()
                .enumerate()
                .map(|(index, result)| {
                    scheduler::to_metrics(&experiment.population, index, result, &experiment.config)
                })
                .collect();
            if options.checks {
                check_contenders(&experiment, &mut metrics, counts);
            }
            for (index, metric) in metrics.iter().enumerate() {
                experiment.record_result(index, metric);
            }
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
        // Mean body size of the evaluated generation: bodies that only grow
        // make every later generation slower to simulate.
        let genomes = &experiment.population.genomes;
        let mean = |part: fn(&evolution_simulator::evolution::Genome) -> usize| {
            genomes.iter().map(part).sum::<usize>() as f64 / genomes.len().max(1) as f64
        };
        println!(
            "{scope} {seed} {generation} {generation_best:.2} {:.2} {} {:.2} {:.2}",
            experiment.archive.qd_score,
            experiment.archive.behavior_count(),
            mean(|g| g.node_count),
            mean(|g| g.muscle_count)
        );
        best = experiment
            .archive
            .entries
            .iter()
            .map(|elite| elite.fitness)
            .fold(best, f32::max);
        println!("{scope} seed {seed} generation {generation} archive best {best:.2} m");
        top = top_bodies(&experiment, TOP_BODIES);
        experiment
            .prepare_next_batch()
            .with_context(|| format!("seed {seed} generation {generation} breeding"))?;
    }
    // `EVOLUTION_AB_SAVE=<dir>` writes each seed's final experiment as
    // `<dir>/seed-<seed>.evo`, for `physics_audit` and `size_report`.
    if let Some(dir) = std::env::var_os("EVOLUTION_AB_SAVE") {
        let path = std::path::Path::new(&dir).join(format!("seed-{seed}.evo"));
        evolution_simulator::storage::save(&path, &experiment)?;
    }
    let qd = experiment.archive.qd_score;
    let cells = experiment.archive.behavior_count();
    if best.is_finite() {
        println!("{scope} seed {seed} summary: best {best:.2} m, qd {qd:.2}, cells {cells}");
    } else {
        println!("{scope} seed {seed} summary: archive empty, qd {qd:.2}, cells {cells}");
    }
    if let Some(path) = &options.save {
        evolution_simulator::storage::save(std::path::Path::new(path), &experiment)?;
    }
    print_body_mix(scope, seed, &top);
    print_robustness(scope, seed, &experiment);
    print_common_grid(scope, seed, &experiment);
    let weights = evolution_simulator::qd::emitter_weights(&experiment.emitter_stats);
    println!(
        "{scope} seed {seed} emitter shares: {}",
        evolution_simulator::qd::Emitter::ALL
            .iter()
            .map(|e| format!("{} {:.2}", e.label(), weights[e.index()]))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok((best, qd))
}

/// How much of their archive distance the 50 best global elites keep under
/// the game's fine check (4x rate and solver passes, full trial, no screen)
/// from a perturbed pose that no run's own check used, whatever check the
/// run itself used.
fn print_robustness(scope: &str, seed: u64, experiment: &Experiment) {
    let mut elites: Vec<_> = experiment
        .archive
        .entries
        .iter()
        .filter(|elite| elite.fitness.is_finite() && elite.fitness > 0.0)
        .collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(TOP_BODIES);
    if elites.is_empty() {
        return;
    }
    let mut unit = Population::default();
    for elite in &elites {
        let mut creature = elite.creature.clone();
        // A second pose from the same rule: the id seeds the perturbation.
        creature.id ^= 0x9e37_79b9;
        scheduler::perturb(&mut creature);
        creature.id ^= 0x9e37_79b9;
        unit.push(creature);
    }
    let cfg = Config {
        fidelity: Some(physics::Fidelity::fine()),
        screen: None,
        population: elites.len(),
        ..experiment.config.clone()
    };
    let results = cpu_engine::evaluate(&unit, &cfg);
    let mut kept: Vec<f32> = elites
        .iter()
        .zip(&results)
        .map(|(elite, result)| result.fitness.max(0.0) / elite.fitness)
        .collect();
    let halved = kept.iter().filter(|&&k| k < 0.5).count();
    println!(
        "{scope} seed {seed} top-{TOP_BODIES} elites under the fine check: median share kept {:.2}, below half {halved} of {}",
        median(&mut kept),
        elites.len()
    );
}

/// QD score of the global archive's behavior elites re-binned on one fixed
/// grid (the archive shape before any experiment: contact 6, cadence 8,
/// height 6, feet 5), so runs whose archives have different shapes compare on
/// the same ground. Also the reserve size and the distinct body plans held.
fn print_common_grid(scope: &str, seed: u64, experiment: &Experiment) {
    let mut cells: HashMap<[u8; 4], f32> = HashMap::new();
    let mut plans = std::collections::HashSet::new();
    for elite in &experiment.archive.entries {
        plans.insert(&elite.topology);
        if evolution_simulator::qd::is_morphology_niche(&elite.niche) || elite.fitness <= 0.0 {
            continue;
        }
        let d = &elite.descriptor;
        let bin = |v: f32, high: f32, n: f32| {
            ((v.clamp(0.0, high) / high * n).floor().min(n - 1.0)) as u8
        };
        let low = 0.15f32;
        let top = (0.6 * evolution_simulator::evolution::max_bone_length()).max(2.0 * low);
        let height = ((d.mean_height.max(low) / low).ln() / (top / low).ln()).clamp(0.0, 1.0);
        let key = [
            bin(d.ground_contact, 1.0, 6.0),
            bin(d.gait_frequency, 6.0, 8.0),
            bin(height, 1.0, 6.0),
            (d.feet.round() as i32).clamp(1, 5) as u8,
        ];
        let slot = cells.entry(key).or_insert(f32::MIN);
        *slot = slot.max(elite.fitness);
    }
    let qd: f64 = cells.values().map(|&f| f as f64).sum();
    println!(
        "{scope} seed {seed} common grid: qd {qd:.2}, cells {}, reserve {}, body plans {}",
        cells.len(),
        experiment.archive.morphology_count(),
        plans.len()
    );
}

struct BodySize {
    nodes: usize,
    muscles: usize,
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
                muscles: creature.muscles.len(),
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
    let muscles: Vec<usize> = bodies.iter().map(|body| body.muscles).collect();
    println!(
        "{scope} seed {seed} top-{TOP_BODIES} muscles: mean {:.2}, most {}",
        muscles.iter().sum::<usize>() as f32 / muscles.len() as f32,
        muscles.iter().max().unwrap_or(&0)
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
