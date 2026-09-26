//! CPU cost of each environment effect and level on one fixed population.
//!
//! Builds a deterministic random population once, then times
//! `cpu_engine::evaluate` on the calm world and on every level of every
//! `environment::EFFECTS` entry, so a new effect is covered without touching
//! this file. The rate is CPU evaluation only. The GPU kernel is unchanged
//! because the effect values travel in its existing uniform buffer.
//!
//! Usage:
//!   cargo run --release --example effect_cost -- [count] [seconds] [seed]
//! Defaults: 2048 bodies, 2.0 s trials, seed 38. Every pass samples the calm
//! world at its start, middle, and end, reports each configuration's ratio to
//! the calm mean, then takes the median ratio across passes, so slow drift in
//! CPU speed cancels. The creatures/s column is the best rate seen, the least
//! noisy absolute number on a machine that other work shares.
use anyhow::{Context, Result};
use evolution_simulator::{config::Config, cpu_engine, engine, environment::EFFECTS, evolution};
use std::time::Instant;

const DEFAULT_COUNT: usize = 2048;
const DEFAULT_SECONDS: f32 = 2.0;
const DEFAULT_SEED: u64 = 38;
const PASSES: usize = 8;

#[derive(Clone)]
struct Variant {
    effect: &'static str,
    level: usize,
    world: &'static str,
    cfg: Config,
}

fn main() -> Result<()> {
    // Low priority and an evaluation pool sized like the scheduler's, so the
    // numbers match a real six-thread CPU evaluation on this workstation.
    engine::lower_thread_priority();
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(engine::cpu_threads().max(1))
        .start_handler(|_| engine::lower_thread_priority())
        .build_global();

    let count: usize = positional(1).unwrap_or(DEFAULT_COUNT);
    let seconds: f32 = positional(2).unwrap_or(DEFAULT_SECONDS);
    let seed: u64 = positional(3).unwrap_or(DEFAULT_SEED);
    let base = Config {
        population: count,
        duration: seconds,
        random_seed: false,
        seed,
        ..Config::default()
    };
    let population = evolution::create(&base).context("random population")?;
    // One untimed run so caches and frequency are warm before the first pass.
    let _ = cpu_engine::evaluate(&population, &base);

    let variants = variants(&base);
    let mut rates: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    let mut best_m = vec![f32::NEG_INFINITY; variants.len()];
    for _ in 0..PASSES {
        // Sample the calm world three times through the pass; a contention
        // burst that hits one sample does not distort the whole baseline.
        let mut calm_rates = vec![measure_once(&population, &base).rate];
        let mut pass_rates = Vec::with_capacity(variants.len());
        for (index, variant) in variants.iter().enumerate() {
            if index == variants.len() / 2 {
                calm_rates.push(measure_once(&population, &base).rate);
            }
            let run = measure_once(&population, &variant.cfg);
            pass_rates.push(run.rate);
            best_m[index] = best_m[index].max(run.best_m);
        }
        calm_rates.push(measure_once(&population, &base).rate);
        let calm = calm_rates.iter().sum::<f64>() / calm_rates.len() as f64;
        for (index, rate) in pass_rates.into_iter().enumerate() {
            rates[index].push(rate);
            ratios[index].push(rate / calm);
        }
    }

    println!(
        "effect_cost: {} bodies, {:.2} s trials, seed {seed}, {} CPU threads, standard fidelity, {PASSES} passes",
        population.genomes.len(),
        seconds,
        engine::cpu_threads().max(1)
    );
    println!(
        "{:<11} {:>5}  {:<20} {:>12} {:>7} {:>8}",
        "effect", "level", "world", "creatures/s", "% calm", "best m"
    );
    for (index, variant) in variants.iter().enumerate() {
        print_row(
            variant,
            rates[index].iter().copied().fold(0.0, f64::max),
            100.0 * median(&mut ratios[index]),
            best_m[index],
        );
    }
    Ok(())
}

/// The calm world first, then every level of every effect.
fn variants(base: &Config) -> Vec<Variant> {
    let mut variants = vec![Variant {
        effect: "calm",
        level: 0,
        world: "default world",
        cfg: base.clone(),
    }];
    for effect in &EFFECTS {
        for level in 0..effect.levels.len() {
            let mut cfg = base.clone();
            effect.set_level(&mut cfg, level);
            variants.push(Variant {
                effect: effect.name,
                level,
                world: effect.levels[level],
                cfg,
            });
        }
    }
    variants
}

fn positional<T: std::str::FromStr>(index: usize) -> Option<T> {
    std::env::args()
        .nth(index)
        .and_then(|value| value.parse().ok())
}

fn measure_once(population: &evolution::Population, cfg: &Config) -> Measurement {
    let started = Instant::now();
    let results = cpu_engine::evaluate(population, cfg);
    let elapsed = started.elapsed().as_secs_f64();
    Measurement {
        rate: population.genomes.len() as f64 / elapsed,
        best_m: results
            .iter()
            .map(|result| result.fitness)
            .filter(|fitness| fitness.is_finite())
            .fold(f32::NEG_INFINITY, f32::max),
    }
}

struct Measurement {
    rate: f64,
    best_m: f32,
}

/// Middle entry of the sorted values, so one slow pass cannot move the ratio.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn print_row(variant: &Variant, rate: f64, percent_calm: f64, best_m: f32) {
    let best = if best_m.is_finite() {
        format!("{best_m:.2}")
    } else {
        "n/a".to_string()
    };
    println!(
        "{:<11} {:>5}  {:<20} {rate:>12.1} {percent_calm:>7.1} {best:>8}",
        variant.effect, variant.level, variant.world
    );
}
