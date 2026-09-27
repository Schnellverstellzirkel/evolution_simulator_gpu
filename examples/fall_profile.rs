//! Share of simulated trial time that comes after a creature has fallen.
//!
//! A fallen creature's score is fixed at its fall, so every step after it is
//! work that cannot change the fitness. This reports how much of a population's
//! trial time that is, by body size, on the CPU engine.
//!
//! Usage: cargo run --release --example fall_profile -- <checkpoint> [count]
use anyhow::{Context, Result};
use evolution_simulator::{cpu_engine, evolution::Population, physics, storage};
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .context("usage: fall_profile <checkpoint> [count]")?;
    let e = storage::load(std::path::Path::new(&path))?;
    let count: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(e.config.population)
        .min(e.config.population);
    let mut population = Population::default();
    for i in 0..count {
        population.push(e.population.creature(i));
    }
    let cfg = e.config.clone();
    let results = cpu_engine::evaluate(&population, &cfg);
    let settle = physics::settle() as f64 * physics::dt() as f64;
    let trial = cfg.duration as f64;
    let total = settle + trial;
    let mut after_fall = 0.0f64;
    let mut fallen = 0usize;
    let mut by_nodes: BTreeMap<usize, (usize, usize, f64)> = BTreeMap::new();
    let mut times: Vec<f64> = Vec::new();
    for (index, result) in results.iter().enumerate() {
        let nodes = population.genomes[index].node_count;
        let entry = by_nodes.entry(nodes).or_default();
        entry.0 += 1;
        if result.fall_time > 0.0 {
            let lost = trial - f64::from(result.fall_time);
            after_fall += lost;
            fallen += 1;
            entry.1 += 1;
            entry.2 += lost;
            times.push(f64::from(result.fall_time));
        }
    }
    times.sort_by(f64::total_cmp);
    let pick = |q: f64| {
        times
            .get(((times.len() as f64 - 1.0) * q) as usize)
            .copied()
    };
    println!(
        "{count} creatures, {:.1} s trials plus {settle:.2} s settling",
        trial
    );
    println!(
        "fallen {fallen} ({:.1}%), fall time p10 {:?} s, median {:?} s, p90 {:?} s",
        100.0 * fallen as f64 / count as f64,
        pick(0.1),
        pick(0.5),
        pick(0.9)
    );
    println!(
        "simulated time after a fall: {:.1}% of all steps",
        100.0 * after_fall / (count as f64 * total)
    );
    // Share of all steps a GPU saves by dropping fallen creatures only at
    // segment boundaries (seconds after settling).
    let boundary_sets: [&[f64]; 5] = [
        &[2.0],
        &[2.0, 10.0],
        &[1.0, 3.0, 10.0],
        &[1.0, 3.0, 10.0, 30.0],
        &[0.5, 1.5, 4.0, 12.0, 30.0],
    ];
    for bounds in boundary_sets {
        let mut saved = 0.0f64;
        for (index, result) in results.iter().enumerate() {
            let _ = index;
            if result.fall_time > 0.0 {
                let fall = f64::from(result.fall_time);
                let stop = bounds.iter().copied().find(|&b| b >= fall).unwrap_or(trial);
                saved += trial - stop;
            }
        }
        println!(
            "segments at {bounds:?} s: {:.1}% of all steps saved",
            100.0 * saved / (count as f64 * total)
        );
    }
    println!("nodes creatures fallen after-fall share");
    for (nodes, (n, f, lost)) in by_nodes {
        println!(
            "{nodes:5} {n:9} {f:6} {:6.1}%",
            100.0 * lost / (n as f64 * total)
        );
    }
    Ok(())
}
