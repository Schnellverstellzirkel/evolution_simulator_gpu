//! How much of its parent's distance a child keeps, per structural operator.
//! Takes the best elites of a checkpoint's global archive, applies each
//! operator once to each elite (and repairs the body as breeding does, but
//! without the small parameter mutation that follows in breeding), and scores
//! parents and children on the GPU. A row for that parameter
//! mutation alone is the baseline.
//!
//! Usage: cargo run --release --example mutation_audit -- <checkpoint> [elites] [seconds]
//! Every operator runs by name, so `EVOLUTION_ANATOMY` need not be set.
mod common;
use anyhow::Context;
use evolution_simulator::{
    config::Config,
    evolution::{self, Creature, Rng},
    storage,
};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .context("usage: mutation_audit <checkpoint> [elites] [seconds]")?;
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(500);
    let seconds: f32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let experiment = storage::load(std::path::Path::new(path))?;
    let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    let cfg = Config {
        duration: seconds,
        screen: None,
        random_seed: false,
        ..experiment.config.clone()
    };
    let mut engine = common::open()?;
    let mut score = |creatures: &[Creature]| -> anyhow::Result<Vec<f32>> {
        if creatures.is_empty() {
            return Ok(Vec::new());
        }
        Ok(common::score_creatures(&mut engine, creatures, &cfg)?
            .iter()
            .map(|r| {
                if r.fitness.is_finite() {
                    r.fitness
                } else {
                    0.0
                }
            })
            .collect())
    };
    let parents: Vec<Creature> = elites.iter().map(|e| e.creature.clone()).collect();
    let parent_scores = score(&parents)?;
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    eprintln!(
        "{} elites, {seconds} s trials, parent median {:.2} m, mean {:.1} nodes / {:.1} muscles",
        parents.len(),
        quantile(&parent_scores, 0.5),
        mean(
            &parents
                .iter()
                .map(|c| c.nodes.len() as f32)
                .collect::<Vec<_>>()
        ),
        mean(
            &parents
                .iter()
                .map(|c| c.muscles.len() as f32)
                .collect::<Vec<_>>()
        ),
    );
    println!(
        "| operator | applied | child/parent median | p75 | keeps 90% | beats parent | nodes | muscles |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    let mut rows: Vec<(String, Vec<(usize, Creature)>)> = Vec::new();
    let local: Vec<(usize, Creature)> = parents
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut rng = Rng::new(0xa0d17, 0, i);
            (
                i,
                evolution::mutate_locally(p.clone(), &cfg, &mut rng, 0.035),
            )
        })
        .collect();
    rows.push(("parameter mutation 0.035 (baseline)".into(), local));
    for (k, name) in evolution::structural_operator_names()
        .into_iter()
        .enumerate()
    {
        let children: Vec<(usize, Creature)> = parents
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let mut child = p.clone();
                let mut rng = Rng::new(0xa0d17, k as u32 + 1, i);
                let donor = &parents[(i * 7 + 3) % parents.len()];
                let changed = evolution::apply_structural_operator(
                    name,
                    &mut child,
                    &cfg,
                    &mut rng,
                    Some(donor),
                )?;
                changed.then_some((i, child))
            })
            .collect();
        rows.push((name.to_string(), children));
    }
    for (name, children) in rows {
        let bodies: Vec<Creature> = children.iter().map(|(_, c)| c.clone()).collect();
        let scores = score(&bodies)?;
        let mut ratios = Vec::new();
        let (mut keeps, mut beats, mut counted) = (0, 0, 0);
        let (mut nodes, mut muscles) = (0.0f32, 0.0f32);
        for ((i, child), s) in children.iter().zip(&scores) {
            nodes += child.nodes.len() as f32 - parents[*i].nodes.len() as f32;
            muscles += child.muscles.len() as f32 - parents[*i].muscles.len() as f32;
            let parent = parent_scores[*i];
            if parent < 1.0 {
                continue;
            }
            let ratio = s.max(0.0) / parent;
            ratios.push(ratio);
            counted += 1;
            keeps += usize::from(ratio >= 0.9);
            beats += usize::from(*s > parent);
        }
        let n = children.len().max(1) as f32;
        let share = |k: usize| 100.0 * k as f32 / counted.max(1) as f32;
        println!(
            "| {name} | {:.0}% | {:.2} | {:.2} | {:.0}% | {:.0}% | {:+.2} | {:+.2} |",
            100.0 * children.len() as f32 / parents.len() as f32,
            quantile(&ratios, 0.5),
            quantile(&ratios, 0.75),
            share(keeps),
            share(beats),
            nodes / n,
            muscles / n,
        );
    }
    Ok(())
}

fn quantile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted[((sorted.len() - 1) as f32 * q).round() as usize]
}
