//! How much of its parent's distance a child keeps, per structural operator.
//! Takes the best elites of a checkpoint's global archive, applies each
//! operator to each elite (`variants` times, each with its own random
//! stream) and repairs the body as breeding does, within the growth step a
//! child of that parent may take (`evolution::child_limits`), but without
//! the small parameter mutation that follows in breeding. Then it scores
//! parents and children in full 20 s trials, in the checkpoint's world. A
//! row for that parameter mutation alone is the baseline, and every operator
//! has a second row, `<operator> + parameter mutation`, with that mutation
//! after it.
//!
//! The table prints, per operator: how often the operator fit the body, the
//! share of the parent's distance the child keeps (median and 75th
//! percentile), how many children keep 90% and how many beat their parent,
//! the nodes and muscles the child gained, how many children would enter the
//! global archive (a child enters when it is faster than the elite that holds
//! its cell), the distance those entrants add to the archive per 1,000
//! children, and how many land in a cell nobody holds (an archive that has
//! refined its cells has many empty ones, and a child there enters whatever
//! its distance, so those children are not counted as entrants).
//!
//! Usage: cargo run --release --example mutation_audit -- <checkpoint> [elites] [seconds] [variants]
//! Every operator runs by name, so no setting is needed to switch one on.
mod common;
use anyhow::Context;
use evolution_simulator::{
    config::Config,
    evolution::{self, Creature, Population, Rng},
    qd::{self, Niche},
    scheduler, storage,
};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .context("usage: mutation_audit <checkpoint> [elites] [seconds] [variants]")?;
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(500);
    let seconds: f32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let variants: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
    let experiment = storage::load_any_version(std::path::Path::new(path))?;
    let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
    // The fastest elite of each cell, for the archive test.
    let occupant: HashMap<&Niche, f32> = elites
        .iter()
        .filter(|e| !qd::is_morphology_niche(&e.niche))
        .map(|e| (&e.niche, e.fitness))
        .collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    let cfg = Config {
        duration: seconds,
        screen: None,
        rungs: None,
        random_seed: false,
        ..experiment.config.clone()
    };
    let mut engine = common::open()?;
    // Fitness and archive cell of every creature, in order.
    let mut score = |creatures: &[Creature]| -> anyhow::Result<Vec<(f32, Niche)>> {
        if creatures.is_empty() {
            return Ok(Vec::new());
        }
        let mut pop = Population::default();
        for c in creatures {
            pop.push(c.clone());
        }
        let results = common::score(&mut engine, &pop, &cfg)?;
        Ok(results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let m = scheduler::to_metrics(&pop, i, r, &cfg);
                let g = &pop.genomes[i];
                let d = qd::descriptor(
                    &pop.nodes[g.node_start..g.node_start + g.node_count],
                    &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count],
                    m.behavior,
                );
                let fitness = if r.fitness.is_finite() {
                    r.fitness
                } else {
                    0.0
                };
                (fitness, experiment.archive.cell_of(d))
            })
            .collect())
    };
    let parents: Vec<Creature> = elites.iter().map(|e| e.creature.unpack()).collect();
    // Donors for the operators that take limbs from another elite. Breeding
    // draws one at random from the island's archive, so the audit draws one
    // at random from the whole global archive.
    let everyone: Vec<&qd::Elite> = experiment
        .archive
        .entries
        .iter()
        .filter(|e| !qd::is_morphology_niche(&e.niche))
        .collect();
    let donor_of = |i: usize, v: usize| -> Creature {
        let mut rng = Rng::new(0xd0409, v as u32, i);
        everyone[rng.index(everyone.len())].creature.unpack()
    };
    let parent_scores: Vec<f32> = score(&parents)?.into_iter().map(|(f, _)| f).collect();
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    eprintln!(
        "{} elites, {variants} variants, {seconds} s trials, parent median {:.2} m, mean {:.1} nodes / {:.1} muscles",
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
        "| operator | applied | child/parent median | p75 | keeps 90% | beats parent | nodes | muscles | enters archive | archive gain per 1k | new cell |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    // The children of one operator: (parent index, child).
    let mut rows: Vec<(String, Vec<(usize, Creature)>)> = Vec::new();
    let local: Vec<(usize, Creature)> = parents
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            let cfg = &cfg;
            (0..variants).map(move |v| {
                let mut rng = Rng::new(0xa0d17, v as u32 * 1000, i);
                (
                    i,
                    evolution::mutate_locally(p.clone(), cfg, &mut rng, 0.035),
                )
            })
        })
        .collect();
    rows.push(("parameter mutation 0.035 (baseline)".into(), local));
    // Every operator alone, then every operator followed by the small
    // parameter mutation that the children of the other operators get in
    // breeding (a compound child gets none there).
    let names = evolution::structural_operator_names();
    let passes: Vec<(usize, &str, bool)> = [false, true]
        .into_iter()
        .flat_map(|noise| {
            names
                .iter()
                .enumerate()
                .map(move |(k, name)| (k, *name, noise))
        })
        .collect();
    for (k, name, noise) in passes {
        let children: Vec<(usize, Creature)> = parents
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                let cfg = &cfg;
                let donor_of = &donor_of;
                (0..variants).filter_map(move |v| {
                    let mut child = p.clone();
                    let mut rng = Rng::new(0xa0d17, k as u32 + 1 + v as u32 * 1000, i);
                    let donor = donor_of(i, v);
                    // The limits breeding gives a child of this parent.
                    let limited = evolution::child_limits(cfg, p, evolution::GROWTH_STEP);
                    let limits = limited.as_ref().unwrap_or(cfg);
                    let changed = evolution::apply_structural_operator(
                        name,
                        &mut child,
                        limits,
                        &mut rng,
                        Some(&donor),
                    )?;
                    if !changed {
                        return None;
                    }
                    if noise {
                        let mut rng = Rng::new(0xa0d18, k as u32 + 1 + v as u32 * 1000, i);
                        child = evolution::mutate_locally(child, limits, &mut rng, 0.035);
                    }
                    Some((i, child))
                })
            })
            .collect();
        let label = if noise {
            format!("{name} + parameter mutation")
        } else {
            name.to_string()
        };
        rows.push((label, children));
    }
    for (name, children) in rows {
        let bodies: Vec<Creature> = children.iter().map(|(_, c)| c.clone()).collect();
        let scores = score(&bodies)?;
        let mut ratios = Vec::new();
        let (mut keeps, mut beats, mut counted) = (0, 0, 0);
        let (mut nodes, mut muscles) = (0.0f32, 0.0f32);
        let (mut enters, mut gain, mut fresh) = (0usize, 0.0f64, 0usize);
        for ((i, child), (s, niche)) in children.iter().zip(&scores) {
            nodes += child.nodes.len() as f32 - parents[*i].nodes.len() as f32;
            muscles += child.muscles.len() as f32 - parents[*i].muscles.len() as f32;
            match occupant.get(niche) {
                Some(&held) if *s > held && *s > 0.0 => {
                    enters += 1;
                    gain += f64::from(*s - held);
                }
                None if *s > 0.0 => fresh += 1,
                _ => {}
            }
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
            "| {name} | {:.0}% | {:.2} | {:.2} | {:.0}% | {:.0}% | {:+.2} | {:+.2} | {:.2}% | {:.2} | {:.2}% |",
            100.0 * children.len() as f32 / (parents.len() * variants) as f32,
            quantile(&ratios, 0.5),
            quantile(&ratios, 0.75),
            share(keeps),
            share(beats),
            nodes / n,
            muscles / n,
            100.0 * enters as f32 / n,
            1000.0 * gain / f64::from(n),
            100.0 * fresh as f32 / n,
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
