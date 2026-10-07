//! How much of its parent's distance a child keeps, per structural operator.
//! The audit applies each operator to each of the best elites of a
//! checkpoint's global archive, `variants` times, each time with its own
//! random stream. It repairs the body as breeding does and keeps the child
//! within the growth step that breeding allows (`evolution::child_limits`),
//! but it leaves out the small parameter mutation that follows in breeding.
//! Then it scores parents and children in full trials in the checkpoint's
//! world.
//!
//! The first row is the baseline, that parameter mutation alone. Next come
//! the operators alone, then the same operators with the parameter mutation
//! after each, as `<operator> + parameter mutation`. A row prints how often
//! the operator fit the body, the share of the parent's distance that the
//! child keeps (median and 75th percentile), how many children keep 90% and
//! how many beat their parent, and the mean change in nodes and in muscles.
//! The four ratio columns leave out parents that reached less than 1 m.
//!
//! A row also prints how many children would enter the global archive. A
//! child enters when it is faster than the elite that holds its cell. The
//! next column is the distance those entrants add to the archive per 1,000
//! children. The last column is how many children land in a cell nobody
//! holds. An archive that has refined its cells has many empty ones. A child
//! there enters whatever its distance, so it is not counted as an entrant.
//!
//! Usage: cargo run --release --example mutation_audit -- <checkpoint> [elites] [seconds] [variants]
//! The defaults are 500 elites, 20 s trials and 1 variant. Every operator
//! runs by name, so no setting is needed to switch one on.
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
    // The fastest elite of each cell, for the archive test. Reserve elites
    // hold no cell.
    let occupant: HashMap<&Niche, f32> = elites
        .iter()
        .filter(|e| !qd::is_morphology_niche(&e.niche))
        .map(|e| (&e.niche, e.fitness))
        .collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    // Full trials in the checkpoint's world: no early screen, no early rungs.
    let cfg = Config {
        duration: seconds,
        screen: None,
        rungs: None,
        random_seed: false,
        ..experiment.config.clone()
    };
    let mut engine = common::open()?;
    // Fitness and archive cell of every creature, in order. A fitness that is
    // not finite counts as 0.
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
    // draws 4 elites of the slot's archive at random and keeps the one whose
    // size differs most from the child's (`structural_mutation_among`). The
    // audit draws one elite at random from the whole global archive.
    let everyone: Vec<&qd::Elite> = experiment
        .archive
        .entries
        .iter()
        .filter(|e| !qd::is_morphology_niche(&e.niche))
        .collect();
    // The donor of variant `v` of elite `i`, the same for every operator.
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
    // The rows of the table: a label and its children, each as (parent
    // index, child).
    let mut rows: Vec<(String, Vec<(usize, Creature)>)> = Vec::new();
    // The baseline row: the parameter mutation alone.
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
    // parameter mutation that breeding gives the child of an operator that is
    // not compound. The child of a compound operator gets none there
    // (`anatomy::is_compound`).
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
            // A child enters when it is faster than the elite of its cell. In
            // an empty cell it counts as new, not as an entrant. Either way it
            // must have moved (a distance above 0).
            match occupant.get(niche) {
                Some(&held) if *s > held && *s > 0.0 => {
                    enters += 1;
                    gain += f64::from(*s - held);
                }
                None if *s > 0.0 => fresh += 1,
                _ => {}
            }
            // A parent under 1 m gives no usable ratio. Its children count in
            // the node, muscle and archive columns but not in the ratio ones.
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
