//! Scores one elite alone, inside batches of its neighbours and with other ids, to see
//! whether its distance depends on anything but its genes.
//!
//! Before those tests it prints the lineage of the archive's three best
//! elites, scoring each parent again when its genes are kept. Then it prints
//! a table of the 15 best elites with their stored and replayed distances.
//! Every score is a GPU trial of the save's config at the standard physics
//! with the early screen off. The early rung rules stay as the save has them,
//! and the first line says whether it has any (`rungs true`). An elite marked
//! `fine` has a confirmation trial's score, so its replay here can differ.
//! Usage: replay_probe <checkpoint.evo> <archive: g or island number> [rank]
//!
//! `g` is the global archive. A number picks that entry of the save's
//! `islands`. `rank` counts from 0, the best elite, and picks the elite for
//! the tests. It defaults to 0.
mod common;
use evolution_simulator::{config::Config, storage};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let e = storage::load(std::path::Path::new(&args[1]))?;
    let archive = if args[2] == "g" {
        &e.archive
    } else {
        &e.islands[args[2].parse::<usize>()?]
    };
    let rank: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut elites: Vec<_> = archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    // `champ` is the elite at `rank`, the one under test.
    let elite = elites[rank];
    let champ = elite.creature.unpack();
    let cfg = Config {
        screen: None,
        ..e.config.clone()
    };
    let mut engine = common::open()?;
    println!(
        "stored {:.3}, id {}, rungs {:?}",
        elite.fitness,
        champ.id,
        e.config.rungs.is_some()
    );
    // The three best elites: their lineage records. When the parent's genes
    // are kept, the parent is scored again beside the score recorded for it.
    for x in elites.iter().take(3) {
        let a = e.lineage.get(&x.creature.id);
        println!(
            "lineage of {}: {:?}",
            x.creature.id,
            a.map(|a| (
                a.parent,
                a.fitness,
                a.generation,
                a.change.clone(),
                a.creature.is_empty()
            ))
        );
        if let Some(p) = a.and_then(|a| a.parent).and_then(|id| e.lineage.get(&id)) {
            if !p.creature.is_empty() {
                let r = common::score_creatures(&mut engine, &[p.creature.unpack()], &cfg)?;
                println!(
                    "  parent replay now {:.3} (recorded {:.3}) nodes {} muscles {} vs child nodes {} muscles {}",
                    r[0].fitness,
                    p.fitness,
                    p.creature.node_count(),
                    p.creature.muscle_count(),
                    x.creature.node_count(),
                    x.creature.muscle_count()
                );
            }
            println!(
                "  parent: fitness {} generation {} change {:?} genes kept {}",
                p.fitness,
                p.generation,
                p.change,
                !p.creature.is_empty()
            );
        }
    }
    // The 15 best elites in one batch: stored distance against replayed
    // distance, with the bookkeeping that might explain a gap.
    let top: Vec<_> = elites
        .iter()
        .take(15)
        .map(|x| x.creature.unpack())
        .collect();
    let r = common::score_creatures(&mut engine, &top, &cfg)?;
    println!(
        "generation {}, autochange step {}",
        e.generation, e.config.autochange_step
    );
    for (i, x) in elites.iter().take(15).enumerate() {
        println!(
            "rank {i}: stored {:.3} replay {:.3} improved_gen {} age {} visits {} protected_until {} fine {} graduate {} emitter {:?}",
            x.fitness,
            r[i].fitness,
            x.improved_generation,
            e.generation.saturating_sub(x.improved_generation),
            x.visits,
            x.protected_until,
            x.fine,
            x.graduate,
            x.emitter
        );
    }
    // `champ` alone, three times. The scores should be equal.
    for k in 0..3 {
        let r = common::score_creatures(&mut engine, std::slice::from_ref(&champ), &cfg)?;
        println!("alone #{k}: {:.3}", r[0].fitness);
    }
    // `champ` scored first in a batch of `n`, then last in the same batch.
    // The rest of the batch is the best elites after rank 0.
    for n in [8usize, 64, 400] {
        let mut batch: Vec<_> = elites
            .iter()
            .skip(1)
            .take(n - 1)
            .map(|x| x.creature.unpack())
            .collect();
        batch.insert(0, champ.clone());
        let r = common::score_creatures(&mut engine, &batch, &cfg)?;
        let mut tail = batch.clone();
        tail.rotate_left(1);
        let r2 = common::score_creatures(&mut engine, &tail, &cfg)?;
        println!(
            "in a batch of {n}: first {:.3}, last {:.3}",
            r[0].fitness,
            r2[n - 1].fitness
        );
    }
    // `champ` with some bits of its id flipped and its genes unchanged.
    for delta in [1u64, 2, 1 << 20, 1 << 40] {
        let mut c = champ.clone();
        c.id ^= delta;
        let r = common::score_creatures(&mut engine, &[c], &cfg)?;
        println!("id xor {delta}: {:.3}", r[0].fitness);
    }
    Ok(())
}
