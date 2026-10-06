//! Scores one elite alone, inside batches of its neighbours and with other ids, to see
//! whether its distance depends on anything but its genes.
//! Usage: replay_probe <checkpoint.evo> <archive: g or island> [rank]
mod common;
use evolution_simulator::{config::Config, storage};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let e = storage::load(std::path::Path::new(&args[1]))?;
    let archive = if args[2] == "g" { &e.archive } else { &e.islands[args[2].parse::<usize>()?] };
    let rank: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut elites: Vec<_> = archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let elite = elites[rank];
    let champ = elite.creature.unpack();
    let cfg = Config { screen: None, ..e.config.clone() };
    let mut engine = common::open()?;
    println!("stored {:.3}, id {}, rungs {:?}", elite.fitness, champ.id, e.config.rungs.is_some());
    for x in elites.iter().take(3) {
        let a = e.lineage.get(&x.creature.id);
        println!("lineage of {}: {:?}", x.creature.id, a.map(|a| (a.parent, a.fitness, a.generation, a.change.clone(), a.creature.is_empty())));
        if let Some(p) = a.and_then(|a| a.parent).and_then(|id| e.lineage.get(&id)) {
            if !p.creature.is_empty() {
                let r = common::score_creatures(&mut engine, &[p.creature.unpack()], &cfg)?;
                println!("  parent replay now {:.3} (recorded {:.3}) nodes {} muscles {} vs child nodes {} muscles {}", r[0].fitness, p.fitness, p.creature.node_count(), p.creature.muscle_count(), x.creature.node_count(), x.creature.muscle_count());
            }
            println!("  parent: fitness {} generation {} change {:?} genes kept {}", p.fitness, p.generation, p.change, !p.creature.is_empty());
        }
    }
    let top: Vec<_> = elites.iter().take(15).map(|x| x.creature.unpack()).collect();
    let r = common::score_creatures(&mut engine, &top, &cfg)?;
    println!("generation {}, autochange step {}", e.generation, e.config.autochange_step);
    for (i, x) in elites.iter().take(15).enumerate() {
        println!("rank {i}: stored {:.3} replay {:.3} improved_gen {} age {} visits {} protected_until {} fine {} graduate {} emitter {:?}", x.fitness, r[i].fitness, x.improved_generation, e.generation.saturating_sub(x.improved_generation), x.visits, x.protected_until, x.fine, x.graduate, x.emitter);
    }
    for k in 0..3 {
        let r = common::score_creatures(&mut engine, &[champ.clone()], &cfg)?;
        println!("alone #{k}: {:.3}", r[0].fitness);
    }
    for n in [8usize, 64, 400] {
        let mut batch: Vec<_> = elites.iter().skip(1).take(n - 1).map(|x| x.creature.unpack()).collect();
        batch.insert(0, champ.clone());
        let r = common::score_creatures(&mut engine, &batch, &cfg)?;
        let mut tail = batch.clone();
        tail.rotate_left(1);
        let r2 = common::score_creatures(&mut engine, &tail, &cfg)?;
        println!("in a batch of {n}: first {:.3}, last {:.3}", r[0].fitness, r2[n - 1].fitness);
    }
    for delta in [1u64, 2, 1 << 20, 1 << 40] {
        let mut c = champ.clone();
        c.id ^= delta;
        let r = common::score_creatures(&mut engine, &[c], &cfg)?;
        println!("id xor {delta}: {:.3}", r[0].fitness);
    }
    Ok(())
}
