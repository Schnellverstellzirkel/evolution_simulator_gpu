//! Scores an elite in each world of the autochange ladder up to the save's step, to find
//! the world its stored score came from. Usage: world_probe <checkpoint.evo> <island> [rank]
mod common;
use evolution_simulator::{config::Config, environment, storage};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let e = storage::load(std::path::Path::new(&args[1]))?;
    let archive = if args[2] == "g" { &e.archive } else { &e.islands[args[2].parse::<usize>()?] };
    let rank: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut elites: Vec<_> = archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let elite = elites[rank];
    let creature = elite.creature.unpack();
    let mut engine = common::open()?;
    let ladder = environment::autochange_ladder();
    let top = e.config.autochange_step as usize;
    println!("stored {:.3}; ladder steps up to {top}:", elite.fitness);
    for (i, &(idx, level)) in ladder.iter().enumerate().take(top + 2) {
        println!("  step {i}: {} to level {level}", environment::EFFECTS[idx].name);
    }
    println!("screen in the save: {:?}; rungs {:?}", e.config.screen, e.config.rungs);
    {
        // Chaos check: the same genes, each node coordinate nudged by a relative 1e-6 or 1e-4.
        for eps in [1e-7f32, 1e-6, 1e-5, 1e-4] {
            let mut batch = Vec::new();
            for k in 0..16u32 {
                let mut c = creature.clone();
                let mut h = (k as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                for n in c.nodes.iter_mut() {
                    h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let u = ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                    n.diameter *= 1.0 + eps * u;
                }
                batch.push(c);
            }
            let cfg = Config { screen: None, rungs: None, ..e.config.clone() };
            let r = common::score_creatures(&mut engine, &batch, &cfg)?;
            let mut v: Vec<f32> = r.iter().map(|x| x.fitness).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            println!("diameter nudged by {eps:e}: min {:.2} median {:.2} max {:.2}", v[0], v[8], v[15]);
        }
    }
    for flags in [0u8, evolution_simulator::rungs::EXEMPT, evolution_simulator::rungs::AUDIT, evolution_simulator::rungs::EXEMPT | evolution_simulator::rungs::AUDIT, evolution_simulator::rungs::RESHAPED | evolution_simulator::rungs::EXEMPT, evolution_simulator::rungs::YOUNG | evolution_simulator::rungs::EXEMPT] {
        let mut pop = evolution_simulator::evolution::Population::default();
        pop.push(creature.clone());
        pop.flags = vec![flags];
        let r = common::score(&mut engine, &pop, &e.config)?;
        println!("save config as is, flags {flags:#x}: {:.3} fall {:.2} screened {:?}", r[0].fitness, r[0].fall_time, r[0].screened);
    }
    for (name, cfg) in [
        ("save config as is", e.config.clone()),
        ("screen off", Config { screen: None, ..e.config.clone() }),
        ("rungs off", Config { rungs: None, ..e.config.clone() }),
        ("screen and rungs off", Config { screen: None, rungs: None, ..e.config.clone() }),
    ] {
        let r = common::score_creatures(&mut engine, &[creature.clone()], &cfg)?;
        println!("{name}: {:.3} fall {:.2}", r[0].fitness, r[0].fall_time);
    }
    for k in (0..=top).rev() {
        let mut cfg = e.config.clone();
        for step in (k..top).rev() {
            let (idx, level) = ladder[step];
            environment::EFFECTS[idx].set_level(&mut cfg, level - 1);
        }
        let cfg = Config { screen: None, rungs: None, ..cfg };
        let r = common::score_creatures(&mut engine, &[creature.clone()], &cfg)?;
        println!("world after {k} steps: {:.3} (brambles {} wind {} slope {} friction {})", r[0].fitness, cfg.brambles, cfg.wind, cfg.slope, cfg.ground_friction);
    }
    Ok(())
}
