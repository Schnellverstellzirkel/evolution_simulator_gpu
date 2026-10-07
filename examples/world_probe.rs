//! Scores one elite of a save in each world of the autochange ladder up to the
//! save's step, to find the world its stored score came from. The elite is the
//! one at `rank` by distance (0, the best, by default) in the global archive
//! (`g`) or in the save's archive with that island number. In the save's own
//! world it is also scored with the early screen and the rungs on and off and
//! under each creature flag. A chaos check scores 16 copies of it with slightly
//! changed node diameters.
//! Usage: world_probe <checkpoint.evo> <island|g> [rank]
mod common;
use evolution_simulator::{config::Config, environment, storage};

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
    let elite = elites[rank];
    let creature = elite.creature.unpack();
    let mut engine = common::open()?;
    let ladder = environment::autochange_ladder();
    let top = e.config.autochange_step as usize;
    // The save has applied the first `top` steps of the ladder. The list adds
    // the next two steps.
    println!("stored {:.3}; ladder steps up to {top}:", elite.fitness);
    for (i, &(idx, level)) in ladder.iter().enumerate().take(top + 2) {
        println!(
            "  step {i}: {} to level {level}",
            environment::EFFECTS[idx].name
        );
    }
    // The screen and rungs the loaded game would use for its next generation.
    println!(
        "screen in the save: {:?}; rungs {:?}",
        e.config.screen, e.config.rungs
    );
    {
        // Chaos check: 16 copies of the creature, each node diameter scaled by
        // 1 + eps * u, where u is a fixed pseudo-random number from -1 to 1.
        // Each copy keeps its own u values for every eps. The spread of the
        // scores shows how much a tiny change of the body moves the score.
        // These are full trials, with no screen and no rungs.
        for eps in [1e-7f32, 1e-6, 1e-5, 1e-4] {
            let mut batch = Vec::new();
            for k in 0..16u32 {
                let mut c = creature.clone();
                let mut h = (k as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                for n in c.nodes.iter_mut() {
                    h = h
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                    n.diameter *= 1.0 + eps * u;
                }
                batch.push(c);
            }
            let cfg = Config {
                screen: None,
                rungs: None,
                ..e.config.clone()
            };
            let r = common::score_creatures(&mut engine, &batch, &cfg)?;
            let mut v: Vec<f32> = r.iter().map(|x| x.fitness).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            println!(
                "diameter nudged by {eps:e}: min {:.2} median {:.2} max {:.2}",
                v[0], v[8], v[15]
            );
        }
    }
    // The save's config as it is, with the creature flags the game gives out
    // (`rungs`): none, exempt from the early rungs, the audit lane, and the
    // nursery bars of the early screen.
    for flags in [
        0u8,
        evolution_simulator::rungs::EXEMPT,
        evolution_simulator::rungs::AUDIT,
        evolution_simulator::rungs::EXEMPT | evolution_simulator::rungs::AUDIT,
        evolution_simulator::rungs::RESHAPED | evolution_simulator::rungs::EXEMPT,
        evolution_simulator::rungs::YOUNG | evolution_simulator::rungs::EXEMPT,
    ] {
        let mut pop = evolution_simulator::evolution::Population::default();
        pop.push(creature.clone());
        pop.flags = vec![flags];
        let r = common::score(&mut engine, &pop, &e.config)?;
        println!(
            "save config as is, flags {flags:#x}: {:.3} fall {:.2} screened {:?}",
            r[0].fitness, r[0].fall_time, r[0].screened
        );
    }
    // The same elite with the early screen and the rungs on or off.
    for (name, cfg) in [
        ("save config as is", e.config.clone()),
        (
            "screen off",
            Config {
                screen: None,
                ..e.config.clone()
            },
        ),
        (
            "rungs off",
            Config {
                rungs: None,
                ..e.config.clone()
            },
        ),
        (
            "screen and rungs off",
            Config {
                screen: None,
                rungs: None,
                ..e.config.clone()
            },
        ),
    ] {
        let r = common::score_creatures(&mut engine, std::slice::from_ref(&creature), &cfg)?;
        println!("{name}: {:.3} fall {:.2}", r[0].fitness, r[0].fall_time);
    }
    // The world after `k` ladder steps, from the save's step down to none. It
    // is the save's world with the steps from `k` on undone, the last first,
    // each by one level. The screen and the rungs are off, so each trial runs
    // in full.
    for k in (0..=top).rev() {
        let mut cfg = e.config.clone();
        for step in (k..top).rev() {
            let (idx, level) = ladder[step];
            environment::EFFECTS[idx].set_level(&mut cfg, level - 1);
        }
        let cfg = Config {
            screen: None,
            rungs: None,
            ..cfg
        };
        let r = common::score_creatures(&mut engine, std::slice::from_ref(&creature), &cfg)?;
        println!(
            "world after {k} steps: {:.3} (brambles {} wind {} slope {} friction {})",
            r[0].fitness, cfg.brambles, cfg.wind, cfg.slope, cfg.ground_friction
        );
    }
    Ok(())
}
