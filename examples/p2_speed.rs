//! Times the GPU engine on the population of a save, under physics v2: one
//! warm-up pass, then `repeats` timed passes over the same creatures.
//! Prints creatures/s and creature-steps/s (steps a creature simulated before
//! it fell or finished), then the same for each lane class alone with its
//! muscle-rounds histogram, and a hash of every creature's result bits, which
//! two runs of one population compare.
//!
//! Usage: p2_speed <save.evo | dump.bin> [count] [repeats]
use evolution_simulator::{
    config::Config,
    engine::{self, Engine},
    evolution::Population,
    storage, warp_kernel,
};
use std::time::{Duration, Instant};

fn run(
    engine: &mut impl Engine,
    pop: &Population,
    cfg: &Config,
) -> anyhow::Result<(f64, f64, f64, u64)> {
    let start = Instant::now();
    engine.submit(pop.clone(), cfg)?;
    let done = loop {
        if let Some(done) = engine.poll()? {
            break done;
        }
        engine.wait(Duration::from_millis(5));
    };
    let seconds = start.elapsed().as_secs_f64();
    let rate = f64::from(cfg.fidelity().rate);
    let total = f64::from(cfg.duration) * rate;
    let steps: f64 = done
        .results
        .iter()
        .map(|r| {
            let t = if r.fall_time > 0.0 {
                f64::from(r.fall_time) * rate
            } else if r.screened > 0.0 {
                f64::from(r.screened) * rate
            } else {
                total
            };
            t.min(total)
        })
        .sum();
    // FNV-1a over the result bits, in population order.
    let hash = bytemuck::cast_slice::<_, u8>(&done.results)
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
    Ok((
        pop.genomes.len() as f64 / seconds,
        steps / seconds,
        steps / done.busy_seconds.max(1e-9),
        hash,
    ))
}

/// The creatures of `pop` in lane class `w`.
fn class_subset(pop: &Population, w: usize) -> Population {
    let mut sub = Population::default();
    for (i, g) in pop.genomes.iter().enumerate() {
        if warp_kernel::class_of(g.node_count, g.muscle_count) == Some(w) {
            sub.push(pop.creature(i));
        }
    }
    sub
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("save path");
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let repeats: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(3);
    let mut pop = Population::default();
    let mut cfg = if path.ends_with(".bin") {
        // A creature dump (settings, population, elites) of a save this game
        // no longer reads.
        type Dump = (
            Config,
            Population,
            Vec<(evolution_simulator::evolution::Creature, Config, f32)>,
        );
        let (settings, all, _): Dump = bincode::deserialize(&std::fs::read(path)?)?;
        for i in 0..count.min(all.genomes.len()) {
            pop.push(all.creature(i));
        }
        settings
    } else {
        let e = storage::load(std::path::Path::new(path))?;
        for i in 0..count.min(e.ring_len()) {
            pop.push(e.creature(i));
        }
        e.config.clone()
    };
    cfg.screen = None;
    let mut engine = engine::gpu_engine("RTX 4060", 64)?;
    eprintln!("engine: {}", engine.name());
    run(&mut engine, &pop, &cfg)?;
    for _ in 0..repeats {
        let (creatures, steps, busy, hash) = run(&mut engine, &pop, &cfg)?;
        println!(
            "{} creatures: {creatures:.0} creatures/s, {:.1}M creature-steps/s ({:.1}M per GPU-busy second), results {hash:016x}",
            pop.genomes.len(),
            steps / 1e6,
            busy / 1e6
        );
    }
    for w in warp_kernel::CLASSES {
        let sub = class_subset(&pop, w);
        if sub.genomes.is_empty() {
            continue;
        }
        let mut rounds = [0usize; warp_kernel::ROUNDS + 1];
        for g in &sub.genomes {
            rounds[g.muscle_count.div_ceil(w)] += 1;
        }
        println!(
            "{w}-lane class: {} creatures, muscle rounds 0 to {}: {rounds:?}",
            sub.genomes.len(),
            warp_kernel::ROUNDS
        );
        for _ in 0..repeats {
            let (creatures, steps, busy, hash) = run(&mut engine, &sub, &cfg)?;
            println!(
                "  {creatures:.0} creatures/s, {:.1}M creature-steps/s ({:.1}M per GPU-busy second), results {hash:016x}",
                steps / 1e6,
                busy / 1e6
            );
        }
    }
    Ok(())
}
