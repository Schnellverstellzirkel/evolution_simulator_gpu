//! Times the GPU engine on the population of a save, under physics v2: one
//! warm-up pass, then `repeats` timed passes over the same creatures.
//! Prints creatures/s and creature-steps/s (steps a creature simulated before
//! it fell or finished).
//!
//! Usage: p2_speed <save.evo | dump.bin> [count] [repeats]
use evolution_simulator::{
    config::Config,
    engine::{self, Engine},
    evolution::Population,
    storage,
};
use std::time::{Duration, Instant};

fn run(
    engine: &mut impl Engine,
    pop: &Population,
    cfg: &Config,
) -> anyhow::Result<(f64, f64, f64)> {
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
    Ok((
        pop.genomes.len() as f64 / seconds,
        steps / seconds,
        steps / done.busy_seconds.max(1e-9),
    ))
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
        let (creatures, steps, busy) = run(&mut engine, &pop, &cfg)?;
        println!(
            "{} creatures: {creatures:.0} creatures/s, {:.1}M creature-steps/s ({:.1}M per GPU-busy second)",
            pop.genomes.len(),
            steps / 1e6,
            busy / 1e6
        );
    }
    Ok(())
}
