//! Times the GPU engine on the population of a save, under physics v2: one
//! warm-up pass, then `repeats` timed passes over the same creatures.
//! Prints creatures/s and creature-steps/s (steps a creature simulated before
//! it fell or finished). `EVOLUTION_CUDA=0` selects Vulkan.
//!
//! Usage: p2_speed <save.evo> [count] [repeats]
use evolution_simulator::{
    config::Config,
    engine::{self, Engine},
    evolution::Population,
    storage,
};
use std::time::{Duration, Instant};

fn run(engine: &mut impl Engine, pop: &Population, cfg: &Config) -> anyhow::Result<(f64, f64)> {
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
    Ok((pop.genomes.len() as f64 / seconds, steps / seconds))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("save path");
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let repeats: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(3);
    let e = storage::load(std::path::Path::new(path))?;
    let mut cfg = e.config.clone();
    cfg.screen = None;
    let mut pop = Population::default();
    for i in 0..count.min(e.ring_len()) {
        pop.push(e.creature(i));
    }
    let mut engine =
        engine::gpu_engine("RTX 4060", 64, evolution_simulator::gpu::DEFAULT_STEP_RANGE)?;
    eprintln!("engine: {}", engine.name());
    run(&mut engine, &pop, &cfg)?;
    for _ in 0..repeats {
        let (creatures, steps) = run(&mut engine, &pop, &cfg)?;
        println!(
            "{} creatures: {creatures:.0} creatures/s, {:.1}M creature-steps/s",
            pop.genomes.len(),
            steps / 1e6
        );
    }
    Ok(())
}
