//! Shared by the diagnostic examples: they score and replay creatures on the
//! GPU engine, the game's scoring kernel, and nowhere else. There is no CPU
//! fallback, so a machine without a working primary GPU fails loudly.
#![allow(dead_code)]
use anyhow::{Context, Result};
use evolution_simulator::{
    config::Config,
    creature_kernel::GpuResult,
    engine::{self, Engine, Recording, ThreadedEngine},
    evolution::{Creature, Population},
    gpu::DEFAULT_STEP_RANGE,
};
use std::time::Duration;

/// Creatures per submitted unit. Small enough for the GPU memory that the
/// owner's game leaves free.
const UNIT: usize = 50_000;

/// How long a replay waits for the GPU.
const REPLAY_PATIENCE: Duration = Duration::from_secs(300);

/// Opens the primary GPU and makes it the engine that records replays.
pub fn open() -> Result<ThreadedEngine> {
    let engine = engine::gpu_engine("RTX 4060", 64, DEFAULT_STEP_RANGE)
        .context("the primary GPU did not open (set EVOLUTION_DEVICES=primary, and check nvidia-smi for free memory)")?;
    eprintln!("engine: {}", engine.name());
    engine.publish_replays();
    Ok(engine)
}

/// Scores every creature of `pop` with `cfg` on the GPU, in population order.
pub fn score(engine: &mut ThreadedEngine, pop: &Population, cfg: &Config) -> Result<Vec<GpuResult>> {
    let total = pop.genomes.len();
    let mut results = Vec::with_capacity(total);
    for begin in (0..total).step_by(UNIT) {
        let end = (begin + UNIT).min(total);
        let mut unit = Population::default();
        for i in begin..end {
            unit.push(pop.creature(i));
        }
        let cfg = Config {
            population: unit.genomes.len(),
            ..cfg.clone()
        };
        engine.submit(unit, &cfg)?;
        let done = loop {
            if let Some(done) = engine.poll()? {
                break done;
            }
            engine.wait(Duration::from_millis(5));
        };
        results.extend(done.results);
    }
    Ok(results)
}

/// Scores a list of creatures.
pub fn score_creatures(
    engine: &mut ThreadedEngine,
    creatures: &[Creature],
    cfg: &Config,
) -> Result<Vec<GpuResult>> {
    let mut pop = Population::default();
    for c in creatures {
        pop.push(c.clone());
    }
    score(engine, &pop, cfg)
}

/// Records one creature's full trial (no early screen) with the scoring
/// kernel. Frames are at the trial's own fidelity, one per step. Call `open`
/// first.
pub fn record(creature: &Creature, cfg: &Config) -> Result<Recording> {
    let cfg = Config {
        screen: None,
        population: 1,
        ..cfg.clone()
    };
    engine::record_on_gpu(creature, &cfg, REPLAY_PATIENCE).context("the GPU did not record the replay")
}
