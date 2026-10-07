//! This file holds the helpers that score and replay creatures for the
//! diagnostic examples and for the GPU tests `cuda_physics`, `screening` and
//! `rungs`, which include it with `#[path]`. Every score and replay comes from
//! the game's scoring kernel on the GPU. There is no CPU fallback, so `open`
//! fails on a machine whose primary GPU does not open.
// Each file that includes this one uses only some of its functions.
#![allow(dead_code)]
use anyhow::{Context, Result};
use evolution_simulator::{
    config::Config,
    creature_kernel::GpuResult,
    engine::{self, Engine, Recording, ThreadedEngine},
    evolution::{Creature, Population},
};
use std::time::Duration;

/// Creatures per submitted unit. Small enough for the GPU memory that the
/// owner's game leaves free.
const UNIT: usize = 50_000;

/// How long `record` waits for the GPU to return a replay.
const REPLAY_PATIENCE: Duration = Duration::from_secs(300);

/// Opens the CUDA device whose name contains `RTX 4060`, the primary GPU, and
/// makes it the engine that records replays. `record` works only while the
/// returned engine is alive.
pub fn open() -> Result<ThreadedEngine> {
    let engine = engine::gpu_engine("RTX 4060", 64)
        .context("the primary GPU did not open (set EVOLUTION_DEVICES=primary, and check nvidia-smi for free memory)")?;
    eprintln!("engine: {}", engine.name());
    engine.publish_replays();
    Ok(engine)
}

/// Scores every creature of `pop` with `cfg` on the GPU, in population order.
/// The creatures go to the engine in units of `UNIT`, one unit at a time, and
/// the population's trial flags go with them.
pub fn score(
    engine: &mut ThreadedEngine,
    pop: &Population,
    cfg: &Config,
) -> Result<Vec<GpuResult>> {
    let total = pop.genomes.len();
    let mut results = Vec::with_capacity(total);
    for begin in (0..total).step_by(UNIT) {
        let end = (begin + UNIT).min(total);
        let mut unit = Population::default();
        for i in begin..end {
            unit.push(pop.creature(i));
        }
        if !pop.flags.is_empty() {
            unit.flags = pop.flags[begin..end].to_vec();
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

/// Scores `creatures` in list order with `score`.
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

/// Records one creature's full trial with the scoring kernel. It removes the
/// early screen from `cfg` and leaves `cfg.rungs` as it is. The frames run at
/// the trial's own fidelity, one per step, and the first ones show the start
/// pose during the settling steps. Call `open` first and keep its engine alive.
pub fn record(creature: &Creature, cfg: &Config) -> Result<Recording> {
    let cfg = Config {
        screen: None,
        population: 1,
        ..cfg.clone()
    };
    engine::record_on_gpu(creature, &cfg, REPLAY_PATIENCE)
        .context("the GPU did not record the replay")
}
