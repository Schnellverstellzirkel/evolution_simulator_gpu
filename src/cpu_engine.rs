//! CPU evaluation of physics v2: the fast lane engine for whole populations
//! (`cpu_v2`, 16 creatures per SIMD group) and the scalar reference for single
//! trials and replays (`physics2`). Both give the same results bit for bit
//! (`tests/physics2_lanes.rs`).
use crate::{config::Config, creature_kernel::GpuResult, evolution::Population, physics};

/// Node positions of one creature's full trial: entry `t` is the state after
/// `t` steps. The replay shows exactly this.
pub fn trajectory(creature: &crate::evolution::Creature, cfg: &Config) -> Vec<Vec<[f32; 2]>> {
    replay(creature, cfg).0
}

/// A creature's recorded trial and the result the engine scored for it, from
/// one run, so a replay can never disagree with its own score.
pub fn replay(
    creature: &crate::evolution::Creature,
    cfg: &Config,
) -> (Vec<Vec<[f32; 2]>>, GpuResult) {
    crate::physics2::replay(creature, cfg)
}

/// Cost of transport of one creature over a full CPU trial: muscle work in
/// joules per kilogram per meter (the work is what drains the muscles' energy
/// stores). Diagnostic only, never fitness. `None` when the creature did not
/// move forward.
pub fn transport_cost(creature: &crate::evolution::Creature, cfg: &Config) -> Option<f32> {
    let (distance, work) = crate::physics2::trial_work(creature, cfg);
    let mass: f32 = physics::nodes(creature).iter().map(|n| n.mass).sum();
    (distance > 0.01 && mass > 0.0).then(|| work as f32 / (mass * distance))
}

/// Evaluates every creature of `unit` and returns results in unit order.
pub fn evaluate(unit: &Population, cfg: &Config) -> Vec<GpuResult> {
    crate::cpu_v2::evaluate(unit, cfg)
}
