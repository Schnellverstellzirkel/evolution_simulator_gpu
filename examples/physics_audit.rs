//! What the scoring kernel records of elites' trials, per elite and in total.
//! Replays elites spread over the archive's rank order on the GPU engine
//! (the scoring kernel with recording) and prints the distance of the replay
//! beside the archive's, the body, the fall time, the share of steps with no
//! node on the ground, the largest ground push on one node, the lowest muscle
//! energy store and the steps with a joint past its break angle.
//! Diagnostic only.
//!
//! The GPU kernel does not expose solver ledgers (muscle work, energy the
//! solver gained or lost, the momentum balance, friction that pushed a node
//! the way it slid, cost of transport, bone load), so the audit has no such
//! columns.
//! With `random` in place of a checkpoint it audits a random first generation.
//! Usage: cargo run --release --example physics_audit <checkpoint.evo|random> [count]
mod common;
use evolution_simulator::{config::Config, physics, storage};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("checkpoint");
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let (config, mut creatures): (Config, Vec<(f32, evolution_simulator::evolution::Creature)>) =
        if path == "random" {
            let cfg = Config {
                population: count,
                random_seed: false,
                ..Config::default()
            };
            let pop = evolution_simulator::evolution::create(&cfg)?;
            let list = (0..count).map(|i| (0.0, pop.creature(i))).collect();
            (cfg, list)
        } else {
            let e = storage::load(std::path::Path::new(&path))?;
            let list = e
                .archive
                .entries
                .iter()
                .map(|x| (x.fitness, x.creature.clone()))
                .collect();
            (e.config.clone(), list)
        };
    let cfg = Config {
        screen: None,
        ..config
    };
    creatures.sort_by(|a, b| b.0.total_cmp(&a.0));
    let elites = creatures;
    let _engine = common::open()?;
    println!(
        "{} elites; replaying {count} spread over the rank order",
        elites.len()
    );
    println!(
        "rank  archive_m replay_m nodes muscles kg  fell_s  contact_free%  max_ground_N  min_store  broken_steps%"
    );
    let (mut tendon_muscles, mut all_muscles) = (0usize, 0usize);
    let mut replayed = 0usize;
    for k in 0..count {
        let rank = (elites.len() - 1) * k / (count - 1).max(1);
        let (archive_m, c) = &elites[rank];
        let recording = common::record(c, &cfg)?;
        let mass: f32 = physics::nodes(c).iter().map(|n| n.mass).sum();
        all_muscles += c.muscles.len();
        tendon_muscles += c.muscles.iter().filter(|m| m.tendon > 0.0).count();
        let (mut free, mut broken_steps, mut steps) = (0usize, 0usize, 0usize);
        let (mut max_ground, mut min_store) = (0.0f32, 1.0f32);
        if let Some(forces) = &recording.forces {
            // The first frames repeat the start pose in the air, by design.
            let settle = cfg.fidelity().settle() as usize;
            steps = forces.ground.len().saturating_sub(settle);
            free = forces
                .ground
                .iter()
                .skip(settle)
                .filter(|frame| frame.iter().all(|&n| n <= 0.0))
                .count();
            max_ground = forces
                .ground
                .iter()
                .skip(settle)
                .flatten()
                .copied()
                .fold(0.0, f32::max);
            min_store = forces
                .energy
                .iter()
                .skip(settle)
                .flatten()
                .copied()
                .fold(1.0, f32::min);
            broken_steps = forces
                .broken
                .iter()
                .skip(settle)
                .filter(|&&b| b != 0)
                .count();
        }
        let share = |n: usize| 100.0 * n as f32 / steps.max(1) as f32;
        println!(
            "{rank:5} {:9.2} {:8.2} {:5} {:7} {:5.1} {:7.2} {:13.1} {:13.0} {:10.2} {:13.1}",
            archive_m,
            recording.result.fitness,
            c.nodes.len(),
            c.muscles.len(),
            mass,
            recording.result.fall_time,
            share(free),
            max_ground,
            min_store,
            share(broken_steps),
        );
        replayed += 1;
    }
    println!("replayed {replayed} elites");
    println!(
        "muscles with an elastic tendon: {tendon_muscles} of {all_muscles} ({:.0}%)",
        100.0 * tendon_muscles as f64 / all_muscles.max(1) as f64
    );
    Ok(())
}
