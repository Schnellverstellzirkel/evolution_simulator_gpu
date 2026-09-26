//! Body size and foot slip of the fastest archive elites: how long each body
//! is, what it weighs, and how far its feet slide while touching the ground.
//! With `EVOLUTION_LEDGER` set, it also prints where the three fastest
//! bodies' forward momentum comes from.
//! Usage: cargo run --release --example size_report <checkpoint.evo> [count]
use evolution_simulator::{cpu_engine, evolution::Population, physics, storage};
fn main() {
    let path = std::env::args().nth(1).expect("checkpoint");
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let e = storage::load(std::path::Path::new(&path)).unwrap();
    let mut elites: Vec<_> = e.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let settle = physics::settle() as usize;
    let amplitude = physics::terrain_amplitude(e.config.terrain);
    println!("distance_m  nodes  length_m  longest_bone_m  mass_kg  slip_m  slip_share");
    let mut lengths = Vec::new();
    let mut shares = Vec::new();
    for elite in elites.iter().take(count) {
        let c = &elite.creature;
        let nodes = physics::nodes(c);
        let mass: f32 = nodes.iter().map(|n| n.mass).sum();
        let length: f32 = c.bones.iter().map(|b| b.rest_length).sum();
        let longest = c.bones.iter().map(|b| b.rest_length).fold(0.0, f32::max);
        let frames = cpu_engine::trajectory(c, &e.config);
        // A node slides when it moves sideways while resting on the ground.
        let mut slip = 0.0f32;
        for t in settle + 1..frames.len() {
            for (j, node) in nodes.iter().enumerate() {
                let (now, before) = (frames[t][j], frames[t - 1][j]);
                let floor = physics::terrain(now[0], amplitude).0 + node.radius;
                if now[1] <= floor + 0.002 && before[1] <= floor + 0.002 {
                    slip += (now[0] - before[0]).abs();
                }
            }
        }
        let share = slip / elite.fitness.abs().max(0.01);
        lengths.push(length);
        shares.push(share);
        println!(
            "{:10.1}  {:5}  {:8.2}  {:14.2}  {:7.2}  {:6.1}  {:10.2}",
            elite.fitness,
            c.nodes.len(),
            length,
            longest,
            mass,
            slip,
            share
        );
    }
    lengths.sort_by(f32::total_cmp);
    shares.sort_by(f32::total_cmp);
    if !lengths.is_empty() {
        println!(
            "median body length {:.2} m, median slip per meter traveled {:.2}",
            lengths[lengths.len() / 2],
            shares[shares.len() / 2]
        );
    }
    if std::env::var_os("EVOLUTION_LEDGER").is_none() {
        return;
    }
    let names = [
        "integration speed cap",
        "ground contact",
        "velocity-pass speed cap",
        "velocity-pass constraints",
        "projection COM shift",
        "muscle forces",
    ];
    for elite in elites.iter().take(3) {
        *cpu_engine::LEDGER.lock().unwrap() = [0.0; 6];
        let mut pop = Population::default();
        pop.push(elite.creature.clone());
        let result = cpu_engine::evaluate(&pop, &e.config);
        let mass: f32 = physics::nodes(&elite.creature).iter().map(|n| n.mass).sum();
        println!(
            "ledger for {:.1} m ({:.0} kg), kg*m/s over the trial:",
            result[0].fitness, mass
        );
        for (name, v) in names.iter().zip(*cpu_engine::LEDGER.lock().unwrap()) {
            println!("  {name:28} {v:+12.1}");
        }
    }
}
