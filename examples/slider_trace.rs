//! Prints a creature's replay step by step: its center of mass, and for each
//! node the horizontal move over the step and the height above its radius.
//! A node that stays at height 0 while it moves forward is sliding.
//! Usage: cargo run --release --example slider_trace -- <creature.json> <first second> <steps>
use evolution_simulator::{config::Config, cpu_engine, evolution::Creature, physics};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let creature: Creature =
        serde_json::from_reader(std::fs::File::open(&args[1]).expect("creature JSON"))
            .expect("a creature");
    let first: f32 = args.get(2).map_or(10.0, |v| v.parse().expect("seconds"));
    let steps: usize = args.get(3).map_or(40, |v| v.parse().expect("steps"));
    let cfg = Config::default();
    let (frames, result) = cpu_engine::replay(&creature, &cfg);
    let nodes = physics::body(&creature.nodes, &creature.bones);
    let mass: f32 = nodes.iter().map(|n| n.mass).sum();
    let rate = cfg.fidelity().rate as f32;
    let settle = cfg.fidelity().settle() as usize;
    println!(
        "fitness {:.2} fall {:.2} feet {}",
        result.fitness,
        result.fall_time,
        result.feet()
    );
    for (j, n) in nodes.iter().enumerate() {
        println!("node {j}: mass {:.3} radius {:.3}", n.mass, n.radius);
    }
    let start = (settle + (first * rate) as usize).max(1);
    for t in start..(start + steps).min(frames.len()) {
        let com = |axis: usize| {
            frames[t]
                .iter()
                .zip(&nodes)
                .map(|(p, n)| p[axis] * n.mass)
                .sum::<f32>()
                / mass
        };
        let mut row = format!("{:6} com {:8.4} {:6.3} |", t - settle, com(0), com(1));
        for (j, (p, n)) in frames[t].iter().zip(&nodes).enumerate() {
            row += &format!(" {:+.4},{:.4}", p[0] - frames[t - 1][j][0], p[1] - n.radius);
        }
        println!("{row}");
    }
}
