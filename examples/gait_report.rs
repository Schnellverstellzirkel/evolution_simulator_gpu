//! Gait check for the best archive elites: compares how often node velocities
//! reverse (a jiggle signature) with the creature's fastest muscle rhythm.
//! Usage: cargo run --release --example gait_report <checkpoint.evo> [count]
use evolution_simulator::{cpu_engine, physics, storage};
fn main() {
    let path = std::env::args().nth(1).expect("checkpoint");
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let e = storage::load(std::path::Path::new(&path)).unwrap();
    let mut elites: Vec<_> = e.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let rate = physics::rate() as f32;
    let settle = physics::settle() as usize;
    println!("fitness  nodes muscles  muscle_hz  reversal_hz  ratio");
    let mut ratios = Vec::new();
    for elite in elites.iter().take(count) {
        let c = &elite.creature;
        let frames = cpu_engine::trajectory(c, &e.config);
        let steps = frames.len() - 1 - settle;
        let mut reversals = 0usize;
        for node in 0..c.nodes.len() {
            let mut last = 0.0f32;
            for t in settle + 1..frames.len() {
                let v = frames[t][node][0] - frames[t - 1][node][0];
                if v.abs() > 1e-5 {
                    if last != 0.0 && v.signum() != last.signum() {
                        reversals += 1;
                    }
                    last = v;
                }
            }
        }
        let seconds = steps as f32 / rate;
        let reversal_hz = reversals as f32 / c.nodes.len() as f32 / seconds / 2.0;
        let muscle_hz = c.muscles.iter().map(|m| 1.0 / m.period).fold(0.0, f32::max);
        let ratio = reversal_hz / muscle_hz.max(1e-3);
        ratios.push(ratio);
        println!(
            "{:7.2}  {:5} {:7}  {:9.2}  {:11.2}  {:5.2}",
            elite.fitness,
            c.nodes.len(),
            c.muscles.len(),
            muscle_hz,
            reversal_hz,
            ratio
        );
    }
    ratios.sort_by(f32::total_cmp);
    println!(
        "median ratio of node oscillation to fastest muscle rhythm: {:.2}",
        ratios[ratios.len() / 2]
    );
}
