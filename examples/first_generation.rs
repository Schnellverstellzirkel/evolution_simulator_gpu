//! Distances of a fresh random population on the CPU engine: a quick check
//! that a physics change does not hand random bodies free propulsion.
//! Usage: cargo run --release --example first_generation [count] [seconds]
use evolution_simulator::{config::Config, cpu_engine, evolution};
fn main() {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let duration: f32 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20.0);
    let cfg = Config {
        population: count,
        duration,
        random_seed: false,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).unwrap();
    let mut distances: Vec<f32> = cpu_engine::evaluate(&pop, &cfg)
        .iter()
        .map(|r| r.fitness)
        .filter(|f| f.is_finite() && *f > -1e10)
        .collect();
    distances.sort_by(f32::total_cmp);
    let at = |q: f32| distances[((distances.len() - 1) as f32 * q) as usize];
    println!(
        "{} bodies, {duration} s: median {:.2} m, 99% {:.2} m, best {:.2} m",
        distances.len(),
        at(0.5),
        at(0.99),
        at(1.0)
    );
}
