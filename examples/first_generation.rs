//! Scores a fresh random population on the GPU engine and prints the median,
//! 99th percentile and best distance. It is a quick check that a physics change
//! does not give random bodies free propulsion.
//!
//! Usage: `cargo run --release --example first_generation [count] [seconds]`
//! The defaults are 20000 bodies and 20 s trials, in the default world with no
//! early screen.
mod common;
use evolution_simulator::{config::Config, evolution};
fn main() -> anyhow::Result<()> {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let duration: f32 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20.0);
    // A fixed seed, and no early screen so that every trial runs in full.
    let cfg = Config {
        population: count,
        duration,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let pop = evolution::create(&cfg)?;
    let mut engine = common::open()?;
    // A failed trial scores `evolution::FAILED` (-1e20) and is left out.
    let mut distances: Vec<f32> = common::score(&mut engine, &pop, &cfg)?
        .iter()
        .map(|r| r.fitness)
        .filter(|f| f.is_finite() && *f > -1e10)
        .collect();
    distances.sort_by(f32::total_cmp);
    // The distance at quantile `q` of the sorted distances.
    let at = |q: f32| distances[((distances.len() - 1) as f32 * q) as usize];
    println!(
        "{} bodies, {duration} s: median {:.2} m, 99% {:.2} m, best {:.2} m",
        distances.len(),
        at(0.5),
        at(0.99),
        at(1.0)
    );
    Ok(())
}
