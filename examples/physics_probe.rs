//! How a fresh random population fares under the selected physics
//! (`EVOLUTION_PHYSICS=2` for the v2 prototype): how many fall and when,
//! how far the survivors get, and the evaluation rate.
//! Usage: cargo run --release --example physics_probe [count] [seconds]
use evolution_simulator::{config::Config, cpu_engine, evolution};

fn main() -> anyhow::Result<()> {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let seconds: f32 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20.0);
    let cfg = Config {
        population: count,
        duration: seconds,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let pop = evolution::create(&cfg)?;
    let start = std::time::Instant::now();
    let results = cpu_engine::evaluate(&pop, &cfg);
    let wall = start.elapsed().as_secs_f64();
    let mut falls: Vec<f32> = results
        .iter()
        .filter(|r| r.fall_time > 0.0)
        .map(|r| r.fall_time)
        .collect();
    falls.sort_by(f32::total_cmp);
    let mut standing: Vec<f32> = results
        .iter()
        .filter(|r| r.fall_time == 0.0)
        .map(|r| r.fitness)
        .collect();
    standing.sort_by(f32::total_cmp);
    let q = |v: &[f32], p: f32| {
        if v.is_empty() {
            f32::NAN
        } else {
            v[((v.len() - 1) as f32 * p) as usize]
        }
    };
    let steps: f32 = results
        .iter()
        .map(|r| {
            if r.fall_time > 0.0 {
                r.fall_time
            } else {
                seconds
            }
        })
        .sum::<f32>()
        * cfg.fidelity().rate as f32;
    println!(
        "{count} bodies, {seconds} s: {} fell (fall time p10 {:.2} s, median {:.2} s, p90 {:.2} s); {} stood, distance median {:.2} m, p90 {:.2} m, best {:.2} m; {:.0} creature-steps/s",
        falls.len(),
        q(&falls, 0.1),
        q(&falls, 0.5),
        q(&falls, 0.9),
        standing.len(),
        q(&standing, 0.5),
        q(&standing, 0.9),
        q(&standing, 1.0),
        steps as f64 / wall,
    );
    Ok(())
}
