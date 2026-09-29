//! Prints the best elites' archive distance beside the distance of their
//! replay, which must match. Usage: replay_match <save> [count]
use anyhow::Result;
use evolution_simulator::{gpu::Gpu, storage};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("save path");
    let count: usize = args.next().map_or(10, |a| a.parse().unwrap());
    let experiment = storage::load(std::path::Path::new(&path))?;
    let _gpu = Gpu::new("RTX 4060")?;
    let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    for (k, e) in elites.iter().enumerate() {
        let cfg = e.replay_config(&experiment.config);
        let (_, result) = evolution_simulator::engine::replay(&e.creature, &cfg);
        println!("#{}: archive {:.4} replay {:.4} fine {}", k + 1, e.fitness, result.fitness, e.fine);
    }
    Ok(())
}
