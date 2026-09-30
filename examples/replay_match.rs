//! Prints the best elites' archive distance beside their GPU replay's, which
//! must be equal. Usage: replay_match <save> [count]
mod common;
use anyhow::Result;
use evolution_simulator::storage;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("save path");
    let count: usize = args.next().map_or(10, |a| a.parse().unwrap());
    let experiment = storage::load(std::path::Path::new(&path))?;
    let _engine = common::open()?;
    let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    for (k, e) in elites.iter().take(count).enumerate() {
        let (creature, cfg) = e.replay_of(&experiment.config);
        let recording = common::record(&creature, &cfg)?;
        println!(
            "#{}: archive {:.4} replay {:.4}{}",
            k + 1,
            e.fitness,
            recording.result.fitness,
            if e.fine { " (confirmation trial)" } else { "" }
        );
    }
    Ok(())
}
