//! Prints the best elites' archive distance beside their GPU replay's, which
//! must be equal. Usage: replay_match <save> [count]
//!
//!   replay_match <save> --retest <count> <out.csv>
//!
//! scores the save's best `count` elites in one batch, full 20 s trials at the
//! standard rate, and writes one row per elite: its archive distance, the
//! re-test distance and its fall time.
mod common;
use anyhow::{Context, Result};
use evolution_simulator::storage;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("save path");
    if args.get(1).is_some_and(|a| a == "--retest") {
        let count: usize = args.get(2).context("--retest needs a count")?.parse()?;
        let out = args.get(3).context("--retest needs an output path")?;
        return retest(path, count, out);
    }
    let count: usize = args.get(1).map_or(10, |a| a.parse().unwrap());
    let experiment = storage::load(std::path::Path::new(path))?;
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

fn retest(path: &str, count: usize, out: &str) -> Result<()> {
    let experiment = storage::load(std::path::Path::new(path))?;
    let mut engine = common::open()?;
    let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    let creatures: Vec<_> = elites.iter().map(|e| e.creature.unpack()).collect();
    let cfg = evolution_simulator::config::Config {
        screen: None,
        ..experiment.config.clone()
    };
    let results = common::score_creatures(&mut engine, &creatures, &cfg)?;
    let mut text = String::from("id,archive,fine,distance,fall_time\n");
    for (e, r) in elites.iter().zip(&results) {
        text += &format!(
            "{},{},{},{},{}\n",
            e.creature.id, e.fitness, e.fine as u8, r.fitness, r.fall_time
        );
    }
    std::fs::write(out, text)?;
    println!("{}: {} elites re-tested", out, results.len());
    Ok(())
}

