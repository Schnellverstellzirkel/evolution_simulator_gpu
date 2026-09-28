//! Would a physics change stop muscle monsters? Takes an evenly spaced sample
//! of a checkpoint's population once, then scores that sample on the CPU
//! engine under whatever physics the binary and environment give, with full
//! 20 s trials and no screening. Comparing the scores by muscle count across
//! physics variants shows whether many-muscle bodies lose their lead.
//!
//! Usage:
//!   cargo run --release --example monster_check -- sample <checkpoint> <count> <out.bin>
//!   cargo run --release --example monster_check -- eval <sample.bin> [seconds]
//! `eval` prints one line per creature: nodes, muscles, distance (m).
use anyhow::{Context, Result, ensure};
use bincode::Options;
use evolution_simulator::{
    config::Config,
    cpu_engine,
    evolution::{Creature, Population},
};
use std::io::Read;

#[derive(serde::Deserialize)]
struct Prefix {
    _config: Config,
    _pending: Option<Config>,
    _generation: u32,
    population: Population,
}

fn options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(16 << 30)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("sample") => {
            ensure!(args.len() == 4, "sample <checkpoint> <count> <out.bin>");
            let mut file = std::io::BufReader::new(std::fs::File::open(&args[1])?);
            let mut magic = [0; 8];
            file.read_exact(&mut magic)?;
            ensure!(&magic[..6] == b"EVORUS", "not a checkpoint");
            let prefix: Prefix =
                options().deserialize_from(zstd::stream::read::Decoder::new(file)?)?;
            let pop = prefix.population;
            let count: usize = args[2].parse()?;
            let step = (pop.genomes.len() / count).max(1);
            let sample: Vec<Creature> = (0..pop.genomes.len())
                .step_by(step)
                .take(count)
                .map(|i| pop.creature(i))
                .collect();
            std::fs::write(&args[3], options().serialize(&sample)?)?;
            eprintln!(
                "{} of {} creatures written to {}",
                sample.len(),
                pop.genomes.len(),
                args[3]
            );
        }
        Some("eval") => {
            let path = args.get(1).context("eval <sample.bin> [seconds]")?;
            let seconds: f32 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(20.0);
            let sample: Vec<Creature> = options().deserialize(&std::fs::read(path)?)?;
            let mut pop = Population::default();
            for creature in &sample {
                pop.push(creature.clone());
            }
            let cfg = Config {
                population: sample.len(),
                duration: seconds,
                random_seed: false,
                screen: None,
                ..Config::default()
            };
            let start = std::time::Instant::now();
            let results = cpu_engine::evaluate(&pop, &cfg);
            eprintln!(
                "{} creatures, {seconds} s trials, in {:.1} s",
                sample.len(),
                start.elapsed().as_secs_f64()
            );
            for (creature, result) in sample.iter().zip(&results) {
                println!(
                    "{} {} {}",
                    creature.nodes.len(),
                    creature.muscles.len(),
                    result.fitness
                );
            }
        }
        _ => anyhow::bail!("usage: monster_check sample|eval ..."),
    }
    Ok(())
}
