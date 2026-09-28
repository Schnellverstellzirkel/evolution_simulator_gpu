//! Rewrites a save in the current format, so the game will open it. Saves
//! hold only the archives and search state, so this works only for a save
//! made under the current physics: an older one loses its archives on
//! loading and would come out as a random population, and is refused.
//! Usage: cargo run --release --example upgrade_save -- <old.evo> <new.evo>
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: upgrade_save <old.evo> <new.evo>");
    let started = std::time::Instant::now();
    let experiment = evolution_simulator::storage::load(std::path::Path::new(&args[1]))?;
    let loaded = started.elapsed().as_secs_f64();
    anyhow::ensure!(
        !experiment.archive.entries.is_empty(),
        "{} has no archive under the current physics; nothing to keep",
        args[1]
    );
    evolution_simulator::storage::save(std::path::Path::new(&args[2]), &experiment)?;
    println!(
        "{}: generation {}, {} creatures, loaded in {loaded:.1} s, saved in {:.1} s",
        args[2],
        experiment.generation,
        experiment.config.population,
        started.elapsed().as_secs_f64() - loaded
    );
    Ok(())
}
