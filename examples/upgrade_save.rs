//! Rewrites a save from an older game version in the current format, so the
//! game will open it. Its archive is emptied and its population is scored
//! again from scratch under the current physics, as `storage::load` does for
//! old saves. Meant for benchmark checkpoints.
//! Usage: cargo run --release --example upgrade_save -- <old.evo> <new.evo>
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: upgrade_save <old.evo> <new.evo>");
    let started = std::time::Instant::now();
    let experiment = evolution_simulator::storage::load(std::path::Path::new(&args[1]))?;
    let loaded = started.elapsed().as_secs_f64();
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
