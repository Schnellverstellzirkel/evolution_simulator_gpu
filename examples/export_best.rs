//! Writes an elite of a checkpoint as creature JSON (for `creature_gif`):
//! the best one, or with a rank fraction the elite that far down the
//! archive ranked by distance (0.5 is the median elite).
//! Usage: cargo run --release --example export_best -- <checkpoint.evo> <out.json> [rank fraction]
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let e = evolution_simulator::storage::load(std::path::Path::new(&args[1]))?;
    let fraction: f32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let mut ranked: Vec<_> = e.archive.entries.iter().collect();
    ranked.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let rank = ((ranked.len() - 1) as f32 * fraction.clamp(0.0, 1.0)).round() as usize;
    let elite = ranked[rank];
    std::fs::write(&args[2], serde_json::to_string(&elite.creature)?)?;
    println!(
        "rank {rank} of {}: {:.2} m, {} nodes, {} muscles",
        ranked.len(),
        elite.fitness,
        elite.creature.nodes.len(),
        elite.creature.muscles.len()
    );
    Ok(())
}
