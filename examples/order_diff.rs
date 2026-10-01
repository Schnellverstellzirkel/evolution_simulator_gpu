//! Scores one population in its own order and in other orders and unit
//! sizes, and says which creatures got different results (a diagnostic).
use evolution_simulator::{engine::{self, Engine}, evolution::Population, storage};
use std::time::Duration;
fn run(engine: &mut impl Engine, pop: &Population, cfg: &evolution_simulator::config::Config) -> anyhow::Result<Vec<u32>> {
    engine.submit(pop.clone(), cfg)?;
    let done = loop {
        if let Some(d) = engine.poll()? { break d; }
        engine.wait(Duration::from_millis(5));
    };
    Ok(done.results.iter().map(|r| r.fitness.to_bits()).collect())
}
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(300_000);
    let e = storage::load(std::path::Path::new(&args[1]))?;
    let mut pop = Population::default();
    for i in 0..count.min(e.ring_len()) { pop.push(e.creature(i)); }
    let mut cfg = e.config.clone();
    cfg.screen = None;
    let mut engine = engine::gpu_engine("RTX 4060", 64)?;
    let base = run(&mut engine, &pop, &cfg)?;
    let n = pop.genomes.len();
    // Orders: reversed, rotated, and strided.
    let orders: Vec<(&str, Vec<usize>)> = vec![
        ("reversed", (0..n).rev().collect()),
        ("rotated", (0..n).map(|i| (i + n / 3) % n).collect()),
        ("strided", (0..n).map(|i| (i * 7919) % n).collect()),
    ];
    for (name, order) in orders {
        let mut shuffled = Population::default();
        for &i in &order { shuffled.push(pop.creature(i)); }
        let got = run(&mut engine, &shuffled, &cfg)?;
        let diff = order.iter().enumerate().filter(|&(k, &i)| got[k] != base[i]).count();
        println!("{name}: {diff} of {n} differ from the original order");
    }
    // A creature alone.
    let mut differ = 0;
    for i in (0..n).step_by(n / 200 + 1) {
        let mut one = Population::default();
        one.push(pop.creature(i));
        if run(&mut engine, &one, &cfg)?[0] != base[i] { differ += 1; }
    }
    println!("{differ} of 200 creatures scored alone differ");
    Ok(())
}
