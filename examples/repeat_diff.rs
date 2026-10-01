//! Scores one population several times in one process and says which
//! creatures changed between the runs (a diagnostic).
use evolution_simulator::{engine::{self, Engine}, evolution::Population, storage, warp_kernel};
use std::time::Duration;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(300_000);
    let (pop, mut cfg) = if args[1] == "random" {
        let cfg = evolution_simulator::config::Config { population: count, random_seed: false, ..Default::default() };
        (evolution_simulator::evolution::create(&cfg)?, cfg)
    } else {
        let e = storage::load(std::path::Path::new(&args[1]))?;
        let mut pop = Population::default();
        for i in 0..count.min(e.ring_len()) { pop.push(e.creature(i)); }
        (pop, e.config.clone())
    };
    cfg.screen = None;
    let mut engine = engine::gpu_engine("RTX 4060", 64)?;
    let mut runs = Vec::new();
    for _ in 0..6 {
        engine.submit(pop.clone(), &cfg)?;
        let done = loop {
            if let Some(d) = engine.poll()? { break d; }
            engine.wait(Duration::from_millis(5));
        };
        runs.push(done.results);
    }
    let last = runs.last().unwrap();
    for (k, run) in runs.iter().enumerate() {
        let diff: Vec<usize> = (0..run.len()).filter(|&i| run[i].fitness.to_bits() != last[i].fitness.to_bits()).collect();
        let failed = run.iter().filter(|r| !(r.fitness > -1e10)).count();
        println!("run {k}: {} differ from the last run, {failed} failed", diff.len());
        if failed > 1000 {
            let share = |pred: &dyn Fn(usize) -> bool| {
                let f = run.iter().enumerate().filter(|(i, r)| !(r.fitness > -1e10) && pred(*i)).count() as f32 / failed as f32;
                let all = (0..run.len()).filter(|&i| pred(i)).count() as f32 / run.len() as f32;
                (f, all)
            };
            let c = |i: usize| pop.creature(i);
            let lig = share(&|i| c(i).bones.iter().any(|b| b.ligament > 0.0));
            let sen = share(&|i| c(i).muscles.iter().any(|m| m.sensor < 2));
            let big = share(&|i| c(i).nodes.len() >= 10);
            let wide = share(&|i| c(i).nodes.iter().any(|n| n.x.abs() > 3.0));
            println!("  failing vs all: ligament {lig:?} sensors {sen:?} >=10 nodes {big:?} nodes beyond 3 m {wide:?}");
        }
        if k < 2 && !diff.is_empty() {
            let mut classes = std::collections::BTreeMap::new();
            let mut firsts = diff.iter().take(8).copied().collect::<Vec<_>>();
            for &i in &diff {
                let g = &pop.genomes[i];
                let w = warp_kernel::class_of(g.node_count, g.muscle_count).unwrap_or(0);
                *classes.entry((w, g.muscle_count.div_ceil(w.max(1)))).or_insert(0usize) += 1;
            }
            println!("  (lane class, rounds) of the differing: {classes:?}");
            firsts.sort();
            println!("  first indices {firsts:?}, last {:?}", diff.last());
        }
    }
    Ok(())
}
