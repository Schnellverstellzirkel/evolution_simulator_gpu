//! Memory held by a loaded checkpoint: process RSS after loading and the
//! size of the experiment's largest parts.
//!
//! Usage: cargo run --release --example mem_report -- <checkpoint>
use anyhow::{Context, Result};
use evolution_simulator::storage;

fn rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
        })
        .unwrap_or(0.0)
        / 1024.0
}

fn mib<T>(v: &Vec<T>) -> f64 {
    (v.capacity() * std::mem::size_of::<T>()) as f64 / 1048576.0
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: mem_report <checkpoint>")?;
    println!("RSS before load: {:.0} MiB", rss_mib());
    let e = storage::load(std::path::Path::new(&path))?;
    println!("RSS after load: {:.0} MiB", rss_mib());
    let p = &e.population;
    println!(
        "population: {} creatures, {:.0} MiB (genomes {:.0}, nodes {:.0}, bones {:.0}, muscles {:.0})",
        p.genomes.len(),
        p.bytes() as f64 / 1048576.0,
        mib(&p.genomes),
        mib(&p.nodes),
        mib(&p.bones),
        mib(&p.muscles)
    );
    println!(
        "per slot: scores {:.0}, parent_scores {:.0}, ranks {:.0}, parents {:.0}, trial_metrics {:.0}, candidate_emitters {:.0}, candidate_cma {:.0}, candidate_parent_ids {:.0}, protected_until {:.0}, screen_distance {:.0} MiB",
        mib(&e.scores),
        mib(&e.parent_scores),
        mib(&e.ranks),
        mib(&e.parents),
        mib(&e.trial_metrics),
        mib(&e.candidate_emitters),
        mib(&e.candidate_cma),
        mib(&e.candidate_parent_ids),
        mib(&e.protected_until),
        mib(&e.screen_distance)
    );
    let elites = |a: &evolution_simulator::qd::QdArchive| a.entries.len();
    println!(
        "archives: global {} elites, islands {:?}, lineage {}, history {}, reseed {}, fossils {}",
        elites(&e.archive),
        e.islands.iter().map(elites).collect::<Vec<_>>(),
        e.lineage.len(),
        e.history.len(),
        e.reseed.len(),
        e.fossils.len()
    );
    Ok(())
}
