//! Times the CPU v2 engine on the population of a save: creatures/s over the
//! first N creatures, full 20 s trials, no screen, on the rayon pool
//! (`RAYON_NUM_THREADS`). Prints a checksum of the distances so a change can
//! be compared bit for bit.
//!
//! Usage: p2_cpu_speed <save.evo> [count] [repeats]
use evolution_simulator::{cpu_v2, evolution::Population, physics2, storage};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("save path");
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let repeats: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(2);
    let e = storage::load(std::path::Path::new(path))?;
    let mut cfg = e.config.clone();
    cfg.screen = None;
    let mut pop = Population::default();
    // P2_REPLICATE=k pushes each creature k times (more creatures per body
    // plan, as a 3M generation has).
    let copies: usize = std::env::var("P2_REPLICATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    for i in 0..count.min(e.population.genomes.len()) {
        for _ in 0..copies {
            pop.push(e.population.creature(i));
        }
    }
    if std::env::var_os("P2_CMP").is_some() {
        let a = physics2::evaluate(&pop, &cfg);
        let b = cpu_v2::evaluate(&pop, &cfg);
        let mut bad = 0;
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let same = x.fitness.to_bits() == y.fitness.to_bits()
                && x.fall_time.to_bits() == y.fall_time.to_bits()
                && x.gait_frequency.to_bits() == y.gait_frequency.to_bits()
                && x.ground_contact.to_bits() == y.ground_contact.to_bits()
                && x.height_sum.to_bits() == y.height_sum.to_bits();
            if !same {
                bad += 1;
                if bad <= 5 {
                    println!(
                        "creature {i}: scalar {} m fall {} / lanes {} m fall {}",
                        x.fitness, x.fall_time, y.fitness, y.fall_time
                    );
                }
            }
        }
        println!("{} creatures, {bad} differ", a.len());
        return Ok(());
    }
    for _ in 0..repeats {
        let start = Instant::now();
        let results = cpu_v2::evaluate(&pop, &cfg);
        let seconds = start.elapsed().as_secs_f64();
        let mut hash = 0xcbf29ce484222325u64;
        for r in &results {
            for bits in [
                r.fitness.to_bits(),
                r.fall_time.to_bits(),
                r.gait_frequency.to_bits(),
            ] {
                hash = (hash ^ u64::from(bits)).wrapping_mul(0x100000001b3);
            }
        }
        println!(
            "{} creatures on {} threads: {:.0} creatures/s, checksum {hash:016x}",
            pop.genomes.len(),
            rayon::current_num_threads(),
            pop.genomes.len() as f64 / seconds
        );
    }
    Ok(())
}
