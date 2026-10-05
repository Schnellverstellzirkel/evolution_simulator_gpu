//! Host cost of packing one ring block for the GPU, on the CPU alone: the
//! pack into new buffers followed by a copy into a staging buffer (the path
//! before host arenas), against the pack into the buffers of the previous
//! unit (`warp_kernel::pack_reusing`), which the engine copies from directly.
//! Prints wall seconds and page faults per block for each.
//! Usage: pack_bench [creatures] [repeats] [save]
use evolution_simulator::{config::Config, evolution, warp_kernel};
use std::time::Instant;

fn faults() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = stat.rsplit_once(')').map_or("", |r| r.1);
    let f: Vec<&str> = rest.split_whitespace().collect();
    let n = |i: usize| f.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    n(7) + n(9)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args.get(1).map_or(196_608, |v| v.parse().expect("number"));
    let repeats: usize = args.get(2).map_or(6, |v| v.parse().expect("number"));
    // The ring of a save has evolved bodies; without one, new random bodies.
    let (pop, cfg) = match args.get(3) {
        Some(path) => {
            let e = evolution_simulator::storage::load_any_version(std::path::Path::new(path))?;
            let block = &e.blocks[0];
            ((*block.population).clone(), (*block.config).clone())
        }
        None => {
            let cfg = Config {
                population: count,
                random_seed: false,
                ..Config::default()
            };
            (evolution::create(&cfg)?, cfg)
        }
    };
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    let mut staging: Vec<u8> = Vec::new();
    let mut spare = Vec::new();
    for round in 0..repeats {
        let (f0, t0) = (faults(), Instant::now());
        let batches = warp_kernel::pack(&pop, &indices, &cfg)?;
        let mut bytes = 0;
        for b in &batches {
            let w = b.wave.as_ref().unwrap();
            for data in [
                bytemuck::cast_slice::<u32, u8>(&w.lanes[..]),
                bytemuck::cast_slice(&w.muscles[..]),
                bytemuck::cast_slice(&w.ends[..]),
                bytemuck::cast_slice(&w.heads[..]),
            ] {
                if staging.len() < bytes + data.len() {
                    staging.resize(bytes + data.len(), 0);
                }
                staging[bytes..bytes + data.len()].copy_from_slice(data);
                bytes += data.len();
            }
        }
        drop(batches);
        let fresh = (t0.elapsed().as_secs_f64(), faults() - f0);
        let (f0, t0) = (faults(), Instant::now());
        let batches = warp_kernel::pack_reusing(&pop, &indices, &cfg, &mut spare)?;
        let reused = (t0.elapsed().as_secs_f64(), faults() - f0);
        spare.extend(batches);
        println!(
            "pack_bench: round {round}, {} creatures, {:.1} MB: new buffers and staging copy {:.3} s, {} page faults; reused buffers {:.3} s, {} page faults",
            pop.genomes.len(),
            bytes as f64 / 1e6,
            fresh.0,
            fresh.1,
            reused.0,
            reused.1
        );
    }
    Ok(())
}
