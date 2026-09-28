//! Times loading a save: reading the file, the full `storage::load`, and
//! saving it again to a temporary file and loading that. With `--fill N` it
//! first runs N generations with made-up scores spread over the archive
//! cells, so a save holds full archives even when no GPU is free to score
//! the population.
//! Usage: cargo run --release --example load_profile -- <save.evo> [--fill N]
use std::io::Read;
use std::time::Instant;
fn main() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from(std::env::args().nth(1).expect("a save"));
    let started = Instant::now();
    let mut file = std::fs::File::open(&path)?;
    let mut buffer = vec![0u8; 1 << 20];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
    }
    println!(
        "read {:.0} MB in {:.2} s",
        bytes as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );
    let fill: u32 = std::env::args()
        .skip_while(|a| a != "--fill")
        .nth(1)
        .map_or(0, |v| v.parse().expect("generations"));
    let started = Instant::now();
    let mut e = evolution_simulator::storage::load(&path)?;
    println!(
        "storage::load {:.2} s: generation {}, population {}, archive {}",
        started.elapsed().as_secs_f64(),
        e.generation,
        e.config.population,
        e.archive.entries.len(),
    );
    for _ in 0..fill {
        fill_archives(&mut e)?;
        e.prepare_next_batch()?;
    }
    if fill > 0 {
        fill_archives(&mut e)?;
        println!(
            "filled: generation {}, archive {} elites, islands {} elites, lineage {}",
            e.generation,
            e.archive.entries.len(),
            e.islands.iter().map(|i| i.entries.len()).sum::<usize>(),
            e.lineage.len()
        );
    }
    let out = std::env::temp_dir().join(format!("load-profile-{}.evo", std::process::id()));
    let started = Instant::now();
    evolution_simulator::storage::save(&out, &e)?;
    let size = std::fs::metadata(&out)?.len();
    println!(
        "storage::save {:.2} s, {:.1} MB",
        started.elapsed().as_secs_f64(),
        size as f64 / 1e6
    );
    drop(e);
    let started = Instant::now();
    let again = evolution_simulator::storage::load(&out)?;
    let _ = std::fs::remove_file(&out);
    println!(
        "storage::load of that save {:.2} s: generation {}, population {}, archive {}",
        started.elapsed().as_secs_f64(),
        again.generation,
        again.population.genomes.len(),
        again.archive.entries.len()
    );
    Ok(())
}

/// Made-up scores and behaviors, spread over the archive cells by a hash
/// of the slot, then the game's archive step.
fn fill_archives(e: &mut evolution_simulator::storage::Experiment) -> anyhow::Result<()> {
    for i in 0..e.config.population {
        let h =
            (i as u64 ^ u64::from(e.generation) << 40).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 20;
        e.scores[i] = 1.0 + (h % 1000) as f32 * 0.1;
        e.trial_metrics[i] = evolution_simulator::qd::TrialMetrics {
            ground_contact: ((h >> 10) % 6) as f32 / 6.0 + 0.05,
            gait_frequency: ((h >> 13) % 8) as f32 * 0.75 + 0.1,
            mean_height: ((h >> 16) % 6) as f32 * 0.3 + 0.05,
            feet: ((h >> 19) % 5) as f32,
            ..Default::default()
        };
    }
    e.evaluated = e.config.population;
    e.stage = evolution_simulator::storage::Stage::Evaluated;
    e.archive_batch()
}
