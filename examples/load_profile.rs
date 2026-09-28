//! Times loading a save: reading the file, zstd decompression alone, the
//! full `storage::load`, and saving it again to a temporary file.
//! Usage: cargo run --release --example load_profile -- <save.evo>
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
    let started = Instant::now();
    let e = evolution_simulator::storage::load(&path)?;
    println!(
        "storage::load {:.2} s: generation {}, population {}, archive {}",
        started.elapsed().as_secs_f64(),
        e.generation,
        e.config.population,
        e.archive.entries.len(),
    );
    let out = std::env::temp_dir().join(format!("load-profile-{}.evo", std::process::id()));
    let started = Instant::now();
    evolution_simulator::storage::save(&out, &e)?;
    let size = std::fs::metadata(&out)?.len();
    let _ = std::fs::remove_file(&out);
    println!(
        "storage::save {:.2} s, {:.0} MB",
        started.elapsed().as_secs_f64(),
        size as f64 / 1e6
    );
    Ok(())
}
