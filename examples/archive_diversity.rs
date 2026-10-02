//! How varied the archives of a save are.
//!
//! For the global archive and every island, nursery and hub: the filled
//! cells, the ways of moving they cover, the distinct body plans and body
//! types, the clades (elites with a common oldest recorded ancestor), the
//! share of a reference grid of body shapes they occupy, and the same for the
//! elites within 1% of the best distance (a distance that many bodies tie at
//! is a speed limit of the physics, and there the archive has nothing left to
//! select on). Then the global archive's elites per shape and size class.
//!
//! Usage: archive_diversity <save>
#[path = "diversity_common/mod.rs"]
mod diversity;
use anyhow::Result;
use evolution_simulator::{qd, storage};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: archive_diversity <save>");
    let path = std::path::Path::new(&path);
    let header = storage::peek(path)?;
    println!(
        "{}: generation {}, qd version {} (this build {})",
        path.display(),
        header.generation,
        header.qd_version,
        qd::VERSION
    );
    let experiment = storage::load_any_version(path)?;
    let islands = storage::island_count();
    let archives = std::iter::once(("global".to_owned(), &experiment.archive)).chain(
        experiment.islands.iter().enumerate().map(|(k, island)| {
            let name = if k < storage::ISOLATED_ISLANDS {
                format!("island {k}")
            } else if k == storage::hub_island() {
                "hub".to_owned()
            } else {
                format!("nursery {}", k - islands)
            };
            (name, island)
        }),
    );
    for (name, archive) in archives {
        if archive.behavior_count() == 0 {
            println!("{name}: empty");
            continue;
        }
        println!(
            "{name}: {}",
            diversity::measure(archive, &experiment, diversity::Part::All).line()
        );
        let best = archive
            .entries
            .iter()
            .filter(|e| !qd::is_morphology_niche(&e.niche))
            .map(|e| e.fitness)
            .fold(f32::MIN, f32::max);
        println!(
            "    best {best:.2} m, within 1% of it: {}",
            diversity::measure(archive, &experiment, diversity::Part::NearBest).line()
        );
    }
    // The global archive by body class.
    let mut classes: std::collections::BTreeMap<(u8, u8), (usize, f32, f32)> = Default::default();
    for e in experiment
        .archive
        .entries
        .iter()
        .filter(|e| !qd::is_morphology_niche(&e.niche))
    {
        let entry = classes
            .entry((e.niche.0[2], e.niche.0[5]))
            .or_insert((0, 0.0, f32::MIN));
        entry.0 += 1;
        entry.1 += e.fitness;
        entry.2 = entry.2.max(e.fitness);
    }
    println!("global archive by shape class and size class: cells, mean m, best m");
    for ((shape, size), (cells, sum, best)) in classes {
        println!(
            "    shape {shape} size {size}: {cells} cells, mean {:.1} m, best {best:.1} m",
            sum / cells as f32
        );
    }
    Ok(())
}
