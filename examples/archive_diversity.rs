//! How varied the archives of a save are. The tool prints a block for the
//! global archive and one for each archive in the save's island list: the
//! isolated islands, the hub, the wild islands, the nurseries of new random
//! bodies and the nurseries of reshaped bodies. It needs no GPU.
//!
//! A block has two lines of numbers from `diversity::measure`. The first line
//! covers every elite that fills a cell. It gives the filled cells, the ways of
//! moving they cover, the distinct body plans and body types, the clades, the
//! classes of a reference grid of body shapes they occupy, and the spread of
//! their body shapes. A clade is the set of elites with the same oldest
//! recorded ancestor. The second line gives the best distance and the same
//! numbers for the elites within 1% of it. Many bodies tie at the top
//! distance, which is a speed limit of the physics, and the archive has
//! nothing left to select on there. An empty archive prints one line that says
//! so. At the end the tool prints the cells, the mean distance and the best
//! distance of the global archive for each shape class and size class.
//!
//! Usage: `archive_diversity <save>`
//!
//! A save of an older version loads too, with the scores it measured then.
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
    // The global archive first, then the archives of the island list in their
    // order, each with a name. The list holds the islands, then the nursery of
    // new random bodies of each island, then the nursery of reshaped bodies of
    // each island. A nursery is named by the number of its island.
    let archives = std::iter::once(("global".to_owned(), &experiment.archive)).chain(
        experiment.islands.iter().enumerate().map(|(k, island)| {
            let name = if k < storage::ISOLATED_ISLANDS {
                format!("island {k}")
            } else if k == storage::hub_island() {
                "hub".to_owned()
            } else if k < islands {
                format!("wild island {k}")
            } else if k < 2 * islands {
                format!("nursery {}", k - islands)
            } else {
                format!("reshaped {}", k - 2 * islands)
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
    // For each shape class and size class of the global archive: the cells,
    // the sum of their distances and the best distance.
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
