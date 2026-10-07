//! How many elites of each archive are not in canonical bone order, the order a
//! replay (`Population::push`) gives them. It checks the global archive and the
//! first 12 archives in `islands`, which it calls arenas. For each one it prints
//! the elite count, how many are not canonical, and the ranks (0 is the best) of
//! the first eight non-canonical ones.
//! Usage: canon_check <save>
use evolution_simulator::{evolution::canonicalize_bone_order, storage};

fn main() -> anyhow::Result<()> {
    let e = storage::load(std::path::Path::new(
        &std::env::args().nth(1).expect("checkpoint"),
    ))?;
    // The global archive, then the first 12 archives in `islands`.
    let mut archives: Vec<(String, &evolution_simulator::qd::QdArchive)> =
        vec![("global".into(), &e.archive)];
    for (i, a) in e.islands.iter().enumerate().take(12) {
        archives.push((format!("arena {i}"), a));
    }
    for (name, a) in archives {
        // Best first, so an index is a rank.
        let mut elites: Vec<_> = a.entries.iter().collect();
        elites.sort_by(|x, y| y.fitness.total_cmp(&x.fitness));
        // The ranks of the elites that canonicalizing the bone order changes.
        let changed: Vec<usize> = elites
            .iter()
            .enumerate()
            .filter(|(_, x)| {
                let c = x.creature.unpack();
                let mut d = c.clone();
                canonicalize_bone_order(&mut d);
                c.bones[..] != d.bones[..]
                    || c.nodes[..] != d.nodes[..]
                    || c.muscles[..] != d.muscles[..]
            })
            .map(|(i, _)| i)
            .collect();
        println!(
            "{name}: {} elites, {} not canonical, first ranks {:?}",
            elites.len(),
            changed.len(),
            &changed[..changed.len().min(8)]
        );
    }
    Ok(())
}
