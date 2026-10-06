//! How many elites of each archive are not in canonical bone order, the order a replay
//! (Population::push) gives them. Usage: canon_check <checkpoint.evo>
use evolution_simulator::{evolution::canonicalize_bone_order, storage};

fn main() -> anyhow::Result<()> {
    let e = storage::load(std::path::Path::new(&std::env::args().nth(1).expect("checkpoint")))?;
    let mut archives: Vec<(String, &evolution_simulator::qd::QdArchive)> = vec![("global".into(), &e.archive)];
    for (i, a) in e.islands.iter().enumerate().take(12) { archives.push((format!("arena {i}"), a)); }
    for (name, a) in archives {
        let mut elites: Vec<_> = a.entries.iter().collect();
        elites.sort_by(|x, y| y.fitness.total_cmp(&x.fitness));
        let changed: Vec<usize> = elites.iter().enumerate().filter(|(_, x)| {
            let c = x.creature.unpack();
            let mut d = c.clone();
            canonicalize_bone_order(&mut d);
            c.bones[..] != d.bones[..] || c.nodes[..] != d.nodes[..] || c.muscles[..] != d.muscles[..]
        }).map(|(i, _)| i).collect();
        println!("{name}: {} elites, {} not canonical, first ranks {:?}", elites.len(), changed.len(), &changed[..changed.len().min(8)]);
    }
    Ok(())
}
