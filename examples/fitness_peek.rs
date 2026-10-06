//! Top fitness values per archive of a save, and how many elites sit within
//! 0.1 m of each archive's best. Usage: fitness_peek <checkpoint.evo>
use evolution_simulator::storage;

fn main() -> anyhow::Result<()> {
    let e = storage::load(std::path::Path::new(&std::env::args().nth(1).expect("checkpoint")))?;
    let show = |name: String, a: &evolution_simulator::qd::QdArchive| {
        let mut f: Vec<f32> = a.entries.iter().map(|x| x.fitness).collect();
        f.sort_by(|a, b| b.total_cmp(a));
        let best = f.first().copied().unwrap_or(0.0);
        let near = f.iter().filter(|&&x| x > best - 0.1).count();
        println!("{name}: {} elites, best {best:.3}, within 0.1 m {near}, top5 {:.3?}", f.len(), &f[..f.len().min(5)]);
    };
    show("global".into(), &e.archive);
    for (i, a) in e.islands.iter().enumerate().take(12) {
        show(format!("arena {i}"), a);
    }
    let arena: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(4);
    let mut top: Vec<_> = e.islands[arena].entries.iter().collect();
    top.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    for x in top.iter().step_by(400).take(14) {
        println!("{:.5} id {} nodes {} muscles {} age {} emitter {:?} niche {:?} change {:?}", x.fitness, x.creature.id, x.creature.node_count(), x.creature.muscle_count(), e.generation - x.improved_generation.min(e.generation), x.emitter, x.niche, e.lineage.get(&x.creature.id).map(|a| a.change.clone()));
    }
    println!("generation {}", e.generation);
    Ok(())
}
