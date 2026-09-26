//! Prints the ancestor chain of a checkpoint's best elite.
fn main() {
    let path = std::env::args().nth(1).expect("checkpoint");
    let e = evolution_simulator::storage::load(std::path::Path::new(&path)).unwrap();
    let best = e
        .archive
        .entries
        .iter()
        .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        .unwrap();
    println!(
        "{} lineage records, best elite {:.2} m",
        e.lineage.len(),
        best.fitness
    );
    for a in e.ancestry(best.creature.id, 400).iter().take(15) {
        println!(
            "gen {:3}  {:6.2} m  {} nodes  {}",
            a.generation,
            a.fitness,
            a.creature.nodes.len(),
            a.change
        );
    }
}
