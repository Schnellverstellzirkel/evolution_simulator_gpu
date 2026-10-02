//! What a save's island archives and nurseries hold: how many distinct body
//! plans, how old each body plan is, and how much of each island descends
//! from new random bodies (graduates of its nursery). Reads the archives
//! only (`storage::load_archives`), so it needs no GPU and little memory.
//! Usage: island_report <save>
use anyhow::Result;
use evolution_simulator::{qd, storage};
use std::collections::{HashMap, HashSet};

/// Generations since the last change of body plan along the elite's
/// recorded ancestry, and the generation of its oldest recorded ancestor.
fn ages(experiment: &storage::Experiment, elite: &qd::Elite) -> (u32, u32) {
    let now = experiment.generation;
    let chain = experiment.ancestry(elite.creature.id, usize::MAX);
    let mut plan_start = elite.improved_generation;
    for ancestor in &chain {
        if qd::Topology::of(&ancestor.creature) == elite.topology {
            plan_start = plan_start.min(ancestor.generation);
        } else {
            break;
        }
    }
    let root = chain
        .last()
        .map_or(elite.improved_generation, |a| a.generation);
    (now.saturating_sub(plan_start), root)
}

fn quantiles(values: &[f32]) -> String {
    if values.is_empty() {
        return "none".into();
    }
    let mut values = values.to_vec();
    values.sort_by(f32::total_cmp);
    let q = |p: f64| values[((values.len() - 1) as f64 * p) as usize];
    format!(
        "10% {:.1} 50% {:.1} 90% {:.1} max {:.1}",
        q(0.1),
        q(0.5),
        q(0.9),
        values[values.len() - 1]
    )
}

fn report(name: &str, archive: &qd::QdArchive, experiment: &storage::Experiment) {
    let elites: Vec<&qd::Elite> = archive
        .entries
        .iter()
        .filter(|e| !qd::is_morphology_niche(&e.niche))
        .collect();
    let reserve: Vec<&qd::Elite> = archive
        .entries
        .iter()
        .filter(|e| qd::is_morphology_niche(&e.niche))
        .collect();
    if elites.is_empty() {
        println!("{name}: empty, reserve {}", reserve.len());
        return;
    }
    let mut plans: HashMap<&qd::Topology, usize> = HashMap::new();
    for e in &elites {
        *plans.entry(&e.topology).or_default() += 1;
    }
    let reserve_plans: HashSet<&qd::Topology> = reserve.iter().map(|e| &e.topology).collect();
    // Skeletons: the bones alone, whatever the muscles. Classes: node and
    // muscle counts.
    let mut skeletons: HashSet<(u32, Vec<(u32, u32)>)> = HashSet::new();
    let mut classes: HashSet<(usize, usize)> = HashSet::new();
    let mut nodes_hist: std::collections::BTreeMap<usize, usize> = Default::default();
    for e in &elites {
        let nodes = e.topology.nodes as u32;
        skeletons.insert((
            nodes,
            e.topology.edges.iter().copied().filter(|&(a, b)| a < nodes && b < nodes).collect(),
        ));
        classes.insert((e.creature.nodes.len(), e.creature.muscles.len()));
        *nodes_hist.entry(e.creature.nodes.len()).or_default() += 1;
    }
    let largest = plans.values().copied().max().unwrap_or(0);
    let single = plans.values().filter(|&&n| n == 1).count();
    let fitness: Vec<f32> = elites.iter().map(|e| e.fitness).collect();
    let mut plan_age: Vec<f32> = Vec::new();
    let mut root: Vec<f32> = Vec::new();
    let mut graduates = 0;
    let mut from_random = 0;
    for e in &elites {
        let (age, root_generation) = ages(experiment, e);
        plan_age.push(age as f32);
        root.push(root_generation as f32);
        graduates += usize::from(e.graduate);
        let chain = experiment.ancestry(e.creature.id, usize::MAX);
        if chain.last().is_some_and(|a| a.change.contains("new random body")) {
            from_random += 1;
        }
    }
    let best = fitness.iter().copied().fold(f32::MIN, f32::max);
    let tied: Vec<&&qd::Elite> = elites.iter().filter(|e| e.fitness >= best - 1e-4).collect();
    let mut tied_bits: HashMap<u32, usize> = HashMap::new();
    for e in &elites {
        *tied_bits.entry(e.fitness.to_bits()).or_default() += 1;
    }
    let mut ties: Vec<usize> = tied_bits.values().copied().filter(|&n| n > 1).collect();
    ties.sort_by(|a, b| b.cmp(a));
    println!(
        "    {} cells tie with the best within 0.1 mm (mean nodes {:.1}, {} plans); {} fitness values shared by more than one cell, largest groups {:?}",
        tied.len(),
        tied.iter().map(|e| e.creature.nodes.len() as f32).sum::<f32>() / tied.len().max(1) as f32,
        tied.iter().map(|e| &e.topology).collect::<HashSet<_>>().len(),
        ties.len(),
        &ties[..ties.len().min(8)]
    );
    let young = |limit: f32| plan_age.iter().filter(|&&a| a < limit).count();
    println!(
        "    {} skeletons, {} node and muscle count classes; cells by node count: {}",
        skeletons.len(),
        classes.len(),
        nodes_hist.iter().map(|(n, c)| format!("{n}:{c}")).collect::<Vec<_>>().join(" ")
    );
    println!(
        "{name}: {} cells, {} body plans ({} held by one cell, largest plan {} cells), reserve {} entries of {} plans ({} not in a cell); fitness {}",
        elites.len(),
        plans.len(),
        single,
        largest,
        reserve.len(),
        reserve_plans.len(),
        reserve_plans.iter().filter(|t| !plans.contains_key(*t)).count(),
        quantiles(&fitness),
    );
    println!(
        "    plan age (generations since the last body change): {}; younger than 10: {}, 50: {}, 200: {}; oldest ancestor generation {}; direct graduates {graduates}, rooted in a random body {from_random}",
        quantiles(&plan_age),
        young(10.0),
        young(50.0),
        young(200.0),
        quantiles(&root),
    );
    // Fitness by plan age: do the young plans hold the slow cells?
    let mut by_age: Vec<(f32, f32)> = plan_age.iter().copied().zip(fitness_of(&elites)).collect();
    by_age.sort_by(|a, b| a.0.total_cmp(&b.0));
    let third = by_age.len() / 3;
    if third > 0 {
        let mean = |part: &[(f32, f32)]| part.iter().map(|p| p.1).sum::<f32>() / part.len() as f32;
        println!(
            "    mean fitness of the youngest third of plans {:.1}, middle {:.1}, oldest {:.1}",
            mean(&by_age[..third]),
            mean(&by_age[third..2 * third]),
            mean(&by_age[2 * third..])
        );
    }
}

fn fitness_of(elites: &[&qd::Elite]) -> Vec<f32> {
    elites.iter().map(|e| e.fitness).collect()
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("usage: island_report <save>");
    let experiment = storage::load_archives(std::path::Path::new(&path))?;
    println!(
        "{path}: generation {}, population {}, {} lineage records",
        experiment.generation,
        experiment.config.population,
        experiment.lineage.len()
    );
    if std::env::var_os("HISTORY").is_some() {
        for stats in experiment.history.iter().step_by(25) {
            println!(
                "history generation {}: best {:.1} m, qd {:.0}, cells {}",
                stats.generation, stats.best, stats.qd_score, stats.archive_cells
            );
        }
    }
    println!("{:?}", experiment.config);
    let islands = storage::island_count();
    for island in 0..islands {
        let Some(archive) = experiment.islands.get(island) else {
            continue;
        };
        let name = if island == storage::hub_island() {
            format!("hub {island}")
        } else {
            format!("island {island}")
        };
        report(&name, archive, &experiment);
        if let Some(nursery) = experiment.islands.get(storage::nursery_of(island)) {
            report(&format!("  nursery {island}"), nursery, &experiment);
        }
        if let Some(reshaped) = experiment.islands.get(storage::reshaped_of(island)) {
            report(&format!("  reshaped {island}"), reshaped, &experiment);
        }
    }
    // The distinct plans across the four isolated islands.
    let mut all: HashSet<&qd::Topology> = HashSet::new();
    for island in experiment.islands.iter().take(storage::ISOLATED_ISLANDS) {
        for e in &island.entries {
            all.insert(&e.topology);
        }
    }
    println!("distinct plans over the four isolated islands (cells and reserves): {}", all.len());
    Ok(())
}
