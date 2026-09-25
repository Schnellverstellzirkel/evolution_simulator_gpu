//! How many creatures share an exact body plan (bones and muscle attachments)?
use std::collections::HashMap;
fn main() {
    let path = std::env::args().nth(1).unwrap_or("bench/w3-seed38-100k.evo".into());
    let e = evolution_simulator::storage::load(std::path::Path::new(&path)).unwrap();
    let mut groups: HashMap<Vec<u32>, usize> = HashMap::new();
    for i in 0..e.population.genomes.len() {
        let c = e.population.creature(i);
        let mut key = vec![c.nodes.len() as u32];
        key.extend(c.bones.iter().flat_map(|b| [b.a, b.b]));
        key.push(u32::MAX);
        key.extend(c.muscles.iter().flat_map(|m| [m.bone_a, m.bone_b]));
        *groups.entry(key).or_default() += 1;
    }
    let mut sizes: Vec<usize> = groups.values().copied().collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let total: usize = sizes.iter().sum();
    let padded16: usize = sizes.iter().map(|s| s.div_ceil(16) * 16).sum();
    println!(
        "{} creatures, {} body plans, largest {:?}, 16-lane efficiency {:.1}%",
        total,
        sizes.len(),
        &sizes[..sizes.len().min(10)],
        100.0 * total as f64 / padded16 as f64
    );
}
