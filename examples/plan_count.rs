//! Distinct bone trees ("plans") in a save's archives and bred ring, or in a
//! creature dump (docs/plan-2m.md, the tree-count row). A per-plan baked
//! kernel is compiled for one tree, so this counts how many trees cover how
//! much of the work.
//!
//! A plan is the node tree rooted at the head, written as the parent array in
//! breadth-first order. Three forms are counted:
//! - canonical: children sorted by their subtree's shape, so isomorphic trees
//!   share a plan. The neck (bone 0) stays the head's first child, because
//!   the physics treats it apart from the head's other bones. The packer can
//!   reorder a body's bones into this form.
//! - packer BFS: breadth-first with children in the packer's bone order.
//! - packer: the packer's parent array as it is (bone order, no reordering).
//!
//! Lane-steps weight each creature by its lane count W (W = 2, 4, 8 by class
//! = max(ceil(nodes / 4), ceil(muscles / 16))) over a full trial; screening
//! is not modelled, so the column shows only the effect of body size.
//!
//! ```text
//! cargo run --release --example plan_count -- <save.evo | dump.bin>...
//! ```
use anyhow::Result;
use evolution_simulator::{
    config::Config,
    evolution::{Bone, Creature, Population},
    storage,
};
use std::collections::HashMap;
use std::path::Path;

/// One body's tree: node count, muscle count and the packer's parent array
/// (entry `r - 1` is the parent of record node `r`; record 0 is the head).
struct Body {
    muscles: usize,
    parent: Vec<u8>,
}

fn body(bones: &[Bone], muscles: usize) -> Option<Body> {
    let n = bones.len() + 1;
    let mut record = vec![usize::MAX; n.max(1)];
    record[0] = 0;
    let mut parent = Vec::with_capacity(bones.len());
    for (j, b) in bones.iter().enumerate() {
        let (a, c) = (b.a as usize, b.b as usize);
        if a >= n || c >= n || record[a] == usize::MAX || record[c] != usize::MAX {
            return None;
        }
        record[c] = j + 1;
        parent.push(record[a] as u8);
    }
    Some(Body { muscles, parent })
}

fn children(parent: &[u8]) -> Vec<Vec<usize>> {
    let mut kids = vec![Vec::new(); parent.len() + 1];
    for (i, &p) in parent.iter().enumerate() {
        kids[p as usize].push(i + 1);
    }
    kids
}

/// Breadth-first parent array, visiting each node's children in the order
/// `kids` lists them.
fn bfs(kids: &[Vec<usize>]) -> Vec<u8> {
    let mut queue = vec![0usize];
    let mut index = vec![0u8; kids.len()];
    let mut out = Vec::with_capacity(kids.len() - 1);
    let mut head = 0;
    while head < queue.len() {
        let v = queue[head];
        head += 1;
        for &c in &kids[v] {
            index[c] = queue.len() as u8;
            queue.push(c);
            out.push(index[v]);
        }
    }
    out
}

/// The canonical form: children ordered by their subtree's code (AHU), the
/// neck first at the head, then breadth-first.
fn canonical(parent: &[u8]) -> Vec<u8> {
    let mut kids = children(parent);
    let n = kids.len();
    let mut code: Vec<Vec<u8>> = vec![Vec::new(); n];
    // Record order is topological (parents first), so reverse order builds
    // every child's code before its parent's.
    for v in (0..n).rev() {
        let mut list = kids[v].clone();
        let neck = if v == 0 && !list.is_empty() {
            Some(list.remove(0))
        } else {
            None
        };
        list.sort_by(|&x, &y| code[x].cmp(&code[y]).then(x.cmp(&y)));
        let mut c = vec![b'('];
        if let Some(k) = neck {
            c.extend_from_slice(&code[k]);
            c.push(b'|');
        }
        for &k in &list {
            c.extend_from_slice(&code[k]);
        }
        c.push(b')');
        code[v] = c;
        if let Some(k) = neck {
            list.insert(0, k);
        }
        kids[v] = list;
    }
    bfs(&kids)
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn lanes(nodes: usize, muscles: usize) -> usize {
    let class = nodes.div_ceil(4).max(muscles.div_ceil(16));
    class.next_power_of_two().clamp(2, 8)
}

fn print_set(name: &str, bodies: &[Body]) {
    let total = bodies.len();
    if total == 0 {
        println!("\n== {name}: no bodies");
        return;
    }
    // (count, lane weight) per canonical plan.
    let mut plans: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();
    let mut packer_bfs: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut packer: HashMap<&[u8], usize> = HashMap::new();
    let mut keyed: HashMap<(Vec<u8>, usize), usize> = HashMap::new();
    let mut lane_total = 0usize;
    let mut node_sum = 0usize;
    for b in bodies {
        let n = b.parent.len() + 1;
        let w = lanes(n, b.muscles);
        lane_total += w;
        node_sum += n;
        let c = canonical(&b.parent);
        let e = plans.entry(c.clone()).or_default();
        e.0 += 1;
        e.1 += w;
        *keyed.entry((c, w)).or_default() += 1;
        *packer_bfs.entry(bfs(&children(&b.parent))).or_default() += 1;
        *packer.entry(&b.parent).or_default() += 1;
    }
    let mut ranked: Vec<(Vec<u8>, usize, usize)> =
        plans.into_iter().map(|(k, (c, w))| (k, c, w)).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top = |counts: &mut Vec<usize>, k: usize| -> f64 {
        counts.iter().take(k).sum::<usize>() as f64 / total as f64
    };
    let mut bfs_counts: Vec<usize> = packer_bfs.values().copied().collect();
    bfs_counts.sort_unstable_by(|a, b| b.cmp(a));
    let mut raw_counts: Vec<usize> = packer.values().copied().collect();
    raw_counts.sort_unstable_by(|a, b| b.cmp(a));
    let mut canon_counts: Vec<usize> = ranked.iter().map(|r| r.1).collect();
    let mut by_lanes: Vec<usize> = ranked.iter().map(|r| r.2).collect();
    by_lanes.sort_unstable_by(|a, b| b.cmp(a));

    println!("\n== {name}");
    println!(
        "{total} bodies, mean {:.2} nodes; distinct trees: {} canonical, {} packer BFS, {} packer order; {} (tree, W) kernel keys",
        node_sum as f64 / total as f64,
        ranked.len(),
        bfs_counts.len(),
        raw_counts.len(),
        keyed.len()
    );
    println!("top k | creatures canonical | lane-steps canonical | creatures packer BFS | creatures packer order");
    let mut canon_lane_cum = 0usize;
    let mut next = 0usize;
    for k in [1, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000] {
        while next < k.min(ranked.len()) {
            canon_lane_cum += ranked[next].2;
            next += 1;
        }
        println!(
            "{k:>5} | {:>6.1}% | {:>6.1}% | {:>6.1}% | {:>6.1}%",
            100.0 * top(&mut canon_counts, k),
            100.0 * canon_lane_cum as f64 / lane_total as f64,
            100.0 * top(&mut bfs_counts, k),
            100.0 * top(&mut raw_counts, k),
        );
    }
    let reach = |counts: &[usize], share: f64| -> usize {
        let mut cum = 0;
        for (i, &c) in counts.iter().enumerate() {
            cum += c;
            if cum as f64 >= share * total as f64 {
                return i + 1;
            }
        }
        counts.len()
    };
    let lane_reach = |share: f64| -> usize {
        let mut cum = 0;
        for (i, r) in ranked.iter().enumerate() {
            cum += r.2;
            if cum as f64 >= share * lane_total as f64 {
                return i + 1;
            }
        }
        ranked.len()
    };
    for share in [0.5, 0.7, 0.85, 0.9, 0.95, 0.99] {
        println!(
            "trees for {:>2.0}%: {} canonical by creatures ({} ranked by lane-steps), {} packer BFS",
            share * 100.0,
            reach(&canon_counts, share),
            lane_reach(share),
            reach(&bfs_counts, share)
        );
    }
    let singles = canon_counts.iter().filter(|&&c| c == 1).count();
    println!("canonical trees held by one body: {singles}");
    let mut by_nodes: std::collections::BTreeMap<usize, (usize, usize)> = Default::default();
    for r in &ranked {
        let e = by_nodes.entry(r.0.len() + 1).or_default();
        e.0 += r.1;
        e.1 += 1;
    }
    let line: Vec<String> = by_nodes
        .iter()
        .map(|(n, (c, d))| format!("{n}:{:.1}%/{d}", 100.0 * *c as f64 / total as f64))
        .collect();
    println!("by node count (share of bodies / distinct trees): {}", line.join(" "));
    println!("rank | bodies | share | cumulative | nodes | hash | canonical BFS parent array");
    let mut cum = 0;
    for (i, (key, count, _)) in ranked.iter().take(20).enumerate() {
        cum += count;
        let array: Vec<String> = key.iter().map(u8::to_string).collect();
        println!(
            "{:>4} | {count:>7} | {:>5.2}% | {:>6.2}% | {:>2} | {:016x} | [{}]",
            i + 1,
            100.0 * *count as f64 / total as f64,
            100.0 * cum as f64 / total as f64,
            key.len() + 1,
            fnv(key),
            array.join(",")
        );
    }
    let t30 = top(&mut canon_counts, 30);
    let t100 = top(&mut canon_counts, 100);
    println!(
        "gate: top 100 {:.1}% (>= 85% makes per-plan the default: {}), top 30 {:.1}% (>= 70%: {}; < 50% rejects per-plan: {})",
        100.0 * t100,
        if t100 >= 0.85 { "pass" } else { "fail" },
        100.0 * t30,
        if t30 >= 0.7 { "pass" } else { "fail" },
        if t30 < 0.5 { "yes" } else { "no" }
    );
}

fn from_creatures<'a>(creatures: impl Iterator<Item = &'a Creature>, bad: &mut usize) -> Vec<Body> {
    creatures
        .filter_map(|c| {
            let b = body(&c.bones, c.muscles.len());
            *bad += usize::from(b.is_none());
            b
        })
        .collect()
}

fn from_population(p: &Population, range: std::ops::Range<usize>, bad: &mut usize) -> Vec<Body> {
    p.genomes[range]
        .iter()
        .filter_map(|g| {
            let b = body(&p.bones[g.bone_start..g.bone_start + g.bone_count], g.muscle_count);
            *bad += usize::from(b.is_none());
            b
        })
        .collect()
}

fn main() -> Result<()> {
    for path in std::env::args().skip(1) {
        let mut bad = 0;
        println!("\n######## {path}");
        if path.ends_with(".bin") {
            type Dump = (Config, Population, Vec<(Creature, Config, f32)>);
            let (_, population, elites): Dump = bincode::deserialize(&std::fs::read(&path)?)?;
            let n = population.genomes.len();
            print_set("dump population", &from_population(&population, 0..n, &mut bad));
            print_set(
                "dump elites",
                &from_creatures(elites.iter().map(|e| &e.0), &mut bad),
            );
        } else {
            let header = storage::peek(Path::new(&path))?;
            let e = storage::load_any_version(Path::new(&path))?;
            println!(
                "save version {}, generation {}, population {}",
                header.qd_version, header.generation, header.population
            );
            let islands = e.islands.len() / 2;
            let island_elites: Vec<&Creature> = e.islands[..islands]
                .iter()
                .flat_map(|a| a.entries.iter().map(|x| &x.creature))
                .collect();
            let nursery_elites: Vec<&Creature> = e.islands[islands..]
                .iter()
                .flat_map(|a| a.entries.iter().map(|x| &x.creature))
                .collect();
            print_set(
                &format!("island archives ({islands} islands incl. the hub)"),
                &from_creatures(island_elites.iter().copied(), &mut bad),
            );
            print_set(
                "nursery archives",
                &from_creatures(nursery_elites.iter().copied(), &mut bad),
            );
            print_set(
                "global archive",
                &from_creatures(e.archive.entries.iter().map(|x| &x.creature), &mut bad),
            );
            if !e.reseed.is_empty() {
                // Elites queued for a new world: a save made right after a
                // world change holds its elites here.
                print_set(
                    "reseed queue (elites waiting for the new world)",
                    &from_creatures(e.reseed.iter(), &mut bad),
                );
            }
            let block = &e.blocks[0];
            print_set(
                &format!("bred ring block 0 (slots {}..{})", block.first, block.first + block.len()),
                &from_population(&block.population, 0..block.len(), &mut bad),
            );
            let mut ring = Vec::new();
            for b in &e.blocks {
                ring.extend(from_population(&b.population, 0..b.len(), &mut bad));
            }
            print_set(&format!("bred ring, all {} blocks", e.blocks.len()), &ring);
        }
        if bad > 0 {
            println!("bodies skipped (bones not in canonical order): {bad}");
        }
    }
    Ok(())
}
