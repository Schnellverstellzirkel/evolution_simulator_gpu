//! How regular the best bodies of a save are: what share of the best elites
//! of the global archive have limbs in pairs (two leaf limbs of the same
//! number of bones and nearly the same lengths hanging from one node), pairs
//! that are mirror images (their first bones lean to opposite sides), and
//! segments (a trunk with leaf limbs on two or more of its nodes in a row),
//! and the mean number of leaf limbs, nodes and muscles. A body that evolves
//! from regular parts is what repeating and mirroring operators aim at.
//!
//! Usage: cargo run --release --example body_regularity -- <save> [elites]
use anyhow::Context;
use evolution_simulator::{evolution::Creature, storage};

struct Shape {
    leaf_limbs: usize,
    paired: bool,
    mirrored: bool,
    segments: usize,
}

fn shape(c: &Creature) -> Shape {
    let n = c.nodes.len();
    let mut children = vec![Vec::new(); n];
    let mut parent_bone = vec![None; n];
    for (i, b) in c.bones.iter().enumerate() {
        children[b.a as usize].push(i);
        parent_bone[b.b as usize] = Some(i);
    }
    // Leaf limbs: from each leaf node (not the head) up to the node where the
    // body branches, as chains of bones from the top down.
    let mut limbs: Vec<(usize, Vec<usize>)> = Vec::new();
    for leaf in 1..n {
        if !children[leaf].is_empty() {
            continue;
        }
        let mut chain = Vec::new();
        let mut node = leaf;
        while let Some(bone) = parent_bone[node] {
            let top = c.bones[bone].a as usize;
            chain.push(bone);
            if top == 0 || children[top].len() != 1 || parent_bone[top].is_none() {
                break;
            }
            node = top;
        }
        chain.reverse();
        let top = c.bones[chain[0]].a as usize;
        if top != 0 {
            limbs.push((top, chain));
        }
    }
    let lengths =
        |chain: &[usize]| -> Vec<f32> { chain.iter().map(|&b| c.bones[b].rest_length).collect() };
    let (mut paired, mut mirrored) = (false, false);
    for (i, (top, x)) in limbs.iter().enumerate() {
        for (other, y) in limbs.iter().skip(i + 1) {
            if top != other || x.len() != y.len() {
                continue;
            }
            let (lx, ly) = (lengths(x), lengths(y));
            if lx
                .iter()
                .zip(&ly)
                .all(|(a, b)| a.max(*b) <= 1.1 * a.min(*b))
            {
                paired = true;
                let lean = |chain: &[usize]| {
                    let b = c.bones[chain[0]];
                    c.nodes[b.b as usize].x - c.nodes[b.a as usize].x
                };
                mirrored |= lean(x) * lean(y) < 0.0;
            }
        }
    }
    // Segments: the longest run of trunk nodes, each carrying a leaf limb, on
    // one path down the body.
    let carries: Vec<bool> = (0..n)
        .map(|node| limbs.iter().any(|(top, _)| *top == node))
        .collect();
    let mut best = 0;
    let mut run = vec![0usize; n];
    for (node, &has) in carries.iter().enumerate() {
        // Bones are in parent-first order, so a parent's run is known.
        let up = parent_bone[node].map_or(0, |b| run[c.bones[b].a as usize]);
        run[node] = if has { up + 1 } else { 0 };
        best = best.max(run[node]);
    }
    Shape {
        leaf_limbs: limbs.len(),
        paired,
        mirrored,
        segments: best,
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .context("usage: body_regularity <save> [elites]")?;
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(300);
    let experiment = storage::load_any_version(std::path::Path::new(path))?;
    let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    let shapes: Vec<(Shape, usize, usize)> = elites
        .iter()
        .map(|e| {
            (
                shape(&e.creature),
                e.creature.nodes.len(),
                e.creature.muscles.len(),
            )
        })
        .collect();
    let n = shapes.len().max(1) as f64;
    let share = |f: &dyn Fn(&Shape) -> bool| {
        100.0 * shapes.iter().filter(|(s, _, _)| f(s)).count() as f64 / n
    };
    let mean = |f: &dyn Fn(&(Shape, usize, usize)) -> f64| shapes.iter().map(f).sum::<f64>() / n;
    println!(
        "{path}: best {} elites: limbs in pairs {:.1}%, mirrored pairs {:.1}%, two or more segments {:.1}%, leaf limbs {:.2}, nodes {:.2}, muscles {:.2}",
        shapes.len(),
        share(&|s| s.paired),
        share(&|s| s.mirrored),
        share(&|s| s.segments >= 2),
        mean(&|t| t.0.leaf_limbs as f64),
        mean(&|t| t.1 as f64),
        mean(&|t| t.2 as f64),
    );
    Ok(())
}
