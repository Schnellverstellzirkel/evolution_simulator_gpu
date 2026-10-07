//! How regular the best bodies of a save are. The tool takes the best elites
//! of the global archive and prints what share of them have limbs in pairs,
//! pairs that are mirror images, and two or more segments. It also prints the
//! mean number of leaf limbs, nodes and muscles. The repeating and mirroring
//! mutation operators aim at bodies built from regular parts.
//!
//! A leaf limb is the chain of bones from a leaf node up to the node where the
//! body branches. It counts only when that node is not the head. Two leaf
//! limbs are a pair when they hang from one node, have the same number of
//! bones, and the longer bone of each matching pair is at most 10% longer than
//! the shorter. A pair is a mirror image when the first bones of its limbs
//! lean to opposite sides. The segment count of a body is the length of its
//! longest chain of parent and child nodes that all carry a leaf limb.
//!
//! Usage: `cargo run --release --example body_regularity -- <save> [elites]`
//!
//! `elites` is how many of the best elites to measure, 300 by default. A save
//! of an older version loads too, with the scores it measured then. The tool
//! needs no GPU.
use anyhow::Context;
use evolution_simulator::{evolution::Creature, storage};

/// How regular one body is.
struct Shape {
    /// The number of leaf limbs.
    leaf_limbs: usize,
    /// Two leaf limbs form a pair.
    paired: bool,
    /// Some pair has first bones that lean to opposite sides.
    mirrored: bool,
    /// The length of the longest chain of parent and child nodes that all
    /// carry a leaf limb.
    segments: usize,
}

/// Measures one body: its leaf limbs, whether they pair up and mirror, and its
/// segments.
fn shape(c: &Creature) -> Shape {
    let n = c.nodes.len();
    // For each node: the bones that start at it, and the bone that ends at it.
    let mut children = vec![Vec::new(); n];
    let mut parent_bone = vec![None; n];
    for (i, b) in c.bones.iter().enumerate() {
        children[b.a as usize].push(i);
        parent_bone[b.b as usize] = Some(i);
    }
    // Leaf limbs: for each leaf node except the head, the chain of bones from
    // the node where the body branches down to the leaf. A chain that starts at
    // the head is not a limb. Each limb is kept with its top node.
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
    // Pairs: two limbs from one node with the same number of bones, each bone
    // within 10% of the one at the same place in the other limb.
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
                // The side a limb leans to: the x offset of the end of its
                // first bone from its start in the start pose.
                let lean = |chain: &[usize]| {
                    let b = c.bones[chain[0]];
                    c.nodes[b.b as usize].x - c.nodes[b.a as usize].x
                };
                mirrored |= lean(x) * lean(y) < 0.0;
            }
        }
    }
    // Segments: the longest chain of parent and child nodes that all carry a
    // leaf limb. `run[node]` is the length of the chain that ends at `node`.
    let carries: Vec<bool> = (0..n)
        .map(|node| limbs.iter().any(|(top, _)| *top == node))
        .collect();
    let mut best = 0;
    let mut run = vec![0usize; n];
    for (node, &has) in carries.iter().enumerate() {
        // A node continues the chain of its parent. Nodes come in index order,
        // so the parent's chain is known only when the parent has the lower
        // index.
        // TODO: a parent with a higher index, as `split_crowded_joint` makes,
        // reads 0 here, so a chain can be cut short.
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
    // Each elite's shape, node count and muscle count.
    let shapes: Vec<(Shape, usize, usize)> = elites
        .iter()
        .map(|e| {
            (
                shape(&e.creature.unpack()),
                e.creature.node_count(),
                e.creature.muscle_count(),
            )
        })
        .collect();
    let n = shapes.len().max(1) as f64;
    // The percentage of the elites whose shape `f` accepts.
    let share = |f: &dyn Fn(&Shape) -> bool| {
        100.0 * shapes.iter().filter(|(s, _, _)| f(s)).count() as f64 / n
    };
    // The mean of `f` over the elites.
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
