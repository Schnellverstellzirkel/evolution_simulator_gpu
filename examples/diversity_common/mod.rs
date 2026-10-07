//! This file measures how varied an archive is. It counts the distinct body
//! plans and the clades of the elites, and it tells how much of body shape
//! space they cover. The examples `archive_diversity`, `archive_bench` and
//! `search_ab` include it with `#[path]` and print its lines.
// Each example that includes this file uses only some of its items.
#![allow(dead_code)]
use evolution_simulator::{
    config::Config,
    environment::EFFECTS,
    qd::{self, Elite, QdArchive},
    storage::Experiment,
};
use std::collections::{HashMap, HashSet};

/// The world as a line of text. It lists the effects that are off their calm
/// level, each with its level name, or says `calm`. The autochange step
/// follows.
pub fn world_line(cfg: &Config) -> String {
    let on: Vec<String> = EFFECTS
        .iter()
        .filter(|e| e.level(cfg) != e.calm)
        .map(|e| format!("{} {}", e.name, e.levels[e.level(cfg)]))
        .collect();
    format!(
        "{} (autochange step {})",
        if on.is_empty() {
            "calm".to_owned()
        } else {
            on.join(", ")
        },
        cfg.autochange_step
    )
}

/// The effects whose level differs between two worlds, as a line of text. An
/// entry reads like `Mud Dry to Damp`. The line says `no effect changed` when
/// no level differs.
pub fn world_difference(before: &Config, after: &Config) -> String {
    let changes: Vec<String> = EFFECTS
        .iter()
        .filter(|e| e.level(before) != e.level(after))
        .map(|e| {
            format!(
                "{} {} to {}",
                e.name,
                e.levels[e.level(before)],
                e.levels[e.level(after)]
            )
        })
        .collect();
    if changes.is_empty() {
        "no effect changed".to_owned()
    } else {
        changes.join(", ")
    }
}

/// What one archive holds, as `measure` counts it.
#[derive(Clone, Debug, Default)]
pub struct Diversity {
    /// Behavior elites measured, one per filled cell. The morphology reserve
    /// is left out.
    pub cells: usize,
    /// Distinct ways of moving: cells ignoring the body classes.
    pub moves: usize,
    /// Distinct body plans (a node count with its bone and muscle wiring).
    pub plans: usize,
    /// Distinct body types: node and muscle counts, as the History tab counts them.
    pub types: usize,
    /// The number of clades. A clade is the elites that share one oldest
    /// recorded ancestor.
    pub clades: usize,
    /// The effective number of clades (see `effective`).
    pub clades_effective: f64,
    /// The share of the elites that the largest clade holds.
    pub largest_clade: f64,
    /// Occupied classes of a fixed reference grid of body shapes (node
    /// count, aspect and total bone length).
    pub kinds: usize,
    /// The effective number of the occupied classes of that grid.
    pub kinds_effective: f64,
    /// Mean distance from an elite to its fifth nearest elite in body shape.
    /// It comes from a sample of at most 1,000 elites and is 0 for fewer
    /// than 6.
    pub spread: f64,
}

/// The number of classes of the reference grid: 5 node classes, 4 aspect
/// classes and 4 classes of total bone length. `reference_kind` gives the class
/// of a body.
pub const REFERENCE_KINDS: usize = 5 * 4 * 4;

/// The class of `value`: how many of the ascending `edges` it reaches.
fn class(value: f32, edges: &[f32]) -> usize {
    edges.iter().filter(|&&edge| value >= edge).count()
}

/// An elite's body measures: node count, aspect of its start pose, total
/// bone length (m) and limbs (leaves of the bone tree).
pub fn body(elite: &Elite) -> (usize, f32, f32, usize) {
    let c = &elite.creature.unpack();
    let length: f32 = c.bones.iter().map(|b| b.rest_length).sum();
    let mut degree = vec![0usize; c.nodes.len()];
    for b in c.bones.iter() {
        degree[b.a as usize] += 1;
        degree[b.b as usize] += 1;
    }
    let leaves = degree.iter().filter(|&&d| d == 1).count();
    (c.nodes.len(), elite.descriptor.aspect_ratio, length, leaves)
}

/// The class of an elite's body in the reference grid, an index below
/// `REFERENCE_KINDS`. Node classes split at 8, 10, 13 and 17 nodes, aspect
/// classes at 1, 2 and 4, and bone length classes at 1.5, 3 and 6 m.
pub fn reference_kind(elite: &Elite) -> usize {
    let (nodes, aspect, length, _) = body(elite);
    let n = class(nodes as f32, &[8.0, 10.0, 13.0, 17.0]);
    let a = class(aspect, &[1.0, 2.0, 4.0]);
    let l = class(length, &[1.5, 3.0, 6.0]);
    (n * 4 + a) * 4 + l
}

/// The effective number of classes. `counts` holds the members of each class
/// and `total` is their sum. The result is `exp` of the entropy of the class
/// shares. It equals the class count when the classes are the same size, and
/// it is smaller when a few classes hold most members. It is 0 when `total` is
/// 0.
fn effective(counts: impl Iterator<Item = usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let entropy: f64 = counts
        .map(|c| {
            let p = c as f64 / total as f64;
            -p * p.ln()
        })
        .sum();
    entropy.exp()
}

/// The root (oldest recorded ancestor) of each of `ids`. The map also holds the
/// ancestors met on the way. A walk up the lineage stops at an id whose root is
/// already known, so each chain is walked once.
pub fn roots(experiment: &Experiment, ids: &[u64]) -> HashMap<u64, u64> {
    let mut known: HashMap<u64, u64> = HashMap::new();
    for &id in ids {
        let mut chain = vec![id];
        let mut current = id;
        let root = loop {
            if let Some(&root) = known.get(&current) {
                break root;
            }
            match experiment.lineage.get(&current).and_then(|a| a.parent) {
                Some(parent) if experiment.lineage.contains_key(&parent) => {
                    chain.push(parent);
                    current = parent;
                }
                _ => break current,
            }
        };
        for step in chain {
            known.insert(step, root);
        }
    }
    known
}

/// Mean distance from an elite to its fifth nearest elite in a body shape
/// space. The axes are log node count, log length, log aspect and limbs, each
/// scaled to about 0 to 1. It is 0 for fewer than 6 elites. The caller keeps
/// the set small, because every pair of elites is compared.
fn spread(elites: &[&Elite]) -> f64 {
    if elites.len() < 6 {
        return 0.0;
    }
    let points: Vec<[f32; 4]> = elites
        .iter()
        .map(|e| {
            let (nodes, aspect, length, leaves) = body(e);
            [
                (nodes as f32 / 3.0).ln() / (32.0f32 / 3.0).ln(),
                (length.max(0.1) / 0.3).ln() / (40.0f32 / 0.3).ln(),
                (aspect.max(0.0625) / 0.0625).ln() / 256.0f32.ln(),
                leaves as f32 / 20.0,
            ]
        })
        .collect();
    let mut total = 0.0;
    let mut near = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        near.clear();
        for (j, q) in points.iter().enumerate() {
            if i != j {
                near.push((0..4).map(|k| (p[k] - q[k]).powi(2)).sum::<f32>());
            }
        }
        near.select_nth_unstable_by(4, f32::total_cmp);
        total += near[4].sqrt() as f64;
    }
    total / points.len() as f64
}

/// Which elites of an archive to measure.
#[derive(Clone, Copy)]
pub enum Part {
    /// Every behavior elite.
    All,
    /// The given number of fastest elites.
    Top(usize),
    /// Those within 1% of the best distance.
    NearBest,
}

/// Measures the behavior elites of `archive`, or the part of them that `part`
/// names. The clades come from the lineage in `experiment`. The spread comes
/// from a sample of at most 1,000 elites.
pub fn measure(archive: &QdArchive, experiment: &Experiment, part: Part) -> Diversity {
    let mut elites: Vec<&Elite> = archive
        .entries
        .iter()
        .filter(|e| !qd::is_morphology_niche(&e.niche))
        .collect();
    match part {
        Part::All => {}
        Part::Top(top) => {
            elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            elites.truncate(top);
        }
        Part::NearBest => {
            let best = elites.iter().map(|e| e.fitness).fold(f32::MIN, f32::max);
            elites.retain(|e| e.fitness >= 0.99 * best);
        }
    }
    let mut moves = HashSet::new();
    let mut plans = HashSet::new();
    let mut types = HashSet::new();
    let mut kinds: HashMap<usize, usize> = HashMap::new();
    for e in &elites {
        // A way of moving is the niche without its shape class (axis 2) and
        // its size class (axis 5).
        moves.insert((e.niche.0[0], e.niche.0[1], e.niche.0[3], e.niche.0[4]));
        plans.insert(&e.topology);
        types.insert((e.descriptor.nodes, e.descriptor.muscles));
        *kinds.entry(reference_kind(e)).or_default() += 1;
    }
    let ids: Vec<u64> = elites.iter().map(|e| e.creature.id).collect();
    let root_of = roots(experiment, &ids);
    let mut clades: HashMap<u64, usize> = HashMap::new();
    for id in &ids {
        *clades.entry(root_of[id]).or_default() += 1;
    }
    let n = elites.len();
    Diversity {
        cells: n,
        moves: moves.len(),
        plans: plans.len(),
        types: types.len(),
        clades: clades.len(),
        clades_effective: effective(clades.values().copied(), n),
        largest_clade: clades.values().copied().max().unwrap_or(0) as f64 / n.max(1) as f64,
        kinds: kinds.len(),
        kinds_effective: effective(kinds.values().copied(), n),
        spread: {
            // At most 1,000 elites from every archive, chosen by a hash of
            // their ids, so spreads compare across archive sizes.
            let mut sample = elites.clone();
            sample.sort_by_key(|e| {
                let mut h = e.creature.id.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                h ^= h >> 29;
                h.wrapping_mul(0xbf58_476d_1ce4_e5b9)
            });
            sample.truncate(1000);
            spread(&sample)
        },
    }
}

impl Diversity {
    /// All the measures on one line of text, as the examples print them.
    pub fn line(&self) -> String {
        format!(
            "cells {} moves {} plans {} types {} clades {} (eff {:.1}, largest {:.2}) kinds {} of {} (eff {:.1}) spread {:.3}",
            self.cells,
            self.moves,
            self.plans,
            self.types,
            self.clades,
            self.clades_effective,
            self.largest_clade,
            self.kinds,
            REFERENCE_KINDS,
            self.kinds_effective,
            self.spread,
        )
    }
}
