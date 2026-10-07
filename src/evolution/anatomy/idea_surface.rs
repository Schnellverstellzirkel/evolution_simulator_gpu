//! Idea operators for what the body touches the ground with and where its
//! mass sits: node grip and size, and organ ballast.
//!
//! A node's size (`diameter`) sets its mass, and its grip (`friction`) is the
//! most direct way to tell a foot that pushes from a foot that slides. Every
//! operator here is a whole change on its own, so its child gets no parameter
//! noise, and the operators of this file share one pick slot (`GAIT_FILES` in
//! `mod.rs`).
//!
//! The sources are Hirose (1993, snakes move on ground that grips more
//! sideways than along the body, so a front that slides and a back that grips
//! is a ratchet), Alexander (2003, light distal limbs swing cheaply), Herr and
//! Popovic (2008, mass moved against the gait balances angular momentum), and
//! Sims (1994, evolved block sizes).
use super::ideas::{body_nodes, bulk, coin, drive, grip, inner_nodes, leaf_nodes, set};
use super::limbs::pick;
use super::rhythm::{foot, matching_limbs};
use super::{BoneIds, Context, Operator, child_bones, muscles_on};
use crate::config::Config;
use crate::evolution::{Creature, MAX_ORGAN_MASS, MIN_ORGAN_MASS, Rng};

/// This file's operators, by name. Add each new one here. The pick slot of
/// this file chooses by position in this list, so the order decides what a
/// fixed seed picks.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("claw_feet", claw_feet),
    ("ski_feet", ski_feet),
    ("ratchet_pair", ratchet_pair),
    ("heavy_feet", heavy_feet),
    ("light_feet", light_feet),
    ("heavy_trunk", heavy_trunk),
    ("light_trunk", light_trunk),
    ("friction_gradient_trunk", friction_gradient_trunk),
    ("bulk_by_height", bulk_by_height),
    ("swap_node_surfaces", swap_node_surfaces),
    ("equalize_nodes", equalize_nodes),
    ("polarize_nodes", polarize_nodes),
    ("redraw_one_node", redraw_one_node),
    ("knee_slip", knee_slip),
    ("dead_weight_diet", dead_weight_diet),
    ("muscle_hub_bulk", muscle_hub_bulk),
    ("organ_diet_or_feast", organ_diet_or_feast),
];

/// Every foot (a node with no bone below it) grips as hard as it can and gets
/// 15% narrower, so lighter: a claw.
fn claw_feet(c: &mut Creature, cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let mut changed = false;
    for n in leaf_nodes(c) {
        let node = &mut c.nodes[n];
        set(&mut node.friction, cfg.max_friction, &mut changed);
        let d = (node.diameter * 0.85).max(cfg.min_size);
        set(&mut node.diameter, d, &mut changed);
    }
    changed
}

/// Every foot slides as easily as it can and gets 15% wider, like a ski.
fn ski_feet(c: &mut Creature, cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let mut changed = false;
    for n in leaf_nodes(c) {
        let node = &mut c.nodes[n];
        set(&mut node.friction, cfg.min_friction, &mut changed);
        let d = (node.diameter * 1.15).min(cfg.max_size);
        set(&mut node.diameter, d, &mut changed);
    }
    changed
}

/// In a pair of matching limbs (`matching_limbs`) one foot grips and the other
/// slides, and a coin picks which. The pair works as a ratchet (Hirose 1993):
/// the gripping limb pushes while the sliding limb is dragged forward.
fn ratchet_pair(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let pairs = matching_limbs(c);
    if pairs.is_empty() {
        return false;
    }
    let (x, y) = pairs.get(rng.index(pairs.len()));
    let (grips, slides) = if coin(rng) { (x, y) } else { (y, x) };
    let (g, s) = (foot(c, grips), foot(c, slides));
    let mut changed = false;
    set(&mut c.nodes[g].friction, cfg.max_friction, &mut changed);
    set(&mut c.nodes[s].friction, cfg.min_friction, &mut changed);
    changed
}

/// Feet get 20 to 35% wider, which makes them heavier and harder to push
/// around.
fn heavy_feet(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = rng.range(1.2, 1.35);
    scale_nodes(c, cfg, &leaf_nodes(c), by)
}

/// Feet get 20 to 30% narrower and lighter. Light distal parts swing at less
/// cost (Alexander 2003).
fn light_feet(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = rng.range(0.7, 0.8);
    scale_nodes(c, cfg, &leaf_nodes(c), by)
}

/// The inner nodes (the trunk and the joints of limbs) get 15 to 30% wider.
fn heavy_trunk(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = rng.range(1.15, 1.3);
    scale_nodes(c, cfg, &inner_nodes(c), by)
}

/// The inner nodes get 15 to 25% narrower.
fn light_trunk(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = rng.range(0.75, 0.85);
    scale_nodes(c, cfg, &inner_nodes(c), by)
}

/// Multiplies the diameter of each of `nodes` by `by`, within `cfg.min_size`
/// and `cfg.max_size`. Returns whether any diameter changed.
fn scale_nodes(c: &mut Creature, cfg: &Config, nodes: &[usize], by: f32) -> bool {
    let mut changed = false;
    for &n in nodes {
        let d = (c.nodes[n].diameter * by).clamp(cfg.min_size, cfg.max_size);
        set(&mut c.nodes[n].diameter, d, &mut changed);
    }
    changed
}

/// Grip falls (or rises) in a line along x from the rearmost node to the
/// foremost, over the whole grip range and every node but the head. A coin
/// picks the direction. One end of the body drags and the other end holds.
fn friction_gradient_trunk(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let nodes = body_nodes(c);
    if nodes.len() < 3 {
        return false;
    }
    let (low, high) = nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), &n| {
        (l.min(c.nodes[n].x), h.max(c.nodes[n].x))
    });
    if high - low < 0.05 {
        return false;
    }
    let (from, to) = if coin(rng) { (0.0, 1.0) } else { (1.0, 0.0) };
    let mut changed = false;
    for &n in &nodes {
        let t = (c.nodes[n].x - low) / (high - low);
        let g = grip(cfg, from + (to - from) * t);
        set(&mut c.nodes[n].friction, g, &mut changed);
    }
    changed
}

/// Each node but the head moves 30 to 70% of the way to a size set by its
/// height: the lowest node toward the largest size and the highest toward the
/// smallest. This keeps the weight low (a low center of mass resists tipping).
fn bulk_by_height(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let nodes = body_nodes(c);
    if nodes.len() < 3 {
        return false;
    }
    let (low, high) = nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), &n| {
        (l.min(c.nodes[n].y), h.max(c.nodes[n].y))
    });
    if high - low < 0.05 {
        return false;
    }
    let strength = rng.range(0.3, 0.7);
    let mut changed = false;
    for &n in &nodes {
        let t = (c.nodes[n].y - low) / (high - low);
        let want = bulk(cfg, 1.0 - t);
        let d = c.nodes[n].diameter + strength * (want - c.nodes[n].diameter);
        set(&mut c.nodes[n].diameter, d, &mut changed);
    }
    changed
}

/// Two nodes (never the head) trade their grip and size.
fn swap_node_surfaces(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let nodes = body_nodes(c);
    if nodes.len() < 2 {
        return false;
    }
    let (x, y) = (nodes[rng.index(nodes.len())], nodes[rng.index(nodes.len())]);
    if x == y {
        return false;
    }
    let (a, b) = (c.nodes[x], c.nodes[y]);
    let mut changed = false;
    set(&mut c.nodes[x].friction, b.friction, &mut changed);
    set(&mut c.nodes[y].friction, a.friction, &mut changed);
    set(&mut c.nodes[x].diameter, b.diameter, &mut changed);
    set(&mut c.nodes[y].diameter, a.diameter, &mut changed);
    changed
}

/// Size and grip of every node move 40% toward the mean of the body.
fn equalize_nodes(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    spread_nodes(c, 0.6)
}

/// Size and grip of every node get 40% farther from the mean of the body,
/// which makes heavy nodes heavier and slippery ones more slippery.
fn polarize_nodes(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    spread_nodes(c, 1.4)
}

/// Scales how far each node but the head is from the mean size and the mean
/// grip of those nodes by `factor`. `repair` clamps what leaves the limits.
fn spread_nodes(c: &mut Creature, factor: f32) -> bool {
    let nodes = body_nodes(c);
    if nodes.len() < 2 {
        return false;
    }
    let n = nodes.len() as f32;
    let mean_d = nodes.iter().map(|&i| c.nodes[i].diameter).sum::<f32>() / n;
    let mean_f = nodes.iter().map(|&i| c.nodes[i].friction).sum::<f32>() / n;
    let mut changed = false;
    for &i in &nodes {
        let d = mean_d + factor * (c.nodes[i].diameter - mean_d);
        let f = mean_f + factor * (c.nodes[i].friction - mean_f);
        set(&mut c.nodes[i].diameter, d, &mut changed);
        set(&mut c.nodes[i].friction, f, &mut changed);
    }
    changed
}

/// One node but the head gets a new random size and grip.
fn redraw_one_node(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let nodes = body_nodes(c);
    let Some(n) = pick(&nodes, rng) else {
        return false;
    };
    let (d, f) = (bulk(cfg, rng.unit()), grip(cfg, rng.unit()));
    let mut changed = false;
    set(&mut c.nodes[n].diameter, d, &mut changed);
    set(&mut c.nodes[n].friction, f, &mut changed);
    changed
}

/// Every node with exactly one bone below it, such as the knee of a limb, gets
/// the lowest grip, so a limb that kneels slides on its knee instead of
/// sticking.
fn knee_slip(c: &mut Creature, cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let children = child_bones(c);
    let mut changed = false;
    for (n, below) in children.iter().enumerate().take(c.nodes.len()).skip(1) {
        // A knee has one bone below it and one above.
        if below.len() == 1 {
            set(&mut c.nodes[n].friction, cfg.min_friction, &mut changed);
        }
    }
    changed
}

/// Nodes that no muscle or only weak muscles reach (less than a third of the
/// drive of the strongest muscle) shrink to the smallest size, because mass
/// that no strong muscle drives only adds load. A muscle reaches a node when it
/// has an end on a bone at that node. The head is left alone.
fn dead_weight_diet(c: &mut Creature, cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let best = c.muscles.iter().map(drive).fold(0.0f32, f32::max);
    if best <= 0.0 {
        return false;
    }
    let mut changed = false;
    for n in 1..c.nodes.len() {
        let touching: BoneIds = (0..c.bones.len())
            .filter(|&b| c.bones[b].a as usize == n || c.bones[b].b as usize == n)
            .collect();
        let reach = muscles_on(c, &touching, false)
            .iter()
            .map(|&m| drive(&c.muscles[m]))
            .fold(0.0f32, f32::max);
        if reach < best / 3.0 {
            set(&mut c.nodes[n].diameter, cfg.min_size, &mut changed);
        }
    }
    changed
}

/// The node with the most muscles on its bones grows to the largest size, so
/// the mass sits where the muscles pull. On a tie the node with the highest
/// index wins.
fn muscle_hub_bulk(c: &mut Creature, cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let load = |n: usize| -> usize {
        let touching: BoneIds = (0..c.bones.len())
            .filter(|&b| c.bones[b].a as usize == n || c.bones[b].b as usize == n)
            .collect();
        muscles_on(c, &touching, false).len()
    };
    let Some(hub) = (1..c.nodes.len()).max_by_key(|&n| load(n)) else {
        return false;
    };
    let mut changed = false;
    set(&mut c.nodes[hub].diameter, cfg.max_size, &mut changed);
    changed
}

/// Every organ gets half as heavy, or 1.6 times as heavy, within the organ mass
/// limits.
fn organ_diet_or_feast(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = if coin(rng) { 0.5 } else { 1.6 };
    let mut changed = false;
    for bone in &mut c.bones {
        if bone.organ_mass > 0.0 {
            let mass = (bone.organ_mass * by).clamp(MIN_ORGAN_MASS, MAX_ORGAN_MASS);
            set(&mut bone.organ_mass, mass, &mut changed);
        }
    }
    changed
}
