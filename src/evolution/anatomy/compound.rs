//! Compound operators. Each one changes several parts of the body together
//! and keeps them consistent with each other, so a child is a larger step
//! than one edit and still has a chance of keeping its parent's gait.
//! `anatomy/mod.rs` lists the operators here in `BASE_OPERATORS` and marks
//! them in `COMPOUND`. `legs.rs` and several `gait_*.rs` files use the helpers
//! that this file defines, such as `close_ring` and `shed_tips`.
//!
//! The sources are Sims (1994, limb pairs added together with mirrored
//! timing), Hornby and Pollack (2001, repeated parts with a gradient),
//! Cheney et al. (2013, regular and symmetric bodies), Lessin, Fussell and
//! Miikkulainen (2013, whole modules exchanged) and Beyer and Schwefel (2002,
//! correlated mutation: genes that act together move together).
//!
//! Operators here that add or remove bones close the motor ring themselves
//! (`close_ring`): the body goes into canonical order and every pair of
//! consecutive bones without a muscle gets a passive spring. Left to `repair`
//! the ring would get random active muscles, and a later `repair` would add
//! more for every pair the canonical order moves.
//!
//! The operators that add nodes also remove the idlest limb tips (`shed_tips`),
//! so a body that takes such a move grows by one node at most. Without that,
//! 16 seeds of 30 generations grew the bodies in the game's ring (`ring::Ring`)
//! to 11.0 nodes against 8.2 (QD x0.84, within the noise of one arm). A bigger
//! body costs the GPU more per creature. On the lane-group kernel of that run,
//! a body past 8 or 16 nodes took twice the lanes of one below.
//!
//! A child that one of these operators made gets no parameter noise after it
//! (`evolution::offspring`), because the move is the whole change. In
//! `examples/mutation_audit.rs` the noise cut the share of the 13 operators'
//! children that enter the archive from 9.6% to 4.5% on one save and from
//! 8.3% to 1.3% on another.
use super::controller::active_on;
use super::extra::{PASSIVE_STIFFNESS, drive};
use super::junctions::{
    add, add_node, keep_strokes, pos, scale, shift_branch, spans, sub, turn_branch,
};
use super::limbs::{clamped, limb_roots, pick};
use super::muscles::{actuation, ring, shared_node, shift_timing, turn};
use super::rhythm::{
    leaf_limbs, leaf_limbs_at, limbs_front_to_back, matching_limbs, muscle_groups, tip_y,
};
use super::{
    BoneIds, Context, MuscleIds, Operator, branch, branch_nodes, child_bones, copy_branch_limited,
    degree, fit_stroke, is_neck, muscles_on, new_muscle, parent_bones, remove_parts, room, span,
};
use crate::config::Config;
use crate::evolution::{
    Bone, Bounded, Creature, MAX_MUSCLES, MAX_NODES, Muscle, Muscles, Rng, canonicalize_bone_order,
    max_bone_length, max_stroke,
};

/// Puts the bones in canonical order and gives each pair of consecutive
/// bones that has no muscle a passive spring. At the muscle limit the muscles
/// off the ring with the least drive go first, and a pair stays without a
/// spring if no room is left. Returns false when the skeleton is invalid.
/// Bone numbers change, so this is the last step of an operator.
pub(super) fn close_ring(c: &mut Creature, cfg: &Config, rng: &mut Rng) -> bool {
    if !canonicalize_bone_order(c) {
        return false;
    }
    let n = c.bones.len();
    if n < 2 {
        return true;
    }
    let joined = |c: &Creature, a: usize, b: usize| {
        c.muscles.iter().any(|m| {
            let ends = (m.bone_a as usize, m.bone_b as usize);
            ends == (a, b) || ends == (b, a)
        })
    };
    let missing: Bounded<(u8, u8), MAX_NODES> = (0..n)
        .map(|a| (a, (a + 1) % n))
        .filter(|&(a, b)| (n > 2 || a < b) && !joined(c, a, b))
        .map(|(a, b)| (a as u8, b as u8))
        .collect();
    while c.muscles.len() + missing.len() > cfg.max_muscles {
        let weakest = (0..c.muscles.len())
            .filter(|&i| !ring(c, &c.muscles[i]))
            .min_by(|&x, &y| drive(&c.muscles[x]).total_cmp(&drive(&c.muscles[y])));
        let Some(weakest) = weakest else {
            break;
        };
        c.muscles.remove(weakest);
    }
    for &(a, b) in &missing {
        if c.muscles.len() < cfg.max_muscles {
            let mut m = crate::evolution::muscle(a as usize, b as usize, &c.bones, &c.nodes, rng);
            m.short = m.long;
            m.stiffness = PASSIVE_STIFFNESS;
            c.muscles.push(m);
        }
    }
    true
}

/// Removes up to `count` limb tips (a bone that ends in a leaf, not the neck)
/// among the nodes before `first_new`, the node count before an operator
/// added its parts, so those stay. Each time it takes the tip with the least
/// drive of three random ones, with its muscles. It leaves at least three
/// nodes. Returns how many went. Operators that add nodes use it to give back
/// what they added, all or part of it, so a body that takes such a move does
/// not keep growing: bodies that only grow cost the GPU more with every
/// creature.
pub(super) fn shed_tips(c: &mut Creature, first_new: usize, count: usize, rng: &mut Rng) -> usize {
    let (mut first_new, mut shed) = (first_new, 0);
    while shed < count && c.nodes.len() > 3 {
        let mut degrees = [0u8; MAX_NODES];
        for bone in &c.bones {
            degrees[bone.a as usize] += 1;
            degrees[bone.b as usize] += 1;
        }
        let tips: BoneIds = (0..c.bones.len())
            .filter(|&b| {
                let node = c.bones[b].b as usize;
                !is_neck(c, b) && node < first_new && degrees[node] == 1
            })
            .collect();
        if tips.is_empty() {
            break;
        }
        let work = |b: usize| -> f32 {
            muscles_on(c, &[b], false)
                .iter()
                .map(|&i| drive(&c.muscles[i]))
                .sum()
        };
        let idlest = (0..3)
            .map(|_| tips[rng.index(tips.len())])
            .min_by(|&x, &y| work(x).total_cmp(&work(y)))
            .expect("three picks");
        let node = c.bones[idlest].b as usize;
        remove_parts(c, &[idlest], &[node]);
        first_new -= 1;
        shed += 1;
    }
    shed
}

/// The muscle of `group` with the most drive, or `None` for an empty group.
pub(super) fn strongest(c: &Creature, group: &[usize]) -> Option<usize> {
    group
        .iter()
        .copied()
        .max_by(|&x, &y| drive(&c.muscles[x]).total_cmp(&drive(&c.muscles[y])))
}

/// The muscle off `group` with the most drive: the main driver of the gait
/// the group has to work with. `None` when every muscle is in `group`.
pub(super) fn lead_muscle(c: &Creature, group: &[usize]) -> Option<usize> {
    (0..c.muscles.len())
        .filter(|i| !group.contains(i))
        .max_by(|&x, &y| drive(&c.muscles[x]).total_cmp(&drive(&c.muscles[y])))
}

/// The phase of the strongest muscle on `limb`, or `None` when it has none.
pub(super) fn limb_phase(c: &Creature, limb: &[usize]) -> Option<f32> {
    let on = muscles_on(c, limb, false);
    strongest(c, &on).map(|i| c.muscles[i].phase)
}

/// Moves every muscle of `group` (phase and touchdown reset) by one common
/// amount, so that `anchor` lands on phase `target` and the group keeps its
/// own timing. Returns whether anything moved.
fn retime_group(c: &mut Creature, group: &[usize], anchor: usize, target: f32) -> bool {
    let shift = turn(c.muscles[anchor].phase, target);
    if shift.abs() < 1.0e-4 {
        return false;
    }
    shift_group(c, group, shift);
    true
}

/// Moves the phase and the touchdown reset of every muscle of `group` by
/// `shift` cycles.
pub(super) fn shift_group(c: &mut Creature, group: &[usize], shift: f32) {
    for &i in group {
        shift_timing(&mut c.muscles[i], shift);
    }
}

/// Scales `bones`, a branch or a limb's chain, about the joint its first
/// bone hangs from by `factor` (within the bone limits). Returns false and
/// changes nothing when the factor that fits is within 2% of 1. The caller
/// keeps the strokes (`keep_strokes`).
pub(super) fn scale_bones(c: &mut Creature, bones: &[usize], factor: f32) -> bool {
    let (mut low, mut high) = (0.0f32, f32::INFINITY);
    for &b in bones {
        let length = c.bones[b].rest_length;
        low = low.max(0.03 / length);
        high = high.min(max_bone_length() / length);
    }
    let factor = factor.clamp(low, high.max(low));
    if (factor - 1.0).abs() < 0.02 {
        return false;
    }
    let pivot = c.nodes[c.bones[bones[0]].a as usize];
    for n in branch_nodes(c, bones) {
        let node = &mut c.nodes[n];
        [node.x, node.y] = clamped(
            pivot.x + (node.x - pivot.x) * factor,
            pivot.y + (node.y - pivot.y) * factor,
        );
    }
    for &b in bones {
        c.bones[b].rest_length *= factor;
    }
    true
}

/// `scale_bones` for the branch that starts at `root`.
fn scale_limb(c: &mut Creature, root: usize, factor: f32) -> bool {
    scale_bones(c, &branch(c, root), factor)
}

/// Sets joint `j` against one of its stops and leaves it a small flex back
/// from it (as `brace_joint` does). A joint that is already against a stop on
/// that side, or has no range, stays.
fn brace(c: &mut Creature, j: usize, upper: bool, rng: &mut Rng) -> bool {
    let b = c.bones[j];
    let stop = if upper { b.max_angle } else { b.min_angle };
    if b.max_angle - b.min_angle <= 0.05 || stop.abs() < 0.03 {
        return false;
    }
    turn_branch(c, j, stop);
    let flex = rng.range(0.03, 0.2);
    (c.bones[j].min_angle, c.bones[j].max_angle) = if stop > 0.0 {
        (-flex, 0.0)
    } else {
        (0.0, flex)
    };
    true
}

/// Makes every muscle on the bone that ends in `foot` sense that foot's
/// touchdown, with reset phases that keep their order (as `touchdown_package`
/// does).
fn reflex_foot(c: &mut Creature, foot: usize, rng: &mut Rng) -> bool {
    let Some(bone) = parent_bones(c)[foot] else {
        return false;
    };
    let muscles = muscles_on(c, &[bone], false);
    let Some(&first) = muscles.first() else {
        return false;
    };
    let reset = rng.unit();
    let origin = c.muscles[first].phase;
    for &i in &muscles {
        let m = c.muscles[i];
        let (a, b) = (c.bones[m.bone_a as usize], c.bones[m.bone_b as usize]);
        let Some(sensor) = [a.a, a.b, b.a, b.b]
            .iter()
            .position(|&n| n as usize == foot)
        else {
            continue;
        };
        let m = &mut c.muscles[i];
        m.sensor = sensor as u32;
        m.reset = (reset + m.phase - origin).rem_euclid(1.0);
    }
    true
}

/// A random foot (a node with one bone, not the head) among the nodes of
/// `bones`, or `None` when there is none.
fn pick_foot(c: &Creature, bones: &[usize], rng: &mut Rng) -> Option<usize> {
    let feet: BoneIds = branch_nodes(c, bones)
        .into_iter()
        .filter(|&n| n != 0 && degree(c, n) == 1)
        .collect();
    pick(&feet, rng)
}

/// Gives the limbs at the front of the body a longer or shorter reach than
/// the limbs at the back, in a smooth gradient from one end to the other (the
/// longest limb is 1.08 to 1.7 times the shortest). Muscles keep their stroke
/// relative to their span. A gradient is how Hornby and Pollack's repeated
/// parts differ from one another, and it tilts the body without a new bone.
pub(crate) fn limb_length_gradient(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs = limbs_front_to_back(c);
    if limbs.len() < 2 {
        return false;
    }
    let spread = rng.range(1.08f32.ln(), 1.7f32.ln()).exp();
    let sign = if rng.unit() < 0.5 { 1.0 } else { -1.0 };
    let last = (limbs.len() - 1) as f32;
    let before = spans(c);
    let mut changed = false;
    for (rank, limb) in limbs.iter().enumerate() {
        let along = (rank as f32 / last - 0.5) * sign;
        changed |= scale_limb(c, limb[0], spread.powf(along));
    }
    if changed {
        keep_strokes(c, &before);
    }
    changed
}

/// Whether two limbs have the same shape: bone by bone, each bone hangs from
/// the bone at the same place in the other limb.
fn same_tree(c: &Creature, x: &[usize], y: &[usize]) -> bool {
    let parents = parent_bones(c);
    let place = |limb: &[usize], k: usize| {
        parents[c.bones[limb[k]].a as usize].and_then(|p| limb.iter().position(|&b| b == p))
    };
    x.len() == y.len() && (0..x.len()).all(|k| place(x, k) == place(y, k))
}

/// Makes one limb the mirror image of a limb of the same shape: its bones get
/// the other limb's lengths, joint ranges (mirrored, so it bends the other
/// way) and organs, its nodes get the other limb's sizes and grips and take
/// the mirrored pose about its own joint, and its muscles are replaced by
/// copies of the other limb's, half a cycle later. A body that has two
/// similar limbs becomes a symmetric pair that alternates, as Sims' creatures
/// did and as Cheney et al. found regular bodies do. In three of ten moves the
/// limb is copied without the reflection (the same pose and ranges), in phase
/// or half a cycle later.
pub(crate) fn symmetrize_limb_pair(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let matching = matching_limbs(c);
    let options: Bounded<u16, 512> = (0..matching.len())
        .filter(|&k| {
            let (x, y) = matching.get(k);
            same_tree(c, x, y)
        })
        .map(|k| k as u16)
        .collect();
    let Some(&k) = options.get(rng.index(options.len().max(1))) else {
        return false;
    };
    let (x, y) = matching.get(k as usize);
    let (source, target) = if rng.unit() < 0.5 { (*x, *y) } else { (*y, *x) };
    let (from_bones, from_muscles) = actuation(c, source[0]);
    let (to_bones, to_muscles) = actuation(c, target[0]);
    if from_bones.len() != to_bones.len() {
        return false;
    }
    let dropped: MuscleIds = to_muscles
        .iter()
        .copied()
        .filter(|&i| !ring(c, &c.muscles[i]))
        .collect();
    if c.muscles.len() - dropped.len() + from_muscles.len() > cfg.max_muscles {
        return false;
    }
    // Usually the mirror image, half a cycle later. Sometimes the same limb on
    // the other joint (not reflected), in phase or half a cycle later.
    let mirror = rng.unit() < 0.7;
    let lag = if mirror || rng.unit() < 0.5 { 0.5 } else { 0.0 };
    let side = if mirror { -1.0 } else { 1.0 };
    let before = spans(c);
    let origin_from = pos(c, c.bones[source[0]].a as usize);
    let origin_to = pos(c, c.bones[target[0]].a as usize);
    for (&s, &t) in source.iter().zip(target.iter()) {
        let (bone, end) = (c.bones[s], c.nodes[c.bones[s].b as usize]);
        let [x, y] = clamped(
            origin_to[0] + side * (end.x - origin_from[0]),
            origin_to[1] + (end.y - origin_from[1]),
        );
        let node = &mut c.nodes[c.bones[t].b as usize];
        (node.x, node.y, node.diameter, node.friction) = (x, y, end.diameter, end.friction);
        let copy = &mut c.bones[t];
        copy.rest_length = bone.rest_length;
        (copy.min_angle, copy.max_angle) = if mirror {
            (-bone.max_angle, -bone.min_angle)
        } else {
            (bone.min_angle, bone.max_angle)
        };
        copy.organ_mass = bone.organ_mass;
        copy.organ_at = bone.organ_at;
    }
    keep_strokes(c, &before);
    let map = |b: u32| {
        to_bones[from_bones
            .iter()
            .position(|&x| x == b as usize)
            .expect("limb bone")] as u32
    };
    let copies: Muscles = from_muscles
        .iter()
        .map(|&i| {
            let old = c.muscles[i];
            let mut m = Muscle {
                bone_a: map(old.bone_a),
                bone_b: map(old.bone_b),
                phase: (old.phase + lag).rem_euclid(1.0),
                reset: (old.reset + lag).rem_euclid(1.0),
                ..old
            };
            fit_stroke(c, &mut m, Some(&old));
            m
        })
        .collect();
    let mut index = 0;
    c.muscles.retain(|_| {
        index += 1;
        !dropped.contains(&(index - 1))
    });
    c.muscles.extend(copies);
    true
}

/// Sets the phases of the leaf limbs by their place along the body, front to
/// back, to one of five patterns: all together (a hop), alternating (a walk),
/// the front half against the back half (a bound), or a wave that runs from
/// front to back or from back to front. Then, half the time, it starts the
/// whole gait at another point of its cycle, because a gait catches or not
/// by how it starts (`shift_gait_start`). Unlike `limb_phase_pattern`, the
/// pattern follows the limbs' positions, so it keeps its meaning when the
/// limbs are numbered in another order.
pub(crate) fn retime_gait_by_position(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs = limbs_front_to_back(c);
    if limbs.len() < 2 {
        return false;
    }
    let groups = muscle_groups(c, &limbs);
    let leads: Bounded<(usize, usize), MAX_NODES> = groups
        .iter()
        .enumerate()
        .filter_map(|(rank, group)| Some((rank, strongest(c, group)?)))
        .collect();
    if leads.len() < 2 {
        return false;
    }
    let count = limbs.len() as f32;
    let pattern = rng.index(5);
    let origin = c.muscles[leads[0].1].phase;
    let mut changed = false;
    for &(rank, lead) in &leads {
        let offset = match pattern {
            0 => 0.0,
            1 => 0.5 * (rank % 2) as f32,
            2 => {
                if 2 * rank >= limbs.len() {
                    0.5
                } else {
                    0.0
                }
            }
            3 => rank as f32 / count,
            _ => -(rank as f32) / count,
        };
        changed |= retime_group(c, &groups[rank], lead, origin + offset);
    }
    if rng.unit() < 0.5 {
        let longest = c.muscles.iter().map(|m| m.period).fold(0.0, f32::max);
        let dt = rng.range(0.05, 0.95) * longest;
        for m in &mut c.muscles {
            m.phase = (m.phase + dt / m.period).rem_euclid(1.0);
        }
        changed = true;
    }
    changed
}

/// Braces every joint of a limb that has two or more, all against the stop
/// on one side, each with a small flex left: the limb takes a bent, rigid
/// shape and its muscles work against the stops. The best elites of a save
/// held 40 to 80% of their joints against a stop (`brace_joint` does one
/// joint, this does the chain).
pub(crate) fn brace_limb_chain(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let roots: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| branch(c, b).len() >= 2)
        .collect();
    let Some(root) = pick(&roots, rng) else {
        return false;
    };
    let upper = rng.unit() < 0.5;
    let mut changed = false;
    for j in branch(c, root) {
        changed |= brace(c, j, upper, rng);
    }
    changed
}

/// Moves the muscles that share a phase together, whatever limb they drive:
/// either the whole group of active muscles within 6% of a cycle of a chosen
/// muscle moves by 5 to 30% of a cycle, or two such groups, at least 15% of a
/// cycle apart, swap places. Muscles that run one motor program move as one,
/// the way a correlated mutation moves genes that act together (Beyer and
/// Schwefel 2002), which no change to one limb or one gene can do.
pub(crate) fn phase_cluster_move(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let active: MuscleIds = (0..c.muscles.len())
        .filter(|&i| c.muscles[i].long > c.muscles[i].short)
        .collect();
    if active.len() < 4 {
        return false;
    }
    let near = |c: &Creature, centre: f32| -> MuscleIds {
        active
            .iter()
            .copied()
            .filter(|&i| turn(centre, c.muscles[i].phase).abs() <= 0.06)
            .collect()
    };
    let seed = c.muscles[active[rng.index(active.len())]].phase;
    let group = near(c, seed);
    if group.len() < 3 {
        return false;
    }
    if rng.unit() < 0.5 {
        let shift = rng.range(0.05, 0.3) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
        shift_group(c, &group, shift);
        return true;
    }
    let apart: MuscleIds = active
        .iter()
        .copied()
        .filter(|&i| turn(seed, c.muscles[i].phase).abs() > 0.15)
        .collect();
    let Some(&other) = apart.get(rng.index(apart.len().max(1))) else {
        return false;
    };
    let seed_other = c.muscles[other].phase;
    let second = near(c, seed_other);
    if second.len() < 3 {
        return false;
    }
    let step = turn(seed, seed_other);
    shift_group(c, &group, step);
    shift_group(c, &second, -step);
    true
}

/// Muscles that pull the same pair of bones form a bundle, and evolved bodies
/// hold a few large ones (the best elites of a save kept 30 and more muscles
/// on one pair of bones). Moves two to four muscles of a bundle of five or
/// more onto the joint between one of its bones and a bone that touches it
/// (not the other bundle bone): the force moves to another joint, each moved
/// muscle keeps its timing and the shape of its stroke, the bundle keeps at
/// least three muscles, and the body gains none. No change to one muscle does
/// that, because a muscle is one in thirty of its bundle.
pub(crate) fn reassign_bundle(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let body: &Creature = c;
    let n = body.bones.len();
    let mut count = [0u8; MAX_NODES * MAX_NODES];
    let key = |x: u32, y: u32| (x.min(y) as usize) * MAX_NODES + x.max(y) as usize;
    for m in &body.muscles {
        let k = key(m.bone_a, m.bone_b);
        count[k] = count[k].saturating_add(1);
    }
    let bundles: Bounded<usize, 64> = (0..n * MAX_NODES)
        .filter(|&k| count[k] >= 5)
        .take(64)
        .collect();
    let Some(&bundle) = bundles.get(rng.index(bundles.len().max(1))) else {
        return false;
    };
    let (x, y) = ((bundle / MAX_NODES) as u32, (bundle % MAX_NODES) as u32);
    // The target: one bundle bone with a bone it shares a node with, not the
    // other bundle bone.
    let neighbours = |bone: u32| {
        (0..n as u32)
            .filter(move |&z| z != bone && z != x && z != y)
            .filter(move |&z| shared_node(body, bone as usize, z as usize).is_some())
            .map(move |z| (bone, z))
    };
    let options: Bounded<(u32, u32), 64> = neighbours(x).chain(neighbours(y)).take(64).collect();
    let Some(&(keep, target)) = options.get(rng.index(options.len().max(1))) else {
        return false;
    };
    let node = shared_node(body, keep as usize, target as usize).expect("neighbours share a node");
    let members: MuscleIds = (0..body.muscles.len())
        .filter(|&i| key(body.muscles[i].bone_a, body.muscles[i].bone_b) == bundle)
        .collect();
    let moved = (2 + rng.index(3)).min(members.len() - 3);
    // Spread over the bundle: every `step`-th muscle from a random start.
    let step = members.len() / moved;
    let first = rng.index(members.len());
    for k in 0..moved {
        let i = members[(first + k * step) % members.len()];
        let old = c.muscles[i];
        let mut m = old;
        let near = rng.range(0.05, 0.5);
        let at = |bone: u32| {
            if c.bones[bone as usize].a == node {
                near
            } else {
                1.0 - near
            }
        };
        (m.bone_a, m.bone_b) = (keep, target);
        (m.anchor_a, m.anchor_b) = (at(keep), at(target));
        fit_stroke(c, &mut m, Some(&old));
        c.muscles[i] = m;
    }
    true
}

/// For each bone, how many bones its branch has and which they are (a set of
/// bone numbers as bits). Bones are in parent-first order, so one pass from
/// the last bone to the first gathers every branch.
fn branch_sizes(c: &Creature) -> ([u8; MAX_NODES], [u32; MAX_NODES]) {
    let parents = parent_bones(c);
    let mut sizes = [1u8; MAX_NODES];
    let mut sets: [u32; MAX_NODES] = std::array::from_fn(|b| 1u32 << b);
    for b in (0..c.bones.len()).rev() {
        if let Some(above) = parents[c.bones[b].a as usize] {
            sizes[above] += sizes[b];
            sets[above] |= sets[b];
        }
    }
    (sizes, sets)
}

/// Whether limb `root` (its branch, as the set of bones `branch`) has muscles
/// among its bones and the bone above it: a program to give away.
fn has_program(c: &Creature, root: usize, branch: u32) -> bool {
    let Some(above) = parent_bones(c)[c.bones[root].a as usize] else {
        return false;
    };
    let set = branch | 1 << above;
    c.muscles
        .iter()
        .any(|m| set >> m.bone_a & 1 == 1 && set >> m.bone_b & 1 == 1)
}

/// The muscles limb `r` of `c` would lose to the program of limb `d` of
/// `donor` (its muscles off the motor ring) and the copies that program makes
/// on `c`: the donor's muscles on the bones at the same places, their strokes
/// in proportion to their spans on `c`, everything else as it was. `None`
/// when the limbs have a different number of bones.
fn program_of(c: &Creature, donor: &Creature, r: usize, d: usize) -> Option<(MuscleIds, Muscles)> {
    let (to_bones, to_muscles) = actuation(c, r);
    let (from_bones, from_muscles) = actuation(donor, d);
    if to_bones.len() != from_bones.len() {
        return None;
    }
    let dropped: MuscleIds = to_muscles
        .iter()
        .copied()
        .filter(|&i| !ring(c, &c.muscles[i]))
        .collect();
    let map = |b: u32| {
        to_bones[from_bones
            .iter()
            .position(|&x| x == b as usize)
            .expect("limb bone")] as u32
    };
    let copies: Muscles = from_muscles
        .iter()
        .map(|&i| {
            let old = donor.muscles[i];
            let mut m = Muscle {
                bone_a: map(old.bone_a),
                bone_b: map(old.bone_b),
                ..old
            };
            let (was, now) = (span(donor, &old).max(0.05), span(c, &m).max(0.05));
            m.short = (old.short / was * now).max(0.01);
            m.long = (old.long / was * now).max(m.short);
            m
        })
        .collect();
    Some((dropped, copies))
}

/// Replaces the muscles in `dropped` with `copies`, after moving the copies
/// in time (phase and touchdown reset, all by one amount) so that the copy
/// with the most drive lands on phase `keep`, when there is one.
fn replace_program(c: &mut Creature, dropped: &[usize], mut copies: Muscles, keep: Option<f32>) {
    let lead = copies
        .iter()
        .copied()
        .max_by(|x, y| drive(x).total_cmp(&drive(y)));
    if let (Some(keep), Some(lead)) = (keep, lead) {
        let shift = turn(lead.phase, keep);
        for m in &mut copies {
            m.phase = (m.phase + shift).rem_euclid(1.0);
            m.reset = (m.reset + shift).rem_euclid(1.0);
        }
    }
    let mut index = 0;
    c.muscles.retain(|_| {
        index += 1;
        !dropped.contains(&(index - 1))
    });
    c.muscles.extend(copies);
}

/// Gives one of this body's limbs the muscle program of a limb of another
/// elite that has the same number of bones: the donor's muscles (their
/// attachment places along the bones, their strokes in proportion to their
/// spans, duty, strength, touchdown sensors and the timing among them) replace
/// the limb's own, and the whole program is moved in time so its strongest
/// muscle keeps the phase the limb's strongest muscle had. The skeleton stays.
/// Crossover of two bodies of one plan mixes their genes in place; this does
/// it between plans, one limb at a time (Lessin, Fussell and Miikkulainen
/// 2013 exchange whole modules the same way).
pub(crate) fn transplant_limb_program(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    cx: &Context,
) -> bool {
    let Some(donor) = cx.donor() else {
        return false;
    };
    // Limbs by number of bones, the donor's only when they have a program.
    let (mine_sizes, _) = branch_sizes(c);
    let (their_sizes, their_sets) = branch_sizes(donor);
    let mine: Bounded<(u8, u8), MAX_NODES> = limb_roots(c)
        .into_iter()
        .map(|r| (r as u8, mine_sizes[r]))
        .collect();
    let theirs: Bounded<(u8, u8), MAX_NODES> = limb_roots(donor)
        .into_iter()
        .filter(|&d| has_program(donor, d, their_sets[d]))
        .map(|d| (d as u8, their_sizes[d]))
        .collect();
    let mut pairs: Bounded<(u8, u8), 1024> = Bounded::new();
    for &(r, size) in &mine {
        for &(d, other) in &theirs {
            if size == other && !pairs.is_full() {
                pairs.push((r, d));
            }
        }
    }
    let Some(&(r, d)) = pairs.get(rng.index(pairs.len().max(1))) else {
        return false;
    };
    let Some((dropped, copies)) = program_of(c, donor, r as usize, d as usize) else {
        return false;
    };
    if c.muscles.len() - dropped.len() + copies.len() > cfg.max_muscles {
        return false;
    }
    let keep = strongest(c, &dropped).map(|i| c.muscles[i].phase);
    replace_program(c, &dropped, copies, keep);
    true
}

/// Gives the body the gait of another elite: its leaf limbs, from front to
/// back, take the muscle programs of the other elite's leaf limbs at the same
/// places in the order, where the two have the same number of bones and the
/// donor's limb has a program, with the timing among the limbs the donor had.
/// The whole set is moved in time so that the strongest muscle of the first
/// limb that takes a program keeps the phase it had. Needs two limbs that
/// match. The skeleton stays. Where `transplant_limb_program` moves one
/// limb's program, this moves a whole set of limbs, as modules are exchanged
/// in Lessin, Fussell and Miikkulainen (2013). In the hub island the donor can
/// come from any isolated island.
pub(crate) fn transplant_gait(
    c: &mut Creature,
    cfg: &Config,
    _rng: &mut Rng,
    cx: &Context,
) -> bool {
    let Some(donor) = cx.donor() else {
        return false;
    };
    let (mine, theirs) = (limbs_front_to_back(c), limbs_front_to_back(donor));
    let mut dropped = MuscleIds::new();
    let mut copies = Muscles::new();
    // The shift that keeps the first limb's place in the cycle: from the phase
    // its strongest new muscle has in the donor to the phase its old one had.
    let mut shift = None;
    let mut limbs = 0;
    for (x, y) in mine.iter().zip(theirs.iter()) {
        if x.len() != y.len() {
            continue;
        }
        let Some((out, program)) = program_of(c, donor, x[0], y[0]) else {
            continue;
        };
        if program.is_empty() || copies.len() + program.len() > MAX_MUSCLES {
            continue;
        }
        if limbs == 0 {
            let lead = program
                .iter()
                .copied()
                .max_by(|p, q| drive(p).total_cmp(&drive(q)))
                .map(|m| m.phase);
            let old = strongest(c, &out).map(|i| c.muscles[i].phase);
            shift = lead.zip(old).map(|(lead, old)| turn(lead, old));
        }
        for i in out {
            if !dropped.contains(&i) && !dropped.is_full() {
                dropped.push(i);
            }
        }
        copies.extend_from_slice(&program);
        limbs += 1;
    }
    if limbs < 2 || c.muscles.len() - dropped.len() + copies.len() > cfg.max_muscles {
        return false;
    }
    if let Some(shift) = shift {
        for m in &mut copies {
            m.phase = (m.phase + shift).rem_euclid(1.0);
            m.reset = (m.reset + shift).rem_euclid(1.0);
        }
    }
    replace_program(c, &dropped, copies, None);
    true
}

/// Retunes one limb's controller in several ways at once: two to four of five
/// changes, each applied to every active muscle that works the limb. The
/// stroke widens or narrows about its middle, the posture shifts, the phase
/// moves, the duty changes (with the phase following so each contraction keeps
/// its middle) and the strength scales. The limb operators each change one of
/// these on some limb, so a limb that needs two of them moved together takes
/// several of them in a row, and the first is lost on the way.
pub(crate) fn retune_limb_package(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    // A driven limb at random: a limb at random, and another when it has no
    // active muscle (the same as a pick among the driven limbs, without
    // scanning them all).
    let mut roots = limb_roots(c);
    let group = loop {
        if roots.is_empty() {
            return false;
        }
        let k = rng.index(roots.len());
        let group = active_on(c, roots[k]);
        if !group.is_empty() {
            break group;
        }
        roots.swap_remove(k);
    };
    let mut kinds = [0usize, 1, 2, 3, 4];
    for k in 0..kinds.len() {
        let other = k + rng.index(kinds.len() - k);
        kinds.swap(k, other);
    }
    let top = max_stroke();
    let mut changed = false;
    for &kind in &kinds[..2 + rng.index(3)] {
        let sign = if rng.unit() < 0.5 { -1.0 } else { 1.0 };
        let (factor, share, nudge, duty) = (
            rng.range(0.75f32.ln(), 1.35f32.ln()).exp(),
            sign * rng.range(0.04, 0.12),
            sign * rng.range(0.02, 0.12),
            sign * rng.range(0.03, 0.15),
        );
        for &i in &group {
            let m = &mut c.muscles[i];
            let before = *m;
            match kind {
                0 => {
                    let (middle, half) =
                        ((m.short + m.long) * 0.5, (m.long - m.short) * 0.5 * factor);
                    m.short = (middle - half).max(0.01);
                    m.long = (middle + half).clamp(m.short, top);
                }
                1 => {
                    let stroke = m.long - m.short;
                    m.short = (m.short + share * stroke).clamp(0.01, (top - stroke).max(0.01));
                    m.long = (m.short + stroke).min(top);
                }
                2 => {
                    m.phase = (m.phase + nudge).rem_euclid(1.0);
                    m.reset = (m.reset + nudge).rem_euclid(1.0);
                }
                3 => {
                    let new = (m.duty + duty).clamp(0.05, 0.95);
                    m.phase = (m.phase + 0.5 * (new - m.duty)).rem_euclid(1.0);
                    m.duty = new;
                }
                _ => m.stiffness = (m.stiffness * factor).clamp(1.0, 120.0),
            }
            changed |= *m != before;
        }
    }
    changed
}

/// Trims a body in one move: the idlest limb tips go (as `prune_idle_limb`
/// picks them, one to four by the size of the body) and up to three of the
/// weakest muscles off the motor ring with them. It applies to bodies of five
/// nodes or more with three or more muscles to a node. The best elites of a
/// save gain more from losing idle parts than from any other single change
/// (36% of the children of `prune_idle_limb` beat their parent), and an elite
/// that carries several takes several of these steps one after the other,
/// each a separate child.
pub(crate) fn trim_body(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    // Only a body with muscles to spare: three or more to a node. Young
    // bodies have under two, and every muscle of theirs pulls.
    if c.nodes.len() <= 4 || c.muscles.len() < 3 * c.nodes.len() {
        return false;
    }
    let mut next = c.clone();
    // The share a body can lose grows with its size.
    let tips = 1 + rng.index(1 + (c.nodes.len() / 8).min(3));
    let shed = shed_tips(&mut next, c.nodes.len(), tips, rng);
    let mut cut = 0;
    for _ in 0..rng.index(1 + (c.muscles.len() / 12).min(3)) {
        let weakest = (0..next.muscles.len())
            .filter(|&i| !ring(&next, &next.muscles[i]))
            .min_by(|&x, &y| drive(&next.muscles[x]).total_cmp(&drive(&next.muscles[y])));
        let Some(weakest) = weakest else {
            break;
        };
        next.muscles.remove(weakest);
        cut += 1;
    }
    if shed + cut == 0 || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// The operators that add one new part to the body (a tip, a toe and heel, a
/// joint with a muscle across it, a copy of a limb, a lever), none of which
/// closes the ring itself.
const NEW_PART: [Operator; 5] = [
    super::limbs::grow_actuated_tip,
    super::junctions::grow_heel_toe,
    super::limbs::split_bone_actuated,
    super::limbs::copy_limb,
    super::junctions::grow_lever_spur,
];

/// Adds a new part and fits it into the body in the same move: its muscles
/// are timed against the gait's main driver (in phase, a quarter, a half or
/// three quarters of a cycle later), half the time its joint is braced
/// against a stop, and some of the time its foot senses touchdown. A part
/// that arrives with a random program rarely works with the gait; one that
/// arrives timed to it has a chance. The part comes from the first of three
/// `NEW_PART` operators, tried from a random start, that fits the body. The
/// idlest limb tips go for the added nodes (`shed_tips`), so the body ends no
/// bigger.
pub(crate) fn grow_integrated_limb(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    cx: &Context,
) -> bool {
    let mut next = c.clone();
    let (before, first_muscle) = (next.bones.len(), next.muscles.len());
    let start = rng.index(NEW_PART.len());
    let grown = (0..3).any(|k| NEW_PART[(start + k) % NEW_PART.len()](&mut next, cfg, rng, cx));
    if !grown {
        return false;
    }
    let focus: BoneIds = (before..next.bones.len()).collect();
    // The muscles the new part brought. A muscle that was only moved onto the
    // part (a split bone's, a lever's) is not among them and keeps its timing.
    let group: MuscleIds = (first_muscle..next.muscles.len()).collect();
    if let (Some(lead), Some(mine)) = (lead_muscle(&next, &group), strongest(&next, &group)) {
        let offset = [0.0, 0.25, 0.5, 0.75][rng.index(4)] + rng.range(-0.03, 0.03);
        let target = next.muscles[lead].phase + offset;
        retime_group(&mut next, &group, mine, target);
    }
    if !focus.is_empty() && rng.unit() < 0.5 {
        let upper = rng.unit() < 0.5;
        brace(&mut next, focus[0], upper, rng);
    }
    if rng.unit() < 0.4
        && let Some(foot) = pick_foot(&next, &focus, rng)
    {
        reflex_foot(&mut next, foot, rng);
    }
    let added = next.nodes.len() - c.nodes.len();
    let shed = shed_tips(&mut next, c.nodes.len(), added, rng);
    if shed < added || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Gives the limb that starts at `root` a muscle across its joint to the bone
/// above it, timed like `template` but moved to phase `lead_phase`. Returns
/// whether it added one: it does not when there is no bone above, no room for
/// a muscle or an active muscle already joins the two. The anchors lie 0.3 to
/// 0.9 of the way along the limb's first bone and 0.3 to 0.8 along the bone
/// above.
pub(super) fn hinge_muscle(
    c: &mut Creature,
    cfg: &Config,
    root: usize,
    template: &Muscle,
    lead_phase: f32,
    rng: &mut Rng,
) -> bool {
    let Some(above) = parent_bones(c)[c.bones[root].a as usize] else {
        return false;
    };
    if c.muscles.len() >= cfg.max_muscles {
        return false;
    }
    let joined = c.muscles.iter().any(|m| {
        let ends = (m.bone_a as usize, m.bone_b as usize);
        (ends == (root, above) || ends == (above, root)) && m.long > m.short
    });
    if joined {
        return false;
    }
    let anchors = (rng.range(0.3, 0.9), rng.range(0.3, 0.8));
    let mut m = new_muscle(c, root, above, anchors, Some(template), rng);
    // A gentle stroke around the span, whatever stroke the template has.
    fit_stroke(c, &mut m, None);
    let shift = turn(m.phase, lead_phase);
    m.phase = (m.phase + shift).rem_euclid(1.0);
    m.reset = (m.reset + shift).rem_euclid(1.0);
    c.muscles.push(m);
    true
}

/// Adds a limb together with its mirror image, hung from another node of the
/// body: one copy as the source limb is, one reflected about the vertical
/// through the node, with the joint ranges mirrored. Each new limb gets a
/// muscle across its joint (`hinge_muscle`, unless the copy brought an active
/// one), timed like the muscles at the joint it hangs from, the first in
/// phase with the gait's main driver (or a quarter cycle later) and the mirror
/// image half a cycle after it. The source is one of the two lowest limbs of
/// one or two bones, which are the ones that reach the ground, and the pair
/// hangs from one of the three nodes nearest to the source's joint in height.
/// The idlest limb tips go for the added nodes (`shed_tips`), and the body
/// ends at most one node bigger. Sims (1994) grew creatures whose limbs came
/// in such pairs.
pub(crate) fn mirrored_limb_pair(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut legs: super::Limbs = leaf_limbs(c)
        .into_iter()
        .filter(|limb| limb.len() <= 2 && room(c, cfg, 2 * limb.len(), 0))
        .collect();
    legs.sort_stable_by(|x, y| tip_y(c, x).total_cmp(&tip_y(c, y)));
    let Some(limb) = legs.get(rng.index(legs.len().clamp(1, 2))) else {
        return false;
    };
    let root = limb[0];
    let joint = c.bones[root].a as usize;
    let inside = branch_nodes(c, limb);
    let mut places: BoneIds = (1..c.nodes.len())
        .filter(|&n| n != joint && !inside.contains(&n))
        .collect();
    places.sort_stable_by(|&x, &y| {
        let level = |n: usize| (c.nodes[n].y - c.nodes[joint].y).abs();
        level(x).total_cmp(&level(y))
    });
    let Some(&at) = places.get(rng.index(places.len().clamp(1, 3))) else {
        return false;
    };
    let base = [0.0, 0.25][rng.index(2)];
    let (from, to) = (c.nodes[joint], c.nodes[at]);
    let above = parent_bones(c)[at];
    let template = above
        .and_then(|b| {
            let near: MuscleIds = muscles_on(c, &[b], false)
                .into_iter()
                .filter(|&i| c.muscles[i].long > c.muscles[i].short)
                .collect();
            strongest(c, &near)
        })
        .or_else(|| strongest(c, &(0..c.muscles.len()).collect::<MuscleIds>()))
        .map(|i| c.muscles[i]);
    let Some(template) = template else {
        return false;
    };
    let lead = lead_muscle(c, &[]).map_or(template.phase, |i| c.muscles[i].phase);
    let mut next = c.clone();
    let forward = |[x, y]: [f32; 2]| [to.x + (x - from.x), to.y + (y - from.y)];
    let backward = |[x, y]: [f32; 2]| [to.x - (x - from.x), to.y + (y - from.y)];
    let first = next.bones.len();
    let Some(_) = copy_branch_limited(&mut next, cfg, root, at, forward, false, base, 1) else {
        return false;
    };
    let second = next.bones.len();
    let Some(_) = copy_branch_limited(&mut next, cfg, root, at, backward, true, base + 0.5, 1)
    else {
        return false;
    };
    for (bone, shift) in [(first, base), (second, base + 0.5)] {
        hinge_muscle(&mut next, cfg, bone, &template, lead + shift, rng);
    }
    let added = next.nodes.len() - c.nodes.len();
    let shed = shed_tips(&mut next, c.nodes.len(), added, rng);
    if shed + 1 < added || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Copies the trunk bone `trunk` with the leaf limbs on its child node and
/// inserts the copy after it in the chain, as `repeat_body_segment` does, with
/// a gradient: the copy is `taper` times the size of the original (its bone,
/// its limbs and their strokes), its muscles run `phase` of a cycle later, and
/// it carries at most `quota` of the muscles across the trunk's upper joint
/// and of each limb (the ones with the most drive). Returns the copy's bone,
/// or `None` when the segment cannot be copied.
fn repeat_segment(
    c: &mut Creature,
    cfg: &Config,
    trunk: usize,
    phase: f32,
    taper: f32,
    quota: usize,
) -> Option<usize> {
    let children = child_bones(c);
    let parents = parent_bones(c);
    let (a, b) = (c.bones[trunk].a as usize, c.bones[trunk].b as usize);
    let above = parents[a]?;
    let limbs = leaf_limbs_at(c, &children, b);
    if limbs.is_empty() || is_neck(c, trunk) {
        return None;
    }
    let length = (c.bones[trunk].rest_length * taper).clamp(0.03, max_bone_length());
    let taper = length / c.bones[trunk].rest_length;
    let mut joint: MuscleIds = (0..c.muscles.len())
        .filter(|&i| {
            let ends = (c.muscles[i].bone_a as usize, c.muscles[i].bone_b as usize);
            ends == (trunk, above) || ends == (above, trunk)
        })
        .collect();
    joint.sort_stable_by(|&x, &y| drive(&c.muscles[y]).total_cmp(&drive(&c.muscles[x])));
    joint.truncate(quota);
    let bones: usize = 1 + limbs.iter().map(|&l| branch(c, l).len()).sum::<usize>();
    if !room(c, cfg, bones, joint.len()) {
        return None;
    }
    let offset = scale(sub(pos(c, b), pos(c, a)), taper);
    let before = spans(c);
    let node = add_node(c, b, add(pos(c, b), offset));
    let copy = c.bones.len();
    c.bones.push(Bone {
        a: b as u32,
        b: node as u32,
        rest_length: length,
        organ_mass: 0.0,
        organ_at: 0.5,
        ..c.bones[trunk]
    });
    for &i in &joint {
        let m = c.muscles[i];
        let map = |x: u32| (if x as usize == trunk { copy } else { trunk }) as u32;
        let mut new = Muscle {
            bone_a: map(m.bone_a),
            bone_b: map(m.bone_b),
            phase: (m.phase + phase).rem_euclid(1.0),
            reset: (m.reset + phase).rem_euclid(1.0),
            ..m
        };
        fit_stroke(c, &mut new, Some(&m));
        c.muscles.push(new);
    }
    // The rest of the body below the trunk hangs from the copy, and the
    // muscles that worked that joint move with it.
    for &child in &children[b] {
        if limbs.contains(&child) {
            continue;
        }
        c.bones[child].a = node as u32;
        shift_branch(c, child, offset);
        for m in &mut c.muscles {
            if m.bone_a as usize == trunk && m.bone_b as usize == child {
                m.bone_a = copy as u32;
            }
            if m.bone_b as usize == trunk && m.bone_a as usize == child {
                m.bone_b = copy as u32;
            }
        }
    }
    let (root, top) = (pos(c, b), pos(c, node));
    for &limb in &limbs {
        let (first_bone, first_muscle) = (c.bones.len(), c.muscles.len());
        let place = |p: [f32; 2]| add(top, scale(sub(p, root), taper));
        copy_branch_limited(c, cfg, limb, node, place, false, phase, quota)?;
        for bone in &mut c.bones[first_bone..] {
            bone.rest_length *= taper;
        }
        for m in &mut c.muscles[first_muscle..] {
            m.short *= taper;
            m.long *= taper;
        }
    }
    keep_strokes(c, &before);
    Some(copy)
}

/// Repeats a trunk segment one or two times down the chain, each copy a
/// little larger or smaller than the one before it (0.8 to 1.25 times) and
/// its muscles a fixed step of 0.1 to 0.3 of a cycle later (or earlier) than
/// the one before it: repeated parts with a gradient in size and in timing,
/// which is how Hornby and Pollack's generative bodies gain regular,
/// many-limbed shapes, and a travelling wave of contraction down the body. The
/// copies carry at most two muscles per joint and limb. The idlest limb tips
/// go for the added nodes (`shed_tips`), and the body ends at most one node
/// bigger.
pub(crate) fn segment_chain(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let children = child_bones(c);
    let trunks: BoneIds = (0..c.bones.len())
        .filter(|&j| {
            !is_neck(c, j) && !leaf_limbs_at(c, &children, c.bones[j].b as usize).is_empty()
        })
        .collect();
    let Some(mut trunk) = pick(&trunks, rng) else {
        return false;
    };
    let repeats = 1 + rng.index(2);
    let step = rng.range(0.1, 0.3) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    let taper = rng.range(0.8, 1.25);
    let mut next = c.clone();
    let mut made = 0;
    for _ in 0..repeats {
        let mut trial = next.clone();
        let Some(copy) = repeat_segment(&mut trial, cfg, trunk, step, taper, 2) else {
            break;
        };
        next = trial;
        trunk = copy;
        made += 1;
    }
    if made == 0 {
        return false;
    }
    let added = next.nodes.len() - c.nodes.len();
    let shed = shed_tips(&mut next, c.nodes.len(), added, rng);
    if shed + 1 < added || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

#[cfg(test)]
mod tests {
    use super::super::{Operator, tests::grown};
    use super::*;
    use crate::evolution::{Population, repair};

    /// Runs `op` on each of `bodies`. A changed body must pass
    /// `check(before, after)`. An unchanged one must be as it was. Returns how
    /// many changed.
    fn run(op: Operator, bodies: &[Creature], check: impl Fn(&Creature, &Creature)) -> usize {
        let cfg = Config::default();
        let mut applied = 0;
        for (i, body) in bodies.iter().enumerate() {
            let mut c = body.clone();
            let cx = Context::of(None);
            if op(&mut c, &cfg, &mut Rng::new(41, 0, i), &cx) {
                applied += 1;
                assert!(c.nodes.len() <= cfg.max_nodes && c.muscles.len() <= cfg.max_muscles);
                check(body, &c);
                repair(&mut c, &cfg, &mut Rng::new(43, 0, i));
                let mut pop = Population::default();
                pop.push(c);
                let one = Config {
                    population: 1,
                    ..cfg.clone()
                };
                pop.validate(&one).expect("a valid body");
            } else {
                assert!(
                    c.nodes == body.nodes && c.bones == body.bones && c.muscles == body.muscles
                );
            }
        }
        applied
    }

    /// The test bodies with their first muscle-bearing limb twinned, so every
    /// one has a pair of limbs of the same shape.
    fn twinned() -> Vec<Creature> {
        let cfg = Config::default();
        grown()
            .into_iter()
            .filter_map(|mut c| {
                let root = limb_roots(&c)
                    .into_iter()
                    .find(|&b| !muscles_on(&c, &actuation(&c, b).0, true).is_empty())?;
                let joint = c.bones[root].a as usize;
                super::super::copy_branch(&mut c, &cfg, root, joint, |p| p, false, 0.25)?;
                repair(&mut c, &cfg, &mut Rng::new(3, 0, 0));
                Some(c)
            })
            .collect()
    }

    /// The test bodies with extra muscles (copies of their own with a changed
    /// phase) up to three to a node, as evolved bodies have.
    fn dense() -> Vec<Creature> {
        grown()
            .into_iter()
            .map(|mut c| {
                let want = (3 * c.nodes.len() + 2).min(MAX_MUSCLES);
                for k in 0..want.saturating_sub(c.muscles.len()) {
                    let mut m = c.muscles[k % c.muscles.len()];
                    m.phase = (m.phase + 0.1 * k as f32).rem_euclid(1.0);
                    c.muscles.push(m);
                }
                c
            })
            .collect()
    }

    fn ring_closed(c: &Creature) -> bool {
        let n = c.bones.len();
        (0..n).all(|a| {
            let b = (a + 1) % n;
            c.muscles.iter().any(|m| {
                let ends = (m.bone_a as usize, m.bone_b as usize);
                ends == (a, b) || ends == (b, a)
            })
        })
    }

    #[test]
    fn limb_length_gradient_scales_limbs_by_their_place() {
        let applied = run(limb_length_gradient, &grown(), |before, after| {
            assert_eq!(after.nodes.len(), before.nodes.len());
            assert_eq!(after.bones.len(), before.bones.len());
            assert_eq!(after.muscles.len(), before.muscles.len());
            let longer = (0..before.bones.len())
                .filter(|&b| after.bones[b].rest_length > before.bones[b].rest_length)
                .count();
            let shorter = (0..before.bones.len())
                .filter(|&b| after.bones[b].rest_length < before.bones[b].rest_length)
                .count();
            assert!(longer + shorter > 0);
        });
        assert!(applied >= 20, "applied to {applied} of 160");
    }

    #[test]
    fn symmetrize_limb_pair_mirrors_one_limb_onto_the_other() {
        let applied = run(symmetrize_limb_pair, &twinned(), |before, after| {
            assert_eq!(after.nodes.len(), before.nodes.len());
            assert_eq!(after.bones.len(), before.bones.len());
            // Some pair of limbs is now alike in lengths and ranges: mirror
            // images, or the same.
            let pairs = matching_limbs(after);
            let alike = pairs.iter().any(|(x, y)| {
                let each = |same: bool| {
                    x.iter().zip(y.iter()).all(|(&p, &q)| {
                        let (a, b) = (after.bones[p], after.bones[q]);
                        let (low, high) = if same {
                            (b.min_angle, b.max_angle)
                        } else {
                            (-b.max_angle, -b.min_angle)
                        };
                        (a.rest_length - b.rest_length).abs() < 1e-4
                            && (a.min_angle - low).abs() < 1e-4
                            && (a.max_angle - high).abs() < 1e-4
                    })
                };
                each(true) || each(false)
            });
            assert!(alike, "no alike pair");
        });
        assert!(applied >= 20, "applied to {applied}");
    }

    #[test]
    fn retime_gait_by_position_sets_a_pattern_over_the_limbs() {
        let applied = run(retime_gait_by_position, &twinned(), |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len());
            assert!(
                before
                    .muscles
                    .iter()
                    .zip(&after.muscles)
                    .any(|(x, y)| x.phase != y.phase)
            );
        });
        assert!(applied >= 20, "applied to {applied}");
    }

    #[test]
    fn brace_limb_chain_leaves_each_joint_a_flex_against_one_stop() {
        let applied = run(brace_limb_chain, &grown(), |before, after| {
            assert_eq!(after.bones.len(), before.bones.len());
            let braced = after
                .bones
                .iter()
                .zip(&before.bones)
                .filter(|(a, b)| a.min_angle != b.min_angle || a.max_angle != b.max_angle)
                .count();
            assert!(braced >= 1);
        });
        assert!(applied >= 20, "applied to {applied}");
    }

    #[test]
    fn phase_cluster_move_keeps_every_muscle_but_moves_a_group() {
        let applied = run(phase_cluster_move, &twinned(), |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len());
            let moved = before
                .muscles
                .iter()
                .zip(&after.muscles)
                .filter(|(x, y)| x.phase != y.phase)
                .count();
            assert!(moved >= 3, "moved {moved}");
        });
        assert!(applied >= 20, "applied to {applied}");
    }

    #[test]
    fn grow_integrated_limb_keeps_the_size_and_closes_the_ring() {
        let applied = run(grow_integrated_limb, &grown(), |before, after| {
            // A new part comes in and the idlest tips go out.
            assert!(after.nodes.len() <= before.nodes.len());
            assert!(after.muscles != before.muscles);
            assert!(ring_closed(after));
        });
        assert!(applied >= 60, "applied to {applied}");
    }

    #[test]
    fn mirrored_limb_pair_adds_two_limbs_and_gives_tips_back() {
        let applied = run(mirrored_limb_pair, &grown(), |before, after| {
            // The pair adds at least two nodes and the idlest tips go, so the
            // body grows by one node at most.
            assert!(after.nodes.len() <= before.nodes.len() + 1);
            assert!(ring_closed(after));
            // Two bones with mirrored joint ranges hang from one node.
            let mirrored = (0..after.bones.len()).any(|p| {
                (p + 1..after.bones.len()).any(|q| {
                    let (x, y) = (after.bones[p], after.bones[q]);
                    x.a == y.a
                        && (x.rest_length - y.rest_length).abs() < 1e-4
                        && (x.min_angle + y.max_angle).abs() < 1e-4
                        && (x.max_angle + y.min_angle).abs() < 1e-4
                })
            });
            assert!(mirrored, "no mirrored pair");
        });
        assert!(applied >= 20, "applied to {applied}");
    }

    #[test]
    fn segment_chain_repeats_a_segment_and_gives_tips_back() {
        let applied = run(segment_chain, &grown(), |before, after| {
            assert!(after.nodes.len() <= before.nodes.len() + 1);
            assert!(ring_closed(after));
        });
        assert!(applied >= 20, "applied to {applied}");
    }

    #[test]
    fn reassign_bundle_moves_muscles_to_a_neighbouring_joint() {
        // Bodies with a bundle: the first muscle's bone pair gets six more.
        let bodies: Vec<Creature> = grown()
            .into_iter()
            .map(|mut c| {
                let m = c.muscles[0];
                for _ in 0..6 {
                    c.muscles.push(m);
                }
                c
            })
            .collect();
        let applied = run(reassign_bundle, &bodies, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len());
            let moved = before
                .muscles
                .iter()
                .zip(&after.muscles)
                .filter(|(x, y)| (x.bone_a, x.bone_b) != (y.bone_a, y.bone_b))
                .count();
            assert!((2..=4).contains(&moved), "moved {moved}");
        });
        assert!(applied >= 40, "applied to {applied}");
    }

    #[test]
    fn transplant_limb_program_gives_a_limb_the_donors_muscles() {
        let cfg = Config::default();
        let all = grown();
        let donor = all[all.len() / 2].clone();
        let mut applied = 0;
        for (i, body) in all.iter().enumerate() {
            let mut c = body.clone();
            let cx = Context::of(Some(&donor));
            if transplant_limb_program(&mut c, &cfg, &mut Rng::new(47, 0, i), &cx) {
                applied += 1;
                assert_eq!(c.bones, body.bones);
                assert!(c.muscles.len() <= cfg.max_muscles);
                assert!(c.muscles != body.muscles);
                repair(&mut c, &cfg, &mut Rng::new(49, 0, i));
                let mut pop = Population::default();
                pop.push(c);
                let one = Config {
                    population: 1,
                    ..cfg.clone()
                };
                pop.validate(&one).expect("a valid body");
            }
        }
        assert!(applied >= 40, "applied to {applied}");
    }

    #[test]
    fn retune_limb_package_changes_one_limbs_muscles_in_several_ways() {
        let applied = run(retune_limb_package, &grown(), |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len());
            assert_eq!(after.bones, before.bones);
            let (mut stroke, mut phase, mut strength) = (0, 0, 0);
            for (x, y) in before.muscles.iter().zip(&after.muscles) {
                stroke += usize::from((x.short, x.long) != (y.short, y.long));
                phase += usize::from(x.phase != y.phase);
                strength += usize::from(x.stiffness != y.stiffness);
            }
            assert!(stroke + phase + strength > 0);
        });
        assert!(applied >= 100, "applied to {applied}");
    }

    #[test]
    fn shedding_tips_removes_old_leaves_and_leaves_a_valid_body() {
        let cfg = Config::default();
        let mut removed = 0;
        for (i, body) in grown().iter().enumerate() {
            let mut c = body.clone();
            let mut rng = Rng::new(53, 0, i);
            let nodes = c.nodes.len();
            let shed = shed_tips(&mut c, nodes, 2, &mut rng);
            assert!(shed <= 2 && c.nodes.len() + shed == nodes);
            removed += shed;
            assert!(close_ring(&mut c, &cfg, &mut rng));
            assert!(ring_closed(&c));
            repair(&mut c, &cfg, &mut Rng::new(55, 0, i));
            let mut pop = Population::default();
            pop.push(c);
            let one = Config {
                population: 1,
                ..cfg.clone()
            };
            pop.validate(&one).expect("a valid body");
        }
        assert!(removed >= 200, "removed {removed} tips");
    }

    #[test]
    fn transplant_gait_gives_the_front_limbs_the_donors_programs() {
        let cfg = Config::default();
        let all = twinned();
        let donor = all[all.len() / 2].clone();
        let mut applied = 0;
        for (i, body) in all.iter().enumerate() {
            let mut c = body.clone();
            let cx = Context::of(Some(&donor));
            if transplant_gait(&mut c, &cfg, &mut Rng::new(57, 0, i), &cx) {
                applied += 1;
                assert_eq!(c.bones, body.bones);
                assert!(c.muscles.len() <= cfg.max_muscles);
                repair(&mut c, &cfg, &mut Rng::new(59, 0, i));
                let mut pop = Population::default();
                pop.push(c);
                let one = Config {
                    population: 1,
                    ..cfg.clone()
                };
                pop.validate(&one).expect("a valid body");
            }
        }
        assert!(applied >= 10, "applied to {applied}");
    }

    #[test]
    fn trim_body_removes_tips_or_muscles_and_keeps_the_body_valid() {
        let applied = run(trim_body, &dense(), |before, after| {
            assert!(after.nodes.len() <= before.nodes.len());
            assert!(
                after.nodes.len() < before.nodes.len()
                    || after.muscles.len() < before.muscles.len()
            );
            assert!(ring_closed(after));
        });
        assert!(applied >= 100, "applied to {applied}");
    }

    #[test]
    fn branch_sizes_and_programs_agree_with_the_slow_ways() {
        for body in twinned() {
            let (sizes, sets) = branch_sizes(&body);
            for b in 0..body.bones.len() {
                let limb = branch(&body, b);
                assert_eq!(sizes[b] as usize, limb.len());
                assert!(limb.iter().all(|&x| sets[b] >> x & 1 == 1));
                assert_eq!(sets[b].count_ones() as usize, limb.len());
                if !is_neck(&body, b) {
                    assert_eq!(
                        has_program(&body, b, sets[b]),
                        !actuation(&body, b).1.is_empty()
                    );
                }
            }
        }
    }
}
