//! Gait operators for a flexing back, and for tails and necks that balance a
//! gait. They share one pick slot (`GAIT_FILES` in `mod.rs`) and are compound,
//! so a child made by one gets no parameter noise after it. A leg is any leaf
//! limb (`rhythm::leaf_limbs`, a tail included), the trunk is every bone in no
//! leg and not the neck, and a spine joint joins two trunk bones.
//!
//! The sources are the bounding gaits of fast mammals, where the back flexes
//! once per stride in time with the legs and stores energy in elastic tissue
//! (Alexander 1988, Hildebrand 1959 on the gallop), tails and necks that move
//! the centre of mass against the legs (Libby et al. 2012, Full and Koditschek
//! 1999 on templates), the central pattern generators that lock a back to the
//! legs with a fixed phase (Ijspeert 2008), and the finding that bodies of
//! segments with a joint between them move well (Sims 1994, Lipson and Pollack
//! 2000).
use super::Operator;
use super::compound::{close_ring, hinge_muscle, lead_muscle, shed_tips, shift_group, strongest};
use super::junctions::{add, add_node, keep_strokes, pos, scale, spans, sub, turn_branch};
use super::limbs::{clamped, pick};
use super::muscles::{shared_node, turn};
use super::rhythm::leaf_limbs;
use super::{
    BoneIds, Context, MuscleIds, branch, branch_nodes, is_neck, muscles_on, parent_bones, room,
};
use crate::config::Config;
use crate::evolution::{
    Bone, Creature, JOINT_LIMIT, MAX_NODES, Muscle, NO_SENSOR, Rng, max_bone_length,
};

/// This file's operators, by name. Add each new one here. The pick slot draws
/// an index into this list, so reordering it changes the search for a seed.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("spine_flex_muscle", spine_flex_muscle),
    ("spine_lock_to_legs", spine_lock_to_legs),
    ("split_spine_bone", split_spine_bone),
    ("grow_counterweight_tail", grow_counterweight_tail),
    ("weight_tail_tip", weight_tail_tip),
    ("plant_tail_prop", plant_tail_prop),
    ("tail_swing_against_legs", tail_swing_against_legs),
    ("neck_bob_muscle", neck_bob_muscle),
    ("split_neck_bone", split_neck_bone),
    ("stiffen_trunk_joints", stiffen_trunk_joints),
    ("loosen_trunk_joints", loosen_trunk_joints),
    ("spine_phase_wave", spine_phase_wave),
    ("elastic_spine", elastic_spine),
    ("arch_back", arch_back),
];

/// For each bone, whether it is in a leg (a leaf limb, a tail included).
fn leg_flags(c: &Creature) -> [bool; MAX_NODES] {
    let mut flags = [false; MAX_NODES];
    for limb in leaf_limbs(c) {
        for b in limb {
            flags[b] = true;
        }
    }
    flags
}

/// Whether `b` is a trunk bone: in no leg and not the neck.
fn is_trunk(c: &Creature, legs: &[bool; MAX_NODES], b: usize) -> bool {
    !legs[b] && !is_neck(c, b)
}

/// The trunk bones that hang from a bone that is in no leg: the joints of the
/// back, including the one at the base of the neck.
fn trunk_joints(c: &Creature, legs: &[bool; MAX_NODES]) -> BoneIds {
    let parents = parent_bones(c);
    (0..c.bones.len())
        .filter(|&j| is_trunk(c, legs, j))
        .filter(|&j| parents[c.bones[j].a as usize].is_some_and(|p| !legs[p]))
        .collect()
}

/// The trunk bones that hang from another trunk bone: the joints of the
/// spine proper, without the one at the base of the neck.
fn spine_joints(c: &Creature, legs: &[bool; MAX_NODES]) -> BoneIds {
    let parents = parent_bones(c);
    (0..c.bones.len())
        .filter(|&j| is_trunk(c, legs, j))
        .filter(|&j| parents[c.bones[j].a as usize].is_some_and(|p| is_trunk(c, legs, p)))
        .collect()
}

/// The driven muscles of the back: those with a stroke (`long` above `short`)
/// and both ends on trunk bones.
fn spine_muscles(c: &Creature, legs: &[bool; MAX_NODES]) -> MuscleIds {
    (0..c.muscles.len())
        .filter(|&i| {
            let m = &c.muscles[i];
            is_trunk(c, legs, m.bone_a as usize)
                && is_trunk(c, legs, m.bone_b as usize)
                && m.long > m.short
        })
        .collect()
}

/// A copy of the muscle that drives the legs, to time the back and the tail
/// by: the muscle with the most drive among those that have a stroke and an end
/// on a leg, or else the muscle with the most drive in the body.
fn leg_lead(c: &Creature, legs: &[bool; MAX_NODES]) -> Option<Muscle> {
    let on: MuscleIds = (0..c.muscles.len())
        .filter(|&i| {
            let m = &c.muscles[i];
            (legs[m.bone_a as usize] || legs[m.bone_b as usize]) && m.long > m.short
        })
        .collect();
    strongest(c, &on)
        .or_else(|| lead_muscle(c, &[]))
        .map(|i| c.muscles[i])
}

/// The direction (+1 or -1 along x) that points from the head to the rear.
fn rear_direction(c: &Creature) -> f32 {
    let others = &c.nodes[1..];
    if others.is_empty() {
        return -1.0;
    }
    let mean = others.iter().map(|n| n.x).sum::<f32>() / others.len() as f32;
    if c.nodes[0].x > mean { -1.0 } else { 1.0 }
}

/// The leaf limb that works as a tail: the first one whose tip lies within
/// 0.02 m of the rearmost node (the head does not count) and has its underside
/// more than 0.05 m above the ground.
fn tail_limb(c: &Creature) -> Option<BoneIds> {
    let dir = rear_direction(c);
    let rearmost = c.nodes[1..]
        .iter()
        .map(|n| n.x * dir)
        .fold(f32::MIN, f32::max);
    leaf_limbs(c).into_iter().find(|limb| {
        let t = c.nodes[c.bones[limb[limb.len() - 1]].b as usize];
        t.x * dir >= rearmost - 0.02 && t.y - 0.5 * t.diameter > 0.05
    })
}

/// Shifts the phase and the touchdown reset of every muscle of `group` by one
/// amount, so the strongest one lands on `target` and the others keep their
/// offsets from it. Returns whether anything moved. It does nothing for an
/// empty group, or when the strongest muscle is within 0.01 of a cycle of the
/// target.
fn retime_to(c: &mut Creature, group: &[usize], target: f32) -> bool {
    let Some(anchor) = strongest(c, group) else {
        return false;
    };
    let shift = turn(c.muscles[anchor].phase, target);
    if shift.abs() < 0.01 {
        return false;
    }
    shift_group(c, group, shift);
    true
}

/// Cuts bone `j` at `frac` of its length with a new node and a new bone
/// below it, and returns the new bone. The bone keeps its upper part and its
/// children stay on the lower node, so they now hang from the new bone. Every
/// muscle end and organ stays at the same point of the body, so the pose does
/// not change. The new joint starts with a range of 0.15 to 0.5 rad each way.
/// A touchdown sensor that would end up on the new node is dropped.
fn split_bone(c: &mut Creature, j: usize, frac: f32, rng: &mut Rng) -> usize {
    let old = c.bones[j];
    let (a, b) = (old.a as usize, old.b as usize);
    let at = add(pos(c, a), scale(sub(pos(c, b), pos(c, a)), frac));
    let mid = add_node(c, b, at);
    let new = c.bones.len();
    c.bones[j].b = mid as u32;
    c.bones[j].rest_length = old.rest_length * frac;
    c.bones.push(Bone {
        a: mid as u32,
        b: b as u32,
        rest_length: old.rest_length * (1.0 - frac),
        min_angle: -rng.range(0.15, 0.5),
        max_angle: rng.range(0.15, 0.5),
        organ_mass: 0.0,
        organ_at: 0.5,
    });
    if old.organ_mass > 0.0 && old.organ_at > frac {
        c.bones[new].organ_mass = old.organ_mass;
        c.bones[new].organ_at = ((old.organ_at - frac) / (1.0 - frac)).clamp(0.0, 1.0);
        c.bones[j].organ_mass = 0.0;
        c.bones[j].organ_at = 0.5;
    } else if old.organ_mass > 0.0 {
        c.bones[j].organ_at = (old.organ_at / frac).clamp(0.0, 1.0);
    }
    for m in c.muscles.iter_mut() {
        let originals = [m.bone_a as usize, m.bone_b as usize];
        let mut moved = [false; 2];
        #[allow(clippy::needless_range_loop)]
        for end in 0..2 {
            let (bone, anchor) = if end == 0 {
                (&mut m.bone_a, &mut m.anchor_a)
            } else {
                (&mut m.bone_b, &mut m.anchor_b)
            };
            if *bone as usize != j {
                continue;
            }
            if *anchor > frac {
                *bone = new as u32;
                *anchor = ((*anchor - frac) / (1.0 - frac)).clamp(0.0, 1.0);
                moved[end] = true;
            } else {
                *anchor = (*anchor / frac).clamp(0.0, 1.0);
            }
        }
        if m.sensor != NO_SENSOR {
            let end = (m.sensor / 2) as usize;
            let upper_end = m.sensor % 2 == 0;
            if end < 2 && originals[end] == j {
                let keeps = if moved[end] { !upper_end } else { upper_end };
                if !keeps {
                    m.sensor = NO_SENSOR;
                }
            }
        }
    }
    new
}

/// Adds a muscle across a random spine joint that no driven muscle bends. The
/// muscle is a copy of the strongest leg muscle with a gentle stroke around its
/// own span (`hinge_muscle`), and it runs in phase with that muscle or half a
/// cycle after it. A flexing back adds length to the stride of a galloping
/// mammal, because the hind legs reach farther forward while the fore legs
/// reach back (Hildebrand 1959).
pub(crate) fn spine_flex_muscle(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leg_flags(c);
    let joints = spine_joints(c, &legs);
    let Some(j) = pick(&joints, rng) else {
        return false;
    };
    let Some(lead) = leg_lead(c, &legs) else {
        return false;
    };
    let phase = lead.phase + if rng.unit() < 0.5 { 0.0 } else { 0.5 };
    hinge_muscle(c, cfg, j, &lead, phase.rem_euclid(1.0), rng)
}

/// Puts every driven muscle of the back on the period of the strongest leg
/// muscle and shifts their phases by one common amount, so that the strongest
/// of them runs in phase with the leg muscle, a quarter of a cycle after it or
/// half a cycle after it. The others keep their offsets from it. A back and
/// legs on one clock is the coupling that a central pattern generator gives an
/// animal (Ijspeert 2008), and it stops the back from beating against the
/// stride.
pub(crate) fn spine_lock_to_legs(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leg_flags(c);
    let back = spine_muscles(c, &legs);
    let Some(lead) = leg_lead(c, &legs) else {
        return false;
    };
    if back.is_empty() {
        return false;
    }
    let offset = [0.0, 0.25, 0.5][rng.index(3)];
    let before: Vec<(f32, f32)> = back
        .iter()
        .map(|&i| (c.muscles[i].period, c.muscles[i].phase))
        .collect();
    for &i in &back {
        c.muscles[i].period = lead.period;
    }
    retime_to(c, &back, (lead.phase + offset).rem_euclid(1.0));
    back.iter()
        .zip(before)
        .any(|(&i, b)| (c.muscles[i].period, c.muscles[i].phase) != b)
}

/// Cuts a trunk bone longer than 0.12 m in two, at 0.4 to 0.6 of its length,
/// so the back has one more joint. A muscle bends the new joint and runs 0.17
/// to 0.33 of a cycle behind the strongest leg muscle. A back of two or more
/// segments can arch and stretch (the cheetah's spine) where one rigid trunk
/// cannot (Sims 1994 built segmented bodies with a joint between segments).
/// The body gives back an idle limb tip if it has one (`shed_tips`), so the
/// creature stays as big as it was. Then `close_ring` closes the motor ring.
pub(crate) fn split_spine_bone(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    let legs = leg_flags(c);
    let long: BoneIds = (0..c.bones.len())
        .filter(|&b| is_trunk(c, &legs, b) && c.bones[b].rest_length > 0.12)
        .collect();
    let Some(j) = pick(&long, rng) else {
        return false;
    };
    let Some(lead) = leg_lead(c, &legs) else {
        return false;
    };
    let mut next = c.clone();
    let before = spans(&next);
    let new = split_bone(&mut next, j, rng.range(0.4, 0.6), rng);
    keep_strokes(&mut next, &before);
    let phase = lead.phase + rng.range(0.17, 0.33);
    if !hinge_muscle(&mut next, cfg, new, &lead, phase.rem_euclid(1.0), rng) {
        return false;
    }
    shed_tips(&mut next, c.nodes.len(), 1, rng);
    if !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Hangs a tail of two light bones from the rearmost trunk node that sits off
/// the ground. The tail points backward and up and its joints are narrow. A
/// muscle across each joint swings it half a cycle against the legs, and
/// three quarters of a cycle for the second joint. A swinging tail moves the
/// centre of mass against the leg thrust, as the tails of running lizards and
/// cheetahs do (Libby et al. 2012). The body gives back its idlest limb tip.
pub(crate) fn grow_counterweight_tail(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 2, 2) {
        return false;
    }
    let legs = leg_flags(c);
    let parents = parent_bones(c);
    let dir = rear_direction(c);
    let Some(root) = (1..c.nodes.len())
        .filter(|&n| parents[n].is_some_and(|p| is_trunk(c, &legs, p)) && c.nodes[n].y >= 0.15)
        .max_by(|&p, &q| (c.nodes[p].x * dir).total_cmp(&(c.nodes[q].x * dir)))
    else {
        return false;
    };
    let Some(lead) = leg_lead(c, &legs) else {
        return false;
    };
    let mean = c.bones.iter().map(|b| b.rest_length).sum::<f32>() / c.bones.len().max(1) as f32;
    let first_len = (mean * rng.range(0.7, 1.2)).clamp(0.06, max_bone_length());
    let second_len = (mean * rng.range(0.6, 1.1)).clamp(0.06, max_bone_length());
    let lift = rng.range(0.1, 0.6);
    let lift2 = (lift + rng.range(-0.4, 0.3)).clamp(-0.1, 0.9);
    let start = pos(c, root);
    let p1 = clamped(
        start[0] + dir * first_len * lift.cos(),
        start[1] + first_len * lift.sin(),
    );
    let p2 = clamped(
        p1[0] + dir * second_len * lift2.cos(),
        p1[1] + second_len * lift2.sin(),
    );
    let mut next = c.clone();
    let n1 = add_node(&mut next, root, p1);
    let n2 = add_node(&mut next, root, p2);
    for n in [n1, n2] {
        next.nodes[n].diameter = (next.nodes[n].diameter * 0.7).max(cfg.min_size);
    }
    let first = next.bones.len();
    for (a, b) in [(root, n1), (n1, n2)] {
        let d = sub(pos(&next, b), pos(&next, a));
        let mut bone = Bone::new(a as u32, b as u32, d[0].hypot(d[1]).max(0.03));
        bone.min_angle = -rng.range(0.3, 0.8);
        bone.max_angle = rng.range(0.3, 0.8);
        next.bones.push(bone);
    }
    let phase = lead.phase + 0.5;
    let before = next.muscles.len();
    hinge_muscle(&mut next, cfg, first, &lead, phase.rem_euclid(1.0), rng);
    hinge_muscle(
        &mut next,
        cfg,
        first + 1,
        &lead,
        (phase + 0.25).rem_euclid(1.0),
        rng,
    );
    if next.muscles.len() == before {
        return false;
    }
    shed_tips(&mut next, c.nodes.len(), 1, rng);
    if !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Makes the tip node of the tail heavier, 0.5 to 0.9 of the widest node a
/// body may have, so the tail swings as a counterweight. Mass at the end of a
/// long light lever gives the tail the most moment for the least extra body
/// mass (Libby et al. 2012 used a tail with a mass at its end).
pub(crate) fn weight_tail_tip(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(tail) = tail_limb(c) else {
        return false;
    };
    let tip = c.bones[tail[tail.len() - 1]].b as usize;
    let wanted = (cfg.max_size * rng.range(0.5, 0.9)).clamp(cfg.min_size, cfg.max_size);
    if wanted <= c.nodes[tip].diameter * 1.15 {
        return false;
    }
    c.nodes[tip].diameter = wanted;
    true
}

/// Turns the tail down and back until its tip reaches the ground, and braces
/// its root joint with a small flex, so the tail props the body like the tail
/// of a kangaroo. The body is not lifted by more than a few centimetres.
pub(crate) fn plant_tail_prop(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(tail) = tail_limb(c) else {
        return false;
    };
    let root = tail[0];
    let dir = rear_direction(c);
    let (from, to) = (
        pos(c, c.bones[root].a as usize),
        pos(c, c.bones[root].b as usize),
    );
    let current = (to[1] - from[1]).atan2(to[0] - from[0]);
    let below = rng.range(0.9, 1.3);
    let target = (-below.sin()).atan2(dir * below.cos());
    let mut turn_by = target - current;
    while turn_by > std::f32::consts::PI {
        turn_by -= std::f32::consts::TAU;
    }
    while turn_by < -std::f32::consts::PI {
        turn_by += std::f32::consts::TAU;
    }
    if turn_by.abs() < 0.15 {
        return false;
    }
    let mut next = c.clone();
    let before = spans(&next);
    turn_branch(&mut next, root, turn_by);
    if next.nodes[0].y - c.nodes[0].y > 0.06 {
        return false;
    }
    let flex = rng.range(0.05, 0.2);
    (next.bones[root].min_angle, next.bones[root].max_angle) = (-flex, flex);
    keep_strokes(&mut next, &before);
    c.clone_from(&next);
    true
}

/// Shifts the tail's muscles to run half a cycle against the strongest leg
/// muscle, on the same period. A tail that swings opposite to the legs takes
/// up the angular momentum of the stride, as in the running lizards of Libby
/// et al. (2012), where tail and body counter-rotate.
pub(crate) fn tail_swing_against_legs(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leg_flags(c);
    let Some(tail) = tail_limb(c) else {
        return false;
    };
    let Some(lead) = leg_lead(c, &legs) else {
        return false;
    };
    let group = muscles_on(c, &tail, false);
    if group.is_empty() {
        return false;
    }
    let before: Vec<f32> = group.iter().map(|&i| c.muscles[i].period).collect();
    for &i in &group {
        c.muscles[i].period = lead.period;
    }
    let moved = retime_to(c, &group, (lead.phase + 0.5).rem_euclid(1.0));
    moved
        || group
            .iter()
            .zip(before)
            .any(|(&i, p)| c.muscles[i].period != p)
}

/// Gives the neck a muscle across its base joint, timed a quarter or half a
/// cycle after the strongest leg muscle, so the head bobs with the stride.
/// Horses and pigeons move the head against the legs to keep the centre of
/// mass over the feet, and a swinging head is a counterweight at the end of a
/// long lever.
pub(crate) fn neck_bob_muscle(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(neck) = (0..c.bones.len()).find(|&b| is_neck(c, b)) else {
        return false;
    };
    let base = c.bones[neck].a.max(c.bones[neck].b) as usize;
    let legs = leg_flags(c);
    let roots: BoneIds = (0..c.bones.len())
        .filter(|&b| is_trunk(c, &legs, b) && c.bones[b].a as usize == base)
        .collect();
    let Some(root) = pick(&roots, rng) else {
        return false;
    };
    let Some(lead) = leg_lead(c, &legs) else {
        return false;
    };
    let phase = lead.phase + if rng.unit() < 0.5 { 0.25 } else { 0.5 };
    hinge_muscle(c, cfg, root, &lead, phase.rem_euclid(1.0), rng)
}

/// Cuts the neck in two, so the head sits on a neck with two joints, and
/// bends the new joint with a muscle that runs a quarter of a cycle behind the
/// legs. The head stays the head and the neck stays attached to it. Two neck
/// joints let the head move against the trunk with little change in the
/// height of the shoulders, as the long neck of a giraffe or a heron does.
/// The body gives back its idlest limb tip.
pub(crate) fn split_neck_bone(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    let Some(neck) = (0..c.bones.len()).find(|&b| is_neck(c, b)) else {
        return false;
    };
    if c.bones[neck].a != 0 || c.bones[neck].rest_length < 0.1 {
        return false;
    }
    let legs = leg_flags(c);
    let Some(lead) = leg_lead(c, &legs) else {
        return false;
    };
    let mut next = c.clone();
    let before = spans(&next);
    let new = split_bone(&mut next, neck, rng.range(0.4, 0.6), rng);
    keep_strokes(&mut next, &before);
    let phase = (lead.phase + 0.25).rem_euclid(1.0);
    if !hinge_muscle(&mut next, cfg, new, &lead, phase, rng) {
        return false;
    }
    shed_tips(&mut next, c.nodes.len(), 1, rng);
    if !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Narrows every trunk joint with a stop more than 0.05 rad beyond a flex of
/// 0.05 to 0.15 rad, so that neither of its stops lies beyond that flex. If a
/// joint narrowed, it also shortens the stroke of every muscle between a trunk
/// joint and a bone in no leg to 40% of what it was, about its midpoint. The
/// trunk becomes one near-rigid frame that the legs move. Many fast animals
/// with a short stride hold the trunk still and let the legs do the work
/// (Cheney et al. 2013 found rigid, regular bodies move well).
pub(crate) fn stiffen_trunk_joints(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leg_flags(c);
    let joints = trunk_joints(c, &legs);
    let flex = rng.range(0.05, 0.15);
    let mut changed = false;
    for &j in &joints {
        let bone = &mut c.bones[j];
        if bone.max_angle > flex + 0.05 || bone.min_angle < -flex - 0.05 {
            bone.min_angle = bone.min_angle.max(-flex);
            bone.max_angle = bone.max_angle.min(flex);
            changed = true;
        }
    }
    if !changed {
        return false;
    }
    let across: MuscleIds = (0..c.muscles.len())
        .filter(|&i| {
            let m = &c.muscles[i];
            let (a, b) = (m.bone_a as usize, m.bone_b as usize);
            (joints.contains(&a) && !legs[b]) || (joints.contains(&b) && !legs[a])
        })
        .collect();
    for &i in &across {
        let m = &mut c.muscles[i];
        let (mid, half) = (0.5 * (m.short + m.long), 0.2 * (m.long - m.short));
        m.short = mid - half;
        m.long = mid + half;
    }
    true
}

/// Widens every trunk joint with a stop more than 0.05 rad inside a flex of 0.5
/// to 0.9 rad, so that both of its stops reach at least that flex. It also
/// gives a random trunk joint a muscle if no driven muscle bends it, timed half
/// a cycle after the strongest leg muscle. A loose trunk can arch and stretch
/// with the stride (a bounding weasel or a galloping horse), which a stiff one
/// cannot.
pub(crate) fn loosen_trunk_joints(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leg_flags(c);
    let joints = trunk_joints(c, &legs);
    let Some(driven) = pick(&joints, rng) else {
        return false;
    };
    let flex = rng.range(0.5, 0.9).min(JOINT_LIMIT);
    let mut changed = false;
    for &j in &joints {
        let bone = &mut c.bones[j];
        if bone.max_angle < flex - 0.05 || bone.min_angle > -flex + 0.05 {
            bone.min_angle = bone.min_angle.min(-flex);
            bone.max_angle = bone.max_angle.max(flex);
            changed = true;
        }
    }
    if let Some(lead) = leg_lead(c, &legs) {
        let phase = (lead.phase + 0.5).rem_euclid(1.0);
        changed |= hinge_muscle(c, cfg, driven, &lead, phase, rng);
    }
    changed
}

/// Orders the driven muscles of the back by the distance from the head to the
/// node their two bones share, gives them the period of the first, and steps
/// their phase from its phase by 0.08 to 0.25 of a cycle from one to the next,
/// toward the tail or toward the head. The result is a travelling wave of
/// bending along the back, as in the swimming of a fish or the trot of a
/// salamander (Ijspeert 2008, a chain of coupled oscillators along the spine).
/// It needs two driven spine muscles.
pub(crate) fn spine_phase_wave(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leg_flags(c);
    let mut back = spine_muscles(c, &legs);
    if back.len() < 2 {
        return false;
    }
    let head = pos(c, 0);
    let distance = |c: &Creature, i: usize| -> f32 {
        let m = &c.muscles[i];
        shared_node(c, m.bone_a as usize, m.bone_b as usize).map_or(0.0, |n| {
            let d = sub(pos(c, n as usize), head);
            d[0].hypot(d[1])
        })
    };
    back.sort_stable_by(|&x, &y| distance(c, x).total_cmp(&distance(c, y)));
    let step = rng.range(0.08, 0.25) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    let (period, base) = (c.muscles[back[0]].period, c.muscles[back[0]].phase);
    for (k, &i) in back.iter().enumerate() {
        let m = &mut c.muscles[i];
        let target = (base + step * k as f32).rem_euclid(1.0);
        let shift = turn(m.phase, target);
        m.phase = target;
        m.reset = (m.reset + shift).rem_euclid(1.0);
        m.period = period;
    }
    true
}

/// Gives the driven muscles of the back an elastic tendon of 0.3 to 0.8, so
/// the back stores the energy of a stretch and returns it. A muscle whose
/// tendon is already within 0.1 of that, or stiffer, keeps it. The cheetah and
/// the horse recover a large part of the stride's energy in the spring-like
/// back and its tendons (Alexander 1988), and an elastic back needs less muscle
/// work for the same flex.
pub(crate) fn elastic_spine(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leg_flags(c);
    let back = spine_muscles(c, &legs);
    let tendon = rng.range(0.3, 0.8);
    let mut changed = false;
    for &i in &back {
        if c.muscles[i].tendon < tendon - 0.1 {
            c.muscles[i].tendon = tendon;
            changed = true;
        }
    }
    changed
}

/// Starts a random trunk joint bent by 0.08 to 0.3 rad toward a hump (the part
/// below the joint tips down), and moves its stops with it, so the joint can
/// still reach every pose it could before, within the joint limit. A back that
/// starts arched is loaded like a spring and stretches out as the body extends
/// (the gather phase of the bounding gait). Only joints that keep the starting
/// pose at least 0.02 rad inside both stops are used.
pub(crate) fn arch_back(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leg_flags(c);
    let joints = trunk_joints(c, &legs);
    let Some(j) = pick(&joints, rng) else {
        return false;
    };
    let amount = rng.range(0.08, 0.3);
    let pivot = pos(c, c.bones[j].a as usize);
    let below = branch_nodes(c, &branch(c, j));
    if below.is_empty() {
        return false;
    }
    let mean_after = |angle: f32| -> f32 {
        let (sin, cos) = angle.sin_cos();
        below
            .iter()
            .map(|&n| {
                let d = sub(pos(c, n), pivot);
                d[0] * sin + d[1] * cos
            })
            .sum::<f32>()
            / below.len() as f32
    };
    let angle = if mean_after(amount) < mean_after(-amount) {
        amount
    } else {
        -amount
    };
    let bone = c.bones[j];
    let (min, max) = (bone.min_angle - angle, bone.max_angle - angle);
    if min > -0.02 || max < 0.02 {
        return false;
    }
    let before = spans(c);
    turn_branch(c, j, angle);
    c.bones[j].min_angle = min;
    c.bones[j].max_angle = max;
    keep_strokes(c, &before);
    true
}
