//! Gait operators for whole body plans, where a plan is a leg count, a leg
//! spacing along the trunk and a timing among the legs, and the three have to
//! change together or the gait breaks. Some operators build a named plan
//! (quadruped, hexapod, myriapod, hopper, swinger, biped, counterweight
//! runner), some add, shed, fuse or split legs, and two give legs the timing or
//! the proportions of hoofed runners. The operators of this file share one pick
//! slot (`GAIT_FILES` in `mod.rs`) and are compound, so a child gets no
//! parameter noise after one. The ideas come from Sims (1994) and Lipson and
//! Pollack (2000) on mirrored leg pairs, Alexander on gait timing and duty
//! factor, and Full and Koditschek (1999) on the spring-mass template that runs
//! under every plan.
use super::compound::{close_ring, hinge_muscle, limb_phase, shed_tips, shift_group, strongest};
use super::extra::limb_drive;
use super::junctions::{add_node, keep_strokes, lift, shift_branch, spans, turn_branch};
use super::limbs::clamped;
use super::rhythm::{foot, leaf_limbs, muscle_groups};
use super::{
    BoneIds, Context, Limbs, Operator, branch, branch_nodes, copy_branch_limited, muscles_on,
    parent_bones, remove_parts, room,
};
use crate::config::Config;
use crate::evolution::{Bone, Creature, JOINT_LIMIT, Muscle, Rng, max_bone_length};
use std::f32::consts::PI;

/// This file's operators, by name, for `GAIT_FILES` in `mod.rs`. Add each new
/// one here. Their order is part of the search, because a pick draws the n-th
/// entry of the list.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("quadruped_plan", quadruped_plan),
    ("hexapod_tripod", hexapod_tripod),
    ("kangaroo_hopper", kangaroo_hopper),
    ("myriapod_wave", myriapod_wave),
    ("gibbon_swinger", gibbon_swinger),
    ("counterweight_runner", counterweight_runner),
    ("shed_leg_pair", shed_leg_pair),
    ("append_leg_pair", append_leg_pair),
    ("fuse_legs_into_one", fuse_legs_into_one),
    ("split_leg_in_two", split_leg_in_two),
    ("reduce_to_biped", reduce_to_biped),
    ("pronking_stot", pronking_stot),
    ("unguligrade_legs", unguligrade_legs),
];

/// The x coordinate of the hip where `limb` attaches.
fn hip_x(c: &Creature, limb: &[usize]) -> f32 {
    c.nodes[c.bones[limb[0]].a as usize].x
}

/// The legs of a plan, called walkers: the leaf limbs with at most three bones
/// and a tip no more than 0.35 of the body's height plus 0.05 above the lowest
/// tip. The list is ordered by the x of the hip and then of the foot, so the
/// rearmost leg is first.
fn walkers(c: &Creature) -> Limbs {
    let all = leaf_limbs(c);
    let tip_y = |l: &BoneIds| c.nodes[foot(c, l)].y;
    let low = all.iter().map(tip_y).fold(f32::MAX, f32::min);
    let top = c.nodes.iter().map(|n| n.y).fold(0.0, f32::max);
    let mut out: Limbs = all
        .into_iter()
        .filter(|l| l.len() <= 3 && tip_y(l) <= low + 0.35 * top + 0.05)
        .collect();
    out.sort_stable_by(|p, q| {
        hip_x(c, p)
            .total_cmp(&hip_x(c, q))
            .then(c.nodes[foot(c, p)].x.total_cmp(&c.nodes[foot(c, q)].x))
    });
    out
}

/// The nodes in no leaf limb, the head excluded. The hip of a leg is one.
fn trunk_nodes(c: &Creature) -> BoneIds {
    let inside: BoneIds = leaf_limbs(c)
        .iter()
        .flat_map(|l| branch_nodes(c, l))
        .collect();
    (1..c.nodes.len()).filter(|n| !inside.contains(n)).collect()
}

/// Moves the muscles of each leg of `legs` (phase and touchdown reset
/// together) so that the strongest muscle of leg `i` runs at phase
/// `base + pattern(i)`. A leg with no muscle is skipped, and a muscle on two
/// legs belongs to the first.
fn retime(c: &mut Creature, legs: &[BoneIds], base: f32, pattern: impl Fn(usize) -> f32) {
    let groups = muscle_groups(c, legs);
    for (i, group) in groups.iter().enumerate() {
        let Some(anchor) = strongest(c, group) else {
            continue;
        };
        let shift = base + pattern(i) - c.muscles[anchor].phase;
        shift_group(c, group, shift);
    }
}

/// The phase offset of leg `i`, rear first, in a gait of leg pairs. The two
/// legs of a pair are half a cycle apart and the next pair has them swapped,
/// so diagonal legs step together: a trot for four legs, a tripod gait for six.
fn alternate(i: usize) -> f32 {
    0.5 * ((i / 2 + i % 2) % 2) as f32
}

/// Scales the leg `limb` about its hip by `factor`. Its nodes move and the rest
/// length of each bone scales, within 0.03 and `max_bone_length()`. The caller
/// lifts the body, clamps the nodes and keeps the strokes (`scale_legs`).
fn scale_leg(c: &mut Creature, limb: &[usize], factor: f32) {
    let pivot = c.nodes[c.bones[limb[0]].a as usize];
    for n in branch_nodes(c, limb) {
        let node = &mut c.nodes[n];
        node.x = pivot.x + (node.x - pivot.x) * factor;
        node.y = pivot.y + (node.y - pivot.y) * factor;
    }
    for &b in limb {
        let bone = &mut c.bones[b];
        bone.rest_length = (bone.rest_length * factor).clamp(0.03, max_bone_length());
    }
}

/// Scales each leg of `limbs` about its hip by `factor`. It raises the body if
/// a node went below the ground, clamps the nodes into the start region and
/// scales every muscle's stroke by how much its span changed.
fn scale_legs(c: &mut Creature, limbs: &[BoneIds], factor: f32) {
    let before = spans(c);
    for limb in limbs {
        scale_leg(c, limb, factor);
    }
    lift(c);
    for n in &mut c.nodes {
        [n.x, n.y] = clamped(n.x, n.y);
    }
    keep_strokes(c, &before);
}

/// Multiplies the stiffness of the active muscles on `limb` by `stiffness`
/// (within 1 to 120) and raises their tendon to at least `tendon`. A muscle
/// with no stroke is left alone.
fn tune_leg(c: &mut Creature, limb: &[usize], stiffness: f32, tendon: f32) {
    for i in muscles_on(c, limb, false) {
        let m = &mut c.muscles[i];
        if m.long > m.short {
            m.stiffness = (m.stiffness * stiffness).clamp(1.0, 120.0);
            m.tendon = m.tendon.max(tendon).clamp(0.0, 1.0);
        }
    }
}

/// A trunk node to hang a leg copied from the hip `from`: the best of three
/// random ones, with few legs on it, far in x from the nearest hip (up to 1)
/// and near the height of `from`. With `other` set, `from` itself is not a
/// candidate. It is `None` when there is no candidate.
fn pick_site(
    c: &Creature,
    legs: &[BoneIds],
    from: usize,
    rng: &mut Rng,
    other: bool,
) -> Option<usize> {
    let hips: BoneIds = legs.iter().map(|l| c.bones[l[0]].a as usize).collect();
    let sites: BoneIds = trunk_nodes(c)
        .into_iter()
        .filter(|&n| !other || n != from)
        .collect();
    if sites.is_empty() {
        return None;
    }
    let score = |n: usize| {
        let here = hips.iter().filter(|&&h| h == n).count() as f32;
        let gap = hips
            .iter()
            .map(|&h| (c.nodes[h].x - c.nodes[n].x).abs())
            .fold(f32::MAX, f32::min)
            .min(1.0);
        here - 3.0 * gap + 2.0 * (c.nodes[n].y - c.nodes[from].y).abs()
    };
    (0..3)
        .map(|_| sites[rng.index(sites.len())])
        .min_by(|&p, &q| score(p).total_cmp(&score(q)))
}

/// Copies the leg `limb` so that its hip sits on node `at`, reflected about the
/// vertical through the hip when `mirror` is set. The copy brings at most three
/// muscles, the ones with the most drive, and their phases move by `phase`
/// cycles. Returns the new root bone, or `None` when there is no room.
fn copy_leg(
    c: &mut Creature,
    cfg: &Config,
    limb: &[usize],
    at: usize,
    mirror: bool,
    phase: f32,
) -> Option<usize> {
    let root = limb[0];
    let (f, t) = (c.nodes[c.bones[root].a as usize], c.nodes[at]);
    let place = |[x, y]: [f32; 2]| {
        let dx = if mirror { f.x - x } else { x - f.x };
        [t.x + dx, t.y + (y - f.y)]
    };
    copy_branch_limited(c, cfg, root, at, place, mirror, phase, 3)
}

/// Copies random walkers onto the trunk until there are `target` of them, in at
/// most 8 copies. It stops at the first copy that does not fit. A copy onto the
/// hip of its source is reflected, and any other copy is reflected half the
/// time. The copies keep the phase of their source, because the caller retimes
/// the legs.
fn grow_legs(c: &mut Creature, cfg: &Config, rng: &mut Rng, target: usize) {
    for _ in 0..8 {
        let legs = walkers(c);
        if legs.is_empty() || legs.len() >= target {
            return;
        }
        let limb = &legs[rng.index(legs.len())];
        let from = c.bones[limb[0]].a as usize;
        let Some(at) = pick_site(c, &legs, from, rng, false) else {
            return;
        };
        let mirror = at == from || rng.unit() < 0.5;
        if copy_leg(c, cfg, limb, at, mirror, 0.0).is_none() {
            return;
        }
    }
}

/// Puts the leg count and timing of a plan on the body. It grows the walkers to
/// `target`, sheds idle limb tips of the old body (`shed_tips`), up to half as
/// many as the nodes it added, retimes the walkers by `pattern` from the phase
/// of the rearmost, and commits the child. It returns false when the body has
/// no walker, when fewer than `keep` are left after the growth, when the
/// rearmost has no muscle, or when the child does not differ.
fn plan_legs(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    target: usize,
    keep: usize,
    pattern: impl Fn(usize) -> f32,
) -> bool {
    if walkers(c).is_empty() {
        return false;
    }
    let mut next = c.clone();
    let base = next.nodes.len();
    grow_legs(&mut next, cfg, rng, target);
    let added = next.nodes.len() - base;
    shed_tips(&mut next, base, added / 2, rng);
    let legs = walkers(&next);
    if legs.len() < keep {
        return false;
    }
    let Some(start) = limb_phase(&next, &legs[0]) else {
        return false;
    };
    retime(&mut next, &legs, start, pattern);
    commit(c, next, cfg, rng)
}

/// The last step of an operator. It closes the motor ring of the child `next`
/// (`close_ring`, which puts the bones in canonical order) and copies `next`
/// into `c` if it differs. Returns whether it did.
fn commit(c: &mut Creature, mut next: Creature, cfg: &Config, rng: &mut Rng) -> bool {
    if !close_ring(&mut next, cfg, rng) {
        return false;
    }
    if next.nodes == c.nodes && next.bones == c.bones && next.muscles == c.muscles {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Hangs a swinging bone from node `site`. The bone is `length` long and points
/// `angle` radians from the +x axis (its tip is clamped into the start region).
/// It has a joint range of 0.4 to 0.9 rad each way and carries a weight of
/// `mass` at 0.85 of its length. A hinge muscle with the rhythm of `template`
/// swings it at phase `phase`. Returns false when there is no room for one node
/// and two muscles, or when `site` is the head, which has no bone above it.
#[allow(clippy::too_many_arguments)]
fn add_swinger(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    site: usize,
    angle: f32,
    length: f32,
    mass: f32,
    phase: f32,
    template: &Muscle,
) -> bool {
    if !room(c, cfg, 1, 2) || parent_bones(c)[site].is_none() {
        return false;
    }
    let s = c.nodes[site];
    let [x, y] = clamped(s.x + length * angle.cos(), s.y + length * angle.sin());
    let tip = add_node(c, site, [x, y]);
    let (dx, dy) = (x - s.x, y - s.y);
    let mut bone = Bone::new(site as u32, tip as u32, dx.hypot(dy).max(0.05));
    bone.min_angle = -rng.range(0.4, 0.9);
    bone.max_angle = rng.range(0.4, 0.9);
    bone.organ_mass = mass;
    bone.organ_at = 0.85;
    c.bones.push(bone);
    hinge_muscle(c, cfg, c.bones.len() - 1, template, phase, rng)
}

/// The mean rest length of the bones of `legs`, or 0.3 when they have none.
fn mean_bone(c: &Creature, legs: &[BoneIds]) -> f32 {
    let (sum, n) = legs
        .iter()
        .flat_map(|l| l.iter())
        .fold((0.0, 0), |(s, n), &b| (s + c.bones[b].rest_length, n + 1));
    if n == 0 { 0.3 } else { sum / n as f32 }
}

/// A copy of the strongest muscle on `legs`, as a template for a new one. It is
/// `None` when the legs have no muscle.
fn template_of(c: &Creature, legs: &[BoneIds]) -> Option<Muscle> {
    let on: BoneIds = legs.iter().flat_map(|l| l.iter().copied()).collect();
    let muscles = muscles_on(c, &on, false);
    strongest(c, &muscles).map(|i| c.muscles[i])
}

/// Turns the body into a quadruped. Legs are copied onto the trunk, spread
/// along it, until there are four. Then the legs take one of four gaits at
/// random: a trot (diagonals together), a walk (a quarter cycle between the
/// legs in turn), a bound (pairs together) or a rotary gallop. A body with four
/// legs or more only takes the new timing. It fails if fewer than three legs
/// result.
pub(crate) fn quadruped_plan(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let gait = rng.index(4);
    plan_legs(c, cfg, rng, 4, 3, |i| match gait {
        0 => alternate(i),
        1 => 0.5 * (i % 2) as f32 + 0.25 * ((i / 2) % 2) as f32,
        2 => 0.4 * ((i / 2) % 2) as f32,
        _ => 0.12 * (i % 2) as f32 + 0.45 * ((i / 2) % 2) as f32,
    })
}

/// Turns the body into a hexapod. Legs are copied onto the trunk until there
/// are six, and the legs step as two tripods. Insects hold three legs on the
/// ground at every moment, so the body is always statically stable, which is
/// why the tripod gait is a safe place to start. A body with six legs or more
/// only takes the new timing. It fails if fewer than five legs result.
pub(crate) fn hexapod_tripod(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    plan_legs(c, cfg, rng, 6, 5, alternate)
}

/// Turns the body into a many-legged one, a myriapod. Legs are copied until
/// there are eight (when the room allows) and the legs step in a wave. Each
/// pair is a fixed step of 0.08 to 0.2 of a cycle later than the pair behind
/// it, or earlier in a random half of the calls, and the two legs of a pair are
/// half a cycle apart. This is the metachronal wave of a centipede, and the
/// phase lag of a central pattern generator down a chain. It fails if fewer
/// than four legs result.
pub(crate) fn myriapod_wave(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let step = rng.range(0.08, 0.2) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    plan_legs(c, cfg, rng, 8, 4, |i| {
        (i / 2) as f32 * step + 0.5 * (i % 2) as f32
    })
}

/// Turns the body into a hopper like a kangaroo. The two rearmost legs grow
/// 1.15 to 1.4 times longer, get muscles 1.1 to 1.4 times stiffer and a tendon
/// raised to 0.4 to 0.9 (the Achilles tendon of a hopper stores and returns the
/// energy of a landing), and push together. Any other legs shrink to 0.6 to
/// 0.85 and step a quarter cycle off. A tail grows backward from the rearmost
/// trunk node higher than 0.05. It is 0.8 to 1.4 times the mean bone length of
/// the rear legs, swings half a cycle from the hop and has a weight near its
/// end, to balance the body. One idle limb tip of the old body goes, so the body
/// does not grow. It fails if the body has fewer than two legs or no room for
/// the tail.
pub(crate) fn kangaroo_hopper(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = walkers(c);
    if legs.len() < 2 || !room(c, cfg, 1, 2) {
        return false;
    }
    let (Some(start), Some(template)) = (limb_phase(c, &legs[0]), template_of(c, &legs[..2]))
    else {
        return false;
    };
    let mut next = c.clone();
    let base = next.nodes.len();
    let length = mean_bone(c, &legs[..2]);
    scale_legs(&mut next, &legs[..2], rng.range(1.15, 1.4));
    scale_legs(&mut next, &legs[2..], rng.range(0.6, 0.85));
    for hind in &legs[..2] {
        tune_leg(&mut next, hind, rng.range(1.1, 1.4), rng.range(0.4, 0.9));
    }
    retime(&mut next, &legs[..2], start, |_| 0.0);
    retime(&mut next, &legs[2..], start, |_| 0.25);
    let Some(site) = trunk_nodes(&next)
        .into_iter()
        .filter(|&n| next.nodes[n].y > 0.05)
        .min_by(|&p, &q| next.nodes[p].x.total_cmp(&next.nodes[q].x))
    else {
        return false;
    };
    // Backward, from 0.1 rad above the horizontal to 0.3 rad below it.
    let angle = PI + rng.range(-0.1, 0.3);
    let mass = rng.range(0.05, 0.12);
    let reach = length * rng.range(0.8, 1.4);
    if !add_swinger(
        &mut next,
        cfg,
        rng,
        site,
        angle,
        reach,
        mass,
        start + 0.5,
        &template,
    ) {
        return false;
    }
    shed_tips(&mut next, base, 1, rng);
    commit(c, next, cfg, rng)
}

/// Turns the body into a swinger like a gibbon walking upright. The two
/// foremost legs become long arms, 1.3 to 1.6 times longer, with joint ranges
/// 1.35 times wider (within the joint limit) and muscle strokes 1.3 times
/// longer, and they swing hand over hand, half a cycle apart. Long pendulum
/// limbs swing at a low natural frequency, so each swing covers a long stride.
/// It fails if the body has fewer than two legs.
pub(crate) fn gibbon_swinger(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = walkers(c);
    if legs.len() < 2 {
        return false;
    }
    let arms = &legs[legs.len() - 2..];
    let Some(start) = limb_phase(c, &arms[0]) else {
        return false;
    };
    let mut next = c.clone();
    scale_legs(&mut next, arms, rng.range(1.3, 1.6));
    for arm in arms {
        for &b in arm.iter() {
            let bone = &mut next.bones[b];
            bone.min_angle = (bone.min_angle * 1.35).max(-JOINT_LIMIT);
            bone.max_angle = (bone.max_angle * 1.35).min(JOINT_LIMIT);
        }
        for i in muscles_on(&next, arm, false) {
            let m = &mut next.muscles[i];
            m.long = m.short + (m.long - m.short) * 1.3;
        }
    }
    retime(&mut next, arms, start, |i| 0.5 * i as f32);
    commit(c, next, cfg, rng)
}

/// Turns the body into a two-legged runner with a counterweight. The two
/// longest legs step half a cycle apart, and the other legs keep their timing.
/// A short arm with a weight at its end grows from the highest trunk node. It
/// is 0.6 to 0.9 times the mean bone length of the two legs and swings half a
/// cycle from the rearmost of them, so that its angular momentum works against
/// the swing of the legs, as the arms of a running person do. One idle limb tip
/// of the old body goes, so the body does not grow. It fails if the body has
/// fewer than two legs or no room for the arm.
pub(crate) fn counterweight_runner(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut legs = walkers(c);
    if legs.len() < 2 || !room(c, cfg, 1, 2) {
        return false;
    }
    let total = |l: &BoneIds| l.iter().map(|&b| c.bones[b].rest_length).sum::<f32>();
    // The two longest legs, the rearmost first.
    legs.sort_stable_by(|p, q| total(q).total_cmp(&total(p)));
    let mut pair = [legs[0], legs[1]];
    pair.sort_by(|p, q| hip_x(c, p).total_cmp(&hip_x(c, q)));
    let (Some(start), Some(template)) = (limb_phase(c, &pair[0]), template_of(c, &pair)) else {
        return false;
    };
    let mut next = c.clone();
    let base = next.nodes.len();
    retime(&mut next, &pair, start, |i| 0.5 * i as f32);
    let Some(site) = trunk_nodes(&next)
        .into_iter()
        .max_by(|&p, &q| next.nodes[p].y.total_cmp(&next.nodes[q].y))
    else {
        return false;
    };
    // Forward, from 0.2 rad below the horizontal to 0.7 rad above it.
    let angle = rng.range(-0.2, 0.7);
    let mass = rng.range(0.08, 0.2);
    let length = mean_bone(c, &pair) * rng.range(0.6, 0.9);
    if !add_swinger(
        &mut next,
        cfg,
        rng,
        site,
        angle,
        length,
        mass,
        start + 0.5,
        &template,
    ) {
        return false;
    }
    shed_tips(&mut next, base, 1, rng);
    commit(c, next, cfg, rng)
}

/// Drops a pair of neighbouring legs from a body with four or more legs and
/// seven or more nodes. Two pairs are tried and the one with the least muscle
/// drive goes. The remaining legs are retimed as one alternating gait. Fewer
/// legs cost fewer nodes, and a plan with fewer legs keeps its stability only
/// if the rest alternate.
pub(crate) fn shed_leg_pair(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = walkers(c);
    if legs.len() < 4 || c.nodes.len() < 7 {
        return false;
    }
    let pick = |rng: &mut Rng| rng.index(legs.len() - 1);
    let pair_drive = |i: usize| limb_drive(c, &legs[i]) + limb_drive(c, &legs[i + 1]);
    let (p, q) = (pick(rng), pick(rng));
    let i = if pair_drive(p) <= pair_drive(q) { p } else { q };
    let bones: BoneIds = legs[i].iter().chain(legs[i + 1].iter()).copied().collect();
    let nodes: BoneIds = branch_nodes(c, &bones);
    let mut next = c.clone();
    remove_parts(&mut next, &bones, &nodes);
    let left = walkers(&next);
    if left.len() < 2 {
        return false;
    }
    let start = limb_phase(&next, &left[0]).unwrap_or(0.0);
    retime(&mut next, &left, start, alternate);
    commit(c, next, cfg, rng)
}

/// Adds a pair of legs at another trunk node: two copies of a random leg, one
/// of them reflected (chosen at random). The first copy steps half a cycle from
/// the leg and the second in its phase, so the two copies are half a cycle
/// apart. The body gains a segment of two legs, as a hexapod or a centipede
/// does. Up to half as many older limb tips as new nodes go.
pub(crate) fn append_leg_pair(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = walkers(c);
    if legs.is_empty() {
        return false;
    }
    let limb = &legs[rng.index(legs.len())];
    if !room(c, cfg, 2 * limb.len(), 0) {
        return false;
    }
    let from = c.bones[limb[0]].a as usize;
    let Some(at) = pick_site(c, &legs, from, rng, true) else {
        return false;
    };
    let mut next = c.clone();
    let base = next.nodes.len();
    let mirror = rng.unit() < 0.5;
    if copy_leg(&mut next, cfg, limb, at, mirror, 0.5).is_none()
        || copy_leg(&mut next, cfg, limb, at, !mirror, 0.0).is_none()
    {
        return false;
    }
    let added = next.nodes.len() - base;
    shed_tips(&mut next, base, added / 2, rng);
    commit(c, next, cfg, rng)
}

/// Fuses two neighbouring legs into one strong leg. Two pairs of neighbours are
/// tried and the pair whose hips are nearer in x is fused. The leg of the pair
/// with more drive stays, 1.15 to 1.3 times longer with its muscles 1.3 to 1.6
/// times stiffer, and the other goes. Needs three or more legs. One thick leg
/// has the force of two without the bones and joints of the second, as the
/// single toe of a horse does.
pub(crate) fn fuse_legs_into_one(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = walkers(c);
    if legs.len() < 3 {
        return false;
    }
    let gap = |i: usize| (hip_x(c, &legs[i + 1]) - hip_x(c, &legs[i])).abs();
    let (p, q) = (rng.index(legs.len() - 1), rng.index(legs.len() - 1));
    let i = if gap(p) <= gap(q) { p } else { q };
    let (keep, drop) = if limb_drive(c, &legs[i]) >= limb_drive(c, &legs[i + 1]) {
        (&legs[i], &legs[i + 1])
    } else {
        (&legs[i + 1], &legs[i])
    };
    let mut next = c.clone();
    scale_legs(&mut next, std::slice::from_ref(keep), rng.range(1.15, 1.3));
    tune_leg(&mut next, keep, rng.range(1.3, 1.6), 0.0);
    let nodes: BoneIds = branch_nodes(c, drop);
    remove_parts(&mut next, drop, &nodes);
    commit(c, next, cfg, rng)
}

/// Splits a leg into two at the same hip. The copy is reflected. The original
/// turns counterclockwise and the copy clockwise, each by 0.12 to 0.3 rad, so
/// legs that hang down spread apart. Every muscle on the original, and every
/// copied muscle whose first bone is the copy's root, becomes three quarters as
/// stiff. The two legs step together or half a cycle apart. Two thin legs under
/// one hip widen the stance and can alternate where one leg could not. Up to
/// half as many older limb tips as new nodes go.
pub(crate) fn split_leg_in_two(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = walkers(c);
    if legs.is_empty() {
        return false;
    }
    let limb = &legs[rng.index(legs.len())];
    if !room(c, cfg, limb.len(), 3) {
        return false;
    }
    let root = limb[0];
    let at = c.bones[root].a as usize;
    let phase = if rng.unit() < 0.5 { 0.5 } else { 0.0 };
    let lean = rng.range(0.12, 0.3);
    let mut next = c.clone();
    let base = next.nodes.len();
    let before = next.muscles.len();
    let Some(copy) = copy_leg(&mut next, cfg, limb, at, true, phase) else {
        return false;
    };
    let pose = spans(&next);
    turn_branch(&mut next, root, lean);
    turn_branch(&mut next, copy, -lean);
    keep_strokes(&mut next, &pose);
    let mut both = BoneIds::from_slice(limb);
    both.extend(branch(&next, copy));
    // The muscles before `before` are the original's. A copied muscle counts
    // when it starts at the copy's root bone.
    for i in muscles_on(&next, &both, false) {
        if i < before || next.muscles[i].bone_a as usize == copy {
            let m = &mut next.muscles[i];
            m.stiffness = (m.stiffness * 0.75).clamp(1.0, 120.0);
        }
    }
    let added = next.nodes.len() - base;
    shed_tips(&mut next, base, added / 2, rng);
    commit(c, next, cfg, rng)
}

/// Reduces a body with three or more legs to a biped. The two legs with the
/// most drive stay, 1.05 to 1.15 times longer, and step half a cycle apart.
/// Every other leg goes. It fails if that would leave fewer than three nodes.
/// Walking on two legs frees the rest of the body for a balance arm or a tail,
/// and costs fewer nodes.
pub(crate) fn reduce_to_biped(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut legs = walkers(c);
    if legs.len() < 3 {
        return false;
    }
    // Most drive first, so the first two stay.
    legs.sort_stable_by(|p, q| limb_drive(c, q).total_cmp(&limb_drive(c, p)));
    let mut next = c.clone();
    scale_legs(&mut next, &legs[..2], rng.range(1.05, 1.15));
    let mut bones = BoneIds::new();
    let mut nodes = BoneIds::new();
    for leg in &legs[2..] {
        bones.extend(leg.iter().copied());
        nodes.extend(branch_nodes(c, leg));
    }
    if c.nodes.len() - nodes.len() < 3 {
        return false;
    }
    remove_parts(&mut next, &bones, &nodes);
    let left = walkers(&next);
    if left.len() < 2 {
        return false;
    }
    let start = limb_phase(&next, &left[0]).unwrap_or(0.0);
    retime(&mut next, &left, start, |i| 0.5 * (i % 2) as f32);
    commit(c, next, cfg, rng)
}

/// Makes every leg push at once, as a gazelle does when it stots: all legs
/// take the phase of the rearmost, and the tendon of their muscles is raised to
/// 0.4 to 0.8 so each landing is stored and returned in the next push. Needs
/// three or more legs.
pub(crate) fn pronking_stot(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = walkers(c);
    if legs.len() < 3 {
        return false;
    }
    let Some(start) = limb_phase(c, &legs[0]) else {
        return false;
    };
    let mut next = c.clone();
    let tendon = rng.range(0.4, 0.8);
    for leg in &legs {
        tune_leg(&mut next, leg, 1.0, tendon);
    }
    retime(&mut next, &legs, start, |_| 0.0);
    commit(c, next, cfg, rng)
}

/// Gives every leg of two or more bones the proportions of a runner on its
/// toes: the bone at the foot 1.3 to 1.7 times longer, the bone at the hip 0.7
/// to 0.9 as long, and a tendon of at least 0.3 in the leg's muscles. Long
/// light distal segments and short muscled proximal ones are how hoofed runners
/// lengthen their stride without a heavier swing (Hildebrand, Alexander). It
/// fails if no leg has two bones.
pub(crate) fn unguligrade_legs(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs: Limbs = walkers(c).into_iter().filter(|l| l.len() >= 2).collect();
    if legs.is_empty() {
        return false;
    }
    let (distal, proximal) = (rng.range(1.3, 1.7), rng.range(0.7, 0.9));
    let mut next = c.clone();
    let before = spans(&next);
    for leg in &legs {
        let (upper, lower) = (leg[0], leg[1]);
        let hip = next.nodes[next.bones[upper].a as usize];
        let knee = next.nodes[next.bones[upper].b as usize];
        // The upper bone scales about the hip. The knee moves by `offset`, and
        // so does everything below it.
        let offset = [
            (knee.x - hip.x) * (proximal - 1.0),
            (knee.y - hip.y) * (proximal - 1.0),
        ];
        next.nodes[next.bones[upper].b as usize].x += offset[0];
        next.nodes[next.bones[upper].b as usize].y += offset[1];
        next.bones[upper].rest_length *= proximal;
        shift_branch(&mut next, lower, offset);
        // The distal bone grows about its own upper end.
        let last = leg[leg.len() - 1];
        let top = next.nodes[next.bones[last].a as usize];
        let tip = next.bones[last].b as usize;
        let node = &mut next.nodes[tip];
        node.x = top.x + (node.x - top.x) * distal;
        node.y = top.y + (node.y - top.y) * distal;
        next.bones[last].rest_length =
            (next.bones[last].rest_length * distal).clamp(0.03, max_bone_length());
    }
    lift(&mut next);
    for n in &mut next.nodes {
        [n.x, n.y] = clamped(n.x, n.y);
    }
    keep_strokes(&mut next, &before);
    for leg in &legs {
        tune_leg(&mut next, leg, 1.0, 0.3);
    }
    commit(c, next, cfg, rng)
}
