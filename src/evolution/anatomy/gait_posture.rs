//! Gait operators: posture: the trunk carried clear of the ground, feet under the load.
//!
//! The operators of this file share one pick slot and are compound: each is a
//! whole, coherent change to the body, and its child gets no parameter noise.
//!
//! The sources are Alexander (1977 and after: mammals carry the body on
//! straight, vertical legs, which cuts the muscle force needed to stand and
//! lets the duty factor fall as speed rises), Full and Koditschek (1999,
//! templates and anchors: a spring-mass body over legs that stand under the
//! centre of mass), Sims (1994) and Lipson and Pollack (2000, whose evolved
//! walkers had to learn to stand before they walked) and Cheney et al. (2013,
//! where the bodies that moved far were low and regular). Mammals also
//! carry the heavy organs in the trunk and keep the feet light (Hildebrand's
//! limb loading), because a light foot swings fast and costs little to move.
//!
//! Joint angles are measured from the starting pose, so an operator that turns
//! a limb in the starting pose also shifts the joint's range by the same angle
//! (`pose_turn`). The stops stay where they were in the world and only the
//! pose the creature starts in changes.
use super::junctions::{keep_strokes, spans, turn_branch};
use super::limbs::clamped;
use super::rhythm::{foot, hip, leaf_limbs};
use super::{BoneIds, Context, Operator, branch, branch_nodes, parent_bones};
use crate::config::Config;
use crate::evolution::{
    Creature, MAX_ORGAN_MASS, Rng, bone_point, max_bone_length, organ_center, organ_range,
};
use crate::physics::node_mass;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

/// An angle brought into -pi..pi.
fn wrap(mut a: f32) -> f32 {
    while a > PI {
        a -= TAU;
    }
    while a < -PI {
        a += TAU;
    }
    a
}

/// One of `items`, or none when empty.
fn choose<T: Copy>(items: &[T], rng: &mut Rng) -> Option<T> {
    items.get(rng.index(items.len().max(1))).copied()
}

/// The direction of bone `b` in the pose, counterclockwise from +x.
fn heading(c: &Creature, b: usize) -> f32 {
    let (p, q) = (
        c.nodes[c.bones[b].a as usize],
        c.nodes[c.bones[b].b as usize],
    );
    (q.y - p.y).atan2(q.x - p.x)
}

/// Legs with a foot: leaf limbs of at least two bones.
fn feet_legs(c: &Creature) -> Vec<BoneIds> {
    leaf_limbs(c).into_iter().filter(|l| l.len() >= 2).collect()
}

/// Turns the branch of bone `j` by `t` in the starting pose and moves the
/// joint's range by the same angle, so its stops stay where they were.
fn pose_turn(c: &mut Creature, j: usize, t: f32) {
    turn_branch(c, j, t);
    let bone = &mut c.bones[j];
    bone.min_angle -= t;
    bone.max_angle -= t;
    bone.clamp_range();
}

/// Moves the whole body down or up so its lowest node touches the ground.
fn settle(c: &mut Creature) {
    let low = c.nodes.iter().map(|n| n.y).fold(f32::MAX, f32::min);
    for n in c.nodes.iter_mut() {
        n.y -= low;
    }
}

/// Scales a leg about its hip by `k`: nodes and rest lengths together.
/// Returns false, touching nothing, when a bone would get shorter than 0.04 m
/// or longer than the longest bone.
fn scale_leg(c: &mut Creature, leg: &[usize], k: f32) -> bool {
    if !leg
        .iter()
        .all(|&b| (0.04..=max_bone_length()).contains(&(c.bones[b].rest_length * k)))
    {
        return false;
    }
    let h = c.nodes[hip(c, leg)];
    for n in branch_nodes(c, leg) {
        let node = &mut c.nodes[n];
        [node.x, node.y] = clamped(h.x + k * (node.x - h.x), h.y + k * (node.y - h.y));
    }
    for &b in leg {
        c.bones[b].rest_length *= k;
    }
    true
}

/// Mass-weighted centre of the body in the starting pose: nodes and organs.
fn centre_of_mass(c: &Creature) -> [f32; 2] {
    let mut sum = [0.0f32; 2];
    let mut mass = 0.0f32;
    for n in c.nodes.iter() {
        let m = node_mass(n.diameter);
        sum[0] += m * n.x;
        sum[1] += m * n.y;
        mass += m;
    }
    for b in c.bones.iter().filter(|b| b.organ_mass > 0.0) {
        let p = bone_point(*b, &c.nodes, b.organ_at);
        sum[0] += b.organ_mass * p[0];
        sum[1] += b.organ_mass * p[1];
        mass += b.organ_mass;
    }
    [sum[0] / mass.max(1e-6), sum[1] / mass.max(1e-6)]
}

/// Bones that are in no leg: the trunk, the neck and the tips that are not feet.
fn trunk_bones(c: &Creature) -> BoneIds {
    let legs = feet_legs(c);
    (0..c.bones.len())
        .filter(|b| !legs.iter().any(|l| l.contains(b)))
        .collect()
}

/// Adds `amount` of organ mass to bone `to`. A new organ sits at `at`. An
/// organ already there moves to the mass-weighted mean of its place and `at`,
/// so the pooled mass keeps its centre.
fn pool_organ(c: &mut Creature, to: usize, amount: f32, at: f32) {
    let bone = &mut c.bones[to];
    bone.organ_at = if bone.organ_mass > 0.0 {
        (bone.organ_mass * bone.organ_at + amount * at) / (bone.organ_mass + amount)
    } else {
        at
    };
    bone.organ_mass += amount;
}

/// Scales every leg by `k` (all together, so the legs stay level) and puts
/// the body back on the ground.
fn scale_all_legs(c: &mut Creature, k: f32) -> bool {
    let legs = feet_legs(c);
    let before = spans(c);
    let mut changed = false;
    for leg in &legs {
        changed |= scale_leg(c, leg, k);
    }
    if !changed {
        return false;
    }
    settle(c);
    keep_strokes(c, &before);
    true
}

/// Straightens a knee: turns the part of a leg below one joint until the bone
/// continues the line of the bone above, with a bend left of at most a third
/// of the original. A straight leg is a strut, not a lever: it carries the
/// load through bone and needs little muscle force to hold the trunk up
/// (Alexander), and it lifts the hip higher over the same bone lengths.
pub(crate) fn straighten_leg_knee(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut joints: Vec<(usize, f32)> = Vec::new();
    for leg in feet_legs(c) {
        for i in 1..leg.len() {
            let bend = wrap(heading(c, leg[i - 1]) - heading(c, leg[i]));
            if (0.15..=1.5).contains(&bend.abs()) {
                joints.push((leg[i], bend));
            }
        }
    }
    let Some((bone, bend)) = choose(&joints, rng) else {
        return false;
    };
    let before = spans(c);
    pose_turn(c, bone, bend * rng.range(0.67, 1.0));
    keep_strokes(c, &before);
    true
}

/// Turns a whole leg about its hip until the line from hip to foot is
/// vertical, so the foot stands directly under the hip. The leg keeps its
/// shape, joint stops and muscles. A foot under the hip puts the ground force
/// along the leg, where a foot ahead or behind puts a torque on the hip
/// (Full and Koditschek's anchor: the leg stands under the load).
pub(crate) fn foot_under_hip(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs: Vec<(usize, f32)> = feet_legs(c)
        .into_iter()
        .filter_map(|leg| {
            let (h, f) = (c.nodes[hip(c, &leg)], c.nodes[foot(c, &leg)]);
            let turn = wrap(-FRAC_PI_2 - (f.y - h.y).atan2(f.x - h.x));
            (f.y < h.y - 0.03 && (0.08..=1.2).contains(&turn.abs())).then_some((leg[0], turn))
        })
        .collect();
    let Some((root, turn)) = choose(&legs, rng) else {
        return false;
    };
    let before = spans(c);
    pose_turn(c, root, turn);
    keep_strokes(c, &before);
    true
}

/// Moves the hip of a leg whose foot is more than 0.12 m from the body's
/// centre of mass to the trunk node that brings the foot nearest below that
/// centre, keeping the leg's shape. Feet under the centre of mass carry the
/// weight without a pitching torque, while feet far from it have to be braced
/// by the other legs. The muscles from the leg to the bone above the old hip
/// move to the bone above the new one.
pub(crate) fn hip_toward_mass_centre(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let com = centre_of_mass(c)[0];
    let legs = feet_legs(c);
    let inside: BoneIds = legs.iter().flat_map(|l| branch_nodes(c, l)).collect();
    let far: Vec<BoneIds> = legs
        .iter()
        .copied()
        .filter(|l| hip(c, l) != 0 && (c.nodes[foot(c, l)].x - com).abs() > 0.12)
        .collect();
    let Some(leg) = choose(&far, rng) else {
        return false;
    };
    let (from, tip) = (hip(c, &leg), c.nodes[foot(c, &leg)]);
    let reach = tip.x - c.nodes[from].x;
    let error = (tip.x - com).abs();
    let parents = parent_bones(c);
    let Some(at) = (1..c.nodes.len())
        .filter(|&n| n != from && !inside.contains(&n) && parents[n].is_some())
        .min_by(|&p, &q| {
            let miss = |n: usize| (c.nodes[n].x + reach - com).abs();
            miss(p).total_cmp(&miss(q))
        })
    else {
        return false;
    };
    if (c.nodes[at].x + reach - com).abs() > error - 0.06 {
        return false;
    }
    let before = spans(c);
    let offset = [
        c.nodes[at].x - c.nodes[from].x,
        c.nodes[at].y - c.nodes[from].y,
    ];
    for n in branch_nodes(c, &branch(c, leg[0])) {
        let node = &mut c.nodes[n];
        [node.x, node.y] = clamped(node.x + offset[0], node.y + offset[1]);
    }
    c.bones[leg[0]].a = at as u32;
    if let (Some(old), Some(new)) = (parents[from], parents[at]) {
        for m in c.muscles.iter_mut() {
            let (x, y) = (m.bone_a as usize, m.bone_b as usize);
            if (x, y) == (leg[0], old) {
                m.bone_b = new as u32;
            } else if (x, y) == (old, leg[0]) {
                m.bone_a = new as u32;
            }
        }
    }
    settle(c);
    keep_strokes(c, &before);
    true
}

/// Turns the starting pose of one leg to the middle of its joint ranges: each
/// joint whose range is off centre by more than 0.08 rad starts at the middle
/// of its range, and the range is centred on it. The leg then starts in
/// mid-stance, with room to swing both ways, instead of at one stop where half
/// of the first stroke is wasted against it. (The pose that `pose_joint_at_stop`
/// gives is the opposite, a braced one.)
pub(crate) fn centre_leg_rest_angles(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let off = |c: &Creature, b: usize| 0.5 * (c.bones[b].min_angle + c.bones[b].max_angle);
    let legs: Vec<BoneIds> = feet_legs(c)
        .into_iter()
        .filter(|l| l.iter().any(|&b| off(c, b).abs() > 0.08))
        .collect();
    let Some(leg) = choose(&legs, rng) else {
        return false;
    };
    let before = spans(c);
    for &b in leg.iter() {
        let mid = off(c, b);
        if mid.abs() > 0.08 {
            pose_turn(c, b, mid);
        }
    }
    keep_strokes(c, &before);
    true
}

/// Lengthens every leg by 10 to 30%, scaling each about its hip, and lifts the
/// trunk to match. The legs keep their proportions, joint stops and muscles,
/// so the gait is the same scaled up: the hip is higher over the ground and
/// each stride covers more of it (stride length grows with leg length, and
/// tall mammals cover more ground per step).
pub(crate) fn raise_stance_height(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    scale_all_legs(c, rng.range(1.1, 1.3))
}

/// Shortens every leg by 10 to 25% about its hip, and the body settles lower.
/// A low body has a low centre of mass, so it tips less when a leg lifts and
/// its feet are close to the ground at the end of a swing (Cheney et al.: low
/// bodies were the stable movers). The legs keep their proportions and
/// muscles.
pub(crate) fn crouch_legs(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    scale_all_legs(c, rng.range(0.75, 0.9))
}

/// Moves the organ mass of every leg bone into the trunk, onto trunk bones
/// that can hold an organ and have room for it. A mammal's heavy organs sit in
/// the trunk and its feet are light, and a light foot swings with little
/// muscle work: the energy of a swing grows with the mass moved.
pub(crate) fn organs_to_trunk(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let sources: BoneIds = feet_legs(c)
        .iter()
        .flat_map(|l| l.iter().copied())
        .filter(|&b| c.bones[b].organ_mass > 0.0)
        .collect();
    let center = organ_center(&c.nodes);
    let trunk: BoneIds = trunk_bones(c)
        .into_iter()
        .filter(|&b| organ_range(&c.bones[b], &c.nodes, center).is_some())
        .collect();
    let mut changed = false;
    for &from in sources.iter() {
        let amount = c.bones[from].organ_mass;
        let fits: BoneIds = trunk
            .iter()
            .copied()
            .filter(|&t| MAX_ORGAN_MASS - c.bones[t].organ_mass >= amount)
            .collect();
        let Some(to) = choose(&fits, rng) else {
            continue;
        };
        let Some((low, high)) = organ_range(&c.bones[to], &c.nodes, center) else {
            continue;
        };
        let at = rng.range(low, high);
        c.bones[from].organ_mass = 0.0;
        pool_organ(c, to, amount, at);
        changed = true;
    }
    changed
}

/// Moves the organ mass of the lower bones of one leg to its first bone, next
/// to the hip. Mass near the hip adds little to the leg's moment of inertia, so
/// the swing needs less torque and goes faster (Hildebrand: the muscle mass of
/// a running limb sits at its top, and the lower bones are light).
pub(crate) fn organs_to_hip(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let center = organ_center(&c.nodes);
    let legs: Vec<BoneIds> = feet_legs(c)
        .into_iter()
        .filter(|l| {
            organ_range(&c.bones[l[0]], &c.nodes, center).is_some()
                && l[1..].iter().any(|&b| {
                    let m = c.bones[b].organ_mass;
                    m > 0.0 && MAX_ORGAN_MASS - c.bones[l[0]].organ_mass >= m
                })
        })
        .collect();
    let Some(leg) = choose(&legs, rng) else {
        return false;
    };
    let Some((low, _)) = organ_range(&c.bones[leg[0]], &c.nodes, center) else {
        return false;
    };
    let mut changed = false;
    for &from in &leg[1..] {
        let amount = c.bones[from].organ_mass;
        if amount > 0.0 && MAX_ORGAN_MASS - c.bones[leg[0]].organ_mass >= amount {
            c.bones[from].organ_mass = 0.0;
            pool_organ(c, leg[0], amount, low);
            changed = true;
        }
    }
    changed
}

/// Moves the highest organ to a lower place that can hold it, at the low end
/// of that bone's allowed stretch. The body's mass sits lower, so the centre
/// of mass drops, and a low centre of mass makes the body harder to tip over a
/// planted foot (an inverted pendulum falls slower the lower its mass). The
/// total organ mass does not change.
pub(crate) fn sink_organ_mass(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let center = organ_center(&c.nodes);
    let height = |b: usize, t: f32| bone_point(c.bones[b], &c.nodes, t)[1];
    let Some(from) = (0..c.bones.len())
        .filter(|&b| c.bones[b].organ_mass > 0.0)
        .max_by(|&p, &q| height(p, c.bones[p].organ_at).total_cmp(&height(q, c.bones[q].organ_at)))
    else {
        return false;
    };
    let amount = c.bones[from].organ_mass;
    let top = height(from, c.bones[from].organ_at);
    // For each other bone with room, its lowest allowed point.
    let lower: Vec<(usize, f32)> = (0..c.bones.len())
        .filter(|&b| b != from && MAX_ORGAN_MASS - c.bones[b].organ_mass >= amount)
        .filter_map(|b| {
            let (low, high) = organ_range(&c.bones[b], &c.nodes, center)?;
            let t = if height(b, low) <= height(b, high) {
                low
            } else {
                high
            };
            (height(b, t) < top - 0.04).then_some((b, t))
        })
        .collect();
    let Some((to, at)) = choose(&lower, rng) else {
        return false;
    };
    c.bones[from].organ_mass = 0.0;
    pool_organ(c, to, amount, at);
    true
}

/// Shortens a part that drags: a one-bone tip (a tail or a snout) whose end
/// lies within 0.12 m of the ground and that is longer than 0.12 m loses 25 to
/// 55% of its length. A dragging tip rubs the ground and costs friction work
/// each step, while a short stub lets the body rest on its feet. The tip
/// keeps its direction, muscles and joint range.
pub(crate) fn shorten_dragging_tip(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let tips: BoneIds = leaf_limbs(c)
        .into_iter()
        .filter(|l| l.len() == 1)
        .map(|l| l[0])
        .filter(|&b| c.nodes[c.bones[b].b as usize].y < 0.12 && c.bones[b].rest_length > 0.12)
        .collect();
    let Some(bone) = choose(&tips, rng) else {
        return false;
    };
    let k = rng.range(0.45, 0.75);
    let before = spans(c);
    let a = c.nodes[c.bones[bone].a as usize];
    let tip = &mut c.nodes[c.bones[bone].b as usize];
    [tip.x, tip.y] = clamped(a.x + k * (tip.x - a.x), a.y + k * (tip.y - a.y));
    c.bones[bone].rest_length = (c.bones[bone].rest_length * k).max(0.06);
    settle(c);
    keep_strokes(c, &before);
    true
}

/// Turns the foremost and the rearmost leg outward by 0.1 to 0.3 rad each,
/// about their hips, so the feet stand wider apart along the trunk than the
/// hips do. A longer base under the same body resists pitching, the way a
/// quadruped's front and hind feet plant well apart. A leg that would lean
/// more than 0.8 rad from vertical does not turn.
pub(crate) fn widen_stance_fore_aft(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = feet_legs(c);
    let at = |l: &BoneIds| c.nodes[hip(c, l)].x;
    let (Some(front), Some(rear)) = (
        legs.iter().max_by(|p, q| at(p).total_cmp(&at(q))),
        legs.iter().min_by(|p, q| at(p).total_cmp(&at(q))),
    ) else {
        return false;
    };
    if at(front) - at(rear) < 0.12 {
        return false;
    }
    // Lean from straight down, positive when the foot is ahead (+x).
    let lean = |l: &BoneIds| {
        let (h, f) = (c.nodes[hip(c, l)], c.nodes[foot(c, l)]);
        (f.x - h.x).atan2(h.y - f.y)
    };
    let t = rng.range(0.1, 0.3);
    if (lean(front) + t).abs() > 0.8 || (lean(rear) - t).abs() > 0.8 {
        return false;
    }
    let (front, rear) = (front[0], rear[0]);
    let before = spans(c);
    pose_turn(c, front, t);
    pose_turn(c, rear, -t);
    keep_strokes(c, &before);
    true
}

/// Bends a leg into the zigzag of a mammal's limb: the first two bones lean
/// opposite ways, 0.25 to 0.5 rad off vertical, with the foot under the hip.
/// A leg behind the middle of the body has its knee forward, a leg in front of
/// it has its knee back, as the hind legs and fore legs of a dog do. The bend
/// stores the load in the knee like a spring and lets the leg fold up short
/// during the swing.
pub(crate) fn zigzag_leg_bend(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let middle = c.nodes.iter().map(|n| n.x).sum::<f32>() / c.nodes.len() as f32;
    let legs = feet_legs(c);
    let Some(leg) = choose(&legs, rng) else {
        return false;
    };
    let (l1, l2) = (c.bones[leg[0]].rest_length, c.bones[leg[1]].rest_length);
    let side = if c.nodes[hip(c, &leg)].x < middle {
        1.0
    } else {
        -1.0
    };
    let upper = side * rng.range(0.25, 0.5);
    let ratio = -l1 * upper.sin() / l2.max(0.03);
    if ratio.abs() > 0.9 {
        return false;
    }
    // Absolute headings: straight down is -pi/2, and a lean ahead adds to it.
    let first = wrap(upper - FRAC_PI_2 - heading(c, leg[0]));
    // Turning the first bone's branch turns the second bone by the same angle.
    let second = wrap(ratio.asin() - FRAC_PI_2 - (heading(c, leg[1]) + first));
    if first.abs() > 1.2 || second.abs() > 1.5 || first.abs() + second.abs() < 0.1 {
        return false;
    }
    let before = spans(c);
    pose_turn(c, leg[0], first);
    pose_turn(c, leg[1], second);
    keep_strokes(c, &before);
    true
}

/// Lengthens the leg whose foot hangs highest above the ground until its foot
/// reaches the ground, scaling it about its hip by 1.05 to 1.8. A foot that
/// never touches the ground carries no load, so the other legs carry the body
/// and the body leans on them. A leg that reaches the ground shares the load
/// and the stride.
pub(crate) fn ground_hanging_foot(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let low = c.nodes.iter().map(|n| n.y).fold(f32::MAX, f32::min);
    let foot_y = |l: &BoneIds| c.nodes[foot(c, l)].y;
    let Some((leg, k)) = feet_legs(c)
        .into_iter()
        .filter(|l| foot_y(l) > low + 0.08)
        .filter_map(|l| {
            let h = c.nodes[hip(c, &l)].y;
            let k = (h - low) / (h - foot_y(&l));
            (h - foot_y(&l) > 0.05 && (1.05..=1.8).contains(&k)).then_some((l, k))
        })
        .max_by(|p, q| foot_y(&p.0).total_cmp(&foot_y(&q.0)))
    else {
        return false;
    };
    let before = spans(c);
    if !scale_leg(c, &leg, k) {
        return false;
    }
    settle(c);
    keep_strokes(c, &before);
    true
}

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("straighten_leg_knee", straighten_leg_knee),
    ("foot_under_hip", foot_under_hip),
    ("hip_toward_mass_centre", hip_toward_mass_centre),
    ("centre_leg_rest_angles", centre_leg_rest_angles),
    ("raise_stance_height", raise_stance_height),
    ("crouch_legs", crouch_legs),
    ("organs_to_trunk", organs_to_trunk),
    ("organs_to_hip", organs_to_hip),
    ("sink_organ_mass", sink_organ_mass),
    ("shorten_dragging_tip", shorten_dragging_tip),
    ("widen_stance_fore_aft", widen_stance_fore_aft),
    ("zigzag_leg_bend", zigzag_leg_bend),
    ("ground_hanging_foot", ground_hanging_foot),
];
