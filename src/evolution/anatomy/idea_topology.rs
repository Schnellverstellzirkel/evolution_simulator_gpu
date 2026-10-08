//! Idea operators that grow, cut and link parts of the body: a balance arm, a
//! toe ahead of the foremost foot, a kickstand, a forked tip, a longer or
//! shorter trunk, inchworm ends, a leg copied twice, a stub or a tail tip
//! removed, a muscle that braces a joint or links two legs, and a free
//! pendulum.
//!
//! Each operator is a whole change on its own, so no parameter noise follows
//! it, and the operators of this file share one pick slot (`GAIT_FILES` in
//! `mod.rs`). Those that add parts stay within `Config::max_nodes` and
//! `max_muscles`.
//!
//! The sources are Herr and Popovic (2008, arms swung against the legs cancel
//! the angular momentum of a gait), Hirose (1993, an inchworm crawls on a
//! front that slides and a rear that grips), Sims (1994, evolved bodies grow
//! parts that carry a muscle and its timing), and Bongard and Pfeifer (2003,
//! bodies grow part by part).
use super::ideas::{bulk, coin, leaf_nodes};
use super::limbs::{clamped, narrow, pick};
use super::rhythm::leaf_limbs;
use super::{
    BoneIds, Context, Operator, child_bones, copy_branch_limited, fit_stroke, is_neck, muscles_on,
    new_muscle, parent_bones, remove_parts, room, span,
};
use crate::config::Config;
use crate::evolution::{Bone, Creature, NodeGene, Rng, max_bone_length};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("sprout_balance_arm", sprout_balance_arm),
    ("forefoot_probe", forefoot_probe),
    ("grow_kickstand", grow_kickstand),
    ("fork_tip", fork_tip),
    ("lengthen_trunk", lengthen_trunk),
    ("shorten_trunk", shorten_trunk),
    ("inchworm_ends", inchworm_ends),
    ("tripod_copy", tripod_copy),
    ("drop_shortest_stub", drop_shortest_stub),
    ("brace_trunk_pair", brace_trunk_pair),
    ("leg_link_muscle", leg_link_muscle),
    ("shed_tail_tip", shed_tail_tip),
    ("dangling_pendulum", dangling_pendulum),
];

/// The median bone length of the body (the longer middle one when the count is
/// even), or 0.2 m for a body with no bones. New parts are sized from it.
fn typical_bone(c: &Creature) -> f32 {
    let mut lengths: Vec<f32> = c.bones.iter().map(|b| b.rest_length).collect();
    lengths.sort_by(|a, b| a.total_cmp(b));
    lengths.get(lengths.len() / 2).copied().unwrap_or(0.2)
}

/// Appends a bone from node `from`, pointing at `angle` (radians, 0 along +x)
/// and `length` long (held between 0.04 m and the longest bone), with a new
/// node of `diameter` at its end. The node takes the grip of `from` and is kept
/// inside the region where nodes may start (`clamped`). The joint gets a narrow
/// range (`narrow`). Returns the new bone's index.
fn add_bone(
    c: &mut Creature,
    from: usize,
    angle: f32,
    length: f32,
    diameter: f32,
    rng: &mut Rng,
) -> usize {
    let length = length.clamp(0.04, max_bone_length());
    let start = c.nodes[from];
    let [x, y] = clamped(
        start.x + length * angle.cos(),
        start.y + length * angle.sin(),
    );
    c.nodes.push(NodeGene {
        x,
        y,
        diameter,
        friction: start.friction,
    });
    let mut bone = Bone::new(from as u32, c.nodes.len() as u32 - 1, length);
    narrow(&mut bone, rng);
    c.bones.push(bone);
    c.bones.len() - 1
}

/// A light arm grows up from an inner node (one with bones below it, other
/// than the end of the neck). It is 0.6 to 1.0 times a typical bone long, leans
/// up to 0.6 rad off vertical and ends in a node of the smallest size, with a
/// narrow joint. A muscle joins it to the bone the inner node hangs from. The
/// muscle has the timing of a random muscle of the body, half a cycle later, so
/// the arm swings against the gait as arms do in a runner (Herr and Popovic
/// 2008). Does nothing for a body with no muscle.
fn sprout_balance_arm(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 1, 1) || c.muscles.is_empty() {
        return false;
    }
    let parents = parent_bones(c);
    let children = child_bones(c);
    let hubs: BoneIds = (1..c.nodes.len())
        .filter(|&n| !children[n].is_empty() && parents[n].is_some_and(|b| !is_neck(c, b)))
        .collect();
    let Some(hub) = pick(&hubs, rng) else {
        return false;
    };
    let parent = parents[hub].expect("hub has a parent bone");
    let angle = std::f32::consts::FRAC_PI_2 + rng.range(-0.6, 0.6);
    let length = typical_bone(c) * rng.range(0.6, 1.0);
    let arm = add_bone(c, hub, angle, length, cfg.min_size, rng);
    let mut template = c.muscles[rng.index(c.muscles.len())];
    template.phase = (template.phase + 0.5).rem_euclid(1.0);
    let anchors = (rng.range(0.4, 0.9), rng.range(0.3, 0.7));
    let m = new_muscle(c, arm, parent, anchors, Some(&template), rng);
    c.muscles.push(m);
    true
}

/// A toe grows forward from the foremost foot (the leaf with the largest x). It
/// points from 0.5 rad down to 0.1 rad up, is half as long as the bone that
/// ends in the foot (at least 5 cm) and ends in a node of the smallest size. A
/// muscle joins the toe to that bone, with the timing of the first muscle on
/// that bone (or else of the first muscle of the body). The toe's far end
/// senses touchdown, and a touchdown sets the muscle to a phase a quarter of a
/// cycle after its start phase. The toe can land before the foot does, so the
/// touch of the probe resets the gait ahead of the foot itself.
fn forefoot_probe(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    let parents = parent_bones(c);
    let Some(foot) = leaf_nodes(c)
        .into_iter()
        .filter(|&n| parents[n].is_some_and(|b| !is_neck(c, b)))
        .max_by(|&a, &b| c.nodes[a].x.total_cmp(&c.nodes[b].x))
    else {
        return false;
    };
    let tip = parents[foot].expect("foot has a parent bone");
    let angle = rng.range(-0.5, 0.1);
    let length = (c.bones[tip].rest_length * 0.5).max(0.05);
    let toe = add_bone(c, foot, angle, length, cfg.min_size, rng);
    let template = muscles_on(c, &[tip], false)
        .first()
        .map(|&m| c.muscles[m])
        .or_else(|| c.muscles.first().copied());
    let mut m = new_muscle(c, toe, tip, (0.8, 0.5), template.as_ref(), rng);
    // The toe's far end senses the landing (bone_a's second endpoint).
    m.sensor = 1;
    m.reset = (m.phase + 0.25).rem_euclid(1.0);
    c.muscles.push(m);
    true
}

/// A passive bone grows back and down from the rearmost inner node (other than
/// the end of the neck), a kickstand that props the tail end. It points 0.2 to
/// 0.7 rad behind straight down, is 0.7 to 1.1 times a typical bone long, ends
/// in a node of the smallest size and has a joint range of 0.1 rad each way.
fn grow_kickstand(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 1, 0) {
        return false;
    }
    let parents = parent_bones(c);
    let children = child_bones(c);
    let Some(rear) = (1..c.nodes.len())
        .filter(|&n| !children[n].is_empty() && parents[n].is_some_and(|b| !is_neck(c, b)))
        .min_by(|&a, &b| c.nodes[a].x.total_cmp(&c.nodes[b].x))
    else {
        return false;
    };
    let angle = -std::f32::consts::FRAC_PI_2 - rng.range(0.2, 0.7);
    let length = typical_bone(c) * rng.range(0.7, 1.1);
    let stand = add_bone(c, rear, angle, length, cfg.min_size, rng);
    c.bones[stand].min_angle = -0.1;
    c.bones[stand].max_angle = 0.1;
    true
}

/// A leaf bone (not the neck) gets a sibling at the joint it hangs from: a bone
/// turned 0.4 to 0.8 rad to one side of it, 0.7 to 1.0 times as long, with a
/// node of the same size at its end. A muscle joins the pair, with the timing
/// of the first muscle on the leaf bone (or a random one if it has none), half
/// a cycle later. The pair is a foot with two toes.
fn fork_tip(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    let parents = parent_bones(c);
    let tips: BoneIds = leaf_nodes(c)
        .into_iter()
        .filter_map(|n| parents[n])
        .filter(|&b| !is_neck(c, b))
        .collect();
    let Some(tip) = pick(&tips, rng) else {
        return false;
    };
    let bone = c.bones[tip];
    let (a, b) = (c.nodes[bone.a as usize], c.nodes[bone.b as usize]);
    let turn = rng.range(0.4, 0.8) * if coin(rng) { 1.0 } else { -1.0 };
    let angle = (b.y - a.y).atan2(b.x - a.x) + turn;
    let sibling = add_bone(
        c,
        bone.a as usize,
        angle,
        bone.rest_length * rng.range(0.7, 1.0),
        c.nodes[bone.b as usize].diameter,
        rng,
    );
    let template = muscles_on(c, &[tip], false).first().map(|&m| c.muscles[m]);
    let mut m = new_muscle(c, sibling, tip, (0.7, 0.7), template.as_ref(), rng);
    m.phase = (m.phase + 0.5).rem_euclid(1.0);
    c.muscles.push(m);
    true
}

/// Scales the bones of the trunk (bones whose child has bones below it, apart
/// from the neck) and refits the strokes of the muscles on them.
fn scale_trunk(c: &mut Creature, by: f32) -> bool {
    let children = child_bones(c);
    let trunk: BoneIds = (0..c.bones.len())
        .filter(|&b| !is_neck(c, b) && !children[c.bones[b].b as usize].is_empty())
        .collect();
    if trunk.is_empty() {
        return false;
    }
    // Strokes keep their share of the span they act across.
    let ratios: Vec<(f32, f32)> = c
        .muscles
        .iter()
        .map(|m| {
            let s = span(c, m).max(0.05);
            (m.short / s, m.long / s)
        })
        .collect();
    for &b in &trunk {
        let length = (c.bones[b].rest_length * by).clamp(0.05, max_bone_length());
        let bone = &mut c.bones[b];
        let (a, z) = (bone.a as usize, bone.b as usize);
        bone.rest_length = length;
        // Move the child side of the bone with it so spans see the change.
        let (dx, dy) = (c.nodes[z].x - c.nodes[a].x, c.nodes[z].y - c.nodes[a].y);
        let now = dx.hypot(dy).max(1.0e-6);
        let (nx, ny) = (
            c.nodes[a].x + dx / now * length,
            c.nodes[a].y + dy / now * length,
        );
        let (mx, my) = (nx - c.nodes[z].x, ny - c.nodes[z].y);
        // The whole branch below the child shifts with it.
        for node in super::branch_nodes(c, &super::branch(c, b)) {
            c.nodes[node].x += mx;
            c.nodes[node].y = (c.nodes[node].y + my).max(0.0);
        }
    }
    for (i, &(short, long)) in ratios.iter().enumerate() {
        let s = span(c, &c.muscles[i]).max(0.05);
        c.muscles[i].short = (short * s).max(0.01);
        c.muscles[i].long = (long * s).max(c.muscles[i].short);
    }
    true
}

/// The trunk's bones grow by 12 to 25%: a longer body between the same legs.
fn lengthen_trunk(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    scale_trunk(c, rng.range(1.12, 1.25))
}

/// The trunk's bones shrink by 10 to 20%: a compact body.
fn shorten_trunk(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    scale_trunk(c, rng.range(0.8, 0.9))
}

/// The foremost and the rearmost feet become the ends of an inchworm: the
/// front slides, the rear grips, and a long muscle between their bones
/// stretches and pulls the body along (Hirose 1993).
fn inchworm_ends(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let parents = parent_bones(c);
    let feet: BoneIds = leaf_nodes(c)
        .into_iter()
        .filter(|&n| parents[n].is_some_and(|b| !is_neck(c, b)))
        .collect();
    let by_x = |a: &&usize, b: &&usize| c.nodes[**a].x.total_cmp(&c.nodes[**b].x);
    let (Some(&front), Some(&rear)) = (feet.iter().max_by(by_x), feet.iter().min_by(by_x)) else {
        return false;
    };
    let (bf, br) = (
        parents[front].expect("foot has a bone"),
        parents[rear].expect("foot has a bone"),
    );
    if bf == br || c.nodes[front].x - c.nodes[rear].x < 0.1 {
        return false;
    }
    c.nodes[front].friction = cfg.min_friction;
    c.nodes[rear].friction = cfg.max_friction;
    if c.muscles.is_empty() {
        return false;
    }
    let mut template = c.muscles[rng.index(c.muscles.len())];
    template.duty = 0.5;
    let mut m = new_muscle(c, br, bf, (1.0, 1.0), Some(&template), rng);
    m.tendon = 0.0;
    m.stiffness = m.stiffness.max(60.0);
    c.muscles.push(m);
    true
}

/// A leg is copied twice onto its own hip, a third and two thirds of a cycle
/// behind, so one leg becomes a three-beat group.
fn tripod_copy(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leaf_limbs(c);
    let driven: BoneIds = (0..legs.len())
        .filter(|&i| !muscles_on(c, &legs[i], false).is_empty())
        .collect();
    let Some(i) = pick(&driven, rng) else {
        return false;
    };
    let leg = legs[i];
    let quota = muscles_on(c, &leg, false).len();
    if !room(c, cfg, 2 * leg.len(), 2 * quota) {
        return false;
    }
    let hip = c.bones[leg[0]].a as usize;
    let mut made = false;
    for (k, lag) in [1.0 / 3.0, 2.0 / 3.0].into_iter().enumerate() {
        let dx = 0.1 * (k as f32 + 1.0) * if coin(rng) { 1.0 } else { -1.0 };
        made |= copy_branch_limited(
            c,
            cfg,
            leg[0],
            hip,
            |p| clamped(p[0] + dx, p[1]),
            false,
            lag,
            quota,
        )
        .is_some();
    }
    made
}

/// The shortest leaf bone goes if it is under 7 cm: a stub that carries no
/// foot and only adds mass.
fn drop_shortest_stub(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.nodes.len() <= 4 {
        return false;
    }
    let parents = parent_bones(c);
    let Some((node, bone)) = leaf_nodes(c)
        .into_iter()
        .filter_map(|n| parents[n].map(|b| (n, b)))
        .filter(|&(_, b)| !is_neck(c, b) && c.bones[b].rest_length < 0.07)
        .min_by(|x, y| {
            c.bones[x.1]
                .rest_length
                .total_cmp(&c.bones[y.1].rest_length)
        })
    else {
        return false;
    };
    remove_parts(c, &[bone], &[node]);
    true
}

/// A stiff, nearly fixed muscle joins two neighbouring trunk bones, with a
/// tendon at full strength: the joint between them becomes a rigid frame.
fn brace_trunk_pair(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let children = child_bones(c);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (b, bone) in c.bones.iter().enumerate() {
        for &next in &children[bone.b as usize] {
            pairs.push((b, next));
        }
    }
    let Some((x, y)) = pick(&pairs, rng) else {
        return false;
    };
    if c.muscles.iter().any(|m| {
        (m.bone_a as usize, m.bone_b as usize) == (x, y)
            || (m.bone_a as usize, m.bone_b as usize) == (y, x)
    }) {
        return false;
    }
    if c.muscles.is_empty() {
        return false;
    }
    let mut template = c.muscles[rng.index(c.muscles.len())];
    template.tendon = 1.0;
    template.stiffness = 100.0;
    let mut m = new_muscle(c, x, y, (0.7, 0.3), Some(&template), rng);
    m.long = m.short + 0.01;
    c.muscles.push(m);
    true
}

/// A weak, springy muscle links the tips of two different legs, so one leg's
/// stretch pulls on the other's.
fn leg_link_muscle(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let legs = leaf_limbs(c);
    if legs.len() < 2 {
        return false;
    }
    let (x, y) = (rng.index(legs.len()), rng.index(legs.len()));
    if x == y {
        return false;
    }
    let (bx, by) = (legs[x][legs[x].len() - 1], legs[y][legs[y].len() - 1]);
    if bx == by || c.muscles.is_empty() {
        return false;
    }
    let mut template = c.muscles[rng.index(c.muscles.len())];
    template.tendon = rng.range(0.6, 1.0);
    template.stiffness = (template.stiffness * 0.4).max(1.0);
    let mut m = new_muscle(c, bx, by, (0.8, 0.8), Some(&template), rng);
    fit_stroke(c, &mut m, Some(&template));
    c.muscles.push(m);
    true
}

/// The leaf farthest from the head goes, with its bone: a tail or toe that
/// only drags.
fn shed_tail_tip(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.nodes.len() <= 4 {
        return false;
    }
    let parents = parent_bones(c);
    let head = [c.nodes[0].x, c.nodes[0].y];
    let Some((node, bone)) = leaf_nodes(c)
        .into_iter()
        .filter_map(|n| parents[n].map(|b| (n, b)))
        .filter(|&(_, b)| !is_neck(c, b))
        .max_by(|x, y| {
            let far = |n: usize| (c.nodes[n].x - head[0]).hypot(c.nodes[n].y - head[1]);
            far(x.0).total_cmp(&far(y.0))
        })
    else {
        return false;
    };
    remove_parts(c, &[bone], &[node]);
    true
}

/// A bone hangs straight down from an inner node (other than the end of the
/// neck), with the widest joint range and no muscle. It is 0.5 to 0.9 times a
/// typical bone long and ends in a heavy node, one in the upper half of the size
/// range. It is a pendulum that swings with the gait and moves the weight about.
fn dangling_pendulum(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 1, 0) {
        return false;
    }
    let parents = parent_bones(c);
    let children = child_bones(c);
    let hubs: BoneIds = (1..c.nodes.len())
        .filter(|&n| !children[n].is_empty() && parents[n].is_some_and(|b| !is_neck(c, b)))
        .collect();
    let Some(hub) = pick(&hubs, rng) else {
        return false;
    };
    let length = typical_bone(c) * rng.range(0.5, 0.9);
    let weight = bulk(cfg, rng.range(0.5, 1.0));
    let bob = add_bone(c, hub, -std::f32::consts::FRAC_PI_2, length, weight, rng);
    c.bones[bob].min_angle = -crate::evolution::JOINT_LIMIT;
    c.bones[bob].max_angle = crate::evolution::JOINT_LIMIT;
    true
}
