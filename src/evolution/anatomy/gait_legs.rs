//! Gait operators for legs: each gives a leg one feature of a mammal leg, such
//! as a knee, an ankle, a foot, longer distal bones, a stop or a tendon. A leg
//! is a leaf limb (`rhythm::leaf_limbs`), and a foot is the end of a leg of at
//! least two bones. `OPS` is one entry of `GAIT_FILES` in `mod.rs`, so the
//! operators of this file share one pick slot and are compound: each is a
//! whole change, and its child gets no parameter noise. The ideas come from
//! the biology of walking and from evolutionary robotics: Alexander, Full and
//! Koditschek, Sims (1994), Lipson and Pollack (2000), Cheney et al. (2013)
//! and Ijspeert.
use super::compound::{close_ring, lead_muscle, shed_tips, strongest};
use super::extra::drive;
use super::junctions::{
    add, add_node, keep_strokes, lift, pos, scale, shift_branch, spans, sub, turn_branch,
};
use super::limbs::narrow;
use super::muscles::turn;
use super::rhythm::leaf_limbs;
use super::{BoneIds, Context, MuscleIds, Operator, muscles_on, new_muscle, room};
use crate::config::Config;
use crate::evolution::{Bone, Creature, Muscle, Rng, max_bone_length};

/// This file's operators, by name. Add each new one here. Breeding picks an
/// operator by its position in this list, so a new order gives a different
/// search for the same seed.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("lengthen_lower_leg", lengthen_lower_leg),
    ("add_ankle_joint", add_ankle_joint),
    ("bend_stick_leg_at_knee", bend_stick_leg_at_knee),
    ("lock_knee_extension", lock_knee_extension),
    ("fold_leg_zigzag", fold_leg_zigzag),
    ("straighten_leg_column", straighten_leg_column),
    ("flatten_foot_sole", flatten_foot_sole),
    ("raise_heel_digitigrade", raise_heel_digitigrade),
    ("tendon_the_ankle", tendon_the_ankle),
    ("grow_forward_foot", grow_forward_foot),
    ("set_leg_proportions", set_leg_proportions),
    ("harden_or_pad_foot", harden_or_pad_foot),
    ("lag_knee_behind_hip", lag_knee_behind_hip),
];

/// A random leg (leaf limb) for which `ok` holds, or `None` when no leg does.
/// It takes one draw from `rng` either way.
fn pick_leg(c: &Creature, rng: &mut Rng, ok: impl Fn(&BoneIds) -> bool) -> Option<BoneIds> {
    let mut legs = leaf_limbs(c);
    legs.retain(|leg| ok(leg));
    legs.get(rng.index(legs.len().max(1))).copied()
}

/// The vector along bone `j`, from its parent node to its child node.
fn bone_vector(c: &Creature, j: usize) -> [f32; 2] {
    sub(pos(c, c.bones[j].b as usize), pos(c, c.bones[j].a as usize))
}

/// The signed angle from bone `upper` to bone `lower` (counterclockwise is
/// positive). Zero is a straight leg.
fn bend(c: &Creature, upper: usize, lower: usize) -> f32 {
    let (p, q) = (bone_vector(c, upper), bone_vector(c, lower));
    (p[0] * q[1] - p[1] * q[0]).atan2(p[0] * q[0] + p[1] * q[1])
}

/// The signed angle (-pi to pi) that turns direction `from` onto direction
/// `to`. Counterclockwise is positive.
fn angle_between(from: [f32; 2], to: [f32; 2]) -> f32 {
    (from[0] * to[1] - from[1] * to[0]).atan2(from[0] * to[0] + from[1] * to[1])
}

/// Whether every node is inside the region where nodes may start.
fn inside(c: &Creature) -> bool {
    let extent = 2.0 * max_bone_length();
    c.nodes
        .iter()
        .all(|n| n.x.abs() <= extent && (0.0..=extent).contains(&n.y))
}

/// Moves a muscle's phase to `lead` by the shortest way round the cycle, and
/// its touchdown reset by the same amount.
fn follow(m: &mut Muscle, lead: f32) {
    let shift = turn(m.phase, lead);
    m.phase = (m.phase + shift).rem_euclid(1.0);
    m.reset = (m.reset + shift).rem_euclid(1.0);
}

/// The timing template of a new muscle: a copy of the strongest muscle with an
/// end on `bones`, or of the strongest muscle of the body when none has one.
fn leg_template(c: &Creature, bones: &[usize]) -> Option<Muscle> {
    let on = muscles_on(c, bones, false);
    strongest(c, &on)
        .or_else(|| lead_muscle(c, &[]))
        .map(|i| c.muscles[i])
}

/// Cuts leaf bone `first` into two at fraction `t` of its length and returns
/// the index of the new lower bone. The old end node moves to the cut, shifted
/// by `side`, and becomes the joint. A copy of it stays at the old tip and is
/// the new end of the leg. The lower bone gets a narrow joint range. Muscle
/// anchors and the organ go to the part that holds their place.
fn insert_joint(c: &mut Creature, first: usize, t: f32, side: [f32; 2], rng: &mut Rng) -> usize {
    let old = c.bones[first];
    let (a, b) = (pos(c, old.a as usize), pos(c, old.b as usize));
    let cut = add(add(a, scale(sub(b, a), t)), side);
    let tip = add_node(c, old.b as usize, b);
    c.nodes[old.b as usize].x = cut[0];
    c.nodes[old.b as usize].y = cut[1];
    let upper = (cut[0] - a[0]).hypot(cut[1] - a[1]).max(0.03);
    let lower = (b[0] - cut[0]).hypot(b[1] - cut[1]).max(0.03);
    c.bones[first].rest_length = upper;
    let second = c.bones.len();
    let mut part = Bone::new(old.b, tip as u32, lower);
    narrow(&mut part, rng);
    // The part that holds position `at` of the old bone, and where along that
    // part the position lies.
    let split = |at: f32| {
        if at <= t {
            (first, at / t)
        } else {
            (second, (at - t) / (1.0 - t))
        }
    };
    if old.organ_mass > 0.0 {
        let (holder, at) = split(old.organ_at);
        if holder == first {
            c.bones[first].organ_at = at;
        } else {
            c.bones[first].organ_mass = 0.0;
            c.bones[first].organ_at = 0.5;
            part.organ_mass = old.organ_mass;
            part.organ_at = at;
        }
    }
    c.bones.push(part);
    for m in &mut c.muscles {
        for (bone, anchor) in [
            (&mut m.bone_a, &mut m.anchor_a),
            (&mut m.bone_b, &mut m.anchor_b),
        ] {
            if *bone as usize == first {
                let (holder, at) = split(*anchor);
                *bone = holder as u32;
                *anchor = at;
            }
        }
    }
    second
}

/// Lengthens the last bone of a leg of two or more bones (the lower leg or the
/// foot) by 1.2 to 1.7 times, so the leg is longer and the muscles keep their
/// stroke. The factor stops at the longest bone allowed, and a factor under 1.1
/// is refused. Cursorial mammals put the length in the distal bones, where it
/// adds stride for little muscle mass, and a longer shank makes a longer step
/// for the same swing of the hip (Alexander).
pub(crate) fn lengthen_lower_leg(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(leg) = pick_leg(c, rng, |leg| leg.len() >= 2) else {
        return false;
    };
    let last = leg[leg.len() - 1];
    let length = c.bones[last].rest_length;
    let factor = rng.range(1.2, 1.7).min(max_bone_length() / length);
    if factor < 1.1 {
        return false;
    }
    let before = spans(c);
    let along = bone_vector(c, last);
    shift_branch(c, last, scale(along, factor - 1.0));
    c.bones[last].rest_length = length * factor;
    lift(c);
    // TODO: `c` is already edited here, so a refusal leaves the longer leg in
    // place and the caller reads the body as unchanged. Editing a copy, as
    // `fold_leg_zigzag` does, would fix that.
    if !inside(c) {
        return false;
    }
    keep_strokes(c, &before);
    true
}

/// Gives a leg of two or more bones an ankle: the last 20 to 40% of the last
/// bone becomes a foot segment on its own joint. A new muscle across the ankle
/// takes the timing of the leg's strongest muscle (`leg_template`) with its
/// phase raised by 0.25. A leg with an ankle can keep the foot flat on the
/// ground while the shank swings over it. The last bone must be at least 0.1
/// long, and the body needs room for a node and a muscle.
pub(crate) fn add_ankle_joint(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    let Some(leg) = pick_leg(c, rng, |leg| {
        leg.len() >= 2 && c.bones[leg[leg.len() - 1]].rest_length >= 0.1
    }) else {
        return false;
    };
    let last = leg[leg.len() - 1];
    let template = leg_template(c, &leg);
    let mut next = c.clone();
    let before = spans(&next);
    let foot = rng.range(0.2, 0.4);
    let second = insert_joint(&mut next, last, 1.0 - foot, [0.0, 0.0], rng);
    keep_strokes(&mut next, &before);
    let anchors = (rng.range(0.4, 0.9), rng.range(0.2, 0.7));
    let mut m = new_muscle(&next, last, second, anchors, template.as_ref(), rng);
    if let Some(t) = template {
        follow(&mut m, t.phase + 0.25);
    }
    next.muscles.push(m);
    if !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Bends a leg of one straight bone near its middle: the bone is cut at 40 to
/// 60% of its length into a thigh and a shank, and the knee between them sticks
/// out sideways by 12 to 30% of the length. A new muscle across the knee takes
/// the timing of the leg's strongest muscle (`leg_template`) with its phase
/// raised by 0.25. Sims (1994) and Lipson and Pollack (2000) got their walkers
/// from jointed legs, and a bent leg can shorten in the swing and extend in the
/// stance, which a straight stick cannot. The bone must be at least 0.14 long,
/// and the body needs room for a node and a muscle.
pub(crate) fn bend_stick_leg_at_knee(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    let Some(leg) = pick_leg(c, rng, |leg| {
        leg.len() == 1 && c.bones[leg[0]].rest_length >= 0.14
    }) else {
        return false;
    };
    let bone = leg[0];
    let length = c.bones[bone].rest_length;
    let along = bone_vector(c, bone);
    let side = if rng.unit() < 0.5 { 1.0 } else { -1.0 };
    let push = side * length * rng.range(0.12, 0.3) / along[0].hypot(along[1]).max(1.0e-6);
    let template = leg_template(c, &leg);
    let mut next = c.clone();
    let before = spans(&next);
    let second = insert_joint(
        &mut next,
        bone,
        rng.range(0.4, 0.6),
        [-along[1] * push, along[0] * push],
        rng,
    );
    lift(&mut next);
    if !inside(&next) {
        return false;
    }
    keep_strokes(&mut next, &before);
    let anchors = (rng.range(0.5, 0.9), rng.range(0.1, 0.5));
    let mut m = new_muscle(&next, bone, second, anchors, template.as_ref(), rng);
    if let Some(t) = template {
        follow(&mut m, t.phase + 0.25);
    }
    next.muscles.push(m);
    if !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Stops a bent knee from extending past the pose it starts in (a 0.02 to 0.08
/// rad allowance) and raises its flexion limit to a draw of 0.7 to 1.2 rad when
/// that is wider. The knee must bend 0.12 rad or more, and a knee with under
/// 0.1 rad of extension range is left alone. The knee can fold to bring the
/// foot up in the swing, and it acts as a strut that takes the load in the
/// stance, as the knee of a mammal does: a joint with a stop turns muscle pull
/// into support without a muscle holding it.
pub(crate) fn lock_knee_extension(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(leg) = pick_leg(c, rng, |leg| leg.len() >= 2) else {
        return false;
    };
    let i = 1 + rng.index(leg.len() - 1);
    let angle = bend(c, leg[i - 1], leg[i]);
    if angle.abs() < 0.12 {
        return false;
    }
    let allowance = rng.range(0.02, 0.08);
    let flexion = rng.range(0.7, 1.2);
    let knee = &mut c.bones[leg[i]];
    // A shank turned counterclockwise from the thigh extends clockwise.
    if angle > 0.0 {
        if knee.min_angle > -0.1 {
            return false;
        }
        knee.min_angle = -allowance;
        knee.max_angle = knee.max_angle.max(flexion);
    } else {
        if knee.max_angle < 0.1 {
            return false;
        }
        knee.max_angle = allowance;
        knee.min_angle = knee.min_angle.min(-flexion);
    }
    true
}

/// Folds a straight leg into a zigzag: the lower part turns about a joint by
/// 0.5 to 1.0 rad to a random side, so the foot comes up and the leg is a
/// shorter, springier Z. The joint must bend less than 0.25 rad to start with.
/// Folded legs swing through with little foot lift (a cat or a dog hind leg),
/// and a compliant bent leg stores and returns energy as the spring of a
/// spring mass walker does (Full and Koditschek).
pub(crate) fn fold_leg_zigzag(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(leg) = pick_leg(c, rng, |leg| {
        leg.len() >= 2 && bend(c, leg[0], leg[1]).abs() < 0.25
    }) else {
        return false;
    };
    let i = 1 + rng.index(leg.len() - 1);
    if bend(c, leg[i - 1], leg[i]).abs() >= 0.25 {
        return false;
    }
    let side = if rng.unit() < 0.5 { 1.0 } else { -1.0 };
    let before = spans(c);
    let mut next = c.clone();
    turn_branch(&mut next, leg[i], side * rng.range(0.5, 1.0));
    if !inside(&next) {
        return false;
    }
    keep_strokes(&mut next, &before);
    c.clone_from(&next);
    true
}

/// Straightens a bent leg into a column: at a joint that bends 0.3 rad or more,
/// the lower part turns back by 90 to 100% of the bend. A straight leg holds
/// the body up with bone and not with muscle, as the legs of an elephant do,
/// and it reaches farthest for a given hip swing. The knee keeps its joint
/// range.
pub(crate) fn straighten_leg_column(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(leg) = pick_leg(c, rng, |leg| {
        (1..leg.len()).any(|i| bend(c, leg[i - 1], leg[i]).abs() >= 0.3)
    }) else {
        return false;
    };
    let bent: BoneIds = (1..leg.len())
        .filter(|&i| bend(c, leg[i - 1], leg[i]).abs() >= 0.3)
        .collect();
    let i = bent[rng.index(bent.len())];
    let angle = bend(c, leg[i - 1], leg[i]);
    let before = spans(c);
    let mut next = c.clone();
    turn_branch(&mut next, leg[i], -angle * rng.range(0.9, 1.0));
    if !inside(&next) {
        return false;
    }
    keep_strokes(&mut next, &before);
    c.clone_from(&next);
    true
}

/// Turns the last bone of a leg that points down to point forward, flat
/// along the ground, so it is a sole. A flat foot (a bear or a human) has a
/// long contact and a wide base, and the ankle can roll over it. A foot turns
/// only if its drop is more than 0.3 of its length and the turn is 0.3 rad or
/// more. It ends level to within a slope of 0.1.
pub(crate) fn flatten_foot_sole(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(leg) = pick_leg(c, rng, |leg| {
        leg.len() >= 2
            && bone_vector(c, leg[leg.len() - 1])[1]
                < -0.3 * c.bones[leg[leg.len() - 1]].rest_length
    }) else {
        return false;
    };
    let last = leg[leg.len() - 1];
    let turn = angle_between(bone_vector(c, last), [1.0, rng.range(-0.1, 0.1)]);
    if turn.abs() < 0.3 {
        return false;
    }
    let before = spans(c);
    let mut next = c.clone();
    turn_branch(&mut next, last, turn);
    if !inside(&next) {
        return false;
    }
    keep_strokes(&mut next, &before);
    c.clone_from(&next);
    true
}

/// Lengthens a short foot segment (under 0.9 times the bone above it) to 1.0 to
/// 1.3 times the bone above it and stands it nearly vertical, so the heel is
/// held high and the animal walks on its toes. A digitigrade foot (a dog, a
/// cat, a horse) makes the leg longer at no cost in the thigh and gives a long
/// lever for the ankle (Alexander). The new length stops at the longest bone
/// allowed. The toe node loses a fifth of its diameter, down to the smallest
/// size, to keep the swing light.
pub(crate) fn raise_heel_digitigrade(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(leg) = pick_leg(c, rng, |leg| {
        let n = leg.len();
        n >= 2 && c.bones[leg[n - 1]].rest_length < 0.9 * c.bones[leg[n - 2]].rest_length
    }) else {
        return false;
    };
    let n = leg.len();
    let (last, above) = (leg[n - 1], leg[n - 2]);
    let want = (c.bones[above].rest_length * rng.range(1.0, 1.3))
        .min(max_bone_length())
        .max(0.03);
    let factor = want / c.bones[last].rest_length;
    let before = spans(c);
    let mut next = c.clone();
    // Down, with a lean of up to 0.2 rad.
    let turn = angle_between(bone_vector(&next, last), [rng.range(-0.2, 0.2), -1.0]);
    turn_branch(&mut next, last, turn);
    let along = bone_vector(&next, last);
    shift_branch(&mut next, last, scale(along, factor - 1.0));
    next.bones[last].rest_length = want;
    let toe = next.bones[last].b as usize;
    next.nodes[toe].diameter = (next.nodes[toe].diameter * 0.8).max(cfg.min_size);
    lift(&mut next);
    if !inside(&next) {
        return false;
    }
    keep_strokes(&mut next, &before);
    c.clone_from(&next);
    true
}

/// Gives an elastic tendon of 0.4 to 0.9 to the active muscles that end on the
/// last bone of a leg of two or more bones and have a tendon under 0.2. Each
/// muscle gets its own value. The tendon stores the stretch of the muscle as
/// the foot lands and gives it back in the push off, as the Achilles tendon of
/// a running mammal does (Alexander). It makes a bouncing, hopping gait cheap
/// to reach.
pub(crate) fn tendon_the_ankle(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(leg) = pick_leg(c, rng, |leg| leg.len() >= 2) else {
        return false;
    };
    let n = leg.len();
    let last = leg[n - 1];
    let on: MuscleIds = muscles_on(c, &leg[n - 2..], false)
        .into_iter()
        .filter(|&i| {
            let m = &c.muscles[i];
            (m.bone_a as usize == last || m.bone_b as usize == last)
                && drive(m) > 0.0
                && m.tendon < 0.2
        })
        .collect();
    if on.is_empty() {
        return false;
    }
    for i in on {
        c.muscles[i].tendon = rng.range(0.4, 0.9);
    }
    true
}

/// Grows a foot onto the tip of a leg of two or more bones: one short bone,
/// 30 to 60% of the last one and 0.04 to 0.4 long, pointing forward and down by
/// up to 0.6 rad. A new muscle across the ankle takes the timing of the leg's
/// strongest muscle (`leg_template`) with its phase raised by 0.25. The foot
/// gives the leg a toe to push off and a lever for the stance. Its tip is the
/// leg's new end, and the idlest tip elsewhere goes back. The operator refuses
/// when no other tip can go.
pub(crate) fn grow_forward_foot(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    let Some(leg) = pick_leg(c, rng, |leg| leg.len() >= 2) else {
        return false;
    };
    let last = leg[leg.len() - 1];
    let tip = c.bones[last].b as usize;
    let length = (c.bones[last].rest_length * rng.range(0.3, 0.6)).clamp(0.04, 0.4);
    let lean = rng.range(-0.6, 0.0);
    let template = leg_template(c, &leg);
    let mut next = c.clone();
    let at = add(pos(c, tip), [length * lean.cos(), length * lean.sin()]);
    let node = add_node(&mut next, tip, at);
    let mut foot = Bone::new(tip as u32, node as u32, length);
    foot.min_angle = -rng.range(0.3, 0.8);
    foot.max_angle = rng.range(0.3, 0.8);
    next.bones.push(foot);
    let bone = next.bones.len() - 1;
    lift(&mut next);
    if !inside(&next) {
        return false;
    }
    let anchors = (rng.range(0.5, 0.9), rng.range(0.2, 0.7));
    let mut m = new_muscle(&next, last, bone, anchors, template.as_ref(), rng);
    if let Some(t) = template {
        follow(&mut m, t.phase + 0.25);
    }
    next.muscles.push(m);
    if shed_tips(&mut next, c.nodes.len(), 1, rng) < 1 || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Re-cuts a leg of two or more bones in new proportions with the same total
/// length. Each bone is scaled by a weight from 0.75 to 1.35. The weights rise
/// down the leg for a leg that is long at the bottom, as in the running
/// mammals, whose distal bones are long and light, and fall down the leg for
/// one that is long at the top, the digging and kicking form. The total length
/// stays the same unless a bone hits a limit: 0.03 at the least and the
/// longest bone allowed at the most. A leg where every bone would change by
/// less than 8% is left alone. The nodes follow the directions of the bones,
/// and the strokes of muscles keep their ratio.
pub(crate) fn set_leg_proportions(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(leg) = pick_leg(c, rng, |leg| leg.len() >= 2) else {
        return false;
    };
    let n = leg.len();
    let distal = rng.unit() < 0.5;
    // The weight of bone `i`, counted from the top: 0.75 rising to 1.35 at the
    // bottom for a distal leg, and the reverse otherwise.
    let weight = |i: usize| {
        let t = i as f32 / (n - 1) as f32;
        0.75 + 0.6 * if distal { t } else { 1.0 - t }
    };
    let total: f32 = leg.iter().map(|&b| c.bones[b].rest_length).sum();
    let weighted: f32 = (0..n)
        .map(|i| weight(i) * c.bones[leg[i]].rest_length)
        .sum();
    let limit = max_bone_length();
    // How much each bone grows: its weight, scaled so the total length stays
    // the same, kept inside the length limits.
    let factors: Vec<f32> = (0..n)
        .map(|i| {
            let length = c.bones[leg[i]].rest_length;
            (weight(i) * total / weighted).clamp(0.03 / length, (limit / length).max(0.03 / length))
        })
        .collect();
    if factors.iter().all(|f| (f - 1.0).abs() < 0.08) {
        return false;
    }
    let before = spans(c);
    let mut next = c.clone();
    for (i, &bone) in leg.iter().enumerate() {
        let b = c.bones[bone];
        let vector = sub(pos(c, b.b as usize), pos(c, b.a as usize));
        let from = pos(&next, b.a as usize);
        let to = add(from, scale(vector, factors[i]));
        next.nodes[b.b as usize].x = to[0];
        next.nodes[b.b as usize].y = to[1];
        next.bones[bone].rest_length *= factors[i];
    }
    lift(&mut next);
    if !inside(&next) {
        return false;
    }
    keep_strokes(&mut next, &before);
    c.clone_from(&next);
    true
}

/// Gives every foot (the end node of each leg of two or more bones) one form.
/// A hoof is 0.55 to 0.8 times as wide, and its friction rises by 50 to 80% of
/// the way to the maximum. A pad is 1.25 to 1.6 times as wide, and its friction
/// rises by 30 to 60% of the way. Both stay inside the size and friction limits
/// of `cfg`. A small hard tip lands on one point with little swung mass (a
/// horse), and a wide soft pad spreads the load and does not slip (a bear).
/// Grip is a body property, not a fitness term. The operator reports a change
/// only if the diameter or friction of some foot moves by more than 0.0001.
pub(crate) fn harden_or_pad_foot(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut legs = leaf_limbs(c);
    legs.retain(|leg| leg.len() >= 2);
    if legs.is_empty() {
        return false;
    }
    let hoof = rng.unit() < 0.5;
    let (size, grip) = if hoof {
        (rng.range(0.55, 0.8), rng.range(0.5, 0.8))
    } else {
        (rng.range(1.25, 1.6), rng.range(0.3, 0.6))
    };
    let mut changed = false;
    for leg in &legs {
        let foot = c.bones[leg[leg.len() - 1]].b as usize;
        let node = &mut c.nodes[foot];
        let diameter = (node.diameter * size).clamp(cfg.min_size, cfg.max_size);
        let friction = (node.friction + grip * (cfg.max_friction - node.friction))
            .clamp(cfg.min_friction, cfg.max_friction);
        changed |=
            (diameter - node.diameter).abs() > 1.0e-4 || (friction - node.friction).abs() > 1.0e-4;
        node.diameter = diameter;
        node.friction = friction;
    }
    changed
}

/// Retimes the knee muscles of each leg of two or more bones that has both a
/// hip muscle and a knee muscle. A hip muscle joins the leg's top bone to a
/// bone outside the leg, and a knee muscle joins two bones of the leg. The knee
/// muscles of a leg move together, so that the strongest of them has a phase
/// 0.25 above the phase of the strongest hip muscle. In 3 calls of 10 the lag
/// is 0.75 instead, in every leg of that call. A leg that is already within
/// 0.001 of its target is left alone. The knee flexes as the hip swings, a
/// phase relation that central pattern generators of walking animals keep fixed
/// (Ijspeert), so the leg is a coordinated whole.
pub(crate) fn lag_knee_behind_hip(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let lag = if rng.unit() < 0.7 { 0.25 } else { 0.75 };
    let mut changed = false;
    for leg in leaf_limbs(c) {
        if leg.len() < 2 {
            continue;
        }
        let hip: MuscleIds = (0..c.muscles.len())
            .filter(|&i| {
                let m = &c.muscles[i];
                let (x, y) = (m.bone_a as usize, m.bone_b as usize);
                (x == leg[0] && !leg.contains(&y)) || (y == leg[0] && !leg.contains(&x))
            })
            .collect();
        let knee = muscles_on(c, &leg, true);
        let (Some(h), Some(k)) = (strongest(c, &hip), strongest(c, &knee)) else {
            continue;
        };
        let shift = turn(c.muscles[k].phase, c.muscles[h].phase + lag);
        if shift.abs() < 1.0e-3 {
            continue;
        }
        for &i in &knee {
            let m = &mut c.muscles[i];
            m.phase = (m.phase + shift).rem_euclid(1.0);
            m.reset = (m.reset + shift).rem_euclid(1.0);
        }
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::super::tests::bodies;
    use super::*;
    use crate::evolution::repair;

    /// Runs every operator of this file on 120 test bodies. A body an operator
    /// changes stays within the node and muscle caps and has one bone fewer
    /// than nodes, and a body it refuses is left as it was.
    #[test]
    fn leg_operators_keep_the_body_valid() {
        let cfg = Config::default();
        for (name, op) in OPS {
            let mut applied = 0;
            for (i, body) in bodies(&cfg, 120).into_iter().enumerate() {
                let mut c = body.clone();
                let cx = Context::of(None);
                if op(&mut c, &cfg, &mut Rng::new(71, 0, i), &cx) {
                    applied += 1;
                    assert!(c.nodes.len() <= cfg.max_nodes && c.muscles.len() <= cfg.max_muscles);
                    assert_eq!(c.bones.len() + 1, c.nodes.len(), "{name}");
                    repair(&mut c, &cfg, &mut Rng::new(73, 0, i));
                } else {
                    assert!(
                        c.nodes == body.nodes && c.bones == body.bones && c.muscles == body.muscles,
                        "{name} changed a body it refused"
                    );
                }
            }
            eprintln!("{name}: {applied} of 120");
        }
    }
}
