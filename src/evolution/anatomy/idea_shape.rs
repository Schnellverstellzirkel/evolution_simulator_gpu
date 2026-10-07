//! Idea operators for the proportions of bones and the ranges of joints:
//! golden and equal bones, tall or short legs, stubby or long toes, a knee
//! that cannot bend backward, joints locked or opened. A leg is a limb that
//! ends in a foot (`leaf_limbs`).
//!
//! Every operator is a whole change on its own, so its child gets no
//! parameter noise, and the operators of this file share one pick slot
//! (`GAIT_FILES` in `mod.rs`).
//!
//! The sources are Alexander (2003, leg segments in a ratio that suits the
//! gait), Thompson (1917, growth by proportion), and Sims (1994, joint
//! ranges as a gene).
use super::ideas::{bone_length, coin, scale_branch, set, some_leg};
use super::limbs::pick;
use super::rhythm::leaf_limbs;
use super::{BoneIds, Context, Operator, is_neck};
use crate::config::Config;
use crate::evolution::{Creature, JOINT_LIMIT, Rng};

/// This file's operators, by name. Add each new one here. The pick slot of
/// this file chooses by position in this list, so the order decides what a
/// fixed seed picks.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("golden_leg", golden_leg),
    ("equal_bones_leg", equal_bones_leg),
    ("leg_scale_all", leg_scale_all),
    ("range_center_leg", range_center_leg),
    ("range_breathe", range_breathe),
    ("knee_stop", knee_stop),
    ("hip_wide_knee_narrow", hip_wide_knee_narrow),
    ("bone_size_nudge", bone_size_nudge),
    ("toe_length_swing", toe_length_swing),
    ("thigh_shank_swap", thigh_shank_swap),
    ("lock_one_joint", lock_one_joint),
    ("open_one_joint", open_one_joint),
];

/// A leg of two bones or more gets bone lengths in a golden-ratio progression.
/// Each bone is 0.618 times as long as the one before it, counted from the
/// root or from the foot, and a coin picks which. The leg keeps its total
/// length, up to the bone length limits. The golden ratio is a trial with no
/// theory behind it.
fn golden_leg(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 2, false) else {
        return false;
    };
    let total: f32 = leg.iter().map(|&b| c.bones[b].rest_length).sum();
    let ratio = 0.618_034f32;
    let weights: Vec<f32> = (0..leg.len()).map(|k| ratio.powi(k as i32)).collect();
    let sum: f32 = weights.iter().sum();
    let root_first = coin(rng);
    let mut changed = false;
    for (k, &b) in leg.iter().enumerate() {
        let w = if root_first {
            weights[k]
        } else {
            weights[leg.len() - 1 - k]
        };
        set(
            &mut c.bones[b].rest_length,
            bone_length(total * w / sum),
            &mut changed,
        );
    }
    changed
}

/// A leg of two bones or more gives all its bones the leg's mean length.
fn equal_bones_leg(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 2, false) else {
        return false;
    };
    let mean = leg.iter().map(|&b| c.bones[b].rest_length).sum::<f32>() / leg.len() as f32;
    let mut changed = false;
    for &b in &leg {
        set(&mut c.bones[b].rest_length, bone_length(mean), &mut changed);
    }
    changed
}

/// Every leg scales about its hip by one factor, 1.1 to 1.3 or 0.8 to 0.9 (a
/// coin picks which), and the strokes of the muscles inside it scale with it.
/// This makes the whole animal taller or lower on its legs.
fn leg_scale_all(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let factor = if coin(rng) {
        rng.range(1.1, 1.3)
    } else {
        rng.range(0.8, 0.9)
    };
    let mut changed = false;
    for leg in leaf_limbs(c).iter() {
        changed |= scale_branch(c, leg[0], factor);
    }
    changed
}

/// Every joint of a leg centers its range on the starting pose: both sides get
/// the mean of the two reaches.
fn range_center_leg(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 1, false) else {
        return false;
    };
    let mut changed = false;
    for &b in &leg {
        let reach = 0.5 * (c.bones[b].max_angle - c.bones[b].min_angle);
        set(&mut c.bones[b].min_angle, -reach, &mut changed);
        set(&mut c.bones[b].max_angle, reach, &mut changed);
    }
    changed
}

/// Every joint range but the neck's narrows by 20% or widens by 20%, up to
/// `JOINT_LIMIT`.
fn range_breathe(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = if coin(rng) { 0.8 } else { 1.2 };
    let mut changed = false;
    for b in 0..c.bones.len() {
        if is_neck(c, b) {
            continue;
        }
        let low = (c.bones[b].min_angle * by).clamp(-JOINT_LIMIT, 0.0);
        let high = (c.bones[b].max_angle * by).clamp(0.0, JOINT_LIMIT);
        set(&mut c.bones[b].min_angle, low, &mut changed);
        set(&mut c.bones[b].max_angle, high, &mut changed);
    }
    changed
}

/// The second joint of a leg may bend only the way its leg already folds: the
/// range on the other side closes to 0.05 rad, so the knee cannot bend
/// backward.
fn knee_stop(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 2, false) else {
        return false;
    };
    let (upper, lower) = (c.bones[leg[0]], c.bones[leg[1]]);
    let vector = |b: crate::evolution::Bone| {
        let (a, z) = (c.nodes[b.a as usize], c.nodes[b.b as usize]);
        (z.x - a.x, z.y - a.y)
    };
    let (u, l) = (vector(upper), vector(lower));
    // The sign of the turn from the upper bone to the lower tells which way
    // the knee is bent now. The range closes on the opposite side.
    let turn = u.0 * l.1 - u.1 * l.0;
    let knee = &mut c.bones[leg[1]];
    let mut changed = false;
    if turn >= 0.0 {
        set(&mut knee.min_angle, -0.05, &mut changed);
    } else {
        set(&mut knee.max_angle, 0.05, &mut changed);
    }
    changed
}

/// In every leg of two bones or more the root joint widens by half of its
/// room to the limit and the other joints narrow by 30%: a hip that swings wide
/// and lower joints that move less.
fn hip_wide_knee_narrow(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let mut changed = false;
    for leg in leaf_limbs(c).iter().filter(|l| l.len() >= 2) {
        for (k, &b) in leg.iter().enumerate() {
            let bone = &mut c.bones[b];
            let (low, high) = if k == 0 {
                (
                    bone.min_angle - 0.5 * (JOINT_LIMIT + bone.min_angle),
                    bone.max_angle + 0.5 * (JOINT_LIMIT - bone.max_angle),
                )
            } else {
                (bone.min_angle * 0.7, bone.max_angle * 0.7)
            };
            set(
                &mut bone.min_angle,
                low.clamp(-JOINT_LIMIT, 0.0),
                &mut changed,
            );
            set(
                &mut bone.max_angle,
                high.clamp(0.0, JOINT_LIMIT),
                &mut changed,
            );
        }
    }
    changed
}

/// Every bone but the neck changes length by the same 5 to 10%, all longer or
/// all shorter.
fn bone_size_nudge(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = if coin(rng) {
        rng.range(1.05, 1.1)
    } else {
        rng.range(0.9, 0.95)
    };
    let mut changed = false;
    for b in 0..c.bones.len() {
        if !is_neck(c, b) {
            let length = bone_length(c.bones[b].rest_length * by);
            set(&mut c.bones[b].rest_length, length, &mut changed);
        }
    }
    changed
}

/// The last bone of every leg of two bones or more shortens to 70% (stubby
/// toes) or lengthens by 35% (long toes).
fn toe_length_swing(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = if coin(rng) { 0.7 } else { 1.35 };
    let mut changed = false;
    for leg in leaf_limbs(c).iter().filter(|l| l.len() >= 2) {
        let toe = leg[leg.len() - 1];
        let length = bone_length(c.bones[toe].rest_length * by);
        set(&mut c.bones[toe].rest_length, length, &mut changed);
    }
    changed
}

/// The first two bones of a leg swap lengths: the thigh takes the shank's
/// length and the shank takes the thigh's. A leg whose two bones differ by less
/// than 0.01 m stays as it is.
fn thigh_shank_swap(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 2, false) else {
        return false;
    };
    let (x, y) = (c.bones[leg[0]].rest_length, c.bones[leg[1]].rest_length);
    if (x - y).abs() < 0.01 {
        return false;
    }
    c.bones[leg[0]].rest_length = bone_length(y);
    c.bones[leg[1]].rest_length = bone_length(x);
    true
}

/// One joint of a leg of two bones or more locks (a range of 0.05 rad each
/// way): a rigid link in place of a hinge. Only a joint with a range wider than
/// 0.2 rad can lock.
fn lock_one_joint(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 2, false) else {
        return false;
    };
    let wide: BoneIds = leg
        .iter()
        .copied()
        .filter(|&b| c.bones[b].max_angle - c.bones[b].min_angle > 0.2)
        .collect();
    let Some(b) = pick(&wide, rng) else {
        return false;
    };
    c.bones[b].min_angle = -0.05;
    c.bones[b].max_angle = 0.05;
    true
}

/// One joint of a leg opens to the widest range: a free hinge. Only a joint
/// whose range is more than 0.2 rad short of the widest can open.
fn open_one_joint(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 1, false) else {
        return false;
    };
    let tight: BoneIds = leg
        .iter()
        .copied()
        .filter(|&b| c.bones[b].max_angle - c.bones[b].min_angle < 2.0 * JOINT_LIMIT - 0.2)
        .collect();
    let Some(b) = pick(&tight, rng) else {
        return false;
    };
    c.bones[b].min_angle = -JOINT_LIMIT;
    c.bones[b].max_angle = JOINT_LIMIT;
    true
}
