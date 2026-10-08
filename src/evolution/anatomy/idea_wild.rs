//! Idea operators that try something odd: flip a whole body, leave one leg
//! to do all the work, add rubber bands, mute a limb's reflexes, scramble a
//! limb's timing, and adapt a body to the slope or the hurdles of its world.
//!
//! The operators of this file share one pick slot (`GAIT_FILES` in `mod.rs`),
//! so a poor one costs little. Each is a whole change on its own, so no
//! parameter noise follows it. Most are experiments that may lose, and only
//! `arms_against_legs` names a source (Herr and Popovic 2008, arms swung
//! against the legs).
use super::ideas::{by_drive, coin, scale_branch, set, some_leg, wrap};
use super::limbs::pick;
use super::rhythm::leaf_limbs;
use super::{BoneIds, Context, Operator, is_neck, muscles_on, new_muscle, room};
use crate::config::Config;
use crate::evolution::{Creature, NO_SENSOR, Rng};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("mirror_flip_body", mirror_flip_body),
    ("flip_joint_ranges", flip_joint_ranges),
    ("lean_trunk", lean_trunk),
    ("slope_lean", slope_lean),
    ("limb_roulette", limb_roulette),
    ("muscle_confetti", muscle_confetti),
    ("scooter_mode", scooter_mode),
    ("arms_against_legs", arms_against_legs),
    ("hurdle_legs", hurdle_legs),
    ("pogo_everything", pogo_everything),
    ("elastic_bands", elastic_bands),
    ("follow_the_strongest", follow_the_strongest),
    ("equal_strength", equal_strength),
    ("sense_other_end", sense_other_end),
    ("numb_limb", numb_limb),
];

/// The whole body is reflected about the head's vertical line, with every
/// joint range reflected too. Apart from effects with a direction, such as a
/// slope or a wind, the physics is the same both ways, so a body that walks
/// backward walks forward after this, and a body that walks forward walks
/// backward: a cheap way to turn around. Does nothing to a body whose nodes all
/// lie within 3 cm of that line.
fn mirror_flip_body(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let x0 = c.nodes[0].x;
    if c.nodes.iter().all(|n| (n.x - x0).abs() < 0.03) {
        return false;
    }
    for n in &mut c.nodes {
        n.x = 2.0 * x0 - n.x;
    }
    for b in &mut c.bones {
        (b.min_angle, b.max_angle) = (-b.max_angle, -b.min_angle);
    }
    true
}

/// The joints of one leg bend the other way, or, half the time, those of every
/// bone but the neck: each range (min, max) becomes (-max, -min), so a knee
/// that bent forward bends back. A body with its nodes where they were stands
/// the same and moves differently.
fn flip_joint_ranges(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let bones: BoneIds = match some_leg(c, rng, 1, false) {
        Some(leg) if coin(rng) => leg,
        _ => (0..c.bones.len()).filter(|&b| !is_neck(c, b)).collect(),
    };
    let mut changed = false;
    for &b in &bones {
        let bone = &mut c.bones[b];
        let (low, high) = (-bone.max_angle, -bone.min_angle);
        set(&mut bone.min_angle, low, &mut changed);
        set(&mut bone.max_angle, high, &mut changed);
    }
    changed
}

/// Shifts every node above the median height along x, by `by` times its height
/// above the median (forward when `by` is positive): a lean of the upper body.
/// The median is the higher middle one when the count is even. The head is left
/// out of both the median and the shift, and a body with fewer than 3 nodes
/// besides the head is left alone. Returns whether any node moved.
fn lean(c: &mut Creature, by: f32) -> bool {
    let mut ys: Vec<f32> = (1..c.nodes.len()).map(|n| c.nodes[n].y).collect();
    if ys.len() < 3 {
        return false;
    }
    ys.sort_by(|a, b| a.total_cmp(b));
    let median = ys[ys.len() / 2];
    let mut changed = false;
    for n in 1..c.nodes.len() {
        let above = (c.nodes[n].y - median).max(0.0);
        let x = c.nodes[n].x + by * above;
        set(&mut c.nodes[n].x, x, &mut changed);
    }
    changed
}

/// The upper body leans forward or back: each node above the median height
/// shifts by 0.15 to 0.4 of its height above it.
fn lean_trunk(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = rng.range(0.15, 0.4) * if coin(rng) { 1.0 } else { -1.0 };
    lean(c, by)
}

/// On a slope the upper body leans into the hill, each node shifting by 0.15 to
/// 0.35 of its height above the median: forward when the ground rises ahead,
/// back when it falls. Does nothing when `Config::slope` (rise over run) is
/// within 0.01 of zero.
fn slope_lean(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if cfg.slope.abs() < 0.01 {
        return false;
    }
    lean(c, rng.range(0.15, 0.35) * cfg.slope.signum())
}

/// One leg that has a muscle is picked, and every muscle with an end on it
/// takes a random phase: a limb gone out of step.
fn limb_roulette(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 1, true) else {
        return false;
    };
    let mut changed = false;
    for m in muscles_on(c, &leg, false) {
        set(&mut c.muscles[m].phase, rng.unit(), &mut changed);
    }
    changed
}

/// Up to three muscles join random pairs of bones. Each joins two different
/// bones that no muscle joins yet, at random anchors, and has a random rhythm
/// of its own. A body with fewer than 3 bones is left alone.
fn muscle_confetti(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let bones = c.bones.len();
    if bones < 3 || !room(c, cfg, 0, 1) {
        return false;
    }
    let wanted = 1 + rng.index(3);
    let mut added = 0;
    for _ in 0..wanted * 3 {
        if added == wanted || !room(c, cfg, 0, 1) {
            break;
        }
        let (a, b) = (rng.index(bones), rng.index(bones));
        let taken = c.muscles.iter().any(|m| {
            (m.bone_a as usize, m.bone_b as usize) == (a, b)
                || (m.bone_a as usize, m.bone_b as usize) == (b, a)
        });
        if a == b || taken {
            continue;
        }
        let anchors = (rng.unit(), rng.unit());
        let m = new_muscle(c, a, b, anchors, None, rng);
        c.muscles.push(m);
        added += 1;
    }
    added > 0
}

/// One leg that has a muscle keeps its drive. Every muscle with an end on
/// another leg and none on that one goes slack, a weak spring each (a stiffness
/// of 1 and a tendon of 0.8), so the others trail as passive struts, like a
/// scooter's foot on the ground. Does nothing with fewer than two legs.
fn scooter_mode(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leaf_limbs(c);
    if legs.len() < 2 {
        return false;
    }
    let Some(keep) = some_leg(c, rng, 1, true) else {
        return false;
    };
    let kept = muscles_on(c, &keep, false);
    let mut changed = false;
    for leg in legs.iter().filter(|l| l[0] != keep[0]) {
        for m in muscles_on(c, leg, false) {
            if kept.contains(&m) {
                continue;
            }
            set(&mut c.muscles[m].stiffness, 1.0, &mut changed);
            set(&mut c.muscles[m].tendon, 0.8, &mut changed);
        }
    }
    changed
}

/// Every muscle with an end on a limb whose root sits above the median root
/// height (an arm) starts half a cycle later, so an arm that swung with the
/// legs now swings against them (Herr and Popovic 2008). The median is the
/// higher middle one when the count is even, so the operator needs at least
/// three limbs to change anything. A muscle with an end on two arms moves once.
fn arms_against_legs(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let limbs = leaf_limbs(c);
    if limbs.len() < 2 {
        return false;
    }
    let root_y = |l: &BoneIds| c.nodes[c.bones[l[0]].a as usize].y;
    let mut heights: Vec<f32> = limbs.iter().map(root_y).collect();
    heights.sort_by(|a, b| a.total_cmp(b));
    let median = heights[heights.len() / 2];
    let mut moved = vec![false; c.muscles.len()];
    let mut changed = false;
    for limb in limbs.iter().filter(|l| root_y(l) > median) {
        for m in muscles_on(c, limb, false) {
            if !moved[m] {
                moved[m] = true;
                let phase = wrap(c.muscles[m].phase + 0.5);
                set(&mut c.muscles[m].phase, phase, &mut changed);
            }
        }
    }
    changed
}

/// Each leg shorter than 2.5 times the hurdle height grows toward that length,
/// by at most 1.4 times and within the bone limits (`scale_branch`): a leg that
/// cannot clear a step is no use. A leg's length is the sum of its bone
/// lengths. Does nothing when the hurdle height (`Config::hurdles`) is under
/// 0.01 m.
fn hurdle_legs(c: &mut Creature, cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if cfg.hurdles < 0.01 {
        return false;
    }
    let needed = 2.5 * cfg.hurdles;
    let mut changed = false;
    for leg in leaf_limbs(c).iter() {
        let length: f32 = leg.iter().map(|&b| c.bones[b].rest_length).sum();
        if length < needed {
            changed |= scale_branch(c, leg[0], (needed / length).min(1.4));
        }
    }
    changed
}

/// Every muscle gets a tendon of at least 0.8, a duty of a quarter and the
/// median period, and the whole body takes the phase of the first muscle: a
/// pogo stick.
fn pogo_everything(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.is_empty() {
        return false;
    }
    let mut periods: Vec<f32> = c.muscles.iter().map(|m| m.period).collect();
    periods.sort_by(|a, b| a.total_cmp(b));
    let period = periods[periods.len() / 2];
    let phase = c.muscles[0].phase;
    let mut changed = false;
    for m in &mut c.muscles {
        let tendon = m.tendon.max(0.8);
        set(&mut m.tendon, tendon, &mut changed);
        set(&mut m.duty, 0.25, &mut changed);
        set(&mut m.period, period, &mut changed);
        set(&mut m.phase, phase, &mut changed);
    }
    changed
}

/// Up to three rubber bands: passive muscles (a stiffness of 1) with the
/// stiffest tendon, each anchored at the middle of two bones that share a node
/// and have no muscle between them. A band has the rhythm of a random muscle.
fn elastic_bands(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.is_empty() || !room(c, cfg, 0, 1) {
        return false;
    }
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for x in 0..c.bones.len() {
        for y in x + 1..c.bones.len() {
            let (p, q) = (c.bones[x], c.bones[y]);
            let adjacent = p.a == q.a || p.a == q.b || p.b == q.a || p.b == q.b;
            let taken = c.muscles.iter().any(|m| {
                (m.bone_a as usize, m.bone_b as usize) == (x, y)
                    || (m.bone_a as usize, m.bone_b as usize) == (y, x)
            });
            if adjacent && !taken {
                pairs.push((x, y));
            }
        }
    }
    let wanted = 1 + rng.index(3);
    let mut added = 0;
    while added < wanted && !pairs.is_empty() && room(c, cfg, 0, 1) {
        let (x, y) = pairs.swap_remove(rng.index(pairs.len()));
        let mut template = c.muscles[rng.index(c.muscles.len())];
        template.stiffness = 1.0;
        template.tendon = 1.0;
        let m = new_muscle(c, x, y, (0.5, 0.5), Some(&template), rng);
        c.muscles.push(m);
        added += 1;
    }
    added > 0
}

/// The strongest muscle (by `drive`, stiffness times stroke) gives its duty and
/// its tendon to every muscle that shares a bone with it.
fn follow_the_strongest(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 2 {
        return false;
    }
    let best = by_drive(c)[0];
    let lead = c.muscles[best];
    let mut changed = false;
    for i in 0..c.muscles.len() {
        let m = c.muscles[i];
        let near = m.bone_a == lead.bone_a
            || m.bone_a == lead.bone_b
            || m.bone_b == lead.bone_a
            || m.bone_b == lead.bone_b;
        if i != best && near {
            set(&mut c.muscles[i].duty, lead.duty, &mut changed);
            set(&mut c.muscles[i].tendon, lead.tendon, &mut changed);
        }
    }
    changed
}

/// Every muscle gets the median stiffness: no muscle is the strong one.
fn equal_strength(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 2 {
        return false;
    }
    let mut stiffness: Vec<f32> = c.muscles.iter().map(|m| m.stiffness).collect();
    stiffness.sort_by(|a, b| a.total_cmp(b));
    let median = stiffness[stiffness.len() / 2];
    let mut changed = false;
    for m in &mut c.muscles {
        set(&mut m.stiffness, median, &mut changed);
    }
    changed
}

/// Touchdown sensors move to the other end of their bone (a `sensor` of 0 and 1
/// swap, as do 2 and 3): a foot that sensed its landing at the toe senses it at
/// the heel.
fn sense_other_end(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let mut changed = false;
    for m in &mut c.muscles {
        if m.sensor != NO_SENSOR {
            m.sensor ^= 1;
            changed = true;
        }
    }
    changed
}

/// One leg that has a sensing muscle is picked, and every muscle with an end on
/// it loses its touchdown sensor and runs open-loop.
fn numb_limb(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let sensing: Vec<BoneIds> = leaf_limbs(c)
        .iter()
        .filter(|l| {
            muscles_on(c, l, false)
                .iter()
                .any(|&m| c.muscles[m].sensor != NO_SENSOR)
        })
        .copied()
        .collect();
    let Some(leg) = pick(&sensing, rng) else {
        return false;
    };
    for m in muscles_on(c, &leg, false) {
        c.muscles[m].sensor = NO_SENSOR;
    }
    true
}
