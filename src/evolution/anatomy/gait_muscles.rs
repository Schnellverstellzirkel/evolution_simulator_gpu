//! Gait operators: muscle layouts that make strokes efficient: antagonists, two-joint muscles, springs.
//!
//! The operators of this file share one pick slot and are compound: each is a
//! whole, coherent change to the body, and its child gets no parameter noise.
//!
//! The sources are Sims (1994, muscles as pulling pairs), Alexander (1984 and
//! 1988, two-joint muscles, tendons that store the energy of a stride, and
//! light distal limbs), Full and Koditschek (1999, a stiff stance leg and a
//! light swing leg as one template) and Cheney et al. (2013, repeated
//! parts). None of the operators adds a node. A muscle waveform rises from
//! long to short over `duty` of the cycle, starting at `phase`, and falls over
//! the rest, so a muscle with a long duty pulls slowly and steadily and one
//! with a short duty pulls quickly.
use super::compound::strongest;
use super::extra::drive;
use super::limbs::pick;
use super::muscles::{flipped, ring, shared_node, shift_timing, torque, turn};
use super::rhythm::leaf_limbs;
use super::{
    BoneIds, Context, MuscleIds, Operator, branch, child_bones, fit_stroke, is_neck, muscles_on,
    new_muscle, parent_bones, pick_each, room, span,
};
use crate::bounded::Bounded;
use crate::config::Config;
use crate::evolution::{Creature, MAX_MUSCLES, Muscle, NO_SENSOR, Rng, bone_point};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("reciprocal_extensor", reciprocal_extensor),
    ("hip_knee_strap", hip_knee_strap),
    ("stance_swing_split", stance_swing_split),
    ("stance_swing_roles", stance_swing_roles),
    ("improve_lever_arm", improve_lever_arm),
    ("muscle_to_all_legs", muscle_to_all_legs),
    ("stance_cocontraction", stance_cocontraction),
    ("elastic_shank_tendon", elastic_shank_tendon),
    ("lock_antagonist_pairs", lock_antagonist_pairs),
    ("push_off_muscle", push_off_muscle),
    ("repurpose_idle_muscle", repurpose_idle_muscle),
    ("catapult_release", catapult_release),
    ("second_hip_anchor", second_hip_anchor),
];

/// Least torque (lever times pull, in m^2) that counts as turning a bone.
const MIN_TORQUE: f32 = 1.0e-3;

/// Whether a muscle shortens at all.
fn active(m: &Muscle) -> bool {
    m.long > m.short + 1.0e-4
}

/// Every leg with the bone above its first bone.
fn legs_with_hip(c: &Creature) -> Vec<(BoneIds, usize)> {
    let parents = parent_bones(c);
    leaf_limbs(c)
        .iter()
        .copied()
        .filter_map(|limb| {
            let above = parents[c.bones[limb[0]].a as usize]?;
            Some((limb, above))
        })
        .collect()
}

/// The active muscles with both ends on `bones`.
fn active_between(c: &Creature, bones: &[usize]) -> MuscleIds {
    muscles_on(c, bones, true)
        .into_iter()
        .filter(|&i| active(&c.muscles[i]))
        .collect()
}

/// The active muscles with both ends on a leg or the bone above it.
fn leg_muscles(c: &Creature, limb: &[usize], above: usize) -> MuscleIds {
    let mut bones = BoneIds::from_slice(&[above]);
    bones.extend(limb.iter().copied());
    active_between(c, &bones)
}

/// Whether an active muscle joins bones `x` and `z`.
fn joined(c: &Creature, x: usize, z: usize) -> bool {
    c.muscles.iter().any(|m| {
        let ends = (m.bone_a as usize, m.bone_b as usize);
        (ends == (x, z) || ends == (z, x)) && active(m)
    })
}

/// The muscle with `bone` as its `bone_a`, if it has an end on `bone`.
fn facing(m: &Muscle, bone: usize) -> Option<Muscle> {
    if m.bone_a as usize == bone {
        Some(*m)
    } else if m.bone_b as usize == bone {
        Some(flipped(m))
    } else {
        None
    }
}

/// The phase halfway between two phases, along the shorter way round.
fn mean_phase(x: f32, y: f32) -> f32 {
    (x + 0.5 * turn(x, y)).rem_euclid(1.0)
}

/// A closing muscle of a leg joint, a bone `r` outside the joint's limb and
/// an anchor on it from which a muscle would open the joint: its torque on
/// the joint runs the other way. The closer is turned so its `bone_a` is the
/// limb's bone.
fn find_opener(c: &Creature, rng: &mut Rng) -> Option<(Muscle, usize, f32)> {
    let legs = legs_with_hip(c);
    let parents = parent_bones(c);
    pick_each(rng, |push: &mut dyn FnMut((Muscle, usize, f32))| {
        for (limb, above) in &legs {
            let on = leg_muscles(c, limb, *above);
            for &k in limb.iter() {
                let Some(p) = parents[c.bones[k].a as usize] else {
                    continue;
                };
                let below = branch(c, k);
                for &i in on.iter() {
                    let Some(closer) = facing(&c.muscles[i], k) else {
                        continue;
                    };
                    if closer.bone_b as usize != p {
                        continue;
                    }
                    let closing = torque(c, &closer);
                    if closing.abs() < MIN_TORQUE {
                        continue;
                    }
                    for r in (0..c.bones.len()).filter(|r| !below.contains(r)) {
                        if joined(c, k, r) {
                            continue;
                        }
                        for anchor in [0.0, 0.5, 1.0] {
                            let opener = Muscle {
                                bone_b: r as u32,
                                anchor_b: anchor,
                                ..closer
                            };
                            let opening = torque(c, &opener);
                            if opening * closing < 0.0 && opening.abs() >= MIN_TORQUE {
                                push((closer, r, anchor));
                            }
                        }
                    }
                }
            }
        }
    })
}

/// The muscle that would open the joint of `closer` through bone `r` at
/// `anchor`, with the closer's timing and no touchdown sensor.
fn opener_muscle(c: &Creature, closer: &Muscle, r: usize, anchor: f32, rng: &mut Rng) -> Muscle {
    let mut m = new_muscle(
        c,
        closer.bone_a as usize,
        r,
        (closer.anchor_a, anchor),
        Some(closer),
        rng,
    );
    m.sensor = NO_SENSOR;
    m
}

/// Gives `m` the reciprocal rhythm of `closer`: it rises over the closer's
/// rest, starts when the closer's rise ends, and has 0.7 of its strength.
fn make_reciprocal(m: &mut Muscle, closer: &Muscle) {
    m.duty = (1.0 - closer.duty).clamp(0.15, 0.85);
    m.phase = (closer.phase + closer.duty).rem_euclid(1.0);
    m.reset = m.phase;
    m.stiffness = (closer.stiffness * 0.7).clamp(1.0, 120.0);
}

/// Adds the extensor of a leg joint that has a flexor: a muscle from the
/// leg's bone to a bone on the other side of the joint, running in the
/// reciprocal rhythm. It rises while the flexor falls (its duty is the
/// flexor's rest, it starts when the flexor's rise ends) and has 0.7 of the
/// flexor's strength. A joint that a muscle can both close and open keeps a
/// leg moving without a stop to fall back on, as the flexor and extensor
/// pairs of mammal limbs do (Sims 1994, muscles in opposed pairs).
fn reciprocal_extensor(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let Some((closer, r, anchor)) = find_opener(c, rng) else {
        return false;
    };
    let mut m = opener_muscle(c, &closer, r, anchor, rng);
    make_reciprocal(&mut m, &closer);
    c.muscles.push(m);
    true
}

/// Adds a muscle across two joints of a leg, from the bone above the thigh to
/// the shank (or from the thigh to the next bone down, on a longer leg), like
/// the rectus femoris and the gastrocnemius. One contraction then moves hip
/// and knee together. Its phase is halfway between the phases of the muscles
/// at the two joints, it has half the strength of the weaker of them, and it
/// carries an elastic tendon of 0.5, as these long muscles of mammals end in
/// long tendons (Alexander 1988).
fn hip_knee_strap(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let legs = legs_with_hip(c);
    // (proximal bone, distal bone, muscle at the proximal joint, muscle at the distal joint)
    let Some((x, z, up, down)) =
        pick_each(rng, |push: &mut dyn FnMut((usize, usize, usize, usize))| {
            for (limb, above) in &legs {
                for level in 1..limb.len() {
                    let x = if level == 1 { *above } else { limb[level - 2] };
                    let (mid, z) = (limb[level - 1], limb[level]);
                    if joined(c, x, z) {
                        continue;
                    }
                    let upper = active_between(c, &[x, mid]);
                    let lower = active_between(c, &[mid, z]);
                    if let (Some(u), Some(d)) = (strongest(c, &upper), strongest(c, &lower)) {
                        push((x, z, u, d));
                    }
                }
            }
        })
    else {
        return false;
    };
    let (up, down) = (c.muscles[up], c.muscles[down]);
    let template = if drive(&up) >= drive(&down) { up } else { down };
    let anchors = (rng.range(0.3, 0.8), rng.range(0.2, 0.5));
    let mut m = new_muscle(c, x, z, anchors, Some(&template), rng);
    fit_stroke(c, &mut m, None);
    m.sensor = NO_SENSOR;
    m.phase = mean_phase(up.phase, down.phase);
    m.reset = m.phase;
    m.stiffness = (0.5 * up.stiffness.min(down.stiffness)).clamp(1.0, 120.0);
    m.tendon = m.tendon.max(0.5);
    c.muscles.push(m);
    true
}

/// A leg's strongest muscle across its hip (the joint to the bone above).
fn hinge_driver(c: &Creature, limb: &[usize], above: usize) -> Option<usize> {
    strongest(c, &active_between(c, &[limb[0], above]))
}

/// Splits a leg's hip work into a stance muscle and a swing muscle. The
/// existing hip muscle becomes the stance muscle: strong (1.2 times) and slow
/// (a rise over 0.6 to 0.75 of the cycle, which holds the foot on the ground
/// and pushes steadily). A new light muscle (0.4 of the strength, a rise over
/// a quarter of the cycle, 60% of the stroke, attached 0.15 nearer the joint)
/// starts when the stance rise ends and brings the leg back quickly. This is
/// the stiff stance leg and light swing leg of Full and Koditschek's template
/// of running animals, and the unequal times of stance and swing in
/// Alexander's duty factor.
fn stance_swing_split(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let legs: Vec<(usize, usize)> = legs_with_hip(c)
        .iter()
        .filter_map(|(limb, above)| Some((hinge_driver(c, limb, *above)?, limb[0])))
        .collect();
    let Some((i, root)) = pick(&legs, rng) else {
        return false;
    };
    let Some(stance) = facing(&c.muscles[i], root) else {
        return false;
    };
    // An anchor moved 0.15 of a bone nearer the joint the two bones share.
    let nearer = |bone: u32, anchor: f32| -> f32 {
        match shared_node(c, stance.bone_a as usize, stance.bone_b as usize) {
            Some(n) if c.bones[bone as usize].a == n => (anchor - 0.15).max(0.0),
            Some(_) => (anchor + 0.15).min(1.0),
            None => anchor,
        }
    };
    let anchors = (
        nearer(stance.bone_a, stance.anchor_a),
        nearer(stance.bone_b, stance.anchor_b),
    );
    let mut swing = new_muscle(
        c,
        stance.bone_a as usize,
        stance.bone_b as usize,
        anchors,
        Some(&stance),
        rng,
    );
    let duty = stance.duty.clamp(0.6, 0.75);
    let trim = 0.2 * (swing.long - swing.short);
    swing.short += trim;
    swing.long -= trim;
    swing.duty = 0.25;
    swing.phase = (stance.phase + duty).rem_euclid(1.0);
    swing.reset = swing.phase;
    swing.sensor = NO_SENSOR;
    swing.stiffness = (stance.stiffness * 0.4).clamp(1.0, 120.0);
    swing.tendon = 0.0;
    let slow = &mut c.muscles[i];
    slow.duty = duty;
    slow.stiffness = (slow.stiffness * 1.2).clamp(1.0, 120.0);
    c.muscles.push(swing);
    true
}

/// Gives the muscles of one leg unequal jobs without adding any: the
/// strongest becomes the stance muscle (strength 1.25 times, a rise over 0.55
/// to 0.8 of the cycle), the others become swing muscles (0.7 times, a rise
/// over at most 0.3 of the cycle), and the swing muscles move together so the
/// first starts when the stance rise ends. The leg pushes long and steadily
/// and recovers quickly (Alexander's duty factor; Full and Koditschek 1999).
fn stance_swing_roles(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs: Vec<(BoneIds, usize)> = legs_with_hip(c)
        .into_iter()
        .filter(|(limb, above)| leg_muscles(c, limb, *above).len() >= 2)
        .collect();
    let Some((limb, above)) = pick(&legs, rng) else {
        return false;
    };
    let group = leg_muscles(c, &limb, above);
    let Some(lead) = strongest(c, &group) else {
        return false;
    };
    let before: Bounded<Muscle, MAX_MUSCLES> = group.iter().map(|&i| c.muscles[i]).collect();
    {
        let m = &mut c.muscles[lead];
        m.duty = m.duty.clamp(0.55, 0.8);
        m.stiffness = (m.stiffness * 1.25).clamp(1.0, 120.0);
    }
    let stance = c.muscles[lead];
    let others: MuscleIds = group.iter().copied().filter(|&i| i != lead).collect();
    let target = (stance.phase + stance.duty).rem_euclid(1.0);
    let shift = turn(c.muscles[others[0]].phase, target);
    for &i in others.iter() {
        let m = &mut c.muscles[i];
        m.duty = m.duty.min(0.3);
        m.stiffness = (m.stiffness * 0.7).clamp(1.0, 120.0);
        shift_timing(m, shift);
    }
    group
        .iter()
        .zip(&before)
        .any(|(&i, old)| c.muscles[i] != *old)
}

/// The signed lever arm of a muscle about the joint node it works at: the
/// arm of the bone end that starts at that node, against the line of pull.
fn lever(c: &Creature, m: &Muscle, node: u32) -> f32 {
    let n = c.nodes[node as usize];
    let (own, own_anchor, other, other_anchor) = if c.bones[m.bone_a as usize].a == node {
        (m.bone_a, m.anchor_a, m.bone_b, m.anchor_b)
    } else {
        (m.bone_b, m.anchor_b, m.bone_a, m.anchor_a)
    };
    let p = bone_point(c.bones[own as usize], &c.nodes, own_anchor);
    let q = bone_point(c.bones[other as usize], &c.nodes, other_anchor);
    let length = (q[0] - p[0]).hypot(q[1] - p[1]).max(0.02);
    ((p[0] - n.x) * (q[1] - p[1]) - (p[1] - n.y) * (q[0] - p[0])) / length
}

/// Moves the attachments of a leg's muscle with a poor lever arm to the pair
/// of points along its two bones (0.2 to 0.9 of the way) that gives the
/// longest arm about the joint, keeping the direction of its torque and the
/// shape of its stroke, when that is at least 1.25 times the arm it had. A
/// muscle that pulls along the bone it moves only strains the joint; one at a
/// good arm turns it (Alexander's work on the moment arms of limb muscles).
fn improve_lever_arm(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let mut options: Vec<(usize, f32, f32)> = Vec::new();
    for (limb, above) in legs_with_hip(c) {
        for i in leg_muscles(c, &limb, above) {
            let m = c.muscles[i];
            let Some(node) = shared_node(c, m.bone_a as usize, m.bone_b as usize) else {
                continue;
            };
            let old = lever(c, &m, node);
            if old.abs() < 1.0e-4 {
                continue;
            }
            let mut best = (old.abs() * 1.25, None);
            for ia in 2..=9 {
                for ib in 2..=9 {
                    let t = Muscle {
                        anchor_a: ia as f32 * 0.1,
                        anchor_b: ib as f32 * 0.1,
                        ..m
                    };
                    let arm = lever(c, &t, node);
                    if arm * old > 0.0 && arm.abs() > best.0 && span(c, &t) >= 0.05 {
                        best = (arm.abs(), Some((t.anchor_a, t.anchor_b)));
                    }
                }
            }
            if let Some((a, b)) = best.1 {
                options.push((i, a, b));
            }
        }
    }
    let Some((i, a, b)) = pick(&options, rng) else {
        return false;
    };
    let old = c.muscles[i];
    let mut m = Muscle {
        anchor_a: a,
        anchor_b: b,
        ..old
    };
    fit_stroke(c, &mut m, Some(&old));
    c.muscles[i] = m;
    true
}

/// Copies a muscle of one leg to the same joint of every other leg that
/// lacks one there: the same pair of levels (the bone above the hip, the
/// thigh, the shank), the same anchors, and the timing the source has against
/// its own leg's strongest muscle, added to each leg's own timing (to the
/// source's phase plus half a cycle for a leg with no muscles). The strokes
/// are refitted to each leg. Regular bodies with repeated parts move further
/// (Cheney et al. 2013, Lipson and Pollack 2000).
fn muscle_to_all_legs(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = legs_with_hip(c);
    // The bones of a leg by level: the bone above, then the leg's bones.
    let levels = |limb: &BoneIds, above: usize| {
        let mut bones = BoneIds::from_slice(&[above]);
        bones.extend(limb.iter().copied());
        bones
    };
    let Some((source, i, la, lb)) =
        pick_each(rng, |push: &mut dyn FnMut((usize, usize, usize, usize))| {
            for (s, (limb, above)) in legs.iter().enumerate() {
                let bones = levels(limb, *above);
                let at = |b: u32| bones.iter().position(|&x| x == b as usize);
                for &i in leg_muscles(c, limb, *above).iter() {
                    let m = &c.muscles[i];
                    if let (Some(la), Some(lb)) = (at(m.bone_a), at(m.bone_b)) {
                        push((s, i, la, lb));
                    }
                }
            }
        })
    else {
        return false;
    };
    let m = c.muscles[i];
    let reference = |leg: usize| -> Option<f32> {
        let (limb, above) = &legs[leg];
        strongest(c, &leg_muscles(c, limb, *above)).map(|k| c.muscles[k].phase)
    };
    let Some(source_phase) = reference(source) else {
        return false;
    };
    let mut copies: Bounded<Muscle, MAX_MUSCLES> = Bounded::new();
    for (t, (limb, above)) in legs.iter().enumerate() {
        if t == source || c.muscles.len() + copies.len() >= cfg.max_muscles {
            continue;
        }
        let bones = levels(limb, *above);
        if bones.len() <= la.max(lb) {
            continue;
        }
        let (x, z) = (bones[la], bones[lb]);
        if joined(c, x, z) {
            continue;
        }
        let own = reference(t).unwrap_or(source_phase + 0.5);
        let mut copy = m;
        copy.bone_a = x as u32;
        copy.bone_b = z as u32;
        shift_timing(&mut copy, turn(m.phase, own + turn(source_phase, m.phase)));
        fit_stroke(c, &mut copy, Some(&m));
        copies.push(copy);
    }
    if copies.is_empty() {
        return false;
    }
    c.muscles.extend(copies);
    true
}

/// Adds a muscle that opposes a leg's flexor and contracts with it, so the
/// joint stiffens during the flexor's rise instead of folding under the
/// body's weight: it has the flexor's phase and duty, half its strength, and
/// almost no stroke (0.93 to 1.06 of its span), so it holds more than it
/// moves. Animals stiffen a limb in stance by co-contracting antagonists
/// (the stiff spring-leg of Full and Koditschek 1999).
fn stance_cocontraction(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let Some((closer, r, anchor)) = find_opener(c, rng) else {
        return false;
    };
    let mut m = opener_muscle(c, &closer, r, anchor, rng);
    let s = span(c, &m).max(0.05);
    m.short = 0.93 * s;
    m.long = 1.06 * s;
    m.phase = closer.phase;
    m.reset = closer.reset;
    m.duty = closer.duty;
    m.stiffness = (closer.stiffness * 0.5).clamp(1.0, 120.0);
    c.muscles.push(m);
    true
}

/// Puts an elastic tendon (0.4 to 0.8) on the muscles at the foot of a leg of
/// two bones or more, and cuts their longest length by 7% so ground load
/// stretches the tendon past it. The tendon stores the energy of a landing and
/// gives it back at push-off, which costs the muscle nothing (Alexander 1988,
/// the spring in the leg tendons of running animals).
fn elastic_shank_tendon(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs: Vec<(BoneIds, usize)> = legs_with_hip(c)
        .into_iter()
        .filter(|(limb, _)| limb.len() >= 2)
        .collect();
    let Some((limb, _)) = pick(&legs, rng) else {
        return false;
    };
    let foot = limb[limb.len() - 1];
    let tendon = rng.range(0.4, 0.8);
    let mut changed = false;
    for i in muscles_on(c, &[foot], false) {
        let m = &mut c.muscles[i];
        if !active(m) || m.tendon >= tendon {
            continue;
        }
        m.tendon = tendon;
        m.long = (m.long * 0.93).max(m.short + 1.0e-3);
        changed = true;
    }
    changed
}

/// A pair of active muscles of a leg that attach to one bone and pull its
/// joint in opposite directions, the stronger first, that `wanted` accepts.
fn antagonist_pair(
    c: &Creature,
    rng: &mut Rng,
    wanted: impl Fn(&Muscle, &Muscle) -> bool,
) -> Option<(usize, usize)> {
    let legs = legs_with_hip(c);
    pick_each(rng, |push: &mut dyn FnMut((usize, usize))| {
        for (limb, above) in &legs {
            let on = leg_muscles(c, limb, *above);
            for &k in limb.iter() {
                for (x, &i) in on.iter().enumerate() {
                    for &j in on[x + 1..].iter() {
                        let (Some(p), Some(q)) =
                            (facing(&c.muscles[i], k), facing(&c.muscles[j], k))
                        else {
                            continue;
                        };
                        let (tp, tq) = (torque(c, &p), torque(c, &q));
                        if tp * tq < 0.0 && tp.abs() >= MIN_TORQUE && tq.abs() >= MIN_TORQUE {
                            let (hi, lo) = if drive(&c.muscles[i]) >= drive(&c.muscles[j]) {
                                (i, j)
                            } else {
                                (j, i)
                            };
                            if wanted(&c.muscles[hi], &c.muscles[lo]) {
                                push((hi, lo));
                            }
                        }
                    }
                }
            }
        }
    })
}

/// Puts the two muscles of an antagonist pair on a leg into one reciprocal
/// rhythm: the weaker takes the stronger's period, starts when the stronger's
/// rise ends and rises over the rest of the cycle. Pairs that already do are
/// skipped. A joint driven by two muscles that fight for part of the cycle
/// wastes energy, and a pair that alternates is what a central pattern
/// generator drives (Ijspeert 2008, half-centre oscillators).
fn lock_antagonist_pairs(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let off = |hi: &Muscle, lo: &Muscle| {
        turn(lo.phase, hi.phase + hi.duty).abs() > 0.05
            || (lo.duty - (1.0 - hi.duty)).abs() > 0.05
            || lo.period != hi.period
    };
    let Some((hi, lo)) = antagonist_pair(c, rng, off) else {
        return false;
    };
    let lead = c.muscles[hi];
    let m = &mut c.muscles[lo];
    let shift = turn(m.phase, (lead.phase + lead.duty).rem_euclid(1.0));
    shift_timing(m, shift);
    m.duty = (1.0 - lead.duty).clamp(0.15, 0.85);
    m.period = lead.period;
    true
}

/// Adds a push-off muscle across the last joint of a leg of two bones or
/// more when nothing drives that joint: from the foot bone to the bone
/// before it, a quarter-cycle rise starting at 0.8 of the hip muscle's rise,
/// so it fires at the end of stance. It is 1.1 times as strong as the hip
/// muscle and has a tendon of 0.6, like the plantar flexors that deliver
/// most of the push in a stride (Alexander 1988).
fn push_off_muscle(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let legs: Vec<(usize, usize, usize)> = legs_with_hip(c)
        .into_iter()
        .filter(|(limb, _)| limb.len() >= 2)
        .filter_map(|(limb, above)| {
            let (foot, before) = (limb[limb.len() - 1], limb[limb.len() - 2]);
            let driver = hinge_driver(c, &limb, above)?;
            (!joined(c, foot, before)).then_some((foot, before, driver))
        })
        .collect();
    let Some((foot, before, driver)) = pick(&legs, rng) else {
        return false;
    };
    let template = c.muscles[driver];
    let anchors = (rng.range(0.3, 0.9), rng.range(0.3, 0.8));
    let mut m = new_muscle(c, foot, before, anchors, Some(&template), rng);
    fit_stroke(c, &mut m, None);
    m.sensor = NO_SENSOR;
    m.duty = 0.25;
    m.phase = (template.phase + 0.8 * template.duty).rem_euclid(1.0);
    m.reset = m.phase;
    m.stiffness = (template.stiffness * 1.1).clamp(1.0, 120.0);
    m.tendon = m.tendon.max(0.6);
    c.muscles.push(m);
    true
}

/// Turns the least useful muscle of the body (the least drive of three
/// picked off the motor ring) into the extensor of a leg joint that has a
/// flexor, in the flexor's reciprocal rhythm. The body gains an antagonist
/// and loses a muscle that did little, so it grows no heavier in muscles,
/// which is the move for bodies at their muscle limit.
fn repurpose_idle_muscle(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let free: MuscleIds = (0..c.muscles.len())
        .filter(|&i| !ring(c, &c.muscles[i]))
        .collect();
    if free.len() < 2 {
        return false;
    }
    let idlest = (0..3)
        .map(|_| free[rng.index(free.len())])
        .min_by(|&x, &y| drive(&c.muscles[x]).total_cmp(&drive(&c.muscles[y])))
        .expect("three picks");
    let Some((closer, r, anchor)) = find_opener(c, rng) else {
        return false;
    };
    // Never replace the closer itself or a stronger muscle.
    let idle = c.muscles[idlest];
    let is_closer = facing(&idle, closer.bone_a as usize)
        .is_some_and(|m| m.bone_b == closer.bone_b && m.anchor_a == closer.anchor_a);
    if is_closer || drive(&idle) >= drive(&closer) {
        return false;
    }
    let mut m = opener_muscle(c, &closer, r, anchor, rng);
    make_reciprocal(&mut m, &closer);
    c.muscles[idlest] = m;
    true
}

/// Makes an antagonist pair of a leg into a catapult: the stronger muscle
/// fires quickly (a rise over 0.2 of the cycle) with an elastic tendon of at
/// least 0.6, and the weaker loads it slowly (a rise over 0.8, 1.15 times as
/// strong), ending just as the stronger starts. The slow muscle stretches the
/// tendon and the quick release returns the energy, the way a leg stores
/// energy for a jump (Alexander's catapult mechanisms; Bobbert 2001).
fn catapult_release(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let off = |hi: &Muscle, lo: &Muscle| {
        hi.tendon < 0.6 || (hi.duty - 0.2).abs() > 0.05 || (lo.duty - 0.8).abs() > 0.05
    };
    let Some((hi, lo)) = antagonist_pair(c, rng, off) else {
        return false;
    };
    let spring = &mut c.muscles[hi];
    spring.duty = 0.2;
    spring.tendon = spring.tendon.max(0.6);
    let (phase, period) = (spring.phase, spring.period);
    let loader = &mut c.muscles[lo];
    loader.duty = 0.8;
    loader.period = period;
    let shift = turn(loader.phase, (phase - 0.8).rem_euclid(1.0));
    shift_timing(loader, shift);
    loader.stiffness = (loader.stiffness * 1.15).clamp(1.0, 120.0);
    true
}

/// Adds a second muscle from a leg's first bone to another bone at the hip
/// (a sibling at the same node, or the bone above the one the leg hangs
/// from), pulling the same way as the hip muscle, and shares the work: each
/// of the two has 0.65 of the old strength and the timing of the old one. The
/// force of a leg spreads over two trunk bones the way the gluteal muscles
/// fan out over the pelvis, so one trunk bone does not carry the whole
/// reaction.
fn second_hip_anchor(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let parents = parent_bones(c);
    let children = child_bones(c);
    let mut options: Vec<(usize, usize, usize)> = Vec::new();
    for (limb, above) in legs_with_hip(c) {
        let Some(driver) = hinge_driver(c, &limb, above) else {
            continue;
        };
        let root = limb[0];
        let node = c.bones[root].a as usize;
        let mut others: BoneIds = children[node].iter().copied().collect();
        others.extend(parents[c.bones[above].a as usize]);
        for &q in others.iter() {
            if q != above && q != root && !is_neck(c, q) && !joined(c, root, q) {
                options.push((driver, root, q));
            }
        }
    }
    let Some((driver, root, q)) = pick(&options, rng) else {
        return false;
    };
    let Some(old) = facing(&c.muscles[driver], root) else {
        return false;
    };
    let node = c.bones[root].a;
    let near = if c.bones[q].a == node {
        rng.range(0.1, 0.4)
    } else if c.bones[q].b == node {
        rng.range(0.6, 0.9)
    } else {
        return false;
    };
    let mut m = new_muscle(c, root, q, (old.anchor_a, near), Some(&old), rng);
    if m.sensor >= 2 {
        // The sensor sat on the bone above, which this muscle no longer touches.
        m.sensor = NO_SENSOR;
    }
    let (new_pull, old_pull) = (torque(c, &m), torque(c, &old));
    if new_pull * old_pull <= 0.0 || new_pull.abs() < MIN_TORQUE {
        return false;
    }
    let share = (old.stiffness * 0.65).clamp(1.0, 120.0);
    m.stiffness = share;
    c.muscles[driver].stiffness = share;
    c.muscles.push(m);
    true
}
