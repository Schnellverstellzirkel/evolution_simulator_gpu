//! Gait operators for body plan variety: organ ballast, rhythm waves along a
//! limb, tendons, joint ranges, allometry, homeosis, gene duplication and
//! broken symmetry.
//!
//! The operators of this file share one pick slot and are compound: each is a
//! whole, coherent change to the body, and its child gets no parameter noise.
//!
//! The sources are Sims (1994, evolved limb mass and joint ranges), Lipson
//! and Pollack (2000, timing along a limb and broken symmetry), Ijspeert
//! (2008, central pattern generator harmonics and coupled phases), Alexander
//! (1988 and 2003, elastic legs, passive joints and hopping), Thompson (1917,
//! allometry and taper), Bateson (1894, homeosis) and Ohno (1970, gene
//! duplication followed by divergence).
use super::limbs::pick;
use super::rhythm::{leaf_limbs, matching_limbs};
use super::{BoneIds, Context, MuscleIds, Operator, branch, muscles_on, room};
use crate::config::Config;
use crate::evolution::{
    Creature, JOINT_LIMIT, MAX_ORGAN_MASS, MIN_ORGAN_MASS, Rng, max_bone_length, max_stroke,
    min_muscle_period,
};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("ballast_slide", ballast_slide),
    ("ballast_split", ballast_split),
    ("ballast_flail_tip", ballast_flail_tip),
    ("ballast_trade", ballast_trade),
    ("duty_sweep_along_limb", duty_sweep_along_limb),
    ("harmonic_twin", harmonic_twin),
    ("pulse_and_coast", pulse_and_coast),
    ("phase_follow_neighbor", phase_follow_neighbor),
    ("spring_leg", spring_leg),
    ("passive_joint", passive_joint),
    ("tendon_gradient", tendon_gradient),
    ("widen_range_pair", widen_range_pair),
    ("asymmetric_range_for_hop", asymmetric_range_for_hop),
    ("allometric_limb", allometric_limb),
    ("taper_by_depth", taper_by_depth),
    ("homeotic_swap", homeotic_swap),
    ("diverged_twin", diverged_twin),
    ("break_one_mirror", break_one_mirror),
];

// Helpers.

/// The bones that carry an organ.
fn organ_bones(c: &Creature) -> BoneIds {
    (0..c.bones.len())
        .filter(|&b| c.bones[b].organ_mass > 0.0)
        .collect()
}

/// A random leg with at least `bones` bones and, with `driven`, a muscle on it.
fn pick_leg(c: &Creature, rng: &mut Rng, bones: usize, driven: bool) -> Option<BoneIds> {
    let legs = leaf_limbs(c);
    let fit: BoneIds = (0..legs.len())
        .filter(|&i| {
            legs[i].len() >= bones && (!driven || !muscles_on(c, &legs[i], false).is_empty())
        })
        .collect();
    pick(&fit, rng).map(|i| legs[i])
}

/// How deep along `limb` a muscle sits: the position of its deepest bone in
/// the limb, from 0 at the root.
fn depth_in(c: &Creature, limb: &[usize], m: usize) -> usize {
    let at = |b: u32| limb.iter().position(|&x| x == b as usize).unwrap_or(0);
    at(c.muscles[m].bone_a).max(at(c.muscles[m].bone_b))
}

/// Shifts a muscle's phase, wrapping to [0, 1).
fn shift_phase(c: &mut Creature, m: usize, by: f32) {
    c.muscles[m].phase = (c.muscles[m].phase + by).rem_euclid(1.0);
}

/// A bone length kept within the body's limits.
fn bone_length(length: f32) -> f32 {
    length.clamp(0.05, max_bone_length())
}

// Ballast: where the organ mass sits.

/// Moves an organ along its bone toward the tip or the root by 10 to 40% of
/// the bone.
fn ballast_slide(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(b) = pick(&organ_bones(c), rng) else {
        return false;
    };
    let step = rng.range(0.1, 0.4) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    let at = (c.bones[b].organ_at + step).clamp(0.0, 1.0);
    if at == c.bones[b].organ_at {
        return false;
    }
    c.bones[b].organ_at = at;
    true
}

/// Splits an organ into two on neighbouring bones with the same total mass.
fn ballast_split(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let heavy: BoneIds = organ_bones(c)
        .into_iter()
        .filter(|&b| c.bones[b].organ_mass >= 2.0 * MIN_ORGAN_MASS)
        .collect();
    let Some(b) = pick(&heavy, rng) else {
        return false;
    };
    let (a, z) = (c.bones[b].a, c.bones[b].b);
    let free: BoneIds = (0..c.bones.len())
        .filter(|&o| o != b && c.bones[o].organ_mass == 0.0)
        .filter(|&o| {
            let (p, q) = (c.bones[o].a, c.bones[o].b);
            p == a || p == z || q == a || q == z
        })
        .collect();
    let Some(o) = pick(&free, rng) else {
        return false;
    };
    let half = 0.5 * c.bones[b].organ_mass;
    c.bones[b].organ_mass = half;
    c.bones[o].organ_mass = half;
    c.bones[o].organ_at = 0.5;
    true
}

/// Gathers a leg's organ mass at its tip, so the leg swings like a flail.
fn ballast_flail_tip(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick_leg(c, rng, 2, false) else {
        return false;
    };
    let total: f32 = leg.iter().map(|&b| c.bones[b].organ_mass).sum();
    if total <= 0.0 {
        return false;
    }
    for &b in &leg {
        c.bones[b].organ_mass = 0.0;
    }
    let tip = leg[leg.len() - 1];
    c.bones[tip].organ_mass = total.clamp(MIN_ORGAN_MASS, MAX_ORGAN_MASS);
    c.bones[tip].organ_at = 1.0;
    true
}

/// Two organs swap their masses and places along their bones.
fn ballast_trade(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let organs = organ_bones(c);
    if organs.len() < 2 {
        return false;
    }
    let x = organs[rng.index(organs.len())];
    let y = organs[rng.index(organs.len())];
    if x == y || c.bones[x].organ_mass == c.bones[y].organ_mass {
        return false;
    }
    let (mx, ax) = (c.bones[x].organ_mass, c.bones[x].organ_at);
    c.bones[x].organ_mass = c.bones[y].organ_mass;
    c.bones[x].organ_at = c.bones[y].organ_at;
    c.bones[y].organ_mass = mx;
    c.bones[y].organ_at = ax;
    true
}

// Waves: rhythm along a limb.

/// Muscle duty changes in steps from a leg's root to its tip, so the tip works
/// a shorter (or longer) share of the cycle than the root.
fn duty_sweep_along_limb(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick_leg(c, rng, 2, true) else {
        return false;
    };
    let on = muscles_on(c, &leg, false);
    let sweep = rng.range(0.2, 0.5) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    let deepest = (leg.len() - 1).max(1) as f32;
    for &m in &on {
        let k = depth_in(c, &leg, m) as f32 / deepest;
        c.muscles[m].duty = (c.muscles[m].duty * (1.0 - sweep * k)).clamp(0.05, 0.95);
    }
    true
}

/// A second muscle on the same two bones at half or double the period, with
/// a small stroke, adds a second beat to the gait.
fn harmonic_twin(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.is_empty() || !room(c, cfg, 0, 1) {
        return false;
    }
    let mut twin = c.muscles[rng.index(c.muscles.len())];
    let period = twin.period * if rng.unit() < 0.5 { 0.5 } else { 2.0 };
    if !(min_muscle_period()..=10.0).contains(&period) {
        return false;
    }
    twin.period = period;
    twin.long = twin.short + 0.3 * (twin.long - twin.short);
    twin.stiffness *= 0.5;
    twin.phase = rng.unit();
    c.muscles.push(twin);
    true
}

/// One muscle kicks once per cycle: a short duty and a high stiffness, and
/// the limb coasts the rest of the cycle.
fn pulse_and_coast(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.is_empty() {
        return false;
    }
    let m = rng.index(c.muscles.len());
    c.muscles[m].duty = rng.range(0.08, 0.2);
    c.muscles[m].stiffness *= rng.range(1.3, 2.0);
    true
}

/// A leg's muscles follow a neighbouring leg's lead phase with a fixed lag.
fn phase_follow_neighbor(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leaf_limbs(c);
    let driven: BoneIds = (0..legs.len())
        .filter(|&i| !muscles_on(c, &legs[i], false).is_empty())
        .collect();
    if driven.len() < 2 {
        return false;
    }
    let x = driven[rng.index(driven.len())];
    let y = driven[rng.index(driven.len())];
    if x == y {
        return false;
    }
    let lead_x = muscles_on(c, &legs[x], false)[0];
    let on_y = muscles_on(c, &legs[y], false);
    if on_y.contains(&lead_x) {
        return false;
    }
    let lag = rng.range(0.1, 0.4);
    let shift = c.muscles[lead_x].phase + lag - c.muscles[on_y[0]].phase;
    for &m in &on_y {
        shift_phase(c, m, shift);
    }
    true
}

// Tendons.

/// Every muscle of a leg gets a tendon, stiffer toward the foot, so the leg
/// stores and returns energy like a spring.
fn spring_leg(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick_leg(c, rng, 2, true) else {
        return false;
    };
    let on = muscles_on(c, &leg, false);
    let (root, tip) = (rng.range(0.1, 0.3), rng.range(0.5, 0.9));
    let deepest = (leg.len() - 1).max(1) as f32;
    for &m in &on {
        let k = depth_in(c, &leg, m) as f32 / deepest;
        c.muscles[m].tendon = root + (tip - root) * k;
    }
    true
}

/// One muscle gets a long, weak stroke and a stiff tendon, which makes its
/// joint a passive hinge that the tendon carries.
fn passive_joint(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.is_empty() {
        return false;
    }
    let pick = rng.index(c.muscles.len());
    let m = &mut c.muscles[pick];
    m.long = (m.short + 1.3 * (m.long - m.short)).min(max_stroke().max(m.short));
    m.stiffness *= 0.3;
    m.tendon = rng.range(0.6, 1.0);
    true
}

/// The tendons of a leg's muscles follow a gradient from root to tip.
fn tendon_gradient(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick_leg(c, rng, 2, true) else {
        return false;
    };
    let on = muscles_on(c, &leg, false);
    let (from, to) = (rng.unit(), rng.unit());
    let deepest = (leg.len() - 1).max(1) as f32;
    for &m in &on {
        let k = depth_in(c, &leg, m) as f32 / deepest;
        c.muscles[m].tendon = from + (to - from) * k;
    }
    true
}

// Joint ranges.

/// Two joints of a leg widen their ranges toward the joint limit by 20 to
/// 60%, so the leg can sweep farther.
fn widen_range_pair(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick_leg(c, rng, 2, false) else {
        return false;
    };
    let first = rng.index(leg.len() - 1);
    let mut changed = false;
    for &b in &leg[first..first + 2] {
        let f = rng.range(0.2, 0.6);
        let bone = &mut c.bones[b];
        let (low, high) = (
            bone.min_angle - (JOINT_LIMIT + bone.min_angle) * f,
            bone.max_angle + (JOINT_LIMIT - bone.max_angle) * f,
        );
        changed |= low != bone.min_angle || high != bone.max_angle;
        bone.min_angle = low.max(-JOINT_LIMIT);
        bone.max_angle = high.min(JOINT_LIMIT);
    }
    changed
}

/// A joint turns one way only: one side of its range closes and the other
/// widens, which makes a one-way hinge for hopping.
fn asymmetric_range_for_hop(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.bones.len() < 2 {
        return false;
    }
    let b = 1 + rng.index(c.bones.len() - 1);
    let bone = &mut c.bones[b];
    if rng.unit() < 0.5 {
        bone.min_angle = -0.05;
        bone.max_angle += 0.5 * (JOINT_LIMIT - bone.max_angle);
    } else {
        bone.max_angle = 0.05;
        bone.min_angle -= 0.5 * (JOINT_LIMIT + bone.min_angle);
    }
    true
}

// Allometry.

/// A leg's bone lengths scale with the cube root of the mass each bone
/// carries (the bones below it), so roots grow and tips shrink.
fn allometric_limb(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick_leg(c, rng, 2, false) else {
        return false;
    };
    let n = leg.len() as f32;
    let mean_length: f32 = leg.iter().map(|&b| c.bones[b].rest_length).sum::<f32>() / n;
    // Bone k carries the n - k bones from it to the tip.
    let mean_carried = (n + 1.0) / 2.0;
    for (k, &b) in leg.iter().enumerate() {
        let carried = n - k as f32;
        c.bones[b].rest_length = bone_length(mean_length * (carried / mean_carried).cbrt());
    }
    true
}

/// Bones of a leg get shorter (or longer) step by step from root to tip.
fn taper_by_depth(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick_leg(c, rng, 2, false) else {
        return false;
    };
    let ratio = if rng.unit() < 0.5 {
        rng.range(0.75, 0.92)
    } else {
        rng.range(1.08, 1.33)
    };
    let mid = (leg.len() - 1) as f32 / 2.0;
    for (k, &b) in leg.iter().enumerate() {
        let length = c.bones[b].rest_length * ratio.powf(k as f32 - mid);
        c.bones[b].rest_length = bone_length(length);
    }
    true
}

// Homeosis, duplication and symmetry.

/// A leg and the trunk swap their muscle programs (period, phase, duty,
/// stiffness, tendon), so a limb takes another part's role in the same body.
fn homeotic_swap(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leaf_limbs(c);
    let Some(leg) = pick_leg(c, rng, 1, true) else {
        return false;
    };
    let on_leg = muscles_on(c, &leg, false);
    let trunk: MuscleIds = (0..c.muscles.len())
        .filter(|&m| legs.iter().all(|l| !muscles_on(c, l, false).contains(&m)))
        .collect();
    let count = on_leg.len().min(trunk.len());
    if count == 0 {
        return false;
    }
    for k in 0..count {
        let (x, y) = (on_leg[k], trunk[k]);
        let (a, b) = (c.muscles[x], c.muscles[y]);
        for (to, from) in [(x, b), (y, a)] {
            let m = &mut c.muscles[to];
            m.period = from.period;
            m.phase = from.phase;
            m.duty = from.duty;
            m.stiffness = from.stiffness;
            m.tendon = from.tendon;
        }
    }
    true
}

/// Copies a leg onto its own hip and lets the copy diverge: its timing, its
/// bone lengths and its joint ranges change, so the twin can take a new role.
fn diverged_twin(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick_leg(c, rng, 1, true) else {
        return false;
    };
    let quota = muscles_on(c, &leg, false).len();
    if !room(c, cfg, leg.len(), quota) {
        return false;
    }
    let hip = c.bones[leg[0]].a as usize;
    let dx = rng.range(-0.15, 0.15);
    let lag = rng.range(0.1, 0.4);
    let Some(root) = super::copy_branch_limited(
        c,
        cfg,
        leg[0],
        hip,
        |p| super::limbs::clamped(p[0] + dx, p[1]),
        false,
        lag,
        quota,
    ) else {
        return false;
    };
    let twin = branch(c, root);
    for &b in &twin {
        let bone = &mut c.bones[b];
        bone.rest_length = bone_length(bone.rest_length * rng.range(0.85, 1.15));
        bone.min_angle = (bone.min_angle * rng.range(0.7, 1.3)).clamp(-JOINT_LIMIT, 0.0);
        bone.max_angle = (bone.max_angle * rng.range(0.7, 1.3)).clamp(0.0, JOINT_LIMIT);
    }
    for m in muscles_on(c, &twin, false) {
        c.muscles[m].duty = (c.muscles[m].duty + rng.range(-0.1, 0.1)).clamp(0.05, 0.95);
    }
    true
}

/// In a pair of matching limbs, one limb's phase moves by 0.05 to 0.15 of a
/// cycle and its bones change length by 5 to 12%, which breaks a symmetric
/// gait that many elites share.
fn break_one_mirror(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let pairs = matching_limbs(c);
    if pairs.is_empty() {
        return false;
    }
    let (x, y) = pairs.get(rng.index(pairs.len()));
    let limb = if rng.unit() < 0.5 { *x } else { *y };
    let sign = if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    let shift = sign * rng.range(0.05, 0.15);
    for m in muscles_on(c, &limb, false) {
        shift_phase(c, m, shift);
    }
    let scale = 1.0 + sign * rng.range(0.05, 0.12);
    for &b in &limb {
        c.bones[b].rest_length = bone_length(c.bones[b].rest_length * scale);
    }
    true
}
