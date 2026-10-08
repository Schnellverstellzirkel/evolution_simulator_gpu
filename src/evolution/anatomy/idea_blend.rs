//! Idea operators that take one group of genes from a second elite, the
//! donor, and leave the rest of the body alone: muscle timing, node surfaces,
//! joint ranges, tendons, organs, stroke ratios, duty cycles, leg lengths or
//! tempo. Each is a crossover at the level of a gene group (`graft_donor_limb`
//! crosses whole limbs), and parts are matched by index up to the shorter body.
//!
//! None does anything without a donor, every one is a whole change that gets
//! no parameter noise, and the operators of this file share one pick slot.
//! The sources are Vassiliades and Mouret (2018, a step along the line
//! between two elites finds good variants faster than isotropic noise,
//! Iso+LineDD), Hutchinson et al. (2026, discrete crossover of genes between
//! elites), Lessin, Fussell and Miikkulainen (2013, exchange of whole modules)
//! and Cully and Demiris (2017, behavioural diversity from recombination).
use super::ideas::{coin, phase_gap, set, wrap};
use super::rhythm::leaf_limbs;
use super::{Context, Operator};
use crate::config::Config;
use crate::evolution::{
    Creature, JOINT_LIMIT, MAX_ORGAN_MASS, MIN_ORGAN_MASS, Rng, max_bone_length, min_muscle_period,
};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("isoline_timing_step", isoline_timing_step),
    ("donor_surfaces", donor_surfaces),
    ("donor_ranges", donor_ranges),
    ("donor_tendons", donor_tendons),
    ("donor_organs", donor_organs),
    ("donor_stroke_ratio", donor_stroke_ratio),
    ("donor_duty_profile", donor_duty_profile),
    ("donor_leg_lengths", donor_leg_lengths),
    ("donor_tempo", donor_tempo),
];

/// A step along the line from this body to the donor's, as a share of the
/// distance. Four times in five it is 0.25 to 0.9, between the two bodies.
/// Otherwise it is a step of 0.1 to 0.3 the other way, past this body and away
/// from the donor.
fn step(rng: &mut Rng) -> f32 {
    if rng.unit() < 0.8 {
        rng.range(0.25, 0.9)
    } else {
        -rng.range(0.1, 0.3)
    }
}

/// The rhythm genes of each muscle (phase, period, duty, stiffness) move along
/// the line to the donor's muscle at the same index, all by one `step`
/// (Vassiliades and Mouret 2018). The phase takes the short way round the
/// cycle.
fn isoline_timing_step(c: &mut Creature, _cfg: &Config, rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    let n = c.muscles.len().min(d.muscles.len());
    if n == 0 {
        return false;
    }
    let t = step(rng);
    let mut changed = false;
    for i in 0..n {
        let (m, o) = (c.muscles[i], d.muscles[i]);
        let phase = wrap(m.phase + t * phase_gap(m.phase, o.phase));
        let period = (m.period + t * (o.period - m.period)).clamp(min_muscle_period(), 10.0);
        let duty = (m.duty + t * (o.duty - m.duty)).clamp(0.05, 0.95);
        let stiffness = (m.stiffness + t * (o.stiffness - m.stiffness)).clamp(1.0, 120.0);
        set(&mut c.muscles[i].phase, phase, &mut changed);
        set(&mut c.muscles[i].period, period, &mut changed);
        set(&mut c.muscles[i].duty, duty, &mut changed);
        set(&mut c.muscles[i].stiffness, stiffness, &mut changed);
    }
    changed
}

/// Node sizes and grips move halfway to the donor's, except the head's.
fn donor_surfaces(c: &mut Creature, cfg: &Config, _rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    let n = c.nodes.len().min(d.nodes.len());
    let mut changed = false;
    for i in 1..n {
        let size = 0.5 * (c.nodes[i].diameter + d.nodes[i].diameter);
        let grip = 0.5 * (c.nodes[i].friction + d.nodes[i].friction);
        set(
            &mut c.nodes[i].diameter,
            size.clamp(cfg.min_size, cfg.max_size),
            &mut changed,
        );
        set(
            &mut c.nodes[i].friction,
            grip.clamp(cfg.min_friction, cfg.max_friction),
            &mut changed,
        );
    }
    changed
}

/// Joint ranges move halfway to the donor's, bone by bone.
fn donor_ranges(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    let n = c.bones.len().min(d.bones.len());
    let mut changed = false;
    // The neck (bone 0 at the head) keeps its range.
    for i in 1..n {
        let low = (0.5 * (c.bones[i].min_angle + d.bones[i].min_angle)).clamp(-JOINT_LIMIT, 0.0);
        let high = (0.5 * (c.bones[i].max_angle + d.bones[i].max_angle)).clamp(0.0, JOINT_LIMIT);
        set(&mut c.bones[i].min_angle, low, &mut changed);
        set(&mut c.bones[i].max_angle, high, &mut changed);
    }
    changed
}

/// Tendons are copied from the donor's muscles at the same index.
fn donor_tendons(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    let n = c.muscles.len().min(d.muscles.len());
    let mut changed = false;
    for i in 0..n {
        set(&mut c.muscles[i].tendon, d.muscles[i].tendon, &mut changed);
    }
    changed
}

/// Organs take the donor's masses and places, bone by bone. A bone where the
/// donor has no organ loses its own.
fn donor_organs(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    let n = c.bones.len().min(d.bones.len());
    let mut changed = false;
    for i in 0..n {
        let mass = d.bones[i].organ_mass;
        let mass = if mass > 0.0 {
            mass.clamp(MIN_ORGAN_MASS, MAX_ORGAN_MASS)
        } else {
            0.0
        };
        set(&mut c.bones[i].organ_mass, mass, &mut changed);
        set(&mut c.bones[i].organ_at, d.bones[i].organ_at, &mut changed);
    }
    changed
}

/// The ratio of each muscle's `short` to its `long` becomes the donor muscle's,
/// kept between 0.1 and 0.95, and the muscle keeps its own `long`.
fn donor_stroke_ratio(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    let n = c.muscles.len().min(d.muscles.len());
    let mut changed = false;
    for i in 0..n {
        let o = d.muscles[i];
        if o.long <= 0.0 {
            continue;
        }
        let ratio = (o.short / o.long).clamp(0.1, 0.95);
        let short = (c.muscles[i].long * ratio).max(0.01);
        set(&mut c.muscles[i].short, short, &mut changed);
    }
    changed
}

/// Duty cycles are copied from the donor's muscles at the same index.
fn donor_duty_profile(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    let n = c.muscles.len().min(d.muscles.len());
    let mut changed = false;
    for i in 0..n {
        set(&mut c.muscles[i].duty, d.muscles[i].duty, &mut changed);
    }
    changed
}

/// Each leg's total length moves toward the length of the leg at the same place
/// in the donor's leg order, by one random share (0.4 to 1.0) of the distance
/// and by at most 30% either way. All bones of a leg scale together. Legs are
/// listed by the index of their foot node.
fn donor_leg_lengths(c: &mut Creature, _cfg: &Config, rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    let (mine, theirs) = (leaf_limbs(c), leaf_limbs(d));
    let n = mine.len().min(theirs.len());
    if n == 0 {
        return false;
    }
    let total = |c: &Creature, l: &[usize]| l.iter().map(|&b| c.bones[b].rest_length).sum::<f32>();
    let share = rng.range(0.4, 1.0);
    let mut changed = false;
    for i in 0..n {
        let (own, want) = (total(c, &mine[i]), total(d, &theirs[i]));
        if own <= 0.0 {
            continue;
        }
        let ratio = (1.0 + share * (want / own - 1.0)).clamp(0.7, 1.3);
        for &b in mine[i].iter() {
            let length = (c.bones[b].rest_length * ratio).clamp(0.05, max_bone_length());
            set(&mut c.bones[b].rest_length, length, &mut changed);
        }
    }
    changed
}

/// The mean period of the body moves to the donor's, all the way or halfway by
/// a coin flip, by a factor of 0.7 to 1.4 at most. Every period is scaled by
/// the same factor, so the ratios among clocks stay.
fn donor_tempo(c: &mut Creature, _cfg: &Config, rng: &mut Rng, cx: &Context) -> bool {
    let Some(d) = cx.donor() else { return false };
    if c.muscles.is_empty() || d.muscles.is_empty() {
        return false;
    }
    let mean =
        |c: &Creature| c.muscles.iter().map(|m| m.period).sum::<f32>() / c.muscles.len() as f32;
    let share = if coin(rng) { 1.0 } else { 0.5 };
    let factor = (1.0 + share * (mean(d) / mean(c) - 1.0)).clamp(0.7, 1.4);
    let mut changed = false;
    for m in &mut c.muscles {
        let period = (m.period * factor).clamp(min_muscle_period(), 10.0);
        set(&mut m.period, period, &mut changed);
    }
    changed
}
