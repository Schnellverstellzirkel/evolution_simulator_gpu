//! Idea operators for clocks and coupling: how muscles synchronise, run in
//! waves, run backward, or drift against each other.
//!
//! Every operator changes only timing (phase, period, duty, touchdown reset).
//! Each is a whole change on its own, so its child gets no parameter noise,
//! and the operators of this file share one pick slot (`GAIT_FILES` in
//! `mod.rs`).
//!
//! The sources are Kuramoto (1975, coupled oscillators pull each other's phase
//! toward the group), Collins and Stewart (1993) and Ijspeert (2008, a gait as
//! coupled oscillators with fixed phase lags), Golubitsky and Stewart (2003,
//! symmetry of the coupling decides which gaits exist) and Weyl (1916, steps of
//! the golden ratio spread points more evenly than any other step).
use super::ideas::{by_drive, circular_mean, coin, phase_gap, set, some_leg, wrap};
use super::limbs::pick;
use super::rhythm::{leaf_limbs, matching_limbs};
use super::{BoneIds, Context, MuscleIds, Operator, bone_point, muscles_on};
use crate::config::Config;
use crate::evolution::{Creature, NO_SENSOR, Rng, min_muscle_period};

/// The golden ratio's fractional part.
const GOLDEN: f32 = 0.618_034;

/// This file's operators, by name. Add each new one here. The pick slot of
/// this file chooses by position in this list, so the order decides what a
/// fixed seed picks.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("kuramoto_pull", kuramoto_pull),
    ("kuramoto_push", kuramoto_push),
    ("golden_phases", golden_phases),
    ("three_phase_split", three_phase_split),
    ("traveling_wave_by_distance", traveling_wave_by_distance),
    ("time_reverse_limb", time_reverse_limb),
    ("time_reverse_body", time_reverse_body),
    ("detune_one", detune_one),
    ("tempo_shift", tempo_shift),
    ("harmonic_ladder", harmonic_ladder),
    ("duty_golden", duty_golden),
    ("duty_complement_pair", duty_complement_pair),
    ("phase_quantize", phase_quantize),
    ("swap_two_phases", swap_two_phases),
    ("reset_advance", reset_advance),
    ("conductor", conductor),
    ("halves_antiphase", halves_antiphase),
    ("phase_from_height", phase_from_height),
];

/// Where a muscle acts: the point midway between its two attachments.
fn muscle_point(c: &Creature, m: usize) -> [f32; 2] {
    let m = c.muscles[m];
    let a = bone_point(c.bones[m.bone_a as usize], &c.nodes, m.anchor_a);
    let b = bone_point(c.bones[m.bone_b as usize], &c.nodes, m.anchor_b);
    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
}

/// Whether two muscles act on a common bone.
fn neighbours(c: &Creature, x: usize, y: usize) -> bool {
    let (p, q) = (c.muscles[x], c.muscles[y]);
    p.bone_a == q.bone_a || p.bone_a == q.bone_b || p.bone_b == q.bone_a || p.bone_b == q.bone_b
}

/// Moves each muscle's phase by `pull` of the way to the circular mean of the
/// muscles it shares a bone with (a negative `pull` moves it away). All
/// phases are read before any is written.
fn couple(c: &mut Creature, pull: f32) -> bool {
    let before: Vec<f32> = c.muscles.iter().map(|m| m.phase).collect();
    let mut changed = false;
    for x in 0..c.muscles.len() {
        let near = (0..c.muscles.len())
            .filter(|&y| y != x && neighbours(c, x, y))
            .map(|y| before[y]);
        let Some(mean) = circular_mean(near) else {
            continue;
        };
        let phase = wrap(before[x] + pull * phase_gap(before[x], mean));
        set(&mut c.muscles[x].phase, phase, &mut changed);
    }
    changed
}

/// Each muscle's phase moves a quarter of the way to the mean phase of the
/// muscles it shares a bone with: the gait locks together (Kuramoto 1975).
fn kuramoto_pull(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    couple(c, 0.25)
}

/// Each muscle's phase moves a quarter of the way away from the mean phase of
/// its neighbours: the gait comes apart into out-of-step parts.
fn kuramoto_push(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    couple(c, -0.25)
}

/// Muscles in order take phases a golden-ratio step apart (Weyl 1916), which
/// spreads any number of muscles evenly round the cycle.
fn golden_phases(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 2 {
        return false;
    }
    let base = c.muscles[0].phase;
    let step = if coin(rng) { GOLDEN } else { 1.0 - GOLDEN };
    let mut changed = false;
    for (k, m) in c.muscles.iter_mut().enumerate() {
        set(&mut m.phase, wrap(base + k as f32 * step), &mut changed);
    }
    changed
}

/// Moves the muscles of each limb together so the first muscle of limb `i`
/// sits at the first limb's phase plus `offset(i)`. A muscle on two limbs
/// moves once.
fn offset_limbs(c: &mut Creature, limbs: &[BoneIds], offset: impl Fn(usize) -> f32) -> bool {
    let Some(lead) = limbs
        .iter()
        .find_map(|l| muscles_on(c, l, false).first().copied())
    else {
        return false;
    };
    let base = c.muscles[lead].phase;
    let mut moved = vec![false; c.muscles.len()];
    let mut changed = false;
    for (i, limb) in limbs.iter().enumerate() {
        let on = muscles_on(c, limb, false);
        let Some(&first) = on.first() else { continue };
        let shift = base + offset(i) - c.muscles[first].phase;
        for &m in &on {
            if !moved[m] {
                moved[m] = true;
                let phase = wrap(c.muscles[m].phase + shift);
                set(&mut c.muscles[m].phase, phase, &mut changed);
            }
        }
    }
    changed
}

/// Three or more legs run in three phases, a third of a cycle apart, in
/// order (a three-beat canter or a tripod rotation).
fn three_phase_split(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leaf_limbs(c);
    if legs.len() < 3 {
        return false;
    }
    let forward = coin(rng);
    offset_limbs(c, &legs, |i| {
        let k = (i % 3) as f32 / 3.0;
        if forward { k } else { -k }
    })
}

/// Phase follows the distance from the head: a wave of contraction that runs
/// the length of the body at one speed (Ijspeert 2008), 0.3 to 1 cycle from
/// head to tail, in either direction.
fn traveling_wave_by_distance(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if c.muscles.len() < 2 {
        return false;
    }
    let head = [c.nodes[0].x, c.nodes[0].y];
    let far: Vec<f32> = (0..c.muscles.len())
        .map(|m| {
            let p = muscle_point(c, m);
            (p[0] - head[0]).hypot(p[1] - head[1])
        })
        .collect();
    let longest = far.iter().cloned().fold(0.0f32, f32::max);
    if longest < 0.05 {
        return false;
    }
    let cycles = rng.range(0.3, 1.0) * if coin(rng) { 1.0 } else { -1.0 };
    let base = c.muscles[0].phase - cycles * far[0] / longest;
    let mut changed = false;
    for (m, d) in far.iter().enumerate() {
        set(
            &mut c.muscles[m].phase,
            wrap(base + cycles * d / longest),
            &mut changed,
        );
    }
    changed
}

/// Reflects the phases of `ids` about the phase of the first of them, which
/// plays that part of the gait backward in time.
fn reflect(c: &mut Creature, ids: &MuscleIds) -> bool {
    let Some(&first) = ids.first() else {
        return false;
    };
    let pivot = c.muscles[first].phase;
    let mut changed = false;
    for &m in ids {
        let phase = wrap(2.0 * pivot - c.muscles[m].phase);
        set(&mut c.muscles[m].phase, phase, &mut changed);
    }
    changed
}

/// One leg's muscles play backward in time: the order in which its joints
/// fire reverses.
fn time_reverse_limb(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 2, true) else {
        return false;
    };
    let on = muscles_on(c, &leg, false);
    reflect(c, &on)
}

/// The whole gait plays backward in time.
fn time_reverse_body(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let all: MuscleIds = (0..c.muscles.len()).collect();
    reflect(c, &all)
}

/// One muscle's period changes by 1 to 4%, so it drifts against the others
/// and the gait beats instead of repeating at once.
fn detune_one(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let ids: Vec<usize> = (0..c.muscles.len()).collect();
    let Some(m) = pick(&ids, rng) else {
        return false;
    };
    let by = 1.0 + rng.range(0.01, 0.04) * if coin(rng) { 1.0 } else { -1.0 };
    let period = (c.muscles[m].period * by).clamp(min_muscle_period(), 10.0);
    let mut changed = false;
    set(&mut c.muscles[m].period, period, &mut changed);
    changed
}

/// Every period changes together by 4 to 12%: a faster or slower gait with
/// the same shape.
fn tempo_shift(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let by = 1.0 + rng.range(0.04, 0.12) * if coin(rng) { 1.0 } else { -1.0 };
    let mut changed = false;
    for m in &mut c.muscles {
        let period = (m.period * by).clamp(min_muscle_period(), 10.0);
        set(&mut m.period, period, &mut changed);
    }
    changed
}

/// Legs in order run at a period divided by 1, 2, 3, 1, 2, 3 and so on: a
/// ladder of harmonics, every leg a whole number of beats to the cycle.
fn harmonic_ladder(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leaf_limbs(c);
    if legs.len() < 2 {
        return false;
    }
    let base = c.muscles.iter().map(|m| m.period).fold(0.0f32, f32::max);
    let mut changed = false;
    for (i, leg) in legs.iter().enumerate() {
        let period = (base / (1 + i % 3) as f32).max(min_muscle_period());
        for m in muscles_on(c, leg, false) {
            set(&mut c.muscles[m].period, period, &mut changed);
        }
    }
    changed
}

/// One leg's muscles run on for 0.618 of the cycle (or 0.382 of it), the
/// golden split of work and rest.
fn duty_golden(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 1, true) else {
        return false;
    };
    let duty = if coin(rng) { GOLDEN } else { 1.0 - GOLDEN };
    let mut changed = false;
    for m in muscles_on(c, &leg, false) {
        set(&mut c.muscles[m].duty, duty, &mut changed);
    }
    changed
}

/// One limb of a matching pair works for the share of the cycle that the other
/// rests, starting when it stops: one limb pushes while the other swings, with
/// no overlap and no gap.
fn duty_complement_pair(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let pairs = matching_limbs(c);
    if pairs.is_empty() {
        return false;
    }
    let (x, y) = pairs.get(rng.index(pairs.len()));
    let (on_x, on_y) = (muscles_on(c, x, false), muscles_on(c, y, false));
    if on_x.is_empty() || on_y.is_empty() || on_x.iter().any(|m| on_y.contains(m)) {
        return false;
    }
    let mut changed = false;
    for (&p, &q) in on_x.iter().zip(on_y.iter()) {
        let (phase, duty) = (c.muscles[p].phase, c.muscles[p].duty);
        set(
            &mut c.muscles[q].duty,
            (1.0 - duty).clamp(0.05, 0.95),
            &mut changed,
        );
        set(&mut c.muscles[q].phase, wrap(phase + duty), &mut changed);
        c.muscles[q].period = c.muscles[p].period;
    }
    changed
}

/// Every phase snaps to the nearest step: eighth or sixth of a cycle (chosen
/// randomly), which tidies a gait whose phases drifted into a near-pattern.
fn phase_quantize(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let steps = if coin(rng) { 8.0 } else { 6.0 };
    let mut changed = false;
    for m in &mut c.muscles {
        let snapped = wrap((m.phase * steps).round() / steps);
        set(&mut m.phase, snapped, &mut changed);
    }
    changed
}

/// Two muscles trade phases.
fn swap_two_phases(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 2 {
        return false;
    }
    let (x, y) = (rng.index(c.muscles.len()), rng.index(c.muscles.len()));
    if x == y {
        return false;
    }
    let (p, q) = (c.muscles[x].phase, c.muscles[y].phase);
    let mut changed = false;
    set(&mut c.muscles[x].phase, q, &mut changed);
    set(&mut c.muscles[y].phase, p, &mut changed);
    changed
}

/// A muscle with a touchdown sensor jumps 15 to 35% of a cycle ahead of its
/// own phase when the foot lands, so each landing advances the gait.
fn reset_advance(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let ahead = rng.range(0.15, 0.35);
    let mut changed = false;
    for m in &mut c.muscles {
        if m.sensor != NO_SENSOR {
            set(&mut m.reset, wrap(m.phase + ahead), &mut changed);
        }
    }
    changed
}

/// The strongest muscle leads. The others follow it in order of strength, each
/// a share of the cycle after the one before, so the gait runs as a wave
/// from the muscle that does the most.
fn conductor(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 3 {
        return false;
    }
    let order = by_drive(c);
    let lead = c.muscles[order[0]].phase;
    let spread = rng.range(0.5, 1.0);
    let n = order.len() as f32;
    let mut changed = false;
    for (k, &m) in order.iter().enumerate().skip(1) {
        set(
            &mut c.muscles[m].phase,
            wrap(lead + spread * k as f32 / n),
            &mut changed,
        );
    }
    changed
}

/// The muscles in the rear half of the body (by position along it) run half
/// a cycle after those in the front half: a body that works in two halves
/// against each other.
fn halves_antiphase(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 2 {
        return false;
    }
    let xs: Vec<f32> = (0..c.muscles.len())
        .map(|m| muscle_point(c, m)[0])
        .collect();
    let mut sorted = xs.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let middle = sorted[sorted.len() / 2];
    let mut changed = false;
    for (m, &x) in xs.iter().enumerate() {
        if x < middle {
            let shifted = wrap(c.muscles[m].phase + 0.5);
            set(&mut c.muscles[m].phase, shifted, &mut changed);
        }
    }
    changed
}

/// Phase follows height: higher muscles fire later (or earlier) by up to a
/// half cycle, a wave that rises through the body.
fn phase_from_height(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 2 {
        return false;
    }
    let ys: Vec<f32> = (0..c.muscles.len())
        .map(|m| muscle_point(c, m)[1])
        .collect();
    let (low, high) = ys
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &y| (l.min(y), h.max(y)));
    if high - low < 0.05 {
        return false;
    }
    let cycles = rng.range(0.2, 0.5) * if coin(rng) { 1.0 } else { -1.0 };
    let base = c.muscles[0].phase - cycles * (ys[0] - low) / (high - low);
    let mut changed = false;
    for (m, y) in ys.iter().enumerate() {
        set(
            &mut c.muscles[m].phase,
            wrap(base + cycles * (y - low) / (high - low)),
            &mut changed,
        );
    }
    changed
}
