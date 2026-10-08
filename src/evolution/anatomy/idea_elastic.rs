//! Idea operators for springs, strokes and muscle strength: tendons, lever
//! arms, the stroke a muscle can use well, how a body spends its strength,
//! and the period of a leg that swings like a pendulum.
//!
//! Every operator is a whole change on its own, so its child gets no
//! parameter noise, and the operators of this file share one pick slot.
//!
//! The sources are Hill (1938, a muscle gives its peak power shortening at
//! about a third of its top speed), Alexander (1976, the swing time of a leg
//! is that of a pendulum of its length, and 1988, tendons return the energy
//! of a stride), Blickhan (1989) and Full and Koditschek (1999, a running
//! body behaves like a mass on a spring leg), Mochon and McMahon (1980, a
//! walking leg swings as a pendulum) and Pratt and Williamson (1995, a spring
//! in series with an actuator).
use super::ideas::{by_drive, coin, drive, set, some_leg};
use super::limbs::pick;
use super::muscles::shared_node;
use super::rhythm::leaf_limbs;
use super::{Context, Operator, fit_stroke, muscles_on, span};
use crate::config::Config;
use crate::evolution::{Creature, NO_SENSOR, Rng, max_stroke, min_muscle_period};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("slip_leg", slip_leg),
    ("hill_power_stroke", hill_power_stroke),
    ("pendulum_period_retune", pendulum_period_retune),
    ("series_elastic_actuators", series_elastic_actuators),
    ("tendon_longest_muscles", tendon_longest_muscles),
    ("tendon_strip_one", tendon_strip_one),
    ("stretch_reserve", stretch_reserve),
    ("peak_shaving", peak_shaving),
    ("stiffness_budget_shift", stiffness_budget_shift),
    ("rich_get_richer", rich_get_richer),
    ("tendon_reflex_pair", tendon_reflex_pair),
    ("gear_down", gear_down),
    ("gear_up", gear_up),
    ("equal_leverage_pair", equal_leverage_pair),
    ("marathon_gait", marathon_gait),
    ("sprint_gait", sprint_gait),
    ("tendon_tide", tendon_tide),
];

/// A leg becomes a spring-mass leg (Blickhan 1989, Full and Koditschek 1999).
/// Every muscle on it gets a stiff tendon (0.75 to 1.0), a `long` 15% longer
/// (capped at the longest muscle length) and 30% less stiffness. They all take
/// one period, the mean of the leg's muscles, so the spring and the muscles
/// keep time together.
fn slip_leg(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 1, true) else {
        return false;
    };
    let on = muscles_on(c, &leg, false);
    let period = on.iter().map(|&m| c.muscles[m].period).sum::<f32>() / on.len() as f32;
    let tendon = rng.range(0.75, 1.0);
    let mut changed = false;
    for &m in &on {
        let m = &mut c.muscles[m];
        let long = (m.long * 1.15).min(max_stroke().max(m.short));
        set(&mut m.tendon, tendon, &mut changed);
        set(&mut m.long, long, &mut changed);
        let stiffness = m.stiffness * 0.7;
        set(&mut m.stiffness, stiffness, &mut changed);
        set(&mut m.period, period, &mut changed);
    }
    changed
}

/// Sets the stroke of the muscles of one leg, or of all muscles, so they
/// shorten at about 0.3 of their top speed of 8 lengths a second, where a
/// muscle gives its peak power (Hill 1938). The shortening speed is the stroke
/// over the time the muscle is on. The new stroke is 5% to 75% of `long`. A
/// coin flip picks the leg or all muscles, and all muscles are taken when no
/// leg has a muscle on it.
fn hill_power_stroke(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let ids = match some_leg(c, rng, 1, true) {
        Some(leg) if coin(rng) => muscles_on(c, &leg, false),
        _ => (0..c.muscles.len()).collect(),
    };
    let mut changed = false;
    for &m in &ids {
        let m = &mut c.muscles[m];
        let on_time = (m.duty * m.period).max(0.05);
        let share = (1.0 - 2.4 * on_time).clamp(0.25, 0.95);
        let short = (m.long * share).max(0.01);
        set(&mut m.short, short, &mut changed);
    }
    changed
}

/// Every period of the body is scaled by one factor so that the mean period
/// becomes the swing time of a pendulum as long as the longest leg, 2 pi
/// sqrt(L / g), or twice that (Alexander 1976, Mochon and McMahon 1980). The
/// factor stays between 0.6 and 1.6, and the operator does nothing when it is
/// within 3% of 1. Ratios between clocks stay.
fn pendulum_period_retune(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.is_empty() {
        return false;
    }
    let legs = leaf_limbs(c);
    let length = |l: &[usize]| l.iter().map(|&b| c.bones[b].rest_length).sum::<f32>();
    let Some(longest) = legs.iter().map(|l| length(l)).max_by(|a, b| a.total_cmp(b)) else {
        return false;
    };
    let swing = std::f32::consts::TAU * (longest.max(0.05) / 9.81).sqrt();
    // A stride is a swing and a stance: the cycle is one swing, or two.
    let target = swing * if coin(rng) { 1.0 } else { 2.0 };
    let mean = c.muscles.iter().map(|m| m.period).sum::<f32>() / c.muscles.len() as f32;
    let factor = (target / mean).clamp(0.6, 1.6);
    if (factor - 1.0).abs() < 0.03 {
        return false;
    }
    let mut changed = false;
    for m in &mut c.muscles {
        let period = (m.period * factor).clamp(min_muscle_period(), 10.0);
        set(&mut m.period, period, &mut changed);
    }
    changed
}

/// Each muscle on a leg gets a medium tendon (at least a random 0.4 to 0.6),
/// like the spring of a series elastic actuator (Pratt and Williamson 1995),
/// and 20% more stiffness to pay for the give.
fn series_elastic_actuators(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 1, true) else {
        return false;
    };
    let mut changed = false;
    for m in muscles_on(c, &leg, false) {
        let m = &mut c.muscles[m];
        let tendon = m.tendon.max(rng.range(0.4, 0.6));
        set(&mut m.tendon, tendon, &mut changed);
        let stiffness = (m.stiffness * 1.2).min(120.0);
        set(&mut m.stiffness, stiffness, &mut changed);
    }
    changed
}

/// The third (rounded up) of the muscles with the longest span (the most
/// stretch to store) get a tendon of at least 0.7.
fn tendon_longest_muscles(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 3 {
        return false;
    }
    let mut ids: Vec<usize> = (0..c.muscles.len()).collect();
    ids.sort_by(|&a, &b| span(c, &c.muscles[b]).total_cmp(&span(c, &c.muscles[a])));
    let mut changed = false;
    for &m in &ids[..c.muscles.len().div_ceil(3)] {
        let tendon = c.muscles[m].tendon.max(0.7);
        set(&mut c.muscles[m].tendon, tendon, &mut changed);
    }
    changed
}

/// One springy muscle (a tendon above 0.1) loses its tendon and pulls 30%
/// harder instead.
fn tendon_strip_one(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let springy: Vec<usize> = (0..c.muscles.len())
        .filter(|&m| c.muscles[m].tendon > 0.1)
        .collect();
    let Some(m) = pick(&springy, rng) else {
        return false;
    };
    let m = &mut c.muscles[m];
    m.tendon = 0.0;
    m.stiffness = (m.stiffness * 1.3).min(120.0);
    true
}

/// One muscle gets a stroke 30% longer on its long side with a tendon 0.2
/// stiffer, so it starts each pull already stretched and the tendon holds
/// the energy.
fn stretch_reserve(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(m) = pick(&(0..c.muscles.len()).collect::<Vec<_>>(), rng) else {
        return false;
    };
    let m = &mut c.muscles[m];
    let long = (m.long * 1.3).min(max_stroke().max(m.short));
    let tendon = (m.tendon + 0.2).min(1.0);
    let mut changed = false;
    set(&mut m.long, long, &mut changed);
    set(&mut m.tendon, tendon, &mut changed);
    changed
}

/// The two strongest muscles (by `drive`) give up a quarter of their stiffness
/// and the two weakest take 30% more, so no one muscle carries the gait.
fn peak_shaving(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 4 {
        return false;
    }
    let ids = by_drive(c);
    let n = ids.len();
    let mut changed = false;
    for &m in &ids[..2] {
        let v = c.muscles[m].stiffness * 0.75;
        set(&mut c.muscles[m].stiffness, v, &mut changed);
    }
    for &m in &ids[n - 2..] {
        let v = (c.muscles[m].stiffness * 1.3).min(120.0);
        set(&mut c.muscles[m].stiffness, v, &mut changed);
    }
    changed
}

/// The muscles of one leg lose a quarter of their stiffness and the muscles of
/// another leg gain a quarter. Only legs with a muscle take part, and a muscle
/// that works on both legs stays as it is.
fn stiffness_budget_shift(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leaf_limbs(c);
    let driven: Vec<usize> = (0..legs.len())
        .filter(|&i| !muscles_on(c, &legs[i], false).is_empty())
        .collect();
    if driven.len() < 2 {
        return false;
    }
    let (x, y) = (
        driven[rng.index(driven.len())],
        driven[rng.index(driven.len())],
    );
    if x == y {
        return false;
    }
    let (from, to) = (
        muscles_on(c, &legs[x], false),
        muscles_on(c, &legs[y], false),
    );
    let mut changed = false;
    for &m in &from {
        if to.contains(&m) {
            continue;
        }
        let v = c.muscles[m].stiffness * 0.75;
        set(&mut c.muscles[m].stiffness, v, &mut changed);
    }
    for &m in &to {
        if from.contains(&m) {
            continue;
        }
        let v = (c.muscles[m].stiffness * 1.25).min(120.0);
        set(&mut c.muscles[m].stiffness, v, &mut changed);
    }
    changed
}

/// The strongest muscle (by `drive`) pulls 30% harder and the weakest 20% less.
fn rich_get_richer(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.len() < 2 {
        return false;
    }
    let ids = by_drive(c);
    let (best, worst) = (ids[0], ids[ids.len() - 1]);
    if drive(&c.muscles[best]) == drive(&c.muscles[worst]) {
        return false;
    }
    let mut changed = false;
    let up = (c.muscles[best].stiffness * 1.3).min(120.0);
    let down = c.muscles[worst].stiffness * 0.8;
    set(&mut c.muscles[best].stiffness, up, &mut changed);
    set(&mut c.muscles[worst].stiffness, down, &mut changed);
    changed
}

/// Muscles that sense touchdown get a tendon of at least 0.5: the landing
/// stretches the tendon, and the reflex then lets it go.
fn tendon_reflex_pair(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let mut changed = false;
    for m in &mut c.muscles {
        if m.sensor != NO_SENSOR {
            let tendon = m.tendon.max(0.5);
            set(&mut m.tendon, tendon, &mut changed);
        }
    }
    changed
}

/// Moves each anchor of each muscle in `ids` the share `by` of the way to the
/// end of its bone at the joint the muscle acts across (with `toward`) or to
/// the other end (without it), and refits the stroke to the new span with its
/// old proportions. A muscle whose bones share no node is skipped, and one
/// whose anchors move by 0.001 or less is left as it was. Returns whether any
/// muscle changed.
fn move_anchors(c: &mut Creature, ids: &[usize], by: f32, toward: bool) -> bool {
    let mut changed = false;
    for &i in ids {
        let old = c.muscles[i];
        let Some(joint) = shared_node(c, old.bone_a as usize, old.bone_b as usize) else {
            continue;
        };
        let end = |bone: u32, node: u32| -> f32 {
            let at_b = c.bones[bone as usize].b == node;
            if at_b == toward { 1.0 } else { 0.0 }
        };
        let (ta, tb) = (end(old.bone_a, joint), end(old.bone_b, joint));
        let mut m = old;
        m.anchor_a = clamp_unit(old.anchor_a + by * (ta - old.anchor_a));
        m.anchor_b = clamp_unit(old.anchor_b + by * (tb - old.anchor_b));
        fit_stroke(c, &mut m, Some(&old));
        if (m.anchor_a - old.anchor_a).abs() > 1.0e-3 || (m.anchor_b - old.anchor_b).abs() > 1.0e-3
        {
            c.muscles[i] = m;
            changed = true;
        }
    }
    changed
}

/// Clamps a position to the unit interval [0, 1].
fn clamp_unit(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// The muscles of a leg move their anchors 40% closer to the joint they
/// bend: a short lever arm, a fast and gentle muscle.
fn gear_down(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 1, true) else {
        return false;
    };
    let on = muscles_on(c, &leg, false);
    move_anchors(c, &on, 0.4, true)
}

/// The muscles of a leg move their anchors 40% of the way toward the far end
/// of their bones, away from the joint they bend: a long lever arm, a strong
/// and slow muscle.
fn gear_up(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = some_leg(c, rng, 1, true) else {
        return false;
    };
    let on = muscles_on(c, &leg, false);
    move_anchors(c, &on, 0.4, false)
}

/// Two muscles across the same pair of bones take the mean of their anchors,
/// so they pull on the same lever.
fn equal_leverage_pair(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for x in 0..c.muscles.len() {
        for y in x + 1..c.muscles.len() {
            let (p, q) = (c.muscles[x], c.muscles[y]);
            if (p.bone_a, p.bone_b) == (q.bone_a, q.bone_b)
                || (p.bone_a, p.bone_b) == (q.bone_b, q.bone_a)
            {
                pairs.push((x, y));
            }
        }
    }
    let Some((x, y)) = pick(&pairs, rng) else {
        return false;
    };
    let (p, q) = (c.muscles[x], c.muscles[y]);
    let same = (p.bone_a, p.bone_b) == (q.bone_a, q.bone_b);
    let (qa, qb) = if same {
        (q.anchor_a, q.anchor_b)
    } else {
        (q.anchor_b, q.anchor_a)
    };
    let (a, b) = (0.5 * (p.anchor_a + qa), 0.5 * (p.anchor_b + qb));
    let mut changed = false;
    set(&mut c.muscles[x].anchor_a, a, &mut changed);
    set(&mut c.muscles[x].anchor_b, b, &mut changed);
    if same {
        set(&mut c.muscles[y].anchor_a, a, &mut changed);
        set(&mut c.muscles[y].anchor_b, b, &mut changed);
    } else {
        set(&mut c.muscles[y].anchor_a, b, &mut changed);
        set(&mut c.muscles[y].anchor_b, a, &mut changed);
    }
    changed
}

/// Every muscle works a quarter longer each cycle and pulls a fifth less: a
/// body for distance.
fn marathon_gait(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let mut changed = false;
    for m in &mut c.muscles {
        let duty = (m.duty * 1.25).min(0.95);
        let stiffness = m.stiffness * 0.8;
        set(&mut m.duty, duty, &mut changed);
        set(&mut m.stiffness, stiffness, &mut changed);
    }
    changed
}

/// Every muscle works a fifth shorter each cycle and pulls a quarter harder:
/// a body for speed in short bursts.
fn sprint_gait(c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    let mut changed = false;
    for m in &mut c.muscles {
        let duty = (m.duty * 0.8).max(0.05);
        let stiffness = (m.stiffness * 1.25).min(120.0);
        set(&mut m.duty, duty, &mut changed);
        set(&mut m.stiffness, stiffness, &mut changed);
    }
    changed
}

/// Every tendon gets 0.15 stiffer, or every tendon loses half of its
/// stiffness. The whole body gets more or less elastic at once.
fn tendon_tide(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let up = coin(rng);
    let mut changed = false;
    for m in &mut c.muscles {
        let tendon = if up {
            (m.tendon + 0.15).min(1.0)
        } else {
            m.tendon * 0.5
        };
        set(&mut m.tendon, tendon, &mut changed);
    }
    changed
}
