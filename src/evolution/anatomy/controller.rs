//! Operators that retune the controller of a whole limb or of a pair of
//! muscles at one joint, without touching the skeleton: scale a limb's stroke,
//! shift its posture, taper its strength along the chain, copy one limb's
//! rhythm onto another limb with an offset, and set two muscles of one joint
//! to alternate or to act together.
//!
//! Like the other muscle and rhythm operators they change no bone or node,
//! so the motor ring stays as it is. They only touch active muscles (a
//! stroke longer than zero), never the passive ring.
use super::limbs::{limb_roots, pick};
use super::{BoneIds, Context, MuscleIds, branch, muscles_on};
use crate::config::Config;
use crate::evolution::{
    Bounded, CLOCK_RATIOS, Creature, MAX_MUSCLES, MAX_NODES, NO_SENSOR, Rng, max_stroke,
    min_muscle_period,
};

/// The active muscles (with a stroke) that have an end on the limb starting
/// at `root`.
pub(super) fn active_on(c: &Creature, root: usize) -> MuscleIds {
    muscles_on(c, &branch(c, root), false)
        .into_iter()
        .filter(|&i| c.muscles[i].long > c.muscles[i].short)
        .collect()
}

/// Limb roots that have at least one active muscle.
fn driven_limbs(c: &Creature) -> BoneIds {
    limb_roots(c)
        .into_iter()
        .filter(|&b| !active_on(c, b).is_empty())
        .collect()
}

/// Scales the stroke of every active muscle on a limb about its middle by one
/// factor (0.6 to 1.6): the limb swings through a wider or narrower arc with
/// the same clock, so its muscles contract more slowly or more quickly.
pub(crate) fn limb_stroke_scale(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(root) = pick(&driven_limbs(c), rng) else {
        return false;
    };
    let factor = rng.range(0.6f32.ln(), 1.6f32.ln()).exp();
    let mut changed = false;
    for i in active_on(c, root) {
        let m = &mut c.muscles[i];
        let (middle, half) = ((m.short + m.long) * 0.5, (m.long - m.short) * 0.5 * factor);
        let short = (middle - half).max(0.01);
        let long = (middle + half).clamp(short, max_stroke());
        changed |= (short, long) != (m.short, m.long);
        (m.short, m.long) = (short, long);
    }
    changed
}

/// Moves both ends of every active muscle's stroke on a limb by the same
/// share (5 to 15%) of its stroke, up or down: the limb rests more bent or
/// more stretched and swings around a new posture.
pub(crate) fn limb_posture_shift(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(root) = pick(&driven_limbs(c), rng) else {
        return false;
    };
    let share = rng.range(0.05, 0.15) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    let mut changed = false;
    for i in active_on(c, root) {
        let m = &mut c.muscles[i];
        let (stroke, top) = (m.long - m.short, max_stroke());
        let short = (m.short + share * stroke).clamp(0.01, (top - stroke).max(0.01));
        let long = (short + stroke).min(top);
        changed |= (short, long) != (m.short, m.long);
        (m.short, m.long) = (short, long);
    }
    changed
}

/// Tapers the strength of a limb's active muscles along the chain, parents
/// first: stiffness changes by a factor of 1.2 to 1.8 between the first and
/// the last muscle, stronger at the root or at the tip (a hip that drives
/// and a foot that yields, or the reverse).
pub(crate) fn taper_limb_strength(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs: BoneIds = driven_limbs(c)
        .into_iter()
        .filter(|&b| active_on(c, b).len() > 1)
        .collect();
    let Some(root) = pick(&limbs, rng) else {
        return false;
    };
    let mut muscles = active_on(c, root);
    let key = |i: usize| c.muscles[i].bone_a.min(c.muscles[i].bone_b);
    muscles.sort_stable_by(|&i, &j| key(i).cmp(&key(j)));
    let ratio = rng.range(1.2, 1.8) * if rng.unit() < 0.5 { 1.0 } else { -1.0 };
    let last = (muscles.len() - 1) as f32;
    let mut changed = false;
    for (rank, &i) in muscles.iter().enumerate() {
        let along = rank as f32 / last * 2.0 - 1.0;
        let factor = ratio.abs().powf(along * ratio.signum());
        let m = &mut c.muscles[i];
        let stiffness = (m.stiffness * factor).clamp(1.0, 120.0);
        changed |= stiffness != m.stiffness;
        m.stiffness = stiffness;
    }
    changed
}

/// Copies the rhythm of one limb onto a different limb of any shape, later
/// by a quarter, a half or three quarters of a cycle (plus a little noise):
/// the target's k-th muscle takes the phase, duty and touchdown reset of the
/// source's k-th muscle (wrapping when the target has more). The two limbs
/// share one step pattern at a fixed lag.
pub(crate) fn copy_limb_rhythm(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs = driven_limbs(c);
    let mut pairs: Bounded<(u8, u8), { MAX_NODES * MAX_NODES }> = Bounded::new();
    for &from in &limbs {
        let a = branch(c, from);
        for &to in &limbs {
            if !branch(c, to).iter().any(|b| a.contains(b)) {
                pairs.push((from as u8, to as u8));
            }
        }
    }
    let Some((from, to)) = pick(&pairs, rng) else {
        return false;
    };
    let (from, to) = (from as usize, to as usize);
    let source = active_on(c, from);
    let target = active_on(c, to);
    let offset = [0.25, 0.5, 0.75][rng.index(3)] + rng.range(-0.04, 0.04);
    let mut changed = false;
    for (k, &j) in target.iter().enumerate() {
        let s = c.muscles[source[k % source.len()]];
        let m = &mut c.muscles[j];
        let phase = (s.phase + offset).rem_euclid(1.0);
        let reset = (s.reset + offset).rem_euclid(1.0);
        changed |= (phase, s.duty, reset) != (m.phase, m.duty, m.reset);
        (m.phase, m.duty, m.reset) = (phase, s.duty, reset);
    }
    changed
}

/// Two active muscles across the same pair of bones (a joint's opener and
/// closer, or two synergists) are set half a cycle apart, or into the same
/// phase, with the second one's duty and touchdown reset following.
pub(crate) fn retune_muscle_pair(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let active: MuscleIds = (0..c.muscles.len())
        .filter(|&i| c.muscles[i].long > c.muscles[i].short)
        .collect();
    let key = |i: usize| {
        let m = &c.muscles[i];
        (m.bone_a.min(m.bone_b), m.bone_a.max(m.bone_b))
    };
    let mut pairs: Bounded<(u8, u8), { MAX_MUSCLES * MAX_MUSCLES / 2 }> = Bounded::new();
    for (n, &i) in active.iter().enumerate() {
        for &j in &active[n + 1..] {
            if key(i) == key(j) {
                pairs.push((i as u8, j as u8));
            }
        }
    }
    let Some((i, j)) = pick(&pairs, rng) else {
        return false;
    };
    let (i, j) = (i as usize, j as usize);
    let lag = if rng.unit() < 0.5 { 0.5 } else { 0.0 };
    let lead = c.muscles[i];
    let m = &mut c.muscles[j];
    let phase = (lead.phase + lag).rem_euclid(1.0);
    let reset = (lead.reset + lag).rem_euclid(1.0);
    let changed = (phase, lead.duty, reset) != (m.phase, m.duty, m.reset);
    (m.phase, m.duty, m.reset) = (phase, lead.duty, reset);
    changed
}

/// Puts the muscles of a limb on a different clock: a simple multiple of the
/// body's base clock (`CLOCK_RATIOS`), so the limb steps faster or slower than
/// the rest and the whole gait still repeats exactly. Skipped when the change
/// would leave some muscle at a ratio outside that set.
pub(crate) fn limb_clock_ratio(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(root) = pick(&driven_limbs(c), rng) else {
        return false;
    };
    let limb = active_on(c, root);
    let Some(anchor) = (0..c.muscles.len()).find(|i| !limb.contains(i)) else {
        return false;
    };
    // The rest of the body's clock is the reference the ratio applies to.
    let period = c.muscles[anchor].period * CLOCK_RATIOS[rng.index(CLOCK_RATIOS.len())];
    if !(min_muscle_period()..=10.0).contains(&period)
        || limb.iter().all(|&i| c.muscles[i].period == period)
    {
        return false;
    }
    let mut periods: Bounded<f32, MAX_MUSCLES> = c.muscles.iter().map(|m| m.period).collect();
    for &i in &limb {
        periods[i] = period;
    }
    let base = periods[0];
    let in_set = |p: f32| {
        CLOCK_RATIOS
            .iter()
            .any(|r| ((p / base).ln() - r.ln()).abs() < 0.003)
    };
    if !periods.iter().all(|&p| in_set(p)) {
        return false;
    }
    for (m, p) in c.muscles.iter_mut().zip(periods) {
        m.period = p;
    }
    true
}

/// Starts the same gait at another point of its cycle: every muscle's clock
/// moves ahead by one common time, so the steady gait is unchanged and only
/// the start differs. On the best elites of a 120-generation save, 83 of 124
/// parent-to-child jumps of 1.5x and 10 m or more were reached again by one of
/// 7 start offsets of the parent alone, and an offset left the top 100 at a
/// median 8% of their distance: whether a gait catches depends on its start.
pub(crate) fn shift_gait_start(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(longest) = c.muscles.iter().map(|m| m.period).reduce(f32::max) else {
        return false;
    };
    let dt = rng.range(0.05, 0.95) * longest;
    for m in &mut c.muscles {
        m.phase = (m.phase + dt / m.period).rem_euclid(1.0);
    }
    true
}

/// Puts every muscle of a limb back on the body's base clock.
pub(crate) fn limb_clock_lock(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(base) = c.muscles.first().map(|m| m.period) else {
        return false;
    };
    let off: BoneIds = driven_limbs(c)
        .into_iter()
        .filter(|&b| active_on(c, b).iter().any(|&i| c.muscles[i].period != base))
        .collect();
    let Some(root) = pick(&off, rng) else {
        return false;
    };
    for i in active_on(c, root) {
        c.muscles[i].period = base;
    }
    true
}

/// The foot nodes of a muscle's two bones (nodes with one bone, not the
/// head), as sensor indices (0 and 1 are the first bone's ends, 2 and 3 the
/// second's).
fn sensable_feet(c: &Creature, m: &crate::evolution::Muscle) -> Bounded<u32, 4> {
    let (a, b) = (c.bones[m.bone_a as usize], c.bones[m.bone_b as usize]);
    [a.a, a.b, b.a, b.b]
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n != 0 && super::degree(c, n as usize) == 1)
        .map(|(k, _)| k as u32)
        .collect()
}

/// A muscle without a sensor starts to sense the touchdown of a foot at one
/// of its ends, with a random cycle position to restart at: a reflex that
/// fires the muscle when its foot lands.
pub(crate) fn reflex_on_muscle(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let options: Bounded<(usize, Bounded<u32, 4>), MAX_MUSCLES> = (0..c.muscles.len())
        .filter(|&i| c.muscles[i].long > c.muscles[i].short && c.muscles[i].sensor == NO_SENSOR)
        .map(|i| (i, sensable_feet(c, &c.muscles[i])))
        .filter(|(_, feet)| !feet.is_empty())
        .collect();
    if options.is_empty() {
        return false;
    }
    let (i, feet) = &options[rng.index(options.len())];
    let m = &mut c.muscles[*i];
    m.sensor = feet[rng.index(feet.len())];
    m.reset = rng.unit();
    true
}

/// Every active muscle with an end on a foot senses that foot's touchdown,
/// all at once, keeping the muscles' phase order in their reset positions.
pub(crate) fn reflex_all_feet(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut changed = false;
    let shift = rng.unit();
    for i in 0..c.muscles.len() {
        let m = c.muscles[i];
        if m.long <= m.short {
            continue;
        }
        let Some(&sensor) = sensable_feet(c, &m).first() else {
            continue;
        };
        let reset = (m.phase + shift).rem_euclid(1.0);
        changed |= (m.sensor, m.reset) != (sensor, reset);
        c.muscles[i].sensor = sensor;
        c.muscles[i].reset = reset;
    }
    changed
}

/// Moves the reset position of every sensing muscle on a limb by one step (5
/// to 25% of a cycle), so the reflex restarts the limb earlier or later in
/// its cycle without changing its clock.
pub(crate) fn reflex_reset_shift(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| {
            muscles_on(c, &branch(c, b), false)
                .iter()
                .any(|&i| c.muscles[i].sensor != NO_SENSOR)
        })
        .collect();
    let Some(root) = pick(&limbs, rng) else {
        return false;
    };
    let step = rng.range(0.05, 0.25) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    for i in muscles_on(c, &branch(c, root), false) {
        let m = &mut c.muscles[i];
        if m.sensor != NO_SENSOR {
            m.reset = (m.reset + step).rem_euclid(1.0);
        }
    }
    true
}

/// Clears the touchdown sensors of every muscle on a limb that has any, so
/// the limb runs on the clock alone (the reverse of `touchdown_package`).
pub(crate) fn release_touchdown(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| {
            muscles_on(c, &branch(c, b), false)
                .iter()
                .any(|&i| c.muscles[i].sensor != NO_SENSOR)
        })
        .collect();
    let Some(root) = pick(&limbs, rng) else {
        return false;
    };
    for i in muscles_on(c, &branch(c, root), false) {
        c.muscles[i].sensor = NO_SENSOR;
    }
    true
}

/// Rounds the phases of a limb's active muscles to the nearest eighth of a
/// cycle, measured from the limb's first muscle, so near-alternations become
/// exact ones and the limb's steps line up on a regular grid.
pub(crate) fn snap_limb_phases(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs: BoneIds = driven_limbs(c)
        .into_iter()
        .filter(|&b| active_on(c, b).len() > 1)
        .collect();
    let Some(root) = pick(&limbs, rng) else {
        return false;
    };
    let muscles = active_on(c, root);
    let origin = c.muscles[muscles[0]].phase;
    let mut changed = false;
    for &i in &muscles[1..] {
        let m = &mut c.muscles[i];
        let snapped = origin + ((m.phase - origin) * 8.0).round() / 8.0;
        let phase = snapped.rem_euclid(1.0);
        changed |= (phase - m.phase).abs() > 1e-6;
        m.phase = phase;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::super::{Operator, tests::grown};
    use super::*;

    fn run(op: Operator, bodies: &[Creature], check: impl Fn(&Creature, &Creature)) -> usize {
        let cfg = Config::default();
        let mut applied = 0;
        for (i, body) in bodies.iter().enumerate() {
            let mut c = body.clone();
            let cx = Context { donor: None };
            if op(&mut c, &cfg, &mut Rng::new(41, 0, i), &cx) {
                applied += 1;
                assert_eq!((&c.nodes, &c.bones), (&body.nodes, &body.bones));
                assert_eq!(c.muscles.len(), body.muscles.len());
                check(body, &c);
            } else {
                assert!(c.muscles == body.muscles);
            }
        }
        applied
    }

    fn changed(before: &Creature, after: &Creature) -> Vec<usize> {
        (0..before.muscles.len())
            .filter(|&i| before.muscles[i] != after.muscles[i])
            .collect()
    }

    fn on_one_limb(before: &Creature, changed: &[usize]) -> bool {
        limb_roots(before).into_iter().any(|root| {
            let on = active_on(before, root);
            changed.iter().all(|i| on.contains(i))
        })
    }

    #[test]
    fn stroke_scale_keeps_the_middle_and_scales_one_limb() {
        let applied = run(limb_stroke_scale, &grown(), |before, after| {
            let moved = changed(before, after);
            assert!(on_one_limb(before, &moved));
            for i in moved {
                let (x, y) = (before.muscles[i], after.muscles[i]);
                assert!(y.short >= 0.01 && y.long >= y.short && y.long <= max_stroke());
                assert_eq!(
                    (x.phase, x.stiffness, x.period),
                    (y.phase, y.stiffness, y.period)
                );
                assert!(y.long > y.short);
            }
        });
        assert!(applied >= 120, "applied {applied}");
    }

    #[test]
    fn posture_shift_keeps_every_stroke_length() {
        let applied = run(limb_posture_shift, &grown(), |before, after| {
            let moved = changed(before, after);
            assert!(on_one_limb(before, &moved));
            for i in moved {
                let (x, y) = (before.muscles[i], after.muscles[i]);
                assert!(y.short >= 0.01 && y.long <= max_stroke());
                assert!(y.long - y.short <= (x.long - x.short) + 1e-5);
                assert_eq!((x.phase, x.stiffness), (y.phase, y.stiffness));
            }
        });
        assert!(applied >= 120, "applied {applied}");
    }

    #[test]
    fn taper_changes_only_stiffness_along_one_limb() {
        let applied = run(taper_limb_strength, &grown(), |before, after| {
            let moved = changed(before, after);
            assert!(on_one_limb(before, &moved));
            for i in moved {
                let (x, y) = (before.muscles[i], after.muscles[i]);
                assert_eq!(
                    crate::evolution::Muscle {
                        stiffness: x.stiffness,
                        ..y
                    },
                    x
                );
                assert!((1.0..=120.0).contains(&y.stiffness));
            }
        });
        assert!(applied >= 60, "applied {applied}");
    }

    #[test]
    fn copy_limb_rhythm_gives_the_target_the_source_rhythm_at_a_lag() {
        let applied = run(copy_limb_rhythm, &grown(), |before, after| {
            let moved = changed(before, after);
            assert!(!moved.is_empty());
            for i in moved {
                let (x, y) = (before.muscles[i], after.muscles[i]);
                assert_eq!(
                    crate::evolution::Muscle {
                        phase: x.phase,
                        duty: x.duty,
                        reset: x.reset,
                        tendon: 0.0,
                        ..y
                    },
                    x
                );
            }
        });
        assert!(applied >= 60, "applied {applied}");
    }

    #[test]
    fn muscle_pair_ends_in_phase_or_half_a_cycle_apart() {
        let applied = run(retune_muscle_pair, &grown(), |before, after| {
            let moved = changed(before, after);
            assert_eq!(moved.len(), 1);
            let j = moved[0];
            let y = after.muscles[j];
            let ok = (0..after.muscles.len()).any(|i| {
                let x = after.muscles[i];
                i != j
                    && (x.bone_a.min(x.bone_b), x.bone_a.max(x.bone_b))
                        == (y.bone_a.min(y.bone_b), y.bone_a.max(y.bone_b))
                    && [0.0f32, 0.5].iter().any(|lag| {
                        let d = (y.phase - x.phase - lag).rem_euclid(1.0);
                        !(1e-4..=1.0 - 1e-4).contains(&d)
                    })
            });
            assert!(ok);
        });
        assert!(applied >= 10, "applied {applied}");
    }

    #[test]
    fn release_touchdown_clears_sensors_of_one_limb() {
        let mut bodies = grown();
        for c in &mut bodies {
            if let Some(m) = c.muscles.iter_mut().find(|m| m.long > m.short) {
                m.sensor = 0;
            }
        }
        let applied = run(release_touchdown, &bodies, |before, after| {
            let moved = changed(before, after);
            assert!(!moved.is_empty());
            for i in moved {
                assert_eq!(after.muscles[i].sensor, NO_SENSOR);
                assert_ne!(before.muscles[i].sensor, NO_SENSOR);
            }
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn snapping_puts_phase_gaps_on_eighths() {
        let applied = run(snap_limb_phases, &grown(), |before, after| {
            let moved = changed(before, after);
            assert!(on_one_limb(before, &moved));
            for i in moved {
                let (x, y) = (before.muscles[i], after.muscles[i]);
                assert_eq!((x.duty, x.stiffness), (y.duty, y.stiffness));
                assert!((y.phase - x.phase + 0.5).rem_euclid(1.0) - 0.5 < 0.0625 + 1e-4);
            }
        });
        assert!(applied >= 40, "applied {applied}");
    }

    #[test]
    fn limb_clock_ratio_keeps_every_period_on_the_ratio_set() {
        let applied = run(limb_clock_ratio, &grown(), |before, after| {
            let base = after.muscles[0].period;
            for m in &after.muscles {
                assert!(
                    CLOCK_RATIOS
                        .iter()
                        .any(|r| ((m.period / base).ln() - r.ln()).abs() < 0.003),
                    "ratio {}",
                    m.period / base
                );
            }
            assert!(changed(before, after).iter().all(|&i| {
                let (x, y) = (before.muscles[i], after.muscles[i]);
                crate::evolution::Muscle {
                    period: x.period,
                    ..y
                } == x
            }));
        });
        assert!(applied >= 30, "applied {applied}");
    }

    #[test]
    fn a_repaired_body_keeps_its_limb_clocks() {
        let cfg = Config::default();
        let mut kept = 0;
        for (i, mut c) in grown().into_iter().enumerate() {
            let cx = Context { donor: None };
            if !limb_clock_ratio(&mut c, &cfg, &mut Rng::new(5, 0, i), &cx) {
                continue;
            }
            let before: Vec<f32> = c.muscles.iter().map(|m| m.period).collect();
            crate::evolution::repair(&mut c, &cfg, &mut Rng::new(6, 0, i));
            let after: Vec<f32> = c.muscles.iter().map(|m| m.period).collect();
            if before == after && after.iter().any(|&p| p != after[0]) {
                kept += 1;
            }
        }
        assert!(kept >= 20, "kept {kept}");
    }

    #[test]
    fn limb_clock_lock_puts_a_limb_on_the_base_clock() {
        let cfg = Config::default();
        let bodies: Vec<Creature> = grown()
            .into_iter()
            .enumerate()
            .filter_map(|(i, mut c)| {
                let cx = Context { donor: None };
                limb_clock_ratio(&mut c, &cfg, &mut Rng::new(9, 0, i), &cx).then_some(c)
            })
            .collect();
        let applied = run(limb_clock_lock, &bodies, |before, after| {
            let base = after.muscles[0].period;
            let moved = changed(before, after);
            assert!(moved.iter().all(|&i| after.muscles[i].period == base));
        });
        assert!(applied >= 10, "applied {applied}");
    }

    #[test]
    fn reflex_on_muscle_senses_a_foot_of_its_own_bones() {
        let applied = run(reflex_on_muscle, &grown(), |before, after| {
            let moved = changed(before, after);
            assert_eq!(moved.len(), 1);
            let m = after.muscles[moved[0]];
            assert_eq!(before.muscles[moved[0]].sensor, NO_SENSOR);
            assert!(sensable_feet(after, &m).contains(&m.sensor));
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn reflex_all_feet_and_reset_shift_only_touch_sensors_and_resets() {
        for op in [reflex_all_feet as Operator, reflex_reset_shift] {
            let mut bodies = grown();
            for c in &mut bodies {
                if let Some(m) = c.muscles.iter_mut().find(|m| m.long > m.short) {
                    m.sensor = 0;
                }
            }
            let applied = run(op, &bodies, |before, after| {
                for i in changed(before, after) {
                    let (x, y) = (before.muscles[i], after.muscles[i]);
                    assert_eq!(
                        crate::evolution::Muscle {
                            sensor: x.sensor,
                            reset: x.reset,
                            ..y
                        },
                        x
                    );
                }
            });
            assert!(applied >= 20, "applied {applied}");
        }
    }
}
