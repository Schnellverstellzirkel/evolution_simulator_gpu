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
use super::{Context, branch, muscles_on};
use crate::config::Config;
use crate::evolution::{Creature, NO_SENSOR, Rng, max_stroke};

/// The active muscles (with a stroke) that have an end on the limb starting
/// at `root`.
fn active_on(c: &Creature, root: usize) -> Vec<usize> {
    muscles_on(c, &branch(c, root), false)
        .into_iter()
        .filter(|&i| c.muscles[i].long > c.muscles[i].short)
        .collect()
}

/// Limb roots that have at least one active muscle.
fn driven_limbs(c: &Creature) -> Vec<usize> {
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
    let limbs: Vec<usize> = driven_limbs(c)
        .into_iter()
        .filter(|&b| active_on(c, b).len() > 1)
        .collect();
    let Some(root) = pick(&limbs, rng) else {
        return false;
    };
    let mut muscles = active_on(c, root);
    muscles.sort_by_key(|&i| c.muscles[i].bone_a.min(c.muscles[i].bone_b));
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
    let mut pairs = Vec::new();
    for &from in &limbs {
        let a = branch(c, from);
        for &to in &limbs {
            if !branch(c, to).iter().any(|b| a.contains(b)) {
                pairs.push((from, to));
            }
        }
    }
    let Some((from, to)) = pick(&pairs, rng) else {
        return false;
    };
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
    let active: Vec<usize> = (0..c.muscles.len())
        .filter(|&i| c.muscles[i].long > c.muscles[i].short)
        .collect();
    let key = |i: usize| {
        let m = &c.muscles[i];
        (m.bone_a.min(m.bone_b), m.bone_a.max(m.bone_b))
    };
    let mut pairs = Vec::new();
    for (n, &i) in active.iter().enumerate() {
        for &j in &active[n + 1..] {
            if key(i) == key(j) {
                pairs.push((i, j));
            }
        }
    }
    let Some((i, j)) = pick(&pairs, rng) else {
        return false;
    };
    let lag = if rng.unit() < 0.5 { 0.5 } else { 0.0 };
    let lead = c.muscles[i];
    let m = &mut c.muscles[j];
    let phase = (lead.phase + lag).rem_euclid(1.0);
    let reset = (lead.reset + lag).rem_euclid(1.0);
    let changed = (phase, lead.duty, reset) != (m.phase, m.duty, m.reset);
    (m.phase, m.duty, m.reset) = (phase, lead.duty, reset);
    changed
}

/// Clears the touchdown sensors of every muscle on a limb that has any, so
/// the limb runs on the clock alone (the reverse of `touchdown_package`).
pub(crate) fn release_touchdown(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs: Vec<usize> = limb_roots(c)
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
    let limbs: Vec<usize> = driven_limbs(c)
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
    use super::super::{Operator, tests::bodies};
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

    fn grown() -> Vec<Creature> {
        bodies(&Config::default(), 160)
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
}
