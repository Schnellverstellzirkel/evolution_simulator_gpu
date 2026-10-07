//! These operators set a whole gait as a phase pattern over the legs (walk,
//! trot, pace, canter, gallop, bound, tripod, metachronal waves) or change the
//! duty factor of the legs. They change muscle timing only, share one pick slot
//! (`GAIT_FILES` in `mod.rs`) and are compound, so a child gets no parameter
//! noise after one. A leg is a leaf limb with a driven muscle, the legs run
//! from front to back in girdles of two (`Slot`), and the strongest muscle of
//! the first leg is the clock that the others are set against. The gait names
//! follow Hildebrand's footfall patterns, Alexander's duty factors and the
//! central pattern generator view of gaits (Collins and Stewart 1993,
//! couplings between oscillators fix the phase lags).
use super::compound::{shift_group, strongest};
use super::extra::drive;
use super::muscles::turn;
use super::rhythm::{limbs_front_to_back, muscle_groups};
use super::{Context, MuscleIds};
use crate::config::Config;
use crate::evolution::{Creature, Rng};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, super::Operator)] = &[
    ("quarter_beat_walk", quarter_beat_walk),
    ("diagonal_trot", diagonal_trot),
    ("lateral_pace", lateral_pace),
    ("three_beat_canter", three_beat_canter),
    ("spread_gallop", spread_gallop),
    ("half_bound", half_bound),
    ("alternating_tripod", alternating_tripod),
    ("paired_leg_wave", paired_leg_wave),
    ("double_ripple_wave", double_ripple_wave),
    ("shared_duty_factor", shared_duty_factor),
    ("fore_hind_duty_split", fore_hind_duty_split),
    ("snap_leg_lags", snap_leg_lags),
    ("reverse_leg_sequence", reverse_leg_sequence),
    ("change_leading_leg", change_leading_leg),
];

/// A leg with an active muscle: all its muscles, and the strongest one.
struct Leg {
    /// Every muscle with an end on the leg. A muscle on two legs belongs to the
    /// front one.
    muscles: MuscleIds,
    /// The muscle with the most drive. Its phase is taken as the leg's phase.
    lead: usize,
}

/// The legs with a driven muscle, from front to back.
fn legs_front_to_back(c: &Creature) -> Vec<Leg> {
    muscle_groups(c, &limbs_front_to_back(c))
        .iter()
        .filter_map(|group| {
            let active: MuscleIds = group
                .iter()
                .copied()
                .filter(|&i| drive(&c.muscles[i]) > 0.0)
                .collect();
            Some(Leg {
                muscles: *group,
                lead: strongest(c, &active)?,
            })
        })
        .collect()
}

/// The driven legs, if there are at least `least` of them.
fn legs_at_least(c: &Creature, least: usize) -> Option<Vec<Leg>> {
    Some(legs_front_to_back(c)).filter(|legs| legs.len() >= least)
}

/// Where a leg stands in the gait: its girdle (0 is the front one), its side
/// in the girdle (0 or 1) and how many girdles there are. The legs run from
/// front to back in girdles of two, so ranks 0 and 1 are the first girdle,
/// ranks 2 and 3 the second, and so on. A 2D body has no left and right, so
/// the two legs of a girdle take the place of the two sides.
#[derive(Clone, Copy)]
struct Slot {
    girdle: usize,
    side: usize,
    girdles: usize,
}

impl Slot {
    /// The slot of the leg at `rank` (0 is the front leg) among `legs` legs.
    fn of(rank: usize, legs: usize) -> Self {
        Self {
            girdle: rank / 2,
            side: rank % 2,
            girdles: legs.div_ceil(2),
        }
    }

    /// 0 for the front girdle to 1 for the back one. It is 0 when there is only
    /// one girdle.
    fn along(self) -> f32 {
        if self.girdles < 2 {
            0.0
        } else {
            self.girdle as f32 / (self.girdles - 1) as f32
        }
    }

    /// Whether the girdle is in the front half. With an odd number of girdles
    /// the middle one counts as front.
    fn fore(self) -> bool {
        2 * self.girdle < self.girdles
    }

    /// The side as an offset in cycles: 0 for side 0 and half a cycle for
    /// side 1.
    fn half(self) -> f32 {
        0.5 * self.side as f32
    }
}

/// Moves a leg's muscles (phase and touchdown reset) so its lead muscle lands
/// on phase `target`. Returns whether anything moved. A shift under 0.0001 of a
/// cycle counts as none.
fn retime_leg(c: &mut Creature, leg: &Leg, target: f32) -> bool {
    let shift = turn(c.muscles[leg.lead].phase, target);
    if shift.abs() < 1.0e-4 {
        return false;
    }
    shift_group(c, &leg.muscles, shift);
    true
}

/// Sets every leg to the offset (in cycles) `offset` gives for its slot,
/// measured from the first leg, which stays where it is. Returns whether any
/// leg moved.
fn set_pattern(c: &mut Creature, legs: &[Leg], offset: impl Fn(Slot) -> f32) -> bool {
    let origin = c.muscles[legs[0].lead].phase;
    let base = offset(Slot::of(0, legs.len()));
    let mut changed = false;
    for (rank, leg) in legs.iter().enumerate() {
        let target = origin + offset(Slot::of(rank, legs.len())) - base;
        changed |= retime_leg(c, leg, target.rem_euclid(1.0));
    }
    changed
}

/// A random sign: -1 or 1, each half of the time.
fn sign(rng: &mut Rng) -> f32 {
    if rng.unit() < 0.5 { -1.0 } else { 1.0 }
}

/// The offset of each leg's lead muscle from the first leg's, in cycles. Each
/// is the signed shortest turn (`turn`), so it lies from -0.5 up to 0.5.
fn offsets(c: &Creature, legs: &[Leg]) -> Vec<f32> {
    let origin = c.muscles[legs[0].lead].phase;
    legs.iter()
        .map(|leg| turn(origin, c.muscles[leg.lead].phase))
        .collect()
}

/// Sets absolute lead phases `targets`, one per leg. Returns whether any leg
/// moved.
fn set_targets(c: &mut Creature, legs: &[Leg], targets: &[f32]) -> bool {
    let mut changed = false;
    for (leg, &t) in legs.iter().zip(targets) {
        changed |= retime_leg(c, leg, t.rem_euclid(1.0));
    }
    changed
}

/// A walk: the legs of a girdle half a cycle apart and the girdles a quarter
/// cycle apart in all, so the feet land at quarter beats. The back girdle
/// follows the same side as the front one (lateral sequence, most
/// mammals) or the opposite side (diagonal sequence, primates and some
/// lizards). Four feet down most of the time keep the body statically stable,
/// which is the stride a body takes before it can run (Hildebrand 1965).
pub(crate) fn quarter_beat_walk(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 3) else {
        return false;
    };
    let lag = 0.25 * sign(rng);
    set_pattern(c, &legs, |s| s.half() + lag * s.along())
}

/// A trot: the two legs of a girdle half a cycle apart, and each leg moves
/// with the leg diagonally across from it (front left with back right). The
/// diagonal pairs keep the body balanced in two-leg support and cancel
/// pitching, so a trot is the most economical run at medium speed (Alexander
/// 1989, Full and Koditschek 1999 on the bouncing template).
pub(crate) fn diagonal_trot(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 3) else {
        return false;
    };
    if legs.len() > 4 {
        return false;
    }
    set_pattern(c, &legs, |s| 0.5 * ((s.girdle + s.side) % 2) as f32)
}

/// A pace: the legs on one side move together and the sides alternate, with
/// the back girdle landing a little ahead of or behind the front one. A pace
/// keeps the feet from striking each other on long bodies, and camels, giraffes
/// and some dogs use it. The body rolls instead of pitching.
pub(crate) fn lateral_pace(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(legs) = legs_at_least(c, 3) else {
        return false;
    };
    let lag = rng.range(0.02, 0.07) * sign(rng);
    set_pattern(c, &legs, |s| s.half() + lag * s.along())
}

/// A three-beat canter: one back leg lands first, the other back leg lands
/// together with its diagonal front leg, and the second front leg (the
/// leading leg) lands last, a third of a cycle apart. Which side leads is
/// random. The long diagonal pair gives a stride that is faster than the trot
/// with a soft, rocking suspension.
pub(crate) fn three_beat_canter(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 4) else {
        return false;
    };
    let third = 1.0 / 3.0;
    let flip = rng.index(2);
    set_pattern(c, &legs, |s| {
        let side = s.side ^ flip;
        match (s.fore(), side) {
            (false, 0) => 0.0,
            (false, _) => third,
            (true, 0) => third,
            (true, _) => 2.0 * third,
        }
    })
}

/// A four-beat gallop: the back legs land one after the other a tenth of a
/// cycle apart, then half a cycle later the front legs do, with the order of
/// the front legs either crossing (transverse gallop: horses, dogs) or
/// following the back legs' side order (rotary gallop: cheetahs turning,
/// rodents). Long flight phases between the beats come from the half-cycle
/// gap, as in the fast runs of Alexander's dynamic similarity work.
pub(crate) fn spread_gallop(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(legs) = legs_at_least(c, 4) else {
        return false;
    };
    let gap = rng.range(0.08, 0.14);
    let rotary = rng.index(2);
    let flip = rng.index(2);
    set_pattern(c, &legs, |s| {
        let side = s.side ^ flip;
        if s.fore() {
            0.5 + gap * (side ^ rotary ^ 1) as f32
        } else {
            gap * side as f32
        }
    })
}

/// A half bound: both front legs land together, both back legs land together
/// a third to two fifths of a cycle later, with a small lead of one side in
/// each pair. Small mammals bound this way, and the spine can work with the
/// legs, which a trot does not allow (Alexander on rodent and weasel runs).
pub(crate) fn half_bound(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(legs) = legs_at_least(c, 4) else {
        return false;
    };
    let hind = rng.range(0.3, 0.4) * sign(rng);
    let skew = rng.range(0.03, 0.09);
    set_pattern(c, &legs, |s| {
        let end = if s.fore() { 0.0 } else { hind };
        end + skew * s.side as f32
    })
}

/// An alternating tripod for bodies with five or more legs: the legs of
/// alternate sides in alternate girdles (front first, middle second, back
/// third) step together and the other three step half a cycle later. Each
/// half of the stride stands on a stable triangle, the gait that insects use
/// when they run (Full and Tu 1991, Cruse's walking controllers).
pub(crate) fn alternating_tripod(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 5) else {
        return false;
    };
    set_pattern(c, &legs, |s| 0.5 * ((s.girdle + s.side) % 2) as f32)
}

/// A wave of stepping along the body in girdles: the two legs of a girdle
/// stay half a cycle apart, and each girdle is ahead of or behind the one in
/// front by a small step. Waves that run from back to front (direct) or from
/// front to back (retrograde) are the metachronal gaits of centipedes and
/// millipedes. Several legs are always in stance, which makes a smooth
/// push.
pub(crate) fn paired_leg_wave(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 4) else {
        return false;
    };
    let step = rng.range(0.06, 0.18) * sign(rng);
    set_pattern(c, &legs, |s| s.half() + step * s.girdle as f32)
}

/// Two waves of stepping run along the body at once: the girdles' lags add up
/// to two full cycles from front to back, so the legs a half body apart step
/// together. The sides are half a cycle apart. This is the ripple of stick
/// insects and some millipedes, with shorter waves than a single sweep and
/// so a steadier body.
pub(crate) fn double_ripple_wave(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 5) else {
        return false;
    };
    let dir = sign(rng);
    set_pattern(c, &legs, |s| {
        s.half() + dir * 2.0 * s.girdle as f32 / s.girdles as f32
    })
}

/// Sets every driven muscle of every leg to one duty factor, a walking one
/// (0.55 to 0.75: each foot is down longer than it is up) or a running one
/// (0.25 to 0.45), and moves each phase so the middle of every contraction
/// stays where it was. Alexander's duty factor is the one number that
/// separates a walk from a run, and a gait that moves the duty of all legs
/// together keeps its footfall order.
pub(crate) fn shared_duty_factor(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = legs_front_to_back(c);
    if legs.is_empty() {
        return false;
    }
    let target = if rng.unit() < 0.5 {
        rng.range(0.55, 0.75)
    } else {
        rng.range(0.25, 0.45)
    };
    let mut changed = false;
    for leg in &legs {
        changed |= change_duty(c, &leg.muscles, |_| target);
    }
    changed
}

/// Gives the front half of the legs a longer duty factor and the back half a
/// shorter one by the same amount, or the reverse, keeping every contraction
/// centered. Front legs of most mammals carry more of the weight and stay on
/// the ground longer, while the back legs push and leave early (Alexander and
/// Jayes 1983, Lee et al. 2004 on limb forces).
pub(crate) fn fore_hind_duty_split(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 2) else {
        return false;
    };
    let amount = rng.range(0.08, 0.2) * sign(rng);
    let mut changed = false;
    for (rank, leg) in legs.iter().enumerate() {
        let front = 2 * rank < legs.len();
        let by = if front { amount } else { -amount };
        changed |= change_duty(c, &leg.muscles, |d| d + by);
    }
    changed
}

/// Sets the duty of each driven muscle in `group` to `f(duty)` (within 0.05
/// to 0.95), moving its phase and reset by half the change, so the middle of
/// its contraction stays put.
fn change_duty(c: &mut Creature, group: &[usize], f: impl Fn(f32) -> f32) -> bool {
    let mut changed = false;
    for &i in group {
        let m = &mut c.muscles[i];
        if drive(m) <= 0.0 {
            continue;
        }
        let duty = f(m.duty).clamp(0.05, 0.95);
        if (duty - m.duty).abs() < 1.0e-3 {
            continue;
        }
        let shift = 0.5 * (duty - m.duty);
        m.phase = (m.phase + shift).rem_euclid(1.0);
        m.reset = (m.reset + shift).rem_euclid(1.0);
        m.duty = duty;
        changed = true;
    }
    changed
}

/// Rounds every leg's lag behind the first leg to the nearest step of a
/// regular beat: halves, thirds, quarters or sixths of a cycle. A gait that
/// evolved with lags that drifted becomes the named gait it was closest to,
/// the way coupled oscillators lock to simple ratios (Golubitsky et al. 1999,
/// symmetry of gait networks).
pub(crate) fn snap_leg_lags(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(legs) = legs_at_least(c, 2) else {
        return false;
    };
    let beats = [2.0, 3.0, 4.0, 6.0][rng.index(4)];
    let origin = c.muscles[legs[0].lead].phase;
    let targets: Vec<f32> = offsets(c, &legs)
        .iter()
        .map(|d| origin + (d * beats).round() / beats)
        .collect();
    set_targets(c, &legs, &targets)
}

/// Reverses the order in which the legs step: each leg's lag behind the first
/// leg changes sign. A wave from back to front becomes one from front to back
/// and a lateral walk becomes a diagonal one, while each leg keeps its own
/// stroke. Gaits of the same footfall spacing differ in which way the weight
/// is passed along the body.
pub(crate) fn reverse_leg_sequence(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 3) else {
        return false;
    };
    let origin = c.muscles[legs[0].lead].phase;
    let targets: Vec<f32> = offsets(c, &legs).iter().map(|d| origin - d).collect();
    set_targets(c, &legs, &targets)
}

/// Swaps the timing of the two legs of every girdle, so the side that led now
/// follows. A horse changes its leading leg in a canter or a gallop to turn
/// and to rest the muscles of one side, and the other way of the same gait is
/// a different local optimum for a body that is not symmetric.
pub(crate) fn change_leading_leg(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(legs) = legs_at_least(c, 2) else {
        return false;
    };
    let origin = c.muscles[legs[0].lead].phase;
    let mut targets: Vec<f32> = offsets(c, &legs).iter().map(|d| origin + d).collect();
    for pair in targets.as_chunks_mut::<2>().0 {
        pair.swap(0, 1);
    }
    set_targets(c, &legs, &targets)
}

#[cfg(test)]
mod tests {
    use super::super::tests::bodies;
    use super::*;
    use crate::evolution::repair;

    #[test]
    fn each_operator_keeps_the_body_and_moves_only_timing() {
        let cfg = Config::default();
        for &(name, op) in OPS {
            let mut applied = 0;
            for (i, body) in bodies(&cfg, 160).into_iter().enumerate() {
                let mut c = body.clone();
                let cx = Context::of(None);
                if op(&mut c, &cfg, &mut Rng::new(71, 0, i), &cx) {
                    applied += 1;
                    assert_eq!(c.nodes, body.nodes, "{name}");
                    assert_eq!(c.bones, body.bones, "{name}");
                    assert_eq!(c.muscles.len(), body.muscles.len(), "{name}");
                    assert!(c.muscles != body.muscles, "{name}");
                    repair(&mut c, &cfg, &mut Rng::new(73, 0, i));
                } else {
                    assert!(c.muscles == body.muscles, "{name}");
                }
            }
            assert!(
                applied > 0 || name.contains("tripod") || name.contains("ripple"),
                "{name}"
            );
        }
    }
}
