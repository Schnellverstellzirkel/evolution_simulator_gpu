//! Gait operators: sensing and reflexes that coordinate legs.
//!
//! The genome senses one thing: a muscle can restart its clock at a chosen
//! cycle position (`reset`) when one of the four ends of its two bones, which
//! must be a foot, touches the ground (`sensor`). From that, these operators
//! build the rules of Cruse's walknet and of the half-centre coupling in
//! central pattern generators (Full and Koditschek's templates, Alexander's
//! duty factor): a landing starts the stance stroke, a landing triggers the
//! next leg through a small bridge muscle, legs alternate or wave along the
//! body with their reflexes set to match.
//!
//! In every operator "the start of the stroke" is cycle position 0, where a
//! muscle begins to contract. A leg's reset puts each of its sensing muscles
//! where it would be when the leg's strongest muscle is at position 0, so a
//! landing keeps the leg's own timing and only restarts it.
//!
//! The operators share one pick slot and are compound: each is a whole change
//! and its child gets no parameter noise.
use super::compound::strongest;
use super::limbs::pick;
use super::rhythm::{foot, leaf_limbs, tip_x};
use super::{
    BoneIds, Context, Limbs, MuscleIds, Operator, is_neck, muscles_on, new_muscle, room, span,
};
use crate::config::Config;
use crate::evolution::{Creature, NO_SENSOR, Rng};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("landing_starts_stroke", landing_starts_stroke),
    ("quick_lift_reflex", quick_lift_reflex),
    ("cross_leg_trigger", cross_leg_trigger),
    ("fore_to_hind_trigger", fore_to_hind_trigger),
    ("mutual_leg_trigger", mutual_leg_trigger),
    ("reflex_wave_along_legs", reflex_wave_along_legs),
    ("alternate_legs_with_reflex", alternate_legs_with_reflex),
    ("fore_hind_pairing", fore_hind_pairing),
    ("landing_flexes_trunk", landing_flexes_trunk),
    ("touchdown_stiffener", touchdown_stiffener),
    ("spread_leg_reflex", spread_leg_reflex),
    ("trigger_chain_along_legs", trigger_chain_along_legs),
    ("stance_duty_with_reflex", stance_duty_with_reflex),
];

/// The legs (leaf limbs of at least two bones), from the rearmost foot to the
/// foremost.
fn legs_by_x(c: &Creature) -> Limbs {
    let mut legs: Limbs = leaf_limbs(c)
        .iter()
        .copied()
        .filter(|l| l.len() >= 2)
        .collect();
    legs.sort_stable_by(|p, q| tip_x(c, p).total_cmp(&tip_x(c, q)));
    legs
}

/// The muscles with a stroke that have an end on the leg.
fn active_on(c: &Creature, leg: &[usize]) -> MuscleIds {
    muscles_on(c, leg, false)
        .into_iter()
        .filter(|&i| c.muscles[i].long > c.muscles[i].short)
        .collect()
}

/// The sensor index (0 to 3) that reads node `node` for muscle `i`.
fn sensor_at(c: &Creature, i: usize, node: usize) -> Option<u32> {
    let m = &c.muscles[i];
    let (a, b) = (c.bones[m.bone_a as usize], c.bones[m.bone_b as usize]);
    [a.a, a.b, b.a, b.b]
        .iter()
        .position(|&n| n as usize == node)
        .map(|k| k as u32)
}

/// The leg's active muscles that can sense its foot.
fn sensing(c: &Creature, leg: &[usize]) -> MuscleIds {
    let f = foot(c, leg);
    active_on(c, leg)
        .into_iter()
        .filter(|&i| sensor_at(c, i, f).is_some())
        .collect()
}

/// Legs whose foot a muscle can sense.
fn sensing_legs(c: &Creature) -> Limbs {
    legs_by_x(c)
        .into_iter()
        .filter(|l| !sensing(c, l).is_empty())
        .collect()
}

/// Makes every sensing muscle of the leg sense its foot, restarting at the
/// cycle position `reset(m, lead)` where `lead` is the leg's strongest
/// muscle. Returns whether anything changed.
fn arm(
    c: &mut Creature,
    leg: &[usize],
    reset: impl Fn(&crate::evolution::Muscle, &crate::evolution::Muscle) -> f32,
) -> bool {
    let Some(lead) = strongest(c, &active_on(c, leg)).map(|i| c.muscles[i]) else {
        return false;
    };
    let f = foot(c, leg);
    let mut changed = false;
    for i in sensing(c, leg) {
        let Some(sensor) = sensor_at(c, i, f) else {
            continue;
        };
        let r = reset(&c.muscles[i], &lead).rem_euclid(1.0);
        changed |= (c.muscles[i].sensor, c.muscles[i].reset) != (sensor, r);
        c.muscles[i].sensor = sensor;
        c.muscles[i].reset = r;
    }
    changed
}

/// Reset that puts a muscle where it is when the leg's lead starts a stroke.
fn at_stroke_start(m: &crate::evolution::Muscle, lead: &crate::evolution::Muscle) -> f32 {
    m.phase - lead.phase
}

/// Moves the phase and reset of every muscle on the leg so that its strongest
/// muscle lands on phase `target`. Returns whether it moved.
fn retime(c: &mut Creature, leg: &[usize], target: f32) -> bool {
    let Some(lead) = strongest(c, &active_on(c, leg)).map(|i| c.muscles[i].phase) else {
        return false;
    };
    let shift = (target - lead + 0.5).rem_euclid(1.0) - 0.5;
    if shift.abs() < 1.0e-4 {
        return false;
    }
    for i in muscles_on(c, leg, false) {
        let m = &mut c.muscles[i];
        m.phase = (m.phase + shift).rem_euclid(1.0);
        if m.sensor != NO_SENSOR {
            m.reset = (m.reset + shift).rem_euclid(1.0);
        }
    }
    true
}

/// The phase of the leg's strongest muscle.
fn lead_phase(c: &Creature, leg: &[usize]) -> Option<f32> {
    strongest(c, &active_on(c, leg)).map(|i| c.muscles[i].phase)
}

/// Adds a short, gentle muscle from the foot bone of `leg` to `to_bone`, which
/// senses the foot and fires when it lands. It keeps the leg's clock and
/// phase, so the clock and the landing agree. Returns whether it was added.
fn bridge(c: &mut Creature, cfg: &Config, leg: &[usize], to_bone: usize, rng: &mut Rng) -> bool {
    let tip = leg[leg.len() - 1];
    if to_bone == tip || !room(c, cfg, 0, 1) {
        return false;
    }
    let Some(lead) = strongest(c, &active_on(c, leg)).map(|i| c.muscles[i]) else {
        return false;
    };
    let joined = c.muscles.iter().any(|m| {
        let ends = (m.bone_a as usize, m.bone_b as usize);
        (ends == (tip, to_bone) || ends == (to_bone, tip)) && m.long > m.short
    });
    if joined {
        return false;
    }
    let anchors = (rng.range(0.5, 0.95), rng.range(0.3, 0.7));
    let mut m = new_muscle(c, tip, to_bone, anchors, Some(&lead), rng);
    let length = span(c, &m).max(0.05);
    if length > 1.2 {
        return false;
    }
    m.short = (length * 0.85).max(0.01);
    m.long = length * 1.05;
    m.stiffness = (lead.stiffness * 0.7).clamp(1.0, 120.0);
    m.duty = rng.range(0.15, 0.35);
    m.tendon = 0.0;
    m.sensor = 1;
    m.reset = 0.0;
    c.muscles.push(m);
    true
}

/// The leg nearest to `leg` by foot position, other than itself.
fn neighbour(c: &Creature, legs: &Limbs, k: usize) -> Option<usize> {
    let x = tip_x(c, &legs[k]);
    (0..legs.len()).filter(|&j| j != k).min_by(|&p, &q| {
        (tip_x(c, &legs[p]) - x)
            .abs()
            .total_cmp(&(tip_x(c, &legs[q]) - x).abs())
    })
}

/// A leg whose foot a muscle can sense gets a landing reflex: every muscle
/// that sees the foot restarts at the beginning of its stroke when the foot
/// lands, so a landing starts the stance push at once (Cruse: touchdown
/// triggers stance). The leg keeps its timing relative to its strongest
/// muscle.
fn landing_starts_stroke(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick(&sensing_legs(c), rng) else {
        return false;
    };
    arm(c, &leg, at_stroke_start)
}

/// A leg that senses its foot restarts its muscles close to the end of their
/// contraction when the foot lands, so the stance is cut short and the foot
/// lifts soon after loading. It gives a quick, light step as in running
/// animals, where stance is short (Alexander's duty factor below one half).
fn quick_lift_reflex(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick(&sensing_legs(c), rng) else {
        return false;
    };
    let early = rng.range(0.05, 0.15);
    arm(c, &leg, |m, _| m.duty - early)
}

/// A small muscle runs from the foot bone of one leg to the hip bone of the
/// nearest other leg and fires when the first foot lands, so the landing of
/// one leg kicks the swing or push of the other directly through the body
/// (Cruse's rule that a leg's touchdown triggers its neighbour).
fn cross_leg_trigger(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = legs_by_x(c);
    if legs.len() < 2 {
        return false;
    }
    let k = rng.index(legs.len());
    let Some(j) = neighbour(c, &legs, k) else {
        return false;
    };
    bridge(c, cfg, &legs[k], legs[j][0], rng)
}

/// The next leg back is put half a cycle behind a leg, and the front leg's
/// landing triggers it through a bridge muscle: the hind leg steps after the
/// fore leg touches down, as in the walking cat and dog where the hind foot
/// lands where the fore foot just was.
fn fore_to_hind_trigger(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = legs_by_x(c);
    if legs.len() < 2 {
        return false;
    }
    let k = 1 + rng.index(legs.len() - 1);
    let Some(phase) = lead_phase(c, &legs[k]) else {
        return false;
    };
    let moved = retime(c, &legs[k - 1], phase + 0.5);
    bridge(c, cfg, &legs[k], legs[k - 1][0], rng) | moved
}

/// Two neighbouring legs trigger each other: each one's landing fires a
/// bridge muscle on the other, and the second leg is put half a cycle after
/// the first. It is the half-centre pair of a central pattern generator with
/// the coupling carried by the feet, so the two legs alternate and keep to it.
fn mutual_leg_trigger(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = legs_by_x(c);
    if legs.len() < 2 {
        return false;
    }
    let k = rng.index(legs.len() - 1);
    let Some(phase) = lead_phase(c, &legs[k]) else {
        return false;
    };
    let moved = retime(c, &legs[k + 1], phase + 0.5);
    let forward = bridge(c, cfg, &legs[k], legs[k + 1][0], rng);
    let back = bridge(c, cfg, &legs[k + 1], legs[k][0], rng);
    moved | forward | back
}

/// Every leg that can sense its foot gets a landing reflex, and the legs are
/// put on a travelling wave from the foremost back: each leg's stroke starts
/// an equal share of a cycle (a quarter, or one over the leg count) after the
/// one in front. It is the metachronal wave of a many-legged walker, held
/// together by reflexes.
fn reflex_wave_along_legs(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = legs_by_x(c);
    if legs.len() < 2 {
        return false;
    }
    let step = if rng.unit() < 0.5 {
        0.25
    } else {
        1.0 / legs.len() as f32
    };
    let Some(base) = lead_phase(c, &legs[legs.len() - 1]) else {
        return false;
    };
    let mut changed = false;
    for (rank, leg) in legs.iter().rev().enumerate() {
        changed |= retime(c, leg, base + step * rank as f32);
        changed |= arm(c, leg, at_stroke_start);
    }
    changed
}

/// Legs take turns: counted along the body, every other leg is put half a
/// cycle behind the first, and each gets a landing reflex that restarts its
/// stroke. Neighbours then alternate as in a trot or a walk (the contralateral
/// half-cycle rule of Cruse and of Sims' mirrored limbs), and the reflex holds
/// the alternation when the body speeds up or slows down.
fn alternate_legs_with_reflex(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = legs_by_x(c);
    if legs.len() < 2 {
        return false;
    }
    let first = rng.index(2);
    let Some(base) = lead_phase(c, &legs[first]) else {
        return false;
    };
    let mut changed = false;
    for (rank, leg) in legs.iter().enumerate() {
        let offset = if (rank + first).is_multiple_of(2) {
            0.0
        } else {
            0.5
        };
        changed |= retime(c, leg, base + offset);
        changed |= arm(c, leg, at_stroke_start);
    }
    changed
}

/// The legs split into a rear group and a front group. Legs of one group
/// step together and the groups are a half, a quarter or three quarters of a
/// cycle apart, each leg with a landing reflex. It is a bound (half) or a
/// gallop (quarter) as in galloping horses, where fore pair and hind pair
/// each act as one.
fn fore_hind_pairing(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = legs_by_x(c);
    if legs.len() < 2 {
        return false;
    }
    let half = legs.len() / 2;
    let offset = [0.5, 0.25, 0.75][rng.index(3)];
    let Some(base) = lead_phase(c, &legs[legs.len() - 1]) else {
        return false;
    };
    let mut changed = false;
    for (rank, leg) in legs.iter().enumerate() {
        let target = if rank < half { base + offset } else { base };
        changed |= retime(c, leg, target);
        changed |= arm(c, leg, at_stroke_start);
    }
    changed
}

/// A leg's landing fires a small muscle between its foot bone and the nearest
/// bone of the trunk, which flexes the spine at each footfall. The impact
/// then carries into the back, the way the spine of a galloping cheetah
/// gathers and extends with each stride.
fn landing_flexes_trunk(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = legs_by_x(c);
    let mut in_leg = [false; crate::evolution::MAX_NODES];
    for leg in &legs {
        for &b in leg {
            in_leg[b] = true;
        }
    }
    let trunk: BoneIds = (0..c.bones.len())
        .filter(|&b| !in_leg[b] && !is_neck(c, b))
        .collect();
    let Some(leg) = pick(&legs, rng) else {
        return false;
    };
    let x = tip_x(c, &leg);
    let mid =
        |b: usize| (c.nodes[c.bones[b].a as usize].x + c.nodes[c.bones[b].b as usize].x) * 0.5;
    let Some(bone) = trunk
        .iter()
        .copied()
        .min_by(|&p, &q| (mid(p) - x).abs().total_cmp(&(mid(q) - x).abs()))
    else {
        return false;
    };
    bridge(c, cfg, &leg, bone, rng)
}

/// A second muscle across the last joint of a leg (foot bone to the bone
/// above) has a spring tendon, a long contraction and a landing reflex. It
/// extends the leg through stance and stores the landing in its tendon, the
/// way a limb stiffens when it takes load (Full and Koditschek's spring-loaded
/// leg template).
fn touchdown_stiffener(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick(&legs_by_x(c), rng) else {
        return false;
    };
    let (tip, above) = (leg[leg.len() - 1], leg[leg.len() - 2]);
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let Some(lead) = strongest(c, &active_on(c, &leg)).map(|i| c.muscles[i]) else {
        return false;
    };
    let anchors = (rng.range(0.3, 0.7), rng.range(0.4, 0.8));
    let mut m = new_muscle(c, tip, above, anchors, Some(&lead), rng);
    let length = span(c, &m).max(0.05);
    m.short = (length * 0.8).max(0.01);
    m.long = length * 1.1;
    m.duty = rng.range(0.5, 0.7);
    m.stiffness = (lead.stiffness * 1.3).clamp(1.0, 120.0);
    m.tendon = rng.range(0.4, 0.8);
    m.sensor = 1;
    m.reset = 0.0;
    c.muscles.push(m);
    true
}

/// A leg that already has a landing reflex passes its rule to every leg that
/// has none. Each target muscle restarts at the same distance from its own
/// phase as the source muscle does, so all legs follow one rule (Cruse's
/// walknet uses one rule set for every leg) while keeping their own timing.
fn spread_leg_reflex(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = sensing_legs(c);
    let armed = |c: &Creature, l: &BoneIds| {
        sensing(c, l)
            .iter()
            .any(|&i| c.muscles[i].sensor != NO_SENSOR)
    };
    let sources: Limbs = legs.iter().filter(|l| armed(c, l)).cloned().collect();
    let Some(source) = pick(&sources, rng) else {
        return false;
    };
    let rule: MuscleIds = sensing(c, &source)
        .into_iter()
        .filter(|&i| c.muscles[i].sensor != NO_SENSOR)
        .collect();
    let mut changed = false;
    let targets: Limbs = legs.iter().filter(|l| !armed(c, l)).cloned().collect();
    for leg in &targets {
        let f = foot(c, leg);
        for (k, i) in sensing(c, leg).into_iter().enumerate() {
            let from = c.muscles[rule[k % rule.len()]];
            let Some(sensor) = sensor_at(c, i, f) else {
                continue;
            };
            let m = &mut c.muscles[i];
            m.sensor = sensor;
            m.reset = (m.phase + from.reset - from.phase).rem_euclid(1.0);
            changed = true;
        }
    }
    changed
}

/// The legs from the front back are linked in a chain: each leg's landing
/// fires a bridge muscle on the next leg back, and each is put half a cycle
/// after the one before it (up to three links). A step then runs down the
/// body as one reflex chain, as in walking insects, where each leg's
/// placement cues the leg behind it.
fn trigger_chain_along_legs(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = legs_by_x(c);
    if legs.len() < 2 {
        return false;
    }
    let Some(mut phase) = lead_phase(c, &legs[legs.len() - 1]) else {
        return false;
    };
    let mut changed = false;
    for k in (1..legs.len()).rev().take(3) {
        phase += 0.5;
        changed |= retime(c, &legs[k - 1], phase);
        changed |= bridge(c, cfg, &legs[k], legs[k - 1][0], rng);
    }
    changed
}

/// A leg is set to a walking duty factor (0.6 to 0.75, long stance with
/// overlap between legs) or a running one (0.3 to 0.4, short stance), keeping
/// the middle of each contraction, and gets a landing reflex that starts the
/// stroke. Alexander's duty factor separates walking from running, and the
/// reflex keeps the stance beginning at touchdown whatever the duty.
fn stance_duty_with_reflex(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(leg) = pick(&sensing_legs(c), rng) else {
        return false;
    };
    let duty = if rng.unit() < 0.5 {
        rng.range(0.6, 0.75)
    } else {
        rng.range(0.3, 0.4)
    };
    let mut changed = false;
    for i in active_on(c, &leg) {
        let m = &mut c.muscles[i];
        let phase = (m.phase + 0.5 * (duty - m.duty)).rem_euclid(1.0);
        changed |= duty != m.duty;
        m.phase = phase;
        m.duty = duty;
    }
    changed | arm(c, &leg, at_stroke_start)
}
