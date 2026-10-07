//! Gait operators: symmetry and repetition: pairs, mirrored halves, repeated segments.
//!
//! The operators of this file share one pick slot and are compound: each is a
//! whole, coherent change to the body, and its child gets no parameter noise.
//!
//! The sources are Sims (1994, symmetric limb pairs with mirrored timing),
//! Lipson and Pollack (2000, legs as repeated rigid bars), Cheney et al.
//! (2013, regular and symmetric bodies travel further), Hornby and Pollack
//! (2001, repeated parts) and the central pattern generator view of gait
//! (one oscillator program run by every segment, each shifted in phase: a
//! walk is alternation, a metachronal wave is a constant lag per segment).
//! Animals repeat one leg design along the body and run it under a shared
//! program, so these operators move a body toward that form in one step.
use super::compound::{close_ring, limb_phase, scale_bones, shed_tips};
use super::extra::{drive, limb_drive};
use super::junctions::{add, add_node, keep_strokes, pos, shift_branch, spans, sub};
use super::limbs::{clamped, pick};
use super::muscles::actuation;
use super::rhythm::{foot, hip, leaf_limbs, leaf_limbs_at, tip_x};
use super::{
    BoneIds, Context, Operator, branch, branch_nodes, child_bones, copy_branch_limited, fit_stroke,
    is_neck, parent_bones, remove_parts, room,
};
use crate::config::Config;
use crate::evolution::{Bone, Bounded, Creature, MAX_NODES, Muscle, Rng, max_bone_length};

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[
    ("clone_best_leg", clone_best_leg),
    ("mirror_body_halves", mirror_body_halves),
    ("repeat_equal_segment", repeat_equal_segment),
    ("repeat_segment_mirrored", repeat_segment_mirrored),
    ("equalize_leg_reach", equalize_leg_reach),
    ("average_leg_pair", average_leg_pair),
    ("share_program_alternating", share_program_alternating),
    ("share_program_wave", share_program_wave),
    ("twin_leg_antiphase", twin_leg_antiphase),
    ("mirror_hip_position", mirror_hip_position),
    ("step_leg_along_trunk", step_leg_along_trunk),
    ("share_joint_ranges", share_joint_ranges),
    ("copy_foot_to_all_legs", copy_foot_to_all_legs),
];

// Helpers.

/// The horizontal x distance from a leg's hip to its tip (negative: behind the hip).
fn reach_x(c: &Creature, limb: &[usize]) -> f32 {
    tip_x(c, limb) - c.nodes[hip(c, limb)].x
}

/// Whether two legs point to opposite sides (one reaches forward, one backward).
fn opposed(c: &Creature, x: &[usize], y: &[usize]) -> bool {
    reach_x(c, x) * reach_x(c, y) < -1.0e-4
}

/// The index in `legs` of the leg with the most drive, if it has any.
fn best_leg(c: &Creature, legs: &[BoneIds]) -> Option<usize> {
    let best = (0..legs.len())
        .max_by(|&x, &y| limb_drive(c, &legs[x]).total_cmp(&limb_drive(c, &legs[y])))?;
    (limb_drive(c, &legs[best]) > 0.0).then_some(best)
}

/// Whether leg `y` has the same structure as leg `x`, with optional mirroring:
/// matching node offsets from the hip and matching joint ranges.
fn same_pose(c: &Creature, x: &[usize], y: &[usize], mirror: bool) -> bool {
    if x.len() != y.len() {
        return false;
    }
    let (hx, hy) = (pos(c, hip(c, x)), pos(c, hip(c, y)));
    x.iter().zip(y.iter()).all(|(&p, &q)| {
        let (bp, bq) = (c.bones[p], c.bones[q]);
        let (np, nq) = (pos(c, bp.b as usize), pos(c, bq.b as usize));
        let dx = if mirror {
            -(np[0] - hx[0])
        } else {
            np[0] - hx[0]
        };
        let (lo, hi) = if mirror {
            (-bp.max_angle, -bp.min_angle)
        } else {
            (bp.min_angle, bp.max_angle)
        };
        (dx - (nq[0] - hy[0])).abs() < 0.03
            && ((np[1] - hx[1]) - (nq[1] - hy[1])).abs() < 0.03
            && (lo - bq.min_angle).abs() < 0.05
            && (hi - bq.max_angle).abs() < 0.05
    })
}

/// Instructions to replace a target leg with a copy of a source leg.
#[derive(Clone, Copy)]
struct Job {
    source: usize,
    target: usize,
    mirror: bool,
    phase: f32,
}

/// Replaces each job's target leg by a copy of its source leg at the
/// target's hip (reflected with `mirror`, muscles `phase` of a cycle later),
/// then closes the ring. Jobs the body has no room for are dropped from the
/// end. Returns whether the creature changed.
fn replant(c: &mut Creature, cfg: &Config, jobs: &[Job], rng: &mut Rng) -> bool {
    let mut next = c.clone();
    let mut gone_bones = BoneIds::new();
    let mut gone_nodes = BoneIds::new();
    let mut done = 0;
    for job in jobs {
        let at = c.bones[job.target].a as usize;
        let from = pos(c, c.bones[job.source].a as usize);
        let to = pos(c, at);
        let place = |p: [f32; 2]| {
            let dx = if job.mirror {
                from[0] - p[0]
            } else {
                p[0] - from[0]
            };
            [to[0] + dx, to[1] + p[1] - from[1]]
        };
        if copy_branch_limited(
            &mut next,
            cfg,
            job.source,
            at,
            place,
            job.mirror,
            job.phase,
            usize::MAX,
        )
        .is_none()
        {
            break;
        }
        let old = branch(c, job.target);
        gone_nodes.extend(branch_nodes(c, &old));
        gone_bones.extend(old);
        done += 1;
    }
    if done == 0 {
        return false;
    }
    remove_parts(&mut next, &gone_bones, &gone_nodes);
    if next.nodes.len() > c.nodes.len() + 2 || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// The x middle, span, and left boundary of the body's nodes.
fn extent(c: &Creature) -> (f32, f32, f32) {
    let (lo, hi) = c.nodes.iter().fold((f32::MAX, f32::MIN), |(lo, hi), n| {
        (lo.min(n.x), hi.max(n.x))
    });
    (0.5 * (lo + hi), (hi - lo).max(0.2), lo)
}

// Operators.

/// Makes every other leg a copy of the best one, the leg with the most muscle
/// drive. Each leg keeps its hip and its own place in the stride (its lead
/// muscle keeps its phase), and a leg that points the other way gets the
/// mirror image. A body whose legs all share one design is what Sims' and
/// Cheney's regular creatures were, and the best leg's design is proven by
/// the distance the body already covers.
pub(crate) fn clone_best_leg(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let legs = leaf_limbs(c);
    if legs.len() < 2 {
        return false;
    }
    let Some(best) = best_leg(c, &legs) else {
        return false;
    };
    let source = &legs[best];
    let source_phase = limb_phase(c, source);
    let mut jobs: Bounded<Job, MAX_NODES> = Bounded::new();
    for (i, leg) in legs.iter().enumerate() {
        if i == best {
            continue;
        }
        let mirror = opposed(c, source, leg);
        if same_pose(c, source, leg, mirror) {
            continue;
        }
        let phase = match (limb_phase(c, leg), source_phase) {
            (Some(own), Some(lead)) => own - lead,
            _ => 0.0,
        };
        jobs.push(Job {
            source: source[0],
            target: leg[0],
            mirror,
            phase,
        });
    }
    replant(c, cfg, &jobs, rng)
}

/// Makes the half of the body on one side of its middle the mirror image of
/// the other half: each leg of the source half is reflected onto the leg of
/// the other half that lies nearest to its mirrored hip position, and that
/// leg runs half a cycle later (a trot) or with it (a bound). Fore and hind
/// limbs of a mammal are close to such mirror images of each other, which
/// is why a cheetah and a horse can turn a single bone plan into two
/// ends of one gait (Alexander's gait work).
pub(crate) fn mirror_body_halves(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leaf_limbs(c);
    let (centre, span, _) = extent(c);
    let margin = 0.04 * span;
    let side = |l: &[usize]| c.nodes[hip(c, l)].x - centre;
    let from_front = rng.unit() < 0.5;
    let lag = [0.5, 0.5, 0.0][rng.index(3)];
    let mut taken = [false; MAX_NODES];
    let mut jobs: Bounded<Job, MAX_NODES> = Bounded::new();
    for source in legs.iter() {
        let sx = side(source);
        if (from_front && sx <= margin) || (!from_front && sx >= -margin) {
            continue;
        }
        let target = (0..legs.len())
            .filter(|&t| {
                let tx = side(&legs[t]);
                !taken[t] && tx.abs() > margin && (tx > 0.0) != (sx > 0.0)
            })
            .min_by(|&p, &q| {
                let d = |t: usize| (side(&legs[t]) + sx).abs();
                d(p).total_cmp(&d(q))
            });
        let Some(t) = target else {
            continue;
        };
        if (side(&legs[t]) + sx).abs() > 0.4 * span {
            continue;
        }
        taken[t] = true;
        jobs.push(Job {
            source: source[0],
            target: legs[t][0],
            mirror: true,
            phase: lag,
        });
    }
    replant(c, cfg, &jobs, rng)
}

/// Copies the trunk bone `trunk` with the leg limbs on its child node and
/// inserts the copy after it, the same size as the original. The copy's
/// muscles run `phase` of a cycle later, and with `mirror` its limbs are
/// reflected about the trunk joint. At most three muscles cross each joint
/// copy. Returns the copy's bone.
fn repeat_segment(
    c: &mut Creature,
    cfg: &Config,
    trunk: usize,
    phase: f32,
    mirror: bool,
) -> Option<usize> {
    let children = child_bones(c);
    let parents = parent_bones(c);
    let (a, b) = (c.bones[trunk].a as usize, c.bones[trunk].b as usize);
    let above = parents[a]?;
    let limbs = leaf_limbs_at(c, &children, b);
    if limbs.is_empty() || is_neck(c, trunk) {
        return None;
    }
    let length = c.bones[trunk].rest_length.clamp(0.03, max_bone_length());
    let mut joint: Bounded<usize, 16> = (0..c.muscles.len())
        .filter(|&i| {
            let ends = (c.muscles[i].bone_a as usize, c.muscles[i].bone_b as usize);
            ends == (trunk, above) || ends == (above, trunk)
        })
        .take(16)
        .collect();
    joint.sort_stable_by(|&x, &y| drive(&c.muscles[y]).total_cmp(&drive(&c.muscles[x])));
    joint.truncate(3);
    let bones: usize = 1 + limbs.iter().map(|&l| branch(c, l).len()).sum::<usize>();
    if !room(c, cfg, bones, joint.len()) {
        return None;
    }
    let offset = sub(pos(c, b), pos(c, a));
    let before = spans(c);
    let node = add_node(c, b, add(pos(c, b), offset));
    let copy = c.bones.len();
    c.bones.push(Bone {
        a: b as u32,
        b: node as u32,
        rest_length: length,
        organ_mass: 0.0,
        organ_at: 0.5,
        ..c.bones[trunk]
    });
    for &i in &joint {
        let m = c.muscles[i];
        let map = |x: u32| (if x as usize == trunk { copy } else { trunk }) as u32;
        let mut new = Muscle {
            bone_a: map(m.bone_a),
            bone_b: map(m.bone_b),
            phase: (m.phase + phase).rem_euclid(1.0),
            reset: (m.reset + phase).rem_euclid(1.0),
            ..m
        };
        fit_stroke(c, &mut new, Some(&m));
        c.muscles.push(new);
    }
    // The rest of the body below the trunk hangs from the copy.
    for &child in &children[b] {
        if limbs.contains(&child) {
            continue;
        }
        c.bones[child].a = node as u32;
        shift_branch(c, child, offset);
        for m in &mut c.muscles {
            if m.bone_a as usize == trunk && m.bone_b as usize == child {
                m.bone_a = copy as u32;
            }
            if m.bone_b as usize == trunk && m.bone_a as usize == child {
                m.bone_b = copy as u32;
            }
        }
    }
    let (root, top) = (pos(c, b), pos(c, node));
    for &limb in &limbs {
        let place = |p: [f32; 2]| {
            let d = sub(p, root);
            add(top, if mirror { [-d[0], d[1]] } else { d })
        };
        copy_branch_limited(c, cfg, limb, node, place, mirror, phase, 3)?;
    }
    keep_strokes(c, &before);
    Some(copy)
}

/// Shared body of the two segment repeaters: picks a trunk bone that carries
/// legs, repeats it once at the same size, and gives idle tips back.
fn repeat_one(c: &mut Creature, cfg: &Config, rng: &mut Rng, phase: f32, mirror: bool) -> bool {
    let children = child_bones(c);
    let trunks: BoneIds = (0..c.bones.len())
        .filter(|&j| {
            !is_neck(c, j) && !leaf_limbs_at(c, &children, c.bones[j].b as usize).is_empty()
        })
        .collect();
    let Some(trunk) = pick(&trunks, rng) else {
        return false;
    };
    let mut next = c.clone();
    if repeat_segment(&mut next, cfg, trunk, phase, mirror).is_none() {
        return false;
    }
    let added = next.nodes.len() - c.nodes.len();
    let shed = shed_tips(&mut next, c.nodes.len(), added, rng);
    if shed + 1 < added || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Repeats a trunk segment with its legs once, the copy the same size as the
/// original and its muscles half a cycle later: a body of identical
/// segments whose neighbours alternate, as in a centipede or the legs of a
/// walking insect, which the central pattern generator literature describes
/// as one oscillator per segment coupled in antiphase.
pub(crate) fn repeat_equal_segment(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    repeat_one(c, cfg, rng, 0.5, false)
}

/// Repeats a trunk segment with its legs once as a mirror image: the legs of
/// the copy lean the opposite way and their joint ranges are reflected, so
/// the two segments push toward and away from each other like the fore and
/// hind limbs of a quadruped. The copy runs with the original or half a
/// cycle later.
pub(crate) fn repeat_segment_mirrored(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let phase = [0.0, 0.5][rng.index(2)];
    repeat_one(c, cfg, rng, phase, true)
}

/// The total length of a leg's bones.
fn reach(c: &Creature, limb: &[usize]) -> f32 {
    limb.iter().map(|&b| c.bones[b].rest_length).sum()
}

/// Makes two legs of unequal reach equal: the shorter leg grows to the
/// reach of the longer (six times in ten) or the longer shrinks to the
/// shorter, with every muscle keeping its stroke relative to its span. Legs
/// of equal length stride in step, and a leg that is much shorter than its
/// partner leaves a limp.
pub(crate) fn equalize_leg_reach(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leaf_limbs(c);
    if legs.len() < 2 {
        return false;
    }
    let first = rng.index(legs.len());
    let near: BoneIds = (0..legs.len())
        .filter(|&j| {
            let (p, q) = (reach(c, &legs[first]), reach(c, &legs[j]));
            j != first && p.max(q) / p.min(q).max(0.01) > 1.06 && p.max(q) <= 2.0 * p.min(q)
        })
        .collect();
    let Some(second) = pick(&near, rng) else {
        return false;
    };
    let (short, long) = if reach(c, &legs[first]) < reach(c, &legs[second]) {
        (first, second)
    } else {
        (second, first)
    };
    let ratio = reach(c, &legs[long]) / reach(c, &legs[short]);
    let before = spans(c);
    let changed = if rng.unit() < 0.6 {
        scale_bones(c, &legs[short], ratio)
    } else {
        scale_bones(c, &legs[long], 1.0 / ratio)
    };
    if changed {
        keep_strokes(c, &before);
    }
    changed
}

/// A leg as bone lengths and turning angles: each bone's length and its
/// angle relative to the direction of the bone before it.
fn polar(c: &Creature, limb: &[usize]) -> (Bounded<f32, MAX_NODES>, Bounded<f32, MAX_NODES>) {
    let (mut lengths, mut turns) = (Bounded::new(), Bounded::new());
    let mut before = 0.0f32;
    for &b in limb {
        let (from, to) = (pos(c, c.bones[b].a as usize), pos(c, c.bones[b].b as usize));
        let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
        let angle = dx.atan2(-dy);
        lengths.push(dx.hypot(dy));
        turns.push(
            (angle - before + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI,
        );
        before = angle;
    }
    (lengths, turns)
}

/// Sets a leg to the given lengths and turning angles (reflected with
/// `flip`), and its joint ranges (`ranges`, reflected the same way).
fn pose_leg(
    c: &mut Creature,
    limb: &[usize],
    lengths: &[f32],
    turns: &[f32],
    ranges: &[(f32, f32)],
    flip: bool,
) {
    let mut at = pos(c, hip(c, limb));
    let mut angle = 0.0f32;
    for (k, &b) in limb.iter().enumerate() {
        angle += if flip { -turns[k] } else { turns[k] };
        let length = lengths[k].clamp(0.03, max_bone_length());
        at = clamped(at[0] + length * angle.sin(), at[1] - length * angle.cos());
        let node = c.bones[b].b as usize;
        [c.nodes[node].x, c.nodes[node].y] = at;
        c.bones[b].rest_length = length;
        (c.bones[b].min_angle, c.bones[b].max_angle) = if flip {
            (-ranges[k].1, -ranges[k].0)
        } else {
            ranges[k]
        };
    }
}

/// Replaces the shapes of two legs with the average of their shapes: each
/// bone gets the mean length, the mean turn from the bone before it and the
/// mean joint range (a leg that points the other way is mirrored first), and
/// the foot nodes the mean size and friction. Both legs end up the same
/// shape, one the mirror image of the other if they point apart. The
/// muscles keep their timing and their strokes follow the new spans.
pub(crate) fn average_leg_pair(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leaf_limbs(c);
    if legs.len() < 2 {
        return false;
    }
    let first = rng.index(legs.len());
    let same: BoneIds = (0..legs.len())
        .filter(|&j| j != first && legs[j].len() == legs[first].len())
        .collect();
    let Some(second) = pick(&same, rng) else {
        return false;
    };
    let (x, y) = (&legs[first], &legs[second]);
    let flip = opposed(c, x, y);
    let ((lx, tx), (ly, ty)) = (polar(c, x), polar(c, y));
    let mut lengths: Bounded<f32, MAX_NODES> = Bounded::new();
    let mut turns: Bounded<f32, MAX_NODES> = Bounded::new();
    let mut ranges: Bounded<(f32, f32), MAX_NODES> = Bounded::new();
    let mut moved = 0.0f32;
    for k in 0..x.len() {
        let (p, q) = (c.bones[x[k]], c.bones[y[k]]);
        let other_turn = if flip { -ty[k] } else { ty[k] };
        let other_range = if flip {
            (-q.max_angle, -q.min_angle)
        } else {
            (q.min_angle, q.max_angle)
        };
        lengths.push(0.5 * (lx[k] + ly[k]));
        turns.push(0.5 * (tx[k] + other_turn));
        ranges.push((
            0.5 * (p.min_angle + other_range.0),
            0.5 * (p.max_angle + other_range.1),
        ));
        moved += (lx[k] - ly[k]).abs() + (tx[k] - other_turn).abs();
    }
    if moved < 0.05 {
        return false;
    }
    let before = spans(c);
    pose_leg(c, x, &lengths, &turns, &ranges, false);
    pose_leg(c, y, &lengths, &turns, &ranges, flip);
    for k in 0..x.len() {
        let (p, q) = (c.bones[x[k]].b as usize, c.bones[y[k]].b as usize);
        let diameter = 0.5 * (c.nodes[p].diameter + c.nodes[q].diameter);
        let friction = 0.5 * (c.nodes[p].friction + c.nodes[q].friction);
        (c.nodes[p].diameter, c.nodes[q].diameter) = (diameter, diameter);
        (c.nodes[p].friction, c.nodes[q].friction) = (friction, friction);
    }
    keep_strokes(c, &before);
    true
}

/// Copies the best leg's muscle program (period, duty, stiffness, tendon,
/// sensor, stroke) to every leg with the same bone count, each shifted in phase
/// by `shift(rank)`, where rank counts from the back and is zero for the best.
/// Returns whether anything changed.
fn share_program(c: &mut Creature, shift: impl Fn(i32, usize) -> Option<f32>) -> bool {
    let legs = leaf_limbs(c);
    let Some(best) = best_leg(c, &legs) else {
        return false;
    };
    let mut group: BoneIds = (0..legs.len())
        .filter(|&i| legs[i].len() == legs[best].len())
        .collect();
    let tip = |i: usize| tip_x(c, &legs[i]);
    group.sort_stable_by(|&p, &q| tip(p).total_cmp(&tip(q)));
    let n = group.len();
    let rank_best = group.iter().position(|&i| i == best).unwrap_or(0) as i32;
    let (_, source_muscles) = actuation(c, legs[best][0]);
    let key = |limb: &[usize], m: &Muscle| {
        let at = |b: u32| {
            limb.iter()
                .position(|&x| x == b as usize)
                .unwrap_or(usize::MAX)
        };
        (at(m.bone_a), at(m.bone_b))
    };
    let mut changed = false;
    for (rank, &i) in group.iter().enumerate() {
        if i == best {
            continue;
        }
        let Some(lag) = shift(rank as i32 - rank_best, n) else {
            return false;
        };
        let (_, muscles) = actuation(c, legs[i][0]);
        for t in muscles {
            let old = c.muscles[t];
            let wanted = key(&legs[i], &old);
            let Some(&s) = source_muscles
                .iter()
                .find(|&&s| key(&legs[best], &c.muscles[s]) == wanted)
            else {
                continue;
            };
            let src = c.muscles[s];
            let mut m = Muscle {
                period: src.period,
                duty: src.duty,
                stiffness: src.stiffness,
                tendon: src.tendon,
                sensor: src.sensor,
                phase: (src.phase + lag).rem_euclid(1.0),
                reset: (src.reset + lag).rem_euclid(1.0),
                ..old
            };
            fit_stroke(c, &mut m, Some(&src));
            changed |= m != old;
            c.muscles[t] = m;
        }
    }
    changed
}

/// Makes every leg with the best leg's number of bones run the best leg's
/// muscle program, alternating along the body: the neighbours of the best leg
/// run half a cycle later, their neighbours with it again. Legs that share one
/// program differ only by phase, as the segments of a walking animal do under
/// one central pattern generator (Full and Koditschek's template idea), so a
/// mutation that improves one leg's stroke improves all of them.
pub(crate) fn share_program_alternating(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    share_program(c, |distance, _| Some(0.5 * distance.rem_euclid(2) as f32))
}

/// Makes every leg with the best leg's number of bones run the best leg's
/// muscle program with a constant lag of one over the leg count per place
/// along the body, forwards or backwards: a metachronal wave, the gait of
/// a millipede and of the swimmerets of a crayfish. It needs three legs.
pub(crate) fn share_program_wave(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let sign = if rng.unit() < 0.5 { 1.0 } else { -1.0 };
    share_program(c, |distance, n| {
        (n >= 3).then(|| sign * distance as f32 / n as f32)
    })
}

/// Hangs a copy of a leg from the same hip in the same pose, its muscles half
/// a cycle later: the left and right leg of a pair seen from the side, which
/// share a place and alternate. `twin_limb` makes the copy in phase, so the
/// two legs move as one. Idle tips go back to keep the body from growing.
pub(crate) fn twin_leg_antiphase(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut legs = leaf_limbs(c);
    legs.retain(|l| l.len() <= 3 && hip(c, l) != 0 && room(c, cfg, l.len(), 0));
    let Some(leg) = pick(&legs, rng) else {
        return false;
    };
    let mut next = c.clone();
    let at = hip(c, &leg);
    if copy_branch_limited(&mut next, cfg, leg[0], at, |p| p, false, 0.5, usize::MAX).is_none() {
        return false;
    }
    let added = next.nodes.len() - c.nodes.len();
    let shed = shed_tips(&mut next, c.nodes.len(), added, rng);
    if shed + 1 < added || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Moves a leg's hip to the trunk node nearest the mirror image of another
/// leg's hip about the middle of the body, when no leg hangs there yet and the
/// moved leg is on the same side as the other: the legs of the body come in
/// mirrored positions, as hind and fore limbs do. The leg keeps its shape and
/// muscles, and the strokes of the muscles across its first bone follow.
pub(crate) fn mirror_hip_position(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leaf_limbs(c);
    if legs.len() < 2 {
        return false;
    }
    let (centre, span, _) = extent(c);
    let hx = |l: &[usize]| c.nodes[hip(c, l)].x;
    let first = rng.index(legs.len());
    let wanted = 2.0 * centre - hx(&legs[first]);
    if (wanted - hx(&legs[first])).abs() < 0.15 * span {
        return false;
    }
    if legs.iter().any(|l| (hx(l) - wanted).abs() < 0.12 * span) {
        return false;
    }
    let same_side: BoneIds = (0..legs.len())
        .filter(|&j| {
            j != first
                && hip(c, &legs[j]) != hip(c, &legs[first])
                && (hx(&legs[j]) - centre) * (hx(&legs[first]) - centre) > 0.0
        })
        .collect();
    let Some(mover) = pick(&same_side, rng) else {
        return false;
    };
    let limb = &legs[mover];
    let root = limb[0];
    let from = hip(c, limb);
    let inside = branch_nodes(c, limb);
    let Some(at) = (1..c.nodes.len())
        .filter(|&n| n != from && !inside.contains(&n))
        .min_by(|&p, &q| {
            (c.nodes[p].x - wanted)
                .abs()
                .total_cmp(&(c.nodes[q].x - wanted).abs())
        })
    else {
        return false;
    };
    if (c.nodes[at].x - wanted).abs() > 0.12 * span || at == from {
        return false;
    }
    let parents = parent_bones(c);
    let before = spans(c);
    let offset = sub(pos(c, at), pos(c, from));
    for n in inside {
        let node = &mut c.nodes[n];
        [node.x, node.y] = clamped(node.x + offset[0], node.y + offset[1]);
    }
    c.bones[root].a = at as u32;
    if let (Some(old), Some(new)) = (parents[from], parents[at]) {
        for m in &mut c.muscles {
            let (x, y) = (m.bone_a as usize, m.bone_b as usize);
            if (x, y) == (root, old) {
                m.bone_b = new as u32;
            } else if (x, y) == (old, root) {
                m.bone_a = new as u32;
            }
        }
    }
    keep_strokes(c, &before);
    true
}

/// Copies a leg of up to three bones to the trunk node one leg spacing away
/// (the distance to its nearest neighbour's hip) forwards or backwards, where
/// no leg hangs yet, in a phase a third, a quarter or half a cycle off: a
/// row of equally spaced, identical legs in a travelling wave, the layout
/// of a many-legged walker. Idle tips go back.
pub(crate) fn step_leg_along_trunk(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leaf_limbs(c);
    let candidates: BoneIds = (0..legs.len())
        .filter(|&i| legs[i].len() <= 3 && hip(c, &legs[i]) != 0 && room(c, cfg, legs[i].len(), 0))
        .collect();
    let Some(i) = pick(&candidates, rng) else {
        return false;
    };
    let leg = &legs[i];
    let (_, span, _) = extent(c);
    let x = c.nodes[hip(c, leg)].x;
    let gap = legs
        .iter()
        .enumerate()
        .filter(|&(j, l)| j != i && hip(c, l) != hip(c, leg))
        .map(|(_, l)| (c.nodes[hip(c, l)].x - x).abs())
        .fold(f32::MAX, f32::min);
    let gap = if gap < f32::MAX {
        gap.max(0.08)
    } else {
        0.35 * span
    };
    let direction = if rng.unit() < 0.5 { 1.0 } else { -1.0 };
    let wanted = x + direction * gap;
    let inside = branch_nodes(c, leg);
    let Some(at) = (1..c.nodes.len())
        .filter(|&n| {
            n != hip(c, leg)
                && !inside.contains(&n)
                && legs.iter().all(|l| !branch_nodes(c, l).contains(&n))
        })
        .min_by(|&p, &q| {
            (c.nodes[p].x - wanted)
                .abs()
                .total_cmp(&(c.nodes[q].x - wanted).abs())
        })
    else {
        return false;
    };
    let nearby = |l: &BoneIds| (c.nodes[hip(c, l)].x - c.nodes[at].x).abs() < 0.4 * gap;
    if (c.nodes[at].x - wanted).abs() > 0.4 * gap || legs.iter().any(nearby) {
        return false;
    }
    let (from, to) = (pos(c, hip(c, leg)), pos(c, at));
    let lag = [1.0 / 3.0, 0.25, 0.5][rng.index(3)] * direction;
    let mut next = c.clone();
    let place = |p: [f32; 2]| add(to, sub(p, from));
    if copy_branch_limited(&mut next, cfg, leg[0], at, place, false, lag, usize::MAX).is_none() {
        return false;
    }
    let added = next.nodes.len() - c.nodes.len();
    let shed = shed_tips(&mut next, c.nodes.len(), added, rng);
    if shed + 1 < added || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Gives every leg with the best leg's number of bones the best leg's joint
/// ranges, bone by bone (reflected for a leg that points the other way). The
/// shape and muscles stay. Legs that can flex the same amount at each joint
/// let one stride suit every leg, as the matched joint ranges of paired
/// limbs do.
pub(crate) fn share_joint_ranges(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leaf_limbs(c);
    let Some(best) = best_leg(c, &legs) else {
        return false;
    };
    let mut changed = false;
    for (i, leg) in legs.iter().enumerate() {
        if i == best || leg.len() != legs[best].len() {
            continue;
        }
        let flip = opposed(c, &legs[best], leg);
        for (&s, &t) in legs[best].iter().zip(leg.iter()) {
            let (lo, hi) = (c.bones[s].min_angle, c.bones[s].max_angle);
            let (lo, hi) = if flip { (-hi, -lo) } else { (lo, hi) };
            let bone = &mut c.bones[t];
            changed |= (bone.min_angle - lo).abs() > 1.0e-4 || (bone.max_angle - hi).abs() > 1.0e-4;
            (bone.min_angle, bone.max_angle) = (lo, hi);
        }
    }
    changed
}

/// Gives the foot of every leg of two bones or more the size and friction of
/// the best leg's foot. A foot that works on one leg grips the same way on
/// the others, and a body whose feet differ at random only has the feet of
/// its worst leg to slip on.
pub(crate) fn copy_foot_to_all_legs(
    c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut legs = leaf_limbs(c);
    legs.retain(|l| l.len() >= 2);
    let Some(best) = best_leg(c, &legs) else {
        return false;
    };
    let source = c.nodes[foot(c, &legs[best])];
    let mut changed = false;
    for (i, leg) in legs.iter().enumerate() {
        if i == best {
            continue;
        }
        let node = &mut c.nodes[c.bones[leg[leg.len() - 1]].b as usize];
        changed |= (node.diameter - source.diameter).abs() > 1.0e-4
            || (node.friction - source.friction).abs() > 1.0e-4;
        (node.diameter, node.friction) = (source.diameter, source.friction);
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::super::tests::bodies;
    use super::*;
    use crate::evolution::repair;

    #[test]
    fn every_symmetry_operator_leaves_a_valid_body() {
        let cfg = Config::default();
        for (name, op) in OPS {
            let mut applied = 0;
            for (i, body) in bodies(&cfg, 120).into_iter().enumerate() {
                let mut c = body.clone();
                let cx = Context::of(None);
                if op(&mut c, &cfg, &mut Rng::new(71, 0, i), &cx) {
                    applied += 1;
                    assert!(c.nodes.len() <= cfg.max_nodes, "{name}");
                    assert!(c.muscles.len() <= cfg.max_muscles, "{name}");
                    repair(&mut c, &cfg, &mut Rng::new(73, 0, i));
                } else {
                    assert!(
                        c.nodes == body.nodes && c.bones == body.bones && c.muscles == body.muscles,
                        "{name} changed a body it refused"
                    );
                }
            }
            assert!(applied > 0, "{name} never applied");
        }
    }
}
