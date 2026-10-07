//! Leg operators. They make the parts of a mammal-like gait cheap to reach by
//! mutation: legs that hang under the trunk, come in front and back pairs,
//! spread along the trunk, and step in a fixed phase against each other.
//!
//! The sources are Sims (1994, limbs added in pairs with mirrored timing),
//! Lipson and Pollack (2000, legs as repeated rigid bars with actuators at the
//! hinges), Cheney et al. (2013, regular and symmetric bodies move further)
//! and Stanley (2007, regularity from repetition and symmetry). Like the other
//! compound operators, each is a whole change, and the ones that add nodes
//! give the idlest tips back.
use super::compound::{close_ring, hinge_muscle, lead_muscle, limb_phase, shed_tips, strongest};
use super::junctions::{add_node, keep_strokes, spans, turn_branch};
use super::limbs::clamped;
use super::rhythm::{leaf_limbs, tip_x};
use super::{
    BoneIds, Context, MuscleIds, branch, branch_nodes, child_bones, copy_branch_limited, is_neck,
    muscles_on, parent_bones, room,
};
use crate::config::Config;
use crate::evolution::{Bone, Creature, Rng};

/// The leaf limb whose tip is nearest to `x` along the body, if any.
fn nearest_leg(c: &Creature, legs: &[BoneIds], x: f32) -> Option<BoneIds> {
    legs.iter()
        .min_by(|p, q| (tip_x(c, p) - x).abs().total_cmp(&(tip_x(c, q) - x).abs()))
        .copied()
}

/// Hangs a new leg of two bones from a node of the trunk: a thigh that points
/// down and a shank below the knee, which bends a little forward or back. A
/// muscle across the hip and a muscle across the knee drive it, and half the
/// time a second hip muscle pulls the other way half a cycle later (a flexor
/// and an extensor). The leg steps against the nearest existing leg: half a
/// cycle after it (a walk or a trot), a quarter or three quarters (a
/// gallop) or with it (a bound), and the knee runs a quarter cycle behind
/// the hip. Without a leg, the leg takes the timing of the gait's main driver.
pub(crate) fn sprout_leg(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 2, 3) {
        return false;
    }
    let parents = parent_bones(c);
    // The trunk: the nodes in no leg.
    let legs = leaf_limbs(c);
    let in_leg = legs
        .iter()
        .flat_map(|leg| leg.iter())
        .fold(0u32, |mask, &b| mask | 1 << c.bones[b].b);
    let hips: BoneIds = (1..c.nodes.len())
        .filter(|&n| parents[n].is_some() && (in_leg >> n) & 1 == 0 && c.nodes[n].y > 0.12)
        .collect();
    let Some(&hip) = hips.get(rng.index(hips.len().max(1))) else {
        return false;
    };
    let Some(above) = parents[hip] else {
        return false;
    };
    let count: usize = legs.iter().map(|l| l.len()).sum();
    let base = if count == 0 {
        0.3
    } else {
        legs.iter()
            .flat_map(|l| l.iter().map(|&b| c.bones[b].rest_length))
            .sum::<f32>()
            / count as f32
    };
    let thigh = (base * rng.range(0.8, 1.4)).clamp(0.06, 0.6);
    let shank = (base * rng.range(0.8, 1.4)).clamp(0.06, 0.6);
    let h = c.nodes[hip];
    let lean = rng.range(-0.35, 0.35);
    let [kx, ky] = clamped(h.x + thigh * lean.sin(), h.y - thigh * lean.cos());
    let bend = lean + rng.range(-0.5, 0.5);
    let [fx, fy] = clamped(kx + shank * bend.sin(), ky - shank * bend.cos());
    let mut next = c.clone();
    let knee = add_node(&mut next, hip, [kx, ky]);
    let foot = add_node(&mut next, hip, [fx, fy]);
    let first = next.bones.len();
    for (a, b) in [(hip, knee), (knee, foot)] {
        let [dx, dy] = [
            next.nodes[b].x - next.nodes[a].x,
            next.nodes[b].y - next.nodes[a].y,
        ];
        let mut bone = Bone::new(a as u32, b as u32, dx.hypot(dy).max(0.03));
        bone.min_angle = -rng.range(0.3, 0.8);
        bone.max_angle = rng.range(0.3, 0.8);
        next.bones.push(bone);
    }
    // Timing: against the nearest leg, or the main driver.
    let reference = nearest_leg(c, &legs, h.x)
        .and_then(|limb| limb_phase(c, &limb))
        .or_else(|| lead_muscle(c, &[]).map(|i| c.muscles[i].phase));
    let Some(reference) = reference else {
        return false;
    };
    let offset = [0.5, 0.5, 0.25, 0.75, 0.0][rng.index(5)];
    let all: MuscleIds = (0..c.muscles.len()).collect();
    let near = muscles_on(c, &[above], false);
    let pool = if near.is_empty() { &all } else { &near };
    let Some(template) = strongest(c, pool).map(|i| c.muscles[i]) else {
        return false;
    };
    let hip_phase = reference + offset;
    let before = next.muscles.len();
    hinge_muscle(&mut next, cfg, first, &template, hip_phase, rng);
    hinge_muscle(&mut next, cfg, first + 1, &template, hip_phase + 0.25, rng);
    // The extensor: the hip muscle again, attached the other way, half a cycle later.
    if next.muscles.len() == before + 2 && rng.unit() < 0.5 && room(&next, cfg, 0, 1) {
        let mut extensor = next.muscles[before];
        extensor.anchor_a = 1.0 - extensor.anchor_a;
        extensor.anchor_b = (1.0 - extensor.anchor_b).clamp(0.0, 1.0);
        extensor.phase = (extensor.phase + 0.5).rem_euclid(1.0);
        extensor.reset = (extensor.reset + 0.5).rem_euclid(1.0);
        next.muscles.push(extensor);
    }
    if next.muscles.len() == before {
        return false;
    }
    let shed = shed_tips(&mut next, c.nodes.len(), 2, rng);
    if shed < 1 || !close_ring(&mut next, cfg, rng) {
        return false;
    }
    c.clone_from(&next);
    true
}

/// Copies a leg to the other end of the trunk, to the node nearest the
/// mirror image of its hip about the middle of the body, so a biped becomes a
/// quadruped. The copy is translated or reflected, and its muscles run in the
/// phase of one four-legged gait: half a cycle after the leg (a trot, where
/// diagonal legs step together), with it (a bound), or a fifth of a cycle off
/// either way (a gallop).
pub(crate) fn mirror_leg_fore_aft(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let mut legs = leaf_limbs(c);
    legs.retain(|limb| limb.len() <= 3 && room(c, cfg, limb.len(), 0));
    let Some(limb) = legs.get(rng.index(legs.len().max(1))) else {
        return false;
    };
    let root = limb[0];
    let joint = c.bones[root].a as usize;
    if joint == 0 {
        return false;
    }
    let (lo, hi) = c.nodes.iter().fold((f32::MAX, f32::MIN), |(lo, hi), n| {
        (lo.min(n.x), hi.max(n.x))
    });
    let centre = 0.5 * (lo + hi);
    let wanted = 2.0 * centre - c.nodes[joint].x;
    if (wanted - c.nodes[joint].x).abs() < 0.12 * (hi - lo).max(0.2) {
        return false;
    }
    let inside = branch_nodes(c, limb);
    let Some(at) = (1..c.nodes.len())
        .filter(|&n| n != joint && !inside.contains(&n))
        .min_by(|&p, &q| {
            (c.nodes[p].x - wanted)
                .abs()
                .total_cmp(&(c.nodes[q].x - wanted).abs())
        })
    else {
        return false;
    };
    if (c.nodes[at].x - c.nodes[joint].x).abs() < 0.08 * (hi - lo).max(0.2) {
        return false;
    }
    let (from, to) = (c.nodes[joint], c.nodes[at]);
    let mirror = rng.unit() < 0.5;
    let shift = [0.5, 0.5, 0.0, 0.2, 0.8][rng.index(5)];
    let mut next = c.clone();
    let place = |[x, y]: [f32; 2]| {
        let dx = if mirror { from.x - x } else { x - from.x };
        [to.x + dx, to.y + (y - from.y)]
    };
    if copy_branch_limited(&mut next, cfg, root, at, place, mirror, shift, usize::MAX).is_none() {
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

/// Moves a leg's hip along the trunk to the nearest node that lies farther
/// from the middle of the body, so the legs spread out toward the ends
/// instead of bunching. The leg keeps its shape and muscles, and the strokes
/// of the muscles from its first bone to the bone above are refitted.
pub(crate) fn spread_leg_attachment(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let legs = leaf_limbs(c);
    let Some(limb) = legs.get(rng.index(legs.len().max(1))) else {
        return false;
    };
    let root = limb[0];
    let from = c.bones[root].a as usize;
    if from == 0 {
        return false;
    }
    let centre = c.nodes.iter().map(|n| n.x).sum::<f32>() / c.nodes.len() as f32;
    let side = |n: usize| (c.nodes[n].x - centre).abs();
    let inside = branch_nodes(c, limb);
    let Some(at) = (1..c.nodes.len())
        .filter(|&n| n != from && !inside.contains(&n) && side(n) > side(from) + 0.04)
        .min_by(|&p, &q| {
            let d = |n: usize| {
                (c.nodes[n].x - c.nodes[from].x).abs() + (c.nodes[n].y - c.nodes[from].y).abs()
            };
            d(p).total_cmp(&d(q))
        })
    else {
        return false;
    };
    let parents = parent_bones(c);
    let before = spans(c);
    let offset = [
        c.nodes[at].x - c.nodes[from].x,
        c.nodes[at].y - c.nodes[from].y,
    ];
    for n in branch_nodes(c, &branch(c, root)) {
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

/// Turns a leg so its first bone points down, within a small lean, and its
/// feet end up under its hip: a leg that holds the trunk up, not a lever that
/// pushes sideways. Only legs that point 0.1 to 1.3 rad off straight down turn.
pub(crate) fn tuck_leg_under(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let children = child_bones(c);
    let legs: BoneIds = leaf_limbs(c)
        .iter()
        .map(|limb| limb[0])
        .filter(|&b| !is_neck(c, b) && !children[c.bones[b].a as usize].is_empty())
        .collect();
    let Some(&root) = legs.get(rng.index(legs.len().max(1))) else {
        return false;
    };
    let (a, b) = (
        c.nodes[c.bones[root].a as usize],
        c.nodes[c.bones[root].b as usize],
    );
    let angle = (b.y - a.y).atan2(b.x - a.x);
    let target = -std::f32::consts::FRAC_PI_2 + rng.range(-0.15, 0.15);
    let mut turn = target - angle;
    while turn > std::f32::consts::PI {
        turn -= std::f32::consts::TAU;
    }
    while turn < -std::f32::consts::PI {
        turn += std::f32::consts::TAU;
    }
    if !(0.1..=1.3).contains(&turn.abs()) {
        return false;
    }
    let before = spans(c);
    turn_branch(c, root, turn);
    keep_strokes(c, &before);
    true
}

#[cfg(test)]
mod tests {
    use super::super::{Operator, tests::bodies};
    use super::*;
    use crate::evolution::repair;

    fn run(op: Operator, check: impl Fn(&Creature, &Creature)) -> usize {
        let cfg = Config::default();
        let mut applied = 0;
        for (i, body) in bodies(&cfg, 160).into_iter().enumerate() {
            let mut c = body.clone();
            let cx = Context::of(None);
            if op(&mut c, &cfg, &mut Rng::new(61, 0, i), &cx) {
                applied += 1;
                assert!(c.nodes.len() <= cfg.max_nodes && c.muscles.len() <= cfg.max_muscles);
                check(&body, &c);
                repair(&mut c, &cfg, &mut Rng::new(63, 0, i));
            } else {
                assert!(
                    c.nodes == body.nodes && c.bones == body.bones && c.muscles == body.muscles
                );
            }
        }
        applied
    }

    #[test]
    fn sprout_leg_adds_a_leg_and_gives_a_tip_back() {
        let n = run(sprout_leg, |before, after| {
            assert!(after.nodes.len() <= before.nodes.len() + 1);
            assert!(after.nodes.len() >= before.nodes.len() - 1);
        });
        assert!(n >= 20, "applied to {n}");
    }

    #[test]
    fn mirror_leg_fore_aft_copies_a_leg_across_the_trunk() {
        let n = run(mirror_leg_fore_aft, |before, after| {
            assert!(after.bones.len() >= before.bones.len());
        });
        assert!(n >= 10, "applied to {n}");
    }

    #[test]
    fn spread_leg_attachment_moves_one_hip_outward() {
        let n = run(spread_leg_attachment, |before, after| {
            assert_eq!(after.nodes.len(), before.nodes.len());
            let moved = (0..before.bones.len())
                .filter(|&b| after.bones[b].a != before.bones[b].a)
                .count();
            assert_eq!(moved, 1);
        });
        assert!(n >= 10, "applied to {n}");
    }

    #[test]
    fn tuck_leg_under_keeps_the_shape() {
        let n = run(tuck_leg_under, |before, after| {
            assert_eq!(after.bones.len(), before.bones.len());
            assert_eq!(after.muscles.len(), before.muscles.len());
        });
        assert!(n >= 10, "applied to {n}");
    }
}
