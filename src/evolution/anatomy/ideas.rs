//! Helpers the idea files (`idea_*.rs`) share.
use super::limbs::clamped;
use super::limbs::pick;
use super::rhythm::leaf_limbs;
use super::{BoneIds, MuscleIds, branch, branch_nodes, child_bones, muscles_on};
use crate::config::Config;
use crate::evolution::{Creature, Muscle, Rng, max_bone_length};

/// A change smaller than this does not count as a change.
const EPSILON: f32 = 1.0e-4;

/// Sets `*slot` to `value` and records whether that changed it.
pub(super) fn set(slot: &mut f32, value: f32, changed: &mut bool) {
    if (*slot - value).abs() > EPSILON {
        *slot = value;
        *changed = true;
    }
}

/// Nodes with no bone below them, apart from the head: the feet and tips.
pub(super) fn leaf_nodes(c: &Creature) -> BoneIds {
    let children = child_bones(c);
    (1..c.nodes.len())
        .filter(|&n| children[n].is_empty())
        .collect()
}

/// Every node but the head.
pub(super) fn body_nodes(c: &Creature) -> BoneIds {
    (1..c.nodes.len()).collect()
}

/// Nodes that are neither the head nor a leaf: the trunk and the joints of
/// limbs.
pub(super) fn inner_nodes(c: &Creature) -> BoneIds {
    let children = child_bones(c);
    (1..c.nodes.len())
        .filter(|&n| !children[n].is_empty())
        .collect()
}

/// Friction at `t` of the way from the lowest to the highest the body may
/// have.
pub(super) fn grip(cfg: &Config, t: f32) -> f32 {
    cfg.min_friction + t.clamp(0.0, 1.0) * (cfg.max_friction - cfg.min_friction)
}

/// Node diameter at `t` of the way from the smallest to the largest.
pub(super) fn bulk(cfg: &Config, t: f32) -> f32 {
    cfg.min_size + t.clamp(0.0, 1.0) * (cfg.max_size - cfg.min_size)
}

/// The node two bones share, if they share one.
pub(super) fn shared_node(c: &Creature, x: usize, y: usize) -> Option<u32> {
    let (p, q) = (c.bones[x], c.bones[y]);
    [p.a, p.b].into_iter().find(|n| *n == q.a || *n == q.b)
}

/// A leg with at least `bones` bones and, with `driven`, a muscle on it.
pub(super) fn some_leg(c: &Creature, rng: &mut Rng, bones: usize, driven: bool) -> Option<BoneIds> {
    let legs = leaf_limbs(c);
    let fit: BoneIds = (0..legs.len())
        .filter(|&i| {
            legs[i].len() >= bones && (!driven || !muscles_on(c, &legs[i], false).is_empty())
        })
        .collect();
    pick(&fit, rng).map(|i| legs[i])
}

/// How hard a muscle can pull: stiffness times its stroke.
pub(super) fn drive(m: &Muscle) -> f32 {
    m.stiffness * (m.long - m.short).max(0.0)
}

/// Muscle indices sorted by drive, strongest first.
pub(super) fn by_drive(c: &Creature) -> MuscleIds {
    let mut ids: MuscleIds = (0..c.muscles.len()).collect();
    ids.sort_stable_by(|&a, &b| drive(&c.muscles[b]).total_cmp(&drive(&c.muscles[a])));
    ids
}

/// The muscles' phases wrapped into [0, 1).
pub(super) fn wrap(phase: f32) -> f32 {
    phase.rem_euclid(1.0)
}

/// The circular mean of phases (each a share of a cycle), or None when they
/// cancel out.
pub(super) fn circular_mean(phases: impl Iterator<Item = f32>) -> Option<f32> {
    let (mut s, mut k) = (0.0f32, 0.0f32);
    for p in phases {
        let a = p * std::f32::consts::TAU;
        s += a.sin();
        k += a.cos();
    }
    (s.hypot(k) > 1.0e-3).then(|| wrap(s.atan2(k) / std::f32::consts::TAU))
}

/// The signed shortest distance from phase `a` to phase `b`, in (-0.5, 0.5].
pub(super) fn phase_gap(a: f32, b: f32) -> f32 {
    let d = wrap(b - a);
    if d > 0.5 { d - 1.0 } else { d }
}

/// A coin flip.
pub(super) fn coin(rng: &mut Rng) -> bool {
    rng.unit() < 0.5
}

/// Scales the branch that starts at bone `root` about its root joint by
/// `factor`, within the bone limits, and the strokes of the muscles inside it
/// with it. False when the factor is within 2% of 1 after the limits.
pub(super) fn scale_branch(c: &mut Creature, root: usize, factor: f32) -> bool {
    let bones = branch(c, root);
    let low = bones
        .iter()
        .map(|&b| 0.03 / c.bones[b].rest_length)
        .fold(0.0, f32::max);
    let high = bones
        .iter()
        .map(|&b| max_bone_length() / c.bones[b].rest_length)
        .fold(f32::MAX, f32::min);
    let factor = factor.clamp(low, high.max(low));
    if (factor - 1.0).abs() < 0.02 {
        return false;
    }
    let pivot = c.nodes[c.bones[root].a as usize];
    for n in branch_nodes(c, &bones) {
        let node = &mut c.nodes[n];
        [node.x, node.y] = clamped(
            pivot.x + (node.x - pivot.x) * factor,
            pivot.y + (node.y - pivot.y) * factor,
        );
    }
    for &b in &bones {
        c.bones[b].rest_length *= factor;
    }
    for i in muscles_on(c, &bones, true) {
        c.muscles[i].short *= factor;
        c.muscles[i].long *= factor;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::tests::bodies;
    use super::super::{
        Context, idea_blend, idea_elastic, idea_shape, idea_surface, idea_timing, idea_topology,
        idea_wild,
    };
    use crate::config::Config;
    use crate::evolution::Rng;

    /// Every idea operator that says it changed a body did change it, and
    /// each fits a fair share of bodies, so none is dead code.
    #[test]
    fn idea_operators_change_the_body_they_report_and_fit_some_bodies() {
        let cfg = Config {
            slope: 0.1,
            hurdles: 0.2,
            ..Config::default()
        };
        let mut bodies = bodies(&cfg, 160);
        // Classic growth leaves no tendons and no organs, so half of the
        // bodies get some to give the operators that need them something to
        // work on.
        for (i, body) in bodies.iter_mut().enumerate().filter(|(i, _)| i % 2 == 0) {
            for (k, m) in body.muscles.iter_mut().enumerate() {
                m.tendon = [0.0, 0.3, 0.6][(k + i) % 3];
            }
            for (k, b) in body.bones.iter_mut().enumerate().skip(1).step_by(3) {
                b.organ_mass = 0.05;
                b.organ_at = 0.3 + 0.1 * (k % 4) as f32;
            }
        }
        let donor = bodies[bodies.len() / 2].clone();
        let lists = [
            idea_surface::OPS,
            idea_elastic::OPS,
            idea_timing::OPS,
            idea_topology::OPS,
            idea_blend::OPS,
            idea_wild::OPS,
            idea_shape::OPS,
        ];
        let mut dead = Vec::new();
        for (name, op) in lists.iter().flat_map(|l| l.iter()) {
            let mut applied = 0;
            for (i, body) in bodies.iter().enumerate() {
                for variant in 0..2u32 {
                    let mut c = body.clone();
                    let mut rng = Rng::new(29, variant, i);
                    let cx = Context {
                        donor: Some(&donor),
                    };
                    let said = op(&mut c, &cfg, &mut rng, &cx);
                    let changed =
                        c.nodes != body.nodes || c.bones != body.bones || c.muscles != body.muscles;
                    assert_eq!(
                        said,
                        changed,
                        "{name} on body {i}: returned {said} but the body {} changed",
                        if changed { "was" } else { "was not" }
                    );
                    applied += usize::from(said);
                }
            }
            eprintln!("{name}: applied to {applied} of {}", 2 * bodies.len());
            if applied < 8 {
                dead.push((*name, applied));
            }
        }
        assert!(dead.is_empty(), "rarely or never applied: {dead:?}");
    }
}
