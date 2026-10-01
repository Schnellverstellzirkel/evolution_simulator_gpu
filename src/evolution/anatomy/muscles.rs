//! Operators that add, move, split, fuse and retime muscles.
use super::limbs::{pick, split_bone_at};
use super::{
    BoneIds, Context, MuscleIds, branch, is_neck, long_enough, map_node, muscles_on, needed,
    neighbours, new_muscle, parent_bones, paths, pick_each, room,
};
use crate::config::Config;
use crate::evolution::{
    Bounded, Creature, MAX_MUSCLES, MAX_NODES, Muscle, Muscles, Rng, STRENGTH_MIN,
};

/// Adds a muscle between two nodes with two joints between them (three bones
/// on the way), timed like an existing muscle across one of those bones: one
/// contraction moves two joints together.
pub(crate) fn add_biarticular_muscle(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let (n, paths) = (c.nodes.len(), paths(c));
    let pairs: Bounded<(u8, u8), { MAX_NODES * MAX_NODES / 2 }> = (0..n)
        .flat_map(|u| (u + 1..n).map(move |v| (u, v)))
        .filter(|&(u, v)| (paths[u] ^ paths[v]).count_ones() == 3)
        .map(|(u, v)| (u as u8, v as u8))
        .collect();
    if pairs.is_empty() {
        return false;
    }
    let (u, v) = pairs[rng.index(pairs.len())];
    let (u, v) = (u as usize, v as usize);
    let way = paths[u] ^ paths[v];
    let bones: BoneIds = (0..c.bones.len()).filter(|&b| way >> b & 1 == 1).collect();
    let timing = muscles_on(c, &bones, false);
    if timing.is_empty() {
        return false;
    }
    let template = c.muscles[timing[rng.index(timing.len())]];
    let m = new_muscle(u, v, Some(&template), rng);
    c.muscles.push(m);
    true
}

/// Moves one end of a muscle to a node a bone away from it, so the muscle
/// then controls a different joint.
pub(crate) fn move_muscle_to_neighbor(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let (paths, needed) = (paths(c), needed(c));
    // (muscle, whether its `b` end moves, the node that end moves to)
    let options = |push: &mut dyn FnMut((usize, bool, usize))| {
        for (i, m) in c.muscles.iter().enumerate() {
            if needed.contains(&i) {
                continue;
            }
            for (end_b, node, other) in [
                (false, m.node_a as usize, m.node_b as usize),
                (true, m.node_b as usize, m.node_a as usize),
            ] {
                for to in neighbours(c, node) {
                    if to != other && long_enough(&paths, to, other) {
                        push((i, end_b, to));
                    }
                }
            }
        }
    };
    let Some((i, end_b, to)) = pick_each(rng, options) else {
        return false;
    };
    if end_b {
        c.muscles[i].node_b = to as u32;
    } else {
        c.muscles[i].node_a = to as u32;
    }
    true
}

/// Duplicates a muscle; the copy moves one end to a neighbouring node or
/// shifts its phase a little, so one connection can specialize into two.
pub(crate) fn split_muscle(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.is_empty() || !room(c, cfg, 0, 1) {
        return false;
    }
    let old = c.muscles[rng.index(c.muscles.len())];
    let mut m = old;
    let mut moved = false;
    if rng.unit() < 0.5 {
        let paths = paths(c);
        let end_b = rng.unit() < 0.5;
        let (node, other) = if end_b {
            (old.node_b as usize, old.node_a as usize)
        } else {
            (old.node_a as usize, old.node_b as usize)
        };
        let to: BoneIds = neighbours(c, node)
            .into_iter()
            .filter(|&n| n != other && long_enough(&paths, n, other))
            .collect();
        if let Some(to) = pick(&to, rng) {
            if end_b {
                m.node_b = to as u32;
            } else {
                m.node_a = to as u32;
            }
            moved = true;
        }
    }
    if !moved {
        m.phase = (m.phase + rng.range(-0.1, 0.1)).rem_euclid(1.0);
    }
    c.muscles.push(m);
    true
}

/// Merges two muscles between the same two nodes with similar phase into
/// one with averaged timing and the strength of both (two muscles side by
/// side pull as one stronger one).
pub(crate) fn fuse_similar_muscles(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    // `q` seen from `p`'s side.
    let facing = |p: &Muscle, q: &Muscle| {
        if (q.node_a, q.node_b) == (p.node_b, p.node_a) {
            q.flipped()
        } else {
            *q
        }
    };
    let mut pairs: Bounded<(u8, u8), { MAX_MUSCLES * MAX_MUSCLES / 2 }> = Bounded::new();
    for (i, p) in c.muscles.iter().enumerate() {
        for (j, q) in c.muscles.iter().enumerate().skip(i + 1) {
            let q = facing(p, q);
            if (q.node_a, q.node_b) == (p.node_a, p.node_b) && turn(p.phase, q.phase).abs() < 0.15 {
                pairs.push((i as u8, j as u8));
            }
        }
    }
    if pairs.is_empty() {
        return false;
    }
    let (i, j) = pairs[rng.index(pairs.len())];
    let (i, j) = (i as usize, j as usize);
    let q = facing(&c.muscles[i], &c.muscles[j]);
    let p = &mut c.muscles[i];
    p.duty = 0.5 * (p.duty + q.duty);
    p.strength = (p.strength + q.strength).min(1.0);
    p.phase = (p.phase + 0.5 * turn(p.phase, q.phase)).rem_euclid(1.0);
    p.reset = (p.reset + 0.5 * turn(p.reset, q.reset)).rem_euclid(1.0);
    c.muscles.remove(j);
    true
}

/// For a joint that already has a muscle closing it, adds a muscle that
/// opens it: from the same node on the child side to a node outside the
/// child's limb, checked geometrically to turn the child the other way, with
/// the phase half a cycle from the closer.
///
/// A muscle between the two sides of a joint can only close it when it runs
/// along the bones, so the opener goes to a node on the other side of the
/// child's line in the pose. Bodies without such a node are skipped.
pub(crate) fn add_antagonist(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let paths = paths(c);
    // (closer with its child-side end first, the opener's other node)
    let options = |push: &mut dyn FnMut((Muscle, usize))| {
        for m in &c.muscles {
            let span = paths[m.node_a as usize] ^ paths[m.node_b as usize];
            if span.count_ones() != 2 {
                continue;
            }
            let (x, y) = (span.trailing_zeros() as usize, 31 - span.leading_zeros() as usize);
            let Some(joint) = shared_node(c, x, y) else {
                continue;
            };
            for k in [x, y] {
                if c.bones[k].a != joint {
                    continue;
                }
                for closer in [*m, m.flipped()] {
                    let (u, v) = (closer.node_a as usize, closer.node_b as usize);
                    if paths[u] >> k & 1 == 0 || paths[v] >> k & 1 == 1 {
                        continue;
                    }
                    let closing = torque(c, joint as usize, u, v);
                    if closing.abs() < MIN_TORQUE {
                        continue;
                    }
                    for r in (0..c.nodes.len()).filter(|&r| paths[r] >> k & 1 == 0) {
                        if r != v && long_enough(&paths, u, r) {
                            let opening = torque(c, joint as usize, u, r);
                            if opening * closing < 0.0 && opening.abs() >= MIN_TORQUE {
                                push((closer, r));
                            }
                        }
                    }
                }
            }
        }
    };
    let Some((closer, r)) = pick_each(rng, options) else {
        return false;
    };
    let mut m = new_muscle(closer.node_a as usize, r, Some(&closer), rng);
    m.phase = (closer.phase + 0.5).rem_euclid(1.0);
    c.muscles.push(m);
    true
}

/// Exchanges the destinations of two muscles: A-B and C-D become A-D and
/// C-B (skipping pairs that would join two nodes a bone apart).
pub(crate) fn swap_muscle_routes(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let (paths, needed) = (paths(c), needed(c));
    let free: MuscleIds = (0..c.muscles.len()).filter(|i| !needed.contains(i)).collect();
    let mut pairs: Bounded<(u8, u8), { MAX_MUSCLES * MAX_MUSCLES / 2 }> = Bounded::new();
    for (n, &i) in free.iter().enumerate() {
        for &j in &free[n + 1..] {
            let (p, q) = (c.muscles[i], c.muscles[j]);
            if p.node_a != q.node_a
                && p.node_b != q.node_b
                && p.node_a != q.node_b
                && q.node_a != p.node_b
                && long_enough(&paths, p.node_a as usize, q.node_b as usize)
                && long_enough(&paths, q.node_a as usize, p.node_b as usize)
            {
                pairs.push((i as u8, j as u8));
            }
        }
    }
    if pairs.is_empty() {
        return false;
    }
    let (i, j) = pairs[rng.index(pairs.len())];
    let (i, j) = (i as usize, j as usize);
    let (p, q) = (c.muscles[i], c.muscles[j]);
    c.muscles[i].node_b = q.node_b;
    c.muscles[j].node_b = p.node_b;
    true
}

/// Spreads several muscles that end at one node over a bone at that node, at
/// even steps along it, keeping their timing, so they act at different
/// leverages. The bone is split at the fan points (a new node and joint for
/// each muscle but the first), and each muscle ends at its own.
pub(crate) fn fan_muscle_attachments(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    // Every muscle end: (muscle, whether it is the `b` end, node).
    let ends: Bounded<(usize, bool, u32), { 2 * MAX_MUSCLES }> = c
        .muscles
        .iter()
        .enumerate()
        .flat_map(|(i, m)| [(i, false, m.node_a), (i, true, m.node_b)])
        .collect();
    // A bone at `node` long enough for its parts.
    let fan_bones = |node: usize| -> BoneIds {
        c.bones
            .iter()
            .enumerate()
            .filter(|(_, b)| (b.a as usize == node || b.b as usize == node) && b.rest_length >= 0.1)
            .map(|(j, _)| j)
            .collect()
    };
    let seeds: Bounded<usize, { 2 * MAX_MUSCLES }> = (0..ends.len())
        .filter(|&s| {
            let node = ends[s].2;
            ends.iter().filter(|e| e.2 == node).count() >= 2 && !fan_bones(node as usize).is_empty()
        })
        .collect();
    if seeds.is_empty() {
        return false;
    }
    let node = ends[seeds[rng.index(seeds.len())]].2 as usize;
    let group: Bounded<(usize, bool), 3> = ends
        .iter()
        .filter(|e| e.2 as usize == node)
        .map(|&(i, end_b, _)| (i, end_b))
        .take(3)
        .collect();
    let Some(j) = pick(&fan_bones(node), rng) else {
        return false;
    };
    // How many muscles fit, with a new node each but the first.
    let g = group
        .len()
        .min(1 + cfg.max_nodes.min(MAX_NODES).saturating_sub(c.nodes.len()));
    if g < 2 {
        return false;
    }
    let from_pivot = c.bones[j].a as usize == node;
    // Fan point k lies k / g of the way from `node`; as a fraction of the
    // bone from its pivot end it is k / g or 1 - k / g.
    let point = |k: usize| {
        if from_pivot { k as f32 / g as f32 } else { 1.0 - k as f32 / g as f32 }
    };
    let mut fractions: Bounded<f32, 3> = (1..g).map(point).collect();
    fractions.sort_stable_by(f32::total_cmp);
    let mut made: Bounded<(f32, usize), 3> = Bounded::new();
    let (mut whole, mut done) = (j, 0.0f32);
    for &f in &fractions {
        let (mid, second) = split_bone_at(c, whole, (f - done) / (1.0 - done), rng);
        made.push((f, mid));
        whole = second;
        done = f;
    }
    // The first muscle stays at `node`; muscle k ends at fan point k.
    let paths = paths(c);
    for (k, &(i, end_b)) in group.iter().enumerate().take(g).skip(1) {
        let Some(&(_, to)) = made.iter().find(|(f, _)| (f - point(k)).abs() < 1e-6) else {
            continue;
        };
        let m = &mut c.muscles[i];
        let other = if end_b { m.node_a } else { m.node_b } as usize;
        if !long_enough(&paths, to, other) {
            continue;
        }
        if end_b {
            m.node_b = to as u32;
        } else {
            m.node_a = to as u32;
        }
    }
    true
}

/// Replaces a muscle across four or more bones with two muscles through a
/// node on the way between its ends, starting in phase.
pub(crate) fn relay_muscle(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let (paths, needed) = (paths(c), needed(c));
    // Nodes on the way between a muscle's ends that leave a muscle of at
    // least two bones on each side.
    let via = |i: usize, sink: &mut dyn FnMut(usize)| {
        let m = c.muscles[i];
        let (u, v) = (m.node_a as usize, m.node_b as usize);
        let way = paths[u] ^ paths[v];
        for w in 0..c.nodes.len() {
            let (first, second) = (paths[u] ^ paths[w], paths[w] ^ paths[v]);
            if w != u
                && w != v
                && first & second == 0
                && first | second == way
                && first.count_ones() >= 2
                && second.count_ones() >= 2
            {
                sink(w);
            }
        }
    };
    let options: MuscleIds = (0..c.muscles.len())
        .filter(|&i| {
            let mut any = false;
            via(i, &mut |_| any = true);
            !needed.contains(&i) && any
        })
        .collect();
    let Some(i) = pick(&options, rng) else {
        return false;
    };
    let mut ways = BoneIds::new();
    via(i, &mut |w| ways.push(w));
    let w = ways[rng.index(ways.len())];
    let old = c.muscles[i];
    // The first half keeps the muscle. The second half is the added muscle.
    c.muscles[i].node_b = w as u32;
    let second = new_muscle(w, old.node_b as usize, Some(&old), rng);
    c.muscles.push(second);
    true
}

/// Copies the muscle pattern of one limb (ends, strength and timing) onto
/// another limb with the same number of bones. The recipient keeps its
/// skeleton.
///
/// A limb's pattern is every muscle among its bones and the bone above it.
/// Bones map by their order in the branch, and so do the nodes they join. The
/// recipient loses its own pattern, except the muscles `repair` would put
/// back.
pub(crate) fn copy_actuation_to_limb(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs: Bounded<(usize, BoneIds, MuscleIds), MAX_NODES> = (0..c.bones.len())
        .filter(|&b| !is_neck(c, b))
        .map(|b| {
            let (bones, muscles) = actuation(c, b);
            (b, bones, muscles)
        })
        .collect();
    let mut pairs: Bounded<(u8, u8), { MAX_NODES * MAX_NODES }> = Bounded::new();
    for (x, from) in limbs.iter().enumerate() {
        for (y, to) in limbs.iter().enumerate() {
            if from.1.len() == to.1.len()
                && !from.2.is_empty()
                && !from.1.contains(&to.0)
                && !to.1.contains(&from.0)
            {
                pairs.push((x as u8, y as u8));
            }
        }
    }
    if pairs.is_empty() {
        return false;
    }
    let (x, y) = pairs[rng.index(pairs.len())];
    let (from, to) = (&limbs[x as usize], &limbs[y as usize]);
    let needed = needed(c);
    let dropped: MuscleIds = to.2.iter().copied().filter(|i| !needed.contains(i)).collect();
    if c.muscles.len() - dropped.len() + from.2.len() > cfg.max_muscles {
        return false;
    }
    let copies: Muscles = from
        .2
        .iter()
        .filter_map(|&i| {
            let old = c.muscles[i];
            let a = map_node(c, &from.1, &to.1, old.node_a)?;
            let b = map_node(c, &from.1, &to.1, old.node_b)?;
            (a != b).then_some(Muscle { node_a: a, node_b: b, ..old })
        })
        .collect();
    let mut index = 0;
    c.muscles.retain(|_| {
        index += 1;
        !dropped.contains(&(index - 1))
    });
    c.muscles.extend(copies);
    true
}

/// Weakens every active muscle across a branch by one factor (0.3 to 0.7),
/// keeping geometry and timing: a limb that fights the gait becomes a quieter
/// support.
pub(crate) fn quiet_muscle_group(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let active = |c: &Creature, root: usize| -> MuscleIds {
        muscles_on(c, &branch(c, root), false)
            .into_iter()
            .filter(|&i| c.muscles[i].active())
            .collect()
    };
    let roots: BoneIds = (0..c.bones.len())
        .filter(|&b| !is_neck(c, b) && !active(c, b).is_empty())
        .collect();
    if roots.is_empty() {
        return false;
    }
    let group = active(c, roots[rng.index(roots.len())]);
    let factor = rng.range(0.3, 0.7);
    for i in group {
        let m = &mut c.muscles[i];
        m.strength = (m.strength * factor).max(STRENGTH_MIN);
    }
    true
}

/// Least torque (lever times pull, in m^2) that counts as turning a bone.
const MIN_TORQUE: f32 = 1.0e-3;

/// The torque the pull of a muscle puts on the limb below `joint` through its
/// end `u` toward its end `v`, about the joint, in the pose. Positive turns
/// the limb counterclockwise.
fn torque(c: &Creature, joint: usize, u: usize, v: usize) -> f32 {
    let (j, p, q) = (c.nodes[joint], c.nodes[u], c.nodes[v]);
    (p.x - j.x) * (q.y - p.y) - (p.y - j.y) * (q.x - p.x)
}

/// The signed shortest step from phase `from` to phase `to` (-0.5 to 0.5).
fn turn(from: f32, to: f32) -> f32 {
    (to - from + 0.5).rem_euclid(1.0) - 0.5
}

/// The node bones `x` and `y` share, if any.
pub(super) fn shared_node(c: &Creature, x: usize, y: usize) -> Option<u32> {
    let (p, q) = (c.bones[x], c.bones[y]);
    [p.a, p.b].into_iter().find(|&n| n == q.a || n == q.b)
}

/// The bones of the limb that starts at `root` followed by the bone above
/// it, and the muscles lying among those bones.
pub(super) fn actuation(c: &Creature, root: usize) -> (BoneIds, MuscleIds) {
    let mut bones = branch(c, root);
    bones.extend(parent_bones(c)[c.bones[root].a as usize]);
    let muscles = muscles_on(c, &bones, true);
    (bones, muscles)
}

#[cfg(test)]
mod tests {
    use super::super::{Operator, span_of, tests::bodies};
    use super::*;

    /// Runs `op` on test bodies. A change must
    /// pass `check(before, after)`. No change must leave the body as it was.
    /// Returns how often the operator applied, out of 160 tries.
    fn each_change(op: Operator, check: impl Fn(&Creature, &Creature)) -> usize {
        let cfg = Config::default();
        let mut applied = 0;
        for (i, body) in bodies(&cfg, 80).iter().enumerate() {
            for variant in 0..2u32 {
                let mut c = body.clone();
                let mut rng = Rng::new(21, variant, i);
                let cx = Context { donor: None };
                if op(&mut c, &cfg, &mut rng, &cx) {
                    applied += 1;
                    assert!(c.muscles.len() <= cfg.max_muscles);
                    check(body, &c);
                } else {
                    assert_eq!(c.nodes, body.nodes);
                    assert_eq!(c.bones, body.bones);
                    assert_eq!(c.muscles, body.muscles);
                }
            }
        }
        applied
    }

    /// Indices where two muscle lists of equal length differ.
    fn differing(before: &[Muscle], after: &[Muscle]) -> Vec<usize> {
        assert_eq!(before.len(), after.len());
        (0..before.len())
            .filter(|&i| before[i] != after[i])
            .collect()
    }

    fn same_timing(p: &Muscle, q: &Muscle) -> bool {
        (p.period, p.phase, p.duty) == (q.period, q.phase, q.duty)
    }

    /// Whether two nodes are joined by a bone.
    fn adjacent(c: &Creature, x: u32, y: u32) -> bool {
        neighbours(c, x as usize).contains(&(y as usize))
    }

    #[test]
    fn biarticular_muscle_spans_two_joints() {
        let applied = each_change(add_biarticular_muscle, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            assert_eq!(after.bones, before.bones);
            let m = after.muscles.last().unwrap();
            let paths = paths(after);
            let way = span_of(&paths, m);
            assert_eq!(way.count_ones(), 3);
            let bones: Vec<usize> = (0..after.bones.len()).filter(|&b| way >> b & 1 == 1).collect();
            let timed_like = muscles_on(before, &bones, false)
                .into_iter()
                .any(|i| before.muscles[i].phase == m.phase);
            assert!(timed_like, "phase comes from a muscle across the way");
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn moved_muscle_end_goes_to_a_neighbouring_node() {
        let applied = each_change(move_muscle_to_neighbor, |before, after| {
            assert_eq!(after.bones, before.bones);
            let changed = differing(&before.muscles, &after.muscles);
            assert_eq!(changed.len(), 1);
            let (p, q) = (before.muscles[changed[0]], after.muscles[changed[0]]);
            let (from, to) = if p.node_a == q.node_a {
                (p.node_b, q.node_b)
            } else {
                assert_eq!(p.node_b, q.node_b);
                (p.node_a, q.node_a)
            };
            assert_ne!(from, to);
            assert!(adjacent(before, from, to));
            assert!(same_timing(&p, &q));
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn split_muscle_adds_a_close_copy() {
        let applied = each_change(split_muscle, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            let copy = after.muscles.last().unwrap();
            let original = before.muscles.iter().any(|m| {
                let same = (m.node_a, m.node_b) == (copy.node_a, copy.node_b);
                let one_end_moved = (m.node_a == copy.node_a && adjacent(before, m.node_b, copy.node_b))
                    || (m.node_b == copy.node_b && adjacent(before, m.node_a, copy.node_a));
                (same && turn(m.phase, copy.phase).abs() <= 0.1)
                    || (one_end_moved && m.phase == copy.phase)
            });
            assert!(original);
        });
        assert!(applied >= 150, "applied {applied}");
    }

    #[test]
    fn fuse_similar_muscles_removes_exactly_one() {
        let applied = each_change(fuse_similar_muscles, |before, after| {
            assert_eq!(after.muscles.len() + 1, before.muscles.len());
        });
        assert!(applied > 0, "applied {applied}");
        // A split muscle that stayed between the same nodes can fuse back.
        let cfg = Config::default();
        let cx = Context { donor: None };
        let mut fused = 0;
        for (i, mut c) in bodies(&cfg, 40).into_iter().enumerate() {
            let mut rng = Rng::new(22, 0, i);
            let count = c.muscles.len();
            if !split_muscle(&mut c, &cfg, &mut rng, &cx) {
                continue;
            }
            let copy = *c.muscles.last().unwrap();
            if !c.muscles[..count].iter().any(|m| {
                (m.node_a, m.node_b) == (copy.node_a, copy.node_b)
                    && turn(m.phase, copy.phase).abs() < 0.15
            }) {
                continue;
            }
            assert!(fuse_similar_muscles(&mut c, &cfg, &mut rng, &cx));
            assert_eq!(c.muscles.len(), count);
            fused += 1;
        }
        assert!(fused > 5, "fused {fused}");
    }

    #[test]
    fn antagonist_turns_the_child_limb_against_its_closer() {
        let applied = each_change(add_antagonist, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            let opener = *after.muscles.last().unwrap();
            let paths = paths(before);
            // A closer at the joint of a child bone whose limb holds the
            // opener's first node and not its second turns the limb the other
            // way, half a cycle apart.
            let closer = before.muscles.iter().any(|m| {
                [*m, m.flipped()].iter().any(|m| {
                    let span = span_of(&paths, m);
                    if m.node_a != opener.node_a || span.count_ones() != 2 {
                        return false;
                    }
                    let (x, y) = (span.trailing_zeros() as usize, 31 - span.leading_zeros() as usize);
                    let Some(joint) = shared_node(before, x, y) else {
                        return false;
                    };
                    let (u, v, r) = (m.node_a as usize, m.node_b as usize, opener.node_b as usize);
                    torque(before, joint as usize, u, v) * torque(before, joint as usize, u, r) < 0.0
                        && turn(m.phase, opener.phase).abs() > 0.499
                })
            });
            assert!(closer);
        });
        assert!(applied >= 60, "applied {applied}");
    }

    #[test]
    fn swap_muscle_routes_exchanges_two_destinations() {
        let applied = each_change(swap_muscle_routes, |before, after| {
            let changed = differing(&before.muscles, &after.muscles);
            assert_eq!(changed.len(), 2);
            let [i, j] = [changed[0], changed[1]];
            let (p, q) = (before.muscles[i], before.muscles[j]);
            let (r, s) = (after.muscles[i], after.muscles[j]);
            assert_eq!((r.node_a, r.node_b), (p.node_a, q.node_b));
            assert_eq!((s.node_a, s.node_b), (q.node_a, p.node_b));
            assert!(same_timing(&p, &r) && same_timing(&q, &s));
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn fan_spreads_muscle_ends_evenly_along_a_bone() {
        let applied = each_change(fan_muscle_attachments, |before, after| {
            let (new_nodes, new_bones) = (
                after.nodes.len() - before.nodes.len(),
                after.bones.len() - before.bones.len(),
            );
            assert!((1..=2).contains(&new_nodes) && new_nodes == new_bones);
            let moved: Vec<usize> = (0..before.muscles.len())
                .filter(|&i| before.muscles[i] != after.muscles[i])
                .collect();
            assert!(!moved.is_empty() && moved.len() <= new_nodes);
            for &i in &moved {
                let (p, q) = (before.muscles[i], after.muscles[i]);
                assert!(same_timing(&p, &q));
                // Exactly one end moved, onto a node the operator made.
                let ends = [(p.node_a, q.node_a), (p.node_b, q.node_b)];
                let moved_ends: Vec<_> = ends.iter().filter(|(x, y)| x != y).collect();
                assert_eq!(moved_ends.len(), 1);
                assert!(moved_ends[0].1 as usize >= before.nodes.len());
            }
            // The new nodes sit on the line of the bone they split, at even
            // steps along it.
            let n = before.nodes.len();
            for new in n..after.nodes.len() {
                let q = after.nodes[new];
                let on_a_bone = before.bones.iter().any(|b| {
                    let (a, e) = (before.nodes[b.a as usize], before.nodes[b.b as usize]);
                    let (dx, dy) = (e.x - a.x, e.y - a.y);
                    let t = ((q.x - a.x) * dx + (q.y - a.y) * dy) / (dx * dx + dy * dy);
                    let off = ((q.x - a.x) - t * dx).hypot((q.y - a.y) - t * dy);
                    off < 1e-3 && (0.3..0.7).contains(&t) || off < 1e-3 && (0.3..0.7).contains(&(1.0 - t))
                });
                assert!(on_a_bone, "new node {new} is on a split bone");
            }
        });
        assert!(applied >= 20, "applied {applied}");
    }

    #[test]
    fn relay_routes_a_muscle_through_a_node_between() {
        let applied = each_change(relay_muscle, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            let kept = &after.muscles[..before.muscles.len()];
            let changed = differing(&before.muscles, kept);
            assert_eq!(changed.len(), 1);
            let old = before.muscles[changed[0]];
            let (first, second) = (after.muscles[changed[0]], *after.muscles.last().unwrap());
            assert_eq!(
                (first.node_a, first.node_b, second.node_a, second.node_b),
                (old.node_a, second.node_a, first.node_b, old.node_b)
            );
            let paths = paths(before);
            let (u, v, w) = (old.node_a as usize, old.node_b as usize, first.node_b as usize);
            let (up, down) = (paths[u] ^ paths[w], paths[w] ^ paths[v]);
            assert!(up & down == 0 && up | down == paths[u] ^ paths[v], "w lies on the way");
            assert!(first.phase == old.phase && second.phase == old.phase);
        });
        assert!(applied >= 10, "applied {applied}");
    }

    #[test]
    fn copied_actuation_matches_the_source_limb() {
        let applied = each_change(copy_actuation_to_limb, |_, after| {
            // Some pair of limbs now has every source muscle copied onto the
            // recipient's matching nodes.
            let limbs: Vec<_> = (0..after.bones.len())
                .filter(|&b| !is_neck(after, b))
                .map(|b| (b, actuation(after, b)))
                .collect();
            let copied = limbs.iter().any(|(from, (fb, fm))| {
                limbs.iter().any(|(to, (tb, _))| {
                    from != to
                        && fb.len() == tb.len()
                        && !fb.contains(to)
                        && !tb.contains(from)
                        && !fm.is_empty()
                        && fm.iter().all(|&i| {
                            let s = after.muscles[i];
                            let (a, b) = (
                                map_node(after, fb, tb, s.node_a),
                                map_node(after, fb, tb, s.node_b),
                            );
                            after.muscles.iter().any(|m| {
                                Some((m.node_a, m.node_b)) == a.zip(b)
                                    && (m.strength, m.phase) == (s.strength, s.phase)
                            })
                        })
                })
            });
            assert!(copied);
        });
        assert!(applied >= 20, "applied {applied}");
    }

    #[test]
    fn quiet_group_weakens_one_branch_by_one_factor() {
        let applied = each_change(quiet_muscle_group, |before, after| {
            let changed = differing(&before.muscles, &after.muscles);
            assert!(!changed.is_empty());
            let factors: Vec<f32> = changed
                .iter()
                .map(|&i| {
                    let (p, q) = (before.muscles[i], after.muscles[i]);
                    assert!(same_timing(&p, &q));
                    assert_eq!((p.node_a, p.node_b), (q.node_a, q.node_b));
                    q.strength / p.strength
                })
                .collect();
            assert!(factors.iter().all(|f| (0.29..=0.71).contains(f)));
            assert!(factors.iter().all(|f| (f - factors[0]).abs() < 1e-2));
            let one_branch = (0..before.bones.len()).any(|root| {
                let on = muscles_on(before, &branch(before, root), false);
                changed.iter().all(|i| on.contains(i))
            });
            assert!(one_branch);
        });
        assert!(applied >= 150, "applied {applied}");
    }
}
