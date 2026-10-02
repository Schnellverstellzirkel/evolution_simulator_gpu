//! Operators that add, move, split, fuse and retime muscles.
use super::{
    BoneIds, Context, MuscleIds, branch, fit_stroke, is_neck, muscles_on, new_muscle, parent_bones,
    pick_each, room,
};
use crate::config::Config;
use crate::evolution::{
    Bounded, Creature, MAX_MUSCLES, MAX_NODES, Muscle, Muscles, Rng, bone_point,
};

/// Adds a muscle between two bones separated by one intermediate bone (for
/// example trunk to lower leg), timed like an existing muscle on either end:
/// one contraction moves two joints together.
pub(crate) fn add_biarticular_muscle(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let n = c.bones.len();
    let pairs: Bounded<(usize, usize), { MAX_NODES * MAX_NODES / 2 }> = (0..n)
        .flat_map(|x| (x + 1..n).map(move |z| (x, z)))
        .filter(|&(x, z)| path_between(c, x, z).len() == 1)
        .collect();
    if pairs.is_empty() {
        return false;
    }
    let (x, z) = pairs[rng.index(pairs.len())];
    let timing = muscles_on(c, &[x, z], false);
    if timing.is_empty() {
        return false;
    }
    let template = c.muscles[timing[rng.index(timing.len())]];
    let anchors = (rng.unit(), rng.unit());
    let m = new_muscle(c, x, z, anchors, Some(&template), rng);
    c.muscles.push(m);
    true
}

/// Moves one end of a muscle to a bone that shares a node with its current
/// bone, near that shared node, and refits its stroke to the new span. The
/// muscle then controls a different joint.
pub(crate) fn move_muscle_to_neighbor(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    // (muscle, whether its `b` end moves, the bone that end moves to)
    let options = |push: &mut dyn FnMut((usize, bool, usize))| {
        for (i, m) in c.muscles.iter().enumerate() {
            if ring(c, m) {
                continue;
            }
            for (end_b, bone, other) in [(false, m.bone_a, m.bone_b), (true, m.bone_b, m.bone_a)] {
                for y in 0..c.bones.len() {
                    if y != bone as usize
                        && y != other as usize
                        && shared_node(c, bone as usize, y).is_some()
                    {
                        push((i, end_b, y));
                    }
                }
            }
        }
    };
    let Some((i, end_b, y)) = pick_each(rng, options) else {
        return false;
    };
    let old = c.muscles[i];
    let bone = (if end_b { old.bone_b } else { old.bone_a }) as usize;
    let node = shared_node(c, bone, y).expect("neighbours share a node");
    let near = rng.range(0.0, 0.25);
    let anchor = if c.bones[y].a == node {
        near
    } else {
        1.0 - near
    };
    let mut m = old;
    if end_b {
        (m.bone_b, m.anchor_b) = (y as u32, anchor);
    } else {
        (m.bone_a, m.anchor_a) = (y as u32, anchor);
    }
    fit_stroke(c, &mut m, Some(&old));
    c.muscles[i] = m;
    true
}

/// Duplicates a muscle; the copy moves one anchor or shifts its phase a
/// little, so one connection can specialize into two.
pub(crate) fn split_muscle(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.muscles.is_empty() || !room(c, cfg, 0, 1) {
        return false;
    }
    let old = c.muscles[rng.index(c.muscles.len())];
    let mut m = old;
    if rng.unit() < 0.5 {
        let shift = rng.range(-0.2, 0.2);
        if rng.unit() < 0.5 {
            m.anchor_a = (m.anchor_a + shift).clamp(0.0, 1.0);
        } else {
            m.anchor_b = (m.anchor_b + shift).clamp(0.0, 1.0);
        }
        fit_stroke(c, &mut m, Some(&old));
    } else {
        m.phase = (m.phase + rng.range(-0.1, 0.1)).rem_euclid(1.0);
    }
    c.muscles.push(m);
    true
}

/// Merges two muscles on the same pair of bones with nearby anchors and
/// similar phase into one with averaged genes.
pub(crate) fn fuse_similar_muscles(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    // `q` seen from `p`'s side.
    let facing = |p: &Muscle, q: &Muscle| {
        if (q.bone_a, q.bone_b) == (p.bone_b, p.bone_a) {
            flipped(q)
        } else {
            *q
        }
    };
    let mut pairs: Bounded<(u8, u8), { MAX_MUSCLES * MAX_MUSCLES / 2 }> = Bounded::new();
    for (i, p) in c.muscles.iter().enumerate() {
        for (j, q) in c.muscles.iter().enumerate().skip(i + 1) {
            let q = facing(p, q);
            if (q.bone_a, q.bone_b) == (p.bone_a, p.bone_b)
                && (p.anchor_a - q.anchor_a).abs() < 0.25
                && (p.anchor_b - q.anchor_b).abs() < 0.25
                && turn(p.phase, q.phase).abs() < 0.15
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
    let q = facing(&c.muscles[i], &c.muscles[j]);
    let p = &mut c.muscles[i];
    let mean = |x: f32, y: f32| 0.5 * (x + y);
    p.anchor_a = mean(p.anchor_a, q.anchor_a);
    p.anchor_b = mean(p.anchor_b, q.anchor_b);
    p.short = mean(p.short, q.short);
    p.long = mean(p.long, q.long);
    p.duty = mean(p.duty, q.duty);
    p.stiffness = mean(p.stiffness, q.stiffness);
    p.phase = (p.phase + 0.5 * turn(p.phase, q.phase)).rem_euclid(1.0);
    p.reset = (p.reset + 0.5 * turn(p.reset, q.reset)).rem_euclid(1.0);
    c.muscles.remove(j);
    true
}

/// For a joint whose bones already have a muscle that closes it, adds a
/// muscle that opens it: from the child bone to a bone on the other side of
/// the joint (a sibling or the bone beyond), checked geometrically to rotate
/// the child the other way, with the phase half a cycle from the closer.
///
/// A muscle between the two bones of a joint can only close it, because its
/// anchors lie on the bones' center lines. So the opener goes from the child
/// to a bone outside the child's branch whose attachment point lies on the
/// other side of the child's line in the pose. Bodies without such a bone
/// are skipped.
pub(crate) fn add_antagonist(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    // (closer with the child bone as `bone_a`, opener's other bone, anchor)
    let options = |push: &mut dyn FnMut((Muscle, usize, f32))| {
        for m in &c.muscles {
            for closer in [*m, flipped(m)] {
                let (k, q) = (closer.bone_a as usize, closer.bone_b as usize);
                let joint = c.bones[k].a;
                let closing = torque(c, &closer);
                if (c.bones[q].a != joint && c.bones[q].b != joint) || closing.abs() < MIN_TORQUE {
                    continue;
                }
                let limb = branch(c, k);
                for r in (0..c.bones.len()).filter(|r| !limb.contains(r)) {
                    for anchor in [0.0, 0.5, 1.0] {
                        let opener = Muscle {
                            bone_b: r as u32,
                            anchor_b: anchor,
                            ..closer
                        };
                        let opening = torque(c, &opener);
                        if opening * closing < 0.0 && opening.abs() >= MIN_TORQUE {
                            push((closer, r, anchor));
                        }
                    }
                }
            }
        }
    };
    let Some((closer, r, anchor)) = pick_each(rng, options) else {
        return false;
    };
    let k = closer.bone_a as usize;
    let mut m = new_muscle(c, k, r, (closer.anchor_a, anchor), Some(&closer), rng);
    m.phase = (closer.phase + 0.5).rem_euclid(1.0);
    c.muscles.push(m);
    true
}

/// Exchanges the destinations of two muscles: A-B and C-D become A-D and
/// C-B (skipping pairs that would join a bone to itself), strokes refitted.
pub(crate) fn swap_muscle_routes(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let free: MuscleIds = (0..c.muscles.len())
        .filter(|&i| !ring(c, &c.muscles[i]))
        .collect();
    let mut pairs: Bounded<(u8, u8), { MAX_MUSCLES * MAX_MUSCLES / 2 }> = Bounded::new();
    for (n, &i) in free.iter().enumerate() {
        for &j in &free[n + 1..] {
            let (p, q) = (c.muscles[i], c.muscles[j]);
            if p.bone_a != q.bone_a
                && p.bone_b != q.bone_b
                && p.bone_a != q.bone_b
                && q.bone_a != p.bone_b
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
    for (index, old, to) in [(i, p, q), (j, q, p)] {
        let mut m = Muscle {
            bone_b: to.bone_b,
            anchor_b: to.anchor_b,
            ..old
        };
        fit_stroke(c, &mut m, Some(&old));
        c.muscles[index] = m;
    }
    true
}

/// Spreads the anchors of several muscles that share a bone and sit close
/// together evenly along that bone, keeping their timing, so they act at
/// different leverages.
pub(crate) fn fan_muscle_attachments(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    // Every muscle end: (muscle, whether it is the `b` end, bone, anchor).
    let ends: Bounded<(usize, bool, u32, f32), { 2 * MAX_MUSCLES }> = c
        .muscles
        .iter()
        .enumerate()
        .flat_map(|(i, m)| {
            [
                (i, false, m.bone_a, m.anchor_a),
                (i, true, m.bone_b, m.anchor_b),
            ]
        })
        .collect();
    let close = |p: &(usize, bool, u32, f32), q: &(usize, bool, u32, f32)| {
        p.2 == q.2 && (p.3 - q.3).abs() < 0.15
    };
    let seeds: Bounded<usize, { 2 * MAX_MUSCLES }> = (0..ends.len())
        .filter(|&s| ends.iter().filter(|e| close(&ends[s], e)).count() >= 2)
        .collect();
    if seeds.is_empty() {
        return false;
    }
    let seed = ends[seeds[rng.index(seeds.len())]];
    let mut group: Bounded<_, { 2 * MAX_MUSCLES }> =
        ends.iter().filter(|e| close(&seed, e)).copied().collect();
    group.sort_stable_by(|p, q| p.3.total_cmp(&q.3));
    for (n, &(i, end_b, _, _)) in group.iter().enumerate() {
        let old = c.muscles[i];
        let mut m = old;
        let anchor = (n as f32 + 0.5) / group.len() as f32;
        if end_b {
            m.anchor_b = anchor;
        } else {
            m.anchor_a = anchor;
        }
        fit_stroke(c, &mut m, Some(&old));
        c.muscles[i] = m;
    }
    true
}

/// Replaces a muscle between two bones that are not neighbours with two
/// muscles through a bone on the path between them, starting in phase.
pub(crate) fn relay_muscle(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let path_of = |i: usize| {
        let m = c.muscles[i];
        path_between(c, m.bone_a as usize, m.bone_b as usize)
    };
    let options: MuscleIds = (0..c.muscles.len())
        .filter(|&i| !ring(c, &c.muscles[i]) && !path_of(i).is_empty())
        .collect();
    if options.is_empty() {
        return false;
    }
    let i = &options[rng.index(options.len())];
    let path = path_of(*i);
    let old = c.muscles[*i];
    let via = path[rng.index(path.len())];
    let at = rng.unit();
    // The first half keeps the muscle. The second half is the added muscle.
    let mut first = Muscle {
        bone_b: via as u32,
        anchor_b: at,
        ..old
    };
    fit_stroke(c, &mut first, Some(&old));
    let second = new_muscle(
        c,
        via,
        old.bone_b as usize,
        (at, old.anchor_b),
        Some(&old),
        rng,
    );
    c.muscles[*i] = first;
    c.muscles.push(second);
    true
}

/// Copies the muscle pattern of one limb (attachments, relative strokes and
/// timing) onto another limb with the same number of bones, scaled to the
/// recipient's lengths. The recipient keeps its skeleton.
///
/// A limb's pattern is every muscle among its bones and the bone above it.
/// Bones map by their order in the branch. The recipient loses its own
/// pattern, except the ring muscles that `repair` would put back.
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
    let dropped: MuscleIds =
        to.2.iter()
            .copied()
            .filter(|&i| !ring(c, &c.muscles[i]))
            .collect();
    if c.muscles.len() - dropped.len() + from.2.len() > cfg.max_muscles {
        return false;
    }
    let map = |b: u32| {
        to.1[from
            .1
            .iter()
            .position(|&x| x == b as usize)
            .expect("limb bone")]
    };
    let copies: Muscles = from
        .2
        .iter()
        .map(|&i| {
            let old = c.muscles[i];
            let mut m = Muscle {
                bone_a: map(old.bone_a) as u32,
                bone_b: map(old.bone_b) as u32,
                ..old
            };
            fit_stroke(c, &mut m, Some(&old));
            m
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

/// Shrinks the strokes of every muscle in a branch by one factor (0.3 to
/// 0.7), keeping geometry and timing: a limb that fights the gait becomes a
/// quieter support.
///
/// Each muscle keeps its relaxed length (`long`) and contracts less.
pub(crate) fn quiet_muscle_group(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let active = |c: &Creature, root: usize| -> MuscleIds {
        muscles_on(c, &branch(c, root), false)
            .into_iter()
            .filter(|&i| c.muscles[i].long > c.muscles[i].short)
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
        m.short = m.long - (m.long - m.short) * factor;
    }
    true
}

/// Least torque (lever times pull, in m^2) that counts as turning a bone.
const MIN_TORQUE: f32 = 1.0e-3;

/// The torque a muscle's pull puts on its `bone_a` about that bone's parent
/// node, in the pose. Positive turns it counterclockwise.
fn torque(c: &Creature, m: &Muscle) -> f32 {
    let bone = c.bones[m.bone_a as usize];
    let joint = c.nodes[bone.a as usize];
    let p = bone_point(bone, &c.nodes, m.anchor_a);
    let q = bone_point(c.bones[m.bone_b as usize], &c.nodes, m.anchor_b);
    (p[0] - joint.x) * (q[1] - p[1]) - (p[1] - joint.y) * (q[0] - p[0])
}

/// The same muscle with its ends swapped.
fn flipped(m: &Muscle) -> Muscle {
    Muscle {
        bone_a: m.bone_b,
        bone_b: m.bone_a,
        anchor_a: m.anchor_b,
        anchor_b: m.anchor_a,
        // Sensor endpoints are numbered bone_a.a, bone_a.b, bone_b.a, bone_b.b.
        sensor: if m.sensor < 4 {
            (m.sensor + 2) % 4
        } else {
            m.sensor
        },
        ..*m
    }
}

/// The signed shortest step from phase `from` to phase `to` (-0.5 to 0.5).
fn turn(from: f32, to: f32) -> f32 {
    (to - from + 0.5).rem_euclid(1.0) - 0.5
}

/// Whether a muscle joins consecutively numbered bones. `repair` keeps
/// a muscle on each such pair, so rerouting one would only make repair add a
/// random muscle in its place.
pub(super) fn ring(c: &Creature, m: &Muscle) -> bool {
    let n = c.bones.len();
    let (x, y) = (m.bone_a as usize, m.bone_b as usize);
    (x + 1) % n == y || (y + 1) % n == x
}

/// The node bones `x` and `y` share, if any.
pub(super) fn shared_node(c: &Creature, x: usize, y: usize) -> Option<u32> {
    let (p, q) = (c.bones[x], c.bones[y]);
    [p.a, p.b].into_iter().find(|&n| n == q.a || n == q.b)
}

/// The bones on the path between bones `x` and `z`, without them. It is
/// empty when they share a node.
fn path_between(c: &Creature, x: usize, z: usize) -> BoneIds {
    let parents = parent_bones(c);
    // A bone and the bones above it, up to the neck.
    let up = |mut bone: usize| {
        let mut out = BoneIds::from_slice(&[bone]);
        while let Some(above) = parents[c.bones[bone].a as usize] {
            out.push(above);
            bone = above;
        }
        out
    };
    let (from_x, from_z) = (up(x), up(z));
    if let Some(i) = from_x.iter().position(|&b| b == z) {
        return BoneIds::from_slice(&from_x[1..i]);
    }
    if let Some(i) = from_z.iter().position(|&b| b == x) {
        return BoneIds::from_slice(&from_z[1..i]);
    }
    // Both climb to a common bone. The path turns at that bone's lower node.
    let common = from_x
        .iter()
        .enumerate()
        .find_map(|(i, b)| from_z.iter().position(|y| y == b).map(|j| (i, j)));
    let Some((i, j)) = common else {
        return BoneIds::new();
    };
    from_x[1..i].iter().chain(&from_z[1..j]).copied().collect()
}

/// The bones of the limb that starts at `root` followed by the bone above
/// it, and the muscles with both ends on those bones.
pub(super) fn actuation(c: &Creature, root: usize) -> (BoneIds, MuscleIds) {
    let mut bones = branch(c, root);
    bones.extend(parent_bones(c)[c.bones[root].a as usize]);
    let muscles = muscles_on(c, &bones, true);
    (bones, muscles)
}

#[cfg(test)]
mod tests {
    use super::super::{Operator, tests::bodies};
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
                    assert_eq!(c.bones, body.bones);
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

    #[test]
    fn biarticular_muscle_spans_two_joints() {
        let applied = each_change(add_biarticular_muscle, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            let m = after.muscles.last().unwrap();
            let (x, z) = (m.bone_a as usize, m.bone_b as usize);
            assert_eq!(path_between(after, x, z).len(), 1);
            let timed_like = muscles_on(before, &[x, z], false)
                .into_iter()
                .any(|i| before.muscles[i].phase == m.phase);
            assert!(timed_like, "phase comes from a muscle on either end");
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn moved_muscle_end_goes_to_a_neighbouring_bone() {
        let applied = each_change(move_muscle_to_neighbor, |before, after| {
            let changed = differing(&before.muscles, &after.muscles);
            assert_eq!(changed.len(), 1);
            let (p, q) = (before.muscles[changed[0]], after.muscles[changed[0]]);
            let (from, to) = if p.bone_a == q.bone_a {
                (p.bone_b, q.bone_b)
            } else {
                assert_eq!(p.bone_b, q.bone_b);
                (p.bone_a, q.bone_a)
            };
            assert_ne!(from, to);
            assert!(shared_node(before, from as usize, to as usize).is_some());
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
                (m.bone_a, m.bone_b) == (copy.bone_a, copy.bone_b)
                    && (m.anchor_a - copy.anchor_a).abs() <= 0.2
                    && (m.anchor_b - copy.anchor_b).abs() <= 0.2
                    && turn(m.phase, copy.phase).abs() <= 0.1
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
        // A split muscle can always fuse back.
        let cfg = Config::default();
        let cx = Context { donor: None };
        for (i, mut c) in bodies(&cfg, 40).into_iter().enumerate() {
            let mut rng = Rng::new(22, 0, i);
            if !split_muscle(&mut c, &cfg, &mut rng, &cx) {
                continue;
            }
            let count = c.muscles.len();
            assert!(fuse_similar_muscles(&mut c, &cfg, &mut rng, &cx));
            assert_eq!(c.muscles.len(), count - 1);
        }
    }

    #[test]
    fn antagonist_turns_the_child_bone_against_its_closer() {
        let applied = each_change(add_antagonist, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            let opener = *after.muscles.last().unwrap();
            let k = opener.bone_a as usize;
            let joint = before.bones[k].a;
            assert!(!branch(before, k).contains(&(opener.bone_b as usize)));
            let opening = torque(before, &opener);
            assert!(opening.abs() >= MIN_TORQUE);
            // A closer at the child's joint turns it the other way, half a
            // cycle apart.
            let closer = before.muscles.iter().any(|m| {
                [*m, flipped(m)].iter().any(|m| {
                    let q = before.bones[m.bone_b as usize];
                    m.bone_a as usize == k
                        && (q.a == joint || q.b == joint)
                        && torque(before, m) * opening < 0.0
                        && turn(m.phase, opener.phase).abs() > 0.499
                })
            });
            assert!(closer);
        });
        assert!(applied >= 90, "applied {applied}");
    }

    #[test]
    fn swap_muscle_routes_exchanges_two_destinations() {
        let applied = each_change(swap_muscle_routes, |before, after| {
            let changed = differing(&before.muscles, &after.muscles);
            assert_eq!(changed.len(), 2);
            let [i, j] = [changed[0], changed[1]];
            let (p, q) = (before.muscles[i], before.muscles[j]);
            let (r, s) = (after.muscles[i], after.muscles[j]);
            assert_eq!((r.bone_a, r.bone_b), (p.bone_a, q.bone_b));
            assert_eq!((s.bone_a, s.bone_b), (q.bone_a, p.bone_b));
            assert!(same_timing(&p, &r) && same_timing(&q, &s));
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn fan_spreads_close_anchors_evenly() {
        let applied = each_change(fan_muscle_attachments, |before, after| {
            let changed = differing(&before.muscles, &after.muscles);
            assert!(!changed.is_empty());
            // The moved ends share one bone and sit at (k + 0.5) / n.
            let mut moved = Vec::new();
            for &i in &changed {
                let (p, q) = (before.muscles[i], after.muscles[i]);
                assert_eq!((p.bone_a, p.bone_b), (q.bone_a, q.bone_b));
                assert!(same_timing(&p, &q));
                if p.anchor_a != q.anchor_a {
                    moved.push((q.bone_a, q.anchor_a));
                }
                if p.anchor_b != q.anchor_b {
                    moved.push((q.bone_b, q.anchor_b));
                }
            }
            assert!(!moved.is_empty());
            assert!(moved.iter().all(|e| e.0 == moved[0].0));
            let even = (moved.len().max(2)..=64).any(|n| {
                moved.iter().all(|&(_, anchor)| {
                    let slot = anchor * n as f32 - 0.5;
                    (slot - slot.round()).abs() < 1e-4
                })
            });
            assert!(even, "{moved:?}");
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn relay_routes_a_muscle_through_a_bone_between() {
        let applied = each_change(relay_muscle, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            let kept = &after.muscles[..before.muscles.len()];
            let changed = differing(&before.muscles, kept);
            assert_eq!(changed.len(), 1);
            let old = before.muscles[changed[0]];
            let (first, second) = (after.muscles[changed[0]], *after.muscles.last().unwrap());
            let path = path_between(before, old.bone_a as usize, old.bone_b as usize);
            assert!(path.contains(&(first.bone_b as usize)));
            assert_eq!(
                (first.bone_a, first.bone_b, second.bone_a, second.bone_b),
                (old.bone_a, second.bone_a, first.bone_b, old.bone_b)
            );
            assert!(first.phase == old.phase && second.phase == old.phase);
        });
        assert!(applied >= 50, "applied {applied}");
    }

    #[test]
    fn copied_actuation_matches_the_source_limb() {
        let applied = each_change(copy_actuation_to_limb, |_, after| {
            // Some pair of limbs now has every source muscle copied onto the
            // recipient's matching bones.
            let limbs: Vec<_> = (0..after.bones.len())
                .filter(|&b| !is_neck(after, b))
                .map(|b| (b, actuation(after, b)))
                .collect();
            let copied = limbs.iter().any(|(from, (fb, fm))| {
                limbs.iter().any(|(to, (tb, _))| {
                    let map = |b: u32| tb[fb.iter().position(|&x| x == b as usize).unwrap()];
                    from != to
                        && fb.len() == tb.len()
                        && !fb.contains(to)
                        && !tb.contains(from)
                        && !fm.is_empty()
                        && fm.iter().all(|&i| {
                            let s = after.muscles[i];
                            after.muscles.iter().any(|m| {
                                (m.bone_a as usize, m.bone_b as usize)
                                    == (map(s.bone_a), map(s.bone_b))
                                    && (m.anchor_a, m.anchor_b, m.phase)
                                        == (s.anchor_a, s.anchor_b, s.phase)
                            })
                        })
                })
            });
            assert!(copied);
        });
        assert!(applied >= 50, "applied {applied}");
    }

    #[test]
    fn quiet_group_shrinks_one_branch_by_one_factor() {
        let applied = each_change(quiet_muscle_group, |before, after| {
            let changed = differing(&before.muscles, &after.muscles);
            assert!(!changed.is_empty());
            let factors: Vec<f32> = changed
                .iter()
                .map(|&i| {
                    let (p, q) = (before.muscles[i], after.muscles[i]);
                    assert_eq!(p.long, q.long);
                    assert!(same_timing(&p, &q));
                    (q.long - q.short) / (p.long - p.short)
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
