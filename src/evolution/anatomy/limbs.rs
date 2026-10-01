//! Operators that copy, grow, fuse, move and reshape whole limbs.
use super::{
    BoneIds, Context, MuscleIds, branch, branch_nodes, child_bones, copy_branch, is_neck,
    mask_of, muscles_on, new_muscle, parent_bones, paths, remove_parts, room, span_of,
};
use crate::config::Config;
use crate::evolution::{
    Bone, Bounded, Creature, MAX_MUSCLES, MAX_NODES, Muscle, NodeGene, Rng, body_extent,
    max_bone_length,
};

/// Two bones count as nearly aligned when the cosine of the angle between
/// them is at least this (within about 26 degrees).
const ALIGNED: f32 = 0.9;

/// Copies a complete branch (several bones, their joints, the muscles inside
/// it and the muscles from its root to the bone above) onto the same joint or
/// another joint, mirrored or not, with the copied muscles shifted by one of
/// 0, 1/4, 1/2 or 3/4 of a cycle. A working bent leg becomes a second leg.
pub(crate) fn copy_limb(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let roots: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| room(c, cfg, branch(c, b).len(), 0))
        .collect();
    let Some(root) = pick(&roots, rng) else {
        return false;
    };
    let joint = c.bones[root].a as usize;
    let at = if rng.unit() < 0.5 {
        joint
    } else {
        1 + rng.index(c.nodes.len() - 1)
    };
    let mirror = rng.unit() < 0.5;
    let phase = rng.index(4) as f32 * 0.25;
    let (from, to) = (c.nodes[joint], c.nodes[at]);
    let side = if mirror { -1.0 } else { 1.0 };
    let place = |[x, y]: [f32; 2]| [to.x + side * (x - from.x), to.y + y - from.y];
    copy_branch(c, cfg, root, at, place, mirror, phase).is_some()
}

/// Extends a limb tip with a short bone (a fraction of the tip bone), a joint
/// with a narrow range, and a muscle from the new node to the top of the tip
/// bone, timed like a muscle near it. A direct route to ankles and toes.
pub(crate) fn grow_actuated_tip(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    let children = child_bones(c);
    let tips: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| children[c.bones[b].b as usize].is_empty())
        .collect();
    let Some(tip) = pick(&tips, rng) else {
        return false;
    };
    let bone = c.bones[tip];
    let (a, b) = (c.nodes[bone.a as usize], c.nodes[bone.b as usize]);
    let length = (bone.rest_length * rng.range(0.25, 0.5)).max(0.03);
    let angle = (b.y - a.y).atan2(b.x - a.x) + rng.range(-1.5, 1.5);
    let [x, y] = clamped(b.x + length * angle.cos(), b.y + length * angle.sin());
    c.nodes.push(NodeGene { x, y, ..b });
    let mut toe = Bone::new(bone.b, c.nodes.len() as u32 - 1, length);
    narrow(&mut toe, rng);
    c.bones.push(toe);
    // A muscle from the toe's tip to the top of the tip bone: across the toe
    // joint and the tip bone's own.
    let template = nearby_muscle(c, &[tip], rng);
    let m = new_muscle(c.nodes.len() - 1, bone.a as usize, template.as_ref(), rng);
    c.muscles.push(m);
    true
}

/// Splits `bone` at `t` of its length from its pivot end: a new node there,
/// the bone shortened to the pivot's side, and a new bone with a narrow
/// joint range from the new node to the old child node. The organ stays
/// where it was, on whichever part holds that point, and muscles keep their
/// nodes. Returns the new node and the new bone.
pub(super) fn split_bone_at(c: &mut Creature, first: usize, t: f32, rng: &mut Rng) -> (usize, usize) {
    let old = c.bones[first];
    let (a, b) = (c.nodes[old.a as usize], c.nodes[old.b as usize]);
    let mid = c.nodes.len() as u32;
    c.nodes.push(NodeGene {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        diameter: (a.diameter + b.diameter) * 0.5,
        friction: (a.friction + b.friction) * 0.5,
    });
    let second = c.bones.len();
    let mut lower = Bone::new(mid, old.b, old.rest_length * (1.0 - t));
    narrow(&mut lower, rng);
    c.bones[first].b = mid;
    c.bones[first].rest_length = old.rest_length * t;
    if old.organ_mass > 0.0 {
        if old.organ_at <= t {
            c.bones[first].organ_at = old.organ_at / t;
        } else {
            c.bones[first].organ_mass = 0.0;
            c.bones[first].organ_at = 0.5;
            lower.organ_mass = old.organ_mass;
            lower.organ_at = (old.organ_at - t) / (1.0 - t);
        }
    }
    c.bones.push(lower);
    (mid as usize, second)
}

/// Splits a bone at a random point between 30% and 70% of its length. The new
/// joint starts with a narrow range, muscles stay on the nodes they were on,
/// and a muscle across the new joint (timed like one nearby) makes it an
/// elbow or knee under control instead of a floppy hinge.
pub(crate) fn split_bone_actuated(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 1, 1) {
        return false;
    }
    // Both parts must stay at least 3 cm long.
    let long: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| c.bones[b].rest_length >= 0.1)
        .collect();
    let Some(first) = pick(&long, rng) else {
        return false;
    };
    let t = rng.range(0.3, 0.7);
    let old = c.bones[first];
    let (_, second) = split_bone_at(c, first, t, rng);
    let template = nearby_muscle(c, &[first, second], rng);
    // From one end of the old bone to the other: across both parts.
    let m = new_muscle(old.a as usize, old.b as usize, template.as_ref(), rng);
    c.muscles.push(m);
    true
}

/// Fuses two nearly aligned bones that meet at a node with no other bone
/// (not the head or the neck) into one bone; muscles on either keep their
/// place on the body. Evolution can decide which regions stay rigid.
pub(crate) fn fuse_bones(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    if c.nodes.len() <= 3 {
        return false;
    }
    let parents = parent_bones(c);
    let children = child_bones(c);
    let pairs: Bounded<(usize, usize), MAX_NODES> = (1..c.nodes.len())
        .filter_map(|joint| {
            let upper = parents[joint]?;
            let &[lower] = &children[joint][..] else {
                return None;
            };
            let a = c.nodes[c.bones[upper].a as usize];
            let (n, m) = (c.nodes[joint], c.nodes[c.bones[lower].b as usize]);
            let (u, v) = ([n.x - a.x, n.y - a.y], [m.x - n.x, m.y - n.y]);
            let cosine = (u[0] * v[0] + u[1] * v[1]) / (u[0].hypot(u[1]) * v[0].hypot(v[1]));
            let length = (m.x - a.x).hypot(m.y - a.y);
            (!is_neck(c, upper) && cosine >= ALIGNED && length <= max_bone_length())
                .then_some((upper, lower))
        })
        .collect();
    let Some((upper, lower)) = pick(&pairs, rng) else {
        return false;
    };
    fuse_pair(c, upper, lower);
    true
}

/// Fuses bone `upper` and the one bone `lower` below it into one bone from
/// the top of `upper` to the tip of `lower`. A muscle that ended at the joint
/// between them ends at the nearer end of the fused bone.
pub(super) fn fuse_pair(c: &mut Creature, upper: usize, lower: usize) {
    let joint = c.bones[upper].b;
    let end = c.bones[lower].b;
    let top = c.bones[upper].a;
    let start = c.nodes[top as usize];
    let stop = c.nodes[end as usize];
    let d = [stop.x - start.x, stop.y - start.y];
    let length = d[0].hypot(d[1]);
    // Where a point of the body lies along the fused bone.
    let along = |p: [f32; 2]| {
        (((p[0] - start.x) * d[0] + (p[1] - start.y) * d[1]) / (length * length)).clamp(0.0, 1.0)
    };
    let at = |b: Bone, t: f32| {
        let (p, q) = (c.nodes[b.a as usize], c.nodes[b.b as usize]);
        [p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t]
    };
    let organ = [upper, lower]
        .into_iter()
        .map(|b| c.bones[b])
        .find(|b| b.organ_mass > 0.0)
        .map(|b| (b.organ_mass, along(at(b, b.organ_at))));
    let joint_at = along([c.nodes[joint as usize].x, c.nodes[joint as usize].y]);
    for m in &mut c.muscles {
        let to = if joint_at < 0.5 { top } else { end };
        if m.node_a == joint {
            m.node_a = to;
        }
        if m.node_b == joint {
            m.node_b = to;
        }
    }
    let fused = &mut c.bones[upper];
    fused.b = end;
    fused.rest_length = length.max(0.03);
    if let Some((mass, at)) = organ {
        fused.organ_mass = mass;
        fused.organ_at = at;
    }
    // Muscles that joined a node to itself, or two nodes a bone apart, are
    // for `repair` to drop; those on the joint are gone with its node.
    remove_parts(c, &[lower], &[joint as usize]);
}

/// Moves a branch, with its internal shape and muscles, to another node of
/// the body (not inside the branch, not the head). Muscles from the branch
/// root to the old bone above move to the bone above the new node.
pub(crate) fn relocate_limb(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let moves: Bounded<(u8, u8), { MAX_NODES * MAX_NODES }> = limb_roots(c)
        .into_iter()
        .flat_map(|root| {
            let inside = branch_nodes(c, &branch(c, root));
            let from = c.bones[root].a as usize;
            (1..c.nodes.len())
                .filter(move |n| *n != from && !inside.contains(n))
                .map(move |n| (root as u8, n as u8))
        })
        .collect();
    let Some((root, at)) = pick(&moves, rng) else {
        return false;
    };
    let (root, at) = (root as usize, at as usize);
    let from = c.bones[root].a as usize;
    let parents = parent_bones(c);
    let offset = [
        c.nodes[at].x - c.nodes[from].x,
        c.nodes[at].y - c.nodes[from].y,
    ];
    for n in branch_nodes(c, &branch(c, root)) {
        let node = &mut c.nodes[n];
        [node.x, node.y] = clamped(node.x + offset[0], node.y + offset[1]);
    }
    // Muscles across the root joint (from the top of the bone above to the
    // branch) move to the joint at the new node: from the top of the bone
    // above it.
    if let (Some(old), Some(new)) = (parents[from], parents[at]) {
        let (paths, hinge) = (paths(c), mask_of(&[root, old]));
        let (old_top, new_top) = (c.bones[old].a, c.bones[new].a);
        for m in &mut c.muscles {
            if span_of(&paths, m) == hinge {
                if m.node_a == old_top {
                    m.node_a = new_top;
                } else if m.node_b == old_top {
                    m.node_b = new_top;
                }
            }
        }
    }
    c.bones[root].a = at as u32;
    true
}

/// Scales every bone of a branch by one factor (0.7 to 1.4, within the bone
/// limits), so a limb gets longer or shorter without scrambling its parts.
pub(crate) fn reshape_limb(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some(root) = pick(&limb_roots(c), rng) else {
        return false;
    };
    let bones = branch(c, root);
    let lengths = || bones.iter().map(|&b| c.bones[b].rest_length);
    let low = lengths().map(|l| 0.03 / l).fold(0.7, f32::max);
    let high = lengths().map(|l| max_bone_length() / l).fold(1.4, f32::min);
    let factor = rng.range(low.ln(), high.ln()).exp();
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
    true
}

/// Copies a branch of `cx.donor` (another archive elite) onto a node of this
/// body, with the donor's joints, internal muscles and their timing. With
/// even odds it replaces a branch of this body instead of adding one.
pub(crate) fn graft_donor_limb(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    cx: &Context,
) -> bool {
    let Some(donor) = cx.donor else {
        return false;
    };
    let Some(graft) = pick(&limb_roots(donor), rng) else {
        return false;
    };
    let graft_bones = branch(donor, graft);
    let graft_muscles = muscles_on(donor, &graft_bones, true);
    // The branch of this body the graft replaces (none when it adds), and
    // the node the graft goes on.
    let (gone, at) = match pick(&limb_roots(c), rng) {
        Some(root) if rng.unit() < 0.5 => (branch(c, root), c.bones[root].a as usize),
        _ => (BoneIds::new(), 1 + rng.index(c.nodes.len() - 1)),
    };
    let gone_nodes = branch_nodes(c, &gone);
    let gone_muscles = muscles_on(c, &gone, false).len();
    if c.nodes.len() - gone_nodes.len() + graft_bones.len() > cfg.max_nodes.min(MAX_NODES)
        || c.muscles.len() - gone_muscles + graft_muscles.len() > cfg.max_muscles.min(MAX_MUSCLES)
    {
        return false;
    }
    remove_parts(c, &gone, &gone_nodes);
    let at = at - gone_nodes.iter().filter(|&&n| n < at).count();
    let from = donor.nodes[donor.bones[graft].a as usize];
    let to = c.nodes[at];
    let mut node_of = [usize::MAX; MAX_NODES];
    node_of[donor.bones[graft].a as usize] = at;
    for &b in &graft_bones {
        let old = donor.bones[b];
        let n = donor.nodes[old.b as usize];
        let [x, y] = clamped(to.x + n.x - from.x, to.y + n.y - from.y);
        node_of[old.b as usize] = c.nodes.len();
        c.nodes.push(NodeGene { x, y, ..n });
        c.bones.push(Bone {
            a: node_of[old.a as usize] as u32,
            b: node_of[old.b as usize] as u32,
            ..old
        });
    }
    for i in graft_muscles {
        let mut m = donor.muscles[i];
        let (a, b) = (node_of[m.node_a as usize], node_of[m.node_b as usize]);
        if a == usize::MAX || b == usize::MAX {
            continue;
        }
        m.node_a = a as u32;
        m.node_b = b as u32;
        c.muscles.push(m);
    }
    true
}

/// Bones that can start a limb: every bone but the neck.
pub(super) fn limb_roots(c: &Creature) -> BoneIds {
    (0..c.bones.len()).filter(|&b| !is_neck(c, b)).collect()
}

/// A random item of `items`, or `None` when there is none.
pub(super) fn pick<T: Copy>(items: &[T], rng: &mut Rng) -> Option<T> {
    (!items.is_empty()).then(|| items[rng.index(items.len())])
}

/// A starting position moved inside the region where nodes may start.
pub(super) fn clamped(x: f32, y: f32) -> [f32; 2] {
    let extent = body_extent();
    [x.clamp(-extent, extent), y.clamp(0.0, extent)]
}

/// Gives a new joint a narrow range around its starting angle.
pub(super) fn narrow(bone: &mut Bone, rng: &mut Rng) {
    bone.min_angle = -rng.range(0.15, 0.5);
    bone.max_angle = rng.range(0.15, 0.5);
}

/// A muscle on one of `bones` to take the timing from, or any muscle of the
/// body when none is on them.
fn nearby_muscle(c: &Creature, bones: &[usize], rng: &mut Rng) -> Option<Muscle> {
    let near = muscles_on(c, bones, false);
    let pool: MuscleIds = if near.is_empty() {
        (0..c.muscles.len()).collect()
    } else {
        near
    };
    pick(&pool, rng).map(|i| c.muscles[i])
}

#[cfg(test)]
mod tests {
    use super::super::{Operator, tests::bodies};
    use super::*;

    /// Runs `op` on 80 test bodies, checks each changed body with `check`
    /// (before, after) and each unchanged one for equality, and returns how
    /// many it changed.
    fn applied(op: Operator, mut check: impl FnMut(&Creature, &Creature)) -> usize {
        let cfg = Config::default();
        let bodies = bodies(&cfg, 80);
        let donor = bodies[40].clone();
        let cx = Context {
            donor: Some(&donor),
        };
        let mut count = 0;
        for (i, body) in bodies.iter().enumerate() {
            let mut c = body.clone();
            let mut rng = Rng::new(5, 0, i);
            if op(&mut c, &cfg, &mut rng, &cx) {
                check(body, &c);
                count += 1;
            } else {
                assert!(
                    c.nodes == body.nodes && c.bones == body.bones && c.muscles == body.muscles
                );
            }
        }
        count
    }

    fn lengths(c: &Creature, bones: &[usize]) -> Vec<f32> {
        bones.iter().map(|&b| c.bones[b].rest_length).collect()
    }

    #[test]
    fn copy_limb_adds_as_many_bones_as_the_branch_has() {
        let n = applied(copy_limb, |before, after| {
            let added = after.bones.len() - before.bones.len();
            assert!(added >= 1);
            assert_eq!(after.nodes.len() - before.nodes.len(), added);
            assert!(after.muscles.len() >= before.muscles.len());
            let copied: Vec<f32> = after.bones[before.bones.len()..]
                .iter()
                .map(|b| b.rest_length)
                .collect();
            assert!(
                limb_roots(before)
                    .into_iter()
                    .any(|r| lengths(before, &branch(before, r)) == copied)
            );
        });
        assert!(n >= 8, "copy_limb applied to {n} of 80");
    }

    #[test]
    fn grow_actuated_tip_adds_a_short_actuated_bone_at_a_tip() {
        let n = applied(grow_actuated_tip, |before, after| {
            assert_eq!(after.nodes.len(), before.nodes.len() + 1);
            assert_eq!(after.bones.len(), before.bones.len() + 1);
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            let new = before.bones.len();
            let toe = after.bones[new];
            assert!(child_bones(before)[toe.a as usize].is_empty());
            let tip = (0..new).find(|&b| before.bones[b].b == toe.a).unwrap();
            assert!(toe.rest_length <= (0.5 * before.bones[tip].rest_length).max(0.03) + 1e-6);
            assert!(toe.max_angle - toe.min_angle <= 1.0);
            let m = after.muscles.last().unwrap();
            assert_eq!(
                (m.node_a as usize, m.node_b),
                (after.nodes.len() - 1, before.bones[tip].a)
            );
        });
        assert!(n >= 8, "grow_actuated_tip applied to {n} of 80");
    }

    #[test]
    fn split_bone_actuated_keeps_attachments_and_adds_a_muscle_across() {
        let n = applied(split_bone_actuated, |before, after| {
            assert_eq!(after.nodes.len(), before.nodes.len() + 1);
            assert_eq!(after.bones.len(), before.bones.len() + 1);
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            let second = before.bones.len();
            let lower = after.bones[second];
            let first = (0..second).find(|&b| after.bones[b].b == lower.a).unwrap();
            let whole = before.bones[first].rest_length;
            assert_eq!(lower.b, before.bones[first].b);
            assert!((after.bones[first].rest_length + lower.rest_length - whole).abs() < 1e-5);
            let t = after.bones[first].rest_length / whole;
            assert!((0.3 - 1e-4..=0.7 + 1e-4).contains(&t));
            assert!(lower.max_angle - lower.min_angle <= 1.0);
            // Every old attachment stays on the same node.
            for (old, new) in before.muscles.iter().zip(&after.muscles) {
                assert_eq!((old.node_a, old.node_b), (new.node_a, new.node_b));
            }
            // The new muscle runs across both parts of the old bone.
            let m = after.muscles.last().unwrap();
            assert_eq!(
                (m.node_a, m.node_b),
                (before.bones[first].a, before.bones[first].b)
            );
        });
        assert!(n >= 8, "split_bone_actuated applied to {n} of 80");
    }

    #[test]
    fn fuse_bones_removes_one_node_and_one_bone() {
        let n = applied(fuse_bones, |before, after| {
            assert_eq!(after.nodes.len(), before.nodes.len() - 1);
            assert_eq!(after.bones.len(), before.bones.len() - 1);
            assert!(after.muscles.len() <= before.muscles.len());
            let total = |c: &Creature| c.bones.iter().map(|b| b.rest_length).sum::<f32>();
            assert!(total(after) <= total(before) + 1e-4);
            assert!(total(after) >= 0.95 * total(before));
        });
        assert!(n >= 8, "fuse_bones applied to {n} of 80");
    }

    #[test]
    fn relocate_limb_moves_one_branch_root_and_keeps_its_parts() {
        let n = applied(relocate_limb, |before, after| {
            assert_eq!(after.nodes.len(), before.nodes.len());
            assert_eq!(after.muscles.len(), before.muscles.len());
            let moved: Vec<usize> = (0..before.bones.len())
                .filter(|&b| after.bones[b].a != before.bones[b].a)
                .collect();
            assert_eq!(moved.len(), 1);
            let root = moved[0];
            let at = after.bones[root].a as usize;
            assert!(at != 0 && !branch_nodes(before, &branch(before, root)).contains(&at));
            for (x, y) in before.bones.iter().zip(&after.bones) {
                assert_eq!((x.b, x.rest_length), (y.b, y.rest_length));
            }
        });
        assert!(n >= 8, "relocate_limb applied to {n} of 80");
    }

    #[test]
    fn reshape_limb_scales_a_branch_by_one_factor() {
        let n = applied(reshape_limb, |before, after| {
            let changed: Vec<usize> = (0..before.bones.len())
                .filter(|&b| after.bones[b].rest_length != before.bones[b].rest_length)
                .collect();
            let mut limb = branch(before, changed[0]);
            limb.sort();
            assert_eq!(changed, limb);
            let factor = after.bones[changed[0]].rest_length / before.bones[changed[0]].rest_length;
            assert!((0.7 - 1e-4..=1.4 + 1e-4).contains(&factor));
            for &b in &changed {
                let ratio = after.bones[b].rest_length / before.bones[b].rest_length;
                assert!((ratio - factor).abs() < 1e-4);
            }
        });
        assert!(n >= 8, "reshape_limb applied to {n} of 80");
    }

    #[test]
    fn graft_donor_limb_adds_or_replaces_with_a_donor_branch() {
        let cfg = Config::default();
        let donor = bodies(&cfg, 80)[40].clone();
        let (mut added, mut replaced) = (0, 0);
        applied(graft_donor_limb, |before, after| {
            // The body ends with a copy of a whole donor branch.
            let limb = limb_roots(&donor)
                .into_iter()
                .map(|r| branch(&donor, r))
                .filter(|bones| {
                    let tail = after.bones.len().checked_sub(bones.len());
                    tail.is_some_and(|t| {
                        after.bones[t..].iter().zip(bones).all(|(x, &b)| {
                            let y = donor.bones[b];
                            (x.rest_length, x.min_angle, x.max_angle)
                                == (y.rest_length, y.min_angle, y.max_angle)
                        })
                    })
                })
                .max_by_key(|limb| limb.len())
                .expect("the body ends with a donor branch");
            if after.nodes.len() == before.nodes.len() + limb.len() {
                added += 1;
            } else {
                replaced += 1;
            }
        });
        assert!(
            added >= 8 && replaced >= 8,
            "added {added}, replaced {replaced} of 80"
        );
        let cx = Context { donor: None };
        let mut c = donor.clone();
        assert!(!graft_donor_limb(&mut c, &cfg, &mut Rng::new(1, 0, 0), &cx));
    }
}
