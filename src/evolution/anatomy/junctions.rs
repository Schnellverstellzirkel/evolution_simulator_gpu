//! Operators that restructure junctions and segments of the skeleton.
use super::{
    BoneIds, Context, branch, branch_nodes, child_bones, copy_branch, is_neck, mask_of,
    muscles_on, new_muscle, parent_bones, paths, remove_parts, room, span_of,
};
use crate::config::Config;
use crate::evolution::{Bone, Bounded, Creature, MAX_MUSCLES, Muscle, NodeGene, Rng};

/// Where three or more bones meet, puts a short new bone between the node
/// and a new node, and moves some of the child branches to the new node, so
/// a crowded junction becomes two joints (a shoulder and a hip region).
pub(crate) fn split_crowded_joint(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let children = child_bones(c);
    let crowded: BoneIds = (1..c.nodes.len())
        .filter(|&n| children[n].len() >= 2)
        .collect();
    if crowded.is_empty() || !room(c, cfg, 1, 0) {
        return false;
    }
    let node = crowded[rng.index(crowded.len())];
    // At least one child branch moves and at least one stays.
    let mut kids = children[node];
    for i in 0..kids.len() {
        let j = i + rng.index(kids.len() - i);
        kids.swap(i, j);
    }
    let moved = &kids[..1 + rng.index(kids.len() - 1)];
    // The new bone points toward the moved branches and is a fraction of
    // their length. The moved branches slide out by it and keep their shape.
    let mut toward = [0.0; 2];
    let mut length = 0.0;
    for &b in moved {
        toward = add(toward, sub(pos(c, c.bones[b].b as usize), pos(c, node)));
        length += c.bones[b].rest_length / moved.len() as f32;
    }
    let offset = scale(unit(toward), (0.3 * length).clamp(0.04, 0.25));
    let joint = add_node(c, node, add(pos(c, node), offset));
    add_narrow_bone(c, node, joint, rng);
    for &b in moved {
        c.bones[b].a = joint as u32;
        shift_branch(c, b, offset);
    }
    true
}

/// Collapses a short bone between two junctions (its child node has two or
/// more child bones) so its child branches meet at the parent node. Muscles
/// on the removed bone move to a neighbouring bone or go.
pub(crate) fn merge_branch_joints(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let children = child_bones(c);
    let parents = parent_bones(c);
    let eligible: BoneIds = (0..c.bones.len())
        .filter(|&j| !is_neck(c, j) && children[c.bones[j].b as usize].len() >= 2)
        .collect();
    if eligible.is_empty() {
        return false;
    }
    // The shorter of two random candidates goes.
    let x = eligible[rng.index(eligible.len())];
    let y = eligible[rng.index(eligible.len())];
    let bone = if c.bones[x].rest_length <= c.bones[y].rest_length {
        x
    } else {
        y
    };
    let (a, b) = (c.bones[bone].a as usize, c.bones[bone].b as usize);
    if parents[a].is_none() {
        return false;
    }
    let offset = sub(pos(c, a), pos(c, b));
    for &child in &children[b] {
        c.bones[child].a = a as u32;
        shift_branch(c, child, offset);
    }
    // Ends on the removed node move to the joint at the tip of the bone
    // above. A muscle that then joins that node to itself or to a node a
    // bone away goes with `repair`.
    for m in &mut c.muscles {
        if m.node_a as usize == b {
            m.node_a = a as u32;
        }
        if m.node_b as usize == b {
            m.node_b = a as u32;
        }
    }
    c.muscles.retain(|m| m.node_a != m.node_b);
    remove_parts(c, &[bone], &[b]);
    true
}

/// Copies a trunk bone (one with child bones) together with the leaf limbs
/// on its child node and their muscles, and inserts the copy after it in the
/// chain: a route to segmented, many-legged bodies.
pub(crate) fn repeat_body_segment(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let children = child_bones(c);
    let parents = parent_bones(c);
    // A leaf limb is a child branch without junctions: a chain to one tip.
    let limbs_at = |node: usize| -> BoneIds {
        children[node]
            .iter()
            .copied()
            .filter(|&l| {
                branch(c, l)
                    .iter()
                    .all(|&x| children[c.bones[x].b as usize].len() <= 1)
            })
            .collect()
    };
    let trunks: BoneIds = (0..c.bones.len())
        .filter(|&j| !is_neck(c, j) && !limbs_at(c.bones[j].b as usize).is_empty())
        .collect();
    if trunks.is_empty() {
        return false;
    }
    let trunk = trunks[rng.index(trunks.len())];
    let (a, b) = (c.bones[trunk].a as usize, c.bones[trunk].b as usize);
    let Some(above) = parents[a] else {
        return false;
    };
    let limbs = limbs_at(b);
    let segment: BoneIds = std::iter::once(trunk)
        .chain(limbs.iter().flat_map(|&l| branch(c, l)))
        .collect();
    if !room(c, cfg, segment.len(), muscles_on(c, &segment, false).len()) {
        return false;
    }
    let phase = rng.index(4) as f32 * 0.25;
    let offset = sub(pos(c, b), pos(c, a));
    let paths = paths(c);
    let above_top = c.bones[above].a;
    // The copy continues the trunk bone in the same direction.
    let node = add_node(c, b, add(pos(c, b), offset));
    c.bones.push(Bone {
        a: b as u32,
        b: node as u32,
        ..c.bones[trunk]
    });
    // Muscles across the trunk's upper joint get a copy across the new one.
    let upper = mask_of(&[trunk, above]);
    let old_count = c.muscles.len();
    for i in 0..old_count {
        let m = c.muscles[i];
        if span_of(&paths, &m) != upper {
            continue;
        }
        // Its ends are the top of the bone above and the trunk's tip; the
        // copy runs from the trunk's top to the copy's tip.
        let new = Muscle {
            node_a: if m.node_a == above_top { a as u32 } else { node as u32 },
            node_b: if m.node_b == above_top { a as u32 } else { node as u32 },
            phase: (m.phase + phase).rem_euclid(1.0),
            ..m
        };
        c.muscles.push(new);
    }
    // The rest of the body below the trunk now hangs from the copy, and the
    // muscles that worked that joint move with it.
    for &child in &children[b] {
        if limbs.contains(&child) {
            continue;
        }
        c.bones[child].a = node as u32;
        shift_branch(c, child, offset);
        let joint = mask_of(&[trunk, child]);
        for m in &mut c.muscles[..old_count] {
            // Across the trunk and this child before: from the trunk's top
            // to the child's tip, and now from the trunk's tip.
            if span_of(&paths, m) == joint {
                if m.node_a == a as u32 {
                    m.node_a = b as u32;
                } else if m.node_b == a as u32 {
                    m.node_b = b as u32;
                }
            }
        }
    }
    for &limb in &limbs {
        copy_branch(c, cfg, limb, node, |p| add(p, offset), false, phase);
    }
    true
}

/// Turns a limb tip into a heel and a toe: two short bones from the tip, one
/// pointing forward and one back, with their own joint ranges and a muscle
/// from each to the top of the tip bone.
pub(crate) fn grow_heel_toe(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let children = child_bones(c);
    let parents = parent_bones(c);
    let tips: BoneIds = (1..c.nodes.len())
        .filter(|&n| children[n].is_empty())
        .collect();
    if tips.is_empty() || !room(c, cfg, 2, 2) {
        return false;
    }
    let tip = tips[rng.index(tips.len())];
    let Some(leg) = parents[tip] else {
        return false;
    };
    let near = muscles_on(c, &[leg], false);
    let template = (!near.is_empty()).then(|| c.muscles[near[rng.index(near.len())]]);
    // Forward is +x, the direction distance is scored in.
    for direction in [1.0, -1.0] {
        let length = (rng.range(0.2, 0.5) * c.bones[leg].rest_length).max(0.04);
        let at = add(pos(c, tip), [direction * length, 0.0]);
        let node = add_node(c, tip, at);
        add_narrow_bone(c, tip, node, rng);
        // From the toe or heel to the top of the leg bone: across its joint
        // and the leg's own.
        let m = new_muscle(node, c.bones[leg].a as usize, template.as_ref(), rng);
        c.muscles.push(m);
    }
    true
}

/// Grows a short bone with a narrow joint range from a joint node, and moves
/// one attachment of a muscle on a bone at that joint onto the new bone's
/// tip. A lever that changes the muscle's leverage (like an elbow process).
pub(crate) fn grow_lever_spur(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let children = child_bones(c);
    let parents = parent_bones(c);
    let joints: BoneIds = (1..c.nodes.len())
        .filter(|&n| !children[n].is_empty())
        .collect();
    if joints.is_empty() || !room(c, cfg, 1, 0) {
        return false;
    }
    let joint = joints[rng.index(joints.len())];
    let Some(above) = parents[joint] else {
        return false;
    };
    let at_joint: BoneIds = std::iter::once(above)
        .chain(children[joint].iter().copied())
        .collect();
    // Muscle ends on a node of a bone at the joint (but never one that would
    // join the spur to the joint itself): (muscle, whether it is end a).
    let ends: Bounded<(usize, bool), { 2 * MAX_MUSCLES }> = c
        .muscles
        .iter()
        .enumerate()
        .flat_map(|(i, m)| [(i, true, m.node_a, m.node_b), (i, false, m.node_b, m.node_a)])
        .filter(|&(_, _, node, other)| {
            other as usize != joint
                && at_joint
                    .iter()
                    .any(|&b| c.bones[b].a == node || c.bones[b].b == node)
        })
        .map(|(i, first, _, _)| (i, first))
        .collect();
    if ends.is_empty() {
        return false;
    }
    let (muscle, first) = ends[rng.index(ends.len())];
    // The spur continues the bone above past the joint, turned by up to one
    // radian. Its narrow range keeps it nearly rigid with that bone.
    let along = unit(sub(pos(c, joint), pos(c, c.bones[above].a as usize)));
    let (sin, cos) = rng.range(-1.0, 1.0).sin_cos();
    let direction = [
        along[0] * cos - along[1] * sin,
        along[0] * sin + along[1] * cos,
    ];
    let length = (rng.range(0.15, 0.35) * c.bones[above].rest_length).clamp(0.04, 0.3);
    let node = add_node(c, joint, add(pos(c, joint), scale(direction, length)));
    add_narrow_bone(c, joint, node, rng);
    let m = &mut c.muscles[muscle];
    if first {
        m.node_a = node as u32;
    } else {
        m.node_b = node as u32;
    }
    true
}

/// Reflects a branch across the line of its root bone and swaps and negates
/// the joint limits inside it, so the limb bends the other way. Muscles stay
/// on their nodes, so they keep their roles.
pub(crate) fn reverse_bend(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let children = child_bones(c);
    // The root bone lies on the mirror line and keeps its joint, so a branch
    // needs a bone below the root to change.
    let roots: BoneIds = (0..c.bones.len())
        .filter(|&j| !is_neck(c, j) && !children[c.bones[j].b as usize].is_empty())
        .collect();
    if roots.is_empty() {
        return false;
    }
    let root = roots[rng.index(roots.len())];
    let bones = branch(c, root);
    let origin = pos(c, c.bones[root].a as usize);
    let line = unit(sub(pos(c, c.bones[root].b as usize), origin));
    for node in branch_nodes(c, &bones[1..]) {
        let v = sub(pos(c, node), origin);
        let along = v[0] * line[0] + v[1] * line[1];
        let [x, y] = sub(add(origin, scale(line, 2.0 * along)), v);
        c.nodes[node].x = x;
        c.nodes[node].y = y;
    }
    // Below the root both a bone and its reference bone are reflected, so
    // every joint angle there changes sign.
    for &j in &bones[1..] {
        let bone = &mut c.bones[j];
        (bone.min_angle, bone.max_angle) = (-bone.max_angle, -bone.min_angle);
    }
    true
}

fn pos(c: &Creature, node: usize) -> [f32; 2] {
    [c.nodes[node].x, c.nodes[node].y]
}

fn add(p: [f32; 2], q: [f32; 2]) -> [f32; 2] {
    [p[0] + q[0], p[1] + q[1]]
}

fn sub(p: [f32; 2], q: [f32; 2]) -> [f32; 2] {
    [p[0] - q[0], p[1] - q[1]]
}

fn scale(p: [f32; 2], s: f32) -> [f32; 2] {
    [p[0] * s, p[1] * s]
}

/// The direction of `v`, or straight down when `v` is too short to have one.
fn unit(v: [f32; 2]) -> [f32; 2] {
    let length = v[0].hypot(v[1]);
    if length > 1.0e-6 {
        scale(v, 1.0 / length)
    } else {
        [0.0, -1.0]
    }
}

/// Adds a node at `at` with the size and grip of node `like`.
fn add_node(c: &mut Creature, like: usize, at: [f32; 2]) -> usize {
    c.nodes.push(NodeGene {
        x: at[0],
        y: at[1],
        ..c.nodes[like]
    });
    c.nodes.len() - 1
}

/// Adds a bone from node `a` to node `b`, as long as they are apart in the
/// pose, with a narrow joint range. Returns its index.
fn add_narrow_bone(c: &mut Creature, a: usize, b: usize, rng: &mut Rng) -> usize {
    let [dx, dy] = sub(pos(c, b), pos(c, a));
    c.bones.push(Bone {
        min_angle: -rng.range(0.1, 0.4),
        max_angle: rng.range(0.1, 0.4),
        ..Bone::new(a as u32, b as u32, dx.hypot(dy))
    });
    c.bones.len() - 1
}

/// Moves every node below `bone` (the child nodes of its branch) by `offset`.
fn shift_branch(c: &mut Creature, bone: usize, offset: [f32; 2]) {
    for node in branch_nodes(c, &branch(c, bone)) {
        c.nodes[node].x += offset[0];
        c.nodes[node].y += offset[1];
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Operator, degree, tests::bodies};
    use super::*;

    /// Runs `op` on 160 grown bodies, checks every changed body with `check`
    /// (before, after) and returns how many it applied to. A body it does not
    /// apply to must come back unchanged.
    fn run(op: Operator, check: impl Fn(&Creature, &Creature)) -> usize {
        let cfg = Config::default();
        let mut applied = 0;
        for (i, body) in bodies(&cfg, 160).into_iter().enumerate() {
            let mut c = body.clone();
            let mut rng = Rng::new(21, 0, i);
            let cx = Context { donor: None };
            if op(&mut c, &cfg, &mut rng, &cx) {
                applied += 1;
                check(&body, &c);
            } else {
                assert_eq!(format!("{body:?}"), format!("{c:?}"));
            }
        }
        applied
    }

    /// Whether a bone is as long in the pose as its rest length, so parts
    /// that moved kept their shape.
    fn fits(c: &Creature, bone: &Bone) -> bool {
        let [dx, dy] = sub(pos(c, bone.b as usize), pos(c, bone.a as usize));
        (dx.hypot(dy) - bone.rest_length).abs() < 1.0e-3
    }

    fn all_fit(c: &Creature) -> bool {
        c.bones.iter().all(|b| fits(c, b))
    }

    #[test]
    fn split_crowded_joint_moves_branches_to_a_new_joint() {
        let applied = run(split_crowded_joint, |before, c| {
            assert_eq!(c.nodes.len(), before.nodes.len() + 1);
            assert_eq!(c.muscles.len(), before.muscles.len());
            let new = c.bones.last().unwrap();
            let (node, joint) = (new.a as usize, new.b as usize);
            let (was, now) = (child_bones(before), child_bones(c));
            assert!(was[node].len() >= 2);
            assert!(!now[joint].is_empty());
            assert!(now[node].len() >= 2, "one branch stays beside the new bone");
            assert_eq!(now[node].len() + now[joint].len(), was[node].len() + 1);
            assert!(all_fit(c));
        });
        eprintln!("split_crowded_joint: {applied} of 160");
        assert!(applied >= 40, "applied to {applied} of 160");
    }

    #[test]
    fn merge_branch_joints_removes_one_bone_and_keeps_the_rest() {
        let applied = run(merge_branch_joints, |before, c| {
            assert_eq!(c.nodes.len(), before.nodes.len() - 1);
            assert_eq!(c.bones.len(), before.bones.len() - 1);
            let sorted = |c: &Creature| {
                let mut l: Vec<f32> = c.bones.iter().map(|b| b.rest_length).collect();
                l.sort_by(f32::total_cmp);
                l
            };
            let (was, now) = (sorted(before), sorted(c));
            assert!((0..was.len()).any(|k| {
                let mut w = was.clone();
                w.remove(k);
                w == now
            }));
            assert_eq!(degree(c, 0), 1, "the head keeps only its neck");
            assert!(all_fit(c));
        });
        eprintln!("merge_branch_joints: {applied} of 160");
        assert!(applied >= 40, "applied to {applied} of 160");
    }

    #[test]
    fn repeat_body_segment_inserts_a_copy_of_the_trunk_bone() {
        let applied = run(repeat_body_segment, |before, c| {
            let added = c.nodes.len() - before.nodes.len();
            assert!(added >= 2, "a trunk bone and at least one limb");
            assert_eq!(c.bones.len(), before.bones.len() + added);
            assert!(c.muscles.len() >= before.muscles.len());
            let copy = c.bones[before.bones.len()];
            let trunk = before
                .bones
                .iter()
                .find(|t| t.b == copy.a)
                .expect("the copy starts where the trunk bone ends");
            assert_eq!(copy.rest_length, trunk.rest_length);
            assert_eq!(
                (copy.min_angle, copy.max_angle),
                (trunk.min_angle, trunk.max_angle)
            );
            assert!(!child_bones(c)[copy.b as usize].is_empty());
            // copy_branch keeps copied nodes above the ground line, so only
            // the old bones (some moved below the copy) and the copy must fit.
            assert!(c.bones[..=before.bones.len()].iter().all(|b| fits(c, b)));
        });
        eprintln!("repeat_body_segment: {applied} of 160");
        assert!(applied >= 80, "applied to {applied} of 160");
    }

    #[test]
    fn grow_heel_toe_adds_two_bones_at_a_former_leaf() {
        let applied = run(grow_heel_toe, |before, c| {
            assert_eq!(c.nodes.len(), before.nodes.len() + 2);
            assert_eq!(c.muscles.len(), before.muscles.len() + 2);
            let n = c.bones.len();
            let (toe, heel) = (c.bones[n - 2], c.bones[n - 1]);
            assert_eq!(toe.a, heel.a);
            let tip = toe.a as usize;
            assert!(child_bones(before)[tip].is_empty(), "a former leaf");
            assert!(pos(c, toe.b as usize)[0] > pos(c, tip)[0]);
            assert!(pos(c, heel.b as usize)[0] < pos(c, tip)[0]);
            let leg = parent_bones(c)[tip].unwrap();
            for bone in [n - 2, n - 1] {
                let (toe, top) = (c.bones[bone].b, c.bones[leg].a);
                assert!(c.muscles.iter().any(|m| m.node_a == toe && m.node_b == top));
            }
            assert!(all_fit(c));
        });
        eprintln!("grow_heel_toe: {applied} of 160");
        assert!(applied >= 120, "applied to {applied} of 160");
    }

    #[test]
    fn grow_lever_spur_moves_one_muscle_end_to_the_spur_tip() {
        let applied = run(grow_lever_spur, |before, c| {
            assert_eq!(c.nodes.len(), before.nodes.len() + 1);
            assert_eq!(c.muscles.len(), before.muscles.len());
            let bone = *c.bones.last().unwrap();
            let tip = bone.b;
            assert!(-0.4 <= bone.min_angle && bone.max_angle <= 0.4);
            assert!(!child_bones(before)[bone.a as usize].is_empty());
            // One muscle end sits on the spur's tip, and only that one.
            let on_spur = c
                .muscles
                .iter()
                .map(|m| usize::from(m.node_a == tip) + usize::from(m.node_b == tip))
                .sum::<usize>();
            assert_eq!(on_spur, 1);
            assert!(all_fit(c));
        });
        eprintln!("grow_lever_spur: {applied} of 160");
        assert!(applied >= 120, "applied to {applied} of 160");
    }

    #[test]
    fn reverse_bend_negates_and_swaps_the_joint_limits_in_the_branch() {
        let applied = run(reverse_bend, |before, c| {
            assert_eq!(c.nodes.len(), before.nodes.len());
            assert_eq!(c.muscles.len(), before.muscles.len());
            let flipped = |j: usize| {
                let (old, new) = (before.bones[j], c.bones[j]);
                Bone {
                    min_angle: -old.max_angle,
                    max_angle: -old.min_angle,
                    ..old
                } == new
            };
            // Some root bone keeps its joint while every bone below it flips
            // and every other bone stays.
            let root = (0..c.bones.len()).find(|&r| {
                let below = &branch(c, r)[1..];
                !below.is_empty()
                    && (0..c.bones.len()).all(|j| {
                        if below.contains(&j) {
                            flipped(j)
                        } else {
                            c.bones[j] == before.bones[j]
                        }
                    })
            });
            assert!(root.is_some());
            assert!(all_fit(c));
        });
        eprintln!("reverse_bend: {applied} of 160");
        assert!(applied >= 120, "applied to {applied} of 160");
    }
}

/// Turns bone `j`'s branch about its pivot by `angle` (counterclockwise),
/// lifting the body if a node would go below the ground.
fn turn_branch(c: &mut Creature, j: usize, angle: f32) {
    let pivot = c.nodes[c.bones[j].a as usize];
    let (sin, cos) = angle.sin_cos();
    for n in branch_nodes(c, &branch(c, j)) {
        let (dx, dy) = (c.nodes[n].x - pivot.x, c.nodes[n].y - pivot.y);
        c.nodes[n].x = pivot.x + dx * cos - dy * sin;
        c.nodes[n].y = pivot.y + dx * sin + dy * cos;
    }
    let low = c.nodes.iter().map(|n| n.y).fold(0.0, f32::min);
    for n in &mut c.nodes {
        n.y -= low;
    }
}

/// A joint (not the neck) and one of its stops, the stop as an angle from
/// the starting pose.
fn joint_and_stop(c: &Creature, rng: &mut Rng) -> Option<(usize, f32)> {
    let joints: BoneIds = (0..c.bones.len())
        .filter(|&j| !is_neck(c, j) && c.bones[j].max_angle - c.bones[j].min_angle > 0.05)
        .collect();
    let j = *joints.get(rng.index(joints.len().max(1)))?;
    let b = c.bones[j];
    Some((
        j,
        if rng.unit() < 0.5 {
            b.min_angle
        } else {
            b.max_angle
        },
    ))
}

/// Starts a joint near one of its stops and measures its range from there, so
/// the stops stay where they were and only the starting pose changes. The
/// best elites of a 120-generation save ran with their joints 0.3 to 0.6 rad
/// from the pose their genome starts in, so every trial began by folding.
pub(crate) fn pose_joint_at_stop(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some((j, stop)) = joint_and_stop(c, rng) else {
        return false;
    };
    let angle = stop * rng.range(0.5, 1.0);
    turn_branch(c, j, angle);
    c.bones[j].min_angle -= angle;
    c.bones[j].max_angle -= angle;
    true
}

/// Sets a joint against one of its stops and leaves it only a small flex back
/// from it, so the skeleton holds a braced shape by itself. The best elites of
/// a 120-generation save held 40 to 80% of their joints against a stop with
/// muscles (as much steady force as oscillating force) and hopped as one
/// rigid frame; mid-ranked elites held 8 to 18% and slid.
pub(crate) fn brace_joint(c: &mut Creature, _cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let Some((j, stop)) = joint_and_stop(c, rng) else {
        return false;
    };
    turn_branch(c, j, stop);
    let flex = rng.range(0.03, 0.2);
    (c.bones[j].min_angle, c.bones[j].max_angle) = if stop > 0.0 {
        (-flex, 0.0)
    } else {
        (0.0, flex)
    };
    true
}
