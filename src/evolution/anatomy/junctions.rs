//! This file holds eight anatomy operators. Six restructure the junctions and
//! segments of the skeleton and two start a joint near one of its stops.
//! `mod.rs` lists them in `BASE_OPERATORS`. The other anatomy files use the
//! helpers here to add nodes, move or turn branches and keep muscle strokes.
use super::rhythm::leaf_limbs_at;
use super::{
    BoneIds, Context, branch, branch_nodes, child_bones, copy_branch, fit_stroke, is_neck,
    muscles_on, new_muscle, parent_bones, remove_parts, room, span,
};
use crate::config::Config;
use crate::evolution::{Bone, Bounded, Creature, MAX_MUSCLES, Muscle, NodeGene, Rng};

/// Where three or more bones meet, adds a short bone from that node to a new
/// node and moves some of the child branches onto the new node, so one crowded
/// junction becomes two.
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
    let mut kids = BoneIds::from_slice(&children[node]);
    for i in 0..kids.len() {
        let j = i + rng.index(kids.len() - i);
        kids.swap(i, j);
    }
    let moved = &kids[..1 + rng.index(kids.len() - 1)];
    // The new bone points toward the moved branches and is a fraction of
    // their mean length. The moved branches slide out by it and keep their
    // shape.
    let mut toward = [0.0; 2];
    let mut length = 0.0;
    for &b in moved {
        toward = add(toward, sub(pos(c, c.bones[b].b as usize), pos(c, node)));
        length += c.bones[b].rest_length / moved.len() as f32;
    }
    let offset = scale(unit(toward), (0.3 * length).clamp(0.04, 0.25));
    let before = spans(c);
    let joint = add_node(c, node, add(pos(c, node), offset));
    add_narrow_bone(c, node, joint, rng);
    for &b in moved {
        c.bones[b].a = joint as u32;
        shift_branch(c, b, offset);
    }
    keep_strokes(c, &before);
    true
}

/// Removes a bone whose child node is a junction (it has two or more child
/// bones), so those child branches hang from the removed bone's parent node.
/// Muscle ends on the removed bone move to the tip of the bone above it. A
/// muscle that then joins that bone to itself is dropped.
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
    let Some(above) = parents[a] else {
        return false;
    };
    let before = spans(c);
    let offset = sub(pos(c, a), pos(c, b));
    for &child in &children[b] {
        c.bones[child].a = a as u32;
        shift_branch(c, child, offset);
    }
    // Muscle ends on the removed bone move to the joint at the tip of the
    // bone above. `remove_parts` then drops a muscle that joins that bone to
    // itself.
    for m in &mut c.muscles {
        if m.bone_a as usize == bone {
            m.bone_a = above as u32;
            m.anchor_a = 1.0;
        }
        if m.bone_b as usize == bone {
            m.bone_b = above as u32;
            m.anchor_b = 1.0;
        }
    }
    // `keep_strokes` matches muscles by index, so it runs before
    // `remove_parts` drops any.
    keep_strokes(c, &before);
    remove_parts(c, &[bone], &[b]);
    true
}

/// Copies a trunk bone together with the leaf limbs on its child node and their
/// muscles, and inserts the copy after it in the chain. A trunk bone is a bone
/// other than the neck whose child node has leaf limbs. The rest of the body
/// below the trunk then hangs from the copy. Copied muscles shift their phase
/// by the same random number of quarter cycles, from none to three. It builds
/// segmented, many-legged bodies one segment at a time.
pub(crate) fn repeat_body_segment(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let children = child_bones(c);
    let parents = parent_bones(c);
    let trunks: BoneIds = (0..c.bones.len())
        .filter(|&j| {
            !is_neck(c, j) && !leaf_limbs_at(c, &children, c.bones[j].b as usize).is_empty()
        })
        .collect();
    if trunks.is_empty() {
        return false;
    }
    let trunk = trunks[rng.index(trunks.len())];
    let (a, b) = (c.bones[trunk].a as usize, c.bones[trunk].b as usize);
    let Some(above) = parents[a] else {
        return false;
    };
    let limbs = leaf_limbs_at(c, &children, b);
    let segment: BoneIds = std::iter::once(trunk)
        .chain(limbs.iter().flat_map(|&l| branch(c, l)))
        .collect();
    if !room(c, cfg, segment.len(), muscles_on(c, &segment, false).len()) {
        return false;
    }
    let phase = rng.index(4) as f32 * 0.25;
    let offset = sub(pos(c, b), pos(c, a));
    let before = spans(c);
    // The copy continues the trunk bone in the same direction.
    let node = add_node(c, b, add(pos(c, b), offset));
    let copy = c.bones.len();
    c.bones.push(Bone {
        a: b as u32,
        b: node as u32,
        ..c.bones[trunk]
    });
    // Muscles across the trunk's upper joint get a copy across the new one.
    for i in 0..c.muscles.len() {
        let m = c.muscles[i];
        let ends = (m.bone_a as usize, m.bone_b as usize);
        if ends != (trunk, above) && ends != (above, trunk) {
            continue;
        }
        let map = |x: u32| (if x as usize == trunk { copy } else { trunk }) as u32;
        let mut new = Muscle {
            bone_a: map(m.bone_a),
            bone_b: map(m.bone_b),
            phase: (m.phase + phase).rem_euclid(1.0),
            ..m
        };
        fit_stroke(c, &mut new, Some(&m));
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
        for m in &mut c.muscles {
            if m.bone_a as usize == trunk && m.bone_b as usize == child {
                m.bone_a = copy as u32;
            }
            if m.bone_b as usize == trunk && m.bone_a as usize == child {
                m.bone_b = copy as u32;
            }
        }
    }
    for &limb in &limbs {
        copy_branch(c, cfg, limb, node, |p| add(p, offset), false, phase);
    }
    keep_strokes(c, &before);
    true
}

/// Grows a toe and a heel on a limb tip: two short bones from the tip node, one
/// pointing forward (+x) and one back, each with a narrow joint range and a
/// muscle to the bone that ends at the tip. Both muscles copy the rhythm of one
/// muscle on that bone, or get a random rhythm if it has none.
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
        let bone = add_narrow_bone(c, tip, node, rng);
        let anchors = (rng.range(0.3, 1.0), rng.range(0.3, 0.9));
        let m = new_muscle(c, bone, leg, anchors, template.as_ref(), rng);
        c.muscles.push(m);
    }
    true
}

/// Grows a short bone with a narrow joint range from a joint node and moves one
/// end of a muscle on a bone at that joint onto the tip of the new bone. The
/// spur is a lever that changes the muscle's leverage, like an elbow process.
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
    // Muscle ends on a bone at the joint: (muscle, whether it is end a).
    let ends: Bounded<(usize, bool), { 2 * MAX_MUSCLES }> = c
        .muscles
        .iter()
        .enumerate()
        .flat_map(|(i, m)| [(i, true, m.bone_a), (i, false, m.bone_b)])
        .filter(|&(_, _, bone)| at_joint.contains(&(bone as usize)))
        .map(|(i, first, _)| (i, first))
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
    let before = spans(c);
    let node = add_node(c, joint, add(pos(c, joint), scale(direction, length)));
    let spur = add_narrow_bone(c, joint, node, rng) as u32;
    let m = &mut c.muscles[muscle];
    if first {
        m.bone_a = spur;
        m.anchor_a = 1.0;
    } else {
        m.bone_b = spur;
        m.anchor_b = 1.0;
    }
    keep_strokes(c, &before);
    true
}

/// Reflects a branch across the line of its root bone and swaps and negates
/// the joint limits of the bones below the root, so the limb bends the other
/// way. The root bone stays as it is. Muscle anchors are positions along
/// bones, so the muscles keep their roles.
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
    let before = spans(c);
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
    keep_strokes(c, &before);
    true
}

pub(super) fn pos(c: &Creature, node: usize) -> [f32; 2] {
    [c.nodes[node].x, c.nodes[node].y]
}

pub(super) fn add(p: [f32; 2], q: [f32; 2]) -> [f32; 2] {
    [p[0] + q[0], p[1] + q[1]]
}

pub(super) fn sub(p: [f32; 2], q: [f32; 2]) -> [f32; 2] {
    [p[0] - q[0], p[1] - q[1]]
}

pub(super) fn scale(p: [f32; 2], s: f32) -> [f32; 2] {
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
pub(super) fn add_node(c: &mut Creature, like: usize, at: [f32; 2]) -> usize {
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
pub(super) fn shift_branch(c: &mut Creature, bone: usize, offset: [f32; 2]) {
    for node in branch_nodes(c, &branch(c, bone)) {
        c.nodes[node].x += offset[0];
        c.nodes[node].y += offset[1];
    }
}

/// Every muscle's span in the pose.
pub(super) fn spans(c: &Creature) -> Bounded<f32, MAX_MUSCLES> {
    c.muscles.iter().map(|m| span(c, m)).collect()
}

/// Scales the stroke of each muscle that existed when `before` was taken by
/// how much its span changed since, so it pulls as it did in the old pose.
pub(super) fn keep_strokes(c: &mut Creature, before: &[f32]) {
    let after = spans(c);
    for ((m, old), new) in c.muscles.iter_mut().zip(before).zip(after) {
        let ratio = new.max(0.05) / old.max(0.05);
        m.short *= ratio;
        m.long *= ratio;
    }
}

/// Raises the whole body if a node would lie below the ground.
pub(super) fn lift(c: &mut Creature) {
    let low = c.nodes.iter().map(|n| n.y).fold(0.0, f32::min);
    for n in &mut c.nodes {
        n.y -= low;
    }
}

/// Turns bone `j`'s branch about its pivot by `angle` (counterclockwise),
/// lifting the body if a node would go below the ground.
pub(super) fn turn_branch(c: &mut Creature, j: usize, angle: f32) {
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
            let cx = Context::of(None);
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
            let leg = parent_bones(c)[tip].unwrap() as u32;
            for bone in [n - 2, n - 1] {
                let bone = bone as u32;
                assert!(
                    c.muscles
                        .iter()
                        .any(|m| m.bone_a == bone && m.bone_b == leg)
                );
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
            let spur = (c.bones.len() - 1) as u32;
            let bone = c.bones[spur as usize];
            assert!(-0.4 <= bone.min_angle && bone.max_angle <= 0.4);
            assert!(!child_bones(before)[bone.a as usize].is_empty());
            let on_spur: Vec<f32> = c
                .muscles
                .iter()
                .filter_map(|m| {
                    if m.bone_a == spur {
                        Some(m.anchor_a)
                    } else if m.bone_b == spur {
                        Some(m.anchor_b)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(on_spur, vec![1.0]);
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
