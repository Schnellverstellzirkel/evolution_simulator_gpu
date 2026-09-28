//! Structural mutations that change a working assembly of bones, joints and
//! muscles together: copy, grow, fuse, reconnect and retime whole limbs, so a
//! child keeps more of its parent's gait than a single random edit allows.
//!
//! The operators are an experiment (`EVOLUTION_ANATOMY`): off by default,
//! `all` or `1` for every operator, or a comma-separated list of operator
//! names. With it on, the structural emitter picks uniformly among the
//! classic operators and the enabled ones below, and tries up to four times
//! when the chosen operator does not apply to the body.
//!
//! Conventions every operator follows:
//! - The creature arrives repaired, so its bones are in canonical order:
//!   bone `j` joins its parent node `a` to its child node `b`, node 0 is the
//!   head, and the bone at the head is the neck. Bones an operator adds keep
//!   that convention (`a` is the node already in the tree).
//! - An operator returns whether it changed the creature. It keeps within
//!   `cfg.max_nodes` and `cfg.max_muscles`, and it never removes the head or
//!   the neck. `repair_with` runs after it (in `offspring`), which clamps
//!   genes, restores canonical order and the muscle ring, and lines the nodes
//!   up with the bone lengths.
//! - Muscles an operator adds start passive when `Context::neutral` is set
//!   (`neutralize`), as the classic operators do.
use super::{Bone, Creature, Muscle, NodeGene, Rng, bone_point, neutralize};
use crate::config::Config;

mod junctions;
mod limbs;
mod muscles;
mod rhythm;

/// What an operator may use besides the creature.
pub(super) struct Context<'a> {
    /// Added muscles start passive.
    pub neutral: bool,
    /// Another archive elite, for operators that graft from a second body.
    pub donor: Option<&'a Creature>,
}

pub(super) type Operator = fn(&mut Creature, &Config, &mut Rng, &Context) -> bool;

/// Every operator, by name. The names are what `EVOLUTION_ANATOMY` lists.
pub(super) const OPERATORS: [(&str, Operator); 30] = [
    ("copy_limb", limbs::copy_limb),
    ("grow_actuated_tip", limbs::grow_actuated_tip),
    ("split_bone_actuated", limbs::split_bone_actuated),
    ("fuse_bones", limbs::fuse_bones),
    ("relocate_limb", limbs::relocate_limb),
    ("reshape_limb", limbs::reshape_limb),
    ("graft_donor_limb", limbs::graft_donor_limb),
    ("split_crowded_joint", junctions::split_crowded_joint),
    ("merge_branch_joints", junctions::merge_branch_joints),
    ("repeat_body_segment", junctions::repeat_body_segment),
    ("grow_heel_toe", junctions::grow_heel_toe),
    ("grow_lever_spur", junctions::grow_lever_spur),
    ("reverse_bend", junctions::reverse_bend),
    ("add_biarticular_muscle", muscles::add_biarticular_muscle),
    ("move_muscle_to_neighbor", muscles::move_muscle_to_neighbor),
    ("split_muscle", muscles::split_muscle),
    ("fuse_similar_muscles", muscles::fuse_similar_muscles),
    ("add_antagonist", muscles::add_antagonist),
    ("swap_muscle_routes", muscles::swap_muscle_routes),
    ("fan_muscle_attachments", muscles::fan_muscle_attachments),
    ("relay_muscle", muscles::relay_muscle),
    ("copy_actuation_to_limb", muscles::copy_actuation_to_limb),
    ("quiet_muscle_group", muscles::quiet_muscle_group),
    ("redistribute_joint_flex", rhythm::redistribute_joint_flex),
    ("mutate_matching_limbs", rhythm::mutate_matching_limbs),
    ("chain_phase_wave", rhythm::chain_phase_wave),
    ("limb_phase_pattern", rhythm::limb_phase_pattern),
    ("limb_duty_cycle", rhythm::limb_duty_cycle),
    ("touchdown_package", rhythm::touchdown_package),
    ("redistribute_organ_mass", rhythm::redistribute_organ_mass),
];

/// The operators `EVOLUTION_ANATOMY` enables, as indices into `OPERATORS`.
pub(super) fn enabled() -> &'static [usize] {
    static ENABLED: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    ENABLED.get_or_init(|| parse(std::env::var("EVOLUTION_ANATOMY").ok().as_deref()))
}

fn parse(value: Option<&str>) -> Vec<usize> {
    let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return Vec::new();
    };
    if matches!(value, "1" | "all" | "on" | "true") {
        return (0..OPERATORS.len()).collect();
    }
    value
        .split(',')
        .filter_map(|name| OPERATORS.iter().position(|(n, _)| *n == name.trim()))
        .collect()
}

/// Runs operator `index` of `OPERATORS`.
pub(super) fn apply(
    index: usize,
    creature: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    cx: &Context,
) -> bool {
    (OPERATORS[index].1)(creature, cfg, rng, cx)
}

// Shared helpers. They read the skeleton from `a` (parent) and `b` (child),
// so they also work on bones an operator appended.

/// Bones at a node.
pub(super) fn degree(c: &Creature, node: usize) -> usize {
    c.bones
        .iter()
        .filter(|b| b.a as usize == node || b.b as usize == node)
        .count()
}

/// For each node, the bone whose child it is (`None` for the head).
pub(super) fn parent_bones(c: &Creature) -> Vec<Option<usize>> {
    let mut parent = vec![None; c.nodes.len()];
    for (index, bone) in c.bones.iter().enumerate() {
        if let Some(slot) = parent.get_mut(bone.b as usize) {
            *slot = Some(index);
        }
    }
    parent
}

/// For each node, the bones it is the parent of.
pub(super) fn child_bones(c: &Creature) -> Vec<Vec<usize>> {
    let mut children = vec![Vec::new(); c.nodes.len()];
    for (index, bone) in c.bones.iter().enumerate() {
        if let Some(list) = children.get_mut(bone.a as usize) {
            list.push(index);
        }
    }
    children
}

/// Whether `bone` is the neck (it touches the head).
pub(super) fn is_neck(c: &Creature, bone: usize) -> bool {
    c.bones[bone].a == 0 || c.bones[bone].b == 0
}

/// The branch that starts with `bone`: it and every bone below its child
/// node, parents before children.
pub(super) fn branch(c: &Creature, bone: usize) -> Vec<usize> {
    let children = child_bones(c);
    let mut out = vec![bone];
    let mut next = 0;
    while next < out.len() {
        let node = c.bones[out[next]].b as usize;
        out.extend(children.get(node).into_iter().flatten().copied());
        next += 1;
    }
    out
}

/// The child nodes of a branch's bones (every node below its root joint).
pub(super) fn branch_nodes(c: &Creature, bones: &[usize]) -> Vec<usize> {
    bones.iter().map(|&b| c.bones[b].b as usize).collect()
}

/// Muscles with both ends (`both`) or at least one end on `bones`.
pub(super) fn muscles_on(c: &Creature, bones: &[usize], both: bool) -> Vec<usize> {
    let on = |b: u32| bones.contains(&(b as usize));
    (0..c.muscles.len())
        .filter(|&i| {
            let m = &c.muscles[i];
            if both {
                on(m.bone_a) && on(m.bone_b)
            } else {
                on(m.bone_a) || on(m.bone_b)
            }
        })
        .collect()
}

/// Whether the body has room for `nodes` more nodes and `muscles` more
/// muscles.
pub(super) fn room(c: &Creature, cfg: &Config, nodes: usize, muscles: usize) -> bool {
    c.nodes.len() + nodes <= cfg.max_nodes.min(64) && c.muscles.len() + muscles <= cfg.max_muscles
}

/// Removes `bones` and `nodes` (and every muscle on a removed bone), and
/// renumbers what is left. The caller makes sure the rest stays one tree.
pub(super) fn remove_parts(c: &mut Creature, bones: &[usize], nodes: &[usize]) {
    let node_map = renumber(c.nodes.len(), nodes);
    let bone_map = renumber(c.bones.len(), bones);
    let mut index = 0;
    c.nodes.retain(|_| {
        index += 1;
        node_map[index - 1] != usize::MAX
    });
    let mut index = 0;
    c.bones.retain(|_| {
        index += 1;
        bone_map[index - 1] != usize::MAX
    });
    for bone in &mut c.bones {
        bone.a = node_map[bone.a as usize] as u32;
        bone.b = node_map[bone.b as usize] as u32;
    }
    c.muscles.retain_mut(|m| {
        let (a, b) = (bone_map[m.bone_a as usize], bone_map[m.bone_b as usize]);
        if a == usize::MAX || b == usize::MAX || a == b {
            return false;
        }
        m.bone_a = a as u32;
        m.bone_b = b as u32;
        true
    });
}

/// New index of each of `len` items after `removed` go (`usize::MAX` for a
/// removed one).
fn renumber(len: usize, removed: &[usize]) -> Vec<usize> {
    let mut map = vec![0; len];
    let mut next = 0;
    for (index, slot) in map.iter_mut().enumerate() {
        if removed.contains(&index) {
            *slot = usize::MAX;
        } else {
            *slot = next;
            next += 1;
        }
    }
    map
}

/// A muscle's span: the distance between its attachment points in the pose.
pub(super) fn span(c: &Creature, m: &Muscle) -> f32 {
    let a = bone_point(c.bones[m.bone_a as usize], &c.nodes, m.anchor_a);
    let b = bone_point(c.bones[m.bone_b as usize], &c.nodes, m.anchor_b);
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Sets a muscle's stroke around its span in the pose, keeping the ratios of
/// `short` and `long` to the span that `template` has (or 0.8 and 1.1).
pub(super) fn fit_stroke(c: &Creature, m: &mut Muscle, template: Option<&Muscle>) {
    let (short, long) = template
        .map(|t| {
            let s = span(c, t).max(0.05);
            (t.short / s, t.long / s)
        })
        .unwrap_or((0.8, 1.1));
    let length = span(c, m).max(0.05);
    m.short = (length * short).max(0.01);
    m.long = (length * long).max(m.short);
}

/// A new muscle from `bone_a` to `bone_b` with the given anchors. Its rhythm
/// (period, phase, duty, stiffness, sensor, reset) comes from `template`, or
/// is random without one; its stroke fits its span. Passive with `neutral`.
pub(super) fn new_muscle(
    c: &Creature,
    bone_a: usize,
    bone_b: usize,
    anchors: (f32, f32),
    template: Option<&Muscle>,
    rng: &mut Rng,
    neutral: bool,
) -> Muscle {
    let mut m = match template {
        Some(t) => *t,
        None => super::muscle(bone_a, bone_b, &c.bones, &c.nodes, rng),
    };
    m.bone_a = bone_a as u32;
    m.bone_b = bone_b as u32;
    m.anchor_a = anchors.0.clamp(0.0, 1.0);
    m.anchor_b = anchors.1.clamp(0.0, 1.0);
    fit_stroke(c, &mut m, template);
    if neutral {
        neutralize(&mut m);
    }
    m
}

/// Copies the branch that starts at `bone` onto node `at`, placing each
/// copied node at `place(original position)`. The copy brings its joint
/// ranges (mirrored with `mirror`) and every muscle inside the branch, plus
/// the muscles from the branch root to the bone above it, reattached to the
/// bone above `at` when there is one. Copied muscles shift their phase by
/// `phase` and start passive with `neutral`. Returns the new root bone, or
/// `None` without room.
#[allow(clippy::too_many_arguments)]
pub(super) fn copy_branch(
    c: &mut Creature,
    cfg: &Config,
    bone: usize,
    at: usize,
    place: impl Fn([f32; 2]) -> [f32; 2],
    mirror: bool,
    phase: f32,
    neutral: bool,
) -> Option<usize> {
    let bones = branch(c, bone);
    let parents = parent_bones(c);
    let above_source = parents[c.bones[bone].a as usize];
    let above_target = parents.get(at).copied().flatten();
    let inside = muscles_on(c, &bones, true);
    let hinge: Vec<usize> = match (above_source, above_target) {
        (Some(src), Some(_)) => muscles_on(c, &[bone], false)
            .into_iter()
            .filter(|&i| {
                let m = &c.muscles[i];
                (m.bone_a as usize == src) != (m.bone_b as usize == src) && !inside.contains(&i)
            })
            .collect(),
        _ => Vec::new(),
    };
    if !room(c, cfg, bones.len(), inside.len() + hinge.len()) {
        return None;
    }
    let mut new_bone = std::collections::HashMap::new();
    let mut new_node = std::collections::HashMap::new();
    new_node.insert(c.bones[bone].a as usize, at);
    for &b in &bones {
        let old = c.bones[b];
        let child = old.b as usize;
        let n = c.nodes[child];
        let [x, y] = place([n.x, n.y]);
        let node = c.nodes.len();
        c.nodes.push(NodeGene {
            x: x.clamp(-super::body_extent(), super::body_extent()),
            y: y.clamp(0.0, super::body_extent()),
            ..n
        });
        new_node.insert(child, node);
        let (min, max) = if mirror {
            (-old.max_angle, -old.min_angle)
        } else {
            (old.min_angle, old.max_angle)
        };
        new_bone.insert(b, c.bones.len());
        c.bones.push(Bone {
            a: new_node[&(old.a as usize)] as u32,
            b: node as u32,
            min_angle: min,
            max_angle: max,
            ..old
        });
    }
    let remap = |b: u32| -> u32 {
        match new_bone.get(&(b as usize)) {
            Some(&n) => n as u32,
            None => above_target.expect("hinge muscles need a bone above") as u32,
        }
    };
    for i in inside.into_iter().chain(hinge) {
        let mut m = c.muscles[i];
        m.bone_a = remap(m.bone_a);
        m.bone_b = remap(m.bone_b);
        if m.bone_a == m.bone_b {
            continue;
        }
        m.phase = (m.phase + phase).rem_euclid(1.0);
        if neutral {
            neutralize(&mut m);
        }
        c.muscles.push(m);
    }
    Some(new_bone[&bone])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::{Population, grow_for_benchmark, random_creature_from, repair_with};

    /// Repaired bodies of 3 to 16 nodes grown with the classic operators.
    pub(super) fn bodies(cfg: &Config, count: usize) -> Vec<Creature> {
        (0..count)
            .map(|i| {
                let mut rng = Rng::new(7, 1, i);
                let mut c = random_creature_from(cfg, &mut rng);
                c.id = i as u64 + 1;
                grow_for_benchmark(&mut c, cfg, 11, 3 + i % 14);
                c
            })
            .collect()
    }

    #[test]
    fn operator_names_parse_and_are_unique() {
        assert!(parse(None).is_empty());
        assert!(parse(Some("")).is_empty());
        assert_eq!(parse(Some("all")).len(), OPERATORS.len());
        assert_eq!(parse(Some("1")).len(), OPERATORS.len());
        assert_eq!(parse(Some("fuse_bones, copy_limb")), vec![3, 0]);
        for (i, (name, _)) in OPERATORS.iter().enumerate() {
            assert_eq!(OPERATORS.iter().position(|(n, _)| n == name), Some(i));
        }
    }

    #[test]
    fn every_operator_leaves_a_valid_body() {
        let cfg = Config::default();
        let bodies = bodies(&cfg, 160);
        let donor = bodies[bodies.len() / 2].clone();
        for (index, (name, _)) in OPERATORS.iter().enumerate() {
            let mut applied = 0;
            for (i, body) in bodies.iter().enumerate() {
                for neutral in [false, true] {
                    let mut c = body.clone();
                    let mut rng = Rng::new(13, index as u32, i);
                    let cx = Context {
                        neutral,
                        donor: Some(&donor),
                    };
                    if !apply(index, &mut c, &cfg, &mut rng, &cx) {
                        continue;
                    }
                    applied += 1;
                    assert!(c.nodes.len() <= cfg.max_nodes, "{name} grew past max_nodes");
                    assert!(
                        c.muscles.len() <= cfg.max_muscles,
                        "{name} grew past max_muscles"
                    );
                    repair_with(&mut c, &cfg, &mut rng, neutral);
                    let mut pop = Population::default();
                    pop.push(c.clone());
                    let check = Config {
                        population: 1,
                        ..cfg.clone()
                    };
                    if let Err(error) = pop.validate(&check) {
                        panic!("{name} on body {i} (neutral {neutral}): {error:#}\n{c:?}");
                    }
                }
            }
            eprintln!("{name}: applied to {applied} of {}", 2 * bodies.len());
        }
    }

    #[test]
    fn copy_branch_brings_its_bones_and_muscles() {
        let cfg = Config::default();
        for body in bodies(&cfg, 60) {
            let leaf = (0..body.bones.len())
                .find(|&b| !is_neck(&body, b) && degree(&body, body.bones[b].b as usize) == 1);
            let Some(leaf) = leaf else { continue };
            let mut c = body.clone();
            let at = c.bones[leaf].a as usize;
            let inside = muscles_on(&c, &branch(&c, leaf), true).len();
            let before = (c.nodes.len(), c.bones.len(), c.muscles.len());
            let Some(root) = copy_branch(&mut c, &cfg, leaf, at, |p| p, false, 0.5, false) else {
                continue;
            };
            assert_eq!(c.nodes.len(), before.0 + 1);
            assert_eq!(c.bones.len(), before.1 + 1);
            assert!(c.muscles.len() >= before.2 + inside);
            assert_eq!(c.bones[root].a as usize, at);
            assert_eq!(c.bones[root].rest_length, body.bones[leaf].rest_length);
        }
    }

    #[test]
    fn remove_parts_renumbers_bones_nodes_and_muscles() {
        let cfg = Config::default();
        for body in bodies(&cfg, 60) {
            let leaf = (0..body.bones.len())
                .find(|&b| !is_neck(&body, b) && degree(&body, body.bones[b].b as usize) == 1);
            let Some(leaf) = leaf else { continue };
            let mut c = body.clone();
            let tip = c.bones[leaf].b as usize;
            remove_parts(&mut c, &[leaf], &[tip]);
            assert_eq!(c.nodes.len(), body.nodes.len() - 1);
            assert_eq!(c.bones.len(), body.bones.len() - 1);
            for bone in &c.bones {
                assert!((bone.a as usize) < c.nodes.len() && (bone.b as usize) < c.nodes.len());
            }
            for m in &c.muscles {
                assert!((m.bone_a as usize) < c.bones.len() && (m.bone_b as usize) < c.bones.len());
            }
        }
    }
}
