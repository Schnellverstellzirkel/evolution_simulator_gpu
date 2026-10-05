//! Structural mutations that change a working assembly of bones, joints and
//! muscles together: copy, grow, fuse, reconnect and retime whole limbs, so a
//! child keeps more of its parent's gait than a single random edit allows.
//!
//! Every operator is on (docs/anatomy-operators.md has the audit and the
//! search A/B). The structural emitter picks uniformly among the classic
//! operators, the ones below, one slot that the `SHARED_SLOT`
//! operators share and one for the `CONTROLLER_SLOT` operators, and tries up to four times when the chosen operator
//! does not apply to the body.
//!
//! Conventions every operator follows:
//! - The creature arrives repaired, so its bones are in canonical order:
//!   bone `j` joins its parent node `a` to its child node `b`, node 0 is the
//!   head, and the bone at the head is the neck. Bones an operator adds keep
//!   that convention (`a` is the node already in the tree).
//! - An operator returns whether it changed the creature. It keeps within
//!   `cfg.max_nodes` and `cfg.max_muscles`, and it never removes the head or
//!   the neck. `repair` runs after it (in `offspring`), which clamps
//!   genes, restores canonical order and the muscle ring, and lines the nodes
//!   up with the bone lengths.
use super::{Bone, Bounded, Creature, MAX_MUSCLES, MAX_NODES, Muscle, NodeGene, Rng, bone_point};
use crate::config::Config;

mod compound;
mod controller;
mod extra;
mod gait_bio;
mod gait_legs;
mod gait_muscles;
mod gait_phase;
mod gait_plans;
mod gait_posture;
mod gait_reflex;
mod gait_spine;
mod gait_symmetry;
mod idea_blend;
mod idea_elastic;
mod idea_shape;
mod idea_surface;
mod idea_topology;
mod idea_timing;
mod idea_wild;
mod ideas;

mod junctions;
mod legs;
mod limbs;
mod muscles;
mod rhythm;

/// Indices of bones or nodes of one body.
pub(super) type BoneIds = Bounded<usize, MAX_NODES>;
/// Indices of muscles of one body.
pub(super) type MuscleIds = Bounded<usize, MAX_MUSCLES>;
/// For each node, the bones it is the parent of (`child_bones`).
pub(super) type Children = [BoneIds; MAX_NODES];
/// Lists of bones, one per limb.
pub(super) type Limbs = Bounded<BoneIds, MAX_NODES>;

/// What an operator may use besides the creature.
pub(super) struct Context<'a> {
    /// Another archive elite, for operators that graft from a second body.
    pub donor: Option<&'a Creature>,
}

pub(super) type Operator = fn(&mut Creature, &Config, &mut Rng, &Context) -> bool;

/// Every operator, by name: the ones below, then those of the gait files.
pub(super) static OPERATORS: std::sync::LazyLock<Vec<(&'static str, Operator)>> =
    std::sync::LazyLock::new(|| {
        BASE_OPERATORS
            .iter()
            .chain(GAIT_FILES.iter().flat_map(|file| file.iter()))
            .copied()
            .collect()
    });

/// The gait operators, one list per file. Each file's operators share one
/// pick slot, so a hundred of them do not crowd out the others, and every one
/// is a compound operator (a whole change, no parameter noise after it).
const GAIT_FILES: &[&[(&str, Operator)]] = &[
    gait_legs::OPS,
    gait_spine::OPS,
    gait_phase::OPS,
    gait_muscles::OPS,
    gait_symmetry::OPS,
    gait_reflex::OPS,
    gait_posture::OPS,
    gait_plans::OPS,
    gait_bio::OPS,
    idea_surface::OPS,
    idea_elastic::OPS,
    idea_timing::OPS,
    idea_topology::OPS,
    idea_blend::OPS,
    idea_wild::OPS,
    idea_shape::OPS,
];

/// The operators before the gait files.
const BASE_OPERATORS: &[(&str, Operator)] = &[
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
    ("mirror_limb_timing", extra::mirror_limb_timing),
    ("swap_limb_programs", extra::swap_limb_programs),
    ("copy_muscle_to_partner", extra::copy_muscle_to_partner),
    ("twin_limb", extra::twin_limb),
    ("grow_matching_tips", extra::grow_matching_tips),
    ("nudge_limb_phase", extra::nudge_limb_phase),
    ("cadence_stride_trade", extra::cadence_stride_trade),
    ("scale_muscle_leverage", extra::scale_muscle_leverage),
    ("scale_limb_strength", extra::scale_limb_strength),
    ("prune_weakest_muscle", extra::prune_weakest_muscle),
    ("prune_idle_limb", extra::prune_idle_limb),
    ("merge_leaf_bones", extra::merge_leaf_bones),
    ("leg_to_dragging_end", extra::leg_to_dragging_end),
    ("lift_dragging_end", extra::lift_dragging_end),
    ("limb_stroke_scale", controller::limb_stroke_scale),
    ("limb_posture_shift", controller::limb_posture_shift),
    ("taper_limb_strength", controller::taper_limb_strength),
    ("copy_limb_rhythm", controller::copy_limb_rhythm),
    ("retune_muscle_pair", controller::retune_muscle_pair),
    ("release_touchdown", controller::release_touchdown),
    ("snap_limb_phases", controller::snap_limb_phases),
    ("limb_clock_ratio", controller::limb_clock_ratio),
    ("limb_clock_lock", controller::limb_clock_lock),
    ("reflex_on_muscle", controller::reflex_on_muscle),
    ("reflex_all_feet", controller::reflex_all_feet),
    ("reflex_reset_shift", controller::reflex_reset_shift),
    ("shift_gait_start", controller::shift_gait_start),
    ("pose_joint_at_stop", junctions::pose_joint_at_stop),
    ("brace_joint", junctions::brace_joint),
    ("limb_length_gradient", compound::limb_length_gradient),
    ("symmetrize_limb_pair", compound::symmetrize_limb_pair),
    ("retime_gait_by_position", compound::retime_gait_by_position),
    ("brace_limb_chain", compound::brace_limb_chain),
    ("phase_cluster_move", compound::phase_cluster_move),
    ("grow_integrated_limb", compound::grow_integrated_limb),
    ("mirrored_limb_pair", compound::mirrored_limb_pair),
    ("segment_chain", compound::segment_chain),
    ("reassign_bundle", compound::reassign_bundle),
    ("transplant_limb_program", compound::transplant_limb_program),
    ("retune_limb_package", compound::retune_limb_package),
    ("transplant_gait", compound::transplant_gait),
    ("trim_body", compound::trim_body),
    ("sprout_leg", legs::sprout_leg),
    ("mirror_leg_fore_aft", legs::mirror_leg_fore_aft),
    ("spread_leg_attachment", legs::spread_leg_attachment),
    ("tuck_leg_under", legs::tuck_leg_under),
];

/// Operators that share one pick slot: together they are as likely as one
/// other operator. They keep much of a parent's gait in the audit, but with a
/// slot each the search got worse, and with one shared slot it did not
/// (docs/anatomy-operators.md).
const SHARED_SLOT: &[&str] = &[
    "mirror_limb_timing",
    "swap_limb_programs",
    "copy_muscle_to_partner",
    "nudge_limb_phase",
    "cadence_stride_trade",
    "scale_muscle_leverage",
    "scale_limb_strength",
    "prune_weakest_muscle",
];

/// Grafts a limb of `donor` onto `c` (`graft_donor_limb`), for crossover
/// between different body plans.
pub(super) fn graft_from(c: &mut Creature, cfg: &Config, rng: &mut Rng, donor: &Creature) -> bool {
    let cx = Context { donor: Some(donor) };
    limbs::graft_donor_limb(c, cfg, rng, &cx)
}
/// The controller operators (`controller.rs`) share a second pick slot. With a
/// slot each they looked slightly worse in the search (9 seeds), so they
/// share one as the gentle group does.
const CONTROLLER_SLOT: &[&str] = &[
    "limb_stroke_scale",
    "limb_posture_shift",
    "taper_limb_strength",
    "copy_limb_rhythm",
    "retune_muscle_pair",
    "release_touchdown",
    "snap_limb_phases",
    "limb_clock_ratio",
    "limb_clock_lock",
    "reflex_on_muscle",
    "reflex_all_feet",
    "reflex_reset_shift",
];

/// The compound operators (`compound.rs`). Each is a whole change by
/// itself, so a child that one of them made gets no parameter noise after it:
/// the noise would only blur a move that was built to be coherent.
const COMPOUND: &[&str] = &[
    "limb_length_gradient",
    "symmetrize_limb_pair",
    "retime_gait_by_position",
    "brace_limb_chain",
    "phase_cluster_move",
    "grow_integrated_limb",
    "mirrored_limb_pair",
    "segment_chain",
    "reassign_bundle",
    "transplant_limb_program",
    "retune_limb_package",
    "transplant_gait",
    "trim_body",
    "sprout_leg",
    "mirror_leg_fore_aft",
    "spread_leg_attachment",
    "tuck_leg_under",
];

/// The gait file that holds operator `name`, if any.
fn gait_file(name: &str) -> Option<usize> {
    GAIT_FILES
        .iter()
        .position(|file| file.iter().any(|(n, _)| *n == name))
}

/// Whether operator `index` of `OPERATORS` is a compound one.
pub(super) fn is_compound(index: usize) -> bool {
    enabled().compound.get(index).copied().unwrap_or(false)
}

/// The enabled operators, as indices into `OPERATORS`.
pub(super) struct Enabled {
    /// For each operator, whether it is a compound one (`COMPOUND`).
    pub compound: Vec<bool>,
    /// Operators with a pick slot each.
    pub single: Vec<usize>,
    /// Operators that share one pick slot (`SHARED_SLOT`).
    pub shared: Vec<usize>,
    /// Operators that share the second pick slot (`CONTROLLER_SLOT`).
    pub controller: Vec<usize>,
    /// The operators of each gait file, one pick slot per file.
    pub gait: Vec<Vec<usize>>,
}

/// The operators, split by pick slot.
pub(super) fn enabled() -> &'static Enabled {
    static ENABLED: std::sync::OnceLock<Enabled> = std::sync::OnceLock::new();
    ENABLED.get_or_init(|| split((0..OPERATORS.len()).collect()))
}

fn split(indices: Vec<usize>) -> Enabled {
    let mut enabled = Enabled {
        compound: OPERATORS
            .iter()
            .map(|(name, _)| COMPOUND.contains(name) || gait_file(name).is_some())
            .collect(),
        single: Vec::new(),
        shared: Vec::new(),
        controller: Vec::new(),
        gait: vec![Vec::new(); GAIT_FILES.len()],
    };
    for i in indices {
        let name = OPERATORS[i].0;
        if let Some(file) = gait_file(name) {
            enabled.gait[file].push(i);
        } else if SHARED_SLOT.contains(&name) {
            enabled.shared.push(i);
        } else if CONTROLLER_SLOT.contains(&name) {
            enabled.controller.push(i);
        } else {
            enabled.single.push(i);
        }
    }
    enabled
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

/// Picks one of the options `each` passes to its sink, as picking from the
/// collected list would (one `rng.index` draw over their count), without
/// storing them: `each` runs twice, once to count and once to find the one
/// picked. For option lists too long to hold on the stack.
pub(super) fn pick_each<T>(rng: &mut Rng, each: impl Fn(&mut dyn FnMut(T))) -> Option<T> {
    let mut count = 0usize;
    each(&mut |_| count += 1);
    if count == 0 {
        return None;
    }
    let target = rng.index(count);
    let mut seen = 0usize;
    let mut chosen = None;
    each(&mut |item| {
        if seen == target {
            chosen = Some(item);
        }
        seen += 1;
    });
    chosen
}

/// Bones at a node.
pub(super) fn degree(c: &Creature, node: usize) -> usize {
    c.bones
        .iter()
        .filter(|b| b.a as usize == node || b.b as usize == node)
        .count()
}

/// For each node, the bone whose child it is (`None` for the head).
pub(super) fn parent_bones(c: &Creature) -> Bounded<Option<usize>, MAX_NODES> {
    let mut parent = Bounded::filled(c.nodes.len(), None);
    for (index, bone) in c.bones.iter().enumerate() {
        if let Some(slot) = parent.get_mut(bone.b as usize) {
            *slot = Some(index);
        }
    }
    parent
}

/// For each node, the bones it is the parent of.
pub(super) fn child_bones(c: &Creature) -> Children {
    let mut children: Children = std::array::from_fn(|_| BoneIds::new());
    for (index, bone) in c.bones.iter().enumerate() {
        if let Some(list) = children[..c.nodes.len()].get_mut(bone.a as usize) {
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
pub(super) fn branch(c: &Creature, bone: usize) -> BoneIds {
    branch_in(c, &child_bones(c), bone)
}

/// `branch` with the child lists (`child_bones`) the caller already has, for
/// callers that take many branches of one body.
pub(super) fn branch_in(c: &Creature, children: &Children, bone: usize) -> BoneIds {
    let mut out = BoneIds::from_slice(&[bone]);
    let mut next = 0;
    while next < out.len() {
        let node = c.bones[out[next]].b as usize;
        out.extend(
            children[..c.nodes.len()]
                .get(node)
                .into_iter()
                .flatten()
                .copied(),
        );
        next += 1;
    }
    out
}

/// The child nodes of a branch's bones (every node below its root joint).
pub(super) fn branch_nodes(c: &Creature, bones: &[usize]) -> BoneIds {
    bones.iter().map(|&b| c.bones[b].b as usize).collect()
}

/// Muscles with both ends (`both`) or at least one end on `bones`.
pub(super) fn muscles_on(c: &Creature, bones: &[usize], both: bool) -> MuscleIds {
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
    c.nodes.len() + nodes <= cfg.max_nodes.min(MAX_NODES)
        && c.muscles.len() + muscles <= cfg.max_muscles.min(MAX_MUSCLES)
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
fn renumber(len: usize, removed: &[usize]) -> BoneIds {
    let mut map = BoneIds::filled(len, 0);
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
/// is random without one; its stroke fits its span.
pub(super) fn new_muscle(
    c: &Creature,
    bone_a: usize,
    bone_b: usize,
    anchors: (f32, f32),
    template: Option<&Muscle>,
    rng: &mut Rng,
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
    m
}

/// Copies the branch that starts at `bone` onto node `at`, placing each
/// copied node at `place(original position)`. The copy brings its joint
/// ranges (mirrored with `mirror`) and every muscle inside the branch, plus
/// the muscles from the branch root to the bone above it, reattached to the
/// bone above `at` when there is one. Copied muscles shift their phase by
/// `phase`. Returns the new root bone, or
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
) -> Option<usize> {
    copy_branch_limited(c, cfg, bone, at, place, mirror, phase, usize::MAX)
}

/// `copy_branch` that brings at most `quota` muscles: when the branch and
/// its hinge have more, the copy keeps the ones with the most drive.
#[allow(clippy::too_many_arguments)]
pub(super) fn copy_branch_limited(
    c: &mut Creature,
    cfg: &Config,
    bone: usize,
    at: usize,
    place: impl Fn([f32; 2]) -> [f32; 2],
    mirror: bool,
    phase: f32,
    quota: usize,
) -> Option<usize> {
    let bones = branch(c, bone);
    let parents = parent_bones(c);
    let above_source = parents[c.bones[bone].a as usize];
    let above_target = parents.get(at).copied().flatten();
    let inside = muscles_on(c, &bones, true);
    let hinge: MuscleIds = match (above_source, above_target) {
        (Some(src), Some(_)) => muscles_on(c, &[bone], false)
            .into_iter()
            .filter(|&i| {
                let m = &c.muscles[i];
                (m.bone_a as usize == src) != (m.bone_b as usize == src) && !inside.contains(&i)
            })
            .collect(),
        _ => MuscleIds::new(),
    };
    let mut brought: MuscleIds = inside.into_iter().chain(hinge).collect();
    if brought.len() > quota {
        brought.sort_stable_by(|&x, &y| {
            extra::drive(&c.muscles[y]).total_cmp(&extra::drive(&c.muscles[x]))
        });
        brought.truncate(quota);
    }
    if !room(c, cfg, bones.len(), brought.len()) {
        return None;
    }
    // Where each copied bone and node went (`usize::MAX` for the rest).
    let mut new_bone = [usize::MAX; MAX_NODES];
    let mut new_node = [usize::MAX; MAX_NODES];
    new_node[c.bones[bone].a as usize] = at;
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
        new_node[child] = node;
        let (min, max) = if mirror {
            (-old.max_angle, -old.min_angle)
        } else {
            (old.min_angle, old.max_angle)
        };
        new_bone[b] = c.bones.len();
        c.bones.push(Bone {
            a: new_node[old.a as usize] as u32,
            b: node as u32,
            min_angle: min,
            max_angle: max,
            ..old
        });
    }
    let remap = |b: u32| -> u32 {
        match new_bone[b as usize] {
            usize::MAX => above_target.expect("hinge muscles need a bone above") as u32,
            n => n as u32,
        }
    };
    for i in brought {
        let mut m = c.muscles[i];
        m.bone_a = remap(m.bone_a);
        m.bone_b = remap(m.bone_b);
        if m.bone_a == m.bone_b {
            continue;
        }
        m.phase = (m.phase + phase).rem_euclid(1.0);
        c.muscles.push(m);
    }
    Some(new_bone[bone])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::{Population, grow_for_benchmark, random_creature_from, repair};

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
    fn operator_names_are_unique() {
        for (i, (name, _)) in OPERATORS.iter().enumerate() {
            assert_eq!(OPERATORS.iter().position(|(n, _)| n == name), Some(i));
        }
        let on = enabled();
        assert_eq!(on.shared.len(), SHARED_SLOT.len());
        assert_eq!(on.controller.len(), CONTROLLER_SLOT.len());
        assert_eq!(
            on.single.len()
                + on.shared.len()
                + on.controller.len()
                + on.gait.iter().map(Vec::len).sum::<usize>(),
            OPERATORS.len()
        );
        assert!(on.single.windows(2).all(|w| w[0] < w[1]), "table order");
    }

    #[test]
    fn every_operator_leaves_a_valid_body() {
        // The default limits, and tight ones where bodies sit at the limits.
        let tight = Config {
            max_nodes: 8,
            max_muscles: 8,
            ..Config::default()
        };
        for cfg in [Config::default(), tight] {
            let bodies = bodies(&cfg, 160);
            let donor = bodies[bodies.len() / 2].clone();
            for (index, (name, _)) in OPERATORS.iter().enumerate() {
                let mut applied = 0;
                for (i, body) in bodies.iter().enumerate() {
                    for variant in 0..2u32 {
                        let mut c = body.clone();
                        let mut rng = Rng::new(13, index as u32 + 1000 * variant, i);
                        let cx = Context {
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
                        repair(&mut c, &cfg, &mut rng);
                        let mut pop = Population::default();
                        pop.push(c.clone());
                        let check = Config {
                            population: 1,
                            ..cfg.clone()
                        };
                        if let Err(error) = pop.validate(&check) {
                            panic!(
                                "{name} on body {i} (max_nodes {}, max_muscles {}): {error:#}\n{c:?}",
                                cfg.max_nodes, cfg.max_muscles
                            );
                        }
                    }
                }
                eprintln!(
                    "{name}: applied to {applied} of {} (max_nodes {})",
                    2 * bodies.len(),
                    cfg.max_nodes
                );
            }
        }
    }

    #[test]
    fn an_operator_is_a_function_of_its_stream_and_its_body() {
        let cfg = Config::default();
        let bodies = bodies(&cfg, 40);
        let donor = bodies[20].clone();
        for (index, (name, _)) in OPERATORS.iter().enumerate() {
            for (i, body) in bodies.iter().enumerate() {
                let cx = Context {
                    donor: Some(&donor),
                };
                let (mut x, mut y) = (body.clone(), body.clone());
                let a = apply(index, &mut x, &cfg, &mut Rng::new(61, index as u32, i), &cx);
                let b = apply(index, &mut y, &cfg, &mut Rng::new(61, index as u32, i), &cx);
                assert_eq!(a, b, "{name} on body {i}");
                assert!(
                    x.nodes == y.nodes && x.bones == y.bones && x.muscles == y.muscles,
                    "{name} on body {i} is not a function of its stream"
                );
            }
        }
    }

    /// Bodies at the default caps: grown toward 32 nodes and filled toward
    /// 96 muscles, where the bounded arrays are full.
    fn full_bodies(cfg: &Config, count: usize) -> Vec<Creature> {
        (0..count)
            .map(|i| {
                let mut rng = Rng::new(3, 2, i);
                let mut c = random_creature_from(cfg, &mut rng);
                c.id = i as u64 + 1;
                grow_for_benchmark(&mut c, cfg, 5 + i as u64, 22 + i % 11);
                let target = cfg.max_muscles - i % 4;
                while c.muscles.len() < target {
                    let (a, b) = (rng.index(c.bones.len()), rng.index(c.bones.len()));
                    if a != b {
                        let m = crate::evolution::muscle(a, b, &c.bones, &c.nodes, &mut rng);
                        c.muscles.push(m);
                    }
                }
                repair(&mut c, cfg, &mut rng);
                c
            })
            .collect()
    }

    #[test]
    fn operators_fit_the_bounded_arrays_at_the_caps() {
        let cfg = Config::default();
        assert_eq!(
            (cfg.max_nodes, cfg.max_muscles),
            (MAX_NODES, MAX_MUSCLES),
            "the default caps are the array capacities"
        );
        let bodies = full_bodies(&cfg, 48);
        assert!(bodies.iter().any(|c| c.nodes.len() == MAX_NODES));
        assert!(bodies.iter().any(|c| c.muscles.len() == MAX_MUSCLES));
        let check = Config {
            population: 1,
            ..cfg.clone()
        };
        for (index, (name, _)) in OPERATORS.iter().enumerate() {
            for (i, body) in bodies.iter().enumerate() {
                for variant in 0..3u32 {
                    let mut c = body.clone();
                    let mut rng = Rng::new(29, index as u32 + 1000 * variant, i);
                    let donor = &bodies[(i + 1 + variant as usize) % bodies.len()];
                    let cx = Context { donor: Some(donor) };
                    if !apply(index, &mut c, &cfg, &mut rng, &cx) {
                        continue;
                    }
                    repair(&mut c, &cfg, &mut rng);
                    let mut pop = Population::default();
                    pop.push(c.clone());
                    if let Err(error) = pop.validate(&check) {
                        panic!("{name} on full body {i}: {error:#}");
                    }
                }
            }
        }
        // Lineages that stay at the caps: random operators one after another.
        for (i, body) in bodies.iter().enumerate().take(24) {
            let mut c = body.clone();
            let mut rng = Rng::new(31, 0, i);
            for _ in 0..100 {
                let donor = &bodies[rng.index(bodies.len())];
                let cx = Context { donor: Some(donor) };
                let index = rng.index(OPERATORS.len());
                if apply(index, &mut c, &cfg, &mut rng, &cx) {
                    c = crate::evolution::local_mutation(c, &cfg, &mut rng, 0.1);
                    repair(&mut c, &cfg, &mut rng);
                }
            }
            let mut pop = Population::default();
            pop.push(c);
            pop.validate(&check).unwrap();
        }
    }

    #[test]
    fn chained_operators_keep_bodies_valid_at_tight_limits() {
        // Lineages at the limits pile up muscles: apply random operators one
        // after another, repairing in between as breeding does.
        let cfg = Config {
            max_nodes: 8,
            max_muscles: 8,
            ..Config::default()
        };
        let bodies = bodies(&cfg, 160);
        let check = Config {
            population: 1,
            ..cfg.clone()
        };
        for (i, body) in bodies.iter().enumerate() {
            let mut c = body.clone();
            let mut rng = Rng::new(17, 0, i);
            for step in 0..200 {
                let donor = &bodies[rng.index(bodies.len())];
                let cx = Context { donor: Some(donor) };
                let index = rng.index(OPERATORS.len());
                if !apply(index, &mut c, &cfg, &mut rng, &cx) {
                    continue;
                }
                // Breeding follows with a parameter mutation; the game test
                // at these limits uses mutation 5.
                c = crate::evolution::local_mutation(c, &cfg, &mut rng, 0.175);
                repair(&mut c, &cfg, &mut rng);
                let mut pop = Population::default();
                pop.push(c.clone());
                if let Err(error) = pop.validate(&check) {
                    panic!(
                        "{} on body {i} step {step}: {error:#}\n{c:?}",
                        OPERATORS[index].0
                    );
                }
            }
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
            let Some(root) = copy_branch(&mut c, &cfg, leaf, at, |p| p, false, 0.5) else {
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
