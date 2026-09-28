//! Operators that restructure junctions and segments of the skeleton.
use super::Context;
use crate::config::Config;
use crate::evolution::{Creature, Rng};

/// Where three or more bones meet, puts a short new bone between the node
/// and a new node, and moves some of the child branches to the new node, so
/// a crowded junction becomes two joints (a shoulder and a hip region).
pub(crate) fn split_crowded_joint(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Collapses a short bone between two junctions (its child node has two or
/// more child bones) so its child branches meet at the parent node. Muscles
/// on the removed bone move to a neighbouring bone or go.
pub(crate) fn merge_branch_joints(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Copies a trunk bone (one with child bones) together with the leaf limbs
/// on its child node and their muscles, and inserts the copy after it in the
/// chain: a route to segmented, many-legged bodies.
pub(crate) fn repeat_body_segment(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Turns a limb tip into a heel and a toe: two short bones from the tip, one
/// pointing forward and one back, with their own joint ranges and a muscle
/// from each to the tip bone.
pub(crate) fn grow_heel_toe(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Grows a short bone with a narrow joint range from a joint node, and moves
/// one attachment of a muscle on a bone at that joint onto the new bone's
/// tip. A lever that changes the muscle's leverage (like an elbow process).
pub(crate) fn grow_lever_spur(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Reflects a branch across the line of its root bone and swaps and negates
/// the joint limits inside it, so the limb bends the other way. Anchors along
/// bones stay, so its muscles keep their roles.
pub(crate) fn reverse_bend(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}
