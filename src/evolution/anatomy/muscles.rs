//! Operators that add, move, split, fuse and retime muscles.
use super::Context;
use crate::config::Config;
use crate::evolution::{Creature, Rng};

/// Adds a muscle between two bones separated by one intermediate bone (for
/// example trunk to lower leg), timed like an existing muscle on either end:
/// one contraction moves two joints together.
pub(crate) fn add_biarticular_muscle(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Moves one end of a muscle to a bone that shares a node with its current
/// bone, near that shared node, and refits its stroke to the new span. The
/// muscle then controls a different joint.
pub(crate) fn move_muscle_to_neighbor(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Duplicates a muscle; the copy moves one anchor or shifts its phase a
/// little, so one connection can specialize into two.
pub(crate) fn split_muscle(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Merges two muscles on the same pair of bones with nearby anchors and
/// similar phase into one with averaged genes.
pub(crate) fn fuse_similar_muscles(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// For a joint whose bones already have a muscle that closes it, adds a
/// muscle that opens it: from the child bone to a bone on the other side of
/// the joint (a sibling or the bone beyond), checked geometrically to rotate
/// the child the other way, with the phase half a cycle from the closer.
pub(crate) fn add_antagonist(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Exchanges the destinations of two muscles: A-B and C-D become A-D and
/// C-B (skipping pairs that would join a bone to itself), strokes refitted.
pub(crate) fn swap_muscle_routes(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Spreads the anchors of several muscles that share a bone and sit close
/// together evenly along that bone, keeping their timing, so they act at
/// different leverages.
pub(crate) fn fan_muscle_attachments(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Replaces a muscle between two bones that are not neighbours with two
/// muscles through a bone on the path between them, starting in phase.
pub(crate) fn relay_muscle(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Copies the muscle pattern of one limb (attachments, relative strokes and
/// timing) onto another limb with the same number of bones, scaled to the
/// recipient's lengths. The recipient keeps its skeleton.
pub(crate) fn copy_actuation_to_limb(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Shrinks the strokes of every muscle in a branch by one factor (0.3 to
/// 0.7), keeping geometry and timing: a limb that fights the gait becomes a
/// quieter support.
pub(crate) fn quiet_muscle_group(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}
