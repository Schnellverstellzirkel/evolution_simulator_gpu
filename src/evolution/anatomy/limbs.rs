//! Operators that copy, grow, fuse, move and reshape whole limbs.
use super::Context;
use crate::config::Config;
use crate::evolution::{Creature, Rng};

/// Copies a complete branch (several bones, their joints, the muscles inside
/// it and the muscles from its root to the bone above) onto the same joint or
/// another joint, mirrored or not, with the copied muscles shifted by one of
/// 0, 1/4, 1/2 or 3/4 of a cycle. A working bent leg becomes a second leg.
pub(crate) fn copy_limb(_c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    false
}

/// Extends a limb tip with a short bone (a fraction of the tip bone), a joint
/// with a narrow range, and a muscle from the new bone to the bone above,
/// timed like a muscle near it. A direct route to ankles and toes.
pub(crate) fn grow_actuated_tip(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Splits a bone at a random point between 30% and 70% of its length. The new
/// joint starts with a narrow range, existing attachments stay where they are
/// on the body, and a muscle across the new joint (timed like one nearby)
/// makes it an elbow or knee under control instead of a floppy hinge.
pub(crate) fn split_bone_actuated(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Fuses two nearly aligned bones that meet at a node with no other bone
/// (not the head or the neck) into one bone; muscles on either keep their
/// place on the body. Evolution can decide which regions stay rigid.
pub(crate) fn fuse_bones(_c: &mut Creature, _cfg: &Config, _rng: &mut Rng, _cx: &Context) -> bool {
    false
}

/// Moves a branch, with its internal shape and muscles, to another node of
/// the body (not inside the branch, not the head). Muscles from the branch
/// root to the old bone above move to the bone above the new node.
pub(crate) fn relocate_limb(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Scales every bone of a branch by one factor (0.7 to 1.4, within the bone
/// limits) and the strokes of the muscles inside it with them, so a limb
/// gets longer or shorter without scrambling its parts.
pub(crate) fn reshape_limb(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Copies a branch of `cx.donor` (another archive elite) onto a node of this
/// body, with the donor's joints, internal muscles and their timing. With
/// even odds it replaces a branch of this body instead of adding one.
pub(crate) fn graft_donor_limb(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}
