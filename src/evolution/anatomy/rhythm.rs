//! Operators that change joint ranges, timing patterns and mass together.
use super::Context;
use crate::config::Config;
use crate::evolution::{Creature, Rng};

/// Narrows one joint's range and widens a neighbouring joint's range by the
/// same angle (within `JOINT_LIMIT`): flexibility moves along the limb.
pub(crate) fn redistribute_joint_flex(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Finds two branches of the same shape (same bone count, similar lengths)
/// and applies one random change to both: bone lengths, joint ranges or
/// anchors. Their timing difference stays.
pub(crate) fn mutate_matching_limbs(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Along a chain of bones (a path down one branch), sets the phase of the
/// muscles on successive bones to grow by one step per bone: a contraction
/// wave, for curling and crawling.
pub(crate) fn chain_phase_wave(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Shifts whole limbs (the branches at one junction, or every leaf branch) to
/// a pattern of phase offsets: all together, alternating halves, or evenly
/// staggered, keeping the timing inside each limb.
pub(crate) fn limb_phase_pattern(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Changes the duty of every muscle in a limb by one amount and moves their
/// phases so each contraction keeps its middle: a slower push with a quicker
/// return, or the reverse.
pub(crate) fn limb_duty_cycle(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Picks a foot (a leaf node) and makes every muscle on a bone at that foot
/// sense it, with reset phases that keep their current order, so landing
/// restarts the limb's movement as a whole.
pub(crate) fn touchdown_package(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}

/// Moves part of one organ's mass to another bone (creating an organ there
/// if needed), keeping the total organ mass and the organ limits.
pub(crate) fn redistribute_organ_mass(
    _c: &mut Creature,
    _cfg: &Config,
    _rng: &mut Rng,
    _cx: &Context,
) -> bool {
    false
}
