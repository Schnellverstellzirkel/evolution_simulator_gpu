//! Gait operators: phase patterns of whole gaits: walk, trot, pace, canter, gallop, bound, pronk, waves.
//!
//! The operators of this file share one pick slot and are compound: each is a
//! whole, coherent change to the body, and its child gets no parameter noise.
use super::Operator;

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[];
