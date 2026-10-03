//! Gait operators: whole body plans: quadruped, hexapod, hopper, myriapod, and moves between them.
//!
//! The operators of this file share one pick slot and are compound: each is a
//! whole, coherent change to the body, and its child gets no parameter noise.
use super::Operator;

/// This file's operators, by name. Add each new one here.
pub(super) const OPS: &[(&str, Operator)] = &[];
