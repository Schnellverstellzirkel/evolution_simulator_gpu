//! A creature for the unit tests of the UI modules.

use crate::evolution::{Bone, Creature, Muscle, NodeGene};

pub(super) fn test_creature() -> Creature {
    Creature {
        nodes: vec![
            NodeGene {
                x: 0.0,
                y: 0.0,
                diameter: 0.2,
                friction: 0.8,
            },
            NodeGene {
                x: 0.5,
                y: 0.0,
                diameter: 0.2,
                friction: 0.8,
            },
            NodeGene {
                x: 1.0,
                y: 0.0,
                diameter: 0.2,
                friction: 0.8,
            },
        ]
        .into(),
        bones: vec![Bone::new(0, 1, 0.5), Bone::new(1, 2, 0.5)].into(),
        muscles: vec![Muscle {
            bone_a: 0,
            bone_b: 1,
            anchor_a: 0.5,
            anchor_b: 0.5,
            short: 0.4,
            long: 0.6,
            period: 0.8,
            phase: 0.0,
            duty: 0.5,
            stiffness: 10.0,
            sensor: crate::evolution::NO_SENSOR,
            reset: 0.0,
            tendon: 0.0,
        }]
        .into(),
        id: 7,
    }
}
