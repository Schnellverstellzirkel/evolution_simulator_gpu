//! Physics checks on the CUDA kernel, the only physics: frozen muscles act
//! like unpowered ones, overlapping nodes stay finite, a creature scores the
//! same in any batch, nodes stay on top of rough ground, joints keep their
//! range and do not spin through a half turn, and a body without drive
//! neither travels nor rises.
//!
//! Needs the RTX 4060 and is ignored by default. Run it with
//!
//!     cargo test --release --test cuda_physics -- --ignored
/// The helpers the example tools share (`examples/common/mod.rs`). They open
/// the GPU engine, score creatures and record replays.
#[path = "../examples/common/mod.rs"]
mod common;
use evolution_simulator::{
    config::Config,
    creature_kernel::GpuResult,
    engine::ThreadedEngine,
    evolution::{self, Bone, Creature, Muscle, NodeGene, Population},
    physics,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// The GPU engine all tests of this file share, opened on first use. It also
/// records replays. A test that fails while it holds the engine poisons the
/// lock, and the next test takes the engine anyway.
fn gpu() -> MutexGuard<'static, ThreadedEngine> {
    static GPU: OnceLock<Mutex<ThreadedEngine>> = OnceLock::new();
    GPU.get_or_init(|| Mutex::new(common::open().expect("the GPU")))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Every creature's result on the GPU, in population order.
fn evaluate(pop: &Population, cfg: &Config) -> Vec<GpuResult> {
    common::score(&mut gpu(), pop, cfg).expect("GPU trials")
}

/// The distance in meters that one creature scores on its own. This is its
/// fitness.
fn evaluate_one(creature: &Creature, cfg: &Config) -> f32 {
    let mut pop = Population::default();
    pop.push(creature.clone());
    evaluate(&pop, cfg)[0].fitness
}

/// A creature's full trial recorded by the scoring kernel, and its result.
/// Frame `physics::settle() + n` holds the node positions after `n` steps.
/// The frames before it repeat the start pose.
fn replay(creature: &Creature, cfg: &Config) -> (Vec<Vec<[f32; 2]>>, GpuResult) {
    // Opens the shared engine on first use, which makes it the replay GPU.
    drop(gpu());
    let recording = common::record(creature, cfg).expect("a GPU replay");
    (recording.frames, recording.result)
}

/// The settings the tests start from. They use a fixed seed, 32 creatures and
/// 1 s trials on flat ground. A test overrides what it needs.
fn config() -> Config {
    Config {
        population: 32,
        random_seed: false,
        duration: 1.0,
        ..Default::default()
    }
}

/// A muscle frozen at the target length it starts with has no drive. Its body
/// scores the same as one whose muscles also have no stiffness.
#[test]
#[ignore = "needs the RTX 4060"]
fn frozen_muscles_behave_like_unpowered_muscles() {
    let cfg = config();
    let mut fixed = evolution::create(&cfg).unwrap().creature(0);
    for muscle in &mut fixed.muscles {
        let initial = physics::target(muscle, 0.0);
        muscle.short = initial;
        muscle.long = initial;
    }
    let mut unpowered = fixed.clone();
    for muscle in &mut unpowered.muscles {
        muscle.stiffness = 0.0;
    }
    let fixed_score = evaluate_one(&fixed, &cfg);
    let unpowered_score = evaluate_one(&unpowered, &cfg);
    assert!(
        (fixed_score - unpowered_score).abs() < 1e-5,
        "frozen {fixed_score} m, unpowered {unpowered_score} m"
    );
}

/// Three nodes on the same point, joined by two bones shorter than a node is
/// wide, still score a finite distance.
#[test]
#[ignore = "needs the RTX 4060"]
fn overlapping_nodes_remain_finite() {
    let c = Creature {
        nodes: vec![
            NodeGene {
                x: 0.,
                y: 0.,
                diameter: 0.08,
                friction: 0.5
            };
            3
        ]
        .into(),
        bones: vec![Bone::new(0, 1, 0.03), Bone::new(1, 2, 0.03)].into(),
        muscles: vec![Muscle {
            bone_a: 0,
            bone_b: 1,
            anchor_a: 0.5,
            anchor_b: 0.5,
            short: 0.1,
            long: 0.2,
            period: 1.,
            phase: 0.,
            duty: 0.5,
            stiffness: 80.,
            sensor: 255,
            reset: 0.0,
            tendon: 0.0,
        }]
        .into(),
        id: 1,
    };
    assert!(evaluate_one(&c, &config()).is_finite());
}

/// A creature scores the same alone as in a batch of other body sizes, in
/// another order. Its fitness, ground contact and gait frequency match bit
/// for bit.
#[test]
#[ignore = "needs the RTX 4060"]
fn a_creature_scores_the_same_in_any_batch() {
    let cfg = Config {
        population: 8,
        duration: 0.5,
        max_nodes: evolution_simulator::kernel::MAX_NODES,
        max_muscles: 256,
        min_size: 0.01,
        min_friction: 0.0,
        ..config()
    };
    // Eight bodies of different sizes. Each has its nodes on a circle, joined
    // in a chain by bones, with muscles between neighboring bones.
    let mut mixed = Population::default();
    for (i, count) in [3, 5, 6, 8, 9, 17, 24, 32].into_iter().enumerate() {
        let nodes: Vec<_> = (0..count)
            .map(|j| {
                let angle = j as f32 / count as f32 * std::f32::consts::TAU;
                NodeGene {
                    x: angle.cos() * 0.3,
                    y: angle.sin() * 0.3 + 0.4,
                    diameter: 0.02,
                    friction: 0.5,
                }
            })
            .collect();
        let bones: Vec<_> = (0..count - 1)
            .map(|j| {
                let a = &nodes[j];
                let b = &nodes[j + 1];
                Bone::new(
                    j as u32,
                    (j + 1) as u32,
                    (a.x - b.x).hypot(a.y - b.y).max(0.03),
                )
            })
            .collect();
        let muscle_links = if bones.len() > 2 { bones.len() } else { 1 };
        let muscles = (0..muscle_links)
            .map(|j| Muscle {
                bone_a: j as u32,
                bone_b: ((j + 1) % bones.len()) as u32,
                anchor_a: 0.0,
                anchor_b: 1.0,
                short: 0.06,
                long: 0.1,
                period: 1.,
                phase: 0.2,
                duty: 0.5,
                stiffness: 20.,
                sensor: 255,
                reset: 0.0,
                tendon: 0.0,
            })
            .collect();
        mixed.push(Creature {
            nodes: nodes.into(),
            bones: bones.into(),
            muscles,
            id: i as u64,
        });
    }
    mixed.validate(&cfg).unwrap();
    let order = [7usize, 5, 3, 1, 6, 4, 0, 2];
    let mut batch = Population::default();
    for &i in &order {
        batch.push(mixed.creature(i));
    }
    let combined = evaluate(&batch, &cfg);
    assert!(
        combined
            .iter()
            .all(|r| r.fitness.is_finite() && r.fitness > evolution::FAILED)
    );
    for (slot, &i) in order.iter().enumerate() {
        let mut alone = Population::default();
        alone.push(mixed.creature(i));
        let separate = evaluate(&alone, &cfg)[0];
        let together = combined[slot];
        assert_eq!(
            together.fitness.to_bits(),
            separate.fitness.to_bits(),
            "creature {i}: {} m in the batch, {} m alone",
            together.fitness,
            separate.fitness
        );
        assert_eq!(
            together.ground_contact.to_bits(),
            separate.ground_contact.to_bits(),
            "creature {i}"
        );
        assert_eq!(
            together.gait_frequency.to_bits(),
            separate.gait_frequency.to_bits(),
            "creature {i}"
        );
    }
}

/// On rough ground with a slope, no node sinks more than 2 cm below its floor
/// in any frame recorded after a step. A node's floor is the height of its
/// center when it rests on the ground, which depends on its radius and the
/// local slope.
#[test]
#[ignore = "needs the RTX 4060"]
fn nodes_stay_on_top_of_rough_ground() {
    let cfg = Config {
        population: 16,
        duration: 3.0,
        terrain: 3,
        slope: 0.15,
        ..config()
    };
    let amplitude = physics::terrain_amplitude(cfg.terrain);
    let pop = evolution::create(&cfg).unwrap();
    for i in 0..pop.genomes.len() {
        let creature = pop.creature(i);
        let (frames, _) = replay(&creature, &cfg);
        for frame in &frames[physics::settle() as usize + 1..] {
            for (node, gene) in frame.iter().zip(&creature.nodes) {
                let (height, slope) = physics::terrain_with_slope(node[0], amplitude, cfg.slope);
                let floor = height + gene.diameter * 0.5 * (1.0 + slope * slope).sqrt();
                assert!(
                    // The kernel moves a sunk node back out along the
                    // ground's normal in every substep, so a node can sit
                    // only a little inside the floor computed here.
                    node[1] >= floor - 0.02,
                    "node sank to {} below {floor}",
                    node[1]
                );
            }
        }
    }
}

/// Signed angle (rad) at node `pivot` from the direction of node `reference`
/// to the direction of node `child`, for the node positions `p` of one frame.
fn joint_angle(p: &[[f32; 2]], pivot: usize, reference: usize, child: usize) -> f32 {
    let u = [p[reference][0] - p[pivot][0], p[reference][1] - p[pivot][1]];
    let v = [p[child][0] - p[pivot][0], p[child][1] - p[pivot][1]];
    (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1])
}

/// A joint stays within the range its bone carries. The test narrows every
/// range to 0.3 rad on each side of the start pose and checks each frame up
/// to the fall or the break.
#[test]
#[ignore = "needs the RTX 4060"]
fn joints_stay_within_their_evolved_range() {
    let cfg = Config {
        population: 64,
        duration: 10.0,
        ..config()
    };
    let pop = evolution::create(&cfg).unwrap();
    let mut worst = 0.0f32;
    for i in 0..pop.genomes.len() {
        let mut creature = pop.creature(i);
        // Every joint may turn 0.3 rad either way from its start pose.
        for bone in &mut creature.bones {
            bone.min_angle = -0.3;
            bone.max_angle = 0.3;
        }
        let joints = physics::joints(&creature.nodes, &creature.bones);
        let start: Vec<[f32; 2]> = creature.nodes.iter().map(|n| [n.x, n.y]).collect();
        let (frames, result) = replay(&creature, &cfg);
        // Scoring ends when the head falls or a joint breaks. The recording
        // goes on with limp muscles, and a limp body may fold any way.
        let end = if result.fall_time > 0.0 {
            physics::settle() as usize
                + (result.fall_time * physics::rate() as f32).round() as usize
                + 1
        } else {
            frames.len()
        }
        .min(frames.len());
        for (bone, joint) in creature.bones.iter().zip(&joints) {
            let Some(reference) = joint.reference else {
                continue;
            };
            let (pivot, child) = (bone.a as usize, bone.b as usize);
            let rest = joint_angle(&start, pivot, reference, child);
            for frame in &frames[..end] {
                // The joint's turn from its start pose, wrapped to a half
                // turn either way.
                let offset = (joint_angle(frame, pivot, reference, child) - rest
                    + std::f32::consts::PI)
                    .rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI;
                worst = worst.max(offset.abs() - 0.3);
            }
        }
    }
    // A joint can pass its limit for a few steps. Past `physics::JOINT_BREAK`
    // (0.5 rad) it breaks and scoring ends. The bound is far from the half
    // turn a wheel would need.
    assert!(worst < 0.75, "a joint left its range by {worst} rad");
}

/// With the widest joint ranges, no joint turns through a half turn from its
/// start pose in any frame of a trial.
#[test]
#[ignore = "needs the RTX 4060"]
fn full_joint_ranges_do_not_spin_through_a_half_turn() {
    let cfg = Config {
        population: 64,
        duration: 10.0,
        ..config()
    };
    let mut pop = evolution::create(&cfg).unwrap();
    for bone in &mut pop.bones {
        bone.min_angle = -evolution::JOINT_LIMIT;
        bone.max_angle = evolution::JOINT_LIMIT;
    }
    let mut widest_turn = 0.0f32;
    for i in 0..pop.genomes.len() {
        let creature = pop.creature(i);
        let joints = physics::joints(&creature.nodes, &creature.bones);
        let (frames, _) = replay(&creature, &cfg);
        for (bone, joint) in creature.bones.iter().zip(&joints) {
            let Some(reference) = joint.reference else {
                continue;
            };
            let (pivot, child) = (bone.a as usize, bone.b as usize);
            let mut previous = joint_angle(&frames[0], pivot, reference, child);
            // The joint's total turn since the first frame, summed from its
            // change between frames. A joint that spins keeps adding up
            // instead of wrapping at a half turn.
            let mut unwrapped = 0.0f32;
            for frame in &frames[1..] {
                let current = joint_angle(frame, pivot, reference, child);
                let delta = (current - previous + std::f32::consts::PI)
                    .rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI;
                unwrapped += delta;
                widest_turn = widest_turn.max(unwrapped.abs());
                previous = current;
            }
        }
    }
    assert!(
        widest_turn < std::f32::consts::PI,
        "a joint turned past a half turn: {widest_turn} rad"
    );
}

#[test]
#[ignore = "needs the RTX 4060"]
fn a_body_without_drive_does_not_travel() {
    // Muscles whose target never changes cannot drive, so nothing but the
    // solver could move these bodies sideways on flat ground.
    let cfg = Config {
        population: 32,
        duration: 5.0,
        ..config()
    };
    let mut pop = evolution::create(&cfg).unwrap();
    for muscle in &mut pop.muscles {
        muscle.short = muscle.long;
    }
    let results = evaluate(&pop, &cfg);
    let worst = results.iter().map(|r| r.fitness.abs()).fold(0.0, f32::max);
    eprintln!("worst drift without drive: {worst} m");
    // A collapsing body can slide a little through real friction, but it
    // must never travel.
    assert!(worst < 0.5, "a body drifted {worst} m with no muscle drive");
}

#[test]
#[ignore = "needs the RTX 4060"]
fn a_passive_body_never_rises_above_its_start() {
    // Without muscle drive, gravity can only lower a body. Rising above its
    // starting height would be energy the solver created.
    let cfg = Config {
        population: 32,
        duration: 5.0,
        ..config()
    };
    let mut pop = evolution::create(&cfg).unwrap();
    for muscle in &mut pop.muscles {
        muscle.short = muscle.long;
    }
    let settle = physics::settle() as usize;
    for i in 0..pop.genomes.len() {
        let creature = pop.creature(i);
        let masses: Vec<f32> = physics::nodes(&creature).iter().map(|n| n.mass).collect();
        let total: f32 = masses.iter().sum();
        let (frames, _) = replay(&creature, &cfg);
        // The height of the body's center of mass in one frame.
        let height = |frame: &Vec<[f32; 2]>| {
            frame
                .iter()
                .zip(&masses)
                .map(|(p, m)| p[1] * m)
                .sum::<f32>()
                / total
        };
        // The start is the first frame recorded after a step.
        let start = height(&frames[settle + 1]);
        let highest = frames[settle + 1..]
            .iter()
            .map(height)
            .fold(f32::MIN, f32::max);
        assert!(
            highest <= start + 0.02,
            "body {i} rose from {start} m to {highest} m without muscle drive"
        );
    }
}
