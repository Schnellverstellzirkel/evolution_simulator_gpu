//! Scores must describe the motion the replay actually shows, including the
//! instant when a fall or broken joint ends a trial.
use evolution_simulator::{
    config::Config,
    cpu_engine,
    creature_kernel::GpuResult,
    evolution::{self, Bone, Creature, Muscle, NodeGene, Population},
    physics::{self, Fidelity},
};

fn config(fidelity: Fidelity) -> Config {
    Config {
        population: 8,
        random_seed: false,
        seed: 38,
        duration: 1.25,
        fidelity: Some(fidelity),
        ..Config::default()
    }
}

fn evaluate(creature: &Creature, cfg: &Config) -> GpuResult {
    let mut population = Population::default();
    population.push(creature.clone());
    cpu_engine::evaluate(&population, cfg)[0]
}

fn center_x(creature: &Creature, frame: &[[f32; 2]]) -> f32 {
    let nodes = physics::nodes(creature);
    let mass: f32 = nodes.iter().map(|node| node.mass).sum();
    frame
        .iter()
        .zip(nodes)
        .map(|(position, node)| position[0] * node.mass)
        .sum::<f32>()
        / mass
}

fn terminal_frame(result: &GpuResult, cfg: &Config) -> usize {
    let fidelity = cfg.fidelity();
    let timed_steps = if result.fall_time > 0.0 {
        (result.fall_time * fidelity.rate as f32).round() as u32
    } else {
        cfg.steps()
    };
    (fidelity.settle() + timed_steps) as usize
}

fn passive_triangle(fallen: bool) -> Creature {
    let positions = if fallen {
        [[0.4, 0.06], [0.0, 0.6], [-0.6, 0.06]]
    } else {
        [[0.0, 0.8], [-0.5, 0.05], [0.5, 0.05]]
    };
    let nodes: Vec<_> = positions
        .iter()
        .map(|&[x, y]| NodeGene {
            x,
            y,
            diameter: 0.1,
            friction: 0.8,
        })
        .collect();
    let connections = if fallen {
        [(0, 1), (1, 2)]
    } else {
        [(0, 1), (0, 2)]
    };
    let bones = connections
        .into_iter()
        .map(|(a, b)| {
            let distance = (nodes[a].x - nodes[b].x).hypot(nodes[a].y - nodes[b].y);
            Bone::new(a as u32, b as u32, distance)
        })
        .collect();
    Creature {
        nodes,
        bones,
        muscles: Vec::new(),
        id: 0,
        mutability: 1.0,
    }
}

#[test]
fn evaluated_distance_matches_mass_weighted_terminal_replay_frame() {
    for fidelity in [Fidelity::standard(), Fidelity::fine()] {
        let cfg = config(fidelity);
        let mut population = evolution::create(&cfg).unwrap();
        population.push(passive_triangle(false));
        population.push(passive_triangle(true));
        let results = cpu_engine::evaluate(&population, &cfg);
        assert!(results.iter().any(|result| result.fall_time == 0.0));
        assert!(results.iter().any(|result| result.fall_time > 0.0));
        for (slot, result) in results.iter().enumerate() {
            let creature = population.creature(slot);
            let frames = cpu_engine::trajectory(&creature, &cfg);
            assert_eq!(frames.len(), (fidelity.settle() + cfg.steps() + 1) as usize);
            let terminal = terminal_frame(result, &cfg);
            let replay_distance = center_x(&creature, &frames[terminal]);
            assert!(
                (result.fitness - replay_distance).abs() < 1e-5,
                "{fidelity:?}, creature {slot}, frame {terminal}: score {}, replay {replay_distance}",
                result.fitness
            );
        }
    }
}

#[test]
fn partial_simd_groups_keep_results_in_order_and_match_single_creatures() {
    for fidelity in [Fidelity::standard(), Fidelity::fine()] {
        let cfg = config(fidelity);
        let template = evolution::create(&cfg).unwrap().creature(0);
        let mut population = Population::default();
        // One complete SIMD group and a partially filled group of the same
        // body plan, with distinct parameters so a slot mixup is observable.
        for index in 0..19 {
            let mut creature = template.clone();
            creature.nodes[0].diameter *= 1.0 + index as f32 * 0.01;
            for muscle in &mut creature.muscles {
                muscle.phase = (muscle.phase + index as f32 * 0.03125).fract();
            }
            population.push(creature);
        }
        let combined = cpu_engine::evaluate(&population, &cfg);
        for (index, result) in combined.iter().enumerate() {
            let single = evaluate(&population.creature(index), &cfg);
            // Contact masks are stored as f32 bit patterns, so compare bytes
            // rather than float equality (a valid mask may encode a NaN).
            assert_eq!(
                bytemuck::bytes_of(result),
                bytemuck::bytes_of(&single),
                "{fidelity:?}, creature {index}: grouped {result:?}, single {single:?}"
            );
        }
    }
}

#[test]
fn extending_a_fallen_creatures_trial_keeps_its_score_at_the_fall() {
    let creature = passive_triangle(true);
    for fidelity in [Fidelity::standard(), Fidelity::fine()] {
        let short_cfg = Config {
            duration: 0.25,
            ..config(fidelity)
        };
        let long_cfg = Config {
            duration: 3.0,
            ..short_cfg.clone()
        };
        let short = evaluate(&creature, &short_cfg);
        let long = evaluate(&creature, &long_cfg);
        assert!(short.fall_time > 0.0 && short.fall_time < short_cfg.duration);
        assert_eq!(short.fall_time, long.fall_time);
        assert_eq!(short.fitness, long.fitness);
        let frames = cpu_engine::trajectory(&creature, &long_cfg);
        let terminal = terminal_frame(&long, &long_cfg);
        let neck = creature.bones[0].b as usize;
        assert!(frames[terminal][0][1] < frames[terminal][neck][1]);
        let at_fall = center_x(&creature, &frames[terminal]);
        assert!((long.fitness - at_fall).abs() < 1e-5);
        assert!(
            (center_x(&creature, frames.last().unwrap()) - at_fall).abs() > 1e-3,
            "fixture must keep moving after its score freezes"
        );
    }
}

#[test]
fn extending_a_broken_joints_trial_keeps_its_score_at_the_break() {
    // Long-range muscles pull a narrow-jointed chain against the ground.
    // Keep the head above the chain so a head fall cannot stand in for a break.
    let node_count = 16;
    let nodes: Vec<_> = (0..node_count)
        .map(|i| {
            let [x, y] = match i {
                0 => [-0.2, 0.8],
                1 => [-0.15, 0.6],
                _ => [(i - 2) as f32 * 0.08, 0.08 + (i % 2) as f32 * 0.04],
            };
            NodeGene {
                x,
                y,
                diameter: 0.08,
                friction: 0.8,
            }
        })
        .collect();
    let bones: Vec<_> = (0..node_count - 1)
        .map(|i| {
            let a = nodes[i];
            let b = nodes[i + 1];
            let mut bone = Bone::new(i as u32, (i + 1) as u32, (a.x - b.x).hypot(a.y - b.y));
            bone.min_angle = 0.0;
            bone.max_angle = 0.0;
            bone
        })
        .collect();
    let muscles = (1..bones.len())
        .map(|i| Muscle {
            bone_a: i as u32,
            bone_b: ((i + node_count / 2) % bones.len()) as u32,
            anchor_a: 1.0,
            anchor_b: 0.0,
            short: 0.1,
            long: 0.3,
            period: 0.2 + (i % 3) as f32 * 0.05,
            phase: (i % 7) as f32 / 7.0,
            duty: 0.5,
            stiffness: 120.0,
            sensor: evolution::NO_SENSOR,
            reset: 0.0,
        })
        .collect();
    let creature = Creature {
        nodes,
        bones,
        muscles,
        id: 0,
        mutability: 1.0,
    };
    let cfg = Config {
        duration: 3.0,
        ..config(Fidelity::standard())
    };
    let result = evaluate(&creature, &cfg);
    assert!(result.fall_time > 0.0 && result.fall_time < cfg.duration - 0.25);
    let frames = cpu_engine::trajectory(&creature, &cfg);
    let terminal = terminal_frame(&result, &cfg);
    let frame = &frames[terminal];
    let neck = creature.bones[0].b as usize;
    let joints = physics::joints(&creature.nodes, &creature.bones);
    assert!(
        frame[0][1] >= frame[neck][1],
        "the head must remain upright"
    );
    assert!(physics::broken_joint(frame, &creature.bones, &joints));
    let shorter = Config {
        duration: result.fall_time + 0.1,
        ..cfg.clone()
    };
    let shorter_result = evaluate(&creature, &shorter);
    assert_eq!(shorter_result.fall_time, result.fall_time);
    assert_eq!(shorter_result.fitness, result.fitness);
    assert!((center_x(&creature, frame) - result.fitness).abs() < 1e-5);
}

#[test]
fn public_physics_evaluation_uses_the_same_physics_as_cpu_replay() {
    for fidelity in [Fidelity::standard(), Fidelity::fine()] {
        for terrain in [0, 3] {
            let cfg = Config {
                population: 4,
                terrain,
                ..config(fidelity)
            };
            let population = evolution::create(&cfg).unwrap();
            let results = cpu_engine::evaluate(&population, &cfg);
            for (slot, result) in results.iter().enumerate() {
                let legacy = physics::evaluate(&population.creature(slot), &cfg);
                assert!(
                    (legacy - result.fitness).abs() < 1e-5,
                    "{fidelity:?}, terrain {terrain}, creature {slot}: physics::evaluate {legacy}, CPU {}",
                    result.fitness
                );
            }
        }
    }
}
