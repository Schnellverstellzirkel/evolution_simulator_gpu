use evolution_simulator::physics::Fidelity;
use evolution_simulator::{
    config::Config,
    evolution::{self, Bone, Creature, Muscle, NodeGene},
    gpu::Gpu,
    physics::{self, Node},
    storage::{self, Experiment, Stage},
};
use std::path::PathBuf;
fn config() -> Config {
    Config {
        population: 32,
        random_seed: false,
        duration: 1.0,
        ..Default::default()
    }
}
fn path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("evolution-{}-{name}.evo", std::process::id()))
}
fn stress_creature() -> Creature {
    let node_count = 64;
    let nodes: Vec<_> = (0..node_count)
        .map(|i| NodeGene {
            x: (i as f32 - (node_count - 1) as f32 * 0.5) * 0.12,
            y: 0.09 + (i % 2) as f32 * 0.05,
            diameter: 0.08,
            friction: 0.5,
        })
        .collect();
    let bones: Vec<_> = (0..node_count - 1)
        .map(|i| {
            let a = &nodes[i];
            let b = &nodes[i + 1];
            Bone::new(i as u32, (i + 1) as u32, (a.x - b.x).hypot(a.y - b.y))
        })
        .collect();
    let muscles: Vec<_> = (0..bones.len())
        .map(|i| Muscle {
            bone_a: i as u32,
            bone_b: ((i + 31) % bones.len()) as u32,
            anchor_a: if i % 2 == 0 { 0.0 } else { 1.0 },
            anchor_b: if i % 3 == 0 { 1.0 } else { 0.0 },
            short: 0.02,
            long: 0.24,
            period: 0.1 + (i % 5) as f32 * 0.07,
            phase: (i % 7) as f32 / 7.0,
            duty: 0.5,
            stiffness: 120.0,
            sensor: 255,
            reset: 0.0,
        })
        .collect();
    Creature {
        nodes,
        bones,
        muscles,
        id: 0,
        mutability: 1.0,
    }
}

#[test]
fn seed_is_repeatable_and_selection_conserves_population() {
    let cfg = config();
    let a = evolution::create(&cfg).unwrap();
    let b = evolution::create(&cfg).unwrap();
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.bones, b.bones);
    assert_eq!(a.muscles, b.muscles);
    let scores: Vec<_> = (0..cfg.population).map(|i| i as f32).collect();
    let ranks = evolution::ranking(&scores);
    let parents = evolution::survivors(&cfg, 0, &ranks);
    assert_eq!(parents.len(), cfg.population / 2);
    let next = evolution::reproduce(&a, &cfg, 0, &parents).unwrap();
    next.validate(&cfg).unwrap();
    assert_eq!(next.genomes.len(), cfg.population);
    let again = evolution::reproduce(&a, &cfg, 0, &parents).unwrap();
    assert_eq!(next.nodes, again.nodes);
    assert_eq!(next.bones, again.bones);
    assert_eq!(next.muscles, again.muscles);
}
#[test]
fn zero_mutation_copies_genetics() {
    let cfg = Config {
        mutation: 0.,
        ..config()
    };
    let p = evolution::create(&cfg).unwrap();
    let parents: Vec<_> = (0..cfg.population / 2).collect();
    let next = evolution::reproduce(&p, &cfg, 0, &parents).unwrap();
    for i in 0..cfg.population {
        let a = p.creature(parents[i / 2]);
        let b = next.creature(i);
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.bones, b.bones);
        assert_eq!(a.muscles, b.muscles);
    }
}
#[test]
fn mutation_keeps_valid_graphs_at_limits() {
    let cfg = Config {
        population: 64,
        max_nodes: 8,
        max_muscles: 8,
        mutation: 5.,
        ..config()
    };
    let mut p = evolution::create(&cfg).unwrap();
    for generation in 0..80 {
        let parents: Vec<_> = (0..cfg.population / 2).collect();
        p = evolution::reproduce(&p, &cfg, generation, &parents).unwrap();
        p.validate(&cfg).unwrap();
    }
}
#[test]
fn flat_ground_contact_resolves_nodes_and_can_be_disabled() {
    let cfg = config();
    let mut node = Node {
        pos: [0., -0.1],
        vel: [1., -2.],
        radius: 0.04,
        friction: 1.,
        mass: 0.1,
        failed: 0.,
    };
    physics::collide(&mut node, &cfg);
    assert!(node.pos[1] >= node.radius - 1e-6);
    assert_eq!(node.vel, [0., 0.]);
    let cfg = Config {
        ground: false,
        ..cfg
    };
    node.pos = [0., 0.];
    node.vel = [1., -2.];
    physics::collide(&mut node, &cfg);
    assert_eq!(node.pos, [0., 0.]);
    assert_eq!(node.vel, [1., -2.]);
}
#[test]
fn muscle_cycle_is_continuous_and_periodic() {
    let m = Muscle {
        bone_a: 0,
        bone_b: 1,
        anchor_a: 1.0,
        anchor_b: 0.0,
        short: 0.1,
        long: 0.3,
        period: 2.,
        phase: 0.,
        duty: 0.4,
        stiffness: 30.,
        sensor: 255,
        reset: 0.0,
    };
    assert!((physics::target(&m, 0.) - 0.3).abs() < 1e-6);
    assert!((physics::target(&m, 0.8) - 0.1).abs() < 1e-6);
    assert!((physics::target(&m, 0.79999) - physics::target(&m, 0.80001)).abs() < 1e-5);
    assert!((physics::target(&m, 0.23) - physics::target(&m, 2.23)).abs() < 1e-6);
}

#[test]
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
    let fixed_score = physics::evaluate(&fixed, &cfg);
    let unpowered_score = physics::evaluate(&unpowered, &cfg);
    assert!((fixed_score - unpowered_score).abs() < 1e-5);
}

#[test]
fn bone_lengths_hold_and_off_center_muscles_rotate_bones() {
    let cfg = Config {
        gravity: 0.0,
        ground: false,
        air_retention: 1.0,
        ..config()
    };
    let genes = [
        NodeGene {
            x: 0.0,
            y: 0.0,
            diameter: 0.08,
            friction: 0.5,
        },
        NodeGene {
            x: 1.0,
            y: 0.0,
            diameter: 0.08,
            friction: 0.5,
        },
        NodeGene {
            x: 1.0,
            y: 1.0,
            diameter: 0.08,
            friction: 0.5,
        },
        NodeGene {
            x: 0.0,
            y: 1.0,
            diameter: 0.08,
            friction: 0.5,
        },
    ];
    let creature = Creature {
        nodes: genes.to_vec(),
        bones: vec![
            Bone::new(0, 1, 1.0),
            Bone::new(1, 2, 1.0),
            Bone::new(2, 3, 1.0),
        ],
        muscles: vec![Muscle {
            bone_a: 0,
            bone_b: 2,
            anchor_a: 0.25,
            anchor_b: 0.75,
            short: 0.1,
            long: 0.4,
            period: 1.0,
            phase: 0.25,
            duty: 0.5,
            stiffness: 40.0,
            sensor: 255,
            reset: 0.0,
        }],
        id: 1,
        mutability: 1.0,
    };
    let mut nodes = physics::nodes(&creature);
    physics::step(
        &mut nodes,
        &creature.bones,
        &creature.muscles,
        &cfg,
        physics::settle() + 1,
    );
    for bone in &creature.bones {
        let a = nodes[bone.a as usize].pos;
        let b = nodes[bone.b as usize].pos;
        let length = (a[0] - b[0]).hypot(a[1] - b[1]);
        assert!(
            (length - bone.rest_length).abs() < 0.002,
            "{bone:?}: {length}"
        );
    }
    assert!(nodes[0].vel[1] > 0.0);
    assert!(nodes[0].vel[1] > nodes[1].vel[1]);
    assert!(nodes[3].vel[1] < nodes[2].vel[1]);
}

#[test]
fn bone_lengths_hold_under_sustained_muscle_and_ground_forces() {
    let cfg = Config {
        gravity: 30.0,
        ground: true,
        air_retention: 1.0,
        ..config()
    };
    let creature = stress_creature();
    let mut body = physics::nodes(&creature);

    let mut max_error = 0.0f32;
    for tick in 0..560 {
        physics::step(&mut body, &creature.bones, &creature.muscles, &cfg, tick);
        for bone in &creature.bones {
            let a = body[bone.a as usize].pos;
            let b = body[bone.b as usize].pos;
            max_error = max_error.max((a[0] - b[0]).hypot(a[1] - b[1]) - bone.rest_length);
            max_error = max_error.max(bone.rest_length - (a[0] - b[0]).hypot(a[1] - b[1]));
        }
    }
    assert!(max_error < 0.0001, "maximum bone length error: {max_error}");
    assert!(body.iter().all(|node| node.pos[1] >= node.radius - 1e-6));
}

#[test]
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
        ],
        bones: vec![Bone::new(0, 1, 0.03), Bone::new(1, 2, 0.03)],
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
        }],
        id: 1,
        mutability: 1.,
    };
    assert!(physics::evaluate(&c, &config()).is_finite());
}
#[test]
fn partial_checkpoint_resumes_identically() {
    let mut e = Experiment::new(config()).unwrap();
    e.stage = Stage::Evaluating;
    for i in 0..7 {
        e.scores[i] = physics::evaluate(&e.population.creature(i), &e.config);
    }
    e.evaluated = 7;
    let checkpoint = path("partial");
    storage::save(&checkpoint, &e).unwrap();
    let mut loaded = storage::load(&checkpoint).unwrap();
    assert_eq!(loaded.evaluated, 7);
    for i in 7..e.config.population {
        e.scores[i] = physics::evaluate(&e.population.creature(i), &e.config);
        loaded.scores[i] = physics::evaluate(&loaded.population.creature(i), &loaded.config);
    }
    assert_eq!(e.scores, loaded.scores);
    e.evaluated = e.config.population;
    loaded.evaluated = loaded.config.population;
    e.rank();
    loaded.rank();
    e.select();
    loaded.select();
    e.reproduce().unwrap();
    loaded.reproduce().unwrap();
    assert_eq!(e.population.nodes, loaded.population.nodes);
    assert_eq!(e.population.muscles, loaded.population.muscles);
    // A stale/incomplete temporary write cannot corrupt the committed checkpoint.
    std::fs::write(checkpoint.with_extension("evo.tmp"), b"partial").unwrap();
    assert!(storage::load(&checkpoint).is_ok());
    let _ = std::fs::remove_file(checkpoint.with_extension("evo.tmp"));
    let _ = std::fs::remove_file(checkpoint);
}
#[test]
fn invalid_settings_and_checkpoints_are_rejected() {
    let mut cfg = config();
    cfg.population = 3;
    assert!(cfg.validate().is_err());
    cfg = config();
    cfg.gravity = f32::NAN;
    assert!(cfg.validate().is_err());
    let p = path("corrupt");
    std::fs::write(&p, b"garbage!").unwrap();
    assert!(storage::load(&p).is_err());
    let _ = std::fs::remove_file(p);
}
#[test]
fn statistics_count_every_creature() {
    let mut e = Experiment::new(config()).unwrap();
    e.scores = (0..e.config.population)
        .map(|i| i as f32 / 10. - 1.)
        .collect();
    e.evaluated = e.config.population;
    e.rank();
    let s = &e.history[0];
    assert_eq!(
        s.histogram.iter().map(|x| x.1 as usize).sum::<usize>(),
        e.config.population
    );
    assert_eq!(
        s.species.iter().map(|x| x.2 as usize).sum::<usize>(),
        e.config.population
    );
    assert_eq!(s.percentiles.len(), 29);
    assert!(s.best >= s.median && s.median >= s.worst);
}

#[test]
fn history_and_checksums_are_validated_on_load() {
    let mut e = Experiment::new(config()).unwrap();
    e.scores = (0..e.config.population).map(|i| i as f32 * 0.1).collect();
    e.evaluated = e.config.population;
    e.rank();
    let checkpoint = path("history");
    storage::save(&checkpoint, &e).unwrap();
    assert_eq!(storage::load(&checkpoint).unwrap().history.len(), 1);
    let mut bytes = std::fs::read(&checkpoint).unwrap();
    let end = bytes.len() - 1;
    bytes[end] ^= 1;
    std::fs::write(&checkpoint, bytes).unwrap();
    assert!(storage::load(&checkpoint).is_err());
    e.history[0].percentiles.clear();
    storage::save(&checkpoint, &e).unwrap();
    assert!(storage::load(&checkpoint).is_err());
    let _ = std::fs::remove_file(checkpoint);
}

#[test]
#[ignore = "requires a Vulkan GPU; run explicitly on the workstation"]
fn gpu_matches_cpu_and_handles_partial_workgroups() {
    let cfg = Config {
        population: 10,
        duration: 0.5,
        ..config()
    };
    let pop = evolution::create(&cfg).unwrap();
    let mut gpu = Gpu::new("RTX 4060").unwrap();
    let scores = gpu
        .evaluate(&pop, &(0..10).collect::<Vec<_>>(), &cfg)
        .unwrap();
    assert_eq!(scores.len(), 10);
    assert!(
        scores
            .iter()
            .all(|s| s.is_finite() && *s > evolution::FAILED)
    );
    // The CPU SIMD engine (also the replay) runs the same physics as the GPU
    // kernel at both fidelities; short trials keep rounding differences from
    // growing chaotically. Single trials leave out the perturbed contender
    // check, whose fall and break decisions can flip on rounding.
    let indices: Vec<usize> = (0..10).collect();
    for fidelity in [Fidelity::standard(), Fidelity::fine()] {
        let cfg = Config {
            fidelity: Some(fidelity),
            ..cfg.clone()
        };
        let gpu_scores = gpu
            .sched
            .as_mut()
            .unwrap()
            .evaluate_single(&pop, &indices, &cfg)
            .unwrap();
        let cpu = evolution_simulator::cpu_engine::evaluate(&pop, &cfg);
        for (i, (gpu_result, cpu_result)) in gpu_scores.iter().zip(&cpu).enumerate() {
            assert!(
                (gpu_result.fitness - cpu_result.fitness).abs() < 0.05,
                "{fidelity:?} score {i}: GPU {}, CPU engine {}",
                gpu_result.fitness,
                cpu_result.fitness
            );
        }
    }
    let cfg = Config {
        population: 8,
        max_nodes: 64,
        max_muscles: 256,
        min_size: 0.01,
        min_friction: 0.0,
        ..cfg
    };
    let mut mixed = evolution::Population::default();
    for (i, count) in [3, 5, 6, 8, 9, 17, 33, 64].into_iter().enumerate() {
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
            })
            .collect();
        mixed.push(Creature {
            nodes,
            bones,
            muscles,
            id: i as u64,
            mutability: 1.,
        });
    }
    mixed.validate(&cfg).unwrap();
    let order = [7usize, 5, 3, 1, 6, 4, 0, 2];
    let scores = gpu.evaluate(&mixed, &order, &cfg).unwrap();
    assert_eq!(scores.len(), 8);
    assert!(
        scores
            .iter()
            .all(|s| s.is_finite() && *s > evolution::FAILED)
    );
    let combined = gpu.evaluate_with_metrics(&mixed, &order, &cfg).unwrap();
    let mut separated = vec![evolution_simulator::qd::EvaluationMetrics::default(); order.len()];
    for (slot, &creature) in order.iter().enumerate() {
        let metrics = gpu
            .evaluate_with_metrics(&mixed, &[creature], &cfg)
            .unwrap();
        separated[slot] = metrics[0];
    }
    for (combined, separated) in combined.iter().zip(&separated) {
        assert!(
            (combined.fitness - separated.fitness).abs() < 1e-4,
            "combined fitness {} vs separate {}",
            combined.fitness,
            separated.fitness
        );
        assert!(
            (combined.behavior.ground_contact - separated.behavior.ground_contact).abs() < 1e-5,
            "combined contact {} vs separate {}",
            combined.behavior.ground_contact,
            separated.behavior.ground_contact
        );
        assert!(
            (combined.behavior.vertical_oscillation - separated.behavior.vertical_oscillation)
                .abs()
                < 1e-4,
            "combined vertical {} vs separate {}",
            combined.behavior.vertical_oscillation,
            separated.behavior.vertical_oscillation
        );
        assert!(
            (combined.behavior.gait_frequency - separated.behavior.gait_frequency).abs() < 1e-4,
            "combined gait {} vs separate {}",
            combined.behavior.gait_frequency,
            separated.behavior.gait_frequency
        );
    }
}

#[test]
fn nodes_stay_on_top_of_rough_ground() {
    let cfg = Config {
        population: 16,
        duration: 3.0,
        terrain: 3,
        ..config()
    };
    let amplitude = physics::terrain_amplitude(cfg.terrain);
    let pop = evolution::create(&cfg).unwrap();
    for i in 0..pop.genomes.len() {
        let creature = pop.creature(i);
        let frames = evolution_simulator::cpu_engine::trajectory(&creature, &cfg);
        for frame in &frames[physics::settle() as usize + 1..] {
            for (node, gene) in frame.iter().zip(&creature.nodes) {
                let (height, slope) = physics::terrain(node[0], amplitude);
                let floor = height + gene.diameter * 0.5 * (1.0 + slope * slope).sqrt();
                assert!(
                    node[1] >= floor - 0.01,
                    "node sank to {} below {floor}",
                    node[1]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan GPU; run explicitly on the workstation"]
fn gpu_matches_cpu_on_rough_ground() {
    let base = Config {
        population: 64,
        // Contacts with small bumps amplify rounding differences quickly, so
        // compare a short trial.
        duration: 0.2,
        ..config()
    };
    let pop = evolution::create(&base).unwrap();
    let mut gpu = Gpu::new("RTX 4060").unwrap();
    for terrain in 0..physics::TERRAIN_AMPLITUDES.len() as u8 {
        let cfg = Config {
            terrain,
            ..base.clone()
        };
        let scores = gpu
            .evaluate(&pop, &(0..64).collect::<Vec<_>>(), &cfg)
            .unwrap();
        let cpu = cpu_reference(&pop, &cfg);
        for (i, (&gpu_score, &cpu_score)) in scores.iter().zip(&cpu).enumerate() {
            assert!(
                (gpu_score - cpu_score).abs() < 0.05,
                "roughness {terrain}, score {i}: GPU {gpu_score}, CPU engine {cpu_score}",
            );
        }
    }
}

/// Angle at `pivot` from the reference end to the child end.
fn joint_angle(p: &[[f32; 2]], pivot: usize, reference: usize, child: usize) -> f32 {
    let u = [p[reference][0] - p[pivot][0], p[reference][1] - p[pivot][1]];
    let v = [p[child][0] - p[pivot][0], p[child][1] - p[pivot][1]];
    (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1])
}

#[test]
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
        for bone in &mut creature.bones {
            bone.min_angle = -0.3;
            bone.max_angle = 0.3;
        }
        let joints = physics::joints(&creature.nodes, &creature.bones);
        let start: Vec<[f32; 2]> = creature.nodes.iter().map(|n| [n.x, n.y]).collect();
        let frames = evolution_simulator::cpu_engine::trajectory(&creature, &cfg);
        // The trial ends when the head falls or a joint breaks; a limp body
        // may fold any way afterwards.
        let base = creature.bones[0].b as usize;
        let end = frames
            .iter()
            .enumerate()
            .skip(physics::settle() as usize + 1)
            .find(|(_, frame)| {
                frame[0][1] < frame[base][1]
                    || physics::broken_joint(frame, &creature.bones, &joints)
            })
            .map_or(frames.len(), |(tick, _)| tick + 1);
        for (bone, joint) in creature.bones.iter().zip(&joints) {
            let Some(reference) = joint.reference else {
                continue;
            };
            let (pivot, child) = (bone.a as usize, bone.b as usize);
            let rest = joint_angle(&start, pivot, reference, child);
            for frame in &frames[..end] {
                let offset = (joint_angle(frame, pivot, reference, child) - rest
                    + std::f32::consts::PI)
                    .rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI;
                worst = worst.max(offset.abs() - 0.3);
            }
        }
    }
    // Later length and ground passes can push a joint past its limit for a
    // few steps; beyond physics::JOINT_BREAK the joint breaks and the trial
    // ends, far from the half turn a wheel would need.
    assert!(worst < 0.75, "a joint left its range by {worst} rad");
}

#[test]
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
        let frames = evolution_simulator::cpu_engine::trajectory(&creature, &cfg);
        for (bone, joint) in creature.bones.iter().zip(&joints) {
            let Some(reference) = joint.reference else {
                continue;
            };
            let (pivot, child) = (bone.a as usize, bone.b as usize);
            let mut previous = joint_angle(&frames[0], pivot, reference, child);
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
#[ignore = "requires a Vulkan GPU; run explicitly on the workstation"]
fn gpu_matches_cpu_with_narrow_joints() {
    let cfg = Config {
        population: 64,
        duration: 0.2,
        ..config()
    };
    let mut pop = evolution::create(&cfg).unwrap();
    for bone in &mut pop.bones {
        bone.min_angle = -0.2;
        bone.max_angle = 0.2;
    }
    let mut gpu = Gpu::new("RTX 4060").unwrap();
    let scores = gpu
        .evaluate(&pop, &(0..64).collect::<Vec<_>>(), &cfg)
        .unwrap();
    let cpu = cpu_reference(&pop, &cfg);
    for (i, (&gpu_score, &cpu_score)) in scores.iter().zip(&cpu).enumerate() {
        assert!(
            (gpu_score - cpu_score).abs() < 0.05,
            "score {i}: GPU {gpu_score}, CPU engine {cpu_score}",
        );
    }
}

/// Scores from the CPU-only scheduler, which runs the same trials as the GPU
/// scheduler: every creature gets the perturbed fine-physics contender check.
fn cpu_reference(pop: &evolution::Population, cfg: &Config) -> Vec<f32> {
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    evolution_simulator::scheduler::Scheduler::cpu_only(4)
        .unwrap()
        .evaluate(pop, &indices, cfg)
        .unwrap()
        .into_iter()
        .map(|m| m.fitness)
        .collect()
}
