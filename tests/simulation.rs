use evolution_simulator::physics::Fidelity;
use evolution_simulator::{
    config::Config,
    evolution::{self, Bone, Creature, Muscle, NodeGene},
    gpu::Gpu,
    physics::{self, Node},
    qd::{Elite, Emitter, QdArchive},
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
fn seed_is_repeatable_and_breeding_conserves_population() {
    let cfg = config();
    let a = evolution::create(&cfg).unwrap();
    let b = evolution::create(&cfg).unwrap();
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.bones, b.bones);
    assert_eq!(a.muscles, b.muscles);

    let scores: Vec<_> = (0..cfg.population).map(|i| i as f32).collect();
    let mut first = Experiment::new(cfg.clone()).unwrap();
    let mut second = Experiment::new(cfg.clone()).unwrap();
    first.scores.clone_from(&scores);
    second.scores.clone_from(&scores);
    first.evaluated = cfg.population;
    second.evaluated = cfg.population;
    first.archive_batch().unwrap();
    second.archive_batch().unwrap();
    first.prepare_next_batch().unwrap();
    second.prepare_next_batch().unwrap();
    assert_eq!(first.population.genomes.len(), cfg.population);
    first.population.validate(&cfg).unwrap();
    assert_eq!(first.population.nodes, second.population.nodes);
    assert_eq!(first.population.bones, second.population.bones);
    assert_eq!(first.population.muscles, second.population.muscles);
}
fn assert_genomes_close(a: &Creature, b: &Creature) {
    let close = |x: f32, y: f32| (x - y).abs() <= 1e-4;
    assert_eq!(a.nodes.len(), b.nodes.len());
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert!((x.x - y.x).abs() <= 1e-4 && (x.y - y.y).abs() <= 1e-4);
        assert!(close(x.diameter, y.diameter) && close(x.friction, y.friction));
    }
    assert_eq!(a.bones.len(), b.bones.len());
    for (x, y) in a.bones.iter().zip(&b.bones) {
        assert_eq!((x.a, x.b), (y.a, y.b));
        assert!(close(x.rest_length, y.rest_length));
        assert!(close(x.min_angle, y.min_angle) && close(x.max_angle, y.max_angle));
        assert!(close(x.organ_mass, y.organ_mass) && close(x.organ_at, y.organ_at));
    }
    assert_eq!(a.muscles.len(), b.muscles.len());
    for (x, y) in a.muscles.iter().zip(&b.muscles) {
        assert_eq!(
            (x.bone_a, x.bone_b, x.sensor),
            (y.bone_a, y.bone_b, y.sensor)
        );
        assert!(close(x.anchor_a, y.anchor_a) && close(x.anchor_b, y.anchor_b));
        assert!(close(x.short, y.short) && close(x.long, y.long));
        assert!(close(x.period, y.period) && close(x.phase, y.phase));
        assert!(close(x.duty, y.duty) && close(x.stiffness, y.stiffness));
        assert!(close(x.reset, y.reset));
    }
}
#[test]
fn zero_mutation_copies_genetics() {
    let cfg = Config {
        mutation: 0.,
        ..config()
    };
    let population = evolution::create(&cfg).unwrap();
    let parent = population.creature(0);
    let mut archive = QdArchive::default();
    archive.entries.push(Elite {
        niche: Default::default(),
        descriptor: Default::default(),
        creature: parent.clone(),
        fitness: 1.0,
        emitter: Emitter::Cma,
        improved_generation: 0,
        protected_until: 0,
        visits: 0,
        topology: evolution_simulator::qd::topology_of_population(&population, 0),
    });
    let plans: Vec<_> = (0..8)
        .map(|_| evolution::CandidatePlan {
            emitter: Emitter::Cma,
            parent: Some(0),
            cma: None,
            mate: None,
        })
        .collect();
    let slots: Vec<usize> = (0..8).collect();
    let children = evolution::emit_offspring(&[archive], &[], &plans, &slots, &cfg, 0, 0);
    for child in &children {
        assert_genomes_close(child, &parent);
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
    let mut e = Experiment::new(cfg.clone()).unwrap();
    for generation in 0..80 {
        for (i, score) in e.scores.iter_mut().enumerate() {
            *score = generation as f32 * 0.1 + i as f32;
        }
        e.evaluated = cfg.population;
        e.archive_batch().unwrap();
        e.prepare_next_batch().unwrap();
        e.population.validate(&cfg).unwrap();
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
    e.archive_batch().unwrap();
    loaded.archive_batch().unwrap();
    e.prepare_next_batch().unwrap();
    loaded.prepare_next_batch().unwrap();
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
    // The environment energy multipliers reach the kernel through the uniform
    // buffer; both engines must apply them identically. Short trials keep
    // rounding differences from growing chaotically.
    let effect = Config {
        muscle_energy: 0.35,
        muscle_recovery: 0.1,
        ..cfg.clone()
    };
    let gpu_metrics = gpu.evaluate_with_metrics(&pop, &indices, &effect).unwrap();
    let cpu = evolution_simulator::cpu_engine::evaluate(&pop, &effect);
    for (i, (gpu_result, cpu_result)) in gpu_metrics.iter().zip(&cpu).enumerate() {
        assert!(
            (gpu_result.fitness - cpu_result.fitness).abs() < 0.05,
            "energy effects score {i}: GPU {}, CPU engine {}",
            gpu_result.fitness,
            cpu_result.fitness
        );
    }
    // The slope and wind effects reach the kernel through the same uniform
    // buffer; both engines must apply the terrain tilt and the horizontal
    // wind force identically. Sloped ground contacts amplify rounding
    // differences quickly, so compare a short trial, as the rough-ground
    // test does.
    let weather = Config {
        duration: 0.1,
        slope: 0.15,
        wind: -3.0,
        ..cfg.clone()
    };
    let gpu_metrics = gpu.evaluate_with_metrics(&pop, &indices, &weather).unwrap();
    let cpu = evolution_simulator::cpu_engine::evaluate(&pop, &weather);
    for (i, (gpu_result, cpu_result)) in gpu_metrics.iter().zip(&cpu).enumerate() {
        assert!(
            (gpu_result.fitness - cpu_result.fitness).abs() < 0.05,
            "slope and wind score {i}: GPU {}, CPU engine {}",
            gpu_result.fitness,
            cpu_result.fitness
        );
    }
    // Mud and gaps reach the kernel through the same uniform buffer. Sunk
    // floors and pit walls change contacts quickly, so compare a short trial
    // at both effects at once.
    let muddy_gaps = Config {
        duration: 0.1,
        mud: 0.10,
        gaps: 0.8,
        ..cfg.clone()
    };
    let gpu_metrics = gpu
        .evaluate_with_metrics(&pop, &indices, &muddy_gaps)
        .unwrap();
    let cpu = evolution_simulator::cpu_engine::evaluate(&pop, &muddy_gaps);
    for (i, (gpu_result, cpu_result)) in gpu_metrics.iter().zip(&cpu).enumerate() {
        assert!(
            (gpu_result.fitness - cpu_result.fitness).abs() < 0.05,
            "mud and gaps score {i}: GPU {}, CPU engine {}",
            gpu_result.fitness,
            cpu_result.fitness
        );
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
        slope: 0.15,
        ..config()
    };
    let amplitude = physics::terrain_amplitude(cfg.terrain);
    let pop = evolution::create(&cfg).unwrap();
    for i in 0..pop.genomes.len() {
        let creature = pop.creature(i);
        let frames = evolution_simulator::cpu_engine::trajectory(&creature, &cfg);
        for frame in &frames[physics::settle() as usize + 1..] {
            for (node, gene) in frame.iter().zip(&creature.nodes) {
                let (height, slope) = physics::terrain_with_slope(node[0], amplitude, cfg.slope);
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

#[test]
fn world_change_retests_archive_elites_in_generational_mode() {
    let mut e = Experiment::new(config()).unwrap();
    let all: Vec<usize> = (0..e.config.population).collect();
    let metrics = evolution_simulator::scheduler::Scheduler::cpu_only(2)
        .unwrap()
        .evaluate(&e.population, &all, &e.config)
        .unwrap();
    for (i, m) in metrics.iter().enumerate() {
        e.scores[i] = m.fitness;
        e.trial_metrics[i] = m.behavior;
    }
    e.evaluated = e.config.population;
    e.rank();
    e.archive_batch().unwrap();
    let elites: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    assert!(!elites.is_empty());
    e.update_config(Config {
        terrain: 1,
        ..e.config.clone()
    })
    .unwrap();
    let mut handed_over = Vec::new();
    e.prepare_next_batch_streaming(4, |pop, range, _| {
        handed_over.extend(range.map(|i| pop.creature(i).id));
        Ok(())
    })
    .unwrap();
    for id in elites {
        assert!(
            (0..e.config.population).any(|i| e.population.creature(i).id == id),
            "elite {id} was dropped by the world change"
        );
        assert!(
            handed_over.contains(&id),
            "elite {id} never reached a device"
        );
    }
}

#[test]
fn autosave_rotation_keeps_the_newest_and_spares_manual_saves() {
    let dir = std::env::temp_dir().join(format!("evolution-rotate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for seed in 0..5 {
        std::fs::write(dir.join(format!("seed-{seed}-auto.evo")), b"x").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::fs::write(dir.join("my-champion.evo"), b"x").unwrap();
    std::fs::write(dir.join("seed-9-auto.evo.tmp"), b"x").unwrap();
    assert_eq!(storage::rotate_autosaves(&dir, 3), 2);
    let mut left: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    // The fresh .tmp may belong to a save in progress, so it stays.
    assert_eq!(
        left,
        [
            "my-champion.evo",
            "seed-2-auto.evo",
            "seed-3-auto.evo",
            "seed-4-auto.evo",
            "seed-9-auto.evo.tmp"
        ]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn archive_scores_never_exceed_the_replayed_distance() {
    let mut e = Experiment::new(config()).unwrap();
    let all: Vec<usize> = (0..e.config.population).collect();
    let metrics = evolution_simulator::scheduler::Scheduler::cpu_only(2)
        .unwrap()
        .evaluate(&e.population, &all, &e.config)
        .unwrap();
    // Pretend another engine scored every creature 100 m farther than the
    // replay engine does.
    for (i, m) in metrics.iter().enumerate() {
        e.scores[i] = m.fitness + 100.0;
        e.trial_metrics[i] = m.behavior;
    }
    e.evaluated = e.config.population;
    e.rank();
    e.archive_batch().unwrap();
    assert!(!e.archive.entries.is_empty());
    for elite in &e.archive.entries {
        let replay = evolution_simulator::cpu_engine::evaluate(
            &{
                let mut pop = evolution::Population::default();
                pop.push(elite.creature.clone());
                pop
            },
            &e.config,
        )[0]
        .fitness;
        assert!(
            elite.fitness <= replay + 1e-4,
            "archive shows {} m but the replay reaches {replay} m",
            elite.fitness
        );
    }
}

#[test]
fn meteor_strike_can_be_undone() {
    let mut e = Experiment::new(config()).unwrap();
    let all: Vec<usize> = (0..e.config.population).collect();
    let metrics = evolution_simulator::scheduler::Scheduler::cpu_only(2)
        .unwrap()
        .evaluate(&e.population, &all, &e.config)
        .unwrap();
    for (i, m) in metrics.iter().enumerate() {
        e.scores[i] = m.fitness;
        e.trial_metrics[i] = m.behavior;
    }
    e.evaluated = e.config.population;
    e.rank();
    e.archive_batch().unwrap();
    let count = |e: &Experiment| {
        e.archive.entries.len() + e.islands.iter().map(|i| i.entries.len()).sum::<usize>()
    };
    let before = count(&e);
    let mut ids: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    ids.sort_unstable();
    let lost = e.meteor(0.5);
    assert!(lost > 0);
    assert_eq!(count(&e), before - lost);
    assert_eq!(e.undo_meteor(), lost);
    assert_eq!(count(&e), before);
    assert!(e.fossils.is_empty());
    let mut back: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    back.sort_unstable();
    assert_eq!(back, ids);
}

#[test]
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
    let results = evolution_simulator::cpu_engine::evaluate(&pop, &cfg);
    let worst = results.iter().map(|r| r.fitness.abs()).fold(0.0, f32::max);
    eprintln!("worst drift without drive: {worst} m");
    // A collapsing body can slide a little through real friction, but it
    // must never travel.
    assert!(worst < 0.5, "a body drifted {worst} m with no muscle drive");
}

#[test]
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
    let settle = evolution_simulator::physics::settle() as usize;
    for i in 0..pop.genomes.len() {
        let creature = pop.creature(i);
        let masses: Vec<f32> = evolution_simulator::physics::nodes(&creature)
            .iter()
            .map(|n| n.mass)
            .collect();
        let total: f32 = masses.iter().sum();
        let (frames, _) = evolution_simulator::cpu_engine::replay(&creature, &cfg);
        let height = |frame: &Vec<[f32; 2]>| {
            frame
                .iter()
                .zip(&masses)
                .map(|(p, m)| p[1] * m)
                .sum::<f32>()
                / total
        };
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

/// Mean distance of a whole random population, ignoring failed trials.
fn mean_distance(pop: &evolution::Population, cfg: &Config) -> f32 {
    let results = evolution_simulator::cpu_engine::evaluate(pop, cfg);
    let mut total = 0.0;
    let mut count = 0.0f32;
    for result in &results {
        if result.fitness.is_finite() {
            total += result.fitness;
            count += 1.0;
        }
    }
    total / count.max(1.0)
}

/// A six-node walker evolved under the calm defaults (population 2048,
/// 30 generations, seed 38, 15 s trials). It travels about 8 m in 5 s with
/// full energy stores, and loses more than half of that when the heat wave
/// shrinks the stores or drought slows recovery.
fn energy_dependent_walker() -> Creature {
    Creature {
        nodes: vec![
            NodeGene {
                x: -0.5593486,
                y: 0.61968184,
                diameter: 0.12,
                friction: 0.9475196,
            },
            NodeGene {
                x: -0.41776797,
                y: 0.3038751,
                diameter: 0.06,
                friction: 0.7486749,
            },
            NodeGene {
                x: -0.116856754,
                y: 0.3178858,
                diameter: 0.108820364,
                friction: 0.66710335,
            },
            NodeGene {
                x: -0.039948717,
                y: 0.43371728,
                diameter: 0.0877441,
                friction: 0.9974566,
            },
            NodeGene {
                x: 0.19000307,
                y: 0.37280446,
                diameter: 0.07844837,
                friction: 0.65871984,
            },
            NodeGene {
                x: -0.27397153,
                y: 0.3718743,
                diameter: 0.08361293,
                friction: 0.65,
            },
        ],
        bones: vec![
            Bone {
                a: 0,
                b: 1,
                rest_length: 0.34609097,
                min_angle: -1.6000074,
                max_angle: 0.7977927,
                organ_mass: 0.01,
                organ_at: 0.5291506,
            },
            Bone {
                a: 1,
                b: 2,
                rest_length: 0.30123723,
                min_angle: -0.65684575,
                max_angle: 1.6548456,
                organ_mass: 0.0,
                organ_at: 0.5,
            },
            Bone {
                a: 2,
                b: 3,
                rest_length: 0.13903877,
                min_angle: -1.5915323,
                max_angle: 0.633044,
                organ_mass: 0.0,
                organ_at: 0.5,
            },
            Bone {
                a: 3,
                b: 4,
                rest_length: 0.23788273,
                min_angle: -1.9740598,
                max_angle: 0.9501501,
                organ_mass: 0.010241741,
                organ_at: 0.4697004,
            },
            Bone {
                a: 3,
                b: 5,
                rest_length: 0.24205624,
                min_angle: -0.82507557,
                max_angle: 2.0943952,
                organ_mass: 0.0,
                organ_at: 0.5,
            },
        ],
        muscles: vec![
            Muscle {
                bone_a: 0,
                bone_b: 1,
                anchor_a: 0.9631201,
                anchor_b: 0.6989166,
                short: 0.18162505,
                long: 0.18962646,
                period: 0.5951622,
                phase: 0.42644,
                duty: 0.4687704,
                stiffness: 35.06757,
                sensor: 0,
                reset: 0.43866375,
            },
            Muscle {
                bone_a: 1,
                bone_b: 2,
                anchor_a: 0.5982309,
                anchor_b: 0.66608244,
                short: 0.08557357,
                long: 0.18462573,
                period: 0.5951622,
                phase: 0.016101224,
                duty: 0.38960835,
                stiffness: 84.78909,
                sensor: 1,
                reset: 0.19877157,
            },
            Muscle {
                bone_a: 2,
                bone_b: 3,
                anchor_a: 0.08075203,
                anchor_b: 0.08479669,
                short: 0.21270484,
                long: 0.5398381,
                period: 0.5951622,
                phase: 0.59647053,
                duty: 0.18719086,
                stiffness: 35.218353,
                sensor: 255,
                reset: 0.2708392,
            },
            Muscle {
                bone_a: 3,
                bone_b: 0,
                anchor_a: 0.58470476,
                anchor_b: 0.39771268,
                short: 0.5982563,
                long: 0.9116529,
                period: 0.5951622,
                phase: 0.07857889,
                duty: 0.4873734,
                stiffness: 57.975662,
                sensor: 255,
                reset: 0.6004214,
            },
            Muscle {
                bone_a: 1,
                bone_b: 2,
                anchor_a: 0.2654894,
                anchor_b: 0.3410796,
                short: 0.10381305,
                long: 0.11306952,
                period: 0.5951622,
                phase: 0.5893883,
                duty: 0.2278046,
                stiffness: 73.649185,
                sensor: 255,
                reset: 0.91381097,
            },
            Muscle {
                bone_a: 2,
                bone_b: 4,
                anchor_a: 0.06489495,
                anchor_b: 0.092396125,
                short: 0.20794365,
                long: 0.53504324,
                period: 0.5951622,
                phase: 0.9986213,
                duty: 0.22768927,
                stiffness: 33.61832,
                sensor: 255,
                reset: 0.2898468,
            },
            Muscle {
                bone_a: 4,
                bone_b: 0,
                anchor_a: 0.57180727,
                anchor_b: 0.3885777,
                short: 0.6111155,
                long: 0.9209887,
                period: 0.5951622,
                phase: 0.5952037,
                duty: 0.37932187,
                stiffness: 71.13563,
                sensor: 255,
                reset: 0.60610723,
            },
            Muscle {
                bone_a: 3,
                bone_b: 4,
                anchor_a: 0.07086325,
                anchor_b: 0.26907945,
                short: 0.059728526,
                long: 0.1037927,
                period: 0.5951622,
                phase: 0.13944362,
                duty: 0.62057906,
                stiffness: 31.938673,
                sensor: 255,
                reset: 0.008283809,
            },
        ],
        id: 47_300,
        mutability: 0.87713593,
    }
}

#[test]
fn heat_wave_and_drought_reduce_distance() {
    // The same walker under three worlds; only the muscle energy multipliers
    // differ. A smaller store or slower recovery must cost it real distance.
    let base = Config {
        population: 16,
        duration: 5.0,
        ..config()
    };
    let mut pop = evolution::Population::default();
    for i in 0..16 {
        let mut walker = energy_dependent_walker();
        walker.id = i as u64;
        pop.push(walker);
    }
    let calm = mean_distance(&pop, &base);
    let heat = mean_distance(
        &pop,
        &Config {
            muscle_energy: 0.35,
            ..base.clone()
        },
    );
    let drought = mean_distance(
        &pop,
        &Config {
            muscle_recovery: 0.1,
            ..base.clone()
        },
    );
    eprintln!("mean distance: calm {calm} m, heat wave {heat} m, drought {drought} m");
    assert!(
        heat < calm - 1.0,
        "heat wave must cost distance: {heat} m vs {calm} m"
    );
    assert!(
        drought < calm - 1.0,
        "drought must cost distance: {drought} m vs {calm} m"
    );
}

#[test]
fn uphill_slope_and_headwind_reduce_distance() {
    // The same walker on flat ground, up a 25% hill, and into a gale. Both
    // effects are forces, never scoring terms, and each must cost real
    // distance.
    let base = Config {
        population: 16,
        duration: 5.0,
        ..config()
    };
    let mut pop = evolution::Population::default();
    for i in 0..16 {
        let mut walker = energy_dependent_walker();
        walker.id = i as u64;
        pop.push(walker);
    }
    let calm = mean_distance(&pop, &base);
    let uphill = mean_distance(
        &pop,
        &Config {
            slope: 0.25,
            ..base.clone()
        },
    );
    let headwind = mean_distance(
        &pop,
        &Config {
            wind: -6.0,
            ..base.clone()
        },
    );
    eprintln!("mean distance: calm {calm} m, 25% uphill {uphill} m, gale {headwind} m");
    assert!(
        uphill < calm - 0.5,
        "uphill must cost distance: {uphill} m vs {calm} m"
    );
    assert!(
        headwind < calm - 0.5,
        "a headwind must cost distance: {headwind} m vs {calm} m"
    );
    // With the ground disabled the slope must not act at all, so both
    // free-fall trials are identical.
    let free = mean_distance(
        &pop,
        &Config {
            ground: false,
            ..base.clone()
        },
    );
    let free_hill = mean_distance(
        &pop,
        &Config {
            ground: false,
            slope: 0.25,
            ..base.clone()
        },
    );
    assert_eq!(
        free, free_hill,
        "slope must not act while the ground is off"
    );
}

#[test]
fn mud_reduces_distance_and_spares_a_groundless_run() {
    // The same walker on dry ground and in the deepest mud. Sunk feet drag,
    // so the mud must cost it real distance. A body that never touches the
    // ground pays nothing: with the ground off, mud changes no result.
    let base = Config {
        population: 16,
        duration: 5.0,
        ..config()
    };
    let mut pop = evolution::Population::default();
    for i in 0..16 {
        let mut walker = energy_dependent_walker();
        walker.id = i as u64;
        pop.push(walker);
    }
    let dry = mean_distance(&pop, &base);
    let damp = mean_distance(
        &pop,
        &Config {
            mud: 0.02,
            ..base.clone()
        },
    );
    let muddy = mean_distance(
        &pop,
        &Config {
            mud: 0.05,
            ..base.clone()
        },
    );
    let deep = mean_distance(
        &pop,
        &Config {
            mud: 0.10,
            ..base.clone()
        },
    );
    eprintln!("mean distance: dry {dry} m, damp {damp} m, muddy {muddy} m, deep {deep} m");
    assert!(
        deep < dry - 0.5,
        "deep mud must cost distance: {deep} m vs {dry} m"
    );
    assert!(
        damp <= dry && muddy <= dry && deep <= dry,
        "mud must never help: dry {dry}, damp {damp}, muddy {muddy}, deep {deep}"
    );
    let free = mean_distance(
        &pop,
        &Config {
            ground: false,
            ..base.clone()
        },
    );
    let free_mud = mean_distance(
        &pop,
        &Config {
            ground: false,
            mud: 0.10,
            ..base.clone()
        },
    );
    assert_eq!(free, free_mud, "mud must not act while the ground is off");
}

#[test]
fn gaps_stop_a_walker_where_solid_ground_lets_it_run() {
    // The same walker on solid ground and over chasms. The first pit opens
    // where the walker would otherwise run, so it must fall or stop early.
    let base = Config {
        population: 16,
        duration: 5.0,
        ..config()
    };
    let mut pop = evolution::Population::default();
    for i in 0..16 {
        let mut walker = energy_dependent_walker();
        walker.id = i as u64;
        pop.push(walker);
    }
    let solid = mean_distance(&pop, &base);
    let chasms = mean_distance(
        &pop,
        &Config {
            gaps: 1.5,
            ..base.clone()
        },
    );
    eprintln!("mean distance: solid {solid} m, chasms {chasms} m");
    assert!(
        chasms < solid - 2.0,
        "chasms must stop the walker: {chasms} m vs {solid} m"
    );
    let free = mean_distance(
        &pop,
        &Config {
            ground: false,
            ..base.clone()
        },
    );
    let free_gaps = mean_distance(
        &pop,
        &Config {
            ground: false,
            gaps: 1.5,
            ..base.clone()
        },
    );
    assert_eq!(free, free_gaps, "gaps must not act while the ground is off");
}

/// The four-node sled evolution built while friction could push the body in
/// any direction: a flat chain of nodes resting on the ground, driven by
/// muscles that pull it together along its length. Its momentum came almost
/// entirely from the center-of-mass shift of the bone passes.
fn sled_creature() -> Creature {
    let spacing = 0.7;
    let diameter = 0.16;
    let nodes: Vec<_> = (0..4)
        .map(|i| NodeGene {
            x: i as f32 * spacing,
            y: diameter * 0.5,
            diameter,
            friction: 1.0,
        })
        .collect();
    let bones: Vec<_> = (0..3)
        .map(|i| Bone::new(i as u32, i as u32 + 1, spacing))
        .collect();
    // Each muscle spans two bones, from bone i's start to bone i+1's end, so
    // contracting it drags the chain together while every node stays down.
    let muscles: Vec<_> = (0..2)
        .map(|i| Muscle {
            bone_a: i as u32,
            bone_b: i as u32 + 1,
            anchor_a: 0.0,
            anchor_b: 1.0,
            short: spacing * 1.05,
            long: spacing * 2.0,
            period: 1.0,
            phase: i as f32 * 0.5,
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
fn a_four_node_sled_stays_slow() {
    // This known exploit shape reached 2,267 m in 60 s while the friction cap
    // still let the bone passes push it forward. Friction may now push only
    // while the feet stay planted, so a body that slides cannot propel itself.
    let cfg = Config {
        population: 1,
        duration: 60.0,
        random_seed: false,
        ..config()
    };
    let mut pop = evolution::Population::default();
    pop.push(sled_creature());
    let result = evolution_simulator::cpu_engine::evaluate(&pop, &cfg);
    let distance = result[0].fitness;
    // A fall scores FAILED, which would also pass a pure upper bound.
    assert!(
        distance > evolution::FAILED && distance < 10.0,
        "the sled exploit traveled {distance} m in 60 s"
    );
}

#[test]
fn random_bodies_get_no_free_propulsion() {
    // Solver exploits show up first in random bodies: the uncapped planted
    // feet let a random body travel 224 m in 20 s, against under 10 m with
    // honest friction. None of 4,096 random bodies should reach 20 m in 10 s.
    let cfg = Config {
        population: 4096,
        duration: 10.0,
        ..config()
    };
    let pop = evolution::create(&cfg).unwrap();
    let best = evolution_simulator::cpu_engine::evaluate(&pop, &cfg)
        .iter()
        .map(|r| r.fitness)
        .filter(|f| f.is_finite())
        .fold(f32::MIN, f32::max);
    assert!(best < 20.0, "a random body traveled {best} m in 10 s");
}
