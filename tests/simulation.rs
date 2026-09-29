use evolution_simulator::physics::Fidelity;
use evolution_simulator::{
    config::Config,
    evolution::{self, Bone, Creature, Muscle, NodeGene},
    gpu::Gpu,
    physics,
    qd::{Elite, Emitter, QdArchive},
    storage::{self, Experiment, Stage},
};
use std::path::PathBuf;
/// One creature's score on the production CPU engine.
fn evaluate_one(creature: &Creature, cfg: &Config) -> f32 {
    let mut pop = evolution_simulator::evolution::Population::default();
    pop.push(creature.clone());
    evolution_simulator::cpu_engine::evaluate(&pop, cfg)[0].fitness
}

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
            reserve: false,
        })
        .collect();
    let slots: Vec<usize> = (0..8).collect();
    let children = evolution::emit_offspring(
        &[archive],
        &Default::default(),
        &[],
        &plans,
        &slots,
        &cfg,
        0,
        0,
    );
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
    let fixed_score = evaluate_one(&fixed, &cfg);
    let unpowered_score = evaluate_one(&unpowered, &cfg);
    assert!((fixed_score - unpowered_score).abs() < 1e-5);
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
    assert!(evaluate_one(&c, &config()).is_finite());
}
#[test]
fn a_checkpoint_mid_generation_resumes_from_its_archives() {
    // Saves keep the archives and search state, not the generation in
    // progress: a game saved before its first archive starts again from the
    // same random population.
    let mut e = Experiment::new(config()).unwrap();
    e.stage = Stage::Evaluating;
    for i in 0..7 {
        e.scores[i] = evaluate_one(&e.population.creature(i), &e.config);
    }
    e.evaluated = 7;
    let checkpoint = path("partial");
    storage::save(&checkpoint, &e).unwrap();
    let loaded = storage::load(&checkpoint).unwrap();
    assert_eq!(loaded.evaluated, 0);
    assert_eq!(loaded.stage, Stage::Ready);
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
#[ignore = "optional CPU/GPU diagnostic; not a physics acceptance gate"]
fn gpu_cpu_diagnostic_handles_partial_workgroups() {
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
    // Optional cross-engine diagnostics at both fidelities. Single trials
    // leave out the perturbed contender check; contacts may diverge over time.
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
    // Cross-engine energy-effect diagnostic; the GPU result remains the score
    // authority in a GPU run.
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
    // Cross-engine slope and wind diagnostic. Sloped contacts can amplify
    // engine differences, so this uses a short trial.
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
    // Cross-engine mud and gap diagnostic with a short trial.
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
    // Cross-engine hurdle and quake diagnostic, including the packed id seed.
    let shaking_steps = Config {
        duration: 0.1,
        hurdles: 0.2,
        quake: 0.25,
        ..cfg.clone()
    };
    let gpu_metrics = gpu
        .evaluate_with_metrics(&pop, &indices, &shaking_steps)
        .unwrap();
    let cpu = evolution_simulator::cpu_engine::evaluate(&pop, &shaking_steps);
    for (i, (gpu_result, cpu_result)) in gpu_metrics.iter().zip(&cpu).enumerate() {
        assert!(
            (gpu_result.fitness - cpu_result.fitness).abs() < 0.05,
            "hurdles and quake score {i}: GPU {}, CPU engine {}",
            gpu_result.fitness,
            cpu_result.fitness
        );
    }
    // The quake alone must not depend on the batch: a creature evaluated in a
    // full group meets the same ground as when it runs alone through replay.
    let alone = Config {
        population: 1,
        ..shaking_steps.clone()
    };
    let mut single_pop = evolution::Population::default();
    single_pop.push(pop.creature(0));
    let single = evolution_simulator::cpu_engine::evaluate(&single_pop, &alone)[0].fitness;
    assert_eq!(
        single, cpu[0].fitness,
        "the same creature must meet the same quake ground in any batch"
    );
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
                    // v2's contacts push a sunk node out by a fifth of its
                    // depth per step, so a node may sit a little inside.
                    node[1] >= floor - 0.02,
                    "node sank to {} below {floor}",
                    node[1]
                );
            }
        }
    }
}

#[test]
#[ignore = "optional CPU/GPU diagnostic; not a physics acceptance gate"]
fn gpu_cpu_diagnostic_on_rough_ground() {
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
#[ignore = "optional CPU/GPU diagnostic; not a physics acceptance gate"]
fn gpu_cpu_diagnostic_with_narrow_joints() {
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
fn archive_keeps_the_selected_engine_score_without_cpu_rescoring() {
    let mut e = Experiment::new(config()).unwrap();
    let all: Vec<usize> = (0..e.config.population).collect();
    let metrics = evolution_simulator::scheduler::Scheduler::cpu_only(2)
        .unwrap()
        .evaluate(&e.population, &all, &e.config)
        .unwrap();
    // Treat this evaluation engine's output as authoritative. Archive
    // insertion must not silently replace its score with a CPU replay.
    let mut expected = std::collections::HashMap::new();
    for (i, m) in metrics.iter().enumerate() {
        e.scores[i] = m.fitness + 100.0;
        e.trial_metrics[i] = m.behavior;
        expected.insert(e.population.genomes[i].id, e.scores[i]);
    }
    e.evaluated = e.config.population;
    e.rank();
    e.archive_batch().unwrap();
    assert!(!e.archive.entries.is_empty());
    for elite in &e.archive.entries {
        assert_eq!(elite.fitness, expected[&elite.creature.id]);
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

/// Mean height of the first creature's nodes at the end of its replay.
fn mean_height(pop: &evolution::Population, cfg: &Config) -> f32 {
    let frames = evolution_simulator::cpu_engine::trajectory(&pop.creature(0), cfg);
    let last = frames.last().expect("frames");
    last.iter().map(|p| p[1]).sum::<f32>() / last.len() as f32
}

/// The best creature of a physics v2 evolution (seed 40, 100,000 creatures,
/// 30 generations, 20 s trials): a 7-node hopper with 17 muscles that goes
/// 36 m in 20 s. The tests below run it in worlds that differ by one effect.
fn energy_dependent_walker() -> Creature {
    serde_json::from_str(include_str!("fixtures/v2_hopper.json")).expect("hopper fixture")
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
        heat < calm * 0.75,
        "heat wave must cost distance: {heat} m vs {calm} m"
    );
    assert!(
        drought < calm * 0.75,
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
        uphill < calm * 0.9,
        "uphill must cost distance: {uphill} m vs {calm} m"
    );
    assert!(
        headwind < calm * 0.9,
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
        deep < dry * 0.9,
        "deep mud must cost distance: {deep} m vs {dry} m"
    );
    assert!(
        // A hopper's contact sequence is chaotic, so shallow mud may land a
        // little either side of dry.
        damp <= dry + 0.5 && muddy <= dry + 0.5 && deep <= dry + 0.5,
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
        duration: 12.0,
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
        chasms < solid * 0.6,
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

#[test]
fn hurdles_reduce_distance_and_spare_a_groundless_run() {
    // The same walker on clear ground and over raised steps. Every step forces
    // a climb or a leap, so taller hurdles must cost real distance.
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
    let clear = mean_distance(&pop, &base);
    let low = mean_distance(
        &pop,
        &Config {
            hurdles: 0.08,
            ..base.clone()
        },
    );
    let high = mean_distance(
        &pop,
        &Config {
            hurdles: 0.20,
            ..base.clone()
        },
    );
    let walls = mean_distance(
        &pop,
        &Config {
            hurdles: 0.35,
            ..base.clone()
        },
    );
    eprintln!("mean distance: clear {clear} m, low {low} m, high {high} m, walls {walls} m");
    assert!(
        low < clear * 0.9,
        "low hurdles must cost distance: {low} m vs {clear} m"
    );
    assert!(
        high < clear * 0.9,
        "high hurdles must cost distance: {high} m vs {clear} m"
    );
    assert!(
        walls < clear * 0.9,
        "walls must cost distance: {walls} m vs {clear} m"
    );
    // With the ground off the steps cannot act at all.
    let free = mean_distance(
        &pop,
        &Config {
            ground: false,
            ..base.clone()
        },
    );
    let free_walls = mean_distance(
        &pop,
        &Config {
            ground: false,
            hurdles: 0.35,
            ..base.clone()
        },
    );
    assert_eq!(
        free, free_walls,
        "hurdles must not act while the ground is off"
    );
}

#[test]
fn earthquake_gives_each_creature_its_own_repeatable_ground() {
    // The same walker under still ground and in the strongest quake. Every
    // creature meets its own bump phase and height from its id, so distances
    // change, and the same id always meets the same ground twice.
    let base = Config {
        population: 16,
        duration: 5.0,
        ..config()
    };
    let quake = Config {
        quake: 0.25,
        ..base.clone()
    };
    let mut pop = evolution::Population::default();
    for i in 0..16u64 {
        let mut walker = energy_dependent_walker();
        walker.id = i;
        pop.push(walker);
    }
    let calm = evolution_simulator::cpu_engine::evaluate(&pop, &base);
    let first = evolution_simulator::cpu_engine::evaluate(&pop, &quake);
    let second = evolution_simulator::cpu_engine::evaluate(&pop, &quake);
    let changed = calm
        .iter()
        .zip(&first)
        .filter(|(a, b)| a.fitness != b.fitness)
        .count();
    eprintln!(
        "quake: {changed}/16 distances changed; calm mean {:.3} m, quake mean {:.3} m",
        calm.iter().map(|r| r.fitness).sum::<f32>() / 16.0,
        first.iter().map(|r| r.fitness).sum::<f32>() / 16.0
    );
    assert!(
        changed >= 8,
        "the quake must move most walkers: {changed}/16 changed"
    );
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(
            a.fitness, b.fitness,
            "the same creature id must meet the same ground twice"
        );
    }
    // The replay of one creature must score exactly what its batch trial
    // scored, so the recorded ground and the scored ground agree.
    let creature = pop.creature(0);
    let (_, replay) = evolution_simulator::cpu_engine::replay(&creature, &quake);
    assert_eq!(replay.fitness, first[0].fitness);
    // Two different ids must not be forced onto one pattern.
    let scores: Vec<f32> = [7u64, 8]
        .into_iter()
        .map(|id| {
            let mut one = evolution::Population::default();
            let mut walker = energy_dependent_walker();
            walker.id = id;
            one.push(walker);
            evolution_simulator::cpu_engine::evaluate(&one, &quake)[0].fitness
        })
        .collect();
    assert_ne!(
        scores[0], scores[1],
        "different creature ids must meet different ground"
    );
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

#[test]
fn cost_of_transport_counts_muscle_work_per_kilogram_and_meter() {
    // The seed 40 champion of a GPU evolution under physics v2 (43.9 m in
    // 20 s, 4.1 kg, muscle drain about 2,400 J).
    let cfg = Config {
        duration: 20.0,
        ..config()
    };
    let hopper: Creature =
        serde_json::from_str(include_str!("fixtures/p2_muscle_champion.json")).unwrap();
    let cost = evolution_simulator::cpu_engine::transport_cost(&hopper, &cfg)
        .expect("the champion moves forward");
    assert!(cost.is_finite() && cost > 5.0 && cost < 40.0, "cost {cost}");
    // With no muscle drive only the dampers work, so the cost is far lower.
    let mut still = hopper.clone();
    for muscle in &mut still.muscles {
        muscle.short = muscle.long;
    }
    let idle = evolution_simulator::cpu_engine::transport_cost(&still, &cfg);
    assert!(
        idle.is_none_or(|c| c < cost * 0.5),
        "{idle:?} against {cost}"
    );
}

#[test]
fn water_slows_the_walker_and_lifts_a_falling_body() {
    // The same walker on dry ground and in rising water. Water resists every
    // stroke and floats the body, so the walker must lose real distance, and
    // more of it as the water deepens. Water needs no ground: with the ground
    // off a submerged body still feels it.
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
    let shallow = mean_distance(
        &pop,
        &Config {
            water: 0.35,
            ..base.clone()
        },
    );
    let deep = mean_distance(
        &pop,
        &Config {
            water: 0.9,
            ..base.clone()
        },
    );
    eprintln!("mean distance: dry {dry} m, shallows {shallow} m, deep {deep} m");
    assert!(
        shallow < dry * 0.9,
        "shallows must cost distance: {shallow} m vs {dry} m"
    );
    assert!(
        deep < shallow,
        "deep water must cost more than shallows: {deep} m vs {shallow} m"
    );
    let floating = Config {
        ground: false,
        gravity: 9.8,
        ..base.clone()
    };
    let sunk = Config {
        water: 0.9,
        ..floating.clone()
    };
    let air = mean_height(&pop, &floating);
    let wet = mean_height(&pop, &sunk);
    eprintln!("mean height after 5 s without ground: air {air} m, water {wet} m");
    assert!(
        wet > air + 0.5,
        "buoyancy must slow a falling body: {wet} m vs {air} m"
    );
}

#[test]
fn ice_patches_cost_the_walker_distance() {
    // Bands of ice take the friction a walker's feet push with. The walker
    // starts on dry ground and meets a band within a few meters of running.
    let base = Config {
        population: 16,
        duration: 8.0,
        ..config()
    };
    let mut pop = evolution::Population::default();
    for i in 0..16 {
        let mut walker = energy_dependent_walker();
        walker.id = i as u64;
        pop.push(walker);
    }
    let dry = mean_distance(&pop, &base);
    let frost = mean_distance(
        &pop,
        &Config {
            patches: 0.5,
            ..base.clone()
        },
    );
    let black = mean_distance(
        &pop,
        &Config {
            patches: 0.95,
            ..base.clone()
        },
    );
    eprintln!("mean distance: dry {dry} m, frost {frost} m, black ice {black} m");
    assert!(
        black < dry * 0.9,
        "black ice must cost distance: {black} m vs {dry} m"
    );
    assert!(
        black <= frost,
        "black ice must cost at least what frost does"
    );
    // A groundless run meets no ice.
    let floating = Config {
        ground: false,
        ..base.clone()
    };
    let iced = Config {
        patches: 0.95,
        ..floating.clone()
    };
    assert_eq!(
        mean_distance(&pop, &floating),
        mean_distance(&pop, &iced),
        "ice patches must change nothing without ground"
    );
}
