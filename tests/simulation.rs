use evolution_simulator::physics::Fidelity;
use evolution_simulator::{
    config::Config,
    evolution::{self, Bone, Creature, Muscle, NodeGene},
    gpu::Gpu,
    physics,
    qd::{Elite, Emitter, QdArchive},
    storage::{self, Experiment},
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
/// Made-up standard results: the score grows with the birth slot.
fn by_slot(
    pop: &evolution::Population,
    _: &Config,
) -> anyhow::Result<Vec<evolution_simulator::qd::EvaluationMetrics>> {
    Ok(pop
        .genomes
        .iter()
        .map(|g| evolution_simulator::qd::EvaluationMetrics {
            fitness: evolution::slot_of_id(g.id) as f32 * 0.1,
            ..Default::default()
        })
        .collect())
}
/// Scores from the CPU-only scheduler, as a ring evaluator.
fn cpu_scheduler() -> impl FnMut(
    &evolution::Population,
    &Config,
) -> anyhow::Result<Vec<evolution_simulator::qd::EvaluationMetrics>> {
    let mut sched = evolution_simulator::scheduler::Scheduler::cpu_only(2).unwrap();
    move |pop, cfg| {
        let all: Vec<usize> = (0..pop.genomes.len()).collect();
        sched.evaluate(pop, &all, cfg)
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

    let mut first = Experiment::new(cfg.clone()).unwrap();
    let mut second = Experiment::new(cfg.clone()).unwrap();
    first.run_generation(&mut by_slot).unwrap();
    second.run_generation(&mut by_slot).unwrap();
    assert_eq!(first.ring_len(), cfg.population);
    first.validate().unwrap();
    for (x, y) in first.blocks.iter().zip(&second.blocks) {
        assert_eq!(x.population.nodes, y.population.nodes);
        assert_eq!(x.population.bones, y.population.bones);
        assert_eq!(x.population.muscles, y.population.muscles);
    }
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
        graduate: false,
        fine: false,
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
    let batches = evolution::emit_offspring_batches(&[archive], &[], &plans, &slots, &cfg, 0, 0);
    let mut children = evolution::Population {
        genomes: vec![Default::default(); 8],
        ..Default::default()
    };
    children.append_batches(&slots, batches);
    for k in 0..8 {
        assert_genomes_close(&children.creature(k), &parent);
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
    for _ in 0..80 {
        e.run_generation(&mut by_slot).unwrap();
        e.validate().unwrap();
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
        tendon: 0.0,
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
            tendon: 0.0,
        }],
        id: 1,
    };
    assert!(evaluate_one(&c, &config()).is_finite());
}
#[test]
fn a_checkpoint_before_the_first_archive_starts_the_same_game() {
    // Saves keep the archives and search state, not the ring: a game saved
    // before its first archive starts again from the same random bodies.
    let e = Experiment::new(config()).unwrap();
    let checkpoint = path("partial");
    storage::save(&checkpoint, &e).unwrap();
    let loaded = storage::load(&checkpoint).unwrap();
    assert_eq!(loaded.evaluated, 0);
    for (x, y) in e.blocks.iter().zip(&loaded.blocks) {
        assert_eq!(x.population.nodes, y.population.nodes);
        assert_eq!(x.population.muscles, y.population.muscles);
    }
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
fn statistics_count_every_archive_elite() {
    let mut e = Experiment::new(config()).unwrap();
    e.run_generation(&mut by_slot).unwrap();
    let s = &e.history[0];
    assert!(s.archive_cells > 0);
    assert_eq!(
        s.histogram.iter().map(|x| x.1 as usize).sum::<usize>(),
        s.archive_cells
    );
    assert_eq!(
        s.species.iter().map(|x| x.2 as usize).sum::<usize>(),
        s.archive_cells
    );
    assert_eq!(s.percentiles.len(), 29);
    assert!(s.best >= s.median && s.median >= s.worst);
}

#[test]
fn history_and_checksums_are_validated_on_load() {
    let mut e = Experiment::new(config()).unwrap();
    e.run_generation(&mut by_slot).unwrap();
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
            .evaluate(&pop, &indices, &cfg)
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
                tendon: 0.0,
            })
            .collect();
        mixed.push(Creature {
            nodes,
            bones,
            muscles,
            id: i as u64,
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
        let (frames, result) = evolution_simulator::cpu_engine::replay(&creature, &cfg);
        // The trial ends when the head falls or a joint breaks; a limp body
        // may fold any way afterwards.
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

/// Scores from the CPU-only scheduler, which runs the same standard trials
/// as the GPU scheduler.
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
fn a_world_change_retests_archive_elites() {
    let mut e = Experiment::new(config()).unwrap();
    let mut evaluate = cpu_scheduler();
    e.run_generation(&mut evaluate).unwrap();
    let elites: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    assert!(!elites.is_empty());
    e.update_config_now(Config {
        terrain: 1,
        ..e.config.clone()
    })
    .unwrap();
    // One lap of the ring breeds every block again, the queued elites first.
    let mut handed_over = Vec::new();
    for _ in 0..e.blocks.len() {
        e.step(&mut evaluate).unwrap();
        let k = (e.cursor + e.blocks.len() - 1) % e.blocks.len();
        handed_over.extend(e.blocks[k].population.genomes.iter().map(|g| g.id));
    }
    for id in elites {
        assert!(
            handed_over.contains(&id),
            "elite {id} was dropped by the world change"
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
    let mut cpu = cpu_scheduler();
    // Treat this evaluation engine's output as authoritative. Archive
    // insertion must not silently replace its score with a CPU replay. The
    // confirmation trials score higher, so the standard score stands.
    let mut expected = std::collections::HashMap::new();
    e.step(&mut |pop, cfg| {
        let mut metrics = cpu(pop, cfg)?;
        let confirm = cfg.fidelity.is_some();
        for (g, m) in pop.genomes.iter().zip(&mut metrics) {
            if confirm {
                m.fitness += 1000.0;
            } else {
                m.fitness += 100.0;
                expected.insert(g.id, m.fitness);
            }
        }
        Ok(metrics)
    })
    .unwrap();
    assert!(!e.archive.entries.is_empty());
    for elite in &e.archive.entries {
        assert_eq!(elite.fitness, expected[&elite.creature.id]);
    }
}

#[test]
fn meteor_strike_can_be_undone() {
    let mut e = Experiment::new(config()).unwrap();
    e.run_generation(&mut cpu_scheduler()).unwrap();
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

