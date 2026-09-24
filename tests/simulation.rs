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
            Bone {
                a: i as u32,
                b: (i + 1) as u32,
                rest_length: (a.x - b.x).hypot(a.y - b.y),
            }
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
    };
    assert!((physics::target(&m, 0.) - 0.3).abs() < 1e-6);
    assert!((physics::target(&m, 0.8) - 0.1).abs() < 1e-6);
    assert!((physics::target(&m, 0.79999) - physics::target(&m, 0.80001)).abs() < 1e-5);
    assert!((physics::target(&m, 0.23) - physics::target(&m, 2.23)).abs() < 1e-6);
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
            Bone {
                a: 0,
                b: 1,
                rest_length: 1.0,
            },
            Bone {
                a: 1,
                b: 2,
                rest_length: 1.0,
            },
            Bone {
                a: 2,
                b: 3,
                rest_length: 1.0,
            },
        ],
        muscles: vec![Muscle {
            bone_a: 0,
            bone_b: 2,
            anchor_a: 0.25,
            anchor_b: 0.75,
            short: 0.1,
            long: 0.1,
            period: 1.0,
            phase: 0.0,
            duty: 0.5,
            stiffness: 40.0,
        }],
        id: 1,
        mutability: 1.0,
    };
    let mut nodes = physics::nodes(&creature);
    physics::step(&mut nodes, &creature.bones, &creature.muscles, &cfg, 0);
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
        bones: vec![
            Bone {
                a: 0,
                b: 1,
                rest_length: 0.03,
            },
            Bone {
                a: 1,
                b: 2,
                rest_length: 0.03,
            },
        ],
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
#[ignore = "requires a Vulkan GPU; CPU/GPU bone and muscle smoke test"]
fn gpu_bones_match_cpu_for_off_center_muscle() {
    let cfg = Config {
        population: 2,
        duration: 0.1,
        gravity: 0.0,
        ground: false,
        air_retention: 1.0,
        ..config()
    };
    let creature = Creature {
        nodes: vec![
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
        ],
        bones: vec![
            Bone {
                a: 0,
                b: 1,
                rest_length: 1.0,
            },
            Bone {
                a: 1,
                b: 2,
                rest_length: 1.0,
            },
            Bone {
                a: 2,
                b: 3,
                rest_length: 1.0,
            },
        ],
        muscles: vec![Muscle {
            bone_a: 0,
            bone_b: 2,
            anchor_a: 0.25,
            anchor_b: 0.75,
            short: 0.1,
            long: 0.1,
            period: 1.0,
            phase: 0.0,
            duty: 0.5,
            stiffness: 40.0,
        }],
        id: 1,
        mutability: 1.0,
    };
    let mut gpu = Gpu::new("RTX 4060").unwrap();
    let actual = gpu.trajectory(&creature, &cfg, 1).unwrap();
    let mut expected = physics::nodes(&creature);
    physics::step(&mut expected, &creature.bones, &creature.muscles, &cfg, 0);
    for (a, b) in actual.iter().zip(&expected) {
        for component in 0..2 {
            assert!((a.pos[component] - b.pos[component]).abs() < 0.002);
            assert!((a.vel[component] - b.vel[component]).abs() < 0.02);
        }
    }
    let mut collapsed = creature.clone();
    for node in &mut collapsed.nodes {
        node.x = 0.0;
        node.y = 0.0;
    }
    collapsed.muscles[0].stiffness = 0.0;
    let actual = gpu.trajectory(&collapsed, &cfg, 1).unwrap();
    let mut expected = physics::nodes(&collapsed);
    physics::step(&mut expected, &collapsed.bones, &collapsed.muscles, &cfg, 0);
    for state in [&actual, &expected] {
        assert!(
            state
                .iter()
                .all(|node| node.vel[0].hypot(node.vel[1]) < 1e-4)
        );
        for bone in &collapsed.bones {
            let a = state[bone.a as usize].pos;
            let b = state[bone.b as usize].pos;
            assert!(((a[0] - b[0]).hypot(a[1] - b[1]) - bone.rest_length).abs() < 1e-4);
        }
    }
    let mut asymmetric_mass = creature.clone();
    for (node, diameter) in asymmetric_mass
        .nodes
        .iter_mut()
        .zip([0.06, 0.08, 0.12, 0.10])
    {
        node.diameter = diameter;
    }
    asymmetric_mass.muscles[0].stiffness = 0.0;
    let mut population = evolution::Population::default();
    population.push(asymmetric_mass.clone());
    let gpu_fitness = gpu.evaluate(&population, &[0], &cfg).unwrap()[0];
    let cpu_fitness = physics::evaluate(&asymmetric_mass, &cfg);
    assert!(
        gpu_fitness.abs() < 0.002,
        "stationary COM moved: {gpu_fitness}"
    );
    assert!((gpu_fitness - cpu_fitness).abs() < 0.002);
    for count in [3, 5, 64] {
        let genes: Vec<_> = (0..count)
            .map(|index| NodeGene {
                x: index as f32 * 0.1,
                y: 0.2,
                diameter: 0.08,
                friction: 0.5,
            })
            .collect();
        let bones: Vec<_> = (0..count - 1)
            .map(|index| Bone {
                a: index as u32,
                b: (index + 1) as u32,
                rest_length: 0.1,
            })
            .collect();
        let creature = Creature {
            nodes: genes,
            muscles: vec![Muscle {
                bone_a: 0,
                bone_b: (bones.len() - 1) as u32,
                anchor_a: 0.25,
                anchor_b: 0.75,
                short: 0.1,
                long: 0.1,
                period: 1.0,
                phase: 0.0,
                duty: 0.5,
                stiffness: 40.0,
            }],
            bones,
            id: count as u64,
            mutability: 1.0,
        };
        let actual = gpu.trajectory(&creature, &cfg, 1).unwrap();
        let mut expected = physics::nodes(&creature);
        physics::step(&mut expected, &creature.bones, &creature.muscles, &cfg, 0);
        for (a, b) in actual.iter().zip(&expected) {
            for component in 0..2 {
                assert!((a.pos[component] - b.pos[component]).abs() < 0.002);
                assert!((a.vel[component] - b.vel[component]).abs() < 0.02);
            }
        }
    }
    let stress = stress_creature();
    let stress_cfg = Config {
        duration: 1.0,
        gravity: 30.0,
        ground: true,
        air_retention: 1.0,
        ..cfg
    };
    for steps in [1, 201, 320] {
        let actual = gpu.trajectory(&stress, &stress_cfg, steps).unwrap();
        for node in &actual {
            assert!(
                node.pos
                    .iter()
                    .chain(node.vel.iter())
                    .all(|v| v.is_finite())
            );
            assert!(node.pos[1] >= node.radius - 1e-5 || steps <= physics::SETTLE);
        }
        for bone in &stress.bones {
            let a = actual[bone.a as usize].pos;
            let b = actual[bone.b as usize].pos;
            assert!(((a[0] - b[0]).hypot(a[1] - b[1]) - bone.rest_length).abs() < 0.0001);
        }
    }
    let population = evolution::create(&cfg).unwrap();
    let scores = gpu.evaluate(&population, &[1, 0], &cfg).unwrap();
    assert_eq!(scores.len(), 2);
    assert!(
        scores
            .iter()
            .all(|score| score.is_finite() && *score > evolution::FAILED)
    );
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
    for i in 0..4 {
        let c = pop.creature(i);
        for steps in [1, 32, 201, 205] {
            let actual = gpu.trajectory(&c, &cfg, steps).unwrap();
            let mut expected = physics::nodes(&c);
            for t in 0..steps {
                physics::step(&mut expected, &c.bones, &c.muscles, &cfg, t);
            }
            for (a, b) in actual.iter().zip(&expected) {
                for k in 0..2 {
                    assert!(
                        (a.pos[k] - b.pos[k]).abs() < 0.002,
                        "positions {a:?} vs {b:?}, step {steps}"
                    );
                    assert!(
                        (a.vel[k] - b.vel[k]).abs() < 0.02,
                        "velocities {a:?} vs {b:?}"
                    );
                }
            }
        }
    }
    let scores = gpu
        .evaluate(&pop, &(0..10).collect::<Vec<_>>(), &cfg)
        .unwrap();
    assert_eq!(scores.len(), 10);
    assert!(
        scores
            .iter()
            .all(|s| s.is_finite() && *s > evolution::FAILED)
    );
    let cfg = Config {
        population: 8,
        max_nodes: 64,
        max_muscles: 256,
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
                Bone {
                    a: j as u32,
                    b: (j + 1) as u32,
                    rest_length: (a.x - b.x).hypot(a.y - b.y).max(0.03),
                }
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
