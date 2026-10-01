//! GPU repeatability on a fixed population and fixed settings.
//!
//! These ignored tests need the NVIDIA GPU and its CUDA engine, which owns
//! the score. Run them on the workstation with:
//!
//!     cargo test --release --test gpu_repeatability -- --ignored
use evolution_simulator::{config::Config, evolution, gpu::Gpu};

/// The GPU tests run one at a time: the replay GPU is published in a
/// process-wide slot, so a second test's GPU would take it over.
static ONE_GPU_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore = "requires the GPU; run explicitly on the workstation"]
fn gpu_repeats_scores_for_identical_trials() {
    let _one_gpu_test = ONE_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = Config {
        population: 32,
        random_seed: false,
        duration: 0.5,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).expect("population");
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
    eprintln!("GPU engine: {}", gpu.names());
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    let first = scheduler
        .evaluate(&pop, &indices, &cfg)
        .expect("first GPU trial");
    let second = scheduler
        .evaluate(&pop, &indices, &cfg)
        .expect("repeat GPU trial");

    assert_eq!(first.len(), second.len());
    for (i, (a, b)) in first.iter().zip(&second).enumerate() {
        assert_eq!(a.fitness.to_bits(), b.fitness.to_bits(), "creature {i}");
        assert_eq!(
            [
                a.behavior.ground_contact.to_bits(),
                a.behavior.vertical_oscillation.to_bits(),
                a.behavior.gait_frequency.to_bits(),
                a.behavior.mean_height.to_bits(),
                a.behavior.feet.to_bits(),
                a.screen_x.to_bits(),
            ],
            [
                b.behavior.ground_contact.to_bits(),
                b.behavior.vertical_oscillation.to_bits(),
                b.behavior.gait_frequency.to_bits(),
                b.behavior.mean_height.to_bits(),
                b.behavior.feet.to_bits(),
                b.screen_x.to_bits(),
            ],
            "GPU metrics changed for creature {i}"
        );
        assert_eq!(
            (a.screened, a.excluded),
            (b.screened, b.excluded),
            "creature {i}"
        );
    }
}

#[test]
#[ignore = "requires the GPU; run explicitly on the workstation"]
fn gpu_replays_show_the_gpu_score() {
    let _one_gpu_test = ONE_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
    use evolution_simulator::{engine, physics};
    // Full trials: a replay never stops at the screen.
    let cfg = Config {
        population: 96,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).expect("population");
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    let scores = scheduler
        .evaluate(&pop, &indices, &cfg)
        .expect("GPU trials");
    let fidelity = cfg.fidelity();
    let total = (fidelity.settle() + cfg.steps()) as usize;
    let started = std::time::Instant::now();
    let mut falls = 0;
    for (i, score) in scores.iter().enumerate() {
        let creature = pop.creature(i);
        let recording = engine::record_on_gpu(&creature, &cfg, std::time::Duration::from_secs(30))
            .expect("a GPU replay");
        assert_eq!(
            recording.result.fitness.to_bits(),
            score.fitness.to_bits(),
            "creature {i}: replay {} m, score {} m",
            recording.result.fitness,
            score.fitness
        );
        assert_eq!(recording.frames.len(), total + 1, "creature {i}");
        // The center of mass where the trial ended is the score.
        let nodes = physics::nodes(&creature);
        let mass: f32 = nodes.iter().map(|n| n.mass).sum();
        let end = if recording.result.fall_time > 0.0 {
            falls += 1;
            fidelity.settle() as usize
                + (recording.result.fall_time * fidelity.rate as f32).round() as usize
        } else {
            total
        };
        let center: f32 = recording.frames[end]
            .iter()
            .zip(&nodes)
            .map(|(p, n)| p[0] * n.mass)
            .sum::<f32>()
            / mass;
        if score.fitness > -1e10 {
            assert!(
                (center - score.fitness).abs() < 1e-3,
                "creature {i}: frame {end} center {center} m, score {} m",
                score.fitness
            );
        }
    }
    eprintln!(
        "{} replays ({falls} falls) in {:.2} s",
        scores.len(),
        started.elapsed().as_secs_f64()
    );
}

/// The muscle energy, muscle force and contact forces a recording carries
/// are the kernel's own values and in range, the recorded broken joints are
/// the scoring test's, and recording does not change the score. The nodes
/// are renumbered so bone `j` no longer ends at node `j + 1`, as in most
/// evolved bodies.
#[test]
#[ignore = "requires the GPU; run explicitly on the workstation"]
fn recorded_forces_are_in_range_and_keep_the_score() {
    let _one_gpu_test = ONE_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
    use evolution_simulator::engine;
    let cfg = Config {
        population: 400,
        random_seed: false,
        duration: 1.5,
        screen: None,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).expect("population");
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    let scores = scheduler
        .evaluate(&pop, &indices, &cfg)
        .expect("GPU trials");
    let settle = evolution_simulator::physics::settle() as usize;
    let fidelity = cfg.fidelity();
    let (mut checked, mut live) = (0, 0usize);
    for (i, score) in scores.iter().enumerate() {
        let mut creature = pop.creature(i);
        evolution_simulator::evolution::canonicalize_bone_order(&mut creature);
        if creature.muscles.is_empty() || score.fitness <= -1e10 {
            continue;
        }
        let n = creature.nodes.len();
        let relabel = |k: u32| if k == 0 { 0 } else { n as u32 - k };
        let mut nodes = creature.nodes;
        for (k, node) in creature.nodes.iter().enumerate() {
            nodes[relabel(k as u32) as usize] = *node;
        }
        creature.nodes = nodes;
        for bone in &mut creature.bones {
            bone.a = relabel(bone.a);
            bone.b = relabel(bone.b);
        }
        for muscle in &mut creature.muscles {
            muscle.node_a = relabel(muscle.node_a);
            muscle.node_b = relabel(muscle.node_b);
        }
        let recording = engine::record_on_gpu(&creature, &cfg, std::time::Duration::from_secs(30))
            .expect("a GPU replay");
        assert_eq!(recording.result.fitness.to_bits(), score.fitness.to_bits());
        let frames = recording.frames.len();
        let forces = recording.forces.expect("recorded forces");
        assert!(
            forces
                .energy
                .iter()
                .flatten()
                .all(|e| (0.0..=1.0).contains(e))
                && forces.muscle.iter().flatten().all(|f| f.is_finite())
                && forces
                    .ground
                    .iter()
                    .flatten()
                    .all(|f| f.is_finite() && *f >= 0.0),
            "creature {i} recorded an energy outside [0, 1] or a force that is not finite"
        );
        assert_eq!(forces.energy.len(), frames);
        assert_eq!(forces.muscle.len(), frames);
        assert_eq!(forces.ground.len(), frames);
        assert_eq!(forces.broken.len(), frames);
        // No broken joint before the trial ended, since a break ends it.
        let terminal = if recording.result.fall_time > 0.0 {
            settle + (recording.result.fall_time * fidelity.rate as f32).round() as usize
        } else {
            frames - 1
        };
        assert!(
            forces.broken[..terminal].iter().all(|&b| b == 0),
            "creature {i} shows a broken joint before its trial ended"
        );
        live += forces.ground.iter().flatten().filter(|&&f| f > 0.0).count();
        checked += 1;
        if checked >= 40 {
            break;
        }
    }
    eprintln!("{checked} creatures, {live} recorded contact forces above zero");
    assert!(checked >= 10, "too few creatures with muscles");
    assert!(live > 0, "no contact force was recorded");
}

#[test]
#[ignore = "requires a GPU; run alone: --ignored a_lost_gpu (other GPU tests in parallel change its results)"]
fn a_lost_gpu_is_reopened_and_gives_the_same_results() {
    let cfg = Config {
        population: 30_000,
        random_seed: false,
        duration: 2.0,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).expect("population");
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    let undisturbed = scheduler
        .evaluate(&pop, &indices, &cfg)
        .expect("undisturbed run");
    // The GPU is lost while its first units are in flight; the game must reopen it
    // and finish the run on it.
    scheduler.simulate_gpu_loss_after(0);
    let disturbed = scheduler
        .evaluate(&pop, &indices, &cfg)
        .expect("run with a lost GPU");
    let notices = scheduler.take_notices();
    assert!(
        notices.iter().any(|n| n.contains("GPU is back")),
        "the GPU was not reopened: {notices:?}"
    );
    assert!(
        scheduler.names().contains("CUDA"),
        "the run must still have its GPU: {}",
        scheduler.names()
    );
    for (i, (a, b)) in undisturbed.iter().zip(&disturbed).enumerate() {
        assert_eq!(a.fitness.to_bits(), b.fitness.to_bits(), "creature {i}");
        assert_eq!(
            a.behavior.ground_contact.to_bits(),
            b.behavior.ground_contact.to_bits(),
            "creature {i}"
        );
    }
}

/// A chain whose joints may not bend, pulled by long-range muscles: it
/// breaks a joint within a few seconds.
/// `variant` changes the muscle rhythm.
fn breaking_chain(variant: usize) -> evolution::Creature {
    use evolution::{Bone, Creature, Muscle, NodeGene};
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
            let (a, b) = (nodes[i], nodes[i + 1]);
            let mut bone = Bone::new(i as u32, (i + 1) as u32, (a.x - b.x).hypot(a.y - b.y));
            bone.min_angle = 0.0;
            bone.max_angle = 0.0;
            bone
        })
        .collect();
    let muscles = (1..bones.len())
        .map(|i| Muscle {
            node_a: i as u32,
            node_b: ((i + node_count / 2) % node_count) as u32,
            strength: 1.0,
            period: 0.2 + ((i + variant) % 3) as f32 * 0.05,
            phase: ((i + variant) % 7) as f32 / 7.0,
            duty: 0.5,
            sensor: evolution::NO_SENSOR,
            reset: 0.0,
        })
        .collect();
    Creature {
        nodes: nodes.into(),
        bones: bones.into(),
        muscles,
        id: variant as u64,
    }
}

/// A recording marks the joints the scoring kernel breaks: the recorded
/// bits appear at the frame where the trial ended and not before.
#[test]
#[ignore = "requires the GPU; run explicitly on the workstation"]
fn recorded_broken_joints_are_the_kernels() {
    let _one_gpu_test = ONE_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
    use evolution_simulator::engine;
    let cfg = Config {
        random_seed: false,
        duration: 3.0,
        screen: None,
        ..Config::default()
    };
    let mut pop = evolution::Population::default();
    for variant in 0..8 {
        pop.push(breaking_chain(variant));
    }
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    let scores = scheduler
        .evaluate(&pop, &indices, &cfg)
        .expect("GPU trials");
    let fidelity = cfg.fidelity();
    let settle = fidelity.settle() as usize;
    let mut breaks = 0usize;
    for (i, score) in scores.iter().enumerate() {
        let creature = pop.creature(i);
        let recording = engine::record_on_gpu(&creature, &cfg, std::time::Duration::from_secs(120))
            .expect("a GPU replay");
        assert_eq!(recording.result.fitness.to_bits(), score.fitness.to_bits());
        let broken = recording.forces.expect("recorded forces").broken;
        let terminal = if recording.result.fall_time > 0.0 {
            settle + (recording.result.fall_time * fidelity.rate as f32).round() as usize
        } else {
            broken.len() - 1
        };
        assert!(
            broken[..terminal].iter().all(|&b| b == 0),
            "creature {i} shows a broken joint before its trial ended"
        );
        breaks += usize::from(broken[terminal] != 0);
    }
    eprintln!(
        "{breaks} of {} trials ended on a recorded broken joint",
        scores.len()
    );
    assert!(breaks * 2 >= scores.len(), "too few breaks");
}
