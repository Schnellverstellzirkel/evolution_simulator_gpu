//! The GPU is the only physics authority. These ignored tests need a GPU and
//! check what its results must satisfy: identical trials repeat, a replay
//! shows the score, a creature scores the same alone as in a batch, and the
//! early screen stops only creatures below the bar. Run them on the
//! workstation with:
//!
//!     EVOLUTION_DEVICES=primary cargo test --release --test gpu_repeatability -- --ignored
use evolution_simulator::{
    config::Config,
    evolution::{self, Bone, Creature, Muscle, NodeGene},
    gpu::Gpu,
    physics::Screen,
};

/// The GPU tests run one at a time: the replay GPU is published in a
/// process-wide slot, so a second test's GPU would take it over.
static ONE_GPU_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[ignore = "requires a Vulkan GPU; run explicitly on the workstation"]
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
    assert!(
        gpu.startup_warning.is_none(),
        "the primary GPU did not open: {}",
        gpu.startup_warning.as_deref().unwrap_or("unknown")
    );
    // The GPU runs CUDA on NVIDIA when it loads, else Vulkan. With
    // EVOLUTION_CUDA=1 the test refuses a Vulkan fallback.
    eprintln!("GPU engine: {}", gpu.names());
    if evolution_simulator::cuda_engine::forced() {
        assert!(
            gpu.names().contains("CUDA"),
            "EVOLUTION_CUDA=1 but the GPU opened as {}",
            gpu.names()
        );
    }
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    let first = scheduler
        .evaluate_single(&pop, &indices, &cfg)
        .expect("first GPU trial");
    let second = scheduler
        .evaluate_single(&pop, &indices, &cfg)
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
            (a.screened, a.unchecked),
            (b.screened, b.unchecked),
            "creature {i}"
        );
    }
}

#[test]
#[ignore = "requires a Vulkan GPU; run explicitly on the workstation"]
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
    assert!(
        gpu.startup_warning.is_none(),
        "the primary GPU did not open"
    );
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    let scores = scheduler
        .evaluate_single(&pop, &indices, &cfg)
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
    assert!(gpu.startup_warning.is_none(), "the GPU did not open");
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    let undisturbed = scheduler
        .evaluate_single(&pop, &indices, &cfg)
        .expect("undisturbed run");
    // The GPU is lost while its first units are in flight; the game must reopen it
    // and finish the run on it.
    scheduler.simulate_gpu_loss_after(0);
    let disturbed = scheduler
        .evaluate_single(&pop, &indices, &cfg)
        .expect("run with a lost GPU");
    let notices = scheduler.take_notices();
    assert!(
        notices.iter().any(|n| n.contains("GPU is back")),
        "the GPU was not reopened: {notices:?}"
    );
    assert!(
        scheduler.devices.iter().any(|d| d.engine.max_nodes() >= 64),
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

/// A creature scores the same alone as inside a mixed batch of bodies with
/// different node counts, in any order.
#[test]
#[ignore = "requires a GPU; run explicitly on the workstation"]
fn a_creature_scores_the_same_alone_as_in_a_batch() {
    let _one_gpu_test = ONE_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = Config {
        population: 8,
        duration: 0.5,
        random_seed: false,
        ..Config::default()
    };
    let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
    assert!(gpu.startup_warning.is_none(), "the GPU did not open");
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

/// The early screen stops creatures below the bar at the screen and keeps
/// their distance there. Creatures above it run the full trial unchanged.
#[test]
#[ignore = "requires a GPU; run explicitly on the workstation"]
fn the_screen_stops_creatures_below_the_bar_and_leaves_survivors_alone() {
    let _one_gpu_test = ONE_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
    const SCREEN_SECONDS: f32 = 2.0;
    let base = Config {
        population: 256,
        duration: 6.0,
        random_seed: false,
        seed: 41,
        screen: None,
        ..Config::default()
    };
    let screened = |bar| Config {
        screen: Some(Screen {
            seconds: SCREEN_SECONDS,
            bar,
        }),
        ..base.clone()
    };
    let pop = evolution::create(&base).expect("population");
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
    assert!(gpu.startup_warning.is_none(), "the GPU did not open");
    let scheduler = gpu.sched.as_mut().expect("scheduler");
    // No bar: every trial runs in full and records its distance at the screen.
    let full = scheduler
        .evaluate_single(&pop, &indices, &screened(f32::NEG_INFINITY))
        .expect("trials without a bar");
    assert!(full.iter().all(|r| !r.screened));
    let mut distances: Vec<f32> = full.iter().map(|r| r.screen_x).collect();
    distances.sort_by(f32::total_cmp);
    let bar = distances[distances.len() / 2];
    let short = scheduler
        .evaluate_single(
            &pop,
            &indices,
            &Config {
                duration: SCREEN_SECONDS,
                ..base.clone()
            },
        )
        .expect("short trials");
    let results = scheduler
        .evaluate_single(&pop, &indices, &screened(bar))
        .expect("trials with a bar");
    let mut stopped = 0;
    for (i, ((r, f), s)) in results.iter().zip(&full).zip(&short).enumerate() {
        assert_eq!(
            r.screen_x.to_bits(),
            f.screen_x.to_bits(),
            "creature {i}: the distance at the screen must not depend on the bar"
        );
        if r.screened {
            stopped += 1;
            assert!(f.screen_x < bar, "creature {i} is above the bar");
            let tolerance = 1e-4 * s.fitness.abs().max(1.0);
            assert!(
                (r.fitness - s.fitness).abs() <= tolerance,
                "creature {i}: screened {} vs a {SCREEN_SECONDS} s trial {}",
                r.fitness,
                s.fitness
            );
        } else {
            assert_eq!(
                r.fitness.to_bits(),
                f.fitness.to_bits(),
                "creature {i}: a survivor's trial must not change"
            );
            assert_eq!(
                r.behavior.ground_contact.to_bits(),
                f.behavior.ground_contact.to_bits()
            );
        }
    }
    assert!(stopped > 0, "the median bar must stop some creatures");
}
