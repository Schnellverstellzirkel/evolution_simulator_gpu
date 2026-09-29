//! GPU repeatability on a fixed population and fixed settings.
//!
//! This ignored test needs a Vulkan GPU and does not compare against the CPU.
//! Cross-engine results are not an acceptance gate: in a GPU run the GPU owns
//! the score. Run this on the workstation with:
//!
//!     cargo test --release --test gpu_repeatability -- --ignored
use evolution_simulator::{config::Config, evolution, gpu::Gpu};

#[test]
#[ignore = "requires a Vulkan GPU; run explicitly on the workstation"]
fn gpu_repeats_scores_for_identical_trials() {
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

/// Physics v2 repeats bit for bit on one GPU, on each backend: Vulkan and
/// CUDA each score the same population twice.
#[test]
#[ignore = "requires a GPU; run explicitly on the workstation"]
fn each_backend_repeats_v2_scores() {
    let cfg = Config {
        population: 256,
        random_seed: false,
        duration: 3.0,
        screen: None,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).expect("population");
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    for setting in ["0", "1"] {
        // SAFETY: the variable is read when a GPU opens; this test runs its
        // backends one after another and no other thread reads it.
        unsafe { std::env::set_var("EVOLUTION_CUDA", setting) };
        let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
        assert!(
            gpu.startup_warning.is_none(),
            "the primary GPU did not open"
        );
        eprintln!("GPU engine: {}", gpu.names());
        if setting == "1" {
            assert!(gpu.names().contains("CUDA"), "opened {}", gpu.names());
        }
        let scheduler = gpu.sched.as_mut().expect("scheduler");
        let first = scheduler
            .evaluate_single(&pop, &indices, &cfg)
            .expect("first");
        let second = scheduler
            .evaluate_single(&pop, &indices, &cfg)
            .expect("second");
        for (i, (a, b)) in first.iter().zip(&second).enumerate() {
            assert_eq!(a.fitness.to_bits(), b.fitness.to_bits(), "creature {i}");
            assert_eq!(
                (
                    a.behavior.ground_contact.to_bits(),
                    a.behavior.vertical_oscillation.to_bits(),
                    a.behavior.gait_frequency.to_bits(),
                    a.behavior.mean_height.to_bits(),
                    a.behavior.feet.to_bits()
                ),
                (
                    b.behavior.ground_contact.to_bits(),
                    b.behavior.vertical_oscillation.to_bits(),
                    b.behavior.gait_frequency.to_bits(),
                    b.behavior.mean_height.to_bits(),
                    b.behavior.feet.to_bits()
                ),
                "creature {i}"
            );
        }
    }
}
