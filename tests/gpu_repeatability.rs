//! GPU repeatability on a fixed population and fixed settings.
//!
//! This ignored test needs a Vulkan GPU and does not compare against the CPU.
//! Cross-engine results are not an acceptance gate: in a GPU run the GPU owns
//! the score. Run this on the workstation with:
//!
//!     cargo test --release --test gpu_repeatability -- --ignored
use evolution_simulator::{config::Config, evolution, gpu::Gpu};

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

/// Physics v2 repeats bit for bit on one GPU, on each backend: Vulkan and
/// CUDA each score the same population twice.
#[test]
#[ignore = "requires a GPU; run explicitly on the workstation"]
fn each_backend_repeats_v2_scores() {
    let _one_gpu_test = ONE_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
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

/// The muscle energy, muscle force and contact forces a v2 recording carries
/// are the kernel's own values: they agree with the CPU prototype's replay of
/// the same creature, and recording does not change the score.
#[test]
#[ignore = "requires a GPU; run explicitly on the workstation"]
fn recorded_forces_match_the_prototype_on_each_backend() {
    use evolution_simulator::{engine, physics2};
    let cfg = Config {
        population: 400,
        random_seed: false,
        duration: 1.5,
        screen: None,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).expect("population");
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    for setting in ["0", "1"] {
        // SAFETY: the variable is read when a GPU opens; the backends run one
        // after another and no other thread reads it.
        unsafe { std::env::set_var("EVOLUTION_CUDA", setting) };
        let mut gpu = Gpu::new("RTX 4060").expect("GPU scheduler");
        assert!(
            gpu.startup_warning.is_none(),
            "the primary GPU did not open"
        );
        let name = gpu.names();
        let scheduler = gpu.sched.as_mut().expect("scheduler");
        let scores = scheduler
            .evaluate_single(&pop, &indices, &cfg)
            .expect("GPU trials");
        let (mut entries, mut close_energy, mut close_force, mut ground_entries) =
            (0usize, 0usize, 0usize, 0usize);
        let mut ground_close = 0usize;
        let mut live = 0usize;
        let mut checked = 0;
        for (i, score) in scores.iter().enumerate() {
            let mut creature = pop.creature(i);
            evolution_simulator::evolution::canonicalize_bone_order(&mut creature);
            if creature.muscles.is_empty() || score.fitness <= -1e10 {
                continue;
            }
            let recording =
                engine::record_on_gpu(&creature, &cfg, std::time::Duration::from_secs(30))
                    .expect("a GPU replay");
            assert_eq!(recording.result.fitness.to_bits(), score.fitness.to_bits());
            let forces = recording.forces.expect("recorded forces");
            let (frames, _, cpu) = physics2::replay_forces(&creature, &cfg);
            assert_eq!(forces.energy.len(), frames.len());
            assert_eq!(forces.muscle.len(), frames.len());
            assert_eq!(forces.ground.len(), frames.len());
            // Only the first second and a bit after settling: contact
            // sequences drift apart later.
            let start = evolution_simulator::physics::settle() as usize;
            for t in start..(start + 80).min(frames.len()) {
                for k in 0..creature.muscles.len() {
                    entries += 1;
                    close_energy +=
                        usize::from((forces.energy[t][k] - cpu.energy[t][k]).abs() < 0.01);
                    close_force +=
                        usize::from((forces.muscle[t][k] - cpu.muscle[t][k]).abs() < 0.5);
                }
                for n in 0..creature.nodes.len() {
                    ground_entries += 1;
                    live += usize::from(forces.ground[t][n] > 0.0);
                    ground_close +=
                        usize::from((forces.ground[t][n] - cpu.ground[t][n]).abs() < 0.5);
                }
            }
            checked += 1;
            if checked >= 40 {
                break;
            }
        }
        eprintln!(
            "{name}: {checked} creatures, energy {close_energy}/{entries}, force {close_force}/{entries}, ground {ground_close}/{ground_entries} close to the prototype ({live} recorded contact forces above zero)"
        );
        assert!(checked >= 10, "too few creatures with muscles");
        assert!(live > 0, "{name}: no contact force was recorded");
        assert!(close_energy * 100 >= entries * 99, "{name}: energy");
        assert!(close_force * 100 >= entries * 98, "{name}: muscle force");
        assert!(
            ground_close * 100 >= ground_entries * 98,
            "{name}: ground force"
        );
    }
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
