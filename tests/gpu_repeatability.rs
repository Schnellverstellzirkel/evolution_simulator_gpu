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
                a.screen2_x.to_bits(),
            ],
            [
                b.behavior.ground_contact.to_bits(),
                b.behavior.vertical_oscillation.to_bits(),
                b.behavior.gait_frequency.to_bits(),
                b.behavior.mean_height.to_bits(),
                b.behavior.feet.to_bits(),
                b.screen_x.to_bits(),
                b.screen2_x.to_bits(),
            ],
            "GPU metrics changed for creature {i}"
        );
        assert_eq!((a.screened, a.unchecked), (b.screened, b.unchecked), "creature {i}");
    }
}
