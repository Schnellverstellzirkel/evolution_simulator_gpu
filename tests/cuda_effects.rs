//! The CUDA kernel (the physics authority) under every environment effect.
//!
//! Needs the RTX 4060 and is ignored by default. Run it with
//!
//!     cargo test --release --test cuda_effects -- --ignored
use evolution_simulator::{
    config::Config,
    engine::{self, Engine},
    evolution::{self, Creature},
};
use std::time::Duration;

/// Every environment effect compiles into the CUDA kernel and changes how far
/// a body gets, and no body's trial fails.
#[test]
#[ignore = "needs the RTX 4060"]
fn the_cuda_kernel_feels_every_effect() {
    // SAFETY: read when the engine opens, before any other thread runs.
    unsafe { std::env::set_var("EVOLUTION_CUDA", "1") };
    let base = Config {
        population: 256,
        random_seed: false,
        screen: None,
        duration: 8.0,
        ..Config::default()
    };
    let mut pop = evolution::create(&base).unwrap();
    let first_hopper = pop.genomes.len();
    // An elite of an evolved population that walks about 2 m/s under this
    // kernel.
    for id in 0..8u64 {
        let mut walker: Creature =
            serde_json::from_str(include_str!("fixtures/warp_walker.json")).unwrap();
        walker.id = id;
        pop.push(walker);
    }
    let mut gpu = engine::gpu_engine("RTX 4060", 32, 64).expect("GPU");
    assert!(gpu.name().contains("CUDA"), "opened {}", gpu.name());
    let mut run = |cfg: &Config| {
        gpu.submit(pop.clone(), cfg).unwrap();
        loop {
            if let Some(done) = gpu.poll().unwrap() {
                break done.results;
            }
            gpu.wait(Duration::from_millis(20));
        }
    };
    let calm = run(&base);
    assert!(calm.iter().all(|r| r.fitness > -1e19), "a calm trial failed");
    let walked = (first_hopper..calm.len())
        .map(|i| calm[i].fitness)
        .fold(f32::NEG_INFINITY, f32::max);
    eprintln!("the walkers get up to {walked:.2} m");
    assert!(walked > 5.0, "the walkers stand still ({walked} m)");
    let effects: Vec<(&str, Config)> = vec![
        ("terrain", Config { terrain: 2, ..base.clone() }),
        ("slope", Config { slope: 0.1, ..base.clone() }),
        ("mud", Config { mud: 0.06, ..base.clone() }),
        ("water", Config { water: 0.35, ..base.clone() }),
        ("gaps", Config { gaps: 0.5, ..base.clone() }),
        ("hurdles", Config { hurdles: 0.1, ..base.clone() }),
        ("ice", Config { patches: 0.95, ..base.clone() }),
        ("quake", Config { quake: 0.05, ..base.clone() }),
        ("wind", Config { wind: 2.0, ..base.clone() }),
        ("air", Config { air_retention: 0.98, ..base.clone() }),
        ("gravity", Config { gravity: 5.0, ..base.clone() }),
        ("grip", Config { ground_friction: 0.5, ..base.clone() }),
        ("heat", Config { muscle_energy: 0.3, ..base.clone() }),
    ];
    for (name, cfg) in effects {
        let results = run(&cfg);
        assert!(
            results.iter().all(|r| r.fitness > -1e19),
            "{name}: a trial failed"
        );
        let change = (0..calm.len())
            .map(|i| (results[i].fitness - calm[i].fitness).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{name}: bodies move up to {change:.3} m differently");
        assert!(change > 0.005, "{name} changes no body's distance");
    }
}
