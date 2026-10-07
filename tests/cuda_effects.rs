//! Scores bodies on the CUDA kernel in a calm world and under each
//! environment effect in its list, which lacks drought and brambles. No trial
//! may fail, and each effect must change how far some body gets by more than
//! 5 mm.
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

/// Scores a fixed set of bodies in a calm world, then under each effect. No
/// trial fails, the stored walkers get more than 5 m in the calm world, and
/// each effect changes how far some body gets.
#[test]
#[ignore = "needs the RTX 4060"]
fn the_cuda_kernel_feels_every_effect() {
    let base = Config {
        population: 256,
        random_seed: false,
        screen: None,
        duration: 8.0,
        ..Config::default()
    };
    let mut pop = evolution::create(&base).unwrap();
    let first_walker = pop.genomes.len();
    // Eight copies of the stored walker, an elite of a population evolved
    // under an earlier kernel. It no longer passes the 5 m check below (see
    // `docs/backlog.md`). Each copy has its own id, because the earthquake
    // gives every id its own bumps.
    for id in 0..8u64 {
        let mut walker: Creature =
            serde_json::from_str(include_str!("fixtures/warp_walker.json")).unwrap();
        walker.id = id;
        pop.push(walker);
    }
    let mut gpu = engine::gpu_engine("RTX 4060", 32).expect("GPU");
    assert!(gpu.name().contains("CUDA"), "opened {}", gpu.name());
    // Scores the whole population under `cfg`.
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
    assert!(
        calm.iter().all(|r| r.fitness > -1e19),
        "a calm trial failed"
    );
    let walked = (first_walker..calm.len())
        .map(|i| calm[i].fitness)
        .fold(f32::NEG_INFINITY, f32::max);
    eprintln!("the walkers get up to {walked:.2} m");
    assert!(walked > 5.0, "the walkers stand still ({walked} m)");
    // Each effect with the calm settings plus that one change.
    let effects: Vec<(&str, Config)> = vec![
        (
            "terrain",
            Config {
                terrain: 2,
                ..base.clone()
            },
        ),
        (
            "slope",
            Config {
                slope: 0.1,
                ..base.clone()
            },
        ),
        (
            "mud",
            Config {
                mud: 0.06,
                ..base.clone()
            },
        ),
        (
            "water",
            Config {
                water: 0.35,
                ..base.clone()
            },
        ),
        (
            "gaps",
            Config {
                gaps: 0.5,
                ..base.clone()
            },
        ),
        (
            "hurdles",
            Config {
                hurdles: 0.1,
                ..base.clone()
            },
        ),
        (
            "ice",
            Config {
                patches: 0.95,
                ..base.clone()
            },
        ),
        (
            "quake",
            Config {
                quake: 0.05,
                ..base.clone()
            },
        ),
        (
            "wind",
            Config {
                wind: 2.0,
                ..base.clone()
            },
        ),
        (
            "air",
            Config {
                air_retention: 0.98,
                ..base.clone()
            },
        ),
        (
            "gravity",
            Config {
                gravity: 5.0,
                ..base.clone()
            },
        ),
        (
            "grip",
            Config {
                ground_friction: 0.5,
                ..base.clone()
            },
        ),
        (
            "heat",
            Config {
                muscle_energy: 0.3,
                ..base.clone()
            },
        ),
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
