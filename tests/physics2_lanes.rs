//! The fast CPU engine of physics v2 (`cpu_v2`, 16 creatures per SIMD group)
//! equals the scalar reference (`physics2::evaluate`) bit for bit, in calm
//! and in rough worlds, with and without the early screen.
use evolution_simulator::{
    config::Config,
    cpu_v2,
    creature_kernel::GpuResult,
    evolution::{self, Population},
    physics::Screen,
    physics2,
};

/// A population in which many creatures share a skeleton: the random
/// population, plus mutated copies of its bodies that keep the bones.
fn population(cfg: &Config) -> Population {
    let mut pop = evolution::create(cfg).unwrap();
    let n = pop.genomes.len();
    for i in 0..n {
        for copy in 1..4u32 {
            let mut c = pop.creature(i);
            c.id = c.id.wrapping_add(1000 * u64::from(copy));
            for m in &mut c.muscles {
                m.phase = (m.phase + 0.13 * copy as f32).fract();
                m.stiffness *= 1.0 + 0.1 * copy as f32;
            }
            for node in &mut c.nodes {
                node.friction = (node.friction + 0.05 * copy as f32).min(1.0);
            }
            pop.push(c);
        }
    }
    pop
}

fn compare(cfg: &Config, what: &str) {
    let pop = population(cfg);
    let reference = physics2::evaluate(&pop, cfg);
    let lanes = cpu_v2::evaluate(&pop, cfg);
    assert_eq!(reference.len(), lanes.len());
    // The comparison means something: bodies touched the ground and moved.
    assert!(reference.iter().any(|r| r.ground_contact > 0.0));
    assert!(reference.iter().any(|r| r.fitness.abs() > 0.01));
    let mut differ = 0;
    for (i, (a, b)) in reference.iter().zip(&lanes).enumerate() {
        let (x, y): (&GpuResult, &GpuResult) = (a, b);
        if bytemuck::bytes_of(x) != bytemuck::bytes_of(y) {
            differ += 1;
            if differ <= 3 {
                eprintln!(
                    "{what}: creature {i} differs: reference {} m fell {}, lanes {} m fell {}",
                    x.fitness, x.fall_time, y.fitness, y.fall_time
                );
            }
        }
    }
    assert_eq!(
        differ,
        0,
        "{what}: {differ} of {} results differ",
        lanes.len()
    );
}

fn base() -> Config {
    Config {
        population: 300,
        duration: 6.0,
        random_seed: false,
        screen: None,
        ..Config::default()
    }
}

#[test]
fn lanes_equal_the_reference_on_calm_ground() {
    compare(&base(), "calm");
}

#[test]
fn lanes_equal_the_reference_with_the_early_screen() {
    let cfg = Config {
        screen: Some(Screen {
            seconds: 1.5,
            bar: 0.0,
        }),
        ..base()
    };
    compare(&cfg, "screened");
}

#[test]
fn lanes_equal_the_reference_in_rough_worlds() {
    for (what, cfg) in [
        (
            "mud and wind",
            Config {
                mud: 0.05,
                wind: 1.5,
                ..base()
            },
        ),
        (
            "bumps, slope and quake",
            Config {
                terrain: 2,
                slope: 0.08,
                quake: 0.5,
                ..base()
            },
        ),
        (
            "gaps and hurdles",
            Config {
                gaps: 0.6,
                hurdles: 0.3,
                ..base()
            },
        ),
        (
            "heavy gravity, thin air, tired muscles",
            Config {
                gravity: 14.7,
                air_retention: 0.9,
                muscle_energy: 0.5,
                muscle_recovery: 0.4,
                ..base()
            },
        ),
    ] {
        compare(&cfg, what);
    }
}
