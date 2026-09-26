//! GPU/CPU agreement on evolved creature fixtures at fine (4x) fidelity.
//!
//! These tests need a Vulkan GPU and are ignored by default. On the
//! workstation run them with:
//!
//!     cargo test --release --test engine_agreement -- --ignored
//!
//! `gpu_matches_cpu_and_handles_partial_workgroups` in `tests/simulation.rs`
//! compares random populations at standard and fine fidelity. The two tests
//! here cover what random bodies cannot reach: an evolved walker and the
//! four-node sled exploit, plus the deterministic perturbed contender check
//! from `scheduler::perturb`.
use evolution_simulator::{
    config::Config,
    evolution::{self, Bone, Creature, Muscle, NodeGene, Population, Rng},
    gpu::Gpu,
    physics::Fidelity,
};

/// Trial length for the unperturbed evolved fixtures. Five seconds at the
/// fine rate is 1,200 timed steps after 400 settle steps, long enough for the
/// walker to travel several body lengths and for the planted-foot rule to
/// show on the sled. It is also short enough that the two engines' rounding
/// differences stay bounded; the existing random-body fine checks use 0.2 to
/// 0.5 s for the same reason.
const EVOLVED_TRIAL_SECONDS: f32 = 5.0;

/// Allowed fitness gap for the unperturbed evolved fixtures.
///
/// The random-population tests allow 0.05 m over short trials. Five seconds
/// is ten times the simulated time, and contacts are chaotic: a footfall can
/// land on a different step in the two implementations, after which the
/// contact sequence diverges. Those flips are real and grow with time, so the
/// tolerance has to be wider than the short-trial one. The measured gap at
/// five seconds on the workstation is 0.15 m for the walker (the sled agrees
/// to the millimetre), so 0.5 m is about three times the observed drift and
/// still an order of magnitude below the walker's travel. A wrong solver
/// change or a flipped joint break moves the score much farther than 0.5 m.
const EVOLVED_TOLERANCE: f32 = 0.5;

/// Trial length for the perturbed contender check. The check runs the
/// configured trial in production, but engine agreement cannot survive a long
/// chaotic trial: this fixture's unperturbed gap already grows past 2 m over
/// ten seconds. One second keeps the comparison meaningful (about 240 fine
/// steps) while staying deterministic, the same compromise the existing
/// fine-fidelity tests make.
const CHECK_TRIAL_SECONDS: f32 = 1.0;

/// Allowed fitness gap for the perturbed contender check, the same 0.05 m the
/// existing standard/fine random comparisons use. The measured gap at one
/// second is 0.002 m for the perturbed walker and 0.000 m for the sled, so the
/// tolerance leaves more than an order of magnitude of headroom. A fall or
/// joint-break decision that flips to a different step shifts the score far
/// more than that, so it still fails instead of hiding.
const CHECK_TOLERANCE: f32 = 0.05;

/// A trial failed when the engine returned the broken-joint sentinel. A head
/// fall is not a failure: both engines still return a finite distance at the
/// fall. The tests assert that status agrees rather than either fall time.
fn failed(fitness: f32) -> bool {
    !fitness.is_finite() || fitness <= evolution::FAILED
}

fn fine_config(population: usize, duration: f32) -> Config {
    Config {
        population,
        random_seed: false,
        duration,
        fidelity: Some(Fidelity::fine()),
        ..Config::default()
    }
}

/// The exact perturbation `scheduler::perturb` applies to a contender: a
/// deterministic per-creature `Rng` seeded from `id ^ 0x5eed_7a11`, up to
/// 2 cm of node offset (never downward, so the pose does not start in the
/// ground), and grip scaled by +/- 10% and clamped to the gene range. This is
/// a deliberate copy because the production function is private; if it
/// changes, this test must change with it.
fn perturb(creature: &mut Creature) {
    let mut rng = Rng::new(creature.id ^ 0x5eed_7a11, 0, 0);
    for node in &mut creature.nodes {
        node.x += rng.range(-0.02, 0.02);
        node.y += rng.range(0.0, 0.02);
        node.friction = (node.friction * rng.range(0.9, 1.1)).clamp(0.0, 1.0);
    }
}

/// Asserts the failure agreement and the fitness tolerance for one creature.
/// A joint-break flip scores the `FAILED` sentinel on one engine, so it
/// cannot pass the distance bound and surfaces here; a fall flip (both scores
/// finite) passes only while the fall landed close enough that the two
/// distances remain within `tolerance` of each other.
fn assert_pair(label: &str, index: usize, gpu: f32, cpu: f32, tolerance: f32) {
    assert_eq!(
        failed(gpu),
        failed(cpu),
        "{label} {index}: engines disagree on whether the trial failed: GPU {gpu}, CPU engine {cpu}"
    );
    if failed(gpu) {
        return;
    }
    let gap = (gpu - cpu).abs();
    assert!(
        gap <= tolerance,
        "{label} {index}: fine-fidelity scores differ by {gap} m (GPU {gpu}, CPU engine {cpu}), tolerance {tolerance} m"
    );
}

/// Opens the primary GPU and refuses to continue when it did not open: with
/// the CPU fallback the comparison would silently pass against the CPU.
fn open_gpu() -> Gpu {
    let gpu = Gpu::new("RTX 4060").expect("GPU");
    assert!(
        gpu.startup_warning.is_none(),
        "the primary GPU did not open ({}); the comparison would silently run on the CPU",
        gpu.startup_warning.as_deref().unwrap_or("unknown")
    );
    gpu
}

fn evaluate_on_gpu(gpu: &mut Gpu, pop: &Population, cfg: &Config) -> Vec<f32> {
    let indices: Vec<usize> = (0..pop.genomes.len()).collect();
    gpu.sched
        .as_mut()
        .expect("scheduler")
        .evaluate_single(pop, &indices, cfg)
        .expect("GPU evaluation")
        .into_iter()
        .map(|metrics| metrics.fitness)
        .collect()
}

/// Item 110: one fine-fidelity trial per evolved fixture on the GPU and on
/// the production CPU engine, compared on fitness, failure status, and the
/// normalized behavior metrics. The fixtures are small and fixed, so the test
/// is deterministic for a given engine pair.
#[test]
#[ignore = "requires a Vulkan GPU; run explicitly on the workstation"]
fn evolved_creatures_agree_at_fine_fidelity() {
    let mut pop = Population::default();
    pop.push(energy_dependent_walker());
    pop.push(sled_creature());
    let cfg = fine_config(pop.genomes.len(), EVOLVED_TRIAL_SECONDS);

    let mut gpu = open_gpu();
    let gpu_results = gpu
        .sched
        .as_mut()
        .expect("scheduler")
        .evaluate_single(&pop, &(0..pop.genomes.len()).collect::<Vec<_>>(), &cfg)
        .expect("GPU evaluation");
    let cpu_results = evolution_simulator::cpu_engine::evaluate(&pop, &cfg);
    assert_eq!(gpu_results.len(), cpu_results.len());

    for (i, (gpu_result, cpu_result)) in gpu_results.iter().zip(&cpu_results).enumerate() {
        let label = if i == 0 { "walker" } else { "sled" };
        assert_pair(
            label,
            i,
            gpu_result.fitness,
            cpu_result.fitness,
            EVOLVED_TOLERANCE,
        );
        if failed(gpu_result.fitness) {
            continue;
        }
        // Behavior metrics follow the same chaotic contacts as fitness, so
        // they cannot be tighter than the fitness tolerance. Compare them at
        // a coarse level only: the contact fraction, mean height, and feet
        // count should track, while gait phase can drift.
        let cpu_metrics = evolution_simulator::scheduler::to_metrics(&pop, i, cpu_result, &cfg);
        assert!(
            (gpu_result.behavior.ground_contact - cpu_metrics.behavior.ground_contact).abs() < 0.2,
            "{label} {i}: ground contact {} vs {}",
            gpu_result.behavior.ground_contact,
            cpu_metrics.behavior.ground_contact
        );
        assert!(
            (gpu_result.behavior.mean_height - cpu_metrics.behavior.mean_height).abs() < 0.1,
            "{label} {i}: mean height {} vs {}",
            gpu_result.behavior.mean_height,
            cpu_metrics.behavior.mean_height
        );
        assert_eq!(
            gpu_result.behavior.feet, cpu_metrics.behavior.feet,
            "{label} {i}: foot count differs"
        );
        eprintln!(
            "{label} {i}: GPU {:.4} m, CPU engine {:.4} m, contact {:.3}/{:.3}, height {:.3}/{:.3}, feet {}",
            gpu_result.fitness,
            cpu_result.fitness,
            gpu_result.behavior.ground_contact,
            cpu_metrics.behavior.ground_contact,
            gpu_result.behavior.mean_height,
            cpu_metrics.behavior.mean_height,
            gpu_result.behavior.feet,
        );
    }
}

/// Item 111: the perturbed contender check across engines. The check exists
/// because a gait that only works from one exact pose should not hold an
/// archive cell: it copies a contender, applies `scheduler::perturb`, and runs
/// the copy at fine fidelity. This test reproduces that exact deterministic
/// perturbed population (the evolved walker and the sled, each under its own
/// id's perturbation stream), runs one fine trial per copy on both engines,
/// and asserts the scores agree within `CHECK_TOLERANCE`.
///
/// A fall or joint-break decision can still flip on a rounding difference; the
/// test makes no attempt to force those decisions bit-identical. It allows a
/// flip only while both engines' distances stay within the tolerance of each
/// other, which is exactly the fitness assertion. That covers a fall at the
/// trailing edge of a trial, and it rejects a joint-break flip, because the
/// `FAILED` sentinel is far outside any finite distance. `docs/validation.md`
/// records this decision and its limits.
#[test]
#[ignore = "requires a Vulkan GPU; run explicitly on the workstation"]
fn perturbed_contenders_agree_at_fine_fidelity() {
    let mut source = Population::default();
    source.push(energy_dependent_walker());
    source.push(sled_creature());

    let mut perturbed = Population::default();
    let mut moved = 0usize;
    for i in 0..source.genomes.len() {
        let mut creature = source.creature(i);
        let before = creature.nodes.clone();
        perturb(&mut creature);
        if creature.nodes != before {
            moved += 1;
        }
        perturbed.push(creature);
    }
    assert_eq!(moved, source.genomes.len(), "the perturbation did nothing");

    let cfg = fine_config(perturbed.genomes.len(), CHECK_TRIAL_SECONDS);
    let mut gpu = open_gpu();
    let gpu_scores = evaluate_on_gpu(&mut gpu, &perturbed, &cfg);
    let cpu_results = evolution_simulator::cpu_engine::evaluate(&perturbed, &cfg);
    assert_eq!(gpu_scores.len(), cpu_results.len());
    assert_eq!(gpu_scores.len(), source.genomes.len());

    for (i, &gpu) in gpu_scores.iter().enumerate() {
        let cpu = cpu_results[i].fitness;
        let label = if i == 0 { "walker" } else { "sled" };
        eprintln!(
            "{label} {i}: GPU {gpu:.4} m, CPU engine {cpu:.4} m (CPU fall_time {:.3}, head_shake {:.3})",
            cpu_results[i].fall_time, cpu_results[i].head_shake
        );
        assert_pair(label, i, gpu, cpu, CHECK_TOLERANCE);
    }
}

/// A six-node walker evolved under the calm defaults (population 2048,
/// 30 generations, seed 38, 15 s trials). It travels about 8 m in 5 s with
/// full energy stores, and loses more than half of that when the heat wave
/// shrinks the stores or drought slows recovery. Copied from
/// `tests/simulation.rs` so the fine-fidelity comparison uses the same
/// fixture the CPU hot/cold tests use.
fn energy_dependent_walker() -> Creature {
    Creature {
        nodes: vec![
            NodeGene {
                x: -0.5593486,
                y: 0.61968184,
                diameter: 0.12,
                friction: 0.9475196,
            },
            NodeGene {
                x: -0.41776797,
                y: 0.3038751,
                diameter: 0.06,
                friction: 0.7486749,
            },
            NodeGene {
                x: -0.116856754,
                y: 0.3178858,
                diameter: 0.108820364,
                friction: 0.66710335,
            },
            NodeGene {
                x: -0.039948717,
                y: 0.43371728,
                diameter: 0.0877441,
                friction: 0.9974566,
            },
            NodeGene {
                x: 0.19000307,
                y: 0.37280446,
                diameter: 0.07844837,
                friction: 0.65871984,
            },
            NodeGene {
                x: -0.27397153,
                y: 0.3718743,
                diameter: 0.08361293,
                friction: 0.65,
            },
        ],
        bones: vec![
            Bone {
                a: 0,
                b: 1,
                rest_length: 0.34609097,
                min_angle: -1.6000074,
                max_angle: 0.7977927,
                organ_mass: 0.01,
                organ_at: 0.5291506,
            },
            Bone {
                a: 1,
                b: 2,
                rest_length: 0.30123723,
                min_angle: -0.65684575,
                max_angle: 1.6548456,
                organ_mass: 0.0,
                organ_at: 0.5,
            },
            Bone {
                a: 2,
                b: 3,
                rest_length: 0.13903877,
                min_angle: -1.5915323,
                max_angle: 0.633044,
                organ_mass: 0.0,
                organ_at: 0.5,
            },
            Bone {
                a: 3,
                b: 4,
                rest_length: 0.23788273,
                min_angle: -1.9740598,
                max_angle: 0.9501501,
                organ_mass: 0.010241741,
                organ_at: 0.4697004,
            },
            Bone {
                a: 3,
                b: 5,
                rest_length: 0.24205624,
                min_angle: -0.82507557,
                max_angle: 2.0943952,
                organ_mass: 0.0,
                organ_at: 0.5,
            },
        ],
        muscles: vec![
            Muscle {
                bone_a: 0,
                bone_b: 1,
                anchor_a: 0.9631201,
                anchor_b: 0.6989166,
                short: 0.18162505,
                long: 0.18962646,
                period: 0.5951622,
                phase: 0.42644,
                duty: 0.4687704,
                stiffness: 35.06757,
                sensor: 0,
                reset: 0.43866375,
            },
            Muscle {
                bone_a: 1,
                bone_b: 2,
                anchor_a: 0.5982309,
                anchor_b: 0.66608244,
                short: 0.08557357,
                long: 0.18462573,
                period: 0.5951622,
                phase: 0.016101224,
                duty: 0.38960835,
                stiffness: 84.78909,
                sensor: 1,
                reset: 0.19877157,
            },
            Muscle {
                bone_a: 2,
                bone_b: 3,
                anchor_a: 0.08075203,
                anchor_b: 0.08479669,
                short: 0.21270484,
                long: 0.5398381,
                period: 0.5951622,
                phase: 0.59647053,
                duty: 0.18719086,
                stiffness: 35.218353,
                sensor: 255,
                reset: 0.2708392,
            },
            Muscle {
                bone_a: 3,
                bone_b: 0,
                anchor_a: 0.58470476,
                anchor_b: 0.39771268,
                short: 0.5982563,
                long: 0.9116529,
                period: 0.5951622,
                phase: 0.07857889,
                duty: 0.4873734,
                stiffness: 57.975662,
                sensor: 255,
                reset: 0.6004214,
            },
            Muscle {
                bone_a: 1,
                bone_b: 2,
                anchor_a: 0.2654894,
                anchor_b: 0.3410796,
                short: 0.10381305,
                long: 0.11306952,
                period: 0.5951622,
                phase: 0.5893883,
                duty: 0.2278046,
                stiffness: 73.649185,
                sensor: 255,
                reset: 0.91381097,
            },
            Muscle {
                bone_a: 2,
                bone_b: 4,
                anchor_a: 0.06489495,
                anchor_b: 0.092396125,
                short: 0.20794365,
                long: 0.53504324,
                period: 0.5951622,
                phase: 0.9986213,
                duty: 0.22768927,
                stiffness: 33.61832,
                sensor: 255,
                reset: 0.2898468,
            },
            Muscle {
                bone_a: 4,
                bone_b: 0,
                anchor_a: 0.57180727,
                anchor_b: 0.3885777,
                short: 0.6111155,
                long: 0.9209887,
                period: 0.5951622,
                phase: 0.5952037,
                duty: 0.37932187,
                stiffness: 71.13563,
                sensor: 255,
                reset: 0.60610723,
            },
            Muscle {
                bone_a: 3,
                bone_b: 4,
                anchor_a: 0.07086325,
                anchor_b: 0.26907945,
                short: 0.059728526,
                long: 0.1037927,
                period: 0.5951622,
                phase: 0.13944362,
                duty: 0.62057906,
                stiffness: 31.938673,
                sensor: 255,
                reset: 0.008283809,
            },
        ],
        id: 47_300,
        mutability: 0.87713593,
    }
}

/// The four-node sled evolution built while friction could push the body in
/// any direction: a flat chain of nodes resting on the ground, driven by
/// muscles that pull it together along its length. Its momentum came almost
/// entirely from the center-of-mass shift of the bone passes. Copied from
/// `tests/simulation.rs`; the planted-foot rule must keep it slow on both
/// engines.
fn sled_creature() -> Creature {
    let spacing = 0.7;
    let diameter = 0.16;
    let nodes: Vec<_> = (0..4)
        .map(|i| NodeGene {
            x: i as f32 * spacing,
            y: diameter * 0.5,
            diameter,
            friction: 1.0,
        })
        .collect();
    let bones: Vec<_> = (0..3)
        .map(|i| Bone::new(i as u32, i as u32 + 1, spacing))
        .collect();
    // Each muscle spans two bones, from bone i's start to bone i+1's end, so
    // contracting it drags the chain together while every node stays down.
    let muscles: Vec<_> = (0..2)
        .map(|i| Muscle {
            bone_a: i as u32,
            bone_b: i as u32 + 1,
            anchor_a: 0.0,
            anchor_b: 1.0,
            short: spacing * 1.05,
            long: spacing * 2.0,
            period: 1.0,
            phase: i as f32 * 0.5,
            duty: 0.5,
            stiffness: 120.0,
            sensor: 255,
            reset: 0.0,
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
