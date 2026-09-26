//! Regression tests for search changes measured with `examples/search_ab.rs`.
//!
//! Items 83 and 67: a bounded, periodic CPU re-test of archive elites keeps the
//! lower score, so a lucky trial cannot hold a cell. Since item 67 the re-test
//! is a fresh deterministic perturbation of the elite (the contender check's
//! pose and grip shift), because the stored score already folded in the
//! unperturbed standard trial. It is opt-in through `EVOLUTION_ELITE_REFRESH`
//! (generations) and off by default.
use evolution_simulator::{
    config::Config,
    cpu_engine,
    evolution::{self, Creature, Population},
    qd::{self, Elite, Emitter},
    scheduler,
    storage::{self, Experiment},
};

fn config(seed: u64) -> Config {
    Config {
        population: 16,
        duration: 0.1,
        seed,
        random_seed: false,
        max_nodes: 8,
        max_muscles: 12,
        ..Config::default()
    }
}

/// A slower config for the tests that need a real gait difference between the
/// exact pose and its fresh perturbation.
fn perturb_config(seed: u64) -> Config {
    Config {
        population: 64,
        duration: 0.5,
        seed,
        random_seed: false,
        max_nodes: 8,
        max_muscles: 12,
        ..Config::default()
    }
}

/// The standard single-trial fitness of one creature on the CPU engine, the
/// same call the archive-admission check and the refresh use.
fn standard_fitness(creature: &Creature, cfg: &Config) -> f32 {
    let mut unit = Population::default();
    unit.push(creature.clone());
    let results = cpu_engine::evaluate(&unit, cfg);
    scheduler::to_metrics(&unit, 0, &results[0], cfg).fitness
}

/// The score the current refresh stores: the standard trial of the fresh
/// deterministic perturbation the refresh applies.
fn refreshed_fitness(creature: &Creature, cfg: &Config) -> f32 {
    let mut perturbed = creature.clone();
    storage::perturb_elite(&mut perturbed);
    standard_fitness(&perturbed, cfg)
}

/// An archive entry with a chosen cadence so each test creature gets its own
/// cell. Only the fitness is dishonest; the refresh reads the creature.
fn stored_elite(creature: &Creature, fitness: f32, cadence: f32) -> Elite {
    let descriptor = qd::Descriptor {
        nodes: creature.nodes.len() as u16,
        muscles: creature.muscles.len() as u16,
        ground_contact: 0.5,
        gait_frequency: cadence,
        aspect_ratio: 1.0,
        vertical_oscillation: 0.0,
        mean_height: 0.5,
        feet: 2.0,
    };
    Elite {
        niche: descriptor.niche(),
        descriptor,
        creature: creature.clone(),
        fitness,
        emitter: Emitter::Cma,
        improved_generation: 0,
        protected_until: 0,
        visits: 0,
        topology: qd::Topology::of(creature),
    }
}

/// Random creatures whose standard trial scores, so a refresh has a real
/// score to lower to.
fn viable_creatures(experiment: &Experiment, count: usize) -> Vec<Creature> {
    let mut out = Vec::new();
    for index in 0..experiment.config.population {
        let creature = experiment.population.creature(index);
        let fitness = standard_fitness(&creature, &experiment.config);
        if fitness.is_finite() && fitness > evolution::FAILED {
            out.push(creature);
            if out.len() == count {
                break;
            }
        }
    }
    out
}

#[test]
fn a_lucky_elite_is_lowered_after_a_refresh_and_a_stable_one_is_kept() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    let creatures = viable_creatures(&experiment, 2);
    assert_eq!(creatures.len(), 2, "no viable random creature in the pool");
    let stable_fresh = refreshed_fitness(&creatures[0], &experiment.config);
    let lucky_fresh = refreshed_fitness(&creatures[1], &experiment.config);
    let stable_niche = stored_elite(&creatures[0], stable_fresh, 0.5).niche;
    let lucky_niche = stored_elite(&creatures[1], lucky_fresh, 2.5).niche;
    assert_ne!(stable_niche, lucky_niche, "test needs two cells");
    experiment
        .archive
        .entries
        .push(stored_elite(&creatures[0], stable_fresh, 0.5));
    experiment
        .archive
        .entries
        .push(stored_elite(&creatures[1], lucky_fresh + 500.0, 2.5));
    experiment.archive.rebuild_indices();
    let qd_before = experiment.archive.qd_score;
    assert_eq!(experiment.archive.behavior_count(), 2);

    // Interval 1 is due at generation 0 and reaches both entries.
    let lowered = experiment.refresh_elites(1);
    assert_eq!(lowered, 1, "only the lucky elite should change");
    let stable = &experiment.archive.entries[0];
    let lucky = &experiment.archive.entries[1];
    assert!(
        (stable.fitness - stable_fresh).abs() < 1e-3,
        "stable elite moved from {stable_fresh} to {}",
        stable.fitness
    );
    assert!(
        (lucky.fitness - lucky_fresh).abs() < 1e-3,
        "lucky elite {} was not lowered to its fresh perturbed trial {lucky_fresh}",
        lucky.fitness
    );
    assert!(lucky.fitness < lucky_fresh + 500.0);
    assert_eq!(stable.niche, stable_niche, "refresh must keep cells");
    assert_eq!(lucky.niche, lucky_niche, "refresh must keep cells");
    assert!(
        experiment.archive.qd_score < qd_before,
        "qd score must drop with the lowered elite"
    );
}

#[test]
fn a_fresh_perturbation_can_disprove_the_admitted_standard_trial() {
    let experiment_config = perturb_config(43);
    let mut experiment = Experiment::new(experiment_config.clone()).unwrap();
    let mut found = None;
    for index in 0..experiment_config.population {
        let creature = experiment.population.creature(index);
        let standard = standard_fitness(&creature, &experiment.config);
        if !standard.is_finite() || standard <= evolution::FAILED {
            continue;
        }
        let fresh = refreshed_fitness(&creature, &experiment.config);
        if fresh.is_finite() && fresh < standard - 1e-4 {
            found = Some((creature, standard, fresh));
            break;
        }
    }
    let (creature, standard, fresh) =
        found.expect("no random creature was weaker under perturbation; the test proves nothing");
    // The archive-admission check stores the exact-pose standard trial.
    experiment
        .archive
        .entries
        .push(stored_elite(&creature, standard, 0.5));
    experiment.archive.rebuild_indices();
    assert_eq!(experiment.refresh_elites(1), 1);
    let elite = &experiment.archive.entries[0];
    assert!(
        (elite.fitness - fresh).abs() < 1e-3,
        "stored {} instead of the fresh perturbed {fresh}",
        elite.fitness
    );
    assert!(elite.fitness < standard);
}

#[test]
fn a_refresh_cycle_is_bounded_and_rotates_through_the_archive() {
    let mut experiment = Experiment::new(config(39)).unwrap();
    let creatures = viable_creatures(&experiment, 8);
    assert_eq!(creatures.len(), 8, "not enough viable random creatures");
    for (index, creature) in creatures.iter().enumerate() {
        let fitness = standard_fitness(creature, &experiment.config) + 100.0;
        let cadence = 0.25 + index as f32 * 0.5;
        experiment
            .archive
            .entries
            .push(stored_elite(creature, fitness, cadence));
    }
    experiment.archive.rebuild_indices();
    assert_eq!(experiment.archive.behavior_count(), 8);

    // Not due before the interval boundary, due after it.
    assert_eq!(experiment.refresh_elites(2), 0, "must not run early");
    experiment.generation = 1;
    assert_eq!(experiment.refresh_elites(2), 4, "first batch of four");
    experiment.generation = 3;
    assert_eq!(experiment.refresh_elites(2), 4, "rotated second batch");
    experiment.generation = 5;
    assert_eq!(
        experiment.refresh_elites(2),
        0,
        "already corrected entries are left alone"
    );
}

#[test]
fn elite_refresh_is_opt_in_through_the_environment() {
    unsafe {
        std::env::remove_var("EVOLUTION_ELITE_REFRESH");
    }
    assert_eq!(storage::elite_refresh_interval(), 0);
    let mut experiment = Experiment::new(config(40)).unwrap();
    let creatures = viable_creatures(&experiment, 1);
    assert_eq!(creatures.len(), 1, "no viable random creature in the pool");
    let true_fitness = standard_fitness(&creatures[0], &experiment.config);
    experiment
        .archive
        .entries
        .push(stored_elite(&creatures[0], true_fitness + 500.0, 0.5));
    experiment.archive.rebuild_indices();

    assert_eq!(
        experiment.refresh_elites_from_env(),
        0,
        "refresh must be off by default"
    );
    unsafe {
        std::env::set_var("EVOLUTION_ELITE_REFRESH", "1");
    }
    assert_eq!(storage::elite_refresh_interval(), 1);
    assert_eq!(
        experiment.refresh_elites_from_env(),
        1,
        "the variable must enable the refresh"
    );
    unsafe {
        std::env::remove_var("EVOLUTION_ELITE_REFRESH");
    }
    assert_eq!(storage::elite_refresh_interval(), 0);
}
