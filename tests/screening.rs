//! Early screening (`physics::Screen`): a standing creature below the bar at
//! the screen time stops, keeps its distance there, and enters no archive.
//! The bar is the distance at the screen that the top share reached. These
//! tests feed the experiment made-up results. `gpu_repeatability.rs` checks
//! that the kernel stops screened creatures and leaves survivors alone.
use evolution_simulator::{
    config::Config,
    physics,
    qd::{EvaluationMetrics, TrialMetrics},
    storage::Experiment,
};

fn config(seed: u64) -> Config {
    Config {
        population: 256,
        duration: 8.0,
        random_seed: false,
        seed,
        ..Config::default()
    }
}

/// What the scoring kernel returns for creature `i` under the experiment's
/// current bar: a spread of distances at the screen, and creatures below the
/// bar stopped there.
fn made_up_result(i: usize, bar: f32) -> EvaluationMetrics {
    let screen_x = ((i * 37) % 101) as f32 * 0.1;
    let screened = screen_x < bar;
    EvaluationMetrics {
        fitness: if screened { screen_x } else { screen_x + 5.0 },
        behavior: TrialMetrics {
            ground_contact: 0.5,
            gait_frequency: (i % 8) as f32 * 0.75 + 0.1,
            mean_height: 0.5,
            feet: 2.0,
            ..TrialMetrics::default()
        },
        unchecked: false,
        screened,
        screen_x,
        fine: false,
    }
}

#[test]
fn screened_creatures_enter_no_archive_and_the_bar_keeps_the_top_share() {
    let mut experiment = Experiment::new(config(43)).unwrap();
    // The first generation has no bar and runs every trial in full.
    let first = experiment
        .config
        .screen
        .expect("screening is on by default");
    assert_eq!(first.bar, f32::NEG_INFINITY);
    assert_eq!(first.seconds, physics::screen_seconds().unwrap());
    for generation in 0..3 {
        let bar = experiment.config.screen.unwrap().bar;
        let mut screened_ids = Vec::new();
        let mut distances = Vec::new();
        for i in 0..experiment.config.population {
            let metric = made_up_result(i, bar);
            if metric.screened {
                screened_ids.push(experiment.population.genomes[i].id);
            }
            distances.push(metric.screen_x);
            experiment.record_result(i, &metric);
        }
        if generation == 0 {
            assert!(screened_ids.is_empty(), "no bar in the first generation");
        } else {
            assert!(!screened_ids.is_empty(), "the bar must stop some creatures");
        }
        experiment.evaluated = experiment.config.population;
        experiment.archive_batch().unwrap();
        for elite in experiment
            .archive
            .entries
            .iter()
            .chain(experiment.islands.iter().flat_map(|island| &island.entries))
        {
            assert!(
                !screened_ids.contains(&elite.creature.id),
                "a screened creature entered an archive"
            );
        }
        experiment.prepare_next_batch().unwrap();
        let bar = experiment.config.screen.unwrap().bar;
        let kept = distances.iter().filter(|&&d| d >= bar).count() as f32 / distances.len() as f32;
        assert!(
            (kept - physics::screen_keep()).abs() < 0.05,
            "generation {generation}: bar {bar} keeps {kept}"
        );
    }
}

#[test]
fn a_generation_without_a_bar_sets_one_after_a_quarter_of_its_results() {
    let mut experiment = Experiment::new(config(53)).unwrap();
    let bar = |e: &Experiment| e.config.screen.unwrap().bar;
    let results: Vec<_> = (0..experiment.config.population)
        .map(|i| made_up_result(i, f32::NEG_INFINITY))
        .collect();
    for (i, metric) in results.iter().enumerate() {
        experiment.record_result(i, metric);
        experiment.arm_screen_early();
        if i + 1 < 64 {
            assert_eq!(bar(&experiment), f32::NEG_INFINITY, "result {i}");
        }
    }
    let armed = bar(&experiment);
    assert!(
        armed.is_finite(),
        "a quarter of the results must set the bar"
    );
    let kept = results[..64].iter().filter(|r| r.screen_x >= armed).count() as f32 / 64.0;
    assert!(
        (kept - physics::screen_keep()).abs() < 0.05,
        "the bar keeps {kept} of the sample"
    );
    // A world change forgets the old world's distances and the bar.
    let mut rough = experiment.config.clone();
    rough.terrain = 3;
    experiment.update_config(rough).unwrap();
    assert_eq!(bar(&experiment), f32::NEG_INFINITY);
}
