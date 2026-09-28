//! The second screening rung in the experiment (`physics::SCREEN2_SECONDS`):
//! its bar keeps `physics::SCREEN2_KEEP` of the first screen's survivors, and
//! a creature it stops enters no archive.
use evolution_simulator::{config::Config, cpu_engine, physics, scheduler, storage::Experiment};

#[test]
fn the_second_bar_keeps_its_share_of_the_survivors_and_its_stops_enter_no_archive() {
    let cfg = Config {
        population: 512,
        duration: 36.0,
        random_seed: false,
        seed: 43,
        ..Config::default()
    };
    let mut experiment = Experiment::new(cfg).unwrap();
    let first = experiment.config.screen.unwrap();
    assert_eq!(first.bar, f32::NEG_INFINITY);
    let second = first.second.expect("a second rung");
    assert_eq!(
        (second.seconds, second.bar),
        (physics::SCREEN2_SECONDS, f32::NEG_INFINITY)
    );
    let mut late_stops = 0;
    for generation in 0..3 {
        let screen = experiment.config.screen.unwrap();
        let results = cpu_engine::evaluate(&experiment.population, &experiment.config);
        for (i, result) in results.iter().enumerate() {
            let metric =
                scheduler::to_metrics(&experiment.population, i, result, &experiment.config);
            experiment.record_result(i, &metric);
        }
        let late_ids: Vec<u64> = (0..experiment.config.population)
            .filter(|&i| results[i].screened > 0.0 && !screen.stopped_first(results[i].screened))
            .map(|i| experiment.population.genomes[i].id)
            .collect();
        if screen.second.unwrap().bar == f32::NEG_INFINITY {
            assert!(
                late_ids.is_empty(),
                "generation {generation}: no second bar yet"
            );
        }
        late_stops += late_ids.len();
        experiment.evaluated = experiment.config.population;
        experiment.archive_batch().unwrap();
        for elite in experiment
            .archive
            .entries
            .iter()
            .chain(experiment.islands.iter().flat_map(|island| &island.entries))
        {
            assert!(
                !late_ids.contains(&elite.creature.id),
                "a creature the second rung stopped entered an archive"
            );
        }
        let first_distances: Vec<f32> = results.iter().map(|r| r.screen_x).collect();
        let late_distances: Vec<f32> = results
            .iter()
            .map(|r| {
                if screen.stopped_first(r.screened) {
                    f32::NAN
                } else {
                    r.screen2_x
                }
            })
            .collect();
        experiment.prepare_next_batch().unwrap();
        let next = experiment.config.screen.unwrap();
        let late_bar = next.second.unwrap().bar;
        let survivors: Vec<f32> = first_distances
            .iter()
            .zip(&late_distances)
            .filter(|&(&x, late)| x >= next.bar && !late.is_nan())
            .map(|(_, &late)| late)
            .collect();
        let kept =
            survivors.iter().filter(|&&d| d >= late_bar).count() as f32 / survivors.len() as f32;
        assert!(
            (kept - physics::SCREEN2_KEEP).abs() < 0.05,
            "generation {generation}: second bar {late_bar} keeps {kept} of {} survivors",
            survivors.len()
        );
    }
    assert!(late_stops > 0, "the second rung must stop some creatures");
}
