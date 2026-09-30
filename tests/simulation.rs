//! Search rules that need no physics: the seed, the mutation operators at
//! their limits, saves, world changes and the meteor. Scores are made up,
//! because the GPU is the only engine that scores creatures.
use evolution_simulator::{
    config::Config,
    evolution,
    qd::TrialMetrics,
    storage::{self, Experiment, Stage},
};
use std::path::PathBuf;

fn config() -> Config {
    Config {
        population: 32,
        random_seed: false,
        duration: 1.0,
        ..Default::default()
    }
}

fn path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("evolution-{}-{name}.evo", std::process::id()))
}

/// Gives every creature a score and a behavior of its own and archives them.
fn archive_made_up_results(e: &mut Experiment) {
    for i in 0..e.config.population {
        e.scores[i] = 1.0 + i as f32;
        e.trial_metrics[i] = TrialMetrics {
            ground_contact: 0.5,
            gait_frequency: (i % 8) as f32 * 0.75 + 0.1,
            mean_height: 0.5,
            feet: 2.0,
            ..TrialMetrics::default()
        };
    }
    e.evaluated = e.config.population;
    e.rank();
    e.archive_batch().unwrap();
}

#[test]
fn seed_is_repeatable_and_breeding_conserves_population() {
    let cfg = config();
    let a = evolution::create(&cfg).unwrap();
    let b = evolution::create(&cfg).unwrap();
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.bones, b.bones);
    assert_eq!(a.muscles, b.muscles);

    let scores: Vec<_> = (0..cfg.population).map(|i| i as f32).collect();
    let mut first = Experiment::new(cfg.clone()).unwrap();
    let mut second = Experiment::new(cfg.clone()).unwrap();
    first.scores.clone_from(&scores);
    second.scores.clone_from(&scores);
    first.evaluated = cfg.population;
    second.evaluated = cfg.population;
    first.archive_batch().unwrap();
    second.archive_batch().unwrap();
    first.prepare_next_batch().unwrap();
    second.prepare_next_batch().unwrap();
    assert_eq!(first.population.genomes.len(), cfg.population);
    first.population.validate(&cfg).unwrap();
    assert_eq!(first.population.nodes, second.population.nodes);
    assert_eq!(first.population.bones, second.population.bones);
    assert_eq!(first.population.muscles, second.population.muscles);
}

#[test]
fn mutation_keeps_valid_graphs_at_limits() {
    let cfg = Config {
        population: 64,
        max_nodes: 8,
        max_muscles: 8,
        mutation: 5.,
        ..config()
    };
    let mut e = Experiment::new(cfg.clone()).unwrap();
    for generation in 0..80 {
        for (i, score) in e.scores.iter_mut().enumerate() {
            *score = generation as f32 * 0.1 + i as f32;
        }
        e.evaluated = cfg.population;
        e.archive_batch().unwrap();
        e.prepare_next_batch().unwrap();
        e.population.validate(&cfg).unwrap();
    }
}

#[test]
fn a_checkpoint_mid_generation_resumes_from_its_archives() {
    // Saves keep the archives and search state, not the generation in
    // progress: a game saved before its first archive starts again from the
    // same random population.
    let mut e = Experiment::new(config()).unwrap();
    e.stage = Stage::Evaluating;
    for i in 0..7 {
        e.scores[i] = i as f32;
    }
    e.evaluated = 7;
    let checkpoint = path("partial");
    storage::save(&checkpoint, &e).unwrap();
    let loaded = storage::load(&checkpoint).unwrap();
    assert_eq!(loaded.evaluated, 0);
    assert_eq!(loaded.stage, Stage::Ready);
    assert_eq!(e.population.nodes, loaded.population.nodes);
    assert_eq!(e.population.muscles, loaded.population.muscles);
    // A file that is not a save does not load.
    std::fs::write(&checkpoint, b"garbage!").unwrap();
    assert!(storage::load(&checkpoint).is_err());
    let _ = std::fs::remove_file(checkpoint);
}

#[test]
fn statistics_count_every_creature() {
    let mut e = Experiment::new(config()).unwrap();
    e.scores = (0..e.config.population)
        .map(|i| i as f32 / 10. - 1.)
        .collect();
    e.evaluated = e.config.population;
    e.rank();
    let s = &e.history[0];
    assert_eq!(
        s.histogram.iter().map(|x| x.1 as usize).sum::<usize>(),
        e.config.population
    );
    assert_eq!(
        s.species.iter().map(|x| x.2 as usize).sum::<usize>(),
        e.config.population
    );
    assert!(s.best >= s.median && s.median >= s.worst);
}

#[test]
fn world_change_retests_archive_elites_in_generational_mode() {
    let mut e = Experiment::new(config()).unwrap();
    archive_made_up_results(&mut e);
    let elites: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    assert!(!elites.is_empty());
    e.update_config(Config {
        terrain: 1,
        ..e.config.clone()
    })
    .unwrap();
    let mut handed_over = Vec::new();
    e.prepare_next_batch_streaming(4, |pop, range, _| {
        handed_over.extend(range.map(|i| pop.creature(i).id));
        Ok(())
    })
    .unwrap();
    for id in elites {
        assert!(
            (0..e.config.population).any(|i| e.population.creature(i).id == id),
            "elite {id} was dropped by the world change"
        );
        assert!(
            handed_over.contains(&id),
            "elite {id} never reached a device"
        );
    }
}

#[test]
fn autosave_rotation_keeps_the_newest_and_spares_manual_saves() {
    let dir = std::env::temp_dir().join(format!("evolution-rotate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for seed in 0..5 {
        std::fs::write(dir.join(format!("seed-{seed}-auto.evo")), b"x").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::fs::write(dir.join("my-champion.evo"), b"x").unwrap();
    std::fs::write(dir.join("seed-9-auto.evo.tmp"), b"x").unwrap();
    assert_eq!(storage::rotate_autosaves(&dir, 3), 2);
    let mut left: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    // The fresh .tmp may belong to a save in progress, so it stays.
    assert_eq!(
        left,
        [
            "my-champion.evo",
            "seed-2-auto.evo",
            "seed-3-auto.evo",
            "seed-4-auto.evo",
            "seed-9-auto.evo.tmp"
        ]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn meteor_strike_can_be_undone() {
    let mut e = Experiment::new(config()).unwrap();
    archive_made_up_results(&mut e);
    let count = |e: &Experiment| {
        e.archive.entries.len() + e.islands.iter().map(|i| i.entries.len()).sum::<usize>()
    };
    let before = count(&e);
    let mut ids: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    ids.sort_unstable();
    let lost = e.meteor(0.5);
    assert!(lost > 0);
    assert_eq!(count(&e), before - lost);
    assert_eq!(e.undo_meteor(), lost);
    assert_eq!(count(&e), before);
    assert!(e.fossils.is_empty());
    let mut back: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    back.sort_unstable();
    assert_eq!(back, ids);
}
