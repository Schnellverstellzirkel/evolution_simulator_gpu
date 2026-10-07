//! Tests of breeding and of the `storage::Experiment` game, scored by `by_slot`
//! instead of the GPU. They check that a fixed seed gives the same bodies, that
//! breeding keeps the population, copies genes and keeps bodies valid at their
//! limits, and that a block holds the same creatures whatever gene arena it is
//! bred into. They also check saves, the statistics, world changes, autosave
//! rotation, the score an archive keeps, meteor strikes and the muscle
//! waveform. The tests of archives, islands and the ring are in
//! `tests/search_state.rs`.

use evolution_simulator::{
    config::Config,
    evolution::{self, Creature, Muscle},
    physics,
    qd::{Elite, Emitter, EvaluationMetrics, QdArchive},
    storage::{self, Experiment},
};
use std::path::PathBuf;

/// The settings of these tests: 32 creatures, 1 s trials and the default seed,
/// which stays fixed because `random_seed` is off. A test overrides the fields
/// it needs.
fn config() -> Config {
    Config {
        population: 32,
        random_seed: false,
        duration: 1.0,
        ..Default::default()
    }
}

/// Made-up results for a block. A creature scores 0.1 times its birth slot, the
/// ring slot it was born in, so the score grows with the slot. Every other
/// metric has its default. The settings are ignored, so a confirmation trial
/// scores the same as a standard one.
fn by_slot(pop: &evolution::Population, _: &Config) -> anyhow::Result<Vec<EvaluationMetrics>> {
    Ok(pop
        .genomes
        .iter()
        .map(|g| EvaluationMetrics {
            fitness: evolution::slot_of_id(g.id) as f32 * 0.1,
            ..Default::default()
        })
        .collect())
}

/// A path for a save in the temporary directory. The file name holds the
/// process id and `name`, and each test uses its own `name`.
fn path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("evolution-{}-{name}.evo", std::process::id()))
}

#[test]
fn seed_is_repeatable_and_breeding_conserves_population() {
    let cfg = config();
    let a = evolution::create(&cfg).unwrap();
    let b = evolution::create(&cfg).unwrap();
    assert_eq!(a.nodes, b.nodes);
    assert_eq!(a.bones, b.bones);
    assert_eq!(a.muscles, b.muscles);

    // Two games with the same settings hold the same ring after a generation.
    let mut first = Experiment::new(cfg.clone()).unwrap();
    let mut second = Experiment::new(cfg.clone()).unwrap();
    first.run_generation(&mut by_slot).unwrap();
    second.run_generation(&mut by_slot).unwrap();
    assert_eq!(first.ring_len(), cfg.population);
    first.validate().unwrap();
    for (x, y) in first.blocks.iter().zip(&second.blocks) {
        assert_eq!(x.population.nodes, y.population.nodes);
        assert_eq!(x.population.bones, y.population.bones);
        assert_eq!(x.population.muscles, y.population.muscles);
    }
}

/// Asserts that two creatures have the same numbers of nodes, bones and
/// muscles, the same bone ends, muscle bones and muscle sensors, and every
/// other gene within 1e-4. It leaves out the ids and the muscles' tendons.
fn assert_genomes_close(a: &Creature, b: &Creature) {
    let close = |x: f32, y: f32| (x - y).abs() <= 1e-4;
    assert_eq!(a.nodes.len(), b.nodes.len());
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert!(close(x.x, y.x) && close(x.y, y.y));
        assert!(close(x.diameter, y.diameter) && close(x.friction, y.friction));
    }
    assert_eq!(a.bones.len(), b.bones.len());
    for (x, y) in a.bones.iter().zip(&b.bones) {
        assert_eq!((x.a, x.b), (y.a, y.b));
        assert!(close(x.rest_length, y.rest_length));
        assert!(close(x.min_angle, y.min_angle) && close(x.max_angle, y.max_angle));
        assert!(close(x.organ_mass, y.organ_mass) && close(x.organ_at, y.organ_at));
    }
    assert_eq!(a.muscles.len(), b.muscles.len());
    for (x, y) in a.muscles.iter().zip(&b.muscles) {
        assert_eq!(
            (x.bone_a, x.bone_b, x.sensor),
            (y.bone_a, y.bone_b, y.sensor)
        );
        assert!(close(x.anchor_a, y.anchor_a) && close(x.anchor_b, y.anchor_b));
        assert!(close(x.short, y.short) && close(x.long, y.long));
        assert!(close(x.period, y.period) && close(x.phase, y.phase));
        assert!(close(x.duty, y.duty) && close(x.stiffness, y.stiffness));
        assert!(close(x.reset, y.reset));
    }
}

/// Breeds eight CMA children of `parent`, the only elite of the archive. A CMA
/// child planned with no CMA emitter is its parent plus gene noise scaled by
/// `cfg.mutation`, so with `cfg.mutation` at 0 it is a repaired copy of the
/// parent.
fn zero_mutation_children(cfg: &Config, parent: &Creature) -> Vec<Creature> {
    let mut archive = QdArchive::default();
    archive.entries.push(Elite {
        niche: Default::default(),
        descriptor: Default::default(),
        creature: parent.clone().into(),
        fitness: 1.0,
        emitter: Emitter::Cma,
        improved_generation: 0,
        protected_until: 0,
        visits: 0,
        topology: Default::default(),
        graduate: false,
        fine: false,
    });
    let plans: Vec<_> = (0..8)
        .map(|_| evolution::CandidatePlan {
            emitter: Emitter::Cma,
            parent: Some(0),
            cma: None,
            mate: None,
            seed: false,
        })
        .collect();
    let slots: Vec<usize> = (0..8).collect();
    let mut children = evolution::Population::default();
    children.breed(
        8,
        None,
        &mut [],
        &[archive],
        &[],
        &plans,
        &slots,
        &slots,
        cfg,
        0,
        0,
    );
    (0..8).map(|k| children.creature(k)).collect()
}

#[test]
fn zero_mutation_copies_genetics() {
    let cfg = Config {
        mutation: 0.,
        ..config()
    };
    let population = evolution::create(&cfg).unwrap();
    // A random body gets some muscles after its last repair, so their periods
    // are not on the body's clock yet. Breeding repairs every child, which puts
    // them on the clock. So a child of a random body can differ from it in
    // those periods, and only a repaired body is copied exactly. The test
    // therefore copies a repaired child and not the random body.
    let repaired = zero_mutation_children(&cfg, &population.creature(0)).swap_remove(0);
    for child in zero_mutation_children(&cfg, &repaired) {
        assert_genomes_close(&child, &repaired);
    }
}

/// A block holds the same creatures when it is bred into a new gene arena, into
/// the arena of an earlier block of the same size, and into one whose parts are
/// too small, so that children go after the parts. `Population::breed` gives
/// each run of children a part of the arena.
#[test]
fn breeding_into_a_reused_arena_gives_the_same_creatures() {
    let cfg = Config {
        population: 64,
        ..config()
    };
    let parents = evolution::create(&cfg).unwrap();
    // The 64 random bodies are the elites of one archive.
    let mut archive = QdArchive::default();
    for i in 0..parents.genomes.len() {
        archive.entries.push(Elite {
            niche: Default::default(),
            descriptor: Default::default(),
            creature: parents.creature(i).into(),
            fitness: 1.0,
            emitter: Emitter::Structural,
            improved_generation: 0,
            protected_until: 0,
            visits: 0,
            topology: evolution_simulator::qd::topology_of_population(&parents, i),
            graduate: false,
            fine: false,
        });
    }
    // One archive in the list, so every child breeds from it.
    let archive = [archive];
    // 9,000 children make several runs of `BREED_CHUNK`, and each run gets a
    // part of the arena.
    let count = 9_000;
    let emitters = [
        Emitter::Structural,
        Emitter::Novelty,
        Emitter::Restart,
        Emitter::Cma,
    ];
    // Position 5 holds a given elite, as a reseeded elite does. Every other
    // position is bred.
    let positions: Vec<usize> = (0..count).filter(|&k| k != 5).collect();
    let plans: Vec<_> = positions
        .iter()
        .map(|&k| evolution::CandidatePlan {
            emitter: emitters[k % 4],
            parent: Some(k % 64),
            cma: None,
            mate: None,
            seed: false,
        })
        .collect();
    let slots: Vec<usize> = positions.iter().map(|&k| 1000 + k).collect();
    // The elite at position 5 is parent 7.
    let lead = || vec![(5, parents.creature(7))];
    let breed = |arena: &mut evolution::Population, round: u64| {
        arena.breed(
            count,
            None,
            &mut lead(),
            &archive,
            &[],
            &plans,
            &slots,
            &positions,
            &cfg,
            3,
            round,
        )
    };
    let mut fresh = evolution::Population::default();
    breed(&mut fresh, 2);
    // Reused: the arena already held a block of this size, bred in round 1.
    let mut reused = evolution::Population::default();
    breed(&mut reused, 1);
    breed(&mut reused, 2);
    // Too small: the arena's creatures hold no genes, so each part has little
    // room. `breed` returns how many children went after the parts.
    let mut small = evolution::Population {
        genomes: vec![Default::default(); count],
        ..Default::default()
    };
    let late = breed(&mut small, 2);
    assert!(late > 0);
    for arena in [&reused, &small] {
        for k in 0..count {
            let (a, b) = (fresh.creature(k), arena.creature(k));
            assert_eq!(a.id, b.id);
            assert_eq!(a.nodes, b.nodes);
            assert_eq!(a.bones, b.bones);
            assert_eq!(a.muscles, b.muscles);
        }
    }
    assert_eq!(fresh.creature(5).nodes, parents.creature(7).nodes);
}

/// Bodies capped at 8 nodes and 8 muscles, bred at 5 times the normal mutation
/// strength, stay valid for 80 generations: `Experiment::validate` passes after
/// each one.
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
    for _ in 0..80 {
        e.run_generation(&mut by_slot).unwrap();
        e.validate().unwrap();
    }
}

/// A muscle's target length starts at `long` and falls to `short` at the end of
/// the duty share of the period. Here the period is 2 s and the duty is 0.4, so
/// the fall ends at 0.8 s. The length then rises again. It has no jump where
/// the fall turns into the rise, and it repeats every period.
#[test]
fn muscle_cycle_is_continuous_and_periodic() {
    let m = Muscle {
        bone_a: 0,
        bone_b: 1,
        anchor_a: 1.0,
        anchor_b: 0.0,
        short: 0.1,
        long: 0.3,
        period: 2.,
        phase: 0.,
        duty: 0.4,
        stiffness: 30.,
        sensor: 255,
        reset: 0.0,
        tendon: 0.0,
    };
    assert!((physics::target(&m, 0.) - 0.3).abs() < 1e-6);
    assert!((physics::target(&m, 0.8) - 0.1).abs() < 1e-6);
    assert!((physics::target(&m, 0.79999) - physics::target(&m, 0.80001)).abs() < 1e-5);
    assert!((physics::target(&m, 0.23) - physics::target(&m, 2.23)).abs() < 1e-6);
}

#[test]
fn a_checkpoint_before_the_first_archive_starts_the_same_game() {
    // A save keeps the archives and the search state, not the ring. A game
    // saved before it has any elite starts again from the same random bodies.
    let e = Experiment::new(config()).unwrap();
    let checkpoint = path("partial");
    storage::save(&checkpoint, &e).unwrap();
    let loaded = storage::load(&checkpoint).unwrap();
    assert_eq!(loaded.evaluated, 0);
    for (x, y) in e.blocks.iter().zip(&loaded.blocks) {
        assert_eq!(x.population.nodes, y.population.nodes);
        assert_eq!(x.population.muscles, y.population.muscles);
    }
    // A leftover temporary file, like one from an interrupted save, does not
    // change what `load` reads.
    std::fs::write(checkpoint.with_extension("evo.tmp"), b"partial").unwrap();
    assert!(storage::load(&checkpoint).is_ok());
    let _ = std::fs::remove_file(checkpoint.with_extension("evo.tmp"));
    let _ = std::fs::remove_file(checkpoint);
}

/// An odd population, a gravity that is not a number and a file that is not a
/// save are all rejected.
#[test]
fn invalid_settings_and_checkpoints_are_rejected() {
    let mut cfg = config();
    cfg.population = 3;
    assert!(cfg.validate().is_err());
    cfg = config();
    cfg.gravity = f32::NAN;
    assert!(cfg.validate().is_err());
    let p = path("corrupt");
    std::fs::write(&p, b"garbage!").unwrap();
    assert!(storage::load(&p).is_err());
    let _ = std::fs::remove_file(p);
}

/// The statistics row of the first generation adds up: its histogram bins and
/// its body types each count `archive_cells` elites. It has one value for each
/// of the 29 percentiles, and best, median and worst are in order.
#[test]
fn statistics_count_every_archive_elite() {
    let mut e = Experiment::new(config()).unwrap();
    e.run_generation(&mut by_slot).unwrap();
    let s = &e.history[0];
    assert!(s.archive_cells > 0);
    assert_eq!(
        s.histogram.iter().map(|x| x.1 as usize).sum::<usize>(),
        s.archive_cells
    );
    assert_eq!(
        s.species.iter().map(|x| x.2 as usize).sum::<usize>(),
        s.archive_cells
    );
    assert_eq!(s.percentiles.len(), 29);
    assert!(s.best >= s.median && s.median >= s.worst);
}

/// A save keeps the history of the generations run. `load` turns down a save
/// with a flipped bit, which the checksum of the compressed data catches, and a
/// save with an invalid history row.
#[test]
fn history_and_checksums_are_validated_on_load() {
    let mut e = Experiment::new(config()).unwrap();
    e.run_generation(&mut by_slot).unwrap();
    let checkpoint = path("history");
    storage::save(&checkpoint, &e).unwrap();
    assert_eq!(storage::load(&checkpoint).unwrap().history.len(), 1);
    // The last byte of the file belongs to the checksum that ends the
    // compressed data.
    let mut bytes = std::fs::read(&checkpoint).unwrap();
    let end = bytes.len() - 1;
    bytes[end] ^= 1;
    std::fs::write(&checkpoint, bytes).unwrap();
    assert!(storage::load(&checkpoint).is_err());
    // A history row with no percentiles is invalid.
    e.history[0].percentiles.clear();
    storage::save(&checkpoint, &e).unwrap();
    assert!(storage::load(&checkpoint).is_err());
    let _ = std::fs::remove_file(checkpoint);
}

/// A world change clears the global archive and queues the elites of the main
/// islands to be evaluated again. Within one lap of the ring each elite the
/// global archive held is back in a block.
#[test]
fn a_world_change_retests_archive_elites() {
    let mut e = Experiment::new(config()).unwrap();
    let mut evaluate = by_slot;
    e.run_generation(&mut evaluate).unwrap();
    let elites: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    assert!(!elites.is_empty());
    // Rougher ground is a change of the world.
    e.update_config_now(Config {
        terrain: 1,
        ..e.config.clone()
    })
    .unwrap();
    // One lap of the ring breeds every block again. Each block places the
    // queued elites before it breeds children.
    let mut handed_over = Vec::new();
    for _ in 0..e.blocks.len() {
        e.step(&mut evaluate).unwrap();
        let k = (e.cursor + e.blocks.len() - 1) % e.blocks.len();
        handed_over.extend(e.blocks[k].population.genomes.iter().map(|g| g.id));
    }
    for id in elites {
        assert!(
            handed_over.contains(&id),
            "elite {id} was dropped by the world change"
        );
    }
}

/// `rotate_autosaves` keeps the newest autosaves and removes the rest. It
/// leaves a save the player named and a temporary file that is less than ten
/// minutes old.
#[test]
fn autosave_rotation_keeps_the_newest_and_spares_manual_saves() {
    let dir = std::env::temp_dir().join(format!("evolution-rotate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Five autosaves. The pauses give them different modification times, so
    // seed 0 is the oldest and seed 4 the newest.
    for seed in 0..5 {
        std::fs::write(dir.join(format!("seed-{seed}-auto.evo")), b"x").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::fs::write(dir.join("my-champion.evo"), b"x").unwrap();
    std::fs::write(dir.join("seed-9-auto.evo.tmp"), b"x").unwrap();
    // Keeping 3 of the 5 autosaves removes 2 files.
    assert_eq!(storage::rotate_autosaves(&dir, 3), 2);
    let mut left: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    // The fresh `.tmp` file may belong to a save in progress, so it stays.
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

/// An elite keeps the score the evaluator gave it in the standard trial. The
/// archive does not rescore it, and a confirmation trial that scores higher
/// changes nothing.
#[test]
fn archive_keeps_the_engine_score_without_rescoring() {
    let mut e = Experiment::new(config()).unwrap();
    // The evaluator's score is final, and archive insertion must not replace
    // it. Here a standard trial adds 100 to the score and a confirmation trial
    // adds 1000. A creature that gets a confirmation trial takes the lower of
    // the two scores, so the standard score stands.
    let mut expected = std::collections::HashMap::new();
    // A step absorbs one block of the ring. Step until an elite arrives, for at
    // most 16 steps.
    for _ in 0..16 {
        e.step(&mut |pop, cfg| {
            let mut metrics = by_slot(pop, cfg)?;
            // Only a confirmation trial sets the fidelity.
            let confirm = cfg.fidelity.is_some();
            for (g, m) in pop.genomes.iter().zip(&mut metrics) {
                if confirm {
                    m.fitness += 1000.0;
                } else {
                    m.fitness += 100.0;
                    expected.insert(g.id, m.fitness);
                }
            }
            Ok(metrics)
        })
        .unwrap();
        if !e.archive.entries.is_empty() {
            break;
        }
    }
    assert!(!e.archive.entries.is_empty());
    for elite in &e.archive.entries {
        assert_eq!(elite.fitness, expected[&elite.creature.id]);
    }
}

/// A meteor strike removes elites from the archives and keeps them as fossils.
/// `undo_meteor` puts every one back.
#[test]
fn meteor_strike_can_be_undone() {
    let mut e = Experiment::new(config()).unwrap();
    e.run_generation(&mut by_slot).unwrap();
    // The elites in the global archive, the islands and their nurseries.
    let count = |e: &Experiment| {
        e.archive.entries.len() + e.islands.iter().map(|i| i.entries.len()).sum::<usize>()
    };
    let before = count(&e);
    let mut ids: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    ids.sort_unstable();
    // The strike spares the fastest elite of each of an archive's fastest body
    // plans, and a tiny archive has few plans, so it may lose nothing.
    let lost = e.meteor(0.5);
    assert_eq!(count(&e), before - lost);
    assert_eq!(e.undo_meteor(), lost);
    assert_eq!(count(&e), before);
    assert!(e.fossils.is_empty());
    let mut back: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    back.sort_unstable();
    assert_eq!(back, ids);
}
