use evolution_simulator::{
    config::Config,
    evolution::{self, Creature, Muscle},
    physics,
    qd::{Elite, Emitter, QdArchive},
    storage::{self, Experiment},
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
/// Made-up standard results: the score grows with the birth slot.
fn by_slot(
    pop: &evolution::Population,
    _: &Config,
) -> anyhow::Result<Vec<evolution_simulator::qd::EvaluationMetrics>> {
    Ok(pop
        .genomes
        .iter()
        .map(|g| evolution_simulator::qd::EvaluationMetrics {
            fitness: evolution::slot_of_id(g.id) as f32 * 0.1,
            ..Default::default()
        })
        .collect())
}
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
fn assert_genomes_close(a: &Creature, b: &Creature) {
    let close = |x: f32, y: f32| (x - y).abs() <= 1e-4;
    assert_eq!(a.nodes.len(), b.nodes.len());
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert!((x.x - y.x).abs() <= 1e-4 && (x.y - y.y).abs() <= 1e-4);
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
/// Breeds eight zero-mutation CMA children of `parent` (the only elite).
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
    // A new random body may carry muscles that no repair has put on the
    // body's clock yet, and breeding repairs every child. So the first
    // generation of children may differ from the random body in those
    // periods, and only a repaired body is copied exactly.
    let repaired = zero_mutation_children(&cfg, &population.creature(0)).swap_remove(0);
    for child in zero_mutation_children(&cfg, &repaired) {
        assert_genomes_close(&child, &repaired);
    }
}
/// A block bred into a reused arena, into one whose parts are too small (so
/// children go after the parts), and into a new one holds the same creatures.
#[test]
fn breeding_into_a_reused_arena_gives_the_same_creatures() {
    let cfg = Config {
        population: 64,
        ..config()
    };
    let parents = evolution::create(&cfg).unwrap();
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
    let archive = [archive];
    let count = 9_000;
    let emitters = [
        Emitter::Structural,
        Emitter::Novelty,
        Emitter::Restart,
        Emitter::Cma,
    ];
    // Position 5 holds a reseeded elite; the rest are bred.
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
    // Reused: the arena held another block of this size.
    let mut reused = evolution::Population::default();
    breed(&mut reused, 1);
    breed(&mut reused, 2);
    // Too small: the last block's creatures had no genes.
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
    // Saves keep the archives and search state, not the ring: a game saved
    // before its first archive starts again from the same random bodies.
    let e = Experiment::new(config()).unwrap();
    let checkpoint = path("partial");
    storage::save(&checkpoint, &e).unwrap();
    let loaded = storage::load(&checkpoint).unwrap();
    assert_eq!(loaded.evaluated, 0);
    for (x, y) in e.blocks.iter().zip(&loaded.blocks) {
        assert_eq!(x.population.nodes, y.population.nodes);
        assert_eq!(x.population.muscles, y.population.muscles);
    }
    // A stale/incomplete temporary write cannot corrupt the committed checkpoint.
    std::fs::write(checkpoint.with_extension("evo.tmp"), b"partial").unwrap();
    assert!(storage::load(&checkpoint).is_ok());
    let _ = std::fs::remove_file(checkpoint.with_extension("evo.tmp"));
    let _ = std::fs::remove_file(checkpoint);
}
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

#[test]
fn history_and_checksums_are_validated_on_load() {
    let mut e = Experiment::new(config()).unwrap();
    e.run_generation(&mut by_slot).unwrap();
    let checkpoint = path("history");
    storage::save(&checkpoint, &e).unwrap();
    assert_eq!(storage::load(&checkpoint).unwrap().history.len(), 1);
    let mut bytes = std::fs::read(&checkpoint).unwrap();
    let end = bytes.len() - 1;
    bytes[end] ^= 1;
    std::fs::write(&checkpoint, bytes).unwrap();
    assert!(storage::load(&checkpoint).is_err());
    e.history[0].percentiles.clear();
    storage::save(&checkpoint, &e).unwrap();
    assert!(storage::load(&checkpoint).is_err());
    let _ = std::fs::remove_file(checkpoint);
}

#[test]
fn a_world_change_retests_archive_elites() {
    let mut e = Experiment::new(config()).unwrap();
    let mut evaluate = by_slot;
    e.run_generation(&mut evaluate).unwrap();
    let elites: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    assert!(!elites.is_empty());
    e.update_config_now(Config {
        terrain: 1,
        ..e.config.clone()
    })
    .unwrap();
    // One lap of the ring breeds every block again, the queued elites first.
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
fn archive_keeps_the_engine_score_without_rescoring() {
    let mut e = Experiment::new(config()).unwrap();
    // Treat this evaluation engine's output as authoritative. Archive
    // insertion must not replace its score. The confirmation trials score
    // higher, so the standard score stands.
    let mut expected = std::collections::HashMap::new();
    // A step absorbs one block of the ring; step until elites arrive, at most
    // one ring.
    for _ in 0..16 {
        e.step(&mut |pop, cfg| {
            let mut metrics = by_slot(pop, cfg)?;
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

#[test]
fn meteor_strike_can_be_undone() {
    let mut e = Experiment::new(config()).unwrap();
    e.run_generation(&mut by_slot).unwrap();
    let count = |e: &Experiment| {
        e.archive.entries.len() + e.islands.iter().map(|i| i.entries.len()).sum::<usize>()
    };
    let before = count(&e);
    let mut ids: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    ids.sort_unstable();
    // The strike spares the fastest elite of each body plan, so a tiny
    // archive may lose nothing.
    let lost = e.meteor(0.5);
    assert_eq!(count(&e), before - lost);
    assert_eq!(e.undo_meteor(), lost);
    assert_eq!(count(&e), before);
    assert!(e.fossils.is_empty());
    let mut back: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
    back.sort_unstable();
    assert_eq!(back, ids);
}
