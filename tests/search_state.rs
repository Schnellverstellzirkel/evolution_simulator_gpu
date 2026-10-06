use evolution_simulator::{
    config::Config,
    evolution::{self, Population},
    qd::{Descriptor, Elite, Emitter, EvaluationMetrics, Niche, QdArchive, TrialMetrics},
    storage::{self, Experiment},
};
use serde::Serialize;
use std::{collections::BTreeMap, path::PathBuf};

fn config(seed: u64) -> Config {
    Config {
        population: 128,
        seed,
        random_seed: false,
        max_nodes: 8,
        max_muscles: 12,
        ..Config::default()
    }
}

#[test]
fn configuration_defaults_and_float_boundaries_are_validated() {
    Config::default().validate().unwrap();
    type FloatCase = (&'static str, fn(&mut Config) -> &mut f32, f32, f32);
    let fields: [FloatCase; 13] = [
        ("duration", |c| &mut c.duration, 0.1, 300.0),
        ("mutation", |c| &mut c.mutation, 0.0, 10.0),
        ("gravity", |c| &mut c.gravity, 0.0, 100.0),
        ("air retention", |c| &mut c.air_retention, 0.0, 1.02),
        ("ground friction", |c| &mut c.ground_friction, 0.0, 20.0),
        ("muscle energy", |c| &mut c.muscle_energy, 0.05, 2.0),
        ("muscle recovery", |c| &mut c.muscle_recovery, 0.05, 2.0),
        ("slope", |c| &mut c.slope, -0.6, 0.6),
        ("wind", |c| &mut c.wind, -20.0, 20.0),
        ("minimum diameter", |c| &mut c.min_size, 0.01, 1.0),
        ("maximum diameter", |c| &mut c.max_size, 0.01, 1.0),
        ("minimum friction", |c| &mut c.min_friction, 0.0, 1.0),
        ("maximum friction", |c| &mut c.max_friction, 0.0, 1.0),
    ];
    let base = Config {
        min_size: 0.01,
        max_size: 1.0,
        min_friction: 0.0,
        max_friction: 1.0,
        ..config(38)
    };
    for (name, field, minimum, maximum) in fields {
        for value in [minimum, maximum] {
            let mut cfg = base.clone();
            *field(&mut cfg) = value;
            assert!(cfg.validate().is_ok(), "rejected {name} = {value}");
        }
        for value in [
            minimum.next_down(),
            maximum.next_up(),
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ] {
            let mut cfg = base.clone();
            *field(&mut cfg) = value;
            assert!(cfg.validate().is_err(), "accepted {name} = {value}");
        }
    }
}

#[test]
fn configuration_integer_limits_and_ordered_bounds_are_validated() {
    type IntegerCase = (&'static str, fn(&mut Config) -> &mut usize, usize, usize);
    let fields: [IntegerCase; 5] = [
        ("population", |c| &mut c.population, 2, 20_000_000),
        ("nodes", |c| &mut c.max_nodes, 3, evolution::MAX_NODES),
        ("muscles", |c| &mut c.max_muscles, 3, evolution::MAX_MUSCLES),
        ("GPU budget", |c| &mut c.gpu_budget_mib, 32, 6144),
        ("RAM budget", |c| &mut c.ram_budget_mib, 64, 24576),
    ];
    let base = Config {
        population: 2,
        max_nodes: 3,
        max_muscles: evolution::MAX_MUSCLES,
        ram_budget_mib: 24576,
        ..config(38)
    };
    for (name, field, minimum, maximum) in fields {
        for value in [minimum, maximum] {
            let mut cfg = base.clone();
            *field(&mut cfg) = value;
            assert!(cfg.validate().is_ok(), "rejected {name} = {value}");
        }
        for value in [minimum - 1, maximum + 1, usize::MAX] {
            let mut cfg = base.clone();
            *field(&mut cfg) = value;
            assert!(cfg.validate().is_err(), "accepted {name} = {value}");
        }
    }
    let invalid = [
        Config {
            population: 3,
            ..base.clone()
        },
        Config {
            min_size: 0.2,
            max_size: 0.1,
            ..base.clone()
        },
        Config {
            min_friction: 0.8,
            max_friction: 0.7,
            ..base.clone()
        },
        Config {
            max_nodes: 8,
            max_muscles: 7,
            ..base.clone()
        },
        Config {
            terrain: u8::MAX,
            ..base.clone()
        },
    ];
    for cfg in invalid {
        assert!(cfg.validate().is_err(), "accepted invalid config: {cfg:?}");
    }
    for terrain in 0..evolution_simulator::physics::TERRAIN_AMPLITUDES.len() {
        Config {
            terrain: terrain as u8,
            ..base.clone()
        }
        .validate()
        .unwrap();
    }
}

#[test]
fn configuration_rejects_population_above_the_ram_budget() {
    let mut cfg = Config {
        ram_budget_mib: 64,
        ..config(38)
    };
    // Use neighboring even population sizes on either side of the RAM limit.
    cfg.population = (cfg.ram_budget_mib * 1024 * 1024 / 1200) & !1;
    cfg.validate().unwrap();
    cfg.population += 2;
    assert!(cfg.validate().is_err());
}

#[test]
fn behavior_archive_keeps_exactly_the_fastest_creature_in_each_cell() {
    let population = evolution::create(&config(38)).unwrap();
    let descriptors: Vec<_> = (0..6)
        .map(|cell| Descriptor {
            ground_contact: cell as f32 / 5.0,
            gait_frequency: 1.0,
            mean_height: 0.5,
            feet: 2.0,
            ..Descriptor::default()
        })
        .collect();
    let mut archive = QdArchive::default();
    let mut expected = BTreeMap::<Niche, (f32, u64)>::new();
    for (round, score) in [-5.0, 4.0, 4.0, 3.0, 9.0, -1.0, 12.0]
        .into_iter()
        .enumerate()
    {
        for (cell, &descriptor) in descriptors.iter().enumerate() {
            let index = round * descriptors.len() + cell;
            let fitness = score + cell as f32;
            let niche = descriptor.niche_in(&evolution_simulator::qd::ISLAND_CLASSES);
            let old = expected.get(&niche).copied();
            let should_insert = old.is_none_or(|(best, _)| fitness > best);
            let offer = archive.offer(
                &population,
                index,
                descriptor,
                fitness,
                false,
                Emitter::Structural,
                round as u32,
                0,
            );
            assert_eq!(offer.inserted, should_insert);
            assert_eq!(offer.new_niche, old.is_none());
            if should_insert {
                expected.insert(niche, (fitness, population.genomes[index].id));
            }
            assert_eq!(archive.entries.len(), expected.len());
            assert_eq!(archive.behavior_count(), expected.len());
            let actual: BTreeMap<_, _> = archive
                .entries
                .iter()
                .map(|elite| (elite.niche.clone(), (elite.fitness, elite.creature.id)))
                .collect();
            assert_eq!(
                actual.len(),
                archive.entries.len(),
                "duplicate archive cell"
            );
            assert_eq!(actual, expected);
            assert_eq!(
                archive.qd_score,
                expected
                    .values()
                    .map(|(score, _)| score.max(0.0) as f64)
                    .sum::<f64>()
            );
        }
        // Loading a checkpoint rebuilds these indices; the same insertion rule
        // must continue to hold afterward.
        archive.rebuild_indices();
    }
    let before = encoded(&archive);
    for fitness in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        evolution::FAILED,
    ] {
        let offer = archive.offer(
            &population,
            0,
            descriptors[0],
            fitness,
            false,
            Emitter::Restart,
            100,
            0,
        );
        assert!(!offer.inserted);
        assert_eq!(encoded(&archive), before);
    }
}

#[test]
fn bodies_of_other_shapes_and_sizes_keep_cells_of_their_own() {
    let population = evolution::create(&config(38)).unwrap();
    // One way of moving, offered by bodies of every shape and size.
    let way = |nodes: u16, aspect_ratio: f32| Descriptor {
        ground_contact: 0.5,
        gait_frequency: 1.0,
        mean_height: 0.5,
        feet: 2.0,
        nodes,
        aspect_ratio,
        ..Descriptor::default()
    };
    let bodies = [
        (5, 0.6),
        (5, 1.8),
        (5, 4.0),
        (10, 0.6),
        (10, 1.8),
        (10, 4.0),
        (20, 0.6),
        (20, 1.8),
        (20, 4.0),
    ];
    let classes: std::collections::BTreeSet<Niche> = bodies
        .iter()
        .map(|&(nodes, aspect)| {
            way(nodes, aspect).niche_in(&evolution_simulator::qd::ISLAND_CLASSES)
        })
        .collect();
    assert_eq!(
        classes.len(),
        evolution_simulator::qd::ISLAND_CLASSES.classes()
    );
    let mut archive = QdArchive::default();
    archive.set_refined(true);
    for (round, fitness) in [3.0, 1.0, 5.0].into_iter().enumerate() {
        for (k, &(nodes, aspect)) in bodies.iter().enumerate() {
            archive.offer(
                &population,
                round * bodies.len() + k,
                way(nodes, aspect),
                fitness,
                false,
                Emitter::Structural,
                round as u32,
                0,
            );
        }
    }
    // Every body class shares the way of moving, and each keeps its fastest.
    assert_eq!(archive.behavior_count(), classes.len());
    assert_eq!(archive.movement_count(), 1);
    // The statistics of distance read one elite for the way of moving.
    assert_eq!(archive.best_per_way_of_moving().len(), 1);
    assert!(archive.entries.iter().all(|elite| elite.fitness == 5.0));
    // Bodies of one class compete for its cell, whatever their exact size.
    let before = archive.entries.len();
    let rival = archive.offer(&population, 0, way(6, 0.5), 9.0, false, Emitter::Cma, 3, 0);
    assert!(rival.inserted && !rival.new_niche);
    assert_eq!(archive.entries.len(), before);
}

#[test]
fn an_archive_keeps_one_elite_per_way_of_moving_until_it_is_refined() {
    let population = evolution::create(&config(38)).unwrap();
    let way = |nodes: u16, aspect_ratio: f32| Descriptor {
        ground_contact: 0.5,
        gait_frequency: 1.0,
        mean_height: 0.5,
        feet: 2.0,
        nodes,
        aspect_ratio,
        ..Descriptor::default()
    };
    // A new archive, such as a nursery or one that is still climbing.
    let mut climbing = QdArchive::default();
    for (k, &(nodes, aspect)) in [(5, 0.6), (10, 1.8), (20, 4.0)].iter().enumerate() {
        climbing.offer(
            &population,
            k,
            way(nodes, aspect),
            1.0 + k as f32,
            false,
            Emitter::Restart,
            0,
            0,
        );
    }
    // The three bodies share one way of moving, and the fastest keeps it.
    assert_eq!(climbing.behavior_count(), 1);
    assert_eq!(climbing.entries[0].fitness, 3.0);
    // Refined, the same elites take the cells of their body classes.
    let mut refined = QdArchive::default();
    refined.set_refined(true);
    let slower = Elite {
        fitness: 1.0,
        descriptor: way(5, 0.6),
        ..climbing.entries[0].clone()
    };
    assert!(refined.absorb(&slower));
    assert!(refined.absorb(&climbing.entries[0]));
    assert_eq!(refined.behavior_count(), 2);
    assert_eq!(refined.movement_count(), 1);
}

#[test]
fn island_migration_never_duplicates_a_cell_or_replaces_a_faster_elite() {
    let population = evolution::create(&config(38)).unwrap();
    let mut source = QdArchive::default();
    source.offer(
        &population,
        0,
        Descriptor::default(),
        5.0,
        false,
        Emitter::Restart,
        0,
        0,
    );
    let mut target = QdArchive::default();
    assert!(target.absorb(&source.entries[0]));
    target.visit(0);
    for (fitness, accepted) in [(3.0, false), (5.0, false), (8.0, true), (7.0, false)] {
        let mut migrant = source.entries[0].clone();
        migrant.fitness = fitness;
        migrant.creature.id += 1;
        let previous = target.entries[0].clone();
        assert_eq!(target.absorb(&migrant), accepted);
        assert_eq!(target.entries.len(), 1);
        assert_eq!(target.entries[0].visits, 1);
        assert_eq!(target.entries[0].fitness, previous.fitness.max(fitness));
        if !accepted {
            assert_eq!(target.entries[0].creature.id, previous.creature.id);
        }
        assert_eq!(target.qd_score, target.entries[0].fitness as f64);
    }
}

// Synthetic results isolate search state from the physics engine and make these
// regression tests cheap. The score follows the generation and the birth
// slot, and every island receives several distinct cadence cells.
fn synthetic(
    generation: u32,
) -> impl FnMut(&Population, &Config) -> anyhow::Result<Vec<EvaluationMetrics>> {
    move |pop, _| {
        Ok(pop
            .genomes
            .iter()
            .map(|g| {
                let i = evolution::slot_of_id(g.id);
                EvaluationMetrics {
                    fitness: 10.0 + generation as f32 + i as f32,
                    behavior: TrialMetrics {
                        ground_contact: 0.5,
                        gait_frequency: ((i / storage::island_count()) % 8) as f32 * 0.75 + 0.1,
                        mean_height: 0.5,
                        feet: 2.0,
                        ..TrialMetrics::default()
                    },
                    ..EvaluationMetrics::default()
                }
            })
            .collect())
    }
}

/// Runs one generation of the ring on synthetic results.
fn run_synthetic(experiment: &mut Experiment) {
    let generation = experiment.generation;
    experiment
        .run_generation(&mut synthetic(generation))
        .unwrap();
}

fn encoded<T: Serialize>(value: &T) -> Vec<u8> {
    bincode::serialize(value).unwrap()
}

fn assert_same_population(a: &Population, b: &Population) {
    assert_eq!(a.genomes.len(), b.genomes.len());
    for index in 0..a.genomes.len() {
        let a = a.creature(index);
        let b = b.creature(index);
        assert_eq!(a.id, b.id, "creature {index} identity");
        assert_eq!(a.nodes, b.nodes, "creature {index} nodes");
        assert_eq!(a.bones, b.bones, "creature {index} bones");
        assert_eq!(a.muscles, b.muscles, "creature {index} muscles");
    }
}

fn assert_same_archive(a: &QdArchive, b: &QdArchive) {
    assert!(
        encoded(&a.entries) == encoded(&b.entries),
        "archive elites differ"
    );
    // Rebuilding an empty archive sums no scores, producing -0.0 rather than
    // Default's +0.0. They represent the same score and search state.
    assert_eq!(a.qd_score, b.qd_score);
    assert_eq!(a.behavior_count(), b.behavior_count());
    assert_eq!(a.morphology_count(), b.morphology_count());
}

/// The same ring, archives and search state.
fn assert_same_state(a: &Experiment, b: &Experiment) {
    a.validate().unwrap();
    b.validate().unwrap();
    assert_eq!(a.blocks.len(), b.blocks.len());
    for (x, y) in a.blocks.iter().zip(&b.blocks) {
        assert_eq!(x.first, y.first);
        assert_same_population(&x.population, &y.population);
        assert_eq!(*x.config, *y.config);
        for (p, q) in x.births.iter().zip(&y.births) {
            assert_eq!(
                (p.emitter, p.cma, p.parent_id, p.mate, p.protection),
                (q.emitter, q.cma, q.parent_id, q.mate, q.protection)
            );
        }
    }
    assert_eq!(a.config, b.config);
    assert_eq!(a.generation, b.generation);
    assert_eq!(a.breed_round, b.breed_round);
    assert_eq!(a.evaluated, b.evaluated);
    assert_eq!(a.cursor, b.cursor);
    assert!(
        encoded(&a.cma_emitters) == encoded(&b.cma_emitters),
        "CMA state differs"
    );
    assert_same_archive(&a.archive, &b.archive);
    assert_eq!(a.islands.len(), b.islands.len());
    for (a, b) in a.islands.iter().zip(&b.islands) {
        assert_same_archive(a, b);
    }
}

fn births(e: &Experiment) -> impl Iterator<Item = &storage::Birth> {
    e.blocks.iter().flat_map(|b| &b.births)
}

#[test]
fn the_ring_breeds_every_emitter_and_stays_valid() {
    for seed in [7, 38, 91] {
        let mut experiment = Experiment::new(config(seed)).unwrap();
        assert_eq!(experiment.ring_len(), 128);
        for _ in 0..4 {
            run_synthetic(&mut experiment);
            experiment.validate().unwrap();
        }
        assert!(!experiment.cma_emitters.is_empty(), "CMA must be exercised");
        let emitters: Vec<Emitter> = births(&experiment).map(|b| b.emitter).collect();
        assert!(emitters.contains(&Emitter::Cma));
        assert!(emitters.contains(&Emitter::Structural));
        assert!(emitters.contains(&Emitter::Novelty));
        assert_eq!(experiment.history.len(), 4);
    }
}

#[test]
fn the_ring_is_smaller_than_a_large_generation() {
    let ring = storage::RingShape {
        block: 4096,
        blocks: 3,
    };
    let experiment = Experiment::with_ring(
        Config {
            population: 3 * 4096 + 1000,
            ..config(38)
        },
        ring,
    )
    .unwrap();
    assert_eq!(experiment.ring_len(), 3 * 4096);
    assert_eq!(experiment.blocks.len(), 3);
    assert!(experiment.blocks.iter().all(|b| b.len() == 4096));
    experiment.validate().unwrap();
}

#[test]
fn a_saved_game_keeps_its_ring_and_history_records_it() {
    let ring = storage::RingShape {
        block: 32,
        blocks: 3,
    };
    let mut experiment = Experiment::with_ring(config(38), ring).unwrap();
    assert_eq!(experiment.blocks.len(), 3);
    run_synthetic(&mut experiment);
    assert!(experiment.history.iter().all(|s| s.ring == ring));
    let path = std::env::temp_dir().join(format!("ring-shape-{}.evo", std::process::id()));
    storage::save(&path, &experiment).unwrap();
    let loaded = storage::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded.ring, ring);
    assert_eq!(loaded.blocks.len(), 3);
    assert_eq!(loaded.ring_len(), 96);
}

#[test]
fn the_island_reserves_breed_and_count_their_visits() {
    let mut experiment = Experiment::new(Config {
        population: 512,
        ..config(38)
    })
    .unwrap();
    for _ in 0..12 {
        run_synthetic(&mut experiment);
    }
    let reserve: Vec<_> = experiment
        .islands
        .iter()
        .flat_map(|island| &island.entries)
        .filter(|elite| evolution_simulator::qd::is_morphology_niche(&elite.niche))
        .collect();
    // The global archive holds behavior elites only.
    assert_eq!(experiment.archive.morphology_count(), 0);
    assert!(
        !reserve.is_empty(),
        "the synthetic run must fill the reserve"
    );
    assert!(
        reserve.iter().any(|elite| elite.visits > 0),
        "reserve entries never became parents"
    );
}

#[test]
fn a_record_is_confirmed_and_keeps_the_lower_score() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    // The confirmation trial finds half the standard distance.
    let mut confirmed = 0;
    experiment
        .step(&mut |pop, cfg| {
            let mut metrics = synthetic(0)(pop, cfg)?;
            if cfg.fidelity.is_some() {
                confirmed += metrics.len();
                for m in &mut metrics {
                    m.fitness *= 0.5;
                }
            }
            Ok(metrics)
        })
        .unwrap();
    assert!(confirmed > 0, "the first block sets records");
    // Every island record is a confirmed score: it lies below any
    // unconfirmed standard score of its island.
    for island in &experiment.islands[..storage::island_count()] {
        let Some(best) = island
            .entries
            .iter()
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        else {
            continue;
        };
        assert!(best.fine, "the record holder shows its confirmation trial");
        for elite in island.entries.iter().filter(|e| !e.fine) {
            assert!(elite.fitness <= best.fitness);
        }
    }
}

struct Checkpoint(PathBuf);

impl Checkpoint {
    fn new(name: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "evolution-{name}-{}-{nonce}.evo",
            std::process::id()
        )))
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_file(self.0.with_extension("evo.tmp"));
    }
}

#[test]
fn saving_twice_replaces_the_checkpoint_with_the_latest_state() {
    let checkpoint = Checkpoint::new("replace-save");
    let mut experiment = Experiment::new(config(38)).unwrap();
    storage::save(&checkpoint.0, &experiment).unwrap();
    let original = storage::load(&checkpoint.0).unwrap();
    assert_eq!(original.generation, 0);
    assert!(original.archive.entries.is_empty());

    run_synthetic(&mut experiment);
    storage::save(&checkpoint.0, &experiment).unwrap();
    let replaced = storage::load(&checkpoint.0).unwrap();
    assert_eq!(replaced.generation, 1);
    assert_eq!(encoded(&experiment.archive), encoded(&replaced.archive));
    assert!(!checkpoint.0.with_extension("evo.tmp").exists());
}

#[test]
fn a_loaded_save_breeds_one_ring_and_repeats_its_search() {
    let mut uninterrupted = Experiment::new(config(38)).unwrap();
    for _ in 0..3 {
        run_synthetic(&mut uninterrupted);
    }
    assert!(!uninterrupted.cma_emitters.is_empty());
    let checkpoint = Checkpoint::new("archive-resume");
    storage::save(&checkpoint.0, &uninterrupted).unwrap();
    let mut first = storage::load(&checkpoint.0).unwrap();
    let mut second = storage::load(&checkpoint.0).unwrap();
    assert_same_archive(&first.archive, &uninterrupted.archive);
    // Loading breeds every block of the ring from the saved archives.
    assert_eq!(
        first.breed_round,
        uninterrupted.breed_round + first.blocks.len() as u64
    );
    assert_same_state(&first, &second);
    run_synthetic(&mut first);
    run_synthetic(&mut second);
    assert_same_state(&first, &second);
}

#[test]
fn each_island_gives_a_fifth_of_its_slots_to_a_nursery_and_a_tenth_to_another() {
    use evolution_simulator::qd;
    let (islands, arenas) = (storage::island_count(), storage::arena_count());
    let mut slots = vec![0usize; arenas];
    // Whole cycles of every main and wild island.
    for slot in 0..10_000 * qd::SLOT_CYCLE {
        slots[qd::arena_of_slot(slot, arenas)] += 1;
    }
    for island in 0..islands {
        let rounds =
            slots[island] + slots[storage::nursery_of(island)] + slots[storage::reshaped_of(island)];
        assert_eq!(slots[island], rounds - 6 * rounds / qd::SLOT_CYCLE);
        assert_eq!(slots[storage::nursery_of(island)], 4 * rounds / qd::SLOT_CYCLE);
        assert_eq!(
            slots[storage::reshaped_of(island)],
            2 * rounds / qd::SLOT_CYCLE
        );
    }
    assert!(qd::is_reshaped_arena(storage::reshaped_of(0), arenas));
    assert!(!qd::is_reshaped_arena(storage::nursery_of(0), arenas));
}

#[test]
fn a_save_holds_the_islands_and_loads_with_empty_reshaped_nurseries() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    for _ in 0..3 {
        run_synthetic(&mut experiment);
    }
    let older = storage::island_count() * 2;
    assert!(
        experiment.islands[older..]
            .iter()
            .any(|nursery| nursery.behavior_count() > 0),
        "the reshaped nurseries hold bodies before the save"
    );
    let checkpoint = Checkpoint::new("reshaped-nurseries");
    storage::save(&checkpoint.0, &experiment).unwrap();
    let loaded = storage::load(&checkpoint.0).unwrap();
    assert_eq!(loaded.islands.len(), storage::arena_count());
    // Loading breeds the ring, which counts visits of the parents, so the
    // elites are compared by cell, creature and score.
    let held = |archive: &QdArchive| -> Vec<(evolution_simulator::qd::Niche, u64, u32)> {
        let mut held: Vec<_> = archive
            .entries
            .iter()
            .map(|e| (e.niche.clone(), e.creature.id, e.fitness.to_bits()))
            .collect();
        held.sort();
        held
    };
    for (arena, island) in loaded.islands.iter().enumerate() {
        if arena < older {
            assert_eq!(held(island), held(&experiment.islands[arena]));
        } else {
            assert!(island.entries.is_empty());
        }
    }
    loaded.validate().unwrap();
}

#[test]
fn checkpoint_preserves_stalled_island_optimizer() {
    let mut uninterrupted = Experiment::new(Config {
        population: 1024,
        ..config(38)
    })
    .unwrap();
    run_synthetic(&mut uninterrupted);
    run_synthetic(&mut uninterrupted);
    // Advance the record age past the 30-generation optimizer rotation without
    // spending 30 generations evaluating bodies. The archived scores and CMA
    // state remain valid; the loaded ring must use the same alternate design.
    uninterrupted.generation = 40;
    uninterrupted.history.clear();
    uninterrupted.island_progress = uninterrupted
        .islands
        .iter()
        .map(|island| (island.best_fitness(), 1))
        .collect();
    assert!(
        uninterrupted
            .islands
            .iter()
            .take(evolution_simulator::qd::MAIN_ISLANDS)
            .all(|island| island.behavior_count() > 1)
    );
    uninterrupted.validate().unwrap();
    let checkpoint = Checkpoint::new("stalled-island-resume");
    storage::save(&checkpoint.0, &uninterrupted).unwrap();
    let first = storage::load(&checkpoint.0).unwrap();
    let second = storage::load(&checkpoint.0).unwrap();
    assert_eq!(first.island_progress, second.island_progress);
    assert!(
        births(&first)
            .filter_map(|b| b.cma)
            .any(|index| first.cma_emitters[index].optimizing())
    );
    assert_same_state(&first, &second);
}

#[test]
fn checkpoint_rejects_invalid_optimizer_resume_metadata() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut experiment);
    let islands = experiment.islands.len();
    let checkpoint = Checkpoint::new("invalid-resume");
    for progress in [
        vec![(f32::NAN, 0); islands],
        vec![(f32::INFINITY, 0); islands],
        vec![(1.0, experiment.generation + 1); islands],
        vec![(1.0, 0); islands + 1],
        vec![(1.0, 0); 65],
    ] {
        experiment.island_progress = progress;
        storage::save(&checkpoint.0, &experiment).unwrap();
        assert!(storage::load(&checkpoint.0).is_err());
    }
    // No planning pass has run yet after loading a save or after an empty
    // island receives its first elite. Both forms are valid resume states.
    for progress in [Vec::new(), vec![(f32::NEG_INFINITY, 0); islands]] {
        experiment.island_progress = progress;
        storage::save(&checkpoint.0, &experiment).unwrap();
        let restored = storage::load(&checkpoint.0).unwrap();
        assert_eq!(restored.island_progress.len(), islands);
    }
}

#[test]
fn a_save_from_other_physics_is_turned_down() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    experiment.qd_version = evolution_simulator::qd::OLDEST_LOADABLE - 1;
    let checkpoint = Checkpoint::new("old-physics");
    storage::save(&checkpoint.0, &experiment).unwrap();
    let error = storage::load(&checkpoint.0).err().unwrap().to_string();
    assert!(error.contains("physics version"), "{error}");
}

#[test]
fn a_save_from_the_previous_cell_layout_loads_with_every_elite_in_its_new_cell() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    for _ in 0..3 {
        run_synthetic(&mut experiment);
    }
    // As the previous version saved them: the niche has no shape or size
    // class, and the global archive and every island hold one elite per cell.
    let old_layout = |archive: &mut QdArchive| {
        let mut seen = std::collections::HashSet::new();
        archive.entries.retain_mut(|elite| {
            if evolution_simulator::qd::is_morphology_niche(&elite.niche) {
                return true;
            }
            elite.niche.0[2] = 0;
            elite.niche.0[5] = 0;
            seen.insert(elite.niche.clone())
        });
        archive.rebuild_indices();
    };
    old_layout(&mut experiment.archive);
    for island in &mut experiment.islands {
        old_layout(island);
    }
    let elites = experiment.archive.entries.len();
    assert!(elites > 1);
    experiment.qd_version = evolution_simulator::qd::OLDEST_LOADABLE;
    let checkpoint = Checkpoint::new("previous-layout");
    storage::save(&checkpoint.0, &experiment).unwrap();
    assert!(storage::check(&checkpoint.0).is_ok());
    let restored = storage::load(&checkpoint.0).unwrap();
    assert_eq!(restored.qd_version, evolution_simulator::qd::VERSION);
    assert_eq!(restored.archive.entries.len(), elites);
    // The global archive was refined at the load: its elites are in cells of
    // the global classes.
    assert!(restored.archive.refined());
    for elite in &restored.archive.entries {
        if !evolution_simulator::qd::is_morphology_niche(&elite.niche) {
            assert_eq!(
                elite.niche,
                elite
                    .descriptor
                    .niche_in(&evolution_simulator::qd::GLOBAL_CLASSES)
            );
        }
    }
    // The islands are young, so each keeps one elite per way of moving.
    for archive in &restored.islands {
        for elite in &archive.entries {
            if !evolution_simulator::qd::is_morphology_niche(&elite.niche) && !archive.refined() {
                assert_eq!(elite.niche, elite.descriptor.movement_niche());
            }
        }
    }
    restored.validate().unwrap();
}

#[test]
fn a_save_keeps_the_ancestors_of_the_global_archive_and_every_elites_record() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    for _ in 0..6 {
        run_synthetic(&mut experiment);
    }
    let checkpoint = Checkpoint::new("lineage");
    storage::save(&checkpoint.0, &experiment).unwrap();
    let restored = storage::load(&checkpoint.0).unwrap();
    let mut chains = 0;
    // The save holds the islands and their nurseries of new bodies.
    let held = experiment.islands.iter().take(storage::island_count() * 2);
    let archives = std::iter::once(&experiment.archive).chain(held);
    for elite in archives.flat_map(|archive| &archive.entries) {
        // Every living elite has its record back, creature included.
        let record = restored.lineage.get(&elite.creature.id).unwrap();
        assert_eq!(record.creature.unpack().nodes, elite.creature.unpack().nodes);
        assert_eq!(record.creature.unpack().muscles, elite.creature.unpack().muscles);
        let before = experiment.lineage.get(&elite.creature.id).unwrap();
        assert_eq!(record.fitness, before.fitness);
        assert_eq!(record.rung, before.rung);
    }
    // The global archive's elites keep the whole chain of their ancestors.
    for elite in &experiment.archive.entries {
        let before = experiment.ancestry(elite.creature.id, usize::MAX);
        let after = restored.ancestry(elite.creature.id, usize::MAX);
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert_eq!(a.creature.unpack().nodes, b.creature.unpack().nodes);
            assert_eq!(a.change, b.change);
        }
        chains += before.len();
    }
    assert!(chains > experiment.archive.entries.len());
    assert!(restored.lineage.len() <= experiment.lineage.len());
}

/// An archive at its plateau: one elite for every way of moving, all at the
/// same distance, with bodies of every shape and size. Each keeps the cell of
/// the layout before the body classes.
fn plateau_archive() -> QdArchive {
    archive_of_all_ways_of_moving(|_| 1000.0)
}

fn archive_of_all_ways_of_moving(fitness: impl Fn(u32) -> f32) -> QdArchive {
    let mut heights = [0.0f32; 6];
    let mut h = 0.15f32;
    while h < 4.0 {
        let bin = Descriptor {
            mean_height: h,
            ..Descriptor::default()
        }
        .niche()
        .0[3] as usize;
        if heights[bin] == 0.0 {
            heights[bin] = h;
        }
        h *= 1.02;
    }
    assert!(heights.iter().all(|&h| h > 0.0));
    let bodies = evolution::create(&Config {
        population: 1440,
        ..config(38)
    })
    .unwrap();
    let mut archive = QdArchive::default();
    let mut n = 0u32;
    for contact in 0..6u32 {
        for cadence in 0..8u32 {
            for &mean_height in &heights {
                for feet in 1..=5u32 {
                    n += 1;
                    let descriptor = Descriptor {
                        ground_contact: (contact as f32 + 0.5) / 6.0,
                        gait_frequency: (cadence as f32 + 0.5) * 0.75,
                        mean_height,
                        feet: feet as f32,
                        // Bodies of both shapes and sizes.
                        nodes: if n.is_multiple_of(2) { 6 } else { 16 },
                        aspect_ratio: if n.is_multiple_of(3) { 0.8 } else { 3.0 },
                        ..Descriptor::default()
                    };
                    let creature = bodies.creature(n as usize - 1);
                    archive.entries.push(Elite {
                        niche: descriptor.movement_niche(),
                        descriptor,
                        topology: evolution_simulator::qd::Topology::of(&creature),
                        creature: creature.into(),
                        fitness: fitness(n),
                        emitter: Emitter::Cma,
                        improved_generation: 0,
                        protected_until: 0,
                        visits: 0,
                        graduate: false,
                        fine: false,
                    });
                }
            }
        }
    }
    archive.rebuild_indices();
    archive
}

#[test]
fn an_island_is_refined_when_its_archive_is_old_enough() {
    use evolution_simulator::qd;
    let mut experiment = Experiment::new(config(38)).unwrap();
    // The global archive never breeds, so it starts refined; an island climbs
    // without classes.
    assert!(experiment.archive.refined());
    run_synthetic(&mut experiment);
    experiment.islands[0] = plateau_archive();
    let before = experiment.islands[0].behavior_count();
    while experiment.generation < qd::REFINE_AFTER - 1 {
        run_synthetic(&mut experiment);
    }
    for island in &experiment.islands[..storage::island_count()] {
        assert!(!island.refined());
    }
    run_synthetic(&mut experiment);
    assert_eq!(experiment.generation, qd::REFINE_AFTER);
    // Island 0, the hub and the wild islands moved to the cells of their
    // body classes (the other isolated islands follow 10 generations apart).
    // The elites of other shapes and sizes sit in cells of their own, and
    // none was lost.
    for (index, island) in experiment.islands[..storage::island_count()].iter().enumerate() {
        assert_eq!(
            island.refined(),
            index == 0 || index >= storage::ISOLATED_ISLANDS,
            "island {index}"
        );
    }
    // (A nursery's cohort may take cells of body classes that were empty.)
    assert!(experiment.islands[0].behavior_count() >= before);
    assert_eq!(experiment.islands[0].movement_count(), before);
    assert!(
        experiment.islands[0]
            .entries
            .iter()
            .any(|e| e.niche.0[2] != 0 || e.niche.0[5] != 0)
    );
    // The nurseries of new random bodies stay as they were.
    for island in 0..storage::island_count() {
        assert!(!experiment.islands[storage::nursery_of(island)].refined());
    }
    for elite in &experiment.islands[0].entries {
        if !qd::is_morphology_niche(&elite.niche) {
            assert_eq!(elite.niche, elite.descriptor.niche_in(&qd::ISLAND_CLASSES));
        }
    }
    experiment.validate().unwrap();
}

#[test]
fn a_version_54_save_loads_into_the_finer_global_classes() {
    use evolution_simulator::qd;
    let mut experiment = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut experiment);
    // Island 2 and the global archive were refined in the island layout.
    let mut refined = plateau_archive();
    // Bodies of in-between shapes and sizes, which the island classes lump.
    for (k, elite) in refined.entries.iter_mut().enumerate() {
        let (aspect_ratio, nodes) = [(0.7, 7), (1.0, 7), (0.7, 9), (0.7, 11)][k % 4];
        elite.descriptor.aspect_ratio = aspect_ratio;
        elite.descriptor.nodes = nodes;
    }
    refined.set_refined(true);
    refined.rebin();
    experiment.islands[2] = refined.clone();
    experiment.archive = refined;
    experiment.qd_version = 54;
    let checkpoint = Checkpoint::new("version-54");
    storage::save(&checkpoint.0, &experiment).unwrap();
    let restored = storage::load(&checkpoint.0).unwrap();
    // The island kept its refined layout, and another stayed as it was.
    assert!(restored.islands[2].refined() && !restored.islands[0].refined());
    assert_eq!(restored.islands[2].behavior_count(), 1440);
    for elite in &restored.islands[2].entries {
        if !qd::is_morphology_niche(&elite.niche) {
            assert_eq!(elite.niche, elite.descriptor.niche_in(&qd::ISLAND_CLASSES));
        }
    }
    // The global archive moved to the cells of its own, finer, classes.
    assert!(restored.archive.refined());
    assert_eq!(restored.archive.behavior_count(), 1440);
    let global_classes: std::collections::BTreeSet<_> = restored
        .archive
        .entries
        .iter()
        .map(|e| (e.niche.0[2], e.niche.0[5]))
        .collect();
    let island_classes: std::collections::BTreeSet<_> = restored.islands[2]
        .entries
        .iter()
        .map(|e| (e.niche.0[2], e.niche.0[5]))
        .collect();
    assert!(global_classes.len() > island_classes.len());
    for elite in &restored.archive.entries {
        if !qd::is_morphology_niche(&elite.niche) {
            assert_eq!(elite.niche, elite.descriptor.niche_in(&qd::GLOBAL_CLASSES));
        }
    }
    restored.validate().unwrap();
}

#[test]
fn a_save_of_an_older_version_leaves_every_island_coarse_until_it_is_old_enough() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut experiment);
    experiment.islands[2] = plateau_archive();
    experiment.qd_version = evolution_simulator::qd::OLDEST_LOADABLE;
    let checkpoint = Checkpoint::new("older-save");
    storage::save(&checkpoint.0, &experiment).unwrap();
    let mut restored = storage::load(&checkpoint.0).unwrap();
    // Every elite is in a way of moving of its own, and the global archive is
    // refined.
    assert!(!restored.islands[2].refined() && restored.archive.refined());
    assert_eq!(restored.islands[2].behavior_count(), 1440);
    // The game had run for 30 generations when it was saved, so its islands
    // are refined at the next generation boundary.
    // Island 2 refines 20 generations after island 0.
    restored.generation = evolution_simulator::qd::REFINE_AFTER + 20;
    restored.history.truncate(restored.generation as usize);
    run_synthetic(&mut restored);
    assert!(restored.islands[2].refined() && restored.islands[0].refined());
    assert_eq!(restored.islands[2].behavior_count(), 1440);
    assert!(
        restored.islands[2]
            .entries
            .iter()
            .any(|e| e.niche.0[2] != 0 || e.niche.0[5] != 0)
    );
}

#[test]
fn a_parent_of_a_rare_clade_is_preferred_when_the_elites_are_level() {
    use evolution_simulator::{evolution::Rng, qd};
    // Every elite has the same distance, so local competition cannot tell them
    // apart, and one elite's clade is rare.
    let archive = plateau_archive();
    let mut rarity = vec![0.0f32; archive.entries.len()];
    rarity[7] = 1.0;
    let draws = |rarity: &[f32]| {
        (0..4000)
            .filter(|&k| {
                let mut rng = Rng::new(38, 1, k);
                archive.sample_local_competitive(&mut rng, None, rarity) == Some(7)
            })
            .count()
    };
    // Eight elites meet in a tournament, so a given one is drawn about 8 in
    // 1,440 times at random and almost every time it meets the others when
    // its clade is rare.
    const { assert!(qd::RARITY_WEIGHT > 0.0) };
    let level = draws(&[]);
    let rare = draws(&rarity);
    assert!(level < 10, "{level} draws without the bonus");
    assert!(rare > 4 * level.max(2), "{rare} draws with the bonus");
}

#[test]
fn the_global_archive_has_finer_body_classes_than_an_island() {
    use evolution_simulator::qd::{GLOBAL_CLASSES, ISLAND_CLASSES};
    assert!(GLOBAL_CLASSES.shapes() > ISLAND_CLASSES.shapes());
    assert!(GLOBAL_CLASSES.sizes() > ISLAND_CLASSES.sizes());
    assert_eq!(GLOBAL_CLASSES.shape_names.len(), GLOBAL_CLASSES.shapes());
    assert_eq!(GLOBAL_CLASSES.size_names.len(), GLOBAL_CLASSES.sizes());
    // Bodies one island class lumps together have cells of their own in the
    // global archive.
    let body = |nodes: u16, aspect_ratio: f32| Descriptor {
        ground_contact: 0.5,
        gait_frequency: 1.0,
        mean_height: 0.5,
        feet: 2.0,
        nodes,
        aspect_ratio,
        ..Descriptor::default()
    };
    let (compact, tall) = (body(7, 1.0), body(7, 0.7));
    assert_eq!(
        compact.niche_in(&ISLAND_CLASSES),
        tall.niche_in(&ISLAND_CLASSES)
    );
    assert_ne!(
        compact.niche_in(&GLOBAL_CLASSES),
        tall.niche_in(&GLOBAL_CLASSES)
    );
    let (small, medium) = (body(9, 1.0), body(11, 1.0));
    assert_eq!(
        small.niche_in(&ISLAND_CLASSES),
        medium.niche_in(&ISLAND_CLASSES)
    );
    assert_ne!(
        small.niche_in(&GLOBAL_CLASSES),
        medium.niche_in(&GLOBAL_CLASSES)
    );
}

#[test]
fn a_world_change_checkpoint_retests_the_same_elites() {
    let mut uninterrupted = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut uninterrupted);
    assert!(!uninterrupted.island_progress.is_empty());
    let mut changed = uninterrupted.config.clone();
    changed.gravity += 1.0;
    uninterrupted.update_config_now(changed).unwrap();
    // The main islands start over; the wild islands keep their own worlds.
    assert!(main_islands_empty(&uninterrupted));
    assert!(!uninterrupted.reseed.is_empty());

    let checkpoint = Checkpoint::new("world-change");
    storage::save(&checkpoint.0, &uninterrupted).unwrap();
    let restored = storage::load(&checkpoint.0).unwrap();
    assert_eq!(restored.config, uninterrupted.config);
    assert!(restored.archive.entries.is_empty());
    // The queued elites are bred back into the loaded ring first.
    assert!(restored.reseed.is_empty());
    for creature in uninterrupted.reseed.iter() {
        assert!(
            births_ids(&restored).any(|id| id == creature.id),
            "elite {} is not in the ring",
            creature.id
        );
    }
}

#[test]
fn a_world_change_keeps_the_layout_of_a_refined_archive() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut experiment);
    // Island 1 and the global archive are refined, the other islands are not.
    let mut refined = plateau_archive();
    refined.set_refined(true);
    refined.rebin();
    experiment.islands[1] = refined.clone();
    experiment.archive = refined;
    assert!(!experiment.islands[2].refined());
    let mut changed = experiment.config.clone();
    changed.gravity += 1.0;
    experiment.update_config_now(changed).unwrap();
    // Every archive is empty and keeps its layout, the nurseries of new
    // random bodies are coarse and the nurseries of reshaped bodies refined.
    // The elites of island 1 wait to be tested again.
    assert_eq!(experiment.islands.len(), storage::arena_count());
    for (arena, island) in experiment.islands.iter().enumerate() {
        if evolution_simulator::qd::is_wild(arena % storage::island_count()) {
            continue;
        }
        assert!(island.entries.is_empty());
        assert_eq!(
            island.refined(),
            arena == 1 || arena >= storage::reshaped_of(0),
            "arena {arena}"
        );
    }
    assert!(experiment.archive.entries.is_empty() && experiment.archive.refined());
    assert!(experiment.reseed.len() >= 1440);
    // The elites that were tested again sit in the cells of their body classes.
    for _ in 0..4 {
        run_synthetic(&mut experiment);
    }
    let island = &experiment.islands[1];
    assert!(island.refined() && !experiment.islands[2].refined());
    assert!(
        island
            .entries
            .iter()
            .any(|e| e.niche.0[2] != 0 || e.niche.0[5] != 0)
    );
    for elite in &island.entries {
        if !evolution_simulator::qd::is_morphology_niche(&elite.niche) {
            assert_eq!(
                elite.niche,
                elite
                    .descriptor
                    .niche_in(&evolution_simulator::qd::ISLAND_CLASSES)
            );
        }
    }
    experiment.validate().unwrap();
    // Without a refined archive the islands start empty, as they always did.
    let mut plain = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut plain);
    let mut changed = plain.config.clone();
    changed.gravity += 1.0;
    plain.update_config_now(changed).unwrap();
    assert!(main_islands_empty(&plain) && plain.archive.refined());
}

fn births_ids(e: &Experiment) -> impl Iterator<Item = u64> + '_ {
    e.blocks
        .iter()
        .flat_map(|b| b.population.genomes.iter().map(|g| g.id))
}

#[test]
fn a_world_change_at_the_boundary_keeps_the_state_valid() {
    let mut uninterrupted = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut uninterrupted);
    let mut changed = uninterrupted.config.clone();
    changed.ground_friction *= 0.5;
    uninterrupted.update_config(changed.clone()).unwrap();
    assert!(uninterrupted.pending.is_some());
    run_synthetic(&mut uninterrupted);
    assert!(uninterrupted.pending.is_none());
    assert_eq!(
        uninterrupted.config.ground_friction,
        changed.ground_friction
    );
    // The world changed at the boundary: the archives start over and the
    // old elites wait to be tested again, the first of them in the block
    // bred right after the boundary.
    assert!(main_islands_empty(&uninterrupted));
    assert!(!uninterrupted.reseed.is_empty());
    uninterrupted.validate().unwrap();

    let checkpoint = Checkpoint::new("boundary-world-change");
    storage::save(&checkpoint.0, &uninterrupted).unwrap();
    let restored = storage::load(&checkpoint.0).unwrap();
    assert_eq!(restored.config, uninterrupted.config);
    assert!(restored.pending.is_none());
    assert_eq!(restored.generation, uninterrupted.generation);
    assert_eq!(restored.evaluated, 0);
    assert!(restored.archive.entries.is_empty());
    for creature in uninterrupted.reseed.iter() {
        assert!(births_ids(&restored).any(|id| id == creature.id));
    }
}

#[test]
fn a_world_change_rescores_the_archive_in_the_new_world() {
    let mut experiment = Experiment::new(Config {
        duration: 2.0,
        ..config(38)
    })
    .unwrap();
    let mut evaluate = synthetic(0);
    experiment.run_generation(&mut evaluate).unwrap();
    assert!(!experiment.archive.entries.is_empty());
    let calm_champion = experiment
        .archive
        .entries
        .iter()
        .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        .unwrap()
        .fitness;
    assert!(calm_champion.is_finite());

    // A terrain change for the next generation: the world changes at the
    // boundary and the archives start over.
    experiment
        .update_config(Config {
            terrain: 3,
            ..experiment.config.clone()
        })
        .unwrap();
    experiment.run_generation(&mut evaluate).unwrap();
    assert!(
        experiment.archive.entries.is_empty(),
        "the world change must clear the old scores"
    );
    assert!(!experiment.reseed.is_empty());
    // The next generation tests the queued elites in the rough world.
    experiment.run_generation(&mut evaluate).unwrap();
    assert!(!experiment.archive.entries.is_empty());
}

#[test]
fn rebuilding_islands_resets_records_from_the_previous_partition() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut experiment);
    // A checkpoint saved with a different island count gets new, empty
    // islands on the next absorption. Its previous records cannot apply.
    experiment.islands.pop();
    experiment.island_progress = vec![(1.0e9, 0); experiment.islands.len()];
    let generation = experiment.generation;
    experiment.step(&mut synthetic(generation)).unwrap();
    assert_eq!(experiment.islands.len(), storage::arena_count());
    assert_eq!(experiment.island_progress.len(), experiment.islands.len());
    for (island, &(record, generation)) in
        experiment.islands.iter().zip(&experiment.island_progress)
    {
        assert_eq!(record, island.best_fitness());
        assert_eq!(generation, experiment.generation);
    }
}

#[test]
fn an_island_migration_is_recorded_and_summarized() {
    use evolution_simulator::worker::{IslandSummary, MigrationSummary};
    let mut experiment = Experiment::new(config(38)).unwrap();
    run_synthetic(&mut experiment);
    assert!(experiment.last_migration.is_none());
    // The generation before a migration boundary.
    experiment.generation = storage::MIGRATION_INTERVAL - 1;
    run_synthetic(&mut experiment);
    let (generation, exchange) = experiment.last_migration.clone().unwrap();
    assert_eq!(generation, storage::MIGRATION_INTERVAL);
    assert_eq!(exchange.len(), storage::island_count());
    // Every isolated island sends to the hub; the hub sends nothing.
    for (island, &(sent, kept)) in exchange.iter().enumerate() {
        if island == storage::hub_island() {
            assert_eq!((sent, kept), (0, 0));
        } else if evolution_simulator::qd::is_wild(island) {
            // A wild island's best run again in the hub's world first.
            assert_eq!(kept, 0);
        } else {
            assert!(sent > 0);
            assert!(kept <= sent);
        }
    }
    let migration = MigrationSummary {
        generation,
        exchange: exchange.clone(),
    };
    let (sent, kept) = migration.hub_received();
    assert_eq!(sent, exchange.iter().map(|e| e.0).sum::<usize>());
    assert_eq!(kept, exchange.iter().map(|e| e.1).sum::<usize>());

    for (index, island) in experiment.islands[..evolution_simulator::qd::MAIN_ISLANDS]
        .iter()
        .enumerate()
    {
        let nurseries = [
            &experiment.islands[storage::nursery_of(index)],
            &experiment.islands[storage::reshaped_of(index)],
        ];
        let summary = IslandSummary::of(island, nurseries, Default::default());
        assert_eq!(summary.cells, island.behavior_count());
        assert_eq!(summary.origins.iter().sum::<usize>(), summary.cells);
        assert!(!summary.top.is_empty() && summary.top.len() <= 3);
        assert!(summary.top.windows(2).all(|w| w[0].0 >= w[1].0));
        assert_eq!(summary.best, summary.top[0].0);
        assert_eq!(summary.leader.as_ref().unwrap().id, summary.top[0].1.id);
        assert_eq!(summary.best, island.best_fitness());
    }
    let empty = IslandSummary::of(
        &QdArchive::default(),
        [&QdArchive::default(), &QdArchive::default()],
        Default::default(),
    );
    assert!(empty.best.is_nan() && empty.leader.is_none() && empty.cells == 0);
}

/// The island a creature was born in, from the ring slot its id carries.
fn birth_island(id: u64) -> usize {
    evolution_simulator::qd::island_of_slot(evolution::slot_of_id(id), storage::island_count())
}

#[test]
fn isolated_islands_only_hold_their_own_descendants() {
    let mut experiment = Experiment::new(Config {
        population: 500,
        ..config(38)
    })
    .unwrap();
    let hub = storage::hub_island();
    // After a world change an island holds its elites tested again, a graduate
    // among them without its mark.
    let check = |experiment: &Experiment, reseeded: bool| {
        for (arena, island) in experiment.islands.iter().enumerate() {
            // A nursery belongs to one island like the island's archive.
            let index = arena % storage::island_count();
            if index == hub {
                continue;
            }
            for elite in &island.entries {
                // Only nursery slots fill the nursery of new bodies (the
                // nursery of reshaped bodies also takes the new body plans
                // that island slots bred), and an island archive holds
                // nursery bodies only as marked graduates.
                let slot = evolution::slot_of_id(elite.creature.id);
                let nursery_slot =
                    evolution_simulator::qd::is_nursery_slot(slot, storage::island_count());
                if arena >= storage::reshaped_of(0) {
                    assert!(!elite.graduate);
                } else if arena >= storage::island_count() {
                    assert!(nursery_slot && !elite.graduate);
                } else if !elite.graduate && !reseeded {
                    assert!(!nursery_slot);
                }
                // The elite and every recorded ancestor were born here, or,
                // from generation 50, on the island before it in the ring of
                // stepping stones.
                let isolated = storage::ISOLATED_ISLANDS;
                let born_ok = |born: usize| {
                    born == index
                        || (experiment.generation >= 50
                            && index < isolated
                            && born < isolated
                            && born != hub)
                };
                // Walk the chain by the ids that key the lineage records. A
                // record keeps its genes only for the chains the lineage tab
                // shows (`prune_lineage`), and the others hold an empty
                // creature with id 0, which reads as born on island 0.
                let mut chain = Some(elite.creature.id);
                while let Some(id) = chain {
                    let Some(record) = experiment.lineage.get(&id) else {
                        break;
                    };
                    assert!(
                        born_ok(birth_island(id)),
                        "island {index} holds a creature from another island"
                    );
                    chain = record.parent;
                }
                assert!(born_ok(birth_island(elite.creature.id)));
            }
        }
        for cma in &experiment.cma_emitters {
            assert!(cma.island < storage::arena_count());
        }
    };
    for generation in 0..2 * storage::MIGRATION_INTERVAL + 3 {
        if generation == storage::MIGRATION_INTERVAL + 5 {
            // A world change queues each island's elites for its own slots.
            let mut changed = experiment.config.clone();
            changed.gravity += 1.0;
            experiment.update_config(changed).unwrap();
        }
        run_synthetic(&mut experiment);
        check(&experiment, generation > storage::MIGRATION_INTERVAL + 5);
    }
    // The nurseries sent cohorts to their islands.
    assert!(experiment.graduations.iter().any(|g| g.sent > 0));
    // The isolated islands sent copies to the hub.
    let (_, exchange) = experiment.last_migration.clone().unwrap();
    assert!(
        exchange[..storage::ISOLATED_ISLANDS]
            .iter()
            .all(|&(sent, _)| sent > 0)
    );
}

/// Whether every archive of the main islands (and their nurseries) is empty.
fn main_islands_empty(e: &Experiment) -> bool {
    e.islands.iter().enumerate().all(|(arena, island)| {
        evolution_simulator::qd::is_wild(arena % storage::island_count()) || island.entries.is_empty()
    })
}
