use bincode::Options;
use evolution_simulator::{
    config::Config,
    evolution::{self, Population},
    qd::{Descriptor, Emitter, Niche, QdArchive, TrialMetrics},
    storage::{self, Experiment, Stage},
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
    let fields: [FloatCase; 11] = [
        ("duration", |c| &mut c.duration, 0.1, 300.0),
        ("mutation", |c| &mut c.mutation, 0.0, 10.0),
        ("gravity", |c| &mut c.gravity, 0.0, 100.0),
        ("air retention", |c| &mut c.air_retention, 0.0, 1.02),
        ("ground friction", |c| &mut c.ground_friction, 0.0, 20.0),
        ("muscle energy", |c| &mut c.muscle_energy, 0.05, 2.0),
        ("muscle recovery", |c| &mut c.muscle_recovery, 0.05, 2.0),
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
        ("nodes", |c| &mut c.max_nodes, 3, 64),
        ("muscles", |c| &mut c.max_muscles, 3, 256),
        ("GPU budget", |c| &mut c.gpu_budget_mib, 32, 6144),
        ("RAM budget", |c| &mut c.ram_budget_mib, 64, 24576),
    ];
    let base = Config {
        population: 2,
        max_nodes: 3,
        max_muscles: 256,
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
            let niche = descriptor.niche();
            let old = expected.get(&niche).copied();
            let should_insert = old.is_none_or(|(best, _)| fitness > best);
            let offer = archive.offer(
                &population,
                index,
                descriptor,
                fitness,
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
            Emitter::Restart,
            100,
            0,
        );
        assert!(!offer.inserted);
        assert_eq!(encoded(&archive), before);
    }
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
// regression tests cheap. Every island receives several distinct cadence cells.
fn archive_synthetic_results(experiment: &mut Experiment) {
    for i in 0..experiment.config.population {
        experiment.scores[i] = 10.0 + experiment.generation as f32 + i as f32;
        experiment.trial_metrics[i] = TrialMetrics {
            ground_contact: 0.5,
            gait_frequency: ((i / storage::island_count()) % 8) as f32 * 0.75 + 0.1,
            mean_height: 0.5,
            feet: 2.0,
            ..TrialMetrics::default()
        };
    }
    experiment.evaluated = experiment.config.population;
    experiment.stage = Stage::Evaluated;
    experiment.archive_batch().unwrap();
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
        assert_eq!(a.mutability, b.mutability, "creature {index} mutability");
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

fn assert_same_next_batch(a: &Experiment, b: &Experiment) {
    a.validate().unwrap();
    b.validate().unwrap();
    assert_same_population(&a.population, &b.population);
    assert_eq!(a.config, b.config);
    assert_eq!(a.generation, b.generation);
    assert_eq!(a.breed_round, b.breed_round);
    assert_eq!(a.stage, b.stage);
    assert_eq!(a.evaluated, b.evaluated);
    assert_eq!(a.candidate_emitters, b.candidate_emitters);
    assert_eq!(a.candidate_cma, b.candidate_cma);
    assert_eq!(a.candidate_parent_ids, b.candidate_parent_ids);
    assert_eq!(a.candidate_mates, b.candidate_mates);
    assert_eq!(a.protected_until, b.protected_until);
    assert!(
        encoded(&a.cma_emitters) == encoded(&b.cma_emitters),
        "CMA state differs"
    );
    assert_same_archive(&a.archive, &b.archive);
    assert_eq!(a.islands.len(), b.islands.len());
    for (a, b) in a.islands.iter().zip(&b.islands) {
        assert_same_archive(a, b);
    }
    assert!(a.scores.iter().all(|score| score.is_nan()));
    assert!(b.scores.iter().all(|score| score.is_nan()));
}

#[test]
fn archive_breeding_is_repeatable_and_valid_across_streaming_slice_sizes() {
    for seed in [7, 38, 91] {
        let mut whole = Experiment::new(config(seed)).unwrap();
        let mut streamed = Experiment::new(config(seed)).unwrap();
        for slice in [1, 17, 63, 256] {
            archive_synthetic_results(&mut whole);
            archive_synthetic_results(&mut streamed);
            whole.prepare_next_batch().unwrap();
            let mut delivered = Population::default();
            streamed
                .prepare_next_batch_streaming(slice, |population, range, cfg| {
                    assert_eq!(range.start, delivered.genomes.len());
                    assert!(range.len() <= slice);
                    assert_eq!(cfg.seed, seed);
                    for index in range {
                        delivered.push(population.creature(index));
                    }
                    Ok(())
                })
                .unwrap();
            delivered.validate(&whole.config).unwrap();
            assert_same_population(&delivered, &whole.population);
            assert_same_next_batch(&whole, &streamed);
        }
        assert!(!whole.cma_emitters.is_empty(), "CMA must be exercised");
        assert!(whole.candidate_emitters.contains(&Emitter::Structural));
        assert!(whole.candidate_emitters.contains(&Emitter::Novelty));
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
    assert_same_population(&experiment.population, &original.population);

    archive_synthetic_results(&mut experiment);
    experiment.prepare_next_batch().unwrap();
    storage::save(&checkpoint.0, &experiment).unwrap();
    let replaced = storage::load(&checkpoint.0).unwrap();
    assert_eq!(replaced.generation, original.generation + 1);
    assert_same_population(&experiment.population, &replaced.population);
    assert_eq!(encoded(&experiment.archive), encoded(&replaced.archive));
    assert!(!checkpoint.0.with_extension("evo.tmp").exists());
}

#[test]
fn checkpoint_restores_archive_and_cma_for_identical_next_generation() {
    let mut uninterrupted = Experiment::new(config(38)).unwrap();
    for _ in 0..3 {
        archive_synthetic_results(&mut uninterrupted);
        uninterrupted.prepare_next_batch().unwrap();
    }
    archive_synthetic_results(&mut uninterrupted);
    assert!(!uninterrupted.cma_emitters.is_empty());
    let checkpoint = Checkpoint::new("archive-resume");
    storage::save(&checkpoint.0, &uninterrupted).unwrap();
    let mut restored = storage::load(&checkpoint.0).unwrap();
    uninterrupted.prepare_next_batch().unwrap();
    restored.prepare_next_batch().unwrap();
    assert_same_next_batch(&uninterrupted, &restored);

    // A steady-state resume must also retain the RNG salt from earlier rounds.
    let slots: Vec<_> = (0..uninterrupted.config.population).rev().collect();
    uninterrupted.breed_slots(&slots).unwrap();
    uninterrupted.breed_slots(&slots).unwrap();
    let steady_checkpoint = Checkpoint::new("steady-resume");
    storage::save(&steady_checkpoint.0, &uninterrupted).unwrap();
    restored = storage::load(&steady_checkpoint.0).unwrap();
    uninterrupted.breed_slots(&slots).unwrap();
    restored.breed_slots(&slots).unwrap();
    assert_eq!(uninterrupted.breed_round, 3);
    assert_same_next_batch(&uninterrupted, &restored);
}

#[test]
fn checkpoint_preserves_stalled_island_optimizer_next_generation() {
    let mut uninterrupted = Experiment::new(config(38)).unwrap();
    archive_synthetic_results(&mut uninterrupted);
    uninterrupted.prepare_next_batch().unwrap();
    archive_synthetic_results(&mut uninterrupted);
    // Advance the record age past the 30-generation optimizer rotation without
    // spending 30 generations evaluating bodies. The archived scores and CMA
    // state remain valid; the next batch must use the same alternate design.
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
            .all(|island| island.behavior_count() > 1)
    );
    uninterrupted.validate().unwrap();
    let checkpoint = Checkpoint::new("stalled-island-resume");
    storage::save(&checkpoint.0, &uninterrupted).unwrap();
    let mut restored = storage::load(&checkpoint.0).unwrap();
    assert_eq!(uninterrupted.island_progress, restored.island_progress);
    uninterrupted.prepare_next_batch().unwrap();
    restored.prepare_next_batch().unwrap();
    assert!(
        uninterrupted
            .candidate_cma
            .iter()
            .flatten()
            .any(|&index| uninterrupted.cma_emitters[index].optimizing())
    );
    assert_same_next_batch(&uninterrupted, &restored);
}

#[test]
fn v3_checkpoint_keeps_current_archives_and_can_continue_breeding() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    archive_synthetic_results(&mut experiment);
    experiment.prepare_next_batch().unwrap();
    archive_synthetic_results(&mut experiment);
    // V3 serialized only Experiment. Keep a fixture using that exact payload,
    // independent of the format emitted by the current save implementation.
    let payload = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(&experiment)
        .unwrap();
    let mut bytes = b"EVORUST3".to_vec();
    bytes.extend(zstd::stream::encode_all(payload.as_slice(), 3).unwrap());
    let checkpoint = Checkpoint::new("v3-resume");
    std::fs::write(&checkpoint.0, bytes).unwrap();

    let mut restored = storage::load(&checkpoint.0).unwrap();
    assert_eq!(restored.qd_version, experiment.qd_version);
    assert_eq!(restored.stage, Stage::Archived);
    assert_eq!(restored.generation, experiment.generation);
    assert_eq!(restored.scores, experiment.scores);
    assert_same_population(&restored.population, &experiment.population);
    assert!(encoded(&restored.archive) == encoded(&experiment.archive));
    assert!(encoded(&restored.islands) == encoded(&experiment.islands));
    assert!(encoded(&restored.cma_emitters) == encoded(&experiment.cma_emitters));
    assert_eq!(restored.history.len(), experiment.history.len());
    // V3 did not save record ages, so exact stalled continuation is available
    // only for V4; old files must still load without discarding valid archives.
    assert!(restored.island_progress.is_empty());
    restored.prepare_next_batch().unwrap();
    restored.validate().unwrap();
}

#[test]
fn checkpoint_rejects_invalid_optimizer_resume_metadata() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    archive_synthetic_results(&mut experiment);
    experiment.prepare_next_batch().unwrap();
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
    // No planning pass has run yet after loading a V3 save or after an empty
    // island receives its first elite. Both forms are valid resume states.
    for progress in [Vec::new(), vec![(f32::NEG_INFINITY, 0); islands]] {
        experiment.island_progress = progress;
        storage::save(&checkpoint.0, &experiment).unwrap();
        let restored = storage::load(&checkpoint.0).unwrap();
        assert_eq!(restored.island_progress, experiment.island_progress);
    }
}

#[test]
fn old_physics_checkpoint_clears_stale_islands_and_queued_reseeds() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    archive_synthetic_results(&mut experiment);
    experiment.prepare_next_batch().unwrap();
    archive_synthetic_results(&mut experiment);
    experiment
        .reseed
        .push(experiment.archive.entries[0].creature.clone());
    experiment.qd_version -= 1;
    assert!(!experiment.cma_emitters.is_empty());
    assert!(
        experiment
            .islands
            .iter()
            .all(|island| !island.entries.is_empty())
    );
    let historical_representatives = encoded(&experiment.history[0].representatives);
    let checkpoint = Checkpoint::new("old-physics-islands");
    storage::save(&checkpoint.0, &experiment).unwrap();

    let mut restored = storage::load(&checkpoint.0).unwrap();
    assert_eq!(restored.qd_version, evolution_simulator::qd::VERSION);
    assert_eq!(restored.stage, Stage::Ready);
    assert_eq!(restored.evaluated, 0);
    assert!(restored.scores.iter().all(|score| score.is_nan()));
    assert!(restored.archive.entries.is_empty());
    assert!(restored.islands.is_empty());
    assert!(restored.island_progress.is_empty());
    assert!(restored.cma_emitters.is_empty());
    assert!(restored.reseed.is_empty());
    assert_eq!(restored.history.len(), 1);
    assert_eq!(
        encoded(&restored.history[0].representatives),
        historical_representatives
    );
    restored.validate().unwrap();

    let slots: Vec<_> = (0..restored.config.population).collect();
    restored.breed_slots(&slots).unwrap();
    assert!(
        restored
            .candidate_emitters
            .iter()
            .all(|&emitter| emitter == Emitter::Restart)
    );
    assert!(restored.candidate_parent_ids.iter().all(Option::is_none));
    restored.validate().unwrap();
}

#[test]
fn ready_environment_change_checkpoint_retests_the_same_elites() {
    let mut uninterrupted = Experiment::new(config(38)).unwrap();
    archive_synthetic_results(&mut uninterrupted);
    uninterrupted.prepare_next_batch().unwrap();
    assert!(!uninterrupted.island_progress.is_empty());
    let mut changed = uninterrupted.config.clone();
    changed.gravity += 1.0;
    uninterrupted.update_config(changed.clone()).unwrap();
    assert!(uninterrupted.islands.is_empty());
    assert!(!uninterrupted.reseed.is_empty());

    let checkpoint = Checkpoint::new("ready-world-change");
    storage::save(&checkpoint.0, &uninterrupted).unwrap();
    let mut restored = storage::load(&checkpoint.0).unwrap();
    assert_eq!(restored.config, changed);
    assert!(restored.island_progress.is_empty());
    assert!(restored.archive.entries.is_empty());
    assert!(restored.cma_emitters.is_empty());
    assert_eq!(encoded(&restored.reseed), encoded(&uninterrupted.reseed));
    let elite_ids: Vec<_> = restored.reseed.iter().map(|creature| creature.id).collect();
    let slots: Vec<_> = (0..restored.config.population).collect();
    uninterrupted.breed_slots(&slots).unwrap();
    restored.breed_slots(&slots).unwrap();
    assert_same_next_batch(&uninterrupted, &restored);
    assert!(elite_ids.iter().all(|id| {
        restored
            .population
            .genomes
            .iter()
            .any(|genome| genome.id == *id)
    }));
}

#[test]
fn steady_environment_change_checkpoint_keeps_boundary_state_valid() {
    let mut uninterrupted = Experiment::new(config(38)).unwrap();
    archive_synthetic_results(&mut uninterrupted);
    uninterrupted.prepare_next_batch().unwrap();
    uninterrupted.stage = Stage::Evaluating;
    let mut changed = uninterrupted.config.clone();
    changed.ground_friction *= 0.5;
    uninterrupted.update_config(changed.clone()).unwrap();
    assert!(uninterrupted.pending.is_some());
    let slots: Vec<_> = (0..uninterrupted.config.population).collect();
    uninterrupted.scores.fill(20.0);
    uninterrupted.evaluated = uninterrupted.config.population;
    uninterrupted.archive_slots(&slots);
    uninterrupted.breed_slots(&slots).unwrap();
    uninterrupted.finish_steady_generation(0).unwrap();
    assert!(uninterrupted.islands.is_empty());
    assert!(!uninterrupted.reseed.is_empty());

    let checkpoint = Checkpoint::new("steady-world-change");
    storage::save(&checkpoint.0, &uninterrupted).unwrap();
    let mut restored = storage::load(&checkpoint.0).unwrap();
    assert_eq!(restored.config, changed);
    assert!(restored.pending.is_none());
    assert!(restored.island_progress.is_empty());
    assert_eq!(encoded(&restored.reseed), encoded(&uninterrupted.reseed));
    restored.validate().unwrap();
    assert_same_population(&uninterrupted.population, &restored.population);
    assert_eq!(restored.generation, uninterrupted.generation);
    assert_eq!(restored.breed_round, uninterrupted.breed_round);
    assert_eq!(restored.stage, Stage::Evaluating);
    assert_eq!(restored.evaluated, 0);
    assert!(restored.archive.entries.is_empty());
    assert!(restored.cma_emitters.is_empty());
    uninterrupted.breed_slots(&slots).unwrap();
    restored.breed_slots(&slots).unwrap();
    assert_same_next_batch(&uninterrupted, &restored);
}

#[test]
fn rebuilding_islands_resets_records_from_the_previous_partition() {
    let mut experiment = Experiment::new(config(38)).unwrap();
    archive_synthetic_results(&mut experiment);
    experiment.prepare_next_batch().unwrap();
    // A checkpoint saved with a different island count is repartitioned on
    // the next planning pass. Its previous partition's records cannot apply.
    experiment.islands.pop();
    experiment.island_progress = vec![(1.0e9, 0); experiment.islands.len()];
    let slots: Vec<_> = (0..experiment.config.population).collect();
    experiment.breed_slots(&slots).unwrap();
    assert_eq!(experiment.islands.len(), storage::island_count());
    assert_eq!(experiment.island_progress.len(), experiment.islands.len());
    for (island, &(record, generation)) in
        experiment.islands.iter().zip(&experiment.island_progress)
    {
        assert_eq!(record, island.best_fitness());
        assert_eq!(generation, experiment.generation);
    }
}
