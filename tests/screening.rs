//! These tests check the early screen (`physics::Screen`) on the CUDA kernel
//! and in the search ring. A creature below the bar at the screen time stops
//! there like a fall and enters no archive. Every other creature runs its
//! full trial unchanged, and so does every replay. The ring sets the bar from
//! its newest results and clears it when the world changes.
//!
//! They need the RTX 4060 and are ignored by default. Run them with
//!
//!     cargo test --release --test screening -- --ignored
/// The GPU helpers that the example tools share: open the engine and score a
/// population.
#[path = "../examples/common/mod.rs"]
mod common;
use evolution_simulator::{
    config::Config,
    creature_kernel::GpuResult,
    engine::ThreadedEngine,
    evolution::{self, Population},
    physics::{self, Screen},
    scheduler,
    storage::Experiment,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// The GPU engine all tests of this file share. It also records replays.
fn gpu() -> MutexGuard<'static, ThreadedEngine> {
    static GPU: OnceLock<Mutex<ThreadedEngine>> = OnceLock::new();
    GPU.get_or_init(|| Mutex::new(common::open().expect("the GPU")))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Every creature's result on the GPU, in population order.
fn evaluate(pop: &Population, cfg: &Config) -> Vec<GpuResult> {
    common::score(&mut gpu(), pop, cfg).expect("GPU trials")
}

/// The screen time of the tests that score a population themselves. A test
/// that runs an `Experiment` gets the game's screen time from
/// `physics::screen_seconds()`.
const SCREEN_SECONDS: f32 = 2.0;

/// Settings for 64 creatures from a fixed seed, with trials of `duration`
/// seconds and no screen.
fn config(duration: f32) -> Config {
    Config {
        population: 64,
        duration,
        random_seed: false,
        seed: 41,
        ..Config::default()
    }
}

/// `cfg` with a screen at `SCREEN_SECONDS` and the same `bar` for every
/// creature.
fn screened(cfg: &Config, bar: f32) -> Config {
    Config {
        screen: Some(Screen::uniform(SCREEN_SECONDS, bar)),
        ..cfg.clone()
    }
}

/// Whether `a` and `b` differ by at most 1e-4 of the larger of `|a|`, `|b|`
/// and 1.
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0)
}

/// The median of the results' distances at the screen. Half of the
/// creatures are below it.
fn median_bar(results: &[GpuResult]) -> f32 {
    let mut distances: Vec<f32> = results.iter().map(|r| r.screen_x).collect();
    distances.sort_by(f32::total_cmp);
    distances[distances.len() / 2]
}

/// A standing creature below the bar at the screen time stops with the result
/// and the behavior of a trial that lasts the screen time. Every other
/// creature scores as it does with no bar. The bar never changes the distance
/// recorded at the screen.
#[test]
#[ignore = "needs the RTX 4060"]
fn screened_creatures_end_at_the_screen_and_survivors_run_in_full() {
    let cfg = config(6.0);
    let pop = evolution::create(&cfg).unwrap();
    // No bar: every trial runs in full and records its screen distance.
    let full = evaluate(&pop, &screened(&cfg, f32::NEG_INFINITY));
    assert!(full.iter().all(|r| r.screened == 0.0));
    let bar = median_bar(&full);
    // A trial that lasts as long as the screen time, with no screen. A
    // screened creature must score what it scores here.
    let short = evaluate(&pop, &config(SCREEN_SECONDS));
    let screen_cfg = screened(&cfg, bar);
    let results = evaluate(&pop, &screen_cfg);
    let mut stopped = 0;
    // Per creature, `r` is the result under the bar, `f` the result with no
    // bar and `s` the result of the short trial.
    for (i, ((r, f), s)) in results.iter().zip(&full).zip(&short).enumerate() {
        assert_eq!(
            r.screen_x.to_bits(),
            f.screen_x.to_bits(),
            "creature {i}: the screen distance must not depend on the bar"
        );
        // A creature that fell by the screen time ends in its fall. The
        // screen does not stop it.
        let fell_first = f.fall_time > 0.0 && f.fall_time <= SCREEN_SECONDS + 1e-4;
        if !fell_first && f.screen_x < bar {
            stopped += 1;
            assert!(r.screened > 0.0, "creature {i} is below the bar");
            assert!(
                close(r.fitness, s.fitness),
                "creature {i}: screened {} vs a {SCREEN_SECONDS} s trial {}",
                r.fitness,
                s.fitness
            );
            // The behavior totals of a screened creature end at the screen,
            // so its metrics match those of the short trial.
            let a = scheduler::to_metrics(&pop, i, r, &screen_cfg);
            let b = scheduler::to_metrics(&pop, i, s, &config(SCREEN_SECONDS));
            assert!(a.screened && !b.screened);
            assert!(close(a.behavior.ground_contact, b.behavior.ground_contact));
            assert!(close(a.behavior.mean_height, b.behavior.mean_height));
            assert!(close(a.behavior.gait_frequency, b.behavior.gait_frequency));
            assert_eq!(a.behavior.feet, b.behavior.feet);
        } else {
            assert_eq!(r.screened, 0.0, "creature {i} passed or fell first");
            assert_eq!(
                r.fitness.to_bits(),
                f.fitness.to_bits(),
                "creature {i}: a survivor's trial must not change"
            );
            assert_eq!(r.ground_contact.to_bits(), f.ground_contact.to_bits());
            assert_eq!(r.gait_frequency.to_bits(), f.gait_frequency.to_bits());
        }
    }
    assert!(stopped > 0, "the median bar must stop some creatures");
}

/// A replay runs the full trial and scores what the full trial scored, even
/// under a bar that is above every creature.
#[test]
#[ignore = "needs the RTX 4060"]
fn replays_run_the_full_trial_even_under_a_screen() {
    let cfg = config(4.0);
    let pop = evolution::create(&cfg).unwrap();
    // The first `evaluate` of a run opens the GPU engine, and that engine
    // records the replays below.
    let full = evaluate(&pop, &cfg);
    // An infinite bar would stop every creature that still stands at the
    // screen time.
    let everyone_below = screened(&cfg, f32::INFINITY);
    for (i, expected) in full.iter().enumerate().take(4) {
        let (_, replayed, _) = evolution_simulator::engine::replay(
            &pop.creature(i),
            &everyone_below,
            std::time::Duration::from_secs(60),
        )
        .expect("a GPU replay");
        // The replay has no screen, so it ends where the full trial ended.
        assert_eq!(replayed.screened, 0.0);
        assert_eq!(replayed.fitness.to_bits(), expected.fitness.to_bits());
    }
}

/// No screened creature enters an archive. The bar that a generation ends with
/// keeps about the share of its distances that `physics::screen_keep()` names.
#[test]
#[ignore = "needs the RTX 4060"]
fn screened_creatures_enter_no_archive_and_the_bar_keeps_the_top_share() {
    let cfg = Config {
        population: 256,
        duration: 8.0,
        random_seed: false,
        seed: 43,
        ..Config::default()
    };
    let mut experiment = Experiment::new(cfg).unwrap();
    // A new game has the screen on and no bar yet, so its first trials run in
    // full.
    let first = experiment
        .config
        .screen
        .expect("screening is on by default");
    assert_eq!(first.bar, f32::NEG_INFINITY);
    assert_eq!(first.seconds, physics::screen_seconds().unwrap());
    for generation in 0..3 {
        // These collect the ids of the screened creatures and every distance
        // at the screen from the standard trials of the generation. A
        // confirmation trial has a fidelity of its own, so it is left out.
        let mut screened_ids: Vec<u64> = Vec::new();
        let mut distances: Vec<f32> = Vec::new();
        // How many creatures were screened once the first standard block was in.
        let mut first_block = None;
        experiment
            .run_generation(&mut |pop, cfg| {
                let results = evaluate(pop, cfg);
                if cfg.fidelity.is_none() {
                    for (g, r) in pop.genomes.iter().zip(&results) {
                        if r.screened > 0.0 {
                            screened_ids.push(g.id);
                        }
                        distances.push(r.screen_x);
                    }
                    first_block.get_or_insert(screened_ids.len());
                }
                Ok(results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| scheduler::to_metrics(pop, i, r, cfg))
                    .collect())
            })
            .unwrap();
        if generation == 0 {
            assert_eq!(first_block, Some(0), "no bar in the first block");
        }
        // The global archive and the island archives hold no screened
        // creature. The island list includes the nurseries.
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
        // The bar that the generation ends with keeps about the share of the
        // generation's distances that `physics::screen_keep()` names.
        let bar = experiment.config.screen.unwrap().bar;
        let kept = distances.iter().filter(|&&d| d >= bar).count() as f32 / distances.len() as f32;
        assert!(
            (kept - physics::screen_keep()).abs() < 0.05,
            "generation {generation}: bar {bar} keeps {kept}"
        );
    }
}

/// The results of the first block arm the bar. The results of the next block
/// move it. A world change clears it.
#[test]
#[ignore = "needs the RTX 4060"]
fn the_bar_arms_after_the_first_block_and_moves_with_every_block() {
    let cfg = Config {
        population: 256,
        duration: 8.0,
        random_seed: false,
        seed: 53,
        ..Config::default()
    };
    let mut experiment = Experiment::new(cfg).unwrap();
    let bar = |e: &Experiment| e.config.screen.unwrap().bar;
    // TODO: out of date. The default ring has 8 blocks, so its first block
    // holds 32 creatures. Wild islands also take 2 of every 10 slots. They
    // have bars of their own, so 64 creatures give the bar of the evolved
    // creatures at most 52 distances. `physics::screen_bar` needs 64.
    assert_eq!(experiment.blocks[0].len(), 64);
    let mut distances: Vec<f32> = Vec::new();
    // Evaluates and absorbs the block at the cursor and records the distances
    // at the screen of its standard trials.
    let step = |e: &mut Experiment, distances: &mut Vec<f32>| {
        e.step(&mut |pop, cfg| {
            let results = evaluate(pop, cfg);
            if cfg.fidelity.is_none() {
                distances.extend(results.iter().map(|r| r.screen_x));
            }
            Ok(results
                .iter()
                .enumerate()
                .map(|(i, r)| scheduler::to_metrics(pop, i, r, cfg))
                .collect())
        })
        .unwrap()
    };
    step(&mut experiment, &mut distances);
    let armed = bar(&experiment);
    assert!(armed.is_finite(), "the first block must set the bar");
    let kept = distances.iter().filter(|&&d| d >= armed).count() as f32 / 64.0;
    assert!(
        (kept - physics::screen_keep()).abs() < 0.05,
        "the bar keeps {kept} of the sample"
    );
    // A block already in the ring keeps the bar it was bred with. The block
    // bred in place of the absorbed one takes the new bar.
    assert_eq!(
        experiment.blocks[1].config.screen.unwrap().bar,
        f32::NEG_INFINITY
    );
    assert_eq!(experiment.blocks[0].config.screen.unwrap().bar, armed);
    // The second block moves the bar to the distance that the best share of
    // both blocks' results reached. The block bred in its place takes it.
    step(&mut experiment, &mut distances);
    let moved = physics::screen_bar(distances.iter().copied(), physics::screen_keep());
    assert_eq!(bar(&experiment).to_bits(), moved.to_bits());
    assert_eq!(experiment.blocks[1].config.screen.unwrap().bar, moved);
    assert_eq!(experiment.blocks[0].config.screen.unwrap().bar, armed);
    // A world change forgets the old world's distances and the bar.
    let mut rough = experiment.config.clone();
    rough.terrain = 3;
    experiment.update_config_now(rough).unwrap();
    assert_eq!(bar(&experiment), f32::NEG_INFINITY);
}
