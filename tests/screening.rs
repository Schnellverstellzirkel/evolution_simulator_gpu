//! Early screening (`physics::Screen`) on the CUDA kernel: at the screen
//! time a standing creature below the bar stops like a fall, keeping its
//! distance there, and enters no archive; survivors run the full trial
//! unchanged.
//!
//! Needs the RTX 4060 and is ignored by default. Run it with
//!
//!     cargo test --release --test screening -- --ignored
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

const SCREEN_SECONDS: f32 = 2.0;

fn config(duration: f32) -> Config {
    Config {
        population: 64,
        duration,
        random_seed: false,
        seed: 41,
        ..Config::default()
    }
}

fn screened(cfg: &Config, bar: f32) -> Config {
    Config {
        screen: Some(Screen::uniform(SCREEN_SECONDS, bar)),
        ..cfg.clone()
    }
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0)
}

/// The recorded distance at the screen, and the bar at its median.
fn median_bar(results: &[GpuResult]) -> f32 {
    let mut distances: Vec<f32> = results.iter().map(|r| r.screen_x).collect();
    distances.sort_by(f32::total_cmp);
    distances[distances.len() / 2]
}

#[test]
#[ignore = "needs the RTX 4060"]
fn screened_creatures_end_at_the_screen_and_survivors_run_in_full() {
    let cfg = config(6.0);
    let pop = evolution::create(&cfg).unwrap();
    // No bar: every trial runs in full and records its screen distance.
    let full = evaluate(&pop, &screened(&cfg, f32::NEG_INFINITY));
    assert!(full.iter().all(|r| r.screened == 0.0));
    let bar = median_bar(&full);
    let short = evaluate(&pop, &config(SCREEN_SECONDS));
    let screen_cfg = screened(&cfg, bar);
    let results = evaluate(&pop, &screen_cfg);
    let mut stopped = 0;
    for (i, ((r, f), s)) in results.iter().zip(&full).zip(&short).enumerate() {
        assert_eq!(
            r.screen_x.to_bits(),
            f.screen_x.to_bits(),
            "creature {i}: the screen distance must not depend on the bar"
        );
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

#[test]
#[ignore = "needs the RTX 4060"]
fn replays_run_the_full_trial_even_under_a_screen() {
    let cfg = config(4.0);
    let pop = evolution::create(&cfg).unwrap();
    let full = evaluate(&pop, &cfg);
    let everyone_below = screened(&cfg, f32::INFINITY);
    for (i, expected) in full.iter().enumerate().take(4) {
        let (_, replayed, _) = evolution_simulator::engine::replay(
            &pop.creature(i),
            &everyone_below,
            std::time::Duration::from_secs(60),
        )
        .expect("a GPU replay");
        assert_eq!(replayed.screened, 0.0);
        assert_eq!(replayed.fitness.to_bits(), expected.fitness.to_bits());
    }
}

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
    // The first generation has no bar and runs every trial in full.
    let first = experiment
        .config
        .screen
        .expect("screening is on by default");
    assert_eq!(first.bar, f32::NEG_INFINITY);
    assert_eq!(first.seconds, physics::screen_seconds().unwrap());
    for generation in 0..3 {
        // Standard trials of the generation: ids of screened creatures and
        // every distance at the screen.
        let mut screened_ids: Vec<u64> = Vec::new();
        let mut distances: Vec<f32> = Vec::new();
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
        let bar = experiment.config.screen.unwrap().bar;
        let kept = distances.iter().filter(|&&d| d >= bar).count() as f32 / distances.len() as f32;
        assert!(
            (kept - physics::screen_keep()).abs() < 0.05,
            "generation {generation}: bar {bar} keeps {kept}"
        );
    }
}

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
    // The ring's first block is a quarter of the generation.
    assert_eq!(experiment.blocks[0].len(), 64);
    let mut distances: Vec<f32> = Vec::new();
    // Absorbs the block at the cursor and records its distances at the screen.
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
    // The next block runs with the bar.
    assert_eq!(
        experiment.blocks[1].config.screen.unwrap().bar,
        f32::NEG_INFINITY
    );
    assert_eq!(experiment.blocks[0].config.screen.unwrap().bar, armed);
    // The second block moves the bar to the share of both blocks' results,
    // and the block bred then takes it.
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
