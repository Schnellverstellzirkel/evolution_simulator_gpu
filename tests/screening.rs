//! Early screening (`physics::Screen`): at the screen time a standing
//! creature below the bar stops like a fall, keeping its distance there, and
//! enters no archive; survivors run the full trial unchanged.
use evolution_simulator::{
    config::Config,
    cpu_engine,
    creature_kernel::GpuResult,
    evolution,
    physics::{self, Screen},
    scheduler,
    storage::Experiment,
};

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
        screen: Some(Screen {
            seconds: SCREEN_SECONDS,
            bar,
        }),
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
fn screened_creatures_end_at_the_screen_and_survivors_run_in_full() {
    let cfg = config(6.0);
    let pop = evolution::create(&cfg).unwrap();
    // No bar: every trial runs in full and records its screen distance.
    let full = cpu_engine::evaluate(&pop, &screened(&cfg, f32::NEG_INFINITY));
    assert!(full.iter().all(|r| r.screened == 0.0));
    let bar = median_bar(&full);
    let short = cpu_engine::evaluate(&pop, &config(SCREEN_SECONDS));
    let screen_cfg = screened(&cfg, bar);
    let results = cpu_engine::evaluate(&pop, &screen_cfg);
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
fn replays_run_the_full_trial_even_under_a_screen() {
    let cfg = config(4.0);
    let pop = evolution::create(&cfg).unwrap();
    let full = cpu_engine::evaluate(&pop, &cfg);
    let everyone_below = screened(&cfg, f32::INFINITY);
    for (i, expected) in full.iter().enumerate().take(4) {
        let (_, replayed) = cpu_engine::replay(&pop.creature(i), &everyone_below);
        assert_eq!(replayed.screened, 0.0);
        assert_eq!(replayed.fitness.to_bits(), expected.fitness.to_bits());
    }
}

#[test]
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
        let results = cpu_engine::evaluate(&experiment.population, &experiment.config);
        for (i, result) in results.iter().enumerate() {
            let metric =
                scheduler::to_metrics(&experiment.population, i, result, &experiment.config);
            experiment.record_result(i, &metric);
        }
        let screened_ids: Vec<u64> = (0..experiment.config.population)
            .filter(|&i| results[i].screened > 0.0)
            .map(|i| experiment.population.genomes[i].id)
            .collect();
        if generation == 0 {
            assert!(screened_ids.is_empty(), "no bar in the first generation");
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
        let distances: Vec<f32> = results.iter().map(|r| r.screen_x).collect();
        experiment.prepare_next_batch().unwrap();
        let bar = experiment.config.screen.unwrap().bar;
        let kept = distances.iter().filter(|&&d| d >= bar).count() as f32 / distances.len() as f32;
        assert!(
            (kept - physics::screen_keep()).abs() < 0.05,
            "generation {generation}: bar {bar} keeps {kept}"
        );
    }
}

/// The GPU kernel screens like the CPU engine. Rounding can move a creature
/// right at the bar to the other side, so only creatures more than 1 cm from
/// it must agree on the decision.
#[test]
#[ignore = "requires a Vulkan GPU"]
fn gpu_screens_like_the_cpu_engine() {
    let cfg = Config {
        population: 96,
        duration: 3.0,
        random_seed: false,
        seed: 47,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).unwrap();
    let screen = |bar| Config {
        screen: Some(Screen { seconds: 1.0, bar }),
        ..cfg.clone()
    };
    let recorded = cpu_engine::evaluate(&pop, &screen(f32::NEG_INFINITY));
    let bar = median_bar(&recorded);
    let screen_cfg = screen(bar);
    let cpu = cpu_engine::evaluate(&pop, &screen_cfg);
    let mut gpu = evolution_simulator::gpu::Gpu::new("RTX 4060").unwrap();
    let indices: Vec<usize> = (0..cfg.population).collect();
    let gpu = gpu
        .sched
        .as_mut()
        .unwrap()
        .evaluate_single(&pop, &indices, &screen_cfg)
        .unwrap();
    let mut screened = 0;
    for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
        if (recorded[i].screen_x - bar).abs() < 0.01 {
            continue;
        }
        assert_eq!(
            g.screened,
            c.screened > 0.0,
            "creature {i}: screen decision"
        );
        screened += usize::from(g.screened);
        assert!(
            (g.fitness - c.fitness).abs() < 0.05,
            "creature {i}: GPU {} vs CPU {}",
            g.fitness,
            c.fitness
        );
    }
    assert!(screened > 0, "the median bar must screen some creatures");
}
