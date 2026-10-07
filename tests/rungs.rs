//! The early rungs (`rungs`) and the early screen on the CUDA kernel. A rung
//! stops a creature at 1 s or 2.5 s like a screened one, the host replays each
//! decision from the kernel's trace, and a creature that runs on scores as it
//! does without rules. Audit creatures skip every rule, exempt creatures skip
//! the rungs they are exempt from, and young and reshaped creatures face the
//! screen bar of their own kind.
//!
//! Needs the RTX 4060 and is ignored by default. Run it with
//!
//!     cargo test --release --test rungs -- --ignored
#[path = "../examples/common/mod.rs"]
mod common;
use evolution_simulator::{
    config::Config,
    creature_kernel::GpuResult,
    engine::ThreadedEngine,
    evolution::{self, Population},
    physics::Screen,
    rungs::{self, Rung, Rungs},
};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// The GPU engine that all tests here share. It opens on first use, and the
/// guard keeps the tests one at a time on it, even after one of them panicked.
fn gpu() -> MutexGuard<'static, ThreadedEngine> {
    static GPU: OnceLock<Mutex<ThreadedEngine>> = OnceLock::new();
    GPU.get_or_init(|| Mutex::new(common::open().expect("the GPU")))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Scores every creature of `pop` with `cfg` on the shared GPU, in population
/// order.
fn evaluate(pop: &Population, cfg: &Config) -> Vec<GpuResult> {
    common::score(&mut gpu(), pop, cfg).expect("GPU trials")
}

/// The period of creature `i`'s first muscle, or 0 for a body without
/// muscles. `kernel::pack` hands the kernel the same value as the sixth
/// feature of the rung rule.
fn period(pop: &Population, i: usize) -> f32 {
    let g = &pop.genomes[i];
    if g.muscle_count > 0 {
        pop.muscles[g.muscle_start].period
    } else {
        0.0
    }
}

/// A mix of flagged creatures runs 6 s trials twice, without rules and with
/// an R1 rule and an R2 rule. The host applies the rules to the traces of the
/// first run, and the second run must stop exactly those creatures. A stopped
/// creature ends at its rung with its distance there. A creature that runs on
/// keeps its fitness bits and its steps.
#[test]
#[ignore = "needs the RTX 4060"]
fn the_kernel_stops_what_the_host_replays_and_spares_audit_and_exempt_creatures() {
    let cfg = Config {
        population: 3000,
        duration: 6.0,
        random_seed: false,
        seed: 41,
        ..Config::default()
    };
    let mut pop = evolution::create(&cfg).unwrap();
    pop.flags = (0..pop.genomes.len())
        .map(|i| match i % 7 {
            0 => rungs::AUDIT,
            1 => rungs::EXEMPT,
            2 => rungs::EXEMPT_R1,
            3 => rungs::EXEMPT_R2,
            _ => 0,
        })
        .collect();
    // The trials without rules, against which every ruled trial is checked.
    let full = evaluate(&pop, &cfg);
    // A trial is alive at rung `k` when it ran past the rung's step.
    let alive = |r: &GpuResult, k: usize| r.rung_trace().steps() > rungs::RUNG_STEPS[k];
    // The median of the distances at 2.5 s (150 steps) is the bias of R2.
    let mut d150: Vec<f32> = full
        .iter()
        .filter(|r| alive(r, 1))
        .map(|r| r.rung_trace().distance(1))
        .collect();
    d150.sort_by(f32::total_cmp);
    assert!(d150.len() > 1000, "{} creatures alive at 2.5 s", d150.len());
    let median = d150[d150.len() / 2];
    // The bias of R1 comes from the distances at 1 s (60 steps).
    let mut d60: Vec<f32> = full
        .iter()
        .filter(|r| alive(r, 0))
        .map(|r| r.rung_trace().distance(0))
        .collect();
    d60.sort_by(f32::total_cmp);
    // R1 stops the lowest 1 in 20 by distance at 1 s. R2 stops a creature
    // whose distance at 2.5 s plus 0.01 times its rhythm period is below the
    // median, except in band 3, where R2 is off.
    let mut r1 = Rung::NEVER;
    r1.weights[0] = 1.0;
    r1.bias = d60[d60.len() / 20];
    let mut r2 = Rung::NEVER;
    r2.weights[0] = 1.0;
    r2.weights[5] = 0.01;
    r2.bias = median;
    r2.off = 1 << 3;
    let ruled_cfg = Config {
        rungs: Some(Rungs([r1, r2])),
        ..cfg.clone()
    };
    let ruled = evaluate(&pop, &ruled_cfg);
    let (mut stopped1, mut stopped2, mut spared) = (0, 0, 0);
    for (i, (r, f)) in ruled.iter().zip(&full).enumerate() {
        let (t, ft) = (r.rung_trace(), f.rung_trace());
        let flags = pop.flags[i];
        assert_eq!(
            t.audit(),
            flags & rungs::AUDIT != 0,
            "creature {i}: the audit bit"
        );
        // The cadence band at 1 s does not depend on the rules.
        if alive(f, 0) {
            assert_eq!(t.band(0), ft.band(0), "creature {i}");
        }
        // The features at rung `k`, rebuilt on the host from the trace of the
        // run without rules.
        let features = |k: usize| rungs::features(&ft, k, period(&pop, i));
        // What the rules should do to this creature: 0 lets it run on, 1 stops
        // it at R1 and 2 stops it at R2 (the codes of `RungTrace::stopped_by`).
        let expected = if flags & rungs::AUDIT != 0 {
            0
        } else if flags & rungs::exempt_bits(0) == 0
            && alive(f, 0)
            && r1.stops(&features(0), ft.band(0))
        {
            1
        } else if flags & rungs::exempt_bits(1) == 0
            && alive(f, 1)
            && r2.stops(&features(1), ft.band(1))
        {
            2
        } else {
            0
        };
        assert_eq!(t.stopped_by(), expected, "creature {i}, flags {flags}");
        match expected {
            0 => {
                spared += 1;
                assert_eq!(r.screened, 0.0, "creature {i}");
                assert_eq!(
                    r.fitness.to_bits(),
                    f.fitness.to_bits(),
                    "creature {i}: a survivor's trial"
                );
                assert_eq!(r.rung_trace().words[2] >> 16, ft.words[2] >> 16);
            }
            k => {
                if k == 1 {
                    stopped1 += 1;
                } else {
                    stopped2 += 1;
                }
                assert!(r.screened > 0.0, "creature {i}");
                assert_eq!(t.steps(), rungs::RUNG_STEPS[k as usize - 1], "creature {i}");
                assert_eq!(
                    r.fitness, r.screen_x,
                    "creature {i}: it keeps its distance there"
                );
                assert!(
                    (r.fitness - ft.distance(k as usize - 1)).abs()
                        <= 0.002 * r.fitness.abs().max(1.0),
                    "creature {i}: {} against {}",
                    r.fitness,
                    ft.distance(k as usize - 1)
                );
            }
        }
    }
    eprintln!("{stopped1} stopped at 1 s, {stopped2} at 2.5 s, {spared} ran on");
    // Each outcome must occur often enough for the checks above to mean
    // something.
    assert!(stopped1 > 20 && stopped2 > 20 && spared > 500);
}

#[test]
#[ignore = "needs the RTX 4060"]
fn an_audit_creature_runs_past_a_screen_that_stops_everyone_else() {
    // The screen is at 2 s and its bar is out of reach, so it stops every
    // creature still running then, except the audit creatures.
    let cfg = Config {
        population: 700,
        duration: 6.0,
        random_seed: false,
        seed: 43,
        screen: Some(Screen::uniform(2.0, f32::INFINITY)),
        ..Config::default()
    };
    let mut pop = evolution::create(&cfg).unwrap();
    pop.flags = (0..pop.genomes.len())
        .map(|i| if i % 5 == 0 { rungs::AUDIT } else { 0 })
        .collect();
    let results = evaluate(&pop, &cfg);
    for (i, r) in results.iter().enumerate() {
        // A trial that ended in a fall by the screen time is never screened.
        let fell_first = r.fall_time > 0.0 && r.fall_time <= 2.0 + 1e-4;
        if pop.flags[i] & rungs::AUDIT != 0 || fell_first {
            assert_eq!(r.screened, 0.0, "creature {i}");
        } else {
            assert!(r.screened > 0.0, "creature {i}");
        }
    }
}

#[test]
#[ignore = "needs the RTX 4060"]
fn a_nursery_creature_is_held_to_the_screen_bar_of_its_own_kind() {
    // Evolved and reshaped creatures face a bar nothing reaches, and young
    // ones a bar that stops nobody.
    let cfg = Config {
        population: 700,
        duration: 6.0,
        random_seed: false,
        seed: 44,
        screen: Some(Screen {
            seconds: 2.0,
            bar: f32::INFINITY,
            young_bar: f32::NEG_INFINITY,
            reshaped_bar: f32::INFINITY,
        }),
        ..Config::default()
    };
    let mut pop = evolution::create(&cfg).unwrap();
    pop.flags = (0..pop.genomes.len())
        .map(|i| match i % 3 {
            0 => 0,
            1 => rungs::YOUNG,
            _ => rungs::RESHAPED,
        })
        .collect();
    let results = evaluate(&pop, &cfg);
    for (i, r) in results.iter().enumerate() {
        // A trial that ended in a fall by the screen time is never screened.
        let fell_first = r.fall_time > 0.0 && r.fall_time <= 2.0 + 1e-4;
        if fell_first || pop.flags[i] == rungs::YOUNG {
            assert_eq!(r.screened, 0.0, "creature {i}");
        } else {
            assert!(r.screened > 0.0, "creature {i}");
        }
    }
}
