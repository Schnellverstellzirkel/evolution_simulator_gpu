//! Backlog item 48: what the CPU engine's whole-group early exit costs and
//! saves. `EVOLUTION_EARLY_EXIT` stops a SIMD group once every real lane has
//! fallen or failed. This example times `cpu_engine::evaluate` with and
//! without the flag on the same first-generation random population, on the
//! calm default world and on a harsh one (3 g, a shrunken muscle energy
//! store). It reports creatures/s, the ratio of each paired pass, the share
//! of SIMD groups that exited early, the share of lanes that fell, and the
//! share of configured physics steps the exited groups skipped.
//!
//! Usage: cargo run --release --example early_exit_cost [count] [seconds] [passes]
use evolution_simulator::{config::Config, cpu_engine, evolution};
use std::sync::atomic::Ordering;
use std::time::Instant;

struct Pass {
    rate: f64,
    seconds: f64,
    exit_share: f64,
    fallen_share: f64,
    skipped_share: f64,
}

fn timed(early: bool, pop: &evolution::Population, cfg: &Config) -> Pass {
    unsafe {
        if early {
            std::env::set_var("EVOLUTION_EARLY_EXIT", "1");
        } else {
            std::env::remove_var("EVOLUTION_EARLY_EXIT");
        }
    }
    cpu_engine::EARLY_EXIT_GROUPS.store(0, Ordering::Relaxed);
    cpu_engine::EARLY_EXIT_GROUPS_TOTAL.store(0, Ordering::Relaxed);
    cpu_engine::EARLY_EXIT_TICKS.store(0, Ordering::Relaxed);
    cpu_engine::EARLY_EXIT_TICKS_FULL.store(0, Ordering::Relaxed);
    let start = Instant::now();
    let results = cpu_engine::evaluate(pop, cfg);
    let seconds = start.elapsed().as_secs_f64();
    let exited = cpu_engine::EARLY_EXIT_GROUPS.load(Ordering::Relaxed);
    let groups = cpu_engine::EARLY_EXIT_GROUPS_TOTAL.load(Ordering::Relaxed);
    let ticks = cpu_engine::EARLY_EXIT_TICKS.load(Ordering::Relaxed);
    let ticks_full = cpu_engine::EARLY_EXIT_TICKS_FULL.load(Ordering::Relaxed);
    let fallen = results.iter().filter(|r| r.fall_time > 0.0).count();
    Pass {
        rate: results.len() as f64 / seconds,
        seconds,
        exit_share: if groups == 0 {
            0.0
        } else {
            exited as f64 / groups as f64
        },
        fallen_share: fallen as f64 / results.len().max(1) as f64,
        skipped_share: if ticks_full == 0 {
            0.0
        } else {
            1.0 - ticks as f64 / ticks_full as f64
        },
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

fn main() {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let duration: f32 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20.0);
    let passes: usize = std::env::args()
        .nth(3)
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let calm = Config {
        population: count,
        duration,
        random_seed: false,
        ..Config::default()
    };
    let mut harsh = calm.clone();
    harsh.gravity = 3.0 * 9.8;
    harsh.muscle_energy = 0.35;
    let pop = evolution::create(&calm).unwrap();
    // Warm caches and threads before the timed passes.
    cpu_engine::evaluate(&pop, &calm);
    println!("{count} bodies, {duration} s trials, {passes} paired passes, thin-LTO release build");
    for (name, cfg) in [("calm", &calm), ("harsh 3g/0.35 energy", &harsh)] {
        let mut off = Vec::new();
        let mut on = Vec::new();
        let mut ratios = Vec::new();
        for pass in 0..passes {
            // Alternating order keeps a slow pass from landing on the same
            // side every time.
            let order = if pass % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            let mut pair = [0.0; 2];
            for early in order {
                let p = timed(early, &pop, cfg);
                println!(
                    "{name} pass {pass} exit={} {:.1}/s ({:.3} s) exited {:.1}% of groups, {:.1}% lanes fallen, {:.1}% steps skipped",
                    early as u8,
                    p.rate,
                    p.seconds,
                    p.exit_share * 100.0,
                    p.fallen_share * 100.0,
                    p.skipped_share * 100.0
                );
                if early {
                    on.push(p.rate);
                    pair[1] = p.rate;
                } else {
                    off.push(p.rate);
                    pair[0] = p.rate;
                }
            }
            ratios.push(pair[1] / pair[0]);
        }
        println!(
            "{name}: median paired ratio {:.3}, best {:.1}/s with the exit, {:.1}/s without",
            median(ratios.clone()),
            on.iter().copied().fold(f64::MIN, f64::max),
            off.iter().copied().fold(f64::MIN, f64::max),
        );
    }
}
