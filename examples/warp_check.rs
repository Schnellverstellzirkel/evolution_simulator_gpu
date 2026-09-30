//! Checks of the GPU physics on a save: the best archive elites scored at
//! the standard rate and at twice the rate, and the distances of a random
//! first generation. A gait that holds under twice the rate does not live on
//! the step size, and random bodies must not move far on their own.
//!
//! Usage: warp_check <save.evo | dump.bin> [elites] [random bodies]
use evolution_simulator::{
    config::Config,
    creature_kernel::GpuResult,
    engine::{self, Engine},
    evolution::{self, Population},
    physics::Fidelity,
    storage,
};
use std::time::Duration;

fn evaluate(
    engine: &mut impl Engine,
    pop: Population,
    cfg: &Config,
) -> anyhow::Result<Vec<GpuResult>> {
    engine.submit(pop, cfg)?;
    loop {
        if let Some(done) = engine.poll()? {
            return Ok(done.results);
        }
        engine.wait(Duration::from_millis(5));
    }
}

fn quantile(values: &[f32], q: f32) -> f32 {
    let mut sorted: Vec<f32> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > -1e10)
        .collect();
    sorted.sort_by(f32::total_cmp);
    if sorted.is_empty() {
        return f32::NAN;
    }
    sorted[((sorted.len() - 1) as f32 * q) as usize]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("save path");
    let top: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(100);
    let random: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    // The best elites: (creature, trial settings, archive distance).
    let best: Vec<(evolution_simulator::evolution::Creature, Config, f32)> =
        if path.ends_with(".bin") {
            // A creature dump (settings, population, elites) of a save this game
            // no longer reads.
            type Dump = (
                Config,
                Population,
                Vec<(evolution_simulator::evolution::Creature, Config, f32)>,
            );
            let (_, _, elites): Dump = bincode::deserialize(&std::fs::read(path)?)?;
            elites
        } else {
            let experiment = storage::load(std::path::Path::new(path))?;
            let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
            elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            elites
                .iter()
                .map(|e| {
                    let (creature, cfg) = e.replay_of(&experiment.config);
                    (creature, cfg, e.fitness)
                })
                .collect()
        };
    let mut engine = engine::gpu_engine("RTX 4060", 64)?;
    eprintln!("engine: {}", engine.name());
    let mut pop = Population::default();
    let mut archive = Vec::new();
    let mut cfg = Config::default();
    for (creature, elite_cfg, fitness) in best.into_iter().take(top) {
        cfg = elite_cfg;
        pop.push(creature);
        archive.push(fitness);
    }
    cfg.screen = None;
    cfg.fidelity = None;
    let standard = evaluate(&mut engine, pop.clone(), &cfg)?;
    let double = evaluate(
        &mut engine,
        pop.clone(),
        &Config {
            fidelity: Some(Fidelity {
                rate: 2 * Fidelity::standard().rate,
                ..Fidelity::standard()
            }),
            ..cfg.clone()
        },
    )?;
    // The same kernel at four times the rate is the reference.
    let reference = evaluate(
        &mut engine,
        pop.clone(),
        &Config {
            fidelity: Some(Fidelity::fine()),
            ..cfg.clone()
        },
    )?;
    {
        let ratios: Vec<f32> = standard
            .iter()
            .zip(&reference)
            .map(|(s, r)| s.fitness / r.fitness.max(0.05))
            .collect();
        let close = standard
            .iter()
            .zip(&reference)
            .filter(|(s, r)| (s.fitness - r.fitness).abs() <= 0.1 * r.fitness.abs().max(0.5))
            .count();
        let fell = standard.iter().filter(|r| r.fall_time > 0.0).count();
        let fell_early = standard
            .iter()
            .filter(|r| r.fall_time > 0.0 && r.fall_time <= 0.5)
            .count();
        let reference_fell = reference.iter().filter(|r| r.fall_time > 0.0).count();
        println!(
            "against 4x rate: median ratio {:.3}, {close} of {} within 10%, {fell} fall at 60 Hz ({fell_early} within 0.5 s), {reference_fell} at 4x",
            quantile(&ratios, 0.5),
            standard.len()
        );
    }
    let mut kept = 0;
    let mut ratios = Vec::new();
    for (k, ((a, s), d)) in archive.iter().zip(&standard).zip(&double).enumerate() {
        let ratio = d.fitness / s.fitness;
        ratios.push(ratio);
        if (ratio - 1.0).abs() <= 0.2 {
            kept += 1;
        }
        if k < 20 {
            println!(
                "#{:>3}: archive {:7.2} m, 60 Hz {:7.2} m (fell {:5.2} s), 120 Hz {:7.2} m (fell {:5.2} s)",
                k + 1,
                a,
                s.fitness,
                s.fall_time,
                d.fitness,
                d.fall_time
            );
        }
    }
    let s_dist: Vec<f32> = standard.iter().map(|r| r.fitness).collect();
    let walkers: Vec<(f32, f32)> = standard
        .iter()
        .zip(&double)
        .filter(|(s, _)| s.fitness > 5.0)
        .map(|(s, d)| (s.fitness, d.fitness))
        .collect();
    let walker_kept = walkers
        .iter()
        .filter(|(s, d)| (d / s - 1.0).abs() <= 0.2)
        .count();
    println!(
        "{} elites pass 5 m at 60 Hz, {walker_kept} of them within 20% at 120 Hz, sum {:.1} m at 60 Hz and {:.1} m at 120 Hz",
        walkers.len(),
        walkers.iter().map(|w| w.0).sum::<f32>(),
        walkers.iter().map(|w| w.1).sum::<f32>()
    );
    let d_dist: Vec<f32> = double.iter().map(|r| r.fitness).collect();
    println!(
        "top {top}: 60 Hz median {:.2} m best {:.2} m; 120 Hz median {:.2} m best {:.2} m; 120/60 ratio median {:.3}; {kept} of {} within 20%",
        quantile(&s_dist, 0.5),
        quantile(&s_dist, 1.0),
        quantile(&d_dist, 0.5),
        quantile(&d_dist, 1.0),
        quantile(&ratios, 0.5),
        standard.len()
    );

    if random > 0 {
        let fresh_cfg = Config {
            population: random,
            random_seed: false,
            screen: None,
            ..Config::default()
        };
        let fresh = evolution::create(&fresh_cfg)?;
        let results = evaluate(&mut engine, fresh, &fresh_cfg)?;
        let distances: Vec<f32> = results.iter().map(|r| r.fitness).collect();
        let fell = results.iter().filter(|r| r.fall_time > 0.0).count();
        println!(
            "{random} random bodies, {} s: median {:.2} m, 99% {:.2} m, best {:.2} m, {fell} fell",
            fresh_cfg.duration,
            quantile(&distances, 0.5),
            quantile(&distances, 0.99),
            quantile(&distances, 1.0)
        );
    }
    Ok(())
}
