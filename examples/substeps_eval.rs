//! Measurement branch only. Scores archive elites, random first-generation
//! bodies or elite replays on the GPU with the kernel that the EVOLUTION_*
//! settings of `creature_kernel::dev_env` select.
//!
//! Usage:
//!   substeps_eval dist <save> <count> <std|fine> <out.csv> [perturb]
//!   substeps_eval random <count>
//!   substeps_eval slip <save> <count>
use evolution_simulator::{
    config::Config,
    creature_kernel::GpuResult,
    engine::{self, Engine},
    evolution::{self, Population},
    physics::{self, Fidelity},
    storage,
};
use std::time::Duration;

fn score(
    engine: &mut impl Engine,
    pop: &Population,
    cfg: &Config,
) -> anyhow::Result<Vec<GpuResult>> {
    engine.submit(pop.clone(), cfg)?;
    loop {
        if let Some(done) = engine.poll()? {
            return Ok(done.results);
        }
        engine.wait(Duration::from_millis(5));
    }
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    sorted[((sorted.len() - 1) as f32 * q) as usize]
}

fn slip_of(nodes: &[physics::Node], frames: &[Vec<[f32; 2]>], result: &GpuResult, cfg: &Config) -> (f32, f32) {
    let fidelity = cfg.fidelity();
    let settle = fidelity.settle() as usize;
    let terminal = if result.fall_time > 0.0 {
        settle + (result.fall_time * fidelity.rate as f32).round() as usize
    } else {
        frames.len() - 1
    };
    let mass: f32 = nodes.iter().map(|node| node.mass).sum();
    let distance = frames[terminal]
        .iter()
        .zip(nodes)
        .map(|(position, node)| position[0] * node.mass)
        .sum::<f32>()
        / mass;
    let mut slip = 0.0f32;
    let amplitude = physics::terrain_amplitude(cfg.terrain);
    let floor = |position: [f32; 2], radius: f32| {
        let (height, slope) = physics::terrain_with_slope(position[0], amplitude, cfg.slope);
        height + radius * (1.0 + slope * slope).sqrt()
    };
    if cfg.ground && terminal > settle {
        for (j, node) in nodes.iter().enumerate() {
            for t in settle + 2..=terminal {
                let (now, before) = (frames[t][j], frames[t - 1][j]);
                if now[1] <= floor(now, node.radius) + 0.002
                    && before[1] <= floor(before, node.radius) + 0.002
                {
                    slip += (now[0] - before[0]).abs();
                }
            }
        }
    }
    (distance, slip)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).expect("mode").as_str();
    let mut engine = engine::gpu_engine("RTX 4060", 64, evolution_simulator::gpu::DEFAULT_STEP_RANGE)?;
    eprintln!("engine: {}", engine.name());
    match mode {
        "dist" => {
            let e = storage::load(std::path::Path::new(&args[2]))?;
            let count: usize = args[3].parse()?;
            let mut cfg = e.config.clone();
            cfg.screen = None;
            if args[4] == "fine" {
                cfg.fidelity = Some(Fidelity::fine());
            }
            let mut elites: Vec<_> = e.archive.entries.iter().collect();
            elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            let mut pop = Population::default();
            let perturbed = args.get(6).is_some_and(|v| v == "perturb");
            for elite in elites.iter().take(count) {
                let mut c = elite.creature.clone();
                if perturbed {
                    evolution_simulator::scheduler::perturb(&mut c);
                }
                pop.push(c);
            }
            let results = score(&mut engine, &pop, &cfg)?;
            let mut out = String::from("id,archive,distance,fall_time\n");
            for (elite, r) in elites.iter().zip(&results) {
                out += &format!("{},{},{},{}\n", elite.creature.id, elite.fitness, r.fitness, r.fall_time);
            }
            std::fs::write(&args[5], out)?;
            println!("wrote {} rows", results.len());
        }
        "random" => {
            let count: usize = args[2].parse()?;
            let cfg = Config {
                population: count,
                duration: 20.0,
                random_seed: false,
                screen: None,
                ..Config::default()
            };
            let pop = evolution::create(&cfg)?;
            let results = score(&mut engine, &pop, &cfg)?;
            let mut d: Vec<f32> = results
                .iter()
                .map(|r| r.fitness)
                .filter(|f| f.is_finite() && *f > -1e10)
                .collect();
            d.sort_by(f32::total_cmp);
            let over1 = d.iter().filter(|x| **x > 1.0).count();
            let over5 = d.iter().filter(|x| **x > 5.0).count();
            println!(
                "{} random bodies: median {:.3} m, p99 {:.3} m, best {:.3} m, above 1 m {}, above 5 m {}",
                d.len(),
                quantile(&d, 0.5),
                quantile(&d, 0.99),
                quantile(&d, 1.0),
                over1,
                over5
            );
        }
        "slip" => {
            let e = storage::load(std::path::Path::new(&args[2]))?;
            let count: usize = args[3].parse()?;
            engine.publish_replays();
            let mut elites: Vec<_> = e.archive.entries.iter().collect();
            elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            let mut shares = Vec::new();
            let mut slips = Vec::new();
            let mut dists = Vec::new();
            for elite in elites.iter().take(count) {
                let c = &elite.creature;
                let nodes = physics::nodes(c);
                let (frames, result) = engine::replay(c, &e.config);
                let (distance, slip) = slip_of(&nodes, &frames, &result, &e.config);
                shares.push(slip / distance.abs().max(0.01));
                slips.push(slip);
                dists.push(distance);
            }
            shares.sort_by(f32::total_cmp);
            slips.sort_by(f32::total_cmp);
            dists.sort_by(f32::total_cmp);
            println!(
                "{} elite replays: median distance {:.1} m, median slip {:.1} m, median slip per meter {:.3}, p25 {:.3}, p75 {:.3}",
                shares.len(),
                quantile(&dists, 0.5),
                quantile(&slips, 0.5),
                quantile(&shares, 0.5),
                quantile(&shares, 0.25),
                quantile(&shares, 0.75),
            );
        }
        _ => panic!("mode"),
    }
    Ok(())
}
