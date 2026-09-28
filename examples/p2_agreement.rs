//! Physics v2 on the GPU against the CPU prototype (`EVOLUTION_PHYSICS=2`
//! required): evaluates the same creatures with `physics2::evaluate` and with
//! the v2 kernel on the named Vulkan device, and prints how far their
//! distances and fall times differ, over time spans short and long (chaotic
//! trials drift apart, so the long ones are distributions). Also prints
//! both engines' rates.
//!
//! Usage: cargo run --release --example p2_agreement -- [count] [seconds] [device] [creature.json ...]
use evolution_simulator::{
    config::Config,
    engine::{self, Engine},
    evolution::{self, Creature, Population},
    physics2,
};
use std::time::{Duration, Instant};

fn gpu(pop: &Population, cfg: &Config, device: &str) -> anyhow::Result<(Vec<f32>, Vec<f32>, f64)> {
    let mut engine = engine::gpu_engine(device, 16, evolution_simulator::gpu::DEFAULT_STEP_RANGE)?;
    // A first run builds the pipelines.
    engine.submit(pop.clone(), cfg)?;
    loop {
        if engine.poll()?.is_some() {
            break;
        }
        engine.wait(Duration::from_millis(20));
    }
    let start = Instant::now();
    engine.submit(pop.clone(), cfg)?;
    let done = loop {
        if let Some(done) = engine.poll()? {
            break done;
        }
        engine.wait(Duration::from_millis(20));
    };
    let seconds = start.elapsed().as_secs_f64();
    Ok((
        done.results.iter().map(|r| r.fitness).collect(),
        done.results.iter().map(|r| r.fall_time).collect(),
        seconds,
    ))
}

fn quantile(values: &mut [f32], q: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    if values.is_empty() {
        return f32::NAN;
    }
    values[((values.len() - 1) as f32 * q) as usize]
}

fn main() -> anyhow::Result<()> {
    anyhow::ensure!(physics2::enabled(), "set EVOLUTION_PHYSICS=2");
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let seconds: f32 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(10.0);
    let device = args.get(3).cloned().unwrap_or_else(|| "NVIDIA".into());
    let cfg = Config {
        population: count,
        duration: seconds,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let mut pop = evolution::create(&cfg)?;
    for path in args.iter().skip(4) {
        let creature: Creature = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        pop.push(creature);
    }
    let start = Instant::now();
    let cpu = physics2::evaluate(&pop, &cfg);
    let cpu_seconds = start.elapsed().as_secs_f64();
    let (gpu_fitness, gpu_fall, gpu_seconds) = gpu(&pop, &cfg, &device)?;
    let n = pop.genomes.len();
    let mut gaps: Vec<f32> = Vec::new();
    let (mut same_fall, mut both_fell, mut fell_one) = (0usize, 0usize, 0usize);
    for i in 0..n {
        let (a, b) = (cpu[i].fitness, gpu_fitness[i]);
        if a > -1e19 && b > -1e19 {
            gaps.push((a - b).abs());
        }
        let (fa, fb) = (cpu[i].fall_time > 0.0, gpu_fall[i] > 0.0);
        if fa && fb {
            both_fell += 1;
            if (cpu[i].fall_time - gpu_fall[i]).abs() < 0.5 / 60.0 {
                same_fall += 1;
            }
        } else if fa != fb {
            fell_one += 1;
        }
    }
    let within =
        |t: f32| gaps.iter().filter(|&&g| g <= t).count() as f32 / gaps.len().max(1) as f32;
    let (w1, w10, w100) = (within(0.01), within(0.1), within(1.0));
    let mut sorted = gaps.clone();
    println!(
        "{n} creatures, {seconds} s: distance gap median {:.4} m, p90 {:.4} m, p99 {:.4} m, max {:.3} m; within 1 cm {:.1}%, 10 cm {:.1}%, 1 m {:.1}%",
        quantile(&mut sorted, 0.5),
        quantile(&mut sorted, 0.9),
        quantile(&mut sorted, 0.99),
        quantile(&mut sorted, 1.0),
        100.0 * w1,
        100.0 * w10,
        100.0 * w100
    );
    println!(
        "falls: both {both_fell} (same step {same_fall}), only one engine {fell_one}; CPU {:.0} creatures/s ({} threads), GPU {:.0} creatures/s",
        n as f64 / cpu_seconds,
        rayon::current_num_threads(),
        n as f64 / gpu_seconds
    );
    let mut cpu_best: Vec<f32> = cpu.iter().map(|r| r.fitness).collect();
    let mut gpu_best = gpu_fitness.clone();
    println!(
        "distance: CPU median {:.3} m p90 {:.3} m best {:.2} m; GPU median {:.3} m p90 {:.3} m best {:.2} m",
        quantile(&mut cpu_best, 0.5),
        quantile(&mut cpu_best, 0.9),
        quantile(&mut cpu_best, 1.0),
        quantile(&mut gpu_best, 0.5),
        quantile(&mut gpu_best, 0.9),
        quantile(&mut gpu_best, 1.0)
    );
    for i in count..n {
        println!(
            "{}: CPU {:.2} m (fell {:.2} s), GPU {:.2} m (fell {:.2} s)",
            args[4 + i - count],
            cpu[i].fitness,
            cpu[i].fall_time,
            gpu_fitness[i],
            gpu_fall[i]
        );
    }
    Ok(())
}
