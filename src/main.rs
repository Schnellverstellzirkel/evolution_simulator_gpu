use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use evolution_simulator::{
    config::Config,
    gpu::Gpu,
    physics,
    storage::{self, Experiment, Stage},
    ui,
};
use rayon::prelude::*;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Parser)]
#[command(
    version,
    about = "A GPU accelerated laboratory for evolving walking creatures"
)]
struct Cli {
    #[arg(long, global = true, default_value = "RTX 4060")]
    gpu: String,
    #[command(subcommand)]
    command: Option<Action>,
}
#[derive(Subcommand)]
enum Action {
    /// Evolve without opening a window. Ctrl+C checkpoints the current generation.
    Headless {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        population: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value_t = 10)]
        generations: u32,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, default_value = "runs/latest.evo")]
        checkpoint: PathBuf,
        #[arg(long)]
        duration: Option<f32>,
        #[arg(long)]
        throughput: bool,
    },
    /// Record real GPU and optional CPU timings to CSV.
    Benchmark {
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "1000,100000,1000000,3000000"
        )]
        populations: Vec<usize>,
        #[arg(long, default_value_t = 15.0)]
        duration: f32,
        #[arg(long, default_value_t = 1)]
        generations: u32,
        #[arg(long)]
        cpu: bool,
        #[arg(long, default_value = "runs/benchmark.csv")]
        output: PathBuf,
    },
}
fn main() -> Result<()> {
    // Reserve two logical CPUs for the window system and other desktop applications.
    rayon::ThreadPoolBuilder::new()
        .num_threads(
            std::thread::available_parallelism()
                .map_or(4, usize::from)
                .saturating_sub(2)
                .max(1),
        )
        .build_global()?;
    let cli = Cli::parse();
    let result = match cli.command {
        None => ui::launch(&cli.gpu),
        Some(Action::Headless {
            config,
            population,
            seed,
            generations,
            resume,
            checkpoint,
            duration,
            throughput,
        }) => {
            let mut e = if let Some(path) = resume {
                storage::load(&path)?
            } else {
                let mut cfg: Config = if let Some(path) = config {
                    serde_json::from_reader(std::fs::File::open(path)?)?
                } else {
                    Config::default()
                };
                if let Some(n) = population {
                    cfg.population = n;
                }
                if let Some(s) = seed {
                    cfg.seed = s;
                    cfg.random_seed = false;
                }
                if let Some(t) = duration {
                    cfg.duration = t;
                }
                if throughput {
                    cfg.throughput = true;
                }
                Experiment::new(cfg)?
            };
            if throughput {
                e.config.throughput = true;
            }
            let mut gpu = Gpu::new(&cli.gpu)?;
            eprintln!(
                "GPU: {} | {} creatures | seed {}",
                gpu.name, e.config.population, e.config.seed
            );
            let stop = Arc::new(AtomicBool::new(false));
            let signal = stop.clone();
            ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))?;
            let until = e.generation.saturating_add(generations);
            let run_result: Result<()> = (|| {
                while e.generation < until && !stop.load(Ordering::Relaxed) {
                    match e.stage {
                        Stage::Ready | Stage::Evaluating => {
                            e.stage = Stage::Evaluating;
                            let batch = e.config.batch_size();
                            let end = (e.evaluated + batch).min(e.config.population);
                            let start = Instant::now();
                            let scores = gpu.evaluate(
                                &e.population,
                                &(e.evaluated..end).collect::<Vec<_>>(),
                                &e.config,
                            )?;
                            e.evaluation_seconds += start.elapsed().as_secs_f64();
                            e.scores[e.evaluated..end].copy_from_slice(&scores);
                            e.evaluated = end;
                            if end == e.config.population {
                                e.stage = Stage::Evaluated;
                            } else if end.is_multiple_of(batch * 16) {
                                eprintln!(
                                    "Generation {}: {:.1}%",
                                    e.generation,
                                    100.0 * end as f64 / e.config.population as f64
                                );
                            }
                        }
                        Stage::Evaluated => {
                            e.rank();
                            let s = e.history.last().unwrap();
                            println!(
                                "generation={} best={:.4}m median={:.4}m failed={} evaluation={:.3}s",
                                s.generation, s.best, s.median, s.failed, s.seconds
                            );
                        }
                        Stage::Ranked => e.select(),
                        Stage::Selected => {
                            e.reproduce()?;
                            if e.config.checkpoint_interval > 0
                                && e.generation.is_multiple_of(e.config.checkpoint_interval)
                            {
                                storage::save(&checkpoint, &e)?;
                            }
                        }
                    }
                }
                Ok(())
            })();
            storage::save(&checkpoint, &e)?;
            storage::export_csv(&checkpoint.with_extension("csv"), &e.history)?;
            eprintln!("Saved {}", checkpoint.display());
            run_result
        }
        Some(Action::Benchmark {
            populations,
            duration,
            generations,
            cpu,
            output,
        }) => {
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut csv = csv::Writer::from_path(&output)?;
            csv.write_record([
                "gpu",
                "population",
                "generation",
                "duration_s",
                "creation_s",
                "gpu_evaluation_s",
                "cpu_evaluation_s",
                "generation_s",
                "evaluations_per_s",
                "population_bytes",
                "gpu_allocated_bytes",
                "failed",
            ])?;
            let mut gpu = Gpu::new(&cli.gpu)?;
            for count in populations {
                let cfg = Config {
                    population: count,
                    seed: 38,
                    random_seed: false,
                    duration,
                    throughput: true,
                    ..Default::default()
                };
                let start = Instant::now();
                let mut e = Experiment::new(cfg)?;
                let creation = start.elapsed().as_secs_f64();
                // Warm the pipeline and buffers before collecting GPU execution timings.
                gpu.evaluate(
                    &e.population,
                    &(0..count.min(1024)).collect::<Vec<_>>(),
                    &e.config,
                )?;
                for generation in 0..generations {
                    let total = Instant::now();
                    let start = Instant::now();
                    for begin in (0..count).step_by(16384) {
                        let end = (begin + 16384).min(count);
                        let s = gpu.evaluate(
                            &e.population,
                            &(begin..end).collect::<Vec<_>>(),
                            &e.config,
                        )?;
                        e.scores[begin..end].copy_from_slice(&s);
                    }
                    let gpu_seconds = start.elapsed().as_secs_f64();
                    e.evaluation_seconds = gpu_seconds;
                    e.evaluated = count;
                    let cpu_seconds = if cpu {
                        let start = Instant::now();
                        let result: Vec<_> = (0..count)
                            .into_par_iter()
                            .map(|i| physics::evaluate(&e.population.creature(i), &e.config))
                            .collect();
                        std::hint::black_box(result);
                        Some(start.elapsed().as_secs_f64())
                    } else {
                        None
                    };
                    e.rank();
                    let failed = e.history.last().unwrap().failed;
                    e.select();
                    let bytes = e.population.bytes();
                    e.reproduce()?;
                    let generation_seconds =
                        total.elapsed().as_secs_f64() - cpu_seconds.unwrap_or(0.0);
                    csv.serialize((
                        &gpu.name,
                        count,
                        generation,
                        duration,
                        creation,
                        gpu_seconds,
                        cpu_seconds,
                        generation_seconds,
                        count as f64 / gpu_seconds,
                        bytes,
                        gpu.allocated_bytes,
                        failed,
                    ))?;
                    csv.flush()?;
                    println!(
                        "{} creatures | generation {} | GPU {:.3}s | complete {:.3}s | {:.0}/s | RAM {:.1} MiB | failed {}{}",
                        count,
                        generation,
                        gpu_seconds,
                        generation_seconds,
                        count as f64 / gpu_seconds,
                        bytes as f64 / 1048576.,
                        failed,
                        cpu_seconds
                            .map(|t| format!(" | CPU {t:.3}s ({:.1}×)", t / gpu_seconds))
                            .unwrap_or_default()
                    );
                }
            }
            println!("Benchmark saved to {}", output.display());
            Ok(())
        }
    };
    result.context("Evolution Simulator")
}
