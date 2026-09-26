use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use evolution_simulator::{
    config::Config,
    gpu::Gpu,
    physics, search_benchmark,
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
    /// Compare fixed-seed evolutionary search runs and export morphology, lineage, and timing data.
    SearchBenchmark {
        #[arg(long, value_delimiter = ',', default_value = "38,39,40,41,42")]
        seeds: Vec<u64>,
        #[arg(long, default_value_t = 1000)]
        population: usize,
        #[arg(long, default_value_t = 300)]
        generations: u32,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        duration: Option<f32>,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "1,5,10,20,50,100,150,200"
        )]
        milestones: Vec<f32>,
        #[arg(long, default_value = "benchmarks/search-baseline")]
        output_dir: PathBuf,
        #[arg(long, value_enum, default_value_t = SearchVariant::MorphologyReserve)]
        variant: SearchVariant,
    },
    /// Evaluate a fixed checkpoint population repeatedly (kernel diagnostics).
    EvalBench {
        #[arg(long, default_value = "bench/w3-seed38-100k.evo")]
        checkpoint: PathBuf,
        /// Evaluate only the first N creatures.
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 3)]
        repeat: u32,
        /// Grow each body with structural mutations to at least this many nodes (W4).
        #[arg(long)]
        grow: Option<usize>,
        /// Write per-creature fitness and behavior as little-endian f32 records.
        #[arg(long)]
        dump: Option<PathBuf>,
        /// Compare against an earlier dump and report differences.
        #[arg(long)]
        compare: Option<PathBuf>,
        /// Override the trial duration (diagnostics only).
        #[arg(long)]
        duration: Option<f32>,
        /// Evaluate directly on the named Vulkan device (bodies up to 16 nodes).
        #[arg(long)]
        engine: Option<String>,
    },
    /// Summarize an existing checkpoint's record curve and archive morphology.
    Analyze {
        checkpoint: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        champion: Option<PathBuf>,
    },
}
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum SearchVariant {
    BehaviorOnly,
    #[default]
    MorphologyReserve,
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
                            // The scheduler keeps every engine busy across a whole
                            // generation; smaller batches would add a tail each.
                            let batch = if gpu.async_capable() {
                                e.config.population
                            } else {
                                e.config.batch_size()
                            };
                            let end = (e.evaluated + batch).min(e.config.population);
                            let start = Instant::now();
                            let metrics = gpu.evaluate_with_metrics(
                                &e.population,
                                &(e.evaluated..end).collect::<Vec<_>>(),
                                &e.config,
                            )?;
                            e.evaluation_seconds += start.elapsed().as_secs_f64();
                            for (offset, metric) in metrics.iter().enumerate() {
                                e.scores[e.evaluated + offset] = metric.fitness;
                                e.trial_metrics[e.evaluated + offset] = metric.behavior;
                            }
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
                        Stage::Evaluated | Stage::Ranked | Stage::Selected => {
                            e.archive_batch()?;
                            let s = e.history.last().unwrap();
                            println!(
                                "generation={} best={:.4}m median={:.4}m niches={} qd={:.2} failed={} evaluation={:.3}s",
                                s.generation,
                                s.best,
                                s.median,
                                s.archive_cells,
                                s.qd_score,
                                s.failed,
                                s.seconds
                            );
                        }
                        Stage::Archived => {
                            e.prepare_next_batch()?;
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
                    let batch = e.config.batch_size();
                    for begin in (0..count).step_by(batch) {
                        let end = (begin + batch).min(count);
                        let metrics = gpu.evaluate_with_metrics(
                            &e.population,
                            &(begin..end).collect::<Vec<_>>(),
                            &e.config,
                        )?;
                        for (offset, metric) in metrics.iter().enumerate() {
                            e.scores[begin + offset] = metric.fitness;
                            e.trial_metrics[begin + offset] = metric.behavior;
                        }
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
                    e.archive_batch()?;
                    let failed = e.history.last().unwrap().failed;
                    let bytes = e.population.bytes();
                    e.prepare_next_batch()?;
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
        Some(Action::SearchBenchmark {
            seeds,
            population,
            generations,
            config,
            duration,
            milestones,
            output_dir,
            variant,
        }) => {
            let mut cfg: Config = if let Some(path) = config {
                serde_json::from_reader(std::fs::File::open(path)?)?
            } else {
                Config::default()
            };
            cfg.population = population;
            if let Some(duration) = duration {
                cfg.duration = duration;
            }
            search_benchmark::run(
                &cli.gpu,
                cfg,
                &seeds,
                generations,
                &milestones,
                &output_dir,
                matches!(variant, SearchVariant::MorphologyReserve),
            )
        }
        Some(Action::EvalBench {
            checkpoint,
            limit,
            repeat,
            grow,
            dump,
            compare,
            duration,
            engine,
        }) => {
            let e = storage::load(&checkpoint)?;
            let mut cfg = e.config.clone();
            if let Some(duration) = duration {
                cfg.duration = duration;
            }
            cfg.throughput = true;
            let count = limit.unwrap_or(cfg.population).min(cfg.population);
            let mut population = evolution_simulator::evolution::Population::default();
            for i in 0..count {
                let mut c = e.population.creature(i);
                if let Some(target) = grow {
                    evolution_simulator::evolution::grow_for_benchmark(&mut c, &cfg, 38, target);
                }
                population.push(c);
            }
            cfg.population = count;
            let mut histogram = std::collections::BTreeMap::new();
            let mut steps_nodes = 0u64;
            for g in &population.genomes {
                *histogram.entry(g.node_count).or_insert(0usize) += 1;
                steps_nodes += g.node_count as u64;
            }
            eprintln!(
                "Workload: {} creatures, mean nodes {:.2}, node histogram {:?}",
                count,
                steps_nodes as f64 / count as f64,
                histogram
            );
            // `--engine cpu` runs the CPU SIMD engine alone; another name opens that
            // Vulkan device alone (bodies up to 16 nodes).
            let mut engine: Option<Box<dyn evolution_simulator::engine::Engine>> = match engine {
                Some(name) if name == "cpu" => {
                    Some(Box::new(evolution_simulator::engine::cpu_engine(
                        std::thread::available_parallelism().map_or(4, usize::from),
                    )?))
                }
                Some(name) => Some(Box::new(evolution_simulator::engine::gpu_engine(
                    &name,
                    16,
                    evolution_simulator::gpu::DEFAULT_STEP_RANGE,
                )?)),
                None => None,
            };
            let mut gpu = if engine.is_none() {
                Some(Gpu::new(&cli.gpu)?)
            } else {
                None
            };
            let indices: Vec<usize> = (0..count).collect();
            let batch = cfg.batch_size();
            let mut out = Vec::new();
            for r in 0..repeat {
                let start = Instant::now();
                out.clear();
                for chunk in indices.chunks(batch) {
                    if let Some(engine) = engine.as_mut() {
                        engine.submit(population.subset(chunk), &cfg)?;
                        let done = loop {
                            if let Some(done) = engine.poll()? {
                                break done;
                            }
                            engine.wait(std::time::Duration::from_millis(50));
                        };
                        out.extend(chunk.iter().zip(&done.results).map(|(&i, r)| {
                            evolution_simulator::scheduler::to_metrics(&population, i, r, &cfg)
                        }));
                    } else {
                        out.extend(gpu.as_mut().unwrap().evaluate_with_metrics(
                            &population,
                            chunk,
                            &cfg,
                        )?);
                    }
                }
                let seconds = start.elapsed().as_secs_f64();
                eprintln!(
                    "Repeat {r}: {count} creatures in {seconds:.3} s, {:.0} creatures/s",
                    count as f64 / seconds
                );
            }
            let records: Vec<f32> = out
                .iter()
                .flat_map(|m| {
                    [
                        m.fitness,
                        m.behavior.ground_contact,
                        m.behavior.vertical_oscillation,
                        m.behavior.gait_frequency,
                    ]
                })
                .collect();
            if let Some(path) = dump {
                std::fs::write(&path, bytemuck::cast_slice(&records))?;
            }
            if let Some(path) = compare {
                let bytes = std::fs::read(&path)?;
                let reference: &[f32] = bytemuck::cast_slice(&bytes);
                anyhow::ensure!(reference.len() == records.len(), "Dump size differs");
                let mut exact = 0usize;
                let mut close = 0usize;
                let mut max_fitness_diff = 0f32;
                let mut fitness_diffs = Vec::new();
                for (a, b) in reference.chunks(4).zip(records.chunks(4)) {
                    if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
                        exact += 1;
                    } else if (a[0] - b[0]).abs() <= 1e-3 * a[0].abs().max(1.0) {
                        close += 1;
                    }
                    let d = (a[0] - b[0]).abs();
                    if d.is_finite() {
                        max_fitness_diff = max_fitness_diff.max(d);
                        fitness_diffs.push(d);
                    }
                }
                fitness_diffs.sort_by(f32::total_cmp);
                let n = reference.len() / 4;
                eprintln!(
                    "Compare: {exact}/{n} bit-exact, {close} more within 0.1% fitness, median fitness diff {:.3e}, p99 {:.3e}, max {:.3e}",
                    fitness_diffs[fitness_diffs.len() / 2],
                    fitness_diffs[fitness_diffs.len() * 99 / 100],
                    max_fitness_diff
                );
            }
            Ok(())
        }
        Some(Action::Analyze {
            checkpoint,
            output,
            champion,
        }) => search_benchmark::write_checkpoint_analysis(
            &checkpoint,
            output.as_deref(),
            champion.as_deref(),
        ),
    };
    result.context("Evolution Simulator")
}
