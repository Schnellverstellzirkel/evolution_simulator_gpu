//! The entry point of the program: it sets up the Rayon pool and reads the
//! command line. With no subcommand it opens the game window (`ui::launch`).
//! `headless` evolves without a window and saves a checkpoint and a history
//! CSV. `eval-bench` scores the creatures of a saved game again and again on
//! the GPU, for kernel diagnostics.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use evolution_simulator::{
    config::Config,
    gpu::Gpu,
    ring::Ring,
    storage::{self, Experiment},
    threads, ui,
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

// The command line: the global `--gpu` and an optional subcommand. A `///`
// comment on a field or a variant below is its `--help` text, so the fields
// that have no help text have `//` comments instead.
#[derive(Parser)]
#[command(
    version,
    about = "A GPU accelerated laboratory for evolving walking creatures"
)]
struct Cli {
    // The GPU that scores creatures: the first CUDA device whose name contains
    // this text, ignoring case. Every subcommand takes it.
    #[arg(long, global = true, default_value = "RTX 4060")]
    gpu: String,
    #[command(subcommand)]
    command: Option<Action>,
}
// The subcommands. With none, the game window opens.
#[derive(Subcommand)]
enum Action {
    /// Evolve without opening a window. Ctrl+C checkpoints the current generation.
    Headless {
        // A JSON settings file for a new experiment.
        #[arg(long)]
        config: Option<PathBuf>,
        // Creatures per generation, for a new experiment.
        #[arg(long)]
        population: Option<usize>,
        // A fixed seed for a new experiment. Without it the settings decide,
        // and the default settings take a seed from the clock.
        #[arg(long)]
        seed: Option<u64>,
        // Generations to run. With `resume` it is that many more.
        #[arg(long, default_value_t = 10)]
        generations: u32,
        // A checkpoint to continue. It keeps the settings saved in it, so
        // `config`, `population`, `seed` and `duration` do not apply to it.
        #[arg(long)]
        resume: Option<PathBuf>,
        // Where the checkpoint goes. The history goes beside it as CSV, with
        // the extension `csv`.
        #[arg(long, default_value = "runs/latest.evo")]
        checkpoint: PathBuf,
        // The trial duration in seconds, for a new experiment.
        #[arg(long)]
        duration: Option<f32>,
        // Evaluate in the large batches of throughput mode
        // (`Config::batch_size`). The default settings have it on already. The
        // flag turns it on for a `config` file or a `resume` checkpoint that
        // has it off.
        #[arg(long)]
        throughput: bool,
    },
    /// Evaluate a fixed checkpoint population repeatedly (kernel diagnostics).
    EvalBench {
        // A save whose ring is evaluated. The default is a local file that is
        // not in the repository.
        #[arg(long, default_value = "bench/w3-seed38-100k.evo")]
        checkpoint: PathBuf,
        /// Evaluate only the first N creatures.
        #[arg(long)]
        limit: Option<usize>,
        // How many times the whole population is evaluated.
        #[arg(long, default_value_t = 3)]
        repeat: u32,
        /// Grow each body with structural mutations to at least this many nodes.
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
        /// Evaluate directly on the named GPU, without the scheduler.
        #[arg(long)]
        engine: Option<String>,
        /// Screen like the game: a first pass of standard trials without a
        /// bar sets the screen's bar from its distances, then every repeat
        /// runs with that bar.
        #[arg(long)]
        screened: bool,
        /// Keep only creatures of at most this many nodes (small bodies, for
        /// benchmarks beside another program on the GPU).
        #[arg(long)]
        max_nodes: Option<usize>,
    },
}
fn main() -> Result<()> {
    // `threads::init` reads the process's CPUs before any thread is pinned, so
    // it comes first. The Rayon pool breeds on every CPU but two: the first
    // runs the worker thread and the second the GPU engine thread, so commands
    // and frames always find a core during a breeding burst. Pool threads are
    // SCHED_BATCH at nice 10 (`threads`).
    threads::init();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads::pool_threads())
        .start_handler(|_| threads::pool_thread_start())
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
            // The flag applies to a resumed save too, which keeps its other
            // settings.
            if throughput {
                e.config.throughput = true;
            }
            let mut gpu = Gpu::new(&cli.gpu)?;
            eprintln!(
                "GPU: {} | {} creatures | seed {}",
                gpu.name, e.config.population, e.config.seed
            );
            // Ctrl+C only sets `stop`. The loop below ends after its current
            // step, and the checkpoint is saved after it.
            let stop = Arc::new(AtomicBool::new(false));
            let signal = stop.clone();
            ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed))?;
            let until = e.generation.saturating_add(generations);
            let sched = gpu.sched.as_mut().expect("scheduler");
            let mut ring = Ring::default();
            // A closure, so that an error in the loop still reaches the
            // `ring.stop` and the save below.
            let run_result: Result<()> = (|| {
                ring.start(&mut e, sched);
                while e.generation < until && !stop.load(Ordering::Relaxed) {
                    let started = Instant::now();
                    sched.pump()?;
                    // Waits up to 20 ms for finished work, then absorbs every
                    // finished block.
                    let step = ring.step(&mut e, sched, Duration::from_millis(20), usize::MAX)?;
                    e.evaluation_seconds += started.elapsed().as_secs_f64();
                    if step.generations == 0 {
                        continue;
                    }
                    // A step that ended a generation prints the newest row of
                    // the history. `niches` is the number of filled cells of
                    // the global archive and `qd` is its QD score.
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
                    if e.config.checkpoint_interval > 0
                        && e.generation.is_multiple_of(e.config.checkpoint_interval)
                    {
                        storage::save(&checkpoint, &e)?;
                    }
                }
                Ok(())
            })();
            // The run ends by its generation count, by Ctrl+C or by an error.
            // Each of them saves the checkpoint and the history.
            ring.stop(sched);
            storage::save(&checkpoint, &e)?;
            storage::export_csv(&checkpoint.with_extension("csv"), &e.history)?;
            eprintln!("Saved {}", checkpoint.display());
            run_result
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
            screened,
            max_nodes,
        }) => {
            let e = storage::load(&checkpoint)?;
            let mut cfg = e.config.clone();
            if let Some(duration) = duration {
                cfg.duration = duration;
            }
            cfg.throughput = true;
            // The creatures of the loaded game's ring, which a save does not
            // hold: `load` bred them from the archives. `--max-nodes` leaves
            // out the bodies with more nodes.
            let source: Vec<_> = e
                .blocks
                .iter()
                .flat_map(|b| (0..b.len()).map(move |j| (b, j)))
                .filter(|(b, j)| {
                    max_nodes.is_none_or(|max| b.population.genomes[*j].node_count <= max)
                })
                .collect();
            let count = limit.unwrap_or(source.len()).min(source.len());
            let mut population = evolution_simulator::evolution::Population::default();
            for &(block, j) in &source[..count] {
                let mut c = block.population.creature(j);
                // `--grow` adds nodes with the game's structural mutations.
                // Its random stream comes from the fixed seed 38 and the
                // creature's id.
                if let Some(target) = grow {
                    evolution_simulator::evolution::grow_for_benchmark(&mut c, &cfg, 38, target);
                }
                population.push(c);
            }
            cfg.population = count;
            let mut histogram = std::collections::BTreeMap::new();
            let mut node_sum = 0u64;
            for g in &population.genomes {
                *histogram.entry(g.node_count).or_insert(0usize) += 1;
                node_sum += g.node_count as u64;
            }
            eprintln!(
                "Workload: {} creatures, mean nodes {:.2}, node histogram {:?}",
                count,
                node_sum as f64 / count as f64,
                histogram
            );
            // `--engine <name>` opens that GPU alone, without the scheduler.
            // The 64 asks for bodies up to 64 nodes, and the engine lowers it
            // to the kernel's limit (`kernel::MAX_NODES`).
            let mut engine: Option<Box<dyn evolution_simulator::engine::Engine>> = match engine {
                Some(name) => Some(Box::new(evolution_simulator::engine::gpu_engine(
                    &name, 64,
                )?)),
                None => None,
            };
            let mut gpu = if engine.is_none() {
                Some(Gpu::new(&cli.gpu)?)
            } else {
                None
            };
            let indices: Vec<usize> = (0..count).collect();
            // `--screened`: a save loads with no screen bar, so a first pass
            // runs every trial in full and records each creature's distance at
            // the screen. The bar is the distance that the best share of them
            // (`physics::screen_keep`) reached there, as in the game, and the
            // repeats run with it.
            if screened && let Some(screen) = cfg.screen {
                let gpu = gpu
                    .as_mut()
                    .context("--screened needs the default GPU path")?;
                let start = Instant::now();
                let sample = gpu.sched.as_mut().context("scheduler")?.evaluate(
                    &population,
                    &indices,
                    &cfg,
                )?;
                let first: Vec<f32> = sample.iter().map(|m| m.screen_x).collect();
                let bar = evolution_simulator::physics::screen_bar(
                    first.iter().copied(),
                    evolution_simulator::physics::screen_keep(),
                );
                let passed = first.iter().filter(|&&x| x >= bar).count();
                eprintln!(
                    "Screen bar from a {:.3} s pass: {bar:.3} m at {} s ({passed} pass)",
                    start.elapsed().as_secs_f64(),
                    screen.seconds,
                );
                // How well the bar keeps the creatures that end best in the
                // full trials of this pass.
                let mut order: Vec<usize> = (0..sample.len()).collect();
                order.sort_by(|&a, &b| sample[b].fitness.total_cmp(&sample[a].fitness));
                for share in [0.001, 0.01, 0.1] {
                    let top = &order[..((sample.len() as f64 * share) as usize).max(1)];
                    let kept = top.iter().filter(|&&i| first[i] >= bar).count();
                    eprintln!(
                        "Final top {:.1}% ({} creatures): {kept} pass the bar",
                        share * 100.0,
                        top.len()
                    );
                }
                cfg.screen = Some(evolution_simulator::physics::Screen::uniform(
                    screen.seconds,
                    bar,
                ));
            }
            let batch = cfg.batch_size();
            // The metrics of the last repeat, for `--dump` and `--compare`.
            let mut out = Vec::new();
            let rate = cfg.fidelity().rate as f64;
            // The kernel runs no settling steps before a trial. It repeats the
            // start pose, so nothing is added to the step counts for it.
            let settle = 0.0;
            for r in 0..repeat {
                let start = Instant::now();
                out.clear();
                // Physics steps the creatures ran in this repeat. Only the
                // `--engine` path counts them, because only it gets the raw
                // results, which hold the seconds each creature ran.
                let mut steps = 0f64;
                for chunk in indices.chunks(batch) {
                    // With `--engine` the engine takes the chunk directly.
                    // Otherwise `Gpu` evaluates it through the scheduler.
                    if let Some(engine) = engine.as_mut() {
                        engine.submit(population.subset(chunk), &cfg)?;
                        let done = loop {
                            if let Some(done) = engine.poll()? {
                                break done;
                            }
                            engine.wait(Duration::from_millis(50));
                        };
                        for r in &done.results {
                            // The seconds this creature ran: until it fell,
                            // until the screen stopped it, or the whole trial.
                            let ended = if r.fall_time > 0.0 {
                                r.fall_time
                            } else if r.screened > 0.0 {
                                r.screened
                            } else {
                                cfg.duration
                            };
                            steps += settle + f64::from(ended) * rate;
                        }
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
                if steps > 0.0 {
                    eprintln!(
                        "  {:.0} creature-steps/s ({:.0} steps per creature, settling included)",
                        steps / seconds,
                        steps / count as f64
                    );
                }
            }
            // Four f32s per creature, in this order: fitness, ground contact,
            // vertical oscillation and gait frequency. `--dump` writes them and
            // `--compare` reads them back.
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
                    // A creature is bit-exact when all four values match bit
                    // for bit. Otherwise it is close when its fitness is within
                    // 0.1%, or within 1 mm when the fitness is under 1 m.
                    if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
                        exact += 1;
                    } else if (a[0] - b[0]).abs() <= 1e-3 * a[0].abs().max(1.0) {
                        close += 1;
                    }
                    // The fitness differences of the creatures that have a
                    // finite one, for the median, the 99th percentile and the
                    // maximum below.
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
    };
    result.context("Evolution Simulator")
}
