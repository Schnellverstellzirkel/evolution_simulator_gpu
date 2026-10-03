//! The CPU side of the search at full scale, without a GPU: the production
//! ring absorbs and breeds blocks on a save's archives, with scores that a
//! cheap function of each creature's genes stands in for. It measures the
//! archive and breeding seconds per generation for a given archive layout,
//! and `perf record` on it shows where they go.
//!
//! The stand-in score is the world's top distance for a body whose rhythm
//! period is under 0.5 s (so many bodies tie there, to the last digits, as
//! evolved bodies do) and less for a longer one. Its behavior follows the
//! start pose: the share of nodes near the ground, the clock's frequency, the
//! body's height and its feet. Bodies below half the top distance are
//! screened, as the early screen stops them.
//!
//! Usage: archive_bench <save, or `new` for a new game> <population> <generations> [change-at]
//!
//! With `change-at` the world changes before that generation of the run, to
//! the next autochange step. The stand-in scores ignore the world, so the
//! elites tested again score as before, and the run shows what an archive
//! costs while it refills.
#[path = "diversity_common/mod.rs"]
mod diversity;
use anyhow::Result;
use evolution_simulator::{
    config::Config,
    creature_kernel::RungTrace,
    evolution::Population,
    qd::{EvaluationMetrics, TrialMetrics},
    storage,
};

/// The top distance of the stand-in world.
const CAP: f32 = 34.07;

fn noise(id: u64) -> f32 {
    let mut h = id.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 32;
    (h & 0xff_ffff) as f32 / 0x80_0000 as f32 - 1.0
}

fn evaluate(population: &Population, _: &Config) -> Result<Vec<EvaluationMetrics>> {
    Ok(population
        .genomes
        .iter()
        .map(|g| {
            let nodes = &population.nodes[g.node_start..g.node_start + g.node_count];
            let muscles = &population.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
            let period = muscles.first().map_or(0.5, |m| m.period);
            let low = nodes.iter().map(|n| n.y).fold(f32::INFINITY, f32::min);
            let high = nodes.iter().map(|n| n.y).fold(f32::NEG_INFINITY, f32::max);
            let mean = nodes.iter().map(|n| n.y).sum::<f32>() / nodes.len().max(1) as f32;
            let feet = nodes.iter().filter(|n| n.y < low + 0.1).count() as f32;
            let quality = (1.2 * (-(period - 0.21) / 1.5).exp()).min(1.0);
            // Bodies at the top tie to about seven digits, as evolved ones do.
            let fitness = if quality >= 0.999 {
                CAP * (1.0 + 3e-7 * noise(g.id))
            } else {
                CAP * quality * (0.9 + 0.1 * noise(g.id).abs())
            };
            EvaluationMetrics {
                fitness,
                behavior: TrialMetrics {
                    ground_contact: (1.0 - mean / 0.8).clamp(0.02, 0.98),
                    vertical_oscillation: 0.1,
                    gait_frequency: (1.0 / period.max(0.05)).clamp(0.0, 6.0),
                    mean_height: high - low + 0.1,
                    feet,
                },
                screened: fitness < 0.5 * CAP,
                excluded: false,
                screen_x: fitness * 0.25,
                fine: false,
                trace: RungTrace::default(),
            }
        })
        .collect())
}

/// Processor time of this process (user and system): other work on a shared
/// machine moves it less than the wall clock.
fn cpu_seconds() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage fills the struct it is given.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

fn main() -> Result<()> {
    evolution_simulator::engine::lower_thread_priority();
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(evolution_simulator::engine::rayon_threads())
        .start_handler(|_| evolution_simulator::engine::lower_thread_priority())
        .build_global();
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: archive_bench <save> <population> <generations>");
    let population: usize = args.next().expect("population").parse()?;
    let generations: u32 = args.next().expect("generations").parse()?;
    let change_at: Option<u32> = args.next().map(|n| n.parse()).transpose()?;
    let mut experiment = if path == "new" {
        storage::Experiment::new(Config {
            population,
            random_seed: false,
            seed: 38,
            ..Config::default()
        })?
    } else {
        storage::load_for_population(std::path::Path::new(&path), population)?
    };
    println!(
        "{path}: generation {}, global archive {} cells, islands {}",
        experiment.generation,
        experiment.archive.behavior_count(),
        experiment
            .islands
            .iter()
            .map(|i| i.behavior_count().to_string())
            .collect::<Vec<_>>()
            .join(" "),
    );
    println!("world {}", diversity::world_line(&experiment.config));
    let progress = |experiment: &storage::Experiment| {
        let archives = std::iter::once(&experiment.archive).chain(&experiment.islands);
        archives
            .take(storage::island_count() + 1)
            .enumerate()
            .map(|(k, a)| {
                // Islands 1 to 5 of this list also have a record generation.
                let record = k
                    .checked_sub(1)
                    .and_then(|k| experiment.island_progress.get(k))
                    .map_or(String::new(), |p| format!(" record gen {}", p.1));
                format!(
                    "{k}: {} ways of moving, refined {}{record}",
                    a.movement_count(),
                    a.refined()
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    println!("{}", progress(&experiment));
    let mut total = [0.0f64; 2];
    let mut cpu_total = 0.0;
    for step in 0..generations {
        if change_at == Some(step) {
            let before = experiment.config.clone();
            let mut cfg = before.clone();
            let next = cfg.autochange_step;
            anyhow::ensure!(
                evolution_simulator::environment::apply_autochange_step(&mut cfg, next),
                "autochange step {next} changes nothing"
            );
            cfg.autochange_step = next + 1;
            experiment.update_config_now(cfg)?;
            println!(
                "world change: {}",
                diversity::world_difference(&before, &experiment.config)
            );
        }
        let generation = experiment.generation;
        let started = std::time::Instant::now();
        let cpu_before = cpu_seconds();
        experiment.run_generation(&mut evaluate)?;
        let cpu = cpu_seconds() - cpu_before;
        cpu_total += cpu;
        let [archive, breeding] = std::mem::take(&mut experiment.stage_seconds);
        total[0] += archive;
        total[1] += breeding;
        let changed = experiment
            .archive
            .entries
            .iter()
            .filter(|e| e.improved_generation == generation)
            .count();
        println!(
            "generation {generation}: archive {archive:.2} s, breeding {breeding:.2} s (whole generation {:.1} s with scoring, {cpu:.1} CPU s), global archive {} cells{}, islands refined {}, {changed} changed",
            started.elapsed().as_secs_f64(),
            experiment.archive.behavior_count(),
            if experiment.archive.refined() {
                " (refined)"
            } else {
                ""
            },
            experiment.islands.iter().filter(|i| i.refined()).count(),
        );
    }
    println!("{}", progress(&experiment));
    println!(
        "mean per generation: archive {:.2} s, breeding {:.2} s, {:.1} CPU s",
        total[0] / generations as f64,
        total[1] / generations as f64,
        cpu_total / generations as f64
    );
    println!(
        "global archive: {}",
        diversity::measure(&experiment.archive, &experiment, diversity::Part::All).line()
    );
    println!(
        "within 1% of the best: {}",
        diversity::measure(&experiment.archive, &experiment, diversity::Part::NearBest).line()
    );
    Ok(())
}
