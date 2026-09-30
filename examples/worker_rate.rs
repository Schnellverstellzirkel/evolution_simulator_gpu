//! Runs the game's worker headless (the ring on the GPU) and prints the end
//! to end creature rate, the peak resident memory, and a digest of every
//! generation's statistics, which two runs of one seed must share.
//! Usage: worker_rate [population] [generations] [seed]
use evolution_simulator::{
    config::Config,
    gpu::Gpu,
    worker::{Command, Worker},
};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let arg = |n: usize, d: u64| {
        std::env::args()
            .nth(n)
            .map_or(d, |v| v.parse().expect("number"))
    };
    let population = arg(1, 1_000_000) as usize;
    let generations = arg(2, 12) as usize;
    let seed = arg(3, 38);
    let gpu = Gpu::new("RTX 4060")?;
    // A measurement must not pause itself when it runs under
    // tools/pause-game.sh, so it watches a private pause directory.
    let pause_dir = std::env::temp_dir().join(format!("worker-rate-{}", std::process::id()));
    let worker = Worker::spawn_with_pause_dir(gpu, eframe::egui::Context::default(), pause_dir);
    worker.send(Command::New(Config {
        population,
        seed,
        random_seed: false,
        checkpoint_interval: 0,
        ..Config::default()
    }));
    worker.send(Command::Run {
        continuous: true,
        guided: false,
    });
    let started = Instant::now();
    let mut marks: Vec<(usize, Instant)> = Vec::new();
    let history = loop {
        if let Some(snapshot) = worker.view.lock().unwrap().take() {
            anyhow::ensure!(snapshot.error.is_none(), "{:?}", snapshot.error);
            let n = snapshot.history.len();
            if marks.last().is_none_or(|m| m.0 != n) {
                marks.push((n, Instant::now()));
                eprintln!(
                    "{:6.1} s: {} generations, best {:.2} m",
                    started.elapsed().as_secs_f64(),
                    n,
                    snapshot.history.last().map_or(0.0, |s| s.best)
                );
            }
            if n >= generations {
                break snapshot.history.clone();
            }
        }
        anyhow::ensure!(started.elapsed() < Duration::from_secs(1800), "too slow");
        std::thread::sleep(Duration::from_millis(20));
    };
    // Rate over the generations after the first two (warm-up).
    let first = marks.iter().find(|m| m.0 == 3).map(|m| m.1);
    let last = marks.last().map(|m| (m.0, m.1));
    if let (Some(first), Some((n, last))) = (first, last) {
        let done = (n - 3) * population;
        println!(
            "worker_rate: population {population}, generations 3 to {n}: {:.0} creatures/s, {:.2} s per generation",
            done as f64 / last.duration_since(first).as_secs_f64(),
            last.duration_since(first).as_secs_f64() / (n - 3) as f64
        );
    }
    // Every generation's statistics, bit for bit.
    use std::hash::{Hash, Hasher};
    let mut digest = std::collections::hash_map::DefaultHasher::new();
    for s in history.iter().take(generations) {
        (
            s.generation,
            s.best.to_bits(),
            s.median.to_bits(),
            s.mean.to_bits(),
            s.failed,
            s.archive_cells,
            s.qd_score.to_bits(),
        )
            .hash(&mut digest);
        for p in &s.percentiles {
            p.to_bits().hash(&mut digest);
        }
        println!(
            "worker_rate: generation {} best {:.3} m, median {:.3} m, cells {}, qd {:.1}",
            s.generation, s.best, s.median, s.archive_cells, s.qd_score
        );
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .map_or(f64::NAN, |kb| kb / 1048576.0)
    };
    println!(
        "worker_rate: RSS {:.2} GB, peak RSS {:.2} GB, statistics digest {:016x}",
        field("VmRSS:"),
        field("VmHWM:"),
        digest.finish()
    );
    worker.send(Command::Shutdown);
    Ok(())
}
