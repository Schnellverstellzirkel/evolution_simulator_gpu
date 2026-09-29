//! Runs the game's worker headless (steady loop, GPU) and prints the end to
//! end creature rate. Usage: worker_rate [population] [generations] [seed]
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
    let worker = Worker::spawn(gpu, eframe::egui::Context::default());
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
    loop {
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
                break;
            }
        }
        anyhow::ensure!(started.elapsed() < Duration::from_secs(1800), "too slow");
        std::thread::sleep(Duration::from_millis(20));
    }
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
    worker.send(Command::Shutdown);
    Ok(())
}
