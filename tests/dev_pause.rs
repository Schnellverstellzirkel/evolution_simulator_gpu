//! A developer pause (`dev_pause`) holds evaluation back and resumes it
//! without changing the search: a paused and resumed run of a fixed seed
//! matches an undisturbed one.

use evolution_simulator::{
    config::Config,
    gpu::Gpu,
    scheduler::Scheduler,
    worker::{Command, Snapshot, Worker},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

fn pause_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "evolution-dev-pause-{tag}-{}-{:?}",
        std::process::id(),
        Instant::now()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

type Row = (u32, u32, u32, u32, usize, usize, u64, Vec<u32>);

/// Runs `generations` generations of a continuous run. With `pause`, the
/// run is paused through the request file once the second generation is
/// under way, held for a while, and resumed by removing the file.
fn run(gpu: Gpu, cfg: Config, generations: usize, pause: bool, tag: &str) -> Vec<Row> {
    let dir = pause_dir(tag);
    let worker = Worker::spawn_with_pause_dir(gpu, eframe::egui::Context::default(), dir.clone());
    worker.send(Command::New(cfg));
    worker.send(Command::Run {
        continuous: true,
        guided: false,
    });
    let deadline = Instant::now() + Duration::from_secs(900);
    let mut paused = !pause;
    let mut last: Option<Snapshot> = None;
    let history = loop {
        if let Some(snapshot) = worker.view.lock().unwrap().take() {
            assert!(snapshot.error.is_none(), "{:?}", snapshot.error);
            last = Some(snapshot);
        }
        if let Some(snapshot) = &last {
            if !paused && snapshot.generation >= 1 && snapshot.evaluated > 0 {
                paused = true;
                hold(&worker, &dir);
                last = None;
                continue;
            }
            if snapshot.history.len() >= generations {
                break snapshot.history.clone();
            }
        }
        assert!(Instant::now() < deadline, "the run did not advance");
        std::thread::sleep(Duration::from_millis(20));
    };
    worker.send(Command::Shutdown);
    drop(worker);
    let _ = std::fs::remove_dir_all(&dir);
    history[..generations]
        .iter()
        .map(|s| {
            (
                s.generation,
                s.best.to_bits(),
                s.median.to_bits(),
                s.mean.to_bits(),
                s.failed,
                s.archive_cells,
                s.qd_score.to_bits(),
                s.percentiles.iter().map(|p| p.to_bits()).collect(),
            )
        })
        .collect()
}

/// Pauses the worker through the request file, checks that nothing is
/// evaluated while it is paused, and resumes it.
fn hold(worker: &Worker, dir: &std::path::Path) {
    let asked = Instant::now();
    std::fs::write(dir.join("pause"), "test").unwrap();
    while !dir.join("paused").exists() {
        assert!(
            asked.elapsed() < Duration::from_secs(120),
            "the pause was not acknowledged"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let closed = asked.elapsed();
    assert!(worker.dev_pause.view().is_some_and(|v| v.closed));
    let progress = |worker: &Worker| {
        std::thread::sleep(Duration::from_millis(400));
        worker
            .view
            .lock()
            .unwrap()
            .take()
            .map(|s| (s.generation, s.completed, s.evaluated))
    };
    let before = progress(worker);
    let after = progress(worker);
    if let (Some(before), Some(after)) = (before, after) {
        assert_eq!(before, after, "the run advanced while paused");
    }
    std::fs::remove_file(dir.join("pause")).unwrap();
    let removed = Instant::now();
    while dir.join("paused").exists() {
        assert!(removed.elapsed() < Duration::from_secs(60), "no resume");
        std::thread::sleep(Duration::from_millis(10));
    }
    eprintln!(
        "Developer pause: closed after {:.2} s, resumed after {:.2} s",
        closed.as_secs_f64(),
        removed.elapsed().as_secs_f64()
    );
}

fn cpu() -> Gpu {
    let sched = Scheduler::cpu_only(2).unwrap();
    Gpu {
        name: sched.names(),
        allocated_bytes: 0,
        sched: Some(sched),
        startup_warning: None,
    }
}

#[test]
fn a_paused_cpu_run_matches_an_undisturbed_one() {
    let cfg = Config {
        population: 24_576,
        duration: 2.0,
        seed: 11,
        random_seed: false,
        checkpoint_interval: 0,
        ..Config::default()
    };
    let undisturbed = run(cpu(), cfg.clone(), 3, false, "cpu-a");
    let paused = run(cpu(), cfg, 3, true, "cpu-b");
    assert_eq!(undisturbed, paused, "the pause changed the search");
}

#[test]
#[ignore = "requires a GPU"]
fn a_paused_gpu_run_matches_an_undisturbed_one() {
    let cfg = Config {
        population: 100_000,
        duration: 10.0,
        seed: 38,
        random_seed: false,
        checkpoint_interval: 0,
        ..Config::default()
    };
    let gpu = || Gpu::new("RTX 4060").unwrap();
    let undisturbed = run(gpu(), cfg.clone(), 6, false, "gpu-a");
    let paused = run(gpu(), cfg, 6, true, "gpu-b");
    assert_eq!(undisturbed, paused, "the pause changed the search");
}
