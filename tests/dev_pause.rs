//! A developer pause (`dev_pause`) holds evaluation back and resumes it
//! without changing the search: a paused and resumed run of a fixed seed
//! matches an undisturbed one. The test asks for the pause with a request
//! file, as `tools/pause-game.sh` does.
//!
//! Needs the RTX 4060 and is ignored by default. Run it with
//!
//!     cargo test --release --test dev_pause -- --ignored

use evolution_simulator::{
    config::Config,
    gpu::Gpu,
    worker::{Command, Snapshot, Worker},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

/// A new empty directory for one run's pause files. It is not the game's own
/// pause directory (`dev_pause::dir`), so the test cannot pause the owner's
/// game.
fn pause_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "evolution-dev-pause-{tag}-{}-{:?}",
        std::process::id(),
        Instant::now()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// What the test compares for one generation: its number, the bits of its
/// best, median and mean distance, its failed count and archive cells, the
/// bits of its QD score, and the bits of its percentiles. Floats compare by
/// their bits, so the runs must match exactly.
type Row = (u32, u32, u32, u32, usize, usize, u64, Vec<u32>);

/// Runs `generations` generations of a continuous run on `gpu` and returns a
/// `Row` for each. With `pause`, the run is paused through the request file
/// once the second generation has absorbed its first evaluations, held for a
/// while, and resumed by removing the file. `tag` names the run's pause
/// directory.
fn run(gpu: Gpu, cfg: Config, generations: usize, pause: bool, tag: &str) -> Vec<Row> {
    let dir = pause_dir(tag);
    // The worker only asks this context to repaint, so one without a window
    // will do.
    let worker = Worker::spawn_with_pause_dir(gpu, eframe::egui::Context::default(), dir.clone());
    worker.send(Command::New(cfg));
    worker.send(Command::Run {
        continuous: true,
        guided: false,
    });
    let deadline = Instant::now() + Duration::from_secs(900);
    // True once the pause is done, or from the start when this run takes none.
    let mut pause_done = !pause;
    let mut last: Option<Snapshot> = None;
    let history = loop {
        if let Some(snapshot) = worker.view.lock().unwrap().take() {
            assert!(snapshot.error.is_none(), "{:?}", snapshot.error);
            last = Some(snapshot);
        }
        if let Some(snapshot) = &last {
            if !pause_done && snapshot.generation >= 1 && snapshot.evaluated > 0 {
                pause_done = true;
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

/// Pauses the worker through the request file `pause` in `dir`. It waits for
/// the worker's acknowledgement file `paused`, checks that the engines are
/// closed and that the worker's progress does not move while it is paused,
/// then removes the request and waits for the acknowledgement to go.
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
    // The generation and counts in the newest snapshot after 0.4 s, or None
    // when the worker has published none since the last look.
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

/// Runs the same fixed seed twice on the GPU, once undisturbed and once with a
/// pause and resume, and compares the histories of both bit for bit.
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
