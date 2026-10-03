//! Runs the game's worker headless (the ring on the GPU) and prints the end
//! to end creature rate, the peak resident memory, the ring's shape, how
//! long the champion's replay took to record (asked every 5 s, as a player
//! clicking it), and a digest of every generation's statistics, which two
//! runs of one seed must share.
//! With a fifth argument above 0 each press applies the next world preset
//! (rough hills, icy slope, swamp, desert, obstacle course, heavy world, then
//! the calm world) instead: most of them are several effects away, so their
//! kernels are not ready yet.
//! With a sixth argument the game opens that save instead of a new one (its
//! population, world and archives), and `generations` counts from the saved
//! generation: this measures an evolved population of big bodies.
//! Usage: worker_rate [population] [generations] [seed] [button seconds] [presets] [save]
//! With button seconds above 0 an effect button is pressed that often: wind,
//! mud, water, ice patches, gaps and hurdles go on one after another, then
//! off in reverse, so every press is one level away from the world before it.
//! The stage log's world_change_discarded column shows what a world change
//! throws away and its kernel_wait_seconds column what it waited for kernels;
//! each press prints the kernel wait until the next one. The search then
//! differs from a run without.
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
    let button = arg(4, 0);
    let presets = arg(5, 0) > 0;
    let save = std::env::args().nth(6).map(std::path::PathBuf::from);
    // Generations of the save, so the count and the rate start from there.
    let base = match &save {
        Some(path) => evolution_simulator::storage::check(path)?.generation as usize,
        None => 0,
    };
    let population = match &save {
        Some(path) => evolution_simulator::storage::check(path)?.population as usize,
        None => population,
    };
    let gpu = Gpu::new("RTX 4060")?;
    // A measurement must not pause itself when it runs under
    // tools/pause-game.sh, so it watches a private pause directory.
    let pause_dir = std::env::temp_dir().join(format!("worker-rate-{}", std::process::id()));
    // The worker asks its context for a repaint at every snapshot, and the
    // context keeps each request until a frame runs. The game runs one every
    // 16 ms. Here the loop below runs empty ones, so the pending requests do
    // not pile up and fault in fresh memory.
    let ctx = eframe::egui::Context::default();
    let worker = Worker::spawn_with_pause_dir(gpu, ctx.clone(), pause_dir);
    match &save {
        Some(path) => worker.send(Command::Load(path.clone())),
        None => worker.send(Command::New(Config {
            population,
            seed,
            random_seed: false,
            checkpoint_interval: 0,
            ..Config::default()
        })),
    }
    worker.send(Command::Run {
        continuous: true,
        guided: false,
    });
    let started = Instant::now();
    let mut marks: Vec<(usize, Instant)> = Vec::new();
    // Minor page faults of the process at each generation mark, and the major
    // ones (a page read back from swap, which another program's memory use
    // can cause) kept apart.
    let stat_fields = || -> (u64, u64) {
        let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
        // Fields after the command name: minflt is field 10, majflt field 12.
        let rest = stat.rsplit_once(')').map_or("", |r| r.1);
        let f: Vec<&str> = rest.split_whitespace().collect();
        let n = |i: usize| f.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        (n(7), n(9))
    };
    let mut fault_marks: Vec<u64> = Vec::new();
    // Resident pages at each mark: faults beyond the growth of the resident
    // set faulted memory that was given back and faulted in again.
    let resident = || -> u64 {
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse().ok())
            .unwrap_or(0)
    };
    let mut resident_marks: Vec<u64> = Vec::new();
    let mut major_marks: Vec<u64> = Vec::new();
    // The champion's replay, asked every 5 s from its own thread like the
    // UI's replay thread.
    type Champion = std::sync::Arc<(
        evolution_simulator::evolution::Creature,
        evolution_simulator::config::Config,
    )>;
    let champion: std::sync::Arc<std::sync::Mutex<Option<Champion>>> = Default::default();
    let replay_seconds: std::sync::Arc<std::sync::Mutex<Vec<f64>>> = Default::default();
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let replays = {
        let (champion, replay_seconds, done) =
            (champion.clone(), replay_seconds.clone(), done.clone());
        std::thread::spawn(move || {
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_secs(5));
                let Some(c) = champion.lock().unwrap().clone() else {
                    continue;
                };
                let asked = Instant::now();
                if evolution_simulator::engine::replay(&c.0, &c.1, Duration::from_secs(60))
                    .is_some()
                {
                    replay_seconds
                        .lock()
                        .unwrap()
                        .push(asked.elapsed().as_secs_f64());
                }
            }
        })
    };
    let mut last_button = Instant::now();
    const BUTTONS: [&str; 6] = ["Wind", "Mud", "Water", "Ice patches", "Gaps", "Hurdles"];
    let mut presses = 0usize;
    let mut wait_at_press = evolution_simulator::cuda_engine::kernel_wait_seconds();
    let history = loop {
        if let Some(snapshot) = worker.view.lock().unwrap().take() {
            anyhow::ensure!(snapshot.error.is_none(), "{:?}", snapshot.error);
            if snapshot.champion.is_some() {
                *champion.lock().unwrap() = snapshot.champion.clone();
            }
            if button > 0 && last_button.elapsed() >= Duration::from_secs(button) {
                last_button = Instant::now();
                let wait = evolution_simulator::cuda_engine::kernel_wait_seconds();
                if presses > 0 {
                    eprintln!(
                        "          kernel wait since press {presses}: {:.2} s",
                        wait - wait_at_press
                    );
                }
                wait_at_press = wait;
                let mut config = snapshot.config.clone();
                if presets {
                    let all = &evolution_simulator::environment::PRESETS;
                    let step = presses % (all.len() + 1);
                    if step < all.len() {
                        all[step].apply(&mut config);
                        eprintln!("press {}: {}", presses + 1, all[step].name);
                    } else {
                        for effect in &evolution_simulator::environment::EFFECTS {
                            if effect.name != "Autochange environment" {
                                effect.set_level(&mut config, effect.calm);
                            }
                        }
                        eprintln!("press {}: calm world", presses + 1);
                    }
                } else {
                    let step = presses % (2 * BUTTONS.len());
                    let (name, level) = if step < BUTTONS.len() {
                        (BUTTONS[step], 1)
                    } else {
                        (BUTTONS[2 * BUTTONS.len() - 1 - step], 0)
                    };
                    let effect = evolution_simulator::environment::EFFECTS
                        .iter()
                        .find(|e| e.name == name)
                        .expect("effect");
                    effect.set_level(&mut config, level);
                    eprintln!("press {}: {name} level {level}", presses + 1);
                }
                presses += 1;
                worker.send(Command::Configure(config));
            }
            let n = snapshot.history.len();
            // A loaded game shows an empty history until the save is read.
            if n < base {
                continue;
            }
            if marks.last().is_none_or(|m| m.0 != n) {
                marks.push((n, Instant::now()));
                let (minor, major) = stat_fields();
                eprintln!(
                    "{:6.1} s: {} generations, best {:.2} m, {} minor page faults since the last generation",
                    started.elapsed().as_secs_f64(),
                    n,
                    snapshot.history.last().map_or(0.0, |s| s.best),
                    minor - fault_marks.last().copied().unwrap_or(0)
                );
                fault_marks.push(minor);
                major_marks.push(major);
                resident_marks.push(resident());
                let (late, fresh) = evolution_simulator::storage::take_breed_late();
                let (hit, map, unmap) = evolution_simulator::block_alloc::large_blocks();
                eprintln!(
                    "          large blocks since the start: {hit} reused, {map} mapped, {unmap} unmapped; children bred after their arena part: {late}, blocks bred into a new arena: {fresh}"
                );
            }
            if n >= base + generations {
                break snapshot.history.clone();
            }
        }
        anyhow::ensure!(started.elapsed() < Duration::from_secs(1800), "too slow");
        let _ = ctx.run_ui(eframe::egui::RawInput::default(), |_| {});
        std::thread::sleep(Duration::from_millis(20));
    };
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(ring) = history.last().map(|s| s.ring) {
        println!(
            "worker_rate: ring of {} blocks of {} creatures",
            ring.blocks, ring.block
        );
    }
    // Rate over the generations after the first two (warm-up).
    let first = marks.iter().find(|m| m.0 == base + 3).map(|m| m.1);
    let last = marks.last().map(|m| (m.0, m.1));
    if let (Some(first), Some((n, last))) = (first, last) {
        let done = (n - base - 3) * population;
        println!(
            "worker_rate: population {population}, generations {} to {n}: {:.0} creatures/s, {:.2} s per generation",
            base + 3,
            done as f64 / last.duration_since(first).as_secs_f64(),
            last.duration_since(first).as_secs_f64() / (n - base - 3) as f64
        );
        let at = |marks_of: &Vec<u64>, g: usize| {
            marks.iter().position(|m| m.0 == g).map(|i| marks_of[i])
        };
        if let (Some(a), Some(b)) = (at(&fault_marks, base + 3), at(&fault_marks, n)) {
            println!(
                "worker_rate: {:.0} minor page faults per generation over generations {} to {n}",
                (b - a) as f64 / (n - base - 3) as f64,
                base + 3
            );
            if let (Some(a), Some(b)) = (at(&major_marks, base + 3), at(&major_marks, n)) {
                println!(
                    "worker_rate: {:.0} major page faults per generation (swap reads)",
                    (b - a) as f64 / (n - base - 3) as f64
                );
            }
            // Per generation, the faults not accounted for by new resident memory.
            let regen: u64 = (base + 4..=n)
                .filter_map(|g| {
                    let i = marks.iter().position(|m| m.0 == g)?;
                    let faults = fault_marks[i] - fault_marks[i - 1];
                    let grown = resident_marks[i].saturating_sub(resident_marks[i - 1]);
                    Some(faults.saturating_sub(grown))
                })
                .sum();
            println!(
                "worker_rate: {:.0} page faults per generation beyond the growth of resident memory, generations {} to {n}",
                regen as f64 / (n - base - 3) as f64,
                base + 4
            );
        }
    }
    // Every generation's statistics, bit for bit.
    use std::hash::{Hash, Hasher};
    let mut digest = std::collections::hash_map::DefaultHasher::new();
    for s in history.iter().skip(base).take(generations) {
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
    println!(
        "worker_rate: engine threads waited {:.2} s for kernels",
        evolution_simulator::cuda_engine::kernel_wait_seconds()
    );
    let _ = replays.join();
    let mut times = replay_seconds.lock().unwrap().clone();
    if !times.is_empty() {
        times.sort_by(f64::total_cmp);
        let at = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
        println!(
            "worker_rate: {} replays, p50 {:.3} s, p95 {:.3} s, max {:.3} s",
            times.len(),
            at(0.5),
            at(0.95),
            times[times.len() - 1]
        );
    }
    worker.send(Command::Shutdown);
    Ok(())
}
