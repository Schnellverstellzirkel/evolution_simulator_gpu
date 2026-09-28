//! Evolves a small population under the selected physics (`EVOLUTION_PHYSICS=2`
//! for the v2 prototype) and draws the best creature's trial as a filmstrip:
//! one stick figure per `every` seconds, left to right, over the ground, so a
//! gait can be judged from a still image. Also prints its speed, how often a
//! node touches the ground, and the fastest node speed.
//!
//! Usage: cargo run --release --example filmstrip -- [generations] [population] [seconds] [out.png]
use evolution_simulator::{config::Config, cpu_engine, scheduler, storage::Experiment};
use image::{Rgb, RgbImage};

fn line(img: &mut RgbImage, a: (f32, f32), b: (f32, f32), color: Rgb<u8>) {
    let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) as usize).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
            let (px, py) = (x as i64 + dx, y as i64 + dy);
            if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let generations: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(30);
    let population: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5000);
    let seconds: f32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let out = args
        .get(4)
        .cloned()
        .unwrap_or_else(|| "filmstrip.png".into());
    let seed: u64 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(40);
    let cfg = Config {
        population,
        seed,
        duration: seconds,
        random_seed: false,
        ..Config::default()
    };
    let mut experiment = Experiment::new(cfg)?;
    for generation in 0..generations {
        let results = cpu_engine::evaluate(&experiment.population, &experiment.config);
        for (index, result) in results.iter().enumerate() {
            let metrics =
                scheduler::to_metrics(&experiment.population, index, result, &experiment.config);
            experiment.record_result(index, &metrics);
        }
        experiment.evaluated = experiment.config.population;
        experiment.archive_batch()?;
        if generation + 1 < generations {
            experiment.prepare_next_batch()?;
        }
    }
    let mut ranked: Vec<_> = experiment.archive.entries.iter().collect();
    ranked.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    // The five best elites: a one-line check each (energy and friction
    // ledgers under physics v2), saved next to the dump as -1 to -4.
    for (rank, elite) in ranked.iter().take(5).enumerate() {
        if let Ok(path) = std::env::var("EVOLUTION_FILM_DUMP") {
            let path = if rank == 0 {
                path
            } else {
                path.replace(".json", &format!("-{rank}.json"))
            };
            std::fs::write(&path, serde_json::to_string(&elite.creature)?)?;
        }
        let (frames, result) = cpu_engine::replay(&elite.creature, &experiment.config);
        let e = evolution_simulator::physics2::ENERGY.with(|e| e.get());
        let rate = experiment.config.fidelity().rate as f32;
        let trial = &frames[evolution_simulator::physics::settle() as usize..];
        let mut fastest = 0.0f32;
        for pair in trial.windows(2) {
            for (a, b) in pair[0].iter().zip(&pair[1]) {
                fastest = fastest.max((b[0] - a[0]).hypot(b[1] - a[1]) * rate);
            }
        }
        let mass: f32 = evolution_simulator::physics::nodes(&elite.creature)
            .iter()
            .map(|n| n.mass)
            .sum();
        println!(
            "rank {rank}: {:.1} m ({:.1} m replayed, fell {:.2} s), {} nodes, {} muscles, {:.1} kg, fastest node {:.1} m/s; muscle work {:.0} J, gained otherwise {:.0} J, lost {:.0} J; friction along the slip {:.1}% of {:.0} N s; no contact {:.0}%",
            elite.fitness,
            result.fitness,
            result.fall_time,
            elite.creature.nodes.len(),
            elite.creature.muscles.len(),
            mass,
            fastest,
            e[0],
            e[1],
            e[2],
            100.0 * e[9] / e[10].max(1e-9),
            e[10],
            100.0 * e[5] / e[6].max(1.0)
        );
    }
    let best = ranked[0];
    let creature = best.creature.clone();
    let (frames, result) = cpu_engine::replay(&creature, &experiment.config);
    let ledger = evolution_simulator::physics2::LEDGER.with(|l| l.get());
    println!(
        "momentum ledger: ground and wind impulse {:.2} N s, momentum change {:.2} N s, made by the integrator {:.2} N s",
        ledger[0],
        ledger[1],
        ledger[1] - ledger[0]
    );
    let rate = experiment.config.fidelity().rate as f32;
    let start = evolution_simulator::physics::settle() as usize;
    let trial = &frames[start..];
    let nodes = trial[0].len();
    let mut fastest = 0.0f32;
    let mut touching = 0usize;
    for pair in trial.windows(2) {
        for (a, b) in pair[0].iter().zip(&pair[1]) {
            fastest = fastest.max((b[0] - a[0]).hypot(b[1] - a[1]) * rate);
        }
        if pair[1].iter().any(|p| p[1] < 0.06) {
            touching += 1;
        }
    }
    println!(
        "best: {} nodes, {} bones, {} muscles; archive {:.2} m, replay {:.2} m in {seconds} s ({:.2} m/s), fell at {:.2} s; fastest node {:.1} m/s; a node near the ground in {:.0}% of steps",
        nodes,
        creature.bones.len(),
        creature.muscles.len(),
        best.fitness,
        result.fitness,
        result.fitness / seconds,
        result.fall_time,
        fastest,
        100.0 * touching as f32 / trial.len().max(1) as f32
    );
    // Filmstrip: 16 poses, each drawn in its own panel around the body.
    let panels = 16usize;
    let (pw, ph) = (160u32, 200u32);
    // One scale for every panel, so the whole trial fits the tallest pose.
    let (mut width, mut height) = (0.1f32, 0.1f32);
    for frame in trial.iter().step_by(10) {
        let xs = frame.iter().map(|p| p[0]);
        let (lo, hi) = xs
            .clone()
            .fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
        width = width.max(hi - lo);
        height = height.max(frame.iter().map(|p| p[1]).fold(0.0f32, f32::max));
    }
    let scale = (0.9 * pw as f32 / width)
        .min(0.8 * (ph as f32 - 30.0) / height)
        .min(300.0);
    let mut img = RgbImage::from_pixel(pw * panels as u32, ph, Rgb([245, 248, 252]));
    for k in 0..panels {
        let frame = &trial[(k * (trial.len() - 1)) / (panels - 1)];
        let cx = frame.iter().map(|p| p[0]).sum::<f32>() / nodes as f32;
        let ox = k as f32 * pw as f32 + pw as f32 / 2.0;
        let ground = ph as f32 - 30.0;
        let to = |p: &[f32; 2]| (ox + (p[0] - cx) * scale, ground - p[1] * scale);
        line(
            &mut img,
            (k as f32 * pw as f32, ground),
            ((k + 1) as f32 * pw as f32, ground),
            Rgb([90, 140, 60]),
        );
        for bone in &creature.bones {
            line(
                &mut img,
                to(&frame[bone.a as usize]),
                to(&frame[bone.b as usize]),
                Rgb([40, 40, 60]),
            );
        }
        let head = to(&frame[0]);
        line(
            &mut img,
            (head.0 - 3.0, head.1),
            (head.0 + 3.0, head.1),
            Rgb([200, 60, 60]),
        );
        line(
            &mut img,
            (head.0, head.1 - 3.0),
            (head.0, head.1 + 3.0),
            Rgb([200, 60, 60]),
        );
        line(
            &mut img,
            ((k + 1) as f32 * pw as f32 - 1.0, 0.0),
            ((k + 1) as f32 * pw as f32 - 1.0, ph as f32),
            Rgb([200, 200, 210]),
        );
    }
    img.save(&out)?;
    println!("filmstrip: {out}");
    Ok(())
}
