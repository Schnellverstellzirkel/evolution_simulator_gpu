//! Replays one creature (a JSON file from `filmstrip` with
//! EVOLUTION_FILM_DUMP) under the selected physics (`EVOLUTION_PHYSICS=2` for
//! the v2 prototype) and draws a chronophotograph: every pose of a short
//! window at fixed intervals, in world coordinates, light to dark, so a gait
//! cycle can be judged from one image. Also prints its speed, flight time,
//! fastest node, lowest head, and the momentum and energy ledgers.
//!
//! Usage: cargo run --release --example strobe -- <creature.json> <out.png> [start s] [window s] [every steps] [trial s]
use evolution_simulator::{config::Config, cpu_engine, evolution::Creature, physics, physics2};
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
    let creature: Creature = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let out = args.get(2).cloned().unwrap_or_else(|| "strobe.png".into());
    let from: f32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(30.0);
    let window: f32 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let every: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(4);
    let seconds: f32 = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(60.0);
    let cfg = Config {
        random_seed: false,
        duration: seconds,
        ..Config::default()
    };
    let (frames, result) = cpu_engine::replay(&creature, &cfg);
    let rate = cfg.fidelity().rate as f32;
    let start = physics::settle() as usize;
    let trial = &frames[start..];
    let nodes = physics::nodes(&creature);
    let mass: f32 = nodes.iter().map(|n| n.mass).sum();
    let ledger = physics2::LEDGER.with(|l| l.get());
    let energy = physics2::ENERGY.with(|e| e.get());
    let mut fastest = 0.0f32;
    let mut low_head = f32::INFINITY;
    for pair in trial.windows(2) {
        for (a, b) in pair[0].iter().zip(&pair[1]) {
            fastest = fastest.max((b[0] - a[0]).hypot(b[1] - a[1]) * rate);
        }
        low_head = low_head.min(pair[1][0][1]);
    }
    println!(
        "{} nodes, {} muscles, {:.2} kg: {:.2} m in {} s ({:.2} m/s), fell at {:.2} s; fastest node {:.1} m/s; lowest head {:.3} m",
        creature.nodes.len(),
        creature.muscles.len(),
        mass,
        result.fitness,
        cfg.duration,
        result.fitness / cfg.duration,
        result.fall_time,
        fastest,
        low_head
    );
    if physics2::enabled() {
        println!(
            "momentum: ground and wind {:.2} N s, change {:.2} N s; energy: muscle work {:.1} J, gained otherwise {:.1} J, lost {:.1} J, momentum balance +{:.1} J -{:.1} J; no contact in {:.1}% of steps",
            ledger[0],
            ledger[1],
            energy[0],
            energy[1],
            energy[2],
            energy[3],
            energy[4],
            100.0 * energy[5] / energy[6].max(1.0)
        );
        println!(
            "in steps without contact: gained {:.1} J, lost {:.1} J; friction {:.1} N s, of which {:.1} N s pushed a node along its slip after the step; the first-law check took {:.1} J",
            energy[7], energy[8], energy[10], energy[9], energy[11]
        );
    }
    if physics2::enabled() {
        let counts = physics2::CONTACT_COUNTS.with(|c| c.get());
        println!("steps by nodes in the contact solve (0, 1, 2, ...): {counts:?}");
    }
    let first = ((from * rate) as usize).min(trial.len() - 1);
    let last = (((from + window) * rate) as usize).min(trial.len() - 1);
    let poses: Vec<&Vec<[f32; 2]>> = trial[first..=last].iter().step_by(every).collect();
    let (mut lo_x, mut hi_x, mut hi_y) = (f32::MAX, f32::MIN, 0.2f32);
    for pose in &poses {
        for p in pose.iter() {
            lo_x = lo_x.min(p[0]);
            hi_x = hi_x.max(p[0]);
            hi_y = hi_y.max(p[1]);
        }
    }
    let (w, h) = (1600u32, 400u32);
    let scale = (0.94 * w as f32 / (hi_x - lo_x).max(0.1))
        .min(0.85 * (h as f32 - 30.0) / hi_y)
        .min(600.0);
    let ground = h as f32 - 20.0;
    let to = |p: &[f32; 2]| {
        (
            0.03 * w as f32 + (p[0] - lo_x) * scale,
            ground - p[1] * scale,
        )
    };
    let mut img = RgbImage::from_pixel(w, h, Rgb([248, 249, 252]));
    line(
        &mut img,
        (0.0, ground),
        (w as f32, ground),
        Rgb([90, 140, 60]),
    );
    // A tick every 10 cm along the ground.
    let mut x = (lo_x * 10.0).floor() / 10.0;
    while x <= hi_x + 0.1 {
        let (sx, sy) = to(&[x, 0.0]);
        line(&mut img, (sx, sy), (sx, sy + 6.0), Rgb([90, 140, 60]));
        x += 0.1;
    }
    for (k, pose) in poses.iter().enumerate() {
        let t = k as f32 / (poses.len().max(2) - 1) as f32;
        let shade = (200.0 - 180.0 * t) as u8;
        let color = Rgb([shade, shade, (shade as f32 * 0.8 + 40.0) as u8]);
        for bone in &creature.bones {
            line(
                &mut img,
                to(&pose[bone.a as usize]),
                to(&pose[bone.b as usize]),
                color,
            );
        }
        let head = to(&pose[0]);
        let red = Rgb([230, (60.0 + 150.0 * (1.0 - t)) as u8, 60]);
        line(
            &mut img,
            (head.0 - 3.0, head.1),
            (head.0 + 3.0, head.1),
            red,
        );
        line(
            &mut img,
            (head.0, head.1 - 3.0),
            (head.0, head.1 + 3.0),
            red,
        );
    }
    img.save(&out)?;
    println!(
        "strobe: {out} ({} poses from {from} s to {:.2} s, one every {every} steps)",
        poses.len(),
        from + window
    );
    Ok(())
}
