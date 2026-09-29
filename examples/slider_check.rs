//! Finds sliders among evolved elites: bodies that drag their nodes along
//! the ground instead of planting feet. Evolves fixed seeds on the CPU with
//! the game's loop, replays the best elites, and prints per elite:
//!
//! - `drag`: forward slide of touching nodes over the body's forward travel
//!   in the same steps. A planted foot moves little while the body passes
//!   over it (walker, near 0); a node dragged along moves with the body
//!   (sled, near 1).
//! - `slip/m`: total slide of touching nodes per meter traveled.
//! - `touch`: share of steps with a node on the ground.
//!
//! With `--film DIR` it also draws each seed's worst slider as two PNG
//! strips: 16 poses over the trial, and 16 poses 1/15 s apart in a fixed
//! frame with ground marks, so a sliding foot is visible.
//!
//! Usage: cargo run --release --example slider_check -- [generations] [population] [seconds] [seeds] [top] [--film DIR] [--dump DIR]
use anyhow::{Context, Result};
use evolution_simulator::{
    config::Config, cpu_engine, creature_kernel::GpuResult, engine, evolution::Creature, physics,
    scheduler, storage::Experiment,
};
use image::{Rgb, RgbImage};

/// Node positions of a recorded trial, one entry per step.
type Frames = Vec<Vec<[f32; 2]>>;

struct Options {
    generations: u32,
    population: usize,
    duration: f32,
    seeds: Vec<u64>,
    top: usize,
    film: Option<String>,
    dump: Option<String>,
}

fn options() -> Result<Options> {
    let mut positionals = Vec::new();
    let (mut film, mut dump) = (None, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--film" => film = Some(args.next().context("--film needs a directory")?),
            "--dump" => dump = Some(args.next().context("--dump needs a directory")?),
            _ => positionals.push(arg),
        }
    }
    let get = |i: usize| positionals.get(i).map(String::as_str);
    Ok(Options {
        generations: get(0).map_or(Ok(20), str::parse)?,
        population: get(1).map_or(Ok(2000), str::parse)?,
        duration: get(2).map_or(Ok(20.0), str::parse)?,
        seeds: get(3)
            .unwrap_or("38")
            .split(',')
            .map(|s| s.trim().parse().context("seed"))
            .collect::<Result<_>>()?,
        top: get(4).map_or(Ok(10), str::parse)?,
        film,
        dump,
    })
}

#[derive(Default, Clone, Copy)]
pub struct Slide {
    pub distance: f32,
    pub drag: f32,
    pub slip_per_meter: f32,
    pub touch: f32,
    pub fall_time: f32,
}

/// Slide measures of one recorded trial (see the module comment).
pub fn slide(
    creature: &Creature,
    frames: &[Vec<[f32; 2]>],
    result: &GpuResult,
    cfg: &Config,
) -> Slide {
    let nodes = physics::body(&creature.nodes, &creature.bones);
    let fidelity = cfg.fidelity();
    let settle = fidelity.settle() as usize;
    let terminal = if result.fall_time > 0.0 {
        (settle + (result.fall_time * fidelity.rate as f32).round() as usize).min(frames.len() - 1)
    } else {
        frames.len() - 1
    };
    let mass: f32 = nodes.iter().map(|n| n.mass).sum();
    let com = |t: usize| -> f32 {
        frames[t]
            .iter()
            .zip(&nodes)
            .map(|(p, n)| p[0] * n.mass)
            .sum::<f32>()
            / mass
    };
    let amplitude = physics::terrain_amplitude(cfg.terrain);
    let down = |p: [f32; 2], radius: f32| {
        let (height, slope) = physics::terrain_with_slope(p[0], amplitude, cfg.slope);
        p[1] <= height + radius * (1.0 + slope * slope).sqrt() + 0.002
    };
    let (mut node_forward, mut body_forward, mut slip) = (0.0f64, 0.0f64, 0.0f64);
    let mut touching_steps = 0usize;
    // The first timed step recenters the settled body; skip it.
    for t in settle + 2..=terminal {
        let step = com(t) - com(t - 1);
        let mut any = false;
        for (j, node) in nodes.iter().enumerate() {
            let (now, before) = (frames[t][j], frames[t - 1][j]);
            if down(now, node.radius) && down(before, node.radius) {
                any = true;
                let dx = now[0] - before[0];
                slip += f64::from(dx.abs());
                if step > 0.0 {
                    node_forward += f64::from(dx.max(0.0));
                    body_forward += f64::from(step);
                }
            }
        }
        touching_steps += usize::from(any);
    }
    let distance = com(terminal);
    Slide {
        distance,
        drag: (node_forward / body_forward.max(1e-9)) as f32,
        slip_per_meter: (slip / f64::from(distance.abs().max(1e-3))) as f32,
        touch: touching_steps as f32 / (terminal.saturating_sub(settle + 1)).max(1) as f32,
        fall_time: result.fall_time,
    }
}

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

/// Draws 16 poses: `spacing` steps apart from `first`, each panel centered
/// on the pose's center (`fixed` false) or on the first pose's center.
fn strip(
    creature: &Creature,
    frames: &[Vec<[f32; 2]>],
    first: usize,
    spacing: usize,
    fixed: bool,
) -> RgbImage {
    let nodes = physics::body(&creature.nodes, &creature.bones);
    let panels = 16usize;
    let (pw, ph) = (160u32, 200u32);
    let pick = |k: usize| (first + k * spacing).min(frames.len() - 1);
    let (mut width, mut height) = (0.1f32, 0.1f32);
    for k in 0..panels {
        let frame = &frames[pick(k)];
        let (lo, hi) = frame
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p[0]), b.max(p[0])));
        width = width.max(hi - lo);
        height = height.max(frame.iter().map(|p| p[1]).fold(0.0f32, f32::max));
    }
    let scale = (0.8 * pw as f32 / width)
        .min(0.8 * (ph as f32 - 30.0) / height)
        .min(300.0);
    let center = |frame: &[[f32; 2]]| frame.iter().map(|p| p[0]).sum::<f32>() / frame.len() as f32;
    let anchor = center(&frames[pick(0)]);
    let mut img = RgbImage::from_pixel(pw * panels as u32, ph, Rgb([245, 248, 252]));
    for k in 0..panels {
        let frame = &frames[pick(k)];
        let cx = if fixed { anchor } else { center(frame) };
        let ox = k as f32 * pw as f32 + pw as f32 / 2.0;
        let ground = ph as f32 - 30.0;
        let to = |p: &[f32; 2]| (ox + (p[0] - cx) * scale, ground - p[1] * scale);
        line(
            &mut img,
            (k as f32 * pw as f32, ground),
            ((k + 1) as f32 * pw as f32, ground),
            Rgb([90, 140, 60]),
        );
        // Ground marks every 10 cm, so sliding shows against them.
        let world_left = cx - (pw as f32 / 2.0) / scale;
        let mut mark = (world_left * 10.0).floor() / 10.0;
        while mark < cx + (pw as f32 / 2.0) / scale {
            let (x, y) = to(&[mark, 0.0]);
            line(&mut img, (x, y), (x, y + 8.0), Rgb([90, 140, 60]));
            mark += 0.1;
        }
        for bone in &creature.bones {
            line(
                &mut img,
                to(&frame[bone.a as usize]),
                to(&frame[bone.b as usize]),
                Rgb([40, 40, 60]),
            );
        }
        for (p, node) in frame.iter().zip(&nodes) {
            let (x, y) = to(p);
            let touching = p[1] <= node.radius + 0.002;
            let color = if touching {
                Rgb([220, 40, 40])
            } else {
                Rgb([60, 90, 200])
            };
            line(&mut img, (x - 2.0, y), (x + 2.0, y), color);
            line(&mut img, (x, y - 2.0), (x, y + 2.0), color);
        }
        let head = to(&frame[0]);
        line(
            &mut img,
            (head.0 - 4.0, head.1 - 4.0),
            (head.0 + 4.0, head.1 + 4.0),
            Rgb([200, 60, 160]),
        );
        line(
            &mut img,
            ((k + 1) as f32 * pw as f32 - 1.0, 0.0),
            ((k + 1) as f32 * pw as f32 - 1.0, ph as f32),
            Rgb([200, 200, 210]),
        );
    }
    img
}

fn main() -> Result<()> {
    engine::lower_thread_priority();
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(engine::rayon_threads())
        .start_handler(|_| engine::lower_thread_priority())
        .build_global();
    let options = options()?;
    println!("seed rank distance_m drag slip_per_m touch fall_s nodes bones muscles mass_kg feet");
    let mut all = Vec::new();
    for &seed in &options.seeds {
        let cfg = Config {
            population: options.population,
            duration: options.duration,
            random_seed: false,
            seed,
            ..Config::default()
        };
        let mut experiment = Experiment::new(cfg)?;
        for generation in 0..options.generations {
            let results = cpu_engine::evaluate(&experiment.population, &experiment.config);
            for (index, result) in results.iter().enumerate() {
                let metrics = scheduler::to_metrics(
                    &experiment.population,
                    index,
                    result,
                    &experiment.config,
                );
                experiment.record_result(index, &metrics);
            }
            experiment.evaluated = experiment.config.population;
            experiment.archive_batch()?;
            if generation + 1 < options.generations {
                experiment.prepare_next_batch()?;
            }
        }
        let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
        elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        elites.truncate(options.top);
        let cfg = experiment.config.clone();
        let mut worst: Option<(f32, Creature, Frames)> = None;
        for (rank, elite) in elites.iter().enumerate() {
            let creature = &elite.creature;
            let (frames, result) = cpu_engine::replay(creature, &cfg);
            let s = slide(creature, &frames, &result, &cfg);
            let nodes = physics::body(&creature.nodes, &creature.bones);
            let mass: f32 = nodes.iter().map(|n| n.mass).sum();
            let row = format!(
                "{seed} {rank} {:.2} {:.3} {:.3} {:.2} {:.2} {} {} {} {:.2} {}",
                s.distance,
                s.drag,
                s.slip_per_meter,
                s.touch,
                s.fall_time,
                creature.nodes.len(),
                creature.bones.len(),
                creature.muscles.len(),
                mass,
                result.feet()
            );
            println!("{row}");
            if let Some(dir) = &options.dump {
                std::fs::create_dir_all(dir)?;
                std::fs::write(
                    format!("{dir}/seed{seed}-rank{rank}.json"),
                    serde_json::to_string(creature)?,
                )?;
            }
            all.push(s);
            if worst.as_ref().is_none_or(|w| s.drag > w.0) {
                worst = Some((s.drag, creature.clone(), frames));
            }
        }
        if let (Some(dir), Some((drag, creature, frames))) = (&options.film, worst) {
            std::fs::create_dir_all(dir)?;
            let settle = cfg.fidelity().settle() as usize;
            let trial = frames.len() - settle;
            strip(&creature, &frames, settle, trial / 16, false)
                .save(format!("{dir}/seed{seed}-trial.png"))?;
            let rate = cfg.fidelity().rate as usize;
            strip(&creature, &frames, settle + 10 * rate, rate / 15, true)
                .save(format!("{dir}/seed{seed}-close.png"))?;
            println!("seed {seed}: filmed the elite with drag {drag:.3} in {dir}");
        }
    }
    let mut drags: Vec<f32> = all.iter().map(|s| s.drag).collect();
    drags.sort_by(f32::total_cmp);
    if !drags.is_empty() {
        let sliders = all.iter().filter(|s| s.drag > 0.5).count();
        println!(
            "{} elites: drag median {:.3}, p90 {:.3}, max {:.3}; {} with drag above 0.5",
            drags.len(),
            drags[drags.len() / 2],
            drags[drags.len() * 9 / 10],
            drags[drags.len() - 1],
            sliders
        );
    }
    Ok(())
}
