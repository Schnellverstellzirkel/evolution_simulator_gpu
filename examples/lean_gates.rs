//! The spirit checks of a physics change, run on a checkpoint's elites.
//!
//! `lean_gates run <checkpoint> <out.json> [top] [record]` scores the top
//! elites of the global archive on the GPU engine (the substep count is the
//! kernel's, so `EVOLUTION_WARP_SUBSTEPS=8` makes it a retest at 4x) and,
//! with `record` set to 1, replays each of them with the scoring kernel and
//! measures the replay: foot slip per metre, cost of transport, the energy
//! the trial gained beyond the muscles' work, the horizontal momentum the
//! contact forces do not explain, the share of airborne steps and the
//! deepest penetration. `lean_gates report <checkpoint> <base.json>
//! <retest.json> [<base-at-generation-10.json>]` prints the gate table.
//! `lean_gates gifs <checkpoint> <dir> [count]` writes a GIF of each of the
//! best movers. `lean_gates export <checkpoint> <path.json> [rank]` writes an
//! elite's genes.
//! Replay-derived numbers come from 60 Hz frames, so they are estimates:
//! velocities are differences of positions.
mod common;
use evolution_simulator::{
    config::Config,
    evolution::Creature,
    physics,
    qd::{Elite, is_morphology_niche},
    storage,
};
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize, Clone)]
struct Row {
    rank: usize,
    archive_m: f32,
    score_m: f32,
    nodes: usize,
    muscles: usize,
    mass: f32,
    contact_bin: u8,
    cadence_bin: u8,
    feet_bin: u8,
    ground_contact: f32,
    gait_hz: f32,
    // Replay measures (zero without `record`).
    replay_m: f32,
    slip_per_m: f32,
    cot: f32,
    work_positive: f32,
    work_net: f32,
    energy_excess: f32,
    momentum_residual: f32,
    airborne: f32,
    penetration_mm: f32,
    speed: f32,
    recorded: bool,
}

fn elites_of(e: &storage::Experiment, top: usize) -> Vec<&Elite> {
    let mut elites: Vec<&Elite> = e
        .archive
        .entries
        .iter()
        .filter(|x| !is_morphology_niche(&x.niche))
        .collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(top);
    elites
}

fn measure(c: &Creature, cfg: &Config, row: &mut Row) -> anyhow::Result<()> {
    let recording = common::record(c, cfg)?;
    let (frames, result) = (&recording.frames, &recording.result);
    let fidelity = cfg.fidelity();
    let settle = fidelity.settle() as usize;
    let dt = physics::dt();
    let nodes = physics::nodes(c);
    let mass: f32 = nodes.iter().map(|n| n.mass).sum();
    let terminal = if result.fall_time > 0.0 {
        (settle + (result.fall_time * fidelity.rate as f32).round() as usize).min(frames.len() - 1)
    } else {
        frames.len() - 1
    };
    let distance = frames[terminal]
        .iter()
        .zip(&nodes)
        .map(|(p, n)| p[0] * n.mass)
        .sum::<f32>()
        / mass;
    row.replay_m = result.fitness;
    let amplitude = physics::terrain_amplitude(cfg.terrain);
    let floor = |p: [f32; 2], radius: f32| {
        let (h, s) = physics::terrain_with_slope(p[0], amplitude, cfg.slope);
        h + radius * (1.0 + s * s).sqrt()
    };
    // Foot slip while planted, and the deepest penetration.
    let (mut slip, mut deepest) = (0.0f32, 0.0f32);
    if cfg.ground && terminal > settle {
        for (j, node) in nodes.iter().enumerate() {
            for t in settle + 1..=terminal {
                let (now, before) = (frames[t][j], frames[t - 1][j]);
                deepest = deepest.max(floor(now, node.radius) - now[1]);
                if t >= settle + 2
                    && now[1] <= floor(now, node.radius) + 0.002
                    && before[1] <= floor(before, node.radius) + 0.002
                {
                    slip += (now[0] - before[0]).abs();
                }
            }
        }
    }
    row.slip_per_m = slip / distance.abs().max(0.01);
    row.penetration_mm = deepest.max(0.0) * 1000.0;
    // Muscle work from the recorded forces and the muscles' lengths.
    let steps = (terminal - settle).max(1);
    let (mut positive, mut net) = (0.0f32, 0.0f32);
    if let Some(forces) = &recording.forces {
        let length = |t: usize, m: &evolution_simulator::evolution::Muscle| {
            let (a, b) = (frames[t][m.node_a as usize], frames[t][m.node_b as usize]);
            (a[0] - b[0]).hypot(a[1] - b[1])
        };
        for t in settle + 1..=terminal {
            for (j, m) in c.muscles.iter().enumerate() {
                let w = forces.muscle[t][j] * (length(t - 1, m) - length(t, m));
                net += w;
                positive += w.max(0.0);
            }
        }
        // Airborne share of the steps.
        let free = (settle + 1..=terminal)
            .filter(|&t| forces.ground[t].iter().all(|&n| n <= 0.0))
            .count();
        row.airborne = free as f32 / steps as f32;
        // Horizontal momentum against the recorded friction on the nodes.
        let velocity = |t: usize| -> Vec<[f32; 2]> {
            (0..nodes.len())
                .map(|j| {
                    [
                        (frames[t + 1][j][0] - frames[t - 1][j][0]) / (2.0 * dt),
                        (frames[t + 1][j][1] - frames[t - 1][j][1]) / (2.0 * dt),
                    ]
                })
                .collect()
        };
        let (t0, t1) = (settle + 2, terminal.saturating_sub(1));
        if t1 > t0 + 10 {
            let momentum = |t: usize| -> f32 {
                velocity(t).iter().zip(&nodes).map(|(v, n)| v[0] * n.mass).sum()
            };
            let delta = momentum(t1) - momentum(t0);
            let impulse: f32 = (t0 + 1..=t1)
                .map(|t| forces.friction.get(t).map_or(0.0, |f| f.iter().sum::<f32>()))
                .sum::<f32>()
                * dt;
            let scale: f32 = (t0 + 1..=t1)
                .map(|t| forces.friction.get(t).map_or(0.0, |f| f.iter().map(|x| x.abs()).sum::<f32>()))
                .sum::<f32>()
                * dt;
            row.momentum_residual = (delta - impulse).abs() / scale.max(1e-3);
            // Energy gained beyond the muscles' work.
            let energy = |t: usize| -> f32 {
                velocity(t)
                    .iter()
                    .zip(&nodes)
                    .zip(&frames[t])
                    .map(|((v, n), p)| 0.5 * n.mass * (v[0] * v[0] + v[1] * v[1]) + n.mass * cfg.gravity * p[1])
                    .sum()
            };
            let gain = energy(t1) - energy(t0);
            row.energy_excess = (gain - net).max(0.0) / positive.max(1e-3);
        }
    }
    row.work_positive = positive;
    row.work_net = net;
    row.cot = positive / (mass * cfg.gravity * distance.abs().max(0.1));
    row.speed = distance / (steps as f32 * dt);
    row.recorded = true;
    Ok(())
}

fn run(args: &[String]) -> anyhow::Result<()> {
    let e = storage::load(std::path::Path::new(&args[0]))?;
    let top: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(300);
    let record = args.get(3).is_some_and(|v| v == "1");
    let mut cfg = Config {
        screen: None,
        ..e.config.clone()
    };
    // FINE=1: the confirmation physics (4x the rate and passes) of the game.
    if std::env::var_os("FINE").is_some() {
        cfg = evolution_simulator::scheduler::confirm_config(&cfg);
        cfg.screen = None;
    }
    let elites = elites_of(&e, top);
    let mut engine = common::open()?;
    let creatures: Vec<Creature> = elites.iter().map(|x| x.creature.clone()).collect();
    let scores = common::score_creatures(&mut engine, &creatures, &cfg)?;
    let mut rows = Vec::new();
    for (rank, (elite, score)) in elites.iter().zip(&scores).enumerate() {
        let niche = elite.descriptor.niche().0;
        let mass: f32 = physics::nodes(&elite.creature).iter().map(|n| n.mass).sum();
        let mut row = Row {
            rank,
            archive_m: elite.fitness,
            score_m: score.fitness,
            nodes: elite.creature.nodes.len(),
            muscles: elite.creature.muscles.len(),
            mass,
            contact_bin: niche[0],
            cadence_bin: niche[1],
            feet_bin: niche[4],
            ground_contact: elite.descriptor.ground_contact,
            gait_hz: elite.descriptor.gait_frequency,
            ..Default::default()
        };
        if record {
            measure(&elite.creature, &cfg, &mut row)?;
        }
        rows.push(row);
    }
    std::fs::write(&args[1], serde_json::to_string(&rows)?)?;
    eprintln!("wrote {} rows to {}", rows.len(), args[1]);
    Ok(())
}

fn quantile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * q).round() as usize]
}

fn load(path: &str) -> anyhow::Result<Vec<Row>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn report(args: &[String]) -> anyhow::Result<()> {
    let e = storage::load(std::path::Path::new(&args[0]))?;
    let base = load(&args[1])?;
    let retest = load(&args[2])?;
    let early = args.get(3).map(|p| load(p)).transpose()?;
    let mark = |ok: bool| if ok { "pass" } else { "FAIL" };
    let history: Vec<f32> = e.history.iter().map(|s| s.best).collect();
    let last = history.last().copied().unwrap_or(0.0);
    // 1. Progress over generations 5 to 30.
    let window = &history[(5usize).min(history.len())..];
    let falls = window.windows(2).filter(|w| w[1] < w[0] - 1e-3).count();
    let archive_best = base.first().map_or(0.0, |r| r.archive_m);
    println!("generations {}, best per generation (5 on): {:?}", history.len(), window.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
    println!(
        "progress: best at the last generation {last:.1} m (archive best {archive_best:.1} m), 0.5x to 2x of 37 m is 18.5 to 74: {}; generation-best falls {falls} times from 5 to {}: {}",
        mark((18.5..=74.0).contains(&archive_best.max(last))),
        history.len(),
        mark(falls == 0)
    );
    // 2. Diversity.
    let best = base.first().map_or(0.0, |r| r.archive_m);
    let near: Vec<&Row> = base.iter().filter(|r| r.archive_m >= 0.5 * best).collect();
    let bins = |f: fn(&Row) -> u8, n: u8| (0..n).filter(|&b| near.iter().any(|r| f(r) == b)).count();
    let all: Vec<&Elite> = e
        .archive
        .entries
        .iter()
        .filter(|x| !is_morphology_niche(&x.niche))
        .collect();
    let near_all: Vec<&&Elite> = all.iter().filter(|x| x.fitness >= 0.5 * best).collect();
    let abins = |i: usize, n: u8| (0..n).filter(|&b| near_all.iter().any(|x| x.descriptor.niche().0[i] == b)).count();
    let (cad, con, fee) = (abins(1, 8), abins(0, 6), abins(4, 5));
    let coverage = e.history.last().map_or(0.0, |s| s.archive_coverage);
    println!(
        "diversity: within 50% of the best ({:.1} m): cadence bins {cad} of 8 (need 5), contact bins {con} of 6 (need 4), feet bins {fee} of 5 (need 3): {}; coverage {:.0}% (need 70): {}",
        0.5 * best,
        mark(cad >= 5 && con >= 4 && fee >= 3),
        100.0 * coverage,
        mark(coverage >= 0.7)
    );
    let _ = bins;
    // 3. Cost of transport.
    let mut cot: Vec<f32> = base.iter().filter(|r| r.recorded && r.replay_m > 1.0).map(|r| r.cot).collect();
    let (p10, p50, p90) = (quantile(&mut cot.clone(), 0.1), quantile(&mut cot.clone(), 0.5), quantile(&mut cot, 0.9));
    let early_median = early.as_ref().map(|rows| {
        let mut v: Vec<f32> = rows.iter().filter(|r| r.recorded && r.replay_m > 1.0).map(|r| r.cot).collect();
        quantile(&mut v, 0.5)
    });
    println!(
        "cost of transport (top {} moving): p10 {p10:.2}, median {p50:.2}, p90 {p90:.2}, spread p90/p10 {:.2} (need 2): {}; generation-10 median {} -> median holds or falls: {}",
        cot.len(),
        p90 / p10,
        mark(p90 / p10 >= 2.0),
        early_median.map_or("n/a".into(), |m| format!("{m:.2}")),
        early_median.map_or("n/a", |m| mark(p50 <= m * 1.02))
    );
    // 4. Retest at 4x the substeps.
    let mut ratios: Vec<f32> = base
        .iter()
        .zip(&retest)
        .filter(|(b, _)| b.score_m > 1.0)
        .map(|(b, r)| r.score_m / b.score_m)
        .collect();
    let under = ratios.iter().filter(|&&r| r < 0.5).count() as f32 / ratios.len().max(1) as f32;
    let (r10, r50) = (quantile(&mut ratios.clone(), 0.1), quantile(&mut ratios, 0.5));
    println!(
        "retest at 4x substeps ({} elites over 1 m): median ratio {r50:.3} (need 0.9): {}, p10 {r10:.3} (need 0.7): {}, share under 0.5 {:.1}% (need 2): {}",
        base.iter().filter(|b| b.score_m > 1.0).count(),
        mark(r50 >= 0.9),
        mark(r10 >= 0.7),
        100.0 * under,
        mark(under <= 0.02)
    );
    // 5. Honesty of the replays.
    let recorded: Vec<&Row> = base.iter().filter(|r| r.recorded && r.replay_m > 1.0).collect();
    let mut slip: Vec<f32> = recorded.iter().map(|r| r.slip_per_m).collect();
    let mut slip_ratio: Vec<f32> = base
        .iter()
        .zip(&retest)
        .filter(|(b, r)| b.recorded && r.recorded && b.slip_per_m > 1e-3 && b.replay_m > 1.0 && r.replay_m > 1.0)
        .map(|(b, r)| r.slip_per_m / b.slip_per_m)
        .collect();
    let slip_median = quantile(&mut slip, 0.5);
    let sr = quantile(&mut slip_ratio, 0.5);
    println!(
        "planted-foot slip: median {slip_median:.2} m per metre; ratio of the 4x retest to the base, median {sr:.3} over {} elites (need 0.95 to 1.05): {}",
        slip_ratio.len(),
        mark((0.95..=1.05).contains(&sr))
    );
    let mut excess: Vec<f32> = recorded.iter().map(|r| r.energy_excess).collect();
    let mut momentum: Vec<f32> = recorded.iter().map(|r| r.momentum_residual).collect();
    let mut pen: Vec<f32> = recorded.iter().map(|r| r.penetration_mm).collect();
    println!(
        "trial energy excess over muscle work: median {:.4}, p90 {:.4} (need at most 0.01 on the top 300): {}",
        quantile(&mut excess.clone(), 0.5),
        quantile(&mut excess.clone(), 0.9),
        mark(quantile(&mut excess, 0.9) <= 0.01)
    );
    println!(
        "horizontal momentum not explained by the recorded friction (replay estimate, relative to the friction impulse): median {:.4}, p90 {:.4}",
        quantile(&mut momentum.clone(), 0.5),
        quantile(&mut momentum, 0.9)
    );
    println!("deepest penetration per elite: median {:.2} mm, p99 {:.2} mm (need 2): {}", quantile(&mut pen.clone(), 0.5), quantile(&mut pen.clone(), 0.99), mark(quantile(&mut pen, 0.99) <= 2.0));
    // 6. Gait types.
    let pair = |r: &Row| (r.cadence_bin, r.contact_bin);
    let mut counts = std::collections::HashMap::new();
    for r in &base {
        *counts.entry(pair(r)).or_insert(0usize) += 1;
    }
    let biggest = counts.values().copied().max().unwrap_or(0) as f32 / base.len().max(1) as f32;
    let hoppers = base.iter().filter(|r| r.ground_contact < 0.4).count() as f32 / base.len().max(1) as f32;
    let mut top20: Vec<(u8, u8)> = base.iter().take(20).map(pair).collect();
    top20.sort_unstable();
    top20.dedup();
    println!(
        "gait types: biggest (cadence, contact) pair holds {:.0}% of the top {} (fail above 80%): {}; contact under 0.4 in {:.0}% (fail above 90%): {}; distinct pairs among the top 20: {} (need 5): {}",
        100.0 * biggest,
        base.len(),
        mark(biggest <= 0.8),
        100.0 * hoppers,
        mark(hoppers <= 0.9),
        top20.len(),
        mark(top20.len() >= 5)
    );
    // Pacing of the top 20.
    println!("top 20: rank  m  speed  gait_hz  contact  airborne  slip/m  cot  nodes muscles  cadence/contact bin");
    for r in base.iter().take(20) {
        println!(
            "{:4} {:7.1} {:6.2} {:7.2} {:8.2} {:9.2} {:7.2} {:5.2} {:5} {:7} {:5}/{}",
            r.rank, r.archive_m, r.speed, r.gait_hz, r.ground_contact, r.airborne, r.slip_per_m, r.cot, r.nodes, r.muscles, r.cadence_bin, r.contact_bin
        );
    }
    let mean = |f: fn(&Row) -> f32| base.iter().map(f).sum::<f32>() / base.len().max(1) as f32;
    let mut muscles: Vec<f32> = base.iter().map(|r| r.muscles as f32).collect();
    let mut nodes: Vec<f32> = base.iter().map(|r| r.nodes as f32).collect();
    println!(
        "elite bodies (top {}): nodes mean {:.1} p90 {:.0}, muscles mean {:.1} p90 {:.0}",
        base.len(),
        mean(|r| r.nodes as f32),
        quantile(&mut nodes, 0.9),
        mean(|r| r.muscles as f32),
        quantile(&mut muscles, 0.9)
    );
    Ok(())
}

fn gifs(args: &[String]) -> anyhow::Result<()> {
    let e = storage::load(std::path::Path::new(&args[0]))?;
    let dir = std::path::Path::new(&args[1]);
    std::fs::create_dir_all(dir)?;
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5);
    let cfg = Config {
        screen: None,
        ..e.config.clone()
    };
    let _engine = common::open()?;
    for (rank, elite) in elites_of(&e, count).into_iter().enumerate() {
        let recording = common::record(&elite.creature, &cfg)?;
        let path = dir.join(format!("mover{}_{:.0}m.gif", rank + 1, recording.result.fitness));
        let frames = evolution_simulator::ui::write_replay_gif(&elite.creature, &cfg, &recording.frames, &path)?;
        println!("{} ({frames} frames, {:.1} m)", path.display(), recording.result.fitness);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("report") => report(&args[1..]),
        Some("gifs") => gifs(&args[1..]),
        Some("export") => {
            // lean_gates export <checkpoint> <path.json> [rank]
            let e = storage::load(std::path::Path::new(&args[1]))?;
            let rank: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
            let elite = elites_of(&e, rank + 1).pop().expect("an elite");
            std::fs::write(&args[2], serde_json::to_string(&elite.creature)?)?;
            Ok(())
        }
        _ => anyhow::bail!("usage: lean_gates run|report|gifs ..."),
    }
}
