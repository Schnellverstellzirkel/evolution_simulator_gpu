//! Prints the best elites' archive distance beside their GPU replay's, which
//! must be equal. Usage: replay_match <save> [count]
//!
//! The substep honesty re-test:
//!
//!   replay_match <save> --retest <count> <out.csv>
//!
//! scores the save's best `count` elites in one batch, full 20 s trials at the
//! standard rate, with the kernel the `EVOLUTION_WARP_*` settings select, and
//! writes one row per elite: its archive distance, the re-test distance and
//! its fall time. Run it with the default settings and once each with
//! `EVOLUTION_WARP_SUBSTEPS` at 2 and 4, then
//!
//!   replay_match --ladder <base.csv> <two.csv> <four.csv>
//!
//! prints the honesty ratios (the distance at 2 and 4 substeps over the
//! distance at the base setting, median and p10).
mod common;
use anyhow::{Context, Result};
use evolution_simulator::storage;
use std::collections::HashMap;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "--ladder") {
        return ladder(&args[1..]);
    }
    let path = args.first().expect("save path");
    if args.get(1).is_some_and(|a| a == "--retest") {
        let count: usize = args.get(2).context("--retest needs a count")?.parse()?;
        let out = args.get(3).context("--retest needs an output path")?;
        return retest(path, count, out);
    }
    let count: usize = args.get(1).map_or(10, |a| a.parse().unwrap());
    let experiment = storage::load(std::path::Path::new(path))?;
    let _engine = common::open()?;
    let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    for (k, e) in elites.iter().take(count).enumerate() {
        let (creature, cfg) = e.replay_of(&experiment.config);
        let recording = common::record(&creature, &cfg)?;
        println!(
            "#{}: archive {:.4} replay {:.4}{}",
            k + 1,
            e.fitness,
            recording.result.fitness,
            if e.fine { " (confirmation trial)" } else { "" }
        );
    }
    Ok(())
}

fn retest(path: &str, count: usize, out: &str) -> Result<()> {
    let experiment = storage::load(std::path::Path::new(path))?;
    let mut engine = common::open()?;
    let mut elites: Vec<_> = experiment.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    let creatures: Vec<_> = elites.iter().map(|e| e.creature.clone()).collect();
    let cfg = evolution_simulator::config::Config {
        screen: None,
        ..experiment.config.clone()
    };
    let results = common::score_creatures(&mut engine, &creatures, &cfg)?;
    let mut text = String::from("id,archive,fine,distance,fall_time\n");
    for (e, r) in elites.iter().zip(&results) {
        text += &format!(
            "{},{},{},{},{}\n",
            e.creature.id,
            e.fitness,
            e.fine as u8,
            r.fitness,
            r.fall_time
        );
    }
    std::fs::write(out, text)?;
    println!("{}: {} elites re-tested", out, results.len());
    Ok(())
}

struct Row {
    archive: f32,
    distance: f32,
}

fn read(path: &str) -> Result<Vec<(u64, Row)>> {
    let text = std::fs::read_to_string(path).with_context(|| path.to_owned())?;
    let mut rows = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let num = |i: usize| f[i].parse::<f32>().unwrap_or(f32::NAN);
        rows.push((
            f[0].parse()?,
            Row {
                archive: num(1),
                distance: num(3),
            },
        ));
    }
    Ok(rows)
}

fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    sorted[((sorted.len() - 1) as f32 * q).round() as usize]
}

fn sorted(mut v: Vec<f32>) -> Vec<f32> {
    v.retain(|x| x.is_finite());
    v.sort_by(f32::total_cmp);
    v
}

fn ladder(paths: &[String]) -> Result<()> {
    anyhow::ensure!(paths.len() == 3, "--ladder needs the base, 2 and 4 substep files");
    let base = read(&paths[0])?;
    let others: Vec<HashMap<u64, Row>> = paths[1..]
        .iter()
        .map(|p| read(p).map(|rows| rows.into_iter().collect()))
        .collect::<Result<_>>()?;
    let names = ["2 substeps", "4 substeps"];
    let same = base.iter().filter(|(_, r)| r.distance.to_bits() == r.archive.to_bits()).count();
    println!(
        "{} elites; the base re-test equals the archive distance for {} (the rest were confirmed at the fine rate)",
        base.len(),
        same
    );
    let distances = sorted(base.iter().map(|(_, r)| r.distance).collect());
    println!(
        "base distance: median {:.2} m, p10 {:.2} m, best {:.2} m",
        quantile(&distances, 0.5),
        quantile(&distances, 0.1),
        quantile(&distances, 1.0)
    );
    for (k, other) in others.iter().enumerate() {
        // Ratio to the base distance; elites under 0.1 m at the base
        // are left out.
        let ratios = sorted(
            base.iter()
                .filter(|(_, r)| r.distance > 0.1)
                .filter_map(|(id, r)| other.get(id).map(|o| o.distance / r.distance))
                .collect(),
        );
        let below = ratios.iter().filter(|&&x| x < 0.6).count();
        let d = sorted(other.values().map(|o| o.distance).collect());
        println!(
            "{}: median ratio {:.3}, p10 {:.3}, p90 {:.3}, {} of {} below 0.6; distance median {:.2} m, best {:.2} m",
            names[k],
            quantile(&ratios, 0.5),
            quantile(&ratios, 0.1),
            quantile(&ratios, 0.9),
            below,
            ratios.len(),
            quantile(&d, 0.5),
            quantile(&d, 1.0)
        );
    }
    Ok(())
}
