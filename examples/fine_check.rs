//! How many of an archive's best elites keep their distance when the trial runs at
//! the fine fidelity (4 substeps) of the record confirmations.
//! Usage: fine_check <checkpoint.evo> <archive: g or island number> [count]
mod common;
use evolution_simulator::{config::Config, storage};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let e = storage::load(std::path::Path::new(&args[1]))?;
    let archive = if args[2] == "g" {
        &e.archive
    } else {
        &e.islands[args[2].parse::<usize>()?]
    };
    let count: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(300);
    let mut elites: Vec<_> = archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    let mut engine = common::open()?;
    let creatures: Vec<_> = elites.iter().map(|x| x.creature.unpack()).collect();
    let standard = Config {
        screen: None,
        rungs: None,
        ..e.config.clone()
    };
    let fine = Config {
        screen: None,
        rungs: None,
        ..evolution_simulator::scheduler::confirm_config(&e.config)
    };
    let a = common::score_creatures(&mut engine, &creatures, &standard)?;
    let b = common::score_creatures(&mut engine, &creatures, &fine)?;
    let (mut kept, mut lost, mut stale, mut held) = (0, 0, 0, 0);
    for (i, x) in elites.iter().enumerate() {
        let (s, f) = (a[i].fitness, b[i].fitness);
        if (s - x.fitness).abs() > 0.05 * x.fitness.abs().max(1.0) {
            stale += 1;
            if stale <= 6 {
                println!(
                    "  mismatch rank {i}: stored {:.3} replay {s:.3} nodes {} muscles {} fine {} improved_gen {} visits {} emitter {:?}",
                    x.fitness,
                    x.creature.node_count(),
                    x.creature.muscle_count(),
                    x.fine,
                    x.improved_generation,
                    x.visits,
                    x.emitter
                );
            }
        }
        if f >= 0.8 * s {
            kept += 1
        } else {
            lost += 1
        }
        if f >= 0.8 * x.fitness {
            held += 1
        }
    }
    let mut ratio: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(s, f)| f.fitness / s.fitness.max(0.01))
        .collect();
    ratio.sort_by(|x, y| x.total_cmp(y));
    let q = |f: f32| ratio[((ratio.len() - 1) as f32 * f) as usize];
    println!(
        "fine over standard: 10% {:.2} 25% {:.2} median {:.2} 75% {:.2} 90% {:.2}; fine farther in {} of {}",
        q(0.1),
        q(0.25),
        q(0.5),
        q(0.75),
        q(0.9),
        ratio.iter().filter(|&&r| r > 1.0).count(),
        ratio.len()
    );
    let mut shown = 0;
    for (i, x) in elites.iter().enumerate() {
        if b[i].fitness < 0.3 * a[i].fitness && shown < 8 {
            shown += 1;
            println!(
                "  collapse rank {i}: standard {:.2} (fell {:.2} s) fine {:.2} (fell {:.2} s, head shake {:.1}) nodes {} muscles {}",
                a[i].fitness,
                a[i].fall_time,
                b[i].fitness,
                b[i].fall_time,
                b[i].head_shake,
                x.creature.node_count(),
                x.creature.muscle_count()
            );
        }
    }
    let mut sd: Vec<f32> = a.iter().map(|r| r.fitness).collect();
    sd.sort_by(|x, y| x.total_cmp(y));
    println!("median standard distance {:.2}", sd[sd.len() / 2]);
    println!(
        "archive {} top {}: archive best {:.2}, standard best {:.2}, fine best {:.2}; keep >=80% at fine: {kept}, lose: {lost}; archive score differs from a standard replay: {stale}; fine replay reaches 80% of the STORED score: {held} of {}",
        args[2],
        elites.len(),
        elites[0].fitness,
        a.iter().map(|r| r.fitness).fold(f32::MIN, f32::max),
        b.iter().map(|r| r.fitness).fold(f32::MIN, f32::max),
        elites.len()
    );
    Ok(())
}
