//! Replays an archive's best elites at the standard trial and at the fine trial,
//! and counts how many keep their distance at the fine one. The fine trial is the
//! game's confirmation trial (`scheduler::confirm_config`): the same world at
//! twice the step rate, 120 steps per second (`physics::Fidelity::fine()`). The
//! tool also lists elites whose stored score a standard replay does not
//! reproduce, and elites that collapse at the fine trial.
//!
//! Usage: `fine_check <checkpoint.evo> <archive: g or island number> [count]`
//! Here `g` is the global archive, a number is an index into
//! `Experiment::islands` (the islands, then their nurseries), and `count`
//! defaults to 300.
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
    // The `count` best elites, best first.
    let mut elites: Vec<_> = archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    let mut engine = common::open()?;
    let creatures: Vec<_> = elites.iter().map(|x| x.creature.unpack()).collect();
    // Both trials run in full, with no early screen and no rungs.
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
    // `a` and `b` hold the standard and the fine results, in the order of
    // `elites`.
    let a = common::score_creatures(&mut engine, &creatures, &standard)?;
    let b = common::score_creatures(&mut engine, &creatures, &fine)?;
    // `kept` and `lost` count the elites whose fine distance is at least 80% of
    // their standard distance, and the others. `held` counts the elites whose
    // fine distance is at least 80% of the stored score. `stale` counts the
    // elites whose standard replay is off the stored score by more than 5% of
    // it or 0.05 m, whichever is larger. The first six are listed.
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
    // Fine distance over standard distance per elite, with the standard
    // distance floored at 0.01 m.
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
    // The elites that lose more than 70% of their distance at the fine trial.
    // The first eight are listed.
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
