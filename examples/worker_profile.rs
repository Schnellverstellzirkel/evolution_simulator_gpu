//! Times the steady-state worker's CPU stages (archive insertion, breeding,
//! generation boundary) without a GPU. Scores and behaviors are made up from
//! a hash of the slot, so the archive fills and turns over like a real run.
//! Usage: worker_profile [population] [unit] [generations]
//! Set EVOLUTION_PROFILE_BREED=1 for the inner timings.
use evolution_simulator::config::Config;
use evolution_simulator::qd::TrialMetrics;
use evolution_simulator::storage::Experiment;
use std::time::Instant;

fn fake(e: &mut Experiment, slots: &[usize], round: u64) {
    for &i in slots {
        let h = (i as u64 ^ (round << 40)).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 20;
        let growth = 1.0 + e.generation as f32 * 0.08;
        e.scores[i] = (1.0 + (h % 1000) as f32 * 0.02) * growth;
        e.trial_metrics[i] = TrialMetrics {
            ground_contact: ((h >> 10) % 6) as f32 / 6.0 + 0.05,
            gait_frequency: ((h >> 13) % 8) as f32 * 0.75 + 0.1,
            mean_height: ((h >> 16) % 6) as f32 * 0.3 + 0.05,
            feet: ((h >> 19) % 5) as f32,
            ..Default::default()
        };
    }
}

fn main() -> anyhow::Result<()> {
    let arg = |n: usize, d: usize| {
        std::env::args()
            .nth(n)
            .map_or(d, |v| v.parse().expect("number"))
    };
    let population = arg(1, 1_000_000);
    let unit = arg(2, 350_000);
    let generations = arg(3, 6);
    let started = Instant::now();
    let mut e = Experiment::new(Config {
        population,
        ..Config::default()
    })?;
    println!(
        "created {} in {:.2} s",
        population,
        started.elapsed().as_secs_f64()
    );
    let all: Vec<usize> = (0..population).collect();
    fake(&mut e, &all, 0);
    e.archive_slots(&all);
    e.breed_slots(&all)?;
    let mut round = 1u64;
    for g in 0..generations {
        let (mut archive, mut breed, mut subset, mut pack) = (0.0, 0.0, 0.0, 0.0);
        let mut failed = 0;
        let gen_start = Instant::now();
        for chunk in all.chunks(unit) {
            round += 1;
            fake(&mut e, chunk, round);
            e.arm_screen_early();
            let t = Instant::now();
            let unit_pop = e.population.subset(chunk);
            subset += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let idx: Vec<usize> = (0..chunk.len()).collect();
            let batches = evolution_simulator::creature_kernel::pack(&unit_pop, &idx)?;
            pack += t.elapsed().as_secs_f64();
            drop(batches);
            let t = Instant::now();
            failed += e.archive_slots(chunk);
            archive += t.elapsed().as_secs_f64();
            let t = Instant::now();
            e.breed_slots(chunk)?;
            breed += t.elapsed().as_secs_f64();
        }
        let t = Instant::now();
        e.finish_steady_generation(failed)?;
        let boundary = t.elapsed().as_secs_f64();
        let [plan, emit, write] = evolution_simulator::storage::take_breed_nanos();
        println!(
            "  subset {subset:.2} s, pack {pack:.2} s\n  breed split: plan {:.2} s, emit {:.2} s, write {:.2} s",
            plan as f64 * 1e-9,
            emit as f64 * 1e-9,
            write as f64 * 1e-9
        );
        println!(
            "gen {g}: archive {archive:.2} s, breed {breed:.2} s, boundary {boundary:.2} s, total {:.2} s, elites {}",
            gen_start.elapsed().as_secs_f64(),
            e.archive.entries.len()
        );
    }
    Ok(())
}
