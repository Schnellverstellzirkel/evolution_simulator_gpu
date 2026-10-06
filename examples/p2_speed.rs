//! Times the GPU engine on the population of a save, under physics v2: one
//! warm-up pass, then `repeats` timed passes over the same creatures.
//! Prints creatures/s and creature-steps/s (steps a creature simulated before
//! it fell or finished), then the same for each lane class alone with its
//! muscle-rounds histogram, and a hash of every creature's result bits, which
//! two runs of one population compare.
//!
//! Usage: p2_speed <save.evo | dump.bin> [count] [repeats] [effects] [screen]
//! A save's creatures are every k-th one of its ring, `count` of them.
//! `effects` is a comma list of effect names (`Wind,Mud`) put on at their
//! first level over the save's world. With `screen` as the fifth argument the
//! timed passes run the game's standard trial: the 5 s screen stops creatures
//! below the bar (the 5 s distance the best tenth reached in the warm-up
//! pass, which runs every trial in full), where the default runs every trial
//! to its end. With `fine` it runs the confirmation trial's settings
//! (`scheduler::confirm_config`) over the whole creature list instead.
use evolution_simulator::{
    config::Config,
    engine::{self, Engine},
    evolution::Population,
    kernel, storage,
};
use std::time::{Duration, Instant};

type Run = (
    f64,
    f64,
    f64,
    u64,
    Vec<evolution_simulator::creature_kernel::GpuResult>,
);

fn run(engine: &mut impl Engine, pop: &Population, cfg: &Config) -> anyhow::Result<Run> {
    let start = Instant::now();
    engine.submit(pop.clone(), cfg)?;
    let done = loop {
        if let Some(done) = engine.poll()? {
            break done;
        }
        engine.wait(Duration::from_millis(5));
    };
    let seconds = start.elapsed().as_secs_f64();
    let rate = f64::from(cfg.fidelity().rate);
    let total = f64::from(cfg.duration) * rate;
    let steps: f64 = done
        .results
        .iter()
        .map(|r| {
            let t = if r.fall_time > 0.0 {
                f64::from(r.fall_time) * rate
            } else if r.screened > 0.0 {
                f64::from(r.screened) * rate
            } else {
                total
            };
            t.min(total)
        })
        .sum();
    // FNV-1a over the result bits, in population order.
    let hash = bytemuck::cast_slice::<_, u8>(&done.results)
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
    Ok((
        pop.genomes.len() as f64 / seconds,
        steps / seconds,
        steps / done.busy_seconds.max(1e-9),
        hash,
        done.results,
    ))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("save path");
    let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let repeats: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(3);
    let mut pop = Population::default();
    let mut cfg = if path.ends_with(".bin") {
        // A creature dump (settings, population, elites) of a save this game
        // no longer reads.
        type Dump = (
            Config,
            Population,
            Vec<(evolution_simulator::evolution::Creature, Config, f32)>,
        );
        let (settings, all, _): Dump = bincode::deserialize(&std::fs::read(path)?)?;
        for i in 0..count.min(all.genomes.len()) {
            pop.push(all.creature(i));
        }
        settings
    } else {
        let e = storage::load(std::path::Path::new(path))?;
        // Every k-th creature of the ring, so the sample holds the mix of
        // every block. A slot's island is its number modulo 5 and its
        // nursery follows a period of 10, so k stays prime to 10.
        let mut stride = (e.ring_len() / count.max(1)).max(1);
        while stride.is_multiple_of(2) || stride.is_multiple_of(5) {
            stride += 1;
        }
        for i in (0..e.ring_len()).step_by(stride).take(count) {
            pop.push(e.creature(i));
        }
        e.config.clone()
    };
    cfg.screen = None;
    let screened = args.get(5).is_some_and(|v| v == "screen");
    let fine = args.get(5).is_some_and(|v| v == "fine");
    for name in args
        .get(4)
        .map_or("", |v| v.as_str())
        .split(',')
        .filter(|n| !n.is_empty())
    {
        let effect = evolution_simulator::environment::EFFECTS
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("no effect {name}"));
        let level = effect.level(&cfg) + 1;
        effect.set_level(&mut cfg, level);
    }
    if fine {
        cfg = evolution_simulator::scheduler::confirm_config(&cfg);
        cfg.screen = None;
    }
    eprintln!("world flags {:#x}", kernel::world_flags(&cfg));
    let mut engine = engine::gpu_engine("RTX 4060", 64)?;
    eprintln!("engine: {}", engine.name());
    let warm = run(&mut engine, &pop, &cfg)?;
    if screened {
        let bar = evolution_simulator::physics::screen_bar(
            warm.4.iter().map(|r| r.rung_trace().distance(2)),
            evolution_simulator::physics::screen_keep(),
        );
        eprintln!("screen bar {bar:.2} m at 5 s");
        cfg.screen = evolution_simulator::physics::screen_seconds()
            .map(|seconds| evolution_simulator::physics::Screen::uniform(seconds, bar));
    }
    for _ in 0..repeats {
        let (creatures, steps, busy, hash, _) = run(&mut engine, &pop, &cfg)?;
        println!(
            "{} creatures: {creatures:.0} creatures/s, {:.1}M creature-steps/s ({:.1}M per GPU-busy second), results {hash:016x}",
            pop.genomes.len(),
            steps / 1e6,
            busy / 1e6
        );
    }
    Ok(())
}
