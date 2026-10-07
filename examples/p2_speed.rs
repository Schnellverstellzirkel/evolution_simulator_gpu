//! Times the GPU engine on the creatures of a save: one warm-up pass, then
//! `repeats` timed passes over the same creatures. Each timed pass prints the
//! creatures per second, the creature-steps per second and the creature-steps
//! per GPU-busy second. A creature-step is a step that a creature simulated
//! before it fell, was stopped early or finished. The line ends with a hash of
//! every creature's result bits, which two runs of one population compare.
//!
//! Usage: p2_speed <save.evo | dump.bin> [count] [repeats] [effects] [screen|fine]
//! The defaults are 50,000 creatures and 3 repeats. A save's creatures are
//! every k-th one of its ring, `count` of them. A `dump.bin` is a bincode
//! creature dump (settings, population, elites) that an older version of the
//! game wrote, and the tool takes its first `count` creatures. It is not the
//! generation dump that `operator_yield` reads.
//! `effects` is a comma list of effect names (`Wind,Mud`, in any case). Each
//! named effect goes one level above its level in the save's world, up to its
//! top level. Give an empty string for no effects.
//! With `screen` as the fifth argument the timed passes run the 5 s screen of
//! the game's standard trial. The screen stops creatures below the bar, which
//! is the 5 s distance the best tenth reached in the warm-up pass. The warm-up
//! pass has no screen, and neither do the timed passes by default. With `fine`
//! the tool runs the confirmation trial's settings
//! (`scheduler::confirm_config`) over the whole creature list instead.
//! A save brings the early rungs that its audit window supports (R1 at 1 s, R2
//! at 2.5 s). A `dump.bin` has none. The rungs stay on in every mode except
//! `fine`. Set `EVOLUTION_NO_RUNGS` to measure without them.
use evolution_simulator::{
    config::Config,
    engine::{self, Engine},
    evolution::Population,
    kernel, storage,
};
use std::time::{Duration, Instant};

/// One pass: creatures per second, creature-steps per second, creature-steps
/// per GPU-busy second, the hash of the result bits, and the results.
type Run = (
    f64,
    f64,
    f64,
    u64,
    Vec<evolution_simulator::creature_kernel::GpuResult>,
);

/// Scores `pop` once on `engine` and times the pass from submit to result.
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
    // Steps each creature simulated: until it fell, until the screen or an
    // early rung stopped it (`screened`), or to the end of the trial.
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
        // no longer reads. Nothing in this repository writes the format now.
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
        // every block. The slots repeat in lanes of 10 (the main islands take
        // 8 lanes and the wild islands 2, see `qd::island_of_slot`), so k
        // stays prime to 10 and the sample takes every lane.
        let mut stride = (e.ring_len() / count.max(1)).max(1);
        while stride.is_multiple_of(2) || stride.is_multiple_of(5) {
            stride += 1;
        }
        for i in (0..e.ring_len()).step_by(stride).take(count) {
            pop.push(e.creature(i));
        }
        e.config.clone()
    };
    // No early screen until the `screen` argument sets the bar below.
    cfg.screen = None;
    let screened = args.get(5).is_some_and(|v| v == "screen");
    let fine = args.get(5).is_some_and(|v| v == "fine");
    // Each named effect goes one level up from the save's world.
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
    // The world effects that the kernel compiles in.
    eprintln!("world flags {:#x}", kernel::world_flags(&cfg));
    let mut engine = engine::gpu_engine("RTX 4060", 64)?;
    eprintln!("engine: {}", engine.name());
    let warm = run(&mut engine, &pop, &cfg)?;
    if screened {
        // The bar is the 5 s distance (rung 2) that the best tenth of the
        // warm-up pass reached.
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
