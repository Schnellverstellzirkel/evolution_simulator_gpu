//! Replays one creature (JSON) with the current physics and prints its
//! distance and the ledgers used for the physics v2 comparison: where its
//! horizontal momentum came from, the muscles' work against the mechanical
//! energy it gained otherwise, and friction that pushed a node along the
//! slip it had after the step (lane 0 of the CPU engine, `EVOLUTION_LEDGER`).
//! Usage: cargo run --release --example v1_ledger -- <creature.json> [seconds]
use evolution_simulator::{config::Config, cpu_engine, evolution::Creature, physics};

fn main() -> anyhow::Result<()> {
    // SAFETY: set before any other thread starts.
    unsafe { std::env::set_var("EVOLUTION_LEDGER", "1") };
    let args: Vec<String> = std::env::args().collect();
    let creature: Creature = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let seconds: f32 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let cfg = Config {
        random_seed: false,
        duration: seconds,
        ..Config::default()
    };
    let (frames, result) = cpu_engine::replay(&creature, &cfg);
    let nodes = physics::nodes(&creature);
    let mass: f32 = nodes.iter().map(|n| n.mass).sum();
    let rate = cfg.fidelity().rate as f32;
    let trial = &frames[physics::settle() as usize..];
    let mut fastest = 0.0f32;
    let mut airborne = 0usize;
    for pair in trial.windows(2) {
        for (a, b) in pair[0].iter().zip(&pair[1]) {
            fastest = fastest.max((b[0] - a[0]).hypot(b[1] - a[1]) * rate);
        }
        if pair[1]
            .iter()
            .zip(&nodes)
            .all(|(p, n)| p[1] > n.radius + 0.002)
        {
            airborne += 1;
        }
    }
    let l = *cpu_engine::LEDGER.lock().unwrap();
    let e = *cpu_engine::ENERGY_LEDGER.lock().unwrap();
    let dt = f64::from(cfg.fidelity().dt());
    println!(
        "{} nodes, {} muscles, {:.2} kg: {:.2} m in {seconds} s ({:.2} m/s), fell at {:.2} s; fastest node {:.1} m/s; no node on the ground in {:.0}% of steps",
        creature.nodes.len(),
        creature.muscles.len(),
        mass,
        result.fitness,
        result.fitness / seconds,
        result.fall_time,
        fastest,
        100.0 * airborne as f32 / trial.len().max(1) as f32
    );
    println!(
        "momentum sources (N s): bone passes (planted feet) {:.2} (a {:.2} m center shift), contact friction {:.2}, muscles {:.2}, speed caps {:.2} and {:.2}, velocity passes {:.2}",
        l[4],
        l[4] * dt / f64::from(mass),
        l[1],
        l[5],
        l[0],
        l[2],
        l[3]
    );
    println!(
        "energy: muscle work {:.1} J, gained otherwise {:.1} J, lost {:.1} J; contact friction {:.1} N s, of which {:.1} N s pushed a node along its slip after the step",
        e[0], e[1], e[2], e[3], e[4]
    );
    Ok(())
}
