//! Where does a creature's forward motion come from? Replays a champion with
//! the legacy `physics::step` loop and sums horizontal momentum changes by source.
//! This legacy solver differs from the evaluation and replay engine; use
//! `size_report` with `EVOLUTION_LEDGER=1` for current-physics diagnostics.
//! Usage: cargo run --release --example momentum_ledger <champion.json>
use evolution_simulator::{config::Config, evolution::Creature, physics};
fn main() {
    evolution_simulator::engine::lower_thread_priority();
    let path = std::env::args().nth(1).expect("champion json");
    let json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    let body = if json.get("creature").is_some() {
        json["creature"].clone()
    } else {
        json
    };
    let creature: Creature = serde_json::from_value(body).unwrap();
    let cfg = Config::default();
    let mut c = creature.clone();
    evolution_simulator::evolution::canonicalize_bone_order(&mut c);
    let mut n = physics::nodes(&c);
    let mass: f32 = n.iter().map(|x| x.mass).sum();
    for tick in 0..physics::settle() {
        physics::step(&mut n, &c.bones, &c.muscles, &cfg, tick);
    }
    physics::MOMENTUM_LEDGER.with(|l| l.set([0.0; 5]));
    let start: f32 = n.iter().map(|x| x.pos[0] * x.mass).sum::<f32>() / mass;
    for tick in physics::settle()..physics::settle() + cfg.steps() {
        physics::step(&mut n, &c.bones, &c.muscles, &cfg, tick);
    }
    let end: f32 = n.iter().map(|x| x.pos[0] * x.mass).sum::<f32>() / mass;
    let l = physics::MOMENTUM_LEDGER.with(|l| l.get());
    println!(
        "{} nodes, {} muscles, total mass {:.3} kg",
        c.nodes.len(),
        c.muscles.len(),
        mass
    );
    println!(
        "center of mass moved {:.3} m over {} s (legacy CPU physics)",
        end - start,
        cfg.duration
    );
    engines(&creature, &cfg);
    let names = [
        "integration speed cap",
        "ground contact",
        "velocity-pass speed cap",
        "velocity-pass constraints",
        "projection COM shift",
    ];
    println!("Legacy physics::step ledger (kg*m/s summed over the trial):");
    for (name, v) in names.iter().zip(l) {
        println!("{name:28} {v:+10.3} kg*m/s summed");
    }
}

/// Scores the same creature on the RTX and CPU engines for comparison.
#[allow(dead_code)]
fn engines(creature: &Creature, cfg: &Config) {
    use evolution_simulator::engine::{self, Engine};
    let mut pop = evolution_simulator::evolution::Population::default();
    for _ in 0..32 {
        pop.push(creature.clone());
    }
    let mut list: Vec<Box<dyn Engine>> = vec![Box::new(engine::gpu_engine("RTX", 64, 64).unwrap())];
    let threads = engine::cpu_threads();
    if threads > 0 {
        list.push(Box::new(engine::cpu_engine(threads).unwrap()));
    }
    for e in &mut list {
        e.submit(pop.clone(), cfg).unwrap();
        let done = loop {
            if let Some(d) = e.poll().unwrap() {
                break d;
            }
            e.wait(std::time::Duration::from_millis(20));
        };
        println!("{:45} fitness {:.3} m", e.name(), done.results[0].fitness);
    }
    println!(
        "{:45} fitness {:.3} m",
        "CPU evaluation (replay physics)",
        physics::evaluate(creature, cfg)
    );
    let l = *evolution_simulator::cpu_engine::LEDGER.lock().unwrap();
    let names = [
        "integration speed cap",
        "ground contact",
        "velocity-pass speed cap",
        "velocity-pass constraints",
        "projection COM shift",
        "muscle forces",
    ];
    println!("Engine physics ledger (lane 0, kg*m/s summed over the trial):");
    for (name, v) in names.iter().zip(l) {
        println!("  {name:28} {v:+10.3}");
    }
}
