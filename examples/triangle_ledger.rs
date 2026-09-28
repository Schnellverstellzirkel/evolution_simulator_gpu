//! Where does a triangle's forward motion come from? A triangle is 3 nodes,
//! 2 bones and 1 muscle. This samples random triangles, replays each on the
//! CPU engine with the momentum and energy ledgers on, and prints the ledger
//! of the fastest ones. Momentum entries are kg*m/s summed over the trial;
//! "projection COM shift" is the center-of-mass move of the bone passes and
//! the rebuild (the planted-feet push), times mass over dt.
//! Usage: cargo run --release --example triangle_ledger [samples] [seed]
use evolution_simulator::{
    config::Config,
    cpu_engine,
    evolution::{Bone, Creature, Muscle, NodeGene},
    physics,
};

fn next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / (1u64 << 24) as f32
}

fn triangle(state: &mut u64) -> Creature {
    let upper = 0.25 + 0.5 * next(state);
    let lower = 0.25 + 0.5 * next(state);
    let angle = 0.6 + 1.2 * next(state);
    let nodes = vec![
        NodeGene {
            x: 0.0,
            y: 0.3,
            diameter: 0.06 + 0.1 * next(state),
            friction: 0.4 + 0.6 * next(state),
        },
        NodeGene {
            x: upper,
            y: 0.3,
            diameter: 0.06 + 0.1 * next(state),
            friction: 0.4 + 0.6 * next(state),
        },
        NodeGene {
            x: upper + lower * angle.cos(),
            y: 0.3 - lower * angle.sin(),
            diameter: 0.06 + 0.1 * next(state),
            friction: 0.4 + 0.6 * next(state),
        },
    ];
    let bones = vec![Bone::new(0, 1, upper), Bone::new(1, 2, lower)];
    let close = (upper * upper + lower * lower - 2.0 * upper * lower * (-angle.cos())).sqrt();
    let muscles = vec![Muscle {
        bone_a: 0,
        bone_b: 1,
        anchor_a: 0.0,
        anchor_b: 1.0,
        short: close * (0.4 + 0.4 * next(state)),
        long: close * (1.0 + 0.3 * next(state)),
        period: 0.2 + 0.8 * next(state),
        phase: next(state),
        duty: 0.3 + 0.4 * next(state),
        stiffness: 60.0 + 100.0 * next(state),
        sensor: 255,
        reset: 0.0,
    }];
    Creature {
        nodes,
        bones,
        muscles,
        id: 0,
        mutability: 1.0,
    }
}

fn main() {
    let samples: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let mut state: u64 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    // SAFETY: set before any other thread starts.
    unsafe { std::env::set_var("EVOLUTION_LEDGER", "1") };
    let cfg = Config {
        random_seed: false,
        duration: 20.0,
        ..Config::default()
    };
    let names = [
        "integration speed cap",
        "ground contact",
        "velocity-pass speed cap",
        "velocity-pass constraints",
        "projection COM shift",
        "muscle forces",
    ];
    let mut rows = Vec::new();
    for _ in 0..samples {
        let creature = triangle(&mut state);
        *cpu_engine::LEDGER.lock().unwrap() = [0.0; 6];
        *cpu_engine::ENERGY_LEDGER.lock().unwrap() = [0.0; 5];
        let (_, result) = cpu_engine::replay(&creature, &cfg);
        let mass: f32 = physics::nodes(&creature).iter().map(|n| n.mass).sum();
        let momentum = *cpu_engine::LEDGER.lock().unwrap();
        let energy = *cpu_engine::ENERGY_LEDGER.lock().unwrap();
        let cost = cpu_engine::transport_cost(&creature, &cfg);
        rows.push((
            result.fitness,
            mass,
            result.fall_time,
            momentum,
            energy,
            cost,
        ));
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    let fitness: Vec<f32> = rows.iter().map(|r| r.0).collect();
    println!(
        "{samples} random triangles, 20 s: best {:.2} m, median {:.2} m, p90 {:.2} m",
        fitness[0],
        fitness[samples / 2],
        fitness[samples / 10]
    );
    for (rank, (fit, mass, fall, l, e, cost)) in rows.iter().take(5).enumerate() {
        println!(
            "#{rank}: {fit:.2} m, {mass:.2} kg, fall {fall:.2} s, cost of transport {}",
            cost.map_or("n/a".into(), |c| format!("{c:.1} J/kg/m"))
        );
        for (name, v) in names.iter().zip(l) {
            println!("    {name:28} {v:+10.2} kg*m/s");
        }
        println!(
            "    muscle work {:.1} J, gained otherwise {:.1} J, lost {:.1} J",
            e[0], e[1], e[2]
        );
    }
    // Sum over the fastest tenth: the share of forward momentum per source.
    let top = &rows[..(samples / 10).max(1)];
    let mut sum = [0.0f64; 6];
    for r in top {
        for (s, v) in sum.iter_mut().zip(r.3) {
            *s += v;
        }
    }
    println!("fastest tenth, summed:");
    for (name, v) in names.iter().zip(sum) {
        println!("    {name:28} {v:+10.2} kg*m/s");
    }
}
