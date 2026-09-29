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

const GENES: usize = 14;

/// A triangle from genes in [0, 1].
fn triangle(u: &[f32; GENES]) -> Creature {
    let upper = 0.25 + 0.5 * u[0];
    let lower = 0.25 + 0.5 * u[1];
    let angle = 0.6 + 1.2 * u[2];
    let nodes = vec![
        NodeGene {
            x: 0.0,
            y: 0.3 + upper * 0.6,
            diameter: 0.06 + 0.1 * u[3],
            friction: 0.4 + 0.6 * u[4],
        },
        NodeGene {
            x: upper * 0.8,
            y: 0.3,
            diameter: 0.06 + 0.1 * u[5],
            friction: 0.4 + 0.6 * u[6],
        },
        NodeGene {
            x: upper * 0.8 + lower * angle.cos(),
            y: 0.3 - lower * angle.sin(),
            diameter: 0.06 + 0.1 * u[7],
            friction: 0.4 + 0.6 * u[8],
        },
    ];
    let head = (nodes[0].x - nodes[1].x).hypot(nodes[0].y - nodes[1].y);
    let tail = (nodes[2].x - nodes[1].x).hypot(nodes[2].y - nodes[1].y);
    let close = (nodes[0].x - nodes[2].x).hypot(nodes[0].y - nodes[2].y);
    let bones = vec![Bone::new(0, 1, head), Bone::new(1, 2, tail)];
    let muscles = vec![Muscle {
        bone_a: 0,
        bone_b: 1,
        anchor_a: 0.0,
        anchor_b: 1.0,
        short: close * (0.3 + 0.5 * u[9]),
        long: close * (1.0 + 0.4 * u[10]),
        period: 0.2 + 0.8 * u[11],
        phase: u[12],
        duty: 0.3 + 0.4 * u[13],
        stiffness: 60.0 + 100.0 * u[9],
        sensor: 255,
        reset: 0.0,
        tendon: 0.0,
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
    let v2 = evolution_simulator::physics2::enabled();
    let names = [
        "integration speed cap",
        "ground contact",
        "velocity-pass speed cap",
        "velocity-pass constraints",
        "projection COM shift",
        "muscle forces",
    ];
    let v2_names = [
        "ground and wind impulse (N s)",
        "body momentum change (N s)",
    ];
    let energy_names = [
        "muscle work (J)",
        "energy gained by the integrator (J)",
        "energy lost (J)",
        "kinetic energy added by the momentum balance (J)",
        "kinetic energy removed by the momentum balance (J)",
        "steps without ground contact",
        "steps",
        "gained in steps without contact (J)",
        "lost in steps without contact (J)",
        "friction impulse pushing a node the way it slid (N s)",
        "all friction impulse (N s)",
        "energy removed by the first-law check (J)",
    ];
    // Per creature: fitness, mass, fall time, momentum ledger, energy
    // ledger, cost of transport.
    type Row = (f32, f32, f32, Vec<f64>, Vec<f64>, Option<f32>);
    // A small evolution of triangles by distance: the best of `samples`
    // random ones, then hill-climbing with mutated copies.
    let evaluate = |genes: &[[f32; GENES]]| -> Vec<f32> {
        let mut pop = evolution_simulator::evolution::Population::default();
        for g in genes {
            pop.push(triangle(g));
        }
        cpu_engine::evaluate(
            &pop,
            &Config {
                screen: None,
                ..cfg.clone()
            },
        )
        .iter()
        .map(|r| r.fitness)
        .collect()
    };
    let mut pool: Vec<([f32; GENES], f32)> = Vec::new();
    let randoms: Vec<[f32; GENES]> = (0..samples)
        .map(|_| std::array::from_fn(|_| next(&mut state)))
        .collect();
    for (g, f) in randoms.iter().zip(evaluate(&randoms)) {
        pool.push((*g, f));
    }
    let generations: usize = std::env::var("TRIANGLE_GENERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    for _ in 0..generations {
        pool.sort_by(|a, b| b.1.total_cmp(&a.1));
        pool.truncate(20);
        let mut children: Vec<[f32; GENES]> = Vec::new();
        for parent in &pool {
            for _ in 0..(samples / 20).max(1) {
                let mut g = parent.0;
                for v in &mut g {
                    if next(&mut state) < 0.3 {
                        *v = (*v + (next(&mut state) - 0.5) * 0.3).clamp(0.0, 1.0);
                    }
                }
                children.push(g);
            }
        }
        for (g, f) in children.iter().zip(evaluate(&children)) {
            pool.push((*g, f));
        }
    }
    pool.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut rows: Vec<Row> = Vec::new();
    for (genes, _) in pool.iter().take(20) {
        let creature = triangle(genes);
        *cpu_engine::LEDGER.lock().unwrap() = [0.0; 6];
        *cpu_engine::ENERGY_LEDGER.lock().unwrap() = [0.0; 5];
        let (_, result) = cpu_engine::replay(&creature, &cfg);
        let mass: f32 = physics::nodes(&creature).iter().map(|n| n.mass).sum();
        let (momentum, energy) = if v2 {
            (
                evolution_simulator::physics2::LEDGER
                    .with(|l| l.get())
                    .to_vec(),
                evolution_simulator::physics2::ENERGY
                    .with(|l| l.get())
                    .to_vec(),
            )
        } else {
            (
                cpu_engine::LEDGER.lock().unwrap().to_vec(),
                cpu_engine::ENERGY_LEDGER.lock().unwrap().to_vec(),
            )
        };
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
    let mnames: &[&str] = if v2 { &v2_names } else { &names };
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "{samples} random triangles then {generations} generations of hill-climbing, 20 s, physics {}: top 20 best {:.2} m, 20th {:.2} m",
        if v2 { "v2" } else { "v1" },
        rows[0].0,
        rows[rows.len() - 1].0
    );
    for (rank, (fit, mass, fall, l, e, cost)) in rows.iter().take(5).enumerate() {
        println!(
            "#{rank}: {fit:.2} m, {mass:.2} kg, fall {fall:.2} s, cost of transport {}",
            cost.map_or("n/a".into(), |c| format!("{c:.1} J/kg/m"))
        );
        for (name, v) in mnames.iter().zip(l) {
            println!("    {name:52} {v:+10.2}");
        }
        if v2 {
            for (name, v) in energy_names.iter().zip(e) {
                println!("    {name:52} {v:+10.2}");
            }
        } else {
            println!(
                "    muscle work {:.1} J, gained otherwise {:.1} J, lost {:.1} J",
                e[0], e[1], e[2]
            );
        }
    }
    let top = &rows[..5];
    let mut sum = vec![0.0f64; mnames.len()];
    for r in top {
        for (s, v) in sum.iter_mut().zip(&r.3) {
            *s += v;
        }
    }
    println!("fastest five, summed:");
    for (name, v) in mnames.iter().zip(sum) {
        println!("    {name:52} {v:+10.2}");
    }
}
