//! Steps one creature (a JSON file from `filmstrip` with EVOLUTION_FILM_DUMP)
//! under the selected physics and prints, per step, its fastest node and the
//! body's center, up to the first step whose fastest node passes a limit.
//! Usage: cargo run --release --example physics_trace -- <creature.json> [limit m/s]
use evolution_simulator::{config::Config, cpu_engine, evolution::Creature};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("creature json");
    let limit: f32 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(50.0);
    let creature: Creature = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let cfg = Config {
        random_seed: false,
        duration: 60.0,
        ..Config::default()
    };
    let (frames, result) = cpu_engine::replay(&creature, &cfg);
    let rate = cfg.fidelity().rate as f32;
    let start = evolution_simulator::physics::settle() as usize;
    println!("result {:?}", result.fitness);
    for (t, pair) in frames[start..].windows(2).enumerate() {
        let (mut fastest, mut who) = (0.0f32, 0);
        for (i, (a, b)) in pair[0].iter().zip(&pair[1]).enumerate() {
            let v = (b[0] - a[0]).hypot(b[1] - a[1]) * rate;
            if v > fastest {
                fastest = v;
                who = i;
            }
        }
        if t % 30 == 0 || fastest > limit {
            println!(
                "step {t}: fastest node {who} at {fastest:.2} m/s, positions {:?}",
                pair[1]
            );
        }
        if fastest > limit {
            break;
        }
    }
    Ok(())
}
