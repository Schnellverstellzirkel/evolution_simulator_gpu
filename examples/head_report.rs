//! Posture check for archive elites: when (if ever) each creature fell over,
//! and how far its body turned over the trial (a roller turns whole turns).
//! Usage: cargo run --release --example head_report <checkpoint.evo> [count]
use evolution_simulator::{cpu_engine, physics, storage};
fn main() {
    let path = std::env::args().nth(1).expect("checkpoint");
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let e = storage::load(std::path::Path::new(&path)).unwrap();
    let mut elites: Vec<_> = e.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let rate = physics::rate() as f32;
    let settle = physics::settle() as usize;
    println!("fitness  nodes  fell_at_s  body_turns  widest_neck_tilt_deg");
    for elite in elites.iter().take(count) {
        let c = &elite.creature;
        let base = c.bones[0].b as usize;
        let frames = cpu_engine::trajectory(c, &e.config);
        let neck = |t: usize| {
            let (h, b) = (frames[t][0], frames[t][base]);
            (h[0] - b[0]).atan2(h[1] - b[1])
        };
        let mut fell = None;
        let mut turned = 0.0f32;
        let mut widest = 0.0f32;
        for t in settle + 1..frames.len() {
            let tilt = neck(t);
            let step = (tilt - neck(t - 1) + std::f32::consts::PI)
                .rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;
            turned += step;
            widest = widest.max(tilt.abs());
            if fell.is_none() && frames[t][0][1] < frames[t][base][1] {
                fell = Some((t - settle) as f32 / rate);
            }
        }
        println!(
            "{:7.2}  {:5}  {:>9}  {:10.2}  {:20.0}",
            elite.fitness,
            c.nodes.len(),
            fell.map_or("-".into(), |s| format!("{s:.1}")),
            turned / std::f32::consts::TAU,
            widest.to_degrees()
        );
    }
    let results = cpu_engine::evaluate(
        &{
            let mut pop = evolution_simulator::evolution::Population::default();
            for elite in &e.archive.entries {
                pop.push(elite.creature.clone());
            }
            pop
        },
        &e.config,
    );
    let fallen = results.iter().filter(|r| r.fall_time > 0.0).count();
    println!(
        "archive: {fallen} of {} elites fall over during the trial",
        results.len()
    );
}
