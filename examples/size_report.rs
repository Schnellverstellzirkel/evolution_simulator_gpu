//! Body size and foot slip of the fastest archive elites: how long each body
//! is, what it weighs, and how far its feet slide while touching the ground.
//! With `EVOLUTION_LEDGER` set, it also prints where the three fastest
//! bodies' forward momentum comes from.
//! Usage: cargo run --release --example size_report <checkpoint.evo> [count]
use evolution_simulator::{cpu_engine, evolution::Population, physics, storage};
fn main() {
    let path = std::env::args().nth(1).expect("checkpoint");
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let e = storage::load(std::path::Path::new(&path)).unwrap();
    let mut elites: Vec<_> = e.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let settle = physics::settle() as usize;
    let amplitude = physics::terrain_amplitude(e.config.terrain);
    println!(
        "distance_m  nodes  length_m  longest_bone_m  mass_kg  slip_m  slip_share  peak_head_g"
    );
    let mut lengths = Vec::new();
    let mut shares = Vec::new();
    for elite in elites.iter().take(count) {
        let c = &elite.creature;
        let nodes = physics::nodes(c);
        let mass: f32 = nodes.iter().map(|n| n.mass).sum();
        let length: f32 = c.bones.iter().map(|b| b.rest_length).sum();
        let longest = c.bones.iter().map(|b| b.rest_length).fold(0.0, f32::max);
        let frames = cpu_engine::trajectory(c, &e.config);
        // A node slides when it moves sideways while resting on the ground.
        let mut slip = 0.0f32;
        for t in settle + 1..frames.len() {
            for (j, node) in nodes.iter().enumerate() {
                let (now, before) = (frames[t][j], frames[t - 1][j]);
                let floor = physics::terrain(now[0], amplitude).0 + node.radius;
                if now[1] <= floor + 0.002 && before[1] <= floor + 0.002 {
                    slip += (now[0] - before[0]).abs();
                }
            }
        }
        let share = slip / elite.fitness.abs().max(0.01);
        // Peak head acceleration after the first 0.1 s, in g.
        let rate = physics::rate() as f32;
        let mut peak = 0.0f32;
        // Head shaking: mean head acceleration over about 0.1 s, the measure the
        // head g-force limit uses. Single impacts average out; jiggling does not.
        let mut shake = 0.0f32;
        let mut shake_peak = 0.0f32;
        let alpha = (1.0 / (0.1 * rate)).min(1.0);
        for t in settle + 8..frames.len() {
            let v = |t: usize| {
                [
                    (frames[t][0][0] - frames[t - 1][0][0]) * rate,
                    (frames[t][0][1] - frames[t - 1][0][1]) * rate,
                ]
            };
            let (a, b) = (v(t), v(t - 1));
            let g = (a[0] - b[0]).hypot(a[1] - b[1]) * rate / 9.8;
            peak = peak.max(g);
            shake += (g - shake) * alpha;
            shake_peak = shake_peak.max(shake);
        }
        lengths.push(length);
        shares.push(share);
        println!(
            "{:10.1}  {:5}  {:8.2}  {:14.2}  {:7.2}  {:6.1}  {:10.2}  {:11.1}  {:7.1}",
            elite.fitness,
            c.nodes.len(),
            length,
            longest,
            mass,
            slip,
            share,
            peak,
            shake_peak
        );
    }
    lengths.sort_by(f32::total_cmp);
    shares.sort_by(f32::total_cmp);
    if !lengths.is_empty() {
        println!(
            "median body length {:.2} m, median slip per meter traveled {:.2}",
            lengths[lengths.len() / 2],
            shares[shares.len() / 2]
        );
    }
    // Per-node detail for the fastest elite: which nodes touch the ground,
    // how far each slides while touching, and what each weighs.
    if let Some(elite) = elites.first()
        && std::env::var_os("EVOLUTION_NODE_SLIP").is_some()
    {
        let c = &elite.creature;
        let nodes = physics::nodes(c);
        let frames = cpu_engine::trajectory(c, &e.config);
        println!("node  mass_kg  contact_share  slip_m  lifts");
        for (j, node) in nodes.iter().enumerate() {
            let (mut touching, mut slip, mut lifts, mut was_down) = (0usize, 0.0f32, 0, false);
            for t in settle + 1..frames.len() {
                let (now, before) = (frames[t][j], frames[t - 1][j]);
                let floor = physics::terrain(now[0], amplitude).0 + node.radius;
                let down = now[1] <= floor + 0.002;
                if down {
                    touching += 1;
                    if before[1] <= floor + 0.002 {
                        slip += (now[0] - before[0]).abs();
                    }
                } else if was_down && now[1] > floor + 0.02 {
                    lifts += 1;
                }
                if down || now[1] > floor + 0.02 {
                    was_down = down;
                }
            }
            let share = touching as f32 / (frames.len() - settle - 1) as f32;
            println!(
                "{j:4}  {:7.2}  {share:13.2}  {slip:6.1}  {lifts:5}",
                node.mass
            );
        }
    }
    if std::env::var_os("EVOLUTION_LEDGER").is_none() {
        return;
    }
    let names = [
        "integration speed cap",
        "ground contact",
        "velocity-pass speed cap",
        "velocity-pass constraints",
        "projection COM shift",
        "muscle forces",
    ];
    for elite in elites.iter().take(3) {
        *cpu_engine::LEDGER.lock().unwrap() = [0.0; 6];
        let mut pop = Population::default();
        pop.push(elite.creature.clone());
        let result = cpu_engine::evaluate(&pop, &e.config);
        let mass: f32 = physics::nodes(&elite.creature).iter().map(|n| n.mass).sum();
        println!(
            "ledger for {:.1} m ({:.0} kg), kg*m/s over the trial:",
            result[0].fitness, mass
        );
        for (name, v) in names.iter().zip(*cpu_engine::LEDGER.lock().unwrap()) {
            println!("  {name:28} {v:+12.1}");
        }
    }
}
