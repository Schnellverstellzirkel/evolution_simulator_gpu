//! Body size and foot slip of the fastest archive elites: how long each body
//! is, what it weighs, and how far its feet slide while touching the ground.
//! With `EVOLUTION_LEDGER` set, it also prints where the three fastest
//! bodies' forward momentum comes from.
//! Usage: cargo run --release --example size_report <checkpoint.evo> [count]
use evolution_simulator::{
    config::Config, cpu_engine, creature_kernel::GpuResult, evolution::Population, physics, storage,
};

struct ReplayMetrics {
    distance: f32,
    slip: f32,
}

fn replay_metrics(
    nodes: &[physics::Node],
    frames: &[Vec<[f32; 2]>],
    result: &GpuResult,
    cfg: &Config,
) -> ReplayMetrics {
    let fidelity = cfg.fidelity();
    let settle = fidelity.settle() as usize;
    let terminal = if result.fall_time > 0.0 {
        settle + (result.fall_time * fidelity.rate as f32).round() as usize
    } else {
        frames.len() - 1
    };
    let mass: f32 = nodes.iter().map(|node| node.mass).sum();
    let distance = frames[terminal]
        .iter()
        .zip(nodes)
        .map(|(position, node)| position[0] * node.mass)
        .sum::<f32>()
        / mass;
    let mut slip = 0.0;
    if cfg.ground {
        let amplitude = physics::terrain_amplitude(cfg.terrain);
        let touching = |position: [f32; 2], radius: f32| {
            let (height, slope) = physics::terrain(position[0], amplitude);
            let floor = height + radius * (1.0 + slope * slope).sqrt();
            position[1] <= floor + 0.002
        };
        // The first timed step includes recentering the settled body, which
        // is not travel. Compare only subsequent scored steps; limp motion
        // after the terminal frame must not inflate a frozen distance's slip.
        for t in settle + 2..=terminal {
            for (j, node) in nodes.iter().enumerate() {
                let (now, before) = (frames[t][j], frames[t - 1][j]);
                if touching(now, node.radius) && touching(before, node.radius) {
                    slip += (now[0] - before[0]).abs();
                }
            }
        }
    }
    ReplayMetrics { distance, slip }
}

fn main() {
    let path = std::env::args().nth(1).expect("checkpoint");
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let e = storage::load(std::path::Path::new(&path)).unwrap();
    let mut elites: Vec<_> = e.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let mut selected = Population::default();
    for elite in elites.iter().take(count) {
        selected.push(elite.creature.clone());
    }
    let results = cpu_engine::evaluate(&selected, &e.config);
    println!(
        "archive_m  replay_m  nodes  length_m  longest_bone_m  mass_kg  slip_m  slip_per_replay_m"
    );
    let mut lengths = Vec::new();
    let mut shares = Vec::new();
    for (elite, result) in elites.iter().take(count).zip(&results) {
        let c = &elite.creature;
        let nodes = physics::nodes(c);
        let mass: f32 = nodes.iter().map(|n| n.mass).sum();
        let length: f32 = c.bones.iter().map(|b| b.rest_length).sum();
        let longest = c.bones.iter().map(|b| b.rest_length).fold(0.0, f32::max);
        let frames = cpu_engine::trajectory(c, &e.config);
        let measured = replay_metrics(&nodes, &frames, result, &e.config);
        let share = measured.slip / measured.distance.abs().max(0.01);
        lengths.push(length);
        shares.push(share);
        println!(
            "{:9.1}  {:8.1}  {:5}  {:8.2}  {:14.2}  {:7.2}  {:6.1}  {:17.2}",
            elite.fitness,
            measured.distance,
            c.nodes.len(),
            length,
            longest,
            mass,
            measured.slip,
            share
        );
    }
    lengths.sort_by(f32::total_cmp);
    shares.sort_by(f32::total_cmp);
    if !lengths.is_empty() {
        println!(
            "median body length {:.2} m, median slip per replay meter {:.2}",
            lengths[lengths.len() / 2],
            shares[shares.len() / 2]
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use evolution_simulator::physics::Fidelity;

    fn config() -> Config {
        Config {
            duration: 0.05,
            fidelity: Some(Fidelity {
                rate: 60,
                bone_passes: 2,
                velocity_passes: 1,
            }),
            ..Config::default()
        }
    }

    fn node() -> physics::Node {
        physics::Node {
            radius: 0.1,
            mass: 1.0,
            ..physics::Node::default()
        }
    }

    fn frames(cfg: &Config, samples: &[[f32; 2]]) -> Vec<Vec<[f32; 2]>> {
        let mut frames = vec![vec![samples[0]]; cfg.fidelity().settle() as usize];
        frames.extend(samples.iter().map(|&sample| vec![sample]));
        frames
    }

    #[test]
    fn slip_stops_at_the_frame_where_scoring_stops() {
        let cfg = config();
        let frames = frames(&cfg, &[[0.0, 0.1], [1.0, 0.1], [2.0, 0.1], [50.0, 0.1]]);
        let result = GpuResult {
            fall_time: 2.0 / cfg.fidelity().rate as f32,
            ..GpuResult::default()
        };
        let measured = replay_metrics(&[node()], &frames, &result, &cfg);
        assert_eq!(measured.distance, 2.0);
        assert_eq!(measured.slip, 1.0);
    }

    #[test]
    fn slip_excludes_the_trial_start_recentering_jump() {
        let cfg = config();
        let frames = frames(&cfg, &[[100.0, 0.1], [0.0, 0.1], [0.25, 0.1], [0.5, 0.1]]);
        let measured = replay_metrics(&[node()], &frames, &GpuResult::default(), &cfg);
        assert_eq!(measured.distance, 0.5);
        assert_eq!(measured.slip, 0.5);
    }

    #[test]
    fn slip_uses_the_configured_fidelity_settling_boundary() {
        let cfg = Config {
            fidelity: Some(Fidelity::fine()),
            ..config()
        };
        let mut frames = frames(&cfg, &[[0.0, 0.1], [0.0, 0.1], [0.25, 0.1], [0.5, 0.1]]);
        // This motion occurs after standard settling but during fine settling.
        frames[physics::settle() as usize + 1][0][0] = 100.0;
        let measured = replay_metrics(&[node()], &frames, &GpuResult::default(), &cfg);
        assert_eq!(measured.slip, 0.5);
    }

    #[test]
    fn disabling_ground_disables_ground_slip_measurement() {
        let cfg = Config {
            ground: false,
            ..config()
        };
        let frames = frames(&cfg, &[[0.0, 0.1], [0.0, 0.1], [1.0, 0.1], [2.0, 0.1]]);
        let measured = replay_metrics(&[node()], &frames, &GpuResult::default(), &cfg);
        assert_eq!(measured.distance, 2.0);
        assert_eq!(measured.slip, 0.0);
    }

    #[test]
    fn sloped_contact_includes_the_nodes_normal_radius() {
        let cfg = Config {
            terrain: 4,
            ..config()
        };
        let amplitude = physics::terrain_amplitude(cfg.terrain);
        let point = |x| {
            let (height, slope) = physics::terrain(x, amplitude);
            [x, height + node().radius * (1.0 + slope * slope).sqrt()]
        };
        let frames = frames(&cfg, &[point(0.1), point(0.1), point(0.45)]);
        let measured = replay_metrics(&[node()], &frames, &GpuResult::default(), &cfg);
        assert!((measured.slip - 0.35).abs() < 1e-6);
    }

    #[test]
    fn descending_contact_checks_each_frames_own_ground_height() {
        let cfg = Config {
            terrain: 4,
            ..config()
        };
        let amplitude = physics::terrain_amplitude(cfg.terrain);
        let point = |x| [x, physics::terrain(x, amplitude).0 + node().radius];
        assert!(point(0.45)[1] > point(0.1)[1] + 0.02);
        let frames = frames(&cfg, &[point(0.45), point(0.45), point(0.1)]);
        let measured = replay_metrics(&[node()], &frames, &GpuResult::default(), &cfg);
        assert!((measured.slip - 0.35).abs() < 1e-6);
    }

    #[test]
    fn replay_distance_uses_node_masses() {
        let cfg = config();
        let mut nodes = [node(), node()];
        nodes[1].mass = 3.0;
        let frames = vec![vec![[1.0, 0.1], [3.0, 0.1]]; cfg.fidelity().settle() as usize + 4];
        let measured = replay_metrics(&nodes, &frames, &GpuResult::default(), &cfg);
        assert_eq!(measured.distance, 2.5);
    }
}
