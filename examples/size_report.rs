//! Prints the body size and the foot slip of the fastest `count` elites (12 by
//! default) in a save's global archive. Each elite is replayed on the GPU with
//! the scoring kernel, and its row gives the total and longest bone length, the
//! mass, the distance its nodes slide while they touch the ground, and the
//! head's acceleration. `EVOLUTION_NODE_SLIP` adds a table of contact and slip
//! for each node of the champion. The GPU does not report the muscle work of a
//! trial, so there is no cost of transport column.
//! Usage: cargo run --release --example size_report <checkpoint.evo> [count]
mod common;
use evolution_simulator::{config::Config, creature_kernel::GpuResult, physics, storage};

/// What `replay_metrics` measures in one replay.
struct ReplayMetrics {
    /// Center of mass x at the last scored frame (m).
    distance: f32,
    /// Slip of all nodes together (m).
    slip: f32,
    /// Contact and slip of each node.
    node_slip: Vec<NodeSlip>,
    /// Largest acceleration of the head, from its recorded positions (g).
    position_peak_g: f32,
    /// Largest value of that acceleration averaged over
    /// `physics::HEAD_SHAKE_WINDOW` (g).
    position_shake_peak_g: f32,
}

/// How one node met the ground over the scored steps.
#[derive(Clone, Copy, Default)]
struct NodeSlip {
    /// Share of the scored steps in which the node was down.
    contact_share: f32,
    /// Horizontal distance the node moved between two frames in which it was
    /// down (m).
    slip: f32,
    /// Times the node rose clear of the ground after being down.
    lifts: usize,
}

/// Measures one replay. `frames` holds the node positions before every step
/// and after the last. Frames 0 to `settle` all show the start pose, so the
/// timed trial runs from frame `settle`. Scoring ends at the frame of
/// `result.fall_time`, or at the last frame when the creature did not fall.
/// The distance is the center of mass there. With the ground on, each node's
/// contact, slip and lifts are counted over the steps up to that frame. A node
/// is down when its center is within 2 mm of the floor, and clear when it is
/// more than 2 cm above it. The floor is the ground height under the node plus
/// its radius measured along the ground's normal. It has the bumps and the
/// slope alone, so it misses gaps, hurdles and the earthquake.
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
    let mut node_slip = vec![NodeSlip::default(); nodes.len()];
    if cfg.ground && terminal > settle {
        let amplitude = physics::terrain_amplitude(cfg.terrain);
        let floor = |position: [f32; 2], radius: f32| {
            let (height, slope) = physics::terrain_with_slope(position[0], amplitude, cfg.slope);
            height + radius * (1.0 + slope * slope).sqrt()
        };
        for (j, (node, detail)) in nodes.iter().zip(&mut node_slip).enumerate() {
            let mut touching = 0usize;
            let mut was_down = false;
            for t in settle + 1..=terminal {
                let (now, before) = (frames[t][j], frames[t - 1][j]);
                let current_floor = floor(now, node.radius);
                let down = now[1] <= current_floor + 0.002;
                let clear = now[1] > current_floor + 0.02;
                if down {
                    touching += 1;
                    // Slip starts at the second timed step. The engine this
                    // tool was written for recentered the body in the first
                    // one. The kernel does not, and the tool still skips it.
                    if t >= settle + 2 && before[1] <= floor(before, node.radius) + 0.002 {
                        detail.slip += (now[0] - before[0]).abs();
                    }
                } else if was_down && clear {
                    detail.lifts += 1;
                }
                if down || clear {
                    was_down = down;
                }
            }
            detail.contact_share = touching as f32 / (terminal - settle) as f32;
        }
    }
    // The head's acceleration from its recorded positions, and its average
    // over `HEAD_SHAKE_WINDOW` the way the kernel averages its head shake. The
    // kernel keeps only the last average, so this also finds the peaks. Both
    // start a window and two steps into the trial.
    let rate = fidelity.rate as f32;
    let first_accel = settle + (physics::HEAD_SHAKE_WINDOW * rate).ceil() as usize + 2;
    let alpha = (1.0 / (physics::HEAD_SHAKE_WINDOW * rate)).min(1.0);
    let mut position_peak_g = 0.0f32;
    let mut position_shake = 0.0;
    let mut position_shake_peak_g = 0.0f32;
    for t in first_accel..=terminal {
        let velocity = |step: usize| {
            [
                (frames[step][0][0] - frames[step - 1][0][0]) * rate,
                (frames[step][0][1] - frames[step - 1][0][1]) * rate,
            ]
        };
        let (now, before) = (velocity(t), velocity(t - 1));
        let acceleration_g = (now[0] - before[0]).hypot(now[1] - before[1]) * rate / 9.8;
        position_peak_g = position_peak_g.max(acceleration_g);
        position_shake += (acceleration_g - position_shake) * alpha;
        position_shake_peak_g = position_shake_peak_g.max(position_shake);
    }
    ReplayMetrics {
        distance,
        slip: node_slip.iter().map(|node| node.slip).sum(),
        node_slip,
        position_peak_g,
        position_shake_peak_g,
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("checkpoint");
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let e = storage::load(std::path::Path::new(&path)).unwrap();
    // The engine records the replays, so it stays open until the end.
    let _engine = common::open()?;
    // The global archive's elites, fastest first.
    let mut elites: Vec<_> = e.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let node_detail = std::env::var_os("EVOLUTION_NODE_SLIP").is_some();
    let mut champion_detail = Vec::new();
    println!(
        "position-derived acceleration includes position corrections; engine_shake_g is the engine's recorded value at the scored endpoint"
    );
    // Columns: the archive's distance and the replay's, the node count, the
    // total and the longest bone length, the mass, the slip and the slip per
    // meter of replay distance, then three accelerations in g: the peak and the
    // averaged peak from the recorded positions, and the kernel's averaged
    // value at the end of scoring.
    println!(
        "archive_m  replay_m  nodes  length_m  longest_bone_m  mass_kg  slip_m  slip_per_replay_m  pos_peak_g  pos_shake_peak_g  engine_shake_g"
    );
    let mut lengths = Vec::new();
    let mut shares = Vec::new();
    for (rank, elite) in elites.iter().take(count).enumerate() {
        let c = &elite.creature.unpack();
        let nodes = physics::nodes(c);
        let mass: f32 = nodes.iter().map(|n| n.mass).sum();
        let length: f32 = c.bones.iter().map(|b| b.rest_length).sum();
        let longest = c.bones.iter().map(|b| b.rest_length).fold(0.0, f32::max);
        let recording = common::record(c, &e.config)?;
        let (frames, result) = (recording.frames, recording.result);
        let measured = replay_metrics(&nodes, &frames, &result, &e.config);
        let share = measured.slip / measured.distance.abs().max(0.01);
        lengths.push(length);
        shares.push(share);
        println!(
            "{:9.1}  {:8.1}  {:5}  {:8.2}  {:14.2}  {:7.2}  {:6.1}  {:17.2}  {:10.1}  {:16.1}  {:14.1}",
            elite.fitness,
            measured.distance,
            c.nodes.len(),
            length,
            longest,
            mass,
            measured.slip,
            share,
            measured.position_peak_g,
            measured.position_shake_peak_g,
            result.head_shake / 9.8,
        );
        if rank == 0 && node_detail {
            champion_detail = nodes
                .iter()
                .zip(measured.node_slip)
                .map(|(node, detail)| (node.mass, detail))
                .collect();
        }
    }
    // The median of an even count is the upper of the two middle values.
    lengths.sort_by(f32::total_cmp);
    shares.sort_by(f32::total_cmp);
    if !lengths.is_empty() {
        println!(
            "median body length {:.2} m, median slip per replay meter {:.2}",
            lengths[lengths.len() / 2],
            shares[shares.len() / 2]
        );
    }
    // Per-node detail for the fastest elite: which nodes touch the ground,
    // how far each slides while touching, and what each weighs.
    if !champion_detail.is_empty() {
        println!("node  mass_kg  contact_share  slip_m  lifts");
        for (j, (mass, detail)) in champion_detail.iter().enumerate() {
            println!(
                "{j:4}  {mass:7.2}  {:13.2}  {:6.1}  {:5}",
                detail.contact_share, detail.slip, detail.lifts,
            );
        }
    }
    Ok(())
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
        assert_eq!(measured.node_slip[0].contact_share, 0.0);
        assert_eq!(measured.node_slip[0].lifts, 0);
    }

    #[test]
    fn sloped_contact_includes_the_nodes_normal_radius() {
        let cfg = Config {
            terrain: 4,
            ..config()
        };
        let amplitude = physics::terrain_amplitude(cfg.terrain);
        let point = |x| {
            let (height, slope) = physics::terrain_with_slope(x, amplitude, cfg.slope);
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
        let point = |x| {
            [
                x,
                physics::terrain_with_slope(x, amplitude, cfg.slope).0 + node().radius,
            ]
        };
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

    #[test]
    fn node_contact_and_lifts_stop_at_the_scored_frame() {
        let cfg = config();
        let frames = frames(&cfg, &[[100.0, 0.1], [0.0, 0.1], [0.0, 0.13], [50.0, 0.1]]);
        let result = GpuResult {
            fall_time: 2.0 / cfg.fidelity().rate as f32,
            ..GpuResult::default()
        };
        let measured = replay_metrics(&[node()], &frames, &result, &cfg);
        assert_eq!(measured.node_slip[0].contact_share, 0.5);
        assert_eq!(measured.node_slip[0].lifts, 1);
        assert_eq!(measured.node_slip[0].slip, 0.0);
    }

    #[test]
    fn position_acceleration_excludes_recentering_and_unscored_motion() {
        let cfg = config();
        let mut samples: Vec<_> = (0..=12).map(|t| [t as f32 * 0.125, 0.1]).collect();
        samples[0][0] = 100.0;
        samples[11][0] = 50.0;
        let frames = frames(&cfg, &samples);
        let result = GpuResult {
            fall_time: 10.0 / cfg.fidelity().rate as f32,
            ..GpuResult::default()
        };
        let measured = replay_metrics(&[node()], &frames, &result, &cfg);
        assert_eq!(measured.position_peak_g, 0.0);
        assert_eq!(measured.position_shake_peak_g, 0.0);
    }
}
