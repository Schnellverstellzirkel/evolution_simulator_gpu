//! Muscle energy, muscle force and ground reaction of a recorded trial.
//!
//! The recorded frames hold only node positions, whichever engine scored the
//! trial. This module rebuilds the actuator state from them with the same
//! formulas the kernels use: a muscle only pulls, its drive scales with its
//! stored energy, and work drains the store while the rest of the time refills
//! it. Velocities come from differences between frames, so the numbers are
//! estimates for viewing. They never feed a score.

use crate::config::Config;
use crate::evolution::Creature;
use crate::physics::{self, Node};

/// Per-frame view data of one trial.
#[derive(Default)]
pub struct Forces {
    /// `[frame][muscle]`: stored energy, 1 is rested and 0 is spent.
    pub energy: Vec<Vec<f32>>,
    /// `[frame][muscle]`: force along the muscle (N), positive pulls its ends together.
    pub muscle: Vec<Vec<f32>>,
    /// `[frame][node]`: estimated ground push on the node (N), 0 in the air.
    pub ground: Vec<Vec<f32>>,
    /// `[frame][node]`: friction force on the node (N) when the frames carry
    /// the recorded contact forces; empty for an estimate.
    pub friction: Vec<Vec<f32>>,
}

fn along_bone(frame: &[[f32; 2]], bone: &crate::evolution::Bone, t: f32) -> [f32; 2] {
    let a = frame[bone.a as usize];
    let b = frame[bone.b as usize];
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Rebuilds the forces of a trial. `contact[frame][node]` says which nodes
/// touch the ground and `fall` is the frame at which the trial ended.
pub fn analyze(
    creature: &Creature,
    nodes: &[Node],
    frames: &[Vec<[f32; 2]>],
    contact: &[Vec<bool>],
    fall: Option<u32>,
    config: &Config,
) -> Forces {
    let dt = physics::dt();
    let settle = physics::settle() as usize;
    let limits = physics::limits();
    let capacity = limits.muscle_energy * config.muscle_energy;
    let recovery = limits.muscle_recovery * config.muscle_recovery;
    let count = creature.muscles.len();
    let mut out = Forces::default();
    let mut energy = vec![1.0f32; count];
    for t in 0..frames.len() {
        out.energy.push(energy.clone());
        let mut force = vec![0.0f32; count];
        if t >= 1 && t >= settle {
            let (now, before) = (&frames[t], &frames[t - 1]);
            let time = (t - settle) as f32 * dt;
            let ended = fall.is_some_and(|f| t as u32 > f);
            for (j, m) in creature.muscles.iter().enumerate() {
                let (Some(ba), Some(bb)) = (
                    creature.bones.get(m.bone_a as usize),
                    creature.bones.get(m.bone_b as usize),
                ) else {
                    continue;
                };
                let pa = along_bone(now, ba, m.anchor_a);
                let pb = along_bone(now, bb, m.anchor_b);
                let qa = along_bone(before, ba, m.anchor_a);
                let qb = along_bone(before, bb, m.anchor_b);
                let d = [pb[0] - pa[0], pb[1] - pa[1]];
                let length = d[0].hypot(d[1]).max(1e-6);
                let dir = [d[0] / length, d[1] / length];
                let dv = [
                    ((pb[0] - qb[0]) - (pa[0] - qa[0])) / dt,
                    ((pb[1] - qb[1]) - (pa[1] - qa[1])) / dt,
                ];
                let relative = dv[0] * dir[0] + dv[1] * dir[1];
                let target_speed = if t > settle {
                    (physics::limited_target(m, time)
                        - physics::limited_target(m, (time - dt).max(0.0)))
                        / dt
                } else {
                    0.0
                };
                let drive = (-target_speed * m.stiffness * 0.25).max(0.0) * energy[j];
                let magnitude = if ended {
                    0.0
                } else {
                    (drive + relative * 0.15).clamp(-limits.muscle_force, limits.muscle_force)
                };
                let work = (magnitude * relative).abs() * dt;
                energy[j] = (energy[j] - work / capacity + recovery * dt * (1.0 - energy[j]))
                    .clamp(0.0, 1.0);
                force[j] = magnitude;
            }
        }
        out.muscle.push(force);
    }
    // Ground push from the vertical acceleration of nodes that touch, smoothed
    // over five frames because the contact solver works in position steps.
    let raw: Vec<Vec<f32>> = (0..frames.len())
        .map(|t| {
            (0..nodes.len())
                .map(|i| {
                    let touching = contact
                        .get(t)
                        .and_then(|c| c.get(i))
                        .copied()
                        .unwrap_or(false);
                    if !touching || t == 0 || t + 1 >= frames.len() || t < settle {
                        return 0.0;
                    }
                    let a = (frames[t + 1][i][1] - 2.0 * frames[t][i][1] + frames[t - 1][i][1])
                        / (dt * dt);
                    (nodes[i].mass * (a + config.gravity)).max(0.0)
                })
                .collect()
        })
        .collect();
    for t in 0..frames.len() {
        let (lo, hi) = (t.saturating_sub(2), (t + 3).min(frames.len()));
        out.ground.push(
            (0..nodes.len())
                .map(|i| raw[lo..hi].iter().map(|f| f[i]).sum::<f32>() / (hi - lo) as f32)
                .collect(),
        );
    }
    out
}
