//! Muscle energy, muscle force and ground push of a recorded trial.
//!
//! The scoring kernel records these with every frame (`engine::Recording`).
//! When a replay holds only node positions, `analyze` estimates them from the
//! differences between frames with a simplified form of the kernel's muscle
//! model. The numbers are for viewing and never feed a score.

use crate::config::Config;
use crate::evolution::Creature;
use crate::physics::{self, Node};

/// Per-frame muscle and ground data of one trial, as the scoring kernel
/// recorded it. `analyze` estimates `energy`, `muscle` and `ground` when a
/// replay has no recording.
#[derive(Default)]
pub struct Forces {
    /// `[frame][muscle]`: stored energy, 1 is rested and 0 is spent.
    pub energy: Vec<Vec<f32>>,
    /// `[frame][muscle]`: force along the muscle (N), positive pulls its ends together.
    pub muscle: Vec<Vec<f32>>,
    /// `[frame][node]`: ground push on the node (N), 0 in the air. An estimate
    /// from the node's vertical acceleration when the frames carry no
    /// recorded forces.
    pub ground: Vec<Vec<f32>>,
    /// `[frame][node]`: friction force on the node (N) when the frames carry
    /// the recorded contact forces; empty for an estimate.
    pub friction: Vec<Vec<f32>>,
    /// `[frame]`: bit `j` set when bone `j`'s joint is past its break angle
    /// by the scoring kernel's test; empty for an estimate.
    pub broken: Vec<u64>,
}

/// Position at fraction `t` of the way from the node `a` to the node `b` of
/// `bone` in `frame`.
fn along_bone(frame: &[[f32; 2]], bone: &crate::evolution::Bone, t: f32) -> [f32; 2] {
    let a = frame[bone.a as usize];
    let b = frame[bone.b as usize];
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Estimates the forces of a trial from its `frames` of node positions, which
/// start with the settling frames. `contact[frame][node]` says which nodes
/// touch the ground and `fall` is the frame at which the trial ended, after
/// which the muscles are limp. `friction` and `broken` stay empty. The muscle
/// model is a simplified form of the kernel's: it has no Hill factor, no
/// tendon, no touch sensors and no per-muscle strength.
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
    // The work (J) a rested muscle can do, and the share of its missing energy
    // it regains per second.
    let capacity = limits.muscle_energy * config.muscle_energy;
    let recovery = limits.muscle_recovery * config.muscle_recovery;
    let count = creature.muscles.len();
    let mut out = Forces::default();
    let mut energy = vec![1.0f32; count];
    for t in 0..frames.len() {
        // The store before the work of this frame.
        out.energy.push(energy.clone());
        let mut force = vec![0.0f32; count];
        // The first `settle` frames only show the start pose, so no muscle
        // works in them.
        if t >= 1 && t >= settle {
            let (now, before) = (&frames[t], &frames[t - 1]);
            let time = (t - settle) as f32 * dt;
            // After the trial ended the muscles are limp.
            let ended = fall.is_some_and(|f| t as u32 > f);
            for (j, m) in creature.muscles.iter().enumerate() {
                let (Some(ba), Some(bb)) = (
                    creature.bones.get(m.bone_a as usize),
                    creature.bones.get(m.bone_b as usize),
                ) else {
                    continue;
                };
                // The muscle's two anchors in this frame (p) and in the frame
                // before (q).
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
                // The speed at which the anchors move apart along the muscle,
                // negative while it shortens.
                let relative = dv[0] * dir[0] + dv[1] * dir[1];
                // How fast the rhythm's target length changes, negative while
                // it shortens.
                let target_speed = if t > settle {
                    (physics::limited_target(m, time)
                        - physics::limited_target(m, (time - dt).max(0.0)))
                        / dt
                } else {
                    0.0
                };
                // The kernel's drive: the shortening speed times the stiffness
                // (with its factor of 0.25) and the energy store. It never
                // pushes.
                let drive = (-target_speed * m.stiffness * 0.25).max(0.0) * energy[j];
                // The drive plus the kernel's damper (0.15 of the lengthening
                // speed), capped at the largest muscle force.
                let magnitude = if ended {
                    0.0
                } else {
                    (drive + relative * 0.15).clamp(-limits.muscle_force, limits.muscle_force)
                };
                // The store pays for the work of the force over the frame and
                // recovers toward 1.
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
                    // No push in the air, in the first and last frame, or
                    // while the body settles.
                    if !touching || t == 0 || t + 1 >= frames.len() || t < settle {
                        return 0.0;
                    }
                    // The node's vertical acceleration. The push is the mass
                    // times that acceleration plus gravity, when it is positive.
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
