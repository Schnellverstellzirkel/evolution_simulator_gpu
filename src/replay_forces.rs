//! Stamina, muscle force and ground reaction of a recorded trial.
//!
//! The kernel records these with every frame (`engine::Recording`). When a
//! replay holds only node positions, this module rebuilds the actuator state
//! from them with the formulas the kernel uses: a muscle only pulls, its
//! force scales with the creature's stamina, and the work of all the muscles
//! drains the one store while the rest of the time refills it. Velocities
//! come from differences between frames, so the numbers are estimates for
//! viewing. They never feed a score.

use crate::config::Config;
use crate::evolution::Creature;
use crate::physics::{self, Node};
use crate::physics2::Model;

/// Per-frame view data of one trial.
#[derive(Default)]
pub struct Forces {
    /// `[frame][muscle]`: the creature's stamina (the same for every muscle),
    /// 1 is rested and 0 is spent.
    pub energy: Vec<Vec<f32>>,
    /// `[frame][muscle]`: force along the muscle (N), positive pulls its ends together.
    pub muscle: Vec<Vec<f32>>,
    /// `[frame][node]`: estimated ground push on the node (N), 0 in the air.
    pub ground: Vec<Vec<f32>>,
    /// `[frame][node]`: friction force on the node (N) when the frames carry
    /// the recorded contact forces; empty for an estimate.
    pub friction: Vec<Vec<f32>>,
    /// `[frame]`: bit `j` set when bone `j`'s joint is past its break angle
    /// by the scoring kernel's test; empty for an estimate.
    pub broken: Vec<u64>,
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
    let recovery = physics::limits().muscle_recovery * config.muscle_recovery;
    let model = Model::new(creature, config);
    let capacity: f32 = model.muscles.iter().map(|m| m.store).sum::<f32>() * config.muscle_energy;
    let count = creature.muscles.len();
    let mut out = Forces::default();
    let mut stamina = 1.0f32;
    // The time of each node's last touchdown, once it has had one.
    let mut touched: Vec<Option<f32>> = vec![None; nodes.len()];
    for t in 0..frames.len() {
        out.energy.push(vec![stamina; count]);
        let mut force = vec![0.0f32; count];
        if t >= 1 && t >= settle {
            let (now, before) = (&frames[t], &frames[t - 1]);
            let time = (t - settle) as f32 * dt;
            let ended = fall.is_some_and(|f| t as u32 > f);
            for (i, down) in touched.iter_mut().enumerate() {
                let on = |frame: usize| contact.get(frame).and_then(|c| c.get(i)).copied().unwrap_or(false);
                if on(t) && !on(t - 1) && t > settle {
                    *down = Some(time);
                }
            }
            let mut power = 0.0f32;
            for (j, m) in creature.muscles.iter().enumerate() {
                let (a, b) = (m.node_a as usize, m.node_b as usize);
                let (Some(pa), Some(pb), Some(qa), Some(qb)) =
                    (now.get(a), now.get(b), before.get(a), before.get(b))
                else {
                    continue;
                };
                let d = [pb[0] - pa[0], pb[1] - pa[1]];
                let length = d[0].hypot(d[1]).max(1e-6);
                let dir = [d[0] / length, d[1] / length];
                let dv = [
                    ((pb[0] - qb[0]) - (pa[0] - qa[0])) / dt,
                    ((pb[1] - qb[1]) - (pa[1] - qa[1])) / dt,
                ];
                let relative = dv[0] * dir[0] + dv[1] * dir[1];
                let since = (m.sensor < 2)
                    .then(|| [a, b][m.sensor as usize])
                    .and_then(|node| touched.get(node).copied().flatten());
                let k = &model.muscles[j];
                let drive = k.cap
                    * physics::activation(m, time, since)
                    * stamina
                    * (1.0 + relative * k.hill).clamp(0.0, 1.0);
                let magnitude = if ended {
                    0.0
                } else {
                    (drive + relative * 0.15).clamp(-k.cap, k.cap)
                };
                power += if ended { 0.0 } else { drive * (-relative).max(0.0) };
                force[j] = magnitude;
            }
            stamina = (stamina - power * dt / capacity.max(1e-6) + recovery * dt * (1.0 - stamina))
                .clamp(0.0, 1.0);
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
