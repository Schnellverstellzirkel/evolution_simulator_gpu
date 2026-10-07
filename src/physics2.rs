//! This module holds the physics constants and `Model`, a creature's
//! constants and start pose. `kernel::pack` fills the CUDA kernel's records
//! from a `Model`, and `docs/physics.md` describes the dynamics the kernel
//! runs. `Model` numbers the nodes so that bone `j` ends at node `j + 1` and
//! node 0 is the head. The start pose comes from the bones' rest angles, with
//! the center of mass over x = 0 and the lowest node on the ground.
use crate::{
    config::Config,
    evolution::{Creature, NO_SENSOR},
    physics,
};

/// The time constant (s) of passive joint damping. Each substep the kernel
/// takes the share `substep / time constant` of every joint's turning speed,
/// with equal and opposite pushes that keep the body's momentum.
pub fn joint_damping() -> f32 {
    0.1
}
/// The shortening speed, in muscle lengths per second, at which a muscle's
/// active pull is zero (Hill's force-velocity relation). The pull falls
/// linearly with the speed. A muscle's length here is its longest length, at
/// least 5 cm. This bounds a muscle's power as real muscle does, so a body
/// cannot catapult itself.
pub fn hill_speed() -> f32 {
    8.0
}
/// The largest acceleration (m/s^2) a muscle can give the mass it drives.
/// A muscle's cross-section, and so its force, grows with the mass it moves.
/// Its force cap is `DRIVEN_ACCELERATION` times the lighter of the two
/// subtrees it pulls together (a bone and everything it carries), never above
/// the fixed `Limits` cap. Its energy store scales the same way, because a
/// muscle's store is its own mass. Without this scaling the fixed cap would
/// drive a light limb at thousands of m/s^2.
pub const DRIVEN_ACCELERATION: f32 = 200.0;
/// Air drag on bones (kg/m^3). Every bone feels a force of
/// `AIR_DRAG x length x width x speed x velocity` against the velocity of its
/// midpoint, with the width the mean diameter of its two nodes. This is a
/// flat plate in the flow: half the air's density (1.2 kg/m^3) times a drag
/// coefficient of 1. One substep never takes more than half the bone's speed.
/// Large fast bodies pay for moving air, and the drag only takes energy away.
pub const AIR_DRAG: f32 = 0.6;
/// Water drag on a submerged bone (kg/m^3): the same law as the air's, in a
/// much thicker medium. The value is a fifth of half the water's density
/// (1,000 kg/m^3). The drag scales with the share of the bone under water. A
/// bone's sideways motion pays the full price and its lengthwise motion
/// `WATER_ALONG` of it. So a stroke that pushes water sideways has a net
/// reaction, and a reciprocal stroke does not cancel itself (a fish tail).
/// The push is limited like the air's.
pub const WATER_DRAG: f32 = 100.0;
/// The share of `WATER_DRAG` a bone meets moving along its own length.
pub const WATER_ALONG: f32 = 0.25;
/// A fully submerged node feels this share of its weight as buoyancy. A node
/// partly under water feels it times the submerged share of its diameter.
pub const WATER_BUOYANCY: f32 = 0.7;
/// How far (m) above its resting height on the ground a node may be and still
/// count as touching the ground in the behavior metrics.
pub(crate) const CONTACT_SLACK: f32 = 0.002;
/// How far (m) above its resting height on the ground a node must be to count
/// as lifted in the behavior metrics.
pub(crate) const LIFT_CLEARANCE: f32 = 0.01;

/// One muscle's constants.
#[derive(Clone, Debug)]
pub(crate) struct MuscleModel {
    /// Index of the first bone this muscle pulls.
    pub(crate) bone_a: usize,
    /// Index of the second bone this muscle pulls.
    pub(crate) bone_b: usize,
    /// Anchor point along `bone_a` (0 to 1 along bone length).
    pub(crate) anchor_a: f32,
    /// Anchor point along `bone_b` (0 to 1 along bone length).
    pub(crate) anchor_b: f32,
    /// Hill's relation as a factor on the shortening speed: 1 / (v_max
    /// times the muscle's length, at least 5 cm).
    pub(crate) hill: f32,
    /// Longest length (m), where the elastic tendon starts to pull, and the
    /// tendon's stiffness (N/m; 0 without one).
    pub(crate) long: f32,
    pub(crate) tendon_k: f32,
    /// Maximum contraction distance (m) during one cycle.
    pub(crate) amplitude: f32,
    /// Inverse of muscle contraction period (1/s).
    pub(crate) inv_period: f32,
    pub(crate) phase: f32,
    /// Fraction of cycle during active contraction.
    pub(crate) duty: f32,
    /// Inverse of duty cycle.
    pub(crate) inv_duty: f32,
    /// Inverse of the inactive fraction (1 - duty).
    pub(crate) inv_complement: f32,
    pub(crate) stiffness: f32,
    /// Force cap and energy store over the fixed `Limits` ones (at most 1):
    /// see `DRIVEN_ACCELERATION`.
    pub(crate) strength: f32,
    /// Node whose touchdown restarts the rhythm, if any.
    pub(crate) sensor: Option<usize>,
    /// Phase offset to apply on sensor contact.
    pub(crate) reset: f32,
}

/// A creature's constants for the physics. Nodes are renumbered so that
/// bone `j` ends at node `j + 1` (node 0 is the head), as the GPU kernel packs
/// them.
#[derive(Clone, Debug)]
pub struct Model {
    /// Mass of each node (kg).
    pub(crate) mass: Vec<f32>,
    pub(crate) radius: Vec<f32>,
    /// Friction coefficient of each node.
    pub(crate) friction: Vec<f32>,
    /// Sum of all node masses (kg).
    pub(crate) total_mass: f32,
    pub(crate) inv_mass: f32,
    /// Per bone: pivot node, child node (always the bone's index plus one),
    /// length, parent bone (`None` for the neck), and the relative-angle
    /// range.
    pub(crate) pivot: Vec<usize>,
    pub(crate) child: Vec<usize>,
    pub(crate) length: Vec<f32>,
    pub(crate) parent: Vec<Option<usize>>,
    pub(crate) lo: Vec<f32>,
    pub(crate) hi: Vec<f32>,
    /// Starting relative angle of every bone (the neck: its absolute angle).
    pub(crate) rest: Vec<f32>,
    pub(crate) muscles: Vec<MuscleModel>,
    /// Each muscle's force cap and energy store as multiples of the fixed
    /// `Limits` ones (1 unless muscle strength scales with the body).
    pub(crate) muscle_scale: f32,
    /// Earthquake bump phase and ground amplitude for this creature.
    pub(crate) quake_phase: f32,
    pub(crate) amplitude: f32,
    /// Starting position of each node (x, y coordinates).
    pub(crate) start: Vec<[f32; 2]>,
}

/// A creature's state: the head and the neck, then relative joint angles.
#[derive(Clone, Debug)]
pub struct State {
    /// Head position (x, y).
    pub(crate) x0: [f32; 2],
    /// Head velocity (x, y).
    pub(crate) v0: [f32; 2],
    /// Neck absolute angle (radians).
    pub(crate) th0: f32,
    /// Neck angular velocity (rad/s).
    pub(crate) w0: f32,
    /// Relative joint angles (radians).
    pub(crate) q: Vec<f32>,
    /// Relative joint angular velocities (rad/s).
    pub(crate) qd: Vec<f32>,
    /// Derived by `kinematics`: absolute bone angles and rates, node
    /// positions and velocities.
    pub(crate) th: Vec<f32>,
    pub(crate) om: Vec<f32>,
    pub(crate) pos: Vec<[f32; 2]>,
    pub(crate) vel: Vec<[f32; 2]>,
}

/// `a` wrapped into [-π, π).
pub(crate) fn wrap(a: f32) -> f32 {
    let t = std::f32::consts::TAU;
    a - t * ((a + std::f32::consts::PI) / t).floor()
}

impl Model {
    /// The model of a repaired creature (canonical bone order: bone `j`
    /// joins its parent node `a` to its child node `b`, bone 0 is the neck).
    pub fn new(c: &Creature, cfg: &Config) -> Model {
        let nodes = physics::nodes(c);
        let n = nodes.len();
        // Node `j + 1` is bone `j`'s child.
        let order: Vec<usize> = std::iter::once(0)
            .chain(c.bones.iter().map(|b| b.b as usize))
            .collect();
        let mut record = vec![0; n];
        for (r, &node) in order.iter().enumerate() {
            record[node] = r;
        }
        let mut parent_of_node = vec![None; n];
        for (j, b) in c.bones.iter().enumerate() {
            parent_of_node[b.b as usize] = Some(j);
        }
        let angle = |j: usize| {
            let b = c.bones[j];
            let (p, q) = (&c.nodes[b.a as usize], &c.nodes[b.b as usize]);
            (q.y - p.y).atan2(q.x - p.x)
        };
        let mut rest = Vec::with_capacity(c.bones.len());
        let mut parent = Vec::with_capacity(c.bones.len());
        let (mut lo, mut hi) = (Vec::new(), Vec::new());
        for (j, b) in c.bones.iter().enumerate() {
            // A bone at the head other than the neck turns against the neck,
            // as its joint does in the current physics (`physics::joints`).
            let up = parent_of_node[b.a as usize].or((j > 0).then_some(0));
            parent.push(up);
            let relative = match up {
                Some(p) => wrap(angle(j) - angle(p)),
                None => angle(j),
            };
            rest.push(relative);
            lo.push(relative + b.min_angle);
            hi.push(relative + b.max_angle);
        }
        let limits = physics::limits();
        let muscles = c
            .muscles
            .iter()
            .map(|m| {
                let ba = c.bones[m.bone_a as usize];
                let bb = c.bones[m.bone_b as usize];
                let ends = [ba.a, ba.b, bb.a, bb.b];
                MuscleModel {
                    bone_a: m.bone_a as usize,
                    bone_b: m.bone_b as usize,
                    anchor_a: m.anchor_a,
                    anchor_b: m.anchor_b,
                    hill: if hill_speed() > 0.0 {
                        1.0 / (hill_speed() * m.long.max(0.05))
                    } else {
                        0.0
                    },
                    long: physics::slack_length(&c.bones, &nodes, m),
                    tendon_k: 0.0,
                    amplitude: (m.long - m.short).min(
                        2.0 * limits.muscle_speed * m.period * m.duty.min(1.0 - m.duty)
                            / std::f32::consts::PI,
                    ),
                    inv_period: 1.0 / m.period,
                    phase: m.phase,
                    duty: m.duty,
                    inv_duty: 1.0 / m.duty,
                    inv_complement: 1.0 / (1.0 - m.duty),
                    stiffness: m.stiffness,
                    strength: 1.0,
                    sensor: (m.sensor != NO_SENSOR)
                        .then(|| record[ends[m.sensor as usize] as usize]),
                    reset: m.reset,
                }
            })
            .collect();
        // Muscle strength follows the mass a muscle drives: the lighter of the
        // two subtrees (a bone with everything it carries) it pulls together.
        let mut muscles: Vec<MuscleModel> = muscles;
        let mut subtree: Vec<f32> = (0..c.bones.len())
            .map(|j| nodes[order[j + 1]].mass + if j == 0 { nodes[order[0]].mass } else { 0.0 })
            .collect();
        for j in (1..c.bones.len()).rev() {
            if let Some(p) = parent[j] {
                subtree[p] += subtree[j];
            }
        }
        // Muscles on the same two bones and on the same side of the joint
        // share one strength: copies stacked on one joint (six on one pair
        // made a limb snap back and forth with six muscles' force) add timing,
        // not force. A flexor and an extensor sit on opposite sides and keep
        // a strength each.
        let group = |gene: &crate::evolution::Muscle| {
            let (a, b) = (gene.bone_a as usize, gene.bone_b as usize);
            let (ba, bb) = (c.bones[a], c.bones[b]);
            let pa = crate::evolution::bone_point(ba, &c.nodes, gene.anchor_a);
            let pb = crate::evolution::bone_point(bb, &c.nodes, gene.anchor_b);
            let shared = [ba.a, ba.b].into_iter().find(|n| *n == bb.a || *n == bb.b);
            let j = match shared {
                Some(n) => [c.nodes[n as usize].x, c.nodes[n as usize].y],
                None => [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]) - 1.0],
            };
            let cross = (pa[0] - j[0]) * (pb[1] - j[1]) - (pa[1] - j[1]) * (pb[0] - j[0]);
            (a.min(b), a.max(b), cross >= 0.0)
        };
        let groups: Vec<(usize, usize, bool)> = c.muscles.iter().map(group).collect();
        for (m, gene) in muscles.iter_mut().zip(&c.muscles) {
            let driven = subtree[m.bone_a].min(subtree[m.bone_b]);
            let key = group(gene);
            let sharing = groups.iter().filter(|&&g| g == key).count().max(1) as f32;
            m.strength = (DRIVEN_ACCELERATION * driven / limits.muscle_force).min(1.0) / sharing;
            // The tendon reaches the muscle's force cap when stretched by
            // `TENDON_STRETCH` of its longest length (at the stiffest gene).
            m.tendon_k = gene.tendon * limits.muscle_force * m.strength
                / (crate::evolution::TENDON_STRETCH * m.long.max(0.05));
        }
        let quake = crate::physics::quake_hash(c.id);
        let still = cfg.quake <= 0.0 || !cfg.ground;
        // Muscle strength over the fixed limits; 1 for every body today.
        let muscle_scale = 1.0;
        Model {
            mass: order.iter().map(|&i| nodes[i].mass).collect(),
            radius: order.iter().map(|&i| nodes[i].radius).collect(),
            friction: order.iter().map(|&i| nodes[i].friction).collect(),
            total_mass: order.iter().map(|&i| nodes[i].mass).sum(),
            inv_mass: 1.0 / order.iter().map(|&i| nodes[i].mass).sum::<f32>(),
            pivot: c.bones.iter().map(|b| record[b.a as usize]).collect(),
            child: (1..n).collect(),
            length: c.bones.iter().map(|b| b.rest_length).collect(),
            parent,
            lo,
            hi,
            rest,
            muscles,
            muscle_scale,
            quake_phase: if still {
                0.0
            } else {
                physics::quake_phase(quake)
            },
            amplitude: physics::terrain_amplitude(cfg.terrain)
                + if still {
                    0.0
                } else {
                    cfg.quake * physics::quake_scale(quake)
                },
            start: order
                .iter()
                .map(|&i| [c.nodes[i].x, c.nodes[i].y])
                .collect(),
        }
    }

    /// The ground's height and slope under `x`.
    fn ground(&self, x: f32, cfg: &Config) -> (f32, f32) {
        if !cfg.ground {
            return (f32::NEG_INFINITY, 0.0);
        }
        physics::ground(
            x,
            self.amplitude,
            cfg.slope,
            cfg.gaps,
            cfg.hurdles,
            self.quake_phase,
        )
    }

    /// The starting state: the creature's pose, its center of mass over
    /// x = 0 and its lowest point on the ground, at rest.
    pub fn start(&self, cfg: &Config) -> State {
        let b = self.pivot.len();
        let mut s = State {
            x0: self.start[0],
            v0: [0.0; 2],
            th0: self.rest.first().copied().unwrap_or(0.0),
            w0: 0.0,
            q: self.rest.clone(),
            qd: vec![0.0; b],
            th: vec![0.0; b],
            om: vec![0.0; b],
            pos: vec![[0.0; 2]; self.mass.len()],
            vel: vec![[0.0; 2]; self.mass.len()],
        };
        self.kinematics(&mut s);
        let com = self.center(&s);
        let low = (0..self.mass.len())
            .map(|i| {
                let (h, _) = self.ground(s.pos[i][0] - com[0], cfg);
                s.pos[i][1] - self.radius[i] - h.max(-1e6)
            })
            .fold(f32::INFINITY, f32::min);
        let low = if cfg.ground { low } else { 0.0 };
        s.x0[0] -= com[0];
        s.x0[1] -= low;
        self.kinematics(&mut s);
        s
    }

    /// Absolute bone angles and rates, then node positions and velocities.
    fn kinematics(&self, s: &mut State) {
        s.pos[0] = s.x0;
        s.vel[0] = s.v0;
        for j in 0..self.pivot.len() {
            let (th, om) = match self.parent[j] {
                Some(p) => (s.th[p] + s.q[j], s.om[p] + s.qd[j]),
                None => (s.th0, s.w0),
            };
            s.th[j] = th;
            s.om[j] = om;
            let (sin, cos) = th.sin_cos();
            let (p, c, l) = (self.pivot[j], self.child[j], self.length[j]);
            s.pos[c] = [s.pos[p][0] + l * cos, s.pos[p][1] + l * sin];
            s.vel[c] = [s.vel[p][0] - l * om * sin, s.vel[p][1] + l * om * cos];
        }
    }

    /// The center of mass.
    fn center(&self, s: &State) -> [f32; 2] {
        let mut c = [0.0; 2];
        for (p, m) in s.pos.iter().zip(&self.mass) {
            c[0] += p[0] * m;
            c[1] += p[1] * m;
        }
        [c[0] * self.inv_mass, c[1] * self.inv_mass]
    }
}
