//! The physics model of a creature: the constants the CUDA kernel is built
//! with and `Model`, the creature's constants and starting state, from which
//! `warp_kernel::pack` fills the kernel's records.
//!
//! A creature is a tree of point masses (its nodes) joined by rigid,
//! massless bones. The state is the head's position and velocity, the neck's
//! angle and angular velocity, and one relative angle and angular velocity
//! per other bone. Node positions follow from forward kinematics, so bones
//! keep their exact lengths and a pose is valid by construction.
//! `docs/physics.md` describes the dynamics the kernel runs.
use crate::{config::Config, evolution::Creature, physics};

/// How firmly a joint limit holds: its damper weighs this many times the
/// joint's inertia per step.
pub(crate) const LIMIT_HARDNESS: f32 = 20.0;
/// A joint's ligament (the bone's `ligament` gene) turns its stop into a
/// spring that stores the energy of the motion into the stop and gives it
/// back. The spring's rate runs from `LIGAMENT_STIFFEST` rad/s at a gene
/// just above 0 down to `LIGAMENT_SOFTEST` rad/s at 1, with a damping ratio
/// of `LIGAMENT_DAMPING`. A gene of 0 is the inelastic stop. The rates are
/// per second, so a finer substep sees the same spring.
pub const LIGAMENT_STIFFEST: f32 = 120.0;
pub const LIGAMENT_SOFTEST: f32 = 12.0;
pub const LIGAMENT_DAMPING: f32 = 0.25;
/// The square of the spring's rate for a ligament gene (0 for none).
pub fn ligament_rate_squared(gene: f32) -> f32 {
    if gene <= 0.0 {
        return 0.0;
    }
    let rate = LIGAMENT_STIFFEST + (LIGAMENT_SOFTEST - LIGAMENT_STIFFEST) * gene.min(1.0);
    rate * rate
}
/// Passive joint damping as a time constant (s): every joint resists its
/// relative rotation like tissue does, with a damper sized to the inertia the
/// joint moves.
pub fn joint_damping() -> f32 {
    0.1
}
/// Hill's force-velocity relation: a muscle's active pull falls linearly
/// with its shortening speed and vanishes at this many of its own lengths per
/// second. It bounds a muscle's power the way real muscle does, so a body
/// cannot catapult itself.
pub fn hill_speed() -> f32 {
    8.0
}
/// The largest acceleration (m/s^2) a muscle can give the mass it drives.
/// A muscle's cross-section, and so its force, grows with the mass it moves:
/// force cap = `DRIVEN_ACCELERATION` x the lighter of the two subtrees it
/// pulls together (a bone and everything it carries), never above the fixed
/// `Limits` cap, and its energy store scales the same way (a muscle's store
/// is its own mass). Before, a 100 N muscle drove a 0.05 kg limb at 2,000
/// m/s^2, turned a bone about a radian in one step and made momentum and
/// energy the integrator did not pay for (docs/physics.md).
pub const DRIVEN_ACCELERATION: f32 = 100.0;
/// Air drag on bones (N per m^3/s^2 of length x width): every bone feels
/// `AIR_DRAG x length x width x speed x velocity` against its midpoint's
/// velocity, with the width the mean diameter of its two nodes (a flat plate
/// in the flow: half the air's density, 1.2 kg/m^3, times a drag coefficient
/// of 1). Large fast bodies pay for moving air; it only takes energy away.
pub const AIR_DRAG: f32 = 0.6;
/// Water drag on a submerged bone (N per m^3/s^2 of length x width): the same
/// law as the air's, with a much thicker medium (about a third of half the
/// water's density over the air's: water resists motion across a bone far
/// more than a body's own bones resist air). A bone's sideways motion pays
/// the full price, its lengthwise motion `WATER_ALONG` of it, so a stroke that
/// pushes water sideways has a net reaction and a reciprocal stroke does not
/// cancel itself (a fish tail). The push is limited like the air's.
pub const WATER_DRAG: f32 = 100.0;
/// The share of `WATER_DRAG` a bone meets moving along its own length.
pub const WATER_ALONG: f32 = 0.25;
/// A fully submerged node feels this share of its weight as buoyancy.
pub const WATER_BUOYANCY: f32 = 0.7;
/// Contact tolerance for the behavior metrics (m), as the current engine.
pub(crate) const CONTACT_SLACK: f32 = 0.002;
pub(crate) const LIFT_CLEARANCE: f32 = 0.01;

/// One muscle's constants. A muscle joins two nodes and pulls only: its
/// force is the cap times the strength gene, the activation, the creature's
/// stamina and Hill's factor.
#[derive(Clone, Debug)]
pub(crate) struct MuscleModel {
    /// The two nodes (record order).
    pub(crate) node_a: usize,
    pub(crate) node_b: usize,
    /// Hill's relation as a factor on the shortening speed: 1 / (v_max
    /// times the muscle's length in the start pose, at least 5 cm).
    pub(crate) hill: f32,
    pub(crate) inv_period: f32,
    pub(crate) phase: f32,
    pub(crate) duty: f32,
    pub(crate) reset: f32,
    /// Force cap (N): the fixed `Limits` cap scaled by the mass the muscle
    /// drives (`DRIVEN_ACCELERATION`) and the strength gene.
    pub(crate) cap: f32,
    /// The muscle's share of the creature's stamina store (J): the fixed
    /// energy of a muscle scaled by the mass it drives.
    pub(crate) store: f32,
    /// Node whose touchdown restarts the rhythm, if any.
    pub(crate) sensor: Option<usize>,
}

/// A creature's constants for the physics. Nodes are renumbered so that
/// bone `j` ends at node `j + 1` (node 0 is the head), as the GPU kernel packs
/// them.
#[derive(Clone, Debug)]
pub struct Model {
    pub(crate) mass: Vec<f32>,
    pub(crate) radius: Vec<f32>,
    pub(crate) friction: Vec<f32>,
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
    /// Per bone: the square of its stop's spring rate (0 for an inelastic
    /// stop), see `ligament_rate_squared`.
    pub(crate) ligament: Vec<f32>,
    /// Earthquake bump phase and ground amplitude for this creature.
    pub(crate) quake_phase: f32,
    pub(crate) amplitude: f32,
    pub(crate) start: Vec<[f32; 2]>,
}

/// A creature's state: the head and the neck, then relative joint angles.
#[derive(Clone, Debug)]
pub struct State {
    pub(crate) x0: [f32; 2],
    pub(crate) v0: [f32; 2],
    pub(crate) th0: f32,
    pub(crate) w0: f32,
    pub(crate) q: Vec<f32>,
    pub(crate) qd: Vec<f32>,
    /// Derived by `kinematics`: absolute bone angles and rates, node
    /// positions and velocities.
    pub(crate) th: Vec<f32>,
    pub(crate) om: Vec<f32>,
    pub(crate) pos: Vec<[f32; 2]>,
    pub(crate) vel: Vec<[f32; 2]>,
}

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
        // The mass each node and everything below it weighs: what a muscle
        // pulls at. Node `j + 1` is bone `j`'s child; the head weighs it all.
        let mut subtree: Vec<f32> = (0..c.bones.len())
            .map(|j| nodes[order[j + 1]].mass + if j == 0 { nodes[order[0]].mass } else { 0.0 })
            .collect();
        for j in (1..c.bones.len()).rev() {
            if let Some(p) = parent[j] {
                subtree[p] += subtree[j];
            }
        }
        let carried = |node: usize| {
            let r = record[node];
            if r == 0 { nodes.iter().map(|n| n.mass).sum() } else { subtree[r - 1] }
        };
        let muscles: Vec<MuscleModel> = c
            .muscles
            .iter()
            .map(|m| {
                let ends = [m.node_a as usize, m.node_b as usize];
                let length = physics::muscle_span(&nodes, m);
                // A muscle's strength follows the mass it drives: the lighter
                // of the two subtrees it pulls together.
                let driven = carried(ends[0]).min(carried(ends[1]));
                let scale = (DRIVEN_ACCELERATION * driven / limits.muscle_force).min(1.0);
                MuscleModel {
                    node_a: record[ends[0]],
                    node_b: record[ends[1]],
                    hill: if hill_speed() > 0.0 {
                        1.0 / (hill_speed() * length.max(0.05))
                    } else {
                        0.0
                    },
                    inv_period: 1.0 / m.period,
                    phase: m.phase,
                    duty: m.duty,
                    reset: m.reset,
                    cap: limits.muscle_force * scale * m.strength,
                    store: limits.muscle_energy * scale,
                    sensor: (m.sensor < 2).then(|| record[ends[m.sensor as usize]]),
                }
            })
            .collect();
        let quake = crate::physics::quake_hash(c.id);
        let still = cfg.quake <= 0.0 || !cfg.ground;
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
            ligament: c.bones.iter().map(|b| ligament_rate_squared(b.ligament)).collect(),
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

/// Fastest a bone may turn (rad/s). At 60 Hz a bone at the cap turns 0.25
/// rad per step. The contact solve predicts each node's velocity at the end
/// of the step from the pose at its start, and that prediction misses by
/// about the square of the turn per step: at the current physics's 40 rad/s
/// (0.67 rad per step) evolution whipped a short bone into the ground, the
/// solve planted its tip, and the next pose had the tip sliding forward with
/// the friction that planted it still pushing (docs/physics.md).
pub const SPIN_CAP: f32 = 15.0;
/// How firmly the spin cap holds: its damper weighs this many times the
/// bone's rotational inertia about its pivot.
pub(crate) const SPIN_HARDNESS: f32 = 20.0;
/// Share of a node's depth inside the ground that the contact removes per
/// step.
pub(crate) const PUSH_OUT: f32 = 0.2;
