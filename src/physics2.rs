//! Physics v2 prototype: a planar articulated tree in reduced coordinates.
//!
//! A creature is a tree of point masses (its nodes) joined by rigid,
//! massless bones. The state is the head's position and velocity, the neck's
//! angle and angular velocity, and one relative angle and angular velocity
//! per other bone. Node positions follow from forward kinematics, so bones
//! keep their exact lengths and a pose is valid by construction: no bone
//! projection, rebuild, whole-body lift or settling.
//!
//! Each bone is a rigid body: its child node's mass sits at its far end (the
//! neck also carries the head). Dynamics come from the articulated-body
//! algorithm in planar spatial vectors, expressed in world axes about the
//! head's position at the start of the step so the numbers stay small.
//!
//! Stiff forces are implicit (backward Euler) and folded into the solve, so
//! they stay stable at 60 Hz on light limbs:
//! - Joint limits are angular springs with damping on the relative angle.
//! - Ground contact is a spring and damper along the ground normal at each
//!   touching node.
//! - Friction is viscous along the ground with a coefficient chosen so the
//!   force stays within mu times the normal force: a foot that barely slides
//!   sticks, and a foot that slides feels about mu N. Friction is a real
//!   force, so a sliding body cannot push itself forward, and only a planted
//!   foot can.
//!
//! Muscles keep today's model: a pull that follows the waveform's shortening
//! speed, a light damper, the force cap and the energy store.
use crate::{
    config::Config,
    creature_kernel::GpuResult,
    evolution::{Creature, NO_SENSOR, Population},
    physics,
};
use rayon::prelude::*;

/// How firmly a joint limit holds: its damper weighs this many times the
/// joint's inertia per step.
pub(crate) const LIMIT_HARDNESS: f32 = 20.0;
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
/// Fields per muscle in the v2 kernel's muscle buffer.
pub const MUSCLE_FIELDS: usize = 19;
/// Contact tolerance for the behavior metrics (m), as the current engine.
pub(crate) const CONTACT_SLACK: f32 = 0.002;
pub(crate) const LIFT_CLEARANCE: f32 = 0.01;

/// Planar spatial vector: an angular part and a linear part. It is a motion
/// (angular velocity, velocity of the body point at the origin) or a force
/// (moment about the origin, force).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct V3 {
    w: f32,
    x: f32,
    y: f32,
}
impl V3 {
    const fn new(w: f32, x: f32, y: f32) -> Self {
        Self { w, x, y }
    }
    fn add(self, o: V3) -> V3 {
        V3::new(self.w + o.w, self.x + o.x, self.y + o.y)
    }
    fn sub(self, o: V3) -> V3 {
        V3::new(self.w - o.w, self.x - o.x, self.y - o.y)
    }
    fn scale(self, s: f32) -> V3 {
        V3::new(self.w * s, self.x * s, self.y * s)
    }
    fn dot(self, o: V3) -> f32 {
        self.w * o.w + self.x * o.x + self.y * o.y
    }
}
/// Motion cross product `v x u`.
fn crm(v: V3, u: V3) -> V3 {
    V3::new(0.0, -v.w * u.y + u.w * v.y, v.w * u.x - u.w * v.x)
}
/// Force cross product `v x* f`.
fn crf(v: V3, f: V3) -> V3 {
    V3::new(v.x * f.y - v.y * f.x, -v.w * f.y, v.w * f.x)
}
/// The spatial force of a force `f` acting at point `r`.
fn force_at(r: [f32; 2], f: [f32; 2]) -> V3 {
    V3::new(r[0] * f[1] - r[1] * f[0], f[0], f[1])
}

/// Symmetric 3x3 spatial inertia: [ww, wx, wy, xx, xy, yy].
#[derive(Clone, Copy, Debug, Default)]
struct Sym([f32; 6]);
impl Sym {
    fn mul(&self, v: V3) -> V3 {
        let a = &self.0;
        V3::new(
            a[0] * v.w + a[1] * v.x + a[2] * v.y,
            a[1] * v.w + a[3] * v.x + a[4] * v.y,
            a[2] * v.w + a[4] * v.x + a[5] * v.y,
        )
    }
    /// Adds `k d d^T`.
    fn add_outer(&mut self, k: f32, d: V3) {
        let a = &mut self.0;
        a[0] += k * d.w * d.w;
        a[1] += k * d.w * d.x;
        a[2] += k * d.w * d.y;
        a[3] += k * d.x * d.x;
        a[4] += k * d.x * d.y;
        a[5] += k * d.y * d.y;
    }
    fn add(&mut self, o: &Sym) {
        for (a, b) in self.0.iter_mut().zip(o.0) {
            *a += b;
        }
    }
    /// A point mass `m` at `c`.
    fn point(m: f32, c: [f32; 2]) -> Sym {
        Sym([
            m * (c[0] * c[0] + c[1] * c[1]),
            -m * c[1],
            m * c[0],
            m,
            0.0,
            m,
        ])
    }
    /// The inverse (symmetric positive definite).
    fn inverse(&self) -> Sym {
        let [a, d, e, bb, f, c] = self.0;
        // Rows (a d e), (d bb f), (e f c).
        let c00 = bb * c - f * f;
        let c01 = e * f - d * c;
        let c02 = d * f - e * bb;
        let inv = 1.0 / (a * c00 + d * c01 + e * c02);
        let c11 = a * c - e * e;
        let c12 = d * e - a * f;
        let c22 = a * bb - d * d;
        Sym([
            c00 * inv,
            c01 * inv,
            c02 * inv,
            c11 * inv,
            c12 * inv,
            c22 * inv,
        ])
    }
}

/// One muscle's constants.
#[derive(Clone, Debug)]
pub(crate) struct MuscleModel {
    pub(crate) bone_a: usize,
    pub(crate) bone_b: usize,
    pub(crate) anchor_a: f32,
    pub(crate) anchor_b: f32,
    /// Hill's relation as a factor on the shortening speed: 1 / (v_max
    /// times the muscle's length, at least 5 cm).
    pub(crate) hill: f32,
    /// Longest length (m), where the elastic tendon starts to pull, and the
    /// tendon's stiffness (N/m; 0 without one).
    pub(crate) long: f32,
    pub(crate) tendon_k: f32,
    pub(crate) amplitude: f32,
    pub(crate) inv_period: f32,
    pub(crate) phase: f32,
    pub(crate) duty: f32,
    pub(crate) inv_duty: f32,
    pub(crate) inv_complement: f32,
    pub(crate) stiffness: f32,
    /// Force cap and energy store over the fixed `Limits` ones (at most 1):
    /// see `DRIVEN_ACCELERATION`.
    pub(crate) strength: f32,
    /// Node whose touchdown restarts the rhythm, if any.
    pub(crate) sensor: Option<usize>,
    pub(crate) reset: f32,
}

/// A creature's constants for the v2 physics. Nodes are renumbered so that
/// bone `j` ends at node `j + 1` (node 0 is the head), as the GPU kernel packs
/// them; `order` maps them back to the creature's own numbering.
#[derive(Clone, Debug)]
pub struct Model {
    pub(crate) order: Vec<usize>,
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
    /// Each muscle's force cap and energy store as multiples of the fixed
    /// `Limits` ones (1 unless muscle strength scales with the body).
    pub(crate) muscle_scale: f32,
    /// Air drag coefficient (`AIR_DRAG`); tests of conservation set it to 0.
    pub(crate) air_drag: f32,
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
    pub(crate) energy: Vec<f32>,
    pub(crate) offset: Vec<f32>,
    /// Each node's contact force (normal, friction) of the last step, which
    /// starts the next step's contact solve.
    pub(crate) warm: Vec<[f32; 2]>,
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
    /// Each muscle's force cap and energy store over the fixed `Limits` ones.
    pub fn muscle_strengths(&self) -> Vec<f32> {
        self.muscles.iter().map(|m| m.strength).collect()
    }
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
                    long: m.long,
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
        for (m, gene) in muscles.iter_mut().zip(&c.muscles) {
            let driven = subtree[m.bone_a].min(subtree[m.bone_b]);
            m.strength = (DRIVEN_ACCELERATION * driven / limits.muscle_force).min(1.0);
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
            air_drag: AIR_DRAG,
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
            order,
        }
    }

    /// Node positions in the creature's own numbering.
    fn frame(&self, s: &State) -> Vec<[f32; 2]> {
        let mut frame = vec![[0.0; 2]; self.order.len()];
        for (r, &node) in self.order.iter().enumerate() {
            frame[node] = s.pos[r];
        }
        frame
    }

    /// The ground's height and slope under `x`.
    /// How deep node `i` at `pos` sits in the mud, as a share of the
    /// deepest mud: 0 for a node clear of the surface.
    fn mud_sink(&self, pos: [f32; 2], i: usize, cfg: &Config, mud: f32) -> f32 {
        let (h, slope) = self.ground(pos[0], cfg);
        let secant = (1.0 + slope * slope).sqrt();
        let dry = (pos[1] - h) / secant - self.radius[i];
        (-dry).clamp(0.0, mud) * (1.0 / physics::MUD_FULL_DEPTH)
    }

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
            energy: vec![1.0; self.muscles.len()],
            offset: vec![0.0; self.muscles.len()],
            warm: vec![[0.0; 2]; self.mass.len()],
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

    /// Kinetic energy and gravity's potential (J), and the same with every
    /// term's magnitude, which sizes the first-law check's rounding
    /// allowance. The joint-limit springs are left out on purpose: a limit
    /// switches on from the step's predicted angle, so a joint driven hard
    /// within one step can pass its limit before the spring acts, and the
    /// spring would then hand back energy nobody paid for. Leaving it out
    /// makes the check take that energy back, so a joint limit is an
    /// inelastic stop.
    fn energy(&self, s: &State, gravity: f32) -> (f32, f32) {
        let (mut total, mut scale) = (0.0f32, 0.0f32);
        for i in 0..self.mass.len() {
            let v = s.vel[i];
            let kinetic = 0.5 * self.mass[i] * (v[0] * v[0] + v[1] * v[1]);
            let potential = self.mass[i] * gravity * s.pos[i][1];
            total += kinetic + potential;
            scale += kinetic + potential.abs();
        }
        // Elastic energy stored in the tendons.
        for (m, len) in self.muscles.iter().zip(self.muscle_lengths(s)) {
            if m.tendon_k > 0.0 {
                let stretch = (len - m.long).max(0.0);
                let stored = 0.5 * m.tendon_k * stretch * stretch;
                total += stored;
                scale += stored;
            }
        }
        (total, scale)
    }

    /// Mass times horizontal position, summed over the nodes (kg m).
    fn mass_x(&self, s: &State) -> f32 {
        s.pos.iter().zip(&self.mass).map(|(p, m)| p[0] * m).sum()
    }

    /// Every muscle's length between its two attachment points.
    fn muscle_lengths<'a>(&'a self, s: &'a State) -> impl Iterator<Item = f32> + 'a {
        self.muscles.iter().map(move |m| {
            let point = |bone: usize, t: f32| {
                let (p, c) = (s.pos[self.pivot[bone]], s.pos[self.child[bone]]);
                [p[0] + (c[0] - p[0]) * t, p[1] + (c[1] - p[1]) * t]
            };
            let (a, b) = (point(m.bone_a, m.anchor_a), point(m.bone_b, m.anchor_b));
            let d = [b[0] - a[0], b[1] - a[1]];
            (d[0] * d[0] + d[1] * d[1]).sqrt()
        })
    }

    fn center(&self, s: &State) -> [f32; 2] {
        let mut c = [0.0; 2];
        for (p, m) in s.pos.iter().zip(&self.mass) {
            c[0] += p[0] * m;
            c[1] += p[1] * m;
        }
        [c[0] * self.inv_mass, c[1] * self.inv_mass]
    }

    /// Body (bone) that carries node `n`: the bone whose child it is, or the
    /// neck for the head.
    fn body_of(&self, n: usize) -> usize {
        n.saturating_sub(1)
    }
}

/// A node that may touch the ground this step.
struct Contact {
    node: usize,
    normal: [f32; 2],
    /// Gap to the ground left after the step without the ground (negative:
    /// the node would sink this deep).
    reach: f32,
    tangent: [f32; 2],
    body: usize,
    /// Spatial directions of a unit force along the normal and the ground.
    dn: V3,
    dt: V3,
    /// The node's speed along the normal and the ground after the step
    /// without the ground.
    vn_free: f32,
    vt_free: f32,
    /// The node's speed along the ground at the start of the step. Friction
    /// may not do positive work: it may only push against the mean of this
    /// and the speed after the step.
    vt_start: f32,
    /// Slowest allowed speed along the normal (negative: approaching).
    target: f32,
    mu: f32,
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
/// Most nodes in one step's contact solve: the deepest four. Each contact
/// costs the GPU kernel two responses and two rows of Gauss-Seidel, so the
/// bound sets most of its speed (20k evolved bodies: 22M creature-steps/s at
/// four, 4M at eight with 20 sweeps); evolved walkers rarely have more than
/// four feet down, and a node left out sinks a little and joins the next
/// step's solve.
pub const MAX_CONTACTS: usize = 4;
/// Gauss-Seidel sweeps over the contacts per step, warm-started from each
/// node's contact force of the last step. Against 20 cold sweeps, 8 warm
/// ones moved 1,000 random bodies by at most 0.1 m in 2 s (p99 0.8 mm).
pub(crate) const PGS_ITERATIONS: usize = 8;
/// Sweeps of the planting pass, which starts from the first solve.
pub(crate) const PLANT_SWEEPS: usize = 4;
/// Rounds of the planting pass. One round leaves friction doing positive
/// work: the end pose it measures moves again when the solve changes the
/// forces (through the momentum balance and the turning bones). Two rounds
/// bring friction's positive work on an evolved hopper from 1,725 J to 5 J.
pub(crate) const PLANT_ROUNDS: usize = 2;
/// Share of a node's depth inside the ground that the contact removes per
/// step.
pub(crate) const PUSH_OUT: f32 = 0.2;

/// Everything one trial needs besides the model and state.
struct Scratch {
    inertia: Vec<Sym>,
    bias: Vec<V3>,
    force: Vec<V3>,
    vel: Vec<V3>,
    cvel: Vec<V3>,
    axis: Vec<V3>,
    u_vec: Vec<V3>,
    /// Per joint: the reciprocal of its articulated inertia about its axis.
    d: Vec<f32>,
    /// The inverse of the root's articulated inertia.
    root: Sym,
    u: Vec<f32>,
    acc: Vec<V3>,
    body_of: Vec<usize>,
    muscle_force: Vec<f32>,
    /// Each tendon's pull in the step (N), for the recorded force.
    tendon_pull: Vec<f32>,
    qdd: Vec<f32>,
    /// Horizontal impulse the ground and wind gave this step (N s), for the
    /// momentum ledger.
    impulse_x: f32,
    /// Muscle lengths at the start of the step, the kinetic energy the
    /// momentum balance added this step (J), and the number of ground
    /// contacts, for the energy ledger.
    muscle_length: Vec<f32>,
    balance_energy: f32,
    /// Energy the first-law check took away this step (J).
    first_law: f32,
    contacts: u32,
    /// Per contact this step: node, friction impulse (N s) and the ground's
    /// tangent, to check friction against the slip the step produced.
    friction: Vec<(usize, f32, [f32; 2])>,
    /// Muscle work so far in this trial (J): the sum of |force x relative
    /// speed| x dt that drains the muscle stores, for the cost of transport.
    work_total: f64,
    /// Ground work of the step (J): normal positive, normal negative,
    /// friction positive, friction negative.
    ground_work: [f64; 4],
}

thread_local! {
    /// Momentum ledger of the last `run` on this thread, summed over steps:
    /// horizontal impulse from the ground and wind, and the body's change of
    /// horizontal momentum. The difference is momentum the integrator made.
    pub static WORK: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    pub static LEDGER: std::cell::Cell<[f64; 2]> = const { std::cell::Cell::new([0.0; 2]) };
    /// Energy ledger of the last replay on this thread, summed over steps:
    /// 0 the work the muscles did on the body (their force times the change
    /// of their length); 1 and 2 the mechanical energy (kinetic, gravity's
    /// potential, and the joint-limit springs) gained beyond the muscle work
    /// and the momentum balance in steps where it grew, and lost where it
    /// shrank; 3 and 4 the kinetic energy the momentum balance added and
    /// removed; 5 steps without ground contact; 6 steps; 7 and 8 columns 1
    /// and 2 in steps without contact; 9 friction impulse that pushed a node
    /// the way it slid after the step, of 10 all friction impulse (N s); 11
    /// the energy the first-law check took away; 12 and 13 the work the
    /// ground's normal impulses did on the nodes (impulse times the mean of
    /// the node's speed along the normal before and after the step), positive
    /// and negative; 14 and 15 the same for friction.
    pub static ENERGY: std::cell::Cell<[f64; 16]> = const { std::cell::Cell::new([0.0; 16]) };
    /// Steps of the last replay on this thread by how many nodes took part
    /// in the contact solve (0 to `MAX_CONTACTS`).
    pub static CONTACT_COUNTS: std::cell::Cell<[u32; MAX_CONTACTS + 1]> =
        const { std::cell::Cell::new([0; MAX_CONTACTS + 1]) };
}

/// Per-trial totals, as the GPU kernel keeps them (`GpuResult`).
pub(crate) fn fresh_metrics() -> GpuResult {
    GpuResult {
        vertical_oscillation: 1e20,
        gait_frequency: -1e20,
        ..GpuResult::default()
    }
}

/// Runs one trial and returns its result, recording node positions per step
/// into `frames` when given (entry `t` after `t` steps; the first
/// `physics::settle()` entries repeat the starting pose so the replay
/// viewer's clock matches the current physics).
pub fn run(model: &Model, cfg: &Config, frames: Option<&mut Vec<Vec<[f32; 2]>>>) -> GpuResult {
    run_recorded(model, cfg, frames, None)
}

/// The muscle state and contact forces one recorded frame carries, in the
/// GPU kernels' convention: the energy is the state the frame starts from,
/// the forces are those of the step that led to it.
/// Bit `j` set for each bone `j` whose joint is forced more than
/// `physics::JOINT_BREAK` past its evolved range, which ends the trial. The
/// neck turns freely and never breaks.
pub(crate) fn broken_joints(model: &Model, s: &State) -> u64 {
    (1..model.pivot.len())
        .filter(|&j| {
            model.parent[j].is_some()
                && (s.q[j] < model.lo[j] - physics::JOINT_BREAK
                    || s.q[j] > model.hi[j] + physics::JOINT_BREAK)
        })
        .fold(0, |bits, j| bits | 1 << j)
}

fn push_extras(forces: &mut crate::replay_forces::Forces, model: &Model, s: &State, sc: &Scratch) {
    forces.energy.push(s.energy.clone());
    forces.broken.push(broken_joints(model, s));
    // The recorded force includes the tendon's pull.
    forces.muscle.push(
        sc.muscle_force
            .iter()
            .zip(&sc.tendon_pull)
            .map(|(f, t)| f + t)
            .collect(),
    );
    // Contact forces in the creature's own node numbering, as the frames.
    let mut ground = vec![0.0; s.warm.len()];
    let mut friction = vec![0.0; s.warm.len()];
    for (r, &node) in model.order.iter().enumerate() {
        ground[node] = s.warm[r][0];
        friction[node] = s.warm[r][1];
    }
    forces.ground.push(ground);
    forces.friction.push(friction);
}

/// `run`, also recording each frame's muscle energy, muscle force and ground
/// contact forces into `forces` (one entry per frame, as `frames`).
pub fn run_recorded(
    model: &Model,
    cfg: &Config,
    mut frames: Option<&mut Vec<Vec<[f32; 2]>>>,
    mut forces: Option<&mut crate::replay_forces::Forces>,
) -> GpuResult {
    let fidelity = cfg.fidelity();
    let rate = fidelity.rate as f32;
    let dt = 1.0 / rate;
    let steps = cfg.steps();
    let sample = fidelity.sample_interval();
    let limits = physics::limits();
    let air = fidelity.air_per_step(cfg.air_retention);
    let screen = cfg
        .screen
        .map(|s| (((s.seconds * rate).round() as u32).max(1) - 1, s.bar));
    let b = model.pivot.len();
    let n = model.mass.len();
    let mut s = model.start(cfg);
    let mut sc = Scratch {
        inertia: vec![Sym::default(); b],
        bias: vec![V3::default(); b],
        force: vec![V3::default(); b],
        vel: vec![V3::default(); b],
        cvel: vec![V3::default(); b],
        axis: vec![V3::default(); b],
        u_vec: vec![V3::default(); b],
        d: vec![0.0; b],
        root: Sym::default(),
        u: vec![0.0; b],
        acc: vec![V3::default(); b],
        body_of: (0..n).map(|i| model.body_of(i)).collect(),
        muscle_force: vec![0.0; model.muscles.len()],
        tendon_pull: vec![0.0; model.muscles.len()],
        qdd: vec![0.0; b],
        impulse_x: 0.0,
        muscle_length: vec![0.0; model.muscles.len()],
        balance_energy: 0.0,
        first_law: 0.0,
        contacts: 0,
        friction: Vec::new(),
        work_total: 0.0,
        ground_work: [0.0; 4],
    };
    if let Some(frames) = frames.as_deref_mut() {
        for _ in 0..=fidelity.settle() {
            frames.push(model.frame(&s));
            if let Some(forces) = forces.as_deref_mut() {
                push_extras(forces, model, &s, &sc);
            }
        }
    }
    let mut metrics = fresh_metrics();
    let mut ground_bits = 0u64;
    let mut head_shake = 0.0f32;
    let momentum_x = |s: &State| -> f64 {
        s.vel
            .iter()
            .zip(&model.mass)
            .map(|(v, m)| f64::from(v[0] * m))
            .sum()
    };
    let mut ledger = [0.0f64; 2];
    let diagnose = frames.is_some();
    let mut energy = [0.0f64; 16];
    let mut counts = [0u32; MAX_CONTACTS + 1];
    for step in 0..steps {
        let energy_before = if diagnose {
            f64::from(model.energy(&s, cfg.gravity).0)
        } else {
            0.0
        };
        let time = step as f32 * dt;
        let head_before = s.vel[0];
        let before = momentum_x(&s);
        let head_acc = simulate_step(model, cfg, &mut s, &mut sc, time, dt, air, &limits);
        let _ = head_acc;
        // The step leaves the node positions and velocities current; the
        // neck's angle is kept within one turn.
        s.th0 = wrap(s.th0);
        ledger[0] += f64::from(sc.impulse_x);
        ledger[1] += momentum_x(&s) - before;
        LEDGER.with(|l| l.set(ledger));
        WORK.with(|w| w.set(sc.work_total));
        if diagnose {
            let work: f64 = model
                .muscle_lengths(&s)
                .zip(&sc.muscle_length)
                .zip(&sc.muscle_force)
                .map(|((after, before), force)| f64::from(-force * (after - before)))
                .sum();
            let balance = f64::from(sc.balance_energy);
            let residual =
                f64::from(model.energy(&s, cfg.gravity).0) - energy_before - work - balance;
            energy[0] += work;
            energy[if residual > 0.0 { 1 } else { 2 }] += residual.abs();
            if sc.contacts == 0 {
                energy[if residual > 0.0 { 7 } else { 8 }] += residual.abs();
            }
            energy[if balance > 0.0 { 3 } else { 4 }] += balance.abs();
            // Friction that pushed a node the way it slid after the step.
            for &(node, impulse, tangent) in &sc.friction {
                let slip = s.vel[node][0] * tangent[0] + s.vel[node][1] * tangent[1];
                energy[10] += f64::from(impulse.abs());
                if impulse * slip > 0.0 && slip.abs() > 0.1 {
                    energy[9] += f64::from(impulse.abs());
                }
            }
            energy[11] += f64::from(sc.first_law);
            for (e, g) in energy[12..].iter_mut().zip(sc.ground_work) {
                *e += g;
            }
            energy[5] += f64::from(u8::from(sc.contacts == 0));
            counts[(sc.contacts as usize).min(MAX_CONTACTS)] += 1;
            CONTACT_COUNTS.with(|c| c.set(counts));
            energy[6] += 1.0;
            ENERGY.with(|e| e.set(energy));
        }
        // Metrics, falls and the screen, as the current engine does after
        // each step.
        let mut failed = false;
        let mut center_y = 0.0;
        let mut contacts = 0.0;
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut contact_bits, mut lift_bits) = (
            u64::from(metrics.contact_lo.to_bits()) | u64::from(metrics.contact_hi.to_bits()) << 32,
            u64::from(metrics.lift_lo.to_bits()) | u64::from(metrics.lift_hi.to_bits()) << 32,
        );
        let mut now_bits = 0u64;
        for i in 0..n {
            let [x, y] = s.pos[i];
            if !x.is_finite() || !y.is_finite() || x.abs() > 1e6 || y.abs() > 1e6 {
                failed = true;
            }
            let r = model.radius[i];
            center_y += y;
            low = low.min(y - r);
            high = high.max(y + r);
            if cfg.ground {
                let (h, slope) = model.ground(x, cfg);
                let floor = h + r * (1.0 + slope * slope).sqrt();
                if y <= floor + CONTACT_SLACK {
                    contacts += 1.0;
                    contact_bits |= 1 << i;
                    now_bits |= 1 << i;
                } else if y > floor + LIFT_CLEARANCE {
                    lift_bits |= contact_bits & (1 << i);
                }
            }
        }
        metrics.contact_lo = f32::from_bits(contact_bits as u32);
        metrics.contact_hi = f32::from_bits((contact_bits >> 32) as u32);
        metrics.lift_lo = f32::from_bits(lift_bits as u32);
        metrics.lift_hi = f32::from_bits((lift_bits >> 32) as u32);
        center_y *= 1.0 / n as f32;
        // Touchdowns restart the rhythm of the muscles that sense them.
        let down = now_bits & !ground_bits;
        ground_bits = now_bits;
        if down != 0 && step > 0 {
            let next = time + dt;
            for (k, m) in model.muscles.iter().enumerate() {
                if let Some(node) = m.sensor
                    && down & (1 << node) != 0
                {
                    let clock = next * m.inv_period + m.phase;
                    s.offset[k] = (m.reset - clock).rem_euclid(1.0);
                }
            }
        }
        // Head shaking, averaged over about HEAD_SHAKE_WINDOW.
        if time >= physics::HEAD_SHAKE_WINDOW {
            let dv = [s.vel[0][0] - head_before[0], s.vel[0][1] - head_before[1]];
            let accel = (dv[0] * dv[0] + dv[1] * dv[1]).sqrt() * rate;
            head_shake +=
                (accel - head_shake) * (1.0 / (physics::HEAD_SHAKE_WINDOW * rate)).min(1.0);
        }
        metrics.head_shake = head_shake;
        let broken = broken_joints(model, &s) != 0;
        let neck_base = model.child[0];
        let fell = s.pos[0][1] < s.pos[neck_base][1]
            || broken
            || head_shake > physics::HEAD_SHAKE_LIMIT
            || failed;
        let com_x = model.center(&s)[0];
        let mut ended = false;
        if fell {
            metrics.fall_time = time + dt;
            metrics.fitness = if failed {
                crate::evolution::FAILED
            } else {
                com_x
            };
            if screen.is_some_and(|(tick, _)| step <= tick) {
                metrics.screen_x = metrics.fitness;
            }
            ended = true;
        }
        metrics.ground_contact += contacts;
        metrics.height_sum += high - low;
        metrics.vertical_oscillation = metrics.vertical_oscillation.min(center_y);
        metrics.gait_frequency = metrics.gait_frequency.max(center_y);
        if step == 0 {
            metrics.previous_center_y = center_y;
            metrics.vertical_extremum = center_y;
            metrics.vertical_trend = 0.0;
            metrics.gait_turns = 0.0;
        } else if step % sample == 0 {
            gait_sample(&mut metrics, center_y);
        }
        if let Some((tick, bar)) = screen
            && step == tick
            && !ended
        {
            metrics.screen_x = com_x;
            if com_x < bar {
                metrics.screened = time + dt;
                metrics.fitness = com_x;
                ended = true;
            }
        }
        if let Some(frames) = frames.as_deref_mut() {
            frames.push(model.frame(&s));
            if let Some(forces) = forces.as_deref_mut() {
                push_extras(forces, model, &s, &sc);
            }
        }
        if ended {
            metrics.vertical_oscillation =
                (metrics.gait_frequency - metrics.vertical_oscillation).max(0.0);
            metrics.gait_frequency = metrics.gait_turns * 0.5 / (time + dt);
            if frames.is_none() {
                return metrics;
            }
            // A replay keeps moving after the fall, with limp muscles.
            let result = metrics;
            for later in step + 1..steps {
                let t = later as f32 * dt;
                simulate_step_limp(model, cfg, &mut s, &mut sc, t, dt, air, &limits);
                s.th0 = wrap(s.th0);
                if let Some(frames) = frames.as_deref_mut() {
                    frames.push(model.frame(&s));
                    if let Some(forces) = forces.as_deref_mut() {
                        push_extras(forces, model, &s, &sc);
                    }
                }
            }
            return result;
        }
    }
    metrics.fitness = model.center(&s)[0];
    metrics.vertical_oscillation = (metrics.gait_frequency - metrics.vertical_oscillation).max(0.0);
    metrics.gait_frequency = metrics.gait_turns * 0.5 / (steps as f32 * dt).max(dt);
    metrics
}

pub(crate) fn gait_sample(metrics: &mut GpuResult, center_y: f32) {
    let delta = center_y - metrics.previous_center_y;
    if metrics.vertical_trend == 0.0 {
        if delta.abs() > 0.0005 {
            metrics.vertical_trend = delta.signum();
            metrics.vertical_extremum = center_y;
        }
    } else if metrics.vertical_trend > 0.0 {
        if center_y > metrics.vertical_extremum {
            metrics.vertical_extremum = center_y;
        } else if metrics.vertical_extremum - center_y > 0.005 {
            metrics.gait_turns += 1.0;
            metrics.vertical_trend = -1.0;
            metrics.vertical_extremum = center_y;
        }
    } else if center_y < metrics.vertical_extremum {
        metrics.vertical_extremum = center_y;
    } else if center_y - metrics.vertical_extremum > 0.005 {
        metrics.gait_turns += 1.0;
        metrics.vertical_trend = 1.0;
        metrics.vertical_extremum = center_y;
    }
    metrics.previous_center_y = center_y;
}

#[allow(clippy::too_many_arguments)]
fn simulate_step_limp(
    model: &Model,
    cfg: &Config,
    s: &mut State,
    sc: &mut Scratch,
    time: f32,
    dt: f32,
    air: f32,
    limits: &physics::Limits,
) {
    let saved = s.energy.clone();
    s.energy.iter_mut().for_each(|e| *e = 0.0);
    simulate_step_inner(model, cfg, s, sc, time, dt, air, limits, true);
    s.energy = saved;
}

#[allow(clippy::too_many_arguments)]
fn simulate_step(
    model: &Model,
    cfg: &Config,
    s: &mut State,
    sc: &mut Scratch,
    time: f32,
    dt: f32,
    air: f32,
    limits: &physics::Limits,
) -> f32 {
    simulate_step_inner(model, cfg, s, sc, time, dt, air, limits, false)
}

/// Advances one step from the kinematics already in `s`. Returns the head's
/// acceleration magnitude.
#[allow(clippy::too_many_arguments)]
fn simulate_step_inner(
    model: &Model,
    cfg: &Config,
    s: &mut State,
    sc: &mut Scratch,
    time: f32,
    dt: f32,
    air: f32,
    limits: &physics::Limits,
    limp: bool,
) -> f32 {
    let b = model.pivot.len();
    let n = model.mass.len();
    let rate = 1.0 / dt;
    let origin = s.x0;
    let rel = |p: [f32; 2]| [p[0] - origin[0], p[1] - origin[1]];
    let momentum = |s: &State| -> [f32; 2] {
        let mut p = [0.0f32; 2];
        for (v, m) in s.vel.iter().zip(&model.mass) {
            p[0] += v[0] * m;
            p[1] += v[1] * m;
        }
        p
    };
    let before = momentum(s);
    let vel_start = s.vel.clone();
    sc.ground_work = [0.0; 4];
    sc.friction.clear();
    // For the first-law check at the end of the step.
    let (energy_start, energy_scale) = model.energy(s, cfg.gravity);
    let mass_x_start = model.mass_x(s);
    let mut muscle_start = 0.0f32;
    // Body inertias (true, for the velocity products) and external forces.
    for j in 0..b {
        let c = model.child[j];
        sc.inertia[j] = Sym::point(model.mass[c], rel(s.pos[c]));
        sc.force[j] = V3::default();
    }
    sc.inertia[0].add(&Sym::point(model.mass[0], [0.0, 0.0]));
    // Spatial velocities, joint axes and velocity-product accelerations.
    sc.vel[0] = V3::new(s.w0, s.v0[0], s.v0[1]);
    for j in 1..b {
        let p = model.parent[j].unwrap_or(0);
        let r = rel(s.pos[model.pivot[j]]);
        sc.axis[j] = V3::new(1.0, r[1], -r[0]);
        sc.vel[j] = sc.vel[p].add(sc.axis[j].scale(s.qd[j]));
        sc.cvel[j] = crm(sc.vel[j], sc.axis[j]).scale(s.qd[j]);
    }
    for j in 0..b {
        let iv = sc.inertia[j].mul(sc.vel[j]);
        sc.bias[j] = crf(sc.vel[j], iv);
    }
    // Gravity and wind on every node.
    let gravity = cfg.gravity;
    // Mud (v1's rules): the floor lies `mud` lower, a node sunk in it (in
    // meters over `MUD_FULL_DEPTH`) has its friction budget scaled up and
    // loses a `MUD_DRAG * sink` share of its horizontal speed per second.
    let mud = if cfg.ground { cfg.mud } else { 0.0 };
    let mut mud_impulse = 0.0f32;
    for i in 0..n {
        let m = model.mass[i];
        let mut f = [cfg.wind * m, -gravity * m];
        if mud > 0.0 {
            let sink = model.mud_sink(s.pos[i], i, cfg, mud);
            let drag = -m * physics::MUD_DRAG * sink * s.vel[i][0];
            f[0] += drag;
            mud_impulse += drag * dt;
        }
        let j = sc.body_of[i];
        sc.force[j] = sc.force[j].add(force_at(rel(s.pos[i]), f));
    }
    // Air drag on every bone, at its midpoint. The push is limited so a step
    // of drag never more than halves the speed it acts on.
    let mut air_impulse = [0.0f32; 2];
    for j in 0..b {
        let (p, c) = (model.pivot[j], model.child[j]);
        let mid = [
            0.5 * (s.pos[p][0] + s.pos[c][0]),
            0.5 * (s.pos[p][1] + s.pos[c][1]),
        ];
        let v = [
            0.5 * (s.vel[p][0] + s.vel[c][0]),
            0.5 * (s.vel[p][1] + s.vel[c][1]),
        ];
        let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let width = model.radius[p] + model.radius[c];
        let strength = (model.air_drag * model.length[j] * width * speed)
            .min(0.5 * model.mass[c] * rate)
            .max(0.0);
        let f = [-v[0] * strength, -v[1] * strength];
        sc.force[j] = sc.force[j].add(force_at(rel(mid), f));
        air_impulse[0] += f[0] * dt;
        air_impulse[1] += f[1] * dt;
    }
    // Water below the waterline: buoyancy on every node by the share of its
    // diameter that is submerged, and anisotropic drag on every bone by the
    // share of it that is submerged (the mean of its two nodes').
    let water = cfg.water;
    let mut buoyancy: Vec<f32> = Vec::new();
    let mut water_y0: Vec<f32> = Vec::new();
    if water > 0.0 {
        let submerged: Vec<f32> = (0..n)
            .map(|i| {
                ((water - (s.pos[i][1] - model.radius[i])) / (2.0 * model.radius[i]))
                    .clamp(0.0, 1.0)
            })
            .collect();
        buoyancy = (0..n)
            .map(|i| WATER_BUOYANCY * model.mass[i] * cfg.gravity * submerged[i])
            .collect();
        water_y0 = (0..n).map(|i| s.pos[i][1]).collect();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let j = sc.body_of[i];
            sc.force[j] = sc.force[j].add(force_at(rel(s.pos[i]), [0.0, buoyancy[i]]));
            air_impulse[1] += buoyancy[i] * dt;
        }
        for j in 0..b {
            let (p, c) = (model.pivot[j], model.child[j]);
            let wet = 0.5 * (submerged[p] + submerged[c]);
            let mid = [
                0.5 * (s.pos[p][0] + s.pos[c][0]),
                0.5 * (s.pos[p][1] + s.pos[c][1]),
            ];
            let v = [
                0.5 * (s.vel[p][0] + s.vel[c][0]),
                0.5 * (s.vel[p][1] + s.vel[c][1]),
            ];
            let inverse = 1.0 / model.length[j];
            let axis = [
                (s.pos[c][0] - s.pos[p][0]) * inverse,
                (s.pos[c][1] - s.pos[p][1]) * inverse,
            ];
            let along = v[0] * axis[0] + v[1] * axis[1];
            let lengthwise = [axis[0] * along, axis[1] * along];
            let sideways = [v[0] - lengthwise[0], v[1] - lengthwise[1]];
            let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
            let width = model.radius[p] + model.radius[c];
            let strength = (WATER_DRAG * wet * model.length[j] * width * speed)
                .min(0.5 * model.mass[c] * rate)
                .max(0.0);
            let weak = strength * WATER_ALONG;
            let f = [
                -(sideways[0] * strength + lengthwise[0] * weak),
                -(sideways[1] * strength + lengthwise[1] * weak),
            ];
            sc.force[j] = sc.force[j].add(force_at(rel(mid), f));
            air_impulse[0] += f[0] * dt;
            air_impulse[1] += f[1] * dt;
        }
    }
    // Muscles: today's drive and damper, applied at the attachment points.
    let settle_free = time; // No settling: the rhythm starts at once.
    // The waveform's target length is `long - amplitude * (1 - w)`; its
    // change over the step is the amplitude times the change of `w`.
    let target_speed = |m: &MuscleModel, offset: f32| {
        let wave = |t: f32| {
            let phase = (t * m.inv_period + m.phase + offset).rem_euclid(1.0);
            if phase < m.duty {
                0.5 + 0.5 * (std::f32::consts::PI * (phase * m.inv_duty)).cos()
            } else {
                0.5 - 0.5 * (std::f32::consts::PI * ((phase - m.duty) * m.inv_complement)).cos()
            }
        };
        if settle_free <= 0.0 {
            0.0
        } else {
            m.amplitude * (wave(settle_free) - wave((settle_free - dt).max(0.0))) * rate
        }
    };
    for (k, m) in model.muscles.iter().enumerate() {
        let point = |bone: usize, t: f32| {
            let (p, c) = (model.pivot[bone], model.child[bone]);
            (
                [
                    s.pos[p][0] + (s.pos[c][0] - s.pos[p][0]) * t,
                    s.pos[p][1] + (s.pos[c][1] - s.pos[p][1]) * t,
                ],
                [
                    s.vel[p][0] + (s.vel[c][0] - s.vel[p][0]) * t,
                    s.vel[p][1] + (s.vel[c][1] - s.vel[p][1]) * t,
                ],
            )
        };
        let (pa, va) = point(m.bone_a, m.anchor_a);
        let (pb, vb) = point(m.bone_b, m.anchor_b);
        let d = [pb[0] - pa[0], pb[1] - pa[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-6);
        let inverse = 1.0 / len;
        let dir = [d[0] * inverse, d[1] * inverse];
        let relative = (vb[0] - va[0]) * dir[0] + (vb[1] - va[1]) * dir[1];
        let energy = s.energy[k];
        let mut drive = (-target_speed(m, s.offset[k]) * m.stiffness * 0.25).max(0.0) * energy;
        if m.hill > 0.0 {
            // Shortening is a negative `relative`.
            drive *= (1.0 + relative * m.hill).clamp(0.0, 1.0);
        }
        let cap = limits.muscle_force * m.strength * model.muscle_scale;
        let inv_capacity =
            1.0 / (limits.muscle_energy * cfg.muscle_energy * m.strength * model.muscle_scale);
        let mut magnitude = (drive + relative * 0.15).clamp(-cap, cap);
        if limp {
            magnitude = 0.0;
        }
        // Only active contraction costs energy: the drive against a shortening
        // muscle. The passive damper (0.15 per m/s) and a muscle that is
        // stretched or held cost nothing.
        let work = drive.min(cap) * (-relative).max(0.0) * dt;
        sc.work_total += f64::from(work);
        s.energy[k] = (energy - work * inv_capacity
            + limits.muscle_recovery * cfg.muscle_recovery * dt * (1.0 - energy))
            .clamp(0.0, 1.0);
        sc.muscle_force[k] = magnitude;
        sc.muscle_length[k] = len;
        muscle_start += magnitude * len;
        // A positive magnitude pulls the two points together.
        // The tendon pulls back passively once the muscle is stretched past
        // its longest length (its energy is in `Model::energy`).
        let tendon_pull = m.tendon_k * (len - m.long).max(0.0);
        sc.tendon_pull[k] = tendon_pull;
        let pull = magnitude + tendon_pull;
        let f = [dir[0] * pull, dir[1] * pull];
        sc.force[m.bone_a] = sc.force[m.bone_a].add(force_at(rel(pa), f));
        sc.force[m.bone_b] = sc.force[m.bone_b].sub(force_at(rel(pb), f));
    }
    // Spin cap: a bone turning faster than `SPIN_CAP` meets a rotational
    // drag, a pure torque against its turning that grows steeply past the
    // cap. A pure torque changes no linear momentum, so the cap cannot push
    // the body along, and it keeps every step's rotation small enough for the
    // integrator. The drag is implicit and damps toward standing still, so it
    // only ever takes energy away. (A damper toward the cap itself, the first
    // version, held a bone at the cap when contacts slowed it: it gave energy.)
    let spin = SPIN_CAP;
    for j in 0..b {
        let w = sc.vel[j].w;
        if w.abs() > spin {
            let l = model.length[j];
            let drag =
                SPIN_HARDNESS * model.mass[model.child[j]] * l * l * (w.abs() * (1.0 / spin) - 1.0);
            let turn = V3::new(1.0, 0.0, 0.0);
            sc.inertia[j].add_outer(drag, turn);
            sc.force[j] = sc.force[j].add(turn.scale(-drag * rate * w));
        }
    }
    // The step without the ground.
    for j in 0..b {
        sc.bias[j] = sc.bias[j].sub(sc.force[j]);
    }
    solve(model, s, sc, dt);
    // Ground contacts, solved together at velocity level by projected
    // Gauss-Seidel on the contact impulses: a touching node may approach the
    // ground only as fast as its gap allows (a node inside is pushed out by
    // a share of its depth), normal forces only push, and friction stays
    // within mu times the normal force and opposes sliding. The contacts are
    // inelastic and store no energy, and a sliding body gets no push from
    // friction, so only a planted foot can propel the body.
    let mut contacts: Vec<Contact> = Vec::new();
    if cfg.ground {
        for i in 0..n {
            let [x, y] = s.pos[i];
            let (h, slope) = model.ground(x, cfg);
            let secant = (1.0 + slope * slope).sqrt();
            let normal = [-slope / secant, 1.0 / secant];
            let tangent = [normal[1], -normal[0]];
            let dry = (y - h) / secant - model.radius[i];
            let gap = dry + mud;
            let sink = (-dry).clamp(0.0, mud) * (1.0 / physics::MUD_FULL_DEPTH);
            let j = sc.body_of[i];
            let r = rel(s.pos[i]);
            let v = s.vel[i];
            let w = sc.vel[j].w;
            let beta = |dir: [f32; 2]| w * (-v[1] * dir[0] + v[0] * dir[1]);
            let dn = force_at(r, normal);
            let dtan = force_at(r, tangent);
            let a = sc.acc[j];
            let vn_free = v[0] * normal[0] + v[1] * normal[1] + dt * (dn.dot(a) + beta(normal));
            if gap + dt * vn_free > 0.0 {
                continue;
            }
            contacts.push(Contact {
                node: i,
                normal,
                reach: gap + dt * vn_free,
                tangent,
                body: j,
                dn,
                dt: dtan,
                vn_free,
                vt_free: v[0] * tangent[0] + v[1] * tangent[1] + dt * (dtan.dot(a) + beta(tangent)),
                vt_start: v[0] * tangent[0] + v[1] * tangent[1],
                target: if gap >= 0.0 {
                    -gap * rate
                } else {
                    -gap * PUSH_OUT * rate
                },
                mu: {
                    let mu = model.friction[i]
                        * cfg.ground_friction
                        * (1.0 + physics::MUD_GRIP * sink)
                        * (1.0 + physics::MUD_NORMAL * sink);
                    // Ice patches take a share of the friction.
                    let mu = if cfg.patches > 0.0 {
                        mu * (1.0 - cfg.patches * physics::ice(x))
                    } else {
                        mu
                    };
                    // Static friction: a foot that barely slides holds harder.
                    mu * physics::static_factor(v[0] * tangent[0] + v[1] * tangent[1])
                },
            });
        }
    }
    // At most `MAX_CONTACTS` nodes, the deepest, take part in one step's
    // solve, as the GPU kernel bounds it; the others follow the next step.
    if contacts.len() > MAX_CONTACTS {
        contacts.sort_by(|a, b| a.reach.total_cmp(&b.reach).then(a.node.cmp(&b.node)));
        contacts.truncate(MAX_CONTACTS);
        contacts.sort_by_key(|c| c.node);
    }
    /// One contact's final impulses, for the ground work ledger: node, normal
    /// and friction impulse, normal and tangent.
    type Logged = (usize, f32, f32, [f32; 2], [f32; 2]);
    let mut ground_log: Vec<Logged> = Vec::new();
    let mut impulse = cfg.wind * model.total_mass * dt + mud_impulse + air_impulse[0];
    let mut impulse_y = -cfg.gravity * model.total_mass * dt + air_impulse[1];
    if !contacts.is_empty() {
        // Each contact direction's response: the change of every body's
        // acceleration under a unit force there.
        let m = 2 * contacts.len();
        let rows: Vec<(usize, V3)> = contacts
            .iter()
            .flat_map(|c| [(c.body, c.dn), (c.body, c.dt)])
            .collect();
        // Contact-space matrix: the velocity change along each row per
        // newton along each column, from the column's response. It is
        // symmetric; the lower triangle is mirrored.
        let mut k = vec![0.0f32; m * m];
        let mut da = vec![V3::default(); b];
        let mut dq = vec![0.0f32; b];
        for col in 0..m {
            response(model, sc, &rows[col..=col], &mut da, &mut dq);
            for row in col..m {
                let (body, dir) = rows[row];
                let v = dt * dir.dot(da[body]);
                k[row * m + col] = v;
                k[col * m + row] = v;
            }
        }
        let mut lambda = vec![0.0f32; m];
        for (i, c) in contacts.iter().enumerate() {
            let [n, t] = s.warm[c.node];
            lambda[2 * i] = n;
            lambda[2 * i + 1] = t.clamp(-c.mu * n, c.mu * n);
        }
        let predicted = pgs(&contacts, &k, &mut lambda, PGS_ITERATIONS);
        // The contact forces act on the bodies through one response.
        apply_contacts(model, sc, &contacts, &lambda, &mut da, &mut dq);
        // Plant against the end pose. The solve works on each contact's
        // velocity linearized in the step's start pose, but the step turns
        // the bones, so a foot planted in the start pose can leave the step
        // sliding (a fast-turning bone swings its planted tip forward) while
        // the friction that planted it keeps pushing. So the step is taken,
        // each contact's velocity measured in the end pose (after the
        // momentum balance), and the solve repeated with the difference.
        let mut predicted = predicted;
        for _ in 0..PLANT_ROUNDS {
            let saved = (s.x0, s.v0, s.w0, s.th0, s.q.clone(), s.qd.clone());
            integrate(model, s, sc, dt, 1.0);
            let ground: [f32; 2] = contacts.iter().enumerate().fold([0.0; 2], |g, (i, c)| {
                [
                    g[0] + (lambda[2 * i] * c.dn.x + lambda[2 * i + 1] * c.dt.x) * dt,
                    g[1] + (lambda[2 * i] * c.dn.y + lambda[2 * i + 1] * c.dt.y) * dt,
                ]
            });
            let after = momentum(s);
            let shift = [
                (before[0] + impulse + ground[0] - after[0]) * model.inv_mass,
                (before[1] + impulse_y + ground[1] - after[1]) * model.inv_mass,
            ];
            for (i, c) in contacts.iter_mut().enumerate() {
                let (rn, rt) = (2 * i, 2 * i + 1);
                let v = [s.vel[c.node][0] + shift[0], s.vel[c.node][1] + shift[1]];
                c.vn_free += v[0] * c.normal[0] + v[1] * c.normal[1] - predicted[rn];
                c.vt_free += v[0] * c.tangent[0] + v[1] * c.tangent[1] - predicted[rt];
            }
            (s.x0, s.v0, s.w0, s.th0) = (saved.0, saved.1, saved.2, saved.3);
            s.q = saved.4;
            s.qd = saved.5;
            let old = lambda.clone();
            let mut ends = pgs(&contacts, &k, &mut lambda, PLANT_SWEEPS);
            clean_friction(&contacts, &k, &mut lambda, &mut ends);
            let change: Vec<f32> = lambda.iter().zip(&old).map(|(a, b)| a - b).collect();
            apply_contacts(model, sc, &contacts, &change, &mut da, &mut dq);
            predicted = ends;
        }
        s.warm.iter_mut().for_each(|w| *w = [0.0; 2]);
        for (i, c) in contacts.iter().enumerate() {
            s.warm[c.node] = [lambda[2 * i], lambda[2 * i + 1]];
        }
        ground_log = contacts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                (
                    c.node,
                    lambda[2 * i] * dt,
                    lambda[2 * i + 1] * dt,
                    c.normal,
                    c.tangent,
                )
            })
            .collect();
        for (i, c) in contacts.iter().enumerate() {
            sc.friction
                .push((c.node, lambda[2 * i + 1] * dt, c.tangent));
            impulse += (lambda[2 * i] * c.dn.x + lambda[2 * i + 1] * c.dt.x) * dt;
            impulse_y += (lambda[2 * i] * c.dn.y + lambda[2 * i + 1] * c.dt.y) * dt;
        }
    }
    if contacts.is_empty() {
        s.warm.iter_mut().for_each(|w| *w = [0.0; 2]);
    }
    sc.impulse_x = impulse;
    sc.contacts = contacts.len() as u32;
    let head = integrate(model, s, sc, dt, air);
    for &(node, jn, jt, normal, tangent) in &ground_log {
        let mean = [
            0.5 * (vel_start[node][0] + s.vel[node][0]),
            0.5 * (vel_start[node][1] + s.vel[node][1]),
        ];
        let wn = f64::from(jn * (mean[0] * normal[0] + mean[1] * normal[1]));
        let wt = f64::from(jt * (mean[0] * tangent[0] + mean[1] * tangent[1]));
        sc.ground_work[if wn > 0.0 { 0 } else { 1 }] += wn;
        sc.ground_work[if wt > 0.0 { 2 } else { 3 }] += wt;
    }
    // Momentum balance: the body's momentum changes only by the external
    // impulses (gravity, wind, the ground) and the air. First-order
    // integration in joint coordinates misses that by a little each step,
    // and a flailing body could collect the misses; the difference is spread
    // over the body as one uniform velocity, which moves no node against
    // another.
    let after = momentum(s);
    let expected = [(before[0] + impulse) * air, (before[1] + impulse_y) * air];
    let shift = [
        (expected[0] - after[0]) * model.inv_mass,
        (expected[1] - after[1]) * model.inv_mass,
    ];
    sc.balance_energy = after[0] * shift[0]
        + after[1] * shift[1]
        + 0.5 * model.total_mass * (shift[0] * shift[0] + shift[1] * shift[1]);
    s.v0 = [s.v0[0] + shift[0], s.v0[1] + shift[1]];
    for v in &mut s.vel {
        v[0] += shift[0];
        v[1] += shift[1];
    }
    // First law in flight: a step without ground contact may not gain more
    // energy than the muscles and the wind put in. The muscles' work is
    // their force times the change of their length. Gravity's work is in the
    // potential energy; joint damping and limits, the spin drag and the air
    // only take energy away. First-order integration of a tree turning fast
    // can still come out ahead by up to about 1% a step; the excess is taken
    // from the body's motion relative to its center of mass (every velocity
    // scaled toward the center's), which keeps its momentum. On the ground
    // the same scaling would move planted feet, so there the contacts, which
    // only take energy, are left to it.
    let muscle_end: f32 = model
        .muscle_lengths(s)
        .zip(&sc.muscle_force)
        .map(|(length, force)| force * length)
        .sum();
    let mut work = (muscle_start - muscle_end) + cfg.wind * (model.mass_x(s) - mass_x_start);
    if water > 0.0 {
        // Buoyancy lifts the body: its work is the force times the rise.
        let lift: f32 = (0..n)
            .map(|i| buoyancy[i] * (s.pos[i][1] - water_y0[i]))
            .sum();
        work += lift;
    }
    let (energy_end, _) = model.energy(s, cfg.gravity);
    let excess = energy_end - energy_start - work - (1e-4 + 1e-5 * energy_scale);
    sc.first_law = 0.0;
    if excess > 0.0 && sc.contacts == 0 {
        let center = [expected[0] * model.inv_mass, expected[1] * model.inv_mass];
        let internal: f32 = s
            .vel
            .iter()
            .zip(&model.mass)
            .map(|(v, m)| {
                let (x, y) = (v[0] - center[0], v[1] - center[1]);
                0.5 * m * (x * x + y * y)
            })
            .sum();
        let keep = if internal > 0.0 {
            (1.0 - excess / internal).max(0.0).sqrt()
        } else {
            0.0
        };
        sc.first_law = internal * (1.0 - keep * keep);
        s.w0 *= keep;
        for qd in s.qd.iter_mut().skip(1) {
            *qd *= keep;
        }
        s.v0 = [
            center[0] + keep * (s.v0[0] - center[0]),
            center[1] + keep * (s.v0[1] - center[1]),
        ];
        for v in &mut s.vel {
            v[0] = center[0] + keep * (v[0] - center[0]);
            v[1] = center[1] + keep * (v[1] - center[1]);
        }
    }
    (head[0] * head[0] + head[1] * head[1]).sqrt()
}

/// Semi-implicit Euler on the joint coordinates with the accelerations in
/// `sc`, then air drag on every velocity and the new pose's kinematics.
/// Returns the head's acceleration.
fn integrate(model: &Model, s: &mut State, sc: &Scratch, dt: f32, air: f32) -> [f32; 2] {
    let a0 = sc.acc[0];
    let head = [a0.x - s.w0 * s.v0[1], a0.y + s.w0 * s.v0[0]];
    s.v0 = [
        (s.v0[0] + head[0] * dt) * air,
        (s.v0[1] + head[1] * dt) * air,
    ];
    s.w0 = (s.w0 + a0.w * dt) * air;
    s.x0 = [s.x0[0] + s.v0[0] * dt, s.x0[1] + s.v0[1] * dt];
    s.th0 += s.w0 * dt;
    for ((q, qd), a) in s.q.iter_mut().zip(&mut s.qd).zip(&sc.qdd).skip(1) {
        *qd = (*qd + a * dt) * air;
        *q += *qd * dt;
    }
    model.kinematics(s);
    head
}

/// Projected Gauss-Seidel on the contact impulses: a touching node may
/// approach the ground only as fast as its target allows, normal forces
/// only push, and friction stays within mu times the normal force and
/// opposes the slip. Every row's velocity is kept current as the forces
/// change (a change moves each row by its column of the matrix), so an
/// update reads one number instead of a sum. Returns the rows' velocities.
fn pgs(contacts: &[Contact], k: &[f32], lambda: &mut [f32], sweeps: usize) -> Vec<f32> {
    let m = lambda.len();
    let mut v: Vec<f32> = contacts
        .iter()
        .flat_map(|c| [c.vn_free, c.vt_free])
        .collect();
    for (row, v) in v.iter_mut().enumerate() {
        for (j, l) in lambda.iter().enumerate() {
            *v += k[row * m + j] * l;
        }
    }
    for _ in 0..sweeps {
        for (i, c) in contacts.iter().enumerate() {
            let (rn, rt) = (2 * i, 2 * i + 1);
            let normal = (lambda[rn] + (c.target - v[rn]) * (1.0 / k[rn * m + rn])).max(0.0);
            let change = normal - lambda[rn];
            lambda[rn] = normal;
            for (j, v) in v.iter_mut().enumerate() {
                *v += k[j * m + rn] * change;
            }
            let bound = c.mu * lambda[rn];
            // Friction does work `lambda * (vt_start + vt_end) / 2`, which may
            // not be positive. The speed after the step without this row's own
            // force is `v - k * lambda`, so the force may only oppose
            // `a = vt_start + that`, and only up to `|a| / k` (past that it
            // would reverse the node and push it the way it now moves).
            let stiff = k[rt * m + rt];
            let a = c.vt_start + v[rt] - stiff * lambda[rt];
            let reach = a.abs() / stiff;
            let (low, high) = if a > 0.0 {
                (-bound.min(reach), 0.0)
            } else {
                (0.0, bound.min(reach))
            };
            let friction = (lambda[rt] - v[rt] * (1.0 / stiff)).clamp(low, high);
            let change = friction - lambda[rt];
            lambda[rt] = friction;
            for (j, v) in v.iter_mut().enumerate() {
                *v += k[j * m + rt] * change;
            }
        }
    }
    v
}

/// Removes friction that would do positive work: after the solves, each
/// contact's friction force may only oppose the mean of its node's speed
/// before and after the step, and only up to the size that stops the node.
/// The rows share `v`, so a change is passed on to the other rows, and two
/// sweeps let the contacts settle.
fn clean_friction(contacts: &[Contact], k: &[f32], lambda: &mut [f32], v: &mut [f32]) {
    let m = lambda.len();
    for _ in 0..2 {
        for (i, c) in contacts.iter().enumerate() {
            let rt = 2 * i + 1;
            let stiff = k[rt * m + rt];
            let a = c.vt_start + v[rt] - stiff * lambda[rt];
            let reach = a.abs() / stiff;
            let bound = (c.mu * lambda[2 * i]).min(reach);
            let (low, high) = if a > 0.0 { (-bound, 0.0) } else { (0.0, bound) };
            let friction = lambda[rt].clamp(low, high);
            let change = friction - lambda[rt];
            lambda[rt] = friction;
            for (j, v) in v.iter_mut().enumerate() {
                *v += k[j * m + rt] * change;
            }
        }
    }
}

/// Adds the accelerations that contact forces `lambda` cause, through one
/// response.
fn apply_contacts(
    model: &Model,
    sc: &mut Scratch,
    contacts: &[Contact],
    lambda: &[f32],
    da: &mut [V3],
    dq: &mut [f32],
) {
    let forces: Vec<(usize, V3)> = contacts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let force = c.dn.scale(lambda[2 * i]).add(c.dt.scale(lambda[2 * i + 1]));
            (c.body, force)
        })
        .collect();
    response(model, sc, &forces, da, dq);
    for j in 0..model.pivot.len() {
        sc.acc[j] = sc.acc[j].add(da[j]);
        sc.qdd[j] += dq[j];
    }
}

/// The articulated-body algorithm on the inertias and bias forces in `sc`:
/// fills `sc.acc` (spatial accelerations) and `sc.qdd` (joint accelerations).
/// Joint damping and limits are implicit in each joint's inertia.
fn solve(model: &Model, s: &State, sc: &mut Scratch, dt: f32) {
    let rate = 1.0 / dt;
    let b = model.pivot.len();
    for j in (1..b).rev() {
        let Some(p) = model.parent[j] else { continue };
        let axis = sc.axis[j];
        let uv = sc.inertia[j].mul(axis);
        let mut d = axis.dot(uv);
        let mut tau = 0.0;
        // Passive damping, implicit: it can never overshoot.
        let damping = joint_damping();
        if damping > 0.0 {
            let c = d * (1.0 / damping);
            tau -= c * s.qd[j];
            d += c * dt;
        }
        // Joint limits: an inelastic stop, like a contact. A joint that
        // would pass its limit this step may turn only as far as the limit
        // (one past it is turned back by a fifth of the overshoot per step),
        // held by a stiff implicit damper toward that rate. It stores no
        // energy: a spring here switched on from the step's predicted angle,
        // so a joint driven hard within one step passed it unopposed and the
        // spring then returned energy nobody paid for.
        let (q, qd) = (s.q[j], s.qd[j]);
        let predicted = q + dt * qd;
        let room = if predicted > model.hi[j] {
            Some(model.hi[j] - q)
        } else if predicted < model.lo[j] {
            Some(model.lo[j] - q)
        } else {
            None
        };
        if let Some(room) = room {
            let upper = predicted > model.hi[j];
            let past = (room < 0.0) == upper;
            let target = if past { room * PUSH_OUT } else { room } * rate;
            // Only a joint heading past the target is held.
            if (qd > target) == upper {
                let c = LIMIT_HARDNESS * d * rate;
                tau -= c * (qd - target);
                d += c * dt;
            }
        }
        let u = tau - axis.dot(sc.bias[j]);
        let dinv = 1.0 / d;
        sc.u_vec[j] = uv;
        sc.d[j] = dinv;
        sc.u[j] = u;
        let mut ia = sc.inertia[j];
        ia.add_outer(-dinv, uv);
        let pa = sc.bias[j].add(ia.mul(sc.cvel[j])).add(uv.scale(u * dinv));
        sc.inertia[p].add(&ia);
        sc.bias[p] = sc.bias[p].add(pa);
    }
    // The neck body floats freely.
    sc.root = sc.inertia[0].inverse();
    sc.acc[0] = sc.root.mul(sc.bias[0]).scale(-1.0);
    for j in 1..b {
        let p = model.parent[j].unwrap_or(0);
        let a = sc.acc[p].add(sc.cvel[j]);
        sc.qdd[j] = (sc.u[j] - sc.u_vec[j].dot(a)) * sc.d[j];
        sc.acc[j] = a.add(sc.axis[j].scale(sc.qdd[j]));
    }
}

/// The change of every body's acceleration (`da`) and joint acceleration
/// (`dq`) under spatial forces on bodies, from the articulated inertias that
/// `solve` left in `sc`.
fn response(model: &Model, sc: &Scratch, forces: &[(usize, V3)], da: &mut [V3], dq: &mut [f32]) {
    let b = model.pivot.len();
    // Each body's share of the forces, passed from its subtree to its
    // parent through the articulated inertias (children first).
    let mut p = [V3::default(); 64];
    for &(body, f) in forces {
        p[body] = p[body].sub(f);
    }
    for j in (1..b).rev() {
        dq[j] = -sc.axis[j].dot(p[j]);
        let up = p[j].add(sc.u_vec[j].scale(dq[j] * sc.d[j]));
        let parent = model.parent[j].unwrap_or(0);
        p[parent] = p[parent].add(up);
    }
    da[0] = sc.root.mul(p[0]).scale(-1.0);
    dq[0] = 0.0;
    for k in 1..b {
        let a = da[model.parent[k].unwrap_or(0)];
        dq[k] = (dq[k] - sc.u_vec[k].dot(a)) * sc.d[k];
        da[k] = a.add(sc.axis[k].scale(dq[k]));
    }
}

/// The v2 GPU kernel (`shaders/physics2_creature.wgsl`) for bodies of up to
/// `capacity` nodes.
pub fn shader_source(capacity: usize, workgroup: u32, fidelity: physics::Fidelity) -> String {
    source_from(
        include_str!("../shaders/physics2_creature.wgsl").to_owned(),
        capacity,
        workgroup,
        fidelity,
    )
}

/// The v2 kernel that also records every creature's trial for a replay, the
/// counterpart of `creature_kernel::record_source`: node positions before
/// each step and after the last, in binding 7 as `[creature][frame][slot]`.
/// A frame is `p.stride` vec2f long: the node positions (`STRIDE` slots),
/// then an (energy, force) pair per muscle, then a (normal, friction) contact
/// force per node, and in the last slot the bits of the broken joints
/// (`creature_kernel::frame_stride`). It computes what
/// `shader_source` computes;
/// its only other change is that a creature keeps moving after its trial
/// ends (limp after a fall) while its result stays the one at the end. A
/// trial starts at the settling tick, as scoring does (so chunk boundaries
/// match), and the kernel fills the frames before it with the start pose.
pub fn record_source(capacity: usize, workgroup: u32, fidelity: physics::Fidelity) -> String {
    let mut source = include_str!("../shaders/physics2_creature.wgsl").to_owned();
    for (from, to) in RECORD_EDITS {
        assert_eq!(source.matches(from).count(), 1, "record edit {from:?}");
        source = source.replace(from, to);
    }
    source_from(source, capacity, workgroup, fidelity)
}

/// The changes `record_source` makes to the kernel text. Each must match
/// exactly once.
const RECORD_EDITS: [(&str, &str); 4] = [
    (
        "@group(0) @binding(6) var<storage, read> tile_info: array<vec4u>;\n",
        "@group(0) @binding(6) var<storage, read> tile_info: array<vec4u>;\n\
         @group(0) @binding(7) var<storage, read_write> frames: array<vec2f>;\n",
    ),
    // The recorded frame is `p.stride` vec2f long: the node positions (`STRIDE`
    // of them), then one (energy, force) pair per muscle, then one (normal,
    // friction) contact force per node, and last the bits of the bones whose
    // joint the scoring test finds broken in this pose. The forces of a frame
    // are those of the step that led to it; the energy is the state it starts
    // from.
    (
        "@compute @workgroup_size(WG)\nfn advance(",
        "fn record_extras(base: u32, muscle_count: u32, tile_x: u32, tl: u32, record_base: u32, nn: u32) {\n\
             for (var k = 0u; k < muscle_count; k++) {\n\
                 let field = tile_x + k * MUSCLE_FIELDS * TILE + tl;\n\
                 frames[base + STRIDE + k] = vec2f(muscle_data[field + 14u * TILE], muscle_data[field + 11u * TILE] + muscle_data[field + 18u * TILE]);\n\
             }\n\
             for (var i = 0u; i < MAXN; i++) {\n\
                 if i >= nn { break; }\n\
                 var w = records[record_base + i].b;\n\
                 if i == 0u {\n\
                     w = records[record_base].c;\n\
                 }\n\
                 frames[base + STRIDE + muscle_count + i] = w;\n\
             }\n\
             var broken = vec2u(0u);\n\
             for (var j = 1u; j < MAXB; j++) {\n\
                 if j >= nb { break; }\n\
                 if q[j] < bone_field(j, 2u) - JOINT_BREAK || q[j] > bone_field(j, 3u) + JOINT_BREAK {\n\
                     if j < 32u { broken.x |= 1u << j; } else { broken.y |= 1u << (j - 32u); }\n\
                 }\n\
             }\n\
             frames[base + p.stride - 1u] = bitcast<vec2f>(broken);\n\
         }\n\
         @compute @workgroup_size(WG)\nfn advance(",
    ),
    (
        "    for (var s = 0u; s < p.steps; s++) {\n        if metrics.fall_time > 0.0 || metrics.screened > 0.0 {\n            break;\n        }\n        let tick = p.tick + s;\n",
        "    // The result at the end of the trial; the body keeps moving after it.\n\
         var kept = metrics;\n\
         var done = metrics.fall_time > 0.0 || metrics.screened > 0.0;\n\
         for (var s = 0u; s < p.steps; s++) {\n\
         let tick = p.tick + s;\n\
         if !done && (metrics.fall_time > 0.0 || metrics.screened > 0.0) {\n\
             kept = metrics;\n\
             kept.head_shake = head_shake;\n\
             done = true;\n\
         }\n\
         let frame = (creature * (p.total_steps + 1u) + tick) * p.stride;\n\
         for (var j = 0u; j < MAXN; j++) {\n\
             if j >= nn { break; }\n\
             frames[frame + j] = node_pos(j);\n\
         }\n\
         record_extras(frame, muscle_count, tile.x, tl, record_base, nn);\n\
         if s == 0u && p.tick == SETTLE {\n\
             for (var t = 0u; t < SETTLE; t++) {\n\
                 let before = (creature * (p.total_steps + 1u) + t) * p.stride;\n\
                 for (var j = 0u; j < MAXN; j++) {\n\
                     if j >= nn { break; }\n\
                     frames[before + j] = node_pos(j);\n\
                 }\n\
                 record_extras(before, muscle_count, tile.x, tl, record_base, nn);\n\
             }\n\
         }\n",
    ),
    (
        "    metrics.head_shake = head_shake;\n    results[creature] = metrics;\n",
        "    metrics.head_shake = head_shake;\n\
         if !done {\n\
             kept = metrics;\n\
         }\n\
         results[creature] = kept;\n\
         if p.tick + p.steps >= p.total_steps {\n\
             let frame = (creature * (p.total_steps + 1u) + p.total_steps) * p.stride;\n\
             for (var j = 0u; j < MAXN; j++) {\n\
                 if j >= nn { break; }\n\
                 frames[frame + j] = node_pos(j);\n\
             }\n\
             record_extras(frame, muscle_count, tile.x, tl, record_base, nn);\n\
         }\n",
    ),
];

fn source_from(
    mut source: String,
    capacity: usize,
    workgroup: u32,
    fidelity: physics::Fidelity,
) -> String {
    let limits = physics::limits();
    let replace = |source: String, from: &str, to: String| {
        assert!(source.contains(from), "kernel text {from:?} missing");
        source.replace(from, &to)
    };
    // Workgroup memory holds 48 KB: bodies above 32 nodes keep their table
    // in private (local) memory instead.
    if capacity > 32 {
        source = replace(
            source,
            "var<workgroup> tab: array<f32, TABF * WG>;",
            "var<private> tab: array<f32, TABF>;".into(),
        );
        source = replace(
            source,
            "    return f * WG + lane_id;",
            "    return f;".into(),
        );
    }
    for (from, to) in [
        (
            "const MUSCLE_CAPACITY: f32 = 120.0;",
            format!("const MUSCLE_CAPACITY: f32 = {:?};", limits.muscle_energy),
        ),
        (
            "const MUSCLE_RECOVERY: f32 = 0.5;",
            format!("const MUSCLE_RECOVERY: f32 = {:?};", limits.muscle_recovery),
        ),
        (
            "const MAX_MUSCLE_FORCE: f32 = 100.0;",
            format!("const MAX_MUSCLE_FORCE: f32 = {:?};", limits.muscle_force),
        ),
        (
            "const INV_JOINT_DAMPING: f32 = 10.0;",
            format!(
                "const INV_JOINT_DAMPING: f32 = {:?};",
                if joint_damping() > 0.0 {
                    1.0 / joint_damping()
                } else {
                    0.0
                }
            ),
        ),
        (
            "const LIMIT_HARDNESS: f32 = 20.0;",
            format!("const LIMIT_HARDNESS: f32 = {LIMIT_HARDNESS:?};"),
        ),
        (
            "const JOINT_BREAK: f32 = 0.5;",
            format!("const JOINT_BREAK: f32 = {:?};", physics::JOINT_BREAK),
        ),
        (
            "const MUD_NORMAL: f32 = 2.0;",
            format!("const MUD_NORMAL: f32 = {:?};", physics::MUD_NORMAL),
        ),
        (
            "const MUD_GRIP: f32 = 2.0;",
            format!("const MUD_GRIP: f32 = {:?};", physics::MUD_GRIP),
        ),
        (
            "const MUD_DRAG: f32 = 2.0;",
            format!("const MUD_DRAG: f32 = {:?};", physics::MUD_DRAG),
        ),
        (
            "const MUD_FULL_DEPTH: f32 = 0.1;",
            format!("const MUD_FULL_DEPTH: f32 = {:?};", physics::MUD_FULL_DEPTH),
        ),
        (
            "const SPIN_CAP: f32 = 15.0;",
            format!("const SPIN_CAP: f32 = {SPIN_CAP:?};"),
        ),
        (
            "const INV_SPIN_CAP: f32 = 0.06666667;",
            format!("const INV_SPIN_CAP: f32 = {:?};", 1.0 / SPIN_CAP),
        ),
        (
            "const SPIN_HARDNESS: f32 = 20.0;",
            format!("const SPIN_HARDNESS: f32 = {SPIN_HARDNESS:?};"),
        ),
        (
            "const PGS_SWEEPS: u32 = 20u;",
            format!("const PGS_SWEEPS: u32 = {}u;", PGS_ITERATIONS),
        ),
        (
            "const PLANT_SWEEPS: u32 = 20u;",
            format!("const PLANT_SWEEPS: u32 = {}u;", PLANT_SWEEPS),
        ),
        (
            "const PLANT_ROUNDS: u32 = 2u;",
            format!("const PLANT_ROUNDS: u32 = {}u;", PLANT_ROUNDS),
        ),
        (
            "const AIR_DRAG: f32 = 0.6;",
            format!("const AIR_DRAG: f32 = {AIR_DRAG:?};"),
        ),
        (
            "const WATER_DRAG: f32 = 100.0;",
            format!("const WATER_DRAG: f32 = {WATER_DRAG:?};"),
        ),
        (
            "const WATER_ALONG: f32 = 0.25;",
            format!("const WATER_ALONG: f32 = {WATER_ALONG:?};"),
        ),
        (
            "const WATER_BUOYANCY: f32 = 0.7;",
            format!("const WATER_BUOYANCY: f32 = {WATER_BUOYANCY:?};"),
        ),
        (
            "const ICE_INV: f32 = 0.16666667;",
            format!("const ICE_INV: f32 = {:?};", 1.0 / physics::ICE_SPACING),
        ),
        (
            "const PUSH_OUT: f32 = 0.2;",
            format!("const PUSH_OUT: f32 = {PUSH_OUT:?};"),
        ),
        (
            "const HEAD_SHAKE_LIMIT: f32 = 78.4;",
            format!(
                "const HEAD_SHAKE_LIMIT: f32 = {:?};",
                physics::HEAD_SHAKE_LIMIT
            ),
        ),
        (
            "const HEAD_SHAKE_WINDOW: f32 = 0.1;",
            format!(
                "const HEAD_SHAKE_WINDOW: f32 = {:?};",
                physics::HEAD_SHAKE_WINDOW
            ),
        ),
        (
            "const CONTACT_SLACK: f32 = 0.002;",
            format!("const CONTACT_SLACK: f32 = {CONTACT_SLACK:?};"),
        ),
        (
            "const LIFT_CLEARANCE: f32 = 0.01;",
            format!("const LIFT_CLEARANCE: f32 = {LIFT_CLEARANCE:?};"),
        ),
        (
            "const GAP_DEPTH: f32 = 2.0;",
            format!("const GAP_DEPTH: f32 = {:?};", physics::GAP_DEPTH),
        ),
        (
            "const GAP_RUN: f32 = 0.15;",
            format!("const GAP_RUN: f32 = {:?};", physics::GAP_RUN),
        ),
        (
            "const HURDLE_SPACING: f32 = 3.0;",
            format!("const HURDLE_SPACING: f32 = {:?};", physics::HURDLE_SPACING),
        ),
        (
            "const HURDLE_TOP: f32 = 1.2;",
            format!("const HURDLE_TOP: f32 = {:?};", physics::HURDLE_TOP),
        ),
        (
            "const HURDLE_RUN: f32 = 0.2;",
            format!("const HURDLE_RUN: f32 = {:?};", physics::HURDLE_RUN),
        ),
        ("PHYSICSRATE", format!("{:.1}", fidelity.rate as f32)),
        ("SETTLESTEPSu", format!("{}u", fidelity.settle())),
        (
            "SAMPLEINTERVALu",
            format!("{}u", fidelity.sample_interval()),
        ),
        ("MAXCONTACTSu", format!("{}u", capacity.min(MAX_CONTACTS))),
        ("WGSIZEu", format!("{workgroup}u")),
        ("STRIDE", format!("{capacity}u")),
        ("MAXNODESu", format!("{capacity}u")),
    ] {
        source = replace(source, from, to);
    }
    let contacts = capacity.min(MAX_CONTACTS);
    // Unrolled, the largest bodies' kernels take the driver very long to
    // build; they are rare and keep their loops.
    if capacity > 16 {
        return source;
    }
    unroll_loops(
        &source,
        &[
            ("MAXN", capacity),
            ("MAXB", capacity - 1),
            ("MAXC", contacts),
            ("MAXR", 2 * contacts),
        ],
    )
}

/// Unrolls every loop of `source` that runs a variable from a literal start
/// to one of the size constants in `bounds` (`for (var j = 0u; j < MAXB;
/// j++) {`): each pass becomes `loop { let j = 3u; <body> break; }`, with
/// the body's own `break` and `continue` statements (those not inside a loop
/// of their own) turned into `break`. That keeps `continue`, and keeps
/// `break` for the kernel's guards, which are monotonic (`if j >= nb {
/// break; }`: once one pass stops, every later one stops too). With every
/// index a constant, the driver can keep the arrays in registers; it keeps
/// them in memory for loops it does not unroll.
fn unroll_loops(source: &str, bounds: &[(&str, usize)]) -> String {
    let mut text = source.to_owned();
    loop {
        let Some((start, var, from, to)) = find_unrollable(&text, bounds) else {
            return text;
        };
        let open = start + text[start..].find('{').expect("loop body");
        let close = matching_brace(&text, open);
        let body = loop_level_breaks(&text[open + 1..close]);
        let mut out = String::from("{\n");
        for value in from..to {
            out.push_str(&format!(
                "loop {{\nlet {var} = {value}u;\n{body}\nbreak;\n}}\n"
            ));
        }
        out.push('}');
        text.replace_range(start..=close, &out);
    }
}

/// The first loop `unroll_loops` expands: its start, variable and range.
fn find_unrollable(text: &str, bounds: &[(&str, usize)]) -> Option<(usize, String, usize, usize)> {
    let mut at = 0;
    while let Some(found) = text[at..].find("for (var ") {
        let start = at + found;
        at = start + 9;
        let header_end = start + text[start..].find(')')?;
        let header = &text[start + 9..header_end];
        // "j = 0u; j < MAXB; j++"
        let parts: Vec<&str> = header.split(';').map(str::trim).collect();
        if parts.len() != 3 {
            continue;
        }
        let Some((var, from)) = parts[0].split_once('=') else {
            continue;
        };
        let (var, from) = (var.trim(), from.trim().trim_end_matches('u'));
        let Ok(from) = from.parse::<usize>() else {
            continue;
        };
        let Some((cmp_var, limit)) = parts[1].split_once('<') else {
            continue;
        };
        if cmp_var.trim() != var || parts[2] != format!("{var}++") {
            continue;
        }
        let Some(&(_, to)) = bounds.iter().find(|(name, _)| *name == limit.trim()) else {
            continue;
        };
        return Some((start, var.to_owned(), from, to));
    }
    None
}

fn matching_brace(text: &str, open: usize) -> usize {
    let mut depth = 0usize;
    for (i, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return open + i;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces in the kernel");
}

/// `body` with its `continue;` statements, and `break;` statements not
/// inside a nested loop, replaced by `break;` (for `unroll_loops`).
fn loop_level_breaks(body: &str) -> String {
    // A stack of open braces: whether each belongs to a loop.
    let mut stack: Vec<bool> = Vec::new();
    let mut out = String::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0;
    let mut pending_loop = false;
    while i < bytes.len() {
        let rest = &body[i..];
        if rest.starts_with("for (") || rest.starts_with("loop {") || rest.starts_with("while ") {
            pending_loop = true;
        }
        match bytes[i] {
            b'{' => {
                stack.push(pending_loop);
                pending_loop = false;
            }
            b'}' => {
                stack.pop();
            }
            _ => {}
        }
        let in_loop = stack.iter().any(|&l| l);
        if !in_loop && rest.starts_with("continue;") {
            out.push_str("break;");
            i += "continue;".len();
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Packs creatures for the v2 kernel, grouped by node capacity as the
/// current kernel's `creature_kernel::pack` does, each in its starting state.
pub fn pack(
    pop: &Population,
    indices: &[usize],
    cfg: &Config,
) -> anyhow::Result<Vec<crate::creature_kernel::LaneBatch>> {
    use crate::creature_kernel::{BONE_FIELDS, CAPACITIES, LaneBatch, TILE, capacity_index};
    let mut groups: Vec<Vec<(usize, usize)>> = vec![Vec::new(); CAPACITIES.len()];
    for (slot, &i) in indices.iter().enumerate() {
        let n = pop.genomes[i].node_count;
        anyhow::ensure!((2..=64).contains(&n), "Unsupported body size");
        groups[capacity_index(n)].push((slot, i));
    }
    Ok(groups
        .into_par_iter()
        .enumerate()
        .filter(|(_, g)| !g.is_empty())
        .map(|(group, mut members)| {
            let capacity = CAPACITIES[group];
            // Similar bodies share a warp: same loop counts.
            members.sort_by_key(|&(_, i)| {
                let g = &pop.genomes[i];
                (g.node_count, g.muscle_count, i)
            });
            let count = members.len();
            let mut tiles = Vec::with_capacity(count.div_ceil(TILE));
            let (mut muscle_len, mut bone_len) = (0usize, 0usize);
            for tile in members.chunks(TILE) {
                let muscles = tile
                    .iter()
                    .map(|&(_, i)| pop.genomes[i].muscle_count)
                    .max()
                    .unwrap_or(0);
                let bones = tile
                    .iter()
                    .map(|&(_, i)| pop.genomes[i].bone_count)
                    .max()
                    .unwrap_or(0);
                tiles.push([
                    muscle_len as u32,
                    bone_len as u32,
                    muscles as u32,
                    bones as u32,
                ]);
                muscle_len += muscles * MUSCLE_FIELDS * TILE;
                bone_len += bones * BONE_FIELDS * TILE;
            }
            let mut nodes = vec![physics::Node::default(); count * capacity];
            let mut muscles = vec![0f32; muscle_len.max(1)];
            let mut bones = vec![0f32; bone_len.max(1)];
            let mut info = Vec::with_capacity(count);
            for (j, &(_, i)) in members.iter().enumerate() {
                let creature = pop.creature(i);
                let model = Model::new(&creature, cfg);
                let start = model.start(cfg);
                let g = &pop.genomes[i];
                info.push([
                    g.node_count as u32,
                    g.bone_count as u32,
                    g.muscle_count as u32,
                    physics::quake_hash(g.id),
                ]);
                let records = &mut nodes[j * capacity..(j + 1) * capacity];
                records[0].pos = start.x0;
                records[0].vel = start.v0;
                for b in 0..model.pivot.len() {
                    let (q, qd) = if b == 0 {
                        (start.th0, start.w0)
                    } else {
                        (start.q[b], start.qd[b])
                    };
                    records[b + 1].pos = [q, qd];
                }
                let (tile, lane) = (tiles[j / TILE], j % TILE);
                for b in 0..model.pivot.len() {
                    let field = tile[1] as usize + b * BONE_FIELDS * TILE + lane;
                    let values = [
                        f32::from_bits(model.pivot[b] as u32),
                        model.length[b],
                        if b == 0 {
                            model.friction[0]
                        } else {
                            model.lo[b]
                        },
                        // The neck has no range: its slot holds the muscle
                        // scale.
                        if b == 0 {
                            model.muscle_scale
                        } else {
                            model.hi[b]
                        },
                        model.mass[b + 1],
                        model.radius[b + 1],
                        model.friction[b + 1],
                        if b == 0 { model.mass[0] } else { 0.0 },
                        if b == 0 { model.radius[0] } else { 0.0 },
                    ];
                    for (f, value) in values.into_iter().enumerate() {
                        bones[field + f * TILE] = value;
                    }
                }
                for (k, m) in model.muscles.iter().enumerate() {
                    let field = tile[0] as usize + k * MUSCLE_FIELDS * TILE + lane;
                    // The four endpoint nodes, 6 bits each, and which of
                    // them senses touchdowns (7 for none).
                    let ends = [
                        model.pivot[m.bone_a],
                        m.bone_a + 1,
                        model.pivot[m.bone_b],
                        m.bone_b + 1,
                    ];
                    let sensor = m
                        .sensor
                        .and_then(|n| ends.iter().position(|&e| e == n))
                        .map_or(7, |e| e as u32);
                    let nodes = ends
                        .iter()
                        .enumerate()
                        .fold(sensor << 24, |w, (e, &n)| w | (n as u32) << (6 * e));
                    let values = [
                        f32::from_bits(nodes),
                        m.anchor_a,
                        m.anchor_b,
                        m.amplitude,
                        m.hill,
                        m.inv_period,
                        m.phase,
                        m.duty,
                        m.stiffness,
                        m.inv_duty,
                        m.inv_complement,
                        0.0,
                        m.reset,
                        0.0,
                        1.0,
                        m.strength,
                        m.tendon_k,
                        m.long,
                        0.0,
                    ];
                    for (f, value) in values.into_iter().enumerate() {
                        muscles[field + f * TILE] = value;
                    }
                }
            }
            LaneBatch {
                capacity,
                slots: members.iter().map(|&(slot, _)| slot).collect(),
                creatures: members.iter().map(|&(_, i)| i).collect(),
                nodes,
                info,
                tiles,
                muscles,
                muscle_fields: MUSCLE_FIELDS,
                bones,
                results: None,
            }
        })
        .collect())
}

/// Evaluates every creature of `unit` with the v2 physics.
pub fn evaluate(unit: &Population, cfg: &Config) -> Vec<GpuResult> {
    (0..unit.genomes.len())
        .into_par_iter()
        .map(|i| run(&Model::new(&unit.creature(i), cfg), cfg, None))
        .collect()
}

/// A creature's recorded trial and its result, for the replay viewer.
pub fn replay(creature: &Creature, cfg: &Config) -> (Vec<Vec<[f32; 2]>>, GpuResult) {
    let (frames, result, _) = replay_forces(creature, cfg);
    (frames, result)
}

/// `replay` with the muscle energy, muscle force and ground contact forces of
/// every frame.
pub fn replay_forces(
    creature: &Creature,
    cfg: &Config,
) -> (Vec<Vec<[f32; 2]>>, GpuResult, crate::replay_forces::Forces) {
    let cfg = Config {
        screen: None,
        ..cfg.clone()
    };
    let mut frames = Vec::new();
    let mut forces = crate::replay_forces::Forces::default();
    let result = run_recorded(
        &Model::new(creature, &cfg),
        &cfg,
        Some(&mut frames),
        Some(&mut forces),
    );
    (frames, result, forces)
}

/// Muscle work (J) of a creature's full trial: the sum of |force x relative
/// speed| x dt that drains the muscle stores, for the cost of transport, with
/// the distance the trial scored.
pub fn trial_work(creature: &Creature, cfg: &Config) -> (f32, f64) {
    let cfg = Config {
        screen: None,
        ..cfg.clone()
    };
    WORK.with(|w| w.set(0.0));
    let result = run(&Model::new(creature, &cfg), &cfg, None);
    (result.fitness, WORK.with(|w| w.get()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::{Bone, NodeGene};

    fn chain(nodes: &[[f32; 2]], muscles: bool) -> Creature {
        let genes: Vec<NodeGene> = nodes
            .iter()
            .map(|p| NodeGene {
                x: p[0],
                y: p[1],
                diameter: 0.08,
                friction: 0.6,
            })
            .collect();
        let bones: Vec<Bone> = (1..nodes.len())
            .map(|i| {
                let (a, b) = (nodes[i - 1], nodes[i]);
                Bone::new((i - 1) as u32, i as u32, (b[0] - a[0]).hypot(b[1] - a[1]))
            })
            .collect();
        let mut c = Creature {
            nodes: genes,
            bones,
            muscles: Vec::new(),
            id: 1,
        };
        if !muscles {
            c.muscles.clear();
        }
        c
    }

    fn calm() -> Config {
        Config {
            random_seed: false,
            duration: 5.0,
            screen: None,
            ..Config::default()
        }
    }

    /// Total momentum and angular momentum about the origin.
    fn momentum(model: &Model, s: &State) -> (f32, f32, f32) {
        let (mut px, mut py, mut l) = (0.0, 0.0, 0.0);
        for i in 0..model.mass.len() {
            let m = model.mass[i];
            px += m * s.vel[i][0];
            py += m * s.vel[i][1];
            l += m * (s.pos[i][0] * s.vel[i][1] - s.pos[i][1] * s.vel[i][0]);
        }
        (px, py, l)
    }

    #[test]
    fn bones_keep_their_length_and_a_free_body_keeps_its_momentum() {
        let cfg = Config {
            ground: false,
            gravity: 0.0,
            air_retention: 1.0,
            ..calm()
        };
        let c = chain(&[[0.0, 1.0], [0.0, 0.7], [0.3, 0.6], [0.5, 0.3]], false);
        let mut model = Model::new(&c, &cfg);
        model.air_drag = 0.0;
        let mut s = model.start(&cfg);
        s.w0 = 1.5;
        s.qd[1] = -3.0;
        s.qd[2] = 2.0;
        s.v0 = [0.4, -0.2];
        model.kinematics(&mut s);
        let before = momentum(&model, &s);
        let mut sc = Scratch {
            inertia: vec![Sym::default(); 3],
            bias: vec![V3::default(); 3],
            force: vec![V3::default(); 3],
            vel: vec![V3::default(); 3],
            cvel: vec![V3::default(); 3],
            axis: vec![V3::default(); 3],
            u_vec: vec![V3::default(); 3],
            d: vec![0.0; 3],
            root: Sym::default(),
            u: vec![0.0; 3],
            acc: vec![V3::default(); 3],
            body_of: (0..4).map(|i| model.body_of(i)).collect(),
            muscle_force: Vec::new(),
            tendon_pull: Vec::new(),
            qdd: vec![0.0; 3],
            impulse_x: 0.0,
            muscle_length: vec![0.0; model.muscles.len()],
            balance_energy: 0.0,
            first_law: 0.0,
            contacts: 0,
            friction: Vec::new(),
            work_total: 0.0,
            ground_work: [0.0; 4],
        };
        let limits = physics::limits();
        for step in 0..600 {
            simulate_step(
                &model,
                &cfg,
                &mut s,
                &mut sc,
                step as f32 / 60.0,
                1.0 / 60.0,
                1.0,
                &limits,
            );
            model.kinematics(&mut s);
        }
        for j in 0..3 {
            let (p, c) = (model.pivot[j], model.child[j]);
            let l = (s.pos[c][0] - s.pos[p][0]).hypot(s.pos[c][1] - s.pos[p][1]);
            assert!((l - model.length[j]).abs() < 1e-4, "bone {j} length {l}");
        }
        // Semi-implicit Euler in reduced coordinates keeps momentum to first
        // order in the step (`a_free_body_keeps_its_momentum_at_any_step`).
        let after = momentum(&model, &s);
        assert!(
            (after.0 - before.0).abs() < 0.03 * before.0.abs(),
            "{before:?} {after:?}"
        );
        assert!(
            (after.1 - before.1).abs() < 0.03 * before.1.abs(),
            "{before:?} {after:?}"
        );
    }

    /// Momentum drift of a spinning free body over 10 s at `rate` Hz.
    fn free_drift(rate: u32) -> f32 {
        let cfg = Config {
            ground: false,
            gravity: 0.0,
            air_retention: 1.0,
            ..calm()
        };
        let c = chain(&[[0.0, 1.0], [0.0, 0.7], [0.3, 0.6], [0.5, 0.3]], false);
        let mut model = Model::new(&c, &cfg);
        model.air_drag = 0.0;
        let mut s = model.start(&cfg);
        s.w0 = 1.5;
        s.qd[1] = -3.0;
        s.qd[2] = 2.0;
        s.v0 = [0.4, -0.2];
        model.kinematics(&mut s);
        let before = momentum(&model, &s);
        let mut sc = scratch(&model);
        let limits = physics::limits();
        let dt = 1.0 / rate as f32;
        for step in 0..rate * 10 {
            simulate_step(
                &model,
                &cfg,
                &mut s,
                &mut sc,
                step as f32 * dt,
                dt,
                1.0,
                &limits,
            );
            model.kinematics(&mut s);
        }
        let after = momentum(&model, &s);
        (after.0 - before.0).abs().max((after.1 - before.1).abs())
    }

    fn scratch(model: &Model) -> Scratch {
        let b = model.pivot.len();
        Scratch {
            inertia: vec![Sym::default(); b],
            bias: vec![V3::default(); b],
            force: vec![V3::default(); b],
            vel: vec![V3::default(); b],
            cvel: vec![V3::default(); b],
            axis: vec![V3::default(); b],
            u_vec: vec![V3::default(); b],
            d: vec![0.0; b],
            root: Sym::default(),
            u: vec![0.0; b],
            acc: vec![V3::default(); b],
            body_of: (0..model.mass.len()).map(|i| model.body_of(i)).collect(),
            muscle_force: vec![0.0; model.muscles.len()],
            tendon_pull: vec![0.0; model.muscles.len()],
            qdd: vec![0.0; b],
            impulse_x: 0.0,
            muscle_length: vec![0.0; model.muscles.len()],
            balance_energy: 0.0,
            first_law: 0.0,
            contacts: 0,
            friction: Vec::new(),
            work_total: 0.0,
            ground_work: [0.0; 4],
        }
    }

    #[test]
    fn a_free_body_keeps_its_momentum_at_any_step() {
        // The momentum balance leaves only rounding (the body carries about
        // 1 N s).
        let (coarse, fine) = (free_drift(60), free_drift(600));
        eprintln!("drift at 60 Hz {coarse}, at 600 Hz {fine}");
        assert!(
            coarse < 1e-4 && fine < 1e-4,
            "drift {coarse} at 60 Hz, {fine} at 600 Hz"
        );
    }

    #[test]
    fn air_drag_only_slows_a_body_and_scales_with_length_and_speed() {
        // A thrown body in still air, with no gravity and no ground: the
        // drag takes kinetic energy and momentum away and never adds any.
        let cfg = Config {
            ground: false,
            gravity: 0.0,
            ..calm()
        };
        let run_body = |scale: f32, speed: f32| {
            let c = chain(
                &[
                    [0.0, 1.0],
                    [0.0, 1.0 - 0.5 * scale],
                    [0.5 * scale, 1.0 - 0.5 * scale],
                ],
                false,
            );
            let model = Model::new(&c, &cfg);
            let mut s = model.start(&cfg);
            s.v0 = [speed, 0.0];
            model.kinematics(&mut s);
            let mut sc = scratch(&model);
            let limits = physics::limits();
            let dt = 1.0 / 60.0;
            let start_momentum: f32 = s.vel.iter().zip(&model.mass).map(|(v, m)| v[0] * m).sum();
            let mut previous = energy(&model, &s, cfg.gravity);
            for step in 0..60 {
                simulate_step(
                    &model,
                    &cfg,
                    &mut s,
                    &mut sc,
                    step as f32 * dt,
                    dt,
                    1.0,
                    &limits,
                );
                model.kinematics(&mut s);
                let e = energy(&model, &s, cfg.gravity);
                assert!(
                    e <= previous + 1e-4,
                    "drag added energy: {previous} -> {e} J"
                );
                previous = e;
            }
            let end_momentum: f32 = s.vel.iter().zip(&model.mass).map(|(v, m)| v[0] * m).sum();
            start_momentum - end_momentum
        };
        let slow = run_body(1.0, 4.0);
        let fast = run_body(1.0, 8.0);
        let long = run_body(2.0, 4.0);
        eprintln!("momentum lost in 1 s: slow {slow}, fast {fast}, long {long}");
        assert!(slow > 0.0, "a thrown body must slow down");
        // Twice the speed loses about four times the momentum (the drag
        // grows with speed squared); longer bones pay for more air.
        assert!(fast > 3.0 * slow, "{fast} against {slow}");
        assert!(long > 1.5 * slow, "{long} against {slow}");
    }

    #[test]
    fn a_tendon_pulls_a_stretched_muscle_back_and_adds_no_energy() {
        use crate::evolution::Muscle;
        let cfg = Config {
            ground: false,
            gravity: 0.0,
            ..calm()
        };
        let body = |tendon: f32| {
            let mut c = chain(&[[0.0, 1.0], [0.0, 0.6], [0.4, 0.6]], false);
            c.bones[1].min_angle = -2.5;
            c.bones[1].max_angle = 2.5;
            // A muscle from the head to the end of the leg, stretched well past
            // its longest length, with no drive of its own.
            c.muscles.push(Muscle {
                bone_a: 0,
                bone_b: 1,
                anchor_a: 0.0,
                anchor_b: 1.0,
                short: 0.3,
                long: 0.3,
                period: 1.0,
                phase: 0.0,
                duty: 0.5,
                stiffness: 60.0,
                sensor: NO_SENSOR,
                reset: 0.0,
                tendon,
            });
            let model = Model::new(&c, &cfg);
            let mut s = model.start(&cfg);
            model.kinematics(&mut s);
            let mut sc = scratch(&model);
            let limits = physics::limits();
            let dt = 1.0 / 60.0;
            let length = |model: &Model, s: &State| model.muscle_lengths(s).next().unwrap();
            let start = length(&model, &s);
            let start_energy = model.energy(&s, 0.0).0;
            let mut shortest = start;
            let mut worst_gain = 0.0f32;
            let mut previous = start_energy;
            for step in 0..120 {
                simulate_step(
                    &model,
                    &cfg,
                    &mut s,
                    &mut sc,
                    step as f32 * dt,
                    dt,
                    1.0,
                    &limits,
                );
                model.kinematics(&mut s);
                let e = model.energy(&s, 0.0).0;
                worst_gain = worst_gain.max(e - previous);
                previous = e;
                shortest = shortest.min(length(&model, &s));
            }
            (
                start,
                shortest,
                start_energy,
                worst_gain,
                model.muscles[0].tendon_k,
            )
        };
        let (start, shortest, stored, gain, k) = body(1.0);
        eprintln!(
            "tendon k {k}: length {start} -> {shortest}, stored {stored} J, worst gain {gain} J"
        );
        assert!(
            k > 0.0 && stored > 0.05,
            "the stretched tendon must store energy: {stored}"
        );
        assert!(
            shortest < start - 0.05,
            "the tendon must pull the muscle back"
        );
        assert!(gain < 0.02 * stored, "a step gained {gain} J of {stored} J");
        let (start, shortest, ..) = body(0.0);
        assert!(
            (start - shortest).abs() < 1e-3,
            "without a tendon nothing moves"
        );
    }

    #[test]
    fn a_pulling_muscle_closes_its_joint() {
        use crate::evolution::Muscle;
        let cfg = Config {
            ground: false,
            gravity: 0.0,
            ..calm()
        };
        // Head, neck base and a leg hanging at a right angle, with a muscle
        // from the middle of the neck to the middle of the leg.
        let mut c = chain(&[[0.0, 1.0], [0.0, 0.6], [0.4, 0.6]], false);
        c.bones[1].min_angle = -2.0;
        c.bones[1].max_angle = 2.0;
        c.muscles.push(Muscle {
            bone_a: 0,
            bone_b: 1,
            anchor_a: 0.5,
            anchor_b: 0.5,
            short: 0.05,
            long: 0.4,
            period: 1.0,
            phase: 0.0,
            duty: 0.5,
            stiffness: 60.0,
            sensor: NO_SENSOR,
            reset: 0.0,
            tendon: 0.0,
        });
        let model = Model::new(&c, &cfg);
        let mut frames = Vec::new();
        run(&model, &cfg, Some(&mut frames));
        let span = |f: &Vec<[f32; 2]>| {
            let a = [(f[0][0] + f[1][0]) / 2.0, (f[0][1] + f[1][1]) / 2.0];
            let b = [(f[1][0] + f[2][0]) / 2.0, (f[1][1] + f[2][1]) / 2.0];
            (a[0] - b[0]).hypot(a[1] - b[1])
        };
        let start = physics::settle() as usize;
        let spans: Vec<f32> = frames[start..start + 30].iter().map(span).collect();
        eprintln!("muscle span over the first half second: {spans:?}");
        assert!(
            spans[29] < spans[0] - 0.01,
            "the muscle did not shorten: {spans:?}"
        );
    }

    /// Kinetic energy and gravity's potential, as the first-law check
    /// counts them (ground contacts and joint limits store none).
    fn energy(model: &Model, s: &State, gravity: f32) -> f32 {
        model.energy(s, gravity).0
    }

    #[test]
    fn a_passive_body_never_gains_energy_on_the_ground() {
        let cfg = calm();
        // Moderate spins: a fast spin gains energy in free flight from the
        // first-order integrator alone (`a_free_body_keeps_its_momentum_at_any_step`),
        // which says nothing about the ground.
        for (spin, drop) in [(0.0, 0.5), (4.0, 0.2), (-6.0, 0.0), (8.0, 1.0)] {
            let c = chain(&[[0.0, 0.5], [0.0, 0.3], [-0.25, 0.1], [0.25, 0.05]], false);
            let model = Model::new(&c, &cfg);
            let mut s = model.start(&cfg);
            s.x0[1] += drop;
            s.w0 = spin;
            s.qd[1] = -spin;
            s.qd[2] = spin * 0.5;
            s.v0 = [3.0, 0.0];
            model.kinematics(&mut s);
            let mut sc = scratch(&model);
            let limits = physics::limits();
            let dt = 1.0 / 60.0;
            let mut worst = 0.0f32;
            let mut previous = energy(&model, &s, cfg.gravity);
            for step in 0..600 {
                simulate_step(
                    &model,
                    &cfg,
                    &mut s,
                    &mut sc,
                    step as f32 * dt,
                    dt,
                    1.0,
                    &limits,
                );
                model.kinematics(&mut s);
                let e = energy(&model, &s, cfg.gravity);
                worst = worst.max(e - previous);
                previous = e;
            }
            eprintln!("spin {spin} drop {drop}: largest energy gain in one step {worst} J");
            assert!(
                worst < 0.05,
                "spin {spin}, drop {drop}: a step gained {worst} J"
            );
        }
    }

    #[test]
    fn fast_passive_bodies_on_the_ground_gain_no_energy() {
        // Random bodies thrown along the ground with every bone turning at up
        // to the spin cap. Nothing drives them, so the largest gain of any
        // step must stay small.
        let cfg = calm();
        let mut rng = crate::evolution::Rng::new(7, 0, 0);
        let mut worst_all = (0.0f32, 0.0f32);
        for trial in 0..400 {
            let n = 3 + trial % 4;
            let mut nodes = vec![[0.0f32, 0.3]];
            for _ in 1..n {
                let last = *nodes.last().unwrap();
                let angle = rng.range(-3.0, 3.0);
                let length = rng.range(0.15, 0.8);
                nodes.push([
                    last[0] + length * angle.cos(),
                    (last[1] + length * angle.sin()).max(0.04),
                ]);
            }
            let c = chain(&nodes, false);
            let mut model = Model::new(&c, &cfg);
            model.air_drag = 0.0;
            let mut s = model.start(&cfg);
            s.v0 = [rng.range(-2.0, 8.0), rng.range(-2.0, 0.5)];
            // Every bone turns at up to the cap, in either direction.
            let mut spin = vec![rng.range(-SPIN_CAP, SPIN_CAP)];
            s.w0 = spin[0];
            for j in 1..model.pivot.len() {
                spin.push(rng.range(-SPIN_CAP, SPIN_CAP));
                s.qd[j] = spin[j] - spin[model.parent[j].unwrap_or(0)];
            }
            model.kinematics(&mut s);
            let mut sc = scratch(&model);
            let limits = physics::limits();
            let dt = 1.0 / 60.0;
            let mut previous = energy(&model, &s, cfg.gravity);
            let start = previous;
            for step in 0..60 {
                simulate_step(
                    &model,
                    &cfg,
                    &mut s,
                    &mut sc,
                    step as f32 * dt,
                    dt,
                    1.0,
                    &limits,
                );
                model.kinematics(&mut s);
                let e = energy(&model, &s, cfg.gravity);
                if e - previous > 1.0 {
                    eprintln!(
                        "trial {trial} step {step}: {previous} -> {e} J, qd {:?}, w0 {}, v0 {:?}, heights {:?}",
                        s.qd,
                        s.w0,
                        s.v0,
                        s.pos.iter().map(|p| p[1]).collect::<Vec<_>>()
                    );
                }
                if e - previous > worst_all.0 {
                    worst_all = (e - previous, start);
                }
                previous = e;
            }
        }
        // The contacts plant against the end pose from a velocity predicted
        // in the start pose; at the spin cap that leaves a gain of about a
        // thousandth of the body's energy in the worst step.
        eprintln!("largest gain in one step {:?} J", worst_all);
        assert!(
            worst_all.0 < 0.05 + 0.002 * worst_all.1,
            "a step gained {:?} J",
            worst_all
        );
    }

    #[test]
    fn a_limb_whipped_into_the_ground_adds_no_energy() {
        // A body sliding forward on its belly with a short bone standing up at
        // its back end, whipped backward and down into the ground. Nothing
        // drives it, so no step may gain energy. At 40 rad/s the contact
        // solve planted the tip, the step's large turn left it sliding
        // forward, and the planting friction kicked the body ahead.
        let cfg = calm();
        for spin in [8.0, 15.0, 40.0] {
            for tip in [0.17f32, 0.4] {
                let c = chain(
                    &[[0.0, 0.4], [0.1, 0.1], [0.9, 0.05], [0.9, 0.05 + tip]],
                    false,
                );
                let model = Model::new(&c, &cfg);
                let mut s = model.start(&cfg);
                s.v0 = [6.5, 0.0];
                s.qd[2] = spin;
                model.kinematics(&mut s);
                let mut sc = scratch(&model);
                let limits = physics::limits();
                let dt = 1.0 / 60.0;
                let mut worst = 0.0f32;
                let mut previous = energy(&model, &s, cfg.gravity);
                for step in 0..30 {
                    simulate_step(
                        &model,
                        &cfg,
                        &mut s,
                        &mut sc,
                        step as f32 * dt,
                        dt,
                        1.0,
                        &limits,
                    );
                    model.kinematics(&mut s);
                    let e = energy(&model, &s, cfg.gravity);
                    worst = worst.max(e - previous);
                    previous = e;
                }
                eprintln!("spin {spin}, tip {tip}: largest energy gain in one step {worst} J");
                assert!(
                    worst < 0.05,
                    "spin {spin}, tip {tip}: a step gained {worst} J"
                );
            }
        }
    }

    #[test]
    fn a_second_bone_at_the_head_turns_on_its_own_joint() {
        // Head, a neck hanging down, and a second bone from the head held out
        // sideways, set turning about its joint at the head. It must turn
        // against the neck (the prototype first locked it to the neck's
        // direction and left its node out of the dynamics).
        let cfg = Config {
            ground: false,
            gravity: 0.0,
            ..calm()
        };
        let mut c = chain(&[[0.0, 1.0], [0.0, 0.6]], false);
        c.nodes.push(NodeGene {
            x: 0.4,
            y: 1.0,
            diameter: 0.08,
            friction: 0.6,
        });
        c.bones.push(Bone::new(0, 2, 0.4));
        let model = Model::new(&c, &cfg);
        assert_eq!(model.parent[1], Some(0), "a head bone turns on the neck");
        let mut s = model.start(&cfg);
        s.qd[1] = 3.0;
        model.kinematics(&mut s);
        let angle = |s: &State, node: usize| {
            (s.pos[node][1] - s.pos[0][1]).atan2(s.pos[node][0] - s.pos[0][0])
        };
        let (side, neck) = (angle(&s, 2), angle(&s, 1));
        let mut sc = scratch(&model);
        let limits = physics::limits();
        for step in 0..20 {
            let dt = 1.0 / 60.0;
            simulate_step(
                &model,
                &cfg,
                &mut s,
                &mut sc,
                step as f32 * dt,
                dt,
                1.0,
                &limits,
            );
            model.kinematics(&mut s);
        }
        assert!(s.pos.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
        let turned = (angle(&s, 2) - side) - (angle(&s, 1) - neck);
        assert!(
            turned > 0.05,
            "the side bone turned {turned} against the neck"
        );
    }

    /// Joint accelerations from the joint-space equations H qdd = tau - C,
    /// with H, C and tau summed over the nodes' Jacobians, for comparison
    /// with the articulated-body pass (`damping` holds each joint's damping
    /// coefficient, explicit only).
    fn joint_space_qdd(model: &Model, s: &State, cfg: &Config, damping: &[f64]) -> Vec<f64> {
        let (b, n) = (model.pivot.len(), model.mass.len());
        let dof = b + 2;
        let origin = s.x0;
        let r = |i: usize| {
            [
                f64::from(s.pos[i][0] - origin[0]),
                f64::from(s.pos[i][1] - origin[1]),
            ]
        };
        // Accelerations of the nodes at zero joint acceleration.
        let mut abar = vec![[0.0f64; 2]; n];
        for j in 0..b {
            let (p, c) = (model.pivot[j], j + 1);
            let d = [
                f64::from(s.pos[c][0] - s.pos[p][0]),
                f64::from(s.pos[c][1] - s.pos[p][1]),
            ];
            let w2 = f64::from(s.om[j]).powi(2);
            abar[c] = [abar[p][0] - w2 * d[0], abar[p][1] - w2 * d[1]];
        }
        // Whether node i hangs below bone j.
        let below = |i: usize, j: usize| {
            let mut node = i;
            while node > 0 {
                if node - 1 == j {
                    return true;
                }
                node = model.pivot[node - 1];
            }
            false
        };
        let jac = |i: usize| -> Vec<[f64; 2]> {
            let ri = r(i);
            let mut cols = vec![[1.0, 0.0], [0.0, 1.0], [-ri[1], ri[0]]];
            for j in 1..b {
                if below(i, j) {
                    let rp = r(model.pivot[j]);
                    cols.push([-(ri[1] - rp[1]), ri[0] - rp[0]]);
                } else {
                    cols.push([0.0, 0.0]);
                }
            }
            cols
        };
        let mut h = vec![0.0f64; dof * dof];
        let mut rhs = vec![0.0f64; dof];
        for (i, ab) in abar.iter().enumerate() {
            let m = f64::from(model.mass[i]);
            let cols = jac(i);
            let f = [f64::from(cfg.wind) * m, -f64::from(cfg.gravity) * m];
            for a in 0..dof {
                for bb in 0..dof {
                    h[a * dof + bb] += m * (cols[a][0] * cols[bb][0] + cols[a][1] * cols[bb][1]);
                }
                rhs[a] += cols[a][0] * f[0] + cols[a][1] * f[1]
                    - m * (cols[a][0] * ab[0] + cols[a][1] * ab[1]);
            }
        }
        for j in 1..b {
            rhs[2 + j] -= damping[j] * f64::from(s.qd[j]);
        }
        // Gaussian elimination (small, symmetric positive definite).
        for c in 0..dof {
            for row in c + 1..dof {
                let f = h[row * dof + c] / h[c * dof + c];
                for k in c..dof {
                    h[row * dof + k] -= f * h[c * dof + k];
                }
                rhs[row] -= f * rhs[c];
            }
        }
        let mut x = vec![0.0f64; dof];
        for c in (0..dof).rev() {
            let mut v = rhs[c];
            for k in c + 1..dof {
                v -= h[c * dof + k] * x[k];
            }
            x[c] = v / h[c * dof + c];
        }
        x
    }

    #[test]
    fn joint_space_equations_match_the_articulated_body_pass() {
        let cfg = Config {
            ground: false,
            ..calm()
        };
        let mut rng = crate::evolution::Rng::new(3, 0, 0);
        let mut worst = 0.0f64;
        for trial in 0..50 {
            let n = 3 + trial % 5;
            let mut nodes = vec![[0.0f32, 1.0]];
            for _ in 1..n {
                // Branch from a random earlier node.
                let from = nodes[rng.index(nodes.len())];
                let angle = rng.range(-3.0, 3.0);
                let length = rng.range(0.15, 0.8);
                nodes.push([
                    from[0] + length * angle.cos(),
                    from[1] + length * angle.sin(),
                ]);
            }
            let mut c = chain(&nodes, false);
            // Rewire bones to a random tree in BFS order.
            for i in 1..n {
                let parent = if i == 1 { 0 } else { rng.index(i) };
                let (a, bnode) = (nodes[parent], nodes[i]);
                c.bones[i - 1] = Bone::new(
                    parent as u32,
                    i as u32,
                    (bnode[0] - a[0]).hypot(bnode[1] - a[1]),
                );
            }
            crate::evolution::canonicalize_bone_order(&mut c);
            let model = Model::new(&c, &cfg);
            let mut s = model.start(&cfg);
            s.v0 = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
            s.w0 = rng.range(-5.0, 5.0);
            for j in 1..model.pivot.len() {
                s.qd[j] = rng.range(-5.0, 5.0);
            }
            model.kinematics(&mut s);
            // The articulated-body pass with a vanishing step, so implicit
            // terms drop out.
            let mut sc = scratch(&model);
            let b = model.pivot.len();
            let origin = s.x0;
            let rel = |p: [f32; 2]| [p[0] - origin[0], p[1] - origin[1]];
            for j in 0..b {
                sc.inertia[j] = Sym::point(model.mass[j + 1], rel(s.pos[j + 1]));
                sc.force[j] = V3::default();
            }
            sc.inertia[0].add(&Sym::point(model.mass[0], [0.0, 0.0]));
            sc.vel[0] = V3::new(s.w0, s.v0[0], s.v0[1]);
            for j in 1..b {
                let p = model.parent[j].unwrap_or(0);
                let r = rel(s.pos[model.pivot[j]]);
                sc.axis[j] = V3::new(1.0, r[1], -r[0]);
                sc.vel[j] = sc.vel[p].add(sc.axis[j].scale(s.qd[j]));
                sc.cvel[j] = crm(sc.vel[j], sc.axis[j]).scale(s.qd[j]);
            }
            for j in 0..b {
                let iv = sc.inertia[j].mul(sc.vel[j]);
                sc.bias[j] = crf(sc.vel[j], iv);
            }
            for i in 0..model.mass.len() {
                let m = model.mass[i];
                let f = [cfg.wind * m, -cfg.gravity * m];
                let j = sc.body_of[i];
                sc.force[j] = sc.force[j].add(force_at(rel(s.pos[i]), f));
            }
            for j in 0..b {
                sc.bias[j] = sc.bias[j].sub(sc.force[j]);
            }
            let tiny = 1e-9;
            solve(&model, &s, &mut sc, tiny);
            // Each joint's damping, as the pass sized it.
            let damping: Vec<f64> = (0..b)
                .map(|j| {
                    if j == 0 {
                        0.0
                    } else {
                        f64::from(1.0 / sc.d[j]) / f64::from(joint_damping())
                    }
                })
                .collect();
            let x = joint_space_qdd(&model, &s, &cfg, &damping);
            let a0 = sc.acc[0];
            let head = [a0.x - s.w0 * s.v0[1], a0.y + s.w0 * s.v0[0]];
            let mut aba = vec![f64::from(head[0]), f64::from(head[1]), f64::from(a0.w)];
            aba.extend(sc.qdd[1..].iter().map(|&v| f64::from(v)));
            for (a, j) in aba.iter().zip(&x) {
                worst = worst.max((a - j).abs() / (1.0 + j.abs()));
            }
        }
        eprintln!("largest relative difference {worst:e}");
        assert!(
            worst < 2e-3,
            "joint space and articulated body differ by {worst}"
        );
    }

    #[test]
    fn the_gpu_packing_holds_each_creature_in_its_starting_state() {
        use crate::creature_kernel::{BONE_FIELDS, TILE};
        let cfg = Config {
            population: 70,
            ..calm()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let indices: Vec<usize> = (0..pop.genomes.len()).collect();
        let batches = pack(&pop, &indices, &cfg).unwrap();
        let mut seen = 0;
        for batch in &batches {
            for (j, &i) in batch.creatures.iter().enumerate() {
                seen += 1;
                let model = Model::new(&pop.creature(i), &cfg);
                let start = model.start(&cfg);
                let records = &batch.nodes[j * batch.capacity..];
                assert_eq!(records[0].pos, start.x0);
                assert_eq!(records[1].pos, [start.th0, 0.0]);
                for b in 1..model.pivot.len() {
                    assert_eq!(records[b + 1].pos, [start.q[b], 0.0]);
                }
                let (tile, lane) = (batch.tiles[j / TILE], j % TILE);
                for b in 0..model.pivot.len() {
                    let field = tile[1] as usize + b * BONE_FIELDS * TILE + lane;
                    assert_eq!(batch.bones[field].to_bits() as usize, model.pivot[b]);
                    assert_eq!(batch.bones[field + 4 * TILE], model.mass[b + 1]);
                }
                for (k, m) in model.muscles.iter().enumerate() {
                    let field = tile[0] as usize + k * MUSCLE_FIELDS * TILE + lane;
                    let packed = batch.muscles[field].to_bits();
                    assert_eq!((packed >> 6) & 63, m.bone_a as u32 + 1);
                    assert_eq!((packed >> 18) & 63, m.bone_b as u32 + 1);
                    assert_eq!(packed & 63, model.pivot[m.bone_a] as u32);
                    assert_eq!(batch.muscles[field + 14 * TILE], 1.0);
                }
            }
        }
        assert_eq!(seen, 70);
    }

    #[test]
    fn the_gpu_kernel_compiles_for_every_capacity() {
        use crate::physics::Fidelity;
        for fidelity in [Fidelity::standard(), Fidelity::fine()] {
            for capacity in crate::creature_kernel::CAPACITIES {
                let source = shader_source(capacity, 32, fidelity);
                crate::vk_engine::spirv(&source)
                    .unwrap_or_else(|e| panic!("{capacity}-node v2 kernel: {e:#}"));
            }
        }
    }

    #[test]
    fn the_recording_kernel_compiles_for_every_capacity() {
        use crate::physics::Fidelity;
        for fidelity in [Fidelity::standard(), Fidelity::fine()] {
            for capacity in crate::creature_kernel::CAPACITIES {
                let source = record_source(capacity, 32, fidelity);
                crate::vk_engine::spirv(&source)
                    .unwrap_or_else(|e| panic!("{capacity}-node v2 recording kernel: {e:#}"));
            }
        }
    }

    #[test]
    fn a_passive_body_lands_rests_and_does_not_travel() {
        let cfg = calm();
        let c = chain(&[[0.0, 0.5], [0.0, 0.3], [-0.25, 0.1], [0.25, 0.05]], false);
        let model = Model::new(&c, &cfg);
        let start = model.start(&cfg);
        let top = start.pos.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        let mut frames = Vec::new();
        let result = run(&model, &cfg, Some(&mut frames));
        for frame in &frames {
            for p in frame {
                assert!(p[1] <= top + 1e-3, "a passive body rose to {}", p[1]);
                assert!(p[1] > -0.2, "a node sank to {}", p[1]);
            }
        }
        assert!(
            result.fitness.abs() < 0.05,
            "a passive body traveled {} m",
            result.fitness
        );
    }

    #[test]
    fn a_passive_body_in_mud_sinks_by_the_mud_depth_and_stays_put() {
        let cfg = Config {
            mud: 0.05,
            ..calm()
        };
        let c = chain(&[[0.0, 0.5], [0.0, 0.3], [-0.25, 0.1], [0.25, 0.05]], false);
        let model = Model::new(&c, &cfg);
        let mut frames = Vec::new();
        let result = run(&model, &cfg, Some(&mut frames));
        let last = frames.last().expect("frames");
        let low = last.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let dry = {
            let mut dry_frames = Vec::new();
            run(&Model::new(&c, &calm()), &calm(), Some(&mut dry_frames));
            dry_frames
                .last()
                .expect("frames")
                .iter()
                .map(|p| p[1])
                .fold(f32::MAX, f32::min)
        };
        assert!(low < dry - 0.01, "no sink: {low} against {dry}");
        assert!(low > dry - 0.06, "sank past the mud: {low} against {dry}");
        assert!(result.fitness.abs() < 0.05, "traveled {}", result.fitness);
    }

    #[test]
    fn a_replay_records_one_set_of_forces_per_frame() {
        let cfg = Config {
            duration: 1.0,
            ..calm()
        };
        let c = chain(&[[0.0, 0.5], [0.0, 0.3], [-0.25, 0.1], [0.25, 0.05]], true);
        let (frames, _, forces) = replay_forces(&c, &cfg);
        assert_eq!(forces.energy.len(), frames.len());
        assert_eq!(forces.muscle.len(), frames.len());
        assert_eq!(forces.ground.len(), frames.len());
        assert_eq!(forces.friction.len(), frames.len());
        assert!(
            forces
                .energy
                .iter()
                .flatten()
                .all(|e| (0.0..=1.0).contains(e))
        );
        assert!(forces.ground.iter().all(|f| f.len() == c.nodes.len()));
        // The bodies start resting on the ground, so contact forces appear.
        assert!(forces.ground.iter().flatten().any(|&f| f > 0.0));
        // Before the trial starts nothing is loaded.
        let settle = physics::settle() as usize;
        assert!(forces.ground[..=settle].iter().flatten().all(|&f| f == 0.0));
    }
}
