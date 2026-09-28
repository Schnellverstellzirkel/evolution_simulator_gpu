//! Physics v2 prototype: a planar articulated tree in reduced coordinates
//! (`docs/hpc-assessment.md` section 7.2, `docs/data-architecture.md`
//! section 8). Opt in with `EVOLUTION_PHYSICS=2`.
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

/// Whether the v2 physics replaces the current one (`EVOLUTION_PHYSICS=2`).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("EVOLUTION_PHYSICS").is_ok_and(|v| v.trim() == "2"))
}

/// How firmly a joint limit holds: its damper weighs this many times the
/// joint's inertia per step.
const LIMIT_HARDNESS: f32 = 20.0;
/// Passive joint damping as a time constant (s): every joint resists its
/// relative rotation like tissue does, with a damper sized to the inertia the
/// joint moves (`EVOLUTION_JOINT_DAMPING`, seconds; 0 turns it off).
pub fn joint_damping() -> f32 {
    static TAU: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *TAU.get_or_init(|| {
        std::env::var("EVOLUTION_JOINT_DAMPING")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &f32| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.1)
    })
}
/// Hill's force-velocity relation: a muscle's active pull falls linearly
/// with its shortening speed and vanishes at this many of its own lengths per
/// second (`EVOLUTION_HILL`; 0 turns it off). It bounds a muscle's power the
/// way real muscle does, so a body cannot catapult itself.
pub fn hill_speed() -> f32 {
    static HILL: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *HILL.get_or_init(|| {
        std::env::var("EVOLUTION_HILL")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &f32| v.is_finite() && *v >= 0.0)
            .unwrap_or(8.0)
    })
}
/// Sliding speed (m/s) below which friction holds a foot (as
/// `physics::PLANTED_SPEED`).
pub const STICK_SPEED: f32 = physics::PLANTED_SPEED;
/// Contact tolerance for the behavior metrics (m), as the current engine.
const CONTACT_SLACK: f32 = 0.002;
const LIFT_CLEARANCE: f32 = 0.01;

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
struct MuscleModel {
    bone_a: usize,
    bone_b: usize,
    anchor_a: f32,
    anchor_b: f32,
    /// Hill's relation as a factor on the shortening speed: 1 / (v_max
    /// times the muscle's length, at least 5 cm).
    hill: f32,
    amplitude: f32,
    inv_period: f32,
    phase: f32,
    duty: f32,
    inv_duty: f32,
    inv_complement: f32,
    stiffness: f32,
    /// Node whose touchdown restarts the rhythm, if any.
    sensor: Option<usize>,
    reset: f32,
}

/// A creature's constants for the v2 physics. Nodes are renumbered so that
/// bone `j` ends at node `j + 1` (node 0 is the head), as the GPU kernel packs
/// them; `order` maps them back to the creature's own numbering.
#[derive(Clone, Debug)]
pub struct Model {
    order: Vec<usize>,
    mass: Vec<f32>,
    radius: Vec<f32>,
    friction: Vec<f32>,
    total_mass: f32,
    inv_mass: f32,
    /// Per bone: pivot node, child node (always the bone's index plus one),
    /// length, parent bone (`None` for the neck), and the relative-angle
    /// range.
    pivot: Vec<usize>,
    child: Vec<usize>,
    length: Vec<f32>,
    parent: Vec<Option<usize>>,
    lo: Vec<f32>,
    hi: Vec<f32>,
    /// Starting relative angle of every bone (the neck: its absolute angle).
    rest: Vec<f32>,
    muscles: Vec<MuscleModel>,
    /// Earthquake bump phase and ground amplitude for this creature.
    quake_phase: f32,
    amplitude: f32,
    start: Vec<[f32; 2]>,
}

/// A creature's state: the head and the neck, then relative joint angles.
#[derive(Clone, Debug)]
pub struct State {
    x0: [f32; 2],
    v0: [f32; 2],
    th0: f32,
    w0: f32,
    q: Vec<f32>,
    qd: Vec<f32>,
    energy: Vec<f32>,
    offset: Vec<f32>,
    /// Each node's contact force (normal, friction) of the last step, which
    /// starts the next step's contact solve.
    warm: Vec<[f32; 2]>,
    /// Derived by `kinematics`: absolute bone angles and rates, node
    /// positions and velocities.
    th: Vec<f32>,
    om: Vec<f32>,
    pos: Vec<[f32; 2]>,
    vel: Vec<[f32; 2]>,
}

fn wrap(a: f32) -> f32 {
    let t = std::f32::consts::TAU;
    a - t * ((a + std::f32::consts::PI) / t).floor()
}

impl Model {
    /// The model of a repaired creature (canonical bone order: bone `j`
    /// joins its parent node `a` to its child node `b`, bone 0 is the neck).
    pub fn new(c: &Creature, cfg: &Config) -> Model {
        let nodes = physics::body(&c.nodes, &c.bones);
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
                    sensor: (m.sensor != NO_SENSOR)
                        .then(|| record[ends[m.sensor as usize] as usize]),
                    reset: m.reset,
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
/// the friction that planted it still pushing (docs/physics-v2.md).
pub const SPIN_CAP: f32 = 15.0;
/// How firmly the spin cap holds: its damper weighs this many times the
/// bone's rotational inertia about its pivot.
const SPIN_HARDNESS: f32 = 20.0;
/// Most nodes in one step's contact solve: the deepest four. Each contact
/// costs the GPU kernel two responses and two rows of Gauss-Seidel, so the
/// bound sets most of its speed (20k evolved bodies: 22M creature-steps/s at
/// four, 4M at eight with 20 sweeps); evolved walkers rarely have more than
/// four feet down, and a node left out sinks a little and joins the next
/// step's solve.
pub const MAX_CONTACTS: usize = 4;
/// The contact bound in use (`EVOLUTION_P2_CONTACTS`, a measuring switch;
/// at most `MAX_CONTACTS`).
fn max_contacts() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("EVOLUTION_P2_CONTACTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (1..=MAX_CONTACTS).contains(&n))
            .unwrap_or(MAX_CONTACTS)
    })
}
/// Gauss-Seidel sweeps over the contacts per step, warm-started from each
/// node's contact force of the last step. Against 20 cold sweeps, 8 warm
/// ones moved 1,000 random bodies by at most 0.1 m in 2 s (p99 0.8 mm).
const PGS_ITERATIONS: usize = 8;
/// Sweeps of the planting pass, which starts from the first solve.
const PLANT_SWEEPS: usize = 4;
fn pgs_iterations() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("EVOLUTION_P2_PGS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PGS_ITERATIONS)
    })
}
/// Gauss-Seidel sweeps of the planting pass, which starts from the first
/// solve's forces (`EVOLUTION_P2_PLANT_SWEEPS`, a measuring switch).
fn plant_sweeps() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("EVOLUTION_P2_PLANT_SWEEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PLANT_SWEEPS)
    })
}
fn warm_start() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("EVOLUTION_P2_WARM").map_or(true, |v| v != "0"))
}
/// Share of a node's depth inside the ground that the contact removes per
/// step.
const PUSH_OUT: f32 = 0.2;

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
}

thread_local! {
    /// Momentum ledger of the last `run` on this thread, summed over steps:
    /// horizontal impulse from the ground and wind, and the body's change of
    /// horizontal momentum. The difference is momentum the integrator made.
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
    /// the energy the first-law check took away.
    pub static ENERGY: std::cell::Cell<[f64; 12]> = const { std::cell::Cell::new([0.0; 12]) };
    /// Steps of the last replay on this thread by how many nodes took part
    /// in the contact solve (0 to `MAX_CONTACTS`).
    pub static CONTACT_COUNTS: std::cell::Cell<[u32; MAX_CONTACTS + 1]> =
        const { std::cell::Cell::new([0; MAX_CONTACTS + 1]) };
}

/// Per-trial totals, as the GPU kernel keeps them (`GpuResult`).
fn fresh_metrics() -> GpuResult {
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
pub fn run(model: &Model, cfg: &Config, mut frames: Option<&mut Vec<Vec<[f32; 2]>>>) -> GpuResult {
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
        qdd: vec![0.0; b],
        impulse_x: 0.0,
        muscle_length: vec![0.0; model.muscles.len()],
        balance_energy: 0.0,
        first_law: 0.0,
        contacts: 0,
        friction: Vec::new(),
    };
    if let Some(frames) = frames.as_deref_mut() {
        for _ in 0..=fidelity.settle() {
            frames.push(model.frame(&s));
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
    let mut energy = [0.0f64; 12];
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
        let broken = (1..b).any(|j| {
            model.parent[j].is_some()
                && (s.q[j] < model.lo[j] - physics::JOINT_BREAK
                    || s.q[j] > model.hi[j] + physics::JOINT_BREAK)
        });
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

fn gait_sample(metrics: &mut GpuResult, center_y: f32) {
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
    for i in 0..n {
        let m = model.mass[i];
        let f = [cfg.wind * m, -gravity * m];
        let j = sc.body_of[i];
        sc.force[j] = sc.force[j].add(force_at(rel(s.pos[i]), f));
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
    let inv_capacity = 1.0 / (limits.muscle_energy * cfg.muscle_energy);
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
        let mut magnitude =
            (drive + relative * 0.15).clamp(-limits.muscle_force, limits.muscle_force);
        if limp {
            magnitude = 0.0;
        }
        let work = (magnitude * relative).abs() * dt;
        s.energy[k] = (energy - work * inv_capacity
            + limits.muscle_recovery * cfg.muscle_recovery * dt * (1.0 - energy))
            .clamp(0.0, 1.0);
        sc.muscle_force[k] = magnitude;
        sc.muscle_length[k] = len;
        muscle_start += magnitude * len;
        // A positive magnitude pulls the two points together.
        let f = [dir[0] * magnitude, dir[1] * magnitude];
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
            let gap = (y - h) / secant - model.radius[i];
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
                target: if gap >= 0.0 {
                    -gap * rate
                } else {
                    -gap * PUSH_OUT * rate
                },
                mu: model.friction[i] * cfg.ground_friction,
            });
        }
    }
    // At most `MAX_CONTACTS` nodes, the deepest, take part in one step's
    // solve, as the GPU kernel bounds it; the others follow the next step.
    if contacts.len() > max_contacts() {
        contacts.sort_by(|a, b| a.reach.total_cmp(&b.reach).then(a.node.cmp(&b.node)));
        contacts.truncate(max_contacts());
        contacts.sort_by_key(|c| c.node);
    }
    let mut impulse = cfg.wind * model.total_mass * dt;
    let mut impulse_y = -cfg.gravity * model.total_mass * dt;
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
        if warm_start() {
            for (i, c) in contacts.iter().enumerate() {
                let [n, t] = s.warm[c.node];
                lambda[2 * i] = n;
                lambda[2 * i + 1] = t.clamp(-c.mu * n, c.mu * n);
            }
        }
        let predicted = pgs(&contacts, &k, &mut lambda, pgs_iterations());
        // The contact forces act on the bodies through one response.
        apply_contacts(model, sc, &contacts, &lambda, &mut da, &mut dq);
        // Plant against the end pose. The solve works on each contact's
        // velocity linearized in the step's start pose, but the step turns
        // the bones, so a foot planted in the start pose can leave the step
        // sliding (a fast-turning bone swings its planted tip forward) while
        // the friction that planted it keeps pushing. So the step is taken,
        // each contact's velocity measured in the end pose (after the
        // momentum balance), and the solve repeated with the difference.
        {
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
            pgs(&contacts, &k, &mut lambda, plant_sweeps());
            let change: Vec<f32> = lambda.iter().zip(&old).map(|(a, b)| a - b).collect();
            apply_contacts(model, sc, &contacts, &change, &mut da, &mut dq);
        }
        s.warm.iter_mut().for_each(|w| *w = [0.0; 2]);
        for (i, c) in contacts.iter().enumerate() {
            s.warm[c.node] = [lambda[2 * i], lambda[2 * i + 1]];
        }
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
    let work = (muscle_start - muscle_end) + cfg.wind * (model.mass_x(s) - mass_x_start);
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
            let friction = (lambda[rt] - v[rt] * (1.0 / k[rt * m + rt])).clamp(-bound, bound);
            let change = friction - lambda[rt];
            lambda[rt] = friction;
            for (j, v) in v.iter_mut().enumerate() {
                *v += k[j * m + rt] * change;
            }
        }
    }
    v
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
    let limits = physics::limits();
    let replace = |source: String, from: &str, to: String| {
        assert!(source.contains(from), "kernel text {from:?} missing");
        source.replace(from, &to)
    };
    let mut source = include_str!("../shaders/physics2_creature.wgsl").to_owned();
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
            format!("const PGS_SWEEPS: u32 = {}u;", pgs_iterations()),
        ),
        (
            "const PLANT_SWEEPS: u32 = 20u;",
            format!("const PLANT_SWEEPS: u32 = {}u;", plant_sweeps()),
        ),
        (
            "const WARM: bool = false;",
            format!("const WARM: bool = {};", warm_start()),
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
        ("MAXCONTACTSu", format!("{}u", capacity.min(max_contacts()))),
        (
            "SELECTTREE",
            (capacity <= 8 && std::env::var("EVOLUTION_P2_SELECT").is_ok_and(|v| v == "1"))
                .to_string(),
        ),
        ("WGSIZEu", format!("{workgroup}u")),
        ("STRIDE", format!("{capacity}u")),
        ("MAXNODESu", format!("{capacity}u")),
    ] {
        source = replace(source, from, to);
    }
    source
}

/// Packs creatures for the v2 kernel, grouped by node capacity as the
/// current kernel's `creature_kernel::pack` does, each in its starting state.
pub fn pack(
    pop: &Population,
    indices: &[usize],
    cfg: &Config,
) -> anyhow::Result<Vec<crate::creature_kernel::LaneBatch>> {
    use crate::creature_kernel::{
        BONE_FIELDS, CAPACITIES, LaneBatch, MUSCLE_FIELDS, TILE, capacity_index,
    };
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
                        model.hi[b],
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
                    let sensor = m.sensor.map_or(NO_SENSOR, |n| n as u32);
                    let values = [
                        f32::from_bits(m.bone_a as u32 | (m.bone_b as u32) << 8 | sensor << 16),
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
                    ];
                    for (f, value) in values.into_iter().enumerate() {
                        muscles[field + f * TILE] = value;
                    }
                }
            }
            LaneBatch {
                capacity,
                plan: None,
                slots: members.iter().map(|&(slot, _)| slot).collect(),
                creatures: members.iter().map(|&(_, i)| i).collect(),
                nodes,
                info,
                tiles,
                muscles,
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
    let cfg = Config {
        screen: None,
        ..cfg.clone()
    };
    let mut frames = Vec::new();
    let result = run(&Model::new(creature, &cfg), &cfg, Some(&mut frames));
    (frames, result)
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
            mutability: 1.0,
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
        let model = Model::new(&c, &cfg);
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
            qdd: vec![0.0; 3],
            impulse_x: 0.0,
            muscle_length: vec![0.0; model.muscles.len()],
            balance_energy: 0.0,
            first_law: 0.0,
            contacts: 0,
            friction: Vec::new(),
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
        let model = Model::new(&c, &cfg);
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
            qdd: vec![0.0; b],
            impulse_x: 0.0,
            muscle_length: vec![0.0; model.muscles.len()],
            balance_energy: 0.0,
            first_law: 0.0,
            contacts: 0,
            friction: Vec::new(),
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
            let model = Model::new(&c, &cfg);
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
        use crate::creature_kernel::{BONE_FIELDS, MUSCLE_FIELDS, TILE};
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
                    assert_eq!(packed & 0xff, m.bone_a as u32);
                    assert_eq!((packed >> 8) & 0xff, m.bone_b as u32);
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
}
