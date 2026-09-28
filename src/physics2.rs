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

/// Joint-limit spring stiffness (N m per radian beyond the range).
pub const JOINT_STIFFNESS: f32 = 400.0;
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
    /// Solves `self x = b` (symmetric positive definite).
    fn solve(&self, b: V3) -> V3 {
        let [a, d, e, bb, f, c] = self.0;
        // Rows (a d e), (d bb f), (e f c).
        let c00 = bb * c - f * f;
        let c01 = e * f - d * c;
        let c02 = d * f - e * bb;
        let det = a * c00 + d * c01 + e * c02;
        let inv = 1.0 / det;
        let c11 = a * c - e * e;
        let c12 = d * e - a * f;
        let c22 = a * bb - d * d;
        V3::new(
            (c00 * b.w + c01 * b.x + c02 * b.y) * inv,
            (c01 * b.w + c11 * b.x + c12 * b.y) * inv,
            (c02 * b.w + c12 * b.x + c22 * b.y) * inv,
        )
    }
}

/// One muscle's constants.
#[derive(Clone, Debug)]
struct MuscleModel {
    bone_a: usize,
    bone_b: usize,
    anchor_a: f32,
    anchor_b: f32,
    long: f32,
    amplitude: f32,
    inv_period: f32,
    phase: f32,
    duty: f32,
    stiffness: f32,
    /// Node whose touchdown restarts the rhythm, if any.
    sensor: Option<usize>,
    reset: f32,
}

/// A creature's constants for the v2 physics.
#[derive(Clone, Debug)]
pub struct Model {
    mass: Vec<f32>,
    radius: Vec<f32>,
    friction: Vec<f32>,
    total_mass: f32,
    /// Per bone: pivot node, child node, length, parent bone (`None` for the
    /// neck), and the relative-angle range and break angles.
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
            let up = parent_of_node[b.a as usize];
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
                    long: m.long,
                    amplitude: (m.long - m.short).min(
                        2.0 * limits.muscle_speed * m.period * m.duty.min(1.0 - m.duty)
                            / std::f32::consts::PI,
                    ),
                    inv_period: 1.0 / m.period,
                    phase: m.phase,
                    duty: m.duty,
                    stiffness: m.stiffness,
                    sensor: (m.sensor != NO_SENSOR).then(|| ends[m.sensor as usize] as usize),
                    reset: m.reset,
                }
            })
            .collect();
        let quake = crate::physics::quake_hash(c.id);
        let still = cfg.quake <= 0.0 || !cfg.ground;
        Model {
            mass: nodes.iter().map(|n| n.mass).collect(),
            radius: nodes.iter().map(|n| n.radius).collect(),
            friction: nodes.iter().map(|n| n.friction).collect(),
            total_mass: nodes.iter().map(|n| n.mass).sum(),
            pivot: c.bones.iter().map(|b| b.a as usize).collect(),
            child: c.bones.iter().map(|b| b.b as usize).collect(),
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
            start: c.nodes.iter().map(|n| [n.x, n.y]).collect(),
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
            energy: vec![1.0; self.muscles.len()],
            offset: vec![0.0; self.muscles.len()],
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

    fn center(&self, s: &State) -> [f32; 2] {
        let mut c = [0.0; 2];
        for (p, m) in s.pos.iter().zip(&self.mass) {
            c[0] += p[0] * m;
            c[1] += p[1] * m;
        }
        [c[0] / self.total_mass, c[1] / self.total_mass]
    }

    /// Body (bone) that carries node `n`: the bone whose child it is, or the
    /// neck for the head.
    fn body_of(&self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            self.child.iter().position(|&c| c == n).unwrap_or(0)
        }
    }
}

/// A node that may touch the ground this step.
struct Contact {
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
/// How firmly the spin cap holds: its damper weighs this many times the
/// bone's rotational inertia about its pivot.
const SPIN_HARDNESS: f32 = 20.0;
/// Gauss-Seidel sweeps over the contacts per step.
const PGS_ITERATIONS: usize = 20;
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
    d: Vec<f32>,
    u: Vec<f32>,
    acc: Vec<V3>,
    body_of: Vec<usize>,
    muscle_force: Vec<f32>,
    qdd: Vec<f32>,
    /// Horizontal impulse the ground and wind gave this step (N s), for the
    /// momentum ledger.
    impulse_x: f32,
}

thread_local! {
    /// Momentum ledger of the last `run` on this thread, summed over steps:
    /// horizontal impulse from the ground and wind, and the body's change of
    /// horizontal momentum. The difference is momentum the integrator made.
    pub static LEDGER: std::cell::Cell<[f64; 2]> = const { std::cell::Cell::new([0.0; 2]) };
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
        u: vec![0.0; b],
        acc: vec![V3::default(); b],
        body_of: (0..n).map(|i| model.body_of(i)).collect(),
        muscle_force: vec![0.0; model.muscles.len()],
        qdd: vec![0.0; b],
        impulse_x: 0.0,
    };
    if let Some(frames) = frames.as_deref_mut() {
        for _ in 0..=fidelity.settle() {
            frames.push(s.pos.clone());
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
    for step in 0..steps {
        let time = step as f32 * dt;
        let head_before = s.vel[0];
        let before = momentum_x(&s);
        let head_acc = simulate_step(model, cfg, &mut s, &mut sc, time, dt, air, &limits);
        let _ = head_acc;
        s.th0 = wrap(s.th0);
        model.kinematics(&mut s);
        ledger[0] += f64::from(sc.impulse_x);
        ledger[1] += momentum_x(&s) - before;
        LEDGER.with(|l| l.set(ledger));
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
        center_y /= n as f32;
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
            let accel = (s.vel[0][0] - head_before[0]).hypot(s.vel[0][1] - head_before[1]) * rate;
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
            frames.push(s.pos.clone());
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
                model.kinematics(&mut s);
                if let Some(frames) = frames.as_deref_mut() {
                    frames.push(s.pos.clone());
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
    let target_speed = |m: &MuscleModel, offset: f32| {
        let wave = |t: f32| {
            let phase = (t * m.inv_period + m.phase + offset).rem_euclid(1.0);
            let w = if phase < m.duty {
                0.5 + 0.5 * (std::f32::consts::PI * phase / m.duty).cos()
            } else {
                0.5 - 0.5 * (std::f32::consts::PI * (phase - m.duty) / (1.0 - m.duty)).cos()
            };
            m.long - m.amplitude * (1.0 - w)
        };
        if settle_free <= 0.0 {
            0.0
        } else {
            (wave(settle_free) - wave((settle_free - dt).max(0.0))) / dt
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
        let len = d[0].hypot(d[1]).max(1e-6);
        let dir = [d[0] / len, d[1] / len];
        let relative = (vb[0] - va[0]) * dir[0] + (vb[1] - va[1]) * dir[1];
        let energy = s.energy[k];
        let mut drive = (-target_speed(m, s.offset[k]) * m.stiffness * 0.25).max(0.0) * energy;
        let hill = hill_speed();
        if hill > 0.0 {
            // Shortening is a negative `relative`.
            drive *= (1.0 + relative / (hill * m.long.max(0.05))).clamp(0.0, 1.0);
        }
        let mut magnitude =
            (drive + relative * 0.15).clamp(-limits.muscle_force, limits.muscle_force);
        if limp {
            magnitude = 0.0;
        }
        let work = (magnitude * relative).abs() * dt;
        s.energy[k] = (energy - work / (limits.muscle_energy * cfg.muscle_energy)
            + limits.muscle_recovery * cfg.muscle_recovery * dt * (1.0 - energy))
            .clamp(0.0, 1.0);
        sc.muscle_force[k] = magnitude;
        // A positive magnitude pulls the two points together.
        let f = [dir[0] * magnitude, dir[1] * magnitude];
        sc.force[m.bone_a] = sc.force[m.bone_a].add(force_at(rel(pa), f));
        sc.force[m.bone_b] = sc.force[m.bone_b].sub(force_at(rel(pb), f));
    }
    // Spin cap: a bone turning faster than `Limits::bone_spin` gets a pure
    // torque back to it. A pure torque changes no linear momentum, so the cap
    // cannot push the body along, and it keeps every step's rotation small
    // enough for the integrator.
    let spin = limits.bone_spin;
    for j in 0..b {
        let w = sc.vel[j].w;
        if w.abs() > spin {
            let l = model.length[j];
            let kappa = SPIN_HARDNESS * model.mass[model.child[j]] * l * l;
            let turn = V3::new(1.0, 0.0, 0.0);
            sc.inertia[j].add_outer(kappa, turn);
            sc.force[j] = sc.force[j].add(turn.scale(-kappa / dt * (w - spin.copysign(w))));
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
                body: j,
                dn,
                dt: dtan,
                vn_free,
                vt_free: v[0] * tangent[0] + v[1] * tangent[1] + dt * (dtan.dot(a) + beta(tangent)),
                target: if gap >= 0.0 {
                    -gap / dt
                } else {
                    -gap * PUSH_OUT / dt
                },
                mu: model.friction[i] * cfg.ground_friction,
            });
        }
    }
    let mut impulse = cfg.wind * model.total_mass * dt;
    let mut impulse_y = -cfg.gravity * model.total_mass * dt;
    if !contacts.is_empty() {
        // Each contact direction's response: the change of every body's
        // acceleration under a unit force there.
        let m = 2 * contacts.len();
        let mut da = vec![V3::default(); m * b];
        let mut dq = vec![0.0f32; m * b];
        for (col, (c, dir)) in contacts
            .iter()
            .flat_map(|c| [(c, c.dn), (c, c.dt)])
            .enumerate()
        {
            response(
                model,
                sc,
                c.body,
                dir,
                &mut da[col * b..(col + 1) * b],
                &mut dq[col * b..(col + 1) * b],
            );
        }
        // Contact-space matrix: velocity change along each row per newton.
        let mut k = vec![0.0f32; m * m];
        for (row, (c, dir)) in contacts
            .iter()
            .flat_map(|c| [(c, c.dn), (c, c.dt)])
            .enumerate()
        {
            for col in 0..m {
                k[row * m + col] = dt * dir.dot(da[col * b + c.body]);
            }
        }
        let mut lambda = vec![0.0f32; m];
        for _ in 0..PGS_ITERATIONS {
            for (i, c) in contacts.iter().enumerate() {
                let (rn, rt) = (2 * i, 2 * i + 1);
                let vn = c.vn_free + (0..m).map(|j| k[rn * m + j] * lambda[j]).sum::<f32>();
                lambda[rn] = (lambda[rn] + (c.target - vn) / k[rn * m + rn]).max(0.0);
                let vt = c.vt_free + (0..m).map(|j| k[rt * m + j] * lambda[j]).sum::<f32>();
                let bound = c.mu * lambda[rn];
                lambda[rt] = (lambda[rt] - vt / k[rt * m + rt]).clamp(-bound, bound);
            }
        }
        for (col, &l) in lambda.iter().enumerate() {
            if l == 0.0 {
                continue;
            }
            for j in 0..b {
                sc.acc[j] = sc.acc[j].add(da[col * b + j].scale(l));
                sc.qdd[j] += l * dq[col * b + j];
            }
        }
        for (i, c) in contacts.iter().enumerate() {
            impulse += (lambda[2 * i] * c.dn.x + lambda[2 * i + 1] * c.dt.x) * dt;
            impulse_y += (lambda[2 * i] * c.dn.y + lambda[2 * i + 1] * c.dt.y) * dt;
        }
    }
    sc.impulse_x = impulse;
    let qdd = &sc.qdd;
    // Integrate: semi-implicit Euler, then air drag on every velocity.
    let a0 = sc.acc[0];
    let head = [a0.x - s.w0 * s.v0[1], a0.y + s.w0 * s.v0[0]];
    s.v0 = [
        (s.v0[0] + head[0] * dt) * air,
        (s.v0[1] + head[1] * dt) * air,
    ];
    s.w0 = (s.w0 + a0.w * dt) * air;
    s.x0 = [s.x0[0] + s.v0[0] * dt, s.x0[1] + s.v0[1] * dt];
    s.th0 += s.w0 * dt;
    for ((q, qd), a) in s.q.iter_mut().zip(&mut s.qd).zip(qdd).skip(1) {
        *qd = (*qd + a * dt) * air;
        *q += *qd * dt;
    }
    // Momentum balance: the body's momentum changes only by the external
    // impulses (gravity, wind, the ground) and the air. First-order
    // integration in joint coordinates misses that by a little each step,
    // and a flailing body could collect the misses; the difference is spread
    // over the body as one uniform velocity, which moves no node against
    // another.
    model.kinematics(s);
    let after = momentum(s);
    let expected = [(before[0] + impulse) * air, (before[1] + impulse_y) * air];
    let shift = [
        (expected[0] - after[0]) / model.total_mass,
        (expected[1] - after[1]) / model.total_mass,
    ];
    s.v0 = [s.v0[0] + shift[0], s.v0[1] + shift[1]];
    for v in &mut s.vel {
        v[0] += shift[0];
        v[1] += shift[1];
    }
    head[0].hypot(head[1])
}

/// The articulated-body algorithm on the inertias and bias forces in `sc`:
/// fills `sc.acc` (spatial accelerations) and `sc.qdd` (joint accelerations).
/// Joint damping and limits are implicit in each joint's inertia.
fn solve(model: &Model, s: &State, sc: &mut Scratch, dt: f32) {
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
            let c = d / damping;
            tau -= c * s.qd[j];
            d += c * dt;
        }
        // Joint limits: an implicit spring and damper beyond the range.
        let predicted = s.q[j] + dt * s.qd[j];
        let beyond = if predicted > model.hi[j] {
            predicted - model.hi[j]
        } else if predicted < model.lo[j] {
            predicted - model.lo[j]
        } else {
            0.0
        };
        if beyond != 0.0 {
            let c = 2.0 * (JOINT_STIFFNESS * d.max(1e-9)).sqrt();
            tau += -JOINT_STIFFNESS * beyond - c * s.qd[j];
            d += JOINT_STIFFNESS * dt * dt + c * dt;
        }
        let u = tau - axis.dot(sc.bias[j]);
        sc.u_vec[j] = uv;
        sc.d[j] = d;
        sc.u[j] = u;
        let mut ia = sc.inertia[j];
        ia.add_outer(-1.0 / d, uv);
        let pa = sc.bias[j].add(ia.mul(sc.cvel[j])).add(uv.scale(u / d));
        sc.inertia[p].add(&ia);
        sc.bias[p] = sc.bias[p].add(pa);
    }
    // The neck body floats freely.
    sc.acc[0] = sc.inertia[0].solve(sc.bias[0].scale(-1.0));
    for j in 1..b {
        let p = model.parent[j].unwrap_or(0);
        let a = sc.acc[p].add(sc.cvel[j]);
        sc.qdd[j] = (sc.u[j] - sc.u_vec[j].dot(a)) / sc.d[j];
        sc.acc[j] = a.add(sc.axis[j].scale(sc.qdd[j]));
    }
}

/// The change of every body's acceleration (`da`) and joint acceleration
/// (`dq`) under a unit spatial force `f` on `body`, from the articulated
/// inertias that `solve` left in `sc`.
fn response(model: &Model, sc: &Scratch, body: usize, f: V3, da: &mut [V3], dq: &mut [f32]) {
    let b = model.pivot.len();
    let mut du = [0.0f32; 64];
    let mut p = f.scale(-1.0);
    let mut j = body;
    while let Some(parent) = model.parent[j] {
        du[j] = -sc.axis[j].dot(p);
        p = p.add(sc.u_vec[j].scale(du[j] / sc.d[j]));
        j = parent;
    }
    da[0] = sc.inertia[0].solve(p.scale(-1.0));
    dq[0] = 0.0;
    for k in 1..b {
        let a = da[model.parent[k].unwrap_or(0)];
        dq[k] = (du[k] - sc.u_vec[k].dot(a)) / sc.d[k];
        da[k] = a.add(sc.axis[k].scale(dq[k]));
    }
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
            u: vec![0.0; 3],
            acc: vec![V3::default(); 3],
            body_of: (0..4).map(|i| model.body_of(i)).collect(),
            muscle_force: Vec::new(),
            qdd: vec![0.0; 3],
            impulse_x: 0.0,
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
        // order in the step (`momentum_drift_shrinks_with_the_step`).
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
            u: vec![0.0; b],
            acc: vec![V3::default(); b],
            body_of: (0..model.mass.len()).map(|i| model.body_of(i)).collect(),
            muscle_force: vec![0.0; model.muscles.len()],
            qdd: vec![0.0; b],
            impulse_x: 0.0,
        }
    }

    #[test]
    fn momentum_drift_shrinks_with_the_step() {
        let (coarse, fine) = (free_drift(60), free_drift(600));
        eprintln!("drift at 60 Hz {coarse}, at 600 Hz {fine}");
        assert!(
            fine < coarse * 0.3,
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

    /// Kinetic energy, gravity's potential, and the energy stored in the
    /// joint-limit springs (ground contacts store none).
    fn energy(model: &Model, s: &State, gravity: f32) -> f32 {
        let nodes: f32 = (0..model.mass.len())
            .map(|i| {
                let v = s.vel[i];
                model.mass[i] * (0.5 * (v[0] * v[0] + v[1] * v[1]) + gravity * s.pos[i][1])
            })
            .sum();
        let joints: f32 = (1..model.pivot.len())
            .map(|j| {
                let beyond = (s.q[j] - model.hi[j]).max(0.0) + (model.lo[j] - s.q[j]).max(0.0);
                0.5 * JOINT_STIFFNESS * beyond * beyond
            })
            .sum();
        nodes + joints
    }

    #[test]
    fn a_passive_body_never_gains_energy_on_the_ground() {
        let cfg = calm();
        // Moderate spins: a fast spin gains energy in free flight from the
        // first-order integrator alone (`momentum_drift_shrinks_with_the_step`),
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
