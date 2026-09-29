//! The fast CPU engine of physics v2: 16 creatures with an identical body plan
//! share one SIMD group. Bone and node indices are uniform across the group,
//! so every value is one `F` (a 16-lane vector, one lane per creature) and the
//! loops over nodes, bones and muscles are the scalar reference's loops
//! (`physics2::run`, `simulate_step_inner`) with every operation done on all
//! lanes at once. Control flow that differs between creatures (contacts, the
//! joint limits, the first-law check, falls) becomes masks and selects.
//!
//! Every lane's arithmetic is the scalar reference's, operation for operation
//! and in the same order (no fused multiply-add), so a lane's result equals
//! `physics2::run`'s for the same creature bit for bit. The only per-lane
//! scalar calls are the libm sine, cosine and the ground function, exactly the
//! reference's. `tests/physics2_lanes.rs` (and the tests below) hold the two
//! engines to that. The reference stays the definition the GPU kernels are
//! measured against; this engine only makes the CPU fallback fast.
use crate::config::Config;
use crate::creature_kernel::GpuResult;
use crate::evolution::Population;
use crate::physics::{self, Limits};
use crate::physics2::{
    LIFT_CLEARANCE, LIMIT_HARDNESS, MAX_CONTACTS, Model, PGS_ITERATIONS, PLANT_ROUNDS,
    PLANT_SWEEPS, PUSH_OUT, SPIN_CAP, SPIN_HARDNESS, State, WATER_ALONG, WATER_BUOYANCY,
    WATER_DRAG, fresh_metrics, gait_sample, joint_damping,
};
use crate::simd::{F, M};
use rayon::prelude::*;
use std::collections::HashMap;

const L: usize = crate::simd::LANES;
const MAXC: usize = MAX_CONTACTS;
const MAXR: usize = 2 * MAX_CONTACTS;

#[inline(always)]
fn sp(x: f32) -> F {
    F::splat(x)
}
#[inline(always)]
fn zero() -> F {
    F::splat(0.0)
}
#[inline(always)]
fn sel(m: M, a: F, b: F) -> F {
    F::select(m, a, b)
}
#[inline(always)]
fn clamp(x: F, lo: F, hi: F) -> F {
    // `f32::clamp`: the low bound first, then the high one.
    let x = sel(x.lt(lo), lo, x);
    sel(x.gt(hi), hi, x)
}
#[inline(always)]
fn arr(x: F) -> [f32; L] {
    x.to_array()
}
#[inline(always)]
fn vec_of(a: [f32; L]) -> F {
    F::load(&a)
}
#[inline(always)]
fn map(x: F, f: impl Fn(f32) -> f32) -> F {
    let a = arr(x);
    vec_of(std::array::from_fn(|l| f(a[l])))
}
fn ge(a: F, b: F) -> M {
    !a.lt(b)
}
fn lane_bit(m: M, l: usize) -> bool {
    (m.bits() >> l) & 1 == 1
}
/// `f32::rem_euclid(1.0)` on every lane.
fn rem1(x: F) -> F {
    let trunc = sel(x.lt(zero()), -(-x).floor(), x.floor());
    let r = x - trunc;
    sel(r.lt(zero()), r + sp(1.0), r)
}

/// `physics::ice` on every lane: how icy the ground is at `x`.
fn ice(x: F) -> F {
    let u = x * sp(1.0 / physics::ICE_SPACING);
    let w = u - u.floor();
    let t = (w - sp(0.5)).abs() * sp(2.0);
    let s = clamp((sp(0.7) - t) * sp(2.5), zero(), sp(1.0));
    s * s * (sp(3.0) - sp(2.0) * s)
}

/// A spatial vector (angular, x, y) on every lane.
#[derive(Clone, Copy)]
struct V3 {
    w: F,
    x: F,
    y: F,
}
impl V3 {
    fn new(w: F, x: F, y: F) -> V3 {
        V3 { w, x, y }
    }
    fn zero() -> V3 {
        V3::new(zero(), zero(), zero())
    }
    fn add(self, o: V3) -> V3 {
        V3::new(self.w + o.w, self.x + o.x, self.y + o.y)
    }
    fn sub(self, o: V3) -> V3 {
        V3::new(self.w - o.w, self.x - o.x, self.y - o.y)
    }
    fn scale(self, s: F) -> V3 {
        V3::new(self.w * s, self.x * s, self.y * s)
    }
    fn dot(self, o: V3) -> F {
        self.w * o.w + self.x * o.x + self.y * o.y
    }
    fn sel(m: M, a: V3, b: V3) -> V3 {
        V3::new(sel(m, a.w, b.w), sel(m, a.x, b.x), sel(m, a.y, b.y))
    }
}
fn crm(v: V3, u: V3) -> V3 {
    V3::new(zero(), -v.w * u.y + u.w * v.y, v.w * u.x - u.w * v.x)
}
fn crf(v: V3, f: V3) -> V3 {
    V3::new(v.x * f.y - v.y * f.x, -v.w * f.y, v.w * f.x)
}
fn force_at(r: [F; 2], f: [F; 2]) -> V3 {
    V3::new(r[0] * f[1] - r[1] * f[0], f[0], f[1])
}

/// Symmetric 3x3 spatial inertia: [ww, wx, wy, xx, xy, yy].
#[derive(Clone, Copy)]
struct Sym([F; 6]);
impl Sym {
    fn zero() -> Sym {
        Sym([zero(); 6])
    }
    fn mul(&self, v: V3) -> V3 {
        let a = &self.0;
        V3::new(
            a[0] * v.w + a[1] * v.x + a[2] * v.y,
            a[1] * v.w + a[3] * v.x + a[4] * v.y,
            a[2] * v.w + a[4] * v.x + a[5] * v.y,
        )
    }
    fn add_outer(&mut self, k: F, d: V3) {
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
    fn point(m: F, c: [F; 2]) -> Sym {
        Sym([
            m * (c[0] * c[0] + c[1] * c[1]),
            -m * c[1],
            m * c[0],
            m,
            zero(),
            m,
        ])
    }
    fn inverse(&self) -> Sym {
        let [a, d, e, bb, f, c] = self.0;
        let c00 = bb * c - f * f;
        let c01 = e * f - d * c;
        let c02 = d * f - e * bb;
        let inv = sp(1.0) / (a * c00 + d * c01 + e * c02);
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
    fn sel(m: M, a: &Sym, b: &Sym) -> Sym {
        Sym(std::array::from_fn(|i| sel(m, a.0[i], b.0[i])))
    }
}

/// One muscle's constants on every lane.
struct Mus {
    /// Lanes that have a muscle at this index (a body's muscles are counted
    /// from 0, so a shorter list leaves the later lanes without one).
    exists: M,
    /// Lanes whose muscle attaches its first (second) end to bone `j`.
    on_a: Vec<M>,
    on_b: Vec<M>,
    /// The node whose touchdown restarts the rhythm, per lane.
    sensor: [Option<usize>; L],
    anchor_a: F,
    anchor_b: F,
    hill: F,
    long: F,
    tendon_k: F,
    amplitude: F,
    inv_period: F,
    phase: F,
    duty: F,
    inv_duty: F,
    inv_complement: F,
    stiffness: F,
    strength: F,
    reset: F,
}

/// A creature state on every lane.
#[derive(Clone)]
struct St {
    x0: [F; 2],
    v0: [F; 2],
    th0: F,
    w0: F,
    q: Vec<F>,
    qd: Vec<F>,
    energy: Vec<F>,
    offset: Vec<F>,
    warm: Vec<[F; 2]>,
    th: Vec<F>,
    om: Vec<F>,
    pos: Vec<[F; 2]>,
    vel: Vec<[F; 2]>,
}

/// Scratch of one step (the reference's `Scratch`).
struct Sc {
    inertia: Vec<Sym>,
    bias: Vec<V3>,
    force: Vec<V3>,
    vel: Vec<V3>,
    cvel: Vec<V3>,
    axis: Vec<V3>,
    u_vec: Vec<V3>,
    d: Vec<F>,
    root: Sym,
    u: Vec<F>,
    acc: Vec<V3>,
    qdd: Vec<F>,
    muscle_force: Vec<F>,
    da: Vec<V3>,
    dq: Vec<F>,
}

/// The contact slots of one step: at most `MAXC` nodes per lane, in node
/// order, each with what the reference's `Contact` holds.
struct Slots {
    /// Lanes where slot `c` holds a contact.
    on: [M; MAXC],
    /// Lanes where slot `c` holds node `i`.
    at: Vec<[M; MAXC]>,
    /// Lanes where slot `c`'s node belongs to body `j`.
    body: Vec<[M; MAXC]>,
    normal: [[F; 2]; MAXC],
    tangent: [[F; 2]; MAXC],
    dn: [V3; MAXC],
    dt: [V3; MAXC],
    vn_free: [F; MAXC],
    vt_free: [F; MAXC],
    vt_start: [F; MAXC],
    target: [F; MAXC],
    mu: [F; MAXC],
}

pub struct Group {
    n: usize,
    b: usize,
    real: usize,
    pivot: Vec<usize>,
    parent: Vec<Option<usize>>,
    mass: Vec<F>,
    radius: Vec<F>,
    friction: Vec<F>,
    length: Vec<F>,
    lo: Vec<F>,
    hi: Vec<F>,
    total_mass: F,
    inv_mass: F,
    muscle_scale: F,
    air_drag: f32,
    muscles: Vec<Mus>,
    amplitude: [f32; L],
    quake_phase: [f32; L],
    start: St,
}

impl Group {
    fn build(models: &[&Model], states: &[State]) -> Group {
        let real = models.len();
        let lane = |l: usize| l.min(real - 1);
        let m0 = models[0];
        let (n, b) = (m0.mass.len(), m0.pivot.len());
        let collect = |f: &dyn Fn(&Model) -> f32| -> F {
            vec_of(std::array::from_fn(|l| f(models[lane(l)])))
        };
        let kmax = models.iter().map(|m| m.muscles.len()).max().unwrap_or(0);
        let mus = |k: usize, default: f32, f: &dyn Fn(&crate::physics2::MuscleModel) -> f32| -> F {
            vec_of(std::array::from_fn(|l| {
                models[lane(l)].muscles.get(k).map_or(default, f)
            }))
        };
        let st = |f: &dyn Fn(&State) -> f32| -> F {
            vec_of(std::array::from_fn(|l| f(&states[lane(l)])))
        };
        let start = St {
            x0: [st(&|s| s.x0[0]), st(&|s| s.x0[1])],
            v0: [st(&|s| s.v0[0]), st(&|s| s.v0[1])],
            th0: st(&|s| s.th0),
            w0: st(&|s| s.w0),
            q: (0..b).map(|j| st(&|s| s.q[j])).collect(),
            qd: (0..b).map(|j| st(&|s| s.qd[j])).collect(),
            energy: (0..kmax)
                .map(|k| st(&|s| s.energy.get(k).copied().unwrap_or(1.0)))
                .collect(),
            offset: (0..kmax)
                .map(|k| st(&|s| s.offset.get(k).copied().unwrap_or(0.0)))
                .collect(),
            warm: (0..n)
                .map(|i| [st(&|s| s.warm[i][0]), st(&|s| s.warm[i][1])])
                .collect(),
            th: (0..b).map(|j| st(&|s| s.th[j])).collect(),
            om: (0..b).map(|j| st(&|s| s.om[j])).collect(),
            pos: (0..n)
                .map(|i| [st(&|s| s.pos[i][0]), st(&|s| s.pos[i][1])])
                .collect(),
            vel: (0..n)
                .map(|i| [st(&|s| s.vel[i][0]), st(&|s| s.vel[i][1])])
                .collect(),
        };
        Group {
            n,
            b,
            real,
            pivot: m0.pivot.clone(),
            parent: m0.parent.clone(),
            mass: (0..n).map(|i| collect(&|m| m.mass[i])).collect(),
            radius: (0..n).map(|i| collect(&|m| m.radius[i])).collect(),
            friction: (0..n).map(|i| collect(&|m| m.friction[i])).collect(),
            length: (0..b).map(|j| collect(&|m| m.length[j])).collect(),
            lo: (0..b).map(|j| collect(&|m| m.lo[j])).collect(),
            hi: (0..b).map(|j| collect(&|m| m.hi[j])).collect(),
            total_mass: collect(&|m| m.total_mass),
            inv_mass: collect(&|m| m.inv_mass),
            muscle_scale: collect(&|m| m.muscle_scale),
            air_drag: m0.air_drag,
            muscles: (0..kmax)
                .map(|k| {
                    let has = |l: usize| l < real && models[l].muscles.len() > k;
                    let bits = |f: &dyn Fn(usize) -> bool| -> M {
                        M((0..L).fold(0u16, |m, l| m | u16::from(has(l) && f(l)) << l))
                    };
                    Mus {
                        exists: bits(&|_| true),
                        on_a: (0..b)
                            .map(|j| bits(&|l| models[l].muscles[k].bone_a == j))
                            .collect(),
                        on_b: (0..b)
                            .map(|j| bits(&|l| models[l].muscles[k].bone_b == j))
                            .collect(),
                        sensor: std::array::from_fn(|l| {
                            if has(l) {
                                models[l].muscles[k].sensor
                            } else {
                                None
                            }
                        }),
                        anchor_a: mus(k, 0.0, &|m| m.anchor_a),
                        anchor_b: mus(k, 0.0, &|m| m.anchor_b),
                        hill: mus(k, 0.0, &|m| m.hill),
                        long: mus(k, 0.0, &|m| m.long),
                        tendon_k: mus(k, 0.0, &|m| m.tendon_k),
                        amplitude: mus(k, 0.0, &|m| m.amplitude),
                        inv_period: mus(k, 1.0, &|m| m.inv_period),
                        phase: mus(k, 0.0, &|m| m.phase),
                        duty: mus(k, 0.5, &|m| m.duty),
                        inv_duty: mus(k, 2.0, &|m| m.inv_duty),
                        inv_complement: mus(k, 2.0, &|m| m.inv_complement),
                        stiffness: mus(k, 0.0, &|m| m.stiffness),
                        strength: mus(k, 1.0, &|m| m.strength),
                        reset: mus(k, 0.0, &|m| m.reset),
                    }
                })
                .collect(),
            amplitude: std::array::from_fn(|l| models[lane(l)].amplitude),
            quake_phase: std::array::from_fn(|l| models[lane(l)].quake_phase),
            start,
        }
    }
}

/// A group running a trial.
struct Sim<'a> {
    g: &'a Group,
    cfg: &'a Config,
    limits: Limits,
    s: St,
    sc: Sc,
    dt: f32,
    /// `1 / dt`, which the reference's step uses (not always the fidelity's
    /// rate, bit for bit).
    rate: f32,
    air: f32,
    /// Lanes still running their trial.
    active: M,
    /// The world has no bumps, tilt, pits or steps on any lane: the ground is
    /// flat and level.
    calm: bool,
}

impl<'a> Sim<'a> {
    fn new(g: &'a Group, cfg: &'a Config) -> Sim<'a> {
        let (n, b) = (g.n, g.b);
        let fidelity = cfg.fidelity();
        let rate = fidelity.rate as f32;
        let calm = cfg.slope == 0.0
            && cfg.gaps == 0.0
            && cfg.hurdles == 0.0
            && g.amplitude.iter().all(|&a| a == 0.0);
        let _ = n;
        Sim {
            g,
            cfg,
            limits: physics::limits(),
            s: g.start.clone(),
            sc: Sc {
                inertia: vec![Sym::zero(); b],
                bias: vec![V3::zero(); b],
                force: vec![V3::zero(); b],
                vel: vec![V3::zero(); b],
                cvel: vec![V3::zero(); b],
                axis: vec![V3::zero(); b],
                u_vec: vec![V3::zero(); b],
                d: vec![zero(); b],
                root: Sym::zero(),
                u: vec![zero(); b],
                acc: vec![V3::zero(); b],
                qdd: vec![zero(); b],
                muscle_force: vec![zero(); g.muscles.len()],
                da: vec![V3::zero(); b],
                dq: vec![zero(); b],
            },
            dt: 1.0 / rate,
            rate: 1.0 / (1.0 / rate),
            air: fidelity.air_per_step(cfg.air_retention),
            active: M(0xffff),
            calm,
        }
    }

    /// The ground's height and slope under `x` on every lane
    /// (`Model::ground`).
    fn ground(&self, x: F) -> (F, F) {
        if !self.cfg.ground {
            return (sp(f32::NEG_INFINITY), zero());
        }
        if self.calm {
            return (zero(), zero());
        }
        let xs = arr(x);
        let mut h = [0.0; L];
        let mut s = [0.0; L];
        for l in 0..L {
            (h[l], s[l]) = physics::ground(
                xs[l],
                self.g.amplitude[l],
                self.cfg.slope,
                self.cfg.gaps,
                self.cfg.hurdles,
                self.g.quake_phase[l],
            );
        }
        (vec_of(h), vec_of(s))
    }

    #[inline(never)]
    fn kinematics(&mut self) {
        let g = self.g;
        let s = &mut self.s;
        s.pos[0] = s.x0;
        s.vel[0] = s.v0;
        for j in 0..g.b {
            let (th, om) = match g.parent[j] {
                Some(p) => (s.th[p] + s.q[j], s.om[p] + s.qd[j]),
                None => (s.th0, s.w0),
            };
            s.th[j] = th;
            s.om[j] = om;
            let ta = arr(th);
            let mut sn = [0.0; L];
            let mut cs = [0.0; L];
            for l in 0..L {
                (sn[l], cs[l]) = ta[l].sin_cos();
            }
            let (sin, cos) = (vec_of(sn), vec_of(cs));
            let (p, c, l) = (g.pivot[j], j + 1, g.length[j]);
            s.pos[c] = [s.pos[p][0] + l * cos, s.pos[p][1] + l * sin];
            s.vel[c] = [s.vel[p][0] - l * om * sin, s.vel[p][1] + l * om * cos];
        }
    }

    fn momentum(&self) -> [F; 2] {
        let mut p = [zero(); 2];
        for (v, m) in self.s.vel.iter().zip(&self.g.mass) {
            p[0] += v[0] * *m;
            p[1] += v[1] * *m;
        }
        p
    }

    fn energy(&self) -> (F, F) {
        let (mut total, mut scale) = (zero(), zero());
        let gravity = sp(self.cfg.gravity);
        for i in 0..self.g.n {
            let v = self.s.vel[i];
            let kinetic = sp(0.5) * self.g.mass[i] * (v[0] * v[0] + v[1] * v[1]);
            let potential = self.g.mass[i] * gravity * self.s.pos[i][1];
            total += kinetic + potential;
            scale += kinetic + potential.abs();
        }
        // Elastic energy stored in the tendons (zero on lanes without one).
        for (k, m) in self.g.muscles.iter().enumerate() {
            let stretch = (self.muscle_length(k) - m.long).max(zero());
            let stored = sp(0.5) * m.tendon_k * stretch * stretch;
            total += stored;
            scale += stored;
        }
        (total, scale)
    }

    fn mass_x(&self) -> F {
        let mut sum = zero();
        for (p, m) in self.s.pos.iter().zip(&self.g.mass) {
            sum += p[0] * *m;
        }
        sum
    }

    /// The value `f(j)` of the bone `j` each lane's mask selects.
    fn pick(&self, masks: &[M], f: impl Fn(usize) -> F) -> F {
        let mut out = zero();
        for (j, &m) in masks.iter().enumerate() {
            out = sel(m, f(j), out);
        }
        out
    }

    /// The position and velocity of the point at `t` along the bone that
    /// each lane's mask selects.
    fn bone_point(&self, masks: &[M], t: F) -> ([F; 2], [F; 2]) {
        let g = self.g;
        let s = &self.s;
        let pp = [
            self.pick(masks, |j| s.pos[g.pivot[j]][0]),
            self.pick(masks, |j| s.pos[g.pivot[j]][1]),
        ];
        let pc = [
            self.pick(masks, |j| s.pos[j + 1][0]),
            self.pick(masks, |j| s.pos[j + 1][1]),
        ];
        let vp = [
            self.pick(masks, |j| s.vel[g.pivot[j]][0]),
            self.pick(masks, |j| s.vel[g.pivot[j]][1]),
        ];
        let vc = [
            self.pick(masks, |j| s.vel[j + 1][0]),
            self.pick(masks, |j| s.vel[j + 1][1]),
        ];
        (
            [pp[0] + (pc[0] - pp[0]) * t, pp[1] + (pc[1] - pp[1]) * t],
            [vp[0] + (vc[0] - vp[0]) * t, vp[1] + (vc[1] - vp[1]) * t],
        )
    }

    fn muscle_length(&self, k: usize) -> F {
        let m = &self.g.muscles[k];
        let (a, _) = self.bone_point(&m.on_a, m.anchor_a);
        let (b, _) = self.bone_point(&m.on_b, m.anchor_b);
        let d = [b[0] - a[0], b[1] - a[1]];
        (d[0] * d[0] + d[1] * d[1]).sqrt()
    }

    fn center_x(&self) -> F {
        let mut c = zero();
        for (p, m) in self.s.pos.iter().zip(&self.g.mass) {
            c += p[0] * *m;
        }
        c * self.g.inv_mass
    }

    /// Semi-implicit Euler on the joint coordinates (`physics2::integrate`).
    fn integrate(&mut self, air: F) {
        let dt = sp(self.dt);
        let a0 = self.sc.acc[0];
        let s = &mut self.s;
        let head = [a0.x - s.w0 * s.v0[1], a0.y + s.w0 * s.v0[0]];
        s.v0 = [
            (s.v0[0] + head[0] * dt) * air,
            (s.v0[1] + head[1] * dt) * air,
        ];
        s.w0 = (s.w0 + a0.w * dt) * air;
        s.x0 = [s.x0[0] + s.v0[0] * dt, s.x0[1] + s.v0[1] * dt];
        s.th0 += s.w0 * dt;
        for j in 1..self.g.b {
            s.qd[j] = (s.qd[j] + self.sc.qdd[j] * dt) * air;
            s.q[j] += s.qd[j] * dt;
        }
        self.kinematics();
    }

    /// The articulated-body algorithm on the inertias and bias forces in `sc`
    /// (`physics2::solve`).
    #[inline(never)]
    fn solve(&mut self) {
        let g = self.g;
        let dt = sp(self.dt);
        let rate = sp(self.rate);
        let damping = joint_damping();
        let sc = &mut self.sc;
        let s = &self.s;
        for j in (1..g.b).rev() {
            let Some(p) = g.parent[j] else { continue };
            let axis = sc.axis[j];
            let uv = sc.inertia[j].mul(axis);
            let mut d = axis.dot(uv);
            let mut tau = zero();
            if damping > 0.0 {
                let c = d * sp(1.0 / damping);
                tau -= c * s.qd[j];
                d += c * dt;
            }
            let (q, qd) = (s.q[j], s.qd[j]);
            let predicted = q + dt * qd;
            let upper = predicted.gt(g.hi[j]);
            let lower = predicted.lt(g.lo[j]);
            let limited = upper | lower;
            if limited.any() {
                let room = sel(upper, g.hi[j] - q, g.lo[j] - q);
                // `past = (room < 0.0) == upper`.
                let past = past_mask(room.lt(zero()), upper);
                let target = sel(past, room * sp(PUSH_OUT), room) * rate;
                // Only a joint heading past the target is held.
                let held = limited & (qd.gt(target) & upper | !qd.gt(target) & !upper);
                let c = sp(LIMIT_HARDNESS) * d * rate;
                let tau_held = tau - c * (qd - target);
                let d_held = d + c * dt;
                tau = sel(held, tau_held, tau);
                d = sel(held, d_held, d);
            }
            let u = tau - axis.dot(sc.bias[j]);
            let dinv = sp(1.0) / d;
            sc.u_vec[j] = uv;
            sc.d[j] = dinv;
            sc.u[j] = u;
            let mut ia = sc.inertia[j];
            ia.add_outer(-dinv, uv);
            let pa = sc.bias[j].add(ia.mul(sc.cvel[j])).add(uv.scale(u * dinv));
            sc.inertia[p].add(&ia);
            sc.bias[p] = sc.bias[p].add(pa);
        }
        sc.root = sc.inertia[0].inverse();
        sc.acc[0] = sc.root.mul(sc.bias[0]).scale(sp(-1.0));
        for j in 1..g.b {
            let p = g.parent[j].unwrap_or(0);
            let a = sc.acc[p].add(sc.cvel[j]);
            sc.qdd[j] = (sc.u[j] - sc.u_vec[j].dot(a)) * sc.d[j];
            sc.acc[j] = a.add(sc.axis[j].scale(sc.qdd[j]));
        }
    }

    /// The change of every body's acceleration (`sc.da`) and joint
    /// acceleration (`sc.dq`) under the spatial forces `forces`, one per
    /// slot, on the body each slot's node belongs to.
    fn response(&mut self, slots: &Slots, forces: &[(usize, V3)]) {
        let g = self.g;
        let sc = &mut self.sc;
        let mut p = vec![V3::zero(); g.b];
        for &(slot, f) in forces {
            for (bd, pb) in p.iter_mut().enumerate() {
                *pb = V3::sel(slots.body[bd][slot], pb.sub(f), *pb);
            }
        }
        for j in (1..g.b).rev() {
            sc.dq[j] = -sc.axis[j].dot(p[j]);
            let up = p[j].add(sc.u_vec[j].scale(sc.dq[j] * sc.d[j]));
            let parent = g.parent[j].unwrap_or(0);
            p[parent] = p[parent].add(up);
        }
        sc.da[0] = sc.root.mul(p[0]).scale(sp(-1.0));
        sc.dq[0] = zero();
        for k in 1..g.b {
            let a = sc.da[g.parent[k].unwrap_or(0)];
            sc.dq[k] = (sc.dq[k] - sc.u_vec[k].dot(a)) * sc.d[k];
            sc.da[k] = a.add(sc.axis[k].scale(sc.dq[k]));
        }
    }

    /// Adds the accelerations that contact forces `lambda` cause; only lanes
    /// in `lanes` (those with a contact) change.
    #[inline(never)]
    fn apply_contacts(&mut self, slots: &Slots, lambda: &[F; MAXR], lanes: M) {
        let forces: Vec<(usize, V3)> = (0..MAXC)
            .filter(|&c| slots.on[c].any())
            .map(|c| {
                (
                    c,
                    slots.dn[c]
                        .scale(lambda[2 * c])
                        .add(slots.dt[c].scale(lambda[2 * c + 1])),
                )
            })
            .collect();
        self.response(slots, &forces);
        for j in 0..self.g.b {
            let (da, dq) = (self.sc.da[j], self.sc.dq[j]);
            self.sc.acc[j] = V3::sel(lanes, self.sc.acc[j].add(da), self.sc.acc[j]);
            self.sc.qdd[j] = sel(lanes, self.sc.qdd[j] + dq, self.sc.qdd[j]);
        }
    }
}

fn past_mask(room_neg: M, upper: M) -> M {
    (room_neg & upper) | (!room_neg & !upper)
}

impl Sim<'_> {
    /// A per-slot quantity `q` of the slot's node: `f(i)` for the node `i`
    /// each lane's slot `c` holds.
    fn at_slot(&self, slots: &Slots, c: usize, f: impl Fn(usize) -> F) -> F {
        let mut out = zero();
        for i in 0..self.g.n {
            out = sel(slots.at[i][c], f(i), out);
        }
        out
    }

    /// One physics step of every lane (`physics2::simulate_step_inner`).
    fn step(&mut self, time: f32) {
        let g = self.g;
        let cfg = self.cfg;
        let (n, b) = (g.n, g.b);
        let dt = sp(self.dt);
        let rate = sp(self.rate);
        let air = sp(self.air);
        let origin = self.s.x0;
        let rel = |p: [F; 2]| [p[0] - origin[0], p[1] - origin[1]];
        let before = self.momentum();
        let (energy_start, energy_scale) = self.energy();
        let mass_x_start = self.mass_x();
        let mut muscle_start = zero();
        // Body inertias (true, for the velocity products) and external forces.
        for j in 0..b {
            let c = j + 1;
            self.sc.inertia[j] = Sym::point(g.mass[c], rel(self.s.pos[c]));
            self.sc.force[j] = V3::zero();
        }
        self.sc.inertia[0].add(&Sym::point(g.mass[0], [zero(), zero()]));
        // Spatial velocities, joint axes and velocity-product accelerations.
        self.sc.vel[0] = V3::new(self.s.w0, self.s.v0[0], self.s.v0[1]);
        for j in 1..b {
            let p = g.parent[j].unwrap_or(0);
            let r = rel(self.s.pos[g.pivot[j]]);
            self.sc.axis[j] = V3::new(sp(1.0), r[1], -r[0]);
            self.sc.vel[j] = self.sc.vel[p].add(self.sc.axis[j].scale(self.s.qd[j]));
            self.sc.cvel[j] = crm(self.sc.vel[j], self.sc.axis[j]).scale(self.s.qd[j]);
        }
        for j in 0..b {
            let iv = self.sc.inertia[j].mul(self.sc.vel[j]);
            self.sc.bias[j] = crf(self.sc.vel[j], iv);
        }
        // Gravity, wind and mud drag on every node.
        let mud = if cfg.ground { cfg.mud } else { 0.0 };
        let mut mud_impulse = zero();
        for i in 0..n {
            let m = g.mass[i];
            let mut f = [sp(cfg.wind) * m, sp(-cfg.gravity) * m];
            if mud > 0.0 {
                let (h, slope) = self.ground(self.s.pos[i][0]);
                let secant = (sp(1.0) + slope * slope).sqrt();
                let dry = (self.s.pos[i][1] - h) / secant - g.radius[i];
                let sink = clamp(-dry, zero(), sp(mud)) * sp(1.0 / physics::MUD_FULL_DEPTH);
                let drag = -m * sp(physics::MUD_DRAG) * sink * self.s.vel[i][0];
                f[0] += drag;
                mud_impulse += drag * dt;
            }
            let j = i.saturating_sub(1);
            self.sc.force[j] = self.sc.force[j].add(force_at(rel(self.s.pos[i]), f));
        }
        // Air drag on every bone, at its midpoint. The push is limited so a
        // step of drag never more than halves the speed it acts on.
        let mut air_impulse = [zero(); 2];
        for j in 0..b {
            let (p, c) = (g.pivot[j], j + 1);
            let (sp_, sc_) = (self.s.pos[p], self.s.pos[c]);
            let (vp, vc) = (self.s.vel[p], self.s.vel[c]);
            let half = sp(0.5);
            let mid = [half * (sp_[0] + sc_[0]), half * (sp_[1] + sc_[1])];
            let v = [half * (vp[0] + vc[0]), half * (vp[1] + vc[1])];
            let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
            let width = g.radius[p] + g.radius[c];
            let strength = (sp(g.air_drag) * g.length[j] * width * speed)
                .min(half * g.mass[c] * rate)
                .max(zero());
            let f = [-v[0] * strength, -v[1] * strength];
            self.sc.force[j] = self.sc.force[j].add(force_at(rel(mid), f));
            air_impulse[0] += f[0] * dt;
            air_impulse[1] += f[1] * dt;
        }
        // Water below the waterline: buoyancy on every node and anisotropic
        // drag on every bone (`physics2::simulate_step_inner`).
        let water = cfg.water;
        let mut buoy: Vec<F> = Vec::new();
        let mut water_y0: Vec<F> = Vec::new();
        if water > 0.0 {
            let submerged: Vec<F> = (0..n)
                .map(|i| {
                    clamp(
                        (sp(water) - (self.s.pos[i][1] - g.radius[i])) / (sp(2.0) * g.radius[i]),
                        zero(),
                        sp(1.0),
                    )
                })
                .collect();
            buoy = (0..n)
                .map(|i| sp(WATER_BUOYANCY) * g.mass[i] * sp(cfg.gravity) * submerged[i])
                .collect();
            water_y0 = (0..n).map(|i| self.s.pos[i][1]).collect();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let j = i.saturating_sub(1);
                self.sc.force[j] =
                    self.sc.force[j].add(force_at(rel(self.s.pos[i]), [zero(), buoy[i]]));
                air_impulse[1] += buoy[i] * dt;
            }
            for j in 0..b {
                let (p, c) = (g.pivot[j], j + 1);
                let (sp_, sc_) = (self.s.pos[p], self.s.pos[c]);
                let (vp, vc) = (self.s.vel[p], self.s.vel[c]);
                let half = sp(0.5);
                let wet = half * (submerged[p] + submerged[c]);
                let mid = [half * (sp_[0] + sc_[0]), half * (sp_[1] + sc_[1])];
                let v = [half * (vp[0] + vc[0]), half * (vp[1] + vc[1])];
                let inverse = sp(1.0) / g.length[j];
                let axis = [(sc_[0] - sp_[0]) * inverse, (sc_[1] - sp_[1]) * inverse];
                let along = v[0] * axis[0] + v[1] * axis[1];
                let lengthwise = [axis[0] * along, axis[1] * along];
                let sideways = [v[0] - lengthwise[0], v[1] - lengthwise[1]];
                let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
                let width = g.radius[p] + g.radius[c];
                let strength = (sp(WATER_DRAG) * wet * g.length[j] * width * speed)
                    .min(half * g.mass[c] * rate)
                    .max(zero());
                let weak = strength * sp(WATER_ALONG);
                let f = [
                    -(sideways[0] * strength + lengthwise[0] * weak),
                    -(sideways[1] * strength + lengthwise[1] * weak),
                ];
                self.sc.force[j] = self.sc.force[j].add(force_at(rel(mid), f));
                air_impulse[0] += f[0] * dt;
                air_impulse[1] += f[1] * dt;
            }
        }
        // Muscles: the drive and damper, applied at the attachment points.
        for k in 0..g.muscles.len() {
            let m = &g.muscles[k];
            let (pa, va) = self.bone_point(&m.on_a, m.anchor_a);
            let (pb, vb) = self.bone_point(&m.on_b, m.anchor_b);
            let d = [pb[0] - pa[0], pb[1] - pa[1]];
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt().max(sp(1e-6));
            let inverse = sp(1.0) / len;
            let dir = [d[0] * inverse, d[1] * inverse];
            let relative = (vb[0] - va[0]) * dir[0] + (vb[1] - va[1]) * dir[1];
            let energy = self.s.energy[k];
            // The waveform's target length is `long - amplitude * (1 - w)`.
            let target_speed = if time <= 0.0 {
                zero()
            } else {
                let wave = |t: f32| -> F {
                    let x = sp(t) * m.inv_period + m.phase + self.s.offset[k];
                    let phase = rem1(x);
                    let first = phase.lt(m.duty);
                    let arg = sel(
                        first,
                        sp(std::f32::consts::PI) * (phase * m.inv_duty),
                        sp(std::f32::consts::PI) * ((phase - m.duty) * m.inv_complement),
                    );
                    let c = map(arg, f32::cos);
                    sel(first, sp(0.5) + sp(0.5) * c, sp(0.5) - sp(0.5) * c)
                };
                m.amplitude * (wave(time) - wave((time - self.dt).max(0.0))) * rate
            };
            let mut drive = (-target_speed * m.stiffness * sp(0.25)).max(zero()) * energy;
            let hilled = drive * clamp(sp(1.0) + relative * m.hill, zero(), sp(1.0));
            drive = sel(m.hill.gt(zero()), hilled, drive);
            let cap = sp(self.limits.muscle_force) * m.strength * g.muscle_scale;
            let inv_capacity = sp(1.0)
                / (sp(self.limits.muscle_energy * cfg.muscle_energy) * m.strength * g.muscle_scale);
            let magnitude = clamp(drive + relative * sp(0.15), -cap, cap);
            // Only active contraction costs energy.
            let work = drive.min(cap) * (-relative).max(zero()) * dt;
            let new_energy = clamp(
                energy - work * inv_capacity
                    + sp(self.limits.muscle_recovery * cfg.muscle_recovery)
                        * dt
                        * (sp(1.0) - energy),
                zero(),
                sp(1.0),
            );
            self.s.energy[k] = sel(m.exists, new_energy, energy);
            self.sc.muscle_force[k] = sel(m.exists, magnitude, zero());
            muscle_start = sel(m.exists, muscle_start + magnitude * len, muscle_start);
            // The tendon pulls back passively once the muscle is stretched past
            // its longest length.
            let tendon_pull = m.tendon_k * (len - m.long).max(zero());
            let pull = magnitude + tendon_pull;
            let f = [dir[0] * pull, dir[1] * pull];
            let fa = force_at(rel(pa), f);
            let fb = force_at(rel(pb), f);
            for j in 0..b {
                self.sc.force[j] = V3::sel(m.on_a[j], self.sc.force[j].add(fa), self.sc.force[j]);
            }
            for j in 0..b {
                self.sc.force[j] = V3::sel(m.on_b[j], self.sc.force[j].sub(fb), self.sc.force[j]);
            }
        }
        // Spin cap: rotational drag past `SPIN_CAP`.
        for j in 0..b {
            let w = self.sc.vel[j].w;
            let fast = w.abs().gt(sp(SPIN_CAP));
            if fast.any() {
                let l = g.length[j];
                let drag = sp(SPIN_HARDNESS)
                    * g.mass[j + 1]
                    * l
                    * l
                    * (w.abs() * sp(1.0 / SPIN_CAP) - sp(1.0));
                let turn = V3::new(sp(1.0), zero(), zero());
                let mut inertia = self.sc.inertia[j];
                inertia.add_outer(drag, turn);
                self.sc.inertia[j] = Sym::sel(fast, &inertia, &self.sc.inertia[j]);
                let pushed = self.sc.force[j].add(turn.scale(-drag * rate * w));
                self.sc.force[j] = V3::sel(fast, pushed, self.sc.force[j]);
            }
        }
        // The step without the ground.
        for j in 0..b {
            self.sc.bias[j] = self.sc.bias[j].sub(self.sc.force[j]);
        }
        self.solve();
        let mut impulse = sp(cfg.wind) * g.total_mass * dt + mud_impulse + air_impulse[0];
        let mut impulse_y = sp(-cfg.gravity) * g.total_mass * dt + air_impulse[1];
        let mut has_contact = M(0);
        if cfg.ground {
            has_contact = self.contacts(before, &mut impulse, &mut impulse_y, mud);
        }
        self.integrate(air);
        // Momentum balance.
        let after = self.momentum();
        let expected = [(before[0] + impulse) * air, (before[1] + impulse_y) * air];
        let shift = [
            (expected[0] - after[0]) * g.inv_mass,
            (expected[1] - after[1]) * g.inv_mass,
        ];
        self.s.v0 = [self.s.v0[0] + shift[0], self.s.v0[1] + shift[1]];
        for v in &mut self.s.vel {
            v[0] += shift[0];
            v[1] += shift[1];
        }
        // First law in flight.
        let mut muscle_end = zero();
        for k in 0..g.muscles.len() {
            muscle_end = sel(
                g.muscles[k].exists,
                muscle_end + self.sc.muscle_force[k] * self.muscle_length(k),
                muscle_end,
            );
        }
        let mut work = (muscle_start - muscle_end) + sp(cfg.wind) * (self.mass_x() - mass_x_start);
        if water > 0.0 {
            // Buoyancy lifts the body: its work is the force times the rise.
            let mut lift = zero();
            for i in 0..n {
                lift += buoy[i] * (self.s.pos[i][1] - water_y0[i]);
            }
            work += lift;
        }
        let (energy_end, _) = self.energy();
        let excess = energy_end - energy_start - work - (sp(1e-4) + sp(1e-5) * energy_scale);
        let fix = excess.gt(zero()) & !has_contact;
        if fix.any() {
            let center = [expected[0] * g.inv_mass, expected[1] * g.inv_mass];
            let mut internal = zero();
            for (v, m) in self.s.vel.iter().zip(&g.mass) {
                let (x, y) = (v[0] - center[0], v[1] - center[1]);
                internal += sp(0.5) * *m * (x * x + y * y);
            }
            let keep = sel(
                internal.gt(zero()),
                (sp(1.0) - excess / internal).max(zero()).sqrt(),
                zero(),
            );
            self.s.w0 = sel(fix, self.s.w0 * keep, self.s.w0);
            for j in 1..b {
                self.s.qd[j] = sel(fix, self.s.qd[j] * keep, self.s.qd[j]);
            }
            let v0 = [
                center[0] + keep * (self.s.v0[0] - center[0]),
                center[1] + keep * (self.s.v0[1] - center[1]),
            ];
            self.s.v0 = [sel(fix, v0[0], self.s.v0[0]), sel(fix, v0[1], self.s.v0[1])];
            for v in &mut self.s.vel {
                let scaled = [
                    center[0] + keep * (v[0] - center[0]),
                    center[1] + keep * (v[1] - center[1]),
                ];
                *v = [sel(fix, scaled[0], v[0]), sel(fix, scaled[1], v[1])];
            }
        }
    }

    /// The ground contacts of the step: detection, the deepest `MAXC` nodes
    /// per lane, the contact-space matrix, projected Gauss-Seidel, and the
    /// planting rounds. Adds the ground's impulse to `impulse`. Returns the
    /// lanes that had any contact.
    #[inline(never)]
    fn contacts(&mut self, before: [F; 2], impulse: &mut F, impulse_y: &mut F, mud: f32) -> M {
        let g = self.g;
        let cfg = self.cfg;
        let (n, b) = (g.n, g.b);
        let dt = sp(self.dt);
        let rate = sp(self.rate);
        let origin = self.s.x0;
        let rel = |p: [F; 2]| [p[0] - origin[0], p[1] - origin[1]];
        // Every node's contact quantities on every lane.
        struct NodeQ {
            normal: [F; 2],
            tangent: [F; 2],
            reach: F,
            dn: V3,
            dt: V3,
            vn_free: F,
            vt_free: F,
            vt_start: F,
            target: F,
            mu: F,
            candidate: M,
        }
        let mut nodes: Vec<NodeQ> = Vec::with_capacity(n);
        for i in 0..n {
            let [x, y] = self.s.pos[i];
            let (h, slope) = self.ground(x);
            let secant = (sp(1.0) + slope * slope).sqrt();
            let normal = [-slope / secant, sp(1.0) / secant];
            let tangent = [normal[1], -normal[0]];
            let dry = (y - h) / secant - g.radius[i];
            let gap = dry + sp(mud);
            let sink = clamp(-dry, zero(), sp(mud)) * sp(1.0 / physics::MUD_FULL_DEPTH);
            let j = i.saturating_sub(1);
            let r = rel(self.s.pos[i]);
            let v = self.s.vel[i];
            let w = self.sc.vel[j].w;
            let beta = |dir: [F; 2]| w * (-v[1] * dir[0] + v[0] * dir[1]);
            let dn = force_at(r, normal);
            let dtan = force_at(r, tangent);
            let a = self.sc.acc[j];
            let vn_free = v[0] * normal[0] + v[1] * normal[1] + dt * (dn.dot(a) + beta(normal));
            let reach = gap + dt * vn_free;
            nodes.push(NodeQ {
                normal,
                tangent,
                reach,
                dn,
                dt: dtan,
                vn_free,
                vt_free: v[0] * tangent[0] + v[1] * tangent[1] + dt * (dtan.dot(a) + beta(tangent)),
                vt_start: v[0] * tangent[0] + v[1] * tangent[1],
                target: sel(ge(gap, zero()), -gap * rate, -gap * sp(PUSH_OUT) * rate),
                mu: {
                    let mu = g.friction[i]
                        * sp(cfg.ground_friction)
                        * (sp(1.0) + sp(physics::MUD_GRIP) * sink)
                        * (sp(1.0) + sp(physics::MUD_NORMAL) * sink);
                    // Ice patches take a share of the friction.
                    let mu = if cfg.patches > 0.0 {
                        mu * (sp(1.0) - sp(cfg.patches) * ice(x))
                    } else {
                        mu
                    };
                    // Static friction: a foot that barely slides holds harder.
                    let slide = (v[0] * tangent[0] + v[1] * tangent[1]).abs();
                    mu * (sp(1.0)
                        + sp(0.25) * clamp((sp(0.02) - slide) * sp(100.0), zero(), sp(1.0)))
                },
                // `continue` when `gap + dt * vn_free > 0`.
                candidate: !reach.gt(zero()) & self.active,
            });
        }
        // The deepest `MAXC` nodes of each lane, in node order.
        let mut chosen = [[usize::MAX; MAXC]; L];
        let mut counts = [0usize; L];
        let reach_l: Vec<[f32; L]> = nodes.iter().map(|q| arr(q.reach)).collect();
        for l in 0..L {
            let mut list: Vec<usize> = (0..n)
                .filter(|&i| lane_bit(nodes[i].candidate, l))
                .collect();
            if list.len() > MAX_CONTACTS {
                list.sort_by(|&a, &b| reach_l[a][l].total_cmp(&reach_l[b][l]).then(a.cmp(&b)));
                list.truncate(MAX_CONTACTS);
                list.sort_unstable();
            }
            counts[l] = list.len();
            chosen[l][..list.len()].copy_from_slice(&list);
        }
        let mut lane_bits = 0u16;
        for (l, &c) in counts.iter().enumerate() {
            lane_bits |= u16::from(c > 0) << l;
        }
        let has = M(lane_bits);
        if !has.any() {
            for w in &mut self.s.warm {
                *w = [zero(); 2];
            }
            return has;
        }
        let mut slots = Slots {
            on: [M(0); MAXC],
            at: vec![[M(0); MAXC]; n],
            body: vec![[M(0); MAXC]; b],
            normal: [[zero(); 2]; MAXC],
            tangent: [[zero(); 2]; MAXC],
            dn: [V3::zero(); MAXC],
            dt: [V3::zero(); MAXC],
            vn_free: [zero(); MAXC],
            vt_free: [zero(); MAXC],
            vt_start: [zero(); MAXC],
            target: [zero(); MAXC],
            mu: [zero(); MAXC],
        };
        #[allow(clippy::needless_range_loop)]
        for c in 0..MAXC {
            let mut on = 0u16;
            for l in 0..L {
                if c < counts[l] {
                    let node = chosen[l][c];
                    on |= 1 << l;
                    let at = &mut slots.at[node][c];
                    *at = M(at.bits() | 1 << l);
                    let body = node.saturating_sub(1);
                    let bm = &mut slots.body[body][c];
                    *bm = M(bm.bits() | 1 << l);
                }
            }
            slots.on[c] = M(on);
            if on == 0 {
                continue;
            }
            let gather = |f: &dyn Fn(&NodeQ) -> F| -> F {
                let mut out = zero();
                for (i, q) in nodes.iter().enumerate() {
                    out = sel(slots.at[i][c], f(q), out);
                }
                out
            };
            slots.normal[c] = [gather(&|q| q.normal[0]), gather(&|q| q.normal[1])];
            slots.tangent[c] = [gather(&|q| q.tangent[0]), gather(&|q| q.tangent[1])];
            slots.dn[c] = V3::new(
                gather(&|q| q.dn.w),
                gather(&|q| q.dn.x),
                gather(&|q| q.dn.y),
            );
            slots.dt[c] = V3::new(
                gather(&|q| q.dt.w),
                gather(&|q| q.dt.x),
                gather(&|q| q.dt.y),
            );
            slots.vn_free[c] = gather(&|q| q.vn_free);
            slots.vt_free[c] = gather(&|q| q.vt_free);
            slots.vt_start[c] = gather(&|q| q.vt_start);
            slots.target[c] = gather(&|q| q.target);
            slots.mu[c] = gather(&|q| q.mu);
        }
        // Contact-space matrix: the velocity change along each row per newton
        // along each column, from the column's response.
        let mut k = [zero(); MAXR * MAXR];
        for col in 0..MAXR {
            let cs = col / 2;
            if !slots.on[cs].any() {
                continue;
            }
            let dir = if col % 2 == 0 {
                slots.dn[cs]
            } else {
                slots.dt[cs]
            };
            self.response(&slots, &[(cs, dir)]);
            for row in col..MAXR {
                let rs = row / 2;
                if !slots.on[rs].any() {
                    continue;
                }
                let rdir = if row % 2 == 0 {
                    slots.dn[rs]
                } else {
                    slots.dt[rs]
                };
                let mut v = zero();
                for bd in 0..b {
                    v = sel(slots.body[bd][rs], dt * rdir.dot(self.sc.da[bd]), v);
                }
                k[row * MAXR + col] = v;
                k[col * MAXR + row] = v;
            }
        }
        // Warm start from the nodes' contact forces of the last step.
        let mut lambda = [zero(); MAXR];
        for c in 0..MAXC {
            if !slots.on[c].any() {
                continue;
            }
            let wn = self.at_slot(&slots, c, |i| self.s.warm[i][0]);
            let wt = self.at_slot(&slots, c, |i| self.s.warm[i][1]);
            lambda[2 * c] = sel(slots.on[c], wn, zero());
            lambda[2 * c + 1] = sel(
                slots.on[c],
                clamp(wt, -slots.mu[c] * wn, slots.mu[c] * wn),
                zero(),
            );
        }
        let mut predicted = pgs(&slots, &k, &mut lambda, PGS_ITERATIONS);
        // The contact forces act on the bodies through one response.
        self.apply_contacts(&slots, &lambda, has);
        // Plant against the end pose.
        for _ in 0..PLANT_ROUNDS {
            let saved = (
                self.s.x0,
                self.s.v0,
                self.s.w0,
                self.s.th0,
                self.s.q.clone(),
                self.s.qd.clone(),
            );
            self.integrate(sp(1.0));
            let mut ground = [zero(); 2];
            for c in 0..MAXC {
                if !slots.on[c].any() {
                    continue;
                }
                let add = [
                    (lambda[2 * c] * slots.dn[c].x + lambda[2 * c + 1] * slots.dt[c].x) * dt,
                    (lambda[2 * c] * slots.dn[c].y + lambda[2 * c + 1] * slots.dt[c].y) * dt,
                ];
                ground = [
                    sel(slots.on[c], ground[0] + add[0], ground[0]),
                    sel(slots.on[c], ground[1] + add[1], ground[1]),
                ];
            }
            let after = self.momentum();
            let shift = [
                (before[0] + *impulse + ground[0] - after[0]) * g.inv_mass,
                (before[1] + *impulse_y + ground[1] - after[1]) * g.inv_mass,
            ];
            for c in 0..MAXC {
                if !slots.on[c].any() {
                    continue;
                }
                let vx = self.at_slot(&slots, c, |i| self.s.vel[i][0]) + shift[0];
                let vy = self.at_slot(&slots, c, |i| self.s.vel[i][1]) + shift[1];
                let normal = slots.normal[c];
                let tangent = slots.tangent[c];
                slots.vn_free[c] = sel(
                    slots.on[c],
                    slots.vn_free[c] + (vx * normal[0] + vy * normal[1] - predicted[2 * c]),
                    slots.vn_free[c],
                );
                slots.vt_free[c] = sel(
                    slots.on[c],
                    slots.vt_free[c] + (vx * tangent[0] + vy * tangent[1] - predicted[2 * c + 1]),
                    slots.vt_free[c],
                );
            }
            (self.s.x0, self.s.v0, self.s.w0, self.s.th0) = (saved.0, saved.1, saved.2, saved.3);
            self.s.q = saved.4;
            self.s.qd = saved.5;
            let old = lambda;
            let mut ends = pgs(&slots, &k, &mut lambda, PLANT_SWEEPS);
            clean_friction(&slots, &k, &mut lambda, &mut ends);
            let change: [F; MAXR] = std::array::from_fn(|r| lambda[r] - old[r]);
            self.apply_contacts(&slots, &change, has);
            predicted = ends;
        }
        // Each node's contact force starts the next step's solve.
        for i in 0..n {
            let mut wn = zero();
            let mut wt = zero();
            for c in 0..MAXC {
                wn = sel(slots.at[i][c], lambda[2 * c], wn);
                wt = sel(slots.at[i][c], lambda[2 * c + 1], wt);
            }
            self.s.warm[i] = [wn, wt];
        }
        for c in 0..MAXC {
            if !slots.on[c].any() {
                continue;
            }
            let on = slots.on[c];
            *impulse = sel(
                on,
                *impulse + (lambda[2 * c] * slots.dn[c].x + lambda[2 * c + 1] * slots.dt[c].x) * dt,
                *impulse,
            );
            *impulse_y = sel(
                on,
                *impulse_y
                    + (lambda[2 * c] * slots.dn[c].y + lambda[2 * c + 1] * slots.dt[c].y) * dt,
                *impulse_y,
            );
        }
        has
    }
}

/// Projected Gauss-Seidel on the contact impulses (`physics2::pgs`). Returns
/// the rows' velocities.
fn pgs(slots: &Slots, k: &[F; MAXR * MAXR], lambda: &mut [F; MAXR], sweeps: usize) -> [F; MAXR] {
    let mut v = [zero(); MAXR];
    for c in 0..MAXC {
        v[2 * c] = slots.vn_free[c];
        v[2 * c + 1] = slots.vt_free[c];
    }
    for row in 0..MAXR {
        if !slots.on[row / 2].any() {
            continue;
        }
        for j in 0..MAXR {
            let mj = slots.on[j / 2];
            if !mj.any() {
                continue;
            }
            v[row] = sel(mj, v[row] + k[row * MAXR + j] * lambda[j], v[row]);
        }
    }
    for _ in 0..sweeps {
        for c in 0..MAXC {
            let m = slots.on[c];
            if !m.any() {
                continue;
            }
            let (rn, rt) = (2 * c, 2 * c + 1);
            let normal = (lambda[rn] + (slots.target[c] - v[rn]) * (sp(1.0) / k[rn * MAXR + rn]))
                .max(zero());
            let change = normal - lambda[rn];
            lambda[rn] = sel(m, normal, lambda[rn]);
            for j in 0..MAXR {
                let mj = slots.on[j / 2] & m;
                if !mj.any() {
                    continue;
                }
                v[j] = sel(mj, v[j] + k[j * MAXR + rn] * change, v[j]);
            }
            let bound = slots.mu[c] * lambda[rn];
            let stiff = k[rt * MAXR + rt];
            let a = slots.vt_start[c] + v[rt] - stiff * lambda[rt];
            let reach = a.abs() / stiff;
            let pos = a.gt(zero());
            let limit = bound.min(reach);
            let low = sel(pos, -limit, zero());
            let high = sel(pos, zero(), limit);
            let friction = clamp(lambda[rt] - v[rt] * (sp(1.0) / stiff), low, high);
            let change = friction - lambda[rt];
            lambda[rt] = sel(m, friction, lambda[rt]);
            for j in 0..MAXR {
                let mj = slots.on[j / 2] & m;
                if !mj.any() {
                    continue;
                }
                v[j] = sel(mj, v[j] + k[j * MAXR + rt] * change, v[j]);
            }
        }
    }
    v
}

/// Removes friction that would do positive work (`physics2::clean_friction`).
fn clean_friction(slots: &Slots, k: &[F; MAXR * MAXR], lambda: &mut [F; MAXR], v: &mut [F; MAXR]) {
    for _ in 0..2 {
        for c in 0..MAXC {
            let m = slots.on[c];
            if !m.any() {
                continue;
            }
            let rt = 2 * c + 1;
            let stiff = k[rt * MAXR + rt];
            let a = slots.vt_start[c] + v[rt] - stiff * lambda[rt];
            let reach = a.abs() / stiff;
            let bound = (slots.mu[c] * lambda[2 * c]).min(reach);
            let pos = a.gt(zero());
            let low = sel(pos, -bound, zero());
            let high = sel(pos, zero(), bound);
            let friction = clamp(lambda[rt], low, high);
            let change = friction - lambda[rt];
            lambda[rt] = sel(m, friction, lambda[rt]);
            for j in 0..MAXR {
                let mj = slots.on[j / 2] & m;
                if !mj.any() {
                    continue;
                }
                v[j] = sel(mj, v[j] + k[j * MAXR + rt] * change, v[j]);
            }
        }
    }
}

/// Runs the trial of every lane of a group; lanes beyond `real` are padding.
fn run_group(g: &Group, cfg: &Config) -> [GpuResult; L] {
    let mut sim = Sim::new(g, cfg);
    let fidelity = cfg.fidelity();
    let rate = fidelity.rate as f32;
    let dt = 1.0 / rate;
    let steps = cfg.steps();
    let sample = fidelity.sample_interval();
    let screen = cfg
        .screen
        .map(|s| (((s.seconds * rate).round() as u32).max(1) - 1, s.bar));
    let (n, b) = (g.n, g.b);
    let mut metrics = [fresh_metrics(); L];
    let mut result = [GpuResult::default(); L];
    let mut contact_bits = [0u64; L];
    let mut lift_bits = [0u64; L];
    let mut ground_bits = [0u64; L];
    let mut done = [false; L];
    let mut head_shake = zero();
    for (l, d) in done.iter_mut().enumerate() {
        *d = l >= g.real;
    }
    let live = |done: &[bool; L]| {
        M(done
            .iter()
            .enumerate()
            .fold(0u16, |m, (l, &d)| m | u16::from(!d) << l))
    };
    sim.active = live(&done);
    let inv_nodes = 1.0 / n as f32;
    let shake_gain = (1.0 / (physics::HEAD_SHAKE_WINDOW * rate)).min(1.0);
    for step in 0..steps {
        if !sim.active.any() {
            break;
        }
        let time = step as f32 * dt;
        let head_before = sim.s.v0;
        sim.step(time);
        let th0 = sim.s.th0;
        sim.s.th0 = th0
            - sp(std::f32::consts::TAU)
                * ((th0 + sp(std::f32::consts::PI)) / sp(std::f32::consts::TAU)).floor();
        // Metrics, falls and the screen, as the reference does after each step.
        let mut failed = M(0);
        let mut center_y = zero();
        let mut low = sp(f32::INFINITY);
        let mut high = sp(f32::NEG_INFINITY);
        let mut touch = vec![0u16; n];
        let mut lift = vec![0u16; n];
        for i in 0..n {
            let [x, y] = sim.s.pos[i];
            failed = failed | !x.abs().le(sp(1e6)) | !y.abs().le(sp(1e6));
            let r = g.radius[i];
            center_y += y;
            low = low.min(y - r);
            high = high.max(y + r);
            if cfg.ground {
                let (h, slope) = sim.ground(x);
                let floor = h + r * (sp(1.0) + slope * slope).sqrt();
                let touching = y.le(floor + sp(crate::physics2::CONTACT_SLACK));
                touch[i] = touching.bits();
                lift[i] = (!touching & y.gt(floor + sp(LIFT_CLEARANCE))).bits();
            }
        }
        center_y = center_y * sp(inv_nodes);
        let com_x = sim.center_x();
        let mut down_any = false;
        let mut downs = [0u64; L];
        let mut now_bits = [0u64; L];
        let mut touching_count = [0.0f32; L];
        for l in 0..L {
            if done[l] {
                continue;
            }
            let mut now = 0u64;
            for i in 0..n {
                if (touch[i] >> l) & 1 == 1 {
                    contact_bits[l] |= 1 << i;
                    now |= 1 << i;
                } else if (lift[i] >> l) & 1 == 1 {
                    lift_bits[l] |= contact_bits[l] & (1 << i);
                }
            }
            touching_count[l] = now.count_ones() as f32;
            now_bits[l] = now;
            downs[l] = now & !ground_bits[l];
            ground_bits[l] = now;
            down_any |= downs[l] != 0;
        }
        // Touchdowns restart the rhythm of the muscles that sense them.
        if down_any && step > 0 {
            let next = time + dt;
            for (k, m) in g.muscles.iter().enumerate() {
                let mut offset = arr(sim.s.offset[k]);
                let (ip, ph, rs) = (arr(m.inv_period), arr(m.phase), arr(m.reset));
                let mut any = false;
                for l in 0..L {
                    let Some(node) = m.sensor[l] else { continue };
                    if !done[l] && downs[l] & (1 << node) != 0 {
                        let clock = next * ip[l] + ph[l];
                        offset[l] = (rs[l] - clock).rem_euclid(1.0);
                        any = true;
                    }
                }
                if any {
                    sim.s.offset[k] = vec_of(offset);
                }
            }
        }
        // Head shaking, averaged over about HEAD_SHAKE_WINDOW.
        if time >= physics::HEAD_SHAKE_WINDOW {
            let dv = [sim.s.v0[0] - head_before[0], sim.s.v0[1] - head_before[1]];
            let accel = (dv[0] * dv[0] + dv[1] * dv[1]).sqrt() * sp(rate);
            head_shake += (accel - head_shake) * sp(shake_gain);
        }
        let mut broken = M(0);
        for j in 1..b {
            if g.parent[j].is_some() {
                broken = broken
                    | sim.s.q[j].lt(g.lo[j] - sp(physics::JOINT_BREAK))
                    | sim.s.q[j].gt(g.hi[j] + sp(physics::JOINT_BREAK));
            }
        }
        let fell = sim.s.pos[0][1].lt(sim.s.pos[1][1])
            | broken
            | head_shake.gt(sp(physics::HEAD_SHAKE_LIMIT))
            | failed;
        let (com, cy, hs, lo_a, hi_a) = (
            arr(com_x),
            arr(center_y),
            arr(head_shake),
            arr(low),
            arr(high),
        );
        for l in 0..L {
            if done[l] {
                continue;
            }
            let m = &mut metrics[l];
            m.contact_lo = f32::from_bits(contact_bits[l] as u32);
            m.contact_hi = f32::from_bits((contact_bits[l] >> 32) as u32);
            m.lift_lo = f32::from_bits(lift_bits[l] as u32);
            m.lift_hi = f32::from_bits((lift_bits[l] >> 32) as u32);
            m.head_shake = hs[l];
            let mut ended = false;
            if lane_bit(fell, l) {
                m.fall_time = time + dt;
                m.fitness = if lane_bit(failed, l) {
                    crate::evolution::FAILED
                } else {
                    com[l]
                };
                if screen.is_some_and(|(tick, _)| step <= tick) {
                    m.screen_x = m.fitness;
                }
                ended = true;
            }
            m.ground_contact += touching_count[l];
            m.height_sum += hi_a[l] - lo_a[l];
            m.vertical_oscillation = m.vertical_oscillation.min(cy[l]);
            m.gait_frequency = m.gait_frequency.max(cy[l]);
            if step == 0 {
                m.previous_center_y = cy[l];
                m.vertical_extremum = cy[l];
                m.vertical_trend = 0.0;
                m.gait_turns = 0.0;
            } else if step % sample == 0 {
                gait_sample(m, cy[l]);
            }
            if let Some((tick, bar)) = screen
                && step == tick
                && !ended
            {
                m.screen_x = com[l];
                if com[l] < bar {
                    m.screened = time + dt;
                    m.fitness = com[l];
                    ended = true;
                }
            }
            if ended {
                m.vertical_oscillation = (m.gait_frequency - m.vertical_oscillation).max(0.0);
                m.gait_frequency = m.gait_turns * 0.5 / (time + dt);
                result[l] = *m;
                done[l] = true;
            }
        }
        sim.active = live(&done);
    }
    let com = arr(sim.center_x());
    for l in 0..L {
        if done[l] {
            continue;
        }
        let m = &mut metrics[l];
        m.fitness = com[l];
        m.vertical_oscillation = (m.gait_frequency - m.vertical_oscillation).max(0.0);
        m.gait_frequency = m.gait_turns * 0.5 / (steps as f32 * dt).max(dt);
        result[l] = *m;
    }
    result
}

/// The skeleton of creature `i` of `pop`, as a hashable key: creatures with
/// equal keys have equal bone parents and pivots. Muscle attachments differ
/// per lane.
fn plan_key(pop: &Population, i: usize) -> Vec<u32> {
    let g = &pop.genomes[i];
    let mut key = Vec::with_capacity(2 + 2 * g.bone_count);
    key.push(g.node_count as u32);
    for bone in &pop.bones[g.bone_start..g.bone_start + g.bone_count] {
        key.extend([bone.a, bone.b]);
    }
    key
}

/// Scores every creature of `pop` with the v2 physics, 16 at a time.
pub fn evaluate(pop: &Population, cfg: &Config) -> Vec<GpuResult> {
    let total = pop.genomes.len();
    let mut plans: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
    for i in 0..total {
        plans.entry(plan_key(pop, i)).or_default().push(i);
    }
    // Within a skeleton, creatures with similar muscle counts share a group,
    // so few lanes wait for the longest muscle list.
    let chunks: Vec<Vec<usize>> = plans
        .into_values()
        .flat_map(|mut members| {
            members.sort_by_key(|&i| (pop.genomes[i].muscle_count, i));
            members.chunks(L).map(<[usize]>::to_vec).collect::<Vec<_>>()
        })
        .collect();
    let done: Vec<(Vec<usize>, [GpuResult; L])> = chunks
        .into_par_iter()
        .map(|members| {
            let models: Vec<Model> = members
                .iter()
                .map(|&i| Model::new(&pop.creature(i), cfg))
                .collect();
            let states: Vec<State> = models.iter().map(|m| m.start(cfg)).collect();
            let refs: Vec<&Model> = models.iter().collect();
            debug_assert!(
                refs.iter()
                    .all(|m| m.pivot == refs[0].pivot && m.parent == refs[0].parent)
            );
            let group = Group::build(&refs, &states);
            let results = run_group(&group, cfg);
            (members, results)
        })
        .collect();
    let mut out = vec![GpuResult::default(); total];
    for (members, results) in done {
        for (l, &i) in members.iter().enumerate() {
            out[i] = results[l];
        }
    }
    out
}
