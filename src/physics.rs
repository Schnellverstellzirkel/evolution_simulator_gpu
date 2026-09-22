use crate::{
    config::Config,
    evolution::{Creature, FAILED, Muscle, NodeGene},
};
pub const DT: f32 = 1.0 / 120.0;
pub const SETTLE: u32 = 200;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub radius: f32,
    pub friction: f32,
    pub mass: f32,
    pub failed: f32,
}
#[inline]
pub fn node(gene: &NodeGene) -> Node {
    Node {
        pos: [gene.x, gene.y],
        vel: [0.0; 2],
        radius: gene.diameter * 0.5,
        friction: gene.friction,
        mass: (0.1 * (gene.diameter / 0.08).powi(2)).clamp(0.02, 10.0),
        failed: 0.0,
    }
}
pub fn nodes(c: &Creature) -> Vec<Node> {
    c.nodes.iter().map(node).collect()
}
pub fn target(m: &Muscle, time: f32) -> f32 {
    let phase = (time / m.period + m.phase).fract();
    let wave = if phase < m.duty {
        0.5 + 0.5 * (std::f32::consts::PI * phase / m.duty).cos()
    } else {
        0.5 - 0.5 * (std::f32::consts::PI * (phase - m.duty) / (1.0 - m.duty)).cos()
    };
    m.short + (m.long - m.short) * wave
}
pub fn center(nodes: &mut [Node]) {
    let x = nodes.iter().map(|n| n.pos[0]).sum::<f32>() / nodes.len() as f32;
    let low = nodes
        .iter()
        .map(|n| n.pos[1] - n.radius)
        .fold(f32::INFINITY, f32::min);
    for n in nodes {
        n.pos[0] -= x;
        n.pos[1] -= low;
        n.vel = [0.0; 2];
    }
}
fn contact(n: &mut Node, normal: [f32; 2], penetration: f32, friction: f32) {
    n.pos[0] += normal[0] * penetration;
    n.pos[1] += normal[1] * penetration;
    let vn = n.vel[0] * normal[0] + n.vel[1] * normal[1];
    if vn < 0.0 {
        n.vel[0] -= vn * normal[0];
        n.vel[1] -= vn * normal[1];
        let speed = n.vel[0].hypot(n.vel[1]);
        let keep = (1.0 - (-vn) * friction / speed.max(1e-8)).max(0.0);
        n.vel[0] *= keep;
        n.vel[1] *= keep;
    }
}
pub fn collide(n: &mut Node, cfg: &Config) {
    let mu = n.friction * cfg.ground_friction;
    if cfg.ground && n.pos[1] < n.radius {
        contact(n, [0.0, 1.0], n.radius - n.pos[1], mu);
    }
    for r in &cfg.obstacles {
        let q = [n.pos[0].clamp(r[0], r[2]), n.pos[1].clamp(r[1], r[3])];
        let d = [n.pos[0] - q[0], n.pos[1] - q[1]];
        let distance = d[0].hypot(d[1]);
        if distance > 1e-8 && distance < n.radius {
            contact(
                n,
                [d[0] / distance, d[1] / distance],
                n.radius - distance,
                mu,
            );
        } else if distance <= 1e-8 {
            let ds = [
                n.pos[0] - r[0],
                r[2] - n.pos[0],
                n.pos[1] - r[1],
                r[3] - n.pos[1],
            ];
            let mut side = 0;
            for k in 1..4 {
                if ds[k] < ds[side] {
                    side = k;
                }
            }
            contact(
                n,
                [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]][side],
                n.radius + ds[side],
                mu,
            );
        }
    }
}
pub fn step(nodes: &mut [Node], muscles: &[Muscle], cfg: &Config, tick: u32) {
    if tick == SETTLE {
        center(nodes);
    }
    let mut old = [Node::default(); 64];
    old[..nodes.len()].copy_from_slice(nodes);
    let time = tick.saturating_sub(SETTLE) as f32 * DT;
    for (i, n) in nodes.iter_mut().enumerate() {
        let mut f = [0.0; 2];
        for m in muscles {
            let other = if m.a as usize == i {
                m.b as usize
            } else if m.b as usize == i {
                m.a as usize
            } else {
                continue;
            };
            let d = [
                old[other].pos[0] - old[i].pos[0],
                old[other].pos[1] - old[i].pos[1],
            ];
            let distance = d[0].hypot(d[1]).max(1e-6);
            let dir = [d[0] / distance, d[1] / distance];
            let relative = (old[other].vel[0] - old[i].vel[0]) * dir[0]
                + (old[other].vel[1] - old[i].vel[1]) * dir[1];
            let force = ((distance - target(m, time)).clamp(-0.25, 0.25) * m.stiffness
                + relative * 0.15)
                .clamp(-30.0, 30.0);
            f[0] += dir[0] * force;
            f[1] += dir[1] * force;
        }
        n.vel[0] = (n.vel[0] + f[0] / n.mass * DT) * cfg.air_retention.sqrt();
        n.vel[1] = (n.vel[1]
            + (f[1] / n.mass - if tick >= SETTLE { cfg.gravity } else { 0.0 }) * DT)
            * cfg.air_retention.sqrt();
        n.pos[0] += n.vel[0] * DT;
        n.pos[1] += n.vel[1] * DT;
        if tick >= SETTLE {
            collide(n, cfg);
        }
        if !n
            .pos
            .iter()
            .chain(n.vel.iter())
            .all(|v| v.is_finite() && v.abs() < 1e6)
        {
            n.failed = 1.0;
            n.pos = [0.0; 2];
            n.vel = [0.0; 2];
        }
    }
}
pub fn evaluate(c: &Creature, cfg: &Config) -> f32 {
    let mut n = nodes(c);
    for tick in 0..SETTLE + cfg.steps() {
        step(&mut n, &c.muscles, cfg, tick);
    }
    fitness(&n)
}
pub fn fitness(n: &[Node]) -> f32 {
    if n.iter().any(|n| n.failed != 0.0) {
        FAILED
    } else {
        n.iter().map(|n| n.pos[0]).sum::<f32>() / n.len() as f32
    }
}
