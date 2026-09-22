use crate::config::Config;
use anyhow::{Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const FAILED: f32 = -1.0e20;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct NodeGene {
    pub x: f32,
    pub y: f32,
    pub diameter: f32,
    pub friction: f32,
}
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Serialize, Deserialize, PartialEq, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct Muscle {
    pub a: u32,
    pub b: u32,
    pub short: f32,
    pub long: f32,
    pub period: f32,
    pub phase: f32,
    pub duty: f32,
    pub stiffness: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub node_start: usize,
    pub node_count: usize,
    pub muscle_start: usize,
    pub muscle_count: usize,
    pub id: u64,
    pub mutability: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Creature {
    pub nodes: Vec<NodeGene>,
    pub muscles: Vec<Muscle>,
    pub id: u64,
    pub mutability: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Population {
    pub genomes: Vec<Genome>,
    pub nodes: Vec<NodeGene>,
    pub muscles: Vec<Muscle>,
}

/// Each creature has its own deterministic stream: thread scheduling cannot change evolution.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64, generation: u32, index: usize) -> Self {
        Self(
            seed ^ (generation as u64).wrapping_mul(0xd1342543de82ef95)
                ^ (index as u64).wrapping_mul(0x9e3779b97f4a7c15),
        )
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + self.unit() * (b - a)
    }
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn delta(&mut self) -> f32 {
        self.range(-1.0, 1.0).powi(7)
    }
}
impl Population {
    pub fn creature(&self, index: usize) -> Creature {
        let g = &self.genomes[index];
        Creature {
            nodes: self.nodes[g.node_start..g.node_start + g.node_count].to_vec(),
            muscles: self.muscles[g.muscle_start..g.muscle_start + g.muscle_count].to_vec(),
            id: g.id,
            mutability: g.mutability,
        }
    }
    pub fn push(&mut self, c: Creature) {
        self.genomes.push(Genome {
            node_start: self.nodes.len(),
            node_count: c.nodes.len(),
            muscle_start: self.muscles.len(),
            muscle_count: c.muscles.len(),
            id: c.id,
            mutability: c.mutability,
        });
        self.nodes.extend(c.nodes);
        self.muscles.extend(c.muscles);
    }
    pub fn bytes(&self) -> usize {
        self.genomes.capacity() * std::mem::size_of::<Genome>()
            + self.nodes.capacity() * std::mem::size_of::<NodeGene>()
            + self.muscles.capacity() * std::mem::size_of::<Muscle>()
    }
    pub fn validate(&self, cfg: &Config) -> Result<()> {
        ensure!(
            self.genomes.len() == cfg.population,
            "Checkpoint population does not match settings"
        );
        for g in &self.genomes {
            ensure!(
                (3..=cfg.max_nodes).contains(&g.node_count) && g.muscle_count <= cfg.max_muscles,
                "Invalid body size"
            );
            ensure!(
                g.node_start
                    .checked_add(g.node_count)
                    .is_some_and(|x| x <= self.nodes.len())
                    && g.muscle_start
                        .checked_add(g.muscle_count)
                        .is_some_and(|x| x <= self.muscles.len()),
                "Invalid genome offset"
            );
            ensure!(
                g.mutability.is_finite() && (0.0..=2.0).contains(&g.mutability),
                "Invalid mutability"
            );
            for n in &self.nodes[g.node_start..g.node_start + g.node_count] {
                ensure!(
                    [n.x, n.y, n.diameter, n.friction]
                        .iter()
                        .all(|x| x.is_finite())
                        && (0.01..=1.0).contains(&n.diameter)
                        && (0.0..=1.0).contains(&n.friction),
                    "Invalid node"
                );
            }
            let edges = &self.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
            let mut adjacency = [0u64; 64];
            for (i, m) in edges.iter().enumerate() {
                ensure!(
                    m.a != m.b
                        && (m.a as usize) < g.node_count
                        && (m.b as usize) < g.node_count
                        && [m.short, m.long, m.period, m.phase, m.duty, m.stiffness]
                            .iter()
                            .all(|x| x.is_finite())
                        && m.short >= 0.01
                        && m.long >= m.short
                        && m.period >= 0.1
                        && (0.05..=0.95).contains(&m.duty)
                        && (1.0..=120.0).contains(&m.stiffness),
                    "Invalid muscle"
                );
                ensure!(
                    !edges[..i]
                        .iter()
                        .any(|p| (p.a == m.a && p.b == m.b) || (p.a == m.b && p.b == m.a)),
                    "Duplicate muscle"
                );
                adjacency[m.a as usize] |= 1u64 << m.b;
                adjacency[m.b as usize] |= 1u64 << m.a;
            }
            ensure!(
                adjacency[..g.node_count]
                    .iter()
                    .all(|n| n.count_ones() >= 2),
                "Isolated or under-connected node"
            );
            let mut reached = 1u64;
            loop {
                let previous = reached;
                for (i, neighbors) in adjacency[..g.node_count].iter().enumerate() {
                    if reached & (1u64 << i) != 0 {
                        reached |= neighbors;
                    }
                }
                if reached == previous {
                    break;
                }
            }
            ensure!(
                reached.count_ones() as usize == g.node_count,
                "Disconnected creature"
            );
        }
        Ok(())
    }
}
fn muscle(a: usize, b: usize, nodes: &[NodeGene], rng: &mut Rng) -> Muscle {
    let length = ((nodes[a].x - nodes[b].x).powi(2) + (nodes[a].y - nodes[b].y).powi(2))
        .sqrt()
        .clamp(0.06, 0.6);
    Muscle {
        a: a as u32,
        b: b as u32,
        short: length * rng.range(0.65, 0.95),
        long: length * rng.range(1.05, 1.35),
        period: rng.range(0.65, 2.6),
        phase: rng.unit(),
        duty: rng.range(0.25, 0.75),
        stiffness: rng.range(20.0, 80.0),
    }
}
fn repair(c: &mut Creature, cfg: &Config, rng: &mut Rng) {
    let count = c.nodes.len();
    let mut seen = [0u64; 64];
    c.muscles.retain(|m| {
        let a = m.a as usize;
        let b = m.b as usize;
        if a >= count || b >= count || a == b || seen[a] & (1u64 << b) != 0 {
            false
        } else {
            seen[a] |= 1u64 << b;
            seen[b] |= 1u64 << a;
            true
        }
    });
    // A ring guarantees connectivity and at least two incident muscles per node.
    for a in 0..count {
        let b = (a + 1) % count;
        if seen[a] & (1u64 << b) == 0 {
            if c.muscles.len() >= cfg.max_muscles
                && let Some(i) = c.muscles.iter().position(|m| {
                    let a = m.a as usize;
                    let b = m.b as usize;
                    (a + 1) % count != b && (b + 1) % count != a
                })
            {
                c.muscles.swap_remove(i);
            }
            c.muscles.push(muscle(a, b, &c.nodes, rng));
            seen[a] |= 1u64 << b;
            seen[b] |= 1u64 << a;
        }
    }
}
fn initial(cfg: &Config, index: usize) -> Creature {
    let mut rng = Rng::new(cfg.seed, 0, index);
    let n = (3 + rng.index(3)).min(cfg.max_nodes);
    let mut c = Creature {
        nodes: (0..n)
            .map(|_| NodeGene {
                x: rng.range(-0.2, 0.2),
                y: rng.range(0.0, 0.4),
                diameter: rng.range(cfg.min_size, cfg.max_size),
                friction: rng.range(cfg.min_friction, cfg.max_friction),
            })
            .collect(),
        muscles: vec![],
        id: index as u64 + 1,
        mutability: 1.0,
    };
    repair(&mut c, cfg, &mut rng);
    for _ in 0..rng.index(n) {
        if c.muscles.len() < cfg.max_muscles {
            let a = rng.index(n);
            let b = rng.index(n);
            if a != b
                && !c.muscles.iter().any(|m| {
                    (m.a == a as u32 && m.b == b as u32) || (m.b == a as u32 && m.a == b as u32)
                })
            {
                c.muscles.push(muscle(a, b, &c.nodes, &mut rng));
            }
        }
    }
    c
}
fn collect_parallel(count: usize, make: impl Fn(usize) -> Creature + Sync) -> Population {
    // Bounded temporary arenas, not a Vec<Creature> with millions of allocations retained.
    let chunks: Vec<Population> = (0..count.div_ceil(4096))
        .into_par_iter()
        .map(|chunk| {
            let mut p = Population::default();
            for i in chunk * 4096..((chunk + 1) * 4096).min(count) {
                p.push(make(i));
            }
            p
        })
        .collect();
    let mut out = Population {
        genomes: Vec::with_capacity(count),
        nodes: Vec::with_capacity(chunks.iter().map(|p| p.nodes.len()).sum()),
        muscles: Vec::with_capacity(chunks.iter().map(|p| p.muscles.len()).sum()),
    };
    for mut chunk in chunks {
        let ns = out.nodes.len();
        let ms = out.muscles.len();
        for g in &mut chunk.genomes {
            g.node_start += ns;
            g.muscle_start += ms;
        }
        out.genomes.extend(chunk.genomes);
        out.nodes.extend(chunk.nodes);
        out.muscles.extend(chunk.muscles);
    }
    out
}
pub fn create(cfg: &Config) -> Result<Population> {
    cfg.validate()?;
    Ok(collect_parallel(cfg.population, |i| initial(cfg, i)))
}
fn mutate(mut c: Creature, cfg: &Config, generation: u32, index: usize) -> Creature {
    let mut rng = Rng::new(cfg.seed, generation, index);
    let strength = cfg.mutation * c.mutability;
    if strength == 0.0 {
        return c;
    }
    for n in &mut c.nodes {
        n.x += rng.delta() * 0.1 * strength;
        n.y += rng.delta() * 0.1 * strength;
        n.diameter = (n.diameter + rng.delta() * 0.02 * strength).clamp(cfg.min_size, cfg.max_size);
        n.friction =
            (n.friction + rng.delta() * 0.1 * strength).clamp(cfg.min_friction, cfg.max_friction);
    }
    for m in &mut c.muscles {
        m.short = (m.short + rng.delta() * 0.1 * strength).clamp(0.02, 0.8);
        m.long = (m.long + rng.delta() * 0.1 * strength).clamp(m.short, 1.0);
        m.period = (m.period + rng.delta() * 0.2 * strength).clamp(0.1, 10.0);
        m.phase = (m.phase + rng.delta() * 0.2 * strength).rem_euclid(1.0);
        m.duty = (m.duty + rng.delta() * 0.1 * strength).clamp(0.05, 0.95);
        m.stiffness = (m.stiffness * (1.0 + rng.delta() * 0.3 * strength)).clamp(1.0, 120.0);
        if rng.unit() < 0.02 * strength {
            m.a = rng.index(c.nodes.len()) as u32;
        }
        if rng.unit() < 0.02 * strength {
            m.b = rng.index(c.nodes.len()) as u32;
        }
    }
    if rng.unit() < 0.04 * strength && c.nodes.len() < cfg.max_nodes {
        let parent = c.nodes[rng.index(c.nodes.len())];
        c.nodes.push(NodeGene {
            x: parent.x + rng.range(-0.1, 0.1),
            y: parent.y + rng.range(-0.1, 0.1),
            diameter: rng.range(cfg.min_size, cfg.max_size),
            friction: rng.range(cfg.min_friction, cfg.max_friction),
        });
    }
    if rng.unit() < 0.04 * strength && c.nodes.len() > 3 {
        let i = rng.index(c.nodes.len());
        c.nodes.remove(i);
        c.muscles.retain(|m| m.a as usize != i && m.b as usize != i);
        for m in &mut c.muscles {
            if m.a as usize > i {
                m.a -= 1;
            }
            if m.b as usize > i {
                m.b -= 1;
            }
        }
    }
    if rng.unit() < 0.04 * strength && c.muscles.len() < cfg.max_muscles {
        let a = rng.index(c.nodes.len());
        let b = rng.index(c.nodes.len());
        if a != b {
            c.muscles.push(muscle(a, b, &c.nodes, &mut rng));
        }
    }
    if rng.unit() < 0.04 * strength && c.muscles.len() > 3 {
        let i = rng.index(c.muscles.len());
        c.muscles.swap_remove(i);
    }
    repair(&mut c, cfg, &mut rng);
    c.mutability = (c.mutability * rng.range(0.8, 1.25)).clamp(0.05, 2.0);
    c
}
pub fn ranking(scores: &[f32]) -> Vec<usize> {
    let mut ranks: Vec<_> = (0..scores.len()).collect();
    ranks.par_sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ranks
}
pub fn survivors(cfg: &Config, generation: u32, ranks: &[usize]) -> Vec<usize> {
    (0..ranks.len() / 2)
        .map(|j| {
            let mut rng = Rng::new(cfg.seed, generation + 1, j);
            let chance = (rng.range(-1.0, 1.0).powi(3) + 1.0) * 0.5;
            if j as f32 / ranks.len() as f32 <= chance {
                ranks[j]
            } else {
                ranks[ranks.len() - 1 - j]
            }
        })
        .collect()
}
pub fn reproduce(
    pop: &Population,
    cfg: &Config,
    generation: u32,
    parents: &[usize],
) -> Result<Population> {
    // Worst-case reserve includes old/new arenas, parallel assembly, fitness/rank tables, and I/O.
    let growth = cfg.population.saturating_mul(96);
    ensure!(
        pop.bytes().saturating_mul(4).saturating_add(growth) < cfg.ram_budget_mib * 1024 * 1024,
        "Evolution would exceed the RAM budget; save and raise the budget before continuing"
    );
    Ok(collect_parallel(cfg.population, |i| {
        let parent = parents[i / 2];
        let mut c = pop.creature(parent);
        if i % 2 == 1 {
            c = mutate(c, cfg, generation + 1, i);
        }
        c.id = (generation as u64 + 1) * cfg.population as u64 + i as u64 + 1;
        c
    }))
}
