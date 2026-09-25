use crate::config::Config;
use crate::qd::{self, CmaEmitter, Emitter, QdArchive};
use anyhow::{Result, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const FAILED: f32 = -1.0e20;
pub const MAX_BONE_LENGTH: f32 = 2.0;
pub const MIN_MUSCLE_PERIOD: f32 = 0.5;

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
pub struct Bone {
    pub a: u32,
    pub b: u32,
    pub rest_length: f32,
}
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Serialize, Deserialize, PartialEq, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct Muscle {
    pub bone_a: u32,
    pub bone_b: u32,
    /// Attachment positions measured from each bone's `a` endpoint.
    pub anchor_a: f32,
    pub anchor_b: f32,
    pub short: f32,
    pub long: f32,
    pub period: f32,
    pub phase: f32,
    pub duty: f32,
    pub stiffness: f32,
    /// Which of the four attachment endpoints (bone_a.a, bone_a.b, bone_b.a,
    /// bone_b.b) senses touchdowns, or `NO_SENSOR`.
    #[serde(default = "no_sensor")]
    pub sensor: u32,
    /// Rhythm phase the muscle jumps to when its sensor touches down.
    #[serde(default)]
    pub reset: f32,
}
/// A muscle without a touchdown sensor.
pub const NO_SENSOR: u32 = 255;
fn no_sensor() -> u32 {
    NO_SENSOR
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct LegacyMuscle {
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
    pub bone_start: usize,
    pub bone_count: usize,
    pub muscle_start: usize,
    pub muscle_count: usize,
    pub id: u64,
    pub mutability: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Creature {
    pub nodes: Vec<NodeGene>,
    pub bones: Vec<Bone>,
    pub muscles: Vec<Muscle>,
    pub id: u64,
    pub mutability: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Population {
    pub genomes: Vec<Genome>,
    pub nodes: Vec<NodeGene>,
    pub bones: Vec<Bone>,
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
            bones: self.bones[g.bone_start..g.bone_start + g.bone_count].to_vec(),
            muscles: self.muscles[g.muscle_start..g.muscle_start + g.muscle_count].to_vec(),
            id: g.id,
            mutability: g.mutability,
        }
    }
    pub fn push(&mut self, c: Creature) {
        let mut c = c;
        canonicalize_bone_order(&mut c);
        self.genomes.push(Genome {
            node_start: self.nodes.len(),
            node_count: c.nodes.len(),
            bone_start: self.bones.len(),
            bone_count: c.bones.len(),
            muscle_start: self.muscles.len(),
            muscle_count: c.muscles.len(),
            id: c.id,
            mutability: c.mutability,
        });
        self.nodes.extend(c.nodes);
        self.bones.extend(c.bones);
        self.muscles.extend(c.muscles);
    }
    pub(crate) fn canonicalize_bones(&mut self) -> Result<()> {
        for index in 0..self.genomes.len() {
            let genome = &self.genomes[index];
            let node_end = genome.node_start.checked_add(genome.node_count);
            let bone_end = genome.bone_start.checked_add(genome.bone_count);
            let muscle_end = genome.muscle_start.checked_add(genome.muscle_count);
            ensure!(
                node_end.is_some_and(|end| end <= self.nodes.len())
                    && bone_end.is_some_and(|end| end <= self.bones.len())
                    && muscle_end.is_some_and(|end| end <= self.muscles.len()),
                "Invalid genome offset"
            );
            let mut creature = self.creature(index);
            if canonicalize_bone_order(&mut creature) {
                let genome = &self.genomes[index];
                self.bones[genome.bone_start..genome.bone_start + genome.bone_count]
                    .copy_from_slice(&creature.bones);
                self.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count]
                    .copy_from_slice(&creature.muscles);
            }
        }
        Ok(())
    }
    pub(crate) fn migrate_actuator_geometry(&mut self, cfg: &Config) {
        let mut migrated = Population::default();
        for index in 0..self.genomes.len() {
            let mut creature = self.creature(index);
            let mut rng = Rng::new(cfg.seed, 0, index);
            repair(&mut creature, cfg, &mut rng);
            migrated.push(creature);
        }
        *self = migrated;
    }
    /// Puts `c` into population slot `slot`. Its genes are appended to the
    /// arenas; `compact` later drops the replaced genes.
    pub fn replace(&mut self, slot: usize, c: Creature) {
        let mut c = c;
        canonicalize_bone_order(&mut c);
        self.genomes[slot] = Genome {
            node_start: self.nodes.len(),
            node_count: c.nodes.len(),
            bone_start: self.bones.len(),
            bone_count: c.bones.len(),
            muscle_start: self.muscles.len(),
            muscle_count: c.muscles.len(),
            id: c.id,
            mutability: c.mutability,
        };
        self.nodes.extend(c.nodes);
        self.bones.extend(c.bones);
        self.muscles.extend(c.muscles);
    }
    /// Rebuilds the arenas without genes of replaced creatures.
    pub fn compact(&mut self) {
        let all: Vec<usize> = (0..self.genomes.len()).collect();
        *self = self.subset(&all);
    }
    /// Copies `indices` into a standalone population; creature `k` of the
    /// result is `indices[k]` of `self`.
    pub fn subset(&self, indices: &[usize]) -> Population {
        let parts: Vec<Population> = indices
            .par_chunks(4096)
            .map(|chunk| {
                let mut part = Population {
                    genomes: Vec::with_capacity(chunk.len()),
                    ..Default::default()
                };
                for &i in chunk {
                    let g = &self.genomes[i];
                    part.genomes.push(Genome {
                        node_start: part.nodes.len(),
                        bone_start: part.bones.len(),
                        muscle_start: part.muscles.len(),
                        ..g.clone()
                    });
                    part.nodes
                        .extend_from_slice(&self.nodes[g.node_start..g.node_start + g.node_count]);
                    part.bones
                        .extend_from_slice(&self.bones[g.bone_start..g.bone_start + g.bone_count]);
                    part.muscles.extend_from_slice(
                        &self.muscles[g.muscle_start..g.muscle_start + g.muscle_count],
                    );
                }
                part
            })
            .collect();
        let mut out = Population {
            genomes: Vec::with_capacity(indices.len()),
            nodes: Vec::with_capacity(parts.iter().map(|p| p.nodes.len()).sum()),
            bones: Vec::with_capacity(parts.iter().map(|p| p.bones.len()).sum()),
            muscles: Vec::with_capacity(parts.iter().map(|p| p.muscles.len()).sum()),
        };
        for mut part in parts {
            let (ns, bs, ms) = (out.nodes.len(), out.bones.len(), out.muscles.len());
            for g in &mut part.genomes {
                g.node_start += ns;
                g.bone_start += bs;
                g.muscle_start += ms;
            }
            out.genomes.extend(part.genomes);
            out.nodes.extend(part.nodes);
            out.bones.extend(part.bones);
            out.muscles.extend(part.muscles);
        }
        out
    }
    pub fn bytes(&self) -> usize {
        self.genomes.capacity() * std::mem::size_of::<Genome>()
            + self.nodes.capacity() * std::mem::size_of::<NodeGene>()
            + self.bones.capacity() * std::mem::size_of::<Bone>()
            + self.muscles.capacity() * std::mem::size_of::<Muscle>()
    }
    pub fn validate(&self, cfg: &Config) -> Result<()> {
        self.validate_with_max_bone(cfg, MAX_BONE_LENGTH, false)
    }
    pub(crate) fn validate_with_max_bone(
        &self,
        cfg: &Config,
        max_bone_length: f32,
        historical: bool,
    ) -> Result<()> {
        ensure!(
            self.genomes.len() == cfg.population,
            "Checkpoint population does not match settings"
        );
        for (genome_index, g) in self.genomes.iter().enumerate() {
            ensure!(
                (3..=cfg.max_nodes).contains(&g.node_count)
                    && g.bone_count == g.node_count - 1
                    && g.muscle_count <= cfg.max_muscles,
                "Invalid body size"
            );
            ensure!(
                g.node_start
                    .checked_add(g.node_count)
                    .is_some_and(|x| x <= self.nodes.len())
                    && g.bone_start
                        .checked_add(g.bone_count)
                        .is_some_and(|x| x <= self.bones.len())
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
                        && (0.0..=1.0).contains(&n.friction)
                        && (historical || (cfg.min_size..=cfg.max_size).contains(&n.diameter))
                        && (historical
                            || (cfg.min_friction..=cfg.max_friction).contains(&n.friction)),
                    "Invalid node"
                );
            }
            let bones = &self.bones[g.bone_start..g.bone_start + g.bone_count];
            let mut bone_adjacency = [0u64; 64];
            for (i, bone) in bones.iter().enumerate() {
                ensure!(
                    bone.a != bone.b
                        && (bone.a as usize) < g.node_count
                        && (bone.b as usize) < g.node_count
                        && bone.rest_length.is_finite()
                        && (0.03..=max_bone_length).contains(&bone.rest_length),
                    "Invalid bone in genome {genome_index}: {bone:?}"
                );
                ensure!(
                    !bones[..i]
                        .iter()
                        .any(|p| (p.a == bone.a && p.b == bone.b)
                            || (p.a == bone.b && p.b == bone.a)),
                    "Duplicate bone"
                );
                bone_adjacency[bone.a as usize] |= 1u64 << bone.b;
                bone_adjacency[bone.b as usize] |= 1u64 << bone.a;
            }
            let mut ordered_nodes = 1u64;
            for bone in bones {
                let parent = 1u64 << bone.a;
                let child = 1u64 << bone.b;
                ensure!(
                    ordered_nodes & parent != 0 && ordered_nodes & child == 0,
                    "Bone constraints are not in parent-first order"
                );
                ordered_nodes |= child;
            }
            ensure!(
                bone_adjacency[..g.node_count].iter().all(|n| *n != 0),
                "Bone skeleton has an unconnected node"
            );
            let mut reached = 1u64;
            loop {
                let previous = reached;
                for (i, neighbors) in bone_adjacency[..g.node_count].iter().enumerate() {
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
                "Disconnected bone skeleton"
            );
            let muscles = &self.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
            let mut muscle_adjacency = [0u64; 64];
            for m in muscles {
                ensure!(
                    m.bone_a != m.bone_b
                        && (m.bone_a as usize) < g.bone_count
                        && (m.bone_b as usize) < g.bone_count
                        && (0.0..=1.0).contains(&m.anchor_a)
                        && (0.0..=1.0).contains(&m.anchor_b)
                        && [
                            m.anchor_a,
                            m.anchor_b,
                            m.short,
                            m.long,
                            m.period,
                            m.phase,
                            m.duty,
                            m.stiffness,
                        ]
                        .iter()
                        .all(|x| x.is_finite())
                        && m.short >= 0.01
                        && m.long >= m.short
                        && m.period >= if historical { 0.1 } else { MIN_MUSCLE_PERIOD }
                        && (0.05..=0.95).contains(&m.duty)
                        && (1.0..=120.0).contains(&m.stiffness),
                    "Invalid muscle attachment or parameters"
                );
                muscle_adjacency[m.bone_a as usize] |= 1u64 << m.bone_b;
                muscle_adjacency[m.bone_b as usize] |= 1u64 << m.bone_a;
            }
            if historical {
                continue;
            }
            ensure!(
                muscle_adjacency[..g.bone_count].iter().all(|n| *n != 0),
                "Every bone must have an attached muscle"
            );
            let mut reached = 1u64;
            loop {
                let previous = reached;
                for (i, neighbors) in muscle_adjacency[..g.bone_count].iter().enumerate() {
                    if reached & (1u64 << i) != 0 {
                        reached |= neighbors;
                    }
                }
                if reached == previous {
                    break;
                }
            }
            ensure!(
                reached.count_ones() as usize == g.bone_count,
                "Disconnected muscle network"
            );
        }
        Ok(())
    }
}
pub fn canonicalize_bone_order(creature: &mut Creature) -> bool {
    let node_count = creature.nodes.len();
    if !(1..=64).contains(&node_count) || creature.bones.len() != node_count - 1 {
        return false;
    }
    let mut ordered_nodes = 1u64;
    let already_ordered = creature.bones.iter().all(|bone| {
        let parent = bone.a as usize;
        let child = bone.b as usize;
        if parent >= node_count || child >= node_count || parent == child {
            return false;
        }
        let parent_bit = 1u64 << parent;
        let child_bit = 1u64 << child;
        if ordered_nodes & parent_bit == 0 || ordered_nodes & child_bit != 0 {
            return false;
        }
        ordered_nodes |= child_bit;
        true
    });
    if already_ordered && ordered_nodes.count_ones() as usize == node_count {
        return true;
    }
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); node_count];
    for (index, bone) in creature.bones.iter().enumerate() {
        let a = bone.a as usize;
        let b = bone.b as usize;
        if a >= node_count || b >= node_count || a == b {
            return false;
        }
        adjacency[a].push((b, index));
        adjacency[b].push((a, index));
    }
    let mut visited = [false; 64];
    let mut queue = [0usize; 64];
    let mut head = 0;
    let mut tail = 1;
    let mut ordered = Vec::with_capacity(creature.bones.len());
    let mut remap = vec![usize::MAX; creature.bones.len()];
    let mut reversed = vec![false; creature.bones.len()];
    visited[0] = true;
    while head < tail {
        let parent = queue[head];
        head += 1;
        for &(child, old_index) in &adjacency[parent] {
            if visited[child] {
                continue;
            }
            visited[child] = true;
            queue[tail] = child;
            tail += 1;
            let old = creature.bones[old_index];
            remap[old_index] = ordered.len();
            reversed[old_index] = old.a as usize != parent;
            ordered.push(Bone {
                a: parent as u32,
                b: child as u32,
                rest_length: old.rest_length,
            });
        }
    }
    if tail != node_count || ordered.len() != creature.bones.len() {
        return false;
    }
    if creature.muscles.iter().any(|muscle| {
        [muscle.bone_a, muscle.bone_b].iter().any(|bone| {
            remap
                .get(*bone as usize)
                .is_none_or(|index| *index == usize::MAX)
        })
    }) {
        return false;
    }
    for muscle in &mut creature.muscles {
        for (bone, anchor) in [
            (&mut muscle.bone_a, &mut muscle.anchor_a),
            (&mut muscle.bone_b, &mut muscle.anchor_b),
        ] {
            let old_index = *bone as usize;
            if reversed[old_index] {
                *anchor = 1.0 - *anchor;
            }
            *bone = remap[old_index] as u32;
        }
    }
    creature.bones = ordered;
    true
}

fn bone(a: usize, b: usize, nodes: &[NodeGene]) -> Bone {
    let dx = nodes[a].x - nodes[b].x;
    let dy = nodes[a].y - nodes[b].y;
    Bone {
        a: a as u32,
        b: b as u32,
        rest_length: dx.hypot(dy).clamp(0.03, MAX_BONE_LENGTH),
    }
}
pub(crate) fn normalize_bone_lengths(c: &mut Creature) {
    for bone in &mut c.bones {
        let a = c.nodes[bone.a as usize];
        let b = c.nodes[bone.b as usize];
        let distance = (a.x - b.x).hypot(a.y - b.y);
        let min = (distance * 0.75).clamp(0.03, MAX_BONE_LENGTH);
        let max = (distance * 1.25).clamp(min, MAX_BONE_LENGTH);
        bone.rest_length = bone.rest_length.clamp(min, max);
    }
}
fn align_nodes_with_bones(c: &mut Creature) {
    let original = c.nodes.clone();
    for bone in &c.bones {
        let a = bone.a as usize;
        let b = bone.b as usize;
        let dx = original[b].x - original[a].x;
        let dy = original[b].y - original[a].y;
        let length = dx.hypot(dy);
        let direction = if length > 1.0e-6 {
            [dx / length, dy / length]
        } else {
            [1.0, 0.0]
        };
        c.nodes[b].x = c.nodes[a].x + direction[0] * bone.rest_length;
        c.nodes[b].y = c.nodes[a].y + direction[1] * bone.rest_length;
    }
}
fn bone_point(bone: Bone, nodes: &[NodeGene], t: f32) -> [f32; 2] {
    let a = nodes[bone.a as usize];
    let b = nodes[bone.b as usize];
    [a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t]
}
fn random_anchor(rng: &mut Rng) -> f32 {
    if rng.unit() < 0.12 {
        if rng.unit() < 0.5 { 0.0 } else { 1.0 }
    } else {
        rng.unit()
    }
}
fn muscle(
    bone_a: usize,
    bone_b: usize,
    bones: &[Bone],
    nodes: &[NodeGene],
    rng: &mut Rng,
) -> Muscle {
    let anchor_a = random_anchor(rng);
    let anchor_b = random_anchor(rng);
    let a = bone_point(bones[bone_a], nodes, anchor_a);
    let b = bone_point(bones[bone_b], nodes, anchor_b);
    let length = (a[0] - b[0]).hypot(a[1] - b[1]).clamp(0.06, 0.6);
    Muscle {
        bone_a: bone_a as u32,
        bone_b: bone_b as u32,
        anchor_a,
        anchor_b,
        short: length * rng.range(0.65, 0.95),
        long: length * rng.range(1.05, 1.35),
        period: rng.range(0.65, 2.6),
        phase: rng.unit(),
        duty: rng.range(0.25, 0.75),
        stiffness: rng.range(20.0, 80.0),
        sensor: if rng.unit() < 0.5 {
            rng.index(4) as u32
        } else {
            NO_SENSOR
        },
        reset: rng.unit(),
    }
}

pub(crate) fn migrate_legacy_creature(
    nodes: Vec<NodeGene>,
    legacy_muscles: &[LegacyMuscle],
    id: u64,
    mutability: f32,
    cfg: &Config,
) -> Creature {
    let mut creature = Creature {
        bones: (0..nodes.len().saturating_sub(1))
            .map(|i| bone(i, i + 1, &nodes))
            .collect(),
        nodes,
        muscles: Vec::with_capacity(legacy_muscles.len()),
        id,
        mutability,
    };
    let mut rng = Rng::new(cfg.seed, 0, id as usize);
    for old in legacy_muscles {
        let a = old.a as usize;
        let b = old.b as usize;
        if a >= creature.nodes.len() || b >= creature.nodes.len() || a == b {
            continue;
        }
        let mut choices = Vec::new();
        for (bone_a, bone) in creature.bones.iter().enumerate() {
            let anchor_a = if bone.a as usize == a {
                Some(0.0)
            } else if bone.b as usize == a {
                Some(1.0)
            } else {
                None
            };
            let Some(anchor_a) = anchor_a else { continue };
            for (bone_b, other) in creature.bones.iter().enumerate() {
                if bone_a == bone_b {
                    continue;
                }
                let anchor_b = if other.a as usize == b {
                    Some(0.0)
                } else if other.b as usize == b {
                    Some(1.0)
                } else {
                    None
                };
                if let Some(anchor_b) = anchor_b {
                    choices.push((bone_a, bone_b, anchor_a, anchor_b));
                }
            }
        }
        if choices.is_empty() {
            continue;
        }
        let (bone_a, bone_b, anchor_a, anchor_b) = choices[rng.index(choices.len())];
        creature.muscles.push(Muscle {
            bone_a: bone_a as u32,
            bone_b: bone_b as u32,
            anchor_a,
            anchor_b,
            short: old.short,
            long: old.long,
            period: old.period,
            phase: old.phase,
            duty: old.duty,
            stiffness: old.stiffness,
            sensor: NO_SENSOR,
            reset: 0.0,
        });
    }
    repair(&mut creature, cfg, &mut rng);
    creature
}

fn bone_path_exists(bones: &[Bone], node_count: usize, start: usize, target: usize) -> bool {
    let mut reached = 1u64 << start;
    loop {
        let previous = reached;
        for bone in bones {
            let a = bone.a as usize;
            let b = bone.b as usize;
            if a < node_count && b < node_count {
                if reached & (1u64 << a) != 0 {
                    reached |= 1u64 << b;
                }
                if reached & (1u64 << b) != 0 {
                    reached |= 1u64 << a;
                }
            }
        }
        if reached == previous {
            return reached & (1u64 << target) != 0;
        }
    }
}
fn repair(c: &mut Creature, cfg: &Config, rng: &mut Rng) {
    for node in &mut c.nodes {
        node.diameter = node.diameter.clamp(cfg.min_size, cfg.max_size);
        node.friction = node.friction.clamp(cfg.min_friction, cfg.max_friction);
    }
    let node_count = c.nodes.len().min(64);
    let candidates = std::mem::take(&mut c.bones);
    for b in candidates {
        let a = b.a as usize;
        let end = b.b as usize;
        if a < node_count
            && end < node_count
            && a != end
            && b.rest_length.is_finite()
            && (0.03..=12.0).contains(&b.rest_length)
            && !bone_path_exists(&c.bones, node_count, a, end)
        {
            c.bones.push(b);
        }
    }
    // Keep a connected, cycle-free skeleton. New links inherit their current
    // length so repair does not teleport a mutated body before physics starts.
    for node in 1..node_count {
        if !bone_path_exists(&c.bones, node_count, 0, node) {
            c.bones.push(bone(0, node, &c.nodes));
        }
    }

    let bone_count = c.bones.len();
    c.muscles.retain_mut(|m| {
        let a = m.bone_a as usize;
        let b = m.bone_b as usize;
        if a >= bone_count || b >= bone_count || a == b {
            return false;
        }
        m.anchor_a = if m.anchor_a.is_finite() {
            m.anchor_a.clamp(0.0, 1.0)
        } else {
            0.5
        };
        m.anchor_b = if m.anchor_b.is_finite() {
            m.anchor_b.clamp(0.0, 1.0)
        } else {
            0.5
        };
        m.short = if m.short.is_finite() {
            m.short.clamp(0.01, 0.8)
        } else {
            0.1
        };
        m.long = if m.long.is_finite() {
            m.long.clamp(m.short, 1.0)
        } else {
            m.short
        };
        m.period = if m.period.is_finite() {
            m.period.clamp(MIN_MUSCLE_PERIOD, 10.0)
        } else {
            1.0
        };
        m.phase = if m.phase.is_finite() {
            m.phase.rem_euclid(1.0)
        } else {
            0.0
        };
        m.duty = if m.duty.is_finite() {
            m.duty.clamp(0.05, 0.95)
        } else {
            0.5
        };
        m.stiffness = if m.stiffness.is_finite() {
            m.stiffness.clamp(1.0, 120.0)
        } else {
            40.0
        };
        true
    });
    if bone_count < 2 {
        normalize_bone_lengths(c);
        canonicalize_bone_order(c);
        align_nodes_with_bones(c);
        return;
    }
    // A motor-link ring keeps every rigid segment addressable to the actuator
    // network while leaving the skeleton itself articulated at its joints.
    for a in 0..bone_count {
        let b = (a + 1) % bone_count;
        if (bone_count > 2 || a < b)
            && !c.muscles.iter().any(|m| {
                (m.bone_a as usize == a && m.bone_b as usize == b)
                    || (m.bone_a as usize == b && m.bone_b as usize == a)
            })
        {
            if c.muscles.len() >= cfg.max_muscles
                && let Some(i) = c.muscles.iter().position(|m| {
                    let x = m.bone_a as usize;
                    let y = m.bone_b as usize;
                    !((x + 1) % bone_count == y || (y + 1) % bone_count == x)
                })
            {
                c.muscles.swap_remove(i);
            }
            if c.muscles.len() < cfg.max_muscles {
                c.muscles.push(muscle(a, b, &c.bones, &c.nodes, rng));
            }
        }
    }
    normalize_bone_lengths(c);
    canonicalize_bone_order(c);
    align_nodes_with_bones(c);
}
fn initial(cfg: &Config, index: usize) -> Creature {
    random_creature(cfg, 0, index)
}
fn random_creature(cfg: &Config, generation: u32, index: usize) -> Creature {
    let mut creature = random_creature_from(cfg, &mut Rng::new(cfg.seed, generation, index));
    creature.id = index as u64 + 1;
    creature
}
fn random_creature_from(cfg: &Config, rng: &mut Rng) -> Creature {
    let n = (3 + rng.index(3)).min(cfg.max_nodes);
    let spacing = rng.range(0.18, 0.28);
    let mut c = Creature {
        nodes: (0..n)
            .map(|i| NodeGene {
                x: (i as f32 - (n - 1) as f32 * 0.5) * spacing + rng.range(-0.025, 0.025),
                y: 0.18 + (i % 2) as f32 * 0.18 + rng.range(-0.025, 0.025),
                diameter: rng.range(cfg.min_size, cfg.max_size),
                friction: rng.range(cfg.min_friction, cfg.max_friction),
            })
            .collect(),
        bones: Vec::with_capacity(n - 1),
        muscles: vec![],
        id: 0,
        mutability: 1.0,
    };
    for i in 0..n - 1 {
        c.bones.push(bone(i, i + 1, &c.nodes));
    }
    for i in 0..c.bones.len() {
        let j = (i + 1) % c.bones.len();
        if c.bones.len() > 2 || i < j {
            c.muscles.push(muscle(i, j, &c.bones, &c.nodes, rng));
        }
    }
    repair(&mut c, cfg, rng);
    for _ in 0..rng.index(n) {
        if c.muscles.len() < cfg.max_muscles {
            let a = rng.index(c.bones.len());
            let b = rng.index(c.bones.len());
            if a != b {
                c.muscles.push(muscle(a, b, &c.bones, &c.nodes, rng));
            }
        }
    }
    c
}
fn collect_parallel(count: usize, make: impl Fn(usize) -> Creature + Sync) -> Population {
    collect_parallel_streaming(count, count.max(1), make, |_, _| Ok(()))
        .expect("infallible slice callback")
}
/// Builds creatures `0..count` in slices of `slice` creatures (each slice in
/// parallel) and calls `on_slice` with the population built so far after each
/// slice. Indices and contents do not depend on the slice size.
fn collect_parallel_streaming(
    count: usize,
    slice: usize,
    make: impl Fn(usize) -> Creature + Sync,
    mut on_slice: impl FnMut(&Population, std::ops::Range<usize>) -> Result<()>,
) -> Result<Population> {
    let mut out = Population {
        genomes: Vec::with_capacity(count),
        ..Default::default()
    };
    for start in (0..count).step_by(slice.max(1)) {
        let end = (start + slice).min(count);
        // Bounded temporary arenas, not a Vec<Creature> with millions of allocations retained.
        let chunks: Vec<Population> = (start..end)
            .step_by(4096)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|chunk| {
                let mut p = Population::default();
                for i in chunk..(chunk + 4096).min(end) {
                    p.push(make(i));
                }
                p
            })
            .collect();
        for mut chunk in chunks {
            let ns = out.nodes.len();
            let bs = out.bones.len();
            let ms = out.muscles.len();
            for g in &mut chunk.genomes {
                g.node_start += ns;
                g.bone_start += bs;
                g.muscle_start += ms;
            }
            out.genomes.extend(chunk.genomes);
            out.nodes.extend(chunk.nodes);
            out.bones.extend(chunk.bones);
            out.muscles.extend(chunk.muscles);
        }
        on_slice(&out, start..end)?;
    }
    Ok(out)
}
pub fn create(cfg: &Config) -> Result<Population> {
    cfg.validate()?;
    Ok(collect_parallel(cfg.population, |i| initial(cfg, i)))
}

#[derive(Clone, Copy, Debug)]
pub struct CandidatePlan {
    pub emitter: Emitter,
    pub parent: Option<usize>,
    pub cma: Option<usize>,
    /// Second archive parent with the same body plan, for crossover.
    pub mate: Option<usize>,
}

pub fn emit_archive_batch(
    current: &Population,
    archive: &[QdArchive],
    cma_emitters: &[CmaEmitter],
    plans: &[CandidatePlan],
    cfg: &Config,
    generation: u32,
) -> Result<Population> {
    emit_archive_batch_streaming(
        current,
        archive,
        cma_emitters,
        plans,
        cfg,
        generation,
        cfg.population.max(1),
        |_, _| Ok(()),
    )
}
/// Like `emit_archive_batch`, but hands each finished slice of offspring to
/// `on_slice` so evaluation can start while the rest is bred.
#[allow(clippy::too_many_arguments)]
pub fn emit_archive_batch_streaming(
    current: &Population,
    archive: &[QdArchive],
    cma_emitters: &[CmaEmitter],
    plans: &[CandidatePlan],
    cfg: &Config,
    generation: u32,
    slice: usize,
    on_slice: impl FnMut(&Population, std::ops::Range<usize>) -> Result<()>,
) -> Result<Population> {
    ensure_archive_batch_memory(current, &archive[0], cfg)?;
    ensure!(plans.len() == cfg.population, "Invalid emitter plan count");
    collect_parallel_streaming(
        cfg.population,
        slice,
        |i| {
            let mut rng = Rng::new(cfg.seed, generation, i);
            let id = (generation as u64) * cfg.population as u64 + i as u64 + 1;
            offspring(&archive[i % archive.len()], cma_emitters, plans[i], cfg, &mut rng, id)
        },
        on_slice,
    )
}

/// Breeds one offspring from its plan with the given random stream.
fn offspring(
    archive: &QdArchive,
    cma_emitters: &[CmaEmitter],
    plan: CandidatePlan,
    cfg: &Config,
    rng: &mut Rng,
    id: u64,
) -> Creature {
    let mut creature = match plan.emitter {
        Emitter::Restart => random_creature_from(cfg, rng),
        Emitter::Cma => {
            if let Some(cma) = plan.cma.and_then(|index| cma_emitters.get(index)) {
                cma.sample_scaled(rng, cfg.mutation)
            } else {
                let parent = &archive.entries[plan.parent.expect("CMA parent")].creature;
                local_mutation(parent.clone(), cfg, rng, 0.12)
            }
        }
        Emitter::Structural => {
            let parent = mated(archive, plan, rng);
            let (child, _) = structural_mutation(parent, cfg, rng);
            local_mutation(child, cfg, rng, 0.035)
        }
        Emitter::Novelty => {
            let parent = mated(archive, plan, rng);
            // Occasional large jumps help lineages cross fitness valleys.
            let scale = if rng.unit() < 0.05 { 2.25 } else { 0.75 };
            let mut child = local_mutation(parent, cfg, rng, scale);
            if rng.unit() < 0.18 {
                let _ = structural_mutation_in_place(&mut child, cfg, rng);
            }
            child
        }
    };
    creature.id = id;
    repair(&mut creature, cfg, rng);
    creature
}

/// The plan's parent, crossed with its mate when it has one.
fn mated(archive: &QdArchive, plan: CandidatePlan, rng: &mut Rng) -> Creature {
    let parent = &archive.entries[plan.parent.expect("archive parent")].creature;
    match plan.mate {
        Some(mate) => crossover(parent, &archive.entries[mate].creature, rng),
        None => parent.clone(),
    }
}

/// Uniform crossover of two creatures with the same body plan: each node,
/// bone, and muscle comes from one parent. Sometimes the whole muscle rhythm
/// (periods and phases) comes from one parent so gaits stay coherent.
pub fn crossover(a: &Creature, b: &Creature, rng: &mut Rng) -> Creature {
    let mut child = a.clone();
    if a.nodes.len() != b.nodes.len()
        || a.bones.len() != b.bones.len()
        || a.muscles.len() != b.muscles.len()
    {
        return child;
    }
    for (node, other) in child.nodes.iter_mut().zip(&b.nodes) {
        if rng.unit() < 0.5 {
            *node = *other;
        }
    }
    for (bone, other) in child.bones.iter_mut().zip(&b.bones) {
        if rng.unit() < 0.5 && (bone.a, bone.b) == (other.a, other.b) {
            bone.rest_length = other.rest_length;
        }
    }
    let rhythm_from_b = if rng.unit() < 0.3 {
        Some(rng.unit() < 0.5)
    } else {
        None
    };
    for (muscle, other) in child.muscles.iter_mut().zip(&b.muscles) {
        if (muscle.bone_a, muscle.bone_b) != (other.bone_a, other.bone_b) {
            continue;
        }
        let (period, phase) = (muscle.period, muscle.phase);
        if rng.unit() < 0.5 {
            *muscle = *other;
        }
        match rhythm_from_b {
            Some(true) => (muscle.period, muscle.phase) = (other.period, other.phase),
            Some(false) => (muscle.period, muscle.phase) = (period, phase),
            None => {}
        }
    }
    child
}

/// Copies a leaf limb as its mirror image around its joint, with copies of the
/// limb's muscles running half a cycle out of phase (alternating legs).
fn duplicate_limb(creature: &mut Creature, cfg: &Config, rng: &mut Rng) -> bool {
    if creature.nodes.len() >= cfg.max_nodes || creature.bones.is_empty() {
        return false;
    }
    let degree = |node: u32| {
        creature
            .bones
            .iter()
            .filter(|b| b.a == node || b.b == node)
            .count()
    };
    let leaves: Vec<usize> = (0..creature.bones.len())
        .filter(|&i| degree(creature.bones[i].b) == 1 || degree(creature.bones[i].a) == 1)
        .collect();
    if leaves.is_empty() {
        return false;
    }
    let limb = leaves[rng.index(leaves.len())];
    let bone = creature.bones[limb];
    let (joint, tip) = if degree(bone.b) == 1 {
        (bone.a, bone.b)
    } else {
        (bone.b, bone.a)
    };
    let attached: Vec<Muscle> = creature
        .muscles
        .iter()
        .filter(|m| m.bone_a as usize == limb || m.bone_b as usize == limb)
        .copied()
        .collect();
    if creature.muscles.len() + attached.len() > cfg.max_muscles {
        return false;
    }
    let pivot = creature.nodes[joint as usize];
    let mut mirror = creature.nodes[tip as usize];
    mirror.x = (2.0 * pivot.x - mirror.x).clamp(-4.0, 4.0);
    let new_node = creature.nodes.len() as u32;
    creature.nodes.push(mirror);
    let new_bone = creature.bones.len() as u32;
    creature.bones.push(Bone {
        a: joint,
        b: new_node,
        rest_length: bone.rest_length,
    });
    for mut m in attached {
        if m.bone_a as usize == limb {
            m.bone_a = new_bone;
        }
        if m.bone_b as usize == limb {
            m.bone_b = new_bone;
        }
        if m.bone_a == m.bone_b {
            continue;
        }
        m.phase = (m.phase + 0.5).rem_euclid(1.0);
        creature.muscles.push(m);
    }
    repair(creature, cfg, rng);
    true
}

/// Gives every muscle the same period, taken from one of them.
fn sync_rhythm(creature: &mut Creature, rng: &mut Rng) -> bool {
    if creature.muscles.len() < 2 {
        return false;
    }
    let period = creature.muscles[rng.index(creature.muscles.len())].period;
    for muscle in &mut creature.muscles {
        muscle.period = period;
    }
    true
}

/// Steady-state breeding: one offspring per plan, for population `slots`.
/// `round` salts the random streams and keeps creature ids unique.
pub fn emit_offspring(
    archive: &[QdArchive],
    cma_emitters: &[CmaEmitter],
    plans: &[CandidatePlan],
    slots: &[usize],
    cfg: &Config,
    generation: u32,
    round: u64,
) -> Vec<Creature> {
    let seed = cfg.seed ^ round.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5bd1_e995;
    plans
        .par_iter()
        .zip(slots)
        .map(|(&plan, &slot)| {
            let mut rng = Rng::new(seed, generation, slot);
            let id = (round << 32) ^ ((generation as u64) << 24) ^ slot as u64 ^ (1 << 63);
            offspring(&archive[slot % archive.len()], cma_emitters, plan, cfg, &mut rng, id)
        })
        .collect()
}

pub fn ensure_archive_batch_memory(
    current: &Population,
    archive: &QdArchive,
    cfg: &Config,
) -> Result<()> {
    let archive_bytes = archive
        .entries
        .iter()
        .map(|elite| {
            elite.creature.nodes.len() * std::mem::size_of::<NodeGene>()
                + elite.creature.bones.len() * std::mem::size_of::<Bone>()
                + elite.creature.muscles.len() * std::mem::size_of::<Muscle>()
                + std::mem::size_of::<Creature>()
        })
        .sum::<usize>();
    let temporary = current
        .bytes()
        .saturating_mul(4)
        .saturating_add(archive_bytes.saturating_mul(3))
        .saturating_add(cfg.population.saturating_mul(96));
    ensure!(
        temporary < cfg.ram_budget_mib * 1024 * 1024,
        "Evolution would exceed the RAM budget; save and raise the budget before continuing"
    );
    Ok(())
}

fn local_mutation(mut creature: Creature, cfg: &Config, rng: &mut Rng, scale: f32) -> Creature {
    let scale = scale * cfg.mutation;
    if scale <= 0.0 {
        return creature;
    }
    for node in &mut creature.nodes {
        node.x = (node.x + qd::gaussian(rng) * 0.10 * scale).clamp(-4.0, 4.0);
        node.y = (node.y + qd::gaussian(rng) * 0.08 * scale).clamp(0.0, 4.0);
        node.diameter =
            (node.diameter + qd::gaussian(rng) * 0.025 * scale).clamp(cfg.min_size, cfg.max_size);
        node.friction = (node.friction + qd::gaussian(rng) * 0.10 * scale)
            .clamp(cfg.min_friction, cfg.max_friction);
    }
    for bone in &mut creature.bones {
        bone.rest_length =
            (bone.rest_length + qd::gaussian(rng) * 0.035 * scale).clamp(0.03, MAX_BONE_LENGTH);
    }
    for muscle in &mut creature.muscles {
        muscle.anchor_a = (muscle.anchor_a + qd::gaussian(rng) * 0.10 * scale).clamp(0.0, 1.0);
        muscle.anchor_b = (muscle.anchor_b + qd::gaussian(rng) * 0.10 * scale).clamp(0.0, 1.0);
        muscle.short = (muscle.short + qd::gaussian(rng) * 0.06 * scale).clamp(0.01, 0.8);
        muscle.long = (muscle.long + qd::gaussian(rng) * 0.08 * scale).clamp(muscle.short, 1.0);
        muscle.period =
            (muscle.period + qd::gaussian(rng) * 0.20 * scale).clamp(MIN_MUSCLE_PERIOD, 10.0);
        muscle.phase = (muscle.phase + qd::gaussian(rng) * 0.12 * scale).rem_euclid(1.0);
        muscle.duty = (muscle.duty + qd::gaussian(rng) * 0.08 * scale).clamp(0.05, 0.95);
        muscle.stiffness =
            (muscle.stiffness * (qd::gaussian(rng) * 0.10 * scale).exp()).clamp(1.0, 120.0);
        muscle.reset = (muscle.reset + qd::gaussian(rng) * 0.12 * scale).rem_euclid(1.0);
        if rng.unit() < 0.05 * scale.min(1.0) {
            muscle.sensor = match rng.index(5) {
                4 => NO_SENSOR,
                endpoint => endpoint as u32,
            };
        }
    }
    creature.mutability = (creature.mutability * (qd::gaussian(rng) * 0.05).exp()).clamp(0.05, 2.0);
    creature
}

fn structural_mutation(mut creature: Creature, cfg: &Config, rng: &mut Rng) -> (Creature, bool) {
    let changed = structural_mutation_in_place(&mut creature, cfg, rng);
    (creature, changed)
}

/// Benchmark workload helper: grows a body with the game's own structural
/// mutations until it has at least `target_nodes` nodes or cannot grow further.
pub fn grow_for_benchmark(creature: &mut Creature, cfg: &Config, seed: u64, target_nodes: usize) {
    let mut rng = Rng::new(seed, u32::MAX, creature.id as usize);
    let mut attempts = 0;
    while creature.nodes.len() < target_nodes.min(cfg.max_nodes) && attempts < 1000 {
        attempts += 1;
        if rng.unit() < 0.5 {
            split_bone(creature, cfg, &mut rng);
        } else {
            duplicate_mirrored_node(creature, cfg, &mut rng);
        }
    }
    repair(creature, cfg, &mut rng);
}

fn structural_mutation_in_place(creature: &mut Creature, cfg: &Config, rng: &mut Rng) -> bool {
    match rng.index(5) {
        0 => split_bone(creature, cfg, rng),
        1 => duplicate_mirrored_node(creature, cfg, rng),
        2 => duplicate_limb(creature, cfg, rng),
        3 => sync_rhythm(creature, rng),
        _ => phase_shift_group(creature, rng),
    }
}

fn split_bone(creature: &mut Creature, cfg: &Config, rng: &mut Rng) -> bool {
    if creature.nodes.len() >= cfg.max_nodes
        || creature.bones.is_empty()
        || creature.bones.iter().all(|bone| bone.rest_length < 0.06)
    {
        return false;
    }
    let eligible: Vec<usize> = creature
        .bones
        .iter()
        .enumerate()
        .filter_map(|(index, bone)| (bone.rest_length >= 0.06).then_some(index))
        .collect();
    let index = eligible[rng.index(eligible.len())];
    let original = creature.bones[index];
    let a = creature.nodes[original.a as usize];
    let b = creature.nodes[original.b as usize];
    let middle = NodeGene {
        x: (a.x + b.x) * 0.5,
        y: (a.y + b.y) * 0.5,
        diameter: (a.diameter + b.diameter) * 0.5,
        friction: (a.friction + b.friction) * 0.5,
    };
    let mid = creature.nodes.len() as u32;
    creature.nodes.push(middle);
    let second_index = creature.bones.len() as u32;
    let first_length = original.rest_length * 0.5;
    creature.bones[index] = Bone {
        a: original.a,
        b: mid,
        rest_length: first_length,
    };
    creature.bones.push(Bone {
        a: mid,
        b: original.b,
        rest_length: original.rest_length - first_length,
    });
    for muscle in &mut creature.muscles {
        if muscle.bone_a as usize == index {
            if muscle.anchor_a <= 0.5 {
                muscle.anchor_a *= 2.0;
            } else {
                muscle.bone_a = second_index;
                muscle.anchor_a = (muscle.anchor_a - 0.5) * 2.0;
            }
        }
        if muscle.bone_b as usize == index {
            if muscle.anchor_b <= 0.5 {
                muscle.anchor_b *= 2.0;
            } else {
                muscle.bone_b = second_index;
                muscle.anchor_b = (muscle.anchor_b - 0.5) * 2.0;
            }
        }
    }
    true
}

fn duplicate_mirrored_node(creature: &mut Creature, cfg: &Config, rng: &mut Rng) -> bool {
    if creature.nodes.len() >= cfg.max_nodes || creature.muscles.len() >= cfg.max_muscles {
        return false;
    }
    if creature.bones.is_empty() || creature.bones.len() < 2 {
        return false;
    }
    let parent_bone_index = rng.index(creature.bones.len());
    let parent_bone = creature.bones[parent_bone_index];
    let source = if rng.unit() < 0.5 {
        parent_bone.a as usize
    } else {
        parent_bone.b as usize
    };
    let center_x = creature.nodes.iter().map(|n| n.x).sum::<f32>() / creature.nodes.len() as f32;
    let mut duplicate = creature.nodes[source];
    duplicate.x = (2.0 * center_x - duplicate.x + rng.range(-0.03, 0.03)).clamp(-4.0, 4.0);
    duplicate.y = (duplicate.y + rng.range(-0.03, 0.03)).clamp(0.0, 4.0);
    let target = creature.nodes.len() as u32;
    creature.nodes.push(duplicate);
    let new_bone = creature.bones.len();
    creature
        .bones
        .push(bone(source, target as usize, &creature.nodes));
    let other_bone = rng.index(new_bone);
    creature.muscles.push(muscle(
        new_bone,
        other_bone,
        &creature.bones,
        &creature.nodes,
        rng,
    ));
    repair(creature, cfg, rng);
    true
}

fn phase_shift_group(creature: &mut Creature, rng: &mut Rng) -> bool {
    if creature.bones.is_empty() || creature.muscles.is_empty() {
        return false;
    }
    let bone = rng.index(creature.bones.len()) as u32;
    let offset = rng.range(-0.25, 0.25);
    let mut changed = false;
    for muscle in &mut creature.muscles {
        if muscle.bone_a == bone || muscle.bone_b == bone {
            muscle.phase = (muscle.phase + offset).rem_euclid(1.0);
            changed = true;
        }
    }
    changed
}
// The original rank-and-reproduce helpers remain for legacy callers. The game
// and CLI use emit_archive_batch and never use the exact-clone pairing rule.
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
    for bone in &mut c.bones {
        bone.rest_length =
            (bone.rest_length + rng.delta() * 0.04 * strength).clamp(0.03, MAX_BONE_LENGTH);
    }
    for m in &mut c.muscles {
        m.anchor_a = (m.anchor_a + rng.delta() * 0.15 * strength).clamp(0.0, 1.0);
        m.anchor_b = (m.anchor_b + rng.delta() * 0.15 * strength).clamp(0.0, 1.0);
        m.short = (m.short + rng.delta() * 0.1 * strength).clamp(0.02, 0.8);
        m.long = (m.long + rng.delta() * 0.1 * strength).clamp(m.short, 1.0);
        m.period = (m.period + rng.delta() * 0.2 * strength).clamp(MIN_MUSCLE_PERIOD, 10.0);
        m.phase = (m.phase + rng.delta() * 0.2 * strength).rem_euclid(1.0);
        m.duty = (m.duty + rng.delta() * 0.1 * strength).clamp(0.05, 0.95);
        m.stiffness = (m.stiffness * (1.0 + rng.delta() * 0.3 * strength)).clamp(1.0, 120.0);
    }
    if rng.unit() < 0.04 * strength {
        let _ = structural_mutation_in_place(&mut c, cfg, &mut rng);
    }
    if rng.unit() < 0.04 * strength && c.muscles.len() < cfg.max_muscles {
        let a = rng.index(c.bones.len());
        let b = rng.index(c.bones.len());
        if a != b {
            c.muscles.push(muscle(a, b, &c.bones, &c.nodes, &mut rng));
        }
    }
    if rng.unit() < 0.04 * strength && c.muscles.len() > c.bones.len() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bone_length_cannot_expand_far_beyond_its_starting_frame() {
        let mut creature = Creature {
            nodes: vec![
                NodeGene {
                    x: 0.0,
                    y: 0.0,
                    diameter: 0.08,
                    friction: 0.5,
                },
                NodeGene {
                    x: 0.5,
                    y: 0.0,
                    diameter: 0.08,
                    friction: 0.5,
                },
            ],
            bones: vec![Bone {
                a: 0,
                b: 1,
                rest_length: 9.6,
            }],
            muscles: vec![],
            id: 0,
            mutability: 1.0,
        };
        normalize_bone_lengths(&mut creature);
        assert_eq!(creature.bones[0].rest_length, 0.625);
        creature.nodes[1].x = 4.0;
        normalize_bone_lengths(&mut creature);
        assert_eq!(creature.bones[0].rest_length, MAX_BONE_LENGTH);
    }

    #[test]
    fn repair_applies_body_bounds_and_starts_bones_at_their_rest_lengths() {
        let cfg = Config::default();
        let mut creature = Creature {
            nodes: vec![
                NodeGene {
                    x: 0.0,
                    y: 0.0,
                    diameter: 0.01,
                    friction: 0.0,
                },
                NodeGene {
                    x: 7.0,
                    y: 0.0,
                    diameter: 0.02,
                    friction: 0.1,
                },
                NodeGene {
                    x: 4.0,
                    y: 0.0,
                    diameter: 0.03,
                    friction: 0.2,
                },
            ],
            bones: vec![
                Bone {
                    a: 0,
                    b: 1,
                    rest_length: 2.0,
                },
                Bone {
                    a: 1,
                    b: 2,
                    rest_length: 2.0,
                },
            ],
            muscles: vec![],
            id: 0,
            mutability: 1.0,
        };
        repair(&mut creature, &cfg, &mut Rng::new(42, 0, 0));
        for node in &creature.nodes {
            assert!((cfg.min_size..=cfg.max_size).contains(&node.diameter));
            assert!((cfg.min_friction..=cfg.max_friction).contains(&node.friction));
        }
        for bone in &creature.bones {
            let a = creature.nodes[bone.a as usize];
            let b = creature.nodes[bone.b as usize];
            assert!(((a.x - b.x).hypot(a.y - b.y) - bone.rest_length).abs() < 1e-5);
        }
        assert!(
            creature
                .muscles
                .iter()
                .all(|m| m.period >= MIN_MUSCLE_PERIOD)
        );
    }

    fn muscle_point(creature: &Creature, bone_id: u32, t: f32) -> [f32; 2] {
        let bone = creature.bones[bone_id as usize];
        let a = creature.nodes[bone.a as usize];
        let b = creature.nodes[bone.b as usize];
        [a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t]
    }

    #[test]
    fn split_bone_preserves_attachments_on_both_halves() {
        let cfg = Config {
            max_nodes: 4,
            ..Config::default()
        };
        for split_index in 0..2 {
            let nodes = vec![
                NodeGene {
                    x: 0.0,
                    y: 0.0,
                    diameter: 0.08,
                    friction: 0.5,
                },
                NodeGene {
                    x: 1.0,
                    y: 0.0,
                    diameter: 0.08,
                    friction: 0.5,
                },
                NodeGene {
                    x: 1.0,
                    y: 1.0,
                    diameter: 0.08,
                    friction: 0.5,
                },
            ];
            let mut creature = Creature {
                nodes,
                bones: vec![
                    Bone {
                        a: 0,
                        b: 1,
                        rest_length: 1.0,
                    },
                    Bone {
                        a: 1,
                        b: 2,
                        rest_length: 1.0,
                    },
                ],
                muscles: vec![
                    Muscle {
                        bone_a: 0,
                        bone_b: 1,
                        anchor_a: 0.25,
                        anchor_b: 0.25,
                        short: 0.1,
                        long: 0.2,
                        period: 1.0,
                        phase: 0.0,
                        duty: 0.5,
                        stiffness: 40.0,
                        sensor: 255,
                        reset: 0.0,
                    },
                    Muscle {
                        bone_a: 0,
                        bone_b: 1,
                        anchor_a: 0.75,
                        anchor_b: 0.75,
                        short: 0.1,
                        long: 0.2,
                        period: 1.0,
                        phase: 0.5,
                        duty: 0.5,
                        stiffness: 40.0,
                        sensor: 255,
                        reset: 0.0,
                    },
                ],
                id: 1,
                mutability: 1.0,
            };
            let old_points: Vec<_> = creature
                .muscles
                .iter()
                .map(|muscle| {
                    [
                        muscle_point(&creature, muscle.bone_a, muscle.anchor_a),
                        muscle_point(&creature, muscle.bone_b, muscle.anchor_b),
                    ]
                })
                .collect();
            let seed = (0..100)
                .find(|&seed| Rng::new(seed, 0, 0).index(2) == split_index)
                .unwrap();
            assert!(split_bone(&mut creature, &cfg, &mut Rng::new(seed, 0, 0)));
            assert_eq!(creature.nodes.len(), 4);
            assert_eq!(creature.bones.len(), 3);
            for (muscle, points) in creature.muscles.iter().zip(old_points) {
                let actual = [
                    muscle_point(&creature, muscle.bone_a, muscle.anchor_a),
                    muscle_point(&creature, muscle.bone_b, muscle.anchor_b),
                ];
                for side in 0..2 {
                    assert!((actual[side][0] - points[side][0]).abs() < 1e-6);
                    assert!((actual[side][1] - points[side][1]).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn canonical_bone_order_preserves_attachment_positions() {
        let mut creature = Creature {
            nodes: (0..4)
                .map(|i| NodeGene {
                    x: i as f32,
                    y: 0.0,
                    diameter: 0.08,
                    friction: 0.5,
                })
                .collect(),
            bones: vec![
                Bone {
                    a: 2,
                    b: 3,
                    rest_length: 1.0,
                },
                Bone {
                    a: 1,
                    b: 0,
                    rest_length: 1.0,
                },
                Bone {
                    a: 2,
                    b: 1,
                    rest_length: 1.0,
                },
            ],
            muscles: vec![Muscle {
                bone_a: 0,
                bone_b: 1,
                anchor_a: 0.25,
                anchor_b: 0.75,
                short: 0.1,
                long: 0.2,
                period: 1.0,
                phase: 0.0,
                duty: 0.5,
                stiffness: 40.0,
                sensor: 255,
                reset: 0.0,
            }],
            id: 1,
            mutability: 1.0,
        };
        let before = [
            muscle_point(
                &creature,
                creature.muscles[0].bone_a,
                creature.muscles[0].anchor_a,
            ),
            muscle_point(
                &creature,
                creature.muscles[0].bone_b,
                creature.muscles[0].anchor_b,
            ),
        ];

        assert!(canonicalize_bone_order(&mut creature));
        assert_eq!(
            creature
                .bones
                .iter()
                .map(|bone| (bone.a, bone.b))
                .collect::<Vec<_>>(),
            vec![(0, 1), (1, 2), (2, 3)]
        );
        let after = [
            muscle_point(
                &creature,
                creature.muscles[0].bone_a,
                creature.muscles[0].anchor_a,
            ),
            muscle_point(
                &creature,
                creature.muscles[0].bone_b,
                creature.muscles[0].anchor_b,
            ),
        ];
        for side in 0..2 {
            assert!((before[side][0] - after[side][0]).abs() < 1e-6);
            assert!((before[side][1] - after[side][1]).abs() < 1e-6);
        }
    }
}
