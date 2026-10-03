pub use crate::bounded::Bounded;
use crate::config::Config;
use crate::qd::{self, CmaEmitter, Emitter, QdArchive};
use anyhow::{Result, ensure};
use bytemuck::Zeroable;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod anatomy;

pub const FAILED: f32 = -1.0e20;
/// Longest bone (m), from `physics::limits()`.
pub fn max_bone_length() -> f32 {
    crate::physics::limits().max_bone
}
/// Largest distance (m) of a starting node from the origin on either axis.
fn body_extent() -> f32 {
    2.0 * max_bone_length()
}
/// Longest muscle length (m), from `physics::limits()`.
pub fn max_stroke() -> f32 {
    crate::physics::limits().max_stroke
}
/// Shortest muscle rhythm period (s), from `physics::limits()`.
pub fn min_muscle_period() -> f32 {
    crate::physics::limits().min_period
}

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
    /// Joint range at node `a` (radians), measured from the starting pose:
    /// how far this bone may turn clockwise (`min_angle`, <= 0) and
    /// counterclockwise (`max_angle`, >= 0) against its reference bone.
    /// Both stay within `JOINT_LIMIT`, so no joint can spin all the way round.
    #[serde(default = "joint_min")]
    pub min_angle: f32,
    #[serde(default = "joint_max")]
    pub max_angle: f32,
    /// Mass (kg) of the organ this bone carries; zero without an organ.
    #[serde(default)]
    pub organ_mass: f32,
    /// Where the organ sits along the bone, from node `a` (0) to `b` (1).
    #[serde(default = "organ_middle")]
    pub organ_at: f32,
}
/// Organ masses (kg). A new organ starts light so it barely changes the gait.
pub const MIN_ORGAN_MASS: f32 = 0.01;
pub const MAX_ORGAN_MASS: f32 = 0.3;
/// Organs sit within this distance (m) of the body's center of mass in the
/// starting pose, measured without organs and without the head, so they stay
/// inside the body instead of weighting the tips of limbs.
pub const ORGAN_RADIUS: f32 = 0.5;
fn organ_middle() -> f32 {
    0.5
}
/// Widest joint range on either side of the starting pose (120 degrees).
pub const JOINT_LIMIT: f32 = 120.0 * std::f32::consts::PI / 180.0;
fn joint_min() -> f32 {
    -JOINT_LIMIT
}
fn joint_max() -> f32 {
    JOINT_LIMIT
}
impl Bone {
    /// A bone with the widest joint range.
    pub fn new(a: u32, b: u32, rest_length: f32) -> Self {
        Self {
            a,
            b,
            rest_length,
            min_angle: -JOINT_LIMIT,
            max_angle: JOINT_LIMIT,
            organ_mass: 0.0,
            organ_at: 0.5,
        }
    }
    /// Keeps the joint range valid.
    pub fn clamp_range(&mut self) {
        self.min_angle = self.min_angle.clamp(-JOINT_LIMIT, 0.0);
        self.max_angle = self.max_angle.clamp(0.0, JOINT_LIMIT);
    }
}
/// Center of mass of the starting pose without the head (node 0) and
/// without organs: the point organs must stay near.
pub fn organ_center(nodes: &[NodeGene]) -> [f32; 2] {
    let body = if nodes.len() > 1 { &nodes[1..] } else { nodes };
    let mut sum = [0.0f32; 2];
    let mut mass = 0.0f32;
    for n in body {
        let m = crate::physics::node_mass(n.diameter);
        sum[0] += n.x * m;
        sum[1] += n.y * m;
        mass += m;
    }
    [sum[0] / mass.max(1e-9), sum[1] / mass.max(1e-9)]
}
/// Positions along `bone` (as a 0-1 range) that lie within `ORGAN_RADIUS`
/// of `center`, or `None` when the whole bone is too far away.
fn organ_range(bone: &Bone, nodes: &[NodeGene], center: [f32; 2]) -> Option<(f32, f32)> {
    let a = nodes[bone.a as usize];
    let b = nodes[bone.b as usize];
    let d = [b.x - a.x, b.y - a.y];
    let f = [a.x - center[0], a.y - center[1]];
    // |f + t d|^2 <= R^2 is a quadratic in t.
    let qa = d[0] * d[0] + d[1] * d[1];
    let qb = 2.0 * (f[0] * d[0] + f[1] * d[1]);
    let qc = f[0] * f[0] + f[1] * f[1] - ORGAN_RADIUS * ORGAN_RADIUS;
    if qa < 1e-12 {
        return (qc <= 0.0).then_some((0.0, 1.0));
    }
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let low = ((-qb - root) / (2.0 * qa)).max(0.0);
    let high = ((-qb + root) / (2.0 * qa)).min(1.0);
    (low <= high).then_some((low, high))
}
/// Keeps every organ valid and inside the body: an organ outside the allowed
/// region slides to the nearest allowed point on its bone, and an organ on a
/// bone that never comes close enough to the body's center is dropped.
fn place_organs(c: &mut Creature) {
    let center = organ_center(&c.nodes);
    for bone in &mut c.bones {
        if !(bone.organ_mass.is_finite() && bone.organ_at.is_finite()) || bone.organ_mass <= 0.0 {
            bone.organ_mass = 0.0;
            bone.organ_at = 0.5;
            continue;
        }
        bone.organ_mass = bone.organ_mass.clamp(MIN_ORGAN_MASS, MAX_ORGAN_MASS);
        match organ_range(bone, &c.nodes, center) {
            Some((low, high)) => bone.organ_at = bone.organ_at.clamp(low, high),
            None => {
                bone.organ_mass = 0.0;
                bone.organ_at = 0.5;
            }
        }
    }
}
/// Adds a light organ to a bone that can hold one, or removes an organ.
fn change_organ(creature: &mut Creature, rng: &mut Rng) -> bool {
    let with: Bounded<usize, MAX_NODES> = (0..creature.bones.len())
        .filter(|&i| creature.bones[i].organ_mass > 0.0)
        .collect();
    if !with.is_empty() && rng.unit() < 0.3 {
        let bone = &mut creature.bones[with[rng.index(with.len())]];
        bone.organ_mass = 0.0;
        bone.organ_at = 0.5;
        return true;
    }
    let center = organ_center(&creature.nodes);
    let free: Bounded<(usize, (f32, f32)), MAX_NODES> = (0..creature.bones.len())
        .filter(|&i| creature.bones[i].organ_mass <= 0.0)
        .filter_map(|i| organ_range(&creature.bones[i], &creature.nodes, center).map(|r| (i, r)))
        .collect();
    if free.is_empty() {
        return false;
    }
    let (index, (low, high)) = free[rng.index(free.len())];
    let bone = &mut creature.bones[index];
    bone.organ_mass = rng.range(MIN_ORGAN_MASS, 0.06);
    bone.organ_at = rng.range(low, high);
    true
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
    /// Elastic tendon in parallel with the muscle, 0 (none) to 1 (stiffest):
    /// once the muscle is stretched past its longest length the tendon pulls
    /// back like a spring, storing the energy of the stretch and returning it.
    #[serde(default)]
    pub tendon: f32,
}
/// Stiffest tendon: it reaches the muscle's force cap when stretched by this
/// share of the muscle's longest length.
pub const TENDON_STRETCH: f32 = 0.25;
/// A muscle without a touchdown sensor.
pub const NO_SENSOR: u32 = 255;
fn no_sensor() -> u32 {
    NO_SENSOR
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Genome {
    pub node_start: usize,
    pub node_count: usize,
    pub bone_start: usize,
    pub bone_count: usize,
    pub muscle_start: usize,
    pub muscle_count: usize,
    pub id: u64,
}
/// Most nodes a body may have (`Config::max_nodes` is at most this).
pub const MAX_NODES: usize = 32;
/// Most muscles a body may have (`Config::max_muscles` is at most this).
pub const MAX_MUSCLES: usize = 96;
pub type Nodes = Bounded<NodeGene, MAX_NODES>;
pub type Bones = Bounded<Bone, MAX_NODES>;
pub type Muscles = Bounded<Muscle, MAX_MUSCLES>;
/// A body's genes, held inline in bounded arrays: a creature is about 6.4 KB
/// and breeding one never allocates.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Creature {
    pub nodes: Nodes,
    pub bones: Bones,
    pub muscles: Muscles,
    pub id: u64,
}
/// `clone_from` copies only the genes in use into the existing arrays, where
/// the derived one would build a whole new creature and move it.
impl Clone for Creature {
    fn clone(&self) -> Self {
        Self {
            nodes: Bounded::from_slice(&self.nodes),
            bones: Bounded::from_slice(&self.bones),
            muscles: Bounded::from_slice(&self.muscles),
            id: self.id,
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.nodes.clone_from(&source.nodes);
        self.bones.clone_from(&source.bones);
        self.muscles.clone_from(&source.muscles);
        self.id = source.id;
    }
}
/// Splits `all` into consecutive parts of the given sizes.
fn split<T>(mut all: &mut [T], sizes: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    let mut out = Vec::new();
    for size in sizes {
        let (head, tail) = std::mem::take(&mut all).split_at_mut(size);
        out.push(head);
        all = tail;
    }
    out
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Population {
    pub genomes: Vec<Genome>,
    pub nodes: Vec<NodeGene>,
    pub bones: Vec<Bone>,
    pub muscles: Vec<Muscle>,
    /// How each creature's trial treats it (`rungs::AUDIT`, `rungs::EXEMPT`),
    /// one byte per genome, set when its block is bred. Empty means no flags.
    #[serde(default)]
    pub flags: Vec<u8>,
}

/// The splitmix64 finalizer: a bijection on 64 bits that mixes every input
/// bit into every output bit.
#[inline(always)]
fn finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
/// Draw `draw` of gene `gene` in the stream with `key`: SplitMix64 started
/// at the key, at counter `gene << 32 | draw`. The key comes from four rounds
/// of the finalizer over the stream's coordinates (`stream_key`), so streams
/// start at unrelated points, and within one stream every (gene, draw) pair
/// gets its own counter.
#[inline(always)]
fn keyed(key: u64, gene: u32, draw: u32) -> u64 {
    let counter = (gene as u64) << 32 | draw as u64;
    finalize(key.wrapping_add(counter.wrapping_mul(0x9e3779b97f4a7c15)))
}
/// A stream key from its coordinates, absorbed one at a time.
fn stream_key(seed: u64, generation: u64, round: u64, slot: u64) -> u64 {
    let mut key = finalize(seed.wrapping_add(0x632be59bd9b4e019));
    for word in [generation, round, slot] {
        key = finalize(key ^ word.wrapping_mul(0xd1342543de82ef95));
    }
    key
}
/// The gene index the cursor draws of a stream use.
const CURSOR: u32 = u32::MAX;
/// 1 / 65535: twelve 16-bit uniforms on 0..65535, scaled by this, have
/// variance 1 to 3e-5.
const GAUSSIAN_SCALE: f32 = 1.0 / 65535.0;
/// Twelve 16-bit uniforms from three draws, summed and centered: a gaussian
/// with mean exactly 0 and variance 1, cut at 6. The sum is an integer, so
/// the value is the same on any machine.
#[inline(always)]
fn twelve_uniforms(words: [u64; 3]) -> f32 {
    let mut sum = 0i64;
    for w in words {
        sum += (w & 0xffff) as i64
            + (w >> 16 & 0xffff) as i64
            + (w >> 32 & 0xffff) as i64
            + (w >> 48) as i64;
    }
    (sum - 6 * 65535) as f32 * GAUSSIAN_SCALE
}

/// Counter-based random numbers. A stream is keyed by its coordinates (the
/// seed, the generation, the breeding round and the ring slot), and every
/// value is a hash of the key and the value's own index: a child is a
/// function of its slot and its plan, whatever thread breeds it and in
/// whatever order.
///
/// The cursor (`next_u64`, `unit`, `gaussian`) walks the stream in order, as
/// structural operators read it. `genes` hands out a sub-stream whose values
/// are keyed by gene index, so parametric noise on a gene does not depend on
/// the draws before it.
pub struct Rng {
    key: u64,
    counter: u32,
}
impl Rng {
    pub fn new(seed: u64, generation: u32, index: usize) -> Self {
        Self::stream(seed, generation, 0, index)
    }
    /// The stream of the child bred for ring `slot` in breeding `round`.
    pub fn stream(seed: u64, generation: u32, round: u64, slot: usize) -> Self {
        Self {
            key: stream_key(seed, generation as u64, round, slot as u64),
            counter: 0,
        }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let value = keyed(self.key, CURSOR, self.counter);
        self.counter = self.counter.wrapping_add(1);
        value
    }
    #[inline]
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
    /// A standard gaussian from three cursor draws (`twelve_uniforms`).
    #[inline]
    pub fn gaussian(&mut self) -> f32 {
        twelve_uniforms([self.next_u64(), self.next_u64(), self.next_u64()])
    }
    /// A new sub-stream keyed by gene index, for one pass of parametric
    /// noise over a body. Takes one cursor draw.
    pub fn genes(&mut self) -> Genes {
        Genes {
            key: self.next_u64(),
        }
    }
}
/// Random values keyed by gene index (`Rng::genes`). Gene `g` gets draws
/// `0, 1, 2` for its gaussian and `3..` for its uniforms.
#[derive(Clone, Copy)]
pub struct Genes {
    key: u64,
}
impl Genes {
    #[inline(always)]
    pub fn gaussian(self, gene: u32) -> f32 {
        twelve_uniforms([
            keyed(self.key, gene, 0),
            keyed(self.key, gene, 1),
            keyed(self.key, gene, 2),
        ])
    }
    /// Uniform on [0, 1): draw `3 + k` of `gene`.
    #[inline(always)]
    pub fn unit(self, gene: u32, k: u32) -> f32 {
        (keyed(self.key, gene, 3 + k) >> 40) as f32 / 16_777_216.0
    }
    #[inline(always)]
    pub fn index(self, gene: u32, k: u32, n: usize) -> usize {
        (keyed(self.key, gene, 3 + k) % n as u64) as usize
    }
}
impl Population {
    pub fn creature(&self, index: usize) -> Creature {
        let g = &self.genomes[index];
        Creature {
            nodes: Bounded::from_slice(&self.nodes[g.node_start..g.node_start + g.node_count]),
            bones: Bounded::from_slice(&self.bones[g.bone_start..g.bone_start + g.bone_count]),
            muscles: Bounded::from_slice(
                &self.muscles[g.muscle_start..g.muscle_start + g.muscle_count],
            ),
            id: g.id,
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
        });
        self.nodes.extend(c.nodes);
        self.bones.extend(c.bones);
        self.muscles.extend(c.muscles);
    }
    /// Copies `indices` into a standalone population; creature `k` of the
    /// result is `indices[k]` of `self`.
    pub fn subset(&self, indices: &[usize]) -> Population {
        const CHUNK: usize = 4096;
        // Gene counts per run of creatures, then every run copies its genes
        // straight into its own part of the output arenas, in parallel.
        let sizes: Vec<[usize; 3]> = indices
            .par_chunks(CHUNK)
            .map(|chunk| {
                chunk.iter().fold([0; 3], |t, &i| {
                    let g = &self.genomes[i];
                    [
                        t[0] + g.node_count,
                        t[1] + g.bone_count,
                        t[2] + g.muscle_count,
                    ]
                })
            })
            .collect();
        let total = sizes
            .iter()
            .fold([0; 3], |t, s| [t[0] + s[0], t[1] + s[1], t[2] + s[2]]);
        let mut out = Population {
            genomes: Vec::with_capacity(indices.len()),
            nodes: Vec::with_capacity(total[0]),
            bones: Vec::with_capacity(total[1]),
            muscles: Vec::with_capacity(total[2]),
            flags: if self.flags.is_empty() {
                Vec::new()
            } else {
                indices.iter().map(|&i| self.flags[i]).collect()
            },
        };
        let mut starts = Vec::with_capacity(sizes.len());
        let mut at = [0usize; 3];
        for s in &sizes {
            starts.push(at);
            at = [at[0] + s[0], at[1] + s[1], at[2] + s[2]];
        }
        let genome_parts = split(
            &mut out.genomes.spare_capacity_mut()[..indices.len()],
            indices.chunks(CHUNK).map(<[usize]>::len),
        );
        let node_parts = split(
            &mut out.nodes.spare_capacity_mut()[..total[0]],
            sizes.iter().map(|s| s[0]),
        );
        let bone_parts = split(
            &mut out.bones.spare_capacity_mut()[..total[1]],
            sizes.iter().map(|s| s[1]),
        );
        let muscle_parts = split(
            &mut out.muscles.spare_capacity_mut()[..total[2]],
            sizes.iter().map(|s| s[2]),
        );
        indices
            .par_chunks(CHUNK)
            .zip(genome_parts)
            .zip(node_parts)
            .zip(bone_parts)
            .zip(muscle_parts)
            .zip(starts)
            .for_each(|(((((chunk, genomes), nodes), bones), muscles), start)| {
                let mut at = [0usize; 3];
                for (&i, genome) in chunk.iter().zip(genomes) {
                    let g = &self.genomes[i];
                    genome.write(Genome {
                        node_start: start[0] + at[0],
                        bone_start: start[1] + at[1],
                        muscle_start: start[2] + at[2],
                        ..g.clone()
                    });
                    for (dst, src) in nodes[at[0]..]
                        .iter_mut()
                        .zip(&self.nodes[g.node_start..g.node_start + g.node_count])
                    {
                        dst.write(*src);
                    }
                    for (dst, src) in bones[at[1]..]
                        .iter_mut()
                        .zip(&self.bones[g.bone_start..g.bone_start + g.bone_count])
                    {
                        dst.write(*src);
                    }
                    for (dst, src) in muscles[at[2]..]
                        .iter_mut()
                        .zip(&self.muscles[g.muscle_start..g.muscle_start + g.muscle_count])
                    {
                        dst.write(*src);
                    }
                    at = [
                        at[0] + g.node_count,
                        at[1] + g.bone_count,
                        at[2] + g.muscle_count,
                    ];
                }
            });
        // SAFETY: the parts cover the first elements of each arena, and every
        // run wrote all of its part.
        unsafe {
            out.genomes.set_len(indices.len());
            out.nodes.set_len(total[0]);
            out.bones.set_len(total[1]);
            out.muscles.set_len(total[2]);
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
        self.validate_with_max_bone(cfg, max_bone_length(), false)
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
                        && (0.03..=max_bone_length).contains(&bone.rest_length)
                        && (-JOINT_LIMIT..=0.0).contains(&bone.min_angle)
                        && (0.0..=JOINT_LIMIT).contains(&bone.max_angle)
                        && (bone.organ_mass == 0.0
                            || (MIN_ORGAN_MASS..=MAX_ORGAN_MASS).contains(&bone.organ_mass))
                        && (0.0..=1.0).contains(&bone.organ_at),
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
                            m.tendon,
                        ]
                        .iter()
                        .all(|x| x.is_finite())
                        && (0.0..=1.0).contains(&m.tendon)
                        && m.short >= 0.01
                        && m.long >= m.short
                        && m.period >= if historical { 0.1 } else { min_muscle_period() }
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
    if !(1..=MAX_NODES).contains(&node_count) || creature.bones.len() != node_count - 1 {
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
    if creature.bones.iter().any(|bone| {
        let (a, b) = (bone.a as usize, bone.b as usize);
        a >= node_count || b >= node_count || a == b
    }) {
        return false;
    }
    let mut visited = [false; MAX_NODES];
    let mut queue = [0usize; MAX_NODES];
    let mut head = 0;
    let mut tail = 1;
    let mut ordered = Bones::new();
    let mut remap = [usize::MAX; MAX_NODES];
    let mut reversed = [false; MAX_NODES];
    visited[0] = true;
    while head < tail {
        let parent = queue[head];
        head += 1;
        // The parent's bones in bone order: the order an adjacency list
        // built bone by bone would hold them in.
        for old_index in 0..creature.bones.len() {
            let bone = creature.bones[old_index];
            let child = if bone.a as usize == parent {
                bone.b as usize
            } else if bone.b as usize == parent {
                bone.a as usize
            } else {
                continue;
            };
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
                organ_at: if reversed[old_index] {
                    1.0 - old.organ_at
                } else {
                    old.organ_at
                },
                ..old
            });
        }
    }
    if tail != node_count || ordered.len() != creature.bones.len() {
        return false;
    }
    let bones = creature.bones.len();
    if creature.muscles.iter().any(|muscle| {
        [muscle.bone_a, muscle.bone_b].iter().any(|bone| {
            remap[..bones]
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
    Bone::new(
        a as u32,
        b as u32,
        dx.hypot(dy).clamp(0.03, max_bone_length()),
    )
}
/// The period ratios a limb may run at against the body's base clock (the
/// first muscle's period). Simple ratios keep the whole gait exactly
/// periodic, repeating every few base cycles.
pub const CLOCK_RATIOS: [f32; 5] = [0.5, 2.0 / 3.0, 1.0, 1.5, 2.0];

/// Every muscle runs on the body's base clock, set by the first muscle, or on
/// a simple multiple of it (`CLOCK_RATIOS`): a period equal to a ratio (to
/// 0.3%, so ratios survive the arithmetic of mutation) keeps it, and any other
/// period falls back to the base, so muscles that repair or an operator adds
/// with a random period join the body's clock.
fn snap_clock_ratios(c: &mut Creature) {
    let Some(base) = c.muscles.first().map(|m| m.period) else {
        return;
    };
    let (low, high) = (min_muscle_period(), 10.0);
    let logs = CLOCK_RATIOS.map(f32::ln);
    for m in &mut c.muscles {
        let log = (m.period / base).ln();
        let best = (0..CLOCK_RATIOS.len())
            .min_by(|&a, &b| (logs[a] - log).abs().total_cmp(&(logs[b] - log).abs()))
            .unwrap_or(2);
        let ratio = if (logs[best] - log).abs() < 0.003
            && (low..=high).contains(&(base * CLOCK_RATIOS[best]))
        {
            CLOCK_RATIOS[best]
        } else {
            1.0
        };
        m.period = base * ratio;
    }
}
pub(crate) fn normalize_bone_lengths(c: &mut Creature) {
    for bone in &mut c.bones {
        let a = c.nodes[bone.a as usize];
        let b = c.nodes[bone.b as usize];
        let distance = (a.x - b.x).hypot(a.y - b.y);
        let min = (distance * 0.75).clamp(0.03, max_bone_length());
        let max = (distance * 1.25).clamp(min, max_bone_length());
        bone.rest_length = bone.rest_length.clamp(min, max);
    }
}
fn align_nodes_with_bones(c: &mut Creature) {
    // The starting positions are needed while the new ones are written in
    // place, so they are copied onto the stack first.
    let mut flat = [0.0f32; 2 * MAX_NODES];
    for (index, node) in c.nodes.iter().enumerate() {
        flat[2 * index] = node.x;
        flat[2 * index + 1] = node.y;
    }
    let original = |index: usize| -> (f32, f32) { (flat[2 * index], flat[2 * index + 1]) };
    for bone in &c.bones {
        let a = bone.a as usize;
        let b = bone.b as usize;
        let (ax, ay) = original(a);
        let (bx, by) = original(b);
        let dx = bx - ax;
        let dy = by - ay;
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
    let length = (a[0] - b[0])
        .hypot(a[1] - b[1])
        .clamp(0.06, 0.6 * max_stroke());
    Muscle {
        bone_a: bone_a as u32,
        bone_b: bone_b as u32,
        anchor_a,
        anchor_b,
        short: length * rng.range(0.55, 0.85),
        long: length * rng.range(1.15, 1.45),
        period: rng.range(0.65, 2.6),
        phase: rng.unit(),
        duty: rng.range(0.25, 0.75),
        stiffness: rng.range(60.0, 120.0),
        sensor: if rng.unit() < 0.5 {
            rng.index(4) as u32
        } else {
            NO_SENSOR
        },
        reset: rng.unit(),
        tendon: 0.0,
    }
}

/// Largest tilt of the neck from vertical in the starting pose.
const HEAD_START_TILT: f32 = std::f32::consts::FRAC_PI_4;
/// Every creature has a head: node 0, as large (and heavy) as a node can be,
/// on a single neck bone. Other bones on the head move to the neck's base,
/// and the neck starts pointing up. A creature whose neck tips below
/// horizontal has fallen (see the engines).
fn shape_head(c: &mut Creature, cfg: &Config) {
    let Some(neck) = c.bones.iter().position(|b| b.a == 0 || b.b == 0) else {
        return;
    };
    let base = (c.bones[neck].a + c.bones[neck].b) as usize;
    for (index, b) in c.bones.iter_mut().enumerate() {
        if index == neck || (b.a != 0 && b.b != 0) {
            continue;
        }
        if b.a == 0 {
            b.a = base as u32;
        } else {
            b.b = base as u32;
        }
        b.rest_length = bone(b.a as usize, b.b as usize, &c.nodes).rest_length;
    }
    c.nodes[0].diameter = cfg.max_size;
    let (bx, by) = (c.nodes[base].x, c.nodes[base].y);
    let (dx, dy) = (c.nodes[0].x - bx, c.nodes[0].y - by);
    let length = dx.hypot(dy).max(0.03);
    let tilt = dx.atan2(dy).clamp(-HEAD_START_TILT, HEAD_START_TILT);
    c.nodes[0].x = bx + length * tilt.sin();
    c.nodes[0].y = by + length * tilt.cos();
}
fn repair(c: &mut Creature, cfg: &Config, rng: &mut Rng) {
    for node in &mut c.nodes {
        node.diameter = node.diameter.clamp(cfg.min_size, cfg.max_size);
        node.friction = node.friction.clamp(cfg.min_friction, cfg.max_friction);
    }
    let node_count = c.nodes.len();
    // Incremental connectivity over the <= 64 nodes: accepted bones always
    // join two components, so the union-find answers the reachability test
    // that a per-candidate graph walk used to run.
    fn root(parent: &mut [u8; 64], mut node: u8) -> u8 {
        while parent[node as usize] != node {
            parent[node as usize] = parent[parent[node as usize] as usize];
            node = parent[node as usize];
        }
        node
    }
    let mut parent: [u8; 64] = std::array::from_fn(|index| index as u8);
    let connected = |parent: &mut [u8; 64], a: u8, b: u8| {
        let (ra, rb) = (root(parent, a), root(parent, b));
        if ra == rb {
            return true;
        }
        parent[ra as usize] = rb;
        false
    };
    let candidates = std::mem::take(&mut c.bones);
    for mut b in candidates {
        b.clamp_range();
        let a = b.a as usize;
        let end = b.b as usize;
        if a < node_count
            && end < node_count
            && a != end
            && b.rest_length.is_finite()
            && (0.03..=12.0).contains(&b.rest_length)
            && !connected(&mut parent, a as u8, end as u8)
        {
            c.bones.push(b);
        }
    }
    // Keep a connected, cycle-free skeleton. New links inherit their current
    // length so repair does not teleport a mutated body before physics starts.
    for node in 1..node_count {
        if !connected(&mut parent, 0, node as u8) {
            c.bones.push(bone(0, node, &c.nodes));
        }
    }
    shape_head(c, cfg);

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
            m.short.clamp(0.01, 0.8 * max_stroke())
        } else {
            0.1
        };
        m.long = if m.long.is_finite() {
            m.long.clamp(m.short, max_stroke())
        } else {
            m.short
        };
        m.period = if m.period.is_finite() {
            m.period.clamp(min_muscle_period(), 10.0)
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
        m.tendon = if m.tendon.is_finite() {
            m.tendon.clamp(0.0, 1.0)
        } else {
            0.0
        };
        true
    });
    if bone_count < 2 {
        normalize_bone_lengths(c);
        canonicalize_bone_order(c);
        align_nodes_with_bones(c);
        place_organs(c);
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
            // At the muscle limit, a muscle off the ring makes room, or else
            // a second muscle on one ring pair: without it a limb's worth of
            // duplicates could leave the network disconnected.
            let pair = |m: &Muscle| (m.bone_a.min(m.bone_b), m.bone_a.max(m.bone_b));
            if c.muscles.len() >= cfg.max_muscles
                && let Some(i) = c
                    .muscles
                    .iter()
                    .position(|m| {
                        let x = m.bone_a as usize;
                        let y = m.bone_b as usize;
                        !((x + 1) % bone_count == y || (y + 1) % bone_count == x)
                    })
                    .or_else(|| {
                        (1..c.muscles.len()).find(|&i| {
                            c.muscles[..i]
                                .iter()
                                .any(|m| pair(m) == pair(&c.muscles[i]))
                        })
                    })
            {
                c.muscles.swap_remove(i);
            }
            if c.muscles.len() < cfg.max_muscles {
                c.muscles.push(muscle(a, b, &c.bones, &c.nodes, rng));
            }
        }
    }
    snap_clock_ratios(c);
    normalize_bone_lengths(c);
    canonicalize_bone_order(c);
    align_nodes_with_bones(c);
    place_organs(c);
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
        bones: Bones::new(),
        muscles: Muscles::new(),
        id: 0,
    };
    for i in 0..n - 1 {
        let mut b = bone(i, i + 1, &c.nodes);
        b.min_angle = -rng.range(0.3, JOINT_LIMIT);
        b.max_angle = rng.range(0.3, JOINT_LIMIT);
        c.bones.push(b);
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
/// Builds creatures `0..count` in parallel, in index order.
fn collect_parallel(count: usize, make: impl Fn(usize) -> Creature + Sync) -> Population {
    let mut out = Population {
        genomes: Vec::with_capacity(count),
        ..Default::default()
    };
    // Bounded temporary arenas, not a Vec<Creature> with millions of allocations retained.
    let chunks: Vec<Population> = (0..count)
        .step_by(4096)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|chunk| {
            // Typical bodies are 3 to 8 nodes; a low estimate only costs a
            // growth reallocation, never a different result.
            let estimate = (chunk + 4096).min(count) - chunk;
            let mut p = Population {
                nodes: Vec::with_capacity(estimate * 8),
                bones: Vec::with_capacity(estimate * 8),
                muscles: Vec::with_capacity(estimate * 10),
                ..Default::default()
            };
            for i in chunk..(chunk + 4096).min(count) {
                p.push(make(i));
            }
            p
        })
        .collect();
    // Reserve the merged arenas exactly so the copies below do not
    // reallocate the growing gene vectors.
    out.nodes
        .reserve_exact(chunks.iter().map(|chunk| chunk.nodes.len()).sum());
    out.bones
        .reserve_exact(chunks.iter().map(|chunk| chunk.bones.len()).sum());
    out.muscles
        .reserve_exact(chunks.iter().map(|chunk| chunk.muscles.len()).sum());
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
    out
}
pub fn create(cfg: &Config) -> Result<Population> {
    cfg.validate()?;
    Ok(collect_parallel(cfg.population, |i| initial(cfg, i)))
}
/// New random bodies for ring slots `first..first + count`.
pub fn random_block(cfg: &Config, first: usize, count: usize) -> Population {
    collect_parallel(count, |k| initial(cfg, first + k))
}
/// Ring slots have fewer than 2^24 places, so a bred creature's id holds its
/// breeding round and its slot, and no two bred creatures share an id.
const SLOT_BITS: u32 = 24;
const BRED: u64 = 1 << 63;
/// Id of the child bred for ring `slot` in breeding round `round`.
pub fn bred_id(round: u64, slot: usize) -> u64 {
    BRED | (round << SLOT_BITS) | slot as u64
}
/// The ring slot a creature was born in, from its id.
pub fn slot_of_id(id: u64) -> usize {
    if id & BRED != 0 {
        (id & ((1 << SLOT_BITS) - 1)) as usize
    } else {
        id.saturating_sub(1) as usize
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CandidatePlan {
    pub emitter: Emitter,
    pub parent: Option<usize>,
    pub cma: Option<usize>,
    /// Second archive parent with the same body plan, for crossover.
    pub mate: Option<usize>,
    /// The parent and the mate are elites of the slot's island, not of the
    /// archive the slot breeds for: a reshaped child for a nursery.
    pub seed: bool,
}

/// The growth-step body rule (the owner's decision of 2026-10-01): a child gains at most this many nodes and muscles
/// over its parent, so bodies grow by steps rather than jumps. In a
/// generation-50 dump 4% of archive entrants had jumped further.
pub const GROWTH_STEP: Option<GrowthStep> = Some(GrowthStep {
    nodes: 4,
    muscles: 4,
});
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowthStep {
    pub nodes: usize,
    pub muscles: usize,
}

/// The body limits a child of `parent` must fit: the config caps, lowered to
/// the parent's size plus `step` when there is one. Every operator's fit
/// check reads `cfg.max_nodes` and `cfg.max_muscles`, so tighter limits
/// reach them as a config (`None` when the config's own caps apply).
pub fn child_limits(cfg: &Config, parent: &Creature, step: Option<GrowthStep>) -> Option<Config> {
    let step = step?;
    let nodes = cfg.max_nodes.min(parent.nodes.len() + step.nodes);
    let muscles = cfg.max_muscles.min(parent.muscles.len() + step.muscles);
    (nodes < cfg.max_nodes || muscles < cfg.max_muscles).then(|| Config {
        max_nodes: nodes,
        max_muscles: muscles,
        ..cfg.clone()
    })
}

/// What breeding did to one child, for `examples/breed_bench.rs`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChildTrace {
    /// The child went through `structural_mutation_from`.
    pub structural: bool,
    /// The structural operator that changed it, as an index into
    /// `structural_operator_names`.
    pub operator: Option<u8>,
}

/// Breeds one offspring from its plan with the given random stream into
/// `child`, which it overwrites.
#[allow(clippy::too_many_arguments)]
fn offspring(
    archive: &QdArchive,
    cma_emitters: &[CmaEmitter],
    plan: CandidatePlan,
    cfg: &Config,
    rng: &mut Rng,
    id: u64,
    step: Option<GrowthStep>,
    child: &mut Creature,
) -> ChildTrace {
    let mut trace = ChildTrace::default();
    let limited = match plan.emitter {
        Emitter::Structural | Emitter::Novelty => plan
            .parent
            .and_then(|p| child_limits(cfg, &archive.entries[p].creature, step)),
        Emitter::Restart | Emitter::Cma => None,
    };
    let cfg = limited.as_ref().unwrap_or(cfg);
    match plan.emitter {
        Emitter::Restart => *child = random_creature_from(cfg, rng),
        Emitter::Cma => {
            if let Some(cma) = plan.cma.and_then(|index| cma_emitters.get(index)) {
                cma.sample_into(rng, cfg.mutation, child);
            } else {
                child.clone_from(&archive.entries[plan.parent.expect("CMA parent")].creature);
                mutate_genes(child, cfg, rng, 0.12);
            }
        }
        Emitter::Structural => {
            mated(archive, plan, cfg, rng, child);
            trace.structural = true;
            trace.operator = structural_mutation_from(child, cfg, rng, archive);
            // A compound operator's change is whole: noise on every gene
            // after it halves how often its child enters the archive
            // (`examples/mutation_audit.rs`).
            let compound = trace.operator.is_some_and(|op| {
                (op as usize)
                    .checked_sub(CLASSIC_COUNT)
                    .is_some_and(anatomy::is_compound)
            });
            if !compound {
                mutate_genes(child, cfg, rng, 0.035);
            }
        }
        Emitter::Novelty => {
            mated(archive, plan, cfg, rng, child);
            // Occasional large jumps help lineages cross fitness valleys.
            let scale = if rng.unit() < 0.05 { 2.25 } else { 0.75 };
            mutate_genes(child, cfg, rng, scale);
            if rng.unit() < 0.18 {
                trace.structural = true;
                trace.operator = structural_mutation_from(child, cfg, rng, archive);
            }
        }
    }
    child.id = id;
    repair(child, cfg, rng);
    trace
}

/// Breeds the child of ring `slot` in breeding `round` from its plan into
/// `child`, as `Population::breed` does, and says what breeding did.
#[allow(clippy::too_many_arguments)]
pub fn breed_child(
    archive: &[QdArchive],
    cma_emitters: &[CmaEmitter],
    plan: CandidatePlan,
    slot: usize,
    cfg: &Config,
    generation: u32,
    round: u64,
    child: &mut Creature,
) -> ChildTrace {
    let mut rng = Rng::stream(cfg.seed, generation, round, slot);
    let source = if plan.seed {
        qd::island_of_slot(slot, archive.len() / qd::ARENA_KINDS)
    } else {
        qd::arena_of_slot(slot, archive.len())
    };
    offspring(
        &archive[source],
        cma_emitters,
        plan,
        cfg,
        &mut rng,
        bred_id(round, slot),
        GROWTH_STEP,
        child,
    )
}

/// The plan's parent, crossed with its mate when it has one, into `child`.
fn mated(
    archive: &QdArchive,
    plan: CandidatePlan,
    cfg: &Config,
    rng: &mut Rng,
    child: &mut Creature,
) {
    let parent = &archive.entries[plan.parent.expect("archive parent")].creature;
    match plan.mate {
        Some(mate) => {
            let mate = &archive.entries[mate].creature;
            if same_shape(parent, mate) {
                crossover_into(parent, mate, rng, child);
            } else {
                // Different body plans: graft one of the mate's limbs, with
                // its muscles and rhythm, onto a copy of the parent.
                child.clone_from(parent);
                for _ in 0..4 {
                    if anatomy::graft_from(child, cfg, rng, mate) {
                        break;
                    }
                }
            }
        }
        None => child.clone_from(parent),
    }
}

/// Whether two creatures have the same nodes, bones and muscle pairs, so
/// `crossover` can pair their genes.
fn same_shape(a: &Creature, b: &Creature) -> bool {
    a.nodes.len() == b.nodes.len()
        && a.bones.len() == b.bones.len()
        && a.muscles.len() == b.muscles.len()
        && a.bones
            .iter()
            .zip(&b.bones)
            .all(|(x, y)| (x.a, x.b) == (y.a, y.b))
        && a.muscles
            .iter()
            .zip(&b.muscles)
            .all(|(x, y)| (x.bone_a, x.bone_b) == (y.bone_a, y.bone_b))
}

/// Uniform crossover of two creatures with the same body plan: each node,
/// bone, and muscle comes from one parent. Sometimes the whole muscle rhythm
/// (periods and phases) comes from one parent so gaits stay coherent.
pub fn crossover(a: &Creature, b: &Creature, rng: &mut Rng) -> Creature {
    let mut child = Creature::default();
    crossover_into(a, b, rng, &mut child);
    child
}
/// `crossover` into `child`, which it overwrites.
fn crossover_into(a: &Creature, b: &Creature, rng: &mut Rng, child: &mut Creature) {
    child.clone_from(a);
    if a.nodes.len() != b.nodes.len()
        || a.bones.len() != b.bones.len()
        || a.muscles.len() != b.muscles.len()
    {
        return;
    }
    for (node, other) in child.nodes.iter_mut().zip(&b.nodes) {
        if rng.unit() < 0.5 {
            *node = *other;
        }
    }
    for (bone, other) in child.bones.iter_mut().zip(&b.bones) {
        if rng.unit() < 0.5 && (bone.a, bone.b) == (other.a, other.b) {
            bone.rest_length = other.rest_length;
            bone.min_angle = other.min_angle;
            bone.max_angle = other.max_angle;
            bone.organ_mass = other.organ_mass;
            bone.organ_at = other.organ_at;
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
    let leaves: Bounded<usize, MAX_NODES> = (0..creature.bones.len())
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
    let attached: Muscles = creature
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
    mirror.x = (2.0 * pivot.x - mirror.x).clamp(-body_extent(), body_extent());
    let new_node = creature.nodes.len() as u32;
    creature.nodes.push(mirror);
    let new_bone = creature.bones.len() as u32;
    // The mirrored limb bends the other way.
    creature.bones.push(Bone {
        a: joint,
        b: new_node,
        rest_length: bone.rest_length,
        min_angle: -bone.max_angle,
        max_angle: -bone.min_angle,
        ..Bone::new(joint, new_node, bone.rest_length)
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
/// Changes the body clock's tempo by a large step, keeping every phase.
fn retime_rhythm(creature: &mut Creature, rng: &mut Rng) -> bool {
    if creature.muscles.is_empty() {
        return false;
    }
    let tempo = (qd::gaussian(rng) * 0.35).exp();
    for muscle in &mut creature.muscles {
        muscle.period = (muscle.period * tempo).clamp(min_muscle_period(), 10.0);
    }
    true
}

/// The structural operator (its index in `structural_operator_names`) that
/// changed each child bred while the log is on, by child id: a diagnostic
/// for the generation dump (`storage`), off otherwise.
static OPERATOR_LOG: std::sync::Mutex<Option<std::collections::HashMap<u64, u8>>> =
    std::sync::Mutex::new(None);
/// Starts or stops recording each child's structural operator.
pub fn record_operators(on: bool) {
    let mut log = OPERATOR_LOG.lock().unwrap_or_else(|e| e.into_inner());
    *log = on.then(|| log.take().unwrap_or_default());
}
/// The operators recorded since the last call, by child id.
pub fn take_operators() -> std::collections::HashMap<u64, u8> {
    OPERATOR_LOG
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .map(std::mem::take)
        .unwrap_or_default()
}

/// Children bred per task. Each run of this many children gets its own part
/// of the block's gene arena.
const BREED_CHUNK: usize = 4096;

/// A node gene with every field zero, to fill arena space not yet written.
const NO_NODE: NodeGene = NodeGene {
    x: 0.0,
    y: 0.0,
    diameter: 0.0,
    friction: 0.0,
};

/// One run of children's part of a block's gene arena: the arena indices
/// where it starts, the genes written so far, and the free space left.
struct ArenaPart<'a> {
    base: [usize; 3],
    at: [usize; 3],
    nodes: &'a mut [NodeGene],
    bones: &'a mut [Bone],
    muscles: &'a mut [Muscle],
}

impl ArenaPart<'_> {
    /// Writes `c`'s genes after the ones already here and returns its genome,
    /// or `None` when the part has no room left for it.
    fn put(&mut self, c: &Creature) -> Option<Genome> {
        let counts = [c.nodes.len(), c.bones.len(), c.muscles.len()];
        let room = [self.nodes.len(), self.bones.len(), self.muscles.len()];
        if (0..3).any(|k| self.at[k] + counts[k] > room[k]) {
            return None;
        }
        let [n, b, m] = self.at;
        self.nodes[n..n + counts[0]].copy_from_slice(&c.nodes);
        self.bones[b..b + counts[1]].copy_from_slice(&c.bones);
        self.muscles[m..m + counts[2]].copy_from_slice(&c.muscles);
        let genome = Genome {
            node_start: self.base[0] + n,
            node_count: counts[0],
            bone_start: self.base[1] + b,
            bone_count: counts[1],
            muscle_start: self.base[2] + m,
            muscle_count: counts[2],
            id: c.id,
        };
        self.at = [n + counts[0], b + counts[1], m + counts[2]];
        Some(genome)
    }
}

/// Genome slots written by parallel tasks, each at its own block position.
#[derive(Clone, Copy)]
struct GenomeOut(*mut Genome);
unsafe impl Send for GenomeOut {}
unsafe impl Sync for GenomeOut {}

impl Population {
    /// Breeds a ring block into this population, which becomes the block's
    /// genes: `count` creatures, the elites of `lead` at their positions and
    /// one child per plan at `positions`, bred for ring `slots` (`round`
    /// salts the random streams and keeps creature ids unique).
    ///
    /// The population is the block's gene arena and is reused from one
    /// breeding to the next, so its memory is allocated and touched once.
    /// Every run of `BREED_CHUNK` children gets a part of the arena sized
    /// from the genes the same positions held last time (from `hint` when
    /// this population held no block of this size), plus a quarter, and
    /// writes each child there as soon as it is bred. A child that does not
    /// fit goes after all parts. Returns how many children went there.
    #[allow(clippy::too_many_arguments)]
    pub fn breed(
        &mut self,
        count: usize,
        hint: Option<&Population>,
        lead: &mut [(usize, Creature)],
        archive: &[QdArchive],
        cma_emitters: &[CmaEmitter],
        plans: &[CandidatePlan],
        slots: &[usize],
        positions: &[usize],
        cfg: &Config,
        generation: u32,
        round: u64,
    ) -> usize {
        assert_eq!(plans.len(), slots.len());
        assert_eq!(plans.len(), positions.len());
        assert_eq!(lead.len() + plans.len(), count);
        // Genes per part: what the part's positions held last time.
        let last: &Population = match hint {
            Some(h) if self.genomes.len() != count => h,
            _ => self,
        };
        let known = last.genomes.len() == count;
        let held = |chunk: &[usize]| -> [usize; 3] {
            let t = chunk.iter().fold([0usize; 3], |t, &k| {
                let s = if known {
                    let g = &last.genomes[k];
                    [g.node_count, g.bone_count, g.muscle_count]
                } else {
                    // A new arena: a typical body.
                    [8, 7, 16]
                };
                [t[0] + s[0], t[1] + s[1], t[2] + s[2]]
            });
            t.map(|x| x + x / 4 + 256)
        };
        for (_, c) in lead.iter_mut() {
            canonicalize_bone_order(c);
        }
        let lead_size = lead.iter().fold([0usize; 3], |t, (_, c)| {
            [
                t[0] + c.nodes.len(),
                t[1] + c.bones.len(),
                t[2] + c.muscles.len(),
            ]
        });
        let parts: Vec<[usize; 3]> = std::iter::once(lead_size)
            .chain(positions.chunks(BREED_CHUNK).map(held))
            .collect();
        let need = parts
            .iter()
            .fold([0usize; 3], |t, s| [t[0] + s[0], t[1] + s[1], t[2] + s[2]]);
        // Grow the arena once, with room for the next block's growth too.
        fn fit<T: Copy>(v: &mut Vec<T>, need: usize, zero: T) {
            if v.len() < need {
                v.reserve_exact((need + need / 4).saturating_sub(v.len()));
                v.resize(need, zero);
            }
        }
        fit(&mut self.nodes, need[0], NO_NODE);
        fit(&mut self.bones, need[1], Bone::zeroed());
        fit(&mut self.muscles, need[2], Muscle::zeroed());
        if self.genomes.len() != count {
            self.genomes.clear();
            self.genomes.resize(count, Genome::default());
        }
        let node_parts = split(&mut self.nodes[..need[0]], parts.iter().map(|s| s[0]));
        let bone_parts = split(&mut self.bones[..need[1]], parts.iter().map(|s| s[1]));
        let muscle_parts = split(&mut self.muscles[..need[2]], parts.iter().map(|s| s[2]));
        let mut base = [0usize; 3];
        let mut arena_parts: Vec<ArenaPart> = Vec::with_capacity(parts.len());
        for (((nodes, bones), muscles), size) in node_parts
            .into_iter()
            .zip(bone_parts)
            .zip(muscle_parts)
            .zip(&parts)
        {
            arena_parts.push(ArenaPart {
                base,
                at: [0; 3],
                nodes,
                bones,
                muscles,
            });
            base = [base[0] + size[0], base[1] + size[1], base[2] + size[2]];
        }
        let mut arena_parts = arena_parts.into_iter();
        let mut lead_part = arena_parts.next().expect("the lead part");
        for (k, c) in lead.iter() {
            self.genomes[*k] = lead_part.put(c).expect("the lead part holds the lead");
        }
        let genomes = GenomeOut(self.genomes.as_mut_ptr());
        let spilled: Vec<Vec<(usize, Creature)>> = arena_parts
            .collect::<Vec<_>>()
            .into_par_iter()
            .zip(plans.par_chunks(BREED_CHUNK))
            .zip(slots.par_chunks(BREED_CHUNK))
            .zip(positions.par_chunks(BREED_CHUNK))
            .map(|(((mut part, plans), slots), positions)| {
                let genomes = genomes;
                let mut spill = Vec::new();
                let recording = OPERATOR_LOG
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_some();
                let mut operators = Vec::new();
                let mut child = Creature::default();
                for ((&plan, &slot), &k) in plans.iter().zip(slots).zip(positions) {
                    let trace = breed_child(
                        archive,
                        cma_emitters,
                        plan,
                        slot,
                        cfg,
                        generation,
                        round,
                        &mut child,
                    );
                    if recording && let Some(operator) = trace.operator {
                        operators.push((child.id, operator));
                    }
                    canonicalize_bone_order(&mut child);
                    match part.put(&child) {
                        // SAFETY: positions are distinct and below `count`,
                        // the genome vector's length, and nothing else
                        // touches the vector while the tasks run.
                        Some(genome) => unsafe { *genomes.0.add(k) = genome },
                        None => spill.push((k, child.clone())),
                    }
                }
                if !operators.is_empty()
                    && let Some(log) = OPERATOR_LOG
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .as_mut()
                {
                    log.extend(operators);
                }
                spill
            })
            .collect();
        // Children that did not fit their part go after every part.
        let mut at = need;
        let mut late = 0;
        for (k, child) in spilled.into_iter().flatten() {
            late += 1;
            fit(&mut self.nodes, at[0] + child.nodes.len(), NO_NODE);
            fit(&mut self.bones, at[1] + child.bones.len(), Bone::zeroed());
            fit(
                &mut self.muscles,
                at[2] + child.muscles.len(),
                Muscle::zeroed(),
            );
            let mut part = ArenaPart {
                base: at,
                at: [0; 3],
                nodes: &mut self.nodes[at[0]..],
                bones: &mut self.bones[at[1]..],
                muscles: &mut self.muscles[at[2]..],
            };
            self.genomes[k] = part.put(&child).expect("room was made");
            at = [
                at[0] + child.nodes.len(),
                at[1] + child.bones.len(),
                at[2] + child.muscles.len(),
            ];
        }
        late
    }
}

fn local_mutation(mut creature: Creature, cfg: &Config, rng: &mut Rng, scale: f32) -> Creature {
    mutate_genes(&mut creature, cfg, rng, scale);
    creature
}

/// Where each gene's noise is keyed (`Genes`): node `i` field `f` at
/// `4 i + f`, bone `i` at `BONE_GENES + 8 i`, the body tempo at `TEMPO_GENE`,
/// muscle `i` at `MUSCLE_GENES + 16 i`.
const BONE_GENES: u32 = 4 * MAX_NODES as u32;
const TEMPO_GENE: u32 = BONE_GENES + 8 * MAX_NODES as u32;
const MUSCLE_GENES: u32 = TEMPO_GENE + 128;

/// Gaussian noise on every gene at `scale` (times the config's mutation
/// strength), clamped to the gene's range. Each gene's noise is keyed by its
/// index, so it does not depend on the body's other genes or on the order
/// they are visited in.
fn mutate_genes(creature: &mut Creature, cfg: &Config, rng: &mut Rng, scale: f32) {
    let scale = scale * cfg.mutation;
    if scale <= 0.0 {
        return;
    }
    let g = rng.genes();
    let extent = body_extent();
    for (i, node) in creature.nodes.iter_mut().enumerate() {
        let at = 4 * i as u32;
        node.x = (node.x + g.gaussian(at) * 0.10 * scale).clamp(-extent, extent);
        node.y = (node.y + g.gaussian(at + 1) * 0.08 * scale).clamp(0.0, extent);
        node.diameter =
            (node.diameter + g.gaussian(at + 2) * 0.025 * scale).clamp(cfg.min_size, cfg.max_size);
        node.friction = (node.friction + g.gaussian(at + 3) * 0.10 * scale)
            .clamp(cfg.min_friction, cfg.max_friction);
    }
    let max_bone = max_bone_length();
    for (i, bone) in creature.bones.iter_mut().enumerate() {
        let at = BONE_GENES + 8 * i as u32;
        bone.rest_length =
            (bone.rest_length + g.gaussian(at) * 0.035 * scale).clamp(0.03, max_bone);
        bone.min_angle += g.gaussian(at + 1) * 0.15 * scale;
        bone.max_angle += g.gaussian(at + 2) * 0.15 * scale;
        bone.clamp_range();
        if bone.organ_mass > 0.0 {
            bone.organ_mass = (bone.organ_mass * (g.gaussian(at + 3) * 0.15 * scale).exp())
                .clamp(MIN_ORGAN_MASS, MAX_ORGAN_MASS);
            bone.organ_at = (bone.organ_at + g.gaussian(at + 4) * 0.10 * scale).clamp(0.0, 1.0);
        }
    }
    // The body's clock speeds up or slows down as a whole.
    let tempo = (g.gaussian(TEMPO_GENE) * 0.10 * scale).exp();
    let (min_period, stroke) = (min_muscle_period(), max_stroke());
    let rare = scale.min(1.0);
    for (i, muscle) in creature.muscles.iter_mut().enumerate() {
        let at = MUSCLE_GENES + 16 * i as u32;
        muscle.anchor_a = (muscle.anchor_a + g.gaussian(at) * 0.10 * scale).clamp(0.0, 1.0);
        muscle.anchor_b = (muscle.anchor_b + g.gaussian(at + 1) * 0.10 * scale).clamp(0.0, 1.0);
        muscle.short = (muscle.short + g.gaussian(at + 2) * 0.06 * scale).clamp(0.01, 0.8 * stroke);
        muscle.long = (muscle.long + g.gaussian(at + 3) * 0.08 * scale).clamp(muscle.short, stroke);
        muscle.period = (muscle.period * tempo).clamp(min_period, 10.0);
        muscle.phase = (muscle.phase + g.gaussian(at + 4) * 0.12 * scale).rem_euclid(1.0);
        muscle.duty = (muscle.duty + g.gaussian(at + 5) * 0.08 * scale).clamp(0.05, 0.95);
        muscle.stiffness =
            (muscle.stiffness * (g.gaussian(at + 6) * 0.10 * scale).exp()).clamp(1.0, 120.0);
        muscle.reset = (muscle.reset + g.gaussian(at + 7) * 0.12 * scale).rem_euclid(1.0);
        // The elastic tendon grows in, tunes, or drops out.
        if g.unit(at + 8, 0) < 0.10 * rare {
            muscle.tendon = if muscle.tendon == 0.0 {
                0.05 + g.unit(at + 8, 1) * 0.45
            } else if g.unit(at + 8, 2) < 0.2 {
                0.0
            } else {
                (muscle.tendon + g.gaussian(at + 9) * 0.2).clamp(0.0, 1.0)
            };
        }
        if g.unit(at + 10, 0) < 0.05 * rare {
            muscle.sensor = match g.index(at + 10, 1, 5) {
                4 => NO_SENSOR,
                endpoint => endpoint as u32,
            };
        }
    }
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

/// `structural_mutation_in_place` for a child bred from `archive`. The
/// anatomy operators join the classic ones (and graft limbs from another
/// elite of the archive).
/// Returns the operator that changed the body, as an index into
/// `structural_operator_names`, or `None` when none fit.
fn structural_mutation_from(
    creature: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    archive: &QdArchive,
) -> Option<u8> {
    structural_mutation_among(creature, cfg, rng, &archive.entries)
}

/// A structural mutation with no archive at hand (the refuge of old
/// champions after a world change), repaired as breeding does. Returns
/// whether the body changed.
pub fn structural_mutation_any(creature: &mut Creature, cfg: &Config, rng: &mut Rng) -> bool {
    let changed = structural_mutation_among(creature, cfg, rng, &[]).is_some();
    if changed {
        repair(creature, cfg, rng);
    }
    changed
}

fn structural_mutation_among(
    creature: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    donors: &[crate::qd::Elite],
) -> Option<u8> {
    let extra = anatomy::enabled();
    let donor = (!donors.is_empty()).then(|| &donors[rng.index(donors.len())].creature);
    let cx = anatomy::Context { donor };
    let classic = CLASSIC_COUNT;
    // An operator that does not fit this body leaves it unchanged; try
    // another, a few times.
    // Each shared group takes one slot, drawn after the others.
    let groups: Bounded<&Vec<usize>, 16> = [&extra.shared, &extra.controller]
        .into_iter()
        .chain(&extra.gait)
        .filter(|group| !group.is_empty())
        .collect();
    let slots = classic + extra.single.len() + groups.len();
    for _ in 0..4 {
        let pick = rng.index(slots);
        let operator = if pick < classic {
            pick
        } else if let Some(&index) = extra.single.get(pick - classic) {
            classic + index
        } else {
            let group = groups[pick - classic - extra.single.len()];
            classic + group[rng.index(group.len())]
        };
        let changed = if operator < classic {
            classic_operator(operator, creature, cfg, rng)
        } else {
            anatomy::apply(operator - classic, creature, cfg, rng, &cx)
        };
        if changed {
            return Some(operator as u8);
        }
    }
    None
}

/// Names of the classic structural operators, in `classic_operator` order.
const CLASSIC_OPERATORS: [&str; CLASSIC_COUNT] = [
    "split_bone",
    "duplicate_mirrored_node",
    "duplicate_limb",
    "retime_rhythm",
    "change_organ",
    "phase_shift_group",
    "rescale_body",
];
const CLASSIC_COUNT: usize = 7;

/// Every structural operator by name: the classic ones, then the anatomy
/// operators. For diagnostics such as `examples/mutation_audit.rs`.
pub fn structural_operator_names() -> Vec<&'static str> {
    let mut names: Vec<&str> = CLASSIC_OPERATORS.to_vec();
    names.extend(anatomy::OPERATORS.iter().map(|(name, _)| *name));
    names
}

/// Applies the structural operator `name` and repairs the body as breeding
/// does. Returns whether the operator changed the body, or `None` for an
/// unknown name.
pub fn apply_structural_operator(
    name: &str,
    creature: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    donor: Option<&Creature>,
) -> Option<bool> {
    let changed = if let Some(pick) = CLASSIC_OPERATORS.iter().position(|n| *n == name) {
        classic_operator(pick, creature, cfg, rng)
    } else {
        let index = anatomy::OPERATORS.iter().position(|(n, _)| *n == name)?;
        let cx = anatomy::Context { donor };
        anatomy::apply(index, creature, cfg, rng, &cx)
    };
    if changed {
        repair(creature, cfg, rng);
    }
    Some(changed)
}

/// The small parameter mutation that follows every structural one in
/// breeding (`local_mutation` at `scale`), for diagnostics.
pub fn mutate_locally(creature: Creature, cfg: &Config, rng: &mut Rng, scale: f32) -> Creature {
    let mut child = local_mutation(creature, cfg, rng, scale);
    repair(&mut child, cfg, rng);
    child
}

fn classic_operator(pick: usize, creature: &mut Creature, cfg: &Config, rng: &mut Rng) -> bool {
    match pick {
        0 => split_bone(creature, cfg, rng),
        1 => duplicate_mirrored_node(creature, cfg, rng),
        2 => duplicate_limb(creature, cfg, rng),
        3 => retime_rhythm(creature, rng),
        4 => change_organ(creature, rng),
        5 => phase_shift_group(creature, rng),
        _ => rescale_body(creature, rng),
    }
}

/// Grows or shrinks the whole body. Lengths scale by `s` and the rhythm slows
/// by `sqrt(s)`, as for animals of similar build under the same gravity, so
/// the gait roughly carries over while the stride scales with the body.
fn rescale_body(creature: &mut Creature, rng: &mut Rng) -> bool {
    let longest_bone = creature
        .bones
        .iter()
        .map(|b| b.rest_length)
        .fold(0.0, f32::max);
    let longest_muscle = creature.muscles.iter().map(|m| m.long).fold(0.0, f32::max);
    if longest_bone <= 0.0 {
        return false;
    }
    // Stay within the body limits instead of distorting the shape.
    let most = (max_bone_length() / longest_bone)
        .min(if longest_muscle > 0.0 {
            max_stroke() / longest_muscle
        } else {
            f32::INFINITY
        })
        .min(1.5);
    let scale = rng.range(0.75f32.ln(), 1.5f32.ln()).exp().min(most);
    if (scale - 1.0).abs() < 0.02 {
        return false;
    }
    let center = creature.nodes.iter().map(|n| n.x).sum::<f32>() / creature.nodes.len() as f32;
    for node in &mut creature.nodes {
        node.x = center + (node.x - center) * scale;
        node.y *= scale;
    }
    for bone in &mut creature.bones {
        bone.rest_length *= scale;
    }
    let tempo = scale.sqrt();
    for muscle in &mut creature.muscles {
        muscle.short *= scale;
        muscle.long *= scale;
        muscle.period *= tempo;
        // Muscle force follows its target's speed, which grows by sqrt(s).
        muscle.stiffness /= tempo;
    }
    true
}

fn split_bone(creature: &mut Creature, cfg: &Config, rng: &mut Rng) -> bool {
    if creature.nodes.len() >= cfg.max_nodes
        || creature.bones.is_empty()
        || creature.bones.iter().all(|bone| bone.rest_length < 0.06)
    {
        return false;
    }
    let eligible: Bounded<usize, MAX_NODES> = creature
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
        ..original
    };
    let mut second = Bone::new(mid, original.b, original.rest_length - first_length);
    // The organ stays where it was, on whichever half now holds that point.
    if original.organ_mass > 0.0 {
        if original.organ_at <= 0.5 {
            creature.bones[index].organ_at = original.organ_at * 2.0;
        } else {
            creature.bones[index].organ_mass = 0.0;
            creature.bones[index].organ_at = 0.5;
            second.organ_mass = original.organ_mass;
            second.organ_at = (original.organ_at - 0.5) * 2.0;
        }
    }
    creature.bones.push(second);
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
    duplicate.x = (2.0 * center_x - duplicate.x + rng.range(-0.03, 0.03))
        .clamp(-body_extent(), body_extent());
    duplicate.y = (duplicate.y + rng.range(-0.03, 0.03)).clamp(0.0, body_extent());
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
pub fn ranking(scores: &[f32]) -> Vec<usize> {
    let mut ranks: Vec<_> = (0..scores.len()).collect();
    ranks.par_sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subset_copies_each_creature_in_index_order() {
        let cfg = Config {
            population: 10_000,
            random_seed: false,
            seed: 9,
            ..Config::default()
        };
        let population = create(&cfg).unwrap();
        // Scrambled order with repeats, longer than one copy chunk.
        let indices: Vec<usize> = (0..9_000).map(|k| (k * 7919 + 13) % 10_000).collect();
        let subset = population.subset(&indices);
        assert_eq!(subset.genomes.len(), indices.len());
        for (k, &i) in indices.iter().enumerate() {
            let (a, b) = (subset.creature(k), population.creature(i));
            assert_eq!(a.nodes, b.nodes);
            assert_eq!(a.bones, b.bones);
            assert_eq!(a.muscles, b.muscles);
            assert_eq!(a.id, b.id);
        }
        assert!(population.subset(&[]).genomes.is_empty());
    }

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
            ]
            .into(),
            bones: vec![Bone::new(0, 1, 9.6)].into(),
            muscles: vec![].into(),
            id: 0,
        };
        normalize_bone_lengths(&mut creature);
        assert_eq!(creature.bones[0].rest_length, 0.625);
        creature.nodes[1].x = 2.0 * max_bone_length();
        normalize_bone_lengths(&mut creature);
        assert_eq!(creature.bones[0].rest_length, max_bone_length());
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
            ]
            .into(),
            bones: vec![Bone::new(0, 1, 2.0), Bone::new(1, 2, 2.0)].into(),
            muscles: vec![].into(),
            id: 0,
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
                .all(|m| m.period >= min_muscle_period())
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
                nodes: nodes.into(),
                bones: vec![Bone::new(0, 1, 1.0), Bone::new(1, 2, 1.0)].into(),
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
                        tendon: 0.0,
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
                        tendon: 0.0,
                    },
                ]
                .into(),
                id: 1,
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
                Bone::new(2, 3, 1.0),
                Bone::new(1, 0, 1.0),
                Bone::new(2, 1, 1.0),
            ]
            .into(),
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
                tendon: 0.0,
            }]
            .into(),
            id: 1,
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

    fn organ_point(creature: &Creature) -> Option<[f32; 2]> {
        creature.bones.iter().find(|b| b.organ_mass > 0.0).map(|b| {
            let a = creature.nodes[b.a as usize];
            let c = creature.nodes[b.b as usize];
            [
                a.x + (c.x - a.x) * b.organ_at,
                a.y + (c.y - a.y) * b.organ_at,
            ]
        })
    }

    #[test]
    fn organs_stay_near_the_body_center_after_every_operator() {
        let cfg = Config::default();
        for index in 0..400 {
            let mut rng = Rng::new(7, 3, index);
            let mut creature = random_creature_from(&cfg, &mut rng);
            for _ in 0..12 {
                if rng.unit() < 0.5 {
                    change_organ(&mut creature, &mut rng);
                }
                let pick = rng.index(CLASSIC_COUNT);
                let _ = classic_operator(pick, &mut creature, &cfg, &mut rng);
                creature = local_mutation(creature, &cfg, &mut rng, 0.75);
                repair(&mut creature, &cfg, &mut rng);
                let center = organ_center(&creature.nodes);
                for bone in &creature.bones {
                    if bone.organ_mass == 0.0 {
                        continue;
                    }
                    assert!((MIN_ORGAN_MASS..=MAX_ORGAN_MASS).contains(&bone.organ_mass));
                    let a = creature.nodes[bone.a as usize];
                    let b = creature.nodes[bone.b as usize];
                    let p = [
                        a.x + (b.x - a.x) * bone.organ_at,
                        a.y + (b.y - a.y) * bone.organ_at,
                    ];
                    let distance = (p[0] - center[0]).hypot(p[1] - center[1]);
                    assert!(
                        distance <= ORGAN_RADIUS + 1e-3,
                        "organ {distance} m from center"
                    );
                }
            }
        }
    }

    #[test]
    fn organs_can_be_grown_and_keep_their_place_through_splits_and_reordering() {
        let cfg = Config::default();
        let mut grown = 0;
        for index in 0..200 {
            let mut rng = Rng::new(11, 0, index);
            let mut creature = random_creature_from(&cfg, &mut rng);
            if !change_organ(&mut creature, &mut rng) {
                continue;
            }
            let Some(before) = organ_point(&creature) else {
                continue;
            };
            grown += 1;
            let organ_bone = creature
                .bones
                .iter()
                .position(|b| b.organ_mass > 0.0)
                .unwrap();
            // Split exactly the organ's bone and check the organ did not move.
            let mut split = creature.clone();
            let others: Vec<f32> = split.bones.iter().map(|b| b.rest_length).collect();
            for (i, b) in split.bones.iter_mut().enumerate() {
                if i != organ_bone {
                    b.rest_length = 0.01;
                }
            }
            if split.bones[organ_bone].rest_length >= 0.06 && split_bone(&mut split, &cfg, &mut rng)
            {
                for (b, &length) in split.bones.iter_mut().zip(&others) {
                    if b.rest_length == 0.01 {
                        b.rest_length = length;
                    }
                }
                let after = organ_point(&split).unwrap();
                assert!((after[0] - before[0]).abs() < 1e-5 && (after[1] - before[1]).abs() < 1e-5);
            }
            // Reversing bone direction keeps the organ at the same point.
            let mut reordered = creature.clone();
            for b in &mut reordered.bones {
                std::mem::swap(&mut b.a, &mut b.b);
                b.organ_at = 1.0 - b.organ_at;
            }
            assert!(canonicalize_bone_order(&mut reordered));
            let after = organ_point(&reordered).unwrap();
            assert!((after[0] - before[0]).abs() < 1e-5 && (after[1] - before[1]).abs() < 1e-5);
        }
        assert!(grown > 100, "only {grown} creatures could grow an organ");
    }

    #[test]
    fn tuning_grows_tendons_and_keeps_them_valid() {
        let cfg = Config::default();
        let mut with_tendon = 0;
        let mut total = 0;
        for index in 0..300 {
            let mut rng = Rng::new(21, 0, index);
            let mut creature = random_creature_from(&cfg, &mut rng);
            assert!(creature.muscles.iter().all(|m| m.tendon == 0.0));
            for _ in 0..10 {
                creature = local_mutation(creature, &cfg, &mut rng, 1.0);
            }
            for m in &creature.muscles {
                assert!((0.0..=1.0).contains(&m.tendon), "tendon {}", m.tendon);
                total += 1;
                with_tendon += usize::from(m.tendon > 0.0);
            }
        }
        // Ten tuning steps at a 10% rate per muscle: a few in ten muscles.
        let share = with_tendon as f32 / total as f32;
        assert!((0.15..0.75).contains(&share), "share with a tendon {share}");
    }

    #[test]
    fn rescaling_scales_lengths_slows_the_rhythm_and_respects_limits() {
        let cfg = Config::default();
        let mut rescaled = 0;
        for index in 0..200 {
            let mut rng = Rng::new(13, 0, index);
            let before = random_creature_from(&cfg, &mut rng);
            let mut after = before.clone();
            if !rescale_body(&mut after, &mut rng) {
                continue;
            }
            rescaled += 1;
            let scale = after.bones[0].rest_length / before.bones[0].rest_length;
            assert!((0.74..=1.51).contains(&scale), "scale {scale}");
            for (a, b) in after.bones.iter().zip(&before.bones) {
                assert!((a.rest_length - b.rest_length * scale).abs() < 1e-4);
                assert!(a.rest_length <= max_bone_length() + 1e-4);
            }
            for (a, b) in after.muscles.iter().zip(&before.muscles) {
                assert!((a.long - b.long * scale).abs() < 1e-4);
                assert!(a.long <= max_stroke() + 1e-4);
                assert!((a.period - b.period * scale.sqrt()).abs() < 1e-4);
            }
            // The shape is kept: every node keeps its place relative to the
            // body's horizontal center, scaled.
            let center =
                |c: &Creature| c.nodes.iter().map(|n| n.x).sum::<f32>() / c.nodes.len() as f32;
            let (ca, cb) = (center(&after), center(&before));
            for (a, b) in after.nodes.iter().zip(&before.nodes) {
                assert!((a.x - ca - (b.x - cb) * scale).abs() < 1e-4);
                assert!((a.y - b.y * scale).abs() < 1e-4);
            }
        }
        assert!(rescaled > 150, "only {rescaled} creatures rescaled");
    }

    #[test]
    fn organ_mass_is_shared_by_its_bone_nodes_at_its_position() {
        let cfg = Config::default();
        let mut rng = Rng::new(5, 0, 0);
        let mut creature = random_creature_from(&cfg, &mut rng);
        let bone = creature.bones[1];
        creature.bones[1].organ_mass = 0.2;
        creature.bones[1].organ_at = 0.25;
        let mut bare = creature.bones;
        bare[1].organ_mass = 0.0;
        let plain = crate::physics::body(&creature.nodes, &bare);
        let with = crate::physics::body(&creature.nodes, &creature.bones);
        let mass = |n: &[crate::physics::Node]| n.iter().map(|x| x.mass).sum::<f32>();
        assert!((mass(&with) - mass(&plain) - 0.2).abs() < 1e-6);
        let com =
            |n: &[crate::physics::Node]| n.iter().map(|x| x.pos[0] * x.mass).sum::<f32>() / mass(n);
        let a = creature.nodes[bone.a as usize];
        let b = creature.nodes[bone.b as usize];
        let organ_x = a.x + (b.x - a.x) * 0.25;
        let expected = (com(&plain) * mass(&plain) + organ_x * 0.2) / (mass(&plain) + 0.2);
        assert!((com(&with) - expected).abs() < 1e-5);
    }

    #[test]
    fn bone_mass_grows_with_the_square_of_its_length() {
        let cfg = Config::default();
        let mut rng = Rng::new(9, 0, 0);
        let mut creature = random_creature_from(&cfg, &mut rng);
        for bone in &mut creature.bones {
            bone.organ_mass = 0.0;
        }
        let mass = |bones: &[Bone]| {
            crate::physics::body(&creature.nodes, bones)
                .iter()
                .map(|n| n.mass)
                .sum::<f32>()
        };
        let nodes_only = mass(&[]);
        let expected: f32 = creature
            .bones
            .iter()
            .map(|b| crate::physics::limits().bone_density * b.rest_length * b.rest_length)
            .sum();
        assert!((mass(&creature.bones) - nodes_only - expected).abs() < 1e-4);
        // Doubling every bone quadruples the bone mass.
        let doubled: Vec<Bone> = creature
            .bones
            .iter()
            .map(|b| Bone {
                rest_length: b.rest_length * 2.0,
                ..*b
            })
            .collect();
        assert!((mass(&doubled) - nodes_only - 4.0 * expected).abs() < 1e-3);
    }

    #[test]
    fn twelve_uniform_gaussians_have_unit_variance() {
        let mut rng = Rng::stream(1, 2, 3, 4);
        let genes = rng.genes();
        for draw in [
            (0..200_000).map(|_| rng.gaussian()).collect::<Vec<f32>>(),
            (0..200_000).map(|g| genes.gaussian(g)).collect(),
        ] {
            let n = draw.len() as f64;
            let mean = draw.iter().map(|&x| x as f64).sum::<f64>() / n;
            let variance = draw.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
            assert!(mean.abs() < 0.01, "mean {mean}");
            assert!((variance - 1.0).abs() < 0.02, "variance {variance}");
            assert!(draw.iter().all(|x| x.abs() <= 6.0));
        }
    }

    #[test]
    fn gene_noise_does_not_depend_on_other_draws() {
        // A gene's noise is keyed by its index: it is the same whichever
        // genes were drawn before it, and streams of other slots differ.
        let a = Rng::stream(5, 1, 2, 3).genes();
        let b = Rng::stream(5, 1, 2, 3).genes();
        let forward: Vec<f32> = (0..64).map(|g| a.gaussian(g)).collect();
        let backward: Vec<f32> = (0..64).rev().map(|g| b.gaussian(g)).collect();
        assert!(forward.iter().eq(backward.iter().rev()));
        let other = Rng::stream(5, 1, 2, 4).genes();
        assert!((0..64).any(|g| other.gaussian(g) != a.gaussian(g)));
        // The same child from its slot's stream, bred twice.
        let cfg = Config::default();
        let body = random_creature_from(&cfg, &mut Rng::new(1, 0, 0));
        let mut x = body.clone();
        let mut y = body.clone();
        mutate_genes(&mut x, &cfg, &mut Rng::stream(9, 3, 7, 11), 1.0);
        mutate_genes(&mut y, &cfg, &mut Rng::stream(9, 3, 7, 11), 1.0);
        assert_eq!(x.nodes, y.nodes);
        assert_eq!(x.muscles, y.muscles);
    }

    #[test]
    fn the_growth_step_limits_what_a_child_gains() {
        let cfg = Config::default();
        let step = GrowthStep {
            nodes: 4,
            muscles: 4,
        };
        let archive = QdArchive::default();
        let mut grew = 0;
        for index in 0..300 {
            let mut rng = Rng::new(17, 0, index);
            let mut parent = random_creature_from(&cfg, &mut rng);
            grow_for_benchmark(&mut parent, &cfg, index as u64, 3 + index % 12);
            let limited = child_limits(&cfg, &parent, Some(step));
            let fit = limited.as_ref().unwrap_or(&cfg);
            let mut child = parent.clone();
            for _ in 0..12 {
                let _ = structural_mutation_from(&mut child, fit, &mut rng, &archive);
                repair(&mut child, fit, &mut rng);
            }
            assert!(child.nodes.len() <= parent.nodes.len() + 4);
            assert!(child.muscles.len() <= parent.muscles.len() + 4);
            grew += usize::from(child.nodes.len() > parent.nodes.len());
        }
        assert!(grew > 50, "only {grew} children grew");
        assert!(
            child_limits(
                &cfg,
                &Creature::clone(&random_creature_from(&cfg, &mut Rng::new(1, 1, 1))),
                None
            )
            .is_none()
        );
    }

    #[test]
    fn crossing_different_body_plans_grafts_a_limb_and_stays_valid() {
        let cfg = Config::default();
        let mut parent = random_creature_from(&cfg, &mut Rng::new(1, 0, 0));
        grow_for_benchmark(&mut parent, &cfg, 3, 5);
        let mut mate = random_creature_from(&cfg, &mut Rng::new(1, 0, 1));
        grow_for_benchmark(&mut mate, &cfg, 4, 9);
        assert!(!same_shape(&parent, &mate));
        assert!(same_shape(&parent, &parent));
        let mut grafted = 0;
        for i in 0..40 {
            let mut child = parent.clone();
            let mut rng = Rng::new(5, 0, i);
            if anatomy::graft_from(&mut child, &cfg, &mut rng, &mate) {
                grafted += 1;
                repair(&mut child, &cfg, &mut rng);
                assert!(child.nodes.len() <= cfg.max_nodes);
                assert!(child.muscles.len() <= cfg.max_muscles);
            }
        }
        assert!(grafted >= 10, "grafted {grafted} of 40");
    }
}
