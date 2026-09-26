use crate::evolution::{Creature, Muscle, Population, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

pub const EMITTER_COUNT: usize = 4;
/// Archive grid: ground contact, gait cadence, vertical bounce, mean body
/// height, and feet (distinct nodes that touched the ground). Bounce keeps a
/// single bin: fewer cells give each one more offspring, which found faster
/// creatures in fixed-seed tests (docs/search-research.md).
const BINS: [u8; 5] = [6, 8, 1, 6, 5];
pub(crate) const ARCHIVE_LIMIT: usize = 6 * 8 * 6 * 5;
pub(crate) const MORPHOLOGY_LIMIT: usize = 64;
pub(crate) const ARCHIVE_CAPACITY: usize = ARCHIVE_LIMIT + MORPHOLOGY_LIMIT;
pub(crate) const HISTORICAL_ARCHIVE_LIMIT: usize = 1 << 20;
pub(crate) const CMA_LIMIT: usize = 96;
pub const VERSION: u32 = 22;
const LOCAL_NEIGHBORS: usize = 5;
const MORPHOLOGY_NICHE_MARKER: u8 = u8::MAX;
/// First byte of an optimizer's niche; behavior niches never reach it and
/// morphology niches use 255.
const OPTIMIZER_NICHE_MARKER: u8 = 254;
/// The niche key of an island's optimizers for one gait cadence band;
/// together with the body plan it identifies one optimizer.
pub fn optimizer_niche(island: usize, cadence: u8) -> Niche {
    let b = (island as u32).to_le_bytes();
    Niche([OPTIMIZER_NICHE_MARKER, b[0], b[1], b[2], b[3], cadence])
}
pub(crate) const MIN_MORPHOLOGY_DESCENDANTS: u64 = 8;
pub(crate) const MORPHOLOGY_PARENT_FRACTION: f32 = 0.10;
// Random immigrants only seed an empty archive: against evolved elites they
// almost never enter it (0.03-0.06% of attempts in fixed-seed tests).
const INITIAL_EMITTER_MIX: [f64; EMITTER_COUNT] = [0.35, 0.35, 0.30, 0.0];

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TrialMetrics {
    pub ground_contact: f32,
    pub vertical_oscillation: f32,
    pub gait_frequency: f32,
    /// Mean height of the body's bounding box during the timed trial (m).
    pub mean_height: f32,
    /// Distinct nodes that touched the ground.
    pub feet: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvaluationMetrics {
    pub fitness: f32,
    pub behavior: TrialMetrics,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Descriptor {
    pub nodes: u16,
    pub muscles: u16,
    pub ground_contact: f32,
    pub gait_frequency: f32,
    pub aspect_ratio: f32,
    pub vertical_oscillation: f32,
    #[serde(default)]
    pub mean_height: f32,
    #[serde(default)]
    pub feet: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Niche(pub [u8; 6]);

pub fn is_morphology_niche(niche: &Niche) -> bool {
    niche.0[0] == MORPHOLOGY_NICHE_MARKER
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Elite {
    pub niche: Niche,
    pub descriptor: Descriptor,
    pub creature: Creature,
    pub fitness: f32,
    pub emitter: Emitter,
    pub improved_generation: u32,
    pub protected_until: u32,
    pub visits: u64,
    pub topology: Topology,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QdArchive {
    pub entries: Vec<Elite>,
    #[serde(skip)]
    lookup: HashMap<Niche, usize>,
    pub qd_score: f64,
    #[serde(skip)]
    least_visited: BTreeSet<(u64, usize)>,
    #[serde(skip)]
    behavior_indices: Vec<usize>,
    #[serde(skip)]
    morphology_indices: Vec<usize>,
    #[serde(skip)]
    behavior_scores: BehaviorScores,
}

#[derive(Clone, Debug, Default)]
struct BehaviorScores {
    novelty: Vec<f32>,
    local_competition: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Topology {
    pub nodes: u8,
    pub edges: Vec<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Emitter {
    #[default]
    Cma,
    Structural,
    Novelty,
    Restart,
}
impl Emitter {
    pub const ALL: [Self; EMITTER_COUNT] =
        [Self::Cma, Self::Structural, Self::Novelty, Self::Restart];
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn from_index(index: usize) -> Self {
        Self::ALL[index.min(EMITTER_COUNT - 1)]
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Cma => "Diagonal CMA-ES",
            Self::Structural => "Morphology",
            Self::Novelty => "Novelty",
            Self::Restart => "Immigrant",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EmitterStats {
    pub attempts: u64,
    pub discoveries: u64,
    pub improvements: u64,
    pub reward: f64,
    pub stagnant_batches: u32,
    pub last_parent: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CmaEmitter {
    pub niche: Niche,
    pub topology: Topology,
    template: Creature,
    mean: Vec<f32>,
    covariance: Vec<f32>,
    path_c: Vec<f32>,
    path_sigma: Vec<f32>,
    sigma: f32,
    pub last_used_generation: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Offer {
    pub inserted: bool,
    pub new_niche: bool,
    pub reward: f64,
}

pub fn descriptor(
    nodes: &[crate::evolution::NodeGene],
    muscles: &[Muscle],
    metrics: TrialMetrics,
) -> Descriptor {
    let (min_x, max_x) = nodes
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), n| {
            (lo.min(n.x), hi.max(n.x))
        });
    let (min_y, max_y) = nodes
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), n| {
            (lo.min(n.y), hi.max(n.y))
        });
    Descriptor {
        nodes: nodes.len() as u16,
        muscles: muscles.len() as u16,
        ground_contact: metrics.ground_contact.clamp(0.0, 1.0),
        gait_frequency: metrics.gait_frequency.clamp(0.0, 20.0),
        aspect_ratio: ((max_x - min_x).max(0.01) / (max_y - min_y).max(0.01)).clamp(0.0625, 16.0),
        vertical_oscillation: metrics.vertical_oscillation.max(0.0),
        mean_height: metrics.mean_height.max(0.0),
        feet: metrics.feet.max(0.0),
    }
}

impl Descriptor {
    pub fn niche(self) -> Niche {
        // Only measured behavior determines archive cells. Morphology remains
        // attached to each descriptor for display and analysis.
        let feet = (self.feet.round() as i32).clamp(1, BINS[4] as i32) as u8 - 1;
        Niche([
            bin(self.ground_contact, 0.0, 1.0, BINS[0]),
            bin(self.gait_frequency, 0.0, 6.0, BINS[1]),
            bin(self.vertical_oscillation, 0.0, 0.8, BINS[2]),
            bin(height_axis(self.mean_height), 0.0, 1.0, BINS[3]),
            feet,
            0,
        ])
    }

    fn behavior(self) -> [f32; 5] {
        [
            self.ground_contact.clamp(0.0, 1.0),
            (self.gait_frequency / 6.0).clamp(0.0, 1.0),
            // Bounce is not an archive axis, so it adds no novelty either.
            0.0,
            height_axis(self.mean_height),
            ((self.feet - 1.0) / (BINS[4] as f32 - 1.0)).clamp(0.0, 1.0),
        ]
    }
}
/// Mean height on a 0..1 log scale from 15 cm to the tallest bodies the
/// bone limit allows, so small and large bodies each get their own cells.
fn height_axis(height: f32) -> f32 {
    let low = 0.15f32;
    let high = (0.6 * crate::evolution::max_bone_length()).max(2.0 * low);
    ((height.max(low) / low).ln() / (high / low).ln()).clamp(0.0, 1.0)
}
fn bin(value: f32, low: f32, high: f32, count: u8) -> u8 {
    (((value.clamp(low, high) - low) / (high - low) * count as f32).floor() as u8).min(count - 1)
}

/// Behavior niches within `radius` grid steps of `center` (itself excluded).
fn neighbor_niches(center: &Niche, radius: i32) -> impl Iterator<Item = Niche> + '_ {
    let side = (2 * radius + 1) as usize;
    let total = side.pow(BINS.len() as u32);
    (0..total).filter_map(move |mut code| {
        let mut cell = [0u8; 6];
        let mut moved = false;
        for (axis, &bins) in BINS.iter().enumerate() {
            let offset = (code % side) as i32 - radius;
            code /= side;
            moved |= offset != 0;
            let value = center.0[axis] as i32 + offset;
            if !(0..bins as i32).contains(&value) {
                return None;
            }
            cell[axis] = value as u8;
        }
        moved.then_some(Niche(cell))
    })
}

fn behavior_distance(a: Descriptor, b: Descriptor) -> f32 {
    let a = a.behavior();
    let b = b.behavior();
    ((0..a.len()).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
}

impl Topology {
    pub fn of(creature: &Creature) -> Self {
        topology_from_parts(&creature.nodes, &creature.bones, &creature.muscles)
    }
}

fn topology_from_parts(
    nodes: &[crate::evolution::NodeGene],
    bones: &[crate::evolution::Bone],
    muscles: &[Muscle],
) -> Topology {
    let offset = nodes.len() as u32;
    let mut edges = Vec::with_capacity(bones.len() + muscles.len());
    edges.extend(bones.iter().map(|b| (b.a.min(b.b), b.a.max(b.b))));
    edges.extend(muscles.iter().map(|m| {
        let a = offset + m.bone_a;
        let b = offset + m.bone_b;
        (a.min(b), a.max(b))
    }));
    edges.sort_unstable();
    Topology {
        nodes: nodes.len() as u8,
        edges,
    }
}

fn topology_equivalent(a: &Topology, b: &Topology) -> bool {
    a == b
}

pub fn topology_equivalent_for_archive(a: &Topology, b: &Topology) -> bool {
    topology_equivalent(a, b)
}

fn morphology_hash(topology: &Topology, salt: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut write = |byte: u8| {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    write(topology.nodes);
    for &(a, b) in &topology.edges {
        for byte in a.to_le_bytes().into_iter().chain(b.to_le_bytes()) {
            write(byte);
        }
    }
    for byte in salt.to_le_bytes() {
        write(byte);
    }
    hash
}

fn morphology_niche(topology: &Topology, salt: u64) -> Niche {
    let hash = morphology_hash(topology, salt);
    Niche([
        MORPHOLOGY_NICHE_MARKER,
        hash as u8,
        (hash >> 8) as u8,
        (hash >> 16) as u8,
        (hash >> 24) as u8,
        (hash >> 32) as u8,
    ])
}

impl QdArchive {
    pub fn rebuild_indices(&mut self) {
        self.lookup.clear();
        self.least_visited.clear();
        self.behavior_indices.clear();
        self.morphology_indices.clear();
        for (i, elite) in self.entries.iter_mut().enumerate() {
            elite.topology.edges.sort_unstable();
            self.lookup.insert(elite.niche.clone(), i);
            self.least_visited.insert((elite.visits, i));
            if is_morphology_niche(&elite.niche) {
                self.morphology_indices.push(i);
            } else {
                self.behavior_indices.push(i);
            }
        }
        self.recompute_score();
        self.refresh_behavior_scores();
    }
    pub fn best_fitness(&self) -> f32 {
        self.entries
            .iter()
            .map(|e| e.fitness)
            .fold(f32::NEG_INFINITY, f32::max)
    }
    pub fn behavior_count(&self) -> usize {
        self.behavior_indices.len()
    }
    pub(crate) fn slot_for(&self, niche: &Niche) -> Option<usize> {
        self.lookup.get(niche).copied()
    }
    pub fn morphology_count(&self) -> usize {
        self.morphology_indices.len()
    }
    /// Fitness a new topology must beat to enter the full topology reserve,
    /// or `None` while the reserve has room.
    pub(crate) fn morphology_floor(&self) -> Option<f32> {
        (self.morphology_count() >= MORPHOLOGY_LIMIT).then(|| {
            self.morphology_indices
                .iter()
                .map(|&i| &self.entries[i])
                .filter(|elite| elite.visits >= MIN_MORPHOLOGY_DESCENDANTS)
                .map(|elite| elite.fitness)
                .fold(f32::INFINITY, f32::min)
        })
    }
    pub fn coverage(&self) -> f32 {
        self.behavior_count() as f32 / ARCHIVE_LIMIT as f32
    }
    pub fn sample_uniform(&self, rng: &mut Rng) -> Option<usize> {
        (!self.behavior_indices.is_empty())
            .then(|| self.behavior_indices[rng.index(self.behavior_indices.len())])
    }
    pub fn sample_novel(&self, rng: &mut Rng, avoid: Option<usize>) -> Option<usize> {
        if self.behavior_count() == 0 {
            return None;
        }
        let mut selected = None;
        let mut best_score = f32::NEG_INFINITY;
        let mut tied = 0usize;
        for &(_, index) in self
            .least_visited
            .iter()
            .filter(|(_, index)| !is_morphology_niche(&self.entries[*index].niche))
            .take(32)
        {
            if Some(index) == avoid && self.behavior_count() > 1 {
                continue;
            }
            let novelty = self
                .behavior_scores
                .novelty
                .get(index)
                .copied()
                .unwrap_or(0.0);
            let visits = self.entries[index].visits;
            let score = novelty + 0.08 / (1.0 + visits as f32).sqrt();
            if score > best_score {
                selected = Some(index);
                best_score = score;
                tied = 1;
            } else if (score - best_score).abs() <= f32::EPSILON {
                tied += 1;
                if rng.index(tied) == 0 {
                    selected = Some(index);
                }
            }
        }
        selected.or_else(|| {
            (!self.behavior_indices.is_empty())
                .then(|| self.behavior_indices[rng.index(self.behavior_indices.len())])
        })
    }
    pub fn sample_local_competitive(&self, rng: &mut Rng, avoid: Option<usize>) -> Option<usize> {
        let behavior = &self.behavior_indices;
        if behavior.is_empty() {
            return None;
        }
        let candidates = behavior.len().clamp(1, 8);
        let mut selected = None;
        let mut best_score = f32::NEG_INFINITY;
        for _ in 0..candidates {
            let mut index = behavior[rng.index(behavior.len())];
            if Some(index) == avoid && behavior.len() > 1 {
                let ordinal = behavior
                    .iter()
                    .position(|&candidate| candidate == index)
                    .unwrap_or(0);
                index = behavior[(ordinal + 1 + rng.index(behavior.len() - 1)) % behavior.len()];
            }
            let local = self
                .behavior_scores
                .local_competition
                .get(index)
                .copied()
                .unwrap_or(0.5);
            let score = local + rng.unit() * 0.02;
            if score > best_score {
                selected = Some(index);
                best_score = score;
            }
        }
        selected
    }
    pub fn sample_morphology(&self, rng: &mut Rng, avoid: Option<usize>) -> Option<usize> {
        let least_visits = self
            .morphology_indices
            .iter()
            .filter(|&&index| Some(index) != avoid)
            .map(|&index| self.entries[index].visits)
            .min()?;
        let mut selected = None;
        let mut tied = 0usize;
        for &index in &self.morphology_indices {
            if Some(index) == avoid || self.entries[index].visits != least_visits {
                continue;
            }
            tied += 1;
            if rng.index(tied) == 0 {
                selected = Some(index);
            }
        }
        selected
    }
    /// Novelty (mean distance to the nearest archived behaviors) and local
    /// competition (share of those neighbors this elite beats), found through
    /// adjacent grid cells instead of comparing every pair.
    pub fn refresh_behavior_scores(&mut self) {
        use rayon::prelude::*;
        let behavior = &self.behavior_indices;
        if behavior.is_empty() {
            self.behavior_scores = BehaviorScores::default();
            return;
        }
        let scores: Vec<(usize, f32, f32)> = behavior
            .par_iter()
            .map(|&index| {
                let elite = &self.entries[index];
                let mut neighbors: Vec<(f32, f32)> = Vec::new();
                for radius in 1..=2 {
                    neighbors.clear();
                    for niche in neighbor_niches(&elite.niche, radius) {
                        if let Some(&slot) = self.lookup.get(&niche) {
                            let other = &self.entries[slot];
                            neighbors.push((
                                behavior_distance(elite.descriptor, other.descriptor),
                                other.fitness,
                            ));
                        }
                    }
                    if neighbors.len() >= LOCAL_NEIGHBORS {
                        break;
                    }
                }
                if neighbors.is_empty() {
                    return (index, 1.0, 1.0);
                }
                neighbors.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                let nearest = &neighbors[..LOCAL_NEIGHBORS.min(neighbors.len())];
                let novelty = nearest.iter().map(|(d, _)| *d).sum::<f32>() / nearest.len() as f32;
                let local = nearest
                    .iter()
                    .map(|(_, f)| match elite.fitness.total_cmp(f) {
                        std::cmp::Ordering::Greater => 1.0,
                        std::cmp::Ordering::Equal => 0.5,
                        std::cmp::Ordering::Less => 0.0,
                    })
                    .sum::<f32>()
                    / nearest.len() as f32;
                (index, novelty, local)
            })
            .collect();
        let mut novelty = vec![0.0; self.entries.len()];
        let mut local_competition = vec![0.5; self.entries.len()];
        for (index, n, l) in scores {
            novelty[index] = n;
            local_competition[index] = l;
        }
        self.behavior_scores = BehaviorScores {
            novelty,
            local_competition,
        };
    }
    pub fn visit(&mut self, index: usize) {
        let elite = &mut self.entries[index];
        self.least_visited.remove(&(elite.visits, index));
        elite.visits += 1;
        self.least_visited.insert((elite.visits, index));
    }
    #[allow(clippy::too_many_arguments)]
    pub fn offer(
        &mut self,
        population: &Population,
        index: usize,
        descriptor: Descriptor,
        fitness: f32,
        emitter: Emitter,
        generation: u32,
        protected_until: u32,
    ) -> Offer {
        if !fitness.is_finite() || fitness <= crate::evolution::FAILED {
            return Offer::default();
        }
        let niche = descriptor.niche();
        if let Some(&slot) = self.lookup.get(&niche) {
            let current = &self.entries[slot];
            if fitness <= current.fitness {
                return Offer::default();
            }
            let candidate_topology = topology_of_population(population, index);
            if generation < current.protected_until && candidate_topology != current.topology {
                return Offer::default();
            }
            let delta = fitness - current.fitness;
            let previous_fitness = current.fitness;
            let visits = current.visits;
            let old_protection = current.protected_until;
            let local_competition = self.local_competition_for(&niche, fitness);
            let elite = &mut self.entries[slot];
            *elite = Elite {
                niche,
                descriptor,
                creature: population.creature(index),
                fitness,
                emitter,
                improved_generation: generation,
                protected_until: protected_until.max(old_protection),
                visits,
                topology: candidate_topology.clone(),
            };
            self.qd_score += fitness.max(0.0) as f64 - previous_fitness.max(0.0) as f64;
            self.behavior_scores = BehaviorScores::default();
            self.remove_morphology_topology(&candidate_topology, fitness);
            return Offer {
                inserted: true,
                new_niche: false,
                reward: ((delta as f64 / (1.0 + previous_fitness.abs() as f64))
                    * (0.5 + local_competition as f64))
                    .clamp(0.01, 1.0),
            };
        }
        if self.behavior_count() >= ARCHIVE_LIMIT {
            return Offer::default();
        }
        let local_competition = self.local_competition_for(&niche, fitness);
        let topology = topology_of_population(population, index);
        self.qd_score += fitness.max(0.0) as f64;
        self.entries.push(Elite {
            niche: niche.clone(),
            descriptor,
            creature: population.creature(index),
            fitness,
            emitter,
            improved_generation: generation,
            protected_until,
            visits: 0,
            topology: topology.clone(),
        });
        let slot = self.entries.len() - 1;
        self.lookup.insert(niche, slot);
        self.least_visited.insert((0, slot));
        self.behavior_indices.push(slot);
        self.behavior_scores = BehaviorScores::default();
        self.remove_morphology_topology(&topology, fitness);
        Offer {
            inserted: true,
            new_niche: true,
            reward: 0.5 + local_competition as f64 * 0.5,
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn offer_morphology(
        &mut self,
        population: &Population,
        index: usize,
        descriptor: Descriptor,
        topology: Topology,
        fitness: f32,
        emitter: Emitter,
        generation: u32,
        protected_until: u32,
    ) -> Offer {
        if !fitness.is_finite() || fitness <= crate::evolution::FAILED {
            return Offer::default();
        }
        let behavior_best = self
            .entries
            .iter()
            .filter(|elite| {
                !is_morphology_niche(&elite.niche)
                    && topology_equivalent(&topology, &elite.topology)
            })
            .map(|elite| elite.fitness)
            .max_by(f32::total_cmp);
        if behavior_best.is_some_and(|best| fitness <= best) {
            return Offer::default();
        }
        if let Some(slot) = self.entries.iter().position(|elite| {
            is_morphology_niche(&elite.niche) && topology_equivalent(&topology, &elite.topology)
        }) {
            let current = &self.entries[slot];
            if fitness <= current.fitness {
                return Offer::default();
            }
            let previous_fitness = current.fitness;
            let visits = current.visits;
            let niche = current.niche.clone();
            self.entries[slot] = Elite {
                niche,
                descriptor,
                creature: population.creature(index),
                fitness,
                emitter,
                improved_generation: generation,
                protected_until: protected_until.max(current.protected_until),
                visits,
                topology,
            };
            return Offer {
                inserted: true,
                new_niche: false,
                reward: ((fitness - previous_fitness) as f64
                    / (1.0 + previous_fitness.abs() as f64))
                    .clamp(0.01, 1.0),
            };
        }

        if self.morphology_count() >= MORPHOLOGY_LIMIT {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, elite)| {
                    is_morphology_niche(&elite.niche) && elite.visits >= MIN_MORPHOLOGY_DESCENDANTS
                })
                .min_by(|(_, a), (_, b)| a.fitness.total_cmp(&b.fitness));
            let Some((slot, elite)) = victim else {
                return Offer::default();
            };
            if fitness <= elite.fitness {
                return Offer::default();
            }
            self.remove_entry(slot);
        }
        let mut salt = 0u64;
        let niche = loop {
            let candidate = morphology_niche(&topology, salt);
            match self.lookup.get(&candidate).copied() {
                None => break candidate,
                Some(slot) if topology_equivalent(&topology, &self.entries[slot].topology) => {
                    return Offer::default();
                }
                Some(_) => salt = salt.wrapping_add(1),
            }
        };
        let elite = Elite {
            niche: niche.clone(),
            descriptor,
            creature: population.creature(index),
            fitness,
            emitter,
            improved_generation: generation,
            protected_until,
            visits: 0,
            topology,
        };
        self.entries.push(elite);
        let slot = self.entries.len() - 1;
        self.lookup.insert(niche, slot);
        self.least_visited.insert((0, slot));
        self.morphology_indices.push(slot);
        Offer {
            inserted: true,
            new_niche: true,
            reward: 1.0,
        }
    }
    /// Adds a copy of `elite` if its behavior niche is empty or it beats the
    /// occupant. Used for island migration; returns whether it was kept.
    pub fn absorb(&mut self, elite: &Elite) -> bool {
        if is_morphology_niche(&elite.niche) {
            return false;
        }
        if let Some(&slot) = self.lookup.get(&elite.niche) {
            let current = &self.entries[slot];
            if elite.fitness <= current.fitness {
                return false;
            }
            self.qd_score += elite.fitness.max(0.0) as f64 - current.fitness.max(0.0) as f64;
            let visits = current.visits;
            self.entries[slot] = Elite {
                visits,
                ..elite.clone()
            };
        } else {
            if self.behavior_count() >= ARCHIVE_LIMIT {
                return false;
            }
            self.qd_score += elite.fitness.max(0.0) as f64;
            self.entries.push(Elite {
                visits: 0,
                ..elite.clone()
            });
            let slot = self.entries.len() - 1;
            self.lookup.insert(elite.niche.clone(), slot);
            self.least_visited.insert((0, slot));
            self.behavior_indices.push(slot);
        }
        self.behavior_scores = BehaviorScores::default();
        true
    }
    fn remove_morphology_topology(&mut self, topology: &Topology, behavior_fitness: f32) {
        if let Some(slot) = self.entries.iter().position(|elite| {
            is_morphology_niche(&elite.niche)
                && elite.fitness <= behavior_fitness
                && topology_equivalent(topology, &elite.topology)
        }) {
            self.remove_entry(slot);
        }
    }
    fn remove_entry(&mut self, slot: usize) {
        let last = self.entries.len() - 1;
        let removed = &self.entries[slot];
        self.lookup.remove(&removed.niche);
        self.least_visited.remove(&(removed.visits, slot));
        let removed_is_morphology = is_morphology_niche(&removed.niche);
        let indices = if removed_is_morphology {
            &mut self.morphology_indices
        } else {
            &mut self.behavior_indices
        };
        if let Some(index) = indices.iter().position(|&entry| entry == slot) {
            indices.swap_remove(index);
        }
        if slot != last {
            let moved = &self.entries[last];
            self.least_visited.remove(&(moved.visits, last));
            let moved_niche = moved.niche.clone();
            let moved_visits = moved.visits;
            let moved_is_morphology = is_morphology_niche(&moved.niche);
            self.entries.swap_remove(slot);
            self.lookup.insert(moved_niche, slot);
            self.least_visited.insert((moved_visits, slot));
            let indices = if moved_is_morphology {
                &mut self.morphology_indices
            } else {
                &mut self.behavior_indices
            };
            if let Some(index) = indices.iter().position(|&entry| entry == last) {
                indices[index] = slot;
            }
        } else {
            self.entries.pop();
        }
        self.behavior_scores = BehaviorScores::default();
    }
    fn local_competition_for(&self, niche: &Niche, fitness: f32) -> f32 {
        let mut compared = 0usize;
        let mut wins = 0.0f32;
        for neighbor in neighbor_niches(niche, 1) {
            let Some(&slot) = self.lookup.get(&neighbor) else {
                continue;
            };
            compared += 1;
            wins += match fitness.total_cmp(&self.entries[slot].fitness) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Less => 0.0,
            };
        }
        if compared == 0 {
            0.5
        } else {
            wins / compared as f32
        }
    }

    fn recompute_score(&mut self) {
        self.qd_score = self
            .entries
            .iter()
            .filter(|elite| !is_morphology_niche(&elite.niche))
            .map(|elite| elite.fitness.max(0.0) as f64)
            .sum();
    }
}
pub fn topology_of_population(population: &Population, index: usize) -> Topology {
    let genome = &population.genomes[index];
    let nodes = &population.nodes[genome.node_start..genome.node_start + genome.node_count];
    let bones = &population.bones[genome.bone_start..genome.bone_start + genome.bone_count];
    let muscles =
        &population.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count];
    topology_from_parts(nodes, bones, muscles)
}

impl EmitterStats {
    pub fn stale(&self) -> bool {
        self.stagnant_batches >= 5
    }
}

pub fn emitter_weights(stats: &[EmitterStats; EMITTER_COUNT]) -> [f64; EMITTER_COUNT] {
    if stats.iter().all(|s| s.attempts == 0) {
        return INITIAL_EMITTER_MIX;
    }
    let total = stats.iter().map(|s| s.attempts).sum::<u64>().max(1) as f64;
    let mut weights = [0.0; EMITTER_COUNT];
    for i in 0..EMITTER_COUNT {
        let prior = INITIAL_EMITTER_MIX[i];
        let mean = if stats[i].attempts == 0 {
            0.5
        } else {
            stats[i].reward
        };
        let exploration = 0.35 * (total.ln_1p() / (stats[i].attempts as f64 + 1.0)).sqrt();
        // Keep half of the prior allocation as an exploration floor while the
        // other half follows archive discoveries and improvements.
        weights[i] = prior * (0.5 + 0.05 + mean + exploration);
    }
    let sum = weights.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    weights.map(|w| w / sum)
}
pub fn choose_emitter(rng: &mut Rng, weights: &[f64; EMITTER_COUNT]) -> Emitter {
    let draw = rng.unit() as f64;
    let mut total = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        total += weight;
        if draw <= total {
            return Emitter::from_index(index);
        }
    }
    Emitter::Restart
}
pub fn record_emitter_batch(
    stats: &mut [EmitterStats; EMITTER_COUNT],
    attempts: &[u64; EMITTER_COUNT],
    discoveries: &[u64; EMITTER_COUNT],
    improvements: &[u64; EMITTER_COUNT],
    rewards: &[f64; EMITTER_COUNT],
) {
    for i in 0..EMITTER_COUNT {
        let s = &mut stats[i];
        s.attempts += attempts[i];
        s.discoveries += discoveries[i];
        s.improvements += improvements[i];
        if attempts[i] > 0 {
            let batch_rate = rewards[i] / attempts[i] as f64;
            s.reward = 0.75 * s.reward + 0.25 * batch_rate;
        }
        if improvements[i] == 0 && discoveries[i] == 0 {
            s.stagnant_batches = s.stagnant_batches.saturating_add(1);
        } else {
            s.stagnant_batches = 0;
        }
    }
}

impl CmaEmitter {
    /// A CMA-ME emitter improving one niche of the archive.
    pub fn new(template: Creature, niche: Niche, generation: u32) -> Self {
        let mean = exploring_parameters(&template);
        let dimensions = mean.len();
        Self {
            niche,
            topology: Topology::of(&template),
            template,
            mean,
            covariance: vec![1.0; dimensions],
            path_c: vec![0.0; dimensions],
            path_sigma: vec![0.0; dimensions],
            sigma: 0.12,
            last_used_generation: generation,
        }
    }
    /// A separable CMA-ES optimizer for one island's body plan, starting from
    /// a fast elite: it searches in physical units and ranks samples by
    /// fitness alone.
    pub fn optimizer(template: Creature, niche: Niche, generation: u32) -> Self {
        let mean = parameters(&template);
        let dimensions = mean.len();
        Self {
            niche,
            topology: Topology::of(&template),
            template,
            mean,
            covariance: vec![1.0; dimensions],
            path_c: vec![0.0; dimensions],
            path_sigma: vec![0.0; dimensions],
            sigma: 0.5,
            last_used_generation: generation,
        }
    }
    /// A new optimizer starting from `template` that keeps the step sizes
    /// this one has learned for the same body plan.
    pub fn recentered(&self, template: Creature, niche: Niche, generation: u32) -> Self {
        let mut next = Self::optimizer(template, niche, generation);
        if next.topology == self.topology && next.mean.len() == self.mean.len() {
            next.covariance.clone_from(&self.covariance);
            next.path_c.clone_from(&self.path_c);
            next.path_sigma.clone_from(&self.path_sigma);
            next.sigma = self.sigma;
        }
        next
    }
    pub fn optimizing(&self) -> bool {
        self.niche.0[0] == OPTIMIZER_NICHE_MARKER
    }
    /// An optimizer whose steps have shrunk this far has converged.
    pub fn converged(&self) -> bool {
        self.optimizing() && self.sigma < 0.05
    }
    pub fn sample(&self, rng: &mut Rng) -> Creature {
        self.sample_scaled(rng, 1.0)
    }
    pub fn sample_scaled(&self, rng: &mut Rng, strength: f32) -> Creature {
        if self.optimizing() {
            self.sample_optimizing(rng, strength)
        } else {
            self.sample_exploring(rng, strength)
        }
    }
    /// Updates the search distribution from scored samples. CMA-ME emitters
    /// get improvement keys; optimizers get fitness.
    pub fn tell(&mut self, population: &Population, samples: &mut Vec<(usize, f32)>) {
        if samples.len() < 2 {
            return;
        }
        samples.retain(|(index, score)| {
            if !score.is_finite() || *score <= crate::evolution::FAILED {
                return false;
            }
            let Some(genome) = population.genomes.get(*index) else {
                return false;
            };
            genome.node_count == self.template.nodes.len()
                && genome.bone_count == self.template.bones.len()
                && genome.muscle_count == self.template.muscles.len()
                && topology_of_population(population, *index) == self.topology
        });
        if samples.len() < 2 {
            return;
        }
        const MAX_UPDATE_SAMPLES: usize = 1024;
        if samples.len() > MAX_UPDATE_SAMPLES {
            samples.select_nth_unstable_by(MAX_UPDATE_SAMPLES, |a, b| b.1.total_cmp(&a.1));
            samples.truncate(MAX_UPDATE_SAMPLES);
        }
        samples.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if self.optimizing() {
            self.tell_optimizing(population, samples);
        } else {
            self.tell_exploring(population, samples);
        }
    }
    fn sample_exploring(&self, rng: &mut Rng, strength: f32) -> Creature {
        let phase_start = self.template.nodes.len() * 4 + self.template.bones.len();
        // Diagonal covariance plus a rank-one term along the evolution path, so
        // parameter changes that keep paying off move together.
        let path_norm = self.path_c.iter().map(|p| p * p).sum::<f32>().sqrt();
        let path_scale = if path_norm > 1e-6 {
            PATH_WEIGHT.sqrt() * gaussian(rng) / path_norm * (self.mean.len() as f32).sqrt()
        } else {
            0.0
        };
        let node_end = self.template.nodes.len() * 4;
        let values: Vec<_> = self
            .mean
            .iter()
            .zip(&self.covariance)
            .zip(&self.path_c)
            .enumerate()
            .map(|(d, ((&mean, &variance), &path))| {
                let step = variance.sqrt() * gaussian(rng) + path * path_scale;
                let value = mean + self.sigma * step * strength;
                // Positions and muscle lengths keep the original 4 m and 1 m
                // scales but are open-ended, so large bodies keep their shape;
                // repair enforces the body limits.
                let muscle_field = d.checked_sub(phase_start).map(|m| m % 8);
                if is_phase_dimension(d, phase_start) {
                    value.rem_euclid(1.0)
                } else if d < node_end && d % 4 == 0 {
                    value
                } else if (d < node_end && d % 4 == 1) || matches!(muscle_field, Some(2 | 3)) {
                    value.max(0.0)
                } else {
                    value.clamp(0.0, 1.0)
                }
            })
            .collect();
        let mut creature = self.template.clone();
        apply_exploring_parameters(&mut creature, &values);
        creature
    }
    fn tell_exploring(&mut self, population: &Population, samples: &[(usize, f32)]) {
        let dimensions = self.mean.len();
        let mu = samples.len().div_ceil(2).max(1);
        let mut weights: Vec<f32> = (0..mu)
            .map(|i| (mu as f32 + 0.5).ln() - ((i + 1) as f32).ln())
            .collect();
        let weight_sum = weights.iter().sum::<f32>().max(f32::MIN_POSITIVE);
        for weight in &mut weights {
            *weight /= weight_sum;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f32>().max(1e-6);
        let old_mean = self.mean.clone();
        let mut new_mean = vec![0.0; dimensions];
        let mut vector = vec![0.0; dimensions];
        let phase_start = self.template.nodes.len() * 4 + self.template.bones.len();
        for (rank, &(index, _)) in samples.iter().take(mu).enumerate() {
            exploring_parameters_into(population, index, &mut vector);
            for d in 0..dimensions {
                if is_phase_dimension(d, phase_start) {
                    new_mean[d] += weights[rank] * wrap_phase(vector[d] - old_mean[d]);
                } else {
                    new_mean[d] += weights[rank] * vector[d];
                }
            }
        }
        for d in 0..dimensions {
            if is_phase_dimension(d, phase_start) {
                new_mean[d] = (old_mean[d] + new_mean[d]).rem_euclid(1.0);
            }
        }
        let n = dimensions as f32;
        let c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (((mu_eff - 1.0).max(0.0) / (n + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);
        let c1 = 2.0 / ((n + 1.3).powi(2) + mu_eff);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff).max(0.0) / ((n + 2.0).powi(2) + mu_eff))
            .min(1.0 - c1);
        let mut norm_sigma = 0.0;
        for d in 0..dimensions {
            let mean_delta = if is_phase_dimension(d, phase_start) {
                wrap_phase(new_mean[d] - old_mean[d])
            } else {
                new_mean[d] - old_mean[d]
            };
            let y = mean_delta / self.sigma.max(1e-6);
            let normalized = y / self.covariance[d].sqrt().max(1e-6);
            self.path_sigma[d] = (1.0 - c_sigma) * self.path_sigma[d]
                + (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt() * normalized;
            norm_sigma += self.path_sigma[d] * self.path_sigma[d];
        }
        norm_sigma = norm_sigma.sqrt();
        let chi = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
        let hsig = norm_sigma / chi.max(1e-6) < 1.4 + 2.0 / (n + 1.0);
        let correction = if hsig { 0.0 } else { c_c * (2.0 - c_c) };
        let mut rank_mu = vec![0.0; dimensions];
        for (rank, &(index, _)) in samples.iter().take(mu).enumerate() {
            exploring_parameters_into(population, index, &mut vector);
            for d in 0..dimensions {
                let coordinate_delta = if is_phase_dimension(d, phase_start) {
                    wrap_phase(vector[d] - old_mean[d])
                } else {
                    vector[d] - old_mean[d]
                };
                let delta = coordinate_delta / self.sigma.max(1e-6);
                rank_mu[d] += weights[rank] * delta * delta;
            }
        }
        for d in 0..dimensions {
            let mean_delta = if is_phase_dimension(d, phase_start) {
                wrap_phase(new_mean[d] - old_mean[d])
            } else {
                new_mean[d] - old_mean[d]
            };
            let y = mean_delta / self.sigma.max(1e-6);
            self.path_c[d] = (1.0 - c_c) * self.path_c[d]
                + if hsig {
                    (c_c * (2.0 - c_c) * mu_eff).sqrt() * y
                } else {
                    0.0
                };
            self.covariance[d] = ((1.0 - c1 - c_mu + c1 * correction) * self.covariance[d]
                + c1 * self.path_c[d] * self.path_c[d]
                + c_mu * rank_mu[d])
                .clamp(0.0025, 4.0);
        }
        self.sigma = (self.sigma
            * ((c_sigma / d_sigma) * (norm_sigma / chi.max(1e-6) - 1.0)).exp())
        .clamp(0.005, 0.35);
        self.mean = new_mean;
    }
    fn sample_optimizing(&self, rng: &mut Rng, strength: f32) -> Creature {
        let layout = Layout::of(&self.template);
        let values: Vec<_> = self
            .mean
            .iter()
            .zip(&self.covariance)
            .enumerate()
            .map(|(d, (&mean, &variance))| {
                mean + self.sigma * strength * layout.scale(d) * variance.sqrt() * gaussian(rng)
            })
            .collect();
        let mut creature = self.template.clone();
        apply_parameters(&mut creature, &values);
        creature
    }
    /// Separable CMA-ES update (Ros & Hansen 2008) from scored samples, fastest
    /// first. Steps are measured in each coordinate's physical scale.
    fn tell_optimizing(&mut self, population: &Population, samples: &[(usize, f32)]) {
        let dimensions = self.mean.len();
        let mu = samples.len().div_ceil(2).max(1);
        let mut weights: Vec<f32> = (0..mu)
            .map(|i| (mu as f32 + 0.5).ln() - ((i + 1) as f32).ln())
            .collect();
        let weight_sum = weights.iter().sum::<f32>().max(f32::MIN_POSITIVE);
        for weight in &mut weights {
            *weight /= weight_sum;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f32>().max(1e-6);
        let layout = Layout::of(&self.template);
        let sigma = self.sigma.max(1e-6);
        // Weighted mean step and weighted squared steps, in units of sigma.
        let mut step = vec![0.0; dimensions];
        let mut rank_mu = vec![0.0; dimensions];
        let mut vector = vec![0.0; dimensions];
        for (rank, &(index, _)) in samples.iter().take(mu).enumerate() {
            parameters_into(population, index, &mut vector);
            for d in 0..dimensions {
                let y = layout.delta(d, vector[d], self.mean[d]) / (sigma * layout.scale(d));
                step[d] += weights[rank] * y;
                rank_mu[d] += weights[rank] * y * y;
            }
        }
        let n = dimensions as f32;
        let c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (((mu_eff - 1.0).max(0.0) / (n + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);
        // A diagonal covariance learns (n + 2) / 3 times faster than a full one.
        let separable = (n + 2.0) / 3.0;
        let c1 = (2.0 / ((n + 1.3).powi(2) + mu_eff) * separable).min(0.5);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff).max(0.0) / ((n + 2.0).powi(2) + mu_eff)
            * separable)
            .min(1.0 - c1);
        let mut norm_sigma = 0.0;
        for ((path, &step), &variance) in
            self.path_sigma.iter_mut().zip(&step).zip(&self.covariance)
        {
            *path = (1.0 - c_sigma) * *path
                + (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt() * step / variance.sqrt().max(1e-6);
            norm_sigma += *path * *path;
        }
        norm_sigma = norm_sigma.sqrt();
        let chi = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
        let hsig = norm_sigma / chi.max(1e-6) < 1.4 + 2.0 / (n + 1.0);
        let correction = if hsig { 0.0 } else { c_c * (2.0 - c_c) };
        for d in 0..dimensions {
            self.path_c[d] = (1.0 - c_c) * self.path_c[d]
                + if hsig {
                    (c_c * (2.0 - c_c) * mu_eff).sqrt() * step[d]
                } else {
                    0.0
                };
            self.covariance[d] = ((1.0 - c1 - c_mu + c1 * correction) * self.covariance[d]
                + c1 * self.path_c[d] * self.path_c[d]
                + c_mu * rank_mu[d])
                .clamp(1e-4, 1e4);
            self.mean[d] = layout.moved(d, self.mean[d], sigma * layout.scale(d) * step[d]);
        }
        self.sigma = (self.sigma
            * ((c_sigma / d_sigma) * (norm_sigma / chi.max(1e-6) - 1.0)).exp())
        .clamp(0.01, 30.0);
        // Fast gaits are fragile: when most samples fail outright, step-length
        // control alone keeps growing the steps, so shrink them instead.
        if samples[samples.len() / 2].1 < 0.25 * samples[0].1 {
            self.sigma = (self.sigma * 0.6).max(0.01);
        }
    }
}

fn is_phase_dimension(dimension: usize, phase_start: usize) -> bool {
    dimension >= phase_start && (dimension - phase_start) % 8 == 5
}
fn wrap_phase(delta: f32) -> f32 {
    (delta + 0.5).rem_euclid(1.0) - 0.5
}

/// Share of each CMA step taken along the normalized evolution path.
const PATH_WEIGHT: f32 = 0.3;

pub(crate) fn gaussian(rng: &mut Rng) -> f32 {
    let u1 = (1.0 - rng.unit()).max(1e-7);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
fn exploring_parameters(creature: &Creature) -> Vec<f32> {
    let mut output = Vec::with_capacity(
        creature.nodes.len() * 4 + creature.bones.len() + creature.muscles.len() * 8,
    );
    for n in &creature.nodes {
        output.extend([
            (n.x + 4.0) / 8.0,
            (n.y / 4.0).max(0.0),
            ((n.diameter - 0.01) / 0.99).clamp(0.0, 1.0),
            n.friction.clamp(0.0, 1.0),
        ]);
    }
    for bone in &creature.bones {
        output.push(
            ((bone.rest_length - 0.03) / (crate::evolution::max_bone_length() - 0.03))
                .clamp(0.0, 1.0),
        );
    }
    for m in &creature.muscles {
        output.extend([
            m.anchor_a.clamp(0.0, 1.0),
            m.anchor_b.clamp(0.0, 1.0),
            ((m.short - 0.01) / 0.79).max(0.0),
            ((m.long - 0.01) / 0.99).max(0.0),
            ((m.period - crate::evolution::min_muscle_period())
                / (10.0 - crate::evolution::min_muscle_period()))
            .clamp(0.0, 1.0),
            m.phase.clamp(0.0, 1.0),
            ((m.duty - 0.05) / 0.90).clamp(0.0, 1.0),
            ((m.stiffness - 1.0) / 119.0).clamp(0.0, 1.0),
        ]);
    }
    output
}
fn exploring_parameters_into(population: &Population, index: usize, output: &mut [f32]) {
    let genome = &population.genomes[index];
    let nodes = &population.nodes[genome.node_start..genome.node_start + genome.node_count];
    let muscles =
        &population.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count];
    let mut i = 0;
    for n in nodes {
        output[i..i + 4].copy_from_slice(&[
            (n.x + 4.0) / 8.0,
            (n.y / 4.0).max(0.0),
            ((n.diameter - 0.01) / 0.99).clamp(0.0, 1.0),
            n.friction.clamp(0.0, 1.0),
        ]);
        i += 4;
    }
    let bones = &population.bones[genome.bone_start..genome.bone_start + genome.bone_count];
    for bone in bones {
        output[i] = ((bone.rest_length - 0.03) / (crate::evolution::max_bone_length() - 0.03))
            .clamp(0.0, 1.0);
        i += 1;
    }
    for m in muscles {
        output[i..i + 8].copy_from_slice(&[
            m.anchor_a.clamp(0.0, 1.0),
            m.anchor_b.clamp(0.0, 1.0),
            ((m.short - 0.01) / 0.79).max(0.0),
            ((m.long - 0.01) / 0.99).max(0.0),
            ((m.period - crate::evolution::min_muscle_period())
                / (10.0 - crate::evolution::min_muscle_period()))
            .clamp(0.0, 1.0),
            m.phase.clamp(0.0, 1.0),
            ((m.duty - 0.05) / 0.90).clamp(0.0, 1.0),
            ((m.stiffness - 1.0) / 119.0).clamp(0.0, 1.0),
        ]);
        i += 8;
    }
}
fn apply_exploring_parameters(creature: &mut Creature, values: &[f32]) {
    let mut i = 0;
    for n in &mut creature.nodes {
        n.x = values[i] * 8.0 - 4.0;
        n.y = values[i + 1] * 4.0;
        n.diameter = 0.01 + values[i + 2] * 0.99;
        n.friction = values[i + 3];
        i += 4;
    }
    for bone in &mut creature.bones {
        bone.rest_length = 0.03 + values[i] * (crate::evolution::max_bone_length() - 0.03);
        i += 1;
    }
    for m in &mut creature.muscles {
        m.anchor_a = values[i].clamp(0.0, 1.0);
        m.anchor_b = values[i + 1].clamp(0.0, 1.0);
        m.short = 0.01 + values[i + 2] * 0.79;
        m.long = (0.01 + values[i + 3] * 0.99).max(m.short);
        m.period = crate::evolution::min_muscle_period()
            + values[i + 4] * (10.0 - crate::evolution::min_muscle_period());
        m.phase = values[i + 5].fract();
        m.duty = 0.05 + values[i + 6] * 0.90;
        m.stiffness = 1.0 + values[i + 7] * 119.0;
        i += 8;
    }
}

/// Where each CMA coordinate lives in a body plan and how far one unit step
/// moves it: per node x, y, diameter, friction; per bone rest length, joint
/// range, and organ mass and position; the shared log period; per muscle
/// anchors, lengths, phase, duty, log stiffness, and touchdown reset phase.
/// An organ mass at or below zero means no organ, so organs can grow and
/// vanish smoothly.
struct Layout {
    nodes: usize,
    bones: usize,
    /// Typical bone length (m), so positions and lengths search relative to
    /// the body's size.
    size: f32,
}
const NODE_SCALES: [f32; 4] = [0.02, 0.02, 0.005, 0.03];
const BONE_SCALES: [f32; 5] = [0.02, 0.1, 0.1, 0.02, 0.05];
const BONE_FIELDS: usize = BONE_SCALES.len();
const PERIOD_SCALE: f32 = 0.05;
const MUSCLE_SCALES: [f32; 8] = [0.05, 0.05, 0.02, 0.02, 0.05, 0.05, 0.1, 0.05];
/// Which scales above are lengths, multiplied by the body's size.
const MUSCLE_LENGTHS: [bool; 8] = [false, false, true, true, false, false, false, false];
impl Layout {
    fn of(template: &Creature) -> Self {
        let size = if template.bones.is_empty() {
            1.0
        } else {
            template.bones.iter().map(|b| b.rest_length).sum::<f32>() / template.bones.len() as f32
        };
        Self {
            nodes: template.nodes.len(),
            bones: template.bones.len(),
            size: size.clamp(0.05, 10.0),
        }
    }
    /// The muscle field of coordinate `d`, if it is one.
    fn muscle_field(&self, d: usize) -> Option<usize> {
        let start = self.nodes * 4 + self.bones * BONE_FIELDS + 1;
        (d >= start).then(|| (d - start) % 8)
    }
    fn scale(&self, d: usize) -> f32 {
        let node_end = self.nodes * 4;
        let bone_end = node_end + self.bones * BONE_FIELDS;
        if d < node_end {
            NODE_SCALES[d % 4] * if d % 4 < 2 { self.size } else { 1.0 }
        } else if d < bone_end {
            let field = (d - node_end) % BONE_FIELDS;
            BONE_SCALES[field] * if field == 0 { self.size } else { 1.0 }
        } else if let Some(field) = self.muscle_field(d) {
            MUSCLE_SCALES[field]
                * if MUSCLE_LENGTHS[field] {
                    self.size
                } else {
                    1.0
                }
        } else {
            PERIOD_SCALE
        }
    }
    /// Phases wrap around the cycle.
    fn wraps(&self, d: usize) -> bool {
        matches!(self.muscle_field(d), Some(4 | 7))
    }
    fn delta(&self, d: usize, value: f32, mean: f32) -> f32 {
        if self.wraps(d) {
            (value - mean + 0.5).rem_euclid(1.0) - 0.5
        } else {
            value - mean
        }
    }
    fn moved(&self, d: usize, mean: f32, step: f32) -> f32 {
        if self.wraps(d) {
            (mean + step).rem_euclid(1.0)
        } else {
            mean + step
        }
    }
}

fn parameters(creature: &Creature) -> Vec<f32> {
    let mut output = vec![
        0.0;
        parameter_count(
            creature.nodes.len(),
            creature.bones.len(),
            creature.muscles.len()
        )
    ];
    write_parameters(
        &creature.nodes,
        &creature.bones,
        &creature.muscles,
        &mut output,
    );
    output
}
fn parameter_count(nodes: usize, bones: usize, muscles: usize) -> usize {
    nodes * 4 + bones * BONE_FIELDS + 1 + muscles * 8
}
fn parameters_into(population: &Population, index: usize, output: &mut [f32]) {
    let genome = &population.genomes[index];
    write_parameters(
        &population.nodes[genome.node_start..genome.node_start + genome.node_count],
        &population.bones[genome.bone_start..genome.bone_start + genome.bone_count],
        &population.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count],
        output,
    );
}
fn write_parameters(
    nodes: &[crate::evolution::NodeGene],
    bones: &[crate::evolution::Bone],
    muscles: &[crate::evolution::Muscle],
    output: &mut [f32],
) {
    let mut i = 0;
    for n in nodes {
        output[i..i + 4].copy_from_slice(&[n.x, n.y, n.diameter, n.friction]);
        i += 4;
    }
    for b in bones {
        output[i..i + BONE_FIELDS].copy_from_slice(&[
            b.rest_length,
            b.min_angle,
            b.max_angle,
            b.organ_mass,
            b.organ_at,
        ]);
        i += BONE_FIELDS;
    }
    output[i] = muscles.first().map_or(1.0, |m| m.period).max(1e-3).ln();
    i += 1;
    for m in muscles {
        output[i..i + 8].copy_from_slice(&[
            m.anchor_a,
            m.anchor_b,
            m.short,
            m.long,
            m.phase,
            m.duty,
            m.stiffness.max(1e-3).ln(),
            m.reset,
        ]);
        i += 8;
    }
}
fn apply_parameters(creature: &mut Creature, values: &[f32]) {
    let mut i = 0;
    for n in &mut creature.nodes {
        n.x = values[i];
        n.y = values[i + 1].max(0.0);
        n.diameter = values[i + 2];
        n.friction = values[i + 3];
        i += 4;
    }
    for bone in &mut creature.bones {
        bone.rest_length = values[i].clamp(0.03, crate::evolution::max_bone_length());
        bone.min_angle = values[i + 1];
        bone.max_angle = values[i + 2];
        bone.clamp_range();
        // Repair keeps organs within their mass range and near the center.
        bone.organ_mass = values[i + 3].max(0.0);
        bone.organ_at = values[i + 4].clamp(0.0, 1.0);
        i += BONE_FIELDS;
    }
    let period = values[i]
        .exp()
        .clamp(crate::evolution::min_muscle_period(), 10.0);
    i += 1;
    let stroke = crate::evolution::max_stroke();
    for m in &mut creature.muscles {
        m.anchor_a = values[i].clamp(0.0, 1.0);
        m.anchor_b = values[i + 1].clamp(0.0, 1.0);
        m.short = values[i + 2].clamp(0.01, 0.8 * stroke);
        m.long = values[i + 3].clamp(m.short, stroke);
        m.period = period;
        m.phase = values[i + 4].rem_euclid(1.0);
        m.duty = values[i + 5].clamp(0.05, 0.95);
        m.stiffness = values[i + 6].exp().clamp(1.0, 120.0);
        m.reset = values[i + 7].rem_euclid(1.0);
        i += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::{CmaEmitter, Layout, Niche};
    use crate::{
        config::Config,
        evolution::{self, Population},
    };

    #[test]
    fn cma_feedback_ignores_candidates_with_changed_topology() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let template = evolution::create(&config).unwrap().creature(0);
        let mut cma = CmaEmitter::new(template.clone(), Niche([0; 6]), 0);
        let original_mean = cma.mean.clone();
        let mut population = Population::default();
        for _ in 0..2 {
            let mut changed = template.clone();
            changed.muscles.push(changed.muscles[0]);
            population.push(changed);
        }
        cma.tell(&population, &mut vec![(0, 2.0), (1, 1.0)]);
        assert_eq!(cma.mean, original_mean);
    }

    #[test]
    fn cma_phase_dimensions_follow_bones_and_eight_value_muscles() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let template = evolution::create(&config).unwrap().creature(0);
        let layout = Layout::of(&template);
        let start = template.nodes.len() * 4 + template.bones.len() * super::BONE_FIELDS + 1;
        assert!(!layout.wraps(start - 1));
        assert!(!layout.wraps(start));
        assert!(layout.wraps(start + 4));
        assert!(layout.wraps(start + 7));
        assert!(layout.wraps(start + 12));
        assert!(!layout.wraps(start + 13));
    }

    #[test]
    fn optimizer_moves_toward_its_faster_samples() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let template = evolution::create(&config).unwrap().creature(0);
        let mut cma = CmaEmitter::optimizer(template.clone(), super::optimizer_niche(0, 0), 0);
        assert!(cma.optimizing() && !cma.converged());
        let mut population = Population::default();
        for shift in [0.05, -0.05, 0.04, -0.04] {
            let mut sample = template.clone();
            for node in &mut sample.nodes {
                node.x += shift;
            }
            population.push(sample);
        }
        let before = cma.mean[0];
        cma.tell(
            &population,
            &mut vec![(0, 4.0), (2, 3.0), (3, 2.0), (1, 1.0)],
        );
        assert!(cma.mean[0] > before);
    }

    #[test]
    fn cma_parameters_round_trip_through_a_creature() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let template = evolution::create(&config).unwrap().creature(0);
        let cma = CmaEmitter::optimizer(template.clone(), super::optimizer_niche(0, 0), 0);
        let mut copy = template.clone();
        super::apply_parameters(&mut copy, &cma.mean);
        assert_eq!(super::parameters(&copy), cma.mean);
    }
}
