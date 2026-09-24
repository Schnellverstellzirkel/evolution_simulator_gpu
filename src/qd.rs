use crate::evolution::{Creature, Muscle, Population, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

pub const EMITTER_COUNT: usize = 4;
pub(crate) const ARCHIVE_LIMIT: usize = 192;
pub(crate) const MORPHOLOGY_LIMIT: usize = 64;
pub(crate) const ARCHIVE_CAPACITY: usize = ARCHIVE_LIMIT + MORPHOLOGY_LIMIT;
pub(crate) const HISTORICAL_ARCHIVE_LIMIT: usize = 13_824;
pub(crate) const CMA_LIMIT: usize = 96;
pub const VERSION: u32 = 8;
const LOCAL_NEIGHBORS: usize = 5;
const MORPHOLOGY_NICHE_MARKER: u8 = u8::MAX;
pub(crate) const MIN_MORPHOLOGY_DESCENDANTS: u64 = 8;
pub(crate) const MORPHOLOGY_PARENT_FRACTION: f32 = 0.10;
// Deliberately exploration-heavy: 70% of the initial batch uses structural,
// novelty, or restart emitters so a stalled lineage cannot dominate for long.
const INITIAL_EMITTER_MIX: [f64; EMITTER_COUNT] = [0.30, 0.30, 0.25, 0.15];

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct TrialMetrics {
    pub ground_contact: f32,
    pub vertical_oscillation: f32,
    pub gait_frequency: f32,
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
    }
}

impl Descriptor {
    pub fn niche(self) -> Niche {
        // Only measured behavior determines archive cells. Morphology remains
        // attached to each descriptor for display and analysis.
        Niche([
            bin(self.ground_contact, 0.0, 1.0, 4),
            bin(self.gait_frequency, 0.0, 6.0, 8),
            bin(self.vertical_oscillation, 0.0, 0.8, 6),
            0,
            0,
            0,
        ])
    }

    fn behavior(self) -> [f32; 3] {
        [
            self.ground_contact.clamp(0.0, 1.0),
            (self.gait_frequency / 6.0).clamp(0.0, 1.0),
            (self.vertical_oscillation / 0.8).clamp(0.0, 1.0),
        ]
    }
}
fn bin(value: f32, low: f32, high: f32, count: u8) -> u8 {
    (((value.clamp(low, high) - low) / (high - low) * count as f32).floor() as u8).min(count - 1)
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
    pub fn refresh_behavior_scores(&mut self) {
        let behavior = &self.behavior_indices;
        let count = behavior.len();
        if count == 0 {
            self.behavior_scores = BehaviorScores::default();
            return;
        }
        let mut novelty = vec![0.0; self.entries.len()];
        let mut local_competition = vec![0.5; self.entries.len()];
        if count == 1 {
            novelty[behavior[0]] = 1.0;
            self.behavior_scores = BehaviorScores {
                novelty,
                local_competition,
            };
            return;
        }
        let mut neighbors = Vec::with_capacity(count - 1);
        for i in 0..count {
            neighbors.clear();
            for j in 0..count {
                if i != j {
                    neighbors.push((
                        behavior_distance(
                            self.entries[behavior[i]].descriptor,
                            self.entries[behavior[j]].descriptor,
                        ),
                        behavior[j],
                    ));
                }
            }
            neighbors.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let nearest = &neighbors[..LOCAL_NEIGHBORS.min(neighbors.len())];
            novelty[behavior[i]] =
                nearest.iter().map(|(distance, _)| *distance).sum::<f32>() / nearest.len() as f32;
            local_competition[behavior[i]] = nearest
                .iter()
                .map(|(_, j)| {
                    match self.entries[behavior[i]]
                        .fitness
                        .total_cmp(&self.entries[*j].fitness)
                    {
                        std::cmp::Ordering::Greater => 1.0,
                        std::cmp::Ordering::Equal => 0.5,
                        std::cmp::Ordering::Less => 0.0,
                    }
                })
                .sum::<f32>()
                / nearest.len() as f32;
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
        if self.entries.is_empty() {
            return 0.5;
        }
        let bounds = [4i32, 8, 6];
        let center = [niche.0[0] as i32, niche.0[1] as i32, niche.0[2] as i32];
        let mut compared = 0usize;
        let mut wins = 0.0f32;
        for a in -1..=1 {
            let x = center[0] + a;
            if !(0..bounds[0]).contains(&x) {
                continue;
            }
            for b in -1..=1 {
                let y = center[1] + b;
                if !(0..bounds[1]).contains(&y) {
                    continue;
                }
                for c in -1..=1 {
                    let z = center[2] + c;
                    if !(0..bounds[2]).contains(&z) {
                        continue;
                    }
                    let neighbor = Niche([x as u8, y as u8, z as u8, 0, 0, 0]);
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
            }
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
    pub fn new(template: Creature, niche: Niche, generation: u32) -> Self {
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
            sigma: 0.12,
            last_used_generation: generation,
        }
    }
    pub fn sample(&self, rng: &mut Rng) -> Creature {
        self.sample_scaled(rng, 1.0)
    }
    pub fn sample_scaled(&self, rng: &mut Rng, strength: f32) -> Creature {
        let phase_start = self.template.nodes.len() * 4 + self.template.bones.len();
        let values: Vec<_> = self
            .mean
            .iter()
            .zip(&self.covariance)
            .enumerate()
            .map(|(d, (&mean, &variance))| {
                let value = mean + self.sigma * variance.sqrt() * gaussian(rng) * strength;
                if is_phase_dimension(d, phase_start) {
                    value.rem_euclid(1.0)
                } else {
                    value.clamp(0.0, 1.0)
                }
            })
            .collect();
        let mut creature = self.template.clone();
        apply_parameters(&mut creature, &values);
        creature
    }
    pub fn tell(&mut self, population: &Population, samples: &mut Vec<(usize, f32)>) {
        if samples.len() < 2 {
            return;
        }
        samples.retain(|(_, score)| score.is_finite() && *score > crate::evolution::FAILED);
        if samples.len() < 2 {
            return;
        }
        const MAX_UPDATE_SAMPLES: usize = 128;
        if samples.len() > MAX_UPDATE_SAMPLES {
            samples.select_nth_unstable_by(MAX_UPDATE_SAMPLES, |a, b| b.1.total_cmp(&a.1));
            samples.truncate(MAX_UPDATE_SAMPLES);
        }
        samples.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
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
            parameters_into(population, index, &mut vector);
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
            parameters_into(population, index, &mut vector);
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
}

fn is_phase_dimension(dimension: usize, phase_start: usize) -> bool {
    dimension >= phase_start && (dimension - phase_start) % 8 == 5
}
fn wrap_phase(delta: f32) -> f32 {
    (delta + 0.5).rem_euclid(1.0) - 0.5
}

pub(crate) fn gaussian(rng: &mut Rng) -> f32 {
    let u1 = (1.0 - rng.unit()).max(1e-7);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
fn parameters(creature: &Creature) -> Vec<f32> {
    let mut output = Vec::with_capacity(
        creature.nodes.len() * 4 + creature.bones.len() + creature.muscles.len() * 8,
    );
    for n in &creature.nodes {
        output.extend([
            ((n.x + 4.0) / 8.0).clamp(0.0, 1.0),
            (n.y / 4.0).clamp(0.0, 1.0),
            ((n.diameter - 0.01) / 0.99).clamp(0.0, 1.0),
            n.friction.clamp(0.0, 1.0),
        ]);
    }
    for bone in &creature.bones {
        output.push(
            ((bone.rest_length - 0.03) / (crate::evolution::MAX_BONE_LENGTH - 0.03))
                .clamp(0.0, 1.0),
        );
    }
    for m in &creature.muscles {
        output.extend([
            m.anchor_a.clamp(0.0, 1.0),
            m.anchor_b.clamp(0.0, 1.0),
            ((m.short - 0.01) / 0.79).clamp(0.0, 1.0),
            ((m.long - 0.01) / 0.99).clamp(0.0, 1.0),
            ((m.period - 0.1) / 9.9).clamp(0.0, 1.0),
            m.phase.clamp(0.0, 1.0),
            ((m.duty - 0.05) / 0.90).clamp(0.0, 1.0),
            ((m.stiffness - 1.0) / 119.0).clamp(0.0, 1.0),
        ]);
    }
    output
}
fn parameters_into(population: &Population, index: usize, output: &mut [f32]) {
    let genome = &population.genomes[index];
    let nodes = &population.nodes[genome.node_start..genome.node_start + genome.node_count];
    let muscles =
        &population.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count];
    let mut i = 0;
    for n in nodes {
        output[i..i + 4].copy_from_slice(&[
            ((n.x + 4.0) / 8.0).clamp(0.0, 1.0),
            (n.y / 4.0).clamp(0.0, 1.0),
            ((n.diameter - 0.01) / 0.99).clamp(0.0, 1.0),
            n.friction.clamp(0.0, 1.0),
        ]);
        i += 4;
    }
    let bones = &population.bones[genome.bone_start..genome.bone_start + genome.bone_count];
    for bone in bones {
        output[i] = ((bone.rest_length - 0.03) / (crate::evolution::MAX_BONE_LENGTH - 0.03))
            .clamp(0.0, 1.0);
        i += 1;
    }
    for m in muscles {
        output[i..i + 8].copy_from_slice(&[
            m.anchor_a.clamp(0.0, 1.0),
            m.anchor_b.clamp(0.0, 1.0),
            ((m.short - 0.01) / 0.79).clamp(0.0, 1.0),
            ((m.long - 0.01) / 0.99).clamp(0.0, 1.0),
            ((m.period - 0.1) / 9.9).clamp(0.0, 1.0),
            m.phase.clamp(0.0, 1.0),
            ((m.duty - 0.05) / 0.90).clamp(0.0, 1.0),
            ((m.stiffness - 1.0) / 119.0).clamp(0.0, 1.0),
        ]);
        i += 8;
    }
}
fn apply_parameters(creature: &mut Creature, values: &[f32]) {
    let mut i = 0;
    for n in &mut creature.nodes {
        n.x = values[i] * 8.0 - 4.0;
        n.y = values[i + 1] * 4.0;
        n.diameter = 0.01 + values[i + 2] * 0.99;
        n.friction = values[i + 3];
        i += 4;
    }
    for bone in &mut creature.bones {
        bone.rest_length = 0.03 + values[i] * (crate::evolution::MAX_BONE_LENGTH - 0.03);
        i += 1;
    }
    for m in &mut creature.muscles {
        m.anchor_a = values[i].clamp(0.0, 1.0);
        m.anchor_b = values[i + 1].clamp(0.0, 1.0);
        m.short = 0.01 + values[i + 2] * 0.79;
        m.long = (0.01 + values[i + 3] * 0.99).max(m.short);
        m.period = 0.1 + values[i + 4] * 9.9;
        m.phase = values[i + 5].fract();
        m.duty = 0.05 + values[i + 6] * 0.90;
        m.stiffness = 1.0 + values[i + 7] * 119.0;
        i += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::is_phase_dimension;

    #[test]
    fn cma_phase_dimensions_follow_bones_and_eight_value_muscles() {
        let phase_start = 5 * 4 + 4;
        assert!(!is_phase_dimension(phase_start, phase_start));
        assert!(is_phase_dimension(phase_start + 5, phase_start));
        assert!(!is_phase_dimension(phase_start + 6, phase_start));
        assert!(is_phase_dimension(phase_start + 13, phase_start));
    }
}
