use super::*;

/// Nanoseconds of breeding spent planning, emitting offspring, and writing
/// them into their block, since the last `take_breed_nanos`.
pub static BREED_NANOS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Returns and clears the breeding timers.
pub fn take_breed_nanos() -> [u64; 3] {
    std::array::from_fn(|i| BREED_NANOS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}
/// Children written after their part of the block's arena (low 32 bits)
/// and blocks bred into a new arena because the old one was still shared
/// (high 32 bits), since the last `take_breed_late`.
pub static BREED_LATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Returns and clears `BREED_LATE` as (late children, new arenas).
pub fn take_breed_late() -> (u64, u64) {
    let v = BREED_LATE.swap(0, std::sync::atomic::Ordering::Relaxed);
    (v & 0xffff_ffff, v >> 32)
}

/// Share of the structural and novelty children of a reshaped nursery that
/// take a limb from a body of another plan (a hybrid; Arnold, 1997).
const RESHAPED_CROSS_SHARE: f32 = 0.3;

/// The emitter mix of `island`: each isolated island leans a few points
/// toward one emitter, so the islands develop different habits (Whitley,
/// 1999). The hub, the wild islands and the nurseries keep the mix.
fn island_weights(weights: &[f64; qd::EMITTER_COUNT], island: usize) -> [f64; qd::EMITTER_COUNT] {
    // (from, to): five points move from one emitter to another.
    let lean = match island {
        0 => Some((Emitter::Structural, Emitter::Cma)),
        1 => Some((Emitter::Novelty, Emitter::Structural)),
        2 => Some((Emitter::Cma, Emitter::Novelty)),
        3 => Some((Emitter::Cma, Emitter::Structural)),
        _ => None,
    };
    let mut out = *weights;
    if let Some((from, to)) = lean {
        let moved = out[from.index()].min(0.05);
        out[from.index()] -= moved;
        out[to.index()] += moved;
    }
    out
}

/// Share of CMA offspring whose parent is one of its island's fastest 1% of
/// elites; the rest sample by local competition. Spending more on the best
/// elites raised the best distance by about half in fixed-seed tests.
const TOP_PARENT_SHARE: f32 = 0.5;
/// Share of structural and novelty children that graft a limb from an elite
/// with a different body plan.
const CROSS_PLAN_MATE_SHARE: f32 = 0.15;
/// Share of those top-elite CMA offspring bred by an island optimizer
/// (separable CMA-ES in physical units) on one of its fastest designs.
const OPTIMIZER_SHARE: f32 = 0.5;
/// How many of an island's fastest elites the breeding plan ranks.
const FASTEST_ELITES: usize = 512;

struct OffspringPlan {
    plan: CandidatePlan,
    parent_id: Option<u64>,
    protection: u32,
}

impl Experiment {
    /// The next elite queued for the island of `slot`. A nursery slot takes
    /// none, so a re-tested elite competes in its island's archive.
    fn reseed_for_slot(&mut self, slot: usize) -> Option<Creature> {
        let islands = island_count();
        if qd::is_nursery_slot(slot, islands) {
            return None;
        }
        self.reseed.pop(qd::island_of_slot(slot, islands))
    }
    /// Chooses emitters, parents, and CMA slots for offspring in `slots`.
    /// The breeding `round` salts the random streams, so no two blocks
    /// repeat a draw.
    fn plan_offspring(
        &mut self,
        cfg: &Config,
        generation: u32,
        round: u64,
        slots: &[usize],
    ) -> Vec<OffspringPlan> {
        let profile = std::env::var_os("EVOLUTION_PROFILE_BREED").is_some();
        let mut plan_times = [0.0f64; 4];
        let mut section = std::time::Instant::now();
        self.ensure_islands();
        for island in &mut self.islands {
            island.ensure_least_visited();
        }
        let weights = qd::emitter_weights(&self.emitter_stats);
        let mut reset_cma = HashMap::<(usize, qd::Niche, u64), usize>::new();
        // CMA slot lookup keyed by (island, niche, body plan): an emitter
        // samples around one island's elite, so it serves only that island.
        // The bucket stores the full key, so the per-offspring probe hashes
        // and compares without cloning the topology vector; clones are only
        // paid when a slot is created or replaced.
        type CmaKey = (usize, qd::Niche, u64);
        struct CmaLookup {
            buckets: HashMap<u64, Vec<(CmaKey, usize)>>,
        }
        impl CmaLookup {
            fn hash(island: usize, niche: &qd::Niche, plan: u64) -> u64 {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                island.hash(&mut hasher);
                niche.hash(&mut hasher);
                plan.hash(&mut hasher);
                hasher.finish()
            }
            fn get(&self, island: usize, niche: &qd::Niche, plan: u64) -> Option<usize> {
                self.buckets
                    .get(&Self::hash(island, niche, plan))?
                    .iter()
                    .find(|((i, n, t), _)| *i == island && n == niche && *t == plan)
                    .map(|(_, index)| *index)
            }
            fn insert(&mut self, key: CmaKey, index: usize) {
                let bucket = self
                    .buckets
                    .entry(Self::hash(key.0, &key.1, key.2))
                    .or_default();
                if let Some(entry) = bucket.iter_mut().find(|(stored, _)| *stored == key) {
                    entry.1 = index;
                } else {
                    bucket.push((key, index));
                }
            }
            fn remove(&mut self, island: usize, niche: &qd::Niche, plan: u64, index: usize) {
                let hash = Self::hash(island, niche, plan);
                let Some(bucket) = self.buckets.get_mut(&hash) else {
                    return;
                };
                bucket.retain(|((i, n, t), stored_index)| {
                    !(*i == island && n == niche && *t == plan && *stored_index == index)
                });
                if bucket.is_empty() {
                    self.buckets.remove(&hash);
                }
            }
        }
        let mut cma_lookup = CmaLookup {
            buckets: HashMap::new(),
        };
        for (index, cma) in self.cma_emitters.iter().enumerate() {
            cma_lookup.insert(
                (cma.island, cma.niche.clone(), cma.topology.plan_key()),
                index,
            );
        }
        let mut used_cma = vec![false; self.cma_emitters.len()];
        let mut out = Vec::with_capacity(slots.len());
        // Phase A: emitter choice and parent sampling against the start-of-batch
        // archive. Each creature has its own deterministic RNG, so parallel order
        // does not change the draws. last_parent is snapshotted instead of updating
        // mid-loop; visit() and CMA slot allocation stay sequential below.
        struct PlanPrep {
            emitter: Emitter,
            parent: Option<usize>,
            parent_id: Option<u64>,
            protection: u32,
            emitter_stale: bool,
            mate: Option<usize>,
            island: usize,
            /// A fast elite whose design's optimizer breeds this offspring.
            optimize: bool,
            /// A reshaped child of an island elite for a reshaped nursery.
            seeded: bool,
        }
        // Each island's elites by body plan key, grouped for crossover
        // partners: (key, slot) sorted, so a plan's elites are one run.
        let by_plan: Vec<Vec<(u64, u32)>> = self
            .islands
            .par_iter()
            .map(|island| {
                let mut keyed: Vec<(u64, u32)> = (0..island.entries.len())
                    .map(|slot| (island.plan_key(slot), slot as u32))
                    .collect();
                keyed.sort_unstable();
                keyed
            })
            .collect();
        // The elites of `island` that share the body plan `key`.
        let same_plan = |island: usize, key: u64| -> &[(u64, u32)] {
            let keyed = &by_plan[island];
            let first = keyed.partition_point(|&(k, _)| k < key);
            let len = keyed[first..].partition_point(|&(k, _)| k == key);
            &keyed[first..first + len]
        };
        plan_times[0] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        let seed = cfg.seed ^ round.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        // Behavior elites per island, fastest first. Both the exploitation
        // pool and the optimizer targets read this order.
        let orders: Vec<Vec<usize>> = self
            .islands
            .iter()
            .map(|island| {
                let mut order: Vec<usize> = (0..island.entries.len())
                    .filter(|&i| !qd::is_morphology_niche(&island.entries[i].niche))
                    .collect();
                // The exploitation pool and the optimizer targets only read
                // the fastest few hundred.
                let faster = |&a: &usize, &b: &usize| {
                    island.entries[b]
                        .fitness
                        .total_cmp(&island.entries[a].fitness)
                        .then(a.cmp(&b))
                };
                if order.len() > FASTEST_ELITES {
                    order.select_nth_unstable_by(FASTEST_ELITES - 1, faster);
                    order.truncate(FASTEST_ELITES);
                }
                order.sort_unstable_by(faster);
                order
            })
            .collect();
        // Each island's fastest elites, for exploitation: 1% of the movement
        // grid (at least 4), however many body classes the archive holds.
        let top_parents: Vec<Vec<usize>> = orders
            .iter()
            .map(|order| {
                let count = (order.len().min(qd::MOVEMENT_CELLS) / 100).max(4);
                order[..count.min(order.len())].to_vec()
            })
            .collect();
        // How rare each elite's clade is in its island, from 0 (the whole
        // archive) to 1 (one elite).
        if qd::RARITY_WEIGHT > 0.0
            && (self.clade_rarity.0 != generation
                || self.clade_rarity.1.len() != self.islands.len())
        {
            let rarities = self
                .islands
                .iter()
                .map(|island| self.clade_rarity_of(island))
                .collect();
            self.clade_rarity = (generation, rarities);
        }
        let rarities = &self.clade_rarity.1;
        let no_rarity = Vec::new();
        // An island's optimizer works on its fastest design: a body plan with
        // a gait cadence band. When the island has not set a record for a
        // while, it turns to its next fastest designs in turn, so one stuck
        // design does not take all local search.
        self.island_progress
            .resize(self.islands.len(), (f32::NEG_INFINITY, generation));
        let optimizer_targets: Vec<Option<usize>> = self
            .islands
            .iter()
            .zip(&top_parents)
            .zip(&orders)
            .zip(&mut self.island_progress)
            .map(|(((island, top), order), progress)| {
                let best = *top.first()?;
                let fitness = island.entries[best].fitness;
                if fitness > progress.0 {
                    *progress = (fitness, generation);
                }
                let mut plans: Vec<usize> = Vec::new();
                // A design is a body plan with a gait cadence band.
                let design = |i: usize| (&island.entries[i].topology, island.entries[i].niche.0[1]);
                for &i in order {
                    if plans.len() >= 4 {
                        break;
                    }
                    if !plans.iter().any(|&p| design(p) == design(i)) {
                        plans.push(i);
                    }
                }
                let turn = (generation.saturating_sub(progress.1) / OPTIMIZER_STALL) as usize;
                Some(plans[turn % plans.len()])
            })
            .collect();
        // A second optimizer target per island: the fastest elite of its
        // rarest clade, so local search also climbs a design the island is
        // about to lose (Fontaine et al., 2020, CMA-ME on several targets).
        let rare_targets: Vec<Option<usize>> = self
            .islands
            .iter()
            .enumerate()
            .map(|(island, archive)| {
                if qd::bio_off(64) {
                    return None;
                }
                let rarity = rarities.get(island)?;
                let top = rarity.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if !top.is_finite() || top <= 0.0 {
                    return None;
                }
                (0..archive.entries.len().min(rarity.len()))
                    .filter(|&i| {
                        rarity[i] >= top && !qd::is_morphology_niche(&archive.entries[i].niche)
                    })
                    .max_by(|&a, &b| {
                        archive.entries[a]
                            .fitness
                            .total_cmp(&archive.entries[b].fitness)
                            .then(b.cmp(&a))
                    })
            })
            .collect();
        plan_times[1] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        let plan_prep: Vec<PlanPrep> = slots
            .par_iter()
            .map(|&i| {
                let mut rng = Rng::new(seed, generation, i);
                let arena = qd::arena_of_slot(i, self.islands.len());
                let reshaped = qd::is_reshaped_arena(arena, self.islands.len());
                // Bodies the island turned away fill a reshaped nursery.
                // While it is empty its slots breed structural children of
                // the island's elites instead.
                let home = qd::island_of_slot(i, island_count());
                let seeded = reshaped
                    && !self.islands[home].entries.is_empty()
                    && self.islands[arena].entries.is_empty();
                // The archive the child's parent comes from.
                let island = if seeded { home } else { arena };
                let archive = &self.islands[island];
                let archive_empty = archive.entries.is_empty();
                let emitter = if seeded {
                    Emitter::Structural
                } else if archive_empty
                    || (arena >= island_count()
                        && !reshaped
                        && rng.unit() < qd::NURSERY_FRESH_SHARE)
                {
                    Emitter::Restart
                } else {
                    qd::choose_emitter(&mut rng, &island_weights(&weights, island))
                };
                let emitter_stale = self.emitter_stats[emitter.index()].stale();
                let avoid = None;
                let mut optimize = false;
                let mut from_reserve = false;
                let parent = if emitter == Emitter::Restart || archive_empty {
                    None
                } else if seeded {
                    archive.sample_local_competitive(
                        &mut rng,
                        avoid,
                        rarities.get(island).unwrap_or(&no_rarity),
                    )
                } else if emitter == Emitter::Structural
                    && rng.unit() < qd::MORPHOLOGY_PARENT_FRACTION
                {
                    // Each island keeps its own morphology reserve.
                    let drawn = archive.sample_morphology(&mut rng, avoid);
                    from_reserve = drawn.is_some();
                    drawn.or_else(|| {
                        let rarity = rarities.get(island).unwrap_or(&no_rarity);
                        archive.sample_local_competitive(&mut rng, avoid, rarity)
                    })
                } else if emitter == Emitter::Novelty && rng.unit() < 0.5 {
                    // Half the novelty parents are far from the others in
                    // body, not in behavior.
                    archive
                        .sample_body_novel(&mut rng)
                        .or_else(|| archive.sample_novel(&mut rng, avoid))
                } else if emitter == Emitter::Novelty || emitter_stale {
                    archive.sample_novel(&mut rng, avoid)
                } else if emitter == Emitter::Cma
                    && !top_parents[island].is_empty()
                    && rng.unit() < TOP_PARENT_SHARE
                {
                    // Half of these come from the island's optimizer for one
                    // of its fastest designs; the rest explore around the top
                    // elites.
                    optimize = rng.unit() < OPTIMIZER_SHARE;
                    let second = rare_targets.get(island).copied().flatten();
                    Some(match second {
                        Some(target) if optimize && rng.unit() < 0.5 => target,
                        _ if optimize => {
                            optimizer_targets[island].unwrap_or(top_parents[island][0])
                        }
                        _ => top_parents[island][rng.index(top_parents[island].len())],
                    })
                } else {
                    let rarity = rarities.get(island).unwrap_or(&no_rarity);
                    archive.sample_local_competitive(&mut rng, avoid, rarity)
                };
                let parent_id = parent.map(|index| archive.entries[index].creature.id);
                let protection = if matches!(emitter, Emitter::Structural | Emitter::Novelty) {
                    generation.saturating_add(qd::PROTECTION_GENERATIONS)
                } else {
                    parent
                        .map(|index| archive.entries[index].protected_until)
                        .unwrap_or(0)
                };
                let mate = match (emitter, parent) {
                    (Emitter::Structural | Emitter::Novelty, Some(p))
                        if !from_reserve && rng.unit() < 0.2 =>
                    {
                        Some(same_plan(island, archive.plan_key(p)))
                            .filter(|group| group.len() > 1)
                            .map(|group| group[rng.index(group.len())].1 as usize)
                            .filter(|&m| m != p)
                    }
                    _ => None,
                };
                // Sometimes the mate has another body plan: the child gets one of
                // its limbs grafted on (see `evolution::mated`).
                let mate = mate.or_else(|| match (emitter, parent) {
                    (Emitter::Structural | Emitter::Novelty, Some(p))
                        if !from_reserve
                            && rng.unit()
                                < if reshaped && !qd::bio_off(32) {
                                    RESHAPED_CROSS_SHARE
                                } else {
                                    CROSS_PLAN_MATE_SHARE
                                } =>
                    {
                        let other = rng.index(archive.entries.len());
                        (other != p && archive.plan_key(other) != archive.plan_key(p))
                            .then_some(other)
                    }
                    _ => None,
                });
                PlanPrep {
                    emitter,
                    parent,
                    parent_id,
                    protection,
                    emitter_stale,
                    mate,
                    island,
                    optimize,
                    seeded,
                }
            })
            .collect();
        plan_times[2] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        for prep in plan_prep {
            let PlanPrep {
                emitter,
                parent,
                parent_id,
                protection,
                emitter_stale,
                mate,
                island,
                optimize,
                seeded,
            } = prep;
            let cma_index = if emitter == Emitter::Cma {
                if let Some(parent_index) = parent {
                    let elite = &self.islands[island].entries[parent_index];
                    let template = elite.creature.unpack();
                    let plan = self.islands[island].plan_key(parent_index);
                    // Each island runs one optimizer per design. It starts from
                    // the design's fastest elite and then follows its own mean,
                    // so recentering on every lucky new best does not throw
                    // away its progress. A converged one restarts.
                    let lookup_niche = if optimize {
                        qd::optimizer_niche(island, elite.niche.0[1])
                    } else {
                        elite.niche.clone()
                    };
                    let converged = |i: &usize| self.cma_emitters[*i].converged() && !used_cma[*i];
                    let mut index = if optimize {
                        cma_lookup
                            .get(island, &lookup_niche, plan)
                            .filter(|i| !converged(i))
                    } else if emitter_stale {
                        reset_cma
                            .get(&(island, lookup_niche.clone(), plan))
                            .copied()
                    } else {
                        cma_lookup.get(island, &lookup_niche, plan)
                    };
                    if index.is_none() {
                        let restart = cma_lookup
                            .get(island, &lookup_niche, plan)
                            .filter(|_| optimize);
                        let replacement = if restart.is_some() {
                            restart
                        } else if self.cma_emitters.len() < qd::CMA_LIMIT {
                            Some(self.cma_emitters.len())
                        } else {
                            self.cma_emitters
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| !used_cma[*i])
                                .min_by_key(|(_, cma)| cma.last_used_generation)
                                .map(|(i, _)| i)
                        };
                        if let Some(slot) = replacement {
                            let mut new = if optimize {
                                // Another optimizer of this island for the
                                // same plan lends its learned step sizes,
                                // unless this is a restart after converging.
                                // Other islands never lend: their step sizes
                                // carry what their search learned.
                                self.cma_emitters
                                    .iter()
                                    .filter(|c| {
                                        c.optimizing()
                                            && c.island == island
                                            && c.topology.plan_key() == plan
                                            && !c.converged()
                                    })
                                    .max_by_key(|c| c.last_used_generation)
                                    .map_or_else(
                                        || {
                                            CmaEmitter::optimizer(
                                                template.clone(),
                                                lookup_niche.clone(),
                                                generation,
                                            )
                                        },
                                        |c| {
                                            c.recentered(
                                                template.clone(),
                                                lookup_niche.clone(),
                                                generation,
                                            )
                                        },
                                    )
                            } else {
                                CmaEmitter::new(template.clone(), elite.niche.clone(), generation)
                            };
                            new.island = island;
                            if slot == self.cma_emitters.len() {
                                self.cma_emitters.push(new);
                                used_cma.push(false);
                            } else {
                                let old = &self.cma_emitters[slot];
                                let (old_island, old_niche, old_plan) =
                                    (old.island, old.niche.clone(), old.topology.plan_key());
                                if cma_lookup.get(old_island, &old_niche, old_plan) == Some(slot) {
                                    cma_lookup.remove(old_island, &old_niche, old_plan, slot);
                                }
                                reset_cma.retain(|_, index| *index != slot);
                                self.cma_emitters[slot] = new;
                            }
                            let new_niche = self.cma_emitters[slot].niche.clone();
                            let new_plan = self.cma_emitters[slot].topology.plan_key();
                            cma_lookup.insert((island, new_niche, new_plan), slot);
                            if emitter_stale && !optimize {
                                reset_cma.insert((island, lookup_niche.clone(), plan), slot);
                            }
                            index = Some(slot);
                        }
                    }
                    if let Some(index) = index {
                        used_cma[index] = true;
                        self.cma_emitters[index].last_used_generation = generation;
                    }
                    index
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(parent_index) = parent {
                self.islands[island].visit(parent_index);
            }
            out.push(OffspringPlan {
                plan: CandidatePlan {
                    emitter,
                    parent,
                    cma: cma_index,
                    mate,
                    seed: seeded,
                },
                parent_id,
                protection,
            });
        }
        plan_times[3] = section.elapsed().as_secs_f64();
        if profile {
            eprintln!(
                "Plan profile: generation {generation}, by_plan {:.6} s, order/optimizer {:.6} s, sampling {:.6} s, cma/visit {:.6} s",
                plan_times[0], plan_times[1], plan_times[2], plan_times[3]
            );
        }
        out
    }
    /// Plans offspring for `slots` in the next breeding round, as a block's
    /// breeding does, and returns the plans with that round. For
    /// `examples/breed_bench.rs`.
    #[doc(hidden)]
    pub fn plan_for_bench(&mut self, slots: &[usize]) -> (Vec<CandidatePlan>, u64) {
        let cfg = self.config.clone();
        self.breed_round += 1;
        let plans = self.plan_offspring(&cfg, self.generation, self.breed_round, slots);
        (
            plans.into_iter().map(|p| p.plan).collect(),
            self.breed_round,
        )
    }
    /// Breeds a block for ring slots `first..first + count` from the current
    /// archives with the current settings, into `arena`: the genes of the
    /// block bred for these slots last time, whose memory the new block
    /// reuses when nothing else holds it. Elites queued by a world change
    /// take the slots of their own islands first. An island without elites
    /// breeds new random bodies.
    pub(super) fn breed_block(
        &mut self,
        first: usize,
        count: usize,
        arena: Arc<Population>,
    ) -> Block {
        let slots: Vec<usize> = (first..first + count).collect();
        let cfg = self.config.clone();
        self.breed_round += 1;
        let started = std::time::Instant::now();
        let planned = self.plan_offspring(&cfg, self.generation, self.breed_round, &slots);
        let planned_at = started.elapsed();
        // The parents as they are now, for the generation dump's rows.
        let dump_parents: Option<Vec<dump::Parent>> = self.dump_breeding().then(|| {
            planned
                .iter()
                .zip(&slots)
                .map(|(p, &slot)| {
                    let arena = if p.plan.seed {
                        qd::island_of_slot(slot, island_count())
                    } else {
                        qd::arena_of_slot(slot, self.islands.len())
                    };
                    let elite = p
                        .plan
                        .parent
                        .and_then(|i| self.islands.get(arena)?.entries.get(i));
                    dump::Parent::of(
                        elite,
                        p.plan
                            .cma
                            .and_then(|c| self.cma_emitters.get(c))
                            .is_some_and(CmaEmitter::optimizing),
                    )
                })
                .collect()
        });
        let mut births: Vec<Birth> = planned
            .iter()
            .map(|p| Birth {
                emitter: p.plan.emitter,
                cma: p.plan.cma,
                parent_id: p.parent_id,
                mate: p.plan.mate.is_some(),
                protection: p.protection,
            })
            .collect();
        // Reseeded elites first, then the children.
        let mut lead: Vec<(usize, Creature)> = Vec::new();
        if !self.reseed.is_empty() {
            for (k, &slot) in slots.iter().enumerate() {
                if let Some(elite) = self.reseed_for_slot(slot) {
                    lead.push((k, elite));
                    births[k] = Birth::RANDOM;
                }
            }
        }
        // After a world change the old champions keep breeding in their
        // island's own slots for a few generations.
        if self.generation < self.refuge.until {
            let islands = island_count();
            let reseeded: std::collections::HashSet<usize> = lead.iter().map(|&(k, _)| k).collect();
            for (k, &slot) in slots.iter().enumerate() {
                if reseeded.contains(&k) || qd::is_nursery_slot(slot, islands) {
                    continue;
                }
                let island = qd::island_of_slot(slot, islands);
                if let Some(child) =
                    self.refuge
                        .child(island, slot, &cfg, self.generation, self.breed_round)
                {
                    lead.push((k, child));
                    births[k] = Birth::RANDOM;
                }
            }
            lead.sort_by_key(|&(k, _)| k);
        }
        // The founder bank breeds in the main islands' own slots, and the hall
        // of fame in the hub's.
        if !self.founders.is_empty() || !self.hall.is_empty() {
            let islands = island_count();
            let hub = hub_island();
            let taken: std::collections::HashSet<usize> = lead.iter().map(|&(k, _)| k).collect();
            for (k, &slot) in slots.iter().enumerate() {
                let island = qd::island_of_slot(slot, islands);
                if taken.contains(&k) || qd::is_nursery_slot(slot, islands) || qd::is_wild(island) {
                    continue;
                }
                let mut rng = evolution::Rng::stream(
                    cfg.seed ^ 0x0066_6f75_6e64,
                    self.generation,
                    self.breed_round,
                    slot,
                );
                let draw = rng.unit();
                let parent = if !self.founders.is_empty() && draw < FOUNDER_SHARE {
                    self.founders[rng.index(self.founders.len())].clone()
                } else if island == hub
                    && !self.hall.is_empty()
                    && draw < FOUNDER_SHARE + HALL_SHARE
                {
                    self.hall[rng.index(self.hall.len())].clone()
                } else {
                    continue;
                };
                let scale = if rng.unit() < 0.1 { 2.0 } else { 0.75 };
                let mut child = evolution::mutate_locally(parent, &cfg, &mut rng, scale);
                if rng.unit() < 0.3 {
                    evolution::structural_mutation_any(&mut child, &cfg, &mut rng);
                }
                child.id = evolution::bred_id(self.breed_round, slot);
                lead.push((k, child));
                births[k] = Birth::RANDOM;
            }
            lead.sort_by_key(|&(k, _)| k);
        }
        // Wild champions in the hub's pen breed in the hub's own slots.
        if !self.pen.is_empty() {
            let islands = island_count();
            let hub = hub_island();
            let taken: std::collections::HashSet<usize> = lead.iter().map(|&(k, _)| k).collect();
            for (k, &slot) in slots.iter().enumerate() {
                if taken.contains(&k)
                    || qd::is_nursery_slot(slot, islands)
                    || qd::island_of_slot(slot, islands) != hub
                {
                    continue;
                }
                let mut rng = evolution::Rng::stream(
                    cfg.seed ^ 0x0070_656e,
                    self.generation,
                    self.breed_round,
                    slot,
                );
                if rng.unit() >= PEN_SHARE {
                    continue;
                }
                let parent = self.pen[rng.index(self.pen.len())].0.clone();
                let scale = if rng.unit() < 0.1 { 2.0 } else { 0.75 };
                let mut child = evolution::mutate_locally(parent, &cfg, &mut rng, scale);
                if rng.unit() < 0.3 {
                    evolution::structural_mutation_any(&mut child, &cfg, &mut rng);
                }
                child.id = evolution::bred_id(self.breed_round, slot);
                lead.push((k, child));
                births[k] = Birth::RANDOM;
            }
            lead.sort_by_key(|&(k, _)| k);
        }
        let mut taken = vec![false; count];
        for &(k, _) in &lead {
            taken[k] = true;
        }
        let positions: Vec<usize> = (0..count).filter(|&k| !taken[k]).collect();
        let bred_slots: Vec<usize> = positions.iter().map(|&k| slots[k]).collect();
        let bred_plans: Vec<CandidatePlan> = positions.iter().map(|&k| planned[k].plan).collect();
        // The arena's memory is reused when no unit or save still holds it;
        // otherwise its sizes guide a new one.
        let (mut population, hint) = match Arc::try_unwrap(arena) {
            Ok(population) => (population, None),
            Err(shared) => (Population::default(), Some(shared)),
        };
        let late = population.breed(
            count,
            hint.as_deref(),
            &mut lead,
            &self.islands,
            &self.cma_emitters,
            &bred_plans,
            &bred_slots,
            &positions,
            &cfg,
            self.generation,
            self.breed_round,
        );
        // Each creature's flags for its trial: the audit lane, and the
        // exemption of nurseries and immigrants from the early rungs.
        // The median fitness of each arena's behavior elites: a parent
        // above it is a strong one.
        let medians: Vec<f32> = if cfg.rungs.is_some() {
            self.islands
                .iter()
                .map(|archive| {
                    let mut v: Vec<f32> = archive
                        .entries
                        .iter()
                        .filter(|e| !qd::is_morphology_niche(&e.niche))
                        .map(|e| e.fitness)
                        .collect();
                    if v.is_empty() {
                        f32::NEG_INFINITY
                    } else {
                        let mid = v.len() / 2;
                        *v.select_nth_unstable_by(mid, f32::total_cmp).1
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        population.flags.clear();
        population.flags.extend((0..count).map(|k| {
            let slot = first + k;
            let mut flags = 0u8;
            if crate::rungs::is_audit(cfg.seed, self.breed_round, slot)
                && !qd::is_wild(qd::island_of_slot(slot, island_count()))
            {
                flags |= crate::rungs::AUDIT;
            }
            let arenas = self.islands.len().max(arena_count());
            let arena = qd::arena_of_slot(slot, arenas);
            // A child whose parent is a strong elite that the rules would
            // stop skips those rungs.
            if let Some(rules) = &cfg.rungs {
                let parent = births[k].parent_id.and_then(|id| self.lineage.get(&id));
                let strong = parent
                    .is_some_and(|a| medians.get(arena).is_none_or(|&median| a.fitness >= median));
                flags |= crate::rungs::parent_exemptions(rules, parent.map(|a| &a.rung), strong);
            }
            if births[k].emitter == Emitter::Restart {
                flags |= crate::rungs::EXEMPT;
            }
            // A nursery body is exempt from the early rungs and held to the
            // screen bar of its own kind.
            if qd::is_reshaped_arena(arena, arenas) {
                flags |= crate::rungs::EXEMPT | crate::rungs::RESHAPED;
            } else if arena >= island_count() {
                flags |= crate::rungs::EXEMPT | crate::rungs::YOUNG;
            }
            flags
        }));
        if let (Some(parents), Some(dump)) = (dump_parents, &self.dump) {
            let reseeded: Vec<usize> = lead.iter().map(|&(k, _)| k).collect();
            dump.lock().unwrap_or_else(|e| e.into_inner()).bred(
                first,
                &population,
                &births,
                parents,
                &reseeded,
            );
        }
        let total = started.elapsed();
        let add = |k: usize, d: std::time::Duration| {
            BREED_NANOS[k].fetch_add(d.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        };
        // Children are written into the arena as they are bred, so the
        // write stage is part of emitting.
        add(0, planned_at);
        add(1, total.saturating_sub(planned_at));
        BREED_LATE.fetch_add(late as u64, std::sync::atomic::Ordering::Relaxed);
        if hint.is_some() {
            BREED_LATE.fetch_add(1 << 32, std::sync::atomic::Ordering::Relaxed);
        }
        Block {
            first,
            population: Arc::new(population),
            births,
            config: Arc::new(cfg),
        }
    }
}
