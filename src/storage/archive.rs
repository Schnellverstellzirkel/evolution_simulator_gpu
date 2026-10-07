//! Absorbs a block's results into the archives and writes a generation's
//! history row.
//!
//! `Experiment::archive_block` offers each creature of a block to the archives
//! it competes in, trains the CMA emitters on the results, updates the emitter
//! statistics and records lineage. `Experiment::push_archive_stats` adds the
//! generation's row to `history`. `Experiment::absorb` in `storage.rs` calls
//! the first, and `end_generation` calls the second.

use super::*;

impl Experiment {
    /// Offers the creatures of block `k` to the archives, using their final
    /// results `finals`, and returns how many trials failed. A trial failed
    /// when its score is not finite or is at or below `FAILED`.
    ///
    /// Each creature goes, in block order, to the archive it breeds for, which
    /// is an island or a nursery of the island. A structural or novelty child
    /// that takes no cell may enter the morphology reserve of that archive. An
    /// island sends a new body plan that gets neither a cell nor a reserve
    /// place to its nursery of reshaped bodies. A creature of a main island is
    /// also a candidate for the global archive, which is offered only the best
    /// candidate of the block for each of its cells. Then the CMA emitters
    /// learn from their samples, the emitter statistics take the block's
    /// attempts, discoveries, improvements and rewards, and every creature
    /// that entered an archive gets a lineage record.
    ///
    /// Screened and excluded results, and every result of a `stale` block,
    /// enter no archive. A block is stale when it ran in a world that has since
    /// changed. With `kinds`, one byte per creature, it also marks what each
    /// creature entered: `dump::ISLAND`, `dump::NURSERY`, `dump::RESERVE` and
    /// `dump::GLOBAL`.
    pub(super) fn archive_block(
        &mut self,
        k: usize,
        finals: &[EvaluationMetrics],
        stale: bool,
        mut kinds: Option<&mut [u8]>,
    ) -> usize {
        // `EVOLUTION_PROFILE_BREED` prints how long each section took.
        // `timings` follows the order of the printed line, which is not the
        // order the sections run in. The prefilter runs first.
        let profile = std::env::var_os("EVOLUTION_PROFILE_BREED").is_some();
        let mut timings = [0.0f64; 7];
        let mut section = std::time::Instant::now();
        self.ensure_islands();
        let block = self.blocks[k].clone();
        let population = &*block.population;
        let births = &block.births;
        let first = block.first;
        // The block positions of the creatures that entered any archive.
        let mut entered: Vec<usize> = Vec::new();
        // An emitter's `last_parent` is a slot of the global archive, and an
        // offer can put another creature in a slot. So the parents are read as
        // ids here, and their slots are found again after the offers.
        let previous_parent_ids: [Option<u64>; qd::EMITTER_COUNT] = std::array::from_fn(|i| {
            self.emitter_stats[i]
                .last_parent
                .and_then(|index| self.archive.entries.get(index))
                .map(|elite| elite.creature.id)
        });
        // Per emitter, for `qd::record_emitter_batch`: the new cells and
        // reserve places it filled, the elites it replaced, and the rewards of
        // both.
        let mut discoveries = [0u64; qd::EMITTER_COUNT];
        let mut improvements = [0u64; qd::EMITTER_COUNT];
        let mut rewards = [0.0f64; qd::EMITTER_COUNT];
        // Per CMA emitter, the creatures it sampled in this block, each with
        // the key that ranks it. An optimizer ranks by distance alone.
        let mut cma_samples = vec![Vec::<(usize, f32)>::new(); self.cma_emitters.len()];
        let optimizers: Vec<bool> = self.cma_emitters.iter().map(|c| c.optimizing()).collect();
        /// What the prefilter works out for one creature.
        struct Prep {
            /// Its behavior descriptor, which picks its cell.
            descriptor: qd::Descriptor,
            emitter: Emitter,
            score: f32,
            /// Its score is its confirmation trial's (`EvaluationMetrics::fine`).
            fine: bool,
            /// The generation until which its niche is protected from other
            /// body plans.
            protection: u32,
            /// It may be offered to the global archive. It has a usable score,
            /// is not screened, comes from a main island, and beats its cell's
            /// elite or finds the cell empty. The scan after the island offers
            /// keeps only the best candidate of each cell.
            behavior_candidate: bool,
            /// A child of the structural or novelty emitter with a usable score
            /// that is not screened. It may enter the morphology reserve of its
            /// archive, and `plan` is its body plan key (0 for any other
            /// creature).
            structural: bool,
            plan: u64,
            /// The result enters no archive: its trial was screened or
            /// excluded, or its block is stale.
            screened: bool,
            /// For a CMA sample with a usable score, the fitness of the global
            /// archive's elite in its cell at the start of the block. None for
            /// any other creature and for an empty cell.
            elite_before: Option<f32>,
        }
        let arenas = self.islands.len().max(arena_count());
        let positions: Vec<usize> = (0..block.len()).collect();
        // There are more than 256 arenas (5 main and 100 wild islands, each
        // with three kinds of archive), so an arena index needs a `u16`.
        const _: () =
            assert!((qd::MAIN_ISLANDS + qd::WILD_ISLANDS) * qd::ARENA_KINDS <= u16::MAX as usize);
        // The archive each creature breeds for and competes in
        // (`qd::arena_of_slot`): its island or a nursery of its island.
        let arena_of: Vec<u16> = positions
            .par_iter()
            .map(|&j| qd::arena_of_slot(first + j, arenas) as u16)
            .collect();
        // The prefilter runs in parallel against the global archive as it
        // stood at the start of the block. It works out each creature's
        // descriptor and whether the creature is a candidate for the global
        // archive. An occupant's fitness only ever rises, so a creature that
        // fails now would fail later too. The offers themselves still go in one
        // at a time, in block order.
        let prep: Vec<Prep> = positions
            .par_iter()
            .map(|&j| {
                let m = &finals[j];
                let score = m.fitness;
                let birth = births[j];
                let emitter = birth.emitter;
                let genome = &population.genomes[j];
                let nodes =
                    &population.nodes[genome.node_start..genome.node_start + genome.node_count];
                let muscles = &population.muscles
                    [genome.muscle_start..genome.muscle_start + genome.muscle_count];
                let descriptor = qd::descriptor(nodes, muscles, m.behavior);
                let screened = stale || m.screened || m.excluded;
                // A creature of a nursery or of a wild island is offered to its
                // own archive only, not to the global archive.
                let nursery = arena_of[j] as usize >= island_count()
                    || qd::is_wild(qd::island_of_slot(first + j, island_count()));
                let valid = score.is_finite() && score > FAILED && !screened;
                let behavior_candidate = if valid && !nursery {
                    let niche = self.archive.cell_of(descriptor);
                    match self.archive.slot_for(&niche) {
                        Some(slot) => score > self.archive.entries[slot].fitness,
                        None => self.archive.behavior_count() < self.archive.limit(),
                    }
                } else {
                    false
                };
                let structural = valid && matches!(emitter, Emitter::Structural | Emitter::Novelty);
                let plan = if structural {
                    qd::plan_key_of_population(population, j)
                } else {
                    0
                };
                let elite_before = (emitter == Emitter::Cma)
                    .then_some(birth.cma)
                    .flatten()
                    .filter(|_| score.is_finite() && score > FAILED)
                    .and_then(|_| self.archive.slot_for(&self.archive.cell_of(descriptor)))
                    .map(|slot| self.archive.entries[slot].fitness);
                Prep {
                    descriptor,
                    emitter,
                    score,
                    fine: m.fine,
                    protection: birth.protection,
                    behavior_candidate,
                    structural,
                    plan,
                    screened,
                    elite_before,
                }
            })
            .collect();
        timings[2] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // Every creature is offered to the archive it breeds for. A child with
        // a new body plan that takes no cell may enter the morphology reserve
        // of that archive. The archives are independent and each takes its
        // offers in block order, so they run in parallel.
        let generation = self.generation;
        /// What one archive took from the block: the positions that entered a
        /// cell or the reserve, the emitter index and offer of each reserve
        /// entry, the positions that entered the reserve, and the positions of
        /// the new body plans of an island that took neither a cell nor a
        /// reserve place.
        type IslandResult = (Vec<usize>, Vec<(usize, qd::Offer)>, Vec<usize>, Vec<usize>);
        // Each archive's creatures, in block order.
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); self.islands.len()];
        for (j, &arena) in arena_of.iter().enumerate() {
            if let Some(list) = members.get_mut(arena as usize) {
                list.push(j);
            }
        }
        let island_results: Vec<IslandResult> = self
            .islands
            .par_iter_mut()
            .zip(&members)
            .enumerate()
            .map(|(island, (archive, members))| {
                let mut entered = Vec::new();
                let mut reserve_offers = Vec::new();
                let mut reserve_entered = Vec::new();
                let mut routed = Vec::new();
                // `parents` maps the id of each elite to its body plan key and
                // whether it is a reserve entry. `bars` maps a body plan key to
                // the best fitness of its behavior elites and of its reserve
                // entry, with negative infinity for none. A reserve place needs
                // a score above both bars of the plan. A plan with no reserve
                // entry also needs a score above the reserve's floor once the
                // reserve is full. `QdArchive::offer_morphology` makes the final
                // check.
                let size = archive.entries.len();
                let mut parents: KeyMap<(u64, bool)> =
                    KeyMap::with_capacity_and_hasher(size, Default::default());
                let mut bars: KeyMap<(f32, f32)> =
                    KeyMap::with_capacity_and_hasher(size, Default::default());
                for (slot, elite) in archive.entries.iter().enumerate() {
                    let morphology = qd::is_morphology_niche(&elite.niche);
                    parents.insert(elite.creature.id, (archive.plan_key(slot), morphology));
                    let bar = bars
                        .entry(archive.plan_key(slot))
                        .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                    if morphology {
                        bar.1 = bar.1.max(elite.fitness);
                    } else {
                        bar.0 = bar.0.max(elite.fitness);
                    }
                }
                for &j in members {
                    let p = &prep[j];
                    if !p.score.is_finite() || p.score <= FAILED || p.screened {
                        continue;
                    }
                    let behavior = archive.offer(
                        population,
                        j,
                        p.descriptor,
                        p.score,
                        p.fine,
                        p.emitter,
                        generation,
                        p.protection,
                    );
                    if behavior.inserted {
                        entered.push(j);
                        if p.structural {
                            let bar = bars
                                .entry(p.plan)
                                .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                            bar.0 = bar.0.max(p.score);
                        }
                        continue;
                    }
                    if !p.structural {
                        continue;
                    }
                    // A reserve place goes to a child whose body plan differs
                    // from its parent's, or to a child of a reserve entry of the
                    // same plan. The parent has to be an elite of this archive
                    // from before the block.
                    let parent = births[j].parent_id.and_then(|id| parents.get(&id));
                    let changed = parent.is_some_and(|&(plan, _)| plan != p.plan);
                    let from_reserve =
                        parent.is_some_and(|&(plan, morphology)| morphology && plan == p.plan);
                    if !(changed || from_reserve) {
                        continue;
                    }
                    let floor = archive.morphology_floor();
                    let admits = match bars.get(&p.plan) {
                        Some(&(behavior, reserve)) if reserve > f32::NEG_INFINITY => {
                            p.score > behavior && p.score > reserve
                        }
                        Some(&(behavior, _)) => {
                            p.score > behavior && floor.is_none_or(|floor| p.score > floor)
                        }
                        None => floor.is_none_or(|floor| p.score > floor),
                    };
                    // A new body plan that an island turns away goes to the
                    // island's nursery of reshaped bodies. Here `island` is the
                    // archive's index in `Experiment::islands`, which lists the
                    // nurseries after the islands. A nursery turns nothing away.
                    let routes = changed && island < island_count();
                    if !admits {
                        if routes {
                            routed.push(j);
                        }
                        continue;
                    }
                    let offer = archive.offer_morphology(
                        population,
                        j,
                        p.descriptor,
                        qd::topology_of_population(population, j),
                        p.score,
                        p.fine,
                        p.emitter,
                        generation,
                        p.protection,
                    );
                    if offer.inserted {
                        entered.push(j);
                        reserve_entered.push(j);
                        let bar = bars
                            .entry(p.plan)
                            .or_insert((f32::NEG_INFINITY, f32::NEG_INFINITY));
                        bar.1 = bar.1.max(p.score);
                        reserve_offers.push((p.emitter.index(), offer));
                    } else if routes {
                        routed.push(j);
                    }
                }
                (entered, reserve_offers, reserve_entered, routed)
            })
            .collect();
        // The turned away body plans compete for cells of their island's
        // nursery of reshaped bodies, where only such bodies compete.
        let routed: Vec<Vec<usize>> = island_results
            .iter()
            .take(island_count())
            .map(|result| result.3.clone())
            .collect();
        let routed_entered: Vec<Vec<usize>> = self.islands[reshaped_of(0)..]
            .par_iter_mut()
            .zip(routed)
            .map(|(archive, routed)| {
                routed
                    .into_iter()
                    .filter(|&j| {
                        let p = &prep[j];
                        archive
                            .offer(
                                population,
                                j,
                                p.descriptor,
                                p.score,
                                p.fine,
                                p.emitter,
                                generation,
                                p.protection,
                            )
                            .inserted
                    })
                    .collect()
            })
            .collect();
        // Per archive, whether it took an entry from this block.
        let mut island_changed: Vec<bool> = island_results
            .iter()
            .map(|(group, _, _, _)| !group.is_empty())
            .collect();
        for (island, entered) in routed_entered.iter().enumerate() {
            island_changed[reshaped_of(island)] |= !entered.is_empty();
        }
        // A wild migrant that entered the hub's archive counts as a win for
        // its wild island.
        if !self.wild_exports.is_empty()
            && let Some((group, _, _, _)) = island_results.get(hub_island())
        {
            for &j in group {
                let id = population.genomes[j].id;
                if let Some(from) = self.wild_exports.remove(&id) {
                    if self.wild_wins.len() < island_count() {
                        self.wild_wins.resize(island_count(), 0);
                    }
                    self.wild_wins[from] += 1;
                }
            }
        }
        // The first elite of a body plan that has not entered a main island
        // before joins the founder bank. The oldest founder leaves when the
        // bank is full.
        {
            for (group, _, _, _) in island_results.iter().take(qd::MAIN_ISLANDS) {
                for &j in group {
                    let plan = qd::plan_key_of_population(population, j);
                    if self.founder_plans.insert(plan) {
                        if self.founders.len() >= FOUNDERS {
                            self.founders.pop_front();
                        }
                        self.founders.push_back(population.creature(j));
                    }
                }
            }
        }
        let mut reserve_offers = Vec::new();
        for (arena, (group, offers, reserves, _)) in island_results.into_iter().enumerate() {
            if let Some(kinds) = kinds.as_deref_mut() {
                let kind = if arena < island_count() {
                    dump::ISLAND
                } else {
                    dump::NURSERY
                };
                for &j in &group {
                    kinds[j] |= kind;
                }
                // A reserve entry carries the reserve mark in place of the
                // island or nursery mark.
                for &j in &reserves {
                    kinds[j] = (kinds[j] & !kind) | dump::RESERVE;
                }
            }
            entered.extend(group);
            // The reserve entries of a nursery count for no emitter. The
            // emitter statistics describe the islands' search.
            if arena < island_count() {
                reserve_offers.extend(offers);
            }
        }
        for group in routed_entered {
            if let Some(kinds) = kinds.as_deref_mut() {
                for &j in &group {
                    kinds[j] |= dump::NURSERY;
                }
            }
            entered.extend(group);
        }
        timings[0] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // The behavior scores depend only on an archive's elites. An archive
        // that took no entry keeps its scores, unless they do not cover its
        // elites. A nursery of reshaped bodies takes entries in every block,
        // so it refreshes once a generation instead (`end_generation`). Most
        // archives are small, so they refresh side by side rather than one
        // after another.
        let refreshed = reshaped_of(0).min(self.islands.len());
        self.islands[..refreshed]
            .par_iter_mut()
            .zip(island_changed.par_iter())
            .for_each(|(island, &changed)| {
                if changed || !island.scores_current() {
                    island.refresh_behavior_scores();
                }
            });
        timings[1] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // The global archive is offered only the best candidate of each of its
        // cells, and the first one when scores tie.
        let mut prep = prep;
        let mut best_by_niche: FastMap<qd::Niche, usize> = FastMap::default();
        for (j, p) in prep.iter().enumerate() {
            if p.behavior_candidate {
                let best = best_by_niche
                    .entry(self.archive.cell_of(p.descriptor))
                    .or_insert(j);
                if prep[*best].score < p.score {
                    *best = j;
                }
            }
        }
        let mut behavior_best = vec![false; prep.len()];
        for &j in best_by_niche.values() {
            behavior_best[j] = true;
        }
        for (p, &best) in prep.iter_mut().zip(&behavior_best) {
            p.behavior_candidate &= best;
        }
        timings[2] += section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // In block order: count the failed trials, offer the candidates to the
        // global archive, count each island result as an attempt of its
        // emitter, and rank the CMA samples.
        let mut attempts = [0u64; qd::EMITTER_COUNT];
        let mut failed = 0usize;
        let mut global_changed = false;
        let mut behavior_inserted = false;
        for (j, prep) in prep.into_iter().enumerate() {
            if !prep.score.is_finite() || prep.score <= FAILED {
                failed += 1;
            }
            let cma = births[j].cma;
            if arena_of[j] as usize >= island_count() {
                // A nursery result is no attempt of its emitter. Its CMA
                // samples rank by distance alone.
                if prep.emitter == Emitter::Cma
                    && let Some(cma) = cma
                    && let Some(samples) = cma_samples.get_mut(cma)
                    && prep.score.is_finite()
                    && prep.score > FAILED
                {
                    samples.push((j, prep.score));
                }
                continue;
            }
            let emitter_index = prep.emitter.index();
            attempts[emitter_index] += 1;
            // The CMA improvement key needs the fitness of the cell's elite
            // before this creature's offer, and only CMA samples use it. Until
            // the first insertion of the block, the prefilter's read of it
            // still holds. After that, the cell is read again, so the key sees
            // the offers made before this one.
            let elite_before = if !behavior_inserted {
                prep.elite_before
            } else {
                (prep.emitter == Emitter::Cma)
                    .then_some(cma)
                    .flatten()
                    .filter(|_| prep.score.is_finite() && prep.score > FAILED)
                    .and_then(|_| {
                        self.archive
                            .slot_for(&self.archive.cell_of(prep.descriptor))
                    })
                    .map(|slot| self.archive.entries[slot].fitness)
            };
            let offer = if prep.behavior_candidate {
                self.archive.offer(
                    population,
                    j,
                    prep.descriptor,
                    prep.score,
                    prep.fine,
                    prep.emitter,
                    self.generation,
                    prep.protection,
                )
            } else {
                qd::Offer::default()
            };
            behavior_inserted |= offer.inserted;
            // A CMA-ME emitter ranks its samples by improvement. First come the
            // creatures that took an empty cell. Then come those that beat their
            // cell's elite, by how much. Next come the ones that fell short of
            // the elite, by how far. Last come the ones that found the cell
            // empty and did not enter. An optimizer ranks by distance alone.
            if prep.emitter == Emitter::Cma
                && let Some(cma) = cma
                && let Some(samples) = cma_samples.get_mut(cma)
                && prep.score.is_finite()
                && prep.score > FAILED
            {
                let key = match elite_before {
                    _ if optimizers[cma] => prep.score,
                    None if offer.inserted => 1.0e6 + prep.score,
                    Some(before) if offer.inserted => 1.0e3 + (prep.score - before),
                    Some(before) => prep.score - before,
                    None => prep.score - 1.0e3,
                };
                samples.push((j, key));
            }
            if offer.inserted {
                global_changed = true;
                entered.push(j);
                if let Some(kinds) = kinds.as_deref_mut() {
                    kinds[j] |= dump::GLOBAL;
                }
                rewards[emitter_index] += offer.reward;
                if offer.new_niche {
                    discoveries[emitter_index] += 1;
                } else {
                    improvements[emitter_index] += 1;
                }
            }
        }
        // The reserve entries of the islands count for their emitters like the
        // global archive's entries.
        for (emitter_index, offer) in reserve_offers {
            rewards[emitter_index] += offer.reward;
            if offer.new_niche {
                discoveries[emitter_index] += 1;
            } else {
                improvements[emitter_index] += 1;
            }
        }
        timings[3] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // The CMA emitters learn from their samples, and the emitter statistics
        // take the block's counts. Then each emitter's last parent is found
        // again by its id, because the offers may have moved it.
        for (emitter, samples) in self.cma_emitters.iter_mut().zip(&mut cma_samples) {
            emitter.tell(population, samples);
        }
        qd::record_emitter_batch(
            &mut self.emitter_stats,
            &attempts,
            &discoveries,
            &improvements,
            &rewards,
        );
        let parent_index_by_id: HashMap<_, _> = self
            .archive
            .entries
            .iter()
            .enumerate()
            .map(|(index, elite)| (elite.creature.id, index))
            .collect();
        for (stats, parent_id) in self.emitter_stats.iter_mut().zip(previous_parent_ids) {
            stats.last_parent = parent_id.and_then(|id| parent_index_by_id.get(&id).copied());
        }
        timings[4] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        if global_changed || !self.archive.scores_current() {
            self.archive.refresh_behavior_scores();
        }
        timings[5] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // A creature that entered several archives is listed once. It gets a
        // lineage record unless it has one already.
        entered.sort_unstable();
        entered.dedup();
        let entered_count = entered.len();
        let records: Vec<(u64, Ancestor)> = entered
            .par_iter()
            .filter_map(|&j| self.ancestor_of(population, births[j], j, &finals[j]))
            .collect();
        self.lineage.extend(records);
        timings[6] = section.elapsed().as_secs_f64();
        // The figure after "block" is the number of creatures in the block.
        if profile {
            eprintln!(
                "Archive profile: generation {}, block {}, island offers {:.6} s, island refresh {:.6} s, prefilter {:.6} s, global offers {:.6} s, cma tell {:.6} s, archive refresh {:.6} s, lineage {:.6} s, entered {}",
                self.generation,
                block.len(),
                timings[0],
                timings[1],
                timings[2],
                timings[3],
                timings[4],
                timings[5],
                timings[6],
                entered_count
            );
        }
        failed
    }
    /// Adds the `history` row of the generation that ends. The row sums up the
    /// global archive: the distances of its elites, how many cells it fills,
    /// its body plans and clades, and three representative creatures. It also
    /// holds the emitter statistics and `failed`, the generation's count of
    /// failed trials. It does nothing when `history` has the row already.
    pub(super) fn push_archive_stats(&mut self, failed: usize) {
        if self.history.len() > self.generation as usize {
            return;
        }
        // The elites that fill a cell, without any entry of the morphology
        // reserve.
        let elites: Vec<_> = self
            .archive
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .collect();
        // The median, the worst, the mean and the percentiles read the best
        // elite of each way of moving, so they mean what they meant before the
        // archive had body classes. The histogram and the body types, which
        // are the node and muscle counts, count every elite in a cell.
        let mut ways: Vec<_> = self
            .archive
            .best_per_way_of_moving()
            .into_iter()
            .map(|slot| &self.archive.entries[slot])
            .collect();
        ways.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let count = ways.len();
        let archive_best = self.archive.best_fitness();
        // The fitness at percentile `p`, from the worst (0) to the best (100)
        // of `ways`, or 0 when there are none.
        let quantile = |p: f32| {
            if count == 0 {
                0.0
            } else {
                ways[((1.0 - p / 100.0) * (count - 1) as f32).round() as usize].fitness
            }
        };
        // The histogram bins the elites by distance in centimeters. `species`
        // counts them by node count and muscle count.
        let mut histogram = BTreeMap::<i32, u32>::new();
        let mut species = BTreeMap::<(usize, usize), u32>::new();
        let mut sum = 0.0f64;
        for elite in &ways {
            sum += elite.fitness as f64;
        }
        for elite in &elites {
            *histogram
                .entry((elite.fitness * 100.0).floor() as i32)
                .or_default() += 1;
            *species
                .entry((elite.creature.node_count(), elite.creature.muscle_count()))
                .or_default() += 1;
        }
        let mut percentiles: Vec<_> = PERCENTILES.iter().map(|&p| quantile(p)).collect();
        // The last percentile is the best fitness of the whole archive.
        if let Some(best_percentile) = percentiles.last_mut() {
            *best_percentile = archive_best.max(0.0);
        }
        // The representatives are the slowest, the median and the fastest
        // entry of the archive, in that order. While the archive is empty they
        // are the ring's first creature three times.
        let mut all_elites: Vec<_> = self.archive.entries.iter().collect();
        all_elites.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let representatives = if all_elites.is_empty() {
            vec![self.blocks[0].population.creature(0); 3]
        } else {
            [all_elites.len() - 1, (all_elites.len() - 1) / 2, 0]
                .map(|i| all_elites[i].creature.unpack())
                .to_vec()
        };
        self.history.push(Stats {
            generation: self.generation,
            best: archive_best.max(0.0),
            median: quantile(50.0),
            worst: quantile(0.0),
            mean: if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            },
            failed,
            seconds: self.evaluation_seconds,
            population: self.config.population,
            percentiles,
            histogram: histogram.into_iter().collect(),
            species: species.into_iter().map(|((n, m), c)| (n, m, c)).collect(),
            representatives,
            config: self.config.clone(),
            archive_cells: elites.len(),
            qd_score: self.archive.qd_score,
            archive_coverage: self.archive.coverage(),
            emitters: self.emitter_stats,
            ring: self.ring,
            plans: elites
                .iter()
                .map(|e| e.topology.plan_key())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            // `plan_born` holds the generation each body plan first appeared
            // in. It drops the plans that left the archive, so a plan that
            // returns starts young. `plan_age` is the median age of the plans
            // in the archive.
            plan_age: {
                let present: std::collections::HashSet<u64> =
                    elites.iter().map(|e| e.topology.plan_key()).collect();
                let generation = self.generation;
                for &plan in &present {
                    self.plan_born.entry(plan).or_insert(generation);
                }
                self.plan_born.retain(|plan, _| present.contains(plan));
                let mut ages: Vec<u32> = self.plan_born.values().map(|&b| generation - b).collect();
                if ages.is_empty() {
                    0.0
                } else {
                    let mid = ages.len() / 2;
                    *ages.select_nth_unstable(mid).1 as f32
                }
            },
            clades: self.effective_clades(elites.iter().map(|e| e.creature.id)),
        });
    }
}
