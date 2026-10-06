use super::*;

impl Experiment {
    /// Offers block `k`'s creatures to the archives in block order, updates
    /// CMA emitters and emitter statistics, and returns how many trials
    /// failed. Screened and excluded results, and every result of a
    /// `stale` block, enter no archive.
    /// With `kinds` it also marks, per position, what the creature entered
    /// (`dump::ISLAND`, `NURSERY`, `RESERVE`, `GLOBAL`).
    pub(super) fn archive_block(
        &mut self,
        k: usize,
        finals: &[EvaluationMetrics],
        stale: bool,
        mut kinds: Option<&mut [u8]>,
    ) -> usize {
        let profile = std::env::var_os("EVOLUTION_PROFILE_BREED").is_some();
        let mut timings = [0.0f64; 7];
        let mut section = std::time::Instant::now();
        self.ensure_islands();
        let block = self.blocks[k].clone();
        let population = &*block.population;
        let births = &block.births;
        let first = block.first;
        let mut entered: Vec<usize> = Vec::new();
        let previous_parent_ids: [Option<u64>; qd::EMITTER_COUNT] = std::array::from_fn(|i| {
            self.emitter_stats[i]
                .last_parent
                .and_then(|index| self.archive.entries.get(index))
                .map(|elite| elite.creature.id)
        });
        let mut discoveries = [0u64; qd::EMITTER_COUNT];
        let mut improvements = [0u64; qd::EMITTER_COUNT];
        let mut rewards = [0.0f64; qd::EMITTER_COUNT];
        let mut cma_samples = vec![Vec::<(usize, f32)>::new(); self.cma_emitters.len()];
        let optimizers: Vec<bool> = self.cma_emitters.iter().map(|c| c.optimizing()).collect();
        // Parallel prefilter: descriptors and behavior-offer eligibility against
        // the start-of-block global archive. Occupant fitness only ever rises,
        // so a snapshot reject stays a live reject. Inserts still commit
        // sequentially in block order.
        struct Prep {
            descriptor: qd::Descriptor,
            emitter: Emitter,
            score: f32,
            fine: bool,
            protection: u32,
            behavior_candidate: bool,
            /// A structural or novelty child may enter its island's
            /// morphology reserve, and has a body plan key.
            structural: bool,
            plan: u64,
            /// Offered to no archive.
            screened: bool,
            /// Fitness of the global archive's elite in this creature's
            /// cell at the start of the block, for a CMA sample.
            elite_before: Option<f32>,
        }
        let arenas = self.islands.len().max(arena_count());
        let positions: Vec<usize> = (0..block.len()).collect();
        // The archive each creature breeds for and competes in.
        let arena_of: Vec<u8> = positions
            .par_iter()
            .map(|&j| qd::arena_of_slot(first + j, arenas) as u8)
            .collect();
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
                // A nursery creature is offered to its nursery only.
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
        // Every creature also competes in its own island's archive, and a
        // new body plan that does not take a behavior cell may enter the
        // island's morphology reserve. Each island's offers resolve in block
        // order and the islands are independent, so the streams run in
        // parallel.
        let generation = self.generation;
        // Per island: the positions that entered, the emitter and offer of
        // each reserve entry, the positions that entered the reserve, and
        // the new body plans that took neither a cell nor a reserve place.
        type IslandResult = (Vec<usize>, Vec<(usize, qd::Offer)>, Vec<usize>, Vec<usize>);
        let island_results: Vec<IslandResult> = self
            .islands
            .par_iter_mut()
            .enumerate()
            .map(|(island, archive)| {
                let mut entered = Vec::new();
                let mut reserve_offers = Vec::new();
                let mut reserve_entered = Vec::new();
                let mut routed = Vec::new();
                // Reserve admission needs a score above the island's best
                // behavior elite and reserve entry of the same body plan, or
                // above the reserve's floor once it is full
                // (`QdArchive::offer_morphology` makes the final check).
                let mut parents: HashMap<u64, (u64, bool)> = HashMap::new();
                let mut bars: HashMap<u64, (f32, f32)> = HashMap::new();
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
                for (j, p) in prep.iter().enumerate() {
                    if arena_of[j] as usize != island {
                        continue;
                    }
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
                    // A reserve place goes to a new body plan, or to a better
                    // child of a reserve entry with the same plan.
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
                    // island's nursery of reshaped bodies.
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
        let mut island_changed: Vec<bool> = island_results
            .iter()
            .map(|(group, _, _, _)| !group.is_empty())
            .collect();
        for (island, entered) in routed_entered.iter().enumerate() {
            island_changed[reshaped_of(island)] |= !entered.is_empty();
        }
        // A wild migrant that took a hub cell counts for its wild island.
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
        // The first elite of a new body plan in a main island joins the
        // founder bank.
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
                for &j in &reserves {
                    kinds[j] = (kinds[j] & !kind) | dump::RESERVE;
                }
            }
            entered.extend(group);
            // Nursery entries count for no emitter: the emitter statistics
            // describe the islands' search.
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
        // The scores depend only on the archive's elites, so an island that
        // took no offer keeps the ones it has. A reshaped nursery takes
        // offers in every block, so it refreshes once a generation instead
        // (`end_generation`).
        for (island, changed) in self
            .islands
            .iter_mut()
            .zip(island_changed)
            .take(reshaped_of(0))
        {
            if changed || !island.scores_current() {
                island.refresh_behavior_scores();
            }
        }
        timings[1] = section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
        // The selected evaluation engine owns the score and behavior. CPU
        // playback and cross-engine comparisons are diagnostics only; they do
        // not edit archive fitness or descriptors.
        let mut prep = prep;
        let mut best_by_niche: HashMap<qd::Niche, usize> = HashMap::new();
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
        let behavior_best: std::collections::HashSet<usize> =
            best_by_niche.values().copied().collect();
        for (j, p) in prep.iter_mut().enumerate() {
            p.behavior_candidate &= behavior_best.contains(&j);
        }
        timings[2] += section.elapsed().as_secs_f64();
        section = std::time::Instant::now();
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
                // Nursery samples rank by distance alone.
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
            // The CMA improvement key needs the cell's fitness before the
            // offers; only CMA samples use it. The prefilter read the cell's
            // elite before any offer of this block; after the first
            // insertion a new read keeps the order.
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
            // CMA-ME improvement ranking: new niches first, then improvement over
            // the niche's elite, then how far short of it a sample fell.
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
        // Island reserve entries count for their emitters like archive entries.
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
        entered.sort_unstable();
        entered.dedup();
        let entered_count = entered.len();
        let records: Vec<(u64, Ancestor)> = entered
            .par_iter()
            .filter_map(|&j| self.ancestor_of(population, births[j], j, &finals[j]))
            .collect();
        self.lineage.extend(records);
        timings[6] = section.elapsed().as_secs_f64();
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
    pub(super) fn push_archive_stats(&mut self, failed: usize) {
        if self.history.len() > self.generation as usize {
            return;
        }
        let elites: Vec<_> = self
            .archive
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .collect();
        // The median, the worst, the mean and the percentiles read the best
        // elite of each way of moving, so they mean what they meant before the
        // archive had body classes. The histogram and the body types count
        // every elite.
        let mut ways: Vec<_> = self
            .archive
            .best_per_way_of_moving()
            .into_iter()
            .map(|slot| &self.archive.entries[slot])
            .collect();
        ways.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let count = ways.len();
        let archive_best = self.archive.best_fitness();
        let quantile = |p: f32| {
            if count == 0 {
                0.0
            } else {
                ways[((1.0 - p / 100.0) * (count - 1) as f32).round() as usize].fitness
            }
        };
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
        if let Some(best_percentile) = percentiles.last_mut() {
            *best_percentile = archive_best.max(0.0);
        }
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
