//! The lineage of the search, kept in `Experiment::lineage`. Each creature
//! that enters an archive gets an `Ancestor` record with its parent and the
//! change that made it. Records are added as blocks are absorbed and pruned
//! once a generation to the ancestors of the living elites. The lineage tab,
//! the clade counts and the clade rarity in parent selection read them.

use super::*;

/// How many steps back from an elite its ancestors are read. The lineage tab
/// shows a chain this long, the clade counts follow parents this far, and
/// `prune_lineage` and a save keep records no farther back.
pub const ANCESTRY_DEPTH: usize = 400;
/// How many of each island's fastest elites keep the genes of their
/// ancestors, besides every elite of the global archive. `prune_lineage` keeps
/// those genes in memory and a save writes them.
pub(super) const ISLAND_LEADERS: usize = 10;

/// One recorded creature in an elite's ancestry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ancestor {
    /// The id of the elite it was bred from, or none when it was not bred
    /// from one of its archive's elites. The record of the parent may have
    /// been pruned.
    pub parent: Option<u64>,
    /// Its genes. `prune_lineage` replaces them with the empty creature in a
    /// record outside the chains the lineage tab shows.
    pub creature: StoredCreature,
    /// Its distance when it entered an archive.
    pub fitness: f32,
    /// The generation it entered an archive in.
    pub generation: u32,
    /// How it was made and what changed from its parent, in words for the
    /// lineage tab. `describe_change` writes it.
    pub change: String,
    /// Its own features at the early rungs (`rungs::profile`), which decide
    /// whether its children skip them.
    pub rung: [u16; 2 * crate::rungs::FEATURES],
}

/// The words for `Ancestor::change`. They name the `emitter` that bred
/// `child` and say if it was `crossed`. When the parent is known they add the
/// nodes, muscles and organs the child gained or lost, and whether its rhythm
/// became synced. For example "reshaped, +2 nodes, +1 muscle".
fn describe_change(
    parent: Option<&Creature>,
    child: &Creature,
    emitter: Emitter,
    crossed: bool,
) -> String {
    let mut parts: Vec<String> = vec![match emitter {
        Emitter::Cma => "fine-tuned".into(),
        Emitter::Structural => "reshaped".into(),
        Emitter::Novelty => "explored".into(),
        Emitter::Restart => "new random body".into(),
    }];
    if crossed {
        parts.push("crossed with a relative".into());
    }
    if let Some(parent) = parent {
        let nodes = child.nodes.len() as i64 - parent.nodes.len() as i64;
        let muscles = child.muscles.len() as i64 - parent.muscles.len() as i64;
        if nodes != 0 {
            parts.push(format!(
                "{nodes:+} node{}",
                if nodes.abs() == 1 { "" } else { "s" }
            ));
        }
        if muscles != 0 {
            parts.push(format!(
                "{muscles:+} muscle{}",
                if muscles.abs() == 1 { "" } else { "s" }
            ));
        }
        let organs = |c: &Creature| c.bones.iter().filter(|b| b.organ_mass > 0.0).count() as i64;
        let organ_change = organs(child) - organs(parent);
        if organ_change != 0 {
            parts.push(format!(
                "{organ_change:+} organ{}",
                if organ_change.abs() == 1 { "" } else { "s" }
            ));
        }
        let synced = |c: &Creature| {
            c.muscles.len() > 1 && c.muscles.iter().all(|m| m.period == c.muscles[0].period)
        };
        if synced(child) && !synced(parent) {
            parts.push("synced rhythm".into());
        }
    }
    parts.join(", ")
}

impl Experiment {
    /// The lineage record of creature `index` of `population`, which just
    /// entered an archive with `result` and was bred as `birth`. It returns
    /// the creature's id with the record, or none when the creature already
    /// has a record. An elite queued again after a world change is such a
    /// creature. The change text compares the creature with the genes in its
    /// parent's record. That record is usually there, because the parent
    /// entered an archive in an earlier block. If it is gone, or if
    /// `prune_lineage` stripped its genes, the text names only the emitter and
    /// the crossover.
    pub(super) fn ancestor_of(
        &self,
        population: &Population,
        birth: Birth,
        index: usize,
        result: &EvaluationMetrics,
    ) -> Option<(u64, Ancestor)> {
        let genome = &population.genomes[index];
        if self.lineage.contains_key(&genome.id) {
            return None;
        }
        let creature = population.creature(index);
        // The period of the first muscle, a feature of the early rungs, as
        // `kernel::pack` takes it.
        let period = if genome.muscle_count > 0 {
            population.muscles[genome.muscle_start].period
        } else {
            0.0
        };
        // A record that `prune_lineage` stripped holds the empty creature.
        // Compared with it, every node of the child would count as new.
        let parent_body = birth
            .parent_id
            .and_then(|id| self.lineage.get(&id))
            .filter(|a| !a.creature.is_empty())
            .map(|a| a.creature.unpack());
        let change = describe_change(parent_body.as_ref(), &creature, birth.emitter, birth.mate);
        Some((
            creature.id,
            Ancestor {
                parent: birth.parent_id,
                fitness: result.fitness,
                generation: self.generation,
                change,
                creature: creature.into(),
                rung: crate::rungs::profile(&result.trace, period),
            },
        ))
    }
    /// Bounds the lineage once a generation. A record stays while a living
    /// elite is at most `ANCESTRY_DEPTH` steps from it, which is as far as the
    /// clade counts and the lineage tab read. A record keeps its creature only
    /// within that distance of an elite the tab shows. Those are the elites of
    /// the global archive and the fastest `ISLAND_LEADERS` of each archive in
    /// `islands`. The other records keep their links and numbers and lose
    /// their genes. The genes of a living elite stay in its archive.
    pub(super) fn prune_lineage(&mut self) {
        let elites = || {
            self.archive
                .entries
                .iter()
                .chain(self.islands.iter().flat_map(|island| island.entries.iter()))
        };
        let living: Vec<u64> = elites().map(|elite| elite.creature.id).collect();
        let mut shown: Vec<u64> = self.archive.entries.iter().map(|x| x.creature.id).collect();
        for island in &self.islands {
            let mut fastest: Vec<&qd::Elite> = island.entries.iter().collect();
            fastest.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            shown.extend(fastest.iter().take(ISLAND_LEADERS).map(|x| x.creature.id));
        }
        // Adds to `within` the ids of `start` and of their ancestors up to
        // `ANCESTRY_DEPTH` steps back, one level of parents at a time.
        let reach = |start: &[u64], within: &mut KeySet| {
            let mut frontier: Vec<u64> = start
                .iter()
                .copied()
                .filter(|id| within.insert(*id))
                .collect();
            for _ in 0..ANCESTRY_DEPTH {
                let mut next = Vec::new();
                for id in &frontier {
                    if let Some(parent) = self.lineage.get(id).and_then(|a| a.parent)
                        && within.insert(parent)
                    {
                        next.push(parent);
                    }
                }
                if next.is_empty() {
                    break;
                }
                frontier = next;
            }
        };
        let mut keep = KeySet::default();
        reach(&living, &mut keep);
        let mut with_genes = KeySet::default();
        reach(&shown, &mut with_genes);
        self.lineage.retain(|id, _| keep.contains(id));
        for (id, record) in self.lineage.iter_mut() {
            if !with_genes.contains(id) && !record.creature.is_empty() {
                record.creature = StoredCreature::default();
            }
        }
    }
    /// The records of creature `id` and of its ancestors, newest first. The
    /// chain stops at a creature with no parent, at the first parent with no
    /// record, or after `limit` records. It is empty when `id` has no record.
    pub fn ancestry(&self, id: u64, limit: usize) -> Vec<&Ancestor> {
        let mut chain = Vec::new();
        let mut current = Some(id);
        while let Some(id) = current {
            let Some(ancestor) = self.lineage.get(&id) else {
                break;
            };
            chain.push(ancestor);
            if chain.len() >= limit {
                break;
            }
            current = ancestor.parent;
        }
        chain
    }
    /// The ids of the chain that `ancestry` walks for creature `id`, newest
    /// first, at most `limit` of them. Use these rather than the records'
    /// `creature.id`. A record outside the chains the lineage tab shows keeps
    /// no genes after `prune_lineage`, and its stored creature has id 0.
    pub fn ancestry_ids(&self, id: u64, limit: usize) -> Vec<u64> {
        let mut ids = Vec::new();
        let mut current = Some(id);
        while let Some(id) = current {
            let Some(ancestor) = self.lineage.get(&id) else {
                break;
            };
            ids.push(id);
            if ids.len() >= limit {
                break;
            }
            current = ancestor.parent;
        }
        ids
    }
    /// The effective number of clades among the elites `ids`, or 0 for no
    /// ids. A clade is the elites that share their oldest recorded ancestor,
    /// found by following parents while each one has a record, for at most
    /// `ANCESTRY_DEPTH` steps. The number is the exponential of the Shannon
    /// entropy of the clade sizes, the Hill number of order 1.
    pub(super) fn effective_clades(&self, ids: impl Iterator<Item = u64>) -> f32 {
        let mut sizes: HashMap<u64, usize> = HashMap::new();
        let mut n = 0usize;
        for id in ids {
            let mut root = id;
            for _ in 0..ANCESTRY_DEPTH {
                match self.lineage.get(&root).and_then(|a| a.parent) {
                    Some(parent) if self.lineage.contains_key(&parent) => root = parent,
                    _ => break,
                }
            }
            *sizes.entry(root).or_default() += 1;
            n += 1;
        }
        if n == 0 {
            return 0.0;
        }
        let entropy: f64 = sizes
            .values()
            .map(|&c| {
                let p = c as f64 / n as f64;
                -p * p.ln()
            })
            .sum();
        entropy.exp() as f32
    }
    /// How rare the clade of each entry of `archive` is, in the order of its
    /// entries. A clade is the elites that share their oldest recorded
    /// ancestor. An entry scores 1 minus the log of the number of behavior
    /// elites in its clade over the log of the number in the archive. Entries
    /// of the morphology reserve get the score of their clade but add nothing
    /// to its size. The scores are scaled down while few elites are near the
    /// best distance. The list is empty for an archive that is not refined.
    pub(super) fn clade_rarity_of(&self, archive: &QdArchive) -> Vec<f32> {
        // A climbing archive keeps one elite per way of moving and the best
        // lineage fills it: a bonus for rare clades would take parents from
        // the climb. A refined archive keeps the lineages of other bodies in
        // cells of their own, and rarity keeps them breeding.
        if !archive.refined() {
            return Vec::new();
        }
        let mut roots: HashMap<u64, u64> = HashMap::new();
        let mut root_of = Vec::with_capacity(archive.entries.len());
        let mut sizes: HashMap<u64, u32> = HashMap::new();
        for elite in &archive.entries {
            let id = elite.creature.id;
            let mut chain = vec![id];
            let mut current = id;
            let root = loop {
                if let Some(&root) = roots.get(&current) {
                    break root;
                }
                match self.lineage.get(&current).and_then(|a| a.parent) {
                    Some(parent) if self.lineage.contains_key(&parent) => {
                        chain.push(parent);
                        current = parent;
                    }
                    _ => break current,
                }
            };
            for step in chain {
                roots.insert(step, root);
            }
            root_of.push(root);
            if !qd::is_morphology_niche(&elite.niche) {
                *sizes.entry(root).or_default() += 1;
            }
        }
        let total = (archive.behavior_count() as f32).ln().max(1.0);
        // While distance still separates the elites, a bonus for rare clades
        // would take parents from the climb. So the scores are scaled. The
        // scale grows with the share of elites within 10% of the best and is
        // 1 when half of them are. Where the distances are level, rarity
        // decides.
        let best = archive
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .map(|elite| elite.fitness)
            .fold(f32::NEG_INFINITY, f32::max);
        let level = archive
            .entries
            .iter()
            .filter(|elite| !qd::is_morphology_niche(&elite.niche))
            .filter(|elite| elite.fitness >= 0.9 * best)
            .count();
        let scale = (2.0 * level as f32 / archive.behavior_count().max(1) as f32).min(1.0);
        root_of
            .iter()
            .map(|root| scale * (1.0 - (sizes.get(root).copied().unwrap_or(1) as f32).ln() / total))
            .collect()
    }
}
