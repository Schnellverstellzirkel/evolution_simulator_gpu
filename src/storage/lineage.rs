use super::*;

/// How far back the lineage tab reads an elite's ancestors.
pub const ANCESTRY_DEPTH: usize = 400;
/// How many of each island's fastest elites keep their ancestors in a save,
/// besides every elite of the global archive.
pub(super) const ISLAND_LEADERS: usize = 10;

/// One recorded creature in an elite's ancestry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ancestor {
    pub parent: Option<u64>,
    pub creature: StoredCreature,
    pub fitness: f32,
    pub generation: u32,
    /// What changed from the parent, for display.
    pub change: String,
    /// Its own features at the early rungs (`rungs::profile`), which decide
    /// whether its children skip them.
    pub rung: [u16; 2 * crate::rungs::FEATURES],
}

/// Short description of how a child differs from its parent.
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
    /// entered an archive with `result`; none when it has one already. A
    /// parent entered an archive in an earlier block, so its record is there.
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
        let period = if genome.muscle_count > 0 {
            population.muscles[genome.muscle_start].period
        } else {
            0.0
        };
        let parent_body = birth
            .parent_id
            .and_then(|id| self.lineage.get(&id))
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
    /// Bounds the lineage. A record stays while a living elite is at most
    /// `ANCESTRY_DEPTH` steps from it, which is as far as the clade count and
    /// the lineage tab read. Its creature stays only for the elites the tab
    /// shows and their ancestors (the global archive and each
    /// island's fastest `ISLAND_LEADERS`); the other records keep their links
    /// and numbers and lose their genes, as in a save.
    pub(super) fn prune_lineage(&mut self) {
        use std::collections::HashSet;
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
        // The ids within `ANCESTRY_DEPTH` steps of `start`, level by level.
        let reach = |start: &[u64], within: &mut HashSet<u64>| {
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
        let mut keep: HashSet<u64> = HashSet::new();
        reach(&living, &mut keep);
        let mut with_genes: HashSet<u64> = HashSet::new();
        reach(&shown, &mut with_genes);
        self.lineage.retain(|id, _| keep.contains(id));
        for (id, record) in self.lineage.iter_mut() {
            if !with_genes.contains(id) && !record.creature.is_empty() {
                record.creature = StoredCreature::default();
            }
        }
    }
    /// Ancestor chain of a creature, newest first (at most `limit` steps).
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
    /// The effective number of clades among `ids`: a clade is the elites
    /// that share their oldest recorded ancestor.
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
    /// For each entry of a refined `archive`, how rare its clade is: 1 minus
    /// the log of the number of behavior elites in the clade over the log of
    /// all of them, where a clade is the elites that share the oldest recorded
    /// ancestor.
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
        // While distance still separates the elites a bonus for rare clades
        // would take parents from the climb. It grows with the share of elites
        // within 10% of the best, and has its full weight when half of them
        // are: where the distances are level, rarity decides.
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
