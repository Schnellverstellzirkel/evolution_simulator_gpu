use super::*;

/// Elites waiting to be evaluated again after a world change, one queue per
/// island. Each returns in a slot of its own island, so a world change mixes
/// no island's creatures into another.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reseed {
    queues: Vec<Vec<Creature>>,
}

impl Reseed {
    pub fn len(&self) -> usize {
        self.queues.iter().map(Vec::len).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(Vec::is_empty)
    }
    pub fn clear(&mut self) {
        self.queues.clear();
    }
    /// Queues `creature` for a slot of `island`.
    pub fn push(&mut self, island: usize, creature: Creature) {
        if self.queues.len() <= island {
            self.queues.resize_with(island + 1, Vec::new);
        }
        self.queues[island].push(creature);
    }
    /// The next creature queued for `island`.
    pub fn pop(&mut self, island: usize) -> Option<Creature> {
        self.queues.get_mut(island)?.pop()
    }
    pub fn iter(&self) -> impl Iterator<Item = &Creature> {
        self.queues.iter().flatten()
    }
    /// Whether every queue belongs to an existing island.
    pub(super) fn fits(&self, islands: usize) -> bool {
        self.queues.len() <= islands
    }
}

/// The champions of each island from before an environment change. A world
/// change can kill every old design at once when the first re-test finds
/// them slow, and random bodies take the islands. For `REFUGE_GENERATIONS`
/// generations a share of each island's slots breed children of its old
/// champions, so an old design gets time to retune its gait to the new world
/// (a refugium, as in island models with migration from a reservoir).
#[derive(Clone, Debug, Default)]
pub struct Refuge {
    champions: Vec<Vec<Creature>>,
    /// The generation the refuge opened, and until which it lasts.
    since: u32,
    pub(super) until: u32,
    /// Per island: until when its refuge lasts (it ends early when the
    /// island recovered, and lasts longer after a heavy loss), and the
    /// island's best distance before the change.
    island_until: Vec<u32>,
    before: Vec<f32>,
}
/// Generations the refuge breeds only gait retunes, before structural
/// changes join (Cheney et al., 2018: a changed body readapts its control
/// first).
const REFUGE_TUNING: u32 = 3;
/// The refuge of an island that kept this share of its best distance ends
/// after `REFUGE_MIN` generations; one that kept under `REFUGE_HEAVY` lasts
/// `REFUGE_LONG` (Branke, 1999: the memory a changed world needs grows with
/// the loss).
const REFUGE_MIN: u32 = 2;
const REFUGE_RECOVERED: f32 = 0.9;
const REFUGE_HEAVY: f32 = 0.5;
const REFUGE_LONG: u32 = 10;

/// The champions of an archive for the refuge: the fastest elite of each
/// body plan, fastest first, up to `REFUGE_CHAMPIONS` plans (Schluter, 2000:
/// a radiation grows from many founders, not from many copies of one).
fn plan_champions(archive: &QdArchive) -> Vec<Creature> {
    let mut order: Vec<usize> = (0..archive.entries.len())
        .filter(|&i| !qd::is_morphology_niche(&archive.entries[i].niche))
        .collect();
    order.sort_by(|&a, &b| {
        archive.entries[b]
            .fitness
            .total_cmp(&archive.entries[a].fitness)
            .then(a.cmp(&b))
    });
    let mut seen = std::collections::HashSet::new();
    order
        .into_iter()
        .filter(|&i| seen.insert(archive.plan_key(i)))
        .take(REFUGE_CHAMPIONS)
        .map(|i| archive.entries[i].creature.unpack())
        .collect()
}
/// Generations the old champions keep breeding after a world change.
const REFUGE_GENERATIONS: u32 = 5;
/// The best elites of each island that go into the refuge.
const REFUGE_CHAMPIONS: usize = 64;
/// Share of an island's own slots that breed from its refuge.
const REFUGE_SHARE: f32 = 0.15;

impl Refuge {
    /// Opens a refuge at `generation` with these champions, and the islands'
    /// best distances before the change.
    fn open(champions: Vec<Vec<Creature>>, before: Vec<f32>, generation: u32) -> Self {
        let until = generation + REFUGE_GENERATIONS;
        Self {
            island_until: vec![until; champions.len()],
            champions,
            since: generation,
            until: generation + REFUGE_LONG,
            before,
        }
    }
    /// At a generation boundary: an island that has won back most of its
    /// best distance closes its refuge, and one still far below keeps it up
    /// to `REFUGE_LONG` generations.
    pub(super) fn review(&mut self, islands: &[QdArchive], generation: u32) {
        if generation >= self.until {
            return;
        }
        for (island, until) in self.island_until.iter_mut().enumerate() {
            let before = self.before.get(island).copied().unwrap_or(f32::NAN);
            let Some(archive) = islands.get(island) else {
                continue;
            };
            if !before.is_finite() || before <= 0.0 {
                continue;
            }
            let kept = archive.best_fitness() / before;
            if generation >= self.since + REFUGE_MIN && kept >= REFUGE_RECOVERED {
                *until = (*until).min(generation);
            } else if kept < REFUGE_HEAVY {
                *until = (*until).max(self.since + REFUGE_LONG);
            }
        }
    }
    /// A child of one of `island`'s champions for `slot`, while the refuge
    /// lasts and the draw picks this slot.
    pub(super) fn child(
        &self,
        island: usize,
        slot: usize,
        cfg: &Config,
        generation: u32,
        round: u64,
    ) -> Option<Creature> {
        if generation >= self.until
            || self
                .island_until
                .get(island)
                .is_some_and(|&u| generation >= u)
        {
            return None;
        }
        let champions = self.champions.get(island).filter(|c| !c.is_empty())?;
        let mut rng = evolution::Rng::stream(cfg.seed ^ 0x7265_6675_6765, generation, round, slot);
        if rng.unit() >= REFUGE_SHARE {
            return None;
        }
        let parent = champions[rng.index(champions.len())].clone();
        // The first generations only retune the gait. Then half the children
        // also take a structural mutation.
        let scale = if rng.unit() < 0.1 { 2.0 } else { 0.75 };
        let mut child = evolution::mutate_locally(parent, cfg, &mut rng, scale);
        if generation >= self.since + REFUGE_TUNING && rng.unit() < 0.5 {
            evolution::structural_mutation_any(&mut child, cfg, &mut rng);
        }
        child.id = evolution::bred_id(round, slot);
        Some(child)
    }
}

impl Experiment {
    /// Applies settings at the next generation boundary.
    pub fn update_config(&mut self, cfg: Config) -> Result<()> {
        self.update_config_at(cfg, false)
    }
    /// Applies settings now. A world change resets the search context at
    /// once. Blocks in flight from the old world are recognized by their own
    /// settings and enter no archive.
    pub fn update_config_now(&mut self, cfg: Config) -> Result<()> {
        self.update_config_at(cfg, true)
    }
    fn update_config_at(&mut self, mut cfg: Config, now: bool) -> Result<()> {
        cfg.validate()?;
        // The autochange step advances in the worker, so a settings update must
        // never rewind a checkpoint-carrying counter to its stale copy.
        cfg.autochange_step = cfg.autochange_step.max(self.config.autochange_step);
        ensure!(
            cfg.population == self.config.population
                && cfg.seed == self.config.seed
                && cfg.random_seed == self.config.random_seed,
            "Population or seed changes require a new experiment"
        );
        ensure!(
            self.bodies_fit(&cfg),
            "Existing bodies exceed these limits; start a new experiment"
        );
        if now {
            self.pending = None;
            let world_changed = fitness_context_changed(&self.config, &cfg);
            if world_changed {
                self.reset_search_context();
            }
            cfg.screen = self.next_screen(cfg.duration);
            cfg.rungs = if world_changed {
                None
            } else {
                self.config.rungs
            };
            self.config = cfg;
        } else {
            self.pending = Some(cfg);
        }
        Ok(())
    }
    /// A meteor strike wipes out `share` of the elites in the global archive
    /// and in every island, chosen at random. Survivors and new offspring
    /// refill the emptied cells, which opens room for new kinds of movement.
    /// The lost elites become fossils so the strike can be undone. Returns how
    /// many elites were lost.
    pub fn meteor(&mut self, share: f32) -> usize {
        let mut rng = evolution::Rng::new(
            self.config.seed ^ 0x6d65_7465_6f72,
            self.generation,
            self.fossils.len(),
        );
        // The strike spares the fastest elite of each body plan, so the
        // survivors are the rare plans and the common ones thin out (Raup,
        // 1986, selective extinction).
        let mut strike = |archive: &mut QdArchive, island: Option<usize>| {
            let spared: std::collections::HashSet<u64> =
                plan_champions(archive).iter().map(|c| c.id).collect();
            let (kept, lost): (Vec<_>, Vec<_>) = std::mem::take(&mut archive.entries)
                .into_iter()
                .partition(|e| spared.contains(&e.creature.id) || rng.unit() >= share);
            archive.entries = kept;
            archive.rebuild_indices();
            lost.into_iter().map(move |elite| (island, elite))
        };
        let mut fossils: Vec<_> = strike(&mut self.archive, None).collect();
        for (index, island) in self.islands.iter_mut().enumerate() {
            fossils.extend(strike(island, Some(index)));
        }
        let lost = fossils.len();
        self.fossils.extend(fossils);
        self.radiate();
        lost
    }
    /// After a meteor or an extinction the survivors of each main island
    /// breed from the refuge for a few generations, as a radiation into the
    /// emptied cells (Lehman and Miikkulainen, 2015).
    fn radiate(&mut self) {
        let champions: Vec<Vec<Creature>> = self
            .islands
            .iter()
            .take(qd::MAIN_ISLANDS)
            .map(plan_champions)
            .collect();
        let before = vec![f32::NAN; champions.len()];
        self.refuge = Refuge::open(champions, before, self.generation);
    }
    /// An extinction wipes out the island whose best creature is slowest. An
    /// isolated island starts over from new random bodies, and the hub from
    /// its next copies, so a stalled island starts over from new designs (Lehman and Miikkulainen,
    /// 2015). The lost elites become fossils, so it can be undone. Returns how
    /// many elites were lost.
    pub fn extinction(&mut self) -> usize {
        let weakest = self
            .islands
            .iter()
            .enumerate()
            .take(island_count())
            .filter(|(_, island)| !island.entries.is_empty())
            .min_by(|a, b| a.1.best_fitness().total_cmp(&b.1.best_fitness()))
            .map(|(index, _)| index);
        let Some(index) = weakest else {
            return 0;
        };
        let lost = std::mem::take(&mut self.islands[index].entries);
        // The island starts over from new bodies, and climbs without classes
        // until it is old enough to be refined again.
        self.islands[index].set_refined(false);
        self.islands[index].rebuild_indices();
        if self.island_epoch.len() <= index {
            self.island_epoch.resize(index + 1, 0);
        }
        self.island_epoch[index] = self.generation;
        let count = lost.len();
        self.fossils
            .extend(lost.into_iter().map(|elite| (Some(index), elite)));
        self.radiate();
        count
    }
    /// Undoes meteor strikes: every fossil returns to its archive if its cell
    /// is empty or holds a slower elite. Returns how many came back.
    pub fn undo_meteor(&mut self) -> usize {
        let mut restored = 0;
        let mut touched = std::collections::BTreeSet::new();
        for (island, elite) in std::mem::take(&mut self.fossils) {
            let archive = match island {
                None => &mut self.archive,
                Some(index) => match self.islands.get_mut(index) {
                    Some(archive) => archive,
                    None => continue,
                },
            };
            let mut elite = elite;
            if !qd::is_morphology_niche(&elite.niche) {
                // The cell in the layout the archive has now.
                elite.niche = archive.cell_of(elite.descriptor);
            }
            match archive.slot_for(&elite.niche) {
                Some(slot) if archive.entries[slot].fitness < elite.fitness => {
                    archive.entries[slot] = elite;
                }
                Some(_) => continue,
                // Later fossils must see this cell as taken.
                None => archive.push_unscored(elite),
            }
            touched.insert(island);
            restored += 1;
        }
        for island in touched {
            match island {
                None => self.archive.rebuild_indices(),
                Some(index) => self.islands[index].rebuild_indices(),
            }
        }
        restored
    }
    /// Clears the archives after the world changed. Their scores no longer
    /// hold, but each island's creatures are queued to compete again under
    /// the new physics in that island's own slots. An archive that was
    /// refined starts again refined, so the re-tested elites keep the cells of
    /// their body classes: the archive refills with evolved bodies, not
    /// random ones, and a climb that spreads over many classes is not at stake.
    /// Only a save written before the first elite re-enters forgets this,
    /// because a save tells a layout by the cells its elites hold.
    pub(super) fn reset_search_context(&mut self) {
        self.reseed.clear();
        self.wild_exports.clear();
        // The nurseries start over; only the islands' creatures are re-tested.
        let mut refined = Vec::new();
        // The refuge takes each island's best. A second change while it lasts
        // keeps the older champions where the islands have none left.
        let mut champions = std::mem::take(&mut self.refuge.champions);
        champions.resize_with(qd::MAIN_ISLANDS, Vec::new);
        // The wild islands live in worlds of their own, which the player's
        // change leaves alone: they keep their archives and nurseries.
        let wild: Vec<(usize, QdArchive)> = if self.islands.len() == arena_count() {
            (0..arena_count())
                .filter(|&a| qd::is_wild(a % island_count()))
                .map(|a| (a, std::mem::take(&mut self.islands[a])))
                .collect()
        } else {
            Vec::new()
        };
        let mut before = vec![f32::NAN; qd::MAIN_ISLANDS];
        for (index, island) in self.islands.iter_mut().take(qd::MAIN_ISLANDS).enumerate() {
            if !island.entries.is_empty() {
                champions[index] = plan_champions(island);
                before[index] = island.best_fitness();
            }
            refined.push(island.refined());
            for elite in std::mem::take(&mut island.entries) {
                self.reseed.push(index, elite.creature.unpack());
            }
        }
        self.archive = QdArchive::starting_global();
        // Fossils are old-world elites: undoing a meteor must not bring them
        // back into the new world's archives.
        self.fossils.clear();
        self.islands = if refined.contains(&true) || !wild.is_empty() {
            new_islands(&refined)
        } else {
            Vec::new()
        };
        for (a, archive) in wild {
            self.islands[a] = archive;
        }
        self.island_progress.clear();
        self.graduations.clear();
        self.reshaped_graduations.clear();
        self.last_migration = None;
        self.emitter_stats = [EmitterStats::default(); qd::EMITTER_COUNT];
        self.cma_emitters.clear();
        // Distances measured in the old world say nothing about the new one,
        // and neither do the audit rows: the rungs disarm and refit.
        self.screen_window.clear();
        self.young_window.clear();
        self.reshaped_window.clear();
        for window in &mut self.wild_windows {
            window.clear();
        }
        self.clade_rarity = (u32::MAX, Vec::new());
        self.rungs.clear();
        self.config.rungs = None;
        self.refuge = Refuge::open(champions, before, self.generation);
    }
}
