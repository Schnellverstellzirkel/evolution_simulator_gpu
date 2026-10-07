//! This module handles what a world change or a catastrophe does to the
//! search. `Reseed` queues the elites that compete again, and `Refuge` keeps
//! the old champions that breed for a few generations. Breeding draws from
//! both. The `Experiment` methods here apply new settings, clear the search
//! context after a world change, and run the meteor strike, the extinction and
//! their undo.

use super::*;

/// Creatures queued to be evaluated again, one queue per island. Each comes
/// back in a slot of its own island, so no island's creatures mix into
/// another's. A world change queues the elites of every main island here.
/// The hub's queue also takes the wild islands' migrants for their trial in
/// the hub's world, and a generation dump queues the elites of every island
/// for its re-run. A save keeps the queues.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reseed {
    queues: Vec<Vec<Creature>>,
}

impl Reseed {
    /// How many creatures are queued, over all islands.
    pub fn len(&self) -> usize {
        self.queues.iter().map(Vec::len).sum()
    }
    /// Whether no island has a creature queued.
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(Vec::is_empty)
    }
    /// Drops every queued creature.
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
    /// Takes the creature queued last for `island`, if there is one.
    pub fn pop(&mut self, island: usize) -> Option<Creature> {
        self.queues.get_mut(island)?.pop()
    }
    /// Every queued creature, island by island.
    pub fn iter(&self) -> impl Iterator<Item = &Creature> {
        self.queues.iter().flatten()
    }
    /// Whether every queue belongs to an existing island.
    pub(super) fn fits(&self, islands: usize) -> bool {
        self.queues.len() <= islands
    }
}

/// The old champions of each main island, kept so that they breed again for a
/// few generations after a world change, a meteor strike or an extinction. A
/// world change can kill every old design at once when the first re-test finds
/// them slow, and random bodies take the islands. For `REFUGE_GENERATIONS`
/// generations a share of each island's slots breed children of its old
/// champions, so an old design gets time to retune its gait to the new world
/// (a refugium, as in island models with migration from a reservoir). An
/// island that wins its distance back closes its refuge sooner, and one that
/// lost most of it keeps the refuge up to `REFUGE_LONG` generations. A save
/// does not keep the refuge.
#[derive(Clone, Debug, Default)]
pub struct Refuge {
    /// Per main island: its champions, the fastest elite of each body plan
    /// (`plan_champions`), fastest first.
    champions: Vec<Vec<Creature>>,
    /// The generation the refuge opened.
    since: u32,
    /// The first generation with no refuge at all. It is `since` plus
    /// `REFUGE_LONG`, and breeding skips the refuge from then on.
    pub(super) until: u32,
    /// Per main island: the first generation after its refuge. It starts at
    /// `since` plus `REFUGE_GENERATIONS`. `review` moves it earlier when the
    /// island recovered and later after a heavy loss.
    island_until: Vec<u32>,
    /// Per main island: its best distance before the change, or NaN when
    /// there is none.
    before: Vec<f32>,
}
/// Generations the refuge breeds only gait retunes, before structural
/// changes join (Cheney et al., 2018: a changed body readapts its control
/// first).
const REFUGE_TUNING: u32 = 3;
/// Generations after the change before the refuge of a recovered island may
/// close.
const REFUGE_MIN: u32 = 2;
/// The share of its best distance from before the change that an island must
/// have again to close its refuge.
const REFUGE_RECOVERED: f32 = 0.9;
/// An island under this share of its best distance from before the change
/// keeps its refuge up to `REFUGE_LONG` generations after the change (Branke,
/// 1999: the memory a changed world needs grows with the loss).
const REFUGE_HEAVY: f32 = 0.5;
/// Generations after the change that the refuge of a heavily hit island lasts.
/// No refuge lasts longer.
const REFUGE_LONG: u32 = 10;

/// The champions of an archive for the refuge: the fastest elite of each
/// body plan, fastest first, up to `REFUGE_CHAMPIONS` plans (Schluter, 2000:
/// a radiation grows from many founders, not from many copies of one). The
/// morphology reserve is left out. A meteor strike spares the same elites.
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
/// Generations the old champions keep breeding after a world change or a
/// catastrophe, unless `review` ends the refuge of an island sooner or keeps
/// it longer.
const REFUGE_GENERATIONS: u32 = 5;
/// The most body plans per island whose fastest elite goes into the refuge.
const REFUGE_CHAMPIONS: usize = 64;
/// Share of an island's own slots that breed from its refuge.
const REFUGE_SHARE: f32 = 0.15;

impl Refuge {
    /// Opens a refuge at `generation` with these champions, and the islands'
    /// best distances before the change (NaN for an island with none).
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
    /// At a generation boundary: an island whose best distance is back at
    /// `REFUGE_RECOVERED` of the old one closes its refuge, but not before
    /// `REFUGE_MIN` generations have passed. An island under `REFUGE_HEAVY` of
    /// its old best keeps its refuge up to `REFUGE_LONG` generations after the
    /// change. An island whose old best is unknown or not above zero stays as
    /// it is.
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
    /// A child of one of `island`'s champions for `slot`. It is `None` when
    /// the island's refuge is over, when the island has no champions, and for
    /// the slots that the `REFUGE_SHARE` draw leaves out. `round` is the
    /// breeding round.
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
        // The stream is salted with "refuge" in ASCII.
        let mut rng = evolution::Rng::stream(cfg.seed ^ 0x7265_6675_6765, generation, round, slot);
        if rng.unit() >= REFUGE_SHARE {
            return None;
        }
        let parent = champions[rng.index(champions.len())].clone();
        // Every child gets a gait retune, a tenth of them with a larger step.
        // Once `REFUGE_TUNING` generations have passed, half the children also
        // take a structural mutation.
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
    /// Applies settings at the next generation boundary. It fails if `cfg` is
    /// invalid, changes the population or a seed, or sets limits that the
    /// bodies in the ring and in the global archive exceed.
    pub fn update_config(&mut self, cfg: Config) -> Result<()> {
        self.update_config_at(cfg, false)
    }
    /// Applies settings now. A world change resets the search context at
    /// once. Blocks in flight from the old world are recognized by their own
    /// settings and enter no archive. It fails in the same cases as
    /// `update_config`.
    pub fn update_config_now(&mut self, cfg: Config) -> Result<()> {
        self.update_config_at(cfg, true)
    }
    /// Checks `cfg`, then either applies it now or keeps it in `pending` for
    /// the generation boundary. Applying it drops any pending settings, resets
    /// the search context if the physics differ, sets the screen bar again and
    /// keeps the early rungs unless the world changed.
    fn update_config_at(&mut self, mut cfg: Config, now: bool) -> Result<()> {
        cfg.validate()?;
        // The autochange step advances at the generation boundary, so the
        // settings the player sent may hold an older count. Keep the larger
        // one, because a checkpoint saves this counter and a settings update
        // must never rewind it.
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
    /// A meteor strike wipes out about `share` of the elites in the global
    /// archive and in every island and nursery, chosen at random. It spares the
    /// fastest elite of each of an archive's `REFUGE_CHAMPIONS` fastest body
    /// plans. Survivors and new offspring refill the emptied cells, which
    /// opens room for new kinds of movement. The lost elites become fossils so
    /// `undo_meteor` can bring them back, and `radiate` opens a refuge for the
    /// survivors. Returns how many elites were lost.
    pub fn meteor(&mut self, share: f32) -> usize {
        // The stream is salted with "meteor" in ASCII. The fossils so far tell
        // two strikes in one generation apart.
        let mut rng = evolution::Rng::new(
            self.config.seed ^ 0x6d65_7465_6f72,
            self.generation,
            self.fossils.len(),
        );
        // The strike spares the plan champions, so the survivors are the rare
        // plans and the common ones thin out (Raup, 1986, selective
        // extinction).
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
    /// emptied cells (Lehman and Miikkulainen, 2015). The new refuge replaces
    /// any refuge that was open.
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
    /// An extinction wipes out the island whose best creature is slowest, so a
    /// stalled island starts over from new designs (Lehman and Miikkulainen,
    /// 2015). The candidates are the main and the wild islands, not the
    /// nurseries or the global archive. An emptied isolated island breeds new
    /// random bodies, and the hub refills from the copies it receives. The
    /// lost elites become fossils, so `undo_meteor` can bring them back.
    /// Returns how many elites were lost, which is 0 when every island is
    /// empty.
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
    /// Undoes meteor strikes and extinctions: every fossil returns to its
    /// archive if its cell is empty or holds a slower elite. A fossil that
    /// finds neither is dropped. Returns how many came back.
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
    /// Clears the search context after the world changed. The old scores no
    /// longer hold, so each main island's elites are queued in `reseed` to
    /// compete again under the new physics, each in a slot of its own island.
    /// The global archive and the nurseries of the main islands start empty,
    /// the fossils are dropped, and the old champions open a new refuge. The
    /// wild islands live in worlds of their own and keep their archives and
    /// nurseries. An archive that was refined starts again refined, so the
    /// re-tested elites keep the cells of their body classes. A coarse archive
    /// would keep one elite per way of moving and lose the classes at every
    /// change. Only a save written before the first elite re-enters forgets
    /// the layout, because a save tells a layout by the cells its elites hold.
    pub(super) fn reset_search_context(&mut self) {
        // The queue starts again from the main islands' elites below. The wild
        // migrants that waited in it for their hub trial are dropped, and their
        // origins are forgotten.
        self.reseed.clear();
        self.wild_exports.clear();
        // The layout of each main island, for the new archives below.
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
        // Each main island's elites are queued to compete again, and its best
        // go to the refuge. The nurseries start over, so only the islands'
        // creatures are re-tested.
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
        // and neither do the audit rows: the rungs disarm and refit. The wild
        // islands' windows start over too, and the clade rarity of the old
        // archives is dropped.
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
