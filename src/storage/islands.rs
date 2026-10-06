use super::*;

/// Generations between the stepping stones of the island ring.
const STONE_INTERVAL: u32 = 50;
/// Generations a graduate is protected against bodies of other plans.
const GRADUATE_GRACE: u32 = 3;

/// What one island's nursery graduated this session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Graduation {
    /// Generation of the last graduation (0 before the first).
    pub generation: u32,
    /// Bodies in the last graduating cohort, and how many the island kept.
    pub sent: usize,
    pub kept: usize,
    /// Bodies the island kept over all graduations this session.
    pub kept_total: usize,
}

/// Island archives: `ISOLATED_ISLANDS` isolated islands, then the hub.
/// Slot `i` breeds from and competes in island `qd::island_of_slot`.
pub fn island_count() -> usize {
    qd::MAIN_ISLANDS + qd::WILD_ISLANDS
}
/// Archives that creatures breed for and compete in: the islands, then one
/// nursery of new random bodies per island, then one nursery of reshaped
/// bodies per island. `Experiment::islands` holds them in this order.
pub fn arena_count() -> usize {
    island_count() * qd::ARENA_KINDS
}
/// The nursery archive of new random bodies of `island` in
/// `Experiment::islands`.
pub fn nursery_of(island: usize) -> usize {
    island_count() + island
}
/// The nursery archive of reshaped bodies of `island`.
pub fn reshaped_of(island: usize) -> usize {
    2 * island_count() + island
}
/// An empty nursery of reshaped bodies of a main island. It starts in the refined layout, so
/// a new body plan has a cell of its own against the bodies of other shapes
/// and sizes, and the nursery holds four times as many bodies as one that
/// keeps one elite per way of moving.
pub(super) fn new_reshaped_nursery() -> QdArchive {
    let mut archive = QdArchive::default();
    archive.set_refined(true);
    archive
}
/// Empty archives for the islands and their two nurseries each. `refined`
/// says which islands start in the refined layout. The others, and the
/// nursery of new random bodies, keep one elite per way of moving.
pub(super) fn new_islands(refined: &[bool]) -> Vec<QdArchive> {
    (0..arena_count())
        .map(|arena| {
            if arena >= reshaped_of(0) && !qd::is_wild(arena - reshaped_of(0)) {
                return new_reshaped_nursery();
            }
            let mut archive = QdArchive::default();
            archive.set_refined(refined.get(arena).copied().unwrap_or(false));
            archive
        })
        .collect()
}
/// Islands that never receive immigrants and breed only from their own
/// elites, so each one evolves its own designs.
pub const ISOLATED_ISLANDS: usize = 4;
/// The hub island: every `MIGRATION_INTERVAL` generations it receives copies
/// of each isolated island's best elites, and it breeds from its own archive
/// like any island. Nothing flows from the hub back.
pub fn hub_island() -> usize {
    ISOLATED_ISLANDS
}

/// Generations between migrations to the hub, and the share of each
/// isolated island's elites copied to it.
pub const MIGRATION_INTERVAL: u32 = 25;

pub const MIGRATION_SHARE: f32 = 0.1;

impl Experiment {
    /// Creates empty island archives if they are missing. They fill from
    /// their own slots' offspring (and queued reseeds). The global archive
    /// is never split among them, because that would mix the islands.
    pub(super) fn ensure_islands(&mut self) {
        if self.islands.len() == arena_count() {
            return;
        }
        self.islands = new_islands(&[]);
        self.island_progress.clear();
        self.graduations.clear();
        self.reshaped_graduations.clear();
        self.last_migration = None;
    }
    /// Refines each island whose archive is `qd::REFINE_AFTER` generations
    /// old: its elites move to the cells of their body classes, and from then
    /// on a body of another shape or size has a cell of its own. Until then
    /// an island keeps one elite per way of moving, so the climb of a new game
    /// pools its lineages as the old archive did. A refined archive stays
    /// refined through a world change (`reset_search_context`), and the
    /// nurseries of new random bodies never refine.
    pub(super) fn refine_archives(&mut self) {
        for island in 0..island_count().min(self.islands.len()) {
            // A wild island keeps one elite per way of moving: a hundred
            // refined islands with two refined nurseries each held 20 GB.
            if qd::is_wild(island) {
                continue;
            }
            let epoch = self.island_epoch.get(island).copied().unwrap_or(0);
            let archive = &mut self.islands[island];
            // The isolated islands refine 10 generations apart, so climbing
            // and refined islands exist side by side (Hornby, 2006).
            let after = qd::REFINE_AFTER
                + if island < ISOLATED_ISLANDS {
                    10 * island as u32
                } else {
                    0
                };
            if !archive.refined() && self.generation.saturating_sub(epoch) >= after {
                archive.set_refined(true);
                archive.rebin();
            }
        }
    }
    /// Every `NURSERY_GENERATIONS` generations each island takes the bodies
    /// of its nurseries that beat its elites, and the global archive takes
    /// those the island kept. The nursery of new random bodies starts over
    /// with new random bodies. The nursery of reshaped bodies keeps every
    /// body and goes on tuning it.
    pub(super) fn graduate_nurseries(&mut self) {
        if self.islands.len() != arena_count()
            || self.generation == 0
            || !self.generation.is_multiple_of(qd::NURSERY_GENERATIONS)
        {
            return;
        }
        self.graduations
            .resize(island_count(), Graduation::default());
        self.reshaped_graduations
            .resize(island_count(), Graduation::default());
        for island in 0..island_count() {
            let nursery = nursery_of(island);
            let mut cohort: Vec<qd::Elite> = std::mem::take(&mut self.islands[nursery].entries)
                .into_iter()
                .filter(|e| !qd::is_morphology_niche(&e.niche))
                .collect();
            self.islands[nursery].rebuild_indices();
            let sent = cohort.len();
            let kept = self.offer_to_island(island, &mut cohort);
            let log = &mut self.graduations[island];
            *log = Graduation {
                generation: self.generation,
                sent,
                kept,
                kept_total: log.kept_total + kept,
            };
            if let Some(progress) = self.island_progress.get_mut(nursery) {
                *progress = (f32::NEG_INFINITY, self.generation);
            }
            // Only the reshaped bodies the island would take are copied.
            let home = &self.islands[island];
            let reshaped = &self.islands[reshaped_of(island)];
            let sent = reshaped.behavior_count();
            let mut winners: Vec<qd::Elite> = reshaped
                .entries
                .iter()
                .filter(|e| home.would_take(e))
                .cloned()
                .collect();
            let kept = self.offer_to_island(island, &mut winners);
            let log = &mut self.reshaped_graduations[island];
            *log = Graduation {
                generation: self.generation,
                sent,
                kept,
                kept_total: log.kept_total + kept,
            };
        }
        // Each island that took bodies and the global archive refresh once.
        for island in &mut self.islands[..island_count()] {
            if !island.scores_current() {
                island.refresh_behavior_scores();
            }
        }
        if !self.archive.scores_current() {
            self.archive.refresh_behavior_scores();
        }
    }
    /// The behavior scores of the nurseries of reshaped bodies, which take
    /// offers in every block, are refreshed once a generation.
    pub(super) fn refresh_reshaped_scores(&mut self) {
        if self.islands.len() == arena_count() {
            for island in 0..island_count() {
                self.islands[reshaped_of(island)].refresh_behavior_scores();
            }
        }
    }
    /// Offers `elites` to `island`, fastest first, marked as graduates: each
    /// takes the cell of an island elite it beats, or an empty cell, and the
    /// global archive takes those the island kept. Returns how many it kept.
    fn offer_to_island(&mut self, island: usize, elites: &mut [qd::Elite]) -> usize {
        elites.sort_unstable_by(|a, b| {
            b.fitness
                .total_cmp(&a.fitness)
                .then_with(|| a.niche.cmp(&b.niche))
        });
        let mut kept = 0;
        // A graduate keeps its cell against bodies of other plans for a few
        // generations, as a young species is protected in NEAT (Stanley and
        // Miikkulainen, 2002). The global archive's copy has no grace.
        let grace = self.generation + GRADUATE_GRACE;
        for elite in elites.iter_mut() {
            elite.graduate = true;
            let mut copy = elite.clone();
            if !qd::bio_off(16) {
                copy.protected_until = copy.protected_until.max(grace);
            }
            if self.islands[island].absorb(&copy) {
                kept += 1;
                self.archive.absorb(elite);
            }
        }
        kept
    }
    /// Every `MIGRATION_INTERVAL` generations the hub receives copies of the
    /// best share of each isolated island's elites. The isolated islands
    /// never receive any.
    pub(super) fn migrate_islands(&mut self) {
        if self.islands.len() != arena_count()
            || !self.generation.is_multiple_of(MIGRATION_INTERVAL)
        {
            return;
        }
        let hub = hub_island();
        let migrants: Vec<Vec<qd::Elite>> = self.islands[..ISOLATED_ISLANDS]
            .iter()
            .map(|island| {
                let mut elites: Vec<&qd::Elite> = island
                    .entries
                    .iter()
                    .filter(|e| !qd::is_morphology_niche(&e.niche))
                    .collect();
                elites.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
                let take =
                    ((elites.len() as f32 * MIGRATION_SHARE).ceil() as usize).min(elites.len());
                elites[..take].iter().map(|e| (*e).clone()).collect()
            })
            .collect();
        let mut exchange = vec![(0, 0); island_count()];
        for (from, group) in migrants.into_iter().enumerate() {
            let to = &mut self.islands[hub];
            let kept = group.iter().filter(|elite| to.absorb(elite)).count();
            exchange[from] = (group.len(), kept);
        }
        // A wild island sends its best tenth by its own world's distance. Each
        // runs once as it is in the hub's world and takes a cell if it is
        // fast enough there; and it enters the hub's pen, where it breeds in
        // the hub's slots for `PEN_GENERATIONS` generations so its line can
        // adapt to the hub's world before it is dropped (owner).
        let until = self.generation + PEN_GENERATIONS;
        for from in qd::MAIN_ISLANDS..island_count() {
            let island = &self.islands[from];
            let mut elites: Vec<&qd::Elite> = island
                .entries
                .iter()
                .filter(|e| !qd::is_morphology_niche(&e.niche))
                .collect();
            elites.sort_unstable_by(|a, b| b.fitness.total_cmp(&a.fitness));
            let take = ((elites.len() as f32 * MIGRATION_SHARE).ceil() as usize).min(elites.len());
            let sent: Vec<Creature> = elites[..take].iter().map(|e| e.creature.unpack()).collect();
            self.pen.extend(sent.iter().map(|c| (c.clone(), until)));
            exchange[from] = (sent.len(), 0);
            for creature in sent {
                self.wild_exports.insert(creature.id, from);
                self.reseed.push(hub, creature);
            }
        }
        self.last_migration = Some((self.generation, exchange));
        self.islands[hub].refresh_behavior_scores();
    }
    /// Every `STONE_INTERVAL` generations each isolated island sends one
    /// elite to the next island in a ring: the fastest elite of its rarest
    /// body plan, so isolation is almost kept and a rare design gets a second
    /// home (Cantu-Paz, 2000, migration topologies).
    pub(super) fn step_stones(&mut self) {
        if qd::bio_off(128)
            || self.islands.len() != arena_count()
            || self.generation == 0
            || !self.generation.is_multiple_of(STONE_INTERVAL)
        {
            return;
        }
        let stones: Vec<Option<qd::Elite>> = (0..ISOLATED_ISLANDS)
            .map(|island| {
                let archive = &self.islands[island];
                let mut counts: HashMap<u64, usize> = HashMap::new();
                for i in 0..archive.entries.len() {
                    if !qd::is_morphology_niche(&archive.entries[i].niche) {
                        *counts.entry(archive.plan_key(i)).or_default() += 1;
                    }
                }
                (0..archive.entries.len())
                    .filter(|&i| !qd::is_morphology_niche(&archive.entries[i].niche))
                    .min_by(|&a, &b| {
                        counts[&archive.plan_key(a)]
                            .cmp(&counts[&archive.plan_key(b)])
                            .then(
                                archive.entries[b]
                                    .fitness
                                    .total_cmp(&archive.entries[a].fitness),
                            )
                            .then(a.cmp(&b))
                    })
                    .map(|i| archive.entries[i].clone())
            })
            .collect();
        for (island, stone) in stones.into_iter().enumerate() {
            if let Some(elite) = stone {
                self.islands[(island + 1) % ISOLATED_ISLANDS].absorb(&elite);
            }
        }
    }
}
