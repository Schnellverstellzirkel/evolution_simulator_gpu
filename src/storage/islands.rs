//! The island layout of the search: how many archives there are and where
//! each one sits in `Experiment::islands`. It also holds the steps of the
//! generation boundary that move bodies between archives. They refine the main
//! islands' archives, graduate the nurseries into their islands, migrate the
//! best elites to the hub and pass stepping stones around the isolated islands.

use super::*;

/// Generations between the stepping stones of the island ring.
const STONE_INTERVAL: u32 = 50;
/// Generations a graduate is protected against bodies of other plans.
const GRADUATE_GRACE: u32 = 3;

/// One island's record of what a nursery graduated this session. An island
/// has one for its nursery of new random bodies and one for its nursery of
/// reshaped bodies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Graduation {
    /// Generation of the last graduation (0 before the first).
    pub generation: u32,
    /// Bodies in the last graduating cohort.
    pub sent: usize,
    /// How many of those bodies the island kept.
    pub kept: usize,
    /// Bodies the island kept over all graduations this session.
    pub kept_total: usize,
}

/// How many islands there are: the `ISOLATED_ISLANDS` isolated islands, the
/// hub, then the `qd::WILD_ISLANDS` (100) wild islands. The first
/// `qd::MAIN_ISLANDS` run in the player's world, and each wild island runs in a
/// world of its own. Ring slot `i` belongs to island
/// `qd::island_of_slot(i, island_count())`.
pub fn island_count() -> usize {
    qd::MAIN_ISLANDS + qd::WILD_ISLANDS
}
/// How many archives creatures breed for and compete in: the islands, then
/// one nursery of new random bodies per island, then one nursery of reshaped
/// bodies per island. `Experiment::islands` holds them in this order.
pub fn arena_count() -> usize {
    island_count() * qd::ARENA_KINDS
}
/// The index in `Experiment::islands` of the nursery of new random bodies of
/// `island`.
pub fn nursery_of(island: usize) -> usize {
    island_count() + island
}
/// The index in `Experiment::islands` of the nursery of reshaped bodies of
/// `island`.
pub fn reshaped_of(island: usize) -> usize {
    2 * island_count() + island
}
/// An empty nursery of reshaped bodies of a main island. It starts in the
/// refined layout, so a new body plan has a cell of its own against the bodies
/// of other shapes and sizes. The nursery then holds up to nine times as many
/// bodies as one that keeps one elite per way of moving, because
/// `qd::ISLAND_CLASSES` has nine body classes.
pub(super) fn new_reshaped_nursery() -> QdArchive {
    let mut archive = QdArchive::default();
    archive.set_refined(true);
    archive
}
/// Empty archives for the islands and their two nurseries each, in the order
/// of `Experiment::islands`. `refined[i]` says whether archive `i` starts in
/// the refined layout, and a missing entry says no. The nurseries of reshaped
/// bodies of the main islands always start refined. An archive that nothing
/// marks keeps one elite per way of moving.
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
/// How many isolated islands there are, the first islands. Each breeds from
/// its own archive, so each evolves its own designs. Bodies from other islands
/// reach one only as a stepping stone from the previous isolated island
/// (`Experiment::step_stones`) or through the founder bank, which all the main
/// islands share. Migration never sends anything to an isolated island.
pub const ISOLATED_ISLANDS: usize = 4;
/// The index of the hub island, which comes right after the isolated islands.
/// Every `MIGRATION_INTERVAL` generations it receives copies of each isolated
/// island's best elites and of each wild island's best
/// (`Experiment::migrate_islands`), and it breeds from its own archive like
/// any island. Migration sends nothing back from the hub.
pub fn hub_island() -> usize {
    ISOLATED_ISLANDS
}

/// Generations between migrations to the hub.
pub const MIGRATION_INTERVAL: u32 = 25;
/// The share of an island's elites, the fastest first, that it sends to the
/// hub at a migration. Each isolated island and each wild island sends this
/// share.
pub const MIGRATION_SHARE: f32 = 0.1;

impl Experiment {
    /// Creates the empty archives of the islands and their nurseries when
    /// `islands` does not hold `arena_count()` of them, as in a new game. It
    /// also clears the progress records, the graduation logs and the last
    /// migration. The archives fill from the offspring of their own slots and
    /// from queued reseeds. The global archive is never split among them,
    /// because that would mix the islands.
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
    /// Refines the archives of the main islands as they age. An island that is
    /// `qd::REFINE_AFTER` generations old moves its elites to the cells of
    /// their body classes, and from then on a body of another shape or size has
    /// a cell of its own. Isolated island `i` waits `10 * i` generations
    /// longer. Until then an island keeps one elite per way of moving, so the
    /// climb of a new game pools its lineages as the old archive did. A
    /// refined archive stays refined through a world change
    /// (`reset_search_context`). The wild islands and the nurseries of new
    /// random bodies never refine.
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
    /// Every `qd::NURSERY_GENERATIONS` generations each island takes the
    /// bodies of its nurseries that beat its elites, and the global archive
    /// takes those the island kept. The nursery of new random bodies is
    /// emptied, its morphology reserve with it, and starts over with new
    /// random bodies. The nursery of reshaped bodies keeps every body and goes
    /// on tuning it. The graduation logs record each cohort. Afterwards the
    /// islands and the global archive refresh their behavior scores.
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
        // Each island that took bodies and the global archive refresh once,
        // the islands side by side.
        self.islands[..island_count()]
            .par_iter_mut()
            .filter(|island| !island.scores_current())
            .for_each(|island| island.refresh_behavior_scores());
        if !self.archive.scores_current() {
            self.archive.refresh_behavior_scores();
        }
    }
    /// Refreshes the behavior scores of the nurseries of reshaped bodies, side
    /// by side. They take offers in every block, so they refresh once a
    /// generation and not after each block.
    pub(super) fn refresh_reshaped_scores(&mut self) {
        if self.islands.len() == arena_count() {
            self.islands[reshaped_of(0)..reshaped_of(island_count())]
                .par_iter_mut()
                .for_each(|nursery| nursery.refresh_behavior_scores());
        }
    }
    /// Offers `elites` to `island`, fastest first, marked as graduates: each
    /// takes the cell of an island elite it beats, or an empty cell, and the
    /// global archive takes those the island kept. The island protects a
    /// graduate it keeps for `GRADUATE_GRACE` generations, and the developer
    /// switch `BIO_OFF` bit 16 removes that. Returns how many it kept.
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
    /// fastest `MIGRATION_SHARE` of each isolated island's elites. Each wild
    /// island also sends its fastest share to the hub's pen. Nothing goes to an
    /// isolated island here. `last_migration` records, per island, how many
    /// elites it sent and how many the hub kept. For a wild island the hub has
    /// kept none yet, because its migrants have still to run in the hub's
    /// world.
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
        // A wild island sends its fastest tenth, ranked by distance in its own
        // world. Each of them runs once in the hub's world and takes a cell if
        // it is fast enough there. Each also enters the hub's pen, where it
        // breeds in the hub's slots for `PEN_GENERATIONS` generations, so its
        // line can adapt to the hub's world before it is dropped (owner).
        let until = self.generation + PEN_GENERATIONS;
        #[allow(clippy::needless_range_loop)]
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
    /// elite to the next isolated island in a ring: the fastest elite of its
    /// rarest body plan, so isolation is almost kept and a rare design gets a
    /// second home (Cantu-Paz, 2000, migration topologies). The developer
    /// switch `BIO_OFF` bit 128 turns it off.
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
