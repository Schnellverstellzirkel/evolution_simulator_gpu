//! The snapshot the UI draws: built from the game and the loop's state, at
//! most five times a second while the game runs.

use super::{
    Card, CardList, IslandSummary, Loop, MapCell, MigrationSummary, Snapshot, end_to_end_rate,
};
use crate::{
    config::Config,
    evolution::Creature,
    gpu::Gpu,
    qd::{self, EmitterStats},
    storage::Experiment,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

impl Loop {
    /// Publishes a snapshot when something changed, at most every 200 ms
    /// while running.
    pub(super) fn publish_if_due(&mut self) {
        if self.changed
            && (self.last_publish.elapsed() > Duration::from_millis(200) || !self.running)
        {
            let build_started = Instant::now();
            let snapshot = self.snapshot();
            self.benchmark
                .snapshot_built(build_started.elapsed().as_secs_f64() * 1e3);
            *self.output.lock().unwrap() = Some(snapshot);
            self.ctx.request_repaint();
            self.changed = false;
            self.last_publish = Instant::now();
        }
    }
    /// The game now, with what the UI asked for once (`Command::Cards`,
    /// `Select`, `Lineage`) and the preview taken from the loop's state.
    fn snapshot(&mut self) -> Snapshot {
        let Some(e) = &self.exp else {
            return self.empty_snapshot();
        };
        if self.history.len() != e.history.len() {
            self.history = Arc::new(e.history.clone());
        }
        if self.want_map {
            let key = (
                self.epoch,
                e.archive.entries.len(),
                e.archive.qd_score.to_bits(),
            );
            if self.map.is_none() || key != self.map_key {
                self.map_key = key;
                self.map = Some(map_table(e));
            }
        }
        // The best elite by distance, the first one on a tie. Only a
        // new record clones a creature.
        let best = e
            .archive
            .entries
            .iter()
            .filter(|elite| elite.fitness.is_finite())
            .reduce(|a, b| if b.fitness > a.fitness { b } else { a });
        let key = best.map(|elite| (self.epoch, elite.creature.id));
        if key != self.champion_key {
            self.champion_key = key;
            self.champion = best.map(|elite| Arc::new(elite.replay_of(&e.config)));
        }
        let live_best = best.map_or(f32::NAN, |elite| elite.fitness.max(0.0));
        let live_median = live_median(e);
        let archive_count = e.archive.entries.len();
        // The ranked archive, built only when the UI asks: one sort
        // and one copy of each kept creature (about 1,500 at 3M).
        let cards = std::mem::take(&mut self.send_cards).then(|| card_list(e));
        Snapshot {
            epoch: self.epoch,
            config: e.config.clone(),
            pending: e.pending.clone(),
            fossils: e.fossils.len(),
            generation: e.generation,
            evaluated: e.evaluated,
            completed: e.evaluated,
            checking: self.ring.confirming(),
            running: self.running,
            history: self.history.clone(),
            events: self.events.clone(),
            map: self.map.clone(),
            selected: self.selected.take(),
            cards,
            preview: self.preview.take(),
            champion: self.champion.clone(),
            live_best,
            live_median,
            lineage: self.lineage.take(),
            gpu: self.gpu.names(),
            engines: engine_rows(&self.gpu),
            end_to_end: end_to_end_rate(&self.generation_marks),
            gpu_bytes: self.gpu.allocated_bytes,
            ram_bytes: ram_bytes(e),
            elapsed: e.evaluation_seconds,
            archive_cells: e.archive.behavior_count(),
            movement_cells: e.archive.movement_count(),
            archive_size: archive_count,
            innovation_reserve_count: e.archive.morphology_count(),
            qd_score: e.archive.qd_score,
            emitters: e.emitter_stats,
            emitter_weights: qd::emitter_weights(&e.emitter_stats),
            islands: island_summaries(e),
            wild_wins: e.wild_wins.clone(),
            strangest: strangest(e),
            migration: e
                .last_migration
                .clone()
                .map(|(generation, exchange)| MigrationSummary {
                    generation,
                    exchange,
                }),
            status: self.status.clone(),
            error: self.error.clone(),
        }
    }
    /// The snapshot while there is no game.
    fn empty_snapshot(&mut self) -> Snapshot {
        Snapshot {
            epoch: self.epoch,
            config: Config::default(),
            pending: None,
            fossils: 0,
            generation: 0,
            evaluated: 0,
            completed: 0,
            checking: 0,
            running: false,
            history: self.history.clone(),
            events: self.events.clone(),
            map: None,
            selected: self.selected.take(),
            cards: None,
            preview: None,
            champion: None,
            live_best: f32::NAN,
            live_median: f32::NAN,
            lineage: None,
            gpu: self.gpu.names(),
            engines: engine_rows(&self.gpu),
            end_to_end: end_to_end_rate(&self.generation_marks),
            gpu_bytes: self.gpu.allocated_bytes,
            ram_bytes: 0,
            elapsed: 0.,
            archive_cells: 0,
            movement_cells: 0,
            archive_size: 0,
            innovation_reserve_count: 0,
            qd_score: 0.0,
            emitters: [EmitterStats::default(); 4],
            emitter_weights: qd::emitter_weights(&[EmitterStats::default(); 4]),
            islands: Vec::new(),
            wild_wins: Vec::new(),
            strangest: None,
            migration: None,
            status: self.status.clone(),
            error: self.error.clone(),
        }
    }
}
/// The behavior elites of the global archive, best first, for the map.
fn map_table(e: &Experiment) -> Arc<Vec<MapCell>> {
    let mut order: Vec<usize> = (0..e.archive.entries.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        e.archive.entries[b]
            .fitness
            .total_cmp(&e.archive.entries[a].fitness)
    });
    Arc::new(
        order
            .into_iter()
            .enumerate()
            .filter(|&(_, i)| {
                !qd::is_morphology_niche(&e.archive.entries[i].niche)
                    && e.archive.entries[i].fitness.is_finite()
            })
            .map(|(rank, i)| {
                let elite = &e.archive.entries[i];
                MapCell {
                    niche: elite.descriptor.niche().0,
                    score: elite.fitness,
                    rank,
                    id: elite.creature.id,
                }
            })
            .collect(),
    )
}
/// The median distance of the best elite of each way of moving, like a
/// history row (NaN before any elite).
fn live_median(e: &Experiment) -> f32 {
    // The best elite of each way of moving, like a history row.
    let mut kept: Vec<f32> = e
        .archive
        .best_per_way_of_moving()
        .into_iter()
        .map(|slot| e.archive.entries[slot].fitness)
        .collect();
    if kept.is_empty() {
        f32::NAN
    } else {
        // The same rank a history row's median uses.
        let rank = ((kept.len() - 1) as f32 * 0.5).round() as usize;
        kept.select_nth_unstable_by(rank, |a, b| b.total_cmp(a));
        kept[rank]
    }
}
/// The whole archive ranked by distance, one copy of each kept creature.
fn card_list(e: &Experiment) -> CardList {
    let archive_count = e.archive.entries.len();
    let mut order: Vec<_> = (0..archive_count).collect();
    order.sort_unstable_by(|&a, &b| {
        e.archive.entries[b]
            .fitness
            .total_cmp(&e.archive.entries[a].fitness)
            .then(a.cmp(&b))
    });
    CardList {
        generation: e.generation,
        config: e.config.clone(),
        cards: Arc::new(
            order
                .into_iter()
                .enumerate()
                .map(|(rank, i)| {
                    let elite = &e.archive.entries[i];
                    Card {
                        index: i,
                        rank,
                        score: elite.fitness,
                        parent_score: f32::NAN,
                        survivor: false,
                        descriptor: Some(elite.descriptor),
                        emitter: Some(elite.emitter),
                        visits: elite.visits,
                        innovation_reserve: qd::is_morphology_niche(&elite.niche),
                        creature: elite.creature.unpack(),
                        fine: elite.fine,
                    }
                })
                .collect(),
        ),
    }
}
/// Memory held by the ring and the archive's creatures.
fn ram_bytes(e: &Experiment) -> usize {
    e.ring_bytes()
        + e.archive
            .entries
            .iter()
            .map(|elite| {
                elite.creature.node_count() * std::mem::size_of::<crate::evolution::NodeGene>()
                    + elite.creature.bone_count() * std::mem::size_of::<crate::evolution::Bone>()
                    + elite.creature.muscle_count()
                        * std::mem::size_of::<crate::evolution::Muscle>()
            })
            .sum::<usize>()
}
/// Each island archive at a glance, in island order.
fn island_summaries(e: &Experiment) -> Vec<IslandSummary> {
    (0..crate::storage::island_count())
        .filter_map(|i| {
            let log =
                |logs: &[crate::storage::Graduation]| logs.get(i).copied().unwrap_or_default();
            let (random, reshaped) = (log(&e.graduations), log(&e.reshaped_graduations));
            Some(IslandSummary::of(
                e.islands.get(i)?,
                [
                    e.islands.get(crate::storage::nursery_of(i))?,
                    e.islands.get(crate::storage::reshaped_of(i))?,
                ],
                crate::storage::Graduation {
                    generation: random.generation.max(reshaped.generation),
                    sent: random.sent + reshaped.sent,
                    kept: random.kept + reshaped.kept,
                    kept_total: random.kept_total + reshaped.kept_total,
                },
            ))
        })
        .collect()
}
/// The main islands' elite with the body farthest from the others.
fn strangest(e: &Experiment) -> Option<Creature> {
    e.islands
        .iter()
        .take(qd::MAIN_ISLANDS)
        .filter_map(|island| island.strangest())
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, elite)| elite.creature.unpack())
}
fn engine_rows(gpu: &Gpu) -> Vec<(String, f64, u64)> {
    gpu.sched.as_ref().map_or_else(Vec::new, |sched| {
        sched
            .devices
            .iter()
            .map(|d| (d.engine.name(), d.rate, d.creatures))
            .collect()
    })
}
