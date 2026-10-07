//! What counts as a record, and which history numbers belong to the world now.
//! `world_records` finds the records of the history within one world, and
//! `live_record` finds the record the running generation has set before its
//! history row exists. `row_in_world` gives the newest history row when it was
//! measured in the world now, and `live_point` gives the running generation's
//! best and median. The Overview and its feed, the History tab, Ways of moving
//! and the viewport call them.

use crate::{storage::Stats, worker::Snapshot};

/// The smallest gain that counts as a new record, in meters: a centimeter.
const RECORD_STEP: f32 = 0.01;
/// A record set in the generation that is running, before its history row
/// exists.
pub(super) struct LiveRecord {
    /// The best distance of the global archive now, in meters.
    pub(super) best: f32,
    /// The newest history row is from another world, so this is the first
    /// best in the world now.
    pub(super) first_in_world: bool,
    /// The history is empty: no generation has finished at all.
    pub(super) first_ever: bool,
}
/// The records of the history, oldest first. Each is a history index, the best
/// distance in that row and whether it is the first best after a world change.
/// A record beats the last one by at least `RECORD_STEP`. A harder world lowers
/// the best, so a world change forgets the last record and records count again
/// from the first generation of the new world.
pub(super) fn world_records(history: &[Stats]) -> Vec<(usize, f32, bool)> {
    let mut best = f32::NEG_INFINITY;
    let mut fresh = false;
    let mut records = Vec::new();
    for (index, stats) in history.iter().enumerate() {
        if index > 0 && stats.config.physics_differs(&history[index - 1].config) {
            // The old records no longer count. The next record is the first
            // of the new world.
            best = f32::NEG_INFINITY;
            fresh = true;
        }
        // A record beats the last one by at least a centimeter, so two
        // records never show the same distance to two decimals. A row with
        // nothing kept has no best.
        if stats.archive_cells > 0 && stats.best.is_finite() && stats.best >= best + RECORD_STEP {
            best = stats.best;
            records.push((index, best, fresh));
            fresh = false;
        }
    }
    records
}
/// The newest history row when it was measured in the world that is live now.
/// It is `None` when there is no row or the newest row is from another world.
/// After a world change the rows are from the old world, and nothing from
/// them may stand for the current world.
pub(super) fn row_in_world(snapshot: &Snapshot) -> Option<&Stats> {
    snapshot
        .history
        .last()
        .filter(|row| !row.config.physics_differs(&snapshot.config))
}
/// The running generation's best and median when its row is not written yet:
/// (generation, best, median). It is `None` once the history reaches the
/// running generation, and while the archive holds no elite.
pub(super) fn live_point(snapshot: &Snapshot) -> Option<(u32, f32, f32)> {
    // The history is behind when it has no row for the running generation.
    let behind = snapshot
        .history
        .last()
        .is_none_or(|last| snapshot.generation > last.generation);
    (behind && snapshot.live_best.is_finite()).then_some((
        snapshot.generation,
        snapshot.live_best,
        snapshot.live_median,
    ))
}
/// The record the running generation has set, before its history row exists.
/// The live best is one when it beats the last record of the history by
/// `RECORD_STEP`, or when it is the first best of the game or of a new world.
/// It is `None` without a `live_point` or a champion. `snapshot.champion`
/// holds the creature of the record.
pub(super) fn live_record(snapshot: &Snapshot) -> Option<LiveRecord> {
    let (_, best, _) = live_point(snapshot)?;
    snapshot.champion.as_ref()?;
    let Some(last) = snapshot.history.last() else {
        return Some(LiveRecord {
            best,
            first_in_world: false,
            first_ever: true,
        });
    };
    if snapshot.config.physics_differs(&last.config) {
        return Some(LiveRecord {
            best,
            first_in_world: true,
            first_ever: false,
        });
    }
    let held = world_records(&snapshot.history)
        .last()
        .map_or(f32::NEG_INFINITY, |&(_, b, _)| b);
    (best >= held + RECORD_STEP).then_some(LiveRecord {
        best,
        first_in_world: false,
        first_ever: false,
    })
}
