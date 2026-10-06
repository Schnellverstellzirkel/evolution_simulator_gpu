//! What counts as a record: the history's records within one world, and the
//! record the running generation has set before its history row exists.

use crate::{storage::Stats, worker::Snapshot};

/// The smallest gain that counts as a new record (m).
const RECORD_STEP: f32 = 0.01;
/// A record set in the generation that is running, before its history row
/// exists.
pub(super) struct LiveRecord {
    pub(super) best: f32,
    /// The first best in a world the history has no row for yet.
    pub(super) first_in_world: bool,
    /// No generation has finished at all.
    pub(super) first_ever: bool,
}
/// History positions where the best distance moved within one world, oldest
/// first, and whether each is the first best after a world change. A harder
/// world lowers the best, so records count again from its first generation.
pub(super) fn world_records(history: &[Stats]) -> Vec<(usize, f32, bool)> {
    let mut best = f32::NEG_INFINITY;
    let mut fresh = false;
    let mut records = Vec::new();
    for (index, stats) in history.iter().enumerate() {
        if index > 0 && stats.config.physics_differs(&history[index - 1].config) {
            best = f32::NEG_INFINITY;
            fresh = true;
        }
        // A record beats the last one by at least a centimeter, so two
        // records never read the same. A row with nothing kept has no best.
        if stats.archive_cells > 0 && stats.best.is_finite() && stats.best >= best + RECORD_STEP {
            best = stats.best;
            records.push((index, best, fresh));
            fresh = false;
        }
    }
    records
}
/// The newest history row when it was measured in the world that is live now.
/// After a world change the rows are from the old world, and nothing from
/// them may stand for the current world.
pub(super) fn row_in_world(snapshot: &Snapshot) -> Option<&Stats> {
    snapshot
        .history
        .last()
        .filter(|row| !row.config.physics_differs(&snapshot.config))
}
/// The running generation's best and median when its row is not written yet:
/// (generation, best, median).
pub(super) fn live_point(snapshot: &Snapshot) -> Option<(u32, f32, f32)> {
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
/// The champion's record, when it beats the last record of this world by a
/// record step. `snapshot.champion` holds its creature.
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
