//! The background autosave: one save at a time, on its own thread.

use crate::storage::{self, Experiment};
use std::{path::PathBuf, thread::JoinHandle};

/// A background autosave, which reports the file and generation it wrote.
#[derive(Default)]
pub(super) struct Autosave {
    thread: Option<JoinHandle<Option<(PathBuf, u32)>>>,
}

impl Autosave {
    /// Waits for a running autosave to end.
    pub(super) fn join(&mut self) {
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
    /// Starts an autosave of `e` when its generation is due and none is
    /// running.
    pub(super) fn start_if_due(&mut self, e: &Experiment) {
        // Benchmarks keep autosaves off even for a loaded
        // checkpoint, which brings its own interval. Right after a
        // world change every elite is on the GPU for its re-test and
        // the archives are empty; a save holds no creatures in
        // flight, so that autosave would hold no elites and replace a
        // good one. The next autosave, after the re-tests, writes.
        if e.config.checkpoint_interval > 0
            && std::env::var_os("EVOLUTION_BENCH_NO_AUTOSAVE").is_none()
            && e.generation.is_multiple_of(e.config.checkpoint_interval)
            && !e.archive.entries.is_empty()
            && self
                .thread
                .as_ref()
                .is_none_or(|handle| handle.is_finished())
        {
            self.join();
            let path = PathBuf::from(format!("runs/seed-{}-auto.evo", e.config.seed));
            // The ring is shared, not copied: the save holds only
            // the archives and the search state.
            let snapshot = e.clone();
            self.thread = Some(std::thread::spawn(move || {
                crate::threads::pin_pool();
                if let Err(err) = storage::save(&path, &snapshot) {
                    eprintln!("Background checkpoint failed: {err:#}");
                    return None;
                }
                if let Some(dir) = path.parent() {
                    // One autosave per experiment piles up: keep the
                    // three most recent experiments' autosaves.
                    storage::rotate_autosaves(dir, 3);
                }
                Some((path, snapshot.generation))
            }));
        }
    }
    /// The file and generation of an autosave that has just finished.
    pub(super) fn finished(&mut self) -> Option<(PathBuf, u32)> {
        if self
            .thread
            .as_ref()
            .is_some_and(|handle| handle.is_finished())
            && let Some(handle) = self.thread.take()
            && let Ok(Some(written)) = handle.join()
        {
            return Some(written);
        }
        None
    }
}
