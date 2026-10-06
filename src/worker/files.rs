//! Loading and saving. A save loads on its own thread while the window
//! shows its progress; a save to disk runs once its "Saving" status has
//! reached the window.

use super::{EventKind, Loop, log_event, log_history_world_changes};
use crate::storage::{self, Experiment};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};

/// A save loading on its own thread.
pub(super) struct Loading {
    path: std::path::PathBuf,
    progress: Arc<storage::Progress>,
    handle: std::thread::JoinHandle<anyhow::Result<Experiment>>,
    started: Instant,
}

impl Loop {
    /// Stops the load in progress, if there is one.
    pub(super) fn cancel_load(&mut self) {
        if let Some(old) = self.loading.take() {
            old.progress.cancel.store(true, Ordering::Relaxed);
        }
    }
    /// `Command::Load`: drops the current game and starts reading the save.
    pub(super) fn start_load(&mut self, path: PathBuf) -> anyhow::Result<()> {
        // An incompatible save is turned down from its header,
        // before gigabytes are read.
        let header = storage::check(&path)?;
        self.cancel_load();
        if let Some(sched) = self.gpu.sched.as_mut() {
            self.ring.stop(sched);
        }
        self.running = false;
        self.ring_meter.clear();
        // Holding the current game while a 3M save loads
        // doubles the memory and can push the machine into
        // swap: let it go first.
        self.exp = None;
        self.preview = None;
        self.lineage = None;
        self.epoch += 1;
        self.history = Arc::new(Vec::new());
        let progress = Arc::new(storage::Progress::default());
        let thread_progress = progress.clone();
        let thread_path = path.clone();
        let handle = std::thread::Builder::new()
            .name("load".into())
            .spawn(move || {
                crate::threads::pin_pool();
                storage::load_with_progress(&thread_path, Some(&thread_progress))
            })?;
        self.status = format!(
            "Loading {} (generation {}, {} creatures)…",
            path.display(),
            header.generation,
            header.population
        );
        self.loading = Some(Loading {
            path,
            progress,
            handle,
            started: Instant::now(),
        });
        Ok(())
    }
    /// Takes a finished load, or shows the progress of one that is not.
    pub(super) fn poll_load(&mut self) {
        let Some(load) = &self.loading else {
            return;
        };
        if load.handle.is_finished() {
            let load = self.loading.take().expect("a load in progress");
            self.finish_load(load);
        } else if self.last_progress.elapsed() > Duration::from_millis(250) {
            let done = load.progress.done.load(Ordering::Relaxed) as f64;
            let total = load.progress.total.load(Ordering::Relaxed).max(1) as f64;
            self.status = if done < total {
                format!(
                    "Loading {}: {:.0}% of {:.0} MB, {:.0} s",
                    load.path.display(),
                    100.0 * done / total,
                    total / 1e6,
                    load.started.elapsed().as_secs_f64()
                )
            } else {
                // A save holds the archives; the population is bred again.
                format!(
                    "Loading {}: breeding the population from the archives, {:.0} s",
                    load.path.display(),
                    load.started.elapsed().as_secs_f64()
                )
            };
            self.last_progress = Instant::now();
            self.changed = true;
        }
    }
    /// A load thread has ended: the loaded game replaces the current one, or
    /// the error is shown.
    fn finish_load(&mut self, load: Loading) {
        match load.handle.join() {
            Ok(Ok(mut next)) => {
                // A loaded game starts with autosave off, like a new
                // one, whatever interval the checkpoint carried. An
                // unattended run (`EVOLUTION_AUTOSTART`) autosaves,
                // as an unattended new game does.
                next.config.checkpoint_interval =
                    if std::env::var_os("EVOLUTION_AUTOSTART").is_some() {
                        crate::ui::AUTOSAVE_INTERVAL
                    } else {
                        0
                    };
                let creature = next
                    .archive
                    .entries
                    .iter()
                    .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
                    .map_or_else(
                        || next.blocks[0].population.creature(0),
                        |elite| elite.creature.unpack(),
                    );
                self.preview = Some((creature, next.config.clone()));
                self.events = Arc::new(Vec::new());
                log_history_world_changes(&mut self.events, &next.history);
                log_event(
                    &mut self.events,
                    next.generation,
                    EventKind::Opened,
                    format!(
                        "Opened {} at generation {}.",
                        load.path.file_name().map_or_else(
                            || load.path.display().to_string(),
                            |name| name.to_string_lossy().into_owned()
                        ),
                        next.generation
                    ),
                );
                self.exp = Some(next);
                self.epoch += 1;
                self.history = Arc::new(Vec::new());
                self.status = format!(
                    "Loaded {} in {:.1} s",
                    load.path.display(),
                    load.started.elapsed().as_secs_f64()
                );
            }
            Ok(Err(err)) => {
                self.error = Some(format!("{err:#}"));
                self.status = format!("Could not load {}", load.path.display());
            }
            Err(_) => {
                self.error = Some("The loading thread stopped unexpectedly".into());
                self.status = format!("Could not load {}", load.path.display());
            }
        }
        self.changed = true;
    }
    /// `Command::Save`: saved once this status reaches the window.
    pub(super) fn request_save(&mut self, path: PathBuf) {
        if self.exp.is_some() {
            // Saved after this status reaches the window.
            self.status = format!("Saving {}…", path.display());
            self.pending_save = Some(path);
        }
    }
    /// `Command::Export`: the history as CSV.
    pub(super) fn export(&mut self, path: PathBuf) -> anyhow::Result<()> {
        if let Some(e) = &self.exp {
            storage::export_csv(&path, &e.history)?;
            self.status = format!("Exported {}", path.display());
        }
        Ok(())
    }
    /// A save runs once its "Saving" status is on screen.
    pub(super) fn save_pending(&mut self) {
        if let Some(path) = self.pending_save.take()
            && let Some(e) = &self.exp
        {
            let started = Instant::now();
            match storage::save(&path, e) {
                Ok(()) => {
                    let seconds = started.elapsed().as_secs_f64();
                    let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
                    self.status = format!("Saved {} in {:.1} s", path.display(), seconds);
                    log_event(
                        &mut self.events,
                        e.generation,
                        EventKind::Saved,
                        format!(
                            "Saved {} ({:.1} MB in {:.1} s).",
                            path.display(),
                            bytes as f64 / 1e6,
                            seconds
                        ),
                    );
                }
                Err(err) => {
                    self.error = Some(format!("{err:#}"));
                    self.status = format!("Could not save {}", path.display());
                }
            }
            self.changed = true;
        }
    }
}
