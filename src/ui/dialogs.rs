//! The File menu's dialogs: save with its overwrite check, open from runs/, the
//! new experiment window, the statistics CSV and the creature JSON and GIF files,
//! and the save state the top bar shows.

use super::{
    App, Tab,
    controls::world_summary,
    export::export_creature_gif,
    text::{ago, file_size, number, seconds_text, species_name},
};
use crate::{
    config::Config,
    evolution::Creature,
    worker::{Command, EventKind, Snapshot},
};
use eframe::egui::{self, Align2, RichText, Vec2};
use std::{path::PathBuf, time::Instant};

/// One save in runs/, for File > Open.
pub(super) struct SaveEntry {
    path: PathBuf,
    modified: Option<std::time::SystemTime>,
    bytes: u64,
    summary: Option<crate::storage::SaveSummary>,
}
impl App {
    pub(super) fn file(&mut self, mode: &'static str) {
        self.file_mode = Some(mode);
        self.file_path = match mode {
            "Export CSV" => "runs/statistics.csv".to_owned(),
            "Open creature JSON" => "runs/creature.json".to_owned(),
            "Export creature JSON" | "Export creature GIF" => {
                let extension = if mode.ends_with("GIF") { "gif" } else { "json" };
                self.playback.as_ref().map_or_else(
                    || format!("runs/creature.{extension}"),
                    |p| {
                        format!(
                            "runs/{}-{:.1}m-{}.{extension}",
                            species_name(&p.creature).replace(' ', "-"),
                            p.distance,
                            p.creature.id
                        )
                    },
                )
            }
            _ => "runs/experiment.evo".to_owned(),
        };
    }
    /// Follows the worker's event log for the save state: a save or an open
    /// marks the experiment saved, a new game marks it unsaved.
    pub(super) fn absorb_events(&mut self, snapshot: &Snapshot) {
        if self.events_seen.0 != snapshot.epoch || snapshot.events.len() < self.events_seen.1 {
            self.events_seen = (snapshot.epoch, 0);
        }
        for event in &snapshot.events[self.events_seen.1..] {
            match event.kind {
                EventKind::Saved | EventKind::Opened => {
                    self.saved = Some((
                        Instant::now(),
                        event.generation,
                        event.kind == EventKind::Opened,
                    ));
                    if event.kind == EventKind::Saved {
                        self.saving = None;
                    }
                }
                EventKind::Started => self.saved = None,
                _ => {}
            }
        }
        self.events_seen.1 = snapshot.events.len();
        if snapshot.error.is_some() {
            self.saving = None;
        }
    }
    /// Asks the worker to save, or first asks the player when the file exists.
    fn save_to(&mut self, path: PathBuf, confirmed: bool) {
        if !confirmed && path.exists() {
            self.overwrite = Some(path);
            return;
        }
        self.saving = Some(Instant::now());
        self.worker.send(Command::Save(path));
    }
    /// Save state for the top bar: saving, saved how long ago, or not saved.
    pub(super) fn save_state(&self) -> (String, bool) {
        if self.saving.is_some() {
            return ("Saving…".to_owned(), true);
        }
        match self.saved {
            Some((at, generation, opened)) => {
                let now = self.snapshot.as_ref().map_or(generation, |s| s.generation);
                let since = now.saturating_sub(generation);
                let ago = seconds_text(at.elapsed().as_secs_f64());
                let verb = if opened { "Opened" } else { "Saved" };
                (
                    if since > 0 {
                        format!("{verb} {ago} ago · {since} generations since")
                    } else {
                        format!("{verb} {ago} ago")
                    },
                    false,
                )
            }
            None => ("Not saved".to_owned(), false),
        }
    }
    /// Opens a saved experiment; the game starts paused on it.
    fn open_experiment(&mut self, path: PathBuf) {
        self.pause();
        self.worker.send(Command::Load(path));
        self.initial = true;
    }
    /// File > Open: the saves in runs/, newest first, and a path field for
    /// a file anywhere else.
    fn open_window(&mut self, ctx: &egui::Context) {
        let Some(saves) = &self.open_list else {
            return;
        };
        let theme = self.theme();
        let mut chosen: Option<PathBuf> = None;
        let mut close = false;
        egui::Window::new("Open a saved experiment")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_width(560.);
                if saves.is_empty() {
                    ui.label(RichText::new("No saves in runs/ yet.").color(theme.muted));
                }
                egui::ScrollArea::vertical()
                    .max_height(360.)
                    .show(ui, |ui| {
                        egui::Grid::new("saves")
                            .num_columns(4)
                            .spacing([14., 8.])
                            .striped(true)
                            .show(ui, |ui| {
                                for save in saves {
                                    let name =
                                        save.path.file_name().map_or_else(String::new, |n| {
                                            n.to_string_lossy().into_owned()
                                        });
                                    ui.vertical(|ui| {
                                        ui.label(RichText::new(name).strong());
                                        if let Some(summary) = &save.summary {
                                            ui.label(
                                                RichText::new(world_summary(&summary.config))
                                                    .small()
                                                    .color(theme.muted),
                                            );
                                        }
                                    });
                                    ui.label(save.summary.as_ref().map_or_else(
                                        || "older format".to_owned(),
                                        |summary| {
                                            format!(
                                                "generation {} · {} creatures",
                                                summary.generation,
                                                number(summary.config.population)
                                            )
                                        },
                                    ));
                                    ui.label(
                                        RichText::new(format!(
                                            "{} · {}",
                                            ago(save.modified),
                                            file_size(save.bytes)
                                        ))
                                        .small()
                                        .color(theme.muted),
                                    );
                                    if ui.button("Open").clicked() {
                                        chosen = Some(save.path.clone());
                                    }
                                    ui.end_row();
                                }
                            });
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Another file");
                    ui.add(egui::TextEdit::singleline(&mut self.file_path).desired_width(300.));
                    if ui.button("Open").clicked() {
                        chosen = Some(PathBuf::from(&self.file_path));
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
        if let Some(path) = chosen {
            self.open_experiment(path);
            close = true;
        }
        if close {
            self.open_list = None;
        }
    }
    pub(super) fn dialogs(&mut self, ctx: &egui::Context) {
        let theme = self.theme();
        self.open_window(ctx);
        if let Some(path) = self.overwrite.clone() {
            let mut answer = None;
            egui::Window::new("Replace the save?")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "{} already exists. Replace it with this experiment?",
                        path.display()
                    ));
                    ui.horizontal(|ui| {
                        if ui.button("Replace").clicked() {
                            answer = Some(true);
                        }
                        if ui.button("Cancel").clicked() {
                            answer = Some(false);
                        }
                    });
                });
            match answer {
                Some(true) => {
                    self.overwrite = None;
                    self.save_to(path, true);
                }
                Some(false) => self.overwrite = None,
                None => {}
            }
        }
        if self.new_dialog {
            egui::Window::new("Start a new experiment")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "Start over with {} new creatures in this world: {}.",
                        number(self.config.population),
                        world_summary(&self.config)
                    ));
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.config.random_seed, "New random seed");
                        if !self.config.random_seed {
                            ui.label("Seed");
                            ui.add(egui::DragValue::new(&mut self.config.seed));
                        }
                    });
                    ui.label("Save the current experiment first if you want to resume it later.");
                    ui.horizontal(|ui| {
                        if ui.button("Save current").clicked() {
                            self.file("Save experiment");
                            self.new_dialog = false;
                        }
                        if ui
                            .add_enabled(
                                self.config.validate().is_ok(),
                                egui::Button::new(
                                    RichText::new("Create population").color(theme.go_text),
                                )
                                .fill(theme.go_fill),
                            )
                            .clicked()
                        {
                            self.pause();
                            self.worker.send(Command::New(self.config.clone()));
                            self.initial = true;
                            self.new_dialog = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.new_dialog = false;
                        }
                    });
                });
        }
        if let Some(mode) = self.file_mode {
            egui::Window::new(mode)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("File path");
                    ui.add(egui::TextEdit::singleline(&mut self.file_path).desired_width(430.));
                    ui.horizontal(|ui| {
                        if ui.button(mode).clicked() {
                            let path = PathBuf::from(&self.file_path);
                            match mode {
                                "Save experiment" => self.save_to(path, false),
                                "Open experiment" => self.open_experiment(path),
                                "Export CSV" => self.worker.send(Command::Export(path)),
                                "Export creature JSON" => {
                                    let creature =
                                        self.playback.as_ref().map(|p| p.creature.clone());
                                    let result = (|| -> anyhow::Result<()> {
                                        let creature = creature.as_ref().ok_or_else(|| {
                                            anyhow::anyhow!("No creature is selected to export")
                                        })?;
                                        if let Some(parent) = path.parent() {
                                            std::fs::create_dir_all(parent)?;
                                        }
                                        serde_json::to_writer_pretty(
                                            std::fs::File::create(&path)?,
                                            creature,
                                        )?;
                                        Ok(())
                                    })();
                                    self.message = Some(result.map_or_else(
                                        |e| e.to_string(),
                                        |_| format!("Creature saved to {}", path.display()),
                                    ));
                                }
                                "Export creature GIF" => {
                                    let result = (|| -> anyhow::Result<usize> {
                                        let playback = self.playback.as_ref().ok_or_else(|| {
                                            anyhow::anyhow!("No creature is selected to export")
                                        })?;
                                        if let Some(parent) = path.parent() {
                                            std::fs::create_dir_all(parent)?;
                                        }
                                        export_creature_gif(playback, &path)
                                    })();
                                    self.message = Some(result.map_or_else(
                                        |e| format!("GIF export failed: {e}"),
                                        |frames| {
                                            format!(
                                                "GIF saved to {} ({frames} frames)",
                                                path.display()
                                            )
                                        },
                                    ));
                                }
                                "Open creature JSON" => {
                                    let config = self
                                        .snapshot
                                        .as_ref()
                                        .map_or_else(Config::default, |s| s.config.clone());
                                    let result = (|| -> anyhow::Result<Creature> {
                                        let mut creature: Creature =
                                            serde_json::from_reader(std::fs::File::open(&path)?)?;
                                        imported_creature(&mut creature).map_err(|why| {
                                            anyhow::anyhow!("{}: {why}", path.display())
                                        })?;
                                        Ok(creature)
                                    })();
                                    match result {
                                        Ok(creature) => {
                                            self.select(creature, config);
                                            self.tab = Tab::Overview;
                                            self.message =
                                                Some(format!("Opened {}", path.display()));
                                        }
                                        Err(e) => self.message = Some(e.to_string()),
                                    }
                                }
                                _ => {}
                            }
                            self.file_mode = None;
                        }
                        if ui.button("Cancel").clicked() {
                            self.file_mode = None;
                        }
                    });
                });
        }
    }
}
/// Validates an imported creature JSON before it reaches the replay engine.
/// Rejects bodies the physics cannot step, with a short reason.
fn imported_creature(creature: &mut Creature) -> Result<(), String> {
    let nodes = creature.nodes.len();
    if !(3..=64).contains(&nodes) || creature.bones.len() + 1 != nodes {
        return Err("expected a connected body with one bone per extra node".into());
    }
    if !crate::evolution::canonicalize_bone_order(creature) {
        return Err("the body plan is not a connected tree".into());
    }
    if creature.nodes.iter().any(|n| {
        ![n.x, n.y, n.diameter, n.friction]
            .iter()
            .all(|v| v.is_finite())
    }) {
        return Err("the file has non-finite node values".into());
    }
    if creature.bones.iter().any(|b| {
        ![
            b.rest_length,
            b.min_angle,
            b.max_angle,
            b.organ_mass,
            b.organ_at,
        ]
        .iter()
        .all(|v| v.is_finite())
    }) {
        return Err("the file has non-finite bone values".into());
    }
    if creature.muscles.iter().any(|m| {
        m.bone_a as usize >= creature.bones.len()
            || m.bone_b as usize >= creature.bones.len()
            || ![m.short, m.long, m.period, m.phase, m.duty, m.stiffness]
                .iter()
                .all(|v| v.is_finite())
    }) {
        return Err("the file has invalid muscles".into());
    }
    Ok(())
}
/// Every .evo file directly in `dir`, newest first, with what its first
/// bytes say about it.
pub(super) fn list_saves(dir: &std::path::Path) -> Vec<SaveEntry> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut saves: Vec<SaveEntry> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "evo"))
        .map(|path| {
            let metadata = std::fs::metadata(&path).ok();
            SaveEntry {
                modified: metadata.as_ref().and_then(|m| m.modified().ok()),
                bytes: metadata.map_or(0, |m| m.len()),
                summary: crate::storage::summary(&path),
                path,
            }
        })
        .collect();
    saves.sort_by_key(|save| std::cmp::Reverse(save.modified));
    saves
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::test_creature;
    #[test]
    fn imported_creatures_round_trip_through_json() {
        let mut creature = test_creature();
        assert!(imported_creature(&mut creature).is_ok());
        let json = serde_json::to_string(&creature).unwrap();
        let mut loaded: Creature = serde_json::from_str(&json).unwrap();
        assert!(imported_creature(&mut loaded).is_ok());
        loaded.bones[0].b = 9;
        assert!(imported_creature(&mut loaded).is_err());
    }
}
