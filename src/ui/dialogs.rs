//! Draws the File menu's windows. One lists the saves in runs/ to open, one
//! starts a new experiment, one asks before a save replaces a file, and one
//! takes the path of a file to save, export or open. The menu in `header` and
//! the export buttons only set a field of `App`, and `App::dialogs` draws the
//! window that field asks for in every frame. The module also keeps the save
//! state that the top bar shows, from the events the worker logs.

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

/// One .evo file that File > Open lists: where it is, when it changed, how big
/// it is and what its first bytes say.
pub(super) struct SaveEntry {
    path: PathBuf,
    modified: Option<std::time::SystemTime>,
    bytes: u64,
    /// The generation and settings at the start of the file. None for another
    /// format, another physics version or a file that cannot be read. The list
    /// then says "older format".
    summary: Option<crate::storage::SaveSummary>,
}
impl App {
    /// Opens the path window for `mode` and fills its path field with the usual
    /// file for it. `mode` is the title of the window and the text of its
    /// button, and it picks what the button does: "Save experiment", "Export
    /// CSV", "Open creature JSON", "Export creature JSON" or "Export creature
    /// GIF". The two creature exports name the file after the creature on
    /// screen, with its species name, distance and id, when there is one.
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
    /// Reads the worker's events that came since the last call and updates the
    /// save state. A save or an open marks the experiment saved or opened at
    /// the generation of the event. An autosave counts as a save. A new
    /// experiment marks it unsaved. A save event ends the Saving state. So does
    /// an error from the worker, because a save that fails logs no event.
    pub(super) fn absorb_events(&mut self, snapshot: &Snapshot) {
        // A new game or a load starts a new log, which is read from its start.
        // The worker keeps only its newest 200 events, so once the log is full
        // its length stops growing and the events after that are not seen.
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
    /// Asks the worker to save to `path`. When the file exists and `confirmed`
    /// is false, it sets `overwrite` instead, and the player is asked first.
    fn save_to(&mut self, path: PathBuf, confirmed: bool) {
        if !confirmed && path.exists() {
            self.overwrite = Some(path);
            return;
        }
        self.saving = Some(Instant::now());
        self.worker.send(Command::Save(path));
    }
    /// What the top bar says about the save, and whether a save is being
    /// written, which makes the bar show a spinner. While a save runs the text
    /// says so. Otherwise it tells when the experiment was saved or opened and
    /// how many generations have run since, or that it is not saved.
    pub(super) fn save_state(&self) -> (String, bool) {
        if self.saving.is_some() {
            return ("Saving…".to_owned(), true);
        }
        match self.saved {
            Some((at, generation, opened)) => {
                let now = self.snapshot.as_ref().map_or(generation, |s| s.generation);
                let since = now.saturating_sub(generation);
                let age = seconds_text(at.elapsed().as_secs_f64());
                let verb = if opened { "Opened" } else { "Saved" };
                (
                    if since > 0 {
                        format!("{verb} {age} ago · {since} generations since")
                    } else {
                        format!("{verb} {age} ago")
                    },
                    false,
                )
            }
            None => ("Not saved".to_owned(), false),
        }
    }
    /// Pauses the game and asks the worker to load the save at `path`, so the
    /// game starts paused on it. It also sets `initial`, so `config` takes its
    /// value from the first snapshot of the next game (`absorb_snapshot`).
    fn open_experiment(&mut self, path: PathBuf) {
        self.pause();
        self.worker.send(Command::Load(path));
        self.initial = true;
    }
    /// File > Open: the saves in runs/, newest first, each with an Open
    /// button, and a path field for a file anywhere else. The window shows
    /// while `open_list` is set. It closes when an Open button is pressed or
    /// the player cancels.
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
    /// Draws the File windows that are open in this frame: the list of saves,
    /// the check before a save replaces a file, the new experiment window and
    /// the path window of `file_mode`.
    pub(super) fn dialogs(&mut self, ctx: &egui::Context) {
        let theme = self.theme();
        self.open_window(ctx);
        // The check before a save replaces a file. `save_to` asks for it when
        // the file exists.
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
        // A new experiment uses `config`. Create population pauses the game
        // and sends `config` to the worker as a new experiment.
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
        // The path window. Its button does what `mode` says and closes the
        // window.
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
                                "Export CSV" => self.worker.send(Command::Export(path)),
                                "Export creature JSON" => {
                                    let result = (|| -> anyhow::Result<()> {
                                        let playback = self.playback.as_ref().ok_or_else(|| {
                                            anyhow::anyhow!("No creature is selected to export")
                                        })?;
                                        if let Some(parent) = path.parent() {
                                            std::fs::create_dir_all(parent)?;
                                        }
                                        serde_json::to_writer_pretty(
                                            std::fs::File::create(&path)?,
                                            &playback.creature,
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
                                    // The creature replays in the world of the
                                    // game now.
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
/// Checks a creature read from a JSON file before it is replayed, and gives a
/// short reason when it fails. The body must be a connected tree with one bone
/// per extra node. The numbers of a node and a bone must be finite, and so must
/// the lengths, rhythm and stiffness of a muscle. A muscle must sit on bones
/// that exist. The check also puts the bones in parent-first order
/// (`canonicalize_bone_order`), which changes `creature`.
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
/// bytes say about it. A file whose time is unknown comes last. A directory
/// that cannot be read gives an empty list.
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
        // A bone that ends on a node the body does not have is not a tree.
        loaded.bones[0].b = 9;
        assert!(imported_creature(&mut loaded).is_err());
    }
}
