//! The top bar (the game's name, Evolve and the File and View menus) and the
//! Help window it opens.

use super::{AUTOSAVE_INTERVAL, App, GAP_S, dialogs::list_saves, text::number};
use crate::{theme::GAP_L, worker::Command};
use eframe::egui::{self, Align2, FontId, RichText, Sense, Stroke, Vec2};
use std::time::{Duration, Instant};

impl App {
    pub(super) fn top(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.add_space(GAP_S);
            let (logo, _) = ui.allocate_exact_size(Vec2::splat(28.), Sense::hover());
            // Three joined nodes, the smallest creature, lit in amber.
            let points = [
                logo.center_top() + Vec2::new(0., 4.),
                logo.left_bottom() + Vec2::new(4., -4.),
                logo.right_bottom() + Vec2::new(-4., -4.),
            ];
            let painter = ui.painter();
            let amber = crate::theme::poster::MUSTARD;
            for i in 0..3 {
                painter.line_segment(
                    [points[i], points[(i + 1) % 3]],
                    Stroke::new(3., amber),
                );
            }
            for point in points {
                painter.circle_filled(point, 4.5, amber);
                painter.circle_filled(point, 1.8, crate::theme::poster::WOOD_DARK);
            }
            // The game's name (owner, 2026-10-03), in chunky condensed letters.
            let mut job = egui::text::LayoutJob::default();
            job.append(
                "exploraMove",
                0.,
                egui::TextFormat {
                    font_id: FontId::new(32., crate::assets::hud_bold()),
                    color: crate::theme::poster::CREAM,
                    extra_letter_spacing: 1.,
                    ..Default::default()
                },
            );
            let title = ui.painter().layout_job(job);
            let (title_rect, _) = ui.allocate_exact_size(title.size(), Sense::hover());
            ui.painter()
                .galley(title_rect.min, title, crate::theme::poster::CREAM);
            ui.add_space(GAP_L);
            let running = self.active();
            let (text, fill, ink, why) = if running {
                (
                    "Pause evolution  (Space)",
                    theme.stop_fill,
                    theme.stop_text,
                    "Stop after the work in flight. The replay keeps playing.",
                )
            } else {
                (
                    "Evolve  (Space)",
                    theme.go_fill,
                    theme.go_text,
                    "Run generation after generation until you pause.",
                )
            };
            if ui
                .add(
                    egui::Button::new(RichText::new(text).size(18.).strong().color(ink))
                        .fill(fill)
                        .min_size(Vec2::new(210., 40.)),
                )
                .on_hover_text(why)
                .clicked()
            {
                if running {
                    self.pause();
                } else {
                    self.run(true, false);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("Help")
                    .on_hover_text("Shortcuts and what each tab shows · F1")
                    .clicked()
                {
                    self.show_help = !self.show_help;
                }
                ui.menu_button("View", |ui| {
                    // The new scale applies when the drag ends: scaling the
                    // UI under the pointer mid-drag moves the slider and threw
                    // it to the other end.
                    let slider = ui.add(
                        egui::Slider::new(&mut self.ui_scale, 0.75..=1.6).text("UI scale"),
                    );
                    if slider.drag_stopped() || (slider.changed() && !slider.dragged()) {
                        ui.ctx().set_zoom_factor(self.ui_scale);
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui.button("New experiment…").clicked() {
                        self.new_dialog = true;
                        ui.close();
                    }
                    if ui.button("Open…").clicked() {
                        self.open_list = Some(list_saves(std::path::Path::new("runs")));
                        self.file_path = "runs/experiment.evo".to_owned();
                        ui.close();
                    }
                    if ui.button("Save…  Ctrl+S").clicked() {
                        self.file("Save experiment");
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Open creature JSON…").clicked() {
                        self.file("Open creature JSON");
                        ui.close();
                    }
                    if ui.button("Export statistics CSV…").clicked() {
                        self.file("Export CSV");
                        ui.close();
                    }
                    if ui.button("Screenshot").clicked() {
                        self.screenshot_pending = true;
                        self.screenshot_waiting = true;
                        self.message = Some("Taking a screenshot…".into());
                        ui.close();
                    }
                    ui.separator();
                    let mut autosave = self.config.checkpoint_interval > 0;
                    if ui
                        .checkbox(
                            &mut autosave,
                            format!("Autosave every {AUTOSAVE_INTERVAL} generations"),
                        )
                        .on_hover_text("Writes runs/seed-<seed>-auto.evo in the background and keeps the three newest.")
                        .changed()
                    {
                        self.config.checkpoint_interval = if autosave { AUTOSAVE_INTERVAL } else { 0 };
                        self.worker.send(Command::Configure(self.config.clone()));
                        self.config_sent = Some(Instant::now());
                    }
                });
                let (state, busy) = self.save_state();
                ui.label(RichText::new(state).color(if busy {
                    crate::theme::poster::MUSTARD
                } else {
                    crate::theme::poster::PAPER
                }))
                .on_hover_text("File > Save writes the experiment to runs/. The game writes nothing on its own unless autosave is on.");
                if busy {
                    ui.spinner();
                    ui.ctx().request_repaint_after(Duration::from_millis(250));
                }
                ui.separator();
                if let Some(s) = &self.snapshot {
                    ui.label(
                        RichText::new(format!(
                            "{} creatures · {:.0} s trials",
                            number(s.config.population),
                            s.config.duration
                        ))
                        .color(crate::theme::poster::PAPER),
                    );
                }
            });
        });
    }
    /// Keyboard shortcuts and what each tab shows.
    pub(super) fn help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let theme = self.theme();
        egui::Window::new("Help")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_max_width(460.);
                ui.heading("Shortcuts");
                egui::Grid::new("help_shortcuts")
                    .num_columns(2)
                    .spacing([18., 6.])
                    .show(ui, |ui| {
                        for (keys, action) in [
                            (
                                "1 to 5",
                                "Overview · Ways of moving · History · Race · Lineage",
                            ),
                            ("Space", "Evolve, or pause evolution"),
                            ("K or a click on the replay", "Play or pause the replay"),
                            ("← / →", "Step the replay one frame"),
                            ("F1 or ?", "Toggle this help"),
                            ("Ctrl+S", "Save the experiment"),
                            ("Drag / scroll", "Pan and zoom the viewport"),
                        ] {
                            ui.label(RichText::new(keys).strong().color(theme.accent));
                            ui.label(action);
                            ui.end_row();
                        }
                    });
                ui.separator();
                if ui.button("How evolution works").clicked() {
                    self.schematic_open = true;
                }
                ui.separator();
                ui.heading("Tabs");
                for (name, why) in [
                    (
                        "Overview",
                        "The champion's replay (or the creature you picked), its playback controls, lineage and the best distance over time.",
                    ),
                    (
                        "Ways of moving",
                        "The best creature for every way of moving, as cards or as a map. Click a creature or a map cell to replay it.",
                    ),
                    (
                        "History",
                        "The best and median distance over time with world changes marked, every record with a replay, the mix of body types, and the distances of one generation.",
                    ),
                    (
                        "Race",
                        "The five fastest archived creatures run their trials side by side with live standings.",
                    ),
                    (
                        "Lineage",
                        "Ancestors of the creature on screen with thumbnails, distance gains and body plan changes.",
                    ),
                ] {
                    ui.label(RichText::new(name).strong());
                    ui.label(RichText::new(why).color(theme.muted));
                    ui.add_space(GAP_S);
                }
            });
    }
}
