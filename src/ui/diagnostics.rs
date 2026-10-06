//! Developer aids: the bar shown while a measurement pauses the game, and the
//! diagnostics drawer under the status line.

use super::{
    App,
    text::{file_size, number},
};
use crate::{
    theme::GAP_L,
    worker::{Command, Snapshot},
};
use eframe::egui;
use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

impl App {
    /// A bar across the window while a developer measurement pauses the
    /// game (`dev_pause`), with the time left and Resume now.
    pub(super) fn dev_pause_bar(&self, ui: &mut egui::Ui) {
        let Some(view) = self.worker.dev_pause.view() else {
            return;
        };
        let theme = self.theme();
        egui::Panel::bottom("dev-pause")
            .frame(
                egui::Frame::new()
                    .fill(theme.panel)
                    .inner_margin(egui::Margin::symmetric(GAP_L as i8, 6)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let left = view.ends_at.saturating_duration_since(Instant::now()).as_secs();
                    let text = if view.closed {
                        format!(
                            "Paused for a developer measurement, resumes in {}:{:02}",
                            left / 60,
                            left % 60
                        )
                    } else {
                        "Pausing for a developer measurement: the last creatures finish their trials"
                            .to_owned()
                    };
                    ui.colored_label(theme.warn, text);
                    if ui
                        .button("Resume now")
                        .on_hover_text("End the developer pause and keep evolving")
                        .clicked()
                    {
                        self.worker.dev_pause.resume_now();
                    }
                });
            });
        ui.ctx().request_repaint_after(Duration::from_millis(500));
    }
    /// The closed-by-default drawer with search and machine numbers, and the
    /// step-by-step run buttons developers use.
    pub(super) fn diagnostics(&self, ui: &mut egui::Ui, s: &Snapshot) {
        let mut frames: Vec<_> = self.frame_times.iter().copied().collect();
        frames.sort_by(f32::total_cmp);
        let p95 = frames.get(frames.len() * 95 / 100).copied().unwrap_or(0.);
        ui.small(format!(
            "{} · seed {} · {} · {} / {} evaluated · {} in confirmation",
            s.gpu,
            s.config.seed,
            if s.running { "running" } else { "paused" },
            number(s.completed),
            number(s.config.population),
            number(s.checking),
        ));
        ui.small(format!(
            "QD score {:.2} · {} behavior niches · {} topology reserves · next batch {}",
            s.qd_score,
            number(s.archive_cells),
            number(s.innovation_reserve_count),
            crate::qd::Emitter::ALL
                .into_iter()
                .zip(s.emitter_weights)
                .map(|(emitter, weight)| format!("{} {:.0}%", emitter.label(), weight * 100.0))
                .collect::<Vec<_>>()
                .join(", "),
        ));
        ui.small(format!(
            "Frame p95 {:.1} ms · end-to-end {:.0} creatures/s · GPU buffers {:.1} MiB · population {:.1} MiB · runs/ {}",
            p95 * 1000.,
            s.end_to_end,
            s.gpu_bytes as f64 / 1048576.,
            s.ram_bytes as f64 / 1048576.,
            file_size(self.runs_bytes)
        ));
        for (name, rate, count) in &s.engines {
            ui.small(format!("{name}: {rate:.0} creatures/s · {count} evaluated"));
        }
        let running = self.active();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!running, egui::Button::new("One generation").small())
                .clicked()
            {
                self.worker.pause.store(false, Ordering::Relaxed);
                self.worker.send(Command::Run {
                    continuous: false,
                    guided: false,
                });
            }
        });
    }
}
/// Total size of the files under a directory, ignoring unreadable entries.
pub(super) fn directory_bytes(root: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}
