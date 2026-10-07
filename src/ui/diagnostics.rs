//! Developer aids. `App::dev_pause_bar` draws the bar that shows while a
//! developer measurement pauses the game (`dev_pause`). `App::diagnostics`
//! fills the drawer under the status line that the Diagnostics button opens.
//! `directory_bytes` sizes the `runs/` directory for that drawer.

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
    /// A bar along the bottom of the window while a developer measurement
    /// pauses the game (`dev_pause`). First it says that the last creatures
    /// finish their trials. Once the engines are closed it counts down to the
    /// latest end of the pause. The Resume now button ends the pause early.
    /// Nothing is drawn when no pause is on.
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
        // Redraw twice a second so the countdown keeps moving.
        ui.ctx().request_repaint_after(Duration::from_millis(500));
    }
    /// The Diagnostics drawer under the status line, closed until the player
    /// opens it. The first line is about the run (GPU, seed, state, progress,
    /// trials in confirmation), the second about the search (QD score, niches,
    /// reserves, the emitter shares of the next batch) and the third about the
    /// machine (frame time, rate, memory, size of `runs/`). One line for each
    /// evaluation engine follows. The One generation button runs a single
    /// generation while evolution is not running.
    pub(super) fn diagnostics(&self, ui: &mut egui::Ui, s: &Snapshot) {
        // The 95th percentile of the recent frame times, in seconds.
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
/// Total size in bytes of the files under `root` and its subdirectories. An
/// entry it cannot read counts as 0, so a missing `root` gives 0. It does not
/// follow links: a link counts by its own size.
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
