//! The drawing methods of `App` for the generation metrics. `metrics` draws the
//! four tiles of the Overview tab. `trend` draws the chart of best and median
//! distance with the world changes marked on it, for the Overview and History
//! tabs. `histogram` draws the distance histogram of the History tab.

use super::{
    App,
    records::{live_point, live_record, row_in_world, world_records},
    text::{number, seconds_text},
};
use crate::{storage::Stats, theme::GAP_M, worker::EventKind};
use eframe::egui::{self, Align2, FontId, Pos2, RichText, Sense, Vec2};
use egui_plot::{Bar, BarChart, Legend, Line, Plot, Points, VLine};

/// A world change on the chart: the first generation in the new world, what
/// changed in words and whether Autochange made the change.
struct WorldMark {
    generation: u32,
    label: String,
    autochange: bool,
}
impl App {
    /// Draws the four tiles at the top of the Overview tab: best distance,
    /// generation, kinds of movement and rate. When no best distance is known
    /// yet, it draws one line of text in place of the tiles. It draws nothing
    /// before the first snapshot.
    pub(super) fn metrics(&self, ui: &mut egui::Ui) {
        let theme = self.theme();
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let history = &snapshot.history;
        // The best distance and the kinds of movement are the archive's
        // numbers now, so they change as results are absorbed. When the
        // archive has no elite, the newest history row of this world stands in.
        let best = if snapshot.live_best.is_finite() {
            snapshot.live_best
        } else if let Some(s) = row_in_world(snapshot) {
            s.best
        } else if !history.is_empty() {
            // The history has rows, but none from this world. The world has
            // just changed and no creature is kept in it yet.
            ui.label(
                RichText::new(
                    "Testing in the new world... The kept creatures are running again under the new rules. The best distance appears here when one is kept.",
                )
                .color(theme.muted),
            );
            return;
        } else {
            ui.label(
                RichText::new(
                    "The first generation is running. Its best creature appears here as soon as one is kept.",
                )
                .color(theme.muted),
            );
            return;
        };
        let cells = if snapshot.movement_cells > 0 {
            snapshot.movement_cells
        } else {
            row_in_world(snapshot).map_or(0, Stats::moves)
        };
        // The gain is the best distance now minus the best of the row 10 back.
        // It counts only when that row is from this world.
        let gain = history
            .len()
            .checked_sub(10)
            .filter(|&earlier| !history[earlier].config.physics_differs(&snapshot.config))
            .map(|earlier| best - history[earlier].best);
        let population = snapshot.config.population.max(1);
        let progress = generation_progress(
            snapshot.completed,
            population,
            snapshot.running,
            snapshot.end_to_end,
        );
        let trial = format!(
            "How far the best creature travels in its {:.0} s trial. Distance is the only score.",
            snapshot.config.duration
        );
        let rate = if snapshot.running && snapshot.end_to_end > 0.0 {
            format!("{} /s", number(snapshot.end_to_end as usize))
        } else {
            "—".to_owned()
        };
        let rate_note = if snapshot.running {
            "creatures scored per second".to_owned()
        } else {
            "Paused".to_owned()
        };
        // One entry per tile: the label, the number, the color of the number,
        // the line of detail and the hover text.
        let tiles = [
            (
                "Best distance",
                format!("{:.2} m", best),
                theme.accent,
                gain.map_or_else(
                    || "so far".to_owned(),
                    |gain| format!("{gain:+.2} m in the last 10 generations"),
                ),
                trial.as_str(),
            ),
            (
                "Generation",
                snapshot.generation.to_string(),
                theme.ink,
                progress,
                "Every generation tries a whole population of new creatures.",
            ),
            (
                "Kinds of movement",
                number(cells),
                theme.ink,
                "different ways of moving kept".to_owned(),
                "Evolution keeps the best creature for each way of moving: how much of the time it touches the ground, its stride rate, its height and how many feet it uses. Each way of moving keeps one creature for every body shape and size.",
            ),
            (
                "Rate",
                rate,
                theme.cold,
                rate_note,
                "How many creatures the machine tries each second, from breeding to score.",
            ),
        ];
        // The tiles share the row in equal widths. Each has its label in
        // capitals at the top left, a line of detail under it and the big
        // number on the right.
        let gap = GAP_M;
        let width = (ui.available_width() - gap * (tiles.len() - 1) as f32) / tiles.len() as f32;
        // A narrow tile shrinks the number and wraps the detail, and every
        // tile takes the height of the tallest.
        let value_size = (width / 8.5).clamp(26., 40.);
        let values: Vec<_> = tiles
            .iter()
            .map(|tile| {
                ui.painter().layout_no_wrap(
                    tile.1.clone(),
                    FontId::new(value_size, crate::assets::hud_bold()),
                    tile.2,
                )
            })
            .collect();
        // The detail wraps in the space that its number leaves.
        let notes: Vec<_> = tiles
            .iter()
            .zip(&values)
            .map(|(tile, value)| {
                ui.painter().layout(
                    tile.3.clone(),
                    FontId::proportional(14.),
                    theme.muted,
                    (width - value.size().x - 44.).max(60.),
                )
            })
            .collect();
        let note_height = notes.iter().map(|g| g.size().y).fold(0., f32::max);
        let tile_height = (value_size * 1.2).max(32. + note_height) + 14.;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for (((name, _, color, _, why), note), value) in
                tiles.into_iter().zip(notes).zip(values)
            {
                let (rect, response) =
                    ui.allocate_exact_size(Vec2::new(width, tile_height), Sense::hover());
                let painter = ui.painter_at(rect);
                crate::theme::plate(ui.painter(), rect.shrink(2.), theme, theme.card, false);
                crate::theme::caps_text(
                    &painter,
                    rect.left_top() + Vec2::new(14., 11.),
                    Align2::LEFT_TOP,
                    name,
                    if width < 280. { 12. } else { 13. },
                    theme.ink,
                );
                painter.galley(rect.left_top() + Vec2::new(14., 32.), note, theme.muted);
                crate::theme::glow_text(
                    &painter,
                    Pos2::new(rect.right() - 14., rect.center().y),
                    Align2::RIGHT_CENTER,
                    value.text(),
                    FontId::new(value_size, crate::assets::hud_bold()),
                    color,
                    false,
                );
                response.on_hover_text(why);
            }
        });
    }
    /// Draws a heading with the view buttons, then the chart of best and median
    /// distance per generation, `height` points tall. The running generation
    /// adds a point to each line. A vertical line marks each world change, and
    /// a point marks each record. It draws nothing before the first snapshot.
    pub(super) fn trend(&self, ui: &mut egui::Ui, height: f32) {
        let Some(s) = &self.snapshot else { return };
        let theme = self.theme();
        // The buttons set these for this frame. `reset` asks to fit the chart
        // again. `zoom` scales the view, and a value below one zooms in. `last`
        // is a number of newest generations to show.
        let mut reset = false;
        let mut zoom = 1.0f64;
        let mut last: Option<f64> = None;
        ui.horizontal(|ui| {
            crate::theme::heading(ui, "Best distance", theme)
                .on_hover_text("Drag to pan. Double-click or Reset view fits the chart again.");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("Reset view")
                    .on_hover_text("Fit every generation and distance again")
                    .clicked()
                {
                    reset = true;
                }
                if ui.small_button("+").on_hover_text("Zoom in").clicked() {
                    zoom = 0.8;
                }
                if ui.small_button("−").on_hover_text("Zoom out").clicked() {
                    zoom = 1.25;
                }
                if ui
                    .small_button("Last 10")
                    .on_hover_text("Show the ten newest generations only")
                    .clicked()
                {
                    last = Some(10.0);
                }
                if ui
                    .small_button("Last 50")
                    .on_hover_text("Show the fifty newest generations only")
                    .clicked()
                {
                    last = Some(50.0);
                }
            });
        });
        let mut plot = Plot::new("fitness_history")
            .height(height)
            .legend(
                Legend::default()
                    .position(egui_plot::Corner::LeftTop)
                    .text_style(egui::TextStyle::Small),
            )
            .x_axis_label("Generation")
            .y_axis_label("Distance (m)")
            .allow_scroll(false);
        if reset {
            plot = plot.reset();
        }
        let live = live_point(s);
        plot.show(ui, |plot| {
            if zoom != 1.0 {
                let bounds = plot.plot_bounds();
                plot.set_plot_bounds_x(scaled_range(bounds.range_x(), zoom));
                plot.set_plot_bounds_y(scaled_range(bounds.range_y(), zoom));
            }
            if let Some(generations) = last {
                // The x range ends just past the newest point. Only the y
                // range stays automatic.
                let end = live.map_or_else(
                    || s.history.last().map_or(1.0, |h| h.generation as f64 + 0.5),
                    |(generation, ..)| generation as f64 + 0.5,
                );
                plot.set_plot_bounds_x((end - generations).max(0.0)..=end);
                plot.set_auto_bounds(egui::Vec2b::new(false, true));
            }
            // Two lines: the best distance and the median over the fastest
            // elite of each way of moving. `i` is an index into
            // `storage::PERCENTILES`, where 28 is the 100th percentile and 14
            // is the 50th.
            for (i, name, color) in [(28, "Best", theme.accent), (14, "Median", theme.cold)] {
                let mut values: Vec<[f64; 2]> = s
                    .history
                    .iter()
                    .map(|h| [h.generation as f64, h.percentiles[i] as f64])
                    .collect();
                // The running generation's point moves as results arrive.
                if let Some((generation, best, median)) = live {
                    let value = if i == 28 { best } else { median };
                    if value.is_finite() {
                        values.push([generation as f64, value as f64]);
                    }
                }
                plot.line(Line::new(name, values).color(color).width(2.5));
            }
            // A vertical line and a short label at each world change. The line
            // stands between the last generation of the old world and the
            // first of the new one. The labels hang from `top`, which is the
            // highest best distance so far and at least 1.
            let top = s
                .history
                .iter()
                .map(|h| h.percentiles[28] as f64)
                .chain(live.map(|(_, best, _)| best as f64))
                .fold(1.0, f64::max);
            for mark in world_marks(&s.events, &s.history) {
                plot.vline(
                    VLine::new(
                        if mark.autochange {
                            "Autochange"
                        } else {
                            "World change"
                        },
                        mark.generation as f64 - 0.5,
                    )
                    .color(theme.warn)
                    .width(1.5),
                );
                plot.text(
                    egui_plot::Text::new(
                        "World change",
                        egui_plot::PlotPoint::new(mark.generation as f64 - 0.4, top),
                        RichText::new(short_label(&mark.label))
                            .small()
                            .color(theme.warn),
                    )
                    .anchor(egui::Align2::LEFT_TOP),
                );
            }
            // A point on the best line for each generation that set a record
            // in its world, and one for the running generation when its
            // champion sets one. A world change starts the records again.
            let mut records: Vec<[f64; 2]> = world_records(&s.history)
                .into_iter()
                .map(|(index, best, _)| [s.history[index].generation as f64, best as f64])
                .collect();
            if let Some(record) = live_record(s) {
                records.push([s.generation as f64, record.best as f64]);
            }
            if !records.is_empty() {
                plot.points(
                    Points::new("Record", records)
                        .color(theme.record)
                        .filled(true)
                        .radius(3.5),
                );
            }
        });
    }
    /// Draws the distances of one history row as a bar chart `height` points
    /// tall. Each bar counts the kept creatures with a distance in its range.
    /// A line of text replaces the chart when the row has no distances. When
    /// trials failed, a line under the chart counts them.
    pub(super) fn histogram(&self, ui: &mut egui::Ui, stats: &Stats, height: f32) {
        // The stored bins are one centimeter wide. The bars cover the lowest
        // bin to the top of the highest, about 40 of them, and none is
        // narrower than a stored bin.
        let (Some(low), Some(high)) = (
            stats.histogram.iter().map(|&(cm, _)| cm).min(),
            stats.histogram.iter().map(|&(cm, _)| cm).max(),
        ) else {
            ui.small("No distances recorded for this generation.");
            return;
        };
        let (low, high) = (low as f64 / 100.0, (high + 1) as f64 / 100.0);
        let width = ((high - low) / 40.0).max(0.01);
        let count = ((high - low) / width).ceil().max(1.0) as usize;
        let mut bins = vec![0u32; count];
        for &(cm, n) in &stats.histogram {
            // A stored bin goes to the bar that holds its middle.
            let value = (cm as f64 + 0.5) / 100.;
            let index = (((value - low) / width).floor() as usize).min(count - 1);
            bins[index] += n;
        }
        let bars = bins
            .iter()
            .enumerate()
            .map(|(i, &n)| Bar::new(low + (i as f64 + 0.5) * width, n as f64).width(width * 0.85))
            .collect();
        Plot::new("histogram")
            .height(height)
            .x_axis_label("Distance (m)")
            .allow_scroll(false)
            .show(ui, |plot| {
                plot.bar_chart(
                    BarChart::new("Creatures", bars)
                        .color(self.theme().accent.gamma_multiply(0.65)),
                );
            });
        if stats.failed > 0 {
            ui.small(format!(
                "{} failed trials are not shown",
                number(stats.failed)
            ));
        }
    }
}
/// Scales a plot range around its center. A factor below one zooms in.
fn scaled_range(
    range: std::ops::RangeInclusive<f64>,
    factor: f64,
) -> std::ops::RangeInclusive<f64> {
    let center = (range.start() + range.end()) / 2.0;
    let half = (range.end() - range.start()) / 2.0 * factor;
    (center - half)..=(center + half)
}
/// A chart label cut to fit beside its line. It keeps the text before the
/// first comma. If that is longer than 22 characters, it keeps the first 20
/// and adds "...".
fn short_label(label: &str) -> String {
    let first = label.split(',').next().unwrap_or(label);
    if first.chars().count() > 22 {
        format!("{}...", first.chars().take(20).collect::<String>())
    } else {
        first.to_owned()
    }
}
/// Every world change, oldest first. An event gives a change as soon as it
/// happens. A pair of neighboring history rows with different physics gives a
/// change that no event covers, such as one that the event log has dropped.
fn world_marks(events: &[crate::worker::Event], history: &[Stats]) -> Vec<WorldMark> {
    let mut marks: Vec<WorldMark> = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::World | EventKind::Autochange))
        .map(|e| WorldMark {
            generation: e.generation,
            // An event's text can go on after the change, for example with
            // the count of kept creatures that run again. The label keeps the
            // first sentence.
            label: e.text.split(". ").next().unwrap_or(&e.text).to_owned(),
            autochange: e.kind == EventKind::Autochange,
        })
        .collect();
    for pair in history.windows(2) {
        let generation = pair[1].generation;
        if pair[1].config.physics_differs(&pair[0].config)
            && !marks.iter().any(|m| m.generation == generation)
        {
            marks.push(WorldMark {
                generation,
                label: crate::worker::world_change_text(&pair[0].config, &pair[1].config)
                    .unwrap_or_else(|| "The world changed".into()),
                // The rule of the worker's events: Autochange is on and its
                // step moved.
                autochange: pair[1].config.autochange_step != pair[0].config.autochange_step
                    && pair[1].config.autochange > 0,
            });
        }
    }
    marks.sort_by_key(|m| m.generation);
    marks
}
/// The Generation tile's second line. It says "Paused" when evolution is
/// stopped. While it runs, it says "Finishing the generation" once every
/// creature has a result. Before that it gives the percent done, rounded down,
/// and the time left at `rate` creatures per second when `rate` is above zero.
fn generation_progress(completed: usize, population: usize, running: bool, rate: f64) -> String {
    let done = completed.min(population);
    if !running {
        return "Paused".to_owned();
    }
    if done >= population {
        return "Finishing the generation".to_owned();
    }
    let percent = (done as f64 / population as f64 * 100.0).floor();
    if rate > 0.0 {
        let left = (population - done) as f64 / rate;
        format!("{percent:.0}% done · about {} left", seconds_text(left))
    } else {
        format!("{percent:.0}% done")
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zoom_scales_a_plot_range_around_its_center() {
        assert_eq!(scaled_range(10.0..=20.0, 0.5), 12.5..=17.5);
        assert_eq!(scaled_range(10.0..=20.0, 2.0), 5.0..=25.0);
        assert_eq!(scaled_range(-4.0..=4.0, 1.0), -4.0..=4.0);
    }
}
