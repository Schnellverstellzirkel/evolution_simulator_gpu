//! The History tab. It has the generation slider, the trend chart, the list of
//! records and the body types through the generations. Below those it shows the
//! picked generation: its diversity, its distances, its body types and its
//! worst, median and best creature. `central_panel` in `ui.rs` calls
//! `App::history`.

use super::{
    App, Tab,
    controls::world_summary,
    records::{live_record, world_records},
    scene::thumbnail,
    text::{number, species_name},
    widgets::{color_dot, species_color},
};
use crate::{storage::Stats, theme::GAP_M};
use eframe::egui::{self, Pos2, Rect, RichText, Sense, Stroke, Vec2};

impl App {
    /// The Records list, newest first. A row has the generation, the distance,
    /// the species name, the world the record was set in and a Replay button.
    /// The first row is the running generation's record, when `live_record`
    /// finds one. Records count again after each world change, like the feed
    /// and the chart.
    fn records_list(&mut self, ui: &mut egui::Ui) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let theme = self.theme();
        let records = world_records(&snapshot.history);
        let live = live_record(snapshot);
        crate::theme::heading(ui, "Records", theme);
        if records.is_empty() && live.is_none() {
            ui.label(
                RichText::new("No records yet. The first best creature lands here.")
                    .small()
                    .color(theme.muted),
            );
            return;
        }
        let mut chosen = None;
        let mut chosen_live = false;
        egui::ScrollArea::vertical()
            .id_salt("records_list")
            .max_height(200.)
            .show(ui, |ui| {
                egui::Grid::new("records_grid")
                    .num_columns(4)
                    .spacing([14., 4.])
                    .striped(true)
                    .show(ui, |ui| {
                        if let Some(record) = &live {
                            ui.label(
                                RichText::new(format!("Gen {}", snapshot.generation))
                                    .small()
                                    .color(theme.muted),
                            );
                            ui.label(RichText::new(format!("{:.2} m", record.best)).strong());
                            let name = snapshot
                                .champion
                                .as_ref()
                                .map(|champion| species_name(&champion.0))
                                .unwrap_or_default();
                            ui.label(
                                RichText::new(format!(
                                    "{name} · {}{}",
                                    world_summary(&snapshot.config),
                                    if record.first_in_world {
                                        " (new world)"
                                    } else {
                                        ""
                                    }
                                ))
                                .small()
                                .color(theme.muted),
                            );
                            if ui.small_button("Replay").clicked() {
                                chosen_live = true;
                            }
                            ui.end_row();
                        }
                        for &(index, best, first) in records.iter().rev() {
                            let stats = &snapshot.history[index];
                            ui.label(
                                RichText::new(format!("Gen {}", stats.generation))
                                    .small()
                                    .color(theme.muted),
                            );
                            ui.label(RichText::new(format!("{best:.2} m")).strong());
                            ui.label(
                                RichText::new(format!(
                                    "{} · {}{}",
                                    stats
                                        .representatives
                                        .last()
                                        .map(species_name)
                                        .unwrap_or_default(),
                                    world_summary(&stats.config),
                                    if first && index > 0 {
                                        " (new world)"
                                    } else {
                                        ""
                                    }
                                ))
                                .small()
                                .color(theme.muted),
                            );
                            if ui.small_button("Replay").clicked() {
                                chosen = Some(index);
                            }
                            ui.end_row();
                        }
                    });
            });
        if chosen_live {
            self.replay_champion();
        } else if let Some(index) = chosen {
            self.replay_history_holder(index);
        }
    }
    /// The body plans, the effective clades and the median plan age of the
    /// global archive at row `index` of the history, as one line of text. A
    /// small chart beside it traces the plans and the clades over at most 100
    /// rows ending at `index`. Each trace is scaled to its own highest value.
    fn diversity_meter(&self, ui: &mut egui::Ui, index: usize) {
        let theme = self.theme();
        let Some(snapshot) = self.snapshot.as_ref() else {
            return;
        };
        let history = &snapshot.history;
        let Some(now) = history.get(index) else {
            return;
        };
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "{} body plans · {:.1} effective clades · plans live {:.0} generations (median)",
                    number(now.plans),
                    now.clades,
                    now.plan_age
                ))
                .strong(),
            )
            .on_hover_text(
                "Body plans: different skeletons among the kept creatures. Effective clades: how many lineages share the archive, counting a lineage by its share (exp of the Shannon entropy).",
            );
            let start = index.saturating_sub(99);
            let window = &history[start..=index];
            let (rect, _) = ui.allocate_exact_size(Vec2::new(200., 28.), Sense::hover());
            let painter = ui.painter();
            painter.rect_filled(rect, 3., theme.card);
            for (values, color) in [
                (window.iter().map(|h| h.plans as f32).collect::<Vec<_>>(), theme.accent),
                (window.iter().map(|h| h.clades).collect::<Vec<_>>(), theme.cold),
            ] {
                let top = values.iter().copied().fold(1.0f32, f32::max);
                let n = values.len().max(2) - 1;
                let points: Vec<Pos2> = values
                    .iter()
                    .enumerate()
                    .map(|(k, v)| {
                        egui::pos2(
                            rect.left() + rect.width() * k as f32 / n as f32,
                            rect.bottom() - 2. - (rect.height() - 4.) * v / top,
                        )
                    })
                    .collect();
                painter.add(egui::Shape::line(points, Stroke::new(1.5, color)));
            }
        });
    }
    /// The chart of the body types through the generations: one column per
    /// generation, and in each column one band per body type, as tall as that
    /// type's share of the row's `species_total`. With more generations than
    /// points of width, each column shows the first generation of a group. A
    /// line marks the picked generation, and a click on the chart picks the
    /// generation under the pointer.
    fn species_history(&mut self, ui: &mut egui::Ui) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        if snapshot.history.is_empty() {
            return;
        }
        let theme = self.theme();
        ui.horizontal_wrapped(|ui| {
            crate::theme::heading(ui, "Body types through generations", theme);
            ui.label(
                RichText::new(
                    "each color is one count of nodes and muscles, named in the list below",
                )
                .small()
                .color(theme.muted),
            );
        });
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 80.), Sense::click());
        let painter = ui.painter_at(rect);
        let history = &snapshot.history;
        let stride = history.len().div_ceil(rect.width().max(1.) as usize).max(1);
        for i in (0..history.len()).step_by(stride) {
            let h = &history[i];
            let body_count = species_total(h);
            let x = rect.left() + rect.width() * i as f32 / history.len() as f32;
            let right = rect.left()
                + rect.width() * (i + stride).min(history.len()) as f32 / history.len() as f32;
            let mut y = rect.bottom();
            for &(nodes, muscles, count) in &h.species {
                let height = rect.height() * count as f32 / body_count.max(1) as f32;
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(x, y - height), Pos2::new(right, y)),
                    0,
                    species_color(nodes, muscles),
                );
                y -= height;
            }
        }
        let x =
            rect.left() + rect.width() * (self.history_index as f32 + 0.5) / history.len() as f32;
        painter.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(2., theme.ink),
        );
        if let Some(pos) = response.hover_pos() {
            let index = (((pos.x - rect.left()) / rect.width()) * history.len() as f32) as usize;
            let index = index.min(history.len() - 1);
            if response.clicked() {
                self.history_latest = false;
                self.history_index = index;
            }
            response.on_hover_text(format!(
                "Generation {} · click to inspect",
                history[index].generation
            ));
        }
    }
    /// The whole tab. With no history yet it shows a short note. Otherwise the
    /// slider picks a row of the history, and Follow latest keeps it on the
    /// newest row. A replay started here selects the creature and switches to
    /// the Overview tab.
    pub(super) fn history(&mut self, ui: &mut egui::Ui) {
        let Some(s) = &self.snapshot else { return };
        let theme = self.theme();
        let len = s.history.len();
        if len == 0 {
            ui.heading("A history waiting to happen");
            ui.label("Run your first generation to build distance curves and creature replays.");
            return;
        }
        ui.horizontal(|ui| {
            ui.heading("History");
            ui.checkbox(&mut self.history_latest, "Follow latest");
            if ui.button("Export CSV").clicked() {
                self.file("Export CSV");
            }
        });
        if self.history_latest {
            self.history_index = len - 1;
        }
        self.history_index = self.history_index.min(len - 1);
        ui.add(egui::Slider::new(&mut self.history_index, 0..=len - 1).text("Generation"))
            .on_hover_text("Disable Follow latest to keep a historical generation selected");
        self.trend(ui, 180.);
        ui.add_space(GAP_M);
        self.records_list(ui);
        ui.add_space(GAP_M);
        self.species_history(ui);
        // A copy, so the rest of the tab can call methods that borrow `self`
        // mutably while it reads these numbers.
        let stats = self.snapshot.as_ref().unwrap().history[self.history_index].clone();
        let body_count = species_total(&stats);
        ui.horizontal(|ui| {
            ui.label(format!(
                "Generation {} · {} creatures tried · {} creatures kept in {} ways of moving",
                stats.generation,
                number(stats.population),
                number(stats.archive_cells),
                number(stats.moves()),
            ));
        });
        self.diversity_meter(ui, self.history_index);
        let mut picked_type = None;
        ui.columns(2, |cols| {
            self.histogram(&mut cols[0], &stats, 155.);
            crate::theme::heading(&mut cols[1], "Body types", theme);
            let mut species = stats.species.clone();
            species.sort_by_key(|&(_, _, n)| std::cmp::Reverse(n));
            egui::ScrollArea::vertical()
                .max_height(180.)
                .show(&mut cols[1], |ui| {
                    for &(n, m, count) in &species {
                        ui.horizontal(|ui| {
                            color_dot(ui, species_color(n, m));
                            // A click replays the fastest kept creature of
                            // this body type (a species view, as NEAT shows
                            // its species).
                            if ui
                                .link(format!(
                                    "{n} nodes / {} bones / {m} muscles",
                                    n.saturating_sub(1)
                                ))
                                .on_hover_text("Replay the fastest kept creature of this body type")
                                .clicked()
                            {
                                picked_type = Some((n, m));
                            }
                            ui.label(format!(
                                "{} · {:.1}%",
                                number(count as usize),
                                100. * count as f32 / body_count.max(1) as f32
                            ));
                        });
                    }
                });
        });
        ui.add_space(GAP_M);
        if let Some((n, m)) = picked_type {
            // The cards are the whole archive as of the last request, not as it
            // was at the picked generation. When they hold no creature of this
            // body type, ask the worker for fresh ones and replay nothing.
            let best = self.cards.as_ref().and_then(|list| {
                list.cards
                    .iter()
                    .filter(|c| c.creature.nodes.len() == n && c.creature.muscles.len() == m)
                    .max_by(|a, b| a.score.total_cmp(&b.score))
                    .map(|c| c.replay_of(&list.config))
            });
            match best {
                Some((creature, config)) => {
                    self.select(creature, config);
                    self.tab = Tab::Overview;
                    return;
                }
                None => self.request_cards(),
            }
        }
        // `representatives` holds the worst, median and best creature, in that
        // order.
        let mut selection = None;
        ui.columns(3, |cols| {
            for (i, ui) in cols.iter_mut().enumerate() {
                ui.label(["Worst creature", "Median creature", "Best creature"][i]);
                let (rect, response) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 100.), Sense::click());
                ui.painter().rect_filled(rect, 8, theme.card);
                thumbnail(
                    &ui.painter_at(rect),
                    &stats.representatives[i],
                    rect.shrink(12.),
                );
                if response.clicked() {
                    selection = Some(stats.representatives[i].clone());
                }
            }
        });
        if let Some(c) = selection {
            self.select(c, stats.config);
            self.tab = Tab::Overview;
        }
    }
}
/// The total that the body type counts of a history row are shares of: the
/// elites the archive kept, or the population when the row has no archive
/// cells.
fn species_total(stats: &Stats) -> usize {
    if stats.archive_cells > 0 {
        stats.archive_cells
    } else {
        stats.population
    }
}
