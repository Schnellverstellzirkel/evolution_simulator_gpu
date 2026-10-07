//! The Ways of moving tab: the archive's cards with their filters, and the heat
//! map of the archive (the island view is in `islands`).

use super::{
    App,
    records::row_in_world,
    scene::thumbnail,
    text::{number, species_name},
    widgets::{Choices, heat_color},
};
use crate::{
    config::Config,
    evolution::{Creature, FAILED},
    theme::Theme,
    worker::{Command, Snapshot},
};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

/// The body classes of the cells the archive tab shows: the global archive's.
const CLASSES: &crate::qd::Classes = &crate::qd::GLOBAL_CLASSES;
/// Movement-axis bin counts, mirroring `qd::MOVEMENT_BINS` (ground contact, cadence,
/// shape, height, feet). The shape and the size classes are filters.
const MAP_BINS: [usize; 5] = [6, 8, 1, 6, 5];
/// Which representation the Behavior archive tab shows.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum ArchiveView {
    Cards,
    Map,
    Islands,
}
/// Which archive cards the player looks at.
#[derive(Clone, Copy, Default, PartialEq)]
pub(super) struct CardFilter {
    /// Feet bin (0 is one foot, 4 is five or more), or every count.
    feet: Option<u8>,
    /// Body size class (`qd::SIZE_NAMES`), or every size.
    size: Option<u8>,
    /// Body shape class (`qd::SHAPE_NAMES`), or every shape.
    shape: Option<u8>,
}
impl CardFilter {
    fn shows(self, card: &crate::worker::Card) -> bool {
        let niche = card.descriptor.map(|d| d.niche().0);
        self.feet
            .is_none_or(|feet| niche.is_some_and(|n| n[4] == feet))
            && self
                .size
                .is_none_or(|size| niche.is_some_and(|n| n[5] == size))
            && self
                .shape
                .is_none_or(|shape| niche.is_some_and(|n| n[2] == shape))
    }
}
impl App {
    /// Renders the Ways of moving tab: archive cards, heat map, or islands view.
    pub(super) fn population(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        self.population_header(ui, theme);
        if self.archive_view == ArchiveView::Islands {
            self.islands_view(ui);
            return;
        }
        if self.archive_view == ArchiveView::Map {
            self.map_filters(ui, theme);
        }
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        Self::sorted_by_note(ui, snapshot, theme);
        if self.archive_view == ArchiveView::Cards {
            let mut filter = self.card_filter;
            Self::card_filters(ui, &mut filter, theme);
            self.card_filter = filter;
        }
        let mut selected = None;
        let mut map_click = None;
        if self.archive_view == ArchiveView::Map {
            let empty = Vec::new();
            let cells = snapshot.map.as_deref().unwrap_or(&empty);
            map_click = paint_archive_map(
                ui,
                cells,
                [
                    self.map_height,
                    self.map_feet,
                    self.map_shape,
                    self.map_size,
                ],
                theme,
            );
        } else {
            self.card_grid(ui, &mut selected);
        }
        if let Some(id) = map_click {
            self.worker.send(Command::Select(id));
        }
        if let Some((creature, config)) = selected {
            self.select(creature, config);
        }
    }
    /// The tab's title, its Map, Cards and Islands choices and the hint.
    fn population_header(&mut self, ui: &mut egui::Ui, theme: Theme) {
        ui.horizontal(|ui| {
            ui.heading("Ways of moving");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.choice(&mut self.archive_view, ArchiveView::Map, "Map")
                    .on_hover_text(
                        "Watch evolution fill the ways of moving. Cells are colored by distance.",
                    );
                ui.choice(&mut self.archive_view, ArchiveView::Cards, "Cards");
                ui.choice(&mut self.archive_view, ArchiveView::Islands, "Islands")
                    .on_hover_text(
                        "The four isolated islands and the hub, each with its best creature, its top elites and its migrants.",
                    );
            });
        });
        ui.label(RichText::new("Click a creature to replay it.").color(theme.muted));
    }
    /// The Map's four filters: body height, feet, shape and size.
    fn map_filters(&mut self, ui: &mut egui::Ui, theme: Theme) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Body height").small().color(theme.muted));
            egui::ComboBox::from_id_salt("map_height")
                .selected_text(self.map_height.map_or("All".to_owned(), height_bin_label))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.map_height, None, "All");
                    for bin in 0..MAP_BINS[3] {
                        ui.selectable_value(&mut self.map_height, Some(bin), height_bin_label(bin));
                    }
                });
            ui.label(RichText::new("Feet").small().color(theme.muted));
            egui::ComboBox::from_id_salt("map_feet")
                .selected_text(self.map_feet.map_or("All".to_owned(), feet_bin_label))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.map_feet, None, "All");
                    for bin in 0..MAP_BINS[4] {
                        ui.selectable_value(&mut self.map_feet, Some(bin), feet_bin_label(bin));
                    }
                });
            ui.label(RichText::new("Shape").small().color(theme.muted));
            egui::ComboBox::from_id_salt("map_shape")
                .selected_text(
                    self.map_shape
                        .map_or("All", |class| CLASSES.shape_names[class]),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.map_shape, None, "All");
                    for (class, name) in CLASSES.shape_names.iter().enumerate() {
                        ui.selectable_value(&mut self.map_shape, Some(class), *name)
                            .on_hover_text(CLASSES.shape_about(class));
                    }
                });
            ui.label(RichText::new("Size").small().color(theme.muted));
            egui::ComboBox::from_id_salt("map_size")
                .selected_text(
                    self.map_size
                        .map_or("All", |class| CLASSES.size_names[class]),
                )
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.map_size, None, "All");
                    for (class, name) in CLASSES.size_names.iter().enumerate() {
                        ui.selectable_value(&mut self.map_size, Some(class), *name)
                            .on_hover_text(CLASSES.size_about(class));
                    }
                });
        });
    }
    /// How many creatures the archive holds and what the ways of moving are
    /// sorted by.
    fn sorted_by_note(ui: &mut egui::Ui, snapshot: &Snapshot, theme: Theme) {
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(format!(
                    "{} creatures in {} ways of moving, sorted by",
                    number(snapshot.archive_cells),
                    number(snapshot.movement_cells)
                ))
                .small()
                .color(theme.muted),
            );
            for (name, why) in [
                (
                    "ground contact",
                    "How much of the trial the creature keeps its nodes on the ground.",
                ),
                (
                    "stride rate",
                    "How many up-and-down body swings the gait makes per second.",
                ),
                (
                    "body height",
                    "The average height of the body above the ground during the trial.",
                ),
                (
                    "feet",
                    "Nodes that touch the ground and lift off again. A node dragged along the ground never lifts, so it is not a foot.",
                ),
            ] {
                ui.label(RichText::new(name).small().color(theme.ink))
                    .on_hover_text(why);
            }
        });
    }
    /// The Cards' filters: feet, size and shape.
    fn card_filters(ui: &mut egui::Ui, filter: &mut CardFilter, theme: Theme) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.;
            ui.label(RichText::new("Feet").small().color(theme.muted));
            if ui.pick(filter.feet.is_none(), "All").clicked() {
                filter.feet = None;
            }
            for bin in 0..MAP_BINS[4] as u8 {
                if ui
                    .pick(
                        filter.feet == Some(bin),
                        feet_bin_label(usize::from(bin))
                            .replace(" feet", "")
                            .replace(" foot", ""),
                    )
                    .clicked()
                {
                    filter.feet = Some(bin);
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.;
            ui.label(RichText::new("Size").small().color(theme.muted));
            if ui.pick(filter.size.is_none(), "All").clicked() {
                filter.size = None;
            }
            for (class, name) in CLASSES.size_names.iter().enumerate() {
                if ui
                    .pick(filter.size == Some(class as u8), *name)
                    .on_hover_text(CLASSES.size_about(class))
                    .clicked()
                {
                    filter.size = Some(class as u8);
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.;
            ui.label(RichText::new("Shape").small().color(theme.muted));
            if ui.pick(filter.shape.is_none(), "All").clicked() {
                filter.shape = None;
            }
            for (class, name) in CLASSES.shape_names.iter().enumerate() {
                if ui
                    .pick(filter.shape == Some(class as u8), *name)
                    .on_hover_text(CLASSES.shape_about(class))
                    .clicked()
                {
                    filter.shape = Some(class as u8);
                }
            }
        });
    }
    /// The archive cards: the ranked list the UI holds, filtered here, so no
    /// card waits for data or moves while the player looks. A newer list
    /// replaces it only when the player asks for it (or opens the tab).
    fn card_grid(&mut self, ui: &mut egui::Ui, selected: &mut Option<(Creature, Config)>) {
        let theme = self.theme();
        let (generation, archive_size) = self
            .snapshot
            .as_ref()
            .map_or((0, 0), |s| (s.generation, s.archive_size));
        let waiting = self
            .cards_requested
            .is_some_and(|at| at.elapsed() < Duration::from_secs(2));
        // A list scored in an earlier world is never shown.
        let world = self.snapshot.as_ref().map(|s| s.config.clone());
        if self.cards.as_ref().is_some_and(|list| {
            world
                .as_ref()
                .is_some_and(|w| list.config.physics_differs(w))
        }) {
            self.cards = None;
        }
        let empty = self.cards.as_ref().is_none_or(|list| list.cards.is_empty());
        if empty && archive_size > 0 && !waiting {
            // Nothing held yet (or the request got lost): ask again.
            self.request_cards();
        }
        let Some(list) = self.cards.clone() else {
            let changed = self
                .snapshot
                .as_ref()
                .is_some_and(|s| !s.history.is_empty() && row_in_world(s).is_none());
            ui.label(
                RichText::new(if changed {
                    "Testing in the new world... Creatures appear here as they are kept."
                } else {
                    "The first generation is running. Its creatures appear here as they are kept."
                })
                .color(theme.muted),
            );
            return;
        };
        if list.generation < generation {
            let mut refresh = false;
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("Showing generation {}.", list.generation))
                        .small()
                        .color(theme.muted),
                );
                refresh = ui
                    .small_button(format!("Show latest (generation {generation})"))
                    .on_hover_text(
                        "The cards stay still while you look; this loads the newest ranking",
                    )
                    .clicked();
            });
            if refresh {
                self.request_cards();
            }
        }
        let filter = self.card_filter;
        let visible: Vec<&crate::worker::Card> = list
            .cards
            .iter()
            .filter(|card| filter.shows(card))
            .collect();
        if visible.is_empty() {
            ui.label(
                RichText::new(if list.cards.is_empty() {
                    "No creatures kept yet."
                } else {
                    "No kept creature matches these filters."
                })
                .color(theme.muted),
            );
            return;
        }
        let columns = (ui.available_width() / 190.).floor().max(2.) as usize;
        let width = (ui.available_width() - (columns - 1) as f32 * 10.) / columns as f32;
        let shown = self.playback.as_ref().map(|p| p.creature.id);
        egui::ScrollArea::vertical()
            .id_salt("population_grid")
            .show_rows(ui, 162., visible.len().div_ceil(columns), |ui, rows| {
                for row in rows {
                    ui.horizontal(|ui| {
                        for card in visible.iter().skip(row * columns).take(columns) {
                            let (rect, response) =
                                ui.allocate_exact_size(Vec2::new(width, 152.), Sense::click());
                            paint_card(
                                ui.painter(),
                                card,
                                rect,
                                response.hovered() || shown == Some(card.creature.id),
                                theme,
                            );
                            if response.clicked() {
                                *selected = Some(card.replay_of(&list.config));
                            }
                            response.on_hover_text(format!(
                                "{}\n{} nodes, {} bones, {} muscles{}\n{}\n{}\nClick to replay",
                                species_name(&card.creature),
                                card.creature.nodes.len(),
                                card.creature.bones.len(),
                                card.creature.muscles.len(),
                                card.descriptor.map_or(String::new(), |d| {
                                    let niche = d.niche().0;
                                    format!(
                                        " · {} {} body",
                                        CLASSES.shape_names[usize::from(niche[2])]
                                            .to_lowercase(),
                                        CLASSES.size_names[usize::from(niche[5])].to_lowercase()
                                    )
                                }),
                                card.emitter.map_or("First generation".to_owned(), |emitter| {
                                    format!("Born {}", origin_words(emitter))
                                }),
                                card.descriptor.map_or_else(
                                    || "Trial done".to_owned(),
                                    |d| format!(
                                        "On the ground {:.0}% of the time · {:.2} strides/s · {:.2} m tall · {:.0} feet",
                                        d.ground_contact * 100.0,
                                        d.gait_frequency,
                                        d.mean_height,
                                        d.feet
                                    ),
                                )
                            ));
                        }
                    });
                }
            });
    }
    /// Asks the worker for the ranked archive; it arrives with a snapshot.
    pub(super) fn request_cards(&mut self) {
        self.cards_requested = Some(Instant::now());
        self.worker.send(Command::Cards);
    }
}
/// Height range of one archive height bin; mirrors `qd::height_axis`.
fn height_bin_range(bin: usize) -> (f32, f32) {
    let low = 0.15f32;
    let high = (0.6 * crate::evolution::max_bone_length()).max(2.0 * low);
    let at = |t: f32| low * (high / low).powf(t);
    (at(bin as f32 / 6.0), at((bin + 1) as f32 / 6.0))
}
fn height_bin_label(bin: usize) -> String {
    let (low, high) = height_bin_range(bin);
    format!("{low:.2} to {high:.2} m")
}
fn feet_bin_label(bin: usize) -> String {
    match bin {
        0 => "1 foot".to_owned(),
        1..=3 => format!("{} feet", bin + 1),
        _ => "5+ feet".to_owned(),
    }
}
/// Heat map of the archive: for each ground contact and cadence pair, the
/// best creature among the height, feet, shape and size bins the filters
/// (`[height, feet, shape, size]`) let through. One color scale spans every
/// cell of the archive, so a color means the same distance whatever the
/// filters. Returns the id of a clicked cell's creature.
fn paint_archive_map(
    ui: &mut egui::Ui,
    cells: &[crate::worker::MapCell],
    [height_bin, feet_bin, shape_bin, size_bin]: [Option<usize>; 4],
    theme: Theme,
) -> Option<u64> {
    let (min, max) = cells
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), cell| {
            (lo.min(cell.score), hi.max(cell.score))
        });
    let range = (max - min).max(1e-6);
    let best = map_best_cells(cells, [height_bin, feet_bin, shape_bin, size_bin]);
    ui.label(
        RichText::new(format!(
            "Ground contact against stride rate · {} cells · the best of each is shown",
            best.len(),
        ))
        .small()
        .color(theme.muted),
    );
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), ui.available_height().max(220.)),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    let plot = Rect::from_min_max(
        rect.left_top() + Vec2::new(56., 34.),
        rect.right_bottom() - Vec2::new(12., 42.),
    );
    if plot.width() < 80. || plot.height() < 80. {
        return None;
    }
    paint_map_axes(&painter, rect, plot, theme);
    if !cells.is_empty() {
        paint_map_legend(&painter, rect, min, max, theme);
    }
    if best.is_empty() {
        painter.text(
            plot.center(),
            Align2::CENTER_CENTER,
            if cells.is_empty() {
                "No creatures kept yet. The map fills as evolution runs."
            } else {
                "No creatures with this height and these feet. Try All."
            },
            FontId::proportional(16.),
            theme.muted,
        );
    }
    paint_map_cells(ui, &painter, plot, &best, (min, range), theme)
}
/// The grid of the map in `plot`: columns, rows and the size of a cell.
fn map_grid(plot: Rect) -> (usize, usize, f32, f32) {
    let columns = MAP_BINS[0];
    let rows = MAP_BINS[1];
    let column_width = plot.width() / columns as f32;
    let row_height = plot.height() / rows as f32;
    (columns, rows, column_width, row_height)
}
/// Best cell and how many ways of moving share each contact and cadence
/// pair under the filters.
fn map_best_cells(
    cells: &[crate::worker::MapCell],
    [height_bin, feet_bin, shape_bin, size_bin]: [Option<usize>; 4],
) -> HashMap<(u8, u8), (crate::worker::MapCell, usize)> {
    let mut best: HashMap<(u8, u8), (crate::worker::MapCell, usize)> = HashMap::new();
    for cell in cells.iter().filter(|cell| {
        height_bin.is_none_or(|bin| usize::from(cell.niche[3]) == bin)
            && feet_bin.is_none_or(|bin| usize::from(cell.niche[4]) == bin)
            && shape_bin.is_none_or(|bin| usize::from(cell.niche[2]) == bin)
            && size_bin.is_none_or(|bin| usize::from(cell.niche[5]) == bin)
    }) {
        let entry = best
            .entry((cell.niche[0], cell.niche[1]))
            .or_insert((*cell, 0));
        entry.1 += 1;
        if cell.score > entry.0.score {
            entry.0 = *cell;
        }
    }
    best
}
/// The map's grid lines, their labels and the two axis titles.
fn paint_map_axes(painter: &egui::Painter, rect: Rect, plot: Rect, theme: Theme) {
    let (columns, rows, column_width, row_height) = map_grid(plot);
    for column in 0..=columns {
        let x = plot.left() + column as f32 * column_width;
        painter.line_segment(
            [Pos2::new(x, plot.top()), Pos2::new(x, plot.bottom())],
            Stroke::new(1., theme.card_border),
        );
        painter.text(
            Pos2::new(x, plot.bottom() + 4.),
            Align2::CENTER_TOP,
            format!("{:.0}%", column as f32 / columns as f32 * 100.),
            FontId::proportional(14.5),
            theme.muted,
        );
    }
    for row in 0..=rows {
        let y = plot.bottom() - row as f32 * row_height;
        painter.line_segment(
            [Pos2::new(plot.left(), y), Pos2::new(plot.right(), y)],
            Stroke::new(1., theme.card_border),
        );
        painter.text(
            Pos2::new(plot.left() - 6., y),
            Align2::RIGHT_CENTER,
            format!("{:.2}", row as f32 / rows as f32 * 6.),
            FontId::proportional(14.5),
            theme.muted,
        );
    }
    painter.text(
        Pos2::new(plot.center().x, plot.bottom() + 24.),
        Align2::CENTER_TOP,
        "Share of the trial on the ground",
        FontId::proportional(14.),
        theme.ink,
    );
    painter.text(
        Pos2::new(rect.left() + 4., plot.top() - 16.),
        Align2::LEFT_BOTTOM,
        "Strides per second",
        FontId::proportional(14.),
        theme.ink,
    );
}
/// The color scale above the map, from the lowest to the highest distance.
fn paint_map_legend(painter: &egui::Painter, rect: Rect, min: f32, max: f32, theme: Theme) {
    let legend = Rect::from_min_max(
        Pos2::new(rect.right() - 272., rect.top() + 8.),
        Pos2::new(rect.right() - 92., rect.top() + 22.),
    );
    let steps = 48;
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        painter.rect_filled(
            Rect::from_min_max(
                Pos2::new(
                    legend.left() + legend.width() * i as f32 / steps as f32,
                    legend.top(),
                ),
                Pos2::new(
                    legend.left() + legend.width() * (i + 1) as f32 / steps as f32,
                    legend.bottom(),
                ),
            ),
            0,
            heat_color(t),
        );
    }
    painter.text(
        Pos2::new(legend.left() - 6., legend.center().y),
        Align2::RIGHT_CENTER,
        format!("{min:.2} m"),
        FontId::proportional(14.5),
        theme.muted,
    );
    painter.text(
        Pos2::new(legend.right() + 6., legend.center().y),
        Align2::LEFT_CENTER,
        format!("{max:.2} m"),
        FontId::proportional(14.5),
        theme.muted,
    );
}
/// The map's cells and their hover and click. Returns the id of a clicked
/// cell's creature.
fn paint_map_cells(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    plot: Rect,
    best: &HashMap<(u8, u8), (crate::worker::MapCell, usize)>,
    (min, range): (f32, f32),
    theme: Theme,
) -> Option<u64> {
    let (_, _, column_width, row_height) = map_grid(plot);
    let mut clicked = None;
    for (&(contact, cadence), (cell, count)) in best {
        let inner = Rect::from_min_size(
            Pos2::new(
                plot.left() + contact as f32 * column_width + 1.5,
                plot.bottom() - (cadence as f32 + 1.) * row_height + 1.5,
            ),
            Vec2::new(column_width - 3., row_height - 3.),
        );
        let color = heat_color((cell.score - min) / range);
        painter.rect_filled(inner, 3, color);
        let luminance = (color.r() as u32 + color.g() as u32 + color.b() as u32) / 3;
        let ink = if luminance > 150 {
            Color32::from_rgb(24, 24, 20)
        } else {
            Color32::WHITE
        };
        if inner.width() > 34. && inner.height() > 16. {
            painter.text(
                inner.center(),
                Align2::CENTER_CENTER,
                format!("{:.1}", cell.score),
                FontId::proportional(14.),
                ink,
            );
        }
        let response = ui.interact(
            inner,
            ui.id().with(("archive_map", contact, cadence)),
            Sense::click(),
        );
        if response.hovered() {
            painter.rect_stroke(
                inner,
                3,
                Stroke::new(2., theme.ink),
                egui::StrokeKind::Inside,
            );
        }
        if response.clicked() {
            clicked = Some(cell.id);
        }
        response.on_hover_text(format!(
            "Rank {} · {:.2} m · {} tall · {}\n{} ways of moving in this cell, the best is shown\nClick to replay",
            cell.rank + 1,
            cell.score,
            height_bin_label(usize::from(cell.niche[3])),
            feet_bin_label(usize::from(cell.niche[4])),
            count,
        ));
    }
    clicked
}
fn paint_card(
    painter: &egui::Painter,
    card: &crate::worker::Card,
    rect: Rect,
    hovered: bool,
    theme: Theme,
) {
    crate::theme::plate(
        painter,
        rect,
        theme,
        if hovered {
            theme.card_hover
        } else {
            theme.card
        },
        hovered,
    );
    thumbnail(
        painter,
        &card.creature,
        Rect::from_min_max(
            rect.left_top() + Vec2::new(12., 46.),
            rect.right_bottom() - Vec2::new(12., 30.),
        ),
    );
    painter.text(
        rect.left_top() + Vec2::new(9., 8.),
        Align2::LEFT_TOP,
        if card.descriptor.is_some() {
            format!("#{}", card.rank + 1)
        } else {
            format!("ID {}", card.creature.id)
        },
        FontId::proportional(14.),
        theme.accent.gamma_multiply(0.85),
    );
    painter.text(
        rect.left_top() + Vec2::new(9., 26.),
        Align2::LEFT_TOP,
        species_name(&card.creature),
        FontId::proportional(14.5),
        theme.ink,
    );
    if card.innovation_reserve {
        crate::theme::caps_text(
            painter,
            rect.right_top() + Vec2::new(-9., 9.),
            Align2::RIGHT_TOP,
            "New body",
            13.,
            theme.cold,
        );
    }
    let (label, score_color) = if !card.score.is_finite() {
        if card.parent_score.is_finite() && card.parent_score > FAILED {
            (format!("Parent {:.3} m", card.parent_score), theme.muted)
        } else if card.parent_score.is_finite() {
            ("Parent failed".into(), theme.warn)
        } else {
            ("Trial pending".into(), theme.muted)
        }
    } else if card.score <= FAILED {
        ("Failed trial".into(), theme.warn)
    } else {
        (
            format!("{:.3} m", card.score),
            if card.survivor {
                theme.accent
            } else {
                theme.ink
            },
        )
    };
    painter.text(
        rect.left_bottom() + Vec2::new(9., -9.),
        Align2::LEFT_BOTTOM,
        label,
        FontId::proportional(15.),
        score_color,
    );
}
/// How a creature came to be, in the words the lineage uses.
fn origin_words(emitter: crate::qd::Emitter) -> &'static str {
    match emitter {
        crate::qd::Emitter::Cma => "fine-tuned from a parent",
        crate::qd::Emitter::Structural => "reshaped from a parent",
        crate::qd::Emitter::Novelty => "exploring a new way of moving",
        crate::qd::Emitter::Restart => "as a new random body",
    }
}
