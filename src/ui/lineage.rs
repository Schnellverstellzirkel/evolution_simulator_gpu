//! The Lineage tab. It lists the recorded ancestors of the creature on screen
//! as tiles, newest first, and marks the tiles where the body plan changed. A
//! click on a tile replays that ancestor in the world of its generation and
//! goes back to the Overview. The worker traces the chain when `App` sends
//! `Command::Lineage`.

use super::{App, Tab, scene::thumbnail, text::species_name};
use crate::{
    config::Config,
    evolution::Creature,
    theme::{GAP_L, GAP_M, Theme},
};
use eframe::egui::{self, Align2, FontId, Pos2, Rect, RichText, Sense, Vec2};

impl App {
    /// The world a generation ran in: the settings its history row kept, or
    /// the live world for a generation without a row yet. Before the first
    /// snapshot it is the default world.
    fn world_of_generation(&self, generation: u32) -> Config {
        let Some(snapshot) = &self.snapshot else {
            return Config::default();
        };
        snapshot
            .history
            .iter()
            .rev()
            .find(|stats| stats.generation == generation)
            .map_or_else(|| snapshot.config.clone(), |stats| stats.config.clone())
    }
    /// The Lineage tab: a heading with a hint, then one full-width tile for each
    /// recorded ancestor of the creature on screen, newest first. While the
    /// list is empty, a line says why. A click on a tile replays that ancestor
    /// in the world of its generation, keeps the list and goes to the Overview.
    pub(super) fn lineage_view(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.heading("Lineage");
            ui.label(
                RichText::new(
                    "Ancestors of the creature on screen, newest first. Click one to replay it.",
                )
                .color(theme.muted),
            );
        });
        if self.lineage.is_empty() {
            ui.add_space(GAP_L);
            ui.label(
                RichText::new(if self.playback.is_none() {
                    "Pick a creature in Ways of moving to trace its ancestry."
                } else if self.lineage_pending {
                    "Requesting ancestors…"
                } else {
                    "No recorded ancestors for this creature yet. Evolve a few generations or pick a kept creature."
                })
                .color(theme.muted),
            );
            return;
        }
        if let Some(p) = &self.playback {
            ui.label(format!(
                "{} · {} nodes, {} bones, {} muscles · {} recorded ancestors",
                species_name(&p.creature),
                p.creature.nodes.len(),
                p.creature.bones.len(),
                p.creature.muscles.len(),
                self.lineage.len()
            ));
        }
        let shown = self.playback.as_ref().map(|p| p.creature.id);
        let mut chosen = None;
        egui::ScrollArea::vertical()
            .id_salt("lineage_view")
            .show(ui, |ui| {
                for k in 0..self.lineage.len() {
                    let step = &self.lineage[k];
                    if paint_lineage_tile(
                        ui,
                        step,
                        self.lineage.get(k + 1),
                        true,
                        shown == Some(step.creature.id),
                        theme,
                        Vec2::new(ui.available_width(), 120.),
                    ) {
                        chosen = Some(k);
                    }
                    ui.add_space(GAP_M);
                }
            });
        if let Some(k) = chosen {
            let config = self.world_of_generation(self.lineage[k].generation);
            self.select_ancestor(self.lineage[k].creature.clone(), config);
            self.tab = Tab::Overview;
        }
    }
}
/// The numbers of nodes, bones and muscles of `creature`, in that order.
fn body_counts(creature: &Creature) -> (usize, usize, usize) {
    (
        creature.nodes.len(),
        creature.bones.len(),
        creature.muscles.len(),
    )
}
/// Whether an ancestor's node, bone or muscle count differs from its parent's.
/// It is false without a parent, which is the case for the oldest recorded
/// ancestor.
fn body_plan_changed(
    step: &crate::worker::LineageStep,
    parent: Option<&crate::worker::LineageStep>,
) -> bool {
    parent.is_some_and(|parent| body_counts(&step.creature) != body_counts(&parent.creature))
}
/// Paints one ancestor tile of size `size`: its thumbnail, generation,
/// distance, gain over its parent and species name. `parent` is the next older
/// ancestor in the list. The tile says "selected" when `current` is set, which
/// is the creature on screen, and "Body plan" when its body plan differs from
/// `parent`'s. Its outline is lit for those two and under the pointer. The
/// hover text gives the counts of nodes, bones and muscles, what changed from
/// the parent, and the parent's generation and distance. Returns whether the
/// player clicked the tile.
fn paint_lineage_tile(
    ui: &mut egui::Ui,
    step: &crate::worker::LineageStep,
    parent: Option<&crate::worker::LineageStep>,
    big: bool,
    current: bool,
    theme: Theme,
    size: Vec2,
) -> bool {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let hovered = response.hovered();
    let painter = ui.painter_at(rect);
    let changed = body_plan_changed(step, parent);
    crate::theme::plate(
        &painter,
        rect,
        theme,
        if hovered {
            theme.card_hover
        } else {
            theme.card
        },
        current || changed || hovered,
    );
    let art_size = (size.y - 16.).clamp(40., 96.);
    thumbnail(
        &painter,
        &step.creature,
        Rect::from_min_size(rect.left_top() + Vec2::splat(8.), Vec2::splat(art_size)),
    );
    let text_x = rect.left() + 16. + art_size;
    painter.text(
        Pos2::new(text_x, rect.top() + 8.),
        Align2::LEFT_TOP,
        format!("Generation {}", step.generation),
        FontId::proportional(15.),
        theme.muted,
    );
    painter.text(
        Pos2::new(text_x, rect.top() + 28.),
        Align2::LEFT_TOP,
        format!("{:.2} m", step.fitness),
        FontId::proportional(19.),
        if big { theme.accent } else { theme.ink },
    );
    painter.text(
        Pos2::new(text_x, rect.top() + 54.),
        Align2::LEFT_TOP,
        format!("{:+.2} m", step.gain),
        FontId::proportional(15.),
        if step.gain >= 0. {
            theme.accent
        } else {
            theme.warn
        },
    );
    painter.text(
        rect.right_bottom() + Vec2::new(-8., -6.),
        Align2::RIGHT_BOTTOM,
        species_name(&step.creature),
        FontId::proportional(14.5),
        theme.muted,
    );
    if current {
        painter.text(
            rect.right_bottom() + Vec2::new(-8., -20.),
            Align2::RIGHT_BOTTOM,
            "selected",
            FontId::proportional(14.5),
            theme.accent,
        );
    }
    if changed {
        crate::theme::caps_text(
            &painter,
            rect.right_top() + Vec2::new(-8., 8.),
            Align2::RIGHT_TOP,
            "Body plan",
            13.,
            theme.accent,
        );
    }
    let (nodes, bones, muscles) = body_counts(&step.creature);
    response
        .on_hover_text(format!(
            "{} · generation {} · {:.2} m ({:+.2} m)\n{} nodes / {} bones / {} muscles\n{}\n{}",
            species_name(&step.creature),
            step.generation,
            step.fitness,
            step.gain,
            nodes,
            bones,
            muscles,
            step.change,
            parent.map_or_else(
                || "oldest recorded ancestor".to_owned(),
                |p| format!("parent: generation {} at {:.2} m", p.generation, p.fitness),
            ),
        ))
        .clicked()
}
