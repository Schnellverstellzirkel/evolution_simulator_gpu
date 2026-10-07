//! The side panel of the window. It shows the world as a summary with the
//! active effects, the presets, one row of levels for every environment effect
//! and the catastrophe buttons. A click that changes the world updates
//! `App::config` and sends it to the worker as `Command::Configure`. The module
//! also holds `world_summary`, which the viewport, the History tab and the
//! dialogs use to name a world, and `worlds_match`, which the frame loop uses
//! to compare two worlds.

use super::{App, GAP_S, text::number};
use crate::{
    config::Config,
    theme::{GAP_M, Theme},
    worker::Command,
};
use eframe::egui::{self, Align2, FontId, RichText, Sense, Stroke, Vec2};
use std::time::Instant;

/// Height of one level button and of the name cell beside it. A bar of levels
/// that wraps onto a second line makes its row taller.
const LEVEL_HEIGHT: f32 = 30.0;
/// Width of the column of effect names in the Effects block.
const EFFECT_NAME_WIDTH: f32 = 100.0;
impl App {
    /// Draws the panel in a vertical scroll area.
    pub(super) fn controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| self.control_contents(ui));
    }
    /// Draws the four blocks one after another. When the player changed the
    /// world, the previous world goes onto `world_undo`, unless the change was
    /// an undo or left the physics as it was. The stack keeps the last 20
    /// worlds. Then the new config goes to the worker as `Command::Configure`,
    /// and `config_sent` notes the time for `App::absorb_snapshot`.
    fn control_contents(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.add_space(GAP_M);
        let mut world_changed = false;
        let world_before = self.config.clone();
        let mut undoing = false;
        crate::theme::section(ui, "World", theme);
        // The world that runs now. The panel shows the world the player asked
        // for, and the two can differ.
        let live = self.snapshot.as_ref().map(|s| s.config.clone());
        world_changed |= self.world_block(ui, theme, &live);
        let (changed, undone) = self.presets_block(ui, theme);
        world_changed |= changed;
        undoing |= undone;
        world_changed |= self.effects_block(ui, theme, &live);
        self.catastrophes_block(ui, theme);
        if world_changed {
            if !undoing && world_before.physics_differs(&self.config) {
                self.world_undo.push(world_before);
                if self.world_undo.len() > 20 {
                    self.world_undo.remove(0);
                }
            }
            self.worker.send(Command::Configure(self.config.clone()));
            self.config_sent = Some(Instant::now());
        }
    }
    /// The World block. It shows the summary of the world the player asked
    /// for, with a Calm world button while an effect is away from calm. When
    /// the physics of `live`, the world that runs now, differ from the panel's,
    /// a line says what runs. Each effect away from calm gets a line with its
    /// reason and an Undo button. Returns whether the player changed the world.
    fn world_block(&mut self, ui: &mut egui::Ui, theme: Theme, live: &Option<Config>) -> bool {
        let mut world_changed = false;
        let calm = world_is_calm(&self.config);
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(world_summary(&self.config))
                    .size(18.)
                    .strong()
                    .color(if calm { theme.ink } else { theme.accent }),
            );
            if !calm
                && ui
                    .small_button("Calm world")
                    .on_hover_text("Set every effect back to the calm world in one change.")
                    .clicked()
            {
                // The autochange level stays as it is.
                for effect in &crate::environment::EFFECTS {
                    if effect.name != "Autochange environment" {
                        effect.set_level(&mut self.config, effect.calm);
                    }
                }
                world_changed = true;
            }
        });
        if let Some(live) = live
            && live.physics_differs(&self.config)
        {
            ui.label(
                RichText::new(format!(
                    "Now: {}. The change starts with the next generation.",
                    world_summary(live)
                ))
                .small()
                .color(theme.warn),
            );
        }
        // One line for each effect away from calm, with its reason below it and
        // an Undo that sets the effect back to calm. The Undo is applied after
        // the loop, because the loop reads `self.config`.
        let mut undo_effect = None;
        for (i, effect) in crate::environment::EFFECTS
            .iter()
            .enumerate()
            .filter(|(_, effect)| effect.name != "Autochange environment")
            .filter(|(_, effect)| effect.level(&self.config) != effect.calm)
        {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(effect_text(effect, &self.config)).color(theme.accent));
                if ui
                    .small_button("Undo")
                    .on_hover_text("Set this effect back to calm")
                    .clicked()
                {
                    undo_effect = Some(i);
                }
            });
            ui.label(RichText::new(effect.why).small().color(theme.muted));
        }
        if let Some(i) = undo_effect {
            let effect = &crate::environment::EFFECTS[i];
            effect.set_level(&mut self.config, effect.calm);
            world_changed = true;
        }
        world_changed
    }
    /// The Presets block: one button for each preset and an Undo last change
    /// button, which takes the newest world off `world_undo`. Returns whether
    /// the world changed and whether that change was the undo.
    fn presets_block(&mut self, ui: &mut egui::Ui, theme: Theme) -> (bool, bool) {
        let mut world_changed = false;
        let mut undoing = false;
        ui.add_space(GAP_S);
        crate::theme::section(ui, "Presets", theme);
        ui.horizontal_wrapped(|ui| {
            for preset in &crate::environment::PRESETS {
                if ui
                    .small_button(preset.name)
                    .on_hover_text(format!(
                        "{} Sets these effects and calms the rest. Undo returns the previous world.",
                        preset.about
                    ))
                    .clicked()
                {
                    preset.apply(&mut self.config);
                    world_changed = true;
                }
            }
            if ui
                .add_enabled(
                    !self.world_undo.is_empty(),
                    egui::Button::new("Undo last change").small(),
                )
                .on_hover_text("Return to the world before your last change here")
                .clicked()
                && let Some(previous) = self.world_undo.pop()
            {
                self.config = previous;
                undoing = true;
                world_changed = true;
            }
        });
        (world_changed, undoing)
    }
    /// The Effects block: a hint, then a row of levels for each effect with the
    /// autochange row last, then the forecast of the next autochange step.
    /// `live` is the world that runs now. Returns whether the player changed
    /// the world.
    fn effects_block(&mut self, ui: &mut egui::Ui, theme: Theme, live: &Option<Config>) -> bool {
        let mut world_changed = false;
        ui.add_space(GAP_S);
        crate::theme::section(ui, "Effects", theme);
        ui.label(
            RichText::new(
                "Click a level to change the world. The best creatures are tested again in the new world.",
            )
            .small()
            .color(theme.muted),
        );
        // One row for every effect and the autochange: the name in a fixed
        // column, then its levels as one segmented bar. The autochange row gets
        // no `live` world, so it is never waiting.
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.y = GAP_S + 2.;
            for effect in crate::environment::EFFECTS
                .iter()
                .filter(|effect| effect.name != "Autochange environment")
            {
                if effect_row(ui, effect, &mut self.config, live.as_ref(), theme) {
                    world_changed = true;
                }
            }
            ui.add_space(GAP_S);
            if let Some(autochange) = crate::environment::EFFECTS
                .iter()
                .find(|effect| effect.name == "Autochange environment")
                && effect_row(ui, autochange, &mut self.config, None, theme)
            {
                world_changed = true;
            }
        });
        let generation = self.snapshot.as_ref().map_or(0, |s| s.generation);
        if let Some(forecast) = autochange_forecast(&self.config, generation) {
            ui.label(RichText::new(forecast).small().color(theme.muted));
        }
        world_changed
    }
    /// The Catastrophes block: Meteor strike, Extinction and an Undo that
    /// brings back the creatures they removed. Each button sends its command to
    /// the worker at once. The Undo is off while `Snapshot::fossils` is 0.
    fn catastrophes_block(&self, ui: &mut egui::Ui, theme: Theme) {
        let fossils = self.snapshot.as_ref().map_or(0, |s| s.fossils);
        ui.add_space(GAP_M);
        crate::theme::section(ui, "Catastrophes", theme).on_hover_text(
            "A catastrophe wipes out creatures that evolution kept. Survivors and newcomers refill the empty places, which makes room for new ways of moving.",
        );
        ui.horizontal_wrapped(|ui| {
            if ui
                .small_button("Meteor strike")
                .on_hover_text("Wipe out half of the kept creatures at random. Undo brings them back.")
                .clicked()
            {
                self.worker.send(Command::Meteor);
            }
            if ui
                .small_button("Extinction")
                .on_hover_text("Wipe out the group whose best creature is slowest, so it starts over from new designs. Undo brings them back.")
                .clicked()
            {
                self.worker.send(Command::Extinction);
            }
            if ui
                .add_enabled(
                    fossils > 0,
                    egui::Button::new(if fossils > 0 {
                        format!("Undo ({})", number(fossils))
                    } else {
                        "Undo".to_owned()
                    })
                    .small(),
                )
                .on_hover_text(format!(
                    "Bring back {} creatures lost to catastrophes, where their place is empty or holds a slower creature.",
                    number(fossils)
                ))
                .clicked()
            {
                self.worker.send(Command::UndoMeteor);
            }
        });
    }
}
/// The next autochange step as a line of text, such as "Next change at
/// generation 100: Wind to Breeze". It is `None` when autochange is off or the
/// ladder has no step left. The worker applies step `autochange_step` when a
/// generation that is a multiple of the interval begins, so the line gives the
/// first multiple of the interval above `generation`.
fn autochange_forecast(config: &Config, generation: u32) -> Option<String> {
    let interval = crate::environment::autochange_interval(config.autochange)?;
    if interval == 0 {
        return None;
    }
    let at = (generation / interval + 1) * interval;
    let ladder = crate::environment::autochange_ladder();
    let &(index, level) = ladder.get(usize::from(config.autochange_step))?;
    let effect = &crate::environment::EFFECTS[index];
    Some(format!(
        "Next change at generation {at}: {} to {}",
        effect.name, effect.levels[level]
    ))
}
/// Whether two configs have every effect, including autochange, at the same
/// level. `App::absorb_snapshot` uses it to see that the worker's world shows
/// the player's last click.
pub(super) fn worlds_match(a: &Config, b: &Config) -> bool {
    crate::environment::EFFECTS
        .iter()
        .all(|effect| effect.level(a) == effect.level(b))
}
/// Whether every effect except the autochange schedule sits at its calm level.
fn world_is_calm(config: &Config) -> bool {
    crate::environment::EFFECTS
        .iter()
        .filter(|effect| effect.name != "Autochange environment")
        .all(|effect| effect.level(config) == effect.calm)
}
/// An effect and its level in a few words: "Wind: Strong", or just "Heat wave"
/// when the level carries the effect's own name.
fn effect_text(effect: &crate::environment::Effect, config: &Config) -> String {
    let level = effect.levels[effect.level(config)];
    if level == effect.name {
        level.to_owned()
    } else {
        format!("{}: {}", effect.name, level)
    }
}
/// The world in a few words: "Calm world", or the effects away from calm,
/// such as "Ground: Rough, 8 cm · Hurdles: Low". The autochange level is not
/// part of it.
pub(super) fn world_summary(config: &Config) -> String {
    let parts: Vec<String> = crate::environment::EFFECTS
        .iter()
        .filter(|effect| effect.name != "Autochange environment")
        .filter(|effect| effect.level(config) != effect.calm)
        .map(|effect| effect_text(effect, config))
        .collect();
    if parts.is_empty() {
        "Calm world".to_owned()
    } else {
        parts.join(" · ")
    }
}
/// One effect as its name and a segmented bar of its levels. The lit segment
/// is the level in `config`. It is filled with `theme.stop_fill` when the
/// effect is away from calm and with `theme.armed_fill` at calm. The row is
/// waiting when `live`, the world that runs now, has another level of this
/// effect than `config` has. Then the name is drawn in `theme.cold` and a
/// `theme.cold` outline marks the level that still runs. Returns true when the
/// player picked another level, which is then set in `config`.
fn effect_row(
    ui: &mut egui::Ui,
    effect: &crate::environment::Effect,
    config: &mut Config,
    live: Option<&Config>,
    theme: Theme,
) -> bool {
    let level = effect.level(config);
    let away = level != effect.calm;
    // The level that runs now. It is `None` while there is no running world.
    let running = live.map(|live| effect.level(live));
    let waiting = running.is_some_and(|running| running != level);
    let color = if waiting {
        theme.cold
    } else if away {
        theme.accent
    } else {
        theme.ink
    };
    // The autochange row shows a short name.
    let label = if effect.name == "Autochange environment" {
        "Autochange"
    } else {
        effect.name
    };
    let mut picked = None;
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 0.;
        let (name_rect, name) =
            ui.allocate_exact_size(Vec2::new(EFFECT_NAME_WIDTH, LEVEL_HEIGHT), Sense::hover());
        ui.painter().text(
            name_rect.left_center(),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(16.5),
            color,
        );
        if let (true, Some(running)) = (waiting, running) {
            name.on_hover_text(format!(
                "Now {}. {} from the next generation.",
                effect.levels[running], effect.levels[level]
            ));
        } else {
            name.on_hover_text(format!("{}. {}", effect.name, effect.why));
        }
        ui.vertical(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(1., 1.);
                let font = FontId::proportional(14.);
                // A segment shows only the text of its level before the first
                // comma. The hover text has the whole level.
                let galleys: Vec<_> = effect
                    .levels
                    .iter()
                    .map(|text| {
                        let short = text.split(',').next().unwrap_or(text);
                        ui.painter()
                            .layout_no_wrap(short.to_owned(), font.clone(), theme.ink)
                    })
                    .collect();
                const PAD: f32 = 8.;
                let count = galleys.len().max(1) as f32;
                // The width of all segments side by side: each text with its
                // padding and 1 point between neighbors.
                let natural: f32 =
                    galleys.iter().map(|g| g.size().x + PAD).sum::<f32>() + count - 1.;
                let room = ui.available_width();
                // One line: the segments share the whole width. Too long for
                // one line, they keep their own width and wrap.
                let extra = if natural <= room {
                    ((room - natural) / count).floor()
                } else {
                    0.
                };
                let last = galleys.len().saturating_sub(1);
                for (i, galley) in galleys.into_iter().enumerate() {
                    let size = Vec2::new(galley.size().x + PAD + extra, LEVEL_HEIGHT);
                    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
                    let lit = i == level;
                    let hovered = response.hovered();
                    let fill = match (lit, away, hovered) {
                        (true, true, _) => theme.stop_fill,
                        (true, false, _) => theme.armed_fill,
                        (false, _, true) => ui.visuals().widgets.hovered.weak_bg_fill,
                        _ => ui.visuals().widgets.inactive.weak_bg_fill,
                    };
                    // Only the outer corners of the first and last segment are
                    // round, so the segments read as one bar.
                    let corner = egui::CornerRadius {
                        nw: if i == 0 { 3 } else { 0 },
                        sw: if i == 0 { 3 } else { 0 },
                        ne: if i == last { 3 } else { 0 },
                        se: if i == last { 3 } else { 0 },
                    };
                    ui.painter().rect_filled(rect, corner, fill);
                    ui.painter().rect_stroke(
                        rect,
                        corner,
                        Stroke::new(if lit { 2.5 } else { 1.5 }, theme.ink),
                        egui::StrokeKind::Inside,
                    );
                    // The outline of the level that still runs.
                    if waiting && running == Some(i) {
                        ui.painter().rect_stroke(
                            rect,
                            corner,
                            Stroke::new(3., theme.cold),
                            egui::StrokeKind::Inside,
                        );
                    }
                    let text_color = if lit && away {
                        theme.go_text
                    } else {
                        theme.ink
                    };
                    let at = rect.center() - galley.size() / 2.;
                    ui.painter()
                        .galley_with_override_text_color(at, galley, text_color);
                    let text = effect.levels[i];
                    let hover = if i == effect.calm {
                        format!("{text}. The calm world.")
                    } else {
                        format!("{text}. {}", effect.why)
                    };
                    if response.on_hover_text(hover).clicked() && i != level {
                        picked = Some(i);
                    }
                }
            });
        });
    });
    if let Some(i) = picked {
        effect.set_level(config, i);
    }
    picked.is_some()
}
