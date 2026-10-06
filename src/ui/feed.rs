//! The event feed of the Overview tab: world changes, records, catastrophes and
//! the hints that suggest a harder world when evolution stalls.

use super::{
    App,
    records::{live_record, world_records},
    text::species_name,
};
use crate::{
    config::Config,
    storage::Stats,
    worker::{Command, EventKind},
};
use eframe::egui::{self, Color32, RichText};
use std::time::Instant;

/// What a line of the event feed lets the player do.
#[derive(Clone, Copy)]
enum FeedAction {
    /// Replay the best creature of this history row.
    Replay(usize),
    /// Replay the champion now, whose record no history row holds yet.
    ReplayChampion,
    /// Bring back creatures lost to catastrophes.
    Undo,
    /// Set this effect (index into `EFFECTS`) to this level.
    Try(usize, usize),
    /// Set the world of this wild island (index among the wild islands).
    Wild(usize),
}
/// Generations without a record before the feed suggests a new world.
const STALL_GENERATIONS: u32 = 25;
/// The feed suggests a new world when the effective clades fell by this share
/// within `COLLAPSE_WINDOW` generations of one world.
const COLLAPSE_SHARE: f32 = 0.25;
const COLLAPSE_WINDOW: usize = 50;
/// The effects a stall hint suggests, in order; the first that can go one
/// level harder wins.
const STALL_EFFECTS: [&str; 13] = [
    "Ground",
    "Hurdles",
    "Grip",
    "Slope",
    "Mud",
    "Brambles",
    "Gaps",
    "Wind",
    "Air",
    "Gravity",
    "Earthquake",
    "Heat wave",
    "Drought",
];
/// One line of the event feed.
struct FeedItem {
    generation: u32,
    text: String,
    color: Color32,
    action: Option<FeedAction>,
}
impl App {
    /// The lines of the event feed, newest first: the worker's events (world
    /// changes, autochange, catastrophes, saves) and the records in the history.
    fn feed_items(&self) -> Vec<FeedItem> {
        let Some(snapshot) = &self.snapshot else {
            return Vec::new();
        };
        let theme = self.theme();
        let history = &snapshot.history;
        let row = |generation: u32| history.iter().rev().find(|s| s.generation == generation);
        let mut items = Vec::new();
        for event in snapshot.events.iter() {
            let mut text = event.text.clone();
            let (color, action) = match event.kind {
                EventKind::Catastrophe => (
                    theme.warn,
                    (snapshot.fossils > 0).then_some(FeedAction::Undo),
                ),
                EventKind::World | EventKind::Autochange => {
                    if let Some(before) = event.generation.checked_sub(1).and_then(row) {
                        if let Some(after) = row(event.generation) {
                            text.push_str(&format!(
                                " Best {:.2} m before, {:.2} m after one generation.",
                                before.best, after.best
                            ));
                        } else if event.generation == snapshot.generation
                            && snapshot.live_best.is_finite()
                        {
                            text.push_str(&format!(
                                " Best {:.2} m before, {:.2} m so far in this generation.",
                                before.best, snapshot.live_best
                            ));
                        }
                    }
                    if event.kind == EventKind::Autochange {
                        text.insert_str(0, "Autochange: ");
                    }
                    (theme.accent, None)
                }
                EventKind::Gpu => (theme.warn, None),
                _ => (theme.muted, None),
            };
            items.push(FeedItem {
                generation: event.generation,
                text,
                color,
                action,
            });
        }
        let records = world_records(history);
        for &(index, best, first_in_world) in &records {
            let stats = &history[index];
            let name = stats
                .representatives
                .last()
                .map(species_name)
                .unwrap_or_default();
            let text = if index == 0 {
                format!("First generation: best {best:.2} m, {name}.")
            } else if first_in_world {
                format!("Best in the new world: {best:.2} m, {name}.")
            } else {
                format!("New record: {best:.2} m, {name}.")
            };
            items.push(FeedItem {
                generation: stats.generation,
                text,
                color: theme.ink,
                action: Some(FeedAction::Replay(index)),
            });
        }
        if let Some(record) = live_record(snapshot) {
            let name = snapshot
                .champion
                .as_ref()
                .map(|champion| species_name(&champion.0))
                .unwrap_or_default();
            let best = record.best;
            items.push(FeedItem {
                generation: snapshot.generation,
                text: if record.first_ever {
                    format!("First generation: best {best:.2} m, {name}.")
                } else if record.first_in_world {
                    format!("Best in the new world: {best:.2} m, {name}.")
                } else {
                    format!("New record: {best:.2} m, {name}.")
                },
                color: theme.ink,
                action: Some(FeedAction::ReplayChampion),
            });
        }
        // A stall: no record in this world for a while. The feed suggests a
        // harder world instead of changing the search silently.
        // Only a record of the live world counts: after a world change the old
        // records say nothing about a stall.
        if let (Some(last), Some(&(index, _, _))) = (history.last(), records.last())
            && !history[index].config.physics_differs(&snapshot.config)
        {
            let since = last.generation.saturating_sub(history[index].generation);
            if since >= STALL_GENERATIONS
                && live_record(snapshot).is_none()
                && let Some((effect, level)) = stall_suggestion(&self.config)
            {
                let effect_ref = &crate::environment::EFFECTS[effect];
                items.push(FeedItem {
                    generation: last.generation,
                    text: format!(
                        "No new record for {since} generations. A new world can open new ways of moving: try {} {}.",
                        effect_ref.name, effect_ref.levels[level]
                    ),
                    color: theme.accent,
                    action: Some(FeedAction::Try(effect, level)),
                });
            }
        }
        // A diversity collapse: the effective clades of the archive fell by a
        // quarter within 50 generations of this world. The feed suggests a
        // new world and never presses it (Lehman and Miikkulainen, 2015: a
        // change restarts radiation).
        if let Some(last) = history.last() {
            let window: Vec<&Stats> = history
                .iter()
                .rev()
                .take(COLLAPSE_WINDOW)
                .take_while(|h| !h.config.physics_differs(&snapshot.config))
                .collect();
            let peak = window.iter().map(|h| h.clades).fold(0.0f32, f32::max);
            if window.len() >= 10
                && peak > 4.0
                && last.clades < (1.0 - COLLAPSE_SHARE) * peak
                && let Some((effect, level)) = stall_suggestion(&self.config)
            {
                let effect_ref = &crate::environment::EFFECTS[effect];
                items.push(FeedItem {
                    generation: last.generation,
                    text: format!(
                        "Lineages are dying out: {:.1} effective clades, down from {peak:.1}. A new world can start a new radiation: try {} {}.",
                        last.clades, effect_ref.name, effect_ref.levels[level]
                    ),
                    color: theme.accent,
                    action: Some(FeedAction::Try(effect, level)),
                });
            }
        }
        // The wild island whose migrants took the most hub cells: its world
        // breeds bodies that also do well in yours. The feed offers it as a
        // world to try (Wang et al., 2019, POET transfer) and never sets it.
        if let Some(last) = history.last()
            && let Some((island, &wins)) = snapshot
                .wild_wins
                .iter()
                .enumerate()
                .skip(crate::qd::MAIN_ISLANDS)
                .max_by_key(|&(i, &w)| (w, std::cmp::Reverse(i)))
            && wins >= 3
        {
            let w = island - crate::qd::MAIN_ISLANDS;
            let worlds = crate::environment::wild_levels(snapshot.config.seed);
            if let Some(levels) = worlds.get(w) {
                items.push(FeedItem {
                    generation: last.generation,
                    text: format!(
                        "Wild island W{} sent the most creatures that won hub cells ({wins}). Its world: {}.",
                        w + 1,
                        crate::environment::wild_name(levels)
                    ),
                    color: theme.accent,
                    action: Some(FeedAction::Wild(w)),
                });
            }
        }
        // Newest first; the sort is stable, so events of one generation keep
        // their order.
        items.reverse();
        items.sort_by_key(|item| std::cmp::Reverse(item.generation));
        items.truncate(60);
        items
    }
    /// The event feed: what happened, newest first, with a button to replay
    /// a record holder or undo a catastrophe.
    pub(super) fn feed(&mut self, ui: &mut egui::Ui, height: f32) {
        let theme = self.theme();
        crate::theme::heading(ui, "What happened", theme);
        let items = self.feed_items();
        let mut chosen = None;
        egui::ScrollArea::vertical()
            .id_salt("event_feed")
            .max_height(height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if items.is_empty() {
                    ui.label(
                        RichText::new("Records, world changes and catastrophes appear here.")
                            .small()
                            .color(theme.muted),
                    );
                }
                for item in &items {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.y = 2.;
                        ui.label(
                            RichText::new(format!("Gen {}", item.generation))
                                .small()
                                .color(theme.muted),
                        );
                        ui.label(RichText::new(&item.text).small().color(item.color));
                        if let Some(action) = item.action {
                            let label = match action {
                                FeedAction::Replay(_) | FeedAction::ReplayChampion => "Replay",
                                FeedAction::Undo => "Undo",
                                FeedAction::Try(..) | FeedAction::Wild(_) => "Try it",
                            };
                            if ui.small_button(label).clicked() {
                                chosen = Some(action);
                            }
                        }
                    });
                }
            });
        match chosen {
            Some(FeedAction::Replay(index)) => self.replay_history_holder(index),
            Some(FeedAction::ReplayChampion) => self.replay_champion(),
            Some(FeedAction::Undo) => self.worker.send(Command::UndoMeteor),
            Some(FeedAction::Try(effect, level)) => {
                crate::environment::EFFECTS[effect].set_level(&mut self.config, level);
                self.worker.send(Command::Configure(self.config.clone()));
                self.config_sent = Some(Instant::now());
            }
            Some(FeedAction::Wild(w)) => {
                let worlds = crate::environment::wild_levels(self.config.seed);
                if let Some(levels) = worlds.get(w) {
                    for effect in crate::environment::EFFECTS
                        .iter()
                        .filter(|e| e.name != "Autochange environment")
                    {
                        effect.set_level(&mut self.config, effect.calm);
                    }
                    for &(e, level) in levels {
                        crate::environment::EFFECTS[e].set_level(&mut self.config, level);
                    }
                    self.worker.send(Command::Configure(self.config.clone()));
                    self.config_sent = Some(Instant::now());
                }
            }
            None => {}
        }
    }
}
/// An effect and the next harder level to try when evolution stalls.
fn stall_suggestion(config: &Config) -> Option<(usize, usize)> {
    STALL_EFFECTS.iter().find_map(|name| {
        let index = crate::environment::EFFECTS
            .iter()
            .position(|effect| effect.name == *name)?;
        let effect = &crate::environment::EFFECTS[index];
        let level = effect.level(config);
        (level + 1 < effect.levels.len()).then_some((index, level + 1))
    })
}
