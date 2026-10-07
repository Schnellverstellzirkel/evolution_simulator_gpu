//! The event feed of the Overview tab. It lists the worker's events, the
//! records of the history and the champion's live record, newest first, and a
//! line may carry a Replay, Undo or Try it button. Three lines are hints that
//! offer a new world: after a stall, after a collapse of the clades and from a
//! wild island. A hint only offers, because the game never changes the world by
//! itself.

use super::{
    App,
    records::{live_record, world_records},
    text::species_name,
};
use crate::{
    config::Config,
    storage::Stats,
    theme::Theme,
    worker::{Command, EventKind, Snapshot},
};
use eframe::egui::{self, Color32, RichText};
use std::time::Instant;

/// What a line of the event feed lets the player do. The button of the line
/// carries it out.
#[derive(Clone, Copy)]
enum FeedAction {
    /// Replay the best creature of this history row (index into
    /// `Snapshot::history`).
    Replay(usize),
    /// Replay the champion now, whose record no history row holds yet.
    ReplayChampion,
    /// Bring back the creatures that catastrophes took (`Command::UndoMeteor`).
    Undo,
    /// Set this effect (index into `EFFECTS`) to this level.
    Try(usize, usize),
    /// Set the world of this wild island (index among the wild islands). Every
    /// other effect goes back to calm.
    Wild(usize),
}
/// Generations without a record in the live world after which the feed
/// suggests a harder one.
const STALL_GENERATIONS: u32 = 25;
/// The effective clades of one world must fall below their peak by more than
/// this share of it before the feed suggests a new world.
const COLLAPSE_SHARE: f32 = 0.25;
/// How many generations of one world the clade check looks back over.
const COLLAPSE_WINDOW: usize = 50;
/// The effects a hint offers, by `EFFECTS` name, in the order they are tried.
/// The first that can go one level higher wins. The stall hint and the collapse
/// hint both pick from it (`stall_suggestion`).
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
    /// The generation the line is dated to. The feed sorts by it.
    generation: u32,
    /// The words of the line.
    text: String,
    /// The color of the words.
    color: Color32,
    /// The button on this line, or `None` for a line with no button.
    action: Option<FeedAction>,
}
impl App {
    /// The lines of the event feed, newest first and at most 60: the worker's
    /// events, the records of the history, the champion's live record and the
    /// hints. It is empty before the first snapshot.
    fn feed_items(&self) -> Vec<FeedItem> {
        let Some(snapshot) = &self.snapshot else {
            return Vec::new();
        };
        let theme = self.theme();
        let mut items = Vec::new();
        Self::push_event_items(snapshot, theme, &mut items);
        let records = Self::push_record_items(snapshot, theme, &mut items);
        self.push_stall_item(snapshot, &records, theme, &mut items);
        self.push_collapse_item(snapshot, theme, &mut items);
        Self::push_wild_item(snapshot, theme, &mut items);
        // Each source added its lines oldest first, and the hints came last.
        // The sort is stable, so after the reverse the lines of one generation
        // also read newest first.
        items.reverse();
        items.sort_by_key(|item| std::cmp::Reverse(item.generation));
        items.truncate(60);
        items
    }
    /// A line for each event of the worker. A catastrophe has an Undo button
    /// while fossils exist. A world change or an autochange also gives the best
    /// distance before and after it, when the history holds the row of the
    /// generation before. The color is `theme.warn` for a catastrophe or a GPU
    /// event, `theme.accent` for a world change and `theme.muted` for any other
    /// event.
    fn push_event_items(snapshot: &Snapshot, theme: Theme, items: &mut Vec<FeedItem>) {
        let history = &snapshot.history;
        let row = |generation: u32| history.iter().rev().find(|s| s.generation == generation);
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
    }
    /// A line with a Replay button for each record of the history, and one for
    /// the champion's live record when `live_record` finds one. A line names
    /// the species of the record holder. Returns the history's records, as
    /// `world_records` gives them, for the stall check.
    fn push_record_items(
        snapshot: &Snapshot,
        theme: Theme,
        items: &mut Vec<FeedItem>,
    ) -> Vec<(usize, f32, bool)> {
        let history = &snapshot.history;
        let records = world_records(history);
        for &(index, best, first_in_world) in &records {
            let stats = &history[index];
            let name = stats
                .representatives
                .last()
                .map(species_name)
                .unwrap_or_default();
            items.push(FeedItem {
                generation: stats.generation,
                text: record_text(index == 0, first_in_world, best, &name),
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
            items.push(FeedItem {
                generation: snapshot.generation,
                text: record_text(record.first_ever, record.first_in_world, record.best, &name),
                color: theme.ink,
                action: Some(FeedAction::ReplayChampion),
            });
        }
        records
    }
    /// The hint to try a harder world, when the newest record is in the live
    /// world and `STALL_GENERATIONS` old or older, and the running generation
    /// has set no record. `records` is what `push_record_items` returned. The
    /// effect to try is the one `stall_suggestion` picks.
    fn push_stall_item(
        &self,
        snapshot: &Snapshot,
        records: &[(usize, f32, bool)],
        theme: Theme,
        items: &mut Vec<FeedItem>,
    ) {
        let history = &snapshot.history;
        // A stall: no record in this world for a while. The feed only suggests
        // a harder world, because the game never changes the world by itself.
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
    }
    /// The hint to try a new world when the effective clades of the archive
    /// collapsed. It looks at the newest `COLLAPSE_WINDOW` rows of the live
    /// world. It needs at least 10 of them, a peak above 4 clades and a newest
    /// row below the peak by more than `COLLAPSE_SHARE` of it. The effect to try
    /// is the one `stall_suggestion` picks.
    fn push_collapse_item(&self, snapshot: &Snapshot, theme: Theme, items: &mut Vec<FeedItem>) {
        let history = &snapshot.history;
        // A diversity collapse: the effective clades of the archive fell by
        // more than `COLLAPSE_SHARE` of their peak within `COLLAPSE_WINDOW`
        // generations of this world. The feed suggests a new world and never
        // sets it (Lehman and Miikkulainen, 2015: a change restarts radiation).
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
    }
    /// The offer of the world of the wild island whose migrants won the most
    /// hub cells, once they have won at least 3. The lowest island wins a tie.
    fn push_wild_item(snapshot: &Snapshot, theme: Theme, items: &mut Vec<FeedItem>) {
        let history = &snapshot.history;
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
    }
    /// Draws the event feed under the heading "What happened", in a scroll area
    /// of at most `height`. A line can have a button that replays a record
    /// holder, undoes a catastrophe or tries a suggested world. The click is
    /// carried out after the area is drawn: a replay selects the creature and
    /// shows the Overview tab, and a world is sent to the worker as new
    /// settings.
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
/// The words of a record line: the first generation of the game, the first best
/// in a new world, or a new record in the same world.
fn record_text(first_ever: bool, first_in_world: bool, best: f32, name: &str) -> String {
    if first_ever {
        format!("First generation: best {best:.2} m, {name}.")
    } else if first_in_world {
        format!("Best in the new world: {best:.2} m, {name}.")
    } else {
        format!("New record: {best:.2} m, {name}.")
    }
}
/// The first effect of `STALL_EFFECTS` that `config` has not set to its highest
/// level, as its index in `EFFECTS` and the next level up. It is `None` when all
/// of them are at their highest.
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
