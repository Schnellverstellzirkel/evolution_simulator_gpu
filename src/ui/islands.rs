//! The Islands view of the Ways of moving tab. It paints one card for each
//! main island, which means the four isolated islands and the hub, and one
//! tile for each wild island. A card shows the island's best creature, its top
//! elites, the share of its elites that each emitter bred, its nurseries and
//! its migration. `population` calls `islands_view`, and a click on a creature
//! replays it.

use super::{
    App,
    scene::thumbnail,
    text::{number, species_name},
};
use crate::{config::Config, evolution::Creature, theme::Theme};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Vec2};

/// Space between island cards.
const ISLAND_GAP: f32 = 10.;
/// Height of an island card.
const ISLAND_HEIGHT: f32 = 346.;
/// Words for the emitter shares of an island's elites, in `Emitter::ALL` order.
/// `IslandSummary::origins` holds the counts. "New" also counts the graduates
/// of the island's nurseries.
const ORIGIN_SHORT: [&str; 4] = ["Tuned", "Reshaped", "Novel", "New"];
/// Their colors: amber, rust, cold blue and olive. `EMITTER_TINT` in
/// `schematic.rs` has the same hues.
const ORIGIN_COLORS: [Color32; 4] = [
    Color32::from_rgb(222, 160, 60),
    Color32::from_rgb(178, 92, 58),
    Color32::from_rgb(96, 154, 196),
    Color32::from_rgb(132, 140, 76),
];
impl App {
    /// The Islands view. It draws the main islands as cards in two columns,
    /// the four isolated islands first and the hub last, and the wild islands
    /// as tiles below them. Each card has a fixed size and fixed places for its
    /// parts, so numbers change without moving anything. A click on a creature
    /// replays it in the world it was scored in. The "How evolution works"
    /// button opens the schematic. The "Strangest body" button replays the
    /// main islands' elite whose body is the most unlike the others.
    pub(super) fn islands_view(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "Islands never mix. The hub gets copies every {} generations.",
                    crate::storage::MIGRATION_INTERVAL
                ))
                .color(theme.muted),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("How evolution works")
                    .on_hover_text("Shows how the islands, the emitters and migration fit together")
                    .clicked()
                {
                    self.schematic_open = true;
                }
            });
        });
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        if snapshot
            .islands
            .iter()
            .all(|island| island.leader.is_none())
        {
            ui.add_space(8.);
            ui.label(
                RichText::new(
                    "The islands fill as the first creatures are kept. Their creatures appear here.",
                )
                .color(theme.muted),
            );
            return;
        }
        let config = snapshot.config.clone();
        let generation = snapshot.generation;
        let strangest = snapshot.strangest.clone();
        if let Some(creature) = strangest
            && ui
                .button("Strangest body")
                .on_hover_text(
                    "Replay the island creature whose body is the most unlike the others",
                )
                .clicked()
        {
            self.select(creature, config.clone());
            return;
        }
        let Some(snapshot) = self.snapshot.as_ref() else {
            return;
        };
        let islands = snapshot.islands.clone();
        let wild_wins = snapshot.wild_wins.clone();
        let migration = snapshot.migration.clone();
        let shown = self.playback.as_ref().map(|p| p.creature.id);
        let width = (ui.available_width() - ISLAND_GAP) / 2.;
        let mut selected = None;
        let mut wild_pick = None;
        egui::ScrollArea::vertical()
            .id_salt("islands_grid")
            .show(ui, |ui| {
                let main = islands.len().min(crate::qd::MAIN_ISLANDS);
                for pair in islands[..main].chunks(2).enumerate() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = ISLAND_GAP;
                        for (offset, island) in pair.1.iter().enumerate() {
                            let index = pair.0 * 2 + offset;
                            let (rect, _) = ui.allocate_exact_size(
                                Vec2::new(width, ISLAND_HEIGHT),
                                Sense::hover(),
                            );
                            let hit = paint_island(
                                ui,
                                rect,
                                index,
                                island,
                                migration.as_ref(),
                                generation,
                                shown,
                                theme,
                            );
                            if let Some(creature) = hit {
                                selected = Some(creature);
                            }
                        }
                    });
                    ui.add_space(ISLAND_GAP);
                }
                if islands.len() > main
                    && let Some(pick) =
                        wild_tiles(ui, &islands[main..], &wild_wins, &config, shown, &theme)
                {
                    wild_pick = Some(pick);
                }
            });
        if let Some((creature, world)) = wild_pick {
            self.select(creature, world);
        } else if let Some(creature) = selected {
            self.select(creature, config);
        }
    }
}
/// What an island card says about its nurseries: their size and best
/// distance, when they graduate next, and what the last graduation kept.
fn nursery_lines(island: &crate::worker::IslandSummary, generation: u32) -> [String; 2] {
    let every = crate::qd::NURSERY_GENERATIONS;
    let next = (generation / every + 1) * every;
    let first = if island.nursery == 0 {
        format!("Nurseries empty. Next: gen {next}")
    } else {
        format!(
            "Nurseries {} bodies, best {:.1} m. Next: gen {next}",
            island.nursery, island.nursery_best
        )
    };
    let g = island.graduation;
    let second = if g.generation == 0 {
        "No graduates yet this session".to_owned()
    } else {
        format!(
            "Gen {}: {} of {} graduates kept, {} in all",
            g.generation, g.kept, g.sent, g.kept_total
        )
    };
    [first, second]
}
/// What an island card says about migration: the last exchange, or when the
/// next one comes.
fn migration_lines(
    migration: Option<&crate::worker::MigrationSummary>,
    island: usize,
    generation: u32,
) -> [String; 2] {
    let next =
        (generation / crate::storage::MIGRATION_INTERVAL + 1) * crate::storage::MIGRATION_INTERVAL;
    let hub = island == crate::storage::hub_island();
    match migration.filter(|m| !m.exchange.is_empty()) {
        Some(m) if hub => {
            let (got, kept) = m.hub_received();
            [
                format!(
                    "Gen {}: got {got} copies from the islands, kept {kept}",
                    m.generation
                ),
                format!("Sends nothing back. Next: gen {next}"),
            ]
        }
        Some(m) => {
            let (sent, kept) = m.exchange.get(island).copied().unwrap_or((0, 0));
            [
                format!(
                    "Gen {}: copied {sent} to the hub, it kept {kept}",
                    m.generation
                ),
                format!("Receives no migrants. Next: gen {next}"),
            ]
        }
        None if hub => [
            "No copies yet this session".to_owned(),
            format!("Copies arrive at generation {next}"),
        ],
        None => [
            "Isolated: receives no migrants".to_owned(),
            format!("Copies go to the hub at generation {next}"),
        ],
    }
}
/// Percent shares that add to 100 (largest remainder), so the legend never
/// reads 99 or 101.
fn percent_shares(counts: &[usize]) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return vec![0; counts.len()];
    }
    let mut shares: Vec<usize> = counts.iter().map(|&c| c * 100 / total).collect();
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(counts[i] * 100 % total));
    let missing = 100 - shares.iter().sum::<usize>();
    for &i in order.iter().take(missing) {
        shares[i] += 1;
    }
    shares
}
/// "Island 1" to "Island 4" for the isolated islands, "Hub" for the hub.
pub(crate) fn island_name(index: usize) -> String {
    if index == crate::storage::hub_island() {
        "Hub".to_owned()
    } else {
        format!("Island {}", index + 1)
    }
}
/// Paints the card of island `index` in `rect` and returns the creature the
/// player clicked. `migration` is the last migration, `generation` is the
/// generation running now, and `shown` is the id of the creature on screen,
/// whose thumbnail is lit.
#[allow(clippy::too_many_arguments)]
fn paint_island(
    ui: &mut egui::Ui,
    rect: Rect,
    index: usize,
    island: &crate::worker::IslandSummary,
    migration: Option<&crate::worker::MigrationSummary>,
    generation: u32,
    shown: Option<u64>,
    theme: Theme,
) -> Option<Creature> {
    let painter = ui.painter().clone();
    crate::theme::plate(&painter, rect, theme, theme.card, false);
    let at = |x: f32, y: f32| rect.left_top() + Vec2::new(x, y);
    crate::theme::caps_text(
        &painter,
        at(12., 12.),
        Align2::LEFT_TOP,
        &island_name(index),
        14.,
        theme.ink,
    );
    painter.text(
        rect.right_top() + Vec2::new(-12., 12.),
        Align2::RIGHT_TOP,
        format!("{} ways of moving", number(island.moves)),
        FontId::proportional(14.5),
        theme.muted,
    );
    let mut clicked = None;
    // The best creature, left; its distance under it.
    let lead_rect = Rect::from_min_size(at(12., 36.), Vec2::new(rect.width() * 0.5 - 18., 104.));
    let best_text = if island.best.is_finite() {
        format!("Best {:.2} m", island.best)
    } else {
        "Empty".to_owned()
    };
    if let Some(leader) = &island.leader {
        let response = ui.interact(
            lead_rect,
            ui.id().with(("island_leader", index)),
            Sense::click(),
        );
        let lit = response.hovered() || shown == Some(leader.id);
        painter.rect_filled(
            lead_rect,
            6,
            if lit { theme.card_hover } else { theme.canvas },
        );
        thumbnail(&painter, leader, lead_rect);
        if response.clicked() {
            clicked = Some(leader.clone());
        }
        response.on_hover_text(format!(
            "{}\n{} nodes, {} muscles\nClick to replay",
            species_name(leader),
            leader.nodes.len(),
            leader.muscles.len()
        ));
    }
    painter.text(
        lead_rect.left_bottom() + Vec2::new(0., 6.),
        Align2::LEFT_TOP,
        best_text,
        FontId::proportional(15.),
        theme.accent,
    );
    // The fastest elites, right, one row each. The first is the leader.
    let list_left = lead_rect.right() + 12.;
    painter.text(
        Pos2::new(list_left, 38.0 + rect.top()),
        Align2::LEFT_TOP,
        "Top elites",
        FontId::proportional(14.),
        theme.muted,
    );
    for (row, (distance, creature)) in island.top.iter().enumerate() {
        let row_rect = Rect::from_min_size(
            Pos2::new(list_left, rect.top() + 56. + row as f32 * 28.),
            Vec2::new(rect.right() - 12. - list_left, 26.),
        );
        let response = ui.interact(
            row_rect,
            ui.id().with(("island_top", index, row)),
            Sense::click(),
        );
        let lit = response.hovered() || shown == Some(creature.id);
        painter.rect_filled(
            row_rect,
            4,
            if lit { theme.card_hover } else { theme.canvas },
        );
        thumbnail(
            &painter,
            creature,
            Rect::from_min_size(row_rect.left_top(), Vec2::new(40., 26.)),
        );
        painter.text(
            row_rect.left_center() + Vec2::new(46., 0.),
            Align2::LEFT_CENTER,
            format!("{distance:.2} m"),
            FontId::proportional(14.5),
            theme.ink,
        );
        if response.clicked() {
            clicked = Some(creature.clone());
        }
        response.on_hover_text(format!("{}\nClick to replay", species_name(creature)));
    }
    // Who bred the island's elites.
    painter.text(
        at(12., 168.),
        Align2::LEFT_TOP,
        "Bred by",
        FontId::proportional(14.),
        theme.muted,
    );
    let bar = Rect::from_min_size(at(12., 184.), Vec2::new(rect.width() - 24., 10.));
    painter.rect_filled(bar, 3, theme.canvas);
    let shares = percent_shares(&island.origins);
    let total: usize = island.origins.iter().sum();
    let mut x = bar.left();
    for (i, &count) in island.origins.iter().enumerate() {
        if total == 0 || count == 0 {
            continue;
        }
        let w = bar.width() * count as f32 / total as f32;
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(x, bar.top()), Vec2::new(w, bar.height())),
            0,
            ORIGIN_COLORS[i],
        );
        x += w;
    }
    let legend_width = (rect.width() - 24.) / 2.;
    for i in 0..island.origins.len() {
        let cell = at(
            12. + (i % 2) as f32 * legend_width,
            200. + (i / 2) as f32 * 16.,
        );
        painter.rect_filled(
            Rect::from_min_size(cell + Vec2::new(0., 3.), Vec2::splat(9.)),
            1,
            ORIGIN_COLORS[i],
        );
        painter.text(
            cell + Vec2::new(14., 0.),
            Align2::LEFT_TOP,
            format!("{} {}%", ORIGIN_SHORT[i], shares[i]),
            FontId::proportional(14.),
            theme.ink,
        );
    }
    let lines: Vec<String> = nursery_lines(island, generation)
        .into_iter()
        .chain(migration_lines(migration, index, generation))
        .collect();
    let mut y = 238.;
    for line in &lines {
        let galley = painter.layout(
            line.clone(),
            FontId::proportional(14.),
            theme.muted,
            rect.width() - 24.,
        );
        let height = galley.size().y;
        painter.galley(at(12., y), galley, theme.muted);
        y += height + 2.;
    }
    clicked
}
/// The wild islands: a line about their worlds, a line that ranks the effects
/// in the worlds of the islands whose migrants took hub cells (once any has),
/// and a grid of small tiles. A tile is colored by its island's best distance
/// against the best of all wild islands, and its hover text names the island's
/// world. `wins` is `Snapshot::wild_wins`. A click returns the island's leader
/// with its world, so the replay runs where the score came from.
fn wild_tiles(
    ui: &mut egui::Ui,
    wild: &[crate::worker::IslandSummary],
    wins: &[u32],
    config: &Config,
    shown: Option<u64>,
    theme: &Theme,
) -> Option<(Creature, Config)> {
    let levels = crate::environment::wild_levels(config.seed);
    ui.add_space(6.);
    ui.label(
        RichText::new(format!(
            "Wild islands: {} worlds of their own. Each sends its best to the hub, where they run again in your world.",
            wild.len()
        ))
        .color(theme.muted),
    );
    // For each effect, the hub cells that migrants from the wild worlds
    // holding it took (Wang et al., 2019, POET).
    let mut by_effect = vec![0u32; crate::environment::EFFECTS.len()];
    for (w, levels) in levels.iter().enumerate() {
        let won = wins.get(crate::qd::MAIN_ISLANDS + w).copied().unwrap_or(0);
        for &(e, _) in levels {
            by_effect[e] += won;
        }
    }
    let mut ranked: Vec<(usize, u32)> = by_effect
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, n)| n > 0)
        .collect();
    ranked.sort_by_key(|&(e, n)| (std::cmp::Reverse(n), e));
    if !ranked.is_empty() {
        ui.label(
            RichText::new(format!(
                "Effects in the worlds of the hub winners: {}",
                ranked
                    .iter()
                    .take(6)
                    .map(|&(e, n)| format!("{} {n}", crate::environment::EFFECTS[e].name))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .small(),
        )
        .on_hover_text(
            "Each wild migrant that took a hub cell counts for every effect of its island's world",
        );
    }
    ui.add_space(4.);
    let top = wild
        .iter()
        .map(|w| w.best)
        .filter(|b| b.is_finite())
        .fold(0.0f32, f32::max)
        .max(0.01);
    let columns = (ui.available_width() / 96.).floor().max(4.) as usize;
    let width = (ui.available_width() - (columns - 1) as f32 * 4.) / columns as f32;
    let mut picked = None;
    for (row, chunk) in wild.chunks(columns).enumerate() {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.;
            for (offset, island) in chunk.iter().enumerate() {
                let w = row * columns + offset;
                let (rect, response) =
                    ui.allocate_exact_size(Vec2::new(width, 40.), Sense::click());
                let share = if island.best.is_finite() {
                    (island.best / top).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let fill = theme.panel.lerp_to_gamma(theme.accent, 0.15 + 0.6 * share);
                let painter = ui.painter();
                painter.rect_filled(rect, 4., fill);
                let lit = response.hovered()
                    || island.leader.as_ref().is_some_and(|c| Some(c.id) == shown);
                if lit {
                    painter.rect_stroke(
                        rect,
                        4.,
                        egui::Stroke::new(2., theme.ink),
                        egui::StrokeKind::Inside,
                    );
                }
                let best = if island.best.is_finite() {
                    format!("{:.1} m", island.best)
                } else {
                    "empty".into()
                };
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("W{}\n{best}", w + 1),
                    egui::FontId::proportional(14.),
                    theme.ink,
                );
                let name = levels
                    .get(w)
                    .map(|l| crate::environment::wild_name(l))
                    .unwrap_or_default();
                let won = wins.get(crate::qd::MAIN_ISLANDS + w).copied().unwrap_or(0);
                let response = response.on_hover_text(format!(
                    "Wild island {}: {name}\nBest {best}, {} cells, {} in its nurseries, {won} hub cells won",
                    w + 1,
                    island.cells,
                    island.nursery
                ));
                if response.clicked()
                    && let (Some(leader), Some(l)) = (&island.leader, levels.get(w))
                {
                    picked = Some((leader.clone(), crate::environment::wild_world(config, l)));
                }
            }
        });
        ui.add_space(4.);
    }
    picked
}
#[cfg(test)]
mod island_view_tests {
    use super::*;

    #[test]
    fn origin_shares_always_add_to_100() {
        assert_eq!(percent_shares(&[0, 0, 0, 0]), vec![0; 4]);
        for counts in [[1, 1, 1, 0], [7, 3, 3, 1], [5, 0, 0, 0], [1, 2, 4, 8]] {
            let shares = percent_shares(&counts);
            assert_eq!(shares.iter().sum::<usize>(), 100, "{counts:?}");
            for (share, count) in shares.iter().zip(counts) {
                assert_eq!(count == 0, *share == 0);
            }
        }
    }
}
