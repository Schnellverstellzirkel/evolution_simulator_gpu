//! The Race tab: archive elites and the player's picks running side by side.

use super::{
    App,
    playback::{FrameMarks, Playback},
    scene::draw_creature,
    text::species_name,
    viewport::fit_zoom,
    widgets::speed_picker,
};
use crate::{
    physics,
    theme::{
        GAP_L, Theme,
        scene::{FALLEN, GROUND_INK},
    },
};
use eframe::egui::{self, Align2, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::time::Duration;

/// One archive elite running in the race view.
pub(super) struct RaceLane {
    /// Why the creature runs: its archive rank, "champion" or "your pick".
    label: String,
    pub(super) playback: Playback,
}
/// Creatures the player sends to the race with "Race it", at most this many
/// beside the champion.
pub(super) const RACE_PICKS: usize = 4;
impl App {
    /// Builds race lanes from the top archive cards once the first page arrives.
    pub(super) fn maybe_build_race(&mut self) {
        if !self.race_pending {
            return;
        }
        if self.race_picks.is_empty() {
            // The top five come with a ranked archive; ask again while none
            // has arrived with creatures in it.
            let kept = self.snapshot.as_ref().is_some_and(|s| s.archive_size > 0);
            let waiting = self
                .cards_requested
                .is_some_and(|at| at.elapsed() < Duration::from_secs(2));
            if kept && !waiting {
                self.request_cards();
            }
            return;
        }
        // The player's picks against the champion.
        let mut lanes: Vec<RaceLane> = Vec::new();
        if let Some((creature, config)) = self.champion()
            && self
                .race_picks
                .iter()
                .all(|(pick, _)| pick.id != creature.id)
        {
            lanes.push(RaceLane {
                label: "champion".to_owned(),
                playback: Playback::new(creature, config),
            });
        }
        for (creature, config) in &self.race_picks {
            lanes.push(RaceLane {
                label: "your pick".to_owned(),
                playback: Playback::new(creature.clone(), config.clone()),
            });
        }
        lanes.sort_by(|a, b| b.playback.distance.total_cmp(&a.playback.distance));
        self.race = lanes;
        self.race_pending = false;
        self.race_camera = 0.0;
    }
    /// The top five kept creatures of a freshly ranked archive race.
    pub(super) fn build_top_race(&mut self, list: &crate::worker::CardList) {
        let mut lanes: Vec<RaceLane> = list
            .cards
            .iter()
            .filter(|card| card.descriptor.is_some() && card.score.is_finite())
            .take(5)
            .map(|card| RaceLane {
                label: format!("archive rank {}", card.rank + 1),
                playback: {
                    let (creature, config) = card.replay_of(&list.config);
                    Playback::new(creature, config)
                },
            })
            .collect();
        // Lanes run in the order their replays finish, so the standings end
        // the way the lanes are listed.
        lanes.sort_by(|a, b| b.playback.distance.total_cmp(&a.playback.distance));
        if lanes.is_empty() {
            return;
        }
        self.race = lanes;
        self.race_pending = false;
        self.race_camera = 0.0;
    }
    /// Clears the race and asks for a fresh set of top elites.
    pub(super) fn restart_race(&mut self) {
        self.race.clear();
        self.race_pending = true;
        self.race_camera = 0.0;
        if self.race_picks.is_empty() {
            self.request_cards();
        }
    }
    /// The top archived elites running side by side, with live standings.
    pub(super) fn race_view(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        self.race_header(ui, theme);
        self.race_controls(ui);
        if self.race.is_empty() {
            self.race_empty_note(ui, theme);
            return;
        }
        let distances: Vec<f32> = self
            .race
            .iter()
            .map(|lane| lane.playback.current_distance())
            .collect();
        let leader = distances
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |(i, _)| i);
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), ui.available_height().max(260.)),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        let board_width = 250.0_f32.min(rect.width() * 0.3);
        let lanes_rect = Rect::from_min_max(
            rect.min,
            Pos2::new(rect.right() - board_width - 12., rect.bottom()),
        );
        // The default zoom follows the lanes' median body height, and the
        // tallest body still has to fit its lane.
        let lane_height = lanes_rect.height() / self.race.len().max(1) as f32;
        let mut heights: Vec<f32> = self.race.iter().map(|lane| lane.playback.height).collect();
        heights.sort_by(f32::total_cmp);
        let median = heights.get(heights.len() / 2).copied().unwrap_or(1.0);
        let tallest = heights.last().copied().unwrap_or(1.0);
        let zoom = fit_zoom(median, lane_height)
            .min(lane_height * 0.8 / tallest.max(0.1))
            .clamp(34.0, 300.0);
        let visible = lanes_rect.width() / zoom;
        // The leader's averaged center of mass, so its stride does not shake
        // the view; the easing below smooths a change of leader.
        // The start line sits a little in from the lane's left edge.
        let target = (self.race[leader].playback.camera_x() - visible * 0.6).max(-visible * 0.08);
        let dt = ui.ctx().input(|i| i.stable_dt).clamp(0.0, 0.1);
        self.race_camera += (target - self.race_camera) * (dt * 4.0).min(1.0);
        let camera = self.race_camera;
        let scene = RaceScene {
            painter: &painter,
            lanes_rect,
            lane_height,
            zoom,
            camera,
            visible,
            leader,
            distances: &distances,
            theme,
        };
        for (i, lane) in self.race.iter().enumerate() {
            Self::paint_race_lane(i, lane, &scene);
        }
        self.paint_race_standings(&painter, rect, lanes_rect, &distances, theme);
    }
    /// The title row: the New race or Top five button and what it means.
    fn race_header(&mut self, ui: &mut egui::Ui, theme: Theme) {
        ui.horizontal(|ui| {
            ui.heading("Race");
            ui.label(
                RichText::new("The fastest kept creatures run their trials side by side.")
                    .color(theme.muted),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.race_picks.is_empty() {
                    if ui
                        .button("New race")
                        .on_hover_text("Take the current top five kept creatures")
                        .clicked()
                    {
                        self.restart_race();
                    }
                } else {
                    if ui
                        .button("Top five")
                        .on_hover_text("Forget your picks and race the top five kept creatures")
                        .clicked()
                    {
                        self.race_picks.clear();
                        self.restart_race();
                    }
                    ui.label(
                        RichText::new(
                            "Your picks against the champion. Race it under any replay adds one.",
                        )
                        .small()
                        .color(theme.muted),
                    );
                }
            });
        });
    }
    /// The Play and Replay buttons and the speed menu.
    fn race_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button(if self.playing {
                    "Pause  (K)"
                } else {
                    "Play  (K)"
                })
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("Replay").clicked() {
                for lane in &mut self.race {
                    lane.playback.reset();
                }
                self.race_camera = 0.0;
            }
            speed_picker(ui, &mut self.speed, "race_speed");
        });
    }
    /// What the tab says while no lane runs.
    fn race_empty_note(&self, ui: &mut egui::Ui, theme: Theme) {
        ui.add_space(GAP_L);
        let waiting = self.race_pending
            && self
                .snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.archive_size > 0);
        ui.label(
            RichText::new(if waiting {
                "Loading the fastest kept creatures…"
            } else {
                "No archived creatures yet. Run a generation, then start a new race."
            })
            .color(theme.muted),
        );
    }
    /// One lane: its world strip, ruler, creature, readouts and frame.
    fn paint_race_lane(i: usize, lane: &RaceLane, scene: &RaceScene) {
        let painter = scene.painter;
        let lanes_rect = scene.lanes_rect;
        let lane_height = scene.lane_height;
        let zoom = scene.zoom;
        let camera = scene.camera;
        let visible = scene.visible;
        let leader = scene.leader;
        let distances = scene.distances;
        let theme = scene.theme;
        let lane_rect = Rect::from_min_max(
            Pos2::new(
                lanes_rect.left(),
                lanes_rect.top() + i as f32 * lane_height + 2.,
            ),
            Pos2::new(
                lanes_rect.right(),
                lanes_rect.top() + (i + 1) as f32 * lane_height - 2.,
            ),
        );
        let is_leader = i == leader;
        // Each lane is a strip of the world: its sky and skyline over a
        // street, the leader's lane framed in orange.
        let ground = lane_rect.bottom() - 16.;
        let lane_painter = painter.with_clip_rect(lane_rect);
        let lane_config = &lane.playback.config;
        let clock = lane.playback.tick as f32 / physics::rate() as f32;
        crate::world_fx::backdrop(
            &lane_painter,
            lane_rect,
            ground,
            camera * zoom + i as f32 * 900.,
            clock,
            lane_config,
            None,
        );
        let span = [
            Pos2::new(lane_rect.left(), ground),
            Pos2::new(lane_rect.right(), ground),
        ];
        crate::world_fx::ground_body(
            &lane_painter,
            lane_rect,
            lane_config,
            &span,
            &[camera, camera + lane_rect.width() / zoom],
            zoom,
        );
        // A tick about every 150 px: 0.5, 1, 2, 5 or 10 m.
        let step = [0.5f32, 1.0, 2.0, 5.0, 10.0]
            .into_iter()
            .find(|step| step * zoom >= 150.0)
            .unwrap_or(10.0);
        let mut x = (camera / step).ceil() * step;
        while x <= camera + visible {
            let px = lane_rect.left() + (x - camera) * zoom;
            lane_painter.line_segment(
                [
                    Pos2::new(px, ground),
                    Pos2::new(px, lane_rect.bottom() - 5.),
                ],
                Stroke::new(1., GROUND_INK),
            );
            lane_painter.line_segment(
                [Pos2::new(px, lane_rect.top()), Pos2::new(px, ground)],
                Stroke::new(1., crate::theme::scene::GRID),
            );
            if i == 0 {
                lane_painter.text(
                    Pos2::new(px + 3., ground - 2.),
                    Align2::LEFT_BOTTOM,
                    if step < 1.0 {
                        format!("{x:.1} m")
                    } else {
                        format!("{x:.0} m")
                    },
                    FontId::proportional(14.5),
                    GROUND_INK,
                );
            }
            x += step;
        }
        let origin = Pos2::new(lane_rect.left() - camera * zoom, ground);
        let playback = &lane.playback;
        let marks = FrameMarks::of(playback);
        draw_creature(
            &lane_painter,
            &playback.nodes,
            &playback.creature,
            origin,
            zoom,
            &marks,
        );
        crate::theme::vignette(&lane_painter, lane_rect, 0.35);
        Self::paint_race_lane_hud(&lane_painter, i, lane, lane_rect, is_leader, distances);
        painter.rect_stroke(
            lane_rect,
            2,
            Stroke::new(
                if is_leader { 2. } else { 1. },
                if is_leader {
                    theme.accent
                } else {
                    theme.card_border
                },
            ),
            egui::StrokeKind::Inside,
        );
    }
    /// A lane's readouts: the creature's name and finish, the live distance and
    /// how it ended.
    fn paint_race_lane_hud(
        lane_painter: &egui::Painter,
        i: usize,
        lane: &RaceLane,
        lane_rect: Rect,
        is_leader: bool,
        distances: &[f32],
    ) {
        let playback = &lane.playback;
        use crate::theme::{
            HudLine, hud_block,
            scene::{HUD, HUD_DIM},
        };
        hud_block(
            lane_painter,
            lane_rect.left_top() + Vec2::splat(6.),
            Align2::LEFT_TOP,
            &[
                HudLine::text(
                    format!("{}. {}", i + 1, species_name(&lane.playback.creature)),
                    15.,
                    if is_leader {
                        HUD
                    } else {
                        crate::theme::scene::HUD_INK
                    },
                ),
                HudLine::text(
                    format!(
                        "finishes at {:.2} m · {}",
                        lane.playback.distance, lane.label
                    ),
                    12.,
                    HUD_DIM,
                ),
            ],
        );
        let mut right = vec![HudLine::value(
            format!("{:.2} m", distances[i]),
            18.,
            if is_leader {
                HUD
            } else {
                crate::theme::scene::HUD_INK
            },
        )];
        if playback.fallen().is_some() {
            right.push(HudLine::text(
                playback.ending.short().to_owned(),
                13.,
                FALLEN,
            ));
        }
        hud_block(
            lane_painter,
            lane_rect.right_top() + Vec2::new(-6., 6.),
            Align2::RIGHT_TOP,
            &right,
        );
    }
    /// The standings board beside the lanes.
    fn paint_race_standings(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        lanes_rect: Rect,
        distances: &[f32],
        theme: Theme,
    ) {
        let board = Rect::from_min_max(
            Pos2::new(lanes_rect.right() + 12., rect.top()),
            rect.right_bottom(),
        );
        crate::theme::plate(painter, board, theme, theme.card, false);
        crate::theme::caps_text(
            painter,
            board.left_top() + Vec2::new(10., 10.),
            Align2::LEFT_TOP,
            "Standings",
            13.5,
            theme.muted,
        );
        let mut order: Vec<usize> = (0..self.race.len()).collect();
        order.sort_by(|&a, &b| distances[b].total_cmp(&distances[a]));
        for (place, &i) in order.iter().enumerate() {
            let y = board.top() + 34. + place as f32 * 26.;
            if y > board.bottom() - 26. {
                break;
            }
            let lane = &self.race[i];
            painter.text(
                Pos2::new(board.left() + 10., y),
                Align2::LEFT_CENTER,
                format!("{}. {}", place + 1, species_name(&lane.playback.creature)),
                FontId::proportional(14.),
                if place == 0 { theme.accent } else { theme.ink },
            );
            painter.text(
                Pos2::new(board.right() - 10., y),
                Align2::RIGHT_CENTER,
                format!("{:.2} m", distances[i]),
                FontId::proportional(14.),
                if place == 0 {
                    theme.accent
                } else {
                    theme.muted
                },
            );
        }
        painter.text(
            board.left_bottom() + Vec2::new(10., -8.),
            Align2::LEFT_BOTTOM,
            "Live distance",
            FontId::proportional(14.5),
            theme.muted,
        );
    }
}
/// What every lane of one frame shares: the zoom and camera of the strips,
/// who leads and each lane's distance.
struct RaceScene<'a> {
    painter: &'a egui::Painter,
    lanes_rect: Rect,
    lane_height: f32,
    zoom: f32,
    camera: f32,
    visible: f32,
    leader: usize,
    distances: &'a [f32],
    theme: Theme,
}
