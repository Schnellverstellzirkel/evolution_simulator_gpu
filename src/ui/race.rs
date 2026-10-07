//! The Race tab. It replays the top five creatures of the ranked archive side
//! by side, or the creatures the player sent with "Race it" against the
//! champion, and it shows live standings. This file adds the race methods to
//! `App`. `ui.rs` holds the race fields and `viewport.rs` adds the picks.

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
        GAP_L, HudLine, Theme, hud_block,
        scene::{FALLEN, GRID, GROUND_INK, HUD, HUD_DIM, HUD_INK},
    },
    worker::CardList,
};
use eframe::egui::{self, Align2, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::time::Duration;

/// One lane of the race: a creature and its replay.
pub(super) struct RaceLane {
    /// Why the creature runs: "archive rank N", "champion" or "your pick".
    label: String,
    /// The lane's replay. It plays and loops like the other replays.
    pub(super) playback: Playback,
}
/// How many creatures the player can send to the race with "Race it". The
/// champion runs beside them. A pick beyond this number drops the oldest one.
pub(super) const RACE_PICKS: usize = 4;
impl App {
    /// Builds the lanes of a race that is waiting for creatures
    /// (`race_pending`). `absorb_snapshot` calls it after each new snapshot.
    /// With picks, the lanes are the picks and the champion. The champion gets
    /// no lane of its own when it is one of the picks. With no picks it asks
    /// for the ranked archive, and `build_top_race` builds the lanes when the
    /// list arrives.
    pub(super) fn maybe_build_race(&mut self) {
        if !self.race_pending {
            return;
        }
        if self.race_picks.is_empty() {
            // The top five come with the ranked archive. Ask for it once the
            // archive holds a creature, and ask again until a list with
            // creatures arrives, but not while a request under 2 s old waits
            // for its answer.
            let kept = self.snapshot.as_ref().is_some_and(|s| s.archive_size > 0);
            let waiting = self
                .cards_requested
                .is_some_and(|at| at.elapsed() < Duration::from_secs(2));
            if kept && !waiting {
                self.request_cards();
            }
            return;
        }
        // The player's picks against the champion. The champion gets no lane
        // of its own when the player picked it.
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
        self.fill_race(lanes);
    }
    /// Starts the race with the first five cards of a ranked archive that have
    /// a descriptor and a finite score. It changes nothing when no card does.
    /// `absorb_snapshot` calls it when a list arrives, if the race is waiting
    /// and the player has no picks.
    pub(super) fn build_top_race(&mut self, list: &CardList) {
        let lanes: Vec<RaceLane> = list
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
        if lanes.is_empty() {
            return;
        }
        self.fill_race(lanes);
    }
    /// Fills the race with `lanes`, the farthest finisher first, so the live
    /// standings end in the order of the lanes. The race stops waiting and the
    /// camera goes back to the start.
    fn fill_race(&mut self, mut lanes: Vec<RaceLane>) {
        lanes.sort_by(|a, b| b.playback.distance.total_cmp(&a.playback.distance));
        self.race = lanes;
        self.race_pending = false;
        self.race_camera = 0.0;
    }
    /// Empties the race and marks it as waiting for creatures. With no picks
    /// it asks for the ranked archive, so the top five can run again. With
    /// picks, `maybe_build_race` builds the lanes from them.
    pub(super) fn restart_race(&mut self) {
        self.race.clear();
        self.race_pending = true;
        self.race_camera = 0.0;
        if self.race_picks.is_empty() {
            self.request_cards();
        }
    }
    /// The Race tab: the title row and the controls, then the lanes on the left
    /// and the standings board on the right. The lanes share one zoom and one
    /// camera. The camera follows the leader, the lane that has gone farthest
    /// so far. While no lane runs, a note takes the place of the lanes.
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
        // The standings board gets the right edge: 250 px, or 30% of the
        // width if that is less. The lanes get the rest, less a gap of 12 px.
        let board_width = 250.0_f32.min(rect.width() * 0.3);
        let lanes_rect = Rect::from_min_max(
            rect.min,
            Pos2::new(rect.right() - board_width - 12., rect.bottom()),
        );
        // The zoom fits the lanes' median body height to a lane, and the
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
        // The camera follows the leader's averaged center of mass, so its
        // stride does not shake the view. The leader stands 60% of the way
        // across the lanes. Near the start the camera stops with the start
        // line 8% in from the left edge. The easing below smooths a change of
        // leader.
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
    /// The title row: the heading and a line about the race. The New race
    /// button at the right edge asks for the current top five. Once the player
    /// has picks, the Top five button takes its place and forgets the picks,
    /// and a note says what the lanes are.
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
    /// The Play or Pause button, the Replay button and the speed menu. Play
    /// sets `playing` and the menu sets `speed`, and every replay shares both.
    /// Replay starts each lane over and puts the camera back at the start.
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
    /// What the tab says while no lane runs: that the fastest kept creatures
    /// are loading, when the archive holds creatures and the race waits for
    /// them, and otherwise that there is nothing to race yet.
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
    /// Paints lane `i` of the race: its strip of the world with a ruler, the
    /// creature, the readouts and a frame.
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
        // street. The ground line sits 16 px above the bottom of the strip.
        let ground = lane_rect.bottom() - 16.;
        let lane_painter = painter.with_clip_rect(lane_rect);
        let playback = &lane.playback;
        let lane_config = &playback.config;
        // The replay clock in seconds moves the sky, so a paused race holds
        // still. Each lane adds 900 px per lane index to the backdrop's
        // camera, so neighboring lanes show different skylines.
        let clock = playback.tick as f32 / physics::rate() as f32;
        crate::world_fx::backdrop(
            &lane_painter,
            lane_rect,
            ground,
            camera * zoom + i as f32 * 900.,
            clock,
            lane_config,
            None,
        );
        // The ground is a flat line across the strip. Its texture follows the
        // lane's world.
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
        // Ticks every 0.5, 1, 2, 5 or 10 m, with the smallest step that spans
        // at least 150 px.
        let step = [0.5f32, 1.0, 2.0, 5.0, 10.0]
            .into_iter()
            .find(|step| step * zoom >= 150.0)
            .unwrap_or(10.0);
        let mut x = (camera / step).ceil() * step;
        while x <= camera + visible {
            let px = lane_rect.left() + (x - camera) * zoom;
            // A tick on the ground and a faint line up through the sky.
            lane_painter.line_segment(
                [
                    Pos2::new(px, ground),
                    Pos2::new(px, lane_rect.bottom() - 5.),
                ],
                Stroke::new(1., GROUND_INK),
            );
            lane_painter.line_segment(
                [Pos2::new(px, lane_rect.top()), Pos2::new(px, ground)],
                Stroke::new(1., GRID),
            );
            // Only the first lane carries the meter labels.
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
        // The world's origin on screen: x = 0 is the start line and y = 0 is
        // the ground.
        let origin = Pos2::new(lane_rect.left() - camera * zoom, ground);
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
        // The frame is thicker and in the accent color on the leader's lane.
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
    /// A lane's readouts. At the top left are the lane's place by final
    /// distance, the creature's name, the distance its replay ends at and why
    /// it runs. At the top right are the live distance and, once the replay
    /// reaches the frame where the trial ended early, how it ended.
    fn paint_race_lane_hud(
        lane_painter: &egui::Painter,
        i: usize,
        lane: &RaceLane,
        lane_rect: Rect,
        is_leader: bool,
        distances: &[f32],
    ) {
        let playback = &lane.playback;
        let ink = if is_leader { HUD } else { HUD_INK };
        hud_block(
            lane_painter,
            lane_rect.left_top() + Vec2::splat(6.),
            Align2::LEFT_TOP,
            &[
                HudLine::text(
                    format!("{}. {}", i + 1, species_name(&playback.creature)),
                    15.,
                    ink,
                ),
                HudLine::text(
                    format!("finishes at {:.2} m · {}", playback.distance, lane.label),
                    12.,
                    HUD_DIM,
                ),
            ],
        );
        let mut right = vec![HudLine::value(format!("{:.2} m", distances[i]), 18., ink)];
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
    /// The standings board beside the lanes. It lists the lanes by live
    /// distance, with the leader in the accent color, and a "Live distance"
    /// caption at the bottom. A row that would run into the caption is left
    /// out.
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
    /// The painter of the whole Race area. Each lane clips it to its strip.
    painter: &'a egui::Painter,
    /// The area of all the strips, left of the standings board.
    lanes_rect: Rect,
    /// The height of one strip in px.
    lane_height: f32,
    /// Pixels per meter.
    zoom: f32,
    /// The world x in meters at the left edge of the strips.
    camera: f32,
    /// The meters of the world across a strip.
    visible: f32,
    /// The index of the lane with the greatest live distance.
    leader: usize,
    /// The live distance of each lane in meters.
    distances: &'a [f32],
    theme: Theme,
}
