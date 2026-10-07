//! The replay viewport. It decides which creature the replay shows, the
//! champion or one the player picked, and paints the scene and the HUD from
//! that creature's `Playback`. It also takes the camera and zoom input and
//! draws the timeline and the playback buttons. The Overview tab shows it, and
//! the Ways of moving tab docks it beside the archive.

use super::{
    App, Tab,
    controls::world_summary,
    playback::{FrameMarks, Playback},
    race::RACE_PICKS,
    records::row_in_world,
    scene::draw_creature,
    text::{number, species_name},
    widgets::speed_picker,
};
use crate::{
    config::Config,
    evolution::Creature,
    physics,
    theme::scene::{FALLEN, GROUND_EDGE, GROUND_INK},
};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

/// Pixels per meter of the camera when the window opens, when a new replay
/// starts and after Reset camera. Until the player zooms by hand, `auto_zoom`
/// replaces it with a zoom that fits the body of the replay.
pub(super) const DEFAULT_CAMERA_ZOOM: f32 = 80.0;
/// Share of the viewport height under the ground line, room for the HUD.
const GROUND_SHARE: f32 = 0.25;
/// Share of the viewport height a body fills at its fitted zoom (`fit_zoom`).
const FIT_HEIGHT_SHARE: f32 = 0.42;
impl App {
    /// Starts the replay of a creature in its world. Its first pose shows at
    /// once while a thread records the replay and waits up to 60 seconds for
    /// the GPU. `receive_replay` takes the recording when it is ready. The
    /// camera resets and follows the creature.
    pub(super) fn set_preview(&mut self, c: Creature, cfg: Config) {
        // Recording takes a while, so it runs off the UI thread. The first
        // pose shows meanwhile, and the recording replaces it.
        self.playback = Some(Playback::preparing(c.clone(), cfg.clone()));
        let (tx, rx) = mpsc::channel();
        let ctx = self.ctx.clone();
        let spawned = std::thread::Builder::new()
            .name("replay".into())
            .spawn(move || {
                let _ = tx.send(Playback::recorded(c, cfg, Duration::from_secs(60)));
                ctx.request_repaint();
            });
        self.replay_wait = spawned.is_ok().then(|| (rx, Instant::now()));
        self.follow = true;
        self.zoom = DEFAULT_CAMERA_ZOOM;
        self.zoom_user = false;
        self.camera = [0.; 2];
    }
    /// Shows a creature the player picked. The replay keeps it until the
    /// player goes back to the champion. The lineage on screen belonged to the
    /// creature shown before, so it is cleared.
    pub(super) fn select(&mut self, creature: Creature, config: Config) {
        self.pinned = true;
        self.set_preview(creature, config);
        self.lineage.clear();
    }
    /// Shows an ancestor from the lineage the player is browsing, keeping the
    /// lineage on screen.
    pub(super) fn select_ancestor(&mut self, creature: Creature, config: Config) {
        self.pinned = true;
        self.set_preview(creature, config);
    }
    /// Shows a champion and follows new ones from now on. It clears the
    /// lineage of the creature shown before.
    pub(super) fn show_champion(&mut self, creature: Creature, config: Config) {
        self.pinned = false;
        self.champion_shown = true;
        self.set_preview(creature, config);
        self.lineage.clear();
    }
    /// The best elite of the global archive now, and the world it is scored
    /// in. The worker sends it as soon as a record is absorbed, mid-generation
    /// too. Until it has sent one, the best creature of the newest history row
    /// stands in, when that row was measured in the live world.
    pub(super) fn champion(&self) -> Option<(Creature, Config)> {
        let snapshot = self.snapshot.as_ref()?;
        if let Some(live) = &snapshot.champion {
            return Some((live.0.clone(), live.1.clone()));
        }
        // A row from before a world change is not this world's champion.
        let stats = row_in_world(snapshot)?;
        Some((stats.representatives.last()?.clone(), stats.config.clone()))
    }
    /// The world changed and nothing measured in it is kept yet.
    fn awaiting_new_world(&self) -> bool {
        self.snapshot.as_ref().is_some_and(|s| {
            !s.history.is_empty() && s.champion.is_none() && row_in_world(s).is_none()
        })
    }
    /// Keeps the replay on the champion unless the player pinned a creature. A
    /// new champion, which a new distance record brings, replaces the one on
    /// screen at once. The frame loop calls it after each snapshot of the
    /// worker.
    pub(super) fn follow_champion(&mut self) {
        let Some((creature, config)) = self.champion() else {
            // The world changed and no creature is kept in it yet: the old
            // champion does not stand for this world, so the view empties.
            if !self.pinned && self.awaiting_new_world() && self.playback.is_some() {
                self.playback = None;
                self.replay_wait = None;
                self.champion_shown = false;
            }
            return;
        };
        let showing = self.playback.as_ref().map(|p| p.creature.id);
        if follows_champion(self.pinned, showing, Some(creature.id)) {
            self.show_champion(creature, config);
        } else if !self.pinned && showing == Some(creature.id) {
            // The creature on screen is the champion already, so the header
            // calls it the champion.
            self.champion_shown = true;
        }
    }
    /// Stops watching a picked creature and shows the champion now.
    fn back_to_champion(&mut self) {
        self.pinned = false;
        if let Some((creature, config)) = self.champion() {
            self.show_champion(creature, config);
        }
    }
    /// The replay header's buttons: Follow, Forces, Reset camera and, while a
    /// picked creature is pinned, Back to champion. Returns whether the player
    /// clicked Back to champion.
    fn viewport_buttons(&mut self, ui: &mut egui::Ui) -> bool {
        let theme = self.theme();
        let mut back = false;
        ui.checkbox(&mut self.follow, "Follow")
            .on_hover_text("Keep the camera on the creature");
        ui.checkbox(&mut self.show_forces, "Forces").on_hover_text(
            "Draw muscle forces (orange) and ground pushes (blue), estimated from the recording",
        );
        if ui.button("Reset camera").clicked() {
            self.zoom = DEFAULT_CAMERA_ZOOM;
            self.zoom_user = false;
            self.camera = [0.; 2];
            self.follow = true;
        }
        if self.pinned {
            back = ui
                .button(RichText::new("Back to champion").color(theme.accent))
                .clicked();
        }
        back
    }
    /// The replay header: a label that says whose replay it is (a picked
    /// creature, the champion or a first-generation creature), the creature's
    /// name with its distance, and the buttons. The buttons share the line when
    /// the view is wide and take a line of their own when it is narrow.
    fn viewport_header(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        let mut back = false;
        // A narrow replay (docked beside the archive) puts its buttons on a
        // line of their own.
        let wide = ui.available_width() > 900.;
        let mut header = |ui: &mut egui::Ui| {
            let (mode, fill, color, why) = if self.pinned {
                (
                    " WATCHING ",
                    theme.card_hover,
                    theme.ink,
                    "A creature you picked. Back to champion shows the best creature again.",
                )
            } else if self.champion_shown {
                (
                    " CHAMPION ",
                    theme.go_fill,
                    theme.go_text,
                    "The best creature so far. The view switches to each new champion as soon as it sets a record.",
                )
            } else {
                (
                    " FIRST GENERATION ",
                    theme.card,
                    theme.muted,
                    "A random creature of the first generation. The champion takes over as soon as the first creature is kept.",
                )
            };
            ui.label(
                RichText::new(mode)
                    .size(14.)
                    .strong()
                    .color(color)
                    .background_color(fill),
            )
            .on_hover_text(why);
            if let Some(p) = &self.playback {
                ui.label(
                    RichText::new(format!("{} · {:.2} m", species_name(&p.creature), p.distance))
                        .strong(),
                )
                .on_hover_text(format!(
                    "{} nodes, {} bones, {} muscles. Creature {}. {:.2} m is the distance this replay reaches, and m/s its speed over the last fifth of a second. The replay is the scoring kernel's own trial, so it shows the GPU score. An island record's archive score comes from its confirmation trial.",
                    p.nodes.len(),
                    p.creature.bones.len(),
                    p.creature.muscles.len(),
                    p.creature.id,
                    p.distance,
                ));
            }
            if wide {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    back = self.viewport_buttons(ui);
                });
            }
        };
        if wide {
            ui.horizontal(|ui| header(ui));
        } else {
            ui.horizontal_wrapped(|ui| header(ui));
        }
        if !wide {
            ui.horizontal_wrapped(|ui| {
                back = self.viewport_buttons(ui);
            });
        }
        if back {
            self.back_to_champion();
        }
    }
    /// The replay's input. A click toggles play and pause, scrolling zooms,
    /// and a drag pans the camera and turns Follow off. Once the player has
    /// zoomed by hand the zoom stays. Until then each frame sets it to the fit
    /// for the body of the replay (`auto_zoom`).
    fn viewport_camera(&mut self, ui: &mut egui::Ui, rect: Rect, response: egui::Response) {
        if response.clicked() {
            self.playing = !self.playing;
        }
        let response =
            response.on_hover_text("Click to pause or play · drag to pan · scroll to zoom");
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.zoom_user = true;
            }
            self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(30., 1200.);
        }
        if response.dragged() {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera[0] -= delta.x / self.zoom;
            self.camera[1] += delta.y / self.zoom;
            self.follow = false;
        }
        if !self.zoom_user
            && let Some(p) = &self.playback
        {
            self.zoom = auto_zoom(p.height, p.peak, rect.height());
        }
        // Developer screenshots: `EVOLUTION_SMOKE_VIEW_ZOOM=<pixels per meter>`
        // sets the zoom in every frame, to frame a wider stretch of the ground.
        if let Some(zoom) = std::env::var("EVOLUTION_SMOKE_VIEW_ZOOM")
            .ok()
            .and_then(|z| z.parse::<f32>().ok())
        {
            self.zoom = zoom;
        }
    }
    /// The scene's frame of reference for this frame: the painter, the camera's
    /// origin, the ground of the world the replay ran in, and the range of
    /// meters on screen.
    fn scene_frame(&self, ui: &egui::Ui, rect: Rect) -> SceneFrame<'_> {
        // The scene is made of egui shapes, which egui batches into its wgpu
        // render pass.
        let painter = ui.painter_at(rect);
        // The world point (`camera[0]`, `camera[1]`) is at the middle of the
        // view across and `GROUND_SHARE` of its height up from the bottom.
        let origin = Pos2::new(
            rect.center().x - self.camera[0] * self.zoom,
            rect.bottom() - rect.height() * GROUND_SHARE + self.camera[1] * self.zoom,
        );
        let cfg = self
            .playback
            .as_ref()
            .map(|p| &p.config)
            .unwrap_or(&self.config);
        // The replay clock, so effects animate with the replay and hold
        // still when it is paused.
        let clock = self
            .playback
            .as_ref()
            .map_or(0.0, |p| p.tick as f32 / physics::rate() as f32);
        // The replay's own creature decides the earthquake ground, through
        // the same id hash the kernel uses.
        let quake_hash = self
            .playback
            .as_ref()
            .map_or(0, |p| crate::physics::quake_hash(p.creature.id));
        let slope = if cfg.ground { cfg.slope } else { 0.0 };
        let gaps = if cfg.ground { cfg.gaps } else { 0.0 };
        let hurdles = if cfg.ground { cfg.hurdles } else { 0.0 };
        let quake = if cfg.ground { cfg.quake } else { 0.0 };
        let mud = if cfg.ground { cfg.mud } else { 0.0 };
        let amplitude = crate::physics::terrain_amplitude(cfg.terrain)
            + quake * crate::physics::quake_scale(quake_hash);
        let phase = if quake > 0.0 {
            crate::physics::quake_phase(quake_hash)
        } else {
            0.0
        };
        let start = (rect.left() - origin.x) / self.zoom;
        let end = (rect.right() - origin.x) / self.zoom;
        let left = start.floor() as i32;
        let right = end.ceil() as i32;
        SceneFrame {
            painter,
            rect,
            origin,
            zoom: self.zoom,
            cfg,
            clock,
            amplitude,
            slope,
            gaps,
            hurdles,
            mud,
            phase,
            start,
            end,
            left,
            right,
        }
    }
    /// Paints the replay's scene under the HUD, from back to front: the sky
    /// and the grid, the ground, the water, the ruler, the creature, the
    /// weather over it and a film look.
    fn paint_scene(&self, ctx: &egui::Context, f: &SceneFrame) {
        let painter = &f.painter;
        let rect = f.rect;
        let cfg = f.cfg;
        let clock = f.clock;
        self.paint_sky_and_grid(f);
        self.paint_ground(f);
        self.paint_water(f);
        self.paint_ruler(f);
        self.paint_replay_creature(ctx, f);
        crate::world_fx::weather(painter, rect, cfg, clock, self.camera[0] * self.zoom);
        // Film look over the scene, under the HUD.
        crate::theme::vignette(painter, rect, 0.55);
        crate::theme::grain(painter, rect, clock, 0.055);
    }
    /// The backdrop with its skyline, the effects in the sky over it, and a
    /// faint vertical line at every meter.
    fn paint_sky_and_grid(&self, f: &SceneFrame) {
        let painter = &f.painter;
        let rect = f.rect;
        let origin = f.origin;
        let cfg = f.cfg;
        let clock = f.clock;
        let left = f.left;
        let right = f.right;
        let world = |x: f32, y: f32| f.world(x, y);
        let height_at = |x: f32, with_hurdles: bool| f.height_at(x, with_hurdles);
        // The skyline stands on the ground under the middle of the view.
        let horizon = if cfg.ground {
            world(
                0.,
                height_at((rect.center().x - origin.x) / self.zoom, false),
            )
            .y
        } else {
            origin.y
        };
        crate::world_fx::backdrop(
            painter,
            rect,
            horizon,
            self.camera[0] * self.zoom,
            clock,
            cfg,
            (cfg.water > 0.0).then(|| world(0., cfg.water).y),
        );
        crate::world_fx::sky(painter, rect, cfg, clock);
        for x in left..=right {
            let pos = world(x as f32, 0.);
            painter.line_segment(
                [
                    Pos2::new(pos.x, rect.top()),
                    Pos2::new(pos.x, rect.bottom()),
                ],
                Stroke::new(1., crate::theme::scene::GRID),
            );
        }
    }
    /// The ground: its fill, mud, edge, structures and what the feet kick up.
    fn paint_ground(&self, f: &SceneFrame) {
        let painter = &f.painter;
        let rect = f.rect;
        let origin = f.origin;
        let cfg = f.cfg;
        let clock = f.clock;
        let start = f.start;
        let end = f.end;
        let mud = f.mud;
        let amplitude = f.amplitude;
        let slope = f.slope;
        let gaps = f.gaps;
        let hurdles = f.hurdles;
        let world = |x: f32, y: f32| f.world(x, y);
        let height_at = |x: f32, with_hurdles: bool| f.height_at(x, with_hurdles);
        if cfg.ground {
            // Sample the ground every 4 pixels (flat ground needs only its two
            // ends) and let `ground_body` fill it down to the bottom of the
            // view. Pits show as notches in the polyline. Mud draws its sunk
            // layer `mud` meters below the surface line.
            let flat = amplitude == 0.0 && slope == 0.0 && gaps == 0.0 && hurdles == 0.0;
            let step = if flat {
                (end - start).max(0.01)
            } else {
                (4.0 / self.zoom).max(0.002)
            };
            let mut x = start;
            let mut line = Vec::new();
            let mut meters = Vec::new();
            let mut mud_line = Vec::new();
            while x <= end + step {
                let height = height_at(x, true);
                line.push(world(x, height));
                meters.push(x);
                if mud > 0.0 {
                    mud_line.push(world(x, height - mud));
                }
                x += step;
            }
            crate::world_fx::ground_body(painter, rect, cfg, &line, &meters, self.zoom);
            if mud > 0.0 {
                // The mud: a band under the surface, an edge along its lower
                // side and a sheen along the surface.
                let fill = crate::theme::scene::MUD;
                for i in 0..line.len().saturating_sub(1) {
                    painter.add(egui::Shape::convex_polygon(
                        vec![line[i], line[i + 1], mud_line[i + 1], mud_line[i]],
                        fill,
                        Stroke::NONE,
                    ));
                }
                painter.add(egui::Shape::line(
                    mud_line,
                    Stroke::new(1.5, crate::theme::scene::MUD_EDGE),
                ));
                let sheen: Vec<Pos2> = line.iter().map(|p| *p + Vec2::new(0., 1.5)).collect();
                painter.add(egui::Shape::line(
                    sheen,
                    Stroke::new(1.5, crate::theme::scene::MUD_SHEEN),
                ));
            }
            // The ground's edge, with a dark shade line just under it.
            let shade: Vec<Pos2> = line.iter().map(|p| *p + Vec2::new(0., 2.)).collect();
            painter.add(egui::Shape::line(
                shade,
                Stroke::new(1.5, Color32::from_black_alpha(110)),
            ));
            painter.add(egui::Shape::line(line, Stroke::new(1.5, GROUND_EDGE)));
            crate::world_fx::structures(painter, rect, cfg, &world, &height_at, (start, end));
            let surface = |sx: f32| world(0., height_at((sx - origin.x) / self.zoom, true)).y;
            // A node whose underside is less than 6 cm plus the mud depth above
            // the ground is a foot. Its speed is how far it moved since the
            // previous frame, in pixels per second.
            let feet: Vec<crate::world_fx::Foot> = self
                .playback
                .as_ref()
                .map(|p| {
                    let rate = physics::rate() as f32;
                    let before = p.frames.get((p.tick as usize).saturating_sub(1));
                    p.nodes
                        .iter()
                        .enumerate()
                        .filter_map(|(id, n)| {
                            let ground_y = height_at(n.pos[0], true);
                            (n.pos[1] - n.radius - ground_y < 0.06 + mud).then(|| {
                                let dx = before
                                    .and_then(|f| f.get(id))
                                    .map_or(0.0, |b| n.pos[0] - b[0]);
                                crate::world_fx::Foot {
                                    at: world(n.pos[0], ground_y),
                                    speed: dx.abs() * rate * self.zoom,
                                    id,
                                }
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            crate::world_fx::ground(
                painter,
                rect,
                cfg,
                clock,
                &surface,
                &|sx| (sx - origin.x) / self.zoom,
                self.zoom,
                &feet,
            );
        }
    }
    /// The water of a flooded world, up to the line `cfg.water` meters above
    /// the flat ground. `world_fx::water` paints nothing when the water effect
    /// is calm.
    fn paint_water(&self, f: &SceneFrame) {
        let painter = &f.painter;
        let rect = f.rect;
        let origin = f.origin;
        let cfg = f.cfg;
        let clock = f.clock;
        let world = |x: f32, y: f32| f.world(x, y);
        crate::world_fx::water(
            painter,
            rect,
            cfg,
            clock,
            world(0., cfg.water).y,
            &|sx| (sx - origin.x) / self.zoom,
            self.zoom,
        );
    }
    /// A tick at every meter on the line at height 0, with a label every few
    /// meters.
    fn paint_ruler(&self, f: &SceneFrame) {
        let painter = &f.painter;
        let left = f.left;
        let right = f.right;
        let world = |x: f32, y: f32| f.world(x, y);
        // A tick at every meter. A label at every 1, 2, 5, 10, 20 or 50 m,
        // the smallest of these that keeps labels 48 pixels apart.
        let every = [1, 2, 5, 10, 20]
            .into_iter()
            .find(|&n| n as f32 * self.zoom >= 48.)
            .unwrap_or(50);
        for x in left..=right {
            let pos = world(x as f32, 0.);
            painter.line_segment([pos, pos + Vec2::new(0., 6.)], Stroke::new(1., GROUND_INK));
            if x.rem_euclid(every) == 0 {
                painter.text(
                    pos + Vec2::new(5., 6.),
                    Align2::LEFT_TOP,
                    format!("{x} m"),
                    FontId::proportional(14.5),
                    GROUND_INK,
                );
            }
        }
    }
    /// The creature of the replay with its center-of-mass trail and its
    /// contact shadows. `ctx` holds the texture of the shadows.
    fn paint_replay_creature(&self, ctx: &egui::Context, f: &SceneFrame) {
        let painter = &f.painter;
        let origin = f.origin;
        let world = |x: f32, y: f32| f.world(x, y);
        let height_at = |x: f32, with_hurdles: bool| f.height_at(x, with_hurdles);
        if let Some(p) = &self.playback {
            // Center-of-mass trail from the last two seconds of recorded
            // frames, fading with age.
            let trail = crate::theme::scene::TRAIL;
            let span = physics::rate().saturating_mul(2).max(1);
            let first = p.tick.saturating_sub(span);
            let mut previous: Option<Pos2> = None;
            for tick in first..=p.tick {
                let Some(com) = p.center_of_mass(tick) else {
                    continue;
                };
                let point = world(com[0], com[1]);
                if let Some(from) = previous {
                    let freshness = 1.0 - (p.tick - tick) as f32 / span as f32;
                    let alpha = (freshness.clamp(0.0, 1.0) * 150.0) as u8;
                    painter.line_segment(
                        [from, point],
                        Stroke::new(
                            2.,
                            Color32::from_rgba_unmultiplied(trail.r(), trail.g(), trail.b(), alpha),
                        ),
                    );
                }
                previous = Some(point);
            }
            if let Some(com) = p.shown_center() {
                let at = world(com[0], com[1]);
                painter.circle_filled(at, 6., trail.gamma_multiply(0.15));
                painter.circle_filled(at, 3., trail);
            }
            // Soft contact shadows, darker and tighter the nearer a node is
            // to the ground under it.
            let glow = crate::assets::Art::Glow.texture(ctx);
            for n in &p.nodes {
                let ground = height_at(n.pos[0], true);
                let lift = (n.pos[1] - n.radius - ground).max(0.0);
                let fade = (1.0 - lift / 0.6).clamp(0.0, 1.0);
                if fade <= 0.0 {
                    continue;
                }
                let at = world(n.pos[0], ground);
                let w = n.radius * self.zoom * (2.6 + lift * 2.0);
                painter.image(
                    glow,
                    Rect::from_center_size(at, Vec2::new(w * 2.0, (w * 0.5).max(6.0))),
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::from_black_alpha((190.0 * fade) as u8),
                );
            }
            let mut marks = FrameMarks::of(p);
            marks.arrows = self.show_forces;
            draw_creature(painter, &p.nodes, &p.creature, origin, self.zoom, &marks);
        }
    }
    /// The HUD over the scene. It has the distance and speed counters of the
    /// creature, a line that says how the trial ended once the replay reaches
    /// that moment, the world, the generation counter with its rate, a note in
    /// the middle while there is no replay to watch, and a frame around the
    /// scene.
    fn paint_hud(&self, f: &SceneFrame) {
        let theme = self.theme();
        let painter = &f.painter;
        let rect = f.rect;
        use crate::theme::{Counter, HudLine, counter, hud_block, scene::HUD};
        // The HUD, laid out like Half-Life 2's: the creature's counters low
        // on the left like HEALTH and SUIT, the generation low on the right
        // like the ammo box with the rate as its reserve, the world as a
        // hint in the top right corner.
        let inset = 12.;
        let size = (rect.height() * 0.085).clamp(20., 32.);
        if let Some(p) = &self.playback {
            let fallen = p.fallen();
            let distance = p.current_distance();
            let left = counter(
                painter,
                rect.left_bottom() + Vec2::new(inset, -inset),
                Align2::LEFT_BOTTOM,
                &Counter {
                    label: "Distance",
                    digits: format!("{distance:.2}"),
                    unit: "m",
                    extra: None,
                    damaged: fallen.is_some(),
                },
                size,
            );
            counter(
                painter,
                left.right_bottom() + Vec2::new(inset, 0.),
                Align2::LEFT_BOTTOM,
                &Counter {
                    label: "Speed",
                    digits: format!("{:.2}", if fallen.is_some() { 0.0 } else { p.speed() }),
                    unit: "m/s",
                    extra: None,
                    damaged: fallen.is_some(),
                },
                size,
            );
            // The trial ended early and the replay has reached that moment.
            if let Some((tick, _)) = fallen {
                hud_block(
                    painter,
                    rect.center_top() + Vec2::new(0., inset),
                    Align2::CENTER_TOP,
                    &[HudLine::text(
                        p.ending.sentence(
                            tick.saturating_sub(physics::settle()) as f32 * physics::dt(),
                        ),
                        16.,
                        Color32::from_rgb(255, 120, 100),
                    )],
                );
            }
            // A replay from before a world change says so.
            let live = self.snapshot.as_ref().map(|s| &s.config);
            let earlier = live.is_some_and(|live| live.physics_differs(&p.config));
            hud_block(
                painter,
                rect.right_top() + Vec2::new(-inset, inset),
                Align2::RIGHT_TOP,
                &[
                    HudLine::label(if earlier {
                        "World · an earlier one"
                    } else {
                        "World"
                    }),
                    HudLine::text(world_summary(&p.config), 15., HUD),
                ],
            );
        }
        if let Some(s) = &self.snapshot {
            // A narrow view has no room beside the creature's counters, so
            // the generation moves to the top left corner.
            let (anchor, align) = if rect.width() < 760. {
                (rect.left_top() + Vec2::splat(inset), Align2::LEFT_TOP)
            } else {
                (
                    rect.right_bottom() - Vec2::splat(inset),
                    Align2::RIGHT_BOTTOM,
                )
            };
            counter(
                painter,
                anchor,
                align,
                &Counter {
                    label: "Gen",
                    digits: s.generation.to_string(),
                    unit: "",
                    extra: Some(format!("{}/s", number(s.end_to_end.max(0.0) as usize))),
                    damaged: false,
                },
                size,
            );
        }
        // The note in the middle says why there is no replay to watch.
        let center_note = if self.playback.is_none() && self.awaiting_new_world() {
            Some("Testing in the new world...")
        } else if self.playback.is_none() {
            Some("Preparing your first population…")
        } else if self.playback.as_ref().is_some_and(|p| p.preparing) {
            Some("Preparing replay...")
        } else if self.playback.as_ref().is_some_and(|p| p.unavailable) {
            Some("The GPU did not record this replay")
        } else {
            None
        };
        if let Some(note) = center_note {
            let mut lines = vec![HudLine::text(note.to_owned(), 20., HUD)];
            // Before the first replay, an engine may wait for a kernel, and
            // two more lines say that the GPU is compiling.
            if self.playback.is_none() && crate::cuda_engine::compiling_world() {
                lines.push(HudLine::text(
                    "The GPU is compiling its kernels for this world.".to_owned(),
                    15.,
                    crate::theme::scene::HUD_INK,
                ));
                lines.push(HudLine::text(
                    "A new game does this once, for up to a minute or two. Later starts are quick."
                        .to_owned(),
                    15.,
                    crate::theme::scene::HUD_INK,
                ));
            }
            hud_block(painter, rect.center(), Align2::CENTER_CENTER, &lines);
        }
        // The frame around the scene.
        painter.rect_stroke(
            rect,
            0,
            Stroke::new(4., theme.ink),
            egui::StrokeKind::Inside,
        );
    }
    /// The time slider and the clock under the scene. A mark on the slider
    /// shows the frame where the trial ended early. Returns whether the player
    /// moved the slider.
    fn viewport_timeline(&mut self, ui: &mut egui::Ui) -> bool {
        let mut sought = false;
        if let Some(p) = &mut self.playback {
            let last_frame = p.last_frame();
            let trial_start = p.trial_start();
            let trial_frames = last_frame.saturating_sub(trial_start);
            ui.horizontal(|ui| {
                ui.label("Time");
                if trial_frames > 0 {
                    // The scrubber takes the row, less room for the clock.
                    ui.spacing_mut().slider_width = (ui.available_width() - 110.).max(120.);
                    let mut frame = p.tick.saturating_sub(trial_start).min(trial_frames);
                    let response = ui.add(
                        egui::Slider::new(&mut frame, 0..=trial_frames)
                            .show_value(false)
                            .text(""),
                    );
                    // Mark the frame where the trial ended early.
                    if let Some((fall_frame, _)) = p.fall {
                        let fraction = fall_frame.saturating_sub(trial_start).min(trial_frames)
                            as f32
                            / trial_frames as f32;
                        let x = egui::lerp(response.rect.x_range(), fraction);
                        ui.painter().line_segment(
                            [
                                Pos2::new(x, response.rect.top() + 3.),
                                Pos2::new(x, response.rect.bottom() - 3.),
                            ],
                            Stroke::new(2., FALLEN),
                        );
                    }
                    if response.changed() {
                        p.seek(frame);
                        sought = true;
                    }
                } else {
                    ui.label("single frame");
                }
                ui.label(format!(
                    "{:.1} / {:.0} s",
                    p.elapsed_seconds(),
                    p.config.duration
                ));
            });
        }
        sought
    }
    /// The button row under the timeline: Play or Pause, Replay, Family tree,
    /// Race it, the two exports and the speed menu. Returns the creature and
    /// world the player asked to race.
    fn viewport_controls(&mut self, ui: &mut egui::Ui) -> Option<(Creature, Config)> {
        let mut race_it = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().button_padding.x = 10.;
            ui.spacing_mut().item_spacing.x = 6.;
            if ui
                .button(if self.playing {
                    "Pause  (K)"
                } else {
                    "Play  (K)"
                })
                .on_hover_text("Pause or play the replay. A click on the replay does the same.")
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("Replay").clicked()
                && let Some(p) = &mut self.playback
            {
                p.reset();
            }
            if ui
                .add_enabled(self.playback.is_some(), egui::Button::new("Family tree"))
                .on_hover_text("The ancestors of this creature, with what changed at each step")
                .clicked()
            {
                self.tab = Tab::Lineage;
            }
            if ui
                .add_enabled(self.playback.is_some(), egui::Button::new("Race it"))
                .on_hover_text("Race this creature against the champion")
                .clicked()
                && let Some(p) = &self.playback
            {
                race_it = Some((p.creature.clone(), p.config.clone()));
            }
            if ui
                .add_enabled(self.playback.is_some(), egui::Button::new("Export GIF"))
                .on_hover_text("Save an animated GIF of this replay under runs/")
                .clicked()
            {
                self.file("Export creature GIF");
            }
            if ui
                .add_enabled(self.playback.is_some(), egui::Button::new("Export JSON"))
                .on_hover_text(
                    "Save this creature as JSON under runs/, to open it again or share it",
                )
                .clicked()
            {
                self.file("Export creature JSON");
            }
            // A menu does not wrap by itself, so start a new line when it will not fit.
            if ui.available_width() < 175. {
                ui.end_row();
            }
            speed_picker(ui, &mut self.speed, "replay_speed");
        });
        race_it
    }
    /// Adds a creature to the race picks and opens the Race tab with a fresh
    /// race. A creature that is picked already moves to the newest place, and
    /// the oldest pick leaves when there are more than `RACE_PICKS`.
    fn race_viewport_creature(&mut self, creature: Creature, config: Config) {
        self.race_picks.retain(|(pick, _)| pick.id != creature.id);
        self.race_picks.push((creature, config));
        if self.race_picks.len() > RACE_PICKS {
            self.race_picks.remove(0);
        }
        self.tab = Tab::Race;
        // The race restarts below. With `prev_tab` set, the frame loop does not
        // restart it again as an opened tab (`opened_tab`).
        self.prev_tab = Tab::Race;
        self.restart_race();
    }
    /// The replay viewport: the header, the scene with its HUD, the timeline
    /// and the playback buttons. `height` is the height of the scene in
    /// points, and the scene is never shorter than 120.
    pub(super) fn viewport(&mut self, ui: &mut egui::Ui, height: f32) {
        self.viewport_header(ui);
        let (rect, response) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), height.max(120.)),
            Sense::click_and_drag(),
        );
        self.viewport_camera(ui, rect, response);
        let frame = self.scene_frame(ui, rect);
        self.paint_scene(ui.ctx(), &frame);
        self.paint_hud(&frame);
        let sought = self.viewport_timeline(ui);
        let race_it = self.viewport_controls(ui);
        // Moving the slider pauses the replay.
        if sought {
            self.playing = false;
        }
        if let Some((creature, config)) = race_it {
            self.race_viewport_creature(creature, config);
        }
    }
    /// Replays the best creature of row `index` of the history, in the world
    /// of that row. It shows it as a picked creature and opens the Overview
    /// tab. It does nothing when there is no such row.
    pub(super) fn replay_history_holder(&mut self, index: usize) {
        let Some((creature, config)) = self.snapshot.as_ref().and_then(|snapshot| {
            let stats = snapshot.history.get(index)?;
            Some((stats.representatives.last()?.clone(), stats.config.clone()))
        }) else {
            return;
        };
        self.select(creature, config);
        self.tab = Tab::Overview;
    }
    /// Replays the champion now as a picked creature and opens the Overview
    /// tab. The Replay button of the running generation's record calls it.
    pub(super) fn replay_champion(&mut self) {
        if let Some((creature, config)) = self.champion() {
            self.select(creature, config);
            self.tab = Tab::Overview;
        }
    }
}
/// What the scene's painting methods share in one frame: the painter, the
/// camera and the world's ground.
struct SceneFrame<'a> {
    /// Paints inside `rect`.
    painter: egui::Painter,
    /// The area of the scene on screen.
    rect: Rect,
    /// The screen position of the world point (0, 0).
    origin: Pos2,
    /// Pixels per meter.
    zoom: f32,
    /// The world to paint: the replay's own, or the settings' world when
    /// there is no replay.
    cfg: &'a Config,
    /// The replay clock in seconds, 0 without a replay.
    clock: f32,
    /// Bump height of the ground (m): the roughness level plus the earthquake
    /// bumps of the replay's creature.
    amplitude: f32,
    /// Rise over run of the ground, 0 in a world without ground.
    slope: f32,
    /// Opening of the pits (m), 0 in a world without ground.
    gaps: f32,
    /// Height of the hurdles (m), 0 in a world without ground.
    hurdles: f32,
    /// Depth of the mud (m), 0 in a world without ground.
    mud: f32,
    /// Phase of the bumps in wave turns, 0 without an earthquake.
    phase: f32,
    /// The world x (m) at the left edge of the scene.
    start: f32,
    /// The world x (m) at the right edge of the scene.
    end: f32,
    /// `start` rounded down to a whole meter.
    left: i32,
    /// `end` rounded up to a whole meter.
    right: i32,
}
impl SceneFrame<'_> {
    /// The screen position of a point of the world.
    fn world(&self, x: f32, y: f32) -> Pos2 {
        Pos2::new(self.origin.x + x * self.zoom, self.origin.y - y * self.zoom)
    }
    /// The ground's height at x, with or without its hurdles.
    fn height_at(&self, x: f32, with_hurdles: bool) -> f32 {
        crate::physics::ground(
            x,
            self.amplitude,
            self.slope,
            self.gaps,
            if with_hurdles { self.hurdles } else { 0.0 },
            self.phase,
        )
        .0
    }
}
/// Pixels per meter at which a body `body_height` meters tall fills
/// `FIT_HEIGHT_SHARE` of a view `view_height` points high. The result stays
/// between 40 and 450, for tiny and huge bodies. The Race tab uses it too.
pub(super) fn fit_zoom(body_height: f32, view_height: f32) -> f32 {
    (FIT_HEIGHT_SHARE * view_height / body_height.max(0.05)).clamp(40.0, 450.0)
}
/// The zoom of a replay until the player zooms by hand. The typical `height`
/// of the body fills its share of the view (`fit_zoom`), and the `peak` of the
/// recording stays in view above the ground, which sits a quarter up from the
/// bottom, unless that would shrink the body to less than 60% of the typical
/// fit. A creature that leaps far higher than it stands keeps that 60% and
/// clips its peak instead of becoming tiny.
fn auto_zoom(height: f32, peak: f32, view_height: f32) -> f32 {
    let typical = fit_zoom(height, view_height);
    let whole = 0.69 * view_height / peak.max(0.05);
    whole.min(typical).max(typical * 0.6).clamp(20.0, 450.0)
}
/// Whether the replay switches to the champion. It does when the player has
/// not pinned a creature, a champion exists and a different creature is on
/// screen. `showing` and `champion` are the ids of the creature on screen and
/// of the champion. A new distance record makes a new champion, so the switch
/// happens as soon as the record lands.
fn follows_champion(pinned: bool, showing: Option<u64>, champion: Option<u64>) -> bool {
    !pinned && champion.is_some() && showing != champion
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_leaper_keeps_its_peak_in_view_without_shrinking_the_body_too_far() {
        let typical = fit_zoom(0.5, 260.0);
        // A mild jump fits whole.
        assert!(auto_zoom(0.5, 1.0, 260.0) < typical);
        assert!(auto_zoom(0.5, 1.0, 260.0) * 1.0 <= 0.72 * 260.0 + 0.01);
        // A huge leap stops at 60% of the typical zoom.
        assert!((auto_zoom(0.5, 30.0, 260.0) - typical * 0.6).abs() < 0.01);
        // A body that never leaves the ground keeps the typical fit.
        assert_eq!(auto_zoom(0.5, 0.5, 260.0), typical);
    }
    #[test]
    fn default_zoom_follows_body_height() {
        let small = fit_zoom(0.3, 260.0);
        let tall = fit_zoom(1.5, 260.0);
        assert!(small > tall);
        assert!((tall * 1.5 / 260.0 - FIT_HEIGHT_SHARE).abs() < 0.01);
        assert_eq!(fit_zoom(0.001, 260.0), 450.0);
        assert_eq!(fit_zoom(100.0, 260.0), 40.0);
    }
    #[test]
    fn a_new_champion_replaces_the_old_one_at_once_unless_pinned() {
        // A record at generation 12 makes creature 42 the champion while 7,
        // the previous champion, plays: the view switches right away.
        assert!(follows_champion(false, Some(7), Some(42)));
        // Nothing on screen yet: the champion shows.
        assert!(follows_champion(false, None, Some(42)));
        // Already showing the champion: nothing to do.
        assert!(!follows_champion(false, Some(42), Some(42)));
        // A creature the player picked stays until Back to champion.
        assert!(!follows_champion(true, Some(7), Some(42)));
        // No finished generation, no champion.
        assert!(!follows_champion(false, Some(7), None));
    }
}
