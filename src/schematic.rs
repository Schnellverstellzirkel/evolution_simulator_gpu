//! This module draws the "How evolution works" window, a painted poster of
//! the search with live numbers from the worker `Snapshot`. The app calls
//! `show` every frame, and buttons in the help window and the Islands view
//! set the flag that opens it. The poster is egui painter shapes on a fixed
//! 1760 x 1040 canvas that scales to the window, a few hundred shapes a
//! frame. Its colors are fixed, in the Team Fortress 2 poster look that
//! `theme` also follows, so it reads the same under every theme.

use crate::assets;
use crate::qd::Emitter;
use crate::storage::{MIGRATION_INTERVAL, MIGRATION_SHARE, hub_island};
use crate::ui::thumbnail;
use crate::worker::Snapshot;
use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Painter, Pos2, Rect, Shape, Stroke, Vec2,
    epaint::PathShape,
};

/// Width of the canvas in canvas units, the poster's own coordinates.
const W: f32 = 1760.0;
/// Height of the canvas in canvas units.
const H: f32 = 1040.0;
/// Below this scale the poster stops shrinking and the window scrolls.
const MIN_SCALE: f32 = 0.7;

/// Dark brown outlines and text.
const INK: Color32 = Color32::from_rgb(50, 34, 26);
/// A lighter brown for the dashes of the arena lanes.
const INK_SOFT: Color32 = Color32::from_rgb(104, 78, 56);
/// The paper behind everything.
const PAPER: Color32 = Color32::from_rgb(232, 210, 160);
/// The sunburst rays on the paper.
const PAPER_RAY: Color32 = Color32::from_rgb(240, 222, 176);
/// The body of cards, signs and steps.
const PANEL: Color32 = Color32::from_rgb(251, 242, 216);
/// Light text, badges, medallions, pills and the hub's name plate.
const CREAM: Color32 = Color32::from_rgb(255, 247, 226);
const RED: Color32 = Color32::from_rgb(184, 56, 50);
const BLU: Color32 = Color32::from_rgb(70, 108, 138);
const MUSTARD: Color32 = Color32::from_rgb(228, 166, 52);
const GRASS: Color32 = Color32::from_rgb(142, 156, 74);
const GRASS_LIGHT: Color32 = Color32::from_rgb(176, 186, 98);
const ROCK: Color32 = Color32::from_rgb(138, 100, 66);
const ROCK_DARK: Color32 = Color32::from_rgb(100, 70, 46);
const DIRT: Color32 = Color32::from_rgb(190, 150, 98);
const WOOD: Color32 = Color32::from_rgb(160, 108, 62);
const SHADOW: Color32 = Color32::from_rgba_premultiplied(40, 26, 14, 70);
/// One color for each emitter, in `Emitter::ALL` order. They have the hues of
/// `ORIGIN_COLORS` in `ui/islands.rs`. The first has dark header text, the
/// rest cream.
const EMITTER_TINT: [Color32; 4] = [
    MUSTARD,
    Color32::from_rgb(184, 84, 48),
    BLU,
    Color32::from_rgb(108, 122, 54),
];
/// Island `i` wears the color of emitter `i`.
const ISLAND_TINT: [Color32; 4] = EMITTER_TINT;

/// The fixed-canvas painter: canvas units in, screen points out. A method that
/// takes `(x, y)` or a list of points reads canvas units. One that takes a
/// `Rect` reads screen points, as `rect` returns them.
struct Scene<'a> {
    /// The painter, clipped to the poster's rectangle.
    p: &'a Painter,
    /// Where the canvas corner `(0, 0)` lies on the screen.
    origin: Pos2,
    /// Screen points per canvas unit.
    k: f32,
    /// Seconds, for the gentle animation.
    t: f32,
}

impl Scene<'_> {
    /// The screen point of the canvas point `(x, y)`.
    fn at(&self, x: f32, y: f32) -> Pos2 {
        self.origin + Vec2::new(x, y) * self.k
    }
    /// The screen rectangle of the canvas rectangle with its corner at
    /// `(x, y)`, `w` wide and `h` high.
    fn rect(&self, x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::from_min_size(self.at(x, y), Vec2::new(w, h) * self.k)
    }
    /// The dark outline stroke, `width` canvas units wide but at least 1.5
    /// points.
    fn stroke(&self, width: f32) -> Stroke {
        Stroke::new((width * self.k).max(1.5), INK)
    }
    /// A filled shape with the thick dark outline every object has.
    fn poly(&self, points: &[(f32, f32)], fill: Color32, outline: f32) {
        let points: Vec<Pos2> = points.iter().map(|&(x, y)| self.at(x, y)).collect();
        self.p.add(Shape::Path(PathShape {
            points,
            closed: true,
            fill,
            stroke: self.stroke(outline).into(),
        }));
    }
    /// A rounded block with a translucent drop shadow, 4 canvas units right
    /// and 6 down. `round` is the corner radius and `outline` the line width,
    /// both in canvas units.
    fn block(&self, r: Rect, fill: Color32, round: f32, outline: f32) {
        self.p.rect_filled(
            r.translate(Vec2::new(4.0, 6.0) * self.k),
            round * self.k,
            SHADOW,
        );
        self.flat(r, fill, round, outline);
    }
    /// A rounded rectangle with the dark outline and no shadow. `block` adds
    /// the shadow.
    fn flat(&self, r: Rect, fill: Color32, round: f32, outline: f32) {
        self.p.rect(
            r,
            round * self.k,
            fill,
            self.stroke(outline),
            egui::StrokeKind::Middle,
        );
    }
    /// A filled ellipse with the dark outline, drawn as a 48-sided shape.
    fn ellipse(&self, cx: f32, cy: f32, rx: f32, ry: f32, fill: Color32) {
        let points: Vec<(f32, f32)> = (0..48)
            .map(|i| {
                let a = i as f32 / 48.0 * std::f32::consts::TAU;
                (cx + rx * a.cos(), cy + ry * a.sin())
            })
            .collect();
        self.poly(&points, fill, 3.5);
    }
    /// The body font at `size` canvas units, never under 10 points.
    fn font(&self, size: f32) -> FontId {
        FontId::new((size * self.k).max(10.0), FontFamily::Proportional)
    }
    /// The bold title font at `size` canvas units, never under 11 points.
    fn title_font(&self, size: f32) -> FontId {
        FontId::new((size * self.k).max(11.0), assets::hud_bold())
    }
    /// Wrapped body text from `(x, y)` in a box `w` wide.
    fn text(&self, x: f32, y: f32, w: f32, text: &str, size: f32, color: Color32) {
        let galley = self.p.layout(
            text.to_owned(),
            self.font(size),
            color,
            (w * self.k).max(1.0),
        );
        self.p.galley(self.at(x, y), galley, color);
    }
    /// One body line centered on `(x, y)`.
    fn label(&self, x: f32, y: f32, text: &str, size: f32, color: Color32) {
        self.p.text(
            self.at(x, y),
            Align2::CENTER_CENTER,
            text,
            self.font(size),
            color,
        );
    }
    /// A chunky title at `pos`, placed by `align`. A title in any color but
    /// `INK` gets a dark copy offset down and right, so light titles stay
    /// readable on a color.
    fn title(&self, pos: (f32, f32), align: Align2, text: &str, size: f32, color: Color32) {
        let font = self.title_font(size);
        if color != INK {
            let d = 2.2 * self.k;
            self.p.text(
                self.at(pos.0, pos.1) + Vec2::new(d, d),
                align,
                text,
                font.clone(),
                INK,
            );
        }
        self.p.text(self.at(pos.0, pos.1), align, text, font, color);
    }
    /// The header band of a card, with its upper corners rounded.
    fn band(&self, x: f32, y: f32, w: f32, h: f32, accent: Color32) {
        self.flat(self.rect(x, y, w, h), accent, 12.0, 3.5);
        self.p
            .rect_filled(self.rect(x + 3.0, y + h - 16.0, w - 6.0, 14.0), 0.0, accent);
        self.p.line_segment(
            [self.at(x + 1.5, y + h), self.at(x + w - 1.5, y + h)],
            self.stroke(3.5),
        );
    }
    /// A card: a light body under a colored header band. With `num` above 0
    /// the band gets a badge with that number and `title`. With 0 the caller
    /// draws the header itself.
    fn card(&self, area: (f32, f32, f32, f32), accent: Color32, num: u32, title: &str) {
        let (x, y, w, h) = area;
        self.block(self.rect(x, y, w, h), PANEL, 12.0, 3.5);
        self.band(x, y, w, 46.0, accent);
        if num > 0 {
            self.badge(x + 32.0, y + 23.0, num);
            let color = if accent == MUSTARD { INK } else { CREAM };
            self.title(
                (x + 62.0, y + 24.0),
                Align2::LEFT_CENTER,
                title,
                27.0,
                color,
            );
        }
    }
    /// Wrapped body text inside a card, below its header band.
    fn card_text(&self, area: (f32, f32, f32, f32), text: &str, size: f32) {
        let (x, y, w, _) = area;
        self.text(x + 18.0, y + 56.0, w - 36.0, text, size, INK);
    }
    /// A path with an arrow head at its end, in `color` over a dark edge. It
    /// needs at least two points.
    fn arrow_path(&self, points: &[(f32, f32)], color: Color32) {
        let mut line: Vec<Pos2> = points.iter().map(|&(x, y)| self.at(x, y)).collect();
        let n = line.len();
        let b = line[n - 1];
        let dir = (b - line[n - 2]).normalized();
        let side = Vec2::new(-dir.y, dir.x);
        let head = 22.0 * self.k;
        let base = b - dir * head;
        // The line stops at the base of the head.
        line[n - 1] = base;
        self.p
            .add(Shape::line(line.clone(), Stroke::new(12.0 * self.k, INK)));
        self.p
            .add(Shape::line(line, Stroke::new(7.0 * self.k, color)));
        self.p.add(Shape::Path(PathShape {
            points: vec![
                b + dir * 3.0,
                base + side * head * 0.8,
                base - side * head * 0.8,
            ],
            closed: true,
            fill: color,
            stroke: self.stroke(3.0).into(),
        }));
    }
    /// A straight arrow from `from` to `to`.
    fn arrow(&self, from: (f32, f32), to: (f32, f32), color: Color32) {
        self.arrow_path(&[from, to], color);
    }
    /// A round cream badge centered on `(x, y)` that shows the number `n`.
    fn badge(&self, x: f32, y: f32, n: u32) {
        let c = self.at(x, y);
        self.p.circle(c, 17.0 * self.k, CREAM, self.stroke(3.0));
        self.p.text(
            c,
            Align2::CENTER_CENTER,
            n.to_string(),
            self.title_font(24.0),
            INK,
        );
    }
    /// A tiny stick creature standing on `(x, y)`. `phase` swings its legs,
    /// `sc` scales it.
    fn critter(&self, x: f32, y: f32, tint: Color32, sc: f32, phase: f32) {
        let hip = self.at(x, y - 15.0 * sc);
        let head = self.at(x, y - 33.0 * sc);
        let swing = phase.sin() * 9.0 * sc;
        let lift_a = phase.cos().max(0.0) * 5.0 * sc;
        let lift_b = (-phase.cos()).max(0.0) * 5.0 * sc;
        let foot_a = self.at(x - 9.0 * sc + swing, y - lift_a);
        let foot_b = self.at(x + 9.0 * sc - swing, y - lift_b);
        let w = sc * self.k;
        for f in [foot_a, foot_b] {
            self.p.line_segment([hip, f], Stroke::new(9.0 * w, INK));
            self.p.line_segment([hip, f], Stroke::new(5.0 * w, tint));
        }
        self.p.line_segment([hip, head], Stroke::new(9.0 * w, INK));
        self.p.line_segment([hip, head], Stroke::new(5.0 * w, tint));
        self.p.circle(head, 7.5 * w, tint, self.stroke(2.5 * sc));
    }
    /// The icon of an emitter in a round cream medallion. `index` is the
    /// emitter's place in `Emitter::ALL`. `pulse`, from 0 to 1, grows a halo
    /// around the medallion and fades it out.
    fn emblem(&self, x: f32, y: f32, index: usize, pulse: f32) {
        let c = self.at(x, y);
        let r = 17.0 * self.k;
        self.p.circle_filled(
            c,
            r + 6.0 * self.k * pulse,
            Color32::from_rgba_unmultiplied(255, 247, 226, (110.0 * (1.0 - pulse)) as u8),
        );
        self.p.circle(c, r, CREAM, self.stroke(3.0));
        let ink = Stroke::new(3.0 * self.k, INK);
        match index {
            // Cma: a gear, turning slowly.
            0 => {
                let pts: Vec<Pos2> = (0..16)
                    .map(|i| {
                        let a = i as f32 / 16.0 * std::f32::consts::TAU + self.t * 0.8;
                        let rr = if (i / 2) % 2 == 0 { 11.5 } else { 8.0 };
                        c + Vec2::new(a.cos(), a.sin()) * rr * self.k
                    })
                    .collect();
                self.p.add(Shape::Path(PathShape {
                    points: pts,
                    closed: true,
                    fill: MUSTARD,
                    stroke: Stroke::new(2.0 * self.k, INK).into(),
                }));
                self.p.circle_filled(c, 3.0 * self.k, INK);
            }
            // Structural: a bone.
            1 => {
                let a = c + Vec2::new(-8.0, 6.0) * self.k;
                let b = c + Vec2::new(8.0, -6.0) * self.k;
                self.p.line_segment([a, b], Stroke::new(4.5 * self.k, INK));
                for (end, sign) in [(a, -1.0), (b, 1.0)] {
                    for (dx, dy) in [(sign * 3.0, sign * 1.0), (-sign * 1.0, -sign * 3.5)] {
                        self.p
                            .circle(end + Vec2::new(dx, dy) * self.k, 3.6 * self.k, CREAM, ink);
                    }
                }
            }
            // Novelty: a star that twinkles.
            2 => {
                let grow = 1.0 + 0.15 * (self.t * 4.0).sin();
                let pts: Vec<Pos2> = (0..10)
                    .map(|i| {
                        let a = i as f32 / 10.0 * std::f32::consts::TAU - 1.57;
                        let rr = if i % 2 == 0 { 12.5 } else { 5.5 } * grow;
                        c + Vec2::new(a.cos(), a.sin()) * rr * self.k
                    })
                    .collect();
                self.p.add(Shape::Path(PathShape {
                    points: pts,
                    closed: true,
                    fill: MUSTARD,
                    stroke: Stroke::new(2.0 * self.k, INK).into(),
                }));
            }
            // Restart: a cracked egg, a new random body.
            _ => {
                self.p
                    .circle(c + Vec2::new(0.0, 1.5) * self.k, 9.0 * self.k, CREAM, ink);
                let zig = [
                    (-6.0, 1.0),
                    (-2.5, -3.0),
                    (1.0, 1.0),
                    (4.5, -3.0),
                    (7.0, 0.5),
                ];
                let pts: Vec<Pos2> = zig
                    .iter()
                    .map(|&(dx, dy)| c + Vec2::new(dx, dy) * self.k)
                    .collect();
                self.p.add(Shape::line(pts, Stroke::new(2.2 * self.k, INK)));
            }
        }
    }
}

/// A best distance as text with one decimal. An archive with no elite has NaN
/// for its best, and that gives "empty".
fn meters(best: f32) -> String {
    if best.is_finite() {
        format!("{best:.1} m")
    } else {
        "empty".into()
    }
}

/// Generations from `generation` to the next multiple of `MIGRATION_INTERVAL`,
/// which is 0 when `generation` is a multiple. The hub gets its copies when the
/// generation counter reaches a multiple, because `migrate_islands` runs right
/// after the counter is raised.
fn until_migration(generation: u32) -> u32 {
    (MIGRATION_INTERVAL - generation % MIGRATION_INTERVAL) % MIGRATION_INTERVAL
}

/// Canvas centers of the grass tops of the four isolated islands: upper left,
/// upper right, lower right, lower left.
const ISLAND_CENTERS: [(f32, f32); 4] = [
    (650.0, 330.0),
    (1100.0, 330.0),
    (1100.0, 560.0),
    (650.0, 560.0),
];
/// Canvas center of the hub island. The sunburst rays start here.
const HUB: (f32, f32) = (875.0, 468.0);

/// The hub: a smaller island in the middle that receives copies of every
/// isolated island's best elites. It shows its leader and its best distance.
fn hub(s: &Scene, snap: Option<&Snapshot>) {
    let (cx, cy) = HUB;
    s.poly(
        &[
            (cx - 92.0, cy + 8.0),
            (cx + 92.0, cy + 8.0),
            (cx + 54.0, cy + 40.0),
            (cx, cy + 58.0),
            (cx - 54.0, cy + 40.0),
        ],
        ROCK,
        3.5,
    );
    s.ellipse(cx, cy, 98.0, 40.0, GRASS);
    s.ellipse(cx, cy - 4.0, 76.0, 26.0, GRASS_LIGHT);
    let summary = snap.and_then(|snap| snap.islands.get(hub_island()));
    if let Some(creature) = summary.and_then(|i| i.leader.as_ref()) {
        thumbnail(s.p, creature, s.rect(cx - 32.0, cy - 62.0, 64.0, 64.0));
    }
    let best = summary.map_or(f32::NAN, |i| i.best);
    s.block(s.rect(cx - 82.0, cy + 6.0, 164.0, 32.0), CREAM, 8.0, 3.0);
    s.title(
        (cx, cy + 22.0),
        Align2::CENTER_CENTER,
        &format!("Hub  {}", meters(best)),
        22.0,
        INK,
    );
}

/// Isolated island `index`: its rock and grass, its niche plots with small
/// creatures, its leader and its sign.
fn island(s: &Scene, snap: Option<&Snapshot>, index: usize) {
    let (cx, cy) = ISLAND_CENTERS[index];
    // Rock underside, then the grass top.
    s.poly(
        &[
            (cx - 132.0, cy + 12.0),
            (cx + 132.0, cy + 12.0),
            (cx + 92.0, cy + 62.0),
            (cx + 32.0, cy + 84.0),
            (cx, cy + 108.0),
            (cx - 36.0, cy + 80.0),
            (cx - 94.0, cy + 58.0),
        ],
        ROCK,
        3.5,
    );
    // Rock crevices for visual texture.
    for ((ax, ay), (bx, by)) in [((-60.0, 40.0), (-20.0, 66.0)), ((52.0, 36.0), (30.0, 64.0))] {
        s.p.line_segment(
            [s.at(cx + ax, cy + ay), s.at(cx + bx, cy + by)],
            Stroke::new(4.0 * s.k, ROCK_DARK),
        );
    }
    s.ellipse(cx, cy, 142.0, 62.0, GRASS);
    s.ellipse(cx, cy - 6.0, 118.0, 42.0, GRASS_LIGHT);
    let summary = snap.and_then(|snap| snap.islands.get(index));
    let cells = summary.map_or(0, |i| i.cells);
    let most = snap
        .map(|snap| snap.islands.iter().map(|i| i.cells).max().unwrap_or(0))
        .unwrap_or(0)
        .max(1);
    // Six niche plots. The number of planted ones follows the island's
    // filled niches compared with the fullest island of the snapshot, the hub
    // and the wild islands included.
    let planted = if cells == 0 {
        0
    } else {
        (cells * 6).div_ceil(most).clamp(1, 6)
    };
    let plots = [
        (-86.0, -6.0),
        (0.0, -16.0),
        (86.0, -6.0),
        (-48.0, 24.0),
        (48.0, 24.0),
        (0.0, 38.0),
    ];
    for (i, (dx, dy)) in plots.iter().enumerate() {
        s.flat(
            s.rect(cx + dx - 26.0, cy + dy - 4.0, 52.0, 20.0),
            DIRT,
            7.0,
            2.5,
        );
        // Plot 1 is the leader's, drawn below.
        if i != 1 && i < planted {
            let phase = s.t * 3.0 + (index * 6 + i) as f32 * 1.3;
            s.critter(
                cx + dx,
                cy + dy + 4.0,
                ISLAND_TINT[(index + i) % 4],
                0.8,
                phase,
            );
        }
    }
    // The island's best creature stands on the middle plot, larger.
    if let Some(creature) = summary.and_then(|i| i.leader.as_ref()) {
        thumbnail(s.p, creature, s.rect(cx - 44.0, cy - 86.0, 88.0, 88.0));
    }
    // The sign: name, best distance, filled niches and the bodies in the
    // island's nurseries.
    let best = summary.map_or(f32::NAN, |i| i.best);
    let nursery = summary.map_or(0, |i| i.nursery);
    let upper = index < 2;
    // Upper islands carry their sign on a post above, lower ones hang it
    // below the rock, so no sign covers the hub or the routes.
    let area = (
        cx - 120.0,
        if upper { cy - 192.0 } else { cy + 76.0 },
        240.0,
        100.0,
    );
    if upper {
        s.p.line_segment(
            [s.at(cx, area.1 + area.3), s.at(cx, cy - 86.0)],
            Stroke::new(6.0 * s.k, INK),
        );
    }
    s.block(s.rect(area.0, area.1, area.2, area.3), PANEL, 10.0, 3.5);
    s.band(area.0, area.1, area.2, 34.0, ISLAND_TINT[index]);
    let head = if ISLAND_TINT[index] == MUSTARD {
        INK
    } else {
        CREAM
    };
    s.title(
        (cx, area.1 + 18.0),
        Align2::CENTER_CENTER,
        &format!("Island {}", index + 1),
        23.0,
        head,
    );
    s.text(
        area.0 + 14.0,
        area.1 + 40.0,
        area.2 - 28.0,
        &format!(
            "best {}   {} niches\nnursery {} bodies",
            meters(best),
            cells,
            nursery
        ),
        18.0,
        INK,
    );
}

/// A boat with a few crates, bobbing on its way. `tint` colors its sail and
/// `bob` is added to `y`.
fn boat(s: &Scene, x: f32, y: f32, tint: Color32, bob: f32) {
    let y = y + bob;
    s.poly(
        &[
            (x - 28.0, y),
            (x + 28.0, y),
            (x + 17.0, y + 16.0),
            (x - 17.0, y + 16.0),
        ],
        WOOD,
        3.0,
    );
    s.p.line_segment([s.at(x, y), s.at(x, y - 34.0)], Stroke::new(4.0 * s.k, INK));
    s.poly(
        &[
            (x + 2.0, y - 34.0),
            (x + 24.0, y - 12.0),
            (x + 2.0, y - 12.0),
        ],
        tint,
        2.5,
    );
    for i in 0..3 {
        let r = s.rect(x - 21.0 + i as f32 * 14.0, y - 10.0, 11.0, 10.0);
        s.flat(r, MUSTARD, 1.0, 2.0);
    }
}

/// The card of emitter `index`, `h` high with its top edge at `y`: its emblem,
/// name, breeding share and a short description. The share is the emitter's
/// live weight from `snap`, or its starting weight before the first snapshot.
fn workshop(s: &Scene, index: usize, snap: Option<&Snapshot>, y: f32, h: f32) {
    let (name, what) = match Emitter::ALL[index] {
        Emitter::Cma => (
            "CMA tuning",
            "Takes an elite and tunes its muscle timing and strength. Every good tweak nudges the next try.",
        ),
        Emitter::Structural => (
            "Anatomy mutations",
            "301 operators add, copy, fuse and move limbs and muscles. Sometimes crosses with a same-plan mate. Children are protected for 3 generations. A new body plan the island turns away goes to a second nursery, where it is tuned until it beats the island's elites.",
        ),
        Emitter::Novelty => (
            "Novelty",
            "Rewards odd ways of moving, so the archive keeps new niches. Also crosses sometimes.",
        ),
        Emitter::Restart => (
            "Immigrants",
            "New random bodies. Each island keeps a nursery of them for 10 generations, then the survivors compete with its elites. Also fills empty islands and re-tests elites after a world change.",
        ),
    };
    let share = snap.map_or(
        crate::qd::emitter_weights(&Default::default())[index],
        |snap| snap.emitter_weights[index],
    );
    let area = (20.0, y, 450.0, h);
    let accent = EMITTER_TINT[index];
    s.card(area, accent, 0, "");
    let head_color = if accent == MUSTARD { INK } else { CREAM };
    let pulse = 0.5 + 0.5 * (s.t * 2.2 + index as f32 * 1.6).sin();
    s.emblem(area.0 + 32.0, y + 23.0, index, pulse);
    s.title(
        (area.0 + 62.0, y + 24.0),
        Align2::LEFT_CENTER,
        name,
        27.0,
        head_color,
    );
    // The share as a pill.
    let pill = s.rect(area.0 + area.2 - 92.0, y + 8.0, 80.0, 30.0);
    s.flat(pill, CREAM, 15.0, 3.0);
    s.p.text(
        pill.center(),
        Align2::CENTER_CENTER,
        format!("{:.0}%", share * 100.0),
        s.title_font(24.0),
        INK,
    );
    s.card_text(area, what, 19.0);
}

/// The trial arena card, with two lanes, the 5 s gate and the finish line. One
/// creature passes the gate and one is stopped at it. The record check card
/// sits further down.
fn arena(s: &Scene) {
    s.card((1290.0, 84.0, 450.0, 356.0), RED, 3, "Trial arena");
    let (lx, ly, lw, lh) = (1308.0, 148.0, 414.0, 92.0);
    s.flat(s.rect(lx, ly, lw, lh), DIRT, 8.0, 3.5);
    // The two lanes.
    for row in 0..2 {
        let y = ly + 32.0 + row as f32 * 38.0;
        for i in 0..14 {
            let x = lx + 14.0 + i as f32 * 28.0;
            s.p.line_segment(
                [s.at(x, y + 6.0), s.at(x + 14.0, y + 6.0)],
                Stroke::new(2.5 * s.k, INK_SOFT),
            );
        }
    }
    // Gate at 5 s (a quarter of the 20 s track), finish line at 20 s.
    let track0 = lx + 28.0;
    let track1 = lx + lw - 34.0;
    let gate_x = track0 + (track1 - track0) * 0.25;
    for x in [gate_x - 12.0, gate_x + 12.0] {
        s.flat(
            s.rect(x - 3.0, ly - 12.0, 6.0, lh + 12.0),
            ROCK_DARK,
            1.0,
            2.5,
        );
    }
    s.flat(s.rect(gate_x - 20.0, ly - 18.0, 40.0, 14.0), RED, 3.0, 3.0);
    for i in 0..8 {
        let c = if i % 2 == 0 { INK } else { CREAM };
        s.flat(
            s.rect(track1 + 6.0, ly + 4.0 + i as f32 * 10.5, 10.0, 10.5),
            c,
            0.0,
            1.0,
        );
    }
    // A creature that passes the gate and one that is stopped at it.
    let t = (s.t / 9.0).fract();
    let run = (t / 0.85).min(1.0);
    s.critter(
        track0 + (track1 - track0 - 6.0) * run,
        ly + 36.0,
        EMITTER_TINT[0],
        0.95,
        s.t * 9.0,
    );
    let stop = (t / 0.25).min(1.0);
    let sx = track0 + (gate_x - 18.0 - track0) * stop;
    if stop < 1.0 {
        s.critter(sx, ly + 74.0, EMITTER_TINT[2], 0.95, s.t * 9.0);
    } else {
        // Stopped: it turns grey and still, with a red cross past the gate.
        s.critter(sx, ly + 74.0, Color32::from_rgb(150, 140, 120), 0.95, 0.0);
        let c = s.at(gate_x + 38.0, ly + 56.0);
        let arm = 9.0 * s.k;
        for d in [Vec2::new(arm, arm), Vec2::new(arm, -arm)] {
            s.p.line_segment([c - d, c + d], Stroke::new(8.0 * s.k, INK));
            s.p.line_segment([c - d, c + d], Stroke::new(4.5 * s.k, RED));
        }
    }
    s.label(gate_x, ly + lh + 18.0, "5 s gate", 18.0, INK);
    s.label(track1 + 4.0, ly + lh + 18.0, "20 s", 18.0, INK);
    s.text(
        1308.0,
        ly + lh + 34.0,
        414.0,
        "Every creature runs a 20 s trial. At 5 s a creature below the top 10% bar of the last generation is stopped and enters no archive. Survivors run on. Score is horizontal distance only.",
        19.0,
        INK,
    );
    // The record check card.
    let rc = (1290.0, 548.0, 450.0, 196.0);
    s.card(rc, BLU, 4, "Record check");
    s.card_text(
        rc,
        "A creature that would beat its island's record runs its trial again from the same pose, with physics at 2x the step rate. The worse distance counts. A second trial that fails the 5 s gate keeps it out.",
        19.0,
    );
}

/// Paints the whole poster into `rect`, scaled to fit and centered. The
/// numbers on it come from `snap`, and are zero or empty without one.
fn paint(ui: &egui::Ui, rect: Rect, snap: Option<&Snapshot>) {
    let p = ui.painter_at(rect);
    let k = (rect.width() / W).min(rect.height() / H);
    let size = Vec2::new(W, H) * k;
    let origin = rect.center() - size * 0.5;
    let t = ui.input(|i| i.time) as f32;
    let s = Scene {
        p: &p,
        origin,
        k,
        t,
    };

    // The poster: paper with sunburst rays from the hub, a double frame.
    p.rect_filled(rect, 0.0, PAPER);
    let hub_c = s.at(HUB.0, HUB.1);
    let reach = (W + H) * k;
    for i in 0..18 {
        let a0 = i as f32 / 18.0 * std::f32::consts::TAU + t * 0.01;
        let a1 = a0 + std::f32::consts::TAU / 36.0;
        p.add(Shape::convex_polygon(
            vec![
                hub_c,
                hub_c + Vec2::new(a0.cos(), a0.sin()) * reach,
                hub_c + Vec2::new(a1.cos(), a1.sin()) * reach,
            ],
            PAPER_RAY,
            Stroke::NONE,
        ));
    }
    s.flat(s.rect(0.0, 0.0, W, H), Color32::TRANSPARENT, 22.0, 6.0);
    p.rect_stroke(
        s.rect(12.0, 12.0, W - 24.0, H - 24.0),
        18.0 * k,
        Stroke::new(3.0 * k, RED),
        egui::StrokeKind::Middle,
    );

    // Title ribbon with the generation.
    s.poly(
        &[
            (520.0, 14.0),
            (1240.0, 14.0),
            (1272.0, 42.0),
            (1240.0, 70.0),
            (520.0, 70.0),
            (488.0, 42.0),
        ],
        RED,
        4.0,
    );
    let generation = snap.map_or(0, |snap| snap.generation);
    s.title(
        (880.0, 43.0),
        Align2::CENTER_CENTER,
        &format!("How evolution works   generation {generation}"),
        38.0,
        CREAM,
    );

    // Parents flow from the island archives into the workshops.
    s.text(
        66.0,
        86.0,
        400.0,
        "Parents come from the island archives.",
        19.0,
        INK,
    );
    // The four isolated islands, the hub between them and the boats that
    // carry copies to the hub. Nothing sails back. Each route runs from the
    // rim of an island, in island order, to the rim of the hub.
    let routes = [
        ((748.0, 392.0), (810.0, 440.0)),
        ((1002.0, 392.0), (940.0, 440.0)),
        ((1002.0, 536.0), (944.0, 498.0)),
        ((748.0, 536.0), (806.0, 498.0)),
    ];
    for i in 0..4 {
        island(&s, snap, i);
    }
    for (a, b) in routes {
        s.arrow(a, b, MUSTARD);
    }
    hub(&s, snap);
    // Each boat sails 80% of its route in 7 s and starts again. The four are
    // a quarter of a cycle apart.
    for (i, (a, b)) in routes.iter().enumerate() {
        let f = (t / 7.0 + i as f32 * 0.25).fract() * 0.8;
        let x = a.0 + (b.0 - a.0) * f;
        let y = a.1 + (b.1 - a.1) * f;
        boat(
            &s,
            x,
            y - 12.0,
            EMITTER_TINT[i],
            (t * 2.4 + i as f32).sin() * 3.0,
        );
    }
    s.arrow((510.0, 330.0), (472.0, 330.0), RED);
    s.arrow((510.0, 560.0), (472.0, 560.0), RED);
    s.badge(40.0, 98.0, 1);
    // The hub gets its copies as the counter is raised at the end of this
    // generation, so the countdown asks about the counter's next value.
    let next = until_migration(generation + 1);
    let when = if next == 0 {
        "The boats sail after this generation".to_owned()
    } else {
        format!("Boats sail in {next} generations")
    };
    s.text(
        520.0,
        84.0,
        740.0,
        &format!(
            "The four islands never mix. Every {MIGRATION_INTERVAL} generations the hub gets a copy of each island's fastest {:.0}% of elites and breeds from them. {when}.",
            MIGRATION_SHARE * 100.0
        ),
        18.0,
        INK,
    );

    // Workshops on the left: one card for each emitter in `Emitter::ALL`
    // order, each with a fixed height that fits its text.
    let mut y = 140.0;
    for (i, h) in [138.0, 244.0, 118.0, 204.0].into_iter().enumerate() {
        workshop(&s, i, snap, y, h);
        y += h + 16.0;
    }

    // Arena on the right and the arrow into it.
    arena(&s);
    s.text(1290.0, 20.0, 300.0, "children go to trial", 19.0, INK);
    s.arrow_path(&[(1282.0, 62.0), (1420.0, 62.0), (1420.0, 84.0)], RED);
    s.text(
        1316.0,
        452.0,
        420.0,
        "Survivors are offered to their island archive and to the global archive.",
        18.0,
        INK,
    );
    s.arrow_path(
        &[
            (1300.0, 478.0),
            (1276.0, 478.0),
            (1276.0, 822.0),
            (1262.0, 822.0),
        ],
        MUSTARD,
    );

    // Global archive.
    let (cells, size_now) = snap.map_or((0, 0), |snap| (snap.movement_cells, snap.archive_size));
    let hall = (520.0, 748.0, 740.0, 172.0);
    s.card(hall, BLU, 5, "Global archive");
    s.card_text(
        hall,
        &format!(
            "{cells} ways of moving filled, {size_now} elites. Every evaluated creature is offered to it and the best one in each way of moving and body shape and size stays. It is the record: no parent comes from it. When the world changes, each island's elites are tested again on their own island, and the emitter stats and CMA state start over."
        ),
        18.5,
    );

    // Champion podium: the leader with the best distance on any island of the
    // snapshot, the hub and the wild islands included.
    let champion = snap.and_then(|snap| {
        snap.islands
            .iter()
            .filter_map(|i| i.leader.as_ref().map(|c| (i.best, c)))
            .max_by(|a, b| a.0.total_cmp(&b.0))
    });
    s.block(s.rect(1360.0, 836.0, 290.0, 66.0), WOOD, 8.0, 3.5);
    s.band(1360.0, 836.0, 290.0, 24.0, MUSTARD);
    let best_text = champion.map_or("no champion yet".to_owned(), |(best, _)| {
        format!("Champion {}", meters(best))
    });
    s.title(
        (1505.0, 882.0),
        Align2::CENTER_CENTER,
        &best_text,
        25.0,
        CREAM,
    );
    if let Some((_, creature)) = champion {
        thumbnail(&p, creature, s.rect(1410.0, 744.0, 190.0, 92.0));
    }

    // How one creature is made, as a strip of five steps.
    s.flat(s.rect(20.0, 926.0, 300.0, 30.0), RED, 8.0, 3.5);
    s.title(
        (170.0, 941.0),
        Align2::CENTER_CENTER,
        "How a child is made",
        23.0,
        CREAM,
    );
    let steps = [
        "Pick a parent from an island archive.",
        "A workshop changes it.",
        "It runs the 20 s trial.",
        "A new island record gets its record check.",
        "It is offered to the archives.",
    ];
    let sw = (W - 40.0 - 4.0 * 14.0) / 5.0;
    for (i, step) in steps.iter().enumerate() {
        let x = 20.0 + i as f32 * (sw + 14.0);
        s.block(s.rect(x, 964.0, sw, 62.0), PANEL, 10.0, 3.5);
        s.badge(x + 30.0, 995.0, i as u32 + 1);
        s.text(x + 60.0, 972.0, sw - 72.0, step, 18.0, INK);
        if i < 4 {
            s.arrow((x + sw + 1.0, 995.0), (x + sw + 13.0, 995.0), MUSTARD);
        }
    }
}

/// Draws the "How evolution works" window while `*open` is true. The window's
/// close button clears `*open`, and the app sets it to open the window.
/// `snapshot` is the latest worker snapshot, or `None` before the first one.
/// While the window is open it asks for a repaint every 33 ms, for the
/// animation.
pub fn show(ctx: &egui::Context, snapshot: Option<&Snapshot>, open: &mut bool) {
    let screen = ctx.content_rect();
    // Most of the screen, at the poster's shape.
    let max = screen.size() * Vec2::new(0.96, 0.92) - Vec2::new(0.0, 36.0);
    let fit = (max.x / W).min(max.y / H);
    let size = Vec2::new(W, H) * fit;
    egui::Window::new("How evolution works")
        .open(open)
        .collapsible(false)
        .resizable(true)
        .default_size(size)
        .default_pos(screen.center() - size * 0.5 - Vec2::new(0.0, 14.0))
        .min_size(Vec2::new(760.0, 440.0))
        .frame(
            egui::Frame::window(&ctx.global_style())
                .inner_margin(0.0)
                .fill(PAPER),
        )
        .show(ctx, |ui| {
            let avail = ui.available_size();
            let k = (avail.x / W).min(avail.y / H).max(MIN_SCALE);
            let canvas = Vec2::new(W, H) * k;
            // The poster fills the window, centered when the window has a
            // different shape. Below the minimum scale it scrolls.
            let draw = |ui: &mut egui::Ui| {
                let size = Vec2::new(avail.x.max(canvas.x), avail.y.max(canvas.y));
                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                paint(ui, rect, snapshot);
            };
            if canvas.x > avail.x + 1.0 || canvas.y > avail.y + 1.0 {
                egui::ScrollArea::both().show(ui, draw);
            } else {
                draw(ui);
            }
        });
    if *open {
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_migration_countdown_hits_zero_on_the_interval() {
        assert_eq!(until_migration(0), 0);
        assert_eq!(until_migration(1), MIGRATION_INTERVAL - 1);
        assert_eq!(until_migration(MIGRATION_INTERVAL), 0);
        assert_eq!(
            until_migration(MIGRATION_INTERVAL + 5),
            MIGRATION_INTERVAL - 5
        );
    }
}
