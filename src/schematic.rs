//! "How evolution works": a painted picture of the search, with live numbers
//! from the worker snapshot. Everything is drawn with egui's painter on a
//! fixed 1000 x 720 canvas that scales to the window. Colors are fixed and
//! do not follow the theme: it is drawn like a worn control-room board, pale
//! text on dark steel plates with amber, rust, cold blue and olive marks, so
//! it reads the same in light and dark mode and matches the scene.

use crate::qd::Emitter;
use crate::storage::{MIGRATION_INTERVAL, MIGRATION_SHARE, hub_island};
use crate::ui::thumbnail;
use crate::worker::Snapshot;
use eframe::egui::{
    self, Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke, Vec2, epaint::PathShape,
};

const W: f32 = 1000.0;
const H: f32 = 720.0;
/// Outlines.
const INK: Color32 = Color32::from_rgb(12, 12, 11);
/// Text on the plates.
const TEXT: Color32 = Color32::from_rgb(226, 220, 204);
const PARCHMENT: Color32 = Color32::from_rgb(40, 41, 38);
const PARCHMENT_DARK: Color32 = Color32::from_rgb(84, 82, 72);
const CREAM: Color32 = Color32::from_rgb(56, 56, 51);
const WOOD: Color32 = Color32::from_rgb(122, 76, 46);
const WOOD_DARK: Color32 = Color32::from_rgb(92, 60, 38);
const GRASS: Color32 = Color32::from_rgb(84, 94, 54);
const GRASS_LIGHT: Color32 = Color32::from_rgb(106, 116, 66);
const ROCK: Color32 = Color32::from_rgb(70, 66, 58);
const DIRT: Color32 = Color32::from_rgb(92, 78, 58);
/// The emitters, in the colors the island cards use for them.
const ROOF: [Color32; 4] = [
    Color32::from_rgb(222, 160, 60),
    Color32::from_rgb(178, 92, 58),
    Color32::from_rgb(96, 154, 196),
    Color32::from_rgb(132, 140, 76),
];
const RED: Color32 = Color32::from_rgb(170, 62, 40);
const GOLD: Color32 = Color32::from_rgb(226, 166, 58);

/// The fixed-canvas painter: virtual coordinates in, screen coordinates out.
struct Scene<'a> {
    p: &'a Painter,
    origin: Pos2,
    k: f32,
}

impl Scene<'_> {
    fn at(&self, x: f32, y: f32) -> Pos2 {
        self.origin + Vec2::new(x, y) * self.k
    }
    fn rect(&self, x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::from_min_size(self.at(x, y), Vec2::new(w, h) * self.k)
    }
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
    fn block(&self, r: Rect, fill: Color32, round: f32, outline: f32) {
        self.p.rect(
            r,
            round * self.k,
            fill,
            self.stroke(outline),
            egui::StrokeKind::Middle,
        );
    }
    fn ellipse(&self, cx: f32, cy: f32, rx: f32, ry: f32, fill: Color32) {
        let points: Vec<(f32, f32)> = (0..40)
            .map(|i| {
                let a = i as f32 / 40.0 * std::f32::consts::TAU;
                (cx + rx * a.cos(), cy + ry * a.sin())
            })
            .collect();
        self.poly(&points, fill, 3.0);
    }
    fn font(&self, size: f32) -> FontId {
        FontId::proportional((size * self.k).max(9.0))
    }
    /// Wrapped text inside `r`, from the top left corner.
    fn text(&self, r: Rect, text: &str, size: f32, color: Color32) {
        let galley = self
            .p
            .layout(text.to_owned(), self.font(size), color, r.width().max(1.0));
        self.p.galley(r.min, galley, color);
    }
    /// One centered line.
    fn label(&self, x: f32, y: f32, text: &str, size: f32, color: Color32) {
        self.p.text(
            self.at(x, y),
            Align2::CENTER_CENTER,
            text,
            self.font(size),
            color,
        );
    }
    /// A cream sign with a title line and body text.
    fn sign(&self, area: (f32, f32, f32, f32), num: u32, title: &str, body: &str) {
        let (x, y, w, h) = area;
        let r = self.rect(x, y, w, h);
        self.block(r, CREAM, 8.0, 3.0);
        let pad = 8.0 * self.k;
        let inner = r.shrink2(Vec2::new(pad, 5.0 * self.k));
        let indent = if num > 0 { 26.0 * self.k } else { 0.0 };
        let head = self.p.layout(
            title.to_owned(),
            self.font(14.5),
            TEXT,
            inner.width() - indent,
        );
        let head_h = head.size().y;
        if num > 0 {
            self.badge(x + 17.0, y + 5.0 + head_h / self.k * 0.5, num);
        }
        self.p.galley(inner.min + Vec2::new(indent, 0.0), head, TEXT);
        self.text(
            Rect::from_min_max(
                inner.min + Vec2::new(0.0, head_h + if num > 0 { 6.0 * self.k } else { 0.0 }),
                inner.max,
            ),
            body,
            12.5,
            TEXT,
        );
    }
    fn arrow(&self, from: (f32, f32), to: (f32, f32), color: Color32) {
        let a = self.at(from.0, from.1);
        let b = self.at(to.0, to.1);
        let dir = (b - a).normalized();
        let side = Vec2::new(-dir.y, dir.x);
        let head = 14.0 * self.k;
        let base = b - dir * head;
        self.p
            .line_segment([a, base], Stroke::new(9.0 * self.k, INK));
        self.p
            .line_segment([a, base], Stroke::new(5.0 * self.k, color));
        self.p.add(Shape::Path(PathShape {
            points: vec![
                b + dir * 3.0,
                base + side * head * 0.75,
                base - side * head * 0.75,
            ],
            closed: true,
            fill: color,
            stroke: self.stroke(2.5).into(),
        }));
    }
    fn badge(&self, x: f32, y: f32, n: u32) {
        let c = self.at(x, y);
        self.p.circle(c, 11.0 * self.k, GOLD, self.stroke(2.5));
        self.p.text(
            c,
            Align2::CENTER_CENTER,
            n.to_string(),
            self.font(13.0),
            INK,
        );
    }
    /// A tiny stick creature for niche plots that the snapshot has no body for.
    fn critter(&self, x: f32, y: f32, tint: Color32) {
        let hip = self.at(x, y - 9.0);
        let head = self.at(x, y - 19.0);
        let foot_a = self.at(x - 7.0, y);
        let foot_b = self.at(x + 7.0, y);
        for f in [foot_a, foot_b] {
            self.p
                .line_segment([hip, f], Stroke::new(6.0 * self.k, INK));
            self.p
                .line_segment([hip, f], Stroke::new(3.0 * self.k, tint));
        }
        self.p
            .line_segment([hip, head], Stroke::new(6.0 * self.k, INK));
        self.p
            .line_segment([hip, head], Stroke::new(3.0 * self.k, tint));
        self.p.circle(head, 4.5 * self.k, tint, self.stroke(2.0));
    }
}

fn meters(best: f32) -> String {
    if best.is_finite() {
        format!("{best:.1} m")
    } else {
        "empty".into()
    }
}

/// Generations until the next migration (0 means this generation's end).
fn until_migration(generation: u32) -> u32 {
    (MIGRATION_INTERVAL - generation % MIGRATION_INTERVAL) % MIGRATION_INTERVAL
}

const ISLAND_CENTERS: [(f32, f32); 4] = [
    (385.0, 190.0),
    (615.0, 190.0),
    (615.0, 462.0),
    (385.0, 462.0),
];
const ISLAND_TINT: [Color32; 4] = ROOF;

/// The hub: a smaller island in the middle that receives copies of every
/// isolated island's best elites.
fn hub(s: &Scene, snap: Option<&Snapshot>) {
    let (cx, cy) = (500.0, 322.0);
    s.poly(
        &[
            (cx - 62.0, cy + 6.0),
            (cx + 62.0, cy + 6.0),
            (cx + 36.0, cy + 30.0),
            (cx, cy + 44.0),
            (cx - 36.0, cy + 30.0),
        ],
        ROCK,
        3.0,
    );
    s.ellipse(cx, cy, 66.0, 28.0, GRASS);
    s.ellipse(cx, cy - 3.0, 52.0, 19.0, GRASS_LIGHT);
    let summary = snap.and_then(|snap| snap.islands.get(hub_island()));
    if let Some(creature) = summary.and_then(|i| i.leader.as_ref()) {
        thumbnail(s.p, creature, s.rect(cx - 20.0, cy - 40.0, 40.0, 40.0));
    }
    let best = summary.map_or(f32::NAN, |i| i.best);
    let plate = s.rect(cx - 54.0, cy + 8.0, 108.0, 20.0);
    s.block(plate, CREAM, 5.0, 2.0);
    s.label(cx, cy + 18.0, &format!("Hub  {}", meters(best)), 11.5, TEXT);
}

fn island(s: &Scene, snap: Option<&Snapshot>, index: usize) {
    let (cx, cy) = ISLAND_CENTERS[index];
    // Rock underside, then the grass top.
    s.poly(
        &[
            (cx - 92.0, cy + 10.0),
            (cx + 92.0, cy + 10.0),
            (cx + 60.0, cy + 46.0),
            (cx + 20.0, cy + 60.0),
            (cx, cy + 74.0),
            (cx - 24.0, cy + 56.0),
            (cx - 62.0, cy + 42.0),
        ],
        ROCK,
        3.0,
    );
    s.ellipse(cx, cy, 96.0, 50.0, GRASS);
    s.ellipse(cx, cy - 5.0, 78.0, 34.0, GRASS_LIGHT);
    let summary = snap.and_then(|snap| snap.islands.get(index));
    let cells = summary.map_or(0, |i| i.cells);
    let most = snap
        .map(|snap| snap.islands.iter().map(|i| i.cells).max().unwrap_or(0))
        .unwrap_or(0)
        .max(1);
    // Six niche plots. The number of planted ones follows the island's
    // filled niches compared with the fullest island.
    let planted = if cells == 0 {
        0
    } else {
        (cells * 6).div_ceil(most).clamp(1, 6)
    };
    let plots = [
        (-52.0, -8.0),
        (0.0, -14.0),
        (52.0, -8.0),
        (-30.0, 14.0),
        (26.0, 14.0),
        (0.0, 24.0),
    ];
    for (i, (dx, dy)) in plots.iter().enumerate() {
        let r = s.rect(cx + dx - 17.0, cy + dy - 4.0, 34.0, 14.0);
        s.block(r, DIRT, 5.0, 2.0);
        if i == 1 {
            continue;
        }
        if i < planted {
            s.critter(cx + dx, cy + dy + 2.0, ISLAND_TINT[(index + i) % 4]);
        }
    }
    // The island's best creature stands on the middle plot, larger.
    let leader = summary.and_then(|i| i.leader.as_ref());
    let stage = s.rect(cx - 24.0, cy - 50.0, 48.0, 50.0);
    if let Some(creature) = leader {
        thumbnail(s.p, creature, stage);
    }
    // Sign above.
    let best = summary.map_or(f32::NAN, |i| i.best);
    let title = format!("Island {}", index + 1);
    let body = format!("best {}\n{} niches", meters(best), cells);
    s.sign((cx - 66.0, cy - 106.0, 132.0, 56.0), 0, &title, &body);
}

fn boat(s: &Scene, x: f32, y: f32, tint: Color32) {
    s.poly(
        &[
            (x - 20.0, y),
            (x + 20.0, y),
            (x + 12.0, y + 12.0),
            (x - 12.0, y + 12.0),
        ],
        WOOD,
        2.5,
    );
    s.p.line_segment([s.at(x, y), s.at(x, y - 24.0)], Stroke::new(3.0 * s.k, INK));
    s.poly(
        &[(x + 1.0, y - 24.0), (x + 17.0, y - 8.0), (x + 1.0, y - 8.0)],
        tint,
        2.0,
    );
    // Cargo: three little crates.
    for i in 0..3 {
        let r = s.rect(x - 15.0 + i as f32 * 10.0, y - 7.0, 8.0, 7.0);
        s.block(r, GOLD, 1.0, 1.5);
    }
}

fn workshop(s: &Scene, index: usize, snap: Option<&Snapshot>) {
    let y = 90.0 + index as f32 * 110.0;
    let x = 14.0;
    let (name, what) = match Emitter::ALL[index] {
        Emitter::Cma => (
            "CMA tuning",
            "Takes an elite and tunes its muscle timing and strength. Every good tweak nudges the next try.",
        ),
        Emitter::Structural => (
            "Anatomy mutations",
            "44 operators add, copy, fuse and move limbs and muscles. Sometimes crosses with a same-plan mate. Children are protected for 3 generations.",
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
    // House: roof and wall.
    s.poly(
        &[
            (x - 2.0, y + 22.0),
            (x + 116.0, y - 2.0),
            (x + 234.0, y + 22.0),
        ],
        ROOF[index],
        3.0,
    );
    let wall = s.rect(x + 6.0, y + 22.0, 220.0, 82.0);
    s.block(wall, CREAM, 4.0, 3.0);
    let inner = wall.shrink2(Vec2::new(7.0 * s.k, 3.0 * s.k));
    let head = format!("{name}   {:.0}%", share * 100.0);
    let g = s.p.layout(head, s.font(14.5), TEXT, inner.width());
    let hh = g.size().y;
    s.p.galley(inner.min, g, TEXT);
    s.text(
        Rect::from_min_max(inner.min + Vec2::new(0.0, hh), inner.max),
        what,
        11.0,
        TEXT,
    );
}

fn arena(s: &Scene) {
    // Track.
    s.sign((770.0, 64.0, 222.0, 28.0), 3, "Trial arena", "");
    let lane = s.rect(776.0, 130.0, 210.0, 34.0);
    s.block(lane, DIRT, 6.0, 3.0);
    for i in 0..9 {
        let x = 786.0 + i as f32 * 22.0;
        s.p.line_segment(
            [s.at(x, 147.0), s.at(x + 10.0, 147.0)],
            Stroke::new(2.0 * s.k, PARCHMENT_DARK),
        );
    }
    // Gate at 5 s (a quarter of the 20 s track), finish line at 20 s.
    let gate_x = 776.0 + 210.0 * 0.25;
    for x in [gate_x - 10.0, gate_x + 10.0] {
        s.block(s.rect(x - 2.0, 104.0, 4.0, 60.0), WOOD_DARK, 1.0, 2.0);
    }
    s.block(s.rect(gate_x - 14.0, 100.0, 28.0, 10.0), RED, 2.0, 2.5);
    let finish_x = 776.0 + 200.0;
    for i in 0..6 {
        let c = if i % 2 == 0 { INK } else { CREAM };
        s.block(
            s.rect(finish_x, 130.0 + i as f32 * 6.0, 8.0, 6.0),
            c,
            0.0,
            1.0,
        );
    }
    s.critter(806.0, 152.0, ROOF[1]);
    s.label(gate_x, 178.0, "5 s gate", 12.5, TEXT);
    s.label(finish_x + 2.0, 178.0, "20 s", 12.5, TEXT);
    s.text(
        s.rect(776.0, 190.0, 214.0, 96.0),
        "Every creature runs a 20 s trial. At 5 s a creature below the top 20% bar of the last generation is stopped and enters no archive. Survivors run on. Score is horizontal distance only.",
        12.0,
        TEXT,
    );
    // Fine check.
    s.sign(
        (770.0, 300.0, 222.0, 150.0),
        5,
        "Fine check",
        "A contender for an archive cell runs again as a nudged copy: pose moved by up to 2 cm, grip changed by up to 10%, physics at 4x the step rate. The worse distance counts. A check that fails the 5 s gate keeps it out.",
    );
}

/// Paints the whole picture into `rect`.
fn paint(ui: &egui::Ui, rect: Rect, snap: Option<&Snapshot>) {
    let p = ui.painter_at(rect);
    let k = (rect.width() / W).min(rect.height() / H);
    let size = Vec2::new(W, H) * k;
    let origin = rect.center() - size * 0.5;
    let s = Scene { p: &p, origin, k };

    // Parchment panel.
    s.block(s.rect(0.0, 0.0, W, H), PARCHMENT, 18.0, 5.0);
    s.block(s.rect(8.0, 8.0, W - 16.0, H - 16.0), PARCHMENT, 14.0, 1.5);

    // Title ribbon.
    s.poly(
        &[
            (290.0, 14.0),
            (710.0, 14.0),
            (730.0, 30.0),
            (710.0, 48.0),
            (290.0, 48.0),
            (270.0, 30.0),
        ],
        RED,
        3.5,
    );
    let generation = snap.map_or(0, |snap| snap.generation);
    s.label(
        500.0,
        31.0,
        &format!("How evolution works  (generation {generation})"),
        20.0,
        TEXT,
    );

    // The four isolated islands, the hub between them, and the boats that
    // carry copies to the hub. Nothing sails back.
    for i in 0..4 {
        island(&s, snap, i);
    }
    s.arrow((440.0, 262.0), (452.0, 296.0), WOOD_DARK);
    s.arrow((560.0, 262.0), (548.0, 296.0), WOOD_DARK);
    s.arrow((462.0, 396.0), (470.0, 362.0), WOOD_DARK);
    s.arrow((538.0, 396.0), (530.0, 362.0), WOOD_DARK);
    hub(&s, snap);
    boat(&s, 398.0, 322.0, ROOF[0]);
    boat(&s, 602.0, 322.0, ROOF[1]);
    let next = until_migration(generation);
    let when = if next == 0 {
        "The boats sail after this generation".to_owned()
    } else {
        format!("Boats sail in {next} generations")
    };
    s.text(
        s.rect(300.0, 52.0, 400.0, 30.0),
        &format!(
            "The four islands never mix. Every {MIGRATION_INTERVAL} generations the hub gets a copy of each island's fastest {:.0}% of elites and breeds from them. {when}.",
            MIGRATION_SHARE * 100.0
        ),
        10.5,
        TEXT,
    );
    s.badge(300.0, 100.0, 1);

    // Workshops on the left.
    for i in 0..4 {
        workshop(&s, i, snap);
    }
    s.badge(116.0, 92.0, 2);
    s.arrow((250.0, 250.0), (284.0, 250.0), RED);
    s.text(
        s.rect(14.0, 60.0, 232.0, 30.0),
        "Parents come from the island archives.",
        11.5,
        TEXT,
    );

    // Arena on the right and the arrow into it.
    arena(&s);
    s.arrow((732.0, 32.0), (872.0, 62.0), RED);
    s.text(
        s.rect(748.0, 12.0, 120.0, 16.0),
        "children go to trial",
        11.0,
        TEXT,
    );
    s.arrow((770.0, 526.0), (626.0, 588.0), GOLD);
    s.text(
        s.rect(776.0, 462.0, 214.0, 50.0),
        "Survivors are offered to their island archive and to the global archive.",
        11.5,
        TEXT,
    );

    // Global archive.
    let (cells, size_now) = snap.map_or((0, 0), |snap| (snap.archive_cells, snap.archive_size));
    let hall = s.rect(262.0, 590.0, 350.0, 116.0);
    s.block(hall, CREAM, 10.0, 3.5);
    s.poly(&[(252.0, 594.0), (437.0, 562.0), (622.0, 594.0)], WOOD, 3.5);
    s.text(
        hall.shrink(9.0 * s.k),
        &format!(
            "Global archive\n{cells} behavior niches filled, {size_now} elites.\nEvery evaluated creature is offered to it and the best one in each niche stays. It is the record: no parent comes from it. When the world changes, each island's elites are tested again on their own island, and the emitter stats and CMA state start over."
        ),
        12.0,
        TEXT,
    );
    s.badge(437.0, 580.0, 6);

    // Champion podium.
    let champion = snap.and_then(|snap| {
        snap.islands
            .iter()
            .filter_map(|i| i.leader.as_ref().map(|c| (i.best, c)))
            .max_by(|a, b| a.0.total_cmp(&b.0))
    });
    s.block(s.rect(650.0, 650.0, 200.0, 50.0), WOOD, 4.0, 3.5);
    s.block(s.rect(650.0, 650.0, 200.0, 14.0), GOLD, 4.0, 3.0);
    let best_text = champion.map_or("no champion yet".to_owned(), |(best, _)| {
        format!("Champion {}", meters(best))
    });
    s.label(750.0, 683.0, &best_text, 14.5, TEXT);
    if let Some((_, creature)) = champion {
        thumbnail(&p, creature, s.rect(700.0, 570.0, 100.0, 76.0));
    }
    s.arrow((622.0, 630.0), (648.0, 630.0), GOLD);

    // How one creature is made.
    s.sign(
        (14.0, 534.0, 232.0, 170.0),
        4,
        "How a child is made",
        "One: pick a parent from an island archive. Two: a workshop changes it. Three: it runs the 20 s trial. Four: a contender gets the fine check. Five: it is offered to the archives. The four islands never share creatures. Only the hub gets copies, when the boats sail.",
    );
}

/// Opens or closes the schematic window. `open` is the UI flag; the window's
/// close button clears it.
pub fn show(ctx: &egui::Context, snapshot: Option<&Snapshot>, open: &mut bool) {
    egui::Window::new("How evolution works")
        .open(open)
        .collapsible(false)
        .resizable(true)
        .default_size(Vec2::new(1000.0, 740.0))
        .min_size(Vec2::new(880.0, 640.0))
        .show(ctx, |ui| {
            let avail = ui.available_size();
            let (rect, _) = ui.allocate_exact_size(
                Vec2::new(avail.x.max(300.0), avail.y.max(220.0)),
                egui::Sense::hover(),
            );
            paint(ui, rect, snapshot);
        });
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
