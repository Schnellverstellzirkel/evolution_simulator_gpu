//! Paints the replay view's world, an overcast industrial wasteland, and the
//! look of each environment effect in it. Every effect is a faint overlay or
//! a few thin shapes in fixed scene colors, so the creature, the meter marks
//! and the text stay readable in both themes. Animation follows the replay
//! clock passed in as `time`, so a paused replay holds still and no shape
//! needs per-frame state. `ui/viewport.rs` calls these functions to paint the
//! replay scene, and `ui/race.rs` calls `backdrop` and `ground_body` for each
//! race lane.
use crate::{assets::Art, config::Config, environment::EFFECTS, theme::hash};
use eframe::egui::{
    self, Color32, Painter, Pos2, Rect, Stroke, Vec2,
    epaint::{Mesh, Vertex},
};

/// How far the effect called `name` is above its calm level: 0 at calm or
/// below, 1 at its harshest level. An unknown name gives 0. The grip also has
/// levels below calm, which `grip` reads.
fn amount(cfg: &Config, name: &str) -> f32 {
    EFFECTS.iter().find(|e| e.name == name).map_or(0.0, |e| {
        let top = e.levels.len().saturating_sub(1).max(1);
        let level = e.level(cfg);
        level.saturating_sub(e.calm) as f32 / (top - e.calm).max(1) as f32
    })
}

/// The grip effect's two sides as `(sandpaper, slipperiness)`. Sandpaper is 1
/// when the grip level is below calm and 0 otherwise. Slipperiness is the
/// grip's `amount`: 0 at calm and below, 1 at the last level.
fn grip(cfg: &Config) -> (f32, f32) {
    let Some(e) = EFFECTS.iter().find(|e| e.name == "Grip") else {
        return (0.0, 0.0);
    };
    let level = e.level(cfg);
    let rough = if level < e.calm { 1.0 } else { 0.0 };
    (rough, amount(cfg, "Grip"))
}

/// `color` at opacity `a`. Values of `a` outside 0 to 1 are clamped.
fn alpha(color: (u8, u8, u8), a: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(color.0, color.1, color.2, (a.clamp(0.0, 1.0) * 255.0) as u8)
}

/// A strip along the polyline `line` that fades from `top` on the line to
/// `bottom` at `depth` pixels below it. A negative `depth` reaches upward. A
/// line of fewer than two points paints nothing.
pub fn band(painter: &Painter, line: &[Pos2], depth: f32, top: Color32, bottom: Color32) {
    if line.len() < 2 {
        return;
    }
    let mut mesh = Mesh::default();
    for (i, p) in line.iter().enumerate() {
        mesh.vertices.push(Vertex {
            pos: *p,
            uv: egui::epaint::WHITE_UV,
            color: top,
        });
        mesh.vertices.push(Vertex {
            pos: *p + Vec2::new(0.0, depth),
            uv: egui::epaint::WHITE_UV,
            color: bottom,
        });
        if i > 0 {
            let k = (i * 2) as u32;
            mesh.indices
                .extend_from_slice(&[k - 2, k - 1, k, k - 1, k + 1, k]);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// Overlays on the sky, drawn over the backdrop and before the ground: a warm
/// glow in a heat wave, haze when the air thickens, wind streaks, and a tint
/// for how far up the autochange ladder the world has climbed. `time` is the
/// replay clock in seconds.
pub fn sky(painter: &Painter, rect: Rect, cfg: &Config, time: f32) {
    let heat = amount(cfg, "Heat wave");
    if heat > 0.0 {
        // A warm glow that grows toward the bottom of the view and fades out
        // toward the top, so the high sky stays as it is.
        let mut mesh = Mesh::default();
        let glow = alpha((226, 150, 64), 0.18 + 0.30 * heat);
        for (pos, color) in [
            (rect.left_top(), Color32::TRANSPARENT),
            (rect.right_top(), Color32::TRANSPARENT),
            (rect.right_bottom(), glow),
            (rect.left_bottom(), glow),
        ] {
            mesh.vertices.push(Vertex {
                pos,
                uv: egui::epaint::WHITE_UV,
                color,
            });
        }
        mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        painter.add(egui::Shape::mesh(mesh));
    }
    let air = amount(cfg, "Air");
    if air > 0.0 {
        painter.rect_filled(rect, 0, alpha((196, 200, 198), 0.08 + 0.22 * air));
    }
    let wind = amount(cfg, "Wind");
    if wind > 0.0 {
        // A headwind blows against +x travel, so streaks move left.
        let count = 5 + (10.0 * wind) as i64;
        let span = rect.width() + 120.0;
        for i in 0..count {
            let speed = 90.0 + 220.0 * wind * (0.6 + hash(i * 3));
            let x = rect.left() - 60.0 + (hash(i) * span - time * speed).rem_euclid(span);
            let y = rect.top() + rect.height() * (0.08 + 0.65 * hash(i * 7 + 1));
            let len = 16.0 + 40.0 * hash(i * 5 + 2) * (0.5 + wind);
            painter.line_segment(
                [Pos2::new(x, y), Pos2::new(x + len, y + len * 0.04)],
                Stroke::new(1.0, alpha((210, 212, 206), 0.10 + 0.14 * wind)),
            );
        }
    }
    if cfg.autochange > 0 {
        // Tint by how far up the autochange ladder the world has climbed, in
        // four stages.
        let lap = crate::environment::autochange_ladder().len().max(4);
        let quarter = (usize::from(cfg.autochange_step).min(lap - 1)) * 4 / lap;
        let tint = [
            (120, 130, 70),
            (220, 170, 70),
            (190, 100, 50),
            (100, 150, 190),
        ][quarter];
        painter.rect_filled(rect, 0, alpha(tint, 0.10));
    }
}

/// The water: a translucent murky body below the waterline at screen height
/// `line_y`, a wavy surface, faint streaks and a few rising bubbles. Nothing
/// is painted when the water effect is calm or the waterline is below the
/// view. The creature and the text are drawn after it, so the water never
/// covers them. `world_x` maps a screen x to meters, so the waves and bubbles
/// scroll with the ground. `pixels_per_meter` is the zoom.
pub fn water(
    painter: &Painter,
    rect: Rect,
    cfg: &Config,
    time: f32,
    line_y: f32,
    world_x: &dyn Fn(f32) -> f32,
    pixels_per_meter: f32,
) {
    let level = amount(cfg, "Water");
    if level <= 0.0 || line_y > rect.bottom() {
        return;
    }
    let top = line_y.clamp(rect.top(), rect.bottom());
    // Murky canal water: green-brown, dark in the depth.
    let body = alpha((52, 74, 62), 0.52 + 0.14 * level);
    let deep = alpha((14, 24, 20), 0.82 + 0.1 * level);
    let mut mesh = Mesh::default();
    for (pos, color) in [
        (Pos2::new(rect.left(), top), body),
        (Pos2::new(rect.right(), top), body),
        (Pos2::new(rect.right(), rect.bottom()), deep),
        (Pos2::new(rect.left(), rect.bottom()), deep),
    ] {
        mesh.vertices.push(Vertex {
            pos,
            uv: egui::epaint::WHITE_UV,
            color,
        });
    }
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    painter.add(egui::Shape::mesh(mesh));
    if line_y >= rect.top() {
        // The surface: a wave that drifts with the clock.
        let surface: Vec<Pos2> = (0..=(rect.width() / 6.0) as usize)
            .map(|i| {
                let x = rect.left() + i as f32 * 6.0;
                let m = world_x(x);
                let wave = (m * 2.6 + time * 2.0).sin() * 1.6 + (m * 6.1 - time * 3.1).sin() * 0.8;
                Pos2::new(x, line_y + wave)
            })
            .collect();
        painter.add(egui::Shape::line(
            surface,
            Stroke::new(2.0, alpha((186, 210, 198), 0.85)),
        ));
    }
    // Faint horizontal streaks below the surface.
    for k in 0..4 {
        let y = top + 16.0 + 22.0 * k as f32 + 6.0 * (time * 0.8 + k as f32).sin();
        if y > rect.bottom() {
            break;
        }
        let points: Vec<Pos2> = (0..=40)
            .map(|i| {
                let x = rect.left() + rect.width() * i as f32 / 40.0;
                let m = world_x(x);
                Pos2::new(x, y + (m * 1.7 + time * 1.5 + k as f32 * 2.0).sin() * 2.5)
            })
            .collect();
        painter.add(egui::Shape::line(
            points,
            Stroke::new(1.0, alpha((170, 200, 190), 0.12)),
        ));
    }
    // Bubbles that rise from the depth and fade before they break the surface.
    let first = (world_x(rect.left()) / 0.5).floor() as i64;
    let last = (world_x(rect.right()) / 0.5).ceil() as i64;
    let span = (rect.bottom() - top).max(1.0);
    for k in first..=last {
        if hash(k * 7) > 0.7 {
            continue;
        }
        let meters = (k as f32 + hash(k * 3)) * 0.5;
        let x = rect.left() + (meters - world_x(rect.left())) * pixels_per_meter;
        let rise = (time * (0.10 + 0.10 * hash(k * 5)) + hash(k * 11)).fract();
        let y = rect.bottom() - rise * span;
        painter.circle_stroke(
            Pos2::new(x, y),
            1.5 + 2.5 * hash(k * 13),
            Stroke::new(1.0, alpha((190, 212, 200), 0.45 * (1.0 - rise))),
        );
    }
}

/// A creature node close to the ground, for mud splashes. `ground` splashes
/// the ones that move faster than 30 pixels per second.
pub struct Foot {
    /// Where the foot meets the ground surface, in screen pixels.
    pub at: Pos2,
    /// Ground speed in pixels per second on screen.
    pub speed: f32,
    /// Index of the node in the creature. It seeds the droplets, so each foot
    /// throws its own pattern.
    pub id: usize,
}

/// Effects on and in the ground, drawn after it: a shade for heavy gravity,
/// dry earth and cracks, sandpaper grains, wet puddles or ice, ice patches,
/// ripples in a quake, brambles, and mud speckles and splashes. `surface`
/// maps a screen x to the ground line's screen y. `world_x` maps a screen x to
/// meters, so cracks and grains scroll with the ground. `pixels_per_meter` is
/// the zoom. `feet` are the nodes near the ground, which splash the mud.
#[allow(clippy::too_many_arguments)]
pub fn ground(
    painter: &Painter,
    rect: Rect,
    cfg: &Config,
    time: f32,
    surface: &dyn Fn(f32) -> f32,
    world_x: &dyn Fn(f32) -> f32,
    pixels_per_meter: f32,
    feet: &[Foot],
) {
    let step = 8.0;
    let n = (rect.width() / step) as usize + 2;
    let line: Vec<Pos2> = (0..n)
        .map(|i| {
            let x = rect.left() + i as f32 * step;
            Pos2::new(x, surface(x).clamp(rect.top() - 50.0, rect.bottom() + 50.0))
        })
        .collect();
    // Heavy gravity: a dark shade under the surface.
    let g = amount(cfg, "Gravity");
    if g > 0.0 {
        band(
            painter,
            &line,
            50.0,
            alpha((0, 0, 0), 0.14 + 0.30 * g),
            alpha((0, 0, 0), 0.0),
        );
    }
    let drought = amount(cfg, "Drought");
    let heat = amount(cfg, "Heat wave");
    let dry = drought.max(0.6 * heat);
    if dry > 0.0 {
        // Dry earth: a tan band under the surface, from a drought or, less,
        // from a heat wave.
        band(
            painter,
            &line,
            70.0,
            alpha((178, 148, 94), 0.20 + 0.40 * dry),
            alpha((178, 148, 94), 0.05 * dry),
        );
    }
    if drought > 0.0 {
        // Cracks: at most one in every 0.7 m of ground, more of them as the
        // drought deepens. Each is a short zigzag down from the surface.
        let first = (world_x(rect.left()) / 0.7).floor() as i64;
        let last = (world_x(rect.right()) / 0.7).ceil() as i64;
        let stroke = Stroke::new(1.2, alpha((30, 24, 16), 0.40 + 0.45 * drought));
        for k in first..=last {
            if hash(k * 11) > 0.35 + 0.65 * drought {
                continue;
            }
            let meters = (k as f32 + hash(k)) * 0.7;
            let x = rect.left() + (meters - world_x(rect.left())) * pixels_per_meter;
            let mut p = Pos2::new(x, surface(x) + 2.0);
            let mut points = vec![p];
            for j in 0..4 {
                p += Vec2::new(
                    (hash(k * 13 + j) - 0.5) * 12.0,
                    5.0 + 9.0 * hash(k * 17 + j) * (0.4 + drought),
                );
                points.push(p);
            }
            painter.add(egui::Shape::line(points, stroke));
        }
    }
    let (rough, slip) = grip(cfg);
    if rough > 0.0 {
        // Sandpaper: dark grains in a thin band under the surface.
        let first = (world_x(rect.left()) / 0.12).floor() as i64;
        let last = (world_x(rect.right()) / 0.12).ceil() as i64;
        for k in first..=last {
            let meters = (k as f32 + hash(k * 3)) * 0.12;
            let x = rect.left() + (meters - world_x(rect.left())) * pixels_per_meter;
            let y = surface(x) + 4.0 + 12.0 * hash(k * 5);
            painter.circle_filled(Pos2::new(x, y), 1.1, alpha((24, 22, 18), 0.65));
        }
    }
    if slip > 0.3 {
        // Wet ground darkens with a bright film. Ice glazes over it.
        let s = (slip - 0.3) / 0.7;
        let ice = slip > 0.9;
        if !ice {
            // Wet: the ground darkens, a bright film runs along it, and puddles
            // with slow ripples lie on it, one in every 1.3 m of ground.
            band(
                painter,
                &line,
                28.0,
                alpha((18, 32, 40), 0.40 + 0.25 * s),
                alpha((18, 32, 40), 0.05),
            );
            let first = (world_x(rect.left()) / 1.3).floor() as i64;
            let last = (world_x(rect.right()) / 1.3).ceil() as i64;
            for k in first..=last {
                let meters = (k as f32 + hash(k * 11)) * 1.3;
                let cx = rect.left() + (meters - world_x(rect.left())) * pixels_per_meter;
                let half = 26.0 + 30.0 * hash(k * 13);
                let mut lens = Vec::new();
                for step in -4..=4_i32 {
                    let t = step as f32 / 4.0;
                    let x = cx + t * half;
                    lens.push(Pos2::new(x, surface(x) + 1.0));
                }
                for step in (-4..=4_i32).rev() {
                    let t = step as f32 / 4.0;
                    let x = cx + t * half;
                    lens.push(Pos2::new(x, surface(x) + 2.0 + 9.0 * (1.0 - t * t)));
                }
                painter.add(egui::Shape::convex_polygon(
                    lens,
                    alpha((78, 110, 120), 0.80 + 0.15 * s),
                    Stroke::new(1.5, alpha((196, 214, 214), 0.90)),
                ));
                // A ripple widens and fades on each puddle.
                let age = (time * 0.5 + hash(k * 17)).fract();
                let r = half * (0.2 + 0.7 * age);
                let y = surface(cx) + 4.0;
                painter.line_segment(
                    [Pos2::new(cx - r, y), Pos2::new(cx + r, y)],
                    Stroke::new(1.2, alpha((255, 255, 255), 0.7 * (1.0 - age))),
                );
            }
        } else {
            let color = (200, 222, 234);
            band(
                painter,
                &line,
                16.0,
                alpha(color, 0.25 + 0.45 * s),
                alpha(color, 0.05),
            );
        }
        let shine: Vec<Pos2> = line.iter().map(|p| *p + Vec2::new(0.0, 3.0)).collect();
        painter.add(egui::Shape::line(
            shine,
            Stroke::new(1.5, alpha((255, 255, 255), 0.30 + 0.40 * s)),
        ));
        if ice {
            // Glints twinkle on the ice. They keep their place on screen.
            for k in 0..14_i64 {
                let x = rect.left() + rect.width() * hash(k * 3 + 1);
                let twinkle = ((time * 2.0 + hash(k) * 6.0).sin() * 0.5 + 0.5).powi(3);
                let c = Pos2::new(x, surface(x) + 5.0 + 6.0 * hash(k * 5));
                let r = 2.0 + 3.0 * twinkle;
                let stroke = Stroke::new(1.0, alpha((255, 255, 255), 0.25 + 0.6 * twinkle));
                painter.line_segment([c - Vec2::new(r, 0.0), c + Vec2::new(r, 0.0)], stroke);
                painter.line_segment([c - Vec2::new(0.0, r), c + Vec2::new(0.0, r)], stroke);
            }
        }
    }
    let frost = amount(cfg, "Ice patches");
    if frost > 0.0 {
        // Ice bands where the friction drops: a pale glaze that fades in and
        // out with the patch (`physics::ice`), with a bright edge and a few
        // glints. Dry stretches between them stay untouched.
        let mut mesh = Mesh::default();
        let mut shine: Vec<(Pos2, f32)> = Vec::new();
        for p in &line {
            let weight = crate::physics::ice(world_x(p.x));
            let top = alpha((156, 196, 220), (0.65 + 0.30 * frost) * weight);
            for (pos, color) in [
                (*p, top),
                (
                    *p + Vec2::new(0.0, 24.0),
                    alpha((90, 136, 172), 0.30 * weight),
                ),
            ] {
                mesh.vertices.push(Vertex {
                    pos,
                    uv: egui::epaint::WHITE_UV,
                    color,
                });
            }
            let k = mesh.vertices.len() as u32;
            if k >= 4 {
                mesh.indices
                    .extend_from_slice(&[k - 4, k - 3, k - 2, k - 3, k - 1, k - 2]);
            }
            shine.push((*p + Vec2::new(0.0, 2.5), weight));
        }
        painter.add(egui::Shape::mesh(mesh));
        for pair in shine.windows(2) {
            let weight = 0.5 * (pair[0].1 + pair[1].1);
            if weight > 0.05 {
                painter.line_segment(
                    [pair[0].0, pair[1].0],
                    Stroke::new(1.5, alpha((255, 255, 255), (0.25 + 0.5 * frost) * weight)),
                );
                // A dark blue edge along the surface keeps the patch readable
                // against the grey sky and the dark ground.
                painter.line_segment(
                    [
                        pair[0].0 - Vec2::new(0.0, 1.0),
                        pair[1].0 - Vec2::new(0.0, 1.0),
                    ],
                    Stroke::new(2.5, alpha((64, 112, 150), 0.85 * weight)),
                );
            }
        }
        let first = (world_x(rect.left()) / crate::physics::ICE_SPACING).floor() as i64;
        let last = (world_x(rect.right()) / crate::physics::ICE_SPACING).ceil() as i64;
        for k in first..=last {
            // One glint at the middle of each patch.
            let meters = (k as f32 + 0.5) * crate::physics::ICE_SPACING;
            let x = rect.left() + (meters - world_x(rect.left())) * pixels_per_meter;
            let twinkle = ((time * 2.0 + hash(k) * 6.0).sin() * 0.5 + 0.5).powi(3);
            let c = Pos2::new(x, surface(x) + 6.0);
            let r = 2.0 + 4.0 * twinkle;
            let stroke = Stroke::new(1.0, alpha((255, 255, 255), 0.3 + 0.6 * twinkle));
            painter.line_segment([c - Vec2::new(r, 0.0), c + Vec2::new(r, 0.0)], stroke);
            painter.line_segment([c - Vec2::new(0.0, r), c + Vec2::new(0.0, r)], stroke);
        }
    }
    let quake = amount(cfg, "Earthquake");
    if quake > 0.0 {
        // Ripples in the ground: wavy lines that pulse. The view never shakes.
        for k in 0..3 {
            let depth = 12.0 + 16.0 * k as f32;
            let points: Vec<Pos2> = line
                .iter()
                .map(|p| {
                    let wave =
                        (p.x * 0.05 - time * 9.0 + k as f32 * 1.7).sin() * (1.0 + 2.5 * quake);
                    Pos2::new(p.x, p.y + depth + wave)
                })
                .collect();
            painter.add(egui::Shape::line(
                points,
                Stroke::new(1.2, alpha((16, 16, 12), 0.25 + 0.30 * quake)),
            ));
        }
    }
    let brambles = amount(cfg, "Brambles");
    if brambles > 0.0 {
        // Thorny tufts along the ground, denser in a thicket.
        let spacing = 0.5 - 0.3 * brambles;
        let first = (world_x(rect.left()) / spacing).floor() as i64;
        let last = (world_x(rect.right()) / spacing).ceil() as i64;
        for k in first..=last {
            let meters = (k as f32 + hash(k * 11)) * spacing;
            let x = rect.left() + (meters - world_x(rect.left())) * pixels_per_meter;
            let base = Pos2::new(x, surface(x));
            let height = (5.0 + 9.0 * hash(k * 13)) * (0.6 + 0.6 * brambles);
            for s in 0..4_i64 {
                let lean = (hash(k * 17 + s) - 0.5) * 1.6;
                let tip = base + Vec2::new(lean * height, -height * (0.6 + 0.4 * hash(k * 19 + s)));
                painter.line_segment([base, tip], Stroke::new(1.3, alpha((58, 66, 30), 0.85)));
                painter.circle_filled(tip, 1.1, alpha((92, 40, 34), 0.8));
            }
        }
    }
    let mud = amount(cfg, "Mud");
    if mud > 0.0 {
        // Speckles in the mud layer.
        let first = (world_x(rect.left()) / 0.15).floor() as i64;
        let last = (world_x(rect.right()) / 0.15).ceil() as i64;
        for k in first..=last {
            let meters = (k as f32 + hash(k * 7)) * 0.15;
            let x = rect.left() + (meters - world_x(rect.left())) * pixels_per_meter;
            let y = surface(x) + 3.0 + hash(k * 9) * (4.0 + 8.0 * mud);
            painter.circle_filled(Pos2::new(x, y), 1.6, alpha((30, 22, 14), 0.60));
        }
        // Splashes: droplets thrown up by feet that move on the ground.
        for foot in feet.iter().filter(|f| f.speed > 30.0) {
            for d in 0..5_i64 {
                let phase = (time * 2.5 + hash(foot.id as i64 * 5 + d) + d as f32 * 0.2).fract();
                let dir = if d % 2 == 0 { -1.0 } else { 1.0 };
                let out = dir * (4.0 + 14.0 * hash(d + 40)) * phase * (0.6 + mud);
                let up = (phase * (1.0 - phase)) * 4.0 * (10.0 + 18.0 * mud * hash(d + 50));
                painter.circle_filled(
                    foot.at + Vec2::new(out, -up),
                    1.8,
                    alpha((74, 58, 38), 0.85 * (1.0 - phase)),
                );
            }
        }
    }
}

/// A vertical gradient across the width of `rect`, from `top` at `y0` to
/// `bottom` at `y1`. The span is clamped to the height of `rect`, and an empty
/// span paints nothing.
fn gradient(painter: &Painter, rect: Rect, y0: f32, y1: f32, top: Color32, bottom: Color32) {
    let (y0, y1) = (y0.max(rect.top()), y1.min(rect.bottom()));
    if y1 <= y0 {
        return;
    }
    let mut mesh = Mesh::default();
    for (pos, color) in [
        (Pos2::new(rect.left(), y0), top),
        (Pos2::new(rect.right(), y0), top),
        (Pos2::new(rect.right(), y1), bottom),
        (Pos2::new(rect.left(), y1), bottom),
    ] {
        mesh.vertices.push(Vertex {
            pos,
            uv: egui::epaint::WHITE_UV,
            color,
        });
    }
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    painter.add(egui::Shape::mesh(mesh));
}

/// Which sky hangs over a world. A storm sky when the weather turns: wind,
/// quakes, heavy gravity, wet ground or mud. The dusk sky, with its low sun,
/// in a heat wave or a drought. The overcast city sky otherwise. When a storm
/// and a dry spell both apply, the stronger wins and a tie goes to dusk.
pub fn sky_art(cfg: &Config) -> Art {
    let (_, slip) = grip(cfg);
    let storm = amount(cfg, "Wind")
        .max(amount(cfg, "Earthquake"))
        .max(amount(cfg, "Gravity"))
        .max(amount(cfg, "Mud"))
        .max(if slip > 0.3 && slip < 0.9 { slip } else { 0.0 });
    let dry = amount(cfg, "Heat wave").max(amount(cfg, "Drought"));
    if dry > 0.0 && dry >= storm {
        Art::SkyDusk
    } else if storm > 0.0 {
        Art::SkyStorm
    } else {
        Art::SkyCity
    }
}

/// One skyline layer: its image, its height as a share of the sky above
/// the horizon, how fast it slides with the camera, and where in the image
/// the view starts.
struct Layer {
    art: Art,
    height: f32,
    parallax: f32,
    start: f32,
}

/// The three skyline layers: far, mid, near.
const LAYERS: [Layer; 3] = [
    Layer {
        art: Art::SkylineFar,
        height: 0.98,
        parallax: 0.05,
        start: 0.62,
    },
    Layer {
        art: Art::SkylineMid,
        height: 0.5,
        parallax: 0.16,
        start: 0.08,
    },
    Layer {
        art: Art::SkylineNear,
        height: 0.66,
        parallax: 0.34,
        start: 0.4,
    },
];

/// A horizontal band of fog across `rect`: clear at screen y `top`, `color` at
/// `bottom`.
fn fog(painter: &Painter, rect: Rect, top: f32, bottom: f32, color: Color32) {
    gradient(painter, rect, top, bottom, Color32::TRANSPARENT, color);
}

/// The backdrop behind every effect, painted in layers from far to near: the
/// sky that `sky_art` picks, a glow and a few shafts of light where the sun
/// burns through, the far city in haze with the great tower climbing into the
/// clouds, a nearer row of old tenements, and in front of them the street's
/// poles, wires, lamps and bare trees. Each layer slides at its own parallax,
/// and fog settles between them. `horizon` is the screen y the skyline stands
/// on and `camera` is the camera's x in pixels. `water` is the screen y of the
/// water surface, or `None` when there is no water. Below the surface the sky
/// and the skyline show again upside down and dimmed, as the canal reflects
/// them.
#[allow(clippy::too_many_arguments)]
pub fn backdrop(
    painter: &Painter,
    rect: Rect,
    horizon: f32,
    camera: f32,
    time: f32,
    cfg: &Config,
    water: Option<f32>,
) {
    let ctx = painter.ctx();
    // The water's surface hides what stands behind it below the line.
    let above = water.map_or(rect, |y| {
        Rect::from_min_max(
            rect.left_top(),
            Pos2::new(rect.right(), y.clamp(rect.top(), rect.bottom())),
        )
    });
    // Paints one image above the water, and its reflection below the surface
    // when there is water. The reflection is the part of the image above the
    // surface, flipped about it and dimmed.
    let draw = |tex: egui::TextureId, dest: Rect, uv: Rect, tint: Color32| {
        painter
            .with_clip_rect(above.intersect(painter.clip_rect()))
            .image(tex, dest, uv, tint);
        let Some(surface) = water.filter(|&y| y > dest.top() && y < rect.bottom()) else {
            return;
        };
        let bottom = dest.bottom().min(surface);
        let t = |y: f32| uv.top() + (y - dest.top()) / dest.height() * uv.height();
        let mirrored = Rect::from_min_max(
            Pos2::new(dest.left(), 2.0 * surface - bottom),
            Pos2::new(dest.right(), 2.0 * surface - dest.top()),
        );
        painter
            .with_clip_rect(Rect::from_min_max(
                Pos2::new(rect.left(), surface),
                rect.right_bottom(),
            ))
            .image(
                tex,
                mirrored,
                Rect::from_min_max(
                    Pos2::new(uv.left(), t(bottom)),
                    Pos2::new(uv.right(), t(dest.top())),
                ),
                Color32::from_gray(120),
            );
    };
    let horizon = horizon.clamp(rect.top() + 40.0, rect.bottom() + 120.0);
    let sky_h = horizon - rect.top();
    painter.rect_filled(rect, 0, crate::theme::scene::SKY_HORIZON);
    // The sky: 360 degrees across 4096 texels, the horizon 17 texels above
    // its bottom row. A view shows about 110 degrees across, with square
    // texels. If the view is too tall for the band, the scale grows until the
    // band covers it, so less of the circle shows.
    let art = sky_art(cfg);
    let size = art.size(ctx);
    let horizon_row = size.y - 17.0;
    let scale = (rect.width() / (size.x * 0.3)).max(sky_h / (horizon_row - 4.0));
    let u0 = 0.3 + camera * 0.015 / (size.x * scale) + time * 0.0006;
    let v_top = (horizon_row - sky_h / scale) / size.y;
    let bottom = (horizon + 17.0 * scale).min(rect.bottom());
    let v_bottom = (horizon_row + (bottom - horizon) / scale) / size.y;
    draw(
        art.texture(ctx),
        Rect::from_min_max(rect.left_top(), Pos2::new(rect.right(), bottom)),
        Rect::from_min_max(
            Pos2::new(u0, v_top),
            Pos2::new(u0 + rect.width() / (size.x * scale), v_bottom),
        ),
        Color32::WHITE,
    );
    // Where the sun burns through the cloud: a wide soft glow with a smaller
    // one on top, and a few shafts of light falling from it. The sun is far
    // away, so it holds its place on screen.
    let sun = Pos2::new(rect.left() + rect.width() * 0.66, rect.top() + sky_h * 0.28);
    let warm = if art == Art::SkyDusk {
        (255, 196, 120)
    } else {
        (255, 244, 220)
    };
    let glow = Art::Glow.texture(ctx);
    let r = sky_h * 0.9;
    painter.image(
        glow,
        Rect::from_center_size(sun, Vec2::splat(r * 2.0)),
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        alpha(warm, 0.22),
    );
    painter.image(
        glow,
        Rect::from_center_size(sun, Vec2::splat(r * 0.6)),
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        alpha(warm, 0.18),
    );
    let mut rays = Mesh::default();
    for k in 0..6 {
        let spread = (k as f32 - 2.5) * 0.16 + (time * 0.05 + k as f32).sin() * 0.02;
        let dir = Vec2::new(spread, 1.0).normalized();
        let side = Vec2::new(dir.y, -dir.x);
        let len = sky_h * 1.4;
        let w = sky_h * (0.05 + 0.04 * hash(k * 7));
        let base = rays.vertices.len() as u32;
        let a = alpha(warm, 0.05 + 0.03 * hash(k * 3));
        for (pos, color) in [
            (sun, a),
            (sun + dir * len + side * w, Color32::TRANSPARENT),
            (sun + dir * len - side * w, Color32::TRANSPARENT),
        ] {
            rays.vertices.push(Vertex {
                pos,
                uv: egui::epaint::WHITE_UV,
                color,
            });
        }
        rays.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    painter.add(egui::Shape::mesh(rays));
    // The skyline, far to near, with fog settling between the layers.
    let haze = crate::theme::scene::SKY_HORIZON;
    for (i, layer) in LAYERS.iter().enumerate() {
        let size = layer.art.size(ctx);
        let h = sky_h * layer.height;
        let texel = h / size.y;
        let w = size.x * texel;
        let u = layer.start + camera * layer.parallax / w;
        let top = horizon + 2.0 - h;
        let shown = Rect::from_min_max(
            Pos2::new(rect.left(), top.max(rect.top())),
            Pos2::new(rect.right(), (horizon + 2.0).min(rect.bottom())),
        );
        if shown.height() > 0.0 {
            let v0 = (shown.top() - top) / h;
            let v1 = (shown.bottom() - top) / h;
            draw(
                layer.art.texture(ctx),
                shown,
                Rect::from_min_max(Pos2::new(u, v0), Pos2::new(u + rect.width() / w, v1)),
                Color32::WHITE,
            );
        }
        if i < 2 {
            let depth = sky_h * (0.34 - 0.12 * i as f32);
            fog(
                painter,
                rect,
                horizon - depth,
                horizon + 2.0,
                Color32::from_rgba_unmultiplied(haze.r(), haze.g(), haze.b(), 110 - 40 * i as u8),
            );
        }
    }
}

/// Weather over the scene, drawn after the creature: rain when the ground is
/// wet or muddy or there is water, blowing dust in a drought, a heat wave or
/// wind, cold mist over ice, and drifting ash in an earthquake. The rain, dust
/// and ash are many thin shapes on the replay clock. `camera` is the camera's
/// x in pixels, which shifts the rain and the dust.
pub fn weather(painter: &Painter, rect: Rect, cfg: &Config, time: f32, camera: f32) {
    let (_, slip) = grip(cfg);
    let wet = if slip > 0.3 && slip < 0.9 { slip } else { 0.0 };
    let rain = wet
        .max(amount(cfg, "Mud") * 0.8)
        .max(amount(cfg, "Water") * 0.4);
    let wind = amount(cfg, "Wind");
    if rain > 0.0 {
        let count = 40 + (140.0 * rain) as i64;
        let slant = -0.18 - 0.5 * wind;
        for i in 0..count {
            let speed = 520.0 + 240.0 * hash(i * 3);
            let x = rect.left()
                + (hash(i) * (rect.width() + 80.0) + camera * 0.6).rem_euclid(rect.width() + 80.0)
                - 40.0;
            let y = rect.top()
                + (hash(i * 5 + 1) * rect.height() + time * speed).rem_euclid(rect.height());
            let len = 10.0 + 12.0 * hash(i * 7);
            let a = Pos2::new(x, y);
            painter.line_segment(
                [a, a + Vec2::new(slant * len, len)],
                Stroke::new(1.0, alpha((200, 210, 214), 0.16 + 0.18 * rain)),
            );
        }
    }
    let dust = amount(cfg, "Drought")
        .max(amount(cfg, "Heat wave") * 0.6)
        .max(wind * 0.7);
    if dust > 0.0 {
        let tint = if amount(cfg, "Drought").max(amount(cfg, "Heat wave")) > 0.0 {
            (200, 170, 120)
        } else {
            (190, 190, 180)
        };
        for i in 0..(30 + (70.0 * dust) as i64) {
            let speed = 40.0 + 180.0 * (dust + wind) * hash(i * 11);
            let x = rect.left()
                + (hash(i * 13) * rect.width() - time * speed + camera * 0.5)
                    .rem_euclid(rect.width());
            let y = rect.top()
                + rect.height() * (0.35 + 0.6 * hash(i * 17))
                + (time * 2.0 + i as f32).sin() * 4.0;
            painter.circle_filled(
                Pos2::new(x, y),
                0.8 + 1.4 * hash(i * 19),
                alpha(tint, 0.25 + 0.35 * dust),
            );
        }
        gradient(
            painter,
            rect,
            rect.top() + rect.height() * 0.45,
            rect.bottom(),
            Color32::TRANSPARENT,
            alpha(tint, 0.10 + 0.14 * dust),
        );
    }
    let frost = amount(cfg, "Ice patches").max(if slip >= 0.9 { 1.0 } else { 0.0 });
    if frost > 0.0 {
        gradient(
            painter,
            rect,
            rect.top() + rect.height() * 0.5,
            rect.bottom(),
            Color32::TRANSPARENT,
            alpha((190, 214, 228), 0.10 + 0.10 * frost),
        );
    }
    let quake = amount(cfg, "Earthquake");
    if quake > 0.0 {
        for i in 0..(20 + (60.0 * quake) as i64) {
            let fall = 30.0 + 50.0 * hash(i * 23);
            let x =
                rect.left() + hash(i * 29) * rect.width() + (time * 1.5 + i as f32).sin() * 10.0;
            let y =
                rect.top() + (hash(i * 31) * rect.height() + time * fall).rem_euclid(rect.height());
            painter.circle_filled(
                Pos2::new(x, y),
                1.0 + hash(i) * 1.2,
                alpha((150, 146, 136), 0.35 * quake + 0.15),
            );
        }
    }
}

/// The ground's body and crust, textured like the street of the world:
/// cobbles over dark concrete in the city, sand over dirt in a drought or a
/// heat wave, sludge in mud, and plain dark concrete when there is water. Mud
/// wins over dry ground, and dry ground over water. `line` is the surface
/// polyline in screen points, `meters` the world x in meters under each of its
/// points, and `ppm` the zoom in pixels per meter. The body fills down to the
/// bottom of `rect`. A line of fewer than two points paints nothing.
pub fn ground_body(
    painter: &Painter,
    rect: Rect,
    cfg: &Config,
    line: &[Pos2],
    meters: &[f32],
    ppm: f32,
) {
    if line.len() < 2 {
        return;
    }
    let dry = amount(cfg, "Drought").max(amount(cfg, "Heat wave") * 0.8);
    let (body, crust) = if amount(cfg, "Mud") > 0.0 {
        (Art::Mud, Art::Mud)
    } else if dry > 0.0 {
        (Art::Dirt, Art::Sand)
    } else if amount(cfg, "Water") > 0.0 {
        (Art::ConcreteDark, Art::ConcreteDark)
    } else {
        (Art::ConcreteDark, Art::Cobble)
    };
    let ctx = painter.ctx();
    // The body: one repeat per 1.6 m, the texture fixed to the ground as it
    // scrolls, dark toward the frame's bottom.
    let tile = 1.6 * ppm;
    let mut mesh = Mesh::with_texture(body.texture(ctx));
    for (i, (p, m)) in line.iter().zip(meters).enumerate() {
        let u = m * ppm / tile;
        let depth = (rect.bottom() - p.y).max(0.0);
        for (pos, v, color) in [
            (*p, 0.0, Color32::from_gray(150)),
            (
                Pos2::new(p.x, rect.bottom()),
                depth / tile,
                Color32::from_gray(58),
            ),
        ] {
            mesh.vertices.push(Vertex {
                pos,
                uv: Pos2::new(u, v + 0.37),
                color,
            });
        }
        if i > 0 {
            let k = (i * 2) as u32;
            mesh.indices
                .extend_from_slice(&[k - 2, k - 1, k, k - 1, k + 1, k]);
        }
    }
    painter.add(egui::Shape::Mesh(mesh.into()));
    // The crust: the street's surface seen at a grazing angle, a band 0.16 m
    // thick, kept between 8 and 26 pixels, squeezed from its texture.
    let thick = (0.16 * ppm).clamp(8.0, 26.0);
    let mut mesh = Mesh::with_texture(crust.texture(ctx));
    for (i, (p, m)) in line.iter().zip(meters).enumerate() {
        let u = m / 0.9;
        for (pos, v, color) in [
            (*p, 0.0, Color32::from_gray(235)),
            (*p + Vec2::new(0.0, thick), 0.5, Color32::from_gray(150)),
        ] {
            mesh.vertices.push(Vertex {
                pos,
                uv: Pos2::new(u, v),
                color,
            });
        }
        if i > 0 {
            let k = (i * 2) as u32;
            mesh.indices
                .extend_from_slice(&[k - 2, k - 1, k, k - 1, k + 1, k]);
        }
    }
    painter.add(egui::Shape::Mesh(mesh.into()));
    // The curb under the crust: a pale concrete edge with a shadow below.
    let curb: Vec<Pos2> = line
        .iter()
        .map(|p| *p + Vec2::new(0.0, thick + 1.5))
        .collect();
    painter.add(egui::Shape::line(
        curb.clone(),
        Stroke::new(3.0, Color32::from_rgb(150, 148, 138)),
    ));
    band(
        painter,
        &curb
            .iter()
            .map(|p| *p + Vec2::new(0.0, 1.5))
            .collect::<Vec<_>>(),
        14.0,
        Color32::from_black_alpha(150),
        Color32::TRANSPARENT,
    );
    // The top of the crust catches the overcast light.
    band(
        painter,
        line,
        thick * 0.5,
        Color32::from_white_alpha(28),
        Color32::TRANSPARENT,
    );
}

/// A convex polygon filled with the texture `art`, tinted by `tint`, `tile`
/// points per repeat, with the texture fixed to the polygon's first point.
/// Fewer than three points paint nothing.
fn textured_fan(painter: &Painter, pts: &[Pos2], art: Art, tile: f32, tint: Color32) {
    if pts.len() < 3 {
        return;
    }
    let mut mesh = Mesh::with_texture(art.texture(painter.ctx()));
    let origin = pts[0];
    for p in pts {
        mesh.vertices.push(Vertex {
            pos: *p,
            uv: ((*p - origin) / tile).to_pos2(),
            color: tint,
        });
    }
    for i in 1..pts.len() as u32 - 1 {
        mesh.indices.extend_from_slice(&[0, i, i + 1]);
    }
    painter.add(egui::Shape::Mesh(mesh.into()));
}

/// Hazard stripes, amber and black, filling `rect`. A `rect` under 2 points
/// wide or tall paints nothing.
fn hazard(painter: &Painter, rect: Rect) {
    if rect.width() < 2.0 || rect.height() < 2.0 {
        return;
    }
    painter.rect_filled(rect, 0, Color32::from_rgb(24, 22, 18));
    let clip = painter.with_clip_rect(rect.intersect(painter.clip_rect()));
    let h = rect.height();
    let mut x = rect.left() - h;
    while x < rect.right() {
        clip.add(egui::Shape::convex_polygon(
            vec![
                Pos2::new(x, rect.bottom()),
                Pos2::new(x + h, rect.top()),
                Pos2::new(x + h + 6.0, rect.top()),
                Pos2::new(x + 6.0, rect.bottom()),
            ],
            Color32::from_rgb(196, 146, 44),
            Stroke::NONE,
        ));
        x += 12.0;
    }
}

/// The ground's built parts: dark pits with hazard marks on their lips, and
/// concrete blocks with a striped top for the hurdles. At the highest hurdle
/// level a block is a dark plate wall with a cold light along it instead.
/// `at(x, y)` maps a point in meters to the screen. `height(x, with_hurdles)`
/// is the ground height at `x` meters, with or without the hurdles. `view` is
/// the range of meters on screen.
pub fn structures(
    painter: &Painter,
    rect: Rect,
    cfg: &Config,
    at: &dyn Fn(f32, f32) -> Pos2,
    height: &dyn Fn(f32, bool) -> f32,
    view: (f32, f32),
) {
    let (left, right) = view;
    let ppm = (at(1.0, 0.0).x - at(0.0, 0.0).x).max(1e-3);
    if cfg.gaps > 0.0 {
        let spacing = crate::physics::gap_spacing(cfg.gaps);
        let half = 0.5 * cfg.gaps;
        let first = (left / spacing).floor() as i64;
        let last = (right / spacing).ceil() as i64;
        for k in first..=last {
            let center = (k as f32 + 0.5) * spacing;
            let (a, b) = (center - half, center + half);
            if b < left || a > right {
                continue;
            }
            let lip_a = at(a, height(a - 0.01, true));
            let lip_b = at(b, height(b + 0.01, true));
            let floor = rect.bottom() + 4.0;
            let mut mesh = Mesh::default();
            let top = Color32::from_rgb(20, 20, 18);
            let bottom = Color32::from_rgb(4, 4, 4);
            for (pos, color) in [
                (lip_a, top),
                (lip_b, top),
                (Pos2::new(lip_b.x, floor), bottom),
                (Pos2::new(lip_a.x, floor), bottom),
            ] {
                mesh.vertices.push(Vertex {
                    pos,
                    uv: egui::epaint::WHITE_UV,
                    color,
                });
            }
            mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
            painter.add(egui::Shape::mesh(mesh));
            // The pit walls: rusted steel sheet piling, and a striped
            // marker on each lip.
            let wall = (0.12 * ppm).clamp(4.0, 14.0);
            for (x0, x1) in [(lip_a.x, lip_a.x + wall), (lip_b.x - wall, lip_b.x)] {
                // The left wall hangs from the left lip's height and the right
                // wall from the right lip's. A pit under one and a half walls
                // wide on screen uses the right lip for both.
                let wall_top = if x0 < lip_b.x - wall * 1.5 {
                    lip_a.y
                } else {
                    lip_b.y
                };
                crate::theme::tiled(
                    painter,
                    Rect::from_min_max(Pos2::new(x0, wall_top), Pos2::new(x1, floor)),
                    Art::Rust,
                    Vec2::splat((0.8 * ppm).max(8.0)),
                    Vec2::new(k as f32 * 0.37, 0.0),
                    Color32::from_gray(120),
                );
            }
            let mark = (0.35 * ppm).clamp(10.0, 42.0);
            let tall = (0.05 * ppm).clamp(3.0, 7.0);
            hazard(
                painter,
                Rect::from_min_max(
                    Pos2::new(lip_a.x - mark, lip_a.y + 1.0),
                    Pos2::new(lip_a.x, lip_a.y + 1.0 + tall),
                ),
            );
            hazard(
                painter,
                Rect::from_min_max(
                    Pos2::new(lip_b.x, lip_b.y + 1.0),
                    Pos2::new(lip_b.x + mark, lip_b.y + 1.0 + tall),
                ),
            );
        }
    }
    if cfg.hurdles > 0.0 {
        use crate::physics::{HURDLE_RUN, HURDLE_SPACING, HURDLE_TOP};
        let half = 0.5 * HURDLE_TOP;
        let first = (left / HURDLE_SPACING).floor() as i64;
        let last = (right / HURDLE_SPACING).ceil() as i64;
        for k in first..=last {
            let center = (k as f32 + 0.5) * HURDLE_SPACING;
            let (s, e) = (center - half - HURDLE_RUN, center + half + HURDLE_RUN);
            if e < left || s > right {
                continue;
            }
            let (tl, tr) = (center - half, center + half);
            // A pit under the hurdle breaks its block apart, so only the
            // ground line shows the hurdle then.
            if [s, tl, center, tr, e]
                .iter()
                .any(|&x| crate::physics::gaps(x, cfg.gaps).0 != 0.0)
            {
                continue;
            }
            let body = [
                at(s, height(s, true)),
                at(tl, height(tl, true)),
                at(tr, height(tr, true)),
                at(e, height(e, true)),
                at(e, height(e, false)),
                at(tr, height(tr, false)),
                at(tl, height(tl, false)),
                at(s, height(s, false)),
            ];
            // Concrete blocks. The highest level is a Combine wall of dark
            // plate with a cold light along it.
            let combine = amount(cfg, "Hurdles") >= 1.0;
            let art = if combine {
                Art::CombinePlate
            } else {
                Art::ConcreteLight
            };
            let tile = (0.9 * ppm).max(8.0);
            textured_fan(
                painter,
                &body,
                art,
                tile,
                Color32::from_gray(if combine { 150 } else { 190 }),
            );
            painter.add(egui::Shape::closed_line(
                body.to_vec(),
                Stroke::new(1.2, Color32::from_rgb(24, 24, 22)),
            ));
            // Shade toward the foot, a lit top edge.
            let foot = body[6].y.max(body[5].y);
            gradient(
                painter,
                Rect::from_min_max(
                    Pos2::new(body[0].x, rect.top()),
                    Pos2::new(body[3].x, rect.bottom()),
                ),
                (body[1].y + foot) * 0.5,
                foot,
                Color32::TRANSPARENT,
                Color32::from_black_alpha(90),
            );
            painter.line_segment(
                [body[1] + Vec2::new(0.0, 1.0), body[2] + Vec2::new(0.0, 1.0)],
                Stroke::new(1.5, alpha((230, 230, 220), 0.35)),
            );
            let (a, b) = (body[1], body[2]);
            if combine {
                let y = a.y + (foot - a.y) * 0.4;
                painter.line_segment(
                    [Pos2::new(a.x + 3.0, y), Pos2::new(b.x - 3.0, y)],
                    Stroke::new(2.0, Color32::from_rgb(130, 210, 255)),
                );
                painter.line_segment(
                    [Pos2::new(a.x + 3.0, y), Pos2::new(b.x - 3.0, y)],
                    Stroke::new(7.0, alpha((120, 200, 255), 0.18)),
                );
                continue;
            }
            let tall = (0.06 * ppm).clamp(3.0, 8.0);
            if (body[6].y - a.y) > tall * 2.0 {
                hazard(
                    painter,
                    Rect::from_min_max(
                        Pos2::new(a.x, a.y.min(b.y) + 1.0),
                        Pos2::new(b.x, a.y.max(b.y) + 1.0 + tall),
                    ),
                );
            }
        }
    }
}
