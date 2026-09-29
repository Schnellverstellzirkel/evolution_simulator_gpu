//! Painted look of the environment effects in the replay view. Every effect
//! is a faint overlay or a few thin shapes, so the creature, the meter marks
//! and the text stay readable. Animation follows the replay clock passed in
//! as `time`, so a paused replay holds still and no shape needs per-frame
//! state. The scene colors are fixed, so the same overlays serve both themes.
use crate::{config::Config, environment::EFFECTS};
use eframe::egui::{
    self, Color32, Painter, Pos2, Rect, Stroke, Vec2,
    epaint::{Mesh, Vertex},
};

/// How far an effect is from calm, 0 (calm) to 1 (harshest level). Grip has
/// two sides, see `grip`.
fn amount(cfg: &Config, name: &str) -> f32 {
    EFFECTS.iter().find(|e| e.name == name).map_or(0.0, |e| {
        let top = e.levels.len().saturating_sub(1).max(1);
        let level = e.level(cfg);
        level.saturating_sub(e.calm) as f32 / (top - e.calm).max(1) as f32
    })
}

/// (sandpaper, slipperiness): the grip effect's two sides, each 0 to 1.
fn grip(cfg: &Config) -> (f32, f32) {
    let Some(e) = EFFECTS.iter().find(|e| e.name == "Grip") else {
        return (0.0, 0.0);
    };
    let level = e.level(cfg);
    let rough = if level < e.calm { 1.0 } else { 0.0 };
    (rough, amount(cfg, "Grip"))
}

fn alpha(color: (u8, u8, u8), a: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(color.0, color.1, color.2, (a.clamp(0.0, 1.0) * 255.0) as u8)
}

/// A cheap deterministic value in [0, 1) for a small integer.
fn hash(n: i64) -> f32 {
    let mut x = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x & 0xFFFF) as f32 / 65536.0
}

/// A strip along the ground line that fades from `top` to `bottom` over
/// `depth` pixels (negative depth reaches upward).
fn band(painter: &Painter, line: &[Pos2], depth: f32, top: Color32, bottom: Color32) {
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

/// Overlays on the sky, drawn before the ground: warmth, haze, wind streaks
/// and a season tint.
pub fn sky(painter: &Painter, rect: Rect, cfg: &Config, time: f32) {
    let heat = amount(cfg, "Heat wave");
    if heat > 0.0 {
        // A warm glow rising from the horizon, so the sky keeps its blue
        // overhead instead of turning grey.
        let mut mesh = Mesh::default();
        let glow = alpha((255, 170, 60), 0.18 + 0.30 * heat);
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
        // Shimmer: thin wavy lines rising off the horizon.
        for k in 0..3 {
            let y = rect.bottom() - rect.height() * (0.30 + 0.06 * k as f32);
            let points: Vec<Pos2> = (0..=40)
                .map(|i| {
                    let x = rect.left() + rect.width() * i as f32 / 40.0;
                    let wave = (x * 0.03 + time * 3.0 + k as f32 * 2.0).sin() * (2.0 + 2.0 * heat);
                    Pos2::new(x, y + wave)
                })
                .collect();
            painter.add(egui::Shape::line(
                points,
                Stroke::new(1.5, alpha((255, 255, 255), 0.10 + 0.14 * heat)),
            ));
        }
    }
    let air = amount(cfg, "Air");
    if air > 0.0 {
        painter.rect_filled(rect, 12, alpha((235, 240, 245), 0.08 + 0.22 * air));
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
            let len = 24.0 + 50.0 * hash(i * 5 + 2) * (0.5 + wind);
            painter.line_segment(
                [Pos2::new(x, y), Pos2::new(x + len, y)],
                Stroke::new(1.5, alpha((255, 255, 255), 0.30 + 0.30 * wind)),
            );
        }
    }
    if cfg.seasons > 0 {
        // The world walks through the seasons' rotation; tint by quarter lap.
        let lap = crate::environment::season_rotation().len().max(4);
        let quarter = (usize::from(cfg.season_step) % lap) * 4 / lap;
        let tint = [
            (120, 200, 90),
            (255, 220, 90),
            (235, 130, 50),
            (150, 190, 255),
        ][quarter];
        painter.rect_filled(rect, 12, alpha(tint, 0.10));
    }
}

/// The water: a translucent blue body below the waterline at screen height
/// `line_y`, a wavy surface, faint streaks and a few rising bubbles. The
/// creature and the text are drawn after it, and the tint stays light enough
/// to read them. `world_x` maps a screen x to meters, so the waves and bubbles
/// scroll with the ground.
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
    let body = alpha((40, 120, 210), 0.20 + 0.10 * level);
    let deep = alpha((20, 70, 160), 0.30 + 0.12 * level);
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
            Stroke::new(2.0, alpha((200, 235, 255), 0.85)),
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
            Stroke::new(1.0, alpha((200, 230, 255), 0.13)),
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
            Stroke::new(1.0, alpha((220, 240, 255), 0.45 * (1.0 - rise))),
        );
    }
}

/// One foot moving fast on the ground, for mud splashes.
pub struct Foot {
    /// Where the foot meets the ground surface, in screen pixels.
    pub at: Pos2,
    /// Ground speed in pixels per second on screen.
    pub speed: f32,
    pub id: usize,
}

/// Effects on and in the ground, drawn after it. `surface` maps a screen x to
/// the ground line's screen y. `world_x` maps a screen x to meters, so cracks
/// and grains scroll with the ground. `pixels_per_meter` is the zoom.
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
    let g = amount(cfg, "Gravity");
    if g > 0.0 {
        band(
            painter,
            &line,
            50.0,
            alpha((10, 30, 10), 0.12 + 0.28 * g),
            alpha((10, 30, 10), 0.0),
        );
    }
    let drought = amount(cfg, "Drought");
    let heat = amount(cfg, "Heat wave");
    let dry = drought.max(0.6 * heat);
    if dry > 0.0 {
        band(
            painter,
            &line,
            70.0,
            alpha((215, 185, 90), 0.20 + 0.40 * dry),
            alpha((215, 185, 90), 0.05 * dry),
        );
    }
    if drought > 0.0 {
        // Cracks every 0.7 m of ground: a short zigzag down from the surface.
        let first = (world_x(rect.left()) / 0.7).floor() as i64;
        let last = (world_x(rect.right()) / 0.7).ceil() as i64;
        let stroke = Stroke::new(1.2, alpha((70, 55, 25), 0.35 + 0.45 * drought));
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
            painter.circle_filled(Pos2::new(x, y), 1.1, alpha((60, 80, 40), 0.55));
        }
    }
    if slip > 0.3 {
        // Wet ground darkens with a bright film. Ice glazes over it.
        let s = (slip - 0.3) / 0.7;
        let ice = slip > 0.9;
        let color = if ice {
            (215, 240, 255)
        } else {
            (140, 190, 235)
        };
        if !ice {
            // Wet: the ground darkens, a bright film runs along it, and puddles
            // with slow ripples lie in the dips.
            band(
                painter,
                &line,
                28.0,
                alpha((20, 50, 110), 0.40 + 0.25 * s),
                alpha((25, 55, 100), 0.05),
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
                    alpha((90, 170, 245), 0.75 + 0.2 * s),
                    Stroke::new(1.5, alpha((240, 250, 255), 0.95)),
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
            let top = alpha((130, 205, 250), (0.65 + 0.30 * frost) * weight);
            for (pos, color) in [
                (*p, top),
                (
                    *p + Vec2::new(0.0, 24.0),
                    alpha((80, 160, 235), 0.30 * weight),
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
                // against the pale sky and the green ground.
                painter.line_segment(
                    [
                        pair[0].0 - Vec2::new(0.0, 1.0),
                        pair[1].0 - Vec2::new(0.0, 1.0),
                    ],
                    Stroke::new(2.5, alpha((40, 100, 190), 0.85 * weight)),
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
                Stroke::new(1.2, alpha((40, 70, 30), 0.20 + 0.25 * quake)),
            ));
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
            painter.circle_filled(Pos2::new(x, y), 1.6, alpha((55, 38, 24), 0.55));
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
                    alpha((92, 64, 40), 0.85 * (1.0 - phase)),
                );
            }
        }
    }
}
