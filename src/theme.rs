//! The look of the game: the palette of both themes, the egui style, and the
//! painted pieces the screens share. The key is grim and industrial: worn
//! concrete greys, olive and rust, amber for what matters now, cold blue for
//! what is secondary, and HUD panels with faintly glowing numbers over the
//! scene. Everything here is a few shapes or one textured quad, so it costs
//! next to nothing per frame.
use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, Response, Sense, Stroke, TextureId, Vec2,
    epaint::{Mesh, Vertex},
    text::{LayoutJob, TextFormat},
};

/// Colors of the scene: the replay, the race lanes and the thumbnails. They
/// are the same in both themes, so the world reads the same in each.
pub mod scene {
    use eframe::egui::Color32;
    /// Overcast sky, from overhead to the haze at the horizon.
    pub const SKY_TOP: Color32 = Color32::from_rgb(58, 66, 72);
    pub const SKY_HORIZON: Color32 = Color32::from_rgb(142, 146, 140);
    /// Far and near skyline silhouettes.
    pub const SKYLINE_FAR: Color32 = Color32::from_rgb(106, 112, 114);
    pub const SKYLINE_NEAR: Color32 = Color32::from_rgb(74, 79, 80);
    /// Packed dirt and broken concrete, from the surface down.
    pub const GROUND_TOP: Color32 = Color32::from_rgb(86, 82, 68);
    pub const GROUND_DEEP: Color32 = Color32::from_rgb(38, 37, 32);
    /// The worn lip along the ground surface.
    pub const GROUND_EDGE: Color32 = Color32::from_rgb(150, 142, 116);
    /// Meter labels and ticks on the ground.
    pub const GROUND_INK: Color32 = Color32::from_rgb(182, 174, 146);
    /// Faint meter lines across the sky.
    pub const GRID: Color32 = Color32::from_rgba_premultiplied(18, 18, 18, 18);
    /// Sludge of the mud layer and its lower edge.
    pub const MUD: Color32 = Color32::from_rgb(60, 44, 26);
    pub const MUD_EDGE: Color32 = Color32::from_rgb(34, 25, 14);
    /// The wet shine on top of the mud.
    pub const MUD_SHEEN: Color32 = Color32::from_rgba_premultiplied(66, 52, 34, 120);
    /// Creature parts: steel bones, flesh muscles, dark outlines.
    pub const OUTLINE: Color32 = Color32::from_rgb(12, 13, 13);
    pub const BONE: Color32 = Color32::from_rgb(184, 186, 176);
    pub const ORGAN: Color32 = Color32::from_rgb(168, 72, 64);
    pub const MUSCLE_REST: Color32 = Color32::from_rgb(206, 132, 112);
    pub const MUSCLE_ACTIVE: Color32 = Color32::from_rgb(158, 26, 20);
    pub const MUSCLE_TIRED: Color32 = Color32::from_rgb(120, 118, 110);
    /// Node shells: slick steel blue for low friction, brass for high.
    pub const NODE_SLICK: Color32 = Color32::from_rgb(150, 176, 190);
    pub const NODE_GRIPPY: Color32 = Color32::from_rgb(196, 170, 112);
    /// The head's eye glows faintly amber.
    pub const EYE: Color32 = Color32::from_rgb(255, 214, 140);
    /// Ring around a node on the ground.
    pub const TOUCHDOWN: Color32 = Color32::from_rgb(255, 200, 96);
    /// A fall, a broken joint.
    pub const FALLEN: Color32 = Color32::from_rgb(226, 74, 52);
    /// Force arrows: muscle pulls and ground pushes.
    pub const FORCE_MUSCLE: Color32 = Color32::from_rgb(244, 140, 36);
    pub const FORCE_GROUND: Color32 = Color32::from_rgb(90, 170, 230);
    /// The HUD over the scene: amber text on smoked glass.
    pub const HUD: Color32 = Color32::from_rgb(255, 196, 84);
    pub const HUD_DIM: Color32 = Color32::from_rgb(196, 160, 96);
    /// Plain text on a HUD panel.
    pub const HUD_INK: Color32 = Color32::from_rgb(222, 216, 200);
    pub const HUD_BACK: Color32 = Color32::from_rgba_premultiplied(8, 8, 7, 150);
    /// The centre of mass and its trail.
    pub const TRAIL: Color32 = Color32::from_rgb(255, 190, 80);
}

/// UI surface colors of the active theme.
#[derive(Clone, Copy)]
pub struct Theme {
    pub dark: bool,
    pub panel: Color32,
    pub canvas: Color32,
    pub card: Color32,
    pub card_hover: Color32,
    pub card_border: Color32,
    /// A faint line along the top edge of a plate.
    pub bevel: Color32,
    pub ink: Color32,
    pub muted: Color32,
    /// Amber: the numbers and choices that matter now.
    pub accent: Color32,
    /// Cold blue: secondary lines and what waits (the median curve, a world
    /// change that starts with the next generation).
    pub cold: Color32,
    /// Rust: warnings (a failed trial, a catastrophe, a world-change mark).
    pub warn: Color32,
    /// A fall or an error.
    pub danger: Color32,
    /// Record markers on the charts.
    pub record: Color32,
    /// The Evolve button: fill and text.
    pub go_fill: Color32,
    pub go_text: Color32,
    /// The Pause evolution button: fill and text.
    pub stop_fill: Color32,
    pub stop_text: Color32,
    /// How strong the worn texture on plates is, 0 to 1.
    pub wear: f32,
}

impl Theme {
    pub fn of(dark: bool) -> Self {
        if dark {
            Self {
                dark,
                panel: Color32::from_rgb(27, 28, 26),
                canvas: Color32::from_rgb(19, 20, 19),
                card: Color32::from_rgb(36, 37, 34),
                card_hover: Color32::from_rgb(48, 48, 43),
                card_border: Color32::from_rgb(61, 61, 55),
                bevel: Color32::from_white_alpha(14),
                ink: Color32::from_rgb(222, 216, 200),
                muted: Color32::from_rgb(150, 146, 132),
                accent: Color32::from_rgb(238, 168, 58),
                cold: Color32::from_rgb(112, 170, 206),
                warn: Color32::from_rgb(212, 120, 70),
                danger: Color32::from_rgb(230, 84, 62),
                record: Color32::from_rgb(255, 112, 44),
                go_fill: Color32::from_rgb(204, 140, 42),
                go_text: Color32::from_rgb(22, 19, 14),
                stop_fill: Color32::from_rgb(92, 52, 30),
                stop_text: Color32::from_rgb(240, 216, 182),
                wear: 1.0,
            }
        } else {
            // Concrete in daylight: the same hues, lighter surfaces, darker
            // amber and blue so text keeps its contrast.
            Self {
                dark,
                panel: Color32::from_rgb(210, 208, 200),
                canvas: Color32::from_rgb(192, 190, 181),
                card: Color32::from_rgb(222, 220, 212),
                card_hover: Color32::from_rgb(234, 229, 216),
                card_border: Color32::from_rgb(160, 156, 144),
                bevel: Color32::from_white_alpha(70),
                ink: Color32::from_rgb(30, 30, 27),
                muted: Color32::from_rgb(86, 84, 76),
                accent: Color32::from_rgb(156, 86, 6),
                cold: Color32::from_rgb(30, 86, 130),
                warn: Color32::from_rgb(150, 70, 28),
                danger: Color32::from_rgb(170, 40, 26),
                record: Color32::from_rgb(200, 66, 12),
                go_fill: Color32::from_rgb(212, 150, 56),
                go_text: Color32::from_rgb(24, 20, 14),
                stop_fill: Color32::from_rgb(122, 66, 36),
                stop_text: Color32::from_rgb(246, 234, 216),
                wear: 0.6,
            }
        }
    }
}

/// The spacing scale the style uses; ui.rs shares it.
pub const GAP_M: f32 = 8.0;
pub const GAP_L: f32 = 16.0;
/// Height of every button, menu and selectable in a row, and the starting
/// height of a row, so a row's items share one center line.
pub const CONTROL_HEIGHT: f32 = 34.0;

fn widget(
    fill: Color32,
    stroke: Color32,
    text: Color32,
    text_width: f32,
) -> egui::style::WidgetVisuals {
    egui::style::WidgetVisuals {
        bg_fill: fill,
        weak_bg_fill: fill,
        bg_stroke: Stroke::new(1.0, stroke),
        corner_radius: egui::CornerRadius::same(2),
        fg_stroke: Stroke::new(text_width, text),
        expansion: 0.0,
    }
}

/// Applies the custom style of the chosen theme.
pub fn apply_style(ctx: &egui::Context, dark: bool) {
    let theme = Theme::of(dark);
    ctx.set_theme(if dark {
        egui::Theme::Dark
    } else {
        egui::Theme::Light
    });
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = Vec2::new(GAP_M, GAP_M);
    style.spacing.button_padding = Vec2::new(12.0, 7.0);
    style.spacing.interact_size = Vec2::new(40.0, CONTROL_HEIGHT);
    style.spacing.slider_width = 120.0;
    style.spacing.window_margin = egui::Margin::same(GAP_L as i8);
    style.spacing.menu_margin = egui::Margin::same(GAP_M as i8);
    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    visuals.override_text_color = Some(theme.ink);
    visuals.weak_text_color = Some(theme.muted);
    visuals.panel_fill = theme.panel;
    visuals.window_fill = theme.panel;
    visuals.faint_bg_color = if dark {
        Color32::from_rgb(33, 34, 31)
    } else {
        Color32::from_rgb(202, 200, 191)
    };
    visuals.extreme_bg_color = if dark {
        Color32::from_rgb(16, 17, 16)
    } else {
        Color32::from_rgb(230, 228, 221)
    };
    visuals.code_bg_color = theme.canvas;
    visuals.hyperlink_color = theme.accent;
    visuals.warn_fg_color = theme.warn;
    visuals.error_fg_color = theme.danger;
    visuals.selection.bg_fill = if dark {
        Color32::from_rgb(98, 68, 22)
    } else {
        Color32::from_rgb(226, 182, 104)
    };
    visuals.selection.stroke = Stroke::new(
        1.0,
        if dark {
            Color32::from_rgb(255, 206, 120)
        } else {
            Color32::from_rgb(60, 34, 4)
        },
    );
    visuals.slider_trailing_fill = true;
    let (line, button, hover, press) = if dark {
        (
            Color32::from_rgb(50, 50, 45),
            Color32::from_rgb(44, 45, 41),
            Color32::from_rgb(58, 57, 50),
            Color32::from_rgb(74, 62, 38),
        )
    } else {
        (
            Color32::from_rgb(172, 168, 156),
            Color32::from_rgb(222, 220, 212),
            Color32::from_rgb(234, 228, 212),
            Color32::from_rgb(226, 196, 140),
        )
    };
    let bright = if dark {
        Color32::from_rgb(244, 238, 222)
    } else {
        Color32::from_rgb(10, 10, 9)
    };
    visuals.widgets.noninteractive = widget(theme.panel, line, theme.ink, 1.0);
    visuals.widgets.inactive = widget(button, theme.card_border, theme.ink, 1.0);
    visuals.widgets.hovered = widget(hover, theme.accent.gamma_multiply(0.7), bright, 1.5);
    visuals.widgets.active = widget(press, theme.accent, bright, 2.0);
    visuals.widgets.open = widget(hover, theme.card_border, bright, 1.0);
    visuals.window_corner_radius = egui::CornerRadius::same(3);
    visuals.menu_corner_radius = egui::CornerRadius::same(3);
    visuals.window_stroke = Stroke::new(1.0, theme.card_border);
    visuals.window_shadow = egui::Shadow {
        offset: [0, 8],
        blur: 22,
        spread: 0,
        color: Color32::from_black_alpha(if dark { 150 } else { 60 }),
    };
    visuals.popup_shadow = egui::Shadow {
        offset: [0, 4],
        blur: 10,
        spread: 0,
        color: Color32::from_black_alpha(if dark { 120 } else { 40 }),
    };
    style.visuals = visuals;
    // Type scale: small print 13 px, body and buttons 16 px, headings 24 px.
    for (style_name, size) in [
        (egui::TextStyle::Small, 13.0),
        (egui::TextStyle::Body, 16.0),
        (egui::TextStyle::Button, 16.0),
        (egui::TextStyle::Monospace, 14.0),
        (egui::TextStyle::Heading, 24.0),
    ] {
        style
            .text_styles
            .insert(style_name, FontId::proportional(size));
    }
    ctx.set_global_style(style);
}

/// A cheap deterministic value in [0, 1) for a small integer.
pub fn hash(n: i64) -> f32 {
    let mut x = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x & 0xFFFF) as f32 / 65536.0
}

/// The two tiling textures: fine film grain and large blotchy stains.
#[derive(Clone, Copy)]
pub enum Texture {
    Grain,
    Stain,
}

const TEXTURE_SIZE: usize = 128;

/// Periodic value noise on a `cells` lattice over the texture, so it tiles.
fn value_noise(x: usize, y: usize, cells: usize, seed: i64) -> f32 {
    let cell = TEXTURE_SIZE as f32 / cells as f32;
    let (fx, fy) = (x as f32 / cell, y as f32 / cell);
    let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
    let (tx, ty) = (fx - ix as f32, fy - iy as f32);
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let at = |i: usize, j: usize| hash(seed + ((i % cells) + (j % cells) * cells) as i64 * 7919);
    let top = at(ix, iy) + (at(ix + 1, iy) - at(ix, iy)) * smooth(tx);
    let bottom = at(ix, iy + 1) + (at(ix + 1, iy + 1) - at(ix, iy + 1)) * smooth(tx);
    top + (bottom - top) * smooth(ty)
}

fn make_texture(kind: Texture) -> egui::ColorImage {
    let mut pixels = Vec::with_capacity(TEXTURE_SIZE * TEXTURE_SIZE);
    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            let v = match kind {
                Texture::Grain => hash((y * TEXTURE_SIZE + x) as i64 + 17),
                Texture::Stain => {
                    0.50 * value_noise(x, y, 4, 1)
                        + 0.28 * value_noise(x, y, 8, 2)
                        + 0.14 * value_noise(x, y, 16, 3)
                        + 0.08 * value_noise(x, y, 32, 4)
                }
            };
            // Light specks above the middle, dark ones below, each as a
            // premultiplied white or black with that much alpha.
            let d = (v - 0.5) * 2.0;
            let d = match kind {
                Texture::Grain => d,
                // Stains: widen the contrast of the smooth field.
                Texture::Stain => (d * 2.2).clamp(-1.0, 1.0),
            };
            let a = (d.abs() * 255.0) as u8;
            pixels.push(if d >= 0.0 {
                Color32::from_rgba_premultiplied(a, a, a, a)
            } else {
                Color32::from_rgba_premultiplied(0, 0, 0, a)
            });
        }
    }
    egui::ColorImage::new([TEXTURE_SIZE, TEXTURE_SIZE], pixels)
}

/// The texture's id, made the first time it is asked for and kept in the
/// context's memory.
pub fn texture(ctx: &egui::Context, kind: Texture) -> TextureId {
    let id = egui::Id::new(("theme_texture", kind as u8));
    if let Some(handle) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return handle.id();
    }
    let options = egui::TextureOptions {
        magnification: egui::TextureFilter::Linear,
        minification: egui::TextureFilter::Linear,
        wrap_mode: egui::TextureWrapMode::Repeat,
        mipmap_mode: None,
    };
    let handle = ctx.load_texture(
        match kind {
            Texture::Grain => "theme-grain",
            Texture::Stain => "theme-stain",
        },
        make_texture(kind),
        options,
    );
    let tex = handle.id();
    ctx.data_mut(|d| d.insert_temp(id, handle));
    tex
}

/// Tiles one texture over `rect`: `tile` points per repeat, `offset` in
/// repeats, `alpha` 0 to 1.
pub fn overlay(
    painter: &egui::Painter,
    rect: Rect,
    kind: Texture,
    tile: f32,
    offset: Vec2,
    alpha: f32,
) {
    if alpha <= 0.0 || rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }
    let tex = texture(painter.ctx(), kind);
    let uv = Rect::from_min_size(
        Pos2::new(offset.x, offset.y),
        Vec2::new(rect.width() / tile, rect.height() / tile),
    );
    let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
    painter.image(tex, rect, uv, Color32::from_rgba_premultiplied(a, a, a, a));
}

/// Worn surface: faint stains and grain over a plate or a panel. `seed`
/// shifts the pattern so neighbouring plates do not repeat each other.
pub fn wear(painter: &egui::Painter, rect: Rect, theme: Theme, seed: f32) {
    if theme.wear <= 0.0 {
        return;
    }
    let offset = Vec2::new(hash(seed as i64), hash(seed as i64 + 5));
    overlay(
        painter,
        rect,
        Texture::Stain,
        420.0,
        offset,
        0.03 * theme.wear,
    );
    overlay(
        painter,
        rect,
        Texture::Grain,
        96.0,
        offset,
        0.035 * theme.wear,
    );
}

/// A worn metal plate: the card fill, stains and grain, a thin border and a
/// faint highlight along the top edge. `lit` draws the border in amber.
pub fn plate(painter: &egui::Painter, rect: Rect, theme: Theme, fill: Color32, lit: bool) {
    painter.rect_filled(rect, 3, fill);
    wear(painter, rect, theme, rect.min.x * 0.37 + rect.min.y * 1.3);
    painter.line_segment(
        [
            rect.left_top() + Vec2::new(3.0, 0.5),
            rect.right_top() + Vec2::new(-3.0, 0.5),
        ],
        Stroke::new(1.0, theme.bevel),
    );
    painter.rect_stroke(
        rect,
        3,
        Stroke::new(
            if lit { 1.5 } else { 1.0 },
            if lit { theme.accent } else { theme.card_border },
        ),
        egui::StrokeKind::Inside,
    );
}

/// A HUD panel over the scene: smoked glass with soft edges.
pub fn hud_panel(painter: &egui::Painter, rect: Rect) {
    painter.rect_filled(rect.expand(1.5), 7, Color32::from_black_alpha(40));
    painter.rect_filled(rect, 6, scene::HUD_BACK);
}

/// Text with a faint glow, like the numbers of a HUD. Returns its rect.
pub fn glow_text(
    painter: &egui::Painter,
    pos: Pos2,
    align: Align2,
    text: impl ToString,
    font: FontId,
    color: Color32,
    glow: bool,
) -> Rect {
    let galley = painter.layout_no_wrap(text.to_string(), font, color);
    let rect = align.anchor_size(pos, galley.size());
    if glow {
        let halo = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 22);
        for (dx, dy) in [
            (-2.0, 0.0),
            (2.0, 0.0),
            (0.0, -2.0),
            (0.0, 2.0),
            (-1.5, -1.5),
            (1.5, -1.5),
            (-1.5, 1.5),
            (1.5, 1.5),
        ] {
            painter.galley_with_override_text_color(
                rect.min + Vec2::new(dx, dy),
                galley.clone(),
                halo,
            );
        }
    }
    painter.galley(rect.min, galley, color);
    rect
}

/// Letter-spaced capitals, the voice of every label on a HUD or a plate.
pub fn caps(text: &str, size: f32, color: Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.append(
        &text.to_uppercase(),
        0.0,
        TextFormat {
            font_id: FontId::proportional(size),
            color,
            extra_letter_spacing: size * 0.12,
            ..Default::default()
        },
    );
    job
}

/// Paints letter-spaced capitals and returns their rect.
pub fn caps_text(
    painter: &egui::Painter,
    pos: Pos2,
    align: Align2,
    text: &str,
    size: f32,
    color: Color32,
) -> Rect {
    let galley = painter.layout_job(caps(text, size, color));
    let rect = align.anchor_size(pos, galley.size());
    painter.galley(rect.min, galley, color);
    rect
}

/// A section title in the side panel: a short amber bar, then capitals.
pub fn section(ui: &mut egui::Ui, text: &str, theme: Theme) -> Response {
    let galley = ui.painter().layout_job(caps(text, 13.0, theme.ink));
    let size = Vec2::new(galley.size().x + 12.0, galley.size().y.max(18.0));
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    let bar = Rect::from_min_size(
        Pos2::new(rect.left(), rect.center().y - 6.0),
        Vec2::new(3.0, 12.0),
    );
    ui.painter().rect_filled(bar, 0, theme.accent);
    ui.painter().galley(
        Pos2::new(rect.left() + 12.0, rect.center().y - galley.size().y / 2.0),
        galley,
        theme.ink,
    );
    response
}

/// One tab of the tab strip: capitals, an amber rule under the open tab.
pub fn tab(ui: &mut egui::Ui, selected: bool, text: &str, theme: Theme) -> Response {
    let galley = ui.painter().layout_job(caps(text, 15.0, theme.ink));
    let size = galley.size() + Vec2::new(24.0, 18.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let hovered = response.hovered();
    if hovered && !selected {
        ui.painter().rect_filled(rect, 2, theme.card);
    }
    let color = if selected {
        theme.accent
    } else if hovered {
        theme.ink
    } else {
        theme.muted
    };
    let at = rect.center() - galley.size() / 2.0 - Vec2::new(0.0, 1.0);
    if selected && theme.dark {
        let halo = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 20);
        for (dx, dy) in [(-1.5, 0.0), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5)] {
            ui.painter().galley_with_override_text_color(
                at + Vec2::new(dx, dy),
                galley.clone(),
                halo,
            );
        }
    }
    ui.painter()
        .galley_with_override_text_color(at, galley, color);
    if selected {
        ui.painter().rect_filled(
            Rect::from_min_max(
                Pos2::new(rect.left() + 6.0, rect.bottom() - 3.0),
                Pos2::new(rect.right() - 6.0, rect.bottom() - 1.0),
            ),
            0,
            theme.accent,
        );
    }
    response
}

/// Adds a band between an outer and an inner rect to a mesh: `outer` color
/// at the outer edge, `inner` at the inner one.
fn ring(mesh: &mut Mesh, outer: Rect, inner: Rect, outer_color: Color32, inner_color: Color32) {
    let base = mesh.vertices.len() as u32;
    for (pos, color) in [
        (outer.left_top(), outer_color),
        (outer.right_top(), outer_color),
        (outer.right_bottom(), outer_color),
        (outer.left_bottom(), outer_color),
        (inner.left_top(), inner_color),
        (inner.right_top(), inner_color),
        (inner.right_bottom(), inner_color),
        (inner.left_bottom(), inner_color),
    ] {
        mesh.vertices.push(Vertex {
            pos,
            uv: egui::epaint::WHITE_UV,
            color,
        });
    }
    for side in 0..4u32 {
        let (a, b) = (side, (side + 1) % 4);
        mesh.indices.extend_from_slice(&[
            base + a,
            base + b,
            base + 4 + b,
            base + a,
            base + 4 + b,
            base + 4 + a,
        ]);
    }
}

/// Darkens the edges of a scene. `strength` is the alpha at the very edge.
pub fn vignette(painter: &egui::Painter, rect: Rect, strength: f32) {
    let mut mesh = Mesh::default();
    let size = rect.size();
    let mid = rect.shrink2(size * 0.04);
    let inner = rect.shrink2(size * Vec2::new(0.22, 0.30));
    let edge = Color32::from_black_alpha((strength.clamp(0.0, 1.0) * 255.0) as u8);
    let half = Color32::from_black_alpha((strength.clamp(0.0, 1.0) * 110.0) as u8);
    ring(&mut mesh, rect, mid, edge, half);
    ring(&mut mesh, mid, inner, half, Color32::TRANSPARENT);
    painter.add(egui::Shape::mesh(mesh));
}

/// Film grain over a scene. `time` moves the pattern, so the grain lives
/// while the scene plays and holds still while it is paused.
pub fn grain(painter: &egui::Painter, rect: Rect, time: f32, alpha: f32) {
    let step = (time * 24.0).floor() as i64;
    let offset = Vec2::new(hash(step), hash(step + 101));
    overlay(painter, rect, Texture::Grain, 110.0, offset, alpha);
}

/// One line of a HUD block.
pub struct HudLine {
    pub text: String,
    pub size: f32,
    pub color: Color32,
    /// Letter-spaced capitals, for labels.
    pub caps: bool,
    /// A faint glow, for the numbers.
    pub glow: bool,
}

impl HudLine {
    /// A small capital label in dim amber.
    pub fn label(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            size: 11.0,
            color: scene::HUD_DIM,
            caps: true,
            glow: false,
        }
    }
    /// A big glowing number.
    pub fn value(text: String, size: f32, color: Color32) -> Self {
        Self {
            text,
            size,
            color,
            caps: false,
            glow: true,
        }
    }
    /// Plain text.
    pub fn text(text: String, size: f32, color: Color32) -> Self {
        Self {
            text,
            size,
            color,
            caps: false,
            glow: false,
        }
    }
}

/// A HUD panel holding lines of text, anchored at `anchor` by `align`, with
/// the text aligned to the same side. Returns the panel's rect.
pub fn hud_block(painter: &egui::Painter, anchor: Pos2, align: Align2, lines: &[HudLine]) -> Rect {
    const PAD: Vec2 = Vec2::new(12.0, 8.0);
    let galleys: Vec<_> = lines
        .iter()
        .map(|line| {
            if line.caps {
                painter.layout_job(caps(&line.text, line.size, line.color))
            } else {
                painter.layout_no_wrap(
                    line.text.clone(),
                    FontId::proportional(line.size),
                    line.color,
                )
            }
        })
        .collect();
    let width = galleys.iter().map(|g| g.size().x).fold(0.0, f32::max);
    let height: f32 = galleys.iter().map(|g| g.size().y).sum();
    let rect = align.anchor_size(anchor, Vec2::new(width, height) + PAD * 2.0);
    hud_panel(painter, rect);
    let mut y = rect.top() + PAD.y;
    for (line, galley) in lines.iter().zip(galleys) {
        let x = match align.x() {
            egui::Align::Max => rect.right() - PAD.x - galley.size().x,
            _ => rect.left() + PAD.x,
        };
        let at = Pos2::new(x, y);
        y += galley.size().y;
        if line.glow {
            let c = line.color;
            let halo = Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 24);
            for (dx, dy) in [
                (-2.0, 0.0),
                (2.0, 0.0),
                (0.0, -2.0),
                (0.0, 2.0),
                (-1.5, -1.5),
                (1.5, -1.5),
                (-1.5, 1.5),
                (1.5, 1.5),
            ] {
                painter.galley_with_override_text_color(
                    at + Vec2::new(dx, dy),
                    galley.clone(),
                    halo,
                );
            }
        }
        painter.galley(at, galley, line.color);
    }
    rect
}
