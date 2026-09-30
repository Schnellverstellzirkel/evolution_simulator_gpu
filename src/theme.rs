//! The look of the game, after Half-Life 2 and the Source engine menus:
//! translucent dark panels with hairline borders over a blurred picture of
//! the city, the menus' orange for what the pointer arms, the HUD's
//! yellow digits with a soft glow in smoked-glass boxes, and cold Combine
//! blue for what is secondary. Everything here is a few shapes or one
//! textured quad, so it costs next to nothing per frame.
use crate::assets::{self, Art};
use eframe::egui::{
    self, Align2, Color32, FontFamily, FontId, Pos2, Rect, Response, Sense, Stroke, Vec2,
    epaint::{Mesh, Vertex},
    text::{LayoutJob, TextFormat},
};

/// Colors of the scene: the replay, the race lanes and the thumbnails.
pub mod scene {
    use eframe::egui::Color32;
    /// Haze at the horizon, and the sky color where no texture reaches.
    pub const SKY_TOP: Color32 = Color32::from_rgb(70, 80, 84);
    pub const SKY_HORIZON: Color32 = Color32::from_rgb(132, 140, 140);
    /// Street ground, from the surface down.
    pub const GROUND_TOP: Color32 = Color32::from_rgb(88, 86, 80);
    pub const GROUND_DEEP: Color32 = Color32::from_rgb(34, 34, 32);
    /// The worn lip along the ground surface.
    pub const GROUND_EDGE: Color32 = Color32::from_rgb(158, 152, 132);
    /// Meter labels and ticks on the ground.
    pub const GROUND_INK: Color32 = Color32::from_rgb(196, 190, 170);
    /// Faint meter lines across the sky.
    pub const GRID: Color32 = Color32::from_rgba_premultiplied(14, 14, 14, 14);
    /// Sludge of the mud layer, its lower edge and its wet shine.
    pub const MUD: Color32 = Color32::from_rgb(60, 44, 26);
    pub const MUD_EDGE: Color32 = Color32::from_rgb(34, 25, 14);
    pub const MUD_SHEEN: Color32 = Color32::from_rgba_premultiplied(66, 52, 34, 120);
    /// Creature parts: steel bones, flesh muscles, dark outlines.
    pub const OUTLINE: Color32 = Color32::from_rgb(10, 10, 10);
    pub const BONE: Color32 = Color32::from_rgb(170, 174, 170);
    pub const BONE_SHINE: Color32 = Color32::from_rgb(226, 230, 224);
    pub const ORGAN: Color32 = Color32::from_rgb(168, 72, 64);
    pub const MUSCLE_REST: Color32 = Color32::from_rgb(196, 120, 100);
    pub const MUSCLE_ACTIVE: Color32 = Color32::from_rgb(160, 22, 16);
    pub const MUSCLE_TIRED: Color32 = Color32::from_rgb(118, 116, 108);
    /// Node shells: slick steel blue for low friction, brass for high.
    pub const NODE_SLICK: Color32 = Color32::from_rgb(150, 178, 196);
    pub const NODE_GRIPPY: Color32 = Color32::from_rgb(204, 170, 104);
    /// The head's eye glows like a HUD light.
    pub const EYE: Color32 = Color32::from_rgb(255, 220, 120);
    /// Ring around a node on the ground.
    pub const TOUCHDOWN: Color32 = Color32::from_rgb(255, 204, 64);
    /// A fall, a broken joint: the HUD's damage red.
    pub const FALLEN: Color32 = Color32::from_rgb(232, 52, 36);
    /// Force arrows: muscle pulls and ground pushes.
    pub const FORCE_MUSCLE: Color32 = Color32::from_rgb(255, 150, 30);
    pub const FORCE_GROUND: Color32 = Color32::from_rgb(90, 180, 240);
    /// The HUD: yellow digits and labels on smoked glass (Half-Life 2's
    /// FgColor 255 220 0 and BgColor 0 0 0 76).
    pub const HUD: Color32 = Color32::from_rgb(255, 220, 0);
    pub const HUD_DIM: Color32 = Color32::from_rgb(230, 196, 40);
    pub const HUD_INK: Color32 = Color32::from_rgb(228, 222, 206);
    pub const HUD_BACK: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 96);
    /// The damaged HUD: a red wash behind red digits.
    pub const HUD_DAMAGED: Color32 = Color32::from_rgba_premultiplied(80, 0, 0, 150);
    /// The centre of mass and its trail.
    pub const TRAIL: Color32 = Color32::from_rgb(255, 210, 60);
}

/// Interface colors.
#[derive(Clone, Copy)]
pub struct Theme {
    /// Side panel, top bar, status line: dark glass over the backdrop.
    pub panel: Color32,
    /// The central area: a lighter veil over the backdrop.
    pub canvas: Color32,
    pub card: Color32,
    pub card_hover: Color32,
    pub card_border: Color32,
    /// A faint line along the top edge of a plate.
    pub bevel: Color32,
    pub ink: Color32,
    pub muted: Color32,
    /// The Source menus' armed orange: the numbers and choices that matter.
    pub accent: Color32,
    /// Combine blue: secondary lines and what waits.
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
    /// The armed (selected) fill and text of a menu item.
    pub armed_fill: Color32,
    pub armed_text: Color32,
}

const fn glass(r: u8, g: u8, b: u8, a: u8) -> Color32 {
    // Premultiplied: the color scaled by its alpha.
    Color32::from_rgba_premultiplied(
        (r as u16 * a as u16 / 255) as u8,
        (g as u16 * a as u16 / 255) as u8,
        (b as u16 * a as u16 / 255) as u8,
        a,
    )
}

impl Theme {
    pub fn get() -> Self {
        Self {
            panel: glass(14, 15, 15, 222),
            canvas: glass(8, 9, 9, 120),
            card: glass(0, 0, 0, 132),
            card_hover: glass(52, 40, 18, 176),
            card_border: glass(255, 255, 255, 34),
            bevel: glass(255, 255, 255, 20),
            ink: Color32::from_rgb(216, 216, 210),
            muted: Color32::from_rgb(150, 150, 146),
            accent: Color32::from_rgb(255, 176, 0),
            cold: Color32::from_rgb(110, 184, 220),
            warn: Color32::from_rgb(220, 120, 60),
            danger: Color32::from_rgb(236, 64, 48),
            record: Color32::from_rgb(255, 120, 40),
            go_fill: Color32::from_rgb(134, 91, 19),
            go_text: Color32::from_rgb(255, 214, 120),
            stop_fill: Color32::from_rgb(96, 30, 22),
            stop_text: Color32::from_rgb(255, 176, 150),
            armed_fill: Color32::from_rgb(134, 91, 19),
            armed_text: Color32::from_rgb(255, 196, 40),
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

/// Applies the style: Source menu colors, the game's fonts, and panels of
/// dark glass so the city shows through behind them.
pub fn apply_style(ctx: &egui::Context) {
    let theme = Theme::get();
    ctx.set_theme(egui::Theme::Dark);
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = Vec2::new(GAP_M, GAP_M);
    style.spacing.button_padding = Vec2::new(12.0, 7.0);
    style.spacing.interact_size = Vec2::new(40.0, CONTROL_HEIGHT);
    style.spacing.slider_width = 120.0;
    style.spacing.window_margin = egui::Margin::same(GAP_L as i8);
    style.spacing.menu_margin = egui::Margin::same(GAP_M as i8);
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(theme.ink);
    visuals.weak_text_color = Some(theme.muted);
    visuals.panel_fill = theme.panel;
    visuals.window_fill = glass(20, 21, 21, 238);
    visuals.faint_bg_color = glass(255, 255, 255, 10);
    visuals.extreme_bg_color = glass(0, 0, 0, 120);
    visuals.text_edit_bg_color = Some(glass(0, 0, 0, 170));
    visuals.code_bg_color = glass(0, 0, 0, 120);
    visuals.hyperlink_color = theme.accent;
    visuals.warn_fg_color = theme.warn;
    visuals.error_fg_color = theme.danger;
    visuals.selection.bg_fill = theme.armed_fill;
    visuals.selection.stroke = Stroke::new(1.0, theme.armed_text);
    visuals.slider_trailing_fill = true;
    let bright = Color32::from_rgb(255, 255, 250);
    visuals.widgets.noninteractive = widget(theme.panel, glass(255, 255, 255, 26), theme.ink, 1.0);
    visuals.widgets.inactive = widget(
        glass(60, 62, 62, 200),
        glass(255, 255, 255, 40),
        theme.ink,
        1.0,
    );
    visuals.widgets.hovered = widget(glass(76, 64, 40, 220), glass(255, 176, 0, 150), bright, 1.5);
    visuals.widgets.active = widget(theme.armed_fill, theme.accent, theme.armed_text, 2.0);
    visuals.widgets.open = widget(
        glass(44, 46, 46, 235),
        glass(255, 255, 255, 40),
        bright,
        1.0,
    );
    visuals.window_corner_radius = egui::CornerRadius::same(6);
    visuals.menu_corner_radius = egui::CornerRadius::same(4);
    visuals.window_stroke = Stroke::new(1.0, glass(255, 255, 255, 46));
    visuals.window_shadow = egui::Shadow {
        offset: [0, 8],
        blur: 26,
        spread: 0,
        color: Color32::from_black_alpha(170),
    };
    visuals.popup_shadow = egui::Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: Color32::from_black_alpha(140),
    };
    style.visuals = visuals;
    // Type scale: small print 13 px, body and buttons 15 px, headings 22 px.
    for (style_name, font) in [
        (egui::TextStyle::Small, FontId::proportional(13.0)),
        (egui::TextStyle::Body, FontId::proportional(15.0)),
        (egui::TextStyle::Button, FontId::proportional(15.0)),
        (egui::TextStyle::Monospace, FontId::monospace(13.0)),
        (
            egui::TextStyle::Heading,
            FontId::new(22.0, assets::hud_bold()),
        ),
    ] {
        style.text_styles.insert(style_name, font);
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

/// A textured quad: `art` tiled so one repeat covers `tile` points, shifted
/// by `offset` repeats, tinted by `tint` (premultiplied).
pub fn tiled(
    painter: &egui::Painter,
    rect: Rect,
    art: Art,
    tile: Vec2,
    offset: Vec2,
    tint: Color32,
) {
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }
    let uv = Rect::from_min_size(
        Pos2::new(offset.x, offset.y),
        Vec2::new(rect.width() / tile.x, rect.height() / tile.y),
    );
    painter.image(art.texture(painter.ctx()), rect, uv, tint);
}

/// The blurred city behind the whole window, like the live scene behind a
/// Half-Life 2 menu. It covers `rect` and keeps its aspect.
pub fn backdrop(painter: &egui::Painter, rect: Rect) {
    let size = Art::Backdrop.size(painter.ctx());
    let scale = (rect.width() / size.x).max(rect.height() / size.y);
    let shown = size * scale;
    let min = rect.center() - shown / 2.0;
    painter.image(
        Art::Backdrop.texture(painter.ctx()),
        Rect::from_min_size(min, shown),
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );
}

/// Kept for the plates: nothing to add over the glass panels.
pub fn wear(_painter: &egui::Painter, _rect: Rect, _theme: Theme, _seed: f32) {}

/// A panel of dark glass: the fill, a hairline border and a faint top edge.
/// `lit` draws the border in the armed orange.
pub fn plate(painter: &egui::Painter, rect: Rect, theme: Theme, fill: Color32, lit: bool) {
    painter.rect_filled(rect, 4, fill);
    painter.line_segment(
        [
            rect.left_top() + Vec2::new(4.0, 0.5),
            rect.right_top() + Vec2::new(-4.0, 0.5),
        ],
        Stroke::new(1.0, theme.bevel),
    );
    painter.rect_stroke(
        rect,
        4,
        Stroke::new(
            if lit { 1.5 } else { 1.0 },
            if lit {
                theme.accent.gamma_multiply(0.8)
            } else {
                theme.card_border
            },
        ),
        egui::StrokeKind::Inside,
    );
}

/// A HUD box: smoked glass with rounded corners, as Half-Life 2 draws it.
pub fn hud_panel(painter: &egui::Painter, rect: Rect) {
    painter.rect_filled(rect, 8, scene::HUD_BACK);
}

/// Paints a galley with a soft glow around it, like the HUD numbers'
/// blurred twin font. `strength` scales the glow.
fn glow_galley(
    painter: &egui::Painter,
    at: Pos2,
    galley: std::sync::Arc<egui::Galley>,
    color: Color32,
    strength: f32,
) {
    if strength > 0.0 {
        let a = |x: f32| (x * strength).clamp(0.0, 255.0) as u8;
        let halo =
            |alpha: u8| Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
        for (radius, alpha, steps) in [(3.5_f32, a(10.0), 12), (1.8, a(22.0), 8)] {
            for i in 0..steps {
                let angle = i as f32 / steps as f32 * std::f32::consts::TAU;
                painter.galley_with_override_text_color(
                    at + Vec2::new(angle.cos(), angle.sin()) * radius,
                    galley.clone(),
                    halo(alpha),
                );
            }
        }
    }
    painter.galley_with_override_text_color(at, galley, color);
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
    glow_galley(
        painter,
        rect.min,
        galley,
        color,
        if glow { 1.0 } else { 0.0 },
    );
    rect
}

/// Letter-spaced capitals, the voice of every label on the HUD.
pub fn caps(text: &str, size: f32, color: Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.append(
        &text.to_uppercase(),
        0.0,
        TextFormat {
            font_id: FontId::new(size, assets::label_bold()),
            color,
            extra_letter_spacing: size * 0.08,
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

/// A section title in the side panel: a short orange bar, then capitals.
pub fn section(ui: &mut egui::Ui, text: &str, theme: Theme) -> Response {
    let galley = ui.painter().layout_job(caps(text, 12.0, theme.ink));
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

/// One tab of the tab strip, drawn like an item of the Half-Life 2 main
/// menu: plain words, brighter when armed, the open one in orange with a
/// thin rule under it.
pub fn tab(ui: &mut egui::Ui, selected: bool, text: &str, theme: Theme) -> Response {
    let font = FontId::new(19.0, FontFamily::Proportional);
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font, theme.ink);
    let size = galley.size() + Vec2::new(22.0, 16.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let hovered = response.hovered();
    let color = if selected {
        theme.accent
    } else if hovered {
        Color32::WHITE
    } else {
        theme.muted
    };
    let at = rect.center() - galley.size() / 2.0 - Vec2::new(0.0, 1.0);
    // A drop shadow keeps the words readable over the city.
    ui.painter().galley_with_override_text_color(
        at + Vec2::new(1.0, 1.5),
        galley.clone(),
        Color32::from_black_alpha(160),
    );
    glow_galley(
        ui.painter(),
        at,
        galley,
        color,
        if selected { 0.8 } else { 0.0 },
    );
    if selected {
        ui.painter().rect_filled(
            Rect::from_min_max(
                Pos2::new(rect.left() + 8.0, rect.bottom() - 3.0),
                Pos2::new(rect.right() - 8.0, rect.bottom() - 1.0),
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

/// The fine film-grain texture, made once and kept in the context.
fn grain_texture(ctx: &egui::Context) -> egui::TextureId {
    let id = egui::Id::new("theme_grain");
    if let Some(handle) = ctx.data(|d| d.get_temp::<egui::TextureHandle>(id)) {
        return handle.id();
    }
    const SIZE: usize = 128;
    let pixels = (0..SIZE * SIZE)
        .map(|i| {
            let d = (hash(i as i64 + 17) - 0.5) * 2.0;
            let a = (d.abs() * 255.0) as u8;
            if d >= 0.0 {
                Color32::from_rgba_premultiplied(a, a, a, a)
            } else {
                Color32::from_rgba_premultiplied(0, 0, 0, a)
            }
        })
        .collect();
    let options = egui::TextureOptions {
        magnification: egui::TextureFilter::Linear,
        minification: egui::TextureFilter::Linear,
        wrap_mode: egui::TextureWrapMode::Repeat,
        mipmap_mode: None,
    };
    let handle = ctx.load_texture(
        "theme-grain",
        egui::ColorImage::new([SIZE, SIZE], pixels),
        options,
    );
    let tex = handle.id();
    ctx.data_mut(|d| d.insert_temp(id, handle));
    tex
}

/// Film grain over a scene. `time` moves the pattern, so the grain lives
/// while the scene plays and holds still while it is paused.
pub fn grain(painter: &egui::Painter, rect: Rect, time: f32, alpha: f32) {
    if alpha <= 0.0 {
        return;
    }
    let step = (time * 24.0).floor() as i64;
    let tex = grain_texture(painter.ctx());
    let uv = Rect::from_min_size(
        Pos2::new(hash(step), hash(step + 101)),
        Vec2::new(rect.width() / 110.0, rect.height() / 110.0),
    );
    let a = (alpha.clamp(0.0, 1.0) * 255.0) as u8;
    painter.image(tex, rect, uv, Color32::from_rgba_premultiplied(a, a, a, a));
}

/// One line of a HUD block.
pub struct HudLine {
    pub text: String,
    pub size: f32,
    pub color: Color32,
    /// Bold letter-spaced capitals, for labels.
    pub caps: bool,
    /// The HUD digit face with its glow, for the numbers.
    pub glow: bool,
}

impl HudLine {
    /// A small capital label in HUD yellow.
    pub fn label(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            size: 10.5,
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

/// A HUD box holding lines of text, anchored at `anchor` by `align`, with
/// the text aligned to the same side. Returns the box's rect.
pub fn hud_block(painter: &egui::Painter, anchor: Pos2, align: Align2, lines: &[HudLine]) -> Rect {
    const PAD: Vec2 = Vec2::new(12.0, 8.0);
    let galleys: Vec<_> = lines
        .iter()
        .map(|line| {
            if line.caps {
                painter.layout_job(caps(&line.text, line.size, line.color))
            } else if line.glow {
                painter.layout_no_wrap(
                    line.text.clone(),
                    FontId::new(line.size, assets::hud()),
                    line.color,
                )
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
            egui::Align::Center => rect.center().x - galley.size().x / 2.0,
            egui::Align::Min => rect.left() + PAD.x,
        };
        let at = Pos2::new(x, y);
        y += galley.size().y;
        glow_galley(
            painter,
            at,
            galley,
            line.color,
            if line.glow { 1.0 } else { 0.0 },
        );
    }
    rect
}

/// A Half-Life 2 counter: a smoked-glass box with its label low on the
/// left and big glowing digits beside it, like HEALTH and SUIT. `unit`
/// follows the digits small, `extra` sits right of them like the reserve
/// count of the ammo box. `damaged` turns it red, as the HUD does when hit.
pub struct Counter<'a> {
    pub label: &'a str,
    pub digits: String,
    pub unit: &'a str,
    pub extra: Option<String>,
    pub damaged: bool,
}

/// Paints a counter anchored at `anchor` by `align` and returns its rect.
/// `size` is the digit height.
pub fn counter(
    painter: &egui::Painter,
    anchor: Pos2,
    align: Align2,
    counter: &Counter,
    size: f32,
) -> Rect {
    let color = if counter.damaged {
        scene::FALLEN
    } else {
        scene::HUD
    };
    let label = painter.layout_job(caps(
        counter.label,
        (size * 0.3).max(9.0),
        color.gamma_multiply(0.9),
    ));
    let digits = painter.layout_no_wrap(
        counter.digits.clone(),
        FontId::new(size, assets::hud()),
        color,
    );
    let unit = painter.layout_no_wrap(
        counter.unit.to_owned(),
        FontId::new(size * 0.42, assets::hud()),
        color,
    );
    let extra = counter.extra.as_ref().map(|text| {
        painter.layout_no_wrap(text.clone(), FontId::new(size * 0.5, assets::hud()), color)
    });
    let pad = size * 0.3;
    let gap = size * 0.28;
    let width = pad
        + label.size().x
        + gap
        + digits.size().x
        + unit.size().x
        + extra.as_ref().map_or(0.0, |g| g.size().x + gap * 1.4)
        + pad;
    let height = digits.size().y + size * 0.08;
    let rect = align.anchor_size(anchor, Vec2::new(width, height));
    painter.rect_filled(
        rect,
        (size * 0.26) as u8,
        if counter.damaged {
            scene::HUD_DAMAGED
        } else {
            scene::HUD_BACK
        },
    );
    let baseline = rect.bottom() - size * 0.22;
    let mut x = rect.left() + pad;
    painter.galley(
        Pos2::new(x, baseline - label.size().y),
        label.clone(),
        color,
    );
    x += label.size().x + gap;
    let digits_width = digits.size().x;
    glow_galley(painter, Pos2::new(x, rect.top()), digits, color, 1.0);
    x += digits_width;
    painter.galley(
        Pos2::new(x + 1.0, baseline - unit.size().y + size * 0.05),
        unit.clone(),
        color,
    );
    x += unit.size().x;
    if let Some(extra) = extra {
        x += gap * 1.4;
        let at = Pos2::new(x, baseline - extra.size().y + size * 0.08);
        glow_galley(painter, at, extra, color, 0.8);
    }
    rect
}
