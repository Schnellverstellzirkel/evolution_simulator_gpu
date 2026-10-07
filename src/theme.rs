//! This module holds the look of the game: its colors, its egui style and the
//! small painters that the interface shares. The look is a warm painted poster
//! after Team Fortress 2 promotional art, and its palette is in `poster`. The
//! replay scene keeps its own colors in `scene` and a smoked-glass HUD, framed
//! like a picture on the poster. `ui.rs`, the modules under `ui/` and
//! `world_fx.rs` draw with it.
use crate::assets::{self, Art};
use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, Response, Sense, Stroke, Vec2,
    epaint::{Mesh, Vertex},
    text::{LayoutJob, TextFormat},
};

/// Colors of the scene: the replay, the race lanes, the creature thumbnails
/// and the exported GIF.
pub mod scene {
    use eframe::egui::Color32;
    /// The sky at the top edge of an exported GIF.
    pub const SKY_TOP: Color32 = Color32::from_rgb(70, 80, 84);
    /// Haze at the horizon. It fills the replay behind the sky texture, tints
    /// the fog between the skyline layers and ends the sky of an exported GIF.
    pub const SKY_HORIZON: Color32 = Color32::from_rgb(132, 140, 140);
    /// The ground fill of an exported GIF.
    pub const GROUND_TOP: Color32 = Color32::from_rgb(88, 86, 80);
    /// Street ground at depth. Nothing uses it now.
    pub const GROUND_DEEP: Color32 = Color32::from_rgb(34, 34, 32);
    /// The worn lip along the ground surface.
    pub const GROUND_EDGE: Color32 = Color32::from_rgb(158, 152, 132);
    /// Meter labels and ticks on the ground.
    pub const GROUND_INK: Color32 = Color32::from_rgb(196, 190, 170);
    /// Faint vertical meter lines across the sky.
    pub const GRID: Color32 = Color32::from_rgba_premultiplied(14, 14, 14, 14);
    /// The sludge of the mud layer.
    pub const MUD: Color32 = Color32::from_rgb(60, 44, 26);
    /// The line along the lower edge of the mud layer.
    pub const MUD_EDGE: Color32 = Color32::from_rgb(34, 25, 14);
    /// The wet shine along the top of the mud layer.
    pub const MUD_SHEEN: Color32 = Color32::from_rgba_premultiplied(66, 52, 34, 120);
    /// The dark outline of bones, muscles, organs and nodes, and the pupil of
    /// the eye.
    pub const OUTLINE: Color32 = Color32::from_rgb(10, 10, 10);
    /// A bone: steel.
    pub const BONE: Color32 = Color32::from_rgb(170, 174, 170);
    /// The bright streak along the upper side of a bone.
    pub const BONE_SHINE: Color32 = Color32::from_rgb(226, 230, 224);
    /// An organ, the disc on a bone that carries organ mass.
    pub const ORGAN: Color32 = Color32::from_rgb(168, 72, 64);
    /// A muscle at rest: pale flesh.
    pub const MUSCLE_REST: Color32 = Color32::from_rgb(196, 120, 100);
    /// A muscle at full contraction: deep red.
    pub const MUSCLE_ACTIVE: Color32 = Color32::from_rgb(160, 22, 16);
    /// A muscle with no energy left: grey.
    pub const MUSCLE_TIRED: Color32 = Color32::from_rgb(118, 116, 108);
    /// A node shell at the lowest friction: slick steel blue.
    pub const NODE_SLICK: Color32 = Color32::from_rgb(150, 178, 196);
    /// A node shell at the highest friction: brass.
    pub const NODE_GRIPPY: Color32 = Color32::from_rgb(204, 170, 104);
    /// The head's eye. It glows like a HUD light.
    pub const EYE: Color32 = Color32::from_rgb(255, 220, 120);
    /// The ring around a node on the ground.
    pub const TOUCHDOWN: Color32 = Color32::from_rgb(255, 204, 64);
    /// The damage red: a fall, a broken joint, a damaged counter.
    pub const FALLEN: Color32 = Color32::from_rgb(232, 52, 36);
    /// The arrows for muscle pulls.
    pub const FORCE_MUSCLE: Color32 = Color32::from_rgb(255, 150, 30);
    /// The arrows for ground pushes.
    pub const FORCE_GROUND: Color32 = Color32::from_rgb(90, 180, 240);
    /// HUD digits and text: yellow on dark glass over the scene.
    pub const HUD: Color32 = Color32::from_rgb(255, 220, 0);
    /// Small HUD labels: a dimmer yellow.
    pub const HUD_DIM: Color32 = Color32::from_rgb(230, 196, 40);
    /// Plain HUD text: a warm off-white.
    pub const HUD_INK: Color32 = Color32::from_rgb(228, 222, 206);
    /// The dark glass behind a HUD box.
    pub const HUD_BACK: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 150);
    /// The red wash behind a damaged counter.
    pub const HUD_DAMAGED: Color32 = Color32::from_rgba_premultiplied(80, 0, 0, 150);
    /// The center of mass and its trail.
    pub const TRAIL: Color32 = Color32::from_rgb(255, 210, 60);
}

/// The interface colors. `Theme::get` returns the one theme the game has, and
/// the `ui` modules take it by value.
#[derive(Clone, Copy)]
pub struct Theme {
    /// Side panel and status line: tan paper.
    pub panel: Color32,
    /// The central area: clear, so the sunburst shows.
    pub canvas: Color32,
    /// The fill of a card: cream.
    pub card: Color32,
    /// The fill of a hovered or lit card: pale mustard.
    pub card_hover: Color32,
    /// The outline of a card: dark brown.
    pub card_border: Color32,
    /// A pale line along the top edge of a plate. Nothing paints it now.
    pub bevel: Color32,
    /// Text and outlines: dark brown.
    pub ink: Color32,
    /// Text of second rank: a lighter brown.
    pub muted: Color32,
    /// Brick red: the numbers and choices that matter.
    pub accent: Color32,
    /// BLU blue: secondary lines and what waits.
    pub cold: Color32,
    /// Rust orange: warnings, such as a failed trial or a catastrophe, and
    /// world-change marks.
    pub warn: Color32,
    /// Red for error messages.
    pub danger: Color32,
    /// Record markers on the charts.
    pub record: Color32,
    /// The fill of the Evolve button: green.
    pub go_fill: Color32,
    /// The text on `go_fill`: cream.
    pub go_text: Color32,
    /// The fill of the Pause evolution button: brick red. A selected effect
    /// level away from calm uses it too.
    pub stop_fill: Color32,
    /// The text on `stop_fill`: cream.
    pub stop_text: Color32,
    /// The fill of what is armed: mustard. A selected effect level at calm
    /// uses it.
    pub armed_fill: Color32,
    /// The text on `armed_fill`: dark brown. Nothing reads it now.
    pub armed_text: Color32,
}

/// The poster palette: dark brown ink, cream and tan paper, and mustard, brick
/// red and BLU blue. `schematic.rs` has its own fixed colors, and most of them
/// match these.
pub mod poster {
    use eframe::egui::Color32;
    /// Dark brown outlines and text.
    pub const INK: Color32 = Color32::from_rgb(50, 34, 26);
    /// Text of second rank. Its contrast is 7.9:1 on `CREAM` and 4.7:1 on
    /// `TAN`.
    pub const INK_SOFT: Color32 = Color32::from_rgb(98, 72, 52);
    /// The paper behind everything.
    pub const PAPER: Color32 = Color32::from_rgb(232, 210, 160);
    /// The paler rays of the sunburst on the paper.
    pub const PAPER_RAY: Color32 = Color32::from_rgb(240, 222, 176);
    /// The side panel and the status line.
    pub const TAN: Color32 = Color32::from_rgb(218, 190, 134);
    /// The fill of cards and windows.
    pub const PANEL: Color32 = Color32::from_rgb(251, 242, 216);
    /// The fill of buttons and text boxes, and the text on dark fills.
    pub const CREAM: Color32 = Color32::from_rgb(255, 247, 226);
    /// Brick red. It fills the Pause evolution button.
    pub const RED: Color32 = Color32::from_rgb(184, 56, 50);
    /// BLU blue, for what is secondary. Nothing uses it directly, and
    /// `Theme::cold` is a deeper shade of it.
    pub const BLU: Color32 = Color32::from_rgb(70, 108, 138);
    /// Mustard, for what is armed.
    pub const MUSTARD: Color32 = Color32::from_rgb(228, 166, 52);
    /// Grass green. It fills the Evolve button.
    pub const GRASS: Color32 = Color32::from_rgb(108, 122, 54);
    /// The top bar: dark wood.
    pub const WOOD_DARK: Color32 = Color32::from_rgb(58, 40, 30);
}

impl Theme {
    /// Returns the interface colors. The game has one theme, so every caller
    /// gets the same colors.
    pub fn get() -> Self {
        use poster::*;
        Self {
            panel: TAN,
            canvas: Color32::TRANSPARENT,
            card: PANEL,
            card_hover: Color32::from_rgb(255, 236, 184),
            card_border: INK,
            bevel: Color32::from_rgba_premultiplied(255, 255, 255, 120),
            ink: INK,
            muted: INK_SOFT,
            accent: Color32::from_rgb(170, 48, 40),
            cold: Color32::from_rgb(52, 90, 122),
            warn: Color32::from_rgb(168, 80, 24),
            danger: Color32::from_rgb(176, 36, 30),
            record: Color32::from_rgb(168, 76, 24),
            go_fill: GRASS,
            go_text: CREAM,
            stop_fill: RED,
            stop_text: CREAM,
            armed_fill: MUSTARD,
            armed_text: INK,
        }
    }
}

/// The medium gap, in points. It is the item spacing and the menu margin of
/// the style, and the `ui` modules use it between blocks.
pub const GAP_M: f32 = 8.0;
/// The large gap, in points. It is the window margin of the style, and the
/// `ui` modules use it for panel margins and between sections.
pub const GAP_L: f32 = 16.0;
/// The minimum height of a button, menu or selectable in a row, and the
/// starting height of a row, so a row's items share one center line.
pub const CONTROL_HEIGHT: f32 = 34.0;

/// The look of one widget state: a box of `fill` with an outline of
/// `stroke_width` in `stroke`. `text` and `text_width` make the stroke for text
/// and icons. The corners have a radius of 5 and the box does not grow.
fn widget(
    fill: Color32,
    stroke: Color32,
    stroke_width: f32,
    text: Color32,
    text_width: f32,
) -> egui::style::WidgetVisuals {
    egui::style::WidgetVisuals {
        bg_fill: fill,
        weak_bg_fill: fill,
        bg_stroke: Stroke::new(stroke_width, stroke),
        corner_radius: egui::CornerRadius::same(5),
        fg_stroke: Stroke::new(text_width, text),
        expansion: 0.0,
    }
}

/// Sets the egui style of the whole window: light panels with dark brown text,
/// thick dark outlines on every control, the spacing from `GAP_M`, `GAP_L` and
/// `CONTROL_HEIGHT`, and the text sizes. The fonts come from
/// `assets::install_fonts`.
pub fn apply_style(ctx: &egui::Context) {
    use poster::*;
    let theme = Theme::get();
    ctx.set_theme(egui::Theme::Light);
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = Vec2::new(GAP_M, GAP_M);
    style.spacing.button_padding = Vec2::new(14.0, 7.0);
    style.spacing.interact_size = Vec2::new(40.0, CONTROL_HEIGHT);
    style.spacing.slider_width = 140.0;
    style.spacing.window_margin = egui::Margin::same(GAP_L as i8);
    style.spacing.menu_margin = egui::Margin::same(GAP_M as i8);
    let mut visuals = egui::Visuals::light();
    visuals.override_text_color = Some(theme.ink);
    visuals.weak_text_color = Some(theme.muted);
    visuals.panel_fill = theme.panel;
    visuals.window_fill = PANEL;
    visuals.faint_bg_color = Color32::from_rgba_premultiplied(40, 26, 14, 14);
    visuals.extreme_bg_color = CREAM;
    visuals.text_edit_bg_color = Some(CREAM);
    visuals.code_bg_color = Color32::from_rgb(240, 226, 190);
    visuals.hyperlink_color = theme.cold;
    visuals.warn_fg_color = theme.warn;
    visuals.error_fg_color = theme.danger;
    visuals.selection.bg_fill = MUSTARD;
    visuals.selection.stroke = Stroke::new(1.5, INK);
    visuals.slider_trailing_fill = true;
    visuals.widgets.noninteractive = widget(PANEL, Color32::from_rgb(150, 118, 84), 1.0, INK, 1.0);
    visuals.widgets.inactive = widget(CREAM, INK, 1.5, INK, 1.0);
    visuals.widgets.hovered = widget(Color32::from_rgb(255, 232, 170), INK, 2.0, INK, 1.5);
    visuals.widgets.active = widget(MUSTARD, INK, 2.0, INK, 2.0);
    visuals.widgets.open = widget(Color32::from_rgb(255, 232, 170), INK, 2.0, INK, 1.0);
    visuals.window_corner_radius = egui::CornerRadius::same(10);
    visuals.menu_corner_radius = egui::CornerRadius::same(8);
    visuals.window_stroke = Stroke::new(3.0, INK);
    visuals.window_shadow = egui::Shadow {
        offset: [4, 6],
        blur: 0,
        spread: 0,
        color: Color32::from_rgba_premultiplied(40, 26, 14, 90),
    };
    visuals.popup_shadow = egui::Shadow {
        offset: [3, 4],
        blur: 0,
        spread: 0,
        color: Color32::from_rgba_premultiplied(40, 26, 14, 80),
    };
    style.visuals = visuals;
    // Type scale in points: small print and code 14, body and buttons 16,
    // headings 25 in the bold HUD face.
    for (style_name, font) in [
        (egui::TextStyle::Small, FontId::proportional(14.0)),
        (egui::TextStyle::Body, FontId::proportional(16.0)),
        (egui::TextStyle::Button, FontId::proportional(16.0)),
        (egui::TextStyle::Monospace, FontId::monospace(14.0)),
        (
            egui::TextStyle::Heading,
            FontId::new(25.0, assets::hud_bold()),
        ),
    ] {
        style.text_styles.insert(style_name, font);
    }
    ctx.set_global_style(style);
}

/// A cheap deterministic hash of `n`, as a value in [0, 1). The world effects
/// in `world_fx.rs` use it to scatter their shapes, and the film grain uses it
/// to build and move its pattern.
pub fn hash(n: i64) -> f32 {
    let mut x = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x & 0xFFFF) as f32 / 65536.0
}

/// Paints `art`, an image that tiles, over `rect` as one textured quad. One
/// repeat covers `tile` points, and `offset` shifts the pattern by that many
/// repeats. `tint` is a premultiplied color that tints the image. An empty
/// `rect` paints nothing.
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

/// Paints the paper behind the whole window over `rect`: poster tan with 14
/// paler rays that fan out from a point above the middle, like a sunburst.
pub fn backdrop(painter: &egui::Painter, rect: Rect) {
    painter.rect_filled(rect, 0, poster::PAPER);
    let center = Pos2::new(rect.center().x, rect.top() + rect.height() * 0.35);
    let reach = rect.size().length();
    let rays = 28;
    for i in (0..rays).step_by(2) {
        let a0 = i as f32 / rays as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / rays as f32 * std::f32::consts::TAU;
        painter.add(egui::Shape::convex_polygon(
            vec![
                center,
                center + Vec2::new(a0.cos(), a0.sin()) * reach,
                center + Vec2::new(a1.cos(), a1.sin()) * reach,
            ],
            poster::PAPER_RAY,
            Stroke::NONE,
        ));
    }
}

/// Does nothing. The flat poster panels have no wear to paint. The side panel
/// in `ui.rs` still calls it, and every argument is unused.
pub fn wear(_painter: &egui::Painter, _rect: Rect, _theme: Theme, _seed: f32) {}

/// Paints a card in `rect`: a translucent drop shadow offset down and to the
/// right, the `fill` and a dark outline inside the rect. `lit` draws the
/// outline in the accent color and thicker.
pub fn plate(painter: &egui::Painter, rect: Rect, theme: Theme, fill: Color32, lit: bool) {
    painter.rect_filled(
        rect.translate(Vec2::new(3.0, 4.0)),
        7,
        Color32::from_rgba_premultiplied(40, 26, 14, 60),
    );
    painter.rect_filled(rect, 7, fill);
    painter.rect_stroke(
        rect,
        7,
        Stroke::new(
            if lit { 3.0 } else { 2.0 },
            if lit { theme.accent } else { theme.card_border },
        ),
        egui::StrokeKind::Inside,
    );
}

/// Paints a HUD box over the scene: dark glass with rounded corners and a thin
/// dark outline.
pub fn hud_panel(painter: &egui::Painter, rect: Rect) {
    painter.rect_filled(rect, 8, scene::HUD_BACK);
    painter.rect_stroke(
        rect,
        8,
        Stroke::new(1.5, Color32::from_rgba_premultiplied(0, 0, 0, 160)),
        egui::StrokeKind::Outside,
    );
}

/// Paints `galley` at `at` in `color`, with a glow of 20 faint copies in two
/// rings around it. `strength` scales the alpha of the glow, and 0 paints the
/// text alone.
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

/// Paints `text` in `font` and `color`, anchored at `pos` by `align`, with a
/// faint glow when `glow` is true. Returns the rect of the text.
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

/// Builds a layout job of `text` as bold capitals at `size` and `color`, with
/// extra letter spacing of 8% of the size.
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

/// Paints `text` as the letter-spaced capitals of `caps`, anchored at `pos` by
/// `align`. Returns the rect of the text.
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

/// A section title in the side panel: a full-width dark wood strip with a
/// mustard block and cream capitals. `_theme` is not used.
pub fn section(ui: &mut egui::Ui, text: &str, _theme: Theme) -> Response {
    strip(ui, text, true)
}

/// The same strip as `section`, only as wide as its text, so it can sit beside
/// other things. `_theme` is not used.
pub fn heading(ui: &mut egui::Ui, text: &str, _theme: Theme) -> Response {
    strip(ui, text, false)
}

/// The dark wood strip that `section` and `heading` draw. With `full` it is as
/// wide as the available space, and otherwise it is as wide as its text.
fn strip(ui: &mut egui::Ui, text: &str, full: bool) -> Response {
    let galley = ui.painter().layout_job(caps(text, 15.0, poster::CREAM));
    let width = if full {
        ui.available_width()
    } else {
        galley.size().x + 40.0
    };
    let size = Vec2::new(width, (galley.size().y + 10.0).max(30.0));
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    let p = ui.painter();
    p.rect_filled(
        rect.translate(Vec2::new(2.0, 3.0)),
        5,
        Color32::from_rgba_premultiplied(40, 26, 14, 60),
    );
    p.rect_filled(rect, 5, poster::WOOD_DARK);
    p.rect_filled(
        Rect::from_min_size(
            rect.left_top() + Vec2::new(8.0, rect.height() / 2.0 - 6.0),
            Vec2::splat(12.0),
        ),
        2,
        poster::MUSTARD,
    );
    p.galley(
        Pos2::new(rect.left() + 28.0, rect.center().y - galley.size().y / 2.0),
        galley,
        poster::CREAM,
    );
    response
}

/// One tab of the tab strip. A small dark square on its left shows the
/// shortcut `key`, and `text` follows it. The selected tab is mustard with a
/// shadow and a thicker outline, a hovered one is pale mustard, and the others
/// are cream. Returns the response, which reports clicks.
pub fn tab(ui: &mut egui::Ui, selected: bool, key: &str, text: &str, theme: Theme) -> Response {
    let font = FontId::new(21.0, assets::hud_bold());
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font, theme.ink);
    let key_galley = ui.painter().layout_no_wrap(
        key.to_owned(),
        FontId::new(13.0, assets::hud_bold()),
        poster::CREAM,
    );
    let cap = 20.0;
    let size = Vec2::new(galley.size().x + cap + 34.0, 40.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let hovered = response.hovered();
    let fill = if selected {
        poster::MUSTARD
    } else if hovered {
        Color32::from_rgb(255, 232, 170)
    } else {
        poster::CREAM
    };
    let p = ui.painter();
    if selected {
        p.rect_filled(
            rect.translate(Vec2::new(2.0, 3.0)),
            8,
            Color32::from_rgba_premultiplied(40, 26, 14, 70),
        );
    }
    p.rect(
        rect,
        8,
        fill,
        Stroke::new(if selected { 3.0 } else { 2.0 }, theme.ink),
        egui::StrokeKind::Inside,
    );
    let cap_rect = Rect::from_center_size(
        Pos2::new(rect.left() + 12.0 + cap / 2.0, rect.center().y),
        Vec2::splat(cap),
    );
    p.rect_filled(cap_rect, 4, poster::WOOD_DARK);
    p.galley(
        cap_rect.center() - key_galley.size() / 2.0,
        key_galley,
        poster::CREAM,
    );
    p.galley(
        Pos2::new(
            cap_rect.right() + 8.0,
            rect.center().y - galley.size().y / 2.0,
        ),
        galley,
        theme.ink,
    );
    response
}

/// Adds a band between the `outer` and `inner` rects to `mesh`, with
/// `outer_color` at the outer edge and `inner_color` at the inner one.
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

/// Darkens the edges of a scene with black that fades toward the middle.
/// `strength` is the alpha at the very edge, from 0 to 1.
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

/// The film-grain texture: 128 by 128 pixels of white and black specks of
/// varying strength. It is made once and kept in the context.
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

/// Paints film grain over `rect`. `alpha` is its strength from 0 to 1, and 0
/// paints nothing. `time` is the scene clock in seconds. The pattern jumps 24
/// times per second of it, so the grain lives while the scene plays and holds
/// still while it is paused.
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

/// One line of a HUD block. `hud_block` lays out a list of them.
pub struct HudLine {
    /// The words or digits of the line.
    pub text: String,
    /// The font size in points.
    pub size: f32,
    /// The text color.
    pub color: Color32,
    /// Bold letter-spaced capitals, for labels.
    pub caps: bool,
    /// The HUD digit face with its glow, for the numbers.
    pub glow: bool,
}

impl HudLine {
    /// A small capital label in dim HUD yellow.
    pub fn label(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            size: 12.5,
            color: scene::HUD_DIM,
            caps: true,
            glow: false,
        }
    }
    /// A line of glowing digits in the HUD digit face, at `size` and `color`.
    pub fn value(text: String, size: f32, color: Color32) -> Self {
        Self {
            text,
            size,
            color,
            caps: false,
            glow: true,
        }
    }
    /// A line of plain text in the interface font, at `size` and `color`.
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

/// Paints a HUD box that holds `lines` one under the other, anchored at
/// `anchor` by `align`. The text sits on the same side as the anchor: left for
/// a left anchor, centered for a center anchor and right for a right anchor.
/// Returns the rect of the box.
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

/// A Half-Life 2 counter: a smoked-glass box with its label low on the left and
/// big glowing digits beside it, like HEALTH and SUIT.
pub struct Counter<'a> {
    /// The caption on the left, painted as small capitals.
    pub label: &'a str,
    /// The number, in big glowing digits.
    pub digits: String,
    /// The unit after the digits, painted small. It may be empty.
    pub unit: &'a str,
    /// A second, smaller text right of the unit, like the reserve count of the
    /// ammo box.
    pub extra: Option<String>,
    /// Turns the counter red, as the HUD does when hit.
    pub damaged: bool,
}

/// Paints `counter` anchored at `anchor` by `align` and returns its rect.
/// `size` is the font size of the digits in points. The label, the unit and
/// the extra text scale with it.
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
        (size * 0.34).max(11.0),
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
