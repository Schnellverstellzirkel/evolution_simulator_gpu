//! Small widgets and color helpers that the tabs, the scene painter and the GIF
//! export share. `Choices` adds choice buttons to `egui::Ui`. `color_dot` draws
//! a small dot and `speed_picker` draws the playback speed menu. `heat_color`
//! colors the archive map, `species_color` colors the body types and
//! `mix_color` blends two colors.

use eframe::egui::{self, Color32, Sense, Vec2};

/// Choice buttons on `egui::Ui` that keep their frame, so every option reads
/// as a button and the chosen one is filled.
pub(super) trait Choices {
    /// Draws a button that looks selected when `selected` is true.
    fn pick(&mut self, selected: bool, text: impl Into<egui::WidgetText>) -> egui::Response;
    /// Draws a button for `value`. It looks selected when `current` equals
    /// `value`, and a click sets `current` to `value`.
    fn choice<T: PartialEq>(
        &mut self,
        current: &mut T,
        value: T,
        text: impl Into<egui::WidgetText>,
    ) -> egui::Response;
}
impl Choices for egui::Ui {
    fn pick(&mut self, selected: bool, text: impl Into<egui::WidgetText>) -> egui::Response {
        self.add(egui::Button::new(text).selected(selected))
    }
    fn choice<T: PartialEq>(
        &mut self,
        current: &mut T,
        value: T,
        text: impl Into<egui::WidgetText>,
    ) -> egui::Response {
        let response = self.pick(*current == value, text);
        if response.clicked() {
            *current = value;
        }
        response
    }
}
/// Cold-to-hot color for a map value `t` from 0 to 1. A `t` outside that range
/// gives the nearest end color.
pub(super) fn heat_color(t: f32) -> Color32 {
    // Cold steel blue through amber to hot rust.
    let cold = Color32::from_rgb(52, 84, 110);
    let mid = Color32::from_rgb(226, 170, 64);
    let hot = Color32::from_rgb(196, 70, 40);
    if t < 0.5 {
        mix_color(cold, mid, t * 2.0)
    } else {
        mix_color(mid, hot, (t - 0.5) * 2.0)
    }
}
/// Draws a filled dot of `color`, 3 points in radius, in a 10 by 10 point cell
/// of `ui`.
pub(super) fn color_dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.), Sense::hover());
    ui.painter().circle_filled(rect.center(), 3., color);
}
/// A compact menu that sets the playback speed `speed` to 0.25, 0.5, 1, 2 or 4
/// times. It is the height of a button, so a row of replay controls stays one
/// line. `id` names the menu in egui, so two pickers in one `ui` need different
/// ids.
pub(super) fn speed_picker(ui: &mut egui::Ui, speed: &mut f32, id: &str) {
    let label = |value: f32| {
        if value < 1.0 {
            format!("Speed {value}×")
        } else {
            format!("Speed {value:.0}×")
        }
    };
    egui::ComboBox::from_id_salt(id)
        .selected_text(label(*speed))
        .show_ui(ui, |ui| {
            for value in [0.25, 0.5, 1.0, 2.0, 4.0] {
                ui.selectable_value(speed, value, label(value));
            }
        })
        .response
        .on_hover_text("Playback speed of the replay");
}
/// The color of the body type with `n` nodes and `m` muscles. It depends only
/// on the two counts, so a body type has the same color in the chart and in the
/// list.
pub(super) fn species_color(n: usize, m: usize) -> Color32 {
    // Muted hues, like paint on old machinery. Stepping the hue by the golden
    // ratio conjugate (0.618034) for each whole number in `n * 257 + m` spreads
    // the body types over the color wheel.
    egui::ecolor::HsvaGamma {
        h: ((n * 257 + m) as f32 * 0.618034).fract(),
        s: 0.45,
        v: 0.72,
        a: 1.,
    }
    .into()
}
/// Blends two colors byte by byte. `t` is clamped to the range 0 to 1, so 0
/// gives `a` and 1 gives `b`. The alpha of `a` and `b` is ignored and the result
/// is opaque.
pub(super) fn mix_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}
