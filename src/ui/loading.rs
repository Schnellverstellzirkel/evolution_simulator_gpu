//! The loading screen: what the game is waiting for while devices open and
//! GPU kernels compile, with a creature walking in place and a few lines to
//! read meanwhile. The progress comes from `crate::loading`.

use super::Theme;
use crate::loading::Progress;
use eframe::egui::{self, Align2, Color32, FontId, Pos2, RichText, Sense, Stroke, Vec2};
use std::time::{Duration, Instant};

/// Lines shown while waiting, a new one every `MESSAGE_SECONDS`.
const MESSAGES: &[&str] = &[
    "Teaching bones which end is up…",
    "Untangling muscles that were wired backwards…",
    "Convincing the GPU that legs are a good idea…",
    "Sharpening the ground so feet have something to push on…",
    "Counting to 60, 1,200 times per creature…",
    "Reminding gravity to show up for work…",
    "Warming up three million tiny hearts…",
    "Translating physics into GPU dialect…",
    "Filing the rough edges off friction…",
    "Asking the compiler nicely to unroll the joints…",
    "Placing the finish line very, very far away…",
    "Hiding the sled exploit where nobody will find it…",
    "Giving every creature exactly one head. Mostly.",
    "Stretching tendons before the big race…",
    "Rolling dice for the first generation's body plans…",
    "Laying out 4 islands for the archive to live on…",
    "Polishing the ice patches. Carefully.",
    "Checking that muscles only pull. They only pull.",
    "Pre-chewing the contact solver…",
    "Charging muscle energy stores to 100%…",
    "Bribing the random number generator for good mutations…",
    "Drawing a ruler on the ground, in meters…",
    "Teaching the replay camera to keep up…",
    "Making sure nobody can walk by vibrating. Again.",
    "Assembling registers, one warp at a time…",
    "Giving the kernel a pep talk about occupancy…",
    "Measuring twice, simulating three million times…",
    "Sorting creatures by size so warps stay busy…",
    "Removing the free energy from the rebuild step…",
    "Adjusting the wind so it blows the wrong way…",
    "Evolving patience. Generation 1.",
    "Waiting for the heat wave to cool down…",
    "Unpacking a fresh supply of evolutionary pressure…",
    "Counting feet. Recounting feet.",
    "Tuning 60 Hz physics to 60 Hz exactly…",
    "Filling the archive's map with empty cells to conquer…",
    "Growing a skeleton from scratch, bone by bone…",
    "Negotiating with NVRTC over register budgets…",
    "Checking that the ground is still flat. Mostly flat.",
    "Your future champions are stretching in the locker room…",
];
const MESSAGE_SECONDS: f32 = 4.0;

/// Why the window is waiting.
pub(super) enum Wait<'a> {
    /// The evaluation devices are opening.
    Opening,
    /// They could not open.
    Failed(&'a str),
    /// A generation cannot start until kernels compile.
    Compiling,
}

/// Draws the full loading screen over the window.
pub(super) fn screen(ctx: &egui::Context, theme: &Theme, wait: Wait, since: Instant) {
    let progress = crate::loading::progress();
    let clock = since.elapsed().as_secs_f32();
    let screen = ctx.content_rect();
    egui::Area::new(egui::Id::new("loading screen"))
        .order(egui::Order::Foreground)
        .fade_in(false)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            // Swallow clicks meant for the panels underneath.
            ui.allocate_rect(screen, Sense::click_and_drag());
            ui.painter()
                .rect_filled(screen, 0.0, theme.canvas.gamma_multiply(0.96));
            let width = (screen.width() - 64.0).clamp(280.0, 620.0);
            let card = egui::Rect::from_center_size(screen.center(), Vec2::new(width, 470.0));
            ui.scope_builder(egui::UiBuilder::new().max_rect(card), |ui| {
                egui::Frame::new()
                    .fill(theme.card)
                    .stroke(Stroke::new(1.0, theme.card_border))
                    .corner_radius(12.0)
                    .inner_margin(24.0)
                    .show(ui, |ui| {
                        ui.set_width(width - 48.0);
                        body(ui, theme, &wait, &progress, clock);
                    });
            });
        });
    ctx.request_repaint_after(Duration::from_millis(33));
}

fn body(ui: &mut egui::Ui, theme: &Theme, wait: &Wait, progress: &Progress, clock: f32) {
    let title = match wait {
        Wait::Opening => "Opening the laboratory",
        Wait::Failed(_) => "The laboratory could not open",
        Wait::Compiling => "Building the physics for your GPU",
    };
    ui.label(RichText::new(title).size(24.0).strong().color(theme.ink));
    ui.add_space(8.0);
    walker(ui, theme, clock);
    ui.add_space(8.0);
    if let Wait::Failed(error) = wait {
        ui.label(RichText::new(*error).color(theme.danger));
        return;
    }
    // The line of the moment, fading in and out.
    let index = (clock / MESSAGE_SECONDS) as usize;
    let phase = (clock / MESSAGE_SECONDS).fract();
    let alpha = (phase * 6.0).min((1.0 - phase) * 6.0).clamp(0.0, 1.0);
    let message = MESSAGES[(index * 7 + 3) % MESSAGES.len()];
    ui.label(
        RichText::new(message)
            .size(16.0)
            .italics()
            .color(theme.accent.gamma_multiply(alpha)),
    );
    ui.add_space(14.0);
    let total = progress.done + progress.running.len() + progress.queued.len();
    if total > 0 {
        let fraction = progress.done as f32 / total as f32;
        ui.add(egui::ProgressBar::new(fraction).text(format!(
            "{} of {total} ready · {:.0} s",
            progress.done,
            progress.elapsed.as_secs_f32()
        )));
        ui.add_space(8.0);
    }
    for (label, took) in progress.running.iter().take(4) {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(RichText::new(label).color(theme.ink));
            ui.label(RichText::new(format!("{:.0} s", took.as_secs_f32())).color(theme.muted));
        });
    }
    if !progress.queued.is_empty() {
        ui.label(
            RichText::new(format!(
                "{} more waiting for a compiler thread",
                progress.queued.len()
            ))
            .color(theme.muted),
        );
    }
    if let Some(left) = estimate(progress) {
        ui.label(RichText::new(format!("About {left:.0} s left")).color(theme.muted));
    }
    if let Some((label, took, cached)) = &progress.last {
        ui.label(
            RichText::new(if *cached {
                format!("Done: {label}, loaded from the cache")
            } else {
                format!("Done: {label} in {:.1} s", took.as_secs_f32())
            })
            .color(theme.muted),
        );
    }
    ui.add_space(10.0);
    ui.label(
        RichText::new(match wait {
            Wait::Opening => {
                "The evaluation devices open on their own thread. Commands you give now wait for them."
            }
            _ if progress.running.iter().any(|(l, _)| l.starts_with("CUDA")) => {
                "Each body size gets its own kernel, compiled once for this GPU and then kept on \
                 disk, so the next start loads it in milliseconds. Larger kernels compile in the \
                 background while evolution runs."
            }
            _ => {
                "Each body size gets its own kernel, which the graphics driver compiles for this \
                 GPU when it is first needed. Larger kernels are built when bodies grow."
            }
        })
        .size(13.0)
        .color(theme.muted),
    );
}

/// Seconds until the running and queued jobs finish, from the compiles seen
/// so far, spread over the three compiler threads.
fn estimate(progress: &Progress) -> Option<f32> {
    if progress.built_seconds.is_empty() || !progress.busy() {
        return None;
    }
    let mean = progress.built_seconds.iter().sum::<f32>() / progress.built_seconds.len() as f32;
    let running: f32 = progress
        .running
        .iter()
        .map(|(_, t)| (mean - t.as_secs_f32()).max(0.5))
        .sum();
    let queued = progress.queued.len() as f32 * mean;
    Some((running + queued) / progress.running.len().clamp(1, 3) as f32)
}

/// A two-legged creature walking in place: a body bone, two legs, and a
/// muscle each, on a ground line that scrolls under it.
fn walker(ui: &mut egui::Ui, theme: &Theme, clock: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 90.0), Sense::hover());
    let painter = ui.painter_at(rect);
    let ground = rect.bottom() - 10.0;
    let stroke = Stroke::new(2.0, theme.muted);
    painter.line_segment(
        [
            Pos2::new(rect.left(), ground),
            Pos2::new(rect.right(), ground),
        ],
        stroke,
    );
    // Tick marks that slide left, so the creature appears to walk right.
    let speed = 40.0;
    let spacing = 30.0;
    let offset = (clock * speed) % spacing;
    let mut x = rect.left() - offset + spacing;
    while x < rect.right() {
        painter.line_segment(
            [Pos2::new(x, ground), Pos2::new(x - 6.0, ground + 6.0)],
            Stroke::new(1.0, theme.muted),
        );
        x += spacing;
    }
    let center = rect.center().x;
    let stride = clock * std::f32::consts::TAU * 1.2;
    let bob = (stride * 2.0).sin().abs() * 3.0;
    let hip = Pos2::new(center - 32.0, ground - 34.0 - bob);
    let shoulder = Pos2::new(center + 28.0, ground - 38.0 - bob);
    let head = Pos2::new(center + 50.0, ground - 56.0 - bob);
    let foot = |base: Pos2, phase: f32| {
        let swing = (stride + phase).sin();
        let lift = (stride + phase).cos().max(0.0) * 8.0;
        Pos2::new(base.x + swing * 16.0, ground - lift)
    };
    let back = foot(hip, 0.0);
    let front = foot(shoulder, std::f32::consts::PI);
    let bone = Stroke::new(4.0, theme.ink);
    let muscle = |a: Pos2, b: Pos2, phase: f32| {
        let pull = 0.5 + 0.5 * (stride + phase).sin();
        let color = Color32::from_rgb(200, 70, 60).gamma_multiply(0.4 + 0.6 * pull);
        painter.line_segment([a, b], Stroke::new(2.0 + 2.0 * pull, color));
    };
    muscle(
        Pos2::new((hip.x + shoulder.x) / 2.0, (hip.y + shoulder.y) / 2.0),
        back,
        0.0,
    );
    muscle(shoulder.lerp(head, 0.5), front, std::f32::consts::PI);
    for [a, b] in [
        [hip, shoulder],
        [shoulder, head],
        [hip, back],
        [shoulder, front],
    ] {
        painter.line_segment([a, b], bone);
    }
    for p in [hip, shoulder, back, front] {
        painter.circle_filled(p, 4.0, theme.ink);
    }
    painter.circle_filled(head, 7.0, theme.accent);
    painter.text(
        Pos2::new(rect.right() - 4.0, rect.top() + 4.0),
        Align2::RIGHT_TOP,
        format!("{:.1} m", clock * speed / 100.0),
        FontId::proportional(12.0),
        theme.muted,
    );
}

/// A small corner note while kernels compile in the background and nothing
/// waits for them.
pub(super) fn toast(ctx: &egui::Context, theme: &Theme) {
    let progress = crate::loading::progress();
    let Some((label, took)) = progress.running.first() else {
        return;
    };
    egui::Area::new(egui::Id::new("loading toast"))
        .order(egui::Order::Foreground)
        .anchor(Align2::RIGHT_BOTTOM, Vec2::new(-16.0, -16.0))
        .interactable(false)
        .fade_in(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(theme.card)
                .stroke(Stroke::new(1.0, theme.card_border))
                .corner_radius(8.0)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        let more = progress.running.len() - 1 + progress.queued.len();
                        ui.label(
                            RichText::new(format!(
                                "Compiling in the background: {label}, {:.0} s{}",
                                took.as_secs_f32(),
                                if more > 0 {
                                    format!(" (+{more} more)")
                                } else {
                                    String::new()
                                }
                            ))
                            .color(theme.muted),
                        );
                    });
                });
        });
    ctx.request_repaint_after(Duration::from_millis(250));
}
