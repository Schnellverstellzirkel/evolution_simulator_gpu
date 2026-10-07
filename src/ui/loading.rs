//! `App::loading_screen` in `ui.rs` calls this module to draw the loading card
//! or the corner note, and both show the progress that `crate::loading`
//! reports. The card covers the window while the GPU opens and the kernels
//! compile, with a creature walking in place and a line to read. A start with
//! no cached kernels compiles about fifty of them, because every effect a
//! world has is compiled into its own kernel and the 100 wild islands run in
//! worlds of their own. The compiled kernels stay on disk and load in
//! milliseconds at the next start.

use crate::loading::{Group, Progress};
use crate::theme::{self, Theme, poster};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
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
    "Teaching 100 islands their own weather…",
    "Baking mud for the swamp islands…",
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
    "Removing the free energy from the solver…",
    "Adjusting the wind so it blows the wrong way…",
    "Evolving patience. Generation 1.",
    "Waiting for the heat wave to cool down…",
    "Unpacking a fresh supply of evolutionary pressure…",
    "Counting feet. Recounting feet.",
    "Tuning 60 Hz physics to 60 Hz exactly…",
    "Filling the archive's map with empty cells to conquer…",
    "Growing a skeleton from scratch, bone by bone…",
    "Negotiating with NVRTC over register budgets…",
    "Hurdles for island 31, gaps for island 32…",
    "Your future champions are stretching in the locker room…",
    "Each island gets a kernel. Each kernel gets a coffee…",
];
const MESSAGE_SECONDS: f32 = 4.0;

/// Why the window is waiting. `App::loading_screen` in `ui.rs` picks it.
pub(super) enum Wait<'a> {
    /// The evaluation devices are opening.
    Opening,
    /// The devices did not open. The string is the reason.
    Failed(&'a str),
    /// The kernels of the starting worlds are compiling.
    Starting,
    /// The first generation cannot start until the kernels of its world
    /// compile.
    World,
}

/// What `screen` draws: why the window waits, and when the game started.
pub(super) struct Card<'a> {
    /// Why the window is waiting. It sets the title, which progress shows and
    /// whether the player may close the card.
    pub wait: Wait<'a>,
    /// When the app started. The walker and the line of the moment count from
    /// here.
    pub since: Instant,
}

/// Draws the loading card over the whole window and returns true when the
/// player pressed the button that closes it. The card shows the progress of the
/// kernels that evolution waits for when `card.wait` is `Wait::World`, and the
/// progress of the starting jobs otherwise. It asks for a repaint every 33 ms,
/// so the walker keeps moving.
pub(super) fn screen(ctx: &egui::Context, theme: Theme, card: &Card) -> bool {
    let progress = match card.wait {
        Wait::World => crate::loading::progress(Group::Needed),
        _ => crate::loading::progress(Group::Startup),
    };
    let clock = card.since.elapsed().as_secs_f32();
    let screen = ctx.content_rect();
    let mut close = false;
    egui::Area::new(egui::Id::new("loading screen"))
        .order(egui::Order::Foreground)
        .fade_in(false)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            // Swallow clicks meant for the panels underneath.
            ui.allocate_rect(screen, Sense::click_and_drag());
            ui.painter().rect_filled(
                screen,
                0.0,
                Color32::from_rgba_premultiplied(40, 26, 14, 205),
            );
            let width = (screen.width() - 48.0).clamp(300.0, 640.0);
            let height = (screen.height() - 32.0).clamp(380.0, 600.0);
            let rect = Rect::from_center_size(screen.center(), Vec2::new(width, height));
            theme::plate(ui.painter(), rect, theme, theme.card, true);
            ui.scope_builder(egui::UiBuilder::new().max_rect(rect.shrink(22.0)), |ui| {
                close = body(ui, theme, card, &progress, clock)
            });
        });
    ctx.request_repaint_after(Duration::from_millis(33));
    close
}

/// Draws the inside of the card from top to bottom: the title, the walker, a
/// line to read, the progress bar, the jobs running now, a note and the button
/// that closes the card. `clock` is the seconds since `Card::since`. After a
/// failure the card stops at the error. Returns true when the player pressed
/// the button.
fn body(ui: &mut egui::Ui, theme: Theme, card: &Card, progress: &Progress, clock: f32) -> bool {
    let title = match card.wait {
        Wait::Opening => "Opening the laboratory",
        Wait::Failed(_) => "The laboratory could not open",
        Wait::Starting => "Building physics for every island",
        Wait::World => "Building physics for this world",
    };
    let galley = ui.painter().layout_job(theme::caps(title, 24.0, theme.ink));
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 34.0), Sense::hover());
    ui.painter().galley(
        rect.left_center() - Vec2::new(0.0, galley.size().y / 2.0),
        galley,
        theme.ink,
    );
    ui.add_space(6.0);
    walker(ui, theme, clock);
    ui.add_space(6.0);
    if let Wait::Failed(error) = card.wait {
        ui.label(RichText::new(error).color(theme.danger).size(15.0));
        return false;
    }
    // The line of the moment, fading in and out.
    let index = (clock / MESSAGE_SECONDS) as usize;
    let phase = (clock / MESSAGE_SECONDS).fract();
    let alpha = (phase * 6.0).min((1.0 - phase) * 6.0).clamp(0.0, 1.0);
    // TODO: 7 divides the 42 lines, so the index reaches only 6 of them. A step
    // that shares no factor with the length of the list would show them all.
    ui.label(
        RichText::new(MESSAGES[(index * 7 + 3) % MESSAGES.len()])
            .size(16.0)
            .italics()
            .color(theme.accent.gamma_multiply(alpha)),
    );
    ui.add_space(12.0);
    // The bar divides by the total, so it waits for the first job.
    let total = progress.total();
    if total > 0 {
        bar(ui, theme, progress, total);
        ui.add_space(8.0);
    }
    // Four running jobs at most. The rest are counted below.
    for (label, took) in progress.running.iter().take(4) {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(RichText::new(label).color(theme.ink));
            ui.label(RichText::new(format!("{:.0} s", took.as_secs_f32())).color(theme.muted));
        });
    }
    if progress.running.len() > 4 {
        ui.label(
            RichText::new(format!(
                "and {} more at the same time",
                progress.running.len() - 4
            ))
            .color(theme.muted),
        );
    }
    if progress.queued > 0 {
        ui.label(
            RichText::new(format!(
                "{} more waiting for a compiler thread",
                progress.queued
            ))
            .color(theme.muted),
        );
    }
    if let Some(left) = progress.seconds_left() {
        ui.label(RichText::new(format!("About {} left", duration_words(left))).color(theme.muted));
    }
    if let Some((label, took, cached)) = &progress.last {
        ui.label(
            RichText::new(if *cached {
                format!("Done: {label}, from the cache")
            } else {
                format!("Done: {label} in {:.1} s", took.as_secs_f32())
            })
            .small()
            .color(theme.muted),
        );
    }
    ui.add_space(10.0);
    ui.label(
        RichText::new(match card.wait {
            Wait::Opening => {
                "The GPU opens on its own thread. Anything you click now waits for it."
            }
            _ => {
                "Every effect a world has is compiled into its own kernel, and each of the 100 \
                 islands has a world of its own. They are compiled once for this GPU and kept on \
                 disk, so the next start takes a moment."
            }
        })
        .size(13.0)
        .color(theme.muted),
    );
    let mut close = false;
    if !matches!(card.wait, Wait::Opening) {
        ui.add_space(10.0);
        close = ui
            .add(
                egui::Button::new(RichText::new("Look around while it finishes").size(15.0))
                    .min_size(Vec2::new(0.0, theme::CONTROL_HEIGHT)),
            )
            .on_hover_text(
                "Compiling goes on in the background. Evolution starts when the kernels it needs are ready.",
            )
            .clicked();
    }
    close
}

/// Draws the progress bar: a cream track with a dark outline, the finished
/// share in grass green, and over it the number of finished jobs and the time
/// since the first job. `total` is `progress.total()` and must be above 0.
fn bar(ui: &mut egui::Ui, theme: Theme, progress: &Progress, total: usize) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 6, poster::CREAM);
    let share = progress.done as f32 / total as f32;
    let mut fill = rect.shrink(2.0);
    fill.set_width(fill.width() * share);
    painter.rect_filled(fill, 4, theme.go_fill);
    painter.rect_stroke(
        rect,
        6,
        Stroke::new(2.0, theme.card_border),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        format!(
            "{} of {total} ready · {}",
            progress.done,
            duration_words(progress.elapsed.as_secs_f32())
        ),
        FontId::proportional(15.0),
        theme.ink,
    );
}

/// Words for a time: "45 s" under a minute, else "3 min 05 s". It rounds to the
/// second and reads a negative time as "0 s".
fn duration_words(seconds: f32) -> String {
    let s = seconds.max(0.0).round() as u32;
    if s < 60 {
        format!("{s} s")
    } else {
        format!("{} min {:02} s", s / 60, s % 60)
    }
}

/// Draws a creature walking in place on a ground that slides under it: bones in
/// dark brown, muscles in brick red that swell as they pull, tan joints and a
/// mustard head. The corner counts the meters it has walked, at 100 points to
/// the meter. `clock` is the seconds since `Card::since`.
fn walker(ui: &mut egui::Ui, theme: Theme, clock: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 96.0), Sense::hover());
    let painter = ui.painter_at(rect);
    let ground = rect.bottom() - 12.0;
    painter.line_segment(
        [
            Pos2::new(rect.left(), ground),
            Pos2::new(rect.right(), ground),
        ],
        Stroke::new(3.0, theme.ink),
    );
    // Tick marks that slide left, so the creature appears to walk right.
    let speed = 44.0;
    let spacing = 32.0;
    let offset = (clock * speed) % spacing;
    let mut x = rect.left() - offset + spacing;
    while x < rect.right() {
        painter.line_segment(
            [Pos2::new(x, ground), Pos2::new(x - 7.0, ground + 7.0)],
            Stroke::new(1.5, theme.muted),
        );
        x += spacing;
    }
    let center = rect.center().x;
    // One turn of `stride` is one cycle of the gait, 1.2 cycles a second. The
    // body bobs four times in a cycle.
    let stride = clock * std::f32::consts::TAU * 1.2;
    let bob = (stride * 2.0).sin().abs() * 3.0;
    let hip = Pos2::new(center - 32.0, ground - 36.0 - bob);
    let shoulder = Pos2::new(center + 28.0, ground - 40.0 - bob);
    let head = Pos2::new(center + 50.0, ground - 60.0 - bob);
    // A foot swings 17 points to each side of `base` and lifts up to 9 points
    // while it swings forward.
    let foot = |base: Pos2, phase: f32| {
        let swing = (stride + phase).sin();
        let lift = (stride + phase).cos().max(0.0) * 9.0;
        Pos2::new(base.x + swing * 17.0, ground - lift)
    };
    let back = foot(hip, 0.0);
    let front = foot(shoulder, std::f32::consts::PI);
    // A muscle from `a` to `b`. It thickens and gets more opaque as it pulls.
    let muscle = |a: Pos2, b: Pos2, phase: f32| {
        let pull = 0.5 + 0.5 * (stride + phase).sin();
        painter.line_segment(
            [a, b],
            Stroke::new(
                2.5 + 3.0 * pull,
                poster::RED.gamma_multiply(0.45 + 0.55 * pull),
            ),
        );
    };
    muscle(hip.lerp(shoulder, 0.5), back, 0.0);
    muscle(shoulder.lerp(head, 0.5), front, std::f32::consts::PI);
    for [a, b] in [
        [hip, shoulder],
        [shoulder, head],
        [hip, back],
        [shoulder, front],
    ] {
        painter.line_segment([a, b], Stroke::new(5.0, theme.ink));
    }
    for p in [hip, shoulder, back, front] {
        painter.circle_filled(p, 5.5, poster::TAN);
        painter.circle_stroke(p, 5.5, Stroke::new(2.0, theme.ink));
    }
    painter.circle_filled(head, 9.0, poster::MUSTARD);
    painter.circle_stroke(head, 9.0, Stroke::new(2.0, theme.ink));
    painter.circle_filled(head + Vec2::new(3.0, -2.0), 2.0, theme.ink);
    painter.text(
        Pos2::new(rect.right() - 2.0, rect.top() + 2.0),
        Align2::RIGHT_TOP,
        format!("{:.1} m", clock * speed / 100.0),
        FontId::proportional(13.0),
        theme.muted,
    );
}

/// Draws a small note in the corner while kernels compile and the card is not
/// up. These are the kernels that evolution needs, such as those of a world
/// change, and with `include_startup` the starting ones too, as after the
/// player closed the card. The note names one running job with its time and
/// counts the other jobs, running or waiting. It draws nothing while no job
/// runs.
pub(super) fn toast(ctx: &egui::Context, theme: Theme, include_startup: bool) {
    let mut progress = crate::loading::progress(Group::Needed);
    if include_startup {
        let startup = crate::loading::progress(Group::Startup);
        progress.running.extend(startup.running);
        progress.queued += startup.queued;
        progress.done += startup.done;
    }
    let Some((label, took)) = progress.running.first() else {
        return;
    };
    let more = progress.running.len() - 1 + progress.queued;
    egui::Area::new(egui::Id::new("loading toast"))
        .order(egui::Order::Foreground)
        .anchor(Align2::RIGHT_BOTTOM, Vec2::new(-16.0, -48.0))
        .interactable(false)
        .fade_in(false)
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(theme.card)
                .stroke(Stroke::new(2.0, theme.card_border))
                .corner_radius(7)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_read_in_seconds_then_minutes() {
        assert_eq!(duration_words(4.4), "4 s");
        assert_eq!(duration_words(59.6), "1 min 00 s");
        assert_eq!(duration_words(185.0), "3 min 05 s");
        assert_eq!(duration_words(-3.0), "0 s");
    }

    #[test]
    fn the_message_index_stays_inside_the_list() {
        for index in 0..10_000usize {
            assert!(MESSAGES[(index * 7 + 3) % MESSAGES.len()].len() > 10);
        }
    }
}
