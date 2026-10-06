mod loading;
mod playback;
mod scene;
#[cfg(test)]
mod test_support;
mod text;
mod widgets;

use crate::{
    config::Config,
    evolution::{Creature, FAILED},
    physics::{self, Node},
    storage::Stats,
    theme::{
        GAP_L, GAP_M, Theme, apply_style,
        scene::{
            BONE, EYE, FALLEN, GROUND_EDGE, GROUND_INK, GROUND_TOP, MUSCLE_ACTIVE, MUSCLE_REST,
            ORGAN, OUTLINE, SKY_HORIZON, SKY_TOP, TOUCHDOWN,
        },
    },
    worker::{Command, EventKind, Snapshot, Worker},
};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Legend, Line, Plot, Points, VLine};
use image::{
    Delay as GifDelay, Frame as GifFrame, Rgba, RgbaImage,
    codecs::gif::{GifEncoder, Repeat as GifRepeat},
};
use playback::{FrameMarks, Playback, broken_nodes, node_contact};
pub(crate) use scene::thumbnail;
use scene::{draw_creature, node_color};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, atomic::Ordering, mpsc},
    time::{Duration, Instant},
};
use text::{ago, file_size, number, seconds_text, species_name};
use widgets::{Choices, color_dot, heat_color, mix_color, species_color, speed_picker};
const DEFAULT_CAMERA_ZOOM: f32 = 80.0;
/// Share of the viewport height under the ground line, room for the HUD.
const GROUND_SHARE: f32 = 0.25;
/// Share of the viewport height a creature fills at the default zoom.
const FIT_HEIGHT_SHARE: f32 = 0.42;
/// Pixels per meter of the default zoom: the creature's height fills
/// `FIT_HEIGHT_SHARE` of the viewport, clamped for tiny and huge bodies.
fn fit_zoom(body_height: f32, view_height: f32) -> f32 {
    (FIT_HEIGHT_SHARE * view_height / body_height.max(0.05)).clamp(40.0, 450.0)
}
/// The player's default zoom: the typical height fills its share of the
/// viewport, and the highest point of the recording stays in view (the ground
/// sits a quarter up from the bottom) unless that would shrink the body to less than
/// 60% of the typical fit. A creature that leaps far higher than it stands
/// keeps that 60% and clips its peak instead of becoming tiny.
fn player_zoom(height: f32, peak: f32, view_height: f32) -> f32 {
    let typical = fit_zoom(height, view_height);
    let whole = 0.69 * view_height / peak.max(0.05);
    whole.min(typical).max(typical * 0.6).clamp(20.0, 450.0)
}
/// The spacing scale: every gap, margin and padding is one of these.
const GAP_S: f32 = 4.0;
/// Height of an effect's level buttons: every effect row is this tall.
const LEVEL_HEIGHT: f32 = 30.0;
/// How long the UI's own messages hold the status line.
const MESSAGE_SECONDS: f32 = 8.0;
/// Generations between autosaves when the player turns autosave on, and in
/// an unattended run.
pub(crate) const AUTOSAVE_INTERVAL: u32 = 10;
/// Exported GIFs render the same scene as the viewport into this frame size.
const GIF_WIDTH: u32 = 400;
const GIF_HEIGHT: u32 = 224;
/// Most frames an exported GIF keeps; a longer trial is sampled evenly.
const GIF_MAX_FRAMES: usize = 360;
/// Pixels per meter cap, so a tiny creature stays in frame whole.
const GIF_MAX_SCALE: f32 = 200.0;
/// Marker carried by a screenshot request, so its reply can be told apart
/// from the benchmark capture.
struct ScreenshotRequest;
/// How long between refreshes of the runs/ disk usage.
const RUNS_REFRESH: Duration = Duration::from_secs(5);
/// Total size of the files under a directory, ignoring unreadable entries.
fn directory_bytes(root: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(entry.path());
            } else {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}
fn save_screenshot(capture: &egui::ColorImage, dir: &std::path::Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let path = dir.join(format!("screenshot-{stamp}.png"));
    let bytes: Vec<u8> = capture.pixels.iter().flat_map(|p| p.to_array()).collect();
    image::save_buffer(
        &path,
        &bytes,
        capture.size[0] as u32,
        capture.size[1] as u32,
        image::ColorType::Rgba8,
    )?;
    Ok(path)
}
pub fn launch(adapter_name: &str) -> anyhow::Result<()> {
    // The UI thread asks for a 1 ms slice, so a frame preempts the breeding
    // threads when it wakes (`threads::short_slice`).
    crate::threads::short_slice();
    let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = wgpu::Backends::VULKAN;
    // Render the UI on the GPU the desktop compositor uses: frames then need no
    // cross-GPU import, and when that is the integrated GPU the discrete GPU is
    // left entirely to evolution. EVOLUTION_RENDER_GPU selects an adapter by name.
    let render_name = std::env::var("EVOLUTION_RENDER_GPU")
        .ok()
        .map(|name| name.to_lowercase());
    let compositor = compositor_vendor();
    let name = adapter_name.to_lowercase();
    setup.native_adapter_selector = Some(Arc::new(move |adapters, surface| {
        let presentable = |a: &&wgpu::Adapter| surface.is_none_or(|s| a.is_surface_supported(s));
        let named = |wanted: &str| {
            adapters
                .iter()
                .filter(presentable)
                .find(|a| a.get_info().name.to_lowercase().contains(wanted))
                .cloned()
        };
        render_name
            .as_deref()
            .and_then(named)
            .or_else(|| {
                compositor.and_then(|vendor| {
                    adapters
                        .iter()
                        .filter(presentable)
                        .find(|a| a.get_info().vendor == vendor)
                        .cloned()
                })
            })
            .or_else(|| named(&name))
            .or_else(|| adapters.iter().find(presentable).cloned())
            .ok_or_else(|| format!("No presentation-capable Vulkan GPU matching {name}"))
    }));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([900.0, 620.0])
            .with_title(
                // Agents take screenshots in real windows on the owner's
                // desktop; the title says so.
                if std::env::vars_os()
                    .any(|(key, _)| key.to_string_lossy().starts_with("EVOLUTION_SMOKE_"))
                {
                    "exploraMove: agent screenshot run, not your game"
                } else {
                    "exploraMove"
                },
            ),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: setup.into(),
            ..Default::default()
        },
        ..Default::default()
    };
    let compute_name = adapter_name.to_owned();
    eframe::run_native(
        "Evolution Laboratory",
        options,
        Box::new(|cc| {
            // Evaluation opens its own CUDA devices on the worker's thread
            // while the window shows its loading screen; the render device
            // only draws.
            let worker = Worker::open(compute_name, cc.egui_ctx.clone());
            Ok(Box::new(App::new(cc, worker)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
/// The body classes of the cells the archive tab shows: the global archive's.
const CLASSES: &crate::qd::Classes = &crate::qd::GLOBAL_CLASSES;
/// Movement-axis bin counts, mirroring `qd::MOVEMENT_BINS` (ground contact, cadence,
/// shape, height, feet). The shape and the size classes are filters.
const MAP_BINS: [usize; 5] = [6, 8, 1, 6, 5];
/// Which representation the Behavior archive tab shows.
#[derive(Clone, Copy, PartialEq)]
enum ArchiveView {
    Cards,
    Map,
    Islands,
}
/// Which archive cards the player looks at.
#[derive(Clone, Copy, Default, PartialEq)]
struct CardFilter {
    /// Feet bin (0 is one foot, 4 is five or more), or every count.
    feet: Option<u8>,
    /// Body size class (`qd::SIZE_NAMES`), or every size.
    size: Option<u8>,
    /// Body shape class (`qd::SHAPE_NAMES`), or every shape.
    shape: Option<u8>,
}
impl CardFilter {
    fn shows(self, card: &crate::worker::Card) -> bool {
        let niche = card.descriptor.map(|d| d.niche().0);
        self.feet
            .is_none_or(|feet| niche.is_some_and(|n| n[4] == feet))
            && self
                .size
                .is_none_or(|size| niche.is_some_and(|n| n[5] == size))
            && self
                .shape
                .is_none_or(|shape| niche.is_some_and(|n| n[2] == shape))
    }
}
/// One archive elite running in the race view.
struct RaceLane {
    /// Why the creature runs: its archive rank, "champion" or "your pick".
    label: String,
    playback: Playback,
}
/// Creatures the player sends to the race with "Race it", at most this many
/// beside the champion.
const RACE_PICKS: usize = 4;
/// Height range of one archive height bin; mirrors `qd::height_axis`.
fn height_bin_range(bin: usize) -> (f32, f32) {
    let low = 0.15f32;
    let high = (0.6 * crate::evolution::max_bone_length()).max(2.0 * low);
    let at = |t: f32| low * (high / low).powf(t);
    (at(bin as f32 / 6.0), at((bin + 1) as f32 / 6.0))
}
fn height_bin_label(bin: usize) -> String {
    let (low, high) = height_bin_range(bin);
    format!("{low:.2} to {high:.2} m")
}
fn feet_bin_label(bin: usize) -> String {
    match bin {
        0 => "1 foot".to_owned(),
        1..=3 => format!("{} feet", bin + 1),
        _ => "5+ feet".to_owned(),
    }
}
fn body_counts(creature: &Creature) -> (usize, usize, usize) {
    (
        creature.nodes.len(),
        creature.bones.len(),
        creature.muscles.len(),
    )
}
/// An ancestor whose node, bone or muscle count differs from its parent's.
fn body_plan_changed(
    step: &crate::worker::LineageStep,
    parent: Option<&crate::worker::LineageStep>,
) -> bool {
    parent.is_some_and(|parent| body_counts(&step.creature) != body_counts(&parent.creature))
}
/// Validates an imported creature JSON before it reaches the replay engine.
/// Rejects bodies the physics cannot step, with a short reason.
fn imported_creature(creature: &mut Creature) -> Result<(), String> {
    let nodes = creature.nodes.len();
    if !(3..=64).contains(&nodes) || creature.bones.len() + 1 != nodes {
        return Err("expected a connected body with one bone per extra node".into());
    }
    if !crate::evolution::canonicalize_bone_order(creature) {
        return Err("the body plan is not a connected tree".into());
    }
    if creature.nodes.iter().any(|n| {
        ![n.x, n.y, n.diameter, n.friction]
            .iter()
            .all(|v| v.is_finite())
    }) {
        return Err("the file has non-finite node values".into());
    }
    if creature.bones.iter().any(|b| {
        ![
            b.rest_length,
            b.min_angle,
            b.max_angle,
            b.organ_mass,
            b.organ_at,
        ]
        .iter()
        .all(|v| v.is_finite())
    }) {
        return Err("the file has non-finite bone values".into());
    }
    if creature.muscles.iter().any(|m| {
        m.bone_a as usize >= creature.bones.len()
            || m.bone_b as usize >= creature.bones.len()
            || ![m.short, m.long, m.period, m.phase, m.duty, m.stiffness]
                .iter()
                .all(|v| v.is_finite())
    }) {
        return Err("the file has invalid muscles".into());
    }
    Ok(())
}
/// Scales a plot range around its center. A factor below one zooms in.
fn scaled_range(
    range: std::ops::RangeInclusive<f64>,
    factor: f64,
) -> std::ops::RangeInclusive<f64> {
    let center = (range.start() + range.end()) / 2.0;
    let half = (range.end() - range.start()) / 2.0 * factor;
    (center - half)..=(center + half)
}
/// One save in runs/, for File > Open.
struct SaveEntry {
    path: PathBuf,
    modified: Option<std::time::SystemTime>,
    bytes: u64,
    summary: Option<crate::storage::SaveSummary>,
}
/// Every .evo file directly in `dir`, newest first, with what its first
/// bytes say about it.
fn list_saves(dir: &std::path::Path) -> Vec<SaveEntry> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut saves: Vec<SaveEntry> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "evo"))
        .map(|path| {
            let metadata = std::fs::metadata(&path).ok();
            SaveEntry {
                modified: metadata.as_ref().and_then(|m| m.modified().ok()),
                bytes: metadata.map_or(0, |m| m.len()),
                summary: crate::storage::summary(&path),
                path,
            }
        })
        .collect();
    saves.sort_by_key(|save| std::cmp::Reverse(save.modified));
    saves
}
/// What a line of the event feed lets the player do.
#[derive(Clone, Copy)]
enum FeedAction {
    /// Replay the best creature of this history row.
    Replay(usize),
    /// Replay the champion now, whose record no history row holds yet.
    ReplayChampion,
    /// Bring back creatures lost to catastrophes.
    Undo,
    /// Set this effect (index into `EFFECTS`) to this level.
    Try(usize, usize),
    /// Set the world of this wild island (index among the wild islands).
    Wild(usize),
}
/// Generations without a record before the feed suggests a new world.
const STALL_GENERATIONS: u32 = 25;
/// The feed suggests a new world when the effective clades fell by this share
/// within `COLLAPSE_WINDOW` generations of one world.
const COLLAPSE_SHARE: f32 = 0.25;
const COLLAPSE_WINDOW: usize = 50;
/// The effects a stall hint suggests, in order; the first that can go one
/// level harder wins.
const STALL_EFFECTS: [&str; 13] = [
    "Ground",
    "Hurdles",
    "Grip",
    "Slope",
    "Mud",
    "Brambles",
    "Gaps",
    "Wind",
    "Air",
    "Gravity",
    "Earthquake",
    "Heat wave",
    "Drought",
];
/// An effect and the next harder level to try when evolution stalls.
fn stall_suggestion(config: &Config) -> Option<(usize, usize)> {
    STALL_EFFECTS.iter().find_map(|name| {
        let index = crate::environment::EFFECTS
            .iter()
            .position(|effect| effect.name == *name)?;
        let effect = &crate::environment::EFFECTS[index];
        let level = effect.level(config);
        (level + 1 < effect.levels.len()).then_some((index, level + 1))
    })
}
/// One line of the event feed.
struct FeedItem {
    generation: u32,
    text: String,
    color: Color32,
    action: Option<FeedAction>,
}
/// The smallest gain that counts as a new record (m).
const RECORD_STEP: f32 = 0.01;
/// History positions where the best distance moved within one world, oldest
/// first, and whether each is the first best after a world change. A harder
/// world lowers the best, so records count again from its first generation.
fn world_records(history: &[Stats]) -> Vec<(usize, f32, bool)> {
    let mut best = f32::NEG_INFINITY;
    let mut fresh = false;
    let mut records = Vec::new();
    for (index, stats) in history.iter().enumerate() {
        if index > 0 && stats.config.physics_differs(&history[index - 1].config) {
            best = f32::NEG_INFINITY;
            fresh = true;
        }
        // A record beats the last one by at least a centimeter, so two
        // records never read the same. A row with nothing kept has no best.
        if stats.archive_cells > 0 && stats.best.is_finite() && stats.best >= best + RECORD_STEP {
            best = stats.best;
            records.push((index, best, fresh));
            fresh = false;
        }
    }
    records
}
/// The newest history row when it was measured in the world that is live now.
/// After a world change the rows are from the old world, and nothing from
/// them may stand for the current world.
fn row_in_world(snapshot: &Snapshot) -> Option<&Stats> {
    snapshot
        .history
        .last()
        .filter(|row| !row.config.physics_differs(&snapshot.config))
}
/// A record set in the generation that is running, before its history row
/// exists.
struct LiveRecord {
    best: f32,
    /// The first best in a world the history has no row for yet.
    first_in_world: bool,
    /// No generation has finished at all.
    first_ever: bool,
}
/// The running generation's best and median when its row is not written yet:
/// (generation, best, median).
fn live_point(snapshot: &Snapshot) -> Option<(u32, f32, f32)> {
    let behind = snapshot
        .history
        .last()
        .is_none_or(|last| snapshot.generation > last.generation);
    (behind && snapshot.live_best.is_finite()).then_some((
        snapshot.generation,
        snapshot.live_best,
        snapshot.live_median,
    ))
}
/// The champion's record, when it beats the last record of this world by a
/// record step. `snapshot.champion` holds its creature.
fn live_record(snapshot: &Snapshot) -> Option<LiveRecord> {
    let (_, best, _) = live_point(snapshot)?;
    snapshot.champion.as_ref()?;
    let Some(last) = snapshot.history.last() else {
        return Some(LiveRecord {
            best,
            first_in_world: false,
            first_ever: true,
        });
    };
    if snapshot.config.physics_differs(&last.config) {
        return Some(LiveRecord {
            best,
            first_in_world: true,
            first_ever: false,
        });
    }
    let held = world_records(&snapshot.history)
        .last()
        .map_or(f32::NEG_INFINITY, |&(_, b, _)| b);
    (best >= held + RECORD_STEP).then_some(LiveRecord {
        best,
        first_in_world: false,
        first_ever: false,
    })
}
/// Heat map of the archive: for each ground contact and cadence pair, the
/// best creature among the height, feet, shape and size bins the filters
/// (`[height, feet, shape, size]`) let through. One color scale spans every
/// cell of the archive, so a color means the same distance whatever the
/// filters. Returns the id of a clicked cell's creature.
fn paint_archive_map(
    ui: &mut egui::Ui,
    cells: &[crate::worker::MapCell],
    [height_bin, feet_bin, shape_bin, size_bin]: [Option<usize>; 4],
    theme: Theme,
) -> Option<u64> {
    let (min, max) = cells
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), cell| {
            (lo.min(cell.score), hi.max(cell.score))
        });
    let range = (max - min).max(1e-6);
    // Best cell and how many ways of moving share each contact and cadence
    // pair under the filters.
    let mut best: HashMap<(u8, u8), (crate::worker::MapCell, usize)> = HashMap::new();
    for cell in cells.iter().filter(|cell| {
        height_bin.is_none_or(|bin| usize::from(cell.niche[3]) == bin)
            && feet_bin.is_none_or(|bin| usize::from(cell.niche[4]) == bin)
            && shape_bin.is_none_or(|bin| usize::from(cell.niche[2]) == bin)
            && size_bin.is_none_or(|bin| usize::from(cell.niche[5]) == bin)
    }) {
        let entry = best
            .entry((cell.niche[0], cell.niche[1]))
            .or_insert((*cell, 0));
        entry.1 += 1;
        if cell.score > entry.0.score {
            entry.0 = *cell;
        }
    }
    ui.label(
        RichText::new(format!(
            "Ground contact against stride rate · {} cells · the best of each is shown",
            best.len(),
        ))
        .small()
        .color(theme.muted),
    );
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), ui.available_height().max(220.)),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    let plot = Rect::from_min_max(
        rect.left_top() + Vec2::new(56., 34.),
        rect.right_bottom() - Vec2::new(12., 42.),
    );
    if plot.width() < 80. || plot.height() < 80. {
        return None;
    }
    let columns = MAP_BINS[0];
    let rows = MAP_BINS[1];
    let column_width = plot.width() / columns as f32;
    let row_height = plot.height() / rows as f32;
    for column in 0..=columns {
        let x = plot.left() + column as f32 * column_width;
        painter.line_segment(
            [Pos2::new(x, plot.top()), Pos2::new(x, plot.bottom())],
            Stroke::new(1., theme.card_border),
        );
        painter.text(
            Pos2::new(x, plot.bottom() + 4.),
            Align2::CENTER_TOP,
            format!("{:.0}%", column as f32 / columns as f32 * 100.),
            FontId::proportional(14.5),
            theme.muted,
        );
    }
    for row in 0..=rows {
        let y = plot.bottom() - row as f32 * row_height;
        painter.line_segment(
            [Pos2::new(plot.left(), y), Pos2::new(plot.right(), y)],
            Stroke::new(1., theme.card_border),
        );
        painter.text(
            Pos2::new(plot.left() - 6., y),
            Align2::RIGHT_CENTER,
            format!("{:.2}", row as f32 / rows as f32 * 6.),
            FontId::proportional(14.5),
            theme.muted,
        );
    }
    painter.text(
        Pos2::new(plot.center().x, plot.bottom() + 24.),
        Align2::CENTER_TOP,
        "Share of the trial on the ground",
        FontId::proportional(14.),
        theme.ink,
    );
    painter.text(
        Pos2::new(rect.left() + 4., plot.top() - 16.),
        Align2::LEFT_BOTTOM,
        "Strides per second",
        FontId::proportional(14.),
        theme.ink,
    );
    if !cells.is_empty() {
        let legend = Rect::from_min_max(
            Pos2::new(rect.right() - 272., rect.top() + 8.),
            Pos2::new(rect.right() - 92., rect.top() + 22.),
        );
        let steps = 48;
        for i in 0..steps {
            let t = i as f32 / (steps - 1) as f32;
            painter.rect_filled(
                Rect::from_min_max(
                    Pos2::new(
                        legend.left() + legend.width() * i as f32 / steps as f32,
                        legend.top(),
                    ),
                    Pos2::new(
                        legend.left() + legend.width() * (i + 1) as f32 / steps as f32,
                        legend.bottom(),
                    ),
                ),
                0,
                heat_color(t),
            );
        }
        painter.text(
            Pos2::new(legend.left() - 6., legend.center().y),
            Align2::RIGHT_CENTER,
            format!("{min:.2} m"),
            FontId::proportional(14.5),
            theme.muted,
        );
        painter.text(
            Pos2::new(legend.right() + 6., legend.center().y),
            Align2::LEFT_CENTER,
            format!("{max:.2} m"),
            FontId::proportional(14.5),
            theme.muted,
        );
    }
    if best.is_empty() {
        painter.text(
            plot.center(),
            Align2::CENTER_CENTER,
            if cells.is_empty() {
                "No creatures kept yet. The map fills as evolution runs."
            } else {
                "No creatures with this height and these feet. Try All."
            },
            FontId::proportional(16.),
            theme.muted,
        );
    }
    let mut clicked = None;
    for (&(contact, cadence), (cell, count)) in &best {
        let inner = Rect::from_min_size(
            Pos2::new(
                plot.left() + contact as f32 * column_width + 1.5,
                plot.bottom() - (cadence as f32 + 1.) * row_height + 1.5,
            ),
            Vec2::new(column_width - 3., row_height - 3.),
        );
        let color = heat_color((cell.score - min) / range);
        painter.rect_filled(inner, 3, color);
        let luminance = (color.r() as u32 + color.g() as u32 + color.b() as u32) / 3;
        let ink = if luminance > 150 {
            Color32::from_rgb(24, 24, 20)
        } else {
            Color32::WHITE
        };
        if inner.width() > 34. && inner.height() > 16. {
            painter.text(
                inner.center(),
                Align2::CENTER_CENTER,
                format!("{:.1}", cell.score),
                FontId::proportional(14.),
                ink,
            );
        }
        let response = ui.interact(
            inner,
            ui.id().with(("archive_map", contact, cadence)),
            Sense::click(),
        );
        if response.hovered() {
            painter.rect_stroke(
                inner,
                3,
                Stroke::new(2., theme.ink),
                egui::StrokeKind::Inside,
            );
        }
        if response.clicked() {
            clicked = Some(cell.id);
        }
        response.on_hover_text(format!(
            "Rank {} · {:.2} m · {} tall · {}\n{} ways of moving in this cell, the best is shown\nClick to replay",
            cell.rank + 1,
            cell.score,
            height_bin_label(usize::from(cell.niche[3])),
            feet_bin_label(usize::from(cell.niche[4])),
            count,
        ));
    }
    clicked
}
/// One ancestor tile with its thumbnail, generation, fitness and gain.
fn paint_lineage_tile(
    ui: &mut egui::Ui,
    step: &crate::worker::LineageStep,
    parent: Option<&crate::worker::LineageStep>,
    big: bool,
    current: bool,
    theme: Theme,
    size: Vec2,
) -> bool {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let hovered = response.hovered();
    let painter = ui.painter_at(rect);
    let changed = body_plan_changed(step, parent);
    crate::theme::plate(
        &painter,
        rect,
        theme,
        if hovered {
            theme.card_hover
        } else {
            theme.card
        },
        current || changed || hovered,
    );
    let art_size = (size.y - 16.).clamp(40., 96.);
    thumbnail(
        &painter,
        &step.creature,
        Rect::from_min_size(rect.left_top() + Vec2::splat(8.), Vec2::splat(art_size)),
    );
    let text_x = rect.left() + 16. + art_size;
    painter.text(
        Pos2::new(text_x, rect.top() + 8.),
        Align2::LEFT_TOP,
        format!("Generation {}", step.generation),
        FontId::proportional(15.),
        theme.muted,
    );
    painter.text(
        Pos2::new(text_x, rect.top() + 28.),
        Align2::LEFT_TOP,
        format!("{:.2} m", step.fitness),
        FontId::proportional(19.),
        if big { theme.accent } else { theme.ink },
    );
    painter.text(
        Pos2::new(text_x, rect.top() + 54.),
        Align2::LEFT_TOP,
        format!("{:+.2} m", step.gain),
        FontId::proportional(15.),
        if step.gain >= 0. {
            theme.accent
        } else {
            theme.warn
        },
    );
    painter.text(
        rect.right_bottom() + Vec2::new(-8., -6.),
        Align2::RIGHT_BOTTOM,
        species_name(&step.creature),
        FontId::proportional(14.5),
        theme.muted,
    );
    if current {
        painter.text(
            rect.right_bottom() + Vec2::new(-8., -20.),
            Align2::RIGHT_BOTTOM,
            "selected",
            FontId::proportional(14.5),
            theme.accent,
        );
    }
    if changed {
        crate::theme::caps_text(
            &painter,
            rect.right_top() + Vec2::new(-8., 8.),
            Align2::RIGHT_TOP,
            "Body plan",
            13.,
            theme.accent,
        );
    }
    let (nodes, bones, muscles) = body_counts(&step.creature);
    response
        .on_hover_text(format!(
            "{} · generation {} · {:.2} m ({:+.2} m)\n{} nodes / {} bones / {} muscles\n{}\n{}",
            species_name(&step.creature),
            step.generation,
            step.fitness,
            step.gain,
            nodes,
            bones,
            muscles,
            step.change,
            parent.map_or_else(
                || "oldest recorded ancestor".to_owned(),
                |p| format!("parent: generation {} at {:.2} m", p.generation, p.fitness),
            ),
        ))
        .clicked()
}
#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Overview,
    Population,
    History,
    Race,
    Lineage,
}
struct App {
    worker: Worker,
    snapshot: Option<Snapshot>,
    config: Config,
    playback: Option<Playback>,
    /// The replay being recorded for `playback`, and when it was asked for.
    replay_wait: Option<(mpsc::Receiver<Playback>, Instant)>,
    /// Seconds the last replay took to appear, for benchmarks.
    replay_seconds: Vec<f32>,
    bench_last_replay: Instant,
    ctx: egui::Context,
    tab: Tab,
    speed: f32,
    playing: bool,
    zoom: f32,
    /// True once the user zoomed by hand; until then the zoom fits the creature.
    zoom_user: bool,
    /// Player view option: draw muscle forces and ground pushes.
    show_forces: bool,
    camera: [f32; 2],
    follow: bool,
    history_index: usize,
    history_latest: bool,
    file_mode: Option<&'static str>,
    /// The saves File > Open lists, newest first, while that window is open.
    open_list: Option<Vec<SaveEntry>>,
    /// When this experiment was last saved or opened, at which generation,
    /// and whether it was an open; None for a new experiment never saved.
    saved: Option<(Instant, u32, bool)>,
    /// A save the worker is writing, since when.
    saving: Option<Instant>,
    /// A save path that exists and waits for the player to confirm.
    overwrite: Option<PathBuf>,
    /// The experiment and number of worker events already read.
    events_seen: (u64, usize),
    file_path: String,
    message: Option<String>,
    /// The message on the status line and when it first showed.
    shown_message: Option<(String, Instant)>,
    /// Since when the GPU has waited for a kernel (`cuda_engine::compiling_world`).
    compiling_since: Option<Instant>,
    new_dialog: bool,
    /// When the UI last sent a settings change to the worker.
    config_sent: Option<Instant>,
    /// Worlds before each change made in the World panel, newest last, for
    /// its Undo button.
    world_undo: Vec<Config>,
    last_frame: Instant,
    frame_times: std::collections::VecDeque<f32>,
    /// The Diagnostics drawer under the status line is open.
    show_perf: bool,
    ui_scale: f32,
    initial: bool,
    smoke_start_pending: bool,
    smoke_preset: Option<usize>,
    started: Instant,
    /// The player closed the loading card; compiling goes on in a corner
    /// note.
    loading_hidden: bool,
    /// A screenshot run shows made-up loading jobs (`loading::demo`).
    loading_demo: bool,
    capture_requested: bool,
    capture_path: Option<String>,
    /// Ancestors of the selected creature, newest first.
    lineage: Vec<crate::worker::LineageStep>,
    /// Creature whose ancestors the UI has already asked the worker for.
    lineage_requested: Option<u64>,
    /// A lineage request is in flight.
    lineage_pending: bool,
    /// The player picked the creature on screen, so the theater stops
    /// following the champion until they go back to it.
    pinned: bool,
    /// The creature on screen is the champion of a finished generation (or
    /// the best elite of a loaded game), not a random first creature.
    champion_shown: bool,
    /// Behavior archive map: occupied cells keyed by their niche bytes.
    /// Map filters; None shows every bin.
    map_height: Option<usize>,
    map_feet: Option<usize>,
    map_shape: Option<usize>,
    map_size: Option<usize>,
    /// Whether the worker was last asked to send the map table.
    map_sent: bool,
    /// The archive cards the player filters for.
    card_filter: CardFilter,
    /// The ranked archive on screen in Ways of moving and for the race. It
    /// changes only when the player opens the tab or asks for the latest.
    cards: Option<crate::worker::CardList>,
    /// When the UI last asked the worker for the ranked archive.
    cards_requested: Option<Instant>,
    /// The tab of the previous frame, to notice when the player opens one.
    prev_tab: Tab,
    archive_view: ArchiveView,
    /// Top archived elites racing side by side.
    race: Vec<RaceLane>,
    race_pending: bool,
    race_camera: f32,
    /// Creatures the player sent to the race, oldest first, with their worlds.
    race_picks: Vec<(Creature, Config)>,
    /// Native benchmark frame intervals, the time each frame began, and
    /// the last control probe time.
    bench_frames: Vec<f32>,
    bench_frame_starts: Vec<Instant>,
    /// CPU time of each measured frame without the vsync wait (eframe's
    /// `cpu_usage`), with the time it began.
    bench_work: Vec<f32>,
    /// The UI thread's major page faults when the measured window began.
    bench_faults: Option<u64>,
    bench_last_ping: Instant,
    bench_pings: u64,
    show_help: bool,
    /// The "How evolution works" window (`schematic::show`).
    pub schematic_open: bool,
    runs_bytes: u64,
    runs_checked: Instant,
    screenshot_pending: bool,
    screenshot_waiting: bool,
    /// Unattended runs: `EVOLUTION_CAPTURE_EVERY=<n>` saves the window to
    /// `runs/progress-gen<g>.png` every n generations; the generation of the
    /// capture in flight, and the last one taken.
    capture_every: Option<u32>,
    capture_generation: Option<u32>,
    captured_generation: u32,
}
impl App {
    fn new(cc: &eframe::CreationContext<'_>, worker: Worker) -> Self {
        let ctx = &cc.egui_ctx;
        // Screenshot runs of the loading screen (no GPU needed).
        // `toast` shows the corner note instead of the card.
        let demo = std::env::var("EVOLUTION_SMOKE_LOADING").ok();
        let loading_demo = demo.is_some();
        if loading_demo {
            crate::loading::demo();
        }
        let mut initial_config = Config::default();
        if let Ok(n) = std::env::var("EVOLUTION_SMOKE_POPULATION")
            && let Ok(n) = n.parse()
        {
            initial_config.population = n;
            initial_config.random_seed = false;
            // Keep the normal periodic autosave in benchmarks unless explicitly disabled.
            if std::env::var_os("EVOLUTION_BENCH_NO_AUTOSAVE").is_some() {
                initial_config.checkpoint_interval = 0;
            }
            initial_config.throughput = n >= 100_000;
        }
        if let Ok(duration) = std::env::var("EVOLUTION_BENCH_DURATION")
            && let Ok(duration) = duration.parse()
        {
            initial_config.duration = duration;
        }
        if std::env::var_os("EVOLUTION_BENCH_THROUGHPUT").is_some() {
            initial_config.throughput = true;
        }
        if std::env::var_os("EVOLUTION_BENCH_RESPONSIVE").is_some() {
            initial_config.throughput = false;
        }
        // Screenshot runs: EVOLUTION_SMOKE_WORLD="Wind=2,Mud=3" starts the
        // game with those effect levels. EVOLUTION_AUTOSTART takes the same
        // list, turns autosave on and starts evolving continuously, for an
        // unattended run.
        let autostart = std::env::var("EVOLUTION_AUTOSTART").ok();
        if autostart.is_some() {
            initial_config.checkpoint_interval = AUTOSAVE_INTERVAL;
        }
        for list in [
            std::env::var("EVOLUTION_SMOKE_WORLD").ok(),
            autostart.clone(),
        ]
        .into_iter()
        .flatten()
        {
            for pair in list.split(',') {
                if let Some((name, level)) = pair.split_once('=')
                    && let Ok(level) = level.trim().parse::<usize>()
                    && let Some(effect) = crate::environment::EFFECTS
                        .iter()
                        .find(|e| e.name.eq_ignore_ascii_case(name.trim()))
                {
                    effect.set_level(&mut initial_config, level);
                }
            }
        }
        let smoke_start_pending =
            std::env::var_os("EVOLUTION_SMOKE_POPULATION").is_some() || autostart.is_some();
        // The game's fonts and style; its art decodes in the background.
        // Developer screenshots: EVOLUTION_SMOKE_ZOOM=0.75 lays a 1440 px
        // window out like a 1920 px one.
        crate::assets::install_fonts(ctx);
        crate::assets::preload(ctx);
        apply_style(ctx);
        let smoke_zoom = std::env::var("EVOLUTION_SMOKE_ZOOM")
            .ok()
            .and_then(|zoom| zoom.parse::<f32>().ok())
            .filter(|zoom| (0.5..=2.0).contains(zoom));
        if let Some(zoom) = smoke_zoom {
            ctx.set_zoom_factor(zoom);
        }
        let smoke_tab = std::env::var("EVOLUTION_SMOKE_TAB").unwrap_or_default();
        if let Some(path) = std::env::var_os("EVOLUTION_SMOKE_CHECKPOINT") {
            worker.send(Command::Load(PathBuf::from(path)));
        } else {
            worker.send(Command::New(initial_config));
        }
        Self {
            worker,
            snapshot: None,
            config: Config::default(),
            playback: None,
            tab: match smoke_tab.as_str() {
                "history" => Tab::History,
                "population" | "map" | "islands" => Tab::Population,
                "race" => Tab::Race,
                "lineage" => Tab::Lineage,
                _ => Tab::Overview,
            },
            speed: 1.0,
            playing: true,
            zoom: DEFAULT_CAMERA_ZOOM,
            zoom_user: false,
            show_forces: std::env::var_os("EVOLUTION_SMOKE_FORCES").is_some(),
            camera: [0.0, 0.0],
            follow: true,
            history_index: 0,
            history_latest: true,
            file_mode: None,
            open_list: None,
            saved: None,
            saving: None,
            overwrite: None,
            events_seen: (u64::MAX, 0),
            file_path: "runs/experiment.evo".into(),
            message: None,
            shown_message: None,
            compiling_since: None,
            new_dialog: false,
            config_sent: None,
            world_undo: Vec::new(),
            last_frame: Instant::now(),
            frame_times: Default::default(),
            show_perf: false,
            ui_scale: smoke_zoom.unwrap_or(1.0),
            initial: true,
            smoke_start_pending,
            smoke_preset: std::env::var("EVOLUTION_SMOKE_PRESET")
                .ok()
                .and_then(|n| n.parse().ok()),
            started: Instant::now(),
            loading_hidden: demo.as_deref() == Some("toast"),
            loading_demo,
            capture_requested: false,
            capture_path: std::env::var("EVOLUTION_SMOKE_CAPTURE").ok(),
            lineage: Vec::new(),
            lineage_requested: None,
            lineage_pending: false,
            pinned: false,
            champion_shown: false,
            map_height: None,
            map_feet: None,
            map_shape: None,
            map_size: None,
            map_sent: false,
            card_filter: Default::default(),
            cards: None,
            cards_requested: None,
            prev_tab: Tab::Overview,
            archive_view: if smoke_tab == "map" {
                ArchiveView::Map
            } else if smoke_tab == "islands" {
                ArchiveView::Islands
            } else {
                ArchiveView::Cards
            },
            race: Vec::new(),
            race_pending: smoke_tab == "race",
            race_camera: 0.0,
            race_picks: Vec::new(),
            bench_frames: Vec::new(),
            bench_frame_starts: Vec::new(),
            bench_work: Vec::new(),
            bench_faults: None,
            replay_wait: None,
            replay_seconds: Vec::new(),
            bench_last_replay: Instant::now(),
            ctx: ctx.clone(),
            bench_last_ping: Instant::now(),
            bench_pings: 0,
            show_help: false,
            schematic_open: std::env::var_os("EVOLUTION_SMOKE_SCHEMATIC").is_some(),
            runs_bytes: 0,
            runs_checked: Instant::now() - RUNS_REFRESH,
            screenshot_pending: false,
            screenshot_waiting: false,
            capture_every: std::env::var("EVOLUTION_CAPTURE_EVERY")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| v > 0),
            capture_generation: None,
            captured_generation: 0,
        }
    }
    fn theme(&self) -> Theme {
        Theme::get()
    }
    /// The loading card while the devices open or the starting worlds'
    /// kernels compile (the player may close it), while the first generation
    /// waits for the kernels of its world, and a corner note for compiles
    /// nobody waits for.
    fn loading_screen(&mut self, ctx: &egui::Context) {
        let theme = self.theme();
        let failed = self
            .worker
            .failed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let first = self
            .snapshot
            .as_ref()
            .is_some_and(|s| s.running && s.history.is_empty() && s.completed == 0);
        let startup = crate::loading::progress(crate::loading::Group::Startup).busy();
        let wait = if self.loading_demo {
            (!self.loading_hidden).then_some(loading::Wait::Starting)
        } else if let Some(error) = failed.as_deref() {
            Some(loading::Wait::Failed(error))
        } else if !self.worker.opened.load(Ordering::Relaxed) {
            Some(loading::Wait::Opening)
        } else if !self.loading_hidden && startup {
            Some(loading::Wait::Starting)
        } else if !self.loading_hidden && first && crate::cuda_engine::compiling_world() {
            Some(loading::Wait::World)
        } else {
            None
        };
        match wait {
            Some(wait) => {
                let card = loading::Card {
                    wait,
                    since: self.started,
                };
                if loading::screen(ctx, theme, &card) {
                    self.loading_hidden = true;
                }
            }
            None => loading::toast(ctx, theme, self.loading_hidden),
        }
    }
    fn active(&self) -> bool {
        self.snapshot.as_ref().is_some_and(|s| s.running)
            && !self.worker.pause.load(Ordering::Relaxed)
    }
    fn run(&mut self, continuous: bool, guided: bool) {
        self.worker.pause.store(false, Ordering::Relaxed);
        self.worker.send(Command::Run { continuous, guided });
    }
    fn pause(&self) {
        self.worker.pause.store(true, Ordering::Relaxed);
        self.worker.send(Command::Pause);
    }
    fn file(&mut self, mode: &'static str) {
        self.file_mode = Some(mode);
        self.file_path = match mode {
            "Export CSV" => "runs/statistics.csv".to_owned(),
            "Open creature JSON" => "runs/creature.json".to_owned(),
            "Export creature JSON" | "Export creature GIF" => {
                let extension = if mode.ends_with("GIF") { "gif" } else { "json" };
                self.playback.as_ref().map_or_else(
                    || format!("runs/creature.{extension}"),
                    |p| {
                        format!(
                            "runs/{}-{:.1}m-{}.{extension}",
                            species_name(&p.creature).replace(' ', "-"),
                            p.distance,
                            p.creature.id
                        )
                    },
                )
            }
            _ => "runs/experiment.evo".to_owned(),
        };
    }
    fn set_preview(&mut self, c: Creature, cfg: Config) {
        // The replay is recorded off the UI thread: the player shows the
        // creature's first pose meanwhile, and the recording replaces it.
        self.playback = Some(Playback::preparing(c.clone(), cfg.clone()));
        let (tx, rx) = mpsc::channel();
        let ctx = self.ctx.clone();
        let spawned = std::thread::Builder::new()
            .name("replay".into())
            .spawn(move || {
                let _ = tx.send(Playback::recorded(c, cfg, Duration::from_secs(60)));
                ctx.request_repaint();
            });
        self.replay_wait = spawned.is_ok().then(|| (rx, Instant::now()));
        self.follow = true;
        self.zoom = DEFAULT_CAMERA_ZOOM;
        self.zoom_user = false;
        self.camera = [0.; 2];
    }
    /// Shows a creature the player picked. The theater keeps it until the
    /// player goes back to the champion.
    fn select(&mut self, creature: Creature, config: Config) {
        self.pinned = true;
        self.set_preview(creature, config);
        self.lineage.clear();
    }
    /// Shows an ancestor from the lineage the player is browsing, keeping the
    /// lineage on screen.
    fn select_ancestor(&mut self, creature: Creature, config: Config) {
        self.pinned = true;
        self.set_preview(creature, config);
    }
    /// Shows a champion and follows new ones from now on.
    fn show_champion(&mut self, creature: Creature, config: Config) {
        self.pinned = false;
        self.champion_shown = true;
        self.set_preview(creature, config);
        self.lineage.clear();
    }
    /// The best elite in the archive now, and the world it is scored in. The
    /// worker sends it as soon as a record is absorbed, mid-generation too;
    /// the newest finished generation's best stands in until then.
    fn champion(&self) -> Option<(Creature, Config)> {
        let snapshot = self.snapshot.as_ref()?;
        if let Some(live) = &snapshot.champion {
            return Some((live.0.clone(), live.1.clone()));
        }
        // A row from before a world change is not this world's champion.
        let stats = row_in_world(snapshot)?;
        Some((stats.representatives.last()?.clone(), stats.config.clone()))
    }
    /// The world changed and nothing measured in it is kept yet.
    fn awaiting_new_world(&self) -> bool {
        self.snapshot.as_ref().is_some_and(|s| {
            !s.history.is_empty() && s.champion.is_none() && row_in_world(s).is_none()
        })
    }
    /// Keeps the theater (on the Overview and docked beside Ways of moving)
    /// on the champion unless the player pinned a creature. A new champion,
    /// which a new distance record brings, replaces the one on screen at
    /// once.
    fn follow_champion(&mut self) {
        let Some((creature, config)) = self.champion() else {
            // The world changed and no creature is kept in it yet: the old
            // champion does not stand for this world, so the view empties.
            if !self.pinned && self.awaiting_new_world() && self.playback.is_some() {
                self.playback = None;
                self.replay_wait = None;
                self.champion_shown = false;
            }
            return;
        };
        let showing = self.playback.as_ref().map(|p| p.creature.id);
        if follows_champion(self.pinned, showing, Some(creature.id)) {
            self.show_champion(creature, config);
        } else if !self.pinned && showing == Some(creature.id) {
            self.champion_shown = true;
        }
    }
    /// The world a generation ran in: the settings its history row kept, or
    /// the live world for a generation without a row yet.
    fn world_of_generation(&self, generation: u32) -> Config {
        let Some(snapshot) = &self.snapshot else {
            return Config::default();
        };
        snapshot
            .history
            .iter()
            .rev()
            .find(|stats| stats.generation == generation)
            .map_or_else(|| snapshot.config.clone(), |stats| stats.config.clone())
    }
    /// Stops watching a picked creature and shows the champion now.
    fn back_to_champion(&mut self) {
        self.pinned = false;
        if let Some((creature, config)) = self.champion() {
            self.show_champion(creature, config);
        }
    }
    fn top(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.add_space(GAP_S);
            let (logo, _) = ui.allocate_exact_size(Vec2::splat(28.), Sense::hover());
            // Three joined nodes, the smallest creature, lit in amber.
            let points = [
                logo.center_top() + Vec2::new(0., 4.),
                logo.left_bottom() + Vec2::new(4., -4.),
                logo.right_bottom() + Vec2::new(-4., -4.),
            ];
            let painter = ui.painter();
            let amber = crate::theme::poster::MUSTARD;
            for i in 0..3 {
                painter.line_segment(
                    [points[i], points[(i + 1) % 3]],
                    Stroke::new(3., amber),
                );
            }
            for point in points {
                painter.circle_filled(point, 4.5, amber);
                painter.circle_filled(point, 1.8, crate::theme::poster::WOOD_DARK);
            }
            // The game's name (owner, 2026-10-03), in chunky condensed letters.
            let mut job = egui::text::LayoutJob::default();
            job.append(
                "exploraMove",
                0.,
                egui::TextFormat {
                    font_id: FontId::new(32., crate::assets::hud_bold()),
                    color: crate::theme::poster::CREAM,
                    extra_letter_spacing: 1.,
                    ..Default::default()
                },
            );
            let title = ui.painter().layout_job(job);
            let (title_rect, _) = ui.allocate_exact_size(title.size(), Sense::hover());
            ui.painter()
                .galley(title_rect.min, title, crate::theme::poster::CREAM);
            ui.add_space(GAP_L);
            let running = self.active();
            let (text, fill, ink, why) = if running {
                (
                    "Pause evolution  (Space)",
                    theme.stop_fill,
                    theme.stop_text,
                    "Stop after the work in flight. The replay keeps playing.",
                )
            } else {
                (
                    "Evolve  (Space)",
                    theme.go_fill,
                    theme.go_text,
                    "Run generation after generation until you pause.",
                )
            };
            if ui
                .add(
                    egui::Button::new(RichText::new(text).size(18.).strong().color(ink))
                        .fill(fill)
                        .min_size(Vec2::new(210., 40.)),
                )
                .on_hover_text(why)
                .clicked()
            {
                if running {
                    self.pause();
                } else {
                    self.run(true, false);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("Help")
                    .on_hover_text("Shortcuts and what each tab shows · F1")
                    .clicked()
                {
                    self.show_help = !self.show_help;
                }
                ui.menu_button("View", |ui| {
                    // The new scale applies when the drag ends: scaling the
                    // UI under the pointer mid-drag moves the slider and threw
                    // it to the other end.
                    let slider = ui.add(
                        egui::Slider::new(&mut self.ui_scale, 0.75..=1.6).text("UI scale"),
                    );
                    if slider.drag_stopped() || (slider.changed() && !slider.dragged()) {
                        ui.ctx().set_zoom_factor(self.ui_scale);
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui.button("New experiment…").clicked() {
                        self.new_dialog = true;
                        ui.close();
                    }
                    if ui.button("Open…").clicked() {
                        self.open_list = Some(list_saves(std::path::Path::new("runs")));
                        self.file_path = "runs/experiment.evo".to_owned();
                        ui.close();
                    }
                    if ui.button("Save…  Ctrl+S").clicked() {
                        self.file("Save experiment");
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Open creature JSON…").clicked() {
                        self.file("Open creature JSON");
                        ui.close();
                    }
                    if ui.button("Export statistics CSV…").clicked() {
                        self.file("Export CSV");
                        ui.close();
                    }
                    if ui.button("Screenshot").clicked() {
                        self.screenshot_pending = true;
                        self.screenshot_waiting = true;
                        self.message = Some("Taking a screenshot…".into());
                        ui.close();
                    }
                    ui.separator();
                    let mut autosave = self.config.checkpoint_interval > 0;
                    if ui
                        .checkbox(
                            &mut autosave,
                            format!("Autosave every {AUTOSAVE_INTERVAL} generations"),
                        )
                        .on_hover_text("Writes runs/seed-<seed>-auto.evo in the background and keeps the three newest.")
                        .changed()
                    {
                        self.config.checkpoint_interval = if autosave { AUTOSAVE_INTERVAL } else { 0 };
                        self.worker.send(Command::Configure(self.config.clone()));
                        self.config_sent = Some(Instant::now());
                    }
                });
                let (state, busy) = self.save_state();
                ui.label(RichText::new(state).color(if busy {
                    crate::theme::poster::MUSTARD
                } else {
                    crate::theme::poster::PAPER
                }))
                .on_hover_text("File > Save writes the experiment to runs/. The game writes nothing on its own unless autosave is on.");
                if busy {
                    ui.spinner();
                    ui.ctx().request_repaint_after(Duration::from_millis(250));
                }
                ui.separator();
                if let Some(s) = &self.snapshot {
                    ui.label(
                        RichText::new(format!(
                            "{} creatures · {:.0} s trials",
                            number(s.config.population),
                            s.config.duration
                        ))
                        .color(crate::theme::poster::PAPER),
                    );
                }
            });
        });
    }
    fn controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| self.control_contents(ui));
    }
    fn control_contents(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.add_space(GAP_M);
        let mut world_changed = false;
        let world_before = self.config.clone();
        let mut undoing = false;
        crate::theme::section(ui, "World", theme);
        let live = self.snapshot.as_ref().map(|s| s.config.clone());
        let calm = world_is_calm(&self.config);
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(world_summary(&self.config))
                    .size(18.)
                    .strong()
                    .color(if calm { theme.ink } else { theme.accent }),
            );
            if !calm
                && ui
                    .small_button("Calm world")
                    .on_hover_text("Set every effect back to the calm world in one change.")
                    .clicked()
            {
                for effect in &crate::environment::EFFECTS {
                    if effect.name != "Autochange environment" {
                        effect.set_level(&mut self.config, effect.calm);
                    }
                }
                world_changed = true;
            }
        });
        if let Some(live) = &live
            && live.physics_differs(&self.config)
        {
            ui.label(
                RichText::new(format!(
                    "Now: {}. The change starts with the next generation.",
                    world_summary(live)
                ))
                .small()
                .color(theme.warn),
            );
        }
        // What is active, one line each, with a one-line reason and an undo
        // that sets that effect back to calm.
        let mut undo_effect = None;
        for (i, effect) in crate::environment::EFFECTS
            .iter()
            .enumerate()
            .filter(|(_, effect)| effect.name != "Autochange environment")
            .filter(|(_, effect)| effect.level(&self.config) != effect.calm)
        {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(effect_text(effect, &self.config)).color(theme.accent));
                if ui
                    .small_button("Undo")
                    .on_hover_text("Set this effect back to calm")
                    .clicked()
                {
                    undo_effect = Some(i);
                }
            });
            ui.label(RichText::new(effect.why).small().color(theme.muted));
        }
        if let Some(i) = undo_effect {
            let effect = &crate::environment::EFFECTS[i];
            effect.set_level(&mut self.config, effect.calm);
            world_changed = true;
        }
        ui.add_space(GAP_S);
        crate::theme::section(ui, "Presets", theme);
        ui.horizontal_wrapped(|ui| {
            for preset in &crate::environment::PRESETS {
                if ui
                    .small_button(preset.name)
                    .on_hover_text(format!(
                        "{} Sets these effects and calms the rest. Undo returns the previous world.",
                        preset.about
                    ))
                    .clicked()
                {
                    preset.apply(&mut self.config);
                    world_changed = true;
                }
            }
            if ui
                .add_enabled(
                    !self.world_undo.is_empty(),
                    egui::Button::new("Undo last change").small(),
                )
                .on_hover_text("Return to the world before your last change here")
                .clicked()
                && let Some(previous) = self.world_undo.pop()
            {
                self.config = previous;
                undoing = true;
                world_changed = true;
            }
        });
        ui.add_space(GAP_S);
        crate::theme::section(ui, "Effects", theme);
        ui.label(
            RichText::new(
                "Click a level to change the world. The best creatures are tested again in the new world.",
            )
            .small()
            .color(theme.muted),
        );
        // One row for every effect and the autochange: the name in a fixed
        // column, then its levels as one segmented bar.
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.y = GAP_S + 2.;
            for effect in crate::environment::EFFECTS
                .iter()
                .filter(|effect| effect.name != "Autochange environment")
            {
                if effect_row(ui, effect, &mut self.config, live.as_ref(), theme) {
                    world_changed = true;
                }
            }
            ui.add_space(GAP_S);
            if let Some(autochange) = crate::environment::EFFECTS
                .iter()
                .find(|effect| effect.name == "Autochange environment")
                && effect_row(ui, autochange, &mut self.config, None, theme)
            {
                world_changed = true;
            }
        });
        let generation = self.snapshot.as_ref().map_or(0, |s| s.generation);
        if let Some(forecast) = autochange_forecast(&self.config, generation) {
            ui.label(RichText::new(forecast).small().color(theme.muted));
        }
        let fossils = self.snapshot.as_ref().map_or(0, |s| s.fossils);
        ui.add_space(GAP_M);
        crate::theme::section(ui, "Catastrophes", theme).on_hover_text(
            "A catastrophe wipes out creatures that evolution kept. Survivors and newcomers refill the empty places, which makes room for new ways of moving.",
        );
        ui.horizontal_wrapped(|ui| {
            if ui
                .small_button("Meteor strike")
                .on_hover_text("Wipe out half of the kept creatures at random. Undo brings them back.")
                .clicked()
            {
                self.worker.send(Command::Meteor);
            }
            if ui
                .small_button("Extinction")
                .on_hover_text("Wipe out the group whose best creature is slowest, so it starts over from new designs. Undo brings them back.")
                .clicked()
            {
                self.worker.send(Command::Extinction);
            }
            if ui
                .add_enabled(
                    fossils > 0,
                    egui::Button::new(if fossils > 0 {
                        format!("Undo ({})", number(fossils))
                    } else {
                        "Undo".to_owned()
                    })
                    .small(),
                )
                .on_hover_text(format!(
                    "Bring back {} creatures lost to catastrophes, where their place is empty or holds a slower creature.",
                    number(fossils)
                ))
                .clicked()
            {
                self.worker.send(Command::UndoMeteor);
            }
        });
        if world_changed {
            if !undoing && world_before.physics_differs(&self.config) {
                self.world_undo.push(world_before);
                if self.world_undo.len() > 20 {
                    self.world_undo.remove(0);
                }
            }
            self.worker.send(Command::Configure(self.config.clone()));
            self.config_sent = Some(Instant::now());
        }
    }
    /// The replay header's buttons: follow, reset camera, and back to the
    /// champion or play the next one. Returns (back, play next).
    fn viewport_buttons(&mut self, ui: &mut egui::Ui) -> bool {
        let theme = self.theme();
        let mut back = false;
        ui.checkbox(&mut self.follow, "Follow")
            .on_hover_text("Keep the camera on the creature");
        ui.checkbox(&mut self.show_forces, "Forces").on_hover_text(
            "Draw muscle forces (orange) and ground pushes (blue), estimated from the recording",
        );
        if ui.button("Reset camera").clicked() {
            self.zoom = DEFAULT_CAMERA_ZOOM;
            self.zoom_user = false;
            self.camera = [0.; 2];
            self.follow = true;
        }
        if self.pinned {
            back = ui
                .button(RichText::new("Back to champion").color(theme.accent))
                .clicked();
        }
        back
    }
    fn viewport(&mut self, ui: &mut egui::Ui, height: f32) {
        let theme = self.theme();
        let mut back = false;
        // A narrow replay (docked beside the archive) puts its buttons on a
        // line of their own.
        let wide = ui.available_width() > 900.;
        let mut header = |ui: &mut egui::Ui| {
            let (mode, fill, color, why) = if self.pinned {
                (
                    " WATCHING ",
                    theme.card_hover,
                    theme.ink,
                    "A creature you picked. Back to champion shows the best creature again.",
                )
            } else if self.champion_shown {
                (
                    " CHAMPION ",
                    theme.go_fill,
                    theme.go_text,
                    "The best creature so far. The view switches to each new champion as soon as it sets a record.",
                )
            } else {
                (
                    " FIRST GENERATION ",
                    theme.card,
                    theme.muted,
                    "A random creature of the first generation. The champion takes over as soon as the first creature is kept.",
                )
            };
            ui.label(
                RichText::new(mode)
                    .size(14.)
                    .strong()
                    .color(color)
                    .background_color(fill),
            )
            .on_hover_text(why);
            if let Some(p) = &self.playback {
                ui.label(
                    RichText::new(format!("{} · {:.2} m", species_name(&p.creature), p.distance))
                        .strong(),
                )
                .on_hover_text(format!(
                    "{} nodes, {} bones, {} muscles. Creature {}. {:.2} m is the distance this replay reaches, and m/s its speed over the last fifth of a second. The replay is the scoring kernel's own trial, so it shows the GPU score. An island record's archive score comes from its confirmation trial.",
                    p.nodes.len(),
                    p.creature.bones.len(),
                    p.creature.muscles.len(),
                    p.creature.id,
                    p.distance,
                ));
            }
            if wide {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    back = self.viewport_buttons(ui);
                });
            }
        };
        if wide {
            ui.horizontal(|ui| header(ui));
        } else {
            ui.horizontal_wrapped(|ui| header(ui));
        }
        if !wide {
            ui.horizontal_wrapped(|ui| {
                back = self.viewport_buttons(ui);
            });
        }
        if back {
            self.back_to_champion();
        }
        let (rect, response) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), height.max(120.)),
            Sense::click_and_drag(),
        );
        if response.clicked() {
            self.playing = !self.playing;
        }
        let response =
            response.on_hover_text("Click to pause or play · drag to pan · scroll to zoom");
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.zoom_user = true;
            }
            self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(30., 1200.);
        }
        if response.dragged() {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera[0] -= delta.x / self.zoom;
            self.camera[1] += delta.y / self.zoom;
            self.follow = false;
        }
        if !self.zoom_user
            && let Some(p) = &self.playback
        {
            self.zoom = player_zoom(p.height, p.peak, rect.height());
        }
        // Developer screenshots: EVOLUTION_SMOKE_VIEW_ZOOM=<pixels per meter>
        // frames a wider stretch of the ground.
        if let Some(zoom) = std::env::var("EVOLUTION_SMOKE_VIEW_ZOOM")
            .ok()
            .and_then(|z| z.parse::<f32>().ok())
        {
            self.zoom = zoom;
        }
        let painter = ui.painter_at(rect);
        // All scene primitives are tessellated into egui's batched wgpu render pass.
        let origin = Pos2::new(
            rect.center().x - self.camera[0] * self.zoom,
            rect.bottom() - rect.height() * GROUND_SHARE + self.camera[1] * self.zoom,
        );
        let world = |x: f32, y: f32| Pos2::new(origin.x + x * self.zoom, origin.y - y * self.zoom);
        let cfg = self
            .playback
            .as_ref()
            .map(|p| &p.config)
            .unwrap_or(&self.config);
        // The replay clock, so effects animate with the replay and hold
        // still when it is paused.
        let clock = self
            .playback
            .as_ref()
            .map_or(0.0, |p| p.tick as f32 / physics::rate() as f32);
        // The replay's own creature decides the earthquake ground, through
        // the same id hash the engines use.
        let quake_hash = self
            .playback
            .as_ref()
            .map_or(0, |p| crate::physics::quake_hash(p.creature.id));
        let slope = if cfg.ground { cfg.slope } else { 0.0 };
        let gaps = if cfg.ground { cfg.gaps } else { 0.0 };
        let hurdles = if cfg.ground { cfg.hurdles } else { 0.0 };
        let quake = if cfg.ground { cfg.quake } else { 0.0 };
        let mud = if cfg.ground { cfg.mud } else { 0.0 };
        let amplitude = crate::physics::terrain_amplitude(cfg.terrain)
            + quake * crate::physics::quake_scale(quake_hash);
        let phase = if quake > 0.0 {
            crate::physics::quake_phase(quake_hash)
        } else {
            0.0
        };
        let height_at = |x: f32, with_hurdles: bool| {
            crate::physics::ground(
                x,
                amplitude,
                slope,
                gaps,
                if with_hurdles { hurdles } else { 0.0 },
                phase,
            )
            .0
        };
        let start = (rect.left() - origin.x) / self.zoom;
        let end = (rect.right() - origin.x) / self.zoom;
        // The skyline stands on the ground under the middle of the view.
        let horizon = if cfg.ground {
            world(
                0.,
                height_at((rect.center().x - origin.x) / self.zoom, false),
            )
            .y
        } else {
            origin.y
        };
        crate::world_fx::backdrop(
            &painter,
            rect,
            horizon,
            self.camera[0] * self.zoom,
            clock,
            cfg,
            (cfg.water > 0.0).then(|| world(0., cfg.water).y),
        );
        crate::world_fx::sky(&painter, rect, cfg, clock);
        let left = start.floor() as i32;
        let right = end.ceil() as i32;
        for x in left..=right {
            let pos = world(x as f32, 0.);
            painter.line_segment(
                [
                    Pos2::new(pos.x, rect.top()),
                    Pos2::new(pos.x, rect.bottom()),
                ],
                Stroke::new(1., crate::theme::scene::GRID),
            );
        }
        if cfg.ground {
            // Sample the ground every few pixels (flat ground needs only its
            // ends) and fill down to the frame with the world's street.
            // Pits carve notches into the polyline; mud draws its sunk layer
            // `mud` meters below the surface line.
            let flat = amplitude == 0.0 && slope == 0.0 && gaps == 0.0 && hurdles == 0.0;
            let step = if flat {
                (end - start).max(0.01)
            } else {
                (4.0 / self.zoom).max(0.002)
            };
            let mut x = start;
            let mut line = Vec::new();
            let mut meters = Vec::new();
            let mut mud_line = Vec::new();
            while x <= end + step {
                let height = height_at(x, true);
                line.push(world(x, height));
                meters.push(x);
                if mud > 0.0 {
                    mud_line.push(world(x, height - mud));
                }
                x += step;
            }
            crate::world_fx::ground_body(&painter, rect, cfg, &line, &meters, self.zoom);
            if mud > 0.0 {
                let fill = crate::theme::scene::MUD;
                for i in 0..line.len().saturating_sub(1) {
                    painter.add(egui::Shape::convex_polygon(
                        vec![line[i], line[i + 1], mud_line[i + 1], mud_line[i]],
                        fill,
                        Stroke::NONE,
                    ));
                }
                painter.add(egui::Shape::line(
                    mud_line,
                    Stroke::new(1.5, crate::theme::scene::MUD_EDGE),
                ));
                let sheen: Vec<Pos2> = line.iter().map(|p| *p + Vec2::new(0., 1.5)).collect();
                painter.add(egui::Shape::line(
                    sheen,
                    Stroke::new(1.5, crate::theme::scene::MUD_SHEEN),
                ));
            }
            let shade: Vec<Pos2> = line.iter().map(|p| *p + Vec2::new(0., 2.)).collect();
            painter.add(egui::Shape::line(
                shade,
                Stroke::new(1.5, Color32::from_black_alpha(110)),
            ));
            painter.add(egui::Shape::line(line, Stroke::new(1.5, GROUND_EDGE)));
            crate::world_fx::structures(&painter, rect, cfg, &world, &height_at, (start, end));
            let surface = |sx: f32| world(0., height_at((sx - origin.x) / self.zoom, true)).y;
            let feet: Vec<crate::world_fx::Foot> = self
                .playback
                .as_ref()
                .map(|p| {
                    let rate = physics::rate() as f32;
                    let before = p.frames.get((p.tick as usize).saturating_sub(1));
                    p.nodes
                        .iter()
                        .enumerate()
                        .filter_map(|(id, n)| {
                            let ground_y = height_at(n.pos[0], true);
                            (n.pos[1] - n.radius - ground_y < 0.06 + mud).then(|| {
                                let dx = before
                                    .and_then(|f| f.get(id))
                                    .map_or(0.0, |b| n.pos[0] - b[0]);
                                crate::world_fx::Foot {
                                    at: world(n.pos[0], ground_y),
                                    speed: dx.abs() * rate * self.zoom,
                                    id,
                                }
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            crate::world_fx::ground(
                &painter,
                rect,
                cfg,
                clock,
                &surface,
                &|sx| (sx - origin.x) / self.zoom,
                self.zoom,
                &feet,
            );
        }
        crate::world_fx::water(
            &painter,
            rect,
            cfg,
            clock,
            world(0., cfg.water).y,
            &|sx| (sx - origin.x) / self.zoom,
            self.zoom,
        );
        // A tick every meter, a label every 1, 2, 5 or 10 m so labels
        // never run into each other.
        let every = [1, 2, 5, 10, 20]
            .into_iter()
            .find(|&n| n as f32 * self.zoom >= 48.)
            .unwrap_or(50);
        for x in left..=right {
            let pos = world(x as f32, 0.);
            painter.line_segment([pos, pos + Vec2::new(0., 6.)], Stroke::new(1., GROUND_INK));
            if x.rem_euclid(every) == 0 {
                painter.text(
                    pos + Vec2::new(5., 6.),
                    Align2::LEFT_TOP,
                    format!("{x} m"),
                    FontId::proportional(14.5),
                    GROUND_INK,
                );
            }
        }
        if let Some(p) = &self.playback {
            // Center-of-mass trail from the last two seconds of recorded
            // frames, fading with age.
            let trail = crate::theme::scene::TRAIL;
            let span = physics::rate().saturating_mul(2).max(1);
            let first = p.tick.saturating_sub(span);
            let mut previous: Option<Pos2> = None;
            for tick in first..=p.tick {
                let Some(com) = p.center_of_mass(tick) else {
                    continue;
                };
                let point = world(com[0], com[1]);
                if let Some(from) = previous {
                    let freshness = 1.0 - (p.tick - tick) as f32 / span as f32;
                    let alpha = (freshness.clamp(0.0, 1.0) * 150.0) as u8;
                    painter.line_segment(
                        [from, point],
                        Stroke::new(
                            2.,
                            Color32::from_rgba_unmultiplied(trail.r(), trail.g(), trail.b(), alpha),
                        ),
                    );
                }
                previous = Some(point);
            }
            if let Some(com) = p.shown_center() {
                let at = world(com[0], com[1]);
                painter.circle_filled(at, 6., trail.gamma_multiply(0.15));
                painter.circle_filled(at, 3., trail);
            }
            // Soft contact shadows, darker and tighter the nearer a node is
            // to the ground under it.
            let glow = crate::assets::Art::Glow.texture(ui.ctx());
            for n in &p.nodes {
                let ground = height_at(n.pos[0], true);
                let lift = (n.pos[1] - n.radius - ground).max(0.0);
                let fade = (1.0 - lift / 0.6).clamp(0.0, 1.0);
                if fade <= 0.0 {
                    continue;
                }
                let at = world(n.pos[0], ground);
                let w = n.radius * self.zoom * (2.6 + lift * 2.0);
                painter.image(
                    glow,
                    Rect::from_center_size(at, Vec2::new(w * 2.0, (w * 0.5).max(6.0))),
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::from_black_alpha((190.0 * fade) as u8),
                );
            }
            let mut marks = FrameMarks::of(p);
            marks.arrows = self.show_forces;
            draw_creature(&painter, &p.nodes, &p.creature, origin, self.zoom, &marks);
        }
        crate::world_fx::weather(&painter, rect, cfg, clock, self.camera[0] * self.zoom);
        // Film look over the scene, under the HUD.
        crate::theme::vignette(&painter, rect, 0.55);
        crate::theme::grain(&painter, rect, clock, 0.055);
        use crate::theme::{Counter, HudLine, counter, hud_block, scene::HUD};
        // The HUD, laid out like Half-Life 2's: the creature's counters low
        // on the left like HEALTH and SUIT, the generation low on the right
        // like the ammo box with the rate as its reserve, the world as a
        // hint in the top right corner.
        let inset = 12.;
        let size = (rect.height() * 0.085).clamp(20., 32.);
        if let Some(p) = &self.playback {
            let fallen = p.fallen();
            let distance = fallen.map_or_else(|| physics::fitness(&p.nodes), |(_, d)| d);
            let left = counter(
                &painter,
                rect.left_bottom() + Vec2::new(inset, -inset),
                Align2::LEFT_BOTTOM,
                &Counter {
                    label: "Distance",
                    digits: format!("{distance:.2}"),
                    unit: "m",
                    extra: None,
                    damaged: fallen.is_some(),
                },
                size,
            );
            counter(
                &painter,
                left.right_bottom() + Vec2::new(inset, 0.),
                Align2::LEFT_BOTTOM,
                &Counter {
                    label: "Speed",
                    digits: format!("{:.2}", if fallen.is_some() { 0.0 } else { p.speed() }),
                    unit: "m/s",
                    extra: None,
                    damaged: fallen.is_some(),
                },
                size,
            );
            if let Some((tick, _)) = fallen {
                hud_block(
                    &painter,
                    rect.center_top() + Vec2::new(0., inset),
                    Align2::CENTER_TOP,
                    &[HudLine::text(
                        p.ending.sentence(
                            tick.saturating_sub(physics::settle()) as f32 * physics::dt(),
                        ),
                        16.,
                        Color32::from_rgb(255, 120, 100),
                    )],
                );
            }
            let live = self.snapshot.as_ref().map(|s| &s.config);
            let earlier = live.is_some_and(|live| live.physics_differs(&p.config));
            hud_block(
                &painter,
                rect.right_top() + Vec2::new(-inset, inset),
                Align2::RIGHT_TOP,
                &[
                    HudLine::label(if earlier {
                        "World · an earlier one"
                    } else {
                        "World"
                    }),
                    HudLine::text(world_summary(&p.config), 15., HUD),
                ],
            );
        }
        if let Some(s) = &self.snapshot {
            // A narrow view has no room beside the creature's counters, so
            // the generation moves to the top left corner.
            let (anchor, align) = if rect.width() < 760. {
                (rect.left_top() + Vec2::splat(inset), Align2::LEFT_TOP)
            } else {
                (
                    rect.right_bottom() - Vec2::splat(inset),
                    Align2::RIGHT_BOTTOM,
                )
            };
            counter(
                &painter,
                anchor,
                align,
                &Counter {
                    label: "Gen",
                    digits: s.generation.to_string(),
                    unit: "",
                    extra: Some(format!("{}/s", number(s.end_to_end.max(0.0) as usize))),
                    damaged: false,
                },
                size,
            );
        }
        let center_note = if self.playback.is_none() && self.awaiting_new_world() {
            Some("Testing in the new world...")
        } else if self.playback.is_none() {
            Some("Preparing your first population…")
        } else if self.playback.as_ref().is_some_and(|p| p.preparing) {
            Some("Preparing replay...")
        } else if self.playback.as_ref().is_some_and(|p| p.unavailable) {
            Some("The GPU did not record this replay")
        } else {
            None
        };
        if let Some(note) = center_note {
            let mut lines = vec![HudLine::text(note.to_owned(), 20., HUD)];
            if self.playback.is_none() && crate::cuda_engine::compiling_world() {
                lines.push(HudLine::text(
                    "The GPU is compiling its kernels for this world.".to_owned(),
                    15.,
                    crate::theme::scene::HUD_INK,
                ));
                lines.push(HudLine::text(
                    "A new game does this once, for up to a minute or two. Later starts are quick."
                        .to_owned(),
                    15.,
                    crate::theme::scene::HUD_INK,
                ));
            }
            hud_block(&painter, rect.center(), Align2::CENTER_CENTER, &lines);
        }
        painter.rect_stroke(
            rect,
            0,
            Stroke::new(4., theme.ink),
            egui::StrokeKind::Inside,
        );
        let mut sought = false;
        let mut race_it = None;
        if let Some(p) = &mut self.playback {
            let last_frame = p.last_frame();
            let trial_start = p.trial_start();
            let trial_frames = last_frame.saturating_sub(trial_start);
            ui.horizontal(|ui| {
                ui.label("Time");
                if trial_frames > 0 {
                    // The scrubber takes the row, less room for the clock.
                    ui.spacing_mut().slider_width = (ui.available_width() - 110.).max(120.);
                    let mut frame = p.tick.saturating_sub(trial_start).min(trial_frames);
                    let response = ui.add(
                        egui::Slider::new(&mut frame, 0..=trial_frames)
                            .show_value(false)
                            .text(""),
                    );
                    if let Some((fall_frame, _)) = p.fall
                        && trial_frames > 0
                    {
                        let fraction = fall_frame.saturating_sub(trial_start).min(trial_frames)
                            as f32
                            / trial_frames as f32;
                        let x = egui::lerp(response.rect.x_range(), fraction);
                        ui.painter().line_segment(
                            [
                                Pos2::new(x, response.rect.top() + 3.),
                                Pos2::new(x, response.rect.bottom() - 3.),
                            ],
                            Stroke::new(2., FALLEN),
                        );
                    }
                    if response.changed() {
                        p.seek(frame);
                        sought = true;
                    }
                } else {
                    ui.label("single frame");
                }
                ui.label(format!(
                    "{:.1} / {:.0} s",
                    p.elapsed_seconds(),
                    p.config.duration
                ));
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().button_padding.x = 10.;
            ui.spacing_mut().item_spacing.x = 6.;
            if ui
                .button(if self.playing {
                    "Pause  (K)"
                } else {
                    "Play  (K)"
                })
                .on_hover_text("Pause or play the replay. A click on the replay does the same.")
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("Replay").clicked()
                && let Some(p) = &mut self.playback
            {
                p.reset();
            }
            if ui
                .add_enabled(self.playback.is_some(), egui::Button::new("Family tree"))
                .on_hover_text("The ancestors of this creature, with what changed at each step")
                .clicked()
            {
                self.tab = Tab::Lineage;
            }
            if ui
                .add_enabled(self.playback.is_some(), egui::Button::new("Race it"))
                .on_hover_text("Race this creature against the champion")
                .clicked()
                && let Some(p) = &self.playback
            {
                race_it = Some((p.creature.clone(), p.config.clone()));
            }
            if ui
                .add_enabled(self.playback.is_some(), egui::Button::new("Export GIF"))
                .on_hover_text("Save an animated GIF of this replay under runs/")
                .clicked()
            {
                self.file("Export creature GIF");
            }
            if ui
                .add_enabled(self.playback.is_some(), egui::Button::new("Export JSON"))
                .on_hover_text(
                    "Save this creature as JSON under runs/, to open it again or share it",
                )
                .clicked()
            {
                self.file("Export creature JSON");
            }
            // A menu does not wrap by itself, so start a new line when it will not fit.
            if ui.available_width() < 175. {
                ui.end_row();
            }
            speed_picker(ui, &mut self.speed, "replay_speed");
        });
        if sought {
            self.playing = false;
        }
        if let Some((creature, config)) = race_it {
            self.race_picks.retain(|(pick, _)| pick.id != creature.id);
            self.race_picks.push((creature, config));
            if self.race_picks.len() > RACE_PICKS {
                self.race_picks.remove(0);
            }
            self.tab = Tab::Race;
            self.prev_tab = Tab::Race;
            self.restart_race();
        }
    }
    fn metrics(&self, ui: &mut egui::Ui) {
        let theme = self.theme();
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let history = &snapshot.history;
        // The best distance and the kinds of movement are the archive's
        // numbers now, so they change as results are absorbed. A history row
        // stands in before the first snapshot has an elite.
        let best = if snapshot.live_best.is_finite() {
            snapshot.live_best
        } else if let Some(s) = row_in_world(snapshot) {
            s.best
        } else if !history.is_empty() {
            ui.label(
                RichText::new(
                    "Testing in the new world... The kept creatures are running again under the new rules. The best distance appears here when one is kept.",
                )
                .color(theme.muted),
            );
            return;
        } else {
            ui.label(
                RichText::new(
                    "The first generation is running. Its best creature appears here as soon as one is kept.",
                )
                .color(theme.muted),
            );
            return;
        };
        let cells = if snapshot.movement_cells > 0 {
            snapshot.movement_cells
        } else {
            row_in_world(snapshot).map_or(0, Stats::moves)
        };
        // Only rows of this world count toward the gain.
        let gain = history
            .len()
            .checked_sub(10)
            .filter(|&earlier| !history[earlier].config.physics_differs(&snapshot.config))
            .map(|earlier| best - history[earlier].best);
        let population = snapshot.config.population.max(1);
        let progress = generation_progress(
            snapshot.completed,
            population,
            snapshot.running,
            snapshot.end_to_end,
        );
        let trial = format!(
            "How far the best creature travels in its {:.0} s trial. Distance is the only score.",
            snapshot.config.duration
        );
        let rate = if snapshot.running && snapshot.end_to_end > 0.0 {
            format!("{} /s", number(snapshot.end_to_end as usize))
        } else {
            "—".to_owned()
        };
        let rate_note = if snapshot.running {
            "creatures scored per second".to_owned()
        } else {
            "Paused".to_owned()
        };
        let tiles = [
            (
                "Best distance",
                format!("{:.2} m", best),
                theme.accent,
                gain.map_or_else(
                    || "so far".to_owned(),
                    |gain| format!("{gain:+.2} m in the last 10 generations"),
                ),
                trial.as_str(),
            ),
            (
                "Generation",
                snapshot.generation.to_string(),
                theme.ink,
                progress,
                "Every generation tries a whole population of new creatures.",
            ),
            (
                "Kinds of movement",
                number(cells),
                theme.ink,
                "different ways of moving kept".to_owned(),
                "Evolution keeps the best creature for each way of moving: how much of the time it touches the ground, its stride rate, its height and how many feet it uses. Each way of moving keeps one creature for every body shape and size.",
            ),
            (
                "Rate",
                rate,
                theme.cold,
                rate_note,
                "How many creatures the machine tries each second, from breeding to score.",
            ),
        ];
        // HUD tiles: a capital label, a big number that glows in the dark
        // theme, and a line of detail.
        let gap = GAP_M;
        let width = (ui.available_width() - gap * (tiles.len() - 1) as f32) / tiles.len() as f32;
        // HUD counters: the label and a line of detail on the left, the
        // glowing number on the right. Narrow tiles shrink the number and
        // wrap the detail, and every tile takes the height of the tallest.
        let value_size = (width / 8.5).clamp(26., 40.);
        let values: Vec<_> = tiles
            .iter()
            .map(|tile| {
                ui.painter().layout_no_wrap(
                    tile.1.clone(),
                    FontId::new(value_size, crate::assets::hud_bold()),
                    tile.2,
                )
            })
            .collect();
        let notes: Vec<_> = tiles
            .iter()
            .zip(&values)
            .map(|(tile, value)| {
                ui.painter().layout(
                    tile.3.clone(),
                    FontId::proportional(14.),
                    theme.muted,
                    (width - value.size().x - 44.).max(60.),
                )
            })
            .collect();
        let note_height = notes.iter().map(|g| g.size().y).fold(0., f32::max);
        let tile_height = (value_size * 1.2).max(32. + note_height) + 14.;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for (((name, _, color, _, why), note), value) in
                tiles.into_iter().zip(notes).zip(values)
            {
                let (rect, response) =
                    ui.allocate_exact_size(Vec2::new(width, tile_height), Sense::hover());
                let painter = ui.painter_at(rect);
                crate::theme::plate(ui.painter(), rect.shrink(2.), theme, theme.card, false);
                crate::theme::caps_text(
                    &painter,
                    rect.left_top() + Vec2::new(14., 11.),
                    Align2::LEFT_TOP,
                    name,
                    if width < 280. { 12. } else { 13. },
                    theme.ink,
                );
                painter.galley(rect.left_top() + Vec2::new(14., 32.), note, theme.muted);
                crate::theme::glow_text(
                    &painter,
                    Pos2::new(rect.right() - 14., rect.center().y),
                    Align2::RIGHT_CENTER,
                    value.text(),
                    FontId::new(value_size, crate::assets::hud_bold()),
                    color,
                    false,
                );
                response.on_hover_text(why);
            }
        });
    }
    fn trend(&self, ui: &mut egui::Ui, height: f32) {
        let Some(s) = &self.snapshot else { return };
        let theme = self.theme();
        let mut reset = false;
        let mut zoom = 1.0f64;
        let mut last: Option<f64> = None;
        ui.horizontal(|ui| {
            crate::theme::heading(ui, "Best distance", theme)
                .on_hover_text("Drag to pan. Double-click or Reset view fits the chart again.");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("Reset view")
                    .on_hover_text("Fit every generation and distance again")
                    .clicked()
                {
                    reset = true;
                }
                if ui.small_button("+").on_hover_text("Zoom in").clicked() {
                    zoom = 0.8;
                }
                if ui.small_button("−").on_hover_text("Zoom out").clicked() {
                    zoom = 1.25;
                }
                if ui
                    .small_button("Last 10")
                    .on_hover_text("Show the ten newest generations only")
                    .clicked()
                {
                    last = Some(10.0);
                }
                if ui
                    .small_button("Last 50")
                    .on_hover_text("Show the fifty newest generations only")
                    .clicked()
                {
                    last = Some(50.0);
                }
            });
        });
        let mut plot = Plot::new("fitness_history")
            .height(height)
            .legend(
                Legend::default()
                    .position(egui_plot::Corner::LeftTop)
                    .text_style(egui::TextStyle::Small),
            )
            .x_axis_label("Generation")
            .y_axis_label("Distance (m)")
            .allow_scroll(false);
        if reset {
            plot = plot.reset();
        }
        let live = live_point(s);
        plot.show(ui, |plot| {
            if zoom != 1.0 {
                let bounds = plot.plot_bounds();
                plot.set_plot_bounds_x(scaled_range(bounds.range_x(), zoom));
                plot.set_plot_bounds_y(scaled_range(bounds.range_y(), zoom));
            }
            if let Some(generations) = last {
                let end = live.map_or_else(
                    || s.history.last().map_or(1.0, |h| h.generation as f64 + 0.5),
                    |(generation, ..)| generation as f64 + 0.5,
                );
                plot.set_plot_bounds_x((end - generations).max(0.0)..=end);
                plot.set_auto_bounds(egui::Vec2b::new(false, true));
            }
            // The best creature and the typical kept one; the percentile
            // index follows `storage::PERCENTILES` (28 is 100, 14 is 50).
            for (i, name, color) in [(28, "Best", theme.accent), (14, "Median", theme.cold)] {
                let mut values: Vec<[f64; 2]> = s
                    .history
                    .iter()
                    .map(|h| [h.generation as f64, h.percentiles[i] as f64])
                    .collect();
                // The running generation's point moves as results arrive.
                if let Some((generation, best, median)) = live {
                    let value = if i == 28 { best } else { median };
                    if value.is_finite() {
                        values.push([generation as f64, value as f64]);
                    }
                }
                plot.line(Line::new(name, values).color(color).width(2.5));
            }
            // A vertical line and a short label where the world changed: the
            // generation that first runs in the new world. Marks come from
            // the worker's events, so one shows when the generation starts,
            // and from the history, so a loaded game still has them.
            let top = s
                .history
                .iter()
                .map(|h| h.percentiles[28] as f64)
                .chain(live.map(|(_, best, _)| best as f64))
                .fold(1.0, f64::max);
            for mark in world_marks(&s.events, &s.history) {
                plot.vline(
                    VLine::new(
                        if mark.autochange {
                            "Autochange"
                        } else {
                            "World change"
                        },
                        mark.generation as f64 - 0.5,
                    )
                    .color(theme.warn)
                    .width(1.5),
                );
                plot.text(
                    egui_plot::Text::new(
                        "World change",
                        egui_plot::PlotPoint::new(mark.generation as f64 - 0.4, top),
                        RichText::new(short_label(&mark.label))
                            .small()
                            .color(theme.warn),
                    )
                    .anchor(egui::Align2::LEFT_TOP),
                );
            }
            // Record markers extend the best line instead of duplicating it.
            // Records count again after a world change.
            let mut records: Vec<[f64; 2]> = world_records(&s.history)
                .into_iter()
                .map(|(index, best, _)| [s.history[index].generation as f64, best as f64])
                .collect();
            if let Some(record) = live_record(s) {
                records.push([s.generation as f64, record.best as f64]);
            }
            if !records.is_empty() {
                plot.points(
                    Points::new("Record", records)
                        .color(theme.record)
                        .filled(true)
                        .radius(3.5),
                );
            }
        });
    }
    fn histogram(&self, ui: &mut egui::Ui, stats: &Stats, height: f32) {
        // The range fits the distances of this generation, in about 40 bars.
        let (Some(low), Some(high)) = (
            stats.histogram.iter().map(|&(cm, _)| cm).min(),
            stats.histogram.iter().map(|&(cm, _)| cm).max(),
        ) else {
            ui.small("No distances recorded for this generation.");
            return;
        };
        let (low, high) = (low as f64 / 100.0, (high + 1) as f64 / 100.0);
        let width = ((high - low) / 40.0).max(0.01);
        let count = ((high - low) / width).ceil().max(1.0) as usize;
        let mut bins = vec![0u32; count];
        for &(cm, n) in &stats.histogram {
            let value = (cm as f64 + 0.5) / 100.;
            let index = (((value - low) / width).floor() as usize).min(count - 1);
            bins[index] += n;
        }
        let bars = bins
            .iter()
            .enumerate()
            .map(|(i, &n)| Bar::new(low + (i as f64 + 0.5) * width, n as f64).width(width * 0.85))
            .collect();
        Plot::new("histogram")
            .height(height)
            .x_axis_label("Distance (m)")
            .allow_scroll(false)
            .show(ui, |plot| {
                plot.bar_chart(
                    BarChart::new("Creatures", bars)
                        .color(self.theme().accent.gamma_multiply(0.65)),
                );
            });
        if stats.failed > 0 {
            ui.small(format!(
                "{} failed trials are not shown",
                number(stats.failed)
            ));
        }
    }
    fn population(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.heading("Ways of moving");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.choice(&mut self.archive_view, ArchiveView::Map, "Map")
                    .on_hover_text(
                        "Watch evolution fill the ways of moving. Cells are colored by distance.",
                    );
                ui.choice(&mut self.archive_view, ArchiveView::Cards, "Cards");
                ui.choice(&mut self.archive_view, ArchiveView::Islands, "Islands")
                    .on_hover_text(
                        "The four isolated islands and the hub, each with its best creature, its top elites and its migrants.",
                    );
            });
        });
        ui.label(RichText::new("Click a creature to replay it.").color(theme.muted));
        if self.archive_view == ArchiveView::Islands {
            self.islands_view(ui);
            return;
        }
        if self.archive_view == ArchiveView::Map {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Body height").small().color(theme.muted));
                egui::ComboBox::from_id_salt("map_height")
                    .selected_text(self.map_height.map_or("All".to_owned(), height_bin_label))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.map_height, None, "All");
                        for bin in 0..MAP_BINS[3] {
                            ui.selectable_value(
                                &mut self.map_height,
                                Some(bin),
                                height_bin_label(bin),
                            );
                        }
                    });
                ui.label(RichText::new("Feet").small().color(theme.muted));
                egui::ComboBox::from_id_salt("map_feet")
                    .selected_text(self.map_feet.map_or("All".to_owned(), feet_bin_label))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.map_feet, None, "All");
                        for bin in 0..MAP_BINS[4] {
                            ui.selectable_value(&mut self.map_feet, Some(bin), feet_bin_label(bin));
                        }
                    });
                ui.label(RichText::new("Shape").small().color(theme.muted));
                egui::ComboBox::from_id_salt("map_shape")
                    .selected_text(
                        self.map_shape
                            .map_or("All", |class| CLASSES.shape_names[class]),
                    )
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.map_shape, None, "All");
                        for (class, name) in CLASSES.shape_names.iter().enumerate() {
                            ui.selectable_value(&mut self.map_shape, Some(class), *name)
                                .on_hover_text(CLASSES.shape_about(class));
                        }
                    });
                ui.label(RichText::new("Size").small().color(theme.muted));
                egui::ComboBox::from_id_salt("map_size")
                    .selected_text(
                        self.map_size
                            .map_or("All", |class| CLASSES.size_names[class]),
                    )
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.map_size, None, "All");
                        for (class, name) in CLASSES.size_names.iter().enumerate() {
                            ui.selectable_value(&mut self.map_size, Some(class), *name)
                                .on_hover_text(CLASSES.size_about(class));
                        }
                    });
            });
        }
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(format!(
                    "{} creatures in {} ways of moving, sorted by",
                    number(snapshot.archive_cells),
                    number(snapshot.movement_cells)
                ))
                .small()
                .color(theme.muted),
            );
            for (name, why) in [
                (
                    "ground contact",
                    "How much of the trial the creature keeps its nodes on the ground.",
                ),
                (
                    "stride rate",
                    "How many up-and-down body swings the gait makes per second.",
                ),
                (
                    "body height",
                    "The average height of the body above the ground during the trial.",
                ),
                (
                    "feet",
                    "Nodes that touch the ground and lift off again. A node dragged along the ground never lifts, so it is not a foot.",
                ),
            ] {
                ui.label(RichText::new(name).small().color(theme.ink))
                    .on_hover_text(why);
            }
        });
        if self.archive_view == ArchiveView::Cards {
            let mut filter = self.card_filter;
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 4.;
                ui.label(RichText::new("Feet").small().color(theme.muted));
                if ui.pick(filter.feet.is_none(), "All").clicked() {
                    filter.feet = None;
                }
                for bin in 0..MAP_BINS[4] as u8 {
                    if ui
                        .pick(
                            filter.feet == Some(bin),
                            feet_bin_label(usize::from(bin))
                                .replace(" feet", "")
                                .replace(" foot", ""),
                        )
                        .clicked()
                    {
                        filter.feet = Some(bin);
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 4.;
                ui.label(RichText::new("Size").small().color(theme.muted));
                if ui.pick(filter.size.is_none(), "All").clicked() {
                    filter.size = None;
                }
                for (class, name) in CLASSES.size_names.iter().enumerate() {
                    if ui
                        .pick(filter.size == Some(class as u8), *name)
                        .on_hover_text(CLASSES.size_about(class))
                        .clicked()
                    {
                        filter.size = Some(class as u8);
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 4.;
                ui.label(RichText::new("Shape").small().color(theme.muted));
                if ui.pick(filter.shape.is_none(), "All").clicked() {
                    filter.shape = None;
                }
                for (class, name) in CLASSES.shape_names.iter().enumerate() {
                    if ui
                        .pick(filter.shape == Some(class as u8), *name)
                        .on_hover_text(CLASSES.shape_about(class))
                        .clicked()
                    {
                        filter.shape = Some(class as u8);
                    }
                }
            });
            self.card_filter = filter;
        }
        let mut selected = None;
        let mut map_click = None;
        if self.archive_view == ArchiveView::Map {
            let empty = Vec::new();
            let cells = snapshot.map.as_deref().unwrap_or(&empty);
            map_click = paint_archive_map(
                ui,
                cells,
                [
                    self.map_height,
                    self.map_feet,
                    self.map_shape,
                    self.map_size,
                ],
                theme,
            );
        } else {
            self.card_grid(ui, &mut selected);
        }
        if let Some(id) = map_click {
            self.worker.send(Command::Select(id));
        }
        if let Some((creature, config)) = selected {
            self.select(creature, config);
        }
    }
    /// The island archives (four isolated islands, then the hub) in two
    /// columns. Each card has a fixed size and
    /// fixed places for its parts, so numbers change without moving anything.
    fn islands_view(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "Islands never mix. The hub gets copies every {} generations.",
                    crate::storage::MIGRATION_INTERVAL
                ))
                .color(theme.muted),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("How evolution works")
                    .on_hover_text("Shows how the islands, the emitters and migration fit together")
                    .clicked()
                {
                    self.schematic_open = true;
                }
            });
        });
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        if snapshot
            .islands
            .iter()
            .all(|island| island.leader.is_none())
        {
            ui.add_space(8.);
            ui.label(
                RichText::new(
                    "The islands fill as the first creatures are kept. Their creatures appear here.",
                )
                .color(theme.muted),
            );
            return;
        }
        let config = snapshot.config.clone();
        let generation = snapshot.generation;
        let strangest = snapshot.strangest.clone();
        if let Some(creature) = strangest
            && ui
                .button("Strangest body")
                .on_hover_text(
                    "Replay the island creature whose body is the most unlike the others",
                )
                .clicked()
        {
            self.select(creature, config.clone());
            return;
        }
        let Some(snapshot) = self.snapshot.as_ref() else {
            return;
        };
        let islands = snapshot.islands.clone();
        let wild_wins = snapshot.wild_wins.clone();
        let migration = snapshot.migration.clone();
        let shown = self.playback.as_ref().map(|p| p.creature.id);
        let width = (ui.available_width() - ISLAND_GAP) / 2.;
        let mut selected = None;
        let mut wild_pick = None;
        egui::ScrollArea::vertical()
            .id_salt("islands_grid")
            .show(ui, |ui| {
                let main = islands.len().min(crate::qd::MAIN_ISLANDS);
                for pair in islands[..main].chunks(2).enumerate() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = ISLAND_GAP;
                        for (offset, island) in pair.1.iter().enumerate() {
                            let index = pair.0 * 2 + offset;
                            let (rect, _) = ui.allocate_exact_size(
                                Vec2::new(width, ISLAND_HEIGHT),
                                Sense::hover(),
                            );
                            let hit = paint_island(
                                ui,
                                rect,
                                index,
                                island,
                                migration.as_ref(),
                                generation,
                                shown,
                                theme,
                            );
                            if let Some(creature) = hit {
                                selected = Some(creature);
                            }
                        }
                    });
                    ui.add_space(ISLAND_GAP);
                }
                if islands.len() > main
                    && let Some(pick) =
                        wild_tiles(ui, &islands[main..], &wild_wins, &config, shown, &theme)
                {
                    wild_pick = Some(pick);
                }
            });
        if let Some((creature, world)) = wild_pick {
            self.select(creature, world);
        } else if let Some(creature) = selected {
            self.select(creature, config);
        }
    }
    /// The archive cards: the ranked list the UI holds, filtered here, so no
    /// card waits for data or moves while the player looks. A newer list
    /// replaces it only when the player asks for it (or opens the tab).
    fn card_grid(&mut self, ui: &mut egui::Ui, selected: &mut Option<(Creature, Config)>) {
        let theme = self.theme();
        let (generation, archive_size) = self
            .snapshot
            .as_ref()
            .map_or((0, 0), |s| (s.generation, s.archive_size));
        let waiting = self
            .cards_requested
            .is_some_and(|at| at.elapsed() < Duration::from_secs(2));
        // A list scored in an earlier world is never shown.
        let world = self.snapshot.as_ref().map(|s| s.config.clone());
        if self.cards.as_ref().is_some_and(|list| {
            world
                .as_ref()
                .is_some_and(|w| list.config.physics_differs(w))
        }) {
            self.cards = None;
        }
        let empty = self.cards.as_ref().is_none_or(|list| list.cards.is_empty());
        if empty && archive_size > 0 && !waiting {
            // Nothing held yet (or the request got lost): ask again.
            self.request_cards();
        }
        let Some(list) = self.cards.clone() else {
            let changed = self
                .snapshot
                .as_ref()
                .is_some_and(|s| !s.history.is_empty() && row_in_world(s).is_none());
            ui.label(
                RichText::new(if changed {
                    "Testing in the new world... Creatures appear here as they are kept."
                } else {
                    "The first generation is running. Its creatures appear here as they are kept."
                })
                .color(theme.muted),
            );
            return;
        };
        if list.generation < generation {
            let mut refresh = false;
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("Showing generation {}.", list.generation))
                        .small()
                        .color(theme.muted),
                );
                refresh = ui
                    .small_button(format!("Show latest (generation {generation})"))
                    .on_hover_text(
                        "The cards stay still while you look; this loads the newest ranking",
                    )
                    .clicked();
            });
            if refresh {
                self.request_cards();
            }
        }
        let filter = self.card_filter;
        let visible: Vec<&crate::worker::Card> = list
            .cards
            .iter()
            .filter(|card| filter.shows(card))
            .collect();
        if visible.is_empty() {
            ui.label(
                RichText::new(if list.cards.is_empty() {
                    "No creatures kept yet."
                } else {
                    "No kept creature matches these filters."
                })
                .color(theme.muted),
            );
            return;
        }
        let columns = (ui.available_width() / 190.).floor().max(2.) as usize;
        let width = (ui.available_width() - (columns - 1) as f32 * 10.) / columns as f32;
        let shown = self.playback.as_ref().map(|p| p.creature.id);
        egui::ScrollArea::vertical()
            .id_salt("population_grid")
            .show_rows(ui, 162., visible.len().div_ceil(columns), |ui, rows| {
                for row in rows {
                    ui.horizontal(|ui| {
                        for card in visible.iter().skip(row * columns).take(columns) {
                            let (rect, response) =
                                ui.allocate_exact_size(Vec2::new(width, 152.), Sense::click());
                            paint_card(
                                ui.painter(),
                                card,
                                rect,
                                response.hovered() || shown == Some(card.creature.id),
                                theme,
                            );
                            if response.clicked() {
                                *selected = Some(card.replay_of(&list.config));
                            }
                            response.on_hover_text(format!(
                                "{}\n{} nodes, {} bones, {} muscles{}\n{}\n{}\nClick to replay",
                                species_name(&card.creature),
                                card.creature.nodes.len(),
                                card.creature.bones.len(),
                                card.creature.muscles.len(),
                                card.descriptor.map_or(String::new(), |d| {
                                    let niche = d.niche().0;
                                    format!(
                                        " · {} {} body",
                                        CLASSES.shape_names[usize::from(niche[2])]
                                            .to_lowercase(),
                                        CLASSES.size_names[usize::from(niche[5])].to_lowercase()
                                    )
                                }),
                                card.emitter.map_or("First generation".to_owned(), |emitter| {
                                    format!("Born {}", origin_words(emitter))
                                }),
                                card.descriptor.map_or_else(
                                    || "Trial done".to_owned(),
                                    |d| format!(
                                        "On the ground {:.0}% of the time · {:.2} strides/s · {:.2} m tall · {:.0} feet",
                                        d.ground_contact * 100.0,
                                        d.gait_frequency,
                                        d.mean_height,
                                        d.feet
                                    ),
                                )
                            ));
                        }
                    });
                }
            });
    }
    /// Replays the best creature recorded for one history entry, through the
    /// same preview path as an archive card click.
    fn replay_history_holder(&mut self, index: usize) {
        let Some((creature, config)) = self.snapshot.as_ref().and_then(|snapshot| {
            let stats = snapshot.history.get(index)?;
            Some((stats.representatives.last()?.clone(), stats.config.clone()))
        }) else {
            return;
        };
        self.select(creature, config);
        self.tab = Tab::Overview;
    }
    /// Replays the champion now, like a record's Replay button.
    fn replay_champion(&mut self) {
        if let Some((creature, config)) = self.champion() {
            self.select(creature, config);
            self.tab = Tab::Overview;
        }
    }
    /// The lines of the event feed, newest first: the worker's events (world
    /// changes, autochange, catastrophes, saves) and the records in the history.
    fn feed_items(&self) -> Vec<FeedItem> {
        let Some(snapshot) = &self.snapshot else {
            return Vec::new();
        };
        let theme = self.theme();
        let history = &snapshot.history;
        let row = |generation: u32| history.iter().rev().find(|s| s.generation == generation);
        let mut items = Vec::new();
        for event in snapshot.events.iter() {
            let mut text = event.text.clone();
            let (color, action) = match event.kind {
                EventKind::Catastrophe => (
                    theme.warn,
                    (snapshot.fossils > 0).then_some(FeedAction::Undo),
                ),
                EventKind::World | EventKind::Autochange => {
                    if let Some(before) = event.generation.checked_sub(1).and_then(row) {
                        if let Some(after) = row(event.generation) {
                            text.push_str(&format!(
                                " Best {:.2} m before, {:.2} m after one generation.",
                                before.best, after.best
                            ));
                        } else if event.generation == snapshot.generation
                            && snapshot.live_best.is_finite()
                        {
                            text.push_str(&format!(
                                " Best {:.2} m before, {:.2} m so far in this generation.",
                                before.best, snapshot.live_best
                            ));
                        }
                    }
                    if event.kind == EventKind::Autochange {
                        text.insert_str(0, "Autochange: ");
                    }
                    (theme.accent, None)
                }
                EventKind::Gpu => (theme.warn, None),
                _ => (theme.muted, None),
            };
            items.push(FeedItem {
                generation: event.generation,
                text,
                color,
                action,
            });
        }
        let records = world_records(history);
        for &(index, best, first_in_world) in &records {
            let stats = &history[index];
            let name = stats
                .representatives
                .last()
                .map(species_name)
                .unwrap_or_default();
            let text = if index == 0 {
                format!("First generation: best {best:.2} m, {name}.")
            } else if first_in_world {
                format!("Best in the new world: {best:.2} m, {name}.")
            } else {
                format!("New record: {best:.2} m, {name}.")
            };
            items.push(FeedItem {
                generation: stats.generation,
                text,
                color: theme.ink,
                action: Some(FeedAction::Replay(index)),
            });
        }
        if let Some(record) = live_record(snapshot) {
            let name = snapshot
                .champion
                .as_ref()
                .map(|champion| species_name(&champion.0))
                .unwrap_or_default();
            let best = record.best;
            items.push(FeedItem {
                generation: snapshot.generation,
                text: if record.first_ever {
                    format!("First generation: best {best:.2} m, {name}.")
                } else if record.first_in_world {
                    format!("Best in the new world: {best:.2} m, {name}.")
                } else {
                    format!("New record: {best:.2} m, {name}.")
                },
                color: theme.ink,
                action: Some(FeedAction::ReplayChampion),
            });
        }
        // A stall: no record in this world for a while. The feed suggests a
        // harder world instead of changing the search silently.
        // Only a record of the live world counts: after a world change the old
        // records say nothing about a stall.
        if let (Some(last), Some(&(index, _, _))) = (history.last(), records.last())
            && !history[index].config.physics_differs(&snapshot.config)
        {
            let since = last.generation.saturating_sub(history[index].generation);
            if since >= STALL_GENERATIONS
                && live_record(snapshot).is_none()
                && let Some((effect, level)) = stall_suggestion(&self.config)
            {
                let effect_ref = &crate::environment::EFFECTS[effect];
                items.push(FeedItem {
                    generation: last.generation,
                    text: format!(
                        "No new record for {since} generations. A new world can open new ways of moving: try {} {}.",
                        effect_ref.name, effect_ref.levels[level]
                    ),
                    color: theme.accent,
                    action: Some(FeedAction::Try(effect, level)),
                });
            }
        }
        // A diversity collapse: the effective clades of the archive fell by a
        // quarter within 50 generations of this world. The feed suggests a
        // new world and never presses it (Lehman and Miikkulainen, 2015: a
        // change restarts radiation).
        if let Some(last) = history.last() {
            let window: Vec<&Stats> = history
                .iter()
                .rev()
                .take(COLLAPSE_WINDOW)
                .take_while(|h| !h.config.physics_differs(&snapshot.config))
                .collect();
            let peak = window.iter().map(|h| h.clades).fold(0.0f32, f32::max);
            if window.len() >= 10
                && peak > 4.0
                && last.clades < (1.0 - COLLAPSE_SHARE) * peak
                && let Some((effect, level)) = stall_suggestion(&self.config)
            {
                let effect_ref = &crate::environment::EFFECTS[effect];
                items.push(FeedItem {
                    generation: last.generation,
                    text: format!(
                        "Lineages are dying out: {:.1} effective clades, down from {peak:.1}. A new world can start a new radiation: try {} {}.",
                        last.clades, effect_ref.name, effect_ref.levels[level]
                    ),
                    color: theme.accent,
                    action: Some(FeedAction::Try(effect, level)),
                });
            }
        }
        // The wild island whose migrants took the most hub cells: its world
        // breeds bodies that also do well in yours. The feed offers it as a
        // world to try (Wang et al., 2019, POET transfer) and never sets it.
        if let Some(last) = history.last()
            && let Some((island, &wins)) = snapshot
                .wild_wins
                .iter()
                .enumerate()
                .skip(crate::qd::MAIN_ISLANDS)
                .max_by_key(|&(i, &w)| (w, std::cmp::Reverse(i)))
            && wins >= 3
        {
            let w = island - crate::qd::MAIN_ISLANDS;
            let worlds = crate::environment::wild_levels(snapshot.config.seed);
            if let Some(levels) = worlds.get(w) {
                items.push(FeedItem {
                    generation: last.generation,
                    text: format!(
                        "Wild island W{} sent the most creatures that won hub cells ({wins}). Its world: {}.",
                        w + 1,
                        crate::environment::wild_name(levels)
                    ),
                    color: theme.accent,
                    action: Some(FeedAction::Wild(w)),
                });
            }
        }
        // Newest first; the sort is stable, so events of one generation keep
        // their order.
        items.reverse();
        items.sort_by_key(|item| std::cmp::Reverse(item.generation));
        items.truncate(60);
        items
    }
    /// The event feed: what happened, newest first, with a button to replay
    /// a record holder or undo a catastrophe.
    fn feed(&mut self, ui: &mut egui::Ui, height: f32) {
        let theme = self.theme();
        crate::theme::heading(ui, "What happened", theme);
        let items = self.feed_items();
        let mut chosen = None;
        egui::ScrollArea::vertical()
            .id_salt("event_feed")
            .max_height(height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if items.is_empty() {
                    ui.label(
                        RichText::new("Records, world changes and catastrophes appear here.")
                            .small()
                            .color(theme.muted),
                    );
                }
                for item in &items {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.y = 2.;
                        ui.label(
                            RichText::new(format!("Gen {}", item.generation))
                                .small()
                                .color(theme.muted),
                        );
                        ui.label(RichText::new(&item.text).small().color(item.color));
                        if let Some(action) = item.action {
                            let label = match action {
                                FeedAction::Replay(_) | FeedAction::ReplayChampion => "Replay",
                                FeedAction::Undo => "Undo",
                                FeedAction::Try(..) | FeedAction::Wild(_) => "Try it",
                            };
                            if ui.small_button(label).clicked() {
                                chosen = Some(action);
                            }
                        }
                    });
                }
            });
        match chosen {
            Some(FeedAction::Replay(index)) => self.replay_history_holder(index),
            Some(FeedAction::ReplayChampion) => self.replay_champion(),
            Some(FeedAction::Undo) => self.worker.send(Command::UndoMeteor),
            Some(FeedAction::Try(effect, level)) => {
                crate::environment::EFFECTS[effect].set_level(&mut self.config, level);
                self.worker.send(Command::Configure(self.config.clone()));
                self.config_sent = Some(Instant::now());
            }
            Some(FeedAction::Wild(w)) => {
                let worlds = crate::environment::wild_levels(self.config.seed);
                if let Some(levels) = worlds.get(w) {
                    for effect in crate::environment::EFFECTS
                        .iter()
                        .filter(|e| e.name != "Autochange environment")
                    {
                        effect.set_level(&mut self.config, effect.calm);
                    }
                    for &(e, level) in levels {
                        crate::environment::EFFECTS[e].set_level(&mut self.config, level);
                    }
                    self.worker.send(Command::Configure(self.config.clone()));
                    self.config_sent = Some(Instant::now());
                }
            }
            None => {}
        }
    }
    /// Every record, newest first: generation, distance, species and the
    /// world it was set in, with a Replay button. Records count again after
    /// each world change, like the feed and the chart.
    fn records_list(&mut self, ui: &mut egui::Ui) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let theme = self.theme();
        let records = world_records(&snapshot.history);
        let live = live_record(snapshot);
        crate::theme::heading(ui, "Records", theme);
        if records.is_empty() && live.is_none() {
            ui.label(
                RichText::new("No records yet. The first best creature lands here.")
                    .small()
                    .color(theme.muted),
            );
            return;
        }
        let mut chosen = None;
        let mut chosen_live = false;
        egui::ScrollArea::vertical()
            .id_salt("records_list")
            .max_height(200.)
            .show(ui, |ui| {
                egui::Grid::new("records_grid")
                    .num_columns(4)
                    .spacing([14., 4.])
                    .striped(true)
                    .show(ui, |ui| {
                        if let Some(record) = &live {
                            ui.label(
                                RichText::new(format!("Gen {}", snapshot.generation))
                                    .small()
                                    .color(theme.muted),
                            );
                            ui.label(RichText::new(format!("{:.2} m", record.best)).strong());
                            let name = snapshot
                                .champion
                                .as_ref()
                                .map(|champion| species_name(&champion.0))
                                .unwrap_or_default();
                            ui.label(
                                RichText::new(format!(
                                    "{name} · {}{}",
                                    world_summary(&snapshot.config),
                                    if record.first_in_world {
                                        " (new world)"
                                    } else {
                                        ""
                                    }
                                ))
                                .small()
                                .color(theme.muted),
                            );
                            if ui.small_button("Replay").clicked() {
                                chosen_live = true;
                            }
                            ui.end_row();
                        }
                        for &(index, best, first) in records.iter().rev() {
                            let stats = &snapshot.history[index];
                            ui.label(
                                RichText::new(format!("Gen {}", stats.generation))
                                    .small()
                                    .color(theme.muted),
                            );
                            ui.label(RichText::new(format!("{best:.2} m")).strong());
                            ui.label(
                                RichText::new(format!(
                                    "{} · {}{}",
                                    stats
                                        .representatives
                                        .last()
                                        .map(species_name)
                                        .unwrap_or_default(),
                                    world_summary(&stats.config),
                                    if first && index > 0 {
                                        " (new world)"
                                    } else {
                                        ""
                                    }
                                ))
                                .small()
                                .color(theme.muted),
                            );
                            if ui.small_button("Replay").clicked() {
                                chosen = Some(index);
                            }
                            ui.end_row();
                        }
                    });
            });
        if chosen_live {
            self.replay_champion();
        } else if let Some(index) = chosen {
            self.replay_history_holder(index);
        }
    }
    /// Builds race lanes from the top archive cards once the first page arrives.
    fn maybe_build_race(&mut self) {
        if !self.race_pending {
            return;
        }
        if self.race_picks.is_empty() {
            // The top five come with a ranked archive; ask again while none
            // has arrived with creatures in it.
            let kept = self.snapshot.as_ref().is_some_and(|s| s.archive_size > 0);
            let waiting = self
                .cards_requested
                .is_some_and(|at| at.elapsed() < Duration::from_secs(2));
            if kept && !waiting {
                self.request_cards();
            }
            return;
        }
        // The player's picks against the champion.
        let mut lanes: Vec<RaceLane> = Vec::new();
        if let Some((creature, config)) = self.champion()
            && self
                .race_picks
                .iter()
                .all(|(pick, _)| pick.id != creature.id)
        {
            lanes.push(RaceLane {
                label: "champion".to_owned(),
                playback: Playback::new(creature, config),
            });
        }
        for (creature, config) in &self.race_picks {
            lanes.push(RaceLane {
                label: "your pick".to_owned(),
                playback: Playback::new(creature.clone(), config.clone()),
            });
        }
        lanes.sort_by(|a, b| b.playback.distance.total_cmp(&a.playback.distance));
        self.race = lanes;
        self.race_pending = false;
        self.race_camera = 0.0;
    }
    /// The top five kept creatures of a freshly ranked archive race.
    fn build_top_race(&mut self, list: &crate::worker::CardList) {
        let mut lanes: Vec<RaceLane> = list
            .cards
            .iter()
            .filter(|card| card.descriptor.is_some() && card.score.is_finite())
            .take(5)
            .map(|card| RaceLane {
                label: format!("archive rank {}", card.rank + 1),
                playback: {
                    let (creature, config) = card.replay_of(&list.config);
                    Playback::new(creature, config)
                },
            })
            .collect();
        // Lanes run in the order their replays finish, so the standings end
        // the way the lanes are listed.
        lanes.sort_by(|a, b| b.playback.distance.total_cmp(&a.playback.distance));
        if lanes.is_empty() {
            return;
        }
        self.race = lanes;
        self.race_pending = false;
        self.race_camera = 0.0;
    }
    /// Clears the race and asks for a fresh set of top elites.
    fn restart_race(&mut self) {
        self.race.clear();
        self.race_pending = true;
        self.race_camera = 0.0;
        if self.race_picks.is_empty() {
            self.request_cards();
        }
    }
    /// Asks the worker for the ranked archive; it arrives with a snapshot.
    fn request_cards(&mut self) {
        self.cards_requested = Some(Instant::now());
        self.worker.send(Command::Cards);
    }
    /// Runs when the player opens a tab: a fresh race, or the latest
    /// ranked archive for Ways of moving.
    fn opened_tab(&mut self, tab: Tab) {
        match tab {
            Tab::Race => self.restart_race(),
            Tab::Population => self.request_cards(),
            _ => {}
        }
    }
    /// Full ancestor list of the selected creature, one row per generation.
    fn lineage_view(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.heading("Lineage");
            ui.label(
                RichText::new(
                    "Ancestors of the creature on screen, newest first. Click one to replay it.",
                )
                .color(theme.muted),
            );
        });
        if self.lineage.is_empty() {
            ui.add_space(GAP_L);
            ui.label(
                RichText::new(if self.playback.is_none() {
                    "Pick a creature in Ways of moving to trace its ancestry."
                } else if self.lineage_pending {
                    "Requesting ancestors…"
                } else {
                    "No recorded ancestors for this creature yet. Evolve a few generations or pick a kept creature."
                })
                .color(theme.muted),
            );
            return;
        }
        if let Some(p) = &self.playback {
            ui.label(format!(
                "{} · {} nodes, {} bones, {} muscles · {} recorded ancestors",
                species_name(&p.creature),
                p.creature.nodes.len(),
                p.creature.bones.len(),
                p.creature.muscles.len(),
                self.lineage.len()
            ));
        }
        let shown = self.playback.as_ref().map(|p| p.creature.id);
        let mut chosen = None;
        egui::ScrollArea::vertical()
            .id_salt("lineage_view")
            .show(ui, |ui| {
                for k in 0..self.lineage.len() {
                    let step = &self.lineage[k];
                    if paint_lineage_tile(
                        ui,
                        step,
                        self.lineage.get(k + 1),
                        true,
                        shown == Some(step.creature.id),
                        theme,
                        Vec2::new(ui.available_width(), 120.),
                    ) {
                        chosen = Some(k);
                    }
                    ui.add_space(GAP_M);
                }
            });
        if let Some(k) = chosen {
            let config = self.world_of_generation(self.lineage[k].generation);
            self.select_ancestor(self.lineage[k].creature.clone(), config);
            self.tab = Tab::Overview;
        }
    }
    /// The top archived elites running side by side, with live standings.
    fn race_view(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.heading("Race");
            ui.label(
                RichText::new("The fastest kept creatures run their trials side by side.")
                    .color(theme.muted),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.race_picks.is_empty() {
                    if ui
                        .button("New race")
                        .on_hover_text("Take the current top five kept creatures")
                        .clicked()
                    {
                        self.restart_race();
                    }
                } else {
                    if ui
                        .button("Top five")
                        .on_hover_text("Forget your picks and race the top five kept creatures")
                        .clicked()
                    {
                        self.race_picks.clear();
                        self.restart_race();
                    }
                    ui.label(
                        RichText::new(
                            "Your picks against the champion. Race it under any replay adds one.",
                        )
                        .small()
                        .color(theme.muted),
                    );
                }
            });
        });
        ui.horizontal(|ui| {
            if ui
                .button(if self.playing {
                    "Pause  (K)"
                } else {
                    "Play  (K)"
                })
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("Replay").clicked() {
                for lane in &mut self.race {
                    lane.playback.reset();
                }
                self.race_camera = 0.0;
            }
            speed_picker(ui, &mut self.speed, "race_speed");
        });
        if self.race.is_empty() {
            ui.add_space(GAP_L);
            let waiting = self.race_pending
                && self
                    .snapshot
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.archive_size > 0);
            ui.label(
                RichText::new(if waiting {
                    "Loading the fastest kept creatures…"
                } else {
                    "No archived creatures yet. Run a generation, then start a new race."
                })
                .color(theme.muted),
            );
            return;
        }
        let distances: Vec<f32> = self
            .race
            .iter()
            .map(|lane| lane.playback.current_distance())
            .collect();
        let leader = distances
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |(i, _)| i);
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), ui.available_height().max(260.)),
            Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        let board_width = 250.0_f32.min(rect.width() * 0.3);
        let lanes_rect = Rect::from_min_max(
            rect.min,
            Pos2::new(rect.right() - board_width - 12., rect.bottom()),
        );
        // The default zoom follows the lanes' median body height, and the
        // tallest body still has to fit its lane.
        let lane_height = lanes_rect.height() / self.race.len().max(1) as f32;
        let mut heights: Vec<f32> = self.race.iter().map(|lane| lane.playback.height).collect();
        heights.sort_by(f32::total_cmp);
        let median = heights.get(heights.len() / 2).copied().unwrap_or(1.0);
        let tallest = heights.last().copied().unwrap_or(1.0);
        let zoom = fit_zoom(median, lane_height)
            .min(lane_height * 0.8 / tallest.max(0.1))
            .clamp(34.0, 300.0);
        let visible = lanes_rect.width() / zoom;
        // The leader's averaged center of mass, so its stride does not shake
        // the view; the easing below smooths a change of leader.
        // The start line sits a little in from the lane's left edge.
        let target = (self.race[leader].playback.camera_x() - visible * 0.6).max(-visible * 0.08);
        let dt = ui.ctx().input(|i| i.stable_dt).clamp(0.0, 0.1);
        self.race_camera += (target - self.race_camera) * (dt * 4.0).min(1.0);
        let camera = self.race_camera;
        for (i, lane) in self.race.iter().enumerate() {
            let lane_rect = Rect::from_min_max(
                Pos2::new(
                    lanes_rect.left(),
                    lanes_rect.top() + i as f32 * lane_height + 2.,
                ),
                Pos2::new(
                    lanes_rect.right(),
                    lanes_rect.top() + (i + 1) as f32 * lane_height - 2.,
                ),
            );
            let is_leader = i == leader;
            // Each lane is a strip of the world: its sky and skyline over a
            // street, the leader's lane framed in orange.
            let ground = lane_rect.bottom() - 16.;
            let lane_painter = painter.with_clip_rect(lane_rect);
            let lane_config = &lane.playback.config;
            let clock = lane.playback.tick as f32 / physics::rate() as f32;
            crate::world_fx::backdrop(
                &lane_painter,
                lane_rect,
                ground,
                camera * zoom + i as f32 * 900.,
                clock,
                lane_config,
                None,
            );
            let span = [
                Pos2::new(lane_rect.left(), ground),
                Pos2::new(lane_rect.right(), ground),
            ];
            crate::world_fx::ground_body(
                &lane_painter,
                lane_rect,
                lane_config,
                &span,
                &[camera, camera + lane_rect.width() / zoom],
                zoom,
            );
            // A tick about every 150 px: 0.5, 1, 2, 5 or 10 m.
            let step = [0.5f32, 1.0, 2.0, 5.0, 10.0]
                .into_iter()
                .find(|step| step * zoom >= 150.0)
                .unwrap_or(10.0);
            let mut x = (camera / step).ceil() * step;
            while x <= camera + visible {
                let px = lane_rect.left() + (x - camera) * zoom;
                lane_painter.line_segment(
                    [
                        Pos2::new(px, ground),
                        Pos2::new(px, lane_rect.bottom() - 5.),
                    ],
                    Stroke::new(1., GROUND_INK),
                );
                lane_painter.line_segment(
                    [Pos2::new(px, lane_rect.top()), Pos2::new(px, ground)],
                    Stroke::new(1., crate::theme::scene::GRID),
                );
                if i == 0 {
                    lane_painter.text(
                        Pos2::new(px + 3., ground - 2.),
                        Align2::LEFT_BOTTOM,
                        if step < 1.0 {
                            format!("{x:.1} m")
                        } else {
                            format!("{x:.0} m")
                        },
                        FontId::proportional(14.5),
                        GROUND_INK,
                    );
                }
                x += step;
            }
            let origin = Pos2::new(lane_rect.left() - camera * zoom, ground);
            let playback = &lane.playback;
            let marks = FrameMarks::of(playback);
            draw_creature(
                &lane_painter,
                &playback.nodes,
                &playback.creature,
                origin,
                zoom,
                &marks,
            );
            crate::theme::vignette(&lane_painter, lane_rect, 0.35);
            use crate::theme::{
                HudLine, hud_block,
                scene::{HUD, HUD_DIM},
            };
            hud_block(
                &lane_painter,
                lane_rect.left_top() + Vec2::splat(6.),
                Align2::LEFT_TOP,
                &[
                    HudLine::text(
                        format!("{}. {}", i + 1, species_name(&lane.playback.creature)),
                        15.,
                        if is_leader {
                            HUD
                        } else {
                            crate::theme::scene::HUD_INK
                        },
                    ),
                    HudLine::text(
                        format!(
                            "finishes at {:.2} m · {}",
                            lane.playback.distance, lane.label
                        ),
                        12.,
                        HUD_DIM,
                    ),
                ],
            );
            let mut right = vec![HudLine::value(
                format!("{:.2} m", distances[i]),
                18.,
                if is_leader {
                    HUD
                } else {
                    crate::theme::scene::HUD_INK
                },
            )];
            if playback.fallen().is_some() {
                right.push(HudLine::text(
                    playback.ending.short().to_owned(),
                    13.,
                    FALLEN,
                ));
            }
            hud_block(
                &lane_painter,
                lane_rect.right_top() + Vec2::new(-6., 6.),
                Align2::RIGHT_TOP,
                &right,
            );
            painter.rect_stroke(
                lane_rect,
                2,
                Stroke::new(
                    if is_leader { 2. } else { 1. },
                    if is_leader {
                        theme.accent
                    } else {
                        theme.card_border
                    },
                ),
                egui::StrokeKind::Inside,
            );
        }
        let board = Rect::from_min_max(
            Pos2::new(lanes_rect.right() + 12., rect.top()),
            rect.right_bottom(),
        );
        crate::theme::plate(&painter, board, theme, theme.card, false);
        crate::theme::caps_text(
            &painter,
            board.left_top() + Vec2::new(10., 10.),
            Align2::LEFT_TOP,
            "Standings",
            13.5,
            theme.muted,
        );
        let mut order: Vec<usize> = (0..self.race.len()).collect();
        order.sort_by(|&a, &b| distances[b].total_cmp(&distances[a]));
        for (place, &i) in order.iter().enumerate() {
            let y = board.top() + 34. + place as f32 * 26.;
            if y > board.bottom() - 26. {
                break;
            }
            let lane = &self.race[i];
            painter.text(
                Pos2::new(board.left() + 10., y),
                Align2::LEFT_CENTER,
                format!("{}. {}", place + 1, species_name(&lane.playback.creature)),
                FontId::proportional(14.),
                if place == 0 { theme.accent } else { theme.ink },
            );
            painter.text(
                Pos2::new(board.right() - 10., y),
                Align2::RIGHT_CENTER,
                format!("{:.2} m", distances[i]),
                FontId::proportional(14.),
                if place == 0 {
                    theme.accent
                } else {
                    theme.muted
                },
            );
        }
        painter.text(
            board.left_bottom() + Vec2::new(10., -8.),
            Align2::LEFT_BOTTOM,
            "Live distance",
            FontId::proportional(14.5),
            theme.muted,
        );
    }
    /// Body plans and effective clades of the global archive at generation
    /// `index` of the history, with their trace over the last 100
    /// generations.
    fn diversity_meter(&self, ui: &mut egui::Ui, index: usize) {
        let theme = self.theme();
        let Some(snapshot) = self.snapshot.as_ref() else {
            return;
        };
        let history = &snapshot.history;
        let Some(now) = history.get(index) else {
            return;
        };
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!(
                    "{} body plans · {:.1} effective clades · plans live {:.0} generations (median)",
                    number(now.plans),
                    now.clades,
                    now.plan_age
                ))
                .strong(),
            )
            .on_hover_text(
                "Body plans: different skeletons among the kept creatures. Effective clades: how many lineages share the archive, counting a lineage by its share (exp of the Shannon entropy).",
            );
            let start = index.saturating_sub(99);
            let window = &history[start..=index];
            let (rect, _) = ui.allocate_exact_size(Vec2::new(200., 28.), Sense::hover());
            let painter = ui.painter();
            painter.rect_filled(rect, 3., theme.card);
            for (values, color) in [
                (window.iter().map(|h| h.plans as f32).collect::<Vec<_>>(), theme.accent),
                (window.iter().map(|h| h.clades).collect::<Vec<_>>(), theme.cold),
            ] {
                let top = values.iter().copied().fold(1.0f32, f32::max);
                let n = values.len().max(2) - 1;
                let points: Vec<egui::Pos2> = values
                    .iter()
                    .enumerate()
                    .map(|(k, v)| {
                        egui::pos2(
                            rect.left() + rect.width() * k as f32 / n as f32,
                            rect.bottom() - 2. - (rect.height() - 4.) * v / top,
                        )
                    })
                    .collect();
                painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
            }
        });
    }
    fn species_history(&mut self, ui: &mut egui::Ui) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        if snapshot.history.is_empty() {
            return;
        }
        let theme = self.theme();
        ui.horizontal_wrapped(|ui| {
            crate::theme::heading(ui, "Body types through generations", theme);
            ui.label(
                RichText::new(
                    "each color is one count of nodes and muscles, named in the list below",
                )
                .small()
                .color(theme.muted),
            );
        });
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 80.), Sense::click());
        let painter = ui.painter_at(rect);
        let history = &snapshot.history;
        let stride = history.len().div_ceil(rect.width().max(1.) as usize).max(1);
        for i in (0..history.len()).step_by(stride) {
            let h = &history[i];
            let body_count = if h.archive_cells > 0 {
                h.archive_cells
            } else {
                h.population
            };
            let x = rect.left() + rect.width() * i as f32 / history.len() as f32;
            let right = rect.left()
                + rect.width() * (i + stride).min(history.len()) as f32 / history.len() as f32;
            let mut y = rect.bottom();
            for &(nodes, muscles, count) in &h.species {
                let height = rect.height() * count as f32 / body_count.max(1) as f32;
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(x, y - height), Pos2::new(right, y)),
                    0,
                    species_color(nodes, muscles),
                );
                y -= height;
            }
        }
        let x =
            rect.left() + rect.width() * (self.history_index as f32 + 0.5) / history.len() as f32;
        painter.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(2., theme.ink),
        );
        if let Some(pos) = response.hover_pos() {
            let index = (((pos.x - rect.left()) / rect.width()) * history.len() as f32) as usize;
            let index = index.min(history.len() - 1);
            if response.clicked() {
                self.history_latest = false;
                self.history_index = index;
            }
            response.on_hover_text(format!(
                "Generation {} · click to inspect",
                history[index].generation
            ));
        }
    }
    fn history(&mut self, ui: &mut egui::Ui) {
        let Some(s) = &self.snapshot else { return };
        let theme = self.theme();
        let len = s.history.len();
        if len == 0 {
            ui.heading("A history waiting to happen");
            ui.label("Run your first generation to build distance curves and creature replays.");
            return;
        }
        ui.horizontal(|ui| {
            ui.heading("History");
            ui.checkbox(&mut self.history_latest, "Follow latest");
            if ui.button("Export CSV").clicked() {
                self.file("Export CSV");
            }
        });
        if self.history_latest {
            self.history_index = len - 1;
        }
        self.history_index = self.history_index.min(len - 1);
        ui.add(egui::Slider::new(&mut self.history_index, 0..=len - 1).text("Generation"))
            .on_hover_text("Disable Follow latest to keep a historical generation selected");
        self.trend(ui, 180.);
        ui.add_space(GAP_M);
        self.records_list(ui);
        ui.add_space(GAP_M);
        self.species_history(ui);
        let stats = self.snapshot.as_ref().unwrap().history[self.history_index].clone();
        let body_count = if stats.archive_cells > 0 {
            stats.archive_cells
        } else {
            stats.population
        };
        ui.horizontal(|ui| {
            ui.label(format!(
                "Generation {} · {} creatures tried · {} creatures kept in {} ways of moving",
                stats.generation,
                number(stats.population),
                number(stats.archive_cells),
                number(stats.moves()),
            ));
        });
        self.diversity_meter(ui, self.history_index);
        let mut picked_type = None;
        ui.columns(2, |cols| {
            self.histogram(&mut cols[0], &stats, 155.);
            crate::theme::heading(&mut cols[1], "Body types", theme);
            let mut species = stats.species.clone();
            species.sort_by_key(|&(_, _, n)| std::cmp::Reverse(n));
            egui::ScrollArea::vertical()
                .max_height(180.)
                .show(&mut cols[1], |ui| {
                    for &(n, m, count) in &species {
                        ui.horizontal(|ui| {
                            color_dot(ui, species_color(n, m));
                            // A click replays the fastest kept creature of
                            // this body type (a species view, as NEAT shows
                            // its species).
                            if ui
                                .link(format!(
                                    "{n} nodes / {} bones / {m} muscles",
                                    n.saturating_sub(1)
                                ))
                                .on_hover_text("Replay the fastest kept creature of this body type")
                                .clicked()
                            {
                                picked_type = Some((n, m));
                            }
                            ui.label(format!(
                                "{} · {:.1}%",
                                number(count as usize),
                                100. * count as f32 / body_count.max(1) as f32
                            ));
                        });
                    }
                });
        });
        ui.add_space(GAP_M);
        if let Some((n, m)) = picked_type {
            let best = self.cards.as_ref().and_then(|list| {
                list.cards
                    .iter()
                    .filter(|c| c.creature.nodes.len() == n && c.creature.muscles.len() == m)
                    .max_by(|a, b| a.score.total_cmp(&b.score))
                    .map(|c| c.replay_of(&list.config))
            });
            match best {
                Some((creature, config)) => {
                    self.select(creature, config);
                    self.tab = Tab::Overview;
                    return;
                }
                None => self.request_cards(),
            }
        }
        let mut selection = None;
        ui.columns(3, |cols| {
            for (i, ui) in cols.iter_mut().enumerate() {
                ui.label(["Worst creature", "Median creature", "Best creature"][i]);
                let (rect, response) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 100.), Sense::click());
                ui.painter().rect_filled(rect, 8, theme.card);
                thumbnail(
                    &ui.painter_at(rect),
                    &stats.representatives[i],
                    rect.shrink(12.),
                );
                if response.clicked() {
                    selection = Some(stats.representatives[i].clone());
                }
            }
        });
        if let Some(c) = selection {
            self.select(c, stats.config);
            self.tab = Tab::Overview;
        }
    }
    /// A bar across the window while a developer measurement pauses the
    /// game (`dev_pause`), with the time left and Resume now.
    fn dev_pause_bar(&self, ui: &mut egui::Ui) {
        let Some(view) = self.worker.dev_pause.view() else {
            return;
        };
        let theme = self.theme();
        egui::Panel::bottom("dev-pause")
            .frame(
                egui::Frame::new()
                    .fill(theme.panel)
                    .inner_margin(egui::Margin::symmetric(GAP_L as i8, 6)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let left = view.ends_at.saturating_duration_since(Instant::now()).as_secs();
                    let text = if view.closed {
                        format!(
                            "Paused for a developer measurement, resumes in {}:{:02}",
                            left / 60,
                            left % 60
                        )
                    } else {
                        "Pausing for a developer measurement: the last creatures finish their trials"
                            .to_owned()
                    };
                    ui.colored_label(theme.warn, text);
                    if ui
                        .button("Resume now")
                        .on_hover_text("End the developer pause and keep evolving")
                        .clicked()
                    {
                        self.worker.dev_pause.resume_now();
                    }
                });
            });
        ui.ctx().request_repaint_after(Duration::from_millis(500));
    }
    /// The closed-by-default drawer with search and machine numbers, and the
    /// step-by-step run buttons developers use.
    fn diagnostics(&self, ui: &mut egui::Ui, s: &Snapshot) {
        let mut frames: Vec<_> = self.frame_times.iter().copied().collect();
        frames.sort_by(f32::total_cmp);
        let p95 = frames.get(frames.len() * 95 / 100).copied().unwrap_or(0.);
        ui.small(format!(
            "{} · seed {} · {} · {} / {} evaluated · {} in confirmation",
            s.gpu,
            s.config.seed,
            if s.running { "running" } else { "paused" },
            number(s.completed),
            number(s.config.population),
            number(s.checking),
        ));
        ui.small(format!(
            "QD score {:.2} · {} behavior niches · {} topology reserves · next batch {}",
            s.qd_score,
            number(s.archive_cells),
            number(s.innovation_reserve_count),
            crate::qd::Emitter::ALL
                .into_iter()
                .zip(s.emitter_weights)
                .map(|(emitter, weight)| format!("{} {:.0}%", emitter.label(), weight * 100.0))
                .collect::<Vec<_>>()
                .join(", "),
        ));
        ui.small(format!(
            "Frame p95 {:.1} ms · end-to-end {:.0} creatures/s · GPU buffers {:.1} MiB · population {:.1} MiB · runs/ {}",
            p95 * 1000.,
            s.end_to_end,
            s.gpu_bytes as f64 / 1048576.,
            s.ram_bytes as f64 / 1048576.,
            file_size(self.runs_bytes)
        ));
        for (name, rate, count) in &s.engines {
            ui.small(format!("{name}: {rate:.0} creatures/s · {count} evaluated"));
        }
        let running = self.active();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!running, egui::Button::new("One generation").small())
                .clicked()
            {
                self.worker.pause.store(false, Ordering::Relaxed);
                self.worker.send(Command::Run {
                    continuous: false,
                    guided: false,
                });
            }
        });
    }
    /// Keyboard shortcuts and what each tab shows.
    fn help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let theme = self.theme();
        egui::Window::new("Help")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_max_width(460.);
                ui.heading("Shortcuts");
                egui::Grid::new("help_shortcuts")
                    .num_columns(2)
                    .spacing([18., 6.])
                    .show(ui, |ui| {
                        for (keys, action) in [
                            (
                                "1 to 5",
                                "Overview · Ways of moving · History · Race · Lineage",
                            ),
                            ("Space", "Evolve, or pause evolution"),
                            ("K or a click on the replay", "Play or pause the replay"),
                            ("← / →", "Step the replay one frame"),
                            ("F1 or ?", "Toggle this help"),
                            ("Ctrl+S", "Save the experiment"),
                            ("Drag / scroll", "Pan and zoom the viewport"),
                        ] {
                            ui.label(RichText::new(keys).strong().color(theme.accent));
                            ui.label(action);
                            ui.end_row();
                        }
                    });
                ui.separator();
                if ui.button("How evolution works").clicked() {
                    self.schematic_open = true;
                }
                ui.separator();
                ui.heading("Tabs");
                for (name, why) in [
                    (
                        "Overview",
                        "The champion's replay (or the creature you picked), its playback controls, lineage and the best distance over time.",
                    ),
                    (
                        "Ways of moving",
                        "The best creature for every way of moving, as cards or as a map. Click a creature or a map cell to replay it.",
                    ),
                    (
                        "History",
                        "The best and median distance over time with world changes marked, every record with a replay, the mix of body types, and the distances of one generation.",
                    ),
                    (
                        "Race",
                        "The five fastest archived creatures run their trials side by side with live standings.",
                    ),
                    (
                        "Lineage",
                        "Ancestors of the creature on screen with thumbnails, distance gains and body plan changes.",
                    ),
                ] {
                    ui.label(RichText::new(name).strong());
                    ui.label(RichText::new(why).color(theme.muted));
                    ui.add_space(GAP_S);
                }
            });
    }
    /// Follows the worker's event log for the save state: a save or an open
    /// marks the experiment saved, a new game marks it unsaved.
    fn absorb_events(&mut self, snapshot: &Snapshot) {
        if self.events_seen.0 != snapshot.epoch || snapshot.events.len() < self.events_seen.1 {
            self.events_seen = (snapshot.epoch, 0);
        }
        for event in &snapshot.events[self.events_seen.1..] {
            match event.kind {
                EventKind::Saved | EventKind::Opened => {
                    self.saved = Some((
                        Instant::now(),
                        event.generation,
                        event.kind == EventKind::Opened,
                    ));
                    if event.kind == EventKind::Saved {
                        self.saving = None;
                    }
                }
                EventKind::Started => self.saved = None,
                _ => {}
            }
        }
        self.events_seen.1 = snapshot.events.len();
        if snapshot.error.is_some() {
            self.saving = None;
        }
    }
    /// Asks the worker to save, or first asks the player when the file exists.
    fn save_to(&mut self, path: PathBuf, confirmed: bool) {
        if !confirmed && path.exists() {
            self.overwrite = Some(path);
            return;
        }
        self.saving = Some(Instant::now());
        self.worker.send(Command::Save(path));
    }
    /// Save state for the top bar: saving, saved how long ago, or not saved.
    fn save_state(&self) -> (String, bool) {
        if self.saving.is_some() {
            return ("Saving…".to_owned(), true);
        }
        match self.saved {
            Some((at, generation, opened)) => {
                let now = self.snapshot.as_ref().map_or(generation, |s| s.generation);
                let since = now.saturating_sub(generation);
                let ago = seconds_text(at.elapsed().as_secs_f64());
                let verb = if opened { "Opened" } else { "Saved" };
                (
                    if since > 0 {
                        format!("{verb} {ago} ago · {since} generations since")
                    } else {
                        format!("{verb} {ago} ago")
                    },
                    false,
                )
            }
            None => ("Not saved".to_owned(), false),
        }
    }
    /// Opens a saved experiment; the game starts paused on it.
    fn open_experiment(&mut self, path: PathBuf) {
        self.pause();
        self.worker.send(Command::Load(path));
        self.initial = true;
    }
    /// File > Open: the saves in runs/, newest first, and a path field for
    /// a file anywhere else.
    fn open_window(&mut self, ctx: &egui::Context) {
        let Some(saves) = &self.open_list else {
            return;
        };
        let theme = self.theme();
        let mut chosen: Option<PathBuf> = None;
        let mut close = false;
        egui::Window::new("Open a saved experiment")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_width(560.);
                if saves.is_empty() {
                    ui.label(RichText::new("No saves in runs/ yet.").color(theme.muted));
                }
                egui::ScrollArea::vertical()
                    .max_height(360.)
                    .show(ui, |ui| {
                        egui::Grid::new("saves")
                            .num_columns(4)
                            .spacing([14., 8.])
                            .striped(true)
                            .show(ui, |ui| {
                                for save in saves {
                                    let name =
                                        save.path.file_name().map_or_else(String::new, |n| {
                                            n.to_string_lossy().into_owned()
                                        });
                                    ui.vertical(|ui| {
                                        ui.label(RichText::new(name).strong());
                                        if let Some(summary) = &save.summary {
                                            ui.label(
                                                RichText::new(world_summary(&summary.config))
                                                    .small()
                                                    .color(theme.muted),
                                            );
                                        }
                                    });
                                    ui.label(save.summary.as_ref().map_or_else(
                                        || "older format".to_owned(),
                                        |summary| {
                                            format!(
                                                "generation {} · {} creatures",
                                                summary.generation,
                                                number(summary.config.population)
                                            )
                                        },
                                    ));
                                    ui.label(
                                        RichText::new(format!(
                                            "{} · {}",
                                            ago(save.modified),
                                            file_size(save.bytes)
                                        ))
                                        .small()
                                        .color(theme.muted),
                                    );
                                    if ui.button("Open").clicked() {
                                        chosen = Some(save.path.clone());
                                    }
                                    ui.end_row();
                                }
                            });
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Another file");
                    ui.add(egui::TextEdit::singleline(&mut self.file_path).desired_width(300.));
                    if ui.button("Open").clicked() {
                        chosen = Some(PathBuf::from(&self.file_path));
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
        if let Some(path) = chosen {
            self.open_experiment(path);
            close = true;
        }
        if close {
            self.open_list = None;
        }
    }
    fn dialogs(&mut self, ctx: &egui::Context) {
        let theme = self.theme();
        self.open_window(ctx);
        if let Some(path) = self.overwrite.clone() {
            let mut answer = None;
            egui::Window::new("Replace the save?")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "{} already exists. Replace it with this experiment?",
                        path.display()
                    ));
                    ui.horizontal(|ui| {
                        if ui.button("Replace").clicked() {
                            answer = Some(true);
                        }
                        if ui.button("Cancel").clicked() {
                            answer = Some(false);
                        }
                    });
                });
            match answer {
                Some(true) => {
                    self.overwrite = None;
                    self.save_to(path, true);
                }
                Some(false) => self.overwrite = None,
                None => {}
            }
        }
        if self.new_dialog {
            egui::Window::new("Start a new experiment")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "Start over with {} new creatures in this world: {}.",
                        number(self.config.population),
                        world_summary(&self.config)
                    ));
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.config.random_seed, "New random seed");
                        if !self.config.random_seed {
                            ui.label("Seed");
                            ui.add(egui::DragValue::new(&mut self.config.seed));
                        }
                    });
                    ui.label("Save the current experiment first if you want to resume it later.");
                    ui.horizontal(|ui| {
                        if ui.button("Save current").clicked() {
                            self.file("Save experiment");
                            self.new_dialog = false;
                        }
                        if ui
                            .add_enabled(
                                self.config.validate().is_ok(),
                                egui::Button::new(
                                    RichText::new("Create population").color(theme.go_text),
                                )
                                .fill(theme.go_fill),
                            )
                            .clicked()
                        {
                            self.pause();
                            self.worker.send(Command::New(self.config.clone()));
                            self.initial = true;
                            self.new_dialog = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.new_dialog = false;
                        }
                    });
                });
        }
        if let Some(mode) = self.file_mode {
            egui::Window::new(mode)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("File path");
                    ui.add(egui::TextEdit::singleline(&mut self.file_path).desired_width(430.));
                    ui.horizontal(|ui| {
                        if ui.button(mode).clicked() {
                            let path = PathBuf::from(&self.file_path);
                            match mode {
                                "Save experiment" => self.save_to(path, false),
                                "Open experiment" => self.open_experiment(path),
                                "Export CSV" => self.worker.send(Command::Export(path)),
                                "Export creature JSON" => {
                                    let creature =
                                        self.playback.as_ref().map(|p| p.creature.clone());
                                    let result = (|| -> anyhow::Result<()> {
                                        let creature = creature.as_ref().ok_or_else(|| {
                                            anyhow::anyhow!("No creature is selected to export")
                                        })?;
                                        if let Some(parent) = path.parent() {
                                            std::fs::create_dir_all(parent)?;
                                        }
                                        serde_json::to_writer_pretty(
                                            std::fs::File::create(&path)?,
                                            creature,
                                        )?;
                                        Ok(())
                                    })();
                                    self.message = Some(result.map_or_else(
                                        |e| e.to_string(),
                                        |_| format!("Creature saved to {}", path.display()),
                                    ));
                                }
                                "Export creature GIF" => {
                                    let result = (|| -> anyhow::Result<usize> {
                                        let playback = self.playback.as_ref().ok_or_else(|| {
                                            anyhow::anyhow!("No creature is selected to export")
                                        })?;
                                        if let Some(parent) = path.parent() {
                                            std::fs::create_dir_all(parent)?;
                                        }
                                        export_creature_gif(playback, &path)
                                    })();
                                    self.message = Some(result.map_or_else(
                                        |e| format!("GIF export failed: {e}"),
                                        |frames| {
                                            format!(
                                                "GIF saved to {} ({frames} frames)",
                                                path.display()
                                            )
                                        },
                                    ));
                                }
                                "Open creature JSON" => {
                                    let config = self
                                        .snapshot
                                        .as_ref()
                                        .map_or_else(Config::default, |s| s.config.clone());
                                    let result = (|| -> anyhow::Result<Creature> {
                                        let mut creature: Creature =
                                            serde_json::from_reader(std::fs::File::open(&path)?)?;
                                        imported_creature(&mut creature).map_err(|why| {
                                            anyhow::anyhow!("{}: {why}", path.display())
                                        })?;
                                        Ok(creature)
                                    })();
                                    match result {
                                        Ok(creature) => {
                                            self.select(creature, config);
                                            self.tab = Tab::Overview;
                                            self.message =
                                                Some(format!("Opened {}", path.display()));
                                        }
                                        Err(e) => self.message = Some(e.to_string()),
                                    }
                                }
                                _ => {}
                            }
                            self.file_mode = None;
                        }
                        if ui.button("Cancel").clicked() {
                            self.file_mode = None;
                        }
                    });
                });
        }
    }
}
impl eframe::App for App {
    fn on_exit(&mut self) {
        if self.bench_frames.is_empty() {
            return;
        }
        if !self.replay_seconds.is_empty() {
            let mut times = self.replay_seconds.clone();
            times.sort_by(f32::total_cmp);
            eprintln!(
                "Native benchmark replays: {} replays, median {:.2} s, max {:.2} s",
                times.len(),
                times[times.len() / 2],
                times[times.len() - 1]
            );
        }
        let report = |label: &str, frames: &[f32]| {
            if frames.is_empty() {
                return;
            }
            let mut sorted = frames.to_vec();
            sorted.sort_by(f32::total_cmp);
            let pct = |q: usize| sorted[(sorted.len() * q / 100).min(sorted.len() - 1)] * 1000.;
            let total: f32 = sorted.iter().sum();
            eprintln!(
                "Native benchmark frames{label}: {} frames, {:.1} FPS, p50 {:.2} ms, p95 {:.2} ms, p99 {:.2} ms, max {:.2} ms",
                sorted.len(),
                sorted.len() as f32 / total.max(1e-6),
                pct(50),
                pct(95),
                pct(99),
                sorted[sorted.len() - 1] * 1000.
            );
            // Frame-time histogram in milliseconds.
            let edges = [8.3f32, 16.7, 25.0, 33.3, 50.0, 100.0];
            let mut counts = [0usize; 7];
            for &f in frames {
                let ms = f * 1000.;
                counts[edges.iter().take_while(|&&e| ms >= e).count()] += 1;
            }
            eprintln!(
                "Native benchmark frame histogram{label}: <8.3 ms {}, 8.3-16.7 {}, 16.7-25 {}, 25-33.3 {}, 33.3-50 {}, 50-100 {}, >=100 {}",
                counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]
            );
        };
        report("", &self.bench_frames);
        report(" (frame time without vsync)", &self.bench_work);
        // Frames that overlapped a ring step that absorbed and bred a block.
        let mut spans = self.worker.breeding.lock().unwrap().clone();
        spans.sort_by_key(|span| span.0);
        let breeding: Vec<f32> = self
            .bench_frames
            .iter()
            .zip(&self.bench_frame_starts)
            .filter(|&(&dt, &start)| {
                let end = start + Duration::from_secs_f32(dt);
                spans.iter().any(|&(a, b, _)| start < b && end > a)
            })
            .map(|(&dt, _)| dt)
            .collect();
        let bred: f32 = spans.iter().map(|(a, b, _)| (*b - *a).as_secs_f32()).sum();
        eprintln!(
            "Native benchmark breeding: {} blocks absorbed and bred, {bred:.2} s",
            spans.len()
        );
        report(" during breeding", &breeding);
        if let (Some(start), Some(end)) = (self.bench_faults, crate::threads::major_faults()) {
            eprintln!(
                "Native benchmark UI thread: {} major faults while measuring",
                end - start
            );
        }
    }
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.runs_checked.elapsed() >= RUNS_REFRESH {
            self.runs_checked = now;
            self.runs_bytes = directory_bytes(std::path::Path::new("runs"));
        }
        if self.smoke_start_pending && self.started.elapsed() >= Duration::from_millis(250) {
            self.worker.send(Command::Run {
                continuous: true,
                guided: false,
            });
            self.smoke_start_pending = false;
        }
        // Screenshot runs: EVOLUTION_SMOKE_PRESET=<number> applies that
        // preset after 4 s, so the chart and feed have a world change.
        if let Some(index) = self.smoke_preset
            && self.started.elapsed() >= Duration::from_secs(4)
        {
            self.smoke_preset = None;
            if let Some(preset) = crate::environment::PRESETS.get(index) {
                preset.apply(&mut self.config);
                self.worker.send(Command::Configure(self.config.clone()));
            }
        }
        self.frame_times.push_back(dt);
        if self.frame_times.len() > 240 {
            self.frame_times.pop_front();
        }
        if self.worker.measuring.load(Ordering::Relaxed) {
            if self.bench_frames.is_empty() {
                self.bench_faults = crate::threads::major_faults();
            }
            self.bench_frames.push(dt);
            self.bench_frame_starts
                .push(now - Duration::from_secs_f32(dt));
            // EVOLUTION_BENCH_REPLAY: ask for the champion's replay every 6 s
            // and time how long it takes to appear.
            if std::env::var_os("EVOLUTION_BENCH_REPLAY").is_some()
                && self.bench_last_replay.elapsed() >= Duration::from_secs(6)
                && self.replay_wait.is_none()
                && let Some((creature, config)) = self.champion()
            {
                self.bench_last_replay = now;
                self.set_preview(creature, config);
            }
            if self.bench_last_ping.elapsed() >= Duration::from_millis(100) {
                self.bench_last_ping = now;
                self.bench_pings += 1;
                // Every tenth probe re-applies the settings, like an
                // environment button, when EVOLUTION_BENCH_SETTINGS_PROBE is set.
                if self.bench_pings.is_multiple_of(10)
                    && std::env::var_os("EVOLUTION_BENCH_SETTINGS_PROBE").is_some()
                {
                    self.worker.send(Command::ConfigureProbe(now));
                } else {
                    self.worker.send(Command::Ping(now));
                }
            }
        }
        if let Some((rx, asked)) = &self.replay_wait
            && let Ok(ready) = rx.try_recv()
        {
            self.replay_seconds.push(asked.elapsed().as_secs_f32());
            self.playback = Some(ready);
            self.replay_wait = None;
        }
        let next = self.worker.view.lock().unwrap().take();
        if let Some(mut next) = next {
            if self.initial
                && next.epoch > 0
                && self
                    .snapshot
                    .as_ref()
                    .is_none_or(|old| old.epoch != next.epoch)
            {
                self.config = next.config.clone();
                self.initial = false;
            } else if self.config_sent.is_none_or(|sent| {
                // A click is acknowledged once the worker's world shows it.
                // A snapshot published before the worker read the click must
                // not put the panel back (autochange would flip to Off), so wait
                // for the match, and give up after a while.
                let acknowledged = worlds_match(
                    &next.pending.clone().unwrap_or_else(|| next.config.clone()),
                    &self.config,
                );
                sent.elapsed() > Duration::from_secs(if acknowledged { 2 } else { 15 })
            }) && self
                .snapshot
                .as_ref()
                .is_none_or(|old| old.epoch == next.epoch)
            {
                // The worker owns the world: autochange advance it, and a change
                // waits in `pending` until the next generation. The panel
                // shows the world the player asked for.
                self.config = next.pending.clone().unwrap_or_else(|| next.config.clone());
            }
            if let Some((c, cfg)) = next.preview.take() {
                // The worker picks the creature of a new game (a random one)
                // and of a loaded game (its best elite).
                // Neither is known to be the champion: the history's best
                // takes over at once below when it differs.
                self.show_champion(c, cfg);
                self.champion_shown = false;
            }
            self.absorb_events(&next);
            if let Some(list) = next.cards.take() {
                self.cards_requested = None;
                if self.race_pending && self.race_picks.is_empty() {
                    self.build_top_race(&list);
                }
                self.cards = Some(list);
            }
            if let Some((c, cfg)) = next.selected.take() {
                // A creature the player clicked on the archive map; it plays
                // in the player docked beside the map.
                self.select(c, cfg);
            }
            if let Some((id, lineage)) = next.lineage.take() {
                if self.playback.as_ref().is_some_and(|p| p.creature.id == id) {
                    self.lineage = lineage;
                }
                self.lineage_pending = false;
            }
            self.snapshot = Some(next);
            self.follow_champion();
            self.maybe_build_race();
        }
        // The worker sends the archive map table only while the map shows.
        let want_map = self.tab == Tab::Population && self.archive_view == ArchiveView::Map;
        if want_map != self.map_sent {
            self.map_sent = want_map;
            self.worker.send(Command::MapTable(want_map));
        }
        let lineage_request =
            if (self.tab == Tab::Overview || self.tab == Tab::Lineage) && self.lineage.is_empty() {
                self.playback
                    .as_ref()
                    .map(|p| p.creature.id)
                    .filter(|&id| self.lineage_requested != Some(id))
            } else {
                None
            };
        if let Some(id) = lineage_request {
            self.lineage_requested = Some(id);
            self.lineage_pending = true;
            self.worker.send(Command::Lineage(id));
        }
        if !ctx.egui_wants_keyboard_input() {
            let pressed = |key| ctx.input(|i| i.key_pressed(key));
            if pressed(egui::Key::Num1) {
                self.tab = Tab::Overview;
            }
            if pressed(egui::Key::Num2) {
                self.tab = Tab::Population;
            }
            if pressed(egui::Key::Num3) {
                self.tab = Tab::History;
            }
            if pressed(egui::Key::Num4) {
                self.tab = Tab::Race;
            }
            if pressed(egui::Key::Num5) {
                self.tab = Tab::Lineage;
            }
            if pressed(egui::Key::F1) || pressed(egui::Key::Questionmark) {
                self.show_help = !self.show_help;
            }
            if pressed(egui::Key::Space) {
                if self.active() {
                    self.pause();
                } else {
                    self.run(true, false);
                }
            }
            if pressed(egui::Key::K) {
                self.playing = !self.playing;
            }
            if let Some(p) = &mut self.playback {
                let elapsed = p.tick.saturating_sub(p.trial_start());
                if pressed(egui::Key::ArrowLeft) {
                    p.seek(elapsed.saturating_sub(1));
                    self.playing = false;
                }
                if pressed(egui::Key::ArrowRight) {
                    p.seek(elapsed.saturating_add(1));
                    self.playing = false;
                }
            }
            if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S)) {
                self.file("Save experiment");
            }
        }
        if self.tab != self.prev_tab {
            self.prev_tab = self.tab;
            self.opened_tab(self.tab);
        }
        if self.playing {
            let frame_dt = physics::dt();
            if frame_dt.is_finite() && frame_dt > 0.0 {
                let speed = self.speed;
                let advance = |p: &mut Playback| {
                    p.accumulator = (p.accumulator + dt.clamp(0.0, 0.1) * speed).min(1.0);
                    let start = Instant::now();
                    while p.accumulator >= frame_dt && start.elapsed() < Duration::from_millis(5) {
                        if p.tick >= p.last_frame() {
                            p.reset();
                        }
                        p.advance();
                        p.accumulator -= frame_dt;
                    }
                };
                if let Some(p) = &mut self.playback {
                    advance(p);
                    p.show_between();
                }
                if self.tab == Tab::Race {
                    for lane in &mut self.race {
                        advance(&mut lane.playback);
                        lane.playback.show_between();
                    }
                }
            }
        }
        // The camera follows the averaged center of mass, paused or not, so
        // a seek also recenters it.
        if self.follow
            && let Some(p) = &self.playback
        {
            self.camera[0] = p.camera_x();
        }
        let theme = self.theme();
        // The blurred city behind everything, as the game world shows
        // behind the Half-Life 2 menus; the panels are dark glass over it.
        crate::theme::backdrop(ui.painter(), ui.ctx().content_rect());
        egui::Panel::top("top")
            .exact_size(68.)
            .frame(
                egui::Frame::new()
                    .fill(crate::theme::poster::WOOD_DARK)
                    .inner_margin(egui::Margin::symmetric(GAP_L as i8, 15)),
            )
            .show(ui, |ui| {
                // A mustard stripe under the bar, like the poster's border.
                let bar = ui.max_rect().expand2(Vec2::new(GAP_L, 15.));
                ui.painter().rect_filled(
                    Rect::from_min_max(
                        Pos2::new(bar.left(), bar.bottom() - 4.),
                        bar.right_bottom(),
                    ),
                    0,
                    crate::theme::poster::MUSTARD,
                );
                self.top(ui)
            });
        self.dev_pause_bar(ui);
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(message) = self.message.take() {
                    self.shown_message = Some((message, Instant::now()));
                }
                if self
                    .shown_message
                    .as_ref()
                    .is_some_and(|(_, at)| at.elapsed().as_secs_f32() > MESSAGE_SECONDS)
                {
                    self.shown_message = None;
                }
                if let Some(s) = &self.snapshot {
                    color_dot(ui, theme.accent);
                    // A world whose kernels nobody compiled yet makes the GPU
                    // wait for them. Waits under a third of a second are
                    // loads of kernels that are ready, not worth a message.
                    if crate::cuda_engine::compiling_world() {
                        self.compiling_since.get_or_insert_with(Instant::now);
                        ui.ctx().request_repaint_after(Duration::from_millis(250));
                    } else {
                        self.compiling_since = None;
                    }
                    let compiling = self
                        .compiling_since
                        .map(|since| since.elapsed())
                        .filter(|waited| *waited > Duration::from_millis(300));
                    match (&self.shown_message, compiling) {
                        (Some((message, _)), _) => ui.label(message),
                        (None, Some(waited)) => ui.label(format!(
                            "Compiling GPU kernels for this world… {:.0} s. Evolution starts when they are ready. A new world compiles once.",
                            waited.as_secs_f32()
                        )),
                        (None, None) => ui.label(&s.status),
                    };
                    if self.shown_message.is_some() {
                        ui.ctx().request_repaint_after(Duration::from_millis(500));
                    }
                    if let Some(error) = &s.error {
                        ui.colored_label(theme.danger, error);
                    }
                    ui.label(
                        RichText::new(format!("{:.0} creatures/s", s.end_to_end))
                            .small()
                            .color(theme.muted),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button(if self.show_perf {
                                "Hide diagnostics"
                            } else {
                                "Diagnostics"
                            })
                            .on_hover_text("Search and machine numbers for developers")
                            .clicked()
                        {
                            self.show_perf = !self.show_perf;
                        }
                    });
                }
            });
            if self.show_perf
                && let Some(s) = &self.snapshot
            {
                self.diagnostics(ui, s);
            }
        });
        egui::Panel::left("controls")
            .default_size(440.)
            .min_size(340.)
            .max_size(520.)
            .resizable(true)
            .frame(
                egui::Frame::new()
                    .fill(theme.panel)
                    .inner_margin(GAP_L as i8),
            )
            .show(ui, |ui| {
                crate::theme::wear(ui.painter(), ui.max_rect().expand(GAP_L), theme, 7.);
                self.controls(ui)
            });
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme.canvas)
                    .inner_margin(GAP_L as i8),
            )
            .show(ui, |ui| {
                let strip = ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = GAP_M;
                    for (key, (tab, label)) in [
                        (Tab::Overview, "Overview"),
                        (Tab::Population, "Ways of moving"),
                        (Tab::History, "History"),
                        (Tab::Race, "Race"),
                        (Tab::Lineage, "Lineage"),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if crate::theme::tab(
                            ui,
                            self.tab == tab,
                            &(key + 1).to_string(),
                            label,
                            theme,
                        )
                        .on_hover_text(format!("Key {}", key + 1))
                        .clicked()
                        {
                            self.tab = tab;
                        }
                    }
                });
                // A thick rule under the tabs, across the panel.
                ui.painter().hline(
                    ui.max_rect().x_range(),
                    strip.response.rect.bottom() + 5.,
                    Stroke::new(3., theme.ink),
                );
                ui.add_space(GAP_L);
                match self.tab {
                    Tab::Overview => {
                        self.metrics(ui);
                        ui.add_space(GAP_M);
                        // The chart keeps a fixed height below the replay and
                        // its controls; the replay takes the rest.
                        const CHART: f32 = 175.;
                        const REPLAY_CONTROLS: f32 = 120.;
                        self.viewport(
                            ui,
                            (ui.available_height() - CHART - REPLAY_CONTROLS).max(180.),
                        );
                        ui.add_space(GAP_M);
                        // The chart and the event feed share the bottom row.
                        let height = (ui.available_height() - 34.).max(80.);
                        let width = ui.available_width();
                        ui.horizontal_top(|ui| {
                            let down = egui::Layout::top_down(egui::Align::Min);
                            ui.allocate_ui_with_layout(
                                Vec2::new(width * 0.62, height + 34.),
                                down,
                                |ui| self.trend(ui, height),
                            );
                            ui.allocate_ui_with_layout(
                                Vec2::new(ui.available_width(), height + 34.),
                                down,
                                |ui| self.feed(ui, height + 10.),
                            );
                        });
                    }
                    Tab::Population => {
                        // The archive on the left, the replay docked on the
                        // right, so browsing never leaves the tab.
                        let height = ui.available_height();
                        let width = ui.available_width();
                        ui.horizontal_top(|ui| {
                            ui.allocate_ui(Vec2::new(width * 0.57, height), |ui| {
                                ui.vertical(|ui| self.population(ui));
                            });
                            ui.allocate_ui(Vec2::new(ui.available_width(), height), |ui| {
                                ui.vertical(|ui| {
                                    self.viewport(ui, (height * 0.55).max(180.));
                                });
                            });
                        });
                    }
                    Tab::History => {
                        egui::ScrollArea::vertical().show(ui, |ui| self.history(ui));
                    }
                    Tab::Race => self.race_view(ui),
                    Tab::Lineage => self.lineage_view(ui),
                }
            });
        self.dialogs(&ctx);
        self.help_window(&ctx);
        self.loading_screen(&ctx);
        crate::schematic::show(&ctx, self.snapshot.as_ref(), &mut self.schematic_open);
        if self.playing || self.active() {
            // Playback and live evolution redraw at the frame cap; the rest of
            // the GPU stays with evolution. EVOLUTION_UI_FPS=0 follows vsync.
            match ui_frame_interval() {
                Some(interval) => ctx.request_repaint_after(interval),
                None => ctx.request_repaint(),
            }
        }
        // Unattended runs: a screenshot every `capture_every` generations.
        if let Some(every) = self.capture_every {
            let generation = self.snapshot.as_ref().map_or(0, |s| s.generation);
            if generation > 0
                && generation.is_multiple_of(every)
                && generation != self.captured_generation
                && !self.screenshot_waiting
            {
                self.captured_generation = generation;
                self.capture_generation = Some(generation);
                self.screenshot_pending = true;
                self.screenshot_waiting = true;
            }
        }
        // Screenshot button: ask the viewport for one frame and save it as PNG.
        if self.screenshot_pending {
            self.screenshot_pending = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
                ScreenshotRequest,
            )));
        }
        // Developer screenshots: EVOLUTION_SMOKE_SEEK=<seconds> holds the replay at that time.
        if self.capture_path.is_some()
            && let Some(seconds) = std::env::var("EVOLUTION_SMOKE_SEEK")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
            && let Some(p) = self.playback.as_mut()
        {
            p.seek((seconds * physics::rate() as f32) as u32);
            self.playing = false;
        }
        // Explicit opt-in capture hook for repeatable native rendering/performance checks.
        if self.capture_path.is_some()
            && self.started.elapsed() > smoke_capture_delay()
            && !self.capture_requested
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            self.capture_requested = true;
        }
        if self.screenshot_waiting || self.capture_path.is_some() {
            for event in ctx.input(|i| i.events.clone()) {
                let egui::Event::Screenshot {
                    user_data, image, ..
                } = event
                else {
                    continue;
                };
                if user_data
                    .data
                    .as_ref()
                    .is_some_and(|data| data.is::<ScreenshotRequest>())
                {
                    self.screenshot_waiting = false;
                    if let Some(generation) = self.capture_generation.take() {
                        let path = format!("runs/progress-gen{generation}.png");
                        let bytes: Vec<u8> =
                            image.pixels.iter().flat_map(|p| p.to_array()).collect();
                        let _ = std::fs::create_dir_all("runs");
                        if let Err(e) = image::save_buffer(
                            &path,
                            &bytes,
                            image.size[0] as u32,
                            image.size[1] as u32,
                            image::ColorType::Rgba8,
                        ) {
                            eprintln!("Screenshot: {e}");
                        }
                        continue;
                    }
                    match save_screenshot(&image, std::path::Path::new("runs")) {
                        Ok(path) => {
                            self.message = Some(format!("Screenshot saved to {}", path.display()));
                            self.runs_checked = Instant::now() - RUNS_REFRESH;
                        }
                        Err(error) => self.message = Some(format!("Screenshot failed: {error}")),
                    }
                } else if let Some(path) = self.capture_path.clone() {
                    let bytes: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                    if let Err(e) = image::save_buffer(
                        &path,
                        &bytes,
                        image.size[0] as u32,
                        image.size[1] as u32,
                        image::ColorType::Rgba8,
                    ) {
                        eprintln!("Screenshot: {e}");
                    }
                    let mut frames: Vec<_> = self.frame_times.iter().copied().collect();
                    frames.sort_by(f32::total_cmp);
                    let p95 = frames.get(frames.len() * 95 / 100).copied().unwrap_or(0.) * 1000.;
                    eprintln!(
                        "Native UI: {} frames, p95 {p95:.2} ms, screenshot {path}",
                        frames.len()
                    );
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
        // eframe's frame time: the previous frame's update, tessellation
        // and paint, without the wait for vsync.
        if self.worker.measuring.load(Ordering::Relaxed)
            && let Some(seconds) = frame.info().cpu_usage
        {
            self.bench_work.push(seconds);
        }
    }
}
/// PCI vendor of the GPU GNOME's compositor renders on: the card tagged
/// `mutter-device-preferred-primary` by udev, otherwise the boot VGA card.
fn compositor_vendor() -> Option<u32> {
    let cards: Vec<_> = std::fs::read_dir("/sys/class/drm")
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card") && !n.contains('-'))
        })
        .collect();
    let read = |path: std::path::PathBuf| std::fs::read_to_string(path).ok();
    let vendor = |card: &std::path::PathBuf| {
        read(card.join("device/vendor"))
            .and_then(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
    };
    let tagged = cards.iter().find(|card| {
        read(card.join("dev")).is_some_and(|dev| {
            read(format!("/run/udev/data/c{}", dev.trim()).into())
                .is_some_and(|data| data.contains("mutter-device-preferred-primary"))
        })
    });
    tagged
        .or_else(|| {
            cards
                .iter()
                .find(|card| read(card.join("device/boot_vga")).is_some_and(|v| v.trim() == "1"))
        })
        .and_then(vendor)
}
fn ui_frame_interval() -> Option<Duration> {
    static INTERVAL: std::sync::OnceLock<Option<Duration>> = std::sync::OnceLock::new();
    *INTERVAL.get_or_init(|| {
        let fps = std::env::var("EVOLUTION_UI_FPS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(60.0);
        // Aim slightly early so vsync-paced frames land on every other 120 Hz refresh.
        (fps > 0.0).then(|| Duration::from_secs_f64(0.97 / fps))
    })
}
fn paint_card(
    painter: &egui::Painter,
    card: &crate::worker::Card,
    rect: Rect,
    hovered: bool,
    theme: Theme,
) {
    crate::theme::plate(
        painter,
        rect,
        theme,
        if hovered {
            theme.card_hover
        } else {
            theme.card
        },
        hovered,
    );
    thumbnail(
        painter,
        &card.creature,
        Rect::from_min_max(
            rect.left_top() + Vec2::new(12., 46.),
            rect.right_bottom() - Vec2::new(12., 30.),
        ),
    );
    painter.text(
        rect.left_top() + Vec2::new(9., 8.),
        Align2::LEFT_TOP,
        if card.descriptor.is_some() {
            format!("#{}", card.rank + 1)
        } else {
            format!("ID {}", card.creature.id)
        },
        FontId::proportional(14.),
        theme.accent.gamma_multiply(0.85),
    );
    painter.text(
        rect.left_top() + Vec2::new(9., 26.),
        Align2::LEFT_TOP,
        species_name(&card.creature),
        FontId::proportional(14.5),
        theme.ink,
    );
    if card.innovation_reserve {
        crate::theme::caps_text(
            painter,
            rect.right_top() + Vec2::new(-9., 9.),
            Align2::RIGHT_TOP,
            "New body",
            13.,
            theme.cold,
        );
    }
    let (label, score_color) = if !card.score.is_finite() {
        if card.parent_score.is_finite() && card.parent_score > FAILED {
            (format!("Parent {:.3} m", card.parent_score), theme.muted)
        } else if card.parent_score.is_finite() {
            ("Parent failed".into(), theme.warn)
        } else {
            ("Trial pending".into(), theme.muted)
        }
    } else if card.score <= FAILED {
        ("Failed trial".into(), theme.warn)
    } else {
        (
            format!("{:.3} m", card.score),
            if card.survivor {
                theme.accent
            } else {
                theme.ink
            },
        )
    };
    painter.text(
        rect.left_bottom() + Vec2::new(9., -9.),
        Align2::LEFT_BOTTOM,
        label,
        FontId::proportional(15.),
        score_color,
    );
}
/// How a creature came to be, in the words the lineage uses.
/// Space between island cards, and a card's height.
const ISLAND_GAP: f32 = 10.;
const ISLAND_HEIGHT: f32 = 346.;
/// Words for the emitter shares of an island's elites, in `Emitter::ALL` order.
const ORIGIN_SHORT: [&str; 4] = ["Tuned", "Reshaped", "Novel", "New"];
/// Their colors: amber, rust, cold blue and olive.
const ORIGIN_COLORS: [Color32; 4] = [
    Color32::from_rgb(222, 160, 60),
    Color32::from_rgb(178, 92, 58),
    Color32::from_rgb(96, 154, 196),
    Color32::from_rgb(132, 140, 76),
];
/// What an island card says about its nurseries: their size and best
/// distance, when they graduate next, and what the last graduation kept.
fn nursery_lines(island: &crate::worker::IslandSummary, generation: u32) -> [String; 2] {
    let every = crate::qd::NURSERY_GENERATIONS;
    let next = (generation / every + 1) * every;
    let first = if island.nursery == 0 {
        format!("Nurseries empty. Next: gen {next}")
    } else {
        format!(
            "Nurseries {} bodies, best {:.1} m. Next: gen {next}",
            island.nursery, island.nursery_best
        )
    };
    let g = island.graduation;
    let second = if g.generation == 0 {
        "No graduates yet this session".to_owned()
    } else {
        format!(
            "Gen {}: {} of {} graduates kept, {} in all",
            g.generation, g.kept, g.sent, g.kept_total
        )
    };
    [first, second]
}
/// What an island card says about migration: the last exchange, or when the
/// next one comes.
fn migration_lines(
    migration: Option<&crate::worker::MigrationSummary>,
    island: usize,
    generation: u32,
) -> [String; 2] {
    let next =
        (generation / crate::storage::MIGRATION_INTERVAL + 1) * crate::storage::MIGRATION_INTERVAL;
    let hub = island == crate::storage::hub_island();
    match migration.filter(|m| !m.exchange.is_empty()) {
        Some(m) if hub => {
            let (got, kept) = m.hub_received();
            [
                format!(
                    "Gen {}: got {got} copies from the islands, kept {kept}",
                    m.generation
                ),
                format!("Sends nothing back. Next: gen {next}"),
            ]
        }
        Some(m) => {
            let (sent, kept) = m.exchange.get(island).copied().unwrap_or((0, 0));
            [
                format!(
                    "Gen {}: copied {sent} to the hub, it kept {kept}",
                    m.generation
                ),
                format!("Receives no migrants. Next: gen {next}"),
            ]
        }
        None if hub => [
            "No copies yet this session".to_owned(),
            format!("Copies arrive at generation {next}"),
        ],
        None => [
            "Isolated: receives no migrants".to_owned(),
            format!("Copies go to the hub at generation {next}"),
        ],
    }
}
/// Percent shares that add to 100 (largest remainder), so the legend never
/// reads 99 or 101.
fn percent_shares(counts: &[usize]) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return vec![0; counts.len()];
    }
    let mut shares: Vec<usize> = counts.iter().map(|&c| c * 100 / total).collect();
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(counts[i] * 100 % total));
    let missing = 100 - shares.iter().sum::<usize>();
    for &i in order.iter().take(missing) {
        shares[i] += 1;
    }
    shares
}
/// "Island 1" to "Island 4" for the isolated islands, "Hub" for the hub.
pub(crate) fn island_name(index: usize) -> String {
    if index == crate::storage::hub_island() {
        "Hub".to_owned()
    } else {
        format!("Island {}", index + 1)
    }
}
/// Paints one island card and returns the creature the player clicked.
#[allow(clippy::too_many_arguments)]
fn paint_island(
    ui: &mut egui::Ui,
    rect: Rect,
    index: usize,
    island: &crate::worker::IslandSummary,
    migration: Option<&crate::worker::MigrationSummary>,
    generation: u32,
    shown: Option<u64>,
    theme: Theme,
) -> Option<Creature> {
    let painter = ui.painter().clone();
    crate::theme::plate(&painter, rect, theme, theme.card, false);
    let at = |x: f32, y: f32| rect.left_top() + Vec2::new(x, y);
    crate::theme::caps_text(
        &painter,
        at(12., 12.),
        Align2::LEFT_TOP,
        &island_name(index),
        14.,
        theme.ink,
    );
    painter.text(
        rect.right_top() + Vec2::new(-12., 12.),
        Align2::RIGHT_TOP,
        format!("{} ways of moving", number(island.moves)),
        FontId::proportional(14.5),
        theme.muted,
    );
    let mut clicked = None;
    // The best creature, left; its distance under it.
    let lead_rect = Rect::from_min_size(at(12., 36.), Vec2::new(rect.width() * 0.5 - 18., 104.));
    let best_text = if island.best.is_finite() {
        format!("Best {:.2} m", island.best)
    } else {
        "Empty".to_owned()
    };
    if let Some(leader) = &island.leader {
        let response = ui.interact(
            lead_rect,
            ui.id().with(("island_leader", index)),
            Sense::click(),
        );
        let lit = response.hovered() || shown == Some(leader.id);
        painter.rect_filled(
            lead_rect,
            6,
            if lit { theme.card_hover } else { theme.canvas },
        );
        thumbnail(&painter, leader, lead_rect);
        if response.clicked() {
            clicked = Some(leader.clone());
        }
        response.on_hover_text(format!(
            "{}\n{} nodes, {} muscles\nClick to replay",
            species_name(leader),
            leader.nodes.len(),
            leader.muscles.len()
        ));
    }
    painter.text(
        lead_rect.left_bottom() + Vec2::new(0., 6.),
        Align2::LEFT_TOP,
        best_text,
        FontId::proportional(15.),
        theme.accent,
    );
    // The next fastest elites, right, one row each.
    let list_left = lead_rect.right() + 12.;
    painter.text(
        Pos2::new(list_left, 38.0 + rect.top()),
        Align2::LEFT_TOP,
        "Top elites",
        FontId::proportional(14.),
        theme.muted,
    );
    for (row, (distance, creature)) in island.top.iter().enumerate() {
        let row_rect = Rect::from_min_size(
            Pos2::new(list_left, rect.top() + 56. + row as f32 * 28.),
            Vec2::new(rect.right() - 12. - list_left, 26.),
        );
        let response = ui.interact(
            row_rect,
            ui.id().with(("island_top", index, row)),
            Sense::click(),
        );
        let lit = response.hovered() || shown == Some(creature.id);
        painter.rect_filled(
            row_rect,
            4,
            if lit { theme.card_hover } else { theme.canvas },
        );
        thumbnail(
            &painter,
            creature,
            Rect::from_min_size(row_rect.left_top(), Vec2::new(40., 26.)),
        );
        painter.text(
            row_rect.left_center() + Vec2::new(46., 0.),
            Align2::LEFT_CENTER,
            format!("{distance:.2} m"),
            FontId::proportional(14.5),
            theme.ink,
        );
        if response.clicked() {
            clicked = Some(creature.clone());
        }
        response.on_hover_text(format!("{}\nClick to replay", species_name(creature)));
    }
    // Who bred the island's elites.
    painter.text(
        at(12., 168.),
        Align2::LEFT_TOP,
        "Bred by",
        FontId::proportional(14.),
        theme.muted,
    );
    let bar = Rect::from_min_size(at(12., 184.), Vec2::new(rect.width() - 24., 10.));
    painter.rect_filled(bar, 3, theme.canvas);
    let shares = percent_shares(&island.origins);
    let total: usize = island.origins.iter().sum();
    let mut x = bar.left();
    for (i, &count) in island.origins.iter().enumerate() {
        if total == 0 || count == 0 {
            continue;
        }
        let w = bar.width() * count as f32 / total as f32;
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(x, bar.top()), Vec2::new(w, bar.height())),
            0,
            ORIGIN_COLORS[i],
        );
        x += w;
    }
    let legend_width = (rect.width() - 24.) / 2.;
    for i in 0..island.origins.len() {
        let cell = at(
            12. + (i % 2) as f32 * legend_width,
            200. + (i / 2) as f32 * 16.,
        );
        painter.rect_filled(
            Rect::from_min_size(cell + Vec2::new(0., 3.), Vec2::splat(9.)),
            1,
            ORIGIN_COLORS[i],
        );
        painter.text(
            cell + Vec2::new(14., 0.),
            Align2::LEFT_TOP,
            format!("{} {}%", ORIGIN_SHORT[i], shares[i]),
            FontId::proportional(14.),
            theme.ink,
        );
    }
    let lines: Vec<String> = nursery_lines(island, generation)
        .into_iter()
        .chain(migration_lines(migration, index, generation))
        .collect();
    let mut y = 238.;
    for line in &lines {
        let galley = painter.layout(
            line.clone(),
            FontId::proportional(14.),
            theme.muted,
            rect.width() - 24.,
        );
        let height = galley.size().y;
        painter.galley(at(12., y), galley, theme.muted);
        y += height + 2.;
    }
    clicked
}
fn origin_words(emitter: crate::qd::Emitter) -> &'static str {
    match emitter {
        crate::qd::Emitter::Cma => "fine-tuned from a parent",
        crate::qd::Emitter::Structural => "reshaped from a parent",
        crate::qd::Emitter::Novelty => "exploring a new way of moving",
        crate::qd::Emitter::Restart => "as a new random body",
    }
}
/// The next autochange step while autochange is on: "Next change at generation 60:
/// Wind to Breeze". The worker applies step `autochange_step` when a generation
/// that is a multiple of the interval begins.
fn autochange_forecast(config: &Config, generation: u32) -> Option<String> {
    let interval = crate::environment::autochange_interval(config.autochange)?;
    if interval == 0 {
        return None;
    }
    let at = (generation / interval + 1) * interval;
    let ladder = crate::environment::autochange_ladder();
    let &(index, level) = ladder.get(usize::from(config.autochange_step))?;
    let effect = &crate::environment::EFFECTS[index];
    Some(format!(
        "Next change at generation {at}: {} to {}",
        effect.name, effect.levels[level]
    ))
}
/// Whether the theater switches to the champion: only when the player has
/// not pinned a creature, a champion exists, and a different creature is on
/// screen. A new distance record makes a new champion, so the switch happens
/// as soon as the record lands.
fn follows_champion(pinned: bool, showing: Option<u64>, champion: Option<u64>) -> bool {
    !pinned && champion.is_some() && showing != champion
}
/// A world change on the chart: the first generation in the new world and a
/// short name for what changed.
struct WorldMark {
    generation: u32,
    label: String,
    autochange: bool,
}
/// A chart label cut to fit beside its line.
fn short_label(label: &str) -> String {
    let first = label.split(',').next().unwrap_or(label);
    if first.chars().count() > 22 {
        format!("{}...", first.chars().take(20).collect::<String>())
    } else {
        first.to_owned()
    }
}
/// Every world change, oldest first. Events give the change as soon as its
/// generation starts. History pairs fill in what events lack, such as after
/// loading a save, when the feed is rebuilt from the history too.
fn world_marks(events: &[crate::worker::Event], history: &[Stats]) -> Vec<WorldMark> {
    let mut marks: Vec<WorldMark> = events
        .iter()
        .filter(|e| matches!(e.kind, EventKind::World | EventKind::Autochange))
        .map(|e| WorldMark {
            generation: e.generation,
            label: e.text.split(". ").next().unwrap_or(&e.text).to_owned(),
            autochange: e.kind == EventKind::Autochange,
        })
        .collect();
    for pair in history.windows(2) {
        let generation = pair[1].generation;
        if pair[1].config.physics_differs(&pair[0].config)
            && !marks.iter().any(|m| m.generation == generation)
        {
            marks.push(WorldMark {
                generation,
                label: crate::worker::world_change_text(&pair[0].config, &pair[1].config)
                    .unwrap_or_else(|| "The world changed".into()),
                autochange: pair[1].config.autochange_step != pair[0].config.autochange_step
                    && pair[1].config.autochange > 0,
            });
        }
    }
    marks.sort_by_key(|m| m.generation);
    marks
}
/// Whether two configs have every effect, including autochange, at the same level.
fn worlds_match(a: &Config, b: &Config) -> bool {
    crate::environment::EFFECTS
        .iter()
        .all(|effect| effect.level(a) == effect.level(b))
}
/// The Generation tile's second line. Percent rounds down, so "100%" only
/// shows when every creature has a result.
fn generation_progress(completed: usize, population: usize, running: bool, rate: f64) -> String {
    let done = completed.min(population);
    if !running {
        return "Paused".to_owned();
    }
    if done >= population {
        return "Finishing the generation".to_owned();
    }
    let percent = (done as f64 / population as f64 * 100.0).floor();
    if rate > 0.0 {
        let left = (population - done) as f64 / rate;
        format!("{percent:.0}% done · about {} left", seconds_text(left))
    } else {
        format!("{percent:.0}% done")
    }
}
/// Whether every effect except the autochange schedule sits at its calm level.
fn world_is_calm(config: &Config) -> bool {
    crate::environment::EFFECTS
        .iter()
        .filter(|effect| effect.name != "Autochange environment")
        .all(|effect| effect.level(config) == effect.calm)
}
/// An effect and its level in a few words: "Mud: Deep", or just "Heat wave"
/// when the level carries the effect's own name.
fn effect_text(effect: &crate::environment::Effect, config: &Config) -> String {
    let level = effect.levels[effect.level(config)];
    if level == effect.name {
        level.to_owned()
    } else {
        format!("{}: {}", effect.name, level)
    }
}
/// The world in a few words: "Calm world", or the effects away from calm,
/// such as "Ground: Rough, 8 cm · Hurdles: Low".
fn world_summary(config: &Config) -> String {
    let parts: Vec<String> = crate::environment::EFFECTS
        .iter()
        .filter(|effect| effect.name != "Autochange environment")
        .filter(|effect| effect.level(config) != effect.calm)
        .map(|effect| effect_text(effect, config))
        .collect();
    if parts.is_empty() {
        "Calm world".to_owned()
    } else {
        parts.join(" · ")
    }
}
/// Width of the effect name column in the World panel.
const EFFECT_NAME_WIDTH: f32 = 100.0;
/// One effect as its name and a segmented bar of its levels. The lit segment
/// is the current level: amber when the effect is away from calm. While a
/// change waits for the next generation, the name turns cold blue and a blue
/// outline marks the level that still runs. Returns true when the player
/// picked another level.
fn effect_row(
    ui: &mut egui::Ui,
    effect: &crate::environment::Effect,
    config: &mut Config,
    live: Option<&Config>,
    theme: Theme,
) -> bool {
    let level = effect.level(config);
    let away = level != effect.calm;
    let running = live.map(|live| effect.level(live));
    let waiting = running.is_some_and(|running| running != level);
    let color = if waiting {
        theme.cold
    } else if away {
        theme.accent
    } else {
        theme.ink
    };
    let label = if effect.name == "Autochange environment" {
        "Autochange"
    } else {
        effect.name
    };
    let mut picked = None;
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 0.;
        let (name_rect, name) =
            ui.allocate_exact_size(Vec2::new(EFFECT_NAME_WIDTH, LEVEL_HEIGHT), Sense::hover());
        ui.painter().text(
            name_rect.left_center(),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(16.5),
            color,
        );
        if let (true, Some(running)) = (waiting, running) {
            name.on_hover_text(format!(
                "Now {}. {} from the next generation.",
                effect.levels[running], effect.levels[level]
            ));
        } else {
            name.on_hover_text(format!("{}. {}", effect.name, effect.why));
        }
        ui.vertical(|ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(1., 1.);
                let font = FontId::proportional(14.);
                let galleys: Vec<_> = effect
                    .levels
                    .iter()
                    .map(|text| {
                        let short = text.split(',').next().unwrap_or(text);
                        ui.painter()
                            .layout_no_wrap(short.to_owned(), font.clone(), theme.ink)
                    })
                    .collect();
                const PAD: f32 = 8.;
                let count = galleys.len().max(1) as f32;
                let natural: f32 =
                    galleys.iter().map(|g| g.size().x + PAD).sum::<f32>() + count - 1.;
                let room = ui.available_width();
                // One line: the segments share the whole width. Too long for
                // one line, they keep their own width and wrap.
                let extra = if natural <= room {
                    ((room - natural) / count).floor()
                } else {
                    0.
                };
                let last = galleys.len().saturating_sub(1);
                for (i, galley) in galleys.into_iter().enumerate() {
                    let size = Vec2::new(galley.size().x + PAD + extra, LEVEL_HEIGHT);
                    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
                    let lit = i == level;
                    let hovered = response.hovered();
                    let fill = match (lit, away, hovered) {
                        (true, true, _) => theme.stop_fill,
                        (true, false, _) => theme.armed_fill,
                        (false, _, true) => ui.visuals().widgets.hovered.weak_bg_fill,
                        _ => ui.visuals().widgets.inactive.weak_bg_fill,
                    };
                    let corner = egui::CornerRadius {
                        nw: if i == 0 { 3 } else { 0 },
                        sw: if i == 0 { 3 } else { 0 },
                        ne: if i == last { 3 } else { 0 },
                        se: if i == last { 3 } else { 0 },
                    };
                    ui.painter().rect_filled(rect, corner, fill);
                    ui.painter().rect_stroke(
                        rect,
                        corner,
                        Stroke::new(if lit { 2.5 } else { 1.5 }, theme.ink),
                        egui::StrokeKind::Inside,
                    );
                    if waiting && running == Some(i) {
                        ui.painter().rect_stroke(
                            rect,
                            corner,
                            Stroke::new(3., theme.cold),
                            egui::StrokeKind::Inside,
                        );
                    }
                    let text_color = match (lit, away) {
                        (true, true) => theme.go_text,
                        (true, false) => theme.ink,
                        _ => theme.ink,
                    };
                    let at = rect.center() - galley.size() / 2.;
                    ui.painter()
                        .galley_with_override_text_color(at, galley, text_color);
                    let text = effect.levels[i];
                    let hover = if i == effect.calm {
                        format!("{text}. The calm world.")
                    } else {
                        format!("{text}. {}", effect.why)
                    };
                    if response.on_hover_text(hover).clicked() && i != level {
                        picked = Some(i);
                    }
                }
            });
        });
    });
    if let Some(i) = picked {
        effect.set_level(config, i);
    }
    picked.is_some()
}
/// Fixed camera and palette of an exported GIF. A frame is rasterized into one
/// reused buffer, so painting allocates nothing beyond the frame itself.
struct GifCamera {
    /// World y at the bottom edge.
    y0: f32,
    /// Pixels per meter.
    scale: f32,
    /// Screen x the followed center of mass stays at, in pixels.
    anchor_x: f32,
}
impl GifCamera {
    fn fit(nodes: &[Node], frames: &[Vec<[f32; 2]>]) -> Self {
        let mut min_y = 0.0f32;
        let mut max_y = 0.4f32;
        for frame in frames {
            for (position, node) in frame.iter().zip(nodes) {
                min_y = min_y.min(position[1] - node.radius);
                max_y = max_y.max(position[1] + node.radius);
            }
        }
        min_y -= 0.15;
        let range = (max_y - min_y).max(0.4);
        let scale = (GIF_HEIGHT as f32 / (range * 1.15)).min(GIF_MAX_SCALE);
        let y0 = 0.5 * (min_y + max_y) - 0.5 * GIF_HEIGHT as f32 / scale;
        Self {
            y0,
            scale,
            anchor_x: GIF_WIDTH as f32 * 0.38,
        }
    }
    fn origin_x(&self, center_x: f32) -> f32 {
        center_x - self.anchor_x / self.scale
    }
    fn screen(&self, origin_x: f32, position: [f32; 2]) -> (f32, f32) {
        (
            (position[0] - origin_x) * self.scale,
            GIF_HEIGHT as f32 - (position[1] - self.y0) * self.scale,
        )
    }
}
fn gif_color(c: Color32) -> Rgba<u8> {
    Rgba([c.r(), c.g(), c.b(), 255])
}
fn gif_put(buffer: &mut RgbaImage, x: f32, y: f32, color: Rgba<u8>) {
    let x = x.round() as i32;
    let y = y.round() as i32;
    if x >= 0 && y >= 0 && (x as u32) < buffer.width() && (y as u32) < buffer.height() {
        buffer.put_pixel(x as u32, y as u32, color);
    }
}
fn gif_disc(buffer: &mut RgbaImage, center: (f32, f32), radius: f32, color: Rgba<u8>) {
    let radius = radius.max(0.5);
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as f32 <= r2 {
                gif_put(buffer, center.0 + dx as f32, center.1 + dy as f32, color);
            }
        }
    }
}
fn gif_ring(buffer: &mut RgbaImage, center: (f32, f32), radius: f32, width: f32, color: Rgba<u8>) {
    let outer = radius + width * 0.5;
    let inner = (radius - width * 0.5).max(0.0);
    let r = outer.ceil() as i32;
    let (outer2, inner2) = (outer * outer, inner * inner);
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as f32;
            if d2 <= outer2 && d2 >= inner2 {
                gif_put(buffer, center.0 + dx as f32, center.1 + dy as f32, color);
            }
        }
    }
}
fn gif_line(buffer: &mut RgbaImage, a: (f32, f32), b: (f32, f32), radius: f32, color: Rgba<u8>) {
    let radius = radius.max(0.5);
    let length = (b.0 - a.0).hypot(b.1 - a.1);
    let steps = (length / (radius * 0.5).max(1.0)).ceil().max(1.0) as u32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        gif_disc(
            buffer,
            (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t),
            radius,
            color,
        );
    }
}
fn gif_cross(buffer: &mut RgbaImage, center: (f32, f32), radius: f32, color: Rgba<u8>) {
    let d = radius.max(3.5);
    gif_line(
        buffer,
        (center.0 - d, center.1 - d),
        (center.0 + d, center.1 + d),
        1.25,
        color,
    );
    gif_line(
        buffer,
        (center.0 - d, center.1 + d),
        (center.0 + d, center.1 - d),
        1.25,
        color,
    );
}
/// Mass-weighted center x of a pose; the GIF camera follows it.
fn pose_center_x(nodes: &[Node], positions: &[[f32; 2]]) -> f32 {
    let mut mass = 0.0;
    let mut x = 0.0;
    for (node, position) in nodes.iter().zip(positions) {
        mass += node.mass;
        x += node.mass * position[0];
    }
    if mass > 0.0 { x / mass } else { 0.0 }
}
/// The viewport scene painted into a pixel buffer for one recorded pose.
struct GifScene<'a> {
    creature: &'a Creature,
    config: &'a Config,
    nodes: &'a [Node],
    camera: &'a GifCamera,
}
impl GifScene<'_> {
    fn render(
        &self,
        buffer: &mut RgbaImage,
        positions: &[[f32; 2]],
        time: f32,
        fallen: bool,
        contact: &[bool],
        broken: &[bool],
    ) {
        let dark = gif_color(OUTLINE);
        let origin_x = self.camera.origin_x(pose_center_x(self.nodes, positions));
        let at = |position: [f32; 2]| self.camera.screen(origin_x, position);
        // The overcast sky, fading to haze toward the ground.
        for (_, y, pixel) in buffer.enumerate_pixels_mut() {
            *pixel = gif_color(mix_color(
                SKY_TOP,
                SKY_HORIZON,
                y as f32 / (GIF_HEIGHT as f32 * 0.8),
            ));
        }
        // A meter grid; it scrolls with the follow camera, so motion reads even
        // when the creature holds its screen position.
        let right = origin_x + GIF_WIDTH as f32 / self.camera.scale;
        let grid = gif_color(Color32::from_rgb(104, 110, 112));
        for meter in origin_x.floor() as i32..=right.ceil() as i32 {
            let screen_x = (meter as f32 - origin_x) * self.camera.scale;
            gif_line(
                buffer,
                (screen_x, 0.0),
                (screen_x, GIF_HEIGHT as f32),
                0.5,
                grid,
            );
        }
        if self.config.ground {
            let hash = physics::quake_hash(self.creature.id);
            let amplitude = physics::terrain_amplitude(self.config.terrain)
                + self.config.quake * physics::quake_scale(hash);
            let phase = if self.config.quake > 0.0 {
                physics::quake_phase(hash)
            } else {
                0.0
            };
            let ground = gif_color(GROUND_TOP);
            let edge = gif_color(GROUND_EDGE);
            for px in 0..GIF_WIDTH {
                let world_x = origin_x + px as f32 / self.camera.scale;
                let (height, _) = physics::ground(
                    world_x,
                    amplitude,
                    self.config.slope,
                    self.config.gaps,
                    self.config.hurdles,
                    phase,
                );
                let surface = GIF_HEIGHT as f32 - (height - self.camera.y0) * self.camera.scale;
                let top = surface.floor().max(0.0) as u32;
                for y in top..GIF_HEIGHT {
                    buffer.put_pixel(px, y, ground);
                }
                if top < GIF_HEIGHT {
                    buffer.put_pixel(px, top, edge);
                    if top + 1 < GIF_HEIGHT {
                        buffer.put_pixel(px, top + 1, edge);
                    }
                }
            }
        }
        for bone in &self.creature.bones {
            let a = at(positions[bone.a as usize]);
            let b = at(positions[bone.b as usize]);
            let half = (self.camera.scale * 0.032).max(3.0) * 0.5;
            gif_line(buffer, a, b, half + 1.5, dark);
            gif_line(buffer, a, b, half, gif_color(BONE));
        }
        for bone in self.creature.bones.iter().filter(|b| b.organ_mass > 0.0) {
            let a = positions[bone.a as usize];
            let b = positions[bone.b as usize];
            let t = bone.organ_at;
            let center = at([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            let r = (0.04 * (bone.organ_mass / 0.1).sqrt() * self.camera.scale).max(2.5);
            gif_disc(buffer, center, r + 1.5, dark);
            gif_disc(buffer, center, r, gif_color(ORGAN));
        }
        for m in &self.creature.muscles {
            let bone_a = self.creature.bones[m.bone_a as usize];
            let bone_b = self.creature.bones[m.bone_b as usize];
            let point = |bone: crate::evolution::Bone, t: f32| {
                let a = positions[bone.a as usize];
                let b = positions[bone.b as usize];
                at([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
            };
            let a = point(bone_a, m.anchor_a);
            let b = point(bone_b, m.anchor_b);
            // A fallen creature's muscles are limp.
            let contraction = if fallen {
                0.0
            } else {
                1. - ((physics::target(m, time) - m.short) / (m.long - m.short).max(1e-5))
            };
            let half = (self.camera.scale * 0.017 * (1. + 0.45 * contraction)).max(2.) * 0.5;
            gif_line(buffer, a, b, half + 1.5, dark);
            gif_line(
                buffer,
                a,
                b,
                half,
                gif_color(mix_color(MUSCLE_REST, MUSCLE_ACTIVE, contraction)),
            );
        }
        for (i, n) in self.nodes.iter().enumerate() {
            let center = at(positions[i]);
            let r = (n.radius * self.camera.scale).max(2.);
            gif_disc(buffer, center, r + 1.5, dark);
            gif_disc(buffer, center, r, gif_color(node_color(n.friction)));
            if contact.get(i).copied().unwrap_or(false) {
                gif_ring(buffer, center, r + 2.5, 2.0, gif_color(TOUCHDOWN));
            }
            if broken.get(i).copied().unwrap_or(false) {
                gif_ring(buffer, center, r + 2.5, 2.0, gif_color(FALLEN));
                gif_cross(buffer, center, r, gif_color(FALLEN));
            }
        }
        if let Some(head) = self.nodes.first() {
            let center = at(positions[0]);
            let r = (head.radius * self.camera.scale).max(2.);
            gif_disc(
                buffer,
                (center.0 + r * 0.4, center.1 - r * 0.2),
                r * 0.3,
                gif_color(EYE),
            );
            gif_disc(
                buffer,
                (center.0 + r * 0.48, center.1 - r * 0.2),
                r * 0.15,
                dark,
            );
            if fallen {
                gif_ring(buffer, center, r + 1.5, 2.0, gif_color(FALLEN));
                gif_cross(buffer, center, r, gif_color(FALLEN));
            }
        }
    }
}
/// Writes an animated GIF of one recorded trial. `ticks` are frame indices in
/// increasing order; the frame delay follows their average spacing, so the GIF
/// plays at the speed the trial was simulated. Returns the frame count.
fn write_creature_gif(
    creature: &Creature,
    config: &Config,
    frames: &[Vec<[f32; 2]>],
    broken_joints: &[u64],
    ticks: &[u32],
    fall: Option<(u32, f32)>,
    path: &std::path::Path,
) -> anyhow::Result<usize> {
    let nodes = &physics::nodes(creature);
    // The camera fits the frames the GIF shows.
    let shown: Vec<Vec<[f32; 2]>> = ticks
        .iter()
        .filter_map(|&tick| frames.get(tick as usize).cloned())
        .collect();
    let camera = GifCamera::fit(nodes, &shown);
    let scene = GifScene {
        creature,
        config,
        nodes,
        camera: &camera,
    };
    let delay = if ticks.len() > 1 {
        let span = ticks[ticks.len() - 1].saturating_sub(ticks[0]) as f32;
        let mean = span / (ticks.len() - 1) as f32;
        ((mean * physics::dt() * 100.0).round() as u32).clamp(2, 200)
    } else {
        10
    };
    let delay = GifDelay::from_numer_denom_ms(delay * 10, 1);
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = GifEncoder::new_with_speed(file, 30);
    encoder.set_repeat(GifRepeat::Infinite)?;
    let mut buffer = RgbaImage::new(GIF_WIDTH, GIF_HEIGHT);
    let mut contact = vec![false; nodes.len()];
    let mut broken = vec![false; nodes.len()];
    let mut written = 0usize;
    for &tick in ticks {
        let Some(frame) = frames.get(tick as usize) else {
            break;
        };
        node_contact(nodes, frame, creature, config, &mut contact);
        let bits = broken_joints.get(tick as usize).copied().unwrap_or(0);
        broken_nodes(creature, bits, &mut broken);
        let time = tick.saturating_sub(physics::settle()) as f32 * physics::dt();
        let fallen = fall.is_some_and(|(fall_tick, _)| tick >= fall_tick);
        scene.render(&mut buffer, frame, time, fallen, &contact, &broken);
        encoder.encode_frame(GifFrame::from_parts(buffer.clone(), 0, 0, delay))?;
        written += 1;
    }
    // Dropping the encoder writes the GIF trailer and flushes the writer.
    drop(encoder);
    Ok(written)
}
/// Samples a playback into at most `GIF_MAX_FRAMES` frames and animates them.
fn export_creature_gif(playback: &Playback, path: &std::path::Path) -> anyhow::Result<usize> {
    let first = playback.trial_start();
    let last = playback.last_frame();
    let total = last.saturating_sub(first) as usize + 1;
    let stride = total.div_ceil(GIF_MAX_FRAMES).max(1);
    let ticks: Vec<u32> = (first..=last).step_by(stride).collect();
    write_creature_gif(
        &playback.creature,
        &playback.config,
        &playback.frames,
        &playback.forces.broken,
        &ticks,
        playback.fall,
        path,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::test_creature;

    #[test]
    fn a_leaper_keeps_its_peak_in_view_without_shrinking_the_body_too_far() {
        let typical = fit_zoom(0.5, 260.0);
        // A mild jump fits whole.
        assert!(player_zoom(0.5, 1.0, 260.0) < typical);
        assert!(player_zoom(0.5, 1.0, 260.0) * 1.0 <= 0.72 * 260.0 + 0.01);
        // A huge leap stops at 60% of the typical zoom.
        assert!((player_zoom(0.5, 30.0, 260.0) - typical * 0.6).abs() < 0.01);
        // A body that never leaves the ground keeps the typical fit.
        assert_eq!(player_zoom(0.5, 0.5, 260.0), typical);
    }
    #[test]
    fn default_zoom_follows_body_height() {
        let small = fit_zoom(0.3, 260.0);
        let tall = fit_zoom(1.5, 260.0);
        assert!(small > tall);
        assert!((tall * 1.5 / 260.0 - FIT_HEIGHT_SHARE).abs() < 0.01);
        assert_eq!(fit_zoom(0.001, 260.0), 450.0);
        assert_eq!(fit_zoom(100.0, 260.0), 40.0);
    }
    #[test]
    fn zoom_scales_a_plot_range_around_its_center() {
        assert_eq!(scaled_range(10.0..=20.0, 0.5), 12.5..=17.5);
        assert_eq!(scaled_range(10.0..=20.0, 2.0), 5.0..=25.0);
        assert_eq!(scaled_range(-4.0..=4.0, 1.0), -4.0..=4.0);
    }
    #[test]
    fn imported_creatures_round_trip_through_json() {
        let mut creature = test_creature();
        assert!(imported_creature(&mut creature).is_ok());
        let json = serde_json::to_string(&creature).unwrap();
        let mut loaded: Creature = serde_json::from_str(&json).unwrap();
        assert!(imported_creature(&mut loaded).is_ok());
        loaded.bones[0].b = 9;
        assert!(imported_creature(&mut loaded).is_err());
    }
    #[test]
    fn a_new_champion_replaces_the_old_one_at_once_unless_pinned() {
        // A record at generation 12 makes creature 42 the champion while 7,
        // the previous champion, plays: the view switches right away.
        assert!(follows_champion(false, Some(7), Some(42)));
        // Nothing on screen yet: the champion shows.
        assert!(follows_champion(false, None, Some(42)));
        // Already showing the champion: nothing to do.
        assert!(!follows_champion(false, Some(42), Some(42)));
        // A creature the player picked stays until Back to champion.
        assert!(!follows_champion(true, Some(7), Some(42)));
        // No finished generation, no champion.
        assert!(!follows_champion(false, Some(7), None));
    }
    #[test]
    fn gif_export_encodes_three_synthetic_frames() {
        use image::AnimationDecoder;
        use image::codecs::gif::GifDecoder;
        let creature = test_creature();
        let config = Config::default();
        let frames: Vec<Vec<[f32; 2]>> = vec![
            vec![[0.0, 0.10], [0.5, 0.10], [1.0, 0.10]],
            vec![[0.1, 0.20], [0.6, 0.20], [1.1, 0.20]],
            vec![[0.2, 0.10], [0.7, 0.10], [1.2, 0.10]],
        ];
        let dir = std::env::temp_dir().join(format!("evolution-gif-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("creature.gif");
        let written =
            write_creature_gif(&creature, &config, &frames, &[], &[0, 1, 2], None, &path).unwrap();
        assert_eq!(written, 3);
        // image::open proves the file is a decodable GIF.
        let first = image::open(&path).unwrap();
        assert_eq!((first.width(), first.height()), (GIF_WIDTH, GIF_HEIGHT));
        let decoder =
            GifDecoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap())).unwrap();
        let decoded = decoder.into_frames().collect_frames().unwrap();
        assert_eq!(decoded.len(), 3);
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
#[cfg(test)]
mod island_view_tests {
    use super::*;

    #[test]
    fn origin_shares_always_add_to_100() {
        assert_eq!(percent_shares(&[0, 0, 0, 0]), vec![0; 4]);
        for counts in [[1, 1, 1, 0], [7, 3, 3, 1], [5, 0, 0, 0], [1, 2, 4, 8]] {
            let shares = percent_shares(&counts);
            assert_eq!(shares.iter().sum::<usize>(), 100, "{counts:?}");
            for (share, count) in shares.iter().zip(counts) {
                assert_eq!(count == 0, *share == 0);
            }
        }
    }
}
/// How long a screenshot run waits before it captures: 8 s, or
/// `EVOLUTION_SMOKE_CAPTURE_AFTER` seconds (developer diagnostic).
fn smoke_capture_delay() -> Duration {
    static DELAY: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *DELAY.get_or_init(|| {
        let seconds = std::env::var("EVOLUTION_SMOKE_CAPTURE_AFTER")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|s| (1.0..=600.0).contains(s))
            .unwrap_or(8.0);
        Duration::from_secs_f64(seconds)
    })
}

/// The wild islands as a grid of small tiles: each tile is colored by its
/// best distance against the best of all wild islands and names its world.
/// A click returns the island's leader with its world, so the replay runs
/// where the score came from.
fn wild_tiles(
    ui: &mut egui::Ui,
    wild: &[crate::worker::IslandSummary],
    wins: &[u32],
    config: &Config,
    shown: Option<u64>,
    theme: &Theme,
) -> Option<(Creature, Config)> {
    let levels = crate::environment::wild_levels(config.seed);
    ui.add_space(6.);
    ui.label(
        RichText::new(format!(
            "Wild islands: {} worlds of their own. Each sends its best to the hub, where they run again in your world.",
            wild.len()
        ))
        .color(theme.muted),
    );
    // Which effects the worlds of the hub winners' islands hold, by the hub
    // cells their migrants took (Wang et al., 2019, POET).
    let mut by_effect = vec![0u32; crate::environment::EFFECTS.len()];
    for (w, levels) in levels.iter().enumerate() {
        let won = wins.get(crate::qd::MAIN_ISLANDS + w).copied().unwrap_or(0);
        for &(e, _) in levels {
            by_effect[e] += won;
        }
    }
    let mut ranked: Vec<(usize, u32)> = by_effect
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, n)| n > 0)
        .collect();
    ranked.sort_by_key(|&(e, n)| (std::cmp::Reverse(n), e));
    if !ranked.is_empty() {
        ui.label(
            RichText::new(format!(
                "Effects in the worlds of the hub winners: {}",
                ranked
                    .iter()
                    .take(6)
                    .map(|&(e, n)| format!("{} {n}", crate::environment::EFFECTS[e].name))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .small(),
        )
        .on_hover_text(
            "Each wild migrant that took a hub cell counts for every effect of its island's world",
        );
    }
    ui.add_space(4.);
    let top = wild
        .iter()
        .map(|w| w.best)
        .filter(|b| b.is_finite())
        .fold(0.0f32, f32::max)
        .max(0.01);
    let columns = (ui.available_width() / 96.).floor().max(4.) as usize;
    let width = (ui.available_width() - (columns - 1) as f32 * 4.) / columns as f32;
    let mut picked = None;
    for (row, chunk) in wild.chunks(columns).enumerate() {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.;
            for (offset, island) in chunk.iter().enumerate() {
                let w = row * columns + offset;
                let (rect, response) =
                    ui.allocate_exact_size(Vec2::new(width, 40.), Sense::click());
                let share = if island.best.is_finite() {
                    (island.best / top).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let fill = theme.panel.lerp_to_gamma(theme.accent, 0.15 + 0.6 * share);
                let painter = ui.painter();
                painter.rect_filled(rect, 4., fill);
                let lit = response.hovered()
                    || island.leader.as_ref().is_some_and(|c| Some(c.id) == shown);
                if lit {
                    painter.rect_stroke(
                        rect,
                        4.,
                        egui::Stroke::new(2., theme.ink),
                        egui::StrokeKind::Inside,
                    );
                }
                let best = if island.best.is_finite() {
                    format!("{:.1} m", island.best)
                } else {
                    "empty".into()
                };
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    format!("W{}\n{best}", w + 1),
                    egui::FontId::proportional(14.),
                    theme.ink,
                );
                let name = levels
                    .get(w)
                    .map(|l| crate::environment::wild_name(l))
                    .unwrap_or_default();
                let won = wins.get(crate::qd::MAIN_ISLANDS + w).copied().unwrap_or(0);
                let response = response.on_hover_text(format!(
                    "Wild island {}: {name}\nBest {best}, {} cells, {} in its nurseries, {won} hub cells won",
                    w + 1,
                    island.cells,
                    island.nursery
                ));
                if response.clicked()
                    && let (Some(leader), Some(l)) = (&island.leader, levels.get(w))
                {
                    picked = Some((leader.clone(), crate::environment::wild_world(config, l)));
                }
            }
        });
        ui.add_space(4.);
    }
    picked
}
