use crate::{
    config::Config,
    evolution::{Creature, FAILED},
    gpu::Gpu,
    physics::{self, Node},
    storage::{PERCENTILES, Stage, Stats},
    worker::{Command, Snapshot, Worker},
};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Legend, Line, Plot, Points};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
const MINT: Color32 = Color32::from_rgb(22, 122, 91);
const ORGAN: Color32 = Color32::from_rgb(190, 78, 104);
const AMBER: Color32 = Color32::from_rgb(164, 96, 24);
const FALLEN: Color32 = Color32::from_rgb(196, 64, 52);
const MUTED: Color32 = Color32::from_rgb(105, 121, 113);
const INK: Color32 = Color32::from_rgb(40, 55, 48);
const PANEL: Color32 = Color32::from_rgb(255, 255, 252);
const CANVAS: Color32 = Color32::from_rgb(244, 247, 242);
const VIEWPORT: Color32 = Color32::from_rgb(151, 203, 245);
const CARD: Color32 = Color32::from_rgb(255, 255, 253);
const CARD_HOVER: Color32 = Color32::from_rgb(238, 247, 241);
const CARD_BORDER: Color32 = Color32::from_rgb(218, 229, 221);
const GROUND: Color32 = Color32::from_rgb(121, 176, 89);
const GROUND_EDGE: Color32 = Color32::from_rgb(66, 118, 55);
const MUSCLE_REST: Color32 = Color32::from_rgb(249, 168, 191);
const MUSCLE_ACTIVE: Color32 = Color32::from_rgb(146, 16, 28);
const DEFAULT_CAMERA_ZOOM: f32 = 80.0;
/// UI surface colors for the active theme. The scene itself (sky, grass,
/// creatures) keeps fixed colors, so the viewport reads the same in both.
#[derive(Clone, Copy)]
struct Theme {
    panel: Color32,
    canvas: Color32,
    card: Color32,
    card_hover: Color32,
    card_border: Color32,
    ink: Color32,
    muted: Color32,
    accent: Color32,
}
impl Theme {
    fn of(dark: bool) -> Self {
        if dark {
            Self {
                panel: Color32::from_rgb(29, 34, 32),
                canvas: Color32::from_rgb(20, 24, 23),
                card: Color32::from_rgb(38, 45, 41),
                card_hover: Color32::from_rgb(48, 57, 52),
                card_border: Color32::from_rgb(62, 73, 67),
                ink: Color32::from_rgb(228, 234, 229),
                muted: Color32::from_rgb(150, 163, 155),
                accent: Color32::from_rgb(88, 205, 155),
            }
        } else {
            Self {
                panel: PANEL,
                canvas: CANVAS,
                card: CARD,
                card_hover: CARD_HOVER,
                card_border: CARD_BORDER,
                ink: INK,
                muted: MUTED,
                accent: MINT,
            }
        }
    }
}
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
fn file_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.0} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} B")
    }
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
/// Applies the custom style of the chosen theme.
fn apply_style(ctx: &egui::Context, dark: bool) {
    let theme = Theme::of(dark);
    ctx.set_theme(if dark {
        egui::Theme::Dark
    } else {
        egui::Theme::Light
    });
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 10.0);
    style.spacing.button_padding = Vec2::new(12.0, 8.0);
    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    visuals.override_text_color = Some(theme.ink);
    visuals.weak_text_color = Some(theme.muted);
    visuals.panel_fill = theme.panel;
    visuals.window_fill = theme.panel;
    visuals.extreme_bg_color = theme.canvas;
    visuals.code_bg_color = if dark { theme.canvas } else { VIEWPORT };
    visuals.faint_bg_color = theme.card_border;
    visuals.selection.bg_fill = if dark {
        Color32::from_rgb(45, 84, 66)
    } else {
        Color32::from_rgb(219, 239, 227)
    };
    visuals.selection.stroke = Stroke::new(1.0, theme.accent);
    visuals.hyperlink_color = theme.accent;
    style.visuals = visuals;
    style
        .text_styles
        .insert(egui::TextStyle::Body, FontId::proportional(14.0));
    style
        .text_styles
        .insert(egui::TextStyle::Heading, FontId::proportional(23.0));
    ctx.set_global_style(style);
}
pub fn launch(adapter_name: &str) -> anyhow::Result<()> {
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
            .with_title("Evolution · Creature Laboratory"),
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
            // Evaluation opens its own Vulkan devices; the render device only draws.
            let gpu = Gpu::new(&compute_name)?;
            Ok(Box::new(App::new(cc, gpu)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
/// Replays a creature's trial as simulated by the evaluation engines.
struct Playback {
    creature: Creature,
    config: Config,
    nodes: Vec<Node>,
    /// Node positions after each step, from the CPU evaluation engine.
    frames: Vec<Vec<[f32; 2]>>,
    tick: u32,
    accumulator: f32,
    /// Frame at which the trial ended early (a fall, a broken joint, or a
    /// shaken head), and the distance the trial kept from that moment.
    fall: Option<(u32, f32)>,
    /// The distance the CPU engine scored for this very recording.
    distance: f32,
}
impl Playback {
    fn new(creature: Creature, config: Config) -> Self {
        let mut normalized = creature.clone();
        crate::evolution::canonicalize_bone_order(&mut normalized);
        // The engine that recorded the frames also decides when the trial
        // ended and how far it got, so the replay shows exactly its score.
        let (frames, result) = crate::cpu_engine::replay(&normalized, &config);
        let nodes = physics::nodes(&normalized);
        let last_frame = frames.len().saturating_sub(1).min(u32::MAX as usize) as u32;
        let fall = (result.fall_time > 0.0).then(|| {
            let tick = physics::settle()
                .saturating_add((result.fall_time * physics::rate() as f32).round() as u32);
            (tick.min(last_frame), result.fitness)
        });
        let mut playback = Self {
            nodes,
            fall,
            distance: result.fitness,
            creature: normalized,
            config,
            tick: physics::settle()
                .min(last_frame)
                .saturating_add(1)
                .min(last_frame),
            frames,
            accumulator: 0.0,
        };
        playback.show();
        playback
    }
    fn reset(&mut self) {
        self.tick = self.trial_start().saturating_add(1).min(self.last_frame());
        self.show();
    }
    fn last_frame(&self) -> u32 {
        self.frames.len().saturating_sub(1).min(u32::MAX as usize) as u32
    }
    fn trial_start(&self) -> u32 {
        physics::settle().min(self.last_frame())
    }
    fn elapsed_seconds(&self) -> f32 {
        self.tick
            .saturating_sub(self.trial_start())
            .min(self.config.steps()) as f32
            * physics::dt()
    }
    fn seek(&mut self, elapsed_frame: u32) {
        self.tick = self
            .trial_start()
            .saturating_add(elapsed_frame)
            .min(self.last_frame());
        self.accumulator = 0.0;
        self.show();
    }
    /// Advances one physics step.
    fn advance(&mut self) {
        self.tick = self.tick.saturating_add(1).min(self.last_frame());
        self.show();
    }
    /// The fall, once the replay has reached it.
    fn fallen(&self) -> Option<(u32, f32)> {
        self.fall.filter(|&(tick, _)| self.tick >= tick)
    }
    fn show(&mut self) {
        if let Some(frame) = self.frames.get(self.tick as usize) {
            for (node, position) in self.nodes.iter_mut().zip(frame) {
                node.pos = *position;
            }
        }
    }
    /// Mass-weighted center of the body at a recorded frame.
    fn center_of_mass(&self, tick: u32) -> Option<[f32; 2]> {
        let frame = self.frames.get(tick as usize)?;
        let mut mass = 0.0;
        let mut center = [0.0; 2];
        for (node, position) in self.nodes.iter().zip(frame) {
            mass += node.mass;
            center[0] += node.mass * position[0];
            center[1] += node.mass * position[1];
        }
        (mass > 0.0).then(|| [center[0] / mass, center[1] / mass])
    }
    /// Center-of-mass speed over the last fifth of a second of recorded
    /// frames, in meters per second.
    fn speed(&self) -> f32 {
        let window = (physics::rate() / 5).max(1);
        let start = self.tick.saturating_sub(window);
        let (Some(now), Some(before)) =
            (self.center_of_mass(self.tick), self.center_of_mass(start))
        else {
            return 0.0;
        };
        let seconds = self.tick.saturating_sub(start) as f32 * physics::dt();
        if seconds <= 0.0 {
            return 0.0;
        }
        (now[0] - before[0]).hypot(now[1] - before[1]) / seconds
    }
    /// Distance covered so far, frozen at the fall the engine recorded.
    fn current_distance(&self) -> f32 {
        self.fallen()
            .map_or_else(|| physics::fitness(&self.nodes), |(_, distance)| distance)
    }
}
/// Behavior-axis bin counts, mirroring `qd::BINS` (ground contact, cadence,
/// bounce, height, feet). Bounce keeps one bin, so it adds no map cell.
const MAP_BINS: [usize; 5] = [6, 8, 1, 6, 5];
/// One occupied behavior cell of the archive map: the creature that holds it.
struct MapCell {
    score: f32,
    rank: usize,
    descriptor: crate::qd::Descriptor,
    emitter: Option<crate::qd::Emitter>,
    creature: Creature,
}
/// A sweep through the archive pages that fills `App::map_cells`.
#[derive(Clone, Copy)]
struct MapScan {
    /// Start offset of the next page to request.
    next: usize,
    /// Archive size when the sweep started.
    total: usize,
}
/// Which representation the Behavior archive tab shows.
#[derive(Clone, Copy, PartialEq)]
enum ArchiveView {
    Cards,
    Map,
}
/// One archive elite running in the race view.
struct RaceLane {
    rank: usize,
    id: u64,
    score: f32,
    playback: Playback,
}
/// Cold-to-hot color for a normalized map value.
fn heat_color(t: f32) -> Color32 {
    let cold = Color32::from_rgb(64, 98, 168);
    let mid = Color32::from_rgb(242, 201, 76);
    let hot = Color32::from_rgb(202, 58, 46);
    if t < 0.5 {
        mix_color(cold, mid, t * 2.0)
    } else {
        mix_color(mid, hot, (t - 0.5) * 2.0)
    }
}
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
/// Invented stems for automatic species names. A stable body-plan hash picks
/// one; the gait word is added after it.
const SPECIES_STEMS: [&str; 16] = [
    "Vex", "Tor", "Quil", "Nym", "Zeb", "Cro", "Fen", "Lum", "Tar", "Wisp", "Brak", "Ovi", "Pyr",
    "Sable", "Dro", "Ril",
];
/// Short deterministic species name from body counts, a body-plan hash and
/// the muscles' commanded rhythm. It uses only creature data, so archive
/// cards, lineage tiles and race lanes agree without asking the worker.
fn species_name(creature: &Creature) -> String {
    let (nodes, bones, muscles) = body_counts(creature);
    // Order-independent sums keep the name stable across bone reordering
    // (playbacks canonicalize their copy of the creature).
    let mut plan = ((nodes as u64) << 42) ^ ((bones as u64) << 21) ^ muscles as u64;
    for bone in &creature.bones {
        plan = plan.wrapping_add(
            (bone.a as u64)
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(bone.b as u64)
                .wrapping_add(((bone.rest_length * 100.0) as u64).wrapping_mul(0xbf58476d1ce4e5b9)),
        );
    }
    for muscle in &creature.muscles {
        plan = plan.wrapping_add(
            (muscle.bone_a as u64)
                .wrapping_mul(0x94d049bb133111eb)
                .wrapping_add(muscle.bone_b as u64)
                .wrapping_add(((muscle.period * 100.0) as u64).wrapping_mul(0x2545f4914f6cdd1d)),
        );
    }
    let stem = SPECIES_STEMS[(plan % SPECIES_STEMS.len() as u64) as usize];
    let form = match bones {
        0..=2 => "ling",
        3..=4 => "pod",
        5..=7 => "form",
        8..=11 => "morph",
        _ => "titan",
    };
    format!("{stem}{form} {}", gait_word(creature))
}
/// Cadence bucket from the muscles' rhythm periods, in cycles per second.
fn gait_word(creature: &Creature) -> &'static str {
    if creature.muscles.is_empty() {
        return "Drifter";
    }
    let mean_period =
        creature.muscles.iter().map(|m| m.period).sum::<f32>() / creature.muscles.len() as f32;
    let hertz = 1.0 / mean_period.max(0.05);
    if hertz < 0.5 {
        "Crawler"
    } else if hertz < 1.0 {
        "Walker"
    } else if hertz < 2.0 {
        "Trotter"
    } else {
        "Sprinter"
    }
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
/// History positions where the all-time best distance moved, oldest first.
fn record_entries(history: &[Stats]) -> Vec<(usize, f32)> {
    let mut best = f32::NEG_INFINITY;
    let mut records = Vec::new();
    for (index, stats) in history.iter().enumerate() {
        if stats.best.is_finite() && stats.best > best {
            best = stats.best;
            records.push((index, stats.best));
        }
    }
    records
}
/// One all-time record in the session hall of fame.
struct FameEntry {
    generation: u32,
    distance: f32,
    creature: Creature,
}
/// Heat map of the occupied archive cells for the selected height and feet
/// bins. Returns the niche key of a clicked cell.
fn paint_archive_map(
    ui: &mut egui::Ui,
    snapshot: &Snapshot,
    cells: &HashMap<[u8; 6], MapCell>,
    scan: Option<&MapScan>,
    height_bin: usize,
    feet_bin: usize,
    theme: Theme,
) -> Option<[u8; 6]> {
    let visible: Vec<(&[u8; 6], &MapCell)> = cells
        .iter()
        .filter(|(niche, _)| niche[3] as usize == height_bin && niche[4] as usize == feet_bin)
        .collect();
    ui.label(
        RichText::new(format!(
            "Ground contact against gait cadence · {} occupied cells in this slice · {} archive elites total",
            visible.len(),
            snapshot.archive_size,
        ))
        .small()
        .color(theme.muted),
    );
    if let Some(scan) = scan {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(
                RichText::new(format!(
                    "Mapping archive pages… {} / {}",
                    scan.next.min(scan.total),
                    scan.total
                ))
                .small()
                .color(theme.accent),
            );
        });
    }
    let (min, max) = visible
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), (_, cell)| {
            (lo.min(cell.score), hi.max(cell.score))
        });
    let range = (max - min).max(1e-6);
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
    }
    for row in 0..=rows {
        let y = plot.bottom() - row as f32 * row_height;
        painter.line_segment(
            [Pos2::new(plot.left(), y), Pos2::new(plot.right(), y)],
            Stroke::new(1., theme.card_border),
        );
    }
    for column in 0..=columns {
        let x = plot.left() + column as f32 * column_width;
        painter.text(
            Pos2::new(x, plot.bottom() + 4.),
            Align2::CENTER_TOP,
            format!("{:.0}%", column as f32 / columns as f32 * 100.),
            FontId::proportional(10.),
            theme.muted,
        );
    }
    for row in 0..=rows {
        let y = plot.bottom() - row as f32 * row_height;
        painter.text(
            Pos2::new(plot.left() - 6., y),
            Align2::RIGHT_CENTER,
            format!("{:.2}", row as f32 / rows as f32 * 6.),
            FontId::proportional(10.),
            theme.muted,
        );
    }
    painter.text(
        Pos2::new(plot.center().x, plot.bottom() + 24.),
        Align2::CENTER_TOP,
        "Ground contact",
        FontId::proportional(11.),
        theme.ink,
    );
    painter.text(
        Pos2::new(rect.left() + 4., plot.top() - 16.),
        Align2::LEFT_BOTTOM,
        "Cadence (Hz)",
        FontId::proportional(11.),
        theme.ink,
    );
    // Legend: cold (slow) to hot (fast), with the distance range.
    if !visible.is_empty() {
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
            FontId::proportional(10.),
            theme.muted,
        );
        painter.text(
            Pos2::new(legend.right() + 6., legend.center().y),
            Align2::LEFT_CENTER,
            format!("{max:.2} m"),
            FontId::proportional(10.),
            theme.muted,
        );
    }
    if visible.is_empty() && scan.is_none() {
        painter.text(
            plot.center(),
            Align2::CENTER_CENTER,
            "No occupied cells in this slice. Pick another height or feet level.",
            FontId::proportional(13.),
            theme.muted,
        );
    }
    let mut clicked = None;
    for (niche, cell) in visible {
        let inner = Rect::from_min_size(
            Pos2::new(
                plot.left() + niche[0] as f32 * column_width + 1.5,
                plot.bottom() - (niche[1] as f32 + 1.) * row_height + 1.5,
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
                FontId::proportional(11.),
                ink,
            );
        }
        let response = ui.interact(inner, ui.id().with(("archive_map", *niche)), Sense::click());
        if response.hovered() {
            painter.rect_stroke(
                inner,
                3,
                Stroke::new(2., theme.ink),
                egui::StrokeKind::Inside,
            );
        }
        if response.clicked() {
            clicked = Some(*niche);
        }
        response.on_hover_text(format!(
            "{} · rank #{} · {:.3} m\n{:.2} m tall · {:.2} aspect ratio · {} feet\n{}\nClick to replay",
            species_name(&cell.creature),
            cell.rank + 1,
            cell.score,
            cell.descriptor.mean_height,
            cell.descriptor.aspect_ratio,
            cell.descriptor.feet.round() as i32,
            cell.emitter
                .map_or("archive elite".to_owned(), |e| e.label().to_owned()),
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
    painter.rect_filled(
        rect,
        8,
        if big || hovered {
            theme.card_hover
        } else {
            theme.card
        },
    );
    let changed = body_plan_changed(step, parent);
    painter.rect_stroke(
        rect,
        8,
        Stroke::new(
            if current { 2. } else { 1. },
            if current || changed {
                theme.accent
            } else {
                theme.card_border
            },
        ),
        egui::StrokeKind::Inside,
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
        FontId::proportional(12.),
        theme.muted,
    );
    painter.text(
        Pos2::new(text_x, rect.top() + 24.),
        Align2::LEFT_TOP,
        format!("{:.2} m", step.fitness),
        FontId::proportional(16.),
        if big { theme.accent } else { theme.ink },
    );
    painter.text(
        Pos2::new(text_x, rect.top() + 46.),
        Align2::LEFT_TOP,
        format!("{:+.2} m", step.gain),
        FontId::proportional(12.),
        if step.gain >= 0. { theme.accent } else { AMBER },
    );
    painter.text(
        rect.right_bottom() + Vec2::new(-8., -6.),
        Align2::RIGHT_BOTTOM,
        species_name(&step.creature),
        FontId::proportional(10.),
        theme.muted,
    );
    if current {
        painter.text(
            rect.right_bottom() + Vec2::new(-8., -20.),
            Align2::RIGHT_BOTTOM,
            "selected",
            FontId::proportional(10.),
            theme.accent,
        );
    }
    if changed {
        painter.text(
            rect.right_top() + Vec2::new(-8., 6.),
            Align2::RIGHT_TOP,
            "BODY PLAN",
            FontId::proportional(9.),
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
    tab: Tab,
    speed: f32,
    playing: bool,
    zoom: f32,
    camera: [f32; 2],
    follow: bool,
    advanced: bool,
    search: String,
    hist_min: f64,
    hist_max: f64,
    bins: u32,
    percentiles: [bool; 29],
    history_index: usize,
    history_latest: bool,
    file_mode: Option<&'static str>,
    file_path: String,
    message: Option<String>,
    new_dialog: bool,
    dirty: bool,
    last_frame: Instant,
    frame_times: std::collections::VecDeque<f32>,
    last_page: usize,
    sort_speed: f32,
    show_perf: bool,
    ui_scale: f32,
    initial: bool,
    smoke_start_pending: bool,
    started: Instant,
    capture_requested: bool,
    capture_path: Option<String>,
    sort_started: Instant,
    card_positions: std::collections::HashMap<u64, Pos2>,
    /// Ancestors of the selected creature, newest first.
    lineage: Vec<crate::worker::LineageStep>,
    /// Creature whose ancestors the UI has already asked the worker for.
    lineage_requested: Option<u64>,
    /// A lineage request is in flight.
    lineage_pending: bool,
    /// Behavior archive map: occupied cells keyed by their niche bytes.
    map_cells: HashMap<[u8; 6], MapCell>,
    map_scan: Option<MapScan>,
    map_generation: u32,
    map_height: usize,
    map_feet: usize,
    archive_view: ArchiveView,
    /// Top archived elites racing side by side.
    race: Vec<RaceLane>,
    race_pending: bool,
    race_page_requested: bool,
    race_camera: f32,
    /// Session hall of fame: all-time records seen so far, oldest first.
    fame: Vec<FameEntry>,
    fame_best: f32,
    fame_seen: usize,
    fame_epoch: u64,
    /// Native benchmark frame intervals and the last control probe time.
    bench_frames: Vec<f32>,
    bench_last_ping: Instant,
    /// Light is the default; the choice lives only in UI state.
    dark: bool,
    show_help: bool,
    runs_bytes: u64,
    runs_checked: Instant,
    screenshot_pending: bool,
    screenshot_waiting: bool,
}
impl App {
    fn new(cc: &eframe::CreationContext<'_>, gpu: Gpu) -> Self {
        let ctx = &cc.egui_ctx;
        apply_style(ctx, false);
        let worker = Worker::spawn(gpu, ctx.clone());
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
        let smoke_start_pending = std::env::var_os("EVOLUTION_SMOKE_POPULATION").is_some();
        let smoke_tab = std::env::var("EVOLUTION_SMOKE_TAB").unwrap_or_default();
        if let Some(path) = std::env::var_os("EVOLUTION_SMOKE_CHECKPOINT") {
            worker.send(Command::Load(PathBuf::from(path)));
        } else {
            worker.send(Command::New(initial_config));
        }
        let mut percentiles = [false; 29];
        percentiles[0] = true;
        percentiles[14] = true;
        percentiles[28] = true;
        Self {
            worker,
            snapshot: None,
            config: Config::default(),
            playback: None,
            tab: match smoke_tab.as_str() {
                "history" => Tab::History,
                "population" | "map" => Tab::Population,
                "race" => Tab::Race,
                "lineage" => Tab::Lineage,
                _ => Tab::Overview,
            },
            speed: 1.0,
            playing: true,
            zoom: DEFAULT_CAMERA_ZOOM,
            camera: [0.0, 0.0],
            follow: true,
            advanced: false,
            search: String::new(),
            hist_min: -1.0,
            hist_max: 8.0,
            bins: 10,
            percentiles,
            history_index: 0,
            history_latest: true,
            file_mode: None,
            file_path: "runs/experiment.evo".into(),
            message: None,
            new_dialog: false,
            dirty: false,
            last_frame: Instant::now(),
            frame_times: Default::default(),
            last_page: usize::MAX,
            sort_speed: 5.0,
            show_perf: false,
            ui_scale: 1.0,
            initial: true,
            smoke_start_pending,
            started: Instant::now(),
            capture_requested: false,
            capture_path: std::env::var("EVOLUTION_SMOKE_CAPTURE").ok(),
            sort_started: Instant::now(),
            lineage: Vec::new(),
            lineage_requested: None,
            lineage_pending: false,
            map_cells: HashMap::new(),
            map_scan: None,
            map_generation: u32::MAX,
            map_height: 0,
            map_feet: 0,
            archive_view: if smoke_tab == "map" {
                ArchiveView::Map
            } else {
                ArchiveView::Cards
            },
            race: Vec::new(),
            race_pending: smoke_tab == "race",
            race_page_requested: false,
            race_camera: 0.0,
            fame: Vec::new(),
            fame_best: 0.0,
            fame_seen: 0,
            fame_epoch: u64::MAX,
            bench_frames: Vec::new(),
            bench_last_ping: Instant::now(),
            card_positions: Default::default(),
            dark: false,
            show_help: false,
            runs_bytes: 0,
            runs_checked: Instant::now() - RUNS_REFRESH,
            screenshot_pending: false,
            screenshot_waiting: false,
        }
    }
    fn theme(&self) -> Theme {
        Theme::of(self.dark)
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
            "Save preset" | "Load preset" => "presets/custom.json".to_owned(),
            "Open creature JSON" => "runs/creature.json".to_owned(),
            "Export creature JSON" => {
                let id = self.playback.as_ref().map_or(0, |p| p.creature.id);
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_millis());
                format!("runs/creature-{id}-{stamp}.json")
            }
            _ => "runs/experiment.evo".to_owned(),
        };
    }
    fn set_preview(&mut self, c: Creature, cfg: Config) {
        self.playback = Some(Playback::new(c, cfg));
        self.follow = true;
        self.zoom = DEFAULT_CAMERA_ZOOM;
        self.camera = [0.; 2];
    }
    fn top(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.add_space(5.);
            let (logo, _) = ui.allocate_exact_size(Vec2::splat(28.), Sense::hover());
            let points = [
                logo.center_top() + Vec2::new(0., 4.),
                logo.left_bottom() + Vec2::new(4., -4.),
                logo.right_bottom() + Vec2::new(-4., -4.),
            ];
            for i in 0..3 {
                ui.painter()
                    .line_segment([points[i], points[(i + 1) % 3]], Stroke::new(2., MINT));
                ui.painter().circle_filled(points[i], 3., MINT);
            }
            ui.label(RichText::new("EVOLUTION").size(22.).strong());
            ui.label(
                RichText::new("CREATURE LABORATORY")
                    .size(10.)
                    .color(theme.muted),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("New experiment").clicked() {
                    self.new_dialog = true;
                }
                if ui
                    .button("Save")
                    .on_hover_text("Save population, settings and progress · Ctrl+S")
                    .clicked()
                {
                    self.file("Save experiment");
                }
                if ui.button("Open").clicked() {
                    self.file("Open experiment");
                }
                if ui
                    .button("Screenshot")
                    .on_hover_text("Save a PNG of the window under runs/")
                    .clicked()
                {
                    self.screenshot_pending = true;
                    self.screenshot_waiting = true;
                    self.message = Some("Taking a screenshot…".into());
                }
                ui.separator();
                if let Some(s) = &self.snapshot {
                    ui.label(
                        RichText::new(format!("GEN {:03}", s.generation))
                            .color(theme.accent)
                            .strong(),
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
        ui.add_space(10.);
        ui.label(RichText::new("EXPERIMENT").small().color(theme.muted));
        ui.heading("Let life find a way.");
        ui.label(RichText::new("More kinds of life. Better walkers.").color(theme.muted));
        ui.add_space(8.);
        let running = self.active();
        let text = if running {
            "Pause evolution"
        } else {
            "Evolve continuously"
        };
        if ui
            .add_sized(
                [ui.available_width(), 40.],
                egui::Button::new(RichText::new(text).strong()).fill(if running {
                    Color32::from_rgb(255, 239, 216)
                } else {
                    Color32::from_rgb(222, 241, 229)
                }),
            )
            .clicked()
        {
            if running {
                self.pause();
            } else {
                self.run(true, false);
            }
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!running, egui::Button::new("One generation"))
                .clicked()
            {
                self.run(false, false);
            }
            if ui
                .add_enabled(!running, egui::Button::new("Guided step"))
                .on_hover_text("Evaluate → update behavior archive → breed from diverse elites")
                .clicked()
            {
                self.worker.pause.store(false, Ordering::Relaxed);
                self.worker.send(Command::Next);
            }
        });
        if let Some(s) = &self.snapshot {
            ui.label(RichText::new(s.stage.label()).color(theme.accent));
            ui.add(
                egui::ProgressBar::new(s.completed as f32 / s.config.population as f32)
                    .text(format!(
                        "{} / {} evaluated",
                        number(s.completed),
                        number(s.config.population)
                    ))
                    .fill(theme.accent.gamma_multiply(0.7)),
            );
        }
        ui.separator();
        let mut before = self.config.clone();
        ui.label(format!(
            "{} creatures · {:.0} s trials",
            number(self.config.population),
            self.config.duration
        ));
        ui.add_space(4.);
        ui.label(RichText::new("Environment").strong());
        ui.label(
            RichText::new(
                "Every change can be undone. Elites are tested again under the new rules.",
            )
            .small()
            .color(theme.muted),
        );
        let mut world_changed = false;
        for effect in &crate::environment::EFFECTS {
            let level = effect.level(&self.config);
            let top = effect.levels.len() - 1;
            ui.horizontal(|ui| {
                ui.label(format!("{}: {}", effect.name, effect.levels[level]))
                    .on_hover_text(effect.why);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(level < top, egui::Button::new(effect.raise).small())
                        .on_hover_text(format!(
                            "{} to: {}. {}",
                            effect.raise,
                            effect.levels[(level + 1).min(top)],
                            effect.why
                        ))
                        .clicked()
                    {
                        effect.set_level(&mut self.config, level + 1);
                        world_changed = true;
                    }
                    if ui
                        .add_enabled(level > 0, egui::Button::new(effect.lower).small())
                        .on_hover_text(format!(
                            "{} to: {}.",
                            effect.lower,
                            effect.levels[level.saturating_sub(1)]
                        ))
                        .clicked()
                    {
                        effect.set_level(&mut self.config, level - 1);
                        world_changed = true;
                    }
                });
            });
        }
        let fossils = self.snapshot.as_ref().map_or(0, |s| s.fossils);
        ui.horizontal(|ui| {
            ui.label("Catastrophe")
                .on_hover_text("A meteor wipes out half of every archive's elites at random. Survivors and newcomers refill the empty cells, which makes room for new kinds of movement.");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(fossils > 0, egui::Button::new("Undo").small())
                    .on_hover_text(format!(
                        "Return {fossils} fossils to their cells where the cell is empty or holds a slower elite."
                    ))
                    .clicked()
                {
                    self.worker.send(Command::UndoMeteor);
                }
                if ui
                    .add(egui::Button::new("Extinction").small())
                    .on_hover_text("Wipe out the island whose best creature is slowest, so it starts over from new designs. Undo brings its elites back.")
                    .clicked()
                {
                    self.worker.send(Command::Extinction);
                }
                if ui
                    .add(egui::Button::new("Meteor strike").small())
                    .on_hover_text("Wipe out half of every archive's elites at random. Undo brings them back.")
                    .clicked()
                {
                    self.worker.send(Command::Meteor);
                }
            });
        });
        if world_changed {
            self.worker.send(Command::Configure(self.config.clone()));
            // Applied already, so it does not count as an unapplied setting.
            before = self.config.clone();
        }
        ui.add_space(4.);
        ui.checkbox(&mut self.advanced, "Advanced controls");
        if self.advanced {
            ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("Find a setting…"));
            let q = self.search.to_lowercase();
            if matches_search(&q, "seed random reproducibility") {
                egui::CollapsingHeader::new("Randomness").show(ui, |ui| {
                    ui.checkbox(
                        &mut self.config.random_seed,
                        "Choose a new seed on creation",
                    );
                    ui.horizontal(|ui| {
                        ui.label("Seed");
                        ui.add(egui::DragValue::new(&mut self.config.seed));
                    });
                    ui.label(
                        RichText::new("The resolved seed is always saved with the experiment.")
                            .small()
                            .color(theme.muted),
                    );
                });
            }
            if matches_search(
                &q,
                "performance gpu ram memory throughput checkpoint autosave",
            ) {
                egui::CollapsingHeader::new("Performance & checkpoints").show(ui, |ui| {
                    ui.checkbox(&mut self.config.throughput, "Maximum throughput")
                        .on_hover_text(
                            "Larger batches for long runs. Selected automatically when you choose 100k or more creatures; uncheck for shorter pauses.",
                        );
                    ui.horizontal(|ui| {
                        ui.label("GPU budget MiB");
                        ui.add(
                            egui::DragValue::new(&mut self.config.gpu_budget_mib)
                                .speed(64)
                                .range(32..=6144),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("RAM budget MiB");
                        ui.add(
                            egui::DragValue::new(&mut self.config.ram_budget_mib)
                                .speed(256)
                                .range(64..=24576),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Autosave every");
                        ui.add(
                            egui::DragValue::new(&mut self.config.checkpoint_interval)
                                .range(0..=1000)
                                .suffix(" gens"),
                        );
                    });
                    ui.small("0 disables automatic checkpoints.");
                });
            }
            if matches_search(&q, "display ui scale window sorting animation") {
                egui::CollapsingHeader::new("Display").show(ui, |ui| {
                    if ui
                        .add(egui::Slider::new(&mut self.ui_scale, 0.75..=1.6).text("UI scale"))
                        .changed()
                    {
                        ui.ctx().set_zoom_factor(self.ui_scale);
                    }
                    ui.add(
                        egui::Slider::new(&mut self.sort_speed, 0.5..=20.0)
                            .text("Sort animation speed"),
                    );
                    ui.checkbox(&mut self.show_perf, "Performance details");
                    if ui.checkbox(&mut self.dark, "Dark theme").changed() {
                        apply_style(ui.ctx(), self.dark);
                    }
                    ui.checkbox(&mut self.show_help, "Show help overlay");
                });
            }
            if matches_search(&q, "histogram minimum maximum bins") {
                egui::CollapsingHeader::new("Histogram").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Min (m)");
                        ui.add(egui::DragValue::new(&mut self.hist_min).speed(0.1));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Max (m)");
                        ui.add(egui::DragValue::new(&mut self.hist_max).speed(0.1));
                    });
                    egui::ComboBox::from_label("Bins / meter")
                        .selected_text(self.bins.to_string())
                        .show_ui(ui, |ui| {
                            for n in [1, 2, 5, 10, 20, 25, 50, 100] {
                                ui.selectable_value(&mut self.bins, n, n.to_string());
                            }
                        });
                });
            }
        }
        if before != self.config {
            self.dirty = true;
        }
        if self.dirty {
            if let Err(e) = self.config.validate() {
                ui.colored_label(AMBER, e.to_string());
            }
            if ui
                .add_enabled(
                    self.config.validate().is_ok(),
                    egui::Button::new("Apply settings"),
                )
                .clicked()
            {
                self.worker.send(Command::Configure(self.config.clone()));
                self.dirty = false;
            }
            ui.label(
                RichText::new(
                    "Changes apply between generations. Seed changes need a new experiment.",
                )
                .small()
                .color(theme.muted),
            );
        }
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Save preset").clicked() {
                self.file("Save preset");
            }
            if ui.small_button("Load preset").clicked() {
                self.file("Load preset");
            }
            if ui
                .add_enabled(
                    self.playback.is_some(),
                    egui::Button::new("Export JSON").small(),
                )
                .on_hover_text("Save the selected creature as JSON under runs/")
                .clicked()
            {
                self.file("Export creature JSON");
            }
            if ui
                .small_button("Open creature")
                .on_hover_text("Replay a creature from a JSON file")
                .clicked()
            {
                self.file("Open creature JSON");
            }
            if ui.small_button("Reset settings").clicked() {
                self.config = Config::default();
                self.dirty = true;
            }
        });
        ui.separator();
        ui.label(RichText::new("Each creature runs its own trial. Faster walkers are more likely to survive; their offspring explore new shapes.").small().color(theme.muted));
    }
    fn viewport(&mut self, ui: &mut egui::Ui, height: f32) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.label(RichText::new("LIVE CREATURE").small().color(theme.muted));
            if let Some(p) = &self.playback {
                ui.label(format!(
                    "#{} · {} nodes / {} bones / {} muscles · replay distance {:.1} m · {:.2} m/s",
                    p.creature.id,
                    p.nodes.len(),
                    p.creature.bones.len(),
                    p.creature.muscles.len(),
                    p.distance,
                    p.speed()
                ))
                .on_hover_text(
                    "The distance this replay reaches, and its mass-weighted center-of-mass speed over the last fifth of a second of recorded frames. An archive score is the worse of this trial and a check from a slightly shifted pose at four times the physics rate, so it is never higher.",
                );
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.follow, "Follow");
                if ui.small_button("Reset camera").clicked() {
                    self.zoom = DEFAULT_CAMERA_ZOOM;
                    self.camera = [0.; 2];
                    self.follow = true;
                }
            });
        });
        let (rect, response) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), height.max(120.)),
            Sense::drag(),
        );
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(30., 1200.);
        }
        if response.dragged() {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera[0] -= delta.x / self.zoom;
            self.camera[1] += delta.y / self.zoom;
            self.follow = false;
        }
        let painter = ui.painter_at(rect);
        // All scene primitives are tessellated into egui's batched wgpu render pass.
        painter.rect_filled(rect, 12, VIEWPORT);
        draw_clouds(&painter, rect, self.camera[0] * self.zoom);
        let origin = Pos2::new(
            rect.center().x - self.camera[0] * self.zoom,
            rect.bottom() - rect.height() * 0.22 + self.camera[1] * self.zoom,
        );
        let world = |x: f32, y: f32| Pos2::new(origin.x + x * self.zoom, origin.y - y * self.zoom);
        let cfg = self
            .playback
            .as_ref()
            .map(|p| &p.config)
            .unwrap_or(&self.config);
        let left = ((rect.left() - origin.x) / self.zoom).floor() as i32;
        let right = ((rect.right() - origin.x) / self.zoom).ceil() as i32;
        for x in left..=right {
            let pos = world(x as f32, 0.);
            painter.line_segment(
                [
                    Pos2::new(pos.x, rect.top()),
                    Pos2::new(pos.x, rect.bottom()),
                ],
                Stroke::new(1., CARD_BORDER),
            );
            painter.text(
                Pos2::new(pos.x + 5., origin.y + 16.),
                Align2::LEFT_TOP,
                format!("{x} m"),
                FontId::proportional(11.),
                MUTED,
            );
        }
        let amplitude = crate::physics::terrain_amplitude(cfg.terrain);
        let slope = if cfg.ground { cfg.slope } else { 0.0 };
        if cfg.ground && (amplitude > 0.0 || slope != 0.0) {
            // Sample the tilted ground every few pixels and fill down to the frame.
            let step = (4.0 / self.zoom).max(0.002);
            let start = (rect.left() - origin.x) / self.zoom;
            let end = (rect.right() - origin.x) / self.zoom;
            let mut x = start;
            let mut line = Vec::new();
            while x <= end + step {
                line.push(world(
                    x,
                    crate::physics::terrain_with_slope(x, amplitude, slope).0,
                ));
                x += step;
            }
            for pair in line.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        a,
                        b,
                        Pos2::new(b.x, rect.bottom()),
                        Pos2::new(a.x, rect.bottom()),
                    ],
                    GROUND,
                    Stroke::NONE,
                ));
            }
            painter.add(egui::Shape::line(line, Stroke::new(2., GROUND_EDGE)));
        } else if cfg.ground {
            painter.rect_filled(
                Rect::from_min_max(
                    Pos2::new(rect.left(), origin.y.clamp(rect.top(), rect.bottom())),
                    rect.right_bottom(),
                ),
                0,
                GROUND,
            );
            painter.line_segment(
                [
                    Pos2::new(rect.left(), origin.y),
                    Pos2::new(rect.right(), origin.y),
                ],
                Stroke::new(2., GROUND_EDGE),
            );
        }
        for x in left..=right {
            let pos = world(x as f32, 0.);
            painter.line_segment([pos, pos + Vec2::new(0., 6.)], Stroke::new(1., MUTED));
            painter.text(
                pos + Vec2::new(5., 12.),
                Align2::LEFT_TOP,
                format!("{x} m"),
                FontId::proportional(11.),
                MUTED,
            );
        }
        if let Some(p) = &self.playback {
            // Center-of-mass trail from the last two seconds of recorded
            // frames, fading with age.
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
                            2.5,
                            Color32::from_rgba_unmultiplied(INK.r(), INK.g(), INK.b(), alpha),
                        ),
                    );
                }
                previous = Some(point);
            }
            if let Some(com) = p.center_of_mass(p.tick) {
                painter.circle_filled(
                    world(com[0], com[1]),
                    3.5,
                    Color32::from_rgba_unmultiplied(INK.r(), INK.g(), INK.b(), 170),
                );
            }
            for n in &p.nodes {
                let shadow = world(n.pos[0], 0.);
                painter.add(egui::Shape::ellipse_filled(
                    shadow,
                    Vec2::new(n.radius * self.zoom * 1.6, 4.),
                    Color32::from_black_alpha(30),
                ));
            }
            draw_creature(
                &painter,
                &p.nodes,
                &p.creature,
                origin,
                self.zoom,
                p.tick.saturating_sub(physics::settle()) as f32 * physics::dt(),
                p.fallen().is_some(),
            );
            match p.fallen() {
                Some((tick, distance)) => {
                    painter.text(
                        rect.left_top() + Vec2::new(18., 16.),
                        Align2::LEFT_TOP,
                        format!("{distance:.2} m"),
                        FontId::proportional(24.),
                        FALLEN,
                    );
                    painter.text(
                        rect.left_top() + Vec2::new(18., 46.),
                        Align2::LEFT_TOP,
                        format!(
                            "Fell over at {:.1} s: head below its neck",
                            tick.saturating_sub(physics::settle()) as f32 * physics::dt()
                        ),
                        FontId::proportional(13.),
                        FALLEN,
                    );
                }
                None => {
                    painter.text(
                        rect.left_top() + Vec2::new(18., 16.),
                        Align2::LEFT_TOP,
                        format!("{:.2} m", physics::fitness(&p.nodes)),
                        FontId::proportional(24.),
                        MINT,
                    );
                }
            }
        } else {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Preparing your first population…",
                FontId::proportional(20.),
                MUTED,
            );
        }
        painter.text(
            rect.left_bottom() + Vec2::new(14., -12.),
            Align2::LEFT_BOTTOM,
            "Drag to pan · scroll to zoom",
            FontId::proportional(11.),
            MUTED,
        );
        let mut sought = false;
        if let Some(p) = &mut self.playback {
            let last_frame = p.last_frame();
            let trial_start = p.trial_start();
            let trial_frames = last_frame.saturating_sub(trial_start);
            ui.horizontal(|ui| {
                ui.label("Time");
                if trial_frames > 0 {
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
        ui.horizontal(|ui| {
            if ui
                .button(if self.playing { "Pause" } else { "Play" })
                .on_hover_text("Pause / play creature")
                .clicked()
            {
                self.playing = !self.playing;
            }
            if ui.button("Replay").clicked()
                && let Some(p) = &mut self.playback
            {
                p.reset();
            }
            if ui.button("Single tick").clicked() {
                self.playing = false;
                if let Some(p) = &mut self.playback {
                    p.advance();
                }
            }
            ui.add(
                egui::Slider::new(&mut self.speed, 0.25..=4.0)
                    .logarithmic(true)
                    .suffix("×")
                    .text("Playback speed"),
            )
            .on_hover_text("Playback speed, from quarter speed to four times speed.");
        });
        if sought {
            self.playing = false;
        }
    }
    fn metrics(&self, ui: &mut egui::Ui) {
        let theme = self.theme();
        // Live creatures/s over complete generations; the last generation's own
        // figure until enough generations have finished.
        let live_rate = self.snapshot.as_ref().map_or(0.0, |s| s.end_to_end);
        if let Some(s) = self.snapshot.as_ref().and_then(|s| s.history.last()) {
            ui.columns(4, |cols| {
                for (ui, (name, value, color)) in cols.iter_mut().zip([
                    ("BEST GAIT SCORE", format!("{:.3} m", s.best), theme.accent),
                    ("QD SCORE", format!("{:.2}", s.qd_score), theme.accent),
                    ("NICHES", number(s.archive_cells), theme.ink),
                    (
                        "EVALUATIONS / SEC",
                        format!(
                            "{:.0}",
                            if live_rate > 0.0 {
                                live_rate
                            } else {
                                s.population as f64 / s.seconds.max(0.001)
                            }
                        ),
                        theme.ink,
                    ),
                ]) {
                    egui::Frame::new()
                        .fill(theme.card)
                        .corner_radius(8)
                        .inner_margin(12)
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.label(RichText::new(name).small().color(theme.muted));
                            ui.label(RichText::new(value).size(22.).color(color));
                        });
                }
            });
        } else {
            ui.label(
                RichText::new("Run a generation to see how far your creatures can travel.")
                    .color(theme.muted),
            );
        }
    }
    fn trend(&self, ui: &mut egui::Ui, height: f32) {
        let Some(s) = &self.snapshot else { return };
        let theme = self.theme();
        Plot::new("fitness_history")
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Generation")
            .y_axis_label("Gait score (m)")
            .allow_scroll(false)
            .show(ui, |plot| {
                for (i, &visible) in self.percentiles.iter().enumerate() {
                    if !visible {
                        continue;
                    }
                    let values: Vec<[f64; 2]> = s
                        .history
                        .iter()
                        .map(|h| [h.generation as f64, h.percentiles[i] as f64])
                        .collect();
                    let (name, color, width) = if i == 28 {
                        ("Best".into(), theme.accent, 2.5)
                    } else if i == 14 {
                        ("Median".into(), AMBER, 2.5)
                    } else if i == 0 {
                        ("Worst".into(), Color32::from_rgb(104, 133, 159), 1.5)
                    } else {
                        (format!("P{}", PERCENTILES[i]), species_color(i, 0), 1.)
                    };
                    plot.line(Line::new(name, values).color(color).width(width));
                }
                // Record markers extend the best line instead of duplicating it.
                let records: Vec<[f64; 2]> = record_entries(&s.history)
                    .into_iter()
                    .map(|(index, best)| [s.history[index].generation as f64, best as f64])
                    .collect();
                if !records.is_empty() {
                    plot.points(
                        Points::new("Record", records)
                            .color(Color32::from_rgb(117, 76, 210))
                            .filled(true)
                            .radius(3.5),
                    );
                }
            });
    }
    fn histogram(&self, ui: &mut egui::Ui, stats: &Stats, height: f32) {
        if !self.hist_min.is_finite()
            || !self.hist_max.is_finite()
            || self.hist_max <= self.hist_min
        {
            ui.colored_label(AMBER, "Histogram minimum must be below maximum.");
            return;
        }
        let count = ((self.hist_max - self.hist_min) * self.bins as f64)
            .ceil()
            .min(4096.) as usize;
        let mut bins = vec![0u32; count];
        let mut outside = stats.failed as u64;
        for &(cm, n) in &stats.histogram {
            let value = (cm as f64 + 0.5) / 100.;
            let index = ((value - self.hist_min) * self.bins as f64).floor();
            if index >= 0. && (index as usize) < count {
                bins[index as usize] += n;
            } else {
                outside += n as u64;
            }
        }
        let bars = bins
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                Bar::new(
                    self.hist_min + (i as f64 + 0.5) / self.bins as f64,
                    n as f64,
                )
                .width(0.85 / self.bins as f64)
            })
            .collect();
        Plot::new("histogram")
            .height(height)
            .x_axis_label("Gait score (m)")
            .allow_scroll(false)
            .show(ui, |plot| {
                plot.bar_chart(
                    BarChart::new("Creatures", bars)
                        .color(self.theme().accent.gamma_multiply(0.65)),
                );
            });
        if outside > 0 {
            ui.small(format!(
                "{outside} outside this range or failed · change range in Advanced"
            ));
        }
    }
    fn population(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.heading("Search archive");
            ui.label(
                RichText::new("Behavior niches and protected topologies · click to replay")
                    .color(theme.muted),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .selectable_value(&mut self.archive_view, ArchiveView::Map, "Map")
                    .on_hover_text(
                        "Watch the search fill behavior space. Cells are colored by gait score.",
                    )
                    .clicked()
                {
                    self.last_page = usize::MAX;
                }
                if ui
                    .selectable_value(&mut self.archive_view, ArchiveView::Cards, "Cards")
                    .clicked()
                {
                    self.map_scan = None;
                    self.last_page = usize::MAX;
                }
            });
        });
        if self.archive_view == ArchiveView::Map {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Body height").small().color(theme.muted));
                egui::ComboBox::from_id_salt("map_height")
                    .selected_text(height_bin_label(self.map_height))
                    .show_ui(ui, |ui| {
                        for bin in 0..MAP_BINS[3] {
                            ui.selectable_value(
                                &mut self.map_height,
                                bin,
                                height_bin_label(bin),
                            );
                        }
                    });
                ui.label(RichText::new("Feet").small().color(theme.muted));
                egui::ComboBox::from_id_salt("map_feet")
                    .selected_text(feet_bin_label(self.map_feet))
                    .show_ui(ui, |ui| {
                        for bin in 0..MAP_BINS[4] {
                            ui.selectable_value(&mut self.map_feet, bin, feet_bin_label(bin));
                        }
                    });
                ui.label(
                    RichText::new(
                        "Bounce adds no cell: its axis has one bin. Click an occupied cell to replay it.",
                    )
                    .small()
                    .color(theme.muted),
                );
            });
        }
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        ui.horizontal(|ui| {
            ui.label(format!("{} niches", number(snapshot.archive_cells)));
            ui.label(format!(
                "{} topology reserves",
                number(snapshot.innovation_reserve_count)
            ));
            ui.label(
                RichText::new(format!("QD score {:.2}", snapshot.qd_score)).color(theme.accent),
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Archive axes:").small().color(theme.muted));
            for (name, why) in [
                (
                    "Ground contact",
                    "How much of the timed trial the creature kept its nodes on the ground.",
                ),
                (
                    "Gait cadence",
                    "How many up-and-down body oscillations the gait completes per second.",
                ),
                (
                    "Mean body height",
                    "The average height of the body's bounding box above the ground during the trial.",
                ),
                (
                    "Feet",
                    "Nodes that touched the ground and lifted off again; a node dragged along the ground never lifts, so it is not a foot.",
                ),
            ] {
                ui.label(RichText::new(name).small().color(theme.ink))
                    .on_hover_text(why);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Next batch:").small().color(theme.muted));
            for (emitter, weight) in crate::qd::Emitter::ALL
                .into_iter()
                .zip(snapshot.emitter_weights)
            {
                ui.label(
                    RichText::new(format!("{} {:.0}%", emitter.label(), weight * 100.0)).small(),
                )
                .on_hover_text(
                    "Emitter shares adapt to recent archive discoveries and improvements.",
                );
            }
        });
        let mut selected = None;
        let mut requested = None;
        let mut map_scan_request = None;
        if self.archive_view == ArchiveView::Map {
            if self.map_scan.is_none()
                && snapshot.archive_size > 0
                && (self.map_cells.is_empty() || self.map_generation != snapshot.generation)
            {
                map_scan_request = Some((snapshot.archive_size, snapshot.generation));
            }
            let clicked = paint_archive_map(
                ui,
                snapshot,
                &self.map_cells,
                self.map_scan.as_ref(),
                self.map_height,
                self.map_feet,
                theme,
            );
            if let Some(cell) = clicked.and_then(|key| self.map_cells.get(&key)) {
                selected = Some((cell.creature.clone(), snapshot.config.clone()));
            }
        } else {
            let columns = (ui.available_width() / 155.).floor().max(2.) as usize;
            let width = (ui.available_width() - (columns - 1) as f32 * 10.) / columns as f32;
            let progress =
                (self.sort_started.elapsed().as_secs_f32() * self.sort_speed / 3.).min(1.);
            let animating = snapshot.stage == Stage::Archived && progress < 1.;
            let ease = progress * progress * (3. - 2. * progress);
            let item_count = if snapshot.archive_size > 0 {
                snapshot.archive_size
            } else {
                snapshot.config.population
            };
            let mut positions = std::collections::HashMap::new();
            egui::ScrollArea::vertical().id_salt("population_grid").show_rows(
                ui, 137., item_count.div_ceil(columns), |ui, rows| {
                    let start = rows.start * columns;
                    if start != self.last_page { requested = Some(start); }
                    for row in rows {
                        ui.horizontal(|ui| {
                            for column in 0..columns {
                                let rank = row * columns + column;
                                if rank >= item_count { break; }
                                let (destination, response) = ui.allocate_exact_size(Vec2::new(width, 127.), Sense::click());
                                if let Some(card) = snapshot.page.iter().find(|c| c.rank == rank) {
                                    positions.insert(card.creature.id, destination.min);
                                    let mut rect = destination;
                                    if animating && let Some(previous) = self.card_positions.get(&card.creature.id) {
                                        rect = destination.translate((*previous - destination.min) * (1. - ease));
                                    }
                                    paint_card(ui.painter(), card, rect, response.hovered(), snapshot.stage, theme);
                                    if response.clicked() {
                                        selected = Some((card.creature.clone(), snapshot.config.clone()));
                                    }
                                    response.on_hover_text(format!(
                                        "{}\nID {}\n{} nodes / {} bones / {} muscles\nMutability {:.2}\n{}\n{}\nClick to replay",
                                        species_name(&card.creature),
                                        card.creature.id,
                                        card.creature.nodes.len(),
                                        card.creature.bones.len(),
                                        card.creature.muscles.len(),
                                        card.creature.mutability,
                                        card.emitter.map_or("Initial population".to_owned(), |emitter| format!("Emitter: {}", emitter.label())),
                                        card.descriptor.map_or_else(
                                            || if card.score.is_finite() { "Current trial evaluated".to_owned() } else { "Current trial pending".to_owned() },
                                            |d| format!("Contact {:.0}% · gait {:.2} Hz · form {:.2} · bob {:.2} m · height {:.2} m · {} feet · {} visits", d.ground_contact * 100.0, d.gait_frequency, d.aspect_ratio, d.vertical_oscillation, d.mean_height, d.feet, card.visits),
                                        )
                                    ));
                                } else {
                                    ui.painter().rect_filled(destination, 8, theme.card);
                                    ui.painter().text(destination.center(), Align2::CENTER_CENTER, "Loading…", FontId::proportional(12.), theme.muted);
                                }
                            }
                        });
                    }
                }
            );
            if animating {
                ui.ctx().request_repaint();
            } else {
                self.card_positions = positions;
            }
        }
        if let Some((total, generation)) = map_scan_request {
            self.begin_map_scan(total, generation);
        }
        if let Some(start) = requested {
            self.worker.send(Command::Page(start));
            self.last_page = start;
        }
        if let Some((creature, config)) = selected {
            self.worker.send(Command::Preview { creature, config });
            self.tab = Tab::Overview;
        }
    }
    /// Starts a sweep through the archive pages that fills the behavior map.
    fn begin_map_scan(&mut self, total: usize, generation: u32) {
        if total == 0 {
            self.map_cells.clear();
            self.map_scan = None;
            return;
        }
        self.map_cells.clear();
        self.map_generation = generation;
        self.map_scan = Some(MapScan { next: 0, total });
        self.last_page = usize::MAX;
        self.worker.send(Command::Page(0));
    }
    /// Folds one published archive page into the behavior map while a sweep runs.
    fn absorb_archive_page(&mut self, snapshot: &Snapshot) {
        let Some(scan) = self.map_scan else { return };
        if snapshot.archive_size != scan.total {
            // The archive moved under the sweep, so start over.
            let total = snapshot.archive_size;
            self.map_cells.clear();
            self.map_generation = snapshot.generation;
            self.map_scan = (total > 0).then_some(MapScan { next: 0, total });
            if total > 0 {
                self.worker.send(Command::Page(0));
            }
            return;
        }
        if snapshot.page_start != scan.next || snapshot.page.is_empty() {
            self.worker.send(Command::Page(scan.next));
            return;
        }
        for card in &snapshot.page {
            let Some(descriptor) = card.descriptor else {
                continue;
            };
            if !card.score.is_finite() {
                continue;
            }
            let key = descriptor.niche().0;
            if self
                .map_cells
                .get(&key)
                .is_none_or(|old| card.score > old.score)
            {
                self.map_cells.insert(
                    key,
                    MapCell {
                        score: card.score,
                        rank: card.rank,
                        descriptor,
                        emitter: card.emitter,
                        creature: card.creature.clone(),
                    },
                );
            }
        }
        self.map_generation = snapshot.generation;
        let next = snapshot.page_start + snapshot.page.len();
        if next >= scan.total {
            self.map_scan = None;
        } else {
            self.map_scan = Some(MapScan {
                next,
                total: scan.total,
            });
            self.worker.send(Command::Page(next));
        }
    }
    /// Appends new all-time bests to the session hall of fame. Records come
    /// from history stats, whose best representative is already in the
    /// snapshot, so no archive page request is needed.
    fn absorb_records(&mut self, snapshot: &Snapshot) {
        if self.fame_epoch != snapshot.epoch {
            self.fame_epoch = snapshot.epoch;
            self.fame.clear();
            self.fame_best = 0.0;
            self.fame_seen = 0;
        }
        if snapshot.history.len() < self.fame_seen {
            self.fame_seen = 0;
        }
        for stats in &snapshot.history[self.fame_seen..] {
            if stats.best.is_finite() && stats.best > self.fame_best {
                self.fame_best = stats.best;
                if let Some(creature) = stats.representatives.last().cloned() {
                    self.fame.push(FameEntry {
                        generation: stats.generation,
                        distance: stats.best,
                        creature,
                    });
                }
            }
        }
        self.fame_seen = snapshot.history.len();
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
        self.worker.send(Command::Preview { creature, config });
        self.tab = Tab::Overview;
    }
    /// Compact timeline of every new all-time best, newest first. Clicking one
    /// replays its record holder.
    fn records_timeline(&mut self, ui: &mut egui::Ui) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let records = record_entries(&snapshot.history);
        if records.is_empty() {
            return;
        }
        let theme = self.theme();
        ui.label(
            RichText::new("RECORDS · EVERY NEW BEST DISTANCE")
                .small()
                .color(theme.muted),
        );
        let mut chosen = None;
        egui::ScrollArea::horizontal()
            .id_salt("records_timeline")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for &(index, best) in records.iter().rev() {
                        let stats = &snapshot.history[index];
                        if ui
                            .small_button(format!("Gen {} · {best:.2} m", stats.generation))
                            .on_hover_text("Click to replay this record holder")
                            .clicked()
                        {
                            chosen = Some(index);
                        }
                    }
                });
            });
        if let Some(index) = chosen {
            self.replay_history_holder(index);
        }
    }
    /// Session record holders, newest first, with a replay button each.
    fn hall_of_fame(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.label(
            RichText::new("HALL OF FAME · SESSION RECORDS")
                .small()
                .color(theme.muted),
        );
        if self.fame.is_empty() {
            ui.label(
                RichText::new("No records yet. The first improvement lands here.")
                    .small()
                    .color(theme.muted),
            );
            return;
        }
        let mut chosen = None;
        egui::ScrollArea::vertical()
            .id_salt("hall_of_fame")
            .max_height(180.)
            .show(ui, |ui| {
                for (place, entry) in self.fame.iter().enumerate().rev() {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!("{}.", place + 1))
                                .small()
                                .color(theme.muted),
                        );
                        ui.label(format!(
                            "Gen {} · {:.2} m",
                            entry.generation, entry.distance
                        ));
                        ui.label(
                            RichText::new(species_name(&entry.creature))
                                .small()
                                .color(theme.muted),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_button("Replay")
                                .on_hover_text("Replay this record holder")
                                .clicked()
                            {
                                chosen = Some(place);
                            }
                        });
                    });
                }
            });
        if let Some(place) = chosen {
            let entry = &self.fame[place];
            let creature = entry.creature.clone();
            let config = self
                .snapshot
                .as_ref()
                .map_or_else(Config::default, |s| s.config.clone());
            self.worker.send(Command::Preview { creature, config });
            self.tab = Tab::Overview;
        }
    }
    /// Builds race lanes from the top archive cards once the first page arrives.
    fn maybe_build_race(&mut self) {
        if !self.race_pending {
            return;
        }
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let (archive_size, page_start) = (snapshot.archive_size, snapshot.page_start);
        if archive_size == 0 {
            return;
        }
        if page_start != 0 {
            if !self.race_page_requested {
                self.race_page_requested = true;
                self.worker.send(Command::Page(0));
            }
            return;
        }
        let config = snapshot.config.clone();
        let lanes: Vec<RaceLane> = snapshot
            .page
            .iter()
            .filter(|card| card.descriptor.is_some() && card.score.is_finite())
            .take(5)
            .map(|card| RaceLane {
                rank: card.rank,
                id: card.creature.id,
                score: card.score,
                playback: Playback::new(card.creature.clone(), config.clone()),
            })
            .collect();
        if lanes.is_empty() {
            return;
        }
        self.race = lanes;
        self.race_pending = false;
        self.race_page_requested = false;
        self.race_camera = 0.0;
    }
    /// Clears the race and asks for a fresh set of top elites.
    fn restart_race(&mut self) {
        self.race.clear();
        self.race_pending = true;
        self.race_page_requested = false;
        self.race_camera = 0.0;
    }
    /// Ancestor chain of the selected creature; the biggest gains stand out and
    /// any ancestor can be replayed.
    fn lineage_strip(&mut self, ui: &mut egui::Ui) {
        if self.lineage.len() < 2 {
            return;
        }
        let theme = self.theme();
        let mut gains: Vec<f32> = self.lineage.iter().map(|step| step.gain).collect();
        gains.sort_by(|a, b| b.total_cmp(a));
        let highlight = gains.get(2).copied().unwrap_or(f32::INFINITY).max(0.01);
        ui.label(
            RichText::new(format!(
                "Lineage · {} ancestors, newest first · click one to replay it",
                self.lineage.len()
            ))
            .small()
            .color(theme.muted),
        );
        let mut chosen = None;
        egui::ScrollArea::horizontal()
            .id_salt("lineage")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for k in 0..self.lineage.len() {
                        let step = &self.lineage[k];
                        if paint_lineage_tile(
                            ui,
                            step,
                            self.lineage.get(k + 1),
                            step.gain >= highlight,
                            k == 0,
                            theme,
                            Vec2::new(176., 88.),
                        ) {
                            chosen = Some(k);
                        }
                    }
                });
            });
        if let Some(k) = chosen {
            let config = self
                .snapshot
                .as_ref()
                .map_or_else(Config::default, |s| s.config.clone());
            self.set_preview(self.lineage[k].creature.clone(), config);
        }
        ui.add_space(6.);
    }
    /// Full ancestor list of the selected creature, one row per generation.
    fn lineage_view(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.heading("Lineage");
            ui.label(
                RichText::new(
                    "Ancestors of the selected creature, newest first. Click one to replay it.",
                )
                .color(theme.muted),
            );
        });
        if self.lineage.is_empty() {
            ui.add_space(12.);
            ui.label(
                RichText::new(if self.playback.is_none() {
                    "Select a creature in the Behavior archive to trace its ancestry."
                } else if self.lineage_pending {
                    "Requesting ancestors…"
                } else {
                    "No recorded ancestors for this creature yet. Evolve a few generations or pick an archive elite."
                })
                .color(theme.muted),
            );
            return;
        }
        if let Some(p) = &self.playback {
            ui.label(format!(
                "#{} · {} nodes / {} bones / {} muscles · {} recorded ancestors",
                p.creature.id,
                p.creature.nodes.len(),
                p.creature.bones.len(),
                p.creature.muscles.len(),
                self.lineage.len()
            ));
        }
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
                        k == 0,
                        theme,
                        Vec2::new(ui.available_width(), 104.),
                    ) {
                        chosen = Some(k);
                    }
                    ui.add_space(6.);
                }
            });
        if let Some(k) = chosen {
            let config = self
                .snapshot
                .as_ref()
                .map_or_else(Config::default, |s| s.config.clone());
            self.set_preview(self.lineage[k].creature.clone(), config);
            self.tab = Tab::Overview;
        }
    }
    /// The top archived elites running side by side, with live standings.
    fn race_view(&mut self, ui: &mut egui::Ui) {
        let theme = self.theme();
        ui.horizontal(|ui| {
            ui.heading("Race");
            ui.label(
                RichText::new("The fastest archived creatures run their trials side by side.")
                    .color(theme.muted),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("New race")
                    .on_hover_text("Take the current top five archived creatures")
                    .clicked()
                {
                    self.restart_race();
                }
            });
        });
        ui.horizontal(|ui| {
            if ui
                .button(if self.playing { "Pause" } else { "Play" })
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
            ui.add(
                egui::Slider::new(&mut self.speed, 0.25..=4.0)
                    .logarithmic(true)
                    .suffix("×")
                    .text("Playback speed"),
            )
            .on_hover_text("Shared with the single-creature replay.");
        });
        if self.race.is_empty() {
            ui.add_space(12.);
            let waiting = self.race_pending
                && self
                    .snapshot
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.archive_size > 0);
            ui.label(
                RichText::new(if waiting {
                    "Loading the top archive elites…"
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
        let board_width = 200.0_f32.min(rect.width() * 0.3);
        let lanes_rect = Rect::from_min_max(
            rect.min,
            Pos2::new(rect.right() - board_width - 12., rect.bottom()),
        );
        let zoom = 34.0;
        let visible = lanes_rect.width() / zoom;
        let target = (distances[leader] - visible * 0.6).max(0.0);
        let dt = ui.ctx().input(|i| i.stable_dt).clamp(0.0, 0.1);
        self.race_camera += (target - self.race_camera) * (dt * 4.0).min(1.0);
        let camera = self.race_camera;
        let lane_height = lanes_rect.height() / self.race.len() as f32;
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
            painter.rect_filled(
                lane_rect,
                8,
                if is_leader {
                    theme.card_hover
                } else {
                    theme.card
                },
            );
            painter.rect_stroke(
                lane_rect,
                8,
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
            let ground = lane_rect.bottom() - 12.;
            painter.line_segment(
                [
                    Pos2::new(lane_rect.left() + 4., ground),
                    Pos2::new(lane_rect.right() - 4., ground),
                ],
                Stroke::new(2., GROUND_EDGE),
            );
            let step = 5.0;
            let mut x = (camera / step).ceil() * step;
            while x <= camera + visible {
                let px = lane_rect.left() + (x - camera) * zoom;
                painter.line_segment(
                    [
                        Pos2::new(px, ground),
                        Pos2::new(px, lane_rect.bottom() - 5.),
                    ],
                    Stroke::new(1., theme.card_border),
                );
                if i == 0 {
                    painter.text(
                        Pos2::new(px + 3., ground - 2.),
                        Align2::LEFT_BOTTOM,
                        format!("{x:.0} m"),
                        FontId::proportional(10.),
                        theme.muted,
                    );
                }
                x += step;
            }
            let origin = Pos2::new(lane_rect.left() - camera * zoom, ground);
            let playback = &lane.playback;
            draw_creature(
                &painter,
                &playback.nodes,
                &playback.creature,
                origin,
                zoom,
                playback.tick.saturating_sub(physics::settle()) as f32 * physics::dt(),
                playback.fallen().is_some(),
            );
            painter.text(
                lane_rect.left_top() + Vec2::new(8., 6.),
                Align2::LEFT_TOP,
                format!("#{}", lane.rank + 1),
                FontId::proportional(13.),
                if is_leader { theme.accent } else { theme.ink },
            );
            painter.text(
                lane_rect.left_top() + Vec2::new(8., 23.),
                Align2::LEFT_TOP,
                format!(
                    "{} · ID {} · best {:.2} m",
                    species_name(&lane.playback.creature),
                    lane.id,
                    lane.score
                ),
                FontId::proportional(11.),
                theme.muted,
            );
            painter.text(
                lane_rect.right_top() + Vec2::new(-8., 6.),
                Align2::RIGHT_TOP,
                format!("{:.2} m", distances[i]),
                FontId::proportional(15.),
                if is_leader { theme.accent } else { theme.ink },
            );
            if playback.fallen().is_some() {
                painter.text(
                    lane_rect.right_top() + Vec2::new(-8., 26.),
                    Align2::RIGHT_TOP,
                    "fell",
                    FontId::proportional(11.),
                    FALLEN,
                );
            }
        }
        let board = Rect::from_min_max(
            Pos2::new(lanes_rect.right() + 12., rect.top()),
            rect.right_bottom(),
        );
        painter.rect_filled(board, 8, theme.card);
        painter.rect_stroke(
            board,
            8,
            Stroke::new(1., theme.card_border),
            egui::StrokeKind::Inside,
        );
        painter.text(
            board.left_top() + Vec2::new(10., 8.),
            Align2::LEFT_TOP,
            "STANDINGS",
            FontId::proportional(11.),
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
                format!("{}. #{}", place + 1, lane.rank + 1),
                FontId::proportional(12.),
                if place == 0 { theme.accent } else { theme.ink },
            );
            painter.text(
                Pos2::new(board.right() - 10., y),
                Align2::RIGHT_CENTER,
                format!("{:.1} m", distances[i]),
                FontId::proportional(12.),
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
            "Live distance · archive rank",
            FontId::proportional(10.),
            theme.muted,
        );
    }
    fn species_history(&mut self, ui: &mut egui::Ui) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        if snapshot.history.is_empty() {
            return;
        }
        let theme = self.theme();
        ui.label(
            RichText::new("BODY TYPES THROUGH GENERATIONS")
                .small()
                .color(theme.muted),
        );
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
            ui.label("Run your first generation to build fitness curves and creature replays.");
            return;
        }
        ui.horizontal(|ui| {
            ui.heading("Generation archive");
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
        egui::CollapsingHeader::new("Percentile curves").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (i, p) in PERCENTILES.iter().enumerate() {
                    ui.checkbox(&mut self.percentiles[i], format!("P{p}"));
                }
            });
        });
        self.trend(ui, 180.);
        self.records_timeline(ui);
        ui.add_space(6.);
        self.hall_of_fame(ui);
        ui.add_space(6.);
        self.species_history(ui);
        let stats = self.snapshot.as_ref().unwrap().history[self.history_index].clone();
        let body_count = if stats.archive_cells > 0 {
            stats.archive_cells
        } else {
            stats.population
        };
        ui.horizontal(|ui| {
            ui.label(format!(
                "Generation {} · seed {} · {} evaluated · {} niches · QD {:.2} · {} failed",
                stats.generation,
                stats.config.seed,
                number(stats.population),
                number(stats.archive_cells),
                stats.qd_score,
                stats.failed
            ));
        });
        ui.columns(2, |cols| {
            self.histogram(&mut cols[0], &stats, 155.);
            cols[1].label(RichText::new("BODY TYPES").small().color(theme.muted));
            let mut species = stats.species.clone();
            species.sort_by_key(|&(_, _, n)| std::cmp::Reverse(n));
            egui::ScrollArea::vertical()
                .max_height(180.)
                .show(&mut cols[1], |ui| {
                    for &(n, m, count) in &species {
                        ui.horizontal(|ui| {
                            color_dot(ui, species_color(n, m));
                            ui.label(format!(
                                "{n} nodes / {} bones / {m} muscles",
                                n.saturating_sub(1)
                            ));
                            ui.label(format!(
                                "{} · {:.1}%",
                                number(count as usize),
                                100. * count as f32 / body_count.max(1) as f32
                            ));
                        });
                    }
                });
        });
        ui.add_space(8.);
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
            self.set_preview(c, stats.config);
            self.tab = Tab::Overview;
        }
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
                                "Overview · Behavior archive · History · Race · Lineage",
                            ),
                            (
                                "Space",
                                "Play or pause the replay. Without a replay it pauses or resumes evolution.",
                            ),
                            ("← / →", "Seek one replay frame"),
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
                ui.heading("Tabs");
                for (name, why) in [
                    (
                        "Overview",
                        "The replay of the selected creature, its playback controls, lineage and fitness trend.",
                    ),
                    (
                        "Behavior archive",
                        "Every behavior niche the search protects, as cards or as a behavior map. Click a creature or an occupied map cell to replay it.",
                    ),
                    (
                        "History & statistics",
                        "Per-generation curves, every new record with a replay, the session hall of fame, the mix of body types, and the distribution of gait scores.",
                    ),
                    (
                        "Race",
                        "The five fastest archived creatures run their trials side by side with live standings.",
                    ),
                    (
                        "Lineage",
                        "Ancestors of the selected creature with thumbnails, fitness gains and body plan changes.",
                    ),
                ] {
                    ui.label(RichText::new(name).strong());
                    ui.label(RichText::new(why).color(theme.muted));
                    ui.add_space(4.);
                }
            });
    }
    fn dialogs(&mut self, ctx: &egui::Context) {
        if self.new_dialog {
            egui::Window::new("Start a new experiment")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "Create {} creatures using the settings in the sidebar.",
                        number(self.config.population)
                    ));
                    ui.label("Save the current experiment first if you want to resume it later.");
                    ui.horizontal(|ui| {
                        if ui.button("Save current").clicked() {
                            self.file("Save experiment");
                            self.new_dialog = false;
                        }
                        if ui
                            .add_enabled(
                                self.config.validate().is_ok(),
                                egui::Button::new("Create population")
                                    .fill(Color32::from_rgb(222, 241, 229)),
                            )
                            .clicked()
                        {
                            self.pause();
                            self.worker.send(Command::New(self.config.clone()));
                            self.initial = true;
                            self.dirty = false;
                            self.last_page = usize::MAX;
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
                                "Save experiment" => self.worker.send(Command::Save(path)),
                                "Open experiment" => {
                                    self.pause();
                                    self.worker.send(Command::Load(path));
                                    self.initial = true;
                                }
                                "Export CSV" => self.worker.send(Command::Export(path)),
                                "Save preset" => {
                                    let result = (|| -> anyhow::Result<()> {
                                        self.config.validate()?;
                                        if let Some(parent) = path.parent() {
                                            std::fs::create_dir_all(parent)?;
                                        }
                                        serde_json::to_writer_pretty(
                                            std::fs::File::create(path)?,
                                            &self.config,
                                        )?;
                                        Ok(())
                                    })();
                                    self.message =
                                        Some(result.map_or_else(
                                            |e| e.to_string(),
                                            |_| "Preset saved".into(),
                                        ));
                                }
                                "Load preset" => {
                                    let result = (|| -> anyhow::Result<Config> {
                                        let cfg: Config =
                                            serde_json::from_reader(std::fs::File::open(path)?)?;
                                        cfg.validate()?;
                                        Ok(cfg)
                                    })();
                                    match result {
                                        Ok(cfg) => {
                                            self.config = cfg;
                                            self.dirty = true;
                                        }
                                        Err(e) => self.message = Some(e.to_string()),
                                    }
                                }
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
                                            self.worker.send(Command::Preview { creature, config });
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
        let mut frames = self.bench_frames.clone();
        frames.sort_by(f32::total_cmp);
        let pct = |q: usize| frames[(frames.len() * q / 100).min(frames.len() - 1)] * 1000.;
        let total: f32 = frames.iter().sum();
        eprintln!(
            "Native benchmark frames: {} frames, {:.1} FPS, p50 {:.2} ms, p95 {:.2} ms, p99 {:.2} ms, max {:.2} ms",
            frames.len(),
            frames.len() as f32 / total.max(1e-6),
            pct(50),
            pct(95),
            pct(99),
            frames[frames.len() - 1] * 1000.
        );
    }
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
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
        self.frame_times.push_back(dt);
        if self.frame_times.len() > 240 {
            self.frame_times.pop_front();
        }
        if self.worker.measuring.load(Ordering::Relaxed) {
            self.bench_frames.push(dt);
            if self.bench_last_ping.elapsed() >= Duration::from_millis(500) {
                self.bench_last_ping = now;
                self.worker.send(Command::Ping(now));
            }
        }
        let next = self.worker.view.lock().unwrap().take();
        if let Some(mut next) = next {
            self.absorb_records(&next);
            self.absorb_archive_page(&next);
            if self
                .snapshot
                .as_ref()
                .is_some_and(|old| old.stage != next.stage)
            {
                self.sort_started = Instant::now();
            }
            if self.initial
                && !next.page.is_empty()
                && self
                    .snapshot
                    .as_ref()
                    .is_none_or(|old| old.epoch != next.epoch)
            {
                self.config = next.config.clone();
                self.initial = false;
            }
            if let Some((c, cfg)) = next.preview.take() {
                self.set_preview(c, cfg);
                self.lineage.clear();
            }
            if let Some(lineage) = next.lineage.take() {
                self.lineage = lineage;
                self.lineage_pending = false;
            }
            self.snapshot = Some(next);
            self.maybe_build_race();
        }
        let lineage_request =
            if (self.tab == Tab::Overview || self.tab == Tab::Lineage) && self.lineage.is_empty() {
                self.playback
                    .as_ref()
                    .filter(|p| self.lineage_requested != Some(p.creature.id))
                    .map(|p| (p.creature.clone(), p.config.clone()))
            } else {
                None
            };
        if let Some((creature, config)) = lineage_request {
            self.lineage_requested = Some(creature.id);
            self.lineage_pending = true;
            self.worker.send(Command::Preview { creature, config });
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
                if self.race.is_empty() {
                    self.race_pending = true;
                }
            }
            if pressed(egui::Key::Num5) {
                self.tab = Tab::Lineage;
            }
            if pressed(egui::Key::F1) || pressed(egui::Key::Questionmark) {
                self.show_help = !self.show_help;
            }
            if pressed(egui::Key::Space) {
                if self.playback.is_some() || (self.tab == Tab::Race && !self.race.is_empty()) {
                    self.playing = !self.playing;
                } else if self.active() {
                    self.pause();
                } else {
                    self.run(true, false);
                }
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
                    if self.follow {
                        let x =
                            p.nodes.iter().map(|n| n.pos[0]).sum::<f32>() / p.nodes.len() as f32;
                        self.camera[0] += (x - self.camera[0]) * (dt * 8.).min(1.0);
                    }
                }
                if self.tab == Tab::Race {
                    for lane in &mut self.race {
                        advance(&mut lane.playback);
                    }
                }
            }
        }
        let theme = self.theme();
        egui::Panel::top("top")
            .exact_size(64.)
            .frame(egui::Frame::new().fill(theme.panel).inner_margin(12))
            .show(ui, |ui| self.top(ui));
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(s) = &self.snapshot {
                    color_dot(ui, theme.accent);
                    ui.label(&s.status);
                    if let Some(error) = &s.error {
                        ui.colored_label(AMBER, error);
                    }
                    ui.label(
                        RichText::new(format!("{:.0} creatures/s", s.end_to_end))
                            .small()
                            .color(theme.muted),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button(if self.show_perf {
                                "Hide performance"
                            } else {
                                "Performance"
                            })
                            .clicked()
                        {
                            self.show_perf = !self.show_perf;
                        }
                        ui.label(RichText::new(&s.gpu).small().color(theme.muted));
                    });
                }
            });
            if self.show_perf && let Some(s) = &self.snapshot {
                let mut frames: Vec<_> = self.frame_times.iter().copied().collect();
                frames.sort_by(f32::total_cmp);
                let p95 = frames.get(frames.len() * 95 / 100).copied().unwrap_or(0.);
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
            }
            if let Some(m) = &self.message {
                ui.label(m);
            }
        });
        egui::Panel::left("controls")
            .default_size(300.)
            .min_size(260.)
            .max_size(440.)
            .resizable(true)
            .frame(egui::Frame::new().fill(theme.panel).inner_margin(16))
            .show(ui, |ui| self.controls(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme.canvas).inner_margin(20))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (tab, label) in [
                        (Tab::Overview, "Overview"),
                        (Tab::Population, "Behavior archive"),
                        (Tab::History, "History & statistics"),
                        (Tab::Race, "Race"),
                        (Tab::Lineage, "Lineage"),
                    ] {
                        let clicked = ui
                            .selectable_value(&mut self.tab, tab, RichText::new(label).size(15.))
                            .clicked();
                        if clicked && tab == Tab::Race && self.race.is_empty() {
                            self.race_pending = true;
                        }
                    }
                });
                ui.add_space(8.);
                match self.tab {
                    Tab::Overview => {
                        self.metrics(ui);
                        ui.add_space(10.);
                        self.viewport(ui, (ui.available_height() * 0.62).max(180.));
                        ui.add_space(8.);
                        self.lineage_strip(ui);
                        self.trend(ui, ui.available_height().max(100.));
                    }
                    Tab::Population => self.population(ui),
                    Tab::History => {
                        egui::ScrollArea::vertical().show(ui, |ui| self.history(ui));
                    }
                    Tab::Race => self.race_view(ui),
                    Tab::Lineage => self.lineage_view(ui),
                }
            });
        self.dialogs(&ctx);
        self.help_window(&ctx);
        if self.playing || self.active() {
            // Playback and live evolution redraw at the frame cap; the rest of
            // the GPU stays with evolution. EVOLUTION_UI_FPS=0 follows vsync.
            match ui_frame_interval() {
                Some(interval) => ctx.request_repaint_after(interval),
                None => ctx.request_repaint(),
            }
        }
        // Screenshot button: ask the viewport for one frame and save it as PNG.
        if self.screenshot_pending {
            self.screenshot_pending = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
                ScreenshotRequest,
            )));
        }
        // Explicit opt-in capture hook for repeatable native rendering/performance checks.
        if self.capture_path.is_some()
            && self.started.elapsed() > Duration::from_secs(8)
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
fn color_dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.), Sense::hover());
    ui.painter().circle_filled(rect.center(), 3., color);
}
fn paint_card(
    painter: &egui::Painter,
    card: &crate::worker::Card,
    rect: Rect,
    hovered: bool,
    stage: Stage,
    theme: Theme,
) {
    painter.rect_filled(
        rect,
        8,
        if hovered {
            theme.card_hover
        } else {
            theme.card
        },
    );
    painter.rect_stroke(
        rect,
        8,
        Stroke::new(1., theme.card_border),
        egui::StrokeKind::Inside,
    );
    thumbnail(painter, &card.creature, rect.shrink2(Vec2::new(10., 23.)));
    painter.text(
        rect.left_top() + Vec2::new(9., 8.),
        Align2::LEFT_TOP,
        if card.descriptor.is_some() || matches!(stage, Stage::Ranked | Stage::Selected) {
            format!("#{}", card.rank + 1)
        } else {
            format!("ID {}", card.creature.id)
        },
        FontId::proportional(11.),
        theme.muted,
    );
    painter.text(
        rect.left_top() + Vec2::new(9., 22.),
        Align2::LEFT_TOP,
        species_name(&card.creature),
        FontId::proportional(10.),
        theme.ink,
    );
    if card.innovation_reserve {
        painter.text(
            rect.right_top() + Vec2::new(-9., 8.),
            Align2::RIGHT_TOP,
            "MORPH",
            FontId::proportional(9.),
            theme.accent,
        );
    }
    let (label, score_color) = if !card.score.is_finite() {
        if card.parent_score.is_finite() && card.parent_score > FAILED {
            (format!("Parent {:.3} m", card.parent_score), theme.muted)
        } else if card.parent_score.is_finite() {
            ("Parent failed".into(), AMBER)
        } else {
            ("Trial pending".into(), theme.muted)
        }
    } else if card.score <= FAILED {
        ("Failed trial".into(), AMBER)
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
        FontId::proportional(12.),
        score_color,
    );
    if stage == Stage::Selected {
        painter.text(
            rect.right_top() + Vec2::new(-9., 8.),
            Align2::RIGHT_TOP,
            if card.survivor {
                "Survives"
            } else {
                "Replaced"
            },
            FontId::proportional(10.),
            if card.survivor { theme.accent } else { AMBER },
        );
    }
}
fn matches_search(q: &str, terms: &str) -> bool {
    q.is_empty() || terms.contains(q)
}
fn number(n: usize) -> String {
    let text = n.to_string();
    let mut out = String::new();
    for (i, c) in text.chars().enumerate() {
        if i > 0 && (text.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
fn species_color(n: usize, m: usize) -> Color32 {
    egui::ecolor::Hsva::new(((n * 257 + m) as f32 * 0.618034).fract(), 0.45, 0.9, 1.).into()
}
/// White clouds in the upper sky, drifting slowly against the camera.
fn draw_clouds(painter: &egui::Painter, rect: Rect, parallax: f32) {
    // Keep cloud centers and their reach clear of the rounded corners, so no
    // cloud paints in the clipped corner squares.
    let left = rect.left() + 40.0;
    let span = (rect.width() - 80.0).max(1.0);
    let top = rect.top() + 50.0;
    let floor = (rect.top() + 120.0).min(rect.bottom() - 60.0).max(top);
    for i in 0..5 {
        let x = left + (i as f32 * 211.0 - parallax * 0.12).rem_euclid(span);
        let y = top + (i * 67) as f32 % (floor - top).max(1.0);
        cloud(painter, Pos2::new(x, y), 0.75 + (i % 3) as f32 * 0.2);
    }
}
fn cloud(painter: &egui::Painter, center: Pos2, s: f32) {
    for &(dx, dy, r) in &[
        (0.0, 0.0, 26.0),
        (-30.0, 7.0, 19.0),
        (29.0, 8.0, 17.0),
        (5.0, -13.0, 17.0),
    ] {
        painter.add(egui::Shape::ellipse_filled(
            center + Vec2::new(dx * s, dy * s),
            Vec2::new(r * s, r * 0.6 * s),
            Color32::from_white_alpha(225),
        ));
    }
}
/// Linear blend between two colors; `t` is clamped to [0, 1].
fn mix_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}
fn draw_creature(
    p: &egui::Painter,
    nodes: &[Node],
    c: &Creature,
    origin: Pos2,
    scale: f32,
    time: f32,
    fallen: bool,
) {
    let position = |n: &Node| origin + Vec2::new(n.pos[0] * scale, -n.pos[1] * scale);
    for bone in &c.bones {
        let a = position(&nodes[bone.a as usize]);
        let b = position(&nodes[bone.b as usize]);
        let width = (scale * 0.032).max(3.0);
        p.line_segment(
            [a, b],
            Stroke::new(width + 3.0, Color32::from_rgb(10, 15, 19)),
        );
        p.line_segment([a, b], Stroke::new(width, Color32::from_rgb(192, 205, 187)));
    }
    // Organs ride on their bones; drawn with the density of a node.
    for bone in c.bones.iter().filter(|b| b.organ_mass > 0.0) {
        let a = nodes[bone.a as usize].pos;
        let b = nodes[bone.b as usize].pos;
        let t = bone.organ_at;
        let center = origin
            + Vec2::new(
                (a[0] + (b[0] - a[0]) * t) * scale,
                -(a[1] + (b[1] - a[1]) * t) * scale,
            );
        let r = (0.04 * (bone.organ_mass / 0.1).sqrt() * scale).max(2.5);
        p.circle_filled(center, r + 1.5, Color32::from_rgb(9, 17, 22));
        p.circle_filled(center, r, ORGAN);
        p.circle_filled(
            center + Vec2::new(-r * 0.25, -r * 0.3),
            r * 0.4,
            Color32::from_white_alpha(40),
        );
    }
    for m in &c.muscles {
        let bone_a = c.bones[m.bone_a as usize];
        let bone_b = c.bones[m.bone_b as usize];
        let point = |bone: crate::evolution::Bone, t: f32| {
            let a = [nodes[bone.a as usize].pos[0], nodes[bone.a as usize].pos[1]];
            let b = [nodes[bone.b as usize].pos[0], nodes[bone.b as usize].pos[1]];
            origin
                + Vec2::new(
                    (a[0] + (b[0] - a[0]) * t) * scale,
                    -(a[1] + (b[1] - a[1]) * t) * scale,
                )
        };
        let a = point(bone_a, m.anchor_a);
        let b = point(bone_b, m.anchor_b);
        // A fallen creature's muscles are limp.
        let contraction = if fallen {
            0.
        } else {
            1. - ((physics::target(m, time) - m.short) / (m.long - m.short).max(1e-5))
        };
        let width = (scale * 0.017 * (1. + 0.45 * contraction)).max(2.);
        p.line_segment(
            [a, b],
            Stroke::new(width + 3., Color32::from_rgb(10, 15, 19)),
        );
        // Pink at rest, deep red at full contraction.
        p.line_segment(
            [a, b],
            Stroke::new(width, mix_color(MUSCLE_REST, MUSCLE_ACTIVE, contraction)),
        );
    }
    for n in nodes {
        let center = position(n);
        let r = (n.radius * scale).max(2.);
        let color =
            egui::ecolor::Hsva::new(0.44 - 0.07 * n.friction, 0.3 + 0.4 * n.friction, 0.95, 1.);
        p.circle_filled(center, r + 1.5, Color32::from_rgb(9, 17, 22));
        p.circle_filled(center, r, Color32::from(color));
        p.circle_filled(
            center + Vec2::new(-r * 0.22, -r * 0.26),
            r * 0.5,
            Color32::from_white_alpha(35),
        );
        p.circle_stroke(center, r, Stroke::new(1., Color32::from_white_alpha(60)));
    }
    // The head (node 0) looks ahead with one eye.
    if let Some(head) = nodes.first() {
        let center = position(head);
        let r = (head.radius * scale).max(2.);
        if fallen {
            p.circle_stroke(center, r + 1.5, Stroke::new(2., FALLEN));
        }
        let eye = center + Vec2::new(r * 0.4, -r * 0.2);
        p.circle_filled(eye, r * 0.3, Color32::WHITE);
        p.circle_filled(
            eye + Vec2::new(r * 0.08, 0.),
            r * 0.15,
            Color32::from_rgb(9, 17, 22),
        );
    }
}
fn thumbnail(p: &egui::Painter, c: &Creature, rect: Rect) {
    let nodes = physics::nodes(c);
    let minx = nodes
        .iter()
        .map(|n| n.pos[0] - n.radius)
        .fold(f32::INFINITY, f32::min);
    let maxx = nodes
        .iter()
        .map(|n| n.pos[0] + n.radius)
        .fold(f32::NEG_INFINITY, f32::max);
    let miny = nodes
        .iter()
        .map(|n| n.pos[1] - n.radius)
        .fold(f32::INFINITY, f32::min);
    let maxy = nodes
        .iter()
        .map(|n| n.pos[1] + n.radius)
        .fold(f32::NEG_INFINITY, f32::max);
    let scale =
        (rect.width() / (maxx - minx).max(0.1)).min(rect.height() / (maxy - miny).max(0.1)) * 0.82;
    let origin =
        rect.center() + Vec2::new(-(minx + maxx) * 0.5 * scale, (miny + maxy) * 0.5 * scale);
    draw_creature(p, &nodes, c, origin, scale, 0., false);
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::{Bone, Muscle, NodeGene};
    fn test_creature() -> Creature {
        Creature {
            nodes: vec![
                NodeGene {
                    x: 0.0,
                    y: 0.0,
                    diameter: 0.2,
                    friction: 0.8,
                },
                NodeGene {
                    x: 0.5,
                    y: 0.0,
                    diameter: 0.2,
                    friction: 0.8,
                },
                NodeGene {
                    x: 1.0,
                    y: 0.0,
                    diameter: 0.2,
                    friction: 0.8,
                },
            ],
            bones: vec![Bone::new(0, 1, 0.5), Bone::new(1, 2, 0.5)],
            muscles: vec![Muscle {
                bone_a: 0,
                bone_b: 1,
                anchor_a: 0.5,
                anchor_b: 0.5,
                short: 0.4,
                long: 0.6,
                period: 0.8,
                phase: 0.0,
                duty: 0.5,
                stiffness: 10.0,
                sensor: crate::evolution::NO_SENSOR,
                reset: 0.0,
            }],
            id: 7,
            mutability: 1.0,
        }
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
    fn species_names_follow_the_body_plan() {
        let creature = test_creature();
        let name = species_name(&creature);
        let mut reversed = creature.clone();
        reversed.bones.reverse();
        assert_eq!(name, species_name(&reversed));
        let mut longer = creature.clone();
        longer.nodes.push(NodeGene {
            x: 1.5,
            y: 0.0,
            diameter: 0.2,
            friction: 0.8,
        });
        longer.bones.push(Bone::new(2, 3, 0.5));
        assert_ne!(name, species_name(&longer));
    }
}
