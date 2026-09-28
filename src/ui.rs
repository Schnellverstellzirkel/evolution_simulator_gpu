use crate::{
    config::Config,
    evolution::{Creature, FAILED},
    gpu::Gpu,
    physics::{self, Node},
    storage::{PERCENTILES, Stage, Stats},
    worker::{Command, EventKind, Snapshot, Worker},
};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Legend, Line, Plot, Points, VLine};
use image::{
    Delay as GifDelay, Frame as GifFrame, Rgba, RgbaImage,
    codecs::gif::{GifEncoder, Repeat as GifRepeat},
};
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
/// Ring around every node touching the ground in the current frame.
const TOUCHDOWN: Color32 = Color32::from_rgb(255, 196, 64);
const DEFAULT_CAMERA_ZOOM: f32 = 80.0;
/// How long the UI's own messages hold the status line.
const MESSAGE_SECONDS: f32 = 8.0;
/// How fast archive cards glide to their new places after a re-sort.
const SORT_SPEED: f32 = 5.0;
/// Generations between autosaves when the player turns autosave on.
const AUTOSAVE_INTERVAL: u32 = 10;
/// Exported GIFs render the same scene as the viewport into this frame size.
const GIF_WIDTH: u32 = 400;
const GIF_HEIGHT: u32 = 224;
/// Most frames an exported GIF keeps; a longer trial is sampled evenly.
const GIF_MAX_FRAMES: usize = 360;
/// Pixels per meter cap, so a tiny creature stays in frame whole.
const GIF_MAX_SCALE: f32 = 200.0;
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
/// How a trial ended early. The engines stop scoring at the first of three
/// events; the replay names the one that happened.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Ending {
    /// The head dropped below its neck.
    Fell,
    /// A joint was forced past its break angle.
    Broke,
    /// The head's averaged acceleration passed the 8 g limit.
    Shook,
}
impl Ending {
    /// Sentence for the replay, given the time of the event in seconds.
    fn sentence(self, seconds: f32) -> String {
        match self {
            Self::Fell => format!("Fell over at {seconds:.1} s: head below its neck"),
            Self::Broke => format!("Broke a joint at {seconds:.1} s"),
            Self::Shook => format!("Shook its head too hard at {seconds:.1} s (over 8 g)"),
        }
    }
    /// A word or two for a race lane.
    fn short(self) -> &'static str {
        match self {
            Self::Fell => "fell",
            Self::Broke => "broke a joint",
            Self::Shook => "shook too hard",
        }
    }
}
/// Replays a creature's trial as simulated by the evaluation engines.
struct Playback {
    creature: Creature,
    config: Config,
    nodes: Vec<Node>,
    /// Joint ranges of the creature, for spotting broken joints per frame.
    joints: Vec<physics::Joint>,
    /// Node positions after each step, from the CPU evaluation engine.
    frames: Vec<Vec<[f32; 2]>>,
    tick: u32,
    accumulator: f32,
    /// Frame at which the trial ended early (a fall, a broken joint, or a
    /// shaken head), and the distance the trial kept from that moment.
    fall: Option<(u32, f32)>,
    /// Which of the three events ended the trial, when one did.
    ending: Ending,
    /// The distance the CPU engine scored for this very recording.
    distance: f32,
    /// Where the follow camera looks at each frame: the body's center of
    /// mass averaged over `CAMERA_WINDOW` seconds on either side. Every
    /// frame is recorded in advance, so the average cancels the swing of
    /// each stride without lagging behind a steady walk.
    track: Vec<f32>,
}
/// Half-width of the follow camera's average of the center of mass (s).
const CAMERA_WINDOW: f32 = 1.0;
/// The follow camera's target per recorded frame: the mass-weighted center of
/// the body, averaged over `CAMERA_WINDOW` seconds on either side.
fn camera_track(frames: &[Vec<[f32; 2]>], nodes: &[Node]) -> Vec<f32> {
    let mass: f32 = nodes.iter().map(|n| n.mass).sum::<f32>().max(1e-6);
    let centers: Vec<f64> = frames
        .iter()
        .map(|frame| {
            let x: f32 = frame.iter().zip(nodes).map(|(p, n)| n.mass * p[0]).sum();
            f64::from(x / mass)
        })
        .collect();
    let mut sums = Vec::with_capacity(centers.len() + 1);
    sums.push(0.0f64);
    for c in &centers {
        sums.push(sums.last().unwrap() + c);
    }
    let half = (CAMERA_WINDOW * physics::rate() as f32).round() as usize;
    (0..centers.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(centers.len()));
            ((sums[b] - sums[a]) / (b - a) as f64) as f32
        })
        .collect()
}
impl Playback {
    fn new(creature: Creature, config: Config) -> Self {
        let mut normalized = creature.clone();
        crate::evolution::canonicalize_bone_order(&mut normalized);
        // The engine that recorded the frames also decides when the trial
        // ended and how far it got, so the replay shows exactly its score.
        let (frames, result) = crate::cpu_engine::replay(&normalized, &config);
        let nodes = physics::nodes(&normalized);
        let joints = physics::joints(&normalized.nodes, &normalized.bones);
        let last_frame = frames.len().saturating_sub(1).min(u32::MAX as usize) as u32;
        let fall = (result.fall_time > 0.0).then(|| {
            let tick = physics::settle()
                .saturating_add((result.fall_time * physics::rate() as f32).round() as u32);
            (tick.min(last_frame), result.fitness)
        });
        let track = camera_track(&frames, &nodes);
        // The head-shake average stops updating when the trial ends, so it
        // still holds the value that ended it. A broken joint shows in the
        // recorded pose at the end (the engine tests the pose after the step).
        let ending = match fall {
            _ if result.head_shake > physics::HEAD_SHAKE_LIMIT => Ending::Shook,
            Some((tick, _)) => {
                let mut broken = vec![false; nodes.len()];
                let broke = [tick, tick.saturating_sub(1)].iter().any(|&t| {
                    frames.get(t as usize).is_some_and(|frame| {
                        broken_nodes(&normalized, frame, &joints, &mut broken);
                        broken.iter().any(|&b| b)
                    })
                });
                if broke { Ending::Broke } else { Ending::Fell }
            }
            None => Ending::Fell,
        };
        let mut playback = Self {
            nodes,
            joints,
            fall,
            ending,
            distance: result.fitness,
            track,
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
    /// Share of the way to the next recorded frame that the replay clock
    /// has gone.
    fn blend(&self) -> f32 {
        (self.accumulator / physics::dt()).clamp(0.0, 1.0)
    }
    /// Places the nodes between the current frame and the next by `blend`,
    /// so motion looks smooth when the screen refreshes faster than the
    /// 60 Hz recording.
    fn show_between(&mut self) {
        let alpha = self.blend();
        let (Some(now), Some(next)) = (
            self.frames.get(self.tick as usize),
            self.frames.get(self.tick as usize + 1),
        ) else {
            return self.show();
        };
        for ((node, a), b) in self.nodes.iter_mut().zip(now).zip(next) {
            node.pos = [a[0] + (b[0] - a[0]) * alpha, a[1] + (b[1] - a[1]) * alpha];
        }
    }
    /// Where the follow camera looks now (see `track`).
    fn camera_x(&self) -> f32 {
        let at = |tick: usize| {
            self.track
                .get(tick)
                .or(self.track.last())
                .copied()
                .unwrap_or(0.0)
        };
        let tick = self.tick as usize;
        let alpha = self.blend();
        at(tick) + (at(tick + 1) - at(tick)) * alpha
    }
    /// Mass-weighted center of the body as drawn.
    fn shown_center(&self) -> Option<[f32; 2]> {
        let mass: f32 = self.nodes.iter().map(|n| n.mass).sum();
        (mass > 0.0).then(|| {
            let x = self.nodes.iter().map(|n| n.mass * n.pos[0]).sum::<f32>();
            let y = self.nodes.iter().map(|n| n.mass * n.pos[1]).sum::<f32>();
            [x / mass, y / mass]
        })
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
/// Marks the nodes touching the ground in `positions`, with the threshold
/// `size_report` uses: a node is down when its center sits within 2 mm of the
/// terrain surface plus its own radius measured along the local normal. Gaps,
/// hurdles and the creature's own quake phase lower and raise the surface here
/// too, so the marks follow the ground that is drawn.
fn node_contact(
    nodes: &[Node],
    positions: &[[f32; 2]],
    creature: &Creature,
    config: &Config,
    out: &mut [bool],
) {
    out.fill(false);
    if !config.ground {
        return;
    }
    let hash = physics::quake_hash(creature.id);
    let amplitude =
        physics::terrain_amplitude(config.terrain) + config.quake * physics::quake_scale(hash);
    let phase = if config.quake > 0.0 {
        physics::quake_phase(hash)
    } else {
        0.0
    };
    for ((node, position), down) in nodes.iter().zip(positions).zip(out.iter_mut()) {
        let (height, slope) = physics::ground(
            position[0],
            amplitude,
            config.slope,
            config.gaps,
            config.hurdles,
            phase,
        );
        let floor = height + node.radius * (1.0 + slope * slope).sqrt();
        *down = position[1] <= floor + 0.002;
    }
}
/// Marks both ends of every bone whose joint is forced past its break angle in
/// `positions`. Mirrors `physics::broken_joint`, one bone at a time, so the
/// drawing can point at the joint that actually broke.
fn broken_nodes(
    creature: &Creature,
    positions: &[[f32; 2]],
    joints: &[physics::Joint],
    out: &mut [bool],
) {
    out.fill(false);
    for (bone, joint) in creature.bones.iter().zip(joints) {
        let Some(reference) = joint.reference else {
            continue;
        };
        let pivot = positions[bone.a as usize];
        let at = |i: usize| [positions[i][0] - pivot[0], positions[i][1] - pivot[1]];
        let (u, v) = (at(reference), at(bone.b as usize));
        let norm = ((u[0] * u[0] + u[1] * u[1]) * (v[0] * v[0] + v[1] * v[1])).sqrt();
        if norm < 1e-12 {
            continue;
        }
        let cos = (u[0] * v[0] + u[1] * v[1]) / norm;
        let sin = (u[0] * v[1] - u[1] * v[0]) / norm;
        if cos * joint.center[0] + sin * joint.center[1] < physics::joint_break_cos(joint.half) {
            out[bone.a as usize] = true;
            out[bone.b as usize] = true;
        }
    }
}
/// A red cross over a node that fell, shook, or broke its joint.
fn draw_break_mark(p: &egui::Painter, center: Pos2, radius: f32) {
    let d = radius.max(3.5);
    let arm = |dx: f32, dy: f32| {
        p.line_segment(
            [
                center + Vec2::new(-dx * d, -dy * d),
                center + Vec2::new(dx * d, dy * d),
            ],
            Stroke::new(2.0, FALLEN),
        );
    };
    arm(1.0, 1.0);
    arm(1.0, -1.0);
}
/// Frame-varying drawing state: muscle time, fallen look, and the per-node
/// marks for ground contact and broken joints.
#[derive(Default)]
struct FrameMarks {
    time: f32,
    fallen: bool,
    contact: Vec<bool>,
    broken: Vec<bool>,
}
impl FrameMarks {
    /// Contact and broken-joint marks of a playback's current frame.
    fn of(playback: &Playback) -> Self {
        let mut marks = Self {
            time: playback.tick.saturating_sub(physics::settle()) as f32 * physics::dt(),
            fallen: playback.fallen().is_some(),
            contact: vec![false; playback.nodes.len()],
            broken: vec![false; playback.nodes.len()],
        };
        if let Some(frame) = playback.frames.get(playback.tick as usize) {
            node_contact(
                &playback.nodes,
                frame,
                &playback.creature,
                &playback.config,
                &mut marks.contact,
            );
            broken_nodes(
                &playback.creature,
                frame,
                &playback.joints,
                &mut marks.broken,
            );
        }
        marks
    }
}
/// Behavior-axis bin counts, mirroring `qd::BINS` (ground contact, cadence,
/// bounce, height, feet). Bounce keeps one bin, so it adds no map cell.
const MAP_BINS: [usize; 5] = [6, 8, 1, 6, 5];
/// Which representation the Behavior archive tab shows.
#[derive(Clone, Copy, PartialEq)]
enum ArchiveView {
    Cards,
    Map,
}
/// One archive elite running in the race view.
struct RaceLane {
    /// Place in the archive ranking.
    rank: usize,
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
/// Scales a plot range around its center. A factor below one zooms in.
fn scaled_range(
    range: std::ops::RangeInclusive<f64>,
    factor: f64,
) -> std::ops::RangeInclusive<f64> {
    let center = (range.start() + range.end()) / 2.0;
    let half = (range.end() - range.start()) / 2.0 * factor;
    (center - half)..=(center + half)
}
/// What a line of the event feed lets the player do.
#[derive(Clone, Copy)]
enum FeedAction {
    /// Replay the best creature of this history row.
    Replay(usize),
    /// Bring back creatures lost to catastrophes.
    Undo,
}
/// One line of the event feed.
struct FeedItem {
    generation: u32,
    text: String,
    color: Color32,
    action: Option<FeedAction>,
}
/// History positions where the best distance moved within one world, oldest
/// first, and whether each is the first best after a world change. A harder
/// world lowers the best, so records count again from its first generation.
fn world_records(history: &[Stats]) -> Vec<(usize, f32, bool)> {
    let mut best = f32::NEG_INFINITY;
    let mut records = Vec::new();
    for (index, stats) in history.iter().enumerate() {
        if index > 0 && stats.config.physics_differs(&history[index - 1].config) {
            best = f32::NEG_INFINITY;
        }
        if stats.best.is_finite() && stats.best > best {
            let first = best == f32::NEG_INFINITY;
            best = stats.best;
            records.push((index, best, first && index > 0));
        }
    }
    records
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
    /// The world the record was set in.
    config: Config,
}
/// Heat map of the archive: for each ground contact and cadence pair, the
/// best creature among the height and feet bins the filters let through.
/// One color scale spans every cell of the archive, so a color means the
/// same distance whatever the filters. Returns the id of a clicked cell's
/// creature.
fn paint_archive_map(
    ui: &mut egui::Ui,
    cells: &[crate::worker::MapCell],
    height_bin: Option<usize>,
    feet_bin: Option<usize>,
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
            FontId::proportional(10.),
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
            FontId::proportional(10.),
            theme.muted,
        );
    }
    painter.text(
        Pos2::new(plot.center().x, plot.bottom() + 24.),
        Align2::CENTER_TOP,
        "Share of the trial on the ground",
        FontId::proportional(11.),
        theme.ink,
    );
    painter.text(
        Pos2::new(rect.left() + 4., plot.top() - 16.),
        Align2::LEFT_BOTTOM,
        "Strides per second",
        FontId::proportional(11.),
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
    if best.is_empty() {
        painter.text(
            plot.center(),
            Align2::CENTER_CENTER,
            if cells.is_empty() {
                "No creatures kept yet. The map fills as evolution runs."
            } else {
                "No creatures with this height and these feet. Try All."
            },
            FontId::proportional(13.),
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
                FontId::proportional(11.),
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
    percentiles: [bool; 29],
    history_index: usize,
    history_latest: bool,
    file_mode: Option<&'static str>,
    file_path: String,
    message: Option<String>,
    /// The message on the status line and when it first showed.
    shown_message: Option<(String, Instant)>,
    new_dialog: bool,
    /// When the UI last sent a settings change to the worker.
    config_sent: Option<Instant>,
    last_frame: Instant,
    frame_times: std::collections::VecDeque<f32>,
    last_page: usize,
    /// The Diagnostics drawer under the status line is open.
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
    /// The player picked the creature on screen, so the theater stops
    /// following the champion until they go back to it.
    pinned: bool,
    /// The creature on screen is the champion of a finished generation (or
    /// the best elite of a loaded game), not a random first creature.
    champion_shown: bool,
    /// A newer champion that replaces the one on screen when its replay loops.
    next_champion: Option<(Creature, Config)>,
    /// Behavior archive map: occupied cells keyed by their niche bytes.
    /// Map filters; None shows every bin.
    map_height: Option<usize>,
    map_feet: Option<usize>,
    /// Whether the worker was last asked to send the map table.
    map_sent: bool,
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
    bench_pings: u64,
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
            percentiles,
            history_index: 0,
            history_latest: true,
            file_mode: None,
            file_path: "runs/experiment.evo".into(),
            message: None,
            shown_message: None,
            new_dialog: false,
            config_sent: None,
            last_frame: Instant::now(),
            frame_times: Default::default(),
            last_page: usize::MAX,
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
            pinned: false,
            champion_shown: false,
            next_champion: None,
            map_height: None,
            map_feet: None,
            map_sent: false,
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
            bench_pings: 0,
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
        self.playback = Some(Playback::new(c, cfg));
        self.follow = true;
        self.zoom = DEFAULT_CAMERA_ZOOM;
        self.camera = [0.; 2];
    }
    /// Shows a creature the player picked. The theater keeps it until the
    /// player goes back to the champion.
    fn select(&mut self, creature: Creature, config: Config) {
        self.pinned = true;
        self.next_champion = None;
        self.set_preview(creature, config);
        self.lineage.clear();
    }
    /// Shows an ancestor from the lineage the player is browsing, keeping the
    /// lineage on screen.
    fn select_ancestor(&mut self, creature: Creature, config: Config) {
        self.pinned = true;
        self.next_champion = None;
        self.set_preview(creature, config);
    }
    /// Shows a champion and follows new ones from now on.
    fn show_champion(&mut self, creature: Creature, config: Config) {
        self.pinned = false;
        self.champion_shown = true;
        self.next_champion = None;
        self.set_preview(creature, config);
        self.lineage.clear();
    }
    /// The best elite of the newest finished generation, and the world it was
    /// scored in.
    fn champion(&self) -> Option<(Creature, Config)> {
        let stats = self.snapshot.as_ref()?.history.last()?;
        Some((stats.representatives.last()?.clone(), stats.config.clone()))
    }
    /// Keeps the theater on the champion unless the player pinned a creature.
    /// A new champion waits until the replay on screen loops, so the player
    /// sees the whole trial, unless the screen shows no champion yet.
    fn follow_champion(&mut self) {
        if self.pinned {
            return;
        }
        let Some((creature, config)) = self.champion() else {
            return;
        };
        let showing = self.playback.as_ref().map(|p| p.creature.id);
        if showing == Some(creature.id) {
            self.next_champion = None;
            return;
        }
        if self
            .next_champion
            .as_ref()
            .is_some_and(|(queued, _)| queued.id == creature.id)
        {
            return;
        }
        if showing.is_none() || !self.champion_shown {
            self.show_champion(creature, config);
        } else {
            self.next_champion = Some((creature, config));
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
            ui.add_space(12.);
            let running = self.active();
            let (text, fill, why) = if running {
                (
                    "Pause evolution  (Space)",
                    Color32::from_rgb(255, 239, 216),
                    "Stop after the work in flight. The replay keeps playing.",
                )
            } else {
                (
                    "Evolve  (Space)",
                    Color32::from_rgb(222, 241, 229),
                    "Run generation after generation until you pause.",
                )
            };
            if ui
                .add(
                    egui::Button::new(RichText::new(text).strong().color(INK))
                        .fill(fill)
                        .min_size(Vec2::new(150., 34.)),
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
                    if ui.checkbox(&mut self.dark, "Dark theme").changed() {
                        apply_style(ui.ctx(), self.dark);
                    }
                    if ui
                        .add(egui::Slider::new(&mut self.ui_scale, 0.75..=1.6).text("UI scale"))
                        .changed()
                    {
                        ui.ctx().set_zoom_factor(self.ui_scale);
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui.button("New experiment…").clicked() {
                        self.new_dialog = true;
                        ui.close();
                    }
                    if ui.button("Open…").clicked() {
                        self.file("Open experiment");
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
                if let Some(s) = &self.snapshot {
                    ui.label(
                        RichText::new(format!(
                            "{} creatures · {:.0} s trials",
                            number(s.config.population),
                            s.config.duration
                        ))
                        .small()
                        .color(theme.muted),
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
        ui.add_space(6.);
        let mut world_changed = false;
        ui.label(RichText::new("World").strong());
        let live = self.snapshot.as_ref().map(|s| s.config.clone());
        let calm = world_is_calm(&self.config);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(world_summary(&self.config)).color(if calm {
                theme.muted
            } else {
                theme.accent
            }));
            if !calm
                && ui
                    .small_button("Calm world")
                    .on_hover_text("Set every effect back to the calm world in one change.")
                    .clicked()
            {
                for effect in &crate::environment::EFFECTS {
                    if effect.name != "Seasons" {
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
                .color(AMBER),
            );
        }
        ui.label(
            RichText::new(
                "Click a level to change the world. The best creatures are tested again in the new world.",
            )
            .small()
            .color(theme.muted),
        );
        egui::Grid::new("world_effects")
            .num_columns(2)
            .spacing([8., 6.])
            .show(ui, |ui| {
                for effect in crate::environment::EFFECTS
                    .iter()
                    .filter(|effect| effect.name != "Seasons")
                {
                    if effect_row(ui, effect, &mut self.config, live.as_ref(), theme) {
                        world_changed = true;
                    }
                    ui.end_row();
                }
            });
        ui.add_space(2.);
        egui::Grid::new("world_seasons")
            .num_columns(2)
            .spacing([8., 6.])
            .show(ui, |ui| {
                if let Some(seasons) = crate::environment::EFFECTS
                    .iter()
                    .find(|effect| effect.name == "Seasons")
                    && effect_row(ui, seasons, &mut self.config, None, theme)
                {
                    world_changed = true;
                }
                ui.end_row();
            });
        let generation = self.snapshot.as_ref().map_or(0, |s| s.generation);
        if let Some(forecast) = season_forecast(&self.config, generation) {
            ui.label(RichText::new(forecast).small().color(theme.muted));
        }
        let fossils = self.snapshot.as_ref().map_or(0, |s| s.fossils);
        ui.add_space(4.);
        ui.label(RichText::new("Catastrophes").strong()).on_hover_text(
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
            self.worker.send(Command::Configure(self.config.clone()));
            self.config_sent = Some(Instant::now());
        }
    }
    fn viewport(&mut self, ui: &mut egui::Ui, height: f32) {
        let theme = self.theme();
        let mut back = false;
        let mut play_next = false;
        ui.horizontal(|ui| {
            let (mode, color, why) = if self.pinned {
                (
                    "WATCHING",
                    theme.ink,
                    "A creature you picked. Back to champion shows the best creature again.",
                )
            } else if self.champion_shown {
                (
                    "CHAMPION",
                    theme.accent,
                    "The best creature so far. The view switches to each new champion when the replay on screen ends.",
                )
            } else {
                (
                    "FIRST GENERATION",
                    theme.muted,
                    "A random creature of the first generation. The champion takes over when the first generation ends.",
                )
            };
            ui.label(RichText::new(mode).small().strong().color(color))
                .on_hover_text(why);
            if let Some(p) = &self.playback {
                ui.label(format!(
                    "{} · {:.2} m · {} nodes, {} bones, {} muscles · {:.2} m/s",
                    species_name(&p.creature),
                    p.distance,
                    p.nodes.len(),
                    p.creature.bones.len(),
                    p.creature.muscles.len(),
                    p.speed(),
                ))
                .on_hover_text(format!(
                    "Creature {}. {:.2} m is the distance this replay reaches, and m/s its speed over the last fifth of a second. The archive keeps the worse of this trial and a check from a slightly shifted pose at four times the physics rate, so its score is never higher.",
                    p.creature.id, p.distance
                ));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.follow, "Follow");
                if ui.small_button("Reset camera").clicked() {
                    self.zoom = DEFAULT_CAMERA_ZOOM;
                    self.camera = [0.; 2];
                    self.follow = true;
                }
                if self.pinned {
                    back = ui
                        .button(RichText::new("Back to champion").color(theme.accent))
                        .clicked();
                } else if let Some((next, _)) = &self.next_champion {
                    play_next = ui
                        .small_button("Play now")
                        .on_hover_text("Show the new champion without waiting")
                        .clicked();
                    ui.label(
                        RichText::new(format!("New champion {} plays next", species_name(next)))
                            .small()
                            .color(theme.accent),
                    );
                }
            });
        });
        if back {
            self.back_to_champion();
        }
        if play_next && let Some((creature, config)) = self.next_champion.take() {
            self.show_champion(creature, config);
        }
        let (rect, response) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), height.max(120.)),
            Sense::click_and_drag(),
        );
        if response.clicked() {
            self.playing = !self.playing;
        }
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
        if cfg.ground
            && (amplitude > 0.0 || slope != 0.0 || gaps > 0.0 || hurdles > 0.0 || mud > 0.0)
        {
            // Sample the ground every few pixels and fill down to the frame.
            // Pits carve notches into the polyline; mud draws its sunk layer
            // `mud` meters below the surface line.
            let step = (4.0 / self.zoom).max(0.002);
            let start = (rect.left() - origin.x) / self.zoom;
            let end = (rect.right() - origin.x) / self.zoom;
            let mut x = start;
            let mut line = Vec::new();
            let mut mud_line = Vec::new();
            while x <= end + step {
                let height = crate::physics::ground(x, amplitude, slope, gaps, hurdles, phase).0;
                line.push(world(x, height));
                if mud > 0.0 {
                    mud_line.push(world(x, height - mud));
                }
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
            if mud > 0.0 {
                let fill = Color32::from_rgb(103, 76, 52);
                for i in 0..line.len().saturating_sub(1) {
                    painter.add(egui::Shape::convex_polygon(
                        vec![line[i], line[i + 1], mud_line[i + 1], mud_line[i]],
                        fill,
                        Stroke::NONE,
                    ));
                }
                painter.add(egui::Shape::line(
                    mud_line,
                    Stroke::new(1., Color32::from_rgb(72, 51, 34)),
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
            if let Some(com) = p.shown_center() {
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
            let marks = FrameMarks::of(p);
            draw_creature(&painter, &p.nodes, &p.creature, origin, self.zoom, &marks);
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
                        p.ending.sentence(
                            tick.saturating_sub(physics::settle()) as f32 * physics::dt(),
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
        if let Some(p) = &self.playback {
            let live = self.snapshot.as_ref().map(|s| &s.config);
            let earlier = live.is_some_and(|live| live.physics_differs(&p.config));
            painter.text(
                rect.right_top() + Vec2::new(-14., 12.),
                Align2::RIGHT_TOP,
                if earlier {
                    format!("{} (an earlier world)", world_summary(&p.config))
                } else {
                    world_summary(&p.config)
                },
                FontId::proportional(13.),
                INK,
            );
        }
        painter.text(
            rect.left_bottom() + Vec2::new(14., -12.),
            Align2::LEFT_BOTTOM,
            "Click to pause · drag to pan · scroll to zoom",
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
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        let Some(s) = snapshot.history.last() else {
            ui.label(
                RichText::new(
                    "The first generation is running. Its best creature appears here when it ends.",
                )
                .color(theme.muted),
            );
            return;
        };
        let history = &snapshot.history;
        let gain = history
            .len()
            .checked_sub(11)
            .map(|earlier| s.best - history[earlier].best);
        let population = snapshot.config.population.max(1);
        let share = snapshot.completed.min(population) as f64 / population as f64;
        let rate = snapshot.end_to_end;
        let progress = if snapshot.running && rate > 0.0 {
            let left = (population - snapshot.completed.min(population)) as f64 / rate;
            format!(
                "{:.0}% done · about {} left",
                share * 100.0,
                seconds_text(left)
            )
        } else if snapshot.running {
            format!("{:.0}% done", share * 100.0)
        } else {
            "Paused".to_owned()
        };
        let trial = format!(
            "How far the best creature travels in its {:.0} s trial. Distance is the only score.",
            snapshot.config.duration
        );
        ui.columns(3, |cols| {
            for (ui, (name, value, color, note, why)) in cols.iter_mut().zip([
                (
                    "BEST DISTANCE",
                    format!("{:.2} m", s.best),
                    theme.accent,
                    gain.map_or_else(
                        || "so far".to_owned(),
                        |gain| format!("{gain:+.2} m in the last 10 generations"),
                    ),
                    trial.as_str(),
                ),
                (
                    "GENERATION",
                    snapshot.generation.to_string(),
                    theme.ink,
                    progress,
                    "Every generation tries a whole population of new creatures.",
                ),
                (
                    "KINDS OF MOVEMENT",
                    number(s.archive_cells),
                    theme.ink,
                    "different ways of moving kept".to_owned(),
                    "Evolution keeps the best creature for each way of moving: how much of the time it touches the ground, its stride rate, its height and how many feet it uses.",
                ),
            ]) {
                egui::Frame::new()
                    .fill(theme.card)
                    .corner_radius(8)
                    .inner_margin(12)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.label(RichText::new(name).small().color(theme.muted))
                            .on_hover_text(why);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(value).size(22.).color(color));
                            ui.label(RichText::new(note).small().color(theme.muted));
                        });
                    });
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
            ui.label(
                RichText::new("Drag to pan · double-click or Reset to fit again")
                    .small()
                    .color(theme.muted),
            );
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
            .legend(Legend::default())
            .x_axis_label("Generation")
            .y_axis_label("Distance (m)")
            .allow_scroll(false);
        if reset {
            plot = plot.reset();
        }
        plot.show(ui, |plot| {
            if zoom != 1.0 {
                let bounds = plot.plot_bounds();
                plot.set_plot_bounds_x(scaled_range(bounds.range_x(), zoom));
                plot.set_plot_bounds_y(scaled_range(bounds.range_y(), zoom));
            }
            if let Some(generations) = last {
                let end = s.history.last().map_or(1.0, |h| h.generation as f64 + 0.5);
                plot.set_plot_bounds_x((end - generations).max(0.0)..=end);
                plot.set_auto_bounds(egui::Vec2b::new(false, true));
            }
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
            // A vertical line where the world changed: the generation that
            // first ran in the new world.
            for pair in s.history.windows(2) {
                if pair[1].config.physics_differs(&pair[0].config) {
                    let season = pair[1].config.season_step != pair[0].config.season_step
                        && pair[1].config.seasons > 0;
                    plot.vline(
                        VLine::new(
                            if season { "Season" } else { "World change" },
                            pair[1].generation as f64 - 0.5,
                        )
                        .color(AMBER)
                        .width(1.5),
                    );
                }
            }
            // Record markers extend the best line instead of duplicating it.
            // Records count again after a world change.
            let records: Vec<[f64; 2]> = world_records(&s.history)
                .into_iter()
                .map(|(index, best, _)| [s.history[index].generation as f64, best as f64])
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
            ui.label(
                RichText::new("The best creature for each way of moving · click one to replay it")
                    .color(theme.muted),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .selectable_value(&mut self.archive_view, ArchiveView::Map, "Map")
                    .on_hover_text(
                        "Watch evolution fill the ways of moving. Cells are colored by distance.",
                    )
                    .clicked()
                {
                    self.last_page = usize::MAX;
                }
                if ui
                    .selectable_value(&mut self.archive_view, ArchiveView::Cards, "Cards")
                    .clicked()
                {
                    self.last_page = usize::MAX;
                }
            });
        });
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
                ui.label(
                    RichText::new("Click a cell to replay its creature.")
                        .small()
                        .color(theme.muted),
                );
            });
        }
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(format!(
                    "{} ways of moving, sorted by",
                    number(snapshot.archive_cells)
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
        let mut selected = None;
        let mut requested = None;
        let mut map_click = None;
        if self.archive_view == ArchiveView::Map {
            let empty = Vec::new();
            let cells = snapshot.map.as_deref().unwrap_or(&empty);
            map_click = paint_archive_map(ui, cells, self.map_height, self.map_feet, theme);
        } else {
            let columns = (ui.available_width() / 155.).floor().max(2.) as usize;
            let width = (ui.available_width() - (columns - 1) as f32 * 10.) / columns as f32;
            let progress = (self.sort_started.elapsed().as_secs_f32() * SORT_SPEED / 3.).min(1.);
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
                                        "{}\n{} nodes, {} bones, {} muscles\n{}\n{}\nClick to replay",
                                        species_name(&card.creature),
                                        card.creature.nodes.len(),
                                        card.creature.bones.len(),
                                        card.creature.muscles.len(),
                                        card.emitter.map_or("First generation".to_owned(), |emitter| format!("Born {}", origin_words(emitter))),
                                        card.descriptor.map_or_else(
                                            || if card.score.is_finite() { "Trial done".to_owned() } else { "Trial running".to_owned() },
                                            |d| format!("On the ground {:.0}% of the time · {:.2} strides/s · {:.2} m tall · {:.0} feet", d.ground_contact * 100.0, d.gait_frequency, d.mean_height, d.feet),
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
        if let Some(id) = map_click {
            self.worker.send(Command::Select(id));
        }
        if let Some(start) = requested {
            self.worker.send(Command::Page(start));
            self.last_page = start;
        }
        if let Some((creature, config)) = selected {
            self.select(creature, config);
            self.tab = Tab::Overview;
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
                        config: stats.config.clone(),
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
        self.select(creature, config);
        self.tab = Tab::Overview;
    }
    /// The lines of the event feed, newest first: the worker's events (world
    /// changes, seasons, catastrophes, saves) and the records in the history.
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
                EventKind::Catastrophe => {
                    (AMBER, (snapshot.fossils > 0).then_some(FeedAction::Undo))
                }
                EventKind::World | EventKind::Season => {
                    if let (Some(before), Some(after)) = (
                        event.generation.checked_sub(1).and_then(row),
                        row(event.generation),
                    ) {
                        text.push_str(&format!(
                            " Best {:.2} m before, {:.2} m after one generation.",
                            before.best, after.best
                        ));
                    }
                    if event.kind == EventKind::Season {
                        text.insert_str(0, "Season: ");
                    }
                    (theme.accent, None)
                }
                _ => (theme.muted, None),
            };
            items.push(FeedItem {
                generation: event.generation,
                text,
                color,
                action,
            });
        }
        for (index, best, first_in_world) in world_records(history) {
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
        ui.label(RichText::new("WHAT HAPPENED").small().color(theme.muted));
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
                                FeedAction::Replay(_) => "Replay",
                                FeedAction::Undo => "Undo",
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
            Some(FeedAction::Undo) => self.worker.send(Command::UndoMeteor),
            None => {}
        }
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
            let (creature, config) = (entry.creature.clone(), entry.config.clone());
            self.select(creature, config);
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
        let mut lanes: Vec<RaceLane> = snapshot
            .page
            .iter()
            .filter(|card| card.descriptor.is_some() && card.score.is_finite())
            .take(5)
            .map(|card| RaceLane {
                rank: card.rank,
                playback: Playback::new(card.creature.clone(), config.clone()),
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
                        Vec2::new(ui.available_width(), 104.),
                    ) {
                        chosen = Some(k);
                    }
                    ui.add_space(6.);
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
        // The leader's averaged center of mass, so its stride does not shake
        // the view; the easing below smooths a change of leader.
        let target = (self.race[leader].playback.camera_x() - visible * 0.6).max(0.0);
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
            let marks = FrameMarks::of(playback);
            draw_creature(
                &painter,
                &playback.nodes,
                &playback.creature,
                origin,
                zoom,
                &marks,
            );
            painter.text(
                lane_rect.left_top() + Vec2::new(8., 6.),
                Align2::LEFT_TOP,
                format!("{}. {}", i + 1, species_name(&lane.playback.creature)),
                FontId::proportional(13.),
                if is_leader { theme.accent } else { theme.ink },
            );
            painter.text(
                lane_rect.left_top() + Vec2::new(8., 23.),
                Align2::LEFT_TOP,
                format!(
                    "finishes at {:.2} m · archive rank {}",
                    lane.playback.distance,
                    lane.rank + 1
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
                    playback.ending.short(),
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
                format!("{}. {}", place + 1, species_name(&lane.playback.creature)),
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
            "Live distance",
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
            ui.label("Run your first generation to build distance curves and creature replays.");
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
                "Generation {} · {} creatures tried · {} ways of moving kept",
                stats.generation,
                number(stats.population),
                number(stats.archive_cells),
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
            self.select(c, stats.config);
            self.tab = Tab::Overview;
        }
    }
    /// The closed-by-default drawer with search and machine numbers, and the
    /// step-by-step run buttons developers use.
    fn diagnostics(&self, ui: &mut egui::Ui, s: &Snapshot) {
        let mut frames: Vec<_> = self.frame_times.iter().copied().collect();
        frames.sort_by(f32::total_cmp);
        let p95 = frames.get(frames.len() * 95 / 100).copied().unwrap_or(0.);
        ui.small(format!(
            "{} · seed {} · stage: {} · {} / {} evaluated · {} in checks",
            s.gpu,
            s.config.seed,
            s.stage.label(),
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
            if ui
                .add_enabled(!running, egui::Button::new("Guided step").small())
                .on_hover_text("Evaluate, then update the archive, then breed, pausing after each")
                .clicked()
            {
                self.worker.pause.store(false, Ordering::Relaxed);
                self.worker.send(Command::Next);
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
                                "Overview · Behavior archive · History · Race · Lineage",
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
                ui.heading("Tabs");
                for (name, why) in [
                    (
                        "Overview",
                        "The champion's replay (or the creature you picked), its playback controls, lineage and the best distance over time.",
                    ),
                    (
                        "Behavior archive",
                        "The best creature for every way of moving, as cards or as a map. Click a creature or a map cell to replay it.",
                    ),
                    (
                        "History & statistics",
                        "Per-generation curves, every new record with a replay, the session hall of fame, the mix of body types, and the distribution of distances.",
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
                                egui::Button::new("Create population")
                                    .fill(Color32::from_rgb(222, 241, 229)),
                            )
                            .clicked()
                        {
                            self.pause();
                            self.worker.send(Command::New(self.config.clone()));
                            self.initial = true;
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
        let next = self.worker.view.lock().unwrap().take();
        if let Some(mut next) = next {
            self.absorb_records(&next);
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
            } else if self
                .config_sent
                .is_none_or(|sent| sent.elapsed() > Duration::from_secs(2))
                && self
                    .snapshot
                    .as_ref()
                    .is_none_or(|old| old.epoch == next.epoch)
            {
                // The worker owns the world: seasons advance it, and a change
                // waits in `pending` until the next generation. The panel
                // shows the world the player asked for.
                self.config = next.pending.clone().unwrap_or_else(|| next.config.clone());
            }
            if let Some((c, cfg)) = next.preview.take() {
                // The worker picks the creature of a new game (a random one)
                // and of a loaded game (its best elite).
                let loaded = !next.history.is_empty();
                self.show_champion(c, cfg);
                self.champion_shown = loaded;
            }
            if let Some((c, cfg)) = next.selected.take() {
                // A creature the player clicked on the archive map.
                self.select(c, cfg);
                self.tab = Tab::Overview;
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
                if self.tab != Tab::Race {
                    self.restart_race();
                }
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
        if self.playing {
            let frame_dt = physics::dt();
            if frame_dt.is_finite() && frame_dt > 0.0 {
                let speed = self.speed;
                // Returns whether the replay reached its end and started over.
                let advance = |p: &mut Playback| {
                    p.accumulator = (p.accumulator + dt.clamp(0.0, 0.1) * speed).min(1.0);
                    let start = Instant::now();
                    let mut looped = false;
                    while p.accumulator >= frame_dt && start.elapsed() < Duration::from_millis(5) {
                        if p.tick >= p.last_frame() {
                            p.reset();
                            looped = true;
                        }
                        p.advance();
                        p.accumulator -= frame_dt;
                    }
                    looped
                };
                let mut looped = false;
                if let Some(p) = &mut self.playback {
                    looped = advance(p);
                    p.show_between();
                }
                if looped
                    && !self.pinned
                    && let Some((creature, config)) = self.next_champion.take()
                {
                    self.show_champion(creature, config);
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
        egui::Panel::top("top")
            .exact_size(64.)
            .frame(egui::Frame::new().fill(theme.panel).inner_margin(12))
            .show(ui, |ui| self.top(ui));
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
                    match &self.shown_message {
                        Some((message, _)) => ui.label(message),
                        None => ui.label(&s.status),
                    };
                    if self.shown_message.is_some() {
                        ui.ctx().request_repaint_after(Duration::from_millis(500));
                    }
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
            .default_size(300.)
            .min_size(260.)
            .max_size(440.)
            .resizable(true)
            .frame(egui::Frame::new().fill(theme.panel).inner_margin(16))
            .show(ui, |ui| self.controls(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme.canvas).inner_margin(20))
            .show(ui, |ui| {
                let before = self.tab;
                ui.horizontal(|ui| {
                    for (tab, label) in [
                        (Tab::Overview, "Overview"),
                        (Tab::Population, "Behavior archive"),
                        (Tab::History, "History & statistics"),
                        (Tab::Race, "Race"),
                        (Tab::Lineage, "Lineage"),
                    ] {
                        ui.selectable_value(&mut self.tab, tab, RichText::new(label).size(15.));
                    }
                });
                if self.tab == Tab::Race && before != Tab::Race {
                    self.restart_race();
                }
                ui.add_space(8.);
                match self.tab {
                    Tab::Overview => {
                        self.metrics(ui);
                        ui.add_space(8.);
                        // The chart keeps a fixed height below the replay and
                        // its controls; the replay takes the rest.
                        const CHART: f32 = 160.;
                        const REPLAY_CONTROLS: f32 = 120.;
                        self.viewport(
                            ui,
                            (ui.available_height() - CHART - REPLAY_CONTROLS).max(180.),
                        );
                        ui.add_space(6.);
                        // The chart and the event feed share the bottom row.
                        let height = (ui.available_height() - 34.).max(80.);
                        let width = ui.available_width();
                        ui.horizontal_top(|ui| {
                            ui.allocate_ui(Vec2::new(width * 0.62, height + 34.), |ui| {
                                self.trend(ui, height);
                            });
                            ui.allocate_ui(Vec2::new(ui.available_width(), height + 34.), |ui| {
                                self.feed(ui, height + 10.);
                            });
                        });
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
/// How a creature came to be, in the words the lineage uses.
fn origin_words(emitter: crate::qd::Emitter) -> &'static str {
    match emitter {
        crate::qd::Emitter::Cma => "fine-tuned from a parent",
        crate::qd::Emitter::Structural => "reshaped from a parent",
        crate::qd::Emitter::Novelty => "exploring a new way of moving",
        crate::qd::Emitter::Restart => "as a new random body",
    }
}
/// The next season step while seasons are on: "Next change at generation 60:
/// Wind to Breeze". The worker applies step `season_step` when a generation
/// that is a multiple of the interval begins.
fn season_forecast(config: &Config, generation: u32) -> Option<String> {
    let interval = *crate::environment::SEASON_INTERVALS.get(usize::from(config.seasons))?;
    if interval == 0 {
        return None;
    }
    let at = (generation / interval + 1) * interval;
    let rotation = crate::environment::season_rotation();
    let &(index, level) = rotation.get(usize::from(config.season_step) % rotation.len())?;
    let effect = &crate::environment::EFFECTS[index];
    Some(format!(
        "Next change at generation {at}: {} to {}",
        effect.name, effect.levels[level]
    ))
}
/// Whether every effect except the seasons schedule sits at its calm level.
fn world_is_calm(config: &Config) -> bool {
    crate::environment::EFFECTS
        .iter()
        .filter(|effect| effect.name != "Seasons")
        .all(|effect| effect.level(config) == effect.calm)
}
/// The world in a few words: "Calm world", or the effects away from calm,
/// such as "Ground: Rough, 8 cm · Hurdles: Low".
fn world_summary(config: &Config) -> String {
    let parts: Vec<String> = crate::environment::EFFECTS
        .iter()
        .filter(|effect| effect.name != "Seasons")
        .filter(|effect| effect.level(config) != effect.calm)
        .map(|effect| format!("{}: {}", effect.name, effect.levels[effect.level(config)]))
        .collect();
    if parts.is_empty() {
        "Calm world".to_owned()
    } else {
        parts.join(" · ")
    }
}
/// One effect as its name and a button per level. The lit button is the
/// current level. Returns true when the player picked another level.
fn effect_row(
    ui: &mut egui::Ui,
    effect: &crate::environment::Effect,
    config: &mut Config,
    live: Option<&Config>,
    theme: Theme,
) -> bool {
    let level = effect.level(config);
    let away = level != effect.calm;
    let waiting = live.is_some_and(|live| effect.level(live) != level);
    let color = if waiting {
        AMBER
    } else if away {
        theme.accent
    } else {
        theme.ink
    };
    let name = ui.label(RichText::new(effect.name).color(color));
    if let (true, Some(live)) = (waiting, live) {
        name.on_hover_text(format!(
            "Now {}. {} from the next generation.",
            effect.levels[effect.level(live)],
            effect.levels[level]
        ));
    } else {
        name.on_hover_text(effect.why);
    }
    let mut picked = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(3., 3.);
        ui.spacing_mut().button_padding = Vec2::new(6., 2.);
        for (i, text) in effect.levels.iter().enumerate() {
            let short = text.split(',').next().unwrap_or(text);
            let hover = if i == effect.calm {
                format!("{text}. The calm world.")
            } else {
                format!("{text}. {}", effect.why)
            };
            if ui
                .selectable_label(i == level, RichText::new(short).small())
                .on_hover_text(hover)
                .clicked()
                && i != level
            {
                picked = Some(i);
            }
        }
    });
    if let Some(i) = picked {
        effect.set_level(config, i);
    }
    picked.is_some()
}
/// A short duration for people: "8 s", "3 min", "2 h".
fn seconds_text(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        "a while".to_owned()
    } else if seconds < 90.0 {
        format!("{:.0} s", seconds.max(1.0))
    } else if seconds < 90.0 * 60.0 {
        format!("{:.0} min", seconds / 60.0)
    } else {
        format!("{:.0} h", seconds / 3600.0)
    }
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
    marks: &FrameMarks,
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
        let contraction = if marks.fallen {
            0.
        } else {
            1. - ((physics::target(m, marks.time) - m.short) / (m.long - m.short).max(1e-5))
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
    for (i, n) in nodes.iter().enumerate() {
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
        if marks.contact.get(i).copied().unwrap_or(false) {
            p.circle_stroke(center, r + 2.5, Stroke::new(2., TOUCHDOWN));
        }
        if marks.broken.get(i).copied().unwrap_or(false) {
            p.circle_stroke(center, r + 2.5, Stroke::new(2., FALLEN));
            draw_break_mark(p, center, r);
        }
    }
    // The head (node 0) looks ahead with one eye.
    if let Some(head) = nodes.first() {
        let center = position(head);
        let r = (head.radius * scale).max(2.);
        if marks.fallen {
            p.circle_stroke(center, r + 1.5, Stroke::new(2., FALLEN));
            draw_break_mark(p, center, r);
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
    draw_creature(p, &nodes, c, origin, scale, &FrameMarks::default());
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
        let dark = gif_color(Color32::from_rgb(10, 15, 19));
        let origin_x = self.camera.origin_x(pose_center_x(self.nodes, positions));
        let at = |position: [f32; 2]| self.camera.screen(origin_x, position);
        for pixel in buffer.pixels_mut() {
            *pixel = gif_color(VIEWPORT);
        }
        // A meter grid; it scrolls with the follow camera, so motion reads even
        // when the creature holds its screen position.
        let right = origin_x + GIF_WIDTH as f32 / self.camera.scale;
        let grid = gif_color(CARD_BORDER);
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
            let ground = gif_color(GROUND);
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
            gif_line(
                buffer,
                a,
                b,
                half,
                gif_color(Color32::from_rgb(192, 205, 187)),
            );
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
            let color =
                egui::ecolor::Hsva::new(0.44 - 0.07 * n.friction, 0.3 + 0.4 * n.friction, 0.95, 1.);
            gif_disc(buffer, center, r + 1.5, dark);
            gif_disc(buffer, center, r, gif_color(Color32::from(color)));
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
                gif_color(Color32::WHITE),
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
    nodes: &[Node],
    frames: &[Vec<[f32; 2]>],
    ticks: &[u32],
    fall: Option<(u32, f32)>,
    path: &std::path::Path,
) -> anyhow::Result<usize> {
    let camera = GifCamera::fit(nodes, frames);
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
    let joints = physics::joints(&creature.nodes, &creature.bones);
    let mut buffer = RgbaImage::new(GIF_WIDTH, GIF_HEIGHT);
    let mut contact = vec![false; nodes.len()];
    let mut broken = vec![false; nodes.len()];
    let mut written = 0usize;
    for &tick in ticks {
        let Some(frame) = frames.get(tick as usize) else {
            break;
        };
        node_contact(nodes, frame, creature, config, &mut contact);
        broken_nodes(creature, frame, &joints, &mut broken);
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
        &playback.nodes,
        &playback.frames,
        &ticks,
        playback.fall,
        path,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::{Bone, Muscle, NodeGene};
    #[test]
    fn the_follow_camera_ignores_the_stride_and_keeps_up_with_the_walk() {
        // One node walking at 2 m/s, swinging 0.3 m back and forth once per
        // second.
        let rate = physics::rate() as f32;
        let frames: Vec<Vec<[f32; 2]>> = (0..600)
            .map(|i| {
                let t = i as f32 / rate;
                vec![[2.0 * t + 0.3 * (std::f32::consts::TAU * t).sin(), 0.5]]
            })
            .collect();
        let nodes = vec![Node {
            mass: 1.0,
            ..Node::default()
        }];
        let track = camera_track(&frames, &nodes);
        let margin = (CAMERA_WINDOW * rate) as usize;
        for (i, x) in track.iter().enumerate().skip(margin).take(600 - 2 * margin) {
            let walk = 2.0 * i as f32 / rate;
            assert!(
                (x - walk).abs() < 0.02,
                "frame {i}: camera {x}, walk {walk}"
            );
        }
    }
    #[test]
    fn zoom_scales_a_plot_range_around_its_center() {
        assert_eq!(scaled_range(10.0..=20.0, 0.5), 12.5..=17.5);
        assert_eq!(scaled_range(10.0..=20.0, 2.0), 5.0..=25.0);
        assert_eq!(scaled_range(-4.0..=4.0, 1.0), -4.0..=4.0);
    }
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
    fn the_season_forecast_names_the_next_step() {
        let mut config = Config {
            seasons: 2,
            ..Config::default()
        };
        assert_eq!(
            season_forecast(&config, 7).as_deref(),
            Some("Next change at generation 10: Wind to Breeze")
        );
        assert_eq!(
            season_forecast(&config, 10).as_deref(),
            Some("Next change at generation 20: Wind to Breeze")
        );
        config.seasons = 0;
        assert_eq!(season_forecast(&config, 7), None);
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
    #[test]
    fn gif_export_encodes_three_synthetic_frames() {
        use image::AnimationDecoder;
        use image::codecs::gif::GifDecoder;
        let creature = test_creature();
        let config = Config::default();
        let nodes = physics::nodes(&creature);
        let frames: Vec<Vec<[f32; 2]>> = vec![
            vec![[0.0, 0.10], [0.5, 0.10], [1.0, 0.10]],
            vec![[0.1, 0.20], [0.6, 0.20], [1.1, 0.20]],
            vec![[0.2, 0.10], [0.7, 0.10], [1.2, 0.10]],
        ];
        let dir = std::env::temp_dir().join(format!("evolution-gif-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("creature.gif");
        let written =
            write_creature_gif(&creature, &config, &nodes, &frames, &[0, 1, 2], None, &path)
                .unwrap();
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
    #[test]
    fn gif_export_samples_a_recorded_trial() {
        let playback = Playback::new(test_creature(), Config::default());
        let dir = std::env::temp_dir().join(format!("evolution-gif-trial-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trial.gif");
        let written = export_creature_gif(&playback, &path).unwrap();
        assert!((3..=GIF_MAX_FRAMES).contains(&written));
        image::open(&path).unwrap();
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
