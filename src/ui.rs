mod feed;
mod islands;
mod loading;
mod overview;
mod playback;
mod population;
mod records;
mod scene;
#[cfg(test)]
mod test_support;
mod text;
mod viewport;
mod widgets;

use crate::{
    config::Config,
    evolution::Creature,
    physics::{self, Node},
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
use image::{
    Delay as GifDelay, Frame as GifFrame, Rgba, RgbaImage,
    codecs::gif::{GifEncoder, Repeat as GifRepeat},
};
use playback::{FrameMarks, Playback, broken_nodes, node_contact};
use population::{ArchiveView, CardFilter};
use records::{live_record, world_records};
pub(crate) use scene::thumbnail;
use scene::{draw_creature, node_color};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::Ordering, mpsc},
    time::{Duration, Instant},
};
use text::{ago, file_size, number, seconds_text, species_name};
use viewport::{DEFAULT_CAMERA_ZOOM, fit_zoom};
use widgets::{color_dot, mix_color, species_color, speed_picker};
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
/// One archive elite running in the race view.
struct RaceLane {
    /// Why the creature runs: its archive rank, "champion" or "your pick".
    label: String,
    playback: Playback,
}
/// Creatures the player sends to the race with "Race it", at most this many
/// beside the champion.
const RACE_PICKS: usize = 4;
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
/// Whether two configs have every effect, including autochange, at the same level.
fn worlds_match(a: &Config, b: &Config) -> bool {
    crate::environment::EFFECTS
        .iter()
        .all(|effect| effect.level(a) == effect.level(b))
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
