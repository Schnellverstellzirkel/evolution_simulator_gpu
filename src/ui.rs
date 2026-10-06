mod controls;
mod diagnostics;
mod feed;
mod header;
mod history;
mod islands;
mod lineage;
mod loading;
mod overview;
mod playback;
mod population;
mod race;
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
            BONE, EYE, FALLEN, GROUND_EDGE, GROUND_TOP, MUSCLE_ACTIVE, MUSCLE_REST, ORGAN, OUTLINE,
            SKY_HORIZON, SKY_TOP, TOUCHDOWN,
        },
    },
    worker::{Command, EventKind, Snapshot, Worker},
};
use controls::{world_summary, worlds_match};
use diagnostics::directory_bytes;
use eframe::egui::{self, Align2, Color32, Pos2, Rect, RichText, Stroke, Vec2};
use image::{
    Delay as GifDelay, Frame as GifFrame, Rgba, RgbaImage,
    codecs::gif::{GifEncoder, Repeat as GifRepeat},
};
use playback::{Playback, broken_nodes, node_contact};
use population::{ArchiveView, CardFilter};
use race::RaceLane;
use scene::node_color;
pub(crate) use scene::thumbnail;
use std::{
    path::PathBuf,
    sync::{Arc, atomic::Ordering, mpsc},
    time::{Duration, Instant},
};
use text::{ago, file_size, number, seconds_text, species_name};
use viewport::DEFAULT_CAMERA_ZOOM;
use widgets::{color_dot, mix_color};
/// The spacing scale: every gap, margin and padding is one of these.
const GAP_S: f32 = 4.0;
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
    /// Runs when the player opens a tab: a fresh race, or the latest
    /// ranked archive for Ways of moving.
    fn opened_tab(&mut self, tab: Tab) {
        match tab {
            Tab::Race => self.restart_race(),
            Tab::Population => self.request_cards(),
            _ => {}
        }
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
