//! The egui interface. This file holds `launch`, `App` (the state of the whole
//! window), its constructor and the frame loop (`ui`). The modules below hold
//! the rest, and most of them add an `impl App` block for their part:
//!
//! - The tabs: `overview` (metrics, trend chart and histogram), `feed` (the
//!   event feed of the Overview tab), `records` (what counts as a record),
//!   `population` (Ways of moving: cards and the archive map), `islands` (the
//!   island view of Ways of moving), `history`, `race` and `lineage`.
//! - The replay: `viewport` (the replay view and the creature it shows),
//!   `playback` (one creature's recorded replay) and `scene` (painting a
//!   creature).
//! - The window: `header` (top bar and Help), `controls` (the side panel),
//!   `diagnostics` (the diagnostics drawer and the developer pause bar),
//!   `dialogs` (the File menu's dialogs), `export` (GIF and screenshot files)
//!   and `loading` (the loading screen).
//! - Shared helpers: `text` (words and numbers) and `widgets`. `test_support`
//!   holds the creature of the unit tests.
mod controls;
mod diagnostics;
mod dialogs;
mod export;
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
    physics,
    theme::{GAP_L, GAP_M, Theme, apply_style},
    worker::{Command, Snapshot, Worker},
};
use controls::worlds_match;
use diagnostics::directory_bytes;
use dialogs::SaveEntry;
use eframe::egui::{self, Pos2, Rect, RichText, Stroke, Vec2};
use export::save_screenshot;
use playback::Playback;
use population::{ArchiveView, CardFilter};
use race::RaceLane;
/// Draws a creature as a small thumbnail. `schematic.rs` uses it too.
pub(crate) use scene::thumbnail;
use std::{
    path::PathBuf,
    sync::{Arc, atomic::Ordering, mpsc},
    time::{Duration, Instant},
};
use viewport::DEFAULT_CAMERA_ZOOM;
use widgets::color_dot;
/// The small gap of the spacing scale, in points. `GAP_M` and `GAP_L` in
/// `theme` are the medium and large gaps, and the `ui` modules space their
/// blocks with the three.
const GAP_S: f32 = 4.0;
/// How long the UI's own messages hold the status line.
const MESSAGE_SECONDS: f32 = 8.0;
/// Generations between autosaves when the player turns autosave on, and in
/// an unattended run.
pub(crate) const AUTOSAVE_INTERVAL: u32 = 10;
/// Marks a screenshot request from the File menu or from
/// `EVOLUTION_CAPTURE_EVERY`, so its reply is not taken for the reply to the
/// capture hook of `EVOLUTION_SMOKE_CAPTURE`.
struct ScreenshotRequest;
/// How often the disk usage of `runs/` is measured again.
const RUNS_REFRESH: Duration = Duration::from_secs(5);
/// Opens the game window and runs it until the player closes it. `adapter_name`
/// is the `--gpu` text, which names the CUDA device that scores creatures. The
/// worker opens that device while the window shows its loading screen. It
/// returns an error if the window fails to start or run.
pub fn launch(adapter_name: &str) -> anyhow::Result<()> {
    // The UI thread asks for a 1 ms slice, so a frame preempts the breeding
    // threads when it wakes (`threads::short_slice`).
    crate::threads::short_slice();
    let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = wgpu::Backends::VULKAN;
    // The window draws on the first of these Vulkan adapters that can present:
    // the one whose name contains `EVOLUTION_RENDER_GPU`, the GPU the desktop
    // compositor uses, the one whose name contains `adapter_name`, then any.
    // The compositor's GPU needs no cross-GPU import for each frame, and when
    // that is the integrated GPU the discrete GPU is left entirely to
    // evolution.
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
                // A run with any `EVOLUTION_SMOKE_*` variable is an agent's
                // screenshot run in a real window on the owner's desktop. The
                // title says so.
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
/// The five tabs, in the order of the tab strip and of the keys 1 to 5.
/// `Population` is the tab called Ways of moving.
#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Overview,
    Population,
    History,
    Race,
    Lineage,
}
/// The state of the whole window: the worker's handle and newest snapshot, the
/// replay on screen, and what each tab, panel and dialog has open or selected.
struct App {
    worker: Worker,
    /// The newest snapshot the worker published. It is `None` before the first.
    snapshot: Option<Snapshot>,
    /// The settings the panels show and edit, which are the world the player
    /// asked for. The worker gets them with `Command::Configure`, and
    /// `absorb_snapshot` takes the worker's own back.
    config: Config,
    /// The replay on screen: the champion or the creature the player picked.
    playback: Option<Playback>,
    /// The replay being recorded for `playback`, and when it was asked for.
    replay_wait: Option<(mpsc::Receiver<Playback>, Instant)>,
    /// Seconds each replay took to appear after it was asked for, for the
    /// benchmark report.
    replay_seconds: Vec<f32>,
    /// When the last replay was requested for benchmarking.
    bench_last_replay: Instant,
    ctx: egui::Context,
    tab: Tab,
    /// Replay speed as a multiple of real time (the Speed menu).
    speed: f32,
    playing: bool,
    /// The replay view's zoom in pixels per meter.
    zoom: f32,
    /// True once the user zoomed by hand; until then the zoom fits the creature.
    zoom_user: bool,
    /// Replay view option, the Forces box: draw muscle forces and ground pushes.
    show_forces: bool,
    /// The replay camera in meters: x along the ground and y up.
    camera: [f32; 2],
    /// The camera follows the creature. Dragging the view turns it off.
    follow: bool,
    /// The row of the history that the History tab shows (the Generation
    /// slider).
    history_index: usize,
    /// The History tab keeps `history_index` on the newest row (the Follow
    /// latest box).
    history_latest: bool,
    /// The title of the file dialog that is open, such as "Save experiment",
    /// which also labels its button. It is `None` when no file dialog shows.
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
    /// The path in the text box of the Open window and of the file dialogs.
    file_path: String,
    /// A message for the status line. The status line takes it on the next
    /// frame and keeps it in `shown_message`.
    message: Option<String>,
    /// The message on the status line and when it first showed.
    shown_message: Option<(String, Instant)>,
    /// Since when the GPU has waited for a kernel (`cuda_engine::compiling_world`).
    compiling_since: Option<Instant>,
    /// The New experiment window is open.
    new_dialog: bool,
    /// When the UI last sent a settings change to the worker.
    config_sent: Option<Instant>,
    /// Worlds before each change of physics made in the World panel, newest
    /// last and at most 20, for its Undo last change button.
    world_undo: Vec<Config>,
    last_frame: Instant,
    /// Seconds of the last 240 frames, for the diagnostics drawer and the
    /// capture report.
    frame_times: std::collections::VecDeque<f32>,
    /// The Diagnostics drawer under the status line is open.
    show_perf: bool,
    /// The UI scale of the View menu, which is egui's zoom factor.
    /// `EVOLUTION_SMOKE_ZOOM` sets the first one.
    ui_scale: f32,
    /// Set at the start, by Create population and by Open. The next snapshot
    /// of a newly started or opened game then replaces `config` with that
    /// game's settings.
    initial: bool,
    /// A screenshot run or an unattended run has yet to start evolving, which
    /// `frame_housekeeping` does 250 ms after the window opened.
    smoke_start_pending: bool,
    /// The preset (an index into `environment::PRESETS`) that a screenshot run
    /// applies 4 s after the window opened. `None` once applied.
    smoke_preset: Option<usize>,
    /// When the window opened.
    started: Instant,
    /// The player closed the loading card; compiling goes on in a corner
    /// note.
    loading_hidden: bool,
    /// A screenshot run shows made-up loading jobs (`crate::loading::demo`).
    loading_demo: bool,
    /// The capture hook has asked for its screenshot.
    capture_requested: bool,
    /// Where the capture hook saves its screenshot (`EVOLUTION_SMOKE_CAPTURE`).
    /// The window closes after that.
    capture_path: Option<String>,
    /// Ancestors of the selected creature, newest first.
    lineage: Vec<crate::worker::LineageStep>,
    /// Creature whose ancestors the UI has already asked the worker for.
    lineage_requested: Option<u64>,
    /// A lineage request is in flight.
    lineage_pending: bool,
    /// The player picked the creature on screen, so the replay stops
    /// following the champion until they go back to it.
    pinned: bool,
    /// The creature on screen is the champion of a finished generation (or
    /// the best elite of a loaded game), not a random first creature.
    champion_shown: bool,
    /// Behavior archive map filters; None shows every bin.
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
    /// Current archive view mode (Cards, Map, or Islands).
    archive_view: ArchiveView,
    /// Top archived elites racing side by side.
    race: Vec<RaceLane>,
    /// The race lanes are not built yet. They are built from the ranked
    /// archive, or from the player's picks, when that data is there.
    race_pending: bool,
    /// The meter at the left edge of the race lanes. It eases toward the
    /// leader.
    race_camera: f32,
    /// Creatures the player sent to the race, oldest first, with their worlds.
    race_picks: Vec<(Creature, Config)>,
    /// Native benchmark frame intervals in seconds, while the benchmark
    /// measures.
    bench_frames: Vec<f32>,
    /// The time each frame in `bench_frames` began.
    bench_frame_starts: Vec<Instant>,
    /// CPU time of each measured frame without the vsync wait (eframe's
    /// `cpu_usage`).
    bench_work: Vec<f32>,
    /// The UI thread's major page faults when the measured window began.
    bench_faults: Option<u64>,
    /// When the worker was last probed for the benchmark.
    bench_last_ping: Instant,
    /// How many probes the benchmark has sent.
    bench_pings: u64,
    show_help: bool,
    /// The "How evolution works" window (`crate::schematic::show`).
    pub schematic_open: bool,
    /// Disk usage in bytes of the runs/ directory.
    runs_bytes: u64,
    runs_checked: Instant,
    /// A screenshot is wanted. `request_screenshots` asks the window for it
    /// with a `ScreenshotRequest` and clears this.
    screenshot_pending: bool,
    /// A `ScreenshotRequest` is out and its reply has not come yet.
    screenshot_waiting: bool,
    /// Unattended runs: `EVOLUTION_CAPTURE_EVERY=<n>` saves the window to
    /// `runs/progress-gen<g>.png` every n generations.
    capture_every: Option<u32>,
    /// The generation of the capture in flight.
    capture_generation: Option<u32>,
    /// The generation of the last capture taken.
    captured_generation: u32,
}
impl App {
    fn new(cc: &eframe::CreationContext<'_>, worker: Worker) -> Self {
        let ctx = &cc.egui_ctx;
        // `EVOLUTION_SMOKE_LOADING` makes a screenshot run of the loading
        // screen with made-up jobs, and it needs no GPU. The value `toast`
        // shows the corner note instead of the card.
        let demo = std::env::var("EVOLUTION_SMOKE_LOADING").ok();
        let loading_demo = demo.is_some();
        if loading_demo {
            crate::loading::demo();
        }
        let mut initial_config = Config::default();
        // `EVOLUTION_SMOKE_POPULATION=<n>` starts a run of n creatures with a
        // fixed seed.
        if let Ok(n) = std::env::var("EVOLUTION_SMOKE_POPULATION")
            && let Ok(n) = n.parse()
        {
            initial_config.population = n;
            initial_config.random_seed = false;
            // `EVOLUTION_BENCH_NO_AUTOSAVE` keeps autosave off in a benchmark.
            // The default settings have it off already, and the worker checks
            // the variable too.
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
        // Screenshot runs: `EVOLUTION_SMOKE_WORLD="Wind=2,Mud=3"` starts the
        // game with those effect levels. `EVOLUTION_AUTOSTART` takes the same
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
        // The game's fonts and style. Its art decodes in the background.
        crate::assets::install_fonts(ctx);
        crate::assets::preload(ctx);
        apply_style(ctx);
        // Developer screenshots: `EVOLUTION_SMOKE_ZOOM=0.75` lays a 1440 px
        // window out like a 1920 px one. It takes 0.5 to 2.
        let smoke_zoom = std::env::var("EVOLUTION_SMOKE_ZOOM")
            .ok()
            .and_then(|zoom| zoom.parse::<f32>().ok())
            .filter(|zoom| (0.5..=2.0).contains(zoom));
        if let Some(zoom) = smoke_zoom {
            ctx.set_zoom_factor(zoom);
        }
        // `EVOLUTION_SMOKE_TAB` opens `history`, `population`, `race` or
        // `lineage`. `map` and `islands` open Ways of moving in that view.
        let smoke_tab = std::env::var("EVOLUTION_SMOKE_TAB").unwrap_or_default();
        // A screenshot run opens the save that `EVOLUTION_SMOKE_CHECKPOINT`
        // names, or else starts a new game.
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
    /// Draws the loading card while the devices open or fail to open, while
    /// the starting worlds' kernels compile and while the first generation
    /// waits for the kernels of its world. The player can close the card of a
    /// kernel wait. Without a card, a corner note shows the compiles that
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
    /// Whether evolution is running and not paused.
    fn active(&self) -> bool {
        self.snapshot.as_ref().is_some_and(|s| s.running)
            && !self.worker.pause.load(Ordering::Relaxed)
    }
    /// Clears the pause and sends `Command::Run`. A run that is `continuous`
    /// goes on generation after generation. A run that is not, or that is
    /// `guided`, stops after one generation.
    fn run(&mut self, continuous: bool, guided: bool) {
        self.worker.pause.store(false, Ordering::Relaxed);
        self.worker.send(Command::Run { continuous, guided });
    }
    /// Sets the pause, so the worker stops at once, and sends `Command::Pause`.
    fn pause(&self) {
        self.worker.pause.store(true, Ordering::Relaxed);
        self.worker.send(Command::Pause);
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
}
impl App {
    /// Once a frame, the checks that need no input: the disk usage of `runs/`,
    /// the delayed start of a screenshot run and its preset.
    fn frame_housekeeping(&mut self, now: Instant) {
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
        // Screenshot runs: `EVOLUTION_SMOKE_PRESET=<number>` applies that
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
    }
    /// Keeps the last 240 frame times for the diagnostics. While a native
    /// benchmark measures, it also records the frame, asks for the champion's
    /// replay when `EVOLUTION_BENCH_REPLAY` is set, and probes the worker every
    /// 100 ms to time how long a control waits.
    fn record_frame(&mut self, dt: f32, now: Instant) {
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
            // `EVOLUTION_BENCH_REPLAY`: ask for the champion's replay every
            // 6 s and time how long it takes to appear.
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
                // environment button, when `EVOLUTION_BENCH_SETTINGS_PROBE`
                // is set.
                if self.bench_pings.is_multiple_of(10)
                    && std::env::var_os("EVOLUTION_BENCH_SETTINGS_PROBE").is_some()
                {
                    self.worker.send(Command::ConfigureProbe(now));
                } else {
                    self.worker.send(Command::Ping(now));
                }
            }
        }
    }
    /// Takes the replay that the replay thread recorded (`set_preview`) once
    /// it is ready, and keeps how long it took for the benchmark report.
    fn receive_replay(&mut self) {
        if let Some((rx, asked)) = &self.replay_wait
            && let Ok(ready) = rx.try_recv()
        {
            self.replay_seconds.push(asked.elapsed().as_secs_f32());
            self.playback = Some(ready);
            self.replay_wait = None;
        }
    }
    /// Takes the worker's newest snapshot, if it published one since the last
    /// frame, and acts on what it carries: the world, the creature to preview,
    /// the event log, the ranked archive, a selected creature and a lineage.
    fn absorb_snapshot(&mut self) {
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
                // A click holds the panel for 2 s once the worker's world shows
                // it, and for 15 s while it does not. A snapshot published
                // before the worker read the click must not put the panel back
                // (autochange would flip to Off).
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
                // The worker owns the world: autochange advances it, and a
                // change waits in `pending` until the next generation. The
                // panel shows the world the player asked for.
                self.config = next.pending.clone().unwrap_or_else(|| next.config.clone());
            }
            if let Some((c, cfg)) = next.preview.take() {
                // The worker picks the creature of a new game (a random one)
                // and of a loaded game (its best elite). Neither is known to
                // be the champion, so `follow_champion` below puts the
                // champion on screen at once when it is another creature.
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
                // A creature the player clicked on the archive map. It plays
                // in the replay docked beside the map.
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
    }
    /// Asks the worker for the archive map table and the lineage when the tab
    /// shows them.
    fn request_tab_data(&mut self) {
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
    }
    /// Reads the keyboard shortcuts unless a text field has the keyboard. The
    /// keys 1 to 5 pick a tab, F1 or ? toggles Help, Space evolves or pauses,
    /// K plays or pauses the replay, the arrow keys step it by one frame, and
    /// Ctrl+S opens the Save dialog.
    fn handle_keys(&mut self, ctx: &egui::Context) {
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
    }
    /// Advances the replay by the time of the frame `dt` at the chosen speed,
    /// and the race lanes too on the Race tab. Then the camera follows the
    /// creature.
    fn advance_replays(&mut self, dt: f32) {
        if self.playing {
            // The replay time of one recorded frame, which is one physics step.
            let step_dt = physics::dt();
            if step_dt.is_finite() && step_dt > 0.0 {
                let speed = self.speed;
                // Steps a replay by one recorded frame for each `step_dt` of
                // replay time it has gathered, and starts it over after its
                // last frame. A frame counts as 0.1 s at most, and the
                // stepping stops after 5 ms of work.
                let advance = |p: &mut Playback| {
                    p.accumulator = (p.accumulator + dt.clamp(0.0, 0.1) * speed).min(1.0);
                    let start = Instant::now();
                    while p.accumulator >= step_dt && start.elapsed() < Duration::from_millis(5) {
                        if p.tick >= p.last_frame() {
                            p.reset();
                        }
                        p.advance();
                        p.accumulator -= step_dt;
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
    }
    /// The top bar: dark wood with a mustard stripe along its lower edge.
    fn top_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("top")
            .exact_size(68.)
            .frame(
                egui::Frame::new()
                    .fill(crate::theme::poster::WOOD_DARK)
                    .inner_margin(egui::Margin::symmetric(GAP_L as i8, 15)),
            )
            .show(ui, |ui| {
                // A mustard stripe along the lower edge of the bar, like the
                // poster's border.
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
    }
    /// The status line at the bottom. It shows the UI's message, else the
    /// kernel compile note, else the worker's status. Then come the error, the
    /// evaluation rate and the Diagnostics button, and the drawer opens under
    /// the line.
    fn status_panel(&mut self, ui: &mut egui::Ui, theme: Theme) {
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
                    // wait for them. Waits under 0.3 s are loads of kernels
                    // that are ready, not worth a message.
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
    }
    /// The tab strip and the tab shown below it.
    fn central_panel(&mut self, ui: &mut egui::Ui, theme: Theme) {
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme.canvas)
                    .inner_margin(GAP_L as i8),
            )
            .show(ui, |ui| {
                self.tab_strip(ui, theme);
                match self.tab {
                    Tab::Overview => self.overview_tab(ui),
                    Tab::Population => self.population_tab(ui),
                    Tab::History => {
                        egui::ScrollArea::vertical().show(ui, |ui| self.history(ui));
                    }
                    Tab::Race => self.race_view(ui),
                    Tab::Lineage => self.lineage_view(ui),
                }
            });
    }
    /// The five tab buttons and the rule under them.
    fn tab_strip(&mut self, ui: &mut egui::Ui, theme: Theme) {
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
                if crate::theme::tab(ui, self.tab == tab, &(key + 1).to_string(), label, theme)
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
    }
    /// Overview: metrics, the replay, the trend chart and the event feed.
    fn overview_tab(&mut self, ui: &mut egui::Ui) {
        self.metrics(ui);
        ui.add_space(GAP_M);
        // The chart row keeps a fixed height below the replay and its
        // controls, and the replay takes the rest, at least 180 points.
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
            ui.allocate_ui_with_layout(Vec2::new(width * 0.62, height + 34.), down, |ui| {
                self.trend(ui, height)
            });
            ui.allocate_ui_with_layout(Vec2::new(ui.available_width(), height + 34.), down, |ui| {
                self.feed(ui, height + 10.)
            });
        });
    }
    /// Ways of moving: the archive on the left and the replay docked on the
    /// right, so browsing never leaves the tab.
    fn population_tab(&mut self, ui: &mut egui::Ui) {
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
    /// Asks the window for the screenshots a run wants: the periodic ones of
    /// an unattended run, the one the File menu asked for, and the one of the
    /// capture hook of a screenshot run, which first holds the replay at
    /// `EVOLUTION_SMOKE_SEEK` seconds.
    fn request_screenshots(&mut self, ctx: &egui::Context) {
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
        // A screenshot wanted by the File menu or by `capture_every`: ask the
        // window for one frame. `handle_screenshot_events` saves it as PNG.
        if self.screenshot_pending {
            self.screenshot_pending = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(
                ScreenshotRequest,
            )));
        }
        // Developer screenshots: `EVOLUTION_SMOKE_SEEK=<seconds>` holds the
        // replay at that time.
        if self.capture_path.is_some()
            && let Some(seconds) = std::env::var("EVOLUTION_SMOKE_SEEK")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
            && let Some(p) = self.playback.as_mut()
        {
            p.seek((seconds * physics::rate() as f32) as u32);
            self.playing = false;
        }
        // The capture hook, for repeatable native rendering and performance
        // checks: `EVOLUTION_SMOKE_CAPTURE` names the file, and the capture
        // comes once `smoke_capture_delay` has passed since the window opened.
        if self.capture_path.is_some()
            && self.started.elapsed() > smoke_capture_delay()
            && !self.capture_requested
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            self.capture_requested = true;
        }
    }
    /// Saves the screenshots egui sends back. The reply to a
    /// `ScreenshotRequest` is saved in `runs/`: as `progress-gen<g>.png` in an
    /// unattended run, or else under a new name that the status line reports.
    /// The reply to the capture hook is saved at `capture_path`, a report of
    /// the frame times is printed, and the window closes.
    fn handle_screenshot_events(&mut self, ctx: &egui::Context) {
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
    }
}
impl eframe::App for App {
    /// Prints the native benchmark report on stderr when a benchmark measured
    /// frames: the replay times, the frame times with and without the vsync
    /// wait and during breeding, and the major faults of the UI thread.
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
    /// Draws one frame. It first updates the state: the frame record, the
    /// replay, the worker's snapshot, the keys and the replay clock. Then it
    /// paints the backdrop, the panels, the dialogs and the loading card. It
    /// ends with the request for the next repaint and for screenshots.
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frame_housekeeping(now);
        self.record_frame(dt, now);
        self.receive_replay();
        self.absorb_snapshot();
        self.request_tab_data();
        self.handle_keys(&ctx);
        if self.tab != self.prev_tab {
            self.prev_tab = self.tab;
            self.opened_tab(self.tab);
        }
        self.advance_replays(dt);
        let theme = self.theme();
        // The blurred city behind everything, as the game world shows
        // behind the Half-Life 2 menus; the panels are dark glass over it.
        crate::theme::backdrop(ui.painter(), ui.ctx().content_rect());
        self.top_panel(ui);
        self.dev_pause_bar(ui);
        self.status_panel(ui, theme);
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
        self.central_panel(ui, theme);
        self.dialogs(&ctx);
        self.help_window(&ctx);
        self.loading_screen(&ctx);
        crate::schematic::show(&ctx, self.snapshot.as_ref(), &mut self.schematic_open);
        if self.playing || self.active() {
            // Playback and live evolution redraw at the frame cap; the rest of
            // the GPU stays with evolution. `EVOLUTION_UI_FPS=0` follows vsync.
            match ui_frame_interval() {
                Some(interval) => ctx.request_repaint_after(interval),
                None => ctx.request_repaint(),
            }
        }
        self.request_screenshots(&ctx);
        self.handle_screenshot_events(&ctx);
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
/// `mutter-device-preferred-primary` by udev, otherwise the boot VGA card. It
/// is `None` when neither is found or the vendor cannot be read.
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
/// The time between repaints while a replay plays or evolution runs: 0.97 of
/// the interval of `EVOLUTION_UI_FPS` frames per second, 60 by default. It is
/// `None` when that is 0 or less, and then the window follows vsync.
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
/// How long a screenshot run waits before it captures: 8 s, or the 1 to 600 s
/// that `EVOLUTION_SMOKE_CAPTURE_AFTER` gives (developer diagnostic).
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
