use crate::{
    config::Config,
    evolution::{Creature, FAILED},
    gpu::{self, Gpu},
    physics::{self, Node},
    storage::{PERCENTILES, Stage, Stats},
    worker::{Command, Snapshot, Worker},
};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Legend, Line, Plot};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
const MINT: Color32 = Color32::from_rgb(22, 122, 91);
const AMBER: Color32 = Color32::from_rgb(164, 96, 24);
const MUTED: Color32 = Color32::from_rgb(105, 121, 113);
const INK: Color32 = Color32::from_rgb(40, 55, 48);
const PANEL: Color32 = Color32::from_rgb(255, 255, 252);
const CANVAS: Color32 = Color32::from_rgb(244, 247, 242);
const VIEWPORT: Color32 = Color32::from_rgb(236, 243, 239);
const CARD: Color32 = Color32::from_rgb(255, 255, 253);
const CARD_HOVER: Color32 = Color32::from_rgb(238, 247, 241);
const CARD_BORDER: Color32 = Color32::from_rgb(218, 229, 221);
const GROUND: Color32 = Color32::from_rgb(220, 234, 222);
const DEFAULT_CAMERA_ZOOM: f32 = 80.0;
pub fn launch(adapter_name: &str) -> anyhow::Result<()> {
    let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = wgpu::Backends::VULKAN;
    let name = adapter_name.to_lowercase();
    setup.native_adapter_selector = Some(Arc::new(move |adapters, surface| {
        adapters
            .iter()
            .find(|a| {
                a.get_info().name.to_lowercase().contains(&name)
                    && surface.is_none_or(|s| a.is_surface_supported(s))
            })
            .cloned()
            .ok_or_else(|| format!("No presentation-capable Vulkan GPU matching {name}"))
    }));
    setup.device_descriptor = Arc::new(gpu::descriptor);
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
    eframe::run_native(
        "Evolution Laboratory",
        options,
        Box::new(|cc| {
            let render = cc
                .wgpu_render_state
                .as_ref()
                .ok_or("GPU renderer unavailable")?;
            let gpu = if std::env::var_os("EVOLUTION_SHARED_DEVICE").is_some() {
                Gpu::from_device(
                    render.device.clone(),
                    render.queue.clone(),
                    render.adapter.get_info().name,
                )?
            } else {
                let (device, queue) = pollster::block_on(
                    render
                        .adapter
                        .request_device(&gpu::descriptor(&render.adapter)),
                )?;
                Gpu::from_device(device, queue, render.adapter.get_info().name)?
            };
            Ok(Box::new(App::new(cc, gpu)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
struct Playback {
    creature: Creature,
    config: Config,
    nodes: Vec<Node>,
    tick: u32,
    accumulator: f32,
}
impl Playback {
    fn new(creature: Creature, config: Config) -> Self {
        let mut nodes = physics::nodes(&creature);
        for tick in 0..physics::SETTLE {
            physics::step(
                &mut nodes,
                &creature.bones,
                &creature.muscles,
                &config,
                tick,
            );
        }
        physics::center(&mut nodes);
        Self {
            creature,
            config,
            nodes,
            tick: physics::SETTLE,
            accumulator: 0.0,
        }
    }
    fn reset(&mut self) {
        *self = Self::new(self.creature.clone(), self.config.clone());
    }
}
#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Overview,
    Population,
    History,
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
}
impl App {
    fn new(cc: &eframe::CreationContext<'_>, gpu: Gpu) -> Self {
        let ctx = &cc.egui_ctx;
        ctx.set_theme(egui::Theme::Light);
        let mut style = (*ctx.global_style()).clone();
        style.spacing.item_spacing = Vec2::new(10.0, 10.0);
        style.spacing.button_padding = Vec2::new(12.0, 8.0);
        style.visuals = egui::Visuals::light();
        style.visuals.override_text_color = Some(INK);
        style.visuals.weak_text_color = Some(MUTED);
        style.visuals.panel_fill = PANEL;
        style.visuals.window_fill = PANEL;
        style.visuals.extreme_bg_color = CANVAS;
        style.visuals.code_bg_color = VIEWPORT;
        style.visuals.faint_bg_color = CARD_BORDER;
        style.visuals.selection.bg_fill = Color32::from_rgb(219, 239, 227);
        style.visuals.selection.stroke = Stroke::new(1.0, MINT);
        style.visuals.hyperlink_color = MINT;
        style
            .text_styles
            .insert(egui::TextStyle::Body, FontId::proportional(14.0));
        style
            .text_styles
            .insert(egui::TextStyle::Heading, FontId::proportional(23.0));
        ctx.set_global_style(style);
        let worker = Worker::spawn(gpu, ctx.clone());
        let mut initial_config = Config::default();
        if let Ok(n) = std::env::var("EVOLUTION_SMOKE_POPULATION")
            && let Ok(n) = n.parse()
        {
            initial_config.population = n;
            initial_config.random_seed = false;
            initial_config.checkpoint_interval = 0;
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
        worker.send(Command::New(initial_config));
        let mut percentiles = [false; 29];
        percentiles[0] = true;
        percentiles[14] = true;
        percentiles[28] = true;
        Self {
            worker,
            snapshot: None,
            config: Config::default(),
            playback: None,
            tab: match std::env::var("EVOLUTION_SMOKE_TAB").as_deref() {
                Ok("history") => Tab::History,
                Ok("population") => Tab::Population,
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
            card_positions: Default::default(),
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
            "Export CSV" => "runs/statistics.csv",
            "Save preset" | "Load preset" => "presets/custom.json",
            _ => "runs/experiment.evo",
        }
        .into();
    }
    fn set_preview(&mut self, c: Creature, cfg: Config) {
        self.playback = Some(Playback::new(c, cfg));
        self.follow = true;
        self.zoom = DEFAULT_CAMERA_ZOOM;
        self.camera = [0.; 2];
    }
    fn top(&mut self, ui: &mut egui::Ui) {
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
            ui.label(RichText::new("CREATURE LABORATORY").size(10.).color(MUTED));
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
                ui.separator();
                if let Some(s) = &self.snapshot {
                    ui.label(
                        RichText::new(format!("GEN {:03}", s.generation))
                            .color(MINT)
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
        ui.add_space(10.);
        ui.label(RichText::new("EXPERIMENT").small().color(MUTED));
        ui.heading("Let life find a way.");
        ui.label(RichText::new("More kinds of life. Better walkers.").color(MUTED));
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
            ui.label(RichText::new(s.stage.label()).color(MINT));
            ui.add(
                egui::ProgressBar::new(s.evaluated as f32 / s.config.population as f32)
                    .text(format!(
                        "{} / {} evaluated",
                        number(s.evaluated),
                        number(s.config.population)
                    ))
                    .fill(MINT.gamma_multiply(0.7)),
            );
        }
        ui.separator();
        let before = self.config.clone();
        ui.label("Population");
        let population_changed = ui
            .add(
                egui::DragValue::new(&mut self.config.population)
                    .speed(100)
                    .range(2..=20_000_000),
            )
            .on_hover_text("Even population. Changing this starts a new experiment.")
            .changed();
        if population_changed && before.population < 100_000 && self.config.population >= 100_000 {
            self.config.throughput = true;
        }
        ui.horizontal(|ui| {
            for (label, n) in [
                ("1k", 1000),
                ("100k", 100000),
                ("1m", 1000000),
                ("3m", 3000000),
            ] {
                if ui.small_button(label).clicked() {
                    self.config.population = n;
                    if n >= 100_000 {
                        self.config.throughput = true;
                    }
                }
            }
        });
        ui.add_space(4.);
        ui.label("Mutation strength");
        ui.add(egui::Slider::new(&mut self.config.mutation, 0.0..=5.0).suffix("×"))
            .on_hover_text(
                "Scales continuous parameter edits. Structural edits, novelty search, and immigrant restarts still run when set to 0.",
            );
        ui.label("Trial duration");
        ui.add(egui::Slider::new(&mut self.config.duration, 1.0..=60.0).suffix(" s"));
        ui.checkbox(&mut self.advanced, "Advanced controls");
        if self.advanced {
            ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("Find a setting…"));
            let q = self.search.to_lowercase();
            if matches_search(&q, "physics gravity air damping friction ground") {
                egui::CollapsingHeader::new("Physics").default_open(true).show(ui,|ui|{
                    ui.add(egui::Slider::new(&mut self.config.gravity,0.0..=30.0).text("Gravity").suffix(" m/s²"));
                    ui.add(egui::Slider::new(&mut self.config.air_retention,0.0..=1.02).text("Air retention")).on_hover_text("Velocity retained per 1/60 s. 1 means no damping; above 1 adds energy.");
                    ui.add(egui::Slider::new(&mut self.config.ground_friction,0.0..=20.0).text("Ground friction"));
                    ui.checkbox(&mut self.config.ground, "Flat ground");
                });
            }
            if matches_search(&q, "body node size friction muscle limits") {
                egui::CollapsingHeader::new("Bodies & mutation bounds")
                    .default_open(true)
                    .show(ui, |ui| {
                        numeric(
                            ui,
                            "Min diameter (m)",
                            &mut self.config.min_size,
                            0.01..=1.0,
                            0.005,
                        );
                        numeric(
                            ui,
                            "Max diameter (m)",
                            &mut self.config.max_size,
                            0.01..=1.0,
                            0.005,
                        );
                        numeric(
                            ui,
                            "Min node friction",
                            &mut self.config.min_friction,
                            0.0..=1.0,
                            0.01,
                        );
                        numeric(
                            ui,
                            "Max node friction",
                            &mut self.config.max_friction,
                            0.0..=1.0,
                            0.01,
                        );
                        ui.horizontal(|ui| {
                            ui.label("Maximum nodes");
                            ui.add(egui::DragValue::new(&mut self.config.max_nodes).range(3..=64));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Maximum muscles");
                            ui.add(
                                egui::DragValue::new(&mut self.config.max_muscles).range(3..=256),
                            );
                        });
                    });
            }
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
                            .color(MUTED),
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
            ui.label(RichText::new("Physics and mutation changes apply between generations. Population and seed changes need a new experiment.").small().color(MUTED));
        }
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Save preset").clicked() {
                self.file("Save preset");
            }
            if ui.small_button("Load preset").clicked() {
                self.file("Load preset");
            }
            if ui.small_button("Reset settings").clicked() {
                self.config = Config::default();
                self.dirty = true;
            }
        });
        ui.separator();
        ui.label(RichText::new("Each creature runs its own trial. Faster walkers are more likely to survive; their offspring explore new shapes.").small().color(MUTED));
    }
    fn viewport(&mut self, ui: &mut egui::Ui, height: f32) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("LIVE CREATURE").small().color(MUTED));
            if let Some(p) = &self.playback {
                ui.label(format!(
                    "#{} · {} nodes / {} bones / {} muscles",
                    p.creature.id,
                    p.nodes.len(),
                    p.creature.bones.len(),
                    p.creature.muscles.len()
                ));
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
        if cfg.ground {
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
                Stroke::new(2., Color32::from_rgb(125, 159, 135)),
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
                (p.tick - physics::SETTLE) as f32 * physics::DT,
            );
            painter.text(
                rect.left_top() + Vec2::new(18., 16.),
                Align2::LEFT_TOP,
                format!("{:.2} m", physics::fitness(&p.nodes)),
                FontId::proportional(24.),
                MINT,
            );
            painter.text(
                rect.right_top() + Vec2::new(-18., 18.),
                Align2::RIGHT_TOP,
                format!(
                    "{:.1} / {:.0} s",
                    (p.tick - physics::SETTLE) as f32 * physics::DT,
                    p.config.duration
                ),
                FontId::proportional(14.),
                MUTED,
            );
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
                    physics::step(
                        &mut p.nodes,
                        &p.creature.bones,
                        &p.creature.muscles,
                        &p.config,
                        p.tick,
                    );
                    p.tick += 1;
                }
            }
            ui.add(
                egui::Slider::new(&mut self.speed, 0.25..=128.0)
                    .logarithmic(true)
                    .suffix("×")
                    .text("Playback"),
            );
        });
    }
    fn metrics(&self, ui: &mut egui::Ui) {
        if let Some(s) = self.snapshot.as_ref().and_then(|s| s.history.last()) {
            ui.columns(4, |cols| {
                for (ui, (name, value, color)) in cols.iter_mut().zip([
                    ("BEST", format!("{:.3} m", s.best), MINT),
                    ("QD SCORE", format!("{:.2}", s.qd_score), MINT),
                    ("NICHES", number(s.archive_cells), INK),
                    (
                        "EVALUATIONS / SEC",
                        format!("{:.0}", s.population as f64 / s.seconds.max(0.001)),
                        INK,
                    ),
                ]) {
                    egui::Frame::new()
                        .fill(CARD)
                        .corner_radius(8)
                        .inner_margin(12)
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.label(RichText::new(name).small().color(MUTED));
                            ui.label(RichText::new(value).size(22.).color(color));
                        });
                }
            });
        } else {
            ui.label(
                RichText::new("Run a generation to see how far your creatures can travel.")
                    .color(MUTED),
            );
        }
    }
    fn trend(&self, ui: &mut egui::Ui, height: f32) {
        let Some(s) = &self.snapshot else { return };
        Plot::new("fitness_history")
            .height(height)
            .legend(Legend::default())
            .x_axis_label("Generation")
            .y_axis_label("Distance (m)")
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
                        ("Best".into(), MINT, 2.5)
                    } else if i == 14 {
                        ("Median".into(), AMBER, 2.5)
                    } else if i == 0 {
                        ("Worst".into(), Color32::from_rgb(104, 133, 159), 1.5)
                    } else {
                        (format!("P{}", PERCENTILES[i]), species_color(i, 0), 1.)
                    };
                    plot.line(Line::new(name, values).color(color).width(width));
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
            .x_axis_label("Distance (m)")
            .allow_scroll(false)
            .show(ui, |plot| {
                plot.bar_chart(BarChart::new("Creatures", bars).color(MINT.gamma_multiply(0.65)));
            });
        if outside > 0 {
            ui.small(format!(
                "{outside} outside this range or failed · change range in Advanced"
            ));
        }
    }
    fn population(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Search archive");
            ui.label(
                RichText::new("Behavior niches and protected topologies · click to replay")
                    .color(MUTED),
            );
        });
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        ui.horizontal(|ui| {
            ui.label(format!("{} niches", number(snapshot.archive_cells)));
            ui.label(format!(
                "{} topology reserves",
                number(snapshot.innovation_reserve_count)
            ));
            ui.label(RichText::new(format!("QD score {:.2}", snapshot.qd_score)).color(MINT));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Next batch:").small().color(MUTED));
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
        let columns = (ui.available_width() / 155.).floor().max(2.) as usize;
        let width = (ui.available_width() - (columns - 1) as f32 * 10.) / columns as f32;
        let progress = (self.sort_started.elapsed().as_secs_f32() * self.sort_speed / 3.).min(1.);
        let animating = snapshot.stage == Stage::Archived && progress < 1.;
        let ease = progress * progress * (3. - 2. * progress);
        let item_count = if snapshot.archive_size > 0 {
            snapshot.archive_size
        } else {
            snapshot.config.population
        };
        let mut selected = None;
        let mut requested = None;
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
                                paint_card(ui.painter(), card, rect, response.hovered(), snapshot.stage);
                                if response.clicked() { selected = Some(card.index); }
                                response.on_hover_text(format!(
                                    "ID {}\n{} nodes / {} bones / {} muscles\nMutability {:.2}\n{}\n{}\nClick to replay",
                                    card.creature.id,
                                    card.creature.nodes.len(),
                                    card.creature.bones.len(),
                                    card.creature.muscles.len(),
                                    card.creature.mutability,
                                    card.emitter.map_or("Initial population".to_owned(), |emitter| format!("Emitter: {}", emitter.label())),
                                    card.descriptor.map_or_else(
                                        || if card.score.is_finite() { "Current trial evaluated".to_owned() } else { "Current trial pending".to_owned() },
                                        |d| format!("Contact {:.0}% · observed gait {:.2} Hz · form {:.2} · bob {:.2} m · {} visits", d.ground_contact * 100.0, d.gait_frequency, d.aspect_ratio, d.vertical_oscillation, card.visits),
                                    )
                                ));
                            } else {
                                ui.painter().rect_filled(destination, 8, CARD);
                                ui.painter().text(destination.center(), Align2::CENTER_CENTER, "Loading…", FontId::proportional(12.), MUTED);
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
        if let Some(start) = requested {
            self.worker.send(Command::Page(start));
            self.last_page = start;
        }
        if let Some(i) = selected {
            self.worker.send(Command::Preview(i));
            self.tab = Tab::Overview;
        }
    }
    fn species_history(&mut self, ui: &mut egui::Ui) {
        let Some(snapshot) = &self.snapshot else {
            return;
        };
        if snapshot.history.is_empty() {
            return;
        }
        ui.label(
            RichText::new("BODY TYPES THROUGH GENERATIONS")
                .small()
                .color(MUTED),
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
            Stroke::new(2., INK),
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
            cols[1].label(RichText::new("BODY TYPES").small().color(MUTED));
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
                ui.painter().rect_filled(rect, 8, CARD);
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
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
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
        let next = self.worker.view.lock().unwrap().take();
        if let Some(mut next) = next {
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
            }
            if std::env::var("EVOLUTION_BENCH_GENERATIONS")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .is_some_and(|target| target > 0 && next.generation >= target && !next.running)
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            self.snapshot = Some(next);
        }
        if !ctx.egui_wants_keyboard_input() {
            if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
                if self.active() {
                    self.pause();
                } else {
                    self.run(true, false);
                }
            }
            if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S)) {
                self.file("Save experiment");
            }
        }
        if self.playing
            && let Some(p) = &mut self.playback
        {
            p.accumulator = (p.accumulator + dt.min(0.1) * self.speed).min(1.0);
            let start = Instant::now();
            while p.accumulator >= physics::DT && start.elapsed() < Duration::from_millis(5) {
                if p.tick >= physics::SETTLE + p.config.steps() {
                    p.reset();
                }
                physics::step(
                    &mut p.nodes,
                    &p.creature.bones,
                    &p.creature.muscles,
                    &p.config,
                    p.tick,
                );
                p.tick += 1;
                p.accumulator -= physics::DT;
            }
            if self.follow {
                let x = p.nodes.iter().map(|n| n.pos[0]).sum::<f32>() / p.nodes.len() as f32;
                self.camera[0] += (x - self.camera[0]) * (dt * 8.).min(1.0);
            }
        }
        egui::Panel::top("top")
            .exact_size(64.)
            .frame(egui::Frame::new().fill(PANEL).inner_margin(12))
            .show(ui, |ui| self.top(ui));
        egui::Panel::bottom("status").show(ui,|ui|{ui.horizontal(|ui|{if let Some(s)=&self.snapshot {color_dot(ui, MINT);
ui.label(&s.status);
if let Some(error)=&s.error {ui.colored_label(AMBER,error);
}ui.with_layout(egui::Layout::right_to_left(egui::Align::Center),|ui|{if ui.small_button(if self.show_perf{"Hide performance"}else{"Performance"}).clicked(){self.show_perf= !self.show_perf;
}ui.label(RichText::new(&s.gpu).small().color(MUTED));
});
}});
if self.show_perf&& let Some(s)=&self.snapshot {let mut frames:Vec<_>=self.frame_times.iter().copied().collect();
frames.sort_by(f32::total_cmp);
let p95=frames.get(frames.len()*95/100).copied().unwrap_or(0.);
ui.small(format!("Frame p95 {:.1} ms · evaluation {:.2} s · {:.0} creatures/s · GPU buffers {:.1} MiB · population {:.1} MiB",p95*1000.,s.elapsed,s.evaluated as f64/s.elapsed.max(0.001),s.gpu_bytes as f64/1048576.,s.ram_bytes as f64/1048576.));
}
if let Some(m)=&self.message {ui.label(m);
}});
        egui::Panel::left("controls")
            .default_size(300.)
            .min_size(260.)
            .max_size(440.)
            .resizable(true)
            .frame(egui::Frame::new().fill(PANEL).inner_margin(16))
            .show(ui, |ui| self.controls(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(CANVAS).inner_margin(20))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (tab, label) in [
                        (Tab::Overview, "Overview"),
                        (Tab::Population, "Behavior archive"),
                        (Tab::History, "History & statistics"),
                    ] {
                        ui.selectable_value(&mut self.tab, tab, RichText::new(label).size(15.));
                    }
                });
                ui.add_space(8.);
                match self.tab {
                    Tab::Overview => {
                        self.metrics(ui);
                        ui.add_space(10.);
                        self.viewport(ui, (ui.available_height() * 0.62).max(180.));
                        ui.add_space(8.);
                        self.trend(ui, ui.available_height().max(100.));
                    }
                    Tab::Population => self.population(ui),
                    Tab::History => {
                        egui::ScrollArea::vertical().show(ui, |ui| self.history(ui));
                    }
                }
            });
        self.dialogs(&ctx);
        if self.active() {
            // Worker snapshots every 200ms already wake the UI; poll gently between them.
            ctx.request_repaint_after(Duration::from_millis(100));
        } else if self.playing {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        // Explicit opt-in capture hook for repeatable native rendering/performance checks.
        if let Some(path) = &self.capture_path {
            if self.started.elapsed() > Duration::from_secs(8) && !self.capture_requested {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                self.capture_requested = true;
            }
            for event in ctx.input(|i| i.events.clone()) {
                if let egui::Event::Screenshot { image, .. } = event {
                    let bytes: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                    if let Err(e) = image::save_buffer(
                        path,
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
) {
    painter.rect_filled(rect, 8, if hovered { CARD_HOVER } else { CARD });
    painter.rect_stroke(
        rect,
        8,
        Stroke::new(1., CARD_BORDER),
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
        MUTED,
    );
    if card.innovation_reserve {
        painter.text(
            rect.right_top() + Vec2::new(-9., 8.),
            Align2::RIGHT_TOP,
            "MORPH",
            FontId::proportional(9.),
            MINT,
        );
    }
    let (label, score_color) = if !card.score.is_finite() {
        if card.parent_score.is_finite() && card.parent_score > FAILED {
            (format!("Parent {:.3} m", card.parent_score), MUTED)
        } else if card.parent_score.is_finite() {
            ("Parent failed".into(), AMBER)
        } else {
            ("Trial pending".into(), MUTED)
        }
    } else if card.score <= FAILED {
        ("Failed trial".into(), AMBER)
    } else {
        (
            format!("{:.3} m", card.score),
            if card.survivor { MINT } else { INK },
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
            if card.survivor { MINT } else { AMBER },
        );
    }
}
fn numeric(
    ui: &mut egui::Ui,
    label: &str,
    v: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    speed: f64,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::DragValue::new(v)
                .range(range)
                .speed(speed)
                .max_decimals(3),
        );
    });
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
fn draw_creature(
    p: &egui::Painter,
    nodes: &[Node],
    c: &Creature,
    origin: Pos2,
    scale: f32,
    time: f32,
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
        let length = physics::target(m, time);
        let contraction = 1. - ((length - m.short) / (m.long - m.short).max(1e-5));
        let width = (scale * 0.017 * (1. + 0.45 * contraction)).max(2.);
        p.line_segment(
            [a, b],
            Stroke::new(width + 3., Color32::from_rgb(10, 15, 19)),
        );
        p.line_segment(
            [a, b],
            Stroke::new(
                width,
                if contraction > 0.5 {
                    AMBER
                } else {
                    Color32::from_rgb(106, 138, 145)
                },
            ),
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
    draw_creature(p, &nodes, c, origin, scale, 0.);
}
