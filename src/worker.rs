use crate::{
    config::Config,
    evolution::Creature,
    gpu::Gpu,
    qd::{self, Descriptor, Emitter, EmitterStats},
    storage::{self, Experiment, Stage, Stats},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};
pub enum Command {
    New(Config),
    Run { continuous: bool, guided: bool },
    Next,
    Pause,
    Configure(Config),
    Save(PathBuf),
    Load(PathBuf),
    Export(PathBuf),
    Page(usize),
    Preview(usize),
    Shutdown,
}
#[derive(Clone)]
pub struct Card {
    pub index: usize,
    pub rank: usize,
    pub score: f32,
    pub parent_score: f32,
    pub survivor: bool,
    pub descriptor: Option<Descriptor>,
    pub emitter: Option<Emitter>,
    pub visits: u64,
    pub creature: Creature,
}
#[derive(Clone)]
pub struct Snapshot {
    pub epoch: u64,
    pub config: Config,
    pub generation: u32,
    pub evaluated: usize,
    pub stage: Stage,
    pub running: bool,
    pub history: Arc<Vec<Stats>>,
    pub page: Vec<Card>,
    pub page_start: usize,
    pub preview: Option<(Creature, Config)>,
    pub gpu: String,
    pub gpu_bytes: u64,
    pub ram_bytes: usize,
    pub elapsed: f64,
    pub archive_cells: usize,
    pub qd_score: f64,
    pub emitters: [EmitterStats; 4],
    pub emitter_weights: [f64; 4],
    pub status: String,
    pub error: Option<String>,
}
pub struct Worker {
    pub tx: Sender<Command>,
    pub view: Arc<Mutex<Option<Snapshot>>>,
    pub pause: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}
impl Worker {
    pub fn spawn(gpu: Gpu, ctx: eframe::egui::Context) -> Self {
        let (tx, rx) = mpsc::channel();
        let view = Arc::new(Mutex::new(None));
        let output = view.clone();
        let pause = Arc::new(AtomicBool::new(false));
        let paused = pause.clone();
        let join = std::thread::Builder::new()
            .name("evolution".into())
            .spawn(move || run(gpu, rx, output, paused, ctx))
            .expect("Start simulation worker");
        Self {
            tx,
            view,
            pause,
            join: Some(join),
        }
    }
    pub fn send(&self, c: Command) {
        let _ = self.tx.send(c);
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.pause.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Command::Shutdown);
        // Finish compute before eframe destroys the Vulkan surface/device resources.
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
fn run(
    mut gpu: Gpu,
    rx: Receiver<Command>,
    output: Arc<Mutex<Option<Snapshot>>>,
    pause: Arc<AtomicBool>,
    ctx: eframe::egui::Context,
) {
    let mut exp: Option<Experiment> = None;
    let mut running = false;
    let mut continuous = false;
    let mut guided = false;
    let mut page = 0usize;
    let mut preview = None;
    let mut status = "Create a population to begin".to_owned();
    let mut error = None;
    let mut last_publish = Instant::now() - Duration::from_secs(1);
    let mut changed = true;
    let mut epoch = 0u64;
    let mut history = Arc::new(Vec::new());
    loop {
        let command = if running {
            rx.try_recv().ok()
        } else {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(c) => Some(c),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(_) => None,
            }
        };
        if let Some(command) = command {
            if matches!(command, Command::Shutdown) {
                break;
            }
            changed = true;
            error = None;
            let result: anyhow::Result<()> = (|| {
                match command {
                    Command::Shutdown => return Ok(()),
                    Command::New(cfg) => {
                        running = false;
                        status = "Creating population…".into();
                        let next = Experiment::new(cfg)?;
                        preview = Some((next.population.creature(0), next.config.clone()));
                        exp = Some(next);
                        epoch += 1;
                        history = Arc::new(Vec::new());
                        page = 0;
                        status = "Population ready".into();
                    }
                    Command::Run {
                        continuous: c,
                        guided: g,
                    } => {
                        continuous = c;
                        guided = g;
                        pause.store(false, Ordering::Relaxed);
                        running = true;
                    }
                    Command::Pause => {
                        running = false;
                        status = "Paused at a completed batch".into();
                    }
                    Command::Next => {
                        guided = true;
                        continuous = false;
                        pause.store(false, Ordering::Relaxed);
                        running = true;
                    }
                    Command::Configure(cfg) => {
                        if let Some(e) = &mut exp {
                            e.update_config(cfg)?;
                            status = "Settings applied or queued for the next generation".into();
                        }
                    }
                    Command::Save(path) => {
                        if let Some(e) = &exp {
                            storage::save(&path, e)?;
                            status = format!("Saved {}", path.display());
                        }
                    }
                    Command::Load(path) => {
                        let next = storage::load(&path)?;
                        preview = Some((next.population.creature(0), next.config.clone()));
                        exp = Some(next);
                        epoch += 1;
                        history = Arc::new(Vec::new());
                        running = false;
                        page = 0;
                        status = format!("Loaded {}", path.display());
                    }
                    Command::Export(path) => {
                        if let Some(e) = &exp {
                            storage::export_csv(&path, &e.history)?;
                            status = format!("Exported {}", path.display());
                        }
                    }
                    Command::Page(start) => {
                        page = start;
                    }
                    Command::Preview(i) => {
                        if let Some(e) = &exp {
                            if let Some(elite) = e.archive.entries.get(i) {
                                preview = Some((elite.creature.clone(), e.config.clone()));
                            } else if e.archive.entries.is_empty() && i < e.config.population {
                                preview = Some((e.population.creature(i), e.config.clone()));
                            }
                        }
                    }
                }
                Ok(())
            })();
            if let Err(e) = result {
                error = Some(format!("{e:#}"));
                running = false;
            }
            // Disconnection is handled separately; shutdown closes the receiver after this iteration.
        }
        if pause.load(Ordering::Relaxed) {
            running = false;
        }
        if running {
            if let Some(e) = &mut exp {
                let result: anyhow::Result<()> = (|| {
                    match e.stage {
                        Stage::Ready | Stage::Evaluating => {
                            e.stage = Stage::Evaluating;
                            let batch = e.config.batch_size();
                            let end = (e.evaluated + batch).min(e.config.population);
                            let indices: Vec<_> = (e.evaluated..end).collect();
                            let start = Instant::now();
                            let metrics =
                                gpu.evaluate_with_metrics(&e.population, &indices, &e.config)?;
                            e.evaluation_seconds += start.elapsed().as_secs_f64();
                            for (offset, metric) in metrics.iter().enumerate() {
                                e.scores[e.evaluated + offset] = metric.fitness;
                                e.trial_metrics[e.evaluated + offset] = metric.behavior;
                            }
                            e.evaluated = end;
                            status = format!("Evaluating generation {}", e.generation);
                            if end == e.config.population {
                                e.stage = Stage::Evaluated;
                                if guided {
                                    running = false;
                                }
                            }
                        }
                        Stage::Evaluated | Stage::Ranked | Stage::Selected => {
                            e.archive_batch()?;
                            status = format!(
                                "Archive: {} niches · QD score {:.2}",
                                e.archive.entries.len(),
                                e.archive.qd_score
                            );
                            if guided {
                                running = false;
                            }
                        }
                        Stage::Archived => {
                            e.prepare_next_batch()?;
                            status = "Breeding from diverse archive elites".into();
                            if e.config.checkpoint_interval > 0
                                && e.generation.is_multiple_of(e.config.checkpoint_interval)
                            {
                                storage::save(
                                    &PathBuf::from(format!("runs/seed-{}-auto.evo", e.config.seed)),
                                    e,
                                )?;
                            }
                            if !continuous || guided {
                                running = false;
                            }
                        }
                    }
                    Ok(())
                })();
                if let Err(err) = result {
                    error = Some(format!("{err:#}"));
                    running = false;
                }
                changed = true;
            } else {
                running = false;
            }
        }
        if changed && (last_publish.elapsed() > Duration::from_millis(200) || !running) {
            let snapshot = if let Some(e) = &exp {
                if history.len() != e.history.len() {
                    history = Arc::new(e.history.clone());
                }
                let archive_count = e.archive.entries.len();
                let item_count = if archive_count > 0 {
                    archive_count
                } else {
                    e.config.population
                };
                if page >= item_count {
                    page = 0;
                }
                let end = (page + 120).min(item_count);
                let cards = if archive_count > 0 {
                    let mut order: Vec<_> = (0..archive_count).collect();
                    order.sort_unstable_by(|&a, &b| {
                        e.archive.entries[b]
                            .fitness
                            .total_cmp(&e.archive.entries[a].fitness)
                    });
                    order
                        .into_iter()
                        .enumerate()
                        .skip(page)
                        .take(end.saturating_sub(page))
                        .map(|(rank, i)| {
                            let elite = &e.archive.entries[i];
                            Card {
                                index: i,
                                rank,
                                score: elite.fitness,
                                parent_score: f32::NAN,
                                survivor: false,
                                descriptor: Some(elite.descriptor),
                                emitter: Some(elite.emitter),
                                visits: elite.visits,
                                creature: elite.creature.clone(),
                            }
                        })
                        .collect()
                } else {
                    (page.min(end)..end)
                        .map(|i| Card {
                            index: i,
                            rank: i,
                            score: e.scores[i],
                            parent_score: e.parent_scores.get(i).copied().unwrap_or(f32::NAN),
                            survivor: false,
                            descriptor: None,
                            emitter: None,
                            visits: 0,
                            creature: e.population.creature(i),
                        })
                        .collect()
                };
                Snapshot {
                    epoch,
                    config: e.config.clone(),
                    generation: e.generation,
                    evaluated: e.evaluated,
                    stage: e.stage,
                    running,
                    history: history.clone(),
                    page: cards,
                    page_start: page,
                    preview: preview.take(),
                    gpu: gpu.name.clone(),
                    gpu_bytes: gpu.allocated_bytes,
                    ram_bytes: e.population.bytes()
                        + e.scores.capacity() * 4
                        + e.trial_metrics.capacity()
                            * std::mem::size_of::<crate::qd::TrialMetrics>()
                        + e.ranks.capacity() * 8
                        + e.parents.capacity() * 8
                        + e.archive
                            .entries
                            .iter()
                            .map(|elite| {
                                elite.creature.nodes.len()
                                    * std::mem::size_of::<crate::evolution::NodeGene>()
                                    + elite.creature.muscles.len()
                                        * std::mem::size_of::<crate::evolution::Muscle>()
                            })
                            .sum::<usize>(),
                    elapsed: e.evaluation_seconds,
                    archive_cells: archive_count,
                    qd_score: e.archive.qd_score,
                    emitters: e.emitter_stats,
                    emitter_weights: qd::emitter_weights(&e.emitter_stats),
                    status: status.clone(),
                    error: error.clone(),
                }
            } else {
                Snapshot {
                    epoch,
                    config: Config::default(),
                    generation: 0,
                    evaluated: 0,
                    stage: Stage::Ready,
                    running: false,
                    history: history.clone(),
                    page: vec![],
                    page_start: 0,
                    preview: None,
                    gpu: gpu.name.clone(),
                    gpu_bytes: gpu.allocated_bytes,
                    ram_bytes: 0,
                    elapsed: 0.,
                    archive_cells: 0,
                    qd_score: 0.0,
                    emitters: [EmitterStats::default(); 4],
                    emitter_weights: qd::emitter_weights(&[EmitterStats::default(); 4]),
                    status: status.clone(),
                    error: error.clone(),
                }
            };
            *output.lock().unwrap() = Some(snapshot);
            ctx.request_repaint();
            changed = false;
            last_publish = Instant::now();
        }
    }
}
