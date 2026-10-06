//! Common interface for the evaluation devices: NVIDIA GPUs through CUDA.
//!
//! An engine accepts units of creatures, evaluates them asynchronously on its
//! own thread, and returns raw per-creature results in unit order. Callers
//! hand over an immutable population, shared with any retained submission,
//! so packing and simulation never hold up the caller.
use crate::{
    config::Config,
    creature_kernel::{self, GpuResult},
    cuda_engine::CudaEngine,
    evolution::{Creature, Population},
};
use anyhow::{Context, Result};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

pub struct Finished {
    pub ticket: u64,
    /// One result per creature, in unit order.
    pub results: Vec<GpuResult>,
    /// Device time spent on this unit.
    pub busy_seconds: f64,
}

/// A creature's trial recorded on the GPU for a replay: node positions
/// before every step and after the last, and the result the kernel scored in
/// the same run.
pub struct Recording {
    pub frames: Vec<Vec<[f32; 2]>>,
    pub result: GpuResult,
    /// The muscle energy, muscle force and ground contact force the kernel
    /// recorded with each frame; `None` only where an engine cannot record them.
    pub forces: Option<crate::replay_forces::Forces>,
}

struct ReplayRequest {
    creature: Creature,
    cfg: Config,
    reply: mpsc::Sender<Result<Recording, String>>,
}

/// The GPU engine that records replays: the primary GPU, which scores the
/// archive's creatures.
static REPLAYS: std::sync::Mutex<Option<mpsc::Sender<ReplayRequest>>> = std::sync::Mutex::new(None);

/// Records `creature`'s trial on the GPU whose scores the archive holds,
/// with the kernel that scores evolution, waiting up to `timeout`. None
/// when no GPU evaluates or it cannot answer in time.
pub fn record_on_gpu(creature: &Creature, cfg: &Config, timeout: Duration) -> Option<Recording> {
    let sender = REPLAYS.lock().unwrap_or_else(|e| e.into_inner()).clone()?;
    let (reply, answer) = mpsc::channel();
    sender
        .send(ReplayRequest {
            creature: creature.clone(),
            cfg: cfg.clone(),
            reply,
        })
        .ok()?;
    match answer.recv_timeout(timeout) {
        Ok(Ok(recording)) => Some(recording),
        Ok(Err(error)) => {
            eprintln!("GPU replay failed: {error}");
            None
        }
        Err(_) => {
            eprintln!("GPU replay did not answer in time");
            None
        }
    }
}

/// A creature's full trial for the replay viewer and the result scored in
/// the same run, recorded by the scoring kernel on the GPU that scores the
/// archive, with the muscle energy, muscle force and ground contact forces it
/// recorded with each frame. A replay runs the full trial, without the early
/// screen. None when the GPU did not answer within `patience`.
pub fn replay(creature: &Creature, cfg: &Config, patience: Duration) -> Option<Replay> {
    let cfg = Config {
        screen: None,
        rungs: None,
        ..cfg.clone()
    };
    // A fine trial records several frames per standard step; the viewer
    // plays standard steps, so it keeps one frame per standard step.
    let every = (cfg.fidelity().rate / crate::physics::Fidelity::standard().rate).max(1) as usize;
    let recording = record_on_gpu(creature, &cfg, patience)?;
    Some(thin(
        recording.frames,
        recording.result,
        recording.forces,
        every,
    ))
}

pub type Replay = (
    Vec<Vec<[f32; 2]>>,
    GpuResult,
    Option<crate::replay_forces::Forces>,
);
/// Keeps every `every`-th frame (the first and the last always).
fn thin(
    frames: Vec<Vec<[f32; 2]>>,
    result: GpuResult,
    forces: Option<crate::replay_forces::Forces>,
    every: usize,
) -> Replay {
    if every <= 1 {
        return (frames, result, forces);
    }
    fn keep<T>(v: Vec<T>, every: usize) -> Vec<T> {
        let last = v.len().saturating_sub(1);
        v.into_iter()
            .enumerate()
            .filter(|(i, _)| i % every == 0 || *i == last)
            .map(|(_, x)| x)
            .collect()
    }
    let forces = forces.map(|f| crate::replay_forces::Forces {
        energy: keep(f.energy, every),
        muscle: keep(f.muscle, every),
        ground: keep(f.ground, every),
        friction: keep(f.friction, every),
        broken: keep(f.broken, every),
    });
    (keep(frames, every), result, forces)
}

/// Whether `cfg` runs a confirmation trial: its physics is finer than the
/// standard (`scheduler::confirm_config`).
pub fn is_confirmation(cfg: &Config) -> bool {
    cfg.fidelity() != crate::physics::Fidelity::standard()
}

pub trait Engine: Send {
    fn name(&self) -> String;
    /// Largest body (in nodes) this engine can evaluate.
    fn max_nodes(&self) -> usize;
    /// Standard units that can be queued now without waiting.
    fn free_slots(&self) -> usize;
    /// Confirmation units that can be queued now without waiting. They have
    /// a slot of their own on a GPU, so standard work never holds them back.
    fn free_confirm_slots(&self) -> usize {
        self.free_slots()
    }
    /// Queues a unit; `unit` holds exactly the unit's creatures, in order.
    fn submit(&mut self, unit: Population, cfg: &Config) -> Result<u64> {
        self.submit_shared(Arc::new(unit), cfg)
    }
    /// Queues an immutable unit without copying its population storage.
    fn submit_shared(&mut self, unit: Arc<Population>, cfg: &Config) -> Result<u64>;
    /// Returns the next finished unit without blocking.
    fn poll(&mut self) -> Result<Option<Finished>>;
    /// Blocks up to `timeout` for the oldest queued unit.
    fn wait(&mut self, timeout: Duration);
    fn allocated_bytes(&self) -> u64 {
        0
    }
}

/// Front end shared by engines that run on their own thread.
pub struct ThreadedEngine {
    name: String,
    max_nodes: usize,
    depth: usize,
    /// Closed on drop so the engine thread finishes queued work and exits.
    jobs: Option<mpsc::Sender<(u64, Arc<Population>, Config)>>,
    done: mpsc::Receiver<Result<Finished, String>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Tickets of the units on the engine thread, and whether each is a
    /// confirmation trial.
    queued: VecDeque<(u64, bool)>,
    ready: VecDeque<Finished>,
    /// Retained until teardown, after all successful results have been delivered.
    failure: Option<String>,
    next_ticket: u64,
    allocated: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Where a GPU engine takes replay requests.
    replays: Option<mpsc::Sender<ReplayRequest>>,
}

impl ThreadedEngine {
    /// Makes this GPU the one that records replays (`replay`).
    pub fn publish_replays(&self) {
        if let Some(sender) = &self.replays {
            *REPLAYS.lock().unwrap_or_else(|e| e.into_inner()) = Some(sender.clone());
        }
    }

    fn fail(&mut self, reason: &str) {
        self.failure
            .get_or_insert_with(|| format!("{} failed: {reason}", self.name));
        self.jobs.take();
    }

    fn receive(&mut self, mut timeout: Duration) {
        while self.failure.is_none() {
            let item = if timeout.is_zero() {
                self.done.try_recv().map_err(|error| match error {
                    mpsc::TryRecvError::Empty => mpsc::RecvTimeoutError::Timeout,
                    mpsc::TryRecvError::Disconnected => mpsc::RecvTimeoutError::Disconnected,
                })
            } else {
                self.done.recv_timeout(timeout)
            };
            match item {
                Ok(Ok(done)) => self.ready.push_back(done),
                Ok(Err(error)) => self.fail(&error),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.fail("result channel disconnected");
                }
                Err(mpsc::RecvTimeoutError::Timeout) => return,
            }
            // Wait for at most the first result; drain the rest without waiting.
            timeout = Duration::ZERO;
        }
    }

    fn submission_error(&mut self) -> anyhow::Error {
        // The worker may have finished older jobs after the initial poll,
        // then closed its job receiver before this send. Keep those results.
        self.receive(Duration::ZERO);
        self.fail("job channel disconnected");
        anyhow::anyhow!(
            "{}",
            self.failure.as_deref().expect("recorded worker failure")
        )
    }
}

impl Engine for ThreadedEngine {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn max_nodes(&self) -> usize {
        self.max_nodes
    }
    fn free_slots(&self) -> usize {
        if self.failure.is_some() {
            0
        } else {
            self.depth.saturating_sub(
                self.queued
                    .iter()
                    .filter(|(_, confirming)| !confirming)
                    .count(),
            )
        }
    }
    fn free_confirm_slots(&self) -> usize {
        if self.failure.is_some() {
            0
        } else {
            CONFIRM_DEPTH.saturating_sub(
                self.queued
                    .iter()
                    .filter(|(_, confirming)| *confirming)
                    .count(),
            )
        }
    }
    fn submit_shared(&mut self, unit: Arc<Population>, cfg: &Config) -> Result<u64> {
        self.receive(Duration::ZERO);
        if let Some(error) = &self.failure {
            anyhow::bail!("{error}");
        }
        let ticket = self.next_ticket;
        if self
            .jobs
            .as_ref()
            .context("Engine is shutting down")?
            .send((ticket, unit, cfg.clone()))
            .is_err()
        {
            return Err(self.submission_error());
        }
        self.next_ticket += 1;
        self.queued.push_back((ticket, is_confirmation(cfg)));
        Ok(ticket)
    }
    fn poll(&mut self) -> Result<Option<Finished>> {
        self.receive(Duration::ZERO);
        let Some(done) = self.ready.pop_front() else {
            if let Some(error) = &self.failure {
                anyhow::bail!("{error}");
            }
            return Ok(None);
        };
        let position = self
            .queued
            .iter()
            .position(|&(ticket, _)| ticket == done.ticket);
        anyhow::ensure!(position.is_some(), "{} returned an unknown unit", self.name);
        self.queued.remove(position.unwrap());
        Ok(Some(done))
    }
    fn wait(&mut self, timeout: Duration) {
        if self.ready.is_empty() {
            self.receive(timeout);
        }
    }
    fn allocated_bytes(&self) -> u64 {
        self.allocated.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for ThreadedEngine {
    fn drop(&mut self) {
        // Join before the process tears down CUDA and the thread pools.
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Results of one completed submission.
pub struct Completed {
    pub ticket: u64,
    /// The raw GPU result of every creature, in unit order.
    pub results: Vec<GpuResult>,
    /// The unit's batches, whose buffers the next unit packs into.
    pub batches: Vec<creature_kernel::LaneBatch>,
    /// For a recording (`CudaEngine::record`): node positions as
    /// `[creature][frame][node]`, with the batch's node stride.
    pub frames: Option<Vec<[f32; 2]>>,
    pub gpu_seconds: f64,
}

/// Submission slots per GPU for standard units. One more unit packs on the
/// engine thread while every slot runs.
pub fn gpu_slots() -> u32 {
    4
}

/// Confirmation units on the engine thread at most: one on the GPU's slot for
/// them and one that packs.
const CONFIRM_DEPTH: usize = 2;

/// Buffer size for `size` bytes of data. Small buffers round up to a power
/// of two, which costs little. Large ones get 25% headroom, so units of
/// slightly different sizes reuse them, without the up to 2x waste of a
/// power of two.
pub fn padded_size(size: u64) -> u64 {
    const LARGE: u64 = 1 << 20;
    let size = size.max(256);
    if size <= LARGE {
        size.next_power_of_two()
    } else {
        (size + size / 4).next_multiple_of(LARGE)
    }
}

/// True when `error` comes from a failed device or pinned host memory
/// allocation, which another process holding GPU memory can cause for a
/// while.
pub fn out_of_memory(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<crate::cuda_engine::CudaError>()
            .is_some_and(crate::cuda_engine::CudaError::out_of_memory)
    })
}

/// What the GPU engine thread needs from a device. `CudaEngine` implements
/// it; tests use a fake that can run out of memory.
trait Device {
    /// Standard units that can be submitted now.
    fn free_slots(&self) -> usize;
    /// Whether a confirmation trial can be submitted now.
    fn confirm_free(&self) -> bool {
        self.free_slots() > 0
    }
    /// Uploads the batches and queues their whole trials. On success the
    /// device takes the batches (`batches` is left empty) and returns them
    /// in `Completed`; on failure they stay.
    fn submit(
        &mut self,
        batches: &mut Vec<creature_kernel::LaneBatch>,
        cfg: &Config,
    ) -> Result<u64>;
    fn poll(&mut self, timeout: Duration) -> Result<Option<Completed>>;
    /// Frees buffers kept for reuse by slots with nothing in flight.
    fn release_idle(&mut self) -> u64;
    fn allocated_bytes(&self) -> u64;
    /// Whether a replay can be recorded now.
    fn replay_free(&self) -> bool;
    /// Queues a whole recorded trial of one batch (`CudaEngine::record`).
    fn record(&mut self, batch: creature_kernel::LaneBatch, cfg: &Config) -> Result<u64>;
    /// Whether a failed submission ran out of memory.
    fn out_of_memory(&self, error: &anyhow::Error) -> bool {
        out_of_memory(error)
    }
}

impl Device for CudaEngine {
    fn free_slots(&self) -> usize {
        CudaEngine::free_slots(self)
    }
    fn confirm_free(&self) -> bool {
        CudaEngine::confirm_free(self)
    }
    fn submit(
        &mut self,
        batches: &mut Vec<creature_kernel::LaneBatch>,
        cfg: &Config,
    ) -> Result<u64> {
        CudaEngine::submit(self, batches, cfg)
    }
    fn poll(&mut self, timeout: Duration) -> Result<Option<Completed>> {
        CudaEngine::poll(self, timeout)
    }
    fn release_idle(&mut self) -> u64 {
        CudaEngine::release_idle(self)
    }
    fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    fn replay_free(&self) -> bool {
        CudaEngine::replay_free(self)
    }
    fn record(&mut self, batch: creature_kernel::LaneBatch, cfg: &Config) -> Result<u64> {
        CudaEngine::record(self, batch, cfg)
    }
}

/// A packed unit waiting for a free slot or for GPU memory.
struct PackedUnit {
    ticket: u64,
    cfg: Config,
    batches: Vec<creature_kernel::LaneBatch>,
}

/// A unit on the GPU: its ticket and how many creatures it holds.
struct RunningUnit {
    ticket: u64,
    count: usize,
    /// A confirmation trial, which has a slot of its own.
    confirming: bool,
}

/// What to do after a submission ran out of memory.
#[derive(Debug, PartialEq)]
enum OutOfMemory {
    /// Keep the unit and try again later; print the line if there is one.
    Retry(Option<String>),
    /// Memory stayed short for this long with nothing running: fail.
    GiveUp(Duration),
}

/// Rides out failed GPU memory allocations instead of failing the GPU,
/// which would stop evolution. Another process
/// can hold GPU memory for a while. While other units run, a unit that does
/// not fit waits for one of them to finish, and fewer units run at once from
/// then on; one more is tried every `RAISE_AFTER`. With nothing running, it
/// retries every `pause` for up to `limit`.
struct MemoryBackoff {
    slots: usize,
    /// Units allowed in flight at once.
    cap: usize,
    raise_at: Option<Instant>,
    /// When a submission first failed with nothing in flight, until one succeeds.
    stalled_since: Option<Instant>,
    retry_at: Option<Instant>,
    waiting_announced: bool,
    short_announced: bool,
    pause: Duration,
    limit: Duration,
}

impl MemoryBackoff {
    const RAISE_AFTER: Duration = Duration::from_secs(10);

    fn new(slots: usize, pause: Duration, limit: Duration) -> Self {
        Self {
            slots,
            cap: slots,
            raise_at: None,
            stalled_since: None,
            retry_at: None,
            waiting_announced: false,
            short_announced: false,
            pause,
            limit,
        }
    }

    /// Whether another unit may be submitted now, with `in_flight` running.
    fn may_submit(&mut self, now: Instant, in_flight: usize) -> bool {
        if self.cap < self.slots && self.raise_at.is_some_and(|at| now >= at) {
            self.cap += 1;
            self.raise_at = (self.cap < self.slots).then(|| now + Self::RAISE_AFTER);
        }
        in_flight < self.cap && self.retry_at.is_none_or(|at| now >= at)
    }

    /// A submission failed for lack of memory with `in_flight` units still
    /// running, after idle slots freed `freed` bytes of cached buffers.
    fn out_of_memory(&mut self, now: Instant, in_flight: usize, freed: u64) -> OutOfMemory {
        if in_flight > 0 {
            // Running units hold memory this one needs: wait for them.
            self.cap = self.cap.min(in_flight);
            self.raise_at = Some(now + Self::RAISE_AFTER);
            let line = (!self.short_announced).then(|| {
                self.short_announced = true;
                format!(
                    "GPU memory is short; running {} of {} units at once",
                    self.cap, self.slots
                )
            });
            return OutOfMemory::Retry(line);
        }
        let first = self.stalled_since.is_none();
        let since = *self.stalled_since.get_or_insert(now);
        let waited = now.duration_since(since);
        if waited >= self.limit {
            return OutOfMemory::GiveUp(waited);
        }
        if first && freed > 0 {
            // Buffers cached for other units were in the way: try at once.
            self.retry_at = None;
            return OutOfMemory::Retry(None);
        }
        self.retry_at = Some(now + self.pause);
        let line = (!self.waiting_announced).then(|| {
            self.waiting_announced = true;
            format!(
                "out of GPU memory; freed cached buffers, waiting up to {:.0} s for memory",
                self.limit.as_secs_f64()
            )
        });
        OutOfMemory::Retry(line)
    }

    /// A submission succeeded; `in_flight` units now run.
    fn submitted(&mut self, now: Instant, in_flight: usize) -> Option<String> {
        self.retry_at = None;
        let mut lines = Vec::new();
        if let Some(since) = self.stalled_since.take()
            && self.waiting_announced
        {
            lines.push(format!(
                "GPU memory available again after {:.1} s",
                now.duration_since(since).as_secs_f64()
            ));
        }
        self.waiting_announced = false;
        if self.short_announced && in_flight >= self.slots {
            self.short_announced = false;
            lines.push(format!(
                "GPU memory suffices again for {} units at once",
                self.slots
            ));
        }
        (!lines.is_empty()).then(|| lines.join("; "))
    }

    /// How long to wait before the next retry when nothing is running.
    fn pause(&self, now: Instant) -> Option<Duration> {
        self.retry_at.map(|at| at.saturating_duration_since(now))
    }
}

/// Opens the NVIDIA GPU named `name` on its own thread, with kernels for
/// bodies up to `max_nodes` nodes. The thread packs the next unit while
/// earlier units run on the GPU.
pub fn gpu_engine(name: &str, max_nodes: usize) -> Result<ThreadedEngine> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (jobs, job_rx) = mpsc::channel::<(u64, Arc<Population>, Config)>();
    let (done_tx, done) = mpsc::channel();
    let (replays, replay_rx) = mpsc::channel::<ReplayRequest>();
    let allocated = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let thread_allocated = allocated.clone();
    let device_name = name.to_owned();
    let slots = gpu_slots() as usize;
    let thread = std::thread::Builder::new()
        .name(format!("gpu-{name}"))
        .spawn(move || {
            crate::threads::pin_engine();
            let opening = crate::loading::start_in(
                format!("Opening {device_name} with CUDA"),
                Some(crate::loading::Group::Startup),
            );
            let engine = match CudaEngine::new(&device_name, max_nodes) {
                Ok(engine) => {
                    opening.finish(false);
                    let _ = ready_tx.send(Ok((engine.name.clone(), engine.max_capacity)));
                    engine
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let name = engine.name.clone();
            let memory = MemoryBackoff::new(
                slots + 1,
                Duration::from_millis(500),
                Duration::from_secs(60),
            );
            // Replays run on the recording variant of the kernel that scores.
            run_units(
                engine,
                &name,
                job_rx,
                done_tx,
                Some(replay_rx),
                &thread_allocated,
                memory,
            );
        })
        .context("GPU engine thread")?;
    let (device_name, max_nodes) = ready_rx.recv().context("GPU engine thread stopped")??;
    Ok(ThreadedEngine {
        name: device_name,
        max_nodes,
        // One unit packs on the engine thread while every slot runs.
        depth: slots + 1,
        jobs: Some(jobs),
        done,
        thread: Some(thread),
        queued: VecDeque::new(),
        ready: VecDeque::new(),
        failure: None,
        next_ticket: 0,
        allocated,
        replays: Some(replays),
    })
}

/// Packs a job into the buffers of finished units (`spare`) where it can.
fn pack_unit(
    ticket: u64,
    unit: Arc<Population>,
    cfg: Config,
    indices: &mut Vec<usize>,
    spare: &mut Vec<creature_kernel::LaneBatch>,
) -> Result<PackedUnit> {
    indices.clear();
    indices.extend(0..unit.genomes.len());
    let batches = crate::kernel::pack_reusing(&unit, indices, &cfg, spare)?;
    Ok(PackedUnit {
        ticket,
        cfg,
        batches,
    })
}

/// The GPU engine thread: packs jobs, runs each as one whole-trial
/// submission, and returns finished units until the job channel closes.
fn run_units<D: Device>(
    mut engine: D,
    name: &str,
    job_rx: mpsc::Receiver<(u64, Arc<Population>, Config)>,
    done_tx: mpsc::Sender<Result<Finished, String>>,
    replays: Option<mpsc::Receiver<ReplayRequest>>,
    allocated: &AtomicU64,
    mut memory: MemoryBackoff,
) {
    let mut recording: Option<InFlightReplay> = None;
    // A packed unit that waits for memory (it goes before new jobs), and
    // submitted units by device ticket.
    let mut waiting: Option<PackedUnit> = None;
    let mut running: Vec<(u64, RunningUnit)> = Vec::new();
    // Jobs that arrived, by kind: a confirmation trial has a slot of its own
    // and goes first, so it never waits behind standard units.
    let mut standard_jobs: VecDeque<(u64, Arc<Population>, Config)> = VecDeque::new();
    let mut confirm_jobs: VecDeque<(u64, Arc<Population>, Config)> = VecDeque::new();
    // Batches of finished units: the next unit packs into their buffers, so
    // the host memory the GPU copies from is mapped and registered once.
    // Confirmation units are small and keep buffers apart from standard ones.
    let mut spare: Vec<creature_kernel::LaneBatch> = Vec::new();
    let mut confirm_spare: Vec<creature_kernel::LaneBatch> = Vec::new();
    let mut indices: Vec<usize> = Vec::new();
    let spare_bytes = |spare: &[creature_kernel::LaneBatch]| -> u64 {
        spare.iter().map(|b| CudaEngine::held_bytes(b) as u64).sum()
    };
    let mut open = true;
    loop {
        if open {
            let idle = running.is_empty()
                && waiting.is_none()
                && recording.is_none()
                && standard_jobs.is_empty()
                && confirm_jobs.is_empty();
            let mut arrived = Vec::new();
            // While idle, wake every few milliseconds for replay requests.
            match (idle, &replays) {
                (true, Some(_)) => match job_rx.recv_timeout(Duration::from_millis(5)) {
                    Ok(job) => arrived.push(job),
                    Err(mpsc::RecvTimeoutError::Disconnected) => open = false,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                },
                (true, None) => match job_rx.recv() {
                    Ok(job) => arrived.push(job),
                    Err(_) => open = false,
                },
                (false, _) => {}
            }
            // Every job that is already there, so a confirmation trial is not
            // stuck in the channel behind a standard unit that waits for a slot.
            loop {
                match job_rx.try_recv() {
                    Ok(job) => arrived.push(job),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        open = false;
                        break;
                    }
                }
            }
            for job in arrived {
                if is_confirmation(&job.2) {
                    confirm_jobs.push_back(job);
                } else {
                    standard_jobs.push_back(job);
                }
            }
        }
        if !open
            && running.is_empty()
            && waiting.is_none()
            && standard_jobs.is_empty()
            && confirm_jobs.is_empty()
            && recording.is_none()
        {
            break;
        }
        // Replays have their own slot and queue, so they never wait behind
        // evaluation.
        if recording.is_none()
            && engine.replay_free()
            && let Some(request) = replays.as_ref().and_then(|rx| rx.try_recv().ok())
        {
            match start_recording(&mut engine, &request) {
                Ok((ticket, layout, total)) => {
                    recording = Some(InFlightReplay {
                        ticket,
                        reply: request.reply,
                        layout,
                        total,
                    });
                }
                Err(error) => {
                    let _ = request.reply.send(Err(format!("{error:#}")));
                }
            }
            allocated.store(
                engine.allocated_bytes() + spare_bytes(&spare) + spare_bytes(&confirm_spare),
                Ordering::Relaxed,
            );
        }
        // One unit goes to the GPU: a unit that waited for memory first, then
        // a confirmation trial (its slot is not the standard units'), then a
        // standard unit.
        let can_confirm = engine.confirm_free();
        let can_standard = engine.free_slots() > 0;
        let may_submit =
            (can_confirm || can_standard) && memory.may_submit(Instant::now(), running.len());
        let next: Option<Result<PackedUnit>> = if !may_submit {
            None
        } else if let Some(unit) = waiting.take() {
            if if is_confirmation(&unit.cfg) {
                can_confirm
            } else {
                can_standard
            } {
                Some(Ok(unit))
            } else {
                waiting = Some(unit);
                None
            }
        } else if can_confirm && let Some((ticket, unit, cfg)) = confirm_jobs.pop_front() {
            Some(pack_unit(
                ticket,
                unit,
                cfg,
                &mut indices,
                &mut confirm_spare,
            ))
        } else if can_standard && let Some((ticket, unit, cfg)) = standard_jobs.pop_front() {
            Some(pack_unit(ticket, unit, cfg, &mut indices, &mut spare))
        } else {
            None
        };
        if let Some(next) = next {
            let mut unit = match next {
                Ok(unit) => unit,
                Err(err) => {
                    let _ = done_tx.send(Err(format!("{err:#}")));
                    return;
                }
            };
            let confirming = is_confirmation(&unit.cfg);
            let count = unit.batches.iter().map(|b| b.slots.len()).sum();
            match engine.submit(&mut unit.batches, &unit.cfg) {
                Ok(device_ticket) => {
                    running.push((
                        device_ticket,
                        RunningUnit {
                            ticket: unit.ticket,
                            count,
                            confirming,
                        },
                    ));
                    if let Some(line) = memory.submitted(Instant::now(), running.len()) {
                        eprintln!("{name}: {line}");
                    }
                }
                // Keep the unit; it runs once memory frees up.
                Err(err) if engine.out_of_memory(&err) => {
                    // Spare host buffers are the first to go.
                    let freed =
                        engine.release_idle() + spare_bytes(&spare) + spare_bytes(&confirm_spare);
                    spare.clear();
                    confirm_spare.clear();
                    waiting = Some(unit);
                    match memory.out_of_memory(Instant::now(), running.len(), freed) {
                        OutOfMemory::Retry(line) => {
                            if let Some(line) = line {
                                eprintln!("{name}: {line}");
                            }
                        }
                        OutOfMemory::GiveUp(waited) => {
                            let _ = done_tx.send(Err(format!(
                                "{err:#} (no GPU memory for {:.0} s)",
                                waited.as_secs_f64()
                            )));
                            return;
                        }
                    }
                }
                Err(err) => {
                    let _ = done_tx.send(Err(format!("{err:#}")));
                    return;
                }
            }
            allocated.store(
                engine.allocated_bytes() + spare_bytes(&spare) + spare_bytes(&confirm_spare),
                Ordering::Relaxed,
            );
            continue;
        }
        if running.is_empty() && recording.is_none() {
            // Nothing in flight: a unit may be waiting out a memory shortage.
            if let Some(pause) = memory.pause(Instant::now()) {
                std::thread::sleep(pause.min(Duration::from_millis(50)));
            }
            continue;
        }
        // Wait briefly for any submission, then check for new jobs.
        match engine.poll(Duration::from_millis(1)) {
            Ok(Some(finished))
                if recording
                    .as_ref()
                    .is_some_and(|replay| replay.ticket == finished.ticket) =>
            {
                let replay = recording.take().expect("a recording");
                let _ = replay
                    .reply
                    .send(recorded(&finished, replay.layout, replay.total));
            }
            Ok(Some(finished)) => {
                // Units on separate streams can finish out of order.
                let Some(position) = running
                    .iter()
                    .position(|(device_ticket, _)| *device_ticket == finished.ticket)
                else {
                    let _ = done_tx.send(Err("unknown GPU submission finished".into()));
                    return;
                };
                let (_, unit) = running.swap_remove(position);
                if finished.results.len() != unit.count {
                    let _ = done_tx.send(Err("a GPU submission returned too few results".into()));
                    return;
                }
                if unit.confirming {
                    confirm_spare.extend(finished.batches);
                } else {
                    spare.extend(finished.batches);
                }
                let message = Finished {
                    ticket: unit.ticket,
                    results: finished.results,
                    busy_seconds: finished.gpu_seconds,
                };
                if done_tx.send(Ok(message)).is_err() {
                    return;
                }
            }
            Ok(None) => {}
            Err(err) => {
                let _ = done_tx.send(Err(format!("{err:#}")));
                return;
            }
        }
    }
}

/// A replay being recorded: its device ticket, where the answer goes, and
/// the body's node count, node stride and trial length.
struct InFlightReplay {
    ticket: u64,
    reply: mpsc::Sender<Result<Recording, String>>,
    layout: FrameLayout,
    total: u32,
}

/// How one recorded frame is laid out (`creature_kernel::frame_stride`): the
/// body's `nodes` positions at the start of `stride` slots, then, when
/// `stride` is longer than `capacity`, an (energy, force) pair per muscle and
/// a (normal, friction) contact force per node, and last the bits of the
/// broken joints. The kernel numbers nodes so
/// that bone `j` ends at node `j + 1`; `order[k]` is the creature's own number
/// of kernel node `k`, as in `physics2::Model`.
struct FrameLayout {
    order: Vec<usize>,
    capacity: usize,
    muscles: usize,
    stride: usize,
}

/// Packs a replay request's creature and queues its recording. Returns the
/// device ticket, the frame layout and the trial length.
fn start_recording<D: Device>(
    engine: &mut D,
    request: &ReplayRequest,
) -> Result<(u64, FrameLayout, u32)> {
    let mut population = Population::default();
    population.push(request.creature.clone());
    let mut batches = crate::kernel::pack(&population, &[0], &request.cfg)?;
    anyhow::ensure!(batches.len() == 1, "A replay packs into one batch");
    let fidelity = request.cfg.fidelity();
    let total = fidelity.settle() + request.cfg.steps();
    let batch = batches.pop().expect("one batch");
    // The kernel's node numbering, from the bone order `pack` gave it.
    let mut creature = request.creature.clone();
    crate::evolution::canonicalize_bone_order(&mut creature);
    let layout = FrameLayout {
        order: std::iter::once(0)
            .chain(creature.bones.iter().map(|b| b.b as usize))
            .collect(),
        capacity: batch.capacity,
        muscles: batch.info.first().map_or(0, |i| i[2] as usize),
        stride: creature_kernel::frame_stride(&batch),
    };
    let ticket = engine.record(batch, &request.cfg)?;
    Ok((ticket, layout, total))
}

/// The replay in a finished recording: one frame per step and one after the
/// last, each with the body's node positions.
fn recorded(finished: &Completed, layout: FrameLayout, total: u32) -> Result<Recording, String> {
    let FrameLayout {
        order,
        capacity,
        muscles,
        stride,
    } = layout;
    let nodes = order.len();
    // Values per kernel node, put back in the creature's node numbering.
    let renumber = |value: &dyn Fn(usize) -> f32| {
        let mut out = vec![0.0; nodes];
        for (k, &node) in order.iter().enumerate() {
            out[node] = value(k);
        }
        out
    };
    let flat = finished
        .frames
        .as_ref()
        .ok_or("the recording returned no frames")?;
    let result = *finished
        .results
        .first()
        .ok_or("the recording returned no result")?;
    let count = total as usize + 1;
    if flat.len() < count * stride {
        return Err("the recording returned too few frames".into());
    }
    let frames = (0..count)
        .map(|t| {
            let mut frame = vec![[0.0; 2]; nodes];
            for (k, &node) in order.iter().enumerate() {
                frame[node] = flat[t * stride + k];
            }
            frame
        })
        .collect();
    let forces = (stride > capacity + muscles + nodes).then(|| {
        let slot = |t: usize, k: usize| flat[t * stride + capacity + k];
        crate::replay_forces::Forces {
            energy: (0..count)
                .map(|t| (0..muscles).map(|k| slot(t, k)[0]).collect())
                .collect(),
            muscle: (0..count)
                .map(|t| (0..muscles).map(|k| slot(t, k)[1]).collect())
                .collect(),
            ground: (0..count)
                .map(|t| renumber(&|k| slot(t, muscles + k)[0]))
                .collect(),
            friction: (0..count)
                .map(|t| renumber(&|k| slot(t, muscles + k)[1]))
                .collect(),
            broken: (0..count)
                .map(|t| {
                    let [lo, hi] = flat[t * stride + stride - 1];
                    u64::from(lo.to_bits()) | u64::from(hi.to_bits()) << 32
                })
                .collect(),
        }
    });
    Ok(Recording {
        frames,
        result,
        forces,
    })
}

fn worker_budget(logical: usize) -> usize {
    (logical / 2).clamp(1, 8)
}

fn rayon_thread_count(logical: usize, requested: Option<usize>) -> usize {
    let budget = worker_budget(logical);
    requested.filter(|&n| n > 0).unwrap_or(budget).min(budget)
}

fn logical_cpus() -> usize {
    std::thread::available_parallelism().map_or(2, usize::from)
}

/// General worker count (archive insertion, breeding, packing): half the
/// logical CPUs, at most eight. `RAYON_NUM_THREADS` can reduce it.
pub fn rayon_threads() -> usize {
    rayon_thread_count(
        logical_cpus(),
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok()),
    )
}

/// Best-effort worker priority reduction so evaluation yields to the desktop.
pub fn lower_thread_priority() {
    #[cfg(target_os = "linux")]
    // Linux applies nice to the calling thread when the id is zero.
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, 10);
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            #[link_name = "GetCurrentThread"]
            fn get_current_thread() -> *mut std::ffi::c_void;
            #[link_name = "SetThreadPriority"]
            fn set_thread_priority(thread: *mut std::ffi::c_void, priority: i32) -> i32;
        }
        // GetCurrentThread returns a pseudo-handle; it must not be closed.
        const THREAD_PRIORITY_LOWEST: i32 = -2;
        unsafe {
            set_thread_priority(get_current_thread(), THREAD_PRIORITY_LOWEST);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct WorkerFixture {
        engine: ThreadedEngine,
        jobs: mpsc::Receiver<(u64, Arc<Population>, Config)>,
        done: mpsc::Sender<Result<Finished, String>>,
    }

    impl WorkerFixture {
        fn new() -> Self {
            let (jobs, job_rx) = mpsc::channel();
            let (done_tx, done) = mpsc::channel();
            Self {
                engine: ThreadedEngine {
                    name: "test worker".into(),
                    max_nodes: 64,
                    depth: 3,
                    jobs: Some(jobs),
                    done,
                    thread: None,
                    queued: VecDeque::new(),
                    ready: VecDeque::new(),
                    failure: None,
                    next_ticket: 0,
                    allocated: Default::default(),
                    replays: None,
                },
                jobs: job_rx,
                done: done_tx,
            }
        }

        fn submit(&mut self) -> u64 {
            self.engine
                .submit(Population::default(), &Config::default())
                .unwrap()
        }
    }

    fn finished(ticket: u64) -> Finished {
        Finished {
            ticket,
            results: vec![GpuResult {
                fitness: ticket as f32 + 0.5,
                ..GpuResult::default()
            }],
            busy_seconds: 0.25,
        }
    }

    #[test]
    fn shared_submission_reaches_the_worker_without_copying_the_population() {
        let mut worker = WorkerFixture::new();
        let unit = Arc::new(Population::default());
        let cfg = Config {
            seed: 91,
            duration: 2.5,
            ..Config::default()
        };
        let engine: &mut dyn Engine = &mut worker.engine;
        let ticket = engine.submit_shared(Arc::clone(&unit), &cfg).unwrap();
        let (received_ticket, received_unit, received_cfg) = worker.jobs.try_recv().unwrap();
        assert_eq!(received_ticket, ticket);
        assert!(Arc::ptr_eq(&received_unit, &unit));
        assert_eq!(received_cfg.seed, 91);
        assert_eq!(received_cfg.duration, 2.5);
    }

    #[test]
    fn owned_submission_moves_population_storage_into_the_shared_unit() {
        let mut worker = WorkerFixture::new();
        let unit = Population {
            nodes: vec![crate::evolution::NodeGene {
                x: 1.25,
                y: 0.5,
                diameter: 0.1,
                friction: 0.8,
            }],
            ..Population::default()
        };
        let node_storage = unit.nodes.as_ptr();
        let engine: &mut dyn Engine = &mut worker.engine;
        let ticket = engine.submit(unit, &Config::default()).unwrap();
        let (received_ticket, received_unit, _) = worker.jobs.try_recv().unwrap();
        assert_eq!(received_ticket, ticket);
        assert_eq!(received_unit.nodes.as_ptr(), node_storage);
        assert_eq!(received_unit.nodes[0].x, 1.25);
    }

    #[test]
    fn wait_preserves_worker_errors_until_poll() {
        let mut worker = WorkerFixture::new();
        worker.submit();
        assert!(worker.done.send(Err("device lost".into())).is_ok());
        worker.engine.wait(Duration::from_millis(1));

        for _ in 0..2 {
            let error = worker.engine.poll().err().expect("worker failure was lost");
            assert!(error.to_string().contains("test worker"));
            assert!(error.to_string().contains("device lost"));
        }
        assert_eq!(worker.engine.free_slots(), 0);
        assert!(
            worker
                .engine
                .submit(Population::default(), &Config::default())
                .is_err()
        );
    }

    #[test]
    fn disconnected_worker_reports_unfinished_tickets() {
        for wait_first in [false, true] {
            let mut worker = WorkerFixture::new();
            worker.submit();
            drop(worker.done);
            if wait_first {
                worker.engine.wait(Duration::from_millis(1));
            }
            for _ in 0..2 {
                let error = worker
                    .engine
                    .poll()
                    .err()
                    .expect("worker disconnect was lost");
                assert!(error.to_string().contains("test worker"));
                assert!(error.to_string().contains("disconnect"));
            }
            assert_eq!(worker.engine.free_slots(), 0);
        }
    }

    #[test]
    fn completed_units_are_delivered_before_worker_error() {
        let mut worker = WorkerFixture::new();
        let completed = [worker.submit(), worker.submit()];
        worker.submit();
        for ticket in completed {
            assert!(worker.done.send(Ok(finished(ticket))).is_ok());
        }
        assert!(worker.done.send(Err("device lost".into())).is_ok());
        for ticket in completed {
            let result = worker.engine.poll().unwrap().unwrap();
            assert_eq!(result.ticket, ticket);
            assert_eq!(result.results[0].fitness, ticket as f32 + 0.5);
            assert_eq!(worker.engine.free_slots(), 0);
        }
        for _ in 0..2 {
            let error = worker.engine.poll().err().expect("worker failure was lost");
            assert!(error.to_string().contains("device lost"));
        }
    }

    #[test]
    fn completed_units_are_delivered_before_disconnect() {
        let mut worker = WorkerFixture::new();
        let completed = [worker.submit(), worker.submit()];
        for ticket in completed {
            assert!(worker.done.send(Ok(finished(ticket))).is_ok());
        }
        drop(worker.done);
        worker.engine.wait(Duration::from_millis(1));
        for ticket in completed {
            let result = worker.engine.poll().unwrap().unwrap();
            assert_eq!(result.ticket, ticket);
            assert_eq!(result.results[0].fitness, ticket as f32 + 0.5);
        }
        let error = worker
            .engine
            .poll()
            .err()
            .expect("worker disconnect was lost");
        assert!(error.to_string().contains("disconnect"));
        assert_eq!(worker.engine.free_slots(), 0);
    }

    #[test]
    fn disconnected_idle_worker_rejects_new_jobs() {
        let mut worker = WorkerFixture::new();
        drop(worker.done);
        assert!(
            worker
                .engine
                .submit(Population::default(), &Config::default())
                .is_err()
        );
        assert_eq!(worker.engine.free_slots(), 0);
        assert!(worker.jobs.try_recv().is_err());
    }

    #[test]
    fn failed_submission_preserves_worker_failure() {
        let mut worker = WorkerFixture::new();
        drop(worker.jobs);
        assert!(
            worker
                .engine
                .submit(Population::default(), &Config::default())
                .is_err()
        );
        let error = worker
            .engine
            .poll()
            .err()
            .expect("submission failure was lost");
        assert!(error.to_string().contains("test worker"));
        assert_eq!(worker.engine.free_slots(), 0);
    }

    #[test]
    fn failed_submission_keeps_results_arriving_after_the_initial_poll() {
        let mut worker = WorkerFixture::new();
        let ticket = worker.submit();
        worker.engine.receive(Duration::ZERO);
        // Model the worker finishing an older job and failing between the next
        // submission's initial poll and its failed send. Test the real handler
        // at that boundary so no thread timing determines the outcome.
        assert!(worker.done.send(Ok(finished(ticket))).is_ok());
        assert!(worker.done.send(Err("device lost".into())).is_ok());
        drop(worker.jobs);
        let error = worker.engine.submission_error();
        assert!(error.to_string().contains("device lost"));
        let result = worker.engine.poll().unwrap().unwrap();
        assert_eq!(result.ticket, ticket);
        assert_eq!(result.results[0].fitness, 0.5);
        let retained = worker.engine.poll().err().expect("worker failure was lost");
        assert_eq!(retained.to_string(), error.to_string());
        assert_eq!(worker.engine.free_slots(), 0);
    }

    #[test]
    fn healthy_worker_stays_available_after_empty_poll_and_wait() {
        let mut worker = WorkerFixture::new();
        assert!(worker.engine.poll().unwrap().is_none());
        worker.engine.wait(Duration::from_millis(1));
        assert!(worker.engine.poll().unwrap().is_none());
        assert_eq!(worker.engine.free_slots(), 3);
        let ticket = worker.submit();
        assert_eq!(worker.jobs.try_recv().unwrap().0, ticket);
    }

    /// The batches of each submission.
    type Layout = Vec<creature_kernel::LaneBatch>;
    /// Node stride and trial length of a recording.
    type Stretch = (usize, u32);

    /// A device that runs every unit at once and can fail submissions for
    /// lack of memory, following `script` (true fails; missing entries succeed).
    struct FakeDevice {
        slots: usize,
        script: VecDeque<bool>,
        /// Submissions in order, with (node stride, trial length) for a recording.
        in_flight: VecDeque<(u64, Layout, Option<Stretch>)>,
        next: u64,
        released: Arc<AtomicU64>,
        most_in_flight: Arc<AtomicU64>,
    }

    impl FakeDevice {
        fn new(slots: usize, script: &[bool]) -> Self {
            Self {
                slots,
                script: script.iter().copied().collect(),
                in_flight: VecDeque::new(),
                next: 0,
                released: Default::default(),
                most_in_flight: Default::default(),
            }
        }
    }

    /// The fake device's out-of-memory error.
    #[derive(Debug)]
    struct NoMemory;
    impl std::fmt::Display for NoMemory {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("out of fake GPU memory")
        }
    }
    impl std::error::Error for NoMemory {}

    impl Device for FakeDevice {
        fn free_slots(&self) -> usize {
            self.slots - self.in_flight.len()
        }
        fn submit(
            &mut self,
            batches: &mut Vec<creature_kernel::LaneBatch>,
            _cfg: &Config,
        ) -> Result<u64> {
            if self.script.pop_front().unwrap_or(false) {
                return Err(anyhow::Error::from(NoMemory).context("GPU buffers"));
            }
            let ticket = self.next;
            self.next += 1;
            self.in_flight
                .push_back((ticket, std::mem::take(batches), None));
            self.most_in_flight
                .fetch_max(self.in_flight.len() as u64, Ordering::Relaxed);
            Ok(ticket)
        }
        fn poll(&mut self, _timeout: Duration) -> Result<Option<Completed>> {
            Ok(self
                .in_flight
                .pop_front()
                .map(|(ticket, batches, recorded)| Completed {
                    ticket,
                    results: {
                        let count = batches.iter().map(|b| b.slots.len()).sum();
                        let mut results = vec![GpuResult::default(); count];
                        for b in &batches {
                            for (&slot, &i) in b.slots.iter().zip(&b.creatures) {
                                results[slot] = GpuResult {
                                    fitness: i as f32,
                                    fall_time: 1.0,
                                    ..GpuResult::default()
                                };
                            }
                        }
                        results
                    },
                    batches,
                    // Frame t puts node j at (t, j).
                    frames: recorded.map(|(stride, total)| {
                        (0..=total)
                            .flat_map(|t| (0..stride).map(move |j| [t as f32, j as f32]))
                            .collect()
                    }),
                    gpu_seconds: 0.001,
                }))
        }
        fn release_idle(&mut self) -> u64 {
            self.released.fetch_add(1, Ordering::Relaxed);
            4096
        }
        fn allocated_bytes(&self) -> u64 {
            0
        }
        fn replay_free(&self) -> bool {
            !self
                .in_flight
                .iter()
                .any(|(_, _, recorded)| recorded.is_some())
        }
        fn record(&mut self, batch: creature_kernel::LaneBatch, cfg: &Config) -> Result<u64> {
            let total = cfg.fidelity().settle() + cfg.steps();
            let ticket = self.next;
            self.next += 1;
            let stretch = (creature_kernel::frame_stride(&batch), total);
            self.in_flight
                .push_back((ticket, vec![batch], Some(stretch)));
            Ok(ticket)
        }
        fn out_of_memory(&self, error: &anyhow::Error) -> bool {
            error.chain().any(|cause| cause.is::<NoMemory>())
        }
    }

    /// Runs `units` jobs of `size` creatures through the GPU thread loop on
    /// `device` and returns what it sent back.
    fn run_fake(
        device: FakeDevice,
        units: usize,
        size: usize,
        limit: Duration,
    ) -> Vec<Result<Finished, String>> {
        let (jobs, job_rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        let cfg = Config {
            population: size,
            duration: 1.0,
            random_seed: false,
            ..Config::default()
        };
        let pop = Arc::new(crate::evolution::create(&cfg).unwrap());
        for ticket in 0..units as u64 {
            jobs.send((ticket, Arc::clone(&pop), cfg.clone())).unwrap();
        }
        drop(jobs);
        let memory = MemoryBackoff::new(device.slots, Duration::from_millis(2), limit);
        run_units(
            device,
            "fake GPU",
            job_rx,
            done_tx,
            None,
            &AtomicU64::new(0),
            memory,
        );
        done.try_iter().collect()
    }

    #[test]
    fn the_gpu_thread_records_replays_beside_evaluation() {
        let (jobs, job_rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        let (replays, replay_rx) = mpsc::channel();
        let cfg = Config {
            population: 8,
            duration: 1.0,
            random_seed: false,
            ..Config::default()
        };
        let pop = Arc::new(crate::evolution::create(&cfg).unwrap());
        let creature = pop.creature(3);
        let thread = std::thread::spawn(move || {
            let memory = MemoryBackoff::new(2, Duration::from_millis(2), Duration::from_secs(10));
            run_units(
                FakeDevice::new(2, &[]),
                "fake GPU",
                job_rx,
                done_tx,
                Some(replay_rx),
                &AtomicU64::new(0),
                memory,
            );
        });
        jobs.send((0, Arc::clone(&pop), cfg.clone())).unwrap();
        let (reply, answer) = mpsc::channel();
        replays
            .send(ReplayRequest {
                creature: creature.clone(),
                cfg: cfg.clone(),
                reply,
            })
            .unwrap();
        let recording = answer
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        let total = cfg.fidelity().settle() + cfg.steps();
        assert_eq!(recording.frames.len(), total as usize + 1);
        for (t, frame) in recording.frames.iter().enumerate() {
            assert_eq!(frame.len(), creature.nodes.len());
            for (j, position) in frame.iter().enumerate() {
                assert_eq!(*position, [t as f32, j as f32]);
            }
        }
        assert_eq!(recording.result.fall_time, 1.0);
        let finished = done.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
        assert_eq!(finished.ticket, 0);
        drop(jobs);
        thread.join().unwrap();
    }

    #[test]
    fn large_buffers_get_a_quarter_of_headroom_and_small_ones_a_power_of_two() {
        assert_eq!(padded_size(0), 256);
        assert_eq!(padded_size(300), 512);
        assert_eq!(padded_size(1 << 20), 1 << 20);
        let mib = 1u64 << 20;
        for size in [mib + 1, 3 * mib, 100 * mib + 7, 513 * mib] {
            let padded = padded_size(size);
            assert!(padded >= size + size / 4, "{size} -> {padded}");
            assert!(padded < size + size / 4 + mib, "{size} -> {padded}");
            assert_eq!(padded % mib, 0);
        }
        // A power of two would have taken 1024 MiB here.
        assert_eq!(padded_size(513 * mib), 642 * mib);
    }

    #[test]
    fn a_unit_that_runs_out_of_gpu_memory_waits_and_finishes() {
        let device = FakeDevice::new(4, &[true, true, true]);
        let released = Arc::clone(&device.released);
        let sent = run_fake(device, 1, 40, Duration::from_secs(10));
        assert_eq!(sent.len(), 1, "one finished unit and no failure");
        let finished = sent[0].as_ref().expect("the unit must not fail");
        assert_eq!(finished.ticket, 0);
        assert_eq!(finished.results.len(), 40);
        for (slot, result) in finished.results.iter().enumerate() {
            assert_eq!(result.fitness, slot as f32, "results stay in unit order");
        }
        assert_eq!(
            released.load(Ordering::Relaxed),
            3,
            "idle buffers freed per failure"
        );
    }

    #[test]
    fn a_unit_that_never_fits_fails_after_the_limit() {
        let device = FakeDevice::new(2, &[true; 10_000]);
        let sent = run_fake(device, 1, 8, Duration::from_millis(30));
        assert_eq!(sent.len(), 1);
        let error = sent[0]
            .as_ref()
            .err()
            .expect("the GPU must fail in the end");
        assert!(error.contains("no GPU memory"), "{error}");
    }

    #[test]
    fn units_run_fewer_at_once_when_memory_is_short() {
        // The second of four units fails while the first runs: it waits for
        // the first, and at most one unit runs at a time afterwards.
        let device = FakeDevice::new(4, &[false, true]);
        let most = Arc::clone(&device.most_in_flight);
        let sent = run_fake(device, 4, 12, Duration::from_secs(10));
        let tickets: Vec<u64> = sent
            .iter()
            .map(|done| done.as_ref().expect("no unit may fail").ticket)
            .collect();
        let mut sorted = tickets.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3]);
        assert_eq!(most.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn memory_backoff_bookkeeping() {
        let start = Instant::now();
        let second = Duration::from_secs(1);
        let mut memory = MemoryBackoff::new(4, Duration::from_millis(500), 60 * second);
        assert!(memory.may_submit(start, 3));
        assert!(!memory.may_submit(start, 4));

        // Out of memory with two running: wait for them, then run two at once.
        let line = memory.out_of_memory(start, 2, 0);
        assert!(matches!(line, OutOfMemory::Retry(Some(ref l)) if l.contains("2 of 4")));
        assert!(!memory.may_submit(start, 2));
        assert!(memory.may_submit(start, 1));
        assert_eq!(memory.out_of_memory(start, 1, 0), OutOfMemory::Retry(None));
        assert!(!memory.may_submit(start, 1));
        // One more is tried every RAISE_AFTER.
        let later = start + MemoryBackoff::RAISE_AFTER;
        assert!(memory.may_submit(later, 1));
        assert!(!memory.may_submit(later, 2));
        assert!(memory.may_submit(later + MemoryBackoff::RAISE_AFTER, 2));
        assert!(memory.may_submit(later + 2 * MemoryBackoff::RAISE_AFTER, 3));
        let line = memory.submitted(later + 2 * MemoryBackoff::RAISE_AFTER, 4);
        assert!(line.is_some_and(|l| l.contains("4 units")));

        // Nothing running: a quiet immediate retry after freeing buffers,
        // then announced pauses until the limit.
        let t = start + 100 * second;
        assert_eq!(memory.out_of_memory(t, 0, 1024), OutOfMemory::Retry(None));
        assert!(memory.may_submit(t, 0));
        let line = memory.out_of_memory(t, 0, 1024);
        assert!(matches!(line, OutOfMemory::Retry(Some(ref l)) if l.contains("waiting")));
        assert!(!memory.may_submit(t, 0));
        assert_eq!(memory.pause(t), Some(Duration::from_millis(500)));
        let t2 = t + Duration::from_millis(500);
        assert!(memory.may_submit(t2, 0));
        assert_eq!(memory.out_of_memory(t2, 0, 0), OutOfMemory::Retry(None));
        let line = memory.submitted(t2 + 2 * second, 1);
        assert!(line.is_some_and(|l| l.contains("available again after 2.5 s")));
        assert_eq!(memory.pause(t2), None);

        // A shortage that outlasts the limit fails.
        let t3 = t2 + 10 * second;
        assert_eq!(
            memory.out_of_memory(t3, 0, 0),
            OutOfMemory::Retry(Some(
                "out of GPU memory; freed cached buffers, waiting up to 60 s for memory".into()
            ))
        );
        assert_eq!(
            memory.out_of_memory(t3 + 61 * second, 0, 0),
            OutOfMemory::GiveUp(61 * second)
        );
    }

    #[test]
    fn general_workers_take_half_the_machine_at_most_eight() {
        for (logical, workers) in [(1, 1), (3, 1), (8, 4), (16, 8), (64, 8)] {
            assert_eq!(rayon_thread_count(logical, None), workers);
        }
        assert_eq!(rayon_thread_count(16, Some(3)), 3);
        assert_eq!(rayon_thread_count(16, Some(0)), 8);
        assert_eq!(rayon_thread_count(16, Some(usize::MAX)), 8);
    }
}
