//! Common interface for every evaluation device (GPUs through Vulkan, CPU SIMD).
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
    vk_engine::{Completed, VkEngine},
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
            eprintln!("GPU replay failed, the CPU replays instead: {error}");
            None
        }
        Err(_) => {
            eprintln!("GPU replay did not answer in time; the CPU replays instead");
            None
        }
    }
}

/// A creature's full trial for the replay viewer and the result scored in
/// the same run, from the engine that scores the archive: the GPU when one
/// evaluates, the CPU engine in a CPU-only game. A replay runs the full
/// trial, without the early screen.
pub fn replay(creature: &Creature, cfg: &Config) -> (Vec<Vec<[f32; 2]>>, GpuResult) {
    let (frames, result, _) = replay_forces(creature, cfg);
    (frames, result)
}

/// `replay` with the muscle energy, muscle force and ground contact forces
/// the engine recorded with each frame.
pub fn replay_forces(
    creature: &Creature,
    cfg: &Config,
) -> (
    Vec<Vec<[f32; 2]>>,
    GpuResult,
    Option<crate::replay_forces::Forces>,
) {
    let cfg = Config {
        screen: None,
        ..cfg.clone()
    };
    if let Some(recording) = record_on_gpu(creature, &cfg, Duration::from_secs(3)) {
        return (recording.frames, recording.result, recording.forces);
    }
    let (frames, result, forces) = crate::physics2::replay_forces(creature, &cfg);
    (frames, result, Some(forces))
}

pub trait Engine: Send {
    fn name(&self) -> String;
    /// Largest body (in nodes) this engine can evaluate.
    fn max_nodes(&self) -> usize;
    /// Units that can be queued now without waiting.
    fn free_slots(&self) -> usize;
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
    queued: VecDeque<u64>,
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
            self.depth.saturating_sub(self.queued.len())
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
        self.queued.push_back(ticket);
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
        let position = self.queued.iter().position(|&ticket| ticket == done.ticket);
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
        // Join before the process tears down Vulkan and the thread pools.
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A unit on its way through the trial segments of `segment_ends`.
struct SegmentedUnit {
    ticket: u64,
    /// Final results by unit position, filled as creatures finish.
    results: Vec<GpuResult>,
    ends: Vec<u32>,
    /// Index into `ends` of the segment that runs next.
    segment: usize,
    cfg: Config,
    busy: f64,
    /// The creatures still running, packed for the next segment.
    batches: Vec<creature_kernel::LaneBatch>,
}

/// Ticks at which a GPU trial pauses to drop fallen creatures, ending with
/// the trial's last tick. A fall ends a trial, so every step a fallen
/// creature would take after it is wasted; at a segment boundary the others
/// are repacked into dense warps. `EVOLUTION_SEGMENTS` lists the pauses in
/// seconds after settling (default `2,10`; empty or `0` for none). On an
/// evolved 3M population 38% of creatures fall, most within a second, and
/// pauses at 2 s and 10 s skip 34% of all steps.
pub(crate) fn segment_ends(cfg: &Config) -> Vec<u32> {
    let fidelity = cfg.fidelity();
    let total = fidelity.settle() + cfg.steps();
    let seconds: Vec<f32> = match std::env::var("EVOLUTION_SEGMENTS") {
        Ok(list) => list
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .filter(|&v: &f32| v > 0.0)
            .collect(),
        Err(_) => vec![2.0, 10.0],
    };
    let mut ends: Vec<u32> = seconds
        .into_iter()
        .map(|s| fidelity.settle() + (s * fidelity.rate as f32).round() as u32)
        // Screened creatures leave right after the screen step.
        .chain(cfg.screen.map(|screen| screen.tick(fidelity) + 1))
        .filter(|&tick| tick < total)
        .collect();
    ends.sort_unstable();
    ends.dedup();
    ends.push(total);
    ends
}

/// What the GPU engine thread needs from a device. `VkEngine` and
/// `CudaEngine` implement it; tests use a fake that can run out of memory.
trait SegmentDevice {
    fn free_slots(&self) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &mut self,
        batches: &[creature_kernel::LaneBatch],
        cfg: &Config,
        start: u32,
        end: u32,
        total: u32,
        chunk: u32,
        read_state: bool,
    ) -> Result<u64>;
    fn poll(&mut self, timeout: Duration) -> Result<Option<Completed>>;
    /// Frees buffers kept for reuse by slots with nothing in flight.
    fn release_idle(&mut self) -> u64;
    fn allocated_bytes(&self) -> u64;
    /// Whether a replay can be recorded now.
    fn replay_free(&self) -> bool;
    /// Queues a whole recorded trial of one batch (`VkEngine::record`).
    fn record(
        &mut self,
        batch: &creature_kernel::LaneBatch,
        cfg: &Config,
        total: u32,
        chunk: u32,
    ) -> Result<u64>;
}

impl SegmentDevice for VkEngine {
    fn free_slots(&self) -> usize {
        VkEngine::free_slots(self)
    }
    fn submit(
        &mut self,
        batches: &[creature_kernel::LaneBatch],
        cfg: &Config,
        start: u32,
        end: u32,
        total: u32,
        chunk: u32,
        read_state: bool,
    ) -> Result<u64> {
        VkEngine::submit(self, batches, cfg, start, end, total, chunk, read_state)
    }
    fn poll(&mut self, timeout: Duration) -> Result<Option<Completed>> {
        VkEngine::poll(self, timeout)
    }
    fn release_idle(&mut self) -> u64 {
        VkEngine::release_idle(self)
    }
    fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    fn replay_free(&self) -> bool {
        VkEngine::replay_free(self)
    }
    fn record(
        &mut self,
        batch: &creature_kernel::LaneBatch,
        cfg: &Config,
        total: u32,
        chunk: u32,
    ) -> Result<u64> {
        VkEngine::record(self, batch, cfg, total, chunk)
    }
}

impl SegmentDevice for CudaEngine {
    fn free_slots(&self) -> usize {
        CudaEngine::free_slots(self)
    }
    fn submit(
        &mut self,
        batches: &[creature_kernel::LaneBatch],
        cfg: &Config,
        start: u32,
        end: u32,
        total: u32,
        chunk: u32,
        read_state: bool,
    ) -> Result<u64> {
        CudaEngine::submit(self, batches, cfg, start, end, total, chunk, read_state)
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
    fn record(
        &mut self,
        batch: &creature_kernel::LaneBatch,
        cfg: &Config,
        total: u32,
        chunk: u32,
    ) -> Result<u64> {
        CudaEngine::record(self, batch, cfg, total, chunk)
    }
}

/// What to do after a submission ran out of memory.
#[derive(Debug, PartialEq)]
enum OutOfMemory {
    /// Keep the unit and try again later; print the line if there is one.
    Retry(Option<String>),
    /// Memory stayed short for this long with nothing running: fail.
    GiveUp(Duration),
}

/// Rides out failed GPU memory allocations instead of retiring the GPU,
/// which would leave the rest of the session on the CPU. Another process
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

/// The GPU a `gpu_engine` thread runs on.
enum Backend {
    Cuda(Box<CudaEngine>),
    Vulkan(Box<VkEngine>),
}

/// Opens the GPU named `name` through CUDA when it is an NVIDIA GPU whose
/// driver and NVRTC load (1.6 to 1.8 times Vulkan's kernel rate on the
/// RTX 4060, docs/performance-log.md), and through Vulkan otherwise.
fn open_backend(name: &str, max_nodes: usize) -> Result<(Backend, String)> {
    if crate::cuda_engine::enabled() {
        match CudaEngine::new(name, max_nodes) {
            Ok(engine) => {
                let name = engine.name.clone();
                return Ok((Backend::Cuda(Box::new(engine)), name));
            }
            Err(error) => eprintln!("CUDA not used ({error:#}); running on Vulkan"),
        }
    }
    let engine = VkEngine::new(name, max_nodes)?;
    let name = engine.name.clone();
    Ok((Backend::Vulkan(Box::new(engine)), name))
}

/// Opens a GPU running the creature-per-lane kernel on its own thread.
/// The thread packs the next unit while earlier units run on the GPU.
pub fn gpu_engine(name: &str, max_nodes: usize, step_range: u32) -> Result<ThreadedEngine> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (jobs, job_rx) = mpsc::channel::<(u64, Arc<Population>, Config)>();
    let (done_tx, done) = mpsc::channel();
    let (replays, replay_rx) = mpsc::channel::<ReplayRequest>();
    let allocated = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let thread_allocated = allocated.clone();
    let device_name = name.to_owned();
    let slots = crate::vk_engine::gpu_slots() as usize;
    let thread = std::thread::Builder::new()
        .name(format!("gpu-{name}"))
        .spawn(move || {
            let (engine, name) = match open_backend(&device_name, max_nodes) {
                Ok(opened) => {
                    let _ = ready_tx.send(Ok(opened.1.clone()));
                    opened
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let memory =
                MemoryBackoff::new(slots, Duration::from_millis(500), Duration::from_secs(60));
            match engine {
                // Each backend records replays with the recording variant of
                // the kernel that scores.
                Backend::Cuda(engine) => run_segments(
                    *engine,
                    &name,
                    job_rx,
                    done_tx,
                    Some(replay_rx),
                    &thread_allocated,
                    step_range,
                    memory,
                ),
                Backend::Vulkan(engine) => run_segments(
                    *engine,
                    &name,
                    job_rx,
                    done_tx,
                    Some(replay_rx),
                    &thread_allocated,
                    step_range,
                    memory,
                ),
            }
        })
        .context("GPU engine thread")?;
    let device_name = ready_rx.recv().context("GPU engine thread stopped")??;
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

/// The GPU engine thread: packs jobs, runs them in trial segments, and
/// returns finished units until the job channel closes.
#[allow(clippy::too_many_arguments)]
fn run_segments<D: SegmentDevice>(
    mut engine: D,
    name: &str,
    job_rx: mpsc::Receiver<(u64, Arc<Population>, Config)>,
    done_tx: mpsc::Sender<Result<Finished, String>>,
    replays: Option<mpsc::Receiver<ReplayRequest>>,
    allocated: &AtomicU64,
    step_range: u32,
    mut memory: MemoryBackoff,
) {
    let mut recording: Option<InFlightReplay> = None;
    // Units whose next trial segment waits for a free slot (they go before
    // new jobs), and submitted segments by device ticket.
    let mut waiting: VecDeque<SegmentedUnit> = VecDeque::new();
    let mut running: Vec<(u64, SegmentedUnit)> = Vec::new();
    let mut pending: Option<(u64, Arc<Population>, Config)> = None;
    let mut open = true;
    loop {
        if pending.is_none() && open {
            let idle = running.is_empty() && waiting.is_empty() && recording.is_none();
            // While idle, wake every few milliseconds for replay requests.
            let job = match (idle, &replays) {
                (true, Some(_)) => job_rx.recv_timeout(Duration::from_millis(5)),
                (true, None) => job_rx
                    .recv()
                    .map_err(|_| mpsc::RecvTimeoutError::Disconnected),
                (false, _) => job_rx.try_recv().map_err(|error| match error {
                    mpsc::TryRecvError::Empty => mpsc::RecvTimeoutError::Timeout,
                    mpsc::TryRecvError::Disconnected => mpsc::RecvTimeoutError::Disconnected,
                }),
            };
            match job {
                Ok(job) => pending = Some(job),
                Err(mpsc::RecvTimeoutError::Disconnected) if idle => open = false,
                Err(_) => {}
            }
        }
        if !open
            && running.is_empty()
            && waiting.is_empty()
            && pending.is_none()
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
            match start_recording(&mut engine, &request, step_range) {
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
            allocated.store(engine.allocated_bytes(), Ordering::Relaxed);
        }
        if engine.free_slots() > 0 && memory.may_submit(Instant::now(), running.len()) {
            let next = match waiting.pop_front() {
                Some(unit) => Some(Ok(unit)),
                None => pending.take().map(|(ticket, unit, cfg)| {
                    let indices: Vec<usize> = (0..unit.genomes.len()).collect();
                    crate::physics2::pack(&unit, &indices, &cfg).map(|batches| SegmentedUnit {
                        ticket,
                        results: vec![GpuResult::default(); indices.len()],
                        ends: segment_ends(&cfg),
                        segment: 0,
                        cfg,
                        busy: 0.0,
                        batches,
                    })
                }),
            };
            if let Some(next) = next {
                let mut unit = match next {
                    Ok(unit) => unit,
                    Err(err) => {
                        let _ = done_tx.send(Err(format!("{err:#}")));
                        return;
                    }
                };
                let total = *unit.ends.last().expect("segment ends");
                // There is no settling phase: trials start at the settling tick.
                let first = unit.cfg.fidelity().settle();
                let start = unit.segment.checked_sub(1).map_or(first, |s| unit.ends[s]);
                let end = unit.ends[unit.segment];
                match engine.submit(
                    &unit.batches,
                    &unit.cfg,
                    start,
                    end,
                    total,
                    step_range,
                    end < total,
                ) {
                    Ok(vk_ticket) => {
                        for batch in &mut unit.batches {
                            batch.release_uploaded();
                        }
                        running.push((vk_ticket, unit));
                        if let Some(line) = memory.submitted(Instant::now(), running.len()) {
                            eprintln!("{name}: {line}");
                        }
                    }
                    // Keep the unit; it runs once memory frees up.
                    Err(err) if crate::vk_engine::out_of_memory(&err) => {
                        let freed = engine.release_idle();
                        waiting.push_front(unit);
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
                allocated.store(engine.allocated_bytes(), Ordering::Relaxed);
                continue;
            }
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
                // Units on separate queues can finish out of order.
                let Some(position) = running
                    .iter()
                    .position(|(vk_ticket, _)| *vk_ticket == finished.ticket)
                else {
                    let _ = done_tx.send(Err("unknown GPU submission finished".into()));
                    return;
                };
                let (_, mut unit) = running.swap_remove(position);
                unit.busy += finished.gpu_seconds;
                let last = unit.segment + 1 == unit.ends.len();
                // Fallen creatures are final; the rest continue in the
                // next segment, repacked into dense warps.
                let mut next = Vec::new();
                for (b, (slots, _, results)) in finished.batches.iter().enumerate() {
                    let mut keep = Vec::new();
                    for (j, result) in results.iter().enumerate() {
                        if last || result.fall_time > 0.0 || result.screened > 0.0 {
                            unit.results[slots[j]] = *result;
                        } else {
                            keep.push(j);
                        }
                    }
                    if !keep.is_empty() {
                        let Some((nodes, muscles)) =
                            finished.state.as_ref().and_then(|state| state.get(b))
                        else {
                            let _ = done_tx.send(Err("GPU segment state missing".into()));
                            return;
                        };
                        next.push(unit.batches[b].repack(&keep, nodes, muscles, results));
                    }
                }
                if next.is_empty() {
                    let message = Finished {
                        ticket: unit.ticket,
                        results: std::mem::take(&mut unit.results),
                        busy_seconds: unit.busy,
                    };
                    if done_tx.send(Ok(message)).is_err() {
                        return;
                    }
                } else {
                    unit.batches = next;
                    unit.segment += 1;
                    waiting.push_front(unit);
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
/// a (normal, friction) contact force per node.
#[derive(Clone, Copy)]
struct FrameLayout {
    nodes: usize,
    capacity: usize,
    muscles: usize,
    stride: usize,
}

/// Packs a replay request's creature and queues its recording. Returns the
/// device ticket, the frame layout and the trial length.
fn start_recording<D: SegmentDevice>(
    engine: &mut D,
    request: &ReplayRequest,
    step_range: u32,
) -> Result<(u64, FrameLayout, u32)> {
    let mut population = Population::default();
    population.push(request.creature.clone());
    let batches = crate::physics2::pack(&population, &[0], &request.cfg)?;
    anyhow::ensure!(batches.len() == 1, "A replay packs into one batch");
    let fidelity = request.cfg.fidelity();
    let total = fidelity.settle() + request.cfg.steps();
    let batch = &batches[0];
    let ticket = engine.record(batch, &request.cfg, total, step_range)?;
    let layout = FrameLayout {
        nodes: request.creature.nodes.len(),
        capacity: batch.capacity,
        muscles: batch.info.first().map_or(0, |i| i[2] as usize),
        stride: creature_kernel::frame_stride(batch),
    };
    Ok((ticket, layout, total))
}

/// The replay in a finished recording: one frame per step and one after the
/// last, each with the body's node positions.
fn recorded(finished: &Completed, layout: FrameLayout, total: u32) -> Result<Recording, String> {
    let FrameLayout {
        nodes,
        capacity,
        muscles,
        stride,
    } = layout;
    let flat = finished
        .frames
        .as_ref()
        .ok_or("the recording returned no frames")?;
    let result = *finished
        .batches
        .first()
        .and_then(|(_, _, results)| results.first())
        .ok_or("the recording returned no result")?;
    let count = total as usize + 1;
    if flat.len() < count * stride {
        return Err("the recording returned too few frames".into());
    }
    let frames = (0..count)
        .map(|t| flat[t * stride..t * stride + nodes].to_vec())
        .collect();
    let forces = (stride >= capacity + muscles + nodes && stride > capacity).then(|| {
        let slot = |t: usize, k: usize| flat[t * stride + capacity + k];
        crate::replay_forces::Forces {
            energy: (0..count)
                .map(|t| (0..muscles).map(|k| slot(t, k)[0]).collect())
                .collect(),
            muscle: (0..count)
                .map(|t| (0..muscles).map(|k| slot(t, k)[1]).collect())
                .collect(),
            ground: (0..count)
                .map(|t| (0..nodes).map(|i| slot(t, muscles + i)[0]).collect())
                .collect(),
            friction: (0..count)
                .map(|t| (0..nodes).map(|i| slot(t, muscles + i)[1]).collect())
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

fn rayon_thread_count(
    logical: usize,
    requested: Option<usize>,
    cpu_requested: Option<usize>,
) -> usize {
    let remaining = worker_budget(logical) - cpu_thread_count(logical, cpu_requested);
    requested
        .filter(|&n| n > 0)
        .unwrap_or(remaining)
        .min(remaining)
}

fn cpu_thread_count(logical: usize, requested: Option<usize>) -> usize {
    // By default the CPU evaluates nothing beside the GPU: at 3M creatures a
    // separate six-thread pool slowed the game (56k against 64k creatures/s
    // end to end) because archive insertion and breeding lost their threads.
    // Breeding and packing need a general worker even during CPU evaluation.
    requested
        .unwrap_or(0)
        .min(worker_budget(logical).saturating_sub(1))
}

fn logical_cpus() -> usize {
    std::thread::available_parallelism().map_or(2, usize::from)
}

fn thread_override(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Breeding and general worker count after reserving the CPU evaluation workers.
/// Both pools share at most eight threads and half the logical CPUs.
/// `RAYON_NUM_THREADS` can reduce, but cannot exceed, the remaining budget.
pub fn rayon_threads() -> usize {
    rayon_thread_count(
        logical_cpus(),
        thread_override("RAYON_NUM_THREADS"),
        thread_override("EVOLUTION_CPU_THREADS"),
    )
}

/// Evaluation workers, defaulting to six while leaving one general worker.
/// `EVOLUTION_CPU_THREADS=0` or a one-worker budget disables the scheduler's CPU
/// engine, leaving the general pool available for breeding and packing.
pub fn cpu_threads() -> usize {
    cpu_thread_count(logical_cpus(), thread_override("EVOLUTION_CPU_THREADS"))
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

/// A CPU engine that evaluates on the general Rayon pool instead of starting
/// its own. Used when the primary GPU cannot open and no separate CPU
/// evaluation pool is configured: evaluation then shares the breeding pool.
pub fn cpu_engine_shared() -> Result<ThreadedEngine> {
    let threads = rayon::current_num_threads();
    let (jobs, job_rx) = mpsc::channel::<(u64, Arc<Population>, Config)>();
    let (done_tx, done) = mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("cpu-eval-shared".into())
        .spawn(move || {
            lower_thread_priority();
            for (ticket, unit, cfg) in job_rx {
                let started = Instant::now();
                let results = crate::cpu_engine::evaluate(&unit, &cfg);
                let message = Finished {
                    ticket,
                    results,
                    busy_seconds: started.elapsed().as_secs_f64(),
                };
                if done_tx.send(Ok(message)).is_err() {
                    break;
                }
            }
        })
        .context("shared CPU evaluation dispatcher")?;
    Ok(ThreadedEngine {
        name: format!(
            "CPU (general pool, {threads} threads, {}-lane SIMD)",
            crate::simd::LANES
        ),
        max_nodes: 64,
        depth: 2,
        jobs: Some(jobs),
        done,
        thread: Some(thread),
        queued: VecDeque::new(),
        ready: VecDeque::new(),
        failure: None,
        next_ticket: 0,
        allocated: Default::default(),
        replays: None,
    })
}

/// Starts a CPU engine on low-priority threads, leaving one general worker in
/// the shared budget. Explicit CPU-only callers can still run one evaluation
/// worker when the budget is one; the scheduler disables that extra pool.
pub fn cpu_engine(threads: usize) -> Result<ThreadedEngine> {
    let threads = cpu_thread_count(logical_cpus(), Some(threads)).max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("cpu-eval-{i}"))
        // Evaluation yields to the UI, the compositor, and breeding.
        .start_handler(|_| lower_thread_priority())
        .build()
        .context("CPU evaluation thread pool")?;
    let (jobs, job_rx) = mpsc::channel::<(u64, Arc<Population>, Config)>();
    let (done_tx, done) = mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("cpu-eval-dispatch".into())
        .spawn(move || {
            for (ticket, unit, cfg) in job_rx {
                let started = Instant::now();
                let results = pool.install(|| crate::cpu_engine::evaluate(&unit, &cfg));
                let message = Finished {
                    ticket,
                    results,
                    busy_seconds: started.elapsed().as_secs_f64(),
                };
                if done_tx.send(Ok(message)).is_err() {
                    break;
                }
            }
        })
        .context("CPU evaluation dispatcher")?;
    Ok(ThreadedEngine {
        name: format!("CPU ({threads} threads, {}-lane SIMD)", crate::simd::LANES),
        max_nodes: 64,
        depth: 2,
        jobs: Some(jobs),
        done,
        thread: Some(thread),
        queued: VecDeque::new(),
        ready: VecDeque::new(),
        failure: None,
        next_ticket: 0,
        allocated: Default::default(),
        replays: None,
    })
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
    fn shared_cpu_engine_evaluates_units_on_the_general_pool() {
        let mut engine = cpu_engine_shared().unwrap();
        let cfg = Config {
            population: 4,
            duration: 0.1,
            random_seed: false,
            ..Config::default()
        };
        let pop = Arc::new(crate::evolution::create(&cfg).unwrap());
        let ticket = engine.submit_shared(Arc::clone(&pop), &cfg).unwrap();
        let done = loop {
            if let Some(done) = engine.poll().unwrap() {
                break done;
            }
            engine.wait(Duration::from_millis(20));
        };
        assert_eq!(done.ticket, ticket);
        assert_eq!(done.results.len(), cfg.population);
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

    /// Slice positions and population indices of each submitted batch.
    type Layout = Vec<(Vec<usize>, Vec<usize>)>;
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

    impl SegmentDevice for FakeDevice {
        fn free_slots(&self) -> usize {
            self.slots - self.in_flight.len()
        }
        fn submit(
            &mut self,
            batches: &[creature_kernel::LaneBatch],
            _cfg: &Config,
            _start: u32,
            _end: u32,
            _total: u32,
            _chunk: u32,
            _read_state: bool,
        ) -> Result<u64> {
            if self.script.pop_front().unwrap_or(false) {
                return Err(
                    anyhow::Error::from(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
                        .context("GPU buffers"),
                );
            }
            let ticket = self.next;
            self.next += 1;
            let layout = batches
                .iter()
                .map(|b| (b.slots.clone(), b.creatures.clone()))
                .collect();
            self.in_flight.push_back((ticket, layout, None));
            self.most_in_flight
                .fetch_max(self.in_flight.len() as u64, Ordering::Relaxed);
            Ok(ticket)
        }
        fn poll(&mut self, _timeout: Duration) -> Result<Option<Completed>> {
            // Every creature falls, so each unit finishes in one segment.
            Ok(self
                .in_flight
                .pop_front()
                .map(|(ticket, layout, recorded)| Completed {
                    ticket,
                    batches: layout
                        .into_iter()
                        .map(|(slots, creatures)| {
                            let results = creatures
                                .iter()
                                .map(|&i| GpuResult {
                                    fitness: i as f32,
                                    fall_time: 1.0,
                                    ..GpuResult::default()
                                })
                                .collect();
                            (slots, creatures, results)
                        })
                        .collect(),
                    state: None,
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
        fn record(
            &mut self,
            batch: &creature_kernel::LaneBatch,
            _cfg: &Config,
            total: u32,
            _chunk: u32,
        ) -> Result<u64> {
            let ticket = self.next;
            self.next += 1;
            let layout = vec![(batch.slots.clone(), batch.creatures.clone())];
            self.in_flight.push_back((
                ticket,
                layout,
                Some((creature_kernel::frame_stride(batch), total)),
            ));
            Ok(ticket)
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
        run_segments(
            device,
            "fake GPU",
            job_rx,
            done_tx,
            None,
            &AtomicU64::new(0),
            64,
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
            run_segments(
                FakeDevice::new(2, &[]),
                "fake GPU",
                job_rx,
                done_tx,
                Some(replay_rx),
                &AtomicU64::new(0),
                64,
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
    fn out_of_memory_is_recognized_through_error_context() {
        use ash::vk;
        for code in [
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ] {
            let error = anyhow::Error::from(code).context("GPU buffers");
            assert!(crate::vk_engine::out_of_memory(&error));
            assert!(crate::vk_engine::out_of_memory(&anyhow::Error::from(code)));
        }
        let lost = anyhow::Error::from(vk::Result::ERROR_DEVICE_LOST).context("fence");
        assert!(!crate::vk_engine::out_of_memory(&lost));
        let text = anyhow::anyhow!("A device memory allocation has failed");
        assert!(!crate::vk_engine::out_of_memory(&text));
    }

    #[test]
    fn large_buffers_get_a_quarter_of_headroom_and_small_ones_a_power_of_two() {
        use crate::vk_engine::padded_size;
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
    fn worker_defaults_leave_half_the_machine_free() {
        for (logical, rayon, cpu) in [(1, 1, 0), (3, 1, 0), (8, 4, 0), (16, 8, 0), (64, 8, 0)] {
            assert_eq!(rayon_thread_count(logical, None, None), rayon);
            assert_eq!(cpu_thread_count(logical, None), cpu);
        }
    }

    #[test]
    fn thread_overrides_respect_the_desktop_budget() {
        assert_eq!(rayon_thread_count(16, Some(3), Some(6)), 2);
        assert_eq!(rayon_thread_count(16, Some(0), None), 8);
        assert_eq!(rayon_thread_count(16, Some(usize::MAX), None), 8);
        assert_eq!(rayon_thread_count(16, None, Some(0)), 8);
        assert_eq!(rayon_thread_count(16, Some(3), Some(0)), 3);
        assert_eq!(rayon_thread_count(16, Some(3), Some(2)), 3);
        assert_eq!(cpu_thread_count(16, Some(0)), 0);
        assert_eq!(cpu_thread_count(16, Some(2)), 2);
        assert_eq!(cpu_thread_count(8, Some(6)), 3);
        assert_eq!(cpu_thread_count(16, Some(usize::MAX)), 7);
        assert_eq!(cpu_thread_count(1, Some(6)), 0);
    }

    #[test]
    fn concurrent_worker_pools_share_one_budget() {
        for logical in [1, 2, 3, 4, 8, 12, 16, 32, 64] {
            for cpu_request in [None, Some(0), Some(1), Some(6), Some(usize::MAX)] {
                for rayon_request in [None, Some(0), Some(1), Some(8), Some(usize::MAX)] {
                    let cpu = cpu_thread_count(logical, cpu_request);
                    let rayon = rayon_thread_count(logical, rayon_request, cpu_request);
                    assert!(rayon >= 1, "general workers must make progress");
                    assert!(
                        cpu + rayon <= 8 && cpu + rayon <= (logical / 2).max(1),
                        "{logical} logical CPUs: {cpu} evaluation + {rayon} general workers"
                    );
                }
            }
        }
    }
}
