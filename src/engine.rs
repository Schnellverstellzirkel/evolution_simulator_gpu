//! Common interface for every evaluation device (GPUs through Vulkan, CPU SIMD).
//!
//! An engine accepts units of creatures, evaluates them asynchronously on its
//! own thread, and returns raw per-creature results in unit order. Callers
//! hand over an immutable population, shared with any retained submission,
//! so packing and simulation never hold up the caller.
use crate::{
    config::Config,
    creature_kernel::{self, GpuResult},
    evolution::Population,
    vk_engine::VkEngine,
};
use anyhow::{Context, Result};
use std::{
    collections::VecDeque,
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

pub struct Finished {
    pub ticket: u64,
    /// One result per creature, in unit order.
    pub results: Vec<GpuResult>,
    /// Device time spent on this unit.
    pub busy_seconds: f64,
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
}

impl ThreadedEngine {
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
fn segment_ends(cfg: &Config) -> Vec<u32> {
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
        // Screened creatures leave right after each screen step.
        .chain(cfg.screen.map(|screen| screen.tick(fidelity) + 1))
        .chain(
            cfg.screen
                .and_then(|screen| screen.second_tick(fidelity))
                .map(|tick| tick + 1),
        )
        .filter(|&tick| tick < total)
        .collect();
    ends.sort_unstable();
    ends.dedup();
    ends.push(total);
    ends
}

/// Opens a Vulkan GPU running the creature-per-lane kernel on its own thread.
/// The thread packs the next unit while earlier units run on the GPU.
pub fn gpu_engine(name: &str, max_nodes: usize, step_range: u32) -> Result<ThreadedEngine> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (jobs, job_rx) = mpsc::channel::<(u64, Arc<Population>, Config)>();
    let (done_tx, done) = mpsc::channel();
    let allocated = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let thread_allocated = allocated.clone();
    let device_name = name.to_owned();
    let thread = std::thread::Builder::new()
        .name(format!("gpu-{name}"))
        .spawn(move || {
            let mut engine = match VkEngine::new(&device_name, max_nodes) {
                Ok(engine) => {
                    let _ = ready_tx.send(Ok(engine.name.clone()));
                    engine
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            // Units whose next trial segment waits for a free slot (they go
            // before new jobs), and submitted segments by Vulkan ticket.
            let mut waiting: VecDeque<SegmentedUnit> = VecDeque::new();
            let mut running: Vec<(u64, SegmentedUnit)> = Vec::new();
            let mut pending: Option<(u64, Arc<Population>, Config)> = None;
            let mut open = true;
            loop {
                if pending.is_none() && open {
                    let job = if running.is_empty() && waiting.is_empty() {
                        job_rx.recv().map_err(|_| ())
                    } else {
                        job_rx.try_recv().map_err(|_| ())
                    };
                    match job {
                        Ok(job) => pending = Some(job),
                        Err(()) if running.is_empty() && waiting.is_empty() => open = false,
                        Err(()) => {}
                    }
                }
                if !open && running.is_empty() && waiting.is_empty() && pending.is_none() {
                    break;
                }
                if engine.free_slots() > 0 {
                    let next = match waiting.pop_front() {
                        Some(unit) => Some(Ok(unit)),
                        None => pending.take().map(|(ticket, unit, cfg)| {
                            let indices: Vec<usize> = (0..unit.genomes.len()).collect();
                            creature_kernel::pack(&unit, &indices).map(|batches| SegmentedUnit {
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
                        let submitted = next.and_then(|unit| {
                            let total = *unit.ends.last().expect("segment ends");
                            let start = unit.segment.checked_sub(1).map_or(0, |s| unit.ends[s]);
                            let end = unit.ends[unit.segment];
                            engine
                                .submit(
                                    &unit.batches,
                                    &unit.cfg,
                                    start,
                                    end,
                                    total,
                                    step_range,
                                    end < total,
                                )
                                .map(|vk_ticket| {
                                    let mut unit = unit;
                                    for batch in &mut unit.batches {
                                        batch.release_uploaded();
                                    }
                                    (vk_ticket, unit)
                                })
                        });
                        match submitted {
                            Ok(entry) => running.push(entry),
                            Err(err) => {
                                let _ = done_tx.send(Err(format!("{err:#}")));
                                return;
                            }
                        }
                        thread_allocated
                            .store(engine.allocated_bytes, std::sync::atomic::Ordering::Relaxed);
                        continue;
                    }
                }
                if running.is_empty() {
                    continue;
                }
                // Wait briefly for any submission, then check for new jobs.
                match engine.poll(Duration::from_millis(1)) {
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
        })
        .context("GPU engine thread")?;
    let device_name = ready_rx.recv().context("GPU engine thread stopped")??;
    Ok(ThreadedEngine {
        name: device_name,
        max_nodes,
        // One unit packs on the engine thread while every slot runs.
        depth: crate::vk_engine::gpu_slots() as usize + 1,
        jobs: Some(jobs),
        done,
        thread: Some(thread),
        queued: VecDeque::new(),
        ready: VecDeque::new(),
        failure: None,
        next_ticket: 0,
        allocated,
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
