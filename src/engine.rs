//! Common interface for every evaluation device (GPUs through Vulkan, CPU SIMD).
//!
//! An engine accepts units of creatures, evaluates them asynchronously on its
//! own thread, and returns raw per-creature results in unit order. Callers
//! hand over a standalone copy of the unit's creatures, so packing and
//! simulation never hold up the caller.
use crate::{
    config::Config,
    creature_kernel::{self, GpuResult},
    evolution::Population,
    vk_engine::VkEngine,
};
use anyhow::{Context, Result};
use std::{
    collections::VecDeque,
    sync::mpsc,
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
    fn submit(&mut self, unit: Population, cfg: &Config) -> Result<u64>;
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
    jobs: Option<mpsc::Sender<(u64, Population, Config)>>,
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
    fn submit(&mut self, unit: Population, cfg: &Config) -> Result<u64> {
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
        anyhow::ensure!(
            self.queued.pop_front() == Some(done.ticket),
            "{} results arrived out of order",
            self.name
        );
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

/// Opens a Vulkan GPU running the creature-per-lane kernel on its own thread.
/// The thread packs the next unit while earlier units run on the GPU.
pub fn gpu_engine(name: &str, max_nodes: usize, step_range: u32) -> Result<ThreadedEngine> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (jobs, job_rx) = mpsc::channel::<(u64, Population, Config)>();
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
            // Submitted units: ticket and length, oldest first.
            let mut running: VecDeque<(u64, u64, usize)> = VecDeque::new();
            let mut pending: Option<(u64, Population, Config)> = None;
            let mut open = true;
            loop {
                if pending.is_none() && open {
                    let job = if running.is_empty() {
                        job_rx.recv().map_err(|_| ())
                    } else {
                        job_rx.try_recv().map_err(|_| ())
                    };
                    match job {
                        Ok(job) => pending = Some(job),
                        Err(()) if running.is_empty() => open = false,
                        Err(()) => {}
                    }
                }
                if !open && running.is_empty() && pending.is_none() {
                    break;
                }
                if engine.free_slots() > 0
                    && let Some((ticket, unit, cfg)) = pending.take()
                {
                    let indices: Vec<usize> = (0..unit.genomes.len()).collect();
                    let submitted = creature_kernel::pack(&unit, &indices).and_then(|packed| {
                        engine.submit(
                            &packed,
                            &cfg,
                            cfg.fidelity().settle() + cfg.steps(),
                            step_range,
                        )
                    });
                    match submitted {
                        Ok(vk_ticket) => running.push_back((ticket, vk_ticket, indices.len())),
                        Err(err) => {
                            let _ = done_tx.send(Err(format!("{err:#}")));
                            return;
                        }
                    }
                    thread_allocated
                        .store(engine.allocated_bytes, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                if running.is_empty() {
                    continue;
                }
                // Wait briefly for the oldest submission, then check for new jobs.
                match engine.poll(Duration::from_millis(1)) {
                    Ok(Some(finished)) => {
                        let (ticket, vk_ticket, len) = running.pop_front().unwrap();
                        debug_assert_eq!(vk_ticket, finished.ticket);
                        let mut results = vec![GpuResult::default(); len];
                        for (slots, _, batch) in finished.batches {
                            for (slot, result) in slots.into_iter().zip(batch) {
                                results[slot] = result;
                            }
                        }
                        let message = Finished {
                            ticket,
                            results,
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
        })
        .context("GPU engine thread")?;
    let device_name = ready_rx.recv().context("GPU engine thread stopped")??;
    Ok(ThreadedEngine {
        name: device_name,
        max_nodes,
        depth: 3,
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
    // Breeding and packing need a general worker even during CPU evaluation.
    requested
        .unwrap_or(6)
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
    let (jobs, job_rx) = mpsc::channel::<(u64, Population, Config)>();
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
        jobs: mpsc::Receiver<(u64, Population, Config)>,
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
        for (logical, rayon, cpu) in [(1, 1, 0), (3, 1, 0), (8, 1, 3), (16, 2, 6), (64, 2, 6)] {
            assert_eq!(rayon_thread_count(logical, None, None), rayon);
            assert_eq!(cpu_thread_count(logical, None), cpu);
        }
    }

    #[test]
    fn thread_overrides_respect_the_desktop_budget() {
        assert_eq!(rayon_thread_count(16, Some(3), Some(6)), 2);
        assert_eq!(rayon_thread_count(16, Some(0), None), 2);
        assert_eq!(rayon_thread_count(16, Some(usize::MAX), None), 2);
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
