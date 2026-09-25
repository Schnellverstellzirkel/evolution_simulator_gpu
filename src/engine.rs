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
    physics,
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
    next_ticket: u64,
    allocated: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl ThreadedEngine {
    fn receive(&mut self, timeout: Duration) -> Result<()> {
        let first = if timeout.is_zero() {
            self.done.try_recv().ok()
        } else {
            self.done.recv_timeout(timeout).ok()
        };
        for item in first.into_iter().chain(self.done.try_iter().collect::<Vec<_>>()) {
            self.ready
                .push_back(item.map_err(|e| anyhow::anyhow!("{} failed: {e}", self.name))?);
        }
        Ok(())
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
        self.depth.saturating_sub(self.queued.len())
    }
    fn submit(&mut self, unit: Population, cfg: &Config) -> Result<u64> {
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.jobs
            .as_ref()
            .context("Engine is shutting down")?
            .send((ticket, unit, cfg.clone()))
            .with_context(|| format!("{} stopped", self.name))?;
        self.queued.push_back(ticket);
        Ok(ticket)
    }
    fn poll(&mut self) -> Result<Option<Finished>> {
        self.receive(Duration::ZERO)?;
        let Some(done) = self.ready.pop_front() else {
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
            let _ = self.receive(timeout);
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
                        engine.submit(&packed, &cfg, physics::SETTLE + cfg.steps(), step_range)
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
    let device_name = ready_rx
        .recv()
        .context("GPU engine thread stopped")??;
    Ok(ThreadedEngine {
        name: device_name,
        max_nodes,
        depth: 3,
        jobs: Some(jobs),
        done,
        thread: Some(thread),
        queued: VecDeque::new(),
        ready: VecDeque::new(),
        next_ticket: 0,
        allocated,
    })
}

/// Starts a CPU engine on `threads` low-priority threads (see `cpu_engine`).
pub fn cpu_engine(threads: usize) -> Result<ThreadedEngine> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("cpu-eval-{i}"))
        // Evaluation yields to the UI, the compositor, and breeding.
        .start_handler(|_| unsafe {
            libc::setpriority(libc::PRIO_PROCESS, 0, 10);
        })
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
        name: format!(
            "CPU ({threads} threads, {}-lane SIMD)",
            crate::simd::LANES
        ),
        max_nodes: 64,
        depth: 2,
        jobs: Some(jobs),
        done,
        thread: Some(thread),
        queued: VecDeque::new(),
        ready: VecDeque::new(),
        next_ticket: 0,
        allocated: Default::default(),
    })
}
