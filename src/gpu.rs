//! Evaluation front end. All creature evaluation runs through the scheduler,
//! which routes work to Vulkan GPUs and keeps CPU evaluation for CPU-only runs
//! or failover after GPU loss.
use crate::{config::Config, evolution::Population, qd::EvaluationMetrics, scheduler::Scheduler};
use anyhow::{Result, ensure};

/// Physics steps per GPU dispatch. Short ranges let display work interleave;
/// the Vulkan engine keeps them nearly free.
pub const DEFAULT_STEP_RANGE: u32 = 64;

pub struct Gpu {
    pub name: String,
    pub allocated_bytes: u64,
    pub sched: Option<Scheduler>,
    /// Why the primary GPU was not used, shown once when a session starts.
    pub startup_warning: Option<String>,
}

impl Gpu {
    /// Opens the named primary GPU plus the other evaluation engines. A
    /// primary that cannot open falls back to the CPU instead of failing.
    pub fn new(name: &str) -> Result<Self> {
        let sched = Scheduler::new(name)?;
        let startup_warning = sched.startup_failure().map(str::to_owned);
        Ok(Self {
            name: sched.names(),
            allocated_bytes: 0,
            sched: Some(sched),
            startup_warning,
        })
    }
    /// The UI passes its render device; evaluation opens its own devices.
    pub fn from_device(_device: wgpu::Device, _queue: wgpu::Queue, name: String) -> Result<Self> {
        Self::new(&name)
    }
    /// Current evaluation backends, including changes after device recovery.
    pub fn names(&self) -> String {
        self.sched
            .as_ref()
            .map_or_else(|| self.name.clone(), Scheduler::names)
    }
    pub fn evaluate(
        &mut self,
        pop: &Population,
        indices: &[usize],
        cfg: &Config,
    ) -> Result<Vec<f32>> {
        Ok(self
            .evaluate_with_metrics(pop, indices, cfg)?
            .into_iter()
            .map(|result| result.fitness)
            .collect())
    }
    pub fn evaluate_with_metrics(
        &mut self,
        pop: &Population,
        indices: &[usize],
        cfg: &Config,
    ) -> Result<Vec<EvaluationMetrics>> {
        ensure!(
            indices.iter().all(|&i| i < pop.genomes.len()),
            "Invalid creature index"
        );
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let sched = self.sched.as_mut().expect("scheduler");
        let metrics = sched.evaluate(pop, indices, cfg);
        self.name = sched.names();
        self.allocated_bytes = sched.allocated_bytes();
        metrics
    }
    /// True when evaluation can be queued without blocking.
    pub fn async_capable(&self) -> bool {
        self.sched.is_some()
    }
    pub fn async_in_flight(&self) -> usize {
        self.sched.as_ref().map_or(0, |s| s.in_flight())
    }
}
