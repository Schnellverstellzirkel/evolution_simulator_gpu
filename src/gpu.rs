//! Evaluation front end. All creature evaluation runs through the scheduler,
//! which routes work to the CUDA engines of the NVIDIA GPUs.
use crate::{config::Config, evolution::Population, qd::EvaluationMetrics, scheduler::Scheduler};
use anyhow::{Result, ensure};

pub struct Gpu {
    /// The names of the devices that evaluate.
    pub name: String,
    /// Bytes allocated on the GPU.
    pub allocated_bytes: u64,
    /// The scheduler for creature evaluation, or `None` if it failed to open.
    pub sched: Option<Scheduler>,
}

impl Gpu {
    /// Opens the named primary GPU plus the other evaluation engines. Fails
    /// when the primary GPU does not open.
    pub fn new(name: &str) -> Result<Self> {
        let sched = Scheduler::new(name)?;
        Ok(Self {
            name: sched.names(),
            allocated_bytes: 0,
            sched: Some(sched),
        })
    }
    /// Current evaluation backends, including changes after device recovery.
    pub fn names(&self) -> String {
        self.sched
            .as_ref()
            .map_or_else(|| self.name.clone(), Scheduler::names)
    }
    /// Evaluates creatures by index and returns their fitness values.
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
    /// Evaluates creatures by index and returns their evaluation metrics.
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
    /// Units on the evaluation engines now.
    pub fn on_engines(&self) -> usize {
        self.sched.as_ref().map_or(0, Scheduler::on_engines)
    }
}
