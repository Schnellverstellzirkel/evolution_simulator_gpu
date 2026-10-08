//! Evaluation front end. All creature evaluation runs through the scheduler,
//! which routes work to the CUDA engines of the NVIDIA GPUs. `Gpu` opens the
//! scheduler and is what the worker thread and the headless modes of `main`
//! hold.
use crate::{config::Config, evolution::Population, qd::EvaluationMetrics, scheduler::Scheduler};
use anyhow::{Result, ensure};

/// The evaluation front end: the `Scheduler` with the device names and the
/// GPU memory it reported.
pub struct Gpu {
    /// The names of the devices that evaluate. `new` and
    /// `evaluate_with_metrics` copy them from the scheduler.
    pub name: String,
    /// Bytes allocated on all the GPUs, as the scheduler reported them at the
    /// last `evaluate_with_metrics` or `refresh_allocated_bytes`. It is 0
    /// before then.
    pub allocated_bytes: u64,
    /// The scheduler for creature evaluation. `new` always fills it. The
    /// worker treats `None` as no engines, and `evaluate_with_metrics`
    /// expects it to be there.
    pub sched: Option<Scheduler>,
}

impl Gpu {
    /// Opens the named primary GPU plus any other GPUs that
    /// `EVOLUTION_DEVICES` lists. Fails when the primary GPU does not open.
    pub fn new(name: &str) -> Result<Self> {
        let sched = Scheduler::new(name)?;
        Ok(Self {
            name: sched.names(),
            allocated_bytes: 0,
            sched: Some(sched),
        })
    }
    /// The names of the evaluation devices now, joined with " + ". The
    /// scheduler answers, so the list follows a device that reopened after a
    /// failure. Without a scheduler it gives the copy in `name`.
    pub fn names(&self) -> String {
        self.sched
            .as_ref()
            .map_or_else(|| self.name.clone(), Scheduler::names)
    }
    /// Evaluates the creatures of `pop` at `indices` with one standard trial
    /// each at `cfg` and returns their fitness values in the order of
    /// `indices`. Fails on an index outside `pop`.
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
    /// Evaluates the creatures of `pop` at `indices` with one standard trial
    /// each at `cfg` and returns their metrics in the order of `indices`.
    /// Fails on an index outside `pop`. Call it on an idle scheduler, because
    /// the scheduler drops the results of other work that finishes meanwhile.
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
    /// Reads the bytes the engines hold from the scheduler into
    /// `allocated_bytes`. The worker calls it on every pass of its loop,
    /// because it never calls `evaluate_with_metrics`. It gives 0 without a
    /// scheduler.
    pub fn refresh_allocated_bytes(&mut self) {
        self.allocated_bytes = self.sched.as_ref().map_or(0, Scheduler::allocated_bytes);
    }
    /// True when a scheduler is open, so evaluation can be queued without
    /// blocking.
    pub fn async_capable(&self) -> bool {
        self.sched.is_some()
    }
    /// Units on the evaluation engines now (`Scheduler::on_engines`), or 0
    /// without a scheduler.
    pub fn on_engines(&self) -> usize {
        self.sched.as_ref().map_or(0, Scheduler::on_engines)
    }
}
