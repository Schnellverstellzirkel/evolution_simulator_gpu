//! Evaluation front end. All creature evaluation runs through the scheduler,
//! which spreads work across the Vulkan GPUs and the CPU SIMD engine.
use crate::{config::Config, evolution::Population, qd::EvaluationMetrics, scheduler::Scheduler};
use anyhow::{Result, ensure};

/// Physics steps per GPU dispatch. Short ranges let display work interleave;
/// the Vulkan engine keeps them nearly free.
pub const DEFAULT_STEP_RANGE: u32 = 64;

pub struct Gpu {
    pub name: String,
    pub allocated_bytes: u64,
    pub sched: Option<Scheduler>,
}

impl Gpu {
    /// Opens the named primary GPU plus the other evaluation engines.
    pub fn new(name: &str) -> Result<Self> {
        let sched = Scheduler::new(name)?;
        Ok(Self {
            name: sched.names(),
            allocated_bytes: 0,
            sched: Some(sched),
        })
    }
    /// The UI passes its render device; evaluation opens its own devices.
    pub fn from_device(_device: wgpu::Device, _queue: wgpu::Queue, name: String) -> Result<Self> {
        Self::new(&name)
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
        let metrics = sched.evaluate(pop, indices, cfg)?;
        self.allocated_bytes = sched.allocated_bytes();
        Ok(metrics)
    }
    /// True when evaluation can be queued without blocking.
    pub fn async_capable(&self) -> bool {
        self.sched.is_some()
    }
    pub fn async_in_flight(&self) -> usize {
        self.sched.as_ref().map_or(0, |s| s.in_flight())
    }
}

/// Replaces the kernel's exact cosine with the default polynomial unless
/// `EVOLUTION_EXACT_COS` is set.
pub(crate) fn apply_fast_cos(source: String) -> String {
    if std::env::var_os("EVOLUTION_EXACT_COS").is_some() {
        return source;
    }
    let fast_cos_function = "fn fast_cos_pi(x:f32)->f32 { let y=(x-0.5)*3.14159265359; let z=y*y; var p=fma(z,-2.50521084e-8,2.75573192e-6); p=fma(z,p,-1.98412698e-4); p=fma(z,p,8.33333377e-3); p=fma(z,p,-1.66666672e-1); p=fma(z,p,1.0); return -y*p; }\n";
    let marker = "fn limited_muscle_length";
    source
        .replace(
            "cos(3.14159265359 * phase * m.inv_duty)",
            "fast_cos_pi(phase * m.inv_duty)",
        )
        .replace(
            "cos(3.14159265359 * (phase - m.duty) * m.inv_complement)",
            "fast_cos_pi((phase - m.duty) * m.inv_complement)",
        )
        .replace(marker, &format!("{fast_cos_function}{marker}"))
}
