use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub population: usize,
    pub seed: u64,
    pub random_seed: bool,
    pub duration: f32,
    pub mutation: f32,
    pub gravity: f32,
    /// Velocity retained per 1/60 second, independent of physics timestep.
    pub air_retention: f32,
    pub ground_friction: f32,
    pub ground: bool,
    pub min_size: f32,
    pub max_size: f32,
    pub min_friction: f32,
    pub max_friction: f32,
    pub max_nodes: usize,
    pub max_muscles: usize,
    pub obstacles: Vec<[f32; 4]>,
    pub gpu_budget_mib: usize,
    pub ram_budget_mib: usize,
    pub throughput: bool,
    pub checkpoint_interval: u32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            population: 1000,
            seed: 38,
            random_seed: true,
            duration: 15.0,
            mutation: 1.0,
            gravity: 3.6,
            air_retention: 0.95,
            ground_friction: 4.0,
            ground: true,
            min_size: 0.08,
            max_size: 0.08,
            min_friction: 0.0,
            max_friction: 1.0,
            max_nodes: 32,
            max_muscles: 96,
            obstacles: vec![],
            gpu_budget_mib: 4096,
            ram_budget_mib: 16384,
            throughput: false,
            checkpoint_interval: 10,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (2..=20_000_000).contains(&self.population) && self.population.is_multiple_of(2),
            "Population must be even, between 2 and 20,000,000"
        );
        ensure!(
            self.duration.is_finite() && (0.1..=300.0).contains(&self.duration),
            "Trial duration must be 0.1–300 seconds"
        );
        ensure!(
            self.mutation.is_finite() && (0.0..=10.0).contains(&self.mutation),
            "Mutation must be 0–10"
        );
        ensure!(
            self.gravity.is_finite() && (0.0..=100.0).contains(&self.gravity),
            "Gravity must be 0–100 m/s²"
        );
        ensure!(
            self.air_retention.is_finite() && (0.0..=1.02).contains(&self.air_retention),
            "Air retention must be 0–1.02"
        );
        ensure!(
            self.ground_friction.is_finite() && (0.0..=20.0).contains(&self.ground_friction),
            "Ground friction must be 0–20"
        );
        ensure!(
            self.min_size.is_finite()
                && self.max_size.is_finite()
                && self.min_size >= 0.01
                && self.max_size <= 1.0
                && self.min_size <= self.max_size,
            "Node diameters must be ordered within 0.01–1 m"
        );
        ensure!(
            self.min_friction.is_finite()
                && self.max_friction.is_finite()
                && self.min_friction >= 0.0
                && self.max_friction <= 1.0
                && self.min_friction <= self.max_friction,
            "Node friction bounds must be ordered within 0–1"
        );
        ensure!(
            (3..=64).contains(&self.max_nodes)
                && (3..=256).contains(&self.max_muscles)
                && self.max_muscles >= self.max_nodes,
            "Body limits: 3–64 nodes; at least as many muscles, up to 256"
        );
        ensure!(
            self.obstacles.len() <= 256,
            "At most 256 rectangular obstacles"
        );
        for r in &self.obstacles {
            ensure!(
                r.iter().all(|v| v.is_finite() && v.abs() <= 10000.0) && r[0] < r[2] && r[1] < r[3],
                "Obstacle coordinates must be finite and ordered (left, bottom, right, top)"
            );
        }
        ensure!(
            (32..=6144).contains(&self.gpu_budget_mib),
            "GPU budget must be 32–6144 MiB"
        );
        ensure!(
            (64..=24576).contains(&self.ram_budget_mib),
            "RAM budget must be 64–24576 MiB"
        );
        ensure!(
            self.population.saturating_mul(1200) < self.ram_budget_mib * 1024 * 1024,
            "Population requires more RAM; increase the budget or reduce population"
        );
        Ok(())
    }
    pub fn steps(&self) -> u32 {
        (self.duration * 120.0).round() as u32
    }
    pub fn batch_size(&self) -> usize {
        let maximum = if self.throughput { 16384 } else { 2048 };
        // Leave space for power-of-two buffer growth and staging resources.
        let bytes_per_creature =
            self.max_nodes.next_power_of_two().max(8) * 32 + self.max_muscles * 32 + 24;
        maximum
            .min(self.gpu_budget_mib * 1024 * 1024 / (bytes_per_creature * 4))
            .max(1)
    }
    pub fn resolved(mut self) -> Self {
        if self.random_seed {
            self.seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
        }
        self
    }
}
