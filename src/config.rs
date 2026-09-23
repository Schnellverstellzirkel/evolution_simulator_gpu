use anyhow::{Result, ensure};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug, PartialEq)]
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
            duration: 18.0,
            mutation: 1.0,
            gravity: 9.8,
            air_retention: 0.985,
            ground_friction: 1.5,
            ground: true,
            min_size: 0.06,
            max_size: 0.12,
            min_friction: 0.65,
            max_friction: 1.0,
            max_nodes: 32,
            max_muscles: 96,
            gpu_budget_mib: 4096,
            ram_budget_mib: 16384,
            throughput: false,
            checkpoint_interval: 10,
        }
    }
}

// JSON settings no longer expose obstacles. The binary checkpoint helper keeps
// the old field slot so existing checkpoints remain readable; its value is
// discarded on load and always empty on save.
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct HumanConfig {
    population: usize,
    seed: u64,
    random_seed: bool,
    duration: f32,
    mutation: f32,
    gravity: f32,
    air_retention: f32,
    ground_friction: f32,
    ground: bool,
    min_size: f32,
    max_size: f32,
    min_friction: f32,
    max_friction: f32,
    max_nodes: usize,
    max_muscles: usize,
    gpu_budget_mib: usize,
    ram_budget_mib: usize,
    throughput: bool,
    checkpoint_interval: u32,
}

impl Default for HumanConfig {
    fn default() -> Self {
        let c = Config::default();
        Self {
            population: c.population,
            seed: c.seed,
            random_seed: c.random_seed,
            duration: c.duration,
            mutation: c.mutation,
            gravity: c.gravity,
            air_retention: c.air_retention,
            ground_friction: c.ground_friction,
            ground: c.ground,
            min_size: c.min_size,
            max_size: c.max_size,
            min_friction: c.min_friction,
            max_friction: c.max_friction,
            max_nodes: c.max_nodes,
            max_muscles: c.max_muscles,
            gpu_budget_mib: c.gpu_budget_mib,
            ram_budget_mib: c.ram_budget_mib,
            throughput: c.throughput,
            checkpoint_interval: c.checkpoint_interval,
        }
    }
}

impl From<HumanConfig> for Config {
    fn from(c: HumanConfig) -> Self {
        Self {
            population: c.population,
            seed: c.seed,
            random_seed: c.random_seed,
            duration: c.duration,
            mutation: c.mutation,
            gravity: c.gravity,
            air_retention: c.air_retention,
            ground_friction: c.ground_friction,
            ground: c.ground,
            min_size: c.min_size,
            max_size: c.max_size,
            min_friction: c.min_friction,
            max_friction: c.max_friction,
            max_nodes: c.max_nodes,
            max_muscles: c.max_muscles,
            gpu_budget_mib: c.gpu_budget_mib,
            ram_budget_mib: c.ram_budget_mib,
            throughput: c.throughput,
            checkpoint_interval: c.checkpoint_interval,
        }
    }
}

impl From<&Config> for HumanConfig {
    fn from(c: &Config) -> Self {
        Self {
            population: c.population,
            seed: c.seed,
            random_seed: c.random_seed,
            duration: c.duration,
            mutation: c.mutation,
            gravity: c.gravity,
            air_retention: c.air_retention,
            ground_friction: c.ground_friction,
            ground: c.ground,
            min_size: c.min_size,
            max_size: c.max_size,
            min_friction: c.min_friction,
            max_friction: c.max_friction,
            max_nodes: c.max_nodes,
            max_muscles: c.max_muscles,
            gpu_budget_mib: c.gpu_budget_mib,
            ram_budget_mib: c.ram_budget_mib,
            throughput: c.throughput,
            checkpoint_interval: c.checkpoint_interval,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct BinaryConfig {
    population: usize,
    seed: u64,
    random_seed: bool,
    duration: f32,
    mutation: f32,
    gravity: f32,
    air_retention: f32,
    ground_friction: f32,
    ground: bool,
    min_size: f32,
    max_size: f32,
    min_friction: f32,
    max_friction: f32,
    max_nodes: usize,
    max_muscles: usize,
    obstacles: Vec<[f32; 4]>,
    gpu_budget_mib: usize,
    ram_budget_mib: usize,
    throughput: bool,
    checkpoint_interval: u32,
}

impl From<&Config> for BinaryConfig {
    fn from(c: &Config) -> Self {
        Self {
            population: c.population,
            seed: c.seed,
            random_seed: c.random_seed,
            duration: c.duration,
            mutation: c.mutation,
            gravity: c.gravity,
            air_retention: c.air_retention,
            ground_friction: c.ground_friction,
            ground: c.ground,
            min_size: c.min_size,
            max_size: c.max_size,
            min_friction: c.min_friction,
            max_friction: c.max_friction,
            max_nodes: c.max_nodes,
            max_muscles: c.max_muscles,
            obstacles: Vec::new(),
            gpu_budget_mib: c.gpu_budget_mib,
            ram_budget_mib: c.ram_budget_mib,
            throughput: c.throughput,
            checkpoint_interval: c.checkpoint_interval,
        }
    }
}

impl From<BinaryConfig> for Config {
    fn from(c: BinaryConfig) -> Self {
        Self {
            population: c.population,
            seed: c.seed,
            random_seed: c.random_seed,
            duration: c.duration,
            mutation: c.mutation,
            gravity: c.gravity,
            air_retention: c.air_retention,
            ground: c.ground,
            ground_friction: c.ground_friction,
            min_size: c.min_size,
            max_size: c.max_size,
            min_friction: c.min_friction,
            max_friction: c.max_friction,
            max_nodes: c.max_nodes,
            max_muscles: c.max_muscles,
            gpu_budget_mib: c.gpu_budget_mib,
            ram_budget_mib: c.ram_budget_mib,
            throughput: c.throughput,
            checkpoint_interval: c.checkpoint_interval,
        }
    }
}

impl Serialize for Config {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            HumanConfig::from(self).serialize(serializer)
        } else {
            BinaryConfig::from(self).serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for Config {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            HumanConfig::deserialize(deserializer).map(Into::into)
        } else {
            BinaryConfig::deserialize(deserializer).map(Into::into)
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
        // Fewer readback fences keep the GPU busier. Responsive mode still stays
        // small enough that pausing and editing settings never feels delayed.
        let maximum = if self.throughput { 65536 } else { 8192 };
        // Leave space for power-of-two buffer growth and staging resources.
        let bytes_per_creature =
            self.max_nodes.next_power_of_two().max(8) * 32 + self.max_muscles * 32 + 80;
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
