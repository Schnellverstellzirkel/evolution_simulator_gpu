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
    /// Ground roughness level (0 = flat), set by the environment effects.
    pub terrain: u8,
    /// Multiplier on each muscle's energy store (heat wave); 1.0 is calm.
    pub muscle_energy: f32,
    /// Multiplier on muscle energy recovery per second (drought); 1.0 is calm.
    pub muscle_recovery: f32,
    /// Ground slope as rise over run; 0.0 is flat, positive tilts the ground
    /// up in the +x direction.
    pub slope: f32,
    /// Steady horizontal wind acceleration (m/s²); 0.0 is calm, positive
    /// pushes nodes in the +x direction.
    pub wind: f32,
    /// Mud sink depth (m); 0.0 is dry ground. Contacting nodes sink by up to
    /// this depth, which raises the effective normal push and multiplies the
    /// friction budget, so dragging feet cost more.
    pub mud: f32,
    /// Pit opening width (m); 0.0 is solid ground. Pits are cut periodically
    /// into the ground with a fixed depth and a spacing that grows with the
    /// width.
    pub gaps: f32,
    /// Raised step height (m); 0.0 is clear ground. Periodic steps with a flat
    /// top and ramp walls rise to this height, spaced `physics::HURDLE_SPACING`
    /// apart.
    pub hurdles: f32,
    /// Earthquake base bump height (m); 0.0 is still ground. Every creature
    /// meets a different bump phase and amplitude, derived deterministically
    /// from its id, so a gait cannot memorize one bump pattern.
    pub quake: f32,
    /// Seasons level: 0 off, 1 slow, 2 normal, 3 fast. When on, the world
    /// advances one step of the `environment::season_rotation` every 20, 10,
    /// or 5 generations.
    pub seasons: u8,
    /// Season rotation steps applied so far. Saved in checkpoints, so a
    /// resumed game continues mid-cycle at the same step.
    pub season_step: u16,
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
    /// Physics resolution for evaluations under this config; `None` is
    /// `Fidelity::standard()`. Runtime only, never saved.
    pub fidelity: Option<crate::physics::Fidelity>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            population: 3_000_000,
            seed: 38,
            random_seed: true,
            duration: 60.0,
            mutation: 1.0,
            gravity: 9.8,
            air_retention: 1.0,
            ground_friction: 1.5,
            ground: true,
            terrain: 0,
            muscle_energy: 1.0,
            muscle_recovery: 1.0,
            slope: 0.0,
            wind: 0.0,
            mud: 0.0,
            gaps: 0.0,
            hurdles: 0.0,
            quake: 0.0,
            seasons: 0,
            season_step: 0,
            min_size: 0.06,
            max_size: 0.12,
            min_friction: 0.65,
            max_friction: 1.0,
            max_nodes: 32,
            max_muscles: 96,
            gpu_budget_mib: 4096,
            ram_budget_mib: 16384,
            throughput: true,
            checkpoint_interval: 10,
            fidelity: None,
        }
    }
}

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
    terrain: u8,
    muscle_energy: f32,
    muscle_recovery: f32,
    slope: f32,
    wind: f32,
    mud: f32,
    gaps: f32,
    hurdles: f32,
    quake: f32,
    seasons: u8,
    season_step: u16,
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
            terrain: c.terrain,
            muscle_energy: c.muscle_energy,
            muscle_recovery: c.muscle_recovery,
            slope: c.slope,
            wind: c.wind,
            mud: c.mud,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            seasons: c.seasons,
            season_step: c.season_step,
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
            fidelity: None,
            population: c.population,
            seed: c.seed,
            random_seed: c.random_seed,
            duration: c.duration,
            mutation: c.mutation,
            gravity: c.gravity,
            air_retention: c.air_retention,
            ground_friction: c.ground_friction,
            ground: c.ground,
            terrain: c.terrain,
            muscle_energy: c.muscle_energy,
            muscle_recovery: c.muscle_recovery,
            slope: c.slope,
            wind: c.wind,
            mud: c.mud,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            seasons: c.seasons,
            season_step: c.season_step,
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
            terrain: c.terrain,
            muscle_energy: c.muscle_energy,
            muscle_recovery: c.muscle_recovery,
            slope: c.slope,
            wind: c.wind,
            mud: c.mud,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            seasons: c.seasons,
            season_step: c.season_step,
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
    terrain: u8,
    muscle_energy: f32,
    muscle_recovery: f32,
    slope: f32,
    wind: f32,
    mud: f32,
    gaps: f32,
    hurdles: f32,
    quake: f32,
    seasons: u8,
    season_step: u16,
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
            terrain: c.terrain,
            muscle_energy: c.muscle_energy,
            muscle_recovery: c.muscle_recovery,
            slope: c.slope,
            wind: c.wind,
            mud: c.mud,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            seasons: c.seasons,
            season_step: c.season_step,
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

impl From<BinaryConfig> for Config {
    fn from(c: BinaryConfig) -> Self {
        Self {
            fidelity: None,
            population: c.population,
            seed: c.seed,
            random_seed: c.random_seed,
            duration: c.duration,
            mutation: c.mutation,
            gravity: c.gravity,
            air_retention: c.air_retention,
            ground: c.ground,
            terrain: c.terrain,
            muscle_energy: c.muscle_energy,
            muscle_recovery: c.muscle_recovery,
            slope: c.slope,
            wind: c.wind,
            mud: c.mud,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            ground_friction: c.ground_friction,
            seasons: c.seasons,
            season_step: c.season_step,
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
            usize::from(self.terrain) < crate::physics::TERRAIN_AMPLITUDES.len(),
            "Unknown ground roughness level"
        );
        ensure!(
            self.muscle_energy.is_finite() && (0.05..=2.0).contains(&self.muscle_energy),
            "Muscle energy multiplier must be 0.05–2"
        );
        ensure!(
            self.muscle_recovery.is_finite() && (0.05..=2.0).contains(&self.muscle_recovery),
            "Muscle recovery multiplier must be 0.05–2"
        );
        ensure!(
            self.slope.is_finite() && (-0.6..=0.6).contains(&self.slope),
            "Slope must be -0.6–0.6 rise over run"
        );
        ensure!(
            self.wind.is_finite() && (-20.0..=20.0).contains(&self.wind),
            "Wind must be -20–20 m/s²"
        );
        ensure!(
            self.mud.is_finite() && (0.0..=0.5).contains(&self.mud),
            "Mud sink depth must be 0–0.5 m"
        );
        ensure!(
            self.gaps.is_finite() && (0.0..=3.0).contains(&self.gaps),
            "Gap width must be 0–3 m"
        );
        ensure!(
            self.hurdles.is_finite() && (0.0..=1.0).contains(&self.hurdles),
            "Hurdle height must be 0–1 m"
        );
        ensure!(
            self.quake.is_finite() && (0.0..=1.0).contains(&self.quake),
            "Quake bump height must be 0–1 m"
        );
        ensure!(
            usize::from(self.seasons) < crate::environment::SEASON_INTERVALS.len(),
            "Unknown seasons level"
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
    /// This config's physics resolution.
    pub fn fidelity(&self) -> crate::physics::Fidelity {
        self.fidelity
            .unwrap_or_else(crate::physics::Fidelity::standard)
    }
    /// Timed steps of a trial (after settling).
    pub fn steps(&self) -> u32 {
        (self.duration * self.fidelity().rate as f32).round() as u32
    }
    pub fn batch_size(&self) -> usize {
        // Fewer readback fences keep the GPU busier. Responsive mode still stays
        // small enough that pausing and editing settings never feels delayed.
        let maximum = std::env::var("EVOLUTION_GPU_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(if self.throughput { 100_000 } else { 8192 });
        // Leave space for power-of-two buffer growth and staging resources.
        let padded_nodes = self.max_nodes.next_power_of_two().max(8);
        // A muscle genome is stored once, with up to four u32 node references;
        // each padded node also owns its state and a NodeAdj range.
        let bytes_per_creature =
            padded_nodes * 40 + self.max_muscles * 72 + self.max_nodes * 12 + 80;
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
