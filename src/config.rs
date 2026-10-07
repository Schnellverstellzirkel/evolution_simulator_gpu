//! `Config` holds the settings of an experiment: the population and seed, the
//! trial length, the physics of the world, the limits on bodies, the memory
//! budgets and the autosave interval. `validate` checks every range. The
//! environment effects (`environment`) change its world fields, and the kernel
//! packing (`kernel`) reads them to set up each trial. A binary save and a JSON
//! preset (`--config`) each go through a private struct that leaves out the
//! runtime-only fields.

use anyhow::{Result, ensure};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The settings of an experiment. `fidelity`, `screen` and `rungs` are runtime
/// only: no save or preset holds them.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Creatures in a generation. An experiment keeps the count it starts with.
    pub population: usize,
    /// Seed of the search's random streams.
    pub seed: u64,
    /// Whether a new game takes its seed from the clock (`resolved`).
    pub random_seed: bool,
    /// Length of a trial (s).
    pub duration: f32,
    /// Multiplier on the size of the random changes to a child's genes, in
    /// breeding and in the CMA samples. 1.0 is the normal strength.
    pub mutation: f32,
    /// Downward acceleration (m/s²), set by the Gravity effect.
    pub gravity: f32,
    /// Velocity retained per 1/60 second, independent of physics timestep.
    pub air_retention: f32,
    /// Multiplier on every node's friction coefficient where it meets the
    /// ground, set by the Grip effect. The calm world is 1.5.
    pub ground_friction: f32,
    /// Whether the world has a ground. Without one, the effects that act on
    /// the ground are off: roughness, slope, mud, gaps, hurdles, quake, ice
    /// patches and brambles.
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
    /// Drag (1/s) on every node that is not a foot (the end of a leg of at
    /// least two bones) while it touches the ground.
    pub brambles: f32,
    /// Water line height (m) above the flat ground; 0.0 is dry. Nodes below
    /// it float and bones meet a viscous medium, so swimming strokes pay.
    pub water: f32,
    /// Ice patch strength; 0.0 is none. On periodic bands of ground the
    /// friction is lowered by this share, so a foot that lands on ice cannot
    /// push.
    pub patches: f32,
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
    /// Autochange level: 0 off, 1 slow, 2 normal, 3 fast. When on, the world
    /// advances one step of the `environment::autochange_ladder` every 100, 50
    /// or 20 generations (`environment::AUTOCHANGE_INTERVALS`).
    pub autochange: u8,
    /// Autochange ladder steps applied so far. Saved in checkpoints, so a
    /// resumed game continues mid-cycle at the same step.
    pub autochange_step: u16,
    /// Minimum node diameter (m).
    pub min_size: f32,
    /// Maximum node diameter (m).
    pub max_size: f32,
    /// Minimum node friction coefficient.
    pub min_friction: f32,
    /// Maximum node friction coefficient.
    pub max_friction: f32,
    /// Most nodes in a body, from 3 up to `evolution::MAX_NODES`.
    pub max_nodes: usize,
    /// Most muscles in a body, from `max_nodes` up to `evolution::MAX_MUSCLES`.
    pub max_muscles: usize,
    /// GPU memory (MiB) that `batch_size` sizes a batch against.
    pub gpu_budget_mib: usize,
    /// Host memory (MiB) for the population. `validate` checks the population
    /// against it at 1,200 bytes a creature.
    pub ram_budget_mib: usize,
    /// Throughput mode (large batches) against responsive mode (small ones).
    /// Only `batch_size` reads it. The game sizes its blocks from
    /// `storage::RingShape`.
    pub throughput: bool,
    /// Generations between autosaves. 0 turns autosave off.
    pub checkpoint_interval: u32,
    /// Physics resolution for evaluations under this config; `None` is
    /// `Fidelity::standard()`. Runtime only, never saved.
    pub fidelity: Option<crate::physics::Fidelity>,
    /// Early screening of standard trials, set by the experiment each
    /// generation; `None` runs every trial in full. Runtime only, never saved.
    pub screen: Option<crate::physics::Screen>,
    /// The early rungs of standard trials (`rungs`), set by the experiment
    /// each generation; `None` runs no rung. Runtime only, never saved.
    pub rungs: Option<crate::rungs::Rungs>,
}

/// The game's normal settings: 3M creatures, 20 s trials and the calm world.
impl Default for Config {
    fn default() -> Self {
        Self {
            population: 3_000_000,
            seed: 38,
            random_seed: true,
            // The owner's fixed trial length (2026-09-28; was 60 s).
            duration: 20.0,
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
            brambles: 0.0,
            water: 0.0,
            patches: 0.0,
            gaps: 0.0,
            hurdles: 0.0,
            quake: 0.0,
            autochange: 0,
            autochange_step: 0,
            min_size: 0.06,
            max_size: 0.12,
            min_friction: 0.65,
            max_friction: 1.0,
            max_nodes: 32,
            max_muscles: 96,
            gpu_budget_mib: 4096,
            ram_budget_mib: 16384,
            throughput: true,
            // The game writes no files on its own (owner, 2026-09-28). The
            // File menu turns autosave on.
            checkpoint_interval: 0,
            fidelity: None,
            screen: None,
            rungs: None,
        }
    }
}

/// `Config` as a text format such as JSON writes it, without the runtime-only
/// fields. A missing field takes its value from `Config::default()`, so a
/// preset lists only the settings it changes.
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
    brambles: f32,
    water: f32,
    patches: f32,
    gaps: f32,
    hurdles: f32,
    quake: f32,
    autochange: u8,
    autochange_step: u16,
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
            brambles: c.brambles,
            water: c.water,
            patches: c.patches,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            autochange: c.autochange,
            autochange_step: c.autochange_step,
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
            screen: None,
            rungs: None,
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
            brambles: c.brambles,
            water: c.water,
            patches: c.patches,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            autochange: c.autochange,
            autochange_step: c.autochange_step,
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
            brambles: c.brambles,
            water: c.water,
            patches: c.patches,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            autochange: c.autochange,
            autochange_step: c.autochange_step,
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

/// `Config` as a binary format writes it, without the runtime-only fields. The
/// settings in a save and in each of its history rows use it. A binary format
/// has no field names, so the order of these fields is the layout in the file,
/// and no field may be missing.
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
    brambles: f32,
    water: f32,
    patches: f32,
    gaps: f32,
    hurdles: f32,
    quake: f32,
    autochange: u8,
    autochange_step: u16,
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
            brambles: c.brambles,
            water: c.water,
            patches: c.patches,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            autochange: c.autochange,
            autochange_step: c.autochange_step,
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
            screen: None,
            rungs: None,
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
            brambles: c.brambles,
            water: c.water,
            patches: c.patches,
            gaps: c.gaps,
            hurdles: c.hurdles,
            quake: c.quake,
            autochange: c.autochange,
            autochange_step: c.autochange_step,
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

/// A text format gets `HumanConfig` and a binary format gets `BinaryConfig`.
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

/// A text format reads `HumanConfig` and a binary format reads `BinaryConfig`.
/// `fidelity`, `screen` and `rungs` come back as `None`.
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
    /// Checks the settings against their ranges. The first one out of range
    /// gives an error, and the player sees its message.
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
            self.brambles.is_finite() && (0.0..=60.0).contains(&self.brambles),
            "Brambles drag must be 0–60 per second"
        );
        ensure!(
            self.water.is_finite() && (0.0..=3.0).contains(&self.water),
            "Water line must be 0–3 m"
        );
        ensure!(
            self.patches.is_finite() && (0.0..=1.0).contains(&self.patches),
            "Ice patch strength must be 0–1"
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
            usize::from(self.autochange) < crate::environment::AUTOCHANGE_INTERVALS.len(),
            "Unknown autochange level"
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
            (3..=crate::evolution::MAX_NODES).contains(&self.max_nodes)
                && (3..=crate::evolution::MAX_MUSCLES).contains(&self.max_muscles)
                && self.max_muscles >= self.max_nodes,
            "Body limits: 3 to {} nodes; at least as many muscles, up to {}",
            crate::evolution::MAX_NODES,
            crate::evolution::MAX_MUSCLES
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
    /// Whether the two settings describe different physics for a fixed
    /// creature's trial. It compares the trial length and the world's physics.
    /// It leaves out the runtime-only fields (`fidelity`, `screen`, `rungs`),
    /// the creation-only settings (body limits, node sizes and friction,
    /// mutation) and the run settings (population, seed, autochange,
    /// throughput, budgets, autosave).
    pub fn physics_differs(&self, other: &Config) -> bool {
        self.duration != other.duration
            || self.gravity != other.gravity
            || self.air_retention != other.air_retention
            || self.ground_friction != other.ground_friction
            || self.ground != other.ground
            || self.terrain != other.terrain
            || self.muscle_energy != other.muscle_energy
            || self.muscle_recovery != other.muscle_recovery
            || self.slope != other.slope
            || self.wind != other.wind
            || self.mud != other.mud
            || self.brambles != other.brambles
            || self.water != other.water
            || self.patches != other.patches
            || self.gaps != other.gaps
            || self.hurdles != other.hurdles
            || self.quake != other.quake
    }
    /// This config's physics resolution: `fidelity`, or `Fidelity::standard()`
    /// when that is `None`.
    pub fn fidelity(&self) -> crate::physics::Fidelity {
        self.fidelity
            .unwrap_or_else(crate::physics::Fidelity::standard)
    }
    /// Steps of a trial: `duration` at this config's step rate, rounded. The
    /// kernel runs all of them from the start pose and settles nothing first.
    /// A replay recording starts with `Fidelity::settle()` frames that repeat
    /// the start pose, and the frame at `Fidelity::settle() + n` shows the pose
    /// after `n` steps.
    pub fn steps(&self) -> u32 {
        (self.duration * self.fidelity().rate as f32).round() as u32
    }
    /// Creatures per batch for the `EvalBench` command of `main.rs`, the only
    /// caller. It is 100,000 in throughput mode and 8,192 in responsive mode,
    /// lowered so that four times a rough estimate of the batch's bytes fits in
    /// `gpu_budget_mib`. It is at least 1.
    pub fn batch_size(&self) -> usize {
        // Fewer readback fences keep the GPU busier. Responsive mode still stays
        // small enough that pausing and editing settings never feels delayed.
        let maximum = if self.throughput { 100_000 } else { 8192 };
        // Leave space for power-of-two buffer growth and staging resources.
        let padded_nodes = self.max_nodes.next_power_of_two().max(8);
        // A rough size of one creature in bytes. The sizes come from the
        // buffers of the earlier GPU kernels and only approximate the CUDA
        // records.
        let bytes_per_creature =
            padded_nodes * 40 + self.max_muscles * 72 + self.max_nodes * 12 + 80;
        maximum
            .min(self.gpu_budget_mib * 1024 * 1024 / (bytes_per_creature * 4))
            .max(1)
    }
    /// This config with `seed` set from the clock (nanoseconds since the Unix
    /// epoch) when `random_seed` is on, and unchanged when it is off.
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
