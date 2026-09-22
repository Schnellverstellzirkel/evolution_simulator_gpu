use crate::{
    config::Config,
    evolution::{self, Creature, FAILED, Population},
};
use anyhow::{Context, Result, ensure};
use bincode::Options;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Stage {
    Ready,
    Evaluating,
    Evaluated,
    Ranked,
    Selected,
}
impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ready => "Ready",
            Self::Evaluating => "Evaluating",
            Self::Evaluated => "Evaluation complete",
            Self::Ranked => "Sorted by fitness",
            Self::Selected => "Survivors selected",
        }
    }
}
pub const PERCENTILES: [f32; 29] = [
    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 20., 30., 40., 50., 60., 70., 80., 90., 91., 92.,
    93., 94., 95., 96., 97., 98., 99., 100.,
];
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stats {
    pub generation: u32,
    pub best: f32,
    pub median: f32,
    pub worst: f32,
    pub mean: f32,
    pub failed: usize,
    pub seconds: f64,
    pub population: usize,
    pub percentiles: Vec<f32>,
    /// Sparse centimeter bins preserve adjustable historical histograms without storing all scores.
    pub histogram: Vec<(i32, u32)>,
    pub species: Vec<(usize, usize, u32)>,
    pub representatives: Vec<Creature>,
    pub config: Config,
}
#[derive(Serialize, Deserialize)]
pub struct Experiment {
    pub config: Config,
    pub pending: Option<Config>,
    pub generation: u32,
    pub population: Population,
    pub scores: Vec<f32>,
    pub evaluated: usize,
    pub stage: Stage,
    pub ranks: Vec<usize>,
    pub parents: Vec<usize>,
    pub history: Vec<Stats>,
    pub evaluation_seconds: f64,
}
impl Experiment {
    pub fn new(config: Config) -> Result<Self> {
        let config = config.resolved();
        let population = evolution::create(&config)?;
        let scores = vec![f32::NAN; config.population];
        Ok(Self {
            config,
            pending: None,
            generation: 0,
            population,
            scores,
            evaluated: 0,
            stage: Stage::Ready,
            ranks: vec![],
            parents: vec![],
            history: vec![],
            evaluation_seconds: 0.0,
        })
    }
    pub fn rank(&mut self) {
        self.ranks = evolution::ranking(&self.scores);
        let mut histogram = BTreeMap::<i32, u32>::new();
        let mut species = BTreeMap::<(usize, usize), u32>::new();
        let mut sum = 0.0f64;
        let mut failed = 0;
        for (&s, g) in self.scores.iter().zip(&self.population.genomes) {
            if s > FAILED && s.is_finite() {
                sum += s as f64;
                *histogram.entry((s * 100.0).floor() as i32).or_default() += 1;
            } else {
                failed += 1;
            }
            *species.entry((g.node_count, g.muscle_count)).or_default() += 1;
        }
        let count = self.scores.len();
        let valid = count - failed;
        let quantile = |p: f32| {
            if valid == 0 {
                0.0
            } else {
                self.scores[self.ranks[((1.0 - p / 100.0) * (valid - 1) as f32).round() as usize]]
            }
        };
        let representatives = [count - 1, (count - 1) / 2, 0]
            .map(|r| self.population.creature(self.ranks[r]))
            .to_vec();
        self.history.push(Stats {
            generation: self.generation,
            best: quantile(100.),
            median: quantile(50.),
            worst: quantile(0.),
            mean: if valid > 0 {
                (sum / valid as f64) as f32
            } else {
                0.
            },
            failed,
            seconds: self.evaluation_seconds,
            population: count,
            percentiles: PERCENTILES.iter().map(|&p| quantile(p)).collect(),
            histogram: histogram.into_iter().collect(),
            species: species.into_iter().map(|((n, m), c)| (n, m, c)).collect(),
            representatives,
            config: self.config.clone(),
        });
        self.stage = Stage::Ranked;
    }
    pub fn select(&mut self) {
        self.parents = evolution::survivors(&self.config, self.generation, &self.ranks);
        self.stage = Stage::Selected;
    }
    pub fn reproduce(&mut self) -> Result<()> {
        let cfg = self.pending.as_ref().unwrap_or(&self.config);
        let next = evolution::reproduce(&self.population, cfg, self.generation, &self.parents)?;
        self.population = next;
        if let Some(c) = self.pending.take() {
            self.config = c;
        }
        self.generation += 1;
        self.stage = Stage::Ready;
        self.evaluated = 0;
        self.scores.fill(f32::NAN);
        self.ranks.clear();
        self.parents.clear();
        self.evaluation_seconds = 0.0;
        Ok(())
    }
    pub fn update_config(&mut self, cfg: Config) -> Result<()> {
        cfg.validate()?;
        ensure!(
            cfg.population == self.config.population
                && cfg.seed == self.config.seed
                && cfg.random_seed == self.config.random_seed,
            "Population or seed changes require a new experiment"
        );
        ensure!(
            self.population
                .genomes
                .iter()
                .all(|g| g.node_count <= cfg.max_nodes && g.muscle_count <= cfg.max_muscles),
            "Existing bodies exceed these limits; start a new experiment"
        );
        if self.stage == Stage::Ready {
            self.config = cfg;
        } else {
            self.pending = Some(cfg);
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        self.population.validate(&self.config)?;
        ensure!(
            self.scores.len() == self.config.population && self.evaluated <= self.scores.len(),
            "Invalid evaluation progress"
        );
        ensure!(
            self.scores[..self.evaluated].iter().all(|s| s.is_finite()),
            "Invalid completed fitness values"
        );
        ensure!(
            self.scores[self.evaluated..].iter().all(|s| s.is_nan()),
            "Invalid pending fitness values"
        );
        if matches!(
            self.stage,
            Stage::Evaluated | Stage::Ranked | Stage::Selected
        ) {
            ensure!(
                self.evaluated == self.scores.len(),
                "Incomplete evaluated generation"
            );
        }
        if matches!(self.stage, Stage::Ranked | Stage::Selected) {
            ensure!(
                self.ranks.len() == self.config.population,
                "Invalid ranking length"
            );
            let mut seen = vec![false; self.ranks.len()];
            for &r in &self.ranks {
                ensure!(r < seen.len() && !seen[r], "Invalid ranking index");
                seen[r] = true;
            }
        }
        if self.stage == Stage::Selected {
            ensure!(
                self.parents.len() == self.config.population / 2
                    && self.parents.iter().all(|&i| i < self.config.population),
                "Invalid parents"
            );
        }
        if let Some(cfg) = &self.pending {
            cfg.validate()?;
            ensure!(
                cfg.population == self.config.population && cfg.seed == self.config.seed
                    && cfg.random_seed == self.config.random_seed
                    && self.population.genomes.iter().all(|g| g.node_count <= cfg.max_nodes && g.muscle_count <= cfg.max_muscles),
                "Invalid pending settings"
            );
        }
        ensure!(
            self.evaluation_seconds.is_finite() && self.evaluation_seconds >= 0.0,
            "Invalid evaluation time"
        );
        ensure!(
            self.history.len() <= self.generation as usize + 1,
            "Invalid history length"
        );
        for (index, stats) in self.history.iter().enumerate() {
            stats.config.validate()?;
            ensure!(
                stats.generation as usize == index
                    && stats.population == stats.config.population
                    && stats.failed <= stats.population,
                "Invalid historical generation"
            );
            ensure!(
                stats.percentiles.len() == PERCENTILES.len()
                    && stats.percentiles.iter().all(|v| v.is_finite())
                    && [stats.best, stats.median, stats.worst, stats.mean]
                        .iter()
                        .all(|v| v.is_finite())
                    && stats.seconds.is_finite()
                    && stats.seconds >= 0.0,
                "Invalid historical statistics"
            );
            ensure!(
                stats.representatives.len() == 3,
                "Missing historical representatives"
            );
            ensure!(
                stats.histogram.iter().map(|(_, n)| *n as u64).sum::<u64>() + stats.failed as u64
                    == stats.population as u64,
                "Invalid histogram totals"
            );
            ensure!(
                stats.species.iter().map(|(_, _, n)| *n as u64).sum::<u64>()
                    == stats.population as u64,
                "Invalid body-type totals"
            );
            let mut representatives = Population::default();
            for creature in &stats.representatives {
                representatives.push(creature.clone());
            }
            representatives.validate(&Config {
                population: 3,
                ..stats.config.clone()
            })?;
        }
        Ok(())
    }
}
const MAGIC: &[u8; 8] = b"EVORUST1";
pub fn save(path: &Path, experiment: &Experiment) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("evo.tmp");
    let file = File::create(&tmp)?;
    let mut out = BufWriter::new(file);
    out.write_all(MAGIC)?;
    let mut encoder = zstd::stream::write::Encoder::new(out, 3)?;
    encoder.include_checksum(true)?;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize_into(&mut encoder, experiment)?;
    let mut out = encoder.finish()?;
    out.flush()?;
    out.get_ref().sync_all()?;
    drop(out);
    std::fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}
pub fn load(path: &Path) -> Result<Experiment> {
    let mut file = BufReader::new(File::open(path).context("Cannot open checkpoint")?);
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    ensure!(&magic == MAGIC, "Unsupported checkpoint format/version");
    let mut decoder = zstd::stream::read::Decoder::new(file)?;
    let experiment: Experiment = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(24 * 1024 * 1024 * 1024)
        .deserialize_from(&mut decoder)?;
    let mut trailing = [0u8; 1];
    ensure!(
        decoder.read(&mut trailing)? == 0,
        "Unexpected trailing checkpoint data"
    );
    experiment.validate()?;
    Ok(experiment)
}
pub fn export_csv(path: &Path, history: &[Stats]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "generation",
        "population",
        "best_m",
        "median_m",
        "worst_m",
        "mean_m",
        "failed",
        "evaluation_seconds",
        "seed",
    ])?;
    for s in history {
        w.serialize((
            s.generation,
            s.population,
            s.best,
            s.median,
            s.worst,
            s.mean,
            s.failed,
            s.seconds,
            s.config.seed,
        ))?;
    }
    w.flush()?;
    Ok(())
}
