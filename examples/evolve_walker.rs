//! Evolves the walker fixture of the effect tests (`energy_dependent_walker`
//! in `tests/simulation.rs`) under the current physics, and prints it as Rust
//! code. Run it after a physics change
//! that leaves the old fixture unable to walk.
//!
//! Recipe: 2,048 creatures, seed 38, 15 s trials, 30 generations on the CPU
//! engine. The archive's elites are then tried in order of fitness in the
//! tests' worlds (16 copies, 5 s trials), and the first one that loses enough
//! distance to every effect is printed, with a margin over the tests'
//! thresholds (losses are shares of the calm distance). It also reports how far a 10 um nudge of the pose moves a
//! fine-fidelity trial. Evolved fast gaits are chaotic there; GPU acceptance
//! now uses repeatability on the GPU rather than this CPU-evolved fixture.
//! See `tests/gpu_repeatability.rs`.
//!
//! Usage: cargo run --release --example evolve_walker -- [generations] [population] [seed]
//! (defaults 30, 2048, 38)
use evolution_simulator::{
    config::Config,
    cpu_engine,
    evolution::{Creature, Population, Rng},
    physics::Fidelity,
    scheduler,
    storage::Experiment,
};

/// Applies the same pose and grip perturbation used for scheduler checks.
fn perturb(creature: &mut Creature) {
    let mut rng = Rng::new(creature.id ^ 0x5eed_7a11, 0, 0);
    for node in &mut creature.nodes {
        node.x += rng.range(-0.02, 0.02);
        node.y += rng.range(0.0, 0.02);
        node.friction = (node.friction * rng.range(0.9, 1.1)).clamp(0.0, 1.0);
    }
}

/// Largest change of the fine-fidelity distance over `seconds` when the pose
/// moves by 10 um in each of three directions.
fn nudge_sensitivity(creature: &Creature, seconds: f32) -> f32 {
    let cfg = Config {
        random_seed: false,
        duration: seconds,
        fidelity: Some(Fidelity::fine()),
        ..Config::default()
    };
    let mut pop = Population::default();
    pop.push(creature.clone());
    for (dx, dy) in [(1e-5, 0.0), (-1e-5, 0.0), (0.0, 1e-5)] {
        let mut nudged = creature.clone();
        for node in &mut nudged.nodes {
            node.x += dx;
            node.y += dy;
        }
        pop.push(nudged);
    }
    let cfg = Config {
        population: pop.genomes.len(),
        ..cfg
    };
    let results = cpu_engine::evaluate(&pop, &cfg);
    results[1..]
        .iter()
        .map(|r| (r.fitness - results[0].fitness).abs())
        .fold(0.0, f32::max)
}

fn mean_distance(creature: &Creature, cfg: &Config) -> f32 {
    let mut pop = Population::default();
    for i in 0..16 {
        let mut copy = creature.clone();
        copy.id = i;
        pop.push(copy);
    }
    let cfg = Config {
        population: 16,
        ..cfg.clone()
    };
    let results = cpu_engine::evaluate(&pop, &cfg);
    let finite: Vec<f32> = results
        .iter()
        .map(|r| r.fitness)
        .filter(|f| f.is_finite())
        .collect();
    finite.iter().sum::<f32>() / finite.len().max(1) as f32
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: u64| args.get(i).map_or(default, |v| v.parse().expect("number"));
    let (generations, population, seed) = (arg(1, 30), arg(2, 2048) as usize, arg(3, 38));
    let cfg = Config {
        population,
        seed,
        duration: 15.0,
        random_seed: false,
        ..Config::default()
    };
    let mut experiment = Experiment::new(cfg)?;
    for generation in 0..generations {
        let results = cpu_engine::evaluate(&experiment.population, &experiment.config);
        for (index, result) in results.iter().enumerate() {
            let metrics =
                scheduler::to_metrics(&experiment.population, index, result, &experiment.config);
            experiment.record_result(index, &metrics);
        }
        experiment.evaluated = experiment.config.population;
        experiment.archive_batch()?;
        eprintln!(
            "generation {generation}: archive best {:.2} m",
            experiment
                .archive
                .entries
                .iter()
                .map(|e| e.fitness)
                .fold(f32::MIN, f32::max)
        );
        experiment.prepare_next_batch()?;
    }
    let mut elites: Vec<_> = experiment.archive.entries.clone();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let base = Config {
        random_seed: false,
        duration: 5.0,
        ..Config::default()
    };
    let world = |change: fn(&mut Config)| {
        let mut cfg = base.clone();
        change(&mut cfg);
        cfg
    };
    let worlds = [
        ("heat", world(|c| c.muscle_energy = 0.35), 0.25),
        ("drought", world(|c| c.muscle_recovery = 0.1), 0.25),
        ("uphill", world(|c| c.slope = 0.25), 0.12),
        ("headwind", world(|c| c.wind = -6.0), 0.12),
        ("chasms", world(|c| c.gaps = 1.5), 0.45),
        ("low hurdles", world(|c| c.hurdles = 0.08), 0.12),
        ("high hurdles", world(|c| c.hurdles = 0.20), 0.12),
        ("walls", world(|c| c.hurdles = 0.35), 0.12),
        ("damp", world(|c| c.mud = 0.02), 0.0),
        ("muddy", world(|c| c.mud = 0.05), 0.0),
        ("deep mud", world(|c| c.mud = 0.10), 0.12),
    ];
    for elite in elites.iter().take(400) {
        let creature = &elite.creature;
        if creature.nodes.len() > 8 {
            continue;
        }
        let calm = mean_distance(creature, &base);
        if calm < 1.5 {
            continue;
        }
        // The chasms' first pit is far off, so that world runs 12 s.
        let long = Config {
            duration: 12.0,
            ..base.clone()
        };
        let solid = mean_distance(creature, &long);
        let losses: Vec<(&str, f32)> = worlds
            .iter()
            .map(|(name, cfg, _)| {
                if *name == "chasms" {
                    let over = Config {
                        gaps: 1.5,
                        ..long.clone()
                    };
                    (
                        *name,
                        (solid - mean_distance(creature, &over)) / solid * calm,
                    )
                } else {
                    (*name, calm - mean_distance(creature, cfg))
                }
            })
            .collect();
        if std::env::var_os("WALKER_VERBOSE").is_some() {
            eprintln!("calm {calm:.2} m, losses {losses:?}");
        }
        if !worlds
            .iter()
            .zip(&losses)
            .all(|((_, _, need), (_, loss))| *loss >= need * calm)
        {
            continue;
        }
        let mut checked = creature.clone();
        perturb(&mut checked);
        let check = nudge_sensitivity(&checked, 1.0);
        let fine = nudge_sensitivity(creature, 5.0);
        eprintln!(
            "chosen: {} nodes, {} muscles, archive {:.2} m, calm 5 s {calm:.2} m, losses {losses:?}, nudge sensitivity {check:.4} m (1 s check), {fine:.4} m (5 s fine)",
            creature.nodes.len(),
            creature.muscles.len(),
            elite.fitness
        );
        println!("{}", serde_json::to_string(creature)?);
        return Ok(());
    }
    anyhow::bail!("no elite loses enough distance to every effect")
}
