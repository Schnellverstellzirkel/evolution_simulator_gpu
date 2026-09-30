//! Deterministic A/B harness for the evolutionary search, on the GPU.
//!
//! Runs the game's ring (`ring::Ring` over the scheduler: every block is
//! scored on the GPU with the early screen, the creatures that would set an
//! island record get their confirmation trial, and the block is absorbed and
//! bred again) on fixed seeds and prints comparable metrics, so a search
//! change can be measured with the same command before and after. The GPU score is final. There is no CPU mode: a
//! machine whose primary GPU does not open fails instead of falling back.
//! Run it with `EVOLUTION_DEVICES=primary` and the GPU lock held shared.
//!
//! Usage:
//!   cargo run --release --example search_ab -- [generations] [population] [duration] [seed,seed,...] [--tag NAME]
//!   cargo run --release --example search_ab -- <tag> [generations] [population] [duration] [seed,seed,...]
//! Defaults: 2 generations, 64 creatures, 1.0 s trials, seeds 38,39.
//! Wall time goes to stderr so stdout is deterministic and diffable.
use anyhow::{Context, Result};
use evolution_simulator::{
    config::Config, engine, evolution::Population, physics, ring::Ring, scheduler,
    storage::Experiment,
};
use std::{
    collections::{BTreeMap, HashMap},
    time::Instant,
};

const DEFAULT_GENERATIONS: u32 = 2;
const DEFAULT_POPULATION: usize = 64;
const DEFAULT_DURATION: f32 = 1.0;
const DEFAULT_SEEDS: &[u64] = &[38, 39];
const TOP_BODIES: usize = 50;

struct Options {
    generations: u32,
    population: usize,
    duration: f32,
    seeds: Vec<u64>,
    tag: Option<String>,
    seed_offset: u64,
    probe: bool,
    save: Option<String>,
    /// `--effect Name=level`: environment effects to apply (level index).
    effects: Vec<(String, usize)>,
}

fn usage() -> &'static str {
    "usage: search_ab [tag] [generations] [population] [duration_seconds] [seed,seed,...] [--tag NAME] [--seed-offset N] [--effect Name=level]"
}

fn options() -> Result<Options> {
    let mut positionals: Vec<String> = Vec::new();
    let mut tag = None;
    let mut probe = false;
    let mut seed_offset = 0u64;
    let mut save = None;
    let mut effects: Vec<(String, usize)> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--probe" {
            probe = true;
        } else if arg == "--seed-offset" {
            seed_offset = args
                .next()
                .context("--seed-offset needs a number")?
                .parse()
                .context("seed offset")?;
        } else if arg == "--effect" {
            let spec = args.next().context("--effect needs Name=level")?;
            let (name, level) = spec.split_once('=').context("--effect needs Name=level")?;
            effects.push((name.to_owned(), level.parse().context("effect level")?));
        } else if arg == "--save" {
            save = Some(args.next().context("--save needs a path")?);
        } else if arg == "--tag" {
            tag = Some(args.next().context("--tag needs a name")?);
        } else if arg == "--help" || arg == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        } else if arg.starts_with("--") {
            anyhow::bail!("unknown option {arg}\n{}", usage());
        } else {
            positionals.push(arg);
        }
    }
    if tag.is_none()
        && positionals
            .first()
            .is_some_and(|arg| arg.parse::<u32>().is_err())
    {
        tag = Some(positionals.remove(0));
    }
    anyhow::ensure!(positionals.len() <= 4, "too many arguments\n{}", usage());
    let generations = positionals
        .first()
        .map(|arg| arg.parse().context("generations"))
        .transpose()?
        .unwrap_or(DEFAULT_GENERATIONS);
    let population = positionals
        .get(1)
        .map(|arg| arg.parse().context("population"))
        .transpose()?
        .unwrap_or(DEFAULT_POPULATION);
    let duration = positionals
        .get(2)
        .map(|arg| arg.parse().context("duration"))
        .transpose()?
        .unwrap_or(DEFAULT_DURATION);
    let seeds = match positionals.get(3) {
        Some(list) => list
            .split(',')
            .map(|seed| seed.trim().parse().context("seed"))
            .collect::<Result<Vec<u64>>>()?,
        None => DEFAULT_SEEDS.to_vec(),
    };
    anyhow::ensure!(!seeds.is_empty(), "need at least one seed");
    Ok(Options {
        generations,
        population,
        duration,
        seeds,
        tag,
        seed_offset,
        probe,
        save,
        effects,
    })
}

fn main() -> Result<()> {
    // Low priority and the shared half-machine budget, like every other run.
    engine::lower_thread_priority();
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(engine::rayon_threads())
        .start_handler(|_| engine::lower_thread_priority())
        .build_global();
    let options = options()?;
    let scope = options.tag.as_deref().unwrap_or("untagged");
    println!(
        "search_ab {scope}: {} generations, population {}, {:.2} s trials, seeds {}",
        options.generations,
        options.population,
        options.duration,
        options
            .seeds
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("{scope} seed generation best_m qd_score cells mean_nodes mean_muscles");
    let started = Instant::now();
    let (mut distances, mut scores) = (Vec::new(), Vec::new());
    for &seed in &options.seeds {
        let seed_started = Instant::now();
        let (best, qd) = run_seed(seed, &options, scope)?;
        eprintln!(
            "search_ab: seed {seed} in {:.1} s wall, {:.1} s CPU so far",
            seed_started.elapsed().as_secs_f64(),
            cpu_seconds()
        );
        distances.push(best);
        scores.push(qd as f32);
    }
    paired_summary(&mut distances, &mut scores);
    eprintln!(
        "search_ab: {} seeds in {:.1} s wall, {:.1} s CPU",
        options.seeds.len(),
        started.elapsed().as_secs_f64(),
        cpu_seconds(),
    );
    Ok(())
}

/// Processor time of this process (user and system), which other work on a
/// shared machine disturbs less than the wall clock.
fn cpu_seconds() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage fills the struct it is given.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

/// Scores the ring's first 20,000 creatures whole and in odd shuffled
/// chunks, standard and confirmation trials, and reports creatures whose
/// result differs by a bit.
fn probe_gpu(sched: &mut scheduler::Scheduler, experiment: &Experiment) -> Result<()> {
    let n = experiment.ring_len().min(20_000);
    let mut ring = Population::default();
    for i in 0..n {
        ring.push(experiment.creature(i));
    }
    let indices: Vec<usize> = (0..n).collect();
    let mut order: Vec<usize> = indices.clone();
    order.sort_by_key(|&i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 7);
    let standard = experiment.config.clone();
    let fine = scheduler::confirm_config(&standard);
    for (name, cfg) in [("standard", &standard), ("confirmation", &fine)] {
        let whole = sched.evaluate(&ring, &indices, cfg)?;
        let mut mismatched = 0;
        let mut behavior = 0;
        for chunk in order.chunks(2999) {
            let part = sched.evaluate(&ring, chunk, cfg)?;
            for (&i, m) in chunk.iter().zip(&part) {
                if m.fitness.to_bits() != whole[i].fitness.to_bits() {
                    mismatched += 1;
                } else if m.behavior.mean_height.to_bits()
                    != whole[i].behavior.mean_height.to_bits()
                    || m.behavior.ground_contact.to_bits()
                        != whole[i].behavior.ground_contact.to_bits()
                {
                    behavior += 1;
                }
            }
        }
        println!(
            "probe {name}: {n} creatures, {mismatched} fitness bits differ, {behavior} behavior bits differ"
        );
    }
    Ok(())
}

/// Runs the game's loop for one seed and returns its archive best and QD score.
fn run_seed(seed: u64, options: &Options, scope: &str) -> Result<(f32, f64)> {
    let cfg = Config {
        population: options.population,
        duration: options.duration,
        random_seed: false,
        seed: seed + options.seed_offset,
        ..Config::default()
    };
    let mut cfg = cfg;
    for (name, level) in &options.effects {
        let effect = evolution_simulator::environment::EFFECTS
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(name))
            .with_context(|| format!("no effect named {name}"))?;
        effect.set_level(&mut cfg, *level);
    }
    cfg.validate()
        .with_context(|| format!("seed {seed} configuration"))?;
    let mut gpu = evolution_simulator::gpu::Gpu::new("RTX 4060")?;
    if let Some(warning) = &gpu.startup_warning {
        anyhow::bail!("the primary GPU did not open: {warning}");
    }
    let mut experiment = Experiment::new(cfg).with_context(|| format!("seed {seed} experiment"))?;
    let mut best = f32::NAN;
    let mut top = Vec::new();
    // Generations whose global best elite was born in a hub slot.
    let mut hub_best: Vec<u32> = Vec::new();
    let mut ring = Ring::default();
    for generation in 0..options.generations {
        {
            // The game's path: the scheduler runs the blocks of the ring with
            // the early screen and the confirmation trials, and the GPU score
            // is final.
            let sched = gpu.sched.as_mut().expect("scheduler");
            if !ring.active() {
                ring.start(&mut experiment, sched);
            }
            while experiment.generation == generation {
                sched.pump()?;
                ring.step(
                    &mut experiment,
                    sched,
                    std::time::Duration::from_millis(4),
                    usize::MAX,
                )?;
            }
            if options.probe && std::env::var_os("PROBE_GPU").is_some() && generation >= 3 {
                probe_gpu(sched, &experiment)?;
            }
        }
        if options.probe {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for e in &experiment.archive.entries {
                e.fitness.to_bits().hash(&mut h);
                e.creature.id.hash(&mut h);
            }
            println!("trace gen {generation}: archive {:016x}", h.finish());
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for block in &experiment.blocks {
                for g in &block.population.genomes {
                    g.id.hash(&mut h);
                }
                for n in &block.population.nodes {
                    n.x.to_bits().hash(&mut h);
                }
            }
            println!("trace gen {generation}: ring {:016x}", h.finish());
        }
        let generation_best = experiment.history.last().map_or(f32::NAN, |s| s.best);
        // Mean body size of the ring: bodies that only grow make every
        // later generation slower to simulate.
        let genomes: Vec<_> = experiment
            .blocks
            .iter()
            .flat_map(|b| &b.population.genomes)
            .collect();
        let mean = |part: fn(&evolution_simulator::evolution::Genome) -> usize| {
            genomes.iter().map(|g| part(g)).sum::<usize>() as f64 / genomes.len().max(1) as f64
        };
        println!(
            "{scope} {seed} {generation} {generation_best:.2} {:.2} {} {:.2} {:.2}",
            experiment.archive.qd_score,
            experiment.archive.behavior_count(),
            mean(|g| g.node_count),
            mean(|g| g.muscle_count)
        );
        best = experiment
            .archive
            .entries
            .iter()
            .map(|elite| elite.fitness)
            .fold(best, f32::max);
        println!("{scope} seed {seed} generation {generation} archive best {best:.2} m");
        if let Some(elite) = experiment
            .archive
            .entries
            .iter()
            .max_by(|a, b| a.fitness.total_cmp(&b.fitness))
        {
            let slot = evolution_simulator::evolution::slot_of_id(elite.creature.id);
            let born = evolution_simulator::qd::island_of_slot(
                slot,
                evolution_simulator::storage::island_count(),
            );
            if born == evolution_simulator::storage::hub_island() {
                hub_best.push(generation);
            }
        }
        top = top_bodies(&experiment, TOP_BODIES);
    }
    ring.stop(gpu.sched.as_mut().expect("scheduler"));
    // `EVOLUTION_AB_SAVE=<dir>` writes each seed's final experiment as
    // `<dir>/seed-<seed>.evo`, for `physics_audit` and `size_report`.
    if let Some(dir) = std::env::var_os("EVOLUTION_AB_SAVE") {
        let path = std::path::Path::new(&dir).join(format!("seed-{seed}.evo"));
        evolution_simulator::storage::save(&path, &experiment)?;
    }
    let qd = experiment.archive.qd_score;
    let cells = experiment.archive.behavior_count();
    if best.is_finite() {
        println!("{scope} seed {seed} summary: best {best:.2} m, qd {qd:.2}, cells {cells}");
    } else {
        println!("{scope} seed {seed} summary: archive empty, qd {qd:.2}, cells {cells}");
    }
    if let Some(path) = &options.save {
        evolution_simulator::storage::save(std::path::Path::new(path), &experiment)?;
    }
    print_body_mix(scope, seed, &top);
    print_robustness(
        scope,
        seed,
        &experiment,
        gpu.sched.as_mut().expect("scheduler"),
    )?;
    if let Some(sched) = gpu.sched.as_ref() {
        println!(
            "{scope} seed {seed} confirmations: {} trials, per evaluated creature {:.5}",
            sched.confirms_submitted,
            sched.confirms_submitted as f64
                / (options.generations as f64 * options.population as f64)
        );
    }
    print_common_grid(scope, seed, &experiment);
    print_island_diversity(scope, seed, &experiment);
    print_islands(scope, seed, &experiment, &hub_best, options.generations);
    let weights = evolution_simulator::qd::emitter_weights(&experiment.emitter_stats);
    println!(
        "{scope} seed {seed} emitter shares: {}",
        evolution_simulator::qd::Emitter::ALL
            .iter()
            .map(|e| format!("{} {:.2}", e.label(), weights[e.index()]))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok((best, qd))
}

/// How much of their archive distance the 50 best global elites keep at four
/// times the rate and solver passes (full trial, no screen) from a slightly
/// perturbed pose that no run used.
fn print_robustness(
    scope: &str,
    seed: u64,
    experiment: &Experiment,
    sched: &mut scheduler::Scheduler,
) -> Result<()> {
    let mut elites: Vec<_> = experiment
        .archive
        .entries
        .iter()
        .filter(|elite| elite.fitness.is_finite() && elite.fitness > 0.0)
        .collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(TOP_BODIES);
    if elites.is_empty() {
        return Ok(());
    }
    let mut unit = Population::default();
    for elite in &elites {
        let mut creature = elite.creature.clone();
        // A second pose from the same rule: the id seeds the perturbation.
        creature.id ^= 0x9e37_79b9;
        perturb(&mut creature);
        creature.id ^= 0x9e37_79b9;
        unit.push(creature);
    }
    let standard = physics::Fidelity::standard();
    let cfg = Config {
        fidelity: Some(physics::Fidelity {
            rate: standard.rate * 4,
            bone_passes: standard.bone_passes * 4,
            velocity_passes: standard.velocity_passes * 4,
        }),
        screen: None,
        population: elites.len(),
        ..experiment.config.clone()
    };
    let indices: Vec<usize> = (0..unit.genomes.len()).collect();
    let results = sched.evaluate(&unit, &indices, &cfg)?;
    let mut kept: Vec<f32> = elites
        .iter()
        .zip(&results)
        .map(|(elite, result)| result.fitness.max(0.0) / elite.fitness)
        .collect();
    let halved = kept.iter().filter(|&&k| k < 0.5).count();
    println!(
        "{scope} seed {seed} top-{TOP_BODIES} elites under the fine check: median share kept {:.2}, below half {halved} of {}",
        median(&mut kept),
        elites.len()
    );
    Ok(())
}

/// QD score of the global archive's behavior elites re-binned on one fixed
/// grid (the archive shape before any experiment: contact 6, cadence 8,
/// height 6, feet 5), so runs whose archives have different shapes compare on
/// the same ground. Also the reserve size and the distinct body plans held.
fn print_common_grid(scope: &str, seed: u64, experiment: &Experiment) {
    let mut cells: HashMap<[u8; 4], f32> = HashMap::new();
    let mut plans = std::collections::HashSet::new();
    for elite in &experiment.archive.entries {
        plans.insert(&elite.topology);
        if evolution_simulator::qd::is_morphology_niche(&elite.niche) || elite.fitness <= 0.0 {
            continue;
        }
        let d = &elite.descriptor;
        let bin = |v: f32, high: f32, n: f32| {
            ((v.clamp(0.0, high) / high * n).floor().min(n - 1.0)) as u8
        };
        let low = 0.15f32;
        let top = (0.6 * evolution_simulator::evolution::max_bone_length()).max(2.0 * low);
        let height = ((d.mean_height.max(low) / low).ln() / (top / low).ln()).clamp(0.0, 1.0);
        let key = [
            bin(d.ground_contact, 1.0, 6.0),
            bin(d.gait_frequency, 6.0, 8.0),
            bin(height, 1.0, 6.0),
            (d.feet.round() as i32).clamp(1, 5) as u8,
        ];
        let slot = cells.entry(key).or_insert(f32::MIN);
        *slot = slot.max(elite.fitness);
    }
    let qd: f64 = cells.values().map(|&f| f as f64).sum();
    println!(
        "{scope} seed {seed} common grid: qd {qd:.2}, cells {}, reserve {}, body plans {}",
        cells.len(),
        experiment
            .islands
            .iter()
            .take(evolution_simulator::storage::island_count())
            .map(|island| island.morphology_count())
            .sum::<usize>(),
        plans.len()
    );
}

/// Each island's best distance and QD score, and when the global best
/// first came from the hub (was born in a hub slot).
fn print_islands(
    scope: &str,
    seed: u64,
    experiment: &Experiment,
    hub_best: &[u32],
    generations: u32,
) {
    let hub = evolution_simulator::storage::hub_island();
    let rows: Vec<String> = experiment
        .islands
        .iter()
        .take(evolution_simulator::storage::island_count())
        .enumerate()
        .map(|(k, island)| {
            format!(
                "{}{k} best {:.2} qd {:.0}",
                if k == hub { "hub " } else { "" },
                island.best_fitness(),
                island.qd_score
            )
        })
        .collect();
    println!("{scope} seed {seed} islands: {}", rows.join(", "));
    match hub_best.first() {
        Some(first) => println!(
            "{scope} seed {seed} global best from the hub: first at generation {first}, {} of {generations} generations",
            hub_best.len()
        ),
        None => println!("{scope} seed {seed} global best from the hub: never"),
    }
}

/// Top elites per island for the diversity report.
const ISLAND_TOP: usize = 20;

/// How much the islands' best elites have in common. For each island's
/// fastest `ISLAND_TOP` behavior elites: how many distinct body plans they
/// hold, the share whose body plan is also among another island's top
/// elites, the share that is the same creature (a migrant copy), and the
/// share whose oldest recorded ancestor is also an oldest ancestor of
/// another island's top elites (common descent). The hub, when there is
/// one, is compared with the others but left out of the means, because it
/// holds copies of the other islands' elites by design.
fn print_island_diversity(scope: &str, seed: u64, experiment: &Experiment) {
    use std::collections::HashSet;
    let hub: Option<usize> = Some(evolution_simulator::storage::hub_island());
    let tops: Vec<Vec<&evolution_simulator::qd::Elite>> = experiment
        .islands
        .iter()
        .take(evolution_simulator::storage::island_count())
        .map(|island| {
            let mut elites: Vec<_> = island
                .entries
                .iter()
                .filter(|e| !evolution_simulator::qd::is_morphology_niche(&e.niche))
                .collect();
            elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            elites.truncate(ISLAND_TOP);
            elites
        })
        .collect();
    let root = |id: u64| {
        experiment
            .ancestry(id, usize::MAX)
            .last()
            .map(|a| a.creature.id)
            .unwrap_or(id)
    };
    let plans: Vec<HashSet<&evolution_simulator::qd::Topology>> = tops
        .iter()
        .map(|top| top.iter().map(|e| &e.topology).collect())
        .collect();
    let ids: Vec<HashSet<u64>> = tops
        .iter()
        .map(|top| top.iter().map(|e| e.creature.id).collect())
        .collect();
    let roots: Vec<HashSet<u64>> = tops
        .iter()
        .map(|top| top.iter().map(|e| root(e.creature.id)).collect())
        .collect();
    let mut rows = Vec::new();
    let (mut plan_sum, mut shared_sum, mut copy_sum, mut kin_sum, mut counted) =
        (0.0, 0.0, 0.0, 0.0, 0usize);
    for (k, top) in tops.iter().enumerate() {
        if top.is_empty() {
            continue;
        }
        // Compare with the other isolated islands (all islands without a hub).
        let others: Vec<usize> = (0..tops.len())
            .filter(|&o| o != k && Some(o) != hub)
            .collect();
        let share = |hit: &dyn Fn(&evolution_simulator::qd::Elite) -> bool| {
            top.iter().filter(|e| hit(e)).count() as f64 / top.len() as f64
        };
        let shared = share(&|e| others.iter().any(|&o| plans[o].contains(&e.topology)));
        let copies = share(&|e| others.iter().any(|&o| ids[o].contains(&e.creature.id)));
        let kin = share(&|e| {
            let r = root(e.creature.id);
            others.iter().any(|&o| roots[o].contains(&r))
        });
        rows.push(format!(
            "{}{k}: plans {} shared {:.2} copies {:.2} kin {:.2}",
            if Some(k) == hub { "hub " } else { "" },
            plans[k].len(),
            shared,
            copies,
            kin
        ));
        if Some(k) != hub {
            plan_sum += plans[k].len() as f64;
            shared_sum += shared;
            copy_sum += copies;
            kin_sum += kin;
            counted += 1;
        }
    }
    let n = counted.max(1) as f64;
    println!(
        "{scope} seed {seed} island diversity (top {ISLAND_TOP}): {}; mean plans {:.1} shared {:.2} copies {:.2} kin {:.2}",
        rows.join(", "),
        plan_sum / n,
        shared_sum / n,
        copy_sum / n,
        kin_sum / n
    );
}

struct BodySize {
    nodes: usize,
    muscles: usize,
    length: f32,
    longest_bone: f32,
    mass: f32,
}

/// The `count` fastest elites of the global archive.
fn top_bodies(experiment: &Experiment, count: usize) -> Vec<BodySize> {
    let mut elites: Vec<_> = experiment
        .archive
        .entries
        .iter()
        .filter(|elite| elite.fitness.is_finite())
        .collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(count);
    elites
        .into_iter()
        .map(|elite| {
            let creature = &elite.creature;
            let mass: f32 = physics::body(&creature.nodes, &creature.bones)
                .iter()
                .map(|node| node.mass)
                .sum();
            let length: f32 = creature.bones.iter().map(|bone| bone.rest_length).sum();
            let longest_bone = creature
                .bones
                .iter()
                .map(|bone| bone.rest_length)
                .fold(0.0, f32::max);
            BodySize {
                nodes: creature.nodes.len(),
                muscles: creature.muscles.len(),
                length,
                longest_bone,
                mass,
            }
        })
        .collect()
}

fn print_body_mix(scope: &str, seed: u64, bodies: &[BodySize]) {
    if bodies.is_empty() {
        println!("{scope} seed {seed} top-{TOP_BODIES}: no scored creatures");
        return;
    }
    let mut nodes: BTreeMap<usize, usize> = BTreeMap::new();
    for body in bodies {
        *nodes.entry(body.nodes).or_default() += 1;
    }
    let mix = nodes
        .iter()
        .map(|(count, bodies)| format!("{count}x{bodies}"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut lengths: Vec<f32> = bodies.iter().map(|body| body.length).collect();
    let mut masses: Vec<f32> = bodies.iter().map(|body| body.mass).collect();
    let longest = bodies
        .iter()
        .map(|body| body.longest_bone)
        .fold(0.0, f32::max);
    println!("{scope} seed {seed} top-{TOP_BODIES} node mix: {mix}");
    println!(
        "{scope} seed {seed} top-{TOP_BODIES}: median length {:.2} m, median mass {:.2} kg, longest bone {:.2} m",
        median(&mut lengths),
        median(&mut masses),
        longest
    );
    let muscles: Vec<usize> = bodies.iter().map(|body| body.muscles).collect();
    println!(
        "{scope} seed {seed} top-{TOP_BODIES} muscles: mean {:.2}, most {}",
        muscles.iter().sum::<usize>() as f32 / muscles.len() as f32,
        muscles.iter().max().unwrap_or(&0)
    );
}

fn paired_summary(distances: &mut Vec<f32>, scores: &mut Vec<f32>) {
    let seeds = distances.len();
    distances.retain(|distance| distance.is_finite());
    scores.retain(|score| score.is_finite());
    if distances.is_empty() || scores.is_empty() {
        println!("paired across {seeds} seeds: no archive entries");
        return;
    }
    let distance_mean = mean(distances);
    let distance_median = median(distances);
    let score_mean = mean(scores);
    let score_median = median(scores);
    println!(
        "paired across {seeds} seeds: best distance mean {distance_mean:.2} m, median {distance_median:.2} m; qd score mean {score_mean:.2}, median {score_median:.2}"
    );
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

/// Upper middle entry of the sorted values, like `size_report`.
fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// Small deterministic change to a creature's starting pose and grip.
fn perturb(creature: &mut evolution_simulator::evolution::Creature) {
    let mut rng = evolution_simulator::evolution::Rng::new(creature.id ^ 0x5eed_7a11, 0, 0);
    for node in &mut creature.nodes {
        node.x += rng.range(-0.02, 0.02);
        node.y += rng.range(0.0, 0.02);
        node.friction = (node.friction * rng.range(0.9, 1.1)).clamp(0.0, 1.0);
    }
}
