//! Host breeding cost with no GPU.
//!
//! It takes archives from a save or grows them with a fake evaluator, then
//! breeds blocks of children through the game's own planning and emitting code.
//! It prints the thread time and cycles per child by emitter, by class
//! (parametric, structural, random body) and by structural operator, with the
//! plan time per slot, the allocations per child and the body size of the
//! children. For each requested thread count it also prints one 64-bit digest
//! of every plan and child, and the digests must agree across thread counts
//! and across runs of one seed.
//!
//! ```text
//! cargo run --release --example breed_bench -- [--save PATH | --grow GENERATIONS]
//!     [--population 200000] [--target-nodes 9] [--blocks 10] [--block 100000]
//!     [--seed 7] [--digest-threads 1,14]
//! ```
use anyhow::{Context, Result, bail, ensure};
use evolution_simulator::{
    config::Config,
    evolution::{self, CandidatePlan, Creature, Population, StoredCreature},
    qd::{self, Emitter, EvaluationMetrics, TrialMetrics},
    storage::{self, Experiment},
};
use rayon::prelude::*;
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---- allocation counting -------------------------------------------------

// The game's global allocator counts allocations per thread and in total once
// `count_allocations` has run.
use evolution_simulator::block_alloc::{allocations, count_allocations, total_allocations};

// ---- clocks --------------------------------------------------------------

fn clock_ns(clock: libc::clockid_t) -> u64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid clock id and a valid timespec.
    unsafe { libc::clock_gettime(clock, &mut t) };
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
/// CPU time of this thread, in nanoseconds.
fn thread_ns() -> u64 {
    clock_ns(libc::CLOCK_THREAD_CPUTIME_ID)
}
/// CPU time of the whole process, in nanoseconds.
fn process_ns() -> u64 {
    clock_ns(libc::CLOCK_PROCESS_CPUTIME_ID)
}

/// The first fields of `perf_event_attr` (`PERF_ATTR_SIZE_VER1`, 72 bytes).
/// `kind` is the C field `type`.
#[repr(C)]
struct PerfAttr {
    kind: u32,
    size: u32,
    config: u64,
    sample_period: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup_events: u32,
    bp_type: u32,
    config1: u64,
    config2: u64,
}
thread_local! {
    /// This thread's cycle counter, a perf file descriptor. It is -2 before the
    /// first call of `thread_cycles` and -1 when the counter could not be opened.
    static CYCLES_FD: Cell<i32> = const { Cell::new(-2) };
}
/// Cycles this thread has run, or `None` without a counter. The first call on
/// a thread opens its counter. The TSC is not used because it does not follow
/// the core clock here.
fn thread_cycles() -> Option<u64> {
    let mut fd = CYCLES_FD.with(Cell::get);
    if fd == -2 {
        let attr = PerfAttr {
            kind: 0,   // PERF_TYPE_HARDWARE
            config: 0, // PERF_COUNT_HW_CPU_CYCLES
            size: std::mem::size_of::<PerfAttr>() as u32,
            sample_period: 0,
            sample_type: 0,
            read_format: 0,
            // Bit 6 is `exclude_hv`. Kernel time counts too, because the
            // thread clock that the cycles are spread over includes it. Only
            // the hypervisor is left out.
            flags: 1 << 6,
            wakeup_events: 0,
            bp_type: 0,
            config1: 0,
            config2: 0,
        };
        // SAFETY: the attribute is a valid, zero-padded perf_event_attr of
        // the size it states; pid 0 and cpu -1 count this thread anywhere.
        fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &attr as *const PerfAttr,
                0,
                -1,
                -1,
                0,
            )
        } as i32;
        let fd = if fd < 0 { -1 } else { fd };
        CYCLES_FD.with(|c| c.set(fd));
    }
    let fd = CYCLES_FD.with(Cell::get);
    if fd < 0 {
        return None;
    }
    let mut value = 0u64;
    // SAFETY: reading 8 bytes into a u64 from a perf counter fd.
    let n = unsafe { libc::read(fd, (&mut value as *mut u64).cast(), 8) };
    (n == 8).then_some(value)
}

// ---- options -------------------------------------------------------------

/// The command line, as the usage at the top of this file shows it.
struct Options {
    /// A save to take the archives from. Without one they are grown.
    save: Option<PathBuf>,
    /// Generations to grow the archives for when there is no save. The default
    /// is 30.
    grow: u32,
    /// Creatures per generation while growing.
    population: usize,
    /// Blocks of children to breed and time.
    blocks: usize,
    /// Children in each block.
    block: usize,
    /// Seed of the grown archives.
    seed: u64,
    /// Thread counts to run the digest at, each in a pool of its own.
    digest_threads: Vec<usize>,
    /// The node count that `fake_evaluate` scores highest.
    target_nodes: f32,
}
/// Reads the command line. An unknown or malformed argument is an error.
fn options() -> Result<Options> {
    let mut o = Options {
        save: None,
        grow: 30,
        population: 200_000,
        blocks: 10,
        block: 100_000,
        seed: 7,
        digest_threads: vec![1, 14],
        target_nodes: 9.0,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--save" => o.save = Some(PathBuf::from(value()?)),
            "--grow" => o.grow = value()?.parse()?,
            "--population" => o.population = value()?.parse()?,
            "--blocks" => o.blocks = value()?.parse()?,
            "--block" => o.block = value()?.parse()?,
            "--seed" => o.seed = value()?.parse()?,
            "--target-nodes" => o.target_nodes = value()?.parse()?,
            "--digest-threads" => {
                o.digest_threads = value()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::parse)
                    .collect::<Result<_, _>>()?
            }
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(o)
}

// ---- the archives --------------------------------------------------------

/// The splitmix64 finalizer, a bijection on 64 bits. `finalize` in
/// `src/evolution.rs` is the same function, and it is private.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
/// A 64-bit hash of a body's genes: the three gene counts and every field of
/// every gene.
fn gene_hash(
    nodes: &[evolution::NodeGene],
    bones: &[evolution::Bone],
    muscles: &[evolution::Muscle],
) -> u64 {
    let mut h = mix(nodes.len() as u64 ^ (bones.len() as u64) << 16 ^ (muscles.len() as u64) << 32);
    let mut eat = |w: u32| h = mix(h ^ w as u64);
    for n in nodes {
        for v in [n.x, n.y, n.diameter, n.friction] {
            eat(v.to_bits());
        }
    }
    for w in bytemuck::cast_slice::<_, u32>(bones) {
        eat(*w);
    }
    for w in bytemuck::cast_slice::<_, u32>(muscles) {
        eat(*w);
    }
    h
}

/// A stand-in for the GPU. The score is a deterministic function of the genes
/// that favors bodies near `target` nodes and 2.2 times `target` muscles, plus
/// noise hashed from the genes. The behavior metrics are hashed from the genes
/// too. The distance at the screen is 0.3 of the score, and a creature below
/// the screen bar in `cfg` counts as screened. It only has to give the
/// archives the shape and body sizes of a real search.
fn fake_evaluate(population: &Population, cfg: &Config, target: f32) -> Vec<EvaluationMetrics> {
    (0..population.genomes.len())
        .into_par_iter()
        .map(|i| {
            let g = &population.genomes[i];
            let nodes = &population.nodes[g.node_start..g.node_start + g.node_count];
            let bones = &population.bones[g.bone_start..g.bone_start + g.bone_count];
            let muscles = &population.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
            let h = gene_hash(nodes, bones, muscles);
            // A number in [0, 1) hashed from the genes, one for each salt `k`.
            let unit = |k: u32| (mix(h ^ k as u64) >> 40) as f32 / 16_777_216.0;
            let size = 1.0
                - 0.12 * (g.node_count as f32 - target).abs()
                - 0.02 * (g.muscle_count as f32 - 2.2 * target).abs();
            let fitness = (size + 3.0 * unit(1) * unit(2)).max(0.0);
            let screen_x = 0.3 * fitness;
            let screened = cfg.screen.is_some_and(|s| screen_x < s.bar);
            EvaluationMetrics {
                fitness,
                behavior: TrialMetrics {
                    ground_contact: unit(3),
                    vertical_oscillation: 0.3 * unit(4),
                    gait_frequency: 0.3 + 3.0 * unit(5),
                    mean_height: 0.1 + 0.8 * unit(6),
                    feet: 1.0 + (unit(7) * 4.0).floor(),
                },
                screened,
                excluded: false,
                screen_x,
                fine: false,
                trace: Default::default(),
            }
        })
        .collect()
}

/// Grows archives from a new game: `o.grow` generations of the production
/// search, scored by `fake_evaluate`. It prints progress every 5 generations
/// and after the last.
fn grown(o: &Options) -> Result<Experiment> {
    // `random_seed: false` keeps `seed`, so one seed grows the same archives
    // every time.
    let cfg = Config {
        population: o.population,
        seed: o.seed,
        random_seed: false,
        ..Config::default()
    };
    let mut e = Experiment::new(cfg)?;
    let started = Instant::now();
    let target = o.target_nodes;
    for generation in 0..o.grow {
        let mut evaluate = |p: &Population, c: &Config| -> Result<Vec<EvaluationMetrics>> {
            Ok(fake_evaluate(p, c, target))
        };
        e.run_generation(&mut evaluate)?;
        if generation % 5 == 4 || generation + 1 == o.grow {
            let (elites, nodes, muscles) = elite_sizes(&e);
            eprintln!(
                "grown generation {:>3}: {elites} island elites, mean {nodes:.1} nodes and {muscles:.1} muscles, {:.0} s",
                e.generation,
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(e)
}

/// Loads a save. An older physics version loads too: breeding reads only
/// the archives and search state, not the physics their scores came from.
fn loaded(path: &Path) -> Result<Experiment> {
    let header = storage::peek(path)?;
    let experiment = storage::load_any_version(path)?;
    eprintln!(
        "loaded {} (physics version {}, this game {})",
        path.display(),
        header.qd_version,
        qd::VERSION
    );
    Ok(experiment)
}

/// The elites in `islands` (the island archives and their nurseries): their
/// number and their mean node and muscle counts.
fn elite_sizes(e: &Experiment) -> (usize, f64, f64) {
    let elites: Vec<Creature> = e
        .islands
        .iter()
        .flat_map(|a| a.entries.iter().map(|x| x.creature.unpack()))
        .collect();
    let n = elites.len().max(1) as f64;
    (
        elites.len(),
        elites.iter().map(|c| c.nodes.len()).sum::<usize>() as f64 / n,
        elites.iter().map(|c| c.muscles.len()).sum::<usize>() as f64 / n,
    )
}

/// The bytes a `StoredCreature` holds: the struct and its one allocation of
/// genes, without the allocator's own overhead.
fn stored_bytes(c: &StoredCreature) -> usize {
    std::mem::size_of::<StoredCreature>()
        + c.node_count() * std::mem::size_of::<evolution::NodeGene>()
        + c.bone_count() * std::mem::size_of::<evolution::Bone>()
        + c.muscle_count() * std::mem::size_of::<evolution::Muscle>()
}

// ---- measurement ---------------------------------------------------------

/// The number of structural operators: the classic ones and the anatomy ones.
fn operator_count() -> usize {
    evolution::structural_operator_names().len()
}
/// The totals for one line of the report: children, and the thread time,
/// cycles and allocations they used. The report has a line for all children,
/// for each of 3 classes, for each of 4 emitters, for each structural operator
/// and one for structural children where no operator fit.
#[derive(Clone, Copy, Default)]
struct Row {
    children: u64,
    ns: u64,
    cycles: f64,
    allocations: u64,
}
impl Row {
    fn add(&mut self, ns: u64, cycles: f64, allocations: u64) {
        self.children += 1;
        self.ns += ns;
        self.cycles += cycles;
        self.allocations += allocations;
    }
    fn merge(&mut self, other: &Row) {
        self.children += other.children;
        self.ns += other.ns;
        self.cycles += other.cycles;
        self.allocations += other.allocations;
    }
}
/// The totals for a run of children, by emitter, by class and by operator.
/// The tallies of separate runs merge into one.
#[derive(Clone)]
struct Tally {
    /// By `Emitter::index`.
    emitter: [Row; 4],
    /// Parametric (no structural change), structural, and random body (the
    /// `Restart` emitter), in that order.
    class: [Row; 3],
    /// By structural operator in the order of `structural_operator_names`,
    /// then a last row for structural children where no operator fit.
    operator: Vec<Row>,
    /// Every child.
    all: Row,
    /// The node count of each child.
    nodes: Vec<u16>,
    /// The muscle count of each child.
    muscles: Vec<u16>,
    /// False once any run had no cycle reading.
    cycles_known: bool,
}
impl Tally {
    fn new() -> Self {
        Self {
            emitter: Default::default(),
            class: Default::default(),
            operator: vec![Row::default(); operator_count() + 1],
            all: Row::default(),
            nodes: Vec::new(),
            muscles: Vec::new(),
            cycles_known: true,
        }
    }
    fn merge(mut self, other: Tally) -> Tally {
        for (a, b) in self.emitter.iter_mut().zip(&other.emitter) {
            a.merge(b);
        }
        for (a, b) in self.class.iter_mut().zip(&other.class) {
            a.merge(b);
        }
        for (a, b) in self.operator.iter_mut().zip(&other.operator) {
            a.merge(b);
        }
        self.all.merge(&other.all);
        self.nodes.extend(other.nodes);
        self.muscles.extend(other.muscles);
        self.cycles_known &= other.cycles_known;
        self
    }
}

/// The cost of one read of the thread clock, in nanoseconds. The two reads
/// around a child add about this much to its time, so each child's time has it
/// subtracted.
fn clock_overhead() -> u64 {
    let runs = 2000;
    let start = thread_ns();
    for _ in 0..runs {
        std::hint::black_box(thread_ns());
    }
    (thread_ns() - start) / runs
}

/// Breeds one block of children with `breed_child` on the rayon pool and
/// returns what they cost, with `overhead` subtracted from each child's time.
/// A task breeds 512 children one after another into one `Creature`. The cycle
/// counter is read around the task, and its cycles are spread over the
/// children in proportion to their thread time.
fn time_block(
    e: &Experiment,
    plans: &[CandidatePlan],
    slots: &[usize],
    round: u64,
    overhead: u64,
) -> Tally {
    const CHUNK: usize = 512;
    plans
        .par_chunks(CHUNK)
        .zip(slots.par_chunks(CHUNK))
        .map(|(plans, slots)| {
            let mut tally = Tally::new();
            // Per child: emitter index, thread ns, allocations, operator and
            // whether it was structural. They wait here because the cycles per
            // ns are known only when the run ends.
            let mut records: Vec<(usize, u64, u64, Option<u16>, bool)> =
                Vec::with_capacity(plans.len());
            let mut child = Creature::default();
            let (c0, t0) = (thread_cycles(), thread_ns());
            for (&plan, &slot) in plans.iter().zip(slots) {
                let a = allocations();
                let t = thread_ns();
                let trace = evolution::breed_child(
                    &e.islands,
                    &e.cma_emitters,
                    plan,
                    slot,
                    &e.config,
                    e.generation,
                    round,
                    &mut child,
                );
                let ns = (thread_ns() - t).saturating_sub(overhead);
                let allocs = allocations() - a;
                tally.nodes.push(child.nodes.len() as u16);
                tally.muscles.push(child.muscles.len() as u16);
                std::hint::black_box(&child);
                records.push((
                    plan.emitter.index(),
                    ns,
                    allocs,
                    trace.operator,
                    trace.structural,
                ));
            }
            let (c1, t1) = (thread_cycles(), thread_ns());
            // Cycles per thread nanosecond over this run of children.
            let rate = match (c0, c1) {
                (Some(a), Some(b)) if t1 > t0 => (b - a) as f64 / (t1 - t0) as f64,
                _ => {
                    tally.cycles_known = false;
                    0.0
                }
            };
            for (emitter, ns, allocs, operator, structural) in records {
                let cycles = ns as f64 * rate;
                tally.emitter[emitter].add(ns, cycles, allocs);
                // 0 parametric, 1 structural, 2 random body.
                let class = if emitter == Emitter::Restart.index() {
                    2
                } else if structural {
                    1
                } else {
                    0
                };
                tally.class[class].add(ns, cycles, allocs);
                if structural {
                    let row = operator.map_or(tally.operator.len() - 1, usize::from);
                    tally.operator[row].add(ns, cycles, allocs);
                }
                tally.all.add(ns, cycles, allocs);
            }
            tally
        })
        .reduce(Tally::new, Tally::merge)
}

/// The ring slots of block number `k`: `block` slots in a row from
/// `k * block` modulo `ring`, the ring length.
fn slots_of(block: usize, k: usize, ring: usize) -> Vec<usize> {
    let first = (k * block) % ring.max(1);
    (first..first + block).collect()
}

/// Breeds the blocks with the production code (`Population::breed`) and
/// returns one hash of every plan and every child, the children in slot order.
/// It prints the allocations per child that each block made.
fn digest(start: &Experiment, o: &Options) -> Result<u64> {
    let mut e = start.clone();
    let ring = e.ring_len();
    let mut h = 0u64;
    // One arena for every block, reused as the ring reuses a block's.
    let mut population = Population::default();
    for k in 0..o.blocks {
        let slots = slots_of(o.block, k, ring);
        let (plans, round) = e.plan_for_bench(&slots);
        for p in &plans {
            h = mix(h ^ p.emitter.index() as u64);
            for x in [p.parent, p.cma, p.mate] {
                h = mix(h ^ x.map_or(u64::MAX, |v| v as u64));
            }
        }
        let positions: Vec<usize> = (0..slots.len()).collect();
        let before = total_allocations();
        population.breed(
            slots.len(),
            None,
            &mut [],
            &e.islands,
            &e.cma_emitters,
            &plans,
            &slots,
            &positions,
            &e.config,
            e.generation,
            round,
        );
        // Every thread's allocations while the block is bred into its arena.
        println!(
            "block {k}: {:.4} allocations per child in Population::breed ({} children)",
            (total_allocations() - before) as f64 / slots.len() as f64,
            slots.len()
        );
        for g in &population.genomes {
            h = mix(h
                ^ gene_hash(
                    &population.nodes[g.node_start..g.node_start + g.node_count],
                    &population.bones[g.bone_start..g.bone_start + g.bone_count],
                    &population.muscles[g.muscle_start..g.muscle_start + g.muscle_count],
                )
                ^ g.id);
        }
    }
    Ok(h)
}

/// The value `p` of the way through `values` when sorted, from 0 (smallest) to
/// 1 (largest), or 0 when `values` is empty. It sorts `values` in place.
fn percentile(values: &mut [u16], p: f64) -> u16 {
    values.sort_unstable();
    values
        .get(((values.len() as f64 - 1.0) * p).round() as usize)
        .copied()
        .unwrap_or(0)
}

/// Prints one line of the report: the row's children, their share of `total`,
/// and the mean thread ns, cycles and allocations per child. `cycles` says
/// whether the cycle counter worked. A row with no children prints nothing.
fn print_row(name: &str, row: &Row, total: u64, cycles: bool) {
    if row.children == 0 {
        return;
    }
    let n = row.children as f64;
    let cycles = if cycles {
        format!("{:>9.0}", row.cycles / n)
    } else {
        format!("{:>9}", "-")
    };
    println!(
        "{name:<28} {:>9} {:>6.1}% {:>9.0} {cycles} {:>8.3}",
        row.children,
        100.0 * n / total.max(1) as f64,
        row.ns as f64 / n,
        row.allocations as f64 / n
    );
}

fn main() -> Result<()> {
    count_allocations();
    let o = options()?;
    let threads = rayon::current_num_threads();
    let start = match &o.save {
        Some(path) => loaded(path)?,
        None => grown(&o)?,
    };
    let (elites, nodes, muscles) = elite_sizes(&start);
    println!(
        "archives: generation {}, {} island arenas, {elites} elites, mean {nodes:.1} nodes and {muscles:.1} muscles, {} CMA emitters, ring {}",
        start.generation,
        start.islands.len(),
        start.cma_emitters.len(),
        start.ring_len()
    );
    // The creatures the experiment holds: elites, lineage records and CMA
    // templates. An elite and a lineage record are a `StoredCreature`, which
    // holds exactly its genes. A CMA template is a whole `Creature`, which has
    // inline arrays, so it counts at that size.
    let stored = [
        (
            "island",
            start.islands.iter().map(|a| a.entries.len()).sum::<usize>(),
        ),
        ("global", start.archive.entries.len()),
        ("lineage", start.lineage.len()),
        ("CMA templates", start.cma_emitters.len()),
    ];
    let count: usize = stored.iter().map(|s| s.1).sum();
    let bytes = start
        .islands
        .iter()
        .flat_map(|a| &a.entries)
        .chain(&start.archive.entries)
        .map(|e| stored_bytes(&e.creature))
        .sum::<usize>()
        + start
            .lineage
            .values()
            .map(|a| stored_bytes(&a.creature))
            .sum::<usize>()
        + start.cma_emitters.len() * std::mem::size_of::<Creature>();
    println!(
        "stored creatures: {} = {count}, {:.0} MB at {:.0} B each on average",
        stored
            .iter()
            .map(|(name, n)| format!("{n} {name}"))
            .collect::<Vec<_>>()
            .join(" + "),
        bytes as f64 / 1e6,
        bytes as f64 / count.max(1) as f64
    );
    // Timing: plan each block, then breed it child by child on the pool.
    // Planning changes the experiment, so it plans on a copy and `start` stays
    // as it was for the digests.
    let overhead = clock_overhead();
    let mut e = start.clone();
    let ring = e.ring_len();
    let mut tally = Tally::new();
    let (mut plan_wall, mut plan_cpu, mut planned) = (0u64, 0u64, 0usize);
    for k in 0..o.blocks {
        let slots = slots_of(o.block, k, ring);
        let (w, c) = (Instant::now(), process_ns());
        let (plans, round) = e.plan_for_bench(&slots);
        plan_cpu += process_ns() - c;
        plan_wall += w.elapsed().as_nanos() as u64;
        planned += slots.len();
        tally = tally.merge(time_block(&e, &plans, &slots, round, overhead));
    }
    let total = tally.all.children;
    println!(
        "\n{} blocks of {} children on {threads} threads; thread clock overhead {overhead} ns per child subtracted",
        o.blocks, o.block
    );
    println!(
        "plan: {:.0} ns wall and {:.0} ns of CPU per slot",
        plan_wall as f64 / planned as f64,
        plan_cpu as f64 / planned as f64
    );
    let cycles = tally.cycles_known;
    if !cycles {
        println!("cycles: no per-thread cycle counter (perf_event_open failed)");
    }
    println!(
        "\n{:<28} {:>9} {:>7} {:>9} {:>9} {:>8}",
        "row", "children", "share", "ns", "cycles", "allocs"
    );
    print_row("all children", &tally.all, total, cycles);
    for (name, row) in ["parametric", "structural", "random body"]
        .iter()
        .zip(&tally.class)
    {
        print_row(name, row, total, cycles);
    }
    println!();
    for (emitter, row) in Emitter::ALL.iter().zip(&tally.emitter) {
        print_row(&format!("emitter {:?}", emitter), row, total, cycles);
    }
    println!();
    // The operators by mean thread time per child, slowest first. Their share
    // is out of the structural children.
    let names = evolution::structural_operator_names();
    let none = names.len();
    let mut order: Vec<usize> = (0..=none).collect();
    order.sort_by(|&a, &b| {
        let per = |i: usize| tally.operator[i].ns as f64 / tally.operator[i].children.max(1) as f64;
        per(b).total_cmp(&per(a))
    });
    let structural = tally.class[1].children;
    for i in order {
        let name = if i == none {
            "(no operator fit)"
        } else {
            names[i]
        };
        print_row(name, &tally.operator[i], structural, cycles);
    }
    println!(
        "\nchildren: nodes mean {:.1} p90 {}, muscles mean {:.1} p90 {}",
        tally.nodes.iter().map(|&n| n as f64).sum::<f64>() / total as f64,
        percentile(&mut tally.nodes, 0.9),
        tally.muscles.iter().map(|&n| n as f64).sum::<f64>() / total as f64,
        percentile(&mut tally.muscles, 0.9),
    );
    // Digests with the production emit path at each thread count.
    let mut digests = Vec::new();
    for &n in &o.digest_threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
        let started = Instant::now();
        let d = pool.install(|| digest(&start, &o))?;
        println!(
            "digest at {n:>2} threads: {d:016x} ({:.1} s)",
            started.elapsed().as_secs_f64()
        );
        digests.push(d);
    }
    ensure!(
        digests.windows(2).all(|w| w[0] == w[1]),
        "the digests differ between thread counts"
    );
    Ok(())
}
