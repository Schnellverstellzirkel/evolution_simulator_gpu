//! The generation dump is a developer diagnostic that
//! `EVOLUTION_DUMP_GENERATION=<generation>[:<path>]` turns on, and `=<path>`
//! alone dumps the next generation. From that generation's start one
//! generation's worth of blocks is bred with the screen bar off, so every
//! trial runs in full, and the island elites are queued to run again among
//! them without entering an archive. The file holds a header, a record per
//! elite of the archives at the start and a row per creature, written as its
//! block is absorbed. `HEADER_BYTES`, `ELITE_BYTES` and `ROW_BYTES` give the
//! layouts, which `examples/dump_stats.rs` and `examples/operator_yield.rs`
//! read.

use super::*;
use std::io::{Seek, SeekFrom};

/// Bit of a row's `entered` byte: the creature entered an island archive.
pub const ISLAND: u8 = 1;
/// Bit of a row's `entered` byte: the creature entered a nursery archive, of
/// either kind.
pub const NURSERY: u8 = 2;
/// Bit of a row's `entered` byte: the creature entered a morphology reserve.
pub const RESERVE: u8 = 4;
/// Bit of a row's `entered` byte: the creature entered the global archive.
pub const GLOBAL: u8 = 8;

/// Bit of a row's `flags` byte: the creature was produced by mating.
pub const MATE: u8 = 1;
/// Bit of a row's `flags` byte: its CMA emitter was optimizing
/// (`CmaEmitter::optimizing`).
pub const OPTIMIZER: u8 = 2;
/// Bit of a row's `flags` byte: its fitness is its confirmation trial's
/// (`EvaluationMetrics::fine`).
pub const FINE: u8 = 4;
/// Bit of a row's `flags` byte: an island elite run again in the dump's
/// generation. It enters no archive.
pub const RERUN: u8 = 8;
/// Bit of a row's `flags` byte: the early screen stopped its trial.
pub const SCREENED: u8 = 16;
/// Bit of a row's `flags` byte: its result enters no archive. The metrics say
/// so (`EvaluationMetrics::excluded`, which the dump also sets for a re-run),
/// or its block ran in a world that has since changed.
pub const EXCLUDED: u8 = 32;
/// Bit of a row's `flags` byte: the creature's parent was in a morphology
/// reserve.
pub const PARENT_RESERVE: u8 = 64;

/// The first 8 bytes of a dump file.
pub const MAGIC: &[u8; 8] = b"EVODUMP1";
/// The format number in the header. Format 1 kept an arena in one byte. The
/// game has more than 256 arenas, so format 2 holds arenas in two bytes.
pub const FORMAT: u32 = 2;
/// Size of the header at the start of the file. It is written when the dump
/// starts and again, with the row count, when the last block is absorbed. All
/// numbers in the file are little endian. The fields by byte:
/// - 0..8: `MAGIC`
/// - 8..12: format u32 (`FORMAT`)
/// - 12..16: `qd::VERSION` u32
/// - 16..20: generation u32
/// - 20..24: population u32
/// - 24..32: seed u64
/// - 32..40: rows u64
/// - 40..44: elites u32
/// - 44..48: the screen bar the generation would have had, f32
/// - 48..52: trial seconds f32
/// - 52..54: steps per second u16
/// - 54: islands u8
/// - 55: spare (format 1: arenas u8)
/// - 56..60: the cell bins (`BINS`), 4 x u8
/// - 60..62: arenas u16
/// - 62..64: 2 spare bytes
pub const HEADER_BYTES: usize = 64;
/// Size of an elite record. One follows the header for each elite of the
/// archives when the dump starts: the islands and their nurseries in the order
/// of `Experiment::islands`, then the global archive. The fields by byte:
/// - 0: arena, low byte. The arena is the archive's index in
///   `Experiment::islands`, and `u16::MAX` for the global archive (format 1:
///   255)
/// - 1: flags u8 (1 morphology reserve, 2 fine, 4 graduate, 8 re-run
///   measured)
/// - 2..4: cell u16 (`cell`)
/// - 4: nodes u8
/// - 5: muscles u8
/// - 6: emitter u8 (`Emitter::index`)
/// - 7: arena, high byte (format 1: spare)
/// - 8..16: creature id u64
/// - 16..20: fitness f32
/// - 20..32: distances at 2.5, 5 and 10 s from its re-run, 3 x f32 (NaN
///   without one)
pub const ELITE_BYTES: usize = 32;
/// Size of a creature row. One follows the elite records for each creature of
/// the dump generation, written as its block is absorbed. The fields by byte:
/// - 0..4: ring slot u32
/// - 4: emitter u8 (`Emitter::index`)
/// - 5: operator, low byte. The operator is an index in
///   `evolution::structural_operator_names` on 12 bits, 0xFFF for none. Its
///   bits 8 to 11 are the high four bits of byte 7.
/// - 6: flags u8 (`MATE`, `OPTIMIZER`, `FINE`, `RERUN`, `SCREENED`,
///   `EXCLUDED`, `PARENT_RESERVE`)
/// - 7: entered in the low four bits (`ISLAND`, `NURSERY`, `RESERVE`,
///   `GLOBAL`), operator bits 8 to 11 in the high four
/// - 8..16: parent id u64 (`u64::MAX` for none)
/// - 16..18: parent cell u16 (`u16::MAX` for none)
/// - 18..20: final cell u16, the cell its own behavior fell in
/// - 20..22: CMA emitter u16 (`u16::MAX` for none)
/// - 22: nodes u8
/// - 23: muscles u8
/// - 24: parent nodes u8
/// - 25: parent muscles u8
/// - 26..28: rhythm period f16, read off its first muscle (0 without muscles)
/// - 28..32: standard fitness f32, the standard trial's distance
/// - 32..36: archive fitness f32, the fitness the archives saw, which is the
///   confirmation trial's when it lowered the score
/// - 36..64: the seven words of `creature_kernel::RungTrace`, u32 each
pub const ROW_BYTES: usize = 64;
/// Bins of the four axes of a cell: contact, cadence, height and feet. They
/// repeat the movement grid of `qd`.
pub const BINS: [u8; 4] = [6, 8, 6, 5];

/// Whether a dump started in this process. There is one per run.
pub static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The dump that `EVOLUTION_DUMP_GENERATION` asks for, or None when the
/// variable is not set. It gives the generation (None for the next boundary)
/// and the file. A bare generation writes `runs/dump-gen<generation>.bin`.
pub fn target() -> Option<(Option<u32>, std::path::PathBuf)> {
    static TARGET: std::sync::OnceLock<Option<(Option<u32>, std::path::PathBuf)>> =
        std::sync::OnceLock::new();
    TARGET
        .get_or_init(|| {
            let value = std::env::var("EVOLUTION_DUMP_GENERATION").ok()?;
            if let Ok(generation) = value.parse::<u32>() {
                return Some((
                    Some(generation),
                    format!("runs/dump-gen{generation}.bin").into(),
                ));
            }
            match value.split_once(':') {
                Some((g, path)) if g.parse::<u32>().is_ok() => Some((g.parse().ok(), path.into())),
                _ => Some((None, value.into())),
            }
        })
        .clone()
}

fn bit(on: bool, flag: u8) -> u8 {
    if on { flag } else { 0 }
}

/// A behavior cell as one number, with the body classes left out. It is
/// `((contact * 8 + cadence) * 6 + height) * 5 + feet`. It is `u16::MAX` for
/// the niche of a morphology reserve or of an optimizer, which sit in no cell.
pub fn cell(niche: &qd::Niche) -> u16 {
    let n = niche.0;
    if qd::is_morphology_niche(niche) || n[0] >= 254 {
        return u16::MAX;
    }
    ((n[0] as u16 * BINS[1] as u16 + n[1] as u16) * BINS[2] as u16 + n[3] as u16) * BINS[3] as u16
        + n[4] as u16
}

/// What breeding knew about a child's parent.
#[derive(Clone, Copy)]
pub struct Parent {
    cell: u16,
    nodes: u8,
    muscles: u8,
    reserve: bool,
    /// The CMA emitter that sampled the child was optimizing.
    optimizer: bool,
}
impl Parent {
    /// The parent as breeding knew it, from `elite` (None for a child with no
    /// parent). `optimizer` says whether the child's CMA emitter was
    /// optimizing.
    pub fn of(elite: Option<&qd::Elite>, optimizer: bool) -> Self {
        Self {
            cell: elite.map_or(u16::MAX, |e| cell(&e.descriptor.niche())),
            nodes: elite.map_or(0, |e| e.creature.node_count().min(255) as u8),
            muscles: elite.map_or(0, |e| e.creature.muscle_count().min(255) as u8),
            reserve: elite.is_some_and(|e| qd::is_morphology_niche(&e.niche)),
            optimizer,
        }
    }
}

/// The host fields of a row, fixed when its block is bred.
#[derive(Clone, Copy)]
pub struct Head {
    /// The row's `flags` byte as bred. `write_rows` adds the bits that the
    /// result sets.
    pub flags: u8,
    slot: u32,
    emitter: u8,
    /// The index of the structural operator that changed the child in
    /// `evolution::structural_operator_names`, `u16::MAX` for none.
    operator: u16,
    parent: u64,
    parent_cell: u16,
    /// The CMA emitter that sampled the child, `u16::MAX` for none.
    cma: u16,
    nodes: u8,
    muscles: u8,
    parent_nodes: u8,
    parent_muscles: u8,
    /// The rhythm period as the bits of an f16.
    period: u16,
    id: u64,
}

/// An elite's record for the dump file, kept until the end of the dump, when
/// the re-run distances are added.
pub struct Elite {
    bytes: [u8; ELITE_BYTES],
    id: u64,
}
impl Elite {
    /// The record of `e` in `arena` (`u16::MAX` for the global archive).
    pub fn of(arena: u16, e: &qd::Elite) -> Self {
        let mut b = [0u8; ELITE_BYTES];
        [b[0], b[7]] = arena.to_le_bytes();
        b[1] = u8::from(qd::is_morphology_niche(&e.niche))
            | u8::from(e.fine) << 1
            | u8::from(e.graduate) << 2;
        b[2..4].copy_from_slice(&cell(&e.descriptor.niche()).to_le_bytes());
        b[4] = e.creature.node_count().min(255) as u8;
        b[5] = e.creature.muscle_count().min(255) as u8;
        b[6] = e.emitter.index() as u8;
        b[8..16].copy_from_slice(&e.creature.id.to_le_bytes());
        b[16..20].copy_from_slice(&e.fitness.to_le_bytes());
        for at in [20, 24, 28] {
            b[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
        Self {
            bytes: b,
            id: e.creature.id,
        }
    }
}

/// The generation dump being written: its file, and the heads of the blocks
/// bred for it until the results of those blocks arrive.
pub struct Dump {
    path: std::path::PathBuf,
    file: BufWriter<File>,
    /// The header bytes. `finish` adds the row count.
    header: [u8; HEADER_BYTES],
    /// Creatures in one generation. Breeding for the dump stops at this count.
    population: usize,
    /// Creatures bred into dump blocks, and blocks still to absorb.
    bred: usize,
    outstanding: usize,
    /// The heads of the creatures in each ring block, for a block bred for the
    /// dump and not yet absorbed.
    heads: Vec<Option<Vec<Head>>>,
    /// The heads of the block just bred, until `assign` files them.
    pending: Option<Vec<Head>>,
    elites: Vec<Elite>,
    /// The ids of the elites queued to run again.
    reruns: std::collections::HashSet<u64>,
    /// Rung distances of the re-run elites, by id.
    measured: HashMap<u64, [f32; 3]>,
    rows: u64,
    started: std::time::Instant,
}

impl Dump {
    /// Starts the dump file at `path`, creating its directory, and writes the
    /// header and the records of `elites`. `bar` is the screen bar the
    /// generation would have had, `reruns` holds the ids of the elites queued
    /// to run again, and `blocks` is the number of blocks in the ring.
    pub fn create(
        path: &Path,
        e: &Experiment,
        bar: f32,
        elites: Vec<Elite>,
        reruns: std::collections::HashSet<u64>,
        blocks: usize,
    ) -> Result<Self> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut file = BufWriter::with_capacity(
            1 << 22,
            File::create(path).with_context(|| format!("creating {}", path.display()))?,
        );
        let mut h = [0u8; HEADER_BYTES];
        h[0..8].copy_from_slice(MAGIC);
        h[8..12].copy_from_slice(&FORMAT.to_le_bytes());
        h[12..16].copy_from_slice(&qd::VERSION.to_le_bytes());
        h[16..20].copy_from_slice(&e.generation.to_le_bytes());
        h[20..24].copy_from_slice(&(e.config.population as u32).to_le_bytes());
        h[24..32].copy_from_slice(&e.config.seed.to_le_bytes());
        h[40..44].copy_from_slice(&(elites.len() as u32).to_le_bytes());
        h[44..48].copy_from_slice(&bar.to_le_bytes());
        h[48..52].copy_from_slice(&e.config.duration.to_le_bytes());
        h[52..54].copy_from_slice(&(e.config.fidelity().rate as u16).to_le_bytes());
        h[54] = island_count() as u8;
        h[56..60].copy_from_slice(&BINS);
        h[60..62].copy_from_slice(&(arena_count() as u16).to_le_bytes());
        // The header and the elites are written again at the end, with
        // the row count and the re-run distances.
        file.write_all(&h)?;
        for elite in &elites {
            file.write_all(&elite.bytes)?;
        }
        Ok(Self {
            path: path.to_owned(),
            file,
            header: h,
            population: e.config.population,
            bred: 0,
            outstanding: 0,
            heads: (0..blocks).map(|_| None).collect(),
            pending: None,
            elites,
            reruns,
            measured: HashMap::new(),
            rows: 0,
            started: std::time::Instant::now(),
        })
    }
    /// How many elites are queued to run again.
    pub fn rerun_count(&self) -> usize {
        self.reruns.len()
    }
    /// Whether creatures are still being bred for the dump.
    pub fn breeding(&self) -> bool {
        self.bred < self.population
    }
    /// Whether all breeding and block absorption is complete.
    pub fn finished(&self) -> bool {
        !self.breeding() && self.outstanding == 0
    }
    /// A block was bred for the dump at ring slot `first`. Its heads wait
    /// until `assign` files them. `parents` holds the parent of each creature.
    /// `reseeded` lists the positions of the creatures that were handed in
    /// instead of bred from a parent, such as the elites queued to run again,
    /// and their rows show no parent.
    pub fn bred(
        &mut self,
        first: usize,
        population: &Population,
        births: &[Birth],
        parents: Vec<Parent>,
        reseeded: &[usize],
    ) {
        let operators = evolution::take_operators();
        let mut heads: Vec<Head> = (0..population.genomes.len())
            .map(|j| {
                let g = &population.genomes[j];
                let birth = births[j];
                let parent = parents[j];
                let period = if g.muscle_count > 0 {
                    population.muscles[g.muscle_start].period
                } else {
                    0.0
                };
                Head {
                    flags: bit(birth.mate, MATE)
                        | bit(parent.optimizer, OPTIMIZER)
                        | bit(parent.reserve, PARENT_RESERVE),
                    slot: (first + j) as u32,
                    emitter: birth.emitter.index() as u8,
                    operator: operators.get(&g.id).copied().unwrap_or(u16::MAX),
                    parent: birth.parent_id.unwrap_or(u64::MAX),
                    parent_cell: parent.cell,
                    cma: birth.cma.map_or(u16::MAX, |c| c.min(65534) as u16),
                    nodes: g.node_count.min(255) as u8,
                    muscles: g.muscle_count.min(255) as u8,
                    parent_nodes: parent.nodes,
                    parent_muscles: parent.muscles,
                    period: crate::creature_kernel::f32_to_f16(period),
                    id: g.id,
                }
            })
            .collect();
        for &j in reseeded {
            let h = &mut heads[j];
            if self.reruns.contains(&h.id) {
                h.flags |= RERUN;
            }
            h.parent = u64::MAX;
            h.parent_cell = u16::MAX;
            h.parent_nodes = 0;
            h.parent_muscles = 0;
            h.flags &= !(MATE | OPTIMIZER | PARENT_RESERVE);
        }
        self.bred += heads.len();
        self.pending = Some(heads);
    }
    /// Files the block just bred (if it was bred for the dump) as ring
    /// block `k`.
    pub fn assign(&mut self, k: usize) {
        if let Some(heads) = self.pending.take() {
            self.heads[k] = Some(heads);
            self.outstanding += 1;
        }
    }
    /// Takes the heads for ring block `k`, if one was bred for the dump.
    pub fn take_head(&mut self, k: usize) -> Option<Vec<Head>> {
        let head = self.heads.get_mut(k)?.take()?;
        self.outstanding -= 1;
        Some(head)
    }
    /// Writes a row for each creature of an absorbed block. `heads` are the
    /// block's heads, `finals` its results and `kinds` what each creature
    /// entered. `stale` says the block ran in a world that has since changed,
    /// so its rows are excluded. The distances of a re-run elite are kept for
    /// `finish`.
    pub fn write_rows(
        &mut self,
        population: &Population,
        heads: &[Head],
        finals: &[EvaluationMetrics],
        kinds: &[u8],
        stale: bool,
    ) -> Result<()> {
        for (j, h) in heads.iter().enumerate() {
            let m = &finals[j];
            let g = &population.genomes[j];
            let nodes = &population.nodes[g.node_start..g.node_start + g.node_count];
            let muscles = &population.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
            let final_cell = cell(&qd::descriptor(nodes, muscles, m.behavior).niche());
            let flags = h.flags
                | bit(m.fine, FINE)
                | bit(m.screened, SCREENED)
                | bit(m.excluded || stale, EXCLUDED);
            if h.flags & RERUN != 0 {
                self.measured
                    .insert(h.id, [1, 2, 3].map(|r| m.trace.distance(r)));
            }
            let mut b = [0u8; ROW_BYTES];
            b[0..4].copy_from_slice(&h.slot.to_le_bytes());
            b[4] = h.emitter;
            // The operator index has 12 bits (0xFFF for none): the low
            // byte, and the high four bits above `entered`'s own four.
            b[5] = h.operator as u8;
            b[6] = flags;
            b[7] = kinds[j] | (((h.operator >> 8) as u8 & 15) << 4);
            b[8..16].copy_from_slice(&h.parent.to_le_bytes());
            b[16..18].copy_from_slice(&h.parent_cell.to_le_bytes());
            b[18..20].copy_from_slice(&final_cell.to_le_bytes());
            b[20..22].copy_from_slice(&h.cma.to_le_bytes());
            b[22] = h.nodes;
            b[23] = h.muscles;
            b[24] = h.parent_nodes;
            b[25] = h.parent_muscles;
            b[26..28].copy_from_slice(&h.period.to_le_bytes());
            b[28..32].copy_from_slice(&m.trace.fitness.to_le_bytes());
            b[32..36].copy_from_slice(&m.fitness.to_le_bytes());
            for (w, word) in m.trace.words.iter().enumerate() {
                b[36 + 4 * w..40 + 4 * w].copy_from_slice(&word.to_le_bytes());
            }
            self.file.write_all(&b)?;
        }
        self.rows += heads.len() as u64;
        Ok(())
    }
    /// Adds the row count and the re-run distances, writes the header and the
    /// elite records again at the start of the file and syncs it. Returns a
    /// line for the log.
    pub fn finish(&mut self) -> Result<String> {
        self.header[32..40].copy_from_slice(&self.rows.to_le_bytes());
        let mut measured = 0;
        for elite in &mut self.elites {
            if let Some(d) = self.measured.get(&elite.id) {
                elite.bytes[1] |= 8;
                for (i, v) in d.iter().enumerate() {
                    elite.bytes[20 + 4 * i..24 + 4 * i].copy_from_slice(&v.to_le_bytes());
                }
                measured += 1;
            }
        }
        self.file.flush()?;
        let file = self.file.get_mut();
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&self.header)?;
        for elite in &self.elites {
            file.write_all(&elite.bytes)?;
        }
        file.sync_all()?;
        Ok(format!(
            "Generation dump written: {} ({} creatures, {} elites, {} of them re-run) in {:.0} s",
            self.path.display(),
            self.rows,
            self.elites.len(),
            measured,
            self.started.elapsed().as_secs_f64()
        ))
    }
}
