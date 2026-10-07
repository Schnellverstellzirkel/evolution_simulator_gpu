//! The generation dump: `EVOLUTION_DUMP_GENERATION=<generation>[:<path>]`
//! (or `=<path>` for the next boundary), a developer diagnostic for the
//! steps ladder. From that generation's start, one
//! generation's worth of blocks is bred with the screen bar off, so every
//! trial runs in full, and the island elites are queued to run again in it
//! (they enter no archive). Each creature of those blocks writes a 64 B row
//! as its block is absorbed; the file ends with the header and a 32 B row
//! per elite of the archives at the start, patched in when the last block
//! is absorbed. Little endian throughout; `examples/dump_stats.rs` and
//! `examples/operator_yield.rs` read it.
//!
//! Header (64 B): magic `EVODUMP1`, format u32 (2), qd version u32,
//! generation u32, population u32, seed u64, rows u64, elites u32, the screen
//! bar the generation would have had f32, trial seconds f32, rate u16,
//! islands u8, a spare byte (format 1: arenas u8), the cell bins (contact,
//! cadence, height, feet) 4 x u8, arenas u16, 2 spare.
//!
//! Elite (32 B): arena low byte, flags u8 (1 reserve, 2 fine, 4 graduate, 8
//! re-run measured), cell u16, nodes u8, muscles u8, emitter u8, arena high
//! byte (format 1: spare; the arena was one byte), id u64, fitness f32,
//! distance at 2.5, 5 and 10 s from its re-run, 3 x f32 (NaN without one).
//! The arena is `u16::MAX` for the global archive (format 1: 255). There are
//! more than 256 arenas, so format 1 dumps of today's layout are wrong.
//!
//! Creature (64 B): slot u32, emitter u8, operator u8 (the low byte of the
//! index in `evolution::structural_operator_names`; 0xFFF for none, whose
//! high four bits are the high nibble of the `entered` byte), flags u8
//! (`MATE`...), entered u8 (`ISLAND`... in the low nibble), parent id u64 (`u64::MAX` for
//! none), parent cell u16, final cell u16, CMA emitter u16 (`u16::MAX` for
//! none), nodes u8, muscles u8, parent nodes u8, parent muscles u8, rhythm
//! period f16, standard fitness f32, archive fitness f32 (after a
//! confirmation), the seven `creature_kernel::RungTrace` words. A cell is
//! `((contact * 8 + cadence) * 6 + height) * 5 + feet`, `u16::MAX` for none.

use super::*;
use std::io::{Seek, SeekFrom};

/// The creature entered an island archive.
pub const ISLAND: u8 = 1;
/// The creature entered a nursery archive.
pub const NURSERY: u8 = 2;
/// The creature entered a morphology reserve.
pub const RESERVE: u8 = 4;
/// The creature entered the global archive.
pub const GLOBAL: u8 = 8;

/// The creature was produced by mating.
pub const MATE: u8 = 1;
/// Its CMA emitter was optimizing (`CmaEmitter::optimizing`).
pub const OPTIMIZER: u8 = 2;
/// Its fitness is its confirmation trial's (`EvaluationMetrics::fine`).
pub const FINE: u8 = 4;
/// An island elite run again in the dump's generation; it enters no archive.
pub const RERUN: u8 = 8;
/// The early screen stopped its trial.
pub const SCREENED: u8 = 16;
/// Its result enters no archive (`EvaluationMetrics::excluded`).
pub const EXCLUDED: u8 = 32;
/// The creature's parent was in a morphology reserve.
pub const PARENT_RESERVE: u8 = 64;

pub const MAGIC: &[u8; 8] = b"EVODUMP1";
/// Format 2 holds arenas in two bytes.
pub const FORMAT: u32 = 2;
pub const HEADER_BYTES: usize = 64;
pub const ELITE_BYTES: usize = 32;
pub const ROW_BYTES: usize = 64;
pub const BINS: [u8; 4] = [6, 8, 6, 5];

/// Whether a dump started in this process; there is one per run.
pub static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The generation (None: the next boundary) and the file.
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

/// A behavior cell as one number.
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
    optimizer: bool,
}
impl Parent {
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
    pub flags: u8,
    slot: u32,
    emitter: u8,
    operator: u16,
    parent: u64,
    parent_cell: u16,
    cma: u16,
    nodes: u8,
    muscles: u8,
    parent_nodes: u8,
    parent_muscles: u8,
    period: u16,
    id: u64,
}

/// A serialized elite for the dump file.
pub struct Elite {
    bytes: [u8; ELITE_BYTES],
    id: u64,
}
impl Elite {
    /// The row of `e` in `arena` (`u16::MAX` for the global archive).
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

/// Writes a generation dump to track which creatures entered archives.
pub struct Dump {
    path: std::path::PathBuf,
    file: BufWriter<File>,
    header: [u8; HEADER_BYTES],
    population: usize,
    /// Creatures bred into dump blocks, and blocks still to absorb.
    bred: usize,
    outstanding: usize,
    heads: Vec<Option<Vec<Head>>>,
    pending: Option<Vec<Head>>,
    elites: Vec<Elite>,
    reruns: std::collections::HashSet<u64>,
    /// Rung distances of the re-run elites, by id.
    measured: HashMap<u64, [f32; 3]>,
    rows: u64,
    started: std::time::Instant,
}

impl Dump {
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
    /// A block was bred for the dump at ring slot `first`.
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
    /// Writes rows for creatures in a block, recording their archive and fitness data.
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
    /// Writes the header and the elites with their re-run distances, and
    /// closes the file. Returns a line for the log.
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
