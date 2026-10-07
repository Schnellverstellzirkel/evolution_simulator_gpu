//! Reads a generation dump (`storage::dump`, `EVOLUTION_DUMP_GENERATION`) and
//! fits a ladder of early stopping rules on its rows, to tell where each row
//! would have stopped. The examples `dump_stats` and `operator_yield` include
//! this file with `#[path]`. Each uses only part of it, so unused items are
//! allowed.
#![allow(dead_code)]
use anyhow::{Context, Result, ensure};
use evolution_simulator::creature_kernel::{RungTrace, f16_to_f32};

// The flag bits below repeat the ones `storage::dump` writes.
/// `Row::entered` bit: the creature entered an island archive.
pub const ISLAND: u8 = 1;
/// `Row::entered` bit: the creature entered a nursery archive.
pub const NURSERY: u8 = 2;
/// `Row::entered` bit: the creature entered a morphology reserve.
pub const RESERVE: u8 = 4;
/// `Row::entered` bit: the creature entered the global archive.
pub const GLOBAL: u8 = 8;
/// `Row::flags` bit: the creature was produced by mating.
pub const MATE: u8 = 1;
/// `Row::flags` bit: its CMA emitter was optimizing.
pub const OPTIMIZER: u8 = 2;
/// `Row::flags` bit: its `Row::score` is its confirmation trial's.
pub const FINE: u8 = 4;
/// `Row::flags` bit: an island elite run again in the dump generation. It
/// enters no archive.
pub const RERUN: u8 = 8;
/// `Row::flags` bit: the early screen stopped its trial.
pub const SCREENED: u8 = 16;
/// `Row::flags` bit: its result enters no archive. Three cases set it. The
/// result was measured in a world that has since changed, the row is an elite
/// re-run, or its confirmation trial was stopped by the screen.
pub const EXCLUDED: u8 = 32;
/// `Row::flags` bit: the creature's parent was in a morphology reserve.
pub const PARENT_RESERVE: u8 = 64;
/// Emitter names by `Row::emitter`, in the order of `qd::Emitter`.
pub const EMITTERS: [&str; 4] = ["cma", "structural", "novelty", "restart"];
/// The `Row::emitter` of the restart emitter, which breeds the immigrants (new
/// random bodies).
pub const RESTART: u8 = 3;
/// Steps of the ladder's four rungs at 60 Hz: 1, 2.5, 5 and 10 s. The game has
/// the first two (`rungs::RUNG_STEPS`), and its 5 s screen is the third. The
/// fourth is a candidate rule that the game does not have.
pub const RUNG_STEPS: [u32; 4] = [60, 150, 300, 600];

/// The 64 byte header of a dump file.
pub struct Header {
    /// `qd::VERSION` of the build that wrote the dump.
    pub qd_version: u32,
    /// The generation at whose start the dump began.
    pub generation: u32,
    /// Creatures per generation.
    pub population: u32,
    /// The experiment's seed.
    pub seed: u64,
    /// Number of rows in the file.
    pub rows: u64,
    /// Number of elites in the file.
    pub elites: u32,
    /// The 5 s screen bar the generation would have had (m). The dump
    /// generation itself runs with the bar off.
    pub bar: f32,
    /// Trial length in seconds.
    pub duration: f32,
    /// Physics steps per second of the standard trial.
    pub rate: u16,
    /// Number of islands. Archives from this index on are nurseries.
    pub islands: u8,
    /// Number of archives that creatures breed for: the islands, then their
    /// nurseries of new random bodies, then their nurseries of reshaped bodies.
    pub arenas: u16,
    /// Bins of the four cell axes: contact, cadence, height, feet.
    pub bins: [u8; 4],
}

/// An elite of an archive when the dump generation began.
#[derive(Clone, Copy)]
pub struct Elite {
    /// The archive it sits in, numbered as `Row::arena` numbers them.
    /// `u16::MAX` for the global archive.
    pub arena: u16,
    /// Bits: 1 it sits in a morphology reserve, 2 its score is a confirmation
    /// trial's, 4 it or its ancestor grew up in a nursery, 8 it was run again
    /// in the dump generation, so `d` is set.
    pub flags: u8,
    /// Its behavior cell (see `axes`), `u16::MAX` for none.
    pub cell: u16,
    /// Its node count.
    pub nodes: u8,
    /// Its muscle count.
    pub muscles: u8,
    /// The emitter that bred it, an index into `EMITTERS`.
    pub emitter: u8,
    /// Its creature id.
    pub id: u64,
    /// Its fitness in the archive: the distance in m.
    pub fitness: f32,
    /// Distances at 2.5, 5 and 10 s from its re-run. NaN without one.
    pub d: [f32; 3],
}
impl Elite {
    /// Whether the elite sits in a morphology reserve (flag bit 1) instead of
    /// a cell of behavior.
    pub fn reserve(&self) -> bool {
        self.flags & 1 != 0
    }
}

/// `Row::operator` when no structural operator changed the child.
pub const NO_OPERATOR: u16 = u16::MAX;

/// One creature of the dump generation: how it was bred, what it entered and
/// how its trial went.
#[derive(Clone, Copy)]
pub struct Row {
    /// Its slot in the generation's population.
    pub slot: u32,
    /// The emitter that bred it, an index into `EMITTERS`.
    pub emitter: u8,
    /// The index in `evolution::structural_operator_names` of the operator
    /// that changed the child, `NO_OPERATOR` for none.
    pub operator: u16,
    /// Bits `MATE`, `OPTIMIZER`, `FINE`, `RERUN`, `SCREENED`, `EXCLUDED` and
    /// `PARENT_RESERVE`.
    pub flags: u8,
    /// Bits `ISLAND`, `NURSERY`, `RESERVE` and `GLOBAL` for the archives it
    /// entered. 0 for none.
    pub entered: u8,
    /// Its parent's creature id, `u64::MAX` for none.
    pub parent: u64,
    /// The cell of its parent (see `axes`), `u16::MAX` for none.
    pub parent_cell: u16,
    /// The cell its own behavior fell in (see `axes`), `u16::MAX` for none.
    pub cell: u16,
    /// The index of the CMA emitter that sampled it, `u16::MAX` for none.
    pub cma: u16,
    /// Its node count.
    pub nodes: u8,
    /// Its muscle count.
    pub muscles: u8,
    /// Its parent's node count, 0 when the parent is unknown.
    pub parent_nodes: u8,
    /// Its parent's muscle count, 0 when the parent is unknown.
    pub parent_muscles: u8,
    /// The rhythm period in seconds, read off its first muscle. 0 without
    /// muscles. The dump stores it in half precision.
    pub period: f32,
    /// The standard trial's fitness: the distance in m.
    pub fitness: f32,
    /// The fitness the archives saw. It is the confirmation trial's when that
    /// trial lowered it.
    pub score: f32,
    /// The standard trial's rung trace: the distances at 1, 2.5, 5 and 10 s,
    /// the early features and the end code.
    pub trace: RungTrace,
}

impl Row {
    /// The archive the row competes in, among the dump's arenas.
    pub fn arena(&self, h: &Header) -> usize {
        evolution_simulator::qd::arena_of_slot(self.slot as usize, h.arenas as usize)
    }
    /// Whether the row competes in a nursery, of either kind.
    pub fn nursery(&self, h: &Header) -> bool {
        self.arena(h) >= h.islands as usize
    }
    /// Whether the row is a bred child. It is not an elite's re-run, it is not
    /// `EXCLUDED`, and its trial left a rung trace.
    pub fn child(&self) -> bool {
        self.flags & (RERUN | EXCLUDED) == 0 && self.trace.steps() > 0
    }
    /// Steps its trial ran. 0 when it left no trace.
    pub fn steps(&self) -> u32 {
        self.trace.steps()
    }
    /// Its distance in m at rung `rung` (0 to 3: 1, 2.5, 5 and 10 s).
    pub fn d(&self, rung: usize) -> f32 {
        self.trace.distance(rung)
    }
    /// Whether the row stands in for an audit creature, which no rung stops.
    /// It picks 1 slot in 128 by a hash of the slot. The game's
    /// `rungs::is_audit` also hashes the seed and the breeding round, which a
    /// row does not carry, so it picks other creatures at the same rate.
    pub fn audit(&self) -> bool {
        splitmix(self.slot as u64 ^ 0xa0d1_7000) & 127 == 0
    }
}

/// One step of SplitMix64 from state `x`, used as a hash of `x`.
pub fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn f32_at(b: &[u8], at: usize) -> f32 {
    f32::from_bits(u32_at(b, at))
}

/// A whole generation dump.
pub struct DumpFile {
    /// The dump's header.
    pub header: Header,
    /// The elites of every archive when the dump generation began.
    pub elites: Vec<Elite>,
    /// One row per creature of the dump generation.
    pub rows: Vec<Row>,
}

/// Reads the dump at `path`. It takes formats 1 and 2. It fails on a file of
/// another kind, on a newer format and on a size that the header does not
/// explain, which is an unfinished dump.
pub fn read(path: &str) -> Result<DumpFile> {
    // The offsets and sizes are those of `storage::dump`: a 64 byte header,
    // then 32 byte elites, then 64 byte rows.
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    ensure!(
        bytes.len() >= 64 && &bytes[0..8] == b"EVODUMP1",
        "{path} is not a generation dump"
    );
    let b = &bytes[..64];
    // Format 1 kept arenas in one byte, which more than 256 arenas overflow.
    let format = u32_at(b, 8);
    ensure!(
        format <= 2,
        "{path}: dump format {format} is newer than this reader"
    );
    let header = Header {
        qd_version: u32_at(b, 12),
        generation: u32_at(b, 16),
        population: u32_at(b, 20),
        seed: u64_at(b, 24),
        rows: u64_at(b, 32),
        elites: u32_at(b, 40),
        bar: f32_at(b, 44),
        duration: f32_at(b, 48),
        rate: u16_at(b, 52),
        islands: b[54],
        arenas: if format >= 2 {
            u16_at(b, 60)
        } else {
            u16::from(b[55])
        },
        bins: [b[56], b[57], b[58], b[59]],
    };
    let elite_end = 64 + header.elites as usize * 32;
    ensure!(
        bytes.len() == elite_end + header.rows as usize * 64,
        "{path}: {} bytes, the header says {} elites and {} rows (an unfinished dump?)",
        bytes.len(),
        header.elites,
        header.rows
    );
    let elites = bytes[64..elite_end]
        .as_chunks::<32>()
        .0
        .iter()
        .map(|b| Elite {
            // Format 1 kept the arena in byte 0 (255 for the global archive).
            // Format 2 splits it between byte 0 (low) and byte 7 (high).
            arena: match (format, b[0]) {
                (1, 255) => u16::MAX,
                (1, arena) => u16::from(arena),
                _ => u16::from_le_bytes([b[0], b[7]]),
            },
            flags: b[1],
            cell: u16_at(b, 2),
            nodes: b[4],
            muscles: b[5],
            emitter: b[6],
            id: u64_at(b, 8),
            fitness: f32_at(b, 16),
            d: [f32_at(b, 20), f32_at(b, 24), f32_at(b, 28)],
        })
        .collect();
    let rows = bytes[elite_end..]
        .as_chunks::<64>()
        .0
        .iter()
        .map(|b| Row {
            slot: u32_at(b, 0),
            emitter: b[4],
            // The operator index has 12 bits: byte 5 and the high nibble of
            // byte 7 (0xFFF for none). `entered` is the low nibble of byte 7.
            operator: match b[5] as u16 | (((b[7] >> 4) as u16) << 8) {
                0xFFF => NO_OPERATOR,
                index => index,
            },
            flags: b[6],
            entered: b[7] & 15,
            parent: u64_at(b, 8),
            parent_cell: u16_at(b, 16),
            cell: u16_at(b, 18),
            cma: u16_at(b, 20),
            nodes: b[22],
            muscles: b[23],
            parent_nodes: b[24],
            parent_muscles: b[25],
            period: f16_to_f32(u16_at(b, 26)),
            fitness: f32_at(b, 28),
            score: f32_at(b, 32),
            // The seven trace words fill bytes 36 to 63.
            trace: RungTrace {
                words: std::array::from_fn(|w| u32_at(b, 36 + 4 * w)),
                fitness: f32_at(b, 28),
            },
        })
        .collect();
    Ok(DumpFile {
        header,
        elites,
        rows,
    })
}

/// The coordinates of `cell` on the four axes: contact, cadence, height, feet.
/// A cell is `((contact * bins[1] + cadence) * bins[2] + height) * bins[3] +
/// feet`. `None` for `u16::MAX`, the cell of a creature with none.
pub fn axes(cell: u16, bins: [u8; 4]) -> Option<[i32; 4]> {
    if cell == u16::MAX {
        return None;
    }
    let c = cell as i32;
    let b = bins.map(i32::from);
    Some([
        c / (b[1] * b[2] * b[3]),
        (c / (b[2] * b[3])) % b[1],
        (c / b[3]) % b[2],
        c % b[3],
    ])
}
/// The cell at coordinates `a` (contact, cadence, height, feet). It undoes
/// `axes`. `None` when a coordinate is outside its axis.
pub fn cell_of(a: [i32; 4], bins: [u8; 4]) -> Option<u16> {
    let b = bins.map(i32::from);
    if (0..4).any(|i| a[i] < 0 || a[i] >= b[i]) {
        return None;
    }
    Some((((a[0] * b[1] + a[1]) * b[2] + a[2]) * b[3] + a[3]) as u16)
}
/// Number of cells for `bins`.
pub fn cells(bins: [u8; 4]) -> usize {
    bins.iter().map(|&b| b as usize).product()
}
/// The cadence band of `cell`: its coordinate on the cadence axis.
pub fn cadence_band(cell: u16, bins: [u8; 4]) -> Option<usize> {
    axes(cell, bins).map(|a| a[1] as usize)
}

/// Per arena and cell, the lowest `value(elite)` over the cell and its radius-1
/// neighbours (81 cells on the four axes). Reserve elites and values that are
/// not finite do not count. NaN when no cell of the neighbourhood has a value.
pub fn neighbourhood_min(d: &DumpFile, value: impl Fn(&Elite) -> f32) -> Vec<Vec<f32>> {
    let bins = d.header.bins;
    let n = cells(bins);
    let mut own = vec![vec![f32::NAN; n]; d.header.arenas as usize];
    for e in &d.elites {
        if e.reserve() || (e.arena as usize) >= own.len() || e.cell as usize >= n {
            continue;
        }
        let v = value(e);
        if v.is_finite() {
            let slot = &mut own[e.arena as usize][e.cell as usize];
            *slot = if slot.is_nan() { v } else { slot.min(v) };
        }
    }
    own.iter()
        .map(|table| {
            (0..n)
                .map(|c| {
                    let a = axes(c as u16, bins).unwrap();
                    let mut best = f32::NAN;
                    for k in 0..81 {
                        let o = [
                            k % 3 - 1,
                            (k / 3) % 3 - 1,
                            (k / 9) % 3 - 1,
                            (k / 27) % 3 - 1,
                        ];
                        if let Some(nc) =
                            cell_of([a[0] + o[0], a[1] + o[1], a[2] + o[2], a[3] + o[3]], bins)
                        {
                            let v = table[nc as usize];
                            if v.is_finite() {
                                best = if best.is_nan() { v } else { best.min(v) };
                            }
                        }
                    }
                    best
                })
                .collect()
        })
        .collect()
}

/// R4's table: per arena and cell, the lowest final distance of the elites over
/// the cell and its neighbours on contact, cadence and height (plus or minus
/// one) and on feet (this bin and the next), 54 cells. Minus infinity when any
/// of those cells that exist is empty. Reserve elites do not count.
pub fn r4_table(d: &DumpFile) -> Vec<Vec<f32>> {
    let bins = d.header.bins;
    let n = cells(bins);
    let mut own = vec![vec![f32::NAN; n]; d.header.arenas as usize];
    for e in &d.elites {
        if e.reserve() || (e.arena as usize) >= own.len() || e.cell as usize >= n {
            continue;
        }
        let slot = &mut own[e.arena as usize][e.cell as usize];
        *slot = if slot.is_nan() {
            e.fitness
        } else {
            slot.min(e.fitness)
        };
    }
    own.iter()
        .map(|table| {
            (0..n)
                .map(|c| {
                    let a = axes(c as u16, bins).unwrap();
                    let mut best = f32::INFINITY;
                    for k in 0..54 {
                        let o = [k % 3 - 1, (k / 3) % 3 - 1, (k / 9) % 3 - 1, k / 27];
                        if let Some(nc) =
                            cell_of([a[0] + o[0], a[1] + o[1], a[2] + o[2], a[3] + o[3]], bins)
                        {
                            let v = table[nc as usize];
                            if v.is_nan() {
                                return f32::NEG_INFINITY;
                            }
                            best = best.min(v);
                        }
                    }
                    best
                })
                .collect()
        })
        .collect()
}

/// Fisher's linear discriminant: weights that give class `a` a higher mean dot
/// product than class `b`. The sums are in f64, and a small ridge keeps the
/// pooled covariance solvable.
pub fn fisher(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<f64> {
    let k = a.first().or(b.first()).map_or(0, Vec::len);
    let mean = |set: &[Vec<f64>]| {
        let mut m = vec![0.0; k];
        for x in set {
            for i in 0..k {
                m[i] += x[i];
            }
        }
        m.iter_mut().for_each(|v| *v /= set.len().max(1) as f64);
        m
    };
    let (ma, mb) = (mean(a), mean(b));
    let mut s = vec![vec![0.0; k]; k];
    for (set, m) in [(a, &ma), (b, &mb)] {
        for x in set {
            for i in 0..k {
                for j in 0..k {
                    s[i][j] += (x[i] - m[i]) * (x[j] - m[j]);
                }
            }
        }
    }
    let n = (a.len() + b.len()).saturating_sub(2).max(1) as f64;
    // The ridge on the diagonal is 1e-9 times the mean variance, which is
    // floored at 1e-12.
    let mean_variance: f64 = (0..k).map(|i| s[i][i] / n).sum::<f64>() / k.max(1) as f64;
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for j in 0..k {
            s[i][j] /= n;
        }
        s[i][i] += 1e-9 * mean_variance.max(1e-12);
    }
    let rhs: Vec<f64> = (0..k).map(|i| ma[i] - mb[i]).collect();
    solve(s, rhs)
}

/// Solves `m x = v` by Gauss-Jordan elimination with row pivoting. A column
/// whose pivot is near zero gives 0.
fn solve(mut m: Vec<Vec<f64>>, mut v: Vec<f64>) -> Vec<f64> {
    let k = v.len();
    for c in 0..k {
        let p = (c..k)
            .max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))
            .unwrap();
        m.swap(c, p);
        v.swap(c, p);
        let d = m[c][c];
        if d.abs() < 1e-300 {
            continue;
        }
        for r in 0..k {
            if r != c {
                let f = m[r][c] / d;
                #[allow(clippy::needless_range_loop)]
                for j in c..k {
                    m[r][j] -= f * m[c][j];
                }
                v[r] -= f * v[c];
            }
        }
    }
    (0..k)
        .map(|i| {
            if m[i][i].abs() < 1e-300 {
                0.0
            } else {
                v[i] / m[i][i]
            }
        })
        .collect()
}

/// The dot product of `w` and `x`.
pub fn dot(w: &[f64], x: &[f64]) -> f64 {
    w.iter().zip(x).map(|(a, b)| a * b).sum()
}

/// The stop threshold of a rung: a creature whose score is below it stops. It
/// leaves `floor(tol * n)` of the `n` protected scores below it. With no
/// protected scores it is minus infinity, so nothing stops.
pub fn threshold(mut protected: Vec<f64>, tol: f64) -> f64 {
    if protected.is_empty() {
        return f64::NEG_INFINITY;
    }
    protected.sort_by(f64::total_cmp);
    let k = (tol * protected.len() as f64).floor() as usize;
    protected[k.min(protected.len() - 1)]
}

/// The `q` quantile (0 to 1) of `values`. It sorts `values` and takes the
/// nearest sorted value, with no interpolation. NaN when `values` is empty.
pub fn quantile(values: &mut [f32], q: f64) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

/// The Spearman rank correlation of the pairs. Tied values share the mean of
/// their ranks. NaN for fewer than 3 pairs.
pub fn spearman(pairs: &[(f32, f32)]) -> f64 {
    let rank = |v: Vec<f32>| {
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
        let mut r = vec![0.0f64; v.len()];
        let mut i = 0;
        while i < order.len() {
            let mut j = i;
            while j + 1 < order.len() && v[order[j + 1]] == v[order[i]] {
                j += 1;
            }
            let mean = (i + j) as f64 / 2.0;
            for &o in &order[i..=j] {
                r[o] = mean;
            }
            i = j + 1;
        }
        r
    };
    let a = rank(pairs.iter().map(|p| p.0).collect());
    let b = rank(pairs.iter().map(|p| p.1).collect());
    let n = a.len() as f64;
    if n < 3.0 {
        return f64::NAN;
    }
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma).powi(2);
        sbb += (b[i] - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

/// The size class of a body with `nodes` nodes: 0 for 3 to 8, 1 for 9 to 16 and
/// 2 for 17 to 32. These edges belong to this tool, not to `qd::Classes`.
pub fn size_class(nodes: u8) -> usize {
    match nodes {
        0..=8 => 0,
        9..=16 => 1,
        _ => 2,
    }
}
/// The names of the size classes, indexed by `size_class`.
pub const SIZE_CLASSES: [&str; 3] = ["3-8 nodes", "9-16 nodes", "17-32 nodes"];

/// A ladder of four early stopping rules, fitted on rows of a dump (`fit`) and
/// applied to any row (`apply`). R1 at 1 s and R2 at 2.5 s are linear rules on
/// early features, as in `rungs`. R3 at 5 s is the screen against a bar. R4 at
/// 10 s is a candidate rule that the game does not have.
pub struct Ladder {
    /// The share of the protected creatures that a rung may stop.
    pub tol: f64,
    /// Whether R3 holds a row to a bar from its parent's neighbourhood
    /// (`bar300`) instead of the generation's one 5 s bar, `bar`.
    pub cell_bars: bool,
    /// Whether R4 is on.
    pub r4: bool,
    /// Whether R2 has a seventh feature, `bar150`: the 2.5 s bar of the row's
    /// parent neighbourhood. It is on unless `RUNG_NO_BAR_FEATURE` is set.
    pub bar_feature: bool,
    /// R1's weights over `features1`.
    pub w1: Vec<f64>,
    /// R1's bias. R1 stops a row whose weighted score is below it.
    pub b1: f64,
    /// R2's weights over `features2`.
    pub w2: Vec<f64>,
    /// R2's bias. R2 stops a row whose weighted score is below it.
    pub b2: f64,
    /// The factor of a per-cell R3 bar: a row must reach `f3` times the lowest
    /// 5 s distance around its parent's cell. It is 0.7 until fitted.
    pub f3: f32,
    /// Per arena, the 80th percentile of the 2.5 s distances of the fitted
    /// rows. It is the 2.5 s bar of a row with no neighbourhood bar.
    pub arena150: Vec<f32>,
    /// The same at 5 s. It is the per-cell R3 bar of a row with no
    /// neighbourhood bar.
    pub arena300: Vec<f32>,
    /// Per arena and cell, the lowest 2.5 s distance of the elites around the
    /// cell (`neighbourhood_min`).
    pub nmin150: Vec<Vec<f32>>,
    /// The same at 5 s.
    pub nmin300: Vec<Vec<f32>>,
    /// Per arena and cell, the lowest final distance of the elites around the
    /// cell (`r4_table`).
    pub nmin_final: Vec<Vec<f32>>,
    /// R4's ratio: the `1 - tol` quantile of final distance over the 10 s
    /// distance. It is infinity until fitted.
    pub r4_ratio: f32,
    /// R4's floor: the median distance at 10 s of each group of rows, keyed by
    /// `emitter_key`.
    pub floor: std::collections::HashMap<(u8, u16, u16), f32>,
    /// The generation's 5 s screen bar from the header (m).
    pub bar: f32,
    /// The header's number of islands.
    pub islands: u8,
    /// The header's number of arenas.
    pub arenas: u16,
}

/// What the ladder did to one row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fate {
    /// Stopped at rung `k`, where 0 to 3 stand for R1 to R4.
    Stopped(usize),
    /// Ran to its natural end: a fall or the last step.
    Ran,
}

impl Ladder {
    /// The group that a row's R4 floor belongs to: its CMA emitter when it has
    /// one, else its emitter kind and its arena.
    fn emitter_key(r: &Row, arena: usize) -> (u8, u16, u16) {
        if r.cma != u16::MAX {
            (0, r.cma, 0)
        } else {
            (r.emitter + 1, 0, arena as u16)
        }
    }
    /// R1's features at 1 s: the distance, the speed over the last half second,
    /// the share of nodes that touched the ground, the head shake, the mean
    /// muscle energy store and the rhythm period.
    pub fn features1(r: &Row) -> [f64; 6] {
        [
            r.d(0) as f64,
            r.trace.speed(0) as f64,
            r.trace.touched(0) as f64,
            r.trace.head_shake(0) as f64,
            r.trace.energy(0) as f64,
            r.period as f64,
        ]
    }
    /// R2's features at 2.5 s: the same six as `features1`, then `bar150` when
    /// `bar_feature` is on.
    pub fn features2(&self, r: &Row, h: &Header) -> Vec<f64> {
        let mut f = vec![
            r.d(1) as f64,
            r.trace.speed(1) as f64,
            r.trace.touched(1) as f64,
            r.trace.head_shake(1) as f64,
            r.trace.energy(1) as f64,
            r.period as f64,
        ];
        if self.bar_feature {
            f.push(self.bar150(r, h) as f64);
        }
        f
    }
    /// The lowest distance in `table` (from `neighbourhood_min`) around the
    /// cell of the row's parent, in the row's arena. NaN when the row has no
    /// usable neighbourhood: a nursery row, an immigrant, a row with no parent
    /// cell, or a neighbourhood whose lowest distance is not above 0.
    fn parent_min(table: &[Vec<f32>], r: &Row, h: &Header) -> f32 {
        let arena = r.arena(h);
        if r.nursery(h) || r.emitter == RESTART || r.parent_cell == u16::MAX {
            return f32::NAN;
        }
        // A neighbourhood whose slowest elite had not moved forward at that
        // time gives no bar. The arena's bar stands in.
        table
            .get(arena)
            .and_then(|t| t.get(r.parent_cell as usize))
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or(f32::NAN)
    }
    /// The 2.5 s bar of a row: the lowest 2.5 s distance around its parent's
    /// cell, or its arena's bar when there is none.
    pub fn bar150(&self, r: &Row, h: &Header) -> f32 {
        let v = Self::parent_min(&self.nmin150, r, h);
        if v.is_finite() {
            v
        } else {
            self.arena150[r.arena(h)]
        }
    }
    /// The R3 bar of a row at 5 s. It is `bar` unless `cell_bars` is on. Then
    /// it is `f3` times the lowest 5 s distance around the row's parent's
    /// cell, or the arena's bar when there is none.
    pub fn bar300(&self, r: &Row, h: &Header) -> f32 {
        if !self.cell_bars {
            return self.bar;
        }
        let v = Self::parent_min(&self.nmin300, r, h);
        if v.is_finite() {
            v * self.f3
        } else {
            self.arena300[r.arena(h)]
        }
    }
    /// Whether the row passes R3: its trial ran past 5 s and its distance
    /// there reaches its bar.
    pub fn pass3(&self, r: &Row, h: &Header) -> bool {
        r.steps() > RUNG_STEPS[2] && r.d(2) >= self.bar300(r, h)
    }

    /// Fits a ladder on `rows` of the dump `d`. `tol` is the share of the
    /// protected creatures that a rung may stop. `cell_bars` gives R3 a bar per
    /// cell, and `r4` turns on the 10 s rung.
    ///
    /// R1 and R2 come from Fisher's discriminant between the rows that pass R3
    /// (class A) and the rest (class B). Each bias leaves a share `tol` of
    /// class A below it. Nursery rows and immigrants stay out of that fit,
    /// because the game exempts them from R1 and R2. With `r4`, its ratio and
    /// floor come from the rows that ran past 10 s.
    pub fn fit(rows: &[&Row], d: &DumpFile, tol: f64, cell_bars: bool, r4: bool) -> Ladder {
        let h = &d.header;
        let arenas = h.arenas as usize;
        let mut per150: Vec<Vec<f32>> = vec![Vec::new(); arenas];
        let mut per300: Vec<Vec<f32>> = vec![Vec::new(); arenas];
        for r in rows {
            if r.steps() > RUNG_STEPS[1] {
                per150[r.arena(h)].push(r.d(1));
            }
            if r.steps() > RUNG_STEPS[2] {
                per300[r.arena(h)].push(r.d(2));
            }
        }
        // Arena bars: the 80th percentile of the arena's own distances.
        let arena150 = per150.iter_mut().map(|v| quantile(v, 0.8)).collect();
        let arena300 = per300.iter_mut().map(|v| quantile(v, 0.8)).collect();
        let mut ladder = Ladder {
            tol,
            cell_bars,
            r4,
            bar_feature: std::env::var_os("RUNG_NO_BAR_FEATURE").is_none(),
            w1: vec![],
            b1: f64::NEG_INFINITY,
            w2: vec![],
            b2: f64::NEG_INFINITY,
            f3: 0.7,
            arena150,
            arena300,
            nmin150: neighbourhood_min(d, |e| e.d[0]),
            nmin300: neighbourhood_min(d, |e| e.d[1]),
            nmin_final: r4_table(d),
            r4_ratio: f32::INFINITY,
            floor: Default::default(),
            bar: h.bar,
            islands: h.islands,
            arenas: h.arenas,
        };
        if cell_bars {
            // f3: the tol lower quantile of d300 / nmin300 over entrants.
            let mut ratios: Vec<f32> = rows
                .iter()
                .filter(|r| r.entered != 0 && r.steps() > RUNG_STEPS[2])
                .filter_map(|r| {
                    let m = Self::parent_min(&ladder.nmin300, r, h);
                    (m.is_finite() && m > 0.0).then(|| r.d(2) / m)
                })
                .collect();
            ladder.f3 = quantile(&mut ratios, tol).clamp(0.0, 1.0);
            if !ladder.f3.is_finite() {
                ladder.f3 = 0.7;
            }
        }
        // R1 and R2: class A passes R3, class B does not. Exempt rows are
        // left out of the fit.
        let subject = |r: &Row| !r.nursery(h) && r.emitter != RESTART;
        let (mut a1, mut b1, mut a2, mut b2) = (vec![], vec![], vec![], vec![]);
        for r in rows.iter().filter(|r| subject(r)) {
            let pass = ladder.pass3(r, h);
            if r.steps() > RUNG_STEPS[0] {
                let f = Self::features1(r).to_vec();
                if f.iter().all(|v| v.is_finite()) {
                    if pass { a1.push(f) } else { b1.push(f) }
                }
            }
            if r.steps() > RUNG_STEPS[1] {
                let f = ladder.features2(r, h);
                if f.iter().all(|v| v.is_finite()) {
                    if pass { a2.push(f) } else { b2.push(f) }
                }
            }
        }
        ladder.w1 = fisher(&a1, &b1);
        ladder.b1 = threshold(a1.iter().map(|f| dot(&ladder.w1, f)).collect(), tol);
        ladder.w2 = fisher(&a2, &b2);
        ladder.b2 = threshold(a2.iter().map(|f| dot(&ladder.w2, f)).collect(), tol);
        if r4 {
            // r4: the (1 - tol) quantile of final / d600 among the rows R4
            // applies to: alive at 10 s, past R3, moving forward.
            let mut ratios: Vec<f32> = rows
                .iter()
                .filter(|r| {
                    r.steps() > RUNG_STEPS[3] && r.d(3) > 0.05 && r.d(2) >= ladder.bar300(r, h)
                })
                .map(|r| r.fitness / r.d(3))
                .filter(|v| v.is_finite())
                .collect();
            ladder.r4_ratio = quantile(&mut ratios, 1.0 - tol);
            let mut groups: std::collections::HashMap<(u8, u16, u16), Vec<f32>> =
                Default::default();
            for r in rows.iter().filter(|r| r.steps() > RUNG_STEPS[3]) {
                groups
                    .entry(Self::emitter_key(r, r.arena(h)))
                    .or_default()
                    .push(r.d(3));
            }
            ladder.floor = groups
                .into_iter()
                .map(|(k, mut v)| (k, quantile(&mut v, 0.5)))
                .collect();
        }
        ladder
    }

    /// The fate of row `r`: the first rung that stops it, else `Fate::Ran`. An
    /// audit row is never stopped. Nursery rows and immigrants skip R1 and R2.
    /// R4 skips the rows of an optimizer.
    pub fn apply(&self, r: &Row, h: &Header) -> Fate {
        if r.audit() {
            return Fate::Ran;
        }
        let n = r.steps();
        let exempt12 = r.nursery(h) || r.emitter == RESTART;
        if !exempt12 && n > RUNG_STEPS[0] {
            let f = Self::features1(r);
            if f.iter().all(|v| v.is_finite()) && dot(&self.w1, &f) < self.b1 {
                return Fate::Stopped(0);
            }
        }
        if !exempt12 && n > RUNG_STEPS[1] {
            let f = self.features2(r, h);
            if f.iter().all(|v| v.is_finite()) && dot(&self.w2, &f) < self.b2 {
                return Fate::Stopped(1);
            }
        }
        if n > RUNG_STEPS[2] && r.d(2) < self.bar300(r, h) {
            return Fate::Stopped(2);
        }
        if self.r4 && n > RUNG_STEPS[3] && r.flags & OPTIMIZER == 0 {
            let arena = r.arena(h);
            let nmin = self
                .nmin_final
                .get(arena)
                .and_then(|t| t.get(r.cell as usize))
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            let floor = self
                .floor
                .get(&Self::emitter_key(r, arena))
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            if nmin.is_finite() && r.d(3) * self.r4_ratio < nmin && r.d(3) < floor {
                return Fate::Stopped(3);
            }
        }
        Fate::Ran
    }
    /// The steps the trial of row `r` runs under `fate`: up to the rung that
    /// stopped it, or all of its own steps.
    pub fn steps(fate: Fate, r: &Row) -> u32 {
        match fate {
            Fate::Stopped(k) => RUNG_STEPS[k],
            Fate::Ran => r.steps(),
        }
    }
}

/// The steps the trial of row `r` runs under the 5 s screen alone, at the
/// generation's bar `h.bar`. It stops at step 300 when its distance there is
/// below the bar. The early rungs R1 and R2 do not apply.
pub fn today_steps(r: &Row, h: &Header) -> u32 {
    if r.steps() > RUNG_STEPS[2] && r.d(2) < h.bar {
        RUNG_STEPS[2]
    } else {
        r.steps()
    }
}

/// Whether row `r` is in the half of the rows to fit a ladder on. The split is
/// a hash of the slot, so it is the same on every run.
pub fn fit_half(r: &Row) -> bool {
    splitmix(r.slot as u64 ^ 0x05ee_df17) & 1 == 0
}
