//! Reader of the generation dump (`storage::dump`, EVOLUTION_DUMP_GENERATION)
//! and a rung ladder applied to its rows, shared by
//! `dump_stats`.
#![allow(dead_code)]
use anyhow::{Context, Result, ensure};
use evolution_simulator::creature_kernel::{RungTrace, f16_to_f32};

pub const ISLAND: u8 = 1;
pub const NURSERY: u8 = 2;
pub const RESERVE: u8 = 4;
pub const GLOBAL: u8 = 8;
pub const MATE: u8 = 1;
pub const OPTIMIZER: u8 = 2;
pub const FINE: u8 = 4;
pub const RERUN: u8 = 8;
pub const SCREENED: u8 = 16;
pub const EXCLUDED: u8 = 32;
pub const PARENT_RESERVE: u8 = 64;
/// Emitter indices (`qd::Emitter`).
pub const EMITTERS: [&str; 4] = ["cma", "structural", "novelty", "restart"];
pub const RESTART: u8 = 3;
/// Steps of the four rungs at 60 Hz: 1, 2.5, 5 and 10 s.
pub const RUNG_STEPS: [u32; 4] = [60, 150, 300, 600];

pub struct Header {
    pub qd_version: u32,
    pub generation: u32,
    pub population: u32,
    pub seed: u64,
    pub rows: u64,
    pub elites: u32,
    pub bar: f32,
    pub duration: f32,
    pub rate: u16,
    pub islands: u8,
    pub arenas: u8,
    pub bins: [u8; 4],
}

#[derive(Clone, Copy)]
pub struct Elite {
    pub arena: u8,
    pub flags: u8,
    pub cell: u16,
    pub nodes: u8,
    pub muscles: u8,
    pub emitter: u8,
    pub id: u64,
    pub fitness: f32,
    /// Distance at 2.5, 5 and 10 s from its re-run (NaN without one).
    pub d: [f32; 3],
}
impl Elite {
    pub fn reserve(&self) -> bool {
        self.flags & 1 != 0
    }
}

/// `Row::operator` when no structural operator changed the child.
pub const NO_OPERATOR: u16 = u16::MAX;

#[derive(Clone, Copy)]
pub struct Row {
    pub slot: u32,
    pub emitter: u8,
    /// The index in `structural_operator_names`, `NO_OPERATOR` for none.
    pub operator: u16,
    pub flags: u8,
    pub entered: u8,
    pub parent: u64,
    pub parent_cell: u16,
    pub cell: u16,
    pub cma: u16,
    pub nodes: u8,
    pub muscles: u8,
    pub parent_nodes: u8,
    pub parent_muscles: u8,
    pub period: f32,
    pub fitness: f32,
    pub score: f32,
    pub trace: RungTrace,
}

impl Row {
    pub fn arena(&self, h: &Header) -> usize {
        evolution_simulator::qd::arena_of_slot(self.slot as usize, h.arenas as usize)
    }
    pub fn nursery(&self, h: &Header) -> bool {
        self.arena(h) >= h.islands as usize
    }
    /// A bred child (not an elite's re-run, not from a stale world).
    pub fn child(&self) -> bool {
        self.flags & (RERUN | EXCLUDED) == 0 && self.trace.steps() > 0
    }
    pub fn steps(&self) -> u32 {
        self.trace.steps()
    }
    pub fn d(&self, rung: usize) -> f32 {
        self.trace.distance(rung)
    }
    /// The audit lane of the plan: 1 in 128 slots (the row carries no
    /// breed round, so the slot alone keys it).
    pub fn audit(&self) -> bool {
        splitmix(self.slot as u64 ^ 0xa0d1_7000) & 127 == 0
    }
}

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

pub struct DumpFile {
    pub header: Header,
    pub elites: Vec<Elite>,
    pub rows: Vec<Row>,
}

pub fn read(path: &str) -> Result<DumpFile> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    ensure!(
        bytes.len() >= 64 && &bytes[0..8] == b"EVODUMP1",
        "{path} is not a generation dump"
    );
    let b = &bytes[..64];
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
        arenas: b[55],
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
            arena: b[0],
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

/// Cell axes: contact, cadence, height, feet.
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
pub fn cell_of(a: [i32; 4], bins: [u8; 4]) -> Option<u16> {
    let b = bins.map(i32::from);
    if (0..4).any(|i| a[i] < 0 || a[i] >= b[i]) {
        return None;
    }
    Some((((a[0] * b[1] + a[1]) * b[2] + a[2]) * b[3] + a[3]) as u16)
}
pub fn cells(bins: [u8; 4]) -> usize {
    bins.iter().map(|&b| b as usize).product()
}
pub fn cadence_band(cell: u16, bins: [u8; 4]) -> Option<usize> {
    axes(cell, bins).map(|a| a[1] as usize)
}

/// Per arena and cell, the minimum over the cell and its radius-1
/// neighbours (81 cells on the four axes) of `value(elite)` among the
/// arena's behavior elites that have it; NaN when none do.
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

/// R4's table: per arena and cell, the minimum of the elites' final
/// distance over the cell and its neighbours on contact, cadence and height
/// (plus or minus one) and on feet (this bin and the next), 54 cells;
/// minus infinity when any of those cells that exist is empty.
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

/// Fisher's linear discriminant: the direction that separates class `a`
/// from class `b`, from sums in f64 with a small ridge.
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
    let trace: f64 = (0..k).map(|i| s[i][i] / n).sum::<f64>() / k.max(1) as f64;
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for j in 0..k {
            s[i][j] /= n;
        }
        s[i][i] += 1e-9 * trace.max(1e-12);
    }
    let rhs: Vec<f64> = (0..k).map(|i| ma[i] - mb[i]).collect();
    solve(s, rhs)
}

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

pub fn dot(w: &[f64], x: &[f64]) -> f64 {
    w.iter().zip(x).map(|(a, b)| a * b).sum()
}

/// The threshold that leaves `floor(tol * n)` of the protected scores below
/// it (stop when a score is below the threshold).
pub fn threshold(mut protected: Vec<f64>, tol: f64) -> f64 {
    if protected.is_empty() {
        return f64::NEG_INFINITY;
    }
    protected.sort_by(f64::total_cmp);
    let k = (tol * protected.len() as f64).floor() as usize;
    protected[k.min(protected.len() - 1)]
}

pub fn quantile(values: &mut [f32], q: f64) -> f32 {
    let values: &mut [f32] = values;
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

/// Spearman rank correlation.
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

/// Body size classes of the plan: 3 to 8, 9 to 16, 17 to 32 nodes.
pub fn size_class(nodes: u8) -> usize {
    match nodes {
        0..=8 => 0,
        9..=16 => 1,
        _ => 2,
    }
}
pub const SIZE_CLASSES: [&str; 3] = ["3-8 nodes", "9-16 nodes", "17-32 nodes"];

/// The rung ladder fitted on one half of the rows (`fit`) and applied to
/// any row.
pub struct Ladder {
    pub tol: f64,
    /// R3 against the generation's own 5 s bar (today's screen) or the
    /// per-cell bars.
    pub cell_bars: bool,
    pub r4: bool,
    /// R2's seventh feature, the parent neighbourhood's 2.5 s bar.
    pub bar_feature: bool,
    pub w1: Vec<f64>,
    pub b1: f64,
    pub w2: Vec<f64>,
    pub b2: f64,
    /// Per-cell R3 factor, and the arena bars at 2.5 and 5 s.
    pub f3: f32,
    pub arena150: Vec<f32>,
    pub arena300: Vec<f32>,
    pub nmin150: Vec<Vec<f32>>,
    pub nmin300: Vec<Vec<f32>>,
    pub nmin_final: Vec<Vec<f32>>,
    pub r4_ratio: f32,
    /// R4's floor: each emitter's median distance at 10 s.
    pub floor: std::collections::HashMap<(u8, u16, u8), f32>,
    pub bar: f32,
    pub islands: u8,
    pub arenas: u8,
}

/// What the ladder did to one row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fate {
    /// Stopped at rung 0 to 3.
    Stopped(usize),
    /// Ran to its natural end (a fall or the last step).
    Ran,
}

impl Ladder {
    fn emitter_key(r: &Row, arena: usize) -> (u8, u16, u8) {
        if r.cma != u16::MAX {
            (0, r.cma, 0)
        } else {
            (r.emitter + 1, 0, arena as u8)
        }
    }
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
    fn parent_min(table: &[Vec<f32>], r: &Row, h: &Header) -> f32 {
        let arena = r.arena(h);
        if r.nursery(h) || r.emitter == RESTART || r.parent_cell == u16::MAX {
            return f32::NAN;
        }
        // A neighbourhood whose slowest elite had not moved forward at that
        // time gives no bar; the arena's bar stands in.
        table
            .get(arena)
            .and_then(|t| t.get(r.parent_cell as usize))
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or(f32::NAN)
    }
    pub fn bar150(&self, r: &Row, h: &Header) -> f32 {
        let v = Self::parent_min(&self.nmin150, r, h);
        if v.is_finite() {
            v
        } else {
            self.arena150[r.arena(h)]
        }
    }
    /// The R3 bar of a row.
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
    pub fn pass3(&self, r: &Row, h: &Header) -> bool {
        r.steps() > RUNG_STEPS[2] && r.d(2) >= self.bar300(r, h)
    }

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
        // R1 and R2: class A passes R3, class B does not; exempt rows are
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
            let mut groups: std::collections::HashMap<(u8, u16, u8), Vec<f32>> = Default::default();
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

    /// The fate of row `r`.
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
    pub fn steps(fate: Fate, r: &Row) -> u32 {
        match fate {
            Fate::Stopped(k) => RUNG_STEPS[k],
            Fate::Ran => r.steps(),
        }
    }
}

/// Today's game: the 5 s screen at the generation's bar.
pub fn today_steps(r: &Row, h: &Header) -> u32 {
    if r.steps() > RUNG_STEPS[2] && r.d(2) < h.bar {
        RUNG_STEPS[2]
    } else {
        r.steps()
    }
}

/// Deterministic half split of the rows, by slot.
pub fn fit_half(r: &Row) -> bool {
    splitmix(r.slot as u64 ^ 0x05ee_df17) & 1 == 0
}
