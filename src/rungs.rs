//! This module holds the early rungs of a trial and the audit lane that fits
//! them. A rung stops a standard trial at 1 s (R1) or 2.5 s (R2) when a linear
//! score of six features that the kernel already measures says the creature
//! will not reach the 5 s bar (R3, `physics::Screen`), and a stopped creature
//! enters no archive. The audit creatures, one slot in 128 of the main
//! islands, run with every rule off, and `Audit::boundary` fits the next rules
//! on their rows at each generation boundary. The blocks bred next carry the
//! rules in `Config::rungs`, nurseries and immigrants are exempt from them,
//! and a breaker per cadence band turns a rung off where it would stop
//! creatures that enter archives.
use crate::creature_kernel::{RungTrace, f16_to_f32, f32_to_f16};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Creature flag in `Population::flags`: an audit creature. It runs with the
/// 5 s screen and both rungs off, so its trial samples the creatures that the
/// rules would have stopped.
pub const AUDIT: u8 = 1;
/// Creature flag: exempt from R1 and R2. It is set on bodies from a nursery
/// and on immigrants, because their bodies are new and most of them stop at
/// the 5 s screen anyway.
pub const EXEMPT: u8 = 2;
/// Creature flag: exempt from R1 because of its parent (`parent_exemptions`).
/// Either the rule would stop the strong parent itself at R1, so the rule
/// misjudges its family, or the parent's features are unknown.
pub const EXEMPT_R1: u8 = 4;
/// Creature flag: exempt from R2 because of its parent, as `EXEMPT_R1`.
pub const EXEMPT_R2: u8 = 8;
/// Creature flag: a body from a nursery of new random bodies. The 5 s screen
/// holds it to the young bar, the distance that the best of its own kind
/// reached (`physics::Screen::young_bar`).
pub const YOUNG: u8 = 16;
/// Creature flag: a body from a nursery of reshaped bodies. The 5 s screen
/// holds it to the bar of its own kind (`physics::Screen::reshaped_bar`).
pub const RESHAPED: u8 = 32;
/// The flag bits that exempt a creature from rung `r` (0 for R1, 1 for R2):
/// `EXEMPT` and the parent bit of that rung. `AUDIT` is not among them,
/// although an audit creature skips every rule.
pub fn exempt_bits(r: usize) -> u8 {
    EXEMPT | if r == 0 { EXEMPT_R1 } else { EXEMPT_R2 }
}

/// An elite's own features at R1, then at R2, as half-precision words. They
/// are all zero when its trial left no trace.
pub fn profile(trace: &RungTrace, period: f32) -> [u16; 2 * FEATURES] {
    if trace.steps() == 0 {
        return [0; 2 * FEATURES];
    }
    let mut out = [0u16; 2 * FEATURES];
    for r in 0..RUNGS {
        let f = features(trace, r, period);
        for i in 0..FEATURES {
            out[r * FEATURES + i] = f32_to_f16(f[i]);
        }
    }
    out
}
/// Which rungs the children of an elite skip under `rules`, as `EXEMPT_R1` and
/// `EXEMPT_R2` bits. `profile` holds the elite's features as the function
/// `profile` made them, or is `None` when it has none, and `strong` says the
/// elite is at or above the median fitness of the archive its child competes
/// in. A child skips both rungs when the features are unknown. A weak elite
/// exempts nothing. A strong one exempts the armed rungs that would stop the
/// elite itself, because a rule that stops a strong elite misjudges its family.
pub fn parent_exemptions(rules: &Rungs, profile: Option<&[u16; 2 * FEATURES]>, strong: bool) -> u8 {
    let Some(p) = profile.filter(|p| p.iter().any(|&w| w != 0)) else {
        return EXEMPT_R1 | EXEMPT_R2;
    };
    if !strong {
        return 0;
    }
    let mut bits = 0;
    for r in 0..RUNGS {
        let f: [f32; FEATURES] = std::array::from_fn(|i| f16_to_f32(p[r * FEATURES + i]));
        if rules.0[r].armed() && rules.0[r].raw_stops(&f) {
            bits |= if r == 0 { EXEMPT_R1 } else { EXEMPT_R2 };
        }
    }
    bits
}

/// One slot in this many holds an audit creature (`is_audit`).
pub const AUDIT_ONE_IN: u64 = 128;
/// Features per rung: distance, speed over the last half second, share of
/// nodes that touched the ground, head shake, mean muscle energy store, the
/// rhythm period.
pub const FEATURES: usize = 6;
/// Number of early rungs: R1 at 1 s and R2 at 2.5 s.
pub const RUNGS: usize = 2;
/// The step each early rung sits at (1 s and 2.5 s at 60 Hz).
pub const RUNG_STEPS: [u32; RUNGS] = [60, 150];
/// The step of the 5 s screen, the third rung (R3), at 60 Hz.
pub const SCREEN_STEPS: u32 = 300;
/// Number of cadence bands that each rung has a breaker for
/// (`RungTrace::BAND_COUNT`).
pub const BANDS: usize = RungTrace::BAND_COUNT;
/// Generations of audit rows the fit pools.
pub const WINDOW: usize = 8;
/// Share of the creatures that pass the 5 s bar that a rung may stop. The fit
/// puts the threshold of a rung where this share of those audit rows scores
/// below it (`Audit::fit_rung`).
pub const BUDGET: f64 = 1e-3;
/// Share of the entrants the 5 s screen would have kept that a rung may stop.
/// No code reads it. The limit that applies to the entrants is
/// `TRUST_STOPPED`.
pub const ENTRANT_BUDGET: f64 = 1e-2;
/// Generations of judgments that a rung is trusted over (`Audit::trusted`). A
/// rung is armed only while the rule fitted before each of them would have
/// stopped few of the creatures that the archives grow from. While the
/// archives still climb, those are the entrants the 5 s screen would have
/// kept. A rule fitted in the first generations, when most creatures that
/// pass the bar barely use their muscles, stops the walkers that the archives
/// grow from. On a plateau (the global best has stood for 30 generations) few
/// creatures enter an archive. Those that do are mostly creatures that fall
/// early and improve a niche of weak bodies, which a rule stops by design.
/// There the guard counts the audit creatures that reach the 5 s bar instead.
const JUDGED: usize = 4;
/// Entrants the 5 s screen would have kept, at least this many over the judged
/// generations, for a rung to be trusted while the archives climb.
const TRUST_ENTRANTS: u32 = 60;
/// Audit creatures that reach the 5 s bar, at least this many over the judged
/// generations, for a rung to be trusted on a plateau.
const TRUST_PASSERS: u32 = 1000;
/// The largest share of those creatures that the rule may have stopped.
const TRUST_STOPPED: f64 = 0.03;
/// Rows of the creatures that reach the 5 s bar that a rung needs before it is
/// armed, enough for a 1 in 1,000 quantile to rest on at least 5 rows. At 400k
/// creatures per generation the window never holds that many, and the rungs
/// stay off. With so few rows the fit stopped the slow walkers that the early
/// archives grow from.
const MIN_PASSING: u64 = 5000;
/// Rows of the other creatures that a rung needs before it is armed, enough
/// for a covariance.
const MIN_OTHER: u64 = 400;
/// Entrant misses per 10k audit rows of a band (entrants the 5 s screen would
/// have kept that the rule would stop) above which the band counts a strike.
const BAND_MISS_LIMIT: f64 = 100.0;
/// Generations in a row over `BAND_MISS_LIMIT` that turn a band's rung off.
const STRIKES: u8 = 3;
/// Generations in a row under `BAND_MISS_LIMIT` that turn it on again.
const RECOVERY: u8 = 3;
/// Bands with fewer audit rows than this in a generation decide nothing.
const BAND_MIN_ROWS: u32 = 100;

/// A developer diagnostic, `EVOLUTION_NO_RUNGS`: no audit lane and no early
/// rungs, to measure the game without them in the same build. The variable is
/// read once, on first use.
pub fn disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("EVOLUTION_NO_RUNGS").is_some())
}

/// Whether the child bred for `slot` in breeding round `round` is an audit
/// creature. A hash of `seed`, `round` and `slot` picks one slot in
/// `AUDIT_ONE_IN`, and `disabled` picks none. Breeding (`breed_block`) also
/// leaves out the slots of wild islands, which run in worlds of their own.
pub fn is_audit(seed: u64, round: u64, slot: usize) -> bool {
    if disabled() {
        return false;
    }
    let mut x = seed
        ^ round.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (slot as u64).wrapping_mul(0xd134_2543_de82_ef95);
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (x ^ (x >> 31)).is_multiple_of(AUDIT_ONE_IN)
}

/// One rung's rule, as the kernel reads it (`Params` in the kernel): stop when
/// the dot product of `weights` with the six features, taken as a chain of
/// fused multiply-adds from zero, is below `bias`. `off` has bit `b` set when
/// band `b` is off.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Rung {
    pub weights: [f32; FEATURES],
    pub bias: f32,
    pub off: u32,
}
impl Rung {
    /// A rule that never stops anything.
    pub const NEVER: Self = Self {
        weights: [0.0; FEATURES],
        bias: f32::NEG_INFINITY,
        off: 0,
    };
    /// Dot product of `weights` with features `f`, via fused multiply-add.
    pub fn score(&self, f: &[f32; FEATURES]) -> f32 {
        let mut s = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..FEATURES {
            s = self.weights[i].mul_add(f[i], s);
        }
        s
    }
    /// Whether the rule stops a creature with features `f` in `band`.
    pub fn stops(&self, f: &[f32; FEATURES], band: usize) -> bool {
        (self.off >> band) & 1 == 0 && self.raw_stops(f)
    }
    /// The same ignoring the bands.
    pub fn raw_stops(&self, f: &[f32; FEATURES]) -> bool {
        f.iter().all(|v| v.is_finite()) && self.score(f) < self.bias
    }
    /// Whether the rule is armed (has a finite bias).
    pub fn armed(&self) -> bool {
        self.bias.is_finite()
    }
}

/// The rules of both early rungs, fixed per block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rungs(pub [Rung; RUNGS]);

/// The features of rung `r` (0 or 1) from a trial's trace and the creature's
/// rhythm period, rounded to half precision as the kernel rounds them.
pub fn features(trace: &RungTrace, r: usize, period: f32) -> [f32; FEATURES] {
    [
        trace.distance(r),
        trace.speed(r),
        trace.touched(r),
        trace.head_shake(r),
        trace.energy(r),
        f16_to_f32(f32_to_f16(period)),
    ]
}
/// The half-precision word the kernel reads for a period (`kernel::pack`).
pub fn period_half(period: f32) -> u16 {
    f32_to_f16(period)
}

/// What one audit creature tells the fit.
#[derive(Clone, Copy, Debug)]
pub struct AuditRow {
    pub trace: RungTrace,
    pub period: f32,
    pub exempt: bool,
    /// The rungs its parent exempted it from (`exempt_bits`).
    pub parent_exempt: u8,
    /// Its block had a screen bar, so `pass3` means something.
    pub bar_known: bool,
    /// Its distance at 5 s, or at its fall before that, was at or above its
    /// block's bar: a creature the 5 s screen would not have stopped and that
    /// reached the bar. The rungs must leave these alone.
    pub pass3: bool,
    /// The 5 s screen would have stopped it: it lived past 5 s below the bar.
    pub below_bar: bool,
    /// It entered an archive.
    pub entrant: bool,
}
impl AuditRow {
    /// Whether the rules never apply to this creature at rung `r`.
    fn skips(&self, r: usize) -> bool {
        self.exempt || self.parent_exempt & exempt_bits(r) != 0
    }
    fn alive(&self, r: usize) -> bool {
        self.trace.steps() > RUNG_STEPS[r]
    }
    fn features(&self, r: usize) -> Option<[f32; FEATURES]> {
        let f = features(&self.trace, r, self.period);
        f.iter().all(|v| v.is_finite()).then_some(f)
    }
}

/// Sums of one class of feature vectors, enough for a pooled covariance.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Sums {
    n: u64,
    sum: [f64; FEATURES],
    xx: [[f64; FEATURES]; FEATURES],
}
impl Sums {
    fn add(&mut self, x: &[f64; FEATURES]) {
        self.n += 1;
        for i in 0..FEATURES {
            self.sum[i] += x[i];
            for j in 0..FEATURES {
                self.xx[i][j] += x[i] * x[j];
            }
        }
    }
}
/// One rung's share of a generation's audit rows: the sums of the creatures
/// that passed 5 s (class A) and of those that did not (class B), and the
/// class A features themselves, which place the threshold.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct RungSet {
    a: Sums,
    b: Sums,
    a_rows: Vec<[u16; FEATURES]>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct GenAudit {
    rows: u32,
    sets: [RungSet; RUNGS],
}
impl GenAudit {
    fn of(rows: &[AuditRow]) -> Self {
        let mut g = Self::default();
        for row in rows {
            g.rows += 1;
            if row.exempt || !row.bar_known {
                continue;
            }
            for r in 0..RUNGS {
                if !row.alive(r) {
                    continue;
                }
                let Some(f) = row.features(r) else { continue };
                let x = f.map(f64::from);
                let set = &mut g.sets[r];
                if row.pass3 {
                    set.a.add(&x);
                    set.a_rows.push(f.map(f32_to_f16));
                } else {
                    set.b.add(&x);
                }
            }
        }
        g
    }
}

/// The rung breaker of one cadence band (`BAND_MISS_LIMIT`).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct Breaker {
    strikes: u8,
    calm: u8,
    off: bool,
}
impl Breaker {
    fn update(&mut self, over: bool) {
        if over {
            self.calm = 0;
            self.strikes = (self.strikes + 1).min(STRIKES);
            if self.strikes >= STRIKES {
                self.off = true;
            }
        } else {
            self.strikes = 0;
            self.calm = (self.calm + 1).min(RECOVERY);
            if self.calm >= RECOVERY {
                self.off = false;
            }
        }
    }
}

/// What a finished generation looked like, for the stage log.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub creatures: u64,
    /// Steps the standard trials of the generation ran.
    pub steps: u64,
    /// Creatures stopped by R1, R2 and the 5 s screen.
    pub stops: [u64; 3],
    pub audit_rows: u32,
    /// The audit rows of the generation's own rules (the rules in force when
    /// its blocks were bred, so the check is out of sample): rows alive at the
    /// rung, those the rule would have stopped, the entrants and the 5 s
    /// passers among them, per rung.
    pub subject: [u32; RUNGS],
    pub stopped: [u32; RUNGS],
    pub entrant_misses: [u32; RUNGS],
    /// Rows alive at the rung whose parent exempted them from it.
    pub parent_skipped: [u32; RUNGS],
    /// Of the entrants a rung would have stopped, those the 5 s screen would
    /// not have stopped: the cost beyond today's game.
    pub extra_misses: [u32; RUNGS],
    pub pass_misses: [u32; RUNGS],
    /// Share of the audit rows in the final top 1% and top 10% by distance
    /// that the ladder (both rungs and the 5 s screen) would have kept, and
    /// the share the 5 s screen alone would have kept.
    pub top1_kept: f32,
    pub top10_kept: f32,
    pub top1_screen: f32,
    pub top10_screen: f32,
    pub armed: [bool; RUNGS],
    pub bands_off: [u8; RUNGS],
    /// Audit rows the fit pools after the boundary.
    pub window_rows: u32,
}
impl Report {
    pub fn steps_per_creature(&self) -> f64 {
        self.steps as f64 / self.creatures.max(1) as f64
    }
    /// Entrant misses of rung `r` per 10k audit rows alive at it.
    pub fn misses_per_10k(&self, r: usize) -> f64 {
        1e4 * f64::from(self.entrant_misses[r]) / f64::from(self.subject[r].max(1))
    }
    /// The same for the entrants the 5 s screen would have kept.
    pub fn extra_misses_per_10k(&self, r: usize) -> f64 {
        1e4 * f64::from(self.extra_misses[r]) / f64::from(self.subject[r].max(1))
    }
}

/// One generation's judgment of a rule: of the entrants the 5 s screen would
/// have kept, how many there were and how many the rule would have stopped,
/// and the same for the audit creatures that reach the 5 s bar. Packed in 16
/// bits each (both counts of a pair are scaled down together past 65,535).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Judged {
    entrants: (u32, u32),
    passers: (u32, u32),
}
impl Judged {
    fn pack(self) -> (u32, u32) {
        let fit = |(n, stopped): (u32, u32)| {
            let scale = n.div_ceil(0xffff).max(1);
            (n / scale, stopped / scale)
        };
        let (n_in, s_in) = fit(self.entrants);
        let (n, s) = fit(self.passers);
        (n_in | n << 16, s_in | s << 16)
    }
    fn unpack((n, s): (u32, u32)) -> Self {
        Self {
            entrants: (n & 0xffff, s & 0xffff),
            passers: (n >> 16, s >> 16),
        }
    }
}

/// The audit lane's state: the rows of the generation in progress, the
/// window of earlier generations, the breakers, and the last report. The
/// window and the breakers are saved.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Audit {
    window: VecDeque<GenAudit>,
    breakers: [[Breaker; BANDS]; RUNGS],
    /// Per rung and generation, what the rule fitted before that generation
    /// would have stopped among the audit creatures alive at the rung and not
    /// exempt: the two counts of `Judged`, packed into the two words (the
    /// layout of the save is kept; an entry of an older save reads as its
    /// entrant counts, as it was written).
    judged: [VecDeque<(u32, u32)>; RUNGS],
    #[serde(skip)]
    rows: Vec<AuditRow>,
    #[serde(skip)]
    tally: Report,
    #[serde(skip)]
    last: Report,
}

impl Audit {
    /// One absorbed block's standard trials: steps, stops and audit rows are
    /// the caller's to count with `note` and `record`.
    pub fn note(&mut self, trace: &RungTrace, screened: bool) {
        self.tally.creatures += 1;
        self.tally.steps += u64::from(trace.steps());
        if screened {
            match trace.stopped_by() {
                1 => self.tally.stops[0] += 1,
                2 => self.tally.stops[1] += 1,
                _ => self.tally.stops[2] += 1,
            }
        }
    }
    pub fn record(&mut self, row: AuditRow) {
        self.rows.push(row);
    }
    pub fn last(&self) -> &Report {
        &self.last
    }
    /// Forgets everything measured in a world that is gone.
    pub fn clear(&mut self) {
        let last = std::mem::take(&mut self.last);
        *self = Self::default();
        self.last = last;
    }
    /// Audit rows in the window.
    pub fn window_rows(&self) -> u32 {
        self.window.iter().map(|g| g.rows).sum()
    }

    /// The generation boundary: judges the rules in force on the generation's
    /// audit rows, moves the breakers, pools the rows into the window and fits
    /// the rules the next blocks carry. `plateau` says whether the archives
    /// have stopped climbing (`trusted`).
    pub fn boundary(&mut self, in_force: Option<Rungs>, plateau: bool) -> Option<Rungs> {
        let rows = std::mem::take(&mut self.rows);
        let mut report = std::mem::take(&mut self.tally);
        report.audit_rows = rows.len() as u32;
        if let Some(rules) = in_force {
            self.judge(&rules, &rows, &mut report);
        }
        self.trial(&rows);
        self.window.push_back(GenAudit::of(&rows));
        while self.window.len() > WINDOW {
            self.window.pop_front();
        }
        let rules = self.fit(plateau);
        report.window_rows = self.window_rows();
        if let Some(rules) = rules {
            for r in 0..RUNGS {
                report.armed[r] = rules.0[r].armed();
                report.bands_off[r] = rules.0[r].off.count_ones() as u8;
            }
        }
        self.last = report;
        rules
    }

    /// Judges the rule the window fits before these rows (the rule that
    /// would have been in force while they were measured), on the entrants
    /// the 5 s screen would have kept and on the creatures that reach the 5 s
    /// bar.
    fn trial(&mut self, rows: &[AuditRow]) {
        for r in 0..RUNGS {
            let Some(rung) = self.fit_rung(r) else {
                self.judged[r].clear();
                continue;
            };
            let mut judged = Judged::default();
            for row in rows
                .iter()
                .filter(|row| !row.skips(r) && row.bar_known && row.alive(r))
            {
                let Some(f) = row.features(r) else { continue };
                let stops = u32::from(rung.raw_stops(&f));
                if row.entrant && !row.below_bar {
                    judged.entrants.0 += 1;
                    judged.entrants.1 += stops;
                }
                if row.pass3 {
                    judged.passers.0 += 1;
                    judged.passers.1 += stops;
                }
            }
            self.judged[r].push_back(judged.pack());
            while self.judged[r].len() > JUDGED {
                self.judged[r].pop_front();
            }
        }
    }
    /// Whether rung `r` may be armed.
    fn trusted(&self, r: usize, plateau: bool) -> bool {
        let mut total = Judged::default();
        for &packed in &self.judged[r] {
            let j = Judged::unpack(packed);
            total.entrants.0 += j.entrants.0;
            total.entrants.1 += j.entrants.1;
            total.passers.0 += j.passers.0;
            total.passers.1 += j.passers.1;
        }
        let (n, stopped, least) = if plateau {
            (total.passers.0, total.passers.1, TRUST_PASSERS)
        } else {
            (total.entrants.0, total.entrants.1, TRUST_ENTRANTS)
        };
        n >= least && f64::from(stopped) <= TRUST_STOPPED * f64::from(n)
    }

    /// The rules in force on the audit rows of their own generation: the
    /// report's miss counts, and the breakers.
    fn judge(&mut self, rules: &Rungs, rows: &[AuditRow], report: &mut Report) {
        let mut band_rows = [[0u32; BANDS]; RUNGS];
        let mut band_misses = [[0u32; BANDS]; RUNGS];
        for row in rows.iter().filter(|r| !r.exempt && r.bar_known) {
            for r in 0..RUNGS {
                if !row.alive(r) {
                    continue;
                }
                let Some(f) = row.features(r) else { continue };
                let band = row.trace.band(r);
                report.subject[r] += 1;
                report.parent_skipped[r] += u32::from(row.parent_exempt & exempt_bits(r) != 0);
                band_rows[r][band] += 1;
                if !row.skips(r) && rules.0[r].stops(&f, band) {
                    report.stopped[r] += 1;
                    report.entrant_misses[r] += u32::from(row.entrant);
                    report.extra_misses[r] += u32::from(row.entrant && !row.below_bar);
                    report.pass_misses[r] += u32::from(row.pass3);
                }
                // The breaker sees what the rule would do with every band on.
                if !row.skips(r)
                    && rules.0[r].armed()
                    && rules.0[r].raw_stops(&f)
                    && row.entrant
                    && !row.below_bar
                {
                    band_misses[r][band] += 1;
                }
            }
        }
        for r in 0..RUNGS {
            if !rules.0[r].armed() {
                continue;
            }
            for b in 0..BANDS {
                if band_rows[r][b] < BAND_MIN_ROWS {
                    continue;
                }
                let per_10k = 1e4 * f64::from(band_misses[r][b]) / f64::from(band_rows[r][b]);
                self.breakers[r][b].update(per_10k > BAND_MISS_LIMIT);
            }
        }
        // The final top 1% and 10% by distance among the audit rows.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by(|&a, &b| rows[b].trace.fitness.total_cmp(&rows[a].trace.fitness));
        // Kept by the screen (or with no bar yet), and by the rungs too.
        let screen_keeps = |row: &AuditRow| !row.below_bar;
        let rungs_keep = |row: &AuditRow| {
            row.exempt
                || !(0..RUNGS).any(|r| {
                    !row.skips(r)
                        && row.alive(r)
                        && row
                            .features(r)
                            .is_some_and(|f| rules.0[r].stops(&f, row.trace.band(r)))
                })
        };
        let share = |count: usize, keeps: &dyn Fn(&AuditRow) -> bool| {
            let top = &order[..count.max(1).min(order.len())];
            if top.is_empty() {
                return 100.0;
            }
            100.0 * top.iter().filter(|&&i| keeps(&rows[i])).count() as f32 / top.len() as f32
        };
        let ladder = |row: &AuditRow| screen_keeps(row) && rungs_keep(row);
        report.top1_kept = share(rows.len() / 100, &ladder);
        report.top10_kept = share(rows.len() / 10, &ladder);
        report.top1_screen = share(rows.len() / 100, &screen_keeps);
        report.top10_screen = share(rows.len() / 10, &screen_keeps);
    }

    /// The rules the window fits, with the breakers' bands off. A rung
    /// needs enough rows of both classes and the trust of `trusted`.
    pub fn fit(&self, plateau: bool) -> Option<Rungs> {
        if disabled() {
            return None;
        }
        let mut rules = [Rung::NEVER; RUNGS];
        let mut any = false;
        #[allow(clippy::needless_range_loop)]
        for r in 0..RUNGS {
            if !self.trusted(r, plateau) {
                continue;
            }
            if let Some(mut rung) = self.fit_rung(r) {
                rung.off = (0..BANDS)
                    .filter(|&b| self.breakers[r][b].off)
                    .fold(0u32, |m, b| m | 1 << b);
                rules[r] = rung;
                any = true;
            }
        }
        any.then_some(Rungs(rules))
    }

    /// Fisher's linear discriminant of the creatures that pass 5 s against
    /// those that do not, from the pooled sums in window order, and the
    /// threshold that leaves `BUDGET` of the class A rows below it.
    fn fit_rung(&self, r: usize) -> Option<Rung> {
        let (mut a, mut b) = (Sums::default(), Sums::default());
        for g in &self.window {
            let set = &g.sets[r];
            for (total, part) in [(&mut a, &set.a), (&mut b, &set.b)] {
                total.n += part.n;
                for i in 0..FEATURES {
                    total.sum[i] += part.sum[i];
                    for j in 0..FEATURES {
                        total.xx[i][j] += part.xx[i][j];
                    }
                }
            }
        }
        if a.n < MIN_PASSING || b.n < MIN_OTHER {
            return None;
        }
        let mean = |s: &Sums| s.sum.map(|v| v / s.n as f64);
        let (ma, mb) = (mean(&a), mean(&b));
        let n = (a.n + b.n - 2) as f64;
        let mut s = vec![vec![0.0f64; FEATURES]; FEATURES];
        for (set, m) in [(&a, &ma), (&b, &mb)] {
            for i in 0..FEATURES {
                for j in 0..FEATURES {
                    s[i][j] += (set.xx[i][j] - set.n as f64 * m[i] * m[j]) / n;
                }
            }
        }
        let trace: f64 = (0..FEATURES).map(|i| s[i][i]).sum::<f64>() / FEATURES as f64;
        for (i, row) in s.iter_mut().enumerate() {
            row[i] += 1e-9 * trace.max(1e-12);
        }
        let rhs: Vec<f64> = (0..FEATURES).map(|i| ma[i] - mb[i]).collect();
        let mut w = solve(s, rhs);
        let largest = w.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if !largest.is_finite() || largest == 0.0 {
            return None;
        }
        w.iter_mut().for_each(|v| *v /= largest);
        let mut rung = Rung::NEVER;
        #[allow(clippy::needless_range_loop)]
        for i in 0..FEATURES {
            rung.weights[i] = w[i] as f32;
        }
        // Class A scores, by the kernel's own arithmetic.
        let mut scores: Vec<f32> = self
            .window
            .iter()
            .flat_map(|g| g.sets[r].a_rows.iter())
            .map(|row| rung.score(&row.map(f16_to_f32)))
            .collect();
        scores.sort_by(f32::total_cmp);
        let k = (BUDGET * scores.len() as f64).floor() as usize;
        let bias = *scores.get(k.min(scores.len().saturating_sub(1)))?;
        if !bias.is_finite() {
            return None;
        }
        rung.bias = bias;
        Some(rung)
    }
}

/// Solves `m x = v` by Gauss-Jordan elimination with partial pivoting.
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

#[cfg(test)]
mod tests;
