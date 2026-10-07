//! Unit tests for `rungs`. They build kernel traces and audit rows by hand, so
//! they need no GPU. They cover the audit sample, the stop rule, the fit, the
//! breaker, the parent exemptions, the trust check and the packing of the
//! judgments it reads. `tests/rungs.rs` checks the rungs on the CUDA kernel.
use super::*;

/// A trace of a creature that ran `steps` steps, built from the packed words
/// the kernel writes. `d60`, `d150` and `d300` are its distances at 1 s, 2.5 s
/// and 5 s, and `bands` are its cadence bands at the two rungs. Its speed at
/// each rung equals its distance there. Every trace has the same other
/// features: all nodes touched the ground, an energy store of 0.5 and a head
/// shake of 1.0. The end code holds the bands and no flag. The fitness is
/// `d300`.
fn trace(d60: f32, d150: f32, d300: f32, steps: u32, bands: [u16; 2]) -> RungTrace {
    let h = |v: f32| u32::from(f32_to_f16(v));
    let pair = |lo: f32, hi: f32| h(lo) | h(hi) << 16;
    let code = bands[0] << 8 | bands[1] << 11;
    RungTrace {
        words: [
            pair(d60, d150),
            pair(d300, d300),
            u32::from(code) | steps << 16,
            pair(d60, d150),
            pair(1.0, 0.5),
            pair(1.0, 0.5),
            pair(1.0, 1.0),
        ],
        fitness: d300,
    }
}

/// An audit row of a creature that reaches the 5 s bar (`pass3`) or not, with
/// its distances at 1 s and 2.5 s moved by `jitter`. A creature that passes is
/// an entrant. One that does not is below the bar.
fn row(pass3: bool, exempt: bool, jitter: f32) -> AuditRow {
    // Creatures that pass 5 s are further along at 2.5 s.
    let d150 = if pass3 { 3.0 + jitter } else { 0.5 + jitter };
    AuditRow {
        trace: trace(
            d150 * 0.4,
            d150,
            if pass3 { 8.0 } else { 1.0 },
            1200,
            [0, 0],
        ),
        period: 0.25,
        exempt,
        parent_exempt: 0,
        bar_known: true,
        pass3,
        below_bar: !pass3,
        entrant: pass3,
    }
}

#[test]
fn one_in_128_slots_are_audit_creatures() {
    let n = 2_000_000usize;
    let audit = (0..n).filter(|&slot| is_audit(5051, 7, slot)).count();
    let share = audit as f64 / n as f64;
    assert!((share - 1.0 / 128.0).abs() < 5e-4, "{share}");
    // The choice is a function of seed, round and slot only. The same inputs
    // agree and another round picks other slots.
    assert_eq!(is_audit(1, 2, 3), is_audit(1, 2, 3));
    let moved = (0..4096)
        .filter(|&s| is_audit(1, 2, s) != is_audit(1, 3, s))
        .count();
    assert!(moved > 0);
}

#[test]
fn a_rule_stops_below_its_bias_and_not_in_an_off_band() {
    // Only the distance counts: the rule stops a creature under 1 m.
    let mut rung = Rung::NEVER;
    rung.weights[0] = 1.0;
    rung.bias = 1.0;
    let low = [0.5, 0.0, 0.0, 0.0, 0.0, 0.2];
    let high = [2.0, 0.0, 0.0, 0.0, 0.0, 0.2];
    assert!(rung.stops(&low, 3));
    assert!(!rung.stops(&high, 3));
    // With band 3 off, the rule leaves band 3 alone and still stops band 2.
    rung.off = 1 << 3;
    assert!(!rung.stops(&low, 3));
    assert!(rung.stops(&low, 2));
    // `Rung::NEVER` stops nothing.
    assert!(!Rung::NEVER.raw_stops(&low));
    // A feature that is not a number never stops a creature.
    let mut bad = low;
    bad[1] = f32::NAN;
    rung.off = 0;
    assert!(!rung.stops(&bad, 0));
}

#[test]
fn the_fit_stops_the_slow_and_spares_the_budget_of_the_fast() {
    // Three generations of 40,000 audit rows, one in eight reaching the 5 s
    // bar. The first fills the window. After that, the rule fitted on the
    // window judges each new generation.
    let mut audit = Audit::default();
    let mut k = 0u32;
    for _ in 0..3 {
        for i in 0..40_000u32 {
            k += 1;
            let jitter = (k.wrapping_mul(2654435761) >> 16) as f32 / 65536.0;
            audit.record(row(i % 8 == 0, false, jitter));
        }
        audit.boundary(None, false);
    }
    let rules = audit.fit(false).expect("enough rows");
    // R2 separates the classes: the slow ones stop, the fast ones go on.
    let mut stopped_slow = 0;
    let mut stopped_fast = 0;
    for i in 0..2000u32 {
        let jitter = (i.wrapping_mul(2654435761) >> 16) as f32 / 65536.0;
        let slow = row(false, false, jitter);
        let fast = row(true, false, jitter);
        stopped_slow += rules.0[1].stops(&slow.features(1).unwrap(), 0) as u32;
        stopped_fast += rules.0[1].stops(&fast.features(1).unwrap(), 0) as u32;
    }
    assert!(stopped_slow > 1900, "{stopped_slow}");
    // The budget is 1 in 1,000 of the fast ones, so about 2 of these 2,000
    // stop. A limit of 6 leaves room for chance.
    assert!(stopped_fast <= 6, "{stopped_fast}");
}

#[test]
fn nothing_is_armed_without_enough_rows_of_both_classes() {
    // Only 1,500 rows reach the 5 s bar. A rung needs `MIN_PASSING` of them.
    let mut audit = Audit::default();
    for i in 0..3000u32 {
        audit.record(row(i % 2 == 0, false, 0.1));
    }
    assert!(audit.boundary(None, false).is_none());
    // Exempt rows (nurseries, immigrants) never enter the fit, however many
    // there are.
    let mut audit = Audit::default();
    for i in 0..50_000u32 {
        audit.record(row(i % 2 == 0, true, i as f32 * 1e-4));
    }
    assert!(audit.boundary(None, false).is_none());
}

#[test]
fn a_band_turns_off_after_three_bad_generations_and_back_on_after_three_good() {
    // `update(true)` reports a generation over `BAND_MISS_LIMIT`.
    let mut b = Breaker::default();
    b.update(true);
    b.update(true);
    assert!(!b.off);
    b.update(true);
    assert!(b.off);
    b.update(false);
    b.update(false);
    assert!(b.off);
    b.update(false);
    assert!(!b.off);
    // A good generation in between resets the strikes.
    let mut b = Breaker::default();
    b.update(true);
    b.update(true);
    b.update(false);
    b.update(true);
    assert!(!b.off);
}

#[test]
fn the_window_keeps_eight_generations_and_survives_a_save() {
    // Eleven generations of 10,000 rows, one in five reaching the 5 s bar.
    let mut audit = Audit::default();
    for g in 0..11u32 {
        for i in 0..10_000u32 {
            audit.record(row(i % 5 == 0, false, ((i + g) % 1000) as f32 * 1e-3));
        }
        audit.boundary(None, false);
    }
    assert_eq!(audit.window.len(), WINDOW);
    // A loaded save fits the same rules.
    let bytes = bincode::serialize(&audit).unwrap();
    let back: Audit = bincode::deserialize(&bytes).unwrap();
    assert_eq!(back.fit(false), audit.fit(false));
    assert!(audit.fit(false).is_some());
}

#[test]
fn children_of_a_parent_the_rules_would_stop_skip_that_rung() {
    // R1 never stops. R2 stops a creature under 1 m at 2.5 s.
    let mut r2 = Rung::NEVER;
    r2.weights[0] = 1.0;
    r2.bias = 1.0;
    let rules = Rungs([Rung::NEVER, r2]);
    let slow = profile(&trace(0.2, 0.5, 1.0, 1200, [0, 0]), 0.25);
    let fast = profile(&trace(1.0, 3.0, 8.0, 1200, [0, 0]), 0.25);
    assert_eq!(parent_exemptions(&rules, Some(&slow), true), EXEMPT_R2);
    assert_eq!(parent_exemptions(&rules, Some(&fast), true), 0);
    // A weak elite that the rules stop gives its children no exemption. That
    // is where the rules should work.
    assert_eq!(parent_exemptions(&rules, Some(&slow), false), 0);
    // An elite with an unknown profile exempts its children from both rungs.
    // The lineage may hold no record of it (`None`), or its trial may have
    // left no trace (all zeros).
    assert_eq!(parent_exemptions(&rules, None, true), EXEMPT_R1 | EXEMPT_R2);
    assert_eq!(
        parent_exemptions(&rules, Some(&[0; 2 * FEATURES]), true),
        EXEMPT_R1 | EXEMPT_R2
    );
}

/// Records a generation of 40,000 audit rows, one in eight reaching the 5 s
/// bar, and ends it at the generation boundary. `g` varies the jitter. `tweak`
/// is called with each row's index and the row, and may change the row.
fn generation(audit: &mut Audit, g: u32, plateau: bool, tweak: impl Fn(u32, &mut AuditRow)) {
    for i in 0..40_000u32 {
        let jitter = ((i + g).wrapping_mul(2654435761) >> 16) as f32 / 65536.0;
        let mut row = row(i % 8 == 0, false, jitter);
        tweak(i, &mut row);
        audit.record(row);
    }
    audit.boundary(None, plateau);
}

/// Makes the early fallers the only entrants. One row in 16 is a creature that
/// falls early with poor features and enters an archive: the 5 s screen keeps
/// it, and the fitted rule stops it.
fn early_fallers_enter(i: u32, row: &mut AuditRow) {
    row.entrant = false;
    if i % 8 == 1 && i % 16 == 1 {
        row.below_bar = false;
        row.entrant = true;
    }
}

#[test]
fn while_the_archives_climb_a_rung_that_stops_their_entrants_is_not_armed() {
    let mut audit = Audit::default();
    for g in 0..6u32 {
        generation(&mut audit, g, false, early_fallers_enter);
    }
    assert!(!audit.trusted(1, false));
    assert!(audit.fit(false).is_none_or(|rules| !rules.0[1].armed()));
}

#[test]
fn on_a_plateau_a_rung_arms_whatever_the_creatures_that_enter_archives_look_like() {
    // On a plateau the guard counts the audit creatures that reach the 5 s
    // bar, so the entrants decide nothing. Try the early fallers that improve
    // a niche of weak bodies as the only entrants, then no entrants at all.
    for tweak in [early_fallers_enter as fn(u32, &mut AuditRow), |_, row| {
        row.entrant = false
    }] {
        let mut audit = Audit::default();
        for g in 0..4u32 {
            generation(&mut audit, g, true, tweak);
        }
        assert!(audit.trusted(0, true) && audit.trusted(1, true));
        assert!(
            audit
                .fit(true)
                .is_some_and(|rules| rules.0[0].armed() && rules.0[1].armed())
        );
    }
}

#[test]
fn a_rung_that_stops_the_creatures_above_the_bar_is_not_armed_on_a_plateau() {
    let mut audit = Audit::default();
    for g in 0..4u32 {
        generation(&mut audit, g, true, |_, _| {});
    }
    assert!(audit.trusted(0, true) && audit.trusted(1, true));
    // A fifth of the creatures that pass the bar are slow at the rungs: the
    // rule fitted before the generation stops them.
    generation(&mut audit, 4, true, |i, row| {
        if i % 8 == 0 && i % 40 == 0 {
            // The parameter `row` hides the function, so name it by its path.
            let slow = self::row(false, false, 0.1);
            row.trace = slow.trace;
        }
    });
    assert!(!audit.trusted(0, true) && !audit.trusted(1, true));
    assert!(
        audit
            .fit(true)
            .is_none_or(|rules| !rules.0[0].armed() && !rules.0[1].armed())
    );
}

#[test]
fn the_judgments_pack_into_two_words_and_scale_down_together() {
    // Counts that fit in 16 bits come back as they went in.
    let j = Judged {
        entrants: (30, 1),
        passers: (2300, 5),
    };
    assert_eq!(Judged::unpack(j.pack()), j);
    let big = Judged {
        entrants: (70_000, 1_400),
        passers: (200_000, 4_000),
    };
    let back = Judged::unpack(big.pack());
    // Past 65,535 both counts of a pair shrink by the same factor, so the 2%
    // share stays.
    assert!(back.passers.0 <= 0xffff && back.entrants.0 <= 0xffff);
    assert!((back.passers.1 as f64 / back.passers.0 as f64 - 0.02).abs() < 1e-3);
    assert!((back.entrants.1 as f64 / back.entrants.0 as f64 - 0.02).abs() < 1e-3);
}

#[test]
fn while_the_archives_climb_the_guard_is_the_entrant_guard_it_always_was() {
    // The entrant guard written out again, outside `Audit`. A rung is trusted
    // when the last 4 generations hold at least 60 counted entrants and the
    // rule the window fitted before each generation would have stopped at most
    // 3% of them. An entrant counts when the 5 s screen keeps it, it is alive
    // at the rung and it is not exempt from the rung.
    let mut audit = Audit::default();
    let mut history: [Vec<(u32, u32)>; RUNGS] = Default::default();
    let mut seen = [false; 2];
    let mut state = 12345u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for g in 0..14u32 {
        // 30,000 audit rows: one in six reaches the 5 s bar, one in 50 is
        // exempt and one in 40 has no bar.
        let rows: Vec<AuditRow> = (0..30_000u32)
            .map(|i| {
                let mut row = row(i % 6 == 0, next() % 50 == 0, (next() % 1000) as f32 * 1e-3);
                row.bar_known = next() % 40 != 0;
                // Entrants are mostly creatures above the bar. A creature
                // below it has a 1 in 250 chance to enter in the first two
                // generations of every five, and 1 in 3,000 in the others.
                let weak = if g % 5 < 2 { 250 } else { 3000 };
                row.entrant = if row.pass3 {
                    next() % 4 == 0
                } else {
                    next() % weak == 0
                };
                row.below_bar = !row.pass3 && next() % 3 != 0;
                row
            })
            .collect();
        #[allow(clippy::needless_range_loop)]
        for r in 0..RUNGS {
            // The rule fitted before this generation judges its rows.
            let Some(rung) = audit.fit_rung(r) else {
                history[r].clear();
                continue;
            };
            let (mut n, mut stopped) = (0u32, 0u32);
            for row in rows.iter().filter(|row| {
                !row.skips(r) && row.bar_known && row.entrant && !row.below_bar && row.alive(r)
            }) {
                let Some(f) = row.features(r) else { continue };
                n += 1;
                stopped += u32::from(rung.raw_stops(&f));
            }
            history[r].push((n, stopped));
            if history[r].len() > JUDGED {
                history[r].remove(0);
            }
        }
        for row in rows {
            audit.record(row);
        }
        audit.boundary(None, false);
        #[allow(clippy::needless_range_loop)]
        for r in 0..RUNGS {
            let (n, stopped) = history[r]
                .iter()
                .fold((0u32, 0u32), |t, &(n, s)| (t.0 + n, t.1 + s));
            let trusted = n >= 60 && f64::from(stopped) <= 0.03 * f64::from(n);
            seen[usize::from(trusted)] = true;
            assert_eq!(
                audit.trusted(r, false),
                trusted,
                "generation {g}, rung {r}: {n} entrants, {stopped} stopped"
            );
        }
    }
    // The test only means something if the guard trusted a rung at least once
    // and refused one at least once.
    assert!(seen[0] && seen[1], "{seen:?}");
}

#[test]
fn a_save_from_before_the_packing_reads_as_the_entrant_counts_it_held() {
    // The two words of a saved generation were (entrants, stopped).
    let old = Judged::unpack((57, 21));
    assert_eq!(old.entrants, (57, 21));
    assert_eq!(old.passers, (0, 0));
}
