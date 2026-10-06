use super::*;

/// A trace of a creature alive past both early rungs with the given
/// features, built from the packed words the kernel writes.
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
    // A function of seed, round and slot only.
    assert_eq!(is_audit(1, 2, 3), is_audit(1, 2, 3));
    let moved = (0..4096)
        .filter(|&s| is_audit(1, 2, s) != is_audit(1, 3, s))
        .count();
    assert!(moved > 0);
}

#[test]
fn a_rule_stops_below_its_bias_and_not_in_an_off_band() {
    let mut rung = Rung::NEVER;
    rung.weights[0] = 1.0;
    rung.bias = 1.0;
    let low = [0.5, 0.0, 0.0, 0.0, 0.0, 0.2];
    let high = [2.0, 0.0, 0.0, 0.0, 0.0, 0.2];
    assert!(rung.stops(&low, 3));
    assert!(!rung.stops(&high, 3));
    rung.off = 1 << 3;
    assert!(!rung.stops(&low, 3));
    assert!(rung.stops(&low, 2));
    // A feature that is not a number never stops a creature.
    assert!(!Rung::NEVER.raw_stops(&low));
    let mut bad = low;
    bad[1] = f32::NAN;
    rung.off = 0;
    assert!(!rung.stops(&bad, 0));
}

#[test]
fn the_fit_stops_the_slow_and_spares_the_budget_of_the_fast() {
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
    // The budget is 1 in 1,000 of the fast ones; the sample is 2,000.
    assert!(stopped_fast <= 6, "{stopped_fast}");
}

#[test]
fn nothing_is_armed_without_rows_of_both_classes() {
    let mut audit = Audit::default();
    for i in 0..3000u32 {
        audit.record(row(i % 2 == 0, false, 0.1));
    }
    assert!(audit.boundary(None, false).is_none());
    // Exempt rows (nurseries, immigrants) never enter the fit.
    let mut audit = Audit::default();
    for i in 0..50_000u32 {
        audit.record(row(i % 2 == 0, true, i as f32 * 1e-4));
    }
    assert!(audit.boundary(None, false).is_none());
}

#[test]
fn a_band_turns_off_after_three_bad_generations_and_back_on_after_three_good() {
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
    let mut audit = Audit::default();
    for g in 0..11u32 {
        for i in 0..10_000u32 {
            audit.record(row(i % 5 == 0, false, ((i + g) % 1000) as f32 * 1e-3));
        }
        audit.boundary(None, false);
    }
    assert_eq!(audit.window.len(), WINDOW);
    let bytes = bincode::serialize(&audit).unwrap();
    let back: Audit = bincode::deserialize(&bytes).unwrap();
    assert_eq!(back.fit(false), audit.fit(false));
    assert!(audit.fit(false).is_some());
}

#[test]
fn children_of_a_parent_the_rules_would_stop_skip_that_rung() {
    let mut r2 = Rung::NEVER;
    r2.weights[0] = 1.0;
    r2.bias = 1.0;
    let rules = Rungs([Rung::NEVER, r2]);
    let slow = profile(&trace(0.2, 0.5, 1.0, 1200, [0, 0]), 0.25);
    let fast = profile(&trace(1.0, 3.0, 8.0, 1200, [0, 0]), 0.25);
    assert_eq!(parent_exemptions(&rules, Some(&slow), true), EXEMPT_R2);
    assert_eq!(parent_exemptions(&rules, Some(&fast), true), 0);
    // A weak elite that the rules stop is where the rules should work.
    assert_eq!(parent_exemptions(&rules, Some(&slow), false), 0);
    // An elite with no profile (an old save, a trial with no trace).
    assert_eq!(parent_exemptions(&rules, None, true), EXEMPT_R1 | EXEMPT_R2);
    assert_eq!(
        parent_exemptions(&rules, Some(&[0; 2 * FEATURES]), true),
        EXEMPT_R1 | EXEMPT_R2
    );
}

/// A generation of 40,000 audit rows, one in eight reaching the 5 s bar.
fn generation(audit: &mut Audit, g: u32, plateau: bool, tweak: impl Fn(u32, &mut AuditRow)) {
    for i in 0..40_000u32 {
        let jitter = ((i + g).wrapping_mul(2654435761) >> 16) as f32 / 65536.0;
        let mut row = row(i % 8 == 0, false, jitter);
        tweak(i, &mut row);
        audit.record(row);
    }
    audit.boundary(None, plateau);
}

/// Some creatures fall early with poor features and enter archives: the 5 s
/// screen keeps them, and the fitted rule stops them.
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
    // The guard counts the creatures above the bar there, so the early
    // fallers that improve a niche of weak bodies, or no entrants at all,
    // decide nothing.
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
    // Past 65,535 both counts shrink by the same factor, so the shares stay.
    assert!(back.passers.0 <= 0xffff && back.entrants.0 <= 0xffff);
    assert!((back.passers.1 as f64 / back.passers.0 as f64 - 0.02).abs() < 1e-3);
    assert!((back.entrants.1 as f64 / back.entrants.0 as f64 - 0.02).abs() < 1e-3);
}

#[test]
fn while_the_archives_climb_the_guard_is_the_entrant_guard_it_always_was() {
    // The trust of each generation, from the rule the window fit before it,
    // by the rule of the entrant guard written out again: the entrants the 5 s
    // screen keeps that are alive at the rung and not exempt, 60 or more over
    // the last 4 generations, at most 3% of them stopped.
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
        let rows: Vec<AuditRow> = (0..30_000u32)
            .map(|i| {
                let mut row = row(i % 6 == 0, next() % 50 == 0, (next() % 1000) as f32 * 1e-3);
                row.bar_known = next() % 40 != 0;
                // Entrants are mostly creatures above the bar; creatures that
                // fall early enter now and then, more in some generations.
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
    // The test only means something if the guard both held and let go.
    assert!(seen[0] && seen[1], "{seen:?}");
}

#[test]
fn a_save_from_before_the_packing_reads_as_the_entrant_counts_it_held() {
    // The two words of a saved generation were (entrants, stopped).
    let old = Judged::unpack((57, 21));
    assert_eq!(old.entrants, (57, 21));
    assert_eq!(old.passers, (0, 0));
}
