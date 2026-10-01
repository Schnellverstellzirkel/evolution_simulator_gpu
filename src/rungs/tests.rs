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
        for i in 0..4000u32 {
            k += 1;
            let jitter = (k.wrapping_mul(2654435761) >> 16) as f32 / 65536.0;
            audit.record(row(i % 8 == 0, false, jitter));
        }
        assert!(audit.boundary(None).is_some() || audit.window_rows() < 8000);
    }
    let rules = audit.fit().expect("enough rows");
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
    for i in 0..300u32 {
        audit.record(row(i % 2 == 0, false, 0.1));
    }
    assert!(audit.boundary(None).is_none());
    // Exempt rows (nurseries, immigrants) never enter the fit.
    let mut audit = Audit::default();
    for i in 0..5000u32 {
        audit.record(row(i % 2 == 0, true, i as f32 * 1e-4));
    }
    assert!(audit.boundary(None).is_none());
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
        for i in 0..1000u32 {
            audit.record(row(i % 5 == 0, false, (i + g) as f32 * 1e-3));
        }
        audit.boundary(None);
    }
    assert_eq!(audit.window.len(), WINDOW);
    let bytes = bincode::serialize(&audit).unwrap();
    let back: Audit = bincode::deserialize(&bytes).unwrap();
    assert_eq!(back.fit(), audit.fit());
    assert!(audit.fit().is_some());
}
