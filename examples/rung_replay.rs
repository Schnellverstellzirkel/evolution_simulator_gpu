//! Replays a candidate rung ladder (R1 at 1 s, R2 at 2.5 s,
//! R3 at 5 s, R4 at 10 s, the 1-in-128 audit lane, nurseries and immigrants
//! exempt from R1 and R2, optimizer children exempt from R4) on a
//! generation dump (EVOLUTION_DUMP_GENERATION), where every trial ran in
//! full. The ladder is fitted on half the rows (by slot) and measured on the
//! other half. It prints steps per creature, the stop share per rung,
//! entrant and pass-R3 misses per 10k creatures per rung and per cadence
//! band, the share of the final top 1% and 10% kept, R4's fire rate, and
//! Spearman of d(10 s) against the final distance.
//!
//! R1 and R2 are Fisher discriminants on the kernel's features (distance,
//! speed over the last half second, share of nodes that touched, head
//! shake, mean muscle energy, rhythm period; R2 adds the parent
//! neighbourhood's 2.5 s bar) with the threshold at the tolerance of the
//! rows that pass R3. R3 is today's 5 s bar or the per-cell bar (factor x
//! the parent neighbourhood's minimum of the elites' 5 s distance). R4 uses
//! the creature's final cell for its 10 s cell, since the kernel does not
//! bin at 10 s yet.
//!
//! Usage: rung_replay <dump.bin> [--tolerance 1e-3]
#[path = "dump_common/mod.rs"]
mod dump_common;
use dump_common::*;
use evolution_simulator::rungs;

struct Outcome {
    label: String,
    steps: f64,
    stops: [f64; 4],
    entrant_misses: [f64; 4],
    pass3_misses: [f64; 2],
    top1: f64,
    top10: f64,
    r4_fire: f64,
}

fn evaluate(label: &str, ladder: Option<&Ladder>, rows: &[&Row], d: &DumpFile) -> Outcome {
    let h = &d.header;
    let n = rows.len().max(1) as f64;
    let mut steps = 0.0;
    let mut stops = [0usize; 4];
    let mut entrant_misses = [0usize; 4];
    let mut pass3_misses = [0usize; 2];
    let mut reached600 = 0usize;
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| rows[b].fitness.total_cmp(&rows[a].fitness));
    let mut kept = vec![true; rows.len()];
    for (i, r) in rows.iter().enumerate() {
        let fate = match ladder {
            Some(l) => l.apply(r, h),
            None => {
                if today_steps(r, h) < r.steps() {
                    Fate::Stopped(2)
                } else {
                    Fate::Ran
                }
            }
        };
        steps += Ladder::steps(fate, r) as f64;
        let alive600 = r.steps() > RUNG_STEPS[3] && !matches!(fate, Fate::Stopped(k) if k < 3);
        reached600 += usize::from(alive600);
        if let Fate::Stopped(k) = fate {
            stops[k] += 1;
            kept[i] = false;
            if r.entered != 0 {
                entrant_misses[k] += 1;
            }
            if k < 2 && ladder.is_some_and(|l| l.pass3(r, h)) {
                pass3_misses[k] += 1;
            }
        }
    }
    let share_kept = |count: usize| {
        let count = count.max(1).min(order.len());
        order[..count].iter().filter(|&&i| kept[i]).count() as f64 / count as f64
    };
    Outcome {
        label: label.to_owned(),
        steps: steps / n,
        stops: stops.map(|s| 100.0 * s as f64 / n),
        entrant_misses: entrant_misses.map(|s| 1e4 * s as f64 / n),
        pass3_misses: pass3_misses.map(|s| 1e4 * s as f64 / n),
        top1: 100.0 * share_kept(order.len() / 100),
        top10: 100.0 * share_kept(order.len() / 10),
        r4_fire: 100.0 * stops[3] as f64 / reached600.max(1) as f64,
    }
}

/// The game's own rules (`rungs`), fitted by the audit lane on the dump's 1 in
/// 128 audit rows alone and applied to every other row with today's 5 s bar.
fn evaluate_live(label: &str, rules: &rungs::Rungs, rows: &[&Row], d: &DumpFile) -> Outcome {
    let h = &d.header;
    let n = rows.len().max(1) as f64;
    let mut steps = 0.0;
    let mut stops = [0usize; 4];
    let mut entrant_misses = [0usize; 4];
    let mut pass3_misses = [0usize; 2];
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| rows[b].fitness.total_cmp(&rows[a].fitness));
    let mut kept = vec![true; rows.len()];
    for (i, r) in rows.iter().enumerate() {
        let exempt = r.nursery(h) || r.emitter == RESTART;
        let mut fate = Fate::Ran;
        if !r.audit() {
            for k in 0..2 {
                if !exempt
                    && r.steps() > RUNG_STEPS[k]
                    && rules.0[k].stops(&rungs::features(&r.trace, k, r.period), 0)
                {
                    fate = Fate::Stopped(k);
                    break;
                }
            }
            if fate == Fate::Ran && today_steps(r, h) < r.steps() {
                fate = Fate::Stopped(2);
            }
        }
        steps += Ladder::steps(fate, r) as f64;
        if let Fate::Stopped(k) = fate {
            stops[k] += 1;
            kept[i] = false;
            if r.entered != 0 {
                entrant_misses[k] += 1;
            }
            if k < 2 && r.d(2) >= h.bar {
                pass3_misses[k] += 1;
            }
        }
    }
    let share_kept = |count: usize| {
        let count = count.max(1).min(order.len());
        order[..count].iter().filter(|&&i| kept[i]).count() as f64 / count as f64
    };
    Outcome {
        label: label.to_owned(),
        steps: steps / n,
        stops: stops.map(|s| 100.0 * s as f64 / n),
        entrant_misses: entrant_misses.map(|s| 1e4 * s as f64 / n),
        pass3_misses: pass3_misses.map(|s| 1e4 * s as f64 / n),
        top1: 100.0 * share_kept(order.len() / 100),
        top10: 100.0 * share_kept(order.len() / 10),
        r4_fire: 0.0,
    }
}

fn print_header() {
    println!(
        "{:<34} {:>7} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} | {:>7} {:>7} {:>7}",
        "ladder",
        "steps",
        "R1%",
        "R2%",
        "R3%",
        "R4%",
        "mis1",
        "mis2",
        "mis3",
        "mis4",
        "p3m1",
        "p3m2",
        "top1%",
        "top10%",
        "R4fire%"
    );
}
fn print(o: &Outcome) {
    println!(
        "{:<34} {:>7.1} {:>6.2} {:>6.2} {:>6.2} {:>6.2} | {:>6.1} {:>6.1} {:>6.1} {:>6.1} | {:>6.1} {:>6.1} | {:>7.2} {:>7.2} {:>7.1}",
        o.label,
        o.steps,
        o.stops[0],
        o.stops[1],
        o.stops[2],
        o.stops[3],
        o.entrant_misses[0],
        o.entrant_misses[1],
        o.entrant_misses[2],
        o.entrant_misses[3],
        o.pass3_misses[0],
        o.pass3_misses[1],
        o.top1,
        o.top10,
        o.r4_fire
    );
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: rung_replay <dump.bin> [--tolerance 1e-3]");
    let tol: f64 = args
        .iter()
        .position(|a| a == "--tolerance")
        .and_then(|i| args.get(i + 1))
        .map_or(Ok(1e-3), |v| v.parse())?;
    let d = read(path)?;
    let h = &d.header;
    let children: Vec<&Row> = d.rows.iter().filter(|r| r.child()).collect();
    let fit: Vec<&Row> = children.iter().copied().filter(|r| fit_half(r)).collect();
    let test: Vec<&Row> = children.iter().copied().filter(|r| !fit_half(r)).collect();
    println!(
        "rung_replay {path}: generation {}, {} children, fitted on {} and measured on {}, tolerance {tol:e}",
        h.generation,
        children.len(),
        fit.len(),
        test.len()
    );
    println!(
        "Columns: steps per creature; stop share per rung (% of creatures); entrant misses per rung (per 10k creatures); pass-R3 misses of R1 and R2 (per 10k); share of the final top 1% and 10% kept; R4 stops among creatures alive at 10 s."
    );
    print_header();
    let none: Vec<&Row> = test.clone();
    let all_audit = Outcome {
        label: "every trial in full".into(),
        steps: none.iter().map(|r| r.steps() as f64).sum::<f64>() / none.len().max(1) as f64,
        stops: [0.0; 4],
        entrant_misses: [0.0; 4],
        pass3_misses: [0.0; 2],
        top1: 100.0,
        top10: 100.0,
        r4_fire: 0.0,
    };
    print(&all_audit);
    print(&evaluate("today: 5 s screen", None, &test, &d));
    let mut main_ladder = None;
    for (cell_bars, r4) in [(false, false), (false, true), (true, false), (true, true)] {
        let ladder = Ladder::fit(&fit, &d, tol, cell_bars, r4);
        let label = format!(
            "R1-R{} {}",
            if r4 { 4 } else { 3 },
            if cell_bars { "cell bars" } else { "today's R3" }
        );
        let o = evaluate(&label, Some(&ladder), &test, &d);
        print(&o);
        if cell_bars && r4 {
            println!(
                "  fitted: R3 factor {:.3}, r4 {:.3}, R1 weights {:?} b {:.4}, R2 weights {:?} b {:.4}",
                ladder.f3,
                ladder.r4_ratio,
                ladder
                    .w1
                    .iter()
                    .map(|w| format!("{w:.3e}"))
                    .collect::<Vec<_>>(),
                ladder.b1,
                ladder
                    .w2
                    .iter()
                    .map(|w| format!("{w:.3e}"))
                    .collect::<Vec<_>>(),
                ladder.b2
            );
        }
        if !cell_bars && r4 {
            main_ladder = Some(ladder);
        }
    }
    // The game's rules: the audit lane's rows are the whole fit, and the
    // window holds one generation.
    {
        let mut audit = rungs::Audit::default();
        for r in children.iter().filter(|r| r.audit()) {
            audit.record(rungs::AuditRow {
                trace: r.trace,
                period: r.period,
                exempt: r.nursery(h) || r.emitter == RESTART,
                parent_exempt: 0,
                bar_known: true,
                pass3: r.d(2) >= h.bar,
                below_bar: r.steps() > RUNG_STEPS[2] && r.d(2) < h.bar,
                entrant: r.entered != 0,
            });
        }
        let rules = audit.boundary(None, false);
        println!(
            "the game's rules, fitted on the {} audit rows alone (one generation) and measured on the rows of the other half:",
            children.iter().filter(|r| r.audit()).count()
        );
        print_header();
        match rules {
            Some(rules) => {
                print(&evaluate_live("R1-R3 live fit", &rules, &test, &d));
                // The entrants the 5 s screen would have kept, and how many
                // of them each rung stops.
                for k in 0..2 {
                    let (mut n, mut stopped) = (0usize, 0usize);
                    for r in test.iter().filter(|r| {
                        r.entered != 0
                            && !(r.steps() > RUNG_STEPS[2] && r.d(2) < h.bar)
                            && !r.nursery(h)
                            && r.emitter != RESTART
                            && r.steps() > RUNG_STEPS[k]
                    }) {
                        n += 1;
                        stopped += usize::from(
                            rules.0[k].stops(&rungs::features(&r.trace, k, r.period), 0),
                        );
                    }
                    println!(
                        "  rung {}: {stopped} of {n} entrants the screen keeps would stop ({:.2}%)",
                        k + 1,
                        100.0 * stopped as f64 / n.max(1) as f64
                    );
                }
                println!("  rules {:?}", rules.0);
            }
            None => println!("  no rung armed"),
        }
    }
    // Each rung alone, on top of today's screen (R3), for the gate of a rung
    // that fails its bar alone.
    println!("each rung alone (R3 is today's screen in every row):");
    print_header();
    let base = Ladder::fit(&fit, &d, tol, false, true);
    for (label, keep) in [
        ("R1 + R3", [true, false, true, false]),
        ("R2 + R3", [false, true, true, false]),
        ("R3 + R4", [false, false, true, true]),
    ] {
        let mut l = Ladder::fit(&fit, &d, tol, false, keep[3]);
        if !keep[0] {
            l.b1 = f64::NEG_INFINITY;
        }
        if !keep[1] {
            l.b2 = f64::NEG_INFINITY;
        }
        print(&evaluate(label, Some(&l), &test, &d));
    }
    drop(base);

    // Per cadence band, for the R1-R4 ladder with today's R3.
    let ladder = main_ladder.expect("the main ladder");
    println!(
        "per cadence band of the final cell (R1-R4, today's R3): creatures, steps, entrant misses per rung per 10k of the band, pass-R3 misses of R1 and R2 per 10k"
    );
    for band in 0..h.bins[1] as usize {
        let rows: Vec<&Row> = test
            .iter()
            .copied()
            .filter(|r| cadence_band(r.cell, h.bins) == Some(band))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let o = evaluate("", Some(&ladder), &rows, &d);
        println!(
            "  band {band}: {:>8} creatures, {:>6.1} steps, misses {:.1} {:.1} {:.1} {:.1}, pass-R3 misses {:.1} {:.1}, top 1% kept {:.1}%",
            rows.len(),
            o.steps,
            o.entrant_misses[0],
            o.entrant_misses[1],
            o.entrant_misses[2],
            o.entrant_misses[3],
            o.pass3_misses[0],
            o.pass3_misses[1],
            o.top1
        );
    }
    println!("per arena kind (R1-R4, today's R3):");
    for (label, nursery) in [("islands", false), ("nurseries", true)] {
        let rows: Vec<&Row> = test
            .iter()
            .copied()
            .filter(|r| r.nursery(h) == nursery)
            .collect();
        let o = evaluate(label, Some(&ladder), &rows, &d);
        println!(
            "  {label}: {} creatures, {:.1} steps, stops {:.1}% {:.1}% {:.1}% {:.1}%, entrant misses {:.1} {:.1} {:.1} {:.1} per 10k",
            rows.len(),
            o.steps,
            o.stops[0],
            o.stops[1],
            o.stops[2],
            o.stops[3],
            o.entrant_misses[0],
            o.entrant_misses[1],
            o.entrant_misses[2],
            o.entrant_misses[3]
        );
    }

    // Tolerance sweep.
    println!("tolerance sweep (R1-R4, today's R3):");
    print_header();
    for t in [1e-4, 1e-3, 1e-2] {
        let l = Ladder::fit(&fit, &d, t, false, true);
        print(&evaluate(&format!("tolerance {t:e}"), Some(&l), &test, &d));
    }

    // R4's predictors.
    let survivors: Vec<(f32, f32)> = test
        .iter()
        .filter(|r| r.steps() > RUNG_STEPS[3])
        .map(|r| (r.d(3), r.fitness))
        .collect();
    let past_screen: Vec<(f32, f32)> = test
        .iter()
        .filter(|r| r.steps() > RUNG_STEPS[3] && r.d(2) >= h.bar)
        .map(|r| (r.d(3), r.fitness))
        .collect();
    println!(
        "Spearman of d(10 s) against the final distance: {:.3} over {} creatures alive at 10 s, {:.3} over the {} that passed today's 5 s bar",
        spearman(&survivors),
        survivors.len(),
        spearman(&past_screen),
        past_screen.len()
    );
    let d300: Vec<(f32, f32)> = test
        .iter()
        .filter(|r| r.steps() > RUNG_STEPS[2])
        .map(|r| (r.d(2), r.fitness))
        .collect();
    println!(
        "Spearman of d(5 s) against the final distance: {:.3} over {} creatures alive at 5 s",
        spearman(&d300),
        d300.len()
    );
    Ok(())
}
