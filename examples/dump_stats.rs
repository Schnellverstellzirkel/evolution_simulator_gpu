//! Reads a generation dump (EVOLUTION_DUMP_GENERATION, see `storage::dump`)
//! and prints what the search needs from it: the ring's body histogram, entrants by archive, the
//! three tail numbers per body class (R1 stop share, entrant share by
//! archive including the reserve, operator histogram), the nursery's
//! entrants, entrant recall per per-cell factor at 5 s, and mean final /
//! d(10) of entrants.
//!
//! Usage: dump_stats <dump.bin>
#[path = "dump_common/mod.rs"]
mod dump_common;
use dump_common::*;
use std::collections::HashMap;

fn pct(part: usize, whole: usize) -> f64 {
    100.0 * part as f64 / whole.max(1) as f64
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_stats <dump.bin>");
    let d = read(&path)?;
    let h = &d.header;
    let names = evolution_simulator::evolution::structural_operator_names();
    let children: Vec<&Row> = d.rows.iter().filter(|r| r.child()).collect();
    let reruns = d.rows.iter().filter(|r| r.flags & RERUN != 0).count();
    let stale = d
        .rows
        .iter()
        .filter(|r| r.flags & EXCLUDED != 0 && r.flags & RERUN == 0)
        .count();
    println!(
        "dump {path}: generation {}, seed {}, population {}, qd version {}, {:.0} s trials at {} Hz",
        h.generation, h.seed, h.population, h.qd_version, h.duration, h.rate
    );
    println!(
        "rows {} ({} children, {} elite re-runs, {} from a changed world), elites {} ({} re-run), screen bar {:.3} m",
        d.rows.len(),
        children.len(),
        reruns,
        stale,
        d.elites.len(),
        d.elites.iter().filter(|e| e.flags & 8 != 0).count(),
        h.bar
    );

    // The ring histogram.
    let hist = |label: &str, values: &mut Vec<u8>| {
        values.sort_unstable();
        let n = values.len().max(1);
        let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let q = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
        println!(
            "ring {label}: mean {mean:.2}, p50 {}, p90 {}, p99 {}, max {}",
            q(0.5),
            q(0.9),
            q(0.99),
            values.last().unwrap()
        );
    };
    hist("nodes", &mut children.iter().map(|r| r.nodes).collect());
    hist("muscles", &mut children.iter().map(|r| r.muscles).collect());
    let mut class_count = [0usize; 3];
    for r in &children {
        class_count[size_class(r.nodes)] += 1;
    }
    println!(
        "ring classes: {}",
        (0..3)
            .map(|c| format!(
                "{} {:.2}%",
                SIZE_CLASSES[c],
                pct(class_count[c], children.len())
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let by_emitter: Vec<String> = (0..4)
        .map(|e| {
            let n = children.iter().filter(|r| r.emitter == e as u8).count();
            format!("{} {:.1}%", EMITTERS[e], pct(n, children.len()))
        })
        .collect();
    println!("emitters: {}", by_emitter.join(", "));

    // Steps per creature.
    let all: f64 = children.iter().map(|r| r.steps() as f64).sum::<f64>() / children.len() as f64;
    let today: f64 = children
        .iter()
        .map(|r| today_steps(r, h) as f64)
        .sum::<f64>()
        / children.len() as f64;
    let fell = children.iter().filter(|r| r.trace.fell()).count();
    println!(
        "steps per creature: {all:.0} with every trial in full, {today:.0} under today's 5 s screen; {:.1}% fell",
        pct(fell, children.len())
    );

    // Entrants by archive.
    let entered = |mask: u8| children.iter().filter(|r| r.entered & mask != 0).count();
    let any = entered(0xff);
    println!(
        "entrants: {any} ({:.3}% of children): island {}, nursery {}, reserve {}, global {}",
        pct(any, children.len()),
        entered(ISLAND),
        entered(NURSERY),
        entered(RESERVE),
        entered(GLOBAL)
    );
    let screened_entrants = children
        .iter()
        .filter(|r| r.entered != 0 && today_steps(r, h) == RUNG_STEPS[2] && r.steps() > 300)
        .count();
    println!(
        "entrants below today's 5 s bar (today's screen would have stopped them): {screened_entrants} ({:.2}% of entrants)",
        pct(screened_entrants, any)
    );

    // R1 at the plan's budget, fitted on half the rows, for the tail number.
    let fit: Vec<&Row> = children.iter().copied().filter(|r| fit_half(r)).collect();
    let ladder = Ladder::fit(&fit, &d, 1e-3, false, false);

    // The three tail numbers per body class.
    println!("tail numbers by body class (R1 fitted on half the rows at 1e-3, today's R3):");
    for class in 0..3 {
        let rows: Vec<&Row> = children
            .iter()
            .copied()
            .filter(|r| size_class(r.nodes) == class)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let at60 = rows.iter().filter(|r| r.steps() > 60).count();
        let r1 = rows
            .iter()
            .filter(|r| ladder.apply(r, h) == Fate::Stopped(0))
            .count();
        let exempt = rows
            .iter()
            .filter(|r| r.nursery(h) || r.emitter == RESTART)
            .count();
        let pass300 = rows
            .iter()
            .filter(|r| r.steps() > 300 && r.d(2) >= h.bar)
            .count();
        println!(
            "  {}: {} children ({:.2}%), alive at 1 s {:.1}%, R1 stops {:.1}% ({:.1}% of those alive; {:.1}% exempt), pass the 5 s bar {:.2}%",
            SIZE_CLASSES[class],
            rows.len(),
            pct(rows.len(), children.len()),
            pct(at60, rows.len()),
            pct(r1, rows.len()),
            pct(r1, at60),
            pct(exempt, rows.len()),
            pct(pass300, rows.len())
        );
        let e = |mask: u8| rows.iter().filter(|r| r.entered & mask != 0).count();
        println!(
            "    entered: any {} ({:.3}%), island {} ({:.3}%), nursery {} ({:.3}%), reserve {} ({:.3}%), global {} ({:.3}%)",
            e(0xff),
            pct(e(0xff), rows.len()),
            e(ISLAND),
            pct(e(ISLAND), rows.len()),
            e(NURSERY),
            pct(e(NURSERY), rows.len()),
            e(RESERVE),
            pct(e(RESERVE), rows.len()),
            e(GLOBAL),
            pct(e(GLOBAL), rows.len())
        );
        let mut ops: HashMap<u8, (usize, usize)> = HashMap::new();
        for r in &rows {
            let o = ops.entry(r.operator).or_default();
            o.0 += 1;
            o.1 += usize::from(r.entered != 0);
        }
        let mut ops: Vec<(u8, (usize, usize))> = ops.into_iter().collect();
        ops.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.0.cmp(&b.0)));
        let label = |o: u8| {
            if o == u8::MAX {
                "none".to_owned()
            } else {
                names
                    .get(o as usize)
                    .map_or(format!("#{o}"), |n| n.to_string())
            }
        };
        println!(
            "    operators: {}",
            ops.iter()
                .take(12)
                .map(|(o, (n, e))| format!(
                    "{} {:.1}% ({e} entered)",
                    label(*o),
                    pct(*n, rows.len())
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let known: Vec<&&Row> = rows.iter().filter(|r| r.parent_nodes > 0).collect();
        let gained = known
            .iter()
            .filter(|r| {
                r.nodes as i32 - r.parent_nodes as i32 > 4
                    || r.muscles as i32 - r.parent_muscles as i32 > 4
            })
            .count();
        let big_parent = known.iter().filter(|r| r.parent_nodes > 13).count();
        println!(
            "    growth: {:.1}% gained more than 4 nodes or 4 muscles over the parent, {:.1}% have a parent above 13 nodes ({} with a known parent)",
            pct(gained, known.len()),
            pct(big_parent, known.len()),
            known.len()
        );
    }
    let entrants: Vec<&&Row> = children.iter().filter(|r| r.entered != 0).collect();
    let jumped = entrants
        .iter()
        .filter(|r| r.parent_nodes > 0 && r.nodes as i32 - r.parent_nodes as i32 > 4)
        .count();
    let jumped_m = entrants
        .iter()
        .filter(|r| r.parent_nodes > 0 && r.muscles as i32 - r.parent_muscles as i32 > 4)
        .count();
    println!(
        "entrants born more than 4 nodes above their parent: {jumped}; more than 4 muscles above: {jumped_m} (of {})",
        entrants.len()
    );

    // The nursery.
    let nursery: Vec<&Row> = children.iter().copied().filter(|r| r.nursery(h)).collect();
    let nursery_entered = nursery.iter().filter(|r| r.entered & NURSERY != 0).count();
    let nursery_passed = nursery
        .iter()
        .filter(|r| r.steps() > 300 && r.d(2) >= h.bar)
        .count();
    println!(
        "nursery: {} children, {} entered their nursery ({:.3}%), {:.2}% pass the island's 5 s bar",
        nursery.len(),
        nursery_entered,
        pct(nursery_entered, nursery.len()),
        pct(nursery_passed, nursery.len())
    );
    let r1_nursery = nursery
        .iter()
        .filter(|r| r.entered & NURSERY != 0 && r.steps() > 60)
        .filter(|r| {
            let f = Ladder::features1(r);
            f.iter().all(|v| v.is_finite()) && dot(&ladder.w1, &f) < ladder.b1
        })
        .count();
    println!(
        "nursery entrants the island R1 would stop if nurseries were not exempt: {r1_nursery} of {nursery_entered}"
    );

    // Entrant recall per per-cell factor at 5 s (the neighbourhood minimum
    // over the parent's cell and its 80 neighbours of the elites' d300).
    let nmin300 = neighbourhood_min(&d, |e| e.d[1]);
    let mut top: Vec<&Row> = children.clone();
    top.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let top1: std::collections::HashSet<u32> = top[..(top.len() / 100).max(1)]
        .iter()
        .map(|r| r.slot)
        .collect();
    let with_min: Vec<(&Row, f32)> = entrants
        .iter()
        .filter(|r| !r.nursery(h) && r.emitter != RESTART && r.parent_cell != u16::MAX)
        .filter_map(|r| {
            let m = nmin300
                .get(r.arena(h))?
                .get(r.parent_cell as usize)
                .copied()?;
            (m.is_finite() && m > 0.0).then_some((**r, m))
        })
        .collect();
    println!(
        "entrant recall at 5 s against factor x the parent neighbourhood's minimum ({} island entrants whose neighbourhood minimum is above 0):",
        with_min.len()
    );
    for factor in [0.3f32, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        let below: Vec<&(&Row, f32)> = with_min
            .iter()
            .filter(|(r, m)| r.d(2) < factor * m)
            .collect();
        let top_below = below.iter().filter(|(r, _)| top1.contains(&r.slot)).count();
        println!(
            "  factor {factor:.1}: {:.2}% of entrants below it, {top_below} of them in the top 1%",
            pct(below.len(), with_min.len())
        );
    }
    let ratios: Vec<f64> = entrants
        .iter()
        .filter(|r| r.steps() > 600 && r.d(3) > 0.05)
        .map(|r| (r.fitness / r.d(3)) as f64)
        .collect();
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "entrants' final / d(10 s): mean {:.3}, median {:.3} ({} entrants)",
        ratios.iter().sum::<f64>() / ratios.len().max(1) as f64,
        sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN),
        ratios.len()
    );
    println!(
        "cell match at 5, 10 and 20 s: not in this dump (the kernel bins a cell only at the end; R4's in-kernel binning at 10 s would add it)"
    );
    Ok(())
}
