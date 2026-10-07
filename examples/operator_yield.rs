//! What each structural operator yields in a real generation. The tool reads
//! a generation dump (`EVOLUTION_DUMP_GENERATION`, see `storage::dump`) and
//! prints, per operator, how many children it made and how many of them
//! entered an archive: any archive, an island archive or the global archive,
//! each with its share of the operator's children. The last column is the
//! distance its children added per 1,000 children, and the rows are sorted by
//! it. Rows for groups of children come last, for comparison: the 13 compound
//! operators, all other structural operators, the CMA emitter, the structural
//! and novelty emitters when no operator changed the child, and new random
//! bodies.
//!
//! Only a child that entered an island archive or a nursery adds distance. It
//! adds how much faster it is than the elite that held its cell when the
//! generation began, or its whole distance when the cell was empty.
//!
//! Usage: operator_yield <dump.bin>
//! A dump of a resumed run:
//!   EVOLUTION_DUMP_GENERATION=<generation>:<path> search_ab <tag> 4 300000 20 38 --load <save>
//! The first generation after the load cannot be dumped, and the run needs two
//! generations beyond the dumped one to absorb its blocks.
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
        .expect("usage: operator_yield <dump.bin>");
    let d = read(&path)?;
    let names = evolution_simulator::evolution::structural_operator_names();
    let children: Vec<&Row> = d.rows.iter().filter(|r| r.child()).collect();
    // The fastest elite of each cell of each archive when the generation
    // began, keyed by (archive, cell). Reserve elites hold no cell.
    let mut occupant: std::collections::HashMap<(u16, u16), f32> = std::collections::HashMap::new();
    for e in &d.elites {
        if e.reserve() || e.cell == u16::MAX {
            continue;
        }
        let slot = occupant.entry((e.arena, e.cell)).or_insert(f32::MIN);
        *slot = slot.max(e.fitness);
    }
    // The distance a child added: how much faster it is than the elite whose
    // cell it entered (its own distance when the cell was empty). A child that
    // entered no island archive and no nursery, or has no cell, added nothing.
    let gain = |r: &Row| -> f64 {
        if r.entered & (ISLAND | NURSERY) == 0 || r.cell == u16::MAX {
            return 0.0;
        }
        let arena = r.arena(&d.header) as u16;
        let held = occupant.get(&(arena, r.cell)).copied().unwrap_or(0.0);
        let score = if r.score.is_finite() {
            r.score
        } else {
            r.fitness
        };
        f64::from((score - held.max(0.0)).max(0.0))
    };
    let structural: Vec<&&Row> = children
        .iter()
        .filter(|r| r.operator != u16::MAX && r.emitter != RESTART)
        .collect();
    println!(
        "dump {path}: generation {}, {} children, {} made by a structural operator",
        d.header.generation,
        children.len(),
        structural.len()
    );
    // operator -> ([children, entered any, entered an island, entered the
    // global archive], distance added)
    let mut table: HashMap<u16, ([usize; 4], f64)> = HashMap::new();
    for r in &structural {
        let t = table.entry(r.operator).or_default();
        t.0[0] += 1;
        t.0[1] += usize::from(r.entered != 0);
        t.0[2] += usize::from(r.entered & ISLAND != 0);
        t.0[3] += usize::from(r.entered & GLOBAL != 0);
        t.1 += gain(r);
    }
    // The most distance added per child first.
    let mut rows: Vec<(u16, ([usize; 4], f64))> = table.into_iter().collect();
    rows.sort_by(|a, b| {
        let rate = |t: &([usize; 4], f64)| t.1 / t.0[0].max(1) as f64;
        rate(&b.1).total_cmp(&rate(&a.1))
    });
    println!("| operator | children | entered any | island | global | distance added per 1k |");
    println!("|---|---:|---:|---:|---:|---:|");
    for (o, (t, added)) in rows {
        // An operator index that this build does not list prints as #index.
        let name = names
            .get(o as usize)
            .map_or(format!("#{o}"), |n| n.to_string());
        println!(
            "| {name} | {} | {} ({:.2}%) | {} ({:.2}%) | {} ({:.2}%) | {:.1} |",
            t[0],
            t[1],
            pct(t[1], t[0]),
            t[2],
            pct(t[2], t[0]),
            t[3],
            pct(t[3], t[0]),
            1000.0 * added / t[0].max(1) as f64,
        );
    }
    // The 13 operators of `src/evolution/anatomy/compound.rs`, as one group
    // beside all the other structural operators. Breeding also treats the four
    // leg operators and the gait and idea operators as compound
    // (`anatomy::is_compound`), but this group leaves them out.
    const COMPOUND: [&str; 13] = [
        "limb_length_gradient",
        "symmetrize_limb_pair",
        "retime_gait_by_position",
        "brace_limb_chain",
        "phase_cluster_move",
        "grow_integrated_limb",
        "mirrored_limb_pair",
        "segment_chain",
        "reassign_bundle",
        "transplant_limb_program",
        "retune_limb_package",
        "transplant_gait",
        "trim_body",
    ];
    let compound = |o: u16| names.get(o as usize).is_some_and(|n| COMPOUND.contains(n));
    let all = |f: &dyn Fn(&Row) -> bool| -> ([usize; 4], f64) {
        let mut t = [0usize; 4];
        let mut added = 0.0;
        for r in children.iter().filter(|r| f(r)) {
            t[0] += 1;
            t[1] += usize::from(r.entered != 0);
            t[2] += usize::from(r.entered & ISLAND != 0);
            t[3] += usize::from(r.entered & GLOBAL != 0);
            added += gain(r);
        }
        (t, added)
    };
    for (label, t) in [
        (
            "all compound operators",
            all(&|r| r.operator != u16::MAX && r.emitter != RESTART && compound(r.operator)),
        ),
        (
            "all other structural operators",
            all(&|r| r.operator != u16::MAX && r.emitter != RESTART && !compound(r.operator)),
        ),
        ("cma emitter", all(&|r| r.emitter == 0)),
        (
            "structural or novelty, no operator",
            all(&|r| r.operator == u16::MAX && (r.emitter == 1 || r.emitter == 2)),
        ),
        ("random bodies", all(&|r| r.emitter == RESTART)),
    ] {
        let (t, added) = t;
        println!(
            "| {label} | {} | {} ({:.2}%) | {} ({:.2}%) | {} ({:.2}%) | {:.1} |",
            t[0],
            t[1],
            pct(t[1], t[0]),
            t[2],
            pct(t[2], t[0]),
            t[3],
            pct(t[3], t[0]),
            1000.0 * added / t[0].max(1) as f64,
        );
    }
    Ok(())
}
