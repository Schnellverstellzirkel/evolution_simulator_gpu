//! What each structural operator yields in a real generation: from a
//! generation dump (EVOLUTION_DUMP_GENERATION, see `storage::dump`) it prints,
//! per operator, how many children it made, how many entered an archive (any
//! archive, an island archive, the global archive) and what share of the
//! operator's children that is. Children of the CMA emitter, children of the
//! structural and novelty emitters that no operator changed, and new random
//! bodies come last, for comparison.
//!
//! Usage: operator_yield <dump.bin>
//! A dump of a resumed run:
//!   EVOLUTION_DUMP_GENERATION=<generation>:<path> search_ab <tag> 2 300000 20 38 --resume <save>
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
    let structural: Vec<&&Row> = children
        .iter()
        .filter(|r| r.operator != u8::MAX && r.emitter != RESTART)
        .collect();
    println!(
        "dump {path}: generation {}, {} children, {} made by a structural operator",
        d.header.generation,
        children.len(),
        structural.len()
    );
    // operator -> (children, any, island, global, children that grew a node)
    let mut table: HashMap<u8, [usize; 4]> = HashMap::new();
    for r in &structural {
        let t = table.entry(r.operator).or_default();
        t[0] += 1;
        t[1] += usize::from(r.entered != 0);
        t[2] += usize::from(r.entered & ISLAND != 0);
        t[3] += usize::from(r.entered & GLOBAL != 0);
    }
    let mut rows: Vec<(u8, [usize; 4])> = table.into_iter().collect();
    rows.sort_by(|a, b| {
        let rate = |t: &[usize; 4]| t[1] as f64 / t[0].max(1) as f64;
        rate(&b.1).total_cmp(&rate(&a.1))
    });
    println!("| operator | children | entered any | island | global |");
    println!("|---|---:|---:|---:|---:|");
    for (o, t) in rows {
        let name = names
            .get(o as usize)
            .map_or(format!("#{o}"), |n| n.to_string());
        println!(
            "| {name} | {} | {} ({:.2}%) | {} ({:.2}%) | {} ({:.2}%) |",
            t[0],
            t[1],
            pct(t[1], t[0]),
            t[2],
            pct(t[2], t[0]),
            t[3],
            pct(t[3], t[0]),
        );
    }
    let all = |f: &dyn Fn(&Row) -> bool| -> [usize; 4] {
        let mut t = [0usize; 4];
        for r in children.iter().filter(|r| f(r)) {
            t[0] += 1;
            t[1] += usize::from(r.entered != 0);
            t[2] += usize::from(r.entered & ISLAND != 0);
            t[3] += usize::from(r.entered & GLOBAL != 0);
        }
        t
    };
    for (label, t) in [
        ("cma emitter", all(&|r| r.emitter == 0)),
        (
            "structural or novelty, no operator",
            all(&|r| r.operator == u8::MAX && (r.emitter == 1 || r.emitter == 2)),
        ),
        ("random bodies", all(&|r| r.emitter == RESTART)),
    ] {
        println!(
            "| {label} | {} | {} ({:.2}%) | {} ({:.2}%) | {} ({:.2}%) |",
            t[0],
            t[1],
            pct(t[1], t[0]),
            t[2],
            pct(t[2], t[0]),
            t[3],
            pct(t[3], t[0]),
        );
    }
    Ok(())
}
