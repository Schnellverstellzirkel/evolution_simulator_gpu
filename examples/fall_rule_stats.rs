//! How the fall rule (the head, node 0, dropping below its neck base, bone
//! 0's child) treats a random population: bodies that start with the head at
//! or below the neck base, bodies whose head is not the highest node, and how
//! many fall within a few seconds. Diagnostic only.
//! With a checkpoint as the third argument it also describes the archive's
//! elites: how far the head starts above its neck base and how tall each body is.
//! Usage: cargo run --release --example fall_rule_stats [count] [seconds] [checkpoint.evo]
use evolution_simulator::{config::Config, cpu_engine, evolution};

fn main() {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);
    let seconds: f32 = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(5.0);
    let cfg = Config {
        population: count,
        duration: seconds,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let pop = evolution::create(&cfg).unwrap();
    let results = cpu_engine::evaluate(&pop, &cfg);
    let (mut head_low, mut head_not_top, mut head_level) = (0, 0, 0);
    let (mut fell_start, mut fell_late, mut stood) = (0, 0, 0);
    let (mut low_fell, mut top_fell, mut low_n, mut top_n) = (0, 0, 0, 0);
    for (i, r) in results.iter().enumerate() {
        let c = pop.creature(i);
        let neck = c.bones[0].b as usize;
        let head = c.nodes[0].y;
        let base = c.nodes[neck].y;
        let highest = c.nodes.iter().map(|n| n.y).fold(f32::MIN, f32::max);
        let below = head < base;
        head_low += usize::from(below);
        head_level += usize::from((head - base).abs() < 0.02);
        head_not_top += usize::from(head + 0.02 < highest);
        let fell = r.fall_time > 0.0;
        if fell && r.fall_time < 1.0 {
            fell_start += 1;
        } else if fell {
            fell_late += 1;
        } else {
            stood += 1;
        }
        if head + 0.02 < highest {
            low_n += 1;
            low_fell += usize::from(fell);
        } else {
            top_n += 1;
            top_fell += usize::from(fell);
        }
    }
    let pct = |n: usize| 100.0 * n as f32 / count as f32;
    println!("{count} random bodies, {seconds} s trials");
    println!(
        "start pose: head below neck base {:.1}%, head level with it (2 cm) {:.1}%, head more than 2 cm under the highest node {:.1}%",
        pct(head_low),
        pct(head_level),
        pct(head_not_top)
    );
    println!(
        "outcome: fell in the first second {:.1}%, fell later {:.1}%, never fell {:.1}%",
        pct(fell_start),
        pct(fell_late),
        pct(stood)
    );
    println!(
        "fell, by whether the head is the top node: head on top {:.1}% of {top_n}, head not on top {:.1}% of {low_n}",
        100.0 * top_fell as f32 / top_n.max(1) as f32,
        100.0 * low_fell as f32 / low_n.max(1) as f32
    );
    if let Some(path) = std::env::args().nth(3) {
        let e = evolution_simulator::storage::load(std::path::Path::new(&path)).unwrap();
        let mut rise = Vec::new();
        let (mut flat, mut level, mut short_neck, mut low_head) = (0usize, 0usize, 0usize, 0usize);
        let total = e.archive.entries.len();
        for elite in &e.archive.entries {
            let c = &elite.creature;
            let neck = c.bones[0].b as usize;
            let dy = c.nodes[0].y - c.nodes[neck].y;
            let ys = c.nodes.iter().map(|n| n.y);
            let (lo, hi) = (
                ys.clone().fold(f32::MAX, f32::min),
                ys.fold(f32::MIN, f32::max),
            );
            rise.push(dy);
            flat += usize::from(hi - lo < 0.3);
            level += usize::from(dy < 0.05);
            short_neck += usize::from(c.bones[0].rest_length < 0.15);
            low_head += usize::from(c.nodes[0].y + 0.05 < hi);
        }
        rise.sort_by(f32::total_cmp);
        let pct = |n: usize| 100.0 * n as f32 / total.max(1) as f32;
        println!(
            "{total} elites: head above neck base at the start, median {:.2} m, p10 {:.2} m; under 5 cm {:.1}%; body under 0.3 m tall {:.1}%; neck bone under 15 cm {:.1}%; head more than 5 cm under the highest node {:.1}%",
            rise[total / 2],
            rise[total / 10],
            pct(level),
            pct(flat),
            pct(short_neck),
            pct(low_head)
        );
    }
}
