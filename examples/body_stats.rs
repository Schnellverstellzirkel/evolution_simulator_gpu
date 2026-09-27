//! Body sizes of a checkpoint's population: how many nodes, bones and muscles
//! each creature has, and what the GPU layouts would cost per creature.
//! It reads only the genome table, so it runs beside the game on a 3M
//! checkpoint (about 200 MB instead of the whole experiment).
//!
//! Usage: cargo run --release --example body_stats -- <checkpoint>
use anyhow::{Context, Result, ensure};
use bincode::Options;
use evolution_simulator::{config::Config, evolution::Genome};
use std::io::Read;

/// How a creature is sorted into a queue class.
type ClassKey = fn(&Genome) -> usize;

#[derive(serde::Deserialize)]
struct Prefix {
    _config: Config,
    _pending: Option<Config>,
    generation: u32,
    genomes: Vec<Genome>,
}

fn percentile(sorted: &[usize], p: f64) -> usize {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: body_stats <checkpoint>")?;
    let mut file = std::io::BufReader::new(std::fs::File::open(&path)?);
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    ensure!(&magic[..6] == b"EVORUS", "not a checkpoint");
    let decoder = zstd::stream::read::Decoder::new(file)?;
    let prefix: Prefix = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(8 * 1024 * 1024 * 1024)
        .deserialize_from(decoder)?;
    let g = &prefix.genomes;
    ensure!(!g.is_empty(), "empty population");
    let n = g.len() as f64;
    println!("generation {}, {} creatures", prefix.generation, g.len());
    for (name, f) in [
        ("nodes", (|x: &Genome| x.node_count) as fn(&Genome) -> usize),
        ("bones", |x| x.bone_count),
        ("muscles", |x| x.muscle_count),
    ] {
        let mut v: Vec<usize> = g.iter().map(f).collect();
        v.sort_unstable();
        let mean = v.iter().sum::<usize>() as f64 / n;
        println!(
            "{name:8} mean {mean:5.2}  p10 {}  p50 {}  p90 {}  p99 {}  max {}",
            percentile(&v, 0.1),
            percentile(&v, 0.5),
            percentile(&v, 0.9),
            percentile(&v, 0.99),
            v[v.len() - 1]
        );
    }
    let mut by_nodes = std::collections::BTreeMap::<usize, (usize, usize, usize)>::new();
    for x in g {
        let e = by_nodes.entry(x.node_count).or_default();
        e.0 += 1;
        e.1 += x.bone_count;
        e.2 += x.muscle_count;
    }
    println!("nodes  share   cumulative  mean bones  mean muscles");
    let mut cumulative = 0.0;
    for (nodes, (count, bones, muscles)) in by_nodes {
        let share = count as f64 / n;
        cumulative += share;
        println!(
            "{nodes:5}  {:5.1}%  {:6.1}%     {:5.2}       {:5.2}",
            100.0 * share,
            100.0 * cumulative,
            bones as f64 / count as f64,
            muscles as f64 / count as f64
        );
    }
    // Today's packed GPU bytes per creature (unpadded): 32 B per node,
    // 60 B per muscle, 36 B per bone, 16 B of info and a 76 B result.
    let today: f64 = g
        .iter()
        .map(|x| 32 * x.node_count + 60 * x.muscle_count + 36 * x.bone_count + 16 + 76)
        .sum::<usize>() as f64
        / n;
    println!("today's packed upload per creature: {today:.0} B (before tile padding)");
    let standard = g
        .iter()
        .filter(|x| x.node_count <= 8 && x.muscle_count <= 20)
        .count();
    println!(
        "bodies with at most 8 nodes and 20 muscles: {:.1}%",
        100.0 * standard as f64 / n
    );

    // Warp loop efficiency: a warp's node and muscle loops run to the largest
    // body among its 32 lanes, so a class that mixes sizes wastes lane slots.
    // Efficiency is the mean count over the mean warp maximum, for warps
    // drawn at random from each class (a queue in birth order).
    let classes: [(&str, ClassKey); 3] = [
        ("one class", |_| 0),
        ("by nodes", |x| x.node_count.min(9)),
        ("by nodes and muscles/4", |x| {
            x.node_count.min(9) * 100 + x.muscle_count.div_ceil(4)
        }),
    ];
    for (name, key) in classes {
        let mut members = std::collections::BTreeMap::<usize, Vec<&Genome>>::new();
        for x in g {
            members.entry(key(x)).or_default().push(x);
        }
        let (mut used, mut slots) = ([0usize; 2], [0usize; 2]);
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        for list in members.values() {
            let warps = list.len().div_ceil(32);
            for _ in 0..warps {
                let mut max = [0usize; 2];
                for _ in 0..32 {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let x = list[(rng % list.len() as u64) as usize];
                    used[0] += x.node_count;
                    used[1] += x.muscle_count;
                    max[0] = max[0].max(x.node_count);
                    max[1] = max[1].max(x.muscle_count);
                }
                slots[0] += 32 * max[0];
                slots[1] += 32 * max[1];
            }
        }
        println!(
            "{name:24} {:3} classes  node loops {:5.1}%  muscle loops {:5.1}%",
            members.len(),
            100.0 * used[0] as f64 / slots[0] as f64,
            100.0 * used[1] as f64 / slots[1] as f64
        );
    }
    Ok(())
}
