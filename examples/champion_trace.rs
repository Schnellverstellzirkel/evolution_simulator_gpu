//! Trace of one island's best elite: where it is, how high, what touches the
//! ground and how much muscle energy is left, every quarter second.
//! Usage: champion_trace <checkpoint.evo> <island> [rank]
mod common;
use evolution_simulator::{config::Config, storage};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let e = storage::load(std::path::Path::new(&args[1]))?;
    let archive = if args[2] == "g" { &e.archive } else { &e.islands[args[2].parse::<usize>()?] };
    let rank: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut elites: Vec<_> = archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let elite = elites[rank];
    let mut engine = common::open()?;
    let creature = elite.creature.unpack();
    println!("fitness {:.2} fine {} nodes {} bones {} muscles {} change: {:?}",
        elite.fitness, elite.fine, creature.nodes.len(), creature.bones.len(), creature.muscles.len(),
        e.lineage.get(&creature.id).map(|a| a.change.clone()));
    for (i, n) in creature.nodes.iter().enumerate() {
        println!("node {i}: x {:.3} y {:.3} d {:.3} friction {:.2}", n.x, n.y, n.diameter, n.friction);
    }
    for (i, b) in creature.bones.iter().enumerate() {
        println!("bone {i}: {}-{} len {:.3} organ {:.3}", b.a, b.b, b.rest_length, b.organ_mass);
    }
    println!("world: brambles {} mud {} wind {} slope {} water {} ground_friction {} patches {} gaps {} hurdles {} quake {}", e.config.brambles, e.config.mud, e.config.wind, e.config.slope, e.config.water, e.config.ground_friction, e.config.patches, e.config.gaps, e.config.hurdles, e.config.quake);
    let (mut c2, mut cfg) = elite.replay_of(&e.config);
    if std::env::var_os("FINE").is_some() { cfg = evolution_simulator::scheduler::confirm_config(&e.config); println!("fine replay"); }
    cfg.rungs = None;
    if std::env::var_os("NO_MUSCLES").is_some() { c2.muscles = Default::default(); println!("muscles removed"); }
    {
        let batch = common::score_creatures(&mut engine, &[c2.clone()], &Config { screen: None, ..cfg.clone() })?;
        println!("batch score of the same creature, same config: {:.3} fall {:.2}", batch[0].fitness, batch[0].fall_time);
    }
    let rec = common::record(&c2, &cfg)?;
    println!("replay {:.2} fall {:.2} frames {}", rec.result.fitness, rec.result.fall_time, rec.frames.len());
    let f = rec.forces.as_ref().expect("forces");
    let dt = cfg.duration / (rec.frames.len() as f32 - 1.0);
    let (mut pos, mut neg) = (0.0f64, 0.0f64);
    let mut per_node = vec![0.0f64; rec.frames[0].len()];
    for k in 1..rec.frames.len() {
        #[allow(clippy::needless_range_loop)]
        for j in 0..rec.frames[k].len() {
            let vx = (rec.frames[k][j][0] - rec.frames[k - 1][j][0]) / dt;
            let fr = f.friction.get(k).and_then(|v| v.get(j)).copied().unwrap_or(0.0);
            let w = (fr * vx * dt) as f64;
            per_node[j] += w;
            if w > 0.0 { pos += w } else { neg += w }
        }
    }
    println!("dt {dt:.5} friction work: positive {pos:.1} J negative {neg:.1} J, per node {per_node:.1?}, friction frames {}", f.friction.len());
    let mass: f32 = evolution_simulator::physics::nodes(&c2).iter().map(|n| n.mass).sum();
    {
        let mut w_pos = 0.0f64;
        let mut w_net = 0.0f64;
        let length = |k: usize, j: usize| -> f32 {
            let m = &c2.muscles[j];
            let pt = |bone: u32, t: f32| {
                let b = &c2.bones[bone as usize];
                let (pa, pb) = (rec.frames[k][b.a as usize], rec.frames[k][b.b as usize]);
                [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t]
            };
            let (p, q) = (pt(m.bone_a, m.anchor_a), pt(m.bone_b, m.anchor_b));
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
        };
        for k in 1..rec.frames.len() {
            for j in 0..c2.muscles.len() {
                let w = (f.muscle[k][j] * (length(k - 1, j) - length(k, j))) as f64;
                w_net += w;
                if w > 0.0 { w_pos += w; }
            }
        }
        println!("muscle work: net {w_net:.1} J, positive part {w_pos:.1} J");
    }
    if let Some(path) = std::env::var_os("DUMP") {
        let bones: Vec<[u32; 2]> = c2.bones.iter().map(|b| [b.a, b.b]).collect();
        let radii: Vec<f32> = c2.nodes.iter().map(|n| n.diameter * 0.5).collect();
        let touching: Vec<Vec<bool>> = f.ground.iter().map(|v| v.iter().map(|&x| x > 0.0).collect()).collect();
        let json = format!("{{\"dt\":{dt},\"bones\":{:?},\"radii\":{:?},\"frames\":{:?},\"touching\":{:?}}}", bones, radii, rec.frames, touching);
        std::fs::write(path, json)?;
    }
    let masses: Vec<f32> = evolution_simulator::physics::nodes(&c2).iter().map(|n| n.mass).collect();
    let com = |k: usize| -> [f32; 2] {
        let mut c = [0.0f32; 2];
        for (j, p) in rec.frames[k].iter().enumerate() { c[0] += masses[j] * p[0]; c[1] += masses[j] * p[1]; }
        [c[0] / mass, c[1] / mass]
    };
    let ke: f32 = (0..rec.frames[0].len()).map(|j| {
        let n = rec.frames.len() - 1;
        let v = [(rec.frames[n][j][0] - rec.frames[n - 1][j][0]) / dt, (rec.frames[n][j][1] - rec.frames[n - 1][j][1]) / dt];
        0.5 * masses[j] * (v[0] * v[0] + v[1] * v[1])
    }).sum();
    println!("final kinetic energy {ke:.1} J");
    let (mut jx, mut jy) = (0.0f64, 0.0f64);
    for k in 1..rec.frames.len() {
        jx += f.friction[k].iter().sum::<f32>() as f64 * dt as f64;
        jy += (f.ground[k].iter().sum::<f32>() - mass * 9.8) as f64 * dt as f64;
    }
    let n = rec.frames.len();
    let (a, b) = (n / 2, n - 1);
    let vx_half = (com(b)[0] - com(a)[0]) / ((b - a) as f32 * dt);
    println!("mass {mass:.1} kg; impulse friction_x {jx:.1} Ns; vertical (ground - weight) {jy:.1} Ns; mean vx second half {vx_half:.2} m/s -> momentum m*vx {:.1}; com start {:?} end {:?}", mass * vx_half, com(0), com(b));
    {
        let nn = rec.frames[0].len();
        let (mut touch, mut slide) = (vec![0usize; nn], vec![0.0f64; nn]);
        let mut degree = vec![0usize; nn];
        for b in creature.bones.iter() { degree[b.a as usize] += 1; degree[b.b as usize] += 1; }
        for k in 1..rec.frames.len() {
            for j in 0..nn {
                if f.ground[k][j] > 0.0 {
                    touch[j] += 1;
                    slide[j] += ((rec.frames[k][j][0] - rec.frames[k - 1][j][0]) / dt).abs() as f64;
                }
            }
        }
        let total = rec.frames.len() as f64;
        let mut line = String::from("contacts (node:degree share% slidespeed):");
        for j in 0..nn {
            if touch[j] > 0 { line += &format!(" {j}:d{} {:.0}% {:.2}m/s;", degree[j], 100.0 * touch[j] as f64 / total, slide[j] / touch[j] as f64); }
        }
        println!("{line}");
    }
    for k in (2000..(2000 + 64 * 4).min(rec.frames.len())).step_by(4) {
        let c = com(k);
        let mut line = format!("f{k} com_x {:.2} |", c[0]);
        for (j, p) in rec.frames[k].iter().enumerate() {
            line += &format!(" {j}:({:+.2},{:.2}){}", p[0] - c[0], p[1], if f.ground[k][j] > 0.0 { "*" } else { "" });
        }
        println!("{line}");
    }
    let step = (rec.frames.len() / 80).max(1);
    println!("t   com_x  com_y  min_y  max_y  groundN  touching  energy_min");
    for (k, frame) in rec.frames.iter().enumerate().step_by(step) {
        let n = frame.len() as f32;
        let cx = frame.iter().map(|p| p[0]).sum::<f32>() / n;
        let cy = frame.iter().map(|p| p[1]).sum::<f32>() / n;
        let miny = frame.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
        let maxy = frame.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        let g: f32 = f.ground.get(k).map_or(0.0, |v| v.iter().sum());
        let t = f.ground.get(k).map_or(0, |v| v.iter().filter(|&&x| x > 0.0).count());
        let en = f.energy.get(k).map_or(1.0, |v| v.iter().copied().fold(1.0, f32::min));
        println!("{:5.2} {:7.2} {:6.2} {:6.2} {:6.2} {:8.0} {:5} {:6.2}", k as f32 / 60.0, cx, cy, miny, maxy, g, t, en);
    }
    Ok(())
}
