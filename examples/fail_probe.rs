//! For the top elites of a checkpoint: the median ratio of the score at the
//! fine physics to the standard score, with parts of the genes changed (a
//! diagnostic).
mod common;
use evolution_simulator::{config::Config, evolution::{self, Creature}, storage};
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    if v.is_empty() { f32::NAN } else { v[v.len() / 2] }
}
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let e = storage::load(std::path::Path::new(&args[1]))?;
    let top: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(300);
    let mut elites: Vec<_> = e.archive.entries.iter().collect();
    elites.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    elites.truncate(top);
    let base: Vec<Creature> = elites.iter().map(|x| x.creature.clone()).collect();
    let std_cfg = Config { screen: None, ..e.config.clone() };
    let mut fine_cfg = evolution_simulator::scheduler::confirm_config(&std_cfg);
    fine_cfg.screen = None;
    let mut engine = common::open()?;
    if let Some(rank) = args.get(3).and_then(|v| v.parse::<usize>().ok()) {
        // Trace one elite at both physics: centre of mass x and head height each second.
        let c = &base[rank];
        let nodes = evolution_simulator::physics::nodes(c);
        let mass: f32 = nodes.iter().map(|n| n.mass).sum();
        for (name, cfg) in [("standard", &std_cfg), ("fine", &fine_cfg)] {
            let rec = common::record(c, cfg)?;
            let rate = cfg.fidelity().rate as usize;
            let settle = cfg.fidelity().settle() as usize;
            println!("{name}: score {:.1} m, fall time {:.2}", rec.result.fitness, rec.result.fall_time);
            for t in (0..20).step_by(1) {
                let f = settle + t * rate;
                if f >= rec.frames.len() { break; }
                let fr = &rec.frames[f];
                let x: f32 = fr.iter().zip(&nodes).map(|(p, n)| p[0] * n.mass).sum::<f32>() / mass;
                let y: f32 = fr.iter().zip(&nodes).map(|(p, n)| p[1] * n.mass).sum::<f32>() / mass;
                println!("  t={t:2} x={x:7.2} y={y:5.2} head_y={:5.2}", fr[0][1]);
            }
        }
        return Ok(());
    }
    let variants: Vec<(&str, Box<dyn Fn(&mut Creature)>)> = vec![
        ("as is", Box::new(|_| {})),
        ("no ligament", Box::new(|c| for b in c.bones.iter_mut() { b.ligament = 0.0 })),
        ("no sensors", Box::new(|c| for m in c.muscles.iter_mut() { m.sensor = evolution::NO_SENSOR })),
        ("strength x0.5", Box::new(|c| for m in c.muscles.iter_mut() { m.strength = (m.strength * 0.5).max(evolution::STRENGTH_MIN) })),
    ];
    for (name, change) in variants {
        let mut cs = base.clone();
        for c in &mut cs { change(c); }
        let a = common::score_creatures(&mut engine, &cs, &std_cfg)?;
        let b = common::score_creatures(&mut engine, &cs, &fine_cfg)?;
        let ratios: Vec<f32> = a.iter().zip(&b).filter(|(x, _)| x.fitness > 1.0).map(|(x, y)| y.fitness / x.fitness).collect();
        let broke: Vec<f32> = a.iter().zip(&b).filter(|(x, y)| x.fitness > 1.0 && y.fitness < 0.5 * x.fitness).map(|(_, y)| y.fall_time).collect();
        let early = broke.iter().filter(|&&t| t > 0.0 && t < 1.0).count();
        let late = broke.iter().filter(|&&t| t >= 1.0).count();
        let none = broke.iter().filter(|&&t| t <= 0.0).count();
        println!("{name}: of {} elites below half at the fine physics, {early} fell before 1 s, {late} fell later, {none} did not fall", broke.len());
        let kept = ratios.iter().filter(|&&r| r >= 0.5).count();
        println!("{name}: standard median {:.1} m, fine/standard median {:.3}, {} of {} keep half", median(a.iter().map(|r| r.fitness).collect()), median(ratios.clone()), kept, ratios.len());
    }
    Ok(())
}
