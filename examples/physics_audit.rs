//! Solver-made energy and momentum of elites under the selected physics
//! (`EVOLUTION_PHYSICS=2` for v2). Replays elites spread over the archive's
//! rank order on the CPU engine and prints, per elite and in total, where its
//! energy and momentum came from: muscle work, mechanical energy the solver
//! added or removed beyond it, the momentum balance and first-law
//! corrections of v2, friction that pushed a node the way it slid, and the
//! steps by contact count. Diagnostic only.
//! With `random` in place of a checkpoint it audits a random first generation.
//! Usage: EVOLUTION_PHYSICS=2 cargo run --release --example physics_audit <checkpoint.evo|random> [count]
use evolution_simulator::{config::Config, cpu_engine, physics, physics2, storage};

fn main() {
    let path = std::env::args().nth(1).expect("checkpoint");
    let count: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    assert!(physics2::enabled(), "set EVOLUTION_PHYSICS=2");
    let (config, mut creatures): (Config, Vec<(f32, evolution_simulator::evolution::Creature)>) =
        if path == "random" {
            let cfg = Config {
                population: count,
                random_seed: false,
                ..Config::default()
            };
            let pop = evolution_simulator::evolution::create(&cfg).unwrap();
            let list = (0..count).map(|i| (0.0, pop.creature(i))).collect();
            (cfg, list)
        } else {
            let e = storage::load(std::path::Path::new(&path)).unwrap();
            let list = e
                .archive
                .entries
                .iter()
                .map(|x| (x.fitness, x.creature.clone()))
                .collect();
            (e.config.clone(), list)
        };
    let cfg = Config {
        screen: None,
        ..config
    };
    creatures.sort_by(|a, b| b.0.total_cmp(&a.0));
    let elites = creatures;
    println!(
        "{} elites; replaying {count} spread over the rank order",
        elites.len()
    );
    println!(
        "rank  archive_m replay_m nodes muscles kg  work_J  gained_J lost_J  bal+J bal-J  firstlaw_J  contact_free%  fric_push%  cost_J/kg/m  fell_s  Nwork+  Nwork-  Fwork+  Fwork-"
    );
    let mut total = [0.0f64; 16];
    let mut with_muscles = 0usize;
    for k in 0..count {
        let rank = (elites.len() - 1) * k / (count - 1).max(1);
        let (archive_m, c) = &elites[rank];
        let (_frames, result) = cpu_engine::replay(c, &cfg);
        let energy = physics2::ENERGY.with(|l| l.get());
        let mass: f32 = physics::nodes(c).iter().map(|n| n.mass).sum();
        let cost = cpu_engine::transport_cost(c, &cfg);
        let steps = energy[6].max(1.0);
        let fric = if energy[10] > 0.0 {
            100.0 * energy[9] / energy[10]
        } else {
            0.0
        };
        println!(
            "{rank:5} {:9.2} {:8.2} {:5} {:7} {:5.1} {:7.1} {:8.1} {:7.1} {:6.2} {:5.2} {:11.2} {:13.1} {:11.1} {:12} {:7.2} {:7.1} {:7.1} {:7.1} {:7.1}",
            archive_m,
            result.fitness,
            c.nodes.len(),
            c.muscles.len(),
            mass,
            energy[0],
            energy[1],
            energy[2],
            energy[3],
            energy[4],
            energy[11],
            100.0 * energy[5] / steps,
            fric,
            cost.map_or("n/a".into(), |v| format!("{v:.1}")),
            result.fall_time,
            energy[12],
            energy[13],
            energy[14],
            energy[15]
        );
        for (t, v) in total.iter_mut().zip(energy) {
            *t += v;
        }
        with_muscles += 1;
    }
    println!(
        "total over {with_muscles}: muscle work {:.1} J, solver gained {:.1} J, lost {:.1} J, balance +{:.2} -{:.2} J, first-law {:.2} J, friction push {:.1}% of {:.0} N s",
        total[0],
        total[1],
        total[2],
        total[3],
        total[4],
        total[11],
        if total[10] > 0.0 {
            100.0 * total[9] / total[10]
        } else {
            0.0
        },
        total[10]
    );
}
