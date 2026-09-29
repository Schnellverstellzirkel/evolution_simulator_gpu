//! Solver-made momentum and energy under physics v2, on bodies that a short
//! CPU evolution produced (random bodies barely move, so they show little).
//! The ledgers are the thread-local `physics2::LEDGER` and `physics2::ENERGY`,
//! filled by `physics2::replay`. Numbers measured on 6ccc472 are in the
//! comments; the thresholds leave room for search noise.
use evolution_simulator::{
    config::Config, cpu_engine, evolution::Creature, physics2, scheduler, storage::Experiment,
};

/// Elites of a short deterministic evolution: 1,500 bodies, 10 generations,
/// 10 s trials, seed 40.
fn elites() -> Vec<Creature> {
    let cfg = Config {
        population: 1500,
        duration: 10.0,
        random_seed: false,
        seed: 40,
        screen: None,
        ..Config::default()
    };
    let mut e = Experiment::new(cfg).unwrap();
    for _ in 0..10 {
        let results = cpu_engine::evaluate(&e.population, &e.config);
        for (i, r) in results.iter().enumerate() {
            let m = scheduler::to_metrics(&e.population, i, r, &e.config);
            e.record_result(i, &m);
        }
        e.evaluated = e.config.population;
        e.archive_batch().unwrap();
        e.prepare_next_batch().unwrap();
    }
    let mut list: Vec<_> = e.archive.entries.iter().collect();
    list.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
    let step = (list.len() / 40).max(1);
    list.iter()
        .step_by(step)
        .map(|x| x.creature.clone())
        .collect()
}

#[test]
fn ground_and_wind_account_for_the_momentum_v2_gains() {
    let cfg = Config {
        duration: 10.0,
        random_seed: false,
        screen: None,
        wind: 0.5,
        ..Config::default()
    };
    for (i, c) in elites().iter().enumerate() {
        let _ = physics2::replay(c, &cfg);
        let [impulse, change] = physics2::LEDGER.with(|l| l.get());
        // The step balances momentum against the external impulses, so the
        // two sums agree apart from air drag.
        let scale = impulse.abs().max(change.abs()).max(1.0);
        assert!(
            (impulse - change).abs() <= 0.1 * scale + 1.0,
            "body {i}: external impulse {impulse:.2} N s, momentum change {change:.2} N s"
        );
    }
}

#[test]
fn v2_bodies_lose_more_energy_than_the_solver_adds() {
    let cfg = Config {
        duration: 10.0,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let (mut net, mut work) = (0.0f64, 0.0f64);
    let (mut friction_push, mut friction) = (0.0f64, 0.0f64);
    for c in elites() {
        let _ = physics2::replay(&c, &cfg);
        let e = physics2::ENERGY.with(|l| l.get());
        work += e[0];
        net += e[1] - e[2];
        friction_push += e[9];
        friction += e[10];
    }
    eprintln!(
        "net residual {net:.1} J, muscle work {work:.1} J, friction push {friction_push:.1} of {friction:.1} N s"
    );
    // Over the elites the residual is dissipation: solver gains stay below
    // what it removes (seed 40 at 20 s: -9,387 J against 4,871 J of work).
    assert!(net < 0.0, "net solver energy {net:.1} J");
    // Friction pushes a node the way it slid for a small share of its
    // impulse (2.7% at 20 s, seed 40; the champion alone had 37%).
    assert!(
        friction_push <= 0.15 * friction.max(1.0),
        "friction pushed {friction_push:.1} of {friction:.1} N s"
    );
}

#[test]
fn a_hopper_does_not_turn_the_ground_into_a_motor() {
    // Whatever ground contacts add to a body's energy through the normal
    // impulses stays a small part of what they take away.
    let cfg = Config {
        duration: 10.0,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let (mut added, mut removed) = (0.0f64, 0.0f64);
    for c in elites() {
        let _ = physics2::replay(&c, &cfg);
        let e = physics2::ENERGY.with(|l| l.get());
        added += e[12] + e[14];
        removed += -(e[13] + e[15]);
    }
    eprintln!("ground work added {added:.1} J, removed {removed:.1} J");
    assert!(
        added < 0.5 * removed,
        "ground added {added:.1} J, removed {removed:.1} J"
    );
}

/// The seed 40 champion of a 40-generation v2 evolution (5,000 bodies, 20 s
/// trials, 63.4 m): a 5-node hopper that reached that distance because
/// friction pushed its foot the way the foot moved. Before the fix friction
/// did +1,725 J of work on its nodes (37% of its friction impulse) against
/// 739 J of muscle work, and removing the planting pass cut its distance to
/// 1.3 m.
#[test]
fn friction_never_does_positive_work_on_the_exploiting_hopper() {
    let creature: Creature =
        serde_json::from_str(include_str!("fixtures/p2_friction_champion.json")).unwrap();
    let cfg = Config {
        duration: 20.0,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let (_, result) = physics2::replay(&creature, &cfg);
    let e = physics2::ENERGY.with(|l| l.get());
    eprintln!(
        "distance {:.2} m, friction work +{:.1} J -{:.1} J, muscle work {:.1} J",
        result.fitness, e[14], e[15], e[0]
    );
    assert!(
        e[14] < 20.0,
        "friction did {:.1} J of positive work on the nodes",
        e[14]
    );
}

/// Rank 881 of a seed 40 evolution: a 1.3 kg body whose capped 100 N muscles
/// turned light limbs about a radian in one step. The integrator then made
/// momentum (18 N s on a body that should hold 1.2) and the momentum balance
/// turned it into kinetic energy: +338 J gained against 26 J of muscle drain,
/// 287 J of it removed again by the balance. With muscle force scaled to the
/// mass a muscle drives, it gains 9 J and the balance removes 0.1 J.
#[test]
fn a_light_body_gains_no_unpaid_energy() {
    let creature: Creature =
        serde_json::from_str(include_str!("fixtures/p2_light_body_r881.json")).unwrap();
    let cfg = Config {
        duration: 20.0,
        random_seed: false,
        screen: None,
        ..Config::default()
    };
    let _ = physics2::replay(&creature, &cfg);
    let e = physics2::ENERGY.with(|l| l.get());
    eprintln!(
        "gained {:.1} J lost {:.1} J, balance +{:.1} -{:.1} J",
        e[1], e[2], e[3], e[4]
    );
    assert!(e[1] - e[2] < 20.0, "net solver energy {:.1} J", e[1] - e[2]);
    assert!(e[4] < 20.0, "the balance removed {:.1} J", e[4]);
}

#[test]
fn a_muscle_is_as_strong_as_the_mass_it_drives() {
    // A muscle between two light limbs gets a small force cap; one that
    // drives a heavy subtree keeps the full cap.
    let cfg = Config {
        random_seed: false,
        ..Config::default()
    };
    let light: Creature =
        serde_json::from_str(include_str!("fixtures/p2_light_body_r881.json")).unwrap();
    let strengths = physics2::Model::new(&light, &cfg).muscle_strengths();
    assert!(strengths.iter().all(|s| (0.0..=1.0).contains(s)));
    assert!(strengths.iter().any(|&s| s < 0.5), "{strengths:?}");
}
