//! Backlog item 48: the CPU engine's whole-group early exit, behind
//! `EVOLUTION_EARLY_EXIT`.
//!
//! A lane's fitness is final once it has fallen or a node has failed, but the
//! default engine keeps accumulating behavior totals (ground contact, height,
//! gait turns, contact bitsets) for the rest of the trial. Stopping the whole
//! group on the last finished lane therefore keeps fitness, fall time and
//! head shake bit for bit, while the descriptor totals of finished lanes are
//! lower than in the full run. The tests below pin both sides of that:
//!
//! - a group where every lane falls exits after the first timed step and
//!   keeps every lane's score;
//! - a group with one live lane never exits, and every field of every lane
//!   (including the live one) is unchanged;
//! - a short (partial) group exits once its real lanes have finished.
//!
//! The flag is read from the environment per group, so a comparison test can
//! toggle it. All tests here share one lock because the variable is
//! process-wide.
use evolution_simulator::{
    config::Config,
    cpu_engine::{self, EARLY_EXIT_GROUPS, EARLY_EXIT_GROUPS_TOTAL},
    creature_kernel::GpuResult,
    evolution::{Bone, Creature, NodeGene, Population},
};
use std::sync::Mutex;
use std::sync::atomic::Ordering;

/// Toggling a process-wide variable while other tests evaluate in parallel
/// would leak into their runs, so every test in this file holds this lock.
static ENV: Mutex<()> = Mutex::new(());

/// A two-node, one-bone body with no muscles. Both variants share the body
/// plan, so the engine puts them in one SIMD group. The head is node 0 and
/// bone 0's child is node 1. With the head gene low and the neck gene high,
/// the settled head sits below its neck base and the creature falls on the
/// first timed step; with the genes the other way around it lies still and
/// never falls.
fn two_node_body(id: u64, head_first: bool) -> Creature {
    let (head_y, neck_y, head_diameter, neck_diameter) = if head_first {
        (0.16, 0.10, 0.12, 0.06)
    } else {
        (0.10, 0.16, 0.06, 0.12)
    };
    Creature {
        nodes: vec![
            NodeGene {
                x: 0.0,
                y: head_y,
                diameter: head_diameter,
                friction: 0.9,
            },
            NodeGene {
                x: 0.6,
                y: neck_y,
                diameter: neck_diameter,
                friction: 0.9,
            },
        ],
        bones: vec![Bone::new(0, 1, 0.6)],
        muscles: Vec::new(),
        id,
        mutability: 1.0,
    }
}

fn test_config() -> Config {
    Config {
        population: 16,
        duration: 1.0,
        random_seed: false,
        ..Config::default()
    }
}

fn population(creatures: impl IntoIterator<Item = Creature>) -> Population {
    let mut pop = Population::default();
    for creature in creatures {
        pop.push(creature);
    }
    pop
}

fn evaluate(early_exit: bool, pop: &Population, cfg: &Config) -> Vec<GpuResult> {
    unsafe {
        if early_exit {
            std::env::set_var("EVOLUTION_EARLY_EXIT", "1");
        } else {
            std::env::remove_var("EVOLUTION_EARLY_EXIT");
        }
    }
    cpu_engine::evaluate(pop, cfg)
}

/// Evaluates with the exit on and reports the two diagnostic counters. The
/// caller holds the environment lock and has reset the counters.
fn evaluate_counted(pop: &Population, cfg: &Config) -> (Vec<GpuResult>, u64, u64) {
    EARLY_EXIT_GROUPS.store(0, Ordering::Relaxed);
    EARLY_EXIT_GROUPS_TOTAL.store(0, Ordering::Relaxed);
    let results = evaluate(true, pop, cfg);
    (
        results,
        EARLY_EXIT_GROUPS.load(Ordering::Relaxed),
        EARLY_EXIT_GROUPS_TOTAL.load(Ordering::Relaxed),
    )
}

/// Compares every f32 bit of two results.
fn assert_same_result(label: &str, a: &GpuResult, b: &GpuResult) {
    macro_rules! same {
        ($field:ident) => {
            assert_eq!(
                a.$field.to_bits(),
                b.$field.to_bits(),
                "{label}: {} is {} vs {}",
                stringify!($field),
                a.$field,
                b.$field
            );
        };
    }
    same!(fitness);
    same!(ground_contact);
    same!(vertical_oscillation);
    same!(gait_frequency);
    same!(previous_center_y);
    same!(vertical_extremum);
    same!(vertical_trend);
    same!(gait_turns);
    same!(height_sum);
    same!(contact_lo);
    same!(contact_hi);
    same!(lift_lo);
    same!(lift_hi);
    same!(ground_lo);
    same!(ground_hi);
    same!(fall_time);
    same!(head_shake);
}

/// A full group of lanes that all fall on the first timed step exits early.
/// Fitness, fall time and head shake match the full run; the behavior totals
/// are lower because the post-fall tail is skipped.
#[test]
fn every_fallen_group_stops_early_and_keeps_every_score() {
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = test_config();
    let pop = population((0..16).map(|i| two_node_body(100 + i, false)));

    let full = evaluate(false, &pop, &cfg);
    assert!(
        full.iter().all(|r| r.fall_time > 0.0),
        "fixture: every lane must fall; got {:?}",
        full.iter().map(|r| r.fall_time).collect::<Vec<_>>()
    );

    let (early, exit_groups, total_groups) = evaluate_counted(&pop, &cfg);
    assert_eq!(total_groups, 1, "one SIMD group for the population");
    assert_eq!(exit_groups, 1, "the group must exit early");

    for (lane, (a, b)) in full.iter().zip(&early).enumerate() {
        assert_eq!(
            a.fitness.to_bits(),
            b.fitness.to_bits(),
            "lane {lane}: fitness {} vs {}",
            a.fitness,
            b.fitness
        );
        assert_eq!(a.fall_time.to_bits(), b.fall_time.to_bits(), "lane {lane}");
        assert_eq!(
            a.head_shake.to_bits(),
            b.head_shake.to_bits(),
            "lane {lane}"
        );
    }
    assert!(
        early
            .iter()
            .zip(&full)
            .any(|(a, b)| a.ground_contact < b.ground_contact),
        "the stopped run must drop the post-fall contact totals"
    );
    assert!(
        early
            .iter()
            .all(|a| a.ground_contact.is_finite() && a.fitness.is_finite()),
        "the stopped run still reports finite totals"
    );
}

/// A group with one lane that never finishes cannot exit, so every field of
/// every lane must match the full run exactly. The live lane must also match
/// its own single-creature evaluation.
#[test]
fn a_mixed_group_with_a_live_lane_is_unchanged() {
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = test_config();
    let mut pop = population([two_node_body(1, true)]);
    for i in 0..15 {
        pop.push(two_node_body(200 + i, false));
    }

    let solo = population([two_node_body(1, true)]);
    let solo_result = evaluate(false, &solo, &cfg)[0];
    assert_eq!(
        solo_result.fall_time, 0.0,
        "fixture: the live lane must not fall"
    );

    EARLY_EXIT_GROUPS.store(0, Ordering::Relaxed);
    EARLY_EXIT_GROUPS_TOTAL.store(0, Ordering::Relaxed);
    let full = evaluate(false, &pop, &cfg);
    let early = evaluate(true, &pop, &cfg);
    assert_eq!(full[0].fall_time, 0.0, "fixture: lane 0 must stay upright");
    assert_eq!(
        EARLY_EXIT_GROUPS.load(Ordering::Relaxed),
        0,
        "a group with a live lane must not exit early"
    );
    assert_eq!(EARLY_EXIT_GROUPS_TOTAL.load(Ordering::Relaxed), 1);

    for (lane, (a, b)) in full.iter().zip(&early).enumerate() {
        assert_same_result(&format!("lane {lane}"), a, b);
    }
    assert_same_result("the live lane alone", &solo_result, &full[0]);
}

/// A partial group (fewer than 16 real lanes) exits as soon as its real lanes
/// have finished, not when the padding copies do.
#[test]
fn a_partial_group_exits_on_its_real_lanes() {
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = test_config();
    // Five real lanes; the remaining eleven repeat the last one and are
    // discarded. Like the first test, the fall happens on the first timed
    // step, so without the exit the run would be nearly a full trial longer.
    let pop = population((0..5).map(|i| two_node_body(300 + i, false)));

    let full = evaluate(false, &pop, &cfg);
    let (early, exit_groups, total_groups) = evaluate_counted(&pop, &cfg);
    assert_eq!(total_groups, 1);
    assert_eq!(exit_groups, 1, "the real five lanes must finish the group");
    for (lane, (a, b)) in full.iter().zip(&early).enumerate() {
        assert_eq!(a.fitness.to_bits(), b.fitness.to_bits(), "lane {lane}");
        assert_eq!(a.fall_time.to_bits(), b.fall_time.to_bits(), "lane {lane}");
    }
}
