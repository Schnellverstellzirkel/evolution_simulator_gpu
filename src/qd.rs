//! The state of the MAP-Elites search: behavior descriptors, archives of
//! elites and the emitters that breed from them.
//! An archive keeps the fastest creature of each cell, where a cell is a way
//! of moving times a body class, and a reserve of new body plans.
//! The module also holds the CMA-ES samplers, the layout of ring slots over
//! islands and nurseries, and the save version `VERSION`.
//! `storage::Experiment` owns the archives and `evolution` breeds from them.
use crate::evolution::{Creature, Muscle, Population, Rng, StoredCreature};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Number of emitter kinds (`Emitter`).
pub const EMITTER_COUNT: usize = 4;
/// The bins of the movement grid: ground contact, gait cadence, mean body
/// height and feet (nodes that touched the ground and lifted off again).
const MOVEMENT_BINS: [u8; 4] = [6, 8, 6, 5];
/// The axes of measured and shape behavior, which local competition and
/// novelty compare across neighboring cells. The node-count class is the
/// last byte of a niche and only separates bodies: neighbors share it.
const NEIGHBOR_AXES: usize = 5;
/// Generations an island keeps one elite per way of moving before its
/// archive is refined to the cells of its body classes. Each isolated island
/// waits 10 generations longer than the one before it, and a wild island is
/// never refined (`Experiment::refine_archives`). Refined from the first
/// generation, the islands climbed 17% slower at generation 40. Refined at
/// generation 30 or 40, they climbed as fast as an archive that stayed coarse.
pub const REFINE_AFTER: u32 = 30;
/// Names of the body shape classes, most compact first, and of the body size
/// classes, smallest first, for every count of classes.
const SHAPE_NAME_SETS: [&[&str]; 6] = [
    &[],
    &["Any"],
    &["Compact", "Long"],
    &["Tall", "Wide", "Long"],
    &["Tall", "Square", "Wide", "Long"],
    &["Tall", "Square", "Wide", "Long", "Needle"],
];
const SIZE_NAME_SETS: [&[&str]; 6] = [
    &[],
    &["Any"],
    &["Small", "Large"],
    &["Small", "Medium", "Large"],
    &["Small", "Medium", "Large", "Giant"],
    &["Tiny", "Small", "Medium", "Large", "Giant"],
];
/// How an archive tells bodies apart once it is refined: the edges of its
/// shape classes (the start pose's width over its height) and of its size
/// classes (node count), and their names.
#[derive(Clone, Copy, Debug)]
pub struct Classes {
    /// Where each shape class after the first starts, as the start pose's
    /// width over its height.
    aspect: &'static [f32],
    /// Where each size class after the first starts, as a node count.
    nodes: &'static [u16],
    /// Names of the shape classes, most compact first.
    pub shape_names: &'static [&'static str],
    /// Names of the size classes, smallest first.
    pub size_names: &'static [&'static str],
}
const ISLAND_ASPECT: [f32; 2] = [1.2, 2.0];
const ISLAND_NODES: [u16; 2] = [9, 12];
const GLOBAL_ASPECT: [f32; 3] = [0.9, 1.4, 2.5];
const GLOBAL_NODES: [u16; 3] = [9, 11, 14];
/// The layout of the islands: each way of moving splits among 3 shapes (under
/// 1.2 times as wide as tall, 1.2 to 2.0, longer) and 3 sizes (up to 8 nodes,
/// 9 to 11, 12 or more), 12,960 cells.
pub const ISLAND_CLASSES: Classes = Classes {
    aspect: &ISLAND_ASPECT,
    nodes: &ISLAND_NODES,
    shape_names: SHAPE_NAME_SETS[ISLAND_ASPECT.len() + 1],
    size_names: SIZE_NAME_SETS[ISLAND_NODES.len() + 1],
};
/// The layout of the global archive, which records every creature and is
/// never a parent source: each way of moving splits among 4 shapes (tall under
/// 0.9, square to 1.4, wide to 2.5, long) and 4 sizes (up to 8 nodes, 9 to 10,
/// 11 to 13, 14 or more), 23,040 cells.
pub const GLOBAL_CLASSES: Classes = Classes {
    aspect: &GLOBAL_ASPECT,
    nodes: &GLOBAL_NODES,
    shape_names: SHAPE_NAME_SETS[GLOBAL_ASPECT.len() + 1],
    size_names: SIZE_NAME_SETS[GLOBAL_NODES.len() + 1],
};
impl Classes {
    /// Number of body shape classes (aspect ratio bins).
    pub const fn shapes(&self) -> usize {
        self.aspect.len() + 1
    }
    /// Number of body size classes (node count bins).
    pub const fn sizes(&self) -> usize {
        self.nodes.len() + 1
    }
    /// Body classes: every shape with every size.
    pub const fn classes(&self) -> usize {
        self.shapes() * self.sizes()
    }
    /// Cells: every way of moving with every body class.
    pub const fn cells(&self) -> usize {
        MOVEMENT_CELLS * self.classes()
    }
    /// What a shape class covers, for hover texts.
    pub fn shape_about(&self, class: usize) -> String {
        let low = class.checked_sub(1).map(|c| self.aspect[c]);
        match (low, self.aspect.get(class)) {
            (None, Some(high)) => format!("Less than {high} times as wide as tall at the start"),
            (Some(low), Some(high)) => {
                format!("{low} to {high} times as wide as tall at the start")
            }
            (Some(low), None) => format!("At least {low} times as wide as tall at the start"),
            (None, None) => "Every shape".to_owned(),
        }
    }
    /// What a size class covers, for hover texts.
    pub fn size_about(&self, class: usize) -> String {
        let low = class.checked_sub(1).map(|c| self.nodes[c]);
        match (low, self.nodes.get(class)) {
            (None, Some(high)) => format!("Up to {} nodes", high - 1),
            (Some(low), Some(high)) => format!("{low} to {} nodes", high - 1),
            (Some(low), None) => format!("{low} nodes or more"),
            (None, None) => "Every size".to_owned(),
        }
    }
    /// The shape class and the size class of a body with this start-pose
    /// aspect and node count.
    fn classes_of(&self, aspect: f32, nodes: u16) -> (u8, u8) {
        (
            self.aspect.iter().filter(|&&edge| aspect >= edge).count() as u8,
            self.nodes.iter().filter(|&&edge| nodes >= edge).count() as u8,
        )
    }
    /// The bins of each byte of a niche.
    fn bins(&self) -> [u8; 6] {
        [
            MOVEMENT_BINS[0],
            MOVEMENT_BINS[1],
            self.shapes() as u8,
            MOVEMENT_BINS[2],
            MOVEMENT_BINS[3],
            self.sizes() as u8,
        ]
    }
}
/// Cells of the movement grid alone: contact, cadence, height and feet.
pub(crate) const MOVEMENT_CELLS: usize = (MOVEMENT_BINS[0] as usize)
    * (MOVEMENT_BINS[1] as usize)
    * (MOVEMENT_BINS[2] as usize)
    * (MOVEMENT_BINS[3] as usize);
/// Most entries the morphology reserve of an archive holds.
pub(crate) const MORPHOLOGY_LIMIT: usize = 64;
/// Most cells a generation's statistics may report. `Experiment::validate`
/// checks a loaded history against it.
pub(crate) const HISTORICAL_ARCHIVE_LIMIT: usize = 1 << 20;
/// Most CMA emitters and optimizers the experiment keeps. When it is full, a
/// new one replaces the one used longest ago.
pub(crate) const CMA_LIMIT: usize = 96;
// 26: a fall ends the trial; behavior totals stop at the fall and average
// over the steps walked.
// 27: every node the ground pushes feels friction, and the lift's friction
// may only slow the body (sliders).
// 28: selected-engine scores and descriptors own archive admission; GPU
//     results are no longer rescored through CPU playback.
// 29: trials last 20 s instead of 60 s; older games move to 20 s on load.
// 30: physics v2 (articulated tree in reduced coordinates) scores the game.
// 31: muscle energy pays for active contraction only.
// 32: air drag on bones.
// 33: evolvable elastic tendons.
// 34: the world gains Water and Ice patches (the saved settings changed).
// 35: creatures lose the unused mutability gene (the save format changed).
// 36: static friction: a foot that barely slides holds 25% harder.
// 37: four isolated islands and a hub; each island keeps its own morphology
//     reserve, CMA emitters and reseed queue (the save format changed).
// 38: every island has a nursery archive for new random bodies (the save
//     format changed).
// 39: muscles have mass (a fixed part plus a part per metre), and an elite
//     remembers whether its score came from its fine check.
// 40: seasons became autochange environment (renamed settings fields, a
//     ladder that only adds effects).
// 41: a tendon starts to pull past the longer of its muscle's longest length
//     and its length in the start pose, and muscle mass follows that length.
// 42: no fine checks; a creature that would set an island record gets one
//     confirmation trial from the same pose at twice the rate and solver
//     passes (the fine fidelity), and the search runs on a ring of blocks.
// 43: the CUDA kernel runs a creature per lane group with substeps instead
//     of planting rounds, warm start and static friction.
// 44: the CUDA contact solve keeps the contact matrix in registers and runs
//     two sweeps.
// 45: breeding draws from counter-based streams keyed by slot and gene with
//     a 12-uniform gaussian, and bodies are held to 32 nodes and 96 muscles.
// 46: the ring's block size and block count are saved with the experiment
//     and recorded in every generation's statistics.
// 47: one substep per 1/60 s step (the L0 rung of the substep ladder).
// 48: the growth-step body rule (a child gains at most 4 nodes and 4 muscles).
// 50: the audit lane and the early rungs: the save holds the audit window
//     (49 is the lean muscle model).
// 53: back to the articulated-body muscles at one substep (owner, for
//     speed; 49 to 52 were the lean muscle model).
// 54: archive cells also follow body shape and size (aspect and node-count
//     classes). A save of version 53 loads by moving each elite to its cell
//     in the new layout.
// 55: the global archive has finer body classes than the islands (4 shapes by
//     4 sizes against 2 by 2). A save of version 53 or 54 loads by moving
//     each elite to its cell in the new layout.
// 56: the Brambles world effect (drag on every node but the feet while it
//     touches the ground). Older saves load with it cleared.
// 57: the islands have 3 shapes by 3 sizes of body class (2 by 2 before), and
//     a save is compressed with long-range matching. A save of version 56
//     loads by moving each elite to its cell in the new layout.
// 58: 100 wild islands beside the isolated islands and the hub, each in a
//     world of its own. The save holds their archives.
// 59: the statistics of a generation gained the body plans, the effective
//     clades and the median plan age.
// 60: muscles are twice as strong (force cap 200 N, 200 m/s^2). Scores of
//     older saves came from weaker muscles.
// 61: a node inside the ground is moved out as a position change with no
//     velocity and no normal impulse (it was a velocity goal, which gave
//     bodies energy and friction grip at one substep). Scores of older saves
//     came from the old contact.
// 62: the wild islands and their reshaped nurseries keep one elite per way of
//     moving and never refine (memory). Saves of 61 and older load as they are.
// 63: every entrant above half its archive's best gets its fine trial and
//     enters with the lower score. Older saves hold standard-trial scores that
//     no replay reaches.
// 64: a new kernel: one thread per creature, position-based dynamics with
//     small substeps, hard joint limits and ground contacts that add no
//     energy. Scores of older saves came from the old physics.
/// The version of the archives and of the physics that scored them. A save of
/// another version is turned down at load unless `loadable` accepts it. Bump
/// it when archive or physics semantics change.
pub const VERSION: u32 = 64;
/// The oldest save version that still loads. Its archives are re-binned, and
/// its elites keep the scores they measured.
pub const OLDEST_LOADABLE: u32 = 53;
/// Whether a save of `version` loads in this game.
pub fn loadable(version: u32) -> bool {
    (OLDEST_LOADABLE..=VERSION).contains(&version)
}
/// Neighbors that novelty and local competition compare an elite with.
const LOCAL_NEIGHBORS: usize = 5;
/// Elites the novelty emitter weighs: the ones visited least.
const LEAST_VISITED: usize = 32;
const MORPHOLOGY_NICHE_MARKER: u8 = u8::MAX;
/// First byte of an optimizer's niche. Behavior niches never reach it and
/// morphology niches use 255.
const OPTIMIZER_NICHE_MARKER: u8 = 254;
/// The main islands: the isolated islands and the hub. They run in the
/// player's world.
pub const MAIN_ISLANDS: usize = 5;
/// Wild islands after the main ones. Each runs in a world of its own, a fixed
/// mix of one to three environment effects that is the same in every game
/// (`environment::wild_levels`, `environment::wild_world`), and sends copies
/// of its best to the hub.
pub const WILD_ISLANDS: usize = 100;
/// Of every `SLOT_LANES` slots, `MAIN_LANES` go to the main islands in turn
/// and the rest to the wild islands in turn.
const SLOT_LANES: usize = 10;
const MAIN_LANES: usize = 8;
/// The island of population slot `slot` among `islands`, and the island's
/// own round of slots that slot is in (which sets its kind of arena).
/// With the wild islands, the main islands take 80% of the slots and the
/// wild islands share the rest.
fn home(slot: usize, islands: usize) -> (usize, usize) {
    if islands != MAIN_ISLANDS + WILD_ISLANDS {
        let islands = islands.max(1);
        return (slot % islands, slot / islands);
    }
    let lane = slot % SLOT_LANES;
    let cycle = slot / SLOT_LANES;
    if lane < MAIN_LANES {
        let k = cycle * MAIN_LANES + lane;
        (k % MAIN_ISLANDS, k / MAIN_ISLANDS)
    } else {
        let k = cycle * (SLOT_LANES - MAIN_LANES) + lane - MAIN_LANES;
        (MAIN_ISLANDS + k % WILD_ISLANDS, k / WILD_ISLANDS)
    }
}
/// The island that population slot `slot` breeds for, among `islands`.
pub fn island_of_slot(slot: usize, islands: usize) -> usize {
    home(slot, islands).0
}
/// Whether `island` is a wild island with a world of its own.
pub fn is_wild(island: usize) -> bool {
    island >= MAIN_ISLANDS
}
/// Each island's slots run in cycles of `SLOT_CYCLE` rounds. In every cycle
/// four rounds belong to the island's nursery of new random bodies and two to
/// its nursery of reshaped bodies, and the rest to the island itself: 20%,
/// 10% and 70% of its slots. Half of the nursery's slots hold fresh random
/// bodies (`NURSERY_FRESH_SHARE`), so a tenth of every generation is new
/// random bodies (owner, 2026-10-03).
pub const SLOT_CYCLE: usize = 20;
/// Generations a nursery cohort develops on its own before its survivors
/// enter the island archive.
pub const NURSERY_GENERATIONS: u32 = 10;
/// Share of nursery slots that hold a fresh random body once the nursery has
/// members. The rest breed from the nursery's own members.
pub const NURSERY_FRESH_SHARE: f32 = 0.5;
/// The kinds of archive a slot breeds for and competes in: the island, its
/// nursery of new random bodies, and its nursery of reshaped bodies (bodies
/// that the island turned away: new body plans of its structural and novelty
/// children that took no cell).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arena {
    /// The island's own archive.
    Island,
    /// The island's nursery of new random bodies.
    Nursery,
    /// The island's nursery of reshaped bodies.
    Reshaped,
}
/// How many kinds of archive each island has.
pub const ARENA_KINDS: usize = 3;
/// The kind of archive that population slot `slot` breeds for, among
/// `islands` islands.
pub fn arena_kind_of_slot(slot: usize, islands: usize) -> Arena {
    match home(slot, islands).1 % SLOT_CYCLE {
        0 | 3 | 10 | 13 => Arena::Nursery,
        5 | 15 => Arena::Reshaped,
        _ => Arena::Island,
    }
}
/// Whether `slot` belongs to a nursery of its island (of either kind).
pub fn is_nursery_slot(slot: usize, islands: usize) -> bool {
    arena_kind_of_slot(slot, islands) != Arena::Island
}
/// The archive that population slot `slot` breeds for and competes in, among
/// `arenas`: the islands first, then one nursery of new random bodies per
/// island in the same order, then one nursery of reshaped bodies per island.
pub fn arena_of_slot(slot: usize, arenas: usize) -> usize {
    if arenas < ARENA_KINDS {
        return 0;
    }
    let islands = arenas / ARENA_KINDS;
    let island = island_of_slot(slot, islands);
    match arena_kind_of_slot(slot, islands) {
        Arena::Island => island,
        Arena::Nursery => islands + island,
        Arena::Reshaped => 2 * islands + island,
    }
}
/// Whether archive `arena` of `arenas` is a nursery of reshaped bodies.
pub fn is_reshaped_arena(arena: usize, arenas: usize) -> bool {
    arenas >= ARENA_KINDS && arena >= 2 * (arenas / ARENA_KINDS)
}
/// The niche key of island `island`'s optimizers for gait cadence band
/// `cadence`. Together with the body plan it identifies one optimizer.
pub fn optimizer_niche(island: usize, cadence: u8) -> Niche {
    let b = (island as u32).to_le_bytes();
    Niche([OPTIMIZER_NICHE_MARKER, b[0], b[1], b[2], b[3], cadence])
}
/// Generations a new body plan is protected against a challenger of another
/// plan.
pub const PROTECTION_GENERATIONS: u32 = 3;
/// Developer diagnostic. Whether bit `bit` is set in the number in the
/// `BIO_OFF` environment variable, which is read once and counts as 0 when it
/// is unset or not a number. A set bit turns one biodiversity idea off, so a
/// run can measure the ideas one at a time: 16 is the grace of graduates, 32
/// the mating share of the reshaped nurseries, 64 the second optimizer target
/// and 128 the stepping stones between islands. The switch is temporary.
pub fn bio_off(bit: u32) -> bool {
    static OFF: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("BIO_OFF")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }) & bit
        != 0
}
/// How much a parent of a rare clade is preferred among the parents a
/// tournament of a refined island compares (`Experiment::clade_rarity_of`).
pub const RARITY_WEIGHT: f32 = 1.0;
/// Times a reserve entry must have been a parent before a new body plan may
/// replace it in a full reserve.
pub(crate) const MIN_MORPHOLOGY_DESCENDANTS: u64 = 8;
/// Share of the structural emitter's children whose parent comes from the
/// island's morphology reserve.
pub(crate) const MORPHOLOGY_PARENT_FRACTION: f32 = 0.10;
/// The share of each emitter before any attempts, in `Emitter::ALL` order.
/// `emitter_weights` scales it afterwards. Random bodies only seed an empty
/// archive: against evolved elites they almost never enter it (0.03-0.06% of
/// attempts in fixed-seed tests). Structural children are 62.5%, and 18% of
/// the novelty children also get a structural operator, so two thirds of the
/// bred children (60% of a generation, after the 10% of fresh random bodies)
/// carry a structural mutation and a third only change numbers (owner,
/// 2026-10-03).
const INITIAL_EMITTER_MIX: [f64; EMITTER_COUNT] = [0.145, 0.625, 0.23, 0.0];

/// What a trial measured of a creature's way of moving, as `descriptor` and
/// the archives read it.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TrialMetrics {
    /// Mean share of the nodes on the ground per step, from 0 to 1.
    pub ground_contact: f32,
    /// Spread of the nodes' mean height over the trial, highest minus lowest
    /// (m).
    pub vertical_oscillation: f32,
    /// Up-and-down cycles per second of the nodes' mean height.
    pub gait_frequency: f32,
    /// Mean height of the body's bounding box during the timed trial (m).
    pub mean_height: f32,
    /// Nodes that touched the ground and lifted off again
    /// (`GpuResult::feet`).
    pub feet: f32,
}

/// What one creature's trial gave: its distance, its behavior metrics and how
/// the trial ended. `scheduler::to_metrics` builds it from the kernel's result.
#[derive(Clone, Copy, Debug, Default)]
pub struct EvaluationMetrics {
    /// The distance the creature travelled (m) where the trial ended: at its
    /// end, at a fall or at the screen.
    pub fitness: f32,
    /// The metrics of the trial.
    pub behavior: TrialMetrics,
    /// The early screen stopped the trial (`physics::Screen`): the creature
    /// never enters an archive.
    pub screened: bool,
    /// The result enters no archive: it was measured in a world that has
    /// since changed, or its confirmation trial was stopped by the screen.
    pub excluded: bool,
    /// Distance at the screen, or at an earlier fall (0 when there was no
    /// screen and no earlier fall).
    pub screen_x: f32,
    /// The fitness is the confirmation trial's (it was worse than the
    /// standard trial), so a replay runs at fine fidelity to show that trial.
    pub fine: bool,
    /// The standard trial's rung trace (distances at 1, 2.5, 5 and 10 s and
    /// the early features), for the generation dump.
    pub trace: crate::creature_kernel::RungTrace,
}

/// What the archives know of a creature's body and way of moving: the counts
/// and the start pose from its genes, and the metrics of its trial, clamped to
/// their ranges (`descriptor`).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Descriptor {
    /// Number of nodes.
    pub nodes: u16,
    /// Number of muscles.
    pub muscles: u16,
    /// `TrialMetrics::ground_contact`.
    pub ground_contact: f32,
    /// `TrialMetrics::gait_frequency`, at most 20.
    pub gait_frequency: f32,
    /// The start pose's width over its height, from 1/16 to 16.
    pub aspect_ratio: f32,
    /// `TrialMetrics::vertical_oscillation`.
    pub vertical_oscillation: f32,
    /// `TrialMetrics::mean_height`.
    #[serde(default)]
    pub mean_height: f32,
    /// `TrialMetrics::feet`.
    #[serde(default)]
    pub feet: f32,
}

/// The key of an entry in an archive. A behavior cell holds the bins of ground
/// contact, gait cadence, body shape class, mean height, feet and body size
/// class, in that order. A morphology reserve entry starts with 255 and holds
/// a hash of its body plan. An optimizer's niche starts with 254
/// (`optimizer_niche`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Niche(pub [u8; 6]);

/// Whether `niche` is the niche of a morphology reserve entry, not a behavior
/// cell.
pub fn is_morphology_niche(niche: &Niche) -> bool {
    niche.0[0] == MORPHOLOGY_NICHE_MARKER
}

/// A creature kept in an archive, with the cell it holds, what it scored and
/// the bookkeeping that breeding reads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Elite {
    /// The cell the elite holds, or its reserve niche.
    pub niche: Niche,
    /// The elite's descriptor, from the trial its fitness came from.
    pub descriptor: Descriptor,
    /// The elite's genes.
    pub creature: StoredCreature,
    /// The distance it travelled (m).
    pub fitness: f32,
    /// The emitter that bred it.
    pub emitter: Emitter,
    /// The generation in which it took its cell.
    pub improved_generation: u32,
    /// Before this generation a challenger of another body plan cannot take
    /// its cell (`QdArchive::offer`).
    pub protected_until: u32,
    /// How many times it was chosen as a parent. A faster elite that takes its
    /// cell keeps the count.
    pub visits: u64,
    /// Its body plan.
    pub topology: Topology,
    /// The elite came from a nursery of its island, copied in when the nursery
    /// graduated (`Experiment::graduate_nurseries`).
    #[serde(default)]
    pub graduate: bool,
    /// Its fitness is its confirmation trial's, so its replay runs at fine
    /// fidelity.
    #[serde(default)]
    pub fine: bool,
}
/// The creature and world of the trial a score came from (`Elite::replay_of`).
pub fn replay_of(
    creature: &Creature,
    fine: bool,
    cfg: &crate::config::Config,
) -> (Creature, crate::config::Config) {
    if fine {
        (creature.clone(), crate::scheduler::confirm_config(cfg))
    } else {
        (creature.clone(), cfg.clone())
    }
}
impl Elite {
    /// The creature and world of the trial this elite's fitness came from:
    /// the standard trial, or the confirmation trial at the fine physics.
    pub fn replay_of(&self, cfg: &crate::config::Config) -> (Creature, crate::config::Config) {
        replay_of(&self.creature.unpack(), self.fine, cfg)
    }
}

/// One archive of elites: the behavior elites, one per cell, and a reserve of
/// up to `MORPHOLOGY_LIMIT` entries. The reserve holds the best creature of a
/// body plan that no behavior elite matches in distance, so a new plan keeps
/// breeding until it finds a cell. The islands, their nurseries and the global
/// archive are all archives, and each has its own layout of cells (`Classes`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct QdArchive {
    /// The behavior elites and the reserve entries, in slot order. Code that
    /// changes it directly runs `rebuild_indices` afterwards.
    pub entries: Vec<Elite>,
    /// The slot of the elite in each behavior cell (`EMPTY_CELL` when none),
    /// indexed by `cell_index`. It is empty until the first elite arrives.
    #[serde(skip)]
    cells: Vec<u32>,
    /// The slots of the morphology reserve, by their hashed niches.
    #[serde(skip)]
    reserve_lookup: HashMap<Niche, usize>,
    /// The archive keeps one elite per way of moving and body class. Until
    /// it is refined it keeps one per way of moving, whatever the body: the
    /// bodies of a climbing archive compete for its cells on distance. A
    /// nursery of new random bodies never refines, and a main island's
    /// nursery of reshaped bodies starts refined.
    #[serde(skip)]
    refined: bool,
    /// The global archive has a layout of its own (`GLOBAL_CLASSES`).
    #[serde(skip)]
    global: bool,
    /// The sum of the behavior elites' distances, each counted as at least 0.
    /// The reserve does not count.
    pub qd_score: f64,
    /// The behavior elites visited least, as (visits, slot), fewest first.
    #[serde(skip)]
    least_visited: Vec<(u64, usize)>,
    /// Whether `least_visited` needs a rebuild (`ensure_least_visited`).
    #[serde(skip)]
    least_visited_dirty: bool,
    /// `Topology::plan_key` of every entry, in entry order.
    #[serde(skip)]
    plan_keys: Vec<u64>,
    /// The slots of the behavior elites.
    #[serde(skip)]
    behavior_indices: Vec<usize>,
    /// The slots of the reserve entries.
    #[serde(skip)]
    morphology_indices: Vec<usize>,
    /// The novelty, local competition and frontier of each entry, by slot.
    #[serde(skip)]
    behavior_scores: BehaviorScores,
    /// Cells whose elite changed since the scores were last computed.
    #[serde(skip)]
    changed_cells: Vec<Niche>,
    /// What parent choice reads besides distance, refreshed with the scores.
    #[serde(skip)]
    traits: ParentTraits,
}

/// Per elite, in entry order: how far its body is from the others (body
/// novelty), refreshed once a generation.
#[derive(Clone, Debug, Default)]
struct ParentTraits {
    body_novelty: Vec<f32>,
}

/// A short summary of a body for body novelty, read from its stored genes: node,
/// bone and muscle counts, leaf count and total bone length.
fn body_embedding(c: &StoredCreature) -> [f32; 5] {
    let bones = c.bones();
    let mut has_child = [false; crate::evolution::MAX_NODES];
    for b in bones {
        if let Some(x) = has_child.get_mut(b.a as usize) {
            *x = true;
        }
    }
    let leaves = (1..c.node_count()).filter(|&n| !has_child[n]).count();
    let length: f32 = bones.iter().map(|b| b.rest_length).sum();
    [
        c.node_count() as f32 / 4.0,
        bones.len() as f32 / 4.0,
        c.muscle_count() as f32 / 8.0,
        leaves as f32 / 2.0,
        length / 2.0,
    ]
}

/// What the scores read of an elite: its behavior vector and its distance,
/// packed apart from the big elite records.
#[derive(Clone, Copy)]
struct ScoreRow {
    behavior: [f32; NEIGHBOR_AXES],
    fitness: f32,
}

/// What parent choice reads of the behavior elites, by slot. A vector is
/// shorter than `entries` when the scores do not cover every entry
/// (`QdArchive::scores_current`).
#[derive(Clone, Debug, Default)]
struct BehaviorScores {
    /// The mean behavior distance to the nearest neighbors.
    novelty: Vec<f32>,
    /// The share of the nearest neighbors the elite beats, a tie counting
    /// half.
    local_competition: Vec<f32>,
    /// How open the elite's surroundings are: 1 / (1 + filled neighbor cells
    /// one step away), so a frontier elite scores high.
    frontier: Vec<f32>,
}

/// A body plan: how many nodes a body has and how its parts connect. A bone
/// is an edge between its two nodes, and a muscle is an edge between its two
/// bones, which are numbered after the nodes. The edges are sorted, so the
/// order in which a body lists its bones and muscles does not matter.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Topology {
    /// Number of nodes.
    pub nodes: u8,
    /// The edges as pairs of ids, the smaller first, in sorted order.
    pub edges: Vec<(u32, u32)>,
}

/// The ways a child is bred. `emitter_weights` shares a breeding round among
/// them, and `EmitterStats` tracks how well each one does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Emitter {
    /// Tunes the numbers of a parent. A CMA-ES sampler draws the child when
    /// one is assigned. Otherwise a small random change of the parent's
    /// numbers makes it.
    #[default]
    Cma,
    /// Changes the body plan of a parent with a structural mutation.
    Structural,
    /// Starts from a parent that is far from the others in behavior or in
    /// body and changes its numbers more widely. Sometimes it also applies a
    /// structural mutation.
    Novelty,
    /// A new random body, an immigrant.
    Restart,
}
impl Emitter {
    /// Every kind, in index order.
    pub const ALL: [Self; EMITTER_COUNT] =
        [Self::Cma, Self::Structural, Self::Novelty, Self::Restart];
    /// The position of this kind in `ALL`.
    pub fn index(self) -> usize {
        self as usize
    }
    /// The kind at position `index` of `ALL`. An index past the end gives the
    /// last kind.
    pub fn from_index(index: usize) -> Self {
        Self::ALL[index.min(EMITTER_COUNT - 1)]
    }
    /// The name of this kind in the statistics.
    pub fn label(self) -> &'static str {
        match self {
            Self::Cma => "Diagonal CMA-ES",
            Self::Structural => "Morphology",
            Self::Novelty => "Novelty",
            Self::Restart => "Immigrant",
        }
    }
}

/// How well one emitter has done: the counts and the reward that
/// `emitter_weights` reads.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EmitterStats {
    /// Children counted for this emitter. Children bred for a nursery do not
    /// count.
    pub attempts: u64,
    /// Children that opened a new cell of the global archive or took a new
    /// place in an island's reserve.
    pub discoveries: u64,
    /// Children that beat the elite of a cell of the global archive or the
    /// reserve entry of their body plan.
    pub improvements: u64,
    /// The reward per attempt, as a running average over batches.
    pub reward: f64,
    /// Batches in a row without a discovery or an improvement.
    pub stagnant_batches: u32,
    /// The slot in the global archive of the parent this emitter used last.
    /// Nothing sets a new parent now, so a new game leaves it `None`. After
    /// each batch it follows its creature to the creature's new slot
    /// (`Experiment::archive_block`).
    pub last_parent: Option<usize>,
}

/// A CMA-ES sampler with a diagonal covariance over the numbers of one body
/// plan. A CMA-ME emitter (`new`) improves one cell of an archive in
/// normalized coordinates and ranks its samples by improvement. An optimizer
/// (`optimizer`) works in physical units on an island's fast design and ranks
/// its samples by distance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CmaEmitter {
    /// The cell the emitter improves. For an optimizer it is the key from
    /// `optimizer_niche`.
    pub niche: Niche,
    /// The body plan of the template. Only samples of this plan update the
    /// emitter (`tell`).
    pub topology: Topology,
    /// The island whose elites this emitter samples around. Its children go
    /// only to that island's slots.
    pub island: usize,
    /// The creature the distribution is centered on. A sample keeps its body
    /// and changes only its numbers.
    template: Creature,
    /// The mean of the distribution, in the coordinates of `parameters` (an
    /// optimizer) or of `exploring_parameters` (a CMA-ME emitter).
    mean: Vec<f32>,
    /// The diagonal of the covariance matrix.
    covariance: Vec<f32>,
    /// The evolution path of the covariance update.
    path_c: Vec<f32>,
    /// The evolution path of the step-size update.
    path_sigma: Vec<f32>,
    /// The step size.
    sigma: f32,
    /// The last generation in which the emitter bred. When `CMA_LIMIT`
    /// emitters exist, breeding replaces the one with the lowest value.
    pub last_used_generation: u32,
}

/// What an offer to an archive did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Offer {
    /// The creature entered the archive.
    pub inserted: bool,
    /// It took an empty cell or a new reserve place and replaced no elite.
    pub new_niche: bool,
    /// What the emitter earns: 0 when the creature did not enter, otherwise
    /// from 0.01 to 1.
    pub reward: f64,
}

/// The descriptor of a creature from its genes (`nodes`, `muscles`) and the
/// metrics of its trial.
pub fn descriptor(
    nodes: &[crate::evolution::NodeGene],
    muscles: &[Muscle],
    metrics: TrialMetrics,
) -> Descriptor {
    let (min_x, max_x) = nodes
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), n| {
            (lo.min(n.x), hi.max(n.x))
        });
    let (min_y, max_y) = nodes
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), n| {
            (lo.min(n.y), hi.max(n.y))
        });
    Descriptor {
        nodes: nodes.len() as u16,
        muscles: muscles.len() as u16,
        ground_contact: metrics.ground_contact.clamp(0.0, 1.0),
        gait_frequency: metrics.gait_frequency.clamp(0.0, 20.0),
        aspect_ratio: ((max_x - min_x).max(0.01) / (max_y - min_y).max(0.01)).clamp(0.0625, 16.0),
        vertical_oscillation: metrics.vertical_oscillation.max(0.0),
        mean_height: metrics.mean_height.max(0.0),
        feet: metrics.feet.max(0.0),
    }
}

impl Descriptor {
    /// The cell of this descriptor in the global archive's layout.
    pub fn niche(self) -> Niche {
        self.niche_in(&GLOBAL_CLASSES)
    }
    /// The cell of this descriptor in `classes`.
    pub fn niche_in(self, classes: &Classes) -> Niche {
        let (shape, size) = classes.classes_of(self.aspect_ratio, self.nodes);
        let feet = (self.feet.round() as i32).clamp(1, MOVEMENT_BINS[3] as i32) as u8 - 1;
        Niche([
            bin(self.ground_contact, 0.0, 1.0, MOVEMENT_BINS[0]),
            bin(self.gait_frequency, 0.0, 6.0, MOVEMENT_BINS[1]),
            shape,
            bin(height_axis(self.mean_height), 0.0, 1.0, MOVEMENT_BINS[2]),
            feet,
            size,
        ])
    }

    /// The cell of the way of moving alone, with the body classes left at
    /// zero: the layout of a nursery and of saves before version 54.
    pub fn movement_niche(self) -> Niche {
        let mut niche = self.niche_in(&ISLAND_CLASSES);
        niche.0[2] = 0;
        niche.0[NEIGHBOR_AXES] = 0;
        niche
    }

    /// The behavior vector novelty compares. Shape joins it once the
    /// archive is refined, as an axis of its cells.
    fn behavior(self, refined: bool) -> [f32; NEIGHBOR_AXES] {
        [
            self.ground_contact.clamp(0.0, 1.0),
            (self.gait_frequency / 6.0).clamp(0.0, 1.0),
            // Shape on a log scale from 1:16 to 16:1.
            if refined {
                ((self.aspect_ratio.max(0.0625).ln() / 16f32.ln() + 1.0) * 0.5).clamp(0.0, 1.0)
            } else {
                0.0
            },
            height_axis(self.mean_height),
            ((self.feet - 1.0) / (MOVEMENT_BINS[3] as f32 - 1.0)).clamp(0.0, 1.0),
        ]
    }
}
/// Mean height on a 0..1 log scale from 15 cm to 0.6 times the longest bone
/// the limit allows, so small and large bodies each get their own cells.
fn height_axis(height: f32) -> f32 {
    let low = 0.15f32;
    let high = (0.6 * crate::evolution::max_bone_length()).max(2.0 * low);
    ((height.max(low) / low).ln() / (high / low).ln()).clamp(0.0, 1.0)
}
/// The bin of `value` among `count` equal bins from `low` to `high`. A value
/// outside the range falls in the first or the last bin.
fn bin(value: f32, low: f32, high: f32, count: u8) -> u8 {
    (((value.clamp(low, high) - low) / (high - low) * count as f32).floor() as u8).min(count - 1)
}

/// The sum of squared differences of two behavior vectors.
fn behavior_squares(a: &[f32; NEIGHBOR_AXES], b: &[f32; NEIGHBOR_AXES]) -> f32 {
    (0..a.len()).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>()
}
/// The behavior distance (the root mean square difference) of a sum of
/// squares.
fn distance_of_squares(squares: f32) -> f32 {
    (squares / NEIGHBOR_AXES as f32).sqrt()
}

impl Topology {
    /// The body plan of `creature`.
    pub fn of(creature: &Creature) -> Self {
        topology_from_parts(&creature.nodes, &creature.bones, &creature.muscles)
    }
    /// A 64-bit key of the body plan: two plans share one only by a
    /// collision of the hash. The key sums one hash per edge, so it does not
    /// depend on the order of the parts.
    pub fn plan_key(&self) -> u64 {
        let sum = self
            .edges
            .iter()
            .fold(0u64, |sum, &(a, b)| sum.wrapping_add(edge_key(a, b)));
        plan_key_of(self.nodes as usize, sum)
    }
}

/// The hash of one edge of a body plan (`topology_from_parts`).
fn edge_key(a: u32, b: u32) -> u64 {
    let mut x = ((a as u64) << 32) | b as u64;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 29;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 32)
}
/// The plan key from the node count and the sum of the edge hashes.
fn plan_key_of(nodes: usize, edge_sum: u64) -> u64 {
    let mut key = edge_sum ^ (nodes as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    key ^= key >> 31;
    key = key.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    key ^ (key >> 29)
}
/// `topology_of_population(..).plan_key()` without building the topology.
pub fn plan_key_of_population(population: &Population, index: usize) -> u64 {
    let genome = &population.genomes[index];
    let bones = &population.bones[genome.bone_start..genome.bone_start + genome.bone_count];
    let muscles =
        &population.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count];
    let offset = genome.node_count as u32;
    let mut sum = 0u64;
    for b in bones {
        sum = sum.wrapping_add(edge_key(b.a.min(b.b), b.a.max(b.b)));
    }
    for m in muscles {
        let (a, b) = (offset + m.bone_a, offset + m.bone_b);
        sum = sum.wrapping_add(edge_key(a.min(b), a.max(b)));
    }
    plan_key_of(genome.node_count, sum)
}

/// The body plan of these parts. A muscle's edge joins its two bones, which
/// are numbered after the nodes.
fn topology_from_parts(
    nodes: &[crate::evolution::NodeGene],
    bones: &[crate::evolution::Bone],
    muscles: &[Muscle],
) -> Topology {
    let offset = nodes.len() as u32;
    let mut edges = Vec::with_capacity(bones.len() + muscles.len());
    edges.extend(bones.iter().map(|b| (b.a.min(b.b), b.a.max(b.b))));
    edges.extend(muscles.iter().map(|m| {
        let a = offset + m.bone_a;
        let b = offset + m.bone_b;
        (a.min(b), a.max(b))
    }));
    edges.sort_unstable();
    Topology {
        nodes: nodes.len() as u8,
        edges,
    }
}

fn topology_equivalent(a: &Topology, b: &Topology) -> bool {
    a == b
}

/// Whether two body plans are the same plan, as the archives count them.
pub fn topology_equivalent_for_archive(a: &Topology, b: &Topology) -> bool {
    topology_equivalent(a, b)
}

/// A 64-bit FNV-1a hash of a body plan and `salt`.
fn morphology_hash(topology: &Topology, salt: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut write = |byte: u8| {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    };
    write(topology.nodes);
    for &(a, b) in &topology.edges {
        for byte in a.to_le_bytes().into_iter().chain(b.to_le_bytes()) {
            write(byte);
        }
    }
    for byte in salt.to_le_bytes() {
        write(byte);
    }
    hash
}

/// The reserve niche of a body plan: the marker byte and five bytes of its
/// hash. A different `salt` gives another niche when two plans collide.
fn morphology_niche(topology: &Topology, salt: u64) -> Niche {
    let hash = morphology_hash(topology, salt);
    Niche([
        MORPHOLOGY_NICHE_MARKER,
        hash as u8,
        (hash >> 8) as u8,
        (hash >> 16) as u8,
        (hash >> 24) as u8,
        (hash >> 32) as u8,
    ])
}

/// A slot value for a cell with no elite.
const EMPTY_CELL: u32 = u32::MAX;

/// The way of moving of a behavior niche: its index among the cells of the
/// movement grid, without the body classes.
fn movement_cell(niche: &Niche) -> usize {
    let n = &niche.0;
    ((n[0] as usize * MOVEMENT_BINS[1] as usize + n[1] as usize) * MOVEMENT_BINS[2] as usize
        + n[3] as usize)
        * MOVEMENT_BINS[3] as usize
        + n[4] as usize
}

/// Where a behavior niche sits in a dense table of every cell of a layout
/// with `bins`, or None for a niche outside the grid.
fn cell_index(niche: &Niche, bins: &[u8; 6]) -> Option<usize> {
    let mut index = 0usize;
    for (axis, &count) in bins.iter().enumerate() {
        if niche.0[axis] >= count {
            return None;
        }
        index = index * count as usize + niche.0[axis] as usize;
    }
    Some(index)
}

impl QdArchive {
    /// An empty global archive in the layout a new game starts with.
    pub fn starting_global() -> Self {
        Self {
            global: true,
            refined: true,
            ..Self::default()
        }
    }
    /// Marks the archive as the global one, which has a layout of its own.
    pub fn set_global(&mut self, global: bool) {
        self.global = global;
    }
    /// The body classes of the archive's cells, once it is refined.
    pub fn classes(&self) -> &'static Classes {
        if self.global {
            &GLOBAL_CLASSES
        } else {
            &ISLAND_CLASSES
        }
    }
    /// Cells the archive may fill.
    pub fn limit(&self) -> usize {
        self.classes().cells()
    }
    /// Elites the archive may hold: its cells and the morphology reserve.
    pub fn capacity(&self) -> usize {
        self.limit() + MORPHOLOGY_LIMIT
    }
    /// Whether the archive keeps an elite for each body class of a way of
    /// moving. An island keeps one per way of moving until it is
    /// `REFINE_AFTER` generations old, and the global archive is refined from
    /// the start.
    pub fn refined(&self) -> bool {
        self.refined
    }
    /// Sets the layout. An archive that holds elites needs `rebin` after it.
    pub fn set_refined(&mut self, refined: bool) {
        self.refined = refined;
    }
    /// Sets the layout from the elites it holds: refined when one of them is
    /// in a cell of a body class other than the first.
    pub fn derive_refined(&mut self) {
        self.refined = self.behavior_indices.iter().any(|&i| {
            let niche = &self.entries[i].niche;
            niche.0[2] != 0 || niche.0[NEIGHBOR_AXES] != 0
        });
    }
    /// The cell of `descriptor` in this archive.
    pub fn cell_of(&self, descriptor: Descriptor) -> Niche {
        if self.refined {
            descriptor.niche_in(self.classes())
        } else {
            descriptor.movement_niche()
        }
    }
    fn niche_of(&self, descriptor: Descriptor) -> Niche {
        self.cell_of(descriptor)
    }
    /// The slot of the elite in behavior cell `index`.
    fn cell_slot(&self, index: usize) -> Option<usize> {
        self.cells
            .get(index)
            .filter(|&&slot| slot != EMPTY_CELL)
            .map(|&slot| slot as usize)
    }
    /// Files `slot` under its niche: a behavior cell or a reserve niche.
    fn index_niche(&mut self, niche: &Niche, slot: usize) {
        if is_morphology_niche(niche) {
            self.reserve_lookup.insert(niche.clone(), slot);
        } else if let Some(index) = cell_index(niche, &self.classes().bins()) {
            if self.cells.is_empty() {
                self.cells = vec![EMPTY_CELL; self.limit()];
            }
            self.cells[index] = slot as u32;
        }
    }
    fn unindex_niche(&mut self, niche: &Niche) {
        if is_morphology_niche(niche) {
            self.reserve_lookup.remove(niche);
        } else if let Some(index) = cell_index(niche, &self.classes().bins())
            && let Some(cell) = self.cells.get_mut(index)
        {
            *cell = EMPTY_CELL;
        }
    }
    pub fn rebuild_indices(&mut self) {
        self.cells.clear();
        self.reserve_lookup.clear();
        self.behavior_indices.clear();
        self.morphology_indices.clear();
        self.plan_keys.clear();
        for i in 0..self.entries.len() {
            let niche = self.entries[i].niche.clone();
            self.entries[i].topology.edges.sort_unstable();
            self.plan_keys.push(self.entries[i].topology.plan_key());
            self.index_niche(&niche, i);
            if is_morphology_niche(&niche) {
                self.morphology_indices.push(i);
            } else {
                self.behavior_indices.push(i);
            }
        }
        self.least_visited_dirty = true;
        self.recompute_score();
        self.refresh_behavior_scores();
    }
    /// Adds `elite` to a free cell (or the reserve) and files it in the
    /// indices. Its scores wait for the next `rebuild_indices`, which a
    /// caller that restores many elites runs once at the end.
    pub fn push_unscored(&mut self, elite: Elite) {
        let slot = self.entries.len();
        self.index_niche(&elite.niche, slot);
        self.plan_keys.push(elite.topology.plan_key());
        if is_morphology_niche(&elite.niche) {
            self.morphology_indices.push(slot);
        } else {
            self.behavior_indices.push(slot);
        }
        self.entries.push(elite);
        self.least_visited_dirty = true;
        self.behavior_scores = BehaviorScores::default();
    }
    /// Moves every behavior elite to the cell its descriptor gives under the
    /// current layout, for archives saved under an older one. When two
    /// elites meet in a cell the faster stays. Elites keep their order.
    pub fn rebin(&mut self) {
        let mut kept: Vec<Elite> = Vec::with_capacity(self.entries.len());
        let mut at: HashMap<Niche, usize> = HashMap::new();
        for mut elite in std::mem::take(&mut self.entries) {
            if !is_morphology_niche(&elite.niche) {
                elite.niche = self.niche_of(elite.descriptor);
                if let Some(&slot) = at.get(&elite.niche) {
                    if elite.fitness > kept[slot].fitness {
                        kept[slot] = elite;
                    }
                    continue;
                }
                at.insert(elite.niche.clone(), kept.len());
            }
            kept.push(elite);
        }
        self.entries = kept;
        self.rebuild_indices();
    }
    pub fn best_fitness(&self) -> f32 {
        self.entries
            .iter()
            .map(|e| e.fitness)
            .fold(f32::NEG_INFINITY, f32::max)
    }
    pub fn behavior_count(&self) -> usize {
        self.behavior_indices.len()
    }
    /// The body plan key of the elite in `slot` (`Topology::plan_key`).
    pub fn plan_key(&self, slot: usize) -> u64 {
        self.plan_keys[slot]
    }
    pub(crate) fn slot_for(&self, niche: &Niche) -> Option<usize> {
        if is_morphology_niche(niche) {
            self.reserve_lookup.get(niche).copied()
        } else {
            cell_index(niche, &self.classes().bins()).and_then(|index| self.cell_slot(index))
        }
    }
    pub fn morphology_count(&self) -> usize {
        self.morphology_indices.len()
    }
    /// Fitness a new topology must beat to enter the full topology reserve,
    /// or `None` while the reserve has room.
    pub(crate) fn morphology_floor(&self) -> Option<f32> {
        (self.morphology_count() >= MORPHOLOGY_LIMIT).then(|| {
            self.morphology_indices
                .iter()
                .map(|&i| &self.entries[i])
                .filter(|elite| elite.visits >= MIN_MORPHOLOGY_DESCENDANTS)
                .map(|elite| elite.fitness)
                .fold(f32::INFINITY, f32::min)
        })
    }
    /// The ways of moving the elites cover, counting cells without their
    /// body classes: the shape and size classes split a way of moving among
    /// bodies, and it is still one way of moving.
    pub fn movement_count(&self) -> usize {
        let mut seen = vec![false; MOVEMENT_CELLS];
        let mut count = 0;
        for &i in &self.behavior_indices {
            let cell = movement_cell(&self.entries[i].niche);
            if !std::mem::replace(&mut seen[cell], true) {
                count += 1;
            }
        }
        count
    }
    /// The slots of the fastest elite of each way of moving, whatever its
    /// body class: the statistics of distance read these, so they mean the
    /// same for an archive with one elite per way of moving and for one with
    /// body classes.
    pub fn best_per_way_of_moving(&self) -> Vec<usize> {
        let mut best: Vec<Option<usize>> = vec![None; MOVEMENT_CELLS];
        for &i in &self.behavior_indices {
            let slot = &mut best[movement_cell(&self.entries[i].niche)];
            if slot.is_none_or(|j| self.entries[i].fitness > self.entries[j].fitness) {
                *slot = Some(i);
            }
        }
        best.into_iter().flatten().collect()
    }
    /// The share of the ways of moving that some elite covers.
    pub fn coverage(&self) -> f32 {
        self.movement_count() as f32 / MOVEMENT_CELLS as f32
    }
    pub fn sample_novel(&self, rng: &mut Rng, avoid: Option<usize>) -> Option<usize> {
        if self.behavior_count() == 0 {
            return None;
        }
        let mut selected = None;
        let mut best_score = f32::NEG_INFINITY;
        let mut tied = 0usize;
        for &(_, index) in &self.least_visited {
            if Some(index) == avoid && self.behavior_count() > 1 {
                continue;
            }
            let novelty = self
                .behavior_scores
                .novelty
                .get(index)
                .copied()
                .unwrap_or(0.0);
            let visits = self.entries[index].visits;
            // Frontier parents first: an elite with empty cells around it
            // can open them (Lehman and Stanley, 2011).
            let frontier = self
                .behavior_scores
                .frontier
                .get(index)
                .copied()
                .unwrap_or(0.0);
            let score = novelty + 0.08 / (1.0 + visits as f32).sqrt() + 0.05 * frontier;
            if score > best_score {
                selected = Some(index);
                best_score = score;
                tied = 1;
            } else if (score - best_score).abs() <= f32::EPSILON {
                tied += 1;
                if rng.index(tied) == 0 {
                    selected = Some(index);
                }
            }
        }
        selected.or_else(|| {
            (!self.behavior_indices.is_empty())
                .then(|| self.behavior_indices[rng.index(self.behavior_indices.len())])
        })
    }
    pub fn sample_local_competitive(
        &self,
        rng: &mut Rng,
        avoid: Option<usize>,
        rarity: &[f32],
    ) -> Option<usize> {
        let behavior = &self.behavior_indices;
        if behavior.is_empty() {
            return None;
        }
        let candidates = behavior.len().clamp(1, 8);
        let mut selected = None;
        let mut best_score = f32::NEG_INFINITY;
        for _ in 0..candidates {
            let mut index = behavior[rng.index(behavior.len())];
            if Some(index) == avoid && behavior.len() > 1 {
                let ordinal = behavior
                    .iter()
                    .position(|&candidate| candidate == index)
                    .unwrap_or(0);
                index = behavior[(ordinal + 1 + rng.index(behavior.len() - 1)) % behavior.len()];
            }
            let local = self
                .behavior_scores
                .local_competition
                .get(index)
                .copied()
                .unwrap_or(0.5);
            let bonus = RARITY_WEIGHT * rarity.get(index).copied().unwrap_or(0.0);
            let score = local + rng.unit() * 0.02 + bonus;
            if score > best_score {
                selected = Some(index);
                best_score = score;
            }
        }
        selected
    }
    pub fn sample_morphology(&self, rng: &mut Rng, avoid: Option<usize>) -> Option<usize> {
        let least_visits = self
            .morphology_indices
            .iter()
            .filter(|&&index| Some(index) != avoid)
            .map(|&index| self.entries[index].visits)
            .min()?;
        let mut selected = None;
        let mut tied = 0usize;
        for &index in &self.morphology_indices {
            if Some(index) == avoid || self.entries[index].visits != least_visits {
                continue;
            }
            tied += 1;
            if rng.index(tied) == 0 {
                selected = Some(index);
            }
        }
        selected
    }
    /// Novelty (mean distance to the nearest archived behaviors) and local
    /// competition (share of those neighbors this elite beats), found through
    /// adjacent grid cells instead of comparing every pair.
    /// Whether the cached novelty and local-competition scores cover every
    /// elite (false after anything reset or grew the archive).
    pub fn scores_current(&self) -> bool {
        self.behavior_scores.novelty.len() == self.entries.len()
            && self.behavior_scores.local_competition.len() == self.entries.len()
    }
    /// Calls `visit` with the slot of the elite in every filled cell within
    /// `radius` grid steps of `center` along the five behavior axes, in the
    /// same node-count class (the cell itself excluded).
    fn for_each_neighbor(&self, center: &Niche, radius: i32, mut visit: impl FnMut(usize)) {
        const _: () = assert!(NEIGHBOR_AXES == 5);
        let bins = self.classes().bins();
        let Some(own) = cell_index(center, &bins) else {
            return;
        };
        if self.cells.is_empty() {
            return;
        }
        let range = |axis: usize| {
            let at = center.0[axis] as i32;
            (at - radius).max(0) as usize..=(at + radius).min(bins[axis] as i32 - 1) as usize
        };
        let (b1, b2, b3, b4, b5) = (
            bins[1] as usize,
            bins[2] as usize,
            bins[3] as usize,
            bins[4] as usize,
            bins[5] as usize,
        );
        let class = center.0[NEIGHBOR_AXES] as usize;
        for a0 in range(0) {
            for a1 in range(1) {
                for a2 in range(2) {
                    for a3 in range(3) {
                        for a4 in range(4) {
                            let index =
                                ((((a0 * b1 + a1) * b2 + a2) * b3 + a3) * b4 + a4) * b5 + class;
                            if index != own {
                                let slot = self.cells[index];
                                if slot != EMPTY_CELL {
                                    visit(slot as usize);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /// For each row of behavior cells that differ only along the last
    /// behavior axis, a bit per filled cell of the row, so a neighbor search
    /// skips a row's empty cells at once. Row `r` holds the cells
    /// `(r / b5 * b4 + a4) * b5 + r % b5`, with `b4` and `b5` the counts of
    /// the last two axes.
    fn filled_rows(&self) -> Vec<u8> {
        const _: () = assert!(MOVEMENT_BINS[3] <= 8);
        let bins = self.classes().bins();
        let (b4, b5) = (bins[4] as usize, bins[5] as usize);
        let mut rows = vec![0u8; self.cells.len() / b4];
        for (index, &slot) in self.cells.iter().enumerate() {
            if slot != EMPTY_CELL {
                rows[index / (b4 * b5) * b5 + index % b5] |= 1 << (index / b5 % b4);
            }
        }
        rows
    }
    /// `for_each_neighbor` row by row through `filled_rows`: the same elites
    /// in the same order, without a look at each empty cell.
    fn for_each_filled_neighbor(
        &self,
        filled: &[u8],
        center: &Niche,
        radius: i32,
        mut visit: impl FnMut(usize),
    ) {
        let bins = self.classes().bins();
        if cell_index(center, &bins).is_none() || self.cells.is_empty() {
            return;
        }
        let range = |axis: usize| {
            let at = center.0[axis] as i32;
            (at - radius).max(0) as usize..=(at + radius).min(bins[axis] as i32 - 1) as usize
        };
        let (b1, b2, b3, b4, b5) = (
            bins[1] as usize,
            bins[2] as usize,
            bins[3] as usize,
            bins[4] as usize,
            bins[5] as usize,
        );
        let class = center.0[NEIGHBOR_AXES] as usize;
        let last = range(4);
        let span = ((1u16 << (last.end() + 1)) - (1u16 << last.start())) as u8;
        let c = &center.0;
        let own = ((c[0] as usize * b1 + c[1] as usize) * b2 + c[2] as usize) * b3 + c[3] as usize;
        for a0 in range(0) {
            for a1 in range(1) {
                for a2 in range(2) {
                    for a3 in range(3) {
                        let outer = ((a0 * b1 + a1) * b2 + a2) * b3 + a3;
                        let mut bits = filled[outer * b5 + class] & span;
                        if outer == own {
                            bits &= !(1 << c[4]);
                        }
                        while bits != 0 {
                            let a4 = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            visit(self.cells[(outer * b4 + a4) * b5 + class] as usize);
                        }
                    }
                }
            }
        }
    }
    /// Novelty and local competition of the behavior elite at `index`, from
    /// the elites in the cells around it. `rows` holds every elite's
    /// behavior vector and distance, `filled` the archive's `filled_rows`.
    fn behavior_score_of(
        &self,
        index: usize,
        rows: &[ScoreRow],
        filled: &[u8],
    ) -> (usize, f32, f32, f32) {
        // The nearest `LOCAL_NEIGHBORS` neighbors, closest first, as
        // (distance, fitness, sum of squares), and how many neighbors there
        // were.
        let mut nearest = [(f32::INFINITY, 0.0f32, f32::INFINITY); LOCAL_NEIGHBORS];
        let mut found = 0usize;
        let mut near = 0usize;
        let own = &rows[index];
        for radius in 1..=2 {
            nearest.fill((f32::INFINITY, 0.0, f32::INFINITY));
            found = 0;
            self.for_each_filled_neighbor(filled, &self.entries[index].niche, radius, |slot| {
                found += 1;
                let other = &rows[slot];
                let squares = behavior_squares(&own.behavior, &other.behavior);
                // The mean and the root keep order, so a sum of squares no
                // smaller than the farthest kept one's is no closer: the
                // test below would turn it down.
                if squares >= nearest[LOCAL_NEIGHBORS - 1].2 {
                    return;
                }
                let distance = distance_of_squares(squares);
                if distance < nearest[LOCAL_NEIGHBORS - 1].0 {
                    let mut at = LOCAL_NEIGHBORS - 1;
                    while at > 0 && nearest[at - 1].0 > distance {
                        nearest[at] = nearest[at - 1];
                        at -= 1;
                    }
                    nearest[at] = (distance, other.fitness, squares);
                }
            });
            if radius == 1 {
                near = found;
            }
            if found >= LOCAL_NEIGHBORS {
                break;
            }
        }
        let frontier = 1.0 / (1.0 + near as f32);
        if found == 0 {
            return (index, 1.0, 1.0, frontier);
        }
        let nearest = &nearest[..LOCAL_NEIGHBORS.min(found)];
        let novelty = nearest.iter().map(|(d, _, _)| *d).sum::<f32>() / nearest.len() as f32;
        let local = nearest
            .iter()
            .map(|(_, f, _)| match own.fitness.total_cmp(f) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Less => 0.0,
            })
            .sum::<f32>()
            / nearest.len() as f32;
        (index, novelty, local, frontier)
    }
    /// Records that the elite of `niche` changed. While the cached scores
    /// still cover every elite they stay, and the next refresh recomputes
    /// only the elites near that cell. A new elite is pushed before this
    /// runs, so the scores no longer cover it and are wiped, and the samplers
    /// read no scores until the next refresh. The reshaped nurseries breed
    /// between refreshes; keeping their scores cost clades
    /// (`docs/rejected-ideas.md`).
    fn note_changed_cell(&mut self, niche: Niche) {
        if self.scores_current() {
            let len = self.entries.len();
            self.behavior_scores.novelty.resize(len, 0.0);
            self.behavior_scores.local_competition.resize(len, 0.5);
            self.behavior_scores.frontier.resize(len, 0.0);
            self.changed_cells.push(niche);
        } else {
            self.behavior_scores = BehaviorScores::default();
            self.changed_cells.clear();
        }
    }
    pub fn refresh_behavior_scores(&mut self) {
        use rayon::prelude::*;
        // The global archive is never a parent source, so nobody reads its
        // scores.
        if self.global {
            self.changed_cells.clear();
            return;
        }
        let changed = std::mem::take(&mut self.changed_cells);
        if self.behavior_indices.is_empty() {
            self.behavior_scores = BehaviorScores::default();
            self.traits = ParentTraits::default();
            return;
        }
        // Only elites within two cells of a changed cell can see a difference.
        let partial = self.scores_current() && changed.len() <= 32;
        let indices: Vec<usize> = if partial {
            self.behavior_indices
                .iter()
                .copied()
                .filter(|&i| {
                    let niche = &self.entries[i].niche;
                    changed.iter().any(|c| {
                        niche.0[NEIGHBOR_AXES] == c.0[NEIGHBOR_AXES]
                            && (0..NEIGHBOR_AXES)
                                .all(|axis| (niche.0[axis] as i32 - c.0[axis] as i32).abs() <= 2)
                    })
                })
                .collect()
        } else {
            self.behavior_indices.clone()
        };
        let rows: Vec<ScoreRow> = self
            .entries
            .par_iter()
            .map(|elite| ScoreRow {
                behavior: elite.descriptor.behavior(self.refined),
                fitness: elite.fitness,
            })
            .collect();
        let filled = self.filled_rows();
        let scores: Vec<(usize, f32, f32, f32)> = indices
            .par_iter()
            .map(|&index| self.behavior_score_of(index, &rows, &filled))
            .collect();
        let len = self.entries.len();
        let (mut novelty, mut local_competition, mut frontier) = if partial {
            let mut f = std::mem::take(&mut self.behavior_scores.frontier);
            f.resize(len, 0.0);
            (
                std::mem::take(&mut self.behavior_scores.novelty),
                std::mem::take(&mut self.behavior_scores.local_competition),
                f,
            )
        } else {
            (vec![0.0; len], vec![0.5; len], vec![0.0; len])
        };
        for (index, n, l, f) in scores {
            novelty[index] = n;
            local_competition[index] = l;
            frontier[index] = f;
        }
        self.behavior_scores = BehaviorScores {
            novelty,
            local_competition,
            frontier,
        };
    }
    /// Recomputes the body novelty: the mean distance of a body summary to
    /// 32 other elites spread evenly over the archive.
    pub fn refresh_traits(&mut self) {
        let len = self.entries.len();
        let behavior = &self.behavior_indices;
        let bodies: Vec<[f32; 5]> = behavior
            .iter()
            .map(|&i| body_embedding(&self.entries[i].creature))
            .collect();
        let samples = bodies.len().min(32);
        let mut body_novelty = vec![0.0; len];
        for (k, &i) in behavior.iter().enumerate() {
            let mut sum = 0.0;
            for s in 0..samples {
                let other = &bodies[(k + 1 + s * bodies.len() / samples.max(1)) % bodies.len()];
                sum += bodies[k]
                    .iter()
                    .zip(other)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
            }
            body_novelty[i] = sum / samples.max(1) as f32;
        }
        self.traits = ParentTraits { body_novelty };
    }
    /// The behavior elite whose body is farthest from the others (highest
    /// body novelty), with that novelty.
    pub fn strangest(&self) -> Option<(f32, &Elite)> {
        self.behavior_indices
            .iter()
            .filter_map(|&i| Some((*self.traits.body_novelty.get(i)?, &self.entries[i])))
            .max_by(|a, b| a.0.total_cmp(&b.0))
    }
    /// A parent far from the other bodies in body summary (body novelty, as
    /// novelty search over morphology, Lehman and Stanley 2011): the best of
    /// 8 behavior elites drawn at random.
    pub fn sample_body_novel(&self, rng: &mut Rng) -> Option<usize> {
        let behavior = &self.behavior_indices;
        if behavior.is_empty() || self.traits.body_novelty.is_empty() {
            return None;
        }
        let novelty = |i: usize| self.traits.body_novelty.get(i).copied().unwrap_or(0.0);
        (0..8)
            .map(|_| behavior[rng.index(behavior.len())])
            .max_by(|&a, &b| novelty(a).total_cmp(&novelty(b)))
    }
    pub fn visit(&mut self, index: usize) {
        self.entries[index].visits += 1;
        self.least_visited_dirty = true;
    }
    /// Rebuilds the least-visited index after batched visits. Consumers of
    /// `least_visited` must call this first; rebuilding once per breeding round
    /// is much cheaper than a balanced-tree update per visit.
    pub fn ensure_least_visited(&mut self) {
        if !self.least_visited_dirty {
            return;
        }
        // The 32 behavior elites with the fewest visits, the lower slot first
        // on a tie.
        let mut all: Vec<(u64, usize)> = self
            .behavior_indices
            .iter()
            .map(|&index| (self.entries[index].visits, index))
            .collect();
        if all.len() > LEAST_VISITED {
            all.select_nth_unstable(LEAST_VISITED - 1);
            all.truncate(LEAST_VISITED);
        }
        all.sort_unstable();
        self.least_visited = all;
        self.least_visited_dirty = false;
    }
    #[allow(clippy::too_many_arguments)]
    pub fn offer(
        &mut self,
        population: &Population,
        index: usize,
        descriptor: Descriptor,
        fitness: f32,
        fine: bool,
        emitter: Emitter,
        generation: u32,
        protected_until: u32,
    ) -> Offer {
        if !fitness.is_finite() || fitness <= crate::evolution::FAILED {
            return Offer::default();
        }
        let niche = self.niche_of(descriptor);
        if let Some(slot) = self.slot_for(&niche) {
            let current = &self.entries[slot];
            if fitness <= current.fitness {
                return Offer::default();
            }
            let candidate_topology = topology_of_population(population, index);
            if generation < current.protected_until && candidate_topology != current.topology {
                return Offer::default();
            }
            let delta = fitness - current.fitness;
            let previous_fitness = current.fitness;
            let visits = current.visits;
            let old_protection = current.protected_until;
            let local_competition = self.local_competition_for(&niche, fitness);
            let elite = &mut self.entries[slot];
            *elite = Elite {
                niche,
                descriptor,
                creature: population.stored(index),
                fitness,
                emitter,
                improved_generation: generation,
                protected_until: protected_until.max(old_protection),
                visits,
                topology: candidate_topology.clone(),
                graduate: false,
                fine,
            };
            self.plan_keys[slot] = candidate_topology.plan_key();
            self.qd_score += fitness.max(0.0) as f64 - previous_fitness.max(0.0) as f64;
            self.note_changed_cell(self.entries[slot].niche.clone());
            self.remove_morphology_topology(&candidate_topology, fitness);
            return Offer {
                inserted: true,
                new_niche: false,
                reward: ((delta as f64 / (1.0 + previous_fitness.abs() as f64))
                    * (0.5 + local_competition as f64))
                    .clamp(0.01, 1.0),
            };
        }
        if self.behavior_count() >= self.limit() {
            return Offer::default();
        }
        let local_competition = self.local_competition_for(&niche, fitness);
        let topology = topology_of_population(population, index);
        self.qd_score += fitness.max(0.0) as f64;
        self.plan_keys.push(topology.plan_key());
        self.entries.push(Elite {
            niche: niche.clone(),
            descriptor,
            creature: population.stored(index),
            fitness,
            emitter,
            improved_generation: generation,
            protected_until,
            visits: 0,
            topology: topology.clone(),
            graduate: false,
            fine,
        });
        let slot = self.entries.len() - 1;
        self.index_niche(&niche, slot);
        self.least_visited_dirty = true;
        self.behavior_indices.push(slot);
        self.note_changed_cell(self.entries[slot].niche.clone());
        self.remove_morphology_topology(&topology, fitness);
        Offer {
            inserted: true,
            new_niche: true,
            reward: 0.5 + local_competition as f64 * 0.5,
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn offer_morphology(
        &mut self,
        population: &Population,
        index: usize,
        descriptor: Descriptor,
        topology: Topology,
        fitness: f32,
        fine: bool,
        emitter: Emitter,
        generation: u32,
        protected_until: u32,
    ) -> Offer {
        if !fitness.is_finite() || fitness <= crate::evolution::FAILED {
            return Offer::default();
        }
        let behavior_best = self
            .behavior_indices
            .iter()
            .map(|&i| &self.entries[i])
            .filter(|elite| topology_equivalent(&topology, &elite.topology))
            .map(|elite| elite.fitness)
            .max_by(f32::total_cmp);
        if behavior_best.is_some_and(|best| fitness <= best) {
            return Offer::default();
        }
        if let Some(slot) = self
            .morphology_indices
            .iter()
            .copied()
            .find(|&i| topology_equivalent(&topology, &self.entries[i].topology))
        {
            let current = &self.entries[slot];
            if fitness <= current.fitness {
                return Offer::default();
            }
            let previous_fitness = current.fitness;
            let visits = current.visits;
            let niche = current.niche.clone();
            self.entries[slot] = Elite {
                niche,
                descriptor,
                creature: population.stored(index),
                fitness,
                emitter,
                improved_generation: generation,
                protected_until: protected_until.max(current.protected_until),
                visits,
                topology,
                graduate: false,
                fine,
            };
            return Offer {
                inserted: true,
                new_niche: false,
                reward: ((fitness - previous_fitness) as f64
                    / (1.0 + previous_fitness.abs() as f64))
                    .clamp(0.01, 1.0),
            };
        }

        if self.morphology_count() >= MORPHOLOGY_LIMIT {
            let victim = self
                .morphology_indices
                .iter()
                .map(|&i| (i, &self.entries[i]))
                .filter(|(_, elite)| elite.visits >= MIN_MORPHOLOGY_DESCENDANTS)
                .min_by(|(_, a), (_, b)| a.fitness.total_cmp(&b.fitness));
            let Some((slot, elite)) = victim else {
                return Offer::default();
            };
            if fitness <= elite.fitness {
                return Offer::default();
            }
            self.remove_entry(slot);
        }
        let mut salt = 0u64;
        let niche = loop {
            let candidate = morphology_niche(&topology, salt);
            match self.reserve_lookup.get(&candidate).copied() {
                None => break candidate,
                Some(slot) if topology_equivalent(&topology, &self.entries[slot].topology) => {
                    return Offer::default();
                }
                Some(_) => salt = salt.wrapping_add(1),
            }
        };
        let elite = Elite {
            niche: niche.clone(),
            descriptor,
            creature: population.stored(index),
            fitness,
            emitter,
            improved_generation: generation,
            protected_until,
            visits: 0,
            topology,
            graduate: false,
            fine,
        };
        self.plan_keys.push(elite.topology.plan_key());
        self.entries.push(elite);
        let slot = self.entries.len() - 1;
        self.index_niche(&niche, slot);
        self.morphology_indices.push(slot);
        Offer {
            inserted: true,
            new_niche: true,
            reward: 1.0,
        }
    }
    /// Adds a copy of `elite` if its behavior niche is empty or it beats the
    /// occupant. Used for island migration and for a nursery's graduation;
    /// the copy takes its cell in this archive's layout. Returns whether it
    /// was kept.
    pub fn absorb(&mut self, elite: &Elite) -> bool {
        if is_morphology_niche(&elite.niche) {
            return false;
        }
        let niche = self.niche_of(elite.descriptor);
        if let Some(slot) = self.slot_for(&niche) {
            let current = &self.entries[slot];
            if elite.fitness <= current.fitness {
                return false;
            }
            self.qd_score += elite.fitness.max(0.0) as f64 - current.fitness.max(0.0) as f64;
            let visits = current.visits;
            self.entries[slot] = Elite {
                niche,
                visits,
                ..elite.clone()
            };
            self.plan_keys[slot] = elite.topology.plan_key();
        } else {
            if self.behavior_count() >= self.limit() {
                return false;
            }
            self.qd_score += elite.fitness.max(0.0) as f64;
            self.plan_keys.push(elite.topology.plan_key());
            self.entries.push(Elite {
                niche: niche.clone(),
                visits: 0,
                ..elite.clone()
            });
            let slot = self.entries.len() - 1;
            self.index_niche(&niche, slot);
            self.least_visited_dirty = true;
            self.behavior_indices.push(slot);
        }
        self.behavior_scores = BehaviorScores::default();
        true
    }
    /// Whether `absorb` would keep `elite`: its cell in this archive's
    /// layout is empty or holds a slower elite.
    pub fn would_take(&self, elite: &Elite) -> bool {
        if is_morphology_niche(&elite.niche) {
            return false;
        }
        match self.slot_for(&self.niche_of(elite.descriptor)) {
            Some(slot) => elite.fitness > self.entries[slot].fitness,
            None => self.behavior_count() < self.limit(),
        }
    }
    fn remove_morphology_topology(&mut self, topology: &Topology, behavior_fitness: f32) {
        if let Some(slot) = self.morphology_indices.iter().copied().find(|&i| {
            let elite = &self.entries[i];
            elite.fitness <= behavior_fitness && topology_equivalent(topology, &elite.topology)
        }) {
            self.remove_entry(slot);
        }
    }
    fn remove_entry(&mut self, slot: usize) {
        let last = self.entries.len() - 1;
        self.least_visited_dirty = true;
        let removed_niche = self.entries[slot].niche.clone();
        self.unindex_niche(&removed_niche);
        let removed = &self.entries[slot];
        let removed_is_morphology = is_morphology_niche(&removed.niche);
        let indices = if removed_is_morphology {
            &mut self.morphology_indices
        } else {
            &mut self.behavior_indices
        };
        if let Some(index) = indices.iter().position(|&entry| entry == slot) {
            indices.swap_remove(index);
        }
        if slot != last {
            let moved = &self.entries[last];
            let moved_niche = moved.niche.clone();
            let moved_is_morphology = is_morphology_niche(&moved.niche);
            self.entries.swap_remove(slot);
            self.plan_keys.swap_remove(slot);
            self.index_niche(&moved_niche, slot);
            let indices = if moved_is_morphology {
                &mut self.morphology_indices
            } else {
                &mut self.behavior_indices
            };
            if let Some(index) = indices.iter().position(|&entry| entry == last) {
                indices[index] = slot;
            }
        } else {
            self.entries.pop();
            self.plan_keys.pop();
        }
        if removed_is_morphology && self.scores_current_before_removal(last + 1) {
            // A reserve entry takes no behavior cell, so no behavior score
            // changes; the scores follow the entry that moved into its place.
            if slot != last {
                self.behavior_scores.novelty.swap_remove(slot);
                self.behavior_scores.local_competition.swap_remove(slot);
            } else {
                self.behavior_scores.novelty.pop();
                self.behavior_scores.local_competition.pop();
            }
        } else {
            self.behavior_scores = BehaviorScores::default();
        }
    }
    /// Whether the cached scores covered `entries` entries.
    fn scores_current_before_removal(&self, entries: usize) -> bool {
        self.behavior_scores.novelty.len() == entries
            && self.behavior_scores.local_competition.len() == entries
    }
    fn local_competition_for(&self, niche: &Niche, fitness: f32) -> f32 {
        let mut compared = 0usize;
        let mut wins = 0.0f32;
        self.for_each_neighbor(niche, 1, |slot| {
            compared += 1;
            wins += match fitness.total_cmp(&self.entries[slot].fitness) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Less => 0.0,
            };
        });
        if compared == 0 {
            0.5
        } else {
            wins / compared as f32
        }
    }

    fn recompute_score(&mut self) {
        self.qd_score = self
            .entries
            .iter()
            .filter(|elite| !is_morphology_niche(&elite.niche))
            .map(|elite| elite.fitness.max(0.0) as f64)
            .sum();
    }
}
pub fn topology_of_population(population: &Population, index: usize) -> Topology {
    let genome = &population.genomes[index];
    let nodes = &population.nodes[genome.node_start..genome.node_start + genome.node_count];
    let bones = &population.bones[genome.bone_start..genome.bone_start + genome.bone_count];
    let muscles =
        &population.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count];
    topology_from_parts(nodes, bones, muscles)
}

impl EmitterStats {
    pub fn stale(&self) -> bool {
        self.stagnant_batches >= 5
    }
}

pub fn emitter_weights(stats: &[EmitterStats; EMITTER_COUNT]) -> [f64; EMITTER_COUNT] {
    if stats.iter().all(|s| s.attempts == 0) {
        return INITIAL_EMITTER_MIX;
    }
    let total = stats.iter().map(|s| s.attempts).sum::<u64>().max(1) as f64;
    let mut weights = [0.0; EMITTER_COUNT];
    for i in 0..EMITTER_COUNT {
        let prior = INITIAL_EMITTER_MIX[i];
        let mean = if stats[i].attempts == 0 {
            0.5
        } else {
            stats[i].reward
        };
        let exploration = 0.35 * (total.ln_1p() / (stats[i].attempts as f64 + 1.0)).sqrt();
        // Keep half of the prior allocation as an exploration floor while the
        // other half follows archive discoveries and improvements.
        weights[i] = prior * (0.5 + 0.05 + mean + exploration);
    }
    let sum = weights.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    weights.map(|w| w / sum)
}
pub fn choose_emitter(rng: &mut Rng, weights: &[f64; EMITTER_COUNT]) -> Emitter {
    let draw = rng.unit() as f64;
    let mut total = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        total += weight;
        if draw <= total {
            return Emitter::from_index(index);
        }
    }
    Emitter::Restart
}
pub fn record_emitter_batch(
    stats: &mut [EmitterStats; EMITTER_COUNT],
    attempts: &[u64; EMITTER_COUNT],
    discoveries: &[u64; EMITTER_COUNT],
    improvements: &[u64; EMITTER_COUNT],
    rewards: &[f64; EMITTER_COUNT],
) {
    for i in 0..EMITTER_COUNT {
        let s = &mut stats[i];
        s.attempts += attempts[i];
        s.discoveries += discoveries[i];
        s.improvements += improvements[i];
        if attempts[i] > 0 {
            let batch_rate = rewards[i] / attempts[i] as f64;
            s.reward = 0.75 * s.reward + 0.25 * batch_rate;
        }
        if improvements[i] == 0 && discoveries[i] == 0 {
            s.stagnant_batches = s.stagnant_batches.saturating_add(1);
        } else {
            s.stagnant_batches = 0;
        }
    }
}

impl CmaEmitter {
    /// A CMA-ME emitter improving one niche of the archive.
    pub fn new(template: Creature, niche: Niche, generation: u32) -> Self {
        let mean = exploring_parameters(&template);
        let dimensions = mean.len();
        Self {
            niche,
            topology: Topology::of(&template),
            island: 0,
            template,
            mean,
            covariance: vec![1.0; dimensions],
            path_c: vec![0.0; dimensions],
            path_sigma: vec![0.0; dimensions],
            sigma: 0.12,
            last_used_generation: generation,
        }
    }
    /// A separable CMA-ES optimizer for one island's body plan, starting from
    /// a fast elite: it searches in physical units and ranks samples by
    /// fitness alone.
    pub fn optimizer(template: Creature, niche: Niche, generation: u32) -> Self {
        let mean = parameters(&template);
        let dimensions = mean.len();
        Self {
            niche,
            topology: Topology::of(&template),
            island: 0,
            template,
            mean,
            covariance: vec![1.0; dimensions],
            path_c: vec![0.0; dimensions],
            path_sigma: vec![0.0; dimensions],
            sigma: 0.5,
            last_used_generation: generation,
        }
    }
    /// A new optimizer on this one's island, starting from `template`, that
    /// keeps the step sizes this one has learned for the same body plan.
    pub fn recentered(&self, template: Creature, niche: Niche, generation: u32) -> Self {
        let mut next = Self::optimizer(template, niche, generation);
        next.island = self.island;
        if next.topology == self.topology && next.mean.len() == self.mean.len() {
            next.covariance.clone_from(&self.covariance);
            next.path_c.clone_from(&self.path_c);
            next.path_sigma.clone_from(&self.path_sigma);
            next.sigma = self.sigma;
        }
        next
    }
    pub fn optimizing(&self) -> bool {
        self.niche.0[0] == OPTIMIZER_NICHE_MARKER
    }
    /// An optimizer whose steps have shrunk this far has converged.
    pub fn converged(&self) -> bool {
        self.optimizing() && self.sigma < 0.05
    }
    pub fn sample(&self, rng: &mut Rng) -> Creature {
        self.sample_scaled(rng, 1.0)
    }
    pub fn sample_scaled(&self, rng: &mut Rng, strength: f32) -> Creature {
        let mut creature = Creature::default();
        self.sample_into(rng, strength, &mut creature);
        creature
    }
    /// `sample_scaled` into `creature`, which it overwrites.
    pub fn sample_into(&self, rng: &mut Rng, strength: f32, creature: &mut Creature) {
        if self.optimizing() {
            self.sample_optimizing(rng, strength, creature)
        } else {
            self.sample_exploring(rng, strength, creature)
        }
    }
    /// Updates the search distribution from scored samples. CMA-ME emitters
    /// get improvement keys; optimizers get fitness.
    pub fn tell(&mut self, population: &Population, samples: &mut Vec<(usize, f32)>) {
        if samples.len() < 2 {
            return;
        }
        samples.retain(|(index, score)| {
            if !score.is_finite() || *score <= crate::evolution::FAILED {
                return false;
            }
            let Some(genome) = population.genomes.get(*index) else {
                return false;
            };
            genome.node_count == self.template.nodes.len()
                && genome.bone_count == self.template.bones.len()
                && genome.muscle_count == self.template.muscles.len()
                && topology_of_population(population, *index) == self.topology
        });
        if samples.len() < 2 {
            return;
        }
        const MAX_UPDATE_SAMPLES: usize = 1024;
        if samples.len() > MAX_UPDATE_SAMPLES {
            samples.select_nth_unstable_by(MAX_UPDATE_SAMPLES, |a, b| b.1.total_cmp(&a.1));
            samples.truncate(MAX_UPDATE_SAMPLES);
        }
        samples.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        if self.optimizing() {
            self.tell_optimizing(population, samples);
        } else {
            self.tell_exploring(population, samples);
        }
    }
    fn sample_exploring(&self, rng: &mut Rng, strength: f32, creature: &mut Creature) {
        let phase_start = self.template.nodes.len() * 4 + self.template.bones.len();
        // Diagonal covariance plus a rank-one term along the evolution path, so
        // parameter changes that keep paying off move together.
        let path_norm = self.path_c.iter().map(|p| p * p).sum::<f32>().sqrt();
        let path_scale = if path_norm > 1e-6 {
            PATH_WEIGHT.sqrt() * gaussian(rng) / path_norm * (self.mean.len() as f32).sqrt()
        } else {
            0.0
        };
        let node_end = self.template.nodes.len() * 4;
        let genes = rng.genes();
        let mut buffer = [0.0f32; MAX_PARAMETERS];
        let values = &mut buffer[..self.mean.len()];
        let noise = self
            .mean
            .iter()
            .zip(&self.covariance)
            .zip(&self.path_c)
            .enumerate()
            .map(|(d, ((&mean, &variance), &path))| {
                let step = variance.sqrt() * genes.gaussian(d as u32) + path * path_scale;
                let value = mean + self.sigma * step * strength;
                // Positions and muscle lengths keep the original 4 m and 1 m
                // scales but are open-ended, so large bodies keep their shape;
                // repair enforces the body limits.
                let muscle_field = d.checked_sub(phase_start).map(|m| m % 8);
                if is_phase_dimension(d, phase_start) {
                    value.rem_euclid(1.0)
                } else if d < node_end && d % 4 == 0 {
                    value
                } else if (d < node_end && d % 4 == 1) || matches!(muscle_field, Some(2 | 3)) {
                    value.max(0.0)
                } else {
                    value.clamp(0.0, 1.0)
                }
            });
        for (slot, value) in values.iter_mut().zip(noise) {
            *slot = value;
        }
        creature.clone_from(&self.template);
        apply_exploring_parameters(creature, values);
    }
    fn tell_exploring(&mut self, population: &Population, samples: &[(usize, f32)]) {
        let dimensions = self.mean.len();
        let mu = samples.len().div_ceil(2).max(1);
        let mut weights: Vec<f32> = (0..mu)
            .map(|i| (mu as f32 + 0.5).ln() - ((i + 1) as f32).ln())
            .collect();
        let weight_sum = weights.iter().sum::<f32>().max(f32::MIN_POSITIVE);
        for weight in &mut weights {
            *weight /= weight_sum;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f32>().max(1e-6);
        let old_mean = self.mean.clone();
        let mut new_mean = vec![0.0; dimensions];
        let mut vector = vec![0.0; dimensions];
        let phase_start = self.template.nodes.len() * 4 + self.template.bones.len();
        for (rank, &(index, _)) in samples.iter().take(mu).enumerate() {
            exploring_parameters_into(population, index, &mut vector);
            for d in 0..dimensions {
                if is_phase_dimension(d, phase_start) {
                    new_mean[d] += weights[rank] * wrap_phase(vector[d] - old_mean[d]);
                } else {
                    new_mean[d] += weights[rank] * vector[d];
                }
            }
        }
        for d in 0..dimensions {
            if is_phase_dimension(d, phase_start) {
                new_mean[d] = (old_mean[d] + new_mean[d]).rem_euclid(1.0);
            }
        }
        let n = dimensions as f32;
        let c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (((mu_eff - 1.0).max(0.0) / (n + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);
        let c1 = 2.0 / ((n + 1.3).powi(2) + mu_eff);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff).max(0.0) / ((n + 2.0).powi(2) + mu_eff))
            .min(1.0 - c1);
        let mut norm_sigma = 0.0;
        for d in 0..dimensions {
            let mean_delta = if is_phase_dimension(d, phase_start) {
                wrap_phase(new_mean[d] - old_mean[d])
            } else {
                new_mean[d] - old_mean[d]
            };
            let y = mean_delta / self.sigma.max(1e-6);
            let normalized = y / self.covariance[d].sqrt().max(1e-6);
            self.path_sigma[d] = (1.0 - c_sigma) * self.path_sigma[d]
                + (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt() * normalized;
            norm_sigma += self.path_sigma[d] * self.path_sigma[d];
        }
        norm_sigma = norm_sigma.sqrt();
        let chi = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
        let hsig = norm_sigma / chi.max(1e-6) < 1.4 + 2.0 / (n + 1.0);
        let correction = if hsig { 0.0 } else { c_c * (2.0 - c_c) };
        let mut rank_mu = vec![0.0; dimensions];
        for (rank, &(index, _)) in samples.iter().take(mu).enumerate() {
            exploring_parameters_into(population, index, &mut vector);
            for d in 0..dimensions {
                let coordinate_delta = if is_phase_dimension(d, phase_start) {
                    wrap_phase(vector[d] - old_mean[d])
                } else {
                    vector[d] - old_mean[d]
                };
                let delta = coordinate_delta / self.sigma.max(1e-6);
                rank_mu[d] += weights[rank] * delta * delta;
            }
        }
        for d in 0..dimensions {
            let mean_delta = if is_phase_dimension(d, phase_start) {
                wrap_phase(new_mean[d] - old_mean[d])
            } else {
                new_mean[d] - old_mean[d]
            };
            let y = mean_delta / self.sigma.max(1e-6);
            self.path_c[d] = (1.0 - c_c) * self.path_c[d]
                + if hsig {
                    (c_c * (2.0 - c_c) * mu_eff).sqrt() * y
                } else {
                    0.0
                };
            self.covariance[d] = ((1.0 - c1 - c_mu + c1 * correction) * self.covariance[d]
                + c1 * self.path_c[d] * self.path_c[d]
                + c_mu * rank_mu[d])
                .clamp(0.0025, 4.0);
        }
        self.sigma = (self.sigma
            * ((c_sigma / d_sigma) * (norm_sigma / chi.max(1e-6) - 1.0)).exp())
        .clamp(0.005, 0.35);
        self.mean = new_mean;
    }
    fn sample_optimizing(&self, rng: &mut Rng, strength: f32, creature: &mut Creature) {
        let layout = Layout::of(&self.template);
        let genes = rng.genes();
        let mut buffer = [0.0f32; MAX_PARAMETERS];
        let values = &mut buffer[..self.mean.len()];
        for (d, ((slot, &mean), &variance)) in values
            .iter_mut()
            .zip(&self.mean)
            .zip(&self.covariance)
            .enumerate()
        {
            *slot = mean
                + self.sigma
                    * strength
                    * layout.scale(d)
                    * variance.sqrt()
                    * genes.gaussian(d as u32);
        }
        creature.clone_from(&self.template);
        apply_parameters(creature, values);
    }
    /// Separable CMA-ES update (Ros & Hansen 2008) from scored samples, fastest
    /// first. Steps are measured in each coordinate's physical scale.
    fn tell_optimizing(&mut self, population: &Population, samples: &[(usize, f32)]) {
        let dimensions = self.mean.len();
        let mu = samples.len().div_ceil(2).max(1);
        let mut weights: Vec<f32> = (0..mu)
            .map(|i| (mu as f32 + 0.5).ln() - ((i + 1) as f32).ln())
            .collect();
        let weight_sum = weights.iter().sum::<f32>().max(f32::MIN_POSITIVE);
        for weight in &mut weights {
            *weight /= weight_sum;
        }
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f32>().max(1e-6);
        let layout = Layout::of(&self.template);
        let sigma = self.sigma.max(1e-6);
        // Weighted mean step and weighted squared steps, in units of sigma.
        let mut step = vec![0.0; dimensions];
        let mut rank_mu = vec![0.0; dimensions];
        let mut vector = vec![0.0; dimensions];
        for (rank, &(index, _)) in samples.iter().take(mu).enumerate() {
            parameters_into(population, index, &mut vector);
            for d in 0..dimensions {
                let y = layout.delta(d, vector[d], self.mean[d]) / (sigma * layout.scale(d));
                step[d] += weights[rank] * y;
                rank_mu[d] += weights[rank] * y * y;
            }
        }
        let n = dimensions as f32;
        let c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (((mu_eff - 1.0).max(0.0) / (n + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
        let c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);
        // A diagonal covariance learns (n + 2) / 3 times faster than a full one.
        let separable = (n + 2.0) / 3.0;
        let c1 = (2.0 / ((n + 1.3).powi(2) + mu_eff) * separable).min(0.5);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff).max(0.0) / ((n + 2.0).powi(2) + mu_eff)
            * separable)
            .min(1.0 - c1);
        let mut norm_sigma = 0.0;
        for ((path, &step), &variance) in
            self.path_sigma.iter_mut().zip(&step).zip(&self.covariance)
        {
            *path = (1.0 - c_sigma) * *path
                + (c_sigma * (2.0 - c_sigma) * mu_eff).sqrt() * step / variance.sqrt().max(1e-6);
            norm_sigma += *path * *path;
        }
        norm_sigma = norm_sigma.sqrt();
        let chi = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
        let hsig = norm_sigma / chi.max(1e-6) < 1.4 + 2.0 / (n + 1.0);
        let correction = if hsig { 0.0 } else { c_c * (2.0 - c_c) };
        for d in 0..dimensions {
            self.path_c[d] = (1.0 - c_c) * self.path_c[d]
                + if hsig {
                    (c_c * (2.0 - c_c) * mu_eff).sqrt() * step[d]
                } else {
                    0.0
                };
            self.covariance[d] = ((1.0 - c1 - c_mu + c1 * correction) * self.covariance[d]
                + c1 * self.path_c[d] * self.path_c[d]
                + c_mu * rank_mu[d])
                .clamp(1e-4, 1e4);
            self.mean[d] = layout.moved(d, self.mean[d], sigma * layout.scale(d) * step[d]);
        }
        self.sigma = (self.sigma
            * ((c_sigma / d_sigma) * (norm_sigma / chi.max(1e-6) - 1.0)).exp())
        .clamp(0.01, 30.0);
        // Fast gaits are fragile: when most samples fail outright, step-length
        // control alone keeps growing the steps, so shrink them instead.
        if samples[samples.len() / 2].1 < 0.25 * samples[0].1 {
            self.sigma = (self.sigma * 0.6).max(0.01);
        }
    }
}

/// Whether coordinate `dimension` is a phase field that wraps modulo 1.
fn is_phase_dimension(dimension: usize, phase_start: usize) -> bool {
    dimension >= phase_start && (dimension - phase_start) % 8 == 5
}
/// Phase difference wrapped to [-0.5, 0.5).
fn wrap_phase(delta: f32) -> f32 {
    (delta + 0.5).rem_euclid(1.0) - 0.5
}

/// Most CMA coordinates of a body at the caps: 4 per node, 5 per bone, the
/// shared period and 8 per muscle.
const MAX_PARAMETERS: usize =
    crate::evolution::MAX_NODES * (4 + BONE_FIELDS) + 1 + crate::evolution::MAX_MUSCLES * 8;
/// Share of each CMA step taken along the normalized evolution path.
const PATH_WEIGHT: f32 = 0.3;

/// A standard gaussian from the stream's cursor (`Rng::gaussian`).
pub(crate) fn gaussian(rng: &mut Rng) -> f32 {
    rng.gaussian()
}
fn exploring_parameters(creature: &Creature) -> Vec<f32> {
    let mut output = Vec::with_capacity(
        creature.nodes.len() * 4 + creature.bones.len() + creature.muscles.len() * 8,
    );
    for n in &creature.nodes {
        output.extend([
            (n.x + 4.0) / 8.0,
            (n.y / 4.0).max(0.0),
            ((n.diameter - 0.01) / 0.99).clamp(0.0, 1.0),
            n.friction.clamp(0.0, 1.0),
        ]);
    }
    for bone in &creature.bones {
        output.push(
            ((bone.rest_length - 0.03) / (crate::evolution::max_bone_length() - 0.03))
                .clamp(0.0, 1.0),
        );
    }
    for m in &creature.muscles {
        output.extend([
            m.anchor_a.clamp(0.0, 1.0),
            m.anchor_b.clamp(0.0, 1.0),
            ((m.short - 0.01) / 0.79).max(0.0),
            ((m.long - 0.01) / 0.99).max(0.0),
            ((m.period - crate::evolution::min_muscle_period())
                / (10.0 - crate::evolution::min_muscle_period()))
            .clamp(0.0, 1.0),
            m.phase.clamp(0.0, 1.0),
            ((m.duty - 0.05) / 0.90).clamp(0.0, 1.0),
            ((m.stiffness - 1.0) / 119.0).clamp(0.0, 1.0),
        ]);
    }
    output
}
fn exploring_parameters_into(population: &Population, index: usize, output: &mut [f32]) {
    let genome = &population.genomes[index];
    let nodes = &population.nodes[genome.node_start..genome.node_start + genome.node_count];
    let muscles =
        &population.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count];
    let mut i = 0;
    for n in nodes {
        output[i..i + 4].copy_from_slice(&[
            (n.x + 4.0) / 8.0,
            (n.y / 4.0).max(0.0),
            ((n.diameter - 0.01) / 0.99).clamp(0.0, 1.0),
            n.friction.clamp(0.0, 1.0),
        ]);
        i += 4;
    }
    let bones = &population.bones[genome.bone_start..genome.bone_start + genome.bone_count];
    for bone in bones {
        output[i] = ((bone.rest_length - 0.03) / (crate::evolution::max_bone_length() - 0.03))
            .clamp(0.0, 1.0);
        i += 1;
    }
    for m in muscles {
        output[i..i + 8].copy_from_slice(&[
            m.anchor_a.clamp(0.0, 1.0),
            m.anchor_b.clamp(0.0, 1.0),
            ((m.short - 0.01) / 0.79).max(0.0),
            ((m.long - 0.01) / 0.99).max(0.0),
            ((m.period - crate::evolution::min_muscle_period())
                / (10.0 - crate::evolution::min_muscle_period()))
            .clamp(0.0, 1.0),
            m.phase.clamp(0.0, 1.0),
            ((m.duty - 0.05) / 0.90).clamp(0.0, 1.0),
            ((m.stiffness - 1.0) / 119.0).clamp(0.0, 1.0),
        ]);
        i += 8;
    }
}
fn apply_exploring_parameters(creature: &mut Creature, values: &[f32]) {
    let mut i = 0;
    for n in &mut creature.nodes {
        n.x = values[i] * 8.0 - 4.0;
        n.y = values[i + 1] * 4.0;
        n.diameter = 0.01 + values[i + 2] * 0.99;
        n.friction = values[i + 3];
        i += 4;
    }
    for bone in &mut creature.bones {
        bone.rest_length = 0.03 + values[i] * (crate::evolution::max_bone_length() - 0.03);
        i += 1;
    }
    for m in &mut creature.muscles {
        m.anchor_a = values[i].clamp(0.0, 1.0);
        m.anchor_b = values[i + 1].clamp(0.0, 1.0);
        m.short = 0.01 + values[i + 2] * 0.79;
        m.long = (0.01 + values[i + 3] * 0.99).max(m.short);
        m.period = crate::evolution::min_muscle_period()
            + values[i + 4] * (10.0 - crate::evolution::min_muscle_period());
        m.phase = values[i + 5].fract();
        m.duty = 0.05 + values[i + 6] * 0.90;
        m.stiffness = 1.0 + values[i + 7] * 119.0;
        i += 8;
    }
}

/// Where each CMA coordinate lives in a body plan and how far one unit step
/// moves it: per node x, y, diameter, friction; per bone rest length, joint
/// range, and organ mass and position; the shared log period; per muscle
/// anchors, lengths, phase, duty, log stiffness, and touchdown reset phase.
/// An organ mass at or below zero means no organ, so organs can grow and
/// vanish smoothly.
struct Layout {
    nodes: usize,
    bones: usize,
    /// Typical bone length (m), so positions and lengths search relative to
    /// the body's size.
    size: f32,
}
const NODE_SCALES: [f32; 4] = [0.02, 0.02, 0.005, 0.03];
const BONE_SCALES: [f32; 5] = [0.02, 0.1, 0.1, 0.02, 0.05];
const BONE_FIELDS: usize = BONE_SCALES.len();
const PERIOD_SCALE: f32 = 0.05;
const MUSCLE_SCALES: [f32; 8] = [0.05, 0.05, 0.02, 0.02, 0.05, 0.05, 0.1, 0.05];
/// Which scales above are lengths, multiplied by the body's size.
const MUSCLE_LENGTHS: [bool; 8] = [false, false, true, true, false, false, false, false];
impl Layout {
    fn of(template: &Creature) -> Self {
        let size = if template.bones.is_empty() {
            1.0
        } else {
            template.bones.iter().map(|b| b.rest_length).sum::<f32>() / template.bones.len() as f32
        };
        Self {
            nodes: template.nodes.len(),
            bones: template.bones.len(),
            size: size.clamp(0.05, 10.0),
        }
    }
    /// The muscle field of coordinate `d`, if it is one.
    fn muscle_field(&self, d: usize) -> Option<usize> {
        let start = self.nodes * 4 + self.bones * BONE_FIELDS + 1;
        (d >= start).then(|| (d - start) % 8)
    }
    fn scale(&self, d: usize) -> f32 {
        let node_end = self.nodes * 4;
        let bone_end = node_end + self.bones * BONE_FIELDS;
        if d < node_end {
            NODE_SCALES[d % 4] * if d % 4 < 2 { self.size } else { 1.0 }
        } else if d < bone_end {
            let field = (d - node_end) % BONE_FIELDS;
            BONE_SCALES[field] * if field == 0 { self.size } else { 1.0 }
        } else if let Some(field) = self.muscle_field(d) {
            MUSCLE_SCALES[field]
                * if MUSCLE_LENGTHS[field] {
                    self.size
                } else {
                    1.0
                }
        } else {
            PERIOD_SCALE
        }
    }
    /// Phases wrap around the cycle.
    fn wraps(&self, d: usize) -> bool {
        matches!(self.muscle_field(d), Some(4 | 7))
    }
    fn delta(&self, d: usize, value: f32, mean: f32) -> f32 {
        if self.wraps(d) {
            (value - mean + 0.5).rem_euclid(1.0) - 0.5
        } else {
            value - mean
        }
    }
    fn moved(&self, d: usize, mean: f32, step: f32) -> f32 {
        if self.wraps(d) {
            (mean + step).rem_euclid(1.0)
        } else {
            mean + step
        }
    }
}

/// Physical CMA parameters in absolute units for optimizer search.
fn parameters(creature: &Creature) -> Vec<f32> {
    let mut output = vec![
        0.0;
        parameter_count(
            creature.nodes.len(),
            creature.bones.len(),
            creature.muscles.len()
        )
    ];
    write_parameters(
        &creature.nodes,
        &creature.bones,
        &creature.muscles,
        &mut output,
    );
    output
}
fn parameter_count(nodes: usize, bones: usize, muscles: usize) -> usize {
    nodes * 4 + bones * BONE_FIELDS + 1 + muscles * 8
}
fn parameters_into(population: &Population, index: usize, output: &mut [f32]) {
    let genome = &population.genomes[index];
    write_parameters(
        &population.nodes[genome.node_start..genome.node_start + genome.node_count],
        &population.bones[genome.bone_start..genome.bone_start + genome.bone_count],
        &population.muscles[genome.muscle_start..genome.muscle_start + genome.muscle_count],
        output,
    );
}
fn write_parameters(
    nodes: &[crate::evolution::NodeGene],
    bones: &[crate::evolution::Bone],
    muscles: &[crate::evolution::Muscle],
    output: &mut [f32],
) {
    let mut i = 0;
    for n in nodes {
        output[i..i + 4].copy_from_slice(&[n.x, n.y, n.diameter, n.friction]);
        i += 4;
    }
    for b in bones {
        output[i..i + BONE_FIELDS].copy_from_slice(&[
            b.rest_length,
            b.min_angle,
            b.max_angle,
            b.organ_mass,
            b.organ_at,
        ]);
        i += BONE_FIELDS;
    }
    output[i] = muscles.first().map_or(1.0, |m| m.period).max(1e-3).ln();
    i += 1;
    for m in muscles {
        output[i..i + 8].copy_from_slice(&[
            m.anchor_a,
            m.anchor_b,
            m.short,
            m.long,
            m.phase,
            m.duty,
            m.stiffness.max(1e-3).ln(),
            m.reset,
        ]);
        i += 8;
    }
}
fn apply_parameters(creature: &mut Creature, values: &[f32]) {
    let mut i = 0;
    for n in &mut creature.nodes {
        n.x = values[i];
        n.y = values[i + 1].max(0.0);
        n.diameter = values[i + 2];
        n.friction = values[i + 3];
        i += 4;
    }
    for bone in &mut creature.bones {
        bone.rest_length = values[i].clamp(0.03, crate::evolution::max_bone_length());
        bone.min_angle = values[i + 1];
        bone.max_angle = values[i + 2];
        bone.clamp_range();
        // Repair keeps organs within their mass range and near the center.
        bone.organ_mass = values[i + 3].max(0.0);
        bone.organ_at = values[i + 4].clamp(0.0, 1.0);
        i += BONE_FIELDS;
    }
    // One log period scales the whole body clock; limbs keep their ratios.
    let base = creature.muscles.first().map_or(1.0, |m| m.period).max(1e-3);
    let scale = values[i]
        .exp()
        .clamp(crate::evolution::min_muscle_period(), 10.0)
        / base;
    i += 1;
    let stroke = crate::evolution::max_stroke();
    for m in &mut creature.muscles {
        m.anchor_a = values[i].clamp(0.0, 1.0);
        m.anchor_b = values[i + 1].clamp(0.0, 1.0);
        m.short = values[i + 2].clamp(0.01, 0.8 * stroke);
        m.long = values[i + 3].clamp(m.short, stroke);
        m.period = (m.period * scale).clamp(crate::evolution::min_muscle_period(), 10.0);
        m.phase = values[i + 4].rem_euclid(1.0);
        m.duty = values[i + 5].clamp(0.05, 0.95);
        m.stiffness = values[i + 6].exp().clamp(1.0, 120.0);
        m.reset = values[i + 7].rem_euclid(1.0);
        i += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::{CmaEmitter, Layout, Niche};
    use crate::{
        config::Config,
        evolution::{self, Population},
    };

    #[test]
    fn row_by_row_neighbor_search_visits_the_cells_one_by_one_does() {
        use super::{Descriptor, Emitter, QdArchive};
        let config = Config {
            population: 300,
            random_seed: false,
            seed: 5,
            ..Config::default()
        };
        let population = evolution::create(&config).unwrap();
        // Refined, so the elites fill cells of every body class.
        let mut archive = QdArchive::default();
        archive.set_refined(true);
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 32) as u32
        };
        for _ in 0..600 {
            let descriptor = Descriptor {
                nodes: (2 + next() % 14) as u16,
                aspect_ratio: (next() % 40) as f32 / 10.0,
                ground_contact: (next() % 100) as f32 / 100.0,
                gait_frequency: (next() % 60) as f32 / 10.0,
                vertical_oscillation: 0.1,
                mean_height: (next() % 300) as f32 / 100.0 + 0.1,
                feet: (next() % 5) as f32 + 1.0,
                ..Default::default()
            };
            let fitness = (next() % 1000) as f32 / 10.0;
            let index = next() as usize % 300;
            archive.offer(
                &population,
                index,
                descriptor,
                fitness,
                false,
                Emitter::Cma,
                0,
                0,
            );
        }
        assert!(archive.behavior_count() > 100);
        let classes = |axis: usize| {
            let values: std::collections::BTreeSet<u8> = archive
                .behavior_indices
                .iter()
                .map(|&i| archive.entries[i].niche.0[axis])
                .collect();
            values.len()
        };
        assert!(classes(2) > 1 && classes(5) > 1);
        let filled = archive.filled_rows();
        for &i in &archive.behavior_indices {
            let niche = &archive.entries[i].niche;
            for radius in 1..=2 {
                let (mut cells, mut rows) = (Vec::new(), Vec::new());
                archive.for_each_neighbor(niche, radius, |slot| cells.push(slot));
                archive.for_each_filled_neighbor(&filled, niche, radius, |slot| rows.push(slot));
                assert_eq!(cells, rows);
            }
        }
    }

    #[test]
    fn partial_score_refresh_matches_a_full_refresh() {
        use super::{Descriptor, Emitter, QdArchive};
        let config = Config {
            population: 400,
            random_seed: false,
            seed: 3,
            ..Config::default()
        };
        let population = evolution::create(&config).unwrap();
        let mut archive = QdArchive::default();
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u32
        };
        for round in 0..40u32 {
            for _ in 0..(1 + round % 5) {
                let index = next() as usize % 400;
                let descriptor = Descriptor {
                    ground_contact: (next() % 100) as f32 / 100.0,
                    gait_frequency: (next() % 60) as f32 / 10.0,
                    vertical_oscillation: 0.1,
                    mean_height: (next() % 300) as f32 / 100.0 + 0.1,
                    feet: (next() % 5) as f32 + 1.0,
                    ..Default::default()
                };
                let fitness = (next() % 1000) as f32 / 10.0;
                archive.offer(
                    &population,
                    index,
                    descriptor,
                    fitness,
                    false,
                    Emitter::Cma,
                    round,
                    0,
                );
            }
            archive.refresh_behavior_scores();
            let mut full = archive.clone();
            full.behavior_scores = Default::default();
            full.changed_cells.clear();
            full.refresh_behavior_scores();
            assert_eq!(
                archive.behavior_scores.novelty,
                full.behavior_scores.novelty
            );
            assert_eq!(
                archive.behavior_scores.local_competition,
                full.behavior_scores.local_competition
            );
        }
        assert!(archive.behavior_count() > 20);
    }

    #[test]
    fn rebinning_moves_each_elite_to_its_cell_and_keeps_the_faster_of_a_shared_one() {
        use super::{Descriptor, Elite, Emitter, QdArchive, Topology};
        let elite = |fitness: f32, nodes: u16, cadence: f32, niche: [u8; 6]| Elite {
            niche: Niche(niche),
            descriptor: Descriptor {
                ground_contact: 0.5,
                gait_frequency: cadence,
                mean_height: 0.5,
                feet: 2.0,
                aspect_ratio: 1.5,
                nodes,
                ..Descriptor::default()
            },
            creature: Default::default(),
            fitness,
            emitter: Emitter::Cma,
            improved_generation: 0,
            protected_until: 0,
            visits: 3,
            topology: Topology::default(),
            graduate: false,
            fine: false,
        };
        let mut archive = QdArchive::default();
        archive.set_refined(true);
        // As an older layout kept them: no shape or size class in the niche.
        let old = [3, 1, 0, 3, 1, 0];
        archive.entries = vec![
            elite(5.0, 6, 1.0, old),
            // Another way of moving.
            elite(7.0, 6, 2.0, [3, 2, 0, 3, 1, 0]),
        ];
        archive.rebin();
        assert_eq!(archive.behavior_count(), 2);
        for e in &archive.entries {
            assert_eq!(e.niche, e.descriptor.niche_in(archive.classes()));
            assert_eq!(
                archive
                    .slot_for(&e.niche)
                    .map(|s| archive.entries[s].fitness),
                Some(e.fitness)
            );
        }
        assert_eq!(archive.qd_score, 12.0);
        // Two elites that land in one cell under the new layout: the faster stays.
        let mut archive = QdArchive::default();
        archive.set_refined(true);
        archive.entries = vec![
            elite(4.0, 6, 1.0, old),
            elite(9.0, 7, 1.0, [3, 1, 1, 3, 1, 0]),
        ];
        archive.rebin();
        assert_eq!(archive.behavior_count(), 1);
        assert_eq!(archive.entries[0].fitness, 9.0);
        assert_eq!(archive.entries[0].visits, 3);
        assert_eq!(archive.movement_count(), 1);
    }

    #[test]
    fn a_plan_key_is_the_same_from_a_topology_and_from_the_genes() {
        let config = Config {
            population: 64,
            random_seed: false,
            seed: 11,
            ..Config::default()
        };
        let population = evolution::create(&config).unwrap();
        let topologies: Vec<_> = (0..population.genomes.len())
            .map(|i| super::topology_of_population(&population, i))
            .collect();
        for (i, topology) in topologies.iter().enumerate() {
            assert_eq!(
                topology.plan_key(),
                super::plan_key_of_population(&population, i)
            );
        }
        // Two bodies share a key exactly when they share a plan.
        for (i, a) in topologies.iter().enumerate() {
            for b in &topologies[i..] {
                assert_eq!(a == b, a.plan_key() == b.plan_key());
            }
        }
        assert!(topologies.iter().any(|t| *t != topologies[0]));
    }

    #[test]
    fn cma_feedback_ignores_candidates_with_changed_topology() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let template = evolution::create(&config).unwrap().creature(0);
        let mut cma = CmaEmitter::new(template.clone(), Niche([0; 6]), 0);
        let original_mean = cma.mean.clone();
        let mut population = Population::default();
        for _ in 0..2 {
            let mut changed = template.clone();
            changed.muscles.push(changed.muscles[0]);
            population.push(changed);
        }
        cma.tell(&population, &mut vec![(0, 2.0), (1, 1.0)]);
        assert_eq!(cma.mean, original_mean);
    }

    #[test]
    fn cma_phase_dimensions_follow_bones_and_eight_value_muscles() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let template = evolution::create(&config).unwrap().creature(0);
        let layout = Layout::of(&template);
        let start = template.nodes.len() * 4 + template.bones.len() * super::BONE_FIELDS + 1;
        assert!(!layout.wraps(start - 1));
        assert!(!layout.wraps(start));
        assert!(layout.wraps(start + 4));
        assert!(layout.wraps(start + 7));
        assert!(layout.wraps(start + 12));
        assert!(!layout.wraps(start + 13));
    }

    #[test]
    fn optimizer_moves_toward_its_faster_samples() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let template = evolution::create(&config).unwrap().creature(0);
        let mut cma = CmaEmitter::optimizer(template.clone(), super::optimizer_niche(0, 0), 0);
        assert!(cma.optimizing() && !cma.converged());
        let mut population = Population::default();
        for shift in [0.05, -0.05, 0.04, -0.04] {
            let mut sample = template.clone();
            for node in &mut sample.nodes {
                node.x += shift;
            }
            population.push(sample);
        }
        let before = cma.mean[0];
        cma.tell(
            &population,
            &mut vec![(0, 4.0), (2, 3.0), (3, 2.0), (1, 1.0)],
        );
        assert!(cma.mean[0] > before);
    }

    #[test]
    fn cma_parameters_round_trip_through_a_creature() {
        let config = Config {
            population: 2,
            random_seed: false,
            ..Config::default()
        };
        let template = evolution::create(&config).unwrap().creature(0);
        let cma = CmaEmitter::optimizer(template.clone(), super::optimizer_niche(0, 0), 0);
        let mut copy = template.clone();
        super::apply_parameters(&mut copy, &cma.mean);
        assert_eq!(super::parameters(&copy), cma.mean);
    }
}
