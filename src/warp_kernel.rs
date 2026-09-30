//! Packing and kernel source of the lane-group CUDA kernel
//! (`shaders/warp_creature.cu`).
//!
//! A creature runs on a group of 8, 16 or 32 lanes of one warp. Lane `i` owns
//! node `i` and the bone that ends there: lane 0 is the head, lane 1 the neck.
//! Lanes follow the bone tree breadth first, so each tree level is a run of
//! neighbouring lanes and the children of one bone are consecutive. Muscles run
//! one per lane, `W` at a time, in at most `ROUNDS` rounds.
//!
//! Every derived constant (masses with the bones, organs and muscles, slack
//! lengths, muscle strengths, joint ranges, the start pose) comes from
//! `physics2::Model`.
use crate::{
    config::Config,
    creature_kernel::LaneBatch,
    evolution::Population,
    physics::{self, Fidelity},
    physics2::{self, Model},
};
use anyhow::{Result, bail};
use rayon::prelude::*;

/// Words per lane of a creature's lane record.
pub const LANE_FIELDS: usize = 12;
/// Words per muscle, read as four 16-byte loads.
pub const MUSCLE_FIELDS: usize = 16;
/// Muscle rounds a group runs: a creature on `W` lanes has at most
/// `ROUNDS * W` muscles.
pub const ROUNDS: usize = 4;
/// Lanes per creature.
pub const CLASSES: [usize; 3] = [8, 16, 32];
/// Largest body the kernel runs.
pub const MAX_NODES: usize = 32;
/// Contacts per substep.
pub const MAX_CONTACTS: usize = 4;
/// Substeps per step, and Gauss-Seidel sweeps per substep (then sweeps that
/// only take back friction that would do positive work).
pub const SUBSTEPS: u32 = 2;
pub const PGS_SWEEPS: u32 = 4;
pub const CLEAN_SWEEPS: u32 = 1;
/// Threads per block, and blocks per multiprocessor the register budget is
/// set for: 4 blocks of 128 threads fit 128 registers per thread.
pub const BLOCK: u32 = 128;
pub const MIN_BLOCKS: u32 = 4;
/// Creatures per kernel launch (a wave).
pub const WAVE: usize = 262_144;

/// The creature data of one lane class, ready for upload.
#[derive(Default)]
pub struct WavePack {
    /// `[creature][field][lane]` lane records.
    pub lanes: Vec<u32>,
    /// `[round][lane][field]` muscle records, per creature at its `heads`
    /// offset.
    pub muscles: Vec<f32>,
    /// `[round][word][lane]` lists of the muscle ends each bone carries: four
    /// byte slots per word (muscle lane times two plus the end), 255 for none.
    pub ends: Vec<u32>,
    /// Two words of four per creature (see `shaders/warp_creature.cu`).
    pub heads: Vec<[u32; 4]>,
}

/// The lane class of a body: the fewest lanes that hold its nodes and its
/// muscles in `ROUNDS` rounds.
pub fn class_of(nodes: usize, muscles: usize) -> Option<usize> {
    CLASSES
        .iter()
        .copied()
        .find(|&w| nodes <= w && muscles <= ROUNDS * w)
}

/// World switches: an effect that is off leaves no code in the kernel.
pub fn world_flags(cfg: &Config) -> u32 {
    let ground = cfg.ground;
    let mut flags = 0u32;
    let mut set = |bit: u32, on: bool| {
        if on {
            flags |= 1 << bit;
        }
    };
    set(0, ground);
    set(
        1,
        ground && (physics::terrain_amplitude(cfg.terrain) != 0.0 || cfg.quake > 0.0),
    );
    set(2, ground && cfg.slope != 0.0);
    set(3, ground && cfg.gaps > 0.0);
    set(4, ground && cfg.hurdles > 0.0);
    set(5, ground && cfg.quake > 0.0);
    set(6, ground && cfg.mud > 0.0);
    set(7, cfg.water > 0.0);
    set(8, ground && cfg.patches > 0.0);
    set(9, cfg.wind != 0.0);
    set(10, cfg.fidelity().air_per_step(cfg.air_retention) != 1.0);
    flags
}

const FLAG_NAMES: [&str; 11] = [
    "GROUND", "TERRAIN", "SLOPE", "GAPS", "HURDLES", "QUAKE", "MUD", "WATER", "ICE", "WIND", "AIR",
];

/// Kernel parameters; the layout of `Params` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub count: u32,
    pub base: u32,
    pub steps: u32,
    pub screen_step: u32,
    pub stride: u32,
    pub gravity: f32,
    pub air: f32,
    pub friction: f32,
    pub terrain: f32,
    pub muscle_energy: f32,
    pub muscle_recovery: f32,
    pub slope: f32,
    pub wind: f32,
    pub mud: f32,
    pub gaps: f32,
    pub hurdles: f32,
    pub quake: f32,
    pub screen_bar: f32,
    pub water: f32,
    pub patches: f32,
    pub air_sub: f32,
    pub inv_muscle_energy: f32,
    pub spare: [f32; 2],
}

/// Parameters of a wave of `count` creatures from `base` of a batch.
pub fn params(cfg: &Config, base: usize, count: usize, stride: usize) -> Params {
    let fidelity = cfg.fidelity();
    let air = fidelity.air_per_step(cfg.air_retention);
    let ground = cfg.ground;
    Params {
        count: count as u32,
        base: base as u32,
        steps: cfg.steps(),
        screen_step: cfg
            .screen
            .map_or(u32::MAX, |screen| screen.tick(fidelity) - fidelity.settle()),
        stride: stride as u32,
        gravity: cfg.gravity,
        air,
        friction: cfg.ground_friction,
        terrain: physics::terrain_amplitude(cfg.terrain),
        muscle_energy: cfg.muscle_energy,
        muscle_recovery: cfg.muscle_recovery,
        slope: if ground { cfg.slope } else { 0.0 },
        wind: cfg.wind,
        mud: if ground { cfg.mud } else { 0.0 },
        gaps: if ground { cfg.gaps } else { 0.0 },
        hurdles: if ground { cfg.hurdles } else { 0.0 },
        quake: if ground { cfg.quake } else { 0.0 },
        screen_bar: cfg.screen.map_or(f32::NEG_INFINITY, |screen| screen.bar),
        water: cfg.water,
        patches: if ground { cfg.patches } else { 0.0 },
        air_sub: air.powf(1.0 / solver_setting("SUBSTEPS", SUBSTEPS) as f32),
        inv_muscle_energy: 1.0 / cfg.muscle_energy,
        spare: [0.0; 2],
    }
}

/// A solver setting, or its developer override `EVOLUTION_WARP_<NAME>` for
/// measuring other settings (never needed to play).
pub fn solver_setting(name: &str, default: u32) -> u32 {
    std::env::var(format!("EVOLUTION_WARP_{name}"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The kernel source for `class` lanes, world `flags`, `fidelity` and, with
/// `record`, the frame output.
pub fn cuda_source(class: usize, flags: u32, fidelity: Fidelity, record: bool) -> String {
    let limits = physics::limits();
    let float = |value: f32| format!("{value:?}f");
    let mut defines: Vec<(String, String)> = vec![
        ("W".into(), format!("{class}u")),
        ("BLOCK".into(), format!("{BLOCK}u")),
        ("MIN_BLOCKS".into(), format!("{}", solver_setting("MIN_BLOCKS", MIN_BLOCKS))),
        ("RATE".into(), format!("{:.1}f", fidelity.rate as f32)),
        ("SETTLE".into(), format!("{}u", fidelity.settle())),
        ("SAMPLE".into(), format!("{}u", fidelity.sample_interval())),
        ("SUBSTEPS".into(), format!("{}u", solver_setting("SUBSTEPS", SUBSTEPS))),
        ("PGS_SWEEPS".into(), format!("{}u", solver_setting("PGS_SWEEPS", PGS_SWEEPS))),
        ("CLEAN_SWEEPS".into(), format!("{}u", solver_setting("CLEAN_SWEEPS", CLEAN_SWEEPS))),
        ("RECORD".into(), (if record { "1" } else { "0" }).into()),
        (
            "DEBUG".into(),
            if record {
                std::env::var("EVOLUTION_WARP_DEBUG").unwrap_or_else(|_| "0".into())
            } else {
                "0".into()
            },
        ),
    ];
    for (bit, name) in FLAG_NAMES.iter().enumerate() {
        defines.push((
            (*name).into(),
            (if flags & (1 << bit) != 0 { "1" } else { "0" }).into(),
        ));
    }
    let constants = [
        ("MUSCLE_CAPACITY", float(limits.muscle_energy)),
        ("MUSCLE_RECOVERY", float(limits.muscle_recovery)),
        ("MAX_MUSCLE_FORCE", float(limits.muscle_force)),
        (
            "INV_JOINT_DAMPING",
            float(if physics2::joint_damping() > 0.0 {
                1.0 / physics2::joint_damping()
            } else {
                0.0
            }),
        ),
        ("LIMIT_HARDNESS", float(physics2::LIMIT_HARDNESS)),
        ("JOINT_BREAK", float(physics::JOINT_BREAK)),
        ("MUD_NORMAL", float(physics::MUD_NORMAL)),
        ("MUD_GRIP", float(physics::MUD_GRIP)),
        ("MUD_DRAG", float(physics::MUD_DRAG)),
        ("MUD_FULL_DEPTH", float(physics::MUD_FULL_DEPTH)),
        ("SPIN_CAP", float(physics2::SPIN_CAP)),
        ("INV_SPIN_CAP", float(1.0 / physics2::SPIN_CAP)),
        ("SPIN_HARDNESS", float(physics2::SPIN_HARDNESS)),
        ("PUSH_OUT", float(physics2::PUSH_OUT)),
        ("AIR_DRAG", float(physics2::AIR_DRAG)),
        ("WATER_DRAG", float(physics2::WATER_DRAG)),
        ("WATER_ALONG", float(physics2::WATER_ALONG)),
        ("WATER_BUOYANCY", float(physics2::WATER_BUOYANCY)),
        ("ICE_INV", float(1.0 / physics::ICE_SPACING)),
        ("HEAD_SHAKE_LIMIT", float(physics::HEAD_SHAKE_LIMIT)),
        ("HEAD_SHAKE_WINDOW", float(physics::HEAD_SHAKE_WINDOW)),
        ("CONTACT_SLACK", float(physics2::CONTACT_SLACK)),
        ("LIFT_CLEARANCE", float(physics2::LIFT_CLEARANCE)),
        ("GAP_DEPTH", float(physics::GAP_DEPTH)),
        ("GAP_RUN", float(physics::GAP_RUN)),
        ("HURDLE_SPACING", float(physics::HURDLE_SPACING)),
        ("HURDLE_TOP", float(physics::HURDLE_TOP)),
        ("HURDLE_RUN", float(physics::HURDLE_RUN)),
    ];
    for (name, value) in constants {
        defines.push((name.into(), value));
    }
    let mut source = String::new();
    for (name, value) in defines {
        source.push_str(&format!("#define {name} {value}\n"));
    }
    source.push_str(include_str!("../shaders/warp_creature.cu"));
    source
}

/// One creature's records before they are joined into a class's buffers.
struct Packed {
    lanes: Vec<u32>,
    muscles: Vec<f32>,
    ends: Vec<u32>,
    head: [u32; 8],
    key: (u32, u32, u32),
}

/// The records of one creature on `w` lanes.
fn pack_creature(model: &Model, cfg: &Config, hash: u32, w: usize) -> Packed {
    let start = model.start(cfg);
    let bones = model.pivot.len();
    let nodes = bones + 1;
    // Breadth-first bone order: lane of each bone, bone of each lane.
    let mut children = vec![Vec::new(); bones];
    for j in 1..bones {
        let parent = model.parent[j].unwrap_or(0);
        children[parent].push(j);
    }
    let mut order = Vec::with_capacity(bones);
    order.push(0usize);
    let mut at = 0;
    while at < order.len() {
        let j = order[at];
        order.extend(children[j].iter().copied());
        at += 1;
    }
    let mut lane_of_bone = vec![0usize; bones];
    for (k, &j) in order.iter().enumerate() {
        lane_of_bone[j] = k + 1;
    }
    let lane_of_node = |node: usize| {
        if node == 0 {
            0
        } else {
            lane_of_bone[node - 1]
        }
    };
    let mut level = vec![0u32; bones];
    let mut ancestors = vec![0u32; bones];
    for &j in &order {
        let (up_level, up_mask) = match model.parent[j] {
            Some(p) if j > 0 => (level[p], ancestors[p]),
            _ => (0, 0),
        };
        level[j] = up_level + 1;
        ancestors[j] = up_mask | 1 << lane_of_bone[j];
    }
    let depth = level.iter().copied().max().unwrap_or(1);
    let mut lanes = vec![0u32; LANE_FIELDS * w];
    let mut put = |field: usize, lane: usize, value: u32| lanes[field * w + lane] = value;
    // The head.
    put(0, 0, model.mass[0].to_bits());
    put(1, 0, model.radius[0].to_bits());
    put(2, 0, model.friction[0].to_bits());
    put(6, 0, start.x0[0].to_bits());
    put(7, 0, start.x0[1].to_bits());
    for j in 0..bones {
        let lane = lane_of_bone[j];
        let node = j + 1;
        let pivot = model.pivot[j];
        let parent_lane = match model.parent[j] {
            Some(p) if j > 0 => lane_of_bone[p],
            _ => 1,
        };
        let first_child = children[j].first().map_or(0, |&c| lane_of_bone[c]);
        let topo = lane_of_node(pivot) as u32
            | (parent_lane as u32) << 5
            | level[j] << 10
            | (first_child as u32) << 15
            | (children[j].len() as u32) << 20;
        put(0, lane, model.mass[node].to_bits());
        put(1, lane, model.radius[node].to_bits());
        put(2, lane, model.friction[node].to_bits());
        put(3, lane, model.length[j].to_bits());
        put(4, lane, if j == 0 { 0.0f32 } else { model.lo[j] }.to_bits());
        put(5, lane, if j == 0 { 0.0f32 } else { model.hi[j] }.to_bits());
        let q = if j == 0 { start.th0 } else { start.q[j] };
        put(6, lane, q.to_bits());
        put(7, lane, if j == 0 { model.mass[0] } else { 0.0 }.to_bits());
        put(8, lane, model.radius[pivot].to_bits());
        put(9, lane, topo);
        put(10, lane, ancestors[j]);
        put(11, lane, node as u32);
    }
    // Muscles and the lists of the muscle ends each bone carries.
    let count = model.muscles.len();
    let rounds = count.div_ceil(w);
    let mut muscles = vec![0f32; rounds * MUSCLE_FIELDS * w];
    let mut lists: Vec<Vec<Vec<u8>>> = vec![vec![Vec::new(); w]; rounds];
    for (k, m) in model.muscles.iter().enumerate() {
        let (round, lane) = (k / w, k % w);
        let la = lane_of_bone[m.bone_a];
        let lb = lane_of_bone[m.bone_b];
        let sensor = m
            .sensor
            .map_or(0, |node| (lane_of_node(node) as u32) << 10 | 1 << 15);
        let packed = la as u32 | (lb as u32) << 5 | sensor;
        let strength = m.strength * model.muscle_scale;
        let limits = physics::limits();
        let values = [
            f32::from_bits(packed),
            m.anchor_a,
            m.anchor_b,
            m.amplitude,
            m.hill,
            m.inv_period,
            m.phase,
            m.duty,
            m.stiffness,
            m.inv_duty,
            m.inv_complement,
            m.reset,
            limits.muscle_force * strength,
            1.0 / (limits.muscle_energy * strength),
            m.tendon_k,
            m.long,
        ];
        let at = ((round * w) + lane) * MUSCLE_FIELDS;
        muscles[at..at + MUSCLE_FIELDS].copy_from_slice(&values);
        lists[round][la].push((2 * lane) as u8);
        lists[round][lb].push((2 * lane + 1) as u8);
    }
    let words = lists
        .iter()
        .flatten()
        .map(|l| l.len().div_ceil(4))
        .max()
        .unwrap_or(0);
    let mut ends = vec![u32::MAX; rounds * words * w];
    for (round, per_lane) in lists.iter().enumerate() {
        for (lane, list) in per_lane.iter().enumerate() {
            for (e, &slot) in list.iter().enumerate() {
                let at = (round * words + e / 4) * w + lane;
                let shift = 8 * (e % 4);
                ends[at] = (ends[at] & !(0xff << shift)) | u32::from(slot) << shift;
            }
        }
    }
    let head = [
        nodes as u32 | depth << 8 | (rounds as u32) << 16 | (words as u32) << 24,
        count as u32,
        hash,
        0,
        0,
        model.total_mass.to_bits(),
        model.inv_mass.to_bits(),
        0,
    ];
    Packed {
        lanes,
        muscles,
        ends,
        head,
        key: (rounds as u32, depth, nodes as u32),
    }
}

/// Packs the creatures at `indices` of `pop` into one batch per lane class.
/// A batch's `capacity` is its lane count, which is also the node stride of a
/// recorded frame (`creature_kernel::frame_stride`).
pub fn pack(pop: &Population, indices: &[usize], cfg: &Config) -> Result<Vec<LaneBatch>> {
    let mut groups: Vec<Vec<(usize, usize)>> = vec![Vec::new(); CLASSES.len()];
    for (slot, &i) in indices.iter().enumerate() {
        let g = &pop.genomes[i];
        let Some(w) = class_of(g.node_count, g.muscle_count) else {
            bail!(
                "Unsupported body for the CUDA kernel: {} nodes, {} muscles",
                g.node_count,
                g.muscle_count
            );
        };
        if g.node_count < 2 {
            bail!("Unsupported body size");
        }
        groups[CLASSES.iter().position(|&c| c == w).unwrap()].push((slot, i));
    }
    let mut batches = Vec::new();
    for (class, members) in groups.into_iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let w = CLASSES[class];
        let mut packed: Vec<(usize, usize, Packed)> = members
            .into_par_iter()
            .map(|(slot, i)| {
                let creature = pop.creature(i);
                let model = Model::new(&creature, cfg);
                let hash = physics::quake_hash(pop.genomes[i].id);
                (slot, i, pack_creature(&model, cfg, hash, w))
            })
            .collect();
        // Similar bodies share a warp: the same loop counts.
        packed.sort_by_key(|(_, i, p)| (p.key, *i));
        let count = packed.len();
        let mut wave = WavePack {
            lanes: Vec::with_capacity(count * LANE_FIELDS * w),
            muscles: Vec::new(),
            ends: Vec::new(),
            heads: Vec::with_capacity(2 * count),
        };
        let mut info = Vec::with_capacity(count);
        for (_, i, p) in &packed {
            let mut head = p.head;
            head[3] = wave.muscles.len() as u32;
            head[4] = wave.ends.len() as u32;
            wave.lanes.extend_from_slice(&p.lanes);
            wave.muscles.extend_from_slice(&p.muscles);
            wave.ends.extend_from_slice(&p.ends);
            wave.heads.push([head[0], head[1], head[2], head[3]]);
            wave.heads.push([head[4], head[5], head[6], head[7]]);
            let g = &pop.genomes[*i];
            info.push([
                g.node_count as u32,
                g.bone_count as u32,
                g.muscle_count as u32,
                head[2],
            ]);
        }
        if wave.muscles.is_empty() {
            wave.muscles.push(0.0);
        }
        if wave.ends.is_empty() {
            wave.ends.push(u32::MAX);
        }
        batches.push(LaneBatch {
            capacity: w,
            slots: packed.iter().map(|(slot, _, _)| *slot).collect(),
            creatures: packed.iter().map(|(_, i, _)| *i).collect(),
            nodes: Vec::new(),
            info,
            tiles: Vec::new(),
            muscles: Vec::new(),
            muscle_fields: MUSCLE_FIELDS,
            bones: Vec::new(),
            results: None,
            wave: Some(wave),
        });
    }
    // The longest trials first.
    batches.reverse();
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_hold_nodes_and_muscles() {
        assert_eq!(class_of(3, 2), Some(8));
        assert_eq!(class_of(8, 32), Some(8));
        assert_eq!(class_of(8, 33), Some(16));
        assert_eq!(class_of(13, 24), Some(16));
        assert_eq!(class_of(17, 10), Some(32));
        assert_eq!(class_of(32, 96), Some(32));
        assert_eq!(class_of(33, 10), None);
    }

    #[test]
    fn a_random_population_packs_breadth_first() {
        let cfg = Config {
            population: 300,
            random_seed: false,
            ..Config::default()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let indices: Vec<usize> = (0..pop.genomes.len()).collect();
        let batches = pack(&pop, &indices, &cfg).unwrap();
        let total: usize = batches.iter().map(|b| b.slots.len()).sum();
        assert_eq!(total, indices.len());
        for batch in &batches {
            let w = batch.capacity;
            let wave = batch.wave.as_ref().unwrap();
            for c in 0..batch.slots.len() {
                let nn = (wave.heads[2 * c][0] & 255) as usize;
                let rec = &wave.lanes[c * LANE_FIELDS * w..(c + 1) * LANE_FIELDS * w];
                for lane in 1..nn {
                    let topo = rec[9 * w + lane];
                    let (pivot, parent, level) = (topo & 31, (topo >> 5) & 31, (topo >> 10) & 31);
                    let parent_level = (rec[9 * w + parent as usize] >> 10) & 31;
                    let pivot_level = (rec[9 * w + pivot as usize] >> 10) & 31;
                    if lane > 1 {
                        assert_eq!(parent_level + 1, level);
                        assert!(pivot_level < level);
                    }
                    let (fc, nch) = ((topo >> 15) & 31, (topo >> 20) & 63);
                    for k in 0..nch {
                        let child = rec[9 * w + (fc + k) as usize];
                        assert_eq!((child >> 5) & 31, lane as u32);
                    }
                }
            }
        }
    }
}
