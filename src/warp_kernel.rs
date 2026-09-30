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
    cuda_engine::HostVec,
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
pub const PGS_SWEEPS: u32 = 2;
pub const CLEAN_SWEEPS: u32 = 1;
/// Threads per block, and blocks per multiprocessor the register budget is
/// set for: 4 blocks of 128 threads fit 128 registers per thread.
pub const BLOCK: u32 = 128;
pub const MIN_BLOCKS: u32 = 4;
/// Creatures per kernel launch (a wave).
pub const WAVE: usize = 262_144;

/// The creature data of one lane class, ready for upload, in host memory
/// the engine copies from directly and reuses for later units.
#[derive(Default)]
pub struct WavePack {
    /// `[creature][field][lane]` lane records.
    pub lanes: HostVec<u32>,
    /// `[round][lane][field]` muscle records, per creature at its `heads`
    /// offset.
    pub muscles: HostVec<f32>,
    /// `[round][word][lane]` lists of the muscle ends each bone carries: four
    /// byte slots per word (muscle lane times two plus the end), 255 for none.
    pub ends: HostVec<u32>,
    /// Two words of four per creature (see `shaders/warp_creature.cu`).
    pub heads: HostVec<[u32; 4]>,
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
        // The substep ladder's switches (docs/plan-2m.md, section 6 item 3),
        // all off in the game: the realized-friction-work ledger, anchored
        // friction with its energy store, spin-adaptive substeps (SUBSTEPS
        // becomes the count for fast steps, calm steps run one), and the
        // diagnostic result words. Rungs: L0 is SUBSTEPS=1; L1 adds LEDGER=1;
        // L2 adds ANCHOR=1; L2.5 is SUBSTEPS=2 ADAPT=1 LEDGER=1 ANCHOR=1.
        ("LEDGER".into(), format!("{}", solver_setting("LEDGER", 0))),
        ("ANCHOR".into(), format!("{}", solver_setting("ANCHOR", 0))),
        ("ADAPT".into(), format!("{}", solver_setting("ADAPT", 0))),
        ("DIAG".into(), format!("{}", solver_setting("DIAG", 0))),
        (
            "PROFILE".into(),
            format!("{}", solver_setting("PROFILE", 0)),
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
    // A developer working on the kernel may point `EVOLUTION_WARP_SOURCE` at
    // a copy of it to skip rebuilds.
    match std::env::var("EVOLUTION_WARP_SOURCE").ok().and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(text) => source.push_str(&text),
        None => source.push_str(include_str!("../shaders/warp_creature.cu")),
    }
    source
}

/// A creature's size in its class's buffers and its place in the sort, from
/// its genes alone: muscle rounds, end-list words per round, tree depth.
#[derive(Clone, Copy)]
struct Size {
    rounds: usize,
    words: usize,
    depth: u32,
    nodes: usize,
}

fn size_of(pop: &Population, i: usize, w: usize) -> Size {
    let g = &pop.genomes[i];
    let bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
    let muscles = &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
    // Bone levels: the neck is 1; a bone is one below the bone ending at its
    // pivot (the neck for a pivot at the head). Genes keep parents first.
    let mut end_of = [usize::MAX; MAX_NODES];
    let mut level = [0u32; MAX_NODES];
    let mut depth = 1;
    for (j, b) in bones.iter().enumerate() {
        let parent = if j == 0 {
            usize::MAX
        } else if b.a == 0 {
            0
        } else {
            end_of[b.a as usize]
        };
        level[j] = if parent == usize::MAX { 1 } else { level[parent] + 1 };
        depth = depth.max(level[j]);
        end_of[b.b as usize] = j;
    }
    let rounds = muscles.len().div_ceil(w);
    let mut count = [[0u8; MAX_NODES]; ROUNDS];
    let mut most = 0usize;
    for (k, m) in muscles.iter().enumerate() {
        for bone in [m.bone_a, m.bone_b] {
            let c = &mut count[k / w][bone as usize];
            *c += 1;
            most = most.max(*c as usize);
        }
    }
    Size {
        rounds,
        words: most.div_ceil(4),
        depth,
        nodes: g.node_count,
    }
}

/// Writes one creature's lane records, muscle records and end lists into
/// its slices of its class's buffers, and returns its two head words.
fn fill_creature(
    model: &Model,
    cfg: &Config,
    hash: u32,
    w: usize,
    size: Size,
    lanes: &mut [u32],
    muscles: &mut [f32],
    ends: &mut [u32],
) -> [u32; 8] {
    let start = model.start(cfg);
    let bones = model.pivot.len();
    let nodes = bones + 1;
    // Breadth-first bone order, with each bone's children as a list.
    let mut first = [usize::MAX; MAX_NODES];
    let mut next = [usize::MAX; MAX_NODES];
    let mut last = [usize::MAX; MAX_NODES];
    let mut children = [0u32; MAX_NODES];
    for j in 1..bones {
        let parent = model.parent[j].unwrap_or(0);
        if first[parent] == usize::MAX {
            first[parent] = j;
        } else {
            next[last[parent]] = j;
        }
        last[parent] = j;
        children[parent] += 1;
    }
    let mut order = [0usize; MAX_NODES];
    let (mut len, mut at) = (1, 0);
    while at < len {
        let mut c = first[order[at]];
        while c != usize::MAX {
            order[len] = c;
            len += 1;
            c = next[c];
        }
        at += 1;
    }
    let mut lane_of_bone = [0usize; MAX_NODES];
    for (k, &j) in order[..bones].iter().enumerate() {
        lane_of_bone[j] = k + 1;
    }
    let lane_of_node = |node: usize| if node == 0 { 0 } else { lane_of_bone[node - 1] };
    let mut level = [0u32; MAX_NODES];
    let mut ancestors = [0u32; MAX_NODES];
    for &j in &order[..bones] {
        let (up_level, up_mask) = match model.parent[j] {
            Some(p) if j > 0 => (level[p], ancestors[p]),
            _ => (0, 0),
        };
        level[j] = up_level + 1;
        ancestors[j] = up_mask | 1 << lane_of_bone[j];
    }
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
        let first_child = if first[j] == usize::MAX { 0 } else { lane_of_bone[first[j]] };
        let topo = lane_of_node(pivot) as u32
            | (parent_lane as u32) << 5
            | level[j] << 10
            | (first_child as u32) << 15
            | children[j] << 20;
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
    // Muscles, and the ends each bone carries: four byte slots per word
    // (muscle lane times two plus the end), 255 for none.
    let limits = physics::limits();
    let mut filled = [[0usize; MAX_NODES]; ROUNDS];
    let words = size.words;
    for (k, m) in model.muscles.iter().enumerate() {
        let (round, lane) = (k / w, k % w);
        let la = lane_of_bone[m.bone_a];
        let lb = lane_of_bone[m.bone_b];
        let sensor = m
            .sensor
            .map_or(0, |node| (lane_of_node(node) as u32) << 10 | 1 << 15);
        let packed = la as u32 | (lb as u32) << 5 | sensor;
        let strength = m.strength * model.muscle_scale;
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
        for (end_lane, slot) in [(la, 2 * lane), (lb, 2 * lane + 1)] {
            let e = filled[round][end_lane];
            filled[round][end_lane] += 1;
            let at = (round * words + e / 4) * w + end_lane;
            let shift = 8 * (e % 4);
            ends[at] = (ends[at] & !(0xff << shift)) | (slot as u32) << shift;
        }
    }
    [
        nodes as u32 | size.depth << 8 | (size.rounds as u32) << 16 | (words as u32) << 24,
        model.muscles.len() as u32,
        hash,
        0,
        0,
        model.total_mass.to_bits(),
        model.inv_mass.to_bits(),
        0,
    ]
}

/// Raw buffer pointers that parallel fills write through, each creature in
/// its own range.
#[derive(Clone, Copy)]
struct Out {
    lanes: *mut u32,
    muscles: *mut f32,
    ends: *mut u32,
    heads: *mut [u32; 4],
}
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

/// Packs the creatures at `indices` of `pop` into one batch per lane class.
/// A batch's `capacity` is its lane count, which is also the node stride of a
/// recorded frame (`creature_kernel::frame_stride`).
pub fn pack(pop: &Population, indices: &[usize], cfg: &Config) -> Result<Vec<LaneBatch>> {
    pack_reusing(pop, indices, cfg, &mut Vec::new())
}

/// `pack` into the memory of batches from `spare` (those of units that
/// finished), taken from it by lane class; the ones not needed stay there.
pub fn pack_reusing(
    pop: &Population,
    indices: &[usize],
    cfg: &Config,
    spare: &mut Vec<LaneBatch>,
) -> Result<Vec<LaneBatch>> {
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
        let mut sized: Vec<(usize, usize, Size)> = members
            .into_par_iter()
            .map(|(slot, i)| (slot, i, size_of(pop, i, w)))
            .collect();
        // Similar bodies share a warp: the same loop counts.
        sized.par_sort_by_key(|&(_, i, z)| (z.rounds, z.depth, z.nodes, i));
        let count = sized.len();
        let (mut muscle_at, mut end_at) = (Vec::with_capacity(count), Vec::with_capacity(count));
        let (mut muscle_len, mut end_len) = (0usize, 0usize);
        for &(_, _, z) in &sized {
            muscle_at.push(muscle_len);
            end_at.push(end_len);
            muscle_len += z.rounds * w * MUSCLE_FIELDS;
            end_len += z.rounds * z.words * w;
        }
        // A spare batch of this class first: its buffers have the sizes
        // this class needed last time.
        let reuse = spare
            .iter()
            .position(|b| b.capacity == w)
            .or((!spare.is_empty()).then_some(spare.len() - 1))
            .map(|at| spare.swap_remove(at));
        let (mut wave, mut slots, mut creatures, mut info) = match reuse {
            Some(mut b) => {
                b.slots.clear();
                b.creatures.clear();
                b.info.clear();
                (b.wave.take().unwrap_or_default(), b.slots, b.creatures, b.info)
            }
            None => Default::default(),
        };
        wave.lanes.reset(count * LANE_FIELDS * w, 0);
        wave.muscles.reset(muscle_len.max(1), 0.0);
        wave.ends.reset(end_len.max(1), u32::MAX);
        wave.heads.reset(2 * count, [0; 4]);
        let out = Out {
            lanes: wave.lanes.as_mut_ptr(),
            muscles: wave.muscles.as_mut_ptr(),
            ends: wave.ends.as_mut_ptr(),
            heads: wave.heads.as_mut_ptr(),
        };
        sized.par_iter().enumerate().for_each(|(c, &(_, i, z))| {
            let out = out;
            let creature = pop.creature(i);
            let model = Model::new(&creature, cfg);
            let hash = physics::quake_hash(pop.genomes[i].id);
            // SAFETY: creature `c` owns lanes [c * LANE_FIELDS * w, ..),
            // muscles from muscle_at[c] and ends from end_at[c], each as long
            // as its size says, and heads 2c and 2c + 1; the ranges are
            // disjoint and inside the buffers allocated above.
            unsafe {
                let lanes = std::slice::from_raw_parts_mut(out.lanes.add(c * LANE_FIELDS * w), LANE_FIELDS * w);
                let muscles = std::slice::from_raw_parts_mut(
                    out.muscles.add(muscle_at[c]),
                    z.rounds * w * MUSCLE_FIELDS,
                );
                let ends = std::slice::from_raw_parts_mut(out.ends.add(end_at[c]), z.rounds * z.words * w);
                let mut head = fill_creature(&model, cfg, hash, w, z, lanes, muscles, ends);
                head[3] = muscle_at[c] as u32;
                head[4] = end_at[c] as u32;
                *out.heads.add(2 * c) = [head[0], head[1], head[2], head[3]];
                *out.heads.add(2 * c + 1) = [head[4], head[5], head[6], head[7]];
            }
        });
        info.extend(sized.iter().enumerate().map(|(c, &(_, i, _))| {
            let g = &pop.genomes[i];
            [
                g.node_count as u32,
                g.bone_count as u32,
                g.muscle_count as u32,
                wave.heads[2 * c][2],
            ]
        }));
        slots.extend(sized.iter().map(|&(slot, _, _)| slot));
        creatures.extend(sized.iter().map(|&(_, i, _)| i));
        batches.push(LaneBatch {
            capacity: w,
            slots,
            creatures,
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
