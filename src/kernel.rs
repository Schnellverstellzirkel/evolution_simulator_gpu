//! Packing and kernel source of the creature kernel (`shaders/creature.cu`).
//!
//! One GPU thread simulates one creature. Each creature has a flat record of
//! node and bone words in `WavePack::lanes`, a run of muscle records in
//! `WavePack::muscles`, and two head words in `WavePack::heads`. Node `j + 1`
//! is the child of bone `j` and node 0 is the head, as `physics2::Model`
//! numbers them.
use crate::{
    config::Config,
    creature_kernel::LaneBatch,
    cuda_engine::HostVec,
    evolution::Population,
    physics::{self, Fidelity},
    physics2::{self, Model},
    rungs::Rung,
};
use anyhow::{Result, bail};
use rayon::prelude::*;

/// Largest body the kernel runs, and the node slots of a recorded frame.
pub const MAX_NODES: usize = 32;
/// Most muscles a body may have.
pub const MAX_MUSCLES: usize = crate::evolution::MAX_MUSCLES;
/// Words per node record: mass, radius, friction, start x, start y, foot.
pub const NODE_WORDS: usize = 6;
/// Words per bone record: pivot node, length, parent bone, joint range low
/// and high, and one spare.
pub const BONE_WORDS: usize = 6;
/// Floats per muscle record (`MusclePull` and `MuscleRhythm` in the kernel).
pub const MUSCLE_FIELDS: usize = 20;
/// Lane classes the engine compiles a kernel for. A creature is one thread,
/// so there is one class, named by the node slots of its frames.
pub const CLASSES: [usize; 1] = [MAX_NODES];
/// Substeps per step: 16 at the standard 60 steps per second (960 per
/// second).
pub const SUBSTEPS: u32 = 16;
/// Threads per block, and blocks per multiprocessor the kernel is built for.
pub const BLOCK: u32 = 128;
pub const MIN_BLOCKS: u32 = 3;
/// Creatures per kernel launch (a wave).
pub const WAVE: usize = 262_144;

/// The creature data of one batch, ready for upload, in host memory the
/// engine copies from directly and reuses for later units.
#[derive(Default)]
pub struct WavePack {
    /// Node and bone records, one run per creature.
    pub lanes: HostVec<u32>,
    /// Muscle records, one run per creature.
    pub muscles: HostVec<f32>,
    /// Unused; kept so the engine's buffers keep their places.
    pub ends: HostVec<u32>,
    /// Two words of four per creature (see `fill_creature`).
    pub heads: HostVec<[u32; 4]>,
}

/// The lane class of a body, or `None` when the kernel cannot run it.
pub fn class_of(nodes: usize, muscles: usize) -> Option<usize> {
    (nodes <= MAX_NODES && muscles <= MAX_MUSCLES).then_some(MAX_NODES)
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
    set(11, ground && cfg.brambles > 0.0);
    flags
}

/// Effect names indexed by their bit position in `world_flags`.
const FLAG_NAMES: [&str; 12] = [
    "GROUND", "TERRAIN", "SLOPE", "GAPS", "HURDLES", "QUAKE", "MUD", "WATER", "ICE", "WIND", "AIR",
    "BRAMBLES",
];

/// The effects compiled into a world's kernel, in words for the loading
/// screen ("Mud, Wind"), or "calm" when it has none beyond flat ground.
pub fn world_label(flags: u32) -> String {
    let on: Vec<String> = FLAG_NAMES
        .iter()
        .enumerate()
        // Ground is in every world that has a floor, and quakes set the
        // terrain switch too, so it names a rough floor only without them.
        .filter(|&(bit, _)| {
            flags & (1 << bit) != 0 && (bit > 1 || (bit == 1 && flags & (1 << 5) == 0))
        })
        .map(|(_, name)| {
            let lower = name.to_lowercase();
            let mut chars = lower.chars();
            chars
                .next()
                .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        })
        .collect();
    if on.is_empty() {
        "calm".to_owned()
    } else {
        on.join(", ")
    }
}

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
    /// The screen bars of a young creature and of a reshaped one
    /// (`rungs::YOUNG`, `rungs::RESHAPED`).
    pub screen_bar_young: f32,
    pub screen_bar_reshaped: f32,
    pub water: f32,
    pub patches: f32,
    /// Velocity kept per substep.
    pub air_sub: f32,
    pub inv_muscle_energy: f32,
    pub brambles: f32,
    /// The early rungs at 1 s and 2.5 s (`rungs::Rung`); a rule that never
    /// stops for a trial without them.
    pub r1: Rung,
    pub r2: Rung,
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
        screen_bar_young: cfg
            .screen
            .map_or(f32::NEG_INFINITY, |screen| screen.young_bar),
        screen_bar_reshaped: cfg
            .screen
            .map_or(f32::NEG_INFINITY, |screen| screen.reshaped_bar),
        water: cfg.water,
        patches: if ground { cfg.patches } else { 0.0 },
        air_sub: air.powf(1.0 / SUBSTEPS as f32),
        inv_muscle_energy: 1.0 / cfg.muscle_energy,
        brambles: if ground { cfg.brambles } else { 0.0 },
        r1: cfg.rungs.map_or(Rung::NEVER, |r| r.0[0]),
        r2: cfg.rungs.map_or(Rung::NEVER, |r| r.0[1]),
    }
}

/// A developer override `EVOLUTION_WARP_<NAME>` of an engine setting (never
/// needed to play).
pub fn solver_setting(name: &str, default: u32) -> u32 {
    std::env::var(format!("EVOLUTION_WARP_{name}"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The kernel source for world `flags`, `fidelity` and, with `record`, the
/// frame output. `_class` is the one lane class.
pub fn cuda_source(_class: usize, flags: u32, fidelity: Fidelity, record: bool) -> String {
    let limits = physics::limits();
    let float = |value: f32| format!("{value:?}f");
    let mut defines: Vec<(String, String)> = vec![
        ("RATE".into(), format!("{:.1}f", fidelity.rate as f32)),
        ("SUBSTEPS".into(), format!("{SUBSTEPS}u")),
        ("SETTLE".into(), format!("{}u", fidelity.settle())),
        ("SAMPLE".into(), format!("{}u", fidelity.sample_interval())),
        ("MAXN".into(), format!("{MAX_NODES}u")),
        ("MAXM".into(), format!("{MAX_MUSCLES}u")),
        ("BLOCK".into(), format!("{BLOCK}u")),
        ("MIN_BLOCKS".into(), format!("{MIN_BLOCKS}")),
        ("RECORD".into(), (if record { "1" } else { "0" }).into()),
    ];
    for (bit, name) in FLAG_NAMES.iter().enumerate() {
        defines.push((
            (*name).into(),
            (if flags & (1 << bit) != 0 { "1" } else { "0" }).into(),
        ));
    }
    let constants = [
        ("MUSCLE_RECOVERY", float(limits.muscle_recovery)),
        ("INV_JOINT_DAMPING", float(1.0 / physics2::joint_damping())),
        ("JOINT_BREAK", float(physics::JOINT_BREAK)),
        ("MUD_NORMAL", float(physics::MUD_NORMAL)),
        ("MUD_GRIP", float(physics::MUD_GRIP)),
        ("MUD_DRAG", float(physics::MUD_DRAG)),
        ("MUD_FULL_DEPTH", float(physics::MUD_FULL_DEPTH)),
        ("BRAMBLE_REACH", float(physics::BRAMBLE_REACH)),
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
    // A developer working on the kernel may point `EVOLUTION_KERNEL_SOURCE` at
    // a copy of it to skip rebuilds.
    match std::env::var("EVOLUTION_KERNEL_SOURCE")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
    {
        Some(text) => source.push_str(&text),
        None => source.push_str(include_str!("../shaders/creature.cu")),
    }
    source
}

/// Writes one creature's node and bone records into `record` and its muscle
/// records into `muscles`; `pack_reusing` writes its two head words.
fn fill_creature(model: &Model, cfg: &Config, record: &mut [u32], muscles: &mut [f32]) {
    let start = model.start(cfg);
    let bones = model.pivot.len();
    let mut children = [0u32; MAX_NODES];
    for j in 1..bones {
        if let Some(p) = model.parent[j] {
            children[p] += 1;
        }
    }
    for i in 0..=bones {
        // A foot ends a leg of at least two bones: no bone below it, and the
        // bone above it is not the neck and has no other child.
        let foot = i > 0 && {
            let j = i - 1;
            children[j] == 0
                && matches!(model.parent[j], Some(p) if j > 0 && p > 0 && children[p] == 1)
        };
        let r = &mut record[i * NODE_WORDS..(i + 1) * NODE_WORDS];
        r[0] = model.mass[i].to_bits();
        r[1] = model.radius[i].to_bits();
        r[2] = model.friction[i].to_bits();
        r[3] = start.pos[i][0].to_bits();
        r[4] = start.pos[i][1].to_bits();
        r[5] = u32::from(foot);
    }
    let bone_at = (bones + 1) * NODE_WORDS;
    for j in 0..bones {
        let r = &mut record[bone_at + j * BONE_WORDS..bone_at + (j + 1) * BONE_WORDS];
        r[0] = model.pivot[j] as u32;
        r[1] = model.length[j].to_bits();
        r[2] = match model.parent[j] {
            Some(p) if j > 0 => p as u32,
            _ => u32::MAX,
        };
        r[3] = model.lo[j].to_bits();
        r[4] = model.hi[j].to_bits();
        r[5] = 0;
    }
    let limits = physics::limits();
    for (k, m) in model.muscles.iter().enumerate() {
        let strength = m.strength * model.muscle_scale;
        let sensor = m.sensor.map_or(u32::MAX, |node| node as u32);
        let a0 = model.pivot[m.bone_a] as u32;
        let b0 = model.pivot[m.bone_b] as u32;
        let nodes = a0 | (m.bone_a as u32 + 1) << 8 | b0 << 16 | (m.bone_b as u32 + 1) << 24;
        let values = [
            // What every substep reads.
            f32::from_bits(nodes),
            m.anchor_a,
            m.anchor_b,
            m.hill,
            limits.muscle_force * strength,
            1.0 / (limits.muscle_energy * strength),
            m.tendon_k,
            m.long,
            // What the rhythm reads once a step.
            m.amplitude,
            m.inv_period,
            m.phase,
            m.duty,
            m.inv_duty,
            m.inv_complement,
            m.stiffness,
            m.reset,
            f32::from_bits(sensor),
            0.0,
            0.0,
            0.0,
        ];
        muscles[k * MUSCLE_FIELDS..(k + 1) * MUSCLE_FIELDS].copy_from_slice(&values);
    }
}

/// Raw buffer pointers that parallel fills write through, each creature in
/// its own range.
#[derive(Clone, Copy)]
struct Out {
    lanes: *mut u32,
    muscles: *mut f32,
    heads: *mut [u32; 4],
}
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

/// Packs the creatures at `indices` of `pop` into one batch. Its `capacity`
/// is the node slots of a recorded frame (`creature_kernel::frame_stride`).
pub fn pack(pop: &Population, indices: &[usize], cfg: &Config) -> Result<Vec<LaneBatch>> {
    pack_reusing(pop, indices, cfg, &mut Vec::new())
}

/// `pack` into the memory of a batch from `spare` (one of a unit that
/// finished), when there is one.
pub fn pack_reusing(
    pop: &Population,
    indices: &[usize],
    cfg: &Config,
    spare: &mut Vec<LaneBatch>,
) -> Result<Vec<LaneBatch>> {
    for &i in indices {
        let g = &pop.genomes[i];
        if g.node_count < 2 || class_of(g.node_count, g.muscle_count).is_none() {
            bail!(
                "Unsupported body for the CUDA kernel: {} nodes, {} muscles",
                g.node_count,
                g.muscle_count
            );
        }
    }
    // Similar bodies run side by side in a warp: the same loop counts.
    let mut sorted: Vec<(usize, usize)> = indices.iter().copied().enumerate().collect();
    sorted.par_sort_by_key(|&(_, i)| {
        let g = &pop.genomes[i];
        (g.muscle_count, g.node_count, i)
    });
    let count = sorted.len();
    let (mut record_at, mut muscle_at) = (Vec::with_capacity(count), Vec::with_capacity(count));
    let (mut record_len, mut muscle_len) = (0usize, 0usize);
    for &(_, i) in &sorted {
        let g = &pop.genomes[i];
        record_at.push(record_len);
        muscle_at.push(muscle_len);
        record_len += g.node_count * NODE_WORDS + g.bone_count * BONE_WORDS;
        muscle_len += g.muscle_count * MUSCLE_FIELDS;
    }
    let (mut wave, mut slots, mut creatures, mut info) = match spare.pop() {
        Some(mut b) => {
            b.slots.clear();
            b.creatures.clear();
            b.info.clear();
            (
                b.wave.take().unwrap_or_default(),
                b.slots,
                b.creatures,
                b.info,
            )
        }
        None => Default::default(),
    };
    wave.lanes.reset(record_len.max(1), 0);
    wave.muscles.reset(muscle_len.max(1), 0.0);
    wave.ends.reset(1, 0);
    wave.heads.reset(2 * count, [0; 4]);
    let out = Out {
        lanes: wave.lanes.as_mut_ptr(),
        muscles: wave.muscles.as_mut_ptr(),
        heads: wave.heads.as_mut_ptr(),
    };
    sorted.par_iter().enumerate().for_each(|(c, &(_, i))| {
        // Captures the whole `Send` wrapper, not its pointer fields.
        #[allow(clippy::redundant_locals)]
        let out = out;
        let g = &pop.genomes[i];
        let creature = pop.creature(i);
        let model = Model::new(&creature, cfg);
        let record_size = g.node_count * NODE_WORDS + g.bone_count * BONE_WORDS;
        let muscle_size = g.muscle_count * MUSCLE_FIELDS;
        let period = if g.muscle_count > 0 {
            pop.muscles[g.muscle_start].period
        } else {
            0.0
        };
        let flags = u32::from(pop.flags.get(i).copied().unwrap_or(0));
        // SAFETY: creature `c` owns its record from record_at[c], its muscles
        // from muscle_at[c], each as long as its size, and heads 2c and
        // 2c + 1; the ranges are disjoint and inside the buffers above.
        unsafe {
            let record = std::slice::from_raw_parts_mut(out.lanes.add(record_at[c]), record_size);
            let muscles =
                std::slice::from_raw_parts_mut(out.muscles.add(muscle_at[c]), muscle_size);
            fill_creature(&model, cfg, record, muscles);
            *out.heads.add(2 * c) = [
                g.node_count as u32,
                g.muscle_count as u32,
                physics::quake_hash(g.id),
                record_at[c] as u32,
            ];
            *out.heads.add(2 * c + 1) = [
                muscle_at[c] as u32,
                model.total_mass.to_bits(),
                model.inv_mass.to_bits(),
                u32::from(crate::rungs::period_half(period)) | flags << 16,
            ];
        }
    });
    info.extend(sorted.iter().enumerate().map(|(c, &(_, i))| {
        let g = &pop.genomes[i];
        [
            g.node_count as u32,
            g.bone_count as u32,
            g.muscle_count as u32,
            wave.heads[2 * c][2],
        ]
    }));
    slots.extend(sorted.iter().map(|&(slot, _)| slot));
    creatures.extend(sorted.iter().map(|&(_, i)| i));
    Ok(vec![LaneBatch {
        capacity: MAX_NODES,
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
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_world_is_named_by_its_effects() {
        // Bit 0 is ground, 1 terrain, 5 quake, 6 mud, 9 wind.
        assert_eq!(world_label(1), "calm");
        assert_eq!(world_label(1 | 1 << 6 | 1 << 9), "Mud, Wind");
        assert_eq!(world_label(1 | 1 << 1), "Terrain");
        assert_eq!(world_label(1 | 1 << 1 | 1 << 5), "Quake");
        assert_eq!(world_label(0), "calm");
    }

    #[test]
    fn a_random_population_packs_every_creature_once() {
        let cfg = Config {
            population: 300,
            random_seed: false,
            ..Config::default()
        };
        let pop = crate::evolution::create(&cfg).unwrap();
        let indices: Vec<usize> = (0..pop.genomes.len()).collect();
        let batches = pack(&pop, &indices, &cfg).unwrap();
        assert_eq!(batches.len(), 1);
        let mut seen = batches[0].slots.clone();
        seen.sort_unstable();
        assert_eq!(seen, indices);
    }
}
