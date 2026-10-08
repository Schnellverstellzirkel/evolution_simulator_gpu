//! Packs creatures into the buffers of the CUDA kernel (`shaders/creature.cu`)
//! and writes the kernel's source text and launch parameters.
//!
//! One GPU thread simulates one creature, which has a record of node and bone
//! words in `WavePack::lanes`, a run of muscle records in `WavePack::muscles`
//! and two head words in `WavePack::heads`. Node `j + 1` is the child end of
//! bone `j` and node 0 is the head, as `physics2::Model` numbers them.
//! `engine` calls `pack_reusing`, and `cuda_engine` compiles `cuda_source` and
//! launches the kernel with `params`.
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

/// Largest body the kernel runs (`MAXN` in the kernel), and the node slots of
/// a recorded frame.
pub const MAX_NODES: usize = 32;
/// Most muscles a body may have (`MAXM` in the kernel).
pub const MAX_MUSCLES: usize = crate::evolution::MAX_MUSCLES;
/// Words per node record: mass, radius, friction, start x, start y, foot. The
/// kernel has its own `NODE_WORDS` of the same value.
pub const NODE_WORDS: usize = 6;
/// Words per bone record: pivot node, length, parent bone (`u32::MAX` for the
/// neck), joint range low and high, and one spare. The kernel has its own
/// `BONE_WORDS` of the same value.
pub const BONE_WORDS: usize = 6;
/// Floats per muscle record, five lines of four. Eight floats are `MusclePull`
/// and nine are `MuscleRhythm` in the kernel, and three are spare. The kernel
/// calls this number `MUSCLE_WORDS`.
pub const MUSCLE_FIELDS: usize = 20;
/// Lane classes the engine compiles a kernel for. A creature is one thread,
/// so there is one class, named by the node slots of its frames.
pub const CLASSES: [usize; 1] = [MAX_NODES];
/// Substeps per step. At the standard 60 steps per second that is 960
/// substeps per second.
pub const SUBSTEPS: u32 = 16;
/// Threads per block.
pub const BLOCK: u32 = 128;
/// Blocks of `BLOCK` threads per multiprocessor that the kernel's
/// `__launch_bounds__` asks the compiler to fit. That limits the registers of a
/// thread.
pub const MIN_BLOCKS: u32 = 3;
/// Most creatures in one kernel launch (a wave).
pub const WAVE: usize = 262_144;

/// The creature data of one batch, ready for upload, in host memory the
/// engine copies from directly and reuses for later units.
#[derive(Default)]
pub struct WavePack {
    /// Node and bone records, one run per creature. A run is its node records
    /// of `NODE_WORDS` words, then its bone records of `BONE_WORDS` words.
    pub lanes: HostVec<u32>,
    /// Muscle records, one run per creature, `MUSCLE_FIELDS` floats each.
    pub muscles: HostVec<f32>,
    /// One word that the kernel does not read. It keeps the engine's buffers
    /// in their places.
    pub ends: HostVec<u32>,
    /// Two head words per creature, four `u32` each (see `pack_reusing`). The
    /// first is the node count, the muscle count, the quake hash and the word
    /// offset of its run in `lanes`. The second is the float offset of its run
    /// in `muscles`, its total mass and the inverse of it as bits, and its
    /// first muscle's period as a half float in the low 16 bits with its
    /// creature flags (`rungs::AUDIT` and the others) in the high 16 bits.
    pub heads: HostVec<[u32; 4]>,
}

/// The lane class of a body, or `None` when the kernel cannot run it.
pub fn class_of(nodes: usize, muscles: usize) -> Option<usize> {
    (nodes <= MAX_NODES && muscles <= MAX_MUSCLES).then_some(MAX_NODES)
}

/// The effect switches of a world: bit `b` is on when the effect
/// `FLAG_NAMES[b]` is in it. An effect that is off leaves no code in the
/// kernel, and the effects of the ground are off in a world without ground.
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

/// Effect names indexed by their bit position in `world_flags`. `cuda_source`
/// defines each name as 1 or 0 in the kernel. `ICE` is the `patches` effect.
const FLAG_NAMES: [&str; 12] = [
    "GROUND", "TERRAIN", "SLOPE", "GAPS", "HURDLES", "QUAKE", "MUD", "WATER", "ICE", "WIND", "AIR",
    "BRAMBLES",
];

/// The effects compiled into a world's kernel, in words for the loading
/// screen ("Mud, Wind"), or "calm" when there is none to name.
pub fn world_label(flags: u32) -> String {
    let on: Vec<String> = FLAG_NAMES
        .iter()
        .enumerate()
        // Ground is in every world that has a floor, so it is not named.
        // Quakes set the terrain switch too, so terrain is named only
        // without a quake.
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

/// The launch parameters of the kernel, passed by value. The layout is the
/// same as `Params` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    /// Creatures in this wave.
    pub count: u32,
    /// Index in the batch of the first creature of this wave.
    pub base: u32,
    /// Timed steps of the trial, after the settling steps.
    pub steps: u32,
    /// The step at whose end the early screen applies, or `u32::MAX` when the
    /// trial has no screen.
    pub screen_step: u32,
    /// Slots in one recorded frame (`creature_kernel::frame_stride`). Only a
    /// recording kernel reads it.
    pub stride: u32,
    /// Gravity (m/s^2).
    pub gravity: f32,
    /// Velocity kept per step. The kernel reads `air_sub` instead.
    pub air: f32,
    /// Ground friction. It multiplies the friction coefficient of each node.
    pub friction: f32,
    /// Bump height (m) of the ground's roughness level.
    pub terrain: f32,
    /// Multiplier on each muscle's energy store. The kernel reads
    /// `inv_muscle_energy` instead.
    pub muscle_energy: f32,
    /// Multiplier on the recovery of the muscle energy stores.
    pub muscle_recovery: f32,
    /// Ground slope, rise over run.
    pub slope: f32,
    /// Horizontal wind acceleration (m/s^2).
    pub wind: f32,
    /// Mud sink depth (m).
    pub mud: f32,
    /// Pit opening width (m).
    pub gaps: f32,
    /// Hurdle height (m).
    pub hurdles: f32,
    /// Earthquake base bump height (m).
    pub quake: f32,
    /// The distance (m) under which the early screen stops a creature that is
    /// neither young nor reshaped. It is negative infinity when the trial has
    /// no screen.
    pub screen_bar: f32,
    /// The screen bar of a young creature (`rungs::YOUNG`).
    pub screen_bar_young: f32,
    /// The screen bar of a reshaped one (`rungs::RESHAPED`).
    pub screen_bar_reshaped: f32,
    /// Water line height (m).
    pub water: f32,
    /// Ice patch strength, the share of friction that a patch takes away.
    pub patches: f32,
    /// Velocity kept per substep.
    pub air_sub: f32,
    /// One over `muscle_energy`.
    pub inv_muscle_energy: f32,
    /// Drag (1/s) on the nodes that are not feet while they touch the ground.
    pub brambles: f32,
    /// The early rung at 1 s (`rungs::Rung`). A trial without rungs carries a
    /// rule that never stops it.
    pub r1: Rung,
    /// The early rung at 2.5 s.
    pub r2: Rung,
}

/// The parameters of the wave of `count` creatures that starts at creature
/// `base` of a batch. `stride` is the number of slots in one recorded frame.
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

/// The developer override `EVOLUTION_WARP_<NAME>` of an engine setting, or
/// `default` when it is unset or not a whole number. For example `CARVEOUT`
/// is the share of each multiprocessor's memory kept as shared memory, in
/// percent. A player never needs it.
pub fn solver_setting(name: &str, default: u32) -> u32 {
    std::env::var(format!("EVOLUTION_WARP_{name}"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The kernel source for world `flags` and `fidelity`: `#define` lines that set
/// the constants and switches of the kernel, then `shaders/creature.cu`. With
/// `record` the kernel also writes every frame of the trial. `_class` is the
/// lane class. It is not read, because there is only one.
pub fn cuda_source(_class: usize, flags: u32, fidelity: Fidelity, record: bool) -> String {
    let limits = physics::limits();
    // A float literal for the kernel. Debug formatting always writes a decimal
    // point or an exponent.
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
    // The physics constants the kernel uses, as float defines.
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
    // an edited copy of it, so a kernel edit needs no Rust rebuild.
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
/// records into `muscles`. `pack_reusing` writes its two head words.
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
        // The node whose touchdown restarts the rhythm, or `u32::MAX` for none.
        let sensor = m.sensor.map_or(u32::MAX, |node| node as u32);
        // The pivot node and the tip node of each of the two bones, a byte
        // each.
        let a0 = model.pivot[m.bone_a] as u32;
        let b0 = model.pivot[m.bone_b] as u32;
        let nodes = a0 | (m.bone_a as u32 + 1) << 8 | b0 << 16 | (m.bone_b as u32 + 1) << 24;
        let values = [
            // `MusclePull`, which every substep reads.
            f32::from_bits(nodes),
            m.anchor_a,
            m.anchor_b,
            m.hill,
            limits.muscle_force * strength,
            1.0 / (limits.muscle_energy * strength),
            m.tendon_k,
            m.long,
            // `MuscleRhythm`, which the rhythm reads once a step.
            m.amplitude,
            m.inv_period,
            m.phase,
            m.duty,
            m.inv_duty,
            m.inv_complement,
            m.stiffness,
            m.reset,
            f32::from_bits(sensor),
            // Spare words that fill the last line.
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
// SAFETY: `pack_reusing` gives each thread only the ranges of its own creature.
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

/// Packs the creatures at `indices` of `pop` into a batch and returns it as a
/// list of one. The `capacity` of the batch is the node slots of a recorded
/// frame (`creature_kernel::frame_stride`). It fails for a body the kernel
/// cannot run.
pub fn pack(pop: &Population, indices: &[usize], cfg: &Config) -> Result<Vec<LaneBatch>> {
    pack_reusing(pop, indices, cfg, &mut Vec::new())
}

/// The batch of `spare` to pack a unit into, given the words of node and bone
/// records, the floats of muscle records and the head words it needs. Among
/// the batches whose memory already holds the unit it is the one with the
/// least, so that big memory stays free for big units. When none holds it, it
/// is the one with the most, which grows. A batch with more than 8 times the
/// memory the unit needs, and a megabyte, is never taken: it stays for the big
/// units, because a small unit that took it would leave the next big unit
/// with a small batch to grow. `None` when `spare` has no batch to take.
fn best_spare(spare: &[LaneBatch], records: usize, muscles: usize, heads: usize) -> Option<usize> {
    let held = |b: &LaneBatch| {
        b.wave.as_ref().map_or(0, |w| {
            w.lanes.held_bytes()
                + w.muscles.held_bytes()
                + w.ends.held_bytes()
                + w.heads.held_bytes()
        })
    };
    let holds = |b: &LaneBatch| {
        b.wave.as_ref().is_some_and(|w| {
            w.lanes.capacity() >= records
                && w.muscles.capacity() >= muscles
                && w.heads.capacity() >= heads
        })
    };
    let need = 4 * records + 4 * muscles + 16 * heads;
    let limit = 8 * need + (1 << 20);
    let taken = || (0..spare.len()).filter(|&i| held(&spare[i]) <= limit);
    taken()
        .filter(|&i| holds(&spare[i]))
        .min_by_key(|&i| (held(&spare[i]), i))
        .or_else(|| taken().max_by_key(|&i| (held(&spare[i]), std::cmp::Reverse(i))))
}

/// `pack` into the memory of a batch from `spare` (the batches of a unit that
/// finished), when there is one: the smallest that holds the unit
/// (`best_spare`). The creatures are packed sorted by muscle count, then node
/// count, then population index. `slots` and `creatures` of the batch say
/// which creature each packed one is.
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
    // Sorting by muscle count, then node count, gives the bodies that run side
    // by side in a warp similar loop counts.
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
    let reused = best_spare(spare, record_len.max(1), muscle_len.max(1), 2 * count)
        .map(|at| spare.swap_remove(at));
    let (mut wave, mut slots, mut creatures, mut info) = match reused {
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
        // The period of the first muscle, a feature of the early rungs.
        let period = if g.muscle_count > 0 {
            pop.muscles[g.muscle_start].period
        } else {
            0.0
        };
        // An empty `pop.flags` means no creature has a flag.
        let flags = u32::from(pop.flags.get(i).copied().unwrap_or(0));
        // SAFETY: creature `c` owns its record from `record_at[c]`, its muscles
        // from `muscle_at[c]`, each as long as its size, and heads `2 * c` and
        // `2 * c + 1`. The ranges are disjoint and inside the buffers above.
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
    // Per creature: its node, bone and muscle counts and its quake hash.
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

    /// A unit packs into the smallest spare batch that holds it, into the
    /// biggest when none does, and never into one that is much too big, which
    /// stays for the big units.
    #[test]
    fn a_unit_packs_into_the_smallest_spare_that_holds_it() {
        let make = |n: usize| {
            let cfg = Config {
                population: n,
                random_seed: false,
                ..Config::default()
            };
            let pop = crate::evolution::create(&cfg).unwrap();
            let indices: Vec<usize> = (0..pop.genomes.len()).collect();
            pack(&pop, &indices, &cfg).unwrap().pop().unwrap()
        };
        let (small, medium, big) = (make(300), make(3000), make(30000));
        let need = |b: &LaneBatch| {
            let w = b.wave.as_ref().unwrap();
            (w.lanes.len(), w.muscles.len(), w.heads.len())
        };
        let (s, m, b) = (need(&small), need(&medium), need(&big));
        let spare = vec![big, medium, small];
        // A small unit takes the small batch, a medium one the medium batch.
        assert_eq!(best_spare(&spare, s.0, s.1, s.2), Some(2));
        assert_eq!(best_spare(&spare, m.0, m.1, m.2), Some(1));
        // A big one takes the big batch, and so does one that nothing holds.
        assert_eq!(best_spare(&spare, b.0, b.1, b.2), Some(0));
        assert_eq!(best_spare(&spare, 10 * b.0, b.1, b.2), Some(0));
        // The big batch is much too big for a small unit, and stays.
        assert_eq!(best_spare(&spare[..1], s.0, s.1, s.2), None);
        assert_eq!(best_spare(&[], 1, 1, 1), None);
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
