//! One-creature-per-lane GPU kernel (`shaders/physics_creature.wgsl`).
//!
//! Creatures are grouped by padded node capacity. Inside a group they are
//! sorted by body size so the 32 creatures that share a warp run similar loop
//! counts. Muscles and bones are packed per 32-creature tile as
//! `[item][field][lane]`, which makes every warp load one coalesced line.
use crate::{
    evolution::Population,
    physics::{self, Node},
};
use anyhow::{Result, ensure};
use rayon::prelude::*;

pub const TILE: usize = 32;
pub const MUSCLE_FIELDS: usize = 11;
pub const BONE_FIELDS: usize = 2;
pub const CAPACITIES: [usize; 12] = [3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub tick: u32,
    pub steps: u32,
    pub stride: u32,
    pub count: u32,
    pub gravity: f32,
    pub air: f32,
    pub friction: f32,
    pub ground: f32,
    pub total_steps: u32,
    pub pad: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuResult {
    pub fitness: f32,
    pub ground_contact: f32,
    pub vertical_oscillation: f32,
    pub gait_frequency: f32,
    pub previous_center_y: f32,
    pub vertical_extremum: f32,
    pub vertical_trend: f32,
    pub gait_turns: f32,
    pub height_sum: f32,
}

/// One node-capacity group, ready for upload.
pub struct LaneBatch {
    pub capacity: usize,
    /// Position of each packed creature within the caller's index slice.
    pub slots: Vec<usize>,
    /// Population index of each packed creature.
    pub creatures: Vec<usize>,
    pub nodes: Vec<Node>,
    pub info: Vec<[u32; 4]>,
    pub tiles: Vec<[u32; 4]>,
    pub muscles: Vec<f32>,
    pub bones: Vec<f32>,
}

pub fn capacity_index(nodes: usize) -> usize {
    CAPACITIES
        .iter()
        .position(|&c| nodes <= c)
        .unwrap_or(CAPACITIES.len() - 1)
}

/// Waveform amplitude: the full stroke, limited so the target length changes
/// at most `MAX_MUSCLE_LENGTH_SPEED` (2 m/s).
pub fn muscle_amplitude(m: &crate::evolution::Muscle) -> f32 {
    (m.long - m.short)
        .min(2.0 * 2.0 * m.period * m.duty.min(1.0 - m.duty) / std::f32::consts::PI)
}

/// Hash of a creature's bone and muscle attachment layout.
pub fn plan_hash(pop: &Population, index: usize) -> u64 {
    let g = &pop.genomes[index];
    let mix = |h: u64, v: u32| (h ^ u64::from(v)).wrapping_mul(0x100_0000_01b3);
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in &pop.bones[g.bone_start..g.bone_start + g.bone_count] {
        h = mix(mix(h, b.a), b.b);
    }
    for m in &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count] {
        h = mix(mix(h, m.bone_a), m.bone_b);
    }
    h
}

pub fn pack(pop: &Population, indices: &[usize]) -> Result<Vec<LaneBatch>> {
    let mut groups: [Vec<(usize, usize)>; CAPACITIES.len()] = Default::default();
    for (slot, &i) in indices.iter().enumerate() {
        let n = pop.genomes[i].node_count;
        ensure!((1..=64).contains(&n), "Unsupported body size");
        groups[capacity_index(n)].push((slot, i));
    }
    Ok(groups
        .into_par_iter()
        .enumerate()
        .filter(|(_, g)| !g.is_empty())
        .map(|(group, mut members)| {
            let capacity = CAPACITIES[group];
            // Identical body plans share a warp: same loop counts and index patterns.
            members.sort_by_cached_key(|&(_, i)| {
                let g = &pop.genomes[i];
                (g.node_count, g.muscle_count, plan_hash(pop, i), i)
            });
            let count = members.len();
            let tile_count = count.div_ceil(TILE);
            let mut tiles = Vec::with_capacity(tile_count);
            let mut muscle_len = 0usize;
            let mut bone_len = 0usize;
            for tile in members.chunks(TILE) {
                let max_muscles = tile
                    .iter()
                    .map(|&(_, i)| pop.genomes[i].muscle_count)
                    .max()
                    .unwrap_or(0);
                let max_bones = tile
                    .iter()
                    .map(|&(_, i)| pop.genomes[i].bone_count)
                    .max()
                    .unwrap_or(0);
                tiles.push([
                    muscle_len as u32,
                    bone_len as u32,
                    max_muscles as u32,
                    max_bones as u32,
                ]);
                muscle_len += max_muscles * MUSCLE_FIELDS * TILE;
                bone_len += max_bones * BONE_FIELDS * TILE;
            }
            let mut nodes = vec![Node::default(); count * capacity];
            let mut info = Vec::with_capacity(count);
            let mut muscles = vec![0f32; muscle_len.max(1)];
            let mut bones = vec![0f32; bone_len.max(1)];

            for (j, &(_, i)) in members.iter().enumerate() {
                let g = &pop.genomes[i];
                let genes = &pop.nodes[g.node_start..g.node_start + g.node_count];
                for (dst, gene) in nodes[j * capacity..].iter_mut().zip(genes) {
                    *dst = physics::node(gene);
                }
                info.push([
                    g.node_count as u32,
                    g.bone_count as u32,
                    g.muscle_count as u32,
                    0,
                ]);
                let tile = tiles[j / TILE];
                let lane = j % TILE;
                let source_bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
                for (b, bone) in source_bones.iter().enumerate() {
                    let field = tile[1] as usize + b * BONE_FIELDS * TILE + lane;
                    bones[field] = f32::from_bits(bone.a | (bone.b << 8));
                    bones[field + TILE] = bone.rest_length;
                }
                let source = &pop.muscles[g.muscle_start..g.muscle_start + g.muscle_count];
                for (m, muscle) in source.iter().enumerate() {
                    let bone_a = source_bones[muscle.bone_a as usize];
                    let bone_b = source_bones[muscle.bone_b as usize];
                    let field = tile[0] as usize + m * MUSCLE_FIELDS * TILE + lane;
                    let values = [
                        f32::from_bits(
                            bone_a.a | (bone_a.b << 8) | (bone_b.a << 16) | (bone_b.b << 24),
                        ),
                        muscle.anchor_a,
                        muscle.anchor_b,
                        muscle_amplitude(muscle),
                        muscle.long,
                        1.0 / muscle.period,
                        muscle.phase,
                        muscle.duty,
                        muscle.stiffness,
                        1.0 / muscle.duty,
                        1.0 / (1.0 - muscle.duty),
                    ];
                    for (f, value) in values.into_iter().enumerate() {
                        muscles[field + f * TILE] = value;
                    }
                }
            }
            LaneBatch {
                capacity,
                slots: members.iter().map(|&(slot, _)| slot).collect(),
                creatures: members.iter().map(|&(_, i)| i).collect(),
                nodes,
                info,
                tiles,
                muscles,
                bones,
            }
        })
        .collect())
}

pub fn shader_source(capacity: usize, workgroup: u32) -> String {
    let mut source = include_str!("../shaders/physics_creature.wgsl").to_owned();
    if capacity >= 24 {
        // Constant bounds let compilers unroll every node and bone loop. That
        // keeps small bodies in registers, but RADV's compile time explodes for
        // large bodies, so those loops use the runtime size instead.
        source = source
            .replace("j < MAXN; j++)", "j < body_nodes; j++)")
            .replace("j < MAXB; j++)", "j < bone_count; j++)");
    }
    let (bone_passes, velocity_passes) = crate::physics::solver_passes();
    let source = source
        .replace(
            "const BONE_SOLVE_ITERATIONS: u32 = 8u;",
            &format!("const BONE_SOLVE_ITERATIONS: u32 = {bone_passes}u;"),
        )
        .replace(
            "const VELOCITY_SOLVE_ITERATIONS: u32 = 4u;",
            &format!("const VELOCITY_SOLVE_ITERATIONS: u32 = {velocity_passes}u;"),
        )
        .replace("PHYSICSRATE", &format!("{:.1}", crate::physics::rate() as f32))
        .replace("SETTLESTEPSu", &format!("{}u", crate::physics::settle()))
        .replace(
            "SAMPLEINTERVALu",
            &format!("{}u", crate::physics::sample_interval()),
        )
        .replace("TURNCOS", &format!("{:.9}", crate::physics::turn_limits().0))
        .replace("TURNTAN", &format!("{:.9}", crate::physics::turn_limits().1))
        .replace("SHAREDLEN", &(capacity * workgroup as usize).to_string())
        .replace("WGSIZEu", &format!("{workgroup}u"))
        .replace("WGSIZE", &workgroup.to_string())
        .replace("MAXNODESu", &format!("{capacity}u"));
    crate::gpu::apply_fast_cos(source)
}
