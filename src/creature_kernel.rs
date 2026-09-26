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
pub const MUSCLE_FIELDS: usize = 15;
/// Per bone: endpoints and joint reference node, rest length, and the joint
/// range constants of `physics::Joint`.
pub const BONE_FIELDS: usize = 9;
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
    /// Bump height of the rough ground (m); 0 is flat.
    pub terrain: f32,
    /// Multiplier on the muscle energy store (heat wave); 1.0 is calm.
    pub muscle_energy: f32,
    /// Multiplier on muscle energy recovery (drought); 1.0 is calm.
    pub muscle_recovery: f32,
    /// Ground slope (rise over run), zeroed when the ground is disabled.
    pub slope: f32,
    /// Steady horizontal wind acceleration (m/s²); positive pushes +x.
    pub wind: f32,
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
    /// Bits of nodes 0-31 / 32-63 that touched the ground (stored as f32 bits).
    pub contact_lo: f32,
    pub contact_hi: f32,
    /// Bits of touching nodes that later lifted clear of the ground again.
    pub lift_lo: f32,
    pub lift_hi: f32,
    /// Nodes grounded after the last step (f32 bits), for sensor touchdowns.
    pub ground_lo: f32,
    pub ground_hi: f32,
    /// Seconds into the trial when the head tipped below its neck base, or 0
    /// if the creature stayed upright. Fitness is the distance at the fall.
    pub fall_time: f32,
    /// Mean head acceleration (m/s^2) over about `physics::HEAD_SHAKE_WINDOW`
    /// seconds, for the head shaking limit.
    pub head_shake: f32,
}
impl GpuResult {
    /// Number of feet: nodes that touched the ground and lifted off again.
    /// A node dragged along the ground never lifts, so it is not a foot.
    pub fn feet(&self) -> u32 {
        self.lift_lo.to_bits().count_ones() + self.lift_hi.to_bits().count_ones()
    }
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
/// at most `physics::limits().muscle_speed`.
pub fn muscle_amplitude(m: &crate::evolution::Muscle) -> f32 {
    (m.long - m.short).min(
        2.0 * crate::physics::limits().muscle_speed * m.period * m.duty.min(1.0 - m.duty)
            / std::f32::consts::PI,
    )
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
                let body_bones = &pop.bones[g.bone_start..g.bone_start + g.bone_count];
                for (dst, state) in nodes[j * capacity..]
                    .iter_mut()
                    .zip(physics::body(genes, body_bones))
                {
                    *dst = state;
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
                let joints = physics::joints(genes, source_bones);
                for (b, (bone, joint)) in source_bones.iter().zip(&joints).enumerate() {
                    let field = tile[1] as usize + b * BONE_FIELDS * TILE + lane;
                    // A free joint points its reference at its own pivot; the
                    // kernel skips it because its half range cosine is -1.
                    let reference = joint.reference.unwrap_or(bone.a as usize) as u32;
                    let values = [
                        f32::from_bits(bone.a | (bone.b << 8) | (reference << 16)),
                        bone.rest_length,
                        joint.center[0],
                        joint.center[1],
                        joint.half[0],
                        joint.half[1],
                        joint.child_share,
                        joint.child_mass,
                        joint.reference_mass,
                    ];
                    for (f, value) in values.into_iter().enumerate() {
                        bones[field + f * TILE] = value;
                    }
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
                        f32::from_bits(muscle.sensor),
                        muscle.reset,
                        0.0,
                        1.0,
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

pub fn shader_source(
    capacity: usize,
    workgroup: u32,
    fidelity: crate::physics::Fidelity,
) -> String {
    let mut source = include_str!("../shaders/physics_creature.wgsl").to_owned();
    if capacity >= 24 {
        // Constant bounds let compilers unroll every node and bone loop. That
        // keeps small bodies in registers, but RADV's compile time explodes for
        // large bodies, so those loops use the runtime size instead.
        source = source
            .replace("j < MAXN; j++)", "j < body_nodes; j++)")
            .replace("j < MAXB; j++)", "j < bone_count; j++)");
    }
    let (bone_passes, velocity_passes) = (fidelity.bone_passes, fidelity.velocity_passes);
    let limits = crate::physics::limits();
    let source = source
        .replace(
            "const MAX_MUSCLE_LENGTH_SPEED: f32 = 2.0;",
            &format!(
                "const MAX_MUSCLE_LENGTH_SPEED: f32 = {:?};",
                limits.muscle_speed
            ),
        )
        .replace(
            "const MUSCLE_CAPACITY: f32 = 15.0;",
            &format!("const MUSCLE_CAPACITY: f32 = {:?};", limits.muscle_energy),
        )
        .replace(
            "const MUSCLE_RECOVERY: f32 = 0.25;",
            &format!("const MUSCLE_RECOVERY: f32 = {:?};", limits.muscle_recovery),
        )
        .replace(
            "const MAX_MUSCLE_FORCE: f32 = 5.0;",
            &format!("const MAX_MUSCLE_FORCE: f32 = {:?};", limits.muscle_force),
        )
        .replace(
            "const PLANTED_SPEED: f32 = 0.01;",
            &format!(
                "const PLANTED_SPEED: f32 = {:?};",
                crate::physics::PLANTED_SPEED
            ),
        )
        .replace(
            "const STANCE_GRIP: f32 = 10.0;",
            &format!(
                "const STANCE_GRIP: f32 = {:?};",
                crate::physics::stance_grip()
            ),
        )
        .replace(
            "const HEAD_SHAKE_LIMIT: f32 = 78.4;",
            &format!(
                "const HEAD_SHAKE_LIMIT: f32 = {:?};",
                crate::physics::HEAD_SHAKE_LIMIT
            ),
        )
        .replace(
            "const HEAD_SHAKE_WINDOW: f32 = 0.1;",
            &format!(
                "const HEAD_SHAKE_WINDOW: f32 = {:?};",
                crate::physics::HEAD_SHAKE_WINDOW
            ),
        )
        .replace(
            "const MAX_NODE_SPEED: f32 = 5.0;",
            &format!("const MAX_NODE_SPEED: f32 = {:?};", limits.node_speed),
        )
        .replace(
            "const MAX_BONE_ANGULAR_SPEED: f32 = 15.0;",
            &format!(
                "const MAX_BONE_ANGULAR_SPEED: f32 = {:?};",
                limits.bone_spin
            ),
        )
        .replace(
            "const BONE_SOLVE_ITERATIONS: u32 = 8u;",
            &format!("const BONE_SOLVE_ITERATIONS: u32 = {bone_passes}u;"),
        )
        .replace(
            "const VELOCITY_SOLVE_ITERATIONS: u32 = 4u;",
            &format!("const VELOCITY_SOLVE_ITERATIONS: u32 = {velocity_passes}u;"),
        )
        .replace("PHYSICSRATE", &format!("{:.1}", fidelity.rate as f32))
        .replace("SETTLESTEPSu", &format!("{}u", fidelity.settle()))
        .replace(
            "SAMPLEINTERVALu",
            &format!("{}u", fidelity.sample_interval()),
        )
        .replace(
            "JOINTBREAKCOS",
            &format!("{:.9}", physics::JOINT_BREAK.cos()),
        )
        .replace(
            "JOINTBREAKSIN",
            &format!("{:.9}", physics::JOINT_BREAK.sin()),
        )
        .replace("TURNCOS", &format!("{:.9}", fidelity.turn_limits().0))
        .replace("TURNTAN", &format!("{:.9}", fidelity.turn_limits().1))
        .replace("SHAREDLEN", &(capacity * workgroup as usize).to_string())
        .replace("WGSIZEu", &format!("{workgroup}u"))
        .replace("WGSIZE", &workgroup.to_string())
        .replace("MAXNODESu", &format!("{capacity}u"));
    crate::gpu::apply_fast_cos(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::Fidelity;

    #[test]
    fn params_layout_matches_the_kernel_uniform() {
        // The WGSL `Params` struct must read every field at these offsets.
        // Adding or reordering a scalar can silently shift later fields.
        assert_eq!(std::mem::offset_of!(Params, gravity), 16);
        assert_eq!(std::mem::offset_of!(Params, total_steps), 32);
        assert_eq!(std::mem::offset_of!(Params, terrain), 36);
        assert_eq!(std::mem::offset_of!(Params, muscle_energy), 40);
        assert_eq!(std::mem::offset_of!(Params, muscle_recovery), 44);
        assert_eq!(std::mem::offset_of!(Params, slope), 48);
        assert_eq!(std::mem::offset_of!(Params, wind), 52);
        assert_eq!(std::mem::size_of::<Params>(), 56);
    }

    #[test]
    fn kernels_compile_for_standard_and_fine_physics() {
        for fidelity in [Fidelity::standard(), Fidelity::fine()] {
            for capacity in CAPACITIES {
                let source = shader_source(capacity, 32, fidelity);
                for constant in [
                    "PHYSICSRATE",
                    "SETTLESTEPSu",
                    "TURNCOS",
                    "TURNTAN",
                    "MAXNODESu",
                ] {
                    assert!(!source.contains(constant), "{constant} left unreplaced");
                }
                crate::vk_engine::spirv(&source)
                    .unwrap_or_else(|e| panic!("{capacity}-node kernel at {fidelity:?}: {e:#}"));
            }
        }
    }
}
