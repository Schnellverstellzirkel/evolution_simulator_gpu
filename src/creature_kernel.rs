//! One-creature-per-lane GPU kernel (`shaders/physics_creature.wgsl`).
//!
//! Creatures are grouped by padded node capacity. Inside a group they are
//! sorted by body size so the 32 creatures that share a warp run similar loop
//! counts. Muscles and bones are packed per 32-creature tile as
//! `[item][field][lane]`, which makes every warp load one coalesced line.
use crate::{
    config::Config,
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

struct GroupBuffers {
    nodes: wgpu::Buffer,
    muscles: wgpu::Buffer,
    bones: wgpu::Buffer,
    results: wgpu::Buffer,
    info: wgpu::Buffer,
    tiles: wgpu::Buffer,
    bind: wgpu::BindGroup,
    /// Byte capacities of nodes, muscles, bones, and creature count.
    capacity: [u64; 4],
}

pub struct CreatureKernel {
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::ComputePipeline>,
    workgroup: u32,
    groups: [Option<GroupBuffers>; CAPACITIES.len()],
    params: Option<(wgpu::Buffer, u64)>,
    readback: Option<(wgpu::Buffer, u64)>,
    params_stride: u64,
    pub allocated_bytes: u64,
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

impl CreatureKernel {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let workgroup = std::env::var("EVOLUTION_LANE_WG")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v == 32 || *v == 64)
            .unwrap_or(32);
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Creature lane resources"),
            entries: &[
                entry(0, storage(false)),
                entry(1, storage(true)),
                entry(2, storage(true)),
                entry(
                    3,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<Params>() as u64
                        ),
                    },
                ),
                entry(4, storage(false)),
                entry(5, storage(true)),
                entry(6, storage(true)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Creature lane pipeline"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = CAPACITIES
            .iter()
            .map(|&capacity| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Creature lane physics"),
                    source: wgpu::ShaderSource::Wgsl(shader_source(capacity, workgroup).into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Creature lane physics"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("advance"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        Ok(Self {
            layout,
            pipelines,
            workgroup,
            groups: Default::default(),
            params: None,
            readback: None,
            params_stride: (std::mem::size_of::<Params>() as u64).next_multiple_of(alignment),
            allocated_bytes: 0,
        })
    }

    fn ensure_buffers(
        &mut self,
        device: &wgpu::Device,
        batches: &[LaneBatch],
        dispatches: u64,
    ) -> Result<()> {
        let create = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(64).next_power_of_two(),
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let params_bytes = dispatches * self.params_stride;
        if self.params.as_ref().is_none_or(|(_, c)| *c < params_bytes) {
            let buffer = create(
                "Creature lane parameters",
                params_bytes,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let size = buffer.size();
            self.params = Some((buffer, size));
            // Bind groups reference the old parameter buffer.
            self.groups = Default::default();
        }
        let result_bytes = batches.iter().map(|b| b.info.len()).sum::<usize>() as u64
            * std::mem::size_of::<GpuResult>() as u64;
        if self
            .readback
            .as_ref()
            .is_none_or(|(_, c)| *c < result_bytes)
        {
            let buffer = create(
                "Creature lane readback",
                result_bytes,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            );
            let size = buffer.size();
            self.readback = Some((buffer, size));
        }
        let limit = device.limits().max_storage_buffer_binding_size;
        for batch in batches {
            let group = capacity_index(batch.capacity);
            let need = [
                std::mem::size_of_val(batch.nodes.as_slice()) as u64,
                std::mem::size_of_val(batch.muscles.as_slice()) as u64,
                std::mem::size_of_val(batch.bones.as_slice()) as u64,
                batch.info.len() as u64,
            ];
            ensure!(
                need[..3].iter().all(|&b| b <= limit),
                "GPU batch exceeds storage binding limits"
            );
            if self.groups[group]
                .as_ref()
                .is_some_and(|g| g.capacity.iter().zip(need).all(|(c, n)| *c >= n))
            {
                continue;
            }
            let count = need[3].max(1).next_power_of_two();
            let nodes = create(
                "Creature lane nodes",
                need[0],
                storage | wgpu::BufferUsages::COPY_SRC,
            );
            let muscles = create("Creature lane muscles", need[1], storage);
            let bones = create("Creature lane bones", need[2], storage);
            let results = create(
                "Creature lane results",
                count * std::mem::size_of::<GpuResult>() as u64,
                storage | wgpu::BufferUsages::COPY_SRC,
            );
            let info = create("Creature lane info", count * 16, storage);
            let tiles = create(
                "Creature lane tiles",
                count.div_ceil(TILE as u64) * 16,
                storage,
            );
            let params = &self.params.as_ref().unwrap().0;
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Creature lane resources"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: nodes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: muscles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bones.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: params,
                            offset: 0,
                            size: std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: results.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: info.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: tiles.as_entire_binding(),
                    },
                ],
            });
            let capacity = [nodes.size(), muscles.size(), bones.size(), count];
            self.groups[group] = Some(GroupBuffers {
                nodes,
                muscles,
                bones,
                results,
                info,
                tiles,
                bind,
                capacity,
            });
        }
        self.allocated_bytes = self
            .groups
            .iter()
            .flatten()
            .map(|g| {
                g.nodes.size()
                    + g.muscles.size()
                    + g.bones.size()
                    + g.results.size()
                    + g.info.size()
                    + g.tiles.size()
            })
            .sum::<u64>()
            + self.params.as_ref().map_or(0, |(b, _)| b.size())
            + self.readback.as_ref().map_or(0, |(b, _)| b.size());
        Ok(())
    }

    /// Uploads, simulates all `steps` in `chunk`-step dispatches, and reads back results.
    pub fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batches: &[LaneBatch],
        cfg: &Config,
        steps: u32,
        chunk: u32,
    ) -> Result<Vec<Vec<GpuResult>>> {
        let dispatches = batches.len() as u64 * u64::from(steps.div_ceil(chunk));
        self.ensure_buffers(device, batches, dispatches)?;
        let params_buffer = &self.params.as_ref().unwrap().0;
        let mut plan = Vec::new();
        for batch in batches {
            let group = capacity_index(batch.capacity);
            let buffers = self.groups[group].as_ref().unwrap();
            queue.write_buffer(&buffers.nodes, 0, bytemuck::cast_slice(&batch.nodes));
            queue.write_buffer(&buffers.muscles, 0, bytemuck::cast_slice(&batch.muscles));
            queue.write_buffer(&buffers.bones, 0, bytemuck::cast_slice(&batch.bones));
            queue.write_buffer(&buffers.info, 0, bytemuck::cast_slice(&batch.info));
            queue.write_buffer(&buffers.tiles, 0, bytemuck::cast_slice(&batch.tiles));
            let groups = batch.info.len().div_ceil(self.workgroup as usize) as u32;
            ensure!(groups <= 65_535, "GPU batch has too many workgroups");
            for tick in (0..steps).step_by(chunk as usize) {
                let offset = plan.len() as u64 * self.params_stride;
                let params = Params {
                    tick,
                    steps: (steps - tick).min(chunk),
                    stride: batch.capacity as u32,
                    count: batch.info.len() as u32,
                    gravity: cfg.gravity,
                    air: crate::physics::air_per_step(cfg.air_retention),
                    friction: cfg.ground_friction,
                    ground: if cfg.ground { 1.0 } else { 0.0 },
                    total_steps: steps,
                    pad: [0; 3],
                };
                queue.write_buffer(params_buffer, offset, bytemuck::bytes_of(&params));
                plan.push((group, offset as u32, groups));
            }
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Creature lane batches"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Creature lane advance"),
                timestamp_writes: None,
            });
            for &(group, offset, groups) in &plan {
                pass.set_pipeline(&self.pipelines[group]);
                pass.set_bind_group(0, &self.groups[group].as_ref().unwrap().bind, &[offset]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
        let readback = &self.readback.as_ref().unwrap().0;
        let mut offset = 0u64;
        for batch in batches {
            let bytes = (batch.info.len() * std::mem::size_of::<GpuResult>()) as u64;
            let group = capacity_index(batch.capacity);
            encoder.copy_buffer_to_buffer(
                &self.groups[group].as_ref().unwrap().results,
                0,
                readback,
                offset,
                bytes,
            );
            offset += bytes;
        }
        queue.submit([encoder.finish()]);
        let flat = crate::gpu::read_buffer::<GpuResult>(device, readback, offset)?;
        let mut out = Vec::with_capacity(batches.len());
        let mut start = 0;
        for batch in batches {
            out.push(flat[start..start + batch.info.len()].to_vec());
            start += batch.info.len();
        }
        Ok(out)
    }
}
