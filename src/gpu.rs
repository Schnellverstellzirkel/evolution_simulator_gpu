use crate::{
    config::Config,
    evolution::{Muscle, Population},
    physics::{self, Node},
    qd::{EvaluationMetrics, TrialMetrics},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta {
    nodes: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NodeAdj {
    start: u32,
    count: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    tick: u32,
    steps: u32,
    stride: u32,
    count: u32,
    gravity: f32,
    air: f32,
    friction: f32,
    ground: f32,
    total_steps: u32,
    pad: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuResult {
    fitness: f32,
    ground_contact: f32,
    // Final output is vertical range and observed cadence; simulation uses min/max scratch.
    vertical_oscillation: f32,
    gait_frequency: f32,
    previous_center_y: f32,
    vertical_extremum: f32,
    vertical_trend: f32,
    gait_turns: f32,
}
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
    pipeline: wgpu::ComputePipeline,
    buffers: Option<Buffers>,
    pub allocated_bytes: u64,
}
struct BucketBuffers {
    nodes: wgpu::Buffer,
    muscles: wgpu::Buffer,
    meta: wgpu::Buffer,
    node_adjacency: wgpu::Buffer,
    results: wgpu::Buffer,
    bind: wgpu::BindGroup,
    capacities: [u64; 4],
}
struct Buffers {
    buckets: [Option<BucketBuffers>; 4],
    params: wgpu::Buffer,
    params_stride: u64,
    params_capacity: u64,
    readback: wgpu::Buffer,
    readback_capacity: u64,
}
struct Batch {
    slots: Vec<usize>,
    creatures: Vec<usize>,
    nodes: Vec<Node>,
    muscles: Vec<Muscle>,
    metadata: Vec<Meta>,
    node_adjacency: Vec<NodeAdj>,
    stride: usize,
}
fn bucket_index(stride: usize) -> usize {
    match stride {
        8 => 0,
        16 => 1,
        32 => 2,
        64 => 3,
        _ => unreachable!("validated GPU bucket stride"),
    }
}
pub async fn adapter(name: &str) -> Result<(wgpu::Instance, wgpu::Adapter)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN).await;
    let adapter = adapters
        .into_iter()
        .find(|a| {
            a.get_info()
                .name
                .to_lowercase()
                .contains(&name.to_lowercase())
        })
        .context(format!("No Vulkan GPU matching {name:?}"))?;
    Ok((instance, adapter))
}
pub fn descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
    let limits = adapter.limits();
    wgpu::DeviceDescriptor {
        label: Some("Evolution GPU"),
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size.min(1 << 30),
            max_buffer_size: limits.max_buffer_size.min(1 << 30),
            ..wgpu::Limits::default()
        },
        ..Default::default()
    }
}
impl Gpu {
    pub fn new(name: &str) -> Result<Self> {
        pollster::block_on(async {
            let (_, a) = adapter(name).await?;
            let (device, queue) = a.request_device(&descriptor(&a)).await?;
            Self::from_device(device, queue, a.get_info().name)
        })
    }
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue, name: String) -> Result<Self> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Muscle physics"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/physics.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Physics resources"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<Params>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Physics pipeline"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batched creature physics"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("advance"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            anyhow::bail!("GPU shader initialization failed: {error}");
        }
        Ok(Self {
            device,
            queue,
            name,
            pipeline,
            buffers: None,
            allocated_bytes: 0,
        })
    }
    fn buffers(&mut self, batches: &[Batch], param_count: u64, cfg: &Config) -> Result<()> {
        let mut specs: [Option<([u64; 4], u64)>; 4] = [None; 4];
        let mut result_count = 0u64;
        for batch in batches {
            ensure!(
                matches!(batch.stride, 8 | 16 | 32 | 64),
                "Unsupported GPU bucket stride"
            );
            let bucket = bucket_index(batch.stride);
            let count = batch.metadata.len() as u64;
            let needed = [
                (std::mem::size_of_val(batch.nodes.as_slice()) as u64).max(32),
                (std::mem::size_of_val(batch.muscles.as_slice()) as u64).max(32),
                count.max(1),
                (std::mem::size_of_val(batch.node_adjacency.as_slice()) as u64).max(32),
            ]
            .map(u64::next_power_of_two);
            ensure!(specs[bucket].is_none(), "Duplicate GPU bucket stride");
            specs[bucket] = Some((needed, count));
            result_count += count;
        }
        let params_alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let params_stride =
            (std::mem::size_of::<Params>() as u64).next_multiple_of(params_alignment.max(1));
        let readback_capacity =
            (result_count.max(1) * std::mem::size_of::<GpuResult>() as u64).next_power_of_two();
        let reusable = self.allocated_bytes < cfg.gpu_budget_mib as u64 * 1024 * 1024
            && self.buffers.as_ref().is_some_and(|existing| {
                existing.params_capacity >= param_count
                    && existing.readback_capacity
                        >= result_count * std::mem::size_of::<GpuResult>() as u64
                    && batches.iter().all(|batch| {
                        let bucket = bucket_index(batch.stride);
                        existing.buckets[bucket].as_ref().is_some_and(|resources| {
                            resources
                                .capacities
                                .iter()
                                .zip(specs[bucket].unwrap().0)
                                .all(|(capacity, needed)| *capacity >= needed)
                        })
                    })
            });
        if reusable {
            return Ok(());
        }

        let bucket_bytes: u64 = specs
            .iter()
            .flatten()
            .map(|(capacities, _)| {
                capacities[0]
                    + capacities[1]
                    + capacities[2]
                        * (std::mem::size_of::<Meta>() as u64
                            + std::mem::size_of::<GpuResult>() as u64)
                    + capacities[3]
            })
            .sum();
        let params_bytes = params_stride * param_count;
        let total = bucket_bytes + params_bytes + readback_capacity;
        ensure!(
            total < cfg.gpu_budget_mib as u64 * 1024 * 1024,
            "GPU batch exceeds memory budget"
        );
        let limits = self.device.limits();
        ensure!(
            specs.iter().flatten().all(|(capacities, _)| {
                capacities[0] <= limits.max_storage_buffer_binding_size
                    && capacities[1] <= limits.max_storage_buffer_binding_size
                    && capacities[2] * std::mem::size_of::<Meta>() as u64
                        <= limits.max_storage_buffer_binding_size
                    && capacities[2] * std::mem::size_of::<GpuResult>() as u64
                        <= limits.max_storage_buffer_binding_size
                    && capacities[3] <= limits.max_storage_buffer_binding_size
            }),
            "GPU batch exceeds storage binding limits"
        );
        ensure!(
            params_bytes <= limits.max_buffer_size && readback_capacity <= limits.max_buffer_size,
            "GPU batch exceeds buffer limits"
        );
        let create = |label, size, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let params = create(
            "Physics parameters",
            params_bytes,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let readback = create(
            "Evaluation readback",
            readback_capacity,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let mut bucket_buffers: [Option<BucketBuffers>; 4] = [const { None }; 4];
        for (bucket, spec) in specs.into_iter().enumerate() {
            let Some((capacities, _)) = spec else {
                continue;
            };
            let nodes = create(
                "Node state",
                capacities[0],
                storage | wgpu::BufferUsages::COPY_SRC,
            );
            let muscles = create("Muscle genomes", capacities[1], storage);
            let meta = create(
                "Creature node counts",
                capacities[2] * std::mem::size_of::<Meta>() as u64,
                storage,
            );
            let node_adjacency = create("Node muscle adjacency", capacities[3], storage);
            let results = create(
                "Fitness and behavior descriptors",
                capacities[2] * std::mem::size_of::<GpuResult>() as u64,
                storage | wgpu::BufferUsages::COPY_SRC,
            );
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Physics resources"),
                layout: &self.pipeline.get_bind_group_layout(0),
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
                        resource: meta.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &params,
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
                        resource: node_adjacency.as_entire_binding(),
                    },
                ],
            });
            bucket_buffers[bucket] = Some(BucketBuffers {
                nodes,
                muscles,
                meta,
                node_adjacency,
                results,
                bind,
                capacities,
            });
        }
        self.buffers = Some(Buffers {
            buckets: bucket_buffers,
            params,
            params_stride,
            params_capacity: param_count,
            readback,
            readback_capacity,
        });
        self.allocated_bytes = total;
        Ok(())
    }
    pub fn evaluate(
        &mut self,
        pop: &Population,
        indices: &[usize],
        cfg: &Config,
    ) -> Result<Vec<f32>> {
        Ok(self
            .evaluate_with_metrics(pop, indices, cfg)?
            .into_iter()
            .map(|result| result.fitness)
            .collect())
    }
    pub fn evaluate_with_metrics(
        &mut self,
        pop: &Population,
        indices: &[usize],
        cfg: &Config,
    ) -> Result<Vec<EvaluationMetrics>> {
        ensure!(
            indices.iter().all(|&i| i < pop.genomes.len()),
            "Invalid creature index"
        );
        let mut out = vec![EvaluationMetrics::default(); indices.len()];
        let batches: Vec<Batch> = [8usize, 16, 32, 64]
            .into_par_iter()
            .filter_map(|stride| {
                let group: Vec<_> = indices
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, &i)| {
                        (pop.genomes[i].node_count.next_power_of_two().max(8) == stride)
                            .then_some((slot, i))
                    })
                    .collect();
                if group.is_empty() {
                    return None;
                }
                let mut nodes = vec![Node::default(); group.len() * stride];
                let muscle_count: usize = group
                    .iter()
                    .map(|&(_, i)| pop.genomes[i].muscle_count)
                    .sum();
                let mut muscles = Vec::<Muscle>::with_capacity(muscle_count * 2);
                let mut node_adjacency = vec![NodeAdj::default(); group.len() * stride];
                let mut metadata = Vec::with_capacity(group.len());
                for (j, &(_, i)) in group.iter().enumerate() {
                    let genome = &pop.genomes[i];
                    let genes =
                        &pop.nodes[genome.node_start..genome.node_start + genome.node_count];
                    for (dst, gene) in nodes[j * stride..j * stride + genes.len()]
                        .iter_mut()
                        .zip(genes)
                    {
                        *dst = physics::node(gene);
                    }
                    metadata.push(Meta {
                        nodes: genes.len() as u32,
                    });
                    let muscle_end = genome.muscle_start + genome.muscle_count;
                    append_adjacency_muscles(
                        &pop.muscles[genome.muscle_start..muscle_end],
                        genes.len(),
                        &mut node_adjacency[j * stride..(j + 1) * stride],
                        &mut muscles,
                    );
                }
                Some(Batch {
                    slots: group.iter().map(|&(slot, _)| slot).collect(),
                    creatures: group.iter().map(|&(_, creature)| creature).collect(),
                    nodes,
                    muscles,
                    metadata,
                    node_adjacency,
                    stride,
                })
            })
            .collect();
        if batches.is_empty() {
            return Ok(out);
        }
        let results = self
            .run(&batches, cfg, physics::SETTLE + cfg.steps(), false)?
            .0;
        for (batch, results) in batches.iter().zip(results) {
            for (j, &slot) in batch.slots.iter().enumerate() {
                let r = results[j];
                let active_steps = cfg.steps();
                let contact_denominator = (active_steps.max(1)
                    * pop.genomes[batch.creatures[j]].node_count as u32)
                    as f32;
                out[slot] = EvaluationMetrics {
                    fitness: r.fitness,
                    behavior: TrialMetrics {
                        ground_contact: (r.ground_contact / contact_denominator).clamp(0.0, 1.0),
                        vertical_oscillation: if r.vertical_oscillation.is_finite() {
                            r.vertical_oscillation.max(0.0)
                        } else {
                            0.0
                        },
                        gait_frequency: if r.gait_frequency.is_finite() {
                            r.gait_frequency.max(0.0)
                        } else {
                            0.0
                        },
                    },
                };
            }
        }
        Ok(out)
    }
    pub fn trajectory(
        &mut self,
        c: &crate::evolution::Creature,
        cfg: &Config,
        steps: u32,
    ) -> Result<Vec<Node>> {
        let stride = c.nodes.len().next_power_of_two().max(8);
        let mut nodes = vec![Node::default(); stride];
        nodes[..c.nodes.len()].copy_from_slice(&physics::nodes(c));
        let metadata = vec![Meta {
            nodes: c.nodes.len() as u32,
        }];
        let mut node_adjacency = vec![NodeAdj::default(); stride];
        let mut muscles = Vec::<Muscle>::with_capacity(c.muscles.len() * 2);
        append_adjacency_muscles(&c.muscles, c.nodes.len(), &mut node_adjacency, &mut muscles);
        let batch = Batch {
            slots: vec![0],
            creatures: vec![0],
            nodes,
            muscles,
            metadata,
            node_adjacency,
            stride,
        };
        let (_, mut result) = self.run(&[batch], cfg, steps, true)?;
        result.truncate(c.nodes.len());
        Ok(result)
    }
    fn run(
        &mut self,
        batches: &[Batch],
        cfg: &Config,
        steps: u32,
        read_nodes: bool,
    ) -> Result<(Vec<Vec<GpuResult>>, Vec<Node>)> {
        ensure!(steps > 0, "GPU simulation requires at least one step");
        ensure!(!batches.is_empty(), "GPU simulation requires a batch");
        let chunk = if cfg.throughput { 1024 } else { 64 };
        let param_count = batches.iter().map(|_| steps.div_ceil(chunk) as u64).sum();
        self.buffers(batches, param_count, cfg)?;
        let buffers = self.buffers.as_ref().unwrap();
        let mut dispatches = Vec::with_capacity(param_count as usize);
        for (bucket, batch) in batches
            .iter()
            .map(|batch| (bucket_index(batch.stride), batch))
        {
            let resources = buffers.buckets[bucket].as_ref().unwrap();
            self.queue
                .write_buffer(&resources.nodes, 0, bytemuck::cast_slice(&batch.nodes));
            self.queue
                .write_buffer(&resources.muscles, 0, bytemuck::cast_slice(&batch.muscles));
            self.queue
                .write_buffer(&resources.meta, 0, bytemuck::cast_slice(&batch.metadata));
            self.queue.write_buffer(
                &resources.node_adjacency,
                0,
                bytemuck::cast_slice(&batch.node_adjacency),
            );
            for tick in (0..steps).step_by(chunk as usize) {
                let offset = dispatches.len() as u64 * buffers.params_stride;
                ensure!(offset <= u32::MAX as u64, "GPU parameter offset overflow");
                let dynamic_offset = offset as u32;
                let params = Params {
                    tick,
                    steps: (steps - tick).min(chunk),
                    stride: batch.stride as u32,
                    count: batch.metadata.len() as u32,
                    gravity: cfg.gravity,
                    air: cfg.air_retention.sqrt(),
                    friction: cfg.ground_friction,
                    ground: if cfg.ground { 1.0 } else { 0.0 },
                    total_steps: steps,
                    pad: [0; 3],
                };
                self.queue
                    .write_buffer(&buffers.params, offset, bytemuck::bytes_of(&params));
                dispatches.push((
                    bucket,
                    dynamic_offset,
                    (batch.metadata.len() * batch.stride).div_ceil(64) as u32,
                ));
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Physics batches"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Advance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (bucket, offset, workgroups) in &dispatches {
                let resources = buffers.buckets[*bucket].as_ref().unwrap();
                pass.set_bind_group(0, &resources.bind, &[*offset]);
                pass.dispatch_workgroups(*workgroups, 1, 1);
            }
        }
        let mut result_offset = 0u64;
        for (bucket, batch) in batches
            .iter()
            .map(|batch| (bucket_index(batch.stride), batch))
        {
            let resources = buffers.buckets[bucket].as_ref().unwrap();
            let bytes = batch.metadata.len() as u64 * std::mem::size_of::<GpuResult>() as u64;
            encoder.copy_buffer_to_buffer(
                &resources.results,
                0,
                &buffers.readback,
                result_offset,
                bytes,
            );
            result_offset += bytes;
        }
        self.queue.submit([encoder.finish()]);
        let flat_scores = read_buffer::<GpuResult>(&self.device, &buffers.readback, result_offset)?;
        let mut scores = Vec::with_capacity(batches.len());
        let mut score_offset = 0;
        for batch in batches {
            let next = score_offset + batch.metadata.len();
            scores.push(flat_scores[score_offset..next].to_vec());
            score_offset = next;
        }
        let states = if read_nodes {
            ensure!(batches.len() == 1, "Node readback requires one batch");
            let bytes = std::mem::size_of_val(batches[0].nodes.as_slice()) as u64;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Validation state"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(
                &buffers.buckets[bucket_index(batches[0].stride)]
                    .as_ref()
                    .unwrap()
                    .nodes,
                0,
                &staging,
                0,
                bytes,
            );
            self.queue.submit([encoder.finish()]);
            read_buffer::<Node>(&self.device, &staging, bytes)?
        } else {
            vec![]
        };
        Ok((scores, states))
    }
}
fn append_adjacency_muscles(
    source: &[Muscle],
    node_count: usize,
    adjacency: &mut [NodeAdj],
    muscles: &mut Vec<Muscle>,
) {
    debug_assert!(node_count <= adjacency.len() && node_count <= 64);
    let mut counts = [0u32; 64];
    let mut cursors = [0usize; 64];
    for muscle in source {
        let a = muscle.a as usize;
        let b = muscle.b as usize;
        debug_assert!(a < node_count && b < node_count && a != b);
        counts[a] += 1;
        counts[b] += 1;
    }

    let mut next = muscles.len();
    for node in 0..node_count {
        adjacency[node] = NodeAdj {
            start: next as u32,
            count: counts[node],
        };
        cursors[node] = next;
        next += counts[node] as usize;
    }
    muscles.resize(next, <Muscle as bytemuck::Zeroable>::zeroed());

    // Fill each node's list in genome order, matching the original force sum order.
    for muscle in source {
        let a = muscle.a as usize;
        let b = muscle.b as usize;
        muscles[cursors[a]] = *muscle;
        cursors[a] += 1;
        muscles[cursors[b]] = *muscle;
        cursors[b] += 1;
    }
}
fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<T>> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..bytes)
        .map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(30)),
    })?;
    rx.recv()??;
    let view = buffer.slice(..bytes).get_mapped_range()?;
    let out = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    buffer.unmap();
    Ok(out)
}
