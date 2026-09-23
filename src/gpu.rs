use crate::{
    config::Config,
    evolution::{Muscle, Population},
    physics::{self, Node},
    qd::{EvaluationMetrics, TrialMetrics},
};
use anyhow::{Context, Result, ensure};
use std::collections::VecDeque;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta {
    nodes: u32,
    muscles: u32,
    start: u32,
    pad: u32,
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
struct Buffers {
    nodes: wgpu::Buffer,
    muscles: wgpu::Buffer,
    meta: wgpu::Buffer,
    params: wgpu::Buffer,
    results: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind: wgpu::BindGroup,
    capacities: [u64; 3],
}
struct Batch<'a> {
    nodes: &'a [Node],
    muscles: &'a [Muscle],
    metadata: &'a [Meta],
    stride: usize,
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
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batched creature physics"),
            layout: None,
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
    fn buffers(
        &mut self,
        node_bytes: u64,
        muscle_bytes: u64,
        count: u64,
        cfg: &Config,
    ) -> Result<()> {
        let needed = [node_bytes.max(32), muscle_bytes.max(32), count.max(1)];
        if self.allocated_bytes < cfg.gpu_budget_mib as u64 * 1024 * 1024
            && self
                .buffers
                .as_ref()
                .is_some_and(|b| b.capacities.iter().zip(needed).all(|(c, n)| *c >= n))
        {
            return Ok(());
        }
        let caps = needed.map(u64::next_power_of_two);
        let total = caps[0]
            + caps[1]
            + caps[2] * (16 + std::mem::size_of::<GpuResult>() as u64 * 2)
            + 48;
        ensure!(
            total < cfg.gpu_budget_mib as u64 * 1024 * 1024,
            "GPU batch exceeds memory budget"
        );
        ensure!(
            caps[0] <= self.device.limits().max_storage_buffer_binding_size
                && caps[1] <= self.device.limits().max_storage_buffer_binding_size,
            "GPU batch exceeds storage binding limits"
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
        let nodes = create(
            "Node state",
            caps[0],
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        let muscles = create("Muscle genomes", caps[1], storage);
        let meta = create("Creature metadata", caps[2] * 16, storage);
        let params = create(
            "Physics parameters",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let results = create(
            "Fitness and behavior descriptors",
            caps[2] * std::mem::size_of::<GpuResult>() as u64,
            storage | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = create(
            "Evaluation readback",
            caps[2] * std::mem::size_of::<GpuResult>() as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physics resources"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[&nodes, &muscles, &meta, &params, &results]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        self.buffers = Some(Buffers {
            nodes,
            muscles,
            meta,
            params,
            results,
            readback,
            bind,
            capacities: caps,
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
        for stride in [8usize, 16, 32, 64] {
            let group: Vec<_> = indices
                .iter()
                .enumerate()
                .filter_map(|(slot, &i)| {
                    (pop.genomes[i].node_count.next_power_of_two().max(8) == stride)
                        .then_some((slot, i))
                })
                .collect();
            if group.is_empty() {
                continue;
            }
            let mut nodes = vec![Node::default(); group.len() * stride];
            let muscle_count = group
                .iter()
                .map(|&(_, i)| pop.genomes[i].muscle_count)
                .sum();
            let mut muscles = Vec::<Muscle>::with_capacity(muscle_count);
            let mut meta = Vec::with_capacity(group.len());
            for (j, &(_, i)) in group.iter().enumerate() {
                let genome = &pop.genomes[i];
                let genes = &pop.nodes[genome.node_start..genome.node_start + genome.node_count];
                for (dst, gene) in nodes[j * stride..j * stride + genes.len()]
                    .iter_mut()
                    .zip(genes)
                {
                    *dst = physics::node(gene);
                }
                let muscle_end = genome.muscle_start + genome.muscle_count;
                let start = muscles.len() as u32;
                meta.push(Meta {
                    nodes: genes.len() as u32,
                    muscles: genome.muscle_count as u32,
                    start,
                    pad: 0,
                });
                muscles.extend_from_slice(&pop.muscles[genome.muscle_start..muscle_end]);
            }
            let results = self
                .run(
                    Batch {
                        nodes: &nodes,
                        muscles: &muscles,
                        metadata: &meta,
                        stride,
                    },
                    cfg,
                    physics::SETTLE + cfg.steps(),
                    false,
                )?
                .0;
            for (j, &(slot, _)) in group.iter().enumerate() {
                let r = results[j];
                let active_steps = cfg.steps();
                let contact_denominator =
                    (active_steps.max(1) * pop.genomes[group[j].1].node_count as u32) as f32;
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
        let meta = [Meta {
            nodes: c.nodes.len() as u32,
            muscles: c.muscles.len() as u32,
            start: 0,
            pad: 0,
        }];
        let (_, mut result) = self.run(
            Batch {
                nodes: &nodes,
                muscles: &c.muscles,
                metadata: &meta,
                stride,
            },
            cfg,
            steps,
            true,
        )?;
        result.truncate(c.nodes.len());
        Ok(result)
    }
    fn run(
        &mut self,
        batch: Batch<'_>,
        cfg: &Config,
        steps: u32,
        read_nodes: bool,
    ) -> Result<(Vec<GpuResult>, Vec<Node>)> {
        let Batch {
            nodes,
            muscles,
            metadata: meta,
            stride,
        } = batch;
        self.buffers(
            std::mem::size_of_val(nodes) as u64,
            std::mem::size_of_val(muscles) as u64,
            meta.len() as u64,
            cfg,
        )?;
        let b = self.buffers.as_ref().unwrap();
        self.queue
            .write_buffer(&b.nodes, 0, bytemuck::cast_slice(nodes));
        self.queue
            .write_buffer(&b.muscles, 0, bytemuck::cast_slice(muscles));
        self.queue
            .write_buffer(&b.meta, 0, bytemuck::cast_slice(meta));
        // Each bounded dispatch retains all intermediate state in GPU buffers.
        let chunk = if cfg.throughput { 256 } else { 64 };
        let max_pending = if cfg.throughput { 8 } else { 2 };
        let mut pending_submissions = VecDeque::with_capacity(max_pending);
        for tick in (0..steps).step_by(chunk) {
            let p = Params {
                tick,
                steps: (steps - tick).min(chunk as u32),
                stride: stride as u32,
                count: meta.len() as u32,
                gravity: cfg.gravity,
                air: cfg.air_retention.sqrt(),
                friction: cfg.ground_friction,
                ground: if cfg.ground { 1.0 } else { 0.0 },
                total_steps: steps,
                pad: [0; 3],
            };
            self.queue
                .write_buffer(&b.params, 0, bytemuck::bytes_of(&p));
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Physics batch"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Advance"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &b.bind, &[]);
                pass.dispatch_workgroups((meta.len() * stride).div_ceil(64) as u32, 1, 1);
            }
            let submission = self.queue.submit([encoder.finish()]);
            pending_submissions.push_back(submission);
            // Keep enough work queued to hide CPU command-encoding gaps. Throughput
            // mode is intended for long runs; responsive mode yields more often.
            if pending_submissions.len() >= max_pending {
                let previous = pending_submissions.pop_front().unwrap();
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(previous),
                    timeout: Some(std::time::Duration::from_secs(30)),
                })?;
            }
        }
        let bytes = meta.len() as u64 * std::mem::size_of::<GpuResult>() as u64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&b.results, 0, &b.readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let scores = read_buffer::<GpuResult>(&self.device, &b.readback, bytes)?;
        let states = if read_nodes {
            let bytes = std::mem::size_of_val(nodes) as u64;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Validation state"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&b.nodes, 0, &staging, 0, bytes);
            self.queue.submit([encoder.finish()]);
            read_buffer::<Node>(&self.device, &staging, bytes)?
        } else {
            vec![]
        };
        Ok((scores, states))
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
