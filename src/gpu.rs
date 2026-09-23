use crate::{
    config::Config,
    evolution::{Muscle, Population},
    physics::{self, Node},
    qd::{EvaluationMetrics, TrialMetrics},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::time::Instant;

#[derive(Default)]
pub struct GpuProfile {
    pub calls: u64,
    pub packing_seconds: f64,
    pub allocation_seconds: f64,
    pub encoding_seconds: f64,
    pub readback_seconds: f64,
    pub shader_seconds: f64,
    pub bucket_seconds: [f64; 6],
}

struct TimestampResources {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readback: wgpu::Buffer,
}

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
struct GpuMuscle {
    a: u32,
    b: u32,
    short: f32,
    long: f32,
    inv_period: f32,
    phase: f32,
    duty: f32,
    stiffness: f32,
    inv_duty: f32,
    inv_complement: f32,
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
    /// Lane kernels store the first-dimension workgroup count here.
    groups_x: u32,
    pad: [u32; 2],
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
    pipeline32: wgpu::ComputePipeline,
    pipeline30: wgpu::ComputePipeline,
    serial_pipeline: Option<wgpu::ComputePipeline>,
    /// Barrier-free one-creature-per-lane kernels for small strides.
    lane_pipelines: [Option<wgpu::ComputePipeline>; 6],
    buffers: Option<Buffers>,
    pub allocated_bytes: u64,
    pub profile: Option<GpuProfile>,
    timestamps: Option<TimestampResources>,
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
    buckets: [Option<BucketBuffers>; 6],
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
    muscles: Vec<GpuMuscle>,
    metadata: Vec<Meta>,
    node_adjacency: Vec<NodeAdj>,
    stride: usize,
}
fn bucket_index(stride: usize) -> usize {
    match stride {
        4 => 0,
        5 => 1,
        8 => 2,
        16 => 3,
        32 => 4,
        64 => 5,
        _ => unreachable!("validated GPU bucket stride"),
    }
}
fn stride_for_nodes(count: usize, split_five: bool) -> usize {
    if count == 5 && split_five {
        5
    } else {
        count.next_power_of_two().max(4)
    }
}
fn split_five_bucket(population: usize) -> bool {
    (population <= 10_000 && std::env::var_os("EVOLUTION_LEGACY_BUCKET5").is_none())
        || std::env::var_os("EVOLUTION_FORCE_BUCKET5").is_some()
}
fn serial_kernels_enabled() -> bool {
    // Paired full-GUI 100k runs showed this private-array path 1.69x slower.
    // Keep it available for explicit tests, but use the workgroup path by default.
    matches!(
        std::env::var("EVOLUTION_KERNEL").ok().as_deref(),
        Some("serial")
    )
}
fn lane_kernels_enabled() -> bool {
    // Barrier-free lane kernels register-spill on long dispatch chunks.
    // Keep them opt-in until the shader stops thrashing.
    std::env::var_os("EVOLUTION_LANE_SHADER")
        .map(|v| v != "0")
        .unwrap_or(false)
}
fn exact_cos_enabled() -> bool {
    std::env::var_os("EVOLUTION_EXACT_COS").is_some()
}
const SERIAL_WORKGROUP: usize = 128;
fn apply_fast_cos(source: String) -> String {
    if exact_cos_enabled() {
        return source;
    }
    let fast_cos_function = "fn fast_cos_pi(x:f32)->f32 { let y=(x-0.5)*3.14159265359; let z=y*y; var p=fma(z,-2.50521084e-8,2.75573192e-6); p=fma(z,p,-1.98412698e-4); p=fma(z,p,8.33333377e-3); p=fma(z,p,-1.66666672e-1); p=fma(z,p,1.0); return -y*p; }\n";
    source
        .replace(
            "cos(3.14159265359*phase*m.inv_duty)",
            "fast_cos_pi(phase*m.inv_duty)",
        )
        .replace(
            "cos(3.14159265359*(phase-m.duty)*m.inv_complement)",
            "fast_cos_pi((phase-m.duty)*m.inv_complement)",
        )
        .replace(
            "fn muscle_length",
            &format!("{fast_cos_function}fn muscle_length"),
        )
}
fn pipeline_chunk_size(cfg: &Config) -> usize {
    std::env::var("EVOLUTION_PIPELINE_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(if cfg.population >= 100_000 {
            16_384
        } else {
            4_096
        })
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
    let timestamp_feature = if std::env::var_os("EVOLUTION_GPU_PROFILE").is_some() {
        adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    } else {
        wgpu::Features::empty()
    };
    wgpu::DeviceDescriptor {
        label: Some("Evolution GPU"),
        required_features: timestamp_feature,
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
        let wide_source = include_str!("../shaders/physics.wgsl");
        ensure!(
            wide_source.matches("WORKGROUPX2").count() == 2
                && wide_source.matches("@workgroup_size(WORKGROUP)").count() == 1
                && wide_source.matches("half=WORKGROUPu").count() == 1,
            "The workgroup physics variant needs its source transform updated"
        );
        let variant_source = |lanes: u32| {
            apply_fast_cos(
                wide_source
                    .replace("WORKGROUPX2", &format!("{}", lanes * 2))
                    .replace("WORKGROUPu", &format!("{lanes}u"))
                    .replace("WORKGROUP", &lanes.to_string()),
            )
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Muscle physics"),
            source: wgpu::ShaderSource::Wgsl(variant_source(64).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batched creature physics"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("advance"),
            compilation_options: Default::default(),
            cache: None,
        });
        let serial_pipeline = if serial_kernels_enabled() {
            let serial_source = apply_fast_cos(
                include_str!("../shaders/physics_serial.wgsl").replace("MAXNODES", "8"),
            );
            ensure!(
                serial_source.matches("@workgroup_size(128)").count() == 1
                    && serial_source.matches("group.x*128u").count() == 1
                    && !serial_source.contains("MAXNODES"),
                "The serial physics source transform needs updating"
            );
            let serial_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Serial creature physics"),
                source: wgpu::ShaderSource::Wgsl(serial_source.into()),
            });
            Some(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Serial whole-body creature physics"),
                    layout: Some(&pipeline_layout),
                    module: &serial_shader,
                    entry_point: Some("advance"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            )
        } else {
            None
        };
        let mut lane_pipelines: [Option<wgpu::ComputePipeline>; 6] = [const { None }; 6];
        if lane_kernels_enabled() {
            let lane_template = include_str!("../shaders/physics_lane.wgsl");
            ensure!(
                lane_template.matches("MAXN").count() == 11,
                "The lane physics source transform needs updating"
            );
            for (bucket, maxn) in [(0usize, 4u32), (1, 5), (2, 8), (3, 16)] {
                let lane_source = apply_fast_cos(lane_template.replace("MAXN", &maxn.to_string()));
                ensure!(
                    !lane_source.contains("MAXN"),
                    "lane MAXN substitution failed"
                );
                let lane_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Lane creature physics"),
                    source: wgpu::ShaderSource::Wgsl(lane_source.into()),
                });
                lane_pipelines[bucket] = Some(device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("Lane creature physics"),
                        layout: Some(&pipeline_layout),
                        module: &lane_shader,
                        entry_point: Some("advance"),
                        compilation_options: Default::default(),
                        cache: None,
                    },
                ));
            }
        }
        let shader32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("32-lane muscle physics"),
            source: wgpu::ShaderSource::Wgsl(variant_source(32).into()),
        });
        let pipeline32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("32-lane creature physics"),
            layout: Some(&pipeline_layout),
            module: &shader32,
            entry_point: Some("advance"),
            compilation_options: Default::default(),
            cache: None,
        });
        let shader30 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("30-lane five-node physics"),
            source: wgpu::ShaderSource::Wgsl(variant_source(30).into()),
        });
        let pipeline30 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("30-lane five-node creature physics"),
            layout: Some(&pipeline_layout),
            module: &shader30,
            entry_point: Some("advance"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            anyhow::bail!("GPU shader initialization failed: {error}");
        }
        let profiling = std::env::var_os("EVOLUTION_GPU_PROFILE").is_some();
        let timestamps = if profiling && device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            Some(TimestampResources {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("Physics timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 12,
                }),
                resolved: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Physics timestamp resolve"),
                    size: 96,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Physics timestamp readback"),
                    size: 96,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            })
        } else {
            None
        };
        Ok(Self {
            device,
            queue,
            name,
            pipeline,
            pipeline32,
            pipeline30,
            serial_pipeline,
            lane_pipelines,
            buffers: None,
            allocated_bytes: 0,
            profile: profiling.then(GpuProfile::default),
            timestamps,
        })
    }
    fn buffers(&mut self, batches: &[Batch], param_count: u64, cfg: &Config) -> Result<()> {
        let mut specs: [Option<([u64; 4], u64)>; 6] = [None; 6];
        let mut result_count = 0u64;
        for batch in batches {
            ensure!(
                matches!(batch.stride, 4 | 5 | 8 | 16 | 32 | 64),
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
        let mut bucket_buffers: [Option<BucketBuffers>; 6] = [const { None }; 6];
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
        if indices.is_empty() {
            return Ok(out);
        }
        let chunk_size = pipeline_chunk_size(cfg);
        let steps = physics::SETTLE + cfg.steps();
        if indices.len() <= chunk_size {
            let packing_started = Instant::now();
            let batches = pack_batches(pop, indices, cfg)?;
            if let Some(profile) = &mut self.profile {
                profile.packing_seconds += packing_started.elapsed().as_secs_f64();
                profile.calls += 1;
            }
            if batches.is_empty() {
                return Ok(out);
            }
            let results = self.run(&batches, cfg, steps, false)?.0;
            merge_metrics(pop, &batches, &results, cfg, 0, &mut out);
            return Ok(out);
        }
        let chunks: Vec<&[usize]> = indices.chunks(chunk_size).collect();
        std::thread::scope(|scope| {
            let mut pending = Some(scope.spawn(|| {
                let started = Instant::now();
                let batches = pack_batches(pop, chunks[0], cfg);
                (started.elapsed().as_secs_f64(), batches)
            }));
            for chunk_index in 0..chunks.len() {
                let (pack_seconds, packed) = pending
                    .take()
                    .expect("pipeline primed")
                    .join()
                    .expect("packing task");
                if let Some(profile) = &mut self.profile {
                    profile.packing_seconds += pack_seconds;
                    profile.calls += 1;
                }
                pending = chunks.get(chunk_index + 1).map(|next| {
                    scope.spawn(|| {
                        let started = Instant::now();
                        let batches = pack_batches(pop, next, cfg);
                        (started.elapsed().as_secs_f64(), batches)
                    })
                });
                let packed = packed?;
                if packed.is_empty() {
                    continue;
                }
                let results = self.run(&packed, cfg, steps, false)?.0;
                merge_metrics(
                    pop,
                    &packed,
                    &results,
                    cfg,
                    chunk_index * chunk_size,
                    &mut out,
                );
            }
            Ok::<(), anyhow::Error>(())
        })?;
        Ok(out)
    }
    pub fn trajectory(
        &mut self,
        c: &crate::evolution::Creature,
        cfg: &Config,
        steps: u32,
    ) -> Result<Vec<Node>> {
        let stride = stride_for_nodes(c.nodes.len(), split_five_bucket(cfg.population));
        let mut nodes = vec![Node::default(); stride];
        nodes[..c.nodes.len()].copy_from_slice(&physics::nodes(c));
        let metadata = vec![Meta {
            nodes: c.nodes.len() as u32,
        }];
        let mut node_adjacency = vec![NodeAdj::default(); stride];
        let mut muscles = Vec::<GpuMuscle>::with_capacity(c.muscles.len() * 2);
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
        let allocation_started = Instant::now();
        let chunk = std::env::var("EVOLUTION_GPU_CHUNK")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(4096);
        let use_32_lanes = (cfg.population <= 10_000
            && std::env::var_os("EVOLUTION_WORKGROUP64").is_none())
            || std::env::var_os("EVOLUTION_WORKGROUP32").is_some();
        let param_count = batches.iter().map(|_| steps.div_ceil(chunk) as u64).sum();
        self.buffers(batches, param_count, cfg)?;
        let allocation_seconds = allocation_started.elapsed().as_secs_f64();
        let encoding_started = Instant::now();
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
            let serial_bucket = self.serial_pipeline.is_some() && bucket <= 2;
            let lane_bucket = !serial_bucket && self.lane_pipelines[bucket].is_some();
            for tick in (0..steps).step_by(chunk as usize) {
                let offset = dispatches.len() as u64 * buffers.params_stride;
                ensure!(offset <= u32::MAX as u64, "GPU parameter offset overflow");
                let dynamic_offset = offset as u32;
                let (workgroups_x, workgroups_y) = if lane_bucket {
                    let total = batch.metadata.len().div_ceil(64) as u32;
                    let groups_x = total.clamp(1, u16::MAX as u32);
                    (groups_x, total.div_ceil(groups_x).max(1))
                } else if serial_bucket {
                    (batch.metadata.len().div_ceil(SERIAL_WORKGROUP) as u32, 1)
                } else {
                    let lanes = if bucket == 1 {
                        30
                    } else if bucket < 5 && use_32_lanes {
                        32
                    } else {
                        64
                    };
                    (
                        (batch.metadata.len() * batch.stride).div_ceil(lanes) as u32,
                        1,
                    )
                };
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
                    groups_x: if lane_bucket { workgroups_x } else { 0 },
                    pad: [0; 2],
                };
                self.queue
                    .write_buffer(&buffers.params, offset, bytemuck::bytes_of(&params));
                dispatches.push((
                    bucket,
                    dynamic_offset,
                    workgroups_x,
                    workgroups_y,
                    serial_bucket,
                    lane_bucket,
                ));
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Physics batches"),
            });
        let detailed_timestamps = self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        let mut used_buckets = [false; 6];
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Advance"),
                timestamp_writes: self.timestamps.as_ref().and_then(|timestamps| {
                    (!detailed_timestamps).then_some(wgpu::ComputePassTimestampWrites {
                        query_set: &timestamps.queries,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    })
                }),
            });
            let mut previous_bucket = None;
            for (bucket, offset, workgroups_x, workgroups_y, serial_bucket, lane_bucket) in
                &dispatches
            {
                pass.set_pipeline(if *serial_bucket {
                    self.serial_pipeline
                        .as_ref()
                        .expect("serial dispatch without serial pipeline")
                } else if *lane_bucket {
                    self.lane_pipelines[*bucket]
                        .as_ref()
                        .expect("lane dispatch without lane pipeline")
                } else if *bucket == 1 {
                    &self.pipeline30
                } else if *bucket < 5 && use_32_lanes {
                    &self.pipeline32
                } else {
                    &self.pipeline
                });
                if detailed_timestamps && previous_bucket != Some(*bucket) {
                    if let Some(timestamps) = &self.timestamps {
                        if let Some(previous) = previous_bucket {
                            pass.write_timestamp(&timestamps.queries, previous as u32 * 2 + 1);
                        }
                        pass.write_timestamp(&timestamps.queries, *bucket as u32 * 2);
                    }
                    previous_bucket = Some(*bucket);
                }
                used_buckets[*bucket] = true;
                let resources = buffers.buckets[*bucket].as_ref().unwrap();
                pass.set_bind_group(0, &resources.bind, &[*offset]);
                pass.dispatch_workgroups(*workgroups_x, *workgroups_y, 1);
            }
            if detailed_timestamps
                && let (Some(timestamps), Some(previous)) = (&self.timestamps, previous_bucket)
            {
                pass.write_timestamp(&timestamps.queries, previous as u32 * 2 + 1);
            }
        }
        if let Some(timestamps) = &self.timestamps {
            if detailed_timestamps {
                encoder.resolve_query_set(&timestamps.queries, 0..12, &timestamps.resolved, 0);
            } else {
                encoder.resolve_query_set(&timestamps.queries, 0..2, &timestamps.resolved, 0);
            }
            encoder.copy_buffer_to_buffer(&timestamps.resolved, 0, &timestamps.readback, 0, 96);
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
        let encoding_seconds = encoding_started.elapsed().as_secs_f64();
        let readback_started = Instant::now();
        let flat_scores = read_buffer::<GpuResult>(&self.device, &buffers.readback, result_offset)?;
        let mut bucket_seconds = [0.0; 6];
        let shader_seconds = if let Some(timestamps) = &self.timestamps {
            let ticks = read_buffer::<u64>(&self.device, &timestamps.readback, 96)?;
            let seconds_per_tick = f64::from(self.queue.get_timestamp_period()) * 1e-9;
            if detailed_timestamps {
                for (bucket, used) in used_buckets.iter().enumerate() {
                    if *used {
                        bucket_seconds[bucket] =
                            ticks[bucket * 2 + 1].saturating_sub(ticks[bucket * 2]) as f64
                                * seconds_per_tick;
                    }
                }
                bucket_seconds.iter().sum()
            } else {
                ticks[1].saturating_sub(ticks[0]) as f64 * seconds_per_tick
            }
        } else {
            0.0
        };
        let readback_seconds = readback_started.elapsed().as_secs_f64();
        if let Some(profile) = &mut self.profile {
            profile.allocation_seconds += allocation_seconds;
            profile.encoding_seconds += encoding_seconds;
            profile.readback_seconds += readback_seconds;
            profile.shader_seconds += shader_seconds;
            for (total, current) in profile.bucket_seconds.iter_mut().zip(bucket_seconds) {
                *total += current;
            }
        }
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
fn pack_batches(pop: &Population, indices: &[usize], cfg: &Config) -> Result<Vec<Batch>> {
    let split_five = split_five_bucket(cfg.population);
    // One pass groups indices by bucket stride instead of six full filter scans.
    let mut buckets: [(usize, Vec<(usize, usize)>); 6] = [
        (4, Vec::new()),
        (5, Vec::new()),
        (8, Vec::new()),
        (16, Vec::new()),
        (32, Vec::new()),
        (64, Vec::new()),
    ];
    for (slot, &i) in indices.iter().enumerate() {
        let stride = stride_for_nodes(pop.genomes[i].node_count, split_five);
        let bucket = match stride {
            4 => 0,
            5 => 1,
            8 => 2,
            16 => 3,
            32 => 4,
            _ => 5,
        };
        buckets[bucket].1.push((slot, i));
    }
    let batches: Vec<Batch> = buckets
        .into_par_iter()
        .filter_map(|(stride, mut group)| {
            if group.is_empty() {
                return None;
            }
            // Similar muscle counts per warp keep force-loop divergence low.
            group.sort_unstable_by_key(|&(_, i)| {
                (pop.genomes[i].muscle_count, pop.genomes[i].node_count, i)
            });
            let mut nodes = vec![Node::default(); group.len() * stride];
            let muscle_count: usize = group
                .iter()
                .map(|&(_, i)| pop.genomes[i].muscle_count)
                .sum();
            let mut muscles = Vec::<GpuMuscle>::with_capacity(muscle_count * 2);
            let mut node_adjacency = vec![NodeAdj::default(); group.len() * stride];
            let mut metadata = Vec::with_capacity(group.len());
            for (j, &(_, i)) in group.iter().enumerate() {
                let genome = &pop.genomes[i];
                let genes = &pop.nodes[genome.node_start..genome.node_start + genome.node_count];
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
    Ok(batches)
}
fn merge_metrics(
    pop: &Population,
    batches: &[Batch],
    results: &[Vec<GpuResult>],
    cfg: &Config,
    base_slot: usize,
    out: &mut [EvaluationMetrics],
) {
    for (batch, results) in batches.iter().zip(results) {
        for (j, &slot) in batch.slots.iter().enumerate() {
            let r = results[j];
            let active_steps = cfg.steps();
            let contact_denominator =
                (active_steps.max(1) * pop.genomes[batch.creatures[j]].node_count as u32) as f32;
            out[base_slot + slot] = EvaluationMetrics {
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
}
fn append_adjacency_muscles(
    source: &[Muscle],
    node_count: usize,
    adjacency: &mut [NodeAdj],
    muscles: &mut Vec<GpuMuscle>,
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
    muscles.resize(next, <GpuMuscle as bytemuck::Zeroable>::zeroed());

    // Fill each node's list in genome order, matching the original force sum order.
    for muscle in source {
        let a = muscle.a as usize;
        let b = muscle.b as usize;
        let packed = GpuMuscle {
            a: muscle.a,
            b: muscle.b,
            short: muscle.short,
            long: muscle.long,
            inv_period: 1.0 / muscle.period,
            phase: muscle.phase,
            duty: muscle.duty,
            stiffness: muscle.stiffness,
            inv_duty: 1.0 / muscle.duty,
            inv_complement: 1.0 / (1.0 - muscle.duty),
        };
        muscles[cursors[a]] = packed;
        cursors[a] += 1;
        muscles[cursors[b]] = packed;
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
