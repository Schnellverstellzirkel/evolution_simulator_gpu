//! Raw Vulkan compute backend for the one-creature-per-lane kernel.
//!
//! wgpu places a full barrier before every dispatch that reuses a read-write
//! buffer, which serializes the independent node-capacity buckets whenever a
//! trial is split into several dispatches. This engine records every bucket's
//! dispatch for a step range, then a single barrier, so short step ranges keep
//! the GPU full. It runs the same WGSL kernel, compiled to SPIR-V with naga.
use crate::{
    config::Config,
    creature_kernel::{self, CAPACITIES, GpuResult, LaneBatch, Params},
};
use anyhow::{Context, Result, bail, ensure};
use ash::vk;
use std::ffi::CStr;

struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    ptr: *mut u8,
}

struct GroupRes {
    nodes: Buf,
    muscles: Buf,
    bones: Buf,
    results: Buf,
    info: Buf,
    tiles: Buf,
    set: vk::DescriptorSet,
}

/// Per-submission resources. Two slots let one batch upload while another runs.
struct Slot {
    groups: Vec<Option<GroupRes>>,
    params: Option<Buf>,
    readback: Option<Buf>,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    pending: Option<Pending>,
}

/// Placement of a submitted batch's creatures, kept until its results are read.
struct Pending {
    ticket: u64,
    layout: Vec<(Vec<usize>, Vec<usize>)>,
    result_count: usize,
}

/// Results of one completed submission: per batch, the slice positions, the
/// population indices, and the raw GPU results.
pub struct Completed {
    pub ticket: u64,
    pub batches: Vec<(Vec<usize>, Vec<usize>, Vec<GpuResult>)>,
    pub gpu_seconds: f64,
}

pub struct VkEngine {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pub name: String,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    slots: Vec<Slot>,
    next_ticket: u64,
    params_stride: u64,
    workgroup: u32,
    timestamp_period: f32,
    /// GPU execution time of the last `run`, from timestamps.
    pub last_gpu_seconds: f64,
    /// Largest node capacity this device has a kernel for.
    pub max_capacity: usize,
    pub allocated_bytes: u64,
}

// Mapped pointers are only touched by the thread that owns the engine.
unsafe impl Send for VkEngine {}

pub fn spirv(source: &str) -> Result<Vec<u32>> {
    let module = naga::front::wgsl::parse_str(source).context("WGSL parse")?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .context("WGSL validation")?;
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::empty(),
        ..Default::default()
    };
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "advance".into(),
    };
    naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline)).context("SPIR-V output")
}

impl VkEngine {
    /// Opens the first Vulkan device whose name contains `name` (case-insensitive).
    /// Kernels are built for bodies up to `max_capacity` nodes.
    pub fn new(name: &str, max_capacity: usize) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load().context("Vulkan loader")?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"Evolution compute")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
                .context("Vulkan instance")?;
            let wanted = name.to_lowercase();
            let physical = instance
                .enumerate_physical_devices()?
                .into_iter()
                .find(|&d| {
                    let props = instance.get_physical_device_properties(d);
                    CStr::from_ptr(props.device_name.as_ptr())
                        .to_string_lossy()
                        .to_lowercase()
                        .contains(&wanted)
                })
                .with_context(|| format!("No Vulkan device matching {name:?}"))?;
            let props = instance.get_physical_device_properties(physical);
            let device_name = CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let families = instance.get_physical_device_queue_family_properties(physical);
            // Prefer a compute-only family: it runs beside the desktop's graphics work.
            let family = families
                .iter()
                .position(|f| {
                    f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
                .or_else(|| {
                    families
                        .iter()
                        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                })
                .context("No compute queue")? as u32;
            let priorities = [1.0];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let device = instance
                .create_device(
                    physical,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
                    None,
                )
                .context("Vulkan device")?;
            let queue = device.get_device_queue(family, 0);
            let memory_properties = instance.get_physical_device_memory_properties(physical);

            let bindings: Vec<_> = (0..7u32)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(if b == 3 {
                            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        } else {
                            vk::DescriptorType::STORAGE_BUFFER
                        })
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let set_layouts = [set_layout];
            let pipeline_layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )?;
            let workgroup = std::env::var("EVOLUTION_LANE_WG")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|v| *v == 32 || *v == 64)
                .unwrap_or(32);
            let mut pipelines = Vec::with_capacity(CAPACITIES.len());
            for &capacity in CAPACITIES.iter().filter(|&&c| c <= max_capacity) {
                let code = spirv(&creature_kernel::shader_source(capacity, workgroup))?;
                let module = device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&code),
                    None,
                )?;
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"advance");
                let info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout);
                let pipeline = device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                    .map_err(|(_, e)| e)?[0];
                device.destroy_shader_module(module, None);
                pipelines.push(pipeline);
            }
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 6 * CAPACITIES.len() as u32 * 8,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: CAPACITIES.len() as u32 * 8,
                },
            ];
            let descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(CAPACITIES.len() as u32 * 8)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let slot_count = 2;
            let command_buffers = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(slot_count),
            )?;
            let mut slots = Vec::new();
            for command_buffer in command_buffers {
                slots.push(Slot {
                    groups: (0..CAPACITIES.len()).map(|_| None).collect(),
                    params: None,
                    readback: None,
                    command_buffer,
                    fence: device.create_fence(&vk::FenceCreateInfo::default(), None)?,
                    query_pool: device.create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )?,
                    pending: None,
                });
            }
            let alignment = props.limits.min_uniform_buffer_offset_alignment;
            Ok(Self {
                _entry: entry,
                instance,
                device,
                queue,
                name: device_name,
                memory_properties,
                set_layout,
                pipeline_layout,
                pipelines,
                descriptor_pool,
                command_pool,
                slots,
                next_ticket: 0,
                params_stride: (std::mem::size_of::<Params>() as u64).next_multiple_of(alignment),
                workgroup,
                timestamp_period: props.limits.timestamp_period,
                last_gpu_seconds: 0.0,
                max_capacity,
                allocated_bytes: 0,
            })
        }
    }

    fn memory_type(&self, bits: u32, wanted: &[vk::MemoryPropertyFlags]) -> Result<u32> {
        for flags in wanted {
            for i in 0..self.memory_properties.memory_type_count {
                if bits & (1 << i) != 0
                    && self.memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(*flags)
                {
                    return Ok(i);
                }
            }
        }
        bail!("No suitable Vulkan memory type")
    }

    /// Creates a persistently mapped buffer. Device-local mappable memory
    /// (ReBAR on discrete GPUs, all memory on integrated GPUs) is preferred.
    fn create_buffer(&self, size: u64, usage: vk::BufferUsageFlags, readback: bool) -> Result<Buf> {
        let size = size.max(256).next_power_of_two();
        unsafe {
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage),
                None,
            )?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let wanted: &[vk::MemoryPropertyFlags] = if readback {
                &[
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_CACHED,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                ]
            } else {
                &[
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                ]
            };
            let memory_type = self.memory_type(requirements.memory_type_bits, wanted)?;
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr =
                self.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    as *mut u8;
            Ok(Buf {
                buffer,
                memory,
                size,
                ptr,
            })
        }
    }

    fn destroy_buffer(&self, buf: Buf) {
        unsafe {
            self.device.unmap_memory(buf.memory);
            self.device.destroy_buffer(buf.buffer, None);
            self.device.free_memory(buf.memory, None);
        }
    }

    fn write<T: bytemuck::Pod>(buf: &Buf, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        assert!(bytes.len() as u64 <= buf.size);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.ptr, bytes.len()) };
    }

    fn ensure_buffers(
        &mut self,
        slot: usize,
        batches: &[LaneBatch],
        dispatches: u64,
    ) -> Result<()> {
        let params_bytes = dispatches * self.params_stride;
        if self.slots[slot]
            .params
            .as_ref()
            .is_none_or(|b| b.size < params_bytes)
        {
            if let Some(old) = self.slots[slot].params.take() {
                self.destroy_buffer(old);
            }
            let params =
                self.create_buffer(params_bytes, vk::BufferUsageFlags::UNIFORM_BUFFER, false)?;
            self.slots[slot].params = Some(params);
            // Descriptor sets reference the parameter buffer; rebuild them.
            for group in 0..CAPACITIES.len() {
                self.drop_group(slot, group);
            }
        }
        let result_bytes = batches.iter().map(|b| b.info.len()).sum::<usize>() as u64
            * std::mem::size_of::<GpuResult>() as u64;
        if self.slots[slot]
            .readback
            .as_ref()
            .is_none_or(|b| b.size < result_bytes)
        {
            if let Some(old) = self.slots[slot].readback.take() {
                self.destroy_buffer(old);
            }
            let readback =
                self.create_buffer(result_bytes, vk::BufferUsageFlags::TRANSFER_DST, true)?;
            self.slots[slot].readback = Some(readback);
        }
        for batch in batches {
            let group = creature_kernel::capacity_index(batch.capacity);
            let need = [
                std::mem::size_of_val(batch.nodes.as_slice()) as u64,
                std::mem::size_of_val(batch.muscles.as_slice()) as u64,
                std::mem::size_of_val(batch.bones.as_slice()) as u64,
                (batch.info.len() * std::mem::size_of::<GpuResult>()) as u64,
                std::mem::size_of_val(batch.info.as_slice()) as u64,
                std::mem::size_of_val(batch.tiles.as_slice()) as u64,
            ];
            if let Some(res) = &self.slots[slot].groups[group] {
                let have = [
                    res.nodes.size,
                    res.muscles.size,
                    res.bones.size,
                    res.results.size,
                    res.info.size,
                    res.tiles.size,
                ];
                if have.iter().zip(need).all(|(h, n)| *h >= n) {
                    continue;
                }
            }
            self.drop_group(slot, group);
            let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
            let res = GroupRes {
                nodes: self.create_buffer(need[0], storage, false)?,
                muscles: self.create_buffer(need[1], storage, false)?,
                bones: self.create_buffer(need[2], storage, false)?,
                results: self.create_buffer(
                    need[3],
                    storage | vk::BufferUsageFlags::TRANSFER_SRC,
                    false,
                )?,
                info: self.create_buffer(need[4], storage, false)?,
                tiles: self.create_buffer(need[5], storage, false)?,
                set: vk::DescriptorSet::null(),
            };
            let set = unsafe {
                self.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&[self.set_layout]),
                )?[0]
            };
            let params = self.slots[slot].params.as_ref().unwrap();
            let infos = [
                (res.nodes.buffer, vk::WHOLE_SIZE),
                (res.muscles.buffer, vk::WHOLE_SIZE),
                (res.bones.buffer, vk::WHOLE_SIZE),
                (params.buffer, std::mem::size_of::<Params>() as u64),
                (res.results.buffer, vk::WHOLE_SIZE),
                (res.info.buffer, vk::WHOLE_SIZE),
                (res.tiles.buffer, vk::WHOLE_SIZE),
            ]
            .map(|(buffer, range)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(range)
            });
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(if binding == 3 {
                            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        } else {
                            vk::DescriptorType::STORAGE_BUFFER
                        })
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
            self.slots[slot].groups[group] = Some(GroupRes { set, ..res });
        }
        self.allocated_bytes = self
            .slots
            .iter()
            .map(|slot| {
                slot.groups
                    .iter()
                    .flatten()
                    .map(|g| {
                        g.nodes.size
                            + g.muscles.size
                            + g.bones.size
                            + g.results.size
                            + g.info.size
                            + g.tiles.size
                    })
                    .sum::<u64>()
                    + slot.params.as_ref().map_or(0, |b| b.size)
                    + slot.readback.as_ref().map_or(0, |b| b.size)
            })
            .sum();
        Ok(())
    }

    fn drop_group(&mut self, slot: usize, group: usize) {
        if let Some(res) = self.slots[slot].groups[group].take() {
            unsafe {
                let _ = self
                    .device
                    .free_descriptor_sets(self.descriptor_pool, &[res.set]);
            }
            for buf in [
                res.nodes,
                res.muscles,
                res.bones,
                res.results,
                res.info,
                res.tiles,
            ] {
                self.destroy_buffer(buf);
            }
        }
    }

    /// Number of submissions that can be queued without waiting.
    pub fn free_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.pending.is_none()).count()
    }

    pub fn in_flight(&self) -> usize {
        self.slots.len() - self.free_slots()
    }

    /// Uploads the batches and queues all `steps` in `chunk`-step ranges
    /// without waiting. Returns a ticket; results arrive through `poll`.
    pub fn submit(
        &mut self,
        batches: &[LaneBatch],
        cfg: &Config,
        steps: u32,
        chunk: u32,
    ) -> Result<u64> {
        ensure!(!batches.is_empty() && steps > 0, "Empty GPU batch");
        ensure!(
            batches.iter().all(|b| b.capacity <= self.max_capacity),
            "Body too large for this device's kernels"
        );
        let slot = self
            .slots
            .iter()
            .position(|s| s.pending.is_none())
            .context("No free GPU submission slot")?;
        let ranges = steps.div_ceil(chunk);
        self.ensure_buffers(slot, batches, u64::from(ranges) * batches.len() as u64)?;
        let mut param_data =
            vec![0u8; (u64::from(ranges) * batches.len() as u64 * self.params_stride) as usize];
        for (r, tick) in (0..steps).step_by(chunk as usize).enumerate() {
            for (b, batch) in batches.iter().enumerate() {
                let offset = ((r * batches.len() + b) as u64 * self.params_stride) as usize;
                let p = Params {
                    tick,
                    steps: (steps - tick).min(chunk),
                    stride: batch.capacity as u32,
                    count: batch.info.len() as u32,
                    gravity: cfg.gravity,
                    air: cfg.air_retention.sqrt(),
                    friction: cfg.ground_friction,
                    ground: if cfg.ground { 1.0 } else { 0.0 },
                    total_steps: steps,
                    pad: [0; 3],
                };
                param_data[offset..offset + std::mem::size_of::<Params>()]
                    .copy_from_slice(bytemuck::bytes_of(&p));
            }
        }
        let resources = &self.slots[slot];
        Self::write(resources.params.as_ref().unwrap(), &param_data);
        for batch in batches {
            let res = resources.groups[creature_kernel::capacity_index(batch.capacity)]
                .as_ref()
                .unwrap();
            Self::write(&res.nodes, &batch.nodes);
            Self::write(&res.muscles, &batch.muscles);
            Self::write(&res.bones, &batch.bones);
            Self::write(&res.info, &batch.info);
            Self::write(&res.tiles, &batch.tiles);
        }
        let device = &self.device;
        let cb = resources.command_buffer;
        let result_count: usize = batches.iter().map(|b| b.info.len()).sum();
        unsafe {
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_reset_query_pool(cb, resources.query_pool, 0, 2);
            // Host writes through mapped memory are visible at submission.
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                resources.query_pool,
                0,
            );
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            // Large buckets first so the small ones fill the tail.
            let mut order: Vec<usize> = (0..batches.len()).collect();
            order.sort_by_key(|&b| std::cmp::Reverse(batches[b].info.len() * batches[b].capacity));
            for r in 0..ranges as usize {
                if r > 0 {
                    device.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
                for &b in &order {
                    let batch = &batches[b];
                    let group = creature_kernel::capacity_index(batch.capacity);
                    let res = resources.groups[group].as_ref().unwrap();
                    device.cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipelines[group],
                    );
                    let offset = ((r * batches.len() + b) as u64 * self.params_stride) as u32;
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_layout,
                        0,
                        &[res.set],
                        &[offset],
                    );
                    let groups = batch.info.len().div_ceil(self.workgroup as usize) as u32;
                    device.cmd_dispatch(cb, groups, 1, 1);
                }
            }
            let to_transfer = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[to_transfer],
                &[],
                &[],
            );
            device.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                resources.query_pool,
                1,
            );
            let readback = resources.readback.as_ref().unwrap();
            let mut offset = 0u64;
            for batch in batches {
                let res = resources.groups[creature_kernel::capacity_index(batch.capacity)]
                    .as_ref()
                    .unwrap();
                let bytes = (batch.info.len() * std::mem::size_of::<GpuResult>()) as u64;
                device.cmd_copy_buffer(
                    cb,
                    res.results.buffer,
                    readback.buffer,
                    &[vk::BufferCopy::default().dst_offset(offset).size(bytes)],
                );
                offset += bytes;
            }
            let to_host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[to_host],
                &[],
                &[],
            );
            device.end_command_buffer(cb)?;
            device.reset_fences(&[resources.fence])?;
            device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[cb])],
                resources.fence,
            )?;
        }
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.slots[slot].pending = Some(Pending {
            ticket,
            layout: batches
                .iter()
                .map(|b| (b.slots.clone(), b.creatures.clone()))
                .collect(),
            result_count,
        });
        Ok(ticket)
    }

    /// Returns the oldest submission's results once it has finished, waiting up
    /// to `timeout`. Submissions complete in order on the single queue.
    pub fn poll(&mut self, timeout: std::time::Duration) -> Result<Option<Completed>> {
        let Some(slot) = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.pending.as_ref().map(|p| (p.ticket, i)))
            .min()
            .map(|(_, i)| i)
        else {
            return Ok(None);
        };
        let resources = &self.slots[slot];
        unsafe {
            match self.device.wait_for_fences(
                &[resources.fence],
                true,
                timeout.as_nanos().min(u128::from(u64::MAX)) as u64,
            ) {
                Ok(()) => {}
                Err(vk::Result::TIMEOUT) => return Ok(None),
                Err(e) => return Err(e.into()),
            }
            let mut ticks = [0u64; 2];
            let gpu_seconds = if self
                .device
                .get_query_pool_results(
                    resources.query_pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
                .is_ok()
            {
                ticks[1].saturating_sub(ticks[0]) as f64 * f64::from(self.timestamp_period) * 1e-9
            } else {
                0.0
            };
            let pending = self.slots[slot].pending.take().unwrap();
            let readback = self.slots[slot].readback.as_ref().unwrap();
            let flat: &[GpuResult] =
                std::slice::from_raw_parts(readback.ptr as *const GpuResult, pending.result_count);
            let mut batches = Vec::with_capacity(pending.layout.len());
            let mut start = 0;
            for (slots, creatures) in pending.layout {
                let end = start + slots.len();
                batches.push((slots, creatures, flat[start..end].to_vec()));
                start = end;
            }
            self.last_gpu_seconds = gpu_seconds;
            Ok(Some(Completed {
                ticket: pending.ticket,
                batches,
                gpu_seconds,
            }))
        }
    }

    /// Waits up to `timeout` for the oldest submission without collecting it.
    pub fn wait(&self, timeout: std::time::Duration) -> Result<()> {
        let Some(slot) = self
            .slots
            .iter()
            .filter_map(|s| s.pending.as_ref().map(|p| (p.ticket, s.fence)))
            .min_by_key(|&(ticket, _)| ticket)
        else {
            return Ok(());
        };
        unsafe {
            match self
                .device
                .wait_for_fences(&[slot.1], true, timeout.as_nanos() as u64)
            {
                Ok(()) | Err(vk::Result::TIMEOUT) => Ok(()),
                Err(e) => Err(e.into()),
            }
        }
    }

    /// Blocking evaluation: submit and wait for this batch alone.
    pub fn run(
        &mut self,
        batches: &[LaneBatch],
        cfg: &Config,
        steps: u32,
        chunk: u32,
    ) -> Result<Vec<Vec<GpuResult>>> {
        while self.in_flight() > 0 {
            self.poll(std::time::Duration::from_secs(60))?;
        }
        let ticket = self.submit(batches, cfg, steps, chunk)?;
        loop {
            if let Some(done) = self.poll(std::time::Duration::from_secs(60))?
                && done.ticket == ticket
            {
                return Ok(done.batches.into_iter().map(|(_, _, r)| r).collect());
            }
        }
    }
}

impl Drop for VkEngine {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        for slot in 0..self.slots.len() {
            for group in 0..CAPACITIES.len() {
                self.drop_group(slot, group);
            }
            if let Some(b) = self.slots[slot].params.take() {
                self.destroy_buffer(b);
            }
            if let Some(b) = self.slots[slot].readback.take() {
                self.destroy_buffer(b);
            }
        }
        unsafe {
            for slot in &self.slots {
                self.device.destroy_query_pool(slot.query_pool, None);
                self.device.destroy_fence(slot.fence, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for &p in &self.pipelines {
                self.device.destroy_pipeline(p, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
