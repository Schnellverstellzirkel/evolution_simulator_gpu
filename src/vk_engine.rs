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

/// Per-submission resources. Each slot submits to its own queue when the
/// device has enough, so units run side by side: the step-range barriers of
/// one unit do not hold back another, and a small unit (a batch of checks)
/// fills the SMs a large one leaves idle instead of running alone.
struct Slot {
    queue: vk::Queue,
    groups: Vec<Option<GroupRes>>,
    params: Option<Buf>,
    readback: Option<Buf>,
    /// Recorded replay frames (binding 7); only the replay slot has one.
    frames: Option<Buf>,
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
    /// Per batch: node count and muscle buffer length read back after the
    /// results, when the trial continues in a later segment.
    state: Option<Vec<(usize, usize)>>,
    /// Recorded frames: their byte offset in the readback buffer and their
    /// count of node positions.
    frames: Option<(u64, usize)>,
}

/// Results of one completed submission: per batch, the slice positions, the
/// population indices, and the raw GPU results.
pub struct Completed {
    pub ticket: u64,
    pub batches: Vec<(Vec<usize>, Vec<usize>, Vec<GpuResult>)>,
    /// Per batch, when asked for: node state and muscle buffer at the end of
    /// the segment, for `LaneBatch::repack`.
    pub state: Option<Vec<(Vec<crate::physics::Node>, Vec<f32>)>>,
    /// For a recording (`VkEngine::record`): node positions as
    /// `[creature][frame][node]`, with the batch's node stride.
    pub frames: Option<Vec<[f32; 2]>>,
    pub gpu_seconds: f64,
}

pub struct VkEngine {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    pub name: String,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    /// Kernels by physics fidelity and node capacity, each built on first
    /// use: a cold driver takes seconds for a small body's kernel and much
    /// longer for a large one, so a start compiles only what its bodies need.
    pipelines: std::collections::HashMap<(crate::physics::Fidelity, usize), vk::Pipeline>,
    /// Recording kernels (`physics2::record_source`) by fidelity and
    /// node capacity, built on first use.
    recording: std::collections::HashMap<(crate::physics::Fidelity, usize), vk::Pipeline>,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    /// Submission slots. The last one is kept for replays, on its own queue
    /// when the device has one, so a replay never waits behind evaluation.
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

/// Descriptor sets the pool holds: one per batch of each submission slot
/// (body-size buckets plus body-plan batches).
const MAX_SETS: u32 = 1024;
/// Descriptor bindings: six storage buffers and the parameters (3) for
/// scoring, and the recorded frames (7) for replays.
const BINDINGS: u32 = 8;

/// Submission slots per GPU (`EVOLUTION_GPU_SLOTS`, 1 to 8, default 4), each
/// on its own queue when the device offers enough.
pub fn gpu_slots() -> u32 {
    std::env::var("EVOLUTION_GPU_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n: &u32| (1..=8).contains(n))
        .unwrap_or(4)
}

/// Buffer size for `size` bytes of data. Small buffers round up to a power
/// of two, which costs little. Large ones get 25% headroom, so units of
/// slightly different sizes reuse them, without the up to 2x waste of a
/// power of two.
pub fn padded_size(size: u64) -> u64 {
    const LARGE: u64 = 1 << 20;
    let size = size.max(256);
    if size <= LARGE {
        size.next_power_of_two()
    } else {
        (size + size / 4).next_multiple_of(LARGE)
    }
}

/// True when `error` comes from a failed device or host memory allocation
/// (Vulkan or CUDA), which another process holding GPU memory can cause for
/// a while.
pub fn out_of_memory(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<vk::Result>(),
            Some(&vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | &vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        ) || cause
            .downcast_ref::<crate::cuda_engine::CudaError>()
            .is_some_and(crate::cuda_engine::CudaError::out_of_memory)
    })
}

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
            // Evaluation slots plus one for replays.
            let slot_count = gpu_slots() + 1;
            let queue_count = families[family as usize].queue_count.clamp(1, slot_count);
            let priorities = vec![1.0; queue_count as usize];
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
            let queues: Vec<vk::Queue> = (0..queue_count)
                .map(|index| device.get_device_queue(family, index))
                .collect();
            let memory_properties = instance.get_physical_device_memory_properties(physical);

            let bindings: Vec<_> = (0..BINDINGS)
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
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: (BINDINGS - 1) * MAX_SETS,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    descriptor_count: MAX_SETS,
                },
            ];
            let descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(MAX_SETS)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let command_buffers = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(slot_count),
            )?;
            let mut slots = Vec::new();
            for (index, command_buffer) in command_buffers.into_iter().enumerate() {
                slots.push(Slot {
                    queue: queues[index % queues.len()],
                    groups: (0..CAPACITIES.len()).map(|_| None).collect(),
                    params: None,
                    readback: None,
                    frames: None,
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
                name: device_name,
                memory_properties,
                set_layout,
                pipeline_layout,
                pipelines: Default::default(),
                recording: Default::default(),
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
    /// Nothing leaks when an allocation fails part way.
    fn create_buffer(&self, size: u64, usage: vk::BufferUsageFlags, readback: bool) -> Result<Buf> {
        let size = padded_size(size);
        unsafe {
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage),
                None,
            )?;
            match self.bind_mapped_memory(buffer, readback) {
                Ok((memory, ptr)) => Ok(Buf {
                    buffer,
                    memory,
                    size,
                    ptr,
                }),
                Err(error) => {
                    self.device.destroy_buffer(buffer, None);
                    Err(error)
                }
            }
        }
    }

    /// Allocates, binds and maps the memory of a new buffer.
    unsafe fn bind_mapped_memory(
        &self,
        buffer: vk::Buffer,
        readback: bool,
    ) -> Result<(vk::DeviceMemory, *mut u8)> {
        unsafe {
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
            let mapped = self
                .device
                .bind_buffer_memory(buffer, memory, 0)
                .and_then(|()| {
                    self.device
                        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                });
            match mapped {
                Ok(ptr) => Ok((memory, ptr as *mut u8)),
                Err(error) => {
                    self.device.free_memory(memory, None);
                    Err(error.into())
                }
            }
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
        read_state: bool,
        frame_bytes: u64,
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
            for group in 0..self.slots[slot].groups.len() {
                self.drop_group(slot, group);
            }
        }
        if self.slots[slot].groups.len() < batches.len() {
            self.slots[slot].groups.resize_with(batches.len(), || None);
        }
        let mut result_bytes = batches.iter().map(|b| b.info.len()).sum::<usize>() as u64
            * std::mem::size_of::<GpuResult>() as u64;
        if read_state {
            result_bytes += batches
                .iter()
                .map(|b| {
                    (std::mem::size_of_val(b.nodes.as_slice())
                        + std::mem::size_of_val(b.muscles.as_slice())) as u64
                })
                .sum::<u64>();
        }
        result_bytes += frame_bytes;
        if frame_bytes > 0
            && self.slots[slot]
                .frames
                .as_ref()
                .is_none_or(|b| b.size < frame_bytes)
        {
            if let Some(old) = self.slots[slot].frames.take() {
                self.destroy_buffer(old);
            }
            let frames = self.create_buffer(
                frame_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                false,
            )?;
            self.slots[slot].frames = Some(frames);
        }
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
        for (group, batch) in batches.iter().enumerate() {
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
            let readable = storage | vk::BufferUsageFlags::TRANSFER_SRC;
            let usages = [readable, readable, storage, readable, storage, storage];
            let mut made: Vec<Buf> = Vec::with_capacity(usages.len());
            let mut failure = None;
            for (bytes, usage) in need.into_iter().zip(usages) {
                match self.create_buffer(bytes, usage, false) {
                    Ok(buf) => made.push(buf),
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            let set = match failure {
                Some(error) => Err(error),
                None => unsafe {
                    self.device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(self.descriptor_pool)
                                .set_layouts(&[self.set_layout]),
                        )
                        .map(|sets| sets[0])
                        .map_err(anyhow::Error::from)
                },
            };
            let set = match set {
                Ok(set) => set,
                Err(error) => {
                    for buf in made {
                        self.destroy_buffer(buf);
                    }
                    return Err(error);
                }
            };
            let Ok([nodes, muscles, bones, results, info, tiles]) = <[Buf; 6]>::try_from(made)
            else {
                unreachable!("one buffer per binding");
            };
            let res = GroupRes {
                nodes,
                muscles,
                bones,
                results,
                info,
                tiles,
                set,
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
            self.slots[slot].groups[group] = Some(res);
        }
        // A recording writes its frames through binding 7. Scoring kernels
        // never use it, so their sets leave it unset.
        if frame_bytes > 0 {
            let frames = self.slots[slot].frames.as_ref().expect("frames buffer");
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(frames.buffer)
                .range(vk::WHOLE_SIZE)];
            let writes: Vec<_> = self.slots[slot].groups[..batches.len()]
                .iter()
                .flatten()
                .map(|group| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(group.set)
                        .dst_binding(BINDINGS - 1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info)
                })
                .collect();
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
        Ok(())
    }

    /// Bytes of buffers a slot keeps for reuse.
    fn slot_bytes(slot: &Slot) -> u64 {
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
            + slot.frames.as_ref().map_or(0, |b| b.size)
    }

    fn recount_allocated(&mut self) {
        self.allocated_bytes = self.slots.iter().map(Self::slot_bytes).sum();
    }

    /// Frees the buffers that idle slots keep for reuse, so a submission that
    /// ran out of device memory can try again. Returns the bytes freed.
    pub fn release_idle(&mut self) -> u64 {
        self.recount_allocated();
        let before = self.allocated_bytes;
        for slot in 0..self.slots.len() {
            if self.slots[slot].pending.is_some() {
                continue;
            }
            for group in 0..self.slots[slot].groups.len() {
                self.drop_group(slot, group);
            }
            if let Some(b) = self.slots[slot].params.take() {
                self.destroy_buffer(b);
            }
            if let Some(b) = self.slots[slot].readback.take() {
                self.destroy_buffer(b);
            }
            if let Some(b) = self.slots[slot].frames.take() {
                self.destroy_buffer(b);
            }
        }
        self.recount_allocated();
        before.saturating_sub(self.allocated_bytes)
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

    /// Number of evaluation submissions that can be queued without waiting.
    pub fn free_slots(&self) -> usize {
        self.evaluation_slots()
            .filter(|&i| self.slots[i].pending.is_none())
            .count()
    }

    /// Submissions in flight, replays included.
    pub fn in_flight(&self) -> usize {
        self.slots.iter().filter(|s| s.pending.is_some()).count()
    }

    fn evaluation_slots(&self) -> std::ops::Range<usize> {
        0..self.slots.len() - 1
    }

    fn replay_slot(&self) -> usize {
        self.slots.len() - 1
    }

    /// Whether a replay can be recorded now.
    pub fn replay_free(&self) -> bool {
        self.slots[self.replay_slot()].pending.is_none()
    }

    /// The recording kernel for `capacity`-node buckets, built on first use.
    fn recording_pipeline(
        &mut self,
        capacity: usize,
        fidelity: crate::physics::Fidelity,
    ) -> Result<vk::Pipeline> {
        if let Some(&pipeline) = self.recording.get(&(fidelity, capacity)) {
            return Ok(pipeline);
        }
        let task = crate::loading::start(crate::cuda_engine::kernel_label(
            "Vulkan", true, fidelity, capacity,
        ));
        let pipeline = Self::compile(
            &self.device,
            self.pipeline_layout,
            &crate::physics2::record_source(capacity, self.workgroup, fidelity),
        )?;
        task.finish(false);
        self.recording.insert((fidelity, capacity), pipeline);
        Ok(pipeline)
    }

    fn compile(
        device: &ash::Device,
        layout: vk::PipelineLayout,
        source: &str,
    ) -> Result<vk::Pipeline> {
        let code = spirv(source)?;
        unsafe {
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"advance");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e);
            device.destroy_shader_module(module, None);
            Ok(pipeline?[0])
        }
    }

    /// The scoring kernel for `capacity`-node buckets at `fidelity`, built
    /// on first use.
    fn standard_pipeline(
        &mut self,
        capacity: usize,
        fidelity: crate::physics::Fidelity,
    ) -> Result<vk::Pipeline> {
        if let Some(&pipeline) = self.pipelines.get(&(fidelity, capacity)) {
            return Ok(pipeline);
        }
        let started = std::time::Instant::now();
        let task = crate::loading::start(crate::cuda_engine::kernel_label(
            "Vulkan", false, fidelity, capacity,
        ));
        let pipeline = Self::compile(
            &self.device,
            self.pipeline_layout,
            &crate::physics2::shader_source(capacity, self.workgroup, fidelity),
        )?;
        task.finish(false);
        if std::env::var_os("EVOLUTION_VK_VERBOSE").is_some() {
            eprintln!(
                "Vulkan: kernel for {capacity} nodes at {fidelity:?} ready after {:.2} s",
                started.elapsed().as_secs_f64()
            );
        }
        self.pipelines.insert((fidelity, capacity), pipeline);
        Ok(pipeline)
    }

    /// Uploads the batches and queues ticks `start..end` of trials that last
    /// `total` ticks, in `chunk`-tick ranges, without waiting. With
    /// `read_state`, the node state and muscle buffers are read back too, so
    /// the trials can continue in another segment. Returns a ticket; results
    /// arrive through `poll`.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &mut self,
        batches: &[LaneBatch],
        cfg: &Config,
        start: u32,
        end: u32,
        total: u32,
        chunk: u32,
        read_state: bool,
    ) -> Result<u64> {
        self.submit_as(
            batches,
            cfg,
            start,
            end,
            total,
            chunk,
            read_state,
            false,
            &[],
        )
    }

    /// Queues a whole trial of one batch on the replay slot with the
    /// recording kernel, which writes every creature's frames. The result
    /// arrives through `poll` with `Completed::frames`. The trial is the one
    /// `submit` scores, computed the same way.
    pub fn record(
        &mut self,
        batch: &LaneBatch,
        cfg: &Config,
        total: u32,
        chunk: u32,
    ) -> Result<u64> {
        ensure!(self.replay_free(), "A replay is already being recorded");
        // Trials start at the settling tick, as scoring does.
        let start = cfg.fidelity().settle();
        // The kernel rebuilds its node table from joint state at every
        // dispatch, so bit-exact frames need scoring's dispatch boundaries.
        let cuts = crate::engine::segment_ends(cfg);
        self.submit_as(
            std::slice::from_ref(batch),
            cfg,
            start,
            total,
            total,
            chunk,
            false,
            true,
            &cuts,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_as(
        &mut self,
        batches: &[LaneBatch],
        cfg: &Config,
        start: u32,
        end: u32,
        total: u32,
        chunk: u32,
        read_state: bool,
        record: bool,
        cuts: &[u32],
    ) -> Result<u64> {
        ensure!(
            !batches.is_empty() && start < end && end <= total,
            "Empty GPU batch"
        );
        ensure!(
            batches.iter().all(|b| b.capacity <= self.max_capacity),
            "Body too large for this device's kernels"
        );
        let fidelity = cfg.fidelity();
        // The free slot with the most buffers to reuse: when memory is short,
        // a new allocation may fail where reuse does not.
        let slot = if record {
            self.replay_slot()
        } else {
            self.evaluation_slots()
                .filter(|&i| self.slots[i].pending.is_none())
                .max_by_key(|&i| (Self::slot_bytes(&self.slots[i]), std::cmp::Reverse(i)))
                .context("No free GPU submission slot")?
        };
        // Dispatches: (first tick, steps), in `chunk`-tick pieces that never
        // cross a cut.
        let mut spans: Vec<(u32, u32)> = Vec::new();
        let mut tick = start;
        while tick < end {
            let stop = cuts
                .iter()
                .copied()
                .find(|&c| c > tick && c < end)
                .unwrap_or(end);
            let steps = (stop - tick).min(chunk);
            spans.push((tick, steps));
            tick += steps;
        }
        let ranges = spans.len() as u32;
        let kernels: Vec<vk::Pipeline> = batches
            .iter()
            .map(|batch| {
                if record {
                    self.recording_pipeline(batch.capacity, fidelity)
                } else {
                    self.standard_pipeline(batch.capacity, fidelity)
                }
            })
            .collect::<Result<_>>()?;
        // Node positions for every creature, before each step and after the last.
        let frame_count: usize = if record {
            batches
                .iter()
                .map(|b| b.info.len() * creature_kernel::frame_stride(b) * (total as usize + 1))
                .sum()
        } else {
            0
        };
        let frame_bytes = (frame_count * std::mem::size_of::<[f32; 2]>()) as u64;
        let buffers = self.ensure_buffers(
            slot,
            batches,
            u64::from(ranges) * batches.len() as u64,
            read_state,
            frame_bytes,
        );
        self.recount_allocated();
        buffers?;
        let mut param_data =
            vec![0u8; (u64::from(ranges) * batches.len() as u64 * self.params_stride) as usize];
        for (r, &(tick, steps)) in spans.iter().enumerate() {
            for (b, batch) in batches.iter().enumerate() {
                let offset = ((r * batches.len() + b) as u64 * self.params_stride) as usize;
                let mut p = creature_kernel::launch_params(
                    cfg,
                    batch.capacity,
                    batch.info.len(),
                    tick,
                    steps,
                    total,
                );
                if record {
                    p.stride = creature_kernel::frame_stride(batch) as u32;
                }
                param_data[offset..offset + std::mem::size_of::<Params>()]
                    .copy_from_slice(bytemuck::bytes_of(&p));
            }
        }
        let resources = &self.slots[slot];
        Self::write(resources.params.as_ref().unwrap(), &param_data);
        for (group, batch) in batches.iter().enumerate() {
            let res = resources.groups[group].as_ref().unwrap();
            Self::write(&res.nodes, &batch.nodes);
            Self::write(&res.muscles, &batch.muscles);
            Self::write(&res.bones, &batch.bones);
            Self::write(&res.info, &batch.info);
            Self::write(&res.tiles, &batch.tiles);
            if let Some(results) = &batch.results {
                Self::write(&res.results, results);
            }
        }
        let device = &self.device;
        let cb = resources.command_buffer;
        let result_count: usize = batches.iter().map(|b| b.info.len()).sum();
        let mut frames_offset = 0u64;
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
                    let res = resources.groups[b].as_ref().unwrap();
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernels[b]);
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
            for (group, batch) in batches.iter().enumerate() {
                let res = resources.groups[group].as_ref().unwrap();
                let bytes = (batch.info.len() * std::mem::size_of::<GpuResult>()) as u64;
                device.cmd_copy_buffer(
                    cb,
                    res.results.buffer,
                    readback.buffer,
                    &[vk::BufferCopy::default().dst_offset(offset).size(bytes)],
                );
                offset += bytes;
            }
            if read_state {
                for (group, batch) in batches.iter().enumerate() {
                    let res = resources.groups[group].as_ref().unwrap();
                    for (buffer, bytes) in [
                        (
                            res.nodes.buffer,
                            std::mem::size_of_val(batch.nodes.as_slice()) as u64,
                        ),
                        (
                            res.muscles.buffer,
                            std::mem::size_of_val(batch.muscles.as_slice()) as u64,
                        ),
                    ] {
                        device.cmd_copy_buffer(
                            cb,
                            buffer,
                            readback.buffer,
                            &[vk::BufferCopy::default().dst_offset(offset).size(bytes)],
                        );
                        offset += bytes;
                    }
                }
            }
            if record {
                frames_offset = offset;
                device.cmd_copy_buffer(
                    cb,
                    resources.frames.as_ref().expect("frames buffer").buffer,
                    readback.buffer,
                    &[vk::BufferCopy::default()
                        .dst_offset(offset)
                        .size(frame_bytes)],
                );
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
                resources.queue,
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
            state: read_state.then(|| {
                batches
                    .iter()
                    .map(|b| (b.nodes.len(), b.muscles.len()))
                    .collect()
            }),
            frames: record.then_some((frames_offset, frame_count)),
        });
        Ok(ticket)
    }

    /// Returns the oldest submission's results once it has finished, waiting up
    /// to `timeout`. Submissions complete in order on the single queue.
    pub fn poll(&mut self, timeout: std::time::Duration) -> Result<Option<Completed>> {
        // Units on different queues finish in any order: return the oldest
        // finished one, waiting up to `timeout` for any of them.
        let mut pending: Vec<(u64, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.pending.as_ref().map(|p| (p.ticket, i)))
            .collect();
        if pending.is_empty() {
            return Ok(None);
        }
        pending.sort_unstable();
        let finished = |this: &Self| -> Result<Option<usize>> {
            for &(_, i) in &pending {
                match unsafe { this.device.get_fence_status(this.slots[i].fence) } {
                    Ok(true) => return Ok(Some(i)),
                    Ok(false) => {}
                    Err(e) => return Err(e.into()),
                }
            }
            Ok(None)
        };
        let slot = match finished(self)? {
            Some(slot) => slot,
            None => {
                let fences: Vec<vk::Fence> =
                    pending.iter().map(|&(_, i)| self.slots[i].fence).collect();
                match unsafe {
                    self.device.wait_for_fences(
                        &fences,
                        false,
                        timeout.as_nanos().min(u128::from(u64::MAX)) as u64,
                    )
                } {
                    Ok(()) => {}
                    Err(vk::Result::TIMEOUT) => return Ok(None),
                    Err(e) => return Err(e.into()),
                }
                match finished(self)? {
                    Some(slot) => slot,
                    None => return Ok(None),
                }
            }
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
            let state = pending.state.map(|sizes| {
                let mut at = readback
                    .ptr
                    .add(pending.result_count * std::mem::size_of::<GpuResult>());
                sizes
                    .into_iter()
                    .map(|(nodes, muscles)| {
                        let node_state =
                            std::slice::from_raw_parts(at as *const crate::physics::Node, nodes)
                                .to_vec();
                        at = at.add(nodes * std::mem::size_of::<crate::physics::Node>());
                        let muscle_state =
                            std::slice::from_raw_parts(at as *const f32, muscles).to_vec();
                        at = at.add(muscles * std::mem::size_of::<f32>());
                        (node_state, muscle_state)
                    })
                    .collect()
            });
            let frames = pending.frames.map(|(offset, count)| {
                std::slice::from_raw_parts(
                    readback.ptr.add(offset as usize) as *const [f32; 2],
                    count,
                )
                .to_vec()
            });
            self.last_gpu_seconds = gpu_seconds;
            Ok(Some(Completed {
                ticket: pending.ticket,
                batches,
                state,
                frames,
                gpu_seconds,
            }))
        }
    }

    /// Waits up to `timeout` for any submission without collecting it.
    pub fn wait(&self, timeout: std::time::Duration) -> Result<()> {
        let fences: Vec<vk::Fence> = self
            .slots
            .iter()
            .filter(|s| s.pending.is_some())
            .map(|s| s.fence)
            .collect();
        if fences.is_empty() {
            return Ok(());
        }
        unsafe {
            match self
                .device
                .wait_for_fences(&fences, false, timeout.as_nanos() as u64)
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
        let ticket = self.submit(batches, cfg, 0, steps, steps, chunk, false)?;
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
            for group in 0..self.slots[slot].groups.len() {
                self.drop_group(slot, group);
            }
            if let Some(b) = self.slots[slot].params.take() {
                self.destroy_buffer(b);
            }
            if let Some(b) = self.slots[slot].readback.take() {
                self.destroy_buffer(b);
            }
            if let Some(b) = self.slots[slot].frames.take() {
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
            for &p in self.pipelines.values().chain(self.recording.values()) {
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
