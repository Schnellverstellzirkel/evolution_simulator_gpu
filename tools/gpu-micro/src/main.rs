//! GPU microbenchmarks for the RTX 4060 Laptop GPU (docs/phase0-measurements.md).
//!
//! Kernels are WGSL compiled to SPIR-V with naga, as in the game, and timed
//! with Vulkan timestamp queries around each dispatch. A child `nvidia-smi`
//! samples the SM clock, memory clock, power and temperature every 100 ms;
//! each timed repetition is paired with the samples taken while it ran, so
//! results can be given per cycle at the measured clock.
//!
//! The tool opens only a Vulkan device whose name contains "RTX 4060" and
//! fails otherwise. It never touches the Radeon.
//!
//! Usage: gpu-micro <test> [--option value ...]
//!   info
//!   compile        (builds every kernel and prints the driver statistics)
//!   bandwidth      --mode read|copy --size 8m [--coherent 0|1] [--data hashed|const]
//!                  [--wg 256] [--groups 144]
//!   chase          --regions 16k,8m,1g [--stride 128] [--order random|linear]
//!   shared-latency
//!   alu-latency    [--ops fma,fadd,imad,sqrt,rsqrt,div]
//!   alu-throughput [--ops fma,fadd,imad,sqrt,rsqrt,div]
//!   shared-bw      [--n 8] [--l 16] [--store 0|1] [--k 1,2,4,8,12,16,24] [--saturate 1]
//!   occupancy      [--cases 0:0,6144:0,6272:0] [--kmax 30]   (shared bytes:live f32 values)
//! Common: --reps 5 (timed repetitions), --target-ms 120 (length of one repetition).
//! `MICRO_DUMP_IR=<dir>` writes each kernel's driver internal representations,
//! if the driver offers any (the NVIDIA driver on this machine offered none).
//!
//! Build: `CARGO_BUILD_JOBS=1 nice -n 10 cargo build --release --offline` in this
//! directory. Run every GPU test under the shared GPU lock (`flock <lock> ...`).
use anyhow::{Context, Result, bail, ensure};
use ash::vk;
use std::ffi::CStr;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEVICE: &str = "RTX 4060";

#[derive(Clone, Copy, Default, Debug)]
struct Sample {
    sm: f64,
    mem: f64,
    power: f64,
    temp: f64,
}

/// nvidia-smi running in the background, one sample every 100 ms.
struct Sampler {
    samples: Arc<Mutex<Vec<(Instant, Sample)>>>,
    child: Child,
}

impl Sampler {
    fn start() -> Result<Self> {
        let mut child = Command::new("nvidia-smi")
            .args([
                "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                "100",
                "-i",
                "0",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("starting nvidia-smi")?;
        let out = child.stdout.take().context("nvidia-smi stdout")?;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let sink = samples.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else { break };
                let v: Vec<f64> = line
                    .split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect();
                if v.len() == 4 {
                    sink.lock().unwrap().push((
                        Instant::now(),
                        Sample {
                            sm: v[0],
                            mem: v[1],
                            power: v[2],
                            temp: v[3],
                        },
                    ));
                }
            }
        });
        Ok(Self { samples, child })
    }

    /// Median of the samples read while [a, b] ran, or the sample nearest
    /// the middle of the window when none fell inside it.
    fn window(&self, a: Instant, b: Instant) -> Sample {
        let all = self.samples.lock().unwrap();
        let inside: Vec<Sample> = all
            .iter()
            .filter(|(t, _)| *t >= a && *t <= b)
            .map(|(_, s)| *s)
            .collect();
        if !inside.is_empty() {
            return median_sample(&inside);
        }
        let mid = a + (b - a) / 2;
        all.iter()
            .min_by_key(|(t, _)| if *t > mid { *t - mid } else { mid - *t })
            .map(|(_, s)| *s)
            .unwrap_or_default()
    }

    fn all(&self) -> Vec<Sample> {
        self.samples
            .lock()
            .unwrap()
            .iter()
            .map(|(_, s)| *s)
            .collect()
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn median(v: &[f64]) -> f64 {
    let mut v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn median_sample(s: &[Sample]) -> Sample {
    Sample {
        sm: median(&s.iter().map(|x| x.sm).collect::<Vec<_>>()),
        mem: median(&s.iter().map(|x| x.mem).collect::<Vec<_>>()),
        power: median(&s.iter().map(|x| x.power).collect::<Vec<_>>()),
        temp: median(&s.iter().map(|x| x.temp).collect::<Vec<_>>()),
    }
}

struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    ptr: *mut u8,
}

struct Gpu {
    _entry: ash::Entry,
    _instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    memory: vk::PhysicalDeviceMemoryProperties,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    queries: vk::QueryPool,
    period_ns: f64,
    exec: Option<ash::khr::pipeline_executable_properties::Device>,
    params: Buf,
    out: Buf,
    sm_count: u32,
}

/// One timed repetition: GPU nanoseconds and the clocks while it ran.
#[derive(Clone, Copy)]
struct Rep {
    ns: f64,
    s: Sample,
}

impl Gpu {
    fn new() -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load().context("Vulkan loader")?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"gpu-micro")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
                .context("Vulkan instance")?;
            let mut chosen = None;
            for d in instance.enumerate_physical_devices()? {
                let props = instance.get_physical_device_properties(d);
                let name = CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                if name.contains(DEVICE) && chosen.is_none() {
                    chosen = Some((d, name, props));
                }
            }
            let (physical, name, props) = chosen.with_context(|| {
                format!("no Vulkan device named {DEVICE:?}; refusing to run on any other GPU")
            })?;
            let extensions: Vec<String> = instance
                .enumerate_device_extension_properties(physical)?
                .iter()
                .map(|e| {
                    CStr::from_ptr(e.extension_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            let has = |n: &CStr| extensions.iter().any(|e| e.as_str() == n.to_str().unwrap());
            let mut sm_count = 0;
            let mut warps_per_sm = 0;
            if has(vk::NV_SHADER_SM_BUILTINS_NAME) {
                let mut sm = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sm);
                instance.get_physical_device_properties2(physical, &mut p2);
                sm_count = sm.shader_sm_count;
                warps_per_sm = sm.shader_warps_per_sm;
            }
            ensure!(sm_count > 0, "the driver did not report the SM count");
            let families = instance.get_physical_device_queue_family_properties(physical);
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
                .context("no compute queue")?;
            ensure!(
                families[family].timestamp_valid_bits > 0,
                "compute queue has no timestamps"
            );
            let priorities = [1.0];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family as u32)
                .queue_priorities(&priorities)];
            let exec_ext = ash::khr::pipeline_executable_properties::NAME;
            let with_exec = has(exec_ext);
            let mut exec_features =
                vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
                    .pipeline_executable_info(true);
            let ext_names = [exec_ext.as_ptr()];
            let mut create = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
            if with_exec {
                create = create
                    .enabled_extension_names(&ext_names)
                    .push_next(&mut exec_features);
            }
            let device = instance
                .create_device(physical, &create, None)
                .context("Vulkan device")?;
            let queue = device.get_device_queue(family as u32, 0);
            let memory = instance.get_physical_device_memory_properties(physical);
            let bindings: Vec<_> = (0..4u32)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let set_layouts = [set_layout];
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )?;
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4 * 64,
            }];
            let descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(64)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family as u32)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(1),
            )?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let queries = device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?;
            let exec = with_exec
                .then(|| ash::khr::pipeline_executable_properties::Device::new(&instance, &device));
            println!(
                "device: {name} | SMs {sm_count} | warps/SM {warps_per_sm} | \
                 max shared/workgroup {} B | timestamp period {} ns ({} valid bits) | \
                 max workgroups x {} | queue family {family}",
                props.limits.max_compute_shared_memory_size,
                props.limits.timestamp_period,
                families[family].timestamp_valid_bits,
                props.limits.max_compute_work_group_count[0],
            );
            let mut gpu = Self {
                _entry: entry,
                _instance: instance,
                device,
                queue,
                memory,
                set_layout,
                layout,
                descriptor_pool,
                cmd,
                fence,
                queries,
                period_ns: props.limits.timestamp_period as f64,
                exec,
                params: Buf {
                    buffer: vk::Buffer::null(),
                    memory: vk::DeviceMemory::null(),
                    size: 0,
                    ptr: std::ptr::null_mut(),
                },
                out: Buf {
                    buffer: vk::Buffer::null(),
                    memory: vk::DeviceMemory::null(),
                    size: 0,
                    ptr: std::ptr::null_mut(),
                },
                sm_count,
            };
            gpu.params = gpu.buffer(256, Mem::Upload)?;
            gpu.out = gpu.buffer(1 << 16, Mem::Upload)?;
            Ok(gpu)
        }
    }

    fn buffer(&self, size: u64, kind: Mem) -> Result<Buf> {
        unsafe {
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                ),
                None,
            )?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            use vk::MemoryPropertyFlags as F;
            // (wanted flags, flags that must be absent)
            let choices: &[(F, F)] = match kind {
                Mem::Device => &[
                    (F::DEVICE_LOCAL, F::HOST_VISIBLE),
                    (F::DEVICE_LOCAL, F::empty()),
                ],
                Mem::Upload => &[
                    (
                        F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT,
                        F::empty(),
                    ),
                    (F::HOST_VISIBLE | F::HOST_COHERENT, F::empty()),
                ],
                Mem::Host => &[
                    (
                        F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
                        F::empty(),
                    ),
                    (F::HOST_VISIBLE | F::HOST_COHERENT, F::empty()),
                ],
            };
            let mut index = None;
            'outer: for (want, avoid) in choices {
                for i in 0..self.memory.memory_type_count {
                    let f = self.memory.memory_types[i as usize].property_flags;
                    if req.memory_type_bits & (1 << i) != 0
                        && f.contains(*want)
                        && !f.intersects(*avoid)
                    {
                        index = Some(i);
                        break 'outer;
                    }
                }
            }
            let index = index.context("no suitable memory type")?;
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(index),
                    None,
                )
                .with_context(|| format!("allocating {size} bytes"))?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr = if matches!(kind, Mem::Device) {
                std::ptr::null_mut()
            } else {
                self.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    as *mut u8
            };
            Ok(Buf {
                buffer,
                memory,
                size,
                ptr,
            })
        }
    }

    fn free(&self, b: Buf) {
        unsafe {
            if !b.ptr.is_null() {
                self.device.unmap_memory(b.memory);
            }
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    fn set(&self, bufs: [&Buf; 4]) -> Result<vk::DescriptorSet> {
        unsafe {
            let layouts = [self.set_layout];
            let set = self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts),
            )?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b.buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);
            Ok(set)
        }
    }

    fn free_set(&self, set: vk::DescriptorSet) {
        unsafe {
            let _ = self
                .device
                .free_descriptor_sets(self.descriptor_pool, &[set]);
        }
    }

    /// Compiles WGSL with naga (as `src/vk_engine.rs` does) and prints the
    /// driver's statistics for the kernel.
    fn pipeline(&self, label: &str, wgsl: &str) -> Result<vk::Pipeline> {
        self.pipeline_with(label, wgsl, &[])
    }

    /// As `pipeline`, and decorates the storage buffers at the given bindings
    /// `Coherent` in the SPIR-V. The driver then serves their loads from L2,
    /// not from the SM's incoherent L1, which WGSL cannot ask for.
    fn pipeline_with(&self, label: &str, wgsl: &str, coherent: &[u32]) -> Result<vk::Pipeline> {
        let module = naga::front::wgsl::parse_str(wgsl)
            .map_err(|e| anyhow::anyhow!("{label}: {}", e.emit_to_string(wgsl)))?;
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .map_err(|e| anyhow::anyhow!("{label}: {}", e.emit_to_string(wgsl)))?;
        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            flags: naga::back::spv::WriterFlags::empty(),
            ..Default::default()
        };
        let stage = naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
        };
        let mut code = naga::back::spv::write_vec(&module, &info, &options, Some(&stage))
            .context("SPIR-V output")?;
        if !coherent.is_empty() {
            code = decorate_coherent(&code, coherent)?;
        }
        unsafe {
            let shader = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let mut flags = vk::PipelineCreateFlags::empty();
            if self.exec.is_some() {
                flags |= vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
                    | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR;
            }
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader)
                        .name(c"main"),
                )
                .layout(self.layout)
                .flags(flags);
            let pipeline = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)?[0];
            self.device.destroy_shader_module(shader, None);
            if let Some(exec) = &self.exec {
                self.stats(exec, pipeline, label)?;
            }
            Ok(pipeline)
        }
    }

    unsafe fn stats(
        &self,
        exec: &ash::khr::pipeline_executable_properties::Device,
        pipeline: vk::Pipeline,
        label: &str,
    ) -> Result<()> {
        unsafe {
            let pinfo = vk::PipelineInfoKHR::default().pipeline(pipeline);
            let count = exec.get_pipeline_executable_properties(&pinfo)?.len();
            for index in 0..count as u32 {
                let einfo = vk::PipelineExecutableInfoKHR::default()
                    .pipeline(pipeline)
                    .executable_index(index);
                let stats = exec.get_pipeline_executable_statistics(&einfo)?;
                let text: Vec<String> = stats
                    .iter()
                    .map(|s| {
                        let name = CStr::from_ptr(s.name.as_ptr()).to_string_lossy();
                        let value = match s.format {
                            vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                                s.value.b32.to_string()
                            }
                            vk::PipelineExecutableStatisticFormatKHR::INT64 => {
                                s.value.i64.to_string()
                            }
                            vk::PipelineExecutableStatisticFormatKHR::UINT64 => {
                                s.value.u64.to_string()
                            }
                            _ => format!("{:.3}", s.value.f64),
                        };
                        format!("{name}={value}")
                    })
                    .collect();
                println!("kernel {label}: {}", text.join(", "));
                if let Some(dir) = std::env::var_os("MICRO_DUMP_IR") {
                    let fp = exec.fp();
                    let raw = self.device.handle();
                    let mut n = 0u32;
                    let _ = (fp.get_pipeline_executable_internal_representations_khr)(
                        raw,
                        &einfo,
                        &mut n,
                        std::ptr::null_mut(),
                    );
                    let mut reps = vec![
                        vk::PipelineExecutableInternalRepresentationKHR::default();
                        n as usize
                    ];
                    let _ = (fp.get_pipeline_executable_internal_representations_khr)(
                        raw,
                        &einfo,
                        &mut n,
                        reps.as_mut_ptr(),
                    );
                    let mut data: Vec<Vec<u8>> =
                        reps.iter().map(|r| vec![0u8; r.data_size]).collect();
                    for (r, d) in reps.iter_mut().zip(&mut data) {
                        r.p_data = d.as_mut_ptr().cast();
                    }
                    let _ = (fp.get_pipeline_executable_internal_representations_khr)(
                        raw,
                        &einfo,
                        &mut n,
                        reps.as_mut_ptr(),
                    );
                    for (i, (r, d)) in reps.iter().zip(&data).enumerate() {
                        let name = CStr::from_ptr(r.name.as_ptr()).to_string_lossy();
                        let path = std::path::Path::new(&dir).join(format!(
                            "{}-{i}-{name}.txt",
                            label.replace([' ', '/', '='], "_")
                        ));
                        std::fs::write(path, d)?;
                    }
                }
            }
            Ok(())
        }
    }

    fn submit(&self) -> Result<()> {
        unsafe {
            let cmds = [self.cmd];
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                self.fence,
            )?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.fence])?;
            Ok(())
        }
    }

    fn fill(&self, b: &Buf, value: u32) -> Result<()> {
        unsafe {
            self.device
                .begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default())?;
            self.device
                .cmd_fill_buffer(self.cmd, b.buffer, 0, vk::WHOLE_SIZE, value);
            self.device.end_command_buffer(self.cmd)?;
        }
        self.submit()
    }

    /// Fills a buffer with hashed words, so no hardware compression can help.
    /// A constant fill (`--data const`) measured the same bandwidth at 64 MiB
    /// and 1 GiB, so this is a precaution.
    fn randomize(&self, b: &Buf, seed: u32) -> Result<()> {
        ensure!(b.size % 16 == 0);
        let pipeline = self.pipeline("randomize", &randomize_wgsl())?;
        let set = self.set([&self.params, b, &self.out, &self.out])?;
        let n = u32::try_from(b.size / 16).context("buffer too large")?;
        self.run(pipeline, set, [self.sm_count * 24, 1, 1], &[n, seed])?;
        self.free_set(set);
        unsafe { self.device.destroy_pipeline(pipeline, None) };
        Ok(())
    }

    /// Runs one dispatch and returns its GPU time in nanoseconds.
    fn run(
        &self,
        pipeline: vk::Pipeline,
        set: vk::DescriptorSet,
        groups: [u32; 3],
        params: &[u32],
    ) -> Result<f64> {
        ensure!(params.len() <= 16);
        unsafe {
            std::ptr::copy_nonoverlapping(
                params.as_ptr().cast::<u8>(),
                self.params.ptr,
                params.len() * 4,
            );
            let d = &self.device;
            d.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default())?;
            d.cmd_reset_query_pool(self.cmd, self.queries, 0, 2);
            d.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
            d.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.queries,
                0,
            );
            d.cmd_dispatch(self.cmd, groups[0], groups[1], groups[2]);
            d.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.queries,
                1,
            );
            d.end_command_buffer(self.cmd)?;
            self.submit()?;
            let mut t = [0u64; 2];
            d.get_query_pool_results(
                self.queries,
                0,
                &mut t,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
            Ok(t[1].wrapping_sub(t[0]) as f64 * self.period_ns)
        }
    }

    /// Finds a work size that makes one dispatch last about `target_ms`.
    fn calibrate(
        &self,
        pipeline: vk::Pipeline,
        set: vk::DescriptorSet,
        groups: [u32; 3],
        params: &dyn Fn(u32) -> Vec<u32>,
        start: u32,
        target_ms: f64,
    ) -> Result<u32> {
        let mut n = start.max(1);
        for _ in 0..4 {
            let ns = self.run(pipeline, set, groups, &params(n))?;
            let scale = target_ms * 1e6 / ns.max(1.0);
            let next = ((n as f64) * scale).clamp(1.0, 4.0e9) as u32;
            if (0.8..1.25).contains(&scale) {
                return Ok(next);
            }
            n = next;
        }
        Ok(n)
    }

    /// Warms up for `warm`, then times `reps` dispatches with their clocks.
    #[allow(clippy::too_many_arguments)]
    fn measure(
        &self,
        sampler: &Sampler,
        pipeline: vk::Pipeline,
        set: vk::DescriptorSet,
        groups: [u32; 3],
        params: &[u32],
        reps: usize,
        warm: Duration,
    ) -> Result<Vec<Rep>> {
        let start = Instant::now();
        let mut n = 0;
        while n < 2 || start.elapsed() < warm {
            self.run(pipeline, set, groups, params)?;
            n += 1;
        }
        let mut windows = Vec::new();
        for _ in 0..reps {
            let a = Instant::now();
            let ns = self.run(pipeline, set, groups, params)?;
            windows.push((a, Instant::now(), ns));
        }
        // Let the sampler deliver the samples of the last repetition.
        std::thread::sleep(Duration::from_millis(250));
        Ok(windows
            .into_iter()
            .map(|(a, b, ns)| Rep {
                ns,
                s: sampler.window(a, b + Duration::from_millis(40)),
            })
            .collect())
    }
}

/// Adds `OpDecorate %var Coherent` for every variable bound at one of
/// `bindings`, right after the module's last existing decoration.
fn decorate_coherent(code: &[u32], bindings: &[u32]) -> Result<Vec<u32>> {
    const OP_DECORATE: u32 = 71;
    const BINDING: u32 = 33;
    const COHERENT: u32 = 23;
    let mut targets = Vec::new();
    let mut after_last = None;
    let mut i = 5;
    while i < code.len() {
        let count = (code[i] >> 16) as usize;
        ensure!(count > 0, "bad SPIR-V");
        if code[i] & 0xffff == OP_DECORATE {
            if count == 4 && code[i + 2] == BINDING && bindings.contains(&code[i + 3]) {
                targets.push(code[i + 1]);
            }
            after_last = Some(i + count);
        }
        i += count;
    }
    ensure!(
        targets.len() == bindings.len(),
        "found {} of {} bindings to decorate",
        targets.len(),
        bindings.len()
    );
    let at = after_last.context("no decorations")?;
    let mut out = code[..at].to_vec();
    for t in targets {
        out.extend([(3 << 16) | OP_DECORATE, t, COHERENT]);
    }
    out.extend_from_slice(&code[at..]);
    Ok(out)
}

#[derive(Clone, Copy)]
enum Mem {
    /// Device-local, not host-visible.
    Device,
    /// Host-visible, device-local when possible (small parameter buffers).
    Upload,
    /// Host-visible system memory.
    Host,
}

struct Args(Vec<String>);

impl Args {
    fn get(&self, name: &str) -> Option<&str> {
        let key = format!("--{name}");
        self.0
            .iter()
            .position(|a| *a == key)
            .and_then(|i| self.0.get(i + 1))
            .map(String::as_str)
    }
    fn num<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T> {
        match self.get(name) {
            Some(v) => v
                .parse()
                .map_err(|_| anyhow::anyhow!("bad value for --{name}: {v}")),
            None => Ok(default),
        }
    }
    fn list(&self, name: &str, default: &str) -> Vec<String> {
        self.get(name)
            .unwrap_or(default)
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

fn parse_size(s: &str) -> Result<u64> {
    let s = s.to_lowercase();
    let (num, mult) = match s.chars().last() {
        Some('k') => (&s[..s.len() - 1], 1u64 << 10),
        Some('m') => (&s[..s.len() - 1], 1 << 20),
        Some('g') => (&s[..s.len() - 1], 1 << 30),
        _ => (s.as_str(), 1),
    };
    Ok(num.parse::<u64>().context("size")? * mult)
}

/// A named metric computed from one repetition.
type Metric<'a> = (&'a str, &'a dyn Fn(&Rep) -> f64);

/// Prints one result line: each metric's median and range over the
/// repetitions, then the clocks sampled during them.
fn report(label: &str, reps: &[Rep], metrics: &[Metric]) {
    let mut line = format!("RESULT {label} |");
    for (name, f) in metrics {
        let v: Vec<f64> = reps.iter().map(f).collect();
        let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let m = median(&v);
        let spread = if m != 0.0 { 100.0 * (hi - lo) / m } else { 0.0 };
        line += &format!(" {name} {m:.3} [{lo:.3} .. {hi:.3}, spread {spread:.1}%] |");
    }
    let sm: Vec<f64> = reps.iter().map(|r| r.s.sm).collect();
    let sm_lo = sm.iter().copied().fold(f64::INFINITY, f64::min);
    let sm_hi = sm.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    line += &format!(
        " n={} ms/rep {:.1} | SM MHz {:.0} [{sm_lo:.0}..{sm_hi:.0}] mem MHz {:.0} power W {:.1} temp C {:.0}",
        reps.len(),
        median(&reps.iter().map(|r| r.ns * 1e-6).collect::<Vec<_>>()),
        median(&sm),
        median(&reps.iter().map(|r| r.s.mem).collect::<Vec<_>>()),
        median(&reps.iter().map(|r| r.s.power).collect::<Vec<_>>()),
        median(&reps.iter().map(|r| r.s.temp).collect::<Vec<_>>()),
    );
    println!("{line}");
}

/// Hz of a repetition's SM clock.
fn hz(r: &Rep) -> f64 {
    r.s.sm * 1e6
}

// ---------------------------------------------------------------- kernels

const HEADER: &str = "@group(0) @binding(0) var<storage, read> p: array<u32, 16>;\n\
@group(0) @binding(3) var<storage, read_write> o: array<u32>;\n";

fn stream_wgsl(wg: u32, copy: bool, coherent: bool) -> String {
    let access = if coherent { "read_write" } else { "read" };
    let body = if copy {
        "let x0 = a[i]; let x1 = a[i + t]; let x2 = a[i + 2u * t]; let x3 = a[i + 3u * t];\n\
         b[i] = x0 ^ k; b[i + t] = x1 ^ k; b[i + 2u * t] = x2 ^ k; b[i + 3u * t] = x3 ^ k;"
    } else {
        "acc = acc ^ a[i] ^ a[i + t] ^ a[i + 2u * t] ^ a[i + 3u * t];"
    };
    let tail = if copy {
        "b[i] = a[i] ^ k;"
    } else {
        "acc = acc ^ a[i];"
    };
    format!(
        r#"{HEADER}
@group(0) @binding(1) var<storage, {access}> a: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> b: array<vec4<u32>>;

@compute @workgroup_size({wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(num_workgroups) nw: vec3<u32>) {{
    let n = p[0];
    let passes = p[1];
    let groups = nw.x;
    let t = groups * {wg}u;
    var acc = vec4<u32>(0u);
    for (var r = 0u; r < passes; r++) {{
        let k = vec4<u32>(r);
        // Rotate the chunk each pass so an SM never reads the same lines twice in a row.
        let g = (wid.x + r * 7u) % groups;
        var i = g * {wg}u + li;
        loop {{
            if (i + 3u * t >= n) {{ break; }}
            {body}
            i += 4u * t;
        }}
        loop {{
            if (i >= n) {{ break; }}
            {tail}
            i += t;
        }}
    }}
    if (acc.x == 0x9e3779b9u && acc.y == 0x7f4a7c15u) {{ o[li] = acc.z; }}
}}
"#
    )
}

fn randomize_wgsl() -> String {
    format!(
        r#"{HEADER}
@group(0) @binding(1) var<storage, read_write> a: array<vec4<u32>>;

fn hash(x: u32) -> u32 {{
    var h = x * 0x9e3779b9u;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    return h ^ (h >> 16u);
}}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let n = p[0];
    let t = nw.x * 256u;
    for (var i = gid.x; i < n; i += t) {{
        let b = (i ^ p[1]) * 4u;
        a[i] = vec4<u32>(hash(b), hash(b + 1u), hash(b + 2u), hash(b + 3u));
    }}
}}
"#
    )
}

fn scatter_wgsl() -> String {
    format!(
        r#"{HEADER}
@group(0) @binding(1) var<storage, read_write> c: array<u32>;
@group(0) @binding(2) var<storage, read_write> next: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= p[0]) {{ return; }}
    let s = p[1];
    c[i * s] = next[i] * s;
}}
"#
    )
}

fn chase_wgsl() -> String {
    let hops = "        j = c[j];\n".repeat(16);
    format!(
        r#"{HEADER}
@group(0) @binding(1) var<storage, read_write> c: array<u32>;

@compute @workgroup_size(1)
fn main() {{
    var j = p[2];
    let n = p[0];
    for (var k = 0u; k < n; k++) {{
{hops}    }}
    o[0] = j;
}}
"#
    )
}

fn shared_chase_wgsl() -> String {
    let hops = "            j = sh[j];\n".repeat(16);
    format!(
        r#"{HEADER}
@group(0) @binding(2) var<storage, read_write> next: array<u32>;
var<workgroup> sh: array<u32, 1024>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) li: u32) {{
    for (var i = li; i < 1024u; i += 32u) {{ sh[i] = next[i]; }}
    workgroupBarrier();
    if (li == 0u) {{
        var j = p[2];
        let n = p[0];
        for (var k = 0u; k < n; k++) {{
{hops}        }}
        o[0] = j;
    }}
}}
"#
    )
}

/// The expression one ALU op applies to `v`; `a` and `b` are runtime constants.
/// The driver reassociates floating point freely: a plain `v + a` chain
/// compiled to one add, and `b / v` chains cancelled in pairs. `abs(v) + a`
/// is one FADD with an operand modifier, and `b / (v + a)` is one add plus
/// one division, which is also the shape the creature kernel uses.
fn op_expr(op: &str, v: &str) -> Result<String> {
    Ok(match op {
        "fma" => format!("fma({v}, a, b)"),
        "fadd" => format!("abs({v}) + a"),
        "sqrt" => format!("sqrt({v})"),
        "rsqrt" => format!("inverseSqrt({v})"),
        "div" => format!("b / ({v} + a)"),
        "imad" => format!("{v} * ia + ib"),
        _ => bail!("unknown op {op}"),
    })
}

fn op_is_int(op: &str) -> bool {
    op == "imad"
}

/// One thread, one dependent chain of 64 ops per loop iteration.
fn latency_wgsl(op: &str) -> Result<String> {
    let int = op_is_int(op);
    let line = format!("        x = {};\n", op_expr(op, "x")?);
    let body = line.repeat(64);
    let init = if int {
        "var x = p[5];"
    } else {
        "var x = bitcast<f32>(p[5]);"
    };
    let out = if int { "x" } else { "bitcast<u32>(x)" };
    Ok(format!(
        r#"{HEADER}
@compute @workgroup_size(1)
fn main() {{
    let n = p[0];
    let a = bitcast<f32>(p[3]);
    let b = bitcast<f32>(p[4]);
    let ia = p[3];
    let ib = p[4];
    {init}
    for (var k = 0u; k < n; k++) {{
{body}    }}
    o[0] = {out};
}}
"#
    ))
}

/// Eight independent chains per thread, 64 ops per loop iteration.
fn throughput_wgsl(op: &str) -> Result<String> {
    let int = op_is_int(op);
    let mut body = String::new();
    for _ in 0..8 {
        for c in 0..8 {
            body += &format!("        x{c} = {};\n", op_expr(op, &format!("x{c}"))?);
        }
    }
    let mut init = String::new();
    for c in 0..8 {
        if int {
            init += &format!("    var x{c} = gid.x * 2654435761u + {c}u;\n");
        } else {
            init += &format!("    var x{c} = f32(gid.x & 1023u) * 1e-4 + 1.0 + f32({c}) * 0.1;\n");
        }
    }
    let sum = (0..8)
        .map(|c| format!("x{c}"))
        .collect::<Vec<_>>()
        .join(" + ");
    let (test, out) = if int {
        ("s == 0x9e3779b9u", "s")
    } else {
        ("s == 1234.5", "bitcast<u32>(s)")
    };
    Ok(format!(
        r#"{HEADER}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let n = p[0];
    let a = bitcast<f32>(p[3]);
    let b = bitcast<f32>(p[4]);
    let ia = p[3];
    let ib = p[4];
{init}    for (var k = 0u; k < n; k++) {{
{body}    }}
    let s = {sum};
    if ({test}) {{ o[0] = {out}; }}
}}
"#
    ))
}

/// Workgroup of 32 lanes; `vec2<f32>` node arrays laid out [index][lane] as
/// in `shaders/physics_creature.wgsl`. Each loop iteration loads (or stores)
/// `l` rows starting at a base that cycles through `n` rows.
fn shared_bw_wgsl(n: u32, l: u32, store: bool) -> String {
    let rows = n + l;
    let mut body = String::new();
    if store {
        body += "        let v = bitcast<vec2<f32>>(vec2<u32>(k, k ^ li));\n";
        for c in 0..l {
            body += &format!("        pos[base + {c}u][li] = v;\n");
        }
    } else {
        for c in (0..l).step_by(2) {
            body += &format!(
                "        acc = acc ^ bitcast<vec2<u32>>(pos[base + {c}u][li]) ^ bitcast<vec2<u32>>(pos[base + {}u][li]);\n",
                c + 1
            );
        }
    }
    let after = if store {
        format!(
            "    workgroupBarrier();\n    acc = bitcast<vec2<u32>>(pos[li % {rows}u][(li * 7u) % 32u]);\n"
        )
    } else {
        String::new()
    };
    format!(
        r#"{HEADER}
var<workgroup> pos: array<array<vec2<f32>, 32>, {rows}>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) li: u32) {{
    for (var i = 0u; i < {rows}u; i++) {{ pos[i][li] = vec2<f32>(f32(i), f32(li)); }}
    workgroupBarrier();
    let iters = p[0];
    var acc = vec2<u32>(0u);
    var base = 0u;
    for (var k = 0u; k < iters; k++) {{
{body}        base = (base + 1u) & {mask}u;
    }}
{after}    if (acc.x == 0x9e3779b9u) {{ o[li] = acc.y; }}
}}
"#,
        mask = n - 1
    )
}

/// Workgroup of 32 lanes holding `words` u32 of workgroup memory and `live`
/// f32 values that stay live across the loop (to raise the register count);
/// lane 0 walks a pointer chain in L2, so a workgroup takes a fixed time no
/// matter how many others share its SM, until the SM runs out of room.
fn occupancy_wgsl(words: u32, live: u32) -> String {
    let hops = "            j = c[j];\n".repeat(16);
    let words = words.max(32);
    let mut init = String::new();
    let mut update = String::new();
    let mut sum = String::from("0.0");
    for v in 0..live {
        init += &format!("        var v{v} = f32(li + {v}u) * 0.37;\n");
        update += &format!("            v{v} = fma(v{v}, fj, {}.0);\n", v + 1);
        sum += &format!(" + v{v}");
    }
    format!(
        r#"{HEADER}
@group(0) @binding(1) var<storage, read_write> c: array<u32>;
var<workgroup> pad: array<u32, {words}>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    pad[(li * 37u + p[5]) % {words}u] = li;
    workgroupBarrier();
    if (li == 0u) {{
        var j = ((wid.x * 97u) % p[1]) * p[4];
        let n = p[0];
{init}        for (var k = 0u; k < n; k++) {{
{hops}            let fj = f32(j & 1u) * 1e-9 + 0.5;
{update}        }}
        let s = {sum};
        if (j == 0xffffffffu || s == 1234.5) {{ o[wid.x % 1024u] = j + pad[(li * 13u) % {words}u] + u32(s); }}
    }}
}}
"#
    )
}

// ---------------------------------------------------------------- tests

/// A random single cycle over `m` slots (Sattolo), or the linear cycle.
fn cycle(m: usize, random: bool, seed: u64) -> Vec<u32> {
    if !random {
        return (0..m).map(|i| ((i + 1) % m) as u32).collect();
    }
    let mut a: Vec<u32> = (0..m as u32).collect();
    let mut s = seed | 1;
    for i in (1..m).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s % i as u64) as usize;
        a.swap(i, j);
    }
    a
}

struct Chain {
    buf: Buf,
    next: Buf,
    scatter: vk::Pipeline,
    set: vk::DescriptorSet,
}

impl Chain {
    fn new(gpu: &Gpu, bytes: u64, max_slots: u64) -> Result<Self> {
        let buf = gpu.buffer(bytes, Mem::Device)?;
        gpu.randomize(&buf, 11)?;
        let next = gpu.buffer(max_slots * 4, Mem::Host)?;
        let scatter = gpu.pipeline("scatter", &scatter_wgsl())?;
        let set = gpu.set([&gpu.params, &buf, &next, &gpu.out])?;
        Ok(Self {
            buf,
            next,
            scatter,
            set,
        })
    }

    /// Writes a cycle over `region` bytes with one element every `stride` bytes.
    fn build(&self, gpu: &Gpu, region: u64, stride: u64, random: bool) -> Result<u32> {
        let m = (region / stride) as usize;
        ensure!(m >= 2 && region <= self.buf.size && (m as u64) * 4 <= self.next.size);
        let next = cycle(m, random, 0x2545_f491_4f6c_dd1d ^ region);
        unsafe { std::ptr::copy_nonoverlapping(next.as_ptr().cast::<u8>(), self.next.ptr, m * 4) };
        let groups = (m as u32).div_ceil(256);
        let (gx, gy) = if groups > 32768 {
            (32768, groups.div_ceil(32768))
        } else {
            (groups, 1)
        };
        let s = (stride / 4) as u32;
        gpu.run(self.scatter, self.set, [gx, gy, 1], &[m as u32, s])?;
        Ok(s)
    }
}

fn bandwidth(gpu: &Gpu, sampler: &Sampler, args: &Args) -> Result<()> {
    let copy = match args.get("mode").unwrap_or("read") {
        "read" => false,
        "copy" => true,
        m => bail!("unknown mode {m}"),
    };
    let size = parse_size(args.get("size").unwrap_or("8m"))?;
    let wg: u32 = args.num("wg", 256)?;
    let groups: u32 = args.num("groups", gpu.sm_count * (1536 / wg).min(24))?;
    let reps: usize = args.num("reps", 5)?;
    let target: f64 = args.num("target-ms", 120.0)?;
    // `size` is the whole footprint: one buffer to read, or two for a copy.
    let each = if copy { size / 2 } else { size };
    let a = gpu.buffer(each, Mem::Device)?;
    let b = gpu.buffer(if copy { each } else { 256 }, Mem::Device)?;
    // `--data const` fills with one repeated word, to show the compression effect.
    let constant = args.get("data") == Some("const");
    if constant {
        gpu.fill(&a, 0x1234_5678)?;
        gpu.fill(&b, 0)?;
    } else {
        gpu.randomize(&a, 1)?;
        gpu.randomize(&b, 2)?;
    }
    // `--coherent 1` serves the reads from L2 instead of L1.
    let coherent = args.num::<u32>("coherent", 0)? == 1;
    let n = (each / 16) as u32;
    let label = format!(
        "{} {} MiB {} data{} wg {wg} groups {groups}",
        if copy { "copy" } else { "read" },
        size >> 20,
        if constant { "constant" } else { "hashed" },
        if coherent { " coherent loads" } else { "" },
    );
    let pipeline = gpu.pipeline_with(
        &label,
        &stream_wgsl(wg, copy, coherent),
        if coherent { &[1] } else { &[] },
    )?;
    let set = gpu.set([&gpu.params, &a, &b, &gpu.out])?;
    let params = |passes: u32| vec![n, passes];
    let passes = gpu.calibrate(pipeline, set, [groups, 1, 1], &params, 4, target)?;
    let r = gpu.measure(
        sampler,
        pipeline,
        set,
        [groups, 1, 1],
        &params(passes),
        reps,
        Duration::from_millis(1500),
    )?;
    let bytes = n as f64 * 16.0 * passes as f64 * if copy { 2.0 } else { 1.0 };
    let sms = gpu.sm_count as f64;
    report(
        &format!("bandwidth {label} passes {passes}"),
        &r,
        &[
            ("GB/s", &|r: &Rep| bytes / r.ns),
            ("B/cycle GPU", &|r: &Rep| bytes / (r.ns * 1e-9 * hz(r))),
            ("B/cycle/SM", &|r: &Rep| bytes / (r.ns * 1e-9 * hz(r) * sms)),
        ],
    );
    gpu.free_set(set);
    gpu.free(a);
    gpu.free(b);
    Ok(())
}

fn chase(gpu: &Gpu, sampler: &Sampler, args: &Args) -> Result<()> {
    let stride = parse_size(args.get("stride").unwrap_or("128"))?;
    let random = match args.get("order").unwrap_or("random") {
        "random" => true,
        "linear" => false,
        o => bail!("unknown order {o}"),
    };
    let reps: usize = args.num("reps", 5)?;
    let target: f64 = args.num("target-ms", 100.0)?;
    let regions: Vec<u64> = args
        .list("regions", "16k,64k,1m,8m,16m,24m,64m,1g")
        .iter()
        .map(|s| parse_size(s))
        .collect::<Result<_>>()?;
    let max = *regions.iter().max().context("no regions")?;
    let chain = Chain::new(gpu, max, max / stride)?;
    let pipeline = gpu.pipeline("chase", &chase_wgsl())?;
    let set = gpu.set([&gpu.params, &chain.buf, &gpu.out, &gpu.out])?;
    for region in regions {
        let s = chain.build(gpu, region, stride, random)?;
        let params = |iters: u32| vec![iters, s, 0];
        let iters = gpu.calibrate(pipeline, set, [1, 1, 1], &params, 256, target)?;
        let r = gpu.measure(
            sampler,
            pipeline,
            set,
            [1, 1, 1],
            &params(iters),
            reps,
            Duration::from_millis(600),
        )?;
        let hops = iters as f64 * 16.0;
        report(
            &format!(
                "chase {} region {} KiB stride {stride} B hops {hops}",
                if random { "random" } else { "linear" },
                region >> 10
            ),
            &r,
            &[
                ("ns/hop", &|r: &Rep| r.ns / hops),
                ("cycles/hop", &|r: &Rep| r.ns * 1e-9 * hz(r) / hops),
            ],
        );
    }
    gpu.free_set(set);
    Ok(())
}

fn shared_latency(gpu: &Gpu, sampler: &Sampler, args: &Args) -> Result<()> {
    let reps: usize = args.num("reps", 5)?;
    let target: f64 = args.num("target-ms", 100.0)?;
    let next = gpu.buffer(4096, Mem::Host)?;
    let perm = cycle(1024, true, 7);
    unsafe { std::ptr::copy_nonoverlapping(perm.as_ptr().cast::<u8>(), next.ptr, 4096) };
    let pipeline = gpu.pipeline("shared chase", &shared_chase_wgsl())?;
    let set = gpu.set([&gpu.params, &gpu.out, &next, &gpu.out])?;
    let params = |iters: u32| vec![iters, 0, 0];
    let iters = gpu.calibrate(pipeline, set, [1, 1, 1], &params, 1024, target)?;
    let r = gpu.measure(
        sampler,
        pipeline,
        set,
        [1, 1, 1],
        &params(iters),
        reps,
        Duration::from_millis(600),
    )?;
    let hops = iters as f64 * 16.0;
    report(
        &format!("shared-latency hops {hops}"),
        &r,
        &[
            ("ns/hop", &|r: &Rep| r.ns / hops),
            ("cycles/hop", &|r: &Rep| r.ns * 1e-9 * hz(r) / hops),
        ],
    );
    gpu.free_set(set);
    gpu.free(next);
    Ok(())
}

fn op_params(op: &str, iters: u32) -> Vec<u32> {
    let (a, b, x): (f32, f32, f32) = match op {
        "fma" => (0.999, 0.001, 1.5),
        "fadd" => (1e-7, 0.0, 1.0),
        "div" => (1.0, 2.0, 3.0),
        _ => (0.0, 0.0, 2.0),
    };
    if op_is_int(op) {
        vec![iters, 0, 0, 1_664_525, 1_013_904_223, 12345]
    } else {
        vec![iters, 0, 0, a.to_bits(), b.to_bits(), x.to_bits()]
    }
}

fn alu_latency(gpu: &Gpu, sampler: &Sampler, args: &Args) -> Result<()> {
    let reps: usize = args.num("reps", 5)?;
    let target: f64 = args.num("target-ms", 100.0)?;
    let set = gpu.set([&gpu.params, &gpu.out, &gpu.out, &gpu.out])?;
    for op in args.list("ops", "fma,fadd,imad,sqrt,rsqrt,div") {
        let pipeline = gpu.pipeline(&format!("latency {op}"), &latency_wgsl(&op)?)?;
        let params = |iters: u32| op_params(&op, iters);
        let iters = gpu.calibrate(pipeline, set, [1, 1, 1], &params, 1024, target)?;
        let r = gpu.measure(
            sampler,
            pipeline,
            set,
            [1, 1, 1],
            &params(iters),
            reps,
            Duration::from_millis(600),
        )?;
        let ops = iters as f64 * 64.0;
        report(
            &format!("alu-latency {op} ops {ops}"),
            &r,
            &[
                ("ns/op", &|r: &Rep| r.ns / ops),
                ("cycles/op", &|r: &Rep| r.ns * 1e-9 * hz(r) / ops),
            ],
        );
    }
    gpu.free_set(set);
    Ok(())
}

fn alu_throughput(gpu: &Gpu, sampler: &Sampler, args: &Args) -> Result<()> {
    let reps: usize = args.num("reps", 5)?;
    let target: f64 = args.num("target-ms", 120.0)?;
    let groups: u32 = args.num("groups", gpu.sm_count * 6 * 8)?;
    let set = gpu.set([&gpu.params, &gpu.out, &gpu.out, &gpu.out])?;
    let sms = gpu.sm_count as f64;
    for op in args.list("ops", "fma,imad,sqrt,rsqrt,div") {
        let pipeline = gpu.pipeline(&format!("throughput {op}"), &throughput_wgsl(&op)?)?;
        let params = |iters: u32| op_params(&op, iters);
        let iters = gpu.calibrate(pipeline, set, [groups, 1, 1], &params, 16, target)?;
        let r = gpu.measure(
            sampler,
            pipeline,
            set,
            [groups, 1, 1],
            &params(iters),
            reps,
            Duration::from_millis(1500),
        )?;
        let ops = groups as f64 * 256.0 * iters as f64 * 64.0;
        report(
            &format!("alu-throughput {op} groups {groups} iters {iters}"),
            &r,
            &[
                ("Gop/s", &|r: &Rep| ops / r.ns),
                ("lane-ops/cycle/SM", &|r: &Rep| {
                    ops / (r.ns * 1e-9 * hz(r) * sms)
                }),
                ("warp-inst/cycle/SM", &|r: &Rep| {
                    ops / 32.0 / (r.ns * 1e-9 * hz(r) * sms)
                }),
            ],
        );
    }
    gpu.free_set(set);
    Ok(())
}

fn shared_bw(gpu: &Gpu, sampler: &Sampler, args: &Args) -> Result<()> {
    let n: u32 = args.num("n", 8)?;
    let l: u32 = args.num("l", 16)?;
    ensure!(n.is_power_of_two() && l.is_multiple_of(2));
    let store = args.num::<u32>("store", 0)? == 1;
    let reps: usize = args.num("reps", 5)?;
    let target: f64 = args.num("target-ms", 120.0)?;
    let bytes_wg = (n + l) * 32 * 8;
    let label = format!(
        "shared-bw {} rows {} ({} B/workgroup) {} per iteration",
        if store { "store" } else { "load" },
        n + l,
        bytes_wg,
        l
    );
    let pipeline = gpu.pipeline(&label, &shared_bw_wgsl(n, l, store))?;
    let set = gpu.set([&gpu.params, &gpu.out, &gpu.out, &gpu.out])?;
    let sms = gpu.sm_count as f64;
    let mut ks: Vec<u32> = args
        .list("k", "1,2,4,8,12,16,20,24")
        .iter()
        .map(|s| s.parse().context("--k"))
        .collect::<Result<_>>()?;
    if args.num::<u32>("saturate", 1)? == 1 {
        // Many waves of workgroups: whatever fits on each SM stays busy.
        ks.push(24 * 16);
    }
    for k in ks {
        let groups = gpu.sm_count * k;
        let params = |iters: u32| vec![iters];
        let iters = gpu.calibrate(pipeline, set, [groups, 1, 1], &params, 64, target)?;
        let r = gpu.measure(
            sampler,
            pipeline,
            set,
            [groups, 1, 1],
            &params(iters),
            reps,
            Duration::from_millis(800),
        )?;
        let bytes = groups as f64 * iters as f64 * l as f64 * 256.0;
        report(
            &format!("{label} | workgroups per SM {k} (groups {groups}) iters {iters}"),
            &r,
            &[
                ("B/cycle/SM", &|r: &Rep| bytes / (r.ns * 1e-9 * hz(r) * sms)),
                ("GB/s", &|r: &Rep| bytes / r.ns),
                ("warp-accesses/cycle/SM", &|r: &Rep| {
                    bytes / 256.0 / (r.ns * 1e-9 * hz(r) * sms)
                }),
            ],
        );
    }
    gpu.free_set(set);
    Ok(())
}

fn occupancy(gpu: &Gpu, sampler: &Sampler, args: &Args) -> Result<()> {
    let reps: usize = args.num("reps", 3)?;
    let kmax: u32 = args.num("kmax", 30)?;
    let region = 4u64 << 20;
    let stride = 128u64;
    let chain = Chain::new(gpu, region, region / stride)?;
    let s = chain.build(gpu, region, stride, true)?;
    let m = (region / stride) as u32;
    // Each case is shared-bytes:live-values, for example 6144:0 or 0:120.
    for case in args.list("cases", "0:0,6144:0,6272:0") {
        let (shared, live) = case.split_once(':').context("--cases shared:live")?;
        let bytes: u32 = shared.parse().context("--cases shared")?;
        let live: u32 = live.parse().context("--cases live")?;
        let words = (bytes / 4).max(32);
        let label = format!("occupancy probe {} B shared, {live} live values", words * 4);
        let pipeline = gpu.pipeline(&label, &occupancy_wgsl(words, live))?;
        let set = gpu.set([&gpu.params, &chain.buf, &gpu.out, &gpu.out])?;
        let params = |iters: u32| vec![iters, m, 0, 0, s, 0];
        let iters = gpu.calibrate(pipeline, set, [gpu.sm_count, 1, 1], &params, 64, 40.0)?;
        // The first workgroup that does not fit starts a second wave, so the
        // time jumps against the previous count. Comparing with the previous
        // count, not with k = 1, keeps a cold clock at k = 1 from hiding it.
        let mut previous = f64::NAN;
        let mut resident = 0;
        for k in 1..=kmax {
            let groups = gpu.sm_count * k;
            let r = gpu.measure(
                sampler,
                pipeline,
                set,
                [groups, 1, 1],
                &params(iters),
                reps,
                Duration::from_millis(100),
            )?;
            let t = median(&r.iter().map(|r| r.ns).collect::<Vec<_>>());
            let ratio = if k == 1 { 1.0 } else { t / previous };
            if ratio < 1.4 && resident == k - 1 {
                resident = k;
            }
            previous = t;
            println!(
                "occupancy {label}: workgroups per SM {k:2} time {:.2} ms, ratio to previous {ratio:.2}, SM MHz {:.0}",
                t * 1e-6,
                median(&r.iter().map(|r| r.s.sm).collect::<Vec<_>>())
            );
        }
        println!(
            "RESULT occupancy {label} workgroup 32 lanes | resident workgroups (warps) per SM {resident}"
        );
        gpu.free_set(set);
    }
    Ok(())
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let test = raw.first().cloned().unwrap_or_else(|| "info".into());
    let args = Args(raw);
    let gpu = Gpu::new()?;
    let sampler = Sampler::start()?;
    std::thread::sleep(Duration::from_millis(300));
    let started = Instant::now();
    match test.as_str() {
        "info" => {}
        // Builds every kernel without running it, for the driver statistics
        // and (with MICRO_DUMP_IR) the SASS.
        "compile" => {
            gpu.pipeline("chase", &chase_wgsl())?;
            gpu.pipeline("shared chase", &shared_chase_wgsl())?;
            gpu.pipeline_with("read coherent", &stream_wgsl(256, false, true), &[1])?;
            for op in ["fma", "fadd", "imad", "sqrt", "rsqrt", "div"] {
                gpu.pipeline(&format!("latency {op}"), &latency_wgsl(op)?)?;
                gpu.pipeline(&format!("throughput {op}"), &throughput_wgsl(op)?)?;
            }
            gpu.pipeline("shared-bw load 24 rows", &shared_bw_wgsl(8, 16, false))?;
            gpu.pipeline("shared-bw store 24 rows", &shared_bw_wgsl(8, 16, true))?;
        }
        "bandwidth" => bandwidth(&gpu, &sampler, &args)?,
        "chase" => chase(&gpu, &sampler, &args)?,
        "shared-latency" => shared_latency(&gpu, &sampler, &args)?,
        "alu-latency" => alu_latency(&gpu, &sampler, &args)?,
        "alu-throughput" => alu_throughput(&gpu, &sampler, &args)?,
        "shared-bw" => shared_bw(&gpu, &sampler, &args)?,
        "occupancy" => occupancy(&gpu, &sampler, &args)?,
        t => bail!("unknown test {t}"),
    }
    let all = sampler.all();
    let sm: Vec<f64> = all.iter().map(|s| s.sm).collect();
    println!(
        "clocks over the whole run ({} samples, {:.1} s): SM MHz median {:.0} min {:.0} max {:.0}; \
         power W max {:.1}; temp C max {:.0}",
        all.len(),
        started.elapsed().as_secs_f64(),
        median(&sm),
        sm.iter().copied().fold(f64::INFINITY, f64::min),
        sm.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        all.iter()
            .map(|s| s.power)
            .fold(f64::NEG_INFINITY, f64::max),
        all.iter().map(|s| s.temp).fold(f64::NEG_INFINITY, f64::max),
    );
    Ok(())
}
