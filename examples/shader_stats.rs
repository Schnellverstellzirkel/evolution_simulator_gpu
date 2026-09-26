//! Prints the driver's compiled-shader statistics (registers, spills, shared
//! memory) for each creature-kernel bucket via VK_KHR_pipeline_executable_properties.
//! Usage: cargo run --release --example shader_stats [device-substring]
use ash::vk;
use std::ffi::CStr;

fn spirv(source: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).expect("WGSL parse");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("WGSL validation");
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .expect("SPIR-V output")
}

fn main() {
    let wanted = std::env::args().nth(1).unwrap_or_else(|| "NVIDIA".into());
    unsafe {
        let entry = ash::Entry::load().expect("Vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .expect("instance");
        let physical = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .find(|&d| {
                let props = instance.get_physical_device_properties(d);
                CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .contains(&wanted)
            })
            .expect("device");
        let queue_family = instance
            .get_physical_device_queue_family_properties(physical)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let mut exec = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        let extensions = [ash::khr::pipeline_executable_properties::NAME.as_ptr()];
        let device = instance
            .create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&extensions)
                    .push_next(&mut exec),
                None,
            )
            .expect("device");
        let exec_fn = ash::khr::pipeline_executable_properties::Device::new(&instance, &device);
        let bindings: Vec<_> = (0..8)
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
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .unwrap();
        let layouts = [set_layout];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
            .unwrap();
        let workgroup = std::env::var("EVOLUTION_LANE_WG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        for &capacity in evolution_simulator::creature_kernel::CAPACITIES.iter() {
            let code = spirv(&evolution_simulator::creature_kernel::shader_source(
                capacity,
                workgroup,
                evolution_simulator::physics::Fidelity::standard(),
            ));
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .unwrap();
            let name = c"advance";
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(name);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)
                .flags(vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR);
            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .unwrap()[0];
            let pinfo = vk::PipelineInfoKHR::default().pipeline(pipeline);
            let executables = exec_fn.get_pipeline_executable_properties(&pinfo).unwrap();
            for index in 0..executables.len() as u32 {
                let einfo = vk::PipelineExecutableInfoKHR::default()
                    .pipeline(pipeline)
                    .executable_index(index);
                let stats = exec_fn.get_pipeline_executable_statistics(&einfo).unwrap();
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
                println!("capacity {capacity:2}: {}", text.join(", "));
            }
            device.destroy_pipeline(pipeline, None);
            device.destroy_shader_module(module, None);
        }
    }
}
