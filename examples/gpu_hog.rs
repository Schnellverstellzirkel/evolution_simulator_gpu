//! Holds device memory on a Vulkan GPU for a while, to test how the game
//! rides out a GPU whose memory another process holds.
//!
//! Usage: cargo run --release --example gpu_hog -- [MiB] [seconds] [device name]
//! Defaults: 6000 MiB for 60 s on the first device whose name contains "RTX".
use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mib: u64 = args.get(1).map_or(Ok(6000), |v| v.parse())?;
    let seconds: f64 = args.get(2).map_or(Ok(60.0), |v| v.parse())?;
    let wanted = args.get(3).map_or("rtx", String::as_str).to_lowercase();
    unsafe {
        let entry = ash::Entry::load()?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )?;
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
            .context("no matching Vulkan device")?;
        let family = instance
            .get_physical_device_queue_family_properties(physical)
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .context("no compute queue")? as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let device = instance.create_device(
            physical,
            &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
            None,
        )?;
        let memory = instance.get_physical_device_memory_properties(physical);
        let index = (0..memory.memory_type_count)
            .find(|&i| {
                memory.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .context("no device-local memory")?;
        let chunk = 256u64 << 20;
        let mut held = Vec::new();
        while (held.len() as u64) * chunk < mib << 20 {
            match device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(chunk)
                    .memory_type_index(index),
                None,
            ) {
                Ok(block) => held.push(block),
                Err(error) => {
                    eprintln!("gpu_hog: stopped at {} MiB: {error}", held.len() * 256);
                    break;
                }
            }
        }
        println!("gpu_hog: holding {} MiB for {seconds} s", held.len() * 256);
        std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
        for block in held {
            device.free_memory(block, None);
        }
        device.destroy_device(None);
        instance.destroy_instance(None);
        println!("gpu_hog: released");
    }
    Ok(())
}
