//! Registers, spills and stack of the lane-group CUDA kernel per lane class,
//! from NVRTC alone (no GPU time). `EVOLUTION_WARP_*` settings apply.
//! Usage: warp_regs [sm_89]
use evolution_simulator::{config::Config, cuda_engine};
fn main() -> anyhow::Result<()> {
    // NVRTC keeps ptxas results in NVIDIA's compute cache, which would hide
    // the report.
    unsafe { std::env::set_var("CUDA_CACHE_DISABLE", "1") };
    let arch = std::env::args().nth(1).unwrap_or_else(|| "sm_89".into());
    let cfg = Config::default();
    for class in evolution_simulator::warp_kernel::CLASSES {
        for record in [false, true] {
            let report = cuda_engine::compile_report(class, &cfg, record, &arch)?;
            println!(
                "{class} lanes{}: {}",
                if record { " (recording)" } else { "" },
                report.replace('\n', " | ")
            );
        }
    }
    Ok(())
}
