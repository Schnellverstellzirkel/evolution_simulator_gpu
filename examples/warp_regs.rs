//! Registers, spills and stack of the CUDA kernels: the lane-group kernel
//! per lane class, and the per-lane stub (`shaders/lane_stub.cu`, see
//! `examples/lane_stub.rs`) at 16, 12 and 8 warps per SM. The ptxas numbers
//! come from NVRTC alone (no GPU time). With `lmem`, each cubin is also
//! loaded on the primary GPU and the driver's local memory per thread is
//! printed beside ptxas's stack frame and spills, so they are reported
//! separately. `EVOLUTION_WARP_*` settings apply.
//! Usage: warp_regs [sm_89] [lmem]
use evolution_simulator::{config::Config, cuda_engine, warp_kernel};

#[path = "lane_stub.rs"]
#[allow(dead_code)]
mod lane_stub;

fn main() -> anyhow::Result<()> {
    // NVRTC keeps ptxas results in NVIDIA's compute cache, which would hide
    // the report.
    unsafe { std::env::set_var("CUDA_CACHE_DISABLE", "1") };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arch = args
        .iter()
        .find(|a| a.starts_with("sm_"))
        .cloned()
        .unwrap_or_else(|| "sm_89".into());
    let cuda = if args.iter().any(|a| a == "lmem") {
        let cu = lane_stub::Cuda::load()?;
        cu.open()?;
        Some(cu)
    } else {
        None
    };
    let nvrtc = lane_stub::Nvrtc::load()?;
    // The driver's view of a compiled kernel, when `lmem` asked for it.
    let driver = |source: &str, file: &str, name: &str| -> anyhow::Result<String> {
        let Some(cu) = &cuda else { return Ok(String::new()) };
        let (cubin, _) = nvrtc.compile(source, file, &lane_stub::options(&arch))?;
        let (regs, local, shared) = cu.function_resources(cu.function(&cubin, name)?)?;
        Ok(format!(" | driver: {regs} registers, {local} B local memory per thread, {shared} B static shared per block"))
    };
    let cfg = Config::default();
    for class in warp_kernel::CLASSES {
        for record in [false, true] {
            let report = cuda_engine::compile_report(class, &cfg, record, &arch)?;
            let source = warp_kernel::cuda_source(class, warp_kernel::world_flags(&cfg), cfg.fidelity(), record);
            println!(
                "lane-group {class} lanes{}: {}{}",
                if record { " (recording)" } else { "" },
                report.replace('\n', " | "),
                driver(&source, "physics_creature.cu", "advance")?
            );
        }
    }
    for (warps, min_blocks) in [(16, 4), (12, 3), (8, 2)] {
        let setup = lane_stub::Setup {
            mpl: 16,
            nb: 3,
            substeps: 2,
            rounds: 3,
            block: 128,
            min_blocks,
        };
        let source = lane_stub::source(&setup.defines());
        let (_, log) = nvrtc.compile(&source, "lane_stub.cu", &lane_stub::options(&arch))?;
        println!(
            "per-lane stub, W = 2, {warps} warps per SM: {} | largest spill {} B{}",
            lane_stub::ptxas_summary(&log),
            lane_stub::max_spill(&log),
            driver(&source, "lane_stub.cu", "lane_stub")?
        );
    }
    Ok(())
}
