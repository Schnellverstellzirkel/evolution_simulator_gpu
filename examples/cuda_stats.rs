//! Prints what the CUDA compiler makes of the creature kernel
//! (`shaders/physics_creature.cu`) for each node capacity and register cap:
//! registers per thread, local memory (stack and spills), shared memory, and
//! resident warps per SM from the driver's occupancy calculator.
//!
//! Usage: cargo run --release --example cuda_stats [caps] [capacities] [fine]
//!
//! `caps` is a comma list of register caps, 0 for the compiler's choice
//! (default `0,128,96,80,64`); `capacities` a comma list of node capacities
//! (default all). A third argument `fine` compiles the fine-fidelity kernels.
//! Needs the CUDA driver and NVRTC (see docs/building.md). Compiling touches
//! the GPU only to load each module.
use evolution_simulator::{
    creature_kernel::CAPACITIES, cuda_engine::CudaEngine, physics::Fidelity,
};

fn list(arg: Option<String>, default: &[usize]) -> Vec<usize> {
    arg.map(|a| a.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let caps = list(args.next(), &[0, 128, 96, 80, 64]);
    let capacities = list(args.next(), &CAPACITIES);
    let fidelity = if args.next().as_deref() == Some("fine") {
        Fidelity::fine()
    } else {
        Fidelity::standard()
    };
    let engine = CudaEngine::open_for_stats("RTX 4060")?;
    println!(
        "{} ({} SMs), {fidelity:?}",
        engine.name,
        engine.multiprocessors()
    );
    println!(
        "| capacity | cap | registers | local bytes | spill stores | spill loads | block threads | shared bytes | warps per SM |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for &capacity in &capacities {
        for &cap in &caps {
            let stats = engine.kernel_stats(capacity, fidelity, (cap > 0).then_some(cap as u32))?;
            // "N bytes stack frame, N bytes spill stores, N bytes spill loads"
            let spill = |what: &str| -> String {
                stats
                    .log
                    .lines()
                    .find(|l| l.contains("spill stores"))
                    .and_then(|l| {
                        l.split(',')
                            .find(|part| part.contains(what))
                            .and_then(|part| {
                                part.split_whitespace().find_map(|w| w.parse::<u32>().ok())
                            })
                    })
                    .map_or_else(|| "?".into(), |n| n.to_string())
            };
            println!(
                "| {capacity} | {} | {} | {} | {} | {} | {} | {} | {} |",
                if cap == 0 {
                    "none".to_string()
                } else {
                    cap.to_string()
                },
                stats.registers,
                stats.local_bytes,
                spill("spill stores"),
                spill("spill loads"),
                stats.threads,
                stats.shared_bytes,
                stats.warps_per_sm
            );
        }
    }
    Ok(())
}
