# Validation results

Measurements below were captured on the local NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB), Ubuntu 24.04 Wayland, with Rust release builds and the simulator's throughput mode. Each creature ran the default 15-second trial. These are workload measurements, not fixed hardware guarantees.

## Generation throughput

| Population | GPU evaluation / generation | Full generation, including ranking and reproduction | Average evaluations/s | Failed trials |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0.34 s | 0.37 s | 294,278 | 0 |
| 1,000,000 | 2.56–2.82 s across a 10-generation run | 2.83–3.11 s | 355,000–383,000 | 0 |
| 3,000,000 | 7.27–7.83 s across three generations | 8.04–8.63 s | 383,000–413,000 | 0 |

At 100,000 creatures, the CPU reference took 5.11 seconds to evaluate the same population and configuration: the observed GPU evaluation was 15× faster. CPU and GPU timings come from the same benchmark executable and machine.

After increasing the batch size and removing per-creature staging allocations, a fresh release run evaluated one million creatures in 2.63 seconds (381,000 evaluations/s) and three million in 6.13 seconds (489,000 evaluations/s). During those runs, `nvidia-smi dmon` reported 82–100% SM utilization; memory-controller utilization remained low because this workload is compute-bound. The three-million run used 720 MiB of process RAM and 35 MiB of tracked GPU buffers.

Ten successive one-million-creature generations completed without failed trials or accumulating memory growth. `/usr/bin/time -v` reported a peak RSS of 1,068,196 KiB (about 1.02 GiB). The three-million-creature, three-generation run peaked at 2,702,408 KiB (about 2.58 GiB).

## Interactive display

The native Wayland screenshot capture rendered the dashboard during a one-million-creature run. Across 240 sampled frames, p95 frame time was 10.36 ms, below the 33 ms target. Closing the window after capture exited cleanly after the worker and GPU work shut down.

The updated light-theme population view rendered during a one-million-creature evaluation at 14.92 ms p95 across 240 frames, also below the 33 ms target.

## Automated checks

- Release test suite: 11 tests passed, including GPU and CPU trajectory agreement, partial GPU workgroups, 3/8/9/17/33/64-node GPU buckets, collider contacts, deterministic evolution, mutation limits, checkpoint resumption, corrupt-checkpoint checksums, and history validation.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.
- Headless interruption: Ctrl+C saved a partial first-generation checkpoint after 26.2% of a one-million-creature generation; resuming that file continued from the saved evaluation count and completed the generation.

Full benchmark records are in [`benchmark-comparison.csv`](benchmark-comparison.csv), [`benchmark-scale.csv`](benchmark-scale.csv), and [`benchmark-soak.csv`](benchmark-soak.csv).
