# Validation results

Throughput figures in the historical sections below predate rigid bones and do not describe the current physics. This change was checked with low-impact correctness tests; no throughput profile was run while the CPU/GPU were busy.

## Rigid-bone correctness audit (2026-09-24)

A 64-node chain with 63 muscles at maximum stiffness ran for 560 CPU steps with gravity and ground contact. The test checks every bone after every step and requires its length to stay within 0.1 mm of its rest length, with all nodes above the ground. The GPU smoke test checks the same 64-node, endpoint-anchored morphology after 1, 201, and 320 steps. The normal CPU/GPU trajectory test also compares positions and velocities through ground contact. A checkpoint migration test reverses a saved skeleton's bone order, reloads it, and verifies the normalized skeleton preserves every muscle attachment point. These checks use the parent-first exact reconstruction added after the eight mass-weighted projection passes; no throughput benchmark was run for that solver change.

The stability checks cover a 2 m/s muscle-target rate limit, a 5 m/s node-speed cap, and a 15 rad/s per-bone rotation limit. A collapsed zero-velocity chain is reconstructed to exact rest lengths without gaining speed, and the GPU smoke test checks the same case alongside CPU/GPU trajectory agreement. Fitness now tracks horizontal center-of-mass displacement, so a posture change alone cannot earn travel distance. The QD version is bumped so saved archives and scores are reevaluated under the changed physics. No throughput profile was run while the CPU/GPU were busy.

## Current native app measurement (2026-09-23)

The graphical Wayland app was run on the NVIDIA RTX 4060 with driver 580.173.02, a fixed seed, 18-second trials, and 30 complete generations. The native benchmark includes GPU evaluation, archive insertion, breeding, worker publication, and active UI rendering. These are sequential runs on the same machine; background desktop activity and GPU clocks can affect small differences.

| Population | Earlier 64-step run | Current app default | Observed speedup |
| ---: | ---: | ---: | ---: |
| 1,000 | 54.69 generations/s | 98.30 generations/s | 1.80× |
| 10,000 | 14.40 generations/s | 21.04 generations/s | 1.46× |
| 100,000 | 1.91 generations/s | 3.18 generations/s | 1.67× |

The current 30-generation runs spent 0.219/0.038/0.048 seconds in evaluation/archive/breeding at 1,000 creatures; 1.023/0.165/0.236 seconds at 10,000; and 7.550/0.900/0.980 seconds at 100,000. A three-generation run at one million creatures completed at 0.435 generations/s, spending 5.270/0.455/1.177 seconds in those stages. All population sizes use 4096-step dispatches by default. Exact 4-node buckets help across population sizes, while the 5-node bucket helps only at smaller populations. The shader keeps behavior metrics in registers and precomputes muscle reciprocals while packing each GPU batch.

On the one-million-creature run, parent planning took 0.151–0.184 s per generation, candidate emission took 0.182–0.222 s, and breeding finalization took about 0.019 s. GPU timestamps from a separate 10-generation run at 100,000 creatures attributed 1.719 s of 2.052 s evaluation time to shader execution: 0.656 s in the 4-node bucket, 0.966 s in the 8-node bucket, and 0.098 s in the 16-node bucket. Packing took 0.163 s. These timings identify where further work may help; profiling adds overhead.

An NVIDIA Nsight guidance review highlighted group barrier stalls as a possible limiter. NVIDIA’s [shader profiler guide](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html) describes barrier stalls as warps waiting for sibling warps and recommends checking whether each group synchronization is needed. An experimental shared muscle-target table reduced duplicate cosine work but performed much worse: 0.92 generations/s at 100,000 creatures, so that variant was removed. Raising the dispatch chunk from 1024 to 4096 steps reduced native 100,000-creature generation time by about 7.5%; a repeated 30-generation run measured 3.18 generations/s, versus 2.96 with 1024-step dispatches. At one million creatures, 4096-step dispatches measured 0.435 generations/s with 100,000-creature batches, versus 0.409 with the earlier default. A 250,000-creature batch did not improve that result. The separate compute device is the graphical default; the full 10× generation-throughput goal remains open.

### Paired kernel comparison at 100,000 creatures (2026-09-23)

Four 20-generation full-GUI runs on revision `5f0349c` used default/workgroup/workgroup/default order. Host load stayed between 1.87 and 2.77 on 16 logical CPUs; CPU pressure was zero during the comparison. GNOME Shell contributed a steady desktop GPU load, and each run recorded 98–100% peak SM use.

| Kernel mode | Generations/s | Evaluation seconds / 20 generations | Runs |
| --- | ---: | ---: | ---: |
| Default (serial private arrays for 4/5/8-node buckets) | 1.828–1.829 | 9.875–9.907 | 2 |
| Workgroup | 3.063–3.104 | 5.449–5.477 | 2 |

The workgroup path delivered 1.69× the throughput of the serial path at that revision. The serial and lane shaders use the earlier node-to-node genome layout, so the bone-physics update disables both paths; the current simulator uses the workgroup shader for every body size.

### Low-impact timestamp profile on the current default (2026-09-24)

A five-generation full-GUI diagnostic at 100,000 creatures ran pinned to CPUs 4–15 at nice priority 10. Before the run, CPU pressure was 0%, load average was 1.82, and the GPU showed 18% desktop utilization. The profiled run completed in 1.44 s; these measurements are diagnostic and should not be compared with unprofiled throughput runs.

| Measurement | Result |
| --- | ---: |
| Evaluation | 1.174 s |
| GPU shader timestamps | 1.027 s |
| Packing | 0.144 s |
| Upload/encode | 0.072 s |
| Bucket 4 / 8 / 16 shader time | 0.413 / 0.577 / 0.037 s |

The 4- and 8-node buckets account for about 96% of timestamped shader time. The reported 1.050 s readback duration is CPU wall time waiting for mapped results and overlaps GPU execution; do not add it to shader time. This points the next kernel experiment toward the 4- and 8-node dispatches. The short run was kept isolated because profiling drives the GPU to full utilization.

### Paired workgroup-size comparison at 100,000 creatures (2026-09-24)

Four five-generation full-GUI runs at revision `7798381` used default/32-lane/32-lane/default order. Each run was pinned to CPUs 4–15 at nice priority 10, with no timestamp instrumentation. Before the sequence, CPU pressure was 0%, load average was 0.92, and GPU desktop utilization was 19%.

| Workgroup size | Run 1 generations/s / evaluation s | Run 2 generations/s / evaluation s | Mean generations/s / evaluation s |
| --- | ---: | ---: | ---: |
| Default (64 lanes for large populations) | 3.477 / 1.170 | 3.492 / 1.169 | 3.485 / 1.170 |
| 32 lanes | 4.119 / 0.955 | 4.002 / 0.981 | 4.061 / 0.968 |

The 32-lane variant improved this short paired sample by 16.5% in full-generation throughput and reduced evaluation time by 17.3%. A follow-up 10-generation full-GUI run with 32 lanes as the default measured 4.539 generations/s (evaluation 1.681 s, archive 0.184 s, breeding 0.336 s). This confirms the default path; longer repeats are still needed for a stable estimate. `EVOLUTION_WORKGROUP64=1` remains available for comparisons.

Measurements below were captured on the local NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB), Ubuntu 24.04 Wayland, with Rust release builds and the simulator's throughput mode. Each creature ran the default 15-second trial. These are workload measurements, not fixed hardware guarantees.

These records predate the MAP-Elites archive and emitter loop. They document GPU simulation throughput and UI responsiveness; they are not performance measurements of the current archive insertion and offspring-generation work.

## Generation throughput

| Population | GPU evaluation / generation | Full generation with the earlier ranking/reproduction loop | Average evaluations/s | Failed trials |
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

- Release test suite: 10 standard tests and the separately invoked Vulkan GPU agreement test passed, covering partial GPU workgroups, 3/5/6/8/9/17/33/64-node GPU buckets, collider contacts, deterministic evolution, mutation limits, checkpoint resumption, corrupt-checkpoint checksums, and history validation.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.
- Headless interruption: Ctrl+C saved a partial first-generation checkpoint after 26.2% of a one-million-creature generation; resuming that file continued from the saved evaluation count and completed the generation.

Full benchmark records are in [`benchmark-comparison.csv`](benchmark-comparison.csv), [`benchmark-scale.csv`](benchmark-scale.csv), and [`benchmark-soak.csv`](benchmark-soak.csv).
