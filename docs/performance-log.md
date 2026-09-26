# Performance log

Current goal: 2,000,000 evaluated creatures/s with fixed 60-second trials, 3 million creatures per generation, and the graphical game at 60 FPS. The measurements below do not establish that goal.

## Bone-cap baseline, Windows headless (2026-09-26)

Seed 38, 100,000 candidates per generation, 20 generations, 60-second trials, existing 2 m bone cap. Hardware/software: RTX 4060 Laptop GPU, driver 610.47, Windows, Rust 1.98.1. Build: native release optimization, LTO disabled, 256 codegen units. Resources: primary GPU only, six CPU evaluation threads, two general workers, low priority. The Radeon did not evaluate creatures.

The 20 logged evaluation stages sum to **415.607 s** for two million candidate slots. Individual stages span 9.749–44.255 s, with the first at 44.255 s. These are headless evaluation-call timings, including the standard trial and fine perturbed check of each candidate. They exclude archive insertion, breeding, checkpoint writes, and rendering. The first call can include pipeline initialization. They are not end-to-end throughput, GUI FPS, or a paired performance comparison, and should not be compared directly with the older 18-second runs below.

The final archive has best distance 165.5846 m, 1,374 behavior cells, and QD score 23,060.27. The top-50 report has median total bone length 2.23 m, longest individual bone 1.81 m, and reported median slip per replay meter 0.89. This baseline supplies evidence for the next physics investigation; no contact-solver change or performance speedup is claimed. The updated slip diagnostic excludes unscored post-fall motion, so its ratios are not directly comparable to earlier reports.

Compact evidence: [generation timings and scores](results/2026-09-26-bone-cap-seed-38/generations.csv), [top-50 body and replay measurements](results/2026-09-26-bone-cap-seed-38/top-50.csv), and [metadata and limits](results/2026-09-26-bone-cap-seed-38/metadata.json). The [validation notes](validation.md) record the unresolved 111.2 m archive / 6.4 m replay outlier and final local check results. The checkpoint and machine-local paths are not committed.

## Historical throughput work (2026-09-25)

The remaining log preserves the earlier goal and workload: 18-second trials, then 2,360 physics steps per evaluation, before the current 60-second default and later physics fixes. Thread allocations, device selections, and performance figures below are historical records, not current workstation run instructions. In particular, Radeon compute is now disabled by default and must remain excluded on the owner's workstation.

## Machine

- Ryzen 7 7840HS (8 cores / 16 threads, Zen 4, AVX-512), 30 GiB RAM.
- RTX 4060 Laptop GPU (AD107, 8 GiB, 105 W limit, driver 580.173.02, ReBAR on).
- Radeon 780M iGPU (RADV). The laptop panel (eDP) is wired to the Radeon.
- Ubuntu 24.04, Wayland, 2880x1800 at 120 Hz.

## What the counters measure

- The perf overlay's `creatures/s` is `evaluated / evaluation_seconds` for the current generation: evaluation stage only.
- The native benchmark (`EVOLUTION_BENCH_GENERATIONS`) now reports evaluation-only and end-to-end creatures/s over the measured generations (after `EVOLUTION_BENCH_WARMUP` generations, default 1), frame-time percentiles over the same window, and control latency (time for a queued UI command to reach the worker).

## Workloads

- W1: fresh population of 100,000, seed 38.
- W3: `bench/w3-seed38-100k.evo`, the W1 population after 100 generations (mean 4.7 nodes, 3-9 nodes). GUI: `EVOLUTION_SMOKE_CHECKPOINT=bench/w3-seed38-100k.evo`.
- W4: W3 bodies grown to at least 16 nodes with the game's structural mutations (`eval-bench --grow 16`).

## Baseline (before)

W1 in the GUI, 100 measured generations, AC power not yet confirmed: evaluation 44.7k/s, end-to-end 42.4k/s, 4.0 FPS (frame p99 490 ms), control latency p99 30.9 s.
W3 in the GUI, 10 generations, AC: evaluation 44.4k/s, end-to-end 43.1k/s, 6.5 FPS, control latency p99 6.5 s.
W3 headless: 49.0k/s. W4 headless: 5.8k/s.

## Changes and results

1. One-creature-per-lane kernel (`shaders/physics_creature.wgsl`). Node state in workgroup memory as [node][lane]; muscles and bones in coalesced 32-creature tiles; each muscle evaluated once and scattered in genome order. Exact small node buckets (3-8, 12, 16, 24, 32, 48, 64). W3 headless 49.0k -> 135k/s; W4 5.8k -> 17.2k/s. Same equations; results differ from the old kernel only by FMA-contraction rounding amplified by chaotic contact. Statistical check: W3 mean fitness shift z = -0.72, W4 z = -1.38; the old kernel's own exact-vs-polynomial cosine switch moves W4 by z = -1.74.
2. Rejected kernel experiments: node constants in global memory (-34%), precomputed muscle weights (-8%), cached velocity-pass directions (-7%), per-muscle amplitude precompute (changed rounding and no gain).
3. Larger GPU submissions (100k instead of 16k creatures): +30-50% headless.
4. Raw Vulkan compute engine (`src/vk_engine.rs`, ash + naga). wgpu inserts a full barrier before every dispatch that reuses a read-write buffer, which serialized the node buckets whenever a trial was split into short dispatches (-22%). The engine records all buckets per step range with one barrier. 16-step dispatches now cost nothing: ~185-190k/s headless, bit-identical to the wgpu path.
5. UI: repaint every frame while evolving or playing (was every 100 ms); worker handles all queued commands each pass; asynchronous two-slot GPU submission. Control latency p99 30.9 s -> ~25 ms.
6. GPU contention: each displayed frame costs compute ~4 ms because GNOME composited on the RTX 4060 (udev rule `61-mutter-preferred-primary-gpu.rules`) and the game UI rendered there too. Idle GUI: 184k -> 133k/s headless. The user approved moving the desktop to the Radeon; the game now renders its UI on the compositor's GPU.
7. Multi-device scheduler (`src/scheduler.rs`): RTX 4060 + Radeon 780M (bodies up to 16 nodes on the Radeon; RADV compile time explodes above that). Radeon alone: 79k/s headless on W3.

8. Desktop moved to the Radeon (udev rule removed, reboot). With nothing but evolution on the RTX 4060, the GUI ran W3 at 160.5k evaluation / 145.3k end-to-end creatures/s at 119.8 FPS (frame p99 9.7 ms).
9. One-second GPU units (one submission per 100k generation) raised W3 in the GUI to 189.8k evaluation / 170.7k end-to-end over 150 generations, 119.8 FPS, frame p99 10.0 ms.
10. Multi-engine architecture (`src/engine.rs`, `src/scheduler.rs`, `src/cpu_engine.rs`, `src/simd.rs`): threaded GPU engines for the RTX 4060 and the Radeon 780M, and an AVX-512 CPU engine that simulates 16 same-plan creatures per vector (4.5k -> 33.5k creatures/s after replacing auto-vectorization with explicit AVX-512). Units are sized by each engine's measured rate, handed out in body-size order, and streamed to the engines while the next generation is bred. All engines match the original kernel within rounding noise on 30k W3 creatures (mean-fitness z: RTX -0.04, Radeon -1.06, CPU -0.39; the original kernel's own cosine switch: -0.94).
11. Division cleanup (mass shares, reciprocals, packed muscle amplitudes): RTX 205k -> 220k creatures/s on W3 headless.
12. Nsight Systems GPU metrics on W3: SMs active 99%, SM issue 58%, about 30% of warp slots occupied (about 120 registers per thread), DRAM about 2%. The kernel is instruction and occupancy bound.
13. Radeon compute starved the desktop in 1 s units (worst frame 592 ms at 1M). 0.05 s units on the Radeon keep the worst frame at 45 ms.
14. Default population raised to 1,000,000 (throughput mode). GUI end-to-end by population with all engines: 300k -> 266k/s, 1M -> 329-370k/s, 3M -> 341k/s.

## Sustained result, exact physics (2026-09-25)

1M creatures, all defaults, 60 measured generations (212.6 s) in the graphical game, AC power: end-to-end 282.2k creatures/s, evaluation-stage 409.8k/s, 116.5 FPS, frame p95 11.8 ms, p99 17.2 ms, max 129 ms, control latency p50 3.4 ms, p99 826 ms. Bodies grow over the run (generation time 2.8 -> 5.2 s) and the GPU reached 87 C at about 73 W.

## Reduced-work physics (user-approved direction)

The user allowed simulation changes that keep the game concept (natural selection of creatures learning to move). RTX throughput on W3 headless by physics rate and projection/velocity passes: 120 Hz 8/4: 235k; 60 Hz 8/4: 424k; 60 Hz 4/2: 524k; 60 Hz 2/2: 582k; 60 Hz 2/1: 613k.

## Open items

- Control latency at 1M: archive insertion and breeding block the worker thread for up to about 1 s.
- NPU: disabled in firmware; a fixed-point engine would be needed (no native fp32).
- End-to-end overlap of archive/breeding with GPU work is limited by the generational algorithm.
