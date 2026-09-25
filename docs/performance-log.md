# Throughput optimization log (2026-09-25)

Goal: 2,000,000 fully evaluated creatures/s (18 s trials, 2,360 physics steps each) in the graphical game at 60 FPS.

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

## Open items

- Re-measure after the compositor move; tune unit sizes per device.
- CPU (AVX-512) evaluator.
- End-to-end overlap of archive/breeding with GPU work is limited by the generational algorithm.
