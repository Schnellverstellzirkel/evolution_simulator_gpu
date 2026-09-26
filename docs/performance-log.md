# Performance log

Current goal: 2,000,000 evaluated creatures/s with fixed 60-second trials, 3 million creatures per generation, and the graphical game at 60 FPS. The measurements below do not establish that goal.

## Environment effect cost (2026-09-26)

Backlog item 46 asks whether each environment effect is cheap to leave on. `examples/effect_cost.rs` builds one deterministic random population (2048 bodies, seed 38, 2.0 s trials) and times CPU `cpu_engine::evaluate` on the calm world and on every level of every `environment::EFFECTS` entry, so a new effect is covered automatically. Each pass samples the calm world at its start, middle, and end. The table reports the best rate of eight passes and the median of the per-pass ratios to calm. Measured on the working tree at HEAD 144848a, which carried the uncommitted ground-roughness, slope, wind, and UI work.

Command:

    nice -n 15 env CARGO_BUILD_JOBS=4 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 cargo run --release --example effect_cost

Same workstation as the Machine section below: Ryzen 7 7840HS (8 cores / 16 threads, Zen 4, AVX-512), Ubuntu 24.04, normal thin-LTO release build. This example is CPU only; the RTX 4060 and the Radeon do no work.

| effect | level | world | creatures/s | % of calm | best m |
|---|---:|---|---:|---:|---:|
| calm | 0 | default world | 148207.1 | 97.1 | 0.77 |
| Ground | 0 | Flat | 161017.7 | 105.7 | 0.77 |
| Ground | 1 | Pebbles, 3 cm | 152155.2 | 96.2 | 0.79 |
| Ground | 2 | Rough, 8 cm | 155389.9 | 103.5 | 0.80 |
| Ground | 3 | Rocky, 15 cm | 155923.6 | 98.1 | 0.99 |
| Ground | 4 | Boulders, 25 cm | 156631.5 | 96.7 | 0.51 |
| Gravity | 0 | Earth | 159307.6 | 104.5 | 0.77 |
| Gravity | 1 | 1.5 g | 155305.9 | 104.7 | 0.34 |
| Gravity | 2 | 2 g | 157489.5 | 105.1 | 0.37 |
| Gravity | 3 | 3 g | 149829.2 | 101.3 | 0.70 |
| Air | 0 | Thin | 162336.1 | 101.4 | 0.77 |
| Air | 1 | Breezy | 159646.7 | 108.2 | 0.86 |
| Air | 2 | Thick | 158298.9 | 100.9 | 0.79 |
| Air | 3 | Syrup | 152289.5 | 97.9 | 0.66 |
| Grip | 0 | Sandpaper | 156064.7 | 102.6 | 0.79 |
| Grip | 1 | Grippy | 157245.5 | 98.4 | 0.77 |
| Grip | 2 | Firm | 155375.9 | 100.5 | 1.03 |
| Grip | 3 | Wet | 167344.1 | 95.0 | 0.83 |
| Grip | 4 | Ice | 155110.4 | 101.1 | 0.50 |
| Heat wave | 0 | Full | 158211.8 | 99.8 | 0.77 |
| Heat wave | 1 | Warm | 153933.9 | 100.0 | 0.76 |
| Heat wave | 2 | Hot | 153114.0 | 105.4 | 0.91 |
| Heat wave | 3 | Heat wave | 158269.8 | 104.2 | 0.82 |
| Drought | 0 | Normal | 157460.8 | 105.8 | 0.77 |
| Drought | 1 | Dry | 152531.9 | 102.1 | 0.77 |
| Drought | 2 | Parched | 157417.7 | 93.9 | 0.78 |
| Drought | 3 | Drought | 149504.5 | 100.0 | 0.79 |
| Slope | 0 | Flat | 155768.7 | 99.9 | 0.77 |
| Slope | 1 | 3% | 151740.2 | 99.5 | 1.09 |
| Slope | 2 | 8% | 150890.7 | 98.2 | 0.98 |
| Slope | 3 | 15% | 153481.9 | 94.9 | 0.72 |
| Slope | 4 | 25% | 146086.0 | 94.4 | 0.75 |
| Wind | 0 | Calm | 153386.3 | 100.1 | 0.77 |
| Wind | 1 | Breeze | 152343.3 | 96.8 | 0.73 |
| Wind | 2 | Strong | 159381.2 | 102.5 | 0.29 |
| Wind | 3 | Gale | 154347.1 | 102.2 | 0.12 |

A second identical run put every row between 93.1% and 104.7% of calm, with Drought level 0 the largest row-to-row change (105.8% then 93.1%, 12.7 points). Rows whose configuration equals calm (Ground Flat, Gravity Earth, Air Thin, Grip Grippy, Heat Full, Drought Normal, Slope Flat, Wind Calm) moved between 93.1% and 105.8% across the two runs, which is the noise floor while other builds share this machine. Absolute creatures/s stayed within 10% for every row.

No environment effect costs more than about 8% extra CPU evaluation time (the worst row is Air/Breezy at 108.2% of calm, barely above the noise floor), so each one is cheap to leave on. This is CPU evaluation cost only; the GPU kernel is unchanged because the effect values travel in its existing uniform buffer.

Addendum: the Mud and Gaps effects added afterwards were measured the same way on the VERSION 23 tree. Both stay inside the noise floor (Mud 102.7 to 106.6% of calm, Gaps 95.9 to 99.2% of calm against a calm row at 103.4%), so the conclusion is unchanged.

| effect | level | world | creatures/s | % of calm | best m |
|---|---:|---|---:|---:|---:|
| calm | 0 | default world | 149791.6 | 103.4 | 0.77 |
| Mud | 1 | Damp | 159457.3 | 103.8 | 0.73 |
| Mud | 2 | Muddy | 150287.5 | 102.7 | 0.85 |
| Mud | 3 | Deep mud | 157463.0 | 106.6 | 0.69 |
| Gaps | 1 | Narrow | 147158.9 | 95.9 | 0.77 |
| Gaps | 2 | Wide | 147171.5 | 96.9 | 0.77 |
| Gaps | 3 | Chasms | 144157.9 | 99.2 | 0.77 |

## Historical version-16 bone-cap baseline, Windows headless (2026-09-26)

This run uses parent physics `bd41746` (QD version 16), with foundation changes captured in `250ad82`. It predates merged revision `abb00cb` and its version-19 friction, planted-foot cap, head-shaking, and CPU archive checks. These timings do not measure the merged code; loading the checkpoint under version 19 invalidates its archive.

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
