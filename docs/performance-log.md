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

Addendum: the Mud, Gaps, Hurdles and Earthquake effects added afterwards were measured the same way on the VERSION 24 tree. Every level stays within about 5% of the calm row (calm itself measured 98.6 to 103.4% across these runs), so they sit inside the run-to-run noise and the conclusion is unchanged. Selected worst levels:

| effect | level | world | creatures/s | % of calm | best m |
|---|---:|---|---:|---:|---:|
| calm | 0 | default world | 149791.6 | 103.4 | 0.77 |
| Mud | 1 | Damp | 159457.3 | 103.8 | 0.73 |
| Mud | 2 | Muddy | 150287.5 | 102.7 | 0.85 |
| Mud | 3 | Deep mud | 157463.0 | 106.6 | 0.69 |
| Gaps | 1 | Narrow | 147158.9 | 95.9 | 0.77 |
| Gaps | 2 | Wide | 147171.5 | 96.9 | 0.77 |
| Gaps | 3 | Chasms | 144157.9 | 99.2 | 0.77 |
| Hurdles | 3 | Walls | 144816.1 | 95.7 | 0.69 |
| Earthquake | 3 | Big one | 145242.9 | 97.4 | 0.67 |

## Whole-group early exit in the CPU engine (2026-09-26)

Backlog item 48. `EVOLUTION_EARLY_EXIT=1` stops a 16-lane SIMD group as soon
as every real lane in it has a failed node or a recorded fall
(`fall_time > 0`); the padding lanes that repeat a real creature do not count.
The stop happens at the end of the tick, after that tick's totals, and a
recorded trial (`replay`) never stops early, so the replay keeps every frame.
The flag is diagnostic and off by default.

The exit is **score-preserving but not descriptor-preserving**, so it cannot
be the default yet. A probe on a 4096-body first-generation population
(seed 38, 3 s trials, `n` = 520 fallen and 3576 upright lanes) found that
`fitness`, `fall_time` and `head_shake` are bit-identical for every lane, and
every field of every upright lane is bit-identical. Descriptor fields of
fallen lanes differ when their group exits: `ground_contact` changed for 107
of the 520 fallen lanes (max difference 784 contact-steps), `height_sum` for
107 (max 80.7 m), `gait_turns` for 61, and the contact/lift/ground bitsets for
25 to 96, because the default engine keeps accumulating those totals while the
limp body lies there. This is the obstacle `docs/physics-audit-2026-09-26.md`
records: freeze terminal results in both engines, update the `to_metrics`
normalization and archive versioning first, then the exit can default on. The
three cases a regression pins are `tests/early_exit.rs`: a group where every
lane falls exits and keeps every score, a group with one live lane never exits
and every field is unchanged, and a partial group exits on its real lanes.

Workload: `examples/early_exit_cost.rs`, the same first-generation random
population as `first_generation` (20,000 bodies, seed 38), alternating paired
passes with the flag on and off on the calm default world, then on a harsh
world (3 g, muscle energy 0.35). The example prints creatures/s, the paired
ratio, the share of groups that exited, the share of fallen lanes and the
share of configured physics steps the exited groups skipped. The step share is
deterministic; the wall rates move with the load on this shared machine.

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
      cargo run --release --example early_exit_cost -- 20000 20 6
    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
      cargo run --release --example early_exit_cost -- 20000 60 4

| world | trial | groups exited | lanes fallen | steps skipped | median paired ratio | best on / best off |
|---|---:|---:|---:|---:|---:|---|
| calm, 20,000 bodies | 20 s | 12.8% | 13.7% | 11.0% | 1.14, 1.12 (two runs) | 46,536 / 43,676 and 62,425 / 58,658 |
| calm, 20,000 bodies | 60 s | 13.4% | 13.9% | 12.4% | 1.27 | 17,856 / 14,519 |
| harsh 3 g / 0.35 energy | 60 s | 7.2% | 11.2% | 6.8% | 1.15 | 15,368 / 13,120 |

The calm first-generation population falls into two clear kinds of group,
because a group shares one body plan: 12.8 to 13.4% of groups have every lane
fallen and leave 86 to 92% of their steps unrun, while the rest run the full
trial. In total the exit removes 11.0 to 12.4% of all configured physics
steps, and the paired wall-time ratio sits at or above that share. The 3 g,
low-energy world counter-intuitively falls less (a limp body lies flat with
the head no lower than its neck base) and saves less. These are single-binary
runs on a machine shared with other workers: within one configuration the
per-pass rate swung by up to 2.6x (the off passes of the second 20 s run,
22,762 to 58,658 creatures/s), so the median ratios and the step shares are
the numbers to keep, not any single pass. The default path is untouched and
`first_generation 20000 20` still reports median -0.07 m, p99 0.34 m, best
11.03 m with the flag unset.

## 3M memory and end-to-end profile (2026-09-26)

Backlog items 57, 59 and 60 ask for the memory use at 3 million creatures, one end-to-end GUI generation at 3M, and the fine-check share of GPU time. All runs below used commit `614e04a` on the workstation listed under Machine, under

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 ...

so the RTX 4060 evaluated and the Radeon did not. The binary was built from a clean tree at 18:34; uncommitted edits from the other workers appeared in `src/` later, so every number here comes from that one binary. Free memory was checked before each 3M run: 24 GiB was available against a measured peak under 6 GiB, so the runs had headroom (system swap use moved from 106 MiB before the session to 189 MiB after it). Per-creature storage was not changed.

### Memory scaling

Headless command pattern, `N` = 100,000, 1,000,000, 3,000,000. The 0.5 s trial keeps evaluation small so the peak shows the data structures, and the checkpoint at the end is part of the peak:

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
      /usr/bin/time -v target/release/evolution-simulator headless \
      --population N --seed 38 --generations 1 --duration 0.5 \
      --checkpoint /tmp/opencode/mem3m/mem-N.evo

`/usr/bin/time -v` reports the peak resident set size. A separate `benchmark` run prints the population arena (`Population::bytes()`, which counts vector capacity) and the GPU allocation:

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
      target/release/evolution-simulator benchmark \
      --populations 100000,1000000,3000000 --duration 0.5 --generations 1 \
      --output /tmp/opencode/mem3m/bench-mem.csv

| population | peak RSS | evaluation | headless wall | checkpoint | population arena | per creature | GPU allocation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 422.1 MiB (0.41 GiB) | 0.49 s | 1.73 s | 30.1 MiB (31,607,539 B) | 47.1 MiB (49,366,912 B) | 494 B | 78.8 MiB |
| 1,000,000 | 2016.6 MiB (1.97 GiB) | 2.96 s | 12.08 s | 309.1 MiB (324,068,377 B) | 388.8 MiB (407,735,296 B) | 408 B | 96.8 MiB |
| 3,000,000 | 5744.1 MiB (5.61 GiB) | 9.09 s | 36.22 s | 942.2 MiB (988,025,720 B) | 1494.4 MiB (1,566,941,184 B) | 522 B | 96.8 MiB |

The first 100k run of the session was cold (shader compilation) with 6.10 s evaluation; the table uses the warm repeat.

Measured growth: 100k to 1M is 1.81 KB of peak RSS per creature, and 1M to 3M is 1.91 KB. A line through the three points is about 1.88 KB per creature plus a 213 MiB base; that fit is a bounded estimate from three points, and it predicts 5,870,254 KB at 3M against 5,881,932 KB measured. The arena itself is 408 to 522 B per creature. The rest of the peak is the per-slot state (scores, parent scores, trial metrics, ranks, candidate tables) plus the one generation of breeding and the checkpoint write holding old and new population memory at once. The `benchmark` process, which does not write a checkpoint, peaked at 5.22 GiB at 3M. The 3M checkpoint is 942 MiB.

### GUI end to end at 3M, default 60 s trials

Two generations, no warm-up, autosave off, stage log on:

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
      EVOLUTION_STAGE_LOG=/tmp/opencode/stage3m-robust2.csv \
      EVOLUTION_SMOKE_POPULATION=3000000 EVOLUTION_BENCH_DURATION=60 \
      EVOLUTION_BENCH_GENERATIONS=2 EVOLUTION_BENCH_WARMUP=0 \
      EVOLUTION_BENCH_NO_AUTOSAVE=1 /usr/bin/time -v \
      target/release/evolution-simulator --gpu "RTX 4060"

Worker summary with the default `EVOLUTION_ROBUST_TRIALS=2`:

- 2 generations in 132.238 s, end to end 45,373 creatures/s. Generation seconds: min 42.182, median 90.056, max 90.056.
- Device busy, totals since start: RTX 4060 5,495,854 standard trials in 128.100 s (112,126/s), CPU (6 threads) 605,631 standard trials in 126.086 s (20,765/s). Packing 3.932 s.
- GUI 15,425 frames at 116.6 FPS, p95 16.60 ms, p99 20.00 ms, max 73.69 ms.
- Control latency 259 probes, p50 2.9 ms, p95 287.3 ms, p99 740.9 ms, max 1,255.6 ms.
- Peak RSS 6,214,900 KB (5.93 GiB).

Stage log rows (generation, evaluation, archive, breeding, end to end):

| generation | evaluation | archive | breeding | end to end |
|---:|---:|---:|---:|---:|
| 0 | 30.226 s | 4.214 s | 7.694 s | 71,120/s |
| 1 | 68.351 s | 11.640 s | 9.888 s | 33,315/s |

The smoke start sends `Command::Run` in continuous mode, where evaluation, archive insertion and breeding overlap on the worker thread. The evaluation column is the generation wall minus the measured archive and breeding work, so the three columns sum to the generation wall but are not independent device timings. The summary's evaluation field wraps whole worker passes and also contains overlapped archive and breeding work; its archive field stays 0.000 s in the continuous path, and its breeding field (3.369 s) only counts the end-of-generation cleanup, which is why it is smaller than the 17.6 s of breeding in the stage log. The device busy totals are the cleaner compute numbers. Standard-trial counters are totals since start and exceed the 3,000,000 generation slots by about 2% (3,060,119 with checks on); the counter also sees work that crosses generation boundaries, so it is not a clean count of one generation. The rates above use the summary's generation wall, not the counters. This end-to-end rate is not comparable with the historical 341k/s at 3M figure, which used the earlier reduced-work physics.

### Fine-check share of GPU time (item 60)

Paired A/B: same fresh seed-38 population, one generation, 3M, 60 s trials, GUI benchmark, warm-up 0; only `EVOLUTION_ROBUST_TRIALS` differs. With 2, each creature whose score could enter an archive, plus every sample from an optimizing island CMA, gets a second trial at 4x physics rate and 4x solver passes; with 1 there are no check trials. In the continuous path the archive fills while results arrive, so eligibility is tested against a live archive.

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
      EVOLUTION_ROBUST_TRIALS=1 EVOLUTION_STAGE_LOG=/tmp/opencode/stage3m-g0-robust1.csv \
      EVOLUTION_SMOKE_POPULATION=3000000 EVOLUTION_BENCH_DURATION=60 \
      EVOLUTION_BENCH_GENERATIONS=1 EVOLUTION_BENCH_WARMUP=0 \
      EVOLUTION_BENCH_NO_AUTOSAVE=1 /usr/bin/time -v \
      target/release/evolution-simulator --gpu "RTX 4060"

| metric | checks on (2) | checks off (1) | difference |
|---|---:|---:|---:|
| generation wall | 43.405 s | 21.241 s | +22.164 s (+104.3%) |
| end-to-end rate | 69,117/s | 141,238/s | 2.04x slower |
| GPU busy | 40.645 s | 18.718 s | +21.927 s (+117.1%) |
| CPU busy | 41.103 s | 13.880 s | +27.223 s (+196.1%) |
| packing | 2.219 s | 1.410 s | +0.809 s |
| stage evaluation remainder | 28.818 s | 8.149 s | +20.669 s |
| stage archive | 5.415 s | 4.486 s | +0.929 s |
| stage breeding | 9.118 s | 8.589 s | +0.529 s |
| peak RSS | 5.65 GiB | 5.43 GiB | +0.22 GiB |

The extra 21.9 s of GPU busy and 22.2 s of generation wall is the check work plus scheduling: 53.9% of the check-on GPU busy and 51.1% of the check-on generation wall. This is a whole-check cost, including perturbing each body, packing it again and dispatching it, not a kernel-internal profile. A second check-on generation-0 sample from the two-generation run above took 42.18 s, 3% below 43.41 s, so read the share as about half, not to two decimals.

Caveats:

- Generation 0 starts with an empty archive, so this share is the cost while the archive fills. In the two-generation check-on run, generation 1 took 90.06 s against 22.82 s for the check-off run's generation 1, but those populations had already diverged, because check-on admission keeps the lower of the two trials. The later-generation share is not cleanly attributable, so no paired later-generation share is reported.
- The GUI path checks only creatures that could enter an archive. The headless path (`Scheduler::evaluate`, used by `headless` and `benchmark`) passes a callback that is true for every creature; the source comment there reads "Without an archive to compare against, every creature is checked." Headless evaluation seconds therefore include a fine check per creature and are not comparable with the GUI numbers in this section.
- GPU busy comes from Vulkan timestamp query spans summed across submissions; it can include stalls inside the command stream.

Everything in the tables is measured, except the 1.88 KB per creature line fit and its 213 MiB base, which are bounded estimates from the three memory points. The following could not be measured: a per-generation split of device busy into standard and check trials (the scheduler exposes run totals only), and a paired later-generation check share (the two arms cannot share an archive).

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
