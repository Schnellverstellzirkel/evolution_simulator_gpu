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

## Between-batch CPU costs (2026-09-27, archive and breeding)

Workload: headless, seed 38, 3 generations, 5 s trials, at 100k and 1M; `EVOLUTION_PROFILE_BREED=1`; 3 runs per side, alternating the baseline and optimized binaries back to back. Figures are medians over generations 1 and 2. The probe now also prints `Archive batch` (archive slots, stats, total), `Archive profile` (island offers, island refresh, prefilter, verify, global offers, cma tell, archive refresh, lineage), and `Plan profile` (by_plan, order/optimizer, sampling, cma/visit).

### Results

| metric | 100k before | 100k after | 1M before | 1M after |
| --- | --- | --- | --- | --- |
| archive total | 0.3057 s | 0.2826 s (-7.5%) | 0.8569 s | 0.8826 s (+3.0%) |
| archive without verify | 0.0540 s | 0.0423 s (-21.7%) | 0.3736 s | 0.3219 s (-13.8%) |
| breeding total | 0.1683 s | 0.1477 s (-12.2%) | 1.8194 s | 1.6907 s (-7.1%) |
| parent plans | 0.0244 s | 0.0174 s | 0.3410 s | 0.2593 s |
| candidate emission | 0.1406 s | 0.1283 s | 1.4160 s | 1.3917 s |
| island offers | 0.0157 s | 0.0078 s | 0.1047 s | 0.0531 s |
| global offers | 0.0093 s | 0.0064 s | 0.0611 s | 0.0424 s |
| verify (CPU replay) | 0.2517 s | 0.2403 s | 0.4833 s | 0.5607 s |

The archive side is dominated by verify, the CPU re-run of the best candidate per cell and per new body plan. That work is unchanged and deterministic; its 1M timing moved by run-to-run CPU contention, which is why the 1M archive total rose while every other archive section fell. At the real 3M / 60 s workload verify will be a larger share still, so the stage-log archive saving there will be smaller than these probe numbers. The 100k pair, where contention was lower, shows the addressable costs falling together.

### Changes

1. `Experiment::archive_slots` computes each descriptor once in the parallel prefilter and reuses it for the per-island offers. The four island offer streams now run in parallel, each preserving its own slot order. The global offers probe the cell fitness only for CMA candidates, and the morphology topology moves instead of cloning.
2. `plan_offspring` derives both the top-parent pool and the optimizer targets from one sorted elite order (was two identical sorts per island), and the CMA slot lookup probes `(niche, topology)` through hash buckets with borrowed keys, so a CMA offspring no longer clones a topology vector.
3. `QdArchive::visit` increments the visit count and marks the least-visited index dirty. The set is rebuilt once per breeding round (`ensure_least_visited`), replacing one balanced-tree remove and insert per visit.
4. `repair_with` answers skeleton connectivity with a union-find over the at most 64 nodes instead of a graph walk per candidate bone.
5. `align_nodes_with_bones` copies the starting coordinates onto the stack for bodies of at most 64 nodes instead of cloning the node vector.
6. `collect_parallel_streaming` preallocates each 4096-creature chunk from a size estimate and reserves the merged gene arenas exactly, removing the growth reallocation copies.

### Equivalence evidence

- `search_ab 3 256 2 38,39` and `search_ab 2 20000 0.5 38,39` stdout are byte-identical between the baseline and optimized builds.
- `cargo test --release`: 136 passed, 0 failed (79 lib, 3 early exit, 5 replay, 4 search improvements, 15 search state, 30 simulation); the GPU tests remain ignored.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` are clean.

### Not kept

- Offspring-level timers (atomic counters plus `Instant::now` per section) distorted emission by 2 to 3 times under six rayon threads, so the split was read from a single-thread run and the counters were removed.
- A `perf stat task-clock` A/B at 1M / 0.1 s trials showed only a 2.8% total CPU change, inside the noise; the wall section medians above resolved the change better.

## Performance campaign baseline (2026-09-27)

Planning measurements for the 2M/s target. Details, sweeps, and the candidate
list are in `docs/performance-campaign.md`. Measured on revision `2b1014c`
plus the concurrent `src/storage.rs` profiling edit, with a fixed snapshot
binary (sha256
`897529257d2eba776da62eddb6ea53fcd17fb967320260de3bfdf6377e825093`). Another
worker's CPU profiling shared the machine, load average 3.8 to 16.4, so the
medians and bests below are the numbers to keep.

Fixed checkpoint:

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
      cargo run --release -q -- headless --population 100000 --seed 38 \
      --generations 1 --duration 60 --checkpoint runs/perf-100k.evo

Kernel sweeps (eval-bench, 60 s trials, round-robin rounds, medians of 7):

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=0 \
      EVOLUTION_ROBUST_TRIALS=1 ... eval-bench --checkpoint runs/perf-100k.evo --repeat 1

    config                         median      best
    default                      92,632/s   94,933/s   (chunk 64, batch 100k, wg 32)
    EVOLUTION_GPU_BATCH=8192     66,326/s   68,963/s
    EVOLUTION_GPU_BATCH=100000   94,056/s   95,358/s
    EVOLUTION_GPU_CHUNK=16        86,426/s   89,289/s
    EVOLUTION_GPU_CHUNK=128       92,573/s   95,919/s
    EVOLUTION_LANE_WG=64          92,456/s   95,072/s
    EVOLUTION_KERNEL + EVOLUTION_PIPELINE_CHUNK  93,137/s   95,822/s (dead variables)

No default changed: batch 100k, chunk 64, and wg 32 are already at the
measured plateau. The two removed kernel variables are no-ops since `bd4c126`.
With the 6-thread CPU engine on the same workload the mixed rate was
39,775/s median (noisy), and with checks forced on every creature 7,439/s
median (3 rounds, very noisy).

Shader stats (`examples/shader_stats.rs`, workgroup 32): 128 to 155 registers
per thread depending on capacity, 5.5 to 65.5 KiB shared per block, and 128
KiB for capacity 64 at workgroup 64. Register occupancy estimate about 27 to
33% per SM, matching the earlier Nsight reading.

Benchmark subcommand, 5 s trials, one generation, checks on (every creature
is fine-checked because `Scheduler::evaluate` has no archive to compare
against):

| population | evaluation | generation | creatures/s |
|---:|---:|---:|---:|
| 100,000 | 0.748 s | 0.960 s | 133,724 |
| 1,000,000 | 7.696 s | 9.620 s | 129,936 |

With `EVOLUTION_ROBUST_TRIALS=1`: 359,967/s at 100k and 703,183/s at 1M. A
separate `--cpu` pass measured 174,486/s on 6 CPU threads at 100k / 5 s.

GUI stage-log runs at 5 s trials: 300k reached 114,159/s end to end (GPU busy
918,629 standard/s, CPU busy 67,538/s, archive 0.61 to 1.51 s, breeding 0.73
to 0.87 s); 1M reached 193,476/s (GPU busy 1,221,507/s, CPU busy 223,955/s,
archive 1.05 to 1.63 s, breeding 2.04 to 2.34 s, packing 2.204 s). The 3M,
60 s production point from the 2026-09-26 section stays at 45,373/s.

## 2M/s campaign, first wave: keep the GPU busy (2026-09-27)

Workload: `runs/evolved-3m.evo` (not committed), a 3M population evolved for 5 generations from seed 38 with 60 s trials and checks off, mean 5.16 nodes. Every row is the graphical game (`EVOLUTION_SMOKE_CHECKPOINT`, `EVOLUTION_BENCH_GENERATIONS=2`, `EVOLUTION_BENCH_WARMUP=1`, `EVOLUTION_BENCH_NO_AUTOSAVE=1`, `EVOLUTION_DEVICES=primary`, `RAYON_NUM_THREADS=8`, `nice -n 15`). One run per row: the search diverges between runs, so check counts and generation times vary by roughly 10%.

| step | end to end | seconds per generation | note |
|---|---:|---:|---|
| baseline (41a2191, 6 CPU threads) | 36,381/s | 82.1 to 82.9 | archive stage 55 to 58 s: CPU verify on the worker |
| shared checks + CPU replay | 33,806/s | 87.4 to 90.1 | archive 33 to 36 s; the GPU now waits on tiny check units |
| + reserve-bar filter | 29,489/s | 90.6 to 112.9 | verify gone (2 inline of 2,796); GPU 96% busy on check units |
| + four GPU queues | 41,842/s | 68.6 to 74.8 | small check units run beside standard units |
| + wall-clock rates | 43,093/s | 69.1 to 70.1 | committed as ffede6e |
| + 3 s units, one check unit at a time | 56,220/s | 48.6 to 58.1 | 6 CPU evaluation threads |
| same, CPU pool of 2 threads | 58,390/s | 47.8 to 55.0 | |
| same, no CPU pool (GPU only) | 64,095/s | 45.4 to 48.3 | inline replays on the worker |
| same, reserve CPU on the general pool | 65,288/s | 45.9 to 46.0 | default now; 119.7 FPS, control p99 1.1 s |

Unit length with the old rate estimator (checks on, 6 CPU threads, one measured generation each): 1 s 45k/s, 2 s 55.9k/s, 3 s 71.6k/s, 5 s 86.9k/s, 8 s 63.9k/s with 9.6 s of GPU idle. Those single generations had very different check counts (51k to 140k), and the old estimator inflated unit sizes, so the two-generation table above is the one to use.

GPU metrics (`nsys --gpu-metrics-devices=0`, busy samples): eval-bench on evolved bodies runs about 24 warps in flight per SM at 44% issue; the 3M GUI run with 1 s units ran about 14 warps at 29%. Warps in flight clustered at 12 to 19, which is why longer units helped. DRAM bandwidth stays at 1 to 3%: the kernel is latency and instruction bound, not memory bound.

Kernel facts measured on the way:

- GPU-only standard throughput, eval-bench, 60 s trials: 98k/s on the first-generation 100k checkpoint (4.68 nodes), 56k/s on the evolved 100k checkpoint (6.51 nodes), 77.5k/s on the first 1M creatures of the evolved 3M checkpoint (5.16 nodes).
- A fine check (240 Hz, 8 bone and 4 velocity passes) costs 5.6 standard trials on 50k evolved-100k bodies and 7.7 on the evolved 3M bodies, not 16.
- Ablations on the evolved 100k population (share of kernel time): muscles 36% (the two waveform evaluations 14%, the force scatter 8%), descriptor metrics with fall and break checks 22%, joint limits 16%, the two bone passes 14%, the velocity pass 8%.
- Simulated time after a fall (`examples/fall_profile.rs`): 25% on the evolved 100k population, 34% on a first generation; the median fall comes within 1 s.
- Body plans: the evolved 3M population has 13,604 plans; the ten largest hold about half the creatures. A kernel specialized for one plan (`EVOLUTION_PLAN_BATCH=1` with `eval-bench --plan-rank`) is bit-exact and 10 to 20% faster on a single-plan population, but splitting mixed units into plan batches measured no gain (82k generic against 60k to 82k creatures/s), so it stays off.

## Segments, early screening and thread budgets (2026-09-27)

Workload: `runs/evolved-3m-v26.evo` (not committed), the evolved 3M checkpoint resumed for 4 generations under `qd::VERSION` 26 with checks off. GUI benchmark as in the section above, one run per row.

| step | end to end | seconds per generation |
|---|---:|---:|
| fall freeze, no segments (`EVOLUTION_SEGMENTS=0`) | 47,024/s | 57.8 to 69.8 |
| GPU segments at 2 s and 10 s | 65,351/s | 39.0 to 52.8 |
| screening off (`EVOLUTION_SCREEN=0`), same binary | 69,805/s | 41.6 to 44.4 |
| screening at 5 s, top 20% continue (default) | 121,885/s | 21.3 to 27.9 |
| same, second run | 137,824/s | 18.6 to 25.0 (last generation 161,445/s) |

Segments are bit-exact against unsegmented runs (100,000 evolved 100k and 500,000 evolved 3M creatures). eval-bench, GPU only: 56.9k -> 63.4k on the evolved 100k checkpoint and 77.2k -> 93.3k on the evolved 3M bodies.

Screening predicts well: on 200k evolved 3M creatures, keeping the top 20% by distance at 5 s kept every creature of the final top 1% and 96% of the final top 10% (Spearman 0.887; at 10 s 0.942; at 2 s 0.730).

Search quality, `examples/search_ab.rs`, 10 seeds (38 to 47), 10,000 creatures, 60 s trials, CPU only:

| variant | generations | best mean (median) | QD mean (median) | cells | wall |
|---|---:|---:|---:|---:|---:|
| no screening | 20 | 270.0 m (235.3) | 22,177 (20,353) | 1,228 | 189.8 s |
| screened, never in an archive | 20 | 243.7 m (239.2) | 19,117 (19,522) | 1,000 | 84.7 s |
| screened, never in an archive | 40 | 420.3 m (429.0) | 52,929 (52,156) | 1,119 | 190.4 s |
| screened may open empty global cells | 20 | 199.8 m (200.0) | 17,727 (16,718) | 1,088 | 84.7 s |
| screened may open empty global cells | 40 | 425.8 m (412.1) | 53,266 (44,642) | 1,191 | 190.9 s |

At equal evaluations screening costs about 14% of QD and a fifth of the cells; at equal time it gives +56% best distance and 2.4x QD for 9% fewer cells. The shipped rule keeps screened creatures out of every archive: letting them open empty cells tied at equal time, lost per evaluation, and would leave 5 s scores in the player's archive. Single seeds swing by 2x between variants, so read only the 10-seed means.

Thread budget at 3M with screening (GUI): 8 general workers 137,824/s; 16 general workers 129,792/s; 8 general plus 8 CPU evaluation workers 121,273/s (the GPU's own rate fell from 174k to 137k standard/s with the CPU evaluating). The game keeps the 8-worker budget.

Kernel experiments that did not help, all bit-exact: grouping 2 or 4 muscles per loop iteration (79.9k, 79.1k, 79.3k creatures/s on 200k evolved bodies); per-plan batches in mixed units (see above).

Muscle waveform (2026-09-27, eval-bench on the evolved 100k checkpoint, GPU only, interleaved runs). Both evaluations of the waveform cost 14% of kernel time, so two ways to evaluate it once per step were tried. Caching each muscle's last target length in a 16th muscle field measured 60.3k to 60.7k creatures/s against 63.8k to 64.8k for the committed kernel: the extra global load and store per muscle cost more than the cosine it saves. A branch-free waveform with one cosine for both halves of the stroke measured 61.2k to 62.8k against 64.2k to 64.6k and was not bit-exact on the GPU (3,162 of 100,000 identical; the driver fuses the multiply-adds differently). Neither landed on the GPU. On the CPU engine the one-cosine waveform is bit-exact (20,000 of 20,000 evolved bodies) and measured 9,588 to 9,790 creatures/s against 9,363 to 9,528 on six threads, so it landed there.

Occupancy (2026-09-27). An nsys trace of the 3M GUI benchmark shows about 12 warps in flight per SM (p50; p90 17) of 48, at 23% issue. The compiled kernel uses 128 to 149 registers per thread, which allows 14 to 16 warps, and at 8 nodes the four shared node arrays took 8,192 bytes per one-warp workgroup, which allows 12. Doubling the declared shared memory (half the warps) cut eval-bench from 62k to 41k creatures/s, so the kernel is latency bound and occupancy pays. Removing 8 of the 10 per-step global loads of muscle genes changed nothing (61.8k to 62.8k against 62.5k to 63.1k), so those loads are not the latency.

The fourth shared array (`scr`, muscle forces and the rebuild's shape copy) is gone: muscle forces add up in `old`, which is free until the velocities integrate, and the rebuild turns each child's position into its offset from its parent in place (children first) before it writes positions back parent-first. Shared memory at 8 nodes fell from 8,192 to 6,144 bytes. Bit-exact on the evolved 100k checkpoint (100,000 of 100,000) and on the first 300,000 creatures of the evolved 3M checkpoint. eval-bench, GPU only, interleaved: evolved 100k 65.1k to 67.4k against 61.4k to 62.4k creatures/s; evolved 3M sample 86.3k to 92.8k against 81.2k to 86.3k. The GUI benchmark measured 165,572/s end to end against 169,308/s for the previous kernel, inside the 10% spread between single runs (this run checked 91,661 contenders against 83,610). Keeping the 19 trial metrics in their global result slot instead of registers brought every bucket up to 8 nodes to 128 registers (from up to 148) and removed the driver's extra shared memory for small bodies, but measured no faster (63.4k to 65.3k against 63.3k to 66.4k creatures/s, bit-exact), so it did not land. The compiler appears to cap the kernel near 128 registers and spill the rest; more than 16 warps per SM would need far fewer live values.

## 30 Hz physics: measured, not adopted (2026-09-27)

`EVOLUTION_PHYSICS_RATE=30` halves the standard steps per trial. In the 3M GUI benchmark it raised the rate from about 130,000 to about 214,000 creatures/s end to end.

Search at equal wall time, `examples/search_ab.rs`, 10 seeds (38 to 47), 10,000 creatures, 60 s trials, screening on, CPU only:

| run | generations | wall | best mean (median) | QD mean (median) | cells | top-50 median length |
|---|---:|---:|---:|---:|---:|---:|
| 60 Hz | 20 | 79.7 s | 243.7 m (239.2) | 19,117 (19,522) | 1,000 | 1.39 m |
| 30 Hz | 20 | 45.4 s | 217.0 m (200.2) | 18,401 (17,831) | | |
| 30 Hz | 32 | 80.0 s | 332.7 m (303.1) | 36,959 (32,678) | 1,090 | 2.94 m |
| 60 Hz | 40 | 190.4 s | 420.3 m (429.0) | 52,929 (52,156) | 1,119 | 3.03 m |
| 30 Hz | 40 | 104.8 s | 373.0 m (366.4) | 53,107 (48,424) | 1,137 | 3.08 m |

At equal time 30 Hz is ahead on every seed but one (best distance higher on 9 of 10, QD on 10 of 10). At equal generations the two rates tie: 30 Hz is higher on 5 of 10 seeds for best, QD, cells, length and mass.

The gain does not survive a finer replay. The twelve fastest elites of three headless 20-generation runs per rate (seeds 39 to 41, 100k creatures, `runs/hz30-s*.evo` and `runs/hz60-s*.evo`) were replayed by `examples/size_report.rs` at twice their evolved rate:

| elites | replayed at | median share of the archive distance kept | below 10% of it |
|---|---:|---:|---:|
| evolved at 60 Hz | 120 Hz | 90% | 1 of 36 |
| evolved at 30 Hz | 60 Hz | 38% | 12 of 36 |

For example, seed 41's 30 Hz elites of rank 5 to 7 cover 63.6, 60.8 and 58.7 m at 30 Hz and 0.2, -0.1 and 0.1 m at 60 Hz. Evolution at 30 Hz finds gaits that depend on the coarse step, so the extra search speed mostly buys exploits of the integrator. The default stays at 60 Hz.

## Peak memory and GPU unit length (2026-09-27)

The 3M GUI benchmark peaked at 18.3 GB RSS (`/usr/bin/time`), up from about 8 GB in the morning's runs; the laptop has 30 GB. Sampling `/proc/<pid>/smaps` during a run: 14.6 GB anonymous heap, 2.0 GB GPU-mapped buffers (`/dev/nvidiactl`), 0.2 GB `[heap]`. Loading the checkpoint alone costs 2.3 GB (`examples/mem_report.rs`). `MALLOC_ARENA_MAX=2` saved only 0.6 GB, so the memory is live data, not allocator fragmentation.

Two causes:

- Gene arenas. Breeding appends each generation's children to the arenas and the generation boundary compacts them into a spare and swaps. Appends grew the arenas by doubling, and the swap kept the old, oversized arena as the spare: at the second boundary the arenas held 8.2 GB plus an 8.0 GB spare by capacity for 2.2 GB of live genes. Now the boundary reserves one generation of children plus an eighth, and shrinks the spare to what the next compaction fills (4.6 GB plus 2.3 GB by capacity). Peak RSS 18.6 -> 14.8 GB in one run each.
- Work in flight. Each GPU unit keeps its creatures, the packed batches and, with segments, the read-back state, and five units per GPU may be queued. With screening a 3 s unit held about a sixth of the population, so nearly all 3M creatures were in flight. The host copies of packed node state and muscle buffers are now freed once uploaded.

GPU unit length after those changes, two or four GUI runs each (`EVOLUTION_UNIT_SECONDS`):

| unit | end to end | peak RSS |
|---:|---:|---:|
| 0.5 s | 168,685 and 161,268/s | 11.3 to 11.4 GB |
| 1 s | 186,134, 181,499, 195,927 and 183,352/s | 12.3 to 12.7 GB |
| 2 s | 180,247 and 174,332/s | 13.4 to 13.9 GB |
| 3 s | 164,244 and 172,891/s | 14.8 to 15.1 GB |

The default is now 1 s: 186,700/s on average against 168,600/s for 3 s, with 2.5 GB less peak memory. FPS stayed at 76 to 81 and control p99 at 1.2 s in every row.

## Control latency (2026-09-27)

Every row is one 3M GUI benchmark run as above. Control latency is the wait of a ping probe sent every 0.5 s. Settings latency (`EVOLUTION_BENCH_SETTINGS_PROBE=1`) re-applies the current settings every 5 s, like an environment button.

| change | end to end | control p95 / p99 | settings median / max |
|---|---:|---:|---:|
| before (1 s units) | 186,700/s mean | 0.6 / 1.2 s | not measured |
| breeding writes the arena in parallel | 171,041 and 197,551/s | 0.42 to 0.61 / 0.57 to 1.0 s | |
| + one finished unit per worker pass | 197,048 and 195,990/s | 0.21 to 0.24 / 0.25 to 0.39 s | |
| same binary, settings probe on | 153,130/s | 7.3 / 8.3 s | 3.0 / 8.8 s |
| + no drain for steady settings changes | 204,820/s | 0.19 / 0.23 s | 0.025 / 0.16 s |

A settings change, meteor, extinction or undo used to wait for all queued GPU work and its checks, which idled the GPU and blocked every other control behind it. In a steady run the change waits for the generation boundary anyway, and work in flight at a boundary already carries over, so those commands now skip the drain. Save and load still drain.

Knobs re-measured after these changes (3M GUI benchmark, two runs each, interleaved), all left at their defaults:

| knob | runs | end to end |
|---|---|---:|
| step range 64 (default, `EVOLUTION_GPU_CHUNK`) | 2 | 216,293 and 204,842/s |
| step range 128 | 2 | 180,661 and 195,423/s |
| step range 256 | 2 | 181,385 and 189,341/s (UI 65 FPS) |
| segments at 2 and 10 s (default, `EVOLUTION_SEGMENTS`) | 2 | 212,034 and 192,205/s |
| segments at 2, 10, 20 and 35 s | 2 | 197,182 and 183,383/s |

An nsys trace with 1 s units shows the GPU at 100% activity for most of the measured generations, with short dips to 75 to 90%, at about 10 warps in flight per SM (p50) and 20% issue.

## Long sessions (2026-09-28)

The owner saw evolution slow sharply after a few dozen generations. Their 3M session reached generation 70 (autosave `runs/seed-1790546317494369957-auto.evo`, not committed). Measured against the evolved 3M checkpoint at generation 9. `examples/body_stats.rs` reports body sizes. The GUI benchmark ran one warm-up and two measured generations with `EVOLUTION_CPU_THREADS=0` and autosave off. `eval-bench` ran on 200,000 creatures with every creature fine-checked.

| | generation 9 | generation 70 |
|---|---:|---:|
| mean nodes / bones / muscles | 6.02 / 5.02 / 8.99 | 10.64 / 9.64 / 33.98 |
| largest body | 11 nodes, 30 muscles | 21 nodes, 96 muscles (the cap) |
| packed GPU data per creature | 1,005 B | 2,818 B |
| `eval-bench`, GPU only | 15,519 and 15,309/s | 3,638 and 3,784/s |
| GUI end to end | 185,000 to 216,000/s (2026-09-27) | 29,720 (warm-up), 36,815, 40,601/s; 38,615/s over the two measured generations |
| evaluation / archive / breeding per generation | | 61.9 to 68.9 s / 2.9 to 3.2 s / 9.0 to 9.3 s |
| peak RSS | about 10 GB | 22.2 GB |
| checkpoint | 1.37 GB | 4.18 GB |

The GPU held 2,502 MHz mean at 44 W and 59 °C over the busy samples, so the slowdown is work per creature, not heat. `mem_report` puts the generation-70 population at 6.4 GB of genes. The arena keeps the live generation, its children and a compaction spare, and the old default autosave cloned the whole experiment every tenth generation on top. That is enough to reach swap on this 32 GB machine.

Fixes landed:

- A continuous run's autosave could not be loaded ("Invalid completed fitness values"). The loader assumed the first `evaluated` scores are the finished ones, which is false when slots are evaluated and re-bred in any order. It now checks only that every score is finite or NaN. Test: `a_continuous_run_checkpoint_with_scattered_scores_loads`.
- Autosave is off by default, and a loaded game starts with it off (owner: as few files as possible).

Measured and not adopted: removal operators (`EVOLUTION_SHRINK=1`: `remove_limb` and `remove_muscle` at the rate of the operators that add nodes). `examples/search_ab.rs` ran 10 seeds (38 to 47), 40 generations and 5,000 creatures, with 60 s trials. It now prints mean nodes and muscles per generation.

| | nodes / muscles at generation 39 | best, mean (median) | QD, mean (median) | CPU wall |
|---|---:|---:|---:|---:|
| current operators | 7.31 / 13.61 | 346 m (280) | 30,075 (17,768) | 196 s |
| with removal | 6.41 / 10.31 | 275 m (278) | 21,329 (18,111) | 182 s |

Removal slows growth by about a quarter. Medians tie, and the means favour the current operators because of one or two strong seeds. Growth is selected for: a muscle has no mass and brings its own energy store and force. The design response is in `docs/data-architecture.md` section 10.

## Muscle mass: measured, not adopted (2026-09-28)

The owner asked for muscle mass to be built and measured, as the physics lever against free body growth (`docs/data-architecture.md` section 10.2, decision 6). The implementation is on branch `claude/muscle-mass` and is not merged. It is in all three engines:

- A muscle's span is the distance between its two attachment points in the starting pose, at least 5 cm (`physics::muscle_span`).
- A muscle weighs `muscle_density` times its span (`EVOLUTION_MUSCLE_DENSITY`, default 1 kg/m). Half of that mass sits at each attachment point and is shared by that bone's two nodes like an organ, so the center of mass is exact (`physics::body`).
- A muscle's energy store is `muscle_energy` times its span: 200 J/m instead of a flat 120 J per muscle (`physics::muscle_capacity`). The heat wave scales it as before. The GPU kernel reads the store as a 16th muscle field.
- `qd::VERSION` 27. The effect-test walker was evolved again with `examples/evolve_walker.rs`. The CPU/GPU agreement tests use the old walker, which moves only 0.66 m at fine fidelity under the new physics but has a steady trajectory there.

200 J/m keeps the median store near today's. `examples/muscle_mass_probe.rs` on the generation-9 3M checkpoint (first 300,000 creatures) gives a median span of 0.53 m (p10 0.13 m, p90 1.23 m), so the median store is about 105 J. At 1 kg/m the median body mass rises from 9.9 kg to 16.3 kg.

Checks on the branch: formatting, all-target clippy, 149 CPU tests and nine report tests pass. The two engine agreement tests and the three GPU simulation tests pass. The agreement gaps are 0.0065 m for the walker at fine fidelity and 0.042 m for its perturbed contender, against tolerances of 0.5 m and 0.05 m. `first_generation 20000 20` gives median -0.06 m, p99 0.27 m, best 4.04 m at 1 kg/m and median -0.05 m, p99 0.26 m, best 2.78 m at 4 kg/m, against median -0.07 m, p99 0.34 m, best 11.03 m without muscle mass. No free propulsion.

Search A/B: `examples/search_ab.rs` with 40 generations, 5,000 creatures, 60 s trials and seeds 38 to 47, CPU only, 4 threads. The baseline arm is cb5119e and reproduces the removal-operator table above exactly.

| | nodes / muscles at generation 39 | top-50 mean nodes | top-50 bone length, median of seeds | best, mean (median) | QD, mean (median) | cells, mean | CPU wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| massless muscles, 120 J each | 7.31 / 13.61 | 7.44 | 2.00 m | 346 m (280) | 30,075 (17,768) | 1,018 | 187 s |
| 1 kg/m, 200 J/m | 7.47 / 13.60 | 7.74 | 4.01 m | 155 m (133) | 16,007 (12,648) | 957 | 202 s |
| 4 kg/m, 200 J/m | 7.51 / 14.21 | 8.73 | 5.58 m | 144 m (121) | 15,707 (13,343) | 830 | 245 s |

Mean nodes / muscles of the population by generation, averaged over the ten seeds:

| generation | 0 | 10 | 20 | 30 | 39 |
|---|---:|---:|---:|---:|---:|
| massless | 4.01 / 3.68 | 5.98 / 8.93 | 6.58 / 11.18 | 6.92 / 12.24 | 7.31 / 13.61 |
| 1 kg/m | 4.01 / 3.68 | 6.19 / 9.27 | 6.77 / 11.01 | 7.02 / 11.91 | 7.47 / 13.60 |
| 4 kg/m | 4.01 / 3.68 | 6.32 / 9.87 | 6.80 / 11.40 | 7.32 / 13.48 | 7.51 / 14.21 |

Best distance per seed (38 to 47): massless 238, 214, 321, 778, 353, 227, 378, 617, 152, 184 m. At 1 kg/m: 57, 80, 57, 162, 128, 421, 139, 255, 152, 103 m. At 4 kg/m: 216, 53, 69, 384, 77, 126, 153, 152, 93, 117 m.

Muscle mass does not stop body growth. At 1 kg/m the population grows at the same rate as without it. The second setting, 4 kg/m, was tried because at 1 kg/m a typical 0.5 m muscle weighs about 5 N against the 100 N a muscle may pull, so the cost might have been too small to matter. At 4 kg/m the population grows faster and the top 50 are larger. Best distance and QD fall by about half at both settings, and archive coverage falls. Two causes are likely and neither was tested. A store that grows with span rewards long muscles, and long muscles need large bodies. Grip grows with the load a foot carries (97e3e9a), so extra weight helps traction. Not measured: the GPU cost of the 16th muscle field.

### Five variants over 80 generations (2026-09-28)

The owner accepts half the best distance if muscle mass stops the muscle monsters, the bodies that jiggle dozens of appendages. The span-scaled store did not lower muscle counts in 40 generations. The next variant keeps muscle mass and gives every muscle the old flat 120 J store. `EVOLUTION_MUSCLE_STORE=120` sets it in all engines (`physics::muscle_capacity`). The owner's problem appeared at generation 70 of a 3M run, so every variant ran to 80 generations. `search_ab` now also prints the top 50's mean and largest muscle count. Settings are as above: seeds 38 to 47, 5,000 creatures, 60 s trials, CPU only, 4 threads. The baseline is cb5119e. Each run's first 40 generations match the 40-generation runs exactly. The top-50 columns describe the final generation's 50 best-scoring creatures.

| | nodes / muscles, gen 39 | nodes / muscles, gen 79 | top-50 mean muscles | most muscles in a seed's top 50 | top-50 mean nodes | top-50 bone length / mass, median of seeds | best, mean (median) | QD, mean (median) | cells, mean | CPU wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| massless muscles, 120 J | 7.31 / 13.61 | 8.37 / 18.56 | 21.7 | 78 | 8.86 | 3.00 m / 8.9 kg | 588 m (490) | 104,828 (57,527) | 1,131 | 477 s |
| 1 kg/m, 200 J/m | 7.47 / 13.60 | 8.55 / 17.43 | 26.7 | 76 | 10.85 | 5.36 m / 34.3 kg | 247 m (195) | 33,506 (23,569) | 1,075 | 468 s |
| 4 kg/m, 200 J/m | 7.66 / 14.93 | 9.05 / 19.96 | 31.4 | 60 | 12.11 | 10.55 m / 196.2 kg | 222 m (181) | 27,975 (25,865) | 957 | 552 s |
| 1 kg/m, flat 120 J | 7.42 / 13.43 | 8.70 / 18.24 | 29.0 | 91 | 11.38 | 6.66 m / 54.8 kg | 289 m (237) | 30,121 (31,108) | 1,072 | 507 s |
| 4 kg/m, flat 120 J | 7.65 / 14.09 | 9.14 / 20.34 | 31.9 | 59 | 11.82 | 9.28 m / 124.3 kg | 245 m (233) | 24,072 (20,586) | 956 | 539 s |

Largest muscle count in each seed's top 50, seeds 38 to 47:

- massless: 16, 9, 17, 78, 9, 11, 23, 17, 49, 46
- 1 kg/m, 200 J/m: 33, 25, 76, 8, 33, 21, 64, 42, 13, 35
- 4 kg/m, 200 J/m: 41, 38, 23, 60, 60, 45, 14, 53, 16, 42
- 1 kg/m, flat 120 J: 62, 46, 19, 72, 17, 25, 57, 25, 91, 52
- 4 kg/m, flat 120 J: 33, 17, 28, 25, 19, 38, 59, 55, 46, 56

No variant holds muscle counts down. The population's mean at generation 79 ranges from 6% lower (17.43 at 1 kg/m with the span store) to 10% higher (20.34 at 4 kg/m with the flat store) than the 18.56 without muscle mass, and the top 50 of the 6% arm carry 23% more muscles than the baseline's top 50. In every variant the top 50 carry more muscles and more nodes than without muscle mass, and they are 1.8 to 3.5 times longer and 4 to 22 times heavier. The largest monster shrinks in the 4 kg/m arms (59 and 60 muscles against 78), but typical top bodies grow. Best distance falls by 51 to 62% and QD by 68 to 77%. Heavier bodies win under muscle mass. One likely reason, untested: grip grows with the load on a foot (97e3e9a), and a muscle's 100 N force limit dwarfs its weight (5 N for a 0.5 m muscle at 1 kg/m). Because no variant held muscle counts down, first_generation and the agreement tests were not re-run for the flat store.

A quicker check on an existing save: `examples/monster_check.rs` samples every 150th creature of the owner's generation-70 3M autosave (20,000 creatures, mean 37.5 muscles) and scores the sample with full 60 s trials, no screening, on the CPU engine under each physics. It takes 45 s to sample and 7 to 10 s per physics.

| muscles | creatures | massless: median, p90, share of the top 10% | 1 kg/m, 200 J/m | 1 kg/m, flat 120 J | 4 kg/m, flat 120 J |
|---|---:|---:|---:|---:|---:|
| 0 to 12 | 3,149 | 0.23 m, 34.3 m, 15% | 0.17 m, 3.0 m, 17% | 0.17 m, 3.2 m, 19% | 0.11 m, 1.2 m, 14% |
| 13 to 20 | 2,870 | 0.48 m, 43.7 m, 15% | 0.29 m, 5.3 m, 22% | 0.29 m, 4.1 m, 21% | 0.16 m, 1.4 m, 15% |
| 21 to 34 | 2,315 | 1.36 m, 69.8 m, 15% | 0.39 m, 5.4 m, 23% | 0.39 m, 4.6 m, 21% | 0.23 m, 2.0 m, 23% |
| 35 to 60 | 8,584 | 2.23 m, 36.1 m, 30% | 0.14 m, 1.9 m, 33% | 0.14 m, 1.9 m, 34% | 0.04 m, 1.3 m, 42% |
| 61 and more | 3,082 | 8.51 m, 44.0 m, 24% | 0.07 m, 1.0 m, 4% | 0.06 m, 1.0 m, 4% | -0.01 m, 0.3 m, 5% |

Today's monsters collapse under muscle mass: the median body with 61 or more muscles falls from 8.51 m to 0.07 m, and its share of the top 10% from 24% to 4%. The 200 J/m store without mass (`EVOLUTION_MUSCLE_DENSITY=0`) leaves that bin at 8.23 m, so the mass causes the collapse. But lean bodies lose most of their distance too (p90 34 m to 3 m), because every gait in the save was tuned to massless muscles. The save test shows what a change does to existing creatures. It cannot show what evolution builds afterwards. The 80-generation runs answer that, and there evolution builds heavy, muscular bodies again.

## Persistent lanes, stage 1: measured, parked (2026-09-28)

Stage 1 of `docs/data-architecture.md` section 11 is built on branch `claude/lanes` (`EVOLUTION_LANES=1`), default off and not merged. The host writes each creature once as a record (starting node state, bone fields, muscle fields, the same values as the tiles). Creatures wait in one queue per class (node capacity, fidelity and physics settings) and optionally per bucket of 4 muscles. Each workgroup is a warp slot whose node state, muscle and bone data stay in device memory between epochs. Every 32 steps the idle lanes of a warp take the next creatures from its queue. A creature still running at the end of an epoch is suspended in its lane and resumes in the next epoch. The host reads back only finished results, queue heads and running lanes. There are no tiles, segments, readbacks or repacks.

Correctness: bit-exact against the segment engine, 100,000 of 100,000 evolved 3M creatures, both with one long epoch and with 2,048-step epochs that suspend creatures. One bug on the way: lanes of different classes used their own node stride and overlapped in the node buffer; every lane now has room for the largest class.

Rates, evolved 3M checkpoint (`runs/evolved-3m-v26.evo`), GPU only, one run per row unless noted:

| workload | segment engine | lanes |
|---|---:|---:|
| `eval-bench` 500k, standard trials only | 72,000 to 77,400/s | 59,000 to 65,400/s |
| `eval-bench` 300k, every creature checked, muscle buckets of 4 | 14,800 to 15,400/s | 9,700 to 10,700/s |
| same, no muscle buckets | | 11,300 to 11,400/s |
| 3M GUI benchmark, end to end (before the time budget) | 243,857/s | 148,121/s |

Why it is slower:

- Registers. The lane kernel compiles to 155 to 168 registers at 4 to 8 nodes, where the segment kernel has 128. By the occupancy rule measured in `docs/phase0-measurements.md`, anything above 128 registers leaves 12 resident warps per SM instead of 16. Variants measured with `examples/shader_stats.rs` at 6 nodes: constants derived inside the step loop 168; derived afresh every 32-step chunk 155; the 19 trial metrics in workgroup memory 167; derived once before the loop with no refill 128. The driver's allocation does not follow live values in any simple way.
- Sparse warps. Fine checks and long survivors spread over many classes and buckets, and a warp holding one long trial takes a full SM slot. At the end of one checked epoch 2,572 lanes ran in 901 warps. The segment engine repacks survivors into dense tiles at every segment, which is what the lanes lack.
- Lockstep epochs were a suspect but not the cause: a shared budget of lane chunks, so costly warps run fewer steps and every warp stops at the same time, changed the checked rate by less than 10%.

What stage 1 would need to win: a kernel at 128 registers or fewer, and GPU-side compaction that moves running creatures into dense warps between epochs. Both are open. The segment engine stays the default.

## GPU memory: ride out a full GPU, smaller buffers (2026-09-28)

The owner started the game while another process held GPU memory. The first submission failed with "A device memory allocation has failed", the scheduler retired the GPU and sent every unit to the CPU for the rest of the session. The GPU thread now keeps a unit whose allocation fails for lack of device or host memory (`vk_engine::out_of_memory` looks for `ERROR_OUT_OF_DEVICE_MEMORY` or `ERROR_OUT_OF_HOST_MEMORY` in the error chain). It frees the buffers that idle slots keep for reuse and tries again. While other units run, the unit waits for one of them to finish, and fewer units run at once from then on; one more is tried every 10 s. With nothing running it retries every 0.5 s for up to 60 s and only then fails as before. It prints one line when it starts waiting and one when memory is back. A submission now prefers the free slot with the most cached buffers, and a failed allocation no longer leaks the buffers made before it. Tests in `src/engine.rs` run the GPU thread's loop on a fake device that runs out of memory.

Buffers used to round up to the next power of two, which wastes up to half of a large buffer. Buffers above 1 MiB now get 25% headroom, rounded up to 1 MiB. Smaller ones still round up to a power of two.

`examples/gpu_hog.rs` holds a chosen amount of GPU memory for a while, to test the recovery by hand.

3M GUI benchmark on `runs/evolved-3m-v26.evo` (`EVOLUTION_BENCH_GENERATIONS=2`, `EVOLUTION_BENCH_WARMUP=1`, `EVOLUTION_BENCH_DURATION=60`, `EVOLUTION_CPU_THREADS=0`, `EVOLUTION_DEVICES=primary`, `RAYON_NUM_THREADS=8`, autosave off), the game's own GPU memory sampled every 0.5 s with `nvidia-smi --query-compute-apps`. One run each, on a machine with other agents' CPU work (load average about 11):

| | peak GPU memory | p90 | median | end to end | peak RSS |
|---|---:|---:|---:|---:|---:|
| power-of-two buffers | 2,253 MiB | 1,811 MiB | 800 MiB | 226,603/s | 10.06 GB |
| 25% headroom | 1,920 MiB | 1,536 MiB | 706 MiB | 212,805/s | 9.96 GB |

Peak GPU memory falls by 15%. The rate difference is inside the 10% spread between single runs.
