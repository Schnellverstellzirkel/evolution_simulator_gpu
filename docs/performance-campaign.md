# Performance campaign

Target: 2,000,000 evaluated creatures/s in the graphical game with fixed 60 s
trials, 3 million creatures per generation, and the UI at 60 FPS. The last
measured production point is 45,373 creatures/s end to end at 3M with 60 s
trials, so the gap is about 44x. This document is measurement and planning
only. No optimization has landed here.

Measured on the worktree at revision 2b1014c plus an in-flight profiling edit
to `src/storage.rs` from a concurrent worker. Later edits to `src/evolution.rs`
and `src/qd.rs` appeared after the snapshot was taken, so none of the runs in
this campaign contain them. A snapshot binary was kept at
`/tmp/opencode/perfcamp/evolution-simulator-snap-2b1014c` (sha256
`897529257d2eba776da62eddb6ea53fcd17fb967320260de3bfdf6377e825093`) so every
run in this campaign used one binary. The fixed kernel checkpoint is
`runs/perf-100k.evo` (gitignored), a seed-38 first-generation population of
100,000 bodies, mean 4.68 nodes, histogram 3: 616, 4: 42,229, 5: 45,877,
6: 11,278, file 29.9 MiB.

The machine was shared with another worker running CPU-heavy profiling for
most of this campaign. Load average moved between 3.8 and 16.4 during the
sweeps. Every knob table uses round-robin rounds so each config sees the same
drift, and the median and best columns are the ones to trust. Sections that
could not be ranked are marked noisy.

## Target decomposition

At 3M creatures and 2M/s, one generation must finish in 1.50 s of wall time.
The 2026-09-26 3M GUI run measured one generation at 90.056 s (generation 1),
and the paired check A/B measured the device busy split in the same session.
Everything below is per 3M generation.

| term | now | target | factor | note |
|---|---:|---:|---:|---|
| generation wall | 90.056 s | 1.500 s | 60x | generation 1, 60 s trials, checks on |
| GPU busy, checks on | 40.645 s | 1.500 s | 27x | paired A/B, one generation |
| GPU busy, standard only | 18.718 s | 1.500 s | 12x | same population, checks off |
| CPU busy, checks on | 41.103 s | hidden or 1.500 s | 27x | 6 evaluation threads |
| archive + breeding wall | 21.528 s | hidden under the GPU | 14x or overlap | stage log, generation 1 |
| check share of GPU busy | 53.9% | below 10% | 5x | checks cost 21.927 s |
| post-fall share of simulated time | 43 to 55% | near 0 | 2x | research figure from item 47 |
| kernel occupancy | 27 to 33% | 60%+ | 1.8 to 2.2x | register bound, see shader stats |
| GPU batch | 100,000 | 100,000+ | 1.0x | already at the measured plateau |

Read as a budget: the standard kernel must produce about 2.0M standards/s
including a reduced check tax, the CPU pipeline must finish its 3M creature
pass in under 1.5 s or run concurrently with GPU work, and the check policy
must stop paying 16 standard equivalents on every archive contender. A factor
of 27x is not reachable with one lever. A plausible stack is check policy
(1.5 to 2x), post-fall compaction (1.8 to 2x), occupancy work (1.8 to 2x),
two creatures per lane for the small buckets (about 2x), and CPU overlap that
removes the 21.5 s serial block. Those together are about 10 to 16x, so the
plan needs at least one structural change beyond tuning: fewer solver passes
at equal agreement, a descriptor pass that does not live in the physics
kernel, or f16 or fixed-point state for the small buckets.

The 5 s GUI runs show the same shape at a smaller scale. Archive and breeding
cost about 3 to 4 us per creature at 1M and do not depend on trial length. The
kernel cost falls with trial length. So at 60 s the kernel share grows and the
CPU pipeline share stays a fixed 3 to 4 us per creature of wall. Both terms
have to move.

## Baseline measurements

All commands in this section ran under:

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 ...

The 3M 60 s rows are from `docs/performance-log.md` (2026-09-26) and are
repeated here for the budget above. The 5 s rows are new.

| run | trial | population | wall per gen | end to end | eval stage | archive | breeding | GPU busy | CPU busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GUI, 3M, checks on | 60 s | 3,000,000 | 90.056 s | 45,373/s (2 gen) | 68.351 s | 11.640 s | 9.888 s | 112,126 std/s | 20,765 std/s |
| GUI, 1M, checks on | 5 s | 1,000,000 | 5.628 s | 193,476/s | 213,184/s | 1.046 to 1.631 s | 2.037 to 2.340 s | 1,221,507 std/s | 223,955 std/s |
| GUI, 300k, checks on | 5 s | 300,000 | 2.911 s | 114,159/s | 120,992/s | 0.612 to 1.514 s | 0.730 to 0.871 s | 918,629 std/s | 67,538 std/s |

The GUI stage columns sum to the generation wall. In the continuous path they
are not independent device timings; the device busy lines are the cleaner
compute numbers, and they count standard trials only, so their rate includes
the busy time of check work in the denominator.

The `benchmark` subcommand, one generation, duration 5 s, output under
`/tmp/opencode/perfcamp/`:

| population | checks | creation | evaluation wall | generation wall | evaluations/s | population bytes | GPU allocation |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100,000 | on | 0.056 s | 0.748 s | 0.960 s | 133,724 | 47.1 MiB | 82.7 MiB |
| 1,000,000 | on | 0.521 s | 7.696 s | 9.620 s | 129,936 | 388.8 MiB | 99.5 MiB |
| 100,000 | off | 0.057 s | 0.278 s | 0.489 s | 359,967 | 47.1 MiB | 71.9 MiB |
| 1,000,000 | off | 0.532 s | 1.422 s | 3.333 s | 703,183 | 388.8 MiB | 88.7 MiB |
| 100,000, `--cpu` | on | 0.056 s | 1.229 s | 1.434 s | 81,346 | 47.1 MiB | 89.0 MiB |

The `--cpu` row adds a separate 6-thread CPU pass: 0.573 s for 100k at 5 s,
or 174,486 creatures/s. That CPU-only rate is about 14.5k/s at 60 s, close to
the 20,765/s the 3M GUI session measured on the evolved mix.

`eval-bench` on `runs/perf-100k.evo`, 60 s trials. The default path runs
checks on every creature (`Scheduler::evaluate` has no archive to compare
against, so every result counts as a contender). The standard-only path sets
`EVOLUTION_ROBUST_TRIALS=1`.

| path | engines | checks | creatures/s | note |
|---|---|---:|---:|---|
| GPU only, standard | RTX 4060 | off | 92,632 (median), 94,933 (best) | wall, not device busy |
| GPU + CPU, standard | RTX 4060 + 6 threads | off | 39,775 (median), 46,971 (best) | noisy, CPU tail dominates |
| GPU + CPU, checks | RTX 4060 + 6 threads | on | 7,439 (median), 13,247 (best) | very noisy, 3 rounds only |
| GPU only, standard, grow 8 | RTX 4060 | off | 24,451 (median), 36,768 (best) | 8-node bodies |
| GPU only, standard, grow 16 | RTX 4060 | off | 7,712 (median) | 16-node bodies |
| GPU + CPU, standard, grow 16 | RTX 4060 + 6 threads | off | 2,474 | 16-node bodies |

Two facts to carry into the campaign. Kernel cost grows about 12x when bodies
go from 4.68 to 16 nodes, so evolved populations are far more expensive than
the first generation. And the CPU engine is the tail for large bodies: at 16
nodes the mixed path is 3.1x slower than the GPU alone.

The check share table from the paired 3M A/B (one generation, 60 s trials) is
the cleanest check-cost number:

| metric | checks on | checks off | difference |
|---|---:|---:|---:|
| generation wall | 43.405 s | 21.241 s | +104.3% |
| end-to-end rate | 69,117/s | 141,238/s | 2.04x slower |
| GPU busy | 40.645 s | 18.718 s | +117.1% |
| CPU busy | 41.103 s | 13.880 s | +196.1% |
| packing | 2.219 s | 1.410 s | +0.809 s |

## Knob sweeps

Method: one variable at a time, same checkpoint, same 60 s duration,
`eval-bench --repeat 1` per process, round-robin over configs in 7 rounds
(3 for the checks-on set), rotating the starting config each round. Medians
and bests are wall creatures/s. The binary, checkpoint, and load range are
stated above.

Standard-only, GPU only (reliable kernel signal):

| config | n | median | best | worst |
|---|---:|---:|---:|---:|
| default (chunk 64, batch 100k, wg 32) | 7 | 92,632 | 94,933 | 62,787 |
| `EVOLUTION_GPU_CHUNK=16` | 7 | 86,426 | 89,289 | 82,054 |
| `EVOLUTION_GPU_CHUNK=32` | 7 | 88,200 | 91,334 | 47,885 |
| `EVOLUTION_GPU_CHUNK=64` | 7 | 92,537 | 95,527 | 42,695 |
| `EVOLUTION_GPU_CHUNK=128` | 7 | 92,573 | 95,919 | 53,194 |
| `EVOLUTION_GPU_CHUNK=256` | 7 | 90,967 | 97,134 | 81,116 |
| `EVOLUTION_GPU_BATCH=8192` | 7 | 66,326 | 68,963 | 61,724 |
| `EVOLUTION_GPU_BATCH=16384` | 7 | 71,252 | 75,089 | 38,654 |
| `EVOLUTION_GPU_BATCH=32768` | 7 | 79,930 | 82,462 | 31,964 |
| `EVOLUTION_GPU_BATCH=65536` | 7 | 86,698 | 90,394 | 36,402 |
| `EVOLUTION_GPU_BATCH=100000` | 7 | 94,056 | 95,358 | 91,301 |
| `EVOLUTION_GPU_BATCH=200000` | 7 | 92,357 | 95,649 | 86,074 |
| `EVOLUTION_LANE_WG=64` | 7 | 92,456 | 95,072 | 84,018 |
| `EVOLUTION_KERNEL=workgroup` + `EVOLUTION_PIPELINE_CHUNK=4096` | 7 | 93,137 | 95,822 | 86,971 |

Standard-only, GPU plus 6 CPU threads (noisy, load 6.5 to 16.4):

| config | n | median | best |
|---|---:|---:|---:|
| default | 7 | 39,775 | 46,971 |
| `EVOLUTION_GPU_BATCH=8192` | 7 | 57,429 | 58,275 |
| `EVOLUTION_GPU_BATCH=16384` | 7 | 25,930 | 29,231 |
| `EVOLUTION_GPU_BATCH=32768` | 7 | 28,150 | 29,645 |
| `EVOLUTION_GPU_BATCH=65536` | 7 | 32,395 | 33,878 |
| `EVOLUTION_GPU_BATCH=100000` | 7 | 45,212 | 45,773 |
| `EVOLUTION_GPU_BATCH=200000` | 7 | 38,326 | 46,175 |
| `EVOLUTION_LANE_WG=64` | 7 | 42,284 | 46,223 |
| `EVOLUTION_GPU_CHUNK=128` | 7 | 40,436 | 45,882 |

Checks on, GPU plus 6 CPU threads (very noisy, 3 rounds, load 8.9 to 15.8):

| config | n | median | best |
|---|---:|---:|---:|
| default | 3 | 7,439 | 13,247 |
| `EVOLUTION_GPU_CHUNK=128` | 3 | 12,830 | 13,733 |
| `EVOLUTION_GPU_BATCH=8192` | 3 | 8,358 | 10,186 |
| `EVOLUTION_GPU_BATCH=32768` | 3 | 3,962 | 4,224 |
| `EVOLUTION_LANE_WG=64` | 3 | 7,976 | 12,654 |

Findings and winners:

- `EVOLUTION_GPU_BATCH`: 100,000 is the winner, which is the shipped
  throughput default. 8,192 costs about 28%, 16,384 to 65,536 costs 6 to 23%.
  200,000 is capped near 124,000 by the 4096 MiB GPU budget and is equal to
  100,000 within noise. Winner: keep 100,000.
- `EVOLUTION_GPU_CHUNK`: 64, 128, and 256 are equal within noise (91 to
  93k median). 16 and 32 are 4 to 7% lower. Winner: keep 64.
- `EVOLUTION_LANE_WG`: 32 and 64 are equal on small bodies (92.6k vs 92.5k).
  Winner: keep 32, because the shader stats below show 64 doubles the shared
  memory per block and the largest capacity needs 128 KiB.
- `EVOLUTION_KERNEL` and `EVOLUTION_PIPELINE_CHUNK` are dead variables. The
  last reader was removed with the legacy wgpu kernels in `bd4c126`, and
  setting both measured equal to default (93,137 vs 92,632, inside noise).
  `bench/noisy_bench.sh` still writes them, which should be cleaned up by
  whoever owns that file.
- Responsive mode (`throughput = false`, batch 8,192) pays about 30% on the
  kernel path. The shipped default is throughput mode, so this only matters
  for item 51 (keep the worker responsive) and for anyone who built with
  `EVOLUTION_BENCH_RESPONSIVE`.
- The checks-on set is too noisy to rank. `chunk 128` led two of three
  rounds, but with a 2 to 3x spread inside a config no default change is
  justified.
- In the noisy mixed standard table, batch 8,192 had the best median (57,429
  against 39,775 default), which contradicts the GPU-only result. That table
  is not rankable, but it is a reminder to re-measure batch size once the CPU
  tail is smaller.
- No `src/gpu.rs` constant changed. The measured plateau already includes the
  shipped defaults, so the numbers above are the deliverable.

## Shader stats

`examples/shader_stats.rs` on the RTX 4060, driver 580.178.04, via
`VK_KHR_pipeline_executable_properties`. Register and shared numbers per
kernel capacity for the standard fidelity. The example was run once per
workgroup size with `EVOLUTION_LANE_WG=32` and `=64`.

| capacity | wg 32 regs | wg 32 shared | wg 64 regs | wg 64 shared | binary wg32 / wg64 |
|---:|---:|---:|---:|---:|---:|
| 3 | 128 | 5,504 B | 128 | 11,008 B | 48,128 / 51,200 |
| 4 | 128 | 6,272 B | 128 | 12,544 B | 49,536 / 53,632 |
| 5 | 155 | 5,120 B | 155 | 10,240 B | 50,560 / 55,680 |
| 6 | 153 | 6,144 B | 153 | 12,288 B | 52,608 / 58,752 |
| 7 | 155 | 7,168 B | 155 | 14,336 B | 53,120 / 60,288 |
| 8 | 147 | 8,192 B | 147 | 16,384 B | 54,656 / 62,848 |
| 12 | 155 | 12,288 B | 155 | 24,576 B | 60,672 / 72,960 |
| 16 | 143 | 16,384 B | 143 | 32,768 B | 63,232 / 79,616 |
| 24 | 141 | 24,576 B | 141 | 49,152 B | 71,040 / 95,616 |
| 32 | 141 | 32,768 B | 141 | 65,536 B | 79,488 / 112,256 |
| 48 | 141 | 49,152 B | 141 | 98,304 B | 96,640 / 145,792 |
| 64 | 141 | 65,536 B | 128 | 131,072 B | 113,664 / 176,128 |

Register occupancy estimate for compute capability 8.9, with 65,536 registers
per SM and a 1,536-thread SM limit: 128 regs is 16 warps or 33%, 141 to 143 is
14 warps or 29%, 147 is 13 warps or 27%, and 153 to 155 is 13 warps or 27%.
That matches the earlier Nsight reading of about 120 registers and about 30%
warp slots. The shared numbers include more than the four node-state arrays
(for example capacity 3 reports 5,504 B against 3,072 B for the arrays), so
the kernel keeps scratch there too.

The shared column is a second ceiling. `vulkaninfo` reports a 49,152 B
`maxComputeSharedMemorySize` for this device, yet the driver compiled the
131,072 B capacity-64 variant, so the real per-SM budget is not known from
that field. If an SM can hold about 100 KiB, capacity 24 and up allow only one
to three resident blocks per SM, which would put large-body occupancy far
below the register estimate. Measuring real achieved occupancy with `nsys`
GPU metrics is the first open item in the occupancy group below.

## Optimization candidates

Each entry has a one-line measure and a rough size (S under a day, M a few
days, L a structural change). Items marked with an AGENTS number are seeded
from the backlog list in `AGENTS.md`.

### Kernel occupancy and register pressure

1. Cut registers per thread toward 96. Measure: shader stats plus eval-bench
   standard GPU-only on `perf-100k.evo`. Size: L.
2. Lower register pressure by splitting physics and descriptor accumulation
   into two kernels. Measure: registers and rate at grow 16. Size: L.
3. Move per-creature node and muscle constants to a uniform or constant
   buffer. Measure: registers and rate. Size: M.
4. Pad the shared node arrays to kill bank conflicts. Measure: rate at fixed
   workload. Size: S.
5. Reuse one shared scratch array instead of separate `old` and `scr`
   buffers. Measure: shared bytes per capacity and grow-16 rate. Size: M.
6. Use 64-lane blocks only for capacities at or below 16 and 32 elsewhere.
   Measure: rate per capacity bucket. Size: S.
7. Put two small-bodied creatures in one lane with paired vector ops.
   Measure: rate on `perf-100k.evo`. Size: L.
8. Process several creatures per block so one node sweep serves all of them.
   Measure: rate and registers. Size: L.
9. Replace per-pass shared-memory barriers with subgroup operations. Measure:
   rate at grow 16 and shader stats. Size: M.
10. Fuse the bone passes into one sweep with fewer barriers. Measure: rate
    plus `tests/engine_agreement.rs`. Size: M.
11. Try one bone pass instead of the default two at equal agreement. Measure:
    rate plus agreement tolerance. Size: M.
12. Retune velocity passes and `dt` against the agreement bound. Measure:
    rate plus agreement. Size: M.
13. Replace the cosine polynomial with a shared lookup table. Measure:
    `EVOLUTION_EXACT_COS` as the control and rate. Size: S.
14. Use 32-bit indices throughout the inner loop. Measure: registers and
    rate. Size: S.
15. Move per-dispatch parameters to push constants. Measure: registers and
    rate. Size: S.
16. Precompute muscle coefficients once per unit instead of per step.
    Measure: rate and a dump comparison. Size: M.
17. Drop descriptor fields the archive never reads from the hot loop.
    Measure: registers and rate. Size: S.
18. Compute final fitness in the kernel and read back one f32 per creature.
    Measure: readback bytes and rate. Size: S.
19. Trim `CAPACITIES` to the observed node histogram. Measure: rate per
    bucket on an evolved checkpoint. Size: S.
20. Measure achieved occupancy and SM issue with `nsys` to confirm the
    register model. Measure: nsys GPU metrics on a 10 s run. Size: S.

### Lane compaction after falls

21. Compact fallen lanes between dispatches (AGENTS 47). Measure: GPU busy
    seconds and rate on a 60 s evolved population. Size: L.
22. Freeze terminal descriptor state in both engines so compaction cannot
    change a score (AGENTS 48 prerequisite). Measure: bit-compare dumps and
    `tests/early_exit.rs`. Size: M.
23. Compact per capacity bucket instead of per dispatch. Measure: rate on the
    3M evolved mix. Size: M.
24. Skip lanes whose body has fallen with a live-lane mask, without moving
    data. Measure: rate at 20 s and 60 s. Size: M.
25. Warp-level early exit when every real lane has fallen. Measure: skipped
    step share and rate. Size: M.
26. Group-level early exit in the GPU kernel to mirror
    `EVOLUTION_EARLY_EXIT`. Measure: step share and dump compare. Size: M.
27. Compact at fixed step windows, for example every 10% of the trial.
    Measure: step share and rate. Size: M.
28. Reorder capacity buckets so small bodies share dispatches after a long
    trial. Measure: rate. Size: S.
29. Track per-bucket fall histograms to size the compaction window. Measure:
    GPU counters in the stage log. Size: S.
30. Land the CPU early exit as default once terminal freeze exists. Measure:
    paired runs plus `examples/first_generation`. Size: M.

### Work batching and dispatch overlap

31. Raise the GPU batch cap once the per-creature buffer shrinks; the current
    124k cap showed no gain over 100k. Measure: eval-bench at 100k, 124k, 200k.
    Size: S.
32. Deepen async submission from two slots to three or four. Measure: GPU
    idle gaps and rate. Size: S.
33. Overlap archive and breeding with the next generation's GPU work. Measure:
    stage wall against the sum of the columns. Size: L.
34. Stream offspring slices to the GPU as breeding produces them. Measure:
    stage log and rate. Size: M.
35. Remove the stage barrier between evaluation and archive insertion.
    Measure: stage log wall. Size: L.
36. Double-buffer score and metric arrays across generations. Measure: RSS
    and stage log. Size: M.
37. Size units from each device's measured rate every batch (AGENTS 56).
    Measure: stage log and device busy. Size: S.
38. Submit check work in the same dispatch as standard work where capacity
    allows. Measure: GPU busy and wall. Size: M.
39. Submit one unit per capacity bucket across the generation. Measure:
    packing seconds. Size: S.
40. Stop cloning populations per unit; reference the parent arena. Measure:
    packing seconds and RSS. Size: M.
41. Keep descriptor sets alive per capacity instead of allocating per unit.
    Measure: packing seconds. Size: S.
42. Adapt the step range to GPU queue depth. Measure: nsys gaps and rate.
    Size: S.
43. Reuse readback buffers in a ring. Measure: allocated bytes and rate.
    Size: S.
44. Expose packing and check tax in the GUI status while a generation runs.
    Measure: status line versus stage log. Size: S.
45. Batch small units across engines to remove per-unit fence waits. Measure:
    device busy and wall. Size: M.

### Fine-check cost and successive halving

46. Successive halving: 10 s screen, 60 s trials for survivors (AGENTS 54).
    Measure: paired best and QD plus device busy. Size: L.
47. End a trial once it cannot beat its cell elite (AGENTS 86). Measure:
    rate and QD. Size: M.
48. Check only the top K per cell instead of every entrant. Measure: check
    count and QD. Size: M.
49. Escalate to the fine check only when the standard score is close to an
    elite. Measure: check share and QD. Size: M.
50. Warm-start the fine check from the standard trajectory. Measure:
    agreement and cost. Size: M.
51. Shrink the check perturbation to the smallest that agreement tolerates.
    Measure: `tests/engine_agreement.rs` and check count. Size: S.
52. Try a 2x rate, 2x pass fine check with a measured rank correlation.
    Measure: paired best and QD. Size: M.
53. Deduplicate identical creatures before checking. Measure: check count.
    Size: S.
54. Spread pending checks across generations instead of holding a round.
    Measure: wall and QD. Size: M.
55. Push more checks onto the CPU when the GPU has standard work queued.
    Measure: CPU busy share and wall. Size: S.
56. Rank the generation first and check only final survivors. Measure: check
    count and QD. Size: M.
57. Triage by margin to the elite with a cheap uncertainty estimate. Measure:
    check count and QD. Size: M.
58. Report the check tax separately from standard work in the benchmark CSV.
    Measure: CSV versus device busy. Size: S.

### CPU archive, breeding, and CMA cost

59. Profile the sub-phases of `archive_slots` (a worker is adding this
    behind `EVOLUTION_PROFILE_BREED`). Measure: seconds per phase. Size: S.
60. Parallelize archive insertion across islands with rayon. Measure: archive
    seconds at 1M. Size: M.
61. Batch CMA sample updates into matrix operations. Measure: archive
    seconds. Size: M.
62. Reuse candidate tables across generations instead of allocating per
    batch. Measure: archive seconds and RSS. Size: S.
63. Store behavior descriptors and scores as struct of arrays. Measure:
    archive seconds. Size: M.
64. Replace hash maps in the hot path with dense vectors or arenas. Measure:
    archive seconds. Size: S.
65. Incremental novelty search with a grid index. Measure: archive seconds at
    1M. Size: M.
66. Batch RNG use in offspring creation. Measure: breeding seconds. Size: S.
67. Write offspring in place instead of cloning parent vectors. Measure:
    breeding seconds and RSS. Size: M.
68. Move emitter feedback out of the per-creature loop. Measure: archive
    seconds. Size: S.
69. Defer lineage pruning to the end of the generation. Measure: archive
    seconds. Size: S.
70. Run archive and breeding on a dedicated thread with a result queue.
    Measure: stage wall against the serial sum. Size: L.
71. Narrow per-creature metadata (scores, parent scores, ranks) to u32.
    Measure: RSS and archive seconds. Size: M.
72. Compact the population arena below 400 B per creature. Measure: RSS at
    3M. Size: M.
73. Stop recomputing `refresh_behavior_scores` for every island every
    generation. Measure: archive seconds. Size: M.
74. Skip archive work for creatures that cannot enter, using a cheap
    prefilter. Measure: archive seconds and QD. Size: M.

### Allocation and memory layout

75. Move creatures to struct of arrays to speed GPU packing. Measure: packing
    seconds and rate. Size: M.
76. Reuse GPU staging buffers across generations. Measure: allocated bytes
    and rate. Size: S.
77. Reuse the population arena across generations without reallocation.
    Measure: stage log and RSS. Size: S.
78. Pool `Arc<Population>` submissions instead of allocating per unit.
    Measure: packing seconds. Size: S.
79. Pack the readback result tighter than the current `GpuResult`. Measure:
    readback bytes. Size: S.
80. Map readback memory persistently. Measure: readback time. Size: M.
81. Prefault the 3M arena once at startup. Measure: first generation time.
    Size: S.
82. Tune `CAPACITIES` to the node histogram of evolved populations. Measure:
    rate per bucket. Size: S.
83. Replace per-slot vectors with one arena allocation. Measure: RSS and
    allocation time. Size: M.

### Checkpoint and snapshot cost

84. Shrink checkpoints by dropping regenerable data (AGENTS 62). Measure:
    checkpoint MiB. Size: M.
85. Delta or incremental autosaves. Measure: autosave seconds. Size: L.
86. Compress checkpoint blocks with lz4 or zstd. Measure: MiB and save
    seconds. Size: S.
87. Memory-map the checkpoint write. Measure: save seconds. Size: M.
88. Track snapshot-copy seconds in the stage log (AGENTS 63). Measure: stage
    log. Size: S.
89. Snapshot by copy-on-write `Arc` instead of a deep copy. Measure: save
    seconds and RSS. Size: M.
90. Autosave only in a GPU-idle window between generations. Measure: wall
    impact. Size: M.
91. Store genomes in u16 or bitfields. Measure: checkpoint MiB. Size: M.
92. Drop elite bodies duplicated across islands from the file. Measure:
    checkpoint MiB. Size: M.

### GUI frame cost

93. Decouple repaint from worker updates; repaint at 60 Hz only. Measure: FPS
    and frame p99. Size: S.
94. Send snapshot deltas to the UI instead of full copies (AGENTS 58).
    Measure: worker seconds and FPS. Size: M.
95. Recompute the archive heatmap only when the archive changes. Measure:
    frame p99. Size: S.
96. Instance-draw creature bodies. Measure: frame p99 at 3M. Size: M.
97. Throttle history and lineage charts while evolving. Measure: frame p99.
    Size: S.
98. Remove per-frame allocations from the paint loop. Measure: frame p99.
    Size: S.
99. Render the scene at a lower resolution while evolving. Measure: frame
    p99. Size: S.
100. Skip work for hidden tabs (race, lineage, history). Measure: frame p99.
    Size: S.
101. Profile one frame with egui timings and record the breakdown. Measure:
    frame p99 plus a trace. Size: S.

### Measurement infrastructure

102. Keep a fixed kernel checkpoint (this campaign uses `runs/perf-100k.evo`).
    Measure: eval-bench median reproducibility. Size: S.
103. Add a script or subcommand that runs the interleaved sweep and prints
    medians. Measure: wall time and spread. Size: M.
104. Add fine-fidelity and checks-off modes to `eval-bench`. Measure: rate
    and check share. Size: S.
105. Report per-capacity timing in the benchmark output. Measure: GPU seconds
    per bucket. Size: S.
106. Record load average, GPU clocks, and the binary hash in every artifact.
    Measure: artifact header versus noise. Size: S.
107. Add a paired A/B mode for two flags or checkpoints. Measure: per-seed
    deltas. Size: S.
108. Put packing seconds and check share in the worker summary. Measure:
    summary versus stage log. Size: S.
109. Add an nsys wrapper with kernel and stage markers. Measure: trace
    artifacts. Size: S.
110. Sample RSS and VRAM during runs. Measure: columns in the stage log.
    Size: S.
111. Establish a three-seed, 100-generation QD regression baseline. Measure:
    `examples/search_ab.rs` deltas. Size: M.
112. Add a microbenchmark gate to CI that fails on a large kernel regression.
    Measure: eval-bench median versus threshold. Size: M.

## Open questions

- Where is the wall between GPU busy time and worker wall time at 3M? The
  stage log columns sum to the generation wall, but the device busy totals
  suggest overlap that the wall does not show. A trace with nsys would settle
  the real serial fraction.
- What is the real per-SM shared memory budget on the RTX 4060? The Vulkan
  field and the compiled 128 KiB block disagree.
- How much of the 43 to 55% post-fall share is in the small-body buckets,
  where compaction is easiest, versus the large buckets, where shared memory
  already limits occupancy?
- Does `EVOLUTION_GPU_BATCH` above 100k help once the batch cap is raised or
  the per-creature buffer shrinks? The 200k run hit the cap.
