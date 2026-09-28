# Phase 0 measurements on the RTX 4060 Laptop GPU

Measured 2026-09-28 on this laptop's RTX 4060 Laptop GPU (AD107, 24 SMs, 48 warps per SM as reported by the driver). These numbers replace estimates in `docs/data-architecture.md` sections 3 and 11 and in `docs/hpc-assessment.md` sections 2, 3 and 9. Every number in the tables is measured unless its row says derived.

## Method

The microbenchmarks are in `tools/gpu-micro/`, a standalone crate that is not part of the game's build. It opens only the Vulkan device whose name contains "RTX 4060" and fails otherwise. Kernels are WGSL, compiled to SPIR-V 1.3 with naga 30 the same way `src/vk_engine.rs` does it, so loads have no bounds checks, as in the game. Each repetition is one dispatch timed with Vulkan timestamp queries. The tool first sizes the work so one dispatch lasts 90 to 125 ms, then warms up for 0.6 to 1.5 s, then times 5 repetitions (3 in some sweeps). The main numbers come from 2 separate runs. Tables give the median of each run and the range over all repetitions in brackets. A child `nvidia-smi` sampled the SM clock, memory clock, power and temperature every 100 ms. Per-cycle numbers divide by the median SM clock sampled during each repetition, so a cycle is an SM clock cycle. Buffers hold hashed words; a constant fill measured the same bandwidth at 64 MiB and 1 GiB. Every GPU run held the shared GPU lock. Two other agents ran their own GPU work between my runs, so the GPU started each run from a different temperature (54 to 78 C). The clocks were not pinned (that needs root). The SM clock stayed at 2,490 to 2,505 MHz in every run except the saturated L2 runs, which drew 85 to 94 W and ran at 2,385 to 2,430 MHz. The memory clock read 8,001 MHz in every sample.

The NVIDIA driver reports statistics through `VK_KHR_pipeline_executable_properties` (registers, shared memory, binary size) but no internal representation, so I could not read SASS. Where the tables say how many instructions an operation compiles to, that comes from the binary size: 16 bytes per instruction.

## 1. L2 bandwidth

WGSL storage-buffer loads go through the SM's L1. At 8 MiB some of the rereads hit in the 24 L1 caches (128 KB each), which inflates the number. So the L2 rows use a copy of the kernel whose source buffer is decorated `Coherent` in the SPIR-V (`--coherent 1`). The driver then serves those loads from L2. An nsys profile of the 8 MiB coherent read showed a 100% L2 hit rate and 0% DRAM use.

Setup: 144 workgroups of 256 threads (48 warps per SM), `vec4<u32>` loads, 4 independent loads per thread per loop iteration.

| test | GB/s, run medians [range] | bytes per cycle, whole GPU | bytes per cycle per SM | SM clock | power |
|---|---:|---:|---:|---:|---:|
| read, 8 MiB | 1,567 and 1,558 [1,552 to 1,575] | 645 and 642 [639 to 658] | 26.9 and 26.8 | 2,370 to 2,430 MHz | 91 W |
| read, 16 MiB | 1,601 and 1,594 [1,588 to 1,616] | 659 and 660 [655 to 669] | 27.5 and 27.5 | 2,408 to 2,430 MHz | 89 to 90 W |
| read plus write (copy), 8 MiB footprint | 1,579 and 1,582 [1,545 to 1,606] | 657 and 653 [647 to 665] | 27.4 and 27.2 | 2,325 to 2,430 MHz | 89 to 94 W |
| read plus write (copy), 16 MiB footprint | 1,583 and 1,584 [1,574 to 1,608] | 661 and 660 [654 to 671] | 27.5 and 27.5 | 2,385 to 2,415 MHz | 85 to 89 W |

Two runs of 5 repetitions per row. A copy counts the bytes read plus the bytes written.

The limit is on the SM side. With fewer SMs busy, each SM moves more (one 1,024-thread workgroup per SM, 8 MiB coherent read, one run of 5 repetitions each):

| SMs busy | GB/s | bytes per cycle, whole GPU | bytes per cycle per busy SM |
|---:|---:|---:|---:|
| 6 | 478 | 192 | 32.0 |
| 12 | 950 | 382 | 31.8 |
| 24 | 1,541 | 652 | 27.2 |

One SM pulls at most about 32 bytes per cycle from L2. With all 24 SMs the GPU reaches about 655 bytes per cycle. nsys reported the L2 at 72 to 73% of its own peak at that point (`L2 Throughput`), so the L2 slices have headroom that shader loads did not reach.

Default (non-coherent) loads over larger footprints, one run of 3 repetitions each: 16 MiB 1,651 GB/s, 24 MiB 1,631, 32 MiB 1,636, 64 MiB 1,541 (at 2,340 MHz). At 64 MiB nsys showed DRAM at 97% of its peak and an 83% L2 hit rate, so rows above 32 MiB mix L2 and DRAM.

## 2. DRAM bandwidth

1 GiB buffer, same kernel with default loads, 3 runs (3, 5 and 5 repetitions) for reads and 2 runs of 5 for copies.

| test | GB/s, run medians [range] | bytes per SM cycle | bytes per memory clock (8,001 MHz) | share of the 256 GB/s spec | SM clock, power |
|---|---:|---:|---:|---:|---|
| read | 250.5, 250.5 and 250.6 [250.5 to 250.6] | 100.6 (at 2,490 MHz) | 31.3 | 98% | 2,490 to 2,505 MHz, 65 to 71 W |
| read plus write (copy) | 243.3 and 244.1 [242.7 to 244.5] | 97.7 and 98.0 | 30.5 | 95% | 2,490 MHz, 71 W |

## 3. Shared-memory throughput

The kernel mimics the creature kernel's node arrays: `vec2<f32>` values laid out [index][lane] (`array<array<vec2<f32>, 32>, rows>`, indexed `pos[row][lane]`), one-warp workgroups (workgroup size 32). Each loop iteration loads (or stores) 16 rows starting at a base that moves one row per iteration, so every access is one warp reading 256 contiguous bytes. The 24-row version uses 6,144 B per workgroup, as the creature kernel does at 8 nodes. The 12-row version uses 3,072 B so that 24 workgroups fit per SM. "Saturated" launches 384 workgroups per SM in waves, so every SM holds as many as fit (16 for 6,144 B).

| workgroups (warps) launched per SM | load, 6,144 B (B per cycle per SM) | load, 3,072 B | store, 6,144 B | store, 3,072 B |
|---:|---:|---:|---:|---:|
| 1 | 32.8 | 20.7 | | |
| 2 | 62.4 | 40.5 | | |
| 4 | 87.6 | 64.4 | 94.2 | |
| 8 | 124.3 [111.8 to 124.3] | 126.0 | 117.4 | |
| 12 | 123.7 | 116.8 | 118.5 | |
| 14 | 120.6 (2 runs: 120.6, 120.6) | | 111.4 | |
| 16 | 122.7 | 122.1 | | 119.9 |
| 20 | | 121.7 | | |
| 24 | 125.3 (only 16 fit) | 123.8 (2 runs: 123.8, 123.8) | | 121.9 |
| saturated | 127.4 (2 runs: 127.4, 127.4) | 127.4 (2 runs: 127.4, 127.3) | 127.5 | 127.5 |

Each cell is the median of 5 repetitions. The range inside a run was 3.2% or less, except the one cell that shows its range. SM clock 2,490 to 2,505 MHz, 33 to 71 W.

Saturated, an SM moves 127.4 bytes per cycle: 0.498 warp accesses of 256 bytes per cycle, one 128-byte wavefront per cycle. An nsys profile of the saturated load run read 100% on `L1 Shared+Attribute Data-Stage Throughput`, so 100% on that nsys row means 128 bytes per cycle per SM for this access pattern. This test has 16 independent loads per warp in flight, so 8 warps per SM already come within 3% of the peak. The creature kernel's loads mostly depend on each other and need more warps for the same rate.

## 4. Latencies by dependent chains

One thread runs a dependent chain. Memory chains are pointer chases with one element every 128 bytes. The cycle counts include the loop's compare and branch, which the ALU rows show is about 0.4 cycles per operation (25 cycles per 64 operations). Each row is 2 runs of 5 repetitions at 2,490 to 2,505 MHz, 24 to 26 W. The range inside every run was 0.1% or less, and the two runs agree to 0.5%.

| chain | ns per step | cycles per step | published Ada figure the design used |
|---|---:|---:|---:|
| shared-memory load (`j = sh[j]`, 4 KB array) | 9.77 | 24.3 | 30 |
| L1 hit (16 KiB and 64 KiB regions) | 24.55 to 24.56 | 61.1 to 61.2 | 43 |
| L2 hit, 1 MiB region, random order | 115.4 to 115.5 | 287.5 | 273 |
| L2 hit, 8 MiB region, random order | 118.9 | 296.1 to 296.2 | 273 |
| L2 hit, 8 MiB region, linear order (1 run) | 115.5 | 287.5 | |
| L2 hit, 16 and 24 MiB regions, random order | 118.7 to 119.3 | 296.8 to 297.6 | |
| DRAM, 64 MiB region, random order | 261.5 to 261.9 | 652.2 to 655.0 | 541 |
| DRAM, 1 GiB region, random order | 257.8 to 258.1 | 642.6 to 645.9 | 541 |
| DRAM, 1 GiB region, linear order (1 run) | 258.7 | 644.1 | |
| FP32 fused multiply-add, `fma(x, a, b)` | 1.76 | 4.39 | 4 |
| FP32 add, `abs(x) + a` | 1.76 | 4.39 | 4 |
| INT32 multiply-add, `x * a + b` | 1.76 | 4.39 | |
| `sqrt(x)` | 6.90 | 17.19 | |
| `inverseSqrt(x)` | 6.90 | 17.19 | |
| division plus add, `b / (x + a)` | 9.42 to 9.48 | 23.60 | |

Every memory hop also computes an address. The shared and global hops in WGSL index an array, so each is an index multiply-add or shift plus the load. That is one likely reason the L1 figure is above the published 43 cycles; without SASS I could not separate the two. The extra 9 cycles for random order at 8 MiB and above, against linear order or a 1 MiB region, look like address-translation misses. That is an inference, not a measurement.

How naga's operations reach the hardware: `sqrt` and `inverseSqrt` compile to one instruction per operation (same binary size as the FMA chain) at 17.2 cycles. WGSL division compiles to two instructions per operation, which fits a reciprocal followed by a fused multiply-add that also takes the `+ a`. So the reciprocal alone is about 23.6 minus 4.4, about 19 cycles (derived). The driver reassociates floating point: a plain `x + a` chain and a plain `b / x` chain both folded to almost nothing, which is why the table uses `abs(x) + a` and `b / (x + a)`.

## 5. Arithmetic throughput

Independent chains: 8 per thread, 8 deep per loop iteration, 1,152 workgroups of 256 threads, 48 warps resident per SM. Each row is 2 runs of 5 repetitions at 2,475 to 2,490 MHz, 34 to 77 W. The range inside every run was 0.1% or less, and the two runs agree to 0.1%.

| operation | lane operations per cycle per SM | warp instructions per cycle per SM | whole GPU at the measured clock |
|---|---:|---:|---:|
| FP32 fused multiply-add | 110.5 | 3.45 | 6.60 T FMA/s (13.2 TFLOPS) |
| FP32 add | 112.0 | 3.50 | 6.66 to 6.70 T/s |
| INT32 multiply-add | 64.0 | 2.00 | 3.80 to 3.82 T/s |
| `sqrt` | 16.0 | 0.50 | 956 G/s |
| `inverseSqrt` | 16.0 | 0.50 | 956 G/s |
| division plus add | 15.9 | 0.50 | 949 G/s |

FP32 reached 86% of the 128 lanes per cycle per SM that the hardware documents. Loop overhead explains at most 5 points of the gap; I did not find the rest. INT32 runs at exactly half the FP32 peak. The special-function unit delivers 16 results per cycle per SM, one eighth of the FP32 peak.

## 6. Occupancy: resident warps per SM

Driver statistics for `creature_kernel::shader_source(capacity, 32, Fidelity::standard())`, from `examples/shader_stats.rs` (measured):

| capacity (nodes) | registers per thread | shared memory per workgroup |
|---:|---:|---:|
| 3 | 128 | 4,480 B |
| 4 | 128 | 5,248 B |
| 5 | 128 | 6,272 B |
| 6 | 128 | 6,272 B |
| 7 | 149 | 5,376 B |
| 8 | 143 | 6,144 B |
| 12 | 147 | 9,216 B |
| 16 | 135 | 12,288 B |

To find what those footprints allow, a probe kernel with one-warp workgroups was launched at 1, 2, 3 and more workgroups per SM. Its register count and shared memory were set to chosen values, and one lane chased pointers in L2 so each workgroup takes the same time. The time stays flat while every workgroup is resident and jumps by 1.6 to 1.9 times at the first count that needs a second wave. One run of 3 or 5 repetitions per count; flat steps varied by under 4%.

| probe registers per thread | probe shared memory | resident warps per SM |
|---:|---:|---:|
| 16 | 128 B | 24 |
| 72 | 128 B | 24 |
| 87 | 128 B | 20 |
| 95 | 128 B | 20 |
| 103 | 128 B | 16 |
| 116 | 128 B | 16 |
| 128 | 128 B | 16 |
| 132 | 6,272 B | 12 |
| 16 | 6,144 B | 16 |
| 16 | 6,272 B | 16 |
| 16 | 9,216 B | 11 |
| 16 | 12,288 B | 8 |

One rule fits every row: warps per SM = min(24, 4 x floor(16,384 / (32 x registers rounded up to 8)), floor(102,400 / shared bytes)). The register file is split among the SM's 4 schedulers, and a warp must fit in one quarter (16,384 registers), so the whole-SM division 65,536 / (32 x registers) overstates occupancy between the steps. At 87 and 95 registers the whole-SM division gives 23 and 21, but 20 were measured. At 132 registers (136 allocated) it gives 15, but 12 were measured. One-warp workgroups cap at 24 per SM. Shared memory allows 100 KB per SM with no visible per-workgroup reservation.

Derived from that rule for the creature kernel: capacities 3 to 6 (128 registers, 6,272 B or less) allow 16 warps per SM. Capacities 7 and 8 (149 and 143 registers) allow 12, limited by registers. Capacity 12 allows 11 (shared memory) and capacity 16 allows 8 (shared memory). The performance log's "128 to 149 registers per thread, which allows 14 to 16 warps" is therefore 12 to 16. I did not run the creature kernel itself for this; the probe matches its footprint.

## 7. Today's kernel under nsys

nsys 2026.3.2 offers the `ad10x-gfxt` GPU metric set for this GPU, which has SM issue, warps, L2 throughput and a shared-memory data-stage row. I profiled `eval-bench` on `runs/evolved-3m-v26.evo` (the first 200,000 creatures, mean 6.02 nodes, 2 repeats per profile) three times, with `EVOLUTION_DEVICES=primary RAYON_NUM_THREADS=4`, sampling at 2 kHz. `eval-bench` goes through `Scheduler::evaluate`, which fine-checks every creature, so each creature runs a standard trial and a fine check. Rates under the profiler: 14,459 and 15,482; 15,530 and 15,257; 14,528 and 15,508 creatures/s, in line with the 15,519 and 15,309/s that the performance log reports for the same checkpoint without a profiler. nvidia-smi during profiles 2 and 3: SM 2,490 MHz median (2,490 to 2,505), memory 8,001 MHz, 60 to 68 W median, 79 W peak, 74 C peak.

Samples with compute in flight above 50% (about 49,000 per profile):

| metric (nsys row) | p50 | mean | p10 | p90 |
|---|---:|---:|---:|---:|
| resident warps per SM (`Active Thread Groups in SM`, one warp per workgroup) | 9 | 7.7 to 7.9 | 2 to 3 | 11 |
| warps in flight, % of 48 per SM (`Compute Warps`) | 18% | 16.2 to 16.5% | 5 to 7% | 23% |
| SM issue, % of 4 per cycle per SM (`SM Issue Active`) | 36% | 31.3 to 31.9% | 9 to 14% | 44% |
| shared-memory data stage (`L1 Shared+Attribute Data-Stage Throughput`) | 22% | 18.7 to 19.0% | 5 to 9% | 27% |
| L2 throughput (`L2 Throughput`) | 7% | 6.6 to 6.7% | 2 to 3% | 9% |
| L2 hit rate | 100% | 99.4% | 98% | 100% |
| L1 hit rate | 63% | 60.8 to 61.7% | 45% | 69 to 71% |
| L1 global data stage | 6% | 5.0 to 5.1% | 2% | 7% |
| DRAM (`VRAM Throughput`) | 1% | 1.2% | 1% | 2% |
| ALU pipe | 28 to 29% | 25.1 to 25.6% | 7 to 11% | 35% |
| FMA heavy pipe | 13% | 11.4 to 11.6% | 3 to 5% | 16% |
| FMA light pipe | 14% | 11.9 to 12.2% | 3 to 6% | 17% |
| special-function pipe | 6% | 4.9 to 5.0% | 1 to 2% | 7% |
| active threads per warp | 26.4 of 32 | 25.0 to 25.6 | 22.6 to 23.7 | 27.2 |

The p50 values agree to within one point across the three profiles; the ranges above span them.

Two conversions. First, the shared-memory row: section 3 shows that 100% on it is 128 bytes per cycle per SM, so 22% is about 28 bytes per cycle per SM (derived). Second, the L2 row: section 1's 1.6 TB/s read measured 72 to 73% on the same row, so 7% is about 10% of the bandwidth that shaders can reach (derived, approximate, because nsys builds this row from the busiest of several L2 sub-units).

Issued warp instructions over each whole profile (`SM Issue Active`, sum): 1.861e12, 1.862e12 and 1.862e12 for 400,000 creature evaluations, or 4.65 million warp instructions per creature for a standard trial plus a fine check. I did not convert this to instructions per creature-step, because I did not count the steps each creature ran.

A note on reading nsys: the `Compute Warps [Avg Warps per Cycle]` row is per TPC, and a TPC holds 2 SMs. The 64 MiB read at 48 warps per SM read 96 on it, the saturated shared-memory run at 15 warps per SM read 30, and the eval-bench profiles read 17 at p50 against 9 on the per-SM thread-group row. The performance log reports "about 24 warps in flight per SM" for eval-bench. The kernel's registers allow at most 16 per SM (section 6), so that figure was probably read from the per-TPC row. If the "10 to 12 warps per SM" of the GUI run in `docs/hpc-assessment.md` section 3.2 came from the same row, the GUI ran 5 to 6 warps per SM. I did not re-profile the GUI, so this stays open.

## Measured against the design estimates

| quantity | estimate | where the estimate is | measured here |
|---|---|---|---|
| L2 bandwidth | about 0.8 TB/s, 320 B per cycle (4090's 1,708 B per cycle scaled to 24 SMs) | data-architecture 3.5 | 1.56 to 1.60 TB/s read, 1.58 TB/s read plus write; 642 to 661 B per cycle for the GPU, 26.8 to 27.5 per SM; one SM alone reaches 32 B per cycle |
| L2 ceiling of today's layout at 520 B per creature-step | 1.54G creature-steps/s, about 1.15M creatures/s | data-architecture 3.5 | derived from the measured bandwidth: about 3.0G creature-steps/s, about 2.3M creatures/s |
| DRAM bandwidth | 256 GB/s (spec) | hpc-assessment 2.1 | 250.5 GB/s read (98%), 243 to 244 GB/s copy; 100.6 B per SM cycle, 31.3 B per memory clock |
| shared-memory throughput | 128 B per cycle per SM | data-architecture 3.5 | 127.4 B per cycle per SM saturated, loads and stores; 124 B with 8 warps of 16 independent loads each |
| shared-memory ceiling of today's layout | about 0.81M creatures/s | data-architecture 3.5 | unchanged, since the throughput matches (derived) |
| FP32 FMA latency | 4 cycles | data-architecture 3.2 | 4.39 cycles per operation in a loop (about 4 without the loop overhead) |
| shared-memory load latency | 30 cycles | data-architecture 3.2 | 24.3 cycles per dependent hop including index arithmetic |
| L1 hit latency | 43 cycles | data-architecture 3.2 | 61.1 cycles per dependent hop including index arithmetic |
| L2 hit latency | 273 cycles | data-architecture 3.2 | 287.5 cycles (1 MiB), 296 to 298 cycles (8 to 24 MiB random) |
| DRAM latency | 541 cycles | data-architecture 3.2 | 643 to 655 cycles, 258 to 262 ns |
| `sqrt`, `inverseSqrt` latency | not estimated | | 17.2 cycles |
| division latency as naga emits it | not estimated | | 23.6 cycles with one add; about 19 for the reciprocal alone (derived) |
| FP32 FMA issue | 128 lanes per cycle per SM, 15.3 TFLOPS at 2.49 GHz | hpc-assessment 2.1 | 110.5 lanes per cycle per SM (86%), 13.2 TFLOPS |
| INT32 issue | half the FP32 rate | hpc-assessment 2.1 | 64.0 lanes per cycle per SM, exactly half of 128 |
| special-function throughput | 16 per cycle per SM, unconfirmed | hpc-assessment 2.1 | 16.0 per cycle per SM (`sqrt`, `inverseSqrt`, reciprocal) |
| resident warps at 128 to 149 registers | 14 to 16 | performance log (occupancy, 2026-09-27) | 16 at 128 registers, 12 at 143 and 149 (register file split by scheduler) |
| registers and shared memory at capacity 6 and 8 | 128 to 149 registers, 6 KB at 8 nodes | hpc-assessment 3.2 | capacity 6: 128 registers, 6,272 B; capacity 8: 143 registers, 6,144 B |
| today's kernel, shared-memory use | about 25% of the ceiling (game) | data-architecture 3.5 | 22% p50, 19% mean (eval-bench, not the game) |
| today's kernel, L2 use | about 17% of the 0.8 TB/s estimate (game) | data-architecture 3.5 | 7% p50 of nsys's L2 peak, about 10% of the measured 1.6 TB/s (eval-bench, not the game) |
| today's kernel, warps in flight | 12 per SM p50 in the game; 24 in eval-bench | data-architecture 3.2, performance log | 9 per SM p50, 11 p90 (eval-bench); the earlier 24 is likely per TPC |
| today's kernel, issue | 17 to 20% (game); 44% (eval-bench) | hpc-assessment 3.2, performance log | 36% p50, 31 to 32% mean (eval-bench) |
| GPU clock under load | 2,490 MHz | hpc-assessment 2.1 | 2,490 to 2,505 MHz; 2,340 to 2,430 MHz when drawing 85 to 100 W |

## Not measured

- SASS. The driver exposes no internal representation, so instruction counts per operation come from binary sizes and the reasons for the L1 latency and the 86% FMA rate stay open.
- Shared load and store instructions per creature-step, and instructions per creature-step, the fourth Phase 0 item of data-architecture section 11. nsys gives totals per creature, and I did not count the steps each creature ran.
- Shared-memory and L2 use of today's kernel in the game. I profiled `eval-bench` only, so the game's 25% and 17% figures are not checked directly.
- Stall reasons (Nsight Graphics GPU Trace), hpc-assessment section 9 item 2.
- The L2's own peak. Shader loads reached 72 to 73% of it, limited on the SM side.
- Pinned clocks. `nvidia-smi -lgc` needs root.

## How to repeat

Build: `cd tools/gpu-micro && CARGO_BUILD_JOBS=1 nice -n 10 cargo build --release --offline`. Run each test under the GPU lock, for example `flock <lock> tools/gpu-micro/target/release/gpu-micro bandwidth --mode read --size 16m --coherent 1`. The tests are `bandwidth`, `chase`, `shared-latency`, `alu-latency`, `alu-throughput`, `shared-bw`, `occupancy` and `compile`; the options are listed at the top of `tools/gpu-micro/src/main.rs`. The nsys profiles used `nsys profile --trace=none --sample=none --cpuctxsw=none --gpu-metrics-devices=0 --gpu-metrics-set=ad10x-gfxt --gpu-metrics-frequency=2000` (10,000 for the microbenchmarks), exported to SQLite and summarized over samples with compute in flight.
