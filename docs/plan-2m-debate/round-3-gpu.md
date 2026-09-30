# Round 3, GPU domain: the kernel and the pipeline it sits in

I hold no GPU lock and ran nothing this round. Numbers are counts from the code and the round 2 files, with the measurement named where one replaces them.

## 1. The finding that reorders the design: the 9% tail

The chair's population has 9% of offspring at 17 to 32 nodes. On the lane-group kernel (one creature per 32-lane warp, tree passes level by level over 7 or 8 levels, 2 muscle rounds), such a creature costs about 9,500 warp instructions per creature-step, 300,000 executed thread-slots. The per-lane classes cost 9,000 to 17,000. In a harmonic mean over the mix, that 9% takes about 70% of the GPU's time if it stays on the lane-group kernel, and the whole population runs at 80 M creature-steps/s at 3 T. So the fallback kernel is not a fallback. Either the large bodies get a per-lane class of their own or a search rule keeps them out of the ring. I design for both: a W = 8 class that costs 20% of the rate, and ga's archive-following cap as the lever that gives it back.

## 2. The kernel

One source, maximal coordinates (physics's specification), templated on W. Every lane owns up to 4 nodes, their rods, and up to 16 muscles. W = ceil(nodes / 4): W = 2 for up to 8 nodes (70% of offspring), W = 4 for 9 to 16 (21%), W = 8 for 17 to 32 (9%). The lane code is identical across classes; only the exchange width and the muscle end lookups change. No W = 1: 4-node bodies are rare at generation 51 and W = 2 costs them 10%. The reduced-coordinate lane-group kernel stays as the scoring kernel for every class until its per-lane class passes its gate, then it is retired, so one physics remains.

Per-lane cross-lane traffic: the chordal LDL is eliminated children first; a rod whose parent rod lives on another lane exchanges one row (3 floats) by shuffle at each of the tree's lane boundaries. Each lane owns 8 / W contact rows, does its own tree solves for them, and the 8x8 direct solve runs redundantly in every lane of the creature after a row exchange (W = 2: 2 lanes x 4 rows, about 50 shuffles; W = 8: 8 x 1 row, about 120). Muscle forces on nodes of other lanes go through a private shared slot per muscle end and the owner lane sums its ends in a fixed list order (today's `ends` list), never through shared atomics, because float atomics are order-dependent and would break determinism.

Registers per lane (peak, the contact phase): persistent 59 (positions and velocities 16, header 10, masses 4, rods 5, friction anchors 8, muscle state 16 as 16-bit pairs of energy and rhythm offset), plus Delassus rows owned 32 at W = 2 or 8 at W = 8, per-contact data 24, factor 8, impulses 8, temporaries 10: about 140 at W = 2, 115 at W = 8. `__launch_bounds__(128, 4)` caps at 128; the compiler will spill 10 to 15 registers at W = 2 to local memory, which is L1 resident and cheap. If the spill exceeds 32 B per lane the W = 2 class runs at 3 blocks per SM (12 warps, 170 registers) instead; under the cap that costs little.

Shared bytes per lane: node position table for cross-lane muscle anchors 64 B, muscle end force slots 32 B, LDL row exchange 16 B, metrics 76 B per creature in lane 0's slot (38 B per lane at W = 2). About 150 B per lane, 77 KB per SM at 16 warps. Fits the 100 KB carveout.

The muscle record stays float32 and I withdraw 16-bit muscle genes as a requirement. The record splits into a per-step part (period, phase, duty, inverse duty, inverse complement, amplitude, stiffness, Hill: 32 B, read once per step for the waveform and drive target, physics accepted this cut) and a per-substep part (end lanes and nodes, anchors, cap, tendon stiffness, slack, inverse capacity: 24 B). L2 traffic at the highest row below (760 M creature-steps/s at 1 substep, 19 muscles): 760 M x 19 x 56 B = 0.8 TB/s, 21% of L2 peak. At 2 substeps and 410 M: 0.7 TB/s. Nothing needs quantizing. ga's ruling on 16-bit genes is moot for the kernel.

Take-up unpack (Model::new on the device): masses with bones, organs and muscle mass, slack lengths, strengths, joint ranges, start pose. About 3,000 instructions per creature, 1% of a 300-step trial, run once per creature by its lanes from the pinned SoA genes. Register cost: none. At take-up no trial state is live, and the unpack's 30 temporaries sit under the 59 persistent registers, far below the 140 peak of the contact phase; the allocator takes the maximum over program points. The unpack uses `__fmul_rn`, `__fadd_rn` and `__fmaf_rn` explicitly so `--fmad=true` cannot contract it, and the gate is a dump of 262k creatures' constants bit-equal to the CPU `Model::new`. Fallback if the gate fails or the kernel grows: a per-wave prologue kernel writing lane records to VRAM (0.5 GB per 262k wave, no PCIe, no CPU). The read itself: I back one DMA of the block's genes to VRAM per block (250 MB/s at 2 M/s, a copy engine, no SMs) over zero-copy reads inside the kernel; the CPU cost is the same zero and the kernel never waits on PCIe latency. os's prefetch question is then moot.

Candidate bit: lane 0 computes the descriptor bins from the metrics in its shared slot (about 30 instructions), reads the island cell's occupant fitness from the 29 KB device table (one L2 load), and sets a bit in the result. Transient, 2 registers. The bar table for ga's rungs is the same table mechanism: one load per rung.

The stub kernel gate (track T0): a kernel with the real state layout, the shared tables, a 16-muscle loop reading split records from L2, the 4-rod LDL with the W = 2 exchange, 8 Delassus tree solves and the 8x8 direct solve on synthetic data, 2 substeps, W = 2, running 300 fake steps per creature over 262k creatures. It passes when ptxas reports at most 128 registers with at most 32 B of spill, nsys shows at least 45% issue-active at 16 warps per SM, and the measured rate extrapolates to at least 250 M creature-steps/s on the mean body at 2 substeps at the clock nsys logs. Below 35% issue or 180 M the design goes to W = 1 for small bodies at 255 registers and the table below drops 20%.

## 3. Rates per class and the harmonic mean

Executed thread-slots per creature-step at 2 substeps, with the three accepted cuts (waveform once per step, ledger in flight, lagged matrix) and a divergence allowance (15% at W = 2 after sorting by exact node and muscle count, 20% at W = 4, 25% at W = 8):

| class | body | useful per step | per lane with exchange | warp instr per creature-step | executed slots | rate at 3 T | rate at 5 T |
|---|---|---:|---:|---:|---:|---:|---:|
| W = 2 | 7 nodes, 17 muscles | 6,000 | 4,100 | 255 | 8,200 | 365 M | 610 M |
| W = 4 | 11 nodes, 28 muscles | 9,400 | 3,900 | 490 | 15,600 | 190 M | 320 M |
| W = 8 | 27 nodes, 50 muscles | 19,000 | 4,800 | 1,200 | 38,400 | 78 M | 130 M |
| lane-group, for comparison | 27 nodes | | | 9,500 | 304,000 | 10 M | 16 M |

Harmonic mean over 70 / 21 / 9 at equal steps per creature: 240 M at 3 T, 400 M at 5 T. Without the 9% tail (ga's cap holding offspring at 16 nodes): 300 M and 500 M. At 1 substep multiply by 1.85 (metrics once per step): 445 M and 740 M with the tail, 555 M and 925 M without.

The mean body's own rate (365 to 610 M) is what the chair's table calls the kernel; the population runs at two thirds of it because of the tail. That is the number to design the search against.

## 4. The multiplier table, from 45 M and 480 steps

| lever | at 3 T | at 5 T | note |
|---|---:|---:|---:|
| generation-51 bodies on today's kernel | 0.6x (27 M) | 0.6x (27 M) | chair's ruling 1 |
| per-lane maximal coordinates, W = 2, 4, 8, 3 cuts, 2 substeps | 8.9x (240 M) | 14.8x (400 M) | section 3 |
| ga's body cap on offspring | 1.25x (300 M) | 1.25x (500 M) | search rule, ga |
| 1 substep (physics P1, 50% odds) | 1.85x | 1.85x | owner |
| device helpers: unpack, descriptor, candidate bit, bar loads | 0.985x | 0.985x | 1.5% of instructions |
| host tax at 2 to 3 busy cores at the cap | 0.93x | 0.93x | cpu and os |
| steps per creature: R1 to R3 | 480 to 300 | | ga and ml |
| steps per creature: R1 to R4 | 480 to 235 | | ga and ml, unsettled |

Creatures per second sustained, generation 51, after helpers and host tax:

| case | with the 9% tail, 3 T | with the tail, 5 T | tail capped, 3 T | tail capped, 5 T |
|---|---:|---:|---:|---:|
| 2 substeps, R1 to R3 | 0.73 M | 1.22 M | 0.92 M | 1.53 M |
| 2 substeps, R1 to R4 | 0.94 M | 1.56 M | 1.17 M | 1.95 M |
| 1 substep, R1 to R3 | 1.36 M | 2.26 M | 1.70 M | 2.83 M |
| 1 substep, R1 to R4 | 1.73 M | 2.89 M | 2.17 M | 3.61 M |

The row I land on: 1 substep with R1 to R4, which reaches 2 M/s at 5 T with the tail, and at 3 T only with the tail capped. With 1 substep and R1 to R3 it needs 5 T. Below the chair's ranges by 20 to 25% because of the tail and the taxes. If 1 substep fails its honesty test, the ceiling is 1.2 to 1.6 M/s at 5 T (2 substeps, R4, cap), and physics's cheap-fidelity screen adds 1.2x on top for 1.4 to 1.9 M/s. That is where it stops: 2 M/s without 1 substep needs everything else and the high power budget, and it is not a promise.

## 5. The pipeline, breeding to archive

1. The CPU breeds a block (50 ms of GPU work, 32k to 256k creatures) in cpu's fixed-array form with the counter RNG, writing genes once into a pinned SoA ring segment. Structural and parametric children alike; no device breeder.
2. The engine DMAs the segment to a VRAM gene ring (one copy engine transfer per block) and bumps the device count word of the block's class queues. Three queues per world (W = 2, 4, 8), each block sorted by exact node and muscle count so warps are uniform.
3. The persistent kernel per class runs waves of blocks; a lane group whose creature ended takes the next index from the spanning counter, unpacks the genes into its registers, and starts. The kernel drains only on a world change.
4. At the end of a creature the group writes its 80 B result with the descriptor bins and the candidate bit into the results ring in pinned host memory, with the block's sequence word after the block's last result.
5. The CPU absorbs blocks in ring order, offers candidates to the archives, runs the confirmation the verdict asks for (see section 8 for the in-kernel version), commits, and breeds the freed slots.

Determinism: a creature's result depends on its genes, its block's settings and the kernel only; lane and wave assignment carry nothing into the arithmetic, warp-uniform loop bounds mask inactive iterations, no atomics touch floats, the counter RNG keys every draw, and absorption is in ring order. One seed, one search, on one GPU.

World change: the host stops bumping the count words; running groups finish (at most 20 s of trial, 15 to 40 ms of wall) and their results are marked as entering no archive; the queues restart with retargeted blocks. Loss: the resident creatures (9,216 lanes' worth, under 20k) plus the queued blocks the ring holds, 0.3 to 1 s of work by data's rule against 26% of a generation today.

Save: CPU archives and search state only; the kernel keeps running; nothing on the device is in a save. Replay: the recording variant of the same per-lane source (RECORD = 1) on the replay slot's high-priority stream; a block slot frees every wave tail, so with 50 ms blocks on 4 to 8 streams the wait is under 25 ms plus the recording. 60 FPS: the window renders on the Radeon; the CPU stays under 3 busy cores at the cap by cpu's numbers; os's rayon-at-14, SCHED_BATCH and short UI slice rules apply; the worker absorbs blocks of 50 ms instead of 786k-creature units.

What stays out: tensor cores (3% of the step, TF32 changes bits), FP16 arithmetic (same rate as FP32), wavefront kernels (no occupancy gain), CUDA graphs (8 launches per second), Radeon physics, the lane-group kernel after T2, 16-bit muscle genes, the structural port, zero-copy reads inside the kernel (DMA per block does the same without in-kernel PCIe latency).

## 6. Tracks and gates

T0, the stub kernel, 3 days: gate in section 2. Runs under the shared lock beside the owner's game; it needs no exclusive lock except for one 30 s rate reading, which I will ask the chair to schedule with os's calibration.

T1, W = 2 class for up to 8 nodes, 1 week after T0: the per-lane source implementing physics's specification, scoring the up-to-8-node batches while the lane-group kernel keeps the rest (the engine already routes by class). Gate: p2_speed on save42.evo's up-to-8-node subset at least 150 M creature-steps/s at 2 substeps; first_generation free propulsion under 0.05 m median; physics_audit ledgers at rounding level; elite re-test ratio at 4x rate at least 0.95 median; a fixed-seed search_ab repeat bit-equal. Game playable at every merge because the class route is a table.

T2, W = 4 and W = 8, 1 week: the same source at the other widths. Gate: harmonic rate on save42.evo's full ring at least 200 M at 2 substeps; the same audits; then the lane-group kernel and its WGSL twin are deleted.

T3, the three cuts, 3 days, can land inside T1: each measured alone on p2_speed with size_report slip and the elite ratio unchanged.

T4, physics's substep ladder, from day 1 in parallel on the current kernel (zero code, then 40 and 60 lines): on a pass, SUBSTEPS = 1 becomes the default of the per-lane kernel with the realized-work ledger and anchored friction inside. Gate: honesty median at least 0.9. Needs the owner.

T5, with data and cpu, 1 week: pinned SoA genes, DMA per block, take-up unpack, results in pinned memory, spanning counter, candidate bit, 50 ms blocks, ring by latency. Gate: unpack bit-equal on 262k; pack thread gone; unhidden host share under 3%; EVOLUTION_STAGE_LOG rate flat over 20 generations with world changes and saves during the run.

T6, with ga and ml: the bar table and checkpoints in the kernel (about 40 lines), gated by their search A/B.

T7, os's calibration after the debate, rechecked on the T1 kernel with a 10-minute trace; the tables above are re-stated on the measured budget.

## 7. Asks

Physics: the specification with the lane boundary exchange spelled out (which rows cross lanes in the chordal order), and the register table at W = 2 with 4 nodes per lane. ga: the body cap on offspring, because the tail costs 20% of rate even with its own class, and the muscle-count histogram per generation, because the per-lane loop runs the warp's maximum and the packer sorts by exact count. data: the three class queues and per-block DMA in the engine API; the results ring with the descriptor bins and the candidate bit at 80 B. cpu: genes in SoA order the unpack can read in one coalesced pass per lane (node fields by node index, muscle fields by muscle index). os: the calibration, and the scheduling of T0's one exclusive 30 s reading. Chair: nobody holds the exclusive lock during rounds; I will not.

## 8. One new idea: confirmation inside the kernel

Today a new island record gets one confirmation trial at 4x rate, score = min, through a host round trip: the block waits for the verdict, the engine queues a confirmation unit, the result comes back 100 to 150 ms later. That round trip is the largest term in data's ring-depth rule (5x the p95 host chain), so it sets the ring at 0.5 to 1 s and the loss on a world change with it.

Instead the lane group that finishes a creature whose fitness beats its island's record (a second word in the 29 KB table) re-runs the same creature at 4x rate at once, in place, constants still in registers, the substep length a runtime parameter, and writes min(standard, confirm) plus both scores in the result. Records are tens per block, so the cost is tens x 4,800 steps against a block's 30 M: about 0.3% of GPU time. The host's verdict finds the confirmation already done; the remaining round trips are the rare cascade cases, a few per generation. Ring depth drops to its 0.3 s floor and a world change discards a third of what a 1 s ring would. Determinism holds because the re-run is a pure function of the creature and the table value at take-up, and the table changes only at block commits in ring order; a stale record causes an extra confirmation, never a missing one, since the host still checks. Measurement: p95 of the per-block host chain with and without (EVOLUTION_PROFILE_BREED), the ring depth data's rule produces from it, and creatures discarded per world change over a 20-generation run with a button press every 3 generations. Expected: chain p95 from 100 to 150 ms down to 40 to 60 ms, ring from 0.6 s to 0.3 s, discard per press from 1.2 M to 0.6 M creatures at 2 M/s.

A smaller one in a line: after a creature's first second, a warp-local permutation of its lanes by contact count (state swapped through shared memory at a checkpoint) would let most warps run MAXC = 2 instead of 4, about 10% of the step; the PROFILE contact histogram decides whether the count is bimodal enough to pay.
