# Data architecture for 2M creatures/s

Date: 2026-09-28. Author: Claude, at the owner's request. This document designs where the data of the evolution loop lives, what shape it has, how it moves, and how the work is orchestrated. It replaces the short sketches in sections 7.3 to 7.5 of `docs/hpc-assessment.md` and corrects two estimates made there (section 9 below). No production code changed for it. One tool was added: `examples/body_stats.rs` reads the genome table of a checkpoint and reports body sizes and warp loop efficiency.

Labels: "measured" numbers come from this machine (the performance log, or runs made for this document). "Estimate" numbers are derived in the text from measured inputs and published hardware figures. The owner's game ran on the RTX 4060 while this was written, so no new GPU measurement was taken. Phase 0 of the assessment (profiling) is where every estimate below gets checked.

## 1. Summary

The method is the usual one for HPC work: inventory every piece of data, count the bytes and accesses at each level of the memory hierarchy, find which level caps the design, then shape the data and the schedule around those caps.

What the inventory shows about today's design:

- Every creature crosses PCIe about 3.6 KB worth of data, in up to seven transfers: an upload, then a readback and a re-upload at each segment boundary. The host copies it about six times in its own memory. At 2M creatures/s that would be about 7 GB/s over PCIe and 16 to 20 GB/s of host memory traffic. Packing alone takes 0.7 to 2.7 s per 3M generation (measured as 2 to 8 s per three-generation run), and at the target a generation lasts 1.5 s.
- Inside the kernel, one creature-step makes about 1,100 shared-memory accesses and reads about 520 B from the L2 cache. Those two numbers set hard ceilings of about 0.8M creatures/s (shared-memory bandwidth) and 1.15M creatures/s (L2 bandwidth). They hold at any occupancy. Today's kernel runs at about a quarter of the first ceiling. The current layout cannot reach 2M/s however well it is tuned.
- An SM holds 384 KB of registers, shared memory and L1 in total. A 6-node creature needs 0.6 to 1 KB on chip in any formulation, once its muscles and the temporaries of a step are counted. So a lean design sits at 12 to 16 resident warps per SM. It does not reach 32 or 48, and the assessment's hint that halving state would roughly double occupancy was wrong. At 12 to 16 warps, 2M/s needs at most about 1,700 instructions per creature-step and at most 8,500 to 11,400 cycles of dependent latency per step. Today's figures are about 4,500 instructions and 86,000 cycles.

The design that follows from those numbers:

1. The whole evolution loop runs on the GPU. The host plans the search (16 B per child), draws the UI and handles files. No per-creature data crosses PCIe.
2. Genomes are stored as fixed-size records in canonical node order, one 128-B-aligned slot per creature (array of structures). A lane reads its own creature's genome once and derives every constant itself. The packed-body format and host packing go away.
3. During a trial, a creature's state never leaves the SM. Every value a step reads is in registers or shared memory, so a step makes no L1 or L2 accesses. Shared memory is used only where topology makes the indices data-dependent, and stays under about 300 accesses per step.
4. Work runs in epochs of about 100 ms. In each epoch, persistent kernels, one per body-size class, run a fixed step budget per lane. Each lane walks its own queue of creatures. It refills at fixed step boundaries and suspends its current creature on the device when the budget runs out. There are no segments, no readbacks and no tails.
5. Results are absorbed in certificate order, and cell winners are chosen by (fitness, certificate). The search is therefore deterministic for a given GPU and settings, as it is today.

Expected effect: with today's physics, stage 1 (section 10) removes host packing, segments and PCIe traffic. The assessment's estimate of 1.3 to 2x still applies to it. With the reduced-coordinate physics, the model in section 3 gives about 1.1M to 2.5M creatures/s on this laptop. The spread comes almost entirely from one number: the average stall per instruction must fall from about 19 cycles to about 6 to 8.

## 2. Inventory: the data today

### 2.1 Body sizes (measured)

`examples/body_stats.rs` on the two evolved checkpoints:

| checkpoint | creatures | nodes: mean (p50, p90, max) | bones | muscles: mean (p50, p90, p99, max) |
|---|---:|---|---:|---|
| evolved 3M, `qd::VERSION` 26, generation 9 | 3,000,000 | 6.02 (6, 7, 11) | 5.02 | 8.99 (9, 13, 19, 30) |
| evolved 100k, generation 20 | 100,000 | 6.51 (6, 8, 12) | 5.51 | 11.77 (11, 19, 26, 43) |

Every body is a tree, so bones = nodes - 1. In the 3M population, 52.5% of bodies have exactly 6 nodes and 95.8% have at most 8. Bodies grow over longer runs: the 100k run at generation 20 has more muscles and a longer tail. Muscles dominate the bytes of a creature, and their count varies the most.

### 2.2 Per-creature data and where it lives (6 nodes, 5 bones, 9 muscles)

| data | form today | bytes | lifetime | accessed |
|---|---|---:|---|---|
| genome in the population arena | `Genome` 64 B, `NodeGene` 16 B, `Bone` 28 B, `Muscle` 48 B, in four vectors | 732 | birth to the next generation boundary | breeding, compaction, `Population::subset`, packing, archive insertion, checkpoints |
| unit copy (`Population::subset`) | the same, copied per work unit | 732 | one unit | packing |
| packed batch (`LaneBatch`) | node state 32 B per node, bone constants 36 B per bone, muscle genes and state 60 B per muscle, 16 B info, 32-creature tiles | 928 plus tile padding | one segment | written to mapped GPU memory |
| GPU buffers | the same layout, device-local and host-visible (written by the CPU over PCIe) | 928 | one segment | kernel: muscles every step, bone fields in the joint-limit pass; readback at each segment end |
| result | 19 f32 | 76 | trial | host at each segment end |
| kernel registers | node constants (5 per node, sized to the bucket), bone constants (7 per bone), 19 metrics, temporaries | 512 to 596 (128 to 149 registers) | one 64-step dispatch | every step |
| kernel shared memory | `pos`, `vel`, `old` per node, laid out as [node][lane] | 144 (192 at 8 nodes) | one dispatch | every pass |

`examples/body_stats.rs` reports 1,005 B of packed GPU data per creature on the evolved 3M checkpoint, before tile padding.

### 2.3 Movement per 3M generation (estimate)

Segments pause at 2 s, at the 5 s screen and at 10 s. At each pause, the whole node, muscle and result buffers come back to the host, about 810 B per creature in the batch including the fallen ones. The survivors are repacked and written again at about 1,005 B each. Running shares: 100% in the first segment, about 65% after 2 s (38% of evolved creatures fall, most within a second), about 20% after the screen, about 19% after 10 s.

| transfer | volume | per creature of the generation |
|---|---:|---:|
| initial upload | 3.0 GB | 1,005 B |
| readback and re-upload at 2 s | 2.4 GB + 2.0 GB | |
| readback and re-upload at the screen | 1.6 GB + 0.6 GB | |
| readback and re-upload at 10 s | 0.5 GB + 0.6 GB | |
| final results | 0.04 GB | |
| total over PCIe | 10.7 GB (6.1 up, 4.5 down) | 3.6 KB |

At today's 200,000 creatures/s that is 0.7 GB/s, which is harmless. At 2M/s it is 4.1 GB/s up and 3.0 GB/s down. A PCIe 4.0 x8 link carries 15.75 GB/s per direction in theory, and a single CPU thread writing through a mapped window gets considerably less. Host memory sees the genome written by breeding, compacted, copied by `subset`, read by packing, then the batch written, read for the upload, read back and repacked. That comes to about 8 to 10 KB of traffic per creature, or 16 to 20 GB/s at the target. The assessment measured the power cost of busy CPU cores: the GPU's standard rate fell from 174,000 to 137,000/s while the CPU evaluated.

### 2.4 Inside the kernel, per creature-step (estimate from the source)

Shared-memory accesses per step, counted pass by pass in `shaders/physics_creature.wgsl` for N nodes, B bones and M muscles. The compiler cannot keep these values in registers across loop iterations, because every index comes from packed data.

| pass | accesses | N = 6, B = 5, M = 9 |
|---|---|---:|
| bone projection, 8 sweeps | 10 per bone per sweep | 400 |
| velocity passes, 4 sweeps | 6 per bone and 6 per node per sweep | 264 |
| muscles | about 15 per muscle (4 positions, 4 velocities, force scatter) | 135 |
| integration, stance, joint limits, rebuild, planted feet, velocity rebuild, metrics | about 36 per node and 16 per bone | 296 |
| total | 60 per node, 120 per bone, 15 per muscle | about 1,095 |

Most accesses move a `vec2f`, so a warp-step moves about 1,095 x 32 lanes x 6.5 B = 228 KB through shared memory. Global memory: each muscle loads 13 words and stores 1 every step (56 B), plus a few bone fields when a joint limit acts. That comes to about 500 to 550 B per creature-step. The muscle data of the resident lanes (384 lanes x 9 x 60 B = 207 KB per SM) does not fit in the L1 left over beside shared memory, so it streams from L2.

## 3. A performance model calibrated on the measurements

### 3.1 The model

The kernel is latency-bound (assessment section 5.1). Per SM, the step rate is set by resident warps and by how long one warp takes for one step:

- warp-steps per cycle per SM = min(W / C, 4 x e / I)
- W is resident warps, C the cycles one warp needs for one step including all stalls, I the warp instructions per step, 4 the schedulers per SM, and e the achievable issue efficiency.

### 3.2 Calibration

Today's GUI runs at about 200,000 creatures/s x 1,330 step-equivalents = 266M creature-steps/s. That is 8.3M warp-steps/s, or 346,000 per SM, so one warp-step completes per SM every 7,200 cycles at 2.49 GHz. With W = 12 (measured p50), C = 12 x 7,200 = 86,000 cycles per step. With I = 4,000 to 5,000, each warp waits 17 to 22 cycles per instruction on average. The model then gives an issue rate of I / 7,200 / 4 = 14 to 17%, close to the measured 17 to 20%. That is a consistency check, not a validation, because I was itself derived from the issue rate.

Published latencies for Ada (RTX 4090, [Luo et al. 2024](https://arxiv.org/abs/2402.13499)): FP32 add and FMA 4 cycles ([Chips and Cheese](https://chipsandcheese.com/p/microbenchmarking-nvidias-rtx-4090)), shared memory 30, L1 hit 43, L2 hit 273, DRAM 541 cycles. An average of 19 cycles per instruction means a large share of instructions wait on shared memory, L2, special-function results or long division and square-root sequences. That fits the kernel's shape. Its bone projection is a sequential sweep in which every bone update reads positions the previous update wrote to shared memory, and it runs 8 times per step, then 4 more sweeps for velocities.

### 3.3 The target in model terms

2M creatures/s at 1,330 step-equivalents is 2.7G creature-steps/s, which is one warp-step per SM every 710 cycles. At 60% issue efficiency that allows I = 0.6 x 4 x 710 = about 1,700 instructions per step, whatever W is. The latency side depends on W:

| resident warps W | allowed C (cycles per step per warp) | average cycles per instruction at I = 1,500 |
|---:|---:|---:|
| 12 | 8,500 | 5.7 |
| 16 | 11,400 | 7.6 |
| 24 | 17,000 | 11.4 |
| 32 | 22,700 | 15.2 |
| today: 12 | 86,000 | about 19 at I = 4,500 |

### 3.4 On-chip memory decides W

An SM has 256 KB of registers and 128 KB of L1 and shared memory, of which shared can take up to 100 KB. Registers are allocated per thread in steps of 8.

| W | lanes per SM | registers per thread | register bytes per lane | shared bytes per lane (100 KB) |
|---:|---:|---:|---:|---:|
| 12 | 384 | 168 | 672 | 267 |
| 16 | 512 | 128 | 512 | 200 |
| 24 | 768 | 80 | 320 | 133 |
| 32 | 1,024 | 64 | 256 | 100 |
| 48 | 1,536 | 40 | 160 | 67 |

Section 5.3 counts what a 6-node creature needs on chip in the new formulation: about 230 to 250 words, or 0.9 to 1 KB, with a muscle bound of 12. The budget at W = 12 is 234 words (168 registers plus 66 shared words). Bodies with at most 8 muscles need about 50 words less and fit at W = 16. A 6-node creature cannot fit at 24. The design point is therefore 12 to 16 warps, and the design must win on I and C: at most about 1,700 instructions and 6 to 8 cycles per instruction.

### 3.5 Ceilings of today's layout

| resource | demand per creature-step | capacity | ceiling |
|---|---:|---:|---:|
| shared-memory bandwidth | 228 KB per warp-step | 128 B per cycle per SM (Luo et al.) | 1.07G creature-steps/s, about 0.81M creatures/s |
| L2 bandwidth | about 520 B | about 0.8 TB/s (estimate: the RTX 4090 measured 1,708 B per cycle across 128 SMs; scaled to 24 SMs at 2.49 GHz) | 1.54G creature-steps/s, about 1.15M creatures/s |
| PCIe and host copies | 3.6 KB and 8 to 10 KB per creature | 15.75 GB/s per direction; about 60 GB/s of practical host DRAM bandwidth | not a hard cap at 2M/s, but it would cost CPU power and GPU clock |

Today's kernel uses about 25% of the shared-memory ceiling and about 17% of the L2 estimate. That is why removing 8 of the 10 muscle loads measured no change on 2026-09-27. At ten times the rate, the same loads would saturate L2. A layout that is harmless at 1x can become the wall at 10x, and these two ceilings are why the step's data has to move on chip.

## 4. Design rules

Each rule comes from a number above.

1. No per-creature data crosses PCIe. The host sends 16-B plans and receives aggregates. Today's 3.6 KB per creature would be 7 GB/s at the target.
2. A creature's state stays on the SM for its whole trial. At most one suspend per lane per epoch writes it out. Today's segment boundaries move 0.8 to 1 KB per creature each time.
3. A step makes no global-memory accesses. The L2 ceiling in section 3.5 is below the target.
4. Shared memory holds only what topology forces to be indexed by data. The budget is 300 accesses per step, a quarter of today's. The shared-memory ceiling is below the target.
5. Constants are derived once per creature, in the lane, from the genome. That costs about one step of work, 0.1% of an average trial, and it removes host packing and the packed-body format.
6. The kernel targets 12 to 16 warps and is written for instruction-level parallelism. Passes do independent work per node, per muscle or per bone. There are no sweeps that chain through shared memory. Divisions and square roots that depend only on the genome run once per creature.
7. Queues are split by body size. Measured with `examples/body_stats.rs` on the 3M checkpoint, warps drawn from a single queue fill 69% of node-loop slots and 52% of muscle-loop slots. Queues by node count reach 99.5% and 70%. Adding muscle buckets of 4 reaches 99.8% and 87% (91% on the 100k checkpoint).
8. The host acts at epoch rate, about 10 times per second, never per unit or per creature.
9. The search stays deterministic for a given GPU and settings, as it is today. Paired A/B runs depend on it.
10. Genes stay f32 and are stored as records. Capacity does not force compression (section 5.8), and each record is read whole, once per trial.

## 5. Data shapes

### 5.1 The genome record

Canonical order: node 0 is the head, nodes are numbered depth-first, and bone j joins node j+1 to its parent. That makes the child index of every bone implicit and static, and it makes parent-first order the natural loop order. `canonicalize_bone_order` already orders bones parent-first. Renumbering the nodes is the missing step. Muscles are sorted by (bone a, bone b).

| part | fields | bytes |
|---|---|---:|
| header | id (u64), parent id (u64), certificate (u32), birth generation (u32), mutability (f32), node count, muscle count, class, emitter (u8 each), parent table (16 x u8) | 48 |
| node | x, y, diameter, friction | 16 each |
| bone | rest length, joint min and max, organ mass, organ position; the child is implicit | 20 each |
| muscle | bone a, bone b, sensor (u8 each), anchors a and b, short, long, period, phase, duty, stiffness, reset | 40 each |

A mean creature of the 3M checkpoint takes 604 B. Records live in 128-B-aligned slots of two sizes. A standard slot is 1,152 B and holds up to 8 nodes and 20 muscles, which covers 95.7% of the evolved 3M checkpoint and 91.5% of the evolved 100k one (measured). A large slot is 2,560 B and holds up to 16 nodes and 48 muscles. Bodies above 16 nodes (none in either checkpoint) fall back to the CPU engine.

Why records and not structure-of-arrays: every consumer takes whole records. Breeding writes a child whole, the archive copies a winner whole, and a lane reads its own creature whole, once. SoA pays off when the lanes of a warp read the same field of neighbouring items. Here each lane reads a different slot at a different time, so SoA would give no coalescing and would scatter every record copy over 30 to 60 arrays. A lane reads its slot with 16-byte vector loads. Each load fetches a 32-B sector and uses half of it. The next load usually finds the other half in L1, so a 604-B genome costs 0.6 to 1.2 KB of DRAM traffic, once per trial.

### 5.2 The plan record

The host decides who breeds. It writes one 16-B plan per child: the certificate (a u32 sequence number, also the RNG stream), emitter and island (u8 each), a structural-operation code (u8), a flags byte, the parent's archive cell (u32), the mate's cell or none (u16 index into a mate table), and the CMA emitter index (u16). The child's random numbers come from a counter-based generator keyed by (run seed, certificate, gene index) ([Salmon et al. 2011](https://doi.org/10.1145/2063384.2063405)). Breeding therefore does not depend on thread order, and the host can reproduce any child exactly from its plan.

### 5.3 The creature inside a lane (6-node class, reduced coordinates, estimate)

This is the layout for the most common class under the physics proposed in the assessment (section 8 below has the step). Words are 32-bit.

| data | words | where | why |
|---|---:|---|---|
| state: root position and velocity; per bone a unit direction and an angular velocity | 19 | registers | read and written every pass |
| node constants: mass; contact radius and friction as an f16 pair | 12 | registers | contacts and dynamics every step; static index |
| bone constants: length, inverse length, parent table (3 bits per node) | 11 | registers | every tree pass |
| joint limits: two limit directions as f16 pairs, break threshold | 15 | shared, lane-sliced | once per step, few instructions |
| muscle state: rhythm phase, energy (bucket bound 12) | 24 | registers | every step; the muscle loop is unrolled to the bucket bound, so indices are static |
| muscle constants: anchors as a unorm16 pair, phase rate, duty and its two reciprocals and the drive gain as f16 (bucket bound 12) | 48 | registers and shared | every step |
| behavior metrics | 12 | registers | every step |
| bone table for muscle gathers (7 words per bone) and force accumulators (3 per bone) | 50 | shared, lane-sliced | a muscle names its bones by data, so these indices are dynamic |
| temporaries at the peak of the step | 40 to 60 | registers | |
| total | 231 to 251 | | budget at W = 12: 234 |

Lane-sliced means stored as [word][lane], so the 32 lanes of a warp hit 32 different banks. Two choices here are deliberate:

- The muscle bone indices stay dynamic. They are gathered through a small per-step bone table in shared memory: 35 stores per step, then 14 loads and 12 read-modify-writes per muscle, about 280 accesses per step in total. The alternative, selecting among 5 bones in registers, costs about 64 instructions per muscle, about 580 per step. That is a third of the instruction budget, so shared memory is the better trade.
- Muscle quantities that change slowly are stored as f16 in the genome itself (section 11, decision 5), so the CPU and GPU start from the same values. Energy stays f32, because its per-step change is about 1e-4.

### 5.4 Suspend record

At an epoch's end, each lane writes its running creature: the state, the muscle state, the metrics, the tick and the slot, about 60 words. That is 240 B, 128-B aligned, one record per lane. With about 9,200 lanes it is 2.4 MB per epoch. A resumed creature re-derives its constants from its genome, which costs about one step.

### 5.5 Result record

64 B, written once, to `results[certificate mod R]`: certificate, slot, fitness, distance at the screen, fall time, screen time, ground contact, vertical oscillation, gait frequency, mean height, feet, flags (fell, screened, broken, failed, fidelity), and the settings version. Today's 19 running totals stay in registers during the trial. Only what the descriptor and the scheduler need is written.

### 5.6 Archive

There is one table per archive (global and four islands), with up to about 2,000 cells each (1,440 behavior niches plus morphology niches). The tables are structure-of-arrays, indexed by cell: best fitness, epoch winner scratch, check-running certificate, elite slot, visits, generation stamps, emitter, descriptor, topology hash. That comes to about 64 B per cell, and 128 KB per archive. SoA fits here, unlike for genomes, because absorption touches one or two fields at random cells, and those small arrays stay in L1 and L2. The elites' genomes sit in a pool of standard and large slots indexed by cell, about 12 MB in total. That fits in the 32 MB L2, where breeding reads parents.

### 5.7 Queues and rings

- Genome rings, one per slot size, hold about three epochs of children.
- Each lane has its own FIFO of 64 slot indices in global memory. The enqueue kernel fills the FIFOs.
- The result ring holds two epochs of result records.
- The check queue is a FIFO per fine-fidelity class.
- Plan buffers are double-buffered by epoch parity.
- A small stats and delta buffer is host-visible.

### 5.8 VRAM map at 2M/s with 100-ms epochs

| item | size |
|---|---:|
| genome rings (600,000 standard slots and 30,000 large) | 770 MB |
| per-lane FIFOs and suspend records (about 9,200 lanes) | 5 MB |
| result ring (2 epochs of 250,000) | 32 MB |
| archives, tables and elite genomes | 13 MB |
| plan buffers | 8 MB |
| screen histogram, stats, deltas | under 1 MB |
| total | under 1 GB of 8 GB |

For comparison, today's host holds about 10 GB at peak for the same run.

### 5.9 Host data

The host keeps a mirror of the archive metadata and elite genomes (about 10 MB), the CMA emitter state, the UI history, and the replays the player asks for. Checkpoints hold the archives, emitters, counters and screen bar, about 10 to 20 MB against 1.1 to 1.4 GB today (section 11, decision 3).

## 6. Data movement

### 6.1 One creature's life

| stage | where the bytes go | bytes |
|---|---|---:|
| plan | host to GPU over PCIe | 16 |
| birth | breed kernel reads parent genomes from L2 and writes the child into a ring slot in DRAM | 604 written |
| lane load | the lane reads its slot with 16-B loads, derives constants into registers and shared memory | 600 to 1,200 read from DRAM |
| trial | registers and shared memory only | 0 off chip per step |
| suspend and resume, only if an epoch ends mid-trial | lane record to DRAM and back | 480, for about one creature in 27 |
| result | one 64-B record to the ring | 64 |
| absorption | result ring and cell tables; winners' genomes copied into the archive pool | 64 read; 604 per winner |

### 6.2 Per epoch at 2M/s (200,000 creatures per 100 ms)

| path | per epoch | per second | share of capacity |
|---|---:|---:|---:|
| host to GPU: plans, CMA emitter updates | 3.3 MB | 33 MB/s | 0.2% of PCIe |
| GPU to host: stats, archive deltas, CMA elites, UI snapshot | about 1 MB | 10 MB/s | under 0.1% |
| DRAM: genomes written, read, results, suspends | 260 to 380 MB | 2.6 to 3.8 GB/s | 1 to 1.5% of 256 GB/s |
| L2 inside a step | 0 | 0 | |
| shared memory inside a step | about 300 accesses | | about 45% of the shared pipe at the target rate |

### 6.3 Per creature, today against this design

| | today | this design |
|---|---:|---:|
| PCIe | 3.6 KB | 16 B |
| host memory traffic | 8 to 10 KB | 16 B |
| host work | breeding, packing, repacking, archive insertion | planning, about 0.26 microseconds (measured as "parent plans" at 1M, mostly on one thread) |
| DRAM on the GPU | about 3.5 KB across segments | 1.3 to 1.9 KB |
| L1 and L2 per step | about 520 B | 0 |
| shared-memory accesses per step | about 1,100 | about 300 |

## 7. Orchestration

### 7.1 The epoch

An epoch is one command buffer holding a fixed sequence of kernels:

| order | kernel | work | time at 2M/s (estimate) |
|---:|---|---|---:|
| 1 | absorb | results finished in the previous epoch, in certificate order: descriptors, cell winners, contender checks, admissions, screen histogram, emitter counts | 0.3 ms |
| 2 | breed | one warp per child: read plan and parents, mutate, repair, canonicalize, write the ring slot | 1 ms |
| 3 | enqueue | prefix sums by class in plan order; assign each child to the lane FIFO with the smallest backlog | 0.1 ms |
| 4 | simulate | one persistent dispatch per class and fidelity, all concurrent, each lane runs K steps | about 97 ms |
| 5 | stats | reductions and archive deltas into the host-visible buffer | 0.1 ms |

Kernels 1 to 3 and 5 take about 1.5% of the epoch, and the GPU runs nothing else while they do. Overlapping them with the simulation is possible later.

The host runs one orchestrator thread. The planner works one epoch ahead, so the GPU never waits for it:

1. Wait on the timeline semaphore for epoch e - 1, in 5-ms slices so UI commands get served.
2. Read epoch e - 1's stats and deltas from the mapped buffer, and update the archive mirror.
3. Plan epoch e + 1 from that mirror, and write its plans into the upload buffer for its parity.
4. Submit epoch e + 1's command buffer. It signals the timeline value e + 1.

There are two command buffers, one per parity, recorded once. They are re-recorded only when the class set or the settings change. The per-epoch parameters (epoch number, settings version, step budget, dispatch sizes) live in a small buffer and in indirect dispatch arguments. One queue suffices. Class kernels overlap because no barrier separates them.

The planner's view of the archive lags by one to two epochs, 100 to 200 ms. That is at most 13% of a generation at the target rate. Today's steady loop has about 1 s of units in flight, so children are bred from an equally stale archive now.

### 7.2 The simulate kernel

Per lane, in outline:

```
if the lane has a suspend record: load it, read the genome, derive constants
for step in 0..K:
    if step % S == 0 and the lane is idle:
        take the next slot from this lane's FIFO; read its genome; derive constants
    if the lane is active:
        advance the creature one step
        if it fell, was screened, broke or finished: write its result; become idle
if the lane is active: write its suspend record
```

Refill cadence S: if every lane reloaded the moment its creature ended, the whole warp would wait through each load. With a cadence, a warp loads all its idle lanes at once. The cost is S/(2L) of idle lane time plus c/S for the loads, where L is about 1,060 steps per creature and c, the load cost, is about 0.5 to 1 step. The optimum is S = sqrt(2Lc), about 32 to 46, for 3 to 4.5% waste. The step budget K is a multiple of S: 28,800 steps per lane per 100-ms epoch at the target.

Tick-dependent branches (screen tick, gait sampling) diverge because lanes hold creatures at different ticks. These branches are a few instructions each. Settling disappears with the new physics. With today's physics it stays a per-lane phase.

### 7.3 Classes and lanes

- Kernel binaries are specialized by node bound, so that node and bone arrays unroll into registers: 4 or fewer, 5, 6, 7, 8, and 9 to 16. With standard and fine fidelity that is 12 binaries.
- Within a binary, warps are bound to a muscle bucket of width 4. The muscle loop runs to the warp's largest body, found by a warp vote. Buckets also set the shared-memory size of the muscle table through a specialization constant, so each (class, bucket) pair is its own dispatch. About 12 to 20 dispatches run at once.
- The host sets the warps per dispatch each epoch, in proportion to the pending step-equivalents in each queue, with a proportional correction on queue depth. A class with no work gets no warps. Fine checks get warps in proportion to their pending work, about 20% of the GPU today.

### 7.4 Determinism

Nothing depends on which lane is faster:

- Each child is assigned to a lane FIFO at enqueue, in plan order, to the lane with the smallest backlog (ties by lane index).
- Every lane runs exactly K steps per epoch.
- The set of results finished in an epoch is therefore a function of the inputs.
- Absorption processes them in certificate order.
- A cell's winner in an epoch is the maximum of (fitness, inverse certificate). This needs no 64-bit atomics. One pass takes the maximum fitness with a 32-bit ordered-integer atomic. A second pass takes, among the entries at that fitness, the smallest certificate with a 32-bit atomic minimum.
- Counts are integer atomics.
- Float sums that feed decisions, such as QD score and emitter rewards, are summed per cell or per emitter by one warp in index order, or in fixed point.
- Ring and queue positions come from prefix sums ([Merrill and Garland 2016](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)).

Dynamic load balancing through a shared atomic queue would be simpler to write. It would make the set of finished creatures, and so the archive, depend on timing. Static FIFOs keep per-lane balance exact within an epoch, because every lane runs K steps. Backlog differences between lanes cost latency, not throughput.

### 7.5 Checks, replays and the screen

- Contenders: absorption compares a result with the cell elites of its island and of the global archive. Screened creatures never qualify, as today.
- One check per cell at a time. The epoch's winning contender for a free cell sets the cell's check-running certificate. The other contenders for that cell are dropped, which is today's rule. The winner's genome slot is pinned until its check returns.
- A check runs in a fine-fidelity class kernel with the deterministic pose perturbation. Absorption takes the minimum of the standard and fine scores and admits or drops the creature. Admission copies the genome into the archive pool.
- CPU replays exist today because the CPU and GPU engines differ in the last bits. With a single physics source compiled for both (assessment 7.4), replays become an audit that samples a few elites per second off the critical path. Without it, admissions to the global archive would stay provisional until the host's replay returns, about 100 per epoch.
- Screen bar: absorption builds a histogram of distances at the screen, 4,096 integer bins of 1 cm around the current bar. At each generation boundary (every 3M absorbed results), one warp scans it and writes the next bar. Each creature carries its birth generation, and the kernel reads the bar of that generation.

### 7.6 What the CPU does

- UI thread: egui at 60 FPS or more. It reads the latest stats snapshot without locks.
- Orchestrator thread: the epoch loop above, and UI commands.
- One or two planner threads: emitter bandit, parent and mate choice, CMA updates, novelty scores, all on the archive mirror. The planner produces 200,000 plans per epoch at a measured 0.26 microseconds per child, which is 52 ms of one thread per 100 ms.
- Replay, export and autosave threads, as today, only for elites the player looks at.

The CPU stays mostly idle. That returns power budget to the GPU under Dynamic Boost.

### 7.7 Settings, world changes and checkpoints

- Settings changes take effect at the next epoch, through the settings version in the parameter buffer, within 100 to 200 ms.
- A world change flushes the lane FIFOs and suspend records and drops results from the old version, about two epochs of work (0.2 s). It then queues the archive's elites for re-testing ahead of new children. About 7,500 elites, at full trial length, take about 10 ms of GPU time at the target rate.
- A checkpoint is written at an epoch boundary. It holds the archives, emitters, counters and screen bar. Children in flight are re-bred after loading, unless the owner wants exact resume (section 11, decision 3).

### 7.8 What the toolchain must provide

- Timeline semaphores and indirect dispatch (Vulkan 1.2 core).
- Concurrent dispatches without barriers (NVIDIA overlaps them when resources allow).
- 32-bit atomics.
- Subgroup ballot and prefix operations, for warp votes and warp-aggregated appends.
- Control over registers per thread and the shared-memory carveout, to hold the 12-to-16-warp design point.

Everything except the last is available from Vulkan with SPIR-V, whether the source is WGSL through naga, GLSL or Slang. Subgroup support in naga still needs checking. The last one is available only in CUDA, where `__launch_bounds__` and the carveout attribute set it. Nsight Compute, which measures the stall reasons this design depends on, also needs CUDA. The toolchain decision in the assessment (section 7.4) therefore follows from this design. Stages 1 and 2 below can be built on what we have. Stages 3 and later want Slang or CUDA.

## 8. The step as a data flow (reduced coordinates, estimate)

The formulation is the one proposed in assessment section 7.2: a planar tree with the root node's position and velocity plus one direction and angular velocity per bone. Directions are stored as unit vectors, not angles, so no pass needs sine or cosine. Integration rotates each direction by a short polynomial and renormalizes it with one reciprocal square root.

| pass | order | reads | writes | instructions (N = 6, B = 5, M = 9) | dependency |
|---|---|---|---|---:|---|
| 1 forward kinematics | parents first | state, lengths, parent table | node positions and velocities (registers), bone gather table (shared) | about 90 | tree depth, at most 5 |
| 2 muscles | independent | gather table, muscle constants and state | muscle state, bone force accumulators (shared) | about 70 per muscle, 630 | none between muscles |
| 3 contacts and friction | independent per node | node positions and velocities, terrain | contact terms | about 40 per node, 240 | none between nodes |
| 4 articulated-body pass, backward | children first | accumulators, contact terms, joint-limit springs | articulated inertia (6 words) and bias (3) per bone | about 70 per bone, 350 | tree depth |
| 5 accelerations and integration | parents first | pass 4 | state | about 40 per bone, 200 | tree depth |
| 6 metrics, falls, screen | independent | node positions | metrics | about 60 | none |
| total | | | | about 1,570 | |

What changes for the dependency chains: today's step runs 12 sweeps over all bones, and in each sweep every bone update waits for the previous one through a shared-memory round trip. Here, passes 2, 3 and 6 carry 6 to 9 independent chains each, and passes 1, 4 and 5 chain only along the tree's depth, with branches in parallel. Contacts and joint limits act as stiff springs folded implicitly into pass 4, which is the standard implicit-spring treatment in the articulated-body method. So neither needs extra iterations. The contact model remains the open research item of the assessment. In particular, the rule that only planted feet may push has to be derived again for this formulation.

Muscles are the largest pass, about 40% of the instructions. The physics decisions that change their cost most:

- The target speed from the analytic derivative of the waveform: one sine, instead of two waveform evaluations and a difference.
- The rhythm phase kept as state and advanced by a constant, instead of recomputed from time.
- Slow muscle genes kept as f16 in the genome.

Rate estimate: I is about 1,500 to 1,700. With an average of 6 to 10 cycles per instruction per warp, C is 9,600 to 16,000 cycles at W = 12. That gives 1.4G to 2.4G creature-steps/s, or 1.1M to 1.9M creatures/s at 1,230 step-equivalents (no settling). Bodies with at most 8 muscles run at W = 16 and do better. The range is about 1.1M to 2.5M/s. Every added cycle of average stall per instruction costs 10 to 15%, so C is the number to track through phases 0 to 4.

### 8.1 Today's physics in the same framework

Stages 1 and 2 (section 10) work with the current kernel. It keeps its in-kernel layout: node state in shared memory, muscles in global memory. So the ceilings of section 3.5 remain at about 0.8M/s and 1.15M/s. Today's rate is a quarter of the lower one, so there is room. The gains come from outside the step:

- No host packing and no PCIe traffic.
- No segment round trips.
- Lanes refill within S steps of a fall or screen, instead of at the next segment boundary.
- Class queues keep warp loops at 99% for nodes and 87 to 91% for muscles.

The assessment's estimate for this path, 300,000 to 450,000 creatures/s, stands.

## 9. Corrections to `docs/hpc-assessment.md`

1. Occupancy. The assessment implied that the new physics's smaller state would roughly double occupancy. Counting muscles and step temporaries, a 6-node creature still needs 0.9 to 1 KB on chip, so occupancy stays at 12 to 16 warps (section 3.4). The gain has to come from instruction count and dependency chains.
2. Instructions per step. The assessment estimated 1,100 to 1,400 for the new physics. The pass-by-pass count in section 8 gives 1,500 to 1,700. The instruction reduction is therefore about 2.5 to 3x, not 3 to 4x.
3. Muscle loads. The assessment took the 2026-09-27 measurement to mean global muscle loads are harmless. They are harmless at today's rate. At the target rate they would exceed the estimated L2 bandwidth (section 3.5).

The assessment's combined estimate is updated to match.

## 10. Build order and gates

| stage | work | physics | gate |
|---:|---|---|---|
| 1 | Genome records in canonical order, uploaded to a ring by the host (0.6 KB per creature over PCIe, about 1.2 GB/s at the target). The kernel derives constants at lane load. Persistent class kernels with lane FIFOs, refill cadence, step-budgeted epochs and device-side suspend. Result ring. Host packing, segments, readbacks and repacks are removed. | today's, bit-exact | same result per creature as today (fall, screen, distance, bit for bit); GUI rate up; epoch overhead under 3%; refill waste under 5% |
| 2 | Breeding, absorption, checks and screen bar on the GPU; host planner. | today's | 10-seed search A/B at equal evaluations against the current loop; host RSS under 1 GB; PCIe under 50 MB/s; two runs of the same seed produce byte-identical archives |
| 3 | Toolchain with register control and Nsight Compute (Slang or CUDA). | today's | profiler confirms W, I and C; equal or better rate |
| 4 | Physics v2 kernel in the same framework. | new | assessment 7.2 gates; K6 kernel at 168 registers or fewer and 267 B of shared memory or less per lane (`examples/shader_stats.rs`); zero global accesses per step; I at 1,700 or less |
| 5 | Kernels specialized for the most common body plans, if profiling shows the muscle gather as a large cost. The ten largest plans hold half the population (performance log). | new | rate on plan-heavy populations |

Stage 1 also answers the question the model cannot: how much of today's GUI inefficiency (12 warps at 23% issue in the game, against 24 warps at 44% in `eval-bench`) comes from orchestration rather than from the kernel.

Phase 0 of the assessment gains four measurements for this design:

- L2 bandwidth on this GPU, to check the 0.8 TB/s estimate.
- Shared-memory throughput.
- Shared-memory and L2 throughput of today's kernel in the game, to check the 25% and 17% figures.
- Shared load and store instructions per creature-step, to check the count of 1,100.

## 11. Decisions for the owner

These come in addition to the four in the assessment.

1. Determinism. Keep the search bit-reproducible for a given GPU and settings? Recommended: yes. It costs the static lane FIFOs of section 7.4 and nothing measurable in throughput.
2. Archive insertion by epoch. The GPU inserts the best contender per cell per epoch in parallel, where today's code offers creatures one at a time. The result for "best per cell" is the same. Emitter rewards are computed against the archive at the start of the epoch, which slightly changes the bandit's feedback. This needs a 10-seed A/B before it lands.
3. Checkpoints. Save only the archives and search state (10 to 20 MB; children in flight are bred again after loading)? Or also save the rings and FIFOs for an exact resume (about 0.8 GB)? Recommended: the small form.
4. Screen bar resolution of 1 cm, instead of an exact percentile.
5. Slow muscle genes as f16. This matters only for the new physics's register budget. Mutation steps and CMA samples are rounded to about 3 significant digits.

## References

- X. Luo et al., [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499), IPDPS 2024. Latency and bandwidth tables for the RTX 4090 (Ada).
- Chips and Cheese, [Microbenchmarking Nvidia's RTX 4090](https://chipsandcheese.com/p/microbenchmarking-nvidias-rtx-4090). Ada L2 bandwidth and FP32 latency.
- NVIDIA, [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html). Registers, warps and shared memory per SM.
- V. Volkov, [Understanding Latency Hiding on GPUs](https://escholarship.org/content/qt1wb7f3h4/qt1wb7f3h4_noSplash_1e32f64125997ee6afa303a150338054.pdf), PhD thesis, 2016.
- K. Gupta, J. Stuart and J. Owens, [A Study of Persistent Threads Style GPU Programming for GPGPU Workloads](https://doi.org/10.1109/InPar.2012.6339596), InPar 2012.
- T. Aila and S. Laine, [Understanding the Efficiency of Ray Traversal on GPUs](https://research.nvidia.com/sites/default/files/publications/aila2009hpg_paper.pdf), HPG 2009.
- J. Novák, V. Havran and C. Dachsbacher, [Path Regeneration for Interactive Path Tracing](https://jannovak.info/publications/PathRegPT/PathRegPT.pdf), Eurographics 2010.
- S. Laine, T. Karras and T. Aila, [Megakernels Considered Harmful](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf), HPG 2013.
- D. Merrill and M. Garland, [Single-pass Parallel Prefix Scan with Decoupled Look-back](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back), NVIDIA technical report, 2016.
- J. Salmon, M. Moraes, R. Dror and D. Shaw, [Parallel Random Numbers: As Easy as 1, 2, 3](https://doi.org/10.1145/2063384.2063405), SC 2011.
- B. Lim, M. Allard, L. Grillotti and A. Cully, [Accelerated Quality-Diversity through Massive Parallelism](https://arxiv.org/pdf/2202.01258), TMLR 2023.
- R. Featherstone, *Rigid Body Dynamics Algorithms*, Springer 2008. Articulated-body method and implicit joint springs.
