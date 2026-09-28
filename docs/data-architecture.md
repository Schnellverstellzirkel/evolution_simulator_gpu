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

Long sessions (section 10, measured on the owner's 3M session at generation 70): bodies grow from 6 nodes and 9 muscles to 10.6 nodes and 34 muscles. Each creature then costs 4.2 times the GPU work, and the game runs 5 times slower (38,600 creatures/s). Host memory reaches 22 GB, and an autosave (4.2 GB, cloned in memory) pushes the machine into swap. The growth is selected for, because a muscle is free work capacity in today's physics. The design therefore scales cost with body size without cliffs: lane groups for large bodies, memory sized for the caps, no population on the host, and no files unless the player saves.

Expected effect: with today's physics, stage 1 (section 11) removes host packing, segments and PCIe traffic. The assessment's estimate of 1.3 to 2x still applies to it. With the reduced-coordinate physics, the model in section 3 gives about 1.1M to 2.5M creatures/s on this laptop. The spread comes almost entirely from one number: the average stall per instruction must fall from about 19 cycles to about 6 to 8.

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

Section 5.3 counts what a 6-node creature needs on chip in the new formulation: about 220 to 240 words, or 0.9 to 1 KB, with a muscle bound of 12. The budget at W = 12 is 234 words (168 registers plus 66 shared words). Bodies with at most 8 muscles need about 50 words less and fit at W = 16. A 6-node creature cannot fit at 24. The design point is therefore 12 to 16 warps, and the design must win on I and C: at most about 1,700 instructions and 6 to 8 cycles per instruction.

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

A mean creature of the evolved 3M checkpoint (generation 9) takes 604 B. A mean creature of the owner's generation-70 session takes 1,771 B. A body at the caps (32 nodes, 96 muscles) takes 5,020 B. Fixed slot sizes do not survive a long session: a slot for 8 nodes and 20 muscles held 95.7% of the generation-9 population and 27.6% of the generation-70 one (measured). Records are therefore variable-size and 128-B aligned in one byte ring. The breed kernel places each epoch's children contiguously, by a prefix sum over their sizes. The ring frees an epoch's range once every creature born in it is absorbed. A contender whose check is still running is copied into a small check pool, so it does not hold its epoch's range.

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
| muscle constants: anchors as a unorm16 pair; duty, its two reciprocals and the drive gain as f16 (bucket bound 12). All muscles of a body share one clock (repair sets every period to the first muscle's), so the phase rate is one word per creature | 37 | registers and shared | every step |
| behavior metrics | 12 | registers | every step |
| bone table for muscle gathers (7 words per bone) and force accumulators (3 per bone) | 50 | shared, lane-sliced | a muscle names its bones by data, so these indices are dynamic |
| temporaries at the peak of the step | 40 to 60 | registers | |
| total | 220 to 240 | | budget at W = 12: 234 |

Lane-sliced means stored as [word][lane], so the 32 lanes of a warp hit 32 different banks. Two choices here are deliberate:

- The muscle bone indices stay dynamic. They are gathered through a small per-step bone table in shared memory: 35 stores per step, then 14 loads and 12 read-modify-writes per muscle, about 280 accesses per step in total. The alternative, selecting among 5 bones in registers, costs about 64 instructions per muscle, about 580 per step. That is a third of the instruction budget, so shared memory is the better trade.
- Muscle quantities that change slowly are stored as f16 in the genome itself (section 12, decision 5), so the CPU and GPU start from the same values. Energy stays f32, because its per-step change is about 1e-4.

### 5.4 Suspend record

At an epoch's end, each lane writes its running creature: the state, the muscle state, the metrics, the tick and the slot, about 60 words. That is 240 B, 128-B aligned, one record per lane. With about 9,200 lanes it is 2.4 MB per epoch. A resumed creature re-derives its constants from its genome, which costs about one step.

### 5.5 Result record

64 B, written once, to `results[certificate mod R]`: certificate, slot, fitness, distance at the screen, fall time, screen time, ground contact, vertical oscillation, gait frequency, mean height, feet, flags (fell, screened, broken, failed, fidelity), and the settings version. Today's 19 running totals stay in registers during the trial. Only what the descriptor and the scheduler need is written.

### 5.6 Archive

There is one table per archive (global and four islands), with up to about 2,000 cells each (1,440 behavior niches plus morphology niches). The tables are structure-of-arrays, indexed by cell: best fitness, check-running certificate, elite slot, visits, generation stamps, emitter, descriptor, topology hash. That comes to about 64 B per cell, and 128 KB per archive. SoA fits here, unlike for genomes, because absorption touches one or two fields at random cells, and those small arrays stay in L1 and L2. The elites' genomes sit in a pool of cap-sized slots (5,120 B) indexed by cell, 51 MB for five archives of 2,000 cells. The unused tail of a slot costs VRAM but no cache, because L2 holds lines, not slots. The genomes themselves take 4 to 13 MB, from generation 9 to generation 70, and stay in the 32 MB L2, where breeding reads parents.

### 5.7 Queues and rings

- The genome ring holds about four epochs of children, in bytes (section 5.1).
- Each lane has its own FIFO of 64 slot indices in global memory. The enqueue kernel fills the FIFOs.
- The result ring holds two epochs of result records.
- The check queue is a FIFO per fine-fidelity class.
- Plan buffers are double-buffered by epoch parity.
- A small stats and delta buffer is host-visible.

### 5.8 VRAM map at 2M/s with 100-ms epochs

| item | size |
|---|---:|
| genome ring (4 epochs of 200,000 at the generation-70 mean of 1.8 KB; 4 GB if every body sat at the caps) | 1.4 GB |
| per-lane FIFOs and suspend records (about 9,200 lanes) | 5 MB |
| result ring (2 epochs of 250,000) | 32 MB |
| archives, tables and elite genome slots | 52 MB |
| plan buffers | 8 MB |
| screen histogram, stats, deltas | under 1 MB |
| total | about 1.5 GB of 8 GB (4.1 GB at the caps) |

For comparison, today's host holds about 10 GB at peak at generation 9 and 22 GB at generation 70. Nothing in the device map grows during a session: every buffer is sized for the caps or for a byte budget at start.

### 5.9 Host data

The host keeps a mirror of the archive metadata and elite genomes (about 10 MB), the CMA emitter state, the UI history, and the replays the player asks for. Checkpoints hold the archives, emitters, counters and screen bar, about 10 to 20 MB against 1.1 to 1.4 GB today (section 12, decision 3).

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
- Contenders are offered to each archive one at a time, in certificate order, by one warp per archive. That keeps today's insertion semantics exactly, emitter rewards included. Contenders are about 5% of results (measured at generation 70: 63,000 to 90,000 global and 63,000 to 92,000 island contenders per 3M generation), so about 10,000 offers per epoch take about 1 ms.
- Counts are integer atomics.
- Float sums that feed decisions, such as QD score and emitter rewards, are summed per cell or per emitter by one warp in index order, or in fixed point.
- Ring and queue positions come from prefix sums ([Merrill and Garland 2016](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)).

Dynamic load balancing through a shared atomic queue would be simpler to write. It would make the set of finished creatures, and so the archive, depend on timing. Static FIFOs keep per-lane balance exact within an epoch, because every lane runs K steps. Backlog differences between lanes cost latency, not throughput.

### 7.5 Checks, replays and the screen

- Contenders: absorption compares a result with the cell elites of its island and of the global archive. Screened creatures never qualify, as today.
- One check per cell at a time. The first contender in certificate order to beat a free cell's elite sets the cell's check-running certificate. The other contenders for that cell are dropped, which is today's rule. The winner's genome slot is pinned until its check returns.
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
- A checkpoint is written at an epoch boundary. It holds the archives, emitters, counters and screen bar. Children in flight are bred again after loading (section 12, decision 3).

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

Stages 1 and 2 (section 11) work with the current kernel. It keeps its in-kernel layout: node state in shared memory, muscles in global memory. So the ceilings of section 3.5 remain at about 0.8M/s and 1.15M/s. Today's rate is a quarter of the lower one, so there is room. The gains come from outside the step:

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

## 10. Long sessions

### 10.1 What a long session does (measured)

The owner noticed that evolution slows sharply after a few dozen generations. The owner's 3M session reached generation 70 and left an autosave. It was measured against the evolved 3M checkpoint at generation 9 with the same binary settings. The GUI benchmark ran one warm-up and two measured generations, with `EVOLUTION_CPU_THREADS=0`. `eval-bench` ran on 200,000 creatures, with every creature fine-checked.

| | generation 9 | generation 70 |
|---|---:|---:|
| mean nodes / bones / muscles | 6.02 / 5.02 / 8.99 | 10.64 / 9.64 / 33.98 |
| largest body | 11 nodes, 30 muscles | 21 nodes, 96 muscles (the muscle cap) |
| bodies within 8 nodes and 20 muscles | 95.7% | 27.6% |
| packed GPU data per creature | 1,005 B | 2,818 B |
| GPU rate, `eval-bench` | 15,400/s | 3,700/s |
| game, end to end | 185,000 to 216,000/s (2026-09-27) | 38,600/s |
| breeding per generation | 2.6 to 3.8 s | 9.0 to 11.1 s |
| peak RSS, autosave off | about 10 GB | 22.2 GB |
| checkpoint | 1.37 GB | 4.18 GB |
| GPU under load | | 2,502 MHz mean, 44 W, 59 °C |

The slowdown is work per creature, not heat. The clock held at 2,502 MHz. Memory adds a cliff on top. At generation 70 the population's genes take 6.4 GB, and the arena keeps three copies (live, children, compaction spare). The session's autosave cloned the whole experiment on the worker thread every tenth generation. With 22 GB already resident, that clone crosses into swap on this 32 GB machine, which likely explains why the drop felt sudden. The autosave itself could not be loaded: validation assumed that the first `evaluated` scores are the finished ones, which is false in a continuous run. That is fixed, with a regression test.

### 10.2 Why bodies grow

- Structural mutation had three operators that add parts (split a bone, mirror a node, duplicate a limb) and none that removes them.
- Repair keeps a muscle between every pair of consecutive bones, so the muscle count rises with the bone count.
- In today's physics a muscle has no mass, and each one brings its own 120 J energy store and up to 100 N of force. More muscles are free work capacity.

Removal operators were measured as a test of the first cause. `EVOLUTION_SHRINK=1` adds `remove_limb` (a leaf node with its bone and muscles) and `remove_muscle` (one muscle outside the consecutive-bone ring), at the same rate as the operators that add nodes. `examples/search_ab.rs` ran 10 seeds (38 to 47), 40 generations and 5,000 creatures, with 60 s trials:

| | nodes / muscles at generation 39 | best, mean (median) | QD, mean (median) | CPU wall |
|---|---:|---:|---:|---:|
| current operators | 7.31 / 13.61 | 346 m (280) | 30,100 (17,800) | 196 s |
| with removal | 6.41 / 10.31 | 275 m (278) | 21,300 (18,100) | 182 s |

Removal slows growth by about a quarter. Medians tie, and the means favour the current operators because of one or two strong seeds. So growth is selected for, not only drift, and removal alone does not fix it without a cost. The flag stays off.

Muscle mass was measured next as a test of the third cause (section 12, decision 6). A muscle weighs `muscle_density` kg per meter of its span in the starting pose, half at each attachment point, and its energy store is 200 J per meter of span instead of a flat 120 J. At the generation-9 3M checkpoint the median span is 0.53 m, so the median store stays near 105 J, and at 1 kg/m muscles raise the median body mass from 9.9 kg to 16.3 kg. Same seeds and settings as the table above:

| | nodes / muscles at generation 39 | top-50 mean nodes | best, mean (median) | QD, mean (median) | cells, mean |
|---|---:|---:|---:|---:|---:|
| massless muscles, 120 J each | 7.31 / 13.61 | 7.44 | 346 m (280) | 30,075 (17,768) | 1,018 |
| 1 kg/m, 200 J/m | 7.47 / 13.60 | 7.74 | 155 m (133) | 16,007 (12,648) | 957 |
| 4 kg/m, 200 J/m | 7.51 / 14.21 | 8.73 | 144 m (121) | 15,707 (13,343) | 830 |

Muscle mass does not stop growth. At 1 kg/m the population grows at the same rate. At 4 kg/m it grows faster and the top 50 are larger (median total bone length 5.6 m against 2.0 m). Distance and QD fall by about half in both.

All variants and the baseline were then run to 80 generations, with a flat 120 J store as a further variant:

| | nodes / muscles, gen 39 | nodes / muscles, gen 79 | top-50 mean muscles | best, mean (median) | QD, mean (median) |
|---|---:|---:|---:|---:|---:|
| massless muscles, 120 J | 7.31 / 13.61 | 8.37 / 18.56 | 21.7 | 588 m (490) | 104,828 (57,527) |
| 1 kg/m, 200 J/m | 7.47 / 13.60 | 8.55 / 17.43 | 26.7 | 247 m (195) | 33,506 (23,569) |
| 4 kg/m, 200 J/m | 7.66 / 14.93 | 9.05 / 19.96 | 31.4 | 222 m (181) | 27,975 (25,865) |
| 1 kg/m, flat 120 J | 7.42 / 13.43 | 8.70 / 18.24 | 29.0 | 289 m (237) | 30,121 (31,108) |
| 4 kg/m, flat 120 J | 7.65 / 14.09 | 9.14 / 20.34 | 31.9 | 245 m (233) | 24,072 (20,586) |

No variant holds muscle counts down, and in every one the best bodies become heavier and more muscular than without muscle mass. On the owner's generation-70 save, today's many-muscle bodies collapse under muscle mass, but so do lean ones, because every gait there was tuned to massless muscles. A likely reason heavy bodies win, untested: grip grows with the load a foot carries, and a muscle's 100 N force limit dwarfs its weight. The implementation stays on branch `claude/muscle-mass` and is not merged. Details are in `docs/performance-log.md`, section "Muscle mass".

### 10.3 Architecture rules for long sessions

1. Cost follows body size without cliffs. Today's kernel loses occupancy in steps as bodies cross the capacity buckets 8, 12, 16 and 24. That is why it slowed 4.2 times where its shared-memory accesses per step grew about 2.1 times (estimate from the counts in section 2.4).
   - In the new design, classes extend to the caps (32 nodes, 96 muscles). A body above the per-lane budget spans a lane group of 2, 4 or 8 lanes. Its muscles, joint limits and gather table are split across the group, and the tree passes exchange values through warp shuffles inside it.
   - The generation-70 mean body needs about 430 words on chip in the layout of section 5.3, twice the per-lane budget. A group of 4 brings it to about 150 words per lane, inside the 16-warp budget.
   - The class table is recomputed every epoch from the live mix, so the mapping follows the bodies as they grow. This is the warp-cooperative mapping Madrona uses for large entities.
2. Nothing grows during a session except the archive, which is bounded by its cells (5 x about 1,500 plus 64 morphology cells) and by lineage pruning. Device buffers are sized at start for the caps or a byte budget (section 5.8). The host holds no population. Today's host holds three copies of it, which scale with body size.
3. The host's cost per creature does not depend on body size. Planning is per child. Breeding, whose cost scales with the genome, runs on the GPU.
4. No clones and no files. Checkpoints hold the archives and search state, and are written only when the player saves. Today's manual save serializes the whole population on the worker thread. At generation 70 that is a 4.2 GB file, and loading the same file took 48 s. The small checkpoint makes that trivial.
5. The speed readout separates body growth from engine speed. When bodies grow, creatures/s falls by design. The status line should also show the work rate (creature-steps per second, or node-steps) and the mean body size. A drop caused by growth then reads as growth, and a drop caused by the engine stands out.
6. A long-session gate. Every stage in section 11 is benchmarked on a generation-70-class population as well as on the generation-9 one. The work rate must not fall more than the body-size ratio explains.

The 2M creatures/s target is defined at a body mix. At the generation-70 mix a creature is about four times the work, so on the same hardware the target means about 500,000/s there. Holding creatures/s constant across a session needs either physics that stops free growth (decision 6) or lower caps.

## 11. Build order and gates

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

## 12. Decisions

Answered by the owner on 2026-09-28:

1. Determinism: keep the search bit-reproducible for a given GPU and settings. Yes.
2. Archive insertion: left to the design. Chosen: keep today's one-at-a-time offers, in certificate order (section 7.4). They cost about 1 ms per epoch, so the parallel alternative and its search A/B are not needed.
3. Checkpoints: only the archives and search state, 10 to 20 MB; children in flight are bred again after loading. Yes. More generally, the game should be as stateless as possible and write as few files as possible. Done now: autosave is off by default and after loading.
4. Screen bar resolution of 1 cm. Yes.
5. Slow muscle genes as f16. Yes.

Open:

6. Muscle mass (section 10.2). The owner asked for it to be built and measured (2026-09-28), and accepts half the best distance if it stops muscle monsters. It does not. Muscles that weigh 1 or 4 kg per meter of span, with a 200 J per meter store or a flat 120 J store, leave the population's muscle count between 6% lower and 10% higher than today's after 80 generations, and the best bodies become heavier and more muscular. It is not adopted. Bounding per-creature cost in long sessions still needs a different physics lever or lower caps.
7. The four decisions of the assessment (new physics formulation, NVIDIA-only fast path, search-side levers, clock pinning) are still open.

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
