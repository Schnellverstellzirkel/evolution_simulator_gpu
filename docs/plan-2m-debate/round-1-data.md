# Round 1, data flow: the pipeline that never starves the kernel

Domain: rings, queues, memory, PCIe, overlap, determinism, the 60 FPS UI. Source of numbers: 00-facts.md, hpc.md, and the code on main (src/ring.rs, src/cuda_engine.rs, src/warp_kernel.rs, src/worker.rs, src/storage.rs, src/scheduler.rs, src/engine.rs, shaders/warp_creature.cu) as of 2026-09-30. Every number I derived is marked "derived" and says what it came from.

## 1. The claim

The kernel is not the only wall. At 2M creatures/s the host does per-creature work that today costs about 16 to 20 microseconds of one CPU thread per creature. Spread over 8 threads that is a ceiling of about 400k to 500k creatures/s before the kernel matters at all, and the CPU load it takes lowers the GPU clock (measured: 8 busy threads cut the GPU rate 174k to 137k, 21%). The upload path costs 3 to 8 GB/s on a PCIe 4.0 x8 link at 2M/s. And the ring is sized in creatures, so at 2M/s it holds only 0.4 s of GPU work, and any host pause longer than that starves the GPU.

So my position: the per-creature path must have no CPU and no PCIe in it. Genes, breeding, packing, descriptors and the candidate filter live on the GPU. The CPU keeps per-block and per-generation control (verdicts on a few thousand candidates, CMA bookkeeping, migration, history, saves, the UI snapshot), which is O(blocks), not O(creatures). This gives back the 21% clock tax, removes the host ceiling, cuts VRAM from 5 to 7.5 GB to about 1 GB and host RSS from 5 to 9 GB to under 1 GB, and lets the ring be sized in seconds without a memory cost.

## 2. What the current pipeline costs, per creature

The path of one creature today (ring.rs, storage.rs, engine.rs, cuda_engine.rs):

1. Breed on the CPU at absorption of its block: plan (emitter, parent, CMA slot), emit (operators), repair, append into the block's arenas. Measured 4.7 s per 3M generation on the rayon pool. Derived: 4.7 s x 8 threads / 3M = 12.5 microseconds of one thread per creature.
2. Pack on the CPU (warp_kernel::pack): Model::new per creature, breadth-first bone order, 12 words per lane, 16 floats per muscle per round, end lists, heads. Measured 0.32 s per 262k on 4 threads. Derived: 4.9 microseconds of one thread per creature. At 2M/s that is 10 of the 16 hardware threads.
3. Copy into pinned staging, then cuMemcpyHtoDAsync (cuda_engine.rs submit_as). Derived record size: W=8, 15 muscles, 2 rounds: lanes 384 B + muscles 1024 B + ends 64 B + heads 32 B = 1.5 KB. W=16, 40 muscles, 3 rounds: 768 + 3072 + about 200 + 32 = 4.1 KB. At 2M/s: 3 to 8 GB/s of upload, plus the same in host memcpy. PCIe 4.0 x8 (nvidia-smi, hpc.md) moves about 12 to 13 GB/s in practice. Bodies grow over generations (the old build went to 9 nodes and 30 to 50 muscles by generation 30), so the record grows toward the 4 KB end. The bus is not saturated but it is a third to two thirds used, on top of a host memcpy of the same size.
4. Simulate. Results are written by creature index (results[cidx] in warp_creature.cu), 80 B each. Readback at 2M/s is 160 MB/s. Nothing.
5. Absorb on the CPU: descriptor and eligibility for every result in parallel (the Prep pass in archive_block), then a sequential commit per island in block order, islands in parallel. Measured 0.6 to 2.3 s per generation. Derived: 1.6 to 6 microseconds of one thread per creature.
6. Breed again (step 1).

Sum of steps 1, 2 and 5: 19 to 23.5 microseconds of one thread per creature. With 8 threads at 100%: 340k to 420k creatures/s. That is the host ceiling of the current design, with the GPU clock tax on top. The measured end-to-end rate of the new main (92k to 167k/s) is below it only because the kernel is slower still. Once the kernel gains 3x, the host is the wall.

The generation as a time budget at 2M/s: 1.5 s. Breeding plus absorption today: 5.3 to 7 s. Off by 3.5 to 4.7x.

Memory today (derived from cuda_engine.rs ensure_buffers and the sizes above): 4 evaluation slots each holding a unit of 196,608 creatures as packed records with 25% headroom, so 0.4 to 1 GB of device buffers plus a pinned staging mirror of the same size per slot, plus a readback buffer, plus the replay slot and its frames. That matches the observed 5 to 7.5 GB of VRAM and the backlog note "CUDA costs 1.2 to 1.6 GB more peak RSS (pinned buffers)". The gene arenas of the ring are 786k x 1.1 to 2.5 KB = 0.9 to 2 GB of host RAM (derived from the gene structs: NodeGene 16 B, Bone 28 B, Muscle 52 B, Genome 56 B; a 7-node, 6-bone, 15-muscle body is 1.1 KB, a 9-node, 8-bone, 40-muscle body is 2.5 KB).

The ring in seconds: 786,432 creatures at 167k/s is 4.7 s of GPU work. At 2M/s it is 0.39 s. Every host pause longer than that starves the GPU: a confirmation round trip (a 4x-rate trial behind a queued wave, 150 to 300 ms today), a snapshot build, a generation boundary, a save, a command.

Replay latency, a UI matter that lives in my domain: a wave launches 96 blocks (4 per SM x 24, from MIN_BLOCKS and the register budget), each looping on the atomic counter until the wave is exhausted, so no block slot frees until the wave's tail. The replay slot's high-priority stream gets a block slot only then. A wave is 262,144 creatures, so at 167k/s a click-to-replay can wait up to about 1.5 s for the GPU. The UI already measures this (replay_seconds).

## 3. Proposals, ranked by expected gain

Gains are in end-to-end creatures/s at the target, assuming the kernel side reaches roughly 1 G creature-steps/s. Each proposal names the measurement that confirms it.

### P1. Genes on the device, unpack at take-up (the pack and the upload disappear)

The ring's gene arenas move to VRAM in the same SoA layout (genomes, nodes, bones, muscles, per block). A lane group that takes a creature (the take-up branch at the top of the kernel's for loop, where it reads heads and the lane record) reads the genome instead and computes what warp_kernel::fill_creature and physics2::Model::new compute today: masses with organs and muscles, slack lengths, strengths, joint ranges, the start pose, the breadth-first bone order, the muscle end lists. This runs once per creature against 500 to 1200 steps of 2 substeps each, so it is under 1% of the creature's cost even if it takes 20k lane instructions (derived: a trial is about 500 steps x 2 substeps x 1500 instructions x W lanes, on the order of 10M lane instructions).

What goes away: the pack (10 hardware threads at 2M/s), 3 to 8 GB/s of upload and host memcpy, the pinned staging mirrors (about 2 to 4 GB of RSS), and the per-slot packed buffers (about 2 to 4 GB of VRAM). What remains to upload in this phase: children bred on the CPU as genes, 1.1 to 2.5 KB each, so 2 to 5 GB/s at 2M/s. That is still a lot, and P2 removes it.

Expected gain: the pack ceiling (about 800k/s on 4 threads, 3.2M/s on all 16 with nothing left for breeding) is gone, and the CPU threads it used give the GPU its clock back. I estimate +10 to +20% GPU rate from the clock alone, from the 21% measurement.

Risk: Model::new must be ported to CUDA and produce the same lane records. Gate: bit-equal lane records between warp_kernel::fill_creature and the device unpack for a 262k population (dump both, compare), then p2_speed unchanged or better.

Measurement: p2_speed with pinned clocks, plus nvidia-smi clocks.sm and power.draw at 100 ms while the game runs, before and after. The clock tax itself needs a table: p2_speed rate with 0, 2, 4, 8, 16 busy CPU threads. I ask the HPC or power domain to own that table (section 5).

### P2. Recipes, not genomes: breed at take-up on the device

A child is a pure function of (seed, generation, breed_round, slot, parent, emitter, operator, CMA state). The CPU already makes it so (Rng::new(seed, generation, index) per slot, salted by breed_round). So a ring entry does not need to hold the child's genes. It holds a recipe of 16 to 32 bytes: parent slab id, mate slab id, emitter, operator, CMA emitter id and its state version, protection, elite_before. The lane group materializes the child at take-up: read the parent's genes from the device archive slab (L2 resident: the archives are 5 islands x 1,440 niches plus reserves, about 7,500 elites x 2.5 KB = 19 MB, in a 32 MB L2), apply the mutation, repair, then unpack as in P1.

The archive slab is append-only with epoch reclamation, because a parent can be replaced by a later block's absorption before the child that references it is taken up (the ring holds 4 blocks; the child of block k is materialized up to 3 absorptions later). A replaced elite's slab entry is freed once every block bred while it was live has been taken up. Derived slab size: 7,500 live plus about 4 blocks x a few thousand replaced entries, about 20k entries x 2.5 KB = 50 MB. CMA state is versioned per block the same way (a few hundred emitters x a few KB).

Order of work: parametric children first (CMA, local mutation, the novelty emitter's local mutation): 65% of offspring by the 35/35/30 shares. Each is a loop over the parent's parameters with a gaussian sample and a clamp, then repair (clamps, period copy, bone start lengths). That is a few thousand lane instructions. Structural children (35%, 64 operators, src/evolution/anatomy at about 7,500 lines with evolution.rs) stay on the CPU in this phase and are uploaded by value: 1M x 2.5 KB = 2.5 GB per generation, 1.7 GB/s at 2M/s. Their CPU cost is the part of the 4.7 s that is structural, which I estimate at 1.6 to 2.5 s per generation. That is still above the 1.5 s budget on 8 threads, so 2M/s needs P3 too. With P1 and P2 alone I estimate a host ceiling near 1M/s.

Why recipes rather than plain resident genomes: the ring must be sized in seconds (P4), and at 2M/s a 1 s ring is 2M entries. As genomes that is 5 GB of VRAM. As recipes it is 32 to 64 MB. The working set (archive slab 50 MB plus ring 64 MB) nearly fits the L2. The whole flood of data becomes a flood of recipes; genes are materialized in registers and written to memory only when a child wins a cell.

Determinism: the recipe fixes the parent by immutable slab id and the CMA state by version, so the child is the same whatever the timing. Bit equality with the CPU breeder is not required (the owner's rule is one seed, one search, on one GPU), but the parametric path should be tested bit-equal with the CPU emitter as a diagnostic, with FMA contraction disabled in the breed code (or accepted and documented).

Risk: the biggest port in the plan. Repair and the CMA sampler are small; the versioned slab and its reclamation are new code with a real bug surface. Gate: 100 generations at 3M with a fixed seed twice, same history (the two_continuous_runs_of_one_seed_agree test at scale), and the emitter discovery rates (emitter_stats) within noise of the CPU breeder over 10 seeds.

Measurement: EVOLUTION_STAGE_LOG breeding seconds per generation before and after (4.7 s to under 0.2 s expected for the CPU part), and the end-to-end rate.

### P3. Structural operators on the device, sorted by operator

The 64 operators are branchy tree edits. One creature per thread with a switch over 64 cases diverges 32 ways in a warp. Sorting a block's structural recipes by operator before launch (the recipes are on the host at plan time; a counting sort over 64 keys on 70k entries is microseconds) makes warps uniform. Derived cost: 1M structural children per generation x about 50k instructions each (a generous guess for an operator plus repair) at 30% issue efficiency on 7.6 T instructions/s is about 0.02 s per generation, under 2% of the 1.5 s budget. Unsorted it would be about 30x worse, around 0.5 s, so the sort is not optional.

This is the phase that takes the CPU out of the per-creature path completely. After it, the CPU does per block: read back results (16 MB per 196k block, 1.5 ms), read back the candidate list (a few thousand genomes, 5 to 10 MB), verdict and confirmations, the exact archive commit on candidates, CMA updates from the block's samples, the plan of the next block (until the plan moves too), and the recipe upload (196k x 16 B = 3 MB). I estimate 30 to 60 ms per block on 1 to 3 threads. A block at 2M/s lasts 100 ms. It fits, with most cores idle, which is what the GPU clock wants.

Risk: the port of src/evolution/anatomy. Each operator must be ported and tested against the CPU version (same recipe, same child, as a diagnostic). It is weeks of work, and it is the price of 2M/s sustained. Gate: mutation_audit numbers per operator on the device within noise of the CPU operator's.

### P4. The ring sized in seconds, blocks sized in milliseconds

Today: RING_SLOTS 786,432, RING_BLOCKS 4, WORK_UNIT 196,608, WAVE 262,144, all in creatures. At 2M/s that is a 0.39 s ring, 100 ms blocks and 130 ms waves. At today's rate it is a 4.7 s ring and 1.5 s waves. Neither is right at both speeds.

Set them from the measured rate: the ring holds about 2 s of GPU work (so a confirmation round trip, a snapshot, a generation boundary or a save never drains it), and a block or wave holds about 50 to 100 ms. With recipes, a 2 s ring at 2M/s is 4M entries x 16 B = 64 MB. As blocks that is 20 to 40 blocks in flight, and the per-block CPU control at P3 sizes (30 to 60 ms on 1 to 3 threads) sustains 10 blocks/s only if it runs on more than one thread in parallel across blocks, or the block is 200 ms. I would start with 200 ms blocks and 10 in flight and measure.

Small waves have a second effect: replay latency. A block slot frees at each wave's tail, and with 4 to 8 waves staggered on separate streams a tail comes every wave_length / streams. At 100 ms waves on 4 streams that is every 25 ms, so a click-to-replay waits about 25 ms plus the 50 ms recording, against up to 1.5 s today. Wave tails (the last survivors of a wave running with the SMs under-occupied, about 50 ms at the end of every wave) are hidden by the next wave's blocks as long as one is queued, which the ring guarantees.

The rate-based sizing also gives backpressure for free: the host stops planning when the ring is full and the GPU never waits when it is not. Nothing in the current code needs to be undone; RING_SLOTS, WORK_UNIT and WAVE become functions of the measured rate.

Expected gain: none in rate at steady state. It is what makes the rate sustained across generations, and it turns replay latency from seconds into tens of milliseconds. Measurement: EVOLUTION_STAGE_LOG per-generation rate over 20 generations with world changes and saves during the run (the stall shows as a dip), and replay_seconds p50 and p95 in the UI benchmark. Also p2_speed at WAVE 16k, 32k, 65k, 131k, 262k on 8 streams, to confirm tails cost under 2% at small waves.

### P5. Candidates, not results: the archive prefilter and the descriptor on the device

The group that finishes a creature has the behavior metrics in registers. It computes the descriptor bins (qd::descriptor is a handful of comparisons once the metrics exist), reads its island's per-niche occupant fitness from a device table (5 islands x 1,440 niches x 4 B = 29 KB, L2 resident), and if it beats the occupant or the cell is empty, appends (index, niche) to a candidate list with one atomicAdd and writes its materialized genome to a candidate slab. The CPU commits only candidates, sequentially per island in block order, exactly as today, so the archive semantics (protection, morphology reserve, the record-setter cascade with confirmations) do not change. After the commit the CPU uploads the new occupant table (29 KB) for the next block on a copy engine, which does not need SMs.

The snapshot prefilter is sound for the same reason the code already gives for its parallel prep: an occupant's fitness only rises, so a reject against the table at the block's start stays a reject. The table may be one block stale for blocks in flight; that only lets a few extra candidates through.

docs/rejected-ideas.md rejected "a device-side contender filter" because its ceiling was 0.5 s of a 10 s generation. The generation is now 1.5 s at the target and absorption is 0.6 to 2.3 s, so the ceiling is now 40% to 150% of the generation, not 5%. The reason changed, so the idea comes back.

Expected gain: absorption falls from 0.6 to 2.3 s per generation of 8-thread CPU work to a few tens of ms per block on one thread. Candidates are a few percent of results early and under 1% in a mature archive (derived: 196k results over 7,200 cells; a cell's occupant beats almost every challenger once the archive is old). Measurement: the STAGE_LOG archive seconds per generation, and the count of candidates per block logged for 50 generations.

### P6. The UI thread and the worker: what must never block

The UI is on the Radeon's render device and reads the worker's snapshot through a Mutex held only for the swap (worker.rs, output.lock and view.lock().take()), so the GPU work cannot stall a frame directly. What can: CPU starvation of the egui thread when 8 rayon workers run flat out, and worker command latency, because commands are read between passes and a pass absorbs one block (a few tenths of a second today).

After P1 to P5 the CPU is 80% idle, which solves the first. For the second, block time falls to 100 to 200 ms, so a button press acts within that. What remains is the snapshot: at 2M/s entrants per 200 ms snapshot are thousands early and about a hundred late, so the worker should publish deltas (changed cells and new cards), which the backlog already asks for. I would measure it: snapshot_build_ms p95 (already collected) must stay under 5 ms in the worker; the egui thread's scheduling latency under load from perf sched must stay under 2 ms; frame time p99 at 60 FPS while the game runs at full rate, from the UI benchmark.

A world change at 2M/s costs more than today in creatures: every block already run or running in the old world "enters no archive". With a 2 s ring that is 4M creatures, more than a generation, thrown away per button press. Blocks not yet on an engine are retargeted (scheduler::retarget), so the loss is what is on the GPU: with 100 ms waves and 4 to 8 in flight, the loss is 0.4 to 0.8 s of work. That is the reason to keep waves small even when the ring is deep: the ring is deep in recipes (cheap to retarget on the host: the recipe carries no world) and shallow in launched waves.

### P7. The Radeon 780M: what I would and would not give it

The owner allows considering it. My numbers: any per-creature stage on the iGPU means genes or records crossing to the RTX through host memory, 2 to 5 GB/s at 2M/s, a third of the PCIe budget, plus the desktop's own memory bandwidth, and the crash risk is on the display device. Breeding on the RTX costs 1 to 3% of its time (P2, P3). The iGPU cannot win that trade. Replays on it would need a second physics that is not the scoring kernel, which the owner's rule forbids. I recommend against any per-creature use of the iGPU, and I say so with the numbers rather than the crash story.

The one job I would try on it, if the UI domain wants it: the archive map and the cards are drawn by egui on the Radeon already; a compute pass that renders the 7,200-cell map and the champion thumbnails from the CPU mirror is UI work, not evaluation, and it is the iGPU's proper role. Gain to throughput: zero. Gain to the UI: frame time when the map is dense. Not my call.

## 4. The memory budget after P1 to P5

VRAM: recipe ring 4M x 16 B = 64 MB; archive slab 50 MB; occupant tables 29 KB; results ring 4M x 80 B = 320 MB (or 128 MB at 32 B once the descriptor is computed on the device and the metrics are not read back); candidate slab 20k x 2.5 KB = 50 MB; replay frames 10 MB; kernels. Under 1 GB, against 5 to 7.5 GB today. That leaves 7 GB for agents' GPU work beside the owner's game, which the owner's rule of 2026-09-29 needs.

Host RAM: the CPU mirror of the archives 20 MB, lineage and history as today, pinned readback buffers 2 x 16 MB for results and 2 x 10 MB for candidates, recipe staging 2 x 3 MB. Under 1 GB of RSS, against 5 to 9 GB.

PCIe at 2M/s: recipes 32 MB/s up, occupant tables and CMA versions under 1 MB/s up, results 160 MB/s down, candidates about 50 MB/s down. Under 0.3 GB/s in each direction, against 3 to 8 GB/s up today. Structural children by value during P2 (before P3): 1.7 GB/s up.

## 5. Determinism under out-of-order completion

Three rules keep one seed one search whatever the timing, and the current code follows the first already:

1. Results are addressed by creature index, never by completion order. The kernel does this (results[cidx]). Which lane group runs which creature is decided by an atomic counter and does not matter.
2. Absorption and breeding happen in ring order on one stream. Stream order is ring order; the archive state a block is bred from is the state after the previous block's commit, as today. A block's exact commit runs on the CPU on candidates in block position order per island, as today.
3. Atomics only where the result is order-independent: the take-up counter, atomicAdd to append candidates (their order in the list does not matter because the CPU sorts by position before the commit), and, if the per-niche winner is ever reduced on the device, atomicMax on a packed 64-bit key of (fitness as ordered bits, inverted position), which is the same winner sequential insertion picks. No float atomics anywhere.

The recipe's immutable parent slab id and CMA state version (P2) are what make rule 2 hold across the lag between breeding and take-up.

## 6. What I need from the other domains

- HPC or power domain: the clock tax table. p2_speed with 0, 2, 4, 8, 16 busy CPU threads, with clocks.sm and power.draw at 100 ms. My +10 to +20% for P1 rests on the one measurement of 174k to 137k.
- Kernel domain: the register cost of the take-up work. Materializing a child and unpacking it (P1, P2) happens in the same kernel as the trial. If it raises the kernel above 128 registers the occupancy falls and the whole plan loses. Two ways out if it does: a separate materialize kernel per wave that writes lane records to a wave buffer (that keeps the packed buffer in VRAM, 0.5 GB per wave, but no PCIe and no CPU), or a __noinline__ take-up function whose registers are reclaimed. I need the kernel domain to say which they prefer and what the occupancy hit is.
- Search domain: whether the plan (plan_offspring: emitter choice, parent sampling from least_visited and local competition scores, CMA slot choice) may be computed one block late from a snapshot of the archive tables, so it can run on the device or on a CPU thread in parallel with the commit. It already runs from "the archives as they stand" at absorption; the question is whether a one-block lag changes the search measurably. Also whether the confirmation cascade tolerates a 2 s ring (more speculative confirmations per block).
- Physics domain: nothing changes in the physics, but the take-up code must produce the same lane records as physics2::Model::new. I need a dump format for lane records so the bit-equality gate of P1 can run.
- UI domain: whether they want the delta snapshot (P6) and whether frame time p99 at 60 FPS under full load is measured today.
- Chair: an owner decision on P7 (I recommend no per-creature use of the iGPU, with the numbers above), and on whether the structural port (P3) may be scheduled, since it is the one that makes 2M/s sustained and it is weeks.

## 7. Order of tracks and the gate of each

1. P4 partially now: WORK_UNIT, WAVE and RING_SLOTS from the measured rate; a replay latency and rate measurement. Days. Gate: replay_seconds p95 under 0.2 s at the current rate, end-to-end rate unchanged.
2. P1: genes in VRAM, device unpack. One to two weeks. Gate: lane records bit-equal, p2_speed not slower, STAGE_LOG shows packing gone, RSS and VRAM down.
3. P5: device descriptor and candidate filter, CPU commit on candidates. One week. Gate: archive seconds per generation under 0.1 s, same history for a fixed seed as the full commit over 20 generations.
4. P2: recipes for parametric children, versioned slab. Two to three weeks. Gate: fixed-seed repeat at 3M, emitter rates within noise, breeding seconds under 0.5 s per generation.
5. P3: structural operators on the device, sorted. Three to four weeks. Gate: mutation_audit per operator within noise, CPU per block under 60 ms, CPU load under 20% at full rate.
6. P4 fully: ring in seconds, 200 ms blocks, deltas to the UI. Gate: rate sustained over 20 generations with saves and world changes during the run, frame time p99 under 16.7 ms.

Each step leaves the game playable and better or faster than before it, which is the owner's merge rule.
