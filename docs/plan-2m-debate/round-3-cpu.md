# Round 3, CPU domain: the host pipeline at implementation level

## 0. The mature archive number: not obtained, and why

I ran `evolution-simulator headless --resume save42.evo --generations 52` with `EVOLUTION_PROFILE_BREED=1` from the warp worktree (main at 29bb266, built 18:11) under the shared lock. The load was refused: save42.evo is physics version 42 and main is version 43 (the bump came with 8af4324, the lane-group kernel). No worktree on the machine is at version 42 with a built binary, and the disk is at 98% with 9 GB free, so a version-42 build (4.5 GB of target) is not safe inside the debate. The measurement needs either a save produced on version 43 (a 50-generation headless run, hours at today's rate) or a version-42 checkout built after the owner frees disk. I keep my estimate of 30 to 60 ms per 196k block and data's prediction of under 20 ms for the refresh as estimates, and the design below bounds the archive stage by construction rather than by that number: anything over 20 ms per 100k block goes incremental (section 2.6).

## 1. The pipeline

Ring order, as today, with one change: absorb and breed overlap through a versioned elite table.

1. The kernel finishes a creature, computes its descriptor bins and a candidate bit against the 29 KB occupant table (data), and writes an 80 B result into the pinned results ring.
2. The worker thread takes the front block's results in ring order and runs the verdict (confirmations for record setters, as today).
3. Commit, on the worker plus one thread per island: candidates only go through `offer`; every result still feeds the screen log, emitter attempts, the failed count, the nursery ranking and the CMA sample lists. Each commit appends new elites to the island's slab and bumps the island's version.
4. Plan for the re-bred block: `plan_offspring` reads the archives at the committed version and produces one 16 B plan per slot.
5. Emit on the pool: parametric plans on the vector path, structural plans sorted by operator on the scalar path. Every child is written once, as SoA genes, straight into the pinned gene ring at its slot. No `Population`, no `ChildBatch`, no `append_batches`, no pack, no staging copy.
6. The engine bumps the take-up counter for the block; the RTX unpacks genes at take-up (Model::new on the device, gpu's track).

Steps 3 and 4 of block k+1 run on the worker while step 5 of block k runs on the pool, because emit reads parents by immutable slab index and never by archive entry.

## 2. The pieces

### 2.1 Bounded arrays

`Bounded<T, const N: usize>`: `[MaybeUninit<T>; N]` plus a `u8` length, `Deref` and `DerefMut` to `[T]`, `push`, `insert`, `remove`, `swap_remove`, `retain`, `extend_from_slice`, `truncate`, `clear`, `iter` through the slice, `Clone` copying `len` elements only, `PartialEq` on the slice. About 150 lines with tests. `Creature` becomes `nodes: Bounded<NodeGene, 32>`, `bones: Bounded<Bone, 32>`, `muscles: Bounded<Muscle, 96>`, `id: u64`: 6.5 KB on the stack, which is fine for a rayon worker (2 MB stacks). Config already caps bodies at 32 and 96, so `push` past capacity returns `false` and the operator reports "does not fit", the same path as today's four retries. Operator temporaries (candidate index lists, the 129 `collect` sites) become `Bounded<u8, 32>` or `Bounded<u8, 96>`. The property test that runs every operator on 160 grown bodies runs unchanged and is the gate that no operator's logic moved.

### 2.2 The RNG

`fn draw(seed, generation, round, slot, gene, draw) -> u64`: the six keys multiplied by six odd constants and xored, then the splitmix64 finalizer twice. Integer only. `unit()` is the top 24 bits over 2^24. `gaussian()` sums twelve 16-bit slices of three draws, subtracts 6 x 65535, converts once with `as f32`, multiplies by one constant. The parametric path keys by gene index, so a gene's noise is independent of lane order; the scalar path keeps a per-child cursor in `draw`, which preserves today's sequential `rng.unit()` semantics inside an operator. A child is a function of (seed, generation, round, slot, plan) and nothing else. The stream changes, which ga ruled a new seed; `qd::VERSION` bumps.

### 2.3 The vector parametric path

Per parent topology, built once and cached by elite slab index: a flat layout of the parent's genes (4 floats per node, 7 per bone, 13 per muscle: about 340 floats at the generation-51 mean body of 8.7 nodes, 7.7 bones, 19.2 muscles), a sigma per position (the constants in `local_mutation`: 0.10, 0.08, 0.025, 0.10 for nodes, 0.035 for rest length, the range and organ terms, the muscle terms), a clamp low and high per position, and a "wrap" mask for phase and reset. The CMA emitters already hold mean, variance and path as flat vectors in the same order (`exploring_parameters`), so they use the same loop with their own sigma vector and the rank-one path term. The loop, 16 lanes on AVX-512: `g = parent + scale * sigma * gaussian(hash)`, then `min(max(g, lo), hi)` or `rem_euclid` under the mask, then scatter into the child's SoA slot in the pinned ring. Multiplicative genes (stiffness, organ mass, tempo) take `exp` of a gaussian: one `exp` polynomial per lane, 16 at a time. The per-child coin flips (tendon growth, sensor change, the whole-body tempo) stay scalar on the child cursor. Then `repair` over bounded arrays: union-find on at most 32 nodes, muscle clamps, `shape_head`, canonical bone order (an insertion sort of at most 31 bones). Estimate at the generation-51 mean body: 340 gaussians at 3 cycles, 1,000; clamps and scatter, 400; repair and canonical order, 500; the cached layout lookup, 100. About 2,000 cycles, against about 50,000 today.

### 2.4 Operator-sorted structural breeding

At plan time the structural plans of a block are counting-sorted by operator index (64 buckets, microseconds) and each bucket is split into rayon chunks of 512 children. One thread runs one operator over hundreds of children in a row: the operator's code and branch history stay hot, and the donor elites it reads (graft, copy-limb) stay in L2. The child still depends only on its slot. Estimate per structural child at the generation-51 body: 8,000 cycles unsorted after the container rewrite, about 6,000 sorted; p90 bodies (12 nodes, 29 muscles) about 1.5x that.

Parent-major order for the parametric path is the same trick and is my new idea (section 8).

### 2.5 plan_offspring

Per slot today: one `Rng::new`, `choose_emitter`, a parent draw (`sample_local_competitive` is 8 probes into two `Vec`s, `sample_novel` a 32-entry scan, `sample_morphology` a scan of the reserve), the protection lookup and the mate draw. About 400 to 800 cycles at today's archive sizes, all reads. Per block: `top_parents` and `orders` sort each island's entries (about 1,500 entries x 5 islands, under 1 ms), the optimizer targets, the emitter weights. With ga's 1,024 emitters the plan also picks a CMA emitter per CMA child, a lookup in a per-island list. Estimate: 600 cycles per slot, 1.8 G cycles per 3M, 0.45 core-seconds, and it parallelizes over slots as it does today. The chair's gate of 2,000 per slot leaves 3x of margin. What would break it: `sample_novel`'s `least_visited` scan growing with the archive; the fix is a per-island heap maintained at commit.

### 2.6 The commit on candidates

Results arrive in position order in the pinned results ring. The worker walks the block once: for every result it updates the screen log, the emitter attempt and reward counters, the CMA sample list of its emitter (score and slot), the nursery lists and the failed count, about 50 cycles each. Candidates (the kernel's bit: beats its island cell's occupant, or the cell is empty) are pushed to their island's list with their descriptor recomputed on the CPU from the metrics in the result (2,000 cycles, on under 5% of results). The five island lists commit in parallel, in position order per island, through `offer` and `offer_morphology` exactly as today. The global archive walk runs on the worker over candidates only. Then the tells: an emitter whose accumulated sample count crossed 200 tells in ring order (ga), about 0.2 M cycles per tell at 178 dimensions with mu 100; at 1,024 emitters and 1.05 M CMA children per generation that is about 5,000 tells, 1 G cycles, 0.25 core-seconds. `refresh_behavior_scores` runs only for islands that inserted, as today; if it measures above 20 ms per 100k block on a mature save, it becomes incremental: an insert updates the novelty of its k nearest entries and its own, O(k N) instead of a full pass. Estimate for the whole stage: 0.15 G cycles of bookkeeping plus 0.3 G of candidate offers plus 1 G of tells per 3M, about 0.35 core-seconds, of which the sequential part per 100k block is under 10 ms on the worker.

### 2.7 The versioned elite table

Each island archive owns an append-only slab `Vec<Creature>` (heap creatures, about 1.4 KB each at generation 51) and a `version: u64`. `Elite` holds a slab index instead of an owned creature. A commit that replaces an occupant appends the new creature and bumps the version; the old record stays until reclaimed. A plan carries the slab indices of parent and mate, so emit reads immutable records while the next commit appends. Reclamation at the generation boundary: records not referenced by any live elite and older than the deepest block in flight are dropped and the slab compacted (a few MB per generation). Memory: 7,500 live elites at 1.4 KB is 10 MB, plus a few thousand replaced records per block. Saves write the archives as today; the slab is rebuilt on load. This is data's device slab moved to the host, and it is what lets 50 ms blocks keep the GPU full: the serial chain per block (verdict, commit, plan) is under 25 ms, and emit runs one block behind on the pool.

### 2.8 Threads and pinning

Rayon at 14 threads, nice 10, `SCHED_BATCH` (os). Pool threads pinned to hardware threads 2 to 15; the worker on CPU 0, the CUDA engine thread on CPU 1 (its SMT sibling), the UI thread and the compositor free on any core but favored by the short EEVDF slice os proposes. The NVIDIA interrupt lands on CPU 7 today; harmless. No thread ever holds a block's worth of work: emit chunks are 512 children (about 3 ms), commits are per block (under 20 ms), so the UI thread never waits behind more than one chunk on any core.

## 3. Core-seconds per 3M at generation 51

Mean body 8.7 nodes, 19.2 muscles; p90 12 nodes, 29 muscles. Mix from ga: 60% mean-like, 30% p90-like, 10% large.

| stage | cycles per creature | per 3M, G cycles |
|---|---:|---:|
| plan | 600 | 1.8 |
| parametric emit, 60% of children | 2,000 (mean), 3,000 (p90) | 4.0 |
| structural emit, 40% of children | 6,000 (mean), 9,000 (p90), 12,000 (large) | 10.6 |
| repair is inside the two rows above | | |
| commit, bookkeeping, tells, candidates | 500 | 1.5 |
| generation boundary (history, nursery graduation, migration, screen bar) | | 0.5 |
| total | about 6,100 | 18.4 |

18.4 G cycles is 4.6 core-seconds at 4.0 GHz, 5.4 at os's 3.4 GHz cap. At 2M/s a generation is 1.5 s, so 3.1 to 3.6 cores busy all the time, 15 to 20 W of package, a host tax of 7 to 9% at the power cap by the chair's rule. That is above the 2 to 3 cores the chair's table assumes, and I say so rather than round it down. The two biggest terms are the structural rows: if breed_bench shows them at 12,000 rather than 6,000 at the mean body, the total goes to 8,500 weighted, the chair's insurance rule fires, and data's parametric recipes on the RTX take the 4.0 G of the parametric row off the host. That leaves the host at 14 G cycles, 3.5 core-seconds, still 2.3 cores at 2M/s, because the structural work stays.

Host ceiling: 14 threads on 7 cores with SMT worth about 1.25x on the branchy path give about 8.5 core-equivalents at 4 GHz, 34 G cycles/s, 5.6M creatures/s in throughput. The latency bound: the serial chain per 100k block is under 25 ms against 50 ms of GPU work at 2M/s, and the ring of 0.3 to 1 s holds 6 to 20 blocks, so the GPU never waits on the host below about 4M/s. The host is not the wall in this design at any row of the chair's table.

## 4. breed_bench

`examples/breed_bench.rs`. Input: a save at the current version, or `--evolve N` to grow an archive at 200k from a fixed seed first (no version-43 mature save exists yet). It breeds 10 blocks of 100k with no GPU: archives loaded, plans made, children emitted into a pinned ring, results faked. Output: thread-nanoseconds per child per emitter and per operator from `CLOCK_THREAD_CPUTIME_ID` (the TSC is unstable here), with `perf stat -e cycles,instructions,page-faults` around the process; `plan_offspring` nanoseconds per slot; allocations per child from a counting allocator; genes per child (mean and p90), so the number is tied to body size; a 64-bit digest of all children, which must match between `RAYON_NUM_THREADS=1` and 14 and between two runs of one seed. Gates: weighted cycles per child under 8,000 (chair), plan under 2,000 per slot, zero allocations on the parametric path and at most 2 on the structural, digests equal, every operator property test green.

## 5. The multiplier table, at 3T and 5T, generation-51 mix

Base: 43 to 45M creature-steps/s at generation 3 is about 27M at generation 51 (the chair's 0.6x), and 480 steps. My levers are the host tax and the host ceiling; the kernel and step levers are the chair's working ranges.

| lever | 3T | 5T | note |
|---|---:|---:|---:|
| per-lane kernel, 2 substeps, creature-steps/s | 300M | 450M | chair's range |
| steps per creature, R1 to R3 | 300 | 300 | conservative ladder |
| host tax (3.1 to 3.6 cores at the cap) | 0.92 | 0.92 | this design |
| creatures/s, 2 substeps | 0.92M | 1.38M | row 1 of the chair's table |
| 1 substep, if it passes | 550M | 800M | physics, 50% odds |
| creatures/s, 1 substep, R1 to R3 | 1.69M | 2.45M | row 3 |
| creatures/s, 1 substep, R1 to R4 at 235 | 2.15M | 3.13M | row 4 |
| host ceiling (throughput) | 5.6M | 5.6M | not a wall |

The design lands on row 3 at 5T (2.45M) and row 3 at 3T (1.7M). Without 1 substep it stops at 0.9 to 1.4M/s, and the host tax is then the smallest of the three factors keeping it there. If 1 substep fails, the cheap-fidelity screen (1.2x on top of R1 to R3) gives 1.1 to 1.65M, and 2M/s needs the 5T budget plus the full ladder: 450M / 235 x 0.92 = 1.76M. Honest answer: without a physics change, 2M/s sustained at generation 51 is not reached on this laptop.

## 6. World change, save, replay, 60 FPS, determinism

World change: plans and genes carry no world; blocks not on the engine are retargeted as today, the engine stops bumping the take-up counter and the kernel drains (data), drained results enter no archive, and the loss is the ring depth of 0.3 to 1 s, not today's 26% of a generation.

Save: archives by value (the slab is not saved), CMA states, emitter statistics, screen state, history, lineage. Load rebreeds the ring as today; the cold-ring burst (os's 350k-cycle run was this) takes about 0.3 s at the new cost.

Replay: the elite's genes are on the host; one recording wave, as today; the worker's step is at most one block, so nothing waits behind it.

60 FPS: no pool unit over about 3 ms, worker serial work under 25 ms per block and preemptible by the UI thread's short slice, rayon at 14 threads, the delta snapshot (data). Gate: os's frame-time histogram, p99 under 10 ms during a full generation.

Determinism: the search is a function of (seed, absorption order, the versioned archive state at each commit, the plan and RNG rules). Thread count, chunking, block timing and wave finish order never enter; the breed_bench digest proves it for the host half on every commit.

## 7. What stays out

Radeon breeding and packing (igpu withdrew them; the fallback trigger is the CPU rewrite failing at more than 3 busy cores, which section 3 says it will not). Recipes on the RTX: insurance behind the breed_bench gate. CPU physics: never, by the power argument of round 1. Huge pages: dropped. A CPU pre-trial: dropped. Bit equality between two breeders: not needed, because one implementation breeds each child (data's rule, which I accept; the integer-exact gaussian stays the rule if a second breeder is ever built).

## 8. One new idea: parent-major breeding order

At generation 51 the archives hold about 7,500 elites and a generation breeds 3M children, so each parent has about 400 children per generation, about 27 per 100k block. Today the children of one parent are scattered across the block by slot, so every child fetches its parent's 1.4 KB of genes from L3 or DRAM: about 25 cache lines, 300 to 500 cycles of the child's 2,000 on the parametric path, and the same again for the cached sigma and clamp layout of section 2.3. Sorting a block's parametric plans by parent slab index (a radix sort of 60k keys, well under 1 ms) puts each parent's 27 children on one thread back to back, so the parent, its layout tables and its CMA vectors stay in L1 and L2 across the run. The child is still a function of its slot, so the search does not change. Estimate: 300 to 500 cycles saved per parametric child, 15 to 25% of that path, 0.6 to 1.0 G cycles per 3M, about 5% of the whole host cost, and the same principle extends the operator sort of section 2.4 (sort structural plans by operator, then by parent). Measurement: breed_bench with `--order slot` against `--order parent`, cycles per parametric child and L2 misses from `perf stat -e l2_cache_misses_from_dc_misses`. Small, but it is free and it compounds with every other host lever.

## 9. Tracks in order, with gates

1. Preallocated arenas and the pinned gene ring written once by the breeder; the CPU pack reads the SoA ring until the device unpack lands (gpu's track), then is deleted. Gate: page faults near zero, staging memcpy gone, STAGE_LOG breeding and packing down, same history for a fixed seed.
2. Bounded arrays, the counter-based RNG, breed_bench with the digest test, `qd::VERSION` bump. Gate: weighted cycles under 8,000 at a generation-51-like body, digests equal at 1 and 14 threads, operator property tests green. Over 8,000 opens data's recipes track.
3. The vector parametric path. Gate: under 2,500 cycles per parametric child at the mean body, zero allocations, digest unchanged from track 2.
4. Operator-sorted structural breeding and parent-major order. Gate: structural under 7,000 cycles at the mean body; the order A/B on breed_bench.
5. The versioned elite slab, absorb and breed overlapped, 50 ms blocks with ga's tells per 200 samples. Gate: the null engine sustains 4M results/s at 100k blocks; same history as the unoverlapped code over 20 generations.
6. Commit on candidates, once the kernel writes the candidate bit. Gate: archive stage under 20 ms per 100k block on a mature save, same history as the full commit over 20 generations, candidates under 5%.
7. Threads, pinning, SCHED_BATCH, the UI slice (with os). Gate: p99 frame time under 10 ms, control latency p99 under 5 ms.

Every track leaves the game on main with no flag, and each has a number that decides whether it stays.
