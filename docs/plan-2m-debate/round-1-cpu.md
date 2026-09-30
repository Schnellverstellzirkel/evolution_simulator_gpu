# Round 1, CPU domain: the Ryzen 7 7840HS in a 2M/s game

Author: CPU architecture expert. Sources: 00-facts.md, hpc.md, the code on main at 29bb266 (src/ring.rs, src/storage.rs, src/evolution.rs, src/qd.rs, src/warp_kernel.rs, src/cuda_engine.rs, src/scheduler.rs, src/engine.rs), and this machine's sysfs. Numbers marked "derived" come from arithmetic on measured values. Numbers marked "estimate" are mine and each comes with the measurement that would replace it.

## 1. What the CPU is here

Zen 4, 8 cores, 16 threads, 5.14 GHz single core, about 4.0 to 4.3 GHz with all 8 cores under vector load. AVX-512 is present (F, BW, DQ, VL, VNNI, BF16, VBMI2, IFMA, VPOPCNTDQ) and runs as two 256-bit halves, so it gives masking and fewer instructions, not more FLOPs per clock. Two 256-bit FMA pipes per core: 32 FP32 FLOPs per clock per core, about 1 TFLOPS for the chip. 32 KB L1d and 1 MB L2 per core, 16 MB L3. The Radeon 780M sits in the same package and draws from the same package power limit and the same DRAM bus. This machine runs amd_pstate in active mode with the performance governor and the performance energy preference, SMT on, transparent huge pages set to madvise only, nvidia-powerd active (so Dynamic Boost is live), and the RTX power limit at 85 W now (default 55, maximum 105). perf is installed, perf_event_paranoid is 1, and the RAPL energy counters exist but are root-only to read.

The one fact that decides this domain: CPU work costs GPU clock. The facts file measured 8 busy CPU threads cutting the GPU from 174k to 137k trials/s, a 21% loss. An all-core AVX-512 load on this chip is 45 to 54 W. Dynamic Boost moves that from the GPU's side. So every proposal below is judged twice, on wall time and on watts.

## 2. What the CPU does today, in cycles

Derived from the facts and the code. Per 3M generation:

- Breeding (plan, emit, repair, write into the block's arenas): 4.7 s wall on the 8-thread rayon pool. That is about 37 core-seconds, 12.5 core-microseconds per child, roughly 50,000 cycles per child.
- Packing (warp_kernel::pack: per creature `pop.creature(i)` with three Vec allocations, `Model::new` with about eight more, then `fill_creature`): 0.27 to 0.32 s per 262k wave on 4 threads, 3.1 to 3.7 s wall per generation, about 4.5 core-microseconds and 18,000 cycles per creature to write a 1.5 KB record.
- Archive stage (descriptors in parallel, island offers on one thread per island, the global archive loop on one thread, `refresh_behavior_scores` on changed islands): 0.6 to 2.3 s wall per generation in the last measurement I have.
- Upload staging: the packed wave is copied once more from pageable Vecs into pinned memory before `cuMemcpyHtoDAsync` (cuda_engine.rs line 1340). At 400 MB per wave that is about 40 ms of one core per wave.
- Page faults: every block allocates fresh arenas (about 180 MB of genes per 196k block) and every wave allocates fresh 400 MB record Vecs. With THP on madvise and no madvise call, the kernel zeroes about 150,000 4 KB pages per wave. Estimate: 30 to 50 ms of kernel time per wave, spread across the filling threads.

Total: about 9 to 11 s of CPU wall time per generation, 55 to 65 core-seconds, against a 20 s GPU generation today. The chain hides behind the GPU now. At 2M/s the generation is 1.5 s.

Where breeding spends its cycles (reading the code, not measured): a child is 7 nodes, 6 bones and 12 muscles, about 900 B of genes. The parametric path (`local_mutation`, `CmaEmitter::sample_exploring`) draws about 160 gaussians per child through scalar Box-Muller (ln, sqrt, cos each), clones the parent into three Vecs, clones again into a `values` Vec, then `repair` clamps everything, `canonicalize_bone_order` sorts, `ChildBatch::push` copies the three Vecs and frees them, and `append_batches` copies once more. The structural path (64 operators, `structural_mutation_from`, up to four tries, `graft_from` with up to four tries) is branchy scalar code over Vecs. Nothing here is vectorized and the allocator is called about ten times per child. 50,000 cycles for 900 B of output is the allocator, the scalar transcendentals and the copies, not the mutation logic.

## 3. The pipeline math at 2M/s

The ring has 4 blocks of 196,608 slots. The worker thread absorbs the front block (verdict, archive, breed) and queues it; the engine thread packs it and uploads it. Both use the same 8-thread rayon pool. Ring order is fixed, so the CPU must finish the front block before the GPU runs dry on the other three.

Derived: at 2M/s the three other blocks are 0.29 s of GPU work. Today's per-block CPU chain is archive 40 to 150 ms plus breed 307 ms on the worker, plus pack 240 ms on the engine thread, contending for the same pool. Even with perfect overlap the worker's own path is 0.35 to 0.45 s per block, so the CPU chain alone caps the game near 196k / 0.45 s, about 440k creatures/s, and with pool contention nearer 300k/s. This is a derived number, and it says the CPU chain becomes the wall at 2 to 3x today's rate, long before 2M/s. The measurement that confirms it: a null engine that returns a fixed result per creature the instant it is submitted, run through `examples/worker_rate.rs` or the game with `EVOLUTION_STAGE_LOG`. The rate it reports is the CPU chain's ceiling with no GPU in the picture. I want this number before round 2.

The budget at 2M/s. If the CPU is allowed 8 cores at 100% it delivers about 32 G cycles per 1.5 s generation, 10,700 cycles per creature. But 8 cores at 100% is 45 to 54 W and costs the GPU about 20%. A CPU budget that leaves the GPU its clock is about 10 to 12 W: three cores at 3.5 GHz, or the whole chip at 25% duty. That is about 10 G cycles per generation, 3,300 cycles per creature for plan, emit, repair, archive and whatever packing remains on the CPU. Today it spends 73,000. The CPU chain needs a 20x cut in core time, not a 7x cut in wall time.

## 4. What the CPU should not do: physics

The old fast CPU engine measured 9,800 evolved creatures/s at 60 s trials on 6 threads (hpc.md), which is 35M creature-steps/s, about 760 cycles per creature-step with 16 creatures per SIMD group. That was 80% of today's GPU kernel rate, which surprised me. With the reduced-coordinate physics (2.5 to 3x fewer instructions) a rewritten AVX-512 lane could reach an estimated 250 to 350 cycles per creature-step, about 100M creature-steps/s on 8 cores, 8 to 10% of the 1.0 to 1.2 G creature-steps/s the target needs. It would cost 45 to 54 W. The measured Dynamic Boost penalty for that load is 21% of the GPU rate. At 2M/s that is 400k creatures/s lost for about 170k gained. Net loss of about 230k/s, and worse as the GPU gets faster. On this laptop, with nvidia-powerd coupling the budgets, the CPU physics lane is a loss and stays out. The measurement that would overturn this: run the GPU at full load and add an 8-thread synthetic AVX-512 load at 3.0 GHz capped clock; if the GPU rate drops less than 8%, the sum could be positive and the lane is worth building. I expect it drops 15% or more.

The same power argument applies to the iGPU, with a different answer. The Radeon 780M is inside the CPU package: 12 RDNA3 compute units, about 4 TFLOPS realistic at 15 to 20 W. That is 3 to 4x the CPU cores' FLOPs per watt. If any package silicon runs physics, it is the iGPU, and then the CPU cores must sit near idle, because both draw from the same package limit. I hand that to the iGPU domain with one warning: the 780M's memory is system DRAM, shared with the CPU chain's 7 to 16 GB/s of streaming traffic and the desktop's scanout.

## 5. Proposals, ranked by expected gain

Gains are on the CPU chain's ceiling and on the GPU's power. The chain's ceiling must go from about 300 to 440k/s to above 2M/s with margin, so the required product is about 6x on wall time and 20x on core time.

### P1. Breeding writes the child once, into pinned memory, in the layout the GPU reads

Expected: packing goes from 12 to 15 core-seconds per generation to zero, the staging copy goes to zero, and the page-fault cost goes to zero. That is about a quarter of today's core time and the whole engine-thread stage. Derived: 3.5 s of wall per generation removed.

The child is written exactly once, by the breeding thread, into a preallocated pinned host buffer (`cuMemAllocHost`, double-buffered per stream, reused forever, so no page faults after the first touch). What is written is the compact SoA genome, about 900 B, not the 1.5 KB lane record. A GPU prologue kernel (GPU domain) turns genes into lane records on the device: it is a trivially parallel walk over 32 nodes and is where `Model::new` belongs, because it runs once per creature and the GPU has 3,072 lanes to spend on it. PCIe traffic drops 40% (900 B against 1.5 KB per creature; at 2M/s that is 1.8 GB/s on a 16 GB/s link). The archive keeps its genome view of the same bytes. The class sort (rounds, depth, nodes) stays on the CPU as a `par_sort` of 196k small keys, about 5 ms, or becomes a device radix sort.

Risks: the GPU pack must produce bit-identical records to the CPU pack (it is integer and copy work plus `Model::new`'s float constants; those must be computed the same way, which is a determinism test, not a physics change). Pinned memory is limited by the driver; 4 waves of 240 MB in flight is 1 GB, fine. If the GPU domain rejects device-side packing, the fallback is P1b: fuse pack into emit on the CPU with `Model::new` rewritten over fixed `[_; 32]` arrays and no allocation. Estimate 18,000 cycles per creature to 2,500. Measurement for either: `perf stat -e cycles,instructions,page-faults` around the breed and pack stages with `EVOLUTION_PROFILE_BREED`, and `EVOLUTION_STAGE_LOG` breeding and evaluation columns before and after.

### P2. Split breeding into a vector path and a scalar path

The emitter mix starts at CMA 35%, structural 35%, novelty 30%, and novelty applies a structural operator to 18% of its children. So about 60% of children are parametric: the same body plan as the parent, genes equal to parent plus sigma times a gaussian, clamped, then a trivial repair. That is a structure-of-arrays vector job and belongs on 16 AVX-512 lanes, or on the GPU (P3). The other 40% are structural and stay scalar.

Vector path estimate per child: 160 gaussians at about 2.5 cycles each with a 16-lane polynomial Box-Muller (log, sqrt and cos as short polynomials on the two 256-bit halves), 400 cycles; clamps and the muscle short-below-long ordering, 200; copy in and out, 200; allocation-free repair (union-find over at most 32 nodes on the stack, muscle clamps), 400. About 1,500 cycles against 50,000. CMA sampling is the same shape (mean, variance, path vectors per emitter, already flat `Vec<f32>`), so it takes the same path.

Scalar path estimate: the 64 operators rewritten over fixed-capacity stack arrays (`MAX_NODES` is 32, muscles at most 96) with no Vec, no clone of the parent through the allocator, and `graft_from` reading the donor in place. 50,000 to about 8,000 cycles per child. The logic does not change, only the containers. This path is branchy and cache-missy (archive entries, donors), which is exactly where SMT pays: run it on 16 threads and expect 1.2 to 1.3x per core at almost no extra power.

Weighted: 0.6 x 1,500 + 0.4 x 8,000 = 4,700 cycles per child, 14 G cycles per generation, about 3.5 core-seconds. That fits 1.5 s on 2.5 cores. It is over the 3,300-cycle power budget from section 3 by 40% until P3 moves the vector path off the CPU.

Determinism: replace the sequential per-slot `Rng` stream with a counter-based generator, a hash of (seed, generation, slot, gene index, draw index). Each gaussian then depends on its own coordinates and not on the order the lanes drew it, so 16-lane batches, scalar code and a GPU thread all produce the same child. Same-slot children stay identical across thread counts, which the search requires. The gaussian must be a fixed polynomial with FMA use pinned (`mul_add` on the CPU, no contraction surprises), or the GPU and CPU versions will differ in the last bit. Risk: the search changes because the random stream changes; that is a new seed, not a new search, and `qd::VERSION` bumps anyway.

Measurement: a `breed_bench` example that breeds one block of 196k from a fixed save and reports cycles per child by emitter (perf stat, instructions and IPC), plus the null-engine ceiling from section 3.

### P3. Ship plans, not children, for the parametric 60%

The plan for a parametric child is 16 bytes: parent index in the device-side elite table, emitter, sigma, and the slot (the RNG coordinates). If the GPU holds the elite table (a few thousand elites at about 1 KB each, a few MB, refreshed per block with only the changed cells), then the GPU breeds the parametric child in a prologue kernel with the same counter-based RNG and the same polynomial gaussian. The CPU's breeding work drops to the structural 40%: 0.4 x 8,000 = 3,200 cycles per child on average, 9.6 G cycles per generation, 2.4 core-seconds, inside the power budget. PCIe drops to about 30 MB/s for plans plus the structural children's 900 B each.

The archive still needs the genes of the children that enter it. They are 0.1 to 1% of a block; the GPU writes back the genes of any creature whose result passes the block's prefilter bar, or the CPU regenerates them from the plan with the identical function, which is the point of counter-based determinism: the child is a pure function of (elite table version, plan). The second option needs no readback at all.

Expected: CPU core time per generation from about 60 core-seconds today to about 4 to 5 (P1 plus P2 plus P3). The CPU chain's ceiling goes from about 300 to 440k/s to an estimated 4 to 6M/s, limited then by the archive stage. Risk: two implementations of the parametric emit (CPU for the archive's regeneration and the search_ab tool, GPU for the ring) must agree bit for bit; that is a test on every commit, not a physics question. This proposal is for the GPU and data-pipeline domains to shape; I own the CPU half.

### P4. Hand the watts to the GPU

Zen 4 at 5.0 GHz runs near 1.35 V, at 3.4 GHz near 1.0 V. Energy per instruction goes roughly with V squared, so the same core-seconds of breeding done at 3.4 GHz cost about 55% of the energy and 45% more wall time. Once P1 to P3 land, the CPU chain has wall time to spare, so a frequency cap converts spare time into GPU watts. Estimate: 10 to 15 W back to the GPU, which on the Ada laptop power curve near 85 W is 5 to 10% more GPU rate. That is 100 to 200k creatures/s at the target for a one-line sysfs change.

The controls, in order of what the game can do itself: spread breeding over all 16 threads at lower priority rather than 8 at full boost (many cores at low duty is the efficient end of the curve, and the firmware lowers all-core clocks itself when 16 threads are active); pin the rayon pool to one SMT sibling per core and leave the other siblings to the UI thread, the CUDA driver's threads and the compositor, so 60 FPS does not fight the breed for a run queue. What only root can do: `cpupower frequency-set -u 3.4GHz` or the energy preference set to `balance_power` (the OS domain decides whether the owner sets this once; the game must not change the system by itself, per the owner's rules), and the RTX power limit raised from 85 to its 105 W maximum with `nvidia-smi -pl`, if nvidia-powerd does not already move it. Measurement: sweep the CPU cap over 2.5, 3.0, 3.4, 4.0 and unlimited GHz during a steady GPU-bound run, logging creatures/s, `nvidia-smi` power.draw and clocks.sm, and package RAPL (root reads `/sys/class/powercap/intel-rapl:0/energy_uj`, or `perf stat -e power/energy-pkg/` after paranoid is set to 0). Five runs of two minutes each, exclusive GPU lock.

### P5. Huge pages and no fresh allocations on the hot path

The arenas per block (180 MB) and, until P1 lands, the wave records (400 MB) are allocated fresh and faulted in every time. Pool them: a ring of arenas sized for the ring, `madvise(MADV_HUGEPAGE)` on each, prefaulted once at start. With THP on madvise this is the only way the game gets 2 MB pages. Estimate: removes 30 to 50 ms of kernel time per wave and cuts dTLB misses on the random parent and archive reads. 5 to 10% of the breed and pack wall time. Measurement: `perf stat -e page-faults,dTLB-load-misses,dTLB-store-misses` and `perf record` showing `clear_page_erms` in the kernel share before and after.

### P6. Run structural operators sorted by operator

Sort a block's structural plans by operator index before emitting (determinism is unaffected, the child is a function of its slot). One thread then runs one operator over thousands of children in a row: warm instruction cache, trained branch predictor, the operator's donor and helper tables hot in L2. Estimate: 1.3 to 1.5x on the scalar path, so 8,000 to about 5,500 cycles per structural child. Measurement: the `breed_bench` cycles per child by operator, sorted against unsorted.

### P7. The archive stage at 2M/s

The prefilter (descriptors) is parallel and about 2,000 cycles per creature, 6 G cycles per generation on 8 threads: 0.2 s. The island offers run one thread per island over about 40k results per block, most rejected in a few hundred cycles at the cell compare: about 10 ms per block. The global loop is one thread over 196k results, about 10 ms. `refresh_behavior_scores` on a changed island is the unknown: it is a parallel pass over the entries and I have no cost for it. Per block the archive stage should fit in 30 to 60 ms after the block-level `Topology` clones in the per-island maps are replaced by references or interned ids. It stays on the CPU: it is sequential in block order by design, it is hash-map work, and it is small. Measurement: `EVOLUTION_PROFILE_BREED` timings already print the seven sections; run one generation of a mature save and read them.

### P8. The ring as a latency buffer

Ring order makes CPU latency per block a hard bound on GPU utilization: CPU latency for block k must be below the GPU time of the other blocks. With 4 blocks that is 0.29 s at 2M/s, tight for a stage that includes a `refresh_behavior_scores` and a block's breed. Raise the ring to a generation in flight, 16 blocks of 196k (3.1M slots), and the bound becomes 1.4 s. Cost: about 3 GB of host RAM for genes (30 GB here), and selection feedback delayed by one generation, which QDax found harmless at batch sizes far above this (hpc.md 5.5). The GA domain should confirm the search is indifferent to a generation of lag. The pipeline domain should size the pinned buffers for it.

### P9. What the CPU can do for a surrogate, if the ML domain wants one

VNNI on Zen 4 does about 64 to 128 int8 multiply-accumulates per cycle per core. A 160-input, 64-hidden, 1-output MLP is 10,000 MACs: 100 to 150 cycles per child. At 3M children per generation that is 0.4 G cycles, 0.1 core-seconds. CPU inference of a small surrogate is free at 2M/s. Training on the last generation's 3M rows at that width is about 30 GMAC per epoch, 0.1 s on the CPU with BF16 or VNNI. So if the GA and ML domains decide a "falls in the first second" or "below the screen bar" predictor keeps the spirit (a screened creature enters no archive, which the owner allows), the CPU can run it at breed time at no cost to the budget. I make no claim about whether it should exist.

## 6. What does not pass from my domain

- AVX-512 physics on the CPU as a scoring lane: section 4, net loss of about 230k/s at the target through Dynamic Boost.
- A CPU pre-trial (a 30-step CPU simulation to catch fallers before upload): 3M x 30 steps at 300 cycles is 27 G cycles, 7 core-seconds per generation, more than the whole budget, and fallers already stop the GPU trial in about 30 steps. Not worth 5% of GPU time.
- FP16 or BF16 arithmetic in breeding: the genes are 900 B; the cost is allocation and transcendentals, not bytes.
- More rayon threads on today's code: the allocator is the bottleneck and it serializes on the thread caches' refill; measure, but expect little.

## 7. Ceiling and the answer to the open questions, from this seat

Question 1 (where the 25x comes from): none of it from the CPU as compute. The CPU's contribution is negative today above about 400k/s (the chain becomes the wall) and its job is to disappear from the critical path and from the power budget. P1 to P3 take the CPU chain from about 60 core-seconds per generation to 4 to 5, and the chain's ceiling from about 300 to 440k/s to 4M/s or more. P4 adds an estimated 5 to 10% GPU rate through power.

Question 2 (the ceiling under 100 W): the budget is shared. Every 10 W the CPU takes costs the GPU 5 to 6% at today's operating point (derived from the 21% loss at an estimated 35 to 40 W of CPU load). A 2M/s game has the CPU at 10 to 12 W average and the GPU at 85 to 105 W. Whether the GPU reaches 1.0 to 1.2 G creature-steps/s at that power is the GPU domain's question; my domain says the CPU can be made to fit inside 10 to 12 W with P1 to P4, and cannot add compute without taking more than it gives.

Question 3 (data flow): the CPU writes each child once, into pinned memory, in SoA genome form, or writes only a 16-byte plan for parametric children. Nothing is packed on the host. The archive and the structural operators stay on the CPU. Determinism comes from a counter-based RNG keyed by slot and gene, which makes the child a pure function that any device can evaluate.

Question 4 (order of tracks): first the null-engine measurement, because it converts my derived 300 to 440k/s chain ceiling into a fact. Then P1 with pooled pinned buffers (P5 rides along), because it removes an entire stage and the game stays playable at every step. Then P2 with the counter-based RNG (a `qd::VERSION` bump). P3 and the GPU prologue kernels after the GPU domain has the elite table. P4 whenever the owner runs the root commands, measured in an afternoon. P6 and P7 as the profile says.

## 8. What I need from the other domains

GPU: a cost and a design for two prologue kernels, genes to lane records (P1) and parametric child from plan plus elite table (P3), and whether a device-side elite table of a few MB can be refreshed per block without stalling the persistent kernel. Also: does the persistent kernel take a stream of 900 B genome records, or does it want the lane layout in host memory as today?

iGPU: the 780M shares the CPU package power limit and the DRAM bus with the CPU chain's 7 to 16 GB/s. I need its measured creature-steps per second per watt against the GPU's, and a statement on desktop safety, before the CPU domain plans around it. If the iGPU takes physics, the CPU chain must be near idle, and P4's cap becomes mandatory.

OS: who owns the root knobs (CPU frequency cap or energy preference, RAPL readability, THP, the RTX power limit, nvidia-powerd's configuration), and whether the owner sets them once. I also need the P4 sweep run under your control of the power counters, and the thread pinning policy for the UI thread and the compositor.

Genetic algorithms: the steady-state emitter mix on a mature save (how much of breeding is parametric), whether the search tolerates one generation of ring lag (P8), and whether a screen-time predictor at breed time keeps the spirit (P9). Also whether the counter-based RNG (a changed random stream) is acceptable as "a new seed".

Physics: which of `Model::new`'s derived constants must be computed on the host and which can be recomputed on the device bit-identically, and the distribution of steps per creature on a mature save, because the CPU per-child budget scales with the GPU steps per creature.

Data pipeline: pinned buffer sizing for a 16-block ring, the block latency bound in section 3 as a design invariant, and whether results can carry the genes of archive candidates back (P3, first option) or the CPU regenerates them (second option).

ML acceleration: whether a small surrogate exists that the GA domain would accept; the CPU can run it at 100 to 150 cycles per child (P9), so cost is not the question, only search quality.
