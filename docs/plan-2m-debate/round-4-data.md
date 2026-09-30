# Round 4, data flow: attacks on K, answers on P

Sources: gpu and physics round 3, src/cuda_engine.rs (prefetch, kernel cache), src/scheduler.rs (recover, retire), the kernel cache on disk (1,465 cubins, 498 MB, no eviction).

## Attacks on Design K

### 1. Registers: the spill is not the risk, the runtime index is

gpu's 140 at W = 2 under a 128 bound spills 12 registers, 48 B per lane. L1 and shared memory share 128 KB per SM; at 150 B of shared per lane and 512 lanes the carveout is 100 KB, so L1 has 28 KB. Spills of 48 B x 512 lanes (24 KB) fit; at 64 B (32 KB) they do not, and every reload is an L2 hit at about 200 cycles instead of 30. Bandwidth is fine (115 GB/s, 3% of L2); latency is not: 12 dependent reloads per substep add about 10% to a 3,000-instruction critical path at 30 cycles and about 40% at 200 unless 16 warps hide it. So the number is 32 B of spill with L1 hits, gpu's own gate; at 64 B W = 2 goes to 3 blocks per SM, and by the measured occupancy curve 12 of 16 warps gives about 0.85x: 365M becomes 310M and the harmonic mean loses 10 to 15%.

The larger risk is what killed the per-thread kernel: runtime-indexed arrays, not the register count. Its 40 to 4,560 B of local memory came from node arrays indexed by a runtime node count. Design K fixes 4 nodes per lane, which is right, but "per-contact data 24 registers" and "up to 12 extra rows" indexed by a runtime contact slot go to local memory whatever the register count, reported as a stack frame, not as spills. Detection: ptxas -v stack frame and lmem bytes beside spill stores. What must change: contact rows unrolled at a fixed MAXC with predication, as today's kernel does, and the active-set loop over compile-time slots.

### 2. The 45% issue rate: name the mechanism or drop a row

Today's lane-group kernel measures 20 to 32% issue-active. K's mechanisms to reach 45%: no __syncwarp in the substep (today about 35 barriers per warp-substep), 9 shuffles per creature-substep instead of 60, and 15 independent muscles for ILP. Two things push back. The muscle scatter to shared memory is about 190 MIO ops per lane-substep, 6% of instructions; the MIO pipe caps at 25% of issue, so that is safe. The split muscle record (24 B per muscle per substep from L2) is a 200-cycle stall per substep per warp unless all 16 loads issue before the first use; that ILP is the first thing spills break. I cannot name a measured mechanism from 30% to 45%, and nobody can before the stub. The cheapest predictor exists today: nsys warp-state sampling on the lane-group kernel (30 minutes, shared lock), reading what share of stalls are barrier and short-scoreboard. If under 30%, removing them cannot reach 45%. At 35% issue-active W = 2 gives 280M, the harmonic mean 190 to 310M, and the table drops a row to 0.65 to 1.0M/s at 2 substeps; below 35% the stub gate sends the design to W = 1 at 255 registers and loses 20% more.

### 7. In-kernel confirmation: a second implementation of the fine trial, for a ring that does not shrink

Three problems, in order of weight.

Determinism seam. The re-run at 4x rate inside the scoring kernel makes the substep length a runtime parameter. Today HS, INV_HS, DT, SAMPLE and air_sub are #defines and the fine fidelity is a separate compiled kernel (KernelKey carries fidelity), and it also changes the solver passes, not only the rate. The cascade cases gpu keeps on the host still run through that fine kernel. Two implementations of the same trial then produce the min(standard, confirm) score, and unless they are bit-equal the score of a record depends on which path ran, which depends on the table version at take-up, which depends on timing. Either every confirmation goes through the kernel (and cascades re-enter the queue with a confirm flag, which brings back the round trip gpu wanted to remove) or the host path is deleted. One implementation is the rule the chair set in round 2, and this breaks it.

Registers. Runtime rate means six persistent registers (HS, INV_HS, DT, the sample interval, the screen tick scaling, air per substep) in a kernel that already spills at 128. That is the wrong kernel to add live state to.

The ring does not drop to 0.3 s. gpu's claim rests on the confirmation round trip being the p95 term of the host chain at 100 to 150 ms. With the 3 reserved slots the round trip is a 4x trial alone, about 70 ms, plus a 1 ms poll, and there is no wave wait because the kernels are persistent. The per-block chain itself (commit plus breed of 100k) is 80 to 100 ms by cpu's numbers, so the p95 is the chain, not the confirmation, and the ring is 0.5 s either way. What in-kernel confirmation buys is 3 block slots (3%). What it costs is a determinism seam and six registers in the tightest kernel. It also does not skip a needed confirmation (gpu is right: the record word only rises, so a stale word is at most a superset), except after a meteor or extinction lowers the record, where the host asks for the missing one and pays one round trip. Verdict: keep host-asked confirmations on the reserved fine kernel; revisit only if the measured p95 chain is under 60 ms and the confirmation term dominates it.

### 9. Kernel variants and cold compiles

Count: per world flag set, 3 classes x (standard, record, fine) = 9 kernels today, and Design K keeps 9 (W = 2, 4, 8) or 18 if both substep counts stay compiled. World flags are 11 bits, but a session visits 10 to 30 flag sets (14 buttons plus autochange every 100 generations). Compile is 1 to 2 s per kernel on the prefetch threads. The disk cache is keyed by source hash, so every kernel edit cold-starts every world, and it has no eviction: 1,465 cubins and 498 MB today on a disk at 98%.

On a change to a world this build has not seen, the drain takes 20 ms and the GPU idles until the three standard kernels exist: 2 s compiled in parallel, 6 s serially; 20 to 60 s of idle per session across new worlds, zero after. What must change: compile the three standard kernels of a new world in parallel at nice 19 and the fine and record variants after; prefetch the neighbours of the current world (each single-button toggle, 14 x 3 kernels, about 60 s of background compile after every change, so the next press is warm); show "Compiling the new world" in the status line; evict the cache by age at 200 files, about 70 MB. K's one source templated on W also means one edit recompiles every variant.

## Answers on Design P

### 10. Zero-copy against DMA: DMA plus a prologue kernel wins, for a reason nobody stated

Not latency. Issue slots. In a per-lane kernel lanes take up at different times, so a take-up runs with one lane active while 31 are predicated off, and the warp still issues every instruction. An in-kernel unpack of 3,000 instructions per creature then costs the warp 3,000 warp-instructions per creature, against a trial that costs the warp about 480 x 1,600 / 32 = 24,000 warp-instruction-equivalents per creature: 12%. In the lane-group kernel the same effect is 8 of 32 lanes active at W = 8: about 3%. A prologue kernel unpacks 32 creatures per warp at full lane use, 100 warp-instruction-equivalents per creature, 0.4%, and writes lane records to VRAM: 150 MB per 100k block, 1.5 GB for 10 blocks in flight. Take-up then reads a 1.5 KB record from VRAM, about 48 loads at L2 latency, issued by one lane: under 1%. So I concede the form to gpu, with the reason corrected: DMA the block's genes to VRAM on a copy engine (2.7 GB/s, no SM), run the prologue unpack per block, take up from VRAM records. VRAM budget rises to about 2 GB (genes 1.35 GB at a 1 s ring plus records for 10 blocks), still 6 GB under the card. The copy engine does not compete with the results ring, which is kernel writes, not copies.

The one-day measurement: p2_speed on save42.evo with the lane-group kernel in three forms: records read from VRAM (today), records read zero-copy from mapped pinned memory, genes read from VRAM with the unpack at take-up. The decision rule: any form within 2% of the first is acceptable; the in-kernel unpack's cost at W = 8 predicts the per-lane cost at 4x. I expect the third form to lose 3 to 5% on the lane-group kernel and 10 to 15% on a per-lane kernel, which is the number that closes zero-copy genes.

### 11. Persistent kernels, faults, out-of-memory, agents on the GPU

Failure modes: a CUDA error in our kernel or an Xid from any process, including an agent's, destroys the context and with it the queues, the mapped pages and every persistent kernel; out-of-memory cannot hit a running persistent kernel but can hit a relaunch after a drain if an agent took VRAM meanwhile. Today scheduler::recover reopens the engine with a 1, 4, 10 s backoff and resubmits unfinished units with their exact inputs; a third failure retires the GPU.

The path in P. Arenas and the results ring are ordinary host memory registered with cuMemHostRegister, not cuMemAllocHost, so a context loss frees nothing on the host. Recovery: reopen the context (1 to 2 s), load cubins from the disk cache, re-register the arenas, reset every head to its tail, re-append the refs of every block whose completion word is unset, relaunch. A partly done block re-runs whole; its creatures are pure functions, so the bits and the history are unchanged; at most 1 s of work repeats. Meanwhile the worker answers commands and the status line says the GPU is reopening. After three failures the game pauses with a message, which is where the current code lands too once cpu_v2 is deleted; docs/building.md should say so.

Agents beside the game: the RTX time-slices between contexts, so the game loses the agent's share, as today; 2 GB of VRAM leaves 6 GB. The new interaction is a relaunch after a drain while an agent holds memory: the engine backs off and retries with fewer blocks resident, the shape of today's memory guard. The design survives the owner's rule.

### 12. Starvation and the generation boundary

The mapped-word spin is the symptom; the ring depth is the cure, and my rule has a hole: 5x the p95 per-block chain excludes the boundary block, because at 30 blocks per generation the boundary is 3% of blocks and the p95 skips it. The rule becomes: depth = clamp(5 x p95 chain, boundary max over the last 3 generations plus 2 blocks, 0.3 s, 1 s).

The boundary on a mature archive is unmeasured (save42 is version 42 against main's 43). Reading end_generation: push_archive_stats sorts the 3M-float screen log (about 100 ms), prune_lineage, graduate_nurseries, migrate_islands every 25 generations (copies plus refresh_behavior_scores on the hub), plus in P ml's fit on 240k audit rows and 1,024 emitters' housekeeping (ms each). Estimate 150 to 400 ms, 500 to 800 ms on a migration generation. A 1 s ring covers it; a 0.5 s ring starves 100 to 300 ms once per 25 generations, under 1%, visible as starved seconds in the stage log. If the boundary exceeds 0.8 s the sort becomes a streaming quantile and migration runs on the versioned elite table from another thread. Measurable now: a fresh 3M game on the current build for 3 generations with EVOLUTION_PROFILE_BREED prints the boundary line; the archive is young, but the sort and the graduation are at full size. One minute under the shared lock.

### 19. Bucketed queues: taken twice or never

The hole: a lane does atomicAdd(head) and gets an index at or beyond the tail, then retires (drain flag or bucket dry). The index is consumed, the ref never ran, and head is past tail. Two rules close it. First, a retiring lane never consumes: it checks tail before the atomicAdd, and a stale read only makes it spin one more round. Second, the host requeues only after the kernel has fully exited (cuStreamSynchronize on its stream after the drain flag), then sets head to tail and re-appends every ref whose result slot still holds the sentinel the host wrote at submit (one word per ring creature, 125 KB of bookkeeping). Under those rules: while a kernel runs, head is monotone and every index below tail is taken exactly once; across a relaunch, a ref is re-appended if and only if its result is unset, so nothing is lost, and a creature completed between the sentinel scan and the exit is at worst run twice, which gives the same bits. The set of live creatures at a world change depends on bucket timing, which the chair allows.

### 23. Glitch risk in the data path

- Prologue unpack replacing Model::new: a constant off by one bit changes the physics silently. Guard: the bit-equal dump on 262k creatures as an ignored GPU test run on every kernel or Model change.
- Results written by the kernel into pinned memory: a torn 80 B read gives a garbage fitness that could enter an archive. Guard: __threadfence_system before the completion word, an acquire load on the host, and the world tag written last; the CPU rejects an unset or foreign tag.
- World-tag exclusion: without it an old-world score enters the new-world archive. Guard: the world-change test asserts no old-tag result was offered.
- Re-run after a context loss: same bits by purity; guard: the two-runs test under EVOLUTION_SIMULATE_GPU_LOSS.
- Candidate bit and bucketed queues: no glitch path; results by index, commit against the real archive.
- K's in-kernel confirmation: the seam of item 7 (a record scored by the weaker of two fine trials); the guard is one implementation.
- Replay pre-recording: frames from the RECORD variant of the scoring kernel, as today.

## Defence, one paragraph

The chair's summary has P reading genes zero-copy at take-up; after item 10 that is withdrawn in favour of DMA and a prologue kernel, and my VRAM budget is 2 GB, not under 100 MB. Everything else in P stands as written, with the ring rule amended in item 12 and the requeue rules of item 19 added.
