# Round 1, GPU domain: what AD107 can do and how to get 25x

Author: the NVIDIA Ada and CUDA kernel expert. Date: 2026-09-30. Sources: 00-facts.md, hpc.md, shaders/warp_creature.cu, src/warp_kernel.rs, src/cuda_engine.rs, src/ring.rs, src/scheduler.rs, the speed logs in scratchpad/warp (speed1.txt, final.txt), docs/rejected-ideas.md. No new GPU runs were made for this round. Every number marked "estimate" is derived in the text and names the measurement that replaces it.

## 1. The ceiling in one table

The chip: 24 SMs, 4 schedulers per SM, one warp instruction per scheduler per clock, 2.49 GHz observed under load. That is 239 G warp instructions/s, or 7.65 T thread instructions/s with all 32 lanes useful. The MIO pipe (shuffles, shared-memory loads and stores, barriers) is one warp instruction per clock per SM, so it is a quarter of the FP32 issue rate: 60 G warp MIO instructions/s for the whole chip. L2 is about 3.8 TB/s theoretical (24 SMs x 64 B/clk). DRAM is 256 GB/s and irrelevant (1% used).

The physics as specified costs, counted pass by pass from warp_creature.cu for a 7-node, 6-bone, 12-muscle body with one lane doing all the work and no overhead (estimate):

| section per substep | thread instructions |
|---|---:|
| kinematics (sincos, positions, velocities) | 120 |
| muscles (12 x about 90: anchors, length, waveform, Hill, energy, tendon, ledger, two force_at) | 1,080 |
| articulated-body backward pass (6 x 70) | 420 |
| root inverse and forward pass | 120 |
| contact detection and selection | 200 |
| contact matrix walk and root closure | 440 |
| projected Gauss-Seidel, 5 sweeps x 4 contacts | 500 |
| contact response, backward and forward | 210 |
| integration, momentum balance, first-law ledger | 160 |
| total per substep | about 3,250 |
| per step (2 substeps + metrics) | about 6,650 |

So the ideal kernel for this physics needs about 6,650 thread instructions per creature-step. At 7.65 T/s that is 1.15 G creature-steps/s at 100% issue and 100% lane use. At a realistic 55% issue that is about 630 M creature-steps/s. At 550 steps per creature that is 1.15 M creatures/s. This is the ceiling for a perfect kernel of today's physics and today's trial policy. The three levers that multiply beyond it are: instructions per step (physics cost), steps per creature (trial policy), and issue efficiency (kernel engineering). 2M/s needs the kernel near its ceiling plus about 1.8x from the other two levers together. It is not reachable by kernel engineering alone on this chip.

Power: the 100 W cap is shared with the CPU. An FMA-dominant kernel draws more per instruction than the shuffle-bound kernel of today, so expect the clock to settle 5 to 15% below 2.49 GHz once the kernel is issue bound. Every busy CPU core costs GPU clock (8 threads once cost 21%). The GPU rate at 2M/s presumes a nearly idle CPU. Measurement: `nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -lms 100` during a 5-minute run of the new kernel, with and without 4 busy CPU threads.

## 2. Where today's kernel loses 25x

Today: 43 to 45 M creature-steps/s, 128 registers, 4 blocks of 128 threads per SM (16 warps), 19 to 21 KB shared per block.

The lane-group design maps node i to lane i and runs every tree pass level by level. Every level costs the full per-level instruction stream for all 32 lanes, but only the lanes whose bone is at that level do work. For a 7-node body with levels of 1, 2, 2, 1 bones, the four tree passes per substep (articulated-body backward, forward, response backward, response forward) use 6 of 32 lane-levels: 19% lane use. The contact matrix walk uses the walker lanes only: 2 to 4 of 8 lanes. The muscle rounds are the good part: 12 muscles over 8 lanes in 2 rounds, 75%. Weighted by the profile in 00-facts.md (contacts 45 to 50%, muscles 15 to 18%, articulated body 10 to 14%), the whole kernel runs at roughly 25 to 30% lane use.

The exchange between levels goes through shuffles and shared memory with a `__syncwarp` around each. I count per substep, per warp of 4 creatures: about 250 shuffles, about 150 shared loads and stores, and about 35 barriers. That is about 430 MIO instructions per warp-substep. At 45 M creature-steps/s with 4 creatures per warp and 2 substeps, that is 22.5 M warp-substeps/s, so about 10 G MIO instructions/s, one sixth of the 60 G/s pipe. Not the wall by itself, but each of those instructions has 20 to 30 cycles of latency, and with 4 warps per scheduler the latency shows as stalls. Estimate of issue efficiency today: about 6,800 warp instructions per warp-step (my count of the stream, 4 creatures) at 11.25 M warp-steps/s is 76 G warp instructions/s, 32% of peak. The measurement that replaces this: Nsight Compute (`ncu --section WarpStateStats --section SchedulerStats`) on p2_speed. ncu is not installed. I need it (section 7).

So the 25x decomposes as: lane use 0.27 to about 0.85 (3x), issue efficiency 0.32 to about 0.55 (1.7x), instructions per creature-step (the physics cost) 1.4 to 2x, steps per creature 1.4x. The product is 8 to 12x from the kernel and 2 to 2.8x from the other domains. That is the mix I propose.

One honest correction to hpc.md: it predicted that reduced coordinates would cut instructions per step by 2.5 to 3x. The measured kernel runs 45 M creature-steps/s where the old maximal-coordinate kernel ran an estimated 250 M. The new physics costs about 5x more per step on this GPU, and the exact contact solve is half of it. The physics bought its invariants with that budget. Every proposal below respects the invariants unless it says so.

## 3. Proposals, ranked by expected gain

### P1. Lanes per creature by body size: W = 1, 2 or 4, with node-count-specialized kernels (expected 6 to 10x on the kernel)

The reverse of the last move. The per-thread physics-v2 kernel that lane groups beat by 1.65x was not a fair per-thread kernel: it compiled to 243 to 255 registers with 40 to 4,560 B of local memory per thread (speed1.txt), because the node count was a runtime value up to 32 and every per-node array went to local memory. Lane groups won by removing spills, not by parallelism. A per-thread kernel whose node count is a compile-time constant (NVRTC already compiles per world; add `NODES` as a define, classes 4, 6, 8, 12) unrolls every node loop, keeps every array in registers, and has 100% lane use in the muscle and contact sections.

The register budget decides W. Live state per creature at 8 nodes, W = 1 (estimate): joint state 16, constants 64 (32 if packed to 16-bit pairs), node positions and velocities during muscles 32, articulated-body temporaries 72, contact rows 32, muscle scalars 24: about 200. That exceeds 128, so 8 warps per SM at 255 registers. At W = 2 (lane j owns bones j, j + 2, j + 4, ...) each lane holds half: about 110 registers, 16 warps per SM, and each tree level of a typical tree has 1 to 2 bones, so the two lanes are both busy at most levels. Parent-child exchange costs one shuffle per word per level: 9 words x 4 levels x 4 passes = 144 shuffles per warp-substep for 16 creatures, 9 per creature-substep, against 60 today. At W = 1 for bodies of at most 4 nodes there are no shuffles at all.

Throughput estimate at W = 2 for 7 nodes: a lane runs half the 3,250 stream plus about 150 for shuffles and syncs plus 15% divergence waste (bodies in a warp sorted by node count, muscle count and depth, which the packer already does) is about 2,100 warp instructions per 16 creature-substeps, 131 per creature-substep, 275 per creature-step with metrics. At 239 G warp instructions/s that is 870 M creature-steps/s at 100% issue, 480 M at 55%. At W = 1 for small bodies, about 240 per creature-step: 1.0 G at 100%, 550 M at 55%. Against 45 M today, 10 to 12x.

Two hazards, both measurable:

1. Muscle constants. 12 muscles x 64 B = 768 B per creature. At W = 2 a lane holds 6 muscles; held as float that is 96 registers, too many. Held in global memory and reloaded every substep, the load is 16 B x 12 x 2 x 1.1 G creature-steps/s = 400 GB/s at the target rate through L1, and per SM the resident set (256 creatures x 768 B = 192 KB) does not fit in L1, so it streams from L2 at about 0.4 to 1.7 TB/s depending on reuse: 10 to 45% of L2 peak. The fix is to pack the muscle record to 32 B (16-bit fields for anchors, amplitude, hill, period, phase, duty, stiffness, cap, capacity, tendon, slack; 5-bit lane indices) and either hold it in registers (6 x 8 words = 48 registers at W = 2, unpacked on use, about 1.5 instructions per value) or in shared memory ([field][creature] layout, 12 x 32 B x 512 creatures = 196 KB, too big; 256 creatures = 98 KB, just fits with no room for anything else). The register form is what I expect to win. This is FP16 storage of constants, not FP16 arithmetic; the values are rounded once when the genome is quantized, so every engine and every replay sees the same constants and determinism holds. The GA domain must accept 16-bit muscle genes (section 7).

2. Divergence on the contact count. Every creature in a warp takes the max contact count of the warp. With 16 to 32 creatures per warp, that is usually 4. Today the lane-group kernel also runs `ncmax` of its 4 creatures. The PGS at 4 contacts is 500 of the 3,250 per substep. I budget it at 4 always.

Measurement: examples/p2_speed on the gen-10 save at 262,144 creatures, 3 timed passes, exclusive GPU, for each of W = 1, 2, 4 and each node class; `--ptxas-options=-v` registers and spills per variant; nsys GPU metrics for SM issue active. Gate for keeping the track alive: 200 M creature-steps/s on the first working variant. Gate for merge: at least 350 M, results bit-equal to the current kernel on the same creatures under the same substep count (the arithmetic order per creature can be kept identical, since only the mapping changes; if the packed constants change bits, compare distances instead and require the elite median ratio within 0.01).

Risk: 3 W values x 4 node classes x worlds is many kernels. NVRTC compile is 1 to 2 s each and the prefetch thread already compiles in the background; cache to disk as now. Bodies above 12 nodes stay on the lane-group kernel (W = 16, 32 classes), which is measured and correct.

### P2. Cut the instruction count of the physics on the GPU's terms (expected 1.3 to 1.5x, needs the physics domain)

These are the cuts I can price. The physics domain must say which keep the invariants.

- Waveform and Hill drive once per step, not per substep. The muscle's target changes at 60 Hz, only the velocity term varies within a step. Saves about 25 per muscle-substep: 300 of 3,250 (9%).
- The first-law ledger accumulates in every muscle and every lane every substep and is only used when there is no contact. Compute it only in the flight branch: about 120 per substep (4%).
- Contact matrix built once per step and reused for the second substep when the contact set is the same. The matrix depends on the pose, which moves by 1/120 s. Saves the walk and closure once per step: 440 of 6,650 (7%). Risk: the PGS then solves an exact LCP of a slightly stale matrix. The friction-work invariant is enforced against velocities from the same matrix, so it holds by construction. The physics domain must measure foot slip on elites (size_report) with this.
- PGS sweeps 4 to 2. The team measured 4 vs 8 as no change; 2 is untested. Saves 200 per substep (6%). Measurement: the elite median distance ratio and foot slip at 2 sweeps.
- Contact count histogram. If 90% of substeps have at most 2 contacts, a kernel variant for MAXC = 2 with a fall-back is worth 300 per substep. Measurement: a counter of nc per substep on the gen-10 save (a 20-line PROFILE addition).

Together, if all pass: about 1,100 of 3,250 per substep, 1.5x. If only the safe ones (waveform once per step, ledger in flight only): 1.14x.

### P3. Steps per creature (expected 1.3 to 1.6x, needs the GA and ML domains)

44% of all GPU work today is the 20% of creatures that run the full 1,200 steps. Every option here is a trial policy the owner rules on.

- Screen at 3 s instead of 5 s: mean steps 0.8 x 180 + 0.2 x 1,200 = 384 against 540: 1.4x. Measurement: the fraction of the final top 1% and top 10% that a 3 s bar keeps, the same test the team ran for the 10 s rung.
- A second rung at 10 s keeping half: measured 1.3x with every top-1% creature kept.
- A GPU-side early stop by a predictor from the first second (COM velocity, height, contact pattern, the metrics already in registers). If it ends 60% of the screened creatures at 60 steps: mean 365 steps, 1.5x. The ML domain owns the recall number. Anything below 99% recall on the top 10% is not worth it.

### P4. Sustaining the rate across generations (no gain, but without it the GPU starves at 2M/s)

At 2M/s a wave of 262,144 creatures is 0.13 s of GPU time and a ring block of 196,608 is 0.1 s. Today's host costs per block: pack 0.24 s on 4 threads, breed 0.31 s (4.7 s per 3M). The GPU would idle two thirds of the time and the CPU work would cost 20% of the GPU clock. What the GPU side needs:

- Pack on the GPU. A pack kernel takes a compact genome (about 500 B: nodes, bones, muscles as the genes) and writes the lane records, muscle records and end lists on the device. It is embarrassingly parallel and costs microseconds per wave. Upload drops from 1.3 KB to 0.5 KB per creature (1 GB/s at 2M/s, PCIe 4.0 x8 is 12 GB/s practical). The data-pipeline domain owns the genome record.
- A device-resident ring. 786k creatures x 500 B = 400 MB on the device. The host sends breeding decisions for parametric children (parent slot, operator, seed: 16 B) and full genomes only for structural children. The CMA and gaussian emitters then run as a kernel. This is the QDax lesson and it also removes the pinned staging copies (1.2 to 1.6 GB RSS).
- Results stay at 80 B per creature (160 MB/s). Archive insertion on the CPU is fine at that rate if it is not the same thread that breeds.
- Waves in flight: keep 8 streams as now; wave tails (the survivors' last 900 steps run on a few warps) overlap across streams. Measurement: nsys timeline of SM active over 30 s at the target rate; the gap between waves must be under 5%.
- CUDA graphs are not needed: 8 launches per second per stream is nothing.

### P5. Kernel engineering inside whatever mapping wins (expected 1.2 to 1.4x)

- `-lgc` clock pinning for measurements only (owner allowed root for it).
- `--use_fast_math` is already the effect of the current options; keep `__sincosf` and `__cosf` (SFU) and check they are not the wall: 12 muscles x 1 cos + 6 bones x 1 sincos per substep is 18 SFU ops per creature-substep, 1.1 G x 2 x 18 = 40 G/s against the SFU rate of 24 SMs x 16/clk x 2.49 GHz = 950 G/s. Not a limit.
- Metrics once per step already; keep the ballot-based fall detection.
- Blocks of 64 threads instead of 128 give the scheduler finer granularity when creatures finish at different times and the block's last warp waits for regeneration; today regeneration is per group, so blocks never wait. Keep 128.
- `__launch_bounds__(BLOCK, MIN_BLOCKS)` per W: at W = 2 target 16 warps per SM (128 registers); at W = 1 allow 255 and 8 warps. Measure both; the rejected-ideas file says caps of 96 and 80 lost 2 to 15% on the lane-group kernel, which is the expected shape.

### P6. What I looked at and reject with numbers

- Tensor cores for the contact closure or the articulated inertias: the closure is a 4 x 4 block matrix of float4 per creature, about 200 instructions of 6,650 (3%), and TF32 changes bits. No.
- FP16 arithmetic: same rate as FP32 on Ada. No. FP16 storage of constants: yes (P1).
- A wavefront split of the substep into separate kernels with state in global memory: the state per creature is 400 B and a substep is 3,000 instructions, so the store and reload would be 10% overhead for no occupancy gain, since registers are not the limit at W = 2. No.
- Warp-specialized producer warps that prefetch the next creature's constants: a creature loads once per 300 to 1,200 steps. Nothing to hide.
- The Radeon 780M: 12 CUs at about 2.7 GHz is 8 TFLOPS nominal with dual issue, 3 to 4 TFLOPS realistic, so at most 20 to 25% of the RTX and it shares the CPU package power that Dynamic Boost already trades against the GPU. It cannot contribute to 25x. It could add 15% at the end if the WGSL kernel is kept alive and the desktop stays responsive. Not in my plan before the CUDA kernel is within 1.5x of the goal.
- CPU AVX-512 evaluation: 1 TFLOPS peak, 7% of the GPU, and it costs GPU clock. No.

## 4. The combined estimate

| lever | conservative | optimistic | owner |
|---|---:|---:|---|
| P1 lanes per creature with node classes and packed constants | 6x | 10x | no |
| P5 tuning inside the mapping | 1.2x | 1.4x | no |
| P2 physics cost cuts | 1.14x | 1.5x | physics |
| P3 steps per creature | 1.3x | 1.6x | yes |
| kernel total (creature-steps/s) | 370 M | 630 M | |
| end to end (creatures/s) | 0.9 M | 2.3 M | |

The conservative end is 6x today and lands at about 0.9 M/s. The optimistic end reaches 2 M/s only if every lever lands, and the optimistic kernel number is at the 55% issue ceiling of section 1. I put 2 M/s sustained at 40% likely on this laptop with all four levers, 0% with the kernel alone.

## 5. Order of tracks with gates

1. Install ncu and profile the current kernel (half a day). Gate: stall reasons and issue active measured, not estimated.
2. P1 at W = 2 for the 8-node class only, with muscle constants in global memory (a week). Gate: 200 M creature-steps/s on the 8-node subset of the gen-10 save. Then packed constants in registers. Gate: 350 M.
3. W = 1 for at most 4 nodes and W = 4 for 12 nodes; mixed populations. Gate: 300 M on the whole gen-10 save, bit-equal or elite-ratio-equal scores.
4. P2 cuts one at a time with the physics domain's slip and stability tests.
5. P4 pack kernel and device ring, with the data-pipeline domain.
6. P3 trial policy with the GA domain, measured on the new kernel.

The game stays playable at every step because each variant is a new kernel behind the same `advance` interface and the lane-group kernel keeps the classes the new one does not cover yet.

## 6. Determinism

The per-lane mapping has no atomics in the arithmetic. The atomic counter assigns creatures to lanes and results are written by creature index, so the result of a creature never depends on which lane ran it. Warp-uniform loop bounds (max node count, max contact count) change which instructions run but not the values, provided the inactive iterations are masked, not merged. Packed 16-bit constants round once at breeding. The confirmation trials at 4x rate keep their own kernel.

## 7. What I need from the other domains

Physics: (a) a histogram of contacts per substep on an evolved population; (b) a yes or no on the waveform once per step, the ledger only in flight, the contact matrix once per step, and 2 sweeps, each with its slip and stability test; (c) the node-count and depth distribution over 100 generations so I size the classes (4, 6, 8, 12 is my guess) and know how often the lane-group fallback runs.

Genetic algorithms: (a) the mean steps per creature under a 3 s bar, a 4 s bar, and a 10 s second rung, and what each keeps of the final top 1% and 10%; (b) whether muscle genes may be stored as 16-bit values (anchors, amplitude, hill, period, phase, duty, stiffness, tendon, slack); (c) the muscle count distribution, because muscles are a third of the instructions and the packed record's size.

ML acceleration: a "will be screened" predictor from 1 s of trial features, its recall on the final top 10%, and its cost in instructions if it runs in the kernel at step 60. Nothing else from ML; there are no dense products in this physics.

Data pipeline: a compact genome record of at most 512 B that a pack kernel can read; a device-resident ring of 786k genomes (400 MB); breeding decisions as 16 B records for parametric children; results at 80 B. How the ring order and the block absorption survive when breeding runs on the device.

CPU: breeding of a 196k block in under 0.1 s on at most 4 threads, or the parametric emitters moved to the GPU and only structural mutation left on the CPU; packing off the CPU entirely; a measurement of the GPU clock against the number of busy cores (1, 2, 4, 8) so the power trade is a number.

OS: Nsight Compute installed and `NVreg_RestrictProfilingToAdminUsers=0` so ncu runs without root; `nvidia-smi -lgc` for benchmarks; whether nvidia-powerd or the platform profile can bias Dynamic Boost toward the GPU while the game runs; a 10-minute thermal trace of GPU clock and power at full load to confirm the sustained clock.

iGPU: nothing in this round. If the CUDA kernel reaches 400 M creature-steps/s, tell me what fraction of a wave the 780M could take through the WGSL kernel without touching the desktop's frame time.
