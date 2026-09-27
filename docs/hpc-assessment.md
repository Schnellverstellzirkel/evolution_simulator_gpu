# Making the simulator much faster: an HPC assessment

Date: 2026-09-28. Author: Claude, at the owner's request. Scope: how the evaluation pipeline could run an order of magnitude faster on the hardware we have, and what that would take. This is research and analysis. No code changed for it.

The owner's game was running on the RTX 4060 while this was written, so no new GPU measurements were taken. Every number marked "measured" comes from `docs/performance-log.md`, `docs/performance-campaign.md` or the runs of 2026-09-27. Numbers marked "estimate" are derived in the text and should be checked with the profiling plan in section 9.

## 1. Summary

The current engine was tuned hard over the last two days: 36,000 creatures/s end to end at 3M creatures became 185,000 to 216,000/s. That tuning worked inside the existing design. The design itself now sets the ceiling, and the ceiling is well below 2M/s.

The arithmetic is simple. At 2M creatures/s, with today's screening policy, the GPU must run about 2.7 billion creature-steps per second. The RTX 4060 Laptop can issue about 7.6 trillion thread instructions per second at its observed clock. With realistic efficiency that leaves a budget of roughly 1,200 to 1,600 instructions per creature-step. Today's kernel spends an estimated 4,000 to 5,000 and uses about 20% of the GPU's issue slots. Both numbers have to move by about 3x at once. No single knob does that.

Three structural changes can close most of the gap. They are independent and multiply:

1. A cheaper physics formulation (section 7.2). Most of today's instructions go to keeping a chain of point masses rigid: two projection passes, a joint-limit pass, a parent-first rebuild, corrective rules for lift and friction, and a settling phase. A planar articulated tree in reduced coordinates is rigid by construction. It removes all of those passes, and its cost per step is a small multiple of the node count. Expected: 3 to 5 times fewer instructions per step. This changes the physics and needs the owner's approval.
2. A persistent GPU kernel that keeps every creature's state in registers for its whole trial, and refills a lane from a work queue the moment its creature falls or is screened (section 7.3). This is the path-regeneration technique from GPU ray tracing. It removes idle lanes, the 64-step reload of state, and the host round trips that segments need today. Expected: 1.3 to 2 times.
3. Control over registers, shared memory and profiling, either by moving the kernel to CUDA or by a language that targets both CUDA and Vulkan (section 7.4). Today the driver decides occupancy and we cannot see instruction-level stall reasons. Expected: 1.2 to 1.8 times, and much faster iteration on everything else.

Together these give an estimated 5x to 18x on the same laptop, so 1M to 3.5M creatures/s. The 2M/s goal is reachable, but only if the physics formulation changes. Tuning the current formulation further tops out at an estimated 300,000 to 400,000/s. A desktop GPU would add a near-linear factor on top (section 8).

What I recommend doing first is not code. It is one to two days of proper profiling (section 9), because every estimate above rests on an instruction count that has not been measured directly.

## 2. The hardware

### 2.1 GPU: RTX 4060 Laptop (AD107, Ada Lovelace)

| property | value | source |
|---|---|---|
| streaming multiprocessors (SMs) | 24, 3,072 FP32 lanes | [AD107 specs](https://videocardz.net/nvidia-geforce-rtx-4060-laptop-gpu) |
| SM layout | 4 partitions, each with one warp scheduler that issues one warp instruction per clock, 16 FP32 lanes, 16 lanes that do FP32 or INT32, 4 load/store units, one special function unit, and a 64 KB register file | [Ada whitepaper, p. 9-10](https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf) |
| registers | 64K 32-bit per SM, at most 255 per thread | [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) |
| resident warps | at most 48 per SM, 24 blocks per SM | Ada tuning guide |
| L1 and shared memory | 128 KB per SM, shared memory up to 100 KB; CUDA can set the carveout per kernel (0, 8, 16, 32, 64 or 100 KB) | Ada tuning guide |
| L2 cache | 32 MB | AD107 specs |
| memory | 8 GB GDDR6, 128-bit, 256 GB/s | AD107 specs |
| FP16 without tensor cores | same rate as FP32 on Ada | Ada whitepaper, throughput table |
| INT32 | half the FP32 rate (only half the lanes do INT32) | Ada whitepaper |
| clocks and power | observed 2,490 MHz under load, 3,105 MHz maximum; power limit 100 W (default 55 W); PCIe 4.0 x8 | `nvidia-smi -q` on this machine |

Peak rates at the observed 2.49 GHz, derived from the table:

- Warp instruction issue: 24 SMs x 4 schedulers x 2.49 GHz = 239 billion warp instructions/s, or 7.65 trillion thread instructions/s when all 32 lanes are active.
- FP32 fused multiply-add: 24 x 128 x 2 x 2.49 GHz = 15.3 TFLOPS.
- INT32: half of that. Integer index arithmetic competes with FP32 on the shared lanes.
- Special functions (reciprocal, square root, sine, cosine): one unit per partition. They are usually documented at 16 results per clock per SM, a quarter of the FP32 rate. I could not confirm that figure for Ada in the current CUDA documentation, so it needs a microbenchmark (section 9).

Consequences for this workload:

- Memory bandwidth is not the limit. Measured DRAM use is 1 to 5% of peak. The whole working set of a running batch fits in the 32 MB L2.
- The limit is instruction issue and latency. The measured issue rate is 17 to 20% of peak, with 10 to 12 resident warps per SM out of 48.
- FP16 arithmetic would not be faster on Ada. It would only halve register and shared-memory footprint, and it would change results.
- Tensor cores do dense small matrix products. Nothing in the current physics has that shape. A reduced-coordinate solver has small dense matrices, but at 3x3 and below they are cheaper in plain FMAs.
- A laptop GPU shares its power and heat budget with the CPU through NVIDIA Dynamic Boost, which moves up to about 15 W between them ([PCWorld](https://www.pcworld.com/article/393737/up-close-with-nvidias-dynamic-boost-feature-for-gaming-laptops.html), [NVIDIA Linux README](https://download.nvidia.com/XFree86/Linux-x86_64/515.43.04/README/dynamicboost.html)). We measured this: with 8 CPU threads evaluating, the GPU's own rate fell from 174,000 to 137,000 standard trials/s. On this machine, CPU work is not free. It costs GPU clock.

### 2.2 CPU: Ryzen 7 7840HS (Zen 4)

| property | value | source |
|---|---|---|
| cores | 8 cores, 16 threads, up to 5.14 GHz | `lscpu` |
| caches | 32 KB L1d and 1 MB L2 per core, 16 MB L3 shared | `lscpu` |
| vector units | two 256-bit FMA units and four 256-bit ALUs per core; AVX-512 instructions are split into two 256-bit halves inside the pipe | [Chips and Cheese, Zen 4 part 1](https://chipsandcheese.com/p/amds-zen-4-part-1-frontend-and-execution-engine) |
| AVX-512 extensions | F, BW, DQ, VL, VNNI, BF16, VBMI2, BITALG, VPOPCNTDQ, IFMA | `/proc/cpuinfo` |

Peak FP32: two 256-bit FMA units give 16 FMAs, or 32 FLOPs, per clock per core. At an estimated 4 GHz all-core clock under vector load, 8 cores give about 1 TFLOPS. That is about 7% of the GPU's peak. AVX-512 on Zen 4 brings fewer instructions and masking, not more FLOPs per clock.

Measured: the CPU engine evaluates about 9,800 evolved creatures/s on 6 threads with 60 s trials. That works out to an estimated 700 to 800 core cycles per creature-step. A fully vectorized step at the GPU kernel's instruction count would need about 150 to 300 cycles, so the CPU engine likely runs at 20 to 40% of what this core can do. The code has per-lane scalar loops (contact bits, touchdown sensors, `to_array` conversions) that would explain it. This is an estimate. Section 9 lists how to profile it.

Even a perfect CPU engine would add at most about 7% to total throughput, and it would slow the GPU through the shared power budget. The CPU's job is orchestration, breeding, the archive, replays and the UI, not bulk evaluation.

### 2.3 Radeon 780M

The integrated GPU drives the display. It has 12 RDNA 3 compute units and shares system memory bandwidth with the CPU. Heavy compute on it crashed the desktop once. It stays out of evaluation.

## 3. The workload

### 3.1 Shape

- 3 million independent trials per generation. No communication between creatures during a trial. This is embarrassingly parallel across creatures.
- Each trial is strictly sequential in time: 100 settling steps, then up to 3,600 steps at 60 Hz.
- Each creature is tiny: typically 4 to 9 nodes, 3 to 8 bones, 5 to 19 muscles. Its whole state is a few hundred bytes.
- Trials end at very different times. About 38% of evolved creatures fall, most within a second. Screening stops 80% of the rest at 5 s. The survivors run 60 s.
- Estimate: under today's screening policy an average creature costs about 1,060 standard steps (100 + 0.8 x 300 + 0.2 x 3,600, before subtracting falls). Fine checks add about a quarter on top in standard-step equivalents, so about 1,330.

Two well-studied problem classes have the same shape:

- Batched small problems. Thousands of independent small computations, each mapped to one thread. The known issues are register pressure and occupancy.
- GPU path tracing. Millions of independent paths that terminate at unpredictable lengths. Its known issues are idle lanes after termination and divergence, and its literature has the fixes (section 5.3).

### 3.2 Measured efficiency of the current kernel

| measurement | value | where |
|---|---|---|
| resident warps per SM | 10 to 12 (p50), 17 (p90), of 48 | nsys GPU metrics, 3M GUI run |
| issue slots used | 17 to 20% | same |
| DRAM bandwidth | 1 to 5% | same |
| registers per thread | 128 to 149 | driver pipeline statistics (`examples/shader_stats.rs`) |
| shared memory | 6 KB per one-warp workgroup at 8 nodes | same |
| occupancy sensitivity | halving resident warps cut throughput to 0.66x | eval-bench, 2026-09-27 |
| time by phase (ablation, evolved bodies) | muscles 36% (waveform 14%, force scatter 8%), metrics 22%, joint limits 16%, bone passes 14%, velocity pass 8% | performance log |

Estimate of instructions per creature-step: 17 to 20% of 239 billion warp instructions/s is about 40 to 48 billion warp instructions/s. At the GUI's roughly 250 million creature-steps/s (200,000 creatures/s x about 1,330 step equivalents), and with 80 to 100% of lanes active, that is about 4,000 to 5,000 thread instructions per creature-step. For a body of 6 nodes, 5 bones and 10 muscles, that is roughly 700 to 800 instructions per node per step.

The kernel source shows where such counts come from. One step runs about 21 loops over nodes and 6 loops over bones, has 48 divisions and 15 square roots or vector lengths in its text, and recomputes several per-creature constants every step. Examples: each muscle's four endpoint weights come from branches on node identity in every step, although they are fixed for the creature's whole life. Node indices are unpacked from bit fields with integer arithmetic in every loop, which uses the half-rate INT32 lanes. The joint-limit pass and the joint-break check each rebuild the joint angle from positions with a square root.

### 3.3 Why tuning inside this design has run out

The 2026-09-27 campaign tried the natural micro-optimizations. Several helped (one fewer shared array, bit-exact, +7.5%; 1 s units, +11%). Several did not: grouping muscles, per-plan batches in mixed units, a waveform cache, a branch-free waveform, metrics moved to global memory, longer or shorter step ranges, extra segments. The pattern is consistent. The compiler caps the kernel near 128 registers and spills the rest. Moving data between registers, shared memory and global memory trades one latency for another. Occupancy cannot rise much without cutting live state, and live state cannot drop much without cutting the passes that need it.

## 4. The target as an instruction budget

| quantity | value |
|---|---|
| creatures/s wanted | 2,000,000 |
| step equivalents per creature (today's policy) | about 1,330 |
| creature-steps/s needed | about 2.7 billion |
| thread instructions/s available at 100% issue, all lanes active | 7.65 trillion |
| realistic issue efficiency x lane utilization | 0.5 x 0.85 to 0.6 x 0.9 |
| thread instructions/s usable | 3.3 to 4.1 trillion |
| budget per creature-step | about 1,200 to 1,600 instructions |
| today | about 4,000 to 5,000 instructions at about 0.2 issue efficiency |

The gap has two factors that multiply: about 3x in instruction count and about 3x in issue efficiency. The step count can move too. Removing the settling phase saves 100 of about 1,060 standard steps. More screening rungs (section 7.6) could cut the average further, but that is an owner decision about the search.

The host side has a budget too. At 2M/s a 3M generation lasts 1.5 s. Breeding and archiving currently cost about 2.6 s and 0.6 to 2.3 s per generation on the worker thread. They overlap with GPU work today, but at 1.5 s per generation they no longer fit. At the target they must run in parallel on all cores, or move to the GPU (section 7.5).

## 5. What the literature says, applied here

### 5.1 Roofline thinking

The roofline model bounds performance by peak compute and by memory bandwidth times arithmetic intensity. For kernels like ours, which are neither FLOP-bound nor bandwidth-bound, the instruction roofline of [Ding and Williams (2019)](https://escholarship.org/uc/item/7q73n52w) uses warp instructions per second as the ceiling and separates issue-bound from latency-bound code. Placed on that chart, our kernel sits at about 20% of the issue ceiling with negligible memory traffic. The work is latency-bound: dependent instruction chains and too few resident warps to cover them. That points at two levers, fewer instructions and more independent work in flight. More bandwidth or a faster FP unit would not help.

### 5.2 Occupancy versus instruction-level parallelism

[Volkov (GTC 2010)](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf) and his thesis [Understanding Latency Hiding on GPUs (2016)](https://escholarship.org/content/qt1wb7f3h4/qt1wb7f3h4_noSplash_1e32f64125997ee6afa303a150338054.pdf) showed that latency can be hidden either by many warps (occupancy) or by independent instructions within one thread (ILP). Code with good ILP runs fast at low occupancy. Our kernel has little ILP by construction. Gauss-Seidel projection updates bone after bone, each update depending on the previous one, and each step depends on the last. So it needs occupancy, and occupancy is capped by registers (128 to 149 per thread allows 13 to 16 warps) and shared memory (6 KB per warp allows about 16). The measured 0.66x slowdown at half the warps confirms it is on the steep part of the curve.

Two ways out:

- Reduce live state per creature, so more creatures fit. The reduced-coordinate formulation (section 7.2) has about half the state variables of the current one and no scratch arrays.
- Add ILP explicitly: two creatures per thread for the smallest bodies, or updates that are independent within a step (Jacobi-style or graph-coloured passes instead of strict Gauss-Seidel).

CUDA exposes `__launch_bounds__` and a per-kernel shared-memory carveout, so these trade-offs can be steered and measured. Vulkan leaves them to the driver. The CUDA best-practice advice is to measure rather than to force occupancy, because spills can cost more than they gain ([NVIDIA forums on launch bounds and spills](https://forums.developer.nvidia.com/t/effect-of-launch-bounds-on-register-usage-and-spillage/303874)).

### 5.3 Variable-length work and divergence

The GPU ray tracing literature solved the "work items end at different times" problem more than a decade ago:

- [Aila and Laine (2009)](https://research.nvidia.com/sites/default/files/publications/aila2009hpg_paper.pdf) introduced persistent threads. The kernel launches just enough warps to fill the GPU, and each warp loops and fetches new work from a global pool instead of exiting.
- [Novák, Havran and Dachsbacher (2010)](https://jannovak.info/publications/PathRegPT/PathRegPT.pdf), "Path regeneration": when a path terminates, its thread starts a new path at once instead of idling until the rest of the warp finishes.
- [Laine, Karras and Aila (2013)](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf), "Megakernels considered harmful": keep a large pool of work alive and run each stage as a separate coherent kernel, because one huge kernel with many branches wastes registers and diverges.
- [Madrona (Shacklett et al., 2023)](https://madrona-engine.github.io/shacklett_siggraph23.pdf), a batch simulator for reinforcement learning, reports over 1.9 million environment steps/s on one GPU. It uses a persistent-threads megakernel in which warps fetch work at warp granularity. It radix-sorts its entity tables by environment each step so that neighbouring threads touch neighbouring data, and it lets some systems run a whole warp cooperatively on one entity.

Our segments (pauses at 2, 5 and 10 s with a host-side repack) are a coarse, host-driven form of the same idea. The principled form keeps creatures on the GPU and refills lanes from a device-side queue with no host round trip.

The megakernel warning also applies. Our kernel handles settling, trials, screening, falls, fine checks, all terrain effects and all behavior metrics in one body. Rarely used paths still reserve registers for the whole kernel. Specialized variants by fidelity and by effect set (flat ground versus rough, mud and gaps) would shrink the common case.

### 5.4 Work efficiency comes before parallel efficiency

The most cost-effective optimization in HPC is usually a cheaper algorithm, not a faster implementation of the same one. For articulated bodies:

- Featherstone's articulated-body algorithm computes forward dynamics of a kinematic tree in O(n) time in reduced (joint) coordinates ([Featherstone 1999](https://journals.sagepub.com/doi/abs/10.1177/02783649922066619); Featherstone, *Rigid Body Dynamics Algorithms*, 2008). Batched GPU implementations exist ([GRiD](https://arxiv.org/pdf/2109.06976); [BARD](https://arxiv.org/pdf/2605.31481), which reports up to 64x higher throughput than baselines at batch 4,096 on an H200). In reduced coordinates, bone lengths cannot drift, so no projection, rebuild or length correction is needed, and joint limits become bounds on a joint angle instead of angle reconstructions from positions.
- [Macklin et al. (2019), "Small Steps in Physics Simulation"](https://mmacklin.com/smallsteps.pdf) found that n substeps with one constraint iteration each beat one step with n iterations, both in accuracy and in stability. Today we run two bone passes per 60 Hz step, and fine checks run four times the passes at four times the rate.
- The 30 Hz experiment of 2026-09-27 showed the cost of an integrator that is only approximately right. Elites evolved at 30 Hz kept a median 38% of their distance at 60 Hz. The fine check at 240 Hz exists partly to catch the same kind of artifact at 60 Hz, and it costs about 20% of GPU time. An integrator whose error is smaller at a given step would make the standard trial more trustworthy and the check cheaper or rarer.

### 5.5 Keep the whole loop on the device

Isaac Gym ([Makoviychuk et al. 2021](https://arxiv.org/pdf/2108.10470)) keeps physics, observations and learning on the GPU and reports 2 to 3 orders of magnitude over CPU-simulator pipelines. Brax ([Freeman et al. 2021](https://arxiv.org/abs/2106.13281)) compiles environment and learner together on the accelerator. For quality-diversity specifically, QDax ([Lim, Allard, Grillotti and Cully, 2022](https://arxiv.org/pdf/2202.01258)) runs MAP-Elites on the accelerator. It found no statistically significant loss of final QD score from very large batches at an equal number of evaluations. Its throughput plateaued at about 30,000 evaluations/s on an A100 at batch 65,536, with 100-step 3D episodes.

We already run 3M-creature batches, so the batch-size lesson is in place. The residency lesson is not. Every creature is bred on the CPU, packed on the CPU, uploaded, simulated, read back, and repacked on the CPU at every segment boundary. Since 2026-09-27 the upload copies are freed after submission, but packing still costs 2 to 8 s per 3-generation run, and the host holds about 10 GB at peak.

### 5.6 Spend less simulation on losers

Screening at 5 s is successive halving with one rung. The hyperparameter-search literature generalizes it: successive halving ([Jamieson and Talwalkar 2016](https://arxiv.org/abs/1502.07943)), Hyperband ([Li et al. 2018](https://jmlr.org/papers/v18/16-558.html)) and ASHA, the asynchronous version built for massive parallelism ([Li et al. 2020](https://arxiv.org/abs/1810.05934)). Our data on 2026-09-27: a second rung at 10 s keeping half the survivors kept every creature of the final top 1% and 86% of the top 10%, for about 1.3x. This is a search-policy decision for the owner, not an engineering one.

### 5.7 Determinism across devices

SPIR-V lets drivers fuse a multiply and an add into one FMA unless the NoContraction decoration is set, and NVIDIA does fuse them ([SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html); [Slang issue on `precise` and NoContraction](https://github.com/shader-slang/slang/issues/12198)). That is one reason our GPU and CPU engines differ in the last bits. Those bits diverge chaotically over a trial, and that forces the CPU replay check before global-archive admission. A single physics source compiled for both targets with contraction disabled and identical operation order would make replays bit-exact. The replay verification step, and the separate CPU reference implementation in `src/physics.rs`, could then go.

### 5.8 General engines are not the answer

| system | reported throughput | hardware | note |
|---|---|---|---|
| Isaac Gym | 540,000 environment steps/s (Ant, 4,096 envs) | A100 | [paper](https://arxiv.org/pdf/2108.10470) |
| MuJoCo MJX | 950,000 humanoid steps/s (batch 8,192); 2.7M on an 8-chip TPU v5 | A100, TPU | [MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html) |
| Madrona | over 1.9M environment steps/s (hide and seek) | one GPU | [paper](https://madrona-engine.github.io/shacklett_siggraph23.pdf) |
| QDax MAP-Elites | about 30,000 evaluations/s, 100-step episodes | A100 | [paper](https://arxiv.org/pdf/2202.01258) |
| this game, today | about 250 million creature-steps/s | RTX 4060 Laptop | estimate from measured creatures/s (section 3.2) |

These engines solve general 3D contact problems, so their per-step cost is orders of magnitude higher than a 2D chain of six point masses. Our specialized kernel is already far faster per step than any of them on a much smaller GPU. Adopting one would make us slower. The lessons to borrow are architectural: residency, persistent workers, sorting, batching.

## 6. Diagnosis of the current architecture

What is sound and should stay:

- One creature per GPU thread, with bodies packed per 32-creature tile in structure-of-arrays layout. That is the right mapping for tiny independent problems.
- Raw Vulkan with several queues, instead of wgpu's conservative barriers.
- Screening, segments and fall freezing. They are the right ideas, applied at host level.
- A CPU fallback engine, and deterministic seeds per creature.

What limits it, from most to least expensive:

1. The formulation. Maximal coordinates (x, y per node) with iterative length projection, a joint-limit pass, a parent-first rebuild, a whole-body lift, a planted-feet friction cap, a speed cap with momentum redistribution, and a settling phase. Many of these exist to repair what an earlier pass broke. Together they account for most of the roughly 4,000 to 5,000 instructions per step.
2. Recomputing constants every step: muscle endpoint weights, packed node indices, joint break thresholds (the last one was fixed on 2026-09-27).
3. Execution granularity. Each dispatch runs 64 steps and then reloads and stores the creature state. Lanes of fallen or screened creatures idle until the next segment boundary. Compaction needs a GPU-to-host readback and a host repack.
4. No control over registers or shared-memory carveout, and no instruction-level profiler in our current toolchain (WGSL through naga to SPIR-V).
5. Three physics implementations (GPU kernel, AVX-512 CPU engine, older CPU reference) that differ in the last bits. That costs a CPU replay for every global-archive contender, triple maintenance for every physics change, and uncertainty in every agreement test.
6. A CPU-resident population. Breeding, packing and archive insertion run on the CPU, with about 10 GB peak RAM at 3M. At the target rate they would be on the critical path.

## 7. Recommendations

Ranked by expected gain times confidence. Each has a validation gate.

### 7.1 Measure properly first (1 to 2 days, no risk)

See section 9. The single most important number, instructions per creature-step by phase, is an estimate today. Nsight Graphics' GPU Trace shader profiler supports Vulkan compute on Ampere and Ada, on Linux, with per-instruction stall reasons ([Shader Profiler docs](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)). It works with the current kernel as it is. This gate decides how much each of the following is worth.

### 7.2 Physics v2: a planar articulated tree in reduced coordinates

Proposal: each creature's state is the root node's position and velocity plus one absolute angle and angular velocity per bone. Node positions follow from forward kinematics. Dynamics come from an O(n) recursive algorithm (Featherstone's articulated-body algorithm specialized to 2D point masses on massless or massive rods). Muscles apply point forces that map to generalized forces through the tree. The ground uses compliant contact at the nodes, with Coulomb friction clamped per contact. Joint limits are angle bounds with a stiff compliant response.

What it removes: both bone projection passes, the joint-limit angle reconstruction, the parent-first rebuild, the turn limit, the whole-body lift, the length-drift problem, and the settling phase (a pose is valid by construction, so it needs no relaxing). The joint-break check becomes a comparison on a joint angle.

Estimated cost per step for 6 nodes and 10 muscles: forward kinematics about 100 instructions, articulated-body passes about 400 to 600, muscles about 300, contacts and friction about 150 to 250, integration and metrics about 150. Total about 1,100 to 1,400 instructions, against about 4,000 to 5,000 today. That is roughly 3 to 4x fewer instructions per step, about 9% fewer steps without settling, and about half the live state.

Risks and open questions:

- It is new physics. Gaits evolved so far would not transfer, and `qd::VERSION` must change. The owner has said the old gameplay is not a reference, but this needs an explicit yes.
- Contact is the hard part. Compliant contact needs small enough steps to stay stable against a stiff ground. Stiffness and damping must be tuned so that feet do not sink or bounce, and the "only planted feet push" rule must be re-established from first principles. The free-propulsion test (`examples/first_generation.rs`) and the momentum ledger must pass before anything lands.
- A fallback within the current physics exists and is worth prototyping in parallel: the same formulation with every constant precomputed, node-count-specialized kernels that keep node state in registers, and fewer and fused passes. Most of that can stay bit-exact. Estimated 1.5 to 2.5x fewer instructions, with no change for the player.

Gate: a CPU prototype that reproduces walking, falling and gait variety on the first-generation population, passes the free-propulsion and energy tests, and measures under 1,500 instructions per creature-step on the GPU.

### 7.3 Persistent lanes with regeneration

Proposal: launch one resident grid per body-size class that stays alive for many seconds. Each lane loads a creature from a device-side queue, keeps its entire state in registers across all its steps, and writes its result to a device-side ring when the creature falls, is screened or finishes. Then it takes the next creature from the queue at once. The host refills the queue and drains results asynchronously, with no per-segment readback and no repack. Queues are sorted by body plan so that the 32 lanes of a warp run the same loop counts. Madrona does this sort with a radix sort each step.

What it removes: idle lanes after falls and screens (today 25 to 34% of simulated time comes after a fall, recovered only at segment boundaries), the load and store of all state every 64 steps, host repacks, and most packing.

Expected: 1.3 to 2x on its own. It works with today's physics too, so it can land first.

Risks: warps then hold creatures at different ticks. Tick-dependent branches (settling, screen tick, gait sampling) diverge. That is small if those branches are cheap, and path tracers live with far worse. Long-running kernels need care with the display watchdog. The RTX does not drive the display here, so the risk is lower than usual.

Gate: an eval-bench with the same population and the same results bit for bit (fall, screen and final distance), at a higher creature rate.

### 7.4 A toolchain that gives us control

Options, in order of preference:

1. [Slang](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/09-targets.html), one source compiled to SPIR-V for Vulkan, to CUDA, and to scalar C++ for the CPU. Nsight Graphics profiles Slang shaders. The CUDA target gives access to Nsight Compute, launch bounds and carveout control. The C++ target gives a CPU reference from the same source (section 5.7). Cost: a new build dependency and porting about 1,150 lines of WGSL.
2. CUDA C++ through NVRTC from Rust (for example with the `cudarc` crate), with the CPU engine kept in Rust. Best profiling and control. It ties the fast path to NVIDIA, with the CPU engine as the fallback that exists anyway.
3. Stay on WGSL and naga and use Nsight Graphics for profiling. The cheapest option, but occupancy stays in the driver's hands and CPU-GPU bit-exactness stays out of reach.

Expected: 1.2 to 1.8x from register and occupancy control alone, plus much shorter measure-change-measure cycles.

### 7.5 Move the loop onto the GPU

Proposal: genomes live in GPU memory in structure-of-arrays form. Parametric mutation (the CMA, gaussian and crossover emitters) runs on the GPU. Archive insertion is a parallel per-cell reduction, which QDax does. Results never leave the GPU except for the UI's snapshot and the elites the player looks at. Structural mutations (adding or removing nodes, bones, muscles) can stay on the CPU at first. They change body sizes and are a minority of offspring.

What it removes: packing (2 to 8 s per 3-generation run), per-unit PCIe traffic, the CPU replay if 7.4 makes engines bit-exact, most of the 10 GB of host memory, and breeding as the next wall after the GPU.

Expected: it does not raise the kernel's rate. It stops the host from becoming the bottleneck once the kernel is 5 to 10x faster, and it frees CPU power for the GPU's budget.

### 7.6 Search-side levers (owner decisions)

These change what is simulated, not how fast:

- More screening rungs (ASHA-style): 1.3x for a 10 s rung keeping half, 1.5x keeping 30% (measured selection quality above).
- Fine checks at 2x instead of 4x the standard rate: checks cost about 20% of GPU time. If physics v2 has a smaller integration error, a 2x check may catch as much.
- Behavior metrics at the gait sampling rate (30 Hz) instead of every step, keeping fall and touchdown detection per step. Metrics are 22% of kernel time today.

### 7.7 What I recommend against

- 30 Hz physics, for the reason measured on 2026-09-27: evolution exploits the coarse step.
- FP16 arithmetic: no faster on Ada, and it changes results.
- Tensor cores: nothing in the physics is a dense product large enough to use them.
- Adopting a general engine (Isaac, MuJoCo, Brax, Genesis): slower per step for our problem (section 5.8).
- More CPU evaluation: at most about 7% more compute, and it takes power from the GPU.
- The Radeon 780M for compute: it crashed the desktop once, and it shares memory bandwidth with the CPU.

## 8. Expected combined effect

The factors are estimates and they multiply only if each holds. Conservative uses the low end of each range, optimistic the high end.

| lever | conservative | optimistic | needs owner decision |
|---|---:|---:|---|
| physics v2, instructions per step (7.2) | 2.5x | 4x | yes |
| no settling phase (7.2) | 1.08x | 1.09x | with 7.2 |
| persistent lanes and regeneration (7.3) | 1.3x | 2x | no |
| register and occupancy control (7.4) | 1.2x | 1.8x | no |
| **kernel total** | **about 5x** | **about 14x** | |
| more screening rungs (7.6) | 1x | 1.3x | yes |
| **end to end** | **about 1M creatures/s** | **about 3.5M creatures/s** | |

The fallback without new physics (bit-exact restructuring plus 7.3 and 7.4) is about 1.5 to 2.5 times 1.3 to 2 times 1.2 to 1.8. That is 2.3x to 9x in theory. Given how much has already been taken, I would expect it to land near the low end, so 300,000 to 450,000/s.

Hardware scales almost linearly for this workload, because it is compute-bound, embarrassingly parallel and fits in cache. An RTX 4090 has 128 SMs and an RTX 5090 has 170 ([RTX 5090 specs](https://videocardz.net/nvidia-geforce-rtx-5090)), against 24 here, at similar clocks and without a laptop power cap. That is roughly 5x and 7x on top of any software gain. Two million creatures/s on a desktop GPU needs only the fallback path. On this laptop it needs the new physics.

## 9. Profiling plan

The goal is to replace every estimate above with a measurement before committing to a design.

GPU:

1. Ceilings. Microbenchmarks on this exact GPU: FP32 FMA issue, INT32 issue, special-function throughput, shared-memory load latency and throughput, and one dependent FMA chain. Record the SM clock with `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv -lms 100` during every run. The laptop throttles on power, and the driver already reports software power-cap events.
2. Instruction profile of the current kernel. Nsight Graphics GPU Trace with the real-time shader profiler, on eval-bench with a fixed checkpoint. Record executed instructions per creature-step by source region, stall reasons (long scoreboard, short scoreboard, MIO throttle, wait, math pipe throttle, not selected), and register and spill counts from SASS. This yields the instruction roofline position and confirms or refutes the 4,000 to 5,000 estimate.
3. Timeline. Nsight Systems as used now for GPU metrics sampling. Add Vulkan tracing once it runs (it crashed on the first attempt and needs investigation), and host ranges around packing, breeding and archive work.
4. If the kernel moves to CUDA: Nsight Compute with the full section set, source-level counters and its built-in roofline charts.

CPU:

1. `perf stat` with the Zen 4 pipeline-utilization metrics (frontend, backend, bad speculation, retiring), which are in the kernel's AMD event tables ([LKML patch adding Zen 4 metrics](https://lkml.iu.edu/hypermail/linux/kernel/2212.1/04303.html)). `perf_event_paranoid` is 1 here, so user-space counting works without root.
2. `perf record` with AMD instruction-based sampling (IBS) for exact attribution of retired instructions and cache misses, or [AMD uProf](https://docs.amd.com/r/en-US/68658-uProf-getting-started-guide/Introduction-to-IBS-Instruction-Based-Sampling), for the CPU engine, breeding and archive insertion.

Method:

- Keep the two benchmark levels separate. eval-bench on fixed checkpoints measures the kernel. The GUI benchmark measures the system.
- Interleave A and B runs, repeat each at least twice (three times for small effects), and report medians and spread. Single GUI runs vary by about 10%.
- Warm up to a steady clock and temperature before measuring. Pin GPU clocks (`nvidia-smi -lgc`, which needs root) if the owner allows it.
- Every physics change keeps the existing gates: engine agreement tests, the free-propulsion check, and search A/B over 10 seeds.

## 10. Roadmap with decision gates

| phase | work | length | gate to continue |
|---|---|---|---|
| 0 | profiling plan above; microbenchmarks; instruction counts by phase | 1 to 2 days | measured instructions per step and stall profile |
| 1 | CPU prototype of physics v2 (scalar Rust, then SIMD); behavior and exploit tests; cost count | 1 to 2 weeks | owner approves the feel of the new physics; under 1,500 instructions per step projected |
| 2 | persistent-lane kernel with regeneration, first on today's physics | 1 week | bit-exact results, higher rate on eval-bench |
| 3 | toolchain decision (Slang or CUDA) and port of the kernel | 1 week | profiler access; equal or better rate |
| 4 | physics v2 kernel on the persistent-lane framework; engine agreement; search A/B | 2 to 3 weeks | agreement tests and search quality at equal time |
| 5 | GPU-resident genomes, breeding and archive | 2 to 3 weeks | host no longer on the critical path |

Phases 0, 2 and 3 need no owner decision and are useful whether or not physics v2 happens. Phase 1 is where the owner decides.

## 11. Decisions for the owner

1. Is a new physics formulation acceptable (section 7.2)? It is the only path to 2M/s on this laptop.
2. Is an NVIDIA-specific fast path acceptable (CUDA or Slang to CUDA), with the CPU engine as the portable fallback (section 7.4)?
3. Search-side levers: a second screening rung, cheaper fine checks, lower-rate behavior metrics (section 7.6).
4. May benchmarks pin GPU clocks with root (`nvidia-smi -lgc`) for cleaner measurements?

## References

- NVIDIA, [Ada Lovelace GPU architecture whitepaper v1.1](https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf).
- NVIDIA, [Ada GPU architecture tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).
- NVIDIA, [Nsight Graphics shader profiler](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html).
- [RTX 4060 Laptop GPU specifications](https://videocardz.net/nvidia-geforce-rtx-4060-laptop-gpu); [RTX 5090 specifications](https://videocardz.net/nvidia-geforce-rtx-5090).
- Chips and Cheese, [AMD's Zen 4 part 1: frontend and execution engine](https://chipsandcheese.com/p/amds-zen-4-part-1-frontend-and-execution-engine).
- AMD, [uProf and instruction-based sampling](https://docs.amd.com/r/en-US/68658-uProf-getting-started-guide/Introduction-to-IBS-Instruction-Based-Sampling); [Zen 4 perf metrics patch](https://lkml.iu.edu/hypermail/linux/kernel/2212.1/04303.html).
- NVIDIA Dynamic Boost: [PCWorld](https://www.pcworld.com/article/393737/up-close-with-nvidias-dynamic-boost-feature-for-gaming-laptops.html), [Linux driver README](https://download.nvidia.com/XFree86/Linux-x86_64/515.43.04/README/dynamicboost.html).
- N. Ding and S. Williams, [An Instruction Roofline Model for GPUs](https://escholarship.org/uc/item/7q73n52w), PMBS 2019.
- V. Volkov, [Better Performance at Lower Occupancy](https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf), GTC 2010; [Understanding Latency Hiding on GPUs](https://escholarship.org/content/qt1wb7f3h4/qt1wb7f3h4_noSplash_1e32f64125997ee6afa303a150338054.pdf), PhD thesis, 2016.
- T. Aila and S. Laine, [Understanding the Efficiency of Ray Traversal on GPUs](https://research.nvidia.com/sites/default/files/publications/aila2009hpg_paper.pdf), HPG 2009.
- J. Novák, V. Havran and C. Dachsbacher, [Path Regeneration for Interactive Path Tracing](https://jannovak.info/publications/PathRegPT/PathRegPT.pdf), Eurographics 2010.
- S. Laine, T. Karras and T. Aila, [Megakernels Considered Harmful: Wavefront Path Tracing on GPUs](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf), HPG 2013.
- B. Shacklett et al., [An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation](https://madrona-engine.github.io/shacklett_siggraph23.pdf), SIGGRAPH 2023.
- R. Featherstone, [A Divide-and-Conquer Articulated-Body Algorithm](https://journals.sagepub.com/doi/abs/10.1177/02783649922066619), IJRR 1999; *Rigid Body Dynamics Algorithms*, Springer 2008.
- B. Plancher et al., [GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients](https://arxiv.org/pdf/2109.06976), ICRA 2022; [BARD: Batched Differentiable Rigid Body Dynamics in PyTorch](https://arxiv.org/pdf/2605.31481).
- M. Macklin et al., [Small Steps in Physics Simulation](https://mmacklin.com/smallsteps.pdf), SCA 2019.
- V. Makoviychuk et al., [Isaac Gym](https://arxiv.org/pdf/2108.10470), NeurIPS 2021.
- C. D. Freeman et al., [Brax](https://arxiv.org/abs/2106.13281), NeurIPS 2021.
- Google DeepMind, [MuJoCo MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html).
- B. Lim, M. Allard, L. Grillotti and A. Cully, [Accelerated Quality-Diversity through Massive Parallelism](https://arxiv.org/pdf/2202.01258), TMLR 2023.
- K. Jamieson and A. Talwalkar, [Non-stochastic Best Arm Identification and Hyperparameter Optimization](https://arxiv.org/abs/1502.07943), AISTATS 2016; L. Li et al., [Hyperband](https://jmlr.org/papers/v18/16-558.html), JMLR 2018; L. Li et al., [A System for Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934), MLSys 2020.
- Khronos, [SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html) (NoContraction); Slang, [issue 12198 on `precise`](https://github.com/shader-slang/slang/issues/12198); Slang, [compilation targets](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/09-targets.html).
- K. Sims, [Evolving Virtual Creatures](https://dl.acm.org/doi/10.1145/192161.192167), SIGGRAPH 1994.
