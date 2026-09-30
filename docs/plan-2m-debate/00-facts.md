# Shared facts for the 2M/s architecture debate

Date: 2026-09-30. Repo: /home/amipo/workspace/evolutionSimulator (read it; do not edit it, do not build in its target/). Owner's rules: AGENTS.md there. The older HPC assessment (2026-09-28, some of it now done) is beside this file as hpc.md.

## The game

2D creatures (nodes with mass, massless bones, pull-only muscles with a waveform, Hill relation, energy store, tendon, touchdown sensors, per-limb clocks) evolve to travel far in 20 s trials at 60 Hz. Fitness is distance only. MAP-Elites with 4 emitters (CMA, structural with 64 operators, novelty, immigrants), 4 isolated islands plus a hub, nurseries, a global archive for display. 3M creatures per generation. Environment effects are buttons (14 effects); autochange adds one every 100 generations. Screening: a trial stops at 5 s if below the bar (5 s distance the top 20% reached). The search must be deterministic per seed on one GPU. Saves are small (archives and search state). The owner wants: the game like a flood of data through CPU, GPU, and any other accelerator, at 2M evaluated creatures per second SUSTAINED for minutes and across generations, in the graphical game at 60 FPS. Spirit to preserve: natural selection of non-glitchy interesting movers.

## Hardware (one laptop)

- RTX 4060 Laptop (AD107, Ada): 24 SMs, 3072 FP32 lanes, 64 KB registers per SM, 100 KB shared max, 32 MB L2, 8 GB GDDR6 at 256 GB/s, observed 2.49 GHz, 100 W cap shared with the CPU through Dynamic Boost (CPU load lowers GPU clocks: 8 busy CPU threads once cut the GPU rate 174k to 137k trials/s).
- Ryzen 7 7840HS (Zen 4): 8 cores 16 threads, AVX-512 (two 256-bit halves), 16 MB L3, about 1 TFLOPS FP32.
- Radeon 780M iGPU (RDNA3, 12 CUs), drives the desktop, shares system memory bandwidth. Heavy compute on it once crashed the desktop. The owner NOW allows considering it ("both gpus!"), with that risk stated.
- 30 GB RAM. Linux 7.0. CUDA driver plus NVRTC; nsys works, ncu is not installed; perf works.

## Current architecture (main, 2026-09-30 evening)

- Physics: one CUDA kernel, shaders/warp_creature.cu, built by src/warp_kernel.rs, driven by src/cuda_engine.rs. One creature per lane group of 8, 16 or 32 lanes (lane i owns node i and the bone ending there), state in registers for the whole trial, tree passes level by level (depth about 5), waves of 262,144 creatures per launch with in-kernel regeneration from an atomic counter. Each 1/60 s step is 2 substeps, each one articulated-body pass (Featherstone, reduced coordinates) and one cold 4-sweep projected Gauss-Seidel contact solve on at most 4 contacts (exact contact matrix built by walkers climbing to the root), plus a friction-clean sweep. Invariants: momentum changes only by external impulses; friction never does positive work; no energy gain in flight. Kernels compiled per world with NVRTC (effects that are off leave no code). 128 registers per lane, 24 to 32 B spills, 19 to 21 KB shared per 128-thread block, 16 warps per SM. CPU and Vulkan engines are being deleted now (GPU is the only authority).
- Search loop: src/ring.rs. A ring of 4 blocks, 786k creatures in flight. Blocks are absorbed in ring order (determinism), each result offered to its island archive, slot bred again on the CPU (rayon). No fine checks; a new island record gets one confirmation trial at 4x rate, score = min. Breeding is about 4.7 s per 3M generation on the CPU (plan, emit with operators, repair). RAM about 5 to 9 GB at 3M per generation.
- Data movement: genomes bred on the CPU into the ring, packed (0.32 s per 262k on 4 threads) into per-creature records, uploaded, results (80 B) read back per wave. Replays are a one-creature recording wave.

## Measurements

- Kernel alone, exclusive GPU, 262,144 creatures of a generation-10 save: old per-thread kernel 26.7M creature-steps/s; new lane-group kernel 43 to 45M (1.65x); with 1 substep + 1 planting round 57 to 59M (2.2x) but planted feet slide more (elite median ratio 0.79 vs 1.03). Profile of the new kernel by section: contact solve 45 to 50% (Gauss-Seidel 16 to 20, matrix walk 13 to 15, response 10 to 12, detection 5), muscles 15 to 18, articulated-body pass 10 to 14. Every tree pass pays the full per-level cost for all lanes at every level.
- Older nsys metrics on the old kernel: SM issue 19%, DRAM 1%, PCIe RX 10%. Latency bound, not bandwidth bound.
- Elites: median 60 Hz distance 34.5 m for the top 300; a 2x rate retest gives ratio 0.99 median. Random bodies gain no distance (median -0.05 m).
- End to end, the new main, game alone, 3M per generation: 92k, 109k, 167k, 144k creatures/s in generations 0 to 3 (mean nodes 5.8 to 6.9). The old build fell from about 110k/s to 27 to 30k/s by generation 30 as bodies grew to 9 nodes and 30 to 50 muscles each; watch for the same.
- Average steps per creature with screening: roughly 500 to 600 (falls end trials early, 80% stop at 5 s = 300 steps, 20% run 1200 steps).
- 2M creatures/s therefore needs about 1.0 to 1.2 G creature-steps/s, about 25x the current kernel, or fewer steps per creature.
- Substep study (old kernel): one solve per step without an end-pose correction makes elites collapse (217 of 300 fall in 0.5 s); 2 substeps hold them and halve foot slip; sweeps 4 vs 8 change nothing.
- Fine checks (now gone) were 40 to 45% of GPU time. The 3M resident population was 23 GB (now a ring).
- Bodies bloat: elites reached 80 to 87 muscles on 13 nodes before muscle mass; muscle mass (0.05 kg + 1 kg/m of slack length) is in.

## What has been rejected with numbers (docs/rejected-ideas.md)

30 Hz physics (integrator exploits), five muscle-mass variants (now one is in), the first persistent-lane engine (register pressure; now solved differently), waveform cache and branch-free waveform on the GPU (slower), a second screening rung and cheaper checks (removed at the time; the owner may accept a second rung now if measured), FP16 (no faster on Ada), tensor cores (no dense products), general engines (Isaac, MuJoCo, Brax: slower per step for this problem).

## Open questions the debate must answer

1. Where do 25x come from: fewer instructions per step, more steps per second per SM, fewer steps per creature, or other silicon (CPU AVX-512, the iGPU), and in what mix, with numbers per lever.
2. What is the physically possible ceiling on this laptop under the shared 100 W budget, and is 2M/s sustained reachable here at all, or only with search-side changes (screening rungs, surrogate models, adaptive trial length) that keep the spirit.
3. What must the data flow look like (rings, GPU-resident genomes, breeding on the GPU or the iGPU, zero-copy, PCIe, memory budget) to sustain the rate for minutes across generations, with determinism per seed.
4. What each implementation track is, with a measurable gate, in an order that keeps the game playable at every merge.
