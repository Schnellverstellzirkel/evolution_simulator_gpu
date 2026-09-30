# Round 1, iGPU domain: the Radeon 780M

Author: the iGPU expert. Date 2026-09-30. Everything marked "measured" I read from this machine or the repository today. Everything else is an estimate with the measurement that would settle it.

## 1. What the 780M is on this laptop (measured)

- RDNA3 Phoenix, 12 CUs = 6 WGPs = 24 SIMD32 units, 768 lanes. DPM clock levels 800, 1100 and 2700 MHz (`pp_dpm_sclk`). Memory clock 800 MHz on a 128-bit LPDDR5X bus (kernel log: "RAM width 128bits LPDDR5"), so about 100 GB/s shared with the CPU.
- Driver: Mesa 25.2.8 RADV, Vulkan 1.4. Subgroup size 64 by default, 32 or 64 selectable with `VK_EXT_subgroup_size_control`. Subgroup shuffle, ballot, arithmetic and clustered ops are all there. 64 KB shared memory per workgroup. `shaderInt64`, `shaderFloat16`, `VK_KHR_shader_clock`, `VK_EXT_device_fault`.
- Queues: one graphics+compute queue (the compositor's), 4 compute-only queues, and global priorities LOW to REALTIME on all of them. LOW needs no privilege. HIGH and REALTIME need CAP_SYS_NICE on amdgpu.
- Memory: the VRAM carve-out is 512 MB and the desktop uses 434 MB of it right now. GTT (system memory the GPU maps) is 16 GB. `VK_EXT_external_memory_host` is supported with 4 KB alignment, so a page-aligned host allocation can be imported as a Vulkan buffer with no copy.
- No ROCm, no HIP, no OpenCL on the Radeon (only the NVIDIA ICD). `/dev/kfd` exists but the userspace is not installed and gfx1103 is not on ROCm's support list.
- Peak issue rate: 24 SIMD32 x 32 lanes x 2.7 GHz = 2.1 T thread instructions/s, against 7.65 T on the RTX 4060. Per SIMD per clock a wave32 instruction issues like an Ada partition's warp instruction, so the honest per-clock ratio is 24 : 96 = 0.25 of the RTX. Under a shared power cap the sustained clock is nearer 2.0 GHz, so 0.2 of the RTX.
- The old per-thread Vulkan engine put the Radeon at 40,000 creatures/s against 180,000 for the RTX in `src/scheduler.rs` (`Device::new(..., 40_000.0)`). That 22% agrees with the arithmetic above.
- Package power (`hwmon` PPT) reads 17 W with the desktop idle. The 7840HS package budget on this laptop is about 54 W and the iGPU draws from it. Dynamic Boost then moves up to 15 W between the package and the RTX. The measured 174k to 137k drop with 8 busy CPU threads is the same mechanism.

## 2. What crashed the desktop (measured, from the kernel journal)

Two events, both from the game's process:

- 2026-09-25 17:55:29: `ring comp_1.1.1 timeout, signaled seq=55, emitted seq=56`, process `evolution-simul` pid 12241. Per-queue reset succeeded. The desktop survived.
- 2026-09-26 13:06:36: the same ring, pid 842696, `signaled seq=12980, emitted seq=12982`. The queue reset failed, amdgpu did a MODE2 whole-GPU reset, "VRAM is lost due to GPU reset", "device wedged, but recovered through reset". The compositor's buffers were in that VRAM. That is the desktop crash.

The cause is a single compute submission that ran longer than the amdgpu lockup timeout. On this kernel (7.0) `modinfo amdgpu` says the default is 2000 ms for every ring, and `lockup_timeout` is unset. So any dispatch longer than 2 s is a coin flip between a queue reset and a desktop reset. The old engine sent trial segments of 16 steps for units of thousands of creatures, and one of them crossed 2 s.

The rules that follow are not optional:

1. Every submission to the Radeon finishes in under 50 ms by design, measured with timestamp queries, and the engine halves its dispatch size whenever a dispatch exceeds 20 ms. Nothing persistent, no in-kernel work loop, no atomic-counter regeneration. A wave's step count is a kernel argument, never unbounded.
2. If a fence wait exceeds 500 ms the engine stops submitting for the rest of the session, drains, and the work goes back to its CPU twin. The scheduler already retires a failed GPU; the retirement threshold for the Radeon is latency, not an error code.
3. Only GTT and imported host memory. Never `DEVICE_LOCAL` on the Radeon. The carve-out is 85% full with the desktop's own buffers, and evicting them is how a reset becomes a black screen. Cap the Radeon engine at 512 MB of GTT.
4. Our queue is a compute-only queue at `QUEUE_GLOBAL_PRIORITY_LOW`. The compositor keeps the graphics queue at its default priority and the hardware scheduler favours it.
5. Run the Radeon work in a helper process that shares memory with the game (memfd plus the same pinned ring). A `VK_ERROR_DEVICE_LOST` or a driver hang then kills the helper, not the game, and the game continues on the CPU twins. The GPU rules above protect the desktop; the process boundary protects the run.
6. Vulkan on RADV only. ROCm would need an unsupported GPU override and user-mode queues whose hang recovery is worse. There is no reason to install a second driver stack on the GPU that draws the screen.

Acceptance test for any Radeon engine: a 10 minute soak with the game's UI at 60 FPS on the Radeon, the engine 80% busy, UI frame time p99 at or under 20 ms (the game's benchmark already prints frame percentiles), and no `ring comp` line in `journalctl -k` afterwards.

## 3. The thesis: the Radeon should displace CPU joules, not add physics

Physics on the Radeon is bounded at about 0.2 to 0.25 of the RTX per clock, and at a shared power cap its watts come out of the CPU package. Under 2M/s the CPU package is the second wall: today one 3M generation costs about 4.7 s of breeding on 8 threads, about 14.6 thread-seconds of packing (0.32 s per 262k on 4 threads, measured), and 0.6 to 2.3 s of archive work. At 2M/s a generation lasts 1.5 s. The host chain is about 5x over budget before the RTX kernel is even touched. That is where the Radeon changes the equation: packing and parametric breeding are data-parallel, memory-light, and they can be written straight into the buffer the RTX DMAs from. The Radeon does them at its 800 to 1100 MHz levels for a few watts, and the CPU cores it frees stop pulling Dynamic Boost away from the RTX. The rate gain is then twofold: the host chain fits, and the RTX clocks higher.

The numbers per lever are in section 4. In short: the host-chain offload is worth about 1.1 to 1.2x on the RTX through power alone, and it is a precondition for any rate above about 400k/s, because at that rate the CPU chain becomes the wall regardless of the kernel. Radeon physics is worth at most 1.1 to 1.15x on top of a search-side rung the RTX can run by itself, and only if the power measurement says its watts are free.

## 4. Proposals ranked by expected gain

### P1. Measure the power split first (one afternoon, no code)

The single number my domain depends on: how much Radeon and CPU load lowers the RTX rate. Run `examples/p2_speed.rs` exclusive on the RTX with (a) the host idle, (b) 8 CPU threads busy (known: 0.79x), (c) a Radeon compute burner at each DPM level (`power_dpm_force_performance_level=manual` and `pp_dpm_sclk`, which the owner runs as root), (d) the burner at 1100 MHz plus 4 busy CPU threads. Log `nvidia-smi --query-gpu=clocks.sm,power.draw` and `hwmon` PPT at 10 Hz. Expected: at 800 to 1100 MHz the Radeon costs the RTX under 3%; at 2700 MHz it costs 10 to 20%, like the 8 CPU threads did. This decides whether P6 is allowed at all and sets the clock cap for P3 and P4.

### P2. The shared ring and the safe Radeon engine (about 1 week)

One page-aligned host allocation per ring block holds genomes, plans, packed lane records and results. The CPU writes plans, the Radeon (imported through `VK_EXT_external_memory_host`) breeds and packs in place, the RTX reads it through `cuMemHostRegister` and `cuMemcpyHtoDAsync` (the CUDA engine already stages uploads in pinned memory; this removes the staging copy) and writes results back into it, and the CPU absorbs. No host memcpy anywhere. PCIe traffic at 2M/s: about 2 to 3 KB of records per creature, so 4 to 6 GB/s of the PCIe 4.0 x8 link's 12 practical GB/s, plus 0.16 GB/s of results. LPDDR bus: about 10 GB/s for the Radeon's reads and writes plus the RTX's DMA, about 15% of 100 GB/s.

Two things to confirm on day one: that amdgpu's userptr import and NVIDIA's host registration accept the same anonymous pages (both use get_user_pages, but nobody has tried it on this machine), and the dispatch-time distribution of a packing kernel over 262k creatures. Gain: none by itself; it is the substrate for P3 and P4 and it removes the staging copy (about 3 GB per generation of memcpy at 3M).

### P3. Packing on the Radeon (with P2, about 1 week)

`warp_kernel::pack` builds a `physics2::Model` per creature (masses with bones, organs and muscles, slack lengths, strengths, joint ranges, the breadth-first lane numbering) and writes 12 x W lane words, up to 4 rounds of 16 x W muscle words and the end lists. That is 4.9 µs per creature per CPU thread today, 14.6 thread-seconds per 3M generation, about 10 cores at 2M/s. As a Radeon kernel it is one workgroup or one wave32 per creature: 20k to 50k instructions of gathers and float math per creature, no allocation. At 2.1 T instructions/s peak and 25% efficiency that is 10 to 25M creatures/s, so 3M in 0.15 to 0.3 s at the 1100 MHz level, under 8 W. Gate: 3M packed in at most 0.5 s, byte-identical to the CPU pack for the same population (the pack is integer and float arithmetic without transcendentals, so bit equality is a fair target; if RADV contracts an FMA differently the CPU twin can be written with explicit `mul_add` to match).

Gain estimate: the CPU chain loses 1.8 s of 8-thread time per generation, and by the P1 curve the RTX gains 5 to 15% clock when those cores go idle. Measurement: `worker_rate` end to end with packing on the CPU versus the Radeon, RTX clocks logged.

### P4. Parametric breeding and immigrants on the Radeon (after P3, 1 to 2 weeks)

The plan step stays on the CPU: it chooses emitters, parents and CMA slots per slot in ring order, which is what makes a seed deterministic. Emission splits by kind:

- CMA children: sample = m + sigma B D z per child, dimension 50 to 300, from a factor the CPU updates per emitter per generation. About 40k flops per child. The rank-mu update stays on the CPU.
- Local gaussian children (`local_mutation`): copy the parent, perturb 100 to 300 floats, clamp. 5 to 10k instructions.
- Immigrants (`Emitter::Restart`): random bodies from the generator, which is a fixed recipe.
- Structural children (64 operators on variable-size trees): stay on the CPU for now. They are the minority and the port is large.

Each child is emitted straight into the packed lane records (P3) without a genome round trip when its genome is not needed on the host, and the genome itself is written beside it for the archive. Parents come from an append-only elite table per generation in the shared ring (new elites are appended at absorption, old versions stay valid until the generation ends), so a child depends only on (seed, generation, slot, parent version) and the result is deterministic for this machine.

Determinism detail: `evolution::Rng` is a 64-bit integer generator; `shaderInt64` lets the Radeon run the same stream bit for bit. `qd::gaussian` uses log, sqrt and cos, whose bits differ between the CPU and ACO. Either accept that the run is deterministic per machine (as with the RTX kernel) or make the gaussian integer-exact (sum of uniforms in fixed point, or an integer ziggurat) so the CPU twin and the Radeon produce the same child. I prefer the second: then the Radeon can drop out mid-session and the run does not change.

Gain estimate: if parametric and immigrant children are 50 to 60% of a generation (the emitter shares adapt; the search expert should give the measured split), breeding on the CPU falls from 4.7 s to about 2 s per 3M, and packing for those children is already done. Combined with P3 the host chain drops from about 8.5 s to about 4 s of 8-thread time per generation. That is still 2.7x over the 1.5 s budget, so the CPU domain must still make structural breeding and the archive about 3x cheaper. Without P3 and P4 they would need 6x. Measurement: `BREED_NANOS` per stage and `EVOLUTION_STAGE_LOG` rows before and after, on the generation-70 long-session save as well as a fresh one.

### P5. Results digest on the Radeon (optional, days)

The RTX writes 80 B per creature into the shared ring. A Radeon pass maps each result to its niche, compares it with the cell record as of the last absorb, and flags the contenders. The archive's records only rise inside a generation, so a creature that cannot beat a stale record cannot beat the current one; the filter is conservative and the CPU still decides in ring order. The CPU domain may prefer to do this with AVX-512 in about 10 ms per 3M, which is fine; I list it because the results already sit in memory the Radeon maps, and because the archive stage (0.6 to 2.3 s) is itself over the 1.5 s budget. Gain: only what the archive expert says the flagged share is; likely the archive's cost is in insertion bookkeeping, not comparison.

### P6. Physics on the Radeon for a 2 s pre-rung (only after P1 to P4, and only if P1 says the watts are free; 2 to 3 weeks)

Trials that score must stay on the RTX: the GPU score is final and a replay must come from the scoring kernel. The Radeon can only run a rung whose survivors then run their full trial on the RTX from step 0. A rung at 5 s duplicates 300 steps for every survivor, so it gains little. A new lenient rung at 2 s (120 steps, keep about 50%) is the version that pays.

Arithmetic with today's mix (80% screened at 300 steps, 20% at 1200 steps, about 480 steps per creature): the RTX running the 2 s rung itself reaches 60 + 240 + 90 = 390 steps per creature, 1.23x, with no Radeon at all. The Radeon taking a share f of the rung at a step rate s of the RTX's balances at f x 120 = s x (480 - 180 f). At s = 0.13 (a port at half the RTX's per-clock efficiency, 2.0 GHz) f = 0.44 and the RTX sees 402 steps per creature. At s = 0.25 (a port as good as the CUDA kernel) f = 0.7 and 354 steps. So the Radeon adds 1.0 to 1.1x over the rung the RTX runs alone, and the rung itself is the search expert's lever, not mine. As the RTX kernel gets faster the share shrinks, because the Radeon's rate does not follow.

The port: the lane-group kernel with subgroup shuffles in WGSL or GLSL, wave32 forced, W = 8 and 16 only (W = 32 bodies stay on the RTX), fixed 120 steps per creature in dispatches of about 8k creatures (about 10 to 20 ms each), no regeneration. The physics rule says WGSL and CUDA change together, so this is a second maintained kernel forever. Register budget: 128 VGPRs per lane gives 12 waves per SIMD; shared memory of about 20 KB per 128 lanes gives 6 waves per SIMD on the 128 KB LDS, which is enough on RDNA.

Search gate before any code: record `com_x` at step 120 for one 262k wave on the RTX (`Params.screen_step = 120` with a bar below any distance keeps `screen_x` while the trial continues), then compute the share of the final top 1% and top 10% kept when the bottom 50% at 2 s is cut. The 5 s rung kept 100% and 96%. If the 2 s rung keeps under 95% of the top 1%, the whole of P6 is dead and the RTX-only rung too.

### P7. What not to do on the Radeon

Scoring trials, confirmation trials that set a record, replays: never, the score authority is the RTX. Persistent kernels: never, the timeout is 2 s. Device-local memory: never. A second driver stack: never. Bit-level agreement with the RTX kernel: not a goal, it is not a goal between CUDA and the CPU either.

## 5. Ceilings for this domain

- Radeon physics: 5 to 9M creature-steps/s (12 to 20% of today's RTX kernel, 0.5 to 0.9% of the 1.0 to 1.2 G creature-steps/s the goal needs). It cannot carry a share of the goal; it can carry a rung.
- Radeon packing: 10 to 25M creatures/s, so 3M in 0.15 to 0.3 s.
- Radeon parametric breeding: 20 to 50M children/s peak, so under 0.2 s per 3M for the parametric half.
- Memory bus: 10 to 15 GB/s of the shared 100 GB/s at 2M/s for the whole host chain plus the RTX's DMA. Not a limit. The compositor's traffic at 60 FPS is a few GB/s.
- Power: the Radeon at 800 to 1100 MHz is a 3 to 8 W device. At 2700 MHz it is a 20 to 30 W device that takes clock from both the CPU and, through Dynamic Boost, the RTX. P1 turns this into a curve.

## 6. What I need from the other domains

- CPU and breeding: a flat, fixed-stride genome and plan record (SoA or fixed-size rows, `f16` for slow muscle genes as the owner allowed) that a shader can read without pointer chasing; the measured split of children by emitter over a long session; agreement on an integer-exact gaussian and on `Rng` as the per-slot stream.
- CUDA kernel: the lane record format (`LANE_FIELDS` 12 x W words, 16-word muscle records, end lists, the two `heads` uint4) as a stable contract with a version number, so the Radeon packer targets it; and whether the engine can take its per-wave input from a registered host buffer instead of its own pinned staging.
- Data flow and ring: which allocation is the ring block, its lifetime, and who owns the page-aligned buffer so that both drivers can import it.
- Search: a verdict on the 2 s rung by the measurement in P6, and whether screened-by-rung creatures need any statistics beyond their 2 s `com_x`.
- HPC and power: agree that P1 is the first measurement of the whole plan, since it also decides how many CPU threads the game may keep busy at all.
