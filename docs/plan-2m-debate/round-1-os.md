# Round 1, Linux systems view: power, scheduling, memory, driver

Author: OS domain. Date: 2026-09-30, 18:20 to 18:45. Everything under "measured" was taken on the laptop today with read-only commands; nothing was changed. Raw traces are in scratchpad/os/ (trace.csv, limit.csv, clk.c).

## 1. What I measured on the box

Platform. Lenovo Legion Slim 5 14APH8. Linux 7.0.0-34-generic, HZ=1000, PREEMPT_DYNAMIC with lazy preemption, NO_HZ_FULL built but not enabled on the command line, sched_ext present and disabled, autogroup on. GNOME on Wayland. NVIDIA 580.178.04 open kernel module with GSP firmware, CUDA 13.0, NVRTC. nsys installed, ncu not, perf works (perf_event_paranoid 1). RAPL energy counters are root-only, but the APU package power (PPT) is readable at /sys/class/hwmon/hwmon9/power1_average, so CPU-side power can be logged without root. RmProfilingAdminOnly is 0, so GPU performance counters are open to user processes.

Clocksource. At boot the kernel measured a TSC warp of 8.1 billion cycles between CPU 0 and CPU 2 and marked the TSC unstable. The clocksource is HPET. Measured with a 2M-call loop: clock_gettime costs 1,377 ns per call on one thread (6,900 cycles) and 1,707 ns per call per thread with 16 threads, because HPET is one MMIO register that all cores serialize on. A TSC clocksource costs about 25 ns. Every Instant::now, every timeout, every hrtimer arm in the kernel and every frame timestamp in egui pays this. It is a firmware problem (the warp is 2.1 s of clock, one core started late), not a kernel setting; the owner should check for a BIOS update and read dmesg for "TSC warp" after the next boot. If it shows every boot, tsc=reliable is a gamble and I would not recommend it blind.

CPU power policy. amd-pstate-epp in active mode, governor performance, EPP performance, ACPI platform_profile performance (choices: low-power, balanced, performance, max-power, custom), boost on, preferred cores on. cpuidle acpi_idle with C1 (1 us), C2 (18 us), C3 (350 us exit latency). Under this policy any runnable thread drives its core to full boost, which is the worst case for the shared power budget below.

GPU power. Power limit 85 W base (default 55, board maximum 105); Notebook Dynamic Boost supported and nvidia-powerd running, which is the 85 to 100 W movement the facts file describes. Runtime D3 fine-grained is on (the GPU powers off when the game closes its engines, so the first open after a pause costs a second or two). Persistence mode is off. PCIe trains to Gen 4 x8 under load and drops to Gen 1 idle. Addressing mode is HMM.

Limiter history. The driver's counters for this 5-day uptime: software power capping active 1,419 s, software thermal slowdown 1,321 s, hardware thermal slowdown 37 ms, hardware power brake 7.5 s. The owner's long sessions spend minutes at the power cap and minutes thermally throttled. "Sustained for minutes" therefore means "at the limiters", not at the clocks a 30 s benchmark shows.

Load trace (the 09-28 search_ab build, so the older per-thread kernel; the platform facts do not depend on the kernel version). Two generations of 786k creatures, 20 s trials, four CPU threads through cpu-slot.sh, sampled every 250 ms:

- GPU phase, 11 s: SM clock 2,490 MHz flat (2,505 once), memory 8,001 MHz, power 72 to 81 W, no limiter reason active, 66 to 68 C, PCIe Gen 4. A second 5 s run drew 74 to 90 W. So at today's 19% issue utilization the GPU sits at its top boost bin with about 10 to 25 W of headroom below the cap. It is clock-bound, not power-bound, today.
- APU package power during the GPU phase with four packing and breeding threads: 11 to 15 W. During the CPU-only tail (breeding after absorption): 45 to 51 W at four threads. All-core breeding on 16 threads will sit at the APU's 54 W sustained limit.

Host cost per creature. perf stat on a one-generation run of 262k creatures (GPU scoring, four host threads): 92 G cycles, 214 G instructions, IPC 2.3, 20.7 s of task-clock in 5.4 s wall, 170k page faults, 2% sys. That is about 350k cycles, 80 us of thread time, per creature, for packing, breeding the next block, verdicts, metrics conversion, and polling. Even if half of it is polling and rayon spin, the host spends 30 to 40 us of thread time per creature.

Memory. 30 GB RAM, 3.2 GB in swap right now with 6 GB free and 17 GB of page cache, kswapd and kcompactd among the busiest kernel threads. THP is madvise-only, so the ring's 5 to 9 GB of Vecs run on 4 KB pages unless the allocator asks. No hugetlb pages. swappiness 60.

Interrupts. The NVIDIA MSI (IRQ 91, 15.9 M interrupts) lands entirely on CPU 7; the amdgpu MSI-X on CPU 13. No irqbalance. Harmless at today's rate.

Game threads today. Rayon pool of 16 threads at nice 10 (main.rs start_handler). The worker thread at nice 0. The window renders on the compositor's GPU, the Radeon, by vendor match (ui.rs), so the RTX has one context, the game's CUDA context, plus gnome-shell's 2 MiB EGL context. The engine thread polls cuEventQuery every 200 us with 1 ms poll timeouts and the scheduler waits in 2 ms slices, so several thousand timer wakeups per second, each with an HPET read.

## 2. Analysis: the ceiling is joules, not cycles

Nobody else on the team owns this number, so here it is. The RTX 4060 in this chassis gets 85 W, plus up to 15 W of Dynamic Boost while the CPU is quiet. Today the kernel draws about 75 W at 2.49 GHz while issuing 19% of the time, that is about 1.45 T thread-instructions/s for 45 M creature-steps/s. Subtracting about 20 W of static, memory and fan power, that is roughly 38 pJ per thread-instruction at this voltage. A kernel that issues 60% of the time at the same clock would want about 240 W. It will not get it. GPU Boost will drop to a lower voltage and frequency point until the board fits in 100 W (or 85 W while the CPU breeds). Energy per instruction falls about 40% between 1.0 V and 0.8 V, so my estimate of the sustained budget on this laptop is about 3 T thread-instructions/s at around 1.9 to 2.1 GHz, not the 7.65 T of the issue peak. Thermal steady state after minutes takes another 5 to 10% (the 1,321 s of thermal slowdown says the cooler cannot hold 100 W for long).

Consequences for question 2 of the facts file:

- At 1.1 G creature-steps/s the sustained budget is about 2,700 executed thread-instructions per creature-step. Today's lane-group kernel executes about 32,000 (1.45 T / 45 M, idle lanes at tree levels included). So the kernel needs roughly a 12x cut in executed thread-instructions per creature-step, and then it also needs to run at 100% issue, which nothing does. At a realistic 60% issue it needs 20x. That is on the edge of possible; a 12x to 20x rewrite that also keeps the physics honest is a research result, not a tuning result.
- Every lever that removes instructions also removes joules, so it counts fully. Every lever that only removes stalls (occupancy, ILP) counts only until the power cap, then it converts into a lower clock. The HPC plan should be stated in executed instructions and joules per creature-step, not in issue efficiency.
- Fewer steps per creature (screening rungs, adaptive trial length) is worth exactly its ratio in joules and is the cheapest lever on this laptop. From my side: every creature-step avoided is a creature-step that did not need to fit under 100 W.
- My answer to "is 2M/s sustained reachable here": with the best imaginable kernel and today's step count, 1.0 to 1.5 M/s sustained is what the power budget allows; 2 M/s needs the step count to fall by 1.5 to 2x on top, or the RTX to spend zero watts on anything but physics, which is proposals 2 and 3 below.

Calibration that settles the estimate (30 minutes, no risk, no root): run the current kernel at four occupancies (launch_bounds or fewer blocks per SM) and one synthetic FMA kernel at 100% issue, each for 30 s, sampling clocks.sm, power.draw, temperature and the limiter reasons at 100 ms, plus `nvidia-smi -q -d POWER` for the live limit. Fit watts against issue rate; read the clock at the cap directly from the FMA run. That single chart replaces every "x times" claim in this debate with a joule budget.

## 3. Proposals, ranked by expected gain

### P1. The host must cost under 4 us of thread time per creature, or the host work leaves the CPU

Measured: 80 us of thread time per creature today (30 to 40 us net of polling). At 2 M/s the CPU has 16 core-seconds per second, so the budget is 8 us per creature with every core busy, and a busy CPU takes the 15 W Dynamic Boost away from the GPU, which the facts file measured as a 21% GPU rate loss with 8 threads. So the real budget is about 4 us per creature on 8 threads, with the other 8 threads idle.

What the OS side can remove now, with numbers:

- No per-creature allocation. 0.65 page faults per creature measured. At 2 M/s that is 1.3 M faults/s, about 1.5 core-seconds per second of fault handling and zeroing, and every large Vec free is an munmap with a TLB shootdown IPI to all 16 CPUs, which scales badly with 16 breeding threads. Preallocate each ring block's arrays once (structure of arrays, fixed capacity per lane class) and reuse them forever. Gate: perf stat page-faults per generation near zero, sys time under 1%.
- Pack straight into write-combined pinned memory (cuMemHostAlloc with WRITECOMBINED) and let the kernel read creature records from host memory at trial start (zero-copy over PCIe Gen 4 x8; at 2 M/s times about 1 KB that is 2 GB/s of a practical 13 GB/s, and each record is read once per 300 to 1,200 steps so the 1 to 2 us PCIe latency is amortized). This deletes the staging memcpy, the cuMemcpyHtoD, the device lane buffers, and the upload bookkeeping. Estimate: packing from 4.9 us to about 1 us of thread time per creature; VRAM freed for physics state.
- Results written by the kernel into pinned host memory (80 B each, 160 MB/s at 2 M/s) with a per-wave sequence word, so the host absorbs continuously instead of after a DtoH copy. This also smooths the CPU load, which helps the 60 FPS side.
- Replace event polling with blocking waits (a stream callback via cuLaunchHostFunc, or cuEventSynchronize on a thread in a blocking-sync context). Frees the few percent of one core the 5,000 timer wakeups per second cost, and removes 5,000 HPET reads per second.

Everything else per creature (breeding at about 25 us of thread time, verdicts, metrics) is not OS work. It either gets 6x cheaper or it moves to a GPU (P2). Measurement for the whole proposal: perf stat cycles per creature on a fixed-seed run, target under 20k cycles per creature (4 us at 5 GHz) with the page-fault count and the sys time beside it.

### P2. Breed and pack on the Radeon 780M, so the RTX keeps its 100 W and the CPU stays quiet

Where the joules go decides this. Physics on the iGPU is a losing trade: the 780M shares the APU's 54 W with the CPU, and every watt it burns there is a watt Dynamic Boost cannot give the RTX, so a 15% gain there costs about 15% on the RTX. Breeding on the iGPU is the winning trade: it is a few thousand operations per child, so 2 M children/s is under 10% of the 780M's 3 to 4 usable TFLOPS at maybe 8 to 12 W, and the CPU goes from all-core breeding at 54 W to near idle, which returns the 15 W boost to the RTX (about +15% GPU rate under load, from the facts file's own 174k to 137k measurement) and stops the thermal soak from the shared heatpipe.

Data flow that makes it work with two GPUs and determinism: the genome ring lives in host memory, allocated as pinned memory for the RTX and imported by the Radeon through Vulkan external host memory (VK_EXT_external_memory_host, which RADV supports). The Radeon writes children in place; the RTX reads them zero-copy as in P1; results land in host memory; the archive verdict runs on the CPU (a few hundred thousand comparisons per second) or on the Radeon. No PCIe copy ever happens on the host's clock. Breeding on the Radeon is a pure function of (seed, block, generation) with an integer RNG, so the search stays deterministic for a seed on this laptop, and ring-order absorption already makes finish order irrelevant.

The crash risk, addressed rather than waved away. The desktop died once under heavy Radeon compute. amdgpu here has gpu_recovery at -1 (auto, enabled on RDNA 3) and the default lockup timeout of 10 s, so a dispatch that runs past 10 s gets the GPU reset under the compositor, which is what a desktop death looks like. Rules for the breeding queue: dispatches of at most 10 to 20 ms of work, a compute-only queue created with VK_EXT_global_priority LOW (RADV allows LOW to unprivileged processes), never more than two dispatches in flight, and a watchdog in the game that stops submitting when the compositor's frame time (readable from the presentation timing of our own window) rises. Gate before any port: a 10-minute soak of a synthetic 20 ms compute loop on the Radeon under GNOME with the sampler logging APU PPT, RTX power and clock, and our own frame times; pass is zero frame drops over 16.7 ms and the RTX rate unchanged. Second gate: one structural operator and one CMA step ported, checked bit-equal against the CPU implementation on 1 M children.

Estimate: this is what makes 2 M/s host-feasible at all (P1 alone leaves breeding at about 25 us of thread time per creature, 50 core-seconds per second at 2 M/s, which the CPU does not have), plus the 15 W of boost back on the RTX. Engineering cost is the biggest on this list: 64 operators plus repair in a compute shader. If the search domain says breeding cannot be expressed data-parallel, the fallback is the RTX breeding into the same host ring (HPC's 7.5); it costs RTX joules but few.

### P3. Owner-side power settings, each a 30 s measurement

- Memory clock lock. DRAM use is 1 to 5%. Locking GDDR6 at its lowest useful P-state (`nvidia-smi -lmc`, root) saves an estimated 5 to 10 W of the 100 W, which at the cap is 5 to 10% more SM clock. Worth nothing today (not capped) and worth its full value once the kernel is dense. Measure: kernel rate, power and clocks.sm at 8,001 versus 5,001 versus 810 MHz memory clock.
- Platform profile max-power. The board limit is 105 W and the current limit is 85 W base. Measure `nvidia-smi -q -d POWER` under load in performance versus max-power. If the base rises to 100 or 105 W, that is +5% at the cap for free.
- CPU frequency cap for breeding. With EPP performance every breeding thread runs its core at 5 GHz, where Zen 4 pays about 3x the power per instruction of 3.3 GHz. Cap scaling_max_freq at 3.3 GHz on the cores rayon uses (or set EPP balance_performance system-wide) and measure the GPU rate under concurrent breeding, APU PPT, and breeding seconds per generation. Estimate: recover 5 to 10% of the GPU rate under load for a 20 to 30% longer breeding pass, which is a win only while breeding is off the critical path, and moot after P2.
- Pinned SM clock for sustained runs. Once at the limiters, a fixed clock (`-lgc`) just below the cap avoids the limiter oscillation and the thermal overshoot. Measure the 10-minute rate with and without; expect a few percent and much flatter generation times.
- A 10-minute trace of the owner's real game with the sampler (scratchpad/os/sample.sh) is the baseline all of these are judged against. Nobody has that trace yet; the limiter counters say it will not look like the 30 s benchmarks.

### P4. Keep 60 FPS under a saturated pipeline

The window is already on the Radeon and rayon is already at nice 10, which is most of it. What is left, cheap, and measurable with the existing control-latency probe (p99 of Command::Ping) plus a frame-time histogram the UI should add:

- Rayon at 14 threads, not 16, so the UI thread, the worker thread and gnome-shell always find a core during a breeding burst. Costs 12% of breeding throughput until P2 makes breeding cheap.
- The UI thread asks EEVDF for a short slice: sched_setattr with sched_runtime of about 1 ms. On this kernel a task with a short requested slice preempts a long-slice task on wakeup, which is exactly a frame thread waking under 16 breeding threads.
- Breeding threads as SCHED_BATCH in addition to nice 10, so the scheduler stops treating them as interactive on wakeup.
- Absorb results continuously (P1's host-resident results) so the worker's per-block work becomes a stream of small steps instead of a few-tenths-of-a-second block, and Command handling never waits behind one.
- Expected: p99 frame time under 10 ms during breeding, p99 control latency under 5 ms. Not measured today; the benchmark's own numbers are the gate.

### P5. Things from my domain that do not pay here, said once

- io_uring: the game writes a few small files; there is no I/O to overlap.
- CPU evaluation: 1 TFLOPS at 54 W against 15 TFLOPS at 100 W, and it steals the RTX's boost. Zero.
- Huge pages for the ring: worth trying only as a one-line madvise(MADV_HUGEPAGE) on the preallocated block arrays of P1; TLB misses were 2.2 M in 92 G cycles, so under 1%. Swap is the real memory risk: 3.2 GB is swapped right now, and a 5 to 9 GB ring that the allocator churns will page. P1's fixed arrays are the fix; mlock is not.
- IRQ affinity: the NVIDIA MSI on CPU 7 is fine at a few thousand interrupts per second. If P1 moves to interrupt-driven waits, keep the engine thread off CPU 7 or on it, either is fine; do not spend time here.
- CUDA graphs: launches are 1 s long today and 130 ms at 2 M/s; launch overhead is noise.

## 4. What I need from the other domains

- HPC and kernel: the executed thread-instruction count per creature-step (from nsys or the SASS count) for the current kernel and for each proposed kernel, so I can turn them into watts with the calibration curve. Also whether the kernel can take creature records from host pinned memory at trial start with one coalesced read per lane group, and write results straight to host memory.
- Search: how much of breeding is data-parallel per child (operators, repair, CMA sampling), so P2 can be sized; and the host cost of a verdict and of to_metrics per result, which must fit in the 4 us per creature host budget.
- Physics: nothing beyond a number: steps per creature under each proposed screening or trial-length policy, because joules scale with it one to one.
- UI: a frame-time histogram in the benchmark report and the presentation timing hook the P2 watchdog needs.
- Owner: five root or firmware actions to test, each a 30 s measurement: `-lmc`, platform profile max-power, a cpufreq cap for breeding cores, `-lgc` for sustained runs, and a BIOS check for the TSC warp. And one 10-minute sampled trace of the real game.

## 5. Order that keeps the game playable at every merge

1. Calibration chart (watts against issue rate, clock at the cap) and the 10-minute trace of the real game. Half a day. Sets every target.
2. P1 pieces in order: preallocated block arrays, zero-copy records and results, blocking waits. Each is a separate merge with page faults and cycles per creature as the gate.
3. P4 scheduling tweaks alongside, with the control-latency probe as the gate.
4. P2 soak test on the Radeon before any port; then one operator; then the rest.
5. P3 owner settings whenever the owner has a minute, since none touches the code.
