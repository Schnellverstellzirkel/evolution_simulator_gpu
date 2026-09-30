# Round 3, OS domain: the power plan, the host that stays quiet, and 60 FPS

The pipeline I back is the converged one from the chair's rulings: CPU breeds in cpu's fixed-array form with the counter RNG, writes genes once into pinned host memory in SoA form, the RTX unpacks at take-up, computes the descriptor bins and the candidate bit, runs the ladder's rung checks in the kernel, writes 80 B results into a pinned host ring, and the CPU commits candidates in ring order. My part is everything around it that decides whether the RTX gets its 100 W and whether the desktop stays at 60 FPS while it does. The design lands on the chair's row "2 substeps, R1 to R3" at 3T (1.0M/s) as the floor and on "1 substep, R1 to R3" at 5T (2.7M/s) as the ceiling; the calibration below decides which, and the host tax I estimate at 5 to 8% is inside every number.

## 1. The power plan as a schedule for the owner

Everything here is a 60 s row unless stated, logged by scratchpad/os/sample.sh (SM clock, memory clock, power, temperature, utilization, limiter reasons, PCIe generation, APU PPT, CPU MHz at 4 Hz) plus `nvidia-smi -q -d POWER` for the live limit and `journalctl -k` for amdgpu events. The RTX load is p2_speed from the warp worktree on the 262k generation-10 dump, three timed passes, which is 3 minutes per row at today's rate. All rows need the exclusive lock and a paused game, because Dynamic Boost reacts to the whole package: my attempt this round found 28 to 54 W of APU load from agent builds with the GPU idle. So the schedule also needs no agent builds during a session; the chair announces the window.

Session A, one 5-minute pause, no root: row 0 (baseline, CPU quiet) and row 1 (2 busy CPU threads, a C spin loop of AVX-512 FMAs in the scratchpad). Session B, one pause, no root: rows 2 and 3 (4 and 8 threads). Session C, one pause, no root: row 4 (16 threads) and row 19a (the CPU rows again with the work in 20% duty bursts, see section 6). Session D, one pause, root: rows 5 and 6 (8 threads with scaling_max_freq 3.4 GHz on all cores; then EPP balance_performance). Session E, one pause, root for DPM: the Radeon rows in igpu's recipe, burner started from a terminal, not from the row script: row 7 Radeon at 100% duty at 800 MHz, row 8 at 1,100 MHz, row 9 at 2,700 MHz (power_dpm_force_performance_level manual and pp_dpm_sclk), row 10 Radeon at 1,100 MHz at 30% duty plus 4 CPU threads. Session F, one pause, root: row 17 (memory clock locked at the lowest P-state, `nvidia-smi -lmc`, then at 5,001 MHz) and row 18 (platform profile max-power, reading the live limit). Session G, one pause, no root but a worktree build: rows 11 to 13, the three synthetic NVRTC kernels (register-only FMA chain; FMA with one shared load and one shuffle per two FMAs; integer and shuffle only), 30 s each at full occupancy, with an nsys GPU-metrics capture for issue rate; and rows 14 to 16, p2_speed at 4, 2 and 1 blocks per SM. Session H, outside the 5-minute rule: row 19, p2_speed looped for 10 minutes for the thermal steady state, which needs the owner away from the game or a longer pause the owner grants once.

That is seven 5-minute pauses plus one 10-minute run, about 50 minutes of GPU across an evening, and the rows that need root are 5, 6, 7 to 10 (DPM), 17 and 18. The outputs, in the order the other domains need them: nJ per warp instruction by mix and the clock at the cap (rows 11 to 13), which fixes the budget between 3T and 5T; the Dynamic Boost curve against CPU threads (rows 0 to 4) and against a capped clock (5, 6), which fixes the host tax; the Radeon cost (7 to 10), which closes or keeps the fallback breeder; the sustained clock (19), which is the number "sustained" means.

## 2. Sustained-run policy

The game never sets clocks or profiles; those are the owner's system settings, so no knob enters the game. The policy is a rule the owner applies once, after the rows exist:

- SM clock: if row 19 shows the limiter oscillating (clock swings over 5% within a minute), pin the SM clock (`-lgc`) at the steady-state value minus 50 MHz. Expected gain is a few percent and flat generation times; if the rate is within 1% unpinned, do not pin.
- Memory clock: lock at the lowest P-state whose row 17 rate is within 1% of the baseline. DRAM use is 1 to 5%; I expect the lowest state to hold and to return 5 to 10 W at the cap, which is 5 to 10% of rate once the kernel is dense and nothing before.
- Platform profile: whichever of performance and max-power gives the higher live limit in row 18, if it moves the rate by 3% or more; otherwise leave it.
- CPU: the breeding cores capped at 3.4 GHz (or EPP balance_performance) if rows 5 and 6 recover 5% or more of the GPU rate against row 3 at the same core-seconds; cpu owns the reading of those rows.
- The TSC: check the BIOS and read dmesg for "TSC warp" after the next boot; if the warp is gone the clocksource returns to TSC on its own and every timestamp drops from 1.4 us to 25 ns. No kernel parameter until that is known.

Every setting is judged by one row and dropped if it moves the rate under 3%.

## 3. The 60 FPS changes

The window already renders on the Radeon and rayon workers are already at nice 10. The remaining changes are in main.rs and worker.rs, about 40 lines, no behaviour change to the search:

- Rayon pool at available_parallelism minus 2 (14 here), so the UI thread, the worker thread and gnome-shell always find a core during a breeding burst. Costs 12% of breeding throughput until cpu's rewrite makes breeding a fraction of the CPU, after which it costs nothing.
- Rayon workers as SCHED_BATCH in the start handler, keeping nice 10, so the scheduler stops crediting them as interactive on wakeup.
- The UI thread requests a 1 ms EEVDF slice with sched_setattr (sched_runtime = 1,000,000 ns) at startup; on this kernel a short-slice task preempts a long-slice task on wakeup, which is a frame thread waking under 14 breeders. The worker thread stays at the default.
- No pinning. The IRQ layout (NVIDIA on CPU 7, amdgpu on CPU 13) is harmless at a few thousand interrupts per second.

Gate: the existing benchmark's control-latency probe at p99 under 5 ms during breeding, and a frame-time histogram (the UI domain adds it to the benchmark report) at p99 under 10 ms. If the histogram shows the drops on the Radeon render side rather than CPU starvation, the fix moves to the UI domain.

## 4. Zero-copy and blocking waits

The engine changes, in cuda_engine.rs and engine.rs, in data's API:

- Genes: the breeder writes each block's SoA gene arrays into pinned host memory allocated once per ring block with cuMemHostAlloc, WRITECOMBINED because the CPU only writes and the RTX only reads, and MAPPED so the kernel takes a device pointer. The kernel's take-up reads the creature's genes through the mapping: 12 to 32 PCIe transactions at about 1.5 us, 20 to 50 us per creature against a 15 to 40 ms trial, 0.1 to 0.3% (data's count, I agree). PCIe traffic at 2M/s and 1.5 KB per generation-51 creature is 3 GB/s of a practical 13 GB/s on Gen 4 x8. If the calibration or the stub kernel shows the take-up stall is not hidden by the other warps, the fallback is one cuMemcpyHtoDAsync per block into VRAM (data's alternative), same watts, 0.5 GB per 262k in flight.
- Results: the kernel writes its 80 B result into a pinned host ring (cached, portable, MAPPED) at the creature's slot, then a per-block completion word. No DtoH copy, no readback buffer, no lane buffers, no staging memcpy. The CPU absorbs candidates while the wave still runs.
- Waits: the primary context gets CU_CTX_SCHED_BLOCKING_SYNC; the engine thread waits on the block's event with cuEventSynchronize, or the kernel's completion is signalled through cuLaunchHostFunc. The 200 us polling loop, the 1 ms poll timeouts and the 2 ms collect waits go, which removes about 5,000 timer wakeups and HPET reads per second and the few percent of a core they cost.
- Allocation: nothing on the hot path allocates. Page faults go from 0.65 per creature (measured) to zero; huge pages are one optional madvise on the block arrays, kept only if the dTLB counter says so (under 1% today).

Gate: perf stat on one generation at generation 51 shows page-faults under 1,000, sys time under 1%, host cycles per creature under 20k, PCIe RX under 4 GB/s in nsys, and the kernel rate on p2_speed within 1% of the copy-based path.

## 5. Watts per lever, at both budgets

Under the cap a joule is the unit. The table starts at 43 to 45M creature-steps/s and 480 steps at the generation-3 body, which the chair rules is 0.6x at the generation-51 body (26 to 27M). Each lever is stated as the fraction of executed instructions or steps it removes, then converted at 3T and 5T. The kernel numbers are the chair's working range (300 to 450M creature-steps/s at 2 substeps for the mix; 550 to 800M at 1 substep), taken as the 3T and 5T ends.

| lever | what it removes | creature-steps/s at 3T | at 5T | note |
|---|---|---:|---:|---|
| today, generation-51 body | base | 27M | 27M | clock-bound, 75 W, not capped |
| per-lane maximal kernel, 2 substeps | 32,000 to about 8,000 to 12,000 executed slots per creature-step, and the MIO share from a third to a few percent | 300M | 450M | the MIO removal is worth 3 to 4x in joules per instruction; it is what separates 3T from 5T |
| 1 substep (owner, physics) | half the substep work, 1.9x in joules | 550M | 800M | 50% odds per physics |
| steps 480 to 300 (R1 to R3) | 37% of steps and joules | x1.6 in creatures/s | same | counted once, ga and ml |
| steps 300 to 235 (R4) | 22% more | x1.28 | same | ml's rate; ga says 5% fire rate |
| memory clock lock | 5 to 10 W of 100 | +5 to 10% | +5 to 10% | only when capped; zero today |
| platform profile max-power | 0 to 5 W | 0 to +5% | 0 to +5% | row 18 decides |
| host tax, CPU at 2 to 3 busy cores | 5 to 15 W of the 15 W Dynamic Boost swing | −5 to −8% | −5 to −8% | rows 1 to 3 decide; −21% if breeding stays as today |
| thermal steady state | 5 to 10% of clock after minutes | −5 to −10% | −5 to −10% | row 19 decides; pinning flattens it |

Creatures per second, host tax and thermal included (multiply by about 0.87): 2 substeps and R1 to R3 give 0.87 to 1.3M/s; 2 substeps and R1 to R4 give 1.1 to 1.7M/s; 1 substep and R1 to R3 give 1.6 to 2.3M/s; 1 substep and R1 to R4 give 2.0 to 3.0M/s. So 2M/s sustained at generation 51 needs 1 substep, or the full ladder at the 5T budget with the memory clock and profile rows both paying. Without 1 substep and at 3T the honest ceiling is 1.1 to 1.3M/s.

The device-side helpers of the converged design, costed in joules as a share of a mean generation-51 creature's trial (300 steps x about 6,000 executed slots at 2 substeps = 1.8M slots; at 1 substep the shares double):

| helper | executed slots per creature | share of a trial | watts at 100 W |
|---|---:|---:|---:|
| take-up unpack (Model::new on the device) | about 2,000 (gpu) | 0.11% | 0.1 W |
| descriptor bins and candidate bit, one L2 read of the occupant table | about 100 | 0.006% | under 0.01 W |
| rung checks, R1 to R4: bar index, one L2 read per rung, two extra distance samples | about 5 per rung plus 10 | 0.002% | under 0.01 W |
| ladder histograms, integer atomics per rung | about 50 | 0.003% | under 0.01 W |
| audit lane: 1% of creatures at 1,200 steps instead of 300 | 9,000 extra steps per 1,000 creatures | 3% | 3 W |
| parametric recipes on the RTX (insurance only) | about 5,000 x 65% | 0.18% | 0.2 W |
| structural operators on the RTX (withdrawn) | about 50,000 x 35% | 1.0% | 1 W |
| spanning take-up counter | one device word read per take-up | under 0.001% | 0 |
| results written to host memory | one 80 B store over PCIe | under 0.01% | 0 |

The audit lane is the only helper that costs a whole percent, and it is 3% of joules for the labels that make the ladder honest; everything else together is under 0.3%. There is no device-side helper in the design whose joule cost changes a row of the table.

## 6. One idea nobody has raised: shape the host work to the boost controller

Dynamic Boost is not a wattmeter on the CPU; nvidia-powerd receives a platform signal and moves the 15 W in steps on its own sampling interval, which is seconds, not milliseconds. If the controller keys on utilization or on a short average, a steady 20% load on three cores looks the same to it as three busy cores and costs the RTX the full 15 W (21% of rate, the measured point at 8 threads), while the same core-seconds delivered as 20 ms bursts at 100% on all cores followed by 80 ms of idle may cost only their true energy share, about 5 W and 5%. The ring already works in blocks of 50 ms of GPU work, so bursting the host chain per block is a scheduling shape, not a knob: breed and commit a whole block on all 14 rayon threads at once, then sleep until the next block. The number: up to 15% of GPU rate for the same CPU work if the controller is utilization-keyed, zero if it is energy-keyed. The measurement is row 19a in session C: rows 1 to 3 repeated with the spin loops at 20% duty in 20 ms bursts, same total core-seconds, and the RTX rate and the live power limit beside them. Ten minutes of GPU, no code in the game, and it decides how cpu's pipeline should pace itself.

## 7. What happens on a world change, a save, a replay click

World change: the host stops bumping the take-up counter for the old world, the kernel drains (data's retarget path), the ring at 0.3 to 1 s loses at most a second of work instead of today's 26% of a generation; nothing in the power or scheduling plan changes. Save: archives and search state only, no ring contents, written by a thread at nice 10; a few MB. Replay click: the replay slot's streams already carry the highest CUDA priority, and with blocking waits the engine thread wakes on the replay's event first; the replay is a one-creature recording wave, under 100 ms of GPU. None of these touch a clock or a profile.

## 8. What stays out

CPU physics (1 TFLOPS at 54 W against the RTX's boost), Radeon physics (closed), io_uring (no I/O), IRQ affinity, CUDA graphs (130 ms launches), persistence mode (the game holds its context), HMM page-faulted access to genes (pinned mapped memory instead), and any clock or profile setting inside the game.

## 9. Determinism

The plan adds no source of nondeterminism: clocks and profiles change speed only; blocking waits change when the host learns a result, and absorption stays in ring order; results written to host memory carry the same bits the DtoH copy carried; the burst pacing of section 6 changes timing only. The search stays a function of the seed on one GPU.

## 10. Tracks, in order, each a separate merge

1. T0, the power table: sessions A to H above, owner-scheduled, no code in the game, one 100-line example and one C spin loop in a worktree. Gate: the nJ per warp instruction and the cap clock exist, the Dynamic Boost curve exists, row 19 exists. Half a day of GPU across an evening.
2. T1, 60 FPS scheduling (section 3): 40 lines. Gate: control latency p99 under 5 ms, frame p99 under 10 ms during breeding, breeding seconds per generation within 15% of before.
3. T2, blocking waits and no hot-path allocation: 150 lines in cuda_engine.rs and the ring's block arrays. Gate: page faults under 1,000 per generation, sys under 1%, rate within 1%.
4. T3, genes in pinned mapped memory and results in a pinned host ring, with data's take-up unpack: the engine half of data's track. Gate: the section 4 numbers, and p2_speed rate within 1% against the copy path; if the take-up stall shows, switch to the per-block DMA and keep the results ring.
5. T4, the sustained-run policy (section 2): applied by the owner after row 19, documented in docs/building.md as system settings, not game settings. Gate: the 10-minute rate flat within 3%.
6. T5, burst pacing of the host chain (section 6): only if row 19a shows over 5%; then it is a change to when cpu's breeder runs, not what it computes.

Where it stops, honestly: at 2 substeps and the 3T mix the whole system sits at 1.1 to 1.3M/s sustained at generation 51, and no row of my table moves that by more than 10%. The two decisions that reach 2M/s are the owner's on 1 substep and the kernel's instruction mix, and the calibration rows tell us within a week which of the two we are betting on.
