# Round 2, OS domain: answers, concessions and the power table

## D1. The 3T sustained budget: its uncertainty and the calibration

The chair is right about the derivation. The 38 pJ came from 55 W of dynamic power over 1.45 T thread-instructions/s at 19% issue, and that mix is heavy in shared-memory loads, shuffles, barriers and half-empty warps. Restated per warp instruction, which is what the hardware pays for: 239 G warp-instructions/s at 19% is 45 G/s, so about 1.2 nJ per issued warp instruction, all-in above static. A register-resident FMA warp instruction on Ada costs far less. The public reference point is a desktop AD102 saturating FP32: about 450 W for 128 SMs x 4 schedulers x 2.7 GHz = 1.38 T warp-instructions/s, about 0.33 nJ per warp instruction including static and memory. So an FMA-dense kernel pays 3 to 4x less per warp instruction than our present mix.

What that does to the budget. At 0.33 nJ per warp instruction, this GPU's full issue rate at 2.49 GHz (239 G/s) would draw about 80 W plus static, right at the 100 W cap. At 1.2 nJ it draws 290 W and clocks down to about 2.0 GHz with a 3 T thread-instruction budget. So the sustained budget on this laptop is between 3 T and 7.6 T thread-instructions/s, a factor of 2.5, and the instruction mix decides where. The rule that survives the uncertainty: shared-memory traffic, shuffles and register spills are the expensive instructions; register-resident FMA and integer work are cheap. That favours gpu's W = 2 register design and physics's per-lane direct solve over anything that walks trees through shared memory, and it means "instructions removed" should be weighted: an MIO instruction removed is worth 3 to 4 FMA instructions removed in joules. My 3 T figure stands as the lower bound for a kernel that keeps today's mix, and I withdraw it as a point estimate for a register-resident kernel.

The calibration run, exactly:

1. Three synthetic kernels compiled through the game's own NVRTC loader (there is no nvcc on the box) in a 100-line example in a worktree: (a) a dependent FMA chain per thread, registers only; (b) the same with one shared-memory load and one shuffle per two FMAs, about 50% MIO; (c) an integer and shuffle-only loop. Each at full occupancy, 30 s.
2. The lane-group kernel through p2_speed on the generation-10 dump at 4, 2 and 1 blocks per SM (a grid cap), 30 s each.
3. Logged at 100 ms: clocks.sm, clocks.mem, power.draw, temperature, the limiter reasons, plus `nvidia-smi -q -d POWER` for the live limit (the csv query reports it N/A on this driver), the APU PPT from hwmon9, and for rows 1 and 2 an nsys GPU-metrics capture for issue rate and pipe utilization. Output: nJ per warp instruction by mix, and the SM clock the limiter settles at for each mix.

Can I run it this round: no. Three reasons, all measured today. The GPU is free now (the 210 MiB process the chair saw was a release-fast build that exited within my 60 s sample), but the APU package reads 28 to 54 W from other agents' CPU jobs with the GPU idle, which moves Dynamic Boost and contaminates every row; p2_speed is not built in target/release (only the 09-28 release-fast examples exist, without p2_speed), so a worktree build is needed; and the synthetic kernels are code that does not exist yet. It needs the exclusive lock, a quiet CPU, and about 20 minutes of GPU time. The chair should schedule it after the debate; I will build the example and the sampler is ready (scratchpad/os/sample.sh).

## D3. Structural operators: CPU, RTX or Radeon

Ranked by engineering cost, lowest first:

1. CPU, in cpu's fixed-array form. The 7,500 lines keep their logic and language; containers change. Debuggable against today's output child by child. One implementation, no determinism seam.
2. RTX, data's sorted port. A second implementation in CUDA C++ through NVRTC, one driver stack, the CPU as the reference, printf available. Weeks.
3. Radeon. The same port in WGSL or GLSL, a second driver stack, a 2 s ring timeout (igpu read modinfo correctly: the default is 2,000 ms; my 10 s was wrong and I withdraw it), no printf, and RADV's FMA contraction to pin. Weeks plus a maintenance tail.

Ranked by watts at 2 M/s, lowest first, with the numbers:

1. RTX: 1 M structural children per generation x about 50k instructions = 5 x 10^10 thread-instructions, at a 3 T/s budget about 17 ms of a 1.5 s generation, about 1% of rate, which under the cap is about 1 W-equivalent.
2. Radeon: 3 to 8 W at 800 to 1,100 MHz (igpu's figure; I have no measurement of my own), but drawn from the APU's 54 W, which is the pool Dynamic Boost takes from.
3. CPU: 40% structural x 8,000 cycles = 9.6 G cycles per generation, 2.4 core-seconds per 1.5 s, 1.6 cores busy. At 5 GHz and 1.35 V about 8 to 12 W; at 3.4 GHz about 4 to 6 W. Dynamic Boost's 15 W transfer is not a step at the first busy core; at 4 to 6 W the RTX loses a fraction of the 15 W, my estimate 4 to 6% of rate under the cap, zero above it.

What I concede: the structural operators. My P2 put all 64 on the Radeon; that was the highest engineering cost for the middle watt cost, and a third implementation of every emitter is exactly the seam the chair named. They should stay on the CPU in cpu's form, sorted by operator (cpu's P6), with the clock capped when the owner allows it. If they ever leave the CPU, the RTX beats the Radeon on both axes.

What I keep, narrowed: the Radeon for packing and the parametric 60% (igpu's P3 and P4), and only conditionally. The condition is the power table: if a Radeon burner at 1,100 MHz beside p2_speed costs the RTX under 3% while 8 CPU threads cost 21%, the Radeon is the cheaper home for that work in watts. If data's recipe scheme on the RTX costs 1 to 3% of rate, the two are within noise of each other in watts, and then one implementation on the RTX beats two, and I concede that too. So the Radeon breeding claim lives or dies on two rows of D4 and on whether a register-resident RTX kernel has headroom under the cap (D1). What would break it outright: the burner row showing more than 3%, or the Radeon pack failing byte-identity with the CPU pack.

On cpu's "every 10 W the CPU takes costs the GPU 5 to 6%": the mechanism is a bounded 15 W transfer, so the cost must saturate. Falsified if the 16-thread row costs the RTX the same as the 8-thread row (both about 21%), which is what I expect; then the rule is "the first 15 W of CPU power cost 21% of GPU rate, the rest cost heat." Both of us should stop quoting a per-watt slope until rows 1 to 4 exist.

On data's "1 to 3% of RTX time": agreed, and under the cap it is 1 to 3% of rate, not of idle time; there is no idle time at the cap. That is the correct way to cost every device-side helper (unpack, breed, descriptor, prefilter): add up their instructions and take that share of the rate.

On ga and ml's steps lever: from my seat it is the only lever that is worth its full ratio with no uncertainty, because a step avoided is a joule avoided whatever the mix. It should be counted once, on the 480 base, and it multiplies with the kernel lever.

## D4. The power table: which rows, and what each needs

Nothing clean this round; the reasons are in D1. The table, each row 60 s unless stated, all logged by scratchpad/os/sample.sh plus `nvidia-smi -q -d POWER`:

- Row 0: p2_speed alone, CPU quiet. Exclusive lock, owner's game paused, no agent builds. The baseline every other row is divided by.
- Rows 1 to 4: p2_speed plus 2, 4, 8, 16 busy CPU threads (a synthetic AVX-512 FMA loop I will write in C; it is in the scratchpad style, no project code). Exclusive lock. This gives the shape of the Dynamic Boost transfer and falsifies or confirms the saturation claim above.
- Rows 5 and 6: 8 threads with scaling_max_freq at 3.4 GHz, and with EPP balance_performance. Owner runs the sysfs writes. This is the row cpu's P4 and my P3 both need; I own the measurement and cpu owns the interpretation.
- Rows 7 to 9: a Radeon compute burner at 800, 1,100 and 2,700 MHz (igpu's loop; DPM forcing is root). Exclusive lock plus the owner for the DPM level. Row 8 is the one that decides my narrowed Radeon claim.
- Row 10: row 8 plus 4 busy CPU threads, the realistic mixed host.
- Rows 11 to 13: the three synthetic kernels of D1 for nJ per warp instruction and the clock at the cap.
- Rows 14 to 16: p2_speed at 4, 2 and 1 blocks per SM.
- Row 17: memory clock locked low (`nvidia-smi -lmc`, root), p2_speed alone.
- Row 18: platform profile max-power, p2_speed alone, reading the live limit.
- Row 19: p2_speed looped for 10 minutes for the thermal steady state, the "sustained" row. Exclusive lock for 10 minutes, so outside a 5-minute pause: it needs the owner away from the game or a longer pause.

Total about 30 minutes of exclusive GPU across rows 0 to 18, which fits in six 5-minute pauses with 2-minute gaps, or one session while the owner is away. Under the shared lock while the owner's game runs, the only honest row is the game's own sustained trace, which I tried to take this round and could not because the process that was on the GPU was not the owner's game. I will take it the next time the owner's game is running for more than five minutes, read-only, and it is the row that tells us whether "sustained" today means the power cap, the thermal limit, or neither.

## D7. Preallocation and huge pages, with cpu

Agreed in one line: preallocate every block's arrays once and reuse them forever, so the hot path allocates nothing and page faults go from 0.65 per creature to zero; huge pages are one optional madvise(MADV_HUGEPAGE) on those arrays, worth under 1% by today's 2.2 M TLB misses in 92 G cycles, to be kept only if the dTLB-miss counter says so. No disagreement.

## What would break the proposals I still back

- P1 (host under 4 us of thread time per creature, zero-copy records and results, blocking waits): breaks if the kernel cannot read a creature record from host memory at take-up without stalling the lane group for more than a few microseconds; the amortization over 300 to 1,200 steps assumes one read per creature. gpu should say whether the take-up branch can prefetch the next record.
- P3 (owner-side power settings): each row of D4 is its own test; a row that moves the rate by under 3% is dropped.
- P4 (60 FPS scheduling): breaks if the frame-time histogram shows the drops are on the Radeon render side rather than CPU starvation; then the fix belongs to the UI domain, not the scheduler.
- The 3 T budget: superseded by rows 11 to 13; until then it is the lower bound and 7.6 T the upper.
