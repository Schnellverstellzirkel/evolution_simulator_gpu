# Round 2, iGPU domain: answers, one measurement, two concessions

Author: the iGPU expert. Date 2026-09-30, 18:40 to 19:05. New measurements are in scratchpad/igpu/ (burner.c, row.sh, sample.sh, trace-*.csv, rate-*.txt).

## D3, the lockup timeout: 2,000 ms on this kernel

`modinfo amdgpu` on the running 7.0.0-34-generic kernel prints, verbatim:

    lockup_timeout:GPU lockup timeout in ms (default: 2000. 0: keep default value. negative: infinity timeout), format: [single value for all] or [GFX,Compute,SDMA,Video]. (string)

The module parameter is unset (`/sys/module/amdgpu/parameters/lockup_timeout` is empty). The 10 s figure os quotes is the older kernel's default (10 s graphics, 60 s compute). This kernel changed it to 2 s for every ring. The two hangs in the journal (2026-09-25 17:55 and 2026-09-26 13:06, both `ring comp_1.1.1 timeout`, both from the game's pid) are consistent with a dispatch crossing 2 s, and the first was recovered by a queue reset while the second escalated to a MODE2 reset with VRAM loss. So the dispatch budget is 2 s, not 10 s, and the design rule stays at 50 ms with a self-halving watchdog. os and I agree on every other rule (LOW-priority compute queue, at most two dispatches in flight, a soak gate before any port).

## D3, the Radeon burner half of the power split: measured this round

Setup. A GLES 3.1 compute burner on the Radeon through Mesa's surfaceless EGL platform (forced to the Mesa vendor; glvnd picks NVIDIA by default, which is a trap for anyone writing a Radeon helper). Dependent FMA chains, 1M invocations per dispatch, the loop count tuned so a dispatch takes about 9 ms, `glFinish` after every one, so the longest submission ever sent was 9.3 ms. The RTX side is `p2_speed` from the warp worktree on the same 262k dump as scratchpad/warp/final.txt, 3 timed passes, under the shared GPU lock. The sampler logs RTX SM clock and power, APU package power (PPT, hwmon amdgpu), the Radeon clock and its busy percent at 4 Hz. The CPU-burner rows are not here: the sandbox refused the spin loops, and they belong to os's table anyway. The chair should schedule them with os.

Radeon alone, 3 s warm-up run: 1,036 dispatches, p50 2.78 ms, p99 7.20 ms, max 9.27 ms, 1.27 T FMA/s (2.5 TFLOPS) at 100% busy. That is the Radeon's real ALU rate on dependent chains: 17% of the RTX's 15 TFLOPS peak, which matches the 0.2 to 0.25 ratio from round 1.

The rows beside p2_speed did not produce a usable number this round, and I say so rather than quote them. Two reasons, both in the traces. First, the gpu domain held the exclusive lock for a p2_speed A/B during my window, and my shared-lock rows ran with a second p2_speed on the RTX: the "no load" row measured 28 to 36M creature-steps/s at 57 W and 2,478 MHz, against the 45M reference, which is contention, not power. Second, the burner launched from the row script did not run (Radeon busy 0 to 18% in the traces, 800 MHz), while the same binary from my shell ran at 100%. So the rows measured nothing about the Radeon. What the round did establish: the burner is ready, it runs at 2.5 TFLOPS with no submission over 10 ms, and `journalctl -k` shows no amdgpu timeout or reset during any of it. The row itself takes 3 minutes of exclusive GPU time (no load, Radeon at 100% duty, Radeon at 30% duty, one p2_speed of 262k x 3 each, the sampler beside it), and I ask the chair to schedule it with os's CPU rows after the debate under the exclusive lock, with the burner started from a terminal. Until then my round 1 estimate stands as an estimate: 3 to 8 W at breeding duty, 20 to 30 W flat out.

## D3, who breeds: what number has to be true for each plan

The chair is right that three plans for one job is two too many. Here is the arithmetic that picks between them, with the number each needs.

- data's recipes on the RTX: needs "1 to 3% of RTX time" to hold at the p90 body and with the structural port done, or the CPU keeps structural children and the RTX only materializes parametric ones. If it holds, packing disappears entirely, because the lane group unpacks the child it just materialized. That beats moving the pack anywhere, including to the Radeon. What breaks it: the structural port (7,500 lines of anatomy, weeks), and a device archive slab that must be versioned for determinism across the breed-to-take-up lag.
- cpu's rewrite: needs 3.5 core-seconds per 3M to hold on a mature population. At 2M/s that is 3.5 core-seconds per 1.5 s, 2.3 cores busy all the time, plus packing unless data's take-up unpack lands. What breaks it: the Dynamic Boost tax of those cores, which os's CPU rows will give (the one point is 8 threads for 21%).
- my Radeon plan: needs the Radeon's watts at breeding duty to cost the RTX less than the CPU cores it replaces would, and the double pinning (amdgpu userptr plus cuMemHostRegister on the same pages) to work. The first is now partly measured above. What breaks it: data's take-up unpack, which makes the pack moot, and a CPU chain that fits in 2 to 3 cores.

My revised position. If data's parametric recipes land (P2 in their file), I withdraw the Radeon pack and the Radeon parametric breeder: there is nothing left per creature for the Radeon to do that the RTX does not do in its own registers at take-up, and a third implementation of every emitter is a seam the owner did not ask for. The Radeon plan stays on the table only as the fallback for the case where the RTX port is not scheduled and the CPU chain, after cpu's rewrite, still needs more than about 3 busy cores at 2M/s. The order that decides this: os's CPU rows, then cpu's rewrite measured with `BREED_NANOS` on the generation-70 save. I ask the chair to record it that way.

What survives from my round 1 regardless: the safety rules for any Radeon use (the UI already runs there, and a Radeon-side archive map or thumbnail renderer is UI work), the zero-copy host ring (`VK_EXT_external_memory_host` on the Radeon side is only needed if the Radeon computes; the RTX side, pinned host memory read at trial start, is os's P1 and data's design, and I back it), and the measured fact that a bounded-dispatch Radeon workload runs beside the desktop without a journal entry.

## D3, the gaussian rule

I adopt cpu's rule and drop my integer-exact one. A fixed polynomial for the inverse normal CDF (one uniform in, about 20 multiply-adds out) evaluated with explicit fused multiply-adds on both sides: `f32::mul_add` on the CPU, `fma()` in GLSL with `precise` (ACO maps it to `v_fma_f32` and contracts nothing else), or the CUDA `fmaf`. Cost per gaussian: about 20 FMAs and one integer draw, so under 10 ns scalar on the CPU and vectorizable, against 12 draws and 12 adds for the sum of uniforms and a log, a sqrt and a cos for Box-Muller. Bit equality between the CPU breeder and any device breeder is then a test, not a hope. Cost to the search: the random stream changes once, which ga must call "a new seed" or not; the distribution is the normal to within the polynomial's error (about 1e-6 relative in the body, with a tail cut around 6 to 7 sigma from the uniform's resolution). I do not expect that to change the search and I would not spend a 10-seed A/B on it.

## D2, the 2 s rung: conceded

My 2 s rung at keep 50% and ga's 2.5 s rung and ml's 2 s checkpoint are the same lever, a rung before the 5 s screen, and my 1.23x number is one point on ga's schedule (150 steps, keep 0.5) and inside ml's calibrated stop. I withdraw it as an independent lever. The only thing that differed was the device: I sized it as work the Radeon could carry, and D6 closes that. ga and ml own the schedule; my only request is that the joint schedule state the mean steps per creature on the 480 base for a mature population, because that number sets the Radeon's, and everyone else's, denominator.

## D6, Radeon physics: closed

I accept the close. My own number is at most 1.0 to 1.1x over a rung the RTX runs alone, and the burner row above is what a Radeon physics engine would cost the RTX, because physics would run the Radeon flat out at 2.7 GHz, not at the 800 MHz breeding duty. I cannot show 1.15x sustained; the arithmetic says the opposite, and it gets worse as the CUDA kernel improves. Radeon physics is off the plan. If anyone reopens it, the entry condition is the chair's: a measured 1.15x end to end with the power trace beside it, on a kernel that is already within 1.5x of the goal.

## What I attack elsewhere

- os, section 2: "every watt the Radeon burns is a watt Dynamic Boost cannot give the RTX". The measured row says how much of that is true at 100% duty; at breeding duty the Radeon's draw scales with busy percent, so the claim needs the duty in it. The row with 30% duty is the closer proxy for breeding.
- data, P7: "2 to 5 GB/s across host memory, a third of the PCIe budget" is the cost of any host-resident ring, including the one data's own P1 keeps for structural children by value (1.7 GB/s in their table). The bandwidth is not the argument against the Radeon; the maintenance seam is, and I have conceded to it above.
- gpu: "the Radeon could add 15% at the end if the WGSL kernel is kept alive". No. The WGSL kernel is the per-thread design the lane-group kernel replaced, it does not run the current physics, and keeping a second physics alive for 10% is the cost the owner's one-physics rule exists to prevent.
- cpu and os on the CPU rows: the one point (8 threads, 21%) cannot be extrapolated linearly to 2 or 3 threads. Zen 4 cores at EPP performance boost to 5 GHz on any runnable thread, so 2 busy threads can draw 15 to 20 W of package power, most of the Dynamic Boost swing. The rows at 2 and 4 threads are the ones that matter for cpu's plan, and a cpufreq cap on the breeding cores (os's P3, cpu's P4) may be worth more than which silicon breeds.

## What would break what I still back

- The safety rules: a driver where LOW priority is not honoured, or a compositor that itself submits long compute (GNOME does not). The 10 minute soak with frame times is the test.
- The zero-copy host ring on the RTX side: PCIe read latency at trial start stalling lane groups; os and data have the same design and the measurement is `p2_speed` with records read from registered host memory against staged uploads.
- The gaussian rule: a shader compiler that contracts across an explicit `fma()` despite `precise`. The bit-equality test on 1M children catches it on day one.
