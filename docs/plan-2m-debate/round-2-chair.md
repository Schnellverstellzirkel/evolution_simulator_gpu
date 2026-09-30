# Round 2, chair: cross-examination

Theme: attack each other's numbers and assumptions. For every proposal you attack, say what number would have to be true for it to work, and what would break it. For every proposal of your own that is attacked below, answer with a number or withdraw the claim. Write arch/round-2-<domain>.md. Keep it under 2,500 words. Plain prose.

Chair's reading of round 1, in one paragraph. The kernel is at 43 to 45M creature-steps/s. Everyone agrees the host chain walls at 300 to 500k creatures/s and must leave the per-creature path. Everyone agrees the Radeon must not score. Beyond that the papers disagree on the size of the largest lever by more than 2x, they count the same steps-per-creature saving under four different names, they put breeding on three different pieces of silicon, and nobody has the power table that three of them say decides their estimate. The product of the conservative ends is about 0.6 to 0.9M/s. The product of the optimistic ends is over 3M/s. Round 2 must narrow that.

Facts the chair verified in the code this round, which several papers got wrong:

- The CUDA kernel runs no settle phase. `Params.steps` is `duration * rate` (1,200) and `screen_step` is `tick - settle` (299). `SETTLE` only offsets recorded replay frames. So the baseline is 480 steps per creature before falls (0.8 x 300 + 0.2 x 1,200), not 550 to 580. ml's proposal 3 (1.15x from settle) is worth 0x and is withdrawn by the chair. ml's other estimates must be restated on the 480 base.
- p2_speed already reports 51.8M creature-steps per GPU-busy second against 45.1M per wall second (scratchpad/warp/final.txt). About 13% of wall time is upload and wave tail on a single 262k wave. That is a free 1.15x for the data-flow design and it is not in anyone's table.
- The old per-thread kernel (26.7M) ran a different step (one step with 2 planting rounds and 8 warm-started sweeps) from the lane-group kernel (two substeps of 4 cold sweeps). The 1.65x is not a clean layout comparison. gpu's claim that "lane groups won by removing spills, not parallelism" is plausible but unproven.

## D1. The kernel: layout, formulation and the instruction count (the lever worth 4x to 12x)

The numbers on the table:

- gpu: the ideal kernel of today's physics costs about 6,650 thread instructions per creature-step (3,250 per substep). Proposes W = 2 lanes per creature in reduced coordinates, node-count-specialized kernels, 16-bit packed muscle constants: 275 warp instructions per creature-step, 370 to 630M creature-steps/s, 6 to 10x. Registers at W = 2 about 110.
- physics: today's lane-group layout tops out at 170 to 240M (4 to 5x) whatever is done inside it. Per-lane (W = 1) maximal coordinates with a direct chordal tree solve and a direct 8x8 contact solve costs about 2,900 thread instructions per step at 1 substep, 5,600 at 2, and reduced coordinates per lane cost 6,400. Registers 90 to 100 at 6 nodes, 140 at 9, 180 at 13. Reaches 1.2G at 6 nodes, 570M at 9 nodes and 35 muscles.
- os: the sustained budget under the 100 W cap is about 3T thread instructions/s at 1.9 to 2.1 GHz, not 7.65T. Today the kernel draws 72 to 81 W at 19% issue and 2.49 GHz, about 38 pJ per issued thread instruction. A kernel at 60% issue "would want 240 W".

Questions:

- gpu: physics counts your W = 2 reduced-coordinate step at about 6,400 per creature-step per lane-equivalent and says the muscle-to-torque mapping and the 3x3 spatial inertias cost 800 more per substep than maximal coordinates. Answer with your own count of the maximal-coordinate step (physics section 3 table) and say whether you would build it. Also: your 6 to 10x assumes 55% issue at 2.49 GHz. Recompute at os's 3T sustained budget. What is your kernel number at 2.0 GHz and what issue rate does your design need to hit 630M?
- physics: your 2,900 per step has the muscle block at 1,200 for 15 muscles (80 per muscle per substep). gpu counts 90 per muscle. Fine. But your Delassus rows (500) and direct solve with one re-solve (300) assume 4 contacts always. What is the count when the warp's max contact count is 4 but the lane's own is 1 or 2, given every lane in a per-lane warp runs the warp's max? And your 90 to 100 registers at 6 nodes: gpu's count for W = 1 reduced coordinates is about 200. Show the register table per phase for maximal coordinates and say what lives in shared memory in the [index][lane] layout and how many bytes per lane that is (this decides occupancy: 100 KB shared per SM at 16 warps is 195 B per lane).
- physics: your proposal 1 (realized-work ledger, anchored friction, 1 substep, 1.8x) is a physics change needing the owner. The "1 substep plus 1 planting round" measurement gave 0.79 median ratio. What ratio do you predict for 1 substep with the ledger and anchored friction, and what is the single cheapest experiment that says whether 1.8x exists (you named one: evolve at 1 substep for 30 generations and re-test at 2 and 4). Is it runnable on the current kernel with a 50-line change?
- os: your 38 pJ per issued thread instruction is derived at 19% issue where a large share of the issued instructions are MIO (shuffles, shared, barriers) and idle-lane tree levels. An FMA-dense kernel has a different energy per instruction. State the uncertainty on the 3T figure and describe the calibration run exactly (which kernel, how long, what is logged), and say whether you can run it this round on the shared GPU lock without pausing the owner's game.
- gpu and physics, both: the 13% wall loss in p2_speed. Is it wave tail or upload? What is the tail of a 262k wave in ms when the last survivors run 900 more steps on a few warps, and does 8 streams already hide it in the game?
- gpu: bodies of 9 to 13 nodes. Your classes are 4, 6, 8, 12 with the lane-group kernel above. physics says per-lane serves at most 7 or 8 nodes. If the population sits at 9 nodes and 30 to 50 muscles by generation 30 (the old build did), what is your kernel's rate on that population? Give one number.

## D2. Steps per creature (the lever worth 1.3x to 1.8x, claimed under four names)

The same saving is on the table as: ml's calibrated early stops at 1, 2, 3 s; ga's cell-aware rung at 2.5 s; igpu's lenient rung at 2 s; gpu's "screen at 3 s instead of 5 s". These cannot multiply. Also on the table: ga's cell-aware bars (1.29x: 480 to 372), ml's elite-relative stops at 8 and 12 s (survivors cut to a mean of 9 s), physics's cheap-fidelity screen (1.25x), physics's flight steps at 1 substep (1.1 to 1.2x), and a second rung at 10 s that gpu quotes as "measured 1.3x with every top-1% kept" while docs/rejected-ideas.md says a second rung at 10, 15 and 20 s lost QD and only 30 s keeping 60% held it, for 9%.

Questions:

- ga and ml: produce one joint steps-per-creature schedule on the 480 base. Rung times, keep fractions, the predictor at each rung, and the resulting mean steps per creature in a fresh population and in a mature one. Each of you writes it in your own file; the chair will read both and note where you still differ. ga: attack ml's elite-relative stop with your own finding that "stop once it cannot beat its cell" was rejected because the cell is unknown. ml: attack ga's per-cell bar with the censoring problem you raised yourself (a learned or tabled screen trains on data it censored) and say how the audit lane fixes it for ga's version too.
- ml: restate every estimate on the 480 base and drop settle. Also: your checkpoint features need 40 more bytes per result. data's plan has results at 80 B and a device-side descriptor. Do you need the features on the host at all, or only the fit's summary statistics?
- gpu: the 10 s rung number. Cite where "1.3x with every top-1% kept" was measured, or withdraw it. The rejected-ideas file says the opposite.
- physics: your cheap-fidelity screen runs the first 5 s at 1 substep and restarts survivors at full fidelity from t = 0. ml rejected exactly this on the ground that exploits pass a coarse screen more often than honest movers (30 Hz elites kept 38%). Answer: is a 1-substep screen's ranking of honest movers good enough that the 20% kept still contains the top 1% at 60 Hz? What is the measurement, and can it run on the current kernel by scoring one 262k wave both ways?
- igpu: your 2 s rung at keep 50% on the RTX alone gives 1.23x. Same lever as ga's 2.5 s rung and ml's 2 s checkpoint. Concede it to ga and ml or show why yours differs.

## D3. Where breeding, packing and the archive prefilter go (three silicon choices for the same work)

- data: recipes of 16 to 32 B in the ring; the RTX materializes the child at take-up from an L2-resident archive slab; structural operators on the RTX sorted by operator; a device-side descriptor and candidate filter; the CPU commits candidates only. Cost to the RTX: 1 to 3% of its time. VRAM under 1 GB.
- cpu: keep breeding on the CPU with a vector path (1,500 cycles per parametric child) and a scalar path over fixed arrays (8,000 per structural child), 4,700 weighted, 3.5 core-seconds per 3M; then ship plans for the parametric 60% to the RTX. Archive stays on the CPU at 30 to 60 ms per block.
- igpu and os: the Radeon packs and breeds parametric children and immigrants into a shared pinned ring imported through VK_EXT_external_memory_host; the CPU keeps structural operators. Radeon at 800 to 1,100 MHz costs 3 to 8 W and returns the CPU's 15 W of Dynamic Boost to the RTX.
- gpu: a pack kernel on the RTX and a device-resident ring of 786k genomes at 500 B (400 MB).

The chair's concern: this is one job proposed on three devices, and two of the plans keep a CPU twin for archive regeneration or failover, so the game would carry two or three implementations of every emitter. Every extra implementation is a determinism seam and a maintenance cost the owner did not ask for.

Questions:

- data: os says every RTX instruction is a joule under the cap, so "1 to 3% of RTX time" is 1 to 3% of rate, which is fine, but the structural operator port is "weeks" and 7,500 lines of anatomy code. Give the port's size in lines and the rate you get without it (recipes for parametric children only, structural by value from the CPU). cpu says the CPU can do all breeding at 3.5 core-seconds per 3M after its rewrite. If that is true, what does your RTX breeding add beyond the clock tax? Answer with the clock tax you assume.
- cpu: os measured 80 us of thread time per creature (350k cycles) on a one-generation run. You derived 73,000 cycles per creature from the facts. That is 5x apart. Reconcile it (polling and rayon spin, the older kernel's segments, page faults) and say what the CPU chain's real ceiling is today. Then: your 4,700-cycle breeding is a rewrite of evolution.rs and anatomy over fixed arrays with a counter-based RNG. How many lines, and does it change any operator's output? The owner's rule is no operator is removed.
- igpu: data says the Radeon cannot win the trade because breeding on the RTX costs 1 to 3% of its time. os says the Radeon wins because it keeps the RTX's 15 W. Both are power claims and neither is measured. Your P1 power split is the measurement. Can you run the Radeon burner part this round (it needs no code beyond a small Vulkan compute loop and nvidia-smi sampling, and the RTX part can be the current game's p2_speed under the shared lock)? If not, say what you need. Also: you say the amdgpu lockup timeout is 2,000 ms on this kernel; os says 10 s. One of you read it wrong. Settle it with the modinfo output.
- os: your P2 puts all 64 structural operators in a compute shader on the Radeon. data puts them on the RTX. cpu keeps them on the CPU. Rank the three by engineering cost and by watts, and say which you would concede.
- data and cpu: the archive prefilter. rejected-ideas rejected a device-side contender filter at a 0.5 s ceiling on a 10 s generation. data says the reason changed. cpu says the archive fits on the CPU in 30 to 60 ms per block. At 2M/s a 196k block is 100 ms. Is the CPU number for a mature archive with refresh_behavior_scores, or an early one? Who measures it (EVOLUTION_PROFILE_BREED on a mature save prints the sections)?
- Everyone who proposes a counter-based RNG (cpu, igpu, data): one rule for the gaussian. cpu wants a fixed polynomial with pinned FMA, igpu wants integer-exact (fixed-point sum of uniforms or an integer ziggurat), data accepts non-bit-equality between the CPU breeder and the device. ga must say whether a changed random stream is "a new seed" and whether a sum-of-uniforms gaussian changes the search. Each of the three: state your rule and its cost per gaussian.

## D4. The power budget (nobody has the table)

Three papers say the same table decides their estimate: p2_speed rate, clocks.sm and power.draw with 0, 2, 4, 8 and 16 busy CPU threads, and with a Radeon burner at each DPM level. The one existing point is 174k to 137k with 8 threads (21%). os adds the limiter history: 1,419 s at the software power cap and 1,321 s of thermal slowdown in five days, so the owner's long sessions run at the limiters and "sustained" means at the cap.

Questions:

- os: you own this table. Say which rows you can produce this round under the shared GPU lock without pausing the owner's game (the game holds the GPU at 162 MiB right now, which is small, so a p2_speed beside it is allowed by the owner's rule of 2026-09-29). If a row needs the exclusive lock and a pause, say so and the chair will schedule it after the debate.
- cpu: your P4 (cap the CPU at 3.4 GHz for breeding) and os's P3 overlap. One of you owns it. Also your claim "every 10 W the CPU takes costs the GPU 5 to 6%" is derived from one point. Say what would falsify it.
- gpu: if the sustained clock is 2.0 GHz and the FMA-dense kernel is power-bound, does your W = 2 design change? An instruction removed is a joule removed; a stall removed is not. Rank your P1, P2 and P5 by instructions removed rather than by issue efficiency gained.

## D5. Sustained rate across generations as bodies grow

The owner's ask is 2M/s sustained for minutes across generations. The old build fell from 110k/s to 27 to 30k/s by generation 30 as bodies grew to 9 nodes and 30 to 50 muscles. physics says muscles are 40% of the step at 15 and 60% at 35, and that 9 to 13 node bodies cost 2 to 3x per step in any layout. Muscle mass is in. Nobody has the size distribution over generations on the new physics.

Questions:

- ga: there are saves at generation 10 (scratchpad/g10.evo), a night run (night.evo), and save.evo and save42.evo from today. examples/size_report and body_stats exist. Report nodes and muscles per creature (mean and p90) by generation from whatever the saves and their history hold, and say what holds bodies below 9 nodes now, if anything. If nothing does, say what search-side rule would (the owner wants no fitness terms; environment effects and physics costs are allowed, and muscle mass is the precedent).
- physics and gpu: give the rate of your design at the p90 body, not the mean body. The plan will be judged at generation 50, not generation 3.

## D6. The Radeon for physics: close it or keep it

igpu: at most 1.0 to 1.1x on top of a rung the RTX runs alone, and shrinking as the RTX gets faster. data: no per-creature use. os: a losing trade in watts. gpu: not before the CUDA kernel is within 1.5x of the goal. The chair proposes to close Radeon physics unless igpu shows a number above 1.15x sustained with the power measurement behind it. igpu: accept or fight, with the number.

## D7. Small items each owner must answer

- ml: proposal 6 (learned routing of confirmations) spends GPU on the spirit, not the rate. Keep it as a track or park it. Say which and why in three sentences.
- ga: P3 (16 blocks of 49k, CMA_LIMIT to 1,024). data wants blocks of 200 ms and 10 in flight at 2M/s, which at today's rate is 30k creatures per block. Are you and data proposing the same ring? Say what block size the emitters want and let data say what the GPU tolerates.
- data: your P4 sizes the ring in seconds and blocks in milliseconds. A world change throws away every block in flight. At 2 s of ring that is 4M creatures at 2M/s. Is a 2 s ring right, or is 0.5 s enough once the host chain is off the critical path?
- cpu and os: THP and page faults. os measured 0.65 faults per creature and 2.2M TLB misses in 92G cycles (under 1%). cpu wants madvise huge pages. Agree on one line: preallocate and reuse, huge pages optional. Say if you disagree.

Write your round-2 file now. Name what would break each proposal you still back.
