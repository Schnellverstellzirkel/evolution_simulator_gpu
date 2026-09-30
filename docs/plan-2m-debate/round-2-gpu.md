# Round 2, GPU domain: answers, revisions, attacks

## D1. The kernel

### My count of the maximal-coordinate step (physics section 3)

Same body as physics: 6 nodes, 5 rods, 15 muscles, 4 contacts, one lane per creature, one substep.

| phase | physics | mine | why mine differs |
|---|---:|---:|---|
| external forces | 150 | 150 | agree |
| muscles, 15 | 1,200 | 1,275 | the 4 force accumulations per muscle go to nodes chosen by a runtime index, so they are shared-memory read-modify-writes (3 MIO ops per end), not register adds |
| rod matrix, LDL, right-hand side, solve, velocity update | 210 | 275 | physics left out the velocity-product term of the rod rows (15 per rod) and the forward and back solve (10 per rod) |
| contact detection | 60 | 60 | flat ground; terrain adds about 100 |
| 8 Delassus rows and the 8x8 assembly | 500 | 530 | agree |
| direct 8x8 solve with one re-solve | 300 | 500 | LDL of 8x8 is about 170, two solves 130, the re-solve repeats the factor after a row drops (170), active-set checks 40 |
| response, integrate, drift projection, ledgers, metrics | 430 | 430 | agree |
| per step, 1 substep | 2,900 | 3,220 | |
| per step, 2 substeps | 5,600 | 6,290 | metrics once |

Against my reduced-coordinate count of 6,650 at 2 substeps, the formulation saves 5% by my count and 15% by physics's. The 2x in physics's table is the substep, not the formulation. The formulation is not an instruction lever.

I would build it anyway, for the register file, not the instruction count. Reduced coordinates per lane carry 9 floats of articulated inertia and bias per bone across the backward pass (72 registers at 8 nodes) plus the walkers' rows; that is what pushed my W = 1 count to 200 and forced W = 2. Maximal coordinates carry 4 floats of state per node and 3 per rod of factor. That makes W = 1 fit at 8 nodes, which removes the 144 shuffles per warp-substep of W = 2 and halves the divergence waste (two lanes of one creature never wait on each other). I price W = 1 over W = 2 at 1.15x in executed instructions. So: build maximal coordinates per lane with the direct solve, keep the reduced-coordinate lane-group kernel as the fallback for bodies above the per-lane class. The drift projection and the joint limits as rows are physics's to prove stable.

Attack on physics's 90 to 100 registers at 6 nodes. The contact phase holds the symmetric 8x8 Delassus matrix (36), a response vector (12), the rod factor (15), state (24) and node force accumulators (12): about 100 live before temporaries, so about 130 at 6 nodes and 150 to 160 at 9. Then the muscle state (energy, rhythm offset, last waveform) is 45 floats at 15 muscles: 175 registers, 8 to 10 warps per SM, or 180 B of shared per lane against the chair's 195 B budget for everything at 16 warps, with the anchor node table (96 B) and the force accumulators (48 B) also wanting to be there. Physics's 20 warps need the whole per-lane shared footprint under 160 B, which I do not see. The fix: drop the last-waveform cache (recompute, one cos; the cache measured slower anyway) and pack energy and offset as one 16-bit pair per muscle (15 registers). Then W = 1 is about 145 registers at 6 nodes, 12 warps per SM, enough if the 15 independent muscles give the ILP physics claims. Physics: show the register table per phase with the muscle state placed.

### The ceiling at os's 3 T budget

Executed thread-slots per creature-step (idle lanes counted, the unit os wants): today about 32,000 (1.45 T / 45 M). My W = 2 reduced design: 275 warp instructions x 32 = 8,800. W = 1 maximal: about 240 x 32 = 7,700. At os's 3 T sustained: 340 M and 390 M creature-steps/s. At 2.0 GHz the issue peak is 192 G warp instructions/s, and hitting 630 M with W = 2 needs 173 G/s, 90% issue. Not reachable. So under os's budget my kernel number is 340 to 390 M, 7.5 to 8.7x today, and the issue rate stops mattering once the cap binds.

I contest the 3 T figure, not the method. os's 38 pJ per executed thread-slot was fitted on a kernel whose issued instructions are a third shuffles, shared-memory ops and barriers, which move data across the SM's crossbar and cost 2 to 3x an FMA that reads and writes the register file, and whose lanes are 70% idle, so the per-warp-instruction overhead (fetch, decode, operand collect) is spread over few useful slots. An FMA-dense kernel at 60% issue with full lanes is a different curve. My estimate of the marginal energy is 12 to 20 pJ per active lane-instruction at 1.0 V, which puts the cap-limited clock at 2.1 to 2.3 GHz and the sustained budget at 4.5 to 5.5 T. Under that: W = 2 at 510 to 620 M, W = 1 maximal at 580 to 710 M. The single measurement that settles it is os's synthetic FMA kernel at 100% issue for 30 s with power, clocks.sm and the limiter reason logged; the clock it holds at the cap is the number. I will take whichever the run gives. Until then my kernel range is 340 to 620 M creature-steps/s at the mean body, 7.5 to 14x.

### The 13% in p2_speed: pack, not tail

p2_speed times `engine.submit` through the engine thread, which runs `pack` (0.32 s per 262k on 4 threads, 00-facts.md) before the upload, and `gpu_seconds` is the event pair around the kernels only (cuda_engine.rs, the `start` event is recorded after the HtoD copies). Wall 4.38 s, GPU 3.81 s: 0.57 s of loss, of which pack is about 0.32 s, the 340 MB upload and its host memcpy about 0.08 s, the readback a few ms, and the tail the rest, 0.1 to 0.15 s.

The tail itself: a block keeps its SM slot until its last group finishes, so after the counter runs out the drain lasts as long as the slowest survivor's remaining trial. A lone warp steps at 12 to 17 us against 34 us under load, so 1,200 steps is 15 to 20 ms per block, and blocks exhaust at different moments, so the SM population falls over about 100 ms. That matches the residual. In the game, freed block slots take queued blocks of the next launch at once, so the tail is hidden whenever a wave is queued behind the running one. So the 1.15x belongs to data's pack removal, already in data's table, not to wave scheduling.

For ga's 49k waves: the drain is 20 to 100 ms whatever the wave size, so a 0.3 s wave pays 10 to 30% unless a wave is queued behind it, which the ring guarantees. The per-lane design raises the floor: 9,216 resident lanes per GPU need at least 4 creatures each, so waves below about 40k cannot amortize the drain.

### The 9-node, 30 to 50 muscle population

At 9 nodes and 40 muscles the reduced-coordinate W = 2 step is: muscles 40 x 90 = 3,600, tree passes on 8 bones 1,000, contacts with a deeper walk 1,500, ledgers 200, per substep 6,300, per step 12,800 plus metrics. Per lane 6,400 plus 15% divergence over 16 creatures: 460 warp instructions per creature-step, 14,700 executed slots. One number: 200 M creature-steps/s at os's 3 T, 340 M at my 5 T. Registers about 145 at W = 2, 12 warps per SM. Muscles are 56% of that step. At 480 steps that is 420k to 700k creatures/s at generation 50 before any step cut. The kernel does not hold 2M/s at the p90 body; muscle count does. The GA domain owns the number that decides sustained rate: muscles per creature at generation 50.

## D2. The 10 s rung: withdrawn

The number is hpc.md section 5.6 ("2026-09-27: a second rung at 10 s keeping half the survivors kept every creature of the final top 1% and 86% of the top 10%, for about 1.3x"), an offline retention count on 60 s trials. docs/rejected-ideas.md records the search result: rungs at 10, 15, 20 or 30 s lost QD; only 30 s keeping 60% held it, for 9%. Retention of the top 1% did not predict the search outcome, so the number is no evidence. Withdrawn. Steps per creature is one lever owned by ga and ml; my "screen at 3 s" was the same lever under another name and leaves my table. The kernel-side cost of whatever they choose is noise: a per-creature bar index with one L2 table load per rung is about 5 instructions, and sampling the distance at two more steps about 10. The lesson for ga and ml: gate on the search A/B, not on retention.

## D3. Breeding: I concede the design to data, and keep the clock tax number

My "pack kernel and device-resident ring at 500 B" is data's P1 and P2 done worse; recipes at take-up are the right form. GPU costs: unpack at take-up about 2,000 instructions per creature against 3.2 M for its trial (0.06%); a parametric child about 5,000 (0.15%); a structural child sorted by operator about 50,000 at a 35% share (0.5%). I take 1 to 2% of RTX time, under data's 1 to 3%. The clock tax of the alternatives: CPU breeding on 8 threads costs 21% of GPU rate (measured); the Radeon at 3 to 8 W takes that much of the 15 W the package would otherwise leave to Dynamic Boost, so 3 to 8%. Ranking in watts: RTX 1 to 2%, Radeon 3 to 8%, CPU 15 to 21%. The trade I back: parametric emitters as recipes on the RTX now, one implementation and no CPU twin; structural children by value from the CPU on at most 2 threads capped at 3.3 GHz; the structural port only if D5's size table says the CPU cannot keep up. One wording of data's breaks: the archive slab is not L2-resident beside the kernel. At 12 warps per SM of per-lane creatures, 9,216 creatures x 1.3 KB = 12 MB of constants stream through L2 every substep; with a 19 MB slab that is 31 MB in a 32 MB L2, which thrashes. Either size the slab at 8 MB (only the parents a block's recipes reference, prefetched per block) or accept DRAM reads at take-up: 2 M x 2.5 KB = 5 GB/s, 2% of DRAM bandwidth, fine. The plan holds; the word "resident" does not.

On the RNG: with no CPU twin, bit equality is moot. Philox 4x32-10 with Box-Muller through `__sincosf` and `__logf` is about 60 instructions per pair of gaussians and deterministic on one GPU. If ga wants streams equal to a CPU diagnostic, the fixed-point sum of 12 uniforms costs about 150 integer instructions per gaussian, under 0.5% of a trial either way.

## D4. Ranked by instructions removed, and the power-bound W question

Executed thread-slots per creature-step, today 32,000. A slot removed is a joule removed; a stall removed is not, once the cap binds. Idle lanes are partly both: a predicated-off lane skips the datapath but the warp instruction still pays fetch, decode, operand collection and issue, which is 40 to 50% of the energy of a full-lane instruction. So I count an idle-lane removal at 0.6 of a real instruction removal.

| lever | executed slots after | slots removed | joule factor | owner |
|---|---:|---:|---:|---|
| 1 substep with physics's ledgers (physics P1) | half of whatever the kernel does | 2x | 1.9x | physics, owner |
| P1 layout, W = 2 reduced | 8,800 | 3.6x (mostly idle lanes and MIO) | about 2.2x | no |
| P1 layout, W = 1 maximal | 7,700 | 4.2x | about 2.5x | physics's formulation, owner |
| steps per creature (ga and ml) | same per step, fewer steps | 1.3 to 1.6x | 1.3 to 1.6x | owner |
| P2 physics cuts (waveform once per step, ledger in flight, lagged matrix, 2 sweeps) | minus 17% | 1.2x | 1.2x | physics |
| maximal coordinates at 2 substeps, by itself | minus 5 to 15% | 1.1x | 1.1x | physics |
| P5 tuning (launch bounds, block shape, SFU) | 0 | 0 | 1.0x at the cap, up to 1.2x while clock-bound | no |

P5 is demoted to "free until the cap, then nothing". P2 moves above it. The layout stays the largest lever the GPU domain owns, and it is an instruction lever, not an occupancy lever: the idle-lane slots are executed today.

Does W = 2 change when power-bound? Yes, toward W = 1. At the cap only executed slots count, and W = 1 executes fewer (no shuffles, no paired-lane wait). W = 1 at 255 registers and 8 warps per SM is fine under the cap if it reaches the issue rate the cap allows, 50% with full lanes at 2.0 GHz and 3 T. Physics's ILP claim (3 in the muscle loop, 4 to 8 in the contact phase) makes that plausible and unmeasured. The stub kernel physics asked for (state, shared tables, a 5-rod LDL loop in isolation) measures it, and I will build it as the track's first gate; it also answers the register question in D1.

## D5. Sustained rate at the p90 body

Given in D1: 200 to 340 M creature-steps/s at 9 nodes and 40 muscles, against 340 to 620 M at the mean body. The step cost is linear in muscles above about 10 of them (90 per muscle-substep against a body-fixed 2,900 per substep at 9 nodes), so the rate at generation 50 is set by the muscle count. Beyond the GA's size table I need the muscle-count histogram per generation: the per-lane muscle loop runs the warp's maximum, so the packer must sort by exact muscle count (today it sorts by rounds, muscle count in eights), and no save yet shows whether muscle mass holds muscles below 30 after 30 generations on the new physics.

## What has to be true, and what breaks it

os's 3 T: true if the FMA run at the cap holds under 2.1 GHz; broken if it holds 2.3 GHz or more.

Physics's 40 to 55% issue at 16 to 20 warps: true if at most 145 registers and 190 B of shared per lane at 6 nodes; broken by the muscle state in shared memory at 15 muscles.

Data's L2-resident slab: true if the whole resident set is under about 24 MB; broken by 12 warps per SM of per-lane creatures at 1.3 KB of constants each (D3).

ml's 40 extra bytes per result: nothing on the GPU (one write per 3.2 M instructions); the cost is the host's and the ring's.

My P1: broken by (a) a per-lane register table above 200 at 8 nodes with the muscle state in registers (6 warps per SM, no 50% issue); (b) muscle-count divergence above 30% in a warp after sorting; (c) muscle constants left at 64 B, which puts the L2 stream at 12 warps per SM above 1.5 TB/s. The stub kernel answers (a) first.
