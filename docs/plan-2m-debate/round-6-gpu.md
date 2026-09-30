# Round 6, GPU domain: where the stub's instructions go, and what removes them

I read shaders/lane_stub.cu, examples/lane_stub.rs, docs/plan-2m-measurements.md and the Power rows. Every count below is from the stub's source against its measured breakdown, per lane per substep at W = 2 (16 creatures per warp), so 1,160 warp instructions per creature-step is 18,560 instructions per lane per step, about 9,300 per substep. My round-3 estimate of 2,100 per lane per substep was wrong by 4.4x, and the reason is in the code, not the layout.

## Q3. The floor, counted from the stub

| section, per lane per substep | stub | what it is | inherent | per-plan baked |
|---|---:|---|---:|---:|
| drag, damping (4 rods, predicated on valid, pivot and grandparent from the topology word) | 240 | forces | 150 | 150 |
| muscles, 10 per lane (2 L2 loads, 4 LDS.128, 8 FRC read-modify-writes, state word in L2, unpacks) | 1,480 | model | 700 | 700 |
| free velocities, node tables, statics | 120 | tables | 60 | 60 |
| rod directions, rows | 110 | rod rows | 100 | 100 |
| factor (parent_mask, predicated q-scans, Schur by bit test) | 300 | LDL | 60 | 60 |
| contact selection, slot values, incidence word sgw | 200 | contacts | 150 | 150 |
| tree solves: batch A (6 rows), batch B (5), z1 (2): per row per rod a parent_mask and 8 predicated selects | 2,700 | tree solve | 630 | 630 |
| Delassus assembly: RCO bit tests per (rod, contact), 36 entries, rf recomputed per row | 1,700 | assembly | 360 | 360 |
| active-set solve, 3 rounds at the warp max, 8x8 LDL with 36 WIJ loads per round, L[8][8] in registers (the 614 B spill) | 1,440 | solve | 320 (PGS) or 480 (1 round) | 480 |
| stash, impulses, ledgers, Euler, anchors | 460 | ledgers | 310 | 310 |
| per substep | about 9,000 | | 2,840 | 3,000 |
| per step at 2 substeps, plus waveform, projection, metrics once per step (stub about 1,100; baked 650) | 18,500 | | 6,300 | 6,650 |
| warp instructions per creature-step | 1,160 | | 395 | 415 |

Inherent means what the physics needs with the topology known at compile time and the muscle ends still runtime (they attach anywhere). The overhead is one thing repeated: the tree is runtime, so every rod update is a parent_mask (8 LOP3 and shifts) plus a scan over 8 slots with predicated moves, about 20 instructions where a baked tree needs one FMA. That is the 20% LOP3, and it is also most of the tree solves, the factor and the assembly: about 4,200 of the 9,000 per substep. The active-set's 8x8 LDL is another 1,000 above a PGS on the same matrix, and its L[8][8] is the stack frame.

Floors per creature-step, warp instructions at W = 2, 8 nodes and 19 muscles: per-plan baked 415 at 2 substeps, 235 at 1 substep, about 290 at spin-adaptive (1.3 substeps mean). Per-lane general tree with the removable overhead removed (state word in registers, stash gone, PGS or one warm round, rf hoisted; the predicated scans stay): about 7,500 per substep, 960 per creature-step at 2 substeps, 530 at 1, 660 spin-adaptive. The lane-group reduced-coordinate kernel, from its measurement (45 M creature-steps/s at 62% of 239 G warp instructions/s, 4 creatures per warp): about 3,300 warp instructions per creature-step now; its floor with the control flow, barriers and muscle stalls trimmed about 2,500 at 2 substeps, 1,350 at 1, and its tree passes are level-serial by construction, so no baking helps it. The plan should carry 415 (baked), 960 (general) and 2,500 (lane-group), not 256.

## Q1. Per-plan kernels

Yes, the runtime topology is the cost, and the rejected single-plan experiment does not transfer: the old kernel was maximal coordinates with projection passes over packed node indices, and specializing it removed index unpacking, a few percent of passes that cost the same either way. Here the topology decides which register slot a rod's parent is, and a register array cannot be indexed at runtime, so the stub scans with predication. Baking the parent array turns each scan into one operand. That is 45% of the substep, not 10%.

What "plan" means for the kernel: the parent array of the rods in canonical breadth-first order, which is the bone tree up to the packer's order. Not the muscles, not the lengths. The number of distinct trees in a mature ring is unknown; rooted unordered trees number 115 at 8 nodes and 4,766 at 12, but MAP-Elites with copy-limb and graft operators concentrates on a few families. My estimate: 300 to 1,000 distinct trees in a generation-51 ring, the top 30 covering 60 to 70% of creatures and the top 100 about 85 to 90%. Measurement: a 50-line example that canonicalizes each ring genome's parent array (the packer already does the breadth-first order), counts distinct trees, and prints the cumulative share and the share of lane-steps per tree, on save42 and on night.evo. Half a day, no GPU.

Cost: a plan fixes the class, so 100 plans x scoring or recording x 60 Hz or 4x = 400 kernels per world at 1 to 2 s each, 10 thread-minutes and 60 MB per world, 1 GB of cache and 2.5 hours once per machine for 15 worlds. Cold-world idle is avoided by one rule: a plan bucket routes to its baked kernel when it exists and to the general kernel otherwise, and the background compiler works the plan list by lane-step share. Warps must be plan-uniform, which is data's bucket machinery keyed by plan.

The middle path needs no compiles: the rod state the scans index goes to shared memory in [rod][lane] layout and the parent index becomes an address, one LDS, one FMA and one STS per rod update instead of 20 instructions. Estimate: tree solves 2,700 to 900, factor 300 to 150, assembly bit tests 1,000 to 400; about 650 warp instructions per creature-step with the solve trimmed, 1.8x on the stub, at about 800 shared operations per lane-substep (Q6 says whether that is a wall). A class of (node count, depth, branching) buys nothing: the parent array is the plan.

## Q2. The solve

The direct active-set solve buys exactly one thing: the contact velocities are solved to rounding in one pass, which is the planted-foot property the 1-substep honesty needs. At 2 substeps the lane-group's PGS holds elites (ratio 1.03), so at 2 substeps the direct solve buys nothing measurable and costs 1,000 instructions per substep plus the spill.

Counts per lane per substep on the formed Delassus matrix: PGS with 4 sweeps and the warm-started impulses, W read from shared, about 600 (4 x 8 rows x 18) and no factor, so no L[8][8] and no spill; direct active set capped at MAX_ROUNDS = 1 with the warm start, about 480, exact whenever the active set did not change since the last substep, which after the first contact substep is most of them, and one round late otherwise; PGS on the rows without forming W needs a tree solve per row update, about 3,200, out. The cheapest that keeps the ledgers is the one-round warm-started active set, because the ledgers are impulse sums and both keep them; its honesty risk is one substep of an under-resolved contact after a set change, which the friction-work ledger catches and which PGS has at every substep. Projected Jacobi is dead because it updates every contact from the same stale velocities, so coupled contacts over-correct together and the sum is a net push that does work the ledger cannot see (random bodies gained 2.1 m).

## Q4. Three paths at the generation-51 mix, 2 substeps

Today's base at the mix: 27 M. The mean body is the stub's body (8 nodes, 19 muscles); the p90 (12 nodes, 29 muscles) at W = 4 costs about 1.7x per step; the tail about 4x.

(a) Lane-group plus trimming. MIO share now: about 430 of 6,600 instructions per warp-substep, 22.5 M warp-substeps/s, 9.7 G MIO issues/s, at the probe's 10.8 G/s. If the probe is the pipe, the kernel is MIO-bound today and 62% is the issue rate that pipe allows. Removing the 35 barriers per substep and half the shuffles lifts the MIO bound 1.7x, but the instruction bound then binds at about 1.3x, because 3,300 warp instructions per creature-step do not shrink with the MIO ops. With the contact round in: about 43 M at the mix, 1.6x. Lessons that transfer: the split muscle record in L2 and the one-word muscle state; nothing else, because its passes are level-serial.

(b) Per-lane general tree, trimmed: 960 per creature-step, 115 M at boost on the mean body, about 65 M at W = 4 on the p90, 15 M on the tail. Harmonic over 70, 21, 9: 65 M. With the growth rule holding the tail at 1%: 98 M. 2.4x to 3.6x.

(c) Per-plan baked, the top 100 plans on baked kernels (85 to 90% of creatures) and the rest on (b): mean body 415 per creature-step, 280 M at boost on the mean body, 150 M on the p90, 30 M on the tail. Harmonic: 120 M with the tail, 170 M with the growth rule. 4.4x to 6.3x. The shared-indexed middle path: 650 per creature-step, 170 M on the mean body unless MIO-bound (Q6), harmonic 75 to 95 M, 2.8x to 3.5x.

I back (c) with the shared-indexed general kernel as its fallback, in that order of measurement, not of building. The 2-day experiment comes first and needs no plan machinery: hand-bake one 8-node tree into the stub (constants for the parent array, parent_mask folded, the q-scans gone, RCO from a constant incidence table indexed by the runtime contact node), run it on plan-uniform synthetic bodies, and read instructions per creature-step and the stack frame from the same report. If it prints under 500 warp instructions per creature-step and no spill, the per-plan track opens; between 500 and 800, the shared-indexed path is the design; above 800, the per-lane layout tops at about 100 M at the mix and the plan says so. In parallel, the distinct-tree count on save42 decides whether 100 plans cover 85%.

## Q6. The two limits for my path

At the FMA limit (124 G warp instructions/s): baked 415 per creature-step gives 300 M on the mean body; general trimmed 960 gives 130 M; the stub 1,160 gives 107 M (it measured 96 M at boost and 73 M locked, consistent).

At the MIO limit (10.8 G MIO issues/s): the baked kernel's shared and shuffle operations per lane-substep are about 500 (muscles 80, tables 30, Delassus 36, W reads in the solve 36 to 250, exchanges 150), 1,000 per lane-step, and a warp instruction serves 16 creatures, so 62 MIO issues per creature-step: 174 M. The shared-indexed general kernel: about 1,600 per lane-step, 100 per creature-step: 108 M, so it would be MIO-bound below its instruction bound. The lane-group kernel: 430 per warp-substep for 4 creatures, 215 per creature-step: 50 M, which is where it sits.

One caution: the MIO probe (a dependent chain with one shared load and one shuffle per two FMAs) measures the pipe's latency-bound rate at that ILP, not its throughput; Ada's documented shared-memory rate is one 32-lane 4-byte access per clock per SM, 60 G/s, 5.5x the probe. A second probe with 8 independent loads in flight per thread settles it in the same session. At 60 G/s no path above is MIO-bound and the baked kernel sits at its FMA bound, 300 M on the mean body; at 10.8 G/s it is 174 M and every shared table in the kernel is a cost to design against.

## Revised multipliers and ceiling

Kernel row at the generation-51 mix, 2 substeps, from 27 M: lane-group trimmed 43 M (1.6x); per-lane general 65 to 98 M (2.4 to 3.6x); per-plan baked 120 to 170 M (4.4 to 6.3x), the range being the tail with or without the growth rule, and the mean-body rate capped at 174 M if the MIO probe holds or 300 M if it does not. The 3 T and 5 T columns collapse into the measured FMA cap (124 G warp instructions/s at 1,965 MHz, about 4 T thread instructions/s); there is no 5 T row any more.

Creatures per second, per-plan path, growth rule in, R4 at 50% (265 steps), host tax 18%, helpers 1.5%: 2 substeps 170 M / 265 x 0.81 = 0.52 M/s; spin-adaptive (1.6x) 0.83 M/s; 1 substep (1.85x) 0.96 M/s. With the tail on the general kernel: 0.37, 0.59 and 0.68 M/s. At generation 100 with the muscle slope at 0.75: 0.39, 0.62 and 0.72 M/s. The honest ceiling of this laptop with this physics is about 1 M/s at generation 51 and 0.7 M/s at generation 100, and it needs the per-plan bake, the growth rule and a passing substep rung. 2 M/s is not reachable here without a cheaper muscle model (10 muscles per lane at 70 each are 700 of the 3,000 inherent instructions per substep, the largest block and the one that grows) or a second GPU. The plan should say so.

What changes my mind: the baked stub under 350 warp instructions per creature-step moves the ceiling to 1.3 M/s; the MIO throughput probe at 60 G/s removes the 174 M cap; a tree count with the top 30 plans covering 90% cuts the compile cost to a tenth and makes per-plan the default.
