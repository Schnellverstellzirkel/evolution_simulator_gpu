# Round 7, GPU domain: the cheaper physics, counted on the stub

I accept the proposal's shape: node-to-node activation muscles with one stamina store, per-node contact impulses coupled by the exact rod solve, symplectic Euler with more substeps. It removes the three blocks the stub spends most on (the muscle state traffic, the Delassus matrix, the active set) and it removes the register arrays that spilled. Where I differ is the count at 2 and 4 substeps: physics's table lets the per-substep fixed costs (forces, tables, ledgers) stay nearly flat as substeps grow, and they do not.

## (a) The count, per lane per substep, from the stub's sections

W = 2 baked (each lane 4 nodes, about 10 muscles), against my round-6 baked floor of 3,000 per substep:

| section | today baked | proposal | why |
|---|---:|---:|---|
| forces, damping | 150 | 100 | joint damping per rod with baked grandparent |
| muscles | 700 | 320 | 10 x (24 model + 8 shared FRC scatter); no state word, no unpacks, no tendon |
| tables | 60 | 60 | node table and FRC clear stay |
| rod rows, factor | 160 | 160 | directions, RHS 100; factor 60 (once per step if lagged) |
| tree solves | 630 | 70 | two rod solves per substep, 35 each with the exchange, no contact rows |
| contacts, Delassus, active set | 990 | 80 | 4 nodes per lane x 10 x 2 passes |
| joint limits, spin as impulses | in the rows | 40 | |
| ledgers, impulses, Euler, anchors | 310 | 200 | no z1 solve, no stash |
| per substep | 3,000 | 1,030 | |
| once per step (rhythm, projection, metrics, angular ledger) | 650 | 480 | |

Per creature-step at W = 2: 2 substeps 2,540 lane instructions, 159 WI (physics 125); 4 substeps with the lagged factor and the muscle geometry held for the step (muscles 160 per substep, rows 40, factor 0): 4 x 750 + 480 = 3,480, 218 WI (physics 170); 1 substep 1,510, 94 WI (physics 90). We agree at 1 substep; at 2 and 4 physics undercounts the ledger and table work that repeats per substep by 25 to 30%.

W = 1 (one creature per lane, 8 nodes, 19 muscles): no exchange, but the node table must be shared memory, because a muscle's two end nodes are runtime indices and a register array cannot take one (2 LDS.128 per muscle instead of 64 predicated selects). Per substep: forces 200, muscles 19 x (24 + 2 LDS + 8 FRC) = 646, tables 24, rod RHS 80, factor 120, solves 2 x 56 = 112, contacts 8 x 10 x 2 = 160, limits 105, ledgers 300: 1,750. Per creature-step: 2 substeps 4,200, 131 WI; 4 substeps lagged 5,830, 182 WI; 1 substep 2,450, 77 WI. W = 1 beats W = 2 by 18%, physics's 15%.

Registers at W = 1, 8 nodes: state 32, masses 8, rod factor 28, anchors 8, force accumulators 16, limb clocks 8, stamina 1, muscle-loop and solve temporaries about 45: about 145, or about 117 with the lagged factor parked in shared memory. Physics's 90 is right at W = 2 and optimistic at W = 1. The stub measured that occupancy does not move the rate, so 12 warps at 145 registers costs nothing; the spill is what matters, and with no L[8][8] and no z[6][4] there is nothing left to spill. W = 1 fits.

MIO at W = 1 per creature-step at 2 substeps: 2 x (19 x 10 + 24) + 20 = 450 lane operations, and a warp instruction serves 32 creatures, so 14 MIO issues per creature-step. At the probe's 10.8 G/s that is 770 M creature-steps/s, below the FMA bound of 950 M (124 G / 131), so at W = 1 the shared scatter is the wall if the probe's number is the pipe. The alternative moves the FRC scatter to registers by predicated selects: 16 more FP instructions per muscle, 2 MIO per muscle: 150 WI and 3.5 MIO issues per creature-step, FMA-bound at 830 M, MIO-bound at 3 G. The throughput probe decides which; both are above 750 M on the mean body.

Rates at the FMA cap on the mean body: W = 2 baked 780 M at 2 substeps, 570 M at 4 lagged, 1.3 G at 1; W = 1 baked 950 M, 680 M, 1.6 G. At the generation-51 mix (physics's 0.85): W = 1 810 M, 580 M, 1.37 G. Creatures per second at 300 steps and 0.81 for host and helpers: 2.2 M, 1.57 M, 3.7 M; at 265 steps 2.5 M, 1.77 M, 4.2 M. So 2 M/s needs the 2-substep physics to pass its retest, or 4 substeps with R4 plus a 15% margin that is not there.

## (b) The host for the 30-generation evolution

The lane-group kernel. The stub has the layout and nothing else: no worlds, falls, screen, metrics, replay or ring; a 30-generation evolution at 3M is the whole game path. The lane-group's muscle round takes the new muscle in days: drop the pivot table and the anchor interpolation, keep the `ends` list as the scatter, one stamina register per group, the trapezoid for `wave`, the tendon gone and the ligament compliance folded into the joint-limit term that already exists in the articulated-body pass. Its contact solve stays: reduced coordinates have no rod solve to couple per-node impulses, and its PGS holds elites. The per-node scheme is a cost item, not a gait change, so it is validated where it will live: the stub for the count, the per-lane kernel for slip and penetration. The elites evolved on the lane-group host are then re-scored on the per-lane kernel, and that transfer (median at least 0.9) is the finer-solver retest the spirit asks for.

## (c) Per-plan baking on the new physics

What still depends on the tree: the rod RHS and factor, the two rod solves per substep, the joint-limit impulses (parent lookups) and the projection. In the general W = 1 kernel the runtime tree turns each of those into the stub's predicated scans: solves 2 x 56 to 2 x 280, factor 120 to 500, limits 105 to 200, projection 250 to 700. At 2 substeps that adds about 2,300 lane instructions on 4,200: 203 WI against 131 baked, 1.55x. At 4 substeps lagged: 264 against 182, 1.45x. The shared-indexed general kernel (rod state in [rod][lane] shared memory, the parent index an address) recovers most of it at about 1.2x over baked, with 60 more MIO operations per substep, which at W = 1 sits on the same pipe question as the scatter. So: per-plan baking pays 1.45 to 1.55x on the new physics, down from 2.9x on today's, and the order changes. Build the general W = 1 kernel with shared-indexed rods first (no compile explosion, one kernel per world and class), and take per-plan baking as a later track worth 1.2 to 1.3x over that, only if the tree count shows 100 plans covering 85%. Rates at the mix, 2 substeps: general W = 1 520 M (1.4 M/s at 300 steps), shared-indexed 680 M (1.85 M/s), baked 810 M (2.2 M/s). 4 substeps lagged: 400 M, 480 M, 580 M (1.08, 1.3, 1.57 M/s).

## (d) The stub variants, one day each

Each variant reports the same lines from examples/lane_stub.rs and its ncu recipe: registers, stack frame, spill stores and loads, issue-active, warp instructions per creature-step, the section shares, the rate locked and at boost, and the DIAG residuals (rn, rr, drift_max).

1. `-DMUSCLE_MODEL=1`: node-to-node activation muscles; two words per muscle (packed node pair and sensor limb, cap times strength, v_max; period, phase, duty); the trapezoid a(t); one stamina register per creature; `roff` per limb; no `mstate`, no tendon; FRC scatter to two nodes. Expect the muscle section from 186 WI to about 50.
2. `-DCONTACT_MODEL=1 -DNPASS=2`: per-node impulses against the node's own mass for every touching node, then the rod solve, twice; removes batches A and B, the Delassus rows, the active set and the stash. Report p99 penetration from rn and the anchor displacement as slip. Expect 336 WI to about 30.
3. `-DSUBSTEPS=4 -DLAGGED_FACTOR=1`: factor once per step, muscle geometry held per step. Expect about 220 WI at W = 2; report drift_max.
4. `-DLIMITS_AS_IMPULSES=1 -DLIGAMENT=<compliance>`: per-joint impulses with the compliance folded in; report the angular ledger residual.
5. `-DW=1`: two days (the stub errors on W != 2): NPL = 8, no exchange, node table in shared memory. Report registers and spill first, then WI and the LDS, STS and SHFL count per step from the SASS.

Variants 1 to 4 compose; the plan's count is all four at W = 2, then at W = 1.

## The physics-lean track as I would gate it

Stub counts within 1.3x of the table above with no spill. Then the lane-group host evolution, 30 generations at 3M under the muscle change: first_generation, the elite retest at 4x the substep count (median 0.9, p10 reported), slip 0.95 to 1.05, the four ledgers at 1e-5, the top-20 replays. Then the per-lane W = 1 kernel: the transfer retest at median 0.9, p99 penetration under 2 mm, and at least 500 M creature-steps/s on the mean body at 2 substeps (55% of its FMA bound), the number that puts 2 M/s within reach. Then the ladder: evolve at 2 substeps of 1/120, retest at 8.

Revised multiplier, from 27 M at the generation-51 mix: general W = 1 kernel 520 M (19x) at 2 substeps, 400 M (15x) at 4 lagged; shared-indexed rods 1.3x on that; per-plan baking 1.2x more. Ceiling at generation 51: 1.4 to 2.2 M/s at 2 substeps, 1.1 to 1.6 M/s at 4, general to baked; at generation 100 with the 0.85 slope, 1.2 to 1.9 and 0.9 to 1.35 M/s. If the MIO probe's 10.8 G/s holds, the W = 1 scatter moves to registers and every number loses 10%. 2 M/s is back on the table, and it rests on the 2-substep retest.
