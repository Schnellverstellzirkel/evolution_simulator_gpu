# Round 6, physics: the floor, counted from the stub

Units: WI = warp instructions per creature-step (the stub's 1,160). At W = 2 a warp holds 16 creatures, so the stub's stream per step is 18,560 instructions per lane pair; the debate's 256 was a stream of about 4,100 with no overhead. Below, what the physics needs against what the stub spends.

## Q3. The floor per creature-step, 8 nodes, 19 muscles, 4 contacts

Inherent work per creature (thread instructions), then the stream cost at W = 2 (split over 2 lanes, plus the exchange), then WI.

| section | inherent per creature | stub WI | floor WI | overhead in the stub |
|---|---:|---:|---:|---|
| muscles, 19 x (60 per substep + 10 per step) | 2,470 | 186 | 77 | shared scatter and gather of forces (about 16 MIO per muscle), state pack and unpack, predication |
| forces, damping (8 nodes x 12, 7 rods x 25) | 270 | 58 | 9 | parent one-hot per rod for the damping couple, predication |
| rod factor, 2 substeps (7 rods x 18 with the lane-1 to lane-0 Schur shuffles) | 250 | in batch A | 16 | parent_mask per rod, NB sibling tests |
| tree batch, 9 rows x 7 rods x 2 passes x (4 flops + 3 shuffles), 2 substeps | 880 | 122 | 63 | parent_mask on every rod of every row and pass (72 calls per substep), slot predication; at W = 1 this is 16 |
| Delassus, 8 x 8 over 7 rods (448 flops) plus 36 exchanges, 2 substeps | 970 | 174 | 50 | rod projections built with topology tests per row per rod, W to shared and back |
| active-set solve, 1 round warm-started (LDL 170, solve 64, checks 30), both lanes, 2 substeps | 530 x 2 | 162 (3 rounds) | 33 | 3 rounds at the warp maximum, about 27 WI each |
| contacts: 8 gaps, deepest 4, 8 rows | 240 | 58 | 8 | ballot-free selection over 8 values, topology for the rod projections |
| projection, once per step (factor 175 + solve 56 + apply 42) | 273 | 64 | 12 | a second full factor even with no anchored node |
| ledgers (momentum 32, angular 112, first law 64), 2 substeps | 420 | in forces | 14 | none counted |
| integrate, metrics, screen | 200 | about 20 | 8 | |
| topology bit tests and packing (LOP3, 20%) | 0 | 232 | 0 | all overhead, spread over the rows above |
| spills (614 B of stores plus the loads) | 0 | about 60 | 0 | |
| total | | 1,160 | about 290 | |

So the floor at W = 2 and 2 substeps for the p50 body is about 290 WI, near the debate's 256, and the stub carries 4x overhead: about 230 of topology bit tests, 200 of exchange and predication in the tree work, 110 in the muscle scatter and packing, 130 in extra active-set rounds, 60 in spills. Joint-limit, spin and tendon rows add about 20 WI. At 1 substep the per-substep rows halve: about 175 WI; spin-adaptive about 200. The per-step rows (waveform, projection, metrics) do not halve, which is why the substep lever is 1.6x and not 2x.

The lane-group reduced-coordinate step at W = 8: its stream per substep (kinematics 60, muscles 360, ABA 400, forward 75, detection 60, matrix walk 300, root 60, PGS 300, response 400, integration 80) is about 2,100 serving 4 creatures, 1,050 WI per step. Measured: 239G x 0.62 / 55M = 2,700 WI on the generation-10 dump, 2.5x its own stream (control flow, record stalls, warp-maximum loop bounds). Its inherent physics is the same 4,500 thread instructions; the layout's floor is set by 25 to 30% lane use at the tree levels, about 500 WI at 2 substeps and 260 at 1.

## Q1. Topology: baking removes the 20% and more

Yes, the runtime topology is the cost. The old kernel's single-plan specialization gained only 10 to 20% because it had no topology inner loop: its passes ran over arrays with runtime lengths, and specialization only trimmed bounds. In the stub, parent_mask runs on every rod of every row in every tree pass (72 times per substep), the Delassus rows and the damping couples read the topology word again, and every register array is indexed through one-hot bit tests to stay out of local memory: the 232 WI of LOP3 plus about half the predication in the tree rows. With the tree baked as constants (rod parents, cliques, elimination order, lane ownership) the tree solve is straight-line code on fixed registers. Estimate for a per-plan kernel: 1,160 to about 550 WI before other trimming, about 400 with MAX_ROUNDS 1 and the spill fixed.

How many plans: the number of rooted unlabeled trees is 48 at 7 nodes, 115 at 8, 286 at 9, 719 at 10. The archive's 7,500 elites descend from a few hundred lineages, and structural children are one operator away from an elite, so I expect 90% of a mature ring in 100 to 300 distinct canonical bone trees and 50% in about 30. The measurement: canonicalize each bone tree (parent sequence in breadth-first order) for every elite of save42 and every creature of one ring block, and print the cumulative share by rank. If the top 30 plans hold under 50% of the ring, per-plan kernels lose; if over 70%, they win.

Compile cost: compiling every plan is out (300 plans x 15 worlds x 3 variants at 1 to 2 s is hours). The workable form: the general-tree kernel runs every plan; a plan above 1% of the ring gets a specialized kernel compiled in the background for the current world, and its bucket switches over when the cubin exists. Thirty plans per world is 30 to 60 s of background compile, cached on disk (cubins are 100 to 200 KB, so the 200-file cap must rise to about 2,000). No cold-world idle, because the general kernel serves meanwhile.

The middle path: the branching pattern is the tree, so fixing it is a plan. Fixing only the level profile (rods per level) keeps the parent index runtime within the previous level, a compare over 2 or 3 candidates instead of a one-hot over 8; that removes about half of the topology cost with under 100 variants at 12 nodes. It is the fallback if the plan count is too high. Muscle end lists stay runtime in every form (about 10 WI).

## Q2. The solve

The direct active set is worth its cost for one thing: planted feet with no residual slip at velocity level, which 1 substep and the honesty ratio need. At 2 substeps the lane-group's 4 cold sweeps plus a clean sweep hold elites (ratio 1.03), so there it buys nothing measurable. Per substep, with the 8 x 8 Delassus formed (every option needs it, about 25 WI):

- Projected Gauss-Seidel, 5 sweeps x 8 rows x 6 flops, redundant on both lanes: about 15 WI; no residual guarantee.
- PGS without forming W, a tree solve per row update: about 165 WI. Never.
- Active set at MAX_ROUNDS 1, warm started, a 2-sweep PGS polish when the round violates: about 17 WI; exact when the warm partition holds, one substep of slip on a transition.
- Active set at 3 rounds (the stub): 81 WI.

Cheapest that keeps the ledgers: form W (consistent impulses for any solver), then the 1-round warm active set with the polish, about 42 WI per substep. Honesty risk low at 2 substeps, medium at 1 substep on touchdowns, which spin-adaptive already runs at 2. MAX_ROUNDS 3 is a diagnostic.

Projected Jacobi is dead because coupled contacts update from the same stale velocities and overshoot together, and the clamp to non-negative impulses keeps the overshoot's push while discarding its pull: a momentum source, measured as random bodies gaining 2.1 m.

## Q4. The three paths at the generation-51 mix, 2 substeps

(a) Lane-group plus trimming. The MIO count settles it: 430 MIO per warp-substep at 55M creature-steps/s on 4-creature warps is 11.8G MIO issues/s, the measured pipe limit (10.8G). The lane-group kernel is at the MIO wall now; its 62% issue is mostly MIO issue. Halving the shuffles takes MIO to about 250 per warp-substep, 1.7x if nothing else binds; the idle tree-level lanes then bind, so about 1.4x realized: 27M x 1.22 x 1.4 = 46M at generation 51. Ceiling 50 to 60M.

(b) Per-lane general tree with the removable overhead gone: forces accumulated in registers in fixed muscle order instead of the shared table (about 60 WI), parent one-hots hoisted out of the row loops (100), MAX_ROUNDS 1 warm (130), the spill (60), the projection's second factor only when a node is anchored (30): 1,160 to about 750 WI at the p50 body. 124G / 750 = 165M at the FMA limit; about 130M at the mix with W = 4 and W = 8 on the tail.

(c) Per-plan kernels in the per-lane layout: about 400 WI at the p50 body, 310M at the FMA limit, about 240M at the mix with the general kernel on the uncovered share. In the lane-group layout baking removes control flow but not idle lanes, about 1.5x over (a); not worth it there.

I back (b) as the next track with (c) as the measured track behind it: (b)'s trimming is needed by (c) too, and (c)'s gain waits on the plan count. The 2-day experiment that decides (c): compile the stub with its synthetic body's topology as #define constants (parent table, cliques, elimination order) and measure the WI drop. Under 600 WI and (c) is real; above 900 and the overhead is elsewhere and (c) is dropped.

## Q6. My path at the two limits

Path (b) at 750 WI: 124G / 750 = 165M FMA-bound. MIO per creature-step about 46 (tree batch 432 per stream at 2 substeps, muscles 160, node table and W 140): 10.8G / 46 = 235M. The smaller: 165M. Path (c) at 400 WI with W = 1 up to 8 nodes (no exchange; the baked tree brings registers under 128): 310M FMA-bound, MIO about 15 per creature-step so 720M MIO-bound; 310M, about 240M at the mix.

## Revised multipliers and ceiling

| lever | debate | now |
|---|---:|---:|
| per-lane maximal coordinates, general tree, trimmed (b) | 8.9 to 14.8x | 4.8x (130M at the mix against 27M) |
| per-plan kernels (c), if the top plans cover 70% | not in the plan | 1.8x over (b) (240M) |
| spin-adaptive substeps L2.5 | 1.6x | 1.5x (per-step rows do not halve) |
| 1 substep L2 | 1.85x | 1.65x |
| the three cuts (waveform once, ledger in flight, lagged matrix) | 1.14x | inside (b) |

Ceiling at generation 51, 300 steps per creature, 18% host tax: path (b) at 2 substeps 0.36M/s, with L2.5 0.53M/s; path (c) 0.66M/s and 0.98M/s. Generation 100 at the 0.75x muscle slope: 0.27, 0.40, 0.49, 0.74M/s. The two power limits no longer split the table: at these instruction counts the kernel is FMA-bound at the 100 W cap, so the 3T and 5T columns collapse to the 124G figure.

So 2M/s sustained is not reachable on this laptop with this physics. The honest number is 0.5M/s with the general-tree kernel and spin-adaptive substeps, 1.0M/s if per-plan kernels cover the ring and L2.5 passes. The remaining 2x would have to come from steps per creature beyond R3 (R4 is 1.28x) or from fewer muscles per body, which is the search's lever: at 19 muscles the muscle model is 77 WI of a 290 WI floor, at 29 it is 118 of 330.
