# Plan measurements as they land

## host-profile (2026-09-30, save42 at generation 51, 196k blocks, CPU times with the GPU shared)

Archive stage per block: median 55 ms, p95 77 ms, max 92 ms (island offers 13 to 15, prefilter 14 to 16, global offers 13, CMA tell 8.5, island refresh 4 to 6, novelty refresh 5.5 median and 12.9 max, lineage 1.4 to 1.8). Scaled to a 100k block: about 30 ms median, 40 ms p95.
Candidates: 1.9% of results beat an island cell (2.7% of unscreened); with reserve and global candidates 2.1%, under the 5% gate. Tells: 4.5 per block of 96 CMA emitters.
Generation boundary: 22 to 41 ms; prune_lineage dominates (21 ms at 18k entries).
Breeding on the same serial chain: 232 ms per 196k block (plan 57, emit 150, write 25), so the host chain is about 290 ms per block today.
Decisions: no incremental novelty (fallback if the refresh p95 passes 20 ms at 100k), no pool-side boundary, ring floor 0.3 s (binds only after host-bounded and host-arenas shrink the breed chain).

## warp-speed (2026-09-30, merged 96c4d40)

Lane-group kernel 1.22x (kernel time at locked clocks 640 to 524 ms on 30k dump creatures; about 53 to 55M creature-steps/s). Elites: median 34.27 m, 2x ratio 0.986; random bodies best -0.01 m. Projected Jacobi rejected: random bodies gained up to 2.1 m. Profile: issue slots 53 to 60% busy, about one eligible warp per scheduler, a fifth of instructions are control flow; the muscle section stalls on global record loads.

## stub-kernel (2026-09-30, merged a125590): gate missed on rate and spill, met on issue

W = 2 per-lane maximal-coordinate stub on 8-node, 19-muscle synthetic bodies, 2 substeps: 128 registers with 424 B stack frame and 614 B spill stores; issue-active 47.9% at 14.9 warps; 72.9M creature-steps/s at 1.89 GHz locked, about 96M at boost (gate 250M). 1,160 warp instructions per creature-step against the debate's 256 estimate: muscles 16%, Delassus assembly 15%, active-set solve 14%, tree batch A 10.5%, projection 5.5%; LOP3 is 20% of instructions (runtime rod topology bit tests and packing). Occupancy does not change the rate (16, 12 and 8 warps give 1,160, 1,142 and 1,074 instructions per step at nearly equal time). The lane-group kernel's warp-state sample: issue-active 61.5 to 62.8% at 15.5 warps, barrier under 0.5%, short scoreboard 18%, wait 33%, long scoreboard 17.5%. So the per-lane layout is worth about 3x at mature bodies (96M against the lane-group's 27M on the generation-51 mix), not the plan's 8.9x to 14.8x, and the plan's multiplier table needs revision.

## tree-count (2026-09-30, merged 92ca335): gate passed, per-plan kernels are the default

Canonical trees (children sorted by subtree shape, neck first) on save42's bred ring of 786k: 10,256 distinct trees; top 30 cover 72.9% of creatures (62.2% of lane-steps), top 100 cover 87.6% (77.7% of lane-steps); 85% of creatures needs 74 trees, 85% of lane-steps 197. Packer bone order as-is gives only 80.6% at top 100, so the packer must canonicalize. The island archives alone: 82.7% at top 100. Gen-10 dump elites: 104 trees, top 30 cover 84%. The general kernel always carries about 12% of creatures and 22% of lane-steps past the top 100.

## bucket-counters (2026-09-30, merged 2dee40f): 0.7% kept; the plan's 1.15x for bucketed queues is refuted

Per-bucket take-up counters on the lane-group kernel: kernel time 3316 to 3293 ms over the three classes on the save42 dump (0.5 to 0.9% per class), results identical. Cause: pack already sorts each wave by (rounds, depth, nodes), so warps are 98 to 99.8% uniform with a single counter; no take-up change can gain more than 1 to 2% while blocks stay sorted. The persistent-kernels row's 1.15x should be re-estimated at about 1.01x unless its blocks are unsorted. Gap-excluded kernel timing (sum of per-warp step intervals under 400 us) measures under the shared lock; tools in the scratchpad.

## End to end on main d7264e4 (2026-10-01, game alone, 3M per generation, autochange Slow)

Generations 0 to 5: 139k, 227k, 197k, 114k, 115k, 149k creatures/s (mean nodes 5.9 to 6.9; confirmations 395 to 1,154 per generation; breeding 1.3 to 1.8 s per generation). Previous main (before the contact solve, bar-stream and frames): 92k to 167k at the same stage; the old build: 85k to 117k, falling to 27k by generation 30.

## baked-stub (2026-10-01, merged 2e946d1): baked 629, general trimmed 914, no spill

W = 2 stub, 8-node 19-muscle bodies, 2 substeps, locked 1.89 GHz, 8 warps per SM at 254 registers with no spill: general trimmed 914 warp instructions per creature-step (82M), baked on tree 0,1,1,2,4,5,6: 629 (121M; about 160M at boost, FMA cap 197M). Other top trees bake at 606 to 664. The runtime tree costs 285 (31%): tree solves 207 vs 40, rod factor 67 vs 20, shuffles 57 vs 23. What is left in baked is physics: muscles about 190, contacts about 230 plus tree rows, drag and damping 43, ledgers 36, rod directions 31; about 275 per substep plus 80 per step. By the plan's rule (500 to 800) the shared-indexed general kernel is the design and per-plan kernels do not open on today's physics. Cheaper rules, not yet checked for honesty: muscle force held per step 549, plus analytic drive 540, cold Gauss-Seidel 2 sweeps 569, all three 483. Baked issues 66 MIO instructions per creature-step (8 G/s at the measured rate, within 1.4x of the probe).
