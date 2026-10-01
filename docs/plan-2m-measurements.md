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

## physics-lean step 1 (2026-10-01, claude/physics-lean): the lean stub counts, above the table everywhere

What was built. `shaders/lane_lean.cu` is the stub of main (2e946d1, trim and baked) with the lean variants behind defines: MUSCLE_MODEL=1, CONTACT_MODEL=1 with NPASS=2, SUBSTEPS=4 with LAGGED_FACTOR=1, LIMITS_AS_IMPULSES=1 with LIGAMENT, and MUSCLE_ANCHORS=1 for the price of the anchor fractions. With every define off it reproduces main: 914 (trim) and 629 (trim, baked on tree 0,1,1,2,4,5,6). `shaders/lane_lean_w1.cu` is a second copy, written W-generic, that runs one creature per lane (W = 1) with a baked tree or the runtime tree. `tools/lane-count.sh` counts one build. All numbers are Nsight Compute at the locked 1.89 GHz, 32,768 synthetic 8-node 19-muscle bodies, 300 steps, with the GPU shared. The model as built: muscles join two nodes, force is cap x strength x trapezoid x stamina x Hill with the damper, one stamina store per creature, the touchdown clocks per limb; contacts are per-node impulses against each node's own mass (normal, then friction clamped to the cone with the clean rule and the anchor in a register) followed by the exact rod solve, twice per substep, with the penetration left recovered by a position nudge; the momentum and angular ledgers of the contact impulses run every substep; joint limits and the spin cap are one angular impulse per joint and step; the lagged factor keeps the rod factor and the drag and damping forces for the step.

Counts at 2 substeps, warp instructions per creature-step (delta from the control), registers, issue-active, rate at the locked clock. Nothing spills except where noted.

| build | general (914) | baked (629) |
|---|---|---|
| muscle model | 800 (-115), 244 regs, 49.7%, 109M | 526 (-103), 230 regs, 43.1%, 146M |
| muscle model with anchor fractions | 858 (+59 over the muscle model), 101M | 588 (+62), 132M |
| contact model, 2 passes | 548 (-366), 212 regs, 45.6%, 147M | 386 (-243), 208 regs, 40.4%, 188M |
| limits as impulses | 934 (+20), 255 regs, 41.8%, 79M | 651 (+22), 255 regs, 42.8%, 117M |
| muscle, contact, limits | 464 (-451), 193 regs, 44.2%, 168M | 308 (-321), 175 regs, 37.8%, 218M |
| the same with the anchor fractions | 522 (+58), 150M | 368 (+59), 188M |
| the same, ledgers audited not enforced | 441 (-23), 172M | 285 (-23), 270M |

Substeps, the same two bases. 4 substeps unlagged is 1,714 (general) and 1,177 (baked); the lagged factor alone gives 1,574 (-140, 255 regs, 106 B of spill) and 1,098 (-80, 24 B of spill). The composed build at 4 substeps with the lagged factor: 689 (106M) general, 489 (130M) baked. At 1 substep: 277 (293M) and 183 (394M). Without the enforced ledgers: 641 and 441 at 4 substeps.

One creature per lane (`lane_lean_w1`, all variants composed, ledgers enforced), against gpu's table (baked 131, 182, 77 at 2 substeps, 4 lagged and 1; general runtime tree 203, 264):

| W = 1 | warp instr | vs table | regs | issue | locked rate |
|---|---|---|---|---|---|
| baked, 2 substeps | 206 | 1.57x | 221, no spill | 55.1% | 463M |
| baked, 4 lagged | 331 | 1.82x | 235, no spill | 42.3% | 216M |
| baked, 1 substep | 115 | 1.49x | 235, no spill | 56.9% | 871M |
| runtime tree, 2 substeps | 354 | 1.75x | 255, 94 B of spill | 33.3% | 158M |
| runtime tree, 4 lagged | 527 | 2.0x | 255, no spill | 35.2% | 113M |
| runtime tree, 1 substep | 209 | | 247, no spill | 36.7% | 310M |
| baked, 2 substeps, anchor fractions | 277 (+71) | | 229 | 58.8% | 368M |
| baked, 2 substeps, ledgers audited | 184 (1.40x) | | 217 | 49.5% | 462M |
| baked, 4 lagged, ledgers audited | 287 (1.58x) | | 225 | 37.1% | 218M |

The W = 2 composed builds against the W = 2 table (baked 160, 220, 95): 308 (1.93x), 489 (2.22x), 183 (1.93x).

The gate (within 1.3x, no spill) is not met by any composed build. Above 1.5x, to be reported as such: W = 2 baked at all three substep counts, W = 1 baked at 2 substeps (1.57x) and 4 lagged (1.82x), W = 1 on the runtime tree at both. W = 1 baked at 1 substep is 1.49x. Each lean variant on its own does what the plan expected in direction and falls short in size: the muscle model removes 103 to 115 where the plan counted about 136 (186 to about 50 unbaked), the contact model removes 243 (baked) and 366 (general) where the plan counted 336 to about 30, the lagged factor removes 80 to 140 of the 4-substep cost, limits add 20.

Where the baked W = 1 count sits against gpu's per-section figures (warp instructions per creature-step at 2 substeps; gpu's lane counts times 2 substeps over 32): muscles 76 (gpu 40; 64 lane instructions per muscle and substep against 34), drag, damping and limits 35 (gpu 19), contact passes and ledgers 44 (29), rod rows, factor and solves 25 (20), projection 11, Euler 5, free velocities 5, node table 3, metrics 3. The two sections that carry most of the gap are the muscles and the explicit forces. The ledgers cost 22 (audit-only builds are the last rows above). The anchor fractions cost 58 to 71 on every base: 13% of the lean build on the general base, 19% baked, 34% at W = 1.

Registers are not 117 to 145. W = 1 baked needs 221 to 235 without a spill, the W = 2 composed builds 175 to 221 (gpu: 90). All of them at 8 warps per SM.

The layout decides the W = 1 rate. With the per-creature records of the stub the baked W = 1 build issued at 22% and ran 197M at the same 206 instructions: 4.2 of 8.9 cycles per issued instruction were short scoreboard. Nsight showed 4-way bank conflicts on every 64-bit shared access (11.6 wavefronts for 2.9 ideal) from a 32-byte element per lane, and 32 L1 tag requests per record load. The record, muscle record, limb clock and held geometry arrays are now interleaved across the warp's 32 creatures (what the per-block prologue kernel writes) and the node and force tables are separate float4 and float2 arrays. Issue-active went to 55% and the rate to 463M, which is 607M at boost by the clock ratio (below). The lane-w1 gate of 500M at 2 substeps is met on this count with ledgers enforced. Grouping the muscles to hide the shared scatter, fixed-point shared atomics for the force sums and float4 force tables each gained nothing and are not kept.

DIAG on the stub's synthetic bodies (violent: strong random muscles, random masses; 0.05 to 0.3% of them still go non-finite under every build including today's rules, and 55 to 90% end with the head below the neck). Medians over creatures. Today's rules, trim: rod drift 0.28 m (the maximum over a trial). Lean, baked tree, 2 substeps: rod drift 5.0e-3 m (W = 1: 2.8e-2), rod residual 4.0e-2 m/s, angular ledger residual 1.3e-7, normal approach residual after the last rod solve 0.62 m/s, penetration per creature (maximum over the trial) 11 mm median and 4.9 m at p99. Without muscles: drift 4.5e-3, penetration 10 mm median, 0.62 m p99. The last rod solve leaves a median 0.6 m/s approach at the contact nodes, so the 2 mm p99 penetration bar needs a contact pass after the last rod solve or a third pass; this is not measured here. 1 substep: drift 9.2e-2 (W = 1: 0.67), so 1 substep stays dishonest. The lagged factor on random trees is unstable on these bodies (median drift 0.37 m against 6.1e-3 unlagged, general, 4 substeps) and stable on the baked tree (6.6e-3) and with muscles at 0.2 of the stub's strength (5e-3): the limit impulse is a velocity target taken at the start of the step and spread over its substeps, and strong muscles push the joint past it. This needs a decision in physics3 (limit at the last substep, or not held).

Rates. The GPU was never idle (another job held it at 96 to 100% all session), so no boost measurement is clean. Rates at the locked clock are Nsight's own kernel durations, and the control reproduces main's 81M and 121M. A boost rate is the locked rate times 1.31, the clock ratio (2.475 against 1.89 GHz; the first stub measured 96M against 72.9M locked, 1.32).

Reproduce: `tools/lane-count.sh lane_lean trim plan=0,1,1,2,4,5,6 baked -DMUSCLE_MODEL=1 -DCONTACT_MODEL=1 -DLIMITS_AS_IMPULSES=1` (substeps=4 -DLAGGED_FACTOR=1 for the lagged build, -DDIAG with the example itself for the residuals); W = 1: `tools/lane-count.sh lane_lean_w1 w=1 -DBAKED=1 min_blocks=2 -DMUSCLE_MODEL=1 -DCONTACT_MODEL=1 -DLIMITS_AS_IMPULSES=1` (add -DMUSCLE_UNROLL=4 with the lagged factor).

## dump (2026-10-01, merged 500faff): the ladder saves 12%, not the plan's 1.5x

Generation-50 dump of a 3M run (seed 5051, best 63.5 m, qd 41,579): steps per creature 838 in full, 397 under today's 5 s screen (the plan assumed 480). Ladder R1 to R3 with today's bar: 348 steps (12% fewer), top 1% kept 100%, top 10% 99.4%, entrant misses 0.0, 6.2 and 15.2 per 10k at R1 to R3 (today's screen alone misses 21.6). R4 stops 0.09% with today's bar (worth nothing); per-cell bars keep more creatures alive (692 steps) and are not a speed lever. Nurseries must be exempt from the early rungs (70% stop at R3 with 117 misses per 10k). 34.3% of creatures fall. Spearman of d(10 s) to final 0.991, of d(5 s) 0.969. Bodies at generation 50: 9.48 nodes mean, p50 8, p90 15, p99 32; 24.8 muscles mean, p90 45; classes 3 to 8 nodes 52.8%, 9 to 16 41.0%, 17 to 32 6.3%. Entrants born more than 4 nodes above their parent: 65 of 11,984; more than 4 muscles: 422. Determinism: two identical search_ab runs on main differed from generation 3 while other agents loaded the GPU; retest on a quiet GPU.
