# Round 2, GA/QD: cross-examination

## D2. One steps-per-creature schedule (joint with ml, on the 480 base)

The four names for the early rung are one rung. Here is the schedule I back. ml writes the same table from the ml side; the chair can diff them.

| rung | step | predictor | mature population: share stopped here | fresh population (generations 1 to 5): share stopped here |
|---|---|---|---|---|
| R1 | 60 (1 s) | ml's calibrated stop on running state (com_x, speed over 0.5 s, contacts, head_shake), tolerance 1 in 1,000 of the would-pass-5 s | 32% | 50% |
| R2 | 150 (2.5 s) | 2.5 s distance against the cell-aware bar (P2) times a factor, or ml's logistic on the same features; one checkpoint, not ml's two at 2 and 3 s | 32% | 20% |
| R3 | 300 (5 s) | today's screen with the bar per arena and per cell (P2) | 26% (keeps 10% of all instead of 20%) | 10% (keeps 20%) |
| R4 | 600 (10 s) | ml's elite-relative stop, neighbourhood minimum, empty cell blocks the stop | 5% | 0% (cell tables empty) |
| full | 1,200 | archive offer | 5% | 20% |

Mean steps: mature 0.32 x 60 + 0.32 x 150 + 0.26 x 300 + 0.05 x 600 + 0.05 x 1,200 = 235. Mature with today's global 20% keep at R3 and no R4 gain: 295. Fresh: 30 + 30 + 30 + 0 + 240 = 330. Falls are not counted and lower all three. Generation 0 runs unscreened until a quarter of it is in (arm_screen_early), as today.

So the joint claim is 480 to about 300 with R1 to R3 (1.6x) and to about 235 with R4 (2.0x) in a mature population. The 235 rests on three unmeasured numbers: the R1 and R2 stop shares (ml's 40/30/10 split is a guess), the per-cell keep share at R3 (my 10% is a guess), and R4's fire rate (below). One dump of a mature 3M generation with the distance at 60, 150, 300 and 600 steps, the final distance, the final cell and the parent's cell gives all three. That dump is the gate for everything in D2 and costs one diagnostic env var.

What would break the schedule: a class of honest movers that stands still for 2 s and then walks (slow rhythm near the period cap). ml's per-cadence-band calibration makes that loss visible; the audit lane (1% of slots with every rule off, chosen by hash) measures it every generation. I accept ml's censoring attack on my per-cell bar: the bar is built from elites' own 5 s distances, which are uncensored (every elite ran full), but its miss rate is only observable on creatures it did not stop, so it needs the audit lane as ml's fitted rules do. Today's 5 s screen has the same blind spot and nobody has measured it since the 0.887 Spearman at its introduction. The audit lane is worth building before any new rung.

### Attack on ml's elite-relative stop at 8 and 12 s

ml says the rejected "stop once it cannot beat its cell" is different because the cell is predicted and the prediction error is measured. Two of the four axes are safe at 8 s: ground contact is a running mean and height is a running mean in log bins. Cadence is turns per second and is stable after a few cycles but sits on bin edges. Feet is the exact count of distinct nodes that touched the ground, binned one per node. It only grows, and the elites that matter are hoppers airborne 40 to 67% of the time (anatomy-operators.md) that can pick up a new contact node in a late stumble. So the prediction error is concentrated on one axis and ml's radius-one neighbourhood covers it. I accept the mechanism.

I attack the size. The stop fires when d(t) x r_t is below the minimum record over the 81 neighbouring cells (four live axes). Two things make it rare:

1. r_t is the 99.9th percentile of final / d(t) among survivors. For a steady gait that ratio is 20/8 = 2.5 at 8 s. The 99.9th percentile includes late starters and is plausibly 4 to 6. Then the stop needs d(8) < record_min / 5, which is a creature on pace for less than half the weakest neighbouring record.
2. Records vary about 10x across a mature grid, so the 81-cell minimum is a weak cell's record, typically a third or less of the parent's own.

A survivor is by definition in the top 20% at 5 s, and CMA and gentle children run at 0.5 to 1.0x their parent's pace. A survivor on pace for under a sixth of its parent's record is uncommon. My estimate: 10 to 30% of survivors stopped, not 80%, saving 0.2 x 0.2 x 600 = 24 steps per creature (5%), not 106 (22%). The number that has to be true for ml's 106: at 8 s, 80% of survivors have d(8) x r_8 below their 81-cell minimum with r_8 at the 99.9th percentile. The dump answers it in one pass. If r_8 must be pulled down to the 99th percentile to make the rule fire, the loss is 1% of survivors, which is 0.2% of creatures and about the whole archive entrant rate. That is the trade the rejected 10 s rung lost.

Where the cell information pays is R3, not R4: at 5 s the whole population is still there, and replacing a global cut with a cell-local one changes who is kept rather than adding a second cut on the already-selected 20%. That is why P2 sits at 20 to 25% and R4 at 5%.

Two agreements with ml: a creature stopped by R4 is treated exactly as a screened one (no archive, CMA tell with the truncated distance), and the island optimizers' children are exempt from R4, since `tell_optimizing` ranks by distance alone and an 8 s ranking would replace its 20 s ranking. They are 9% of creatures, so the exemption costs 0.09 x 0.05 x 600 = 3 steps.

### physics's cheap-fidelity screen and igpu's 2 s rung, against this schedule

They do not multiply with R1 and R2. Under the mature schedule the first 5 s cost 145 of 235 steps. One substep there saves about 72 and restarting survivors at full fidelity costs 30, so the net is 235 to 193 (1.2x), not 1.25x on 480. Measure it (physics's 262k both ways) after R1 to R3 exist; the number that must hold is that the cheap 5 s ranking keeps at least 98% of the full-fidelity top 1%. igpu's 2 s rung at keep 50% and gpu's "screen at 3 s" are R2.

## D3. Random streams and the gaussian

Is a changed stream a new seed? Yes, and I would sign that in `docs/design-decisions.md`. The owner's rule is one seed, one search, on one GPU. It says nothing about which search a seed names. Two conditions keep the search the same search in distribution, and those are what I hold the RNG proposals to:

1. Each draw keeps its distribution: uniform stays uniform on [0, 1), the gaussian stays zero mean, unit variance, symmetric.
2. Draws are independent across slot, gene and draw index. A counter-based hash keyed by (seed, generation, breed round, slot, gene, draw) gives that by construction and removes the order dependence of today's sequential `Rng` per slot, which is an improvement: two implementations that draw genes in a different order still make the same child.

A sum of 12 uniforms (Irwin-Hall) has unit variance exactly, so the CMA covariance and path statistics stay unbiased; its only deviation is thin tails: nothing beyond 6 sigma and about half the gaussian density at 4 sigma. Of the 5 x 10^8 gaussians a generation draws, the gaussian puts about 30k beyond 4 sigma and Irwin-Hall about 15k, every one clamped by the gene's range anyway. CMA-ES is rank based and works with bounded and even uniform mutation distributions; the path length normalisation shifts by under 1%. Verdict: a sum of 8 or 12 uniforms in fixed point does not change the search in any way `search_ab` could see, and I prefer it to a polynomial Box-Muller with pinned FMA because integer arithmetic is bit-equal on every device without a contraction rule. Cost: 12 integer adds per gaussian. Measurement that closes it: `search_ab --gpu`, 10 seeds, 40 generations, equal evaluations, old stream against new: best distance and QD within the seed spread. `qd::VERSION` bumps.

## D5. Body size by generation

Measured this round from scratchpad/warp/save42.evo (version 42, generation 51, muscle mass in for the whole run) with a throwaway example in a version-42 worktree. The other saves are versions 38, 40 and 41 (generations 119, 400 and 10) on earlier physics, and the current code turns them down.

Global archive elites (nodes mean / p50 / p90, muscles mean / p50 / p90, best distance):

| generation | elites | nodes | muscles | best m |
|---|---|---|---|---|
| 0 | 117 | 4.6 / 5 / 5 | 4.6 / 5 / 6 | 0.0 |
| 10 | 1,345 | 6.9 / 6 / 11 | 10.6 / 9 / 18 | 16.3 |
| 20 | 1,367 | 7.5 / 6 / 13 | 13.9 / 12 / 24 | 34.7 |
| 30 | 1,373 | 8.5 / 7 / 16 | 17.1 / 14 / 34 | 37.2 |
| 40 | 1,380 | 8.6 / 7 / 12 | 18.2 / 17 / 26 | 40.4 |
| 50 | 1,381 | 8.6 / 8 / 11 | 19.2 / 19 / 28 | 42.7 |

The top 300 at generation 51: nodes 7.9 / 8 / 9, muscles 21.3 / 21 / 26.

The ring's 786,432 offspring at generation 51, which is what the GPU runs: nodes 8.7 / 7 / 12, muscles 19.2 / 17 / 29. The node histogram has a second hump: 70% of offspring have 8 nodes or fewer, 21% have 9 to 16, and 9% have 17 to 32 nodes (clusters at 25, 29 and 30 nodes of about 10k each) although no elite in the top 300 has more than 9. In the lane-group kernel those run as 32-lane groups. Weighting by lanes (8, 16, 32 per class), the 9% of bodies at 17 nodes and up take about 24% of lane-steps and the 30% at 9 nodes and up take about 53%. Per-lane designs weight by nodes and get a similar split. So the plan is judged at mean 8.7, p90 12, and a 9% tail that costs a quarter of the GPU.

What holds bodies down now: the physics. Elite mean nodes rose from 4.6 to 8.6 in 30 generations and then stopped; the p90 fell from 16 to 11 and the p90 muscle count from 34 to 27 between generations 30 and 50. Muscle mass and the mass-scaled muscle force are doing what they were put in for, at about 9 nodes and 19 muscles, not below 9. Nothing holds offspring. The 17 to 32 node tail comes from the growth operators (copy a branch, twin a limb, repeat a trunk segment, graft) applied to already large parents, the morphology reserve and immigrants, and from generation 30 on they are almost all screened.

The search-side rule I would propose, with no fitness term: a body limit that follows the archive. The genome already has fixed limits (32 nodes, a bone length cap). Make the node cap for a structural child on an island the largest node count among that island's behaviour elites plus two. It is a body limit, like the bone length cap, not a score. It keeps every operator, only bounds how far a child may outgrow anything that has ever worked on its island, and it lets giants appear when a giant has entered the archive. Expected: the 17-plus tail falls from 9% to under 2% of offspring, which is 15 to 20% of lane-steps at generation 50. Measurement: the histogram above from a save at generation 50 with and without the rule, and `search_ab --gpu` over 10 seeds at equal time for best and QD, because the rule could cost the rare large winner. It is the owner's decision.

Rate consequence for everyone's estimates: the generation 0 to 3 rates (92k to 167k/s) were at mean nodes 5.8 to 6.9. At mean 8.7 with the tail, expect about 0.6x of the generation 3 rate on today's kernel, so about 90 to 100k/s at generation 50 before any change.

## D7. Are data and I proposing the same ring?

No, and I concede the block size. What the emitters want is not a block size but two things: a tell every 100 to 500 samples per emitter, and a short lag between a tell and the children bred from it. My 16 blocks of 49k was a way to get the first through block size, and it conflicts with my own P3: at 1,024 emitters and a 49k block a CMA emitter gets 17 samples per block, which is worse than today. The right form: each emitter tells when it has accumulated at least 200 samples across blocks, in ring order, independent of block size. That is deterministic (accumulation order is absorption order) and it makes the block size the GPU's choice. So data's 200 ms blocks are fine.

The ring depth is where we differ. data wants 2 s of ring. At 2M/s that is 4M creatures bred from emitter states 1 to 2 tells old, and a world change discards 4M creatures and makes the player wait 2 s. Asynchronous CMA-ES tolerates a delay of a few tells, so the search survives 2 s but gains nothing from it. I ask for 0.5 to 1 s of ring; data's own P1 to P3 put the per-block CPU work at 30 to 60 ms, which keeps a 1 s ring full. If it cannot, the fix is on the host, not a deeper ring.

## What I still back and what breaks it

- P1 bar table: breaks on nothing; it is the mechanism for R2 to R4 and for physics's cheap-fidelity screen if that survives.
- P2 per-arena and per-cell bars at R3: breaks if the dump shows entrants below 0.7 times the neighbourhood minimum at 5 s in more than 4% of cases; then the factor drops and so does the gain.
- P3 emitter restructuring, restated: tells per 200 accumulated samples per emitter, CMA_LIMIT 1,024, optimizers on 4 designs per island at once. Breaks if `search_ab` at equal evaluations shows no gain over 96 emitters, which would mean the landscape, not the emitter, limits progress.
- P6 GPU-side parametric children: breaks if the two samplers cannot be made bit-equal, which the integer gaussian in D3 exists to prevent.
- The body limit that follows the archive (D5): breaks if a seed's best comes from a body larger than any elite plus two; `search_ab` at equal time decides, and the owner decides before that.
- Withdrawn: 16 blocks of 49k as a requirement.
