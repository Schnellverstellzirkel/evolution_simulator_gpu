# Round 2, ML: restated numbers, the joint schedule, censoring, host bytes

## Corrections first

The chair is right about settle. `warp_kernel.rs` line 145 subtracts `fidelity.settle()` from the screen tick and `Params.steps` is the trial length, so the kernel runs 300 steps for a screened creature and 1,200 for a survivor. My proposal 3 is withdrawn. Every number below is on the 480 base (0.8 x 300 + 0.2 x 1,200, before falls). Falls lower the base for every domain equally, so I leave them out as the chair did.

Restated on 480:

- Proposal 1 (calibrated stops at 1 s, 2 s, 3 s): the absolute saving is unchanged at about 130 steps per creature (0.8 x (0.4 x 240 + 0.3 x 180 + 0.1 x 120)). Mean 480 to 350, 1.37x instead of 1.3x.
- Proposal 2 (elite-relative stop for survivors at 8 s and 12 s): 106 steps per creature unchanged (0.2 x 0.8 x 660). Mean 350 to 244 with proposal 1, 1.97x together.
- Nothing else in my file used the settle count.

The joint schedule with ga below replaces both numbers, because ga's per-cell 5 s bar changes the survivor share that my 8 s and 12 s stops act on, and ga's 2.5 s rung is the same lever as my early checkpoints.

## D2. The joint steps-per-creature schedule (my version)

One ladder, five rungs, each with the predictor that decides it. ga's 2.5 s rung and my 1, 2, 3 s checkpoints merge into two early rungs. ga's 10 s rung and my 8 and 12 s stops merge into two late rungs that use the creature's own running descriptor instead of the parent's cell.

| rung | step | predictor | what it cuts |
|---|---|---|---|
| 1 s | 60 | calibrated score on running state (`com_x`, mean forward speed over the last 0.5 s, contact count, `head_shake`, mean muscle energy, shortest muscle period), threshold set so at most 1 in 1,000 eventual 5 s passers is stopped, from the audit lane | creatures that do not move |
| 2.5 s | 150 | d(2.5) against a per-neighbourhood bar: the elites' stored 2.5 s distance (ga's P2 storage) divided by the calibrated ratio quantile r_2.5 (mine), the arena bar for structural children with no predicted cell | slow movers that cannot reach their neighbourhood |
| 5 s | 300 | ga's per-arena and per-cell bar, the factor calibrated on the audit lane instead of a fixed 0.7 | the current screen, made local |
| 8 s | 480 | elite-relative: predicted cell from the running descriptor, plus one-step neighbours on each axis, bar = min over them of elite final / r_8, empty neighbour blocks the stop | survivors that cannot beat their neighbourhood |
| 12 s | 720 | same with r_12 | the rest of them |

Keep fractions are outputs of the calibration, not inputs. That is the one place I differ from ga's P5, which fixes keep 0.5 at 2.5 s and 0.5 at 10 s. A fixed keep is either too loose in a fresh population or too tight in a mature one; a calibrated threshold with a fixed miss budget adapts every generation and its loss is a measured number. For the estimates I still need keep fractions, so here they are, as guesses to be replaced by one offline replay:

Fresh population (generations 3 to 10, a bar exists, archives sparse, most neighbourhoods have an empty cell): 35% cut at 1 s, 25% at 2.5 s, 20% at 5 s, survivors 20%, of whom 20% stop at 8 s and the rest run full. Mean: 0.35 x 60 + 0.25 x 150 + 0.20 x 300 + 0.20 x (0.2 x 480 + 0.8 x 1,200) = 21 + 37.5 + 60 + 211 = 330 steps. 1.45x.

Mature population (generation 30 and later, full archives, per-cell bars biting, ga's kept share 5 to 10%): 40% cut at 1 s, 25% at 2.5 s, 27% at 5 s, survivors 8%, of whom 85% stop at a mean of 9 s (540 steps). Mean: 24 + 37.5 + 81 + 0.08 x (0.85 x 540 + 0.15 x 1,200) = 24 + 37.5 + 81 + 51 = 194 steps. 2.5x.

Mature, conservative ends of both our guesses (30% at 1 s, 25% at 2.5 s, 35% at 5 s, 10% survive, 60% of them stop at 9 s): 18 + 37.5 + 105 + 0.10 x (324 + 480) = 241 steps. 2.0x.

So the joint schedule is 1.45x fresh and 2.0 to 2.5x mature. Since sustained rate is judged at maturity, the kernel's target for 2M/s is 2M x 194 to 241 = 390 to 480M creature-steps/s, 9 to 11x today's 43 to 45M, not 25x. In a fresh population it is 660M, 15x, but a fresh population lasts minutes. This is also the one lever os's power argument does not discount: a step not run is a joule not spent, so it counts fully under the cap.

Where ga and I still differ, for the chair: ga's late rung compares against the parent's cell, mine against the creature's measured cell at 8 s. By 8 s the descriptor is a running mean over 480 steps and the parent's cell is a prior; the measured cell should be the better predictor and the offline replay will say so (share of survivors whose 8 s cell equals the 20 s cell; I expect 85 to 90%). And ga's floor is right: every entrant needs its 1,200 steps, so the floor is about 6 steps per creature and everything above it is the price of prediction. On the rejected 10 s rung that gpu quotes at "1.3x with every top-1% kept": both sources are true. hpc.md section 5.6 measured the recall (100% of the top 1%, 86% of the top 10%) and rejected-ideas measured the QD loss end to end. A global rung keeps the fastest and loses cells. That is exactly why the late rungs above never stop a creature whose neighbourhood has an empty cell, and why the audit lane reports misses as entrants lost, not as top-1% lost.

## D2. The censoring attack on ga's per-cell bar, and the fix

ga's P2 sets a cell's bar from its elite's own 5 s distance times 0.7. Two things are wrong with where that number comes from.

First, the elite passed the screen, so its 5 s distance is at or above the bar it faced. Every elite in the archive is a passer, and passers are the creatures with a high 5 s distance relative to their final distance. The ratio final / d(5) estimated on elites is therefore biased low, and a factor tuned on it is too tight for the creatures it matters for: the ones that are slower at 5 s and faster at 20 s. Second, once the bar is set, a creature stopped by it never shows its final distance, so the archive never learns the bar was wrong. The rule cannot observe its own error. This is the same mechanism that made the rejected second rung lose QD while keeping the top 1%: the fast-start cells kept their elites and the slow-start cells stopped filling.

The concrete failure: a cell whose elite starts fast and plateaus blocks every lineage that starts slow and finishes fast, and nothing shows it except a QD number that stops rising. The 0.7 is a guess with no feedback.

The audit lane fixes both parts for ga's rule as well as mine. One percent of the slots of every block, chosen by a hash of slot and breed round, run with every rung off. They give 30k uncensored (d(1), d(2.5), d(5), d(8), d(12), final, final cell) rows per 3M generation, across all cells and all emitters, and the calibration of each rung is done on those rows only: the factor at each rung is set so that among audit-lane creatures that would have entered an archive, at most 1 in 1,000 would have been stopped. ga's 0.7 becomes a measured number per rung and per generation. And the audit lane leaks the right way: an audit creature that would have entered does enter, because it ran the full trial and is a legitimate result. So a slow-start lineage that the bars stop still gets 1% of its children through, and once one is an elite its 5 s distance lowers ga's cell bar and its ratio raises my r_t. The rules correct themselves through the archive. Cost: 1% of creatures at 1,200 steps, about 3% of all steps at maturity, less than the 8 to 12 steps per creature it protects against losing in QD.

The audit lane does not fix the nursery starvation ga found (F1); ga's per-arena bars do, and that finding stands on its own.

## D2. Host bytes: I need summaries, not features

Data's results stay at 80 B and the descriptor is computed on the device. Two answers.

Steady state: no per-creature features reach the host. The rung decisions run in the kernel on state it already holds. The calibration needs, per rung, a joint histogram of (distance at the rung, final distance) from the audit lane and a histogram of final / d(t) among survivors: 64 x 64 bins per rung as 32-bit integer counts, five rungs, about 100 KB per generation, reduced on the device with integer atomics (order-independent, so deterministic) and read back once per generation. The threshold search over a histogram is a few thousand comparisons on one CPU thread. No float atomics, no feature rows.

The 80 B record has room anyway. `Result` holds 19 floats; `contact_hi`, `lift_hi` and `ground_hi` are always zero at 32 nodes at most, and `previous_center_y`, `vertical_extremum`, `vertical_trend` and `gait_turns` are working state that `scheduler::to_metrics` never reads. That is 7 floats, enough for five rung distances as fp16 pairs plus a rung-stopped code, at no size change. Only the audit lane's rows are worth sending as rows (30k x 48 B = 1.4 MB per generation, through data's candidate list path), because they are the uncensored labels.

The score at the 1 s rung is a linear function of six features. I fit it as a linear discriminant from class means and a 6 x 6 covariance, which are sums, so they come out of the same integer-atomic reduction. A logistic fit would need rows; it is not needed at this feature count.

## What would break each proposal I still back

Rung at 1 s. Breaks if fewer than 25% of the 5 s losers are separable at 1 s at a 0.1% miss rate. In a mature population most children are children of elites and most of them move, so the immobile share may be lower than the 40% I guessed. The number that must be true: at least 25% of eventual screen failures have `com_x` under 0.05 m and non-positive speed at 1 s, with under 0.1% of eventual passers in the same region. Also breaks if a cadence band (long-period gaits that catch late) shows a miss rate above 1% on the audit lane; then the rung is turned off for that band.

Rung at 2.5 s. Breaks if d(2.5) ranks the final distance much worse than d(5) does (Spearman under 0.75 against 0.887 at 5 s). ga asked physics how many gait cycles 2.5 s covers; that number decides it.

Rungs at 8 s and 12 s. Breaks if more than 15% of survivors change cell between 8 s and 20 s, because the neighbourhood then has to widen and an empty neighbour blocks most stops. Breaks if late accelerators (final / d(8) above r_8) are the entrants rather than the losers; the audit lane shows that as misses per generation. And breaks the CMA optimizer's ranking if all its children are stopped at 8 s; the fix is to exempt the optimizer's children (about 9% of creatures, per ga's F3), and the measurement is the optimizer's record trajectory in `search_ab`.

Audit lane. Cannot break the search; it costs 3% of steps.

Proposal 5 (SAIL-style best-of-2 breeding, now merged with ga's P7). Breaks if the surrogate's Spearman with the 5 s distance is under 0.3 on the next generation. ga owns the train-on-g, test-on-g+1 measurement; I supply the model (a 32-feature linear or 2-layer model trained on the CPU in under 0.2 s per generation, per cpu's P9 numbers, which I accept).

## Answers to points from other files that touch my domain

physics P3 (cheap-fidelity screen). The measurement physics proposes is the right one and it is cheap: score one 262k wave at 1 substep and at 2, compare the 5 s rankings. If the 1-substep screen keeps 99% of the 60 Hz top 1% and 95% of the top 10%, my exploit objection is void for the screen, because exploits that pass a coarse screen only get a full-fidelity trial they then lose. It also stacks partly with my rungs: under the mature schedule the pre-5 s portion is 0.40 x 60 + 0.25 x 150 + 0.27 x 300 = 143 of 194 steps, so running it at half cost saves another 70, 194 to 124 steps. I would run the measurement, since it decides 1.5x on top of the ladder.

igpu's 2 s rung and gpu's "screen at 3 s": the same lever as the 1 s and 2.5 s rungs. One ladder, one calibration.

cpu P9: agreed that CPU inference of a small model is free. Only the breed-time surrogate needs it.

## D7. Proposal 6: park it

Park it, with one condition: physics's first measurement (the realized friction-work ledger on the 300 best elites from replays, no kernel change) also tells whether exploits sit in archive cells now. If that ledger shows archived elites with positive friction work above 1% of muscle work, the routing comes back as a track, because confirmations at 1.6% of GPU time (ga's count) are the cheapest place to spend on the spirit. Until then it spends GPU on a problem nobody has measured.

## What I need this round

From ga: agreement on the ladder above or a marked-up version, and the nursery `sent` and `kept` numbers from a mature save. From data: confirmation that a 100 KB per-generation histogram readback and a 30k-row audit list fit the candidate path. From physics: the cell-stability numbers (8 s cell against 20 s cell per axis) and gait cycles in 2.5 s, from one recording of a 262k wave with the five rung distances written into the free `Result` floats. From the chair: the offline replay tool over one saved generation is the gate for the whole ladder, both ga's rungs and mine, and it is a day of work; it should be the first track scheduled in D2.
