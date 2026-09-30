# Round 3, ML: the rung ladder as a specification

The pipeline is the chair's converged one: CPU breeding into pinned genes, take-up unpack on the RTX, the per-lane kernel, results in pinned host memory absorbed per block in ring order, candidate bit in the kernel, CPU commit. My part is the four rungs, the audit lane, their calibration, and the tool that gates them, written so a coding agent can build it without me.

## Where the design lands

Row "2 substeps, R1 to R4" at my R4 rate: 1.3 to 1.9M/s before the host tax, 1.2 to 1.8M/s after 5 to 8%. If 1 substep passes its honesty test, row 4: 2.3 to 3.4M/s, so 2M/s with margin. If 1 substep fails and the budget is 3T, the honest number is 1.2 to 1.4M/s with the full ladder, and 2M/s is not reached on this laptop. The ladder is the one lever whose gain is the same in joules and in steps, so it holds at either power budget.

## The steps lever at generation 51, per rung

Cumulative mean steps per creature, base 480. Mature is generation 30 and later with full archives and per-cell bars; fresh is generations 3 to 10 with a bar and sparse archives. My column and ga's column differ only on R1, R2 and R4 shares, and the dump (gate 1 below) replaces both.

| lever | mature, my shares | mature, ga's shares | fresh |
|---|---:|---:|---:|
| base | 480 | 480 | 480 |
| R1 at 60: stop share 40% / 32% / 35% | 384 | 403 | 396 |
| R2 at 150: 25% / 32% / 25% | 347 | 355 | 359 |
| R3 per-cell bar: survivors 20% to 8% / 10% / 20% | 239 | 265 | 359 |
| R4 at 600: 85% / 50% of survivors / 20% | 198 | 235 | 335 |
| audit lane 1% at full length | 208 | 245 | 344 |
| optimizer children exempt from R4 | 211 | 248 | 344 |

Creatures per second at the chair's kernel ranges (300M at 3T, 450M at 5T for 2 substeps; 550M and 800M for 1 substep), before the host tax:

| steps | 2 substeps 3T | 2 substeps 5T | 1 substep 3T | 1 substep 5T |
|---:|---:|---:|---:|---:|
| 211 | 1.42M | 2.13M | 2.61M | 3.79M |
| 248 | 1.21M | 1.81M | 2.22M | 3.23M |
| 344 (fresh) | 0.87M | 1.31M | 1.60M | 2.33M |

The fresh row lasts minutes; sustained is the mature row.

## The rungs, as a specification

Kernel side. Every rung is evaluated once, at a fixed step, by the group that runs the creature, on values it already has in the metrics block of `warp_creature.cu` (`com_x`, `center_y`, `contact_bits`, `head_shake`, `step`, the muscle energy stores `s_en`). Rung parameters arrive in the block's `Params` (a struct of 40 floats) and two tables in global memory, and are constants for the block. A stop writes the result exactly as the screen does: `fitness = com_x`, `screened = t + DT`, `screen_x` unchanged, and a rung code. No atomics, no new shared memory, one 32 B read of the tables per rung. Creatures with the audit bit, the optimizer bit (R4 only), confirmation trials and replays skip every rung.

Two running values are added per group in `s_mt`: `x30` (`com_x` sampled 30 steps before each rung, so the speed over the last 0.5 s is `com_x - x30`) and `touched`, the popcount of `contact_bits` (already kept). Both are one register-free shared write per step.

R1 at step 60. Features, six floats: f1 = `com_x`; f2 = `com_x - x30`; f3 = `touched / nn`; f4 = `head_shake`; f5 = mean muscle energy store (a group sum of `s_en` over the muscle lanes, divided by `nmus`); f6 = the body's shared rhythm period (all muscles share one clock after `repair`, so `f1.y` of muscle 0). Score s = w1·f + b1, seven floats in `Params`. Stop if s < 0. The features are scaled on the host into w and b, so the kernel does one dot product.

R2 at step 150. Same six features with f1 and f2 taken at 150, plus f7 = the neighbourhood bar at 2.5 s from ga's table (the minimum over the parent's cell and its axis neighbours of the elites' stored 2.5 s distance, one float per creature in its lane record, the arena bar when unknown). Score s = w2·f + b2, stop if s < 0. R2 is also a linear discriminant, not a bar with a factor: the bar is one of its inputs, so ga's factor and my calibration are one fit.

R3 at step 300. ga's per-arena and per-cell bar, with its factor calibrated as in the next section. The kernel side is ga's P1 table load, unchanged.

R4 at step 600. The group bins its running descriptor exactly as the kernel's candidate bit does (ground contact over 600 steps, `gait_turns` per second, mean height over 600 steps, feet from `lift_bits`). Neighbourhood: the cell and its plus-or-minus-one neighbours on the ground-contact, cadence and height axes; on the feet axis the current bin and the next one up, because feet only grows. That is 3 x 3 x 3 x 2 = 54 cells. The kernel reads one float per cell from a per-arena table of neighbourhood minima, precomputed on the host at breeding: `nmin[arena][cell]` = min over the 54 cells of the elite's final distance, negative infinity if any is empty. Nine arenas x 1,440 cells x 4 B = 52 KB, L2 resident. Stop if `com_x * r4 < nmin` and `nmin` is finite. `r4` is one float in `Params`. Optimizer children carry a bit in the lane record and skip R4.

## Calibration, on the host, once per generation

Data. The rows are the audit-lane results, which reach the host like every other result (results are in pinned memory per block). The five rung distances and the R1 and R2 feature vectors travel in the seven free floats of `Result` (layout below). One thread walks the absorbed blocks of the generation in ring order and, for each audit-lane row, appends it to the generation's audit set: about 30k rows per 3M generation, 64 B each, 2 MB. Everything below is computed from the audit set, never from rules-on creatures, which is what removes the censoring: an audit creature ran the full trial whatever its start looked like.

Labels. Each audit row gets three labels from its own full trial and the archive as it stood when its block was absorbed: `pass3` (its 5 s distance beat its own per-cell bar), `entrant` (it entered an archive, or would have: final distance above its final cell's record at absorption), and its final cadence band (the cadence bin of its final cell, 8 bands).

The discriminant for R1 and R2. Fisher's linear discriminant on the pooled audit set of the last 8 generations (a ring of 8 generation sets, 16 MB): class A is `pass3`, class B the rest; w = Σ⁻¹(μA − μB) with Σ the pooled within-class covariance (6 x 6 for R1, 7 x 7 for R2), sums accumulated in f64 in row order. A 7 x 7 solve is nothing. The direction w is then fixed and the threshold b is searched separately.

Threshold search. For each rung, the miss budget is 1 in 1,000 of the class the rung must not touch: for R1 and R2 the `pass3` rows, for R4 the rows that reached 600 and are `entrant`. Sort those rows' scores; b is set so that floor(0.001 x count) of them fall below the stop threshold, on the conservative side of a tie. With 8 generations pooled, the `pass3` set is 20k rows at maturity and the R4 entrant set a few hundred; when it is under 500 rows, R4 uses the rows that beat their neighbourhood minimum at the final step instead (about 2k), which is the property R4 needs. R3's factor is the same search: f3 = the 0.1% lower quantile of `d300 / nmin_2.5` over entrants.

`r4` is the 99.9th percentile of `final / d600` over the pooled audit rows that reached 600, so the stop needs the creature to be below the neighbourhood minimum even if it accelerated as much as the top 0.1% did.

Per-cadence-band miss report. After fitting, the rules are replayed on the current generation's audit rows (not the pooled ones), which is an out-of-sample check since the pooled window lags by a generation. For each rung and each cadence band: the number of `entrant` rows the rule would have stopped, per 10k audit rows, and the same for `pass3` rows. If a band's entrant misses exceed 100 per 10k for 3 generations running, that band's threshold for that rung is set to negative infinity (the rung is off for that band) until it recovers for 3 generations. Bands are read from the rhythm period feature, which is the kernel's proxy for the final cadence. This is a rule, not a knob.

Device fallback. If per-creature results ever stop reaching the host, the same fit runs from device-reduced statistics: per rung, a 64 x 64 histogram of (score, final distance) over audit rows and a 256-bin histogram of `final / d(t)`, accumulated with 32-bit integer atomics (order-independent, so deterministic), 100 KB per generation, thresholds read off the bins on the conservative side.

Cost. The fit is one thread for about 20 ms at the generation boundary (30k rows x 8 generations, a 7 x 7 solve, four sorts of at most 200k floats). The new parameters go into the next generation's `Config` like the screen bar, so every block bred after the boundary carries them and blocks bred before keep theirs.

## The audit lane

`audit(seed, breed_round, slot)`: `splitmix64(seed ^ breed_round * 0x9E3779B97F4A7C15 ^ slot * 0xD1342543DE82EF95) & 127 == 0`, so 1 in 128 slots, 0.78%. Keyed by breed round, the audit slots move every block and every arena and emitter gets its share over a generation. The bit goes into head word `h1.y`, which the kernel never reads today, and into the block's `Birth` on the host. Audit creatures run every rung off, enter archives like any other result, and are the calibration set. Cost: 8 to 10 steps per creature at maturity, in the table above.

## The Result layout

`Result` keeps its 80 B and 19 floats. `to_metrics` reads `fitness`, `ground_contact`, `vertical_oscillation`, `gait_frequency`, `height_sum`, `contact_lo`, `lift_lo`, `fall_time`, `head_shake`, `screen_x`, `screened`. Seven floats are free: `contact_hi`, `lift_hi`, `ground_hi` are always zero at 32 nodes (`feet()` counts `lift_hi` bits; it changes to `lift_lo` only), and `previous_center_y`, `vertical_extremum`, `vertical_trend`, `gait_turns` are working state that stays in `s_mt` during the trial and is overwritten when the group writes the result. New meaning, as packed halves (fp16 pairs, 32 bits each):

- `contact_hi`: d60, d150
- `lift_hi`: d300, d600
- `ground_hi`: rung code (u16: 0 none, 1 to 4 the rung that stopped it, 8 audit bit), ended step (u16)
- `previous_center_y`: speed at 60, speed at 150
- `vertical_extremum`: touched share at 60, mean energy at 60
- `vertical_trend`: touched share at 150, mean energy at 150
- `gait_turns`: head_shake at 60, R2's neighbourhood bar

fp16 holds a rung distance under 256 m to 0.125 m, enough for calibration; fitness stays f32. `Result` and `creature_kernel::GpuResult` change together.

## The offline replay tool (gate 1 of the steps track)

`EVOLUTION_RUNG_DUMP=<path>`: with the layout above and all rungs off, every absorbed block appends one CSV row per creature: slot, arena, emitter, optimizer flag, audit flag, parent id, parent cell, d60, d150, d300, d600, final, fall step, final cell, entered flag, and the R1 and R2 features. That is ga's dump plus the features; one env var, a developer diagnostic. One generation at 3M is 3M rows, about 400 MB.

`examples/rung_replay.rs <dump> [--tolerance 1e-3] [--window 1]`: fits the ladder on the dump's audit rows (or on a random half of all rows, since with the rungs off every row is uncensored) and evaluates on the other half. It prints, for the ladder and for each rung alone: mean steps per creature before and after, the stop share per rung, entrant misses per 10k, `pass3` misses per 10k, the share of the final top 1% and top 10% kept, all of it per island and per cadence band, and a sweep over tolerances 1e-4, 1e-3, 1e-2. It also prints R4's fire rate among survivors at the calibrated `r4`, which settles the 4x between ga and me. Pass bars for the ladder at 1e-3 per rung: 100% of the top 1% kept, at least 96% of the top 10%, entrant misses at most 30 per 10k, mean steps at most 260 on the generation-51 population. A rung that fails its bar alone is dropped from the ladder before any kernel work.

## What the developer sees

One line per generation in `EVOLUTION_STAGE_LOG`: steps per creature, stop share at R1 to R4, audit rows, entrant misses per 10k per rung, and the number of cadence bands with a rung disabled. The player sees nothing new. The rungs are on when merged, and their loss is a logged number, never a control.

## Determinism

1. Every rung is a pure function of the creature's state and the block's parameters; no atomics touch the decision.
2. The parameters (w1, b1, w2, b2, f3, r4, the band masks, ga's bar tables, `nmin`) are fixed when the block is bred and travel in its `Config`, as the screen bar does today. A block keeps them whatever the timing.
3. The fit runs in `end_generation` on the audit rows in absorption order with f64 sums and stable sorts on one thread. The fitted parameters and the pooled window are saved in the `Experiment`; the window is kept as per-generation score histograms of 1,024 bins, 128 KB, so saves stay small and a load reproduces the same fit.
4. The audit hash is a function of the seed, the breed round and the slot.
5. A world change disarms every rung, exactly as `next_screen` disarms the bar: the audit rows of the old world are discarded, the window restarts, and the rungs rearm when the new world's audit set passes 8k rows, about a quarter generation, mirroring `arm_screen_early`. Blocks in flight from the old world enter no archive as today.
6. Replays and confirmation trials run with every rung off; an elite's score came from a full trial, so its replay is bit for bit its score.
7. A save between generations holds the parameters; a save mid-generation holds the previous boundary's parameters, which is the same rule the screen bar follows.

## 60 FPS and what stays out

Nothing here runs per frame: the fit is 20 ms on one thread at a boundary and the per-block work is appending audit rows. Out: neural or fitted physics (evolution exploits it), tensor cores (no dense product), per-creature host features beyond the free floats, a genome pre-skip (not an evaluation), the iGPU for inference. Proposal 6 stays parked.

## Tracks, in order, with gates

1. Dump and replay tool (shared with ga): the `Result` layout, the env var, `rung_replay`. Two days. Gate: one generation-51 dump exists and the tool prints the table; the rung shares and R4's fire rate become numbers. The game is unchanged for the player.
2. Audit lane, rung distances recorded with no stopping, the stage-log line. Three days. Gate: fitness bit-identical to the current kernel on a 262k wave, rate within 1%, `search_ab` history identical for a fixed seed.
3. R1 in the kernel with the fit and the band masks. One week. Gate: `rung_replay` passes at 1e-3; then `search_ab --gpu`, 10 seeds, 40 generations, equal wall time: best distance and QD not below the current game; live entrant misses at most 30 per 10k.
4. R3 per-cell bars and R2 (ga's table, my discriminant). One week after ga's P1 table exists. Same gate.
5. R4 with `nmin` and the optimizer exemption. One week. Extra gate: fires for at least 30% of survivors at the calibrated `r4`, else it is dropped and ga's 5% stands.
6. If physics's 1-substep passes, nothing here changes; if it fails, the cheap-fidelity screen slots in under R1 to R3 and is gated by the same tool (98% of the full-fidelity top 1% kept).

Each rung goes on by default only after its offline gate, and a rung that loses its live gate is deleted with its code.

## One new idea: the parent's start profile as a feature

Nobody has used what the archive already knows about how a lineage starts. Store each elite's own rung distances (d60, d150, d300, d600, 8 B as fp16) in `Elite`, and give R1 and R2 two more features: the parent's d60 and d150. A child of a parent that covered 0.8 m in its first second and is standing still at 1 s is broken with far more confidence than a child of a parent that took 2 s to get going; the discriminant learns the difference, and slow-start lineages are protected by the feature instead of the band mask. The stored profile also gives ga's per-cell bars at R1 and R2 for free.

Number: R1's stop share at a fixed 1e-3 miss rate from 40% to about 50% of creatures, 24 steps per creature, 10% of the mature mean; R2 a further 5 to 10 steps. Measurement: `rung_replay` with and without the two features (the dump carries the parent id; the profiles come from the run's archive), comparing the stop share at equal miss rate and the band-mask trigger count, which should fall to zero. Cost: 8 B per elite, two floats per lane record, two weights. It changes what is stopped, never what is scored.
