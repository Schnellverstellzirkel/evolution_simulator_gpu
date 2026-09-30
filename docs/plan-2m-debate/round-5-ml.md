# Round 5, ML: convergence

## The design I back

The converged pipeline with all thirteen rulings: CPU breeding into pinned SoA arenas with experiment-constant block size and ring depth (ruling 4), DMA plus a prologue unpack kernel (2), VRAM-resident queue tail and drain flag (3), the three bucket rules (5), registered host memory with whole-block re-run on a context loss (6), the per-lane kernel with data's 3 confirmation slots and 1 replay slot (1), the substep ladder with L2.5 as the primary target (8), the growth-step rule as the owner's candidate (9), the ladder R1 to R4 with nurseries and immigrants exempt from R1 and R2 at first and R4 floored at the emitter's own median d(10) (10), the host tax carried at 15 to 21% until measured (11), and the small items including `feet()` on `lift_lo`, the 128 KB histogram window, `physics_audit` on every new island record and the 4x re-run of audit-lane entrants (12). I reject no ruling. My part of the design is unchanged from round 3 except where rulings 10 and 12 amend it: the audit lane at 1 in 128, the discriminants at R1 and R2 fit on the audit set, R3's factor and r4 calibrated on it, the per-band breaker, the parameters frozen into each block's Config, the fit at the generation boundary in absorption order.

## Tracks, final form

Each is one coding agent, one to two weeks, worktree `/home/amipo/workspace/evolutionSimulator-<track>` on branch `claude/<track>`.

1. `rung-dump` (with ga; one agent, ga's spec for the row, mine for the tool). Owns `src/creature_kernel.rs` (the seven reinterpreted `GpuResult` words as fp16 pairs), the metrics block of `shaders/warp_creature.cu` (write d60, d150, d300, d600, the R1 and R2 feature words and the ended step; no stopping), `src/physics2.rs` (the prototype writes the same seven words), `src/scheduler.rs` (`feet()` on `lift_lo`), the `EVOLUTION_RUNG_DUMP` writer in `src/storage.rs`, and `examples/rung_replay.rs`. Deliverable: the dump env var and the tool that fits a candidate ladder on half the rows and prints steps per creature, stop share per rung, entrant misses per 10k per rung and per cadence band, the share of the final top 1% and 10% kept, R4's fire rate at the calibrated r4 with the emitter-median floor, Spearman of d600 against final, and the 5 s, 10 s and 20 s cell match. Gate: fitness bit-identical on a 262k wave and the fixed-seed history test unchanged; one generation-51 dump exists and the tool prints its table. Dependencies: none; it runs on the lane-group kernel. Order 1.

2. `audit-lane`. Owns the audit hash in `src/storage.rs` (`Birth` flag, the per-generation audit set, the 8-generation histogram window in `Experiment` and its save), the flag in the lane record head in `src/warp_kernel.rs`, the audit-entrant confirmation in `verdict` (ruling 12), the stage-log line (steps per creature, audit rows, mean and p90 nodes and muscles), and `qd::VERSION`. Deliverable: audit creatures flagged and recorded, their entrants re-run at 4x before commit, the fit scaffolding with nothing to fit yet. Gate: the history changes only by the added confirmations (new VERSION), saves grow under 200 KB, the log line prints. Depends on 1. Order 2.

3. `rung-r1`. Owns the step-60 checkpoint in `shaders/warp_creature.cu`, the rung fields of `Params` in `src/warp_kernel.rs`, the rung parameters in `src/config.rs`, and the Fisher fit, threshold search and band breaker in `src/storage.rs` (`end_generation`). Deliverable: R1 live, off for nursery and immigrant slots and for audit, confirmation and replay creatures. Gate: `rung_replay` at 1e-3 keeps 100% of the top 1% and at least 96% of the top 10% with at most 30 entrant misses per 10k; then `search_ab --gpu`, 10 seeds, 40 generations, equal wall time, best distance and QD not below control; live misses at most 30 per 10k. Depends on 1, 2 and the dump showing R1 stops at least 20% of creatures at that budget. Order 3. The per-lane kernel gets the same checkpoint when gpu's kernel lands (about 40 lines in gpu's track).

4. `rung-r2-r3` (with ga's bar-table track, which owns the table and the elites' stored rung distances). Owns the step-150 checkpoint, R2's seven-feature discriminant with the neighbourhood bar as input, and R3's factor calibration on the audit set. Gate: as track 3, plus nursery `kept` per cohort not falling against control. Depends on 3 and ga's table. Order 4.

5. `rung-r4`. Owns the host-built 54-cell neighbourhood-minimum table per arena (52 KB per block), the in-kernel predicted cell at step 600 (shares data's descriptor binning code), r4's calibration, the emitter-median floor as one float per CMA emitter per block and a bar index in the lane record, the optimizer exemption bit, and the `stale()` guard. Gate: fires for at least 30% of survivors at the calibrated r4 with the floor, else the track is deleted; `search_ab` at equal time as above; the median of final / d600 among entrants within the seed spread of control. Depends on 4 and data's candidate-bit code. Order 5.

6. `rung-parent-profile` (only if track 1's dump says R1 is worth building). Owns the four rung distances in `Elite` and two more R1 and R2 features. Gate: `rung_replay` stop share at equal miss rate at least 5 points higher; otherwise deleted. Depends on 3. Order 6.

7. Nursery per-arena fit, in track 3's files, opened only when the dump shows nursery entrants stopped under 0.1% in every band at a 1 in 10,000 budget (ruling 10). Not scheduled until then.

## Three measurements that would change my mind

1. `rung_replay` on the generation-51 dump: if R1 at the 1 in 1,000 budget stops under 20% of creatures (I claim 40%), the early rungs are worth under 0.15x and R1 folds into R2 as one checkpoint at 2.5 s, which removes track 3's kernel work.
2. Spearman of d600 against final among survivors under 0.9, or the 10 s to 20 s cell match under 75%: R4 is dropped and survivors keep their 1,200 steps; the mature mean rises by about 25 steps.
3. `search_ab --gpu` at equal time over 10 seeds with the ladder on: QD below control by more than the seed spread while best distance holds. That means the ladder biases what survives, and the budget tightens to 1 in 10,000 for every rung or the offending rung is deleted, whatever the steps it saved.

## Honest ceiling

Steps per creature at generation 51 with rulings 10 and 12: 240 to 260 (my shares 228 plus the nursery and immigrant exemption, ga's 245 plus the same), against 300 with R1 to R3 only. At generation 100 the step count is about the same; the cost per step is what rises.

| case | generation 51, 3T | generation 51, 5T | generation 100, 3T | generation 100, 5T |
|---|---:|---:|---:|---:|
| 2 substeps, 250 steps, host tax 18% | 0.98M | 1.48M | 0.74M | 1.10M |
| L2.5 at 1.6x | 1.57M | 2.36M | 1.18M | 1.77M |

Three assumptions set it: the chair's kernel range of 300M to 450M creature-steps/s at 2 substeps for the generation-51 mix; a 0.75x muscle slope by generation 100 with the growth rule in (0.65 to 0.82 without it); and the host tax at 18% until os's rows exist (at 5 to 8% every number rises by 12%). So 2M/s sustained at generation 51 needs L2.5 and the 5T budget, or L2.5 at 3T with the host tax down to 8%. At generation 100 it needs all three, and even then lands at 2.0M with no margin. With 2 substeps it is not reached on this laptop at either budget. That is the honest number from my seat, and the ladder is the one lever in it that holds whatever the power budget does.

## Ruling 10

I agree. Exempting nurseries and immigrants from R1 and R2 costs about 12 steps per creature and removes the only failure mode of the ladder that the audit lane cannot see quickly enough (a cohort with a 10-generation window and a 3-generation breaker), and the per-arena fit is the right way back in once the dump shows nursery entrants stopped under 0.1% in every band. The emitter-median floor on R4 answers the sprinter drift by construction, since a stopped sample was in the bottom half of its own emitter's ranking at 10 s and could not have reached the top half that drives the update, and it halves my fire-rate claim to about 50% of survivors, which the dump will confirm or cut further. The `stale()` guard is already in the code, so nothing new is built for it.
