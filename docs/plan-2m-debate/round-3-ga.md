# Round 3, GA/QD: the search pipeline

All numbers for the rung shares and R4 are estimates until the dump (section 6) exists. The dump is the first gate of the steps track.

## 1. The pipeline from breeding to archive

Breeding (CPU, cpu's fixed-array form, counter-based RNG keyed by seed, generation, breed round, slot, gene, draw). Per block, `plan_offspring` runs as today with three changes:

- `CMA_LIMIT` 1,024. An exploring emitter per (island, niche, topology) as now, but slots are no longer recycled by `last_used_generation` until the limit is hit; at 1,024 the recycling rule stays. Memory: mean, diagonal covariance and two paths, about 200 floats each, 4 KB per emitter, 4 MB total.
- Four optimizers per island at once. `optimizer_targets` already picks four designs (body plan plus cadence band) and cycles them every 30 stalled generations; instead all four run and the OPTIMIZER_SHARE children of an island go to them round robin by slot. Twenty optimizers in all.
- Tells per emitter every 200 accumulated samples, in absorption order, independent of block size. A sample is stored as a recipe (breed round, slot, emitter state version, 16 B), not as a parameter vector; at tell time the z vector is regenerated from the recipe with the counter RNG (200 x 178 gaussians, trivial). An emitter issues children under state version v until 200 results of version v have been absorbed; it then tells, bumps the version, and later results of version v still enter archives but not the update. So lambda is exactly 200 and the update is a function of ring order only. Optimizers rank by distance; exploring emitters rank by the CMA-ME improvement key as today.

Screening tables (CPU, per block, uploaded with the block like `Config`): one `bars` array of f32 and, per creature, a u16 bar index and a flags byte in the lane record. Layout: entry 0 is minus infinity (the audit lane and every creature whose bar is unknown); 10 arena bars (5 islands, 5 nurseries); 10 x 1,440 cell bars per rung; ml's R4 record table (10 x 1,440) and r_10 per cadence band. About 100 KB, L2 resident, read once per creature per rung. Each `Elite` stores its distance at 150, 300 and 600 steps (12 B more; the kernel already measures `screen_x`, the other two go in ml's free `Result` floats).

The bars. R2 (step 150) and R3 (step 300) bars for a child are the minimum over the parent's cell and its radius-1 neighbours (81 cells on the four live axes) of the elites' distance at that step, times a factor per rung calibrated on the audit lane (start 0.7 for R3, 0.5 for R2; a factor is a calibrated number, not a setting). Empty neighbourhood, unknown parent cell (immigrants, cross-plan grafts, reserve parents) or a nursery slot: the arena bar, which is the 80th percentile of that arena's own distances at the rung. R4 is ml's rule against the creature's measured cell at 10 s. R1 is ml's calibrated stop at 1 s.

The nursery fix is the arena bar: a nursery's bar is the 80th percentile of the nursery's own 5 s distances, so fresh random bodies compete with each other, as the nursery was designed to do. F1 from round 1 stands: today the nursery is almost certainly dead in mature games, and `Graduation { sent, kept }` from a mature save is the number that shows it.

The audit lane. A creature with `hash(slot, breed_round) % 100 == 0` gets bar index 0 at every rung and the flag that turns R1 and R4 off. It runs 1,200 steps, enters archives like any other, and its row (distance at 60, 150, 300, 600, final, final cell) goes back through data's candidate path. ml calibrates every factor and threshold on those rows only.

The body cap. `max_nodes(island)` is the largest node count among the island's behaviour elites plus two, capped by the global 32. It is a body limit applied in the structural operators' fit check, like the bone length cap: an operator whose child would exceed it does not fit and the picker retries up to four times, as today. No operator is removed. Immigrants and nursery bodies are exempt (they start at 3 to 5 nodes anyway). The owner decides; my estimate is that the 17-plus tail goes from 9% to under 2% of offspring, which is 15 to 20% of lane-steps at generation 50.

Absorption (CPU, ring order). The kernel's candidate bit (data) marks results that beat their cell's occupant or open a cell; the CPU commits candidates per island in block order exactly as `archive_block` does now, offers reserves, records emitter statistics, feeds recipes to the emitters, and tells the emitters that reached 200. Records still go through the 4x confirmation. Nothing in `QdArchive` changes except the three stored rung distances.

## 2. Steps per creature at generation 50

| rung | step | predictor | stopped, mature | stopped, fresh (generations 1 to 5) |
|---|---|---|---|---|
| R1 | 60 | ml's calibrated stop | 32% | 50% |
| R2 | 150 | neighbourhood bar x factor | 32% | 20% |
| R3 | 300 | per-arena and per-cell bar | 26% | 10% |
| R4 | 600 | ml's elite-relative stop | 5% (ml: 7%) | 0% |
| full | 1,200 | archive offer | 5% (ml: 1%) | 20% |

Mature: 235 with my R4 rate, 194 with ml's, 300 with R1 to R3 only. Fresh: 330. The audit lane adds 1% of creatures at 1,200 steps, about 3% of steps at maturity, included above by rounding. Falls are not counted and can only lower these. For the chair's table I land on 300 conservative and 235 optimistic.

## 3. Per-lever table

From 43 to 45M creature-steps/s at the generation-3 body and 480 steps; the kernel rows are the chair's working range at the generation-51 mix.

| lever | at 3T | at 5T | unit | owner |
|---|---|---|---|---|
| per-lane kernel, 2 substeps | 300M | 450M | creature-steps/s | gpu, physics |
| body cap (17-plus tail 9% to 2%) | x1.15 | x1.15 | rate of the mix | ga, owner |
| R1 to R3 | 480 to 300 | same | steps | ga, ml |
| R4 | 300 to 235 | same | steps | ml, ga |
| 1 substep | x1.8 | x1.8 | creature-steps/s | physics, owner |
| streaming bar (section 8) | early generations only | | steps | ga |
| emitter restructuring | x1.2 to 1.8 best distance at equal time, no rate change | | progress | ga |

Creatures/s sustained at generation 50, before the host tax of 5 to 8%:

| row | 3T | 5T |
|---|---:|---:|
| 2 substeps, body cap, R1 to R3 | 1.15M | 1.7M |
| 2 substeps, body cap, R1 to R4 (235) | 1.5M | 2.2M |
| 1 substep, body cap, R1 to R3 | 2.1M | 3.1M |
| 1 substep, body cap, R1 to R4 | 2.6M | 4.0M |

My design lands on row 2 (2 substeps, R1 to R4) and honestly stops at 1.5M/s if the power budget is 3T and R4 fires at my rate; it reaches 2M/s at 5T, or at 3T with 1 substep. If 1 substep fails its honesty test, row 2 is the ceiling and the search side has nothing more than the body cap and the ladder to give; the remaining gap is the kernel's.

## 4. What happens to CMA children under R4

In my own words. R4 stops a survivor at 10 s when its distance, scaled by the calibrated ratio, cannot beat the smallest record in the neighbourhood of the cell it is measured in at 10 s, and never when a neighbouring cell is empty. A stopped child keeps its 10 s distance as its score and is treated as screened: no archive, and it enters its emitter's tell with that score. For an exploring emitter the tell key is improvement over the cell's elite, so a stopped child's key is its 10 s distance minus the elite's record: negative and large. Children that ran full and did not improve have keys that are negative and small; children that improved have positive keys. So R4 compresses the ordering inside the bottom of the ranking, among samples that lost anyway, and leaves the top half, which drives the mean and covariance update, in the same order. That is why exploring emitters do not need an exemption. The island optimizers rank by distance alone, and a population stopped at 10 s would be ranked by a 10 s pace, which is not what the optimizer is optimizing; their children carry the flag that turns R4 off. They are 9% of creatures and the exemption costs about 3 steps per creature. R3's per-cell bar does the same to CMA children as today's global bar: a screened child enters the tell with its 5 s distance, below every survivor.

One consequence to measure: a mature exploring emitter whose cell's elite is far ahead of its mean will have almost all its children stopped at R3 or R4, so its top half is ranked by 5 s and 10 s distances. That is fine for direction, and `converged()` restarts an emitter that no longer improves. The measurement is the emitter discovery and improvement rates in `emitter_stats`, control against ladder.

## 5. Ruling on 16-bit muscle genes

Accepted, with two conditions. The host genome stays f32 and the archive stores f32; the kernel's packed record is the phenotype, and since the GPU score is final and replays come from the same kernel, the rounding is part of the physics, not a disagreement between engines.

Condition one: fixed-point u16 over each gene's clamp range, not fp16. The genes are bounded (anchor 0 to 1, short and long up to the stroke cap, duty 0.05 to 0.95, phase and reset on the unit circle, stiffness 1 to 120 on a log scale, tendon 0 to 1), so uniform quantization gives a step of 1.5 x 10^-5 of the range, three orders of magnitude below the smallest CMA sigma the emitters reach before `converged()` fires. fp16 would give 5 x 10^-4 relative and a step that depends on magnitude, which puts a plateau under the optimizer for the small genes.

Condition two: the period is not a per-muscle 16-bit gene. Every muscle shares one body clock (`repair` copies the first muscle's period) and a limb runs at one of five ratios. So the record carries one f32 base period per creature and a 3-bit ratio code per muscle, which is exact. A period rounded to 16 bits drifts a fast gait by several percent of a cycle over 20 s, and the search would learn the rounding.

Effect on the search: none I can predict, because the quantum is below the search's own resolution. Gates: the top 300 of a save evolved at u16 rescored with f32 constants in a diagnostic build, median ratio at the 0.99 of the 2x-rate retest; and `search_ab --gpu`, 10 seeds, equal evaluations, best and QD within the seed spread. `qd::VERSION` bumps.

## 6. The dump tool

One env var, `EVOLUTION_DUMP_GENERATION=<path>`. At the next generation boundary the worker runs one whole generation with every rung off and the bar at minus infinity (an all-audit generation, 3M x 1,200 steps, about 80 s at today's rate), then writes one binary row per creature and one per elite of the archive at that boundary, and unsets itself. Rows, little-endian:

Creature (64 B): slot u32; arena u8; emitter u8; operator id u8 (255 for none); flags u8 (mate, from reserve, audit); parent id u64; parent cell 6 x u8; cma index u16; nodes u8; muscles u8; distance at steps 60, 150, 300, 600 and final, 5 x f32; fall time f32; final cell 6 x u8; entered u8 (0 none, 1 island, 2 nursery, 3 reserve, 4 global); fine u8; confirmed fitness f32.

Elite (32 B): arena u8; cell 6 x u8; fitness f32; distance at 150, 300 and 600, 3 x f32; nodes u8; muscles u8; id u64.

The kernel writes the four rung distances into ml's free `Result` floats; the rest is host state. ml's offline tool applies a candidate ladder to the rows and prints steps per creature, stopped share per rung, misses per rung and per cadence band, and the share of the final top 1% and 10% kept. This one file decides R1 and R2's shares, R3's factor, R4's fire rate and the cell-match rate between 5, 10 and 20 s. It also answers F1 (nursery rows with `entered`) and gives the operator histogram of the 17-plus tail.

## 7. Gates, per piece

Every search A/B is `search_ab --gpu`, 10 seeds, 40 generations at 3M, fixed seeds, control against change, reporting the median over seeds of best distance, QD score and entrants per generation, plus a fixed-seed repeat (two runs of one seed give one history).

- Streaming bar and per-arena bars (T1): kept share per block in generations 1 to 10 within 18 to 22%; nursery `kept` per cohort above zero in a mature save; equal time A/B best and QD not below control.
- Per-cell bars at R3 (T2): first the dump: at factor 0.7, entrants stopped under 4% of entrants and none of the top 1%; then equal time A/B: best at or above control, QD at or above 0.95x, steps per creature at or below 0.8x of control.
- Emitter restructuring (T3): samples-per-tell histogram shows 200 for every active emitter; equal evaluations A/B: best at or above 1.1x control (the claim is 1.2 to 1.8), QD not below 0.95x. If best is under 1.05x the lever is deleted, code and all.
- R1, R2 and the audit lane (T4, ml's predictors): dump misses under 0.1% of entrants per rung; equal time A/B as T2.
- R4 (T5): dump fire rate and misses; optimizer record trajectories per island not below control; equal time A/B as T2.
- Body cap (T6, owner first): offspring node histogram from a generation-50 save before and after; equal time A/B best not below 0.97x control, since the cap could cost a rare large winner.
- 16-bit genes (T7): section 5.

Order: T1 is a one-day change with no kernel work and fixes a live bug (the nursery) and an early-generation loss (section 8), so it merges first. T2 needs the bar table in the kernel (P1) and the dump. T3 rides on cpu's breeding rewrite and the counter RNG, one `qd::VERSION` bump for both. T4 and T5 after the dump. T6 whenever the owner answers. T7 with gpu's packed record. The game is playable at every merge because each piece is on by default and measured on its own; nothing is a switch.

## 8. The new idea: the bar moves per block, not per generation

Today `next_screen` sets the bar once per generation from the previous generation's `screen_log`. In generations 1 to 10 the population's 5 s distance roughly doubles per generation (best went 0 to 16 m by generation 10 on save42), so a bar that is the previous generation's 80th percentile keeps 40 to 50% of the current one, not 20%. At 45% kept the mean is 0.55 x 300 + 0.45 x 1,200 = 705 steps, against 480 at 20%. The same happens for the first generation after every world change, when the bar restarts at minus infinity and re-arms after a quarter of the generation.

The change: the bar is recomputed at every absorption from a sliding window of the last 786k `screen_x` values (the ring's worth), per arena, and the block bred at that absorption takes it. Determinism holds because the bar is fixed when the block is bred, as now. Under per-cell bars the same thing happens by itself for cells with elites; the arena bars need it explicitly. Expected: generations 1 to 10 and every post-change generation run at about 480 steps instead of about 700, a 1.4x on those generations, which is where the player is watching most closely. Cost: a quantile over a 786k window per block, one `select_nth_unstable` on a copy, about 5 ms. Measurement: the kept share per block in `EVOLUTION_STAGE_LOG` for generations 1 to 10 and after a world change, before and after.

## 9. Determinism, world change, save, replay, 60 FPS

Determinism: every input to a creature's trial (genes, seed, bar indices, the bar table, the R4 table and ratios, ml's weights, the flags) is fixed when its block is bred and travels with the block. Emitter tells use the first 200 absorbed results of a version, in ring order. The audit lane is a hash. The dump generation is a diagnostic and need not be reproducible.

World change: blocks in flight enter no archive (as today; the ring is at most 1 s, so at most 2M creatures). The arena bars restart at minus infinity and re-arm from the sliding window after 64 results per arena. Per-cell bars and the R4 table are off (entry 0) until the re-tested elites (the `reseed` queue) have new rung distances, because old records are not comparable; that is at most one generation. Emitters keep their means; the archive re-scoring handles the rest as now.

Save: emitters (4 MB), pending recipes (1,024 x 200 x 16 B = 3 MB), 12 B more per elite, ml's weights and the calibrated factors. Under 15 MB more. Load rebuilds the tables from the archive.

Replay: unchanged, from the scoring kernel with the same tables, so a screened or stopped replay stops where the trial did.

60 FPS: `plan_offspring` with 1,024 emitters is the same O(1) hashed lookup per child; the by-design optimizer selection is per island per block. The sliding-window quantile is 5 ms per block on the worker. Nothing new touches the UI thread.

## 10. What stays out and why

Surrogate best-of-2 breeding: parked until the dump gives a Spearman above 0.6 between a genome-side predictor and the 5 s distance. Exact-duplicate elimination: measured from the dump first; under 1% it stays out. Per-emitter racing bars: under 2%, out. More islands and finer grids: they absorb evaluations but nothing showed they are needed for the rate, and every island multiplies host work. Periodic-gait extrapolation: a score that is not a full simulation; out. The Radeon: nothing in the search uses it.

## 11. What would stop this design

The dump showing entrants are not predictable at 5 s per cell (more than 4% of entrants below 0.7 times their neighbourhood minimum) removes the per-cell gain and leaves the arena bars: the design drops to row 1 at 1.15M/s (3T). The emitter A/B under 1.05x removes T3 and changes no rate. The body cap losing at equal time keeps the 9% tail and takes 15% off every row. The physics honesty test failing removes rows 3 and 4. Every piece that passes its gate still ships, in the order above.
