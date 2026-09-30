# Round 1, machine learning and tensor acceleration

Author: the ML domain. Read: 00-facts.md, hpc.md, src/ring.rs, src/qd.rs, src/storage.rs (verdict, absorb, archive_block, breed_block), src/physics.rs (Screen, settle), shaders/warp_creature.cu (whole kernel, the metrics and screen section in particular), docs/design-decisions.md, docs/rejected-ideas.md, docs/backlog.md.

My line: a model never produces a score. It decides what to simulate, when to stop a trial, and what to breed. Everything below keeps that line, and I say where an idea would cross it.

## Where the steps go today

Every creature runs 100 settle steps before its timed trial (`Fidelity::settle` is 200 * rate / 120, so 1.67 s at 60 Hz). A screened creature therefore costs 100 + 300 = 400 steps and a survivor 100 + 1,200 = 1,300. With 80% screened and falls ending some trials early the mean is about 520 to 580 steps, which matches the facts file. Split by cause:

- settle: 100 steps for everyone, 17 to 19% of all steps;
- the 80% who fail the 5 s screen: 240 steps per creature on average, 41 to 46%;
- the 20% survivors: 240 steps per creature on average, 41 to 46%.

Two things about the survivors decide my ranking. First, almost none of them enter an archive. Each island has 1,440 niches and records only rise, so a 3M generation produces at most a few thousand archive entries out of 600k full trials. Second, the only consumers of a survivor's full 20 s result are the archive offer (which needs the final distance and the final behavior cell) and the CMA tell, which ranks samples and already receives screened creatures with their truncated 5 s distance (storage.rs, the cma_samples push takes `prep.score` whatever ended the trial). So the search already lives with truncated distances at the bottom of a CMA ranking. That is the door the proposals below walk through.

The kernel already computes, every step, everything a stop rule needs: `com_x`, `center_y`, the contact and lift bit masks, `head_shake`, `gait_turns`, `height_sum`, and the muscle energy stores. A checkpoint costs a few dozen instructions once per creature, against about 2,000 instructions per step. Inference in the kernel is free at this scale. No model here needs the iGPU or a tensor core.

## Proposals, ranked by expected gain

### 1. Calibrated early stop before the screen (1 s, 2 s, 3 s checkpoints)

What: at fixed steps (60, 120, 180 after settle) the kernel evaluates a small fitted model on the creature's running state and stops the trial when the probability of passing the 5 s bar is below a tolerance. The stopped creature keeps its true simulated distance at the stop, is marked screened, enters no archive, and feeds the CMA tell as screened creatures do today. Fitness is untouched.

Model: features at the checkpoint are `com_x`, the mean forward speed over the last 0.5 s, `center_y` relative to the settle pose, the number of contact bits set, `head_shake`, mean muscle energy, and the 5 s bar itself. A logistic model or a 2-layer MLP with 16 hidden units is 100 to 400 FMAs once per creature. Weights are fit on the CPU at the generation boundary from the previous generation's 3M results (the results already carry `screen_x`; the checkpoint features need to be returned too, see needs below), single-threaded, fixed iteration count, so the search stays deterministic per seed. Weights ride in the block's config like the screen bar does, so a block keeps the rule it was bred with.

Calibration: this is a conformal-style threshold, not a trusted classifier. The tolerance is set so that on the previous generation at most 1 in 1,000 creatures that would have passed the 5 s bar is stopped. That is measured directly, because the previous generation's survivors are known.

Estimate: the screened 80% split into creatures that never move (a large share of structural children and every random body: random bodies gain a median -0.05 m) and creatures that move but too slowly. I expect 40% of the screened group identifiable at 1 s, another 30% at 2 s and 10% at 3 s, at the 1 in 1,000 loss level. Steps saved per screened creature: 0.4 * 240 + 0.3 * 180 + 0.1 * 120 = 162, so about 130 per creature overall. Mean steps drop from about 550 to about 420, a 1.3x on throughput at equal kernel speed, and the stopped creatures still count as evaluated exactly like today's screened ones. The uncertainty is the split above. The data to replace the guess exists in one recording of a 3M generation with per-checkpoint features.

Risk: the 5 s screen already lost 14% QD at equal evaluations and won at equal time. An earlier stop is the same trade at a smaller scale, and its loss is bounded by the calibration. The risk that matters for the owner's spirit is a systematic one: bodies that stand still for 2 s and then walk (a slow rhythm, a period near the 2 s limit) would be stopped as a class. The feature set includes the muscle period, and the calibration is measured per cadence band, so the loss is visible per class, not only on average.

Measurement that confirms it: replay one saved 3M generation offline with the rule and report (a) steps saved, (b) the share of the final top 1% and top 10% kept, as the 5 s screen was validated (it kept 100% and 96%), (c) the miss rate per cadence band. Then `search_ab --gpu` at equal GPU time over 10 seeds, best distance and QD. Gate: top 1% kept at 100%, top 10% at 95% or more, best distance and QD not below the current game at equal time.

### 2. Elite-relative stop for survivors (8 s and 12 s checkpoints)

What: after the screen, a survivor is only useful if its final distance beats the record of the cell it will land in, or if it opens an empty cell. At 8 s and 12 s the kernel computes the creature's provisional behavior cell from its running metrics (the same code path as the final descriptor, on the running sums), reads the record of that cell and of its one-step neighbours along each axis from a per-arena table, and stops when `d(t) * r_t` is below the smallest of those records. `r_t` is the 99.9th percentile of `final / d(t)` among the previous generation's survivors, calibrated per checkpoint. An empty neighbour (record minus infinity in the table) blocks the stop, so a creature that might open a cell always runs full. The stopped creature keeps its true distance at the stop and enters no archive.

Why this is not the rejected "stop once it cannot beat its cell": that one was impossible because the final cell is unknown. Here the cell is predicted, the prediction's error is measured (the fraction of survivors whose 8 s cell differs from the 20 s cell, per axis), and the neighbourhood plus the empty-cell block absorb it. Feet only grows, contact and height are running means, cadence is turns per second. I expect 85 to 90% of survivors in the same cell at 8 s and 97% within one step on one axis. That is a number to measure, not to trust.

Estimate: children of the CMA and structural emitters land near their parents' cells, which are occupied by definition, so the rule fires for most of them. Novelty children aim at sparse regions and fire less. If 80% of survivors are stopped at a mean of 9 s (540 timed steps instead of 1,200), the saving is 0.2 * 0.8 * 660 = 106 steps per creature. Together with proposal 1 the mean drops from about 550 to about 310 steps. That is 1.75x end to end at equal kernel speed, and it changes the target for the kernel: 2M/s needs about 620M creature-steps/s instead of 1.0 to 1.2G, so 14x on today's 43 to 45M rather than 25x.

The rule strengthens as a run ages: full archives and grown bodies are exactly where trials are most expensive. It does nothing in generation 0.

Data flow: the table is one float per cell per arena, 1,440 * 9 arenas = 52 KB, built at breeding from the archive records and uploaded with the block. Records only rise, so a table set at breeding time is conservative. Two floats for `r_8` and `r_12`. Determinism holds because the table is part of the block's settings.

Risk: a late accelerator (a creature whose second 10 s are much faster than its first) is lost when its ratio is above `r_t`. That is one in 1,000 by construction on the previous generation, and it is measured every generation on the audit lane (proposal 4). The other risk is the CMA tell receiving 8 s distances for most of its samples. Since it ranks, and since every stopped sample was below the record of its own cell, the ranking among the samples that matter (the top of the CMA population) is unchanged. The optimizer variant (`tell_optimizing`, physical units, ranks by distance alone) is the one to check: if its whole population is stopped at 8 s its ranking becomes an 8 s ranking. The fix is a flag on the CMA optimizer's children that turns the rule off for them; they are a small share.

Measurement: the same offline replay as proposal 1, plus the cell-stability table. Gate: the misses (creatures that were stopped and would have entered an archive) under 0.1% of survivors; `search_ab --gpu` at equal time, 10 seeds, best distance and QD not below the current game.

### 3. Cut the settle phase, or stop during it

Not my domain, but it is 17 to 19% of all steps and nobody else's file mentions it. `docs/design-decisions.md` says a reduced-coordinate pose is valid by construction and the settle was needed by the point-mass chain. If physics agrees the settle can go from 100 steps to 20, the mean drops another 80 steps. From my side: the fall check runs during settle, and if a measurable share of creatures fall or break a joint in settle, that share is already free.

Estimate: 1.15x if settle goes to 20 steps. Measurement: elite distance and fall share at settle 100 versus 20 on the top 300 of a save (`replay_match` style).

### 4. The audit lane and the per-generation fit (the plumbing for 1 and 2)

A learned screen trains on data it censored, and that drift is the classic failure of learned screening. So 1% of the slots of every block, chosen by a hash of the slot and the breed round, run with every stop rule off. They give an unbiased estimate of each rule's miss rate every generation, an uncensored training sample, and the number the game can show a developer (misses per 10k). Cost: 1% of creatures at full length, about 3% of steps.

The fit itself: a logistic model on 8 to 16 features over a 200k subsample, 20 passes, on one CPU thread, about 20 ms per generation; the ratio quantiles and cell-stability tables are one pass over the previous generation's results. Nothing here needs the iGPU, and I recommend against putting ML work on it: the models are too small to justify the crash risk, and any spare iGPU capacity is worth more to breeding and packing (HPC's call).

### 5. Surrogate-assisted breeding (what to breed)

SAIL (Gaier, Asteroth, Mouret 2018) is MAP-Elites with a surrogate that picks which candidates to evaluate. Our version: for a CMA or structural slot, breed 2 candidates instead of 1 and simulate the one a surrogate ranks higher on predicted 5 s distance. Features: the parent's fitness and cell, emitter and operator id, node and muscle counts, mass, bone length sum, CMA sigma, and the size of the genetic change. Training data: 3M labelled rows per generation, for free. Novelty and immigrant children stay untouched, and the choice is best of 2, not best of 16, so the model cannot narrow the search much.

This raises search quality per evaluation, not evaluations per second. It also raises the screen pass rate, so mean steps go up, not down. I put it here because the owner asked for creativity and because the data is already there, but it is a search change to test with `search_ab --gpu` over 10 seeds at equal GPU time, and I would not do it before 1, 2 and 4. Prior evidence in this repo is mixed: the reward bandit on emitter shares tied, and the mutability gene washed out.

A pure genome pre-screen (skip a child without simulating it) is the same model used harder. A skipped child is not an evaluation, so it does not move the 2M/s number, and I would not propose it as a throughput lever.

### 6. Learned routing of confirmation trials (non-glitchy movers)

Confirmations now go only to record setters. A creature that enters a cell without a record is never re-tested, so an integrator exploit can sit in an archive cell. A small model on trial features (head shake near its limit, energy ledger excess count, friction impulse along the slip, fall time) can flag suspicious entrants for a confirmation at the fine physics. This spends a little GPU (a few hundred confirmations per generation) to protect the spirit. It decides what to simulate, not what scores. Needs physics to return one or two exploit indicators per trial (the kernel has the ledger and the friction impulses in hand). Measurement: the share of flagged entrants whose confirmation keeps less than 80% of the standard score, against the same share for unflagged ones.

## What I looked at and reject

Tensor cores for the contact solve. The solve is a projected Gauss-Seidel on at most 4 contacts, an 8 by 8 system, 256 multiply-adds per substep per creature, and it is 16 to 20% of kernel time. To use `mma` you would pack two creatures block-diagonally into a 16 by 16 tf32 fragment and run projected Jacobi instead of Gauss-Seidel. Building the fragments costs 20 to 30 instructions, Jacobi needs damping and more iterations, and the projection breaks the linear form every iteration. Best case is halving the solve, so a 1.1x, with tf32 precision in the contact matrix. Not worth it. The one place a tensor core fits this game is a future neural controller (backlog): a 16 by 16 fp16 layer is one `mma` per creature per step against about 128 lane instructions, and the weights would be per creature, which is fine for `mma` since it needs no shared operand. If the controller track starts, design it around a 16 by 16 fp16 layer from day one.

Neural or fitted contact models. Evolution is an adversary against the physics; it found the 30 Hz integrator error, the spring joint limit and the one-at-a-time contact solve. A learned contact model has holes everywhere and they would be found within generations. This also crosses my line, since the score would come from a model. Rejected.

A low-fidelity kernel for the screen only (one substep, 2.2x faster) with the true kernel for survivors. Multi-fidelity screening is sound in the literature, and only the 60 Hz score would enter archives. But exploits pass a coarse screen more often than honest movers (30 Hz elites kept 38% at 60 Hz), so the screen would select for them, and proposal 1 takes most of the same steps without a fidelity mismatch. Keep as a later experiment, not a track.

Per-operator contextual bandits over the 64 operators. Adaptive operator selection has a literature (Fialho et al. 2010), but the repo's own evidence on reward-following shares is a tie or a loss twice. Low expected gain, not proposed.

## What the search-side levers add up to

At today's kernel speed: proposals 1 and 2 together take the mean from about 550 to about 310 steps (1.75x), settle to 20 steps takes it to about 230 (2.4x). At 230 steps per creature, 2M/s needs about 460M creature-steps/s, which is 10 to 11x the measured lane-group kernel, instead of 25x. Everything above is a per-generation fit and a few checkpoints in the kernel, no new physics and no new archive semantics, and each piece is on or off by its own measured gate. I do not claim 2M/s from my side alone. I claim the kernel target drops by a factor of 2.4.

## What I need from the other domains

From the kernel and HPC domain: a checkpoint hook at a small fixed set of steps (60, 120, 180, 480, 720 after settle) that reads a per-block parameter blob (about 64 floats plus a 52 KB cell-record table in global memory) and can end the trial like the screen does. And the `Result` struct extended with the checkpoint features (about 10 floats: `com_x` and `center_y` at 1, 2, 3, 8 and 12 s) so the fit has its data. That is 40 more bytes per result, 120 MB per 3M generation over PCIe, which is nothing.

From the search domain: agreement that a survivor stopped by proposal 2 is treated exactly like a screened creature (no archive, CMA tell with the truncated distance), a decision on whether the CMA optimizer's children are exempt from it, and the cell-stability numbers (8 s cell versus 20 s cell per axis) from one saved generation.

From the physics domain: a verdict on the settle phase in reduced coordinates, and one or two exploit indicators per trial in `Result` (friction impulse along the slip, count of first-law corrections).

From the data-flow domain: the audit lane (1% of slots by hash, every stop rule off) and the per-generation fit at the generation boundary as part of `end_generation`, with the fitted weights and tables stored in the block config so a save and a load reproduce the same search.

From the measurement domain: one offline tool that replays a saved 3M generation's results through a candidate rule and prints steps saved, misses, and the share of the final top 1% and 10% kept. That tool is the gate for everything I proposed, and it is cheaper than any of the tracks.
