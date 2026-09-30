# Round 1, evolutionary computation and quality diversity

Author: the GA/QD domain. Question owned: how many creature-steps per creature and per unit of search progress are really needed, and how 2M evaluations per second become better creatures instead of noise. Sources: 00-facts.md, hpc.md, and the code on main (src/ring.rs, src/storage.rs, src/qd.rs, src/evolution.rs, src/physics.rs, shaders/warp_creature.cu), docs/design-decisions.md, docs/rejected-ideas.md.

## 1. Where the steps go today

Numbers from the code, not estimates:

- The lane-group kernel simulates no settling steps. `record_frame` repeats the start pose SETTLE + 1 times for replays and the trial starts at step 0. So a screened creature costs 300 steps (5 s at 60 Hz) and a survivor 1,200. Settling is not a lever any more; hpc.md section 4 is out of date on this.
- With the top 20% kept: 0.8 x 300 + 0.2 x 1,200 = 480 steps per creature before falls. Falls bring it to the 500 to 600 range the facts file quotes only if the fall accounting counts fallen survivors at their full 1,200. Either way: half of all GPU steps are seconds 5 to 20 of the 20% survivors. The other half is the 5 s rung on the 80% that lose.
- Confirmation trials (records only, 4x rate and 4x passes, at most 8 speculative per arena per block) cost at most 80 x 16 x 1,200 = 1.5M step-equivalents per 196k block of 94M. That is 1.6%. It is the only anti-glitch work and it is cheap. Do not touch it.
- What the search actually needs: every archive entrant needs its full 1,200 steps because the niche is measured over the whole trial (rejected-ideas.md is right that a trial cannot stop once it cannot beat its cell, if the cell is unknown). If entrants are 0.5% of creatures, the floor is 0.005 x 1,200 = 6 steps per creature. We spend 480. Everything above the floor is the price of predicting who will enter. The 5 s distance is that predictor today (Spearman 0.887 with the 20 s distance, 100% of the top 1% and 96% of the top 10% kept). Steps per creature is therefore a question about predictors, and the room is large in principle but bounded by prediction quality in practice. My honest ceiling for screening policy alone is about 1.6x fewer steps per creature (480 down to about 300). The other 15x of the 25x must come from the kernel and from spending evaluations better, which is a different unit: progress per GPU second.

## 2. Findings in the code that cost search progress now

These are not speculation. They follow from the constants and the loops.

F1. One global screen bar for islands and nurseries. `Experiment::screen_log` is one vector over every absorbed result and `Config::screen.bar` is one number. `archive_block` drops every screened result (`valid = ... && !screened`) for nurseries too. The nursery's fresh random bodies (NURSERY_FRESH_SHARE 0.5 of 10% of slots) gain no distance (facts: median -0.05 m), and the bar is the top 20% of an evolved population at 5 s. So in a mature game nearly every fresh nursery body is screened and never enters its nursery, and the nursery cohort, which is meant to develop for 10 generations, is fed almost nothing. That is 10% of all slots, 300k creatures per generation, 90M steps, spent on trials whose result is thrown away by construction. Measurement: per generation, the share of nursery-slot results with `screened == true`, and the `Graduation { sent, kept }` numbers already kept per island. If sent is near zero in mature saves, the nursery is dead today.

F2. The same bar starves weak cells. A child of an elite in a weak cell can beat that cell at 20 s and still be screened at 5 s because the bar is the population's top 20%. `sample_local_competitive` sends about a quarter of CMA children and most structural children to such parents. Those evaluations are 300 steps each and cannot enter any archive, whatever they do. The archive design says cells compete locally; the screen competes globally. The two disagree and the screen wins.

F3. CMA emitters are oversampled by 3x to 50x. `CMA_LIMIT` is 96 emitters. CMA children are 35% of 3M = 1.05M per generation. Half of those go to top parents and half of those to the island optimizer (TOP_PARENT_SHARE 0.5 x OPTIMIZER_SHARE 0.5), so one optimizer per island receives about 0.35 x 0.25 x 3M / 5 = 52k samples per generation, 3.4k per 196k block. `tell` runs once per block and truncates to the best 1,024 samples (MAX_UPDATE_SAMPLES), then uses mu = 512. So for the optimizer two thirds of its samples never reach the update, and the update that does happen uses a lambda 5 to 50 times larger than CMA-ES theory wants for 100 to 200 dimensions (a 7-node, 6-bone, 15-muscle body is 28 + 30 + 120 = 178 dimensions; the default lambda is 4 + 3 ln n = 20, and the progress rate per generation grows only logarithmically beyond about 10 n while the cost grows linearly). The exploring emitters are in the same state to a smaller degree. This is the batch-size effect from the literature, and it is on the emitter, not on the archive. QDax found no loss from large batches for MAP-Elites because thousands of cells each get few samples; here 96 emitters get 11k each.

F4. Breeding is the next wall and it also costs GPU clock. Breeding is 4.7 s per 3M generation, so 640k children per second on the CPU. The target is 2M per second. And 8 busy CPU threads cut the GPU from 174k to 137k trials/s through Dynamic Boost. At 2M/s the search cannot afford to breed every child on the CPU, and parametric children (CMA and the local mutation) do not need to be: a parent index, a seed and a sigma are enough to materialize them where they are simulated.

F5. Feedback latency is set by the ring, not the generation. The ring holds 786k creatures in 4 blocks. Emitters learn once per block. At 2M/s that is 10 tells per second, which is good, but a block of 196k is far more than any emitter can use per update (F3). Smaller blocks give every emitter faster and cheaper updates, if the GPU stays full at smaller waves (question for the HPC domain).

## 3. Proposals, ranked by expected gain

Units: steps saved per creature (GPU work), or progress per GPU second (best distance and QD at equal wall time in `search_ab --gpu`). Every proposal keeps: distance-only fitness, the GPU score final, determinism per seed (every bar and every seed is fixed when the block is bred, as today), full trials for every archive entrant, and the 4x confirmation for records.

### P1. A bar table in the kernel: screening becomes a policy, not a number

Mechanism. The kernel already reads `p.screen_bar` and `p.screen_step`. Replace the scalar with a small table of bars in global memory (L2 resident, at most a few hundred KB) and give every creature a bar index (a u16 in its lane record). At the screen step the creature compares its distance against `bars[index]`. The CPU fills the table when it breeds the block, so a block's table is part of its settings like `Config` is now, and one seed still gives one search. Kernel cost: one load per creature per rung. This single change enables P2, P3, P4 and P6 with no further kernel work.

Gain: none by itself. Risk: none. Measurement: bit-identical results with a table of one entry equal to the current bar.

### P2. Per-arena and per-cell bars (cell-aware successive halving)

Policy. Each island and each nursery keeps its own bar (fixes F1 at once). Then each cell keeps a bar: for every elite, store its distance at the screen step (`screen_x`, already measured for every creature and simply not kept in `Elite`; 4 bytes). A child's bar is the minimum over its parent's cell and the 3^4 neighbouring cells of the elites' 5 s distances, times a safety factor below 1 (start at 0.7), or the arena bar when those cells are empty. The child's cell is predicted from the parent's cell because most children land near their parent (gentle operators keep 97 to 100% of the parent's gait, CMA children more so). The kernel needs nothing but P1. For a structural child whose cell is unknown, the arena bar applies.

Expected. In a mature archive the strong cells' elites reach far more than the population's 80th percentile at 5 s, so children aimed there are cut harder; children aimed at weak or empty cells are kept, and the nursery lives. Estimate: the kept share falls from 20% to 5 to 10% in mature generations while every eventual entrant is kept, so steps per creature go from 480 to about 0.92 x 300 + 0.08 x 1,200 = 372, 22% fewer. Early generations gain nothing (bars are low everywhere) and lose nothing.

Risks. The cell prediction can be wrong: a child that changes cell is measured against the wrong bar. The neighbourhood minimum and the 0.7 factor absorb most of that. A bar that is too tight cuts future entrants, which is the same failure mode as the rejected second rung, so the first measurement must be offline.

Measurement, in order. (1) Dump one mature 3M generation with `screen_x`, final distance, final cell and parent cell for every creature (the fields exist; the dump is a diagnostic). Compute, for every creature that entered an archive, whether the per-cell rule with factor f would have kept it, for f in 0.5, 0.7, 0.9. Target: 100% of the top 1% entrants and at least 96% of all entrants kept, which is what the current rung achieves. Report the kept share per island. (2) `search_ab --gpu`, 10 seeds, equal wall time, current bar against per-cell bars: best distance, QD, entrants per generation, nursery `kept` counts. Also measure the nursery alone: per-arena bars only, everything else unchanged.

### P3. Emitter restructuring: many small CMA emitters, faster tells, smaller blocks

Policy. Raise `CMA_LIMIT` from 96 to 1,024 or more (each emitter is a mean, a diagonal covariance and two paths of about 200 floats: 4 KB; 2,048 emitters are 8 MB). Give each island optimizers on its 4 fastest designs at once instead of one design in turn (`optimizer_targets` already picks 4 designs and cycles them by OPTIMIZER_STALL). Cap the samples per emitter per block near 256 and spread the rest of the CMA share over more emitters (more cells per island get one; the exploring emitters are per cell already, so this mostly means not letting 96 slots be recycled by `last_used_generation`). Shrink `RING_BLOCKS` from 4 to 16 (49k per block) so tells happen 4x as often with 4x fewer samples each. The ring stays 786k.

Expected. CMA-ES theory (Hansen; Arnold on (mu/mu_w, lambda) progress rates): at fixed evaluation budget, splitting a lambda of 3,400 into 10 updates of 340 gives roughly 3 to 5 times the progress on a smooth landscape, and the truncation to 1,024 today throws away the rest outright. CMA is 35% of the budget, so a 3x gain on it is about 1.5x progress per GPU second for the whole search if the other emitters are unchanged. I put the honest range at 1.2 to 1.8x on best distance at equal time, because the landscape is not smooth and elites move cells.

Risks. Emitters starved below about 30 samples per tell learn noise; the cap must be a floor too. More emitters mean more `plan_offspring` lookup work on the CPU (F4). Smaller blocks add a wave tail on the GPU (the HPC domain must say how much; see section 5). Determinism is unaffected.

Measurement. Print a histogram of samples per tell per emitter (a diagnostic env var). Then `search_ab --gpu`, 10 seeds, 60 generations, equal evaluations: CMA_LIMIT 96 vs 512 vs 2,048 crossed with 4 vs 16 blocks. Report best, QD, and the optimizer's own record trajectory per island.

### P4. Per-emitter racing bar (within P1)

Policy. An optimizer's children in a block are one lambda. The update uses the top half. Any child below the emitter's own median at 5 s is irrelevant to the update and, for an optimizer on the island's fastest design, almost always irrelevant to the archive. So each optimizer gets its own bar: the mu-th best `screen_x` among its previous tell's samples, times the same safety factor. This is racing (Maron and Moore) applied per emitter. With P3's smaller lambdas the bar refreshes every block.

Expected. Optimizer children are 25% of CMA children, about 9% of all creatures. Cutting their kept share from 20% to 10% saves 0.09 x 0.1 x 900 = 8 steps per creature, under 2%. On its own this is small. It matters more if the owner later widens the optimizer share, and it costs nothing once P1 exists. Ranked here for honesty, not for size.

Measurement. Same dump as P2: for optimizer children, the share of archive entrants below the emitter's median at 5 s (expected near zero).

### P5. A second cell-aware rung at 10 s, and an early rung at 2.5 s

The rejected second rung (10, 15, 20 or 30 s on 60 s trials) used a global bar and lost QD. A cell-aware bar changes what a rung cuts: it cuts creatures that will not beat their own neighbourhood, not the population's slow half. So the experiment is worth repeating in that form, and only after P2 has shown its offline numbers.

Rung at 10 s (600 steps): store each elite's 10 s distance too (another 4 bytes). Cut survivors below the neighbourhood's 10 s minimum times the factor. Expected: half of the survivors cut, saving 0.1 x 600 = 60 steps per creature (12%) at 20% keep, less under P2's lower keep. Risk: moderate; the rejected result stands until the cell-aware version is measured the same way.

Rung at 2.5 s (150 steps): only worthwhile if the 2.5 s distance ranks the final distance nearly as well as the 5 s distance does. The gait period floor (`min_muscle_period`) and how long a gait takes to catch (anatomy-operators.md: whether a gait catches depends on how it starts) decide this; the physics domain should say whether 2.5 s covers at least two gait cycles for typical elites. Expected if it works: cut half at 2.5 s, saving 0.5 x 150 = 75 steps per creature (16%). Measurement: Spearman of the 2.5 s distance with the 20 s distance and the recall of entrants, from the same dump with one more sampled distance.

Together with P2 the geometric schedule (150, 300, 600, 1,200 with keep 0.5, 0.25, 0.5 relative) gives 150 + 75 + 75 + 75 = 375 steps per creature at today's keep rates and near 300 under mature cell bars. That is the 1.6x ceiling from section 1.

### P6. Materialize parametric children on the GPU

Policy. For CMA children (35%) and for local-mutation-only children, the CPU sends a plan record: parent slot in a GPU-resident elite table, emitter index, seed, and sigma. The kernel materializes the child in its lane at load time from the parent genome and the emitter's mean and covariance (both small tables uploaded per block). Structural children (the 64 operators, repair, canonical bone order) stay on the CPU; they are the minority and the expensive ones. The result record must carry the seed back so the CPU can rebuild an entrant's genome exactly for the archive (the CPU does the same arithmetic on the same seed; the GPU child is a function of the seed, not of floating point order, if the sampler is written in integer and single-rounded operations, which the data-flow domain must confirm).

Expected. Not steps. It removes half or more of CPU breeding, half of the pack and upload bytes (a plan record is 16 bytes against about 1 KB per packed creature), and it returns GPU clock that CPU load takes today (up to 20%). It is what lets the search feed 2M/s at all (F4).

Risks. Two implementations of the sampler that must agree; the archive needs the child's genome and gets it by recomputation, which is extra CPU work only for entrants (under 1%). Determinism holds by seed.

Measurement. Breeding seconds per 3M generation with and without it, creatures/s end to end, GPU clock under load, and bit equality of a recomputed entrant against a CPU-bred child from the same seed.

### P7. Surrogate pre-selection of children (research track)

Idea. At 3M labeled samples per generation the search produces a training set that any supervised model would envy. A small model (genome and parent features to 5 s distance, trained each generation, evaluated on the next) could rank children before they are simulated. If its Spearman with the 5 s distance is above about 0.6, breeding two candidates per slot and sending the better one raises the quality of what the GPU evaluates. This is pre-screening from surrogate-assisted evolution (Jin 2011 survey). It never replaces a trial; it only chooses which trial to run, so the GPU score stays the only score.

Expected. If Spearman 0.6 holds: 1.2 to 1.5x archive insertions per evaluation. If it is 0.3, nothing. I do not know which, and neither does the literature for this genome. The measurement comes first and is cheap: train on generation g's dump, test on g plus 1, report Spearman and top-20% recall. Only a positive number justifies the design. It also competes with F4 for CPU time, so it would have to run on the GPU or the iGPU between waves.

### P8. Exact-duplicate elimination

Hash each bred genome; a child identical to any creature evaluated in the last N blocks is not sent (its slot breeds again with the next seed). Operators that fail to fit four times and CMA samples with sigma near zero produce such children. Expected 1 to 3% fewer evaluations; measure the collision share per block first, one afternoon. Determinism holds (the hash set is a function of the history).

### P9. Let the archive absorb more: islands and grid size at 3M per generation

The finer grid lost at 16k creatures per generation and the reserve of 256 places was a tie; the backlog says archive changes should be retested at large populations on the GPU. At 3M per generation each of 5 x 1,440 cells sees about 400 offspring, and at 2M/s a generation is 1.5 s. More independent islands (16 isolated plus the hub, each with its own optimizers and reserve) absorb evaluations without any change to what one island does, and isolation won every migration test so far. Expected: unknown until measured; the island literature says diversity and best-of-islands rise with island count when the per-island budget is still adequate (here 187k per island per generation, more than the whole population the current constants were tuned on). Measurement: `search_ab --gpu`, 10 seeds, equal time, 4 vs 8 vs 16 isolated islands, best and QD of the hub and of the union.

## 4. Ranking with numbers

| rank | proposal | gain | unit | risk | first measurement |
|---|---|---|---|---|---|
| 1 | P2 per-arena and per-cell bars (needs P1) | 20 to 25% fewer steps per creature in mature archives; nursery revived | steps and progress | low to medium | offline dump: entrant recall per factor |
| 2 | P3 emitter restructuring and 16 blocks | 1.2 to 1.8x best distance at equal time | progress per GPU second | medium | samples-per-tell histogram, then search_ab |
| 3 | P6 GPU-side parametric children | removes the 640k/s CPU breeding wall, up to 20% GPU clock back | throughput at 2M/s | medium | breeding seconds and clock |
| 4 | P5 cell-aware rungs at 10 s and 2.5 s | 12 to 16% fewer steps each | steps | medium (a global-bar version lost before) | Spearman and recall from the dump |
| 5 | P9 more islands at 3M | unknown, absorbs evaluations | progress | low | search_ab at equal time |
| 6 | P7 surrogate pre-selection | 1.2 to 1.5x insertions per evaluation if the model ranks | progress | high (may be zero) | train and test Spearman |
| 7 | P8 duplicates | 1 to 3% | steps | none | collision share |
| 8 | P4 per-emitter racing bar | under 2% | steps | none once P1 exists | entrant share below emitter median |

Steps per creature under P2 plus P5: about 300, from 480. That is the 1.6x. Everything else in this list is about turning the same GPU seconds into more progress. The 25x is not in my domain; at most 1.6x of it is.

## 5. What I need from the other domains

- HPC/kernel: the cost of a per-creature bar index and table load at the rung steps (P1), and of sampling the distance at two more steps (P5). The smallest wave that keeps the lane-group kernel at full rate with in-kernel regeneration, and the tail cost of a 49k wave (P3): my estimate is that a 1,200-step creature runs about 27 ms at today's per-creature rate, so a 49k wave of about 0.5 s has a 5% tail unless streams overlap waves; I need the real number.
- Data flow: whether a GPU-resident elite table plus per-block emitter tables fit the memory budget beside the owner's game (elites are 5 x 1,504 plus nurseries, under 10 MB; emitter tables under 8 MB), and whether the result record can carry a seed back (P6). Also whether the bar table can travel with the block's settings (P1).
- Physics: how many gait cycles a typical elite completes in 2.5 s, the distribution of fall times under the new physics (how much of the 480 falls already remove), and whether behaviour at 5 s predicts the 20 s cell (the descriptor components are running averages, so they should, but feet and contact bins can flip on one late touchdown). This decides P5 and the neighbourhood radius in P2.
- The owner (through the chair): approval in principle for rungs whose bar comes from the creature's own cell and emitter rather than one population number. It stays natural selection and it stays distance only; the change is that the competition the screen applies is the same local competition the archive applies.
- Anyone with a mature 3M save on the new physics: the dump in P2 is the one measurement that decides P2, P4 and P5 together, and it needs nothing but a diagnostic env var that writes `screen_x`, the sampled distances, the final distance, the cell and the parent cell per creature to a CSV.
