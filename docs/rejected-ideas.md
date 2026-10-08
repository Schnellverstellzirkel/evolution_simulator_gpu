# Rejected ideas

Each idea was measured and lost. Do not redo one without a new reason. Numbers are from the measurement that decided it.

## Search

- Compound operators that add nodes and give none back (a mirrored limb pair, a repeated segment with a gradient, a new part with its timing): 16 seeds, 100k creatures, 30 generations, ring bodies grew to 11.0 nodes and 16.6 muscles against 8.2 and 13.7 on main, QD x0.84 (the standard error of one arm's log ratio is 0.16) and best x1.01. Bigger bodies cost the GPU more per creature. On the lane-group kernel a body past 8 or 16 nodes took twice the lanes. The three (`mirrored_limb_pair`, `segment_chain` and `grow_integrated_limb`) now remove the idlest limb tips for the nodes they add (bodies stay at 8.2 nodes, QD x0.98).
- Removal operators that took a random limb tip or muscle (`remove_limb`, `remove_muscle`, both deleted): 40 generations, 10 seeds, best 275 m against 346 m and QD 21,329 against 30,075. The owner wants more operators, so this is a note about growth, not a reason to prune. The game keeps `prune_idle_limb` and `prune_weakest_muscle`, which take the idlest tip or the weakest muscle of three.
- Neutral new muscles for structural mutations: best 21.3 m against 25.6 m, QD 2,343 against 2,617 over 10 seeds. A split still breaks the gait.
- Elite refresh with a fresh perturbation: best fell 18% and QD did not improve at 60 generations (3 of 10 wins).
- CMA-MAE thresholds: best x1.07 and QD x0.98 at rate 0.1, x0.82 and x1.04 at rate 0.02.
- UCB bandit for emitter shares: x0.96 best and x0.98 QD once the shares followed reward.
- Iso+line variation between same-plan elites: best x1.32 from one seed, QD x0.99.
- Mating rate of structural and novelty children from 0.2 to 0.5: best x1.18, QD x1.06, within noise.
- Self-adapted mutation scale from the `mutability` gene: x1.23 best on the first 18 seeds and x1.00 on nine more. The gene was deleted.
- CMA emitters in normalized units that also search joint ranges and touchdown phases: best x1.22, QD x1.02. The island optimizer, in physical units, does search both.
- A varied first population (chains, bilateral bodies, trees): QD 1,620 against 1,848 over 24 seeds.
- An L-system source for structural children: QD x0.98.
- Distillation of new muscles from the nearest kept muscle: QD x0.91. Lamarckian tuning of new bodies by the CMA share: best x0.87.
- A node-count archive axis alone, on the CPU at 16k creatures per generation: best x0.77 with 0 wins of 8 seeds, and as a replacement for cadence or feet QD x0.73 and x0.65. At 500k on the GPU, a node-count class and a shape class beside the 1,440 ways of moving held best distance and QD (see `docs/design-decisions.md`). One class axis alone (3 sizes or 3 shapes, 4,320 cells) held best distance within 9% on two seeds and gave fewer clades than the 2 by 2 layout of that time.
- A tie margin on archive replacement (a child must beat its elite by 1e-4 of the distance): no change in clades or speed, because the archive already changes by 4 of 1,431 elites per generation at the plateau.
- Island classes from the first generation: a fresh game at 1M creatures reached 32.0 m at generation 40 against 38.5 m with one cell per way of moving (seed 41), and every island was lower. Islands refine after 30 generations instead (`qd::REFINE_AFTER`), the four isolated ones 10 generations apart.
- The rarity bonus for a rare clade at full weight in a climbing island: island QD 27% lower and best 35.1 against 38.5 m at generation 40 (1M, seed 41). It counts in refined islands, scaled by the share of level elites.
- A protection of 30 generations for a new body plan against a challenger of another plan: the global archive's effective clades grew 75% on the owner's save (121 against 69), but a fresh game at 1M had a global QD 51% lower at generation 40, because the window is most of the run. The nursery of reshaped bodies protects new plans instead.
- A margin of 0.5% for a challenger of another body plan to take a cell: no change on the owner's save (252 clades, effective 69.8, against 286 and 69.3).
- A morphology reserve of 1,024 places: 2% more body plans and 14 MB more save on the owner's save.
- A finer grid (8 x 10 x 8): best x0.86, QD x0.83 at 16k creatures.
- A reserve of 256 places: best x1.17, QD x1.01. A reserve parent share of 25%: best x1.02. Both together: best x1.04, QD x0.96.
- Periodic island extinctions: best x0.97 and QD x0.98 every 15 generations, x0.94 and x0.99 every 8 generations with half the grid.
- Migration every 5 generations: the islands converged and reached 847 m against 1,556 m every 25.
- A second screening rung after the 5 s screen, at 10, 15, 20 or 30 s of a 60 s trial: only 30 s keeping 60% held QD, and that one gained 9% end to end. It was removed when trials went to 20 s, with no 20 s measurement, because the owner judged that screening makes less sense at 20 s.
- Screened creatures opening empty archive cells: tied at equal time and lost per evaluation.
- Cheaper contender checks, on 60 s trials against a check at 4x the rate: 2x or 3x the rate, or ending at 20 or 30 s. Every one kept more of the score. The 2x check let exploits through, and the top 50 kept 12% of their distance from an unseen pose against 44%. The confirmation trial now runs at 2x, with the kernel's 16 substeps per step (`physics::Fidelity::fine`).
- Stopping a trial once it cannot beat its cell: impossible, because the cell needs behavior measured over the whole trial.
- A nursery of reshaped bodies bred from fresh structural children of island elites (a cohort that starts over every 10 generations, 5% of the slots taken from the island): QD x0.95 at generation 1,612 after a world change (35,205 against 37,044, 300k a generation). Adding such children to half the slots of the nursery that turned-away bodies feed gave x0.95 against x1.006 without them. The island's own structural children already do that work, so the slots bought nothing.
- Three extras for the reshaped nursery, tried together on top of it, one run each at 300k from generation 1,598: a reseed of its bodies after a world change, routing the elites that lost their cell during the re-test into it, and promotion every generation. They showed no gain: the plans held by all archives were 9,808 against 9,662 without them, and QD at generation 1,633 was 39,578 against 42,999 (the game as it is: 41,359 and 38,976 on two seeds).
- A nursery of new random bodies that is never wiped (300k a generation): its best body went from 5.4 m to 13.4 m in 60 generations against 33 m in the island, and a world change clears the archives of the main islands (autochange makes one every 100, 50 or 20 generations). It stays a 10-generation cohort.
- The island takes the reshaped bodies every 20 generations instead of 10 (300k from the owner's save, one seed): 747 island cells held a graduate against 1,254, and the island plans were the same.
- A refined layout for the nursery of new random bodies as well (one seed): island plans 14,876 against 14,822 with the refined reshaped nursery alone, graduate cells 2,188 against 2,172.
- A nursery of new random bodies at 10% and the reshaped one at 10%, both refined, so the island keeps 80% (one seed): island plans 15,172 (+10.5%), but QD only 140,140 against main's 139,830, and the effective clades of the islands rose 2% where the 5% and 10% split rose 12% (one seed each).
- Keeping the novelty and local-competition scores of the reshaped nurseries between their once-a-generation refreshes. Today every new cell, every absorbed graduate and every elite that leaves wipes them (`QdArchive::note_changed_cell`), so for most of a generation the nurseries pick parents by visits, rarity and chance. archive_bench, 200k for 30 generations, seeds 38 to 40: cells 4% fewer, plans 3 to 4% fewer, effective clades 491, 518 and 562 against 560, 633 and 680. Stand-in scores, no GPU run.
- Parent tournament rules from the biodiversity list: dominated novelty rank, curiosity counts, plan rarity pull, rest for barren elites, clearing per plan, class-first draws and tournament sizes 4, 6, 8 and 12 per island. With the kept variation changes on, search_ab 4 generations at 200k (seeds 8 to 10): clearing and class draws 3.42 m against 5.08 m without them; clearing alone 4.03 m, tournament sizes alone 3.92 m, the pulls alone 3.95 m against 4.99 m (seeds 8, 9). QD scores were level. Distance loses, so they were deleted.
- Island and nursery ideas from the biodiversity list, search_ab 30 generations at 200k, seed 8 against main (26.46 m, 8,200 plans, 368 effective clades): equal nursery slots per plan, a median floor for nursery parents, staggered graduation and a cap of 20 graduates per plan gave 24.10 m, 6,636 plans, 244 clades; staggered and per-plan migration with the plan tie break gave 24.60 m, 7,826 plans, 328 clades, and without the tie break 23.39 m. Reserve eviction from the most crowded body type, 70 generations: effective clades 122 against 275, largest clade 0.16. The hub rescue of stalled islands never fired in 70 generations.
- Not measured, no verdict: a ladder of age layers beyond the two nurseries, deep grids, racing, and dominated novelty search as a whole (only its rank in the parent tournament was measured, above).

## Physics

- 30 Hz physics, before the position-based kernel: 30 Hz elites kept a median 38% of their distance at 60 Hz, so the gains were integrator exploits.
- Muscle mass as a way to hold muscle counts down, in four variants and a massless control (1 or 4 kg/m, span or flat energy store): none held muscle counts down and best distance fell by 51 to 62%. The game still weighs a muscle (0.05 kg plus 1 kg per metre of its slack length, `physics::add_muscle_masses`), but the weight does not limit muscle counts.
- Fewer contacts for speed, in the articulated-body solver that kept the 4 deepest contacts: 2 contacts +33%, 1 contact +65%, but nodes left out of the solve sink. It changed the physics.

## Speed

Most entries below were measured before 2026-10-06 on designs the game no longer has: trial segments, work units sized by GPU time, the CPU engine and the lane-group kernel. `shaders/creature.cu` replaced that kernel and has no lanes, muscle rounds, contact matrix or sweeps.

- The first persistent-lane engine (`claude/lanes`): 148,121 creatures/s end to end against 243,857 for segments. It compiled to 155 to 168 registers and left sparse warps. It would need 128 registers or fewer and GPU-side compaction.
- A muscle waveform cache: 60.3k to 60.7k creatures/s against 63.8k to 64.8k. A branch-free waveform: 61.2k to 62.8k against 64.2k and not bit-exact on the GPU.
- Keeping trial metrics in a global slot instead of registers: no faster.
- Grouping 2 or 4 muscles per loop iteration: 79.9k, 79.1k and 79.3k creatures/s, no change.
- A kernel specialized for one body plan: 10 to 20% faster on a single-plan population, no gain in mixed units.
- Behavior metrics every 2 or 4 steps: no speedup, and it moved 6.8% and 46% of creatures to another cell.
- A device-side contender filter: the ceiling is about 0.5 s of a 10 s generation, and it needs kernel code in three places.
- An overlap thread for archiving and breeding: it can hide only about 1.7 s of a 10 s generation.
- CUDA register caps of 96 or 80, a local-memory table, and blocks of 32, 64 or 128 threads for physics v2: 0.85 to 0.98 of the default.
- One children-first pass for the contact matrix, and the same pass in registers: within noise.
- Contact sweep counts of 8, 4 or 2: 84M to 86M creature-steps/s at every count.
- Projected Jacobi instead of Gauss-Seidel in the lane-group kernel (4 sweeps, relaxation 1, 0.7, 0.5): random bodies travel up to 2.1 m and three to seven times as many fall.
- One children-count bound per step instead of per tree level in the articulated-body pass, and 3 blocks per SM without spills: 3% and 8% slower.
- GPU step ranges of 128 or 256, and extra segment boundaries: slower than 64 and two segments.
- Thread counts: 137,824 creatures/s with 8 general workers, 129,792 with 16, and 121,273 with 8 plus 8 CPU evaluation workers beside a healthy GPU. The CPU engine is deleted. The general pool now takes every CPU but two (`threads::pool_threads`).
- GPU work units of 0.5 s, 2 s or 3 s: 168k, 180k and 168k creatures/s against 187k for 1 s.
- More standard slots with CUDA's default 8 hardware queues: 12 slots took 50.3 s for generation 3 at 1M against 50.8 s for 4, because kernels on streams that share a queue run in order. The queues are set to 32 (`docs/design-decisions.md`). 32 slots took 20.6 s against 18.0 s for 24 even then, with two streams per slot.
- Giving a unit the smallest slot and the smallest spare host buffer that hold it, with all 24 standard slots alike: peak GPU memory 2.9 GB at 1M and 3.7 GB at 3M, against 2.7 and 3.3 GB without it. Under load the small slots are all busy, a small unit takes a big slot, and the next main unit grows another. The first 4 slots for big units are used instead (`docs/design-decisions.md`).
- One engine slot kept free for confirmation trials, with a ring of 5 blocks of 32k: 24.7k against 24.3k creatures/s side by side on a shared GPU, and the same GPU idle time, because a confirmation still waits for a running wave to free the multiprocessors. The game's own fifth slot for confirmation trials, on a stream of the highest priority, is a different layout (`docs/design-decisions.md`).
- Transparent huge pages for large blocks (`MADV_HUGEPAGE`, whole 2 MiB mappings): page faults fell from 450,000 to a few thousand per generation at 1M, but direct compaction stalled single generations for 2 to 5 s while the machine was fragmented. Reuse of freed blocks (`src/block_alloc.rs`) is used instead.
- Substep ladder on the lane-group kernel, rungs above L0 (seed 38, 30 generations at 3M, top 300 re-tested at 4 substeps): a realized-friction-work ledger (best 45 m against 67 m at 1 substep alone, 24.6 m held at 4 substeps against 39.6 m), anchored friction with a store (ratio 0.33, 2.5% of muscle work as friction work) and spin-adaptive substeps (ratio 0.88, 1.07x speed because a warp runs its groups' largest count) all lose to plain 1 substep. The code is at 11818bf in the history (deleted by d809226).
- One kernel for all worlds (effects as runtime flags in the kernel parameters, `#if` blocks as uniform branches): 16% slower than the per-world kernel in the calm world (top 3 of 40 alternating pairs on the 262k dump, 61.8M against 73.8M steps per GPU-busy second, spills 122 to 148 B) and 1.4% slower with four effects on. Warm world changes already cost 0 s without it.
- Loading a muscle round's records and end lists at the top of the round: 2.4% on 16 lanes, but the replay of 138 of 400 random 8-lane bodies differed from its score in the last digits, because the scoring and the recording build fused float operations differently. Each of the two loads alone passed. The test `gpu_replays_show_the_gpu_score` in `tests/gpu_repeatability.rs` compares every replay with its score bit for bit.
- `--ftz=true` and `--use_fast_math` on the 16-lane kernel: 0.3% and 0.8%. A prefetch of the first muscle round before the substep's setup: nothing more.
- A memo of confirmation results by genome: of 169,014 confirmations on the owner's save (generations 1,512 to 1,518), none repeated a genome that an earlier confirmation or the same block had. The plateau creatures that tie the record are all different bodies that reach one exact distance.
- Partial behavior-score refreshes that rescore only the elites a changed cell can reach (each elite's search radius kept, and the cache kept through new cells and removals): 1.64M elites rescored in 10 generations against 2.55M, and the same wall time in archive_bench (measured while the islands still refreshed one after another). A two-cell neighborhood holds about 40% of a body class's cells, so a few changed cells reach most elites. The game already rescores only the elites within two cells of a changed cell while no cell is added or removed.
