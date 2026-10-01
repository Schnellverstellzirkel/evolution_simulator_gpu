# Rejected ideas

Each idea was measured and lost. Do not redo one without a new reason. Numbers are from the measurement that decided it.

## Search

- Removal operators (`remove_limb`, `remove_muscle`): 40 generations, 10 seeds, best 275 m against 346 m and QD 21,329 against 30,075. The owner wants more operators, so this is a note about growth, not a reason to prune.
- Neutral new muscles for structural mutations: best 21.3 m against 25.6 m, QD 2,343 against 2,617 over 10 seeds. A split still breaks the gait.
- Elite refresh with a fresh perturbation: best fell 18% and QD did not improve at 60 generations (3 of 10 wins).
- CMA-MAE thresholds: best x1.07 and QD x0.98 at rate 0.1, x0.82 and x1.04 at rate 0.02.
- UCB bandit for emitter shares: x0.96 best and x0.98 QD once the shares followed reward.
- Iso+line variation between same-plan elites: best x1.32 from one seed, QD x0.99.
- Mating rate of structural and novelty children from 0.2 to 0.5: best x1.18, QD x1.06, within noise.
- Self-adapted mutation scale from the `mutability` gene: x1.23 best on the first 18 seeds and x1.00 on nine more. The gene was deleted.
- CMA that also searches joint ranges and touchdown phases: best x1.22, QD x1.02.
- A varied first population (chains, bilateral bodies, trees): QD 1,620 against 1,848 over 24 seeds.
- An L-system source for structural children: QD x0.98.
- Distillation of new muscles from the nearest kept muscle: QD x0.91. Lamarckian tuning of new bodies by the CMA share: best x0.87.
- A node-count archive axis: best x0.77 with 0 wins of 8 seeds. As a replacement for cadence or feet, QD x0.73 and x0.65. The earlier gain of a body-size axis came from having fewer cells.
- A finer grid (8 x 10 x 8): best x0.86, QD x0.83 at 16k creatures.
- A reserve of 256 places or 25% parent share: best x1.17 and x1.02, QD x1.01 and x0.96.
- Periodic island extinctions: best x0.97 and QD x0.98 every 15 generations, x0.94 and x0.99 every 8 generations with half the grid.
- Migration every 5 generations: the islands converged and reached 847 m against 1,556 m every 25.
- A second screening rung at 10, 15, 20 or 30 s: only 30 s keeping 60% held QD, and the gain was 9% end to end. Screening makes less sense at 20 s trials.
- Screened creatures opening empty archive cells: tied at equal time and lost per evaluation.
- Cheaper contender checks (2x or 3x rate, or ending at 20 or 30 s): every one kept more of the score. The 2x check let exploits through, and the top 50 kept 12% of their distance from an unseen pose against 44%.
- Stopping a trial once it cannot beat its cell: impossible, because the cell needs behavior measured over the whole trial.
- Not measured, no verdict: age-layered populations, deep grids, racing, dominated novelty search.

## Physics

- 30 Hz physics: 30 Hz elites kept a median 38% of their distance at 60 Hz, so the gains were integrator exploits.
- Muscle mass in five variants (1 or 4 kg/m, span or flat energy store): none held muscle counts down and best distance fell by 51 to 62%.
- Fewer contacts for speed: 2 contacts +33%, 1 contact +65%, but nodes left out of the solve sink. It changes the physics.

## Speed

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
- 16 general workers, or 8 CPU evaluation workers beside a healthy GPU: 129,792 and 121,273 creatures/s against 137,824.
- GPU work units of 0.5 s, 2 s or 3 s: 168k, 180k and 168k creatures/s against 187k for 1 s.
- One engine slot kept free for confirmation trials, with a ring of 5 blocks of 32k: 24.7k against 24.3k creatures/s side by side on a shared GPU, and the same GPU idle time, because a confirmation still waits for a running wave to free the multiprocessors.
- Transparent huge pages for large blocks (`MADV_HUGEPAGE`, whole 2 MiB mappings): page faults fell from 450,000 to a few thousand per generation at 1M, but direct compaction stalled single generations for 2 to 5 s while the machine was fragmented. Reuse of freed blocks (`src/block_alloc.rs`) is used instead.
