# Design decisions

What the game does in search, physics and speed, and the number that decided each choice. Each item stays true until a direct measurement of a change shows the game is better or faster without it. Ideas that lost are in `docs/rejected-ideas.md`.

## Search

Fitness is horizontal center-of-mass distance and nothing else. Behavior (ground contact, cadence, body height, lifted feet) picks the archive niche and never changes the score. The physics rules that end a trial (a fall, a broken joint, head shaking above 8 g) keep distance the only objective.

The search is MAP-Elites with four emitters: CMA tuning, anatomy mutations, novelty and random immigrants. Immigrants only seed empty archives. Each island also keeps a nursery: 10% of its slots hold new random bodies and the bodies bred from them, which compete only against each other for 10 generations. Then the survivors enter the island archive and compete on distance alone. The emitter shares stay at the 35% CMA, 35% structural, 30% novelty prior. A reward-following bandit tied with the fixed shares.

Four islands are fully isolated and a fifth hub island receives copies of each isolated island's fastest tenth every 25 generations. Islands that exchanged their best tenth every 5 generations converged on one design within about 15 generations. At 241 generations one seed reached 847 m with migration every 5 generations, 1,243 m with none and 1,556 m every 25.

Each archive has 1,440 niches: 6 ground contact, 8 cadence, 1 vertical oscillation, 6 body height (log scale) and 5 feet. Body height is logarithmic so small and giant bodies do not share one cell. A 64-entry morphology reserve per island gives new body plans offspring without counting toward coverage or QD score. In a 5-seed run the reserve admitted 3.8 times as many first-time body plans at equal best distance (112.8 m against 112.0 m).

Every muscle shares one body clock. This gained 32% best distance on every seed in the first test and was the only change that made bodies of 8 or more nodes competitive (the best such body went from 34 m to 62 m).

The CMA emitter explores in normalized units with a large step (sigma 0.12). Each island also runs a separable CMA-ES in physical units on its fastest body plan. That optimizer needed three things to help: every sample gets the same contender check, it stays on its body plan instead of chasing each new record, and it starts at half the base step. With them one seed went from 455 m to 958 m at 241 generations. A uniform sigma of 0.03 for the exploring emitter lost 40%.

Whole-body rescaling is a structural mutation, so evolution can grow bodies from 0.25 m bones to giants. Together with the log height axis it took one seed from 304 m to 856 m.

There are 51 anatomy operators on top of the 7 classic ones (`docs/anatomy-operators.md`). The first 30 gained 21 to 26% best distance and 16 to 28% QD over 10 seeds at equal evaluations, and the top 50 carried 28% fewer muscles. The owner wants more operator types and never fewer.

Cross-plan crossover grafts a limb with its muscles and rhythm from an elite of another body plan on the same island. Over 18 seeds it gave QD x1.25 and best distance x1.08.

Early screening stops a standard trial at 5 s when the creature is below the bar, the 5 s distance the top 20% reached. Screened creatures enter no archive. The 5 s distance keeps every creature of the final top 1% and 96% of the final top 10% (Spearman 0.887). At equal time screening gives 56% more best distance and 2.4 times the QD, at a cost of 14% QD at equal evaluations. Letting screened creatures open empty cells tied at equal time and lost per evaluation.

Every archive contender gets a check trial from a perturbed pose (nodes moved up to 2 cm, grip varied 10%) at four times the standard rate and solver passes. The lower of the two distances is the fitness. A check at twice the rate let integrator exploits through: the top 50 kept a median 12% of their distance from an unseen pose against 44% with the four-times check.

Trials last 20 s and physics runs at 60 Hz. Elites evolved at 30 Hz kept a median 38% of their distance when replayed at 60 Hz, against 90% for elites evolved at 60 Hz and replayed at 120 Hz.

The search is deterministic for a fixed seed on one GPU. Results are absorbed in a fixed order, and the contender claims on archive cells are ordered.

## Scoring and replays

The GPU score is final. A replay is recorded by the scoring kernel with a frame output on a slot and queue of its own, so its result is bit for bit the archive score (`gpu_replays_show_the_gpu_score`). CPU and GPU agreement is a diagnostic. Bit equality across engines is not expected, because the compilers contract floating point differently.

A GPU that fails is retired and its unfinished units run on the CPU engine with the same creatures. A GPU out of memory keeps its unit, frees idle buffers and retries with fewer units in flight. Buffers above 1 MiB get 25% headroom, which cut peak GPU memory 15%.

Saves hold only the configuration, history, archives, CMA and emitter state, lineage and queued elites. A 3M game saves 3.4 MB in 0.06 s where the full population took 1,389 MB and 24 s. Loading breeds the next generation from the archives in 3.8 s. Autosave is off. An older `qd::VERSION` is turned down by the header before the load starts.

## Physics

The physics is a planar articulated tree in reduced coordinates (`docs/physics.md`). Bones stay rigid by construction, so the projection passes, the rebuild and the lift that a point-mass chain needs are gone. Evolution finds every hole in a physics model, and each rule below closes one:

- Hill's force-velocity relation bounds muscle power. Without it the best creature threw itself 19 m in one throw.
- Contacts are solved together at velocity level, and friction can only oppose the slip. Solved one at a time, sticking contacts carried the load of sliding ones and nodes reached 30 m/s.
- The spin cap is 15 rad/s. At 40 rad/s a short end bone whipped into the ground and 2,887 of 5,761 N s of friction pushed a node along its own slip (the kick sled, 408 m). At 15 rad/s the same body went 0.8 m.
- Joint limits are inelastic stops. A spring limit let a joint pass unopposed and return 25 kJ against 4.9 kJ of muscle work.
- Feet are planted against the end pose of the step. Planting against the start pose left 42% of an elite's friction pushing along its slip. Now it is 0.0 N s.
- Muscle force and energy scale with the mass a muscle drives. Before, a 100 N muscle drove a 0.05 kg limb at 2,000 m/s^2 and one elite gained 338 J against 26 J of muscle drain. With the scaling it gains 9 J.
- Only active contraction is charged to the energy store. Charging the damper and stretched muscles too tied within the noise but paid for work no muscle did.
- Bones feel air drag. Tendons store and return energy. With tendons QD was 13,162 against 6,437 and 12% of the muscles of sampled elites carried one.
- The contact solve keeps the 4 deepest contacts per step. Eight contacts with cold sweeps ran 4.1M creature-steps/s, four with warm sweeps and planting rounds ran 22M and one ran 62M.

Every engine reads the same physics constants. A physics change updates the WGSL and CUDA kernels together, the prototype, and `cpu_v2`.

## Speed

On NVIDIA the game runs CUDA (`src/cuda_engine.rs`). It is 1.8 times Vulkan on an evolved population and equal on a fresh one, so it is never slower. The kernels compile without a register cap, which was 3% faster than a 128 cap.

Work units of 1 s gave 186,700 creatures/s end to end against 168,600 for 3 s units, with 2.5 GB less peak memory. The worker keeps 8 general threads. Adding 8 CPU evaluation workers to a healthy GPU lowered the rate from 137,800 to 121,300 creatures/s. Archive refreshes recompute only the cells near a change, which cut the archive stage from 3.4 s to 2.1 s per generation at 3M. Breeding packs children in batches, and the CPU chain is now under a quarter of a generation, so the GPU is the wall.

Long sessions grow bodies, and every creature then costs more. The gene arenas reserve one generation of children and shrink the spare, which cut peak memory from 18.6 to 14.8 GB. Autosave is off by default because a clone of the experiment pushed a 32 GB machine into swap.

The fast CPU engine runs 16 creatures with one skeleton per SIMD group. It scores 7,800 creatures/s on 4 threads with full groups against 1,000 for the scalar reference, and it is bit-equal to the reference.
