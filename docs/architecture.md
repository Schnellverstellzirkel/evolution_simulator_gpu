# Architecture

`config` owns validated, serializable experiment parameters. `evolution` stores genomes in contiguous arenas and creates deterministic parallel candidate batches. `qd` owns behavioral descriptors, the MAP-Elites archive, emitter allocation, and diagonal CMA-ES state. `physics` is the CPU reference and interactive replay engine. `gpu` implements the simulation and measures behavior in WGSL. `storage` owns generation stages, statistics, and atomic checkpoints. `worker` runs the experiment off the UI thread. `ui` renders controls, the behavior archive, and the creature scene on the same wgpu device as compute. The CLI uses the same experiment model and GPU implementation.

## Physics

The fixed timestep is 1/120 second. Muscles have smooth contraction/extension phases with independent period, phase, duty cycle, and stiffness. Each node gathers incident spring forces using the previous step's positions/velocities, integrates semi-implicitly, applies frame-independent air retention, and resolves flat-ground contact with Coulomb friction. Node mass is proportional to diameter squared and bounded away from zero. Forces and mutation parameters are bounded; nonfinite or runaway states are marked as failed trials.

The first 200 steps settle a body with stationary muscle targets, without gravity or contacts. Nodes are then horizontally centered and placed on the ground, velocities reset, and the timed trial begins. Fitness is average node X displacement in meters, including negative values for backward walkers. Creatures do not collide with other creatures. The GPU also records ground-contact fraction, center-of-mass vertical range, and observed gait cadence from center-of-mass turning points during the timed trial. New experiments default to 9.8 m/s² gravity, 0.985 velocity retention, node friction in [0.65, 1.0], a 1.5 ground-friction multiplier, varied node diameters from 0.06–0.12 m, and 18-second trials on flat ground.

## GPU execution

Creatures are grouped into 8/16/32/64-node buckets. Each workgroup has 64 lanes; multiple small creatures share a workgroup. Node positions and velocities live in workgroup memory during a dispatch. Each lane gathers forces in a stable muscle order, requiring no floating-point atomics. Barriers separate gathering and integration. Padded lanes participate in every barrier.

A batch keeps its transient state in reusable GPU storage buffers between dispatches. Responsive mode limits dispatches to 64 steps; throughput mode uses 1024. All bucket dispatches and result copies are encoded into one command submission, and evaluation reads back one combined result region per batch. Full node readback is used only for replay validation. Population creation and offspring generation use CPU parallelism. Archive insertion and emitter feedback run once after a complete population batch.

CPU and GPU use identical data definitions and equations, but floating-point transcendental functions can differ. Fixed seeds reproduce generation/mutation streams regardless of Rayon scheduling. Chaotic contact dynamics can eventually diverge across adapters, drivers, or shader/compiler changes; cross-device bitwise replay is not promised.

## Memory and history

CPU genomes are packed in arenas with per-creature offsets. Population assembly uses bounded chunks and contiguous concatenation; it does not retain a million heap-allocated creature objects. Evaluation creates only a bounded batch of padded GPU states. The archive grid uses three measured behavior descriptors: ground-contact fraction, observed gait cadence, and center-of-mass vertical oscillation. It has 192 possible cells; each stores its highest-fitness creature, emitter source, topology, protection period, and visit count. Novelty is the mean normalized distance to the five nearest archived behaviors. Local competition compares an elite's fitness with behaviors in adjacent cells; emitter parent selection uses those local scores, while novelty parents favor underexplored and behaviorally isolated cells.

The four emitters start at 30% diagonal CMA-ES over a fixed topology, 30% constructive morphology, 25% novelty, and 15% random immigrants. A UCB-style bandit adjusts allocations from archive discoveries and improvements while retaining a prior-based exploration floor. Novelty samples underexplored and behaviorally isolated niches; an emitter with five stagnant batches restarts from a different niche. Structural edits split a muscle or mirror a node and its incident muscles, while a third structural operator shifts an oscillator group. New morphologies are protected from replacement by another topology for three generations. No offspring is an unconditional exact clone.

Generation history stores archive percentiles, centimeter fitness bins, counts by body size, settings, emitter feedback, and three representative archive elites. It does not retain every generation's full population. Checkpoints preserve the current candidate population, archive, CMA states, emitter feedback, and exact completed-evaluation boundary. Older checkpoints are migrated by retaining their current population and evaluating it into a new archive. Evolution checks its estimated temporary-memory requirement before constructing offspring.

## State transitions

`Ready → Evaluating → Evaluated → Archived → Ready (next generation)`

Guided mode pauses at evaluation, archive update, and offspring creation. Continuous mode repeats them. New parameter sets apply before evaluating a new generation, so one generation never mixes environments or fitness criteria. Saved stage/progress determines which work resumes after loading.

## Background

The archive follows the MAP-Elites quality-diversity pattern: [Mouret and Clune (2015)](https://arxiv.org/abs/1504.04909). The multiple emitter types and reward-based allocation are inspired by [Multi-Emitter MAP-Elites](https://arxiv.org/abs/2007.05352). The fixed-topology diagonal CMA-ES here is a small custom implementation, not the full CMA-ME algorithm. Three-generation morphology protection follows the broader innovation-protection motivation behind [NEAT](https://nn.cs.utexas.edu/downloads/papers/stanley.jair04.pdf), without NEAT's species model.
