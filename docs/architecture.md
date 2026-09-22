# Architecture

`config` owns validated, serializable experiment parameters. `evolution` stores genomes in contiguous arenas and performs deterministic parallel creation/mutation. `physics` is the CPU reference and interactive replay engine. `gpu` implements the same equations in WGSL. `storage` owns generation stages, statistics, and atomic checkpoints. `worker` runs the experiment off the UI thread. `ui` renders native controls and the creature scene on the same wgpu device as compute. The CLI uses the same experiment model and GPU implementation.

## Physics

The fixed timestep is 1/120 second. Muscles have smooth contraction/extension phases with independent period, phase, duty cycle, and stiffness. Each node gathers incident spring forces using the previous step's positions/velocities, integrates semi-implicitly, applies frame-independent air retention, and resolves ground/rectangle contacts with Coulomb friction. Node mass is proportional to diameter squared and bounded away from zero. Forces and mutation parameters are bounded; nonfinite or runaway states are marked as failed trials.

The first 200 steps settle a body with stationary muscle targets, without gravity or contacts. Nodes are then horizontally centered and placed on the ground, velocities reset, and the timed trial begins. Fitness is average node X displacement in meters, including negative values for backward walkers. Creatures do not collide with other creatures.

## GPU execution

Creatures are grouped into 8/16/32/64-node buckets. Each workgroup has 64 lanes; multiple small creatures share a workgroup. Node positions and velocities live in workgroup memory during a dispatch. Each lane gathers forces in a stable muscle order, requiring no floating-point atomics. Barriers separate gathering and integration. Padded lanes participate in every barrier.

A batch keeps its transient state in reusable GPU storage buffers between dispatches. Responsive mode limits dispatches to 32 steps; throughput mode uses 128. Encoding and parameter uploads overlap the previous dispatch; queue fences allow at most two compute submissions in flight so rendering can interleave. Fitness readback asynchronously maps a compact staging buffer only when a bucket has finished; its completion is awaited on the worker, never the UI thread. Full node readback is used only by validation. Population creation, ranking, selection, and mutation run on the CPU using Rayon.

CPU and GPU use identical data definitions and equations, but floating-point transcendental functions can differ. Fixed seeds reproduce generation/mutation streams regardless of Rayon scheduling. Chaotic contact dynamics can eventually diverge across adapters, drivers, or shader/compiler changes; cross-device bitwise replay is not promised.

## Memory and history

CPU genomes are packed in arenas with per-creature offsets. Population assembly uses bounded chunks and contiguous concatenation; it does not retain a million heap-allocated creature objects. Evaluation creates only a bounded batch of padded GPU states. UI snapshots contain a page of at most 120 creature cards and aggregate statistics.

Generation history stores 29 percentiles, centimeter histogram bins, actual body-type counts, settings, and three representative genomes. It does not retain every generation's full population. Checkpoints preserve the current full population and exact completed-evaluation boundary. Evolution checks its estimated temporary-memory requirement before constructing offspring.

## State transitions

`Ready → Evaluating → Evaluated → Ranked → Selected → Ready (next generation)`

Guided mode pauses at each visible stage. Continuous mode repeats them. New parameter sets apply before evaluating a new generation, so one generation never mixes environments or fitness criteria. Saved stage/progress determines which work resumes after loading.
