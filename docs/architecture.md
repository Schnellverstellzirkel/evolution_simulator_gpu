# Architecture

## Modules and execution paths

| Module | Responsibility |
| --- | --- |
| `config` | Validated, serializable experiment settings; defaults are 3M creatures and 60 s trials |
| `evolution` | Arena-packed genomes, deterministic creation and breeding, body repair |
| `qd` | Behavior niches, elite archives, emitter allocation, diagonal CMA state |
| `physics` | Shared masses, geometry, limits, and fidelity settings; also an older scalar simulator |
| `cpu_engine`, `simd` | Production CPU evaluation and recorded replay trajectories, in groups of 16 creatures |
| `creature_kernel` | Body-size buckets and packed GPU inputs |
| `vk_engine` | Vulkan buffers, shader compilation, pipelines, and dispatch |
| `engine`, `scheduler`, `gpu` | Threaded evaluation devices, work scheduling, and the evaluation front end |
| `environment` | Reversible ground, gravity, air, grip, heat wave, and drought levels |
| `storage` | Experiment stages, CPU-checked global archive admission, islands, catastrophes, history, checkpoints, migrations |
| `worker` | Background evolution, command handling, snapshots, and autosaves |
| `ui` | egui dashboard and CPU-recorded creature playback |

The production physics paths are `shaders/physics_creature.wgsl` and `cpu_engine::evaluate`. Replay calls `cpu_engine::replay`, which returns recorded frames and a `GpuResult` from one CPU stepping loop. `cpu_engine::trajectory` is the frames-only wrapper. `physics::evaluate` is a compatibility wrapper that canonicalizes one creature through `Population::push` and evaluates it with the production CPU engine. The optional CPU benchmark evaluates a whole population in one batch to fill SIMD groups. The older `physics::step` loop remains for direct tests and the legacy momentum ledger; it lacks production joint limits and fatigue and is not an authoritative production oracle. Shared helpers such as `physics::body`, `Fidelity`, and `Limits` remain authoritative inputs to every engine. Physics changes must keep the production engines aligned and address the legacy step path deliberately.

## Physics and scoring

Standard fidelity uses 60 steps/s, two bone position-projection passes, and one velocity-constraint pass. `EVOLUTION_PHYSICS_RATE`, `EVOLUTION_BONE_PASSES`, and `EVOLUTION_VELOCITY_PASSES` override these for experiments. Settling lasts about 1.67 s (100 standard steps), before the timed trial. Settling disables gravity and contacts; the body is then centered horizontally by mass, placed on the ground, and its velocities reset.

A skeleton is a connected tree of rigid bones. Bone order is normalized parent-first and muscle anchors are remapped to preserve attachment positions. Each muscle joins two bones at normalized positions along them; forces distribute to the endpoints according to those positions. Joint ranges constrain bending. Bones can also carry organs, whose mass is distributed to their endpoint nodes according to attachment position.

`physics::body` combines three mass contributions: a node's own mass, half of each incident bone's mass, and its share of organ mass. Node mass is `0.1 * (diameter / 0.08)^2` kg, clamped to 0.02–10 kg. Bone mass is `bone_density * rest_length^2`, with default density 4 kg/m². All packing and evaluation paths use these combined masses.

Default limits from `physics::Limits::DEFAULT` are:

| Quantity | Default |
| --- | --- |
| Bone length and muscle long length | 2 m |
| Muscle target speed | 24 m/s |
| Muscle force magnitude | 100 N |
| Node speed | 60 m/s |
| Bone angular speed | 40 rad/s |
| Minimum rhythm period | 0.2 s |
| Muscle energy store | 120 J |
| Recovery | 0.5 of missing energy per second |

The environment effects scale these two baselines per run: `Config::muscle_energy` multiplies the store (heat wave, 1.0 down to 0.35) and `Config::muscle_recovery` multiplies recovery (drought, 1.0 down to 0.1). Both default to 1.0, and every engine applies them to the shared `Limits` values.

Touchdown sensors can restart a muscle's rhythm when its chosen node lands. The active muscle drive is nonnegative and acts only while the target shortens. Lengthening supplies no active push; exhausted muscles have zero active drive. Relative-velocity damping remains part of the force. The current energy debit uses the absolute work of the combined force, including damping; charging only active contraction work remains open work.

Each step integrates forces into predicted positions, projects bone lengths and ground contact, applies joint limits once per step, and rebuilds the tree at exact bone lengths. Grounded nodes are weighted more heavily during the bone and velocity passes (`1 + STANCE_GRIP * node_grip * ground_friction`, with default `STANCE_GRIP = 10`), allowing the body to pivot over planted feet. Joint projection also favors a grounded side when only one side is grounded.

The parent-first rebuild restores exact lengths and recenters by mass. If nodes penetrate the ground, it lifts the whole body; that lift is subtracted when reconstructing vertical velocity, making it a position-only correction. Floor clamps inside bone and joint passes accumulate into the per-node ground push. Per-node Coulomb friction uses that push, and the lift supplies an additional body-wide horizontal correction bounded by the contacting feet's weighted grip.

Planting feet alone can create propulsion through weighted projections. The current solver caps the body's horizontal center-of-mass shift from the projection/rebuild stage to the grip coefficient times accumulated normal correction divided by body mass. It removes excess displacement with a rigid translation, so the feet slip when the budget is exhausted. Normal correction includes contacting-node pushes and the remaining body's share of whole-body lift. Velocity constraints then remove radial bone motion and bound rotation. This implementation is not a general proof of mechanical-energy conservation; `examples/first_generation` checks random bodies for excessive free propulsion after physics changes.

The default environment has gravity 9.8 m/s², air velocity retention 1.0 per 1/60 s, ground-friction multiplier 1.5, node grip 0.65–1.0, node diameters 0.06–0.12 m, and both muscle multipliers at 1.0. Terrain can add deterministic bumps. Creatures do not collide with one another.

Fitness is horizontal center-of-mass displacement after centering the start pose. It has no posture factor, stepping multiplier, size penalty, or energy bonus. A head dropping below its neck base, a joint more than 0.5 rad beyond its range, or excessive head shaking records the distance at that event and disables muscle force. The shaking rule uses the magnitude of per-step head acceleration, exponentially averaged over about 0.1 seconds, with an 8 g limit (`HEAD_SHAKE_LIMIT = 78.4 m/s²`). Its accumulator starts after the initial 0.1 seconds of the timed trial; invalid/nonfinite states receive the failed-trial sentinel. Measured ground contact, vertical oscillation, cadence, body height, and lifted feet are behavior descriptors, separate from fitness.

## Contender checks and replay

In the graphical worker, candidates that could enter the global archive, an island archive, or the topology reserve are held for a check. Optimizer offspring are also checked. The scheduler perturbs starting node positions by up to 2 cm and grip by ±10%, then evaluates at `Fidelity::fine()`: four times the standard rate and solver passes, with rate capped at 960 Hz. At the defaults this is 240 Hz, eight bone passes, and four velocity passes. The returned fitness is the lower distance from the standard and fine trials; descriptors remain from the standard trial.

The blocking `Gpu::evaluate_with_metrics` path, used by the headless CLI, has no archive callback and checks every evaluated candidate. `Scheduler::evaluate_single` explicitly bypasses the additional check for engine comparisons. `EVOLUTION_ROBUST_TRIALS=1` disables checks for diagnostics; the normal default is two trials for contenders.

Global archive admission adds a CPU validation step in `storage`. From each batch it chooses the best eligible candidate for each behavior cell and each new topology, evaluates that subset at standard fidelity on the CPU engine, and retains the lower of the candidate's existing score and this CPU distance. It replaces descriptors with CPU-measured behavior before checking the resulting global niche again. The island path and scheduler checks remain separate; this CPU admission check applies to the global archive players browse.

`cpu_engine::replay` returns `(frames, result)` from the same unperturbed recorded run at the supplied configuration's fidelity. The viewport uses `result.fitness` and `result.fall_time`, including joint-break and head-shaking termination. The archive may hold a lower distance because it keeps the worst of its checks. Cross-device rounding and terminal thresholds can still change outcomes; GPU agreement tests must compare the same fidelity and trial setup.

## GPU and CPU scheduling

The Vulkan kernel evaluates one creature per lane. Bodies are bucketed by node capacity: 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, and 64. Within each bucket, sorting by body size groups similar loop counts. Bones and muscles are packed as `[item][field][lane]` in 32-creature tiles. Node state uses `[node][lane]` workgroup memory; node and bone constants use private arrays. Each muscle is evaluated once and scattered to its endpoint nodes in genome order.

`vk_engine` uses ash and compiles WGSL through naga. It submits all independent bucket dispatches for a step range before inserting the next shared barrier. The default step range is 64. Two submission slots per device permit work to be queued while another unit runs. GPU allocation statistics count application buffers, not total driver VRAM.

`Scheduler` gives devices bounded work units, uses measured rates to adjust their size, and handles node-capacity restrictions. Its default target is about one second for primary GPU and CPU units. Results can arrive out of order; the worker tracks completed flags and maintains a contiguous completed prefix for checkpointing. Contender checks are scheduled separately before their held standard results become final.

Devices carry an explicit GPU or CPU kind and queued units carry a retry count. A GPU that reports a failure is retired, and every unfinished unit, including pending fine checks, is re-submitted to a healthy CPU engine with its exact population, configuration, trial kind and ticket order. A failed submission leaves its creatures in the round for the next engine. A failed CPU is terminal: results already completed are delivered first, then the error persists so no drain loop retries forever. If the primary GPU cannot open, `Scheduler::new` falls back to the CPU instead of failing; when `EVOLUTION_CPU_THREADS=0` disables the dedicated pool, the fallback evaluates on the general Rayon pool. Explicit `gpu_engine` constructors remain strict.

On the owner's workstation, always set `EVOLUTION_DEVICES=primary` and `EVOLUTION_CPU_THREADS=6`. These select the primary compute GPU plus six CPU evaluation workers, excluding the desktop Radeon. Additional GPUs require an explicit `EVOLUTION_DEVICES` selection. Evaluation and global Rayon pools share one budget: half the available logical CPUs, capped at eight workers. Evaluation reserves up to six by default, leaving at least one general worker. On the 16-thread workstation this means six evaluation plus two general workers; an eight-CPU affinity gives three plus one. Global workers honor `RAYON_NUM_THREADS` within the remainder, while `EVOLUTION_CPU_THREADS=0` releases the full budget to them. A one-worker budget disables scheduler CPU evaluation; an explicit CPU-only caller can still run one evaluation worker. The UI has its own wgpu render device and targets 60 FPS while evolving or playing back; `EVOLUTION_RENDER_GPU` and `EVOLUTION_UI_FPS` are diagnostic overrides.

The CPU engine processes 16-lane groups. `simd` uses AVX-512 when enabled at compile time and a portable array implementation otherwise. Linux x86-64 builds select the local CPU through `.cargo/config.toml`; CI overrides that with a generic x86-64 target to exercise the fallback. CPU evaluation runs in a separate low-priority Rayon pool.

## Archives, emitters, and memory

Genomes are stored in contiguous arenas with per-creature offsets. Assembly and GPU upload use bounded chunks rather than retaining millions of separate heap-allocated creature objects. Defaults allow 32 nodes and 96 muscles per body; supported configuration maxima are 64 and 256.

Each behavior archive has `6 × 8 × 1 × 6 × 5 = 1,440` cells: ground contact, gait cadence, vertical oscillation (one bin), logarithmic mean body height, and distinct nodes that touched down and subsequently lifted clear. The latter excludes continuously dragged nodes. A cell retains its highest-fitness eligible creature, with source, topology, protection period, and visit count. New morphologies receive three generations of protection against a different topology. Novelty uses normalized distance to nearby archived behaviors; local competition compares fitness with nearby cells.

A separate reserve holds up to 64 new topologies. Reserve parents get 10% of structural-emitter trials when available and at least eight selected offspring opportunities before ordinary eviction. Reserve entries do not contribute to behavior coverage or QD score.

Four island archives retain independent parent pools, alongside the global display archive. Every 25 generations each island sends its fastest 10% to its neighbor. Initial emitter shares are 35% CMA, 35% structural, 30% novelty, and no immigrants once the archive is established; random immigrants seed empty archives. Reward-based allocation adapts these shares. Structural mutations can split or duplicate limbs, retime oscillators, and rescale bodies. Half the CMA parents come from the fastest 1% of their island; half of those offspring use island optimizers. Optimizers search fixed body plans and rotate to other fast designs after 30 generations without an island record.

History retains summary statistics, centimeter fitness bins, body-size counts, settings, and representative creatures. The current population, archive, islands, CMA state, lineage, and completed evaluation boundary are checkpointed. QD/physics semantics are currently version 19 in `qd::VERSION`; loading supported older states repairs their population and clears obsolete archives before reevaluation. Historical results remain records of their original rules.

## Catastrophes

`Experiment::meteor(0.5)` independently removes each elite with probability one half from the global archive and every island. `Experiment::extinction` clears the nonempty island with the slowest best elite. Both keep the removed entries as fossils with their archive identity. `undo_meteor` returns fossils only to empty cells or cells containing a slower elite, then rebuilds affected indices. The Environment panel exposes Meteor strike, Extinction, and Undo through worker commands. These operations open archive space; they do not change fitness. Fossils are runtime-only (`serde(skip)`), so undo history is not restored from a checkpoint.

## State and storage

The normal generation path is `Ready → Evaluating → Evaluated → Archived → Ready`. Guided mode pauses between evaluation, archive insertion, and offspring creation; continuous mode repeats them. The worker services UI commands and asynchronous evaluation completions separately. World changes collect pending work, invalidate old evaluations and archives, and queue elites to be retested. Generational breeding restores those queued elites before submitting slices to evaluation devices.

V4 checkpoints use a versioned header, compressed binary payload, and checksum. The V4 resume record stores `island_progress`, preserving the optimizer's stall history; V3 remains readable without that metadata. Old QD-version loads clear obsolete global/island archives, optimizer progress, and queued reseeds before reevaluation. Version-16 baseline files therefore cannot preserve their archives under current version-19 physics. A temporary file is flushed and renamed into place. Dashboard autosaves run in a background thread every ten generations by default; rotation keeps the three newest `seed-*-auto.evo` files and clears stale autosave temporary files. The headless CLI saves to its selected checkpoint path. A save contains current progress, so resume avoids repeating completed evaluations when its semantics are still current.

## Validation and build profiles

`cargo fmt --all --check`, `cargo clippy --locked --all-targets -- -D warnings`, and `cargo test --locked --release` are the CPU CI gates. `.github/workflows/ci.yml` runs them on Ubuntu with eight build jobs, serial test execution, and low process priority. It requests up to eight global Rayon threads and six CPU evaluation threads; the game divides its shared CPU budget between the two pools. GPU tests remain ignored and run locally with `cargo test --release --test simulation -- --ignored` and the workstation environment above.

The `release-fast` Cargo profile inherits release optimization but sets `lto = false`, `codegen-units = 256`, and `incremental = true`. It is intended for iteration, adds no linker requirement, and leaves the thin-LTO release profile unchanged. Use consistent profiles and physics settings when comparing measured performance. Current and historical measurements are recorded separately in [validation](validation.md) and the [performance log](performance-log.md).
