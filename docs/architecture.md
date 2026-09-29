# Architecture

## Modules

| Module | Responsibility |
| --- | --- |
| `config` | Validated, serializable experiment settings. The game runs 3M creatures and 20 s trials |
| `evolution`, `evolution/anatomy` | Arena-packed genomes, deterministic creation and breeding, body repair, mutation operators |
| `qd` | Behavior niches, elite archives, emitters, diagonal CMA state, the morphology reserve, `qd::VERSION` |
| `physics` | Limits, node masses, joint constants, the ground functions, screening |
| `physics2` | The scalar reference of the physics, the kernel source builders, the packing and the recording edits for replays |
| `cpu_v2`, `cpu_engine`, `simd` | The fast CPU engine (16 creatures per SIMD group), and its front for evaluation, replays and trajectories |
| `creature_kernel` | GPU data layout (`LaneBatch`, `Params`, `GpuResult`) and the CUDA source builders |
| `vk_engine`, `cuda_engine`, `gpu` | The Vulkan and CUDA backends and the evaluation front end |
| `engine`, `scheduler` | Device threads, replays, GPU failure and out-of-memory recovery, work units, contender checks |
| `environment` | Environment effects, presets and the seasons rotation |
| `storage` | The `Experiment`: islands, emitters, breeding, migration, contender checks, catastrophes, history, saves |
| `worker` | The background evolution thread and the snapshot the UI draws |
| `ui`, `dev_pause` | The egui dashboard and playback. The developer pause used by measurement tools |

## Scoring and replays

The GPU scores every creature. `shaders/physics2_creature.wgsl` (Vulkan) and its CUDA mirror `shaders/physics2_creature.cu` run one creature per lane, bucketed by node capacity (3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48 and 64). CUDA is used on NVIDIA when the driver and NVRTC load. The GPU result and the GPU contender check are final: no CPU run validates, caps or moves a GPU score. The CPU engine scores CPU-only games and takes over units of a GPU that fails.

`engine::replay` sends the creature to the primary GPU's engine thread. It runs the scoring kernel with a frame output (node positions, muscle energy and force, contact forces) on a slot and queue of its own, and returns the frames and the result of that same run. Without a GPU, or when the GPU does not answer in time, the replay runs on the CPU engine.

## Evaluation flow

A generation moves through `Ready`, `Evaluating`, `Evaluated`, `Archived` and back. The worker handles UI commands and evaluation results separately. The scheduler gives each healthy device bounded work units of about 1 s and sizes them from measured rates. Each device runs one contender-check unit at a time. Results arrive out of order, and the worker absorbs them in a fixed order so a fixed seed repeats.

A standard trial stops at 5 s when the creature is below the bar, the 5 s distance the top 20% reached. A screened creature enters no archive. Every creature that could enter an archive or a reserve gets a check trial from a perturbed pose at four times the rate and solver passes, and its fitness is the lower distance. Screening and checks are in the kernel and the scheduler (`physics::screen_seconds`, `scheduler::check_verdict`).

A failed GPU is retired and its unfinished units, including pending checks, run on the CPU engine with the same creatures and settings. A GPU out of memory keeps the unit, frees idle buffers and retries with fewer units in flight.

## Archives and breeding

Each behavior archive has `6 x 8 x 1 x 6 x 5 = 1,440` niches: ground contact, gait cadence, vertical oscillation (one bin), logarithmic body height and lifted feet. A niche keeps its fastest eligible creature. New body plans are protected against a different topology for three generations.

There are five island archives and a global archive. Population slot `i` breeds for island `qd::island_of_slot(i, 5)`. Islands 1 to 4 are isolated: parents, mates, limb donors, reserve parents and CMA emitters all come from their own archive. Island 5 is the hub. Every 25 generations it receives copies of the fastest 10% of each isolated island's elites, and nothing flows back. The global archive collects every island's elites for display and saves, and no parent comes from it. Each island keeps a 64-entry morphology reserve of new body plans. It gets 10% of the island's structural-emitter trials and does not count toward coverage or QD score.

Emitter shares start at 35% CMA, 35% structural and 30% novelty, and immigrants seed empty archives. Half the CMA parents come from the fastest 1% of their island. Each island also runs a separable CMA-ES on its fastest body plan and rotates to another fast design after 30 generations without an island record.

Genomes live in contiguous arenas with per-creature offsets. Breeding writes children in batches straight into the arenas, and the generation boundary compacts them into a spare and swaps.

## Saves and catastrophes

A save (magic header, then a compressed payload) holds the configuration, generation, history, archives, CMA and emitter state, lineage and the queued elites. It holds no population. Loading breeds the next generation from the archives. The header carries the physics version, so an older save is turned down before it loads. Autosave is off. Manual saves go through a temporary file that is flushed and renamed.

Meteor strike removes each elite with probability one half from every archive. Extinction clears the island with the slowest best elite. Both keep the removed entries as fossils in memory, and Undo returns them to empty cells or cells with a slower elite. Fossils are not saved.

A world change collects pending work, invalidates old evaluations and queues each island's elites to be tested again under the new world in that island's own slots.

## Threads

Evaluation and Rayon share a budget of half the logical CPUs, capped at eight. By default the GPU evaluates and all eight go to general workers (archive insertion, breeding, packing). CPU engines stand by for GPU failure and never score while a GPU is healthy. `EVOLUTION_DEVICES=primary` keeps evaluation off the desktop Radeon. The UI has its own render device and targets 60 FPS.

## Checks

CI runs `cargo fmt --all --check`, `cargo clippy --locked --all-targets -- -D warnings` and the release CPU tests on Ubuntu with the portable SIMD fallback. GPU tests are ignored and run on the workstation.
