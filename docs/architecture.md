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
| `engine`, `scheduler` | Device threads, replays, GPU failure and out-of-memory recovery, work units, confirmation trials first |
| `ring` | The blocks in flight between the experiment and the engines, absorbed in ring order |
| `environment` | Environment effects, presets and the autochange ladder |
| `storage` | The `Experiment`: the ring of blocks, islands, emitters, breeding, migration, record confirmations, catastrophes, history, saves |
| `worker` | The background evolution thread and the snapshot the UI draws |
| `ui`, `dev_pause` | The egui dashboard and playback. The developer pause used by measurement tools |

## Scoring and replays

The GPU scores every creature. On NVIDIA, when the driver and NVRTC load, `shaders/warp_creature.cu` runs one creature per group of 8, 16 or 32 lanes of a warp, by its nodes and muscles (`src/warp_kernel.rs`). A unit runs as waves of up to 262,144 creatures, one launch each, and a lane group runs its creature to the end of its trial and takes the next from the wave, so there are no trial segments. Kernels compile per lane class, world (the effects that are on), rate and recording. Elsewhere `shaders/physics2_creature.wgsl` (Vulkan) runs one creature per lane, bucketed by node capacity. The GPU result and the GPU confirmation trial are final: no CPU run validates, caps or moves a GPU score. The CPU engine scores CPU-only games and takes over units of a GPU that fails.

`engine::replay` sends the creature to the primary GPU's engine thread. It runs the scoring kernel with a frame output (node positions, muscle energy and force, contact forces) on a slot and queue of its own, and returns the frames and the result of that same run. Without a GPU, or when the GPU does not answer in time, the replay runs on the CPU engine.

## Evaluation flow

The creatures in flight form a ring of 4 blocks, at most 786,432 creatures (`storage::RING_SLOTS`), whatever the generation size. Ring slot `i` fixes a creature's island and random stream. Each block has its own gene arenas and goes to an engine as one unit without a copy. Engines finish blocks in any order, but the ring (`ring.rs`) absorbs them strictly in ring order: a block is decided against the archives as they stand, gets the confirmation trials that decision asks for, is offered to the archives, and is bred again from them and queued at the back. A block keeps the trial settings it was bred with, so a fixed seed repeats whatever the timing. A generation is a count of 3M evaluations: history rows, autochange, nursery graduation and hub migration happen when the count passes the generation size. The worker absorbs one block per pass and handles UI commands between passes; no command waits for the engines.

A standard trial stops at 5 s when the creature is below the bar, the 5 s distance the top 20% reached. A screened creature enters no archive. Every other standard result is final, with one exception: a creature that would set a new record of its island or nursery gets one confirmation trial from the same pose at the fine fidelity (twice the rate and solver passes), and its fitness is the lower distance (`Experiment::verdict`, `scheduler::confirm_config`). The record-setters of an archive are taken fastest first, each against the record the ones before it set, so no unconfirmed score becomes a record. A block asks for its confirmations as soon as its standard results are in, so they run while the blocks before it are absorbed.

A failed GPU is retired and its unfinished units, including pending confirmations, run on the CPU engine with the same creatures and settings. A GPU out of memory keeps the unit, frees idle buffers and retries with fewer units in flight.

## Archives and breeding

Each behavior archive has `6 x 8 x 1 x 6 x 5 = 1,440` niches: ground contact, gait cadence, vertical oscillation (one bin), logarithmic body height and lifted feet. A niche keeps its fastest eligible creature. New body plans are protected against a different topology for three generations.

There are five island archives and a global archive. Ring slot `i` breeds for island `qd::island_of_slot(i, 5)`. Islands 1 to 4 are isolated: parents, mates, limb donors, reserve parents and CMA emitters all come from their own archive. Island 5 is the hub. Every 25 generations it receives copies of the fastest 10% of each isolated island's elites, and nothing flows back. The global archive collects every island's elites for display and saves, and no parent comes from it. Each island keeps a 64-entry morphology reserve of new body plans. It gets 10% of the island's structural-emitter trials and does not count toward coverage or QD score.

Emitter shares start at 35% CMA, 35% structural and 30% novelty, and immigrants seed empty archives. Each island also keeps a nursery: 10% of its slots hold new random bodies and the bodies bred from them, which compete only against each other for 10 generations. Then the survivors enter the island archive and compete on distance alone. Half the CMA parents come from the fastest 1% of their island. Each island also runs a separable CMA-ES on its fastest body plan and rotates to another fast design after 30 generations without an island record.

Genomes live in contiguous arenas with per-creature offsets, one set of arenas per block. Breeding a block writes its children in batches straight into new arenas, and the old block's arenas are freed when the last unit holding them returns, so nothing needs compaction.

## Saves and catastrophes

A save (magic header, then a compressed payload) holds the configuration, generation, history, archives, CMA and emitter state, lineage and the queued elites. It holds no ring. Loading starts the saved generation again with a ring bred from the archives. The header carries the physics version, so an older save is turned down before it loads. Autosave is off. Manual saves go through a temporary file that is flushed and renamed.

Meteor strike removes each elite with probability one half from every archive. Extinction clears the island with the slowest best elite. Both keep the removed entries as fossils in memory, and Undo returns them to empty cells or cells with a slower elite. Fossils are not saved.

A world change applies at once. Blocks already run or running in the old world enter no archive, blocks not yet on an engine run in the new world, and each island's elites are queued to be tested again under the new world in that island's own slots.

## Threads

Evaluation and Rayon share a budget of half the logical CPUs, capped at eight. By default the GPU evaluates and all eight go to general workers (archive insertion, breeding, packing). CPU engines stand by for GPU failure and never score while a GPU is healthy. `EVOLUTION_DEVICES=primary` keeps evaluation off the desktop Radeon. The UI has its own render device and targets 60 FPS.

## Checks

CI runs `cargo fmt --all --check`, `cargo clippy --locked --all-targets -- -D warnings` and the release CPU tests on Ubuntu with the portable SIMD fallback. GPU tests are ignored and run on the workstation.
