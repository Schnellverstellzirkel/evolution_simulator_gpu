# Agent guide

Read this before you change the code. It covers the game, the owner's rules, how work is organized, and how to build and measure on this machine. Open work is listed in `docs/backlog.md`.

## The game

Evolution Simulator is a Rust game. 2D creatures made of bones, joints and pull-only muscles evolve to travel as far as possible in 20 s trials. Each generation has 3 million creatures, and the GPU scores all of them. The search is MAP-Elites with four emitters (CMA tuning, anatomy mutations, novelty, immigrants) over 4 isolated island archives (each with a nursery that gives new random bodies 10 generations to develop before they compete in the island archive), one hub island that receives copies of their best elites, and one global archive that records everything and is never a parent source. The player watches evolution and changes the world with environment buttons.

`cargo run --release` on `main` is the current game. It must be the best game with no flags.

## Owner's rules

Product:

- Fitness is horizontal distance only. Never add fitness terms, penalties or multipliers, and ask the owner before proposing one. Pressure on behavior comes from physics or environment effects.
- Keep settings few. Trials last 20 s, a generation has 3M creatures, and there are no mutation controls. Environment effects are buttons. The game never changes the world by itself. It may suggest an effect.
- No knobs. Finished work is on by default. A losing experiment is deleted, code and switch. Unfinished work stays on its branch. An environment variable may exist only as a developer diagnostic that a player never needs.
- Never remove a shipped feature or mutation operator unless the owner asks. The owner wants more mutation operator types, never fewer.
- Early screening: a standard trial stops at 5 s when the creature is below the bar, which is the 5 s distance the top 10% reached. A screened creature enters no archive. Replays and elite re-tests run full trials.
- The GPU score is final. A replay comes from the scoring kernel, so it matches its score.
- The search is deterministic for a fixed seed on one GPU.
- Saves are small (archives and search state), and the game writes as few files as possible. Autosave is off. Breaking old saves is fine: bump `qd::VERSION` when archive or physics semantics change, and the save header turns older saves down with a message.
- Speed matters. The goal is 500k evaluated creatures per second, sustained, in the graphical game at 60 FPS.
- The current game is the reference, not the past. A change stays if the game is better or faster now, shown by a direct measurement of that change on its own. Any measured speedup is kept and merged, however small: 1.05x is a win. A track's gate (3x, 2x, 45% issue) is its ambition and decides what to try next, never whether measured gains are thrown away. No comparison to an earlier version is needed, and nobody writes experiment reports. Search changes rest on papers and practice and must not break the search.
- No golden reference. The physics, the muscle model and the kernel that exist today are vibecoded and are not a reference: any of them may be replaced by a cheaper one. The only physics requirement is the spirit of the game: creatures evolve interesting, efficient shapes and gaits, and no glitchy movers (random bodies that travel, feet that slide, energy from nowhere). Bit-equality between two implementations or kernel variants never matters. The one determinism rule is that one build on one GPU gives one search per seed.
- There is one physics (`docs/physics.md`): the CUDA kernel, `shaders/warp_creature.cu`. Nothing else simulates creatures. The game needs an NVIDIA GPU with the CUDA driver and NVRTC.
- Posture rules (for example what counts as a fall) need the owner's approval.

Working:

- Commit small and push often, about every 20 minutes of work. Plain commit messages say what changed, why, and what was measured.
- Write plainly in commits, docs and reports.

## How work is organized

One orchestrator session plans, writes briefs, reviews and merges. Worker agents each take one track in their own worktree and branch:

```
git -C /home/amipo/workspace/evolutionSimulator fetch
git -C /home/amipo/workspace/evolutionSimulator worktree add /home/amipo/workspace/evolutionSimulator-<track> -b claude/<track> origin/main
mkdir -p /home/amipo/workspace/evolutionSimulator-<track>/target
cp -a --reflink=auto /home/amipo/workspace/evolutionSimulator/target/release-fast /home/amipo/workspace/evolutionSimulator-<track>/target/
```

Use absolute paths, because `git -C` resolves a relative worktree path against the repository. Copying `target/release-fast` (or `target/release`) seeds the build cache so dependencies do not rebuild. Never build in the main directory, because the owner's game runs from its `target/release`. Before you edit, check `git status` for changes you did not make. Rebase onto `origin/main` before each push, push your branch, and tell the orchestrator the hash. Only the orchestrator merges into `main`.

## This machine

- 16 threads, an RTX 4060 laptop GPU for compute, and a Radeon 780M that drives the desktop. Never evaluate creatures on the Radeon, because heavy Radeon use once crashed the desktop. Set `EVOLUTION_DEVICES=primary` for every run of the game and the tools.
- Builds, tests and runs may use the whole CPU. The game's general Rayon pool (breeding, archive insertion, packing) takes every logical CPU but two, at nice 10.
- GPU runs share the GPU through `/home/amipo/workspace/evolutionSimulator/target/gpu.lock`. Smoke windows and evolutions that do not measure speed take it shared (`flock -s`), so they run side by side. Speed measurements take it exclusive (`flock -x`), so they run alone. Run at most 2 GPU processes per agent at once, because the owner's game holds 5 to 7.5 GB of the RTX 4060's 8 GB and CUDA fails to open when memory runs out. Since 2026-09-29 the owner allows agent GPU work while their game runs: keep GPU memory low, and retry smaller after an out-of-memory error. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` shows who is on the GPU.
- Speed measurements pause the owner's game while they run: `flock -x target/gpu.lock tools/pause-game.sh <bench>`. The game stops evaluating and breeding and closes its GPU engines, so the GPU memory is free, and it resumes when the command ends. A pause lasts at most 5 minutes, so a measurement must fit in 5 minutes or be split. After a pause the game runs at least 2 minutes before it pauses again. `docs/building.md` has the details.
- Screenshot runs (`EVOLUTION_SMOKE_*`, see `src/ui.rs`) open real windows on the owner's desktop titled "agent screenshot run, not your game". Keep them to seconds and look at every screenshot yourself.

## Build

- Iteration build: `cargo build --profile release-fast` (LTO off, 256 codegen units, incremental). Use the normal release profile (thin LTO) for performance measurements.
- `docs/building.md` has the build details and the environment variables.

## Code map

The physics:

- `shaders/warp_creature.cu` is the CUDA kernel, the only physics, driven by `src/cuda_engine.rs`. A creature runs on a group of 4, 8, 16 or 32 lanes (lane i owns node i and the bone ending there, state in registers, tree passes level by level), and each 1/60 s step is two substeps (`docs/physics.md`). `src/warp_kernel.rs` packs creatures for it from `physics2::Model` and writes its source with the world's effects compiled in.
- `src/physics2.rs` holds the physics constants and `Model`, a creature's constants and starting state, from which `warp_kernel::pack` fills every kernel record.
- `src/physics.rs` holds what the physics and the UI share: limits, fidelity, node and joint constants, the ground functions (bumps, slope, gaps, hurdles, quake), screening.
- `src/engine.rs` runs each GPU on its own thread, one whole-trial submission per unit, and records replays with the scoring kernel (frames carry the muscle energy, muscle force and contact forces). `src/gpu.rs` is the evaluation front end. `src/creature_kernel.rs` holds `GpuResult`, `LaneBatch` and `frame_stride`.
- `docs/physics.md` describes the model: contacts (the deepest 4 per step), friction that may never do positive work, the plant pass, and muscle strength scaled to the mass a muscle moves.

Search and game state:

- `src/qd.rs`: archives, niches, behavior descriptors, the morphology reserve, `qd::VERSION`.
- `src/rungs.rs`: the audit lane and the early rungs (R1 at 1 s, R2 at 2.5 s), their fit at the generation boundary, the breaker per cadence band. The rule runs in the metrics block of `shaders/warp_creature.cu`.
- `src/storage.rs`: the `Experiment` with its ring of blocks, islands, emitters, breeding, migration, record confirmations, and saves.
- `src/ring.rs`: the blocks in flight, absorbed in ring order whatever order the GPU finishes them in.
- `src/evolution.rs` and `src/evolution/anatomy/`: the genome and the mutation operators (see `docs/anatomy-operators.md`).
- `src/scheduler.rs`: routes work to the GPUs and reopens a GPU that fails. A GPU that does not reopen stops evolution.
- `src/worker.rs`: the worker thread and the snapshot the UI draws.
- `src/environment.rs`: environment effects and presets. Add new effects here.
- `src/ui.rs`: the egui interface. `src/theme.rs`: its palette, style and HUD pieces. `src/world_fx.rs`: the replay backdrop and the look of each environment effect. `src/assets.rs`: the art under `assets/ui/`, which `tools/ui_assets.py` builds from CC0 sources (`assets/ui/CREDITS.md`). `src/config.rs`: settings.

## Measurement tools

All example tools score and replay creatures on the GPU engine (`examples/common/mod.rs`), take the GPU lock shared and need `EVOLUTION_DEVICES=primary`. If the primary GPU does not open they fail.

- `examples/search_ab.rs`: fixed-seed search runs through the production ring on the GPU, with the early screen and the record confirmations.
- `EVOLUTION_STAGE_LOG=<path>`: one CSV row per generation (evaluation, archive and breeding seconds, end-to-end rate).
- `examples/size_report.rs <save> [count]`: body length, mass and foot slip for the best elites, from GPU replays. It has no cost of transport column.
- `examples/mutation_audit.rs`: how much of its parent's distance each operator's child keeps.
- `examples/physics_audit.rs`: per elite, what the GPU replay records: contact-free steps, largest ground push, lowest muscle energy store, steps with a broken joint. The kernel keeps no solver energy, momentum or friction ledgers.
- `examples/first_generation.rs`: random-population distances on the GPU engine.
- `examples/replay_match.rs <save>`: the best elites' archive distance beside their replay's.
- `examples/p2_speed.rs`, `examples/worker_rate.rs`: GPU and worker throughput.
- `tools/pause-game.sh`: pauses the owner's game for a speed measurement (`docs/building.md`).

## Docs

- `README.md`: for players.
- `docs/architecture.md`, `docs/physics.md`, `docs/building.md`, `docs/anatomy-operators.md`.
- `docs/design-decisions.md`: what the game does in search, physics and speed, and the number behind each choice.
- `docs/rejected-ideas.md`: real dead ends with their numbers. Add one or two lines only when someone would retry the idea.
- `docs/backlog.md`: open work.

## Measured and rejected

The full list is in `docs/rejected-ideas.md`. Do not redo these without a new reason:

- 30 Hz physics: the search gains were integrator exploits. 60 Hz stays.
- Muscle mass (5 variants): none held muscle counts down, and best distance fell by half.
- The first persistent-lane engine (`claude/lanes`): slower, because of register pressure.
- A muscle waveform cache and a branch-free waveform on the GPU: both slower.
- A second screening rung, cheaper fine checks, and behavior metrics every 2 or 4 steps: removed.
