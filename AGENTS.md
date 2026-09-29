# Agent guide

Read this before you change the code. It covers the game, the owner's rules, how work is organized, and how to build, test and measure on this machine. Open work is listed in `docs/backlog.md`.

## The game

Evolution Simulator is a Rust game. 2D creatures made of bones, joints and pull-only muscles evolve to travel as far as possible in 20 s trials. Each generation has 3 million creatures, and the GPU scores all of them. The search is MAP-Elites with four emitters (CMA tuning, anatomy mutations, novelty, immigrants) over 4 island archives in a ring plus one global archive. The player watches evolution and changes the world with environment buttons.

`cargo run --release` on `main` is the current game. It must be the best game with no flags.

## Owner's rules

Product:

- Fitness is horizontal distance only. Never add fitness terms, penalties or multipliers, and ask the owner before proposing one. Pressure on behavior comes from physics or environment effects.
- Keep settings few. Trials last 20 s, a generation has 3M creatures, and there are no mutation controls. Environment effects are buttons. The game never changes the world by itself. It may suggest an effect.
- No knobs. Finished work is on by default. A losing experiment is deleted, code and switch. Unfinished work stays on its branch. An environment variable may exist only as a developer diagnostic that a player never needs.
- Never remove a shipped feature or mutation operator unless the owner asks. The owner wants more mutation operator types, never fewer.
- Early screening: a standard trial stops at 5 s when the creature is below the bar, which is the 5 s distance the top 20% reached. A screened creature enters no archive. Replays and elite re-tests run full trials.
- The GPU score is final. CPU and GPU agreement is a diagnostic, not a gate. A replay comes from the scoring kernel, so it matches its score.
- The search is deterministic for a fixed seed on one GPU.
- Saves are small (archives and search state), and the game writes as few files as possible. Autosave is off. Breaking old saves is fine: bump `qd::VERSION` when archive or physics semantics change, and the save header turns older saves down with a message.
- Speed matters. The goal is 2M, then 4M, evaluated creatures per second in the graphical game at 60 FPS.
- Search changes rest on evidence (papers and practice) and must not break the search. Test each one with a paired A/B at equal evaluation budgets over 3 seeds, and report best distance, QD score and the top-50 body mix.
- Physics v2 is the game (since 2026-09-29). New physics features go into v2. A physics change updates every GPU kernel, WGSL and CUDA together. CPU ports may differ.
- Posture rules (for example what counts as a fall) need the owner's approval.

Working:

- Commit small and push often, about every 20 minutes of work. Plain commit messages say what changed, why, and what was measured.
- Run the full test suite once per push, not after every edit. Stop an A/B once 3 seeds decide it.
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

- 16 threads, an RTX 4060 laptop GPU for compute, and a Radeon 780M that drives the desktop. Never evaluate creatures on the Radeon, because heavy Radeon use once crashed the desktop. Set `EVOLUTION_DEVICES=primary` for every run of the game, the tests and benchmarks.
- Agents share at most half the machine. Build with `nice -n 19` and `CARGO_BUILD_JOBS=2`. Run long CPU jobs (the full test suite, A/B runs, benchmarks) through `tools/cpu-slot.sh <command>`, which waits for one of two shared 4-thread slots. The game itself may use the whole machine.
- GPU runs share the GPU through `/home/amipo/workspace/evolutionSimulator/target/gpu.lock`. Tests, smoke windows and evolutions that do not measure speed take it shared (`flock -s`), so they run side by side. Speed measurements take it exclusive (`flock -x`), so they run alone. Run at most 2 GPU processes per agent at once, because the owner's game holds 5 to 7.5 GB of the RTX 4060's 8 GB and CUDA fails to open when memory runs out. Since 2026-09-29 the owner allows agent GPU work while their game runs: keep GPU memory low, and retry smaller after an out-of-memory error. `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` shows who is on the GPU.
- Screenshot runs (`EVOLUTION_SMOKE_*`, see `src/ui.rs`) open real windows on the owner's desktop titled "agent screenshot run, not your game". Keep them to seconds and look at every screenshot yourself.

## Build, test, run

- Iteration build: `cargo build --profile release-fast` (LTO off, 256 codegen units, incremental). Use the normal release profile (thin LTO) for performance measurements.
- Before committing: `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test --release`.
- GPU tests are `#[ignore]`d. Run them with `cargo test --release --test simulation -- --ignored`, plus `tests/gpu_repeatability.rs` and the ignored tests in `tests/screening.rs`.
- Run `examples/first_generation` before pushing any physics change. It reports random-population distances (median, p99, best) and catches free propulsion.
- `docs/building.md` has the build details and the list of environment variables.

## Code map

Physics v2 is the game's physics:

- `shaders/physics2_creature.wgsl` is the GPU kernel for Vulkan, driven by `src/vk_engine.rs` and `src/gpu.rs`.
- `shaders/physics2_creature.cu` is its CUDA mirror, used on NVIDIA when the driver and NVRTC load, driven by `src/cuda_engine.rs`. It mirrors the WGSL section by section; change both together.
- `src/physics2.rs` is the scalar CPU prototype: the reference the GPU kernels and the fast CPU engine are tested against, the engine of single CPU replays, and the home of the kernel source builders, the packing (`pack`) and the recording edits for replays.
- `src/cpu_v2.rs` is the fast CPU engine (16 creatures with one skeleton per SIMD group, `src/simd.rs`), bit-equal to the prototype (`tests/physics2_lanes.rs`). It scores CPU-only games and GPU failover. `src/cpu_engine.rs` is the thin front (`evaluate`, `replay`, `trajectory`, `transport_cost`).
- `src/physics.rs` holds what the physics and the UI share: limits, fidelity, node and joint constants, the ground functions (bumps, slope, gaps, hurdles, quake), screening.
- `src/engine.rs` picks the backend and records replays with the scoring kernel (frames carry the muscle energy, muscle force and contact forces). `src/creature_kernel.rs` holds the GPU data layout (`LaneBatch`, `Params`, `GpuResult`, `frame_stride`) and builds the CUDA sources.
- `docs/physics-v2.md` describes the model: contacts (the deepest 4 per step), friction that may never do positive work, the plant pass, and muscle strength scaled to the mass a muscle moves.

Search and game state:

- `src/qd.rs`: archives, niches, behavior descriptors, the morphology reserve, `qd::VERSION`.
- `src/storage.rs`: the `Experiment` with its islands, emitters, breeding, migration, fine checks, and saves.
- `src/evolution.rs` and `src/evolution/anatomy/`: the genome and the mutation operators (see `docs/anatomy-operators.md`).
- `src/scheduler.rs`: routes work to healthy GPUs, with the CPU as failover.
- `src/worker.rs`: the worker thread and the snapshot the UI draws.
- `src/environment.rs`: environment effects and presets. Add new effects here.
- `src/ui.rs`: the egui interface. `src/config.rs`: settings.

## Measurement tools

- `examples/search_ab.rs`: paired fixed-seed search runs through the production archive and breeding path, on the CPU or with `--gpu` through the game's GPU path. At 100k creatures a 6-seed mean detects only about 30% in QD; use 12 or more seeds for smaller effects (`docs/search-research.md` section 15).
- `EVOLUTION_STAGE_LOG=<path>`: one CSV row per generation (evaluation, archive and breeding seconds, end-to-end rate).
- `examples/size_report.rs <save> [count]`: body length, mass and foot slip for the best elites. `EVOLUTION_LEDGER=1` adds where forward momentum comes from.
- `examples/mutation_audit.rs`: how much of its parent's distance each operator's child keeps.
- `examples/physics_audit.rs`: energy, friction and momentum ledgers per elite under v2. `tests/physics_audit.rs` guards against solver-made energy and friction exploits.
- `examples/effect_cost.rs`, `examples/mem_report.rs`, `examples/body_stats.rs`, `examples/momentum_ledger.rs`.

## Docs

- `README.md`: for players.
- `docs/architecture.md`, `docs/building.md`, `docs/validation.md`.
- `docs/search-research.md`: every search experiment and its numbers. Add yours.
- `docs/performance-log.md`: every performance measurement, including rejected ideas. Add yours.
- `docs/research-2026-09-29.md`: literature survey with ranked ideas for search quality and throughput.
- `docs/hpc-assessment.md`, `docs/data-architecture.md`, `docs/phase0-measurements.md`: GPU limits and the v2 data design.
- `docs/anatomy-operators.md`, `docs/ux-audit.md`.
- `docs/backlog.md`: open work.

## Measured and rejected

Do not redo these without a new reason. The numbers are in `docs/performance-log.md` and `docs/search-research.md`.

- 30 Hz physics: the search gains were integrator exploits. 60 Hz stays.
- Muscle mass (5 variants): none held muscle counts down, and best distance fell by half.
- The first persistent-lane engine (`claude/lanes`): slower, because of register pressure.
- A muscle waveform cache and a branch-free waveform on the GPU: both slower.
- A second screening rung, cheaper fine checks, and behavior metrics every 2 or 4 steps: removed.
