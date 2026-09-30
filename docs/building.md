# Building

The normal `release` profile keeps thin LTO and 16 codegen units. For quicker
iteration, `release-fast` inherits the release settings but disables LTO, uses 256
codegen units, and enables incremental compilation:

```bash
CARGO_BUILD_JOBS=8 nice -n 10 cargo build --profile release-fast
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 EVOLUTION_DEVICES=primary nice -n 10 cargo run --profile release-fast
```

The repository's x86_64 Linux Cargo config already enables `target-cpu=native`.
If `mold` is installed, it can speed up linking:

```bash
CARGO_BUILD_JOBS=8 nice -n 10 env RUSTFLAGS="-C target-cpu=native -C link-arg=-fuse-ld=mold" cargo build --profile release-fast
```

Supplying `RUSTFLAGS` replaces Cargo's configured rustflags, so the command
repeats `target-cpu=native` explicitly. Omit it when `mold` is unavailable.

On this machine keep evaluation off the desktop Radeon and cap CPU use. Apply
these settings to every game run, test and benchmark:

```bash
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 EVOLUTION_DEVICES=primary nice -n 10 cargo test --release
```

GPU tests are ignored. Select them explicitly, for example
`cargo test --release --test gpu_repeatability -- --ignored`.

## CUDA on NVIDIA GPUs

On an NVIDIA GPU the game evaluates creatures through CUDA
(`src/cuda_engine.rs`, `shaders/warp_creature.cu`, `src/warp_kernel.rs`).
`examples/warp_regs.rs` prints each kernel's registers and spills from NVRTC
alone; `EVOLUTION_WARP_PROFILE=1` makes one warp print its cycles per kernel
section. `EVOLUTION_WARP_SUBSTEPS`, `EVOLUTION_WARP_PGS_SWEEPS`,
`EVOLUTION_WARP_CLEAN_SWEEPS`, `EVOLUTION_WARP_PLANT_ROUNDS` and
`EVOLUTION_WARP_PLANT_SWEEPS` override the solver settings for measuring
them.
Nothing needs configuring: the build links no CUDA library, and the engine
loads the CUDA driver library and NVRTC when it opens. If either is missing,
or the GPU is not an NVIDIA GPU, the game prints one line and runs on Vulkan.

The driver library (`libcuda.so.1`) comes with the NVIDIA driver. NVRTC comes
with a CUDA toolkit, or, without root, from NVIDIA's pip wheel in a virtual
environment the engine looks for:

```bash
python3 -m venv ~/.local/share/evolution-cuda/venv
~/.local/share/evolution-cuda/venv/bin/pip install "nvidia-cuda-nvrtc==13.0.88"
```

Pick an NVRTC no newer than the driver's CUDA version (`nvidia-smi` shows it;
driver 580 is CUDA 13.0). The kernels compile when the engine opens (the fine
check kernels on first use), about 2 s per kernel the first time. NVIDIA's
compute cache (`~/.nv/ComputeCache`) keeps them, so later starts take about
0.3 s.

Developer diagnostics, never needed to play: `EVOLUTION_CUDA=0` runs on
Vulkan, `EVOLUTION_CUDA=1` makes the GPU tests refuse a Vulkan fallback,
`EVOLUTION_NVRTC=/path/to/libnvrtc.so.13` names another NVRTC, and
`EVOLUTION_CUDA_VERBOSE=1` reports compile times.

## Diagnostic examples

The tools in `examples/` (`search_ab`, `size_report`, `mutation_audit`, `physics_audit`, `first_generation`, `replay_match`, `p2_speed`, `worker_rate`) score and replay creatures on the GPU engine and have no CPU mode. They fail if the primary GPU does not open. They submit at most 50,000 creatures per unit, so they need little GPU memory beside the owner's game. Run them with the lock shared, unless they measure speed:

```bash
EVOLUTION_DEVICES=primary flock -s target/gpu.lock nice -n 19 tools/cpu-slot.sh cargo run --release --example first_generation 20000
```

## Pausing the game for a measurement

A speed measurement needs the GPU and the CPU to itself, while the owner's game may be running. `tools/pause-game.sh <command...>` pauses the running game, runs the command and lets the game resume when the command ends, also on Ctrl-C or an error. Run speed measurements inside the exclusive GPU lock:

```
flock -x target/gpu.lock tools/pause-game.sh tools/cpu-slot.sh <bench>
```

The tool writes the request file `pause` in `$XDG_RUNTIME_DIR/evolution-simulator` (or `/tmp/evolution-simulator-<uid>` without `XDG_RUNTIME_DIR`). The game looks for it four times a second. It stops handing new work to its engines, lets the units already on them finish and be absorbed, closes its GPU engines so their memory is freed, and writes `paused` with its pid. The tool waits up to 60 s for that file and then runs the command. With no game running it runs the command at once. When the command ends the tool removes the request, the game opens its engines again and goes on. It prints how long the game was paused.

A pause lasts at most 5 minutes from when it began, even if the request stays. After a pause the game runs at least 2 minutes before it honors a new request (it writes `waiting` with the time it will), and it never honors the same request twice. The tool warns when a command ran past the 5 minutes, because the game then resumed partway through. Split such a measurement.

While paused the game window shows "Paused for a developer measurement, resumes in m:ss" and a Resume now button. The replay keeps playing. A new replay records on the CPU. Save, open and new game wait until the pause ends. The pause does not change the search: work is held back, never dropped, so a paused run of a fixed seed matches an undisturbed one (`tests/dev_pause.rs`). The code is in `src/dev_pause.rs` and `src/scheduler/suspend.rs`.


## Environment variables

`cargo run --release` is the whole game and needs none of these. Every `EVOLUTION_*` variable is a developer diagnostic or a measuring control. Speed and search settings (GPU slots, batch and unit sizes, workgroup sizes, screening, the anatomy operators, joint damping, Hill speed) are fixed in the code and have no switch. Read the code (`grep -rn EVOLUTION_ src`) for the exact list. The groups are:

- Devices and threads: `EVOLUTION_DEVICES` (`primary` on this machine, never the Radeon), `EVOLUTION_CPU_THREADS` (size of the CPU failover pool, 0 turns it off), `EVOLUTION_RENDER_GPU` (adapter for drawing the window), `EVOLUTION_UI_FPS` (frame rate cap, 0 follows vsync).
- CUDA: `EVOLUTION_CUDA`, `EVOLUTION_NVRTC`, `EVOLUTION_CUDA_VERBOSE` (see the CUDA section).
- Measuring: `EVOLUTION_STAGE_LOG=<path>` writes one CSV row per generation. `EVOLUTION_PROFILE_BREED` prints archive and breeding timings.
- Benchmarks, tests and screenshots: `EVOLUTION_BENCH_*` drives the graphical benchmark mode (generations, duration, warm-up). `EVOLUTION_TEST_*` sizes the ignored GPU tests. `EVOLUTION_SMOKE_*` starts short screenshot runs, and their windows show on the desktop.
- Unattended runs: `EVOLUTION_AUTOSTART="Autochange environment=1"` sets the listed effect levels (the list may be empty), turns autosave on every 10 generations and starts evolving continuously.
