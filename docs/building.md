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

For tests, benchmarks, and game runs on this machine, keep evaluation off the
desktop Radeon and cap CPU use. Apply these environment settings to each such
command:

```bash
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 RUST_TEST_THREADS=1 EVOLUTION_DEVICES=primary nice -n 10 cargo test --release --all-targets
```

Serial test execution prevents independently created evaluation pools from
running concurrently. `--all-targets` includes the size-report diagnostic tests;
GPU tests remain ignored unless explicitly selected. Use the same resource
settings for other test or benchmark commands. Normal release
builds and runs remain available with `--release`.

## CUDA on NVIDIA GPUs

On an NVIDIA GPU the game evaluates creatures through CUDA
(`src/cuda_engine.rs`, `shaders/physics2_creature.cu`). On the RTX 4060 it
runs the kernel 1.6 to 1.8 times as fast as Vulkan, the 3M game on an
evolved population 1.37 times as fast, and a fresh game at the same speed
(docs/performance-log.md).
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

Developer overrides, never needed to play: `EVOLUTION_CUDA=0` runs on
Vulkan, `EVOLUTION_CUDA=1` makes the GPU tests refuse a Vulkan fallback,
`EVOLUTION_NVRTC=/path/to/libnvrtc.so.13` names another NVRTC,
`EVOLUTION_CUDA_MAXREG` sets the register cap (default 128, 0 for the
compiler's choice), `EVOLUTION_CUDA_WG` fixes threads per block (32, 64 or
128), `EVOLUTION_CUDA_STREAMS` limits streams per unit,
`EVOLUTION_CUDA_FLAGS` adds NVRTC options, and `EVOLUTION_CUDA_VERBOSE=1`
reports compile times. `examples/cuda_stats.rs` prints registers, spills,
shared memory and resident warps per node capacity; run it with
`CUDA_CACHE_DISABLE=1` to get the spill counts, which a cached compile does
not report.

## Environment variables

`cargo run --release` is the whole game and needs none of these. Every
`EVOLUTION_*` variable that is left is a developer diagnostic or a measuring
control. Experiment switches were either turned on for good or deleted with
their code (2026-09-29).

Devices and threads
- `EVOLUTION_DEVICES`: which GPUs evaluate. `primary` on this machine, never the Radeon. Other names select extra devices.
- `EVOLUTION_CPU_THREADS`: size of the CPU evaluation pool (0 turns it off). Set it only for CPU-only tools.
- `EVOLUTION_RENDER_GPU`: adapter name for drawing the window.
- `EVOLUTION_UI_FPS`: cap on the window frame rate (0 follows vsync).

CUDA (see the CUDA section above)
- `EVOLUTION_CUDA`, `EVOLUTION_NVRTC`, `EVOLUTION_CUDA_MAXREG`, `EVOLUTION_CUDA_WG`, `EVOLUTION_CUDA_STREAMS`, `EVOLUTION_CUDA_FLAGS`, `EVOLUTION_CUDA_VERBOSE`: engine choice and compile controls.

Scheduling and kernel tuning (they change speed, not results)
- `EVOLUTION_GPU_SLOTS`: submission slots per GPU.
- `EVOLUTION_GPU_BATCH`, `EVOLUTION_GPU_CHUNK`: creatures per batch and steps per dispatch.
- `EVOLUTION_UNIT_SECONDS`, `EVOLUTION_CPU_UNIT_SECONDS`, `EVOLUTION_SECONDARY_UNIT_SECONDS`, `EVOLUTION_SECONDARY_CHUNK`: length and size of work units.
- `EVOLUTION_CHECK_UNITS`: check units queued at once.
- `EVOLUTION_LANE_WG`: Vulkan workgroup size.
- `EVOLUTION_SEGMENTS`: pauses at which the GPU repacks fallen creatures out of warps.
- `EVOLUTION_PLAN_BATCH`: minimum run of one body plan that gets its own specialized kernel (0, the default, is off because it measured no gain on mixed units; `eval-bench --plan-rank` uses it).

Measuring controls
- `EVOLUTION_SCREEN`: seconds of the early screen (`0` turns screening off for comparisons).
- `EVOLUTION_ANATOMY`: `0` restores the classic mutation for comparisons.
- `EVOLUTION_EXACT_COS`: exact cosine in the kernels instead of the polynomial, as a control for its rate and accuracy.
- `EVOLUTION_NODE_SLIP`: contact details for the champion in `size_report`.
- `EVOLUTION_PROFILE_BREED`: prints archive and breeding timings.
- `EVOLUTION_STAGE_LOG`: appends one CSV row per generation.
- `EVOLUTION_DUMP_IR`: dumps the compiled kernel IR.
- `EVOLUTION_STATS_PLAN`: `<checkpoint>:<rank>` adds the specialized kernel to `shader_stats`.

Benchmarks, tests and screenshots
- `EVOLUTION_BENCH_DURATION`, `_GENERATIONS`, `_WARMUP`, `_THROUGHPUT`, `_RESPONSIVE`, `_NO_AUTOSAVE`, `_SETTINGS_PROBE`: the GUI benchmark mode.
- `EVOLUTION_TEST_POPULATION`, `EVOLUTION_TEST_CHECKPOINT`: size and start file of the ignored GPU world-change test.
- `EVOLUTION_SMOKE_CAPTURE`, `_CHECKPOINT`, `_TAB`, `_POPULATION`, `_ZOOM`, `_DARK`: short screenshot runs (their windows show on the desktop).

Deleted with their code: `EVOLUTION_NEUTRAL_SPLITS`, `EVOLUTION_ELITE_REFRESH`,
`EVOLUTION_SHRINK`, `EVOLUTION_EARLY_EXIT` (the exit is always on outside
replays), `EVOLUTION_CHECK_TERRAIN`, `EVOLUTION_ROBUST_TRIALS`,
`EVOLUTION_ISLANDS`, `EVOLUTION_PHYSICS_RATE`, `EVOLUTION_BONE_PASSES`,
`EVOLUTION_VELOCITY_PASSES`, `EVOLUTION_STANCE_GRIP`, `EVOLUTION_SCREEN_KEEP`,
and the physics limit overrides (`EVOLUTION_MAX_MUSCLE_SPEED`,
`EVOLUTION_MAX_MUSCLE_FORCE`, `EVOLUTION_MAX_NODE_SPEED`,
`EVOLUTION_MAX_BONE_SPIN`, `EVOLUTION_MAX_BONE_LENGTH`, `EVOLUTION_MAX_STROKE`,
`EVOLUTION_MIN_MUSCLE_PERIOD`, `EVOLUTION_MUSCLE_ENERGY`,
`EVOLUTION_MUSCLE_RECOVERY`, `EVOLUTION_BONE_DENSITY`). Physics constants now
live in `physics::Limits::DEFAULT` and the functions beside it. The physics v2
branch keeps its own temporary `EVOLUTION_P2_*` switches.
