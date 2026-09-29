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
`cargo test --release --test simulation -- --ignored`.

## CUDA on NVIDIA GPUs

On an NVIDIA GPU the game evaluates creatures through CUDA
(`src/cuda_engine.rs`, `shaders/physics2_creature.cu`). On the RTX 4060 it
runs an evolved population about 1.8 times as fast as Vulkan and a fresh game
at the same speed (`docs/design-decisions.md`).
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

## Pausing the game for a measurement

A speed measurement needs the GPU and the CPU to itself, while the owner's game may be running. `tools/pause-game.sh <command...>` pauses the running game, runs the command and lets the game resume when the command ends, also on Ctrl-C or an error. Run speed measurements inside the exclusive GPU lock:

```
flock -x target/gpu.lock tools/pause-game.sh tools/cpu-slot.sh <bench>
```

The tool writes the request file `pause` in `$XDG_RUNTIME_DIR/evolution-simulator` (or `/tmp/evolution-simulator-<uid>` without `XDG_RUNTIME_DIR`). The game looks for it four times a second. It stops handing new work to its engines, lets the units already on them finish and be absorbed, closes its GPU engines so their memory is freed, and writes `paused` with its pid. The tool waits up to 60 s for that file and then runs the command. With no game running it runs the command at once. When the command ends the tool removes the request, the game opens its engines again and goes on. It prints how long the game was paused.

A pause lasts at most 5 minutes from when it began, even if the request stays. After a pause the game runs at least 2 minutes before it honors a new request (it writes `waiting` with the time it will), and it never honors the same request twice. The tool warns when a command ran past the 5 minutes, because the game then resumed partway through. Split such a measurement.

While paused the game window shows "Paused for a developer measurement, resumes in m:ss" and a Resume now button. The replay keeps playing. A new replay records on the CPU. Save, open and new game wait until the pause ends. The pause does not change the search: work is held back, never dropped, so a paused run of a fixed seed matches an undisturbed one (`tests/dev_pause.rs`). The code is in `src/dev_pause.rs` and `src/scheduler/suspend.rs`.

