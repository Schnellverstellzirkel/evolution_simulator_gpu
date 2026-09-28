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
(`src/cuda_engine.rs`, `shaders/physics_creature.cu`). On the RTX 4060 it
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
