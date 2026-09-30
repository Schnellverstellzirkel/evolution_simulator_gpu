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

## GPU failures

Today a GPU engine that fails is dropped, and the scheduler opens it again after 1, 4 and then 10 s (`recover` in `src/scheduler.rs`). Each reopen submits the unfinished units again with their exact inputs, so they give the same results, and the event feed shows every attempt. After three failed reopens the units go to the CPU engine. The 2M/s plan (`docs/plan-2m.md`) deletes the CPU engine, so there will be no failover physics. From then on the third failed reopen pauses the game with the error in the event feed, and the game does not retry forever.

## Machine settings for measurements

These need root, so the owner runs them. `nvidia-smi -lgc <min>,<max>` pins the RTX 4060's SM clock for a measurement and `nvidia-smi -rgc` releases it. The power rows of the plan decide if a pinned clock also helps long runs.

The RTX's PCIe link idles at Gen 1 x8 and trains up to Gen 4 under load. On 2026-09-30 `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max --format=csv` printed 1 and 4 on the idle machine, and the device's runtime PM (`/sys/bus/pci/devices/0000:01:00.0/power/control`) was `auto`. A speed change is a link retrain of 1 to 5 ms, and no DMA moves during it. Waves that last seconds never notice. Blocks of 50 ms may pay it on their first upload, and a replay click or the first block after a world change waits for it. To check, run `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv -lms 10` beside `worker_rate` and count the gen changes per second. If there is more than one per second, keep the link up with two one-time root settings and measure again:

```bash
sudo nvidia-smi -pm 1
echo on | sudo tee /sys/bus/pci/devices/0000:01:00.0/power/control
```

The first turns on persistence mode and the second keeps the device out of runtime suspend. Together they cost 1 to 2 W at idle.

## The Radeon 780M

The Radeon 780M drives the desktop. The game window renders on it (`launch` in `src/ui.rs` picks the compositor's GPU), and that is its only job in the game. It never evaluates, confirms or replays creatures. The GPU score is final and it comes from the RTX 4060. `EVOLUTION_DEVICES=primary` keeps the Radeon out of evaluation.

Any compute on the Radeon shares the compositor's GPU. It also shares the 512 MB VRAM carve-out, which the desktop fills to about 85%, and the APU's 54 W package budget with the CPU. On this kernel (7.0.0-34) the amdgpu lockup timeout is 2,000 ms for every ring. `modinfo amdgpu` gives that default and the `lockup_timeout` parameter is unset. A submission that runs longer is reset. The kernel journal shows two such resets from this game's process, both from the old Vulkan engine sending trial segments to the Radeon. The lines below were read again with `journalctl -k -b all` on 2026-09-30, and the times are CEST:

- 2026-09-25 17:55:29: `ring comp_1.1.1 timeout, signaled seq=55, emitted seq=56`, process `evolution-simul` pid 12241. The queue reset succeeded and the desktop survived.
- 2026-09-26 13:06:36: the same ring, pid 842696, `signaled seq=12980, emitted seq=12982`. The queue reset failed and amdgpu did a MODE2 whole-GPU reset. The journal says "VRAM is lost due to GPU reset" and "device wedged, but recovered through reset". The compositor's buffers were in that VRAM, so the desktop crashed.

Rules for anything that submits work to the Radeon, including tools and diagnostics:

1. No submission may take longer than 50 ms. Size dispatches from a measured rate and halve them when one exceeds 20 ms. No persistent kernels, no in-kernel work loops, no unbounded step counts. 50 ms is 2.5% of the timeout, and the halving stops a dispatch that drifts from ever reaching it.
2. Use a compute-only queue at `VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR`, which needs no privilege. Never raise the game's priority above the compositor's.
3. Allocate only host-visible memory (GTT, or host memory imported through `VK_EXT_external_memory_host`). Never `DEVICE_LOCAL` on the Radeon. Stay under 512 MB.
4. At most two submissions in flight. If a fence wait exceeds 500 ms, stop submitting for the session and fall back to the CPU path.
5. Run Radeon compute in a helper process that shares memory with the game. A `VK_ERROR_DEVICE_LOST` then kills the helper and the game goes on.
6. Vulkan through RADV only. No ROCm, no OpenCL, no second driver stack on the display device.
7. Before shipping anything that uses the Radeon, soak it for 10 minutes with the game window open and the compositor at 60 FPS. It passes if the UI frame time p99 stays under 20 ms, `journalctl -k` has no `ring comp` or `reset` line, and the RTX rate is unchanged.
8. Never evaluate, confirm or replay creatures on the Radeon.

The Mesa vendor trap: glvnd picks NVIDIA by default, so a tool that opens EGL gets the RTX even when it means the Radeon. Force the Mesa vendor for EGL, or select the RADV physical device by name in Vulkan:

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json EGL_PLATFORM=surfaceless <tool>
```

Check the renderer line the tool prints. If it names NVIDIA, stop, because the tool is on the RTX.

Diagnostics: run `journalctl -k | grep -E "amdgpu.*(timeout|reset)"` after any Radeon experiment. The Radeon is `card2` (`0000:06:00.0`) and the RTX is `card1` (`0000:01:00.0`). `/sys/class/drm/card2/device/gpu_busy_percent` is the Radeon's load. In the amdgpu hwmon, `power1_average` is the APU package power in microwatts and `freq1_input` is the shader clock.

The measurements behind this section are in `docs/plan-2m-debate/round-1-igpu.md` to `round-5-igpu.md`. Radeon physics stays closed: the 780M issues 0.2 to 0.25 of the RTX's instructions per clock, and its watts come from the package budget the CPU needs.

## Environment variables

`cargo run --release` is the whole game and needs none of these. Every `EVOLUTION_*` variable is a developer diagnostic or a measuring control. Speed and search settings (GPU slots, batch and unit sizes, workgroup sizes, screening, the anatomy operators, joint damping, Hill speed) are fixed in the code and have no switch. Read the code (`grep -rn EVOLUTION_ src`) for the exact list. The groups are:

- Devices and threads: `EVOLUTION_DEVICES` (`primary` on this machine, never the Radeon), `EVOLUTION_CPU_THREADS` (size of the CPU failover pool, 0 turns it off), `EVOLUTION_RENDER_GPU` (adapter for drawing the window), `EVOLUTION_UI_FPS` (frame rate cap, 0 follows vsync).
- CUDA: `EVOLUTION_CUDA`, `EVOLUTION_NVRTC`, `EVOLUTION_CUDA_VERBOSE` (see the CUDA section).
- Measuring: `EVOLUTION_STAGE_LOG=<path>` writes one CSV row per generation. `EVOLUTION_PROFILE_BREED` prints archive and breeding timings.
- Benchmarks, tests and screenshots: `EVOLUTION_BENCH_*` drives the graphical benchmark mode (generations, duration, warm-up). `EVOLUTION_TEST_*` sizes the ignored GPU tests. `EVOLUTION_SMOKE_*` starts short screenshot runs, and their windows show on the desktop.
- Unattended runs: `EVOLUTION_AUTOSTART="Autochange environment=1"` sets the listed effect levels (the list may be empty), turns autosave on every 10 generations and starts evolving continuously.
