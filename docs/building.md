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
these settings to every game run and benchmark:

```bash
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 EVOLUTION_DEVICES=primary nice -n 10 cargo run --release
```

## CUDA on NVIDIA GPUs

The game needs an NVIDIA GPU with the CUDA driver and NVRTC. It evaluates
creatures through CUDA and nowhere else (`src/cuda_engine.rs`,
`shaders/warp_creature.cu`, `src/warp_kernel.rs`). The window is drawn
through Vulkan (wgpu), on the GPU the desktop uses.
`examples/warp_regs.rs` prints each kernel's registers and spills from NVRTC
alone; `EVOLUTION_WARP_PROFILE=1` makes one warp print its cycles per kernel
section. `EVOLUTION_WARP_SUBSTEPS`, `EVOLUTION_WARP_PGS_SWEEPS`,
`EVOLUTION_WARP_CLEAN_SWEEPS`, `EVOLUTION_WARP_PLANT_ROUNDS` and
`EVOLUTION_WARP_PLANT_SWEEPS` override the solver settings for measuring
them. `replay_match <save> --retest <count> <out.csv>` re-tests a save's best
elites with the settings in force, and `replay_match --ladder` compares the
files from runs at the default, 2 and 4 substeps (`docs/plan-2m-measurements.md`,
substep ladder). `EVOLUTION_WARP_BUCKETS=1` gives each wave one take-up counter
instead of one per muscle-rounds bucket.
Nothing needs configuring: the build links no CUDA library, and the engine
loads the CUDA driver library and NVRTC when it opens. If either is missing,
or the GPU is not an NVIDIA GPU, the game stops with an error that says so.

The driver library (`libcuda.so.1`) comes with the NVIDIA driver. NVRTC comes
with a CUDA toolkit, or, without root, from NVIDIA's pip wheel in a virtual
environment the engine looks for:

```bash
python3 -m venv ~/.local/share/evolution-cuda/venv
~/.local/share/evolution-cuda/venv/bin/pip install "nvidia-cuda-nvrtc==13.0.88"
```

Pick an NVRTC no newer than the driver's CUDA version (`nvidia-smi` shows it;
driver 580 is CUDA 13.0). The kernels of the current world compile when the
engine opens and when the world changes, and the kernels of every world one
effect level away compile after them at idle priority, about 1 to 2 s per kernel
the first time. They are kept in `~/.cache/evolution-simulator/cuda` (the
200 newest files), so later starts load them in milliseconds.

Developer diagnostics, never needed to play:
`EVOLUTION_NVRTC=/path/to/libnvrtc.so.13` names another NVRTC, and
`EVOLUTION_CUDA_VERBOSE=1` reports compile times.

## Diagnostic examples

The tools in `examples/` (`search_ab`, `size_report`, `mutation_audit`, `physics_audit`, `first_generation`, `replay_match`, `p2_speed`, `worker_rate`, and for a physics change `lean_gates`, which runs the spirit checks on a save's elites: `run`, `report`, `gifs`; `repeat_diff` and `order_diff`, which score one population repeatedly or in other orders and name the creatures that changed) score and replay creatures on the GPU engine. They fail if the primary GPU does not open. They submit at most 50,000 creatures per unit, so they need little GPU memory beside the owner's game. Run them with the lock shared, unless they measure speed:

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

While paused the game window shows "Paused for a developer measurement, resumes in m:ss" and a Resume now button. The replay keeps playing. A new replay cannot be recorded until the pause ends. Save, open and new game wait until the pause ends. The pause does not change the search: work is held back, never dropped, so a paused run of a fixed seed matches an undisturbed one (`tests/dev_pause.rs`). The code is in `src/dev_pause.rs` and `src/scheduler/suspend.rs`.

## GPU failures

A GPU engine that fails is dropped, and the scheduler opens it again after 1, 4 and then 10 s (`recover` in `src/scheduler.rs`). Each reopen submits the unfinished units again with their exact inputs, so they give the same results, and the event feed shows every attempt. There is no failover physics: after the third failed reopen the game stops evolving and shows the error, and it does not retry forever.

## Machine settings for measurements

These need root, so the owner runs them. `nvidia-smi -lgc <min>,<max>` pins the RTX 4060's SM clock for a measurement and `nvidia-smi -rgc` releases it. The power rows of the plan decide if a pinned clock also helps long runs.

`evolution-simulator headless ... --snapshot-at N` also saves the run at generation N next to the checkpoint (`<checkpoint>.gN.evo`), for the checks that compare generation 10 with generation 30.

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
4. At most two submissions in flight. If a fence wait exceeds 500 ms, stop submitting for the session.
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

## Power rows

These rows fix the power budget, the host tax, the sustained clock, the Radeon's cost and four root settings for the 2M/s plan (`docs/plan-2m.md`, section 6, item 1). The row numbers are the ones the plan and `docs/plan-2m-debate/round-2-os.md` use. Every row is logged by `tools/power-sample.sh` at 100 ms: SM and memory clock, GPU power, the live power limit (`enforced.power.limit`, which Dynamic Boost moves between 85 and 100 W), the limiter reasons, temperature, the APU package power (PPT) and the Radeon's clock and load. Its summary averages the samples where the RTX is at least 90% busy.

The tools:

- `examples/power_probe.rs <fma|mio|int|idle> [seconds] [cubin-dir]` runs one synthetic NVRTC kernel at full occupancy (48 warps per SM). `fma` is 8 chaotic FFMA chains per thread in registers. `mio` has one shared-memory load and one shuffle per two FFMAs, so 49% of its instructions go to the MIO pipes. `int` is IMAD, LOP3 and IADD3 chains with one shuffle in six instructions. It prints warp instructions per second from the loop's SASS count (131, 131 and 99 instructions per iteration with NVRTC 13.0 on sm_89, counted with `cuobjdump -sass` on the cubins it writes to `cubin-dir`) and the SM clock the kernel itself saw. The Ada peak is 4 warp instructions per SM per clock. nJ per warp instruction is the busy GPU power divided by that rate.
- `tools/cpu-burn.c <threads> [duty%] [period_ms] [seconds]` keeps N threads on AVX-512 FMAs for duty% of each period, all threads on the same period grid. It stops on SIGTERM and prints the busy core-seconds.
- `examples/p2_speed.rs` scores the 262,144 creatures of the generation-10 dump, one warm-up pass and 5 timed passes, about 28 s. The dump is at `~/.cache/evolution-simulator/power-dump.bin` (copied from the warp track's scratchpad `save_dump.bin`).

A row needs the RTX to itself and a quiet CPU, because Dynamic Boost takes the 15 W from the RTX as soon as the APU draws more. Other GPU work shares the SMs by time slices: the fma probe beside other agents' runs read 1.0 to 2.6 warp instructions per SM per clock at full clock, against a peak of 4. So a calibration row runs under `flock -x target/gpu.lock` with nothing else on the RTX, and the APU column shows how quiet the CPU was.

### The table

Each result is the busy summary of the sampler. For p2_speed rows the rate is the mean of the 5 timed passes in M creature-steps/s, and the tax is 1 minus the rate over row 0's. For probe rows it is G warp instructions/s, warp instructions per SM per clock, and nJ per warp instruction (busy GPU power over the rate, all-in, then above the idle reading). A blank row is the owner's (see below).

| row | load beside the RTX | needs | SM MHz | GPU W | limit W | limiters | APU W | result | taken |
|---|---|---|---|---|---|---|---|---|---|
| 0 | p2_speed alone | | 2490 | 67.9 | 99.9 | none | 29.6 | 12.3 (time-sliced with 3 other GPU processes) | agent, shared lock, 09-30 |
| 1 | p2_speed, 2 CPU threads at 100% | | 2482 | 71.2 | 99.6 | none | 29.5 | 9.4 (3 other GPU processes) | agent, shared lock, 09-30 |
| 2 | p2_speed, 4 CPU threads at 100% | | 2476 | 74.2 | 99.8 | none | 29.5 | 7.0 (5 other GPU processes) | agent, shared lock, 09-30 |
| 3 | p2_speed, 8 CPU threads at 100% | quiet machine | | | | | | not taken (no two free CPU slots) | |
| 4 | p2_speed, 16 CPU threads at 100% | the whole CPU | | | | | | | |
| 5 | row 3 with the CPU capped at 3.4 GHz | root | | | | | | | |
| 6 | row 3 with EPP balance_performance | root | | | | | | | |
| 7 | p2_speed, Radeon burner at 800 MHz, 100% duty | root, Radeon | | | | | | | |
| 8 | p2_speed, Radeon burner at 1,100 MHz, 100% duty | root, Radeon | | | | | | | |
| 9 | p2_speed, Radeon burner at 2,700 MHz, 100% duty | root, Radeon | | | | | | | |
| 10 | p2_speed, Radeon at 1,100 MHz and 30% duty, 4 CPU threads | root, Radeon | | | | | | | |
| 11 | power_probe fma, 30 s | | 2419 (1965 to 2505) | 82.8 | 96.7 | power cap 39%, SW thermal 15%, HW slowdown 5%, HW thermal 5% | 34.0 | 150.2 G/s, 2.64 per SM per clock, 0.55 nJ all-in (1 other GPU process) | agent, shared lock, 09-30 |
| 12 | power_probe mio, 30 s | | 2490 | 61.1 | 99.8 | none | 29.3 | 21.7 G/s, 0.36 per SM per clock, 2.8 nJ all-in (1 other GPU process) | agent, shared lock, 09-30 |
| 13 | power_probe int, 30 s | | 2486 | 73.3 | 99.6 | power cap 3% | 29.4 | 52.1 G/s, 0.87 per SM per clock, 1.41 nJ all-in (3 other GPU processes) | agent, shared lock, 09-30 |
| idle | power_probe idle, 10 s (a context, no kernel) | quiet GPU | | | | | | not usable: 2 other processes kept the GPU busy at 63.4 W | agent, shared lock, 09-30 |
| 14 to 16 | p2_speed at 4, 2 and 1 blocks per SM | engine code | | | | | | not taken, see below | |
| 17 | p2_speed, memory clock locked at 810 and at 6,001 MHz | root | | | | | | | |
| 18 | p2_speed, platform profile max-power | root | | | | | | | |
| 19 | p2_speed for 10 minutes | 10 min of GPU | | | | | | | |
| 19a | p2_speed, 4 and 8 CPU threads at 20% duty in 20 ms bursts | quiet machine | | | | | | not taken | |

The agent rows of 2026-09-30 (21:06 to 21:15) are not the calibration. The exclusive lock waited more than an hour behind other agents' 3M-creature searches, so they were taken under the shared lock with 1 to 5 other processes on the RTX. Those processes share the SMs by time slice, so every rate is a lower bound and the power includes their kernels. Rows 0 to 2 fall from 12.3 to 7.0M creature-steps/s because more processes joined, not because of the CPU threads, so they say nothing about the host tax (43 to 45M is the exclusive rate on this dump). The APU sat near 29 W in all of them and the limit stayed near 100 W. Two things they do show. The FMA kernel reached 2.64 warp instructions per SM per clock of the 4.0 peak even while sharing, and at that rate it met the power limit (39% of samples at the SW power cap, 15% thermal, clock down to 1,965 MHz). So a register-dense kernel does run into the cap on this laptop, and row 11 on a quiet GPU will give the cap clock. The MIO kernel issued only 0.36 warp instructions per SM per clock at full clock and 61 W, which is 0.18 shared loads and shuffles per SM per clock. If that holds on a quiet GPU, an MIO-heavy kernel is bound by the MIO pipe long before it is bound by power.

Rows 14 to 16 are not taken. The engine sizes each wave's grid to the resident capacity and runs up to 8 waves at once on separate streams, so a grid cap in p2_speed does not set the blocks per SM. They need a per-SM cap in `src/cuda_engine.rs`, and the plan's session (section 6, item 1) leaves them out.

### The owner's session

The rows left blank need root, more than half the CPU, the Radeon, or more than 5 minutes of GPU. The rows an agent took ran beside other agents' CPU work, which the APU column shows, so the session takes them again on a quiet machine. The owner runs it in five pauses and one 10-minute window, with no agent builds or GPU work during the session. First, once, from the repository root (the tools build into their own target directory, so the game's `target/release` is untouched):

```bash
CARGO_TARGET_DIR=target/power CARGO_BUILD_JOBS=8 nice -n 10 cargo build --release --example power_probe --example p2_speed
gcc -O2 -march=native -pthread -o target/power/cpu-burn tools/cpu-burn.c
mkdir -p target/power/rows
export EVOLUTION_DEVICES=primary RAYON_NUM_THREADS=4
export P=target/power/release/examples/power_probe D=~/.cache/evolution-simulator/power-dump.bin
export P2="target/power/release/examples/p2_speed $D 262144 5"
row() { n=$1; shift; tools/power-sample.sh target/power/rows/$n.csv -- "$@" 2>&1 | tee target/power/rows/$n.txt; }
burn() { target/power/cpu-burn "$1" "$2" 100 600 & b=$!; $P2; kill $b; wait $b; }
export -f row burn
```

`burn <threads> <duty%>` runs p2_speed beside the burner and stops the burner when p2_speed ends. Each pause below is one command. It pauses the game, runs its rows and lets the game resume.

1. Rows 11 to 13 and the idle reading, no root:

   ```bash
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r11 $P fma 30; row r12 $P mio 30; row r13 $P int 30; row idle $P idle 10'
   ```

2. Rows 0 to 3, no root:

   ```bash
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r00 $P2; row r01 burn 2 100; row r02 burn 4 100; row r03 burn 8 100'
   ```

3. Row 4 and row 19a, no root:

   ```bash
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r04 burn 16 100; row r19a-4 burn 4 20; row r19a-8 burn 8 20'
   ```

4. Row 19, no root, with the game closed (it is longer than a pause): 135 timed passes of about 4.4 s.

   ```bash
   flock -x target/gpu.lock bash -c 'row r19 target/power/release/examples/p2_speed $D 262144 135'
   ```

5. Rows 7 to 10, root for the Radeon's DPM level (the Radeon is `card2`). The burner is igpu's (`burner.c` in the radeon-rows track, built with `gcc -O2 burner.c -o target/power/burner -l:libEGL.so.1 -l:libGLESv2.so.2`, usage `burner <seconds> [target_ms] [duty]`, 10 ms dispatches). Start it from a second terminal with the Mesa vendor forced, check that its first line names the Radeon 780M, and give it 10 s more than the row:

   ```bash
   # terminal 2, before each row (duty 1.0 for rows 7 to 9, 0.3 for row 10):
   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json EGL_PLATFORM=surfaceless target/power/burner 45 10 1.0
   # terminal 1:
   echo manual | sudo tee /sys/class/drm/card2/device/power_dpm_force_performance_level
   echo 0 | sudo tee /sys/class/drm/card2/device/pp_dpm_sclk   # row 7, 800 MHz;  1 is 1,100 MHz (rows 8 and 10), 2 is 2,700 MHz (row 9)
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r07 $P2'
   # row 10: burner at duty 0.3, sclk 1, then: row r10 burn 4 100
   echo auto | sudo tee /sys/class/drm/card2/device/power_dpm_force_performance_level
   journalctl -k --since "15 min ago" | grep -E "amdgpu.*(timeout|reset)"   # must print nothing
   ```

6. Rows 5, 6, 17 and 18, root. Each setting is kept only if it moves the rate by 3% or more. Restore each one before the next:

   ```bash
   # row 5: 8 threads with the CPU capped at 3.4 GHz (compare with row 3)
   echo 3400000 | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r05 burn 8 100'
   for p in /sys/devices/system/cpu/cpufreq/policy*; do sudo cp $p/cpuinfo_max_freq $p/scaling_max_freq; done
   # row 6: 8 threads with EPP balance_performance (the EPP needs the powersave governor)
   sudo cpupower frequency-set -g powersave
   echo balance_performance | sudo tee /sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r06 burn 8 100'
   sudo cpupower frequency-set -g performance
   # row 17: memory clock locked at 810 MHz, then at 6,001 MHz (the listed clocks are 810, 6001, 7001 and 8001)
   sudo nvidia-smi -lmc 810,810
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r17-810 $P2'
   sudo nvidia-smi -lmc 6001,6001
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r17-6001 $P2'
   sudo nvidia-smi -rmc
   # row 18: platform profile max-power (the sampler logs the live limit)
   echo max-power | sudo tee /sys/firmware/acpi/platform_profile
   flock -x target/gpu.lock tools/pause-game.sh bash -c 'row r18 $P2'
   echo performance | sudo tee /sys/firmware/acpi/platform_profile
   ```

Each `target/power/rows/<row>.txt` holds the p2_speed or probe lines and the sampler's summary. The rate of a p2_speed row is the mean of its 5 timed passes, and its host tax is 1 minus its rate over row 0's.

## Environment variables

`cargo run --release` is the whole game and needs none of these. Every `EVOLUTION_*` variable is a developer diagnostic or a measuring control. Speed and search settings (GPU slots, batch and unit sizes, workgroup sizes, screening, the anatomy operators, joint damping, Hill speed) are fixed in the code and have no switch. Read the code (`grep -rn EVOLUTION_ src`) for the exact list. The groups are:

- Devices and threads: `EVOLUTION_DEVICES` (`primary` on this machine; other names add NVIDIA GPUs), `RAYON_NUM_THREADS` (lowers the general worker pool), `EVOLUTION_RENDER_GPU` (adapter for drawing the window), `EVOLUTION_UI_FPS` (frame rate cap, 0 follows vsync).
- CUDA: `EVOLUTION_NVRTC`, `EVOLUTION_CUDA_VERBOSE`, the `EVOLUTION_WARP_*` solver overrides (see the CUDA section).
- Measuring: `EVOLUTION_STAGE_LOG=<path>` writes one CSV row per generation. `EVOLUTION_PROFILE_BREED` prints archive and breeding timings. `EVOLUTION_DUMP_GENERATION=<generation>[:<path>]` runs one generation with the screen bar off, re-runs the island elites in it, and writes a 64 B row per creature and a 32 B row per elite (`storage::dump` has the layout). `dump_stats <path>` and `rung_replay <path>` read it. The CUDA kernel writes the rung trace (distances at 1, 2.5, 5 and 10 s, the early features, the end code with the rung that stopped the trial, the cadence bands and the audit bit) into seven result words nothing else reads. `rung_replay` also fits the game's own rules (`rungs::Audit`) on the dump's audit rows alone and measures them on the rest. `EVOLUTION_NO_RUNGS=1` removes the audit lane and the early rungs, to measure the game without them in the same build (`search_ab` prints one `rungs` line per generation: steps per creature, stops per rung, audit rows, the audit lane's miss estimate and how much of the final top 1% and 10% the ladder and the 5 s screen alone would keep; `--seconds N` stops a run after N seconds of wall time).
- Benchmarks and screenshots: `EVOLUTION_BENCH_*` drives the graphical benchmark mode (generations, duration, warm-up). `EVOLUTION_SMOKE_*` starts short screenshot runs, and their windows show on the desktop.
- Unattended runs: `EVOLUTION_AUTOSTART="Autochange environment=1"` sets the listed effect levels (the list may be empty), turns autosave on every 10 generations and starts evolving continuously.
