# Round 3, iGPU domain: the Radeon's place in the final design

Author: the iGPU expert. Date 2026-09-30, evening. I do not propose a competing pipeline. I back the converged one (per-lane kernel, ladder R1 to R4, CPU breeding into pinned SoA genes, take-up unpack on the RTX, candidate bit in the kernel, ring by measured latency) and state what the Radeon does in it, which is UI work and a documented fallback. Numbers first.

## 1. Where the Radeon sits in the per-lever table

| lever | at 3T | at 5T | owner | Radeon's part |
|---|---:|---:|---|---|
| Radeon physics | 1.00x | 1.00x | closed | none |
| Radeon packing | 1.00x | 1.00x | withdrawn | none: the RTX unpacks from genes at take-up, so there is no pack to move |
| Radeon parametric breeding (fallback) | 1.00x, or 1.03 to 1.10x if the trigger in section 4 fires | same | igpu | conditional |
| UI on the Radeon | 1.00x on throughput; keeps the RTX's one-context state that was measured at 1.28x on 2026-09-25 | same | ui and igpu | permanent |

So the Radeon contributes nothing to the multiplication and the design lands on whichever row the kernel and the ladder reach (the chair's table: 1.0 to 1.5M/s with 2 substeps and R1 to R3, 1.8 to 2.7M/s with 1 substep and R1 to R3). My job in the final design is to keep the Radeon from costing anything: no context switches on the RTX, no desktop crash, and no CPU time in the UI that the breeder needs.

The one measured Radeon number this debate produced: the burner runs dependent FMA chains at 1.27 T FMA/s (2.5 TFLOPS) with every dispatch under 10 ms and no amdgpu timeout or reset in the journal. That is 17% of the RTX's peak and it draws from the APU's package budget. Every other Radeon number is an estimate until os's rows run.

## 2. The Radeon's role: UI rendering, with one concrete change

The window already renders on the Radeon (ui.rs picks the compositor's vendor), so the RTX holds one context, the game's CUDA context, and the compositor's 2 MiB EGL context. That stays. What I add is one change with a number.

The archive map and the cards. The archive view draws each cell as a `painter.rect_filled` (ui.rs around lines 1125 to 1200), so egui tessellates about 7,200 cells per frame, 14,400 triangles, plus the text and strokes, on the UI thread, at 60 FPS. That is 1 to 2 ms of UI-thread CPU per frame by egui's usual tessellation cost of about 100 to 200 ns per rect, which is a core the breeder wants. The snapshot changes at most a few times per second. Proposal: the worker writes the map as a small `ColorImage` (one texel per cell, 90 x 80 for 7,200 cells, 29 KB) into the snapshot, the UI uploads it as a texture once per snapshot epoch and draws one textured quad with nearest filtering, and the hover and selection logic reads the CPU-side cell array as today. The same applies to the champion thumbnails on the cards: render each thumbnail once into a texture atlas when the card list arrives (`Command::Cards`), not every frame. Expected: the UI thread's per-frame cost with the archive view open drops from about 2 to 3 ms to under 0.5 ms, and the tessellation no longer scales with the grid. Measurement: the UI already keeps `frame_times` (ui.rs line 1377); print p50 and p99 with the archive view open before and after, under a running search, and the number of tessellated vertices per frame from `egui::Context::tessellation` stats. Gate: p99 frame time under 10 ms with the archive view open during breeding, which is also os's P4 target. This is UI work and the ui domain owns the code; I raise it because it is the Radeon's only per-frame job and it is the cheapest way to give the breeder a core back.

Nothing else moves to the Radeon. The replay is recorded by the scoring kernel and drawn as egui shapes; that is already right.

## 3. Safety rules for docs/building.md

Proposed text, to be added under a heading "The Radeon 780M" in docs/building.md. It is written for a developer who wants to run anything on the integrated GPU, including the UI and diagnostics.

    The Radeon 780M drives the desktop. Any compute on it shares the
    compositor's GPU, its 512 MB VRAM carve-out (85% used by the desktop)
    and the APU's 54 W package budget with the CPU. On this kernel
    (7.0.0-34) the amdgpu lockup timeout is 2,000 ms for every ring
    (`modinfo amdgpu`, parameter unset). A submission that runs longer is
    reset; the journal shows two such resets from this game's process
    (2026-09-25 17:55, recovered by a queue reset; 2026-09-26 13:06, a
    MODE2 whole-GPU reset that lost VRAM and the desktop). Rules for
    anything that submits work to the Radeon:

    1. No submission may take longer than 50 ms. Size dispatches from a
       measured rate and halve them when one exceeds 20 ms. No persistent
       kernels, no in-kernel work loops, no unbounded step counts.
    2. Use a compute-only queue at VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR (no
       privilege needed). Never raise the game's priority above the
       compositor's.
    3. Allocate only host-visible memory (GTT or imported host memory
       through VK_EXT_external_memory_host). Never DEVICE_LOCAL on the
       Radeon. Stay under 512 MB.
    4. At most two submissions in flight. If a fence wait exceeds 500 ms,
       stop submitting for the session and fall back to the CPU path.
    5. Run Radeon compute in a helper process that shares memory with the
       game, so a VK_ERROR_DEVICE_LOST kills the helper, not the game.
    6. Vulkan through RADV only. No ROCm, no OpenCL, no second driver stack
       on the display device. When using EGL or Vulkan from a tool, force
       the Mesa vendor (`__EGL_VENDOR_LIBRARY_FILENAMES=
       /usr/share/glvnd/egl_vendor.d/50_mesa.json`, or select the RADV
       physical device by name); glvnd picks NVIDIA by default.
    7. Before shipping anything that uses the Radeon, soak it for 10
       minutes with the game window open and the compositor at 60 FPS:
       pass is UI frame time p99 under 20 ms, no `ring comp` or `reset`
       line in `journalctl -k`, and the RTX rate unchanged.
    8. Never evaluate, confirm or replay creatures on the Radeon. The GPU
       score is final and it is the RTX's.

    Diagnostics: `journalctl -k | grep -E "amdgpu.*(timeout|reset)"` after
    any Radeon experiment; `/sys/class/drm/card2/device/gpu_busy_percent`
    and the amdgpu hwmon (`power1_average` is the APU package power in
    microwatts, `freq1_input` the shader clock) for load and power.

Rule 1's 50 ms is 4% of the timeout, and the halving rule keeps a dispatch that drifts (thermal, a heavier body class, the compositor stealing time) from ever reaching it.

## 4. The fallback breeding plan and its exact trigger

The plan, unchanged from round 1 and narrowed in round 2: the Radeon breeds only the parametric children (CMA samples and local gaussian children, about 60% of offspring) and the immigrants, in cpu's fixed-array gene format, from the versioned elite table in the shared host ring, straight into the pinned SoA gene ring the RTX reads at take-up. Structural children stay on the CPU. The Radeon runs at its 800 or 1,100 MHz DPM level under the rules above, in a helper process. Determinism: the same counter-based RNG keyed by (seed, generation, breed round, slot, gene, draw), integer-exact gaussian (section 6), so the child is the same whichever device breeds it, and the search does not change when the Radeon drops out.

It is built only when all three of these are measured true, in this order:

1. cpu's `breed_bench` on save42.evo after the container rewrite shows more than 12,000 weighted cycles per child, or `plan_offspring` above 2,000 cycles per child, so the CPU chain at 2M/s needs more than 3 busy cores (more than 4.5 core-seconds per 3M generation).
2. data's RTX recipes for parametric children are refused or measured above 3% of RTX time at the generation-51 mix.
3. os's power table shows, in the same run, that those 3 or more busy cores cost the RTX more than 5% of rate at the cap, and that the Radeon burner row at the 1,100 MHz level and 30% duty costs the RTX under 3%.

If 1 and 2 hold and 3 fails, the answer is os's frequency cap on the breeding cores, not the Radeon. If 3 holds and 1 or 2 fails, nothing is built. The expected value if it fires is the difference of the two power rows, which cpu estimates at 20 to 25 W of package or up to 10% of RTX rate at the cap, and which I put at 3 to 10%. Cost: a second implementation of the parametric emit and repair (a few hundred lines of GLSL plus the helper), the double pinning test (amdgpu userptr and cuMemHostRegister on the same anonymous pages, one afternoon), and the soak. Gate before merge: 1M children bit-identical to the CPU emitter for the same plan, the soak of rule 7, and `BREED_NANOS` per stage on save42 showing the CPU chain under 3 busy cores at the measured rate.

I concede to cpu's request: nothing of this is built before the `breed_bench` number exists.

## 5. The Radeon rows on os's table: the exact recipe

Purpose: the cost of Radeon load to the RTX's rate and clock, and the APU package power at each Radeon level. Six rows plus one soak. About 6 minutes of exclusive GPU time for the rows; the soak runs beside the owner's game and needs no lock.

Prerequisites. The burner binary is at scratchpad/igpu/burner (source burner.c beside it, `gcc -O2 burner.c -o burner -l:libEGL.so.1 -l:libGLESv2.so.2`). Usage: `burner <seconds> [target_ms=10] [duty=1.0]`. It tunes its loop so one dispatch lasts about `target_ms`, runs for `seconds`, and prints dispatch p50, p99 and max and the FMA rate at the end (it prints only at the end, so never kill it before its time is up; give it 10 s more than the RTX run). Start it from a terminal, not from a script that backgrounds it inside a `case`: the row script's background launch did not run in round 2 (Radeon busy stayed at 0 to 18%). Always with the Mesa vendor forced:

    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
    EGL_PLATFORM=surfaceless scratchpad/igpu/burner 60 10 1.0

Check its first line says `renderer: AMD Radeon 780M`. If it says NVIDIA, stop: it is loading the RTX.

The RTX side: `p2_speed` from a worktree with the current kernel on scratchpad/warp/save_dump.bin, 262,144 creatures, 3 timed passes (`EVOLUTION_DEVICES=primary EVOLUTION_CUDA=1 EVOLUTION_KERNEL_CACHE=scratchpad/warp/kcache target/release/examples/p2_speed scratchpad/warp/save_dump.bin 262144 3`), about 25 s. Under `flock -x target/gpu.lock tools/pause-game.sh` if the owner's game runs. No other GPU process during a row: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` must show only p2_speed, and `fuser target/gpu.lock` only this run. Round 2's rows were spoiled by a second p2_speed.

The sampler: scratchpad/igpu/sample.sh writes `t, rtx_sm_mhz, rtx_w, rtx_limit_reasons, apu_ppt_w, radeon_mhz, radeon_busy` at 4 Hz. Start it 3 s before p2_speed and stop it after. Report per row: the three p2_speed rates (creature-steps/s per wall second and per GPU-busy second), RTX SM clock mean and minimum during the timed passes, RTX power mean, the limiter reasons seen, APU PPT mean, Radeon clock and busy percent mean, and the burner's dispatch p99.

Rows, each one p2_speed run:

1. No load (the reference; repeat it at the end as row 1b to bound drift).
2. Burner at 100% duty, DPM auto (the Radeon will sit at 2,700 MHz: the cost of a flat-out Radeon, which is what physics on it would have cost).
3. Burner at 30% duty, DPM auto.
4. Burner at 10% duty, DPM auto (breeding-like duty).
5. Root: `echo manual > /sys/class/drm/card2/device/power_dpm_force_performance_level; echo 0 > pp_dpm_sclk` (800 MHz), burner at 100% duty.
6. Same with `echo 1 > pp_dpm_sclk` (1,100 MHz), burner at 100% duty. Then `echo auto > power_dpm_force_performance_level`.

The CPU rows (0, 2, 4, 8, 16 busy threads) are os's and run in the same session with the same sampler, so the Radeon rows and the CPU rows share a reference. Row 6 against the 2- and 4-thread CPU rows is the number that decides section 4's condition 3.

The soak, no lock: burner at 30% duty and 10 ms dispatches for 10 minutes with the game window open on the desktop and the search running, the sampler beside it, then `journalctl -k --since "15 min ago" | grep -E "amdgpu.*(timeout|reset)"` must be empty and the UI's frame-time p99 (from `frame_times`, once the UI prints it) under 20 ms. This is the rule 7 soak for the fallback and for any future Radeon use.

## 6. The gaussian, in one paragraph

I withdraw the pinned-FMA polynomial and take the integer sum. The chair's reason is the right one: an integer rule needs no contraction rule in any compiler, and rustc, NVRTC and ACO each get a chance to differ on a polynomial's last bit. A 12-term sum of 16-bit uniforms from splitmix hashes costs about 12 hashes and 12 adds, about 100 integer ops per gaussian, and cuts the tails at 6 sigma with a kurtosis slightly below the normal's. For mutation noise that is harmless, and ga has ruled that a changed stream is a new seed anyway. It applies only if a second breeder is ever built (section 4); with one breeder per child, no bit equality is needed at all.

## 7. The pipeline as I back it, in the Radeon's terms

Breeding on the CPU (cpu), genes written once into pinned SoA host memory, the RTX unpacks at take-up (gpu, data), results and candidate bits in pinned memory (os, data), the CPU commits candidates in ring order, the ring 0.3 to 1 s deep by measured latency (data), blocks of 50 ms (data, ga). The Radeon draws the window and, after section 2, one textured quad for the archive map and an atlas for the thumbnails. Determinism: one implementation per child, ring-order absorption, a fixed pair of GPUs and driver versions. World change: blocks not yet on the RTX are retargeted, blocks on it enter no archive; the Radeon holds no block state, so nothing on it changes. Save: archives and search state only; nothing on the Radeon persists. Replay click: the scoring kernel records one creature on its own stream; the Radeon draws the frames. 60 FPS: the window on the Radeon, section 2's textures, os's scheduling changes (rayon at 14, SCHED_BATCH, the UI slice). What stays out: Radeon physics, Radeon packing, any Radeon breeding before section 4's trigger, ROCm.

Tracks in my domain, in order, each mergeable alone:

1. docs/building.md section (section 3). No code. Gate: none.
2. Archive map and thumbnail textures (section 2, ui domain codes it). Gate: frame p99 under 10 ms with the archive view open during breeding, tessellated vertices per frame down by more than 10x.
3. The Radeon rows on os's table (section 5), after the debate, 6 minutes exclusive. Gate: the row numbers exist with the CPU rows beside them.
4. The fallback breeder (section 4), only on its trigger. Gate: bit-identical children, the soak, the CPU chain under 3 busy cores.

## 8. One new idea: keep the PCIe link at Gen 4

Not raised by anyone. Measured right now on the idle machine: the RTX's link is at Gen 1 x8 (`nvidia-smi --query-gpu=pcie.link.gen.current` prints 1, max 4), the device's runtime PM is `auto`, and os's trace saw the link train up to Gen 4 under load and drop to Gen 1 when idle. A Gen 1 to Gen 4 speed change is a link retrain; on PCIe 4.0 hardware that is 1 to 5 ms during which no DMA moves and the driver's uploads wait. Today's waves are seconds long and the link stays trained. In the converged design the blocks are 50 ms and the uploads are 3 to 10 MB bursts of genes per block with the results dribbling back; if the link's idle timer (the driver's dynamic link speed switching, on the order of tens of milliseconds) fires between bursts, every block pays a retrain on its first upload: 1 to 5 ms of 50 ms, a 2 to 10% stall on the upload path, hidden only if the ring is deep enough that the kernel never waits for a gene block. At the floor ring depth of 0.3 s it is hidden; at a shallow ring after a world change it is not, and it adds to replay latency, which waits for the link too.

Number: 0 to 3% of rate in the steady state (hidden by the ring), 1 to 5 ms on every replay click and on the first block after a world change. Measurement, 10 lines: `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv -lms 10` beside `worker_rate` at 50 ms blocks; count gen transitions per second and the p99 upload latency from the engine's timestamps. If transitions exceed 1 per second, test the fix: `nvidia-smi -pm 1` (persistence mode, root) and `echo on > /sys/bus/pci/devices/0000:01:00.0/power/control` (root; keeps the device out of runtime suspend), then repeat. Cost: 1 to 2 W at idle. It keeps distance-only fitness, one physics and no knobs: a machine setting the owner applies once, documented beside `-lgc`.

## 9. What would break what I back

- The textured archive map: if the hover and selection code depends on egui shape ids per cell. It does not need to; the cell array is on the CPU. If frame p99 does not move, the cost was never tessellation and the change is withdrawn.
- The safety rules: a future kernel raising the amdgpu timeout back to 10 s makes rule 1 conservative, not wrong. A compositor that submits long compute (none here) would break the priority rule.
- The fallback trigger: if `breed_bench` lands under 12,000 cycles the fallback never fires, which is the outcome I expect and prefer.
- The PCIe idea: if the link stays at Gen 4 throughout a search, it is a note in docs/building.md, not a track.
