# Validation results

## Version-20 integration checks (2026-09-26)

The merge through 73ddf68 preserves the planted-foot direction rule, replay scrubber, secondary-device selection tests, and fast build profile. Both fast and release-fast profile names are supported. Formatting, all-target Clippy, 75 CPU tests, and all three RTX agreement tests passed (16.92 s for the GPU tests). The nine report tests passed before this merge; their code is unchanged. The required random-body diagnostic at 20,000 bodies and 20 s measured median -0.07 m, p99 0.34 m, and best 11.03 m. These are diagnostic motion statistics, not a GUI performance result.

## Integration of the concurrent physics and foundation work (2026-09-26)

The integration combines foundation commits `250ad82`/`ca75014` with `abb00cb` and its version-19 physics. Formatting, all-target Clippy, 72 release CPU tests, nine size-report tests, and all three explicit RTX GPU agreement tests passed. The GPU checks took 14.36 s; four GPU tests remain ignored in the default suite. The required 20,000-body, 20-second random-population diagnostic measured median -0.07 m, p99 1.85 m, and best 9.81 m. This is a propulsion check, not a proof of energy conservation or throughput. Remote CI execution is not yet verified.

Checkpoint continuation remains deterministic with CPU validation of archive entrants enabled. The fall fixture was adjusted to keep toppling under stronger ground grip while retaining its frozen-score and terminal-COM assertions. The size report now obtains frames and their result from one `cpu_engine::replay` call and measures per-node slip and visible head motion only through the scored endpoint. Position-derived acceleration is labeled separately from the engine's recorded head-shake value because position-only corrections contribute to visible movement.

## Historical version-16 baseline and foundation checks (2026-09-26)

This section records parent physics revision `bd41746` (QD version 16) and local foundation commit `250ad82`, before the merge of `abb00cb`. The later version-19 physics adds load-aware friction, capped planted-foot propulsion, head-shaking termination, CPU validation of global archive entrants, and replay results recorded with frames. The baseline measurements and test counts below do not validate those merged changes. Loading the baseline checkpoint under version 19 invalidates its archive.

The foundation batch itself left the production contact solver, physics limits, distance-only fitness, and `qd::VERSION` unchanged. It adds regression coverage, aligns the public `physics::evaluate` wrapper with the CPU engine, repairs checkpoint restart state and Windows support, and configures CPU CI. The lower-level legacy `physics::step` remains for existing tests and diagnostics.

### Baseline after the 2 m bone cap

The prescribed headless run used seed 38, 100,000 candidates per generation, 20 generations (logged as 0–19), and 60-second trials. It ran on Windows with an RTX 4060 Laptop GPU, NVIDIA driver 610.47, and Rust 1.98.1. The native release build disabled LTO and used 256 codegen units. `EVOLUTION_DEVICES=primary`, six CPU evaluation threads, two general workers, and low priority kept the desktop Radeon out of evaluation.

| Measurement | Result |
| --- | ---: |
| Final best archive distance | 165.5846 m |
| Final median archive distance | 8.8442 m |
| Final behavior cells | 1,374 / 1,440 |
| Final QD score | 23,060.27 |
| Reported failed candidates across 20 generations | 0 |
| Top-50 reported median total bone length | 2.23 m |
| Longest individual bone among the top 50 | 1.81 m |
| Top-50 reported median slip per replay meter | 0.89 |
| Champion archive / CPU replay distance | 165.6 / 157.3 m |
| Champion total bone length / mass | 1.77 m / 3.20 kg |

There is no pile-up at the 2 m bone cap in this top-50 sample. This single-seed baseline does not establish that size selection is solved for other seeds or longer runs. Foot slip remains material. Archive rank 21 scores 111.2 m but replays at 6.4 m; that discrepancy remains unresolved. Fine perturbed checks and device-sensitive fall/break thresholds can produce different trajectories, but the cause of this outlier has not been established.

`size_report` now measures slip only during the scored CPU replay interval, excludes initial recentering and post-fall motion, uses the configuration's fidelity, and includes terrain slope in contact detection. It divides summed touching-node horizontal slip by absolute terminal replay distance. Its body length is the sum of bone rest lengths, and its even-sample median selects the upper middle entry. The earlier 0.02 and 2.9 slip-per-meter reports used different diagnostic semantics and cannot be compared directly with 0.89. A champion ratio printed as `0.00` is rounded; its reported absolute slip is 0.7 m.

The sanitized [generation log](results/2026-09-26-bone-cap-seed-38/generations.csv), [top-50 report](results/2026-09-26-bone-cap-seed-38/top-50.csv), and [run metadata](results/2026-09-26-bone-cap-seed-38/metadata.json) preserve the printed measurements without machine-local paths or the checkpoint. Timing scope is recorded in the [performance log](performance-log.md).

### Regression and GPU validation status

New coverage exercises configuration boundaries, one fastest eligible archive entry per cell, migration, deterministic valid offspring across streaming slice sizes, checkpoint continuation including stalled island optimizers and environment changes, CPU score/replay agreement at standard and fine fidelity, and frozen scores after falls or joint breaks. `size_report` has focused tests for scored intervals and contact geometry.

Checkpoint format V4 persists island optimizer progress that V3 omitted; V3 remains readable. Obsolete-physics loads clear stale island and reseed state. This is a storage-format repair, with no change to `qd::VERSION` or the current production physics solver.

| Check | Status |
| --- | --- |
| Formatting | `cargo fmt --all --check` passed |
| Clippy | `cargo clippy --locked --all-targets -- -D warnings` passed |
| Release CPU suite | 68 passed: 30 library, 5 replay, 15 search-state, and 18 simulation tests; four GPU tests ignored |
| Size-report example | Seven tests passed |
| Explicit local simulation GPU suite | All three tests passed in the final rerun (3.32 s), with the RTX primary device and no Radeon evaluation |
| Default test selection | Four GPU-dependent tests remain ignored, including the worker test |
| GitHub Actions | CPU workflow configured; remote execution not yet verified |

The checks above were run by the coordinator on foundation commit `250ad82`, before merging the later physics. Two test-only comparison failures were resolved by treating empty-archive cached QD scores of `+0.0` and `-0.0` as numerically equal; exact elite, population, and CMA comparisons remain. No active stepping change was needed.

The local GPU suite covers partial workgroups/body buckets, rough ground, and narrow joints. Its standard/fine comparisons do not establish full-trial agreement for every evolved elite or resolve the rank-21 replay outlier. An evolved-creature fine-fidelity fixture and investigation of threshold-sensitive contender outcomes remain open.

### Historical contact audit correction (version 16)

At the inspected version-16 revision, the friction budget already includes positional clamp/lift displacement in `final_y - predicted_y`; the old description that those corrections were entirely absent was incorrect. The budget still uses the final touching node's own mass, missing support transferred through bones to the rest of the body. Later velocity clamps remove downward motion without adding that normal impulse to friction and can restore slip. This is a source-level finding, not a validated contact-solver fix. See the [physics audit](physics-audit-2026-09-26.md) for the remaining invariants and fixtures.

## Historical validation records

The sections below retain results under their original revisions and workloads. Their trial lengths, caps, kernels, and test counts do not describe the current defaults. In particular, the temporary 10 m bone / 5 m muscle limits were subsequently restored to 2 m, and the current game uses 60-second trials.

## Larger limits and broken joints (2026-09-26)

The physics limits were raised so that large runners can reach a kilometer in a minute: 60 m/s nodes, 40 rad/s bone turning, 24 m/s and 100 N muscles, 0.2 s rhythms, 10 m bones, 5 m muscles, and no air drag by default. The earlier stability checks covered 2 m/s muscle targets, 5 m/s nodes, and 15 rad/s bones.

With the larger limits, the existing joint-range test failed: a few of 64 small random creatures forced a joint through its limits and round a full turn, jamming it up to 2.8 rad outside its range. Any muscle force above 5 N or node speed above 5 m/s was enough. Strong muscles drive the joint faster than the per-step turn limit lets the joint projection pull it back. The fix does not change the solver. A joint forced more than 0.5 rad past its range breaks and ends the trial like a fall, in the CPU engine, the GPU kernel, and the replay. The test now checks joints up to the end of the trial, and a new unit test checks the break threshold. Evolved runners keep their joints within 0.06 rad of their ranges over a full trial.

The GPU kernel change compiles (naga parses the substituted shader for standard and fine physics), but this container has no GPU, so CPU/GPU trajectory agreement was not rerun.

## Rejected whole-creature lane kernel (2026-09-25)

A full-window, five-generation comparison at 5,000 creatures and 1-second trials was run while the desktop compositor remained active. The lane-per-creature experiment evaluated at 17.8k creatures/s (0.280 s evaluation/generation) and completed 14.75 generations/s; the shared-memory kernel evaluated at 46.9k creatures/s (0.107 s evaluation/generation) and completed 34.44 generations/s. This short, non-default-duration trial is diagnostic only, but it clearly regressed, so the experiment was removed. The default 18-second workload remains the acceptance workload.

Throughput figures in the historical sections below predate rigid bones and do not describe the current physics. This change was checked with low-impact correctness tests; no throughput profile was run while the CPU/GPU were busy.

## Rigid-bone correctness audit (2026-09-24)

A 64-node chain with 63 muscles at maximum stiffness ran for 560 CPU steps with gravity and ground contact. The test checks every bone after every step and requires its length to stay within 0.1 mm of its rest length, with all nodes above the ground. The GPU smoke test checks the same 64-node, endpoint-anchored morphology after 1, 201, and 320 steps. The normal CPU/GPU trajectory test also compares positions and velocities through ground contact. A checkpoint migration test reverses a saved skeleton's bone order, reloads it, and verifies the normalized skeleton preserves every muscle attachment point. These checks use the parent-first exact reconstruction added after the eight mass-weighted projection passes; no throughput benchmark was run for that solver change.

The stability checks cover a 2 m/s muscle-target rate limit, a 5 m/s node-speed cap, and a 15 rad/s per-bone rotation limit. A collapsed zero-velocity chain is reconstructed to exact rest lengths without gaining speed, and the GPU smoke test checks the same case alongside CPU/GPU trajectory agreement. Fitness now tracks horizontal center-of-mass displacement, so a posture change alone cannot earn travel distance. The QD version is bumped so saved archives and scores are reevaluated under the changed physics. No throughput profile was run while the CPU/GPU were busy.

## Native app measurement (2026-09-23)

The graphical Wayland app was run on the NVIDIA RTX 4060 with driver 580.173.02, a fixed seed, 18-second trials, and 30 complete generations. The native benchmark includes GPU evaluation, archive insertion, breeding, worker publication, and active UI rendering. These are sequential runs on the same machine; background desktop activity and GPU clocks can affect small differences.

| Population | Earlier 64-step run | Current app default | Observed speedup |
| ---: | ---: | ---: | ---: |
| 1,000 | 54.69 generations/s | 98.30 generations/s | 1.80× |
| 10,000 | 14.40 generations/s | 21.04 generations/s | 1.46× |
| 100,000 | 1.91 generations/s | 3.18 generations/s | 1.67× |

The current 30-generation runs spent 0.219/0.038/0.048 seconds in evaluation/archive/breeding at 1,000 creatures; 1.023/0.165/0.236 seconds at 10,000; and 7.550/0.900/0.980 seconds at 100,000. A three-generation run at one million creatures completed at 0.435 generations/s, spending 5.270/0.455/1.177 seconds in those stages. All population sizes use 4096-step dispatches by default. Exact 4-node buckets help across population sizes, while the 5-node bucket helps only at smaller populations. The shader keeps behavior metrics in registers and precomputes muscle reciprocals while packing each GPU batch.

On the one-million-creature run, parent planning took 0.151–0.184 s per generation, candidate emission took 0.182–0.222 s, and breeding finalization took about 0.019 s. GPU timestamps from a separate 10-generation run at 100,000 creatures attributed 1.719 s of 2.052 s evaluation time to shader execution: 0.656 s in the 4-node bucket, 0.966 s in the 8-node bucket, and 0.098 s in the 16-node bucket. Packing took 0.163 s. These timings identify where further work may help; profiling adds overhead.

An NVIDIA Nsight guidance review highlighted group barrier stalls as a possible limiter. NVIDIA’s [shader profiler guide](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html) describes barrier stalls as warps waiting for sibling warps and recommends checking whether each group synchronization is needed. An experimental shared muscle-target table reduced duplicate cosine work but performed much worse: 0.92 generations/s at 100,000 creatures, so that variant was removed. Raising the dispatch chunk from 1024 to 4096 steps reduced native 100,000-creature generation time by about 7.5%; a repeated 30-generation run measured 3.18 generations/s, versus 2.96 with 1024-step dispatches. At one million creatures, 4096-step dispatches measured 0.435 generations/s with 100,000-creature batches, versus 0.409 with the earlier default. A 250,000-creature batch did not improve that result. The separate compute device is the graphical default; the full 10× generation-throughput goal remains open.

### Paired kernel comparison at 100,000 creatures (2026-09-23)

Four 20-generation full-GUI runs on revision `5f0349c` used default/workgroup/workgroup/default order. Host load stayed between 1.87 and 2.77 on 16 logical CPUs; CPU pressure was zero during the comparison. GNOME Shell contributed a steady desktop GPU load, and each run recorded 98–100% peak SM use.

| Kernel mode | Generations/s | Evaluation seconds / 20 generations | Runs |
| --- | ---: | ---: | ---: |
| Default (serial private arrays for 4/5/8-node buckets) | 1.828–1.829 | 9.875–9.907 | 2 |
| Workgroup | 3.063–3.104 | 5.449–5.477 | 2 |

The workgroup path delivered 1.69× the throughput of the serial path at that revision. The serial and lane shaders use the earlier node-to-node genome layout, so the bone-physics update disables both paths; the current simulator uses the workgroup shader for every body size.

### Low-impact timestamp profile on the current default (2026-09-24)

A five-generation full-GUI diagnostic at 100,000 creatures ran pinned to CPUs 4–15 at nice priority 10. Before the run, CPU pressure was 0%, load average was 1.82, and the GPU showed 18% desktop utilization. The profiled run completed in 1.44 s; these measurements are diagnostic and should not be compared with unprofiled throughput runs.

| Measurement | Result |
| --- | ---: |
| Evaluation | 1.174 s |
| GPU shader timestamps | 1.027 s |
| Packing | 0.144 s |
| Upload/encode | 0.072 s |
| Bucket 4 / 8 / 16 shader time | 0.413 / 0.577 / 0.037 s |

The 4- and 8-node buckets account for about 96% of timestamped shader time. The reported 1.050 s readback duration is CPU wall time waiting for mapped results and overlaps GPU execution; do not add it to shader time. This points the next kernel experiment toward the 4- and 8-node dispatches. The short run was kept isolated because profiling drives the GPU to full utilization.

### Paired workgroup-size comparison at 100,000 creatures (2026-09-24)

Four five-generation full-GUI runs at revision `7798381` used default/32-lane/32-lane/default order. Each run was pinned to CPUs 4–15 at nice priority 10, with no timestamp instrumentation. Before the sequence, CPU pressure was 0%, load average was 0.92, and GPU desktop utilization was 19%.

| Workgroup size | Run 1 generations/s / evaluation s | Run 2 generations/s / evaluation s | Mean generations/s / evaluation s |
| --- | ---: | ---: | ---: |
| Default (64 lanes for large populations) | 3.477 / 1.170 | 3.492 / 1.169 | 3.485 / 1.170 |
| 32 lanes | 4.119 / 0.955 | 4.002 / 0.981 | 4.061 / 0.968 |

The 32-lane variant improved this short paired sample by 16.5% in full-generation throughput and reduced evaluation time by 17.3%. A follow-up 10-generation full-GUI run with 32 lanes as the default measured 4.539 generations/s (evaluation 1.681 s, archive 0.184 s, breeding 0.336 s). This confirms the default path; longer repeats are still needed for a stable estimate. `EVOLUTION_WORKGROUP64=1` remains available for comparisons.

Measurements below were captured on the local NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB), Ubuntu 24.04 Wayland, with Rust release builds and the simulator's throughput mode. Each creature ran the default 15-second trial. These are workload measurements, not fixed hardware guarantees.

These records predate the MAP-Elites archive and emitter loop. They document GPU simulation throughput and UI responsiveness; they are not performance measurements of the current archive insertion and offspring-generation work.

## Generation throughput

| Population | GPU evaluation / generation | Full generation with the earlier ranking/reproduction loop | Average evaluations/s | Failed trials |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0.34 s | 0.37 s | 294,278 | 0 |
| 1,000,000 | 2.56–2.82 s across a 10-generation run | 2.83–3.11 s | 355,000–383,000 | 0 |
| 3,000,000 | 7.27–7.83 s across three generations | 8.04–8.63 s | 383,000–413,000 | 0 |

At 100,000 creatures, the CPU reference took 5.11 seconds to evaluate the same population and configuration: the observed GPU evaluation was 15× faster. CPU and GPU timings come from the same benchmark executable and machine.

After increasing the batch size and removing per-creature staging allocations, a fresh release run evaluated one million creatures in 2.63 seconds (381,000 evaluations/s) and three million in 6.13 seconds (489,000 evaluations/s). During those runs, `nvidia-smi dmon` reported 82–100% SM utilization; memory-controller utilization remained low because this workload is compute-bound. The three-million run used 720 MiB of process RAM and 35 MiB of tracked GPU buffers.

Ten successive one-million-creature generations completed without failed trials or accumulating memory growth. `/usr/bin/time -v` reported a peak RSS of 1,068,196 KiB (about 1.02 GiB). The three-million-creature, three-generation run peaked at 2,702,408 KiB (about 2.58 GiB).

## Interactive display

The native Wayland screenshot capture rendered the dashboard during a one-million-creature run. Across 240 sampled frames, p95 frame time was 10.36 ms, below the 33 ms target. Closing the window after capture exited cleanly after the worker and GPU work shut down.

The updated light-theme population view rendered during a one-million-creature evaluation at 14.92 ms p95 across 240 frames, also below the 33 ms target.

## Automated checks (historical)

- Release test suite: 10 standard tests and the separately invoked Vulkan GPU agreement test passed, covering partial GPU workgroups, 3/5/6/8/9/17/33/64-node GPU buckets, collider contacts, deterministic evolution, mutation limits, checkpoint resumption, corrupt-checkpoint checksums, and history validation.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.
- Headless interruption: Ctrl+C saved a partial first-generation checkpoint after 26.2% of a one-million-creature generation; resuming that file continued from the saved evaluation count and completed the generation.

Full benchmark records are in [`benchmark-comparison.csv`](benchmark-comparison.csv), [`benchmark-scale.csv`](benchmark-scale.csv), and [`benchmark-soak.csv`](benchmark-soak.csv).
