# Evolution · Creature Laboratory

A local Rust evolution game for Ubuntu/Wayland. Creatures learn to move through quality-diversity search: a MAP-Elites archive keeps the best creature in each measured combination of contact, gait cadence, and vertical motion. WGSL compute shaders evaluate independent creatures on the GPU; an egui dashboard lets you inspect the archive and tune the experiment.

The original Processing sketch is preserved in [`old_code.txt`](old_code.txt). This is a modernized simulation, not a bit-for-bit reproduction of its physics or random sequence.

## Original work and license

This project adapts Carykh's **Evolution Simulator** by Cary Huang.

- Original source: [OpenProcessing sketch](https://openprocessing.org/@carykh/205807)
- Original license: [CC BY-SA 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/)
- Modified by: Amipo (Schnellverstellzirkel)
- Changes: Rebuilt in Rust with GPU-accelerated simulation, MAP-Elites quality-diversity search, constructive morphology mutations, a light-themed interface, configurable controls, and flat-ground physics.

The full license text is in [`LICENSE`](LICENSE).

## Run

```bash
cargo run --release
```

The default adapter is this workstation's **NVIDIA RTX 4060 Laptop GPU**. To use the integrated Radeon:

```bash
cargo run --release -- --gpu Radeon
```

Use **Evolve continuously**, **One generation**, or **Guided step**. The search evaluates a batch, inserts better specimens into the behavior archive, then breeds from diverse archive entries. Guided mode pauses at evaluation, archive update, and breeding. Space pauses/resumes evolution; Ctrl+S opens Save. The creature playback controls are independent of population evaluation. Drag the scene to pan, scroll to zoom, and select an archive card to replay it.

The first population has 1,000 creatures. Population presets include 100,000, one million, and three million. Increasing population requires **New experiment**. Populations must be even. No population is silently downsized to fit memory.

## Controls

Main controls expose population, mutation strength, and trial duration. The archive page shows the current next-batch allocation across local CMA search, constructive morphology, novelty search, and random immigrants. Random immigrants only seed an empty archive; the other three start at 35%, 35%, and 30%, and their shares then adapt to discoveries and improvements. **Advanced controls** has a search box and the physics and body settings:

| Original setting | New control / units |
| --- | --- |
| `USE_RANDOM_SEED`, `SEED` | Randomness: choose a seed automatically or supply a fixed seed; resolved seed is always saved |
| `WINDOW_SIZE` | Resizable native window and UI scale |
| `SORT_ANIMATION_SPEED` | Display: sorting transition speed |
| Minimum/maximum node size | Body bounds: diameter in meters; defaults vary from 0.06–0.12 m (original 0.4 world units = 0.08 m) |
| Minimum/maximum node friction | Body bounds: node grip, 0–1 |
| `GRAVITY` | Physics: m/s²; new experiments default to 9.8 |
| `AIR_FRICTION` | Physics: velocity retention per 1/60 second; default 1.0 (no air drag), so a runner's speed is limited by its gait rather than by drag |
| `FRICTION` | Physics: global ground/contact friction multiplier; default 1.5 |
| Default node friction | New bodies start with node grip between 0.65 and 1.0 |
| `MUTABILITY_FACTOR` | Scales continuous parameter edits; structural changes and random immigrants remain available at zero |
| `haveGround` | Physics: flat ground exists |
| Histogram range/density | Histogram minimum/maximum and bins per meter |
| Playback speed, camera zoom | Independent playback controls; mouse zoom and pan |
| Step-by-step, quick, ASAP, continuous | Guided step, One generation, Evolve continuously |
| Historical generation slider / previews | History & statistics: generation slider and worst/median/best replays |

World coordinates are meters with **positive Y upward**. New experiments run 18-second trials on flat ground with high grip. Genetics and physics changes are queued for the next generation; presentation changes are immediate. Genetic bounds apply to newly mutated offspring. Population/seed changes or limits below existing bodies require a new experiment. Use **Apply settings** after editing experiment settings.

A creature whose trial could enter the archive is tested again from a slightly perturbed pose at four times the physics rate and solver passes (240 Hz with the defaults), and keeps the worse of the two distances. Gaits that only work at the coarse physics or from one exact pose lose that way; the rest skip the second trial. A joint forced more than 0.5 rad past its range breaks, which ends the trial like a fall, so no gait can profit from muscles forcing joints round like wheels.

The physics limits allow large, fast bodies: bones up to 10 m, muscles up to 5 m long that change length at up to 24 m/s with up to 100 N, nodes up to 60 m/s, bones turning up to 40 rad/s, rhythms as short as 0.2 s, and muscles that store 120 J and recover half their missing energy per second. Under Earth gravity running speed grows with body size, so the fastest runners evolve into giants; with these limits evolved runners cover about a kilometer in a minute. `EVOLUTION_MAX_NODE_SPEED`, `EVOLUTION_MAX_MUSCLE_FORCE`, `EVOLUTION_MAX_MUSCLE_SPEED`, `EVOLUTION_MIN_MUSCLE_PERIOD`, `EVOLUTION_MAX_BONE_SPIN`, `EVOLUTION_MUSCLE_ENERGY`, `EVOLUTION_MUSCLE_RECOVERY`, `EVOLUTION_MAX_BONE_LENGTH`, and `EVOLUTION_MAX_STROKE` override the physics limits for experiments.

Archive niches use measured ground-contact fraction, center-of-mass gait cadence, mean body height (on a log scale from 15 cm up to the tallest bodies the bone limit allows, so small and giant bodies get separate cells), and feet; body shape remains visible on specimen cards but does not determine archive cells. The 1,440-cell archive stores the fastest creature in each behavior niche. Novelty uses distance to nearby archived behaviors, while local competition compares speed against elites in neighboring behavior cells. Half of CMA offspring descend from their island's fastest 1% of elites; other CMA and structural parents are locally competitive, and novelty and stalled emitters sample underexplored behaviors. Half of those top-elite offspring come from an optimizer: each island runs a separable CMA-ES on its fastest design (a body plan with a gait cadence band), searching node positions, sizes and grip, bone lengths, joint ranges and organs, the rhythm period, and every muscle setting in physical units, and ranking its samples by distance alone. It follows its own search mean instead of jumping to every new record, and restarts from the design's fastest elite once its steps shrink to nothing. When an island sets no record for 30 generations, its optimizer turns to the island's next fastest designs in turn. All optimizer samples get the perturbed fine-physics check, so they are ranked on the same terms. The four islands exchange their fastest tenth of elites only every 25 generations, so each settles on and refines its own design instead of all polishing the same one. Bodies start with 3–5 joint nodes connected by fixed-length bones and can grow. Muscles attach between bones at any normalized point, including endpoints; their forces are shared across the bone joints and rotate the linked segments. All of a body's muscles share one rhythm period and differ only in phase, so every gait repeats exactly; mutation changes the tempo as a whole. Structural emitters split bones while remapping muscle attachments, add a mirrored joint with a connected bone and motor, retime the rhythm, shift a connected oscillator group, or grow or shrink the whole body by up to half: lengths scale together and the rhythm slows with the square root of the size, as for animals of similar build, so the gait roughly carries over. Fresh morphologies are protected from replacement by a different topology for three generations. Default limits are 32 nodes/96 muscles; supported maximums are 64/256. Each skeleton is a connected tree and the motor network connects every bone. A bone can also carry an organ: a weight of 0.01–0.3 kg at any point along it that never touches the ground. Organs must sit within 0.5 m of the body's center of mass in the starting pose, measured without the head and without organs, so they stay inside the body rather than weighting limb tips. Structural mutation grows light organs or removes them, and ordinary mutation shifts their position and mass. An organ's mass is shared by its bone's two nodes in proportion to its position, so the body's center of mass is exact.

Alongside the behavior niches, a 64-entry topology reserve protects promising changed body graphs while they gather offspring trials. When the reserve has entries, ten percent of structural-emitter trials try a reserve parent. A reserve entry receives at least eight selected offspring before it can be evicted to make room for another topology; the behavior archive can absorb it earlier if a behavior elite matches or beats its distance. Fitness alone determines admission; triangles and other small bodies are not penalized. Reserve entries do not add to behavior coverage or QD score. The archive cards label them `MORPH` and report their count separately.

Archive cards show the stored elite's fitness, descriptor, emitter source, and niche visits. The archive keeps alternatives with different body plans and gaits while each niche independently improves.

## Headless experiments

```bash
cargo run --release -- headless --population 1000000 --seed 38 --generations 20 --throughput
cargo run --release -- headless --resume runs/latest.evo --generations 20 --throughput
cargo run --release -- headless --config presets/large-experiment.json --generations 10
```

`--duration` overrides the trial duration for a new experiment. `--checkpoint PATH` changes the checkpoint destination. Ctrl+C stops after the current GPU batch and saves. A completed generation includes evaluation, archive insertion, emitter feedback, and offspring creation. `--generations` counts additional generations when resuming.

The dashboard uses responsive mode, evaluating up to 8,192 creatures per GPU batch, for populations below 100,000. Larger populations automatically start in **Maximum throughput** mode, which raises the batch limit to 100,000; the checkbox can be changed afterward. Both modes adapt batch size to the configured GPU memory budget. Compute uses a separate wgpu device so long batches do not block rendering. The Rayon pool reserves two logical CPUs for the desktop.

## Save and resume

`.evo` checkpoints use a versioned header, a compact binary payload, and Zstandard compression. They contain current genomes, the MAP-Elites archive and visit counts, emitter feedback, CMA states, all settings, resolved seed, completed fitness and behavior metrics, generation stage, and historical summaries/representative creatures. Saves happen at completed batch boundaries; resume skips completed evaluations. Older checkpoints keep their current population and are reevaluated into a new archive. Temporary writes are renamed atomically after flushing.

The dashboard automatically saves every ten generations to `runs/seed-<seed>-auto.evo`; the interval is adjustable, and 0 disables it. Manual saves can preserve partial generations. Full candidate populations and archive elites are stored in checkpoints. Close the app after a requested save reports completion.

Presets are editable JSON files. CSV export contains generation, population, best/median/worst/mean archive fitness, QD score, archive cells and coverage, failed count, evaluation time, and seed. Historical histograms retain centimeter bins; available display densities divide this stored resolution. Values outside the selected histogram range are reported, not silently dropped.

## Development and measurements

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo test --release --test simulation gpu_matches_cpu_and_handles_partial_workgroups -- --ignored --nocapture
cargo run --release -- benchmark --populations 1000,100000,1000000,3000000
cargo run --release -- benchmark --populations 1000,100000 --cpu
cargo run --release -- search-benchmark --variant behavior-only --seeds 38,39,40,41,42 --population 1000 --generations 300 --duration 18 --output-dir /tmp/search-behavior-only
cargo run --release -- search-benchmark --variant morphology-reserve --seeds 38,39,40,41,42 --population 1000 --generations 300 --duration 18 --output-dir /tmp/search-morphology-reserve
cargo run --release -- analyze runs/latest.evo --output /tmp/search-analysis.json --champion /tmp/champion.json
```

`benchmark --generations N` measures successive evolving populations. CSV includes creation, GPU evaluation, optional CPU evaluation, complete generation time, population allocation, and GPU buffer allocation. GPU allocation is tracked application buffer memory, not total driver VRAM; population allocation excludes temporary evolution/checkpoint storage. Use `/usr/bin/time -v` for process peak RSS.

`search-benchmark` runs fixed-seed, headless MAP-Elites experiments and writes per-generation timing, archive, emitter, morphology, lineage, milestone, and record data. `behavior-only` disables the topology reserve for a paired baseline; `morphology-reserve` is the normal search mode. See [`docs/search-benchmark.md`](docs/search-benchmark.md) for the fixed-seed results and measurement limits.

Opt-in native UI smoke capture (closes its own window after capturing):

```bash
EVOLUTION_SMOKE_CAPTURE=/tmp/evolution.png EVOLUTION_SMOKE_POPULATION=1000000 cargo run --release
```

To measure **complete generations in the graphical app**, including evaluation, archive update, breeding, and rendering activity, run:

```bash
EVOLUTION_SMOKE_POPULATION=10000 EVOLUTION_BENCH_GENERATIONS=30 cargo run --release
```

The window starts evolution automatically, prints total generations/s and stage times, then closes. `EVOLUTION_BENCH_DURATION` changes the trial length. Populations of at least 100,000 start in throughput mode; `EVOLUTION_BENCH_THROUGHPUT=1` and `EVOLUTION_BENCH_RESPONSIVE=1` override that choice. `EVOLUTION_GPU_BATCH` overrides the batch limit and `EVOLUTION_GPU_CHUNK` overrides physics steps per dispatch for profiling. `EVOLUTION_GPU_PROFILE=1` adds GPU timestamps for shader time and the 4/5/8/16/32/64-node buckets; `EVOLUTION_PROFILE_BREED=1` reports parent planning and candidate emission times. 32-lane workgroups are the default; `EVOLUTION_WORKGROUP64=1` selects the older 64-lane kernel for comparison, and `EVOLUTION_WORKGROUP32=1` explicitly selects 32 lanes. `EVOLUTION_SHARED_DEVICE=1` restores a shared compute/render device for comparison.

See [`docs/architecture.md`](docs/architecture.md) for the execution model and [`docs/validation.md`](docs/validation.md) for measured results.
