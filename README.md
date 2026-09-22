# Evolution · Creature Laboratory

A local Rust evolution game for Ubuntu/Wayland. Creatures learn to walk through selection and mutation of their nodes, muscles, timing, and grip. WGSL compute shaders evaluate independent creatures on the GPU; an egui dashboard lets you watch, inspect, and tune the experiment.

The original Processing sketch is preserved in [`old_code.txt`](old_code.txt). This is a modernized simulation, not a bit-for-bit reproduction of its physics or random sequence.

## Run

```bash
cargo run --release
```

The default adapter is this workstation's **NVIDIA RTX 4060 Laptop GPU**. To use the integrated Radeon:

```bash
cargo run --release -- --gpu Radeon
```

Use **Evolve continuously**, **One generation**, or **Guided step**. Guided mode pauses after evaluation, sorting, selection, and reproduction. Space pauses/resumes evolution; Ctrl+S opens Save. The creature playback controls are independent of population evaluation. Drag the scene to pan, scroll to zoom, and select a population card or an archived representative to replay it.

The first population has 1,000 creatures. Population presets include 100,000, one million, and three million. Increasing population requires **New experiment**. Populations must be even. No population is silently downsized to fit memory.

## Controls

Main controls expose population, mutation strength, and trial duration. **Advanced controls** has a search box and all the original settings:

| Original setting | New control / units |
| --- | --- |
| `USE_RANDOM_SEED`, `SEED` | Randomness: choose a seed automatically or supply a fixed seed; resolved seed is always saved |
| `WINDOW_SIZE` | Resizable native window and UI scale |
| `SORT_ANIMATION_SPEED` | Display: sorting transition speed |
| Minimum/maximum node size | Body bounds: diameter in meters; original 0.4 world units = 0.08 m |
| Minimum/maximum node friction | Body bounds: node grip, 0–1 |
| `GRAVITY` | Physics: m/s²; default 3.6 corresponds to the original 0.005 at 60 Hz |
| `AIR_FRICTION` | Physics: velocity retention per 1/60 second; timestep independent |
| `FRICTION` | Physics: global ground/contact friction multiplier |
| `MUTABILITY_FACTOR` | Mutation strength; zero preserves offspring genetics |
| `haveGround` | Physics: ground exists |
| `RECTANGLES` | Terrain: add/remove rectangles or load flat/hurdle layouts |
| Histogram range/density | Histogram minimum/maximum and bins per meter |
| Playback speed, camera zoom | Independent playback controls; mouse zoom and pan |
| Step-by-step, quick, ASAP, continuous | Guided step, One generation, Evolve continuously |
| Historical generation slider / previews | History & statistics: generation slider and worst/median/best replays |

World coordinates are meters with **positive Y upward**. Rectangles are `[left, bottom, right, top]`. Genetics and physics changes are queued for the next generation; presentation changes are immediate. Genetic bounds apply to newly mutated offspring. Population/seed changes or limits below existing bodies require a new experiment. Use **Apply settings** after editing experiment settings.

Bodies start with 3–5 nodes and can grow. Default limits are 32 nodes/96 muscles; supported maximums are 64/256. Every body remains a connected graph. Distinct body types are identified by actual node and muscle counts, without the original modulo-10 collisions.

Population cards show a creature's current trial score when available. New offspring display their parent's previous-generation result until their own trial finishes, with the card ID making it clear that they are new specimens.

## Headless experiments

```bash
cargo run --release -- headless --population 1000000 --seed 38 --generations 20 --throughput
cargo run --release -- headless --resume runs/latest.evo --generations 20 --throughput
cargo run --release -- headless --config presets/large-experiment.json --generations 10
```

`--duration` overrides the trial duration for a new experiment. `--checkpoint PATH` changes the checkpoint destination. Ctrl+C stops after the current GPU batch and saves. A completed generation includes evaluation, ranking, selection, and reproduction. `--generations` counts additional generations when resuming.

The dashboard defaults to responsive mode, evaluating up to 8,192 creatures per GPU batch. **Maximum throughput** raises the batch limit to 65,536, uses longer dispatches, and keeps more compute work queued; it is intended for long runs. Both modes adapt batch size to the configured GPU memory budget. The Rayon pool reserves two logical CPUs for the desktop.

## Save and resume

`.evo` checkpoints use a versioned header, a compact binary payload, and Zstandard compression. They contain current genomes, all settings, resolved seed, completed fitness results, generation stage, selected parents when applicable, and historical summaries/representative creatures. Saves happen at completed batch boundaries; resume skips completed evaluations. Temporary writes are renamed atomically after flushing.

The dashboard automatically saves every ten generations to `runs/seed-<seed>-auto.evo`; the interval is adjustable, and 0 disables it. Manual saves can preserve partial generations. Full populations are stored in checkpoints, while the archive keeps statistics and three representatives per generation. Close the app after a requested save reports completion.

Presets are editable JSON files. CSV export contains generation, population, best/median/worst/mean distance, failed count, evaluation time, and seed. Historical histograms retain centimeter bins; available display densities divide this stored resolution. Values outside the selected histogram range are reported, not silently dropped.

## Development and measurements

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo test --release --test simulation gpu_matches_cpu_and_handles_partial_workgroups -- --ignored --nocapture
cargo run --release -- benchmark --populations 1000,100000,1000000,3000000
cargo run --release -- benchmark --populations 1000,100000 --cpu
```

`benchmark --generations N` measures successive evolving populations. CSV includes creation, GPU evaluation, optional CPU evaluation, complete generation time, population allocation, and GPU buffer allocation. GPU allocation is tracked application buffer memory, not total driver VRAM; population allocation excludes temporary evolution/checkpoint storage. Use `/usr/bin/time -v` for process peak RSS.

Opt-in native UI smoke capture (closes its own window after capturing):

```bash
EVOLUTION_SMOKE_CAPTURE=/tmp/evolution.png EVOLUTION_SMOKE_POPULATION=1000000 cargo run --release
```

See [`docs/architecture.md`](docs/architecture.md) for the execution model and [`docs/validation.md`](docs/validation.md) for measured results.
