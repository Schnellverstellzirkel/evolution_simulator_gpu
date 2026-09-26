# Evolution Simulator

A Rust game in which 2D creatures made of bones, joints, and muscles evolve to travel as far as possible. The graphical game starts with **3 million creatures per generation** and **60-second trials**. Fitness is horizontal center-of-mass distance in meters. Gait, height, and ground contact describe archive niches; they do not multiply or penalize the score.

The search combines MAP-Elites, CMA optimizers, structural mutations, novelty search, and immigrants across four island archives. Vulkan compute evaluates creatures alongside a CPU engine. An egui dashboard shows the archive, history, lineage, and CPU-recorded replays.

## Original work and license

This project adapts Carykh's **Evolution Simulator** by Cary Huang.

- Original source: [OpenProcessing sketch](https://openprocessing.org/@carykh/205807)
- Original license: [CC BY-SA 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/)
- Modified by: Amipo (Schnellverstellzirkel)
- Changes: rebuilt in Rust with GPU and CPU simulation, quality-diversity search, evolving body plans, environment effects, and an interactive dashboard.

The license text is in [LICENSE](LICENSE). The original Processing sketch is preserved in [old_code.txt](old_code.txt).

## Build and run

The runtime uses Vulkan for compute and rendering. Linux uses Wayland or X11 for the dashboard; Windows uses the native window system. Rust 1.95 or newer is required by the GUI dependencies. On Ubuntu, install the native build dependencies:

```bash
sudo apt-get install build-essential pkg-config libwayland-dev libxkbcommon-dev \
  libudev-dev libdbus-1-dev libx11-dev libxi-dev libxrandr-dev libxcursor-dev
```

Use this environment for all builds, tests, game runs, and benchmarks on the owner's workstation:

```bash
export CARGO_BUILD_JOBS=8
export RAYON_NUM_THREADS=8
export EVOLUTION_DEVICES=primary
export EVOLUTION_CPU_THREADS=6
nice -n 10 cargo run --release
```

The default compute adapter name is `RTX 4060`; `--gpu NAME` selects another primary adapter. `EVOLUTION_DEVICES=primary` prevents adding the desktop Radeon as an evaluation device. Keep that setting on this workstation: the Radeon drives the desktop and must not evaluate creatures. Evaluation and general Rayon workers share a budget of half the available logical CPUs, capped at eight. On the 16-thread workstation, the settings above give six evaluation workers and two general workers; `RAYON_NUM_THREADS` is limited to the remaining budget. `EVOLUTION_CPU_THREADS=0` makes all eight available to general workers. Smaller machines reduce evaluation workers first, preserving one general worker; a one-worker budget disables the scheduler's CPU engine.

For local iteration, use the named profile:

```bash
nice -n 10 cargo build --profile release-fast
nice -n 10 cargo run --profile release-fast
```

`release-fast` inherits release optimization, disables LTO, uses 256 codegen units, and enables incremental compilation. Its output is in `target/release-fast/`. The normal release profile retains thin LTO. The fast profile does not require a particular linker; Linux x86-64 builds already select the host CPU through [.cargo/config.toml](.cargo/config.toml). Use the normal release profile for comparable measurements.

## Playing

Use **Evolve continuously**, **One generation**, or **Guided step**. Guided mode pauses between evaluation, archive insertion, and breeding. Space pauses or resumes evolution; Ctrl+S opens Save. Select an archive card or historical creature to replay it. Playback has its own controls; drag the scene to pan and scroll to zoom.

Population and trial duration are displayed in the main controls, with no mutation slider. The default game keeps them at three million and 60 seconds. Diagnostic CLI runs and JSON presets can use other sizes or durations. Advanced controls contain seed selection, performance and checkpoint settings, display options, and histogram controls.

Environment buttons raise or lower each effect:

| Effect | Levels |
| --- | --- |
| Ground | Flat, pebbles (3 cm), rough (8 cm), rocky (15 cm), boulders (25 cm) |
| Gravity | Earth, 1.5 g, 2 g, 3 g |
| Air | Thin, breezy, thick, syrup |
| Grip | Grippy, firm, wet, ice |

A world change invalidates the old scores and queues archive elites for evaluation under the new conditions. Effects change the physics; the objective remains distance.

## Creatures and search

Bodies begin with 3–5 nodes connected by a tree of bones. Defaults allow growth to 32 nodes and 96 muscles; configuration supports up to 64 nodes and 256 muscles. Bones and muscle lengths are capped at 2 m by default. Muscles attach along bones, share a rhythm period, and provide active drive only while contracting. Their energy stores deplete with work and recover over time. Bone mass grows with length squared and is included in each engine's node masses.

Each behavior archive has 1,440 possible niches for ground contact, gait cadence, mean body height, and feet that touch down and lift off. Vertical oscillation is recorded but has one archive bin. A separate 64-entry morphology reserve gives new topologies offspring opportunities without adding to behavior coverage or QD score. Four islands retain separate parent pools and exchange their fastest tenth of elites every 25 generations. CMA, structural, and novelty emitter shares adapt to archive discoveries and improvements; immigrants seed empty archives.

Potential archive entrants receive a perturbed trial at four times the standard physics rate and solver passes. The stored fitness is the lower distance from the standard and check trials. Replays run the unperturbed creature through the CPU evaluation engine, so an archive's conservative checked score can differ from the displayed replay distance. See [architecture](docs/architecture.md) for the execution paths and current legacy-physics limitations.

## Headless experiments and diagnostics

Run these after setting the environment above:

```bash
nice -n 10 cargo run --release -- headless --population 100000 --seed 38 --generations 20 --checkpoint runs/seed-38-100k.evo
nice -n 10 cargo run --release -- headless --resume runs/seed-38-100k.evo --generations 20 --checkpoint runs/seed-38-100k.evo
nice -n 10 cargo run --release --example size_report -- runs/seed-38-100k.evo 50
EVOLUTION_LEDGER=1 nice -n 10 cargo run --release --example size_report -- runs/seed-38-100k.evo 10
```

`--generations` counts additional generations when resuming. `--config PATH` loads a JSON preset, `--duration` overrides trial duration for a new experiment, and `--checkpoint PATH` chooses the save destination. Ctrl+C requests a stop and checkpoint after the current evaluation call returns. `size_report` reports elite geometry, mass, travel, and foot slip; `EVOLUTION_LEDGER=1` adds momentum diagnostics.

```bash
nice -n 10 cargo run --release -- benchmark --populations 1000,100000 --duration 60 --generations 3
nice -n 10 cargo run --release -- analyze runs/seed-38-100k.evo --output runs/analysis.json --champion runs/champion.json
```

`benchmark --cpu` adds CPU timings. `search-benchmark` supports fixed seeds and paired search variants; see [search benchmark notes](docs/search-benchmark.md). Historical results in that document and [search research](docs/search-research.md) predate the current physics and should be remeasured before drawing conclusions about today's search.

For complete-generation timing in the graphical app:

```bash
EVOLUTION_SMOKE_POPULATION=100000 EVOLUTION_BENCH_GENERATIONS=20 nice -n 10 cargo run --release
```

This starts evolution, prints stage timings, and closes after the requested generations. `EVOLUTION_BENCH_DURATION` overrides trial length for a diagnostic run. The performance target is two million evaluated creatures per second with the graphical game at 60 FPS; this is a goal, not a measured claim.

## Save and resume

Versioned `.evo` files store the current population, evaluation progress, archives, emitter and CMA state, settings, seed, lineage, and history using a compressed binary payload. Temporary writes are flushed and renamed. Compatible older checkpoints can keep their population while obsolete archives are cleared and reevaluated; not every historical format is guaranteed to load.

The dashboard autosaves every ten generations by default to `runs/seed-<seed>-auto.evo`. It writes autosaves in a background thread and keeps the three newest experiment autosaves. Manual saves can preserve partial-generation progress. The interval is adjustable; zero disables autosave. Wait for a requested manual save to report completion before closing the app. Headless runs write to their chosen checkpoint path and also export history CSV.

## Checks

Use the workstation environment above, with serial test execution to avoid overlapping CPU evaluation pools:

```bash
nice -n 10 cargo fmt --all --check
nice -n 10 cargo clippy --locked --all-targets -- -D warnings
RUST_TEST_THREADS=1 nice -n 10 cargo test --locked --release
RUST_TEST_THREADS=1 nice -n 10 cargo test --locked --release --test simulation -- --ignored
```

GitHub Actions runs formatting, clippy, and release CPU tests on Ubuntu, with the portable CPU vector implementation. GPU tests remain ignored during CI and must be run explicitly on the workstation. No GPU agreement result follows from a passing CPU workflow.

See [architecture](docs/architecture.md), [validation](docs/validation.md), [performance log](docs/performance-log.md), and the owner's current [agent notes](AGENTS.md) for implementation details, measured results, and open work.
