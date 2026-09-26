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

If the primary GPU cannot open, evaluation falls back to the CPU and reports why once; with `EVOLUTION_CPU_THREADS=0` that fallback shares the general Rayon pool. A GPU that fails during a run is retired and its unfinished units, including pending fine checks, are retried on the CPU with the same creatures and settings. A failed CPU stops the session with a persistent error after completed results are stored.

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
| Grip | Sandpaper, grippy, firm, wet, ice |
| Heat wave | Full, warm, hot, heat wave |
| Drought | Normal, dry, parched, drought |
| Slope | Flat, 3%, 8%, 15%, 25% uphill |
| Wind | Calm, breeze, strong, gale (headwind) |
| Mud | Dry, damp, muddy, deep mud |
| Gaps | Solid, narrow, wide, chasms |
| Hurdles | Clear, low, high, walls |
| Earthquake | Still, tremors, quakes, big one |

Hurdles raise periodic steps that a gait must climb or leap. The earthquake gives every creature its own bump phase and height, derived from its id, so no gait can memorize one pattern. A world change invalidates the old scores and queues archive elites for evaluation under the new conditions. Effects change the physics; the objective remains distance.

The **Catastrophe** row adds **Meteor strike**, which removes about half the elites at random from each archive, and **Extinction**, which clears the island with the slowest best creature. **Undo** restores saved fossils where their cells are empty or hold slower elites. Fossils are kept in memory for the current session; catastrophe undo history is not saved in checkpoints.

## Creatures and search

Bodies begin with 3–5 nodes connected by a tree of bones. Defaults allow growth to 32 nodes and 96 muscles; configuration supports up to 64 nodes and 256 muscles. Bones and muscle lengths are capped at 2 m by default. Muscles attach along bones, share a rhythm period, and provide active drive only while contracting. Their energy stores deplete with work and recover over time. Bone mass grows with length squared and is included in each engine's node masses. Touchdown sensors can reset muscle rhythms when a foot lands.

Grounded nodes resist movement during bone and velocity constraints, so the body can pivot over planted feet. A friction cap limits the center-of-mass displacement this can produce, and ground support includes floor clamps and whole-body lift. A fall, a joint driven too far past its range, or head acceleration averaged over about 0.1 seconds exceeding 8 g ends scoring at the distance reached. These physical limits leave distance as the sole objective.

Each behavior archive has 1,440 possible niches for ground contact, gait cadence, mean body height, and feet that touch down and lift off. Vertical oscillation is recorded but has one archive bin. A separate 64-entry morphology reserve gives new topologies offspring opportunities without adding to behavior coverage or QD score. Four islands retain separate parent pools and exchange their fastest tenth of elites every 25 generations. CMA, structural, and novelty emitter shares adapt to archive discoveries and improvements; immigrants seed empty archives.

Potential archive entrants receive a perturbed trial at four times the standard physics rate and solver passes. Before admission to the global archive shown in the dashboard, the best candidates for behavior cells and new body plans also repeat the standard trial on the CPU engine. Their score keeps the lowest distance, and their global archive cell uses the CPU behavior.

Replays return the recorded frames and engine result together through `cpu_engine::replay`. The viewport displays that recording's distance and terminal event; the conservative archive score can be lower. See [architecture](docs/architecture.md) for the execution paths and remaining legacy-physics limitations.

## Headless experiments and diagnostics

Run these after setting the environment above:

```bash
nice -n 10 cargo run --release -- headless --population 100000 --seed 38 --generations 20 --checkpoint runs/seed-38-100k.evo
nice -n 10 cargo run --release -- headless --resume runs/seed-38-100k.evo --generations 20 --checkpoint runs/seed-38-100k.evo
nice -n 10 cargo run --release --example size_report -- runs/seed-38-100k.evo 50
EVOLUTION_LEDGER=1 nice -n 10 cargo run --release --example size_report -- runs/seed-38-100k.evo 10
nice -n 10 cargo run --release --example search_ab -- 2 64 0.5 38,39 --tag baseline
```

`--generations` counts additional generations when resuming. `--config PATH` loads a JSON preset, `--duration` overrides trial duration for a new experiment, and `--checkpoint PATH` chooses the save destination. Ctrl+C requests a stop and checkpoint after the current evaluation call returns. `size_report` reports elite geometry, mass, travel, and foot slip; `EVOLUTION_LEDGER=1` adds momentum diagnostics. `search_ab` runs fixed-seed CPU-only A/B generations and prints best distance, QD score, archive cells, and the top-50 body mix; see [search research](docs/search-research.md).

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

Versioned `.evo` files store the current population, evaluation progress, archives, emitter and CMA state, settings, seed, lineage, and history using a compressed binary payload. Temporary writes are flushed and renamed. V4 checkpoints also retain island optimizer progress so continuation preserves its stall history; V3 files remain readable. Compatible older checkpoints can keep their population while obsolete archives are cleared and reevaluated. Current physics uses QD version 19, so archives from the earlier version-16 baseline are invalidated on load; not every historical format is guaranteed to load.

The dashboard autosaves every ten generations by default to `runs/seed-<seed>-auto.evo`. It writes autosaves in a background thread and keeps the three newest experiment autosaves. Manual saves can preserve partial-generation progress. The interval is adjustable; zero disables autosave. Wait for a requested manual save to report completion before closing the app. Headless runs write to their chosen checkpoint path and also export history CSV.

## Checks

Use the workstation environment above, with serial test execution to avoid overlapping CPU evaluation pools:

```bash
nice -n 10 cargo fmt --all --check
nice -n 10 cargo clippy --locked --all-targets -- -D warnings
RUST_TEST_THREADS=1 nice -n 10 cargo test --locked --release
RUST_TEST_THREADS=1 nice -n 10 cargo test --locked --release --test simulation -- --ignored
```

GitHub Actions runs formatting, clippy, and release CPU tests on Ubuntu, with the portable CPU vector implementation. GPU tests remain ignored during CI and must be run explicitly on the workstation. No GPU agreement result follows from a passing CPU workflow. Physics changes also require the random-population propulsion diagnostic, `nice -n 10 cargo run --release --example first_generation`, with the same workstation environment. This is a diagnostic against solver-created propulsion, not a proof of energy conservation.

See [architecture](docs/architecture.md), [validation](docs/validation.md), [performance log](docs/performance-log.md), and the owner's current [agent notes](AGENTS.md) for implementation details, measured results, and open work.
