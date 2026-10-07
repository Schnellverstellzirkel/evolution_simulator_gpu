# Evolution Simulator

A Rust game in which 2D creatures made of bones, joints, and muscles evolve to travel as far as possible. The game window calls it exploraMove. The graphical game starts with **3 million creatures per generation** and **20-second trials**. Fitness is horizontal center-of-mass distance in meters. Gait, height, and ground contact describe archive niches; they do not multiply or penalize the score.

The search combines MAP-Elites, CMA optimizers, structural mutations, novelty search, and immigrants across four isolated island archives, a hub, and 100 wild islands that each run in a world of their own. A CUDA kernel on an NVIDIA GPU simulates and scores every creature and records the replays. An egui dashboard shows the archive, history, lineage, and replays.

## Original work and license

This project adapts Carykh's **Evolution Simulator** by Cary Huang.

- Original source: [OpenProcessing sketch](https://openprocessing.org/@carykh/205807)
- Original license: [CC BY-SA 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/)
- Modified by: Amipo (Schnellverstellzirkel)
- Changes: rebuilt in Rust with GPU simulation, quality-diversity search, evolving body plans, environment effects, and an interactive dashboard.

The license text is in [LICENSE](LICENSE). The original Processing sketch is preserved in [old_code.txt](old_code.txt).

## Build and run

The game needs an NVIDIA GPU with the CUDA driver and NVRTC: creatures are simulated on CUDA only, and the game stops with an error without them. The CUDA driver library comes with the NVIDIA driver, and NVRTC with a CUDA toolkit or NVIDIA's pip wheel ([building](docs/building.md)). The window is drawn through Vulkan. Linux uses Wayland or X11 for the dashboard; Windows uses the native window system. Rust 1.95 or newer is required by the GUI dependencies. On Ubuntu, install the native build dependencies:

```bash
sudo apt-get install build-essential pkg-config libwayland-dev libxkbcommon-dev \
  libudev-dev libdbus-1-dev libx11-dev libxi-dev libxrandr-dev libxcursor-dev
```

Use this environment for all builds, tests, game runs, and benchmarks on the owner's workstation:

```bash
export EVOLUTION_DEVICES=primary
cargo run --release
```

The default GPU name is `RTX 4060`; `--gpu NAME` selects another NVIDIA GPU. Secondary GPUs are off by default; `EVOLUTION_DEVICES=NAME` adds other NVIDIA GPUs to evaluation, and `EVOLUTION_DEVICES=primary` keeps them off. Keep that setting on this workstation, because the Radeon 780M draws the desktop and the game window and never evaluates creatures ([building](docs/building.md#the-radeon-780m) has the reasons). General Rayon workers (archive insertion, breeding, packing) take every logical CPU but two, at low priority; `RAYON_NUM_THREADS` can lower that.

If the primary GPU cannot open, the game stops and says why. A GPU that fails during a run is reopened and its unfinished units, including pending confirmation trials, run again with the same creatures and settings. A GPU that does not reopen stops the session with a persistent error after completed results are stored.

For local iteration, use the named profile:

```bash
nice -n 10 cargo build --profile release-fast
nice -n 10 cargo run --profile release-fast
```

`release-fast` inherits release optimization, disables LTO, uses 256 codegen units, and enables incremental compilation. Its output is in `target/release-fast/`. The normal release profile retains thin LTO. The fast profile does not require a particular linker; Linux x86-64 builds already select the host CPU through [.cargo/config.toml](.cargo/config.toml). Use the normal release profile for comparable measurements.

## Playing

Press **Evolve** in the top bar, or Space, to run generation after generation; Space or **Pause evolution** stops. The replay follows the champion: the best creature so far. When a new record makes a new champion, the view switches to it at once, mid-generation too, on the Overview and in the player beside Ways of moving. Pick any creature (an archive card, a map cell, a record, an ancestor) to watch it instead, and **Back to champion** returns. K or a click on the replay pauses it, the arrow keys step one frame, drag pans and scroll zooms. Ctrl+S opens Save, F1 opens help.

The top bar shows the population and trial length; the game keeps them at three million and 20 seconds, with no mutation controls. Diagnostic CLI runs and JSON presets can use other sizes or durations. The File menu opens, saves and exports; the New experiment dialog takes a seed; the View menu holds the UI scale. **Diagnostics** in the status line opens a drawer with search and machine numbers and the **One generation** button, which runs one generation and pauses.

Each environment effect is a row with one button per level. A click sets that level, and **Calm world** resets them all:

| Effect | Levels |
| --- | --- |
| Autochange environment | Off, slow, normal, fast: raises one effect by one level every 100, 50 or 20 generations, most benign first, and keeps them |
| Ground | Flat, pebbles (3 cm), rough (8 cm), rocky (15 cm), boulders (25 cm) |
| Gravity | Earth, 1.5 g, 2 g, 3 g |
| Air | Thin, breezy, thick, syrup |
| Grip | Sandpaper, grippy, firm, wet, ice |
| Heat wave | Full, warm, hot, heat wave |
| Drought | Normal, dry, parched, drought |
| Slope | Flat, 3%, 8%, 15%, 25% uphill |
| Wind | Calm, breeze, strong, gale (headwind) |
| Mud | Dry, damp, muddy, deep mud |
| Brambles | Clear, scrub, brambles, thicket: every part that is not a foot is held back while it touches the ground |
| Water | Dry, puddles (12 cm), shallows (35 cm), deep water (90 cm): parts below the waterline are buoyed, and bones push against a thick medium |
| Ice patches | None, frost, ice, black ice: slick bands every 6 m that take away 50%, 80% or 95% of the grip |
| Gaps | Solid, narrow, wide, chasms |
| Hurdles | Clear, low, high, walls |
| Earthquake | Still, tremors, quakes, big one |

Hurdles raise periodic steps that a gait must climb or leap. The earthquake gives every creature its own bump phase and height, derived from its id, so no gait can memorize one pattern. The autochange makes the world harsher on a schedule: every 100, 50, or 20 generations, exactly one effect advances one level. It starts with the most benign effects, air, wind and ground, takes each effect to its first level above calm, and then goes round again with harsher levels, ending with hurdles, gaps and the earthquake. It never lowers an effect, and the world stays as it is after the last step. The step is saved with the experiment, so a resumed run continues mid-cycle, and Off is the default. A world change invalidates the old scores of the main islands and queues their elites for a new test under the new conditions. The wild islands keep their own worlds. Effects change the physics; the objective remains distance.

The **Presets** buttons (Rough hills, Icy slope, Swamp, Desert, Obstacle course, Heavy world and Savanna) set several effects at once and calm the rest. **Undo last change** returns to the world before your last change in the panel.

The **Catastrophes** section has **Meteor strike**, which removes about half the elites at random from each archive and spares the fastest elite of each of an archive's 64 fastest body plans, and **Extinction**, which clears the island with the slowest best creature. An emptied isolated island restarts from new random bodies, and the hub refills from the copies it receives. **Undo** restores saved fossils where their cells are empty or hold slower elites. Fossils are kept in memory for the current session; catastrophe undo history is not saved in checkpoints.

## Creatures and search

Bodies begin with 3 to 5 nodes connected by a tree of bones. Bodies grow to 32 nodes and 96 muscles by default, and bones and muscles are at most 2 m long. Muscles attach along bones, run on one body clock or on a simple multiple of it, and pull only while contracting. Their energy stores deplete with work and recover over time. Bone mass grows with length squared. Touchdown sensors can reset muscle rhythms when a foot lands. The physics is position-based dynamics: each 1/60 s step is 16 substeps in which constraints return every bone to its length and every joint inside its range, and no step adds energy. See [physics](docs/physics.md).

A fall, a joint driven too far past its range, or head acceleration above 8 g ends scoring at the distance reached. Distance is the only objective.

Each island's behavior archive starts with 1,440 ways of moving (ground contact, gait cadence, body height and feet that touch down and lift off). After 30 generations it is refined to 12,960 cells: each way of moving gets a cell for 9 body classes, 3 shapes by the start pose's width over height and 3 sizes by node count, so a long large body and a compact small one no longer compete for a cell. The four isolated islands refine 10 generations apart. A refined island prefers a parent of a rare lineage when the distances of its elites are level, so a lineage that the best one has not taken over keeps breeding.

Four islands are isolated: their children take parents and mates only from their own archive. Only two small exchanges cross between them: every 50 generations each island sends the fastest elite of its rarest body plan to the next one, and a bank of the first elite of each new body plan breeds 1% of the main islands' slots. Every 25 generations a fifth island, the hub, receives copies of each isolated island's fastest tenth of elites and breeds from them with its own. Nothing flows back. 100 wild islands each run in a fixed world of their own, a mix of one to three environment effects that the buttons leave alone. Every 25 generations each wild island sends its fastest tenth to the hub, where the creatures run once in the hub's world and then breed for 30 generations. The global archive records the main islands' elites for display and saves, and no parent comes from it. It has finer body classes than the islands, 4 shapes by 4 sizes for each way of moving (23,040 cells), from the first generation.

Each island keeps a 64-entry reserve of new body plans. About 60% of each generation carries a structural mutation, 30% only changes numbers and 10% are new random bodies. Each island also keeps two protected archives. A nursery of new random bodies lets them compete only against each other for 10 generations before the survivors enter the island archive. A nursery of reshaped bodies keeps the new body plans that the island turned away and tunes them until they beat its elites.

A standard trial stops at 5 s when the creature is below the bar, the 5 s distance the top 10% reached. A screened creature enters no archive. A creature that would take a cell of an archive, when it has reached at least half of that archive's best distance, or that would tie its record, also gets a confirmation trial from the same pose at twice the physics rate, and its fitness is the lower distance. Both are GPU evaluations and the GPU score is final.

Replays are recorded by the GPU that scores the archive (`engine::replay`): the scoring kernel with a frame output, so the replay shows the trial and the distance the archive holds. See [architecture](docs/architecture.md) and [design decisions](docs/design-decisions.md).

## Headless experiments and diagnostics

Run these after setting the environment above:

```bash
nice -n 10 cargo run --release -- headless --population 100000 --seed 38 --generations 20 --checkpoint runs/seed-38-100k.evo
nice -n 10 cargo run --release -- headless --resume runs/seed-38-100k.evo --generations 20 --checkpoint runs/seed-38-100k.evo
nice -n 10 cargo run --release --example size_report -- runs/seed-38-100k.evo 50
nice -n 10 cargo run --release --example size_report -- runs/seed-38-100k.evo 10
nice -n 10 cargo run --release --example search_ab -- 2 64 0.5 38,39 --tag baseline
```

`--generations` counts additional generations when resuming. `--config PATH` loads a JSON preset, `--duration` overrides trial duration for a new experiment, and `--checkpoint PATH` chooses the save destination. Ctrl+C requests a stop and checkpoint after the current evaluation call returns. `size_report` reports elite geometry, mass, travel, and foot slip from GPU replays. `search_ab` runs fixed-seed generations through the production archive and breeding path and prints best distance, QD score, archive cells, and the top-50 body mix.

For complete-generation timing in the graphical app:

```bash
EVOLUTION_SMOKE_POPULATION=100000 EVOLUTION_BENCH_GENERATIONS=20 nice -n 10 cargo run --release
```

This starts evolution, prints stage timings, and closes after the requested generations. `EVOLUTION_BENCH_DURATION` overrides trial length for a diagnostic run. The performance goal is 500,000 evaluated creatures per second, sustained, with the graphical game at 60 FPS. It is a goal, not a measured claim.

## Save and resume

A save holds the settings, generation, history, archives, emitter and CMA state and lineage. It holds no creatures in flight, so it is small, and loading breeds them again from the archives. Each save starts with a header that carries the physics version, so an older save is turned down with a message before it loads. A save of version 53 or newer loads, with its elites placed in the new cells.

The dashboard writes no files on its own: autosave is off by default, and a loaded game starts with it off. When the player turns on File > Autosave every 10 generations, autosaves go to `runs/seed-<seed>-auto.evo` in a background thread, and the three newest experiment autosaves are kept. Wait for a requested manual save to report completion before closing the app. Headless runs write to their chosen checkpoint path and also export history CSV.

## Checks

Use the workstation environment above:

```bash
nice -n 10 cargo fmt --all --check
nice -n 10 cargo clippy --locked --all-targets -- -D warnings
```

See [architecture](docs/architecture.md), [physics](docs/physics.md), [anatomy operators](docs/anatomy-operators.md), [building](docs/building.md), [design decisions](docs/design-decisions.md), [rejected ideas](docs/rejected-ideas.md), and the owner's [agent notes](AGENTS.md). Open work is in [backlog](docs/backlog.md).
