# Evolution · Creature Laboratory

A Rust evolution game for Linux. 2D creatures made of bones, joints and muscles evolve to travel as far as they can in 60 seconds. The only score is horizontal distance. You shape evolution by changing the world: rougher ground, stronger gravity, thicker air, slippery ground. Every change can be undone.

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

Evaluation runs on the NVIDIA RTX 4060 through a raw Vulkan engine, on the integrated Radeon, and on an AVX-512 CPU engine at the same time. The interface renders on the GPU that drives the desktop. `EVOLUTION_DEVICES=primary` keeps evaluation off the Radeon, and `EVOLUTION_CPU_THREADS` sets how many CPU threads evaluate.

## Playing

Press **Evolve continuously** and watch. A new experiment evaluates 3 million creatures per generation with 60 second trials. **One generation** and **Guided step** advance one step at a time. Space pauses and resumes. Ctrl+S saves.

The **Overview** tab replays a creature's scored trial. Drag to pan and scroll to zoom. The **Behavior archive** tab shows the best creature for each kind of movement; click one to replay it and see its lineage. **History & statistics** shows progress over generations.

The **Environment** panel changes the world. Each effect has a button to raise its level and one to lower it:

| Effect | Levels |
| --- | --- |
| Ground | flat, pebbles 3 cm, rough 8 cm, rocky 15 cm, boulders 25 cm |
| Gravity | Earth, 1.5 g, 2 g, 3 g |
| Air | thin (no drag), breezy, thick, syrup |
| Grip | grippy, firm, wet, ice |

When the world changes, the archive's scores no longer hold. Its creatures are tested again under the new rules and compete with the new generation.

## Creatures and physics

A creature is a tree of rigid bones joined at nodes. Muscles connect pairs of bones. A muscle only pulls: it follows a rhythm and drives while its target length shortens. All muscles in a body share one tempo and differ in phase. Each muscle has an energy store that drains with the work it does and recovers over time, and an exhausted muscle cannot drive. Joints have evolved ranges, and a joint forced far past its range breaks and ends the trial. A heavy head must stay above its neck, or the creature falls and its score freezes. Touchdown sensors can restart a muscle's rhythm when a foot lands.

Bones are at most 2 m long and weigh 4 kg per square meter of length (a 1 m bone weighs 4 kg), so larger bodies are heavier. Muscles pull with at most 100 N. Physics runs at 60 Hz with 2 bone passes and 1 velocity pass. Positions are solved first and velocities come from the actual movement. Ground friction resists sliding in proportion to the load a foot carries.

Physics is implemented three times and the implementations must agree: the GPU kernel (`shaders/physics_creature.wgsl`), the CPU engine (`src/cpu_engine.rs`), and an older CPU reference (`src/physics.rs`). The replay shows frames recorded by the CPU engine, so it shows the scored motion.

## Evolution

The search is MAP-Elites. The archive keeps the fastest creature in each behavior cell: 6 ground-contact bins, 8 cadence bins, 6 body-height bins on a log scale, and 5 bins for the number of feet, 1,440 cells in all. A 64-entry morphology reserve protects new body plans while they gather offspring. Four island archives evolve apart and exchange elites every 25 generations.

Offspring come from four emitters. CMA emitters make local steps around elites, and each island runs a CMA-ES optimizer on its fastest design. Structural emitters change the body: split a bone, duplicate a limb, resize the whole body. Novelty emitters sample underexplored behaviors. Random immigrants only seed an empty archive.

A creature that could enter the archive is tested again from a slightly perturbed pose at 4x the physics rate and solver passes, and keeps the worse distance. Gaits that only work because of coarse time steps do not survive.

## Save and resume

`.evo` checkpoints hold the population, the archives, emitter state, settings, the seed, and history. They are versioned and Zstandard-compressed, and writes are atomic. The game autosaves every 10 generations to `runs/seed-<seed>-auto.evo` on a background thread and keeps the autosaves of the three most recent experiments. Files you save yourself are never removed. Breaking changes to physics or the archive rebuild the archive from an older checkpoint.

## Headless use

```bash
cargo run --release -- headless --population 100000 --seed 38 --generations 20 --checkpoint runs/test.evo
cargo run --release -- headless --resume runs/test.evo --generations 20
cargo run --release -- analyze runs/test.evo --output /tmp/analysis.json --champion /tmp/champion.json
cargo run --release -- benchmark --populations 1000,100000,1000000
cargo run --release -- search-benchmark --help
cargo run --release -- eval-bench --help
cargo run --release --example size_report runs/test.evo 12
```

Ctrl+C in headless mode saves after the current batch. `size_report` prints body length, mass and foot slip for the fastest elites, and with `EVOLUTION_LEDGER=1` where their forward momentum comes from.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo test --release --test simulation -- --ignored   # GPU agreement tests
```

`AGENTS.md` has the working rules for this machine and the open work. `docs/architecture.md` describes the execution model, `docs/search-research.md` the measurements behind the search design, and `docs/performance-log.md` the performance history.

In-app benchmark: `EVOLUTION_SMOKE_POPULATION=10000 EVOLUTION_BENCH_GENERATIONS=30 cargo run --release` starts evolution, prints generations per second and frame times, and closes. `EVOLUTION_SMOKE_CAPTURE=/tmp/shot.png` saves a screenshot instead.
