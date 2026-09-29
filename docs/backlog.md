# Backlog

Open work only. Delete an item when it is merged into `main` or measured and rejected (record the numbers in `docs/search-research.md` or `docs/performance-log.md`). Delete this file when it is empty. The branch in brackets is where the item is in progress.

## Physics v2 becomes the game

- Land v2 as the game's physics: switches fixed, mud, `qd::VERSION` 30, GPU tests and thresholds, `first_generation` [claude/physics2].
- v2 CUDA kernel, mirroring `shaders/physics2_creature.wgsl` [claude/p2-cuda].
- v2 replays recorded by the scoring kernel, so a replay matches its score [claude/p2-replay].
- Audit v2 for solver-made energy and momentum, the planted-feet rule, jitter, and where a triangle's motion comes from. Add momentum and energy tests [claude/physdiag2].
- A fast v2 CPU fallback (today the scalar prototype runs about 1,000 creatures/s on 2 threads), then remove the v1 kernels and CPU engine.
- A faster v2 contact solve (v2 runs about 0.27x of v1's creature-steps per second, and contacts are three quarters of its cost). Ideas are in `docs/research-2026-09-29.md`.
- Realism physics in v2: muscle force scaled to muscle size, bones that break under load, static and kinetic friction, ground contact along bones, air drag by bone length times speed squared, energy charged only for contraction work, elastic tendons. The earlier walking levers are on `claude/physics2-levers`.
- Owner question: in evolved bodies 39% have their head more than 5 cm below the highest node. Should the fall rule (head below neck) change for them?

## Interface

- Evolution schematic in a Dofus style, opened from Help and from the Islands view [claude/schematic].
- Muscle energy and a force overlay in the replay [claude/camera].
- Suggest an environment effect when the archive stalls. Fix the Seasons button (the owner saw Slow go back to Off). A visual effect in the world view for every environment effect [claude/world-ui].
- Cost of transport in the playback tooltip and `size_report` [claude/physdiag].
- The Islands view does not save the last migration, so a loaded game shows none until the next one.

## Speed toward 2M and 4M creatures/s

- Overlap archive insertion and breeding with GPU work, and parallelize them. Then measure an evolved 3M save [claude/speed].
- Filter contenders on the GPU, build local-mutation children on the GPU, and cap kernel registers at 128 (`docs/research-2026-09-29.md`).
- A persistent Vulkan pipeline cache. Size work units per device from measured rates. Send only snapshot changes from the worker to the UI. Keep the GUI at 60 FPS and the controls responsive at 3M.
- CUDA costs 1.2 to 1.6 GB more peak RSS (pinned buffers).

## Search

Each item is an A/B over 3 seeds. Winners go on by default and losers are deleted.

- Emitters: bandit emitter shares, line variation between same-plan elites, CMA-MAE thresholds, discrete crossover, self-adapted step sizes (use or remove `Creature.mutability`), CMA over body bounds, joint ranges, sensors and reset phases [claude/emitters].
- Archive and selection: fix the morphology reserve (draw parents from the global reserve and count their visits), a finer archive, body size or limb count as an axis, descriptor review, island model tuning, periodic island extinctions, age layers (ALPS), deep grids, racing, dominated novelty search, generalized early stopping [claude/archive].
- Body encoding: a more varied first population, repeated and mirrored limbs, a generative body grammar [claude/encoding, paused].
- Controller transfer: crossover between body plans, controller distillation, Lamarckian inheritance [claude/transfer, paused].
- Controllers, after v2 lands: an optional neural controller driven by rhythm and touchdown sensors, a rhythm controller per limb, touchdown reflexes, and a mutation that adds antagonist muscle pairs.

## World

- More environment effects and catastrophes that push toward complex, efficient movement, including water. These go into v2.

## Code health

- Delete or finish every experiment switch (`EVOLUTION_NEUTRAL_SPLITS`, `EVOLUTION_ELITE_REFRESH`, `EVOLUTION_SHRINK`, `EVOLUTION_EARLY_EXIT`, `EVOLUTION_CHECK_TERRAIN`), list the remaining diagnostics in `docs/building.md`, and check that the GitHub Actions CI runs [claude/cleanup].
