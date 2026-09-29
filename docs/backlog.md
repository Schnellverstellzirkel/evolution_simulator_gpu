# Backlog

Open work only. Delete an item when it is merged into `main` or measured and rejected (record the numbers in `docs/search-research.md` or `docs/performance-log.md`). Delete this file when it is empty. The branch in brackets is where the item is in progress.

## Physics v2 becomes the game

- Land v2 as the game's physics: switches fixed, mud, `qd::VERSION` 30, GPU tests and thresholds, `first_generation` [claude/physics2].
- v2 CUDA kernel, mirroring `shaders/physics2_creature.wgsl` [claude/p2-cuda].
- v2 replays recorded by the scoring kernel, so a replay matches its score [claude/p2-replay].
- v2 audit tools and momentum and energy tests [claude/physdiag2, merging into claude/physics2].
- Fix v2 friction that does positive work (one evolved champion got 2.3x its muscle work from it) and the unpaid energy of light bodies that fall within 1 s [claude/p2-friction].
- A fast v2 CPU fallback (today the scalar prototype runs about 1,000 creatures/s on 2 threads), then remove the v1 kernels and CPU engine.
- A faster v2 contact solve (v2 runs about 0.27x of v1's creature-steps per second, and contacts are three quarters of its cost). Ideas are in `docs/research-2026-09-29.md`.
- Realism physics in v2: muscle force scaled to muscle size, bones that break under load, static and kinetic friction, ground contact along bones, air drag by bone length times speed squared, energy charged only for contraction work, elastic tendons. The earlier walking levers are on `claude/physics2-levers`.
- Owner question: in evolved bodies 39% have their head more than 5 cm below the highest node. Should the fall rule (head below neck) change for them?

## Interface

- Replay muscle energy and forces are estimates rebuilt from recorded positions. Record the exact values in the v2 kernel's frames once v2 is on main.
- The wet grip level is barely visible in the world view, and the effect label repeats the effect name ("Heat wave: Heat wave").
- The Islands view does not save the last migration, so a loaded game shows none until the next one.

## Speed toward 2M and 4M creatures/s

- Breed the next batch from the previous archive while the GPU works, and write children straight into the arenas. Then measure end to end at 3M on a free GPU and make an evolved 3M save [claude/speed].
- Filter contenders on the GPU, build local-mutation children on the GPU, and cap kernel registers at 128 (`docs/research-2026-09-29.md`).
- A persistent Vulkan pipeline cache. Size work units per device from measured rates. Send only snapshot changes from the worker to the UI. Keep the GUI at 60 FPS and the controls responsive at 3M.
- CUDA costs 1.2 to 1.6 GB more peak RSS (pinned buffers).

## Search

Each item is an A/B over 3 seeds. Winners go on by default and losers are deleted.

- `Creature.mutability` is mutated but unused. Self-adapted mutation scale tied over 18 seeds, so delete the gene once fewer branches are open (it touches the save format and about 32 sites).
- The emitter shares barely move: the weight formula adds a constant 0.55 to a mean reward near 0.002, so shares stay at the 35/35/30 prior. A reward-following bandit tied in the A/B.
- Archive and selection, after v2 lands and on the GPU at larger populations (CPU A/Bs at 2k to 16k creatures were ties or losses, see `docs/search-research.md` section 11): age layers (ALPS), deep grids, racing, dominated novelty search, the migration interval and which elites migrate [claude/archive].
- v2 solves only the 4 deepest contacts per step, so a fifth touching node can sink up to 1.25 cm before it joins. Revisit with the contact solve.
- A generative body grammar whose rules children inherit and mutate, tested on the GPU at 100k (a grammar used only as a seed source tied, and a varied first population lost; see `docs/search-research.md` section 12) [claude/encoding].
- Controller transfer: crossover between body plans, controller distillation, Lamarckian inheritance [claude/transfer, paused].
- Controllers, after v2 lands: an optional neural controller driven by rhythm and touchdown sensors, a rhythm controller per limb, touchdown reflexes, and a mutation that adds antagonist muscle pairs.

## World

- More environment effects and catastrophes that push toward complex, efficient movement, including water. These go into v2.
