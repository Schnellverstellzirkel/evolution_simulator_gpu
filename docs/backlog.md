# Backlog

Open work only. Delete an item when it is merged into `main` or measured and rejected (record the numbers in `docs/search-research.md` or `docs/performance-log.md`). Delete this file when it is empty. The branch in brackets is where the item is in progress.

## Physics v2

- If the game loses its GPU (device lost), it falls back to the CPU (3,700 to 12,000 creatures/s at 8 threads) for the rest of the session [claude/gpu-recover]. Reopen the GPU after a device loss instead, or make the CPU fallback fast. Agent benchmarks beside the owner's game have caused device-lost errors in the benchmark process.
- The walking levers (tendons, joint damping) are on `claude/physics2-levers` and need a port onto v2 and one GPU evolution each.
- v2 runs at about 0.2x of v1's creature-steps per second; CUDA is 1.8x Vulkan. Occupancy is 4 warps at 16 nodes [claude/p2-cuda-speed].
- Remove the v1 kernels and CPU engine now that the v2 CPU engine is fast (3,700 to 12,000 creatures/s at 8 threads, bit-equal to the prototype) [claude/p2-cpu].
- A faster v2 contact solve (contacts are about three quarters of v2's cost). Ideas are in `docs/research-2026-09-29.md`.
- Realism physics in v2: bones that break under load, static and kinetic friction, ground contact along bones, elastic tendons. The earlier walking levers are on `claude/physics2-levers`.
- Owner question: in evolved bodies 39% have their head more than 5 cm below the highest node. Should the fall rule (head below neck) change for them?

## Interface

- The wet grip level is barely visible in the world view, and the effect label repeats the effect name ("Heat wave: Heat wave").

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
- Controllers: an optional neural controller driven by rhythm and touchdown sensors, a rhythm controller per limb (today `repair` copies the first muscle's period to every muscle, so the body shares one clock), and touchdown reflexes.

## World

- More environment effects and catastrophes that push toward complex, efficient movement, including water. These go into v2.
