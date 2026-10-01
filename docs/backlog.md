# Backlog

Open work only. Delete an item when it is merged into `main`. If it is a real dead end, add one or two lines to `docs/rejected-ideas.md`. Delete this file when it is empty. The branch in brackets is where the item is in progress.

## Physics

- CUDA occupancy is 4 warps at 16 nodes [claude/p2-cuda-speed].
- A faster contact solve. Contacts are about 55% of a step, and the cheap changes are listed in `docs/rejected-ideas.md`. A different solver (fewer, larger operations per step) is what is left.
- Realism physics: ground contact along bones. Bones that break under load are deferred until the contact solve is faster (the exact load costs an estimated 20 to 30% of kernel time; the CPU diagnostic is on claude/p2-bone-break).
- Owner question: in evolved bodies 39% have their head more than 5 cm below the highest node. Should the fall rule (head below neck) change for them?

## Interface


## Speed toward 500k creatures/s sustained

- Breed the next batch from the previous archive while the GPU works, and write children straight into the arenas. Then measure end to end at 3M on a free GPU and make an evolved 3M save [claude/speed].
- Build local-mutation children on the GPU. A device-side contender filter has a low ceiling (`docs/rejected-ideas.md`).
- Size work units per device from measured rates. Send only snapshot changes from the worker to the UI. Keep the GUI at 60 FPS and the controls responsive at 3M.
- CUDA costs 1.2 to 1.6 GB more peak RSS (pinned buffers).

## Search

Winners go on by default and losers are deleted.

- The emitter shares barely move: the weight formula adds a constant 0.55 to a mean reward near 0.002, so shares stay at the 35/35/30 prior. A reward-following bandit tied.
- Archive and selection, on the GPU at larger populations (small CPU runs were ties or losses, see `docs/rejected-ideas.md`): age layers (ALPS), deep grids, racing, dominated novelty search, the migration interval and which elites migrate [claude/archive].
- The solve keeps only the 4 deepest contacts per step, so a fifth touching node can sink up to 1.25 cm before it joins. Revisit with the contact solve.
- A generative body grammar whose rules children inherit and mutate, tested on the GPU at 100k (a grammar used only as a seed source tied, and a varied first population lost; see `docs/rejected-ideas.md`) [claude/encoding].
- Controllers: an optional neural controller driven by rhythm and touchdown sensors, a rhythm controller per limb (today `repair` copies the first muscle's period to every muscle, so the body shares one clock), and touchdown reflexes.

## World

- More environment effects and catastrophes (Water and Ice patches landed; ideas: wind gusts, low ceiling, moving ground).
