# Backlog

Open work only. Delete an item when it is merged into `main`. If it is a real dead end, add one or two lines to `docs/rejected-ideas.md`. Delete this file when it is empty. The branch in brackets is where the item is in progress.

## Physics

- Hundreds of different bodies tie at one top distance in an evolved game (46.585 m in 20 s at generation 590 of the owner's game, 34.07 m at generation 1,512, each island its own value to 1e-5). The 400 fastest elites replay to distances within 2 mm of each other, and the five fastest of one island travel at the same constant speed after the first 2 s although their bodies, cells and muscle periods (0.2 to 0.47 s) differ, so the search has nothing to select on at the top. Find what limits them. Muscle strength and energy scale with the mass a muscle drives, so one power per mass would give one speed, but nothing has measured it.

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

- Two runs of one seed diverge by generation 2 to 3 on the shared GPU: `search_ab --probe` at 40k creatures printed different archive and ring digests for two runs of the code of c54876a, of 09d1f73 and of `claude/cells`, while its batch probe (`PROBE_GPU=1`) found no differing bit between whole and chunked scoring on c54876a. At 1M creatures two builds (the old archive and this branch's unrefined one) gave the same archive to every digit on seeds 41 and 42 over 40 generations, so the divergence may need a small population. The owner's rule is one search per seed on one GPU. Find where the runs part.

- The emitter shares barely move: the weight formula adds a constant 0.55 to a mean reward near 0.002, so shares stay at the 35/35/30 prior. A reward-following bandit tied.
- Archive and selection, on the GPU at larger populations (small CPU runs were ties or losses, see `docs/rejected-ideas.md`): age layers beyond the two nurseries (ALPS), deep grids, racing, dominated novelty search, the migration interval and which elites migrate [claude/archive].
- The nurseries were measured at 300k creatures a generation from the owner's save. At 3M each reshaped body gets about five times the children it gets at 300k, so measure the final split of the slots there (5% and 10%), the end-to-end rate on the owner's save (the 5% and 5% design with a coarse reshaped nursery ran 217k against 236k for main on power saver), and whether a larger reshaped share or a longer run still adds plans.
- The island plans stop at about 15,000 because an island has 5,000 cells. A finer class layout for the islands (3 by 3 classes gave 4,894 plans in an island against 2,530, in 9,400 cells) or a deep grid that keeps two or three elites of different plans in a cell would raise the plans the islands breed from. The save limits it: 152 MB for the owner's save with the global archive at 4 by 4 classes, 103 MB before, and 3 by 3 classes in every archive was 170 MB. A global elite takes 3.3 KB, and a history row 20 KB (three representative creatures and the list of body types), so a thinner history or a global archive that stores each creature once would buy room.
- Eight isolated islands instead of four were not tried: each island keeps its own lineages, so the clades of the global archive would grow, at the cost of a save with twice the island archives.
- A save keeps the ancestors of the global archive's elites and each island's fastest 10, so after a load the elites of an island that no chain reaches start as clades of their own, and the rarity bonus counts their descendants from there. A clade id per saved elite would keep the clades through a load.
- A world change tests an island's elites again as new bodies, so a graduate of a nursery loses its mark (`Elite::graduate`) and the island's count of graduate cells restarts.
- The solve keeps only the 4 deepest contacts per step, so a fifth touching node can sink up to 1.25 cm before it joins. Revisit with the contact solve.
- A generative body grammar whose rules children inherit and mutate, tested on the GPU at 100k (a grammar used only as a seed source tied, and a varied first population lost; see `docs/rejected-ideas.md`) [claude/encoding]. `segment_chain`, `limb_length_gradient` and `mirrored_limb_pair` already apply such productions to the body and the child inherits the product. A rule set stored with the creature would be a new part of every save and archive.
- A mutation strength that evolves with each lineage (a step size carried by the creature, Beyer and Schwefel 2002). It needs a new field in the genome and in every save, so it waits for a save version bump.
- The 0.035 parameter mutation after a structural operator lowers how often the child enters the archive in `mutation_audit`: for the 66 older operators 18.6% alone against 2.4% with it on the generation-590 save and 16.7% against 7.1% on the generation-1510 save (the distance added per 1,000 children falls 55% and 21%). On young saves (generation 20, before the archives refined) the entries fell 4 to 5 points and the distance added rose 3 to 7%. A compound child already gets none. Not tested in breeding: drop it for every child an operator changed and compare paired resumes of a late and a young save.
- Controllers: an optional neural controller driven by rhythm and touchdown sensors, a rhythm controller per limb (today `repair` copies the first muscle's period to every muscle, so the body shares one clock), and touchdown reflexes.

## World

- More environment effects and catastrophes (Water and Ice patches landed; ideas: wind gusts, low ceiling, moving ground).
