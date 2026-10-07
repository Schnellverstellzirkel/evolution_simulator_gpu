# Backlog

Open work only. Delete an item when it is merged into `main`. If it is a real dead end, add one or two lines to `docs/rejected-ideas.md`. Delete this file when it is empty. The branch in brackets is where the item is in progress.

## Physics

- `tests/cuda_effects.rs` (ignored, needs the GPU) fails in the calm world: its stored walkers no longer pass 5 m under the current kernel. Replace them with walkers from a current save, and add Brambles to its effect list.

- Realism physics: ground contact along bones, and bones that break under load (a CPU diagnostic is on claude/p2-bone-break).
- The kernel is bound by the L1 cache: each thread keeps its creature in local memory (about 4 KB). Muscle forces held for a whole step were 2.7x faster but cost robustness at the fine trial (88% of elites kept their distance against 95% to 99%), so a faster muscle pass must keep them per substep.
- In a long game an island's best elite sometimes replays far below its stored score (seen late in long games on the old kernel; the stored genome does not reproduce the score under any world or setting). Check it on a long game of the new kernel with `examples/replay_probe.rs` and `examples/fine_check.rs`.
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
- Archive and selection, on the GPU at larger populations (small CPU runs were ties or losses, see `docs/rejected-ideas.md`): age layers beyond the two nurseries (ALPS), deep grids, racing, dominated novelty search, the migration interval and which elites migrate [claude/archive].
- The nurseries were measured at 300k creatures a generation from the owner's save. At 3M each reshaped body gets about five times the children it gets at 300k, so measure the final split of the slots there (5% and 10%), the end-to-end rate on the owner's save (the 5% and 5% design with a coarse reshaped nursery ran 217k against 236k for main on power saver), and whether a larger reshaped share or a longer run still adds plans.
- The island plans stop at about 15,000 because an island has 5,000 cells. A finer class layout for the islands (3 by 3 classes gave 4,894 plans in an island against 2,530, in 9,400 cells) or a deep grid that keeps two or three elites of different plans in a cell would raise the plans the islands breed from. The save limits it: 152 MB for the owner's save with the global archive at 4 by 4 classes, 103 MB before, and 3 by 3 classes in every archive was 170 MB. A global elite takes 3.3 KB, and a history row 20 KB (three representative creatures and the list of body types), so a thinner history or a global archive that stores each creature once would buy room.
- Eight isolated islands instead of four were not tried: each island keeps its own lineages, so the clades of the global archive would grow, at the cost of a save with twice the island archives.
- A save keeps the ancestors of the global archive's elites and each island's fastest 10, so after a load the elites of an island that no chain reaches start as clades of their own, and the rarity bonus counts their descendants from there. A clade id per saved elite would keep the clades through a load.
- A world change tests an island's elites again as new bodies, so a graduate of a nursery loses its mark (`Elite::graduate`) and the island's count of graduate cells restarts.
- A save holds no creatures in flight, so a save made while a world change has the elites on the GPU for their re-test holds few or none of them: the owner's gen 2000 autosave held 0 elites. The autosave now waits until the global archive holds elites again (4e73f49), but a manual save in those generations still loses the elites in flight. Saving the re-test creatures still in the ring with the queue would close it.
- A generative body grammar whose rules children inherit and mutate, tested on the GPU at 100k (a grammar used only as a seed source tied, and a varied first population lost; see `docs/rejected-ideas.md`) [claude/encoding]. `segment_chain`, `limb_length_gradient` and `mirrored_limb_pair` already apply such productions to the body and the child inherits the product. A rule set stored with the creature would be a new part of every save and archive.
- A mutation strength that evolves with each lineage (a step size carried by the creature, Beyer and Schwefel 2002). It needs a new field in the genome and in every save, so it waits for a save version bump.
- The 0.035 parameter mutation after a structural operator lowers how often the child enters the archive in `mutation_audit`: for the 66 older operators 18.6% alone against 2.4% with it on the generation-590 save and 16.7% against 7.1% on the generation-1510 save (the distance added per 1,000 children falls 55% and 21%). On young saves (generation 20, before the archives refined) the entries fell 4 to 5 points and the distance added rose 3 to 7%. A compound child already gets none. Not tested in breeding: drop it for every child an operator changed and compare paired resumes of a late and a young save.
- Controllers: an optional neural controller driven by rhythm and touchdown sensors.

- The 101 idea operators (`idea_*.rs`), `mutation_audit` on a generation-40 save of the new kernel (100 elites, 4 variants): `slope_lean` and `hurdle_legs` never fit (the save's world has no slope and no hurdles); `pendulum_period_retune`, `shed_tail_tip`, `mirror_flip_body` and `arms_against_legs` keep 90% of the parent's distance in under 5% of children and none enters the archive; `tripod_copy` (688 m per 1,000 children against 36 for the parameter mutation), `ratchet_pair`, `knee_slip`, `donor_organs` and `rich_get_richer` (128 to 177) add the most. The owner keeps every operator; the weak ones are listed for a decision.
- The ~100 gait operators (`gait_*.rs`) and the owner's 10/60/30 split of a generation are not measured: run `mutation_audit` and `operator_yield` on them and a fresh game, and check breeding allocations, because five of the gait files and `sprout_leg` use heap vectors where the older operators use bounded types.

## World

- More environment effects and catastrophes (Water and Ice patches landed; ideas: wind gusts, low ceiling, moving ground).
