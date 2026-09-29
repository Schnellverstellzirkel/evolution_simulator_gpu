

## 2026-09-29: the rate drop after ordered blocks was check volume

Owner report: on main 91c4536 the game's rate fell sharply and kept falling over generations, and the world view waited for the generation end before it showed a new champion.

`examples/worker_rate` at 3M, seed 38, CUDA, exclusive GPU lock, owner's game idle: generation 0 ran at 105k/s, generations 3 to 11 at 74k, 84k, 68k, 68k, 62k, 65k, 61k, 59k and 47k/s (63.9k/s over generations 3 to 12). `nvidia-smi` sampled every 500 ms read 98% utilization over the whole run, so the ordered-block pipeline does not leave the GPU idle. The GPU does more work per generation.

The stage log now has the check trials, check and device busy seconds, device idle seconds, mean nodes per body and the share of bodies above 8 nodes per generation. Main ran 174k, 427k, 720k and 350k check trials in generations 0 to 3. A check costs about 4 standard trials. The contender counts (`EVOLUTION_PROFILE_BREED`, now fed by `check_need` too) showed 280k to 610k reserve contenders per generation. A reserve contender is a structural or novelty child above the reserve floor. It had no cell key, so every one got a check. In the ordered blocks, decisions see the archives 12 blocks behind, so the floor they face is stale. The pre-ordered base 62ba7a3, under a shared lock, ran 330k to 440k checks per generation in total.

Fix (bfeb7d8): the reserve keeps one elite per body plan, so a reserve contender carries a key per body plan and the blocks check the best one per plan like an archive cell. Checks per generation fell to 38k, 101k, 197k, 171k, 80k, 148k, 124k, 126k in generations 0 to 7. The rate was 102k, 145k, 99k, 84k, 92k/s in generations 0 to 4, against 94k, 125k, 72k, 81k/s for main in the same window. Best distance matched main for generations 1 to 4 (0.40, 4.71, 8.10, 10.57 m). Generations 5 and 6 of that run read 36k and 29k/s and may have had the owner's game on the GPU, because it runs without the lock. What is left after the fix is mostly optimizer samples (80k to 170k per generation, all checked by design) and body growth: mean nodes per body rose from 5.0 to 7.3 over 8 generations, and bodies above 8 nodes run in the 16-node kernel at a fraction of its occupancy.

Measuring note: with the owner's game running unpaused, a run got a different engine (its first block admitted 560 elites where CUDA admits 165) at half the rate. Time only with `tools/pause-game.sh`.
