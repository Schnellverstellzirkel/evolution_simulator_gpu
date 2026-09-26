# Agent notes

Read this before changing the code. It lists the owner's rules, how to work on this machine, and the open work.

## Live status (Claude session, updated with each push)

Two agent teams work on this repository at the same time and only see each other through git. Pull before you start, commit small, push often, and update this section when you take or finish an item.

- 2026-09-26 14:xx, Claude is working on: (a) the owner's top priority, the mismatch between the behavior tab's scores and the replay. Plan: creatures entering the global archive also run their standard trial on the CPU engine (the engine that records the replay), and their score becomes the worse of all trials. (b) A head g-force limit: if the head accelerates faster than a limit, the creature dies like a fall. The owner asked for this to kill jiggling gaits. Please leave `src/storage.rs` archive insertion, the fall rules in `src/cpu_engine.rs`, and `shaders/physics_creature.wgsl` to Claude until this line changes.
- Just pushed: head shaking limit (a creature dies if its head's mean acceleration over 0.1 s passes 8 g), friction that counts a foot's load (97e3e9a), README rewrite (a41e9df). Next: the behavior tab vs replay mismatch.
- Measured with the 2 m bone cap (100k creatures, 20 generations, seed 38): the fastest bodies are 8 nodes, about 1.15 m of bone, 2.3 kg, 175 to 289 m in 60 s, and slip 0.12 to 0.29 m per meter. Giants are gone. The owner still sees glitchy gaits, for example a tall pyramid that jiggles at the simulation step rate.

## The game

Evolution Simulator is a Rust game. 2D creatures made of bones, joints and muscles evolve to travel as far as possible in 60 s trials. The search is MAP-Elites with CMA, structural, novelty and immigrant emitters over 4 island archives, with 3 million creatures per generation.

Physics exists in three places that must agree:

- the GPU kernel: `shaders/physics_creature.wgsl`, driven by `src/gpu.rs` and `src/vk_engine.rs`
- the AVX-512 CPU engine: `src/cpu_engine.rs` and `src/simd.rs`
- the older CPU reference: `src/physics.rs`

The replay viewport plays frames recorded by the CPU engine (`cpu_engine::trajectory`). `src/creature_kernel.rs` packs creatures for the GPU. `src/scheduler.rs` hands work units to every device. `physics::body()` computes node masses for every engine.

## Owner's product rules

- Fitness is horizontal distance only. Never add fitness terms, penalties or multipliers. Ask the owner before proposing one.
- Pressure on behavior comes from physics or environment effects, never from scoring.
- Keep settings few. The owner wants fixed 60 s trials, 3M creatures, and no mutation controls. Environment effects are buttons.
- Breaking old checkpoints is fine. Bump `qd::VERSION` when archive or physics semantics change.
- The old gameplay is not a reference. Speed matters. The long-term goal is 2M evaluated creatures per second in the graphical game at 60 FPS.
- A physics change must land in all engines and keep CPU/GPU agreement.
- Commit and push to `main` often, with plain commit messages that explain what changed, why, and what was measured.

## Working on this machine

- The laptop has 16 threads, an RTX 4060 for compute, and a Radeon 780M that drives the desktop.
- Use at most half the machine for builds, tests and runs: 8 build jobs and 8 rayon threads, at low priority (`nice`).
- Never evaluate creatures on the Radeon. Set `EVOLUTION_DEVICES=primary` and `EVOLUTION_CPU_THREADS=6` for every run of the game, the tests, and benchmarks. Heavy Radeon use crashed the desktop (mutter/Wayland) once.
- Keep subagent fan-outs small for the same reason. The session limit is 20 concurrent subagents, and 20 at once also ran out the owner's token budget.
- Fast iteration build: `CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 CARGO_PROFILE_RELEASE_INCREMENTAL=true RUSTFLAGS="-C target-cpu=native -C link-arg=-fuse-ld=mold" cargo build --release`. The committed profile uses thin LTO and rebuilds slowly.
- GPU tests are `#[ignore]`d. Run them with `cargo test --release --test simulation -- --ignored`.
- `examples/size_report.rs <checkpoint> [count]` prints body length, mass and foot slip for the best elites. With `EVOLUTION_LEDGER=1` it also prints where their forward momentum comes from.
- Before committing: `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test --release`.

## Done in the 2026-09-26 session

- The GPU and CPU engine disagreed at 4x physics fidelity. The CPU engine applied joint limits inside every bone pass, while the GPU applied them once per step. Fixed in the CPU engine, and the GPU tests now compare like with like (b4f6e20).
- Bones now have mass: bone density times length squared, split between the two joints (`Limits::bone_density`, `EVOLUTION_BONE_DENSITY`). Feet slid 0.43 m per meter traveled before and 0.02 m after.
- Muscles only pull: the drive term cannot push. Exhausted muscles have no drive (`TIRED_DRIVE = 0`), so all work comes from each muscle's energy store.
- Later the same day (commits b8ee76f to d47b2b1): the ground-lift glitch is fixed, the generational path re-tests elites after a world change, environment effects are undoable with gravity, air and grip added, and bones are capped at 2 m. See the items marked Done below.
- Result of a 20-generation run (100k creatures, seed 38) with all three changes: feet no longer slide, but the fastest bodies are still about 20 m long, weigh 750 to 800 kg, and reach 800 m. The CPU engine replays the 801 m champion at only 90 m. Evolution now exploits the ground-contact glitch described in the first item below.

## Next steps

Items marked (owner) were requested by the owner. The rest are suggestions, in rough priority order within each group.

### Creature size and movement realism

1. Done: (owner) the glitched jump. The whole-body lift after the parent-first rebuild is now a position-only correction in both engines (b8ee76f). Archive and CPU replay distances agree again (236 m vs 231 m; before, 90 m vs 801 m).
2. (owner) Stop evolution from favoring huge creatures. Measure again with the 2 m bone cap: run 20 generations (100k creatures, seed 38, `EVOLUTION_DEVICES=primary`) and read `size_report`. If bodies still pile up at the cap, try the physics items below (muscle force scaling, bone breaking).
3. Done: (owner) feet grip with the load they carry (97e3e9a). Floor clamps inside the bone and joint passes count toward a node's push, and the whole-body lift's normal impulse lets the feet resist the body's slide. Re-measure slip with `size_report` after the next changes; if feet still skate, look at the step-rate jiggle (item 1 in Live status).
4. Scale muscle force with muscle size. A longer or thicker muscle should be stronger and heavier, so a giant needs heavy muscles.
5. Let bones break under load. Bone strength grows with cross-section while load grows with mass, so oversized bones fail like real ones.
6. Done: bones and muscle strokes are capped at 2 m again (d47b2b1). Physics alone did not stop giants: after the lift fix, 16 to 22 m bodies still won.
7. Review the whole-body rescale mutation. It exists to grow giants and may no longer be needed.
8. Review the log-scale height archive axis. It gives giants their own cells and protects them.
9. Measure where a triangle's (2 bones, 1 muscle) forward motion comes from with the momentum ledger. It moves in ways the owner thinks should be impossible.
10. Remove or justify the rebuild step that lifts the whole body when a node sinks into the ground. It adds potential energy that no force paid for.
11. Add an energy conservation test: a passive body dropped on flat ground must never gain mechanical energy.
12. Add a momentum test: the projection and rebuild center-of-mass shift in the ledger should stay near zero.
13. Charge muscle energy only for active contraction work, not for passive damping.
14. Add passive elastic tendons as an evolvable part, so gaits can store and return energy honestly.
15. Add static and kinetic friction (a higher coefficient to start sliding than to keep sliding).
16. Give bones ground contact along their length, not only at the joints, so a bone cannot pass through the ground between two nodes.
17. Align or delete `physics::evaluate` (the old CPU reference). It lacks joint limits and fatigue and no longer matches the engines.
18. Consolidate the CPU engine and the old CPU reference into one CPU implementation.
19. Review the fall rule (head below neck) for bodies without a clear head.
20. Add air drag that scales with bone length times speed squared, so large fast bodies pay for moving air.
21. Report cost of transport (energy used per kg per meter) in the UI and in `size_report`, as a diagnostic only, never as fitness.

### Environment effects and catastrophes

22. Done: (owner) every environment effect has raise and lower buttons, and every change re-tests the archive (b82ff21). Effects live in `src/environment.rs`. Add new effects there.
23. (owner) Add more environment effects and catastrophes that create biodiversity and push toward complex, efficient movement.
24. Done: the generational path now puts queued elites back after a world change, before any slice reaches a device (1214bd0). A test guards it.
25. Hurdles: periodic steps whose height rises with each level.
26. Slope: the ground tilts uphill, steeper at each level.
27. Done: air drag levels (Thin, Breezy, Thick, Syrup).
28. Done: gravity levels (Earth, 1.5 g, 2 g, 3 g).
29. Done: grip levels (Grippy, Firm, Wet, Ice).
30. Mud: higher friction with sinking, so dragging feet cost more.
31. Gaps: pits that force jumping or bridging.
32. Water: a viscous medium that favors swimming strokes.
33. Wind: a steady headwind or tailwind.
34. Drought: slower muscle energy recovery.
35. Heat wave: smaller muscle energy store.
36. Meteor: a one-time catastrophe that clears a random share of archive cells.
37. Island extinction: wipe one island's archive and reseed it from the others.
38. Earthquake: a new random terrain each trial, so gaits must be robust.
39. Seasons: effects that cycle automatically every N generations.
40. A curriculum that raises difficulty when the archive stalls, as in POET (Wang et al., 2019).
41. Run the robustness trial on different terrain instead of a 2 cm pose shift (research note B7).
42. An environment panel that lists active effects with their levels, undo buttons, and short explanations.
43. A timeline of effects on the history chart.
44. Save the effect history in checkpoints.
45. Presets that combine effects ("rough hills", "icy slope").
46. Keep each effect cheap: measure creatures per second with each one on.

### Performance

47. Stop simulating fallen creatures on the GPU (lane compaction at dispatch boundaries). Research ranks it first: 43 to 55% of simulated time comes after a fall.
48. Early exit in the CPU engine when every lane in a vector group has fallen.
49. Reduce GPU kernel register pressure (about 120 registers per thread, about 30% occupancy).
50. Cut CPU time between batches: archive insertion, emitter feedback, CMA updates, offspring creation.
51. Keep the worker responsive: controls should never wait behind archive insertion or breeding (about 1 s at 1M creatures).
52. Keep the GUI at 60+ FPS during evolution at 3M creatures.
53. Add a persistent Vulkan pipeline cache and compile pipelines in the background.
54. Successive halving: short trials first, full trials for survivors (10 s ranks predict 60 s ranks with Spearman 0.89 to 0.94).
55. Decide whether the Radeon should evaluate at all by default, since it drives the desktop.
56. Size work units per device from measured rates, and re-measure after each engine change.
57. Measure memory use at 3M creatures and shrink per-creature storage.
58. Send only snapshot changes from worker to UI, not full copies.
59. Profile end to end in the GUI at 3M and record numbers in `docs/performance-log.md`.
60. Check the 4x-fidelity contender check's share of total GPU time.

### Checkpoints and storage

61. Done: autosave rotation keeps the three newest `seed-*-auto.evo` files and removes stale `.evo.tmp` files (`storage::rotate_autosaves`).
62. Shrink checkpoints (1 to 1.5 GB at 3M creatures): store the population compactly and drop data that can be regenerated.
63. Write autosaves off the worker thread so evolution does not stall.
64. Show disk use of `runs/` in the UI.
65. A save and load round-trip test that checks the next generation is identical.

### Search

66. Make structural mutations near-neutral: new parts start with weak muscles so the parent's gait survives (research: split children keep 1 to 2% of the parent's distance).
67. Re-check top elites from fresh perturbations every few generations so steady gaits beat lucky ones.
68. Spend more evaluations on the best elites (CMA-MAE thresholds, curiosity-based parent choice).
69. Let CMA respect the configured body bounds and vary joint ranges, sensors and reset phases (F6).
70. Use or remove `Creature.mutability` (mutated but unused, F8).
71. Add body size or limb count as an archive axis (research: +24% with body size).
72. Tune new bodies for a few generations before they compete (research B2, B3).
73. Encode repetition and symmetry, so limbs can be copied as modules.
74. Give each limb its own rhythm controller.
75. Add a mutation that creates antagonist muscle pairs, now that muscles only pull.
76. Crossover between different body plans.
77. Periodic extinctions per island (Lehman and Miikkulainen, 2015).
78. Age-layered populations (ALPS) so new bodies compete with their own age group.
79. Reflexes built on the touchdown sensors (a muscle that fires when its foot lands).
80. Re-run the GA research lab ablations under the new physics before drawing conclusions from old results.
81. (owner) Improve the evolution algorithm itself. The items below are candidates. Test each with paired runs at equal evaluation budgets over at least 5 seeds, and report best distance, QD score, and the body-size mix of the top 50.
82. Commit a search A/B harness (CPU engine, fixed seeds, equal budgets) so every search change is measured the same way.
83. Re-evaluate archive elites now and then and keep the worse score, so lucky results do not hold cells (noisy fitness).
84. Deep grids for noisy fitness: keep several candidates per cell and let the steady ones win (Flageat and Cully, 2020).
85. Racing: spend extra trials only on creatures whose rank is still uncertain (Hoeffding races, Heidrich-Meisner and Igel, 2009).
86. Generalized early stopping: end any trial that can no longer beat its cell's elite, not only fallen ones (Arza et al., 2024).
87. CMA-MAE annealing thresholds, so emitters keep improving cells that already have elites (Fontaine and Nikolaidis, 2023).
88. Choose emitter shares with a bandit that rewards archive improvement per evaluation.
89. Directional variation: mutate along the difference between two elites with the same body plan (Vassiliades and Mouret, 2018).
90. Discrete gene crossover between elites (Hutchinson et al., 2026).
91. Dominated novelty search as the local competition rule (Bahlous-Boldi et al., 2025).
92. Self-adapt mutation step sizes per lineage (1/5 success rule or log-normal self-adaptation).
93. Protect morphological innovations: lower selection pressure on new bodies for a few generations (Cheney et al., 2018).
94. Controller distillation, so a good gait can move to a different body (Mertan and Cheney, 2025).
95. Lamarckian inheritance: children inherit their parent's tuned controller after a short local search.
96. Review the behavior descriptors (ground contact, cadence, height). Candidates: number of feet in use, gait symmetry, body size.
97. Tune the archive size: fewer, coarser cells give each cell more offspring (research: +27% from dropping one axis).
98. Tune the island model: island count, migration interval, and which elites migrate.
99. Seed the first population with more varied bodies (bilateral, longer chains), not only 3 to 5 node chains.
100. A generative body encoding (grammar or L-system) so larger bodies stay coherent.
101. An optional neural controller: a small network driven by rhythm and touchdown sensors, as an alternative to fixed waveforms.
102. When the archive stalls, suggest an environment effect in the UI instead of changing the search silently.
103. Re-run every research conclusion under the new physics (bone mass, pull-only muscles). All numbers in docs/search-research.md predate it.

### Correctness and tests

104. Audit the top elites for physics exploits after every physics change.
105. Test that the CPU engine and the replay frames match the scored distance.
106. Test that archive insertion keeps one elite per cell and never replaces a faster elite.
107. Test that Config validation rejects bad values and accepts defaults.
108. Test that breeding is deterministic for a fixed seed and every offspring is a valid body.
109. Test that a creature that falls or breaks a joint keeps the score it had at that moment.
110. Add a GPU agreement test at 4x fidelity for evolved creatures, not only random ones.
111. Decide how to test the perturbed contender check across engines. Fall and break decisions can flip on rounding.
112. Run clippy and CPU tests in CI (GitHub Actions), and keep GPU tests local.

### Interface

113. Replay viewer: follow camera, distance ruler, speed readout, center-of-mass trail, playback speed, scrubber, fall marker.
114. Behavior archive map: a heatmap of cells colored by distance, click to replay.
115. History tab: best distance over generations, records timeline, replay of each record holder.
116. Race view: the top elites run side by side with a leaderboard.
117. Creature drawing: muscle activation and fatigue colors, head, organs, touchdown highlights, broken joint marks.
118. Help overlay (F1 or ?), shortcuts for tabs and replay, and a status line with creatures per second.
119. Share a creature: export an animated GIF and a JSON file, and open a creature JSON to replay it.
120. Lineage view: ancestors with thumbnails, mutation labels, gains, and body plan changes highlighted.
121. Show each muscle's energy during replay, to make fatigue visible.
122. A debug overlay for forces and ground reactions.
123. Name species automatically so players can follow them.
124. A hall of fame of record holders across the whole run.
125. Remove settings the owner does not want (histogram controls, budgets) or move them to a debug panel.
126. Tooltips that explain each archive axis in plain words.
127. A screenshot button.
128. A dark theme.

### Code health and docs

129. Done: README.md describes the current game (a41e9df).
130. Update docs/architecture.md: trial length, fitness, physics limits, bone mass, pull-only muscles, the fidelity check.
131. Update docs/validation.md with the new GPU agreement results.
132. Remove the legacy `mutate()` path that the app no longer uses.
133. Remove the empty obstacle slot kept for old checkpoints, since breaking saves is fine.
134. Remove environment variables that no experiment uses any more.
135. Decide what to do with `research/`: commit the harness and results, or ignore the folder.
136. Delete the stray `cuda-keyring_1.1-1_all.deb` files in the repository root.
137. The local branch `wip/cpu-finalist-validation` (not pushed) holds an older owner change that replayed archive finalists on the CPU before they entered the archive. The contender check in 71e9088 replaces it. Delete the branch or port anything missing.
138. Done: `.claude/` is in `.gitignore`.
139. Handle GPU device loss by falling back to the CPU engine instead of stopping.
140. Log per-generation stage times to a file for later analysis.
141. Speed up builds: consider the fast iteration profile as a named Cargo profile.
