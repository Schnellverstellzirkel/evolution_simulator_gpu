# Agent notes

Read this before changing the code. It lists the owner's rules, how to work on this machine, and the open work.

## Live status (updated with each push)

- Codex primary: finished integrating verified foundation commits 250ad82/ca75014 with origin 73ddf68; 75 CPU and three RTX tests pass, with nine report tests passed at the preceding diagnostic revision. Worker failure propagation is now fixed and tested (83 CPU, nine report, three RTX tests). Submitted bodies and configurations are now retained with shared ownership (88 CPU, nine report, three RTX tests). Device recovery batches 1 to 3 are done (see `docs/superpowers/plans/2026-09-26-device-recovery.md`); batch 4, startup CPU fallback and backend reporting, is next and owns src/engine.rs, src/scheduler.rs and src/gpu.rs. Active physics/archive-insertion work remains with Claude.

Two agent teams work on this repository at the same time and only see each other through git. Pull before you start, commit small, push often, and update this section when you take or finish an item.

- 2026-09-26 15:5x, Claude pushed: the sled regression test (2b8ae27..8fd2234); the grip effect now has both slippery and grippier levels with the calm world at Grippy; the live view has green ground, a blue sky with clouds, and muscles that go pink to deep red with contraction. Still open from the earlier list: a full evolution measurement under the planted-feet rule (5905316). Please leave the physics in `src/cpu_engine.rs` and `shaders/physics_creature.wgsl`, and archive insertion in `src/storage.rs`, to Claude until this line changes. Claude stays out of the files the Codex team listed below.
- 2026-09-26, Codex Luna team (coordinated by the primary Codex session) is working on: CPU evaluation backend and top-50 body mix in `src/search_benchmark.rs`, `src/main.rs`, and `docs/search-benchmark.md`; current-physics details in `docs/architecture.md`. Please leave these files to the assigned workers until this line changes.
- Just pushed: a failed GPU is retired and its unfinished units, including pending fine checks, are re-submitted to a healthy CPU with their exact creatures and settings; a failed CPU is terminal and delivers completed output before its persistent error. Rejected submissions keep their creatures in the round. 94 CPU tests, nine report tests, three RTX agreement tests.
- Just pushed: a `fast` Cargo profile and machine-safe build/run instructions in `docs/building.md`.
- Just pushed: replay scrubber and fall marker on the timeline, with seeking paused safely at the selected frame.
- Just pushed: secondary GPU evaluation is opt-in; the default uses the primary GPU and CPU only. `EVOLUTION_DEVICES` can select extra devices explicitly.
- Just pushed: only planted feet may push the body forward (5905316). The capped version still let evolution build a sled: four nodes sliding on the ground all trial reached 2,267 m in 60 s. Friction may now push forward only while the feet on the ground slide slower than 1 cm/s; while they slide it can only oppose the slide. Also fixed a clippy error in the replay clock (`src/ui.rs`).
- Just pushed: planted feet with a friction cap. b57f756 alone was an exploit (random bodies reached 224 m in 20 s because weighting grounded nodes broke momentum conservation). The fix limits the center-of-mass shift from the bone passes to mu times the ground's normal push; random bodies now reach 9.8 m at best (6.7 m without planted feet). `examples/first_generation` checks any physics change for free propulsion this way; run it before pushing physics changes.
- Just pushed: undoable meteor strike (Environment panel, Catastrophe row).
- Just pushed: the replay takes its fall and distance from the CPU engine run that recorded it (`cpu_engine::replay`), and the viewport header shows the replay distance (fb4b594). The global archive only admits scores the CPU replay reproduces (7f3f3a3; test `archive_scores_never_exceed_the_replayed_distance`). Head shaking limit: a creature dies if its head's mean acceleration over 0.1 s passes 8 g (a08fdc4). Friction counts a foot's load (97e3e9a). README rewrite (a41e9df).
- Measured with the 2 m bone cap (100k creatures, 20 generations, seed 38): the fastest bodies are 8 nodes, about 1.15 m of bone, 2.3 kg, 175 to 289 m in 60 s, and slip 0.12 to 0.29 m per meter. Giants are gone. The owner still sees glitchy gaits, for example a tall pyramid that jiggles at the simulation step rate.

## The game

Evolution Simulator is a Rust game. 2D creatures made of bones, joints and muscles evolve to travel as far as possible in 60 s trials. The search is MAP-Elites with CMA, structural, novelty and immigrant emitters over 4 island archives, with 3 million creatures per generation.

Physics exists in three places that must agree:

- the GPU kernel: `shaders/physics_creature.wgsl`, driven by `src/gpu.rs` and `src/vk_engine.rs`
- the AVX-512 CPU engine: `src/cpu_engine.rs` and `src/simd.rs`
- the older CPU reference: `src/physics.rs`

The replay viewport plays frames and the matching scored result recorded by the CPU engine (`cpu_engine::replay`). `src/creature_kernel.rs` packs creatures for the GPU. `src/scheduler.rs` hands work units to every device. `physics::body()` computes node masses for every engine.

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
- Fast iteration build: `cargo build --profile release-fast` inherits release optimization with LTO disabled, 256 codegen units, and incremental compilation. It adds no platform-specific linker requirement. Use the normal thin-LTO release profile for comparable performance measurements.
- GPU tests are `#[ignore]`d. Run them with `cargo test --release --test simulation -- --ignored`.
- `examples/size_report.rs <checkpoint> [count]` prints body length, mass and foot slip for the best elites. With `EVOLUTION_LEDGER=1` it also prints where their forward momentum comes from.
- Before committing: `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test --release`.

## Done in the 2026-09-26 session

- The GPU and CPU engine disagreed at 4x physics fidelity. The CPU engine applied joint limits inside every bone pass, while the GPU applied them once per step. Fixed in the CPU engine, and the GPU tests now compare like with like (b4f6e20).
- Bones now have mass: bone density times length squared, split between the two joints (`Limits::bone_density`, `EVOLUTION_BONE_DENSITY`). Feet slid 0.43 m per meter traveled before and 0.02 m after.
- Muscles only pull: the drive term cannot push. Exhausted muscles have no drive (`TIRED_DRIVE = 0`), so all work comes from each muscle's energy store.
- Later the same day (commits b8ee76f to d47b2b1): the ground-lift glitch is fixed, the generational path re-tests elites after a world change, environment effects are undoable with gravity, air and grip added, and bones are capped at 2 m. See the items marked Done below.
- Historical result before the lift fix and 2 m bone cap: a 20-generation run (100k creatures, seed 38) produced roughly 20 m, 750–800 kg bodies and an 801 m archive champion that replayed at only 90 m. Its low slip measurements did not establish grounded traction. The baseline below is a separate historical measurement before the newer contact and replay-admission fixes.

### Historical bd41746 physics baseline and Codex foundations

- Seed 38, 100k candidates, 20 generations, 60 s trials: final best 165.5846 m, 1,374 behavior cells, QD score 23,060.27. Top-50 median total bone length is 2.23 m; maximum individual bone is 1.81 m. No pile-up at the cap appeared in this sample.
- The champion stores 165.6 m and replays at 157.3 m; rank 21 stores 111.2 m and replays at 6.4 m. The latter was observed before the CPU archive-admission check (7f3f3a3); it is not evidence of a failure of that newer check. Updated scored-interval slip reports a median 0.89 m per replay meter; old ratios are not directly comparable.
- Public CPU evaluation now uses the replay engine. Checkpoint V4 preserves island optimizer progress, while V3 remains readable; stale island/reseed state is cleared on physics-version migration. The foundation commit 250ad82 changed no production stepping equations; the concurrent changes now merged advance qd::VERSION from 16 to 19.
- Resource defaults exclude Radeon compute and share eight workers across the two CPU pools. CPU CI, regression coverage, current docs, and a named fast profile are in place. Before merging concurrent physics work, local checks passed: formatting, all-target clippy, 68 CPU tests, seven size-report tests, and three explicit GPU agreement tests (3.32 s). Merged verification passed: 72 CPU tests, nine size-report tests, three RTX agreement tests (14.36 s), formatting and Clippy. The 20k-body, 20 s random-population check measured median -0.07 m, p99 1.85 m, best 9.81 m. Four GPU tests remain ignored in the default suite. Remote CI execution remains unverified; see docs/validation.md.
- Sanitized baseline data: docs/results/2026-09-26-bone-cap-seed-38/. No checkpoint is committed.

## Next steps

Items marked (owner) were requested by the owner. The rest are suggestions, in rough priority order within each group.

### Creature size and movement realism

1. Done: (owner) the glitched jump. The whole-body lift after the parent-first rebuild is now a position-only correction in both engines (b8ee76f). Archive and CPU replay distances agree again (236 m vs 231 m; before, 90 m vs 801 m).
2. Baseline measured: (owner) stop evolution from favoring huge creatures. The 20-generation, 100k, seed-38 run with the 2 m cap produced top-50 median total bone length 2.23 m and longest individual bone 1.81 m: no pile-up at the cap in this sample. The champion is 1.77 m total bone length and 3.20 kg. Broader seeds and longer runs remain open; this single sample does not justify declaring size selection solved. See docs/results/2026-09-26-bone-cap-seed-38/.
3. Implemented: (owner) feet grip with the load they carry (97e3e9a), with planted feet (b57f756) and a friction cap on projection-induced propulsion (8572d32). Re-measure under the merged physics using the corrected scored-interval size_report; the historical bd41746 median slip of 0.89 is not a current solver result. The earlier source audit is preserved as historical evidence only.
4. Scale muscle force with muscle size. A longer or thicker muscle should be stronger and heavier, so a giant needs heavy muscles.
5. Let bones break under load. Bone strength grows with cross-section while load grows with mass, so oversized bones fail like real ones.
6. Done: bones and muscle strokes are capped at 2 m again (d47b2b1). Physics alone did not stop giants: after the lift fix, 16 to 22 m bodies still won.
7. Review the whole-body rescale mutation. It exists to grow giants and may no longer be needed.
8. Review the log-scale height archive axis. It gives giants their own cells and protects them.
9. Measure where a triangle's (2 bones, 1 muscle) forward motion comes from with the momentum ledger. It moves in ways the owner thinks should be impossible.
10. Remove or justify the rebuild step that lifts the whole body when a node sinks into the ground. It adds potential energy that no force paid for.
11. Done: tests `a_passive_body_never_rises_above_its_start` and `a_body_without_drive_does_not_travel` guard against solver-made energy and propulsion. `examples/first_generation` compares random-population distances across physics changes.
12. Add a momentum test: the projection and rebuild center-of-mass shift in the ledger should stay near zero.
13. Charge muscle energy only for active contraction work, not for passive damping.
14. Add passive elastic tendons as an evolvable part, so gaits can store and return energy honestly.
15. Add static and kinetic friction (a higher coefficient to start sliding than to keep sliding).
16. Give bones ground contact along their length, not only at the joints, so a bone cannot pass through the ground between two nodes.
17. Done: `physics::evaluate` now delegates to the production CPU engine, including configured fidelity, joint limits, fatigue, and fall/break scoring. Standard/fine replay regression coverage was added. The lower-level legacy `physics::step` remains; consolidation is still item 18.
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
36. Done: meteor strike wipes out half of every archive's elites; Undo returns the fossils (`Experiment::meteor`, `undo_meteor`).
37. Done: Extinction wipes out the slowest island (`Experiment::extinction`), undoable with the same fossils as the meteor.
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
55. Done: secondary GPUs are opt-in; the scheduler defaults to `primary`. General and CPU evaluation pools share a budget of at most eight threads and half the logical CPUs (six evaluation plus two general workers on this laptop). Continue setting the explicit workstation environment for every run.
56. Size work units per device from measured rates, and re-measure after each engine change.
57. Measure memory use at 3M creatures and shrink per-creature storage.
58. Send only snapshot changes from worker to UI, not full copies.
59. Profile end to end in the GUI at 3M and record numbers in `docs/performance-log.md`.
60. Check the 4x-fidelity contender check's share of total GPU time.

### Checkpoints and storage

61. Done: autosave rotation keeps the three newest `seed-*-auto.evo` files and removes stale `.evo.tmp` files (`storage::rotate_autosaves`).
62. Shrink checkpoints (1 to 1.5 GB at 3M creatures): store the population compactly and drop data that can be regenerated.
63. Done (existing implementation): autosave serialization and writes use a background thread. Snapshot-copy cost on the worker still needs measurement before claiming stall-free autosaves.
64. Show disk use of `runs/` in the UI.
65. Done: regression tests compare next-generation genomes, archive/CMA state, and offspring metadata after checkpoint round trips, including stalled island optimizers, steady breeding, and environment changes. V4 checkpoints now persist optimizer progress; V3 remains readable. Final integrated checks are recorded in docs/validation.md.

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
105. Done: standard/fine CPU scores are compared with mass-weighted terminal replay frames, including partial SIMD groups. This does not establish agreement for every evolved GPU-scored elite; the historical rank-21 outlier predates the CPU archive-admission check.
106. Done: archive insertion and island migration regression tests cover unique cells and rejection of slower candidates.
107. Done: configuration regression tests cover defaults, float/integer boundaries, ordered bounds, and the population RAM limit.
108. Done: modern archive-breeding tests compare valid offspring across fixed seeds and streaming slice sizes, including CMA, structural, and novelty output.
109. Done: standard/fine regression fixtures verify frozen scores at falls and joint breaks while replay motion continues.
110. Add a GPU agreement test at 4x fidelity for evolved creatures, not only random ones.
111. Decide how to test the perturbed contender check across engines. Fall and break decisions can flip on rounding.
112. Configured: GitHub Actions runs formatting, all-target clippy, and release CPU tests with resource limits and the portable SIMD path. GPU tests remain local and ignored by default. Remote workflow execution is not yet verified.

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

129. Done: README describes current 60 s / 3M defaults, distance-only scoring, archive/search behavior, safe runs, environment buttons, diagnostics, and save/resume.
130. Done: architecture updated from source, including masses, pull-only active drive, fatigue, standard/fine checks, replay semantics, scheduler, and checkpoint state.
131. Done: validation records the local three-test GPU pass and the new 20-generation baseline, with historical workloads clearly separated. Final local checks passed: 68 CPU, seven example, and three explicit GPU tests; remote CI remains unverified.
132. Remove the legacy `mutate()` path that the app no longer uses.
133. Remove the empty obstacle slot kept for old checkpoints, since breaking saves is fine.
134. Remove environment variables that no experiment uses any more.
135. Decide what to do with `research/`: commit the harness and results, or ignore the folder.
136. Delete the stray `cuda-keyring_1.1-1_all.deb` files in the repository root.
137. The local branch `wip/cpu-finalist-validation` (not pushed) holds an older owner change that replayed archive finalists on the CPU before they entered the archive. The contender check in 71e9088 replaces it. Delete the branch or port anything missing.
138. Done: `.claude/` is in `.gitignore`.
139. In progress (Codex primary): worker failures/disconnections now persist through wait/poll without losing completed results. CPU retry after GPU device loss is next; src/engine.rs, src/scheduler.rs, src/gpu.rs are assigned to this work.
140. Log per-generation stage times to a file for later analysis.
141. Done: `release-fast` is the named incremental release profile (LTO off, 256 codegen units); normal release retains thin LTO. No build-speed measurement is claimed.
