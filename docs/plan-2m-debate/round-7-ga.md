# Round 7, GA/QD: the cheaper physics and the search

I accept the proposal's direction. Pull-only, Hill, a rhythm of period, phase and duty, a cost that recovers, touchdown clocks and muscle mass are the genes evolution has been using; the anchors, the stroke targets, the per-muscle store and the tendon were mechanism.

## (a) Genes, operators, the CMA dimension

Genes that go, from `Muscle`: `anchor_a`, `anchor_b` (attachment becomes two node ids), `short` and `long` (an activation model has no target length: the stroke is what force and the joint range allow), `tendon`. `stiffness` becomes `strength`, a fraction of the cap. `period`, `phase`, `duty`, `sensor`, `reset` stay. New gene on `Bone`: `ligament`, the joint-limit compliance. Physics did not say `short` and `long` go, but its force law has no length term, so they do; if it wants a length-dependent force it must say so, because that is where a stroke gene would come back.

Operators that touch removed genes, from the code (writers, not tests):

- Anchors, 15 operators: `move_muscle_to_neighbor`, `split_muscle`, `fuse_similar_muscles`, `add_antagonist`, `swap_muscle_routes`, `fan_muscle_attachments`, `relay_muscle` (muscles.rs); `scale_muscle_leverage`, `lift_dragging_end`, `copy_muscle_to_partner` (extra.rs); `grow_lever_spur`, `merge_branch_joints` (junctions.rs); `split_bone_actuated`, `fuse_bones` (limbs.rs); `mutate_matching_limbs` (rhythm.rs); plus `renumber` in mod.rs. Re-targeting rule: an operation on "a point along a bone" becomes an operation on "a node", and when the node it needs does not exist the operator makes it by splitting the bone there (the existing split-bone code path). So `move_muscle_to_neighbor` moves an end to the neighbouring bone's far node; `fan_muscle_attachments` splits the bone at the fan points and attaches the ends to the new nodes; `scale_muscle_leverage` moves an end to the next node out or in along the limb, or splits the bone to make one; `relay_muscle` relays through a middle node; `grow_lever_spur` is unchanged (the spur tip is a node); the copy and fuse operators copy node ids. Every one keeps its name and its defining test, restated on nodes. This is the mechanism physics named: leverage comes from node placement, and these operators are the ones that place nodes for muscles.
- Strokes, 6 operators: `stroke_scale`, `posture_shift`, `trade cadence against stride`, `quiet a branch's strokes`, `split_bone_actuated`'s "muscle across the new joint", `copied_actuation`. Stroke scale and quiet become strength scale; posture shift becomes a shift of the joint range's middle (`Bone::mutate_range` exists); cadence against stride trades period against strength.
- Stiffness, 6 operators (`taper_limb_strength`, `scale_limb_strength`, `retune a muscle pair`, `fuse_similar_muscles`, `change the weakest muscle`, `passive_ring`): they act on `strength`, same range semantics on 0 to 1.
- Tendon: no operator touches it. Only `local_mutation` grows, tunes or drops a tendon with 10% chance. The same three moves act on `ligament` per bone. Nothing to re-target.

No operator is removed; 27 of 64 change what they read and write. The property test over 160 grown bodies and `mutation_audit` (limb operators keep 2 to 3% of the parent today, timing operators 97 to 100%; the re-targeted ones should land in the same groups) are the gate.

CMA dimension, p50 body (8 nodes, 7 bones, 19 muscles): today 32 + 35 + 1 + 152 = 220; new, with a ligament per bone and 4 genes per muscle (strength, phase, duty, reset): 32 + 42 + 1 + 76 = 151. Thirty percent fewer dimensions, and the leverage search moves into the node positions, already CMA coordinates. The emitter track does not change: lambda 200 stays right (10 n would be 1,500), the tell costs 30% less, and the `Layout` table is the only code that changes. One risk to measure: node x and y now carry shape and leverage, so their CMA scale (0.02 of body size) may be too small for leverage moves; the fix is a measured scale, not a knob.

## (b) Which shapes win, and the muscle-count slope

Stamina per creature charges work, not muscles. Two bodies doing one gait with 20 or 40 muscles do the same work, so stamina alone does not price a muscle. What it prices is wasted work: co-contraction, antagonists pulling against each other, muscles that fire out of phase with the motion. That is exactly where the pre-mass bloat lived (80 to 87 muscles on 13 nodes as braced frames), so stamina flattens the slope for wasteful muscles and leaves useful ones free. Muscle mass stays the price of a muscle as such and is still needed; physics keeps it, and I agree.

The growth-step rule is about offspring cost, not elite counts: the 25, 29 and 30 node clusters are one operator's step on a large parent and stamina does nothing to them. It stays as the cap candidate. Whether it is also needed for muscles is measurable in the 30-generation run: elite muscle p90 at generation 30 against today's 28 to 34 with no rule; under 30 means stamina and mass hold it and the rule is nodes-only for the tail.

One shape effect to watch: without a tendon, hopping efficiency must come from the ligament, so hoppers change from a springy muscle to a joint driven into its stop, and braced stop-riding frames may dominate again. The contact bin and the top-20 pair count in (c) catch it.

## (c) The spirit checks as search numbers

`search_ab --gpu`, 30 generations at 3M, 5 seeds, on the lane-group kernel with the variants behind defines (about 15 minutes per seed at today's rate). The numbers that say "interesting efficient movers":

- Random bodies gain no distance: `first_generation` median within 0.05 m, best under 0.5 m (physics's bar).
- Progress: best distance rising monotonically over generations 5 to 30 on every seed, and generation 30 within 0.5x to 2x of today's 37 m (a different physics scales differently; outside that band the physics is either too weak to walk or catapulting, and the retest says which).
- Diversity: at generation 30, at least 5 of 8 cadence bins, 4 of 6 contact bins and 3 of 5 feet bins each hold an elite within 50% of the best; global behaviour coverage at least 70% (today 95% at generation 30).
- Efficiency: among the top 300 the cost of transport (stamina drawn per metre, the archive can compute it from the ledger) has a p90 to p10 ratio of at least 2, and its median falls or holds from generation 10 to 30. A search that finds distance by burning more is not finding efficiency.
- Honesty: elite retest at the finer rate median 0.9 with p10 reported, planted-foot slip 0.95 to 1.05, ledgers at 1e-5, penetration p99 under 2 mm.
- The owner's eye: the top-20 replays, with the count of distinct (cadence bin, contact bin) pairs among them at least 5.

What says it does not: one (cadence, contact) pair holding more than 80% of the top 300 (one gait type in every cell); ground contact under 0.4 for more than 90% of the top 300 (hoppers only); slip outside the band (sliders); a retest median under 0.9 (catapults or integrator riders); a cost-of-transport median rising over the run (brute force); or fewer than 3 distinct pairs in the top 20. Any one of these fails the variant, and the variant's own knob-free fix (a ledger, a bound) is what physics changes, not the search.

## (d) The population numbers

They do not carry over. Anchors gone means leverage needs nodes, so I expect bodies to gain 1 to 3 nodes at equal function; four genes per muscle instead of eight means a muscle is cheaper for the search to tune, so counts may rise until mass and stamina bite; no tendon changes the hopper family. The generation-51 numbers (8.7 mean, p90 12, the 9% tail) were measured under a physics that will not exist, and the save cannot be read after the `qd::VERSION` bump. The plan must re-measure the ring histogram (mean and p90 of nodes and muscles, the 17-plus share, the operator histogram of the tail) at generation 30 of the run in (c), and the growth rule's tail numbers and gpu's class shares follow from that, not from save42. The rung ladder is unaffected in form: R1's discriminant is refit every generation and its "mean muscle energy" feature becomes stamina; the per-cell bars are built from the new elites' rung distances; R4's emitter-median floor is unchanged.

## My levers, revised

Steps per creature unchanged: 300 with R1 to R3, about 270 with R4. Emitters unchanged in gain, 30% cheaper per tell. The growth rule stays the cap candidate with its numbers to be re-derived. Ceiling with physics's rates: 4 substeps 1.7M/s at 300 steps and about 1.9M/s at 270; 2 substeps 2.3 to 2.5M/s; generation 100 at 0.85x. So 2M/s at 4 substeps needs R4, and it is inside the range for the first time without the 5T budget.

## The physics-lean track as I would gate it

Worktree `claude/physics-lean`. Order: the stub counts per variant (gpu, a day each); the lane-group kernel with the variants behind defines, the genome change in `evolution.rs`, the 27 re-targeted operators with their tests, the `Layout` table and the `qd::VERSION` bump (one week); the 30-generation run of (c) on 5 seeds with `first_generation`, `mutation_audit`, `physics_audit` and `size_report`; then the substep ladder on the new physics (evolve at 2 substeps of 1/120, retest at 8, median 0.9). The run's generation-30 ring histogram replaces save42 for sizing the kernel classes and the growth rule. Nothing compares bits with today's kernel.
