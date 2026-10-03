# Anatomy mutation operators

The structural emitter picks one of 7 classic operators (split a bone, mirrored node, duplicate a leaf limb, retime, organ, phase shift, rescale), one of the anatomy operators in `src/evolution/anatomy/` or one of the compound operators below. All of them are on. It picks uniformly among the classic operators, the anatomy and compound operators that have their own slot, and the shared slots, and tries again (up to four times) when the chosen operator does not fit the body. A child of the structural emitter gets one structural operator and then a small parameter mutation, except a child of a compound operator, which gets none because its move is the whole change. The owner wants more operator types and never fewer.

Every operator keeps the body limits and never touches the head or the neck.

| group (file) | operators |
|---|---|
| limbs (`limbs.rs`) | copy a whole branch with its muscles and timing; grow an actuated tip; split a bone with a narrow new joint and a muscle across it; fuse two aligned bones; move a branch to another node; rescale a branch and its strokes; graft a branch from another elite |
| junctions (`junctions.rs`) | split a crowded junction into two joints; merge two junctions; repeat a trunk segment with its limbs; grow a heel and a toe; grow a lever spur and move a muscle end onto it; reflect a branch so it bends the other way; start a joint near one of its stops (same stops, new starting pose); brace a joint against a stop with a small flex left |
| muscles (`muscles.rs`) | add a muscle across two joints; move a muscle end to a neighbouring bone; split a muscle; fuse two similar muscles; add an antagonist (opposite torque about the joint, half a cycle apart); swap two muscles' destinations; fan clustered anchors along a bone; replace a long muscle with a relay through a middle bone; copy one limb's actuation onto another limb; quiet a branch's strokes |
| rhythm (`rhythm.rs`) | move joint range from one joint to its neighbour; apply one change to two matching limbs; a phase wave down a chain; limb phase patterns (together, alternating, staggered); a limb's duty cycle around each contraction's middle; a coordinated touchdown reset; move organ mass between bones |
| extra (`extra.rs`) | copy a leg to the dragging end of the body; lift the dragging end; twin a limb in place; the same tip on two matching limbs; remove the idlest limb tip; merge the last two bones of a limb; and, sharing one pick slot, eight gentle operators that copy, swap or shift limb programs, trade cadence against stride, and change leverage, strength or the weakest muscle |
| controller (`controller.rs`) | limb stroke scale; limb posture shift; taper limb strength; copy limb rhythm; retune a muscle pair; release touchdown; snap limb phases; put a limb on another clock ratio; lock a limb back on the base clock; reflexes on a muscle, on all feet, and a shifted reflex reset. These share one pick slot. Starting the gait at another point of its cycle has its own slot |
| compound (`compound.rs`) | coherent changes that touch several parts of a body at once, one line each below. Each has its own pick slot |
| legs (`legs.rs`) | four compound operators that make mammal-like legs cheap to reach, described below. Each has its own pick slot and its child gets no parameter noise |
| gait (`gait_*.rs`) | 106 compound operators in eight files that build gaits, listed below. Each file's operators share one pick slot |

A limb may run on its own clock at a ratio of the base clock (1/2, 2/3, 1, 3/2 or 2), so the whole gait still repeats.

## Compound operators

A compound operator changes several parts of a body together and keeps them consistent with each other, so a child is a larger step than one edit and still has a chance of keeping its parent's gait. A child of the structural emitter that a compound operator made gets no parameter mutation after it, because the noise halves how often such a child enters the archive (see the audit below) and it would blur a move that was built to be coherent. Skipping it also saves its breeding time. The operators that add bones close the motor ring themselves, in canonical bone order, with passive springs, so `repair` adds no random active muscles (a later `repair` would add more for every pair the canonical order moves). The operators that add nodes also remove the idlest limb tips (never one of the new parts), because bodies that only grow take more GPU lanes per creature (past 8 or 16 nodes a body takes twice as many). In the hub island a donor elite came from any isolated island, so the transplant exchanges programs between islands.

- `limb_length_gradient`: the leaf limbs scale in a gradient from the front of the body to the back, the longest 1.08 to 1.7 times the shortest, with every muscle keeping its stroke relative to its span (Hornby and Pollack 2001, repeated parts with a gradient).
- `symmetrize_limb_pair`: one limb becomes the mirror image of a limb of the same shape, with its lengths, mirrored joint ranges, node sizes and organs, and its muscles replaced by copies of the other limb's half a cycle later (Sims 1994, symmetric pairs with mirrored timing; Cheney et al. 2013, regular bodies). In three of ten moves the limb is copied without the reflection, in phase or half a cycle later.
- `retime_gait_by_position`: the leaf limbs, ordered front to back, get one of five phase patterns (hop, walk, bound, a wave in either direction), and half the time the whole gait starts at another point of its cycle.
- `brace_limb_chain`: every joint of a limb with two or more bones is set against its stop on one side with a small flex left, which bends the whole limb into a rigid shape.
- `phase_cluster_move`: the active muscles that share a phase, whatever limb they drive, move together by 5 to 30% of a cycle, or two such groups swap places (Beyer and Schwefel 2002, correlated mutation).
- `reassign_bundle`: two to four muscles of a bundle of five or more on one pair of bones move to a joint beside it, each keeping its timing and the shape of its stroke.
- `retune_limb_package`: one limb's controller changes in two to four of five ways together (stroke, posture, phase, duty with the phase following, strength).
- `grow_integrated_limb`: a new part (a tip, a toe and heel, a joint with a muscle across it, a copied limb or a lever) comes with its muscles timed against the gait's main driver, its joint braced half the time and its foot sensing touchdown some of the time.
- `mirrored_limb_pair`: a limb of one or two bones and its mirror image hang from another node of the body, with joint ranges mirrored and a muscle across each joint, half a cycle apart (Sims 1994).
- `segment_chain`: a trunk segment with its limbs repeats once or twice down the chain, each copy 0.8 to 1.25 times the size of the one before and its muscles a step of 0.1 to 0.3 of a cycle later (Hornby and Pollack 2001, repeated segments with a gradient; a wave of contraction down the body).
- `transplant_limb_program`: a limb takes the muscle program of a limb of another elite with the same number of bones, moved in time so its strongest muscle keeps the phase the old one had (Lessin, Fussell and Miikkulainen 2013, whole modules exchanged).
- `transplant_gait`: the leaf limbs, front to back, take the programs of another elite's leaf limbs at the same places in the order (where the bone counts match, for two limbs or more), keeping the donor's timing among them and the front limb's place in the cycle.
- `trim_body`: the idlest limb tips (one to four by the size of the body) and up to three of the weakest muscles off the motor ring go in one move, for bodies with three or more muscles to a node. Young bodies have fewer and need every muscle.

## Leg operators

These four are compound operators in `legs.rs`. They build the parts of a mammal gait: legs under the trunk, in pairs, spread along it, stepping in a fixed phase against each other (Sims 1994, Lipson and Pollack 2000, Cheney et al. 2013, Stanley 2007).

- `sprout_leg`: a leg of two bones (thigh and shank) hangs from a trunk node, pointing down with a small lean and a bent knee. A hip muscle and a knee muscle drive it, the knee a quarter cycle behind the hip, and half the time a second hip muscle pulls the other way half a cycle later. The leg steps against the nearest leg: half a cycle after it, a quarter or three quarters (gallop) or with it (bound). The idlest tips go back.
- `mirror_leg_fore_aft`: a leg of up to three bones is copied to the node nearest the mirror image of its hip about the middle of the trunk, translated or reflected, in the phase of a four-legged gait (trot: half a cycle, bound: none, gallop: a fifth of a cycle either way).
- `spread_leg_attachment`: a leg's hip moves along the trunk to the nearest node that lies farther from the middle of the body. The leg keeps its shape and muscles.
- `tuck_leg_under`: a leg turns so its first bone points down, within a small lean, which puts its foot under its hip. Only legs 0.1 to 1.3 rad off straight down turn.

## Gait operators

These 106 compound operators were added on 2026-10-03 to make interesting, efficient gaits likely by mutation. They live in eight files, and each file's operators share one pick slot. A leg is a leaf limb (`rhythm::leaf_limbs`), the trunk is the nodes in no leg and a foot ends a leg of at least two bones; the head can sit anywhere. None has been audited or measured yet.

- `gait_legs.rs` (knees, ankles, feet and leg proportions): `lengthen_lower_leg`, `add_ankle_joint`, `bend_stick_leg_at_knee`, `lock_knee_extension`, `fold_leg_zigzag`, `straighten_leg_column`, `flatten_foot_sole`, `raise_heel_digitigrade`, `tendon_the_ankle`, `grow_forward_foot`, `set_leg_proportions`, `harden_or_pad_foot`, `lag_knee_behind_hip`.
- `gait_spine.rs` (a flexing back, tails and necks that balance a gait): `spine_flex_muscle`, `spine_lock_to_legs`, `split_spine_bone`, `grow_counterweight_tail`, `weight_tail_tip`, `plant_tail_prop`, `tail_swing_against_legs`, `neck_bob_muscle`, `split_neck_bone`, `stiffen_trunk_joints`, `loosen_trunk_joints`, `spine_phase_wave`, `elastic_spine`, `arch_back`.
- `gait_phase.rs` (whole gait phase patterns and duty factors; timing only): `quarter_beat_walk`, `diagonal_trot`, `lateral_pace`, `three_beat_canter`, `spread_gallop`, `half_bound`, `alternating_tripod`, `paired_leg_wave`, `double_ripple_wave`, `shared_duty_factor`, `fore_hind_duty_split`, `snap_leg_lags`, `reverse_leg_sequence`, `change_leading_leg`.
- `gait_muscles.rs` (antagonists, two-joint muscles, stance and swing muscles, springs): `reciprocal_extensor`, `hip_knee_strap`, `stance_swing_split`, `stance_swing_roles`, `improve_lever_arm`, `muscle_to_all_legs`, `stance_cocontraction`, `elastic_shank_tendon`, `lock_antagonist_pairs`, `push_off_muscle`, `repurpose_idle_muscle`, `catapult_release`, `second_hip_anchor`.
- `gait_symmetry.rs` (pairs, mirrored halves and repeated segments): `clone_best_leg`, `mirror_body_halves`, `repeat_equal_segment`, `repeat_segment_mirrored`, `equalize_leg_reach`, `average_leg_pair`, `share_program_alternating`, `share_program_wave`, `twin_leg_antiphase`, `mirror_hip_position`, `step_leg_along_trunk`, `share_joint_ranges`, `copy_foot_to_all_legs`.
- `gait_reflex.rs` (touchdown reflexes and bridge muscles that let one leg's landing fire another): `landing_starts_stroke`, `quick_lift_reflex`, `cross_leg_trigger`, `fore_to_hind_trigger`, `mutual_leg_trigger`, `reflex_wave_along_legs`, `alternate_legs_with_reflex`, `fore_hind_pairing`, `landing_flexes_trunk`, `touchdown_stiffener`, `spread_leg_reflex`, `trigger_chain_along_legs`, `stance_duty_with_reflex`.
- `gait_posture.rs` (the trunk carried clear of the ground, feet under the load): `straighten_leg_knee`, `foot_under_hip`, `hip_toward_mass_centre`, `centre_leg_rest_angles`, `raise_stance_height`, `crouch_legs`, `organs_to_trunk`, `organs_to_hip`, `sink_organ_mass`, `shorten_dragging_tip`, `widen_stance_fore_aft`, `zigzag_leg_bend`, `ground_hanging_foot`.
- `gait_plans.rs` (whole body plans in one move and moves between them): `quadruped_plan`, `hexapod_tripod`, `kangaroo_hopper`, `myriapod_wave`, `gibbon_swinger`, `counterweight_runner`, `shed_leg_pair`, `append_leg_pair`, `fuse_legs_into_one`, `split_leg_in_two`, `reduce_to_biped`, `pronking_stot`, `unguligrade_legs`.

A generative grammar whose rules children inherit was not built: a rule set would be a new part of the genome and of every save and archive, and a grammar used only as a seed source tied (`docs/rejected-ideas.md`). `segment_chain`, `limb_length_gradient` and `mirrored_limb_pair` apply such productions (repeat with a gradient, scale by place, add a mirrored pair) to the body itself, and the child inherits the result.

## Where big jumps come from

On the best 300 elites of a 120-generation save, rescored in one world, most jumps of 1.5x and 10 m or more changed no structure, and nearly any single gene group of the child gave most of the gain alone. 83 of 124 such parents reached 90% of their child's distance just by starting their own gait at another point of its cycle, and a start offset left the top 100 at a median 8% of their distance. Whether a gait catches depends on how it starts, so `shift_gait_start` moves every clock ahead by one common time.

## What the best creatures do

Filmstrips of the same save show one plan from rank 0 to rank 1000: a low triangle frame packed with muscles. The top elites hold 40 to 80% of their joints against a stop, as much steady muscle force as oscillating force, swing 0.1 rad and are airborne 40 to 67% of the time: a braced frame that hops. Mid-ranked elites hold 8 to 18% at a stop and slide. Every elite runs 0.3 to 0.7 rad per joint from its genome pose. `pose_joint_at_stop` and `brace_joint` let the skeleton take the braced pose itself.

## Audit

`examples/mutation_audit.rs <save> [elites] [seconds] [variants]` applies each operator to each of the best elites of a save (once per variant), within the nodes and muscles a child of that parent may add in breeding (`evolution::child_limits`), and scores parent and child in full 20 s trials in the save's world. It prints how often the operator fits the body, the share of the parent's distance the child keeps (median and 75th percentile), how many children keep 90% and how many beat their parent, the change in nodes and muscles, how many children would enter the global archive (faster than the elite that holds their cell), the distance those entrants add per 1,000 children and how many land in a cell nobody holds. `examples/operator_yield.rs` counts entrants the same way from a real generation (a generation dump).

Two saves, the best 500 elites of each and 3 variants per operator, on main at 0cee0e4 with the compound operators. Gen 590 is the owner's save from generation 590 (elites of 14.5 nodes and 55 muscles, median 46.6 m) and gen 1510 the one from generation 1510 (9.5 nodes and 39 muscles, median 34.1 m, under six world effects). Both were saved before the archives refined into body classes, and loading rebins them. "keeps" is the median share of the parent's distance. "enters" is the share of children faster than the elite of their cell in the global archive, and "new cell" the share that land in a cell nobody holds, which are not counted in "enters". "fits" shows gen 590 first, and "nodes, muscles" is the change on the gen 1510 save.

| operator | fits | nodes, muscles | keeps, gen 590 | enters | new cell | keeps, gen 1510 | enters | new cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parameter mutation alone (baseline) | 100% / 100% | +0.00, +0.05 | 0.44 | 5.07% | 16.33% | 0.95 | 11.20% | 25.40% |
| `graft_donor_limb` | 91% / 95% | +0.53, -4.71 | 0.19 | 5.49% | 33.70% | 0.59 | 4.49% | 47.16% |
| `copy_limb` | 89% / 81% | +1.34, +3.17 | 0.65 | 8.13% | 17.68% | 0.87 | 2.54% | 49.18% |
| `repeat_body_segment` | 7% / 9% | +2.00, +3.13 | 0.09 | 0.00% | 41.90% | 0.95 | 0.78% | 62.50% |
| `limb_phase_pattern` | 98% / 100% | +0.00, +0.03 | 1.00 | 28.78% | 5.70% | 1.00 | 27.73% | 15.60% |
| `prune_idle_limb` | 100% / 100% | -1.00, -3.59 | 0.93 | 20.33% | 10.27% | 0.99 | 23.07% | 38.13% |
| `limb_length_gradient` | 98% / 100% | +0.00, +0.03 | 0.12 | 2.51% | 24.58% | 0.71 | 4.47% | 39.13% |
| `symmetrize_limb_pair` | 76% / 73% | +0.00, +0.44 | 0.87 | 12.23% | 12.40% | 0.92 | 11.36% | 34.07% |
| `retime_gait_by_position` | 98% / 100% | +0.00, +0.03 | 0.05 | 3.19% | 23.01% | 0.03 | 1.33% | 38.47% |
| `brace_limb_chain` | 92% / 97% | +0.00, +0.04 | 0.01 | 3.61% | 26.23% | 0.01 | 6.13% | 38.59% |
| `phase_cluster_move` | 96% / 77% | +0.00, +0.03 | 0.21 | 11.18% | 17.78% | 0.38 | 6.30% | 34.17% |
| `reassign_bundle` | 80% / 97% | +0.00, +0.04 | 0.22 | 5.28% | 20.60% | 0.85 | 8.26% | 31.40% |
| `retune_limb_package` | 100% / 100% | +0.00, +0.03 | 0.99 | 26.53% | 7.87% | 0.99 | 23.40% | 20.40% |
| `grow_integrated_limb` | 99% / 100% | +0.00, -2.60 | 0.09 | 3.65% | 20.96% | 0.30 | 7.34% | 44.66% |
| `mirrored_limb_pair` | 97% / 99% | +0.01, -5.89 | 0.01 | 1.51% | 32.63% | 0.03 | 4.05% | 47.03% |
| `segment_chain` | 68% / 92% | +0.01, -9.08 | 0.05 | 1.87% | 26.82% | 0.02 | 6.17% | 41.73% |
| `transplant_limb_program` | 94% / 86% | +0.00, +0.54 | 0.98 | 18.83% | 9.06% | 0.98 | 17.42% | 23.33% |
| `transplant_gait` | 22% / 39% | +0.00, -0.41 | 0.37 | 4.17% | 14.58% | 0.85 | 6.62% | 36.33% |
| `trim_body` | 90% / 83% | -1.45, -7.11 | 0.48 | 13.26% | 12.52% | 0.97 | 21.79% | 42.45% |

The compound rows read as bigger moves than the rest. The four that rebuild a gait (`retime_gait_by_position`, `brace_limb_chain`, `mirrored_limb_pair`, `segment_chain`) keep a median of 1 to 5% of the parent's distance and their few good children carry the yield, while `retune_limb_package` and `transplant_limb_program` keep nearly all of it. The compound operators land in a cell nobody holds more often than the older ones (36% of their children against 29% on the gen 1510 save, 19% against 13% on the gen 590 save). A parameter mutation after a compound operator lowers how often its child enters the archive, so a compound child gets none. Averaged over the 13 operators the share that enters falls from 9.6% to 4.5% on the gen 1510 save and from 8.3% to 1.3% on the gen 590 save (`mutation_audit` prints a row `<operator> + parameter mutation` for every operator).

The audit starts from the best elites, which are the densest and the hardest to improve, so it is not what breeding sees: `trim_body` has 22% of its children entering the gen 1510 archive here and 0.3% in a real generation of that save, where its parents come from every rank (`examples/operator_yield.rs` counts the real generation). The audit ranks operators by how much of a gait they keep and how often a child beats the elite of its cell. It does not say which ones help the search.
