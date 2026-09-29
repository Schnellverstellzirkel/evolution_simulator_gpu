# Anatomy mutation operators

Date: 2026-09-28. The owner asked for new structural mutations that make a good gait more likely, "in the simplest way possible". A Codex session proposed 30 ideas; Claude built a shared frame and four subagents implemented the operators in `src/evolution/anatomy/`. A second round the same day added 12 more operators, and the owner then asked for two that help a creature whose one strong leg drags the other end of the body. All 44 are on by default; eight of them share one pick slot (see "Second round"). `EVOLUTION_ANATOMY=0` turns them off for comparisons, and a comma-separated list of names enables only those. The owner's rule: no existing operator is ever removed.

## What changes

The classic structural emitter picks one of 7 operators (split a bone, mirrored node, duplicate a leaf limb, retime, organ, phase shift, rescale). With the anatomy operators on it picks uniformly among those 7, the anatomy operators, and one slot shared by the eight gentle second-round operators, and tries again (up to four times) when the chosen operator does not fit the body. With `EVOLUTION_ANATOMY=0`, breeding draws exactly the same random numbers as the classic mutation.

The operators change a working assembly of parts together:

| group (file) | operators |
|---|---|
| limbs (`limbs.rs`) | copy a whole branch with its muscles and timing; grow an actuated tip; split a bone with a narrow new joint and a muscle across it; fuse two aligned bones; move a branch to another node; rescale a branch and its strokes; graft a branch from another elite |
| junctions (`junctions.rs`) | split a crowded junction into two joints; merge two junctions; repeat a trunk segment with its limbs; grow a heel and a toe; grow a lever spur and move a muscle end onto it; reflect a branch so it bends the other way |
| muscles (`muscles.rs`) | add a muscle across two joints; move a muscle end to a neighbouring bone; split a muscle; fuse two similar muscles; add an antagonist (opposite torque about the joint, half a cycle apart); swap two muscles' destinations; fan clustered anchors along a bone; replace a long muscle with a relay through a middle bone; copy one limb's actuation onto another limb; quiet a branch's strokes |
| rhythm (`rhythm.rs`) | move joint range from one joint to its neighbour; apply one change to two matching limbs; a phase wave down a chain; limb phase patterns (together, alternating, staggered); a limb's duty cycle around each contraction's middle; a coordinated touchdown reset; move organ mass between bones |
| second round (`extra.rs`) | copy a leg to the dragging end of the body; lift the dragging end; twin a limb in place; the same tip on two matching limbs; remove the idlest limb tip; merge the last two bones of a limb; and, sharing one pick slot, eight gentle operators that copy, swap or shift limb programs, trade cadence against stride, and change leverage, strength or the weakest muscle |

Every operator keeps the body limits, never touches the head or the neck, and starts added muscles passive when neutral splits are on. A property test runs every operator on 160 grown bodies and validates the repaired result; each operator has its own test of its defining effect.

## How much of the parent a child keeps

`examples/mutation_audit.rs` applies each operator once to each of the 600 best elites of the evolved 3M checkpoint (generation 9, mean 6.0 nodes, 9.1 muscles) and scores parent and child with the CPU engine over 20 s. It leaves out the small parameter mutation that follows in breeding; the first row is that mutation alone. The second-round rows come from later runs of the same audit, which reproduced every earlier row exactly.

| operator | applied | child/parent median | p75 | keeps 90% | beats parent | nodes | muscles |
|---|---:|---:|---:|---:|---:|---:|---:|
| parameter mutation 0.035 (baseline) | 100% | 0.53 | 0.86 | 22% | 9% | +0.00 | +0.05 |
| split_bone | 100% | 0.00 | 0.04 | 2% | 2% | +1.00 | +2.09 |
| duplicate_mirrored_node | 100% | 0.03 | 0.23 | 6% | 3% | +1.00 | +2.59 |
| duplicate_limb | 100% | 0.03 | 0.20 | 3% | 2% | +1.00 | +4.65 |
| retime_rhythm | 100% | 0.36 | 0.90 | 25% | 11% | +0.00 | +0.05 |
| change_organ | 96% | 0.94 | 1.00 | 57% | 25% | +0.00 | +0.05 |
| phase_shift_group | 100% | 0.56 | 0.94 | 28% | 15% | +0.00 | +0.05 |
| rescale_body | 88% | 0.16 | 0.56 | 7% | 3% | +0.00 | +0.04 |
| remove_limb | 100% | 0.01 | 0.16 | 5% | 3% | -1.00 | -3.05 |
| remove_muscle | 79% | 0.30 | 0.94 | 27% | 14% | +0.00 | -0.94 |
| copy_limb | 100% | 0.02 | 0.15 | 6% | 3% | +1.85 | +4.11 |
| grow_actuated_tip | 100% | 0.01 | 0.08 | 3% | 1% | +1.00 | +2.48 |
| split_bone_actuated | 100% | 0.00 | 0.06 | 3% | 2% | +1.00 | +2.73 |
| fuse_bones | 24% | 0.01 | 0.39 | 8% | 6% | -1.00 | -0.76 |
| relocate_limb | 100% | 0.00 | 0.02 | 0% | 0% | +0.00 | +0.73 |
| reshape_limb | 95% | 0.20 | 0.68 | 13% | 6% | +0.00 | +0.05 |
| graft_donor_limb | 100% | 0.01 | 0.03 | 2% | 2% | +0.93 | +0.83 |
| split_crowded_joint | 72% | 0.00 | 0.02 | 1% | 1% | +1.00 | +2.04 |
| merge_branch_joints | 51% | 0.00 | 0.01 | 1% | 1% | -1.00 | -1.19 |
| repeat_body_segment | 100% | 0.01 | 0.04 | 2% | 1% | +2.85 | +5.68 |
| grow_heel_toe | 100% | 0.01 | 0.10 | 4% | 2% | +2.00 | +4.46 |
| grow_lever_spur | 100% | 0.01 | 0.03 | 3% | 2% | +1.00 | +2.03 |
| reverse_bend | 100% | 0.00 | 0.01 | 7% | 4% | +0.00 | +0.05 |
| add_biarticular_muscle | 100% | 0.49 | 0.95 | 28% | 14% | +0.00 | +1.04 |
| move_muscle_to_neighbor | 79% | 0.52 | 0.97 | 31% | 16% | +0.00 | +0.06 |
| split_muscle | 100% | 0.65 | 0.99 | 36% | 17% | +0.00 | +1.05 |
| fuse_similar_muscles | 2% | 0.81 | 0.94 | 33% | 17% | +0.00 | -1.00 |
| add_antagonist | 68% | 0.69 | 0.99 | 37% | 18% | +0.00 | +1.05 |
| swap_muscle_routes | 58% | 0.15 | 0.60 | 14% | 8% | +0.00 | +0.07 |
| fan_muscle_attachments | 93% | 0.36 | 0.88 | 22% | 11% | +0.00 | +0.05 |
| relay_muscle | 79% | 0.10 | 0.68 | 17% | 8% | +0.00 | +1.06 |
| copy_actuation_to_limb | 71% | 0.93 | 1.00 | 51% | 25% | +0.00 | +1.01 |
| quiet_muscle_group | 100% | 0.06 | 0.41 | 6% | 3% | +0.00 | +0.05 |
| redistribute_joint_flex | 100% | 0.02 | 0.48 | 15% | 7% | +0.00 | +0.05 |
| mutate_matching_limbs | 39% | 0.76 | 0.98 | 33% | 15% | +0.00 | +0.01 |
| chain_phase_wave | 100% | 0.14 | 0.53 | 9% | 4% | +0.00 | +0.05 |
| limb_phase_pattern | 72% | 0.48 | 0.92 | 27% | 15% | +0.00 | +0.05 |
| limb_duty_cycle | 100% | 0.45 | 0.84 | 20% | 11% | +0.00 | +0.05 |
| touchdown_package | 100% | 0.23 | 0.91 | 26% | 13% | +0.00 | +0.05 |
| redistribute_organ_mass | 28% | 0.97 | 1.00 | 61% | 32% | +0.00 | +0.01 |
| mirror_limb_timing | 32% | 1.00 | 1.00 | 80% | 39% | +0.00 | +0.00 |
| swap_limb_programs | 32% | 1.00 | 1.00 | 69% | 33% | +0.00 | +0.00 |
| copy_muscle_to_partner | 6% | 0.94 | 1.00 | 57% | 27% | +0.00 | +1.05 |
| twin_limb | 100% | 0.03 | 0.28 | 8% | 4% | +1.78 | +4.17 |
| grow_matching_tips | 39% | 0.02 | 0.33 | 10% | 6% | +2.00 | +5.01 |
| nudge_limb_phase | 100% | 0.50 | 0.87 | 22% | 11% | +0.00 | +0.05 |
| cadence_stride_trade | 76% | 0.43 | 0.80 | 19% | 7% | +0.00 | +0.04 |
| scale_muscle_leverage | 100% | 0.97 | 1.00 | 57% | 29% | +0.00 | +0.05 |
| scale_limb_strength | 100% | 0.52 | 0.88 | 23% | 12% | +0.00 | +0.05 |
| prune_weakest_muscle | 79% | 0.71 | 1.00 | 38% | 20% | +0.00 | -0.94 |
| prune_idle_limb | 100% | 0.03 | 0.46 | 8% | 5% | -1.00 | -2.95 |
| merge_leaf_bones | 74% | 0.00 | 0.02 | 4% | 2% | -1.00 | -1.11 |
| leg_to_dragging_end | 100% | 0.03 | 0.13 | 6% | 3% | +2.00 | +4.32 |
| lift_dragging_end | 54% | 0.18 | 0.73 | 16% | 10% | +0.00 | +1.06 |

Every operator that changes the skeleton, old or new, leaves a child a few percent of its parent's distance: evolved gaits are tuned to their exact bodies. The muscle and rhythm operators keep far more. Copying one limb's actuation onto another keeps a median 93% and beats the parent a quarter of the time; an antagonist keeps 69% and beats it 18% of the time, twice the rate of the parameter mutation alone.

## Search A/B

`examples/search_ab.rs`, seeds 38 to 47, 5,000 creatures, 60 s trials, CPU only. Both arms evaluate the same number of creatures.

| | classic, 40 gen | + anatomy, 40 gen | classic, 80 gen | + anatomy, 80 gen |
|---|---:|---:|---:|---:|
| best distance, mean (median) | 346 m (321) | 437 m (349) | 588 m (575) | 714 m (638) |
| QD score, mean (median) | 30,075 (19,447) | 38,568 (39,126) | 104,828 (60,823) | 121,293 (106,537) |
| seeds with a better best | | 7 of 10 | | 7 of 10 |
| archive cells, mean | 742 | 732 | 1,131 | 1,127 |
| population nodes / muscles at the last generation | 6.40 / 9.72 | 6.74 / 9.22 | 8.37 / 18.56 | 7.94 / 15.07 |
| top-50 muscles, mean (most) | 14.6 (40) | 12.3 (26) | 21.6 (78) | 15.5 (54) |
| CPU wall, 8 and 6 threads | 107 s | 136 s | 338 s | 418 s |

The classic arm reproduces the earlier baseline (346 m and 30,075 at 40 generations; 588 m and 104,828 at 80). With the anatomy operators the best distance is 21 to 26% higher on average, the QD score 16 to 28% higher (75 to 100% higher by median), and the bodies are leaner: the top 50 carry about 28% fewer muscles and the largest has 54 instead of 78.

### By group

The same 40-generation setup with one group enabled at a time (6 threads, so wall times compare only among these rows):

| enabled | best, mean (median) | QD, mean (median) | cells | top-50 muscles, mean (most) |
|---|---:|---:|---:|---:|
| classic only | 346 m (321) | 30,075 (19,447) | 742 | 14.6 (40) |
| + limb operators | 459 m (397) | 30,669 (30,503) | 802 | 16.4 (39) |
| + junction operators | 353 m (319) | 21,078 (21,352) | 741 | 14.9 (43) |
| + muscle operators | 346 m (358) | 24,992 (24,517) | 752 | 14.1 (44) |
| + rhythm operators | 322 m (324) | 31,986 (28,766) | 765 | 11.0 (29) |
| + all 30 | 437 m (349) | 38,568 (39,126) | 732 | 12.3 (26) |

The limb operators carry most of the distance gain and the rhythm operators give the leanest bodies. Junction or muscle operators alone lower the mean QD, but all 30 together give the best QD and nearly the best distance, so all stay on.

## In the game

3M GUI benchmark on `runs/evolved-3m-v26.evo`, one warm-up and two measured generations, `EVOLUTION_CPU_THREADS=0`, one run each:

| | end to end | breeding per generation | peak RSS |
|---|---:|---:|---:|
| classic operators | 253,559/s | 3.57 to 3.65 s | 10.2 GB |
| all anatomy operators | 241,687/s | 3.73 to 3.85 s | 10.2 GB |

The difference is inside the 10% spread between single runs; breeding costs about 5% more.

## Second round

The owner liked the first 30 operators and asked for more ideas. The audit above says muscle and rhythm operators keep a median 50 to 90% of the parent's distance and skeletal ones a few percent, so the second round favoured changes that keep the gait. Twelve operators:

- `mirror_limb_timing`: a limb of the same shape as another takes the other's program (each matching muscle's phase and duty) half a cycle later.
- `swap_limb_programs`: two limbs of the same shape exchange programs, which for equal shapes is the same as swapping their places.
- `copy_muscle_to_partner`: a muscle one limb has and its same-shaped partner lacks is copied onto the partner, timed to the partner's program.
- `twin_limb`: a limb is duplicated in place (same joint, same pose, same muscles, same phase), to diverge later.
- `grow_matching_tips`: the same actuated tip grows on both limbs of a same-shaped pair, mirrored when the limbs point opposite ways.
- `nudge_limb_phase`: every muscle on one limb moves 2 to 12% of a cycle earlier or later.
- `cadence_stride_trade`: the body clock's period and every stroke scale by one factor (0.7 to 1.4), so each muscle keeps its contraction speed and force.
- `scale_muscle_leverage`: both ends of a muscle across one joint move toward or away from the joint by one factor (0.5 to 2), stroke refitted.
- `scale_limb_strength`: the stiffness of every active muscle on one limb scales by one factor (0.6 to 1.6).
- `prune_weakest_muscle`: removes the weakest (stiffness times stroke) of three random muscles outside the motor ring.
- `prune_idle_limb`: removes the idlest of three random limb tips, with its muscles.
- `merge_leaf_bones`: the last two bones of a limb become one bone to the same foot.

The operators that added or removed bones closed the motor ring with passive muscles instead of the random active ones `repair_with` adds. In the audit this raised the upper tail a little (`grow_matching_tips` kept 90% of the parent in 10% of children instead of 6%) and left the medians at a few percent. An exact in-place twin that copied every muscle of the limb, also with both twins at half stiffness so the total drive stayed the parent's, still kept only 2 to 4%: the extra limb mass alone breaks a tuned gait, as `reshape_limb` and `rescale_body` already showed.

The audit and the search disagree. `mirror_limb_timing`, `swap_limb_programs` and `scale_muscle_leverage` keep a median 97 to 100% of the parent and beat it 29 to 39% of the time, more than any first-round operator. Yet the arm that adds the eight operators that change muscles and timing ("gentle" below) searched clearly worse. This was not investigated further. A likely cause is that near-copies of good parents take budget from the operators that explore, the limb group in particular.

### Search A/B, second round

`examples/search_ab.rs` as in the first round: seeds 38 to 47, 5,000 creatures, 60 s trials, CPU only, 3 threads. Arms:

- default: the first-round 30 operators (it reproduced the first round's numbers exactly, at 40 and at 80 generations)
- all: default plus the 12 new operators
- pruned: default without `relocate_limb` and the six junction operators (23 operators), to see whether dropping the operators with the worst audit helps
- gentle: default plus the 8 new operators that change muscles and timing (mirror, swap, copy muscle, nudge, cadence, leverage, strength, prune weakest muscle)
- skeletal: default plus the 4 new operators that add or remove bones (twin, matching tips, prune idle limb, merge leaf bones)

"Paired" is the geometric mean of arm over default across seeds, with the standard error of the mean log ratio.

40 generations:

| arm | best, mean (median) | QD, mean (median) | better best | better QD | paired best | paired QD | population nodes / muscles | top-50 muscles, mean (most) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default (30) | 436 m (349) | 38,568 (39,126) | | | | | 7.21 / 12.21 | 12.3 (26) |
| all (42) | 449 m (559) | 35,517 (35,280) | 4 of 10 | 5 of 10 | x1.05 (+0.04 ± 0.23) | x0.92 (-0.08 ± 0.27) | 7.70 / 13.43 | 19.5 (43) |
| pruned (23) | 513 m (550) | 35,175 (29,467) | 7 of 10 | 4 of 10 | x1.27 (+0.24 ± 0.15) | x0.96 (-0.04 ± 0.19) | 7.04 / 12.52 | 16.0 (29) |
| gentle (38) | 372 m (375) | 26,507 (23,957) | 6 of 10 | 5 of 10 | x0.94 (-0.06 ± 0.17) | x0.73 (-0.31 ± 0.22) | 7.34 / 11.97 | 10.9 (27) |

80 generations:

| arm | best, mean (median) | QD, mean (median) | better best | better QD | paired best | paired QD | population nodes / muscles | top-50 muscles, mean (most) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default (30) | 714 m (638) | 121,293 (106,537) | | | | | 7.94 / 15.07 | 15.5 (54) |
| all (42) | 646 m (696) | 108,203 (123,403) | 3 of 10 | 5 of 10 | x0.88 (-0.13 ± 0.13) | x0.92 (-0.09 ± 0.30) | 8.68 / 16.71 | 22.4 (80) |
| pruned (23) | 731 m (728) | 141,143 (132,369) | 5 of 10 | 6 of 10 | x0.99 (-0.01 ± 0.11) | x1.27 (+0.24 ± 0.19) | 7.54 / 15.43 | 17.9 (47) |
| gentle (38) | 497 m (481) | 68,421 (71,910) | 0 of 10 | 2 of 10 | x0.71 (-0.34 ± 0.08) | x0.63 (-0.46 ± 0.18) | 7.71 / 13.71 | 12.8 (42) |
| skeletal (34) | 744 m (740) | 119,459 (132,817) | 6 of 10 | 5 of 10 | x1.01 (+0.01 ± 0.09) | x1.06 (+0.06 ± 0.15) | 8.21 / 16.18 | 19.3 (48) |

Twenty more seeds (48 to 67) at 40 generations, default against pruned: pruned has the better best in 13 of 20 seeds, best x1.31 (+0.27 ± 0.12) and QD x1.09 (+0.08 ± 0.12). Over all 30 seeds at 40 generations pruning gives best x1.30 (+0.26 ± 0.09, 20 of 30 seeds better) and QD x1.04 (+0.04 ± 0.10).

A split of the gentle arm into its program half (mirror, swap, copy muscle) and its tuning half was started and stopped when the owner asked to wrap up. The program half's first five seeds at 80 generations averaged 494 m best and QD 90,481, against 660 m and 112,884 for default on the same seeds (better best in 2 of 5 seeds).

Wall times varied with other work on the machine, so they are only a rough guide. At 80 generations the pruned arm took 805 s against 1,005 s for default. Its population is smaller (7.54 against 7.94 nodes), which points the same way.

Reading:

- No group of new operators helped at a slot each. The gentle group lost at both lengths, and at 80 generations it lost best distance in every seed. The skeletal group was neutral and grew bodies (top-50 muscles 19.3 against 15.5). All twelve together lost at 80 generations and gave the largest bodies.
- Pruning helped search (best x1.30 over 30 seeds at 40 generations, QD x1.27 at 80), but the owner wants more kinds of mutation, not fewer, so no operator is removed. The pruned arm stays here as a measurement.

### What went on by default

The four skeletal operators went on by default, because they were neutral and the owner wants the variety. The default keeps table order, so it draws exactly like the skeletal arm above.

For the eight gentle operators the guess was that near-copies of good parents take budget from the operators that explore. So they now share one pick slot: the whole group is as likely as one other operator, and a draw of that slot picks one of the eight at random. A single check, `search_ab` with seeds 38 to 40, 5,000 creatures, 60 s trials and 40 generations, compared this against the step before (30 plus 4 skeletal):

| arm | best, mean (median) | QD, mean (median) | better best | better QD | paired best | paired QD | population nodes / muscles | top-50 muscles, mean (most) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 + 4 skeletal | 332 m (393) | 24,332 (23,947) | | | | | 7.08 / 12.01 | 18.1 (32) |
| + gentle 8 in one shared slot | 353 m (376) | 31,472 (31,586) | 2 of 3 | 3 of 3 | x1.10 (+0.09 ± 0.10) | x1.28 (+0.25 ± 0.07) | 7.17 / 11.73 | 13.8 (31) |

It does not lose: QD is higher in all three seeds and the top 50 carry fewer muscles. Three seeds are a small check, so this says the shared slot does no harm, not how much it helps. The gentle operators are on by default in the shared slot.

## Dragging end

The owner saw evolution settle on a creature with one strong leg at the front or the back while the other end drags on the ground, and asked for a mutation that helps it out of that. Two operators in `extra.rs`:

- `leg_to_dragging_end`: the working leg is the leg (a chain from a foot up to where the body branches) whose muscles drive most, by stiffness times stroke. The dragging end is the leg whose foot lies farthest from the working leg's top joint along x in the rest pose, or without another leg the farthest node. The working leg is copied there with `copy_branch`, mirrored front to back about its top joint, half a cycle apart (gallop or bound) or in phase (hop) at random. When the leg at the dragging end drives less than a quarter as much as the working leg, it is removed first, so the mass stays about the same.
- `lift_dragging_end`: adds one muscle from the dragging leg's top bone to the bone above its joint, anchored where its pull turns the leg so the foot rises, and timed like the working leg's strongest muscle, so the end lifts while the working leg pushes.

In the audit the copy keeps a median 3% of the parent, like the other operators that add a limb (`copy_limb` 2%, `twin_limb` 3%). The lifting muscle keeps 18% and beats the parent 10% of the time. One more arm of the same check adds both to the shared-slot default:

| arm | best, mean (median) | QD, mean (median) | better best | better QD | paired best | paired QD | population nodes / muscles | top-50 muscles, mean (most) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared-slot default | 353 m (376) | 31,472 (31,586) | | | | | 7.17 / 11.73 | 13.8 (31) |
| + both dragging-end operators | 260 m (265) | 32,888 (33,227) | 1 of 3 | 1 of 3 | x0.71 (-0.34 ± 0.37) | x1.05 (+0.05 ± 0.11) | 8.05 / 14.49 | 14.3 (26) |

Per seed, best and QD: seed 38 422 m / 36,415 against 158 m / 33,227; seed 39 376 m / 31,586 against 265 m / 30,539; seed 40 260 m / 26,416 against 357 m / 34,898. The mean best is lower, mostly from seed 38, while seed 40 gained and the mean QD rose. The paired best is less than one standard error below zero, so three seeds do not show a clear loss. The population grew by almost one node and nearly three muscles. The owner asked for these operators, so they are on by default; the owner judges them in the game.

## Decision

All 44 anatomy operators are on by default, on top of the 7 classic ones. Each has its own pick slot except the eight gentle second-round operators (`mirror_limb_timing`, `swap_limb_programs`, `copy_muscle_to_partner`, `nudge_limb_phase`, `cadence_stride_trade`, `scale_muscle_leverage`, `scale_limb_strength`, `prune_weakest_muscle`), which share one (`SHARED_SLOT` in `src/evolution/anatomy/mod.rs`). No operator is removed. `EVOLUTION_ANATOMY=0` restores the classic structural mutation exactly (same random draws), and a list of names enables only those, for experiments.

## Third round: controller operators (2026-09-29)

The owner asked for more operator types, starting with an antagonist pair and operators on the controller of a whole limb. `add_antagonist` already does the antagonist pair (a muscle from the child bone to a bone on the far side of the joint, checked to turn it the other way, half a cycle out of phase), so no new operator was written for it. It applies to 98% of the elites of the new audit checkpoint. The clock is shared by every muscle (`repair` copies the first muscle's period to all), so a limb cannot run at its own period. "Slow or speed a limb" is done through stroke, posture and duty instead. Seven operators in `src/evolution/anatomy/controller.rs`, all on by default, all in one new shared pick slot:

- `limb_stroke_scale`: every active muscle on a limb scales its stroke about its middle by 0.6 to 1.6.
- `limb_posture_shift`: both ends of every stroke on a limb move by 5 to 15% of the stroke, up or down.
- `taper_limb_strength`: stiffness rises or falls along the limb's muscles by a factor of 1.2 to 1.8 from the root to the tip.
- `copy_limb_rhythm`: the rhythm (phase, duty, touchdown reset) of one limb goes onto a different limb of any shape, a quarter, half or three quarters of a cycle later. `mirror_limb_timing` only pairs limbs of the same shape at half a cycle.
- `retune_muscle_pair`: two active muscles across the same pair of bones are set half a cycle apart or into one phase.
- `release_touchdown`: clears the touchdown sensors of a limb (the reverse of `touchdown_package`).
- `snap_limb_phases`: the phases of a limb's muscles are rounded to eighths of a cycle from its first muscle.

Audit (`examples/mutation_audit.rs`, 120 best elites of a 40-generation `search_ab` save, seed 38, 5,000 creatures, 20 s trials, CPU; parent median 4.10 m, 8.1 nodes, 16.9 muscles). Elites of this save are insensitive to gentle changes: the parameter mutation alone keeps a median 0.99, so the audit cannot rank gentle operators.

| operator | applied | child/parent median | keeps 90% | beats parent |
|---|---:|---:|---:|---:|
| parameter mutation 0.035 (baseline) | 100% | 0.99 | 83% | 34% |
| add_antagonist | 98% | 1.00 | 84% | 28% |
| nudge_limb_phase | 100% | 0.99 | 85% | 28% |
| limb_stroke_scale | 100% | 1.00 | 80% | 34% |
| limb_posture_shift | 100% | 1.00 | 98% | 39% |
| taper_limb_strength | 100% | 1.00 | 89% | 35% |
| copy_limb_rhythm | 98% | 0.98 | 74% | 32% |
| retune_muscle_pair | 100% | 1.00 | 82% | 38% |

Search A/B (`search_ab --checks`, 5,000 creatures, 60 generations, 20 s trials, seeds 38 to 46, the same binary with `EVOLUTION_ANATOMY` listing the 44 earlier operators for the base arm). Five of the operators with a slot each: best distance x0.71 (one base seed reached 48 m), QD x0.94, better best in 4 and better QD in 5 of 9 seeds. The same five plus `release_touchdown` and `snap_limb_phases`, sharing one slot: best x1.50 (again driven by that seed), QD x1.02, better best in 6 and better QD in 5 of 9 seeds, cells x1.02. That is neutral, so the seven stay, in a shared slot as the eight gentle operators do. There are now 51 operators.
