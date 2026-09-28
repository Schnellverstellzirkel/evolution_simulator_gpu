# Anatomy mutation operators

Date: 2026-09-28. The owner asked for new structural mutations that make a good gait more likely, "in the simplest way possible". A Codex session proposed 30 ideas; Claude built a shared frame and four subagents implemented the operators in `src/evolution/anatomy/`. `EVOLUTION_ANATOMY` enables them (`all`, or a comma-separated list of names).

## What changes

The classic structural emitter picks one of 7 operators (split a bone, mirrored node, duplicate a leaf limb, retime, organ, phase shift, rescale). With the anatomy operators on it picks uniformly among those 7 and the enabled new ones, and tries again (up to four times) when the chosen operator does not fit the body. With the flag off, breeding draws exactly the same random numbers as before.

The operators change a working assembly of parts together:

| group (file) | operators |
|---|---|
| limbs (`limbs.rs`) | copy a whole branch with its muscles and timing; grow an actuated tip; split a bone with a narrow new joint and a muscle across it; fuse two aligned bones; move a branch to another node; rescale a branch and its strokes; graft a branch from another elite |
| junctions (`junctions.rs`) | split a crowded junction into two joints; merge two junctions; repeat a trunk segment with its limbs; grow a heel and a toe; grow a lever spur and move a muscle end onto it; reflect a branch so it bends the other way |
| muscles (`muscles.rs`) | add a muscle across two joints; move a muscle end to a neighbouring bone; split a muscle; fuse two similar muscles; add an antagonist (opposite torque about the joint, half a cycle apart); swap two muscles' destinations; fan clustered anchors along a bone; replace a long muscle with a relay through a middle bone; copy one limb's actuation onto another limb; quiet a branch's strokes |
| rhythm (`rhythm.rs`) | move joint range from one joint to its neighbour; apply one change to two matching limbs; a phase wave down a chain; limb phase patterns (together, alternating, staggered); a limb's duty cycle around each contraction's middle; a coordinated touchdown reset; move organ mass between bones |

Every operator keeps the body limits, never touches the head or the neck, and starts added muscles passive when neutral splits are on. A property test runs every operator on 160 grown bodies and validates the repaired result; each operator has its own test of its defining effect.

## How much of the parent a child keeps

`examples/mutation_audit.rs` applies each operator once to each of the 600 best elites of the evolved 3M checkpoint (generation 9, mean 6.0 nodes, 9.1 muscles) and scores parent and child with the CPU engine over 20 s. It leaves out the small parameter mutation that follows in breeding; the first row is that mutation alone.

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

## Decision

All 30 operators are on by default. `EVOLUTION_ANATOMY=0` restores the classic structural mutation exactly (same random draws), and a list of names enables only those, for experiments.
