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

A generative grammar whose rules children inherit was not built: a rule set would be a new part of the genome and of every save and archive, and a grammar used only as a seed source tied (`docs/rejected-ideas.md`). `segment_chain`, `limb_length_gradient` and `mirrored_limb_pair` apply such productions (repeat with a gradient, scale by place, add a mirrored pair) to the body itself, and the child inherits the result.

## Where big jumps come from

On the best 300 elites of a 120-generation save, rescored in one world, most jumps of 1.5x and 10 m or more changed no structure, and nearly any single gene group of the child gave most of the gain alone. 83 of 124 such parents reached 90% of their child's distance just by starting their own gait at another point of its cycle, and a start offset left the top 100 at a median 8% of their distance. Whether a gait catches depends on how it starts, so `shift_gait_start` moves every clock ahead by one common time.

## What the best creatures do

Filmstrips of the same save show one plan from rank 0 to rank 1000: a low triangle frame packed with muscles. The top elites hold 40 to 80% of their joints against a stop, as much steady muscle force as oscillating force, swing 0.1 rad and are airborne 40 to 67% of the time: a braced frame that hops. Mid-ranked elites hold 8 to 18% at a stop and slide. Every elite runs 0.3 to 0.7 rad per joint from its genome pose. `pose_joint_at_stop` and `brace_joint` let the skeleton take the braced pose itself.

## Audit

`examples/mutation_audit.rs <save> [elites] [seconds] [variants]` applies each operator to each of the best elites of a save (once per variant), within the nodes and muscles a child of that parent may add in breeding (`evolution::child_limits`), and scores parent and child in full 20 s trials in the save's world. It prints how often the operator fits the body, the share of the parent's distance the child keeps (median and 75th percentile), how many children keep 90% and how many beat their parent, the change in nodes and muscles, how many children would enter the global archive (faster than the elite that holds their cell) and the distance those entrants add per 1,000 children. `examples/operator_yield.rs` counts entrants the same way from a real generation (a generation dump).

Two saves, the best 500 elites of each and 3 variants per operator, on the build that has the compound operators. Gen 590 is the owner's save from generation 590 (elites of 14.5 nodes and 55 muscles, median 46.6 m) and gen 1510 is the one from generation 1510 (9.5 nodes and 39 muscles, median 34.1 m, under six world effects). "keeps" is the median share of the parent's distance, "beats parent" the share of children faster than their parent and "enters archive" the share faster than the elite of their cell in the global archive. Each cell shows gen 590 first and gen 1510 second where two numbers share one cell.

| operator | fits | nodes | muscles | keeps, gen 590 | beats parent | enters archive | keeps, gen 1510 | beats parent | enters archive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parameter mutation alone (baseline) | 100% / 100% | +0.00 / +0.00 | +0.03 / +0.05 | 0.42 | 6% | 5.80% | 0.95 | 16% | 15.60% |
| `graft_donor_limb` | 91% / 95% | +0.19 / +0.53 | -5.34 / -4.71 | 0.18 | 5% | 5.20% | 0.56 | 8% | 7.79% |
| `copy_limb` | 89% / 81% | +1.22 / +1.34 | +3.19 / +3.17 | 0.66 | 9% | 8.73% | 0.87 | 4% | 4.01% |
| `repeat_body_segment` | 7% / 9% | +2.00 / +2.00 | +3.49 / +3.13 | 0.09 | 1% | 0.95% | 0.96 | 4% | 3.91% |
| `limb_phase_pattern` | 98% / 100% | +0.00 / +0.00 | +0.02 / +0.03 | 1.00 | 29% | 28.85% | 1.00 | 32% | 32.00% |
| `prune_idle_limb` | 100% / 100% | -1.00 / -1.00 | -4.26 / -3.59 | 0.93 | 21% | 21.53% | 0.99 | 35% | 35.13% |
| `limb_length_gradient` | 98% / 100% | +0.00 / +0.00 | +0.02 / +0.03 | 0.12 | 3% | 2.78% | 0.70 | 6% | 5.73% |
| `symmetrize_limb_pair` | 76% / 73% | +0.00 / +0.00 | +0.28 / +0.44 | 0.86 | 13% | 12.93% | 0.92 | 17% | 16.48% |
| `retime_gait_by_position` | 98% / 100% | +0.00 / +0.00 | +0.02 / +0.03 | 0.05 | 3% | 3.05% | 0.03 | 2% | 1.73% |
| `brace_limb_chain` | 92% / 97% | +0.00 / +0.00 | +0.02 / +0.04 | 0.01 | 4% | 3.90% | 0.01 | 10% | 9.79% |
| `phase_cluster_move` | 96% / 77% | +0.00 / +0.00 | +0.02 / +0.03 | 0.21 | 11% | 11.04% | 0.38 | 8% | 7.94% |
| `reassign_bundle` | 80% / 97% | +0.00 / +0.00 | +0.03 / +0.04 | 0.23 | 6% | 5.86% | 0.83 | 10% | 10.12% |
| `retune_limb_package` | 100% / 100% | +0.00 / +0.00 | +0.02 / +0.03 | 0.99 | 27% | 27.53% | 0.99 | 26% | 25.33% |
| `grow_integrated_limb` | 99% / 100% | +0.00 / +0.00 | -2.84 / -2.60 | 0.09 | 4% | 3.79% | 0.29 | 12% | 11.88% |
| `mirrored_limb_pair` | 97% / 99% | +0.03 / +0.01 | -7.30 / -5.89 | 0.01 | 1% | 1.37% | 0.03 | 7% | 6.95% |
| `segment_chain` | 68% / 92% | +0.05 / +0.01 | -11.43 / -9.08 | 0.05 | 2% | 1.67% | 0.02 | 8% | 8.42% |
| `transplant_limb_program` | 94% / 86% | +0.00 / +0.00 | +1.02 / +0.54 | 0.98 | 20% | 20.03% | 0.98 | 21% | 20.92% |
| `transplant_gait` | 22% / 39% | +0.00 / +0.00 | -0.27 / -0.41 | 0.38 | 4% | 4.46% | 0.84 | 9% | 8.66% |
| `trim_body` | 90% / 83% | -1.66 / -1.45 | -8.32 / -7.11 | 0.48 | 14% | 13.78% | 0.97 | 34% | 33.98% |

The compound rows read as bigger moves than the rest. The four that rebuild a gait (`retime_gait_by_position`, `brace_limb_chain`, `mirrored_limb_pair`, `segment_chain`) keep a median of 1 to 5% of the parent's distance and their few good children carry the yield, while `retune_limb_package` and `transplant_limb_program` keep nearly all of it. A parameter mutation after a compound operator lowers how often its child enters the archive, so a compound child gets none. Averaged over the 13 operators the share that enters falls from 12.9% to 6.7% on the gen 1510 save and from 8.6% to 1.4% on the gen 590 save (`mutation_audit` prints a row `<operator> + parameter mutation` for each).

The audit starts from the best elites, which are the densest and the hardest to improve, so it is not what breeding sees: `trim_body` has 34% of its children entering the gen 1510 archive here and 0.3% in a real generation of that save, where its parents come from every rank (`examples/operator_yield.rs` counts the real generation). The audit ranks operators by how much of a gait they keep and how often a child beats the elite of its cell. It does not say which ones help the search.
