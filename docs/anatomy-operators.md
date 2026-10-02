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

A compound operator changes several parts of a body together and keeps them consistent with each other, so a child is a larger step than one edit and still has a chance of keeping its parent's gait. A child of the structural emitter that a compound operator made gets no parameter mutation after it, because the noise would only blur a move that was built to be coherent (it also saves the noise's breeding time). The operators that add bones close the motor ring themselves, in canonical bone order, with passive springs, so `repair` adds no random active muscles (a later `repair` would add more for every pair the canonical order moves). The operators that add nodes also remove the idlest limb tips (never one of the new parts), because bodies that only grow take more GPU lanes per creature (past 8 or 16 nodes a body takes twice as many). In the hub island a donor elite came from any isolated island, so the transplant exchanges programs between islands.

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

AUDIT_TABLE

The audit ranks operators by how much of a gait they keep and how often a child beats the elite of its cell. It does not say which ones help the search, and the elites it starts from are the ones that are hardest to improve.
