# Anatomy mutation operators

The structural emitter picks one of 7 classic operators (split a bone, mirrored node, duplicate a leaf limb, retime, organ, phase shift, rescale) or one of the anatomy operators in `src/evolution/anatomy/`. All of them are on. It picks uniformly among the classic operators, the anatomy operators that have their own slot, and the shared slots, and tries again (up to four times) when the chosen operator does not fit the body. The owner wants more operator types and never fewer.

Every operator keeps the body limits and never touches the head or the neck.

| group (file) | operators |
|---|---|
| limbs (`limbs.rs`) | copy a whole branch with its muscles and timing; grow an actuated tip; split a bone with a narrow new joint and a muscle across it; fuse two aligned bones; move a branch to another node; rescale a branch and its strokes; graft a branch from another elite |
| junctions (`junctions.rs`) | split a crowded junction into two joints; merge two junctions; repeat a trunk segment with its limbs; grow a heel and a toe; grow a lever spur and move a muscle end onto it; reflect a branch so it bends the other way; start a joint near one of its stops (same stops, new starting pose); brace a joint against a stop with a small flex left |
| muscles (`muscles.rs`) | add a muscle across two joints; move a muscle end to a neighbouring bone; split a muscle; fuse two similar muscles; add an antagonist (opposite torque about the joint, half a cycle apart); swap two muscles' destinations; fan clustered anchors along a bone; replace a long muscle with a relay through a middle bone; copy one limb's actuation onto another limb; quiet a branch's strokes |
| rhythm (`rhythm.rs`) | move joint range from one joint to its neighbour; apply one change to two matching limbs; a phase wave down a chain; limb phase patterns (together, alternating, staggered); a limb's duty cycle around each contraction's middle; a coordinated touchdown reset; move organ mass between bones |
| extra (`extra.rs`) | copy a leg to the dragging end of the body; lift the dragging end; twin a limb in place; the same tip on two matching limbs; remove the idlest limb tip; merge the last two bones of a limb; and, sharing one pick slot, eight gentle operators that copy, swap or shift limb programs, trade cadence against stride, and change leverage, strength or the weakest muscle |
| controller (`controller.rs`) | limb stroke scale; limb posture shift; taper limb strength; copy limb rhythm; retune a muscle pair; release touchdown; snap limb phases; put a limb on another clock ratio; lock a limb back on the base clock; reflexes on a muscle, on all feet, and a shifted reflex reset. These share one pick slot. Starting the gait at another point of its cycle has its own slot |

A limb may run on its own clock at a ratio of the base clock (1/2, 2/3, 1, 3/2 or 2), so the whole gait still repeats.

## Where big jumps come from

On the best 300 elites of a 120-generation save, rescored in one world, most jumps of 1.5x and 10 m or more changed no structure, and nearly any single gene group of the child gave most of the gain alone. 83 of 124 such parents reached 90% of their child's distance just by starting their own gait at another point of its cycle, and a start offset left the top 100 at a median 8% of their distance. Whether a gait catches depends on how it starts, so `shift_gait_start` moves every clock ahead by one common time.

## What the best creatures do

Filmstrips of the same save show one plan from rank 0 to rank 1000: a low triangle frame packed with muscles. The top elites hold 40 to 80% of their joints against a stop, as much steady muscle force as oscillating force, swing 0.1 rad and are airborne 40 to 67% of the time: a braced frame that hops. Mid-ranked elites hold 8 to 18% at a stop and slide. Every elite runs 0.3 to 0.7 rad per joint from its genome pose. `pose_joint_at_stop` and `brace_joint` let the skeleton take the braced pose itself.

## Audit

`examples/mutation_audit.rs` applies each operator once to each of the best elites of a save and scores parent and child over 20 s. It prints the share of the parent's distance the child keeps, how often the child beats its parent, and the change in nodes and muscles. Operators that add a limb keep a median 2 to 3% of the parent. Operators that change timing or leverage keep a median 97 to 100%. The audit ranks operators by how much of a gait they keep. It does not say which ones help the search.
