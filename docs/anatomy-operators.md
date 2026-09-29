# Anatomy mutation operators

The structural emitter picks one of 7 classic operators (split a bone, mirrored node, duplicate a leaf limb, retime, organ, phase shift, rescale) or one of the anatomy operators in `src/evolution/anatomy/`. All of them are on. It picks uniformly among the classic operators, the anatomy operators that have their own slot, and the shared slots, and tries again (up to four times) when the chosen operator does not fit the body. The owner wants more operator types and never fewer.

Every operator keeps the body limits and never touches the head or the neck. A property test runs every operator on 160 grown bodies and validates the repaired result. Each operator has its own test of its defining effect.

| group (file) | operators |
|---|---|
| limbs (`limbs.rs`) | copy a whole branch with its muscles and timing; grow an actuated tip; split a bone with a narrow new joint and a muscle across it; fuse two aligned bones; move a branch to another node; rescale a branch and its strokes; graft a branch from another elite |
| junctions (`junctions.rs`) | split a crowded junction into two joints; merge two junctions; repeat a trunk segment with its limbs; grow a heel and a toe; grow a lever spur and move a muscle end onto it; reflect a branch so it bends the other way |
| muscles (`muscles.rs`) | add a muscle across two joints; move a muscle end to a neighbouring bone; split a muscle; fuse two similar muscles; add an antagonist (opposite torque about the joint, half a cycle apart); swap two muscles' destinations; fan clustered anchors along a bone; replace a long muscle with a relay through a middle bone; copy one limb's actuation onto another limb; quiet a branch's strokes |
| rhythm (`rhythm.rs`) | move joint range from one joint to its neighbour; apply one change to two matching limbs; a phase wave down a chain; limb phase patterns (together, alternating, staggered); a limb's duty cycle around each contraction's middle; a coordinated touchdown reset; move organ mass between bones |
| extra (`extra.rs`) | copy a leg to the dragging end of the body; lift the dragging end; twin a limb in place; the same tip on two matching limbs; remove the idlest limb tip; merge the last two bones of a limb; and, sharing one pick slot, eight gentle operators that copy, swap or shift limb programs, trade cadence against stride, and change leverage, strength or the weakest muscle |
| controller (`controller.rs`) | limb stroke scale; limb posture shift; taper limb strength; copy limb rhythm; retune a muscle pair; release touchdown; snap limb phases. All share one pick slot |

The muscle clock is shared by every muscle (`repair` copies the first muscle's period to all), so a limb cannot run at its own period. Operators change stroke, posture, duty and phase instead.

## Audit

`examples/mutation_audit.rs` applies each operator once to each of the best elites of a save and scores parent and child over 20 s. It prints the share of the parent's distance the child keeps, how often the child beats its parent, and the change in nodes and muscles. Operators that add a limb keep a median 2 to 3% of the parent. Operators that change timing or leverage keep a median 97 to 100%. The audit ranks operators by how much of a gait they keep. It does not say which ones help the search.
