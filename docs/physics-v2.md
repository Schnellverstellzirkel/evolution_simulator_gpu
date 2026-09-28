# Physics v2: a planar articulated tree in reduced coordinates

Date: 2026-09-28. Status: CPU prototype (`src/physics2.rs`), opt in with `EVOLUTION_PHYSICS=2`. The owner approved a new physics formulation on 2026-09-28 (`docs/hpc-assessment.md` section 11). This document records the formulation, what evolution exploited on the way and how each hole was closed, and the measurements.

## Formulation

A creature is a tree of point masses (its nodes) joined by rigid, massless bones. The state is the head's position and velocity, the neck's angle and angular velocity, and one relative angle and angular velocity per other bone. Node positions follow from forward kinematics, so every bone keeps its exact length and every pose is valid. The current physics's bone projection passes, parent-first rebuild, turn limit, whole-body lift and settling phase are gone.

Each bone is a rigid body carrying its child node's mass at its far end; the neck also carries the head. The dynamics are Featherstone's articulated-body algorithm in planar spatial vectors (angular part plus two linear parts), expressed in world axes about the head's position at the start of the step so the numbers stay small in single precision. The neck body floats freely (three degrees of freedom); every other bone turns about its pivot relative to its parent bone.

Forces:

- Gravity and wind on every node.
- Muscles keep the current model: a pull that follows the waveform's shortening speed, times the muscle's stiffness and its energy, a light damper, the force cap and the energy store. Hill's force-velocity relation scales the active pull by `1 - v / v_max`, with `v_max` = 8 muscle lengths per second (`EVOLUTION_HILL`).
- Passive joint damping with a 0.1 s time constant (`EVOLUTION_JOINT_DAMPING`), sized to the inertia each joint moves, implicit in the solve.
- Joint limits: an implicit spring and damper on the relative angle beyond the evolved range. A joint forced 0.5 rad past its range breaks, as today.
- Spin cap: a bone turning faster than `Limits::bone_spin` (40 rad/s) gets a pure torque back to it, implicit. A pure torque changes no linear momentum.
- Ground contact: after the step without the ground, every node that would reach the ground gets a contact. The contact impulses are solved together at velocity level by projected Gauss-Seidel: a touching node may approach the ground only as fast as its gap allows (a node already inside is pushed out by a fifth of its depth per step), normal impulses only push, and friction stays within mu times the normal impulse and opposes sliding. The contact-space matrix comes from the articulated inertias with one cheap bias-only pass per contact direction.
- Momentum balance: after each step the body's momentum must equal its old momentum plus the external impulses (gravity, wind, ground) times the air retention. First-order integration in joint coordinates misses this slightly each step; the difference is applied as one uniform velocity.

Integration is semi-implicit Euler on the joint coordinates.

## What evolution exploited, and the fixes

Each row is a hole found by evolving 5,000 creatures for 60 generations with 60 s trials and looking at the winner (`examples/filmstrip.rs` draws its trial and prints a momentum ledger: impulse from the ground and the wind against the body's change of momentum).

| exploit | symptom | fix |
|---|---|---|
| catapult | 19 m in one throw, then a fall | Hill's force-velocity relation bounds muscle power |
| super-grip friction | sleds at 4 to 6 m/s | friction sized from the step's start velocity could exceed mu N; replaced (see the next rows) |
| sticking contacts carrying the load of sliding ones | 30 m/s, ground impulse matches the gain | contacts are solved together, not one at a time |
| contact that switches on only once a node is inside | a node 4 cm deep pushed out with energy nothing paid for | contacts activate for nodes that would reach the ground within the step |
| spring that pulls before the node touches | contact force negative, contact dropped, node sinks | velocity-level contacts instead of springs |
| friction that reverses a slide | a node flung at 115 m/s in one step | projected Gauss-Seidel: friction can only oppose the slip |
| integrator error at high spin | nodes at 140 to 1,176 m/s, momentum made from nothing | absolute bone spin cap; momentum balance each step |

Tests in `src/physics2.rs`: bones keep their length; a free body keeps its momentum to first order in the step; a pulling muscle closes its joint; a passive body lands, rests, never rises above its start and does not travel; a passive body never gains energy on the ground.
