# Physics v2: a planar articulated tree in reduced coordinates

Date: 2026-09-28. Status: CPU prototype (`src/physics2.rs`), opt in with `EVOLUTION_PHYSICS=2`. The owner approved a new physics formulation on 2026-09-28 (`docs/hpc-assessment.md` section 11). This document records the formulation, what evolution exploited on the way and how each hole was closed, and the measurements.

## Formulation

A creature is a tree of point masses (its nodes) joined by rigid, massless bones. The state is the head's position and velocity, the neck's angle and angular velocity, and one relative angle and angular velocity per other bone. Node positions follow from forward kinematics, so every bone keeps its exact length and every pose is valid. The current physics's bone projection passes, parent-first rebuild, turn limit, whole-body lift and settling phase are gone.

Each bone is a rigid body carrying its child node's mass at its far end; the neck also carries the head. The dynamics are Featherstone's articulated-body algorithm in planar spatial vectors (angular part plus two linear parts), expressed in world axes about the head's position at the start of the step so the numbers stay small in single precision. The neck body floats freely (three degrees of freedom); every other bone turns about its pivot relative to its parent bone.

Forces:

- Gravity and wind on every node.
- Muscles keep the current model: a pull that follows the waveform's shortening speed, times the muscle's stiffness and its energy, a light damper, the force cap and the energy store. Hill's force-velocity relation scales the active pull by `1 - v / v_max`, with `v_max` = 8 muscle lengths per second (`EVOLUTION_HILL`).
- Passive joint damping with a 0.1 s time constant (`EVOLUTION_JOINT_DAMPING`), sized to the inertia each joint moves, implicit in the solve.
- Joint limits: an inelastic stop, like a contact. A joint that would pass its evolved limit within the step may turn only as far as the limit, held by a stiff implicit damper; one already past it is turned back by a fifth of the overshoot per step. A joint forced 0.5 rad past its range breaks, as today.
- Spin cap: a bone turning faster than 15 rad/s (0.25 rad per step) meets an implicit rotational drag toward rest. It is a pure torque, so it changes no linear momentum.
- Ground contact: after the step without the ground, every node that would reach the ground gets a contact, at most the eight deepest per step. The contact impulses are solved together at velocity level by projected Gauss-Seidel: a touching node may approach the ground only as fast as its gap allows (a node already inside is pushed out by a fifth of its depth per step), normal impulses only push, and friction stays within mu times the normal impulse and opposes sliding. The contact-space matrix comes from the articulated inertias with one bias-only pass per contact direction. Each step is then taken once, every contact's velocity measured in the end pose, and the solve repeated with the difference, so a foot is planted against where the step leaves it.
- Momentum balance: after each step the body's momentum must equal its old momentum plus the external impulses (gravity, wind, ground) times the air retention. First-order integration in joint coordinates misses this slightly each step; the difference is applied as one uniform velocity.
- First law in flight: a step without ground contact may not gain more kinetic plus potential energy than the muscles and the wind put in; the excess comes off the motion about the center of mass.
- Mud is not modeled yet.

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
| kick sled (seed 40, 408 m) | a flat body slides at 6.8 m/s; a short end bone whips into the ground at the 40 rad/s cap | at 0.67 rad per step the contact solve plants the tip in the start pose, and the end pose has the tip sliding forward with the planting friction still pushing: 2,887 of 5,761 N s of friction pushed a node along its own slip. Spin cap lowered to 15 rad/s (0.25 rad per step); the same body then goes 0.8 m and 0.5% of its friction points along the slip |
| spin cap that held bones at the cap | random passive bodies gained up to 48 J in one step | the cap was a damper toward the cap speed, so a contact that slowed a bone below it was resisted; it is now a drag toward rest past the cap, which only takes energy |
| joint limit passed within one step (seed 40, a 13-node elite) | 25 kJ of energy beyond 4.9 kJ of muscle work | the limit was a spring switched on from the angle predicted with the step's start velocity, so a joint driven hard within one step passed it unopposed and the spring then returned energy nobody paid for (one step: +20 J). A joint limit is now an inelastic stop, like a contact: a joint that would pass it may turn only as far as the limit, held by a stiff implicit damper, and one already past it is turned back by a fifth of the overshoot per step. The elite now gains nothing beyond its muscles' work (it falls at 3 s, its gait needed the spring) |
| feet planted in the start pose (the same elite) | 42% of its friction pushed a node along the slip it had after the step | the contact solve predicts each contact's velocity in the start pose. Each step is now taken once, each contact's velocity measured in the end pose (after the momentum balance), and the solve repeated with the difference. Friction along the slip: 0.0 N s on the three elites checked |
| a second bone at the head | its node sat on the neck's line and outside the dynamics (NaN once it touched the ground) | such a bone now turns against the neck, as its joint does in the current physics |

`examples/strobe.rs` replays one saved creature and prints the energy ledger (muscle work against the energy the body gained otherwise) and how much friction pushed a node the way it slid after the step, the signature of the kick sled.

## Animations

GIFs from `examples/creature_gif.rs`, 10 s from 20 s into a 60 s trial, 20 frames per second. Both creatures evolved from seed 40 with 5,000 creatures, 60 generations and 60 s trials (`examples/filmstrip.rs`).

- `physics-v2/v2-seed40.gif`: physics v2 with the fixes above. 8 nodes, 17 muscles, 2.9 kg, 90.4 m in 60 s (1.5 m/s). A compact body that hops forward in low bounces; no node touches the ground in 58% of steps. Fastest node 5.4 m/s.
- `physics-v2/v1-seed40.gif`: the current physics. 8 nodes, 12 muscles, 544.8 m in 60 s (9.1 m/s). A stiff frame that glides with its feet on the ground and jiggles; fastest node 59.8 m/s, at the current node speed cap.
- `physics-v2/exploit-kick-sled.gif`: the kick sled under the prototype before the spin fix (408 m).

The v2 GIF was rendered with the physics it evolved under: the 15 rad/s spin cap and the head-bone fix, before the planting pass and the inelastic joint stops. Under the current v2 that body falls within 3 s, because its gait leaned on the joint springs.

## How the winners move, and levers that are not fitness terms

None of the v2 winners so far is a lifelike walker. Seed 38 was a flat hopper, seed 39 a triangle that shuffles, seed 40 first the kick sled and after the fixes a compact body that bounces along in low hops. They share three traits: a low, compact or flat body that cannot fall under the head-below-neck rule, many strong muscles, and short, fast movements. Random first-generation bodies barely move under either physics (`examples/physics_probe.rs`, 2,000 bodies, 10 s: best 0.06 m under v2 and 0.51 m under the current physics), so evolution starts from twitches. Joint damping at 0.03 s or none, or Hill's relation off, moved the v2 best only to 0.08 to 0.12 m.

Physics-side levers, none of them a fitness term, in the order I would try them:

1. Muscle power scaled to the body. A muscle may pull 100 N and hold 120 J recovering at half its store per second, up to 60 W each; the seed-40 hopper's 17 muscles give a 2.9 kg body far more power per kilogram than any animal has. Force and energy store in proportion to muscle size, or muscles with mass, would make a cheap gait worth more than a powerful one.
2. A posture rule for the trial end, beside the head-below-neck rule, for example that the head may not touch the ground. Flat and tumbling bodies pass today's rule easily. This is a rule on when a trial ends, like a fall, so it is the owner's call.
3. Elastic tendons that store and return energy (backlog item 14): running and pendular walking gaits live on stored energy, and an inelastic world gives nothing back.
4. Less joint damping. The 0.1 s time constant makes limbs sluggish, so short, stiff strokes may win over swinging legs; it barely changes how far random bodies get, so this needs an evolution run to judge.

A first-law check backs these up in flight: a step without ground contact may not gain more energy (kinetic plus gravity's potential) than the muscles (force times change of length) and the wind put in, and the excess comes off the motion about the center of mass. A tree turning near the spin cap gains up to about 1% a step from the first-order integrator there. On the ground the same scaling would move planted feet, so it is off; random passive bodies with every bone at the spin cap gain at most 0.12% of their energy in a contact step.

## The GPU kernel

`shaders/physics2_creature.wgsl` runs one creature per lane and mirrors `physics2::simulate_step_inner` expression by expression; the CPU prototype computes the same expressions in the same order (reciprocals of 1/d, 1/duty, the root's inverse inertia and the step rate where the kernel multiplies), so the two drift apart only by rounding (fused multiply-adds on the GPU, and its sine and cosine).

Layout. `physics2::pack` renumbers nodes so that bone j ends at node j + 1 and writes each creature's starting state into the buffers the current kernel uses, so segments, repacking and the scheduler work unchanged: node records hold the head's position and velocity and each bone's angle and rate, plus the last contact forces; bone fields hold the pivot node, length, joint range and the child node's mass, radius and friction; muscle fields are today's.

Registers. Bodies of up to eight nodes keep every per-bone array in registers: a bone's parent and pivot are reached through select chains, the passes children first and parents first run over static bone indices, and each node has its own contact slot, so the contact matrix (lower triangle) and the Gauss-Seidel loops use static indices too. The muscles name their bones by data, so they gather positions and velocities and scatter forces through a per-lane table in workgroup memory (11 words per bone). Larger bodies index memory.

The contact solve is its own function (`contacts`), so a later change to it ports in one place.

### Measured on the RTX 4060 (2026-09-28)

Agreement with the CPU prototype (`examples/p2_agreement.rs`, 512 random bodies and three evolved elites): at 1 s every distance is equal; at 20 s all but two are equal and the worst gap is 0.98 m, on an active elite that travels 28 m. Falls agree to the step. Contact sequences are chaotic, so long trials of active creatures drift apart by rounding.

Rate (`eval-bench`, the first 100,000 creatures of `runs/evolved-3m-v26.evo`, 20 s trials, no screen, straight to the GPU engine):

| kernel | creatures/s | creature-steps/s |
|---|---:|---:|
| current (v1) | 170,120 | 147M |
| v2 | 30,106 | 27.2M |

v2 is 5.4 times slower per step. The contact solve is most of the cost. On 20,000 of the same bodies: eight contacts with 20 cold sweeps ran 4.1M creature-steps/s, four contacts with eight warm sweeps and four planting sweeps 22M, and one contact with one sweep 62M. Select chains for small bodies were slower than indexing memory (19M against 22M), so they are off. The v2 kernels use 168 registers (193 at 12 nodes), so 12 warps fit per SM. The defaults now are four contacts, eight warm-started sweeps and four planting sweeps, in both engines.

### The same evolution under v1 and v2

Seed 40, 100,000 creatures, 30 generations, 20 s trials, on the GPU with the production scheduler (`headless`, 4 rayon threads). v1 is main at f371e70 (tonight's slider fix), built in a separate worktree (`evolutionSimulator-v1cmp`) with a lane-0 energy and friction ledger added to its CPU engine (`examples/v1_ledger.rs`, not pushed). Distances of single creatures are CPU replays of 20 s trials; "median" in the first rows is the population median of the last generation.

| | v1 (main f371e70) | v2 |
|---|---:|---:|
| best archive distance | 17.15 m | 36.17 m |
| population median, last generation | 2.61 m | 2.71 m |
| median elite (archive rank 50%) | 2.51 m | 2.47 m |
| archive cells | 1,315 | 1,293 |
| QD score | 3,495 | 6,489 |
| evaluation per generation (generations 1 to 29) | 1.8 s | 11.0 s |
| wall time for 30 generations | 70 s | 395 s |

Per creature:

- `physics-v2/v1-gpu-seed40-best.gif`: v1 best, 5 nodes, 7 muscles, 3.7 kg, 17.15 m on the GPU and 15.49 m in its CPU replay. A small triangle that scoots forward in short hops, airborne 34% of the time, fastest node 17.8 m/s. Momentum: the bone passes (planted feet) put in 128 N s and contact friction took 124 N s back. Energy: muscles did 965 J, 9 J came from elsewhere. 13 of 288 N s of friction pushed a node along its slip.
- `physics-v2/v1-gpu-seed40-mid.gif`: v1 median elite, 5 nodes, 7 muscles, 15.6 kg, 2.51 m. A long flat plank that lies on the ground and inches forward; its fastest node reaches 47 m/s. Momentum: bone passes 15.1 N s, contact friction -14.6 N s. Energy: muscles did 9 J while 761 J came from elsewhere, that is from the position solver. No friction along the slip.
- `physics-v2/v2-gpu-seed40-best.gif` and `v2-gpu-seed40-mid.gif`: below.

### A GPU evolution under v2

Seed 40, 100,000 creatures, 30 generations, 20 s trials, on the GPU with the production scheduler (`headless`, about 10 s per generation). Best 36.2 m, median elite 2.7 m, 1,293 archive cells, QD 6,489. Ledgers from the CPU replay (`examples/strobe.rs`):

- `physics-v2/v2-gpu-seed40-best.gif`: the best, 7 nodes, 17 muscles, 5.0 kg, 36.4 m in 20 s (1.8 m/s). A compact hopper: a trailing leg kicks off, the body flies low, and no node touches the ground in 46% of steps. Ground impulse 8.16 N s against a momentum change of 8.16 N s; muscles did 4,042 J and 3.7 J came from elsewhere; no friction pushed a node along its slip.
- `physics-v2/v2-gpu-seed40-mid.gif`: the median elite (rank 678 of 1,357), 10 nodes, 24 muscles, 45 kg, 2.5 m in 20 s. A flat plank that lies on the ground and inches forward. Ground impulse -0.01 N s against -0.01 N s; muscles did 45 J and 88 J came from elsewhere, 43 J with six contacts and 23 J with eight: nodes left out of the four-contact solve sink and are pushed out. Its distance is the same at every contact bound, so this is resting jitter, not propulsion.

Tests in `src/physics2.rs`: bones keep their length; a free body keeps its momentum at 60 and 600 Hz; a pulling muscle closes its joint; a second bone at the head turns on its own joint; a passive body lands, rests, never rises above its start and does not travel; a passive body never gains energy on the ground; 400 random passive bodies with every bone at the spin cap gain at most 0.2% of their energy in any step; a limb whipped into the ground adds no energy; the GPU packing holds each creature's starting state; the kernel compiles for every capacity at standard and fine fidelity. `tests/physics2_gpu.rs` (ignored, needs the RTX 4060) compares the kernel with the prototype on 1 s trials.

## Muscle force scaled to the mass a muscle drives

A muscle's force cap and energy store now scale with its size: cap = 100 m/s^2 x the lighter of the two subtrees (a bone and everything it carries) that the muscle pulls together, never above the fixed 100 N, and the store scales the same way. A muscle's cross-section, and so its strength, follows the mass it moves. Before, a 100 N muscle drove a 0.05 kg limb at 2,000 m/s^2, turned a bone about a radian in one step and made momentum and energy the first-order integrator did not pay for: rank 881 of a seed 40 evolution gained 338 J against 26 J of muscle drain, and the momentum balance took 287 J of it back. With the scaling it gains 9 J and the balance removes 0.1 J.

The same GPU evolution (seed 40, 100k, 30 generations, 20 s, Vulkan), with the friction fix and without/with the scaling:

| | best | QD | median | top-50 median body |
|---|---|---|---|---|
| friction fix only | 63.5 m | 13,960 | 4.47 m | 5 nodes, 1.38 m of bone, 2.6 kg |
| muscle size too | 43.9 m | 8,746 | 3.39 m | 4 nodes, 1.67 m of bone, 4.2 kg |

`v2-muscle-size-gpu-seed40-best.gif` is the champion of the second run: 4 nodes, 9 muscles, 4.1 kg, 43.9 m in 20 s (2.2 m/s). Its ledger: muscle drain 2,400 J, energy the solver added 0.0 J, lost 3,258 J, momentum balance +103 J and -106 J, friction work +0.0 J. No elite of the run has unpaid energy (60 elites: gained 1,040 J against 37,637 J of muscle work) or friction that does positive work (14 J in total).
