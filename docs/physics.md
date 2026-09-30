# Physics

A creature is a tree of point masses (nodes) joined by rigid, massless bones. The state is the head's position and velocity, the neck's angle and angular velocity, and one relative angle and angular velocity per other bone. Node positions come from forward kinematics, so every bone keeps its exact length and every pose is valid.

The dynamics are Featherstone's articulated-body algorithm in planar spatial vectors, written in world axes about the head's position at the start of the step so the numbers stay small in single precision. The neck body floats freely. Every other bone turns about its pivot relative to its parent bone. Integration is semi-implicit Euler on the joint coordinates at 60 steps per second. Trials last 20 s after a short settling phase.

The scalar reference is `physics2::simulate_step_inner`. The WGSL kernel, the CUDA kernel and the fast CPU engine (`cpu_v2`) compute the same expressions. The two GPU kernels change together. `cpu_v2` is bit-equal to the reference (`tests/physics2_lanes.rs`).

## Forces and rules

- Gravity, wind and air drag act on every node and bone. Bone drag is `AIR_DRAG (0.6) x length x width x speed x velocity`, limited so one step never more than halves the speed. It only takes energy away.
- Muscles pull only. The pull follows the waveform's shortening speed, times the muscle's stiffness and energy, with a light damper. Hill's force-velocity relation scales the active pull by `1 - v / v_max`, with `v_max` of 8 muscle lengths per second. Only active contraction is charged to the energy store (120 J, recovering half the missing energy per second, before environment effects).
- A muscle's force cap and energy store scale with the mass it drives: 100 m/s^2 times the lighter of the two subtrees it pulls together, never above 100 N.
- Every muscle has a tendon gene (0 to 1). A muscle stretched past its longest length is pulled back by a passive spring in parallel with it. The spring stores and returns the stretch energy and is not charged to the muscle.
- Passive joint damping with a 0.1 s time constant, sized to the inertia each joint moves.
- Joint limits are inelastic stops. A joint that would pass its limit within the step turns only as far as the limit. A joint forced 0.5 rad past its range breaks and ends the trial like a fall.
- Spin cap: a bone turning faster than 15 rad/s meets an implicit drag toward rest. The drag is a pure torque, so it changes no linear momentum.
- Ground contact: every node that would reach the ground within the step gets a contact, at most the 4 deepest per step. The contacts are solved together at velocity level by projected Gauss-Seidel (8 sweeps, warm started). A touching node may approach the ground only as fast as its gap allows, normal impulses only push, and friction stays within mu times the normal impulse and opposes sliding. A foot that barely slides holds 25% harder (static friction). Each step is taken once and the solve is repeated with the contact velocities measured in the end pose (2 planting rounds of 4 sweeps), so a foot is planted against where the step leaves it.
- Momentum balance: after each step the body's momentum equals its old momentum plus the external impulses. The difference from first-order integration is applied as one uniform velocity.
- First law in flight: a step without ground contact may not gain more kinetic plus potential energy than the muscles, the wind and the tendons put in. The excess comes off the motion about the center of mass.

A fall (head below the neck base), a joint break, or head acceleration averaged over about 0.1 s above 8 g ends scoring at the distance reached and disables muscle force. Fitness is horizontal center-of-mass distance and nothing else. Ground contact, cadence, body height and lifted feet are behavior descriptors for the archive.

## Environment effects

Each effect changes the physics and never the objective. Levels are in `src/environment.rs`.

- Muscle energy (heat wave) and recovery (drought) scale the store and its recovery.
- Slope adds `slope * x` to the ground height. Wind adds a steady horizontal acceleration. Air scales the velocity retention. Grip scales friction. Gravity scales gravity.
- Mud lowers the contact floor by a sink depth. The sink scales the normal push, the friction budget and a horizontal drag. A node clear of the surface pays nothing.
- Gaps cut periodic pits of depth 2 m. Hurdles raise periodic steps every 3 m. Earthquake gives every creature its own bumps, with a phase and height derived from its id, so no gait can memorize one pattern. Ground roughness adds fixed bumps.
- Water shallows add drag and buoyancy. Ice patches lower friction on periodic stretches.
- Autochange environment raises one effect one level every 100, 50 or 20 generations, most benign first, and never lowers one.

`physics::ground` combines bumps, per-creature phase, slope, pits and steps in one sample. Constants are in `physics.rs`, `physics2.rs` and the packed kernel parameters.

## Cost

The contact solve is about 55% of a step. The dense contact matrix, the sweeps and the planting rounds each cost a quarter to a third of that section. `docs/rejected-ideas.md` lists what was tried to make it cheaper.

## Audits

`tests/physics_audit.rs` guards against solver-made energy and friction exploits: no elite may gain energy the muscles did not pay for, and friction may never do positive work along a slip. `examples/physics_audit.rs` prints what the GPU replay records per elite (contact-free steps, ground push, muscle energy store, broken joints). The energy, friction and momentum ledgers exist only in the CPU prototype. `examples/first_generation.rs` scores a random population on the GPU (median, p99, best) and catches free propulsion. Run it after any physics change.
