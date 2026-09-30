# Physics

A creature is a tree of point masses (nodes) joined by rigid, massless bones. The state is the head's position and velocity, the neck's angle and angular velocity, and one relative angle and angular velocity per other bone. Node positions come from forward kinematics, so every bone keeps its exact length and every pose is valid.

The dynamics are Featherstone's articulated-body algorithm in planar spatial vectors, written in world axes about the head's position at the start of the step so the numbers stay small in single precision. The neck body floats freely. Every other bone turns about its pivot relative to its parent bone. Integration is semi-implicit Euler on the joint coordinates at 60 steps per second, each step taken as 2 substeps of 1/120 s. Trials last 20 s after a short settling phase.

The physics is the CUDA kernel, `shaders/warp_creature.cu`, and nothing else simulates creatures. The game needs an NVIDIA GPU with the CUDA driver and NVRTC. `src/warp_kernel.rs` packs each creature from `physics2::Model` and writes the kernel source with the constants of `physics.rs` and `physics2.rs` and the world's effects compiled in. Where the rules below say step, the kernel uses the substep's time step.

## Forces and rules

- Gravity, wind and air drag act on every node and bone. Bone drag is `AIR_DRAG (0.6) x length x width x speed x velocity`, limited so one step never more than halves the speed. It only takes energy away.
- Muscles pull only. The pull follows the waveform's shortening speed, times the muscle's stiffness and energy, with a light damper. Hill's force-velocity relation scales the active pull by `1 - v / v_max`, with `v_max` of 8 muscle lengths per second. Only active contraction is charged to the energy store (120 J, recovering half the missing energy per second, before environment effects).
- A muscle's force cap and energy store scale with the mass it drives: 100 m/s^2 times the lighter of the two subtrees it pulls together, never above 100 N.
- Every muscle has a tendon gene (0 to 1). A muscle stretched past its longest length is pulled back by a passive spring in parallel with it. The spring stores and returns the stretch energy and is not charged to the muscle.
- Passive joint damping with a 0.1 s time constant, sized to the inertia each joint moves.
- Joint limits are inelastic stops. A joint that would pass its limit within the step turns only as far as the limit. A joint forced 0.5 rad past its range breaks and ends the trial like a fall.
- Spin cap: a bone turning faster than 15 rad/s meets an implicit drag toward rest. The drag is a pure torque, so it changes no linear momentum.
- Ground contact: a substep is one articulated-body pass with the muscles, gravity, wind, drag and water, then one contact solve. Every node that would reach the ground within the substep gets a contact, at most the 4 deepest. The contacts are solved together at velocity level with the exact contact-space matrix and 2 sweeps of projected Gauss-Seidel from zero impulses, then 1 sweep that only takes back friction that would do positive work. A touching node may approach the ground only as fast as its gap allows, normal impulses only push, and friction stays within mu times the normal impulse and opposes sliding. There are no planting rounds, no warm start and no static friction factor.
- Momentum balance: after each substep the body's momentum equals its old momentum plus the external impulses. The difference from first-order integration is applied as one uniform velocity.
- First law in flight: a substep without ground contact may not gain more kinetic plus potential energy than the muscles, the wind and the tendons put in. The excess comes off the motion about the center of mass.

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

In the CUDA kernel the contact solve (detection, the matrix walk, the sweeps and the response) is about 40% of the instructions of a substep, the muscles about 16% and the articulated-body pass about 14%. Each walker keeps its matrix rows in registers, and the response to the contact impulses comes from the torques the walkers leave at each joint. About a fifth of the instructions are branches, compares and convergence bookkeeping. `docs/rejected-ideas.md` lists what was tried to make it cheaper.

## Substeps

Measured on 300 elites of an evolved population (60 Hz against the same kernel at 4x rate): 2 substeps give a median distance ratio of 1.03; 1 substep gives 0.05, and 1 substep with one planting round 0.79. Holding the muscle forces over both substeps gave 0.41. Random bodies gain nothing (median -0.05 m, best -0.01 m in 20 s).

## Audits

`examples/physics_audit.rs` prints what the GPU replay records per elite (contact-free steps, ground push, muscle energy store, broken joints). The kernel keeps no energy, friction or momentum ledgers. `tests/cuda_physics.rs` checks that bodies stay on the ground, joints stay in range, a body without drive neither travels nor rises, and a creature scores the same in any batch. `examples/first_generation.rs` scores a random population on the GPU (median, p99, best) and catches free propulsion. Run it after any physics change.
