# Physics

A creature is a tree of point masses (nodes) joined by bones. The physics is the CUDA kernel `shaders/creature.cu`, and nothing else simulates creatures. One GPU thread simulates one creature from its start pose to the end of its trial in plain loops. `src/kernel.rs` packs each creature from `physics2::Model` into a flat record and writes the kernel source with the world's effects compiled in. The game needs an NVIDIA GPU with the CUDA driver and NVRTC.

## Step

The kernel uses position-based dynamics with small substeps (Mueller and others, "Small Steps in Physics Simulation", 2019). A trial runs 60 steps per second, and each step is 8 substeps (480 per second). A confirmation trial runs 240 steps per second with the same 8 substeps. Each substep:

1. Forces change the node velocities: gravity, wind, air and water drag, buoyancy, mud, brambles and the muscles.
2. Every node moves by its velocity.
3. Constraints move the nodes, once each in this order. Every bone gets its length back. Every joint goes back inside its range. Every node inside the ground moves out along the ground's normal, and friction takes back up to mu times that move of the node's slide over the substep.
4. Each node's velocity becomes its move over the substep divided by the substep. Then joint damping takes a share of every joint's turning speed.

Constraints only move nodes toward a valid pose, and friction only takes back sliding, so no part of a step adds energy. The muscles are the only source of work. Every node is its own ground contact and there is no limit on how many touch the ground.

## Forces and rules

- A muscle pulls its two anchors together. Its drive follows the rhythm's shortening speed over the step, times its stiffness and its energy store. Hill's relation scales the drive by `1 - v / v_max` with `v_max` of 8 muscle lengths per second. A light damper resists its length change. Only active shortening is charged to the store, which recovers. A muscle's force cap and store scale with the lighter of the two subtrees it pulls together, at most 100 N.
- A muscle stretched past its slack length is pulled back by a passive tendon, not charged to the store.
- A joint's range is a hard limit on the relative angle of a bone and its parent bone. A joint forced 0.5 rad past its range breaks and ends the trial like a fall.
- Joint damping with a 0.1 s time constant. It pushes the joint's nodes with equal and opposite velocities, so it keeps the body's momentum.
- Bone drag is `AIR_DRAG (0.6) x length x width x speed x velocity` at the bone's midpoint, shared by its two nodes, never more than half the bone's speed in one substep.

A fall (head below the neck base), a joint break, a non-finite position, or head acceleration averaged over about 0.1 s above 8 g ends scoring at the distance reached. In a recording the muscles then go limp and the trial plays to its end. Fitness is horizontal center-of-mass distance and nothing else. Ground contact, cadence, body height and lifted feet are behavior descriptors for the archive.

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

One thread per creature keeps each creature's nodes, bones and muscle state in the thread's local memory. On 30,000 evolved creatures of a fresh 20-generation game the kernel runs about 19M creature-steps per second (150M substeps). The muscles take about half of that time.

## Audits

`examples/physics_audit.rs` prints what the GPU replay records per elite (contact-free steps, ground push, muscle energy store, broken joints). `examples/fine_check.rs` replays an archive's best elites at the standard trial and at the fine one, and counts the elites that keep their distance.
