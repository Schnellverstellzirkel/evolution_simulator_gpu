# Physics

A creature is a tree of point masses (nodes) joined by bones. The physics is the CUDA kernel `shaders/creature.cu`, and nothing else simulates creatures. One GPU thread simulates one creature from its start pose to the end of its trial in plain loops. `src/kernel.rs` packs each creature from `physics2::Model` into a flat record and writes the kernel source with the world's effects compiled in. The game needs an NVIDIA GPU with the CUDA driver and NVRTC.

## Step

The kernel uses position-based dynamics with small substeps (Mueller and others, "Small Steps in Physics Simulation", 2019). A trial runs 60 steps per second, and each step is 16 substeps (960 per second). A confirmation trial runs 120 steps per second with the same 16 substeps. Each substep:

1. Forces change the node velocities: gravity, wind, air and water drag, buoyancy, mud, brambles and the muscles.
2. Every node moves by its velocity.
3. Constraints move the nodes, once each in this order. Every bone gets its length back. Every joint goes back inside its range. Every node inside the ground moves out along the ground's normal, and friction takes back up to mu times that move of the node's slide over the substep.
4. Each node's velocity becomes its move over the substep divided by the substep. Then joint damping takes a share of every joint's turning speed.

Constraints only move nodes toward a valid pose, and friction only takes back sliding, so no part of a step adds energy. The muscles are the only source of work. Every node is its own ground contact and there is no limit on how many touch the ground.

## Forces and rules

- A node weighs `0.1 x (diameter / 0.08)^2` kg, from 0.02 to 10 kg. A bone weighs 4 x length^2 kg (length in m), half at each end. An organ adds its mass to the two ends of its bone by where it sits. A muscle weighs 0.05 kg plus 1 kg per metre of its slack length, half at each attachment, shared by that bone's two nodes by where the muscle attaches (`physics::nodes`).
- A muscle pulls its two anchors together. Its drive follows the rhythm's shortening speed over the step, times a quarter of its stiffness and times its energy store. Hill's relation scales the drive by `1 - v / v_max` with `v_max` of 8 muscle lengths per second. A light damper resists its length change. Only active shortening is charged to the store, which recovers. A muscle's force cap and store scale with the lighter of the two subtrees it pulls together: the cap is 200 m/s^2 times that mass, at most 200 N, and the store is the same fraction of 120 J. Muscles on the same two bones and on the same side of the joint split one strength, so stacked copies add timing and not force.
- A muscle stretched past its slack length is pulled back by a passive tendon, not charged to the store.
- A muscle's stroke is capped so that its rhythm never changes its length faster than 24 m/s (`Limits::muscle_speed`, applied when `physics2::Model` is built).
- A joint's range is a hard limit on the relative angle of a bone and its parent bone. A joint forced 0.5 rad past its range breaks and ends the trial like a fall.
- Joint damping with a 0.1 s time constant. It pushes the joint's nodes with equal and opposite velocities, so it keeps the body's momentum.
- Bone drag is `AIR_DRAG (0.6) x length x width x speed x velocity` at the bone's midpoint, shared by its two nodes, never more than half the bone's speed in one substep.

A fall (head below the neck base), a joint break, a non-finite position, or head acceleration averaged over about 0.1 s above 8 g ends scoring at the distance reached. In a recording the muscles then go limp and the trial plays to its end. Fitness is horizontal center-of-mass distance and nothing else. Ground contact, cadence, body height and lifted feet are behavior descriptors for the archive.

## Environment effects

Each effect changes the physics and never the objective. Levels are in `src/environment.rs`.

- Muscle energy (heat wave) and recovery (drought) scale the store and its recovery.
- Slope adds `slope * x` to the ground height. Wind adds a steady horizontal acceleration. Air scales the velocity retention. Grip scales friction. Gravity scales gravity.
- Mud lowers the contact floor by the mud depth. A node that sinks into it gets a larger friction budget, 9 times larger at 10 cm of sink (`MUD_GRIP` and `MUD_NORMAL`), and a horizontal drag of 2 per second at 10 cm (`MUD_DRAG`). A node clear of the surface pays nothing.
- Brambles drag every node that is not a foot, against its horizontal velocity, while its surface is within 1 cm of the ground. A foot ends a leg of at least two bones.
- Gaps cut periodic pits of depth 2 m. Hurdles raise periodic steps every 3 m. Earthquake gives every creature its own bumps, with a phase and height derived from its id, so no gait can memorize one pattern. Ground roughness adds fixed bumps.
- Water shallows add drag and buoyancy. Ice patches lower friction on periodic stretches.
- Autochange environment raises one effect one level every 100, 50 or 20 generations, most benign first, and never lowers one.

`physics::ground` combines bumps, per-creature phase, slope, pits and steps in one sample. Constants are in `physics.rs`, `physics2.rs` and the packed kernel parameters.

## Cost

One thread per creature keeps each creature's constants, bones and muscle energy in the thread's local memory, about 3.5 KB. The node positions, velocities and muscle forces of a body of 16 nodes or fewer are in shared memory, 48 KB per block, so the kernel runs two blocks of 128 threads per multiprocessor (the default of `EVOLUTION_WARP_CARVEOUT`, 100). A larger body keeps them in local memory too. The kernel was bound by the L1 cache: its request path was 95% busy and 88% of the warp stalls were on local and global loads, because the local lines of all resident warps do not fit in L1 and every store goes through to L2. On 224,695 evolved creatures in 3 s trials it ran 3.71M creature-steps per second with all state in local memory and 5.9M with the shared tables. Per GPU-busy second, 40,000 bodies of 17 to 32 nodes run at about 1M creature-steps (the old path) against 13M for bodies of 16 nodes or fewer.

Most of the time goes to the muscles. Computing their forces once per step instead of once per substep made the kernel 2.7 times faster, but then only 88% of the top elites kept their distance at the confirmation trial, against 95% to 99%, so the forces stay per substep.

Sixteen substeps were chosen because the standard trial then agrees with the confirmation trial. On a fresh game of 20 generations at 300k creatures per generation, 95% to 99% of the top 1,000 elites of an archive keep 80% of their distance at the confirmation trial, and the ratio of the two distances has a median of 1.00 with a tenth below 0.94 to 0.96. At 8 substeps 80% to 84% kept it, with a tenth below 0.5 to 0.7. The lane-group kernel that came before kept 45% to 60% at generation 40 and none late in a game.

## Audits

`examples/physics_audit.rs` prints what the GPU replay records per elite (contact-free steps, ground push, muscle energy store, broken joints). `examples/fine_check.rs` replays an archive's best elites at the standard trial and at the fine one, and counts the elites that keep their distance.
