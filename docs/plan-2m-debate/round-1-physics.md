# Round 1: numerical formulation of the creature step

Domain: articulated-body dynamics with contacts, integrator error, exploit-proofing, and the instruction floor per creature-step. Sources: shaders/warp_creature.cu, src/warp_kernel.rs, src/physics2.rs, docs/physics.md, docs/rejected-ideas.md, 00-facts.md, hpc.md. No new measurements were run. Every number below that is not marked "measured" is a count from the code or an estimate, and each proposal names the measurement that would confirm it.

## 1. Where the step's cost goes today, counted

The current kernel is Featherstone's articulated-body algorithm (ABA) in reduced coordinates on a lane group of W = 8, 16 or 32 lanes (lane i owns node i and the bone that ends there). Per 1/60 s step it runs 2 substeps. Per substep, the serial chain of one warp is:

- Kinematics: 2 level loops over the tree depth (about 5 levels), 4 shuffles per level. About 10 dependent shuffle rounds, each 20 to 30 cycles.
- Muscles: up to 4 rounds of W muscles. Per round: 4 float4 global loads, 4 float4 shared loads of the node table, sqrt, reciprocal, cos, about 60 flops, 2 float4 shared stores, a syncwarp, then the end-list gather (shared loads) and another syncwarp. About 400 cycles of chain per round.
- ABA children-first pass: per level, about 60 flops in the lanes of that level, 3 float4 shared stores, a syncwarp, a warp reduce_max, a child loop of 3 float4 shared loads. Only the lanes of the current level do useful work; the other 6 or 7 of 8 idle.
- Forward pass: 5 levels of 3 shuffles plus 10 flops.
- Contact detection: ballots and up to 4 rounds of a group maximum.
- Contact matrix: each of up to 4 walkers climbs 5 levels; per level 2 float4 shared loads, 20 flops, a 4-entry shared row update, 2 syncwarps. Then the root block with 6 shuffles per contact.
- Projected Gauss-Seidel: 5 sweeps x 4 contacts, each update about 12 flops followed by 2 shuffles, all in one dependent chain: 40 dependent shuffles, about 1,000 cycles, with 1 lane doing the arithmetic.
- Response: another children-first pass with shared memory and syncwarps, and another forward pass.
- Integration, momentum balance (4 group sums, 3 shuffle rounds each), and the first-law check, which recomputes every muscle's length and tendon (a second muscle geometry pass) whenever any group in the warp is in flight.

From the measured 43 to 45M creature-steps per second and an issue rate near the older kernel's 19% (not re-measured on the new kernel), the warp issues about 239e9 x 0.2 / 45e6 = about 1,000 warp instructions per creature-step, or about 4,000 per warp-step for its 4 creatures at W = 8, 2,000 per warp-substep. That matches the structural count above within a factor of 1.5. The profile buckets (contact solve 45 to 50%, muscles 15 to 18%, ABA 10 to 14%) agree.

The two things that make this slow are not flops. They are idle lanes (a tree level occupies 1 to 3 of 8 lanes, the PGS occupies 1 lane per contact, the root block occupies lane 1) and dependent chains through shuffles, shared memory and syncwarps (latency 20 to 30 cycles each, about 100 of them per substep) with only 4 warps per scheduler to hide them.

## 2. The wall

1.1G creature-steps per second on 239G warp instructions per second is 217 warp instructions per creature-step at 100% issue. Today's lane-group kernel spends about 1,000 at about 20% issue. So the lane-group layout needs a 5x cut in instructions and perfect issue at the same time. That is not available: a tree level cannot use idle lanes, and the shuffle chains cap issue near 30 to 40% at 16 warps per SM. My estimate for the best lane-group kernel (section 4, proposals 2 and 4 applied) is about 400 to 500 warp instructions per creature-step at 35 to 45% issue, which is 170 to 240M creature-steps per second, 4 to 5x today. That is the ceiling of the layout, not of the physics.

The per-lane layout (one creature per lane, 32 per warp) changes the budget. 217 warp instructions per creature-step become 217 x 32 = 6,900 thread instructions per creature-step at 100% issue, or about 3,500 at 50% issue. The question my domain must answer is whether a full step of a 6-node, 15-muscle body fits in 3,500 thread instructions with no local-memory spills and with enough instruction-level parallelism to hold 50% issue at 12 to 16 warps per SM. Section 3 counts it: about 3,300 with 1 substep in maximal coordinates, about 6,400 with 2 substeps in reduced coordinates. So the target at today's 500 to 600 steps per creature needs per-lane layout, the cheaper formulation and 1 substep, all three. Two of the three give 0.5 to 0.6 of the target, which the search side can close by cutting steps per creature (section 5, proposal 3).

Bodies of 9 to 13 nodes with 30 to 50 muscles cost 2 to 3x more per step in any layout, and evolution grows bodies to that size within 30 generations (measured on the old kernel). The instruction floor for those bodies is 8,000 to 12,000 per step, so the rate falls to a third when the population matures unless the search holds bodies small. Muscle mass is in for that reason. This is the largest uncertainty in every estimate here.

## 3. The cheapest formulation: maximal coordinates with a direct tree solve

The creature is n point masses joined by n-1 massless rigid rods in a tree, with muscles as point forces at interpolated points on the rods. For this system:

- In maximal coordinates (x, y per node) the mass matrix is diagonal. Muscles need no Jacobian: a muscle end at fraction t of a rod puts (1-t) F on one node and t F on the other. Gravity, drag, wind and buoyancy are per-node forces. The current kernel's muscle scatter to spatial body forces and the two tree passes that turn them into joint torques disappear.
- The rod constraints are a tree. The constraint matrix A = J M^-1 J^T (n-1 by n-1) has a nonzero for every pair of rods that share a node. That graph is chordal: the rods at one node form a clique, and cliques chain along the tree. Eliminating rods children first creates no fill. The exact rod tensions come from one sparse LDL factorization and one solve, about 25 flops per rod plus 1 reciprocal per rod. For 5 rods that is about 130 flops. Featherstone's ABA on the same body is about 100 flops per bone and carries 3x3 spatial inertias for what are point masses. This is Baraff's linear-time Lagrange-multiplier method (SIGGRAPH 1996) applied to an acyclic graph; nothing new, only unused here.
- Contacts and joint limits are unilateral rows added on top. A contact is 2 rows on one node (normal, tangent). A joint limit is one row on the 3 nodes of two adjacent rods (the angle's Jacobian). Their Delassus matrix comes from one tree solve per row (about 30 flops per solve at 5 rods): 8 rows for 4 contacts is about 250 flops, against the walkers' 5 levels of shared memory and syncwarps.
- The 2k by 2k contact system (k at most 4) is solved directly: LDL of 8x8 is about 170 flops, then an active-set loop that drops a pulling contact or clamps a friction row past mu N and re-solves. For planted feet (the common case) it converges in 1 to 2 rounds. A direct solve gives exactly planted feet at velocity level; 4 cold PGS sweeps leave residual slip. PGS stays as a fallback for k > 2 re-solves.
- Drift: acceleration-level rod constraints let lengths drift by O(dt^2) per step. One position projection per step through the same factor (about 60 flops plus 1 sqrt per rod) removes it. The projection is mass-weighted, so it moves no momentum. It can add or remove a little energy; the first-law ledger already polices that in flight and the contact solve dominates on the ground.
- Joint damping and the spin cap are torques between adjacent rods, applied as forces on 3 nodes. The joint-limit inelastic stop is a unilateral row as above, with the same predicted-angle trigger the kernel uses now.

Invariants under this formulation:

1. Momentum only from external impulses: rod tensions are equal and opposite on their two nodes by construction, and every internal row (rod, joint limit, joint damping) has a Jacobian whose columns sum to zero. The momentum balance becomes a check, not a correction. Keep it as the cheap ledger it is.
2. Friction never does positive work: same impulse rule as today, plus the realized-pose ledger of proposal 1.
3. No energy gain in flight: same ledger as today, per node, about 10 flops per node.

Per-lane instruction count for 6 nodes, 5 rods, 15 muscles, 4 contacts, one substep:

| phase | thread instructions |
|---|---:|
| external forces per node and rod (gravity, drag, spin cap, water off) | 150 |
| muscles: 15 x (4 float4 constant loads, 4 float4 shared node loads, 50 flops, 4 shared force accumulations) | 1,200 |
| rod matrix assembly and LDL (rows and cliques) | 130 |
| dynamics solve, velocity update | 80 |
| contact detection, top 4 | 60 |
| 8 Delassus rows by tree solves, 8x8 assembly | 500 |
| direct contact solve with 1 re-solve | 300 |
| contact response through the tree | 60 |
| integrate, drift projection | 100 |
| ledgers (momentum check, friction realized work, first law) | 120 |
| metrics, fall, screen, sensors (once per step) | 150 |
| total per step at 1 substep | about 2,900 |
| total per step at 2 substeps | about 5,600 |

The same body in reduced coordinates per lane (ABA, walkers replaced by dense M^-1 through a 7x7 LDL, direct contact solve) counts about 3,200 per substep, 6,400 per step, because the muscle-to-torque mapping and the spatial inertias cost about 800 more per substep. Muscles are the largest block either way: 40% of the step at 15 muscles and 60% at 35. The muscle floor is about 60 to 80 instructions per muscle per substep (two anchor points, length, direction, relative speed, waveform with one cos, Hill, energy, tendon, four force accumulations) and I see no formulation that removes any of it, because each term is a rule of the game. Waveform caching and a branch-free waveform were measured slower.

Registers per lane at 6 nodes: state 24 (positions, velocities), rod factor about 15, node force accumulators 12, muscle constants streamed from global memory per muscle, muscle state (energy, offset, last waveform) in shared memory as [muscle][lane], loop temporaries about 40. About 90 to 100 registers, which allows 20 warps per SM, above the 16 the lane-group kernel holds. At 9 nodes the count is about 140 and spills; at 13 it is about 180. So per-lane serves bodies up to 7 or 8 nodes and a lane-group or a 2-lanes-per-creature class serves the rest. Dynamic indexing by parent rod and by muscle end node is handled by shared-memory tables laid out [index][lane] (conflict-free) rather than register arrays, which would spill to local memory. That is what killed the old per-thread kernel.

Serial depth per lane: the LDL and the tree solves are chains of about 5 dependent steps each, and the 8 Delassus solves are independent, so the scheduler sees ILP of 4 to 8 in the contact phase and about 3 elsewhere (2D vectors and independent muscles). No shuffles, no syncwarps, no cross-lane traffic except through the lane's own shared slots. At 16 to 20 warps per SM and ILP 3, an FMA chain of latency 4 to 6 issues near full rate; the SFU (cos, sqrt, reciprocal, about 4 per muscle and 1 per rod) and shared-memory loads (latency about 25) are the limit. My estimate of the reachable issue rate is 40 to 55%. At 45% and 2,900 instructions per step: 7.65e12 x 0.45 / 2,900 = 1.2G creature-steps per second for 6-node bodies. At 9 nodes and 35 muscles: about 6,000 per step, 570M per second.

## 4. Proposals, ranked by expected gain times confidence

### Proposal 1: close invariant 2 in the realized pose, then run 1 substep (about 1.8x, confidence medium)

The friction clean sweep forbids positive work against the predicted end velocity in the start pose. The realized end pose differs by the rotation of the foot's bone within the substep, and physics2's own comment says that without planting rounds the realized positive friction work on an evolved hopper was 1,725 J, cut to 5 J by two planting rounds. The GPU kernel has no planting rounds; its 2 substeps cut the rotation per solve in half, which cuts that error by about 4x, not to zero. So invariant 2 is only approximately held today. The measurement that says how much: from a replay's frames (they carry per-node contact forces and positions), sum friction impulse times realized tangential displacement per contact per step, over the 300 best elites. That ledger can be built today without a kernel change.

The change: after the step, compute each contact node's realized tangential displacement; if the friction impulse did positive work on it, take that work out of the motion about the center of mass the way the first-law check does (about 20 instructions per contact). Then the invariant holds regardless of the step size, and dropping to 1 substep at 60 Hz becomes an accuracy question rather than an exploit question. Second part, anchored friction: a planted foot keeps its plant position; the friction impulse targets the anchor rather than zero velocity, clamped by mu N, with a stored-energy bound like the tendon's so the anchor never returns more energy than it took. This is how Box2D, PhysX and MuJoCo keep feet from creeping at large steps, and it is what the kernel's "1 substep plus 1 planting round" measurement (0.79 median ratio) was reaching for by a more expensive route.

Estimate: 1 substep with the realized-work ledger and anchored friction runs at 1.8x the rate of 2 substeps (the ledger and anchor cost about 5% of a substep). Risk: the gaits change (a re-evolved population, a qd::VERSION bump) and the coarse step may tune brittle gaits. The 30 Hz history (elites kept 38% of their distance at 60 Hz) is the warning; the difference now is that the three invariants are enforced by ledgers, which the 30 Hz engine did not have.

Measurements: (a) realized friction-work ledger on 300 elites at most 1% of muscle work; (b) elites evolved at 1 substep for 30 generations re-tested at 2 and 4 substeps keep a median ratio of at least 0.9 (the 30 Hz test in the right direction); (c) first_generation.rs random-body free propulsion at most 0.05 m median; (d) size_report foot slip ratio near 1.0 for planted feet; (e) p2_speed rate.

### Proposal 2: maximal coordinates with the direct tree solve and direct contact solve (1.5 to 1.7x on the lane-group kernel; the enabler for per-lane)

Section 3 is the design. On the current lane-group kernel it replaces ABA (10 to 14%), the walker matrix (13 to 15%), the response passes (10 to 12%) and the kinematics (about 5%) with a level-parallel LDL (about 4%), 8 short tree solves (about 6%), one response solve (about 3%) and a drift projection (about 3%), and it replaces the 40-shuffle PGS chain (16 to 20%) with a redundant 8x8 direct solve in every lane (about 8% of issue, no chain). About 60% of today's time per substep, so about 1.6x, before proposal 1.

On a per-lane kernel it is what makes the register budget fit (section 3). Combined with proposal 1 and the per-lane layout it is the path to about 1.2G creature-steps per second on 6-node bodies, and it is the only path I see to the target on this laptop.

Risks: two weeks of reduced-coordinate work become the CPU reference only; joint limits and the spin cap need care as unilateral and pairwise rows; the drift projection must be shown not to feed energy (the flight ledger catches it, but a ledger that triggers every step is a sign of a bad integrator). It is new physics again and needs the owner's yes.

Measurements: rod length drift at most 1e-4 of length after 1,200 steps without projection and at most 1e-6 with it; the momentum ledger at rounding level without the balance step; first_generation.rs free propulsion; elite re-test ratio at 2x rate of at least 0.95 median for a population evolved on it; p2_speed.

### Proposal 3: cheap-fidelity screening, full-fidelity scoring (about 1.3x, confidence medium-high, needs the search domain)

Run the first 5 s at the cheap setting (1 substep, or a lagged Delassus matrix) and the screen decision on that. Survivors restart at full fidelity from t = 0 and only their full-fidelity score counts. Today's steps per creature: 0.8 x 300 + 0.2 x 1,200 = 480 step-equivalents. With the screen at half cost: 0.8 x 150 + 0.2 x (150 + 1,200) = 390. About 1.25x, and 1.35x if the survivors' first 5 s are not repeated but continued (which makes the score a mixed-fidelity number; I would not do that). This is the Hyperband pattern: cheap approximations at the low rungs, the real objective at the top. The rejected "cheaper contender checks" were the opposite direction (cheap scoring), which let exploits in; a cheap screen can only lose true positives, never admit an exploit.

Measurement: on 262k creatures scored both ways, the fraction of the full-fidelity top 1% and top 10% that the cheap screen removes; accept at most 2% and 5%. Determinism holds because the screen is a deterministic function of the seed.

### Proposal 4: flight steps at 1 substep (about 1.1 to 1.2x, confidence high, cheap)

A step with no contact candidate (no node would reach the ground within the step) has nothing stiff in it: muscles carry an implicit damper, joint limits and the spin cap are implicit. Run such steps as 1 substep of 1/60 s and steps with a candidate as 2 substeps. Hoppers and gallopers spend 30 to 50% of their steps in flight; walkers none. The decision is a deterministic function of the state, so determinism holds. The flight ledger polices the energy either way. Cost: an early exit in the substep loop.

Measurement: p2_speed on a generation-10 save; the elite re-test ratio unchanged (at least 1.0 median against 4x rate, as now 1.03).

### Proposal 5: lagged contact matrix (about 1.08x, confidence high, cheap)

The Delassus matrix depends on configuration only and changes by O(omega dt) per substep. Build it in the first substep of a step and reuse it in the second. Saves half of the walk (13 to 15%). The solve then satisfies the constraint against a slightly stale matrix; the foot sinks or slides by the error and the next step corrects it. Invariants unaffected (impulses are applied consistently; the friction clamp and ledgers act on the actual impulses).

Measurement: p2_speed; elite ratio; size_report foot slip.

### Proposal 6: periodic-gait extrapolation (about 1.2x, needs a genome change, confidence low)

A steady gait is a periodic orbit in the moving frame. If the creature's state at t + P equals its state at t up to a translation, on ground that is translation-invariant (flat, slope, wind, air, water, no bumps, no quake, no gaps, no hurdles, no ice patches), its future is that orbit repeated, and the score and every behavior metric are exact extrapolations. Detecting it after 2 matched periods would end most survivors' trials at 2 to 4 s instead of 20. The flaw: muscle periods are continuous genes, so the joint drive is quasi-periodic and the state never repeats. It only works if every muscle's period is a small rational multiple of one per-creature base period (1, 1/2, 1/3, 2), which is a genome change and a search decision. With that change, and half the survivors detected periodic by 4 s, steps per creature fall by about 17%, so about 1.2x. I rank it last because the score would no longer be bit-equal to a full replay (float drift over repeats), which the owner's replay rule may not allow.

Measurement: on 300 elites, the fraction whose state repeats within 1e-4 relative after quantized periods, and the distance error of extrapolation against the full run.

## 5. What the physics cannot give, and what I need

The physics levers above multiply to about 1.8 (proposal 1) x 1.6 (proposal 2 on lane groups) = 2.9x on the current layout, which is about 130M creature-steps per second, or 240k creatures per second at 540 steps per creature. That is not the target. The remaining 8x is the layout (per-lane, section 3: about 3 to 4x in useful work per warp instruction) and issue rate (about 2x), which are the HPC domain's tracks, and the steps per creature (proposals 3, 4 and search rungs: 1.3 to 1.6x), which is the search domain's. The three are independent and multiply: 2.9 x 3.5 x 2 x 1.4 = about 28x on paper, with the realistic range 10 to 25x because each factor lands below its estimate.

What I need from the others:

- HPC: measure the current kernel's warp instructions per creature-step and its issue rate (nsys sm__inst_executed and sm__issue_active over a fixed 262k wave). My 1,000 and 20% are inferred. Then confirm the per-lane register budget (section 3) with a stub kernel that only carries the state and the shared-memory tables, and measure the issue rate of a shared-indexed 5-rod LDL loop in isolation. That decides whether per-lane holds 45% issue or 25%.
- Search: the acceptable false-negative rate for a cheap screen (proposal 3); the steps-per-creature curve under any new rung; whether a per-creature base period with rational multiples is an acceptable genome change (proposal 6); how fast bodies grow to 9 nodes with muscle mass in, because per-lane serves at most 7 or 8 nodes.
- Data flow: sorting within a lane class by muscle count so that the per-lane muscle loop diverges by at most 10% (mean over max muscles per warp); a plan for 2 device classes (per-lane and lane-group) drawing from one ring in a deterministic order.
- Owner, through the chair: 1 substep with the ledgers (proposal 1) and maximal coordinates (proposal 2) are both physics changes with a qd::VERSION bump and a re-evolved population. The anchored-friction rule is a posture-adjacent rule (what a planted foot is) and needs approval.

## 6. Two measurements to run before any of this

1. The realized friction-work ledger from replays of the 300 best elites on the current kernel. If it is already near zero, proposal 1 is a step-size question only. If it is hundreds of joules, the current kernel has an open invariant that the search may already be using, and proposal 1 is a correctness fix first.
2. The evolve-at-1-substep, re-test-at-2-and-4 ratio on the current kernel with the ledger from (1) added. This single run (30 generations, one seed, about an hour) decides whether proposal 1's 1.8x exists. Without the ledger the same run tells whether the current friction rule already suffices at 1 substep, which the 0.79 planting measurement suggests it does not.
