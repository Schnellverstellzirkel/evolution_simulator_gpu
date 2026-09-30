# Round 7, physics: the cheapest physics that keeps the spirit

The premise changed: today's rules are not the reference, the spirit is. The spirit as gates: random bodies gain no distance, elites of the new physics hold under a finer-rate retest, planted feet do not slide, no energy from nowhere, the gaits are efficient interesting movers, one build gives one search per seed. Everything below is counted in warp instructions per creature-step (WI) on a per-lane W = 2 layout with the tree baked per plan, against the 415 WI baked estimate for today's rules at 2 substeps on the p50 body (8 nodes, 19 muscles).

## 1. What a muscle needs to be

The spirit of a muscle is a pull-only actuator with a rhythm and a cost, bounded in power. Evolution finds efficient gaits when four things hold: the actuator cannot push (so bodies must use gravity and the ground), its force is bounded and falls with shortening speed (so it cannot catapult), its rhythm is a few genes with a phase (so gaits are periodic and heritable), and using it costs something that comes back slowly (so a wasteful gait weakens). Everything else in today's muscle is implementation, and most of it costs instructions every substep.

Today's muscle, per substep, about 70 instructions: two anchor points interpolated along two rods (8 FMA for positions, 8 for velocities), length and direction, the relative speed, the waveform's shortening speed from the stored previous waveform value, a drive proportional to that speed times a stiffness gene times the muscle's own energy store, Hill, a damper, the cap, the energy store update, the tendon, the ledger terms, and four force accumulations. Plus a state word per muscle (energy, drive target) read and written in L2 every substep, plus a 64 B record.

The muscle I propose, and what each removal costs the spirit:

- Node to node. A muscle joins two nodes, not two points along rods. Length is |p_b - p_a|, the relative speed is (v_b - v_a) . d, and the force goes to two nodes. Removes the anchor genes and 20 instructions per substep and halves the force accumulations. Leverage still evolves: a muscle that should attach mid-bone attaches to a node placed there, and the add-node operators exist. Karl Sims's creatures and every spring-mass walker since use node-to-node actuators.
- Activation, not a velocity target. Force = cap x strength x a(t) x E x max(0, 1 - v / v_max), pull only. a(t) in [0, 1] is the rhythm: a trapezoid of period, phase and duty (frac, two compares, one clamp, no cosine, no stored previous value). Hill stays, because it is the power bound that keeps catapults out. The damper stays (one instruction). The "stiffness" gene becomes "strength" (a fraction of the cap). The waveform derivative, the cosine and the per-muscle drive-target state go.
- One energy store per creature, not per muscle. Work = force x shortening, summed over muscles into one store that recovers at half the deficit per second, and every muscle's force is scaled by the creature's store. This is stamina, which is what the spirit wants (a wasteful gait weakens the whole animal), and it removes the per-muscle state word entirely: the muscle has no state. The cap scaled by the driven mass and the muscle mass stay, since they are precomputed and they are the pressure that holds muscle counts down.
- No tendon on the muscle. Elastic energy storage is what makes hopping efficient, so it must exist, but a spring on a light node is either too soft to matter or unstable at 1/60 s (round 4: omega DT 4.7), and a compliant row per stretched tendon is the most expensive fix. Put the elasticity where a row already exists: the joint limit gets a compliance gene (a ligament). A joint driven into its range's end stores energy in the limit row and returns it, implicitly and stably, at zero extra rows. The ledger treats it as it treated the tendon.
- Per-limb clocks with touchdown reset stay. They are the only feedback in the game, they cost about 2 instructions per muscle per step amortized, and their state is one word per limb (at most 8 per creature), not per muscle.

Cost per muscle: per step the rhythm (6) and the record load (2 x 16 B); per substep direction and length (7 with one rsqrt), relative speed (5), force (6), work (2), accumulation (4): 24. At 2 substeps and 19 muscles: 19 x 54 / 2 lanes / 16 = 32 WI, against about 100 in the baked estimate. At 4 substeps the geometry can be held for the step and only the velocity terms (8) recomputed: 19 x (30 + 4 x 8) / 32 = 37 WI.

## 2. What the contact solve needs to be

The spirit needs: a touching node cannot enter the ground, friction stays inside the cone and never does positive work, and a planted foot stays where it planted. It does not need the exact coupled solve. Today's solve forms the 8 x 8 Delassus matrix through the rod factor (about 100 WI at 2 substeps with the solve) because the coupled solution is exact for the 4 deepest nodes. The cheap honest form is per-node impulses with the rod solve as the coupling:

1. For every node that would reach the ground within the substep (no deepest-4 selection: every node, 8 x 10 instructions), a normal impulse against its own mass to stop the approach, then a friction impulse against its own mass toward the anchor (or zero slip), clamped to mu times the normal and by the clean rule.
2. The rod solve (the chordal LDL, exact) redistributes those impulses through the body.
3. One more pass of 1 and 2 with the residual approach and slip.

A foot's own-mass impulse is too small by the coupling fraction m_foot / (m_foot + what the rod pulls in); the second pass takes the residual down by that fraction squared, and the anchor pulls any leftover back next substep. The friction rule holds per node exactly, so friction never does positive work by construction; the momentum ledger and the angular ledger see only external impulses; the first law in flight is unchanged. Cost: 2 x (80 + one rod solve of 56) per creature per substep, about 9 WI per substep, against 50 WI at 2 substeps for the coupled solve. Joint limits and the spin cap become the same kind of per-joint impulse (each joint's own effective inertia, the ligament compliance folded in): 8 WI per step instead of rows in a Delassus matrix.

What could go wrong: on a three-foot landing the per-node passes leave a transient penetration (p99 to be measured, gate 2 mm) and a transient slip that the anchor removes next substep (gate: size_report slip ratio 0.95 to 1.05 on the new elites). Projected Jacobi is not this: Jacobi solved coupled rows from stale velocities and clamped the overshoot; this solves each node against its own mass, never overshoots, and lets the exact rod solve do the coupling.

## 3. Integrator and substeps

Symplectic Euler on node velocities and positions, rods exact by the LDL each substep, one drift projection per step with contact nodes at infinite mass, the four ledgers per step (momentum, angular momentum about the center, realized friction and projection work, first law in flight). The projection and the angular ledger stay because the spirit's "no energy from nowhere" includes "no momentum from nowhere", and they cost 20 WI per step together.

The substep rule follows Macklin's small-steps result (SCA 2019): substeps beat iterations. With one cheap pass per substep the honest choice is more substeps, not a smarter solve. The rotation error that made 1 substep dishonest is (omega HS)^2: at 1/240 s it is 16x smaller than at 1/60 s, so the anchors may become a safety net rather than the mechanism, and the honesty ladder inverts: evolve at 2 substeps of 1/120, retest at 8. Three candidates, all state-only and deterministic:

| variant | muscles | rods (factor and solve per substep) | contacts, limits, spin | forces, damping | ledgers, projection | integrate, metrics | total WI | over 415 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 substep of 1/60 | 30 | 16 | 9 | 5 | 20 | 8 | about 90 | 4.6x |
| 2 substeps of 1/120 | 32 | 32 | 18 | 9 | 24 | 8 | about 125 | 3.3x |
| 4 substeps of 1/240 | 37 | 64 | 36 | 18 | 32 | 8 | about 195 | 2.1x |

The rod rows dominate at 4 substeps because the factor is rebuilt every substep as rod directions turn. A lagged factor (rebuilt once per step, solves every substep) is exact to first order and cuts the 4-substep row to 40 WI and the total to about 170 (2.4x); the projection at the step's end removes what the lag leaves. Registers fall with the Delassus and the muscle state gone (about 90 at W = 2, under 128 at W = 1 for bodies up to 8 nodes), so W = 1 with no exchange becomes possible for 70% of offspring, which removes the tree exchange shuffles and about 15% more.

Which one: 4 substeps of 1/240 with the lagged factor is my primary target, 2 substeps of 1/120 the stretch target if its finer-rate retest passes. The floor of this physics on this laptop is about 170 WI per creature-step at the p50 body, about 210 at the p90 body (12 nodes, 29 muscles), which is the first time the p90 body costs under 1.3x the p50 one, because muscles are now a fifth of the step instead of a third.

## 4. Rates and ceiling

At the FMA-bound cap (124G warp instructions/s) and the generation-51 mix (about 0.85x of the p50 rate with the tail on W = 4):

| variant | creature-steps/s at the mix | creatures/s at 300 steps, 18% host tax |
|---|---:|---:|
| 4 substeps, lagged factor (170 WI) | 620M | 1.7M |
| 2 substeps (125 WI) | 840M | 2.3M |
| 1 substep (90 WI) | 1.17G | 3.2M |

Generation 100 at the new muscle slope (muscles are a fifth of the step, so the slope is about 0.85x rather than 0.75x): 1.45M, 1.95M, 2.7M. So 2M/s comes back into reach with the cheap physics: at 4 substeps it needs the steps ladder to R4 (235 steps), at 2 substeps it is there with R1 to R3. The assumptions: the baked-topology kernel at W = 1 or W = 2 as counted, the diagonal contact passing its slip gate, the 3T-or-5T question settled at 124G by the FMA probe.

## 5. What is measured, in order

1. The stub counts every variant in a day each: node-to-node activation muscles (expect the muscle section 186 to about 60 WI unbaked), per-node contact passes replacing the Delassus and the active set (expect 336 to about 40), 4 substeps with the lagged factor (the total). The stub's synthetic bodies are enough for counting.
2. The gaits: the lane-group kernel already runs at 55M and takes the muscle and contact variants behind defines in a week (its physics sections are the same rules on a different layout), so a 30-generation evolution at 3M under the variant comes from it before the per-lane kernel exists. Then: first_generation (random bodies median within 0.05 m, best under 0.5 m), the elite retest at 4x the substep count (median 0.9, p10 reported), size_report planted-foot slip 0.95 to 1.05, p99 penetration under 2 mm, physics_audit's four ledgers at 1e-5, and the replays of the top 20 elites for the owner's eye, with the archive's cost of transport spread and the cadence and contact bins filled as the numeric proxy for "interesting and efficient".
3. The substep ladder on the new physics: evolve at 2 substeps of 1/120 for 30 generations, retest at 8; pass at median 0.9. If it fails, 4 substeps is the physics and the ceiling is 1.7M/s at generation 51 with R1 to R3, 2.2M/s with R4.

What I would not change: pull-only, Hill, the energy cost, the rhythm genes, the touchdown clocks, the muscle mass, the four ledgers, the fall and break rules, distance-only fitness. Those are the spirit; the rest was the tower.
