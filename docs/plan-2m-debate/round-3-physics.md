# Round 3, physics: the formulation as a specification

Scope: the creature step for the per-lane kernel (W = 1, 2, 4, 8, maximal coordinates), the rules that change, the ladder, the gates, the rate per class at generation 51. Breeding, ring, archive and power follow the chair's rulings 7 to 10 unchanged. Row: "1 substep, R1 to R3" if the honesty test passes, else "2 substeps, R1 to R3" times the cheap-fidelity screen.

## 1. State and constants per creature

Nodes i = 0..n-1 (node 0 the head), rods j = 0..n-2 with child node b_j = j + 1, pivot node a_j, parent rod (none for the neck), rest length L_j, joint range [lo_j, hi_j] on the relative angle to the parent rod. Rods are numbered breadth first, so parent(j) < j. Per node: mass (muscle mass included by Model::new), radius, friction. Per muscle: the 16-field record as today. The per-lane kernel reads what warp_kernel::pack writes now.

State: p_i, v_i; muscle energy and rhythm offset per muscle (shared memory, [muscle][lane]); friction anchors, one per contact slot: node index, anchor x along the tangent, valid bit, absorbed work (16 floats, registers); the metrics record (shared); the ledger excess (2 floats). No angles: a relative joint angle, when a rule needs one, is atan2 of the cross and dot of the two rod directions.

Time: DT = 1/60, SUBSTEPS from the ladder (section 6), HS = DT / SUBSTEPS. Unless section 3.7 makes a step a single substep.

## 2. Rows

Every constraint is a row k with a sparse Jacobian J_k over at most 3 nodes, a right-hand side, and a type. The rod rows are equalities and are eliminated exactly. Every other row is an "extra" row; at most MAXE = 12 per substep (4 contacts x 2 rows, plus up to 4 joint or spin rows, the most violated first).

- Rod j: J = [-d_j at a_j, +d_j at b_j], d_j the unit rod direction. Target: rhs_j = -(v_b - v_a) . d_j - |perp(v_b - v_a, d_j)|^2 HS / L_j (the centripetal term), so the rod is rigid to first order; 3.6 removes the rest.
- Contact normal at node c: J = [n at c], n the ground normal from terrain() as today. Target as today (goal = -gap / HS, PUSH_OUT inside the ground). lambda >= 0.
- Contact friction at node c: J = [t at c], t = perp(n). Target: 3.5. |lambda_t| <= mu lambda_n, mu as today.
- Joint limit at rod j against its parent q: the relative angular velocity row. A rod's angular velocity is w_j = cross(d_j, v_b - v_a) / L_j, so J = [-perp(d_j)/L_j at a_j, +perp(d_j)/L_j at b_j] minus the same for q. Active when the predicted relative angle passes lo or hi, with today's goal and PUSH_OUT, and LIMIT_HARDNESS as a compliance alpha = HS / (LIMIT_HARDNESS x I_j) on the diagonal, I_j the rod's inertia about its pivot with its subtree (Model::new precomputes it, as today's d). Unilateral toward the violated bound.
- Spin cap for rod j when |w_j| > SPIN_CAP: the rod's angular velocity row, target 0, compliance alpha = HS / (SPIN_HARDNESS x m_b L_j^2 x (|w_j| / SPIN_CAP - 1)). A pure couple, as today.
- Joint damping: explicit, since HS / tau = 0.17 is far below the explicit stability bound of 2. Torque = -I_j w_rel / tau as a couple on the rod's nodes and the opposite couple on the parent rod. Same semantics as today.

## 3. The substep

3.1 External forces per node as today (gravity, wind, mud drag, buoyancy). Air and water drag on a rod act at its midpoint and go half to each node (a point force at fraction t of a rigid massless rod is (1-t) F and t F on its nodes). Joint damping couples. Muscles (3.2). Free velocity v* = v + HS f / m.

3.2 Muscles: today's rule unchanged (waveform target, drive x energy, Hill, damper, cap, store, tendon, sensors, clocks). The force at anchors at fractions tA, tB on rods A and B goes to the four nodes with weights (1-tA), tA, (1-tB), tB and opposite signs. The muscle-work ledger terms as today.

3.3 Rod factor. A_jk = J_j M^-1 J_k^T is nonzero for j = k (1/m_a + 1/m_b) and for rods sharing a node: s (d_j . d_k) / m_shared, s = +1 when the shared node is the pivot of both or the child of both, else -1. Eliminate rods in reverse index order: the rods at one node form a clique and child rods have higher indices than the rod ending at their pivot, so this order creates no fill. LDL without square roots, j from n-2 down to 0: D_j = A_jj - sum over eliminated children c of L_cj^2 D_c; for each remaining neighbour k (parent rod, uneliminated siblings): L_jk = (A_jk - sum_c L_cj L_ck D_c) / D_j. Storage: D per rod, L per adjacent pair. A tree solve is forward substitution over children then backward over parents, about 4 flops per pair. Exact; no sweeps.

3.4 Extras and their Delassus matrix. Collect the active extras E: contacts by the deepest-4 rule, then limit and spin rows, at most MAXE. For each row k: r = J_rods M^-1 J_k^T (nonzero on the rods at the row's nodes), tree-solve mu = A^-1 r, response y_k = M^-1 (J_k^T - J_rods^T mu) on E's nodes only. W_kl = J_l y_k, symmetric. v** = v* corrected by the rods alone (one tree solve); free values w_k = J_k v**. Compliant rows add alpha_k to W_kk.

3.5 Direct active-set solve. Start with every row active (feet stuck). Solve (W + alpha) lambda = target - w by an e x e LDL, e <= 12. A normal or limit lambda < 0 leaves the set; a friction row past mu lambda_n is fixed at the bound. Re-solve on the reduced set, at most 3 rounds. Warm start from the previous substep's partition (section 11). Friction keeps today's clean rule (oppose the mean of start slip and end slip without its own force); the "25% harder" static factor is dropped because the anchor replaces it.

Anchored friction, the posture-adjacent rule. A contact slot keeps an anchor x_a along the tangent, set when its node first touches. The friction target is (x_a - x_c) / HS instead of 0. When the row hits its bound the foot slides and x_a moves to x_c. A node out of contact for one full step clears its slot. The anchor returns at most the energy it took: the slot sums the friction work it absorbed since planting, and an impulse that would do positive work beyond that store is clamped to the store, like the tendon.

3.6 Apply and integrate. v = v** + sum_k lambda_k y_k; p += v HS. After the step's last substep, the drift projection: c_j = |p_b - p_a| - L_j, tree-solve A mu = -c with the last factor, p_i += (1/m_i) sum over rods at i of (+-) d_j mu_j. Velocities untouched. The projection is internal, so it moves no momentum; its energy is inside the ledgers' allowance.

3.7 Flight step. If at the top of a step no node's predicted gap is <= 0 and no anchor is valid, the step runs as one substep of DT whatever SUBSTEPS is. State-only, so deterministic. Muscles, limits and the spin cap are implicit or stable at DT.

3.8 Lagged matrix. With SUBSTEPS = 2, substep 2 reuses substep 1's rod factor and Delassus matrix when the extra set is identical; targets and free values are recomputed.

3.9 Ledgers, per substep.
- Momentum: sum m_i v_i equals the pre-substep momentum plus the external impulses; the difference is applied as one uniform velocity as today. Here it is rounding only, but it costs 10 flops per node and stops a bug from becoming an exploit.
- Realized friction work: after the position update, per contact slot, work = lambda_t x the node's tangential displacement. Positive work beyond the anchor's store joins the first-law excess.
- First law: as today in flight; the friction excess is removed the same way (the `keep` scaling about the center of mass) on the ground.

3.10 Metrics, fall, joint break, head shake, screen: unchanged, once per step.

## 4. Which rules in docs/physics.md change

Stay word for word: gravity, wind, air drag, muscles, tendon, joint damping, joint limits and break, spin cap, momentum balance, first law in flight, the fall and head-shake rules, the deepest-4 selection, the environment effects, fitness.

Change: the state paragraph (positions and velocities; rods exact by a direct solve, lengths held by one projection per step); "projected Gauss-Seidel, 8 sweeps, warm started" becomes "a small complementarity system solved directly"; "holds 25% harder" and "planted against the end pose" become the anchored-friction paragraph (3.5); one new sentence: friction that does positive work on the realized motion is taken back. SUBSTEPS as the ladder decides.

## 5. Version and re-evolution the owner must accept

qd::VERSION bumps once when the kernel switches; old saves are turned down and the population re-evolves from scratch. The scalar reference becomes physics3.rs (about 800 lines: the spec in plain Rust, with tests for rod drift, the chordal factor against a dense solve, the active set against a 2,000-sweep PGS, the momentum ledger at rounding); physics2.rs goes when the lane-group kernel goes. The one posture-adjacent decision is the anchored foot: it holds where it landed and slides when the cone binds.

## 6. The substep ladder, with pass bars

Each rung on the current lane-group kernel, before any per-lane work, under the shared lock, one seed, 30 generations of search_ab --gpu at 3M, then the top 300 re-tested at 2 and 4 substeps.

| rung | change | pass bar | my odds |
|---|---|---|---|
| L0 | EVOLUTION_WARP_SUBSTEPS=1, no code | median honesty ratio >= 0.9 at 4 substeps | 15% |
| L1 | plus the realized-work ledger (about 40 lines) | same | 30% |
| L2 | plus anchored friction (about 60 lines) | same, and size_report planted-foot slip ratio 0.95 to 1.05 | 50% |
| L3 | L2 on the per-lane kernel (the spec) | same bar; also first_generation median <= 0.05 m | inherits L2 |

Also reported at every rung: p10 of the ratio (warning below 0.6), realized friction work on the 300 as a share of muscle work (bar 1%), first_generation. If L2 fails, SUBSTEPS stays 2 and the cheap-fidelity screen runs the early rungs at 1 substep with the full score at 2 (bar: 95% recall of the full-fidelity top 10%, dilution under 25%).

## 7. Gates on every merge of the physics track

- first_generation.rs: 262k random bodies, median within 0.05 m of zero, best under 0.5 m, no NaN. Anything above is free propulsion and blocks the merge.
- physics_audit.rs with the kernel's ledgers: per elite, momentum residual under 1e-5 kg m/s per step, realized positive friction work under 1% of muscle work, first-law corrections in under 1% of flight substeps, rod length error under 1e-5 L after projection.
- replay_match.rs: archive distance equals the replay's to 1e-4.
- Elite ratio at 4x rate on the top 300: median >= 0.95, p10 >= 0.7.
- Determinism: two runs of one 262k wave are bit-equal.

## 8. Cost and rate per class at generation 51

Classes by node count from the chair's histogram, W by the register table of round 2 (W = 1 peaks at 145 at 6 nodes; each extra lane halves the node state and the Delassus columns and adds about 150 shuffle and sync instructions per creature-substep per lane pair).

| class | offspring share | W | per step, 1 substep | per step, 2 substeps |
|---|---:|---:|---:|---:|
| A: <= 6 nodes, about 13 muscles | 45% | 1 | 3,000 | 5,700 |
| B: 7 to 8 nodes, about 18 muscles | 25% | 2 | 4,400 | 8,400 |
| C: 9 to 16 nodes, about 25 muscles | 21% | 4 | 7,100 | 13,800 |
| D: 17 to 32 nodes, about 30 muscles | 9% | 8 | 7,000 | 13,500 |

Class D is cheap per node because the Delassus stays at most 12 x 12 whatever n is and the rod factor is linear in n. The lane-group kernel serves D (about 11M creature-steps/s there today) until the W = 8 class exists.

Weighted mean per creature-step with a 10% tax for warp-maximum contacts and active-set rounds: 5,000 at 1 substep, 9,700 at 2. The flight step and the lagged matrix (5 to 15% on hoppers, 4% on everyone) are not credited until measured.

| budget | 1 substep, creature-steps/s | 2 substeps |
|---|---:|---:|
| 3T | 600M | 310M |
| 5T | 1.0G | 515M |

Creatures per second at the chair's steps per creature:

| case | 3T | 5T |
|---|---:|---:|
| 1 substep, 300 steps (R1 to R3) | 2.0M | 3.3M |
| 1 substep, 235 steps (R1 to R4) | 2.55M | 4.3M |
| 2 substeps, 300 steps | 1.03M | 1.7M |
| 2 substeps, 300 steps, cheap screen x1.2 | 1.24M | 2.1M |

Before the host tax (5 to 8%). The design lands on row 3 of the chair's table if L2 passes, 2.0M at 3T with no margin and 3.3M at 5T. If L2 fails it lands between rows 1 and 2, at 1.2 to 2.1M with the cheap screen, and 2M sustained then needs the 5T budget. Where it stops: at the p90 body's muscle count. Muscles are 60% of the p90 step and nothing in the physics makes a muscle cheaper than about 70 instructions per substep; if offspring drift past 29 muscles at p90, every number above falls in proportion.

Multiplier table from the chair's base (45M at generation 3 is 27M on the generation-51 mix) and 480 steps:

| lever | 3T | 5T |
|---|---:|---:|
| per-lane layout by class (lane use and issue) | 5x | 5x |
| maximal coordinates, direct solves (instructions) | 1.1x | 1.1x |
| power budget against the 2.49 GHz peak | 0.55x | 0.9x |
| kernel at 2 substeps, from 27M | 310M | 515M |
| 1 substep with ledgers and anchors (if L2 passes) | 1.95x | 1.95x |
| steps 480 to 300 (R1 to R3) | 1.6x | 1.6x |
| result, 1 substep | 2.0M/s | 3.3M/s |
| result, 2 substeps with the cheap screen | 1.24M/s | 2.1M/s |

## 9. Determinism, world change, save, replay, 60 FPS

Every rule is a function of the creature's state and the block's settings; no atomics in arithmetic; warp-uniform loop bounds are masks; the flight-step and lagged-matrix decisions are state functions. A world change recompiles the kernel as today and data's ring discards blocks in flight; anchors and ledgers are per creature. Saves hold archives and search state only. A replay is a one-creature RECORD wave of the scoring kernel, with the friction anchors added to the frame (2 floats per slot) so the audit can show a planted foot. The physics has no host work per step, so 60 FPS is data's and os's problem.

## 10. Tracks with gates, in order

1. Ladder L0 to L2 on the current kernel (one day). Gate: the SUBSTEPS decision of section 6. It runs before any per-lane code.
2. physics3.rs reference plus tests (3 days). Gate: the section 5 tests pass; first_generation clean on the CPU port.
3. W = 1 kernel for class A, other classes on the lane-group kernel behind one advance interface (one week). Gate: 200M creature-steps/s on class A of save42 at 2 substeps, bit-equal determinism, all section 7 gates.
4. W = 2 and W = 4 (one week). Gate: harmonic mean over save42 >= 300M at 2 substeps on the measured budget.
5. W = 8 for D; retire the lane-group kernel and physics2.rs (3 days). Gate: rate on D >= 3x the lane-group's.
6. 1 substep on the per-lane kernel (L3) if L2 passed (2 days). Gate: the section 6 bar.
7. Flight step, lagged matrix, warm active set, each separately (2 days). Gate: at least 3% each in p2_speed with slip and elite ratio unchanged; a lever that does not measure is deleted.

The game is playable at every merge because the lane-group kernel keeps every class the new one does not cover yet.

## 11. One idea not yet raised: cycle-aligned rung distances

A rung compares distances at a fixed time, and a hopper's distance at 5.00 s depends on where in its cycle 5.00 s falls: 0.5 m per cycle at 1 Hz is up to 10% of phase noise on a 5 m distance. The metrics already count vertical turns. Let each rung compare the com_x at the last completed cycle (the value at the last turn before the rung, or the rung's own when no turn happened), so the same gait ranks the same at any phase. The score is untouched; only the rung's predictor changes. Estimate: Spearman with the 20 s distance rises from 0.887 toward 0.92, letting the keep fall from 20% to about 17% at equal recall: 27 steps per creature, 6%, more at the 10 s rung. Cost: 2 floats in the metrics. Measurement: from ga's dump, Spearman and top-10% recall of the cycle-aligned 5 s distance against the raw one; adopt if recall at 17% keep matches today's at 20%.

The warm-started active set (3.5) is the small second: start the direct solve from the previous substep's stuck-or-sliding partition (4 bits per slot). Expected: re-solves fall from about 1 per substep to 0.2, 4 to 8% of the substep. Measurement: a re-solve histogram on save42, then p2_speed.
