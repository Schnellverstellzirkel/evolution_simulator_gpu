# Round 4, physics: red team

Attacks on Design P's ladder and body cap (items 14 to 16), answers on my own physics (items 5 and 6, with two spec changes that came out of writing them), and the glitch list for K (item 23).

## Item 14: R1 stops slow-start lineages before anything notices

Failure mode. R1 is fit on last generation's audit rows: 1 in 128 of 3M is about 23k rows, about 4,600 of them passers of the 5 s bar. A 1-in-1,000 miss budget is 4 or 5 creatures. Any class under about 0.1% of passers is invisible: the fit can stop all of it and meet its budget. Two such classes exist by construction. Long-period gaits: periods run to 10 s (evolution.rs clamps to [min_muscle_period, 10]); a 2 s gait has half a cycle at 1 s and often no center-of-mass motion yet. Nursery bodies: 5% of slots and almost none pass (median -0.05 m), so the nursery's passers are a few hundred per generation.

The circuit breaker per cadence band sees neither, because a slow-start lineage shares its band with fast ones and its misses dilute below 100 per 10k. Three generations before it fires is 30% of the nursery's window, and in those generations the nursery cohort receives only its audit-lane children, 1 in 128 of its flow.

Detection: the dump reports R1 stops among nursery-slot passers and among passers whose longest period exceeds 1.5 s, as separate rates; bar under 1% in both. What changes: nursery and immigrant slots exempt from R1 and R2 (10% of slots, about 15 steps per creature, 5%), and the miss report per body plan and per longest period as well as per band. The exemption comes off if the dump shows the classes safe.

## Item 15: R4 drifts an emitter toward sprinters

Failure mode. An emitter in a mature cell whose 200 samples are all stopped at 10 s tells on d(10). The ranking within the top half of its samples is then a 10 s ranking, and the covariance adapts to what is fast at 10 s. Where d(10) and d(20) disagree systematically, the emitter drifts.

Energy is not the mechanism: the store (120 J, half the deficit recovered per second) reaches equilibrium within 2 to 3 s, so the 10 s to 20 s ratio is steady. Stability is: a gait that falls between 10 and 20 s (head shake creeping up, a joint driven to its break, a hop that grows until it topples) has a higher d(10) and a lower d(20) than a steady one. Under R4 it is stopped at 10 s with its high d(10) and outranks the steady child. The CMA moves toward faster and less stable, sets no records, and spends its 200 samples per tell on a 10 s objective forever. The archive is safe; the emitter's budget is lost, and at 1,024 emitters in mature cells that share can be large.

It does not show in best distance within 40 generations (seed noise 20%). It shows in the per-emitter entrant count: 100% of samples stopped at R4 for 3 tells and 0 entrants. What changes: R4 fires only for samples below the emitter's own median d(10) (ga's racing bar as a floor on R4), so the CMA's top half always runs to 20 s. Cost: R4's saving falls from ml's 85% of survivors toward about 50%, the honest middle of ga's and ml's numbers.

## Item 16: the body cap against the morphology reserve

The cap is the largest elite plus two per island. Elites sit at 8.6 nodes mean and 11 at p90, so the cap is 11 to 13 on most islands, and the reserve's new plans (6 to 8 nodes at entry) can grow under it. Where it bites: an island whose elites are all small (6 nodes, cap 8) cannot discover a 10-node plan that would win there, and the reserve's whole purpose is that discovery. The hub gives such a plan a path only after it wins somewhere else.

The alternative that screens instead of limits: on Design K the W = 8 class runs the 9% tail at 4 to 5x the cost of a W = 2 body, about 28% of GPU time uncapped. R1 by construction stops a 27-node graft that does not move at 60 steps instead of 300, which takes the tail to about 10%. So the cap's gain over R1 alone is 10 to 15% of the rate, and its risk is the reserve. The dump must show the share of tail bodies (17 nodes and up) that pass the 5 s bar (under 1%: cap and R1 are equivalent; over 5%: the cap costs entrants), and whether any top-300 elite of the last 20 generations was born above its island's incumbent plus two. I would not build the cap until that second number is shown to be zero. A physics-side alternative: none I trust. Bone mass would slow big bodies, but a mass rule that shifts the optimum is a fitness term in disguise.

## Item 5: my own physics, with two spec changes

Anchor pumping. The store is friction impulse times realized tangential displacement. The chair's premise that this is "the quantity the integrator gets wrong" is not right: positions are the state, so the displacement is exact; what 1 substep gets wrong is the predicted velocity the impulse is sized from. The ledger's own discretization error is in the safe direction: the true work of an impulse J is J (v_start + v_end) / 2, the ledger charges J v_end HS, and the difference J^2 HS / 2m under-fills the store while absorbing and over-charges it while returning. The store's size is mu N times a stuck foot's creep, 0.8 x 10 N x 1 mm, about 10 mJ per plant against 120 J in the muscles. No pump. It is a stiff tendon at the foot, under the tendon's rule.

Drift projection: yes, in one specific way, and it is the exploit I would evolve. The stretch it removes is (omega HS)^2 / 2 of the rod length: 3% at the 15 rad/s cap and 1/60 s, 0.8% at 1/120 s, 0.1% at a typical 3 rad/s. Energy is not the problem (the rod row removes radial velocity each substep, which is dissipative; the projection changes no velocity). The problem is that the projection is mass-weighted over both nodes, so it moves a planted foot along the ground with no friction impulse and no ledger. A 0.3 m rod swung at 10 rad/s about a planted foot moves the foot 1 to 2 mm per step, 6 to 12 cm/s of free slip. A spin-walker would find it at 1 substep and collapse at 4 (the creep is 16x smaller), so the ladder would catch it, but the fix belongs in the spec: a node with a valid anchor or an active contact has infinite mass in the projection; its rod partner takes the whole correction. When both nodes of a rod are planted the length error waits a step. The momentum ledger would not have noticed, because it counts the contact impulse as external.

A second hole found while answering: the contact target pushes a penetrating node out at PUSH_OUT x depth / HS as a velocity, an external impulse the momentum ledger allows. At 1 substep the rotation error sinks a foot a few millimetres and the push-out returns it with velocity the body did not earn, about 70 mm/s per step, 2% of body weight as a mean upward force. Small, but systematic. Fix: split impulse (Catto's rule): the velocity target of a penetrating contact is 0 and the depth is removed in the projection pass, position only, vertically, with the node's own mass. Both fixes are in the spec now.

Stiffest row in a flight step at 1/60 s. Every compliant row (joint limit and spin cap at hardness 20) is inside the direct solve, backward Euler on those rows, stable at any stiffness. The stiffest explicit element is the tendon: k = cap / (TENDON_STRETCH x long) = 100 / (0.25 x 0.1) = 4,000 N/m on a 0.1 m muscle, omega about 280 rad/s on a 0.05 kg node, omega DT = 4.7 at 1/60 s and 2.4 at 1/120 s, both past the explicit bound of 2. It works today because the tendon force is capped, so it chatters at bounded amplitude, and the first-law ledger takes back gains in flight; on the ground the chatter is a second small momentum source through the contacts, and at 1 substep it doubles. Spec change: the tendon becomes a compliant distance row between its anchor points (compliance 1 / k, active when stretched), unconditionally stable; MAXE rises from 12 to 16.

## Item 6: a rung between L2 and 2 substeps

Two of the chair's candidates and one of mine:

- 1 substep for W = 2 bodies and 2 for larger. Rejected on the owner's rule that there is one physics: the same creature would score differently by its class, and a body at the class boundary would change physics when a mutation adds a node. It also gains only 1.47x (70% of offspring at 1.85x, the rest at 1x, harmonic).
- 2 substeps on the first contact step of each touchdown, 1 elsewhere. Sound: the touchdown is where the foot arrives at speed and the prediction error is largest; once anchored, the foot's velocity is near zero and the error moves to the swinging parts. A 2 Hz walker on two feet touches down 4 times per second, 7% of steps, so the gain is 1.85 / 1.07 = 1.73x.
- Spin-adaptive substeps (mine). The error is (omega HS)^2 in the rod rotation and v HS in the touchdown, so tie the substep count to those: 2 substeps when any rod spins above 5 rad/s or any contact candidate approaches the ground above 0.5 m/s, else 1. State-only, deterministic, one physics, no knob (fixed thresholds, like the spin cap). Walkers rarely exceed 5 rad/s; hoppers do at takeoff and landing. Expected gain 1.5 to 1.7x on the generation-51 mix. Its honesty risk is lower than L2's because the substep count follows the error source, and its exploits (spin just under 5 rad/s) are bounded by the threshold: at 5 rad/s and 1/60 s the rotation error is 0.35% of rod length, the same as 2 substeps at 10 rad/s.

So the ladder gets a rung L2.5, spin-adaptive substeps with anchors, ledgers and the two fixes above, tested with the same 0.9 median bar. My odds: L2 at 1 substep everywhere 45% (down from 50% after the two holes above), L2.5 65%. If L2.5 passes and L2 fails, the table's 1 substep row scales by 1.6 instead of 1.85, which is 1.75M/s at 3T and R1 to R3 instead of 2.0M.

## Item 23: the glitch list for K

| change | how an exploit could enter an archive | guard | gap |
|---|---|---|---|
| Drift projection | planted foot moved by the projection: free slip | infinite-mass contact nodes in the projection (spec now); size_report slip ratio; the honesty ladder | two planted nodes on one rod keep their length error that step |
| PUSH_OUT at velocity level | penetration recovery as an external impulse: a stomp pump | split impulse (spec now); confirmation at 4x for records | non-record entrants are never re-run; ml's routed confirmation or the audit lane would cover them |
| Anchored friction store | returning stored energy in a favourable phase | store bounded by absorbed work, ledger conservative by J^2 HS / 2m, about 10 mJ per plant | none found |
| 1 substep (L2) or spin-adaptive (L2.5) | gaits tuned to the coarse step's error | honesty ratio median 0.9 and p10 reported; confirmation at 4x for records | the median passes while p10 exploiters sit in ordinary cells; only a per-entrant re-run at 4x on a sample (the audit lane's 1 in 128 re-run at 4x, about 3% of GPU) closes it |
| Explicit tendon at 1/60 s | bounded chatter pumping momentum through contacts | tendon as a compliant row (spec now); first-law ledger in flight | none after the row |
| Lagged Delassus matrix | stale matrix: a foot sinks or slides by the error, corrected next step | impulses are applied consistently, ledgers unchanged | none as an energy source; slip is measured by size_report |
| Flight step at DT | integrator error in flight | first-law ledger, allowance 1e-4 + 1e-5 x scale per substep | the allowance per second doubles at DT; tighten to 5e-5 + 5e-6 x scale in flight steps |
| Warm-started active set | a different partition picks a different solution of a degenerate LCP | state-only, so deterministic; W is positive definite when contact nodes are distinct, so the solution is unique | none |
| Waveform once per step, ledger only in flight, explicit joint damping | none: dissipative or timing only | ledgers | none |
| Maximal coordinates as such | a rod row solved exactly cannot drift; joint limits and the spin cap are the same rules as rows | physics_audit rod error 1e-5 L, momentum residual 1e-5 | none |
| Compliant limit rows with PUSH_OUT on the angle | internal torque, moves no momentum | momentum ledger | none |

The one gap that matters is the fourth row: a median honesty bar guards the population, not the individual. The fix is cheap and I ask ga and ml to take it into the ladder: the audit lane's creatures that enter an archive are re-run at 4x before commit, and an entrant that keeps under 80% of its score is logged and refused. That is a few hundred re-runs per generation.

## Defence

Item 5's premise on the anchor is wrong (positions are exact state; the ledger's own error under-fills the store), but the attack found two real holes beside it, projection slip and push-out momentum, and both are now closed by construction rather than by a ledger, which is the better kind of guard.
