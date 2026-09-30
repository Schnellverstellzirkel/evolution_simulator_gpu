# Round 7, ML: the cheaper physics against the ladder and the guards

I accept physics's proposal as the physics-lean track. My levers depend on the guards below holding and on the dump being re-run under the new physics, because every share in the ladder is a property of the population, not of the rules.

## (a) Glitch risk per change

Stamina, rest and sprint. Not an exploit: a body that rests pays the rest in distance, and the energy it spends is muscle work from a store that recovers at a fixed rate, which the first-law ledger counts as input exactly as it counted per-muscle energy. Total work over 20 s is bounded by the capacity plus 20 s of recovery, today's bound summed over muscles. What changes is the pacing distribution, a spirit question for the owner. The audit costs nothing: the rung distances already in `Result` give every creature's pacing profile, so the stage log reports the share of elites with a pause (speed under 10% of their mean for 2 s or more) and the median of final / d600. Gap: none for admission; if the owner finds rest-and-sprint uninteresting, the recovery rate is the constant to tune, like the spin cap.

Ligament. The rejected spring joint limit returned 25 kJ against 4.9 kJ of muscle work because it was an explicit spring at a stiff rate. An implicit compliant row cannot return more than it stored, and the ledger carries its store as it carried the tendon, so the flight ledger bounds a release in the air. The gap is a release on the ground: no ledger checks the energy balance during contact, and the tendon had the same gap. The guard is `physics_audit` per elite over the whole trial: kinetic plus potential gain minus muscle work minus the ligament store's change, at most 1% of muscle work on every new island record (ruling 12). The 25 kJ case would read 500%. The spin cap is a drag and removes energy; the ledger is the bound.

Per-node contacts on a three-foot landing. Friction never does positive work per node by construction, but the rod solve redistributes the impulses, and the realized slip of a foot after the coupling can lie along its friction impulse. That is exactly what the realized-work ledger (friction impulse times realized displacement, plus the projection's displacement times rod tension) was built to catch; bar 1e-5 per step as physics set, and on the top 300 at most 1% of muscle work per trial. Penetration recovery is a position-only split impulse and lifts mass: 0.1 kg x 9.8 x 2 mm is 2 mJ per landing against joules per stride, so with the p99 under 2 mm it is not an energy source worth a ledger; a systematic one shows in the slip ratio gate (0.95 to 1.05 in `size_report`). Gap: the transient penetration is not in any ledger; the 2 mm p99 gate is the guard and it must be measured on the evolved elites, not on random bodies, because evolution will find the landing that penetrates most.

The trapezoid's jump. A force that steps from 0 to the cap in one substep is a hammer at 240 Hz. It creates no energy (Hill bounds the power), and the head-shake rule catches jitter carried as motion. The risk is integrator exploitation: a discontinuous force makes the outcome depend on how the rhythm's edges align with the substep grid, and evolution tunes phase genes to that alignment. The finer-rate retest is the right guard, because a grid-aligned gait breaks when the grid changes. I would also remove the jump by construction: a minimum ramp of 2 substeps on each edge as a constant, not a gene. Cost: one clamp, already counted.

The audit lane's 4x re-run (ruling 12) reports, per generation, over the audit entrants: median ratio and p10; bars median at least 0.9, p10 at least 0.7, and any entrant under 0.5 is excluded like a failed confirmation and counted. Today's kernel gives median 0.99 at 2x; the 30 Hz failure was median 0.38.

## (b) Does the ladder's calibration survive

Yes, because nothing in it is calibrated once. The discriminants and thresholds are refit every generation on the audit set, so a new gait distribution refits itself within the window. The one feature that changes meaning, mean muscle energy, becomes stamina, one float per creature instead of a group sum over muscle lanes, cheaper and more informative. The period stays a gene. A physics change is treated as a world change: disarm, discard the window, rearm after 8k audit rows.

R4 changes in its number, not its mechanism. r4 is the 99.9th percentile of final / d600 among survivors, and stamina pacing widens that distribution: a body that rests from 8 to 12 s and sprints late has a ratio far above a steady gait's 2. If the percentile rises from about 3 to above 4, the stop condition (d600 x r4 below the neighbourhood minimum) fires for few survivors and R4 saves under 10 steps per creature; then it is dropped by its own gate (fires for at least 30% of survivors). The emitter-median floor of ruling 10 still holds. So under the new physics the ladder's mature mean is 300 with R1 to R3 as before, and R4's 20 to 25 steps are conditional on the dump, re-run on the 30-generation population before any rung is calibrated for it.

## (c) The honesty retest

The owner's rule is that elites hold under a finer-rate retest. A retest at 4x the production substep count is meaningful at any production rate, because it measures sensitivity to the grid, and a gait that depends on the grid loses distance whenever the grid moves. So the primary (4 substeps of 1/240) is retested at 16 of 1/960 and the stretch (2 of 1/120) at 8. physics's evolve-at-2, retest-at-8 is the stretch candidate's test, not a replacement for the primary's. Cost: 300 elites at 4x per candidate, minutes.

Bars, from today's data (2x retest median 0.99 now, 60 Hz elites at 120 Hz median 0.90 in the older physics, 30 Hz elites at 60 Hz median 0.38): median at least 0.9, p10 at least 0.7, share under 0.5 at most 2%. The p10 bar is the one that matters for the spirit, because a physics whose median passes on steady walkers and whose tail is hoppers that exploit the grid would fill the hopper cells with glitchy movers.

## (d) The removed muscle state word

Nothing in the ladder or the audit lane needs it. R1 and R2 use per-creature quantities; the audit lane needs the flag and the rung distances; R4 needs the running descriptor. The only reader of per-muscle energy outside the physics is the replay frame (`record_frame` writes each muscle's energy and force for the UI); with one store per creature the frame carries stamina once and the force per muscle, which is `replay_forces.rs` and the UI.

## Revised multipliers for my levers

The ladder's factor is unchanged by the physics: 480 / 300 = 1.6x with R1 to R3, 480 / 250 = 1.9x with R4 at the ruled fire rate, both before the dump. What changes is what it multiplies. At physics's 620M creature-steps/s (4 substeps, lagged factor, FMA cap) and an 18% host tax: 1.7M/s at 300 steps, 2.0M/s at 250. At 840M (2 substeps): 2.3M and 2.8M. Those are physics's kernel numbers; phase 0 showed a design table overrunning by 4.5x, so I carry them at half until the stub count exists: 0.85 to 1.0M/s and 1.15 to 1.4M/s.

## The physics-lean track as I would gate it

1. Stub count per variant, one day each (gpu's defines). Gate: total WI within 1.5x of physics's table, or the table is rewritten before any evolution runs.
2. The 30-generation evolution at 3M on the lane-group kernel with the variants behind defines. Gates, all on that run's elites: `first_generation` random median within 0.05 m and best under 0.5 m; retest at 4x the substep count with median at least 0.9 and p10 at least 0.7; slip ratio 0.95 to 1.05; penetration p99 under 2 mm on the evolved elites; the four ledgers at 1e-5; `physics_audit` trial energy excess at most 1% of muscle work on the top 300; the top-20 replays for the owner, with the cost-of-transport spread, the cadence and contact bins filled, and the pacing report (pause share, median final / d600) beside them.
3. The rung dump re-run on that population and `rung_replay` re-gated (the shares, r4, the cell match), before R1 to R4 are calibrated for the new physics. The ladder's code does not change; its numbers do.
4. The substep ladder on the new physics: primary retested at 16, stretch at 8, same bars.
5. The audit lane's 4x re-run of entrants runs from the first generation of the new physics: the only guard that runs every generation rather than once per candidate.
