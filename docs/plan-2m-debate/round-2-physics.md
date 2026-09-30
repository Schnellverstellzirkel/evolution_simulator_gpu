# Round 2, physics: answers and corrections

Base restated on the chair's 480 steps per creature. No new runs; two checks of the code (EVOLUTION_WARP_SUBSTEPS and EVOLUTION_WARP_PLANT_ROUNDS are live overrides in warp_kernel::solver_setting; p2_speed's wall clock starts before submit and its busy clock is the kernel's start and stop events).

## D1a. Per-lane contact cost at the warp's maximum

Every lane runs the warp's maximum contact count, so the count is at 4 contacts unless the whole warp has fewer. My table already priced 4 contacts always (Delassus 500, direct solve 300). At a warp maximum of 2 it falls to about 150 and 120; I do not budget that. One correction: the active-set loop also runs the warp's maximum number of re-solves. I priced one re-solve; the worst case of three adds about 170, so the step is 3,100 at 6 nodes, not 2,900. The contact histogram gpu asked for (contacts per substep on an evolved wave) is a 20-line PROFILE counter; I cannot run it this round and I do not assume its answer.

## D1b. Registers and shared memory per lane, maximal coordinates, W = 1, 6 nodes, 15 muscles

gpu's 200 for W = 1 in reduced coordinates is right for that formulation. Mine, counted per phase (live registers at the peak of each phase, persistent plus temporaries):

| item | registers |
|---|---:|
| persistent: positions and velocities 24, header (n, muscles, base, step, inv_mass, flags, quake) 10, masses 6, rod lengths and packed topology 7, friction anchors 8 | 55 |
| forces phase: node force accumulators 12, drag temporaries 10 | +22 = 77 |
| muscle loop: 4 float4 constants 16, geometry and drive 14, accumulators 12 | +42 = 97 |
| rod matrix and LDL: 11 entries in place, rhs 5, temporaries 10, accumulators 12 | +38 = 93 |
| contact phase: Delassus 36 (8x8 symmetric), per-contact data 24 (gap, mu, target, vn, vt, node), factor 11, impulses 8, temporaries 10 | +89 = 144 |
| integrate, projection, ledgers | +20 = 75 |

The contact phase sets the peak: 144, not the 90 to 100 I wrote. My round-1 number left the Delassus matrix out of the register count. At 144 the SM holds 14 warps by the register file and 12 by block granularity (3 blocks of 128). Two ways down: keep the Delassus in shared memory (144 B per lane, see below, which the shared budget cannot take at 16 warps), or run the PGS with a tree solve per row update and never form the matrix (about 110 registers, plus about 1,000 instructions per substep). Under the power cap an instruction is a joule and a stall is not, so 12 warps at 144 registers beats 16 warps at 1,000 more instructions. The 6-node class at W = 1 therefore runs at 12 to 14 warps per SM with ILP 3 in the muscle loop and 4 to 8 in the contact phase. My issue-rate estimate at that occupancy is 35 to 45%, down from the 40 to 55% I wrote.

Shared memory per lane in [index][lane] layout: muscle energy 15 x 4 = 60 B, sensor offset 60 B, last waveform value 60 B (or 0 B and one more cos per muscle per substep, about 1% of the step), metrics 76 B (19 floats as today's Result), radius and friction per node 48 B (or from global). Total 196 to 304 B. The budget at 16 warps is 195 B; at 12 warps it is 260 B. So: recompute the waveform, radius and friction from global, and the lane needs 196 B, which fits at 12 warps and not at 16. Muscle constants stay in global memory at 64 B per muscle (gpu's 16-bit packing halves it; the GA domain must accept the quantization), 960 B per creature, 370 KB per SM at 384 resident creatures, served from L2 at about 10 to 20% of its bandwidth at the target rate.

The consequence: at 6 nodes W = 1 is register-bound at 12 to 14 warps, and W = 2 in maximal coordinates (each lane 3 nodes, half the Delassus columns, one 8x8 LDL exchanged by about 50 shuffles) counts about 95 registers and 100 B of shared per lane, 16 to 20 warps, at the cost of about 300 shuffle and sync instructions per creature-substep. I now put W = 2 for 5 to 8 nodes and W = 1 for 4 nodes or fewer, which is gpu's layout with my formulation inside it. gpu's question whether I would build maximal coordinates: yes, because it is the only count under 3,500 per step at 1 substep, and because it makes W = 2 cheap (no spatial inertias to split between lanes, only node vectors).

## D1c. Proposal 1: the predicted ratio and the cheapest experiment

Two different ratios are in play. The 0.79 (1 substep plus 1 planting round) is a transfer ratio: elites evolved at 2 substeps run at 1. The ratio that decides whether 1-substep scoring is honest is the reverse: elites evolved at 1 substep, re-tested at 2 and 4. The 30 Hz history (38%) was that second kind and it is the only precedent.

Predictions. Transfer ratio, elites evolved at 2 substeps run at 1 substep with the realized-work ledger and anchored friction: median 0.85 to 0.95 (anchors hold feet where planting rounds got 0.79; the ledger only removes distance). Honesty ratio, evolved at 1 substep with ledger and anchors, re-tested at 4 substeps: median 0.85 to 0.95, p10 near 0.6. Pass bar: median at least 0.9. My probability that it passes: 50%. Without the ledger and anchors (today's rule at 1 substep): median 0.5 to 0.7, because physics2's own measurement shows friction doing 1,725 J of realized positive work per hopper without planting, and a population evolved at that setting will find it.

Cheapest experiment, in order:

1. Zero code. `EVOLUTION_WARP_SUBSTEPS=1` on search_ab --gpu for 30 generations, one seed, then re-test the top 300 at SUBSTEPS 2 and 4 (the replay_match path with the override). About 25 minutes at today's rate under the shared lock. If the median is at least 0.9, the 1.8x exists today and proposal 1 is a setting; I expect it to fail, and the failure tells how large the hole is.
2. The realized-work ledger: about 40 lines in the kernel. After the substep's kinematics, each walker lane forms its tangential displacement against the contact tangent, multiplies by its friction impulse, and if the product is positive adds it to the same excess that the first-law check removes with `keep`. Re-run step 1.
3. Anchored friction: about 60 lines. Two floats per lane (anchor position, valid flag in the sign), set on the first contact substep, the friction target becomes (anchor minus position along the tangent) per substep instead of zero slip, clamped to mu times normal, and the anchor moves to the foot when the clamp binds. Re-run step 1.

Step 1 and 2 fit in a day on the shared lock. Step 3 is the posture-adjacent rule that needs the owner.

What breaks proposal 1: a median honesty ratio under 0.9 after step 3. Then 1-substep scoring is out and the cheap-fidelity screen (D2) is the fallback use of the same kernel setting.

## D1d. The 13% in p2_speed

It is not wave tail. p2_speed's busy seconds are the kernel's event pair, and the kernel runs until every group has exhausted the counter, so the tail is inside the 51.8M figure. The wall clock includes pack (0.32 s per 262k on 4 threads, 7% of the 4.4 s run), the pinned staging copy and upload (about 340 MB at 262k x 1.3 KB, 0.03 to 0.05 s), the readback, and the poll loop's 5 ms sleeps. So about 10 of the 13 points are host packing, which the data-flow plan removes, and about 3 are copies and polling.

The tail itself, inside busy: 6,144 resident groups (24 SMs x 16 warps x 4 groups at W = 8) advance at about 7,300 steps per second each (45M / 6,144), so a 1,200-step survivor started last runs for 164 ms. The last 6,144 creatures start during the wave's final 0.1 s, so the drain is at most 164 ms with occupancy falling linearly, about 80 ms lost, 2% of a 4.4 s wave. A 10x faster kernel shortens both the wave and the drain by 10x, so the tail stays about 2% per wave at the same wave size, and the game's 8 streams overlap it. Not a lever.

## D1e. gpu's four cuts, my verdicts

- Waveform and Hill target once per step: yes. The rejected "hold the muscle forces" held the force with its damper; holding only the shortening target and recomputing the damper and Hill factor per substep keeps the implicit stabilization. Measurement: elite ratio at 4x rate unchanged.
- First-law ledger only in flight: yes. Per lane it is a branch on the lane's own contact count. The muscle-work accumulation (4 ops per muscle) stays; the second geometry pass goes.
- Contact matrix once per step: yes (my proposal 5). Measurement: size_report slip ratio and the elite ratio.
- Sweeps 4 to 2: no without a warm start. 2 cold sweeps on 4 coupled contacts leave the coupling unsolved and feet sink or slide. With a warm start from the last substep's impulses (8 floats per creature), 2 sweeps are defensible; measure slip. With the direct solve the question disappears.

## D2. The cheap-fidelity screen against ml's attack

ml's argument transfers the 30 Hz result to a screen. The 30 Hz elites were evolved at 30 Hz: selection over 30 generations optimized the exploit, because the coarse physics scored them, chose the parents and set the records. In the cheap screen the coarse physics scores nothing. It decides only which children get a full trial. For an exploit to be selected, a lineage would have to pass the coarse screen by exploiting and then beat its cell at full fidelity to reproduce. A child that does the first and not the second is a wasted survivor: 1,350 steps spent, no archive entry, no offspring. So the failure modes are two, and neither is exploit selection: false negatives (honest movers the coarse screen ranks below the bar) and survivor dilution (coarse-only movers taking places in the kept 20% from honest ones). Both are one number each.

The measurement: one 262k wave from a full-fidelity population, every creature scored twice, once at 1 substep to 5 s and once at 2 substeps to 20 s. Report (a) recall of the full-fidelity top 1% and top 10% within the coarse top 20% (today's 5 s rung keeps 100% and 96%, the bar), (b) the share of the coarse top 20% whose full-fidelity 5 s distance is below the honest bar (dilution), (c) both per cadence band and node count, since a class-wise miss is the loss the spirit cares about. It runs on the current kernel with the two overrides and no code. It settles the static question. It does not settle drift over generations, but drift can enter only through the false-negative rate, which the audit lane ml proposed for its own stops measures every generation for this one too.

Gain restated on 480: 0.8 x 150 + 0.2 x (150 + 1,200) = 390, so 1.23x. If proposal 1 passes its honesty test, this screen is moot because everything runs at 1 substep. It is the fallback for the case that proposal 1 fails, and it combines with ml's checkpoints (a checkpoint inside a coarse screen is the same decision, cheaper). What breaks it: recall of the top 10% under 95%, or dilution above 25% of the kept share.

## D5. Rate at the p90 body

The p90 body from the old build's mature generations: 9 nodes, 8 rods, 35 muscles. Per-lane maximal coordinates, 1 substep: forces 225, muscles 35 x 80 = 2,800, rod LDL 200, solve 120, detection 90, Delassus 730, contact solve 300, response 100, integrate and projection 150, ledgers 180, metrics 200: about 5,100 thread instructions per step, 9,800 at 2 substeps. Registers at 9 nodes peak near 170 at W = 1, so this class runs at W = 2 (about 110 per lane) with about 300 shuffle and sync instructions per creature-substep: 5,400 per creature-step at 1 substep.

Budget: I take os's sustained figure as the working number, 2.8T thread instructions per second (45% issue at 2.0 GHz), and gpu's 4.2T (55% at 2.49 GHz) as the ceiling.

| body | layout | per step, 1 substep | per step, 2 substeps | rate at 2.8T, 1 substep | rate at 2.8T, 2 substeps |
|---|---|---:|---:|---:|---:|
| mean, 6 nodes, 15 muscles | W = 1 or 2 | 3,100 | 5,800 | 900M | 480M |
| p90, 9 nodes, 35 muscles | W = 2 | 5,400 | 10,500 | 520M | 270M |
| 12 nodes, 50 muscles | W = 4 | 8,000 | 15,500 | 350M | 180M |

A generation-50 population at 60% mean-like, 30% p90-like and 10% large gives a harmonic-mean rate of about 600M creature-steps per second at 1 substep and about 320M at 2. At 480 steps per creature: 1.25M/s and 0.67M/s. With a 1.3x search-side cut in steps: 1.6M/s and 0.87M/s. At the 4.2T ceiling multiply by 1.5. So at generation 50 the target needs all of: the per-lane layout, maximal coordinates, 1 substep passing its honesty test, a 1.3x from the rungs, and bodies held near the mean by muscle mass or a search rule. Any one missing puts the sustained rate at 0.7 to 1.3M/s. That is my honest number, and it is 25 to 40% lower than my round-1 figure because of the register correction (D1b) and the power budget.

## What I withdraw or concede

- Withdrawn: "90 to 100 registers at 6 nodes" (now 144 at W = 1, 95 at W = 2) and "40 to 55% issue" (now 35 to 45%).
- Conceded to gpu: W = 2 as the main class. My formulation goes inside it.
- Conceded to the chair: 480 steps base; my 1.25x for the cheap screen becomes 1.23x.
- Kept: maximal coordinates with the direct tree solve (about 45% fewer instructions per substep than reduced coordinates per lane); proposal 1 at 50% odds with a one-day experiment ladder; the flight-step and lagged-matrix cuts, which cost nothing to try.
