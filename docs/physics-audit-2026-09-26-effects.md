# Physics audit: effects and evolved elites, 2026-09-26, revision 08598ff

This is a measurement audit of the current solver on evolved archive elites,
following AGENTS.md item 104. The revision is `08598ff` (`qd::VERSION` 24).
While this audit ran, other workers added uncommitted seasons work to
`src/config.rs`, `src/environment.rs`, `src/qd.rs`, and `src/storage.rs` and
raised `qd::VERSION` to 25, and later in the session the concurrent work also
touched `src/cpu_engine.rs`. Every physics number below comes from a binary
built from the clean `08598ff` tree: the pre-built
`target/release/evolution-simulator` binary and a snapshot built under
`/tmp/opencode/audit-08598ff` from `git archive 08598ff`. A `size_report`
built from that snapshot reproduced the calm report from the first binary byte
for byte, and that calm report was recorded before any of the concurrent edits.
No source file was changed by this audit.

The audit looks for physics exploits on evolved elites: a stored score a replay
does not confirm, forward momentum the ground did not pay for, travel with no
ground contact, and bodies beyond the bone cap or beyond what their mass can
explain. It also exercises the recent effects with fresh checkpoints. The calm
world is the primary workload. Sloped, muddy, gap, hurdle, and quake worlds are
secondary.

## Commands

The calm checkpoint used the requested command on the clean tree, before the
concurrent edits landed:

```text
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 cargo run --release -q -- headless --population 20000 --seed 38 --generations 8 --duration 60 --checkpoint runs/audit-effect-physics.evo
```

The requested reports on that checkpoint:

```text
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 cargo run --release -q --example size_report -- runs/audit-effect-physics.evo 20
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 EVOLUTION_LEDGER=1 cargo run --release -q --example size_report -- runs/audit-effect-physics.evo 20
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 cargo run --release -q --example first_generation -- 20000 20
```

Effect runs and diagnostics used the snapshot binary at
`/tmp/opencode/audit-target/release/evolution-simulator` and examples built
from `/tmp/opencode/audit-08598ff`:

```text
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 /tmp/opencode/audit-target/release/evolution-simulator headless --config /tmp/opencode/audit-mud-slope.json --generations 8 --checkpoint runs/audit-effect-mud-slope.evo
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 /tmp/opencode/audit-target/release/evolution-simulator headless --config /tmp/opencode/audit-mud-only.json --generations 8 --checkpoint runs/audit-effect-mud-only.evo
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 /tmp/opencode/audit-target/release/evolution-simulator headless --config /tmp/opencode/audit-slope-only.json --generations 8 --checkpoint runs/audit-effect-slope-only.evo
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 /tmp/opencode/audit-target/release/evolution-simulator headless --config /tmp/opencode/audit-terrain.json --generations 8 --checkpoint runs/audit-effect-terrain.evo
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 /tmp/opencode/audit-target/release/examples/size_report <checkpoint.evo> 20
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 EVOLUTION_LEDGER=1 /tmp/opencode/audit-target/release/examples/size_report <checkpoint.evo> 20
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 /tmp/opencode/audit-target/release/examples/audit_detail <checkpoint.evo> 10
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 /tmp/opencode/audit-target/release/examples/audit_fidelity <checkpoint.evo> 10
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 EVOLUTION_LEDGER=1 /tmp/opencode/audit-target/release/examples/audit_worlds <checkpoint.evo> 5
nice -n 15 env EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 /tmp/opencode/audit-target/release/examples/audit_first_gen 20000 20
```

`audit_detail`, `audit_worlds`, `audit_fidelity`, and `audit_first_gen` are
temporary examples written for this audit under
`/tmp/opencode/audit-08598ff/examples/`. They are not part of the repository.

## Workload

All runs: 20,000 creatures, seed 38, `random_seed` false, 60 s trials, 8
generations. Each archive is re-tested under its own config. Checkpoints are
under `runs/` and gitignored.

```text
world               settings                              best m     QD score  niches
calm                defaults                              24.5698    1943.56   1084
mud+slope           mud 0.02, slope 0.03                  52.7104    1795.22   1055
mud only            mud 0.02                              17.5994    1394.58   1076
slope only          slope 0.03                            13.3242    1499.45   1070
gaps+hurdles+quake  gaps 0.35, hurdles 0.08, quake 0.05     2.8503     680.03   1018
```

Mud, Slope, and Gaps use their second button level (Damp, 3%, Narrow).
Hurdles and Earthquake also use their second level (Low, Tremors).

## What was measured

- `size_report` on the top 20 archive entries of each checkpoint: stored
  fitness, standard CPU replay distance, node count, total and longest bone,
  mass, foot slip, slip per replay meter, and head accelerations.
- `EVOLUTION_LEDGER=1` on the top 3 entries: the six horizontal momentum terms
  summed over the whole trial.
- `audit_detail` on the top 10: for each scored step, ground contact, the split
  of center-of-mass movement into planted, sliding-contact, and airborne steps,
  and the grounded node slide speed.
- `audit_fidelity` on the top 10: the archive value against fresh standard,
  fine, and pose-perturbed fine CPU evaluations.
- `audit_worlds`: the top 5 of a checkpoint replayed under calm, mud, slope,
  and mud+slope, with the ledger.
- `first_generation` with 20,000 bodies for 20 s in the calm, mud+slope, and
  gaps+hurdles+quake worlds.

## Raw numbers

### Stored fitness against replay

A stored score larger than its CPU replay would be an inflation exploit. It
does not occur in any of the 60 audited rows. The mud+slope world has the
largest gap in the safe direction: rank 12 stores 14.9 m and the standard
replay runs 42.5 m. The standard replay is not the admission value; admission
also folds in a pose-perturbed fine check, and that check can score lower than
the standard replay. Calm top 20 summary: replay never below stored, maximum
replay minus stored +4.2 m, mean +0.94 m. Mud+slope: never below, maximum
+27.6 m, mean +5.23 m. Gaps+hurdles+quake: never below, maximum +0.1 m, mean
+0.01 m.

Calm world, `size_report` top 10 of 20:

```text
archive_m  replay_m  nodes  length_m  longest_bone_m  mass_kg  slip_m  slip_per_replay_m  pos_peak_g  pos_shake_peak_g  engine_shake_g
     24.6      24.6      4      0.92            0.32     1.68    32.5               1.32        13.9               6.8             2.2
     23.2      23.2      4      0.91            0.32     1.68    30.5               1.31        15.6               6.7             2.0
     21.2      21.2      4      0.91            0.33     1.67    30.3               1.43        10.6               5.5             2.3
     19.5      19.5      4      0.91            0.33     1.72    28.1               1.44        17.7               5.9             2.2
     19.5      19.5      4      0.91            0.33     1.64    26.3               1.35        15.2               6.2             4.0
     19.3      19.3      4      1.05            0.37     2.02    25.8               1.34        15.8               6.3             3.9
     19.1      20.1      4      0.92            0.33     1.61    27.8               1.38        15.5               5.2             2.3
     18.1      18.1      4      0.92            0.32     1.69    23.8               1.31        17.8               6.2             0.8
     16.8      16.9      4      0.81            0.29     1.42    22.9               1.36        12.1               4.3             2.7
     16.6      16.7      5      1.87            0.56     4.38     7.3               0.44        13.2               5.5             2.9
median body length 0.92 m, median slip per replay meter 1.36
```

Mud+slope world, `size_report` top 10 of 20:

```text
archive_m  replay_m  nodes  length_m  longest_bone_m  mass_kg  slip_m  slip_per_replay_m  pos_peak_g  pos_shake_peak_g  engine_shake_g
     52.7      54.9      5      1.69            0.64     4.08   114.0               2.08        21.0               7.7             2.2
     47.9      55.4      5      1.70            0.63     4.09   102.4               1.85        20.1               6.5             3.0
     38.9      54.7      5      1.70            0.64     4.13   118.5               2.17        20.3               7.3             3.1
     36.3      38.9      5      1.74            0.64     4.25    94.7               2.43        23.1               6.9             3.6
     36.0      46.2      6      2.21            0.64     5.26   147.7               3.20        18.7               6.8             3.3
     35.1      45.1      5      1.70            0.63     4.17    93.5               2.07        21.3               6.8             2.1
     28.9      34.5      5      1.71            0.63     4.16   103.1               2.99        21.6               7.3             2.6
     27.5      31.8      5      1.73            0.63     4.26    73.0               2.30        23.1               7.2             2.5
     27.2      28.6      5      1.68            0.63     4.02    89.8               3.14        20.8               6.9             2.4
     18.9      18.9      6      2.60            0.92     7.71    74.3               3.93        21.5               8.0             2.1
median body length 1.73 m, median slip per replay meter 3.14
```

The largest individual bone in the whole sample is 2.00 m in the terrain
checkpoint, exactly at the cap and never above it. The mass of every row is
derived by `physics::nodes` from the bones and node diameters, so it cannot be
carried in the checkpoint. The calm sample is 4 to 6 nodes and 0.8 to 4.2 m
long; the terrain sample is 4 nodes and up to 4.84 m in total with 35 kg at
the cap. The mud+slope sample is 4 to 7 nodes and 0.77 to 3.13 m.

### How the scored distance is earned

`audit_detail` splits the center-of-mass movement over the scored steps into
steps where a grounded node moved slower than `physics::PLANTED_SPEED`
(0.01 m/s), steps where every grounded node moved faster than that, and steps
with no contact. `term/len` is the terminal frame over all recorded frames. A
fall freezes scoring at the terminal frame, so a fall time of 0.0 means the
body ran upright for the full trial.

Calm world, top 5:

```text
archive_m replay_m fall_s term/len contact_steps planted_dx sliding_dx airborne_dx mean_slide_m/s max_slide_m/s
     24.6     24.6    0.0 3700/3701            3223      -0.29      19.91        5.01            0.41         3.12
     23.2     23.2    0.0 3700/3701            3176      -0.53      18.01        5.81            0.37         3.25
     21.2     21.2    0.0 3700/3701            3234       0.05      16.75        4.43            0.38         4.03
     19.5     19.5    0.0 3700/3701            3246       0.15      15.90        3.50            0.35         3.31
     19.5     19.5    0.0 3700/3701            3359      -0.30      16.61        3.19            0.28         3.25
```

Mud+slope world, top 5:

```text
archive_m replay_m fall_s term/len contact_steps planted_dx sliding_dx airborne_dx mean_slide_m/s max_slide_m/s
     52.7     54.9    0.0 3700/3701            3565       0.11      53.96        0.47            1.07        23.67
     47.9     55.4    0.0 3700/3701            3583       0.08      54.60        0.30            1.07        24.55
     38.9     54.7    0.0 3700/3701            3594       0.22      53.95        0.12            1.03        23.81
     36.3     38.9    0.0 3700/3701            3587       0.22      38.11        0.16            0.82        24.83
     36.0     46.2    0.0 3700/3701            3596       0.10      45.70        0.03            0.92        23.29
```

The mud+slope checkpoint's top 3 replayed under the calm config. The distance
is higher in the calm world, and the gait keeps its sliding signature:

```text
archive_m replay_m fall_s term/len contact_steps planted_dx sliding_dx airborne_dx mean_slide_m/s max_slide_m/s
     52.7     67.2    0.0 3700/3701            3116       0.23      55.15       11.46            1.16        23.68
     47.9     61.5    0.0 3700/3701            3155       0.23      51.56        9.27            1.07        24.55
     38.9     61.8    0.0 3700/3701            3299       0.06      55.32        6.03            1.07        23.82
```

In the calm and mud+slope worlds every audited elite has a grounded node in
almost every scored step, 2701 to 3597 of the 3600 scored steps, and none
earns its distance airborne. The terrain world is the exception because its
elites fall into pits. The split does not add up to the absolute replay
distance because it starts from the settled pose's center of mass.

Terrain world, top 5:

```text
archive_m replay_m fall_s term/len contact_steps planted_dx sliding_dx airborne_dx mean_slide_m/s max_slide_m/s
      2.9      2.9    2.4 242/3701                0       0.00       0.00        2.39            0.00         0.00
      2.8      2.8    9.1 648/3701                0       0.00       0.00        2.41            0.00         0.00
      2.8      2.9    2.5 250/3701               10       0.00       0.24        2.20            0.16         0.23
      2.8      2.8    0.0 3700/3701                0       0.00       0.00        2.37            0.00         0.00
      2.8      2.8    2.2 233/3701                8       0.00       0.22        2.13            0.17         0.18
```

Most of the terrain top 10 fall within 2.2 to 2.9 s. One falls at 9.1 s, and
one stays upright for 60 s while moving 2.64 m airborne and -0.32 m in sliding
contact. Slip over the scored interval is 0.00 to 0.03 m/m.

### Momentum ledger

Calm champion, plus ranks 2 and 3:

```text
ledger for 24.6 m (2 kg), kg*m/s over the trial:
  integration speed cap                +0.0
  ground contact                       -0.9
  velocity-pass speed cap              +0.0
  velocity-pass constraints            +0.0
  projection COM shift                 +0.4
  muscle forces                        +0.0
ledger for 23.2 m (2 kg), kg*m/s over the trial:
  integration speed cap                +0.0
  ground contact                       -0.2
  velocity-pass speed cap              +0.0
  velocity-pass constraints            -0.0
  projection COM shift                 +1.8
  muscle forces                        +0.0
ledger for 21.2 m (2 kg), kg*m/s over the trial:
  integration speed cap                +0.0
  ground contact                      +31.5
  velocity-pass speed cap              +0.0
  velocity-pass constraints            -0.0
  projection COM shift               -31.8
  muscle forces                        +0.0
```

Mud+slope ranks 1 to 3:

```text
ledger for 54.9 m (4 kg), kg*m/s over the trial:
  integration speed cap                +0.0
  ground contact                     +361.7
  velocity-pass speed cap              +0.0
  velocity-pass constraints            +0.0
  projection COM shift               -321.5
  muscle forces                        -0.0
ledger for 55.4 m (4 kg), kg*m/s over the trial:
  integration speed cap                +0.0
  ground contact                     +312.0
  velocity-pass speed cap              +0.0
  velocity-pass constraints            -0.0
  projection COM shift               -272.9
  muscle forces                        +0.0
ledger for 54.7 m (4 kg), kg*m/s over the trial:
  integration speed cap                +0.0
  ground contact                     +352.2
  velocity-pass speed cap              +0.0
  velocity-pass constraints            -0.0
  projection COM shift               -312.4
  muscle forces                        +0.0
```

Mud+slope champion replayed under the other worlds, `audit_worlds`:

```text
archive_m calm_m mud_m slope_m both_m
     52.7   67.2   59.1    62.9   54.9
  ledger calm: fitness 67.2 m: integration speed cap +0.0, ground contact +336.1, velocity-pass speed cap +0.0, velocity-pass constraints +0.0, projection COM shift -331.6, muscle forces -0.0
  ledger mud: fitness 59.1 m: integration speed cap +0.0, ground contact +321.6, velocity-pass speed cap +0.0, velocity-pass constraints +0.0, projection COM shift -319.1, muscle forces -0.0
  ledger slope: fitness 62.9 m: integration speed cap +0.0, ground contact +370.7, velocity-pass speed cap +0.0, velocity-pass constraints +0.0, projection COM shift -330.2, muscle forces -0.0
  ledger both: fitness 54.9 m: integration speed cap +0.0, ground contact +361.7, velocity-pass speed cap +0.0, velocity-pass constraints +0.0, projection COM shift -321.5, muscle forces -0.0
```

In the calm replay the forward impulse from ground contact is +336.1 kg m/s for
a 4.08 kg body, the projection cap removes -331.6, and the net +4.5 matches a
final speed near 1.1 m/s. Friction pays the momentum. The projection path
removes more than it adds for every one of these fast sliding gaits.

### Fidelity of the top elites

`audit_fidelity` compares the stored value with fresh CPU evaluations: standard
exact, fine exact (4x rate and 4x solver passes), and fine with the same pose
perturbation the admission check uses (`storage::perturb_elite`, node x and y
by up to 2 cm and grip by plus or minus 10%). A standard fall time of 0.0
means the body ran the full 60 s upright.

Calm world, top 10:

```text
archive_m standard_m standard_fall_s fine_m fine_fall_s fine_minus_standard fine_perturbed_m
     24.6       24.6            0.0   -0.0          0.6               -24.6             25.7
     23.2       23.2            0.0   28.6          0.0                +5.3             23.3
     21.2       21.2            0.0   27.4          0.0                +6.2             23.4
     19.5       19.5            0.0   21.7          0.0                +2.2             21.7
     19.5       19.5            0.0   21.3          0.0                +1.9             21.6
     19.3       19.3            0.0   20.7          0.0                +1.4             22.1
     19.1       20.1            0.0   23.5          0.0                +3.4             17.4
     18.1       18.1            0.0   24.2          0.0                +6.1             24.7
     16.8       16.9            0.0   12.9          0.0                -4.0             18.3
     16.6       16.7            0.0   16.8          0.0                +0.1             16.8
```

Mud+slope world, top 10:

```text
archive_m standard_m standard_fall_s fine_m fine_fall_s fine_minus_standard fine_perturbed_m
     52.7       54.9            0.0   39.3          0.0               -15.6             52.6
     47.9       55.4            0.0   38.6          0.0               -16.8             47.2
     38.9       54.7            0.0   37.7          0.0               -17.0             38.9
     36.3       38.9            0.0   36.7          0.0                -2.2             37.6
     36.0       46.2            0.0   27.9          0.0               -18.3             35.1
     35.1       45.1            0.0   33.3          0.0               -11.8             35.9
     28.9       34.5            0.0   30.9          0.0                -3.5             29.9
     27.5       31.8            0.0   25.9          0.0                -5.8             26.3
     27.2       28.6            0.0   28.7          0.0                +0.1             28.5
     18.9       18.9            0.0   16.9          0.0                -2.0              1.8
```

### Random-body control

`first_generation` on 20,000 bodies for 20 s. The calm line is identical to
the recorded baseline before this audit (median -0.07 m, p99 0.34 m, best
11.03 m).

```text
calm: 20000 bodies, 20 s: median -0.07 m, 99% 0.34 m, best 11.03 m
mud+slope: 20000 bodies, 20 s: median -0.11 m, 99% 0.23 m, best 3.74 m
gaps+hurdles+quake: 20000 bodies, 20 s: median -0.05 m, 99% 0.25 m, best 1.15 m
```

### Cross-world replays

The top 5 of each checkpoint under every world. The calm checkpoint's elites
are worst in mud, and the mud+slope checkpoint's elites are best in calm.

```text
source=/home/amipo/workspace/evolutionSimulator-claude/runs/audit-effect-mud-slope.evo
archive_m calm_m mud_m slope_m both_m
     52.7   67.2   59.1    62.9   54.9
     47.9   61.5   60.5    55.0   55.4
     38.9   61.8   49.1    62.5   54.7
     36.3   42.8   38.6    37.2   38.9
     36.0   45.6   43.4    46.3   46.2
source=/home/amipo/workspace/evolutionSimulator-claude/runs/audit-effect-physics.evo
archive_m calm_m mud_m slope_m both_m
     24.6   24.6   -0.3    23.2   22.2
     23.2   23.2   22.0    20.5   20.1
     21.2   21.2   19.2    20.0   18.6
     19.5   19.5    0.3     0.2   13.1
     19.5   19.5   18.6    17.4   16.6
```

## Findings

### No free propulsion and no inflated stored score

The random-body control is unchanged at exactly the recorded calm baseline:
median -0.07 m, p99 0.34 m, best 11.03 m. Random bodies in the effect worlds
top out lower than calm, at 3.74 m in mud+slope and 1.15 m in the terrain
world. The archive-admission rule holds: in the 60 audited rows (20 per world
for calm, mud+slope, and terrain) the stored score is never larger than the
standard CPU replay, and the mean stored value is below the replay in every
world. In the calm and mud+slope worlds every audited elite has a grounded
node in almost every scored step, so no body travels while no foot touches.
The terrain elites' short airborne travel is a fall into a pit. The longest
individual bone anywhere is 2.00 m, exactly at the cap. Mass cannot be forged
because `physics::nodes` derives it from the bones. For the fastest sliding
gait the projection center-of-mass term is negative, so the capped planted-feet
path removes momentum instead of manufacturing it.

### Fidelity sensitivity and the pose-perturbed check

This is the strongest anomaly. The calm champion, creature id 145492, stores
24.6 m. A fresh standard CPU run reproduces 24.6 m and stays upright for the
full 60 s. A fresh fine CPU run (4x rate and 4x solver passes, the same exact
stored pose) falls at 0.6 s and scores -0.0 m. The pose-perturbed fine run
scores 25.7 m. Archive admission takes the worse of the standard trial and a
pose-perturbed fine check (`scheduler::pump_checks_with` perturbs the pose
instead of checking the stored pose; `storage::perturb_elite` mirrors it), and
that check scores above the stored value, so nothing in the pipeline lowers
the stored 24.6 m. The exact pose is never evaluated at fine fidelity by
admission, so an exact-pose, standard-rate-only gait can still become the
archive champion. The second calm elite has the opposite sign (standard
23.2 m, fine 28.6 m), so the divergence is chaotic rather than a one-way
standard-rate failure. Across the calm top 10 the fine-minus-standard
difference runs from -24.6 m to +6.2 m. Across the mud+slope top 10 it runs
from -18.3 m to +0.1 m without a collapse. The fixed evolved-creature agreement
fixtures do not cover this class of fresh champion.

Creature fields of the calm champion, from the JSON export
(`/tmp/opencode/audit-champion-calm.json`): id 145492, 4 nodes, 3 bones with
rest lengths 0.3226, 0.3024, and 0.2984 m (0.92 m in total), 4 muscles, node
diameters 0.068 to 0.12 m, node friction 0.798 to 0.977, mutability 0.9722.
The failing fine run is exactly this stored pose with `Config::fidelity` set
to `Fidelity::fine()` and nothing else changed.

### The fastest gait class slides on every contact

The mud+slope champion, creature id 145187, stores 52.7 m and replays at
54.9 m in its own world, 67.2 m in calm, 59.1 m in mud only, and 62.9 m in
slope only. It has 5 nodes, 4 bones with rest lengths 0.5944, 0.6370, 0.2022,
and 0.2591 m (1.69 m total, longest 0.637 m), 7 muscles, node diameters 0.061
to 0.12 m, node friction 0.656 to 1.0, and a derived mass of 4.08 kg. In the
calm replay, 55.15 m of the scored movement happens in steps where every
grounded node slides faster than 0.01 m/s, 11.46 m is airborne, and 0.23 m
happens on planted feet. The mean grounded slide is 1.16 m/s and the peak is
23.68 m/s, which is 0.39 m in a single 1/60 s step. The friction impulse that
moves the body comes from opposing those backward slides, and the ledger shows
it is paid by ground contact (+336.1 kg m/s) and then clipped back by the
projection cap (-331.6 kg m/s). This is not unearned momentum and it is not a
stored-score inflation. The open question for the physics owner is whether the
normal impulse that sizes that friction, which comes from a position clamp
after a fast ground sweep, is physically plausible at those sweep speeds. The
calm 8-generation run found only the weaker end of this gait class: max slide
3.1 to 7.4 m/s and median slip 1.36 m per replay meter, against 23.7 m/s and
3.14 in the mud+slope run.

### Effects exercised

The terrain world (gaps 0.35 m, hurdles 0.08 m, quake 0.05 m) tops out at
2.85 m. Its elites use 2.00 m bones to try to span the 3.4 m pit spacing, and
most fall within 2.2 to 2.9 s with slip near zero. No exploit signature
appears.

Mud alone lowers the best to 17.6 m and slope alone to 13.3 m against calm
24.6 m, and random bodies do worse under both, so the mud+slope run's 52.7 m
is not an effect loophole. The calm champion scores -0.3 m in mud while the
mud+slope champion scores 67.2 m in calm, which shows that the two runs
selected different gait families and the effect configuration changed the
search trajectory rather than the legality of the winning motion.

## Limits

- One seed (38), one budget (20,000 creatures, 8 generations, 60 s), and one
  duration. The top 20 archive entries per run were reported, with the top 10
  for the detail, fidelity, and random-body tests, and the top 5 for the
  cross-world replays. Other seeds and longer runs can surface other gait
  families.
- Effects covered: calm, slope 3%, mud 0.02, gaps 0.35, hurdles 0.08, and
  quake 0.05. Not covered: gravity, air retention, grip, heat wave, drought,
  wind, ground roughness, seasons, and the catastrophes (meteor and
  extinction).
- Every evaluation here is CPU. No GPU against CPU agreement audit was run.
  The admission check itself runs on a GPU when one is free, while the
  pose-perturbed reproduction here is CPU only.
- The ledger is a net sum for lane 0 over the whole trial, including motion
  after a fall. It is not a per-step or per-contact attribution. The distance
  split uses the same frame sampling as `size_report` and is not a force
  measurement.
- The calm checkpoint predates the concurrent seasons edits; the snapshot
  binary and the pre-built binary are both from clean `08598ff`. The effect
  checkpoints and the snapshot examples are not committed. The checkpoints are
  local under `runs/` (gitignored) and the raw outputs and champion JSONs are
  under `/tmp/opencode`.
