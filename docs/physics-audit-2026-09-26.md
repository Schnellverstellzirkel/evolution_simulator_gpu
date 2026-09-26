# Physics audit: 2026-09-26

This is a source inspection, not a measurement. No physics runs or benchmarks
were performed for this audit. References describe the code inspected before
the platform/default-device changes in this session.

Follow-up in this session: `physics::evaluate` now delegates to the current CPU
engine, and `size_report` measures only scored replay intervals with the correct
fidelity and sloped contact surface. The legacy `physics::step` remains for its
low-level tests and old momentum diagnostic. The concerns below describe the
source inspected before those fixes; the contact solver itself is unchanged.

## Ground support and sliding

`src/cpu_engine.rs` and `shaders/physics_creature.wgsl` reconstruct velocity from
positions, then apply friction to nodes still touching the floor. Their budget
is each node's friction coefficient times `final_y - predicted_y`, converted to
velocity. This difference already includes clamp/lift displacement, but only a
final touching node's own mass contributes to its friction impulse. Support
transferred through bones to the rest of the body is lost from that budget.

Support enters through the initial floor correction, floor clamps inside bone
and joint projection, and the whole-body lift after reconstruction. Velocity
passes also remove downward contact velocity without adding that normal impulse
to friction. They can restore horizontal foot slip after the one-shot friction
operation. Relevant CPU blocks are the ground solve, bone/joint floor clamps,
`lift`, `max_change`, and the `resting` velocity clamp; matching shader blocks
use `ground_lift`, `max_change`, and `VELOCITY_SOLVE_ITERATIONS`.

A proposed flat-ground diagnostic invariant is:

```text
sum(mass * (final_y - predicted_y)) / dt
  = sum(mass * positive_floor_correction) / dt
    + total_mass * whole_body_lift / dt
```

Internal projections and the COM-restoring reconstruction should cancel in the
mass-weighted sum. Sum signed node displacement before clamping the total;
summing only positive displacement also counts internal redistribution. This
is position-solver support, not a complete velocity-level contact impulse.

A full friction fix should account for support at contacts and accumulate
tangential impulses within the Coulomb budget across velocity passes. Repeated
unlimited friction would spend the budget several times. Uniform COM drag is
not equivalent to foot friction. Preserve the existing subtraction of whole-body
lift from reconstructed vertical velocity, which prevents a limb penetrating
the ground from launching the body.

Useful fixtures: a heavy body supported by a light foot; zero grip; no ground;
zero gravity; mirrored sliding; and friction that cannot increase kinetic
energy in an isolated passive contact. Verify CPU/GPU parity at both fidelities
and bump `qd::VERSION` for any physics change. Re-measure seed 38 with 100,000
creatures for 20 generations before claiming a size or slip improvement.

## Muscle energy

Both active engines debit `abs(total_force * relative_velocity) * dt`, charging
passive damping and negative active work. Positive work by the active pull is
`max(-active_force * relative_velocity, 0) * dt`. Attribute the actual capped
force consistently; one decomposition is
`clamp(active + damping) - clamp(damping)` for the active contribution. This is
a smaller physics change than replacing the contact solver, but still requires
CPU/GPU changes, focused tests, and a version bump. Test passive damping,
stretching, positive contraction work, and exhausted muscles separately.

## Evaluator and replay correctness

The legacy `physics::evaluate`/`physics::step` loop lacks current terrain, joint
constraints and breaks, fall-distance retention, fatigue, touchdown resets, and
velocity reconstruction. It uses standard timing even when `cfg.fidelity` is
fine. The optional CPU timing in `src/main.rs` and several integration tests
still use it. Consolidate evaluation on `cpu_engine::evaluate`; use one batch
for benchmarks so SIMD groups fill. Migrate direct step tests and diagnostic
callers before removing the legacy loop.

For replay agreement, compare raw CPU fitness to mass-weighted COM in frame
`cfg.fidelity().settle() + round(fall_time * rate)`, or the last frame if upright.
Test both fidelities, falls, joint breaks, partial SIMD groups, and grouped
versus individual evaluation. The UI still uses standard settle helpers, which can mis-index fine-fidelity
frames. `size_report` now uses the configuration's fidelity for both settling
and the terminal frame.

Add an evolved-creature fixture to the existing short, direct standard/fine
GPU agreement check. Test perturbed scheduler trials separately: terminal
decisions amplify rounding. Both engines consume the shared `physics::Joint` values: `physics::joints`
already quantizes joint centers to snorm16 for both paths. Additional GPU
packing/unpacking can introduce driver-dependent rounding, but there is no
CPU-full-precision versus GPU-quantized center split. Agreement fixtures should
avoid terminal thresholds unless those thresholds are the subject of the test.

## Early exit after falling

A plain CPU all-fallen exit is not currently semantics-preserving. Both active
engines continue accumulating ground contact, height, oscillation, gait turns,
and contact/lift bitsets after a fall. The scheduler divides these totals by
the configured full duration. Numeric failures after falling can also override
the stored distance. First define and freeze terminal results in both engines,
update normalization and archive versioning, then skip further evaluation work.
Replay can continue simulating the limp body for its full animation. Unused
SIMD lanes repeat a real creature, so padding itself does not prevent an exit.

## Diagnostic limits

- `size_report` includes post-fall sliding but divides by frozen archive fitness;
  that can inflate slip per meter. Its terrain floor omits the slope correction.
- A passive-energy test needs actual solver velocities. Finite differences of
  replay positions incorrectly turn the position-only lift into kinetic energy.
- In a no-ground, no-drag fixture, internal muscle forces, bone reconstruction,
  and velocity constraints should preserve horizontal momentum within rounding.
- The momentum example uses legacy physics. Its explicit Radeon engine was
  removed during the platform fix; current diagnostics should use `size_report`
  with `EVOLUTION_LEDGER=1` until the legacy evaluator is consolidated.
