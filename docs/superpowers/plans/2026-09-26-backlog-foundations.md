# Backlog foundations implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development to execute independent tasks and verify each deliverable before committing.

**Goal:** Establish reproducible, safe local validation and protect core search and checkpoint behavior before changing locomotion physics.

**Architecture:** Keep the current engine and archive interfaces. Add targeted regression tests, repair confirmed portability and resource-control defects, and document current behavior. Physics changes follow a separate measured baseline and CPU/GPU comparison.

**Tech Stack:** Rust, Cargo, Rayon, Vulkan/WGSL, GitHub Actions.

**Spec:** AGENTS.md and the owner's instruction to work autonomously through its backlog.

## Global constraints

- Fitness is horizontal distance only; do not introduce scoring terms.
- Use at most eight build jobs and eight Rayon threads at low priority. Every run sets EVOLUTION_DEVICES=primary and EVOLUTION_CPU_THREADS=6.
- Keep simulation semantics aligned across engines; bump qd::VERSION for archive or physics changes.
- Work on the clean main checkout as explicitly requested; commit and push verified batches.
- Coordinator owns builds and integration; agents have disjoint file assignments.
- The user delegated design decisions and explicitly requested no approval questions.

## Review focus

- Checkpoint restart after optimizer stall must breed the same offspring.
- Windows compilation and checkpoint save/load must work on the current host.
- Runtime defaults must not start Radeon compute or exceed the CPU budget.
- Invalid config boundaries and slower archive candidates must be rejected.
- Fallen bodies and replay metrics must retain their existing meaning until a tested cross-engine change.

## Task 1: Validation and resource controls

**Files:** src/engine.rs, src/main.rs, src/scheduler.rs, examples/momentum_ledger.rs; local ignored target/tooling.

- [x] Establish local Rust/MSVC build and record baseline failures.
- [x] Add focused tests for default thread/device selection, then repair confirmed resource-control defects.
- [x] Make platform-specific priority handling compile on Windows without changing Linux behavior.
- [x] Make diagnostic engine selection obey the same opt-in rule for secondary GPUs.
- [x] Run fmt, clippy, and release CPU tests; review the changes.
- [ ] Commit and push the verified foundation batch.

## Task 2: Core correctness coverage

**Files:** tests/search_state.rs, tests/replay_consistency.rs, src/storage.rs; evaluator wrapper and size-report regression coverage.

- [x] Test default and invalid config boundaries.
- [x] Test one archive winner per cell and monotonic replacement.
- [x] Test deterministic valid modern offspring and checkpoint continuation.
- [x] Reproduce and fix any discovered checkpoint defects separately, with format migration/version review.
- [x] Run required checks and review the changes.
- [ ] Commit and push the verified foundation batch.

## Task 3: Build support and documentation

**Files:** README.md, docs/architecture.md, Cargo.toml, .github/workflows/ci.yml.

- [x] Verify current product, engine, and archive behavior from source.
- [x] Update stale documentation without claiming unmeasured results.
- [x] Add a portable fast iteration profile and CPU CI (GPU tests stay local).
- [x] Run required checks and review the changes.
- [ ] Commit and push the verified foundation batch.

## Task 4: Physics baseline and next measured change

**Files:** docs/validation.md, docs/performance-log.md, AGENTS.md; physics files only after baseline.

- [x] Audit ground normal-force accounting, terminal metrics, and reference divergence.
- [x] Run the prescribed seed-38, 100k, 20-generation baseline and size report once build is available.
- [ ] Select and validate the next narrowly defined physics change. The contact audit and unexplained evolved-elite replay outlier require further investigation; this foundation batch intentionally leaves production stepping unchanged.
- [x] Record measured results and update backlog completion accurately.

## Verified handoff (2026-09-26)

- Final local validation: formatting and all-target clippy passed; 68 CPU tests (30 library, 5 replay, 15 search-state, 18 simulation) passed, with four GPU tests ignored by default. Seven size-report tests passed. The three explicit simulation GPU tests passed in 3.32 s on the primary RTX GPU. Remote CI is configured but unverified.
- Baseline: fixed seed 38, 100k candidates, 20 generations, 60 s trials; final best 165.5846 m, 1,374 cells, QD 23,060.27. Top-50 median total bone length 2.23 m, longest individual bone 1.81 m, reported median slip/replay meter 0.89. See `docs/results/2026-09-26-bone-cap-seed-38/`.
- Open evidence: rank 21 stores 111.2 m but CPU replay travels 6.4 m. Diagnostic slip semantics changed, so historical ratios are not directly comparable. No contact-solver fix, energy-conservation claim, search A/B improvement, or performance speedup is asserted.
- Checkpoint V4 persists island optimizer progress and reads V3. The public evaluation wrapper uses the production CPU engine; legacy `physics::step` remains. Production stepping and `qd::VERSION` are unchanged.
- Commit and push remain coordinator actions after this handoff.

## Concurrent integration

Foundation commits 250ad82 and ca75014 were created, but origin advanced to abb00cb before the first push. The merge preserves the newer version-19 physics, CPU archive admission, replay result, and catastrophes. Its validation passed 72 CPU tests, nine report tests, three RTX tests, formatting and Clippy; the required random-population check returned best 9.81 m (20k bodies,20 s). The earlier measurements above describe version16 and do not measure current physics. Next work is engine error propagation and CPU fallback; active physics/archive-insertion work remains with the other team.
