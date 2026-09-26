# Evaluation device recovery

User authorization: autonomous backlog work, no approval questions, verified commits to main. Preserve distance-only scoring and the existing primary-device restriction. Other team owns active physics and archive insertion.

## Intended behavior

A reported GPU failure must leave every unfinished evaluation available for retry on the CPU. Already completed results are delivered once. Standard and perturbed fine trials retain their exact input and configuration. CPU failure ends the operation with a useful error; it never starts an unbounded retry loop. GPU startup failure selects CPU, never an unrequested secondary GPU. This does not promise recovery from a hung driver.

## Implementation batches

1. Make threaded worker errors persistent. Regression tests reproduce errors swallowed by wait, unexpected disconnects, and successful results buffered before failure. Keep wait's existing interface, drain successful results before returning the retained error, and stop accepting submissions after failure. Commit this independent correction first.
2. Share immutable submitted populations using Arc while retaining the public owned-submit convenience method. Scheduler queued units retain population, configuration, original indices, trial kind and device ticket without a second deep copy.
3. Add explicit device kind and retry state. Retire a failed GPU, move incomplete units to CPU retries, preserve completed output, and advance retries during pause/save/drain paths as well as normal rounds. Validate ticket and result count before consuming a unit. Surface CPU errors after delivering any previously completed output.
4. Support CPU fallback when primary initialization fails, including a dispatcher on the existing general Rayon pool when a separate CPU pool is disabled. Keep explicit GPU engine constructors strict. Report the actual active backend and the original failure once.

## Verification

- Use fake engines for deterministic submit/poll/disconnect failures, mixed-device success, interrupted rounds, and failures during perturbed checks. Assert no missing or duplicate population indices and identical saved trial inputs.
- Test bounded CPU failure and malformed engine results.
- Keep at most eight low-priority build/general/evaluation workers combined; every run sets EVOLUTION_DEVICES=primary and EVOLUTION_CPU_THREADS=6.
- Before each commit: formatting, all-target Clippy, release tests. Run explicit RTX agreement tests when scheduler behavior changes. Direct GPU checks must never silently use CPU fallback.
- Update AGENTS live status and push small verified batches, pulling concurrent changes without overwriting them.

## Progress

- [x] Persistent engine error correction
- [x] Shared retained submissions
- [x] Runtime CPU recovery and drain behavior
- [ ] Startup fallback and backend reporting
- [ ] Full validation and documentation

Batch 1 verification: seven failure regressions reproduced the original errors before the fix, including the submission race. All-target release tests passed 83 CPU tests and nine diagnostic tests, with four GPU tests ignored. Three explicit RTX tests passed in3.34s; formatting and Clippy passed. Runtime CPU retry remains pending.

Batch 2 verification: three scheduler regressions first failed for missing shared retention, caller-state normalization and consumed malformed results. Shared and owned submission tests verify allocation identity. Formatting, Clippy, 88 CPU tests, nine report tests and three RTX tests passed (GPU3.31s). Retained snapshots prepare recovery without duplicating body arenas. Automatic retry is still pending.

Batch 3 verification: devices carry an explicit GPU/CPU kind and queued units carry a retry count. A polled GPU failure retires the device and re-submits every unfinished unit, including pending fine checks, to a healthy CPU with its exact population, configuration and ticket order; a submission failure keeps the rejected creatures in the round for the next engine. A failed CPU is terminal: already completed output is delivered first, the error persists on every later collection, and no retry loop starts. Five scheduler regressions first failed before the fix (retry inputs, rejected submission, terminal CPU with buffered output, pending checks, retry state). Formatting, Clippy, 94 CPU tests, nine report tests and three RTX agreement tests passed (GPU13.32s). Startup CPU fallback and backend reporting remain.
