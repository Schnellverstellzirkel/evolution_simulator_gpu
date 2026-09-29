#!/bin/bash
# Runs a long CPU job (the full test suite, A/B runs, benchmarks) in one of two
# slots shared by every worktree, with 4 threads at low priority, so agents
# together stay within half the machine.
# Usage: tools/cpu-slot.sh <command> [args...]
common=$(realpath "$(git rev-parse --git-common-dir)")
locks="$(dirname "$common")/target"
mkdir -p "$locks"
while true; do
  for s in 1 2; do
    exec 9>"$locks/cpu-slot-$s.lock"
    if flock -n 9; then
      export RAYON_NUM_THREADS=4 CARGO_BUILD_JOBS=4
      exec nice -n 19 "$@"
    fi
  done
  sleep 5
done
