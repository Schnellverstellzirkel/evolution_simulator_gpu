#!/bin/bash
# Counts one build of the per-lane stub (examples/lane_lean or lane_lean_w1,
# lane_stub): warp instructions per creature-step, issue-active, resident warps
# and the rate at the locked clock, from Nsight Compute. The GPU lock is shared
# (the run measures instructions, and the rate is the kernel's own duration).
# Usage: tools/lane-count.sh <example> [count=32768] [lane_lean arguments...]
#   tools/lane-count.sh lane_lean trim plan=0,1,1,2,4,5,6 baked -DMUSCLE_MODEL=1
# NCU names the Nsight Compute binary (default ncu). Set LANE_LEAN_KERNEL to
# compile another kernel file than the one built in.
set -e
cd "$(dirname "$0")/.."
bin=${1:?example name}; shift
count=32768
case "${1:-}" in count=*) count=${1#count=}; shift;; esac
export EVOLUTION_DEVICES=primary
out=$(flock -s target/gpu.lock "${NCU:-ncu}" --metrics smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.per_cycle_active,gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second \
  -k regex:lane_ --launch-skip 1 --launch-count 1 "target/release/examples/$bin" count="$count" steps=300 repeat=1 "$@" 2>&1)
echo "$out" | grep -E "^ptxas"
steps=$(echo "$out" | grep -oE "steps done [0-9]+" | head -1 | awk '{print $3}')
inst=$(echo "$out" | grep "smsp__inst_executed.sum" | awk '{print $NF}' | tr -d ,)
issue=$(echo "$out" | grep "issue_active" | awk '{print $NF}')
warps=$(echo "$out" | grep "warps_active" | awk '{print $NF}')
dur=$(echo "$out" | grep "gpu__time_duration" | awk '{print $NF, $(NF-1)}')
clk=$(echo "$out" | grep "per_second" | awk '{print $NF, $(NF-1)}')
python3 - "$steps" "$inst" "$issue" "$warps" "$dur" "$clk" <<'PY'
import sys
steps, inst, issue, warps, dur, clk = sys.argv[1:7]
d, unit = dur.replace(",", "").split()
sec = float(d) * {"ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6, "ms": 1e-3, "msecond": 1e-3, "s": 1.0, "second": 1.0}.get(unit, 1e-9)
print(f"warp instructions per creature-step {float(inst.replace(',', '')) / float(steps):.1f}, "
      f"issue-active {issue} %, {warps} warps per SM cycle, clock {clk}, "
      f"rate {float(steps) / sec / 1e6:.1f}M creature-steps/s")
PY
