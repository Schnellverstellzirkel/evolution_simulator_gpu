#!/usr/bin/env bash
# Min-of-N graphical-app generation benchmark, robust to a busy shared machine.
# Usage: noisy_bench.sh <population> [runs] [generations]
# Env passes through to the app (e.g. EVOLUTION_SUBGROUP_SYNC=1).
set -euo pipefail
POP="${1:?population}"
RUNS="${2:-5}"
GENS="${3:-30}"
cd "$(dirname "$0")/.."

mkdir -p runs/bench
LOG="runs/bench/pop${POP}-$(date +%s).log"
CSV="runs/bench/noisy.csv"
[[ -f "$CSV" ]] || echo "timestamp,population,runs,generations,gen_s_min,gen_s_median,gen_s_all,eval_s_median,archive_s_median,breed_s_median,shader_s_median,load_avg,gpu_clock_min_mhz,gpu_clock_max_mhz" >"$CSV"

samples=()
evals=()
archives=()
breeds=()
shaders=()
clock_min=99999
clock_max=0

for ((r = 1; r <= RUNS; r++)); do
  # Sample clocks while the run is in flight.
  (
    for _ in $(seq 1 200); do
      c=$(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || continue
      [[ -n "$c" ]] || continue
      echo "$c" >>"$LOG.clocks"
      sleep 0.15
    done
  ) &
  sampler=$!
  out=$(env EVOLUTION_SMOKE_POPULATION="$POP" EVOLUTION_BENCH_GENERATIONS="$GENS" \
    taskset -c 4-15 nice -n 5 cargo run --release 2>&1) || {
    echo "$out" | tail -20
    kill "$sampler" 2>/dev/null || true
    exit 1
  }
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  {
    echo "===== run $r ====="
    echo "$out" | grep -E "Native|stages|GPU profile|Breeding profile" | tail -5
  } >>"$LOG"

  g=$(grep -oE '\(([0-9.]+) generations/s\)' <<<"$out" | tr -d '()' | awk '{print $1}' || true)
  e=$(grep -oE 'evaluation ([0-9.]+) s' <<<"$out" | awk '{print $2}' || true)
  a=$(grep -oE 'archive ([0-9.]+) s' <<<"$out" | awk '{print $2}' || true)
  b=$(grep -oE 'breeding ([0-9.]+) s' <<<"$out" | awk '{print $2}' || true)
  # shader seconds / gens when profiled
  s=$(grep -oE 'shader ([0-9.]+) s' <<<"$out" | awk '{print $2}' || true)
  samples+=("$g")
  evals+=("$e")
  archives+=("$a")
  breeds+=("$b")
  shaders+=("${s:-}")
  echo "run $r: $g gen/s eval=$e archive=$a breed=$b shader=$s"
done

# Sampled clocks for this batch.
if [[ -f "$LOG.clocks" ]]; then
  clock_min=$(sort -n "$LOG.clocks" | head -1)
  clock_max=$(sort -n "$LOG.clocks" | tail -1)
fi

IFS=$'\n' sorted=($(printf '%s\n' "${samples[@]}" | sort -n)); unset IFS
n=${#sorted[@]}
min=${sorted[0]}
med=${sorted[$((n / 2))]}
median_of() {
  local -n arr=$1
  local vals=()
  for v in "${arr[@]}"; do [[ -n "$v" ]] && vals+=("$v"); done
  ((${#vals[@]})) || { echo ""; return; }
  IFS=$'\n' s=($(printf '%s\n' "${vals[@]}" | sort -n)); unset IFS
  echo "${s[$((${#s[@]} / 2))]}"
}
load=$(cut -d' ' -f1 /proc/loadavg)
all=$(IFS=,; echo "${samples[*]}")
echo "RESULT pop=$POP runs=$RUNS gen_s_min=$min gen_s_median=$med all=$all load=$load clocks=$clock_min..$clock_maxMHz"
echo "$(date +%s),$POP,$RUNS,$GENS,$min,$med,$all,$(median_of evals),$(median_of archives),$(median_of breeds),$(median_of shaders),$load,$clock_min,$clock_max" >>"$CSV"
echo "log: $LOG"
