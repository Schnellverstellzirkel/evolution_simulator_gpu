#!/usr/bin/env bash
# Min-of-N graphical-app benchmark with low-rate host and GPU telemetry.
# Usage: noisy_bench.sh <population> [runs] [generations] [default|serial|workgroup|lane]
# Keep diagnostic profiling disabled for throughput runs. Other experiment
# variables can still be passed through the environment.
set -euo pipefail
POP="${1:?population}"
RUNS="${2:-5}"
GENS="${3:-30}"
KERNEL="${4:-default}"
cd "$(dirname "$0")/.."

case "$KERNEL" in
  default) kernel_env=() ;;
  serial) kernel_env=(EVOLUTION_KERNEL=serial) ;;
  workgroup) kernel_env=(EVOLUTION_KERNEL=workgroup) ;;
  lane) kernel_env=(EVOLUTION_KERNEL=workgroup EVOLUTION_LANE_SHADER=1) ;;
  *) echo "kernel must be default, serial, workgroup, or lane" >&2; exit 2 ;;
esac
REVISION=$(git rev-parse --short HEAD)

mkdir -p runs/bench
LOG="runs/bench/pop${POP}-${KERNEL}-${REVISION}-$(date +%s)-$$.log"
CSV="runs/bench/noisy-v2.csv"
[[ -f "$CSV" ]] || echo "timestamp,revision,kernel,population,runs,generations,gen_s_min,gen_s_median,gen_s_all,eval_s_median,archive_s_median,breed_s_median,shader_s_median,load_before,load_after,gpu_clock_min_mhz,gpu_clock_max_mhz,gpu_util_min_pct,gpu_util_max_pct" >"$CSV"

samples=()
evals=()
archives=()
breeds=()
shaders=()
batch_load_before=
batch_load_after=
clock_min=99999
clock_max=0
exact_cos=approx
[[ -v EVOLUTION_EXACT_COS ]] && exact_cos=exact
lane_mode=off
[[ "$KERNEL" == lane ]] && lane_mode=on
workgroup32=default
workgroup64=off
if [[ -v EVOLUTION_WORKGROUP64 ]]; then
  workgroup32=off
  workgroup64=forced
fi
if [[ -v EVOLUTION_WORKGROUP32 ]]; then
  workgroup32=forced
  workgroup64=off
fi
force_bucket5=off
[[ -v EVOLUTION_FORCE_BUCKET5 ]] && force_bucket5=on
legacy_bucket5=off
[[ -v EVOLUTION_LEGACY_BUCKET5 ]] && legacy_bucket5=on
{
  echo "config: revision=$REVISION kernel=$KERNEL lane=$lane_mode exact_cos=$exact_cos workgroup32=$workgroup32 workgroup64=$workgroup64 gpu_batch=${EVOLUTION_GPU_BATCH:-auto} gpu_chunk=${EVOLUTION_GPU_CHUNK:-4096} pipeline_chunk=${EVOLUTION_PIPELINE_CHUNK:-auto} force_bucket5=$force_bucket5 legacy_bucket5=$legacy_bucket5"
} >>"$LOG"

for ((r = 1; r <= RUNS; r++)); do
  telemetry="$LOG.run$r.telemetry.csv"
  host_load="$LOG.run$r.vmstat"
  load_before=$(cut -d' ' -f1 /proc/loadavg)
  if ((r == 1)); then batch_load_before=$load_before; fi
  # One persistent sampler per device at 1 Hz avoids the CPU/driver overhead
  # of repeatedly starting nvidia-smi during a busy benchmark.
  nvidia-smi --query-gpu=timestamp,clocks.current.sm,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
    --format=csv,noheader,nounits --loop=1 >"$telemetry" 2>&1 &
  gpu_sampler=$!
  vmstat -w 1 >"$host_load" 2>&1 &
  cpu_sampler=$!
  stop_monitors() {
    kill "$gpu_sampler" "$cpu_sampler" 2>/dev/null || true
    wait "$gpu_sampler" "$cpu_sampler" 2>/dev/null || true
  }
  out=$(env -u EVOLUTION_GPU_PROFILE -u EVOLUTION_PROFILE_BREED \
    -u EVOLUTION_BENCH_THROUGHPUT -u EVOLUTION_BENCH_RESPONSIVE -u EVOLUTION_BENCH_DURATION \
    -u EVOLUTION_KERNEL -u EVOLUTION_LANE_SHADER "${kernel_env[@]}" \
    EVOLUTION_SMOKE_POPULATION="$POP" EVOLUTION_BENCH_GENERATIONS="$GENS" \
    taskset -c 4-15 nice -n 5 cargo run --release 2>&1) || {
    echo "$out" | tail -20
    stop_monitors
    exit 1
  }
  stop_monitors
  load_after=$(cut -d' ' -f1 /proc/loadavg)
  batch_load_after=$load_after
  awk -F, '{gsub(/[[:space:]]/, "", $2); if ($2 ~ /^[0-9]+$/) print $2}' "$telemetry" >>"$LOG.clocks"
  awk -F, '{gsub(/[[:space:]%]/, "", $3); if ($3 ~ /^[0-9]+$/) print $3}' "$telemetry" >>"$LOG.utilization"
  {
    echo "===== run $r ====="
    echo "Host load average before/after: $load_before / $load_after; logical CPUs: $(nproc)"
    echo "Host CPU samples: $host_load"
    echo "GPU samples: $telemetry"
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
gpu_util_min=0
gpu_util_max=0
if [[ -f "$LOG.utilization" ]]; then
  gpu_util_min=$(sort -n "$LOG.utilization" | head -1)
  gpu_util_max=$(sort -n "$LOG.utilization" | tail -1)
fi
echo "RESULT revision=$REVISION kernel=$KERNEL pop=$POP runs=$RUNS gen_s_min=$min gen_s_median=$med all=$all load=$load clocks=${clock_min}..${clock_max}MHz gpu_util=$gpu_util_min..$gpu_util_max%"
echo "$(date +%s),$REVISION,$KERNEL,$POP,$RUNS,$GENS,$min,$med,\"$all\",$(median_of evals),$(median_of archives),$(median_of breeds),$(median_of shaders),$batch_load_before,$batch_load_after,$clock_min,$clock_max,$gpu_util_min,$gpu_util_max" >>"$CSV"
echo "log: $LOG"
