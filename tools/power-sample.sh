#!/bin/bash
# Samples the RTX and the APU for the power rows in docs/building.md: SM and
# memory clock, power (instant and averaged), the live power limit, the
# limiter reasons, temperature, utilization, the P-state, the PCIe link, the
# APU package power (PPT, amdgpu hwmon), the mean CPU clock, and the
# Radeon's shader clock and busy percent.
#
# Usage:
#   tools/power-sample.sh <out.csv> [interval_ms=100] -- <command...>
#       samples while the command runs, then prints a summary of the run
#   tools/power-sample.sh <out.csv> [interval_ms=100]
#       samples until Ctrl-C, then prints the summary
#   tools/power-sample.sh --summary <file.csv> [from_s] [to_s]
#       prints the summary of a window of an earlier file
#
# The summary gives means over all samples and over the busy ones (GPU
# utilization at least 90%), and the share of busy samples each limiter
# was active. Reason bits: 0x1 idle, 0x4 SW power cap, 0x8 HW slowdown,
# 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake.
set -u

summary() {
  awk -F, -v from="${2:-0}" -v to="${3:-1e18}" '
    function hex(s,   i, v) {
      v = 0; s = tolower(substr(s, 3))
      for (i = 1; i <= length(s); i++) v = v * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1
      return v
    }
    function bit(v, b) { return int(v / b) % 2 }
    NR == 1 { next }
    $1 + 0 < from + 0 || $1 + 0 > to + 0 { next }
    {
      n++; sm += $2; pw += $4; ppt += $12; mhz += $13; rmhz += $14; rbusy += $15
      r = hex($9)
      if ($8 + 0 >= 90) {
        b++; bsm += $2; bpw += $4; bavg += $5; blim += $6; bppt += $12; bmhz += $13
        btemp += $7; if ($7 + 0 > tmax) tmax = $7
        if (bmin == "" || $2 + 0 < bmin) bmin = $2 + 0
        if ($2 + 0 > bmax) bmax = $2 + 0
        if (bit(r, 4)) cap++
        if (bit(r, 8)) hws++
        if (bit(r, 32)) swt++
        if (bit(r, 64)) hwt++
        if (bit(r, 128)) brake++
      }
      if (t0 == "") t0 = $1; t1 = $1
    }
    END {
      if (n == 0) { print "no samples in the window"; exit 1 }
      printf "window %.1f to %.1f s, %d samples: SM %.0f MHz, GPU %.1f W, APU PPT %.1f W, CPU %.0f MHz, Radeon %.0f MHz at %.0f%% busy\n",
        t0, t1, n, sm / n, pw / n, ppt / n, mhz / n, rmhz / n, rbusy / n
      if (b == 0) { print "no busy samples (utilization under 90%)"; exit 0 }
      printf "busy, %d samples: SM %.0f MHz (%d to %d), GPU %.1f W (averaged %.1f W), limit %.1f W, %.0f C (max %d), APU PPT %.1f W, CPU %.0f MHz\n",
        b, bsm / b, bmin, bmax, bpw / b, bavg / b, blim / b, btemp / b, tmax, bppt / b, bmhz / b
      printf "limiters over busy samples: power cap %.0f%%, SW thermal %.0f%%, HW slowdown %.0f%%, HW thermal %.0f%%, power brake %.0f%%\n",
        100 * cap / b, 100 * swt / b, 100 * hws / b, 100 * hwt / b, 100 * brake / b
    }' "$1"
}

if [ "${1:-}" = "--summary" ]; then
  shift
  [ $# -ge 1 ] || { echo "usage: $0 --summary <file.csv> [from_s] [to_s]" >&2; exit 2; }
  summary "$@"
  exit
fi
if [ $# -lt 1 ]; then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
out=$1
shift
ms=100
if [ $# -gt 0 ] && [ "$1" != "--" ]; then
  ms=$1
  shift
fi
[ "${1:-}" = "--" ] && shift

ppt_file= radeon_freq= radeon_busy=
for h in /sys/class/hwmon/hwmon*; do
  if [ "$(cat "$h/name" 2>/dev/null)" = amdgpu ]; then
    [ -r "$h/power1_average" ] && ppt_file=$h/power1_average
    [ -r "$h/freq1_input" ] && radeon_freq=$h/freq1_input
    busy=$h/device/gpu_busy_percent
    [ -r "$busy" ] && radeon_busy=$busy
  fi
done

echo "t_s,sm_mhz,mem_mhz,power_w,power_avg_w,limit_w,temp_c,util,reasons,pstate,pcie_gen,apu_ppt_w,cpu_mhz,radeon_mhz,radeon_busy" > "$out"
t0=$EPOCHREALTIME
sample() {
  nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.mem,power.draw.instant,power.draw.average,enforced.power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,pstate,pcie.link.gen.current \
    --format=csv,noheader,nounits -lms "$ms" |
    while IFS= read -r line; do
      awk -v t0="$t0" -v now="$EPOCHREALTIME" -v g="${line// /}" -v ppt="$ppt_file" -v rf="$radeon_freq" -v rb="$radeon_busy" '
        /^cpu MHz/ { s += $4; n++ }
        END {
          p = ""
          f = ""; b = ""
          if (ppt != "" && (getline v < ppt) > 0) p = sprintf("%.1f", v / 1e6)
          if (rf != "" && (getline v < rf) > 0) f = sprintf("%.0f", v / 1e6)
          if (rb != "" && (getline v < rb) > 0) b = v + 0
          printf "%.3f,%s,%s,%.0f,%s,%s\n", now - t0, g, p, n ? s / n : 0, f, b
        }' /proc/cpuinfo
    done >> "$out"
}
sample &
sampler=$!
stop() {
  pkill -P "$sampler" 2>/dev/null
  kill "$sampler" 2>/dev/null
  wait "$sampler" 2>/dev/null
}

if [ $# -gt 0 ]; then
  "$@"
  status=$?
  stop
  summary "$out"
  exit $status
fi
trap 'stop; summary "$out"; exit 0' INT TERM
wait
