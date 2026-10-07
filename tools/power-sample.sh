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

# summary <file.csv> [from_s] [to_s]: prints the means and the limiter shares
# of the samples whose t_s lies between from_s and to_s. The awk program names
# the columns by number, as in the header written below: $1 t_s, $2 sm_mhz,
# $3 mem_mhz, $4 power_w, $5 power_avg_w, $6 limit_w, $7 temp_c, $8 util,
# $9 reasons, $10 pstate, $11 pcie_gen, $12 apu_ppt_w, $13 cpu_mhz,
# $14 radeon_mhz, $15 radeon_busy.
summary() {
  awk -F, -v from="${2:-0}" -v to="${3:-1e18}" '
    # hex(s) reads a value such as 0x0000000000000004, as in the reasons column.
    function hex(s,   i, v) {
      v = 0; s = tolower(substr(s, 3))
      for (i = 1; i <= length(s); i++) v = v * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1
      return v
    }
    # bit(v, b) is 1 when the bit with the value b is set in v.
    function bit(v, b) { return int(v / b) % 2 }
    # Skip the header line and the samples outside the window.
    NR == 1 { next }
    $1 + 0 < from + 0 || $1 + 0 > to + 0 { next }
    {
      # Sums over every sample in the window.
      n++; sm += $2; pw += $4; ppt += $12; mhz += $13; rmhz += $14; rbusy += $15
      r = hex($9)
      # Sums over the busy samples (util of 90 or more), with the lowest and
      # highest SM clock, the highest temperature and the count of samples with
      # each limiter.
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
      # The times of the first and last sample in the window.
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
  # Print the header comment (lines 2 to 20) as the usage text. Keep the header
  # on those lines.
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
out=$1
shift
# Next come an optional interval in milliseconds, then `--` and the command.
ms=100
if [ $# -gt 0 ] && [ "$1" != "--" ]; then
  ms=$1
  shift
fi
[ "${1:-}" = "--" ] && shift

# Find the Radeon in the hwmon directories: the APU package power (in
# microwatts), the shader clock (in hertz) and the busy percent. A file the
# script cannot read leaves its column empty.
ppt_file= radeon_freq= radeon_busy=
for h in /sys/class/hwmon/hwmon*; do
  if [ "$(cat "$h/name" 2>/dev/null)" = amdgpu ]; then
    [ -r "$h/power1_average" ] && ppt_file=$h/power1_average
    [ -r "$h/freq1_input" ] && radeon_freq=$h/freq1_input
    busy=$h/device/gpu_busy_percent
    [ -r "$busy" ] && radeon_busy=$busy
  fi
done

# The first column is the time since the start. The next ten are the nvidia-smi
# fields in the order of its query below. The last four are the APU power, the
# mean CPU clock and the two Radeon values.
echo "t_s,sm_mhz,mem_mhz,power_w,power_avg_w,limit_w,temp_c,util,reasons,pstate,pcie_gen,apu_ppt_w,cpu_mhz,radeon_mhz,radeon_busy" > "$out"
t0=$EPOCHREALTIME
# nvidia-smi prints one line every $ms milliseconds for GPU 0, the RTX. This
# function appends one CSV row for each line. awk adds the time since the
# start, the hwmon values and the mean `cpu MHz` of /proc/cpuinfo. The
# nvidia-smi line goes in without its spaces.
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
# Stops the sampler: first nvidia-smi and the reading loop, which are its
# children, then the sampler itself.
stop() {
  pkill -P "$sampler" 2>/dev/null
  kill "$sampler" 2>/dev/null
  wait "$sampler" 2>/dev/null
}

if [ $# -gt 0 ]; then
  # Sample while the command runs. The script exits with the command's status.
  "$@"
  status=$?
  stop
  summary "$out"
  exit $status
fi
# With no command, sample until Ctrl-C or SIGTERM, then print the summary.
trap 'stop; summary "$out"; exit 0' INT TERM
wait
