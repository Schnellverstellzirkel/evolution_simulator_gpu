#!/bin/bash
# Runs a command while the CPU and the NVIDIA GPU stay cool: waits until both
# are under 70 C before it starts, and stops the command if either passes
# 85 C. Usage: tools/thermal-guard.sh <command> [args...]
cpu() { for h in /sys/class/hwmon/hwmon*; do [ "$(cat "$h/name")" = k10temp ] && echo $(( $(cat "$h/temp1_input") / 1000 )); done | head -1; }
gpu() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
while [ "$(cpu)" -ge 70 ] || [ "$(gpu)" -ge 70 ]; do
  echo "thermal-guard: CPU $(cpu) C, GPU $(gpu) C, waiting to cool below 70 C" >&2
  sleep 15
done
"$@" &
child=$!
while kill -0 "$child" 2>/dev/null; do
  if [ "$(cpu)" -ge 85 ] || [ "$(gpu)" -ge 85 ]; then
    echo "thermal-guard: CPU $(cpu) C, GPU $(gpu) C, stopping the command" >&2
    kill "$child"
    wait "$child"
    exit 124
  fi
  sleep 2
done
wait "$child"
