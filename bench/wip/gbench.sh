#!/usr/bin/env bash
# Usage: gbench.sh <label> <population> <generations> [extra env...]
set -u
S=/tmp/claude-1000/-home-amipo-workspace-evolutionSimulator/7829a527-0f81-43e0-9ef8-bbe1fd69d2e3/scratchpad
label=$1; pop=$2; gens=$3; shift 3
cd /home/amipo/workspace/evolutionSimulator
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits --loop=1 > $S/$label.smi &
smi=$!
start_load=$(cut -d' ' -f1 /proc/loadavg)
env EVOLUTION_SMOKE_POPULATION=$pop EVOLUTION_BENCH_GENERATIONS=$gens EVOLUTION_BENCH_WARMUP=${WARMUP:-2} "$@" ./target/release/evolution-simulator > $S/$label.log 2>&1
kill $smi
echo "load before $start_load after $(cut -d' ' -f1 /proc/loadavg)" >> $S/$label.log
awk -F', ' '{c+=$2; u+=$3; p+=$4; if($5>t)t=$5; n++} END {printf "gpu: mean SM clock %.0f MHz, util %.0f%%, power %.1f W, max temp %d C, samples %d\n", c/n, u/n, p/n, t, n}' $S/$label.smi >> $S/$label.log
grep -E "Native|GPU profile|load|gpu:" $S/$label.log
