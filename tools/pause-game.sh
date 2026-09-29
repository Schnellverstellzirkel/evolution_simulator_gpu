#!/bin/bash
# Pauses the owner's running game for a speed measurement, runs a command,
# and lets the game resume when the command ends (also on Ctrl-C or an
# error). The game frees the GPU and stops evaluating and breeding while it
# is paused. A pause lasts at most 5 minutes; after one the game runs at
# least 2 minutes before it honors a new request. See src/dev_pause.rs.
# Usage: tools/pause-game.sh <command> [args...]
# Speed measurements: flock -x target/gpu.lock tools/pause-game.sh tools/cpu-slot.sh <bench>
set -u
if [ $# -eq 0 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi
if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
  dir="$XDG_RUNTIME_DIR/evolution-simulator"
else
  dir="${TMPDIR:-/tmp}/evolution-simulator-$(id -u)"
fi
mkdir -p "$dir"
request="$dir/pause"
ack="$dir/paused"
waiting="$dir/waiting"
limit=300

trap 'rm -f "$request"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

field() { sed -n "s/^$2=//p" "$1" 2>/dev/null | head -n 1; }
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
now() { date +%s.%N; }
seconds() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", b - a }'; }

asked=$(now)
printf 'pid=%s\nasked=%s\n' "$$" "$asked" > "$request.tmp$$" && mv "$request.tmp$$" "$request"

paused_at=""
# The kernel keeps the first 15 characters of a process name.
if ! pgrep -x evolution-simul > /dev/null; then
  echo "pause-game: no game is running; running the command without a pause."
else
  deadline=$(( ${asked%.*} + 60 ))
  told_waiting=""
  while :; do
    pid=$(field "$ack" pid)
    since=$(field "$ack" since)
    if alive "$pid" && [ -n "$since" ] && [ "$since" -ge "${asked%.*}" ]; then
      paused_at=$(now)
      echo "pause-game: the game (pid $pid) paused after $(seconds "$asked" "$paused_at") s."
      break
    fi
    honors_at=$(field "$waiting" honors_at)
    if [ -z "$told_waiting" ] && alive "$(field "$waiting" pid)" && [ -n "$honors_at" ]; then
      told_waiting=1
      echo "pause-game: the game rests after its last pause and pauses in $(( honors_at - $(date +%s) )) s."
      deadline=$(( honors_at + 60 ))
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "pause-game: no game acknowledged the pause within 60 s; running the command anyway."
      break
    fi
    sleep 0.2
  done
fi

"$@"
status=$?

if [ -n "$paused_at" ]; then
  ended=$(now)
  still=$(field "$ack" pid)
  rm -f "$request"
  took=$(seconds "$paused_at" "$ended")
  echo "pause-game: the game was paused for $took s while the command ran."
  if ! alive "$still"; then
    if awk -v t="$(seconds "$asked" "$ended")" -v l="$limit" 'BEGIN { exit !(t >= l) }'; then
      echo "pause-game: WARNING: the command ran past the 5 minute limit, so the game resumed partway through. Split the measurement."
    else
      echo "pause-game: WARNING: the game resumed before the command finished (the player pressed Resume now)."
    fi
  fi
fi
exit $status
