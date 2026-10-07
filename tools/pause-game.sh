#!/bin/bash
# Pauses the owner's running game for a speed measurement, runs a command,
# and lets the game resume when the command ends (also on Ctrl-C or an
# error). The game frees the GPU and stops evaluating and breeding while it
# is paused. A pause lasts at most 5 minutes; after one the game runs at
# least 2 minutes before it honors a new request. See src/dev_pause.rs.
# Usage: tools/pause-game.sh <command> [args...]
# Speed measurements: flock -x target/gpu.lock tools/pause-game.sh <bench>
#
# The script and the game talk through files in one directory. The script
# writes `pause` to ask for a pause. The game writes `paused` when its engines
# are closed, and `waiting` while it rests after an earlier pause. The script
# exits with the status of the command.
set -u
if [ $# -eq 0 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi
# The same directory as `dir` in src/dev_pause.rs.
if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
  dir="$XDG_RUNTIME_DIR/evolution-simulator"
else
  dir="${TMPDIR:-/tmp}/evolution-simulator-$(id -u)"
fi
mkdir -p "$dir"
request="$dir/pause"
ack="$dir/paused"
waiting="$dir/waiting"
# The longest a pause lasts, in seconds. It is `LONGEST` in src/dev_pause.rs.
limit=300

# Withdraw the request on every exit, so the game resumes. Ctrl-C and SIGTERM
# exit with the usual status (128 plus the signal number) so this trap runs.
trap 'rm -f "$request"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# field <file> <key>: the value of the first `key=value` line in the file.
field() { sed -n "s/^$2=//p" "$1" 2>/dev/null | head -n 1; }
# alive <pid>: succeeds when the pid is not empty and that process exists.
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
# now: the Unix time in seconds, with a fraction.
now() { date +%s.%N; }
# seconds <from> <to>: the time between two values of `now`, with one decimal.
seconds() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", b - a }'; }

# The request holds this pid and the time. The game honors a request once and
# tells requests apart by their file time and contents. The file is written
# through a temporary file and a rename, so the game never reads half of it.
asked=$(now)
printf 'pid=%s\nasked=%s\n' "$$" "$asked" > "$request.tmp$$" && mv "$request.tmp$$" "$request"

paused_at=""
# The game is the process evolution-simulator. The kernel keeps the first 15
# characters of a process name, so pgrep matches evolution-simul.
if ! pgrep -x evolution-simul > /dev/null; then
  echo "pause-game: no game is running; running the command without a pause."
else
  # Wait up to 60 s for the game to answer.
  deadline=$(( ${asked%.*} + 60 ))
  told_waiting=""
  while :; do
    # The answer is the `paused` file. It counts when its game is alive and
    # its `since` time (Unix seconds) is not before this request.
    pid=$(field "$ack" pid)
    since=$(field "$ack" since)
    if alive "$pid" && [ -n "$since" ] && [ "$since" -ge "${asked%.*}" ]; then
      paused_at=$(now)
      echo "pause-game: the game (pid $pid) paused after $(seconds "$asked" "$paused_at") s."
      break
    fi
    # A game that rests after its last pause says when it will honor the
    # request. Then the 60 s start from that time.
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

# Run the measurement.
"$@"
status=$?

if [ -n "$paused_at" ]; then
  ended=$(now)
  still=$(field "$ack" pid)
  rm -f "$request"
  took=$(seconds "$paused_at" "$ended")
  echo "pause-game: the game was paused for $took s while the command ran."
  # The game removes `paused` when it resumes. If the file or its process is
  # gone, the game resumed before the command ended. If the time since the
  # request reached the limit, the limit ended the pause. Otherwise the player
  # pressed Resume now.
  if ! alive "$still"; then
    if awk -v t="$(seconds "$asked" "$ended")" -v l="$limit" 'BEGIN { exit !(t >= l) }'; then
      echo "pause-game: WARNING: the command ran past the 5 minute limit, so the game resumed partway through. Split the measurement."
    else
      echo "pause-game: WARNING: the game resumed before the command finished (the player pressed Resume now)."
    fi
  fi
fi
exit $status
