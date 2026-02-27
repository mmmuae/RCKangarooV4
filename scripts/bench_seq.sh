#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <backend:auto|cuda|sass> [range=78] [dp=16] [duration_sec=18] [runs=3] [binary=./rckangaroo]"
  exit 1
fi

BACKEND="$1"
RANGE="${2:-78}"
DP="${3:-16}"
DURATION="${4:-18}"
RUNS="${5:-3}"
BIN="${6:-./rckangaroo}"

if [[ ! -x "$BIN" ]]; then
  echo "error: binary not executable: $BIN"
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
SPEED_FILE="$TMP_DIR/speeds.txt"

echo "benchmark start: backend=$BACKEND range=$RANGE dp=$DP duration=${DURATION}s runs=$RUNS"

for ((i=1; i<=RUNS; i++)); do
  LOG="$TMP_DIR/run_${i}.log"
  echo "run $i/$RUNS ..."
  set +e
  timeout "${DURATION}s" "$BIN" -backend "$BACKEND" -range "$RANGE" -dp "$DP" >"$LOG" 2>&1
  RC=$?
  set -e
  if [[ $RC -ne 0 && $RC -ne 124 ]]; then
    echo "run $i failed with exit code $RC"
    cat "$LOG"
    exit 1
  fi

  SPEED="$(awk '/BENCH: Speed:/ {v=$3} END {print v}' "$LOG")"
  if [[ -z "$SPEED" ]]; then
    echo "run $i has no BENCH speed line"
    cat "$LOG"
    exit 1
  fi
  echo "$SPEED" >> "$SPEED_FILE"
  echo "run $i speed=${SPEED} MKeys/s"
done

AVG="$(awk '{s+=$1} END {if (NR==0) print 0; else printf "%.2f", s/NR}' "$SPEED_FILE")"
MEDIAN="$(sort -n "$SPEED_FILE" | awk '{a[NR]=$1} END {if (NR==0) {print 0} else if (NR%2==1) {print a[(NR+1)/2]} else {printf "%.2f", (a[NR/2]+a[NR/2+1])/2}}')"
MINV="$(sort -n "$SPEED_FILE" | head -n 1)"
MAXV="$(sort -n "$SPEED_FILE" | tail -n 1)"

echo "summary: backend=$BACKEND avg=${AVG} median=${MEDIAN} min=${MINV} max=${MAXV} MKeys/s"
