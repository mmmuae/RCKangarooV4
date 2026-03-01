#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  scripts/bench_dpf_matrix.sh <pubkey> [range=134] [dp=30] [duration_sec=45] [variants="b256g16 b256g24 b256g32 b256g48 b256g64"] [mode=wild]

Example:
  scripts/bench_dpf_matrix.sh 0244... 134 30 60 "b256g24 b256g32 b256g48 b256g64" wild

Notes:
- Requires running from repo root so ./rckangaroo and ./scripts/wdp_type_report.py are available.
- Uses timeout to stop each run and then validates produced WDP types.
USAGE
  exit 0
fi

PUBKEY="${1:-}"
RANGE="${2:-134}"
DPBITS="${3:-30}"
DURATION="${4:-45}"
VARIANTS="${5:-b256g16 b256g24 b256g32 b256g48 b256g64}"
MODE="${6:-wild}"

if [[ -z "$PUBKEY" ]]; then
  echo "error: pubkey is required"
  exit 1
fi

if [[ ! -x ./rckangaroo ]]; then
  echo "error: ./rckangaroo not found/executable"
  exit 1
fi

if [[ ! -x ./scripts/wdp_type_report.py ]]; then
  echo "error: ./scripts/wdp_type_report.py not executable"
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "bench-dpf: mode=$MODE range=$RANGE dp=$DPBITS duration=${DURATION}s variants=$VARIANTS"

for v in $VARIANTS; do
  SPOOL="$WORKDIR/spool_${v}"
  LOG="$WORKDIR/run_${v}.log"
  mkdir -p "$SPOOL"

  echo "=== variant $v ==="
  set +e
  RCK_SASS_STRICT=1 RCK_SASS_VARIANT="$v" timeout "${DURATION}s" \
    ./rckangaroo \
      -dpf-mode "$MODE" \
      -range "$RANGE" -dp "$DPBITS" -start 0 \
      -pubkey "$PUBKEY" \
      -dpf-worker bench \
      -dpf-dir "$SPOOL" \
      -dpf-flush-records 1000000 \
      -dpf-flush-sec 300 \
      >"$LOG" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 && $rc -ne 124 ]]; then
    echo "run failed rc=$rc"
    tail -n 50 "$LOG"
    continue
  fi

  speed_line="$(awk '/EXPORT\['"$MODE"'\]: Speed:/ {line=$0} END {print line}' "$LOG")"
  speed="$(awk '/EXPORT\['"$MODE"'\]: Speed:/ {v=$3} END {print v}' "$LOG")"
  if [[ -z "$speed" ]]; then
    speed="NA"
  fi
  echo "speed_line: ${speed_line:-<none>}"

  file_count=$(find "$SPOOL" -type f -name '*.wdp' | wc -l | tr -d ' ')
  echo "files: $file_count"
  if [[ "$file_count" != "0" ]]; then
    ./scripts/wdp_type_report.py "$SPOOL" | tail -n 5
  fi
  echo "result: variant=$v speed_mkeys=$speed"
  echo

done
