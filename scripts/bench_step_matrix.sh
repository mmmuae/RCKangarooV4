#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  scripts/bench_step_matrix.sh <pubkey> [range=134] [dp=30] [start=0] [duration_sec=45] [variants="b256g16 b256g24 b256g32 b256g48 b256g64"] [modes="main wild tame both"]

Example:
  scripts/bench_step_matrix.sh 0244... 134 30 0 60 "b256g24 b256g32 b256g48 b256g64" "main wild tame both"

Notes:
- Runs each (variant,mode) with timeout and extracts speed + Err.
- For DPF modes (wild/tame/both), validates output files via wdp_type_report.py.
- Requires ./rckangaroo and ./scripts/wdp_type_report.py from repo root.
USAGE
  exit 0
fi

PUBKEY="${1:-}"
RANGE="${2:-134}"
DPBITS="${3:-30}"
START_HEX="${4:-0}"
DURATION="${5:-45}"
VARIANTS="${6:-b256g16 b256g24 b256g32 b256g48 b256g64}"
MODES="${7:-main wild tame both}"

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

echo "bench-step: range=$RANGE dp=$DPBITS start=$START_HEX duration=${DURATION}s"
echo "variants=$VARIANTS"
echo "modes=$MODES"

for v in $VARIANTS; do
  echo "=== variant $v ==="
  for mode in $MODES; do
    SPOOL="$WORKDIR/spool_${v}_${mode}"
    LOG="$WORKDIR/run_${v}_${mode}.log"
    mkdir -p "$SPOOL"

    base_cmd=(./rckangaroo -range "$RANGE" -dp "$DPBITS" -start "$START_HEX" -pubkey "$PUBKEY")
    if [[ "$mode" == "wild" || "$mode" == "tame" || "$mode" == "both" ]]; then
      base_cmd+=(
        -dpf-mode "$mode"
        -dpf-worker bench
        -dpf-dir "$SPOOL"
        -dpf-flush-records 1000000
        -dpf-flush-sec 300
      )
    fi

    echo "-- mode $mode --"
    set +e
    RCK_SASS_STRICT=1 RCK_SASS_VARIANT="$v" timeout "${DURATION}s" "${base_cmd[@]}" >"$LOG" 2>&1
    rc=$?
    set -e

    if [[ $rc -ne 0 && $rc -ne 124 ]]; then
      echo "run failed rc=$rc"
      tail -n 50 "$LOG"
      continue
    fi

    if [[ "$mode" == "main" ]]; then
      speed_line="$(awk '/^MAIN: Speed:/ {line=$0} END {print line}' "$LOG")"
      speed_val="$(awk '/^MAIN: Speed:/ {v=$3} END {print v}' "$LOG")"
      err_val="$(awk -F'Err: ' '/^MAIN: Speed:/ {split($2,a,/,/); v=a[1]} END {print v}' "$LOG")"
    else
      speed_line="$(awk '/^EXPORT\['"$mode"'\]: Speed:/ {line=$0} END {print line}' "$LOG")"
      speed_val="$(awk '/^EXPORT\['"$mode"'\]: Speed:/ {v=$3} END {print v}' "$LOG")"
      err_val="$(awk -F'Err: ' '/^EXPORT\['"$mode"'\]: Speed:/ {split($2,a,/,/); v=a[1]} END {print v}' "$LOG")"
    fi
    if [[ -z "$speed_val" ]]; then
      speed_val="NA"
    fi
    if [[ -z "$err_val" ]]; then
      err_val="NA"
    fi
    echo "speed_line: ${speed_line:-<none>}"

    if [[ "$mode" == "wild" || "$mode" == "tame" || "$mode" == "both" ]]; then
      file_count=$(find "$SPOOL" -type f -name '*.wdp' | wc -l | tr -d ' ')
      echo "wdp_files=$file_count"
      if [[ "$file_count" != "0" ]]; then
        ./scripts/wdp_type_report.py "$SPOOL" | tail -n 5
      fi
    fi

    echo "result: variant=$v mode=$mode speed_mkeys=$speed_val err=$err_val"
    if [[ "$err_val" != "0" && "$err_val" != "NA" ]]; then
      echo "warning: non-zero Err observed for variant=$v mode=$mode"
    fi
    echo
  done
done
