#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <pubkey_hex> <start_hex> <range_bits> <dp_bits> [binary=./rckangaroo]"
  exit 1
fi

PUBKEY="$1"
START="$2"
RANGE="$3"
DP="$4"
BIN="${5:-./rckangaroo}"

if [[ ! -x "$BIN" ]]; then
  echo "error: binary not executable: $BIN"
  exit 1
fi

exec "$BIN" -pubkey "$PUBKEY" -start "$START" -range "$RANGE" -dp "$DP"
