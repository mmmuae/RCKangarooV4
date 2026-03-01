#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  scripts/build_sass_matrix.sh [sm120_profiles] [sm89_profiles]

Defaults:
  sm120_profiles: "b256g16 b256g24 b256g32 b256g48 b256g64"
  sm89_profiles:  "b256g24"

This builds variant cubins and refreshes default cubins:
  - sass/sm120/rckangaroo_kernels_b*g*.cubin
  - sass/sm89/rckangaroo_kernels_b*g*.cubin
  - sass/sm120/rckangaroo_kernels.cubin (copied from highest score block*group)
  - sass/sm89/rckangaroo_kernels.cubin (copied from highest score block*group)
USAGE
  exit 0
fi

SM120_PROFILES="${1:-b256g16 b256g24 b256g32 b256g48 b256g64}"
SM89_PROFILES="${2:-b256g24}"

build_for_arch() {
  local arch="$1"
  local arch_dir="$2"
  local profiles="$3"
  local best=0
  local best_file=""

  for p in $profiles; do
    if [[ ! "$p" =~ ^b([0-9]+)g([0-9]+)$ ]]; then
      echo "error: invalid profile '$p' (expected b<block>g<group>)"
      exit 1
    fi
    local b="${BASH_REMATCH[1]}"
    local g="${BASH_REMATCH[2]}"
    local out="sass/${arch_dir}/rckangaroo_kernels_b${b}g${g}.cubin"
    ./scripts/build_sass_cubin.sh "$arch" "$g" "$out" "$b"
    local score=$(( b * g ))
    if (( score > best )); then
      best=$score
      best_file="$out"
    fi
  done

  if [[ -z "$best_file" ]]; then
    echo "error: no profiles built for $arch"
    exit 1
  fi

  cp "$best_file" "sass/${arch_dir}/rckangaroo_kernels.cubin"
  echo "default ${arch_dir} cubin <- $(basename "$best_file")"
}

build_for_arch sm_120 sm120 "$SM120_PROFILES"
build_for_arch sm_89 sm89 "$SM89_PROFILES"
