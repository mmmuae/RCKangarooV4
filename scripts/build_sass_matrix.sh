#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  scripts/build_sass_matrix.sh [sm120_groups] [sm89_groups]

Defaults:
  sm120_groups: "16 24 32 48 64"
  sm89_groups:  "24"

This builds variant cubins and refreshes default cubins:
  - sass/sm120/rckangaroo_kernels_g*.cubin
  - sass/sm89/rckangaroo_kernels_g*.cubin
  - sass/sm120/rckangaroo_kernels.cubin (copied from highest sm120 group)
  - sass/sm89/rckangaroo_kernels.cubin (copied from highest sm89 group)
USAGE
  exit 0
fi

SM120_GROUPS="${1:-16 24 32 48 64}"
SM89_GROUPS="${2:-24}"

build_for_arch() {
  local arch="$1"
  local arch_dir="$2"
  local groups="$3"
  local best=0
  local best_file=""

  for g in $groups; do
    ./scripts/build_sass_cubin.sh "$arch" "$g" "sass/${arch_dir}/rckangaroo_kernels_g${g}.cubin"
    if (( g > best )); then
      best=$g
      best_file="sass/${arch_dir}/rckangaroo_kernels_g${g}.cubin"
    fi
  done

  if [[ -z "$best_file" ]]; then
    echo "error: no groups built for $arch"
    exit 1
  fi

  cp "$best_file" "sass/${arch_dir}/rckangaroo_kernels.cubin"
  echo "default ${arch_dir} cubin <- $(basename "$best_file")"
}

build_for_arch sm_120 sm120 "$SM120_GROUPS"
build_for_arch sm_89 sm89 "$SM89_GROUPS"
