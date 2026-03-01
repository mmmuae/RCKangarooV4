#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  scripts/build_sass_cubin.sh <arch> <group_cnt> [output]

Examples:
  scripts/build_sass_cubin.sh sm_120 64 sass/sm120/rckangaroo_kernels.cubin
  scripts/build_sass_cubin.sh sm_120 32 sass/sm120/rckangaroo_kernels_g32.cubin

Notes:
- pure SASS only: this script builds cubin files from RCGpuCore.cu.
- group_cnt sets PNT_GROUP_NEW_GPU at compile time.
USAGE
  exit 0
fi

ARCH="${1:-}"
GROUP="${2:-}"
OUT="${3:-}"

if [[ -z "$ARCH" || -z "$GROUP" ]]; then
  echo "error: arch and group_cnt are required"
  exit 1
fi

if ! [[ "$GROUP" =~ ^[0-9]+$ ]]; then
  echo "error: group_cnt must be integer"
  exit 1
fi
if (( GROUP < 8 || GROUP > 128 || GROUP % 8 != 0 )); then
  echo "error: group_cnt must be divisible by 8 in [8, 128]"
  exit 1
fi

case "$ARCH" in
  sm_120) ARCH_DIR="sm120" ;;
  sm_89) ARCH_DIR="sm89" ;;
  *)
    echo "error: unsupported arch '$ARCH' (expected sm_120 or sm_89)"
    exit 1
    ;;
esac

if [[ -z "$OUT" ]]; then
  OUT="sass/${ARCH_DIR}/rckangaroo_kernels_g${GROUP}.cubin"
fi

if [[ ! -f RCGpuCore.cu ]]; then
  echo "error: RCGpuCore.cu is missing"
  exit 1
fi

CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
NVCC="${NVCC:-$CUDA_PATH/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then
  echo "error: nvcc not found at $NVCC"
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

echo "build: arch=$ARCH group=$GROUP out=$OUT"
"$NVCC" -O3 -std=c++17 -I"$CUDA_PATH/include" -arch="$ARCH" --cubin \
  -DPNT_GROUP_NEW_GPU="$GROUP" \
  RCGpuCore.cu -o "$OUT"

ls -lh "$OUT"
