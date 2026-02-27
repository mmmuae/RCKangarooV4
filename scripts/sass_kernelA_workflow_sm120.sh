#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./sass_work/sm120}"
mkdir -p "$OUT_DIR"
SM120_DEFS="${SM120_DEFS:--DPNT_GROUP_NEW_GPU=16 -DBLOCK_SIZE_NEW_GPU=256}"
SM120_MAXRREGCOUNT="${SM120_MAXRREGCOUNT:-0}"
SM120_DLCM="${SM120_DLCM:-ca}"

CUBIN="$OUT_DIR/rckangaroo_kernels_sm120.cubin"
PTXAS_LOG="$OUT_DIR/ptxas.log"
RES_LOG="$OUT_DIR/resource_usage.txt"
DISASM="$OUT_DIR/disasm_sm120.sass"
KERNELA_DISASM="$OUT_DIR/kernelA_sm120.sass"

echo "building cubin for sm120 ..."
NVCC_CMD=(nvcc -O3 -Xptxas=-v)
if [[ "$SM120_MAXRREGCOUNT" != "0" ]]; then
  NVCC_CMD+=(-Xptxas -maxrregcount="${SM120_MAXRREGCOUNT}")
fi
if [[ -n "$SM120_DLCM" ]]; then
  NVCC_CMD+=(-Xptxas "-dlcm=${SM120_DLCM}")
fi
if [[ -n "$SM120_DEFS" ]]; then
  # shellcheck disable=SC2206
  DEFS_ARR=($SM120_DEFS)
  NVCC_CMD+=("${DEFS_ARR[@]}")
fi
NVCC_CMD+=(-gencode=arch=compute_120,code=sm_120 -cubin RCGpuCore.cu -o "$CUBIN")
"${NVCC_CMD[@]}" 2>"$PTXAS_LOG"

echo "dumping resource usage ..."
cuobjdump --dump-resource-usage "$CUBIN" > "$RES_LOG"

echo "disassembling cubin ..."
nvdisasm "$CUBIN" > "$DISASM"

echo "extracting KernelA disassembly ..."
awk '
  /^\.text\.KernelA:/ {in_kernel=1}
  /^\/\/--------------------- \.text\./ && in_kernel {exit}
  in_kernel {print}
' "$DISASM" > "$KERNELA_DISASM"

echo "done."
echo "sm120 defs: $SM120_DEFS"
echo "sm120 maxrregcount: $SM120_MAXRREGCOUNT"
echo "sm120 dlcm: $SM120_DLCM"
echo "cubin: $CUBIN"
echo "ptxas: $PTXAS_LOG"
echo "resources: $RES_LOG"
echo "kernelA disasm: $KERNELA_DISASM"
