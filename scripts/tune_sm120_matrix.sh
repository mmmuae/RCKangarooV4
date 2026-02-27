#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f Makefile ]]; then
  echo "error: run from project root"
  exit 1
fi

RANGE="${1:-78}"
DP="${2:-16}"
DURATION="${3:-10}"
RUNS="${4:-2}"

declare -a TUNE_GROUPS=(16 24 32)
declare -a TUNE_BLOCKS=(128 256 512)
declare -a TUNE_RREGS=(0 160)
declare -a TUNE_DLCM=("off" "ca")

RESULTS_FILE="tune_sm120_results.tsv"
echo -e "group\tblock\trreg\tdlcm\tavg_mkeys" > "$RESULTS_FILE"

for g in "${TUNE_GROUPS[@]}"; do
  for b in "${TUNE_BLOCKS[@]}"; do
    for r in "${TUNE_RREGS[@]}"; do
      for d in "${TUNE_DLCM[@]}"; do
        echo "==== group=$g block=$b rreg=$r dlcm=$d ===="
        EXTRA_DEFS="-DPNT_GROUP_NEW_GPU=$g -DBLOCK_SIZE_NEW_GPU=$b"
        EXTRA_NVCC="$EXTRA_DEFS"
        if [[ "$d" == "ca" ]]; then
          EXTRA_NVCC="$EXTRA_NVCC -Xptxas -dlcm=ca"
        fi
        if [[ "$r" != "0" ]]; then
          EXTRA_NVCC="$EXTRA_NVCC -Xptxas -maxrregcount=$r"
        fi

        make clean >/dev/null
        CUDA_ARCH_LIST=120 \
        CUDA_MAXRREGCOUNT= \
        CUDA_EXTRA_CCFLAGS="$EXTRA_DEFS" \
        CUDA_EXTRA_NVCCFLAGS="$EXTRA_NVCC" \
        make -j"$(nproc)" all >/tmp/tune_build.log 2>&1 || {
          echo "build failed for g=$g b=$b r=$r d=$d"
          tail -n 40 /tmp/tune_build.log
          continue
        }

        OUT="$(./scripts/bench_seq.sh cuda "$RANGE" "$DP" "$DURATION" "$RUNS" ./rckangaroo)"
        echo "$OUT"
        AVG="$(echo "$OUT" | awk -F 'avg=' '/summary:/ {split($2,a," "); print a[1]}')"
        if [[ -z "$AVG" ]]; then
          echo "warn: no avg parsed"
          continue
        fi
        echo -e "${g}\t${b}\t${r}\t${d}\t${AVG}" >> "$RESULTS_FILE"
      done
    done
  done
done

echo "==== top 10 ===="
sort -t$'\t' -k5,5nr "$RESULTS_FILE" | head -n 10
