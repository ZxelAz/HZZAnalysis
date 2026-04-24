#!/usr/bin/env bash
# Rate-comparison plot across categorisation methods.
set -euo pipefail

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
RESULTS="${HZZ_ROOT}/InferenceModel/Results"

# Each method's combine_output directory. Override via env if needed.
CB_DIR="${CB_DIR:-${RESULTS}/CutBased/combine_output}"
BDT_DIR="${BDT_DIR:-${RESULTS}/BDT/combine_output}"
GATO_DIR="${GATO_DIR:-${RESULTS}/GATO/combine_output}"

OUT_DIR="${HZZ_ROOT}/InferenceModel/Plots/comparison"
mkdir -p "${OUT_DIR}"

python "${HZZ_ROOT}/InferenceModel/functions/plot_rate_comparison.py" \
    --cutbased "${CB_DIR}" \
    --bdt      "${BDT_DIR}" \
    --gato     "${GATO_DIR}" \
    --output   "${OUT_DIR}/rate_comparison.png" \
    --x-lo -3 \
    --x-hi  5

echo "Saved: ${OUT_DIR}/rate_comparison.png"
