#!/usr/bin/env bash
# Produces a single figure with all 41 1D scan subplots.
#
# Usage:
#   bash run_plot_all_1D_scans.sh [METHOD]
#   METHOD ∈ {CutBased, BDT, DNN, GATO}  (default: BDT)
set -euo pipefail

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
METHOD="${1:-BDT}"
TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${METHOD}"
PLOT_SCRIPT="${HZZ_ROOT}/InferenceModel/functions/plot_all_1D_scans.py"
COMBINE_OUT_DIR="${COMBINE_OUT_DIR:-combine_output}"
OUTPUT="${TRIAL_DIR}/${COMBINE_OUT_DIR}/all_1D_scans.png"


mkdir -p "${TRIAL_DIR}/${COMBINE_OUT_DIR}"

echo "============================================================"
echo "Plotting all 1D scans → ${OUTPUT}"
echo "============================================================"

python3 "${PLOT_SCRIPT}" \
    --trial-dir "${TRIAL_DIR}" \
    --combine-out-dir "${COMBINE_OUT_DIR}" \
    --output    "${OUTPUT}"

echo "Done. Saved: ${OUTPUT}"
