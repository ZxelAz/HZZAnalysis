#!/usr/bin/env bash
# Build a datacard from an existing workspace.root + pickles.
#
# Usage:
#   bash run_make_datacard.sh [METHOD]
#   METHOD ∈ {CutBased, BDT, DNN, GATO}  (default: GATO)
set -euo pipefail

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
METHOD="${1:-GATO}"
HZZ_PYTHON="${HZZ_PYTHON:-python}"

TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${METHOD}"

EPS="${TRIAL_DIR}/epsilonA.pkl"
NBKG="${TRIAL_DIR}/N_bkg_category.pkl"
WS="${TRIAL_DIR}/workspace.root"
OUT="${TRIAL_DIR}/datacard.txt"

# STXS bins to include (comma-separated); empty = all.
PROCESSES_CSV=""
CATEGORIES_CSV=""

"${HZZ_PYTHON}" "${HZZ_ROOT}/InferenceModel/functions/make_datacard.py" \
    --epsilonA        "${EPS}" \
    --N_bkg_category  "${NBKG}" \
    --workspace       "${WS}" \
    --output          "${OUT}" \
    --processes-list  "${PROCESSES_CSV}" \
    --categories-list "${CATEGORIES_CSV}"

echo "Datacard written to: ${OUT}"
