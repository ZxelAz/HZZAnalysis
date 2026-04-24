#!/bin/bash
# Wrapper for response_matrix.py — runs fits per (final state × category × STXS bin).
#
# Usage:
#   bash run_response_matrix.sh [METHOD]
#   METHOD ∈ {CutBased, BDT, DNN, GATO}   (default: GATO)

set -euo pipefail

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
METHOD="${1:-GATO}"

SCRIPT_DIR="${HZZ_ROOT}/InferenceModel/functions"
PY_SCRIPT="${SCRIPT_DIR}/response_matrix.py"

# Map method → default categorised input file & branch name.
INPUT_BASE="/eos/user/z/zhiheng/STXS_samples/allYears/categorisation_qqZZggZZ"
case "${METHOD}" in
    CutBased)
        DEFAULT_INPUT="${INPUT_BASE}/stage1p2_combined.root"
        DEFAULT_CATEGORY="run2_category"
        ;;
    BDT)
        DEFAULT_INPUT="${INPUT_BASE}/stage1p2_combined_with_bdt.root"
        DEFAULT_CATEGORY="BDT_category"
        ;;
    DNN)
        DEFAULT_INPUT="${INPUT_BASE}/stage1p2_combined_with_dnn.root"
        DEFAULT_CATEGORY="DNN_category"
        ;;
    GATO)
        DEFAULT_INPUT="${INPUT_BASE}/stage1p2_combined_with_gato.root"
        DEFAULT_CATEGORY="GATO_bin"
        ;;
    *)
        echo "Unknown method '${METHOD}'. Expected one of: CutBased, BDT, DNN, GATO." >&2
        exit 2
        ;;
esac

INPUT_FILE="${INPUT_FILE:-${DEFAULT_INPUT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${HZZ_ROOT}/InferenceModel/Results/${METHOD}}"
VARIABLE="${VARIABLE:-ZZCand_mass}"
CATEGORY_NAME="${CATEGORY_NAME:-${DEFAULT_CATEGORY}}"
BIN_NAME="${BIN_NAME:-HTXS_stage1_2_cat_pTjet30GeV_label}"
MODE_NAME="${MODE_NAME:-production_mode}"
BKG_EVENT_WEIGHT_NAME="${BKG_EVENT_WEIGHT_NAME:-EventWeight_lumi62}"
SIGNAL_EVENT_WEIGHT_NAME="${SIGNAL_EVENT_WEIGHT_NAME:-genWeight}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Running response matrix + DSCB fits"
echo "Method:            ${METHOD}"
echo "Input file:        ${INPUT_FILE}"
echo "Output directory:  ${OUTPUT_DIR}"
echo "Fit variable:      ${VARIABLE}"
echo "Category column:   ${CATEGORY_NAME}"
echo "Bin column:        ${BIN_NAME}"
echo "Mode column:       ${MODE_NAME}"
echo "Bkg weight col:    ${BKG_EVENT_WEIGHT_NAME}"
echo "Sig weight col:    ${SIGNAL_EVENT_WEIGHT_NAME}"
echo "============================================================"

python "${PY_SCRIPT}" \
    --input_file "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --variable "${VARIABLE}" \
    --category_name "${CATEGORY_NAME}" \
    --bin_name "${BIN_NAME}" \
    --mode_name "${MODE_NAME}" \
    --bkg_event_weight_name "${BKG_EVENT_WEIGHT_NAME}" \
    --signal_event_weight_name "${SIGNAL_EVENT_WEIGHT_NAME}"

echo
echo "Done. Outputs written to: ${OUTPUT_DIR}"
echo "Fit plots are in:         ${OUTPUT_DIR}/fit_plots"
