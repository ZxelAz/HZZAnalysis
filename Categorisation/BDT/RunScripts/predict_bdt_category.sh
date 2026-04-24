#!/bin/bash

set -euo pipefail

# -----------------------------
# User configuration
# -----------------------------
: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
TRIAL="${TRIAL:-trial53}"
MODEL_PATH="${HZZ_ROOT}/Categorisation/BDT/Models/${TRIAL}/model.pkl"
INPUT_ROOT="/eos/user/z/zhiheng/STXS_samples/allYears/categorisation_qqZZggZZ/stage1p2_combined.root"
OUTPUT_DIR="/eos/user/z/zhiheng/STXS_samples/allYears/categorisation_qqZZggZZ"
OUTPUT_NAME="stage1p2_combined_with_bdt.root"
TREE_NAME="Events"
CHUNK_SIZE="25 MB"
PREDICT_DEVICE="${PREDICT_DEVICE:-auto}"

# Optional: override features explicitly (if empty, features from model metadata are used)
# FEATURES=(ZZCand_pt ZZCand_eta ZZCand_phi ...)

# -----------------------------
# Environment setup
# -----------------------------
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
set -u

export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

echo "Running BDT inference and writing BDT_category branch"
echo "Model:  ${MODEL_PATH}"
echo "Input:  ${INPUT_ROOT}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "Tree:   ${TREE_NAME}"

CMD=(
    python -u -m Categorisation.BDT.functions.predict_to_root
    --model "${MODEL_PATH}"
    --input "${INPUT_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --output-name "${OUTPUT_NAME}"
    --tree-name "${TREE_NAME}"
    --chunk-size "${CHUNK_SIZE}"
    --predict-device "${PREDICT_DEVICE}"
)

# Uncomment to force feature list from shell instead of model metadata
# if [[ ${#FEATURES[@]} -gt 0 ]]; then
#     CMD+=(--features "${FEATURES[@]}")
# fi

"${CMD[@]}"

echo "Done. Output ROOT written to ${OUTPUT_DIR}"
