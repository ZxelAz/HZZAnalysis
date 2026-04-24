#!/bin/bash
# GATO inference: adds a GATO_bin branch to a categorised ROOT file.

set -euo pipefail

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
GATO_DIR="${HZZ_ROOT}/Categorisation/GATO"

TRIAL="${TRIAL:-trial5}"
N_CATS="${N_CATS:-35}"
MODEL_DIR="${MODEL_DIR:-${GATO_DIR}/Models/trained_models/${TRIAL}/gato/checkpoints/${N_CATS}_bins}"
INPUT_ROOT="${INPUT_ROOT:-/eos/user/z/zhiheng/STXS_samples/allYears/categorisation_qqZZggZZ/stage1p2_combined_with_bdt.root}"
OUTPUT_DIR="${OUTPUT_DIR:-/eos/user/z/zhiheng/STXS_samples/allYears/categorisation_qqZZggZZ}"
OUTPUT_NAME="${OUTPUT_NAME:-stage1p2_combined_with_gato.root}"
TREE_NAME="${TREE_NAME:-Events}"
CHUNK_SIZE="${CHUNK_SIZE:-25 MB}"

set +u
if [ -z "${LCG_VERSION:-}" ]; then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
fi
set -u

export PYTHONPATH="${GATO_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

echo "Running GATO inference and writing GATO_bin branch"
echo "Model dir: ${MODEL_DIR}"
echo "Input:     ${INPUT_ROOT}"
echo "Output:    ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "Bins:      ${N_CATS}"

python -u "${GATO_DIR}/functions/predict_gato_to_root.py" \
    --model-dir "${MODEL_DIR}" \
    --input "${INPUT_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "${OUTPUT_NAME}" \
    --tree-name "${TREE_NAME}" \
    --chunk-size "${CHUNK_SIZE}" \
    --n-cats "${N_CATS}"

echo "GATO inference completed!"
