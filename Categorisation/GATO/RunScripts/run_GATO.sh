#!/bin/bash
# GATO Optimization Script for HZZ STXS Classification.

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
GATO_DIR="${HZZ_ROOT}/Categorisation/GATO"
SCRIPT="${GATO_DIR}/functions/GATO_HZZ.py"

# Configuration
DATA_PATH="${DATA_PATH:-/eos/user/z/zhiheng/STXS_samples/GATOdata.npz}"
OUTPUT_NAME="${OUTPUT_NAME:-trial5}"
GATO_BINS="${GATO_BINS:-35}"   # space-separated list, e.g. "10 15 20 26"
EPOCHS="${EPOCHS:-1000}"

# Loss weights
LAM_YIELD="${LAM_YIELD:-0.0}"
LAM_UNC="${LAM_UNC:-0.0}"
THR_YIELD="${THR_YIELD:-5.0}"
THR_UNC="${THR_UNC:-0.20}"

set -euo pipefail

echo "Setting up environment..."
set +u
if [ -z "${LCG_VERSION:-}" ]; then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
fi
set -u

# Put bundled gatohep package on PYTHONPATH so 'from gatohep...' resolves.
export PYTHONPATH="${GATO_DIR}:${PYTHONPATH:-}"

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

# Model outputs go into Categorisation/GATO/Models/${OUTPUT_NAME}/.
MODELS_DIR="${GATO_DIR}/Models"
mkdir -p "${MODELS_DIR}"
cd "${MODELS_DIR}"   # GATO_HZZ.py writes to ./trained_models/<out> relative to CWD

echo "Starting GATO optimization..."
echo "Data path:  $DATA_PATH"
echo "Output:     $OUTPUT_NAME"
echo "GATO bins:  $GATO_BINS"
echo "Epochs:     $EPOCHS"
echo ""

python -u "${SCRIPT}" \
    --data-path "$DATA_PATH" \
    --epochs $EPOCHS \
    --gato-bins $GATO_BINS \
    --lam-yield $LAM_YIELD \
    --lam-unc $LAM_UNC \
    --thr-yield $THR_YIELD \
    --thr-unc $THR_UNC \
    --out "$OUTPUT_NAME"

echo ""
echo "GATO optimization completed!"
echo "Results saved to: ${MODELS_DIR}/trained_models/${OUTPUT_NAME}/"
