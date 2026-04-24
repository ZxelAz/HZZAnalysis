#!/bin/bash

# Script to run STXS categorization on filtered HZZ data

# Define paths
: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT_DIR="${HZZ_ROOT}/Categorisation/CutBased/functions"
INPUT_PATHS="/eos/user/z/zhiheng/STXS_samples/2022samples/preEE/root_data,/eos/user/z/zhiheng/STXS_samples/2022samples/postEE/root_data,/eos/user/z/zhiheng/STXS_samples/2023samples/preBPix/root_data,/eos/user/z/zhiheng/STXS_samples/2023samples/postBPix/root_data"
YEARS="2022,2022EE,2023preBPix,2023postBPix"
OUTPUT_PATH="/eos/user/z/zhiheng/STXS_samples/allYears/categorisation_qqZZggZZ"
TREE_NAME="Events"
MODE="prduction+qqZZ+ggZZ"
RUN2SCALE=True

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_PATH}"

# Run the categorization script
echo "Running STXS categorization..."
echo "Input paths: ${INPUT_PATHS}"
echo "Years:       ${YEARS}"
echo "Output: ${OUTPUT_PATH}"
echo "Tree:   ${TREE_NAME}"
echo "Mode:   ${MODE}"
echo ""

python "${SCRIPT_DIR}/run2_categorisation.py" \
    --input_paths "${INPUT_PATHS}" \
    --years "${YEARS}" \
    --output_path "${OUTPUT_PATH}" \
    --tree_name "${TREE_NAME}" \
    --mode "${MODE}" \
    --run2scale "${RUN2SCALE}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Categorization completed successfully!"
    echo "Output files saved to: ${OUTPUT_PATH}"
    ls -lh "${OUTPUT_PATH}"
else
    echo ""
    echo "✗ Categorization failed!"
    exit 1
fi
