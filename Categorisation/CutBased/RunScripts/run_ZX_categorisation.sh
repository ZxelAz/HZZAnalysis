#!/bin/bash

# Script to run STXS categorisation on the filtered ZX background files

# Define paths
: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT_DIR="${HZZ_ROOT}/Categorisation/CutBased/functions"
BASE_DIR="/eos/user/z/zhiheng/STXS_samples"

INPUT_FILES="${BASE_DIR}/2022Data/filtered_ZX_2022.root,${BASE_DIR}/2022Data/filtered_ZX_2022EE.root,${BASE_DIR}/2023Data/filtered_ZX_2023preBPix.root,${BASE_DIR}/2023Data/filtered_ZX_2023postBPix.root"
YEARS="2022,2022EE,2023preBPix,2023postBPix"
OUTPUT_PATH="${BASE_DIR}/allYears/ZX_categorisation"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_PATH}"

# Run the categorisation script
echo "Running ZX background categorisation..."
echo ""
echo "Input files:"
IFS=',' read -ra FILES <<< "$INPUT_FILES"
for f in "${FILES[@]}"; do
    echo "  $f"
done
echo ""
echo "Years:  ${YEARS}"
echo "Output: ${OUTPUT_PATH}"
echo ""

python "${SCRIPT_DIR}/run_ZX_categorisation.py" \
    --input_files "${INPUT_FILES}" \
    --years "${YEARS}" \
    --output_path "${OUTPUT_PATH}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ ZX categorisation completed successfully!"
    echo "Output files saved to: ${OUTPUT_PATH}"
    ls -lh "${OUTPUT_PATH}"
else
    echo ""
    echo "✗ ZX categorisation failed!"
    exit 1
fi
