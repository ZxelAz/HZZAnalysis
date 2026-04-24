#!/bin/bash

# Data filtering script for 4 campaigns
# Usage: bash parallel_filter_data.sh

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT="${HZZ_ROOT}/DataPreprocessing/functions/filter_dataframe_forData.py"

run_campaign() {
    local CAMPAIGN_NAME="$1"
    local INPUT_FILE="$2"
    local OUTPUT_FILE="$3"
    local BTAG_THRESHOLD="$4"

    echo ""
    echo "========================================================"
    echo "Campaign: ${CAMPAIGN_NAME}"
    echo "Input file: ${INPUT_FILE}"
    echo "Output file: ${OUTPUT_FILE}"
    echo "btagThreshold: ${BTAG_THRESHOLD}"
    echo "========================================================"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "✗ ERROR: Input file not found: ${INPUT_FILE}"
        return 1
    fi

    # Create output directory if needed
    mkdir -p "$(dirname "$OUTPUT_FILE")"

    echo "Processing ${CAMPAIGN_NAME}..."
    echo ""

    # Run filter script (outputPath should be without .root since script appends it)
    local output_path="${OUTPUT_FILE%.root}"
    
    python "$SCRIPT" \
        --file "$INPUT_FILE" \
        --outputPath "$output_path" \
        --btagThreshold "$BTAG_THRESHOLD"
    
    local status=$?

    if [ $status -eq 0 ]; then
        if [ -f "$OUTPUT_FILE" ]; then
            local file_size
            file_size=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
            echo ""
            echo "✓ ${CAMPAIGN_NAME} completed successfully (${file_size})"
        else
            echo ""
            echo "✗ ${CAMPAIGN_NAME}: Output file not created"
            echo "========================================================"
            return 1
        fi
    else
        echo ""
        echo "✗ ${CAMPAIGN_NAME} failed (exit code: ${status})"
        echo "========================================================"
        return 1
    fi
    echo "========================================================"
}

# Format: "campaign_name|input_file|output_file|btag_threshold"
CAMPAIGNS=(
    "2022preEE|/eos/user/z/zhiheng/STXS_samples/2022Data/2022.root|/eos/user/z/zhiheng/STXS_samples/2022Data/filtered_2022.root|0.2421"
    "2022postEE|/eos/user/z/zhiheng/STXS_samples/2022Data/2022EE.root|/eos/user/z/zhiheng/STXS_samples/2022Data/filtered_2022EE.root|0.2386"
    "2023preBPix|/eos/user/z/zhiheng/STXS_samples/2023Data/2023preBPix.root|/eos/user/z/zhiheng/STXS_samples/2023Data/filtered_2023preBPix.root|0.1917"
    "2023postBPix|/eos/user/z/zhiheng/STXS_samples/2023Data/2023postBPix.root|/eos/user/z/zhiheng/STXS_samples/2023Data/filtered_2023postBPix.root|0.1919"
)

echo "Starting data filtering for 4 campaigns..."
echo ""

failed_campaigns=0

for campaign in "${CAMPAIGNS[@]}"; do
    IFS='|' read -r CAMPAIGN_NAME INPUT_FILE OUTPUT_FILE BTAG_THRESHOLD <<< "$campaign"
    run_campaign "$CAMPAIGN_NAME" "$INPUT_FILE" "$OUTPUT_FILE" "$BTAG_THRESHOLD"
    status=$?
    
    if [ $status -ne 0 ]; then
        failed_campaigns=$((failed_campaigns + 1))
    fi
done

echo ""
echo "=========================================================="
if [ $failed_campaigns -eq 0 ]; then
    echo "✓ All campaigns completed successfully."
else
    echo "✗ Warning: $failed_campaigns campaign(s) failed."
fi
echo "=========================================================="
