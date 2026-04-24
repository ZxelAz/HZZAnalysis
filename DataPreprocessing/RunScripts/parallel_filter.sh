#!/bin/bash

# Parallel filtering script for multiple datasets
# Usage: bash parallel_filter.sh

# /eos/user/a/atarabin/STXS_samples/PROD_samplesNano_2022EE_MC_8d4c03f7
# /eos/user/a/atarabin/STXS_samples/PROD_samplesNano_2022_MC_8d4c03f7
# /eos/user/a/atarabin/STXS_samples/220125/PROD_samplesNano_2023preBPix_MC_8d4c03f7
# /eos/user/a/atarabin/STXS_samples/220125/PROD_samplesNano_2023postBPix_MC_8d4c03f7

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT="${HZZ_ROOT}/DataPreprocessing/functions/filter_dataframe.py"

# Array of datasets to process
DATASETS=("ZZTo4l" "ggTo2e2mu_Contin_MCFM701" "ggTo4e_Contin_MCFM701" "ggTo4mu_Contin_MCFM701" "ggTo4tau_Contin_MCFM701" "ggTo2mu2tau_Contin_MCFM701" "ggTo2e2tau_Contin_MCFM701" "WZZ" "WWZ" "ZZZ" "ggH125" "VBFH125" "ZH125" "WplusH125" "WminusH125" "ttH125" "TTZZ" "TTWW")

run_campaign() {
    local CAMPAIGN_NAME="$1"
    local INPUT_BASE="$2"
    local OUTPUT_BASE="$3"
    local BTAG_THRESHOLD="$4"

    echo ""
    echo "========================================================"
    echo "Campaign: ${CAMPAIGN_NAME}"
    echo "Input base: ${INPUT_BASE}"
    echo "Output base: ${OUTPUT_BASE}"
    echo "btagThreshold: ${BTAG_THRESHOLD}"
    echo "========================================================"

    mkdir -p "${OUTPUT_BASE}"

    declare -A pids

    # Launch all dataset jobs for this campaign in parallel
    for dataset in "${DATASETS[@]}"; do
        local INPUT_FILE="${INPUT_BASE}/${dataset}/ZZ4lAnalysis.root"

        if [ ! -f "$INPUT_FILE" ]; then
            echo "WARNING: ${INPUT_FILE} not found, skipping..."
            continue
        fi

        echo "Launching filter job for: ${dataset}"

        python "$SCRIPT" \
            --file "$INPUT_FILE" \
            --outputPath "$OUTPUT_BASE" \
            --mode "$dataset" \
            --btagThreshold "$BTAG_THRESHOLD" \
            > "${OUTPUT_BASE}/log_${dataset}.txt" 2>&1 &

        pids[$!]=$dataset
    done

    echo ""
    echo "All jobs launched for ${CAMPAIGN_NAME}. Waiting for completion..."
    echo "========================================================"

    local total_datasets=${#pids[@]}
    local completed=0

    for pid in "${!pids[@]}"; do
        local dataset="${pids[$pid]}"
        wait "$pid"
        local status=$?

        completed=$((completed + 1))

        if [ "$status" -eq 0 ]; then
            local file_size
            file_size=$(ls -lh "${OUTPUT_BASE}/${dataset}.root" 2>/dev/null | awk '{print $5}')
            [ -z "$file_size" ] && file_size="unknown"
            echo "✓ ${dataset} completed successfully (${file_size})"
        else
            echo "✗ ${dataset} failed (exit code: ${status})"
        fi
        echo "  Progress: ${completed}/${total_datasets} datasets finished"
    done

    echo ""
    echo "Campaign ${CAMPAIGN_NAME} completed!"
    echo "Check logs in: ${OUTPUT_BASE}/log_*.txt"
    echo "Summary of generated files:"
    echo "========================================================"

    local total_size=0
    for dataset in "${DATASETS[@]}"; do
        if [ -f "${OUTPUT_BASE}/${dataset}.root" ]; then
            local file_size
            local file_bytes
            file_size=$(ls -lh "${OUTPUT_BASE}/${dataset}.root" | awk '{print $5}')
            file_bytes=$(ls -l "${OUTPUT_BASE}/${dataset}.root" | awk '{print $5}')
            total_size=$((total_size + file_bytes))
            echo "  ${dataset}: ${file_size}"
        else
            echo "  ${dataset}: MISSING"
        fi
    done

    echo "========================================================"
    local total_size_h
    total_size_h=$(numfmt --to=iec "$total_size" 2>/dev/null || echo "$((total_size / 1024 / 1024))MB")
    echo "Total size (${CAMPAIGN_NAME}): ${total_size_h}"
}

# Format: "campaign_name|input_base|output_base|btag_threshold"
CAMPAIGNS=(
    "2022postEE|/eos/user/a/atarabin/STXS_samples/PROD_samplesNano_2022EE_MC_8d4c03f7|/eos/user/z/zhiheng/STXS_samples/2022samples/postEE/root_data|0.2386"
    "2022preEE|/eos/user/a/atarabin/STXS_samples/PROD_samplesNano_2022_MC_8d4c03f7|/eos/user/z/zhiheng/STXS_samples/2022samples/preEE/root_data|0.2421"
    "2023preBPix|/eos/user/a/atarabin/STXS_samples/220125/PROD_samplesNano_2023preBPix_MC_8d4c03f7|/eos/user/z/zhiheng/STXS_samples/2023samples/preBPix/root_data|0.1917"
    "2023postBPix|/eos/user/a/atarabin/STXS_samples/220125/PROD_samplesNano_2023postBPix_MC_8d4c03f7|/eos/user/z/zhiheng/STXS_samples/2023samples/postEE/root_data|0.1919"
)

echo "Starting parallel filtering for 4 campaigns..."

for campaign in "${CAMPAIGNS[@]}"; do
    IFS='|' read -r CAMPAIGN_NAME INPUT_BASE OUTPUT_BASE BTAG_THRESHOLD <<< "$campaign"
    run_campaign "$CAMPAIGN_NAME" "$INPUT_BASE" "$OUTPUT_BASE" "$BTAG_THRESHOLD"
done

echo ""
echo "All campaigns completed."
