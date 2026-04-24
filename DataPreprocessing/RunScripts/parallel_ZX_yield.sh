#!/bin/bash

# Parallel Z+X yield calculator script for multiple datasets
# Usage: bash parallel_ZX_yield.sh

# Base paths
INPUT_BASE="/eos/user/a/atarabin/STXS_samples/PROD_samplesNano_2022EE_MC_8d4c03f7"
OUTPUT_BASE="/eos/user/z/zhiheng/STXS_samples/2022samples/postEE/ZXdata"
: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT="${HZZ_ROOT}/DataPreprocessing/functions/ZX_yield_calculator.py"

# Analysis parameters
YEAR="2022EE"
YEAR_MC="2022EE"

# Array of datasets to process
DATASETS=("ZZTo4l" "ggTo2e2mu_Contin_MCFM701" "ggTo4e_Contin_MCFM701" "ggTo4mu_Contin_MCFM701" "ggTo4tau_Contin_MCFM701" "ggTo2mu2tau_Contin_MCFM701" "ggTo2e2tau_Contin_MCFM701" "WZZ" "WWZ" "ZZZ" "ggH125" "VBFH125" "ZH125" "WplusH125" "WminusH125" "ttH125" "TTZZ" "TTWW")

echo "Starting parallel Z+X yield calculation for datasets..."
echo "==========================================="
echo "Year: ${YEAR}"
echo "Year MC: ${YEAR_MC}"
echo "Input base: ${INPUT_BASE}"
echo "Output base: ${OUTPUT_BASE}"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# Associative array to store PIDs
declare -A pids

# Launch all jobs in parallel
for dataset in "${DATASETS[@]}"; do
    INPUT_FILE="${INPUT_BASE}/${dataset}/ZZ4lAnalysis.root"
    OUTPUT_FILE="${OUTPUT_BASE}/${dataset}_ZX.root"
    
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "WARNING: ${INPUT_FILE} not found, skipping..."
        continue
    fi
    
    echo "Launching Z+X yield calculation for: ${dataset}"
    
    # Run in background with output redirection
    python "$SCRIPT" \
        --year "$YEAR" \
        --year_mc "$YEAR_MC" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        > "${OUTPUT_BASE}/log_${dataset}_ZX.txt" 2>&1 &
    
    # Store the process ID
    pids[$!]=$dataset
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo "==========================================="

# Wait for all background jobs and report status
total_datasets=${#DATASETS[@]}
completed=0

for pid in "${!pids[@]}"; do
    dataset="${pids[$pid]}"
    wait $pid
    status=$?
    
    completed=$((completed + 1))
    
    if [ $status -eq 0 ]; then
        file_size=$(ls -lh "${OUTPUT_BASE}/${dataset}_ZX.root" 2>/dev/null | awk '{print $5}')
        echo "✓ ${dataset} completed successfully (${file_size})"
        echo "  Progress: ${completed}/${total_datasets} datasets finished"
    else
        echo "✗ ${dataset} failed (exit code: ${status})"
        echo "  Check log: ${OUTPUT_BASE}/log_${dataset}_ZX.txt"
        echo "  Progress: ${completed}/${total_datasets} datasets finished"
    fi
    
    # Show remaining datasets
    remaining=$((total_datasets - completed))
    if [ $remaining -gt 0 ]; then
        echo "  Remaining datasets (${remaining}): "
        for i in $(seq 0 $((${#DATASETS[@]} - 1))); do
            dataset_name="${DATASETS[$i]}"
            if ! ls "${OUTPUT_BASE}/${dataset_name}_ZX.root" &>/dev/null; then
                echo "    - ${dataset_name}"
            fi
        done
        echo ""
    fi
done

echo ""
echo "All Z+X yield calculations completed!"
echo "Check logs in: ${OUTPUT_BASE}/log_*_ZX.txt"
echo ""
echo "Summary of generated files:"
echo "==========================================="
total_size=0
success_count=0
for dataset in "${DATASETS[@]}"; do
    if [ -f "${OUTPUT_BASE}/${dataset}_ZX.root" ]; then
        file_size=$(ls -lh "${OUTPUT_BASE}/${dataset}_ZX.root" | awk '{print $5}')
        file_bytes=$(ls -l "${OUTPUT_BASE}/${dataset}_ZX.root" | awk '{print $5}')
        total_size=$((total_size + file_bytes))
        success_count=$((success_count + 1))
        echo "  ${dataset}_ZX.root: ${file_size}"
    else
        echo "  ${dataset}_ZX.root: MISSING"
    fi
done
echo "==========================================="
total_size_h=$(numfmt --to=iec $total_size 2>/dev/null || echo "$(($total_size / 1024 / 1024))MB")
echo "Total size: ${total_size_h}"
echo "Success rate: ${success_count}/${#DATASETS[@]} datasets"
