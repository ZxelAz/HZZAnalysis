#!/bin/bash
# Thin wrapper for root_to_numpy.py; runs over all four campaigns.

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT="${HZZ_ROOT}/DataPreprocessing/functions/root_to_numpy.py"

CAMPAIGNS=(
    "2022preEE|/eos/user/z/zhiheng/STXS_samples/2022samples/preEE/root_data|/eos/user/z/zhiheng/STXS_samples/2022samples/preEE/numpy_data"
    "2022postEE|/eos/user/z/zhiheng/STXS_samples/2022samples/postEE/root_data|/eos/user/z/zhiheng/STXS_samples/2022samples/postEE/numpy_data"
    "2023preBPix|/eos/user/z/zhiheng/STXS_samples/2023samples/preBPix/root_data|/eos/user/z/zhiheng/STXS_samples/2023samples/preBPix/numpy_data"
    "2023postBPix|/eos/user/z/zhiheng/STXS_samples/2023samples/postBPix/root_data|/eos/user/z/zhiheng/STXS_samples/2023samples/postBPix/numpy_data"
)

for campaign in "${CAMPAIGNS[@]}"; do
    IFS='|' read -r NAME INDIR OUTDIR <<< "$campaign"
    echo "== ${NAME}: ${INDIR} -> ${OUTDIR}"
    python "${SCRIPT}" \
        --file_path "${INDIR}" \
        --output_dir "${OUTDIR}" \
        --tree_name Events
done
