#!/bin/bash
# Post-hoc BDT trial analysis (validation-score histograms etc.).

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
TRIAL="${TRIAL:-trial51}"
TRIAL_PATH="${HZZ_ROOT}/Categorisation/BDT/Models/${TRIAL}"
DATA_DIR="/eos/user/z/zhiheng/STXS_samples"
OUTPUT_DIR="${TRIAL_PATH}/plots"

mkdir -p "${OUTPUT_DIR}"

if [ -z "${LCG_VERSION:-}" ]; then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
fi
export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH}"

DATA_FILES=(
    "$DATA_DIR"/2023samples/preBPix/root_data/{ggH125,VBFH125,ZH125}.root
    "$DATA_DIR"/2023samples/preBPix/root_data/{WplusH125,WminusH125,ttH125}.root
    "$DATA_DIR"/2023samples/postBPix/root_data/{ggH125,VBFH125,ZH125}.root
    "$DATA_DIR"/2023samples/postBPix/root_data/{WplusH125,WminusH125,ttH125}.root
    "$DATA_DIR"/2023samples/preBPix/root_data/ZZTo4l.root
    "$DATA_DIR"/2023samples/postBPix/root_data/ZZTo4l.root
    "$DATA_DIR"/2022samples/preEE/root_data/{ggH125,VBFH125,ZH125}.root
    "$DATA_DIR"/2022samples/preEE/root_data/{WplusH125,WminusH125,ttH125}.root
    "$DATA_DIR"/2022samples/postEE/root_data/{ggH125,VBFH125,ZH125}.root
    "$DATA_DIR"/2022samples/postEE/root_data/{WplusH125,WminusH125,ttH125}.root
    "$DATA_DIR"/2022samples/preEE/root_data/ZZTo4l.root
    "$DATA_DIR"/2022samples/postEE/root_data/ZZTo4l.root
)

echo "Starting model analysis..."
echo "Trial:  ${TRIAL_PATH}"
echo "Output: ${OUTPUT_DIR}"

python -u -m Categorisation.BDT.functions.analyze_model \
    --trial-path "${TRIAL_PATH}" \
    --data "${DATA_FILES[@]}" \
    --tree-name Events \
    --target HTXS_stage1_2_cat_pTjet30GeV \
    --use-event-weights \
    --class-name-dict merged \
    --output "${OUTPUT_DIR}/bdt_score.pdf"

echo "Analysis complete. Plots saved to: ${OUTPUT_DIR}"
