#!/bin/bash

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT="${HZZ_ROOT}/DataPreprocessing/functions/plotter.py"
PLOTS_DIR="${HZZ_ROOT}/DataPreprocessing/Plots"

input_path="/eos/user/z/zhiheng/STXS_samples/2023samples/preBPix/numpy_data"
# plot_variables options: "All", "Leptons", "WeightEffect", or an explicit list of feature names
plot_variables=("WeightEffect")

if [[ "${plot_variables[0]}" == "WeightEffect" ]]; then
    output_path="${PLOTS_DIR}/weight_effect"
elif [[ "${plot_variables[0]}" == "Leptons" ]]; then
    output_path="${PLOTS_DIR}/leptons"
else
    output_path="${PLOTS_DIR}/signals"
fi

python "${SCRIPT}" \
    --numpy_path ${input_path} \
    --output_dir ${output_path} \
    --plot_variables ${plot_variables[@]}
