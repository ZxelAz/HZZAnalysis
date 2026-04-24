#!/bin/bash
# Thin wrapper for split_year.py; splits 2022 and 2023 data ROOT files by run number.

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT="${HZZ_ROOT}/DataPreprocessing/functions/split_year.py"
BASE="/eos/user/z/zhiheng/STXS_samples"

python "${SCRIPT}" --year 2022 --input_file "${BASE}/2022Data/2022.root"           --output_path "${BASE}/2022Data"
python "${SCRIPT}" --year 2023 --input_file "${BASE}/2023Data/2023.root"           --output_path "${BASE}/2023Data"
