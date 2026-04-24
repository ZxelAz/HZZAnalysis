#!/usr/bin/env bash
set -euo pipefail

# Full pipeline:  make_datacard.py  →  text2workspace.py
#
# Usage:
#   bash run_datacard_pipeline.sh [METHOD]
#   METHOD ∈ {CutBased, BDT, DNN, GATO}   (default: BDT)
#
# Step 1 (make_datacard) runs under the venv/LCG Python (no CMSSW).
# Step 2 (text2workspace) runs under the CMSSW environment.

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
METHOD="${1:-BDT}"
TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${METHOD}"

# Non-CMSSW Python for step 1. Default to current LCG/system python;
# override with HZZ_PYTHON=/path/to/bin/python if a dedicated venv is needed.
HZZ_PYTHON="${HZZ_PYTHON:-python}"

# ---------------------------------------------------------------------------
# Step 1: make_datacard.py  (venv/LCG Python, no CMSSW)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Step 1: generating datacard for method '${METHOD}'"
echo "  trial dir: ${TRIAL_DIR}"
echo "============================================================"

EPS="${TRIAL_DIR}/epsilonA.pkl"
NBKG="${TRIAL_DIR}/N_bkg_category.pkl"
WS="${TRIAL_DIR}/workspace.root"
DATACARD="${TRIAL_DIR}/datacard.txt"

# STXS bins / categories filters; leave empty to use all.
PROCESSES_CSV=""
CATEGORIES_CSV=""

"${HZZ_PYTHON}" "${HZZ_ROOT}/InferenceModel/functions/make_datacard.py" \
    --epsilonA        "${EPS}" \
    --N_bkg_category  "${NBKG}" \
    --workspace       "${WS}" \
    --output          "${DATACARD}" \
    --processes-list  "${PROCESSES_CSV}" \
    --categories-list "${CATEGORIES_CSV}"

echo "Datacard written to: ${DATACARD}"

# ---------------------------------------------------------------------------
# Step 2: text2workspace.py  (CMSSW Python; clear venv/LCG overlays first)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Step 2: running text2workspace for method '${METHOD}'"
echo "============================================================"

# Remove venv / LCG overlays that conflict with CMSSW combine.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV ROOTSYS LD_PRELOAD

if [[ -n "${PATH:-}" ]]; then
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/cvmfs/sft.cern.ch/lcg/views/' | paste -sd: -)"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cvmfs/sft.cern.ch/lcg/views/' | paste -sd: -)"
    export LD_LIBRARY_PATH
fi

CMSSW_COMBINE_DIR="${HZZ_ROOT}/InferenceModel/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit"

if [[ ! -d "${CMSSW_COMBINE_DIR}" ]]; then
    echo "ERROR: CMSSW CombinedLimit not found at ${CMSSW_COMBINE_DIR}"
    exit 2
fi

cd "${CMSSW_COMBINE_DIR}"
eval "$(scramv1 runtime -sh)"
command -v combine >/dev/null || {
    echo "ERROR: combine not found after loading CMSSW runtime."
    exit 127
}

set -x

text2workspace.py "${DATACARD}" \
    -o "${TRIAL_DIR}/datacard_workspace.root" \
    -m 125.38 \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO 'map=.*/proc_GG2H_0J_PTH_0_10:r_proc_GG2H_0J_PTH_0_10[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_0J_PTH_GT10:r_proc_GG2H_0J_PTH_GT10[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_1J_PTH_0_60:r_proc_GG2H_1J_PTH_0_60[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_1J_PTH_120_200:r_proc_GG2H_1J_PTH_120_200[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_1J_PTH_60_120:r_proc_GG2H_1J_PTH_60_120[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_GE2J_MJJ_0_350_PTH_0_60:r_proc_GG2H_GE2J_MJJ_0_350_PTH_0_60[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_GE2J_MJJ_0_350_PTH_120_200:r_proc_GG2H_GE2J_MJJ_0_350_PTH_120_200[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_GE2J_MJJ_0_350_PTH_60_120:r_proc_GG2H_GE2J_MJJ_0_350_PTH_60_120[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_GE2J_MJJ_GT350:r_proc_GG2H_GE2J_MJJ_GT350[1,-100,100]' \
    --PO 'map=.*/proc_GG2H_PTH_GT200:r_proc_GG2H_PTH_GT200[1,-100,100]' \
    --PO 'map=.*/proc_TTH:r_proc_TTH[1,-100,100]' \
    --PO 'map=.*/proc_VBF_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25:r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25[1,-100,100]' \
    --PO 'map=.*/proc_VBF_GE2J_MJJ_60_120:r_proc_QQH_GE2J_MJJ_60_120[1,-100,100]' \
    --PO 'map=.*/proc_VBF_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25:r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25[1,-100,100]' \
    --PO 'map=.*/proc_VBF_GE2J_MJJ_GT350_PTH_GT200:r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200[1,-100,100]' \
    --PO 'map=.*/proc_VBF_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25:r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25[1,-100,100]' \
    --PO 'map=.*/proc_VBF_rest:r_proc_QQH_rest[1,-100,100]' \
    --PO 'map=.*/proc_WminusH_lep_PTV_0_150:r_proc_VH_lep_PTV_0_150[1,-100,100]' \
    --PO 'map=.*/proc_WminusH_lep_PTV_GT150:r_proc_VH_lep_PTV_GT150[1,-100,100]' \
    --PO 'map=.*/proc_WminushadH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25:r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25[1,-100,100]' \
    --PO 'map=.*/proc_WminushadH_GE2J_MJJ_60_120:r_proc_QQH_GE2J_MJJ_60_120[1,-100,100]' \
    --PO 'map=.*/proc_WminushadH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25:r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25[1,-100,100]' \
    --PO 'map=.*/proc_WminushadH_GE2J_MJJ_GT350_PTH_GT200:r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200[1,-100,100]' \
    --PO 'map=.*/proc_WminushadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25:r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25[1,-100,100]' \
    --PO 'map=.*/proc_WminushadH_rest:r_proc_QQH_rest[1,-100,100]' \
    --PO 'map=.*/proc_WplusH_lep_PTV_0_150:r_proc_VH_lep_PTV_0_150[1,-100,100]' \
    --PO 'map=.*/proc_WplusH_lep_PTV_GT150:r_proc_VH_lep_PTV_GT150[1,-100,100]' \
    --PO 'map=.*/proc_WplushadH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25:r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25[1,-100,100]' \
    --PO 'map=.*/proc_WplushadH_GE2J_MJJ_60_120:r_proc_QQH_GE2J_MJJ_60_120[1,-100,100]' \
    --PO 'map=.*/proc_WplushadH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25:r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25[1,-100,100]' \
    --PO 'map=.*/proc_WplushadH_GE2J_MJJ_GT350_PTH_GT200:r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200[1,-100,100]' \
    --PO 'map=.*/proc_WplushadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25:r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25[1,-100,100]' \
    --PO 'map=.*/proc_WplushadH_rest:r_proc_QQH_rest[1,-100,100]' \
    --PO 'map=.*/proc_ZH_lep_PTV_0_150:r_proc_VH_lep_PTV_0_150[1,-100,100]' \
    --PO 'map=.*/proc_ZH_lep_PTV_GT150:r_proc_VH_lep_PTV_GT150[1,-100,100]' \
    --PO 'map=.*/proc_ZhadH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25:r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25[1,-100,100]' \
    --PO 'map=.*/proc_ZhadH_GE2J_MJJ_60_120:r_proc_QQH_GE2J_MJJ_60_120[1,-100,100]' \
    --PO 'map=.*/proc_ZhadH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25:r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25[1,-100,100]' \
    --PO 'map=.*/proc_ZhadH_GE2J_MJJ_GT350_PTH_GT200:r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200[1,-100,100]' \
    --PO 'map=.*/proc_ZhadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25:r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25[1,-100,100]' \
    --PO 'map=.*/proc_ZhadH_rest:r_proc_QQH_rest[1,-100,100]'

set +x

echo "============================================================"
echo "Pipeline complete."
echo "  Datacard:  ${DATACARD}"
echo "  Workspace: ${TRIAL_DIR}/datacard_workspace.root"
echo "============================================================"
