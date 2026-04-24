#!/usr/bin/env bash
# Usage:
#   bash run_text2workspace.sh [METHOD]
#   METHOD ∈ {CutBased, BDT, DNN, GATO}  (default: GATO)
set -euo pipefail

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
METHOD="${1:-GATO}"

# setup CMSSW environment for combine
# Remove external python/LCG overlays that can conflict with CMSSW combine.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV ROOTSYS LD_PRELOAD

PROCESSES_CSV="\
GG2H_0J_PTH_0_10,\
GG2H_0J_PTH_GT10,\
GG2H_1J_PTH_0_60,\
GG2H_1J_PTH_120_200,\
GG2H_1J_PTH_60_120,\
GG2H_GE2J_MJJ_0_350_PTH_0_60,\
GG2H_GE2J_MJJ_0_350_PTH_120_200,\
GG2H_GE2J_MJJ_0_350_PTH_60_120,\
GG2H_GE2J_MJJ_GT350,\
GG2H_PTH_GT200,\
TTH,\
VBF_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25,\
VBF_GE2J_MJJ_60_120,\
VBF_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25,\
VBF_GE2J_MJJ_GT350_PTH_GT200,\
VBF_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25,\
VBF_rest,\
WminusH_lep_PTV_0_150,\
WminusH_lep_PTV_GT150,\
WminushadH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25,\
WminushadH_GE2J_MJJ_60_120,\
WminushadH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25,\
WminushadH_GE2J_MJJ_GT350_PTH_GT200,\
WminushadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25,\
WminushadH_rest,\
WplusH_lep_PTV_0_150,\
WplusH_lep_PTV_GT150,\
WplushadH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25,\
WplushadH_GE2J_MJJ_60_120,\
WplushadH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25,\
WplushadH_GE2J_MJJ_GT350_PTH_GT200,\
WplushadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25,\
WplushadH_rest,\
ZH_lep_PTV_0_150,\
ZH_lep_PTV_GT150,\
ZhadH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25,\
ZhadH_GE2J_MJJ_60_120,\
ZhadH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25,\
ZhadH_GE2J_MJJ_GT350_PTH_GT200,\
ZhadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25,\
ZhadH_rest"

if [[ -n "${PATH:-}" ]]; then
	PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/cvmfs/sft.cern.ch/lcg/views/' | paste -sd: -)"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
	LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cvmfs/sft.cern.ch/lcg/views/' | paste -sd: -)"
	export LD_LIBRARY_PATH
fi

cd ${HZZ_ROOT}/InferenceModel/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit
# scramv1 b clean; scramv1 b -j$(nproc --ignore=2) # always make a clean build, with n - 2 cores on the system

# Load CMSSW runtime so `combine` is available in this shell.
eval "$(scramv1 runtime -sh)"
command -v combine >/dev/null || {
	echo "ERROR: combine not found after loading CMSSW runtime."
	echo "Check that HiggsAnalysis/CombinedLimit is built successfully."
	exit 127
}


TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${METHOD}"
DATACARD="${TRIAL_DIR}/datacard.txt"

IFS=',' read -r -a PROCESSES <<< "${PROCESSES_CSV}"

PO_ARGS=()

for proc in "${PROCESSES[@]}"; do
	proc="${proc//[[:space:]]/}"
	[[ -z "${proc}" ]] && continue
	PO_ARGS+=(--PO "map=.*/proc_${proc}:r_proc_${proc}[1,-10,20]")
done
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
