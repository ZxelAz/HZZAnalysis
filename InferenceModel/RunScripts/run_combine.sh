#!/usr/bin/env bash
# Usage:
#   bash run_combine.sh [METHOD]
#   METHOD ∈ {CutBased, BDT, DNN, GATO}  (default: GATO)
set -euo pipefail

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
METHOD="${1:-GATO}"

# setup CMSSW environment for combine
# Remove external python/LCG overlays that can conflict with CMSSW combine.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV ROOTSYS LD_PRELOAD

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
DATACARD="${TRIAL_DIR}/datacard_workspace.root"
METHOD="MultiDimFit" # or "FitDiagnostics" for full fit output

# Tunables for stability on shared/batch nodes.
# Override at runtime, e.g.:
#   POINTS=200 FLOAT_OTHER_POIS=1 SAVE_INACTIVE_POI=0 bash run_combine.sh
POINTS="${POINTS:-200}"
FLOAT_OTHER_POIS="${FLOAT_OTHER_POIS:-1}"
SAVE_INACTIVE_POI="${SAVE_INACTIVE_POI:-1}"

# Build --setParameters string with all POIs initialized to 1.
POI_INIT="$(python3 - <<'PY' "${DATACARD}"
import sys
import ROOT

root_path = sys.argv[1]
f = ROOT.TFile.Open(root_path)
if not f or f.IsZombie():
	raise RuntimeError(f"Cannot open workspace file: {root_path}")

w = f.Get("w")
if w is None:
	# Fallback: pick first RooWorkspace in file.
	for key in f.GetListOfKeys():
		obj = key.ReadObj()
		if obj and obj.InheritsFrom("RooWorkspace"):
			w = obj
			break
if w is None:
	raise RuntimeError("No RooWorkspace found in file")

mc = w.obj("ModelConfig")
if mc is None:
	raise RuntimeError("ModelConfig not found in workspace")

pois = mc.GetParametersOfInterest()
it = pois.createIterator()
parts = []
v = it.Next()
while v:
	parts.append(f"{v.GetName()}=1")
	v = it.Next()

print(",".join(parts))
PY
)"

if [[ -z "${POI_INIT}" ]]; then
	echo "ERROR: No POIs found in ModelConfig; cannot build --setParameters."
	exit 1
fi

echo "============================================================"
echo "Running combine with method: ${METHOD}"
echo "Datacard: ${DATACARD}"
echo "Initial POIs: ${POI_INIT}"
echo "Grid points: ${POINTS} | floatOtherPOIs: ${FLOAT_OTHER_POIS} | saveInactivePOI: ${SAVE_INACTIVE_POI}"
echo "============================================================"
cd "${TRIAL_DIR}"
OUTPUT_ROOT="${TRIAL_DIR}/higgsCombine.fitresult.${METHOD}.mH125.38.root"
set -x
combine \
	-M "${METHOD}" \
	"${DATACARD}" \
	-t -1 \
	--setParameters "${POI_INIT}" \
	-n .fitresult \
	-m 125.38 \
	--algo grid \
	--points "${POINTS}" \
	-P r_proc_GG2H_0J_PTH_0_10 \
	-P r_proc_GG2H_0J_PTH_GT10 \
	--setParameterRanges r_proc_GG2H_0J_PTH_0_10=0,2:r_proc_GG2H_0J_PTH_GT10=0,2\
	--floatOtherPOIs "${FLOAT_OTHER_POIS}" \
	--saveInactivePOI "${SAVE_INACTIVE_POI}" \
	--cminDefaultMinimizerStrategy 0 --robustFit \
	2>&1 | tee "${TRIAL_DIR}/combine_${METHOD}.log"
COMBINE_STATUS=${PIPESTATUS[0]}
set +x

if [[ ${COMBINE_STATUS} -eq 137 ]]; then
	echo "ERROR: combine was killed with exit 137 (likely out-of-memory)."
	echo "Try fewer points (e.g. POINTS=60) and/or FLOAT_OTHER_POIS=0."
	exit 137
elif [[ ${COMBINE_STATUS} -ne 0 ]]; then
	echo "ERROR: combine failed with exit code ${COMBINE_STATUS}."
	exit ${COMBINE_STATUS}
fi
# combine writes to cwd; move to TRIAL_DIR if not already there
if [[ -f "higgsCombine.fitresult.${METHOD}.mH125.38.root" && "$(pwd)" != "${TRIAL_DIR}" ]]; then
    mv "higgsCombine.fitresult.${METHOD}.mH125.38.root" "${OUTPUT_ROOT}"
fi
if [[ -f "${OUTPUT_ROOT}" ]]; then
    echo "Output ROOT file: ${OUTPUT_ROOT}"
else
    echo "WARNING: Output ROOT file not found! Expected: ${OUTPUT_ROOT}"
fi
echo "Output log: ${TRIAL_DIR}/combine_${METHOD}.log"
echo "Fit summary (POI lines):"
grep -E "Best fit|r_proc_" "${TRIAL_DIR}/combine_${METHOD}.log" || true
# --redefineSignalPOIs "${POIS}"
# -P r_proc_GG2H_0J_PTH_GT10 \
# -P r_proc_GG2H_1J_PTH_0_60 \
# -P r_proc_GG2H_1J_PTH_120_200 \
# -P r_proc_GG2H_1J_PTH_60_120 \
# -P r_proc_GG2H_GE2J_MJJ_0_350_PTH_0_60 \
# -P r_proc_GG2H_GE2J_MJJ_0_350_PTH_120_200 \
# -P r_proc_GG2H_GE2J_MJJ_0_350_PTH_60_120 \
# -P r_proc_GG2H_GE2J_MJJ_GT350 \
# -P r_proc_GG2H_PTH_GT200 \