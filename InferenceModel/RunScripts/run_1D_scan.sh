#!/usr/bin/env bash
set -euo pipefail

# Runs a 1D likelihood scan (--algo grid) for each POI independently,
# floating all other POIs. Produces one ROOT file per POI in TRIAL_DIR.
#
# Usage:
#   bash run_1D_scan.sh [METHOD]
#   METHOD ∈ {CutBased, BDT, DNN, GATO}  (default: GATO)
#
#   POINTS=400 bash run_1D_scan.sh BDT          # more points
#   POI_RANGE=0.5,2 bash run_1D_scan.sh BDT     # override default range
#   BATCH_SIZE=8 bash run_1D_scan.sh BDT        # 8 POIs in parallel per batch
#
# Per-POI ranges are configured in POI_RANGES_MAP below.
# Any POI not listed there falls back to POI_RANGE (default: -5,10).

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV ROOTSYS LD_PRELOAD

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
METHOD="${1:-GATO}"

# Condor may stage executable into /pool/condor/...; ensure CMSSW is reachable.
if [[ ! -d "${HZZ_ROOT}/InferenceModel/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit" ]]; then
	echo "ERROR: Could not locate CMSSW area under HZZ_ROOT='${HZZ_ROOT}'."
	exit 2
fi

CMSSW_BASE_DIR="${HZZ_ROOT}/InferenceModel/CMSSW_14_1_0_pre4"
CMSSW_COMBINE_DIR="${CMSSW_BASE_DIR}/src/HiggsAnalysis/CombinedLimit"

if [[ -n "${PATH:-}" ]]; then
	PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/cvmfs/sft.cern.ch/lcg/views/' | paste -sd: -)"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
	LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/cvmfs/sft.cern.ch/lcg/views/' | paste -sd: -)"
	export LD_LIBRARY_PATH
fi

# Ensure CMS tools are available in non-interactive batch shells.
if ! command -v scramv1 >/dev/null 2>&1; then
	if [[ -f /cvmfs/cms.cern.ch/cmsset_default.sh ]]; then
		# shellcheck source=/dev/null
		source /cvmfs/cms.cern.ch/cmsset_default.sh
	fi
fi

command -v scramv1 >/dev/null || {
	echo "ERROR: scramv1 not found. Source CMS environment first."
	exit 127
}

cd "${CMSSW_COMBINE_DIR}"
eval "$(scramv1 runtime -sh)"
command -v combine >/dev/null || {
	echo "ERROR: combine not found after loading CMSSW runtime."
	exit 127
}
command -v python3 >/dev/null || {
	echo "ERROR: python3 not found after loading CMSSW runtime."
	exit 127
}

TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${METHOD}"
COMBINE_OUT_NAME="${COMBINE_OUT_NAME:-combine_output}"
COMBINE_OUT_DIR="${TRIAL_DIR}/${COMBINE_OUT_NAME}"
DATACARD="${TRIAL_DIR}/datacard_workspace.root"
POINTS="${POINTS:-800}"
POI_RANGE="${POI_RANGE:--5,10}"
MINIMIZER_STRATEGIES="${MINIMIZER_STRATEGIES:-0 1 2}"

# ---------------------------------------------------------------------------
# Per-POI range overrides.  Format:  [r_proc_NAME]="lo,hi"
# Any POI not listed here uses the global POI_RANGE default above.
# ---------------------------------------------------------------------------
declare -A POI_RANGES_MAP
# Tight range: high-yield ggH 0j/1j bins (~10–50 events)
POI_RANGES_MAP[r_proc_GG2H_0J_PTH_0_10]="-0.5,2.5"
POI_RANGES_MAP[r_proc_GG2H_0J_PTH_GT10]="0,2"
POI_RANGES_MAP[r_proc_GG2H_1J_PTH_0_60]="-2,3"
POI_RANGES_MAP[r_proc_GG2H_1J_PTH_60_120]="-2,4"
POI_RANGES_MAP[r_proc_GG2H_1J_PTH_120_200]="-2,4"
# Medium range: ggH 2j and high-pT bins, TTH (~1–10 events)
POI_RANGES_MAP[r_proc_GG2H_GE2J_MJJ_0_350_PTH_0_60]="-8,15"
POI_RANGES_MAP[r_proc_GG2H_GE2J_MJJ_0_350_PTH_60_120]="-5,10"
POI_RANGES_MAP[r_proc_GG2H_GE2J_MJJ_0_350_PTH_120_200]="-5,10"
POI_RANGES_MAP[r_proc_GG2H_GE2J_MJJ_GT350]="-8,10"
POI_RANGES_MAP[r_proc_GG2H_PTH_GT200]="-5,10"
POI_RANGES_MAP[r_proc_TTH]="-5,10"
# Merged QQH bins (VBF + hadronic VH combined)
POI_RANGES_MAP[r_proc_QQH_GE2J_MJJ_60_120]="-5,10"
POI_RANGES_MAP[r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25]="-8,13"
POI_RANGES_MAP[r_proc_QQH_rest]="-8,9"
POI_RANGES_MAP[r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25]="-10,20"
POI_RANGES_MAP[r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200]="-5,13"
POI_RANGES_MAP[r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25]="-8,15"
POI_RANGES_MAP[r_proc_VH_lep_PTV_0_150]="-3,5"
POI_RANGES_MAP[r_proc_VH_lep_PTV_GT150]="-5,15"

POIS=(
	r_proc_GG2H_0J_PTH_0_10
	r_proc_GG2H_0J_PTH_GT10
	r_proc_GG2H_1J_PTH_0_60
	r_proc_GG2H_1J_PTH_120_200
	r_proc_GG2H_1J_PTH_60_120
	r_proc_GG2H_GE2J_MJJ_0_350_PTH_0_60
	r_proc_GG2H_GE2J_MJJ_0_350_PTH_120_200
	r_proc_GG2H_GE2J_MJJ_0_350_PTH_60_120
	r_proc_GG2H_GE2J_MJJ_GT350
	r_proc_GG2H_PTH_GT200
	r_proc_TTH
	r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25
	r_proc_QQH_GE2J_MJJ_60_120
	r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25
	r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200
	r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25
	r_proc_QQH_rest
	r_proc_VH_lep_PTV_0_150
	r_proc_VH_lep_PTV_GT150
)

# Build --setParameters string (all POIs to 1).
POI_INIT="$(python3 - <<'PY' "${DATACARD}"
import sys, ROOT
f = ROOT.TFile.Open(sys.argv[1])
w = f.Get("w")
mc = w.obj("ModelConfig")
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

BATCH_SIZE="${BATCH_SIZE:-5}"

mkdir -p "${COMBINE_OUT_DIR}"
cd "${COMBINE_OUT_DIR}"

# Helper: run a single POI scan in the background, log to file.
run_one() {
	local POI="$1"
	local SHORT="${POI#r_proc_}"
	local NAME="scan1D.${SHORT}"
	local LOG="${COMBINE_OUT_DIR}/combine_${NAME}.log"
	local OUT_ROOT="${COMBINE_OUT_DIR}/higgsCombine.${NAME}.MultiDimFit.mH125.38.root"

	# Resolve per-POI range: deserialize exported map, fall back to global.
	local this_range="${POI_RANGE}"
	while IFS='=' read -r k v; do
		[[ "${k}" == "${POI}" ]] && this_range="${v}" && break
	done <<< "${POI_RANGES_MAP_VALS}"

	echo "[batch] Starting: ${POI}  range=${this_range}  strategies='${MINIMIZER_STRATEGIES}'" >&2
	: > "${LOG}"
	rm -f "${OUT_ROOT}"

	local status=1
	local strategy
	for strategy in ${MINIMIZER_STRATEGIES}; do
		local strategy_opts=(--cminDefaultMinimizerStrategy "${strategy}")
		case "${strategy}" in
			0)
				strategy_opts+=(--cminFallbackAlgo Minuit2,Simplex,0:0.1)
				;;
			1)
				strategy_opts+=(--cminFallbackAlgo Minuit2,Simplex,1:0.1)
				;;
			2)
				strategy_opts+=(--cminFallbackAlgo Minuit2,Simplex,2:0.1)
				;;
		esac

		echo "[batch] Trying ${POI} with minimizer strategy=${strategy}" >> "${LOG}"
		if combine \
			-M MultiDimFit \
			"${DATACARD}" \
			-t -1 \
			--setParameters "${POI_INIT}" \
			-n ".${NAME}" \
			-m 125.38 \
			--algo grid \
			--points "${POINTS}" \
			-P "${POI}" \
			--setParameterRanges "${POI}=${this_range}" \
			--floatOtherPOIs 1 \
			--robustFit 1 \
			--cminDefaultMinimizerStrategy 0 \
			>>"${LOG}" 2>&1; then
			if [[ -s "${OUT_ROOT}" ]]; then
				status=0
				break
			fi
		fi
		echo "[batch] Strategy ${strategy} failed for ${POI}, retrying..." >> "${LOG}"
	done

	if [[ ${status} -eq 0 ]]; then
		echo "[batch] Done:    ${POI}" >&2
	else
		echo "[batch] FAILED:  ${POI} (exit ${status})" >&2
	fi
	return ${status}
}
export -f run_one
# Serialize the associative array for export to subshells.
POI_RANGES_MAP_KEYS="$(printf '%s\n' "${!POI_RANGES_MAP[@]}")"
POI_RANGES_MAP_VALS="$(for k in "${!POI_RANGES_MAP[@]}"; do printf '%s=%s\n' "$k" "${POI_RANGES_MAP[$k]}"; done)"
export DATACARD POI_INIT POINTS POI_RANGE COMBINE_OUT_DIR POI_RANGES_MAP_KEYS POI_RANGES_MAP_VALS

# Split POIS into batches of BATCH_SIZE and run each batch in parallel.
FAILED=()
total=${#POIS[@]}
i=0
batch_num=0

while [[ $i -lt $total ]]; do
	batch_num=$(( batch_num + 1 ))
	batch=( "${POIS[@]:$i:$BATCH_SIZE}" )
	echo "============================================================"
	echo "Batch ${batch_num}: running ${#batch[@]} POIs in parallel"
	for p in "${batch[@]}"; do echo "  ${p}"; done
	echo "============================================================"

	pids=()
	for POI in "${batch[@]}"; do
		run_one "${POI}" &
		pids+=($!)
	done

	# Wait for all jobs in this batch.
	for pid in "${pids[@]}"; do
		if ! wait "${pid}"; then
			FAILED+=("pid=${pid}")
		fi
	done

	i=$(( i + BATCH_SIZE ))
done

echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
	echo "All 1D scans done (${total} POIs, batch size ${BATCH_SIZE})."
	echo "Output directory: ${COMBINE_OUT_DIR}"
else
	echo "Completed with ${#FAILED[@]} failure(s). Check logs in: ${COMBINE_OUT_DIR}"
fi

# ---------------------------------------------------------------------------
# Generate combined plot automatically.
# Serialize POI_RANGES_MAP as a JSON object and pass to the plotter so the
# x-axis ranges match exactly what was used for the scans.
# ---------------------------------------------------------------------------
POI_RANGES_JSON="{"
first=1
for k in "${!POI_RANGES_MAP[@]}"; do
	lo="${POI_RANGES_MAP[$k]%,*}"
	hi="${POI_RANGES_MAP[$k]#*,}"
	[[ ${first} -eq 0 ]] && POI_RANGES_JSON+=","
	POI_RANGES_JSON+="\"${k}\":[${lo},${hi}]"
	first=0
done
POI_RANGES_JSON+="}"

PLOT_SCRIPT="${HZZ_ROOT}/InferenceModel/functions/plot_all_1D_scans.py"
PLOT_OUTPUT="${TRIAL_DIR}/${COMBINE_OUT_NAME}/all_1D_scans.png"

echo "Generating plot: ${PLOT_OUTPUT}"
python3 "${PLOT_SCRIPT}" \
	--trial-dir      "${TRIAL_DIR}" \
	--combine-out-dir "${COMBINE_OUT_NAME}" \
	--output         "${PLOT_OUTPUT}" \
	--poi-ranges-json "${POI_RANGES_JSON}" \
	&& echo "Plot saved: ${PLOT_OUTPUT}" \
	|| echo "WARNING: plot generation failed (scans still available in ${COMBINE_OUT_DIR})"
echo "============================================================"
