#!/usr/bin/env bash
# run_pipeline.sh — end-to-end H→ZZ→4l STXS analysis pipeline.
#
# Stages (in order):
#   filter        DataPreprocessing: filter ROOT → filtered ROOT + numpy arrays + ZX yield
#   categorise    Apply categorisation method (CutBased | BDT | DNN | GATO)
#   inference     Response matrix → datacard → workspace (text2workspace)
#   scan          1D likelihood scans over all 19 POIs
#   plots         Summary scan plots
#
# Usage:
#   bash run_pipeline.sh [OPTIONS]
#
# Options:
#   -m, --method   METHOD     CutBased | BDT | DNN | GATO  (default: GATO)
#   -f, --from     STAGE      Start from stage: filter | categorise | inference | scan | plots
#                             (default: filter)
#   -s, --skip-train          Skip model training; use existing artefact in Models/
#   -n, --dry-run             Print what would run without executing
#   -h, --help

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
METHOD="GATO"
FROM_STAGE="filter"
SKIP_TRAIN=0
DRY_RUN=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--method)    METHOD="$2";     shift 2 ;;
        -f|--from)      FROM_STAGE="$2"; shift 2 ;;
        -s|--skip-train) SKIP_TRAIN=1;  shift   ;;
        -n|--dry-run)   DRY_RUN=1;      shift   ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "${METHOD}" in
    CutBased|BDT|DNN|GATO) ;;
    *) echo "ERROR: Unknown method '${METHOD}'. Choose: CutBased BDT DNN GATO"; exit 1 ;;
esac

case "${FROM_STAGE}" in
    filter|categorise|inference|scan|plots) ;;
    *) echo "ERROR: Unknown stage '${FROM_STAGE}'. Choose: filter categorise inference scan plots"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Resolve HZZ_ROOT to this script's directory
# ---------------------------------------------------------------------------
_SELF="${BASH_SOURCE[0]:-$0}"
export HZZ_ROOT="$(cd "$(dirname "${_SELF}")" && pwd)"

SCRIPTS_DP="${HZZ_ROOT}/DataPreprocessing/RunScripts"
SCRIPTS_BDT="${HZZ_ROOT}/Categorisation/BDT/RunScripts"
SCRIPTS_DNN="${HZZ_ROOT}/Categorisation/DNN/RunScripts"
SCRIPTS_GATO="${HZZ_ROOT}/Categorisation/GATO/RunScripts"
SCRIPTS_CB="${HZZ_ROOT}/Categorisation/CutBased/RunScripts"
SCRIPTS_IM="${HZZ_ROOT}/InferenceModel/RunScripts"

LOG_DIR="${HZZ_ROOT}/logs"
mkdir -p "${LOG_DIR}"
PIPELINE_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S)_${METHOD}.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
STAGE_ORDER=(filter categorise inference scan plots)

stage_index() {
    local s="$1" i=0
    for stage in "${STAGE_ORDER[@]}"; do
        [[ "${stage}" == "${s}" ]] && echo ${i} && return
        (( i++ ))
    done
    echo 99
}

FROM_IDX=$(stage_index "${FROM_STAGE}")

should_run() {
    local idx
    idx=$(stage_index "$1")
    [[ ${idx} -ge ${FROM_IDX} ]]
}

ts() { date '+%H:%M:%S'; }

log() { echo "[$(ts)] $*" | tee -a "${PIPELINE_LOG}"; }

run_step() {
    local label="$1"; shift
    log ">> ${label}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "   DRY-RUN: $*"
    else
        "$@" 2>&1 | tee -a "${PIPELINE_LOG}"
        log "   ${label} — done"
    fi
}

check_file() {
    local f="$1" label="$2"
    if [[ ! -f "${f}" ]]; then
        log "ERROR: Expected file missing after ${label}: ${f}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Print banner
# ---------------------------------------------------------------------------
log "========================================================"
log "  HZZ STXS pipeline"
log "  Method:      ${METHOD}"
log "  From stage:  ${FROM_STAGE}"
log "  Skip train:  ${SKIP_TRAIN}"
log "  Dry run:     ${DRY_RUN}"
log "  HZZ_ROOT:    ${HZZ_ROOT}"
log "  Log:         ${PIPELINE_LOG}"
log "========================================================"

# ---------------------------------------------------------------------------
# STAGE 1: filter
# Filter NanoAOD ROOT files, convert to numpy, compute ZX yield.
# ---------------------------------------------------------------------------
if should_run filter; then
    log "-------- STAGE 1: DataPreprocessing / filter --------"

    # Source LCG view for ROOT + Python
    set +u
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
    set -u
    export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH:-}"

    run_step "Filter MC samples (4 campaigns)"    bash "${SCRIPTS_DP}/parallel_filter.sh"
    run_step "Filter collision data"              bash "${SCRIPTS_DP}/parallel_filter_data.sh"
    run_step "Filter ZX control region"          bash "${SCRIPTS_DP}/parallel_filter_ZXCR.sh"
    run_step "Split 2022/2023 data by run number" bash "${SCRIPTS_DP}/split_year.sh"

    # numpy conversion is needed for ML-based categorisation methods
    if [[ "${METHOD}" != "CutBased" ]]; then
        run_step "Convert filtered ROOT → numpy"  bash "${SCRIPTS_DP}/root_to_numpy.sh"
    fi

    run_step "ZX yield estimate"                  bash "${SCRIPTS_DP}/run_ZX_yield_calculator.sh"

    log "-------- STAGE 1 complete --------"
fi

# ---------------------------------------------------------------------------
# STAGE 2: categorise
# Assign each event to an analysis category; writes a new branch to ROOT.
# ---------------------------------------------------------------------------
if should_run categorise; then
    log "-------- STAGE 2: Categorisation (${METHOD}) --------"

    set +u
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
    set -u
    export PYTHONPATH="${HZZ_ROOT}:${HZZ_ROOT}/Categorisation:${HZZ_ROOT}/Categorisation/GATO:${PYTHONPATH:-}"

    case "${METHOD}" in

        CutBased)
            run_step "Cut-based STXS categorisation"   bash "${SCRIPTS_CB}/run_categorisation.sh"
            run_step "Cut-based ZX categorisation"     bash "${SCRIPTS_CB}/run_ZX_categorisation.sh"
            ;;

        BDT)
            if [[ ${SKIP_TRAIN} -eq 0 ]]; then
                run_step "BDT training (Optuna + XGBoost)" bash "${SCRIPTS_BDT}/run.sh"
            else
                log "   Skipping BDT training (--skip-train)"
            fi
            run_step "BDT inference → BDT_category branch" bash "${SCRIPTS_BDT}/predict_bdt_category.sh"
            ;;

        DNN)
            if [[ ${SKIP_TRAIN} -eq 0 ]]; then
                run_step "DNN training" bash "${SCRIPTS_DNN}/run_DNN.sh"
            else
                log "   Skipping DNN training (--skip-train)"
            fi
            # DNN training script writes the categorised ROOT as part of evaluation.
            # If a separate predict step exists, add it here.
            ;;

        GATO)
            if [[ ${SKIP_TRAIN} -eq 0 ]]; then
                run_step "GATO training" bash "${SCRIPTS_GATO}/run_GATO.sh"
            else
                log "   Skipping GATO training (--skip-train)"
            fi
            run_step "GATO inference → GATO_bin branch" bash "${SCRIPTS_GATO}/predict_gato_bin.sh"
            ;;
    esac

    log "-------- STAGE 2 complete --------"
fi

# ---------------------------------------------------------------------------
# STAGE 3: inference
# Fit m4l shapes, build RooFit workspace, write datacard, run text2workspace.
# ---------------------------------------------------------------------------
if should_run inference; then
    log "-------- STAGE 3: InferenceModel (${METHOD}) --------"

    # Response matrix runs under LCG Python (ROOT + RooFit).
    set +u
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
    set -u
    export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH:-}"

    run_step "Response matrix + DSCB fits"  bash "${SCRIPTS_IM}/run_response_matrix.sh" "${METHOD}"

    TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${METHOD}"
    check_file "${TRIAL_DIR}/workspace.root"   "response matrix"
    check_file "${TRIAL_DIR}/epsilonA.pkl"     "response matrix"

    # Datacard pipeline handles the LCG→CMSSW environment switch internally.
    run_step "Datacard + text2workspace"    bash "${SCRIPTS_IM}/run_datacard_pipeline.sh" "${METHOD}"

    check_file "${TRIAL_DIR}/datacard_workspace.root" "text2workspace"

    log "-------- STAGE 3 complete --------"
fi

# ---------------------------------------------------------------------------
# STAGE 4: scan
# 1D likelihood scans for all 19 POIs (parallel batches, CMSSW combine).
# ---------------------------------------------------------------------------
if should_run scan; then
    log "-------- STAGE 4: 1D likelihood scans (${METHOD}) --------"

    TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${METHOD}"
    check_file "${TRIAL_DIR}/datacard_workspace.root" "stage 3"

    # run_1D_scan.sh handles its own CMSSW environment setup.
    run_step "1D POI scans (19 × batched)" bash "${SCRIPTS_IM}/run_1D_scan.sh" "${METHOD}"

    log "-------- STAGE 4 complete --------"
fi

# ---------------------------------------------------------------------------
# STAGE 5: plots
# Summary scan plots.
# ---------------------------------------------------------------------------
if should_run plots; then
    log "-------- STAGE 5: Plots (${METHOD}) --------"

    set +u
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
    set -u
    export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH:-}"

    run_step "1D scan summary plot" bash "${SCRIPTS_IM}/run_plot_all_1D_scans.sh" "${METHOD}"

    log "-------- STAGE 5 complete --------"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "========================================================"
log "  Pipeline complete  (method: ${METHOD})"
log "  Results:  ${HZZ_ROOT}/InferenceModel/Results/${METHOD}/"
log "  Plots:    ${HZZ_ROOT}/InferenceModel/Results/${METHOD}/combine_output/"
log "  Full log: ${PIPELINE_LOG}"
log "========================================================"
