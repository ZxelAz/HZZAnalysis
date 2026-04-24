#!/usr/bin/env bash
# run_pipeline.sh — end-to-end H→ZZ→4l STXS analysis pipeline.
#
# Stages (in order):
#   filter        DataPreprocessing: filter ROOT → filtered ROOT + numpy arrays + ZX yield
#   categorise    Apply categorisation; adds category column(s) to the output ROOT file.
#                 With --method all, all four methods run in sequence so every column
#                 (run2_category, BDT_category, DNN_category, GATO_bin) lands in the
#                 final chained ROOT file.
#   inference     Response matrix → datacard → workspace (text2workspace), per method
#   scan          1D likelihood scans over all 19 POIs, per method
#   plots         Per-method scan plots; with --method all also produces cross-method
#                 rate comparison (CutBased vs BDT vs GATO).
#
# Usage:
#   bash run_pipeline.sh [OPTIONS]
#
# Options:
#   -m, --method   METHOD     CutBased | BDT | DNN | GATO | all  (default: GATO)
#   -f, --from     STAGE      filter | categorise | inference | scan | plots  (default: filter)
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
        -m|--method)     METHOD="$2";     shift 2 ;;
        -f|--from)       FROM_STAGE="$2"; shift 2 ;;
        -s|--skip-train) SKIP_TRAIN=1;    shift   ;;
        -n|--dry-run)    DRY_RUN=1;       shift   ;;
        -h|--help)
            sed -n '2,23p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "${METHOD}" in
    CutBased|BDT|DNN|GATO|all) ;;
    *) echo "ERROR: Unknown method '${METHOD}'. Choose: CutBased BDT DNN GATO all"; exit 1 ;;
esac

case "${FROM_STAGE}" in
    filter|categorise|inference|scan|plots) ;;
    *) echo "ERROR: Unknown stage '${FROM_STAGE}'. Choose: filter categorise inference scan plots"; exit 1 ;;
esac

# Expand "all" into the ordered list of methods.
if [[ "${METHOD}" == "all" ]]; then
    METHODS_TO_RUN=(CutBased BDT DNN GATO)
else
    METHODS_TO_RUN=("${METHOD}")
fi

# ---------------------------------------------------------------------------
# Paths
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

ts()  { date '+%H:%M:%S'; }
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
    # skip in dry-run mode (files won't exist yet)
    [[ ${DRY_RUN} -eq 1 ]] && return 0
    if [[ ! -f "${f}" ]]; then
        log "ERROR: Expected file missing after ${label}: ${f}"
        exit 1
    fi
}

lcg_setup() {
    set +u
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
    set -u
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
log "========================================================"
log "  HZZ STXS pipeline"
log "  Method(s):   ${METHOD}  →  ${METHODS_TO_RUN[*]}"
log "  From stage:  ${FROM_STAGE}"
log "  Skip train:  ${SKIP_TRAIN}"
log "  Dry run:     ${DRY_RUN}"
log "  HZZ_ROOT:    ${HZZ_ROOT}"
log "  Log:         ${PIPELINE_LOG}"
log "========================================================"

# ---------------------------------------------------------------------------
# STAGE 1: filter
# ---------------------------------------------------------------------------
if should_run filter; then
    log "-------- STAGE 1: DataPreprocessing / filter --------"

    lcg_setup
    export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH:-}"

    run_step "Filter MC samples (4 campaigns)"     bash "${SCRIPTS_DP}/parallel_filter.sh"
    run_step "Filter collision data"               bash "${SCRIPTS_DP}/parallel_filter_data.sh"
    run_step "Filter ZX control region"            bash "${SCRIPTS_DP}/parallel_filter_ZXCR.sh"
    run_step "Split 2022/2023 data by run number"  bash "${SCRIPTS_DP}/split_year.sh"

    # numpy arrays needed by all ML-based methods
    local_methods="${METHODS_TO_RUN[*]}"
    if [[ "${local_methods}" != "CutBased" ]]; then
        run_step "Convert filtered ROOT → numpy"   bash "${SCRIPTS_DP}/root_to_numpy.sh"
    fi

    run_step "ZX yield estimate"                   bash "${SCRIPTS_DP}/run_ZX_yield_calculator.sh"

    log "-------- STAGE 1 complete --------"
fi

# ---------------------------------------------------------------------------
# STAGE 2: categorise
# Each method adds its own branch to the output ROOT.  When METHOD=all the
# methods run in sequence so all four columns accumulate in the chained file:
#   stage1p2_combined.root          ← base (run2_category added by CutBased)
#   stage1p2_combined_with_bdt.root ← base + BDT_category
#   stage1p2_combined_with_dnn.root ← base + DNN_category  (independent chain)
#   stage1p2_combined_with_gato.root← bdt  + GATO_bin
# The final GATO file therefore carries run2_category + BDT_category + GATO_bin.
# ---------------------------------------------------------------------------
if should_run categorise; then
    log "-------- STAGE 2: Categorisation (${METHODS_TO_RUN[*]}) --------"

    lcg_setup
    export PYTHONPATH="${HZZ_ROOT}:${HZZ_ROOT}/Categorisation:${HZZ_ROOT}/Categorisation/GATO:${PYTHONPATH:-}"

    for M in "${METHODS_TO_RUN[@]}"; do
        log "  -- Categorisation: ${M} --"
        case "${M}" in

            CutBased)
                run_step "[CutBased] STXS categorisation → run2_category" \
                    bash "${SCRIPTS_CB}/run_categorisation.sh"
                run_step "[CutBased] ZX categorisation" \
                    bash "${SCRIPTS_CB}/run_ZX_categorisation.sh"
                ;;

            BDT)
                if [[ ${SKIP_TRAIN} -eq 0 ]]; then
                    run_step "[BDT] Training (Optuna + XGBoost)" \
                        bash "${SCRIPTS_BDT}/run.sh"
                else
                    log "   Skipping BDT training (--skip-train)"
                fi
                run_step "[BDT] Inference → BDT_category branch" \
                    bash "${SCRIPTS_BDT}/predict_bdt_category.sh"
                ;;

            DNN)
                if [[ ${SKIP_TRAIN} -eq 0 ]]; then
                    run_step "[DNN] Training + evaluation → DNN_category branch" \
                        bash "${SCRIPTS_DNN}/run_DNN.sh"
                else
                    log "   Skipping DNN training (--skip-train)"
                fi
                ;;

            GATO)
                if [[ ${SKIP_TRAIN} -eq 0 ]]; then
                    run_step "[GATO] Training" \
                        bash "${SCRIPTS_GATO}/run_GATO.sh"
                else
                    log "   Skipping GATO training (--skip-train)"
                fi
                run_step "[GATO] Inference → GATO_bin branch" \
                    bash "${SCRIPTS_GATO}/predict_gato_bin.sh"
                ;;
        esac
    done

    log "-------- STAGE 2 complete --------"
fi

# ---------------------------------------------------------------------------
# STAGE 3: inference (per method)
# ---------------------------------------------------------------------------
if should_run inference; then
    log "-------- STAGE 3: InferenceModel (${METHODS_TO_RUN[*]}) --------"

    lcg_setup
    export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH:-}"

    for M in "${METHODS_TO_RUN[@]}"; do
        log "  -- Inference: ${M} --"
        TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${M}"

        run_step "[${M}] Response matrix + DSCB fits" \
            bash "${SCRIPTS_IM}/run_response_matrix.sh" "${M}"

        check_file "${TRIAL_DIR}/workspace.root" "[${M}] response matrix"
        check_file "${TRIAL_DIR}/epsilonA.pkl"   "[${M}] response matrix"

        # run_datacard_pipeline.sh handles the internal LCG→CMSSW switch.
        run_step "[${M}] Datacard + text2workspace" \
            bash "${SCRIPTS_IM}/run_datacard_pipeline.sh" "${M}"

        check_file "${TRIAL_DIR}/datacard_workspace.root" "[${M}] text2workspace"
    done

    log "-------- STAGE 3 complete --------"
fi

# ---------------------------------------------------------------------------
# STAGE 4: scan (per method)
# ---------------------------------------------------------------------------
if should_run scan; then
    log "-------- STAGE 4: 1D likelihood scans (${METHODS_TO_RUN[*]}) --------"

    for M in "${METHODS_TO_RUN[@]}"; do
        log "  -- Scan: ${M} --"
        TRIAL_DIR="${HZZ_ROOT}/InferenceModel/Results/${M}"
        check_file "${TRIAL_DIR}/datacard_workspace.root" "stage 3 [${M}]"

        # run_1D_scan.sh handles its own CMSSW environment setup.
        run_step "[${M}] 1D POI scans (19 × batched)" \
            bash "${SCRIPTS_IM}/run_1D_scan.sh" "${M}"
    done

    log "-------- STAGE 4 complete --------"
fi

# ---------------------------------------------------------------------------
# STAGE 5: plots
# Per-method scan summary; cross-method rate comparison when METHOD=all.
# ---------------------------------------------------------------------------
if should_run plots; then
    log "-------- STAGE 5: Plots (${METHODS_TO_RUN[*]}) --------"

    lcg_setup
    export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH:-}"

    for M in "${METHODS_TO_RUN[@]}"; do
        run_step "[${M}] 1D scan summary plot" \
            bash "${SCRIPTS_IM}/run_plot_all_1D_scans.sh" "${M}"
    done

    # Cross-method rate comparison (requires CutBased, BDT, and GATO scans).
    if [[ "${METHOD}" == "all" ]]; then
        run_step "[all] Cross-method rate comparison plot" \
            bash "${SCRIPTS_IM}/run_plot_rate_comparison.sh"
    fi

    log "-------- STAGE 5 complete --------"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "========================================================"
log "  Pipeline complete  (method: ${METHOD})"
for M in "${METHODS_TO_RUN[@]}"; do
    log "  Results [${M}]: ${HZZ_ROOT}/InferenceModel/Results/${M}/"
done
if [[ "${METHOD}" == "all" ]]; then
    log "  Comparison plot: ${HZZ_ROOT}/InferenceModel/Plots/comparison/rate_comparison.png"
fi
log "  Full log: ${PIPELINE_LOG}"
log "========================================================"
