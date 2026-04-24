#!/bin/bash
# HZZAnalysis environment setup.
#
# Usage:
#   source setup_env.sh              # sets HZZ_ROOT, LCG view, PYTHONPATH
#   hzz_cmsenv                       # enters CMSSW runtime for Combine

# Resolve HZZ_ROOT to the directory of this script.
_this_script="${BASH_SOURCE[0]:-$0}"
export HZZ_ROOT="$(cd "$(dirname "${_this_script}")" && pwd)"
unset _this_script

# LCG view (Python + ROOT + uproot + XGBoost + PyTorch + TF + TFP)
LCG_VIEW="/cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt"
if [ -f "${LCG_VIEW}/setup.sh" ]; then
    source "${LCG_VIEW}/setup.sh"
else
    echo "[hzz setup] WARNING: ${LCG_VIEW}/setup.sh not found; LCG view not set."
fi

# Expose the analysis packages on PYTHONPATH.
export PYTHONPATH="${HZZ_ROOT}:${HZZ_ROOT}/Categorisation:${HZZ_ROOT}/Categorisation/GATO:${PYTHONPATH}"

# Convenience function to enter the CMSSW runtime for Combine.
hzz_cmsenv () {
    local cmssw_src="${HZZ_ROOT}/InferenceModel/CMSSW_14_1_0_pre4/src"
    if [ ! -d "${cmssw_src}" ]; then
        echo "[hzz_cmsenv] ${cmssw_src} not found."
        return 1
    fi
    ( cd "${cmssw_src}" && eval "$(scramv1 runtime -sh)" && cd "${HZZ_ROOT}" && bash )
}

echo "[hzz setup] HZZ_ROOT=${HZZ_ROOT}"
