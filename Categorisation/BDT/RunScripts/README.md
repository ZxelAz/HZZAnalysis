# Multiclass BDT Training Script

This repository contains a script to train a multiclass Boosted Decision Tree (BDT) classifiers for the STXS categorisation in the Higgs-to-four-lepton channel.

## Environment

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
```
If you want to use the GPU (tested on lxplus-gpu):
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
```

### Stage 0 dictionary:
UNKNOWN = 0,
GG2H_FWDH = 10,
GG2H = 11,
VBF_FWDH = 20,
VBF = 21,
VH2HQQ_FWDH = 22,
VH2HQQ = 23,
QQ2HLNU_FWDH = 30,
QQ2HLNU = 31,
QQ2HLL_FWDH = 40,
QQ2HLL = 41,
GG2HLL_FWDH = 50,
GG2HLL = 51,
TTH_FWDH = 60,
TTH = 61,
BBH_FWDH = 70,
BBH = 71,
TH_FWDH = 80,
TH = 81


