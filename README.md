# HZZAnalysis

End-to-end H→ZZ→4l STXS categorisation pipeline.

## Pipeline

```
POWHEG ROOT
    │
    ├── DataPreprocessing/      filter → numpy → plots
    │
    ├── Categorisation/         adds a category column per event
    │       CutBased/           run2_categorisation (first_step + second_step)
    │       BDT/                XGBoost multiclass argmax
    │       DNN/                PyTorch multiclass argmax
    │       GATO/               gatohep Gaussian-mixture optimiser
    │       Comparison/         cross-method ROC / yield overlays
    │
    └── InferenceModel/         response matrix → datacard → Combine scans
            Results/{CutBased,BDT,DNN,GATO}/    per-method outputs
            CMSSW_14_1_0_pre4/                  Combine build
```

## Quick start

```bash
cd /afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis
source setup_env.sh                         # sets HZZ_ROOT, LCG, PYTHONPATH
```

## Stage entry-points

| Stage              | Runner                                                                | Output                                        |
|--------------------|-----------------------------------------------------------------------|-----------------------------------------------|
| Filter MC          | `DataPreprocessing/RunScripts/parallel_filter.sh`                     | filtered ROOT on EOS                          |
| Filter data        | `DataPreprocessing/RunScripts/parallel_filter_data.sh`                | filtered ROOT on EOS                          |
| Filter ZX CR       | `DataPreprocessing/RunScripts/parallel_filter_ZXCR.sh`                | filtered ZX ROOT on EOS                       |
| ROOT → numpy       | `DataPreprocessing/RunScripts/root_to_numpy.sh`                       | numpy dumps on EOS                            |
| Diagnostic plots   | `DataPreprocessing/RunScripts/plotter.sh`                             | `DataPreprocessing/Plots/`                    |
| Cut-based cat.     | `Categorisation/CutBased/RunScripts/run_categorisation.sh`            | yield tables                                  |
| BDT train          | `Categorisation/BDT/RunScripts/run.sh`                                | `Categorisation/BDT/Models/`                  |
| BDT predict        | `Categorisation/BDT/RunScripts/predict_bdt_category.sh`               | ROOT + `BDT_category` branch                  |
| DNN train          | `Categorisation/DNN/RunScripts/run_DNN.sh`                            | `Categorisation/DNN/Models/`                  |
| GATO train         | `Categorisation/GATO/RunScripts/run_GATO.sh`                          | `Categorisation/GATO/Models/`                 |
| GATO predict       | `Categorisation/GATO/RunScripts/predict_gato_bin.sh`                  | ROOT + `GATO_bin` branch                      |
| Response matrix    | `InferenceModel/RunScripts/run_response_matrix.sh --method <M>`       | `InferenceModel/Results/<M>/epsilonA.pkl`     |
| Datacard           | `InferenceModel/RunScripts/run_datacard_pipeline.sh --method <M>`     | `InferenceModel/Results/<M>/datacard.txt`     |
| Combine 1D scan    | `InferenceModel/RunScripts/run_1D_scan.sh --method <M>`               | `InferenceModel/Results/<M>/combine_output/`  |

`<M>` ∈ `{CutBased, BDT, DNN, GATO}`.

## Results seeded from previous trials

- `InferenceModel/Results/CutBased/` — copied from `inferenceModel/results/trial10/`.
- `InferenceModel/Results/BDT/`      — copied from `inferenceModel/results/trial11/`.
- `InferenceModel/Results/GATO/`     — copied from `inferenceModel/results/trial12/`.
- `InferenceModel/Results/DNN/`      — empty; to be filled once DNN inference is run.

## Originals

The originals under `Thesis/2023Samples/`, `Thesis/multiclass_HZZ_STXS/`,
`Thesis/gato-hep/`, `Thesis/inferenceModel/`, and `Thesis/CMSSW_14_1_0_pre4/` are
left untouched and continue to work. Delete or archive them once the new tree is
verified.
