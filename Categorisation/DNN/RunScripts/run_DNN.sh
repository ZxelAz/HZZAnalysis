#!/bin/bash
# DNN training entry-point.

: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
DATA_DIR="/eos/user/z/zhiheng/STXS_samples"
TRIAL="${TRIAL:-trial2}"
OUTPUT_DIR="${HZZ_ROOT}/Categorisation/DNN/Models/${TRIAL}"
SCRIPT="${HZZ_ROOT}/Categorisation/DNN/RunScripts/train_multiclass_DNN.py"

mkdir -p "${OUTPUT_DIR}"

if [ -z "${LCG_VERSION:-}" ]; then
    source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
fi

export PYTHONPATH="${HZZ_ROOT}:${PYTHONPATH}"

python -u "${SCRIPT}" \
    --data \
        "$DATA_DIR"/2023samples/preBPix/root_data/{ggH125,VBFH125,ZH125}.root \
        "$DATA_DIR"/2023samples/preBPix/root_data/{WplusH125,WminusH125,ttH125}.root \
        "$DATA_DIR"/2023samples/postBPix/root_data/{ggH125,VBFH125,ZH125}.root \
        "$DATA_DIR"/2023samples/postBPix/root_data/{WplusH125,WminusH125,ttH125}.root \
        "$DATA_DIR"/2023samples/preBPix/root_data/ZZTo4l.root \
        "$DATA_DIR"/2023samples/postBPix/root_data/ZZTo4l.root \
        \
        "$DATA_DIR"/2022samples/preEE/root_data/{ggH125,VBFH125,ZH125}.root \
        "$DATA_DIR"/2022samples/preEE/root_data/{WplusH125,WminusH125,ttH125}.root \
        "$DATA_DIR"/2022samples/postEE/root_data/{ggH125,VBFH125,ZH125}.root \
        "$DATA_DIR"/2022samples/postEE/root_data/{WplusH125,WminusH125,ttH125}.root \
        "$DATA_DIR"/2022samples/preEE/root_data/ZZTo4l.root \
        "$DATA_DIR"/2022samples/postEE/root_data/ZZTo4l.root \
    --target HTXS_stage1_2_cat_pTjet30GeV \
    --features ZZCand_pt ZZCand_eta ZZCand_phi \
        PFMET_pt ZZCand_costheta1 ZZCand_costheta2 ZZCand_costhetastar ZZCand_Phi1 \
        JetLeading_pt JetLeading_eta JetLeading_mass JetLeading_phi JetSubLeading_pt JetSubLeading_eta JetSubLeading_mass JetSubLeading_phi \
        deltaEta_jj deltaPhi_jj m_jj nCleanedJetsPt30 \
        LepPt_0 LepPt_1 LepPt_2 LepPt_3 LepPdgId_0 LepPdgId_1 LepPdgId_2 LepPdgId_3 LepEta_0 LepEta_1 LepEta_2 LepEta_3 LepPhi_0 LepPhi_1 LepPhi_2 LepPhi_3 \
        LepPt_4 LepPt_5 LepEta_4 LepEta_5 LepPhi_4 LepPhi_5 LepPdgId_4 LepPdgId_5 \
        ZZCand_nExtraLep \
        "DVBF2j_ME" "DVBF1j_ME" "DWHh_ME" "DZHh_ME" \
        "JetLeading_btag" "JetSubLeading_btag" "nBtagged_filtered" \
        "ZZjj_pt" "ZZCand_KD" \
    --EventWeight EventWeight_lumi138 \
    --run2-yield \
    --class-name-dict merged \
    --weights trainWeight \
    --mode-name production_mode \
    --tree-name Events \
    --output "${OUTPUT_DIR}/model" \
    --plot \
    --use-gpu \
    --hidden-dims 256 128 64 \
    --dropout 0.3 \
    --learning-rate 0.001 \
    --batch-size 256 \
    --n-epochs 100 \
    --weight-decay 1e-5 \
    --activation relu
    #--tune-hyperparameters 30
