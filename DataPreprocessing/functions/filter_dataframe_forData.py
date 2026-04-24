import ROOT
import argparse
import os
from get_genEventSumw import get_genEventSumw

def main():
    parser = argparse.ArgumentParser(description="Filter the Jet columns from a ROOT file for the leading and subleading jet kinematics. \
                                     Then save the dataframe to the 'Events' tree of the output path.")
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to the ROOT file')
    parser.add_argument('--outputPath', '-o', type=str, default='.', help='Output path for the dataframe')
    parser.add_argument('--btagThreshold', '-b', type=float, default=0.2421, help='the b-tagging threshold to apply to the jets')
    args = parser.parse_args()

    file_path = args.file
    output_path = args.outputPath
    b_tag_threshold = args.btagThreshold
    name_tree = "Events"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Load discriminant functions from header file (resolved relative to this script)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT.gInterpreter.Declare(f'#define HZZ_CCONST_DIR "{_HERE}/cconstants"')
    ROOT.gInterpreter.ProcessLine(f'#include "{_HERE}/discriminants.h"')

    rdf = ROOT.RDataFrame(name_tree, file_path)
    new_rdf = rdf.Filter("bestCandIdx > -1") \
        .Filter("ZZCand_mass[bestCandIdx] > 105 && ZZCand_mass[bestCandIdx] < 140") \
        .Define("ZZmass", "ZZCand_mass[bestCandIdx]") \
        .Define("ZZVector", "ROOT::Math::PtEtaPhiMVector(ZZCand_pt[bestCandIdx], ZZCand_eta[bestCandIdx], ZZCand_phi[bestCandIdx], ZZCand_mass[bestCandIdx])") \
        .Define("JetLeading_pt", "JetLeadingIdx != -1 ? Jet_pt[JetLeadingIdx] : -999.0") \
        .Define("JetLeading_eta", "JetLeadingIdx != -1 ? Jet_eta[JetLeadingIdx] : -999.0") \
        .Define("JetLeading_phi", "JetLeadingIdx != -1 ? Jet_phi[JetLeadingIdx] : -999.0") \
        .Define("JetLeading_mass", "JetLeadingIdx != -1 ? Jet_mass[JetLeadingIdx] : -999.0") \
        .Define("JetLeading_btag", "JetLeadingIdx != -1 ? Jet_btagPNetB[JetLeadingIdx] : -999.0") \
        .Define("JetSubLeading_pt", "JetSubleadingIdx != -1 ? Jet_pt[JetSubleadingIdx] : -999.0") \
        .Define("JetSubLeading_eta", "JetSubleadingIdx != -1 ? Jet_eta[JetSubleadingIdx] : -999.0") \
        .Define("JetSubLeading_phi", "JetSubleadingIdx != -1 ? Jet_phi[JetSubleadingIdx] : -999.0") \
        .Define("JetSubLeading_mass", "JetSubleadingIdx != -1 ? Jet_mass[JetSubleadingIdx] : -999.0") \
        .Define("JetSubLeading_btag", "JetSubleadingIdx != -1 ? Jet_btagPNetB[JetSubleadingIdx] : -999.0") \
        .Define("deltaEta_jj", "JetLeadingIdx != -1 && JetSubleadingIdx != -1 ? abs(Jet_eta[JetLeadingIdx] - Jet_eta[JetSubleadingIdx]) : -999.0") \
        .Define("deltaPhi_jj", "JetLeadingIdx != -1 && JetSubleadingIdx != -1 ? ROOT::VecOps::DeltaPhi(Jet_phi[JetLeadingIdx], Jet_phi[JetSubleadingIdx]) : -999.0") \
        .Define("LeadingVector", "JetLeadingIdx != -1 ? ROOT::Math::PtEtaPhiMVector(Jet_pt[JetLeadingIdx], Jet_eta[JetLeadingIdx], Jet_phi[JetLeadingIdx], Jet_mass[JetLeadingIdx]) : ROOT::Math::PtEtaPhiMVector(-999.0, -999.0, -999.0, -999.0)") \
        .Define("SubLeadingVector", "JetSubleadingIdx != -1 ? ROOT::Math::PtEtaPhiMVector(Jet_pt[JetSubleadingIdx], Jet_eta[JetSubleadingIdx], Jet_phi[JetSubleadingIdx], Jet_mass[JetSubleadingIdx]) : ROOT::Math::PtEtaPhiMVector(-999.0, -999.0, -999.0, -999.0)") \
        .Define("m_jj", "JetLeadingIdx != -1 && JetSubleadingIdx != -1 ? (LeadingVector + SubLeadingVector).M() : -999.0") \
        .Define("ZZjj_pt", "JetLeadingIdx != -1 && JetSubleadingIdx != -1 ? (LeadingVector + SubLeadingVector + ZZVector).Pt() : -999.0") \
        .Define("Jet_btagPNetB_filtered", "ROOT::VecOps::RVec<float> filtered; for (int i = 0; i < Jet_pt.size(); i++) { if (Jet_pt[i] > 30 && !Jet_ZZMask[i]) filtered.push_back(Jet_btagPNetB[i]); } return filtered;") \
        .Define("nBtagged_filtered", f"Sum(Jet_btagPNetB_filtered > {b_tag_threshold})")
    
    # Define MET alias for consistency (2023 samples use MET_pt instead of PFMET_pt)
    new_rdf = new_rdf.Define("PFMET_pt", "MET_pt")
    
    # Define discriminants
    new_rdf = new_rdf \
        .Define("DVBF2j_ME", "nCleanedJetsPt30 >= 2 ? DVBF2j_ME(ZZCand_P_JJVBF_SIG_ghv1_1_JHUGen_JECNominal[bestCandIdx], ZZCand_P_JJQCD_SIG_ghg2_1_JHUGen_JECNominal[bestCandIdx], ZZCand_mass[bestCandIdx]) : -999.0") \
        .Define("DVBF1j_ME", "nCleanedJetsPt30 == 1 ? DVBF1j_ME(ZZCand_P_JVBF_SIG_ghv1_1_JHUGen_JECNominal[bestCandIdx], ZZCand_P_JVBF_SIG_ghv1_1_JHUGen_JECNominal_aux[bestCandIdx], ZZCand_P_JQCD_SIG_ghg2_1_JHUGen_JECNominal[bestCandIdx], ZZCand_mass[bestCandIdx]) : -999.0") \
        .Define("DWHh_ME", "nCleanedJetsPt30 >= 2 ? DWHh_ME(ZZCand_P_HadWH_SIG_ghw1_1_JHUGen_JECNominal[bestCandIdx], ZZCand_P_JJQCD_SIG_ghg2_1_JHUGen_JECNominal[bestCandIdx], ZZCand_P_HadWH_SIG_ghw1_1_JHUGen_JECNominal_mavjj[bestCandIdx], ZZCand_P_HadWH_SIG_ghw1_1_JHUGen_JECNominal_mvajj_true[bestCandIdx], ZZCand_mass[bestCandIdx]) : -999.0") \
        .Define("DZHh_ME", "nCleanedJetsPt30 >= 2 ? DZHh_ME(ZZCand_P_HadZH_SIG_ghz1_1_JHUGen_JECNominal[bestCandIdx], ZZCand_P_JJQCD_SIG_ghg2_1_JHUGen_JECNominal[bestCandIdx], ZZCand_P_HadZH_SIG_ghz1_1_JHUGen_JECNominal_mavjj[bestCandIdx], ZZCand_P_HadZH_SIG_ghz1_1_JHUGen_JECNominal_mvajj_true[bestCandIdx], ZZCand_mass[bestCandIdx]) : -999.0") \
        .Define("DVBF2j_ME_noC", "nCleanedJetsPt30 >= 2 ? 1.f/(1.f + ZZCand_P_JJQCD_SIG_ghg2_1_JHUGen_JECNominal[bestCandIdx] / ZZCand_P_JJVBF_SIG_ghv1_1_JHUGen_JECNominal[bestCandIdx]) : -999.0") \
        .Define("DVBF1j_ME_noC", "nCleanedJetsPt30 == 1 ? 1.f/(1.f + ZZCand_P_JQCD_SIG_ghg2_1_JHUGen_JECNominal[bestCandIdx] / (ZZCand_P_JVBF_SIG_ghv1_1_JHUGen_JECNominal[bestCandIdx] * ZZCand_P_JVBF_SIG_ghv1_1_JHUGen_JECNominal_aux[bestCandIdx])) : -999.0") \
        .Define("DWHh_ME_noC", "nCleanedJetsPt30 >= 2 ? 1.f/(1.f + (ZZCand_P_HadWH_SIG_ghw1_1_JHUGen_JECNominal_mvajj_true[bestCandIdx] * ZZCand_P_JJQCD_SIG_ghg2_1_JHUGen_JECNominal[bestCandIdx]) / (ZZCand_P_HadWH_SIG_ghw1_1_JHUGen_JECNominal_mavjj[bestCandIdx] * ZZCand_P_HadWH_SIG_ghw1_1_JHUGen_JECNominal[bestCandIdx])) : -999.0") \
        .Define("DZHh_ME_noC", "nCleanedJetsPt30 >= 2 ? 1.f/(1.f + (ZZCand_P_HadZH_SIG_ghz1_1_JHUGen_JECNominal_mvajj_true[bestCandIdx] * ZZCand_P_JJQCD_SIG_ghg2_1_JHUGen_JECNominal[bestCandIdx]) / (ZZCand_P_HadZH_SIG_ghz1_1_JHUGen_JECNominal_mavjj[bestCandIdx] * ZZCand_P_HadZH_SIG_ghz1_1_JHUGen_JECNominal[bestCandIdx])) : -999.0") 
    
    
    # Define lepton features
    new_rdf = new_rdf \
                    .Define('Z1_lep_hi_idx', "Lepton_pt[ZZCand_Z1l1Idx[bestCandIdx]] >= Lepton_pt[ZZCand_Z1l2Idx[bestCandIdx]] ? ZZCand_Z1l1Idx[bestCandIdx] : ZZCand_Z1l2Idx[bestCandIdx]") \
                    .Define('Z1_lep_lo_idx', "Lepton_pt[ZZCand_Z1l1Idx[bestCandIdx]] >= Lepton_pt[ZZCand_Z1l2Idx[bestCandIdx]] ? ZZCand_Z1l2Idx[bestCandIdx] : ZZCand_Z1l1Idx[bestCandIdx]") \
                    .Define('Z2_lep_hi_idx', "Lepton_pt[ZZCand_Z2l1Idx[bestCandIdx]] >= Lepton_pt[ZZCand_Z2l2Idx[bestCandIdx]] ? ZZCand_Z2l1Idx[bestCandIdx] : ZZCand_Z2l2Idx[bestCandIdx]") \
                    .Define('Z2_lep_lo_idx', "Lepton_pt[ZZCand_Z2l1Idx[bestCandIdx]] >= Lepton_pt[ZZCand_Z2l2Idx[bestCandIdx]] ? ZZCand_Z2l2Idx[bestCandIdx] : ZZCand_Z2l1Idx[bestCandIdx]") \
                    .Define('LepPt', "std::vector<float> LepPt{Lepton_pt[Z1_lep_hi_idx], Lepton_pt[Z1_lep_lo_idx], Lepton_pt[Z2_lep_hi_idx], Lepton_pt[Z2_lep_lo_idx]}; return LepPt") \
                    .Define('LepEta', "std::vector<float> LepEta{Lepton_eta[Z1_lep_hi_idx], Lepton_eta[Z1_lep_lo_idx], Lepton_eta[Z2_lep_hi_idx], Lepton_eta[Z2_lep_lo_idx]}; return LepEta") \
                    .Define('LepPhi', "std::vector<float> LepPhi{Lepton_phi[Z1_lep_hi_idx], Lepton_phi[Z1_lep_lo_idx], Lepton_phi[Z2_lep_hi_idx], Lepton_phi[Z2_lep_lo_idx]}; return LepPhi") \
                    .Define('LepPdgId', "std::vector<int> LepPdgId{Lepton_pdgId[Z1_lep_hi_idx], Lepton_pdgId[Z1_lep_lo_idx], Lepton_pdgId[Z2_lep_hi_idx], Lepton_pdgId[Z2_lep_lo_idx]}; return LepPdgId") \
                    .Define('LepPt_0', "LepPt[0]") \
                    .Define('LepEta_0', "LepEta[0]") \
                    .Define('LepPhi_0', "LepPhi[0]") \
                    .Define('LepPdgId_0', "LepPdgId[0]") \
                    .Define('LepPt_1', "LepPt[1]") \
                    .Define('LepEta_1', "LepEta[1]") \
                    .Define('LepPhi_1', "LepPhi[1]") \
                    .Define('LepPdgId_1', "LepPdgId[1]") \
                    .Define('LepPt_2', "LepPt[2]") \
                    .Define('LepEta_2', "LepEta[2]") \
                    .Define('LepPhi_2', "LepPhi[2]") \
                    .Define('LepPdgId_2', "LepPdgId[2]") \
                    .Define('LepPt_3', "LepPt[3]") \
                    .Define('LepEta_3', "LepEta[3]") \
                    .Define('LepPhi_3', "LepPhi[3]") \
                    .Define('LepPdgId_3', "LepPdgId[3]") \
                    .Define("LepPt_4", "ZZCand_extraLep1Idx[bestCandIdx] != -1 ? Lepton_pt[ZZCand_extraLep1Idx[bestCandIdx]] : -999.0") \
                    .Define("LepEta_4", "ZZCand_extraLep1Idx[bestCandIdx] != -1 ? Lepton_eta[ZZCand_extraLep1Idx[bestCandIdx]] : -999.0") \
                    .Define("LepPhi_4", "ZZCand_extraLep1Idx[bestCandIdx] != -1 ? Lepton_phi[ZZCand_extraLep1Idx[bestCandIdx]] : -999.0") \
                    .Define("LepPdgId_4", "ZZCand_extraLep1Idx[bestCandIdx] != -1 ? Lepton_pdgId[ZZCand_extraLep1Idx[bestCandIdx]] : 0") \
                    .Define("LepPt_5", "ZZCand_extraLep2Idx[bestCandIdx] != -1 ? Lepton_pt[ZZCand_extraLep2Idx[bestCandIdx]] : -999.0") \
                    .Define("LepEta_5", "ZZCand_extraLep2Idx[bestCandIdx] != -1 ? Lepton_eta[ZZCand_extraLep2Idx[bestCandIdx]] : -999.0") \
                    .Define("LepPhi_5", "ZZCand_extraLep2Idx[bestCandIdx] != -1 ? Lepton_phi[ZZCand_extraLep2Idx[bestCandIdx]] : -999.0") \
                    .Define("LepPdgId_5", "ZZCand_extraLep2Idx[bestCandIdx] != -1 ? Lepton_pdgId[ZZCand_extraLep2Idx[bestCandIdx]] : 0") \
                    

    # Persist only the analysis features to keep the Snapshot JIT manageable
    columns_to_save = [
        # ZZ candidate features
        "ZZCand_pt", "ZZCand_eta", "ZZCand_phi", "ZZmass",
        "ZZCand_costheta1", "ZZCand_costheta2", "ZZCand_costhetastar", "ZZCand_Phi1",
        "ZZCand_nExtraLep", "ZZjj_pt",
        # MET
        "PFMET_pt",
        # Jet features
        "JetLeading_pt", "JetLeading_eta", "JetLeading_mass", "JetLeading_phi",
        "JetSubLeading_pt", "JetSubLeading_eta", "JetSubLeading_mass", "JetSubLeading_phi",
        "nCleanedJetsPt30", "nBtagged_filtered",
        "JetLeading_btag", "JetSubLeading_btag",
        # Dijet features
        "deltaEta_jj", "deltaPhi_jj", "m_jj",
        # Lepton features (primary 4 leptons)
        "LepPt_0", "LepPt_1", "LepPt_2", "LepPt_3",
        "LepEta_0", "LepEta_1", "LepEta_2", "LepEta_3",
        "LepPhi_0", "LepPhi_1", "LepPhi_2", "LepPhi_3",
        "LepPdgId_0", "LepPdgId_1", "LepPdgId_2", "LepPdgId_3",
        # Extra lepton features
        "LepPt_4", "LepPt_5",
        "LepEta_4", "LepEta_5",
        "LepPhi_4", "LepPhi_5",
        "LepPdgId_4", "LepPdgId_5",
        # Discriminants
        "DVBF2j_ME", "DVBF1j_ME", "DWHh_ME", "DZHh_ME", "DVBF2j_ME_noC", "DVBF1j_ME_noC", "DWHh_ME_noC", "DZHh_ME_noC", "ZZCand_KD",
    ]
    
    # Snapshot is lazy - trigger execution
    snapshot_result = new_rdf.Snapshot("Events", f"{output_path}.root", columns_to_save)
    print("Snapshot created.")
    # Force execution and get event count
    n_events = snapshot_result.Count().GetValue()
    print(f"Saved {n_events} events to {output_path}.root")

if __name__ == "__main__":
    main()
    