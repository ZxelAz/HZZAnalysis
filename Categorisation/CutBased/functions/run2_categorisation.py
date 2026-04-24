import time
from turtle import mode
import numpy as np 
import matplotlib.pyplot as plt
import ROOT
import argparse
import os
import pandas as pd
# CMS style for plotting
import mplhep as hep
plt.style.use(hep.style.CMS)

all_features = [
        # ZZ candidate features
        "ZZCand_pt", "ZZCand_eta", "ZZCand_phi", "ZZCand_mass",
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
        # Event weights and label
        "EventWeight_lumi18", "EventWeight_lumi9", "EventWeight_lumi138", "production_mode", "EventWeight_lumi62",  
        "HTXS_stage_0", "overallEventWeight", "genWeight", "puWeight", "trainWeight", "HTXS_stage1_2_cat_pTjet30GeV", "HTXS_stage1_2_cat_pTjet30GeV_label", "genEventSumw"
    ]
run2_scale_factor_dict = {
    'ggH125': 48.58/52.23,
    'VBFH125': 3.782/4.078,
    'WplusH125': 0.839913392/0.8889,
    'WminusH125': 0.532731377/0.5677,
    'ZH125': 0.7612/0.9439,
    'ttH125': 	0.5071/0.5700,
    'ZZTo4l': 1.256/1.39,
    'ggTo4e_Contin_MCFM701': 0.00158549/0.00305851,
    'ggTo4mu_Contin_MCFM701': 0.00158549/0.00303575,
    'ggTo4tau_Contin_MCFM701': 0.00158549/0.00303575,
    'ggTo2e2mu_Contin_MCFM701': 0.0031942/0.00624157,
    'ggTo2e2tau_Contin_MCFM701': 0.0031942/0.00624157,
    'ggTo2mu2tau_Contin_MCFM701': 0.0031942/0.00624157,
    'ZZZ': 0.01398/0.0159,
    'WZZ': 0.05565/0.0621,
    'WWZ': 0.1651/0.1851,
    'TTWW': 0.007883/0.008203,
    'TTZZ': 0.001579/0.001572,
    'TTLL_MLL-4to50': 	0.0259/0.03949,
    'TTLL_MLL-50': 0.0259/0.08646
}
production_modes = ['ggH125', 'VBFH125', 'WplusH125', 'WminusH125', 'ZH125', 'ttH125']
qqZZ_bgmodes = ['ZZTo4l']
ggZZ_bgmodes = ['ggTo2e2mu_Contin_MCFM701', 'ggTo4e_Contin_MCFM701', 'ggTo4mu_Contin_MCFM701', 'ggTo4tau_Contin_MCFM701', 'ggTo2mu2tau_Contin_MCFM701', 'ggTo2e2tau_Contin_MCFM701']
EWK_bgmodes = ['WZZ', 'WWZ', 'ZZZ']
TTjets_bgmodes = ['TTZZ', 'TTWW']
all_modes = production_modes + qqZZ_bgmodes + ggZZ_bgmodes + EWK_bgmodes + TTjets_bgmodes

def rdf_To_numpy(rdf, mode_label, variables=all_features, label= "HTXS_stage_0"):
    """Convert specified variables from a ROOT RDataFrame to a dictionary of numpy arrays.
    
    Args:
        rdf: ROOT RDataFrame containing the data
        variables: List of variable names to convert
        mode_label: String label for the production mode
        label: Name of the target label variable
    Returns:
        Dictionary mapping variable names to numpy arrays
    """
    
    # Cast Char_t columns to int to avoid string conversion issues
    char_t_vars = [
        "nCleanedJetsPt30", "nBtagged_filtered", "passedFiducial"
    ]
    
    for var in char_t_vars:
        if var in variables:
            rdf = rdf.Define(f"{var}_int", f"(int){var}")
    
    # Replace Char_t variables with their int versions in the variables list
    vars_to_read = []
    for var in variables:
        if var in char_t_vars:
            vars_to_read.append(f"{var}_int")
        else:
            vars_to_read.append(var)
    
    rdf = rdf.Define('mode_label', f'"{mode_label}"')
    arr = rdf.AsNumpy(vars_to_read + [label] + ['mode_label'])
    
    # Rename the int versions back to original names
    for var in char_t_vars:
        if var in variables and f"{var}_int" in arr:
            arr[var] = arr.pop(f"{var}_int")
    
    # Convert byte strings to regular strings for mode_label if needed
    if arr['mode_label'].dtype.kind in ['O', 'S', 'U']:  # Object, byte string, or unicode
        # Decode if it's bytes
        if arr['mode_label'].dtype.kind == 'S':
            arr['mode_label'] = arr['mode_label'].astype(str)
    
    return arr

def load_arrays_from_root(file_path, name_tree, modes):
    """Load ROOT data from multiple production modes into numpy arrays.
    
    Args:
        file_path: Base path to ROOT files
        name_tree: Name of the TTree to read
        modes: List of production mode names
        
    Returns:
        Dictionary mapping mode names to their numpy array data
    """
    print(f"  Loading {len(modes)} datasets:")
    rdf_dict = {}
    arr_dict = {}
    
    for i, mode in enumerate(modes, 1):
        print(f"    [{i}/{len(modes)}] {mode}...", end=" ", flush=True)
        rdf = ROOT.RDataFrame(name_tree, f"{file_path}/{mode}.root")
        rdf = rdf.Filter('ZZCand_mass[0] < 140')
        arr = rdf_To_numpy(rdf, mode_label=mode)
        
        # Check first array to get event count
        first_key = list(arr.keys())[0]
        n_events = len(arr[first_key])
        print(f"{n_events:,} events")
        
        arr_dict[mode] = arr
    
    return arr_dict


def parse_list_arg(raw_value):
    """Parse comma-separated CLI list into a clean Python list."""
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def infer_year_label(path_value):
    """Infer a year label from the input path basename."""
    return os.path.basename(os.path.normpath(path_value))


def load_arrays_from_multiple_years(input_paths, year_labels, name_tree, modes, run2scale=False):
    """Load and concatenate per-mode arrays from multiple years."""
    if len(input_paths) != len(year_labels):
        raise ValueError("input_paths and year_labels must have the same length")

    combined_arr_dict = {}

    for idx, (input_path, year_label) in enumerate(zip(input_paths, year_labels), 1):
        print(f"  Year [{idx}/{len(input_paths)}]: {year_label}")
        year_arr_dict = load_arrays_from_root(input_path, name_tree, modes)

        # Add year label to each event for each mode.
        for mode in modes:
            first_key = list(year_arr_dict[mode].keys())[0]
            n_events = len(year_arr_dict[mode][first_key])
            year_arr_dict[mode]["year_label"] = np.array([year_label] * n_events, dtype="U32")

        if run2scale:
            print("    Applying Run 3 to Run 2 cross-section scaling factors...")
            for mod in modes:
                scale_factor = run2_scale_factor_dict[mod]
                year_arr_dict[mod]["EventWeight_lumi138"] = year_arr_dict[mod]["EventWeight_lumi138"] * scale_factor
                year_arr_dict[mod]["EventWeight_lumi9"] = year_arr_dict[mod]["EventWeight_lumi9"] * scale_factor

        # Concatenate with previous years mode-by-mode, variable-by-variable.
        for mode in modes:
            if mode not in combined_arr_dict:
                combined_arr_dict[mode] = {k: v.copy() for k, v in year_arr_dict[mode].items()}
            else:
                for var_name, var_data in year_arr_dict[mode].items():
                    combined_arr_dict[mode][var_name] = np.concatenate([combined_arr_dict[mode][var_name], var_data])

    return combined_arr_dict

def merge_modes(arr_dict, modes, variables=all_features + ["HTXS_stage_0"]):
    """Merge arrays from multiple production modes into a single combined dataset.
    
    Args:
        arr_dict: Dictionary mapping mode names to their variable arrays
        modes: List of mode names to merge together
        variables: List of variable names to merge
    
    Returns:
        Dictionary containing concatenated arrays for each variable
    """
    merged = {}
    
    merge_vars = list(variables) + ["mode_label"]
    if all("year_label" in arr_dict[mode] for mode in modes):
        merge_vars.append("year_label")

    for var in merge_vars:
        # Collect arrays for this variable
        arrays_to_concat = []
        for mode in modes:
            if var in arr_dict[mode]:
                arr = arr_dict[mode][var]
                
                # Convert to appropriate dtype
                if var in ["mode_label", "year_label"]:
                    # Keep as string
                    if arr.dtype.kind == 'S':  # byte string
                        arr = arr.astype(str)
                else:
                    # Convert numeric variables to float
                    try:
                        arr = arr.astype(float)
                    except (ValueError, TypeError):
                        print(f"  Warning: Could not convert {var} to float, keeping as is")
                
                arrays_to_concat.append(arr)
        
        if arrays_to_concat:
            merged[var] = np.concatenate(arrays_to_concat)
    
    return merged

def first_step_categorisation(arr):
    """Perform first-step categorization.
    
    Args:
        arr: the merged array containing all events
        
    Returns:
        a dictionary of 7 arrays corresponding to the stage 0 HTXS categorization.
    """
    # Convert numeric columns to float for consistency
    numeric_vars = [
        "ZZCand_nExtraLep", "DVBF2j_ME", "nCleanedJetsPt30", "nBtagged_filtered",
        "DVBF1j_ME", "LepPdgId_4", "LepPdgId_5"
    ]
    
    for var in numeric_vars:
        if var in arr:
            try:
                arr[var] = arr[var].astype(float)
            except (ValueError, TypeError) as e:
                print(f"  Warning: Could not convert {var}: {e}")
    
    # Define masks for each category
    VBF_2jet_tagged_mask = (arr["ZZCand_nExtraLep"]==0) & (arr["DVBF2j_ME"]>0.5) & \
                            (
                                ((np.isin(arr["nCleanedJetsPt30"], [2, 3])) & (arr["nBtagged_filtered"]<=1)) | \
                                ((arr["nCleanedJetsPt30"]==4) & (arr["nBtagged_filtered"]==0))
                            ) 
    
    VH_hadronic_tagged_mask = ~VBF_2jet_tagged_mask & \
                                (
                                    (arr["ZZCand_nExtraLep"]==0) & (arr["DWHh_ME"]>0.5) & (arr["DZHh_ME"]>0.5) & 
                                    ((np.isin(arr["nCleanedJetsPt30"], [2, 3])) | 
                                    ((arr["nCleanedJetsPt30"]==4) & (arr["nBtagged_filtered"]==0)))
                                ) 
    
    extra_OSSF_pair_cond = (arr["LepPdgId_4"]!=0) & (arr["LepPdgId_4"]+arr["LepPdgId_5"]==0) 
    VH_leptonic_tagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & \
                                (
                                (arr["ZZCand_nExtraLep"]<=3) & (arr["nBtagged_filtered"]==0) & \
                                 ((arr["ZZCand_nExtraLep"]==1) | extra_OSSF_pair_cond)
                                ) | \
                                ((arr["ZZCand_nExtraLep"]>=1) & (arr["nCleanedJetsPt30"]==0))
    
    ttH_hadronic_tagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & ~VH_leptonic_tagged_mask & \
                                (arr["nCleanedJetsPt30"]>=4) & (arr["nBtagged_filtered"]>=1) & (arr["ZZCand_nExtraLep"]==0)
    
    ttH_leptonic_tagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & ~VH_leptonic_tagged_mask & ~ttH_hadronic_tagged_mask & \
                                (arr["ZZCand_nExtraLep"]>=1) 
    
    one_jet_tagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & ~VH_leptonic_tagged_mask & ~ttH_hadronic_tagged_mask & ~ttH_leptonic_tagged_mask & \
                                (arr["nCleanedJetsPt30"]==1) & (arr["ZZCand_nExtraLep"]==0) & (arr["DVBF1j_ME"]>0.7)
    Untagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & ~VH_leptonic_tagged_mask & ~ttH_hadronic_tagged_mask & ~ttH_leptonic_tagged_mask & ~one_jet_tagged_mask
    
    # Create arrays for each category based on the masks
    VBF_2jet_tagged_arr = {var: arr[var][VBF_2jet_tagged_mask] for var in arr.keys()}
    VH_hadronic_tagged_arr = {var: arr[var][VH_hadronic_tagged_mask] for var in arr.keys()}
    VH_leptonic_tagged_arr = {var: arr[var][VH_leptonic_tagged_mask] for var in arr.keys()}
    ttH_hadronic_tagged_arr = {var: arr[var][ttH_hadronic_tagged_mask] for var in arr.keys()}
    ttH_leptonic_tagged_arr = {var: arr[var][ttH_leptonic_tagged_mask] for var in arr.keys()}
    one_jet_tagged_arr = {var: arr[var][one_jet_tagged_mask] for var in arr.keys()}
    Untagged_arr = {var: arr[var][Untagged_mask] for var in arr.keys()}

    print(f"VBF_2jet_tagged: {np.sum(VBF_2jet_tagged_mask)} events")
    print(f"VH_hadronic_tagged: {np.sum(VH_hadronic_tagged_mask)} events")
    print(f"VH_leptonic_tagged: {np.sum(VH_leptonic_tagged_mask)} events")
    print(f"ttH_hadronic_tagged: {np.sum(ttH_hadronic_tagged_mask)} events")
    print(f"ttH_leptonic_tagged: {np.sum(ttH_leptonic_tagged_mask)} events")
    print(f"one_jet_tagged: {np.sum(one_jet_tagged_mask)} events")
    print(f"Untagged: {np.sum(Untagged_mask)} events")

    return {
        "VBF_2jet_tagged": VBF_2jet_tagged_arr,
        "VH_hadronic_tagged": VH_hadronic_tagged_arr,
        "VH_leptonic_tagged": VH_leptonic_tagged_arr,
        "ttH_hadronic_tagged": ttH_hadronic_tagged_arr,
        "ttH_leptonic_tagged": ttH_leptonic_tagged_arr,
        "one_jet_tagged": one_jet_tagged_arr,
        "Untagged": Untagged_arr
    }


def second_step_categorisation(arr_dict):
    """Perform second-step categorization.
    
    Args:
        arr_dict: the output dictionary from first_step_categorisation
        
    Returns:
        a dictionary of 22 arrays corresponding to the stage 1.2 HTXS categorization.
    """
    # Convert numeric columns to float for consistency
    numeric_vars = [
        "nCleanedJetsPt30", "ZZCand_pt", "m_jj", "ZZjj_pt"
    ]
    
    for category in arr_dict:
        for var in numeric_vars:
            if var in arr_dict[category]:
                try:
                    arr_dict[category][var] = arr_dict[category][var].astype(float)
                except (ValueError, TypeError) as e:
                    print(f"  Warning: Could not convert {category}/{var}: {e}")
    
    # mask definitions

    # Untagged subcategories
    Untagged_0j_Pt0To10_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==0) & (arr_dict["Untagged"]["ZZCand_pt"]<10)
    Untagged_0j_Pt10To200_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==0) & (arr_dict["Untagged"]["ZZCand_pt"]>=10) & (arr_dict["Untagged"]["ZZCand_pt"]<=200)
    Untagged_1j_Pt0To60_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==1) & (arr_dict["Untagged"]["ZZCand_pt"]<60)
    Untagged_1j_Pt60To120_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==1) & (arr_dict["Untagged"]["ZZCand_pt"]>=60) & (arr_dict["Untagged"]["ZZCand_pt"]<=120)
    Untagged_1j_Pt120To200_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==1) & (arr_dict["Untagged"]["ZZCand_pt"]>120) & (arr_dict["Untagged"]["ZZCand_pt"]<=200)
    Untagged_2j_Pt0To60_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==2) & (arr_dict["Untagged"]["ZZCand_pt"]<60) & (arr_dict["Untagged"]["m_jj"]<=350)
    Untagged_2j_Pt60To120_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==2) & (arr_dict["Untagged"]["ZZCand_pt"]>=60) & (arr_dict["Untagged"]["ZZCand_pt"]<=120) & (arr_dict["Untagged"]["m_jj"]<=350)
    Untagged_2j_Pt120To200_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==2) & (arr_dict["Untagged"]["ZZCand_pt"]>120) & (arr_dict["Untagged"]["ZZCand_pt"]<=200) & (arr_dict["Untagged"]["m_jj"]<=350)
    Untagged_Pt200above_mask = (arr_dict["Untagged"]["ZZCand_pt"]>200)
    Untagged_2j_mjj350above_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==2) & (arr_dict["Untagged"]["m_jj"]>350)

    # VBF 2-jet tagged subcategories
    VBF_2jet_tagged_mjj350To700_mask = (arr_dict["VBF_2jet_tagged"]["m_jj"]>=350) & (arr_dict["VBF_2jet_tagged"]["m_jj"]<700) & (arr_dict["VBF_2jet_tagged"]["ZZCand_pt"]<=200) & (arr_dict["VBF_2jet_tagged"]["ZZjj_pt"]<25) # need to include pt4ljj
    VBF_2jet_tagged_mjj700above_mask = (arr_dict["VBF_2jet_tagged"]["m_jj"]>=700) & (arr_dict["VBF_2jet_tagged"]["ZZCand_pt"]<=200) & (arr_dict["VBF_2jet_tagged"]["ZZjj_pt"]<25) # need to include pt4ljj
    VBF_3jet_tagged_mjj350above_mask = (arr_dict["VBF_2jet_tagged"]["m_jj"]>=350) & (arr_dict["VBF_2jet_tagged"]["ZZCand_pt"]<=200) & (arr_dict["VBF_2jet_tagged"]["nCleanedJetsPt30"]>=3) & (arr_dict["VBF_2jet_tagged"]["ZZjj_pt"]>25) # need to include pt4ljj 3jet? need to ask
    VBF_2jet_tagged_Pt200above_mask = (arr_dict["VBF_2jet_tagged"]["ZZCand_pt"]>200) & (arr_dict["VBF_2jet_tagged"]["m_jj"]>=350) # need to include pt4ljj
    VBF_rest = arr_dict["VBF_2jet_tagged"]["m_jj"]<350

    # VH hadronic tagged subcategories
    VH_hadronic_tagged_mjj60To120_mask = (arr_dict["VH_hadronic_tagged"]["m_jj"]>=60) & (arr_dict["VH_hadronic_tagged"]["m_jj"]<120)
    VH_hadronic_tagged_rest_mask = (arr_dict["VH_hadronic_tagged"]["m_jj"]<60) | (arr_dict["VH_hadronic_tagged"]["m_jj"]>=120)

    # VH leptonic tagged subcategories
    VH_leptonic_tagged_Pt0To150_mask = (arr_dict["VH_leptonic_tagged"]["ZZCand_pt"]<150)
    VH_leptonic_tagged_Pt150above_mask = (arr_dict["VH_leptonic_tagged"]["ZZCand_pt"]>=150)

    # Masks for events captured by defined stage-1.2 categories
    Untagged_categorised_mask = (
        Untagged_0j_Pt0To10_mask |
        Untagged_0j_Pt10To200_mask |
        Untagged_1j_Pt0To60_mask |
        Untagged_1j_Pt60To120_mask |
        Untagged_1j_Pt120To200_mask |
        Untagged_2j_Pt0To60_mask |
        Untagged_2j_Pt60To120_mask |
        Untagged_2j_Pt120To200_mask |
        Untagged_Pt200above_mask |
        Untagged_2j_mjj350above_mask
    )
    VBF_2jet_categorised_mask = (
        VBF_2jet_tagged_mjj350To700_mask |
        VBF_2jet_tagged_mjj700above_mask |
        VBF_3jet_tagged_mjj350above_mask |
        VBF_2jet_tagged_Pt200above_mask |
        VBF_rest
    )
    VH_hadronic_categorised_mask = VH_hadronic_tagged_mjj60To120_mask | VH_hadronic_tagged_rest_mask
    VH_leptonic_categorised_mask = VH_leptonic_tagged_Pt0To150_mask | VH_leptonic_tagged_Pt150above_mask

    # Leftover events not assigned to any category above
    Untagged_uncategorised_mask = ~Untagged_categorised_mask
    VBF_2jet_uncategorised_mask = ~VBF_2jet_categorised_mask
    VH_hadronic_uncategorised_mask = ~VH_hadronic_categorised_mask
    VH_leptonic_uncategorised_mask = ~VH_leptonic_categorised_mask

    

    # Create arrays for each category based on the masks
    Untagged_0j_Pt0To10_arr = {var: arr_dict["Untagged"][var][Untagged_0j_Pt0To10_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_0j_Pt10To200_arr = {var: arr_dict["Untagged"][var][Untagged_0j_Pt10To200_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_1j_Pt0To60_arr = {var: arr_dict["Untagged"][var][Untagged_1j_Pt0To60_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_1j_Pt60To120_arr = {var: arr_dict["Untagged"][var][Untagged_1j_Pt60To120_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_1j_Pt120To200_arr = {var: arr_dict["Untagged"][var][Untagged_1j_Pt120To200_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_2j_Pt0To60_arr = {var: arr_dict["Untagged"][var][Untagged_2j_Pt0To60_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_2j_Pt60To120_arr = {var: arr_dict["Untagged"][var][Untagged_2j_Pt60To120_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_2j_Pt120To200_arr = {var: arr_dict["Untagged"][var][Untagged_2j_Pt120To200_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_Pt200above_arr = {var: arr_dict["Untagged"][var][Untagged_Pt200above_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_2j_mjj350above_arr = {var: arr_dict["Untagged"][var][Untagged_2j_mjj350above_mask] for var in arr_dict["Untagged"].keys()}

    VBF_2jet_tagged_mjj350To700_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_2jet_tagged_mjj350To700_mask] for var in arr_dict["VBF_2jet_tagged"].keys()}
    VBF_2jet_tagged_mjj700above_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_2jet_tagged_mjj700above_mask] for var in arr_dict["VBF_2jet_tagged"].keys()}
    VBF_3jet_tagged_mjj350above_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_3jet_tagged_mjj350above_mask] for var in arr_dict["VBF_2jet_tagged"].keys()}
    VBF_2jet_tagged_Pt200above_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_2jet_tagged_Pt200above_mask] for var in arr_dict["VBF_2jet_tagged"].keys()}
    VBF_rest_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_rest] for var in arr_dict["VBF_2jet_tagged"].keys()}

    VH_hadronic_tagged_mjj60To120_arr = {var: arr_dict["VH_hadronic_tagged"][var][VH_hadronic_tagged_mjj60To120_mask] for var in arr_dict["VH_hadronic_tagged"].keys()}
    VH_hadronic_tagged_rest_arr = {var: arr_dict["VH_hadronic_tagged"][var][VH_hadronic_tagged_rest_mask] for var in arr_dict["VH_hadronic_tagged"].keys()}

    VH_leptonic_tagged_Pt0To150_arr = {var: arr_dict["VH_leptonic_tagged"][var][VH_leptonic_tagged_Pt0To150_mask] for var in arr_dict["VH_leptonic_tagged"].keys()}
    VH_leptonic_tagged_Pt150above_arr = {var: arr_dict["VH_leptonic_tagged"][var][VH_leptonic_tagged_Pt150above_mask] for var in arr_dict["VH_leptonic_tagged"].keys()}

    # Build uncategorised as leftovers across all split parents
    uncategorised_parts = []
    if np.any(Untagged_uncategorised_mask):
        uncategorised_parts.append({
            var: arr_dict["Untagged"][var][Untagged_uncategorised_mask]
            for var in arr_dict["Untagged"].keys()
        })
    if np.any(VBF_2jet_uncategorised_mask):
        uncategorised_parts.append({
            var: arr_dict["VBF_2jet_tagged"][var][VBF_2jet_uncategorised_mask]
            for var in arr_dict["VBF_2jet_tagged"].keys()
        })
    if np.any(VH_hadronic_uncategorised_mask):
        uncategorised_parts.append({
            var: arr_dict["VH_hadronic_tagged"][var][VH_hadronic_uncategorised_mask]
            for var in arr_dict["VH_hadronic_tagged"].keys()
        })
    if np.any(VH_leptonic_uncategorised_mask):
        uncategorised_parts.append({
            var: arr_dict["VH_leptonic_tagged"][var][VH_leptonic_uncategorised_mask]
            for var in arr_dict["VH_leptonic_tagged"].keys()
        })

    if uncategorised_parts:
        uncategorised_arr = {
            var: np.concatenate([part[var] for part in uncategorised_parts])
            for var in uncategorised_parts[0].keys()
        }
    else:
        uncategorised_arr = {
            var: arr_dict["Untagged"][var][:0]
            for var in arr_dict["Untagged"].keys()
        }

    print(f"uncategorised: {len(uncategorised_arr['mode_label'])} events")

    return {
        "Untagged_0j_Pt0To10": Untagged_0j_Pt0To10_arr,
        "Untagged_0j_Pt10To200": Untagged_0j_Pt10To200_arr,
        "Untagged_1j_Pt0To60": Untagged_1j_Pt0To60_arr,
        "Untagged_1j_Pt60To120": Untagged_1j_Pt60To120_arr,
        "Untagged_1j_Pt120To200": Untagged_1j_Pt120To200_arr,
        "Untagged_2j_Pt0To60": Untagged_2j_Pt0To60_arr,
        "Untagged_2j_Pt60To120": Untagged_2j_Pt60To120_arr,
        "Untagged_2j_Pt120To200": Untagged_2j_Pt120To200_arr,
        "Untagged_Pt200above": Untagged_Pt200above_arr,
        "Untagged_2j_mjj350above": Untagged_2j_mjj350above_arr,
        "VBF_1jet_tagged": arr_dict["one_jet_tagged"],
        "VBF_2jet_tagged_mjj350To700": VBF_2jet_tagged_mjj350To700_arr,
        "VBF_2jet_tagged_mjj700above": VBF_2jet_tagged_mjj700above_arr,
        "VBF_3jet_tagged_mjj350above": VBF_3jet_tagged_mjj350above_arr,
        "VBF_2jet_tagged_Pt200above": VBF_2jet_tagged_Pt200above_arr,
        "VBF_rest": VBF_rest_arr,
        "VH_hadronic_tagged_mjj60To120": VH_hadronic_tagged_mjj60To120_arr,
        "VH_hadronic_tagged_rest": VH_hadronic_tagged_rest_arr,
        "VH_leptonic_tagged_Pt0To150": VH_leptonic_tagged_Pt0To150_arr,
        "VH_leptonic_tagged_Pt150above": VH_leptonic_tagged_Pt150above_arr,
        "ttH_hadronic_tagged": arr_dict["ttH_hadronic_tagged"],
        "ttH_leptonic_tagged": arr_dict["ttH_leptonic_tagged"],
        "uncategorised": uncategorised_arr
    }


def compute_yield_table(arr_dict, event_weight=None):
    """Compute yields for each category and production mode.
    
    Args:
        arr_dict: Dictionary mapping category names to their variable arrays
        event_weight: The key for the variable to compute yields on. If None, count unweighted events.
    
    Returns:
        pandas DataFrame with categories as rows and production modes as columns
    """
    # Build the yield table
    yield_table = {}
    for category, arr in arr_dict.items():
        yield_table[category] = {}
        
        # Process production modes (keep individual)
        for mode in production_modes:
            mask = arr["mode_label"] == mode
            if event_weight is not None:
                yield_table[category][mode] = np.sum(arr[event_weight][mask])
            else:
                yield_table[category][mode] = np.sum(mask)
        
        # Process qqZZ modes (keep individual)
        for mode in qqZZ_bgmodes:
            mask = arr["mode_label"] == mode
            if event_weight is not None:
                yield_table[category][mode] = np.sum(arr[event_weight][mask])
            else:
                yield_table[category][mode] = np.sum(mask)
        
        # Group ggZZ modes
        ggZZ_mask = np.isin(arr["mode_label"], ggZZ_bgmodes)
        if event_weight is not None:
            yield_table[category]["ggZZ"] = np.sum(arr[event_weight][ggZZ_mask])
        else:
            yield_table[category]["ggZZ"] = np.sum(ggZZ_mask)
        
        # Group EWK modes
        EWK_mask = np.isin(arr["mode_label"], EWK_bgmodes)
        if event_weight is not None:
            yield_table[category]["EWK"] = np.sum(arr[event_weight][EWK_mask])
        else:
            yield_table[category]["EWK"] = np.sum(EWK_mask)
        
        # Group TTjets modes
        TTjets_mask = np.isin(arr["mode_label"], TTjets_bgmodes)
        if event_weight is not None:
            yield_table[category]["TTjets"] = np.sum(arr[event_weight][TTjets_mask])
        else:
            yield_table[category]["TTjets"] = np.sum(TTjets_mask)
    
    # Convert to DataFrame
    df = pd.DataFrame(yield_table).T
    df = df.fillna(0)  # Fill any missing entries with 0
    
    # Reorder columns: production modes, qqZZ, then grouped backgrounds
    column_order = production_modes + qqZZ_bgmodes + ["ggZZ", "EWK", "TTjets"]
    column_order = [c for c in column_order if c in df.columns]  # Only include columns that exist

    df = df[column_order]
    
    # Calculate signal-to-background ratio
    signal_columns = production_modes
    background_columns = qqZZ_bgmodes + ["ggZZ", "EWK", "TTjets"]
    
    signal_sum = df[signal_columns].sum(axis=1)
    background_sum = df[background_columns].sum(axis=1)
    
    # Avoid division by zero
    df["S/B"] = signal_sum / background_sum.replace(0, np.inf)
    df["S/B"] = df["S/B"].replace(np.inf, 0)  # Replace inf with 0 where background is 0
    
    return df


def save_arrays_to_root(arr_dict, output_path, tree_name="Events"):
    """Save categorized arrays to ROOT files.
    
    Args:
        arr_dict: Dictionary mapping category names to their variable arrays
        output_path: Directory to save the ROOT files
        tree_name: Name of the TTree to create
    """
    import ROOT
    from array import array
    
    for category, arr in arr_dict.items():
        # Create output ROOT file
        root_file = ROOT.TFile(f"{output_path}/{category}.root", "RECREATE")
        tree = ROOT.TTree(tree_name, tree_name)
        
        # Get number of events
        first_key = list(arr.keys())[0]
        n_events = len(arr[first_key])
        arr["category"] = np.array([category]*n_events, dtype='S')  # Add category variable as string
        # Create branches for each variable
        branch_vars = {}
        string_vars = {}
        for var_name, var_data in arr.items():
            # Handle string variables (mode_label, category)
            if var_data.dtype.kind in ['O', 'S', 'U']:
                # Store strings using ROOT's string support
                string_vars[var_name] = ROOT.std.string()
                tree.Branch(var_name, string_vars[var_name])
            else:
                # Numeric variables
                branch_vars[var_name] = array('f', [0.0])  # float array
                tree.Branch(var_name, branch_vars[var_name], f"{var_name}/F")
        
        # Fill the tree
        for i in range(n_events):
            # Fill numeric variables
            for var_name in branch_vars.keys():
                branch_vars[var_name][0] = float(arr[var_name][i])
            # Fill string variables
            for var_name in string_vars.keys():
                str_val = arr[var_name][i]
                # Convert bytes to string if necessary
                if isinstance(str_val, bytes):
                    str_val = str_val.decode('utf-8')
                elif not isinstance(str_val, str):
                    str_val = str(str_val)
                # Clear and assign the string value
                string_vars[var_name].clear()
                string_vars[var_name].append(str_val)
            tree.Fill()
        
        # Write and close
        tree.Write()
        root_file.Close()


def save_all_categories_to_single_root(arr_dict, output_path, tree_name="Events"):
    """Save all categorized arrays to a single ROOT file with category labels.
    
    Args:
        arr_dict: Dictionary mapping category names to their variable arrays
        output_path: Path to output ROOT file (should end in .root)
        tree_name: Name of the TTree to create
    """
    import ROOT
    from array import array
    
    # First, add category labels and merge all arrays
    all_vars = None
    for category, arr in arr_dict.items():
        # Get number of events in this category
        first_key = [k for k in arr.keys() if k != 'category'][0]
        n_events = len(arr[first_key])
        
        # Add category label as a string array
        arr_with_cat = arr.copy()
        arr_with_cat["category"] = np.array([category]*n_events, dtype='U100')
        
        # Merge with previous categories
        if all_vars is None:
            all_vars = arr_with_cat
        else:
            # Concatenate each variable
            for var_name in all_vars.keys():
                if var_name in arr_with_cat:
                    all_vars[var_name] = np.concatenate([all_vars[var_name], arr_with_cat[var_name]])
    
    # Create output ROOT file
    root_file = ROOT.TFile(output_path, "RECREATE")
    tree = ROOT.TTree(tree_name, tree_name)
    
    # Create branches for each variable
    branch_vars = {}
    string_vars = {}
    n_total_events = len(all_vars['category'])
    
    for var_name, var_data in all_vars.items():
        # Handle string variables (mode_label, category)
        if var_data.dtype.kind in ['O', 'S', 'U']:
            string_vars[var_name] = ROOT.std.string()
            tree.Branch(var_name, string_vars[var_name])
        else:
            # Numeric variables
            branch_vars[var_name] = array('f', [0.0])
            tree.Branch(var_name, branch_vars[var_name], f"{var_name}/F")
    
    # Fill the tree
    for i in range(n_total_events):
        # Fill numeric variables
        for var_name in branch_vars.keys():
            branch_vars[var_name][0] = float(all_vars[var_name][i])
        
        # Fill string variables
        for var_name in string_vars.keys():
            str_val = all_vars[var_name][i]
            if isinstance(str_val, bytes):
                str_val = str_val.decode('utf-8')
            elif not isinstance(str_val, str):
                str_val = str(str_val)
            string_vars[var_name].clear()
            string_vars[var_name].append(str_val)
        
        tree.Fill()
    
    # Write and close
    tree.Write()
    root_file.Close()
    print(f"  ✓ Saved {n_total_events:,} events to single ROOT file: {output_path}")


def print_and_save_yield_table(yield_table, output_path, title, filename):
    """Print and save a yield table to CSV.
    
    Args:
        yield_table: pandas DataFrame with yields
        output_path: Directory to save the CSV file
        title: Title to print above the table
        filename: Name of the CSV file to save
    """
    print("\n" + "="*100)
    print(title)
    print("="*100)
    print(yield_table.to_string())
    print("="*100 + "\n")
    yield_table.to_csv(f"{output_path}/{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTXS categorization")
    parser.add_argument("--input_path", type=str, default=None, help="Path to input ROOT files (single-year mode)")
    parser.add_argument("--input_paths", type=str, default=None, help="Comma-separated input paths for multi-year mode")
    parser.add_argument("--years", type=str, default=None, help="Comma-separated year labels matching --input_paths")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output directory")
    parser.add_argument("--tree_name", type=str, default="ZZTree/candTree", help="Name of the TTree in ROOT files")
    parser.add_argument("--mode", type=str, default="production_modes", help="Production mode to process (default: production modes)")
    parser.add_argument("--run2scale", type=bool, default=False, help="Apply Run 3 to Run 2 cross-section scaling factors")
    args = parser.parse_args()

    if args.mode == "production_modes":
        modes = production_modes
    elif args.mode == "all_modes":
        modes = all_modes
    elif args.mode == "prduction+qqZZ+ggZZ":
        modes = production_modes + qqZZ_bgmodes + ggZZ_bgmodes

    start_time = time.time()
    print("\n" + "="*100)
    print("Starting STXS Categorization Pipeline")
    print("="*100)
    input_paths = parse_list_arg(args.input_paths)
    if not input_paths:
        if args.input_path is None:
            raise ValueError("Provide either --input_path (single year) or --input_paths (multi-year).")
        input_paths = [args.input_path]

    year_labels = parse_list_arg(args.years)
    if year_labels and len(year_labels) != len(input_paths):
        raise ValueError("When provided, --years must have the same number of entries as --input_paths.")
    if not year_labels:
        year_labels = [infer_year_label(path_value) for path_value in input_paths]

    print(f"Input paths: {input_paths}")
    print(f"Years:       {year_labels}")
    print(f"Output path: {args.output_path}")
    print(f"Tree name:   {args.tree_name}")
    print(f"Modes:       {args.mode} ({len(modes)} datasets)")
    print("="*100 + "\n")

    # Load data from ROOT files
    print("[1/6] Loading ROOT data from disk...")
    load_start = time.time()
    arr_dict = load_arrays_from_multiple_years(
        input_paths=input_paths,
        year_labels=year_labels,
        name_tree=args.tree_name,
        modes=modes,
        run2scale=args.run2scale,
    )
    load_time = time.time() - load_start
    total_events = sum(len(arr_dict[mode].get(list(arr_dict[mode].keys())[0])) for mode in modes)
    print(f"✓ Loaded {len(modes)} datasets across {len(input_paths)} year(s) with {total_events:,} total events ({load_time:.2f}s)\n")

    # Merge all modes into a single dataset
    print("[2/6] Merging datasets...")
    merge_start = time.time()
    merged_arr = merge_modes(arr_dict, modes)
    merge_time = time.time() - merge_start
    print(f"✓ Merged into single array ({merge_time:.2f}s)")
    print(f"  - Total events: {len(merged_arr['mode_label']):,}\n")

    # First-step categorization
    print("[3/6] Performing first-step (stage 0) categorization...")
    stage0_start = time.time()
    first_step_arr_dict = first_step_categorisation(merged_arr)
    stage0_time = time.time() - stage0_start
    stage0_total = sum(len(arr['mode_label']) for arr in first_step_arr_dict.values())
    print(f"✓ Stage 0 categorization complete ({stage0_time:.2f}s)")
    print(f"  - Categories created: {len(first_step_arr_dict)}")
    print(f"  - Total events in categories: {stage0_total:,}\n")

    # Second-step categorization
    print("[4/6] Performing second-step (stage 1.2) categorization...")
    stage1p2_start = time.time()
    second_step_arr_dict = second_step_categorisation(first_step_arr_dict)
    stage1p2_time = time.time() - stage1p2_start
    stage1p2_total = sum(len(arr['mode_label']) for arr in second_step_arr_dict.values())
    print(f"✓ Stage 1.2 categorization complete ({stage1p2_time:.2f}s)")
    print(f"  - Categories created: {len(second_step_arr_dict)}")
    print(f"  - Total events in categories: {stage1p2_total:,}\n")

    # Save categorized arrays to output directory
    print("[5/6] Saving categorized arrays...")
    save_start = time.time()
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(f"{args.output_path}/stage0", exist_ok=True)
    os.makedirs(f"{args.output_path}/stage1p2", exist_ok=True)
    
    saved_count = 0
    print("  Saving numpy arrays...")
    for category, arr in second_step_arr_dict.items():
        np.savez_compressed(f"{args.output_path}/stage1p2/{category}_data.npz", **arr)
        saved_count += 1
    for category, arr in first_step_arr_dict.items():
        np.savez_compressed(f"{args.output_path}/stage0/{category}_data.npz", **arr)
        saved_count += 1
    print(f"  ✓ Saved {saved_count} numpy arrays")
    
    print("  Saving ROOT files...")
    root_saved = 0
    save_arrays_to_root(second_step_arr_dict, f"{args.output_path}/stage1p2", tree_name=args.tree_name)
    root_saved += len(second_step_arr_dict)
    save_arrays_to_root(first_step_arr_dict, f"{args.output_path}/stage0", tree_name=args.tree_name)
    root_saved += len(first_step_arr_dict)
    print(f"  ✓ Saved {root_saved} ROOT files")
    
    print("  Saving combined ROOT files...")
    save_all_categories_to_single_root(
        second_step_arr_dict, 
        f"{args.output_path}/stage1p2_combined.root", 
        tree_name=args.tree_name
    )
    
    save_time = time.time() - save_start
    print(f"✓ Total: {saved_count} numpy + {root_saved} ROOT files ({save_time:.2f}s)\n")

    # Compute and display weighted yield tables
    print("[6/6] Computing yield tables...")
    yield_start = time.time()
    
    print("\n  Stage 0 yields:")
    yield_stage0_18 = compute_yield_table(first_step_arr_dict, event_weight="EventWeight_lumi138")
    print_and_save_yield_table(yield_stage0_18, args.output_path, 
                               "WEIGHTED YIELD TABLE (Stage 0 STXS Categorization lumi18fb)",
                               "yield_stage0_lumi18.csv")
    yield_stage0_9 = compute_yield_table(first_step_arr_dict, event_weight="EventWeight_lumi9")
    print_and_save_yield_table(yield_stage0_9, args.output_path, 
                               "WEIGHTED YIELD TABLE (Stage 0 STXS Categorization lumi9fb)",
                               "yield_stage0_lumi9.csv")

    print("\n  Stage 1.2 yields:")
    yield_stage1p2_18 = compute_yield_table(second_step_arr_dict, event_weight="EventWeight_lumi138")
    print_and_save_yield_table(yield_stage1p2_18, args.output_path, 
                               "WEIGHTED YIELD TABLE (Stage 1.2 STXS Categorization lumi18fb)",
                               "yield_stage1p2_lumi18.csv")
    
    yield_stage1p2_9 = compute_yield_table(second_step_arr_dict, event_weight="EventWeight_lumi9")
    print_and_save_yield_table(yield_stage1p2_9, args.output_path, 
                               "WEIGHTED YIELD TABLE (Stage 1.2 STXS Categorization lumi9fb)",
                               "yield_stage1p2_lumi9.csv")

    yield_time = time.time() - yield_start
    print(f"\n✓ Yield tables computed and saved ({yield_time:.2f}s)\n")

    end_time = time.time()
    total_time = end_time - start_time
    print("="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"  - Data loading:  {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
    print(f"  - Merging:       {merge_time:.2f}s ({merge_time/total_time*100:.1f}%)")
    print(f"  - Stage 0 cat:   {stage0_time:.2f}s ({stage0_time/total_time*100:.1f}%)")
    print(f"  - Stage 1.2 cat: {stage1p2_time:.2f}s ({stage1p2_time/total_time*100:.1f}%)")
    print(f"  - Saving:        {save_time:.2f}s ({save_time/total_time*100:.1f}%)")
    print(f"  - Yields:        {yield_time:.2f}s ({yield_time/total_time*100:.1f}%)")
    print("="*100)
    print(f"✓ Categorization completed successfully!")
    print(f"✓ Output saved to: {args.output_path}")
    print("="*100 + "\n")