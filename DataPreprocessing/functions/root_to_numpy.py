"""
Script to load ROOT data, convert to numpy arrays, and save to disk.

python ${HZZ_ROOT}/DataPreprocessing/functions/root_to_numpy.py \
    --file_path /eos/user/z/zhiheng/STXS_samples/2023samples/preBPix/root_data \
    --output_dir /eos/user/z/zhiheng/STXS_samples/2023samples/preBPix/numpy_data \
    --tree_name Events
"""
import argparse
import os
import numpy as np
import ROOT
from pathlib import Path

# Feature definitions from plotter.py
non_lepton_features = [
    "ZZCand_pt", "ZZCand_eta", "ZZCand_phi", "ZZCand_mass",
    "ZZCand_costheta1", "ZZCand_costheta2", "ZZCand_costhetastar", "ZZCand_Phi1",
    "ZZCand_nExtraLep", "ZZjj_pt",
    "PFMET_pt",
    "JetLeading_pt", "JetLeading_eta", "JetLeading_mass", "JetLeading_phi",
    "JetSubLeading_pt", "JetSubLeading_eta", "JetSubLeading_mass", "JetSubLeading_phi",
    "nCleanedJetsPt30", "nBtagged_filtered", "JetLeading_btag", "JetSubLeading_btag",
    "deltaEta_jj", "deltaPhi_jj", "m_jj",
    "DVBF2j_ME", "DVBF1j_ME", "DWHh_ME", "DZHh_ME", "DVBF2j_ME_noC", "DVBF1j_ME_noC", "DWHh_ME_noC", "DZHh_ME_noC"
]

lepton_features = [
    "LepPt_0", "LepPt_1", "LepPt_2", "LepPt_3",
    "LepEta_0", "LepEta_1", "LepEta_2", "LepEta_3",
    "LepPhi_0", "LepPhi_1", "LepPhi_2", "LepPhi_3",
    "LepPdgId_0", "LepPdgId_1", "LepPdgId_2", "LepPdgId_3",
    "LepPt_4", "LepPt_5",
    "LepEta_4", "LepEta_5",
    "LepPhi_4", "LepPhi_5",
    "LepPdgId_4", "LepPdgId_5",
]

labels_and_weights = ["EventWeight_lumi18", "EventWeight_lumi9", "EventWeight_lumi138", "HTXS_stage_0", "overallEventWeight", "genWeight"]

all_features = non_lepton_features + lepton_features + labels_and_weights

production_modes = ['ggH125', 'VBFH125', 'WplusH125', 'WminusH125', 'ZH125', 'ttH125']
qqZZ_bgmodes = ['ZZTo4l']
ggZZ_bgmodes = ['ggTo2e2mu_Contin_MCFM701', 'ggTo4e_Contin_MCFM701', 'ggTo4mu_Contin_MCFM701', 'ggTo4tau_Contin_MCFM701', 'ggTo2mu2tau_Contin_MCFM701', 'ggTo2e2tau_Contin_MCFM701']
EWK_bgmodes = ['WZZ', 'WWZ', 'ZZZ']
TTjets_bgmodes = ['TTZZ', 'TTWW']
all_modes = production_modes + qqZZ_bgmodes + ggZZ_bgmodes + EWK_bgmodes + TTjets_bgmodes


def rdf_To_numpy(rdf, mode_label, variables=all_features, label="HTXS_stage_0"):
    """Convert specified variables from a ROOT RDataFrame to a dictionary of numpy arrays.
    
    Args:
        rdf: ROOT RDataFrame containing the data
        variables: List of variable names to convert
        mode_label: String label for the production mode
        label: Name of the target label variable
    Returns:
        Dictionary mapping variable names to numpy arrays
    """
    print(f"[debug] Defining mode_label='{mode_label}' and converting to numpy for variables: {len(variables)} + label + mode_label")
    
    # Get available columns in the RDataFrame
    available_columns = rdf.GetColumnNames()
    print(f"[debug] Available columns in ROOT file: {len(available_columns)} columns")
    
    # Filter variables to only those that exist
    existing_variables = [v for v in variables if v in available_columns]
    missing_variables = [v for v in variables if v not in available_columns]
    
    if missing_variables:
        print(f"[warning] Missing columns: {missing_variables}")
        print(f"[warning] Will proceed with {len(existing_variables)} available columns")
    
    rdf = rdf.Define('mode_label', f'"{mode_label}"')
    result = rdf.AsNumpy(existing_variables + [label] + ['mode_label'])
    print(f"[debug] Converted mode '{mode_label}' to numpy with {len(next(iter(result.values())))} events")
    return result


def load_arrays_from_root(file_path, name_tree, modes):
    """Load ROOT data from multiple production modes into numpy arrays.
    
    Args:
        file_path: Base path to ROOT files
        name_tree: Name of the TTree to read
        modes: List of production mode names
        
    Returns:
        Dictionary mapping mode names to their numpy array data
    """
    rdf_dict = {}
    arr_dict = {}
    for mode in modes:
        path = f"{file_path}/{mode}.root"
        print(f"[debug] Loading mode '{mode}' from {path}")
        try:
            rdf = ROOT.RDataFrame(name_tree, path)
            entries = rdf.Count().GetValue()
            print(f"[debug] RDataFrame entries for '{mode}': {entries}")
            rdf_dict[mode] = rdf
            mode_label = (
                mode if mode in production_modes + qqZZ_bgmodes
                else "ggZZ" if mode in ggZZ_bgmodes
                else "EWK" if mode in EWK_bgmodes
                else "TTjets" if mode in TTjets_bgmodes
                else "Unknown"
            )
            arr_dict[mode] = rdf_To_numpy(rdf, mode_label=mode_label)
        except Exception as e:
            print(f"[error] Failed to process mode '{mode}': {str(e)}")
            print(f"[warning] Skipping mode '{mode}'")
            continue
    return arr_dict


def save_arrays_to_numpy(arr_dict, output_dir):
    """Save numpy arrays to disk as .npy files.
    
    Args:
        arr_dict: Dictionary mapping mode names to their variable arrays
        output_dir: Directory to save the numpy files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for mode, data_dict in arr_dict.items():
        mode_output_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_output_dir, exist_ok=True)
        
        print(f"[info] Saving mode '{mode}' to {mode_output_dir}")
        for var_name, var_data in data_dict.items():
            output_file = os.path.join(mode_output_dir, f"{var_name}.npy")
            np.save(output_file, var_data)
            print(f"[info] Saved {var_name} ({len(var_data)} events) to {output_file}")


def load_arrays_from_numpy(input_dir):
    """Load numpy arrays from disk back into the original dictionary structure.
    
    Args:
        input_dir: Directory containing the saved numpy files organized by mode
        
    Returns:
        Dictionary mapping mode names to their variable arrays (arr_dict[mode][var_name])
    """
    arr_dict = {}
    
    if not os.path.isdir(input_dir):
        print(f"[error] Input directory does not exist: {input_dir}")
        return arr_dict
    
    for mode in os.listdir(input_dir):
        mode_dir = os.path.join(input_dir, mode)
        
        if not os.path.isdir(mode_dir):
            continue
        
        print(f"[info] Loading mode '{mode}' from {mode_dir}")
        arr_dict[mode] = {}
        
        for var_file in os.listdir(mode_dir):
            if var_file.endswith('.npy'):
                var_name = var_file[:-4]  # Remove .npy extension
                var_path = os.path.join(mode_dir, var_file)
                arr_dict[mode][var_name] = np.load(var_path, allow_pickle=True)
                print(f"[info] Loaded {var_name} ({len(arr_dict[mode][var_name])} events) from {var_path}")
    
    total_modes = len(arr_dict)
    print(f"[info] Loaded {total_modes} modes")
    return arr_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ROOT data, convert to numpy, and save to disk.")
    parser.add_argument('--file_path', '-f', type=str, required=True, help='Base path to the ROOT files')
    parser.add_argument('--output_dir', '-o', type=str, default='numpy_data', help='Output directory for numpy files')
    parser.add_argument('--tree_name', '-t', type=str, default='Events', help='Name of the TTree to read')
    parser.add_argument('--modes', '-m', type=str, nargs='+', default=all_modes, help='List of production modes to process')
    args = parser.parse_args()

    print(f"[info] Loading data from {args.file_path}")
    print(f"[info] Processing modes: {args.modes}")
    
    # Load ROOT arrays
    arr_dict = load_arrays_from_root(args.file_path, args.tree_name, args.modes)
    
    # Save to numpy
    print(f"[info] Saving numpy arrays to {args.output_dir}")
    save_arrays_to_numpy(arr_dict, args.output_dir)
    
    print("[info] Done!")
