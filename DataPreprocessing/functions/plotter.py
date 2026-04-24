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

non_lepton_features = [
    # ZZ candidate features
    "ZZCand_pt", "ZZCand_eta", "ZZCand_phi", "ZZCand_mass",
    "ZZCand_costheta1", "ZZCand_costheta2", "ZZCand_costhetastar", "ZZCand_Phi1",
    "ZZCand_nExtraLep", "ZZjj_pt",
    # MET
    "PFMET_pt",
    # Jet features
    "JetLeading_pt", "JetLeading_eta", "JetLeading_mass", "JetLeading_phi",
    "JetSubLeading_pt", "JetSubLeading_eta", "JetSubLeading_mass", "JetSubLeading_phi",
    "nCleanedJetsPt30", "nBtagged_filtered", "JetLeading_btag",
    "JetSubLeading_btag",
    # Dijet features
    "deltaEta_jj", "deltaPhi_jj", "m_jj",
    # Discriminants
    "DVBF2j_ME", "DVBF1j_ME", "DWHh_ME", "DZHh_ME", "DVBF2j_ME_noC", "DVBF1j_ME_noC", "DWHh_ME_noC", "DZHh_ME_noC"
]
lepton_features = [
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
]
all_features = non_lepton_features + lepton_features + ["EventWeight_lumi138", "EventWeight_lumi9"]
range_dict = {
    # ZZ candidate features
    "ZZCand_pt": (0, 1000),
    "ZZCand_eta": (-5, 5),
    "ZZCand_phi": (-np.pi, np.pi),
    "ZZCand_mass": (117,131),
    "ZZCand_costheta1": (-1, 1),
    "ZZCand_costheta2": (-1, 1),
    "ZZCand_costhetastar": (-1, 1),
    "ZZCand_Phi1": (-np.pi, np.pi),
    "ZZCand_nExtraLep": (0, 4),
    "ZZjj_pt": (0, 500),
    # MET
    "PFMET_pt": (0, 250),
    # Jet features
    "JetLeading_pt": (0, 500),
    "JetLeading_eta": (-5, 5),
    "JetLeading_mass": (0, 200),
    "JetLeading_phi": (-3.5, 3.5),
    "JetSubLeading_pt": (0, 500),
    "JetSubLeading_eta": (-5, 5),
    "JetSubLeading_mass": (0, 200),
    "JetSubLeading_phi": (-3.5, 3.5),
    "nCleanedJetsPt30": (0, 10),
    "nBtagged_filtered": (0, 5),
    "JetLeading_btag": (0, 1),
    "JetSubLeading_btag": (0, 1),  
    # Dijet features
    "deltaEta_jj": (0, 10),
    "deltaPhi_jj": (0, 3.5),
    "m_jj": (0, 1500),
    # Lepton features
    "LepPt": (0, 150),
    "LepEta": (-2.5, 2.5),
    "LepPhi": (-np.pi, np.pi),
    "LepPdgId": (-20, 20),
    # Discriminants
    "DVBF2j_ME": (0, 1),
    "DVBF1j_ME": (0, 1),
    "DWHh_ME": (0, 1),
    "DZHh_ME": (0, 1),
    "DVBF2j_ME_noC": (0, 1),
    "DVBF1j_ME_noC": (0, 1),
    "DWHh_ME_noC": (0, 1),
    "DZHh_ME_noC": (0, 1),
}

logscale_dict = {
    # ZZ candidate features
    "ZZCand_pt": True,
    "ZZCand_eta": False,
    "ZZCand_phi": False,
    "ZZCand_mass": False,
    "ZZCand_costheta1": False,
    "ZZCand_costheta2": False,
    "ZZCand_costhetastar": False,
    "ZZCand_Phi1": False,
    "ZZCand_nExtraLep": False,
    "ZZjj_pt": True,
    # MET
    "PFMET_pt": True,
    # Jet features
    "JetLeading_pt": True,
    "JetLeading_eta": False,
    "JetLeading_mass": True,
    "JetLeading_phi": False,
    "JetSubLeading_pt": True,
    "JetSubLeading_eta": False,
    "JetSubLeading_mass": True,
    "JetSubLeading_phi": False,
    "nCleanedJetsPt30": False,
    "nBtagged_filtered": False,
    "JetLeading_btag": True,
    "JetSubLeading_btag": True,
    # Dijet features
    "deltaEta_jj": False,
    "deltaPhi_jj": False,
    "m_jj": True,
    # Lepton features
    "LepPt": False,
    "LepEta": False,
    "LepPhi": False,
    "LepPdgId": False,
    # Discriminants
    "DVBF2j_ME": False,
    "DVBF1j_ME": False,
    "DWHh_ME": True,
    "DZHh_ME": True,
    "DVBF2j_ME_noC": False,
    "DVBF1j_ME_noC": False,
    "DWHh_ME_noC": True,
    "DZHh_ME_noC": True,
}

bin_dict = {
    # ZZ candidate features
    "ZZCand_pt": 100,
    "ZZCand_eta": 50,
    "ZZCand_phi": 20,
    "ZZCand_mass": 100,
    "ZZCand_costheta1": 30,
    "ZZCand_costheta2": 30,
    "ZZCand_costhetastar": 30,
    "ZZCand_Phi1": 20,
    "ZZCand_nExtraLep": 4,
    "ZZjj_pt": 100,
    # MET   
    "PFMET_pt": 50,
    # Jet features
    "JetLeading_pt": 100,
    "JetLeading_eta": 50,
    "JetLeading_mass": 50,
    "JetLeading_phi": 20,
    "JetSubLeading_pt": 100,
    "JetSubLeading_eta": 50,
    "JetSubLeading_mass": 50,
    "JetSubLeading_phi": 20,
    "nCleanedJetsPt30": 10,
    "nBtagged_filtered": 6,
    "JetLeading_btag": 20,
    "JetSubLeading_btag": 20,
    # Dijet features
    "deltaEta_jj": 50,
    "deltaPhi_jj": 30,
    "m_jj": 100,
    # Lepton features
    "LepPt": 50,
    "LepEta": 50,
    "LepPhi": 20,
    "LepPdgId": 40,
    # Discriminants
    "DVBF2j_ME": 50,
    "DVBF1j_ME": 50,
    "DWHh_ME": 50,
    "DZHh_ME": 50,
    "DVBF2j_ME_noC": 50,
    "DVBF1j_ME_noC": 50,
    "DWHh_ME_noC": 50,
    "DZHh_ME_noC": 50,
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

    print(f"[debug] Defining mode_label='{mode_label}' and converting to numpy for variables: {len(variables)} + label + mode_label")
    rdf = rdf.Define('mode_label', f'"{mode_label}"')
    result = rdf.AsNumpy(variables + [label] + ['mode_label'])
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
    return arr_dict

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

def merge_modes(arr_dict, modes, variables=all_features + ["HTXS_stage_0"]):
    """Merge arrays from multiple production modes into a single combined dataset.
    
    Args:
        arr_dict: Dictionary mapping mode names to their variable arrays
        modes: List of mode names to merge together
        variables: List of variable names to merge
    
    Returns:
        Dictionary containing concatenated arrays for each variable
    """
    print(f"[debug] Merging modes {modes} for variables: {len(variables)}")
    merged = {}
    for var in variables + ["mode_label"]:
        merged[var] = np.concatenate([arr_dict[mode][var] for mode in modes])
    total = len(next(iter(merged.values())))
    print(f"[debug] Merged total events: {total}")
    return merged

def plot_vars_hist(
    arr_dict,
    variable,
    plot_modes,
    event_weight,
    range_dict,
    logscale_dict,
    output_dir,
    bin_dict=bin_dict,
    density=True,
    histtype='step',
    linewidth=2,
    filename_suffix="",
):
    os.makedirs(output_dir, exist_ok=True)
    data_list = []
    weights_list = []
    for mode in plot_modes:
        mask = arr_dict[mode][variable] != -999
        data_list.append(arr_dict[mode][variable][mask])
        weights_list.append(arr_dict[mode][event_weight][mask])
    plt.hist(
        data_list,
        bins=bin_dict[variable],
        weights=weights_list,
        density=density,
        histtype=histtype,
        linewidth=linewidth,
        range=range_dict[variable],
        label=plot_modes,
    )
    plt.xlabel(variable)
    plt.ylabel("a.u.")
    plt.xlim(range_dict[variable])
    if logscale_dict.get(variable, False):
        plt.yscale('log')
    plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1))
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    plt.savefig(f"{output_dir}/{variable}{suffix}.pdf", bbox_inches='tight')
    plt.close()

def plot_leptons(
    arr_dict,
    plot_modes,
    event_weight,
    range_dict,
    logscale_dict,
    output_dir,
    variables = ['LepPt', 'LepEta', 'LepPhi'],
    density=True,
    histtype='step',
    linewidth=2,
    filename_suffix="",
):
    for variable in variables:
        # Special handling for LepPt, LepEta, LepPhi: plot pairs (0,1), (2,3), (4,5)
        pairs = [(0, 1), (2, 3), (4, 5)]
        for a_idx, b_idx in pairs:
            os.makedirs(output_dir, exist_ok=True)
            series_data = []
            series_weights = []
            labels = []
            for idx in (a_idx, b_idx):
                for mode in plot_modes:
                    mask = arr_dict[mode][f"{variable}_{idx}"] != -999
                    series_data.append(arr_dict[mode][f"{variable}_{idx}"][mask])
                    series_weights.append(arr_dict[mode][event_weight][mask])
                    labels.append(f"{variable}_{idx} ({mode})")

            plt.hist(
                series_data,
                bins=bin_dict[variable],
                weights=series_weights,
                density=density,
                histtype=histtype,
                linewidth=linewidth,
                range=range_dict[variable],
                label=labels,
            )
            plt.xlabel(variable)
            plt.ylabel("a.u.")
            plt.xlim(range_dict[variable])
            if logscale_dict.get(variable, False):
                plt.yscale('log')
            plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1))
            suffix = f"_{filename_suffix}" if filename_suffix else ""
            plt.savefig(f"{output_dir}/{variable}_{a_idx}{b_idx}{suffix}.pdf", bbox_inches='tight')
            plt.close()
            
def _to_numeric(arr):
    """Coerce a per-event array to a 1-D float64 numpy array.

    Handles three storage shapes produced by our ROOT->numpy conversion:
      - numeric dtype: returned cast to float64.
      - object dtype wrapping 1-element ndarrays (e.g. ZZCand_pt): unwrapped.
      - object dtype of length-1 byte/str chars (e.g. nCleanedJetsPt30): ord().
    Returns (values, valid_mask) of the same length as `arr`.
    """
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float64), np.ones(len(arr), dtype=bool)
    out = np.empty(len(arr), dtype=np.float64)
    valid = np.zeros(len(arr), dtype=bool)
    for i, x in enumerate(arr):
        try:
            if isinstance(x, np.ndarray):
                if x.size == 0:
                    continue
                out[i] = float(x.ravel()[0])
            elif isinstance(x, (bytes, str)):
                if len(x) == 0:
                    continue
                out[i] = float(ord(x[0]))
            else:
                out[i] = float(x)
            valid[i] = True
        except (TypeError, ValueError):
            pass
    return out, valid

def _resolve_feature_meta(feature, range_dict, bin_dict, logscale_dict, data=None):
    """Resolve (key, range, nbins, logy) for a feature, stripping trailing _N for leptons."""
    key = feature
    lepton_prefixes = ("LepPt", "LepEta", "LepPhi", "LepPdgId")
    if "_" in feature:
        head, tail = feature.rsplit("_", 1)
        if head in lepton_prefixes and tail.isdigit():
            key = head
    rng = range_dict.get(key)
    if rng is None and data is not None and len(data) > 0:
        rng = (float(np.min(data)), float(np.max(data)))
    nbins = bin_dict.get(key, 50)
    logy = logscale_dict.get(key, False)
    return key, rng, nbins, logy

def plot_weight_effect_grid(
    arr_dict,
    mode,
    features,
    event_weight,
    range_dict,
    bin_dict,
    logscale_dict,
    output_dir,
    filename_suffix="",
):
    """For one mode, plot every feature as a subplot comparing all vs positive-weight-only events.

    Grid layout: 4 features per row. Each slot has a histogram panel (top) and a
    ratio panel (bottom, positive-only / all). One figure saved per call.
    """
    import matplotlib.gridspec as gridspec

    os.makedirs(output_dir, exist_ok=True)
    ncols = 4
    nfeat = len(features)
    nrows = (nfeat + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    outer = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.3)

    weights_full = arr_dict[mode][event_weight]

    for i, feature in enumerate(features):
        r, c = divmod(i, ncols)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[r, c], height_ratios=[2, 1], hspace=0.05
        )
        ax_hist = fig.add_subplot(inner[0])
        ax_ratio = fig.add_subplot(inner[1], sharex=ax_hist)

        values_raw = arr_dict[mode][feature]
        values, valid = _to_numeric(values_raw)
        mask_all = valid & (values != -999)
        v_all = values[mask_all]
        w_all = weights_full[mask_all]
        w_pos_mask = w_all > 0
        v_pos = v_all[w_pos_mask]
        w_pos = w_all[w_pos_mask]

        _, rng, nbins, logy = _resolve_feature_meta(
            feature, range_dict, bin_dict, logscale_dict, data=v_all
        )

        if rng is None or len(v_all) == 0:
            ax_hist.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_hist.transAxes)
            ax_hist.set_title(feature, fontsize=10)
            ax_ratio.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            continue

        hist_all, edges = np.histogram(v_all, bins=nbins, range=rng, weights=w_all)
        hist_pos, _ = np.histogram(v_pos, bins=edges, weights=w_pos)

        color_all = "C0"
        color_pos = "C1"
        color_ratio = "C2"
        ax_hist.stairs(hist_all, edges, label="all", linewidth=1.5, color=color_all)
        ax_hist.stairs(hist_pos, edges, label="weight > 0", linewidth=1.5, color=color_pos)
        ax_hist.set_title(feature, fontsize=10)
        ax_hist.set_xlim(rng)
        if logy:
            ax_hist.set_yscale("log")
        if c == 0:
            ax_hist.set_ylabel("a.u.", fontsize=9)
        ax_hist.tick_params(axis="x", labelbottom=False)
        ax_hist.tick_params(axis="both", labelsize=8)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(hist_all != 0, hist_pos / hist_all, np.nan)
        ax_ratio.stairs(ratio, edges, linewidth=1.2, color=color_ratio, label="pos / all")
        ax_ratio.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)

        ymin, ymax = 0.5, 1.5
        finite = ratio[np.isfinite(ratio)]
        if finite.size:
            rmin, rmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            ymin = min(ymin, rmin - 0.05)
            ymax = max(ymax, rmax + 0.05)
        ax_ratio.set_ylim(ymin, ymax)
        ax_ratio.set_xlim(rng)
        if c == 0:
            ax_ratio.set_ylabel("pos / all", fontsize=9)
        ax_ratio.set_xlabel(feature, fontsize=9)
        ax_ratio.tick_params(axis="both", labelsize=8)

    # Single shared legend at the top.
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
        if len(labels) >= 3:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10,
                   bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Weight-effect study: mode = {mode}", y=1.01, fontsize=14)

    suffix = f"_{filename_suffix}" if filename_suffix else ""
    out_path = f"{output_dir}/weight_effect_{mode}{suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] wrote {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ROOT data and plot variable histograms.")
    parser.add_argument('--root_path', '-f', type=str, help='Base path to the ROOT files')
    parser.add_argument('--output_dir', '-o', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--modes', '-m', type=str, nargs='+', default=production_modes, help='List of production modes to process')
    parser.add_argument('--plot_variables', '-p', type=str, nargs='+', default=non_lepton_features, help='List of variables to plot')
    parser.add_argument('--numpy_path', '-i', type=str, help='Input directory for numpy files (if loading from numpy instead of ROOT)')
    args = parser.parse_args()

    if args.numpy_path:
        arr_dict = load_arrays_from_numpy(args.numpy_path)
    else:
        arr_dict = load_arrays_from_root(args.root_path, "Events", args.modes)

    # Merge the background modes (include event weights)
    if args.plot_variables == ["Leptons"]:
        merge_vars = lepton_features + ["EventWeight_lumi138", "EventWeight_lumi9"]
    elif args.plot_variables == ["All"] or args.plot_variables == ["WeightEffect"]:
        merge_vars = non_lepton_features + lepton_features + ["EventWeight_lumi138", "EventWeight_lumi9"]
    else:
        merge_vars = args.plot_variables + ["EventWeight_lumi138", "EventWeight_lumi9"]
    arr_dict['WH'] = merge_modes(arr_dict, ['WplusH125', 'WminusH125'], variables=merge_vars)
    arr_dict['ggZZ'] = merge_modes(arr_dict, ggZZ_bgmodes, variables=merge_vars)
    arr_dict['EWK'] = merge_modes(arr_dict, EWK_bgmodes, variables=merge_vars)
    arr_dict['TTjets'] = merge_modes(arr_dict, TTjets_bgmodes, variables=merge_vars)
    arr_dict['Production'] = merge_modes(arr_dict, production_modes, variables=merge_vars)
    arr_dict['Background'] = merge_modes(arr_dict, ggZZ_bgmodes + EWK_bgmodes + TTjets_bgmodes, variables=merge_vars)
    plot_modes = ['ggH125', 'VBFH125', 'WH', 'ZH125', 'ttH125']

    if args.plot_variables == ["Leptons"]:
        # Special plotting for lepton variables
        plot_leptons(
            arr_dict,
            plot_modes=["Production", "Background"],
            event_weight='EventWeight_lumi138',
            range_dict=range_dict,
            logscale_dict=logscale_dict,
            output_dir=args.output_dir,
            variables=['LepPt', 'LepEta', 'LepPhi'],
            filename_suffix="",
        )
    elif args.plot_variables == ["All"]:
        # Plot leptons with special handling
        plot_leptons(
            arr_dict,
            plot_modes=["Production", "Background"],
            event_weight='EventWeight_lumi138',
            range_dict=range_dict,
            logscale_dict=logscale_dict,
            output_dir=args.output_dir,
            variables=['LepPt', 'LepEta', 'LepPhi'],
            filename_suffix="",
        )
        # Plot all non-leptonic variables
        for variable in non_lepton_features:
            print(f"[debug] Plotting variable '{variable}' for modes: {plot_modes}")
            plot_vars_hist(
                arr_dict,
                variable,
                plot_modes,
                event_weight='EventWeight_lumi138',
                range_dict=range_dict,
                logscale_dict=logscale_dict,
                output_dir=args.output_dir,
                filename_suffix="",
            )
    elif args.plot_variables == ["WeightEffect"]:
        study_modes = ['ggH125', 'VBFH125', 'WH', 'ZH125', 'ttH125',
                       'ggZZ', 'ZZTo4l', 'EWK', 'TTjets']
        all_feats = non_lepton_features + lepton_features
        for mode in study_modes:
            if mode not in arr_dict:
                print(f"[warn] mode '{mode}' not in arr_dict — skipping")
                continue
            print(f"[info] Weight-effect grid for mode '{mode}' ({len(all_feats)} features)")
            plot_weight_effect_grid(
                arr_dict,
                mode=mode,
                features=all_feats,
                event_weight='EventWeight_lumi138',
                range_dict=range_dict,
                bin_dict=bin_dict,
                logscale_dict=logscale_dict,
                output_dir=args.output_dir,
            )
    else:
        for variable in args.plot_variables:
            print(f"[debug] Plotting variable '{variable}' for modes: {plot_modes}")
            plot_vars_hist(
                arr_dict,
                variable,
                plot_modes,
                event_weight='EventWeight_lumi138',
                range_dict=range_dict,
                logscale_dict=logscale_dict,
                output_dir=args.output_dir,
                filename_suffix="",
            )



