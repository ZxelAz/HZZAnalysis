import os
import numpy as np
import ROOT
import argparse 
import pickle

# ---------------------------------------------------------------------------
# Python-side STXS stage-1.2 label derivation
# Mirrors the C++ STXSStage12MergedCat / STXSStage12Label functions defined
# in filter_dataframe.py.  Using the integer HTXS code + production_mode
# is immune to the Char_t[20] truncation that affects the string branch.
# ---------------------------------------------------------------------------

def _stxs_merge_cat(cat: int) -> int:
    """Python equivalent of C++ STXSStage12MergedCat."""
    cat = int(cat)
    if cat in (102, 103, 104):                              return 101
    if cat in (114, 115, 116):                              return 113
    if cat in (201, 203, 205):                              return 202
    if cat == 210:                                          return 208
    if cat in (302, 401, 402, 501, 502):                   return 301
    if cat in (304, 305, 403, 404, 405, 503, 504, 505):    return 303
    if cat in (602, 603, 604, 605):                         return 601
    return cat


# Base labels before mode-dependent prefix substitution
_STXS_BASE_LABELS = {
    0:   "UNKNOWN",
    100: "GG2H_FWDH",
    101: "GG2H_PTH_GT200",
    105: "GG2H_0J_PTH_0_10",
    106: "GG2H_0J_PTH_GT10",
    107: "GG2H_1J_PTH_0_60",
    108: "GG2H_1J_PTH_60_120",
    109: "GG2H_1J_PTH_120_200",
    110: "GG2H_GE2J_MJJ_0_350_PTH_0_60",
    111: "GG2H_GE2J_MJJ_0_350_PTH_60_120",
    112: "GG2H_GE2J_MJJ_0_350_PTH_120_200",
    113: "GG2H_GE2J_MJJ_GT350",
    200: "QQ2HQQ_FWDH",
    202: "QQ2HQQ_rest",
    204: "QQ2HQQ_GE2J_MJJ_60_120",
    206: "QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200",
    207: "QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25",
    208: "QQ2HQQ_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25",
    209: "QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25",
    300: "QQ2HLNU_FWDH",
    301: "VH_lep_PTV_0_150",
    303: "VH_lep_PTV_GT150",
    400: "QQ2HLL_FWDH",
    500: "GG2HLL_FWDH",
    600: "TTH_FWDH",
    601: "TTH",
}

# Mode-dependent prefix replacements (mirrors the C++ if-blocks)
_QQ2HQQ_PREFIX = {
    "VBFH125":    "VBF",
    "WplusH125":  "WplushadH",
    "WminusH125": "WminushadH",
    "ZH125":      "ZhadH",
}
_VH_PREFIX = {
    "WplusH125":  "WplusH",
    "WminusH125": "WminusH",
    "ZH125":      "ZH",
}


def stxs_int_to_label(cat_int: int, mode: str) -> str:
    """Python equivalent of C++ STXSStage12Label(cat, mode)."""
    merged = _stxs_merge_cat(cat_int)
    label = _STXS_BASE_LABELS.get(merged, "UNKNOWN")
    if label.startswith("QQ2HQQ"):
        repl = _QQ2HQQ_PREFIX.get(mode)
        if repl:
            return repl + label[6:]   # skip "QQ2HQQ"
    if label.startswith("VH"):
        repl = _VH_PREFIX.get(mode)
        if repl:
            return repl + label[2:]   # skip "VH"
    return label

def _extract_raw_string(value):
    """Extract a plain Python string from any ROOT/NumPy/cppyy representation."""
    import re as _re
    if isinstance(value, str):
        idx = value.find('\x00')
        return value[:idx] if idx >= 0 else value
    if isinstance(value, (bytes, np.bytes_)):
        s = value.rstrip(b'\x00').decode(errors="ignore")
        idx = s.find('\x00')
        return s[:idx] if idx >= 0 else s
    if np.isscalar(value):
        s = str(value)
        idx = s.find('\x00')
        return s[:idx] if idx >= 0 else s
    # Handle numpy arrays or lists/tuples of single-character strings/bytes
    if isinstance(value, (np.ndarray, list, tuple)):
        if hasattr(value, 'dtype') and value.dtype.kind == 'S':
            chars = []
            for x in value:
                c = x.rstrip(b'\x00').decode(errors='ignore')
                if not c or '\x00' in c:
                    break
                chars.append(c)
            return ''.join(chars)
        if hasattr(value, 'dtype') and value.dtype.kind == 'U':
            chars = []
            for x in value:
                if not x or x == '\x00':
                    break
                chars.append(x)
            return ''.join(chars)
        try:
            chars = []
            for x in value:
                if isinstance(x, (bytes, np.bytes_)):
                    c = x.rstrip(b'\x00').decode(errors='ignore')
                elif isinstance(x, str):
                    c = x
                else:
                    c = str(x)
                if not c or c == '\x00':
                    break
                chars.append(c)
            return ''.join(chars)
        except Exception:
            pass
    # Handle cppyy/ROOT char-array objects whose str() looks like { 'G', 'G', ... '0x00' }
    s = str(value)
    if s.startswith('{') and s.endswith('}'):
        chars = []
        for m in _re.findall(r"'([^']*)'", s):
            if m == '0x00' or '\x00' in m:
                break  # C-string null terminator
            chars.append(m)
        return ''.join(chars)
    # Try iterating over arbitrary cppyy iterables (chars as ints)
    try:
        chars = []
        for x in value:
            if isinstance(x, int):
                if x == 0:
                    break
                chars.append(chr(x))
            elif isinstance(x, bytes):
                c = x.rstrip(b'\x00').decode(errors='ignore')
                if not c:
                    break
                chars.append(c)
            elif isinstance(x, str):
                if not x or x == '\x00':
                    break
                chars.append(x)
            else:
                cs = str(x)
                if cs in ('0x00', '\x00', ''):
                    break
                chars.append(cs)
        return ''.join(chars)
    except (TypeError, AttributeError):
        pass
    idx = s.find('\x00')
    return s[:idx] if idx >= 0 else s


def normalize_mode_value(value):
    """Convert ROOT/NumPy/cppyy string-like entries into a plain Python string."""
    return _extract_raw_string(value)


def normalize_string_array(values):
    """Convert a column of ROOT/NumPy string-like entries to plain Python strings."""
    return np.array([normalize_mode_value(v) for v in values], dtype=object)


def _safe_name(value):
    norm = normalize_mode_value(value)
    return norm.replace("/", "_").replace(" ", "_")


def save_fit_result(fit_result, fit_results_dir, name):
    """Persist a RooFitResult into fit_results/fit_results.root."""
    if fit_results_dir is None or fit_result is None:
        return
    os.makedirs(fit_results_dir, exist_ok=True)
    result_path = os.path.join(fit_results_dir, "fit_results.root")
    out_file = ROOT.TFile.Open(result_path, "UPDATE")
    fit_result.Write(name, ROOT.TObject.kOverwrite)
    out_file.Close()

def _patch_string_columns_via_uproot(input_file: str, string_cols: list, dfs: dict) -> None:
    """Overwrite string columns in AsNumpy dicts with full (non-truncated) values read via uproot.

    ROOT's AsNumpy reads variable-length C-string branches into a fixed 20-byte RVec<char> buffer,
    silently truncating any label longer than 19 chars.  uproot reads the same branch correctly.
    """
    import uproot

    cols_to_load = list(set(string_cols) | {"LepPdgId_0", "LepPdgId_1", "LepPdgId_2", "LepPdgId_3"})
    with uproot.open(f"{input_file}:Events") as tree:
        full = tree.arrays(cols_to_load, library="np")

    pdg0 = np.abs(full["LepPdgId_0"])
    pdg1 = np.abs(full["LepPdgId_1"])
    pdg2 = np.abs(full["LepPdgId_2"])
    pdg3 = np.abs(full["LepPdgId_3"])
    mask_4e   = (pdg0 == 11) & (pdg1 == 11) & (pdg2 == 11) & (pdg3 == 11)
    mask_4mu  = (pdg0 == 13) & (pdg1 == 13) & (pdg2 == 13) & (pdg3 == 13)
    mask_2e2mu = ~(mask_4e | mask_4mu)

    fs_masks = {"4e": mask_4e, "4mu": mask_4mu, "2e2mu": mask_2e2mu}
    for fs, df in dfs.items():
        for col in string_cols:
            if col in full:
                df[col] = full[col][fs_masks[fs]]


def split_final_state(input_file):
    df_root = ROOT.RDataFrame("Events", input_file)
    df_4e = df_root.Filter("abs(LepPdgId_0) == 11 && abs(LepPdgId_1) == 11 && abs(LepPdgId_2) == 11 && abs(LepPdgId_3) == 11").AsNumpy()
    df_4mu = df_root.Filter("abs(LepPdgId_0) == 13 && abs(LepPdgId_1) == 13 && abs(LepPdgId_2) == 13 && abs(LepPdgId_3) == 13").AsNumpy()
    df_2e2mu = df_root.Filter(
    "!( (abs(LepPdgId_0) == 11 && abs(LepPdgId_1) == 11 && abs(LepPdgId_2) == 11 && abs(LepPdgId_3) == 11) || "
    "   (abs(LepPdgId_0) == 13 && abs(LepPdgId_1) == 13 && abs(LepPdgId_2) == 13 && abs(LepPdgId_3) == 13) )"
    ).AsNumpy()

    dfs = {"4e": df_4e, "4mu": df_4mu, "2e2mu": df_2e2mu}

    # Detect which columns are C-string branches (variable-length, truncated by AsNumpy)
    # and re-read them properly via uproot.
    f = ROOT.TFile.Open(input_file)
    tree = f.Get("Events")
    string_cols = []
    for branch in tree.GetListOfBranches():
        for leaf in branch.GetListOfLeaves():
            if leaf.GetTypeName() == "Char_t":
                string_cols.append(branch.GetName())
                break
    f.Close()

    if string_cols:
        print(f"Re-reading {len(string_cols)} C-string column(s) via uproot to avoid ROOT truncation: {string_cols}")
        _patch_string_columns_via_uproot(input_file, string_cols, dfs)

    return df_4e, df_4mu, df_2e2mu

def response_matrix_element(df, ew_name, bin_name, bin, category_name, category, genEventSumw_name="genEventSumw"):
    """
    Calculate the response matrix element for a given category and bin (\epsilon A)_{ij}.

    Parameters:
    df (dict): Dictionary containing the data arrays.
    ew_name (str): Name of the event weight array.
    bin_name (str): Name of the bin array. (STXS 1.2 labels)
    bin (int): Bin index.
    category_name (str): Name of the category array. (cut-based, BDT, GATO)
    category (int): Category index.

    Returns:
    tuple: (epsilon, acceptance, epsilonA)
    """
    bin_mask = df[bin_name] == bin
    num_bin = np.sum(df[ew_name][bin_mask])
    print(f"Total events in bin {bin}: {num_bin}")

    # Handle both numeric and string categories; convert element-wise to avoid NumPy dtype issues
    cat_mask = np.array([str(c) == str(category) for c in df[category_name]])
    num_cat = np.sum(df[ew_name][cat_mask & bin_mask])
    print(f"Events in category {category} and bin {bin}: {num_cat}")

    num_total = df[genEventSumw_name][bin_mask][0]
    print(f"Total events in dataset: {num_total}")

    epsilon = num_cat / num_bin
    acceptance = num_bin / num_total
    epsilonA = num_cat / num_total
    print(f"Efficiency (epsilon) for category {category} and bin {bin}: {epsilon:.4f}")
    print(f"Acceptance for bin {bin}: {acceptance:.4f}")
    print(f"Combined efficiency*acceptance (epsilonA) for category {category} and bin {bin}: {epsilonA:.4f}")
    return epsilon, acceptance, epsilonA

def fit_pdf(
    df,
    bin_name,
    bin,
    variable="ZZCand_mass",
    event_weight_name="genWeight",
    output_dir=None,
    fit_results_dir=None,
    final_state=None,
):
    """
    Fit a signal probability density function (PDF) to the data for a given category and bin: f_{ij}(m_{4l}).

    Parameters:
    df (dict): Dictionary containing the data arrays.
    category_name (str): Name of the category array. (cut-based, BDT, GATO)
    category (int): Category index.
    bin_name (str): Name of the bin array. (STXS 1.2 labels)
    bin (int): Bin index.
    variable (str): Name of the variable to fit. Default is "ZZCand_mass".
    event_weight_name (str): Name of the event weight array. Default is "genWeight".

    Returns:
    ROOT.RooFitResult: Fit result object.
    """
    import ROOT
    from ROOT import RooFit

    # Create a RooWorkspace
    w = ROOT.RooWorkspace("w", "w")

    # Define the variable to fit
    w.factory(f"{variable}[105, 140]")
    var = w.var(variable)
    weight_var = ROOT.RooRealVar("event_weight", "event_weight", -1e9, 1e9)

    # Create a dataset for the specified bin
    mask = (df[bin_name] == bin)
    data = df[variable][mask]
    weights = df[event_weight_name][mask]
    dataset = ROOT.RooDataSet(
        "data",
        "data",
        ROOT.RooArgSet(var, weight_var),
        RooFit.WeightVar("event_weight"),
    )
    for value, weight in zip(data, weights):
        var.setVal(value)
        weight_var.setVal(float(weight))
        dataset.add(ROOT.RooArgSet(var, weight_var), float(weight))

    # Define the signal PDF (DSCB)
    w.factory(
        f"RooCrystalBall::signal({variable}, "
        "mean[125, 105, 140], sigma[1.5, 0.1, 10], "
        "alphaL[1.5, 0.01, 10], nL[3, 0.1, 50], "
        "alphaR[1.5, 0.01, 10], nR[3, 0.1, 50])"
    )

    # Signal-only fit
    signal = w.pdf("signal")
    fit_result = signal.fitTo(dataset, RooFit.Save(), RooFit.SumW2Error(True))
    save_fit_result(
        fit_result,
        fit_results_dir,
        f"fitresult_signal_{_safe_name(final_state)}_bin{_safe_name(bin)}",
    )

    # Print fit results
    # print(f"Fit results for category {category} and bin {bin}:")
    # fit_result.Print()

    # Plot data and fitted signal PDF, and save if an output directory is provided.
    if output_dir is not None:
        frame = var.frame(RooFit.Title(f"{variable}: {final_state}, bin={bin}"))
        dataset.plotOn(frame, RooFit.DataError(ROOT.RooAbsData.SumW2))
        signal.plotOn(frame, RooFit.LineColor(ROOT.kRed), RooFit.LineWidth(2))
        n_fit_params = fit_result.floatParsFinal().getSize() if fit_result else 0
        chi2_ndf = frame.chiSquare(n_fit_params) if n_fit_params >= 0 else frame.chiSquare()

        canvas_name = f"c_{final_state}_{bin}"
        canvas = ROOT.TCanvas(canvas_name, canvas_name, 900, 700)
        frame.GetXaxis().SetTitle(variable)
        frame.GetYaxis().SetTitle("Events")
        frame.Draw()

        chi2_legend = ROOT.TLegend(0.62, 0.82, 0.88, 0.90)
        chi2_legend.SetBorderSize(0)
        chi2_legend.SetFillStyle(0)
        chi2_dummy = ROOT.TGraph()
        chi2_legend.AddEntry(chi2_dummy, f"#chi^{{2}}/ndf = {chi2_ndf:.3f}", "")
        chi2_legend.Draw()

        safe_state = str(final_state).replace("/", "_") if final_state is not None else "all"
        safe_bin = str(bin).replace("/", "_")
        plot_base = os.path.join(output_dir, f"fit_{safe_state}_bin{safe_bin}")
        canvas.SaveAs(f"{plot_base}.png")
        print(f"  Saved fit plots: {plot_base}.png")

    return fit_result


def fit_background_erf(
    df,
    mode_name,
    mode,
    variable="ZZCand_mass",
    event_weight_name="genWeight",
    output_dir=None,
    fit_results_dir=None,
    final_state=None,
    skip_mode_filter=False,
    fit_function="Chebychev",
):
    """Fit a background model with an error-function shape for one bin.

    Model:
        bkg(m) = 0.5 * (1 - erf((m - turn) / width))
    
    Parameters:
    skip_mode_filter: If True, use all data in df without filtering by mode (used for pre-filtered merged modes)
    """
    import ROOT
    from ROOT import RooFit

    w = ROOT.RooWorkspace("w_bkg", "w_bkg")
    w.factory(f"{variable}[105, 140]")
    var = w.var(variable)
    weight_var = ROOT.RooRealVar("event_weight", "event_weight", -1e9, 1e9)

    if skip_mode_filter:
        # Data already pre-filtered for merged modes
        data = df[variable]
        weights = df[event_weight_name]
    else:
        # Filter by mode for single-mode fits
        mask = np.array([normalize_mode_value(entry) == mode for entry in df[mode_name]], dtype=bool)
        data = df[variable][mask]
        weights = df[event_weight_name][mask]

    dataset = ROOT.RooDataSet(
        "data_bkg",
        "data_bkg",
        ROOT.RooArgSet(var, weight_var),
        RooFit.WeightVar("event_weight"),
    )
    for value, weight in zip(data, weights):
        var.setVal(value)
        weight_var.setVal(float(weight))
        dataset.add(ROOT.RooArgSet(var, weight_var), float(weight))

    # Polynomial (Chebychev) background parameterization.
    w.factory(f"Chebychev::bkg_pdf({variable}, {{a0[0.5, -1, 1], a1[0.1, -1, 1], a2[0.01, -1, 1]}})")

    bkg_pdf = w.pdf("bkg_pdf")
    fit_result = bkg_pdf.fitTo(dataset, RooFit.Save(), RooFit.SumW2Error(True))
    save_fit_result(
        fit_result,
        fit_results_dir,
        f"fitresult_bkg_{_safe_name(final_state)}_{_safe_name(mode)}",
    )

    if output_dir is not None:
        frame = var.frame(RooFit.Title(f"{variable}: {final_state}, mode={mode}, bkg erf fit"))
        dataset.plotOn(frame, RooFit.DataError(ROOT.RooAbsData.SumW2))
        bkg_pdf.plotOn(frame, RooFit.LineColor(ROOT.kBlue + 1), RooFit.LineWidth(2))
        n_fit_params = fit_result.floatParsFinal().getSize() if fit_result else 0
        chi2_ndf = frame.chiSquare(n_fit_params) if n_fit_params >= 0 else frame.chiSquare()

        canvas_name = f"c_bkg_erf_{final_state}_{mode}"
        canvas = ROOT.TCanvas(canvas_name, canvas_name, 900, 700)
        frame.GetXaxis().SetTitle(variable)
        frame.GetYaxis().SetTitle("Events")
        frame.Draw()

        chi2_legend = ROOT.TLegend(0.62, 0.82, 0.88, 0.90)
        chi2_legend.SetBorderSize(0)
        chi2_legend.SetFillStyle(0)
        chi2_dummy = ROOT.TGraph()
        chi2_legend.AddEntry(chi2_dummy, f"#chi^{{2}}/ndf = {chi2_ndf:.3f}", "")
        chi2_legend.Draw()

        safe_state = str(final_state).replace("/", "_") if final_state is not None else "all"
        safe_mode = str(mode).replace("/", "_")
        plot_base = os.path.join(output_dir, f"fit_bkg_erf_{safe_state}_mode{safe_mode}")
        canvas.SaveAs(f"{plot_base}.png")
        print(f"  Saved background erf fit plot: {plot_base}.png")

    return fit_result


def fit_background_zx_landau(
    df,
    variable="ZZMass",
    output_dir=None,
    fit_results_dir=None,
    final_state=None,
):
    """Fit ZX background with an unweighted Landau model."""
    import ROOT
    from ROOT import RooFit

    w = ROOT.RooWorkspace("w_zx", "w_zx")
    w.factory(f"{variable}[105, 140]")
    var = w.var(variable)

    data = df[variable]
    dataset = ROOT.RooDataSet("data_zx", "data_zx", ROOT.RooArgSet(var))
    for value in data:
        var.setVal(float(value))
        dataset.add(ROOT.RooArgSet(var))

    w.factory(f"RooLandau::bkg_pdf({variable}, mean[120, 80, 140], sigma[5, 0.1, 20])")
    bkg_pdf = w.pdf("bkg_pdf")
    fit_result = bkg_pdf.fitTo(dataset, RooFit.Save())
    save_fit_result(
        fit_result,
        fit_results_dir,
        f"fitresult_bkg_{_safe_name(final_state)}_ZX",
    )

    if output_dir is not None:
        frame = var.frame(RooFit.Title(f"{variable}: {final_state}, mode=ZX, bkg Landau fit"))
        dataset.plotOn(frame)
        bkg_pdf.plotOn(frame, RooFit.LineColor(ROOT.kGreen + 2), RooFit.LineWidth(2))
        n_fit_params = fit_result.floatParsFinal().getSize() if fit_result else 0
        chi2_ndf = frame.chiSquare(n_fit_params) if n_fit_params >= 0 else frame.chiSquare()

        canvas_name = f"c_bkg_landau_{final_state}_ZX"
        canvas = ROOT.TCanvas(canvas_name, canvas_name, 900, 700)
        frame.GetXaxis().SetTitle(variable)
        frame.GetYaxis().SetTitle("Events")
        frame.Draw()

        chi2_legend = ROOT.TLegend(0.62, 0.82, 0.88, 0.90)
        chi2_legend.SetBorderSize(0)
        chi2_legend.SetFillStyle(0)
        chi2_dummy = ROOT.TGraph()
        chi2_legend.AddEntry(chi2_dummy, f"#chi^{{2}}/ndf = {chi2_ndf:.3f}", "")
        chi2_legend.Draw()

        safe_state = str(final_state).replace("/", "_") if final_state is not None else "all"
        plot_base = os.path.join(output_dir, f"fit_bkg_landau_{safe_state}_modeZX")
        canvas.SaveAs(f"{plot_base}.png")
        print(f"  Saved ZX Landau fit plot: {plot_base}.png")

    return fit_result


def main():
    parser = argparse.ArgumentParser(description='Calculate response matrix elements and fit PDFs for STXS categories and bins')
    parser.add_argument('--input_file', required=True, help='Path to input ROOT file')
    parser.add_argument('--output_dir', required=True, help='Directory to save output results')
    parser.add_argument('--variable', default='ZZCand_mass', help='Variable to fit (default: ZZCand_mass)')
    parser.add_argument('--category_name', default='CB_cat', help='Name of category array (default: CB_cat)')
    parser.add_argument('--bin_name', default='HTXS_stage1_2_cat_pTjet30GeV', help='Name of bin array (default: HTXS_stage1_2_cat_pTjet30GeV)')
    parser.add_argument('--mode_name', default='mode', help='Name of mode array (default: mode)')
    parser.add_argument('--bkg_event_weight_name', default='EventWeight_lumi62', help='Name of event weight array (default: EventWeight_lumi62)')
    parser.add_argument('--signal_event_weight_name', default='genWeight', help='Name of event weight array (default: genWeight)')
    parser.add_argument('--skip_zx', action='store_true', help='Skip ZX background loading, fitting, and yield calculation')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fit_plot_dir = os.path.join(args.output_dir, "fit_plots")
    fit_results_dir = os.path.join(args.output_dir, "fit_results")
    os.makedirs(fit_plot_dir, exist_ok=True)
    os.makedirs(fit_results_dir, exist_ok=True)

    print("=" * 60)
    print("loading data and splitting into final states...")
    df = {}
    df['4e'], df['4mu'], df['2e2mu'] = split_final_state(args.input_file)
    print("Data loaded and split into final states.")
    print(f"Number of entries in 4e: {len(df['4e'][args.variable])}")
    print(f"Number of entries in 4mu: {len(df['4mu'][args.variable])}")
    print(f"Number of entries in 2e2mu: {len(df['2e2mu'][args.variable])}")
    print("=" * 60)

    normalized_category_name = f"__normalized__{args.category_name}"
    # Derive bin labels from the integer HTXS code + production_mode to avoid
    # Char_t[20] string truncation (which makes e.g. GG2H_GE2J_MJJ_0_350_PTH_0_60,
    # _PTH_60_120, and _PTH_120_200 all look identical at 19 chars).
    STXS_INT_COL = "HTXS_stage1_2_cat_pTjet30GeV"
    normalized_bin_name = "__stxs_label__"

    for final_state in df.keys():
        df[final_state][normalized_category_name] = normalize_string_array(df[final_state][args.category_name])
        # Derive correct full-length label from integer + mode
        cat_ints = df[final_state][STXS_INT_COL]
        modes    = [normalize_mode_value(m) for m in df[final_state][args.mode_name]]
        df[final_state][normalized_bin_name] = np.array(
            [stxs_int_to_label(c, m) for c, m in zip(cat_ints, modes)],
            dtype=object,
        )

    # ZX bkg
    df_ZX = {}
    if not args.skip_zx:
        # Load ZX files
        df_ZX_years = {}
        df_ZX_years['2022'] = ROOT.RDataFrame("Events", "/eos/user/z/zhiheng/STXS_samples/2022Data/ZX_2022.root").Filter("ZZMass >= 105 && ZZMass <= 140").AsNumpy(["ZZMass", "ZX_Yield", 'FinState'])
        df_ZX_years['2022EE'] = ROOT.RDataFrame("Events", "/eos/user/z/zhiheng/STXS_samples/2022Data/ZX_2022EE.root").Filter("ZZMass >= 105 && ZZMass <= 140").AsNumpy(["ZZMass", "ZX_Yield", 'FinState'])
        df_ZX_years['2023preBPix'] = ROOT.RDataFrame("Events", "/eos/user/z/zhiheng/STXS_samples/2023Data/ZX_2023preBPix.root").Filter("ZZMass >= 105 && ZZMass <= 140").AsNumpy(["ZZMass", "ZX_Yield", 'FinState'])
        df_ZX_years['2023postBPix'] = ROOT.RDataFrame("Events", "/eos/user/z/zhiheng/STXS_samples/2023Data/ZX_2023postBPix.root").Filter("ZZMass >= 105 && ZZMass <= 140").AsNumpy(["ZZMass", "ZX_Yield", 'FinState'])

        print("\n" + "=" * 60)
        print("ZX debug: loaded yearly inputs")
        for year, df_year in df_ZX_years.items():
            n_year = len(df_year["ZZMass"])
            fs_vals, fs_counts = np.unique(df_year["FinState"], return_counts=True)
            print(f"  {year}: events={n_year}, ZX_Yield sum={np.sum(df_year['ZX_Yield']):.6f}")
            print(f"    FinState distribution: {dict(zip(fs_vals.tolist(), fs_counts.tolist()))}")

        # merge into one df
        df_ZX_tot = {}
        for year, df_year in df_ZX_years.items():
            for key in df_year.keys():
                if key not in df_ZX_tot:
                    df_ZX_tot[key] = df_year[key]
                else:
                    df_ZX_tot[key] = np.concatenate((df_ZX_tot[key], df_year[key]))

        print("ZX debug: merged total")
        print(f"  total events={len(df_ZX_tot['ZZMass'])}, ZX_Yield sum={np.sum(df_ZX_tot['ZX_Yield']):.6f}")
        fs_vals_tot, fs_counts_tot = np.unique(df_ZX_tot["FinState"], return_counts=True)
        print(f"  FinState distribution total: {dict(zip(fs_vals_tot.tolist(), fs_counts_tot.tolist()))}")

        # split into final states
        mask_4e = (df_ZX_tot['FinState'] == 0)
        mask_4mu = (df_ZX_tot['FinState'] == 1)
        mask_2e2mu = (df_ZX_tot['FinState'] == 2) | (df_ZX_tot['FinState'] == 3)
        df_ZX['4e'] = {key: arr[mask_4e] for key, arr in df_ZX_tot.items()}
        df_ZX['4mu'] = {key: arr[mask_4mu] for key, arr in df_ZX_tot.items()}
        df_ZX['2e2mu'] = {key: arr[mask_2e2mu] for key, arr in df_ZX_tot.items()}

        print("ZX debug: split by final state")
        for fs in ["4e", "4mu", "2e2mu"]:
            n_fs = len(df_ZX[fs]["ZZMass"])
            y_fs = np.sum(df_ZX[fs]["ZX_Yield"]) if n_fs > 0 else 0.0
            print(f"  {fs}: events={n_fs}, ZX_Yield sum={y_fs:.6f}")
        print("=" * 60)
    else:
        print("Skipping ZX background loading (--skip_zx flag set)")

    epsilon = {}
    acceptance = {}
    epsilonA = {}
    signal_pdf = {}
    N_bkg = {}
    N_bkg_category = {}

    total_iterations = sum(
        len(np.unique(df[key][normalized_category_name])) * len(np.unique(df[key][normalized_bin_name]))
        for key in df.keys()
    )
    current_iteration = 0
    
    for final_state in df.keys():
        print(f"\n{'='*60}")
        print(f"Processing final state: {final_state}")
        print(f"{'='*60}")
        categories = np.unique(df[final_state][normalized_category_name])
        bins = np.unique(df[final_state][normalized_bin_name])
        signal_bins = [bin for bin in bins if bin not in ('UNKNOWN', 'GG2H_FWDH', 'QQ2HLNU_FWDH', 'GG2HLL_FWDH', 'QQ2HLL_FWDH', 'TTH_FWDH', 'WplushadH_FWDH', 'WminushadH_FWDH', 'VBF_FWDH', 'ZhadH_FWDH')]
        print(f"Unique categories: {len(categories)}, Unique bins: {len(bins)}, Signal bins: {len(signal_bins)}")
         
        # backgrounds qqZZ and ggZZ (merged)
        gg_modes = ["ggTo2e2mu_Contin_MCFM701", "ggTo4e_Contin_MCFM701", "ggTo4mu_Contin_MCFM701", "ggTo4tau_Contin_MCFM701", "ggTo2mu2tau_Contin_MCFM701", "ggTo2e2tau_Contin_MCFM701"]

        # Background yield per category for both backgrounds
        qqzz_mode = "ZZTo4l"
        ggzz_mode = "ggZZ"
        for category in categories:
            cat_mask = np.array(
                [str(c) == str(category) for c in df[final_state][normalized_category_name]],
                dtype=bool,
            )
            qqzz_mask = np.array(
                [normalize_mode_value(entry) == qqzz_mode for entry in df[final_state][args.mode_name]],
                dtype=bool,
            )
            ggzz_mask = np.array(
                [normalize_mode_value(entry) in gg_modes for entry in df[final_state][args.mode_name]],
                dtype=bool,
            )
            qqzz_yield = float(np.sum(df[final_state][args.bkg_event_weight_name][cat_mask & qqzz_mask]))
            ggzz_yield = float(np.sum(df[final_state][args.bkg_event_weight_name][cat_mask & ggzz_mask]))
            
            N_bkg_category[(final_state, category)] = {
                qqzz_mode: qqzz_yield,
                ggzz_mode: ggzz_yield,
            }
        
        # Fit qqZZ (ZZTo4l) separately
        print(f"\nProcessing background mode: {qqzz_mode}")
        mask_mode = np.array(
            [normalize_mode_value(entry) == qqzz_mode for entry in df[final_state][args.mode_name]],
            dtype=bool,
        )
        df_bkg_mode = {key: arr[mask_mode] for key, arr in df[final_state].items()}
        fit_result_bkg = fit_background_erf(
            df_bkg_mode,
            args.mode_name,
            qqzz_mode,  
            variable=args.variable,
            event_weight_name=args.bkg_event_weight_name,
            output_dir=fit_plot_dir,
            fit_results_dir=fit_results_dir,
            final_state=f"{final_state}_{qqzz_mode}",
        )
        print(f"  ✓ Background erf fit completed for mode: {qqzz_mode}")
        N_bkg[(final_state, qqzz_mode)] = np.sum(df_bkg_mode[args.bkg_event_weight_name])
        
        # Fit ggZZ (merged: all ggTo... modes)
        print(f"\nProcessing background mode: {ggzz_mode} (merged)")
        mask_mode = np.array(
            [normalize_mode_value(entry) in gg_modes for entry in df[final_state][args.mode_name]],
            dtype=bool,
        )
        df_bkg_mode = {key: arr[mask_mode] for key, arr in df[final_state].items()}
        fit_result_bkg = fit_background_erf(
            df_bkg_mode,
            args.mode_name,
            ggzz_mode,  
            variable=args.variable,
            event_weight_name=args.bkg_event_weight_name,
            output_dir=fit_plot_dir,
            fit_results_dir=fit_results_dir,
            final_state=f"{final_state}_{ggzz_mode}",
            skip_mode_filter=True,
        )
        print(f"  ✓ Background erf fit completed for mode: {ggzz_mode}")
        # calculate ggzz bkg yield
        N_bkg[(final_state, ggzz_mode)] = np.sum(df_bkg_mode[args.bkg_event_weight_name])

        # Fit ZX background
        if not args.skip_zx:
            print(f"\nProcessing background mode: ZX")
            fit_result_bkg = fit_background_zx_landau(
                df_ZX[final_state],
                variable="ZZMass",
                output_dir=fit_plot_dir,
                fit_results_dir=fit_results_dir,
                final_state=f"{final_state}_ZX",
            )
            N_bkg[(final_state, "ZX")] = np.sum(df_ZX[final_state]['ZX_Yield'])
        else:
            print("  Skipping ZX background fit (--skip_zx flag set)")

        # epsilon and pdf for signals 
        for bin in signal_bins:
            bin_str = normalize_mode_value(bin)
            for category in categories:
                category_str = normalize_mode_value(category)
                current_iteration += 1
                print(f"\n[{current_iteration}/{total_iterations}] Processing {final_state} - category {category_str}, bin {bin_str}")
                try:
                    eps, acc, epsA = response_matrix_element(
                        df[final_state],
                        args.signal_event_weight_name,
                        normalized_bin_name,
                        bin_str,
                        normalized_category_name,
                        category_str,
                    )
                    epsilon[(final_state, category_str, bin_str)] = eps
                    acceptance[(final_state, bin_str)] = acc
                    epsilonA[(final_state, category_str, bin_str)] = epsA
                    print(f"  ✓ Response matrix element calculated for bin {bin_str}")
                except Exception as e:
                    print(f"  ✗ Error processing {final_state} - category {category_str}, bin {bin_str}: {e}")
                    import traceback
                    traceback.print_exc()
            signal_pdf[(final_state, category_str, bin_str)] = fit_pdf(
                df[final_state],
                normalized_bin_name,
                bin_str,
                variable=args.variable,
                event_weight_name=args.signal_event_weight_name,
                output_dir=fit_plot_dir,
                fit_results_dir=fit_results_dir,
                final_state=final_state,
            )
            print(f"  ✓ Signal PDF fit completed for bin {bin_str}")
    
    
                
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Total entries processed: {total_iterations}")
    print(f"epsilon dict size: {len(epsilon)}")
    print(f"acceptance dict size: {len(acceptance)}")
    print(f"epsilonA dict size: {len(epsilonA)}")
    print(f"signal_pdf dict size: {len(signal_pdf)}")
    print(f"N_bkg dict size: {len(N_bkg)}")
    print(f"N_bkg_category dict size: {len(N_bkg_category)}")
    print(f"{'='*60}")

    with open(os.path.join(args.output_dir, "epsilon.pkl"), "wb") as f:
        pickle.dump(epsilon, f)
    with open(os.path.join(args.output_dir, "acceptance.pkl"), "wb") as f:
        pickle.dump(acceptance, f)
    with open(os.path.join(args.output_dir, "epsilonA.pkl"), "wb") as f:
        pickle.dump(epsilonA, f)
    with open(os.path.join(args.output_dir, "N_bkg.pkl"), "wb") as f:
        pickle.dump(N_bkg, f)
    with open(os.path.join(args.output_dir, "N_bkg_category.pkl"), "wb") as f:
        pickle.dump(N_bkg_category, f)
    
    print(f"Saved dictionaries to: {args.output_dir}")
    print("printing epsilonA dictionary for verification:")
    print(epsilonA)

                
if __name__ == "__main__":
    main()