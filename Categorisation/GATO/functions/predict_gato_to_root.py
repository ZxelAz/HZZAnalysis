#!/usr/bin/env python3
"""Run GATO GMM inference on a ROOT file containing BDT probability scores
and append a GATO_bin branch."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import uproot

from gatohep.models import gato_gmm_model
from gatohep.plotting_utils import fill_histogram_from_assignments
from gatohep.utils import compute_significance_from_hists


class gato_BDT(gato_gmm_model):
    """Mirrors the architecture used during training in GATO_HZZ.py."""

    def __init__(self, n_cats, dim, temperature=0.3, name="gato_BDT"):
        super().__init__(
            n_cats=n_cats,
            dim=dim,
            temperature=temperature,
            mean_norm="softmax",
            cov_offdiag_damping=0.1,
            name=name,
        )


def _discover_bdt_proba_columns(all_columns):
    """Return sorted list of BDT_proba_<code> column names."""
    pattern = re.compile(r"^BDT_proba_\d+$")
    cols = sorted(
        [c for c in all_columns if pattern.match(c)],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not cols:
        raise ValueError(
            "No BDT_proba_<code> columns found in input ROOT file. "
            "Run the BDT prediction step first to produce these columns."
        )
    return cols


# Columns used for the significance comparison printed at the end
_SIG_WEIGHT_COL   = "EventWeight_lumi138"
_SIG_HTXS_COL     = "HTXS_stage1_2_cat_pTjet30GeV"
_SIG_CUTBASED_COL = "category"
_SIG_BDT_COL      = "BDT_category"

# STXS Stage 1.2 label map (all merged codes expanded to a common name).
# UNKNOWN (0), FWDH codes, and any unmapped code are treated as background.
_STXS_LABEL_MAP: dict = {
    0:   'UNKNOWN',
    # Gluon fusion  (100 = GG2H_FWDH -> background, omitted)
    101: 'GG2H_PTH_GT200', 102: 'GG2H_PTH_GT200',
    103: 'GG2H_PTH_GT200', 104: 'GG2H_PTH_GT200',
    105: 'GG2H_0J_PTH_0_10',
    106: 'GG2H_0J_PTH_GT10',
    107: 'GG2H_1J_PTH_0_60',
    108: 'GG2H_1J_PTH_60_120',
    109: 'GG2H_1J_PTH_120_200',
    110: 'GG2H_GE2J_MJJ_0_350_PTH_0_60',
    111: 'GG2H_GE2J_MJJ_0_350_PTH_60_120',
    112: 'GG2H_GE2J_MJJ_0_350_PTH_120_200',
    113: 'GG2H_GE2J_MJJ_GT350', 114: 'GG2H_GE2J_MJJ_GT350',
    115: 'GG2H_GE2J_MJJ_GT350', 116: 'GG2H_GE2J_MJJ_GT350',
    # VBF  (200 = QQ2HQQ_FWDH -> background, omitted)
    201: 'QQ2HQQ_rest', 202: 'QQ2HQQ_rest',
    203: 'QQ2HQQ_rest', 205: 'QQ2HQQ_rest',
    204: 'QQ2HQQ_GE2J_MJJ_60_120',
    206: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',
    207: 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
    208: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25',
    210: 'QQ2HQQ_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25',
    209: 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
    # qq -> WH  (300 = QQ2HLNU_FWDH -> background, omitted)
    301: 'VH_lep_PTV_0_150', 302: 'VH_lep_PTV_0_150',
    303: 'VH_lep_PTV_GT150', 304: 'VH_lep_PTV_GT150', 305: 'VH_lep_PTV_GT150',
    # qq -> ZH  (400 = QQ2HLL_FWDH -> background, omitted)
    401: 'VH_lep_PTV_0_150', 402: 'VH_lep_PTV_0_150',
    403: 'VH_lep_PTV_GT150', 404: 'VH_lep_PTV_GT150', 405: 'VH_lep_PTV_GT150',
    # gg -> ZH  (500 = GG2HLL_FWDH -> background, omitted)
    501: 'VH_lep_PTV_0_150', 502: 'VH_lep_PTV_0_150',
    503: 'VH_lep_PTV_GT150', 504: 'VH_lep_PTV_GT150', 505: 'VH_lep_PTV_GT150',
    # ttH  (600 = TTH_FWDH -> background, omitted)
    601: 'TTH', 602: 'TTH', 603: 'TTH', 604: 'TTH', 605: 'TTH',
}

# Mapping between STXS 1.2 merged categories and Run2 cut-based categories
_STXS_TO_RUN2_MAPPING: dict = {
    'GG2H_PTH_GT200': ['Untagged_Pt200above'],
    'GG2H_0J_PTH_0_10': ['Untagged_0j_Pt0To10'],
    'GG2H_0J_PTH_GT10': ['Untagged_0j_Pt10To200'],
    'GG2H_1J_PTH_0_60': ['Untagged_1j_Pt0To60'],
    'GG2H_1J_PTH_60_120': ['Untagged_1j_Pt60To120'],
    'GG2H_1J_PTH_120_200': ['Untagged_1j_Pt120To200'],
    'GG2H_GE2J_MJJ_0_350_PTH_0_60': ['Untagged_2j_Pt0To60'],
    'GG2H_GE2J_MJJ_0_350_PTH_60_120': ['Untagged_2j_Pt60To120'],
    'GG2H_GE2J_MJJ_0_350_PTH_120_200': ['Untagged_2j_Pt120To200'],
    'GG2H_GE2J_MJJ_GT350': ['Untagged_2j_mjj350above'],
    'QQ2HQQ_GE2J_MJJ_60_120': ['VH_hadronic_tagged_mjj60To120'],
    'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200': ['VBF_2jet_tagged_Pt200above'],
    'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25': ['VBF_2jet_tagged_mjj350To700'],
    'QQ2HQQ_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25': ['VBF_3jet_tagged_mjj350above'],
    'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25': ['VBF_2jet_tagged_mjj700above'],
    'QQ2HQQ_rest': ['VBF_rest', 'VBF_1jet_tagged', 'VH_hadronic_tagged_rest'],
    'VH_lep_PTV_0_150': ['VH_leptonic_tagged_Pt0To150'],
    'VH_lep_PTV_GT150': ['VH_leptonic_tagged_Pt150above'],
    'TTH': ['ttH_hadronic_tagged', 'ttH_leptonic_tagged'],
}

# Ordered list of signal process names
_SIGNAL_PROCS: list = list(_STXS_TO_RUN2_MAPPING.keys())

# Reverse mapping: Run2 category name -> STXS process name
_RUN2_TO_STXS: dict = {
    run2_cat: stxs_proc
    for stxs_proc, run2_cats in _STXS_TO_RUN2_MAPPING.items()
    for run2_cat in run2_cats
}


def _compute_per_process_significances(
    weights: np.ndarray,
    proc_names: np.ndarray,
    bin_labels,
) -> tuple:
    """For each signal process, compute Z = sqrt(sum_bins Z_bin^2).

    Uses fill_histogram_from_assignments and compute_significance_from_hists
    to match the approach in GATO_HZZ.py.
    Other signal processes are treated as background for each target signal.

    Parameters
    ----------
    weights    : per-event event weights (1-D array)
    proc_names : STXS process name per event (e.g. 'GG2H_PTH_GT200' or 'UNKNOWN')
    bin_labels : categorisation bin label per event (int or str)
    """
    bin_labels = np.asarray(bin_labels)

    # Map arbitrary bin labels to contiguous integer indices
    unique_bins = np.unique(bin_labels)
    bin_to_idx = {b: i for i, b in enumerate(unique_bins)}
    n_bins = len(unique_bins)
    int_bins = np.array([bin_to_idx[b] for b in bin_labels])

    # Fill one histogram per unique process
    all_procs = np.unique(proc_names)
    filled: dict = {}
    for proc in all_procs:
        mask = proc_names == proc
        filled[proc] = fill_histogram_from_assignments(
            int_bins[mask].astype(np.float64),
            weights[mask].astype(np.float64),
            n_bins,
        )

    # Compute significance for each signal process
    per_process_Z: dict = {}
    for sig in _SIGNAL_PROCS:
        if sig not in filled:
            per_process_Z[sig] = 0.0
            continue
        # Background = every other process (other signals + UNKNOWN)
        bkg_hists = [filled[p] for p in filled if p != sig]
        Z = compute_significance_from_hists(filled[sig], bkg_hists)
        per_process_Z[sig] = float(Z)

    Z_vals = [per_process_Z[s] for s in _SIGNAL_PROCS if per_process_Z.get(s, 0) > 0]
    Z_geom_mean = float(np.power(np.prod(Z_vals), 1.0 / len(Z_vals))) if Z_vals else 0.0
    return per_process_Z, Z_geom_mean


def predict_gato_root_file(
    input_file: str,
    output_file: str,
    tree_name: str,
    model: gato_BDT,
    bdt_proba_columns: list,
    chunk_size: str,
) -> None:
    """Read ROOT, compute GATO bin indices, write new ROOT with GATO_bin branch."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nInput ROOT:  {input_path}")
    print(f"Output ROOT: {output_path}")
    print(f"Tree:        {tree_name}")
    print(f"Chunk size:  {chunk_size}")
    print(f"BDT proba columns ({len(bdt_proba_columns)}): {bdt_proba_columns[:5]} ...")

    # Check which extra columns are available for significance computation
    with uproot.open(input_path) as _f:
        _tree_keys = set(_f[tree_name].keys())
    _sig_extra_cols = [_SIG_WEIGHT_COL, _SIG_HTXS_COL, _SIG_CUTBASED_COL, _SIG_BDT_COL]
    compute_sig = all(c in _tree_keys for c in _sig_extra_cols)
    if not compute_sig:
        _missing = [c for c in _sig_extra_cols if c not in _tree_keys]
        print(f"[Note] Significance comparison skipped – columns missing: {_missing}")

    # Accumulators for significance computation
    acc_weights:  list = []
    acc_mode:     list = []
    acc_cat:      list = []
    acc_bdt_cat:  list = []
    acc_gato_bin: list = []

    total_entries = 0
    with uproot.open(input_path) as src:
        tree = src[tree_name]
        n_events = tree.num_entries
        print(f"Events:      {n_events}")

        with uproot.recreate(output_path) as dst:
            first_chunk = True

            for idx, chunk in enumerate(
                tree.iterate(step_size=chunk_size, library="np", how=dict), 1
            ):
                # Stack BDT probability columns into (N, dim) array
                bdt_scores = np.column_stack(
                    [chunk[col].astype(np.float32) for col in bdt_proba_columns]
                )

                # GATO inference: get_bin_indices accepts a plain (N, dim) tensor
                x_tensor = tf.constant(bdt_scores, dtype=tf.float32)
                gato_bins = model.get_bin_indices(x_tensor).numpy().astype(np.int32)

                chunk["GATO_bin"] = gato_bins

                if first_chunk:
                    dst[tree_name] = chunk
                    first_chunk = False
                else:
                    dst[tree_name].extend(chunk)

                # Accumulate arrays for significance computation
                if compute_sig:
                    acc_weights.append(chunk[_SIG_WEIGHT_COL].copy())
                    acc_mode.append(chunk[_SIG_HTXS_COL].copy())
                    acc_cat.append(chunk[_SIG_CUTBASED_COL].copy())
                    acc_bdt_cat.append(chunk[_SIG_BDT_COL].copy())
                    acc_gato_bin.append(gato_bins.copy())

                total_entries += len(gato_bins)
                print(
                    f"  Chunk {idx}: wrote {len(gato_bins)} events "
                    f"(running total: {total_entries})"
                )

    print(f"Finished writing {total_entries} events to: {output_path}")

    # ---------- Significance comparison ----------
    if compute_sig:
        weights_all = np.concatenate(acc_weights)
        htxs_all    = np.concatenate(acc_mode)
        cat_all     = np.concatenate(acc_cat)
        bdt_cat_all = np.concatenate(acc_bdt_cat)
        gato_all    = np.concatenate(acc_gato_bin)

        # Map HTXS integer codes to merged STXS process names
        proc_names = np.array(
            [_STXS_LABEL_MAP.get(int(c), 'UNKNOWN') for c in htxs_all]
        )

        # Cut-based: map Run2 category names to STXS names as bin labels
        cat_as_stxs = np.array(
            [_RUN2_TO_STXS.get(str(c), str(c)) for c in cat_all]
        )

        def _print_block(label, Z_dict, Z_mean):
            print(f"\n{label}")
            for proc in _SIGNAL_PROCS:
                print(f"  {proc}: Z = {Z_dict[proc]:.3f}")
            print(f"\nGeometric mean of Z-scores: {Z_mean:.3f}")

        Z_cut, Z_cut_mean = _compute_per_process_significances(
            weights_all, proc_names, cat_as_stxs,
        )
        _print_block("Computing cut-based categorisation significance...",
                     Z_cut, Z_cut_mean)

        Z_bdt, Z_bdt_mean = _compute_per_process_significances(
            weights_all, proc_names, bdt_cat_all,
        )
        _print_block("Computing BDT argmax categorisation significance...",
                     Z_bdt, Z_bdt_mean)

        Z_gato, Z_gato_mean = _compute_per_process_significances(
            weights_all, proc_names, gato_all,
        )
        _print_block("Computing GATO categorisation significance...",
                     Z_gato, Z_gato_mean)

        # ---------- Bar-chart comparison ----------
        short_labels = [p.replace('QQ2HQQ_GE2J_MJJ_', 'VBF_')
                          .replace('QQ2HQQ_', 'VBF_')
                          .replace('GG2H_GE2J_MJJ_', 'ggH_2j_')
                          .replace('GG2H_', 'ggH_')
                        for p in _SIGNAL_PROCS]

        x = np.arange(len(_SIGNAL_PROCS))
        width = 0.25

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.bar(x - width, [Z_cut[p] for p in _SIGNAL_PROCS],
               width, label=f"Cut-based (geom. mean = {Z_cut_mean:.2f})")
        ax.bar(x,         [Z_bdt[p] for p in _SIGNAL_PROCS],
               width, label=f"BDT argmax (geom. mean = {Z_bdt_mean:.2f})")
        ax.bar(x + width, [Z_gato[p] for p in _SIGNAL_PROCS],
               width, label=f"GATO (geom. mean = {Z_gato_mean:.2f})")

        ax.set_ylabel("Significance Z")
        ax.set_title("Per-process significance comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        _HERE = Path(__file__).resolve().parent
        plot_dir = _HERE.parent / "Plots" / "significance"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "significance_comparison.pdf"
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"\nPlot saved to: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained GATO model, assign bin indices to events "
        "using BDT probability scores, and save ROOT with GATO_bin branch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to GATO checkpoint directory (e.g. .../35_bins)",
    )
    parser.add_argument("--input", required=True, help="Input ROOT file (with BDT_proba_* columns)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--output-name", default=None, help="Output ROOT filename")
    parser.add_argument("--tree-name", default="Events", help="TTree name inside input ROOT")
    parser.add_argument(
        "--chunk-size",
        default="50 MB",
        help="Chunk size for uproot iteration (e.g. '50 MB', '50000 entries')",
    )
    parser.add_argument(
        "--n-cats",
        type=int,
        default=35,
        help="Number of GATO bins (must match trained model)",
    )
    parser.add_argument(
        "--bdt-proba-columns",
        nargs="+",
        default=None,
        help="BDT probability column names; if omitted, auto-detected from ROOT file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input ROOT file not found: {input_path}")

    # Discover BDT probability columns from the input file
    with uproot.open(input_path) as f:
        all_columns = f[args.tree_name].keys()

    bdt_proba_columns = args.bdt_proba_columns or _discover_bdt_proba_columns(all_columns)
    n_dim = len(bdt_proba_columns)

    print("=" * 80)
    print("GATO ROOT Inference")
    print("=" * 80)
    print(f"Model dir:   {args.model_dir}")
    print(f"Input:       {args.input}")
    print(f"Output dir:  {args.output_dir}")
    print(f"n_cats:      {args.n_cats}")
    print(f"BDT dim:     {n_dim}")
    print("=" * 80)

    # Build and restore model
    model = gato_BDT(n_cats=args.n_cats, dim=n_dim)
    model.restore(args.model_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name
    if output_name is None:
        output_name = f"{input_path.stem}_with_gato{input_path.suffix}"
    output_path = output_dir / output_name

    predict_gato_root_file(
        input_file=str(input_path),
        output_file=str(output_path),
        tree_name=args.tree_name,
        model=model,
        bdt_proba_columns=bdt_proba_columns,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
