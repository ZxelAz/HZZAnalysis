#!/usr/bin/env python3
"""
Plot all 19 1D likelihood scans as subplots in a single figure.

Usage
-----
python3 plot_all_1D_scans.py
python3 plot_all_1D_scans.py --trial-dir Results/CutBased --output combine_plots/all_1D_scans.png
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from scipy.interpolate import UnivariateSpline

ROOT.gROOT.SetBatch(True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIAL_DIR_DEFAULT = "${HZZ_ROOT}/InferenceModel/Results/CutBased"

POIS = [
    "r_proc_GG2H_0J_PTH_0_10",
    "r_proc_GG2H_0J_PTH_GT10",
    "r_proc_GG2H_1J_PTH_0_60",
    "r_proc_GG2H_1J_PTH_120_200",
    "r_proc_GG2H_1J_PTH_60_120",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_0_60",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_120_200",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_60_120",
    "r_proc_GG2H_GE2J_MJJ_GT350",
    "r_proc_GG2H_PTH_GT200",
    "r_proc_TTH",
    "r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25",
    "r_proc_QQH_GE2J_MJJ_60_120",
    "r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25",
    "r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200",
    "r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25",
    "r_proc_QQH_rest",
    "r_proc_VH_lep_PTV_0_150",
    "r_proc_VH_lep_PTV_GT150",
]

# Compact x-axis labels (strip r_proc_ prefix, replace underscores with spaces)
def short_label(poi):
    return poi.replace("r_proc_", "").replace("_", " ")


# Per-POI scan ranges — must match POI_RANGES_MAP in run_1D_scan.sh.
# Any POI not listed falls back to POI_RANGE_DEFAULT.
POI_RANGE_DEFAULT = (-5, 10)

POI_RANGES = {
    # Tight: high-yield ggH 0j/1j bins
    "r_proc_GG2H_0J_PTH_0_10":                                   (-0.5, 2.5),
    "r_proc_GG2H_0J_PTH_GT10":                                   (0, 2),
    "r_proc_GG2H_1J_PTH_0_60":                                   (-2, 3),
    "r_proc_GG2H_1J_PTH_60_120":                                 (-2, 4),
    "r_proc_GG2H_1J_PTH_120_200":                                (-2, 4),
    # Medium: ggH 2j, high-pTH, TTH
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_0_60":                      (-8, 15),
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_60_120":                    (-5, 10),
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_120_200":                   (-5, 10),
    "r_proc_GG2H_GE2J_MJJ_GT350":                                (-8, 10),
    "r_proc_GG2H_PTH_GT200":                                     (-5, 10),
    "r_proc_TTH":                                                 (-5, 10),
    # Merged QQH bins (VBF + hadronic VH combined)
    "r_proc_QQH_GE2J_MJJ_60_120":                                (-5, 10),
    "r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25":           (-8, 13),
    "r_proc_QQH_rest":                                           (-8, 9),
    "r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25":         (-10, 20),
    "r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200":                      (-5, 13),
    "r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25":           (-8, 15),
    "r_proc_VH_lep_PTV_0_150":                                  (-3, 5),
    "r_proc_VH_lep_PTV_GT150":                                  (-5, 15),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot all 1D likelihood scans as a single subplot figure."
    )
    parser.add_argument(
        "--combine-out-dir",
        default="combine_output",
        help=(
            "Directory containing the MultiDimFit ROOT files. "
            "Defaults to <trial-dir>/combine_output."
        ),
    )
    parser.add_argument(
        "--trial-dir",
        default=TRIAL_DIR_DEFAULT,
        help="Directory containing the MultiDimFit ROOT files (default: trial9).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output image path. "
            "Defaults to <trial-dir>/combine_plots/all_1D_scans.png."
        ),
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=8.0,
        help="Y-axis maximum for -2 Delta ln L (default: 8.0).",
    )
    parser.add_argument(
        "--y-cut",
        type=float,
        default=60.0,
        help="Drop points with 2*deltaNLL above this before spline fit (default: 60.0).",
    )
    parser.add_argument(
        "--n-spline",
        type=int,
        default=1000,
        help="Number of points for spline evaluation (default: 1000).",
    )
    parser.add_argument(
        "--poi-ranges-json",
        default=None,
        help=(
            "JSON string mapping POI name to [lo, hi] range. "
            "Overrides the internal POI_RANGES dict when provided."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: find where spline crosses a y-level
# ---------------------------------------------------------------------------

def find_crossings(poi_vals, spline_vals, level, best_fit):
    lo, hi = None, None
    for i in range(len(poi_vals) - 1):
        y0 = spline_vals[i] - level
        y1 = spline_vals[i + 1] - level
        if y0 * y1 > 0:
            continue
        x0, x1 = poi_vals[i], poi_vals[i + 1]
        x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
        if x_cross <= best_fit:
            lo = x_cross
        else:
            hi = x_cross
    return lo, hi


# ---------------------------------------------------------------------------
# Read one ROOT file → (poi_arr, dnll_arr, best_fit, xs, ys)
# Returns None on failure.
# ---------------------------------------------------------------------------

def read_scan(root_file, poi, y_cut, n_spline, x_lo=None, x_hi=None):
    if not os.path.isfile(root_file):
        print(f"WARNING: ROOT file not found, skipping: {root_file}")
        return None

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        print(f"WARNING: could not open {root_file}")
        return None

    tree = f.Get("limit")
    if tree is None:
        print(f"WARNING: tree 'limit' not found in {root_file}")
        f.Close()
        return None

    poi_list, dnll_list = [], []
    for ev in tree:
        try:
            poi_list.append(getattr(ev, poi))
            dnll_list.append(getattr(ev, "deltaNLL"))
        except AttributeError:
            print(f"WARNING: branch '{poi}' or 'deltaNLL' missing in {root_file}")
            f.Close()
            return None

    f.Close()

    if not poi_list:
        print(f"WARNING: empty tree in {root_file}")
        return None

    poi_arr = np.array(poi_list)
    dnll_arr = 2.0 * np.array(dnll_list)
    dnll_arr -= dnll_arr.min()

    best_idx = int(np.argmin(dnll_arr))
    best_fit = float(poi_arr[best_idx])

    order = np.argsort(poi_arr)
    poi_arr, dnll_arr = poi_arr[order], dnll_arr[order]
    _, unique_idx = np.unique(poi_arr, return_index=True)
    poi_arr, dnll_arr = poi_arr[unique_idx], dnll_arr[unique_idx]

    mask = dnll_arr <= y_cut

    # Edge points near hard scan bounds often contain minimizer pathologies
    # (negative expected yields / PDF underflow) and create artificial jumps.
    # Keep a small interior margin for spline construction.
    if x_lo is not None and x_hi is not None and x_hi > x_lo:
        margin = 0.02 * (x_hi - x_lo)
        mask &= (poi_arr >= (x_lo + margin)) & (poi_arr <= (x_hi - margin))
    if mask.sum() < 4:
        print(f"WARNING: too few points after y_cut for {poi}")
        return None

    poi_fit = poi_arr[mask]
    dnll_fit = dnll_arr[mask]

    spline = UnivariateSpline(poi_fit, dnll_fit, s=0, k=3, ext=3)
    xs = np.linspace(float(poi_fit[0]), float(poi_fit[-1]), n_spline)
    ys = np.clip(spline(xs), 0.0, None)

    return poi_arr, dnll_arr, best_fit, xs, ys, spline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Allow caller to override per-POI ranges (e.g. from run_1D_scan.sh).
    poi_ranges = POI_RANGES.copy()
    if args.poi_ranges_json:
        overrides = json.loads(args.poi_ranges_json)
        poi_ranges.update({k: tuple(v) for k, v in overrides.items()})

    if args.output is None:
        args.output = os.path.join(args.trial_dir, "combine_plots", "all_1D_scans.png")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n_pois = len(POIS)
    n_cols = 4
    n_rows = math.ceil(n_pois / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), sharey=True)
    axes_flat = axes.flatten()

    # Hide unused subplots
    for idx in range(n_pois, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for idx, poi in enumerate(POIS):
        ax = axes_flat[idx]
        short = poi.replace("r_proc_", "").replace("GG2H_", "")

        # build expected ROOT file path
        name_tag = poi.replace("r_proc_", "")
        root_file = os.path.join(
            args.trial_dir,
            args.combine_out_dir,
            f"higgsCombine.scan1D.{name_tag}.MultiDimFit.mH125.38.root",
        )

        # Resolve per-POI x range
        x_lo, x_hi = poi_ranges.get(poi, POI_RANGE_DEFAULT)

        result = read_scan(root_file, poi, args.y_cut, args.n_spline, x_lo=x_lo, x_hi=x_hi)

        if result is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=9)
            ax.set_title(short_label(poi), fontsize=8)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(0.0, args.y_max)
            continue

        poi_arr, dnll_arr, best_fit, xs, ys, spline = result
        # Keep the full scan visible across x-range even when -2DeltaNLL is
        # above the displayed y-axis ceiling.
        dnll_plot = np.minimum(dnll_arr, args.y_max)
        ys_plot = np.minimum(ys, args.y_max)

        lo_1sig, hi_1sig = find_crossings(xs, ys, 1.0, best_fit)
        lo_2sig, hi_2sig = find_crossings(xs, ys, 4.0, best_fit)

        # scan points
        ax.scatter(poi_arr, dnll_plot, s=6, color="black", zorder=3, label="scan")

        # spline
        ax.plot(xs, ys_plot, color="#1f77b4", linewidth=1.8, zorder=4, label="spline")

        # 1σ / 2σ horizontal lines
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(4.0, color="gray", linestyle=":",  linewidth=1.0)

        # vertical crossings
        for x_c in [lo_1sig, hi_1sig]:
            if x_c is not None:
                ax.axvline(x_c, ymax=1.0 / args.y_max, color="gray",
                           linestyle="--", linewidth=0.8)
        for x_c in [lo_2sig, hi_2sig]:
            if x_c is not None:
                ax.axvline(x_c, ymax=4.0 / args.y_max, color="gray",
                           linestyle=":",  linewidth=0.8)

        # best-fit marker
        ax.plot(best_fit, 0.0, marker="D", color="#d62728",
                markersize=6, zorder=5, label=f"best fit")

        # SM marker
        sm_y = float(np.clip(spline(1.0), 0.0, None))
        ax.plot(1.0, sm_y, marker="*", color="#ff7f0e",
                markersize=9, zorder=5, label="SM (r=1)")

        # annotation: best-fit value with 1σ errors
        err_hi = (hi_1sig - best_fit) if hi_1sig is not None else float("nan")
        err_lo = (best_fit - lo_1sig) if lo_1sig is not None else float("nan")
        if not math.isnan(err_hi) and not math.isnan(err_lo):
            ann = f"{best_fit:.2f}$^{{+{err_hi:.2f}}}_{{-{err_lo:.2f}}}$"
            ax.text(0.04, 0.93, ann, transform=ax.transAxes, fontsize=7.5,
                    va="top", ha="left")

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.0, args.y_max)
        ax.set_title(short_label(poi), fontsize=8, pad=3)
        ax.set_xlabel("signal strength", fontsize=7)
        ax.tick_params(labelsize=7)

    # shared y-axis label on left column only
    for row_ax in axes[:, 0]:
        row_ax.set_ylabel(r"$-2\,\Delta\ln\mathcal{L}$", fontsize=9)

    # legend on first subplot
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[0].legend(handles, labels, fontsize=6.5, loc="upper right",
                        framealpha=0.7)

    fig.suptitle(
        "CMS Internal — 1D likelihood scans (STXS bins, QQH-merged)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
