#!/usr/bin/env python3
"""
Plot 68% CL uncertainty bars for each signal process, comparing three
categorisation methods (cut-based, BDT, GATO) on a single panel.

Each process gets a y-position; the three methods are drawn as horizontal
error bars offset vertically within that slot, styled after CMS HIG-21-018
Figure 14.

Usage
-----
python plot_rate_comparison.py \
    --cutbased  Results/CutBased/combine_output_3 \
    --bdt       Results/BDT/combine_output_4 \
    --gato      Results/GATO/combine_output_4 \
    --output    results/comparison/rate_comparison.png
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import ROOT
from scipy.interpolate import UnivariateSpline

ROOT.gROOT.SetBatch(True)

# ---------------------------------------------------------------------------
# POI list and display labels
# ---------------------------------------------------------------------------

POIS = [
    "r_proc_GG2H_0J_PTH_0_10",
    "r_proc_GG2H_0J_PTH_GT10",
    "r_proc_GG2H_1J_PTH_0_60",
    "r_proc_GG2H_1J_PTH_60_120",
    "r_proc_GG2H_1J_PTH_120_200",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_0_60",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_60_120",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_120_200",
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

LABELS = {
    "r_proc_GG2H_0J_PTH_0_10":                           r"ggH 0j, $p_T^H$ [0,10]",
    "r_proc_GG2H_0J_PTH_GT10":                           r"ggH 0j, $p_T^H > 10$",
    "r_proc_GG2H_1J_PTH_0_60":                           r"ggH 1j, $p_T^H$ [0,60]",
    "r_proc_GG2H_1J_PTH_60_120":                         r"ggH 1j, $p_T^H$ [60,120]",
    "r_proc_GG2H_1J_PTH_120_200":                        r"ggH 1j, $p_T^H$ [120,200]",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_0_60":               r"ggH $\geq$2j, $m_{jj}\!<\!350$, $p_T^H$ [0,60]",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_60_120":             r"ggH $\geq$2j, $m_{jj}\!<\!350$, $p_T^H$ [60,120]",
    "r_proc_GG2H_GE2J_MJJ_0_350_PTH_120_200":            r"ggH $\geq$2j, $m_{jj}\!<\!350$, $p_T^H$ [120,200]",
    "r_proc_GG2H_GE2J_MJJ_GT350":                        r"ggH $\geq$2j, $m_{jj}\!>\!350$",
    "r_proc_GG2H_PTH_GT200":                              r"ggH $p_T^H > 200$",
    "r_proc_TTH":                                         r"ttH",
    "r_proc_QQH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25":  r"qqH $m_{jj}$ [350,700], $p_{T,Hjj}\!<\!25$",
    "r_proc_QQH_GE2J_MJJ_60_120":                        r"qqH $m_{jj}$ [60,120]",
    "r_proc_QQH_GE2J_MJJ_GT350_PTH_0_200_PTHJJ_GT25":    r"qqH $m_{jj}\!>\!350$, $p_{T,Hjj}\!>\!25$",
    "r_proc_QQH_GE2J_MJJ_GT350_PTH_GT200":               r"qqH $m_{jj}\!>\!350$, $p_T^H\!>\!200$",
    "r_proc_QQH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25":    r"qqH $m_{jj}\!>\!700$, $p_{T,Hjj}\!<\!25$",
    "r_proc_QQH_rest":                                    r"qqH rest",
    "r_proc_VH_lep_PTV_0_150":                           r"VH lep $p_T^V$ [0,150]",
    "r_proc_VH_lep_PTV_GT150":                           r"VH lep $p_T^V > 150$",
}

# ---------------------------------------------------------------------------
# Helpers
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


def read_scan(root_file, poi, y_cut=60.0, n_spline=2000):
    """
    Read a MultiDimFit ROOT file and return
    (best_fit, err_lo_68, err_hi_68, err_lo_95, err_hi_95).
    err_lo values are negative (distance below best-fit).
    95% CL values are nan when the spline does not reach Δ2NLL=3.84.
    Returns None on any failure.
    """
    if not os.path.isfile(root_file):
        return None

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        return None

    tree = f.Get("limit")
    if tree is None:
        f.Close()
        return None

    poi_list, dnll_list = [], []
    for ev in tree:
        try:
            poi_list.append(getattr(ev, poi))
            dnll_list.append(getattr(ev, "deltaNLL"))
        except AttributeError:
            f.Close()
            return None

    f.Close()

    if len(poi_list) < 4:
        return None

    poi_arr  = np.array(poi_list)
    dnll_arr = 2.0 * np.array(dnll_list)
    dnll_arr -= dnll_arr.min()

    best_idx = int(np.argmin(dnll_arr))
    best_fit = float(poi_arr[best_idx])

    order = np.argsort(poi_arr)
    poi_arr, dnll_arr = poi_arr[order], dnll_arr[order]
    _, unique_idx = np.unique(poi_arr, return_index=True)
    poi_arr, dnll_arr = poi_arr[unique_idx], dnll_arr[unique_idx]

    mask = dnll_arr <= y_cut
    if mask.sum() < 4:
        return None

    poi_fit  = poi_arr[mask]
    dnll_fit = dnll_arr[mask]

    try:
        spline = UnivariateSpline(poi_fit, dnll_fit, s=0, k=3, ext=3)
    except Exception:
        return None

    xs = np.linspace(float(poi_fit[0]), float(poi_fit[-1]), n_spline)
    ys = np.clip(spline(xs), 0.0, None)

    lo_1sig, hi_1sig = find_crossings(xs, ys, 1.0, best_fit)
    lo_2sig, hi_2sig = find_crossings(xs, ys, 3.84, best_fit)

    err_hi_68 = (hi_1sig - best_fit) if hi_1sig is not None else float("nan")
    err_lo_68 = (lo_1sig - best_fit) if lo_1sig is not None else float("nan")  # negative
    err_hi_95 = (hi_2sig - best_fit) if hi_2sig is not None else float("nan")
    err_lo_95 = (lo_2sig - best_fit) if lo_2sig is not None else float("nan")  # negative

    return best_fit, err_lo_68, err_hi_68, err_lo_95, err_hi_95


def root_file_for_poi(scan_dir, poi):
    """Return the expected ROOT file path for a POI in a combine output dir."""
    proc = poi.replace("r_proc_", "")
    fname = f"higgsCombine.scan1D.{proc}.MultiDimFit.mH125.38.root"
    return os.path.join(scan_dir, fname)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(cutbased_dir, bdt_dir, gato_dir, output, x_lo, x_hi):

    methods = [
        ("Cut-based", cutbased_dir, "#E41A1C"),   # red
        ("BDT",       bdt_dir,      "#377EB8"),   # blue
        ("GATO",      gato_dir,     "#4DAF4A"),   # green
    ]

    n_pois  = len(POIS)
    # Vertical spacing: each POI gets 1 unit; within that 3 methods are
    # offset by ±spacing so they don't overlap.
    dy = 0.22   # offset between methods within one POI slot

    fig_height = max(8, n_pois * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    offsets  = [dy, 0.0, -dy]   # cutbased top, BDT middle, GATO bottom

    for poi_idx, poi in enumerate(POIS):
        y_center = n_pois - 1 - poi_idx   # plot top-to-bottom

        for (label, scan_dir, color), offset in zip(methods, offsets):
            root_file = root_file_for_poi(scan_dir, poi)
            result = read_scan(root_file, poi)

            y_pos = y_center + offset

            if result is None:
                ax.plot(1.0, y_pos, marker="x", color=color,
                        markersize=6, zorder=3)
                continue

            best_fit, err_lo_68, err_hi_68, err_lo_95, err_hi_95 = result

            # --- 95% CL: thin line, no caps, drawn behind the 68% bar
            lo95 = abs(err_lo_95) if not math.isnan(err_lo_95) else None
            hi95 = err_hi_95      if not math.isnan(err_hi_95) else None
            if lo95 is not None or hi95 is not None:
                ax.errorbar(
                    best_fit, y_pos,
                    xerr=[[lo95 if lo95 is not None else 0.0],
                          [hi95 if hi95 is not None else 0.0]],
                    fmt="none",
                    color=color,
                    linewidth=0.9,
                    capsize=2,
                    capthick=0.9,
                    alpha=0.5,
                    zorder=2,
                )

            # --- 68% CL: thick line with centre marker, drawn on top
            lo68 = abs(err_lo_68) if not math.isnan(err_lo_68) else 0.0
            hi68 = err_hi_68      if not math.isnan(err_hi_68) else 0.0

            ax.errorbar(
                best_fit, y_pos,
                xerr=[[lo68], [hi68]],
                fmt="o",
                color=color,
                markersize=5,
                linewidth=1.8,
                capsize=3,
                capthick=1.5,
                zorder=3,
            )

        # light horizontal separator between POI groups
        if poi_idx < n_pois - 1:
            ax.axhline(y_center - 0.5, color="lightgrey", linewidth=0.5, zorder=1)

    # --- SM expectation line
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", zorder=2)

    # --- axes
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.5, n_pois - 0.5)
    ax.set_yticks(range(n_pois))
    ax.set_yticklabels(
        [LABELS.get(poi, poi.replace("r_proc_", "").replace("_", " "))
         for poi in reversed(POIS)],
        fontsize=9,
    )
    ax.set_xlabel(r"Best-fit $\hat{\mu}$ (signal strength)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)

    # --- legend
    method_handles = [
        mlines.Line2D([], [], color=color, marker="o", markersize=6,
                      linewidth=1.8, label=label)
        for label, _, color in methods
    ]
    cl_handles = [
        mlines.Line2D([], [], color="black", marker="o", markersize=5,
                      linewidth=1.8, label="68% CL"),
        mlines.Line2D([], [], color="black", marker="none",
                      linewidth=0.9, alpha=0.5, label="95% CL"),
    ]
    ax.legend(handles=method_handles + cl_handles,
              loc="upper right", fontsize=9, framealpha=0.9)

    # --- CMS label
    ax.text(0.02, 1.01, "CMS", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")
    ax.text(0.10, 1.01, "Simulation Preliminary", transform=ax.transAxes,
            fontsize=10, fontstyle="italic", va="bottom")
    ax.text(0.98, 1.01, r"$H \to ZZ \to 4\ell$, 138 fb$^{-1}$",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom")

    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    import os
    BASE = os.environ.get("HZZ_ROOT", "/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis")
    RESULTS = f"{BASE}/InferenceModel/Results"
    parser = argparse.ArgumentParser(
        description="Compare 68%% CL signal-strength uncertainties across categorisation methods."
    )
    parser.add_argument("--cutbased", default=f"{RESULTS}/CutBased/combine_output",
                        help="Combine output dir for cut-based categorisation")
    parser.add_argument("--bdt",      default=f"{RESULTS}/BDT/combine_output",
                        help="Combine output dir for BDT categorisation")
    parser.add_argument("--gato",     default=f"{RESULTS}/GATO/combine_output",
                        help="Combine output dir for GATO categorisation")
    parser.add_argument("--output",   default=f"{BASE}/InferenceModel/Plots/comparison/rate_comparison.png",
                        help="Output plot path")
    parser.add_argument("--x-lo", type=float, default=-3.0,
                        help="X-axis lower limit (default: -3)")
    parser.add_argument("--x-hi", type=float, default=5.0,
                        help="X-axis upper limit (default: 5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_comparison(
        cutbased_dir=args.cutbased,
        bdt_dir=args.bdt,
        gato_dir=args.gato,
        output=args.output,
        x_lo=args.x_lo,
        x_hi=args.x_hi,
    )
