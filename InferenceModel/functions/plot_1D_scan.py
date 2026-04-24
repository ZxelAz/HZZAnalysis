#!/usr/bin/env python3
"""
Plot a 1D likelihood scan from a Combine MultiDimFit --algo grid output.

Typical usage
-------------
# Run scan first (one POI at a time):
#   combine -M MultiDimFit datacard_workspace.root -m 125.38 \
#       --freezeParameters MH --algo grid --points 200 \
#       -P r_proc_GG2H_0J_PTH_0_10 \
#       --setParameterRanges r_proc_GG2H_0J_PTH_0_10=0,2 \
#       -n .scan1D.GG2H_0J_PTH_0_10

# Then plot:
#   python3 plot_1D_scan.py \
#       --input higgsCombine.scan1D.GG2H_0J_PTH_0_10.MultiDimFit.mH125.38.root \
#       --poi r_proc_GG2H_0J_PTH_0_10 \
#       --output results/trial9/fit_plots/scan1D_GG2H_0J_PTH_0_10.png
"""

import argparse
import math
import os

import numpy as np
import ROOT
from scipy.interpolate import UnivariateSpline

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 1D likelihood scan from Combine MultiDimFit --algo grid output."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input ROOT file produced by combine (-M MultiDimFit --algo grid).",
    )
    parser.add_argument(
        "--poi",
        required=True,
        help="POI branch name to scan (e.g. r_proc_GG2H_0J_PTH_0_10).",
    )
    parser.add_argument(
        "--tree",
        default="limit",
        help="TTree name in the input file (default: limit).",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="X axis minimum. Auto-detected from data if not given.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="X axis maximum. Auto-detected from data if not given.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=8.0,
        help="Y axis maximum for -2DeltaLogL (default: 8.0).",
    )
    parser.add_argument(
        "--y-cut",
        type=float,
        default=7.0,
        help="Drop points with 2*deltaNLL above this value before spline fit (default: 7.0).",
    )
    parser.add_argument(
        "--n-spline",
        type=int,
        default=1000,
        help="Number of evaluation points for the interpolating spline (default: 1000).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output image path (.png or .pdf). "
            "Defaults to scan1D_<poi>.png next to the input file."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: find where spline crosses a y-level
# ---------------------------------------------------------------------------

def find_crossings(poi_vals, spline_vals, level, best_fit):
    """
    Return (lo, hi) where the spline crosses `level`.
    poi_vals and spline_vals must be sorted ascending by poi_vals.
    Returns None for a boundary if no crossing is found on that side.
    """
    lo = None
    hi = None
    for i in range(len(poi_vals) - 1):
        y0, y1 = spline_vals[i] - level, spline_vals[i + 1] - level
        if y0 * y1 > 0:
            continue  # same sign, no crossing
        # linear interpolation
        x0, x1 = poi_vals[i], poi_vals[i + 1]
        x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
        if x_cross <= best_fit:
            lo = x_cross
        else:
            hi = x_cross
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- default output path -----------------------------------------------
    if args.output is None:
        out_dir = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(out_dir, f"scan1D_{args.poi}.png")

    # ---- open ROOT file and read tree --------------------------------------
    f_in = ROOT.TFile.Open(args.input)
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"Could not open input file: {args.input}")

    tree = f_in.Get(args.tree)
    if tree is None:
        raise RuntimeError(f"Tree '{args.tree}' not found in {args.input}")

    poi_list, dnll_list = [], []
    for ev in tree:
        if not hasattr(ev, args.poi) or not hasattr(ev, "deltaNLL"):
            raise RuntimeError(
                f"Branch '{args.poi}' or 'deltaNLL' not found in tree. "
                f"Check the POI name matches the scan output."
            )
        poi_list.append(getattr(ev, args.poi))
        dnll_list.append(getattr(ev, "deltaNLL"))

    if not poi_list:
        raise RuntimeError("No entries found in tree.")

    poi_arr = np.array(poi_list)
    dnll_arr = 2.0 * np.array(dnll_list)

    # ---- shift so minimum is 0 ---------------------------------------------
    min_dnll = dnll_arr.min()
    dnll_arr -= min_dnll

    # ---- best fit ----------------------------------------------------------
    best_idx = int(np.argmin(dnll_arr))
    best_fit = float(poi_arr[best_idx])

    # ---- remove duplicates and sort ----------------------------------------
    order = np.argsort(poi_arr)
    poi_arr, dnll_arr = poi_arr[order], dnll_arr[order]
    _, unique_idx = np.unique(poi_arr, return_index=True)
    poi_arr, dnll_arr = poi_arr[unique_idx], dnll_arr[unique_idx]

    # ---- apply y-cut -------------------------------------------------------
    mask = dnll_arr <= args.y_cut
    poi_fit = poi_arr[mask]
    dnll_fit = dnll_arr[mask]

    # ---- spline interpolation ----------------------------------------------
    spline = UnivariateSpline(poi_fit, dnll_fit, s=0, k=3, ext=3)
    x_min_data = float(poi_fit[0])
    x_max_data = float(poi_fit[-1])
    x_lo = args.x_min if args.x_min is not None else x_min_data
    x_hi = args.x_max if args.x_max is not None else x_max_data
    xs = np.linspace(x_lo, x_hi, args.n_spline)
    ys = np.clip(spline(xs), 0.0, None)

    # ---- compute 1σ and 2σ crossings ---------------------------------------
    lo_1sig, hi_1sig = find_crossings(xs, ys, 1.0, best_fit)
    lo_2sig, hi_2sig = find_crossings(xs, ys, 4.0, best_fit)

    err_hi_1sig = (hi_1sig - best_fit) if hi_1sig is not None else float("nan")
    err_lo_1sig = (lo_1sig - best_fit) if lo_1sig is not None else float("nan")
    err_hi_2sig = (hi_2sig - best_fit) if hi_2sig is not None else float("nan")
    err_lo_2sig = (lo_2sig - best_fit) if lo_2sig is not None else float("nan")

    print(f"\n{'='*60}")
    print(f"  POI : {args.poi}")
    print(f"  Best fit : {best_fit:.4f}")
    if not math.isnan(err_hi_1sig) and not math.isnan(err_lo_1sig):
        print(f"  1σ  :  {best_fit:.3f}  +{err_hi_1sig:.3f}  {err_lo_1sig:.3f}")
    if not math.isnan(err_hi_2sig) and not math.isnan(err_lo_2sig):
        print(f"  2σ  :  {best_fit:.3f}  +{err_hi_2sig:.3f}  {err_lo_2sig:.3f}")
    print(f"{'='*60}\n")

    # ---- ROOT graph and spline TF1 -----------------------------------------
    n_pts = len(poi_arr)
    g_scan = ROOT.TGraph(n_pts)
    for i in range(n_pts):
        g_scan.SetPoint(i, float(poi_arr[i]), float(dnll_arr[i]))
    g_scan.SetMarkerStyle(20)
    g_scan.SetMarkerSize(0.5)
    g_scan.SetMarkerColor(ROOT.kBlack)

    n_spline = len(xs)
    g_spline = ROOT.TGraph(n_spline)
    for i in range(n_spline):
        g_spline.SetPoint(i, float(xs[i]), float(ys[i]))
    g_spline.SetLineColor(ROOT.kBlue + 1)
    g_spline.SetLineWidth(3)

    # ---- canvas ------------------------------------------------------------
    canv = ROOT.TCanvas("canv", "canv", 800, 600)
    canv.SetTickx()
    canv.SetTicky()
    canv.SetLeftMargin(0.12)
    canv.SetBottomMargin(0.13)
    canv.SetRightMargin(0.04)
    canv.SetTopMargin(0.07)

    # draw scan points first to set axis
    g_scan.Draw("AP")
    g_scan.SetTitle("")
    g_scan.GetXaxis().SetTitle(args.poi)
    g_scan.GetXaxis().SetTitleSize(0.045)
    g_scan.GetXaxis().SetTitleOffset(1.1)
    g_scan.GetYaxis().SetTitle("-2 #Delta ln L")
    g_scan.GetYaxis().SetTitleSize(0.045)
    g_scan.GetYaxis().SetTitleOffset(1.1)
    g_scan.GetYaxis().SetRangeUser(0.0, args.y_max)
    g_scan.GetXaxis().SetRangeUser(x_lo, x_hi)

    # overlay spline
    g_spline.Draw("L SAME")

    # draw 1σ / 2σ horizontal lines
    hline1 = ROOT.TLine(x_lo, 1.0, x_hi, 1.0)
    hline1.SetLineColor(ROOT.kGray + 2)
    hline1.SetLineStyle(2)
    hline1.SetLineWidth(2)
    hline1.Draw("same")

    hline4 = ROOT.TLine(x_lo, 4.0, x_hi, 4.0)
    hline4.SetLineColor(ROOT.kGray + 2)
    hline4.SetLineStyle(3)
    hline4.SetLineWidth(2)
    hline4.Draw("same")

    # vertical lines at crossings
    vline_color = ROOT.kGray + 2
    def vline(x, y_top):
        vl = ROOT.TLine(x, 0.0, x, y_top)
        vl.SetLineColor(vline_color)
        vl.SetLineStyle(2)
        vl.Draw("same")
        return vl  # keep reference alive

    vlines = []
    if lo_1sig is not None:
        vlines.append(vline(lo_1sig, 1.0))
    if hi_1sig is not None:
        vlines.append(vline(hi_1sig, 1.0))
    if lo_2sig is not None:
        vlines.append(vline(lo_2sig, 4.0))
    if hi_2sig is not None:
        vlines.append(vline(hi_2sig, 4.0))

    # best-fit marker
    g_bf = ROOT.TGraph(1)
    g_bf.SetPoint(0, best_fit, 0.0)
    g_bf.SetMarkerStyle(34)
    g_bf.SetMarkerSize(2.0)
    g_bf.SetMarkerColor(ROOT.kRed + 1)
    g_bf.Draw("P SAME")

    # SM (r=1) marker
    g_sm = ROOT.TGraph(1)
    g_sm.SetPoint(0, 1.0, float(spline(1.0)))
    g_sm.SetMarkerStyle(29)
    g_sm.SetMarkerSize(2.0)
    g_sm.SetMarkerColor(ROOT.kOrange + 1)
    g_sm.Draw("P SAME")

    # result text
    txt_1sig = ""
    if not math.isnan(err_hi_1sig) and not math.isnan(err_lo_1sig):
        txt_1sig = (
            f"{args.poi} = {best_fit:.3f}"
            f" ^{{+{err_hi_1sig:.3f}}}_{{-{abs(err_lo_1sig):.3f}}}"
        )

    pt = ROOT.TPaveText(0.14, 0.80, 0.80, 0.92, "NDCNB")
    pt.SetTextAlign(12)
    pt.SetTextFont(42)
    pt.SetTextSize(0.032)
    pt.SetFillStyle(0)
    pt.SetBorderSize(0)
    if txt_1sig:
        pt.AddText(txt_1sig)
    pt.Draw()

    # legend
    leg = ROOT.TLegend(0.55, 0.65, 0.94, 0.78)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetTextSize(0.032)
    leg.AddEntry(g_spline, "Spline", "L")
    leg.AddEntry(g_bf, "Best fit", "P")
    leg.AddEntry(g_sm, "SM (r=1)", "P")
    leg.Draw()

    # CMS label
    cms = ROOT.TLatex()
    cms.SetNDC()
    cms.SetTextFont(62)
    cms.SetTextSize(0.05)
    cms.DrawLatex(0.12, 0.945, "CMS")
    cms_sub = ROOT.TLatex()
    cms_sub.SetNDC()
    cms_sub.SetTextFont(52)
    cms_sub.SetTextSize(0.038)
    cms_sub.DrawLatex(0.20, 0.948, "Internal")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    canv.SaveAs(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
