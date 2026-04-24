import argparse
import os

import numpy as np
import ROOT
from scipy.interpolate import griddata

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot 2D likelihood scan from Combine MultiDimFit output.")
    parser.add_argument(
        "--input",
        default="${HZZ_ROOT}/InferenceModel/results/trial9/higgsCombine.fitresult.MultiDimFit.mH125.38.root",
        help="Input ROOT file produced by combine.",
    )
    parser.add_argument(
        "--tree",
        default="limit",
        help="TTree name in input file.",
    )
    parser.add_argument(
        "--x-poi",
        default="r_proc_GG2H_0J_PTH_0_10",
        help="POI branch name for X axis.",
    )
    parser.add_argument(
        "--y-poi",
        default="r_proc_GG2H_0J_PTH_GT10",
        help="POI branch name for Y axis.",
    )
    parser.add_argument("--x-min", type=float, default=0.0, help="X axis minimum.")
    parser.add_argument("--x-max", type=float, default=2.0, help="X axis maximum.")
    parser.add_argument("--y-min", type=float, default=0.0, help="Y axis minimum.")
    parser.add_argument("--y-max", type=float, default=2.0, help="Y axis maximum.")
    parser.add_argument("--n-points", type=int, default=1000, help="Interpolation grid points per axis.")
    parser.add_argument("--n-bins", type=int, default=40, help="Plot bin count per axis.")
    parser.add_argument(
        "--z-max",
        type=float,
        default=25.0,
        help="Maximum value shown on -2DeltaLogL color scale.",
    )
    parser.add_argument(
        "--output",
        default="${HZZ_ROOT}/InferenceModel/results/trial9/fit_plots/scan2D_r_proc_GG2H_0J_PTH_0_10_vs_r_proc_GG2H_0J_PTH_GT10.png",
        help="Output image path (.png or .pdf).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    x_range = [args.x_min, args.x_max]
    y_range = [args.y_min, args.y_max]

    f_in = ROOT.TFile.Open(args.input)
    if not f_in or f_in.IsZombie():
        raise RuntimeError(f"Could not open input file: {args.input}")

    t = f_in.Get(args.tree)
    if t is None:
        raise RuntimeError(f"Tree '{args.tree}' was not found in {args.input}")

    x_vals, y_vals, delta_nll_vals = [], [], []
    for ev in t:
        if not hasattr(ev, args.x_poi) or not hasattr(ev, args.y_poi) or not hasattr(ev, "deltaNLL"):
            continue
        x_vals.append(getattr(ev, args.x_poi))
        y_vals.append(getattr(ev, args.y_poi))
        delta_nll_vals.append(getattr(ev, "deltaNLL"))

    if not x_vals:
        raise RuntimeError(
            f"No valid points found. Check POIs '{args.x_poi}', '{args.y_poi}' and input tree contents."
        )

    points = np.array([x_vals, y_vals]).transpose()
    dnll = np.asarray(delta_nll_vals)

    grid_x, grid_y = np.mgrid[
        x_range[0] : x_range[1] : args.n_points * 1j,
        y_range[0] : y_range[1] : args.n_points * 1j,
    ]
    grid_vals = griddata(points, dnll, (grid_x, grid_y), method="cubic")

    valid = np.isfinite(grid_vals)
    grid_x = grid_x[valid]
    grid_y = grid_y[valid]
    grid_vals = grid_vals[valid]

    if grid_vals.size == 0:
        raise RuntimeError("Interpolation returned no finite values. Increase scan points or adjust ranges.")

    h2d = ROOT.TProfile2D("h2d", "h2d", args.n_bins, x_range[0], x_range[1], args.n_bins, y_range[0], y_range[1])
    for i in range(grid_vals.size):
        h2d.Fill(float(grid_x[i]), float(grid_y[i]), 2.0 * float(grid_vals[i]))

    for ibin in range(1, h2d.GetNbinsX() + 1):
        for jbin in range(1, h2d.GetNbinsY() + 1):
            if h2d.GetBinContent(ibin, jbin) == 0:
                xc = h2d.GetXaxis().GetBinCenter(ibin)
                yc = h2d.GetYaxis().GetBinCenter(jbin)
                h2d.Fill(xc, yc, 999)

    canv = ROOT.TCanvas("canv", "canv", 700, 650)
    canv.SetTickx()
    canv.SetTicky()
    canv.SetLeftMargin(0.12)
    canv.SetBottomMargin(0.12)

    xw = (x_range[1] - x_range[0]) / args.n_bins
    yw = (y_range[1] - y_range[0]) / args.n_bins

    h2d.SetContour(999)
    h2d.SetTitle("")
    h2d.GetXaxis().SetTitle(args.x_poi)
    h2d.GetXaxis().SetTitleSize(0.045)
    h2d.GetXaxis().SetTitleOffset(1.0)
    h2d.GetXaxis().SetRangeUser(x_range[0], x_range[1] - xw)

    h2d.GetYaxis().SetTitle(args.y_poi)
    h2d.GetYaxis().SetTitleSize(0.045)
    h2d.GetYaxis().SetTitleOffset(1.0)
    h2d.GetYaxis().SetRangeUser(y_range[0], y_range[1] - yw)

    h2d.GetZaxis().SetTitle("-2 #Delta ln L")
    h2d.GetZaxis().SetTitleSize(0.045)
    h2d.GetZaxis().SetTitleOffset(0.95)
    h2d.SetMaximum(args.z_max)

    c68 = h2d.Clone("c68")
    c95 = h2d.Clone("c95")
    c68.SetContour(2)
    c68.SetContourLevel(1, 2.30)
    c68.SetLineWidth(3)
    c68.SetLineColor(ROOT.kBlack)
    c95.SetContour(2)
    c95.SetContourLevel(1, 5.99)
    c95.SetLineWidth(3)
    c95.SetLineStyle(2)
    c95.SetLineColor(ROOT.kBlack)

    h2d.Draw("COLZ")

    vline = ROOT.TLine(1.0, y_range[0], 1.0, y_range[1] - yw)
    vline.SetLineColorAlpha(ROOT.kGray + 2, 0.6)
    vline.Draw("same")
    hline = ROOT.TLine(x_range[0], 1.0, x_range[1] - xw, 1.0)
    hline.SetLineColorAlpha(ROOT.kGray + 2, 0.6)
    hline.Draw("same")

    c68.Draw("cont3same")
    c95.Draw("cont3same")

    best_idx = int(np.argmin(grid_vals))
    g_bf = ROOT.TGraph()
    g_bf.SetPoint(0, float(grid_x[best_idx]), float(grid_y[best_idx]))
    g_bf.SetMarkerStyle(34)
    g_bf.SetMarkerSize(2)
    g_bf.SetMarkerColor(ROOT.kBlack)
    g_bf.Draw("P")

    g_sm = ROOT.TGraph()
    g_sm.SetPoint(0, 1.0, 1.0)
    g_sm.SetMarkerStyle(33)
    g_sm.SetMarkerSize(2)
    g_sm.SetMarkerColor(ROOT.kRed + 1)
    g_sm.Draw("P")

    leg = ROOT.TLegend(0.58, 0.67, 0.86, 0.87)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.AddEntry(g_bf, "Best fit", "P")
    leg.AddEntry(c68, "1sigma CL", "L")
    leg.AddEntry(c95, "2sigma CL", "L")
    leg.AddEntry(g_sm, "SM (1,1)", "P")
    leg.Draw()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    canv.SaveAs(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
