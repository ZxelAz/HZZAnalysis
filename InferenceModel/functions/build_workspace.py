#!/usr/bin/env python3
"""
build_workspace.py

Build a RooWorkspace for CMS Combine parametric shape analysis from:
    - DSCB signal PDF parameters  (RooFitResult objects in fit_results.root)
    - Efficiency × acceptance fractions (epsilonA.pkl)
    - Chebychev background shape fits (fit_results.root)
    - Filtered data ROOT files (for building data_obs)

Workspace structure
-------------------
Observable:
    ZZCand_mass (or --variable) in [105, 140] by default

Signal strengths (floating POIs, initial value = 1):
  mu_{stxs_bin}   -- one per STXS bin, shared across all channels

Per (final_state × STXS bin):
  dscb_{fs}_{bin}           -- RooCrystalBall with all parameters fixed to fit results

Nuisance parameters:
  theta_lumi                -- Gaussian-constrained lumi nuisance, N(0,1)
  lumi_obs                  -- global observable for theta_lumi constraint (fixed at 0)
  lumi                      -- RooFormulaVar = lumi_nominal * (1 + sigma_lumi * theta_lumi)

Per (final_state × category × STXS bin):
    n_signal_{fs}_{cat}_{bin} -- fixed expected yield = epsilonA * XS_{bin} * BR * Lumi
    signal_{fs}_{cat}_{bin}   -- channel/process-specific signal PDF cloned from dscb_{fs}_{bin}

Observed data:
    data_obs                  -- RooDataSet built from ZZCand_mass in the four filtered data files

Per (final_state × bkg_mode):  bkg_mode in {ZZTo4l, ggZZ, ZX}
    bkg_{fs}_{mode}           -- Chebychev PDF (ZZTo4l/ggZZ; params a0,a1,a2) or Landau PDF (ZX; params mean,sigma),
                                 all parameters fixed to fit results (non-extended)

Combine datacard shapes lines (per STXS bin):
    shapes signal {fs}_{cat}_{bin} workspace.root w:signal_{fs}_{cat}_{bin}
    shapes qqZZ   {fs}_{cat}_{bin} workspace.root w:bkg_{fs}_ZZTo4l
    shapes ggZZ   {fs}_{cat}_{bin} workspace.root w:bkg_{fs}_ggZZ

Usage:
cd /afs/cern.ch/work/z/zhiheng/Thesis && source gato-hep/.venv/bin/activate && python inferenceModel/functions/build_workspace.py \
    --fit_results InferenceModel/Results/BDT/fit_results/fit_results.root \
    --epsilonA    InferenceModel/Results/BDT/epsilonA.pkl \
    --output      InferenceModel/Results/BDT/workspace.root 2>&1
"""

import os
import pickle
import argparse

import numpy as np
import ROOT


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def safe(s: str) -> str:
    """Convert an arbitrary string to a valid RooFit / C++ identifier."""
    return (
        str(s)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "p")
    )


def _import(workspace, obj, *cmd_args):
    """Call workspace.import() -- the name is a Python keyword, so use getattr."""
    getattr(workspace, "import")(obj, *cmd_args)


def require_params(params: dict, names: tuple[str, ...], context: str) -> list[float]:
    """Return required fit parameters or raise with a clear context message."""
    missing = [name for name in names if name not in params]
    if missing:
        raise KeyError(
            f"Missing required fit parameters {missing} for {context}. "
            f"Available parameters: {sorted(params)}"
        )
    return [params[name] for name in names]


def stxs_bin_to_xs_key(stxs_bin: str) -> str:
    """Map an STXS bin label to the inclusive production-mode XS key."""
    name = str(stxs_bin).upper()

    if name.startswith("GG2H"):
        return "ggH"
    if name.startswith("VBF"):
        return "VBF"
    if name.startswith("WPLUS") or name.startswith("WMINUS") or name.startswith("WH"):
        return "WH"
    if name.startswith("ZH") or name.startswith("ZHADH"):
        return "ZH"
    if name.startswith("TTH"):
        return "ttH"

    raise ValueError(
        f"Cannot map STXS bin '{stxs_bin}' to XS key. "
        "Please extend stxs_bin_to_xs_key() for this naming pattern."
    )

XS = {"ggH": 51.96,
      "VBF": 4.067,
      "WH": 1.443,
      "ZH": 0.9361,
      "ttH": 0.5634} # in pb from https://arxiv.org/pdf/2402.09955

BR = 1.251*10**-4

Lumi = 62.34*1000  # in pb^-1, for scaling yields to expected event counts

DATA_FILES = {
    "2022preEE": "/eos/user/z/zhiheng/STXS_samples/2022Data/filtered_2022.root",
    "2022postEE": "/eos/user/z/zhiheng/STXS_samples/2022Data/filtered_2022EE.root",
    "2023preBPix": "/eos/user/z/zhiheng/STXS_samples/2023Data/filtered_2023preBPix.root",
    "2023postBPix": "/eos/user/z/zhiheng/STXS_samples/2023Data/filtered_2023postBPix.root",
}

# ---------------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------------

def build_workspace(
    fit_results_path: str,
    epsilonA_path: str,
    output_path: str,
    variable: str = "ZZCand_mass",
    mass_lo: float = 105.0,
    mass_hi: float = 140.0,
) -> None:
    # ------------------------------------------------------------------ load
    with open(epsilonA_path, "rb") as fh:
        epsilonA: dict = pickle.load(fh)

    final_states = sorted({k[0] for k in epsilonA})
    categories   = sorted({k[1] for k in epsilonA})
    stxs_bins    = sorted({k[2] for k in epsilonA})

    print(f"Final states : {final_states}")
    print(f"Categories   : {len(categories)}")
    print(f"STXS bins    : {len(stxs_bins)}")
    print(f"epsilonA entries: {len(epsilonA)}  (non-zero: {sum(v > 0 for v in epsilonA.values())})")

    fit_file = ROOT.TFile.Open(fit_results_path)
    if not fit_file or fit_file.IsZombie():
        raise RuntimeError(f"Cannot open {fit_results_path}")
    all_fit_keys = {k.GetName() for k in fit_file.GetListOfKeys()}

    # ------------------------------------------------------------------ workspace
    w = ROOT.RooWorkspace("w", "HZZ STXS parametric workspace")

    # ------------------------------------------------------------------ signal strengths
    print("\nCreating signal strength parameters (mu)...")
    for stxs_bin in stxs_bins:
        mu_name = f"mu_{safe(stxs_bin)}"
        mu = ROOT.RooRealVar(mu_name, f"Signal strength {stxs_bin}", 1.0, -5.0, 20.0)
        _import(w, mu)
    print(f"  Created {len(stxs_bins)} mu parameters.")

    print("\nCreating per-bin XS*BR normalization constants...")
    for stxs_bin in stxs_bins:
        bin_s = safe(stxs_bin)
        xs_key = stxs_bin_to_xs_key(stxs_bin)
        xsbr_val = XS[xs_key] * BR
        xsbr_name = f"xsbr_{bin_s}"
        w.factory(f"{xsbr_name}[{xsbr_val}]")
        w.var(xsbr_name).setConstant(True)
    print(f"  Created {len(stxs_bins)} xsbr constants.")

    # mass variable
    mass_var = ROOT.RooRealVar(variable, variable, mass_lo, mass_hi)
    _import(w, mass_var)


    # Luminosity
    sigma_lumi = 0.025  # 2.5 % relative uncertainty (dummy)
    w.factory(f"lumi_nominal[{Lumi}]")
    w.var("lumi_nominal").setConstant(True)
    w.factory("theta_lumi[-5, 5]")           # floating nuisance
    w.factory(f"expr::lumi('@0*(1+{sigma_lumi}*@1)', lumi_nominal, theta_lumi)")
    w.factory("Gaussian::constr_theta_lumi(theta_lumi, lumi_obs[0, -5, 5], 1)")
    w.var("lumi_obs").setConstant(True)      # global observable fixed at nominal 0
    print(f"  Created lumi nuisance: nominal={Lumi} fb^-1, sigma={sigma_lumi*100:.1f}%.")

    # ------------------------------------------------------------------ DSCB shapes
    print("\nCreating DSCB PDFs from fit results...")
    dscb_available: set = set()  # (fs, stxs_bin) pairs

    for fs in final_states:
        for stxs_bin in stxs_bins:
            fr_name = f"fitresult_signal_{safe(fs)}_bin{safe(stxs_bin)}"
            if fr_name not in all_fit_keys:
                continue
            fit_result = fit_file.Get(fr_name)
            if fit_result is None:
                continue

            # Extract fitted parameter values
            float_pars = fit_result.floatParsFinal()
            params = {
                float_pars[i].GetName(): float_pars[i].getVal()
                for i in range(float_pars.getSize())
            }

            prefix = f"{safe(fs)}_{safe(stxs_bin)}"

            mean_val, sigma_val, alphaL_val, nL_val, alphaR_val, nR_val = require_params(
                params,
                ("mean", "sigma", "alphaL", "nL", "alphaR", "nR"),
                fr_name,
            )

            # Create fixed shape parameters via factory (avoids ownership issues)
            w.factory(f"mean_{prefix}[{mean_val}]")
            w.factory(f"sigma_{prefix}[{sigma_val}]")
            w.factory(f"alphaL_{prefix}[{alphaL_val}]")
            w.factory(f"nL_{prefix}[{nL_val}]")
            w.factory(f"alphaR_{prefix}[{alphaR_val}]")
            w.factory(f"nR_{prefix}[{nR_val}]")

            for pname in (
                f"mean_{prefix}", f"sigma_{prefix}",
                f"alphaL_{prefix}", f"nL_{prefix}",
                f"alphaR_{prefix}", f"nR_{prefix}",
            ):
                w.var(pname).setConstant(True)

            # DSCB PDF
            w.factory(
                f"RooCrystalBall::dscb_{prefix}("
                f"{variable}, mean_{prefix}, sigma_{prefix}, "
                f"alphaL_{prefix}, nL_{prefix}, alphaR_{prefix}, nR_{prefix})"
            )

            dscb_available.add((fs, stxs_bin))

    # ------------------------------------------------------------------ background shapes
    # ZZTo4l and ggZZ use Chebychev (params: a0, a1, a2).
    # ZX uses Landau (params: mean, sigma).
    # All fit results are stored as  fitresult_bkg_{fs}_{mode}_{mode}  in fit_results.root.
    print("\nCreating background PDFs (Chebychev for ZZTo4l/ggZZ, Landau for ZX)...")
    bkg_modes = ["ZZTo4l", "ggZZ"]
    bkg_available: set = set()  # (fs, bkg_mode) pairs

    for fs in final_states:
        fs_s = safe(fs)
        for bkg_mode in bkg_modes:
            fr_name = f"fitresult_bkg_{safe(fs)}_{safe(bkg_mode)}_{safe(bkg_mode)}"
            if fr_name not in all_fit_keys:
                print(f"  WARNING: {fr_name} not found, skipping {fs}/{bkg_mode}")
                continue
            bkg_fr = fit_file.Get(fr_name)
            if bkg_fr is None:
                continue

            float_pars = bkg_fr.floatParsFinal()
            params = {
                float_pars[i].GetName(): float_pars[i].getVal()
                for i in range(float_pars.getSize())
            }
            mode_s = safe(bkg_mode)
            prefix_bkg = f"{fs_s}_{mode_s}"

            if bkg_mode == "ZX":
                mean_val, sigma_val = require_params(params, ("mean", "sigma"), fr_name)
                w.factory(f"mean_{prefix_bkg}[{mean_val}]")
                w.factory(f"sigma_{prefix_bkg}[{sigma_val}]")
                w.var(f"mean_{prefix_bkg}").setConstant(True)
                w.var(f"sigma_{prefix_bkg}").setConstant(True)
                w.factory(
                    f"RooLandau::bkg_{prefix_bkg}("
                    f"{variable}, mean_{prefix_bkg}, sigma_{prefix_bkg})"
                )
                print(f"  Created Landau PDF for {fs}/ZX: mean={mean_val:.3f}, sigma={sigma_val:.3f}")
            else:
                a0, a1, a2 = require_params(params, ("a0", "a1", "a2"), fr_name)
                w.factory(f"a0_{prefix_bkg}[{a0}]")
                w.factory(f"a1_{prefix_bkg}[{a1}]")
                w.factory(f"a2_{prefix_bkg}[{a2}]")
                w.var(f"a0_{prefix_bkg}").setConstant(True)
                w.var(f"a1_{prefix_bkg}").setConstant(True)
                w.var(f"a2_{prefix_bkg}").setConstant(True)
                w.factory(
                    f"Chebychev::bkg_{prefix_bkg}("
                    f"{variable}, {{a0_{prefix_bkg}, a1_{prefix_bkg}, a2_{prefix_bkg}}})"
                )

            bkg_available.add((fs, bkg_mode))

    fit_file.Close()
    print(f"  Created {len(bkg_available)} background PDFs.")
    print(f"  Created {len(dscb_available)} DSCB PDFs.")

    # ------------------------------------------------------------------ build data_obs
    print("\nCreating dummy data_obs...")
    data_obs = ROOT.RooDataSet("data_obs", "observed data", ROOT.RooArgSet(mass_var))
    total_added = 0

    for era, data_path in DATA_FILES.items():
        f_data = ROOT.TFile.Open(data_path)
        tree = f_data.Get("Events")

        added_this_file = 0
        for event in tree:
            value = getattr(event, "ZZmass")
            mass = float(value[0]) if hasattr(value, "__len__") else float(value)

            if mass < mass_lo or mass > mass_hi:
                continue

            mass_var.setVal(mass)
            data_obs.add(ROOT.RooArgSet(mass_var))
            added_this_file += 1

        f_data.Close()
        total_added += added_this_file
        print(f"  Added {added_this_file} events from {era}: {data_path}")

    _import(w, data_obs)
    print(f"  Imported data_obs with {total_added} total events.")

    # ------------------------------------------------------------------ per-channel-per-process signal
    print("\nBuilding per-channel-per-process signal models...")
    models_built = []

    for fs in final_states:
        fs_s = safe(fs)
        for stxs_bin in stxs_bins:
            if (fs, stxs_bin) not in dscb_available:
                continue

            bin_s = safe(stxs_bin)
            prefix = f"{fs_s}_{bin_s}"

            xsbr_val = w.var(f"xsbr_{bin_s}").getVal()
            for cat in categories:
                cat_s = safe(cat)
                model_name = f"{fs_s}_{cat_s}_{bin_s}"

                ea_val = epsilonA.get((fs, cat, stxs_bin), 0.0)

                # One expected yield per (final_state, category, stxs_bin)
                yield_val = ea_val * xsbr_val * Lumi
                n_sig_var = ROOT.RooRealVar(
                    f"n_signal_{model_name}",
                    f"Expected signal yield {fs}/{cat}/{stxs_bin}",
                    yield_val,
                )
                n_sig_var.setConstant(True)
                _import(w, n_sig_var)

                # Store one signal shape PDF per (final_state, category, stxs_bin)
                # by cloning the DSCB shape with a channel/process-specific name.
                base_pdf = w.pdf(f"dscb_{prefix}")
                sig_pdf = base_pdf.clone(f"signal_{model_name}")
                _import(w, sig_pdf)

                models_built.append(model_name)

    # ------------------------------------------------------------------ save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_file = ROOT.TFile.Open(output_path, "RECREATE")
    w.Write()
    out_file.Close()

    print(f"\n{'='*60}")
    print(f"Workspace saved : {output_path}")
    print(f"Channel/process models built   : {len(models_built)}")

    # Print yield table for datacard rate lines
    print(f"\n{'='*60}")
    print("Signal yields for datacard rate lines:")
    print(f"  {'channel':<35}  {'yield':>12}")
    print(f"  {'-'*35}  {'-'*12}")
    for model_name in models_built:
        n_var = w.var(f"n_signal_{model_name}")
        if n_var:
            print(f"  {model_name:<35}  {n_var.getVal():>12.4f}")

    print(f"\nTo inspect the workspace:")
    print(f"  root -l '{output_path}'")
    print(f"  w->Print()")
    print(f"\nCombine datacard shapes lines example:")
    print(f"  shapes signal <fs>_<cat>_<bin> {os.path.basename(output_path)} w:signal_<fs>_<cat>_<bin>")
    print(f"  shapes qqZZ   <fs>_<cat>_<bin> {os.path.basename(output_path)} w:bkg_<fs>_ZZTo4l")
    print(f"  shapes ggZZ   <fs>_<cat>_<bin> {os.path.basename(output_path)} w:bkg_<fs>_ggZZ")
    print(f"  shapes ZX     <fs>_<cat>_<bin> {os.path.basename(output_path)} w:bkg_<fs>_ZX")
    print(f"  (use n_signal_<fs>_<cat>_<bin>.getVal() from workspace for the rate line)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
"""
cd /afs/cern.ch/work/z/zhiheng/Thesis && source gato-hep/.venv/bin/activate && python inferenceModel/functions/build_workspace.py \
    --fit_results InferenceModel/Results/GATO/fit_results/fit_results.root \
    --epsilonA    InferenceModel/Results/GATO/epsilonA.pkl \
    --output      InferenceModel/Results/GATO/workspace.root 2>&1
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build RooWorkspace for Combine STXS parametric shape analysis"
    )
    parser.add_argument(
        "--fit_results", required=True,
        help="Path to fit_results.root (containing RooFitResult objects)",
    )
    parser.add_argument(
        "--epsilonA", required=True,
        help="Path to epsilonA.pkl (dict keyed by (final_state, category, stxs_bin))",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output workspace .root file path",
    )
    parser.add_argument(
        "--variable", default="ZZCand_mass",
        help="Observable variable name (default: ZZCand_mass)",
    )
    parser.add_argument(
        "--mass_lo", type=float, default=105.0,
        help="Lower bound of mass range (default: 105)",
    )
    parser.add_argument(
        "--mass_hi", type=float, default=140.0,
        help="Upper bound of mass range (default: 140)",
    )
    args = parser.parse_args()

    build_workspace(
        fit_results_path=args.fit_results,
        epsilonA_path=args.epsilonA,
        output_path=args.output,
        variable=args.variable,
        mass_lo=args.mass_lo,
        mass_hi=args.mass_hi,
    )
