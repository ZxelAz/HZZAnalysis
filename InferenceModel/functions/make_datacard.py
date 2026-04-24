#!/usr/bin/env python3
"""
make_datacard.py

Generate a CMS Combine datacard for the H→ZZ→4ℓ STXS parametric shape analysis.

The workspace must be built first with build_workspace.py.  The script reads
epsilonA.pkl and N_bkg_category.pkl to reconstruct the same channel list and to compute
Asimov yields (signal at μ=1, fixed backgrounds) used as placeholder observations.

Workspace naming conventions expected:
  signal_{fs}_{bin}      -- extended signal PDF  (RooExtendPdf)
    bkg_{fs}_{mode}        -- background shape PDF  (Chebychev, non-extended)
  theta_lumi             -- floating lumi nuisance (already in workspace)
  constr_theta_lumi      -- Gaussian constraint PDF  (already in workspace)

Nuisance handling
-----------------
theta_lumi is declared with `param 0 1` in the datacard, which tells Combine to
multiply the likelihood by a unit Gaussian.  The workspace also contains
constr_theta_lumi as a safety, but Combine's param directive takes precedence
when the parameter is declared that way.  Remove one if you want to avoid double-
counting once you move beyond the toy stage.

Observations
------------
Real data should be provided as RooDataSets named  data_obs_{fs}_{bin}  added to
the workspace, with a matching `shapes data_obs` line.  For now the observation is
set to the Asimov S+B expectation at μ=1 (integer), which is sufficient to run
toys (-t -1) and expected limits/pulls.

Usage
-----
  python make_datacard.py \\
      --epsilonA  results/trial5/epsilonA.pkl \\
      --N_bkg_category results/trial5/N_bkg_category.pkl \\
      --workspace results/trial5/workspace.root \\
      --output    results/trial5/datacard.txt
  # Run Combine on the result (Asimov expected):
"""

import os
import pickle
import argparse
import textwrap

import numpy as np


# ---------------------------------------------------------------------------
# helpers (mirrors build_workspace.py)
# ---------------------------------------------------------------------------

def safe(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "p")
    )


def stxs_bin_to_xs_key(stxs_bin: str) -> str:
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
    raise ValueError(f"Cannot map STXS bin '{stxs_bin}' to XS key.")


def parse_csv_list(raw):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    items = [item.strip() for item in str(raw).split(",")]
    items = [item for item in items if item]
    return items if items else None


XS   = {"ggH": 51.96, "VBF": 4.067, "WH": 1.443, "ZH": 0.9361, "ttH": 0.5634}
BR   = 1.251e-4
LUMI = 62.34 * 1000  # pb^-1


# ---------------------------------------------------------------------------
# datacard writer
# ---------------------------------------------------------------------------

def make_datacard(
    epsilonA_path: str,
    N_bkg_category_path: str,
    workspace_path: str,
    output_path: str,
    wsname: str = "w",
    processes_list=None,
    categories_list=None,
) -> None:

    processes_list = parse_csv_list(processes_list)
    categories_list = parse_csv_list(categories_list)

    # ------------------------------------------------------------------ load
    with open(epsilonA_path, "rb") as fh:
        epsilonA: dict = pickle.load(fh)
    with open(N_bkg_category_path, "rb") as fh:
        N_bkg_category_raw: dict = pickle.load(fh)
    # Ensure simple float values: (final_state, category) -> {mode: yield}
    N_bkg_category = {
        k: {m: float(np.sum(v)) for m, v in mode_dict.items()}
        for k, mode_dict in N_bkg_category_raw.items()
    }

    final_states = sorted({k[0] for k in epsilonA})
    categories   = categories_list if categories_list else sorted({k[1] for k in epsilonA})
    stxs_bins    = processes_list if processes_list else sorted({k[2] for k in epsilonA})
    bkg_modes    = sorted({m for mode_dict in N_bkg_category.values() for m in mode_dict.keys()})
    
    ws_basename = os.path.basename(workspace_path)

    # ------------------------------------------------------------------ keys
    # A channel exists for each (fs, stxs_bin) that has at least one non-zero
    # epsilonA across all categories — mirrors the build_workspace.py skip logic.
    signal_keys = []
    for fs in final_states:
        for stxs_bin in stxs_bins:
            for cat in categories:
                signal_keys.append(f"ch_{safe(fs)}_{safe(cat)}_{safe(stxs_bin)}")
    n_sig_keys  = len(final_states)*len(categories)*len(stxs_bins)
    n_bkg_procs = len(bkg_modes) * len(categories)   # per-category-final-state background processes
    all_pois = sorted({f"r_{fs}_{stxs_bin}" for (fs, _, stxs_bin), ea in epsilonA.items() if ea > 0})

    print(f"signal fs*proc*cats : {n_sig_keys}")
    print(f"BKG procs: {n_bkg_procs}  {bkg_modes}")
    print(f"POIs     : {len(all_pois)}")

    # ------------------------------------------------------------------ Asimov yields
    # Signal at mu=1:  S = xsbr * Lumi * epsilonA  (per fs, cat, stxs_bin)
    sig_yield = {}
    for fs in final_states:
        for stxs_bin in stxs_bins:
            xs_key   = stxs_bin_to_xs_key(stxs_bin)
            xsbr     = XS[xs_key] * BR
            for cat in categories:  
                ea_ij = epsilonA.get((fs, cat, stxs_bin))
                # if ea_ij <= 0:
                #     ea_ij = 1e-9
                sig_yield[(fs, cat, stxs_bin)] = xsbr * LUMI * ea_ij
    nonzero_sig = [v for v in sig_yield.values() if v is not None and v > 0]
    print("Min/max signal yield per category: ", min(nonzero_sig), max(nonzero_sig))
    # Background per category
    bkg_yield = {}  # (channel_name, mode, category) -> yield
    for fs in final_states:
        for bkg_mode in bkg_modes:
            for cat in categories:
                bkg_yield[fs, cat, bkg_mode] = N_bkg_category.get((fs, cat), {}).get(bkg_mode, 0.0)
                # if bkg_yield[fs, cat, bkg_mode] <= 0:
                #     bkg_yield[fs, cat, bkg_mode] = 1e-9
    nonzero_bkg = [v for v in bkg_yield.values() if v is not None and v > 0]
    print("Min/max background yield per category: ", min(nonzero_bkg), max(nonzero_bkg))
    # ------------------------------------------------------------------ column widths
    max_ch = max(len(key) for key in signal_keys)
    col = max(max_ch + 2, 20)

    key_list = [key for key in signal_keys]

    # ------------------------------------------------------------------ write
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:

        # --- header
        f.write("# CMS Combine datacard -- H->ZZ->4l STXS parametric shape analysis\n")
        f.write("# Generated by make_datacard.py\n")
        f.write("#\n")
        f.write(f"# Workspace : {ws_basename}  (object: {wsname})\n")
        f.write(f"# signal keys  : {len(signal_keys)}  (one per final_state x category x STXS bin)\n")
        f.write("# Signal processes have mu-dependent yields baked into the workspace.\n")
        f.write("# Backgrounds (qqZZ, ggZZ) use non-extended shapes; rates come from datacard.\n")
        f.write("#\n")
        f.write(f"# List of all POIs ({len(all_pois)} total):\n")
        f.write("#\n")
        f.write("imax *\n")
        f.write("jmax *\n")
        f.write("kmax *\n")

        # --- shapes
        f.write("------------\n")
        f.write("# shapes: process  channel  workspace  workspace_object\n")
        f.write(f"shapes  data_obs  *  {ws_basename}  {wsname}:data_obs\n")
        # Mapping for remapped bins
        stxs_bin_remap = {
            "WminushadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25": "WminushadH_GE2J_MJJ_GT350_PTH_GT200",
            "WplushadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25": "WplushadH_GE2J_MJJ_GT350_PTH_GT200",
            "ZhadH_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25": "ZhadH_GE2J_MJJ_60_120",
            "ZhadH_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25": "ZhadH_GE2J_MJJ_GT350_PTH_GT200",
        }
        for fs in final_states:
            for cat in categories:
                for stxs_bin in stxs_bins:
                    remapped_bin = stxs_bin_remap.get(stxs_bin, stxs_bin)
                    f.write(f"shapes  proc_{stxs_bin:<20} cat_{cat}_{fs:<{col}}  {ws_basename}  {wsname}:signal_{fs}_{cat}_{remapped_bin}\n")
                for bkg_mode in bkg_modes:
                    f.write(f"shapes  bkg_{bkg_mode:<20} cat_{cat}_{fs:<{col}}  {ws_basename}  {wsname}:bkg_{fs}_{bkg_mode}\n")
                
       

        # --- observation
        f.write("------------\n")
        f.write("bin         " + "  ".join(f"cat_{cat}_{fs:<{col}}" for fs in final_states for cat in categories) + "\n")
        f.write("observation " + "  ".join(f"{-1:<{col}}" for fs in final_states for cat in categories) + "\n")

        # --- rates block  (columns: one per final_state,category,process)
        f.write("------------\n")
        bin_row   = []
        proc_name = []
        proc_idx  = []
        rate_row  = []
        lumi_row  = []
        sig_idx = 0
        for fs in final_states:
            for cat in categories:
                cat_yield = sum(sig_yield.get((fs, cat, stxs_bin), 0) for stxs_bin in stxs_bins) + sum(bkg_yield.get((fs, cat, bkg_mode), 0) for bkg_mode in bkg_modes)
                sig_idx = 0
                bkg_idx = 1
                for stxs_bin in stxs_bins:
                    if sig_yield[fs, cat, stxs_bin]/cat_yield <= 0.001:
                        print(f"Skipping signal process with negligible yield: {fs} {cat} {stxs_bin}  yield={sig_yield[fs, cat, stxs_bin]:.3f}  cat_yield={cat_yield:.3f}")
                        continue
                    bin_row.append(f"cat_{cat}_{fs}")
                    proc_name.append(f"proc_{stxs_bin}")
                    proc_idx.append(sig_idx)
                    sig_idx -= 1
                    rate_row.append(f"{sig_yield[fs, cat, stxs_bin]*138/62.24}") # scale from 62.24 to 138 fb^-1 temporarily
                    lumi_row.append("1.001")
                for bkg_mode in bkg_modes:
                    if bkg_yield[fs, cat, bkg_mode]/cat_yield <= 0.001:
                        print(f"Skipping background process with negligible yield: {fs} {cat} {bkg_mode}  yield={bkg_yield[fs, cat, bkg_mode]:.3f}  cat_yield={cat_yield:.3f}")
                        continue
                    bin_row.append(f"cat_{cat}_{fs}")
                    proc_name.append(f"bkg_{bkg_mode}")
                    proc_idx.append(bkg_idx)
                    bkg_idx += 1
                    rate_row.append(f"{bkg_yield[fs, cat, bkg_mode]*138/62.24}") # scale from 62.24 to 138 fb^-1 temporarily
                    lumi_row.append("-")

                # Combine requires at least one background per bin.
                # If all were skipped, force-add the dominant mode with a 1e-9 floor.
                if bkg_idx == 1:
                    dominant_mode = max(bkg_modes, key=lambda m: bkg_yield.get((fs, cat, m), 0.0))
                    floor_yield = max(bkg_yield.get((fs, cat, dominant_mode), 0.0), 1e-9)
                    bin_row.append(f"cat_{cat}_{fs}")
                    proc_name.append(f"bkg_{dominant_mode}")
                    proc_idx.append(bkg_idx)
                    bkg_idx += 1
                    rate_row.append(f"{floor_yield * 138/62.24}")
                    lumi_row.append("-")
                    print(f"  Forced floor background: {fs} {cat} {dominant_mode}  yield={floor_yield:.3e}")
                

        f.write("bin      " + "  ".join(f"{b:<{col}}" for b in bin_row) + "\n")
        f.write("process  " + "  ".join(f"{p:<{col}}" for p in proc_name) + "\n")
        f.write("process  " + "  ".join(f"{p:<{col}}" for p in proc_idx) + "\n")
        f.write("rate     " + "  ".join(f"{r:<{col}}" for r in rate_row) + "\n")

        # --- nuisances
        f.write("------------\n")
        f.write("# Gaussian-constrained lumi nuisance (2.5 % dummy uncertainty).\n")
        f.write("# The workspace also contains constr_theta_lumi -- remove one to avoid\n")
        f.write("# double-counting once you move to a real analysis.\n")
        f.write("lumi  lnN  " + "  ".join(f"{x:<{col}}" for x in lumi_row) + "\n")

    n_channels = len(final_states) * len(categories)
    print(f"\nDatacard written: {output_path}")
    print(f"  {n_channels} channels  x  {len(stxs_bins)+len(bkg_modes)} processes  =  {n_channels * (len(stxs_bins)+len(bkg_modes))} columns")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write a CMS Combine datacard for the HZZ STXS parametric analysis"
    )
    parser.add_argument("--epsilonA",  required=True, help="Path to epsilonA.pkl")
    parser.add_argument("--N_bkg_category", required=True, help="Path to N_bkg_category.pkl (ignored with --list-pois)")
    parser.add_argument("--workspace", required=True, help="Path to workspace.root (ignored with --list-pois)")
    parser.add_argument("--output",    default=None,  help="Output datacard .txt path")
    parser.add_argument("--wsname",    default="w",   help="Name of the RooWorkspace object (default: w)")
    parser.add_argument("--processes-list", default=None, help="Optional list of STXS bins to include (default: all in epsilonA.pkl)")
    parser.add_argument("--categories-list", default=None, help="Optional list of categories to include (default: all in epsilonA.pkl)")
    args = parser.parse_args()
    
    make_datacard(
        epsilonA_path=args.epsilonA,
        N_bkg_category_path=args.N_bkg_category,
        workspace_path=args.workspace,
        output_path=args.output,
        wsname=args.wsname,
        processes_list=parse_csv_list(args.processes_list),
        categories_list=parse_csv_list(args.categories_list),
    )
