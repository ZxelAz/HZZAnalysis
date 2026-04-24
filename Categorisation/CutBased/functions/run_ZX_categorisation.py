import ROOT
import numpy as np
import argparse
import os
import sys
import pandas as pd

# Import categorisation functions from run2_categorisation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run2_categorisation import first_step_categorisation, second_step_categorisation

# Columns required by the two-step categorisation + yield weight
ZX_FEATURES = [
    "ZZCand_pt", "ZZCand_nExtraLep", "ZZjj_pt",
    "nCleanedJetsPt30", "nBtagged_filtered",
    "m_jj", "DVBF2j_ME", "DVBF1j_ME", "DWHh_ME", "DZHh_ME",
    "LepPdgId_4", "LepPdgId_5",
    "ZX_Yield",
]

# Integer branches that ROOT's AsNumpy returns as raw bytes on some versions
INT_COLS = ["nCleanedJetsPt30", "nBtagged_filtered", "ZZCand_nExtraLep",
            "LepPdgId_4", "LepPdgId_5"]


def load_zx_file(file_path, year_label):
    rdf = ROOT.RDataFrame("Events", file_path)
    arr = rdf.AsNumpy(ZX_FEATURES)
    # Cast integer branches — ROOT may return them as raw single-byte objects
    for col in INT_COLS:
        if col in arr:
            a = arr[col]
            if a.dtype.kind in ('O', 'S', 'U'):
                arr[col] = np.array([ord(v) if isinstance(v, (bytes, str)) and len(v) == 1
                                     else int(v) for v in a], dtype=np.int32)
            else:
                arr[col] = a.astype(np.int32)
    n = len(arr["ZX_Yield"])
    arr["mode_label"] = np.array(["ZX"] * n, dtype=object)
    arr["year"] = np.array([year_label] * n, dtype=object)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Run STXS categorisation on filtered ZX CR files.")
    parser.add_argument("--input_files", type=str, required=True,
                        help="Comma-separated paths to filtered ZX ROOT files")
    parser.add_argument("--years", type=str, required=True,
                        help="Comma-separated year labels matching --input_files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save the output yield table (CSV)")
    args = parser.parse_args()

    input_files = [f.strip() for f in args.input_files.split(",")]
    years       = [y.strip() for y in args.years.split(",")]

    if len(input_files) != len(years):
        raise ValueError("--input_files and --years must have the same number of entries")

    os.makedirs(args.output_path, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load and concatenate all campaigns
    # ------------------------------------------------------------------
    print(f"Loading {len(input_files)} filtered ZX file(s)...")
    all_arrs = []
    for f, y in zip(input_files, years):
        print(f"  {y}: {f}")
        arr = load_zx_file(f, y)
        n = len(arr["ZX_Yield"])
        print(f"    → {n} events, ZX_Yield sum = {np.sum(arr['ZX_Yield']):.4f}")
        all_arrs.append(arr)

    combined = {key: np.concatenate([a[key] for a in all_arrs]) for key in all_arrs[0]}
    print(f"\nTotal: {len(combined['ZX_Yield'])} events, "
          f"ZX_Yield sum = {np.sum(combined['ZX_Yield']):.4f}")

    # ------------------------------------------------------------------
    # 2) First-step (stage 0) categorisation
    # ------------------------------------------------------------------
    print("\nRunning first-step (stage-0) categorisation...")
    first_step = first_step_categorisation(combined)
    for cat, arr in first_step.items():
        print(f"  {cat}: {len(arr['ZX_Yield'])} events, yield = {np.sum(arr['ZX_Yield']):.4f}")

    # ------------------------------------------------------------------
    # 3) Second-step (stage 1.2) categorisation
    # ------------------------------------------------------------------
    print("\nRunning second-step (stage-1.2) categorisation...")
    second_step = second_step_categorisation(first_step)

    # ------------------------------------------------------------------
    # 4) Compute per-category ZX yields and save
    # ------------------------------------------------------------------
    rows = []
    for cat, arr in second_step.items():
        rows.append({
            "category": cat,
            "n_events": len(arr["ZX_Yield"]),
            "ZX_yield": float(np.sum(arr["ZX_Yield"])),
        })

    df = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)

    print("\n" + "=" * 65)
    print("ZX yield per STXS category (all campaigns combined):")
    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\nTotal ZX yield: {df['ZX_yield'].sum():.4f}")

    out_csv = os.path.join(args.output_path, "ZX_category_yields.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
