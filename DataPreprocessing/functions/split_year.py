#!/usr/bin/env python3
"""
Script to split ROOT files by run number for different data-taking periods.
- 2022: Split into pre-EE (run <= 359021) and EE (run > 359021)
- 2023: Split into preBPix (run <= 369802) and postBPix (run > 369802)
"""

import ROOT
import argparse
import os


def split_year(year, input_file, output_path):
    """
    Split a ROOT file into two periods based on run number.
    
    Args:
        year: Data-taking year ("2022" or "2023")
        input_file: Path to input ROOT file
        output_path: Directory path for output files
    """
    if year == "2022":
        run = 359021
        prefix = ""
        suffix = "EE"
    elif year == "2023":
        run = 369802
        prefix = "preBPix"
        suffix = "postBPix"
    else:
        raise ValueError(f"Unsupported year: {year}. Use '2022' or '2023'")
    
    # Create output directory if it doesn't exist
    year_data_dir = f"{output_path}/{year}Data"
    if not os.path.exists(year_data_dir):
        os.makedirs(year_data_dir)
        print(f"Created output directory: {year_data_dir}")
    
    print(f"Loading input file: {input_file}")
    df = ROOT.RDataFrame("Events", input_file)
    
    # Count total events
    total_events = df.Count().GetValue()
    print(f"Total events in input: {total_events}")
    
    # Split by run number
    output_pre = f"{year_data_dir}/{year}{prefix}.root" 
    output_post = f"{year_data_dir}/{year}{suffix}.root"
    
    print(f"Splitting at run number: {run}")
    print(f"  Creating: {output_pre}")
    df_pre = df.Filter(f"run <= {run}")
    n_pre = df_pre.Count().GetValue()
    df_pre.Snapshot("Events", output_pre)
    
    print(f"  Creating: {output_post}")
    df_post = df.Filter(f"run > {run}")
    n_post = df_post.Count().GetValue()
    df_post.Snapshot("Events", output_post)
    
    print(f"\n✓ Split completed:")
    print(f"  {output_pre}: {n_pre} events")
    print(f"  {output_post}: {n_post} events")
    print(f"  Total: {n_pre + n_post} events (original: {total_events})")


def main():
    parser = argparse.ArgumentParser(
        description='Split ROOT files by run number for different data-taking periods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split 2022 data at run 359021 (pre-EE/EE boundary)
  python ${HZZ_ROOT}/DataPreprocessing/functions/split_year.py --year 2022 --input /eos/user/a/atarabin/STXS_samples/Data2022/ZZ4lAnalysis.root --output /eos/user/z/zhiheng/STXS_samples
  
  # Split 2023 data at run 369802 (preBPix/postBPix boundary)
  python ${HZZ_ROOT}/DataPreprocessing/functions/split_year.py --year 2023 --input /eos/user/a/atarabin/STXS_samples/Data2023/ZZ4lAnalysis.root --output /eos/user/z/zhiheng/STXS_samples
        """
    )
    
    parser.add_argument('--year', required=True, choices=['2022', '2023'],
                        help='Data-taking year')
    parser.add_argument('--input', required=True,
                        help='Path to input ROOT file')
    parser.add_argument('--output', required=True,
                        help='Output directory path')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    try:
        split_year(args.year, args.input, args.output)
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
