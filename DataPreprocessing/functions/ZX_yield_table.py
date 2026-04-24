import ROOT
import argparse
import os
import sys
import numpy as np

def main():
    print("="*60)
    for year in ["2022", "2022EE"]:
        input_file = f"/eos/user/z/zhiheng/STXS_samples/2022Data/ZX_{year}.root"
        df = ROOT.RDataFrame("Events", input_file)
        df = df.Filter("ZZMass > 70")
        df_4e = df.Filter("FinState == 0").AsNumpy()
        df_4mu = df.Filter("FinState == 1").AsNumpy()
        df_2e2mu = df.Filter("FinState == 2").AsNumpy()
        df_2mu2e = df.Filter("FinState == 3").AsNumpy()
        yield_4e = np.sum(df_4e['ZX_Yield']) 
        yield_4mu = np.sum(df_4mu['ZX_Yield'])
        yield_2e2mu = np.sum(df_2e2mu['ZX_Yield'])
        yield_2mu2e = np.sum(df_2mu2e['ZX_Yield'])
        print(f"Year: {year}")
        print(f"4e Yield: {yield_4e}")
        print(f"4mu Yield: {yield_4mu}")
        print(f"2e2mu Yield: {yield_2e2mu}")
        print(f"2mu2e Yield: {yield_2mu2e}")
        print(f"number of events: {len(df_4e['ZX_Yield']) + len(df_4mu['ZX_Yield']) + len(df_2e2mu['ZX_Yield']) + len(df_2mu2e['ZX_Yield'])}")
        print("="*60)

    for year in ["2023preBPix", "2023postBPix"]:
        input_file = f"/eos/user/z/zhiheng/STXS_samples/2023Data/ZX_{year}.root"
        df = ROOT.RDataFrame("Events", input_file)
        df = df.Filter("ZZMass > 70")
        df_4e = df.Filter("FinState == 0").AsNumpy()
        df_4mu = df.Filter("FinState == 1").AsNumpy()
        df_2e2mu = df.Filter("FinState == 2").AsNumpy()
        df_2mu2e = df.Filter("FinState == 3").AsNumpy()
        yield_4e = np.sum(df_4e['ZX_Yield'])
        yield_4mu = np.sum(df_4mu['ZX_Yield'])
        yield_2e2mu = np.sum(df_2e2mu['ZX_Yield'])
        yield_2mu2e = np.sum(df_2mu2e['ZX_Yield'])
        print(f"Year: {year}")
        print(f"4e Yield: {yield_4e}")
        print(f"4mu Yield: {yield_4mu}")
        print(f"2e2mu Yield: {yield_2e2mu}")
        print(f"2mu2e Yield: {yield_2mu2e}")
        print(f"number of events: {len(df_4e['ZX_Yield']) + len(df_4mu['ZX_Yield']) + len(df_2e2mu['ZX_Yield']) + len(df_2mu2e['ZX_Yield'])}")
        print("="*60)

if __name__ == "__main__":
    main()
     