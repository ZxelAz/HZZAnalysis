import os
import numpy as np
import ROOT
import argparse 
import pickle

df_ZX_years = {}
df_ZX_years['2022'] = ROOT.RDataFrame("Events", "/eos/user/z/zhiheng/STXS_samples/2022Data/ZX_2022.root").AsNumpy(["ZZMass", "ZX_Yield"])
df_ZX_years['2022EE'] = ROOT.RDataFrame("Events", "/eos/user/z/zhiheng/STXS_samples/2022Data/ZX_2022EE.root").AsNumpy(["ZZMass", "ZX_Yield"])
df_ZX_years['2023preBPix'] = ROOT.RDataFrame("Events", "/eos/user/z/zhiheng/STXS_samples/2023Data/ZX_2023preBPix.root").AsNumpy(["ZZMass", "ZX_Yield"])
df_ZX_years['2023postBPix'] = ROOT.RDataFrame("Events", "/eos/user/z/zhiheng/STXS_samples/2023Data/ZX_2023postBPix.root").AsNumpy(["ZZMass", "ZX_Yield"])

# merge into one df
df_ZX = {}
for year, df in df_ZX_years.items():
    for key in df.keys():
        if key not in df_ZX:
            df_ZX[key] = df[key]
        else:
            df_ZX[key] = np.concatenate((df_ZX[key], df[key]))

