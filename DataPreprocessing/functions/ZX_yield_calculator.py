import ROOT
import argparse
import os
import sys
from get_genEventSumw import get_genEventSumw
import numpy as np


# Keep C++-side caches for mapping python-computed values back to RDataFrame rows.
# The lookup is keyed by run/lumi/event because rdfentry_ refers to the original tree entry,
# not the compact row index of the filtered control-region dataframe.
ROOT.gInterpreter.Declare(
    """
    #include <string>
    #include <unordered_map>

    static std::unordered_map<std::string, double> gZXYieldValues;
    static std::unordered_map<std::string, int> gFinStateValues;

    std::string makeZXYieldKey(ULong64_t run, ULong64_t lumi, ULong64_t event) {
        return std::to_string(run) + ":" + std::to_string(lumi) + ":" + std::to_string(event);
    }

    void clearZXYieldValues() {
        gZXYieldValues.clear();
    }

    void clearFinStateValues() {
        gFinStateValues.clear();
    }

    void setZXYieldValue(ULong64_t run, ULong64_t lumi, ULong64_t event, double value) {
        gZXYieldValues[makeZXYieldKey(run, lumi, event)] = value;
    }

    void setFinStateValue(ULong64_t run, ULong64_t lumi, ULong64_t event, int value) {
        gFinStateValues[makeZXYieldKey(run, lumi, event)] = value;
    }

    double getZXYieldFromEvent(ULong64_t run, ULong64_t lumi, ULong64_t event) {
        const auto it = gZXYieldValues.find(makeZXYieldKey(run, lumi, event));
        if (it != gZXYieldValues.end()) return it->second;
        return 0.0;
    }

    int getFinStateFromEvent(ULong64_t run, ULong64_t lumi, ULong64_t event) {
        const auto it = gFinStateValues.find(makeZXYieldKey(run, lumi, event));
        if (it != gFinStateValues.end()) return it->second;
        return -1;
    }
    """
)


def write_root_tree_from_dict(data_dict, output_file, tree_name="Events"):
    """Write a ROOT TTree from a dict of columns without dropping any supported data.

    Supports scalar numeric/bool columns and jagged/vector-like numeric/bool columns.
    Raises an exception if a column type cannot be represented.
    """

    def _first_non_null(values):
        for item in values:
            if item is not None:
                return item
        return None

    def _is_sequence_like(item):
        if item is None or isinstance(item, (str, bytes)):
            return False
        # cppyy ROOT::VecOps::RVec can be reported as scalar by numpy helpers.
        if "ROOT.VecOps.RVec" in str(type(item)):
            return True
        if np.isscalar(item):
            return False
        try:
            iter(item)
            return True
        except TypeError:
            return False

    def _infer_numeric_kind(sample):
        if isinstance(sample, (bool, np.bool_)):
            return "bool"
        if isinstance(sample, (int, np.integer)):
            return "int"
        if isinstance(sample, (float, np.floating)):
            return "float"
        return None

    def _infer_sequence_elem_kind(elem):
        kind = _infer_numeric_kind(elem)
        if kind is not None:
            return kind
        # ROOT char vectors often surface as 1-byte str/bytes in Python.
        if isinstance(elem, (bytes, np.bytes_)):
            return "uchar"
        if isinstance(elem, str) and len(elem) == 1:
            return "uchar"
        return None

    def _sequence_kind(seq):
        for elem in seq:
            kind = _infer_sequence_elem_kind(elem)
            if kind is not None:
                return kind
        return "float"

    def _kind_to_leaf(kind):
        if kind == "bool":
            return "O", np.bool_
        if kind == "int":
            return "L", np.int64
        if kind == "float":
            return "D", np.float64
        raise ValueError(f"Unknown scalar kind: {kind}")

    def _kind_to_vector_type(kind):
        if kind == "bool":
            return "unsigned char"
        if kind == "uchar":
            return "unsigned char"
        if kind == "int":
            return "long long"
        if kind == "float":
            return "double"
        raise ValueError(f"Unknown vector kind: {kind}")

    prepared = {}
    n_events = None

    for key, val in data_dict.items():
        values = np.asarray(val, dtype=object) if np.asarray(val).dtype == object else np.asarray(val)

        if values.ndim != 1:
            raise TypeError(f"Column '{key}' has ndim={values.ndim}; only 1D event columns are supported")

        if n_events is None:
            n_events = len(values)
        elif len(values) != n_events:
            raise ValueError(f"Column '{key}' length {len(values)} differs from expected {n_events}")

        sample = _first_non_null(values)
        if sample is None:
            # All-null scalar placeholder; write as float scalar zeros.
            leaf_code, dtype = _kind_to_leaf("float")
            prepared[key] = {
                "mode": "scalar",
                "values": np.zeros(n_events, dtype=dtype),
                "buffer": np.zeros(1, dtype=dtype),
                "leaf": leaf_code,
            }
            continue

        if _is_sequence_like(sample):
            kind = _sequence_kind(sample)
            vec_type = _kind_to_vector_type(kind)
            prepared[key] = {
                "mode": "vector",
                "values": values,
                "buffer": ROOT.std.vector(vec_type)(),
                "kind": kind,
            }
            continue

        scalar_kind = _infer_numeric_kind(sample)
        if scalar_kind is None:
            raise TypeError(f"Column '{key}' has unsupported scalar type: {type(sample)}")
        leaf_code, dtype = _kind_to_leaf(scalar_kind)
        coerced = np.asarray(values, dtype=dtype)
        prepared[key] = {
            "mode": "scalar",
            "values": coerced,
            "buffer": np.zeros(1, dtype=dtype),
            "leaf": leaf_code,
        }

    out_file = ROOT.TFile(output_file, "RECREATE")
    out_tree = ROOT.TTree(tree_name, tree_name)

    for key, info in prepared.items():
        if info["mode"] == "scalar":
            out_tree.Branch(key, info["buffer"], f"{key}/{info['leaf']}")
        else:
            out_tree.Branch(key, info["buffer"])

    for idx in range(n_events):
        for info in prepared.values():
            if info["mode"] == "scalar":
                info["buffer"][0] = info["values"][idx]
            else:
                vec = info["buffer"]
                vec.clear()
                entry = info["values"][idx]
                if entry is None:
                    continue
                for elem in entry:
                    if info["kind"] == "bool":
                        vec.push_back(1 if bool(elem) else 0)
                    elif info["kind"] == "uchar":
                        if isinstance(elem, (bytes, np.bytes_)):
                            vec.push_back(elem[0] if len(elem) > 0 else 0)
                        elif isinstance(elem, str):
                            vec.push_back(ord(elem[0]) if len(elem) > 0 else 0)
                        else:
                            vec.push_back(int(elem) & 0xFF)
                    elif info["kind"] == "int":
                        vec.push_back(int(elem))
                    else:
                        vec.push_back(float(elem))
        out_tree.Fill()

    out_file.Write()
    out_file.Close()

# Helper for RDataFrame: create a dummy RVec with zeros matching input size.
ROOT.gInterpreter.Declare(
    """
    ROOT::VecOps::RVec<unsigned char> addDummyBranch(const ROOT::VecOps::RVec<float>& vec) {
        return ROOT::VecOps::RVec<unsigned char>(vec.size(), 0);
    }
    """
)

# function to split 2023 into pre/post BPix and 2022 into EE/non-EE, as the coefficients are different for these cases
def split_year(year, input_file, output_file):
    if year == "2022":
        run = 359021
        prefix = ""
        suffix = "EE"
    elif year == "2023":
        run = 369802
        prefix = "preBPix"
        suffix = "postBPix"
    df = ROOT.RDataFrame("Events", input_file)
    df_pre = df.Filter(f"run <= {run}").Snapshot("Events", f"{output_file}/{year}_{prefix}.root")
    df_post = df.Filter(f"run > {run}").Snapshot("Events", f"{output_file}/{year}_{suffix}.root")
    print(f"✓ Split {input_file} into {output_file}/{year}_{prefix}.root and {output_file}/{year}_{suffix}.root")

# coefficients
def comb(year):
    if year == "2022": # 2022 from HIG 24 13, 2023 from SPENCER
        cb_SS = np.array([
            1.239, # 4e
            1.093, # 4mu
            1.057, # 2e2mu
            1.254, # 2mu2e
        ])
    elif year == "2022EE":
        cb_SS = np.array([
            1.067, # 4e
            1.015, # 4mu
            1.049, # 2e2mu
            0.905, # 2mu2e
        ])
    elif year == "2023preBPix":
        cb_SS = np.array([
            1.116, # 4e
            1.036, # 4mu
            0.989, # 2e2mu
            1.141, # 2mu2e
        ])
    elif year == "2023postBPix":
        cb_SS = np.array([
            0.795, # 4e
            1.025, # 4mu
            1.074, # 2e2mu
            1.078, # 2mu2e
        ])
    elif year == "2024": 
        cb_SS = np.array([
            0.787, # 4e
            0.960, # 4mu
            0.958, # 2e2mu
            0.749, # 2mu2e
        ])
    return cb_SS

def makeCR(_df, _flag):

    # CRZLLss 21 --> 2097152
    # CRZLLos_2P2F 22 --> 4194304
    # CRZLLos_3P1F 23 --> 8388608
    if _flag == '3P1F':
        bit = '8388608'
    elif _flag == '2P2F':
        bit = '4194304'
    elif _flag == 'SS':
        bit = '2097152'
    elif _flag == 'SIPCR':
        bit = '21'
    else:
        raise Exception("The CR "+_flag+" is not known")

    df_out = ( _df.Filter('ZLLbest'+_flag+'Idx>-1').Define("ZZMass", "ZLLCand_mass[ZLLbest"+_flag+"Idx]")
                                                   .Define("RunNumber", "run")
                                                   .Define("EventNumber", "event") # for uniqueness
                                                   .Define("LumiNumber", "luminosityBlock")
                                                   .Define("CRflag", bit)
                                                   .Define("Z1Mass", "ZLLCand_Z1mass[ZLLbest"+_flag+"Idx]")
                                                   .Define("Z2Mass", "ZLLCand_Z2mass[ZLLbest"+_flag+"Idx]")
                                                   .Define("Z1Flav", "ZLLCand_Z1flav[ZLLbest"+_flag+"Idx]")
                                                   .Define("Z2Flav", "ZLLCand_Z2flav[ZLLbest"+_flag+"Idx]")
                                                   .Define('Leptons_pt', "Concatenate(Electron_pt,Muon_pt)")
                                                   .Define('Leptons_eta', "Concatenate(Electron_eta,Muon_eta)")
                                                   .Define('Leptons_phi', "Concatenate(Electron_phi,Muon_phi)")
                                                   .Define('Leptons_dxy', "Concatenate(Electron_dxy,Muon_dxy)")
                                                   .Define('Leptons_dz', "Concatenate(Electron_dz,Muon_dz)")
                                                   .Define('Leptons_id', "Concatenate(Electron_pdgId,Muon_pdgId)")
                                                   .Define('Leptons_sip', "Concatenate(Electron_sip3d,Muon_sip3d)")
                                                   .Define('Leptons_iso', "Concatenate(Electron_pfRelIso03FsrCorr,Muon_pfRelIso03FsrCorr)")
                                                   .Define('Leptons_isid', "Concatenate(Electron_passBDT,Muon_passID)")
                                                   ## Need to add the LepMissingHit branch for SS FR method
                                                   ## First create a dummy branch for muons filled with zeroes
                                                   .Define('Muon_lostHits', "addDummyBranch(Muon_pt)")
                                                   .Define('Leptons_missinghit', "Concatenate(Electron_lostHits, Muon_lostHits)")
                                                   ## Variable miniAOD-style
                                                   .Define('LeptonPt', "std::vector<float> LepPt{Leptons_pt[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_pt[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_pt[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_pt[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return LepPt")
                                                   .Define('LeptonEta', "std::vector<float> LepEta{Leptons_eta[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_eta[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_eta[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_eta[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return LepEta")
                                                   .Define('LeptonPhi', "std::vector<float> LepPhi{Leptons_phi[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_phi[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_phi[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_phi[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return LepPhi")
                                                   .Define('Leptondxy', "std::vector<float> Lepdxy{Leptons_dxy[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_dxy[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_dxy[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_dxy[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return Lepdxy")
                                                   .Define('Leptondz', "std::vector<float> Lepdz{Leptons_dz[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_dz[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_dz[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_dz[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return Lepdz")
                                                   .Define('LeptonLepId', "std::vector<short> LepLepId{static_cast<short>(Leptons_id[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]]), static_cast<short>(Leptons_id[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]]), static_cast<short>(Leptons_id[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]]), static_cast<short>(Leptons_id[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]])}; return LepLepId")
                                                   .Define('LepSIP', "std::vector<float> LepSIP{Leptons_sip[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_sip[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_sip[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_sip[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return LepSIP")
                                                   .Define('LepCombRelIsoPF', "std::vector<float> LepCombRelIsoPF{Leptons_iso[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_iso[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_iso[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_iso[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return LepCombRelIsoPF")
                                                   .Define('LepisID', "std::vector<bool> LepisID{Leptons_isid[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_isid[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_isid[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_isid[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return Leptons_isid")
                                                   .Define('LepMissingHit', "std::vector<unsigned char> LepMissingHit{Leptons_missinghit[ZLLCand_Z1l1Idx[ZLLbest"+_flag+"Idx]], Leptons_missinghit[ZLLCand_Z1l2Idx[ZLLbest"+_flag+"Idx]], Leptons_missinghit[ZLLCand_Z2l1Idx[ZLLbest"+_flag+"Idx]], Leptons_missinghit[ZLLCand_Z2l2Idx[ZLLbest"+_flag+"Idx]]}; return LepMissingHit")
                                                   .Define('PFMET', "MET_pt")
                                                   ## overallEventWeight contains everything in NanoAODs
                                                   .Define('L1prefiringWeight', "1") ## Dummy
                                                   .Define('KFactor_EW_qqZZ', "1") ## Dummy
                                                   .Define('KFactor_QCD_qqZZ_M', "1") ## Dummy
                                                   .Define('KFactor_QCD_ggZZ_Nominal', '1') ## Dummy
                                                   .Define('xsec', '1') ## Dummy
                                                   )
    return df_out

def FindFinalState(z1_flav, z2_flav):
    if(z1_flav == -121):
        if(z2_flav == +121): return 0 # 4e
        if(z2_flav == +169): return 2 # 2e2mu
    if(z1_flav == -169):
        if(z2_flav == +121): return 3 # 2mu2e
        if(z2_flav == +169): return 1 # 4mu

def findFSZX(df):
    df['FinState'] = np.asarray([FindFinalState(x, y) for x, y in zip(df['Z1Flav'], df['Z2Flav'])], dtype=np.int32)
    return df

def openFR(year):
    
    if (year == "2022" or year == "2022EE" or year == "2023preBPix" or year == "2023postBPix" or year == "2024"):
        fnameFR = "/eos/cms/store/group/phys_higgs/cmshzz4l/cjlst/HIG-25-015/RunIII_byZ1Z2/Moriond26_JES/FAKERATES/%s/FakeRates_SS_%s.root" % (year, year)
    else:
        raise ValueError(f"ERROR: Unsupported year")

    if not os.path.exists(fnameFR):
        raise FileNotFoundError(f"Fake rate file not found: {fnameFR}")

    f = ROOT.TFile.Open(fnameFR)
    FR_mu_EB = f.Get("FR_SS_muon_EB").Clone()
    FR_mu_EE = f.Get("FR_SS_muon_EE").Clone()
    FR_e_EB  = f.Get("FR_SS_electron_EB").Clone()
    FR_e_EE  = f.Get("FR_SS_electron_EE").Clone()

    f.Close()
    del f
    return FR_mu_EB, FR_mu_EE, FR_e_EB, FR_e_EE

def GetFakeRate(lep_Pt, lep_eta, lep_ID):
    if(lep_Pt >= 80):
        my_lep_Pt = 79
    else:
        my_lep_Pt = lep_Pt
    my_lep_ID = abs(lep_ID)
    if((my_lep_Pt > 5) & (my_lep_Pt <= 7)): bin = 0
    if((my_lep_Pt > 7) & (my_lep_Pt <= 10)): bin = 1
    if((my_lep_Pt > 10) & (my_lep_Pt <= 20)): bin = 2
    if((my_lep_Pt > 20) & (my_lep_Pt <= 30)): bin = 3
    if((my_lep_Pt > 30) & (my_lep_Pt <= 40)): bin = 4
    if((my_lep_Pt > 40) & (my_lep_Pt <= 50)): bin = 5
    if((my_lep_Pt > 50) & (my_lep_Pt <= 80)): bin = 6
    if(abs(my_lep_ID) == 11): bin = bin-1 # There is no [5, 7] bin in the electron fake rate # CHECK THIS - SPENCER
    if(my_lep_ID == 11):
        if(abs(lep_eta) < 1.479): return FR_e_EB.GetY()[bin]
        else: return FR_e_EE.GetY()[bin]
    if(my_lep_ID == 13):
        if(abs(lep_eta) < 1.2): return FR_mu_EB.GetY()[bin]
        else: return FR_mu_EE.GetY()[bin]

def ratio(year): # 2022 from HIG 24 13, 2023 from SPENCER
    if year == "2022":
        OS_SS = np.array([
            1.030,   # 4e
            1.165,  # 4mu
            0.966,   # 2e2mu
            1.041,  # 2mu2e
            ])
    elif year == "2022EE":
        OS_SS = np.array([
            0.990,   # 4e
            0.997,  # 4mu
            1.039,   # 2e2mu
            1.016,  # 2mu2e
            ])
    elif year == "2023preBPix":
        OS_SS = np.array([
            0.992,   # 4e
            1.024,  # 4mu
            1.102,   # 2e2mu
            1.024,  # 2mu2e
            ])
    elif year == "2023postBPix":
        OS_SS = np.array([
            1.006,   # 4e
            1.040,  # 4mu
            1.078,   # 2e2mu
            1.025,  # 2mu2e
            ])
    elif year == "2024": 
        OS_SS = np.array([
            0.997,   # 4e
            1.051,  # 4mu
            1.051,   # 2e2mu
            1.024,  # 2mu2e
            ])
    return OS_SS

def ZXYield(df, year, year_mc):
    cb_SS = comb(year_mc)
    OS_SS = ratio(year_mc)
    n_events = len(df['ZZMass'])
    Yield = np.zeros(n_events, float)
    for i in range(n_events):
        finSt  = df['FinState'][i]
        lepPt  = df['LeptonPt'][i]
        lepEta = df['LeptonEta'][i]
        lepID  = df['LeptonLepId'][i]
        Yield[i] = cb_SS[finSt] * OS_SS[finSt] * GetFakeRate(lepPt[2], lepEta[2], lepID[2]) * GetFakeRate(lepPt[3], lepEta[3], lepID[3])

    # if opt.doRun3:
    #     Yield *= 171/109
    return Yield

def main():
    parser = argparse.ArgumentParser(description='Calculate Z+X yields for the 4l analysis')
    parser.add_argument('--year', required=True, help='Data-taking year (e.g., 2022, 2022EE, 2023preBPix, 2023postBPix, 2024)')
    parser.add_argument('--year_mc', required=True, help='Year of the MC samples to use for coefficients (e.g., 2022, 2022EE, 2023preBPix, 2023postBPix, 2024)')
    parser.add_argument('--input_file', required=True, help='Path to the input ROOT file containing the TTree with the necessary branches')
    parser.add_argument('--output_file', required=False, default='output_with_ZX.root', help='Path to the output ROOT file to save the results (default: output_with_ZX.root)')
    args = parser.parse_args()

    # Load fake rates early
    global FR_mu_EB, FR_mu_EE, FR_e_EB, FR_e_EE
    print(f"Loading fake rates for year {args.year}...")
    FR_mu_EB, FR_mu_EE, FR_e_EB, FR_e_EE = openFR(args.year)
    print("Fake rates loaded successfully")

    # Example usage: python ZX_yield_calculator.py --year 2022 --year_mc 2022
    print(f"Calculating Z+X yields for year {args.year} using MC coefficients from {args.year_mc}...")

    df = ROOT.RDataFrame("Events", args.input_file)
    # Check for columns containing 'ZLL'
    column_names = df.GetColumnNames()
    zll_columns = [str(col) for col in column_names if 'ZLL' in str(col)]
    
    if zll_columns:
        print(f"Found {len(zll_columns)} columns with 'ZLL' in the name:")
        for col in zll_columns:
            print(f"  - {col}")
        
        # Create control region dataframe and convert to NumPy
        print("Creating control region...")
        df_CR_rdf = makeCR(df, 'SS')
        print(f"Control region filter applied")
        
        print("Materializing control region data (this may take a moment)...")
        df_CR = df_CR_rdf.AsNumpy()
        print(f"Control region has {len(df_CR['EventNumber'])} events")

        # Convert ROOT ndarrays to standard NumPy arrays
        df_CR = {key: np.asarray(val) for key, val in df_CR.items()}

        # Find final states and calculate yields
        print("Finding final states...")
        df_CR = findFSZX(df_CR)
        print("Calculating Z+X yields...")
        df_CR['ZX_Yield'] = ZXYield(df_CR, args.year, args.year_mc)
        print(f"Calculated Z+X yields for {len(df_CR['ZX_Yield'])} events")
        for fin_state in range(4):
            mask = df_CR['FinState'] == fin_state
            yield_sum = np.sum(df_CR['ZX_Yield'][mask])
            print(f"  FinState {fin_state}: Total ZX_Yield = {yield_sum:.4f} over {np.sum(mask)} events")

        # Add the python-computed ZX_Yield column to the original RDataFrame.
        # Use the event identity instead of rdfentry_, which refers to the original tree entry.
        ROOT.clearZXYieldValues()
        ROOT.clearFinStateValues()
        for run, lumi, event, value, fin_state in zip(
            np.asarray(df_CR['RunNumber']),
            np.asarray(df_CR['LumiNumber']),
            np.asarray(df_CR['EventNumber']),
            np.asarray(df_CR['ZX_Yield'], dtype=np.float64),
            np.asarray(df_CR['FinState'], dtype=np.int32),
        ):
            ROOT.setZXYieldValue(int(run), int(lumi), int(event), float(value))
            ROOT.setFinStateValue(int(run), int(lumi), int(event), int(fin_state))
        df_CR_rdf = df_CR_rdf.Define("ZX_Yield", "getZXYieldFromEvent(run, luminosityBlock, event)")\
                            .Define("FinState", "getFinStateFromEvent(run, luminosityBlock, event)")
        

        # Snapshot directly from RDataFrame to preserve native ROOT branch types.
        try:
            df_CR_rdf.Snapshot("Events", args.output_file)
            print(f"Saved control-region dataframe (with ZX_Yield) to ROOT file: {args.output_file}")
        except Exception as save_err:
            print(f"ROOT snapshot failed ({save_err}). Falling back to manual writer...")
            try:
                df_CR = {key: np.asarray(val) for key, val in df_CR.items()}
                write_root_tree_from_dict(df_CR, args.output_file, "Events")
                print(f"Saved control-region dataframe via fallback writer: {args.output_file}")
            except Exception as fallback_err:
                print(f"Fallback ROOT tree writing failed ({fallback_err}).")

        fallback_path = args.output_file
        if not fallback_path.endswith(".npz"):
            fallback_path = f"{args.output_file}.npz"
        np.savez_compressed(fallback_path, **df_CR)
        print(f"Saved full control-region arrays as NPZ: {fallback_path}")


if __name__ == "__main__":
    main()
