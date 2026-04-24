import ROOT
import pickle
import numpy as np
import pandas as pd

def main():
    path = "${HZZ_ROOT}/InferenceModel/results/trial4"
    epsilon = pickle.load(open(f"{path}/epsilon.pkl", "rb"))
    acceptance = pickle.load(open(f"{path}/acceptance.pkl", "rb"))
    epsilonA = pickle.load(open(f"{path}/epsilonA.pkl", "rb"))
    N_bkg = pickle.load(open(f"{path}/N_bkg.pkl", "rb"))

    print("epsilon:", epsilon)
    print("acceptance:", acceptance)
    print("epsilonA:", epsilonA)
    print("N_bkg:", N_bkg)
    
    # Extract unique final states
    final_states = sorted(set(key[0] for key in epsilonA.keys()))
    
    print(f"\n\nFinal states: {final_states}\n")
    
    # Create a table for each final state
    tables = {}
    for fs in final_states:
        # Filter keys for this final state
        fs_data = {key: val for key, val in epsilonA.items() if key[0] == fs}
        
        # Extract unique categories and bins
        category_map = {}  # Maps original key to clean string
        for key in fs_data.keys():
            cat = key[1]
            # Convert to string representation
            if isinstance(cat, bytes):
                cat_str = cat.decode('utf-8', errors='ignore')
            elif isinstance(cat, (list, tuple)):
                # If it's a list/tuple, join the characters
                cat_str = ''.join(str(c) if not isinstance(c, bytes) else c.decode('utf-8', errors='ignore') for c in cat)
            else:
                cat_str = str(cat)
            
            category_map[cat] = cat_str
        
        categories = sorted(list(set(category_map.values())))
        bins = sorted(set(key[2] for key in fs_data.keys()))
        
        # Create a DataFrame
        data = np.zeros((len(categories), len(bins)))
        for i, cat_str in enumerate(categories):
            # Find original key that maps to this string
            original_cat = None
            for orig_cat, mapped_str in category_map.items():
                if mapped_str == cat_str:
                    original_cat = orig_cat
                    break
            
            for j, bin_val in enumerate(bins):
                key = (fs, original_cat, bin_val)
                if key in fs_data:
                    val = fs_data[key]
                    if isinstance(val, np.ndarray):
                        data[i, j] = float(val)
                    else:
                        data[i, j] = float(val)
        
        df = pd.DataFrame(data, index=categories, columns=bins)
        df.index.name = 'Category'
        tables[fs] = df
        
        print(f"\n{'='*80}")
        print(f"Final State: {fs}")
        print(f"{'='*80}")
        print(df)
        print(f"\nShape: ({len(categories)} categories, {len(bins)} bins)")
    
    # Save tables to CSV
    print(f"\n\n{'='*80}")
    print("Saving tables to CSV...")
    print(f"{'='*80}")
    for fs, df in tables.items():
        csv_path = f"{path}/epsilonA_{fs}.csv"
        df.to_csv(csv_path)
        print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()