"""Data loading functions for multiclass BDT training."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Union

import uproot
from sklearn.preprocessing import LabelEncoder


def load_data(
    data_path: Union[str, List[str]],
    target_column: str,
    feature_columns: List[str],
    tree_name: str,
    weight_column: Optional[str] = None,
    EventWeight_column: Optional[str] = None,
    mode_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], list, np.ndarray]:
    """
    Load data from ROOT file(s) in a memory-efficient way.
    
    Only loads the columns that are actually needed (target, features, weights),
    processing files one at a time to minimize memory usage.
    
    Args:
        data_path: Path to ROOT file(s) - can be single path or list of paths
        target_column: Name of the target/class column
        feature_columns: List of feature column names
        tree_name: Name of the TTree in ROOT file
        weight_column: Name of the weight column (optional)
        EventWeight_column: Name of the EventWeight column (optional)
        mode_name: Name of the mode/category column (optional)
        
    Returns:
        Tuple of (X, y, weights, event_weights, modes, feature_names, class_names)
    """
    if isinstance(data_path, str):
        data_paths = [data_path]
    else:
        data_paths = data_path
    
    print(f"Loading data from {len(data_paths)} ROOT file(s)...")
    print(f"Using TTree: {tree_name}")
    
    columns_to_load = list(set(
        [target_column] + feature_columns +
        ([weight_column] if weight_column else []) +
        ([EventWeight_column] if EventWeight_column else []) +
        ([mode_name] if mode_name else [])
    ))
    print(f"Loading {len(columns_to_load)} columns (target + features + weights)")
    
    x_parts = []
    y_parts = []
    weights_parts = [] if weight_column else None
    event_weights_parts = [] if EventWeight_column else None
    modes_parts = [] if mode_name else None
    categorical_features = set()  # Track which features are categorical
    
    for idx, path_str in enumerate(data_paths):
        data_path_obj = Path(path_str)
        print(f"  [{idx+1}/{len(data_paths)}] Loading: {data_path_obj}...")
        
        try:
            with uproot.open(data_path_obj) as file:
                tree = file[tree_name]
                
                y_part = tree[target_column].array(library='np')
                
                # Load features one by one to handle strings properly
                feature_arrays = []
                for col in feature_columns:
                    col_data = tree[col].array(library='np')
                    # Track categorical (string) features
                    if col_data.dtype.kind in ['O', 'S', 'U']:
                        categorical_features.add(col)
                        if col_data.dtype.kind == 'S':  # byte strings
                            col_data = np.char.decode(col_data, 'utf-8')
                    feature_arrays.append(col_data)
                
                X_part = np.column_stack(feature_arrays)
                
                w_part = None
                if weight_column:
                    w_part = tree[weight_column].array(library='np')
                    n_negative = np.sum(w_part < 0)
                    n_zero = np.sum(w_part == 0)
                    if n_negative > 0 or n_zero > 0:
                        print(f"    Weight column '{weight_column}': {n_negative} negative, {n_zero} zero out of {len(w_part)} samples")
                
                ew_part = None
                if EventWeight_column:
                    ew_part = tree[EventWeight_column].array(library='np')
                
                mode_part = None
                if mode_name:
                    mode_part = tree[mode_name].array(library='np')
                    # Convert bytes to strings if necessary
                    if mode_part.dtype.kind == 'S':
                        mode_part = np.char.decode(mode_part, 'utf-8')
                
                x_parts.append(X_part)
                y_parts.append(y_part)
                if weights_parts is not None:
                    weights_parts.append(w_part)
                if event_weights_parts is not None:
                    event_weights_parts.append(ew_part)
                if modes_parts is not None:
                    modes_parts.append(mode_part)
                
                print(f"    Loaded {len(y_part)} events")
                
        except Exception as e:
            raise RuntimeError(f"Error reading ROOT file {data_path_obj}: {e}") from e
    
    print(f"\nConcatenating data from {len(x_parts)} file(s)...")
    
    x = x_parts[0] if len(x_parts) == 1 else np.vstack(x_parts)
    del x_parts
    
    y = y_parts[0] if len(y_parts) == 1 else np.concatenate(y_parts)
    del y_parts
    
    weights = None
    if weights_parts is not None:
        weights = weights_parts[0] if len(weights_parts) == 1 else np.concatenate(weights_parts)
        del weights_parts
    
    print(f"Total loaded: {len(x)} events with {len(feature_columns)} features")
    
    # Convert all features to float
    x = x.astype(float)

    # Initialize event_weights and modes
    event_weights = None
    modes = None
    
    # Encode labels if they're strings
    label_encoder = LabelEncoder()
    if y.dtype == 'object' or y.dtype.kind == 'U':
        y = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
    else:
        class_names = np.unique(y)
    
    print(f"Number of classes: {len(class_names)}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution:\n{pd.Series(counts, index=unique).sort_index()}")
    
    # Handle event_weights and modes if not already processed
    if event_weights_parts is not None and event_weights is None:
        event_weights = event_weights_parts[0] if len(event_weights_parts) == 1 else np.concatenate(event_weights_parts)
        del event_weights_parts
    
    if modes_parts is not None and modes is None:
        modes = modes_parts[0] if len(modes_parts) == 1 else np.concatenate(modes_parts)
        del modes_parts
    
    return x, y, weights, event_weights, modes, feature_columns, class_names
