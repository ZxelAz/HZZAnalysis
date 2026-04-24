"""Utility functions for multiclass DNN training."""

import pickle
import json
from pathlib import Path
from typing import Union, Dict
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, accuracy_score, log_loss

from .config import STXS_STAGE_0_DICT, STXS_STAGE_1_2_DICT_MERGED


def consolidate_mode(mode_name: str) -> str:
    """
    Consolidate ggZZ and EWK background modes into single categories.
    
    Args:
        mode_name: Original mode name
        
    Returns:
        Consolidated mode name
    """
    # EWK modes
    if mode_name in ['WZZ', 'WWZ', 'ZZZ']:
        return 'EWK'
    # ggZZ continuum modes
    elif 'ggTo' in mode_name and 'Contin_MCFM701' in mode_name:
        return 'ggZZ'
    elif mode_name in ["WplusH125", "WminusH125"]:
        return 'WH'
    else:
        return mode_name


def save_model(
    model,
    feature_names: list,
    class_names: Union[list, np.ndarray],
    output_path: str,
    scaler=None
):
    """
    Save DNN model and metadata.
    
    Args:
        model: Trained PyTorch model
        feature_names: List of feature names
        class_names: List of class names
        output_path: Output path for saved model (without extension)
        scaler: StandardScaler object (optional)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = output_path.with_suffix('.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_structure': {
            'input_dim': model.network[0].in_features,
            'num_classes': model.network[-1].out_features,
            'hidden_dims': [layer.out_features for layer in model.network if isinstance(layer, torch.nn.Linear)][:-1]
        },
        'feature_names': feature_names,
        'class_names': class_names
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save scaler
    if scaler is not None:
        scaler_path = str(output_path) + '_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    metadata = {
        'feature_names': feature_names,
        'class_names': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
        'n_features': len(feature_names),
        'n_classes': len(class_names),
        'model_type': 'DNN'
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


def evaluate_model(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    feature_names: list,
    device: torch.device = None
) -> Dict:
    """
    Evaluate trained DNN model on training and test sets.
    
    Args:
        model: Trained PyTorch model
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        class_names: List of class names
        feature_names: List of feature names
        device: PyTorch device
        
    Returns:
        Dictionary with evaluation metrics and predictions
    """
    print("\nEvaluating model...")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Get probability predictions
    with torch.no_grad():
        x_train_tensor = torch.FloatTensor(x_train).to(device)
        x_test_tensor = torch.FloatTensor(x_test).to(device)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(x_train_tensor).any() or torch.isinf(x_train_tensor).any():
            print("WARNING: NaN or Inf detected in x_train_tensor")
            print(f"  NaN count: {torch.isnan(x_train_tensor).sum().item()}")
            print(f"  Inf count: {torch.isinf(x_train_tensor).sum().item()}")
        
        train_outputs = model(x_train_tensor)
        test_outputs = model(x_test_tensor)
        
        # Check for NaN/Inf in model outputs
        if torch.isnan(train_outputs).any() or torch.isinf(train_outputs).any():
            print("WARNING: NaN or Inf detected in train_outputs")
            print(f"  NaN count: {torch.isnan(train_outputs).sum().item()}")
            print(f"  Inf count: {torch.isinf(train_outputs).sum().item()}")
        
        y_train_proba = torch.softmax(train_outputs, dim=1).cpu().numpy()
        y_test_proba = torch.softmax(test_outputs, dim=1).cpu().numpy()
    
    # Clip probabilities to avoid numerical issues with log_loss
    epsilon = 1e-10
    y_train_proba = np.clip(y_train_proba, epsilon, 1 - epsilon)
    y_test_proba = np.clip(y_test_proba, epsilon, 1 - epsilon)
    
    # Check for NaN in probabilities
    if np.isnan(y_train_proba).any():
        print("ERROR: NaN detected in y_train_proba after clipping")
        print(f"  NaN count: {np.isnan(y_train_proba).sum()}")
        # Replace NaN with uniform distribution
        nan_mask = np.isnan(y_train_proba).any(axis=1)
        y_train_proba[nan_mask] = 1.0 / y_train_proba.shape[1]
    
    if np.isnan(y_test_proba).any():
        print("ERROR: NaN detected in y_test_proba after clipping")
        print(f"  NaN count: {np.isnan(y_test_proba).sum()}")
        # Replace NaN with uniform distribution
        nan_mask = np.isnan(y_test_proba).any(axis=1)
        y_test_proba[nan_mask] = 1.0 / y_test_proba.shape[1]
    
    # Use standard argmax for predictions
    y_train_pred = y_train_proba.argmax(axis=1)
    y_test_pred = y_test_proba.argmax(axis=1)
    
    # Compute metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_logloss = log_loss(y_train, y_train_proba)
    test_logloss = log_loss(y_test, y_test_proba)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Log Loss: {train_logloss:.4f}")
    print(f"Test Log Loss: {test_logloss:.4f}")
    
    # Classification report
    print("\nClassification Report (Test Set):")
    target_names = [STXS_STAGE_1_2_DICT_MERGED.get(c, str(c)) for c in class_names]
    
    classification_rep = classification_report(
        y_test, y_test_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    print(classification_report(
        y_test, y_test_pred,
        target_names=target_names,
        zero_division=0
    ))
    
    # Extract weighted average f1-score
    weighted_f1 = classification_rep['weighted avg']['f1-score']
    print(f"\nWeighted Average F1-Score: {weighted_f1:.4f}")
    
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_logloss': train_logloss,
        'test_logloss': test_logloss,
        'weighted_f1': weighted_f1,
        'y_test_pred': y_test_pred,
    }
    
    return results


def first_step_categorisation(arr):
    """Perform first-step categorization.
    
    Args:
        arr: the merged array containing all events
        
    Returns:
        a dictionary of 7 arrays corresponding to the stage 0 HTXS categorization.
    """
    # Ensure all arrays are proper numpy arrays
    for key in arr:
        if isinstance(arr[key], list):
            arr[key] = np.asarray(arr[key])
        elif not isinstance(arr[key], np.ndarray):
            arr[key] = np.asarray(arr[key])
    
    # Convert numeric columns to appropriate types
    float_vars = [
        "ZZCand_nExtraLep", "DVBF2j_ME", "nBtagged_filtered",
        "DVBF1j_ME", "LepPdgId_4", "LepPdgId_5"
    ]
    int_vars = ["nCleanedJetsPt30"]
    
    for var in float_vars:
        if var in arr:
            try:
                arr[var] = arr[var].astype(float)
            except (ValueError, TypeError) as e:
                print(f"  Warning: Could not convert {var} to float: {e}")
    
    for var in int_vars:
        if var in arr:
            try:
                arr[var] = arr[var].astype(int)
            except (ValueError, TypeError) as e:
                print(f"  Warning: Could not convert {var} to int: {e}")
    
    # Define masks for each category
    VBF_2jet_tagged_mask = (arr["ZZCand_nExtraLep"]==0) & (arr["DVBF2j_ME"]>0.5) &\
                            (
                                ((np.isin(arr["nCleanedJetsPt30"], [2, 3])) & (arr["nBtagged_filtered"]<=1)) | 
                                ((arr["nCleanedJetsPt30"]==4) & (arr["nBtagged_filtered"]==0))
                            ) 
    
    VH_hadronic_tagged_mask = ~VBF_2jet_tagged_mask & \
                                (
                                    (arr["ZZCand_nExtraLep"]==0) & (arr["DWHh_ME"]>0.5) & (arr["DZHh_ME"]>0.5) & \
                                    ((np.isin(arr["nCleanedJetsPt30"], [2, 3])) | 
                                    ((arr["nCleanedJetsPt30"]==4) & (arr["nBtagged_filtered"]==0)))
                                ) 
    
    extra_OSSF_pair_cond = (arr["LepPdgId_4"]!=0) & (arr["LepPdgId_4"]+arr["LepPdgId_5"]==0) 
    VH_leptonic_tagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & \
                                (
                                (arr["ZZCand_nExtraLep"]<=3) & (arr["nBtagged_filtered"]==0) & \
                                 ((arr["ZZCand_nExtraLep"]==1) | extra_OSSF_pair_cond)
                                ) | \
                                ((arr["ZZCand_nExtraLep"]>=1) & (arr["nCleanedJetsPt30"]==0))
    
    ttH_hadronic_tagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & ~VH_leptonic_tagged_mask & \
                                (arr["nCleanedJetsPt30"]>=4) & (arr["nBtagged_filtered"]>=1) & (arr["ZZCand_nExtraLep"]==0)
    
    ttH_leptonic_tagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & ~VH_leptonic_tagged_mask & ~ttH_hadronic_tagged_mask & \
                                (arr["ZZCand_nExtraLep"]>=1) 
    
    one_jet_tagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & ~VH_leptonic_tagged_mask & ~ttH_hadronic_tagged_mask & ~ttH_leptonic_tagged_mask & \
                                (arr["nCleanedJetsPt30"]==1) & (arr["ZZCand_nExtraLep"]==0) & (arr["DVBF1j_ME"]>0.7)
    Untagged_mask = ~VBF_2jet_tagged_mask & ~VH_hadronic_tagged_mask & ~VH_leptonic_tagged_mask & ~ttH_hadronic_tagged_mask & ~ttH_leptonic_tagged_mask & ~one_jet_tagged_mask
    
    # Create arrays for each category based on the masks
    VBF_2jet_tagged_arr = {var: arr[var][VBF_2jet_tagged_mask] for var in arr.keys()}
    VH_hadronic_tagged_arr = {var: arr[var][VH_hadronic_tagged_mask] for var in arr.keys()}
    VH_leptonic_tagged_arr = {var: arr[var][VH_leptonic_tagged_mask] for var in arr.keys()}
    ttH_hadronic_tagged_arr = {var: arr[var][ttH_hadronic_tagged_mask] for var in arr.keys()}
    ttH_leptonic_tagged_arr = {var: arr[var][ttH_leptonic_tagged_mask] for var in arr.keys()}
    one_jet_tagged_arr = {var: arr[var][one_jet_tagged_mask] for var in arr.keys()}
    Untagged_arr = {var: arr[var][Untagged_mask] for var in arr.keys()}

    print(f"VBF_2jet_tagged: {np.sum(VBF_2jet_tagged_mask)} events")
    print(f"VH_hadronic_tagged: {np.sum(VH_hadronic_tagged_mask)} events")
    print(f"VH_leptonic_tagged: {np.sum(VH_leptonic_tagged_mask)} events")
    print(f"ttH_hadronic_tagged: {np.sum(ttH_hadronic_tagged_mask)} events")
    print(f"ttH_leptonic_tagged: {np.sum(ttH_leptonic_tagged_mask)} events")
    print(f"one_jet_tagged: {np.sum(one_jet_tagged_mask)} events")
    print(f"Untagged: {np.sum(Untagged_mask)} events")

    return {
        "VBF_2jet_tagged": VBF_2jet_tagged_arr,
        "VH_hadronic_tagged": VH_hadronic_tagged_arr,
        "VH_leptonic_tagged": VH_leptonic_tagged_arr,
        "ttH_hadronic_tagged": ttH_hadronic_tagged_arr,
        "ttH_leptonic_tagged": ttH_leptonic_tagged_arr,
        "one_jet_tagged": one_jet_tagged_arr,
        "Untagged": Untagged_arr
    }


def second_step_categorisation(arr_dict):
    """Perform second-step categorization.
    
    Args:
        arr_dict: the output dictionary from first_step_categorisation
        
    Returns:
        a dictionary of 22 arrays corresponding to the stage 1.2 HTXS categorization.
    """
    # Convert numeric columns to float for consistency
    numeric_vars = [
        "nCleanedJetsPt30", "ZZCand_pt", "m_jj", "ZZjj_pt"
    ]
    
    for category in arr_dict:
        for var in numeric_vars:
            if var in arr_dict[category]:
                try:
                    arr_dict[category][var] = arr_dict[category][var].astype(float)
                except (ValueError, TypeError) as e:
                    print(f"  Warning: Could not convert {category}/{var}: {e}")
    
    # mask definitions

    # Untagged subcategories
    Untagged_0j_Pt0To10_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==0) & (arr_dict["Untagged"]["ZZCand_pt"]<10)
    Untagged_0j_Pt10To200_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==0) & (arr_dict["Untagged"]["ZZCand_pt"]>=10) & (arr_dict["Untagged"]["ZZCand_pt"]<=200)
    Untagged_1j_Pt0To60_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==1) & (arr_dict["Untagged"]["ZZCand_pt"]<60)
    Untagged_1j_Pt60To120_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==1) & (arr_dict["Untagged"]["ZZCand_pt"]>=60) & (arr_dict["Untagged"]["ZZCand_pt"]<=120)
    Untagged_1j_Pt120To200_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==1) & (arr_dict["Untagged"]["ZZCand_pt"]>120) & (arr_dict["Untagged"]["ZZCand_pt"]<=200)
    Untagged_2j_Pt0To60_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==2) & (arr_dict["Untagged"]["ZZCand_pt"]<60) & (arr_dict["Untagged"]["m_jj"]<=350)
    Untagged_2j_Pt60To120_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==2) & (arr_dict["Untagged"]["ZZCand_pt"]>=60) & (arr_dict["Untagged"]["ZZCand_pt"]<=120) & (arr_dict["Untagged"]["m_jj"]<=350)
    Untagged_2j_Pt120To200_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==2) & (arr_dict["Untagged"]["ZZCand_pt"]>120) & (arr_dict["Untagged"]["ZZCand_pt"]<=200) & (arr_dict["Untagged"]["m_jj"]<=350)
    Untagged_Pt200above_mask = (arr_dict["Untagged"]["ZZCand_pt"]>200)
    Untagged_2j_mjj350above_mask = (arr_dict["Untagged"]["nCleanedJetsPt30"]==2) & (arr_dict["Untagged"]["m_jj"]>350)

    # VBF 2-jet tagged subcategories
    VBF_2jet_tagged_mjj350To700_mask = (arr_dict["VBF_2jet_tagged"]["m_jj"]>=350) & (arr_dict["VBF_2jet_tagged"]["m_jj"]<700) & (arr_dict["VBF_2jet_tagged"]["ZZCand_pt"]<=200) & (arr_dict["VBF_2jet_tagged"]["ZZjj_pt"]<25) # need to include pt4ljj
    VBF_2jet_tagged_mjj700above_mask = (arr_dict["VBF_2jet_tagged"]["m_jj"]>=700) & (arr_dict["VBF_2jet_tagged"]["ZZCand_pt"]<=200) & (arr_dict["VBF_2jet_tagged"]["ZZjj_pt"]<25) # need to include pt4ljj
    VBF_3jet_tagged_mjj350above_mask = (arr_dict["VBF_2jet_tagged"]["m_jj"]>=350) & (arr_dict["VBF_2jet_tagged"]["ZZCand_pt"]<=200) & (arr_dict["VBF_2jet_tagged"]["nCleanedJetsPt30"]>=3) & (arr_dict["VBF_2jet_tagged"]["ZZjj_pt"]>25) # need to include pt4ljj 3jet? need to ask
    VBF_2jet_tagged_Pt200above_mask = (arr_dict["VBF_2jet_tagged"]["ZZCand_pt"]>200) & (arr_dict["VBF_2jet_tagged"]["m_jj"]>=350) # need to include pt4ljj
    VBF_rest = arr_dict["VBF_2jet_tagged"]["m_jj"]<350

    # VH hadronic tagged subcategories
    VH_hadronic_tagged_mjj60To120_mask = (arr_dict["VH_hadronic_tagged"]["m_jj"]>=60) & (arr_dict["VH_hadronic_tagged"]["m_jj"]<120)
    VH_hadronic_tagged_rest_mask = (arr_dict["VH_hadronic_tagged"]["m_jj"]<60) | (arr_dict["VH_hadronic_tagged"]["m_jj"]>=120)

    # VH leptonic tagged subcategories
    VH_leptonic_tagged_Pt0To150_mask = (arr_dict["VH_leptonic_tagged"]["ZZCand_pt"]<150)
    VH_leptonic_tagged_Pt150above_mask = (arr_dict["VH_leptonic_tagged"]["ZZCand_pt"]>=150)

    # Create arrays for each category based on the masks
    Untagged_0j_Pt0To10_arr = {var: arr_dict["Untagged"][var][Untagged_0j_Pt0To10_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_0j_Pt10To200_arr = {var: arr_dict["Untagged"][var][Untagged_0j_Pt10To200_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_1j_Pt0To60_arr = {var: arr_dict["Untagged"][var][Untagged_1j_Pt0To60_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_1j_Pt60To120_arr = {var: arr_dict["Untagged"][var][Untagged_1j_Pt60To120_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_1j_Pt120To200_arr = {var: arr_dict["Untagged"][var][Untagged_1j_Pt120To200_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_2j_Pt0To60_arr = {var: arr_dict["Untagged"][var][Untagged_2j_Pt0To60_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_2j_Pt60To120_arr = {var: arr_dict["Untagged"][var][Untagged_2j_Pt60To120_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_2j_Pt120To200_arr = {var: arr_dict["Untagged"][var][Untagged_2j_Pt120To200_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_Pt200above_arr = {var: arr_dict["Untagged"][var][Untagged_Pt200above_mask] for var in arr_dict["Untagged"].keys()}
    Untagged_2j_mjj350above_arr = {var: arr_dict["Untagged"][var][Untagged_2j_mjj350above_mask] for var in arr_dict["Untagged"].keys()}

    VBF_2jet_tagged_mjj350To700_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_2jet_tagged_mjj350To700_mask] for var in arr_dict["VBF_2jet_tagged"].keys()}
    VBF_2jet_tagged_mjj700above_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_2jet_tagged_mjj700above_mask] for var in arr_dict["VBF_2jet_tagged"].keys()}
    VBF_3jet_tagged_mjj350above_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_3jet_tagged_mjj350above_mask] for var in arr_dict["VBF_2jet_tagged"].keys()}
    VBF_2jet_tagged_Pt200above_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_2jet_tagged_Pt200above_mask] for var in arr_dict["VBF_2jet_tagged"].keys()}
    VBF_rest_arr = {var: arr_dict["VBF_2jet_tagged"][var][VBF_rest] for var in arr_dict["VBF_2jet_tagged"].keys()}

    VH_hadronic_tagged_mjj60To120_arr = {var: arr_dict["VH_hadronic_tagged"][var][VH_hadronic_tagged_mjj60To120_mask] for var in arr_dict["VH_hadronic_tagged"].keys()}
    VH_hadronic_tagged_rest_arr = {var: arr_dict["VH_hadronic_tagged"][var][VH_hadronic_tagged_rest_mask] for var in arr_dict["VH_hadronic_tagged"].keys()}

    VH_leptonic_tagged_Pt0To150_arr = {var: arr_dict["VH_leptonic_tagged"][var][VH_leptonic_tagged_Pt0To150_mask] for var in arr_dict["VH_leptonic_tagged"].keys()}
    VH_leptonic_tagged_Pt150above_arr = {var: arr_dict["VH_leptonic_tagged"][var][VH_leptonic_tagged_Pt150above_mask] for var in arr_dict["VH_leptonic_tagged"].keys()}

    return {
        "Untagged_0j_Pt0To10": Untagged_0j_Pt0To10_arr,
        "Untagged_0j_Pt10To200": Untagged_0j_Pt10To200_arr,
        "Untagged_1j_Pt0To60": Untagged_1j_Pt0To60_arr,
        "Untagged_1j_Pt60To120": Untagged_1j_Pt60To120_arr,
        "Untagged_1j_Pt120To200": Untagged_1j_Pt120To200_arr,
        "Untagged_2j_Pt0To60": Untagged_2j_Pt0To60_arr,
        "Untagged_2j_Pt60To120": Untagged_2j_Pt60To120_arr,
        "Untagged_2j_Pt120To200": Untagged_2j_Pt120To200_arr,
        "Untagged_Pt200above": Untagged_Pt200above_arr,
        "Untagged_2j_mjj350above": Untagged_2j_mjj350above_arr,
        "VBF_1jet_tagged": arr_dict["one_jet_tagged"],
        "VBF_2jet_tagged_mjj350To700": VBF_2jet_tagged_mjj350To700_arr,
        "VBF_2jet_tagged_mjj700above": VBF_2jet_tagged_mjj700above_arr,
        "VBF_3jet_tagged_mjj350above": VBF_3jet_tagged_mjj350above_arr,
        "VBF_2jet_tagged_Pt200above": VBF_2jet_tagged_Pt200above_arr,
        "VBF_rest": VBF_rest_arr,
        "VH_hadronic_tagged_mjj60To120": VH_hadronic_tagged_mjj60To120_arr,
        "VH_hadronic_tagged_rest": VH_hadronic_tagged_rest_arr,
        "VH_leptonic_tagged_Pt0To150": VH_leptonic_tagged_Pt0To150_arr,
        "VH_leptonic_tagged_Pt150above": VH_leptonic_tagged_Pt150above_arr,
        "ttH_hadronic_tagged": arr_dict["ttH_hadronic_tagged"],
        "ttH_leptonic_tagged": arr_dict["ttH_leptonic_tagged"]
    }
