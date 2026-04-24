# Multiclass DNN for STXS Classification

This module implements a Deep Neural Network (DNN) classifier for STXS (Simplified Template Cross Section) categorization using PyTorch.

## Structure

- `trainer.py` - Main DNN trainer class with hyperparameter tuning support
- `utils.py` - Utility functions for model evaluation and saving
- `data_loader.py` - Data loading utilities (shared with BDT)
- `config.py` - Configuration and constants (shared with BDT)
- `plotting.py` - Plotting utilities (shared with BDT)
- `__init__.py` - Module initialization

## Features

### DNN Architecture
- Configurable hidden layers and dimensions
- Batch normalization
- Multiple activation functions (ReLU, ELU, LeakyReLU)
- Dropout for regularization
- Early stopping

### Training Features
- Label merging using STXS_1_2_MERGE_Helper_PARTIAL
- Filtering of categories with insufficient events
- Removal of FWDH categories
- Class label remapping to consecutive integers
- GPU support
- Hyperparameter tuning with Optuna

### Outputs
- Trained DNN model (.pt format)
- Model metadata (JSON)
- Yield tables (CSV)
- Yield table density plots (PDF)
- Confusion matrix (PDF)
- Run2 categorization comparison (optional)

## Usage

### Basic Training

```bash
python train_multiclass_DNN.py \
    --data path/to/data.root \
    --target HTXS_stage1_2_cat_pTjet30GeV \
    --features feat1 feat2 feat3 \
    --tree-name Events \
    --output models/my_dnn_model \
    --plot
```

### With Hyperparameter Tuning

```bash
python train_multiclass_DNN.py \
    --data path/to/data.root \
    --target HTXS_stage1_2_cat_pTjet30GeV \
    --features feat1 feat2 feat3 \
    --tree-name Events \
    --output models/my_dnn_model \
    --tune-hyperparameters 30 \
    --use-gpu \
    --plot
```

### Custom Architecture

```bash
python train_multiclass_DNN.py \
    --data path/to/data.root \
    --target HTXS_stage1_2_cat_pTjet30GeV \
    --features feat1 feat2 feat3 \
    --tree-name Events \
    --hidden-dims 512 256 128 \
    --dropout 0.4 \
    --learning-rate 0.0005 \
    --batch-size 512 \
    --n-epochs 150 \
    --activation elu \
    --output models/my_dnn_model
```

### Using the Run Script

A convenience script is provided:

```bash
bash run_DNN.sh
```

Edit `run_DNN.sh` to modify data paths, hyperparameters, and output directories.

## Key Differences from BDT

1. **Model Type**: Neural network vs gradient boosted trees
2. **Training**: Epoch-based with early stopping vs boosting rounds
3. **Hyperparameters**: Learning rate, batch size, dropout vs tree depth, subsample
4. **Output Format**: PyTorch .pt file vs XGBoost .pkl file
5. **Feature Importance**: Not directly available in DNN (can use gradient-based methods)

## Requirements

- PyTorch
- NumPy
- Pandas
- scikit-learn
- Optuna (for hyperparameter tuning)
- uproot (for ROOT file reading)
- matplotlib (for plotting)

## Notes

- The DNN uses the same label merging strategy as the BDT (STXS_1_2_MERGE_Helper_PARTIAL)
- GPU training is recommended for faster training
- Hyperparameter tuning typically requires 30-100 trials for good results
- Early stopping prevents overfitting with patience=10 epochs
