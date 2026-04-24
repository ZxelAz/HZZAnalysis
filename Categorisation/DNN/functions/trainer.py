"""Main trainer class for multiclass DNN."""

import numpy as np
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, f1_score, accuracy_score
import optuna

from .utils import evaluate_model
from .config import (
    STXS_STAGE_0_DICT,
    STXS_STAGE_1_2_DICT_MERGED,
    STXS_1_2_MERGE_Helper,
    FWDH_CATEGORIES_TO_EXCLUDE,
    MIN_EVENTS_PER_CATEGORY
)


class MulticlassDNN(nn.Module):
    """Deep Neural Network for multiclass classification."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [256, 128, 64], 
                 dropout: float = 0.3, activation: str = 'relu'):
        """
        Initialize DNN architecture.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ('relu', 'elu', 'leaky_relu')
        """
        super(MulticlassDNN, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MulticlassDNNTrainer:
    """Trainer class for multiclass DNN."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names = None
        self.class_names = None
        
        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def _can_stratify(self, labels: np.ndarray) -> bool:
        """Check if stratification is possible."""
        if labels.size == 0:
            return False
        _, counts = np.unique(labels, return_counts=True)
        return counts.min() >= 2
    
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        event_weights: Optional[np.ndarray] = None,
        modes: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        hyperparameter_tuning: bool = False,
        n_trials: int = 30,
        use_gpu: bool = False,
        **model_params
    ) -> Dict:
        """Train the DNN model."""
        print("\n" + "="*60)
        print("Training Multiclass DNN")
        print("="*60)
        
        # Filter categories with < MIN_EVENTS_PER_CATEGORY events
        unique_classes, class_counts = np.unique(y, return_counts=True)
        classes_to_keep = unique_classes[class_counts >= MIN_EVENTS_PER_CATEGORY]
        classes_to_drop = unique_classes[class_counts < MIN_EVENTS_PER_CATEGORY]
        
        if len(classes_to_drop) > 0:
            print(f"\nWARNING: Found {len(classes_to_drop)} category/categories with < {MIN_EVENTS_PER_CATEGORY} events")
            mask = np.isin(y, classes_to_keep)
            x, y = x[mask], y[mask]
            if weights is not None:
                weights = weights[mask]
            if event_weights is not None:
                event_weights = event_weights[mask]
            if modes is not None:
                modes = modes[mask]
            print(f"Remaining events: {len(x)}")
        
        # Drop FWDH categories
        mask_exclude = ~np.isin(y, FWDH_CATEGORIES_TO_EXCLUDE)
        n_excluded = np.sum(~mask_exclude)
        
        if n_excluded > 0:
            print(f"\nDropping FWDH categories {FWDH_CATEGORIES_TO_EXCLUDE}: {n_excluded} events")
            x, y = x[mask_exclude], y[mask_exclude]
            if weights is not None:
                weights = weights[mask_exclude]
            if event_weights is not None:
                event_weights = event_weights[mask_exclude]
            if modes is not None:
                modes = modes[mask_exclude]
            print(f"Remaining events: {len(x)}")
        
        # Merge STXS 1.2 categories (using partial merge)
        print("\nMerging STXS 1.2 categories...")
        for merged_category, original_categories in STXS_1_2_MERGE_Helper.items():
            mask = np.isin(y, original_categories)
            n_merged = np.sum(mask)
            if n_merged > 0:
                y[mask] = merged_category
                print(f"  Merged {original_categories} -> {merged_category}: {n_merged} events")
        
        if weights is not None:
            print(f"\nWeight statistics: min={np.min(weights):.6e}, max={np.max(weights):.6e}, mean={np.mean(weights):.6e}")
        
        # Remap class labels to consecutive integers
        unique_classes_remaining = np.unique(y)
        sorted_original_classes = sorted(unique_classes_remaining)
        class_mapping = {orig_class: new_class for new_class, orig_class in enumerate(sorted_original_classes)}
        
        if isinstance(self.class_names, np.ndarray):
            self.class_names = np.array([c for c in sorted_original_classes])
        else:
            self.class_names = sorted_original_classes
        
        y_remapped = np.array([class_mapping[orig] for orig in y])
        print(f"\nRemapped {len(unique_classes_remaining)} classes to [0, {len(unique_classes_remaining)-1}]")
        print(f"Original class IDs: {sorted_original_classes}")
        print(f"Class names: {[STXS_STAGE_1_2_DICT_MERGED.get(c, str(c)) for c in sorted_original_classes]}")
        
        # Split data
        print("\n" + "="*60)
        print("Splitting data...")
        print("="*60)
        
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y_remapped, test_size=test_size, random_state=self.random_state,
            stratify=y_remapped if self._can_stratify(y_remapped) else None
        )
        
        val_ratio = val_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=val_ratio, random_state=self.random_state,
            stratify=y_trainval if self._can_stratify(y_trainval) else None
        )
        
        # Split weights similarly
        sample_weights = None
        if weights is not None:
            w_trainval, w_test = train_test_split(
                weights, test_size=test_size, random_state=self.random_state,
                stratify=y_remapped if self._can_stratify(y_remapped) else None
            )
            w_train, w_val = train_test_split(
                w_trainval, test_size=val_ratio, random_state=self.random_state,
                stratify=y_trainval if self._can_stratify(y_trainval) else None
            )
            sample_weights = w_train
        
        # Split event_weights and modes for evaluation
        event_weights_test = None
        if event_weights is not None:
            _, event_weights_test = train_test_split(
                event_weights, test_size=test_size, random_state=self.random_state,
                stratify=y_remapped if self._can_stratify(y_remapped) else None
            )
        
        modes_test = None
        if modes is not None:
            _, modes_test = train_test_split(
                modes, test_size=test_size, random_state=self.random_state,
                stratify=y_remapped if self._can_stratify(y_remapped) else None
            )
        
        print(f"Train set: {len(x_train)} samples")
        print(f"Validation set: {len(x_val)} samples")
        print(f"Test set: {len(x_test)} samples")
        
        # Replace sentinel missing values (-999) using train-set medians
        # and add indicator variables for imputed values.
        missing_value = -999.0
        print(f"\nImputing sentinel value {missing_value} with train-set medians...")
        x_train_impute = x_train.astype(float, copy=True)
        x_train_impute[x_train_impute == missing_value] = np.nan
        feature_medians = np.nanmedian(x_train_impute, axis=0)
        feature_medians = np.where(np.isnan(feature_medians), 0.0, feature_medians)
        
        def _impute_missing_with_indicator(x_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_data = x_data.astype(float, copy=True)
            missing_mask = (x_data == missing_value)
            x_data[missing_mask] = np.nan
            nan_rows, nan_cols = np.where(np.isnan(x_data))
            if nan_rows.size > 0:
                x_data[nan_rows, nan_cols] = feature_medians[nan_cols]
            indicator = missing_mask.astype(float)
            return x_data, indicator
        
        x_train, x_train_ind = _impute_missing_with_indicator(x_train)
        x_val, x_val_ind = _impute_missing_with_indicator(x_val)
        x_test, x_test_ind = _impute_missing_with_indicator(x_test)
        
        # Append indicator variables to feature matrix
        x_train = np.hstack([x_train, x_train_ind])
        x_val = np.hstack([x_val, x_val_ind])
        x_test = np.hstack([x_test, x_test_ind])
        
        if self.feature_names is not None:
            indicator_names = [f"{name}_is_missing" for name in self.feature_names]
            self.feature_names = list(self.feature_names) + indicator_names
            print(f"Added {len(indicator_names)} missing-value indicator features")
        
        # Normalize features using StandardScaler
        # print("\nNormalizing features with StandardScaler...")
        # self.scaler = StandardScaler()
        # x_train = self.scaler.fit_transform(x_train)
        # x_val = self.scaler.transform(x_val)
        # x_test = self.scaler.transform(x_test)
        # print("Features scaled: mean=0, std=1")
        
        # Check for NaN/Inf after scaling
        if np.isnan(x_train).any() or np.isinf(x_train).any():
            print("WARNING: NaN or Inf detected after scaling in x_train")
            print(f"  NaN count: {np.isnan(x_train).sum()}")
            print(f"  Inf count: {np.isinf(x_train).sum()}")
            # Replace NaN/Inf with 0
            x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
            x_val = np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0)
            x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
            print("  Replaced NaN/Inf with 0.0")
        
        # Hyperparameter tuning or direct training
        if hyperparameter_tuning:
            print("\n" + "="*60)
            print(f"Hyperparameter tuning with Optuna ({n_trials} trials)...")
            print("="*60)
            
            best_params = self._tune_hyperparameters(
                x_train, y_train, x_val, y_val,
                sample_weights, n_trials, use_gpu
            )
            print(f"\nBest hyperparameters: {best_params}")
        else:
            # Use provided or default parameters
            best_params = {
                'hidden_dims': model_params.get('hidden_dims', [256, 128, 64]),
                'dropout': model_params.get('dropout', 0.3),
                'learning_rate': model_params.get('learning_rate', 0.001),
                'batch_size': model_params.get('batch_size', 256),
                'n_epochs': model_params.get('n_epochs', 100),
                'weight_decay': model_params.get('weight_decay', 1e-5),
                'activation': model_params.get('activation', 'relu')
            }
        
        # Train final model
        print("\n" + "="*60)
        print("Training final model...")
        print("="*60)
        
        self.model = self._train_model(
            x_train, y_train, x_val, y_val,
            sample_weights, best_params, use_gpu
        )
        
        # Evaluate
        eval_results = evaluate_model(
            self.model,
            x_train, y_train,
            x_test, y_test,
            self.class_names,
            self.feature_names,
            device=self.device
        )
        
        results = {
            **eval_results,
            'model': self.model,
            'scaler': self.scaler,
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test,
            'event_weights_test': event_weights_test,
            'modes_test': modes_test,
            'best_params': best_params,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        
        return results
    
    def _train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray],
        params: dict,
        use_gpu: bool
    ) -> nn.Module:
        """Train a single model with given parameters."""
        
        device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
        print(f"Using device: {device}")
        
        # Create model
        input_dim = x_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        model = MulticlassDNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=params['hidden_dims'],
            dropout=params['dropout'],
            activation=params.get('activation', 'relu')
        ).to(device)
        
        # Loss and optimizer
        # Use reduction='none' to apply sample weights manually
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params.get('weight_decay', 1e-5)
        )
        
        # Prepare data loaders
        if sample_weights is not None:
            train_dataset = TensorDataset(
                torch.FloatTensor(x_train),
                torch.LongTensor(y_train),
                torch.FloatTensor(sample_weights)
            )
            print(f"Using sample weights in training (mean={np.mean(sample_weights):.6e})")
        else:
            train_dataset = TensorDataset(
                torch.FloatTensor(x_train),
                torch.LongTensor(y_train)
            )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(x_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(params['n_epochs']):
            # Training
            model.train()
            train_loss = 0.0
            for batch_data in train_loader:
                if sample_weights is not None:
                    batch_x, batch_y, batch_w = batch_data
                    batch_x, batch_y, batch_w = batch_x.to(device), batch_y.to(device), batch_w.to(device)
                else:
                    batch_x, batch_y = batch_data
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss_per_sample = criterion(outputs, batch_y)
                
                # Apply sample weights if available
                if sample_weights is not None:
                    loss = (loss_per_sample * batch_w).mean()
                else:
                    loss = loss_per_sample.mean()
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Check for NaN in training loss
            if np.isnan(train_loss) or np.isinf(train_loss):
                print(f"ERROR: NaN or Inf detected in training loss at epoch {epoch+1}")
                print(f"  train_loss: {train_loss}")
                print("  Stopping training early")
                break
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss_per_sample = criterion(outputs, batch_y)
                    loss = loss_per_sample.mean()
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{params['n_epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return model
    
    def _tune_hyperparameters(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray],
        n_trials: int,
        use_gpu: bool
    ) -> dict:
        """Tune hyperparameters using Optuna."""
        
        def objective(trial):
            # Suggest hyperparameters
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_dims = []
            for i in range(n_layers):
                dim = trial.suggest_categorical(f'hidden_dim_{i}', [64, 128, 256, 512])
                hidden_dims.append(dim)
            
            params = {
                'hidden_dims': hidden_dims,
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
                'n_epochs': 50,  # Fewer epochs for tuning
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu'])
            }
            
            # Train model
            model = self._train_model(
                x_train, y_train, x_val, y_val,
                sample_weights, params, use_gpu
            )
            
            # Evaluate on validation set
            device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
            model.eval()
            with torch.no_grad():
                x_val_tensor = torch.FloatTensor(x_val).to(device)
                outputs = model(x_val_tensor)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
            
            val_logloss = log_loss(y_val, proba)
            
            return val_logloss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Extract best params
        best_trial = study.best_trial
        best_params = {
            'hidden_dims': [best_trial.params[f'hidden_dim_{i}'] for i in range(best_trial.params['n_layers'])],
            'dropout': best_trial.params['dropout'],
            'learning_rate': best_trial.params['learning_rate'],
            'batch_size': best_trial.params['batch_size'],
            'n_epochs': 100,  # Use more epochs for final training
            'weight_decay': best_trial.params['weight_decay'],
            'activation': best_trial.params['activation']
        }
        
        return best_params
