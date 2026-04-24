"""Main trainer class for multiclass BDT."""

import numpy as np
from typing import Dict, Optional

import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import optuna

from .utils import evaluate_model
from .config import (
    STXS_STAGE_0_DICT,
    STXS_STAGE_1_2_DICT_MERGED,
    STXS_STAGE_1_2_DICT_PARTIAL_MERGED,
    STXS_1_2_MERGE_Helper,
    STXS_1_2_MERGE_Helper_PARTIAL,
    DEFAULT_XGBOOST_PARAMS,
    OPTUNA_SEARCH_SPACE,
    FWDH_CATEGORIES_TO_EXCLUDE,
    MIN_EVENTS_PER_CATEGORY
)


class MulticlassBDTTrainer:
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the trainer.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.feature_names = None
        self.class_names = None
    
    def compute_weight_sums(self, y: np.ndarray, weights: np.ndarray) -> Dict:
        """
        Compute the sum of weights for each class label.
        
        Args:
            y: Array of class labels (shape: [n_samples])
            weights: Array of sample weights (shape: [n_samples])
            
        Returns:
            Dictionary mapping class label -> total weight sum
            
        Example:
            >>> y = np.array([0, 0, 1, 1, 2])
            >>> weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> self.compute_weight_sums(y, weights)
            {0: 3.0, 1: 7.0, 2: 5.0}
        """
        unique_classes = np.unique(y)
        weight_sums = {}
        total_event_num = 0
        for cls in unique_classes:
            mask = y == cls
            weight_sums[cls] = np.sum(weights[mask])
            total_event_num += np.sum(mask)
            print(f"  Class {cls}: Total weight sum = {np.sum(weights[mask]):.4f}, Events = {np.sum(mask)}")
        class_weights = {cls: total_event_num / (10 * len(weight_sums) * weight_sums[cls]) for cls in weight_sums}
        print(f"  Class weights: {class_weights}")
        return class_weights

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
        """Train the BDT model."""
        print("\n" + "="*60)
        print("Training Multiclass BDT")
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
        
        # Merge STXS 1.2 categories (using partial merge: keep GG2H_GE2J_MJJ_GT350 and QQ2HQQ_rest unmerged)
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
            self.class_names = [c for c in sorted_original_classes]
        
        y = np.array([class_mapping[cls] for cls in y])
        
        # Split data into train, validation, and test sets
        # First split into train+val and test
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y, test_size=test_size, random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Then split train+val into train and validation
        val_ratio = val_size / (1 - test_size)  # Adjust val_size to remaining data
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=val_ratio, random_state=self.random_state,
            stratify=y_trainval if len(np.unique(y_trainval)) > 1 else None
        )
        
        # Split weights
        if weights is not None:
            w_trainval, w_test = train_test_split(
                weights, test_size=test_size, random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            w_train, w_val = train_test_split(
                w_trainval, test_size=val_ratio, random_state=self.random_state,
                stratify=y_trainval if len(np.unique(y_trainval)) > 1 else None
            )
        else:
            w_train, w_val, w_test = None, None, None
        
        # Split event weights
        if event_weights is not None:
            ew_trainval, ew_test = train_test_split(
                event_weights, test_size=test_size, random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            ew_train, ew_val = train_test_split(
                ew_trainval, test_size=val_ratio, random_state=self.random_state,
                stratify=y_trainval if len(np.unique(y_trainval)) > 1 else None
            )
        else:
            ew_train, ew_val, ew_test = None, None, None
        
        # Split modes
        if modes is not None:
            m_trainval, m_test = train_test_split(
                modes, test_size=test_size, random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            m_train, m_val = train_test_split(
                m_trainval, test_size=val_ratio, random_state=self.random_state,
                stratify=y_trainval if len(np.unique(y_trainval)) > 1 else None
            )
        else:
            m_train, m_val, m_test = None, None, None
        
        print("\nComputing class-balanced sample weights for TRAINING set...")
        class_weights = self.compute_weight_sums(y_train, w_train if w_train is not None else np.ones_like(y_train)) # normalize by weights if provided
        sample_weights = compute_sample_weight(class_weight=class_weights, y=y_train)
        
        if w_train is not None:
            print("Combining with existing event weights...")
            print(f"  Before combining - sample_weights: min={np.min(sample_weights):.6e}, max={np.max(sample_weights):.6e}")
            print(f"  Before combining - w_train: min={np.min(w_train):.6e}, max={np.max(w_train):.6e}")
            sample_weights = sample_weights * w_train
            print(f"  After combining - sample_weights: min={np.min(sample_weights):.6e}, max={np.max(sample_weights):.6e}")
            print(f"  Number of non-positive weights: {np.sum(sample_weights <= 0)}")
            
            # Print negative weights per class
            print("\n  Negative weights per class:")
            for cls in np.unique(y_train):
                mask = y_train == cls
                n_negative = np.sum(sample_weights[mask] < 0)
                n_zero = np.sum(sample_weights[mask] == 0)
                total_in_class = np.sum(mask)
                class_id = self.class_names[cls]
                class_name = STXS_STAGE_1_2_DICT_MERGED.get(class_id, class_id)
                print(f"    Class {cls} ({class_name}): {n_negative} negative, {n_zero} zero out of {total_in_class} samples")
            
            # Drop samples with negative or zero weights
            mask_positive_weights = sample_weights > 0
            n_dropped = np.sum(~mask_positive_weights)
            if n_dropped > 0:
                print(f"\n  Dropping {n_dropped} samples with non-positive weights...")
                x_train = x_train[mask_positive_weights]
                y_train = y_train[mask_positive_weights]
                sample_weights = sample_weights[mask_positive_weights]
                if m_train is not None:
                    m_train = m_train[mask_positive_weights]
                if ew_train is not None:
                    ew_train = ew_train[mask_positive_weights]
                print(f"  Remaining training samples: {len(x_train)}")
        
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Testing samples: {len(x_test)}")
        
        # Configure parameters
        default_params = DEFAULT_XGBOOST_PARAMS.copy()
        default_params.update({
            'num_class': len(np.unique(y)),
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        })
        
        if use_gpu:
            default_params['device'] = 'cuda'
            default_params['n_jobs'] = 1
            print("\nUsing GPU for training")
        
        default_params.update(model_params)
        
        # Prepare validation sample weights for eval_set
        val_sample_weights = None
        if w_val is not None:
            print("\nComputing class-balanced sample weights for VALIDATION set...")
            val_class_weights = self.compute_weight_sums(y_val, w_val)
            val_sample_weights = compute_sample_weight(class_weight=val_class_weights, y=y_val)
            val_sample_weights = val_sample_weights * w_val
            # Filter positive weights
            mask_positive = val_sample_weights > 0
            if np.sum(~mask_positive) > 0:
                print(f"Warning: Filtering {np.sum(~mask_positive)} non-positive weights from validation set")
                x_val = x_val[mask_positive]
                y_val = y_val[mask_positive]
                val_sample_weights = val_sample_weights[mask_positive]
        
        if hyperparameter_tuning:
            print("\nPerforming hyperparameter tuning with Optuna...")
            tuned_params = self._tune_hyperparameters(x_train, y_train, sample_weights, x_val, y_val, val_sample_weights, use_gpu, n_trials)
            # Retrain final model with eval_set to get learning curves
            print("\nRetraining final model with best parameters for learning curve tracking...")
            self.model = xgb.XGBClassifier(**tuned_params)
            eval_set = [(x_train, y_train), (x_val, y_val)]
            self.model.fit(
                x_train, y_train, 
                sample_weight=sample_weights,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model = xgb.XGBClassifier(**default_params)
            # Train with validation set for eval tracking (no early stopping)
            eval_set = [(x_train, y_train), (x_val, y_val)]
            self.model.fit(
                x_train, y_train, 
                sample_weight=sample_weights,
                eval_set=eval_set,
                verbose=False
            )
        
        # Evaluate
        eval_results = evaluate_model(
            self.model,
            x_train, y_train,
            x_test, y_test,
            self.class_names,
            self.feature_names
        )
        
        proba_train = self.model.predict_proba(x_train)
        proba_test = self.model.predict_proba(x_test)
        proba_val = self.model.predict_proba(x_val)

        results = {
            **eval_results,
            'model': self.model,
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'class_names': self.class_names.tolist() if hasattr(self.class_names, 'tolist') else list(self.class_names),
            'event_weights_train': ew_train,
            'event_weights_val': ew_val,
            'event_weights_test': ew_test,
            'modes_train': m_train,
            'modes_val': m_val,
            'modes_test': m_test,
            'sample_weights': sample_weights,
            'evals_result': self.model.evals_result() if hasattr(self.model, 'evals_result') else None,
            'proba_train': proba_train,
            'proba_val': proba_val,
            'proba_test': proba_test
        }
        
        return results
    
    def _tune_hyperparameters(self, x_train, y_train, sample_weights, x_val, y_val, event_weights_val, use_gpu, n_trials=30):
        """Perform hyperparameter tuning with Optuna using validation set with event weights."""
        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': len(np.unique(y_train)),
                'eval_metric': 'mlogloss',
                'max_depth': trial.suggest_int('max_depth', *OPTUNA_SEARCH_SPACE['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *OPTUNA_SEARCH_SPACE['learning_rate'], log=True),
                'n_estimators': trial.suggest_int('n_estimators', *OPTUNA_SEARCH_SPACE['n_estimators']),
                'subsample': trial.suggest_float('subsample', *OPTUNA_SEARCH_SPACE['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *OPTUNA_SEARCH_SPACE['colsample_bytree']),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', *OPTUNA_SEARCH_SPACE['colsample_bylevel']),
                'colsample_bynode': trial.suggest_float('colsample_bynode', *OPTUNA_SEARCH_SPACE['colsample_bynode']),
                'reg_lambda': trial.suggest_float('reg_lambda', *OPTUNA_SEARCH_SPACE['reg_lambda']),
                'reg_alpha': trial.suggest_float('reg_alpha', *OPTUNA_SEARCH_SPACE['reg_alpha']),
                'min_child_weight': trial.suggest_int('min_child_weight', *OPTUNA_SEARCH_SPACE['min_child_weight']),
                'gamma': trial.suggest_float('gamma', *OPTUNA_SEARCH_SPACE['gamma']),
                'random_state': self.random_state,
                'n_jobs': 1 if use_gpu else self.n_jobs,
                'tree_method': 'hist'
            }
            if use_gpu:
                params['device'] = 'cuda'
            
            # Train on training set and evaluate on validation set
            model = xgb.XGBClassifier(**params)
            model.fit(x_train, y_train, sample_weight=sample_weights)
            y_val_pred = model.predict(x_val)
            score = balanced_accuracy_score(y_val, y_val_pred, sample_weight=event_weights_val)
            # score = f1_score(y_val, y_val_pred, sample_weight=event_weights_val, average='macro')
            
            
            return score
        
        study = optuna.create_study(
            direction='maximize',
            study_name='xgb_multiclass_optimization',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        print(f"Starting Bayesian optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest parameters: {study.best_params}")
        print(f"Best validation score: {study.best_value:.4f}")
        
        final_params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'eval_metric': 'mlogloss',
            'random_state': self.random_state,
            'n_jobs': 1 if use_gpu else self.n_jobs,
            'tree_method': 'hist',
            **study.best_params
        }
        if use_gpu:
            final_params['device'] = 'cuda'
        
        return final_params
