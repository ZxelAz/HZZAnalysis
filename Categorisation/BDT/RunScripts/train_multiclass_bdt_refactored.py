#!/usr/bin/env python3
"""
Train a multiclass BDT classifier for STXS categorization.

Refactored version with modular structure.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import sys
import os

# Resolve HZZ_ROOT (two levels up from this script) so the Categorisation package
# is importable regardless of how the script is launched.
_HZZ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _HZZ_ROOT not in sys.path:
    sys.path.insert(0, _HZZ_ROOT)

from Categorisation.BDT.functions import MulticlassBDTTrainer, load_data, save_model
from Categorisation.BDT.functions.config import (
    DEFAULT_XGBOOST_PARAMS,
    STXS_1_2_MERGE_Helper,
    STXS_STAGE_1_2_DICT_MERGED,
    STXS_STAGE_1_2_DICT_PARTIAL_MERGED,
    STXS_STAGE_1_2_DICT,
    STXS_1_2_MERGE_Helper_PARTIAL,
)
from Categorisation.BDT.functions.plotting import (
    plot_confusion_matrix,
    plot_roc_auc_curve,
    plot_feature_importance,
    plot_yield_table_density,
    plot_learning_curves,
    plot_training_size_curve,
    compute_yield_table,
    yield_table_run2_categorisation,
)


def main():
    class_name_dict_options = {
        'merged': STXS_STAGE_1_2_DICT_MERGED,
        'partial_merged': STXS_STAGE_1_2_DICT_PARTIAL_MERGED,
        'stage1_2': STXS_STAGE_1_2_DICT,
    }

    parser = argparse.ArgumentParser(
        description='Train a multiclass BDT classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', '-d', type=str, nargs='+', required=True,
                        help='Path(s) to input ROOT file(s)')
    parser.add_argument('--target', '-t', type=str, required=True,
                        help='Name of the target/class column')
    parser.add_argument('--features', '-f', type=str, nargs='+', required=True,
                        help='List of feature column names')
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Name of the weight column (optional)')
    parser.add_argument('--EventWeight', type=str, default=None,
                        help='Name of the EventWeight column for yield table (optional)')
    parser.add_argument('--mode-name', type=str, default=None,
                        help='Name of the mode/category column (optional)')
    parser.add_argument('--tree-name', type=str, required=True,
                        help='Name of the TTree in ROOT file')
    parser.add_argument('--output', '-o', type=str, default='multiclass_bdt_model',
                        help='Output path for saved model (without extension)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--tune-hyperparameters', type=int, default=None, metavar='N_TRIALS',
                        help='Perform hyperparameter tuning with Optuna using N_TRIALS trials')
    parser.add_argument('--max-depth', type=int, default=DEFAULT_XGBOOST_PARAMS['max_depth'],
                        help='Maximum depth of trees')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_XGBOOST_PARAMS['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--n-estimators', type=int, default=DEFAULT_XGBOOST_PARAMS['n_estimators'],
                        help='Number of boosting rounds')
    parser.add_argument('--subsample', type=float, default=DEFAULT_XGBOOST_PARAMS['subsample'],
                        help='Fraction of samples used for fitting trees')
    parser.add_argument('--colsample-bytree', type=float, default=DEFAULT_XGBOOST_PARAMS['colsample_bytree'],
                        help='Fraction of features used when constructing each tree')
    parser.add_argument('--reg-lambda', type=float, default=DEFAULT_XGBOOST_PARAMS['reg_lambda'],
                        help='L2 regularization term on weights')
    parser.add_argument('--reg-alpha', type=float, default=DEFAULT_XGBOOST_PARAMS['reg_alpha'],
                        help='L1 regularization term on weights')
    parser.add_argument('--min-child-weight', type=int, default=DEFAULT_XGBOOST_PARAMS['min_child_weight'],
                        help='Minimum sum of instance weight needed in a child')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and save plots')
    parser.add_argument('--plot-training-size-curve', action='store_true',
                        help='Generate training size curve (may take longer)')
    parser.add_argument('--run2-yield', action='store_true',
                        help='Generate Run2 categorization yield tables (requires --EventWeight and --mode-name)')
    parser.add_argument('--class-name-dict', type=str, default='merged',
                        choices=class_name_dict_options.keys(),
                        help='Class name dictionary used for yield table label mapping')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for training')
    parser.add_argument('--GATO_data_path', type=str, default=None,
                        help='Path to save BDT data for GATO (optional)')
    
    args = parser.parse_args()
    selected_class_name_dict = class_name_dict_options[args.class_name_dict]
    
    start_total = time.time()
    
    # Initialize trainer
    trainer = MulticlassBDTTrainer(random_state=args.random_state)
    
    # Load data
    print("\n" + "="*60)
    print("STEP 1: Loading data...")
    print("="*60)
    start_load = time.time()
    x, y, weights, event_weights, modes, feature_names, class_names = load_data(
        data_path=args.data,
        target_column=args.target,
        feature_columns=args.features,
        tree_name=args.tree_name,
        weight_column=args.weights,
        EventWeight_column=args.EventWeight,
        mode_name=args.mode_name
    )
    # Set trainer attributes from loaded data
    trainer.feature_names = feature_names
    trainer.class_names = class_names
    elapsed_load = time.time() - start_load
    print(f"\n⏱️  Data loading completed in {elapsed_load:.2f}s ({elapsed_load/60:.2f}m)")
    
    # Train model
    print("\n" + "="*60)
    print("STEP 2: Training model...")
    print("="*60)
    start_train = time.time()
    results = trainer.train(
        x, y, weights,
        event_weights=event_weights,
        modes=modes,
        test_size=args.test_size,
        hyperparameter_tuning=args.tune_hyperparameters is not None,
        n_trials=args.tune_hyperparameters if args.tune_hyperparameters is not None else 30,
        use_gpu=args.use_gpu,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight
    )
    elapsed_train = time.time() - start_train
    print(f"\n⏱️  Training completed in {elapsed_train:.2f}s ({elapsed_train/60:.2f}m)")
    
    # Save model
    print("\n" + "="*60)
    print("STEP 3: Saving model...")
    print("="*60)
    start_save = time.time()
    save_model(
        results['model'],
        trainer.feature_names,
        trainer.class_names,
        args.output
    )
    elapsed_save = time.time() - start_save
    print(f"⏱️  Model saving completed in {elapsed_save:.2f}s")
    
    # Save BDT data for GATO
    if args.GATO_data_path is not None:
        gato_data_path = Path(args.GATO_data_path)
        gato_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            gato_data_path,
            x_train=results['x_train'],
            y_train=results['y_train'],
            proba_train=results['proba_train'],
            ew_train=results['event_weights_train'],
            x_val=results['x_val'],
            y_val=results['y_val'],
            proba_val=results['proba_val'],
            ew_val=results['event_weights_val'],
            x_test=results['x_test'],
            y_test=results['y_test'],
            proba_test=results['proba_test'],
            ew_test=results['event_weights_test'],
            class_names=trainer.class_names.tolist() if isinstance(trainer.class_names, np.ndarray) else trainer.class_names
        )
        print(f"✓ BDT data for GATO saved to {gato_data_path}")

    # Generate plots
    if args.plot:
        print("\n" + "="*60)
        print("STEP 4: Generating plots...")
        print("="*60)
        start_plot = time.time()
        
        plot_dir = Path(args.output).parent / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        if 'feature_importance' in results:
            plot_feature_importance(
                results['feature_importance'],
                output_path=str(plot_dir / '0feature_importance.pdf')
            )
        
        # Plot learning curves (training vs validation loss)
        if results.get('evals_result') is not None:
            plot_learning_curves(
                results['evals_result'],
                output_path=str(plot_dir / '0learning_curves.pdf')
            )
            print("✓ Learning curves saved")

        # Plot ROC AUC curves (per category)
        model_class_names = trainer.class_names if isinstance(trainer.class_names, list) else trainer.class_names.tolist()
        y_test_ids_for_roc = np.array([model_class_names[i] for i in results['y_test']])
        y_test_proba = results['model'].predict_proba(results['x_test'])
        plot_roc_auc_curve(
            y_test_ids_for_roc,
            y_test_proba,
            model_class_names,
            class_name_dict=selected_class_name_dict,
            output_path=str(plot_dir / '0roc_auc_curve.pdf')
        )
        print("✓ ROC AUC curve saved")
        
        # Plot training size curve (optional, can take longer)
        if args.plot_training_size_curve:
            print("\nGenerating training size curve...")
            plot_training_size_curve(
            trainer,
            x, y,
            weights,
            event_weights,
            modes,
            train_fractions=[0.3, 0.5, 0.7, 0.9, 1.0],
            output_path=str(plot_dir / '0training_size_curve.pdf'),
            test_size=args.test_size,
            use_gpu=args.use_gpu,
            hyperparameter_tuning=False,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            reg_alpha=args.reg_alpha,
            min_child_weight=args.min_child_weight
            )
            print("✓ Training size curve saved")
        else:
            print("Skipping training size curve (use --plot-training-size-curve to enable)")

        # Merge STXS 1.2 categories for plotting using FULL merge helper
        # This merges all categories including GG2H_GE2J_MJJ_GT350 and QQ2HQQ_rest
        # (Note: Training uses PARTIAL merge helper which keeps these categories unmerged)
        # Convert from indices back to category IDs
        class_names = trainer.class_names if isinstance(trainer.class_names, list) else trainer.class_names.tolist()
        y_test_ids = np.array([class_names[i] for i in results['y_test']])
        y_test_pred_ids = np.array([class_names[i] for i in results['y_test_pred']])
        
        merged_class_names = class_names.copy()
        categories_to_remove = set()
        
        print("\nMerging STXS 1.2 categories for plotting (using FULL merge helper)...")
        for merged_category, original_categories in STXS_1_2_MERGE_Helper.items():
            # Merge labels (working with category IDs now)
            mask = np.isin(y_test_ids, original_categories)
            mask_pred = np.isin(y_test_pred_ids, original_categories)
            n_merged = np.sum(mask)
            if n_merged > 0:
                y_test_ids[mask] = merged_category
                y_test_pred_ids[mask_pred] = merged_category
                print(f"  Merged {original_categories} -> {merged_category}: {n_merged} events")
            
            # Track categories to remove from class names
            categories_to_remove.update(original_categories)
        
        # Remove merged categories from class names list
        merged_class_names = [cls for cls in merged_class_names if cls not in categories_to_remove]
        print(f"Merged class names: {len(class_names)} -> {len(merged_class_names)} categories")
        
        # Use the merged IDs for all subsequent plotting
        y_test = y_test_ids
        y_test_pred = y_test_pred_ids

        if args.EventWeight is not None and args.mode_name is not None:
            # Compute BDT yield table
            
            yield_table_by_mode, yield_table_by_true_label = compute_yield_table(
                y_test,
                y_test_pred,
                event_weights=results.get('event_weights_test'),
                modes=results.get('modes_test'),
                class_names=merged_class_names,
                class_name_dict=selected_class_name_dict
            )
            # yield_table_by_mode.to_csv(plot_dir / 'BDT_by_mode.csv', index=True)
            yield_table_by_true_label.to_csv(plot_dir / 'YT_BDT.csv', index=True)
            plot_yield_table_density(
                yield_table_by_true_label,
                title='BDT Yield Table Density (by True Label)',
                output_path=str(plot_dir / 'YTDensity_BDT.pdf')
            )
            print("✓ BDT yield tables and density plot saved")
            
            # Compute Run2 categorization yield table
            if args.run2_yield:
                print("\nComputing Run2 categorization yield table on test dataset...")

                (yield_table_stage1p2_by_mode, yield_table_stage1p2_by_true_label) = yield_table_run2_categorisation(
                    results['x_test'],
                    y_test,
                    trainer.feature_names,
                    merged_class_names,
                    results.get('event_weights_test'),
                    results.get('modes_test'),
                    class_name_dict=selected_class_name_dict
                )
                # yield_table_stage1p2_by_mode.to_csv(plot_dir / 'CB_by_mode.csv', index=True)
                yield_table_stage1p2_by_true_label.to_csv(plot_dir / 'YT_CB.csv', index=True)
                plot_yield_table_density(
                    yield_table_stage1p2_by_true_label,
                    title='Run2 Categorization (Stage 1.2) Yield Table Density (by True Label)',
                    output_path=str(plot_dir / 'YTDensity_CB.pdf')
                )
                print("✓ Run2 categorization yield tables and density plot saved")
            else:
                print("\nSkipping Run2 categorization yield tables (use --run2-yield to enable)")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_test,
            y_test_pred,
            merged_class_names,
            output_path=str(plot_dir / '0confusion_matrix.pdf')
        )
        print("✓ Confusion matrix saved")
        
        elapsed_plot = time.time() - start_plot
        print(f"⏱️  Plotting completed in {elapsed_plot:.2f}s")
    
    elapsed_total = time.time() - start_total
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\n⏱️  TOTAL TIME: {elapsed_total:.2f}s ({elapsed_total/60:.2f}m)")
    print(f"    - Data loading: {elapsed_load:.2f}s ({elapsed_load/elapsed_total*100:.1f}%)")
    print(f"    - Training: {elapsed_train:.2f}s ({elapsed_train/elapsed_total*100:.1f}%)")
    print(f"    - Saving: {elapsed_save:.2f}s ({elapsed_save/elapsed_total*100:.1f}%)")
    if args.plot:
        print(f"    - Plotting: {elapsed_plot:.2f}s ({elapsed_plot/elapsed_total*100:.1f}%)")


if __name__ == '__main__':
    main()
