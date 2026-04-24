#!/usr/bin/env python3
"""
Train a multiclass DNN classifier for STXS categorization.

Deep Neural Network implementation using PyTorch.
"""

import os
import sys
from pathlib import Path

# Resolve HZZ_ROOT (three levels up) so the Categorisation package is importable.
_HZZ_ROOT = str(Path(__file__).resolve().parents[3])
if _HZZ_ROOT not in sys.path:
    sys.path.insert(0, _HZZ_ROOT)

import argparse
import time
import numpy as np

from Categorisation.DNN.functions import MulticlassDNNTrainer, load_data, save_model
from Categorisation.DNN.functions.config import (
    STXS_1_2_MERGE_Helper,
    STXS_STAGE_1_2_DICT_MERGED,
    STXS_STAGE_1_2_DICT_PARTIAL_MERGED,
    STXS_STAGE_1_2_DICT,
    STXS_1_2_MERGE_Helper_PARTIAL,
)
from Categorisation.DNN.functions.plotting import (
    plot_confusion_matrix,
    plot_yield_table_density,
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
        description='Train a multiclass DNN classifier',
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
    parser.add_argument('--output', '-o', type=str, default='multiclass_dnn_model',
                        help='Output path for saved model (without extension)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--tune-hyperparameters', type=int, default=None, metavar='N_TRIALS',
                        help='Perform hyperparameter tuning with Optuna using N_TRIALS trials')
    
    # DNN-specific hyperparameters
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--activation', type=str, default='relu', 
                        choices=['relu', 'elu', 'leaky_relu'],
                        help='Activation function')
    
    parser.add_argument('--plot', action='store_true',
                        help='Generate and save plots')
    parser.add_argument('--run2-yield', action='store_true',
                        help='Generate Run2 categorization yield tables (requires --EventWeight and --mode-name)')
    parser.add_argument('--class-name-dict', type=str, default='merged',
                        choices=class_name_dict_options.keys(),
                        help='Class name dictionary used for yield table label mapping')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for training')
    
    args = parser.parse_args()
    selected_class_name_dict = class_name_dict_options[args.class_name_dict]
    
    start_total = time.time()
    
    # Initialize trainer
    trainer = MulticlassDNNTrainer(random_state=args.random_state)
    
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
    
    trainer.feature_names = feature_names
    trainer.class_names = class_names
    
    elapsed_load = time.time() - start_load
    print(f"⏱️  Data loading completed in {elapsed_load:.2f}s")
    
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
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
        activation=args.activation
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
        args.output,
        results.get('scaler')
    )
    elapsed_save = time.time() - start_save
    print(f"⏱️  Model saving completed in {elapsed_save:.2f}s")
    
    # Generate plots
    if args.plot:
        print("\n" + "="*60)
        print("STEP 4: Generating plots...")
        print("="*60)
        start_plot = time.time()
        
        plot_dir = Path(args.output).parent / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert from indices back to category IDs
        class_names = trainer.class_names if isinstance(trainer.class_names, list) else trainer.class_names.tolist()
        y_test_ids = np.array([class_names[i] for i in results['y_test']])
        y_test_pred_ids = np.array([class_names[i] for i in results['y_test_pred']])
        
        merged_class_names = class_names.copy()
        categories_to_remove = set()
        
        
        # Use the merged IDs for all subsequent plotting
        y_test = y_test_ids
        y_test_pred = y_test_pred_ids

        if args.EventWeight is not None and args.mode_name is not None:
            # Compute DNN yield table
            yield_table_by_mode, yield_table_by_true_label = compute_yield_table(
                y_test,
                y_test_pred,
                event_weights=results.get('event_weights_test'),
                modes=results.get('modes_test'),
                class_names=merged_class_names,
                class_name_dict=selected_class_name_dict
            )
            yield_table_by_true_label.to_csv(plot_dir / 'YT_DNN.csv', index=True)
            plot_yield_table_density(
                yield_table_by_true_label,
                title='DNN Yield Table Density (by True Label)',
                output_path=str(plot_dir / 'YTDensity_DNN.pdf')
            )
            print("✓ DNN yield tables and density plot saved")
            
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
