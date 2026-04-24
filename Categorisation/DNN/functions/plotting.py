"""Plotting functions for multiclass BDT results."""

from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .config import STXS_STAGE_1_2_DICT_MERGED, STXS_TO_RUN2_MAPPING, STXS_STAGE_1_2_DICT_PARTIAL_MERGED
from .utils import consolidate_mode, first_step_categorisation, second_step_categorisation


def combine_rows(yield_table: pd.DataFrame, rows_to_combine: list, new_row_name: str) -> pd.DataFrame:
    """
    Combine multiple rows by summing them into a single row.
    
    Args:
        yield_table: Yield table DataFrame
        rows_to_combine: List of row names to combine
        new_row_name: Name for the combined row
        
    Returns:
        Modified yield table with combined row
    """
    # Check which rows exist in the table
    existing_rows = [row for row in rows_to_combine if row in yield_table.index]
    
    if len(existing_rows) == 0:
        # No rows to combine, return table unchanged
        return yield_table
    
    # Sum the rows that exist
    combined_row = yield_table.loc[existing_rows].sum(axis=0)
    
    # Drop the original rows
    yield_table = yield_table.drop(existing_rows, errors='ignore')
    
    # Add the combined row
    combined_df = pd.DataFrame([combined_row], index=[new_row_name])
    yield_table = pd.concat([yield_table, combined_df])
    
    return yield_table


def combine_columns(yield_table: pd.DataFrame, columns_to_combine: list, new_column_name: str) -> pd.DataFrame:
    """
    Combine multiple columns by summing them into a single column.
    
    Args:
        yield_table: Yield table DataFrame
        columns_to_combine: List of column names to combine
        new_column_name: Name for the combined column
        
    Returns:
        Modified yield table with combined column
    """
    # Check which columns exist in the table
    existing_columns = [col for col in columns_to_combine if col in yield_table.columns]
    
    if len(existing_columns) == 0:
        # No columns to combine, return table unchanged
        return yield_table
    
    # Sum the columns that exist
    combined_column = yield_table[existing_columns].sum(axis=1)
    
    # Drop the original columns
    yield_table = yield_table.drop(columns=existing_columns, errors='ignore')
    
    # Add the combined column
    yield_table[new_column_name] = combined_column
    
    return yield_table


def calculate_purity(yield_table: pd.DataFrame) -> pd.Series:
    """
    Calculate purity percentage for each predicted category.
    
    Purity is defined as the fraction of correctly classified events:
    purity = (diagonal element) / (sum of all elements in the row)
    
    For example, for predicted category GG2H:
    purity = GG2H / sum(all columns in GG2H row)
    
    Args:
        yield_table: Yield table DataFrame with predicted categories as index
        
    Returns:
        Series with purity percentage for each predicted category
    """
    purity = pd.Series(dtype=float, index=yield_table.index)
    
    for pred_category in yield_table.index:
        row_sum = yield_table.loc[pred_category].sum()
        
        if row_sum > 0:
            # Check if the predicted category exists as a column (diagonal element)
            if pred_category in yield_table.columns:
                diagonal_value = yield_table.loc[pred_category, pred_category]
                purity[pred_category] = (diagonal_value / row_sum) * 100
            else:
                # If no matching column, purity is 0
                purity[pred_category] = 0.0
        else:
            purity[pred_category] = 0.0
    
    return purity


def calculate_success_rate(yield_table: pd.DataFrame) -> pd.Series:
    """
    Calculate success rate percentage for each true category (column).
    
    Success rate is defined as the fraction of correctly classified events:
    success_rate = (diagonal element) / (sum of all elements in the column)
    
    For example, for true category GG2H:
    success_rate = GG2H / sum(all rows in GG2H column)
    
    Args:
        yield_table: Yield table DataFrame with predicted categories as index
                     and true categories as columns
        
    Returns:
        Series with success rate percentage for each true category
    """
    success_rate = pd.Series(dtype=float, index=yield_table.columns)
    
    for true_category in yield_table.columns:
        col_sum = yield_table[true_category].sum()
        
        if col_sum > 0:
            # Check if the true category exists as a row (diagonal element)
            if true_category in yield_table.index:
                diagonal_value = yield_table.loc[true_category, true_category]
                success_rate[true_category] = (diagonal_value / col_sum) * 100
            else:
                # If no matching row, success rate is 0
                success_rate[true_category] = 0.0
        else:
            success_rate[true_category] = 0.0
    
    return success_rate



def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: Optional[str] = None,
    figsize: tuple = (16, 14),
    annot_threshold: float = 0.05
):
    """
    Plot confusion matrix with improved readability.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save plot (if None, displays plot)
        figsize: Figure size (width, height) in inches
        annot_threshold: Only show annotations for values > threshold (default 0.05 = 5%)
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Create labels with shorter names
    labels = [STXS_STAGE_1_2_DICT_MERGED[c] for c in class_names]
    
    # Create masked array to hide small values (improves readability)
    cm_display = cm.copy()
    cm_display[cm_display < annot_threshold] = 0  # Suppress small values
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=cm_display,  # Show numbers only for values above threshold
        fmt='.3f',
        cmap='Blues',
        cbar_kws={'label': 'Normalized Frequency'},
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={'size': 8},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix (normalized by true label)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_auc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list,
    class_name_dict: dict = STXS_STAGE_1_2_DICT_MERGED,
    output_path: Optional[str] = None,
    curves_per_subplot: int = 3
):
    """
    Plot ROC AUC curves for multiclass BDT outputs, with multiple curves per subplot.

    Args:
        y_true: True class labels (class IDs)
        y_score: Predicted probabilities with shape (n_samples, n_classes)
        class_names: List of class IDs corresponding to columns in y_score
        class_name_dict: Mapping from class ID to class name
        output_path: Path to save plot (if None, displays plot)
        curves_per_subplot: Number of ROC curves to plot per subplot (default: 3)
    """
    if y_score.ndim != 2 or y_score.shape[1] != len(class_names):
        raise ValueError("y_score must be 2D with columns matching class_names")

    n_classes = len(class_names)
    
    # Handle binary case specially
    if n_classes == 2:
        plt.figure(figsize=(10, 8))
        pos_class = class_names[1]
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1], pos_label=pos_class)
        roc_auc_value = auc(fpr, tpr)
        label = class_name_dict.get(pos_class, str(pos_class))
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc_value:.3f})")
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC AUC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right', frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"ROC AUC curve saved to {output_path}")
        else:
            plt.show()
        plt.close()
        return
    
    # Multiclass case: plot multiple curves per subplot
    y_true_bin = label_binarize(y_true, classes=class_names)
    
    # Calculate ROC curves for all valid classes
    roc_data = []  # List of (class_idx, class_id, label, fpr, tpr, auc_value)
    
    for idx, class_id in enumerate(class_names):
        # Skip classes without both positive and negative samples
        if np.unique(y_true_bin[:, idx]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
        roc_auc_value = auc(fpr, tpr)
        label = class_name_dict.get(class_id, str(class_id))
        roc_data.append((idx, class_id, label, fpr, tpr, roc_auc_value))
    
    n_valid_classes = len(roc_data)
    n_subplots = (n_valid_classes + curves_per_subplot - 1) // curves_per_subplot  # Ceiling division
    
    # Create subplot grid
    n_cols = 3
    n_rows = (n_subplots + n_cols - 1) // n_cols  # Ceiling division
    
    figsize = (5 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    # Plot ROC curves
    for subplot_idx in range(n_subplots):
        ax = axes[subplot_idx]
        
        # Determine which curves to plot in this subplot
        start_idx = subplot_idx * curves_per_subplot
        end_idx = min(start_idx + curves_per_subplot, n_valid_classes)
        
        # Plot curves for this subplot
        for curve_idx in range(start_idx, end_idx):
            idx, class_id, label, fpr, tpr, roc_auc_value = roc_data[curve_idx]
            ax.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC = {roc_auc_value:.3f})")
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
        ax.set_title(f'ROC Curves {start_idx+1}-{end_idx}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right', frameon=True)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_subplots, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle('ROC AUC Curves for All Classes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC AUC curves saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    output_path: Optional[str] = None
):
    """
    Plot feature importance.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to plot
        output_path: Path to save plot (if None, displays plot)
    """
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_yield_table_density(
    yield_table: pd.DataFrame,
    title: str = 'Yield Table Density',
    output_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    cmap: str = 'YlOrRd'
):
    """
    Plot density plot (heatmap) of yield table by true label.
    
    Args:
        yield_table: Yield table DataFrame (predicted categories as rows, true categories as columns)
        title: Title for the plot
        output_path: Path to save plot (if None, displays plot)
        figsize: Figure size (width, height) in inches
        cmap: Colormap for the heatmap
    """
    # Remove the purity and success rate rows/columns if they exist
    table_for_plot = yield_table.drop('Purity (%)', axis=1, errors='ignore').drop('Success (%)', axis=0, errors='ignore')
    
    # Normalize by column (true label) to show efficiency/success rate
    table_normalized = table_for_plot.div(table_for_plot.sum(axis=0), axis=1) * 100
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        table_normalized,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        cbar_kws={'label': 'Efficiency (%)'},
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'size': 8}
    )
    plt.xlabel('True Category', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Category', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Density plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_learning_curves(
    evals_result: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Plot training vs validation loss curves.
    
    Args:
        evals_result: Dictionary from model.evals_result() containing training history
        output_path: Path to save plot (if None, displays plot)
        figsize: Figure size (width, height) in inches
    """
    if evals_result is None or len(evals_result) == 0:
        print("No evaluation results available for plotting")
        return
    
    plt.figure(figsize=figsize)
    
    # Get metric name (usually 'mlogloss' for multiclass)
    metric_names = list(list(evals_result.values())[0].keys())
    metric = metric_names[0]
    
    # Plot training and validation curves
    epochs = range(len(evals_result['validation_0'][metric]))
    plt.plot(epochs, evals_result['validation_0'][metric], label='Training Loss', linewidth=2)
    plt.plot(epochs, evals_result['validation_1'][metric], label='Validation Loss', linewidth=2)
    
    plt.xlabel('Boosting Rounds', fontsize=12, fontweight='bold')
    plt.ylabel('Log Loss', fontsize=12, fontweight='bold')
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_training_size_curve(
    trainer,
    x_full: np.ndarray,
    y_full: np.ndarray,
    weights_full: Optional[np.ndarray],
    event_weights_full: Optional[np.ndarray],
    modes_full: Optional[np.ndarray],
    train_fractions: list = [0.2, 0.4, 0.6, 0.8, 1.0],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    **train_params
):
    """
    Plot validation performance vs training set size.
    
    Args:
        trainer: MulticlassBDTTrainer instance
        x_full: Full feature dataset
        y_full: Full labels
        weights_full: Full weights
        event_weights_full: Full event weights
        modes_full: Full modes
        train_fractions: List of training data fractions to evaluate
        output_path: Path to save plot
        figsize: Figure size
        **train_params: Additional parameters to pass to trainer.train()
    """
    from sklearn.metrics import log_loss
    
    train_scores = []
    val_scores = []
    train_sizes = []
    
    print("\nComputing training size curve...")
    
    for fraction in train_fractions:
        print(f"  Training with {fraction*100:.0f}% of data...")
        
        # Sample the data
        n_samples = int(len(x_full) * fraction)
        indices = np.random.choice(len(x_full), n_samples, replace=False)
        
        x_sample = x_full[indices]
        y_sample = y_full[indices]
        w_sample = weights_full[indices] if weights_full is not None else None
        ew_sample = event_weights_full[indices] if event_weights_full is not None else None
        m_sample = modes_full[indices] if modes_full is not None else None
        
        # Train model
        results = trainer.train(
            x_sample, y_sample,
            weights=w_sample,
            event_weights=ew_sample,
            modes=m_sample,
            **train_params
        )
        
        # Evaluate on train and validation sets
        y_train_pred_proba = results['model'].predict_proba(results['x_train'])
        y_val_pred_proba = results['model'].predict_proba(results['x_val'])
        
        train_loss = log_loss(results['y_train'], y_train_pred_proba)
        val_loss = log_loss(results['y_val'], y_val_pred_proba)
        
        train_scores.append(train_loss)
        val_scores.append(val_loss)
        train_sizes.append(len(results['x_train']))
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_scores, 'o-', label='Training Loss', linewidth=2, markersize=8)
    plt.plot(train_sizes, val_scores, 's-', label='Validation Loss', linewidth=2, markersize=8)
    
    plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
    plt.ylabel('Log Loss', fontsize=12, fontweight='bold')
    plt.title('Learning Curve: Performance vs Training Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training size curve saved to {output_path}")
    else:
        plt.show()
    plt.close()


def compute_yield_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_weights: np.ndarray,
    modes: np.ndarray,
    class_names: list,
    class_name_dict: dict = STXS_STAGE_1_2_DICT_MERGED
):
    """
    Compute yield tables per category.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        event_weights: Event weights
        modes: Mode/category labels
        class_names: List of class names
        class_name_dict: Mapping from class ID to class name
        
    Returns:
        Tuple of (yield_table_by_mode, yield_table_by_true_label)
    """
    print("\nComputing yield tables...")
    
    modes_consolidated = np.array([consolidate_mode(m) for m in modes])
    
    # y_pred and y_true contain category IDs, not indices into class_names
    pred_names = [class_name_dict.get(p, f'Class_{p}') for p in y_pred]
    true_names = [class_name_dict.get(t, f'Class_{t}') for t in y_true]
    class_labels = [class_name_dict.get(c, f'Class_{c}') for c in class_names]
    
    df = pd.DataFrame({
        'pred_name': pred_names,
        'mode': modes_consolidated,
        'true_label': true_names,
        'event_weight': event_weights
    })
    
    # Yield table by mode
    yield_table_by_mode = df.pivot_table(
        index='pred_name',
        columns='mode',
        values='event_weight',
        aggfunc='sum',
        fill_value=0
    )
    yield_table_by_mode.index.name = 'Predicted_Category'
    
    # Yield table by true label
    yield_table_by_true_label = df.pivot_table(
        index='pred_name',
        columns='true_label',
        values='event_weight',
        aggfunc='sum',
        fill_value=0
    )
    yield_table_by_true_label.index.name = 'Predicted_Category'

    # Reindex to a square table using the expected class labels
    if len(class_labels) > 0:
        yield_table_by_true_label = yield_table_by_true_label.reindex(
            index=class_labels,
            columns=class_labels,
            fill_value=0
        )

    # Reorder rows and columns to match STXS_TO_RUN2_MAPPING order
    stxs_order = list(STXS_TO_RUN2_MAPPING.keys())
    row_order = [row for row in stxs_order if row in yield_table_by_true_label.index]
    col_order = [col for col in stxs_order if col in yield_table_by_true_label.columns]
    # Add any remaining columns/rows (Success (%) and Purity (%))
    col_order += [col for col in yield_table_by_true_label.columns if col not in col_order]
    row_order += [row for row in yield_table_by_true_label.index if row not in row_order]
    yield_table_by_true_label = yield_table_by_true_label.loc[row_order, col_order]
    
    # Add success rate row for both tables (diagonal/column sum)
    success_rates_by_mode = calculate_success_rate(yield_table_by_mode)
    yield_table_by_mode.loc['Success (%)'] = success_rates_by_mode
    
    # Add purity column and success rate row to yield table by true label (after reordering)
    purity = calculate_purity(yield_table_by_true_label)
    yield_table_by_true_label['Purity (%)'] = purity
    
    success_rates_by_true = calculate_success_rate(yield_table_by_true_label.drop(columns=['Purity (%)']))
    # Add NaN for the Purity column
    success_rates_by_true['Purity (%)'] = np.nan
    yield_table_by_true_label.loc['Success (%)'] = success_rates_by_true
    
    print("Yield table by mode:")
    print(yield_table_by_mode)
    print("\nYield table by true label:")
    print(yield_table_by_true_label)
    
    return yield_table_by_mode, yield_table_by_true_label

def yield_table_run2_categorisation(
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    class_names: list,
    event_weights: np.ndarray,
    modes : np.ndarray,
    class_name_dict: dict = STXS_STAGE_1_2_DICT_MERGED
):
    """
    Compute yield table for Run 2 categorisation (Stage 1.2 only).
    
    Args:
        x_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        class_names: List of class names
        event_weights: Event weights
        modes: Mode/category labels
        class_name_dict: Mapping from class ID to class name
    Returns:
        Tuple of (yield_table_stage1p2_by_mode, yield_table_stage1p2_by_true_label)
    """
    modes_consolidated = np.array([consolidate_mode(m) for m in modes])
    # Ensure all feature arrays are proper 1D numpy arrays
    arr = {feature_names[i]: np.asarray(x_test[:, i], dtype=np.float64) for i in range(len(feature_names))}
    arr['event_weight'] = np.asarray(event_weights, dtype=np.float64)
    # y_test contains category IDs after merging, not indices
    arr['true_label'] = np.array([class_name_dict.get(int(y), f'Class_{int(y)}') for y in y_test])
    arr['mode'] = modes_consolidated
    
    first_step_arr_dict = first_step_categorisation(arr)
    second_step_arr_dict = second_step_categorisation(first_step_arr_dict)

    # Debug: Print categories from Run2 categorisation
    print("\nCategories from Run2 categorisation (stage 1.2):")
    for category, cat_arr in second_step_arr_dict.items():
        n_events = len(cat_arr[feature_names[0]])
        print(f"  {category}: {n_events} events")
    
    # Reconstruct dataframe from stage 1.2 categorized arrays
    stage1p2_data = []
    for category, cat_arr in second_step_arr_dict.items():
        n_events = len(cat_arr[feature_names[0]])
        for i in range(n_events):
            stage1p2_data.append({
                'pred_name': category,
                'true_label': cat_arr['true_label'][i],
                'event_weight': cat_arr['event_weight'][i],
                'mode': cat_arr['mode'][i]
            })
    
    df_stage1p2 = pd.DataFrame(stage1p2_data)

    # Yield tables by mode for stage 1.2
    yield_table_stage1p2_by_mode = df_stage1p2.pivot_table(
        index='pred_name',
        columns='mode',
        values='event_weight',
        aggfunc='sum',
        fill_value=0
    )
    yield_table_stage1p2_by_mode.index.name = 'Predicted_Category_Stage1p2'

    # Yield table by true label for stage 1.2
    yield_table_stage1p2_by_true_label = df_stage1p2.pivot_table(
        index='pred_name',
        columns='true_label',
        values='event_weight',
        aggfunc='sum',
        fill_value=0
    )
    yield_table_stage1p2_by_true_label.index.name = 'Predicted_Category_Stage1p2'

    # Combine rows for stage 1.2 
    yield_table_stage1p2_by_true_label = combine_rows(yield_table_stage1p2_by_true_label, ['VBF_1jet_tagged','VBF_rest', 'VH_hadronic_tagged_rest'], 'QQ2HQQ_rest')
    yield_table_stage1p2_by_true_label = combine_rows(yield_table_stage1p2_by_true_label, ['ttH_hadronic_tagged', 'ttH_leptonic_tagged'], 'TTH')
    
    # Rename rows for stage 1.2 by true label using STXS_TO_RUN2_MAPPING
    # Create reverse mapping: Run2 name -> STXS name, excluding rows that will be combined
    rows_to_exclude = ['VBF_1jet_tagged', 'VBF_rest', 'VH_hadronic_tagged_rest', 'ttH_hadronic_tagged', 'ttH_leptonic_tagged']
    run2_to_stxs_mapping = {}
    for stxs_name, run2_names in STXS_TO_RUN2_MAPPING.items():
        # Skip STXS names that will be combined away
        if stxs_name not in rows_to_exclude:
            for run2_name in run2_names:
                run2_to_stxs_mapping[run2_name] = stxs_name
    
    yield_table_stage1p2_by_true_label = yield_table_stage1p2_by_true_label.rename(index=run2_to_stxs_mapping)
    
    # Add missing rows with 0 events (categories that should exist but have no events)
    all_expected_rows = [key for key in STXS_TO_RUN2_MAPPING.keys() if key not in rows_to_exclude]
    for expected_row in all_expected_rows:
        if expected_row not in yield_table_stage1p2_by_true_label.index:
            # Add row with zeros for all columns
            yield_table_stage1p2_by_true_label.loc[expected_row] = 0.0
            print(f"  Added missing row '{expected_row}' with 0 events")
    
    # Reorder rows and columns to match STXS_TO_RUN2_MAPPING order for stage 1.2
    stxs_order_stage1p2 = list(STXS_TO_RUN2_MAPPING.keys())
    row_order_stage1p2 = [row for row in stxs_order_stage1p2 if row in yield_table_stage1p2_by_true_label.index]
    col_order_stage1p2 = [col for col in stxs_order_stage1p2 if col in yield_table_stage1p2_by_true_label.columns]
    # Add any remaining columns/rows (Success (%) and Purity (%))
    col_order_stage1p2 += [col for col in yield_table_stage1p2_by_true_label.columns if col not in col_order_stage1p2]
    row_order_stage1p2 += [row for row in yield_table_stage1p2_by_true_label.index if row not in row_order_stage1p2]
    yield_table_stage1p2_by_true_label = yield_table_stage1p2_by_true_label.loc[row_order_stage1p2, col_order_stage1p2]
    
    # Add success rate row to stage 1.2 by mode
    success_rates_stage1p2_by_mode = calculate_success_rate(yield_table_stage1p2_by_mode)
    yield_table_stage1p2_by_mode.loc['Success (%)'] = success_rates_stage1p2_by_mode
    
    # Add purity column and success rate row to yield table by true label (after reordering)
    # Stage 1.2 by true label
    purity_stage1p2 = calculate_purity(yield_table_stage1p2_by_true_label)
    yield_table_stage1p2_by_true_label['Purity (%)'] = purity_stage1p2
    
    success_rates_stage1p2_by_true = calculate_success_rate(yield_table_stage1p2_by_true_label.drop(columns=['Purity (%)']))
    success_rates_stage1p2_by_true['Purity (%)'] = np.nan
    yield_table_stage1p2_by_true_label.loc['Success (%)'] = success_rates_stage1p2_by_true
    
    print("Yield tables for Run 2 categorisation computed:")
    print("\nStage 1.2 by mode:")
    print(yield_table_stage1p2_by_mode)
    print("\nStage 1.2 by true label:")
    print(yield_table_stage1p2_by_true_label)
    
    return yield_table_stage1p2_by_mode, yield_table_stage1p2_by_true_label

    