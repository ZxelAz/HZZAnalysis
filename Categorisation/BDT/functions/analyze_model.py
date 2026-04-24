"""Analyze a trained multiclass BDT model and plot validation score histograms."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from .data_loader import load_data
from .config import (STXS_STAGE_1_2_DICT_MERGED, STXS_1_2_MERGE_Helper,
                      STXS_STAGE_1_2_DICT_PARTIAL_MERGED, STXS_1_2_MERGE_Helper_PARTIAL)


def _load_model(model_path: str) -> Tuple[object, Optional[list], Optional[list]]:
	model_path_obj = Path(model_path)
	if not model_path_obj.exists():
		raise FileNotFoundError(f"Model file not found: {model_path_obj}")

	with open(model_path_obj, "rb") as f:
		payload = pickle.load(f)

	if isinstance(payload, dict) and "model" in payload:
		model = payload["model"]
		feature_names = payload.get("feature_names")
		class_names = payload.get("class_names")
		return model, feature_names, class_names

	return payload, None, None


def _select_file_by_suffix(directory: Path, suffix: str) -> Path:
	if not directory.exists():
		raise FileNotFoundError(f"Trial directory not found: {directory}")

	candidates = sorted(directory.glob(f"*{suffix}"))
	if not candidates:
		raise FileNotFoundError(f"No '{suffix}' files found in {directory}")

	preferred = [p for p in candidates if "model" in p.name.lower()]
	return preferred[0] if preferred else candidates[0]


def _load_trial_metadata(trial_path: str) -> Tuple[Path, Path, Optional[list], Optional[list]]:
	trial_dir = Path(trial_path)
	json_path = _select_file_by_suffix(trial_dir, ".json")
	pkl_path = _select_file_by_suffix(trial_dir, ".pkl")

	with open(json_path, "r", encoding="utf-8") as f:
		metadata = json.load(f)

	feature_names = metadata.get("feature_names")
	class_names = metadata.get("class_names")
	return pkl_path, json_path, feature_names, class_names


def _split_validation(
	x: np.ndarray,
	y: np.ndarray,
	event_weights: Optional[np.ndarray],
	test_size: float,
	val_size: float,
	random_state: int
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
	def _can_stratify(labels: np.ndarray) -> bool:
		if labels.size == 0:
			return False
		_, counts = np.unique(labels, return_counts=True)
		return counts.min() >= 2

	x_trainval, x_test, y_trainval, y_test = train_test_split(
		x, y, test_size=test_size, random_state=random_state,
		stratify=y if _can_stratify(y) else None
	)

	val_ratio = val_size / (1 - test_size)
	x_train, x_val, y_train, y_val = train_test_split(
		x_trainval, y_trainval, test_size=val_ratio, random_state=random_state,
		stratify=y_trainval if _can_stratify(y_trainval) else None
	)

	ew_val = None
	if event_weights is not None:
		ew_trainval, ew_test = train_test_split(
			event_weights, test_size=test_size, random_state=random_state,
			stratify=y if _can_stratify(y) else None
		)
		ew_train, ew_val = train_test_split(
			ew_trainval, test_size=val_ratio, random_state=random_state,
			stratify=y_trainval if _can_stratify(y_trainval) else None
		)

	return x_val, y_val, ew_val


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot histogram of BDT scores on the validation set",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	model_group = parser.add_mutually_exclusive_group(required=True)
	model_group.add_argument("--model", help="Path to trained model .pkl")
	model_group.add_argument("--trial-path", help="Path to trial directory containing model.json and model.pkl")
	parser.add_argument("--data", "-d", type=str, nargs='+', required=True,
						help="Path(s) to input ROOT file(s)")
	parser.add_argument("--tree-name", required=True, help="TTree name")
	parser.add_argument("--target", required=True, help="Target/class column name")
	parser.add_argument("--features", nargs='+', default=None,
						help="Feature columns (optional if stored in model)")
	parser.add_argument("--event-weight-col", default=None,
						help="Event weight column (optional)")
	parser.add_argument("--test-size", type=float, default=0.2,
						help="Fraction of data for test split")
	parser.add_argument("--val-size", type=float, default=0.1,
						help="Fraction of data for validation split")
	parser.add_argument("--random-state", type=int, default=42,
						help="Random seed")
	parser.add_argument("--bins", type=int, default=50,
				help="Number of histogram bins")
	parser.add_argument("--range", dest="hist_range", nargs=2, type=float, default=[0.0, 1.0],
						help="Histogram range (min max)")
	parser.add_argument("--use-event-weights", action="store_true",
						help="Weight histogram by event weights")
	parser.add_argument("--class-name-dict", type=str, default='merged',
						choices=['merged', 'partial_merged'],
						help="Class name dictionary for category merging and label mapping")
	parser.add_argument("--output", default=None, help="Output path for plot")

	args = parser.parse_args()

	# Select the appropriate merge helper and dictionary
	if args.class_name_dict == 'partial_merged':
		merge_helper = STXS_1_2_MERGE_Helper_PARTIAL
		stxs_dict = STXS_STAGE_1_2_DICT_PARTIAL_MERGED
		print(f"Using PARTIAL_MERGED dictionary for category merging and labeling")
	else:
		merge_helper = STXS_1_2_MERGE_Helper
		stxs_dict = STXS_STAGE_1_2_DICT_MERGED
		print(f"Using MERGED dictionary for category merging and labeling")

	if args.trial_path:
		model_path, metadata_path, metadata_features, metadata_classes = _load_trial_metadata(args.trial_path)
		model, _, _ = _load_model(str(model_path))
		model_feature_names = metadata_features
		model_class_names = metadata_classes
	else:
		model, model_feature_names, model_class_names = _load_model(args.model)

	feature_names = args.features if args.features is not None else model_feature_names

	if not feature_names:
		raise ValueError("Feature list is required. Provide --features or save model with features.")

	x, y, _, event_weights, _, _, data_class_names = load_data(
		data_path=args.data,
		target_column=args.target,
		feature_columns=feature_names,
		tree_name=args.tree_name,
		EventWeight_column=args.event_weight_col
	)

	# Merge STXS 1.2 categories
	print("\nMerging STXS 1.2 categories...")
	for merged_category, original_categories in merge_helper.items():
		mask = np.isin(y, original_categories)
		n_merged = np.sum(mask)
		if n_merged > 0:
			y[mask] = merged_category
			print(f"  Merged {original_categories} -> {merged_category}: {n_merged} events")

	x_val, y_val, ew_val = _split_validation(
		x, y, event_weights, args.test_size, args.val_size, args.random_state
	)

	proba = model.predict_proba(x_val)
	weights = ew_val if (args.use_event_weights and ew_val is not None) else None
	
	# Plot all classes from model
	if model_class_names is not None:
		class_indices = list(range(len(model_class_names)))
		print(f"Plotting all {len(class_indices)} classes from model")
	else:
		class_indices = list(range(proba.shape[1]))
		print(f"Plotting all {len(class_indices)} classes (no class names available)")

	# Create one figure with all classes
	n_classes = len(class_indices)
	n_cols = 3
	n_rows = (n_classes + n_cols - 1) // n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
	axes = np.atleast_1d(axes).flatten()
	cmap = plt.get_cmap("tab20")

	unique_labels = np.unique(y_val)

	# Plot each class in a subplot
	for plot_idx, class_idx in enumerate(class_indices):
		if class_idx < 0 or class_idx >= proba.shape[1]:
			print(f"⚠ Skipping class index {class_idx}: out of bounds for model with {proba.shape[1]} outputs")
			continue
			
		scores = proba[:, class_idx]
		
		if model_class_names is not None and class_idx < len(model_class_names):
			class_label = model_class_names[class_idx]
			plot_title = f"{class_label}"
		else:
			plot_title = f"Class {class_idx}"

		try:
			class_label_int = int(class_label) if "class_label" in locals() else int(class_idx)
			if class_label_int in stxs_dict:
				plot_title = stxs_dict[class_label_int]
		except Exception:
			pass

		ax = axes[plot_idx]
		
		# Try to determine the true label for this class
		try:
			true_label = int(class_label) if "class_label" in locals() else int(class_idx)
		except Exception:
			true_label = None
		
		# Plot histogram for the signal (true label matches this class)
		if true_label is not None:
			signal_mask = y_val == true_label
			signal_scores = scores[signal_mask]
			signal_weights = weights[signal_mask] if weights is not None else None
			
			if true_label in stxs_dict:
				signal_name = stxs_dict[true_label]
			else:
				signal_name = f"Cat {true_label}"
			
			ax.hist(
				signal_scores,
				bins=args.bins,
				range=tuple(args.hist_range),
				weights=signal_weights,
				density=True,
				histtype="step",
				linewidth=1.5,
				color="blue",
				alpha=1.0,
				label=signal_name if plot_idx == 0 else None
			)
		
		# Plot histogram for the background (all other true labels combined)
		background_mask = y_val != true_label if true_label is not None else np.zeros(len(y_val), dtype=bool)
		background_scores = scores[background_mask]
		background_weights = weights[background_mask] if (weights is not None and background_mask.sum() > 0) else None
		
		if background_mask.sum() > 0:
			ax.hist(
				background_scores,
				bins=args.bins,
				range=tuple(args.hist_range),
				weights=background_weights,
				density=True,
				histtype="step",
				linewidth=1.5,
				color="red",
				alpha=1.0,
				label="Background" if plot_idx == 0 else None
			)
		
		ax.set_xlabel("BDT Score", fontsize=9)
		ax.set_ylabel("Density", fontsize=9)
		ax.set_title(plot_title, fontsize=10, fontweight="bold")
		ax.set_yscale("log")
		ax.grid(True, alpha=0.3)
		ax.tick_params(labelsize=8)

	# Add legend outside the subplots
	if len(unique_labels) > 0:
		handles, labels = axes[0].get_legend_handles_labels()
		if handles:
			fig.legend(handles, ('signal', 'background'), fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))

	# Remove empty subplots
	for idx in range(n_classes, len(axes)):
		fig.delaxes(axes[idx])

	plt.tight_layout(rect=[0, 0, 0.95, 1])

	if args.output:
		output_path = Path(args.output)
		# Save to bdt_score/ subdirectory
		bdt_score_dir = output_path.parent / "bdt_score"
		bdt_score_dir.mkdir(parents=True, exist_ok=True)
		final_output_path = bdt_score_dir / output_path.name
		plt.savefig(final_output_path, dpi=300, bbox_inches="tight")
		print(f"✓ Saved plot: {final_output_path}")
	else:
		plt.show()

	plt.close()


if __name__ == "__main__":
	main()
