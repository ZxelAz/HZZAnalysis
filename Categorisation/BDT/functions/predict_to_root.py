#!/usr/bin/env python3
"""Run multiclass BDT inference on ROOT files and append BDT_category branch."""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import uproot

from .config import STXS_STAGE_1_2_DICT_MERGED


def load_model_payload(model_path: str) -> Tuple[object, Optional[List[str]], Optional[Sequence]]:
    """Load model payload from pickle file."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    with open(model_file, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("feature_names"), payload.get("class_names")

    return payload, None, None


def configure_prediction_device(model, predict_device: str) -> None:
    """Configure XGBoost prediction device.

    Args:
        model: Trained model object (typically xgboost.sklearn wrapper)
        predict_device: One of {'auto', 'cpu', 'cuda'}
    """
    if predict_device == "auto":
        return

    if not hasattr(model, "get_booster"):
        return

    booster = model.get_booster()
    if predict_device == "cpu":
        if hasattr(model, "set_params"):
            model.set_params(device="cpu")
        booster.set_param({"device": "cpu", "predictor": "cpu_predictor"})
    elif predict_device == "cuda":
        if hasattr(model, "set_params"):
            model.set_params(device="cuda")
        booster.set_param({"device": "cuda", "predictor": "gpu_predictor"})


def _build_class_code_array(class_names: Sequence) -> np.ndarray:
    """Convert model class_names to integer STXS codes."""
    if class_names is None:
        raise ValueError("Model payload does not include class_names; cannot map predicted indices to STXS categories.")

    class_name_to_code: Dict[str, int] = {name: code for code, name in STXS_STAGE_1_2_DICT_MERGED.items()}

    stxs_codes = []
    for cls in class_names:
        if isinstance(cls, (np.integer, int)):
            stxs_codes.append(int(cls))
            continue

        cls_str = str(cls)
        try:
            stxs_codes.append(int(cls_str))
            continue
        except ValueError:
            pass

        if cls_str in class_name_to_code:
            stxs_codes.append(class_name_to_code[cls_str])
            continue

        raise ValueError(
            f"Could not map class '{cls}' to numeric STXS code. "
            "Expected integer-like class_names or names from STXS_STAGE_1_2_DICT_MERGED."
        )

    return np.asarray(stxs_codes, dtype=np.int32)


def _decode_if_bytes(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8")
    return arr


def _build_feature_matrix(chunk: Dict[str, np.ndarray], feature_columns: List[str]) -> np.ndarray:
    feature_arrays = []
    for col in feature_columns:
        if col not in chunk:
            raise KeyError(f"Feature column '{col}' is missing in ROOT chunk.")

        col_data = _decode_if_bytes(np.asarray(chunk[col]))
        feature_arrays.append(col_data)

    x_part = np.column_stack(feature_arrays)
    return x_part.astype(float)


def _make_gpu_array(x_cpu: np.ndarray):
    """Return a CuPy array if available, otherwise the original numpy array."""
    try:
        import cupy as cp
        return cp.asarray(x_cpu)
    except Exception:
        return x_cpu


def _to_numpy(arr) -> np.ndarray:
    """Convert CuPy or numpy array to numpy."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return np.asarray(arr)


def predict_root_file(
    input_file: str,
    output_file: str,
    tree_name: str,
    model,
    feature_columns: List[str],
    class_codes: np.ndarray,
    chunk_size: str,
    predict_device: str = "cuda",
) -> None:
    """Predict BDT category and write new ROOT with appended BDT_category branch."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nInput ROOT:  {input_path}")
    print(f"Output ROOT: {output_path}")
    print(f"Tree:        {tree_name}")
    print(f"Chunk size:  {chunk_size}")

    total_entries = 0
    with uproot.open(input_path) as src:
        tree = src[tree_name]
        n_events = tree.num_entries
        print(f"Events:      {n_events}")

        with uproot.recreate(output_path) as dst:
            first_chunk = True

            for idx, chunk in enumerate(tree.iterate(step_size=chunk_size, library="np", how=dict), 1):
                x_part = _build_feature_matrix(chunk, feature_columns)

                # Move input to GPU for inplace prediction (avoids costly DMatrix copy)
                if predict_device == "cuda":
                    x_input = _make_gpu_array(x_part)
                else:
                    x_input = x_part

                y_proba = _to_numpy(model.predict_proba(x_input))
                y_pred_idx = y_proba.argmax(axis=1)
                y_pred_stxs = class_codes[y_pred_idx].astype(np.int32)

                chunk["BDT_category"] = y_pred_stxs

                # Store per-class BDT probability scores
                for ci, code in enumerate(class_codes):
                    chunk[f"BDT_proba_{code}"] = y_proba[:, ci].astype(np.float32)

                if first_chunk:
                    dst[tree_name] = chunk
                    first_chunk = False
                else:
                    dst[tree_name].extend(chunk)

                total_entries += len(y_pred_stxs)
                print(f"  Chunk {idx}: wrote {len(y_pred_stxs)} events (running total: {total_entries})")

    print(f"✓ Finished writing {total_entries} events to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained BDT model, predict STXS category per event, and save ROOT with BDT_category branch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    parser.add_argument("--input", required=True, nargs="+", help="Input ROOT file(s)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tree-name", default="Events", help="TTree name inside input ROOT")
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Feature columns to use; if omitted, feature_names stored in model are used",
    )
    parser.add_argument(
        "--output-suffix",
        default="_with_bdt",
        help="Suffix added to output ROOT basename",
    )
    parser.add_argument(
        "--chunk-size",
        default="50 MB",
        help="Chunk size for uproot iteration (e.g. '50 MB', '50000 entries')",
    )
    parser.add_argument(
        "--predict-device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Device to run XGBoost inference on",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output ROOT filename (useful for single input file)",
    )
        
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, model_feature_names, model_class_names = load_model_payload(args.model)
    configure_prediction_device(model, args.predict_device)

    feature_columns = args.features if args.features is not None else model_feature_names
    if not feature_columns:
        raise ValueError("Feature list is required. Provide --features or save model with feature_names.")

    class_codes = _build_class_code_array(model_class_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BDT ROOT Inference")
    print("=" * 80)
    print(f"Model:          {args.model}")
    print(f"Input files:    {len(args.input)}")
    print(f"Feature count:  {len(feature_columns)}")
    print(f"Output dir:     {output_dir}")
    print(f"Tree name:      {args.tree_name}")
    print(f"Predict device: {args.predict_device}")
    print("=" * 80)

    for input_file in args.input:
        in_path = Path(input_file)
        if not in_path.exists():
            raise FileNotFoundError(f"Input ROOT file not found: {in_path}")

        output_name = args.output_name
        if output_name is None:
            suffix = in_path.suffix if in_path.suffix else ".root"
            output_name = f"{in_path.stem}{args.output_suffix}{suffix}"

        if not output_name:
            output_name = in_path.name if in_path.name else "output_with_bdt.root"

        output_path = output_dir / output_name

        predict_root_file(
            input_file=str(in_path),
            output_file=str(output_path),
            tree_name=args.tree_name,
            model=model,
            feature_columns=list(feature_columns),
            class_codes=class_codes,
            chunk_size=args.chunk_size,
            predict_device=args.predict_device,
        )


if __name__ == "__main__":
    main()
