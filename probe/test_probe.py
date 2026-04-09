import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple, List, Dict

# ensure the CorrectProbe definition can be found in train_qa_correct_probe, avoid pickle deserialization failure
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from train_probe import CorrectProbe  # noqa: F401  # only used for pickle deserialization

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def load_probe(model_path: str):
    """load the trained probe (contains scaler and logistic regression model)"""
    with open(model_path, "rb") as f:
        probe = pickle.load(f)
    if not getattr(probe, "is_fitted", False):
        raise ValueError("The loaded probe is not fitted, check if the model file is correct")
    return probe


def load_test_data(data_path: str, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    read the test data pickle, extract hidden states and labels.
    the data structure of the qa version: each sample's 'question_hs' is
    [num_layers, hidden_size] (the hidden state of the last token of each layer)
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    hidden_states: List[np.ndarray] = []
    labels: List[int] = []
    for sample in data:
        # hs: shape [num_layers, hidden_size]
        hs = np.asarray(sample["question_hs"])
        hidden_states.append(hs)
        labels.append(int(sample["correct"]))

    # X: [num_samples, num_layers, hidden_size]
    X = np.array(hidden_states)
    y = np.array(labels)
    return X, y


def evaluate_probe(probe, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
    """
    infer the test set, return the prediction, probability, distance, correctness information and overall metrics.
    the distance uses the decision_function of logistic, representing the signed distance from the sample to the hyperplane.
    """
    X_scaled = probe.scaler.transform(X)
    probs = probe.model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    distances = probe.model.decision_function(X_scaled)
    correct_flags = preds == y

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc": float(roc_auc_score(y, probs)),
    }

    per_sample = []
    for idx, (label, pred, prob, dist, corr) in enumerate(
        zip(y, preds, probs, distances, correct_flags)
    ):
        per_sample.append(
            {
                "index": idx,
                "label": int(label),
                "pred": int(pred),
                "prob": float(prob),
                "distance": float(dist),
                "correct": bool(corr),
            }
        )

    return {"metrics": metrics, "per_sample": per_sample}


def save_results(results: Dict[str, any], output_path: str):
    """save the test results to JSON, including the overall metrics and per-sample information"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Use the trained QA probe, evaluate all layers")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The directory of the probe models of each layer (the location where train_qa_correct_probe.py is saved)",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        required=True,
        help="The model file prefix, without the _layerXX_correct.pkl part",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The pickle path of the test data (contains question_hs and correct)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory of the test results of each layer",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        required=True,
        help="Optional, limit the maximum number of samples to test (for quick verification)",
    )
    args = parser.parse_args()

    print("=== Loading test data ===")
    X, y = load_test_data(args.data_path, max_samples=args.max_samples)
    print(f"Test sample number: {len(X)}")

    if X.ndim != 3:
        raise ValueError(f"Expected X to be a 3D array [num_samples, num_layers, hidden_size], but got shape {X.shape}")

    num_samples, num_layers, hidden_size = X.shape
    print(f"Data shape: num_samples={num_samples}, num_layers={num_layers}, hidden_size={hidden_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    for layer_idx in range(num_layers):
        print(f"\n=== Evaluating the probe of layer {layer_idx} ===")
        model_path = os.path.join(
            args.model_dir,
            f"{args.model_prefix}_layer{layer_idx}_correct.pkl",
        )
        if not os.path.exists(model_path):
            print(f"Model does not exist, skip: {model_path}")
            continue

        probe = load_probe(model_path)
        print(f"Model loaded: {model_path}")

        # get the features of the layer: [num_samples, hidden_size]
        X_layer = X[:, layer_idx, :]

        results = evaluate_probe(probe, X_layer, y)
        metrics = results["metrics"]
        print("Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        output_path = os.path.join(
            args.output_dir,
            f"{args.model_prefix}_layer{layer_idx}_correct_test_results.json",
        )
        print(f"Saving results to: {output_path}")
        save_results(results, output_path)

    print("\n=== All available layers probe testing completed ===")


if __name__ == "__main__":
    main()