"""Train an MLP classifier on herraje mask features."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = PROJECT_ROOT / "data" / "processed" / "manifest.json"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import FeatureVector, extract_features  # noqa: E402


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list of records")
    return data


def build_dataset(records: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    names: Tuple[str, ...] | None = None

    for record in tqdm(records, desc="Extracting features"):
        mask_rel = record.get("mask_path")
        label = record.get("label")
        if mask_rel is None or label is None:
            continue
        mask_path = PROJECT_ROOT / "data" / "processed" / mask_rel
        if not mask_path.exists():
            continue
        fv: FeatureVector = extract_features(mask_path)
        if names is None:
            names = fv.names
        features.append(fv.values)
        labels.append(str(label))

    if not features:
        raise RuntimeError("No features extracted from manifest")

    X = np.vstack(features)
    y = np.array(labels)
    if names is None:
        names = tuple(f"f{i}" for i in range(X.shape[1]))
    return X, y, names


def stratified_split(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y))
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def train_model(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> Tuple[MLPClassifier, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=seed,
    )
    mlp.fit(X_train_scaled, y_train)
    return mlp, scaler


def evaluate_model(model: MLPClassifier, scaler: StandardScaler, X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    return {
        "accuracy": float(report["accuracy"]),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_results(result_dir: Path, model: MLPClassifier, scaler: StandardScaler, feature_names: Tuple[str, ...], metrics: Dict[str, object]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": feature_names}, result_dir / "model.joblib")
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    latest_dir = RESULTS_DIR / "latest_model"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(result_dir, latest_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT, help="Path to manifest.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    X, y, feature_names = build_dataset(records)
    X_train, X_val, y_train, y_val = stratified_split(X, y, seed=args.seed)

    model, scaler = train_model(X_train, y_train, seed=args.seed)
    train_metrics = evaluate_model(model, scaler, X_train, y_train)
    val_metrics = evaluate_model(model, scaler, X_val, y_val)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f"mlp_{timestamp}"
    save_results(
        result_dir,
        model,
        scaler,
        feature_names,
        {
            "train": train_metrics,
            "validation": val_metrics,
            "labels": sorted(set(y.tolist())),
            "feature_names": list(feature_names),
        },
    )

    print(f"Model saved to {result_dir}")
    print(f"Validation accuracy: {val_metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
