"""Train a size classification submodel using precomputed features."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "submodels" / "size"
RESULTS_ROOT = PROJECT_ROOT / "results" / "submodels" / "size"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.submodels.size import model as size_model  # noqa: E402


def normalise_qualities(qualities: Sequence[str]) -> List[str]:
    if not qualities:
        return ["BUENO"]
    norm = {q.upper() for q in qualities if q}
    if "ALL" in norm:
        return ["ALL"]
    return sorted(norm)


def quality_slug(qualities: Sequence[str]) -> str:
    if not qualities or qualities == ["ALL"]:
        return "all"
    return "-".join(sorted(q.lower() for q in qualities))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Optional explicit path to dataset .npz file")
    parser.add_argument("--family", type=str, default="anillo", help="Herraje family (use '*' for all)")
    parser.add_argument(
        "--quality",
        action="append",
        dest="qualities",
        default=None,
        help="Quality token(s) included in the dataset (default: BUENO)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT, help="Base directory for trained models")
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=[64, 32],
        help="Hidden layer sizes for the MLP (default: 64 32)",
    )
    return parser.parse_args()


def locate_dataset(dataset_path: Optional[Path], family: str, qualities: Sequence[str]) -> Path:
    if dataset_path:
        return dataset_path
    slug = quality_slug(qualities)
    candidate = DEFAULT_DATA_ROOT / family / slug / "dataset.npz"
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found at {candidate}. Use --dataset to specify a path.")
    return candidate


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Sequence[str]]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = tuple(str(name) for name in data["feature_names"].tolist())
    if "size_labels" in data:
        size_labels = [str(label) for label in data["size_labels"].tolist()]
    else:
        size_labels = sorted({str(label) for label in y.tolist()})
    return X, y, feature_names, size_labels


def train_model(X_train: np.ndarray, y_train: np.ndarray, hidden_layers: Sequence[int], seed: int) -> Tuple[MLPClassifier, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=seed,
    )
    mlp.fit(X_scaled, y_train)
    return mlp, scaler


def evaluate(model: MLPClassifier, scaler: StandardScaler, X: np.ndarray, y: np.ndarray) -> dict:
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    return {
        "accuracy": float(report["accuracy"]),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    args = parse_args()
    qualities = normalise_qualities(args.qualities or ["BUENO"])
    if args.family == "*":
        base_dir = DEFAULT_DATA_ROOT
        families = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])
    else:
        families = [args.family]

    any_trained = False
    for family in families:
        try:
            dataset_path = locate_dataset(args.dataset, family, qualities)
        except FileNotFoundError:
            print(f"[WARN] Dataset missing for family '{family}'")
            continue

        print(f"Loading dataset: {dataset_path}")
        X, y, feature_names, size_labels = load_dataset(dataset_path)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(X, y))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, scaler = train_model(X_train, y_train, args.hidden, args.seed)

        train_metrics = evaluate(model, scaler, X_train, y_train)
        val_metrics = evaluate(model, scaler, X_val, y_val)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = quality_slug(qualities)
        output_dir = args.output / family / slug / f"mlp_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "family": family,
            "qualities": qualities,
            "seed": args.seed,
            "hidden_layers": list(args.hidden),
            "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
            "feature_names": list(feature_names),
            "size_labels": list(size_labels),
        }

        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        bundle_path = output_dir / "model.joblib"
        size_model.save_bundle(
            bundle_path,
            family=family,
            size_labels=size_labels,
            feature_names=feature_names,
            model=model,
            scaler=scaler,
            qualities=qualities,
            seed=args.seed,
            dataset=str(dataset_path.relative_to(PROJECT_ROOT)),
        )

        latest_dir = args.output / family / slug / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(output_dir, latest_dir)

        joblib.dump(
            {
                "family": family,
                "qualities": qualities,
                "sizes": size_labels,
                "metrics": metrics,
                "bundle": str(bundle_path.relative_to(PROJECT_ROOT)),
            },
            output_dir / "summary.joblib",
        )

        print(f"Model saved to {output_dir}")
        print(f"Validation accuracy: {val_metrics['accuracy']:.3f}")
        any_trained = True

    if not any_trained:
        raise SystemExit("No families were trained. Check dataset availability or parameters.")


if __name__ == "__main__":
    main()
