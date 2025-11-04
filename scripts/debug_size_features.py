"""Inspect size features for a specific family to diagnose class separation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "submodels" / "size" / "anillo" / "bueno" / "dataset.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to size dataset .npz file")
    parser.add_argument("--labels", nargs="*", default=["T1", "T2"], help="Size labels to compare")
    parser.add_argument(
        "--features",
        nargs="*",
        default=["outer_area_px", "ring_area_px", "thickness_mean_px", "hole_equiv_diameter_mean_px"],
        help="Feature names to summarise",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(str)
    feature_names = [str(name) for name in data["feature_names"].tolist()]

    indices = [feature_names.index(name) for name in args.features if name in feature_names]
    if not indices:
        raise ValueError("None of the requested features are present in the dataset")

    mask = np.isin(y, args.labels)
    X_sel = X[mask][:, indices]
    y_sel = y[mask]

    for feature, idx in zip(args.features, indices):
        values = X_sel[:, args.features.index(feature)]
        print(f"\nFeature: {feature}")
        print("Label\tCount\tMean\tStd\tMin\tMax\tP25\tP50\tP75")
        for size_label in args.labels:
            subset = values[y_sel == size_label]
            if subset.size == 0:
                print(f"{size_label}\t0\t-\t-\t-\t-\t-\t-\t-")
                continue
            stats = {
                "count": subset.size,
                "mean": float(np.mean(subset)),
                "std": float(np.std(subset)),
                "min": float(np.min(subset)),
                "max": float(np.max(subset)),
                "p25": float(np.percentile(subset, 25)),
                "p50": float(np.percentile(subset, 50)),
                "p75": float(np.percentile(subset, 75)),
            }
            print(
                f"{size_label}\t"
                f"{stats['count']}\t"
                f"{stats['mean']:.2f}\t"
                f"{stats['std']:.2f}\t"
                f"{stats['min']:.2f}\t"
                f"{stats['max']:.2f}\t"
                f"{stats['p25']:.2f}\t"
                f"{stats['p50']:.2f}\t"
                f"{stats['p75']:.2f}"
            )


if __name__ == "__main__":
    main()
