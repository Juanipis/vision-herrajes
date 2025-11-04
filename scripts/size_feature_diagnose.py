"""Analyse feature separation across consecutive size labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "submodels" / "size"


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(str)
    feature_names = [str(name) for name in data["feature_names"].tolist()]
    return X, y, feature_names


def describe_feature(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def format_stats(label: str, stats: dict) -> str:
    return (
        f"{label:>4} | n={stats['count']:3d} | "
        f"mean={stats['mean']:.3f} | std={stats['std']:.3f} | "
        f"range=({stats['min']:.3f}, {stats['max']:.3f}) | "
        f"P10={stats['p10']:.3f} | P50={stats['p50']:.3f} | P90={stats['p90']:.3f}"
    )


def inspect_pairs(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    sizes: Sequence[str],
) -> None:
    size_order = sorted({s for s in y if s in sizes}, key=lambda s: sizes.index(s))
    for idx in range(len(size_order) - 1):
        a, b = size_order[idx], size_order[idx + 1]
        print(f"\n=== Pair: {a} vs {b} ===")
        mask_a = y == a
        mask_b = y == b
        for feat_idx, feat_name in enumerate(feature_names):
            vals_a = X[mask_a, feat_idx]
            vals_b = X[mask_b, feat_idx]
            if not vals_a.size or not vals_b.size:
                continue
            stats_a = describe_feature(vals_a)
            stats_b = describe_feature(vals_b)
            overlap = not (stats_a["max"] < stats_b["min"] or stats_b["max"] < stats_a["min"])
            print(f"Feature: {feat_name} | overlap={'YES' if overlap else 'NO'}")
            print("  " + format_stats(a, stats_a))
            print("  " + format_stats(b, stats_b))

        # Also track absolute extremes for debugging
        extremes = []
        for feat_idx, feat_name in enumerate(feature_names):
            vals_a = X[mask_a, feat_idx]
            vals_b = X[mask_b, feat_idx]
            if not vals_a.size or not vals_b.size:
                continue
            extremes.append(
                (
                    feat_name,
                    float(vals_a.min()),
                    float(vals_a.max()),
                    float(vals_b.min()),
                    float(vals_b.max()),
                )
            )
        if extremes:
            print("\n  Extremes (minA, maxA, minB, maxB):")
            for name, min_a, max_a, min_b, max_b in extremes:
                print(f"    {name:>28}: {min_a:.3f}, {max_a:.3f}, {min_b:.3f}, {max_b:.3f}")


def discover_families(root: Path) -> Iterable[Path]:
    for family_dir in sorted(root.iterdir()):
        if family_dir.is_dir():
            yield family_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Base directory for size datasets")
    parser.add_argument("--family", type=str, default="anillo", help="Family to analyse (use '*' for all)")
    parser.add_argument("--qualities", nargs="*", default=["bueno"], help="Qualities slug(s) to inspect")
    parser.add_argument("--sizes", nargs="*", default=["T1", "T2", "T3", "T4"], help="Ordered size labels")
    args = parser.parse_args()

    families = discover_families(args.data_root) if args.family == "*" else [args.data_root / args.family]
    for family_dir in families:
        if not family_dir.is_dir():
            continue
        print(f"\n===== Family: {family_dir.name} =====")
        for quality in args.qualities:
            dataset_path = family_dir / quality / "dataset.npz"
            if not dataset_path.exists():
                print(f"[WARN] Missing dataset: {dataset_path}")
                continue
            print(f"\n--- Quality: {quality} ---")
            X, y, feature_names = load_dataset(dataset_path)
            inspect_pairs(X, y, feature_names, args.sizes)


if __name__ == "__main__":
    main()
