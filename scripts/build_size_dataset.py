"""Build a size-specific dataset for a particular herraje family."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "submodels" / "base" / "manifest.json"
DEFAULT_MASK_ROOT = PROJECT_ROOT / "data" / "submodels" / "base"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "submodels" / "size"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.submodels.size import dataset as size_dataset  # noqa: E402


def normalise_qualities(qualities: Sequence[str]) -> List[str]:
    if not qualities:
        return ["BUENO"]
    norm = {q.upper() for q in qualities if q}
    if "*" in norm:
        return ["ALL"]
    if "ALL" in norm:
        return ["ALL"]
    return sorted(norm)


def quality_slug(qualities: Sequence[str]) -> str:
    if not qualities or qualities == ["ALL"]:
        return "all"
    return "-".join(sorted(q.lower() for q in qualities))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to submodel manifest.json")
    parser.add_argument("--family", type=str, default="anillo", help="Herraje family to target (use '*' for all)")
    parser.add_argument(
        "--quality",
        action="append",
        dest="qualities",
        default=None,
        help="Quality token(s) to include (e.g. BUENO, MALO). Use --quality ALL to disable filtering.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Base directory for size datasets")
    parser.add_argument("--balance", action="store_true", help="Oversample smaller sizes to match the largest class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balancing")
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=DEFAULT_MASK_ROOT,
        help="Root directory where masks referenced by the manifest reside",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qualities = normalise_qualities(args.qualities or ["BUENO"])

    size_dataset.set_processed_root(args.mask_root)

    records = size_dataset.load_processed_manifest(args.manifest)
    families = sorted({str(record.get("label")) for record in records}) if args.family == "*" else [args.family]

    processed_any = False
    for family in families:
        entries = size_dataset.filter_records(records, family, qualities)
        if not entries:
            print(f"[WARN] No entries found for family '{family}' with qualities {qualities}")
            continue

        original_counts = size_dataset.summarise_counts(entries)

        if args.balance:
            entries = size_dataset.rebalance_entries(entries, seed=args.seed)

        balanced_counts = size_dataset.summarise_counts(entries)

        print("Manifest:", args.manifest)
        print("Mask root:", args.mask_root)
        print("Family:", family)
        print("Qualities:", qualities)
        print("Original size counts:", original_counts)
        if args.balance:
            print("Balanced size counts:", balanced_counts)

        print("Extracting features...", flush=True)
        with tqdm(total=len(entries), desc=f"Frames-{family}") as bar:
            features: List[np.ndarray] = []
            labels: List[str] = []
            feature_names = None
            for entry in entries:
                mask_path = size_dataset.PROCESSED_ROOT / entry.mask_path
                if not mask_path.exists():
                    bar.update(1)
                    continue
                fv = size_dataset.extract_size_features_from_path(mask_path)
                if feature_names is None:
                    feature_names = fv.names
                features.append(fv.values)
                labels.append(entry.size_label)
                bar.update(1)

        if not features:
            print(f"[WARN] No features extracted for family '{family}'")
            continue

        X = np.vstack(features)
        y = np.array(labels)
        feature_names = feature_names or size_dataset.SIZE_FEATURE_NAMES

        qual_slug = quality_slug(qualities)
        output_dir = args.output / family / qual_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = size_dataset.to_manifest(entries)
        manifest_path = output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        dataset_path = output_dir / "dataset.npz"
        np.savez_compressed(
            dataset_path,
            X=X,
            y=y,
            feature_names=np.array(feature_names),
            size_labels=np.unique(y),
            qualities=qualities,
            family=family,
            balance=args.balance,
            seed=args.seed,
            generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            manifest_path=str(manifest_path.relative_to(PROJECT_ROOT)),
        )

        summary = {
            "family": family,
            "qualities": qualities,
            "balance": args.balance,
            "seed": args.seed,
            "original_counts": original_counts,
            "balanced_counts": balanced_counts,
            "samples": int(len(entries)),
            "features": len(feature_names),
            "dataset_npz": str(dataset_path.relative_to(PROJECT_ROOT)),
        }

        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print("Dataset saved to", output_dir)
        print("Feature matrix shape:", X.shape)
        print("Sizes:", sorted(np.unique(y)))
        processed_any = True

    if not processed_any:
        raise SystemExit("No datasets were generated. Check manifest, family filter, or qualities.")


if __name__ == "__main__":
    main()
