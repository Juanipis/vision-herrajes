"""Reprocess captured frames for submodels using a dedicated preset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "filter_presets.json"
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.json"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "submodels" / "base"
DEFAULT_PRESET = "PHANSALKAR"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.processing.pipeline import FilterParameters, FilterPipeline  # noqa: E402


def load_preset(name: str) -> FilterParameters:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Preset file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if name not in data:
        raise KeyError(f"Preset '{name}' not present in {CONFIG_PATH}")
    params = FilterParameters.from_dict(data[name])
    # Frames stored in the manifest are already cropped to the ROI.
    params.roi_left_pct = 0.0
    params.roi_right_pct = 0.0
    return params


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
        help="Manifest from the main dataset build (defaults to data/processed/manifest.json)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory containing frame/mask files referenced by the source manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will receive the reprocessed frames/masks",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        help="Preset name to apply when regenerating masks (default: PHANSALKAR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report without writing files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N entries (useful for quick tests)",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Source manifest must be a list of records")
    return data


def iterate_entries(entries: Iterable[Dict[str, object]], limit: Optional[int]) -> Iterable[Dict[str, object]]:
    if limit is None:
        yield from entries
    else:
        for idx, entry in enumerate(entries):
            if idx >= limit:
                break
            yield entry


def reprocess_entry(
    entry: Dict[str, object],
    pipeline: FilterPipeline,
    source_root: Path,
    output_root: Path,
    preset_name: str,
    dry_run: bool,
) -> Optional[Dict[str, object]]:
    label = str(entry.get("label"))
    frame_rel = Path(str(entry.get("frame_path")))
    mask_rel = Path(str(entry.get("mask_path")))

    source_frame_path = source_root / frame_rel
    if not source_frame_path.exists():
        print(f"[WARN] Missing frame for entry: {source_frame_path}")
        return None

    frame = cv2.imread(str(source_frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[WARN] Failed to read frame: {source_frame_path}")
        return None

    mask = pipeline.apply(frame)

    if dry_run:
        return None

    dest_frame_path = output_root / frame_rel
    dest_mask_path = output_root / mask_rel

    ensure_parent(dest_frame_path)
    ensure_parent(dest_mask_path)

    if not dest_frame_path.exists():
        shutil.copy2(source_frame_path, dest_frame_path)

    if not cv2.imwrite(str(dest_mask_path), mask):
        print(f"[WARN] Failed to write mask: {dest_mask_path}")
        return None

    updated = dict(entry)
    updated["frame_path"] = str(dest_frame_path.relative_to(output_root))
    updated["mask_path"] = str(dest_mask_path.relative_to(output_root))
    updated["preset"] = preset_name
    updated["preset_name"] = preset_name
    return updated


def write_manifest(output_root: Path, entries: List[Dict[str, object]]) -> Path:
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    return manifest_path


def write_summary(
    output_root: Path,
    counts: Counter,
    source_manifest: Path,
    source_root: Path,
    preset: str,
    processed: int,
) -> None:
    summary = {
        "source_manifest": str(source_manifest),
        "source_root": str(source_root),
        "preset": preset,
        "processed_entries": processed,
        "labels": dict(counts),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()

    source_manifest = args.source_manifest.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output.resolve()

    entries = load_manifest(source_manifest)
    params = load_preset(args.preset)

    pipeline = FilterPipeline()
    pipeline.set_parameters(params)

    output_root.mkdir(parents=True, exist_ok=True)

    new_entries: List[Dict[str, object]] = []
    label_counts: Counter = Counter()

    iterator = iterate_entries(entries, args.limit)
    total = args.limit if args.limit is not None else len(entries)
    for entry in tqdm(iterator, total=total, desc="Frames"):
        updated = reprocess_entry(entry, pipeline, source_root, output_root, args.preset, args.dry_run)
        if updated is None:
            continue
        new_entries.append(updated)
        label_counts[str(entry.get("label"))] += 1

    if args.dry_run:
        print("Dry run completed; no files were written.")
        return

    manifest_path = write_manifest(output_root, new_entries)
    write_summary(output_root, label_counts, source_manifest, source_root, args.preset, len(new_entries))

    print("Manifest written:", manifest_path)
    print("Total entries:", len(new_entries))
    print("Label counts:", dict(label_counts))


if __name__ == "__main__":
    main()
