"""Dataset construction helpers for size-specific submodels."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .features import SIZE_FEATURE_NAMES, SizeFeatureVector, extract_size_features_from_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASK_ROOT = PROJECT_ROOT / "data" / "submodels" / "base"
PROCESSED_ROOT = DEFAULT_MASK_ROOT

SIZE_PATTERN = re.compile(r"_T(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SizeDatasetEntry:
    family: str
    size_label: str
    quality: str
    video_path: str
    frame_path: str
    mask_path: str
    frame_index: int
    alias: str


def load_processed_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Processed manifest must be a list of records")
    return data


def _detect_quality(stem: str) -> str:
    stem_upper = stem.upper()
    if "MALO" in stem_upper:
        return "MALO"
    if "BUENO" in stem_upper:
        return "BUENO"
    return "UNKNOWN"


def _detect_size(stem: str) -> str | None:
    match = SIZE_PATTERN.search(stem)
    if not match:
        return None
    return f"T{match.group(1)}"


def filter_records(
    records: Iterable[Dict[str, object]],
    family: str,
    qualities: Sequence[str],
) -> List[SizeDatasetEntry]:
    normalized_qualities = {q.upper() for q in qualities if q}
    allow_all = not normalized_qualities or "ALL" in normalized_qualities

    entries: List[SizeDatasetEntry] = []

    for record in records:
        if record.get("label") != family:
            continue
        video = str(record.get("video", ""))
        frame_path = str(record.get("frame_path", ""))
        mask_path = str(record.get("mask_path", ""))
        alias = str(record.get("alias", ""))
        size_label = _detect_size(Path(video).stem)
        if size_label is None:
            continue
        quality = _detect_quality(Path(video).stem)
        if not allow_all and quality not in normalized_qualities:
            continue
        entry = SizeDatasetEntry(
            family=family,
            size_label=size_label,
            quality=quality,
            video_path=video,
            frame_path=frame_path,
            mask_path=mask_path,
            frame_index=int(record.get("frame_index", -1)),
            alias=alias,
        )
        entries.append(entry)

    return entries


def _group_by_size(entries: Sequence[SizeDatasetEntry]) -> Dict[str, List[SizeDatasetEntry]]:
    grouped: Dict[str, List[SizeDatasetEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.size_label, []).append(entry)
    return grouped


def rebalance_entries(
    entries: Sequence[SizeDatasetEntry],
    seed: int = 42,
) -> List[SizeDatasetEntry]:
    grouped = _group_by_size(entries)
    if not grouped:
        return []
    target = max(len(group) for group in grouped.values())
    rng = random.Random(seed)
    balanced: List[SizeDatasetEntry] = []

    for size_label, group in grouped.items():
        if len(group) == target:
            balanced.extend(group)
            continue
        copies = group.copy()
        while len(copies) < target:
            copies.append(rng.choice(group))
        balanced.extend(copies[:target])

    return balanced


def compute_feature_matrix(entries: Sequence[SizeDatasetEntry]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    feature_names: Tuple[str, ...] | None = None

    for entry in entries:
        mask_rel = entry.mask_path
        mask_path = PROCESSED_ROOT / mask_rel
        if not mask_path.exists():
            continue
        fv: SizeFeatureVector = extract_size_features_from_path(mask_path)
        if feature_names is None:
            feature_names = fv.names
        features.append(fv.values)
        labels.append(entry.size_label)

    if not features:
        raise RuntimeError("No features extracted for size dataset")

    X = np.vstack(features)
    y = np.array(labels)
    if feature_names is None:
        feature_names = SIZE_FEATURE_NAMES
    return X, y, feature_names


def summarise_counts(entries: Sequence[SizeDatasetEntry]) -> Dict[str, int]:
    grouped = _group_by_size(entries)
    return {size: len(group) for size, group in grouped.items()}


def to_manifest(entries: Sequence[SizeDatasetEntry]) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []
    for entry in entries:
        manifest.append(
            {
                "family": entry.family,
                "size": entry.size_label,
                "quality": entry.quality,
                "video": entry.video_path,
                "frame_path": entry.frame_path,
                "mask_path": entry.mask_path,
                "frame_index": entry.frame_index,
                "alias": entry.alias,
            }
        )
    return manifest


def set_processed_root(path: Path) -> None:
    global PROCESSED_ROOT
    PROCESSED_ROOT = path.resolve()
