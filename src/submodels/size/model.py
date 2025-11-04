"""Utilities for persisting and running size classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from .features import SIZE_FEATURE_NAMES, SizeFeatureVector, extract_size_features, extract_size_features_from_path


@dataclass(frozen=True)
class SizeModelBundle:
    family: str
    size_labels: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    model: BaseEstimator
    scaler: StandardScaler
    metadata: Dict[str, object]

    def predict(self, mask: np.ndarray) -> Tuple[str, float, np.ndarray, SizeFeatureVector]:
        fv = extract_size_features(mask)
        return self._predict_from_vector(fv)

    def predict_from_path(self, mask_path: Path) -> Tuple[str, float, np.ndarray, SizeFeatureVector]:
        fv = extract_size_features_from_path(mask_path)
        return self._predict_from_vector(fv)

    def transform(self, fv: SizeFeatureVector) -> np.ndarray:
        mapping = {name: idx for idx, name in enumerate(fv.names)}
        ordered = np.array([fv.values[mapping.get(name, 0)] for name in self.feature_names], dtype=np.float32)
        return ordered

    def _predict_from_vector(self, fv: SizeFeatureVector) -> Tuple[str, float, np.ndarray, SizeFeatureVector]:
        ordered = self.transform(fv)
        X = ordered.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[0]
        best_idx = int(np.argmax(probs))
        label = self.size_labels[best_idx]
        return label, float(probs[best_idx]), probs, SizeFeatureVector(ordered, tuple(self.feature_names))


def load_bundle(path: Path) -> SizeModelBundle:
    payload = joblib.load(path)
    required_keys = {"family", "size_labels", "feature_names", "model", "scaler"}
    missing = required_keys.difference(payload)
    if missing:
        raise KeyError(f"Size model bundle missing keys: {missing}")
    feature_names = tuple(payload.get("feature_names", SIZE_FEATURE_NAMES))
    metadata = {k: v for k, v in payload.items() if k not in required_keys}
    return SizeModelBundle(
        family=str(payload["family"]),
        size_labels=tuple(payload["size_labels"]),
        feature_names=feature_names,
        model=payload["model"],
        scaler=payload["scaler"],
        metadata=metadata,
    )


def save_bundle(
    path: Path,
    family: str,
    size_labels: Sequence[str],
    feature_names: Sequence[str],
    model: BaseEstimator,
    scaler: StandardScaler,
    **metadata: object,
) -> None:
    payload: Dict[str, object] = {
        "family": family,
        "size_labels": tuple(size_labels),
        "feature_names": tuple(feature_names),
        "model": model,
        "scaler": scaler,
    }
    payload.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
