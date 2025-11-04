"""Feature extraction tailored for size classification submodels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

SIZE_FEATURE_NAMES: Tuple[str, ...] = (
    "outer_area_px",
    "outer_equiv_diameter_px",
    "hole_equiv_diameter_mean_px",
    "thickness_mean_px",
)


@dataclass(frozen=True)
class SizeFeatureVector:
    values: np.ndarray
    names: Tuple[str, ...]


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return binary


def _prepare_binary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    return binary


def _outer_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _hole_contours(mask: np.ndarray) -> Iterable[np.ndarray]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    internal: list[np.ndarray] = []
    for idx, contour in enumerate(contours):
        parent = hierarchy[0][idx][3]
        if parent < 0:
            continue
        internal.append(contour)
    return internal


def _fit_major_minor_axes(contour: np.ndarray) -> Tuple[float, float]:
    if contour is None or len(contour) < 5:
        return 0.0, 0.0
    (_, _), (major, minor), _ = cv2.fitEllipse(contour)
    major_f = float(max(major, minor))
    minor_f = float(min(major, minor))
    return major_f, minor_f


def extract_size_features(mask: np.ndarray) -> SizeFeatureVector:
    binary = _prepare_binary(mask)
    if binary.sum() == 0:
        return SizeFeatureVector(np.zeros(len(SIZE_FEATURE_NAMES), dtype=np.float32), SIZE_FEATURE_NAMES)

    outer = _outer_contour(binary)
    if outer is None:
        return SizeFeatureVector(np.zeros(len(SIZE_FEATURE_NAMES), dtype=np.float32), SIZE_FEATURE_NAMES)

    outer_area = float(cv2.contourArea(outer))
    outer_perimeter = float(cv2.arcLength(outer, closed=True))
    equiv_diameter = float(np.sqrt(4.0 * outer_area / np.pi)) if outer_area > 0 else 0.0
    major_axis, minor_axis = _fit_major_minor_axes(outer)

    # Bounding box is not needed for the current feature set; retain for potential future use if required.

    hole_contours = list(_hole_contours(binary))
    hole_areas: Sequence[float] = [float(cv2.contourArea(cnt)) for cnt in hole_contours]

    if hole_areas:
        hole_equiv = np.array([np.sqrt(4.0 * area / np.pi) if area > 0 else 0.0 for area in hole_areas], dtype=np.float32)
        hole_equiv_mean = float(hole_equiv.mean())
    else:
        hole_equiv_mean = 0.0

    ring_mask = binary.copy()
    for cnt in hole_contours:
        cv2.drawContours(ring_mask, [cnt], -1, 0, thickness=cv2.FILLED)

    dist = cv2.distanceTransform(ring_mask, cv2.DIST_L2, 5)
    dist_values = dist[ring_mask > 0]
    if dist_values.size:
        # Multiply by 2 to approximate full thickness rather than radius to boundary.
        thickness_mean = float(dist_values.mean() * 2.0)
    else:
        thickness_mean = 0.0

    feature_values = np.array(
        [
            outer_area,
            equiv_diameter,
            hole_equiv_mean,
            thickness_mean,
        ],
        dtype=np.float32,
    )

    return SizeFeatureVector(feature_values, SIZE_FEATURE_NAMES)


def extract_size_features_from_path(mask_path: Path) -> SizeFeatureVector:
    mask = load_mask(mask_path)
    return extract_size_features(mask)
