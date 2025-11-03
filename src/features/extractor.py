"""Feature extraction utilities for binarised herraje masks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

FEATURE_NAMES: Tuple[str, ...] = (
    "hu_1",
    "hu_2",
    "hu_3",
    "hu_4",
    "hu_5",
    "hu_6",
    "hu_7",
    "fd_1",
    "fd_2",
    "fd_3",
    "fd_4",
    "fd_5",
    "fd_6",
    "fd_7",
    "fd_8",
    "fd_9",
    "fd_10",
    "holes",
    "radial_std",
    "radial_min",
    "radial_max",
    "skeleton_endpoints",
    "skeleton_junctions",
    "holes",
    "hole_ar_mean",
    "hole_ar_std",
    "hole_ecc_mean",
    "hole_centroid_ar",
)


def _prepare_binary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    return binary


@dataclass
class FeatureVector:
    values: np.ndarray
    names: Tuple[str, ...]


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return binary


def _largest_contour(contours: Iterable[np.ndarray]) -> np.ndarray | None:
    contours = list(contours)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _hu_moments(contour: np.ndarray) -> np.ndarray:
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale for stability
    with np.errstate(divide="ignore"):
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu


def _fourier_descriptors(contour: np.ndarray, n_components: int = 10) -> np.ndarray:
    contour = contour.squeeze(axis=1).astype(np.float64)
    if contour.ndim != 2 or contour.shape[0] < n_components + 1:
        return np.zeros(n_components, dtype=np.float32)
    complex_signal = contour[:, 0] + 1j * contour[:, 1]
    complex_signal -= complex_signal.mean()
    fft_coeffs = np.fft.fft(complex_signal)
    magnitudes = np.abs(fft_coeffs)
    magnitudes = magnitudes[1:]  # drop DC component
    if magnitudes.size == 0 or magnitudes[0] == 0:
        return np.zeros(n_components, dtype=np.float32)
    magnitudes /= magnitudes[0]
    magnitudes = np.abs(magnitudes[:n_components])
    if magnitudes.size < n_components:
        magnitudes = np.pad(magnitudes, (0, n_components - magnitudes.size))
    return magnitudes.astype(np.float32)


def _find_internal_contours(mask: np.ndarray, area_threshold_ratio: float = 0.01) -> list[np.ndarray]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    total_area = float(np.count_nonzero(mask))
    min_area = total_area * area_threshold_ratio
    internal: list[np.ndarray] = []
    for idx, contour in enumerate(contours):
        parent = hierarchy[0][idx][3]
        if parent < 0:
            continue
        area = cv2.contourArea(contour)
        if area >= min_area:
            internal.append(contour)
    return internal


def _hole_count(mask: np.ndarray) -> int:
    return len(_find_internal_contours(mask))


def _hole_shape_stats(mask: np.ndarray) -> Tuple[float, float, float, float]:
    internal = _find_internal_contours(mask)
    if not internal:
        return 0.0, 0.0, 0.0, 0.0

    aspect_ratios: list[float] = []
    eccentricities: list[float] = []
    centers: list[Tuple[float, float]] = []

    for contour in internal:
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        ar = max(w, h) / max(1, min(w, h))
        aspect_ratios.append(float(ar))

        if len(contour) >= 5:
            (_, _), (major, minor), _ = cv2.fitEllipse(contour)
            major, minor = float(max(major, minor)), float(min(major, minor))
            if major > 1e-6:
                eccentricities.append(float(np.sqrt(1 - (minor / major) ** 2)))

        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            centers.append((cx, cy))

    ar_mean = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
    ar_std = float(np.std(aspect_ratios)) if aspect_ratios else 0.0
    ecc_mean = float(np.mean(eccentricities)) if eccentricities else 0.0

    if len(centers) >= 2:
        centers_arr = np.array(centers)
        min_xy = centers_arr.min(axis=0)
        max_xy = centers_arr.max(axis=0)
        width, height = max_xy - min_xy
        if width < 1e-6 or height < 1e-6:
            centroid_ar = 0.0
        else:
            centroid_ar = float(max(width, height) / min(width, height))
    else:
        centroid_ar = 0.0

    return ar_mean, ar_std, ecc_mean, centroid_ar


def _radial_stats(contour: np.ndarray) -> Tuple[float, float, float]:
    points = contour.squeeze(axis=1).astype(np.float64)
    moments = cv2.moments(points)
    if moments["m00"] == 0:
        return 0.0, 0.0, 0.0
    centroid = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    distances = np.linalg.norm(points - centroid, axis=1)
    if distances.size == 0:
        return 0.0, 0.0, 0.0
    distances = distances / (distances.mean() + 1e-8)
    std = float(np.std(distances))
    min_ratio = float(np.min(distances))
    max_ratio = float(np.max(distances))
    return std, min_ratio, max_ratio


def _skeleton_metrics(mask: np.ndarray) -> Tuple[int, int]:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0, 0
    skeleton = skeletonize(binary).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndimage.convolve(skeleton, kernel, mode="constant") - skeleton
    endpoints = int(np.sum((skeleton == 1) & (neighbor_count == 1)))
    junctions = int(np.sum((skeleton == 1) & (neighbor_count >= 4)))
    return endpoints, junctions


def extract_features(mask_path: Path) -> FeatureVector:
    mask = load_mask(mask_path)
    return extract_features_from_mask(mask)


def extract_features_from_mask(mask: np.ndarray) -> FeatureVector:
    mask = _prepare_binary(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = _largest_contour(contours)
    if main_contour is None:
        return FeatureVector(np.zeros(len(FEATURE_NAMES), dtype=np.float32), FEATURE_NAMES)

    hu = _hu_moments(main_contour)
    fd = _fourier_descriptors(main_contour)
    holes = _hole_count(mask)
    radial_std, radial_min, radial_max = _radial_stats(main_contour)
    endpoints, junctions = _skeleton_metrics(mask)
    hole_count = _hole_count(mask)
    hole_ar_mean, hole_ar_std, hole_ecc_mean, hole_centroid_ar = _hole_shape_stats(mask)

    feature_values = [
        *hu.tolist(),
        *fd.tolist(),
        float(holes),
        float(radial_std),
        float(radial_min),
        float(radial_max),
        float(endpoints),
        float(junctions),
        float(hole_count),
        float(hole_ar_mean),
        float(hole_ar_std),
        float(hole_ecc_mean),
        float(hole_centroid_ar),
    ]

    return FeatureVector(np.array(feature_values, dtype=np.float32), FEATURE_NAMES)
