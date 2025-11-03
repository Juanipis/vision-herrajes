"""Binarisation helpers for the vision herrajes pipeline."""
from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def gaussian_blur(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(0, int(kernel_size))
    if kernel_size <= 0:
        return gray
    kernel_size = _ensure_odd(kernel_size, minimum=1)
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def manual_threshold(gray: np.ndarray, threshold: int) -> np.ndarray:
    threshold = int(np.clip(threshold, 0, 255))
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def adaptive_mean_threshold(gray: np.ndarray, block_size: int, c_value: float) -> np.ndarray:
    block_size = _ensure_odd(block_size)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )


def adaptive_gaussian_threshold(gray: np.ndarray, block_size: int, c_value: float) -> np.ndarray:
    block_size = _ensure_odd(block_size)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )


def _local_mean_std(gray: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    block_size = _ensure_odd(block_size)
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (block_size, block_size))
    sq_mean = cv2.blur(gray_f * gray_f, (block_size, block_size))
    variance = np.maximum(sq_mean - mean * mean, 0.0)
    std = np.sqrt(variance)
    return mean, std


def _binary_from_threshold(gray: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    mask = (gray.astype(np.float32) > threshold).astype(np.uint8) * 255
    return mask


def niblack_threshold(gray: np.ndarray, block_size: int, k: float) -> np.ndarray:
    mean, std = _local_mean_std(gray, block_size)
    threshold = mean + k * std
    return _binary_from_threshold(gray, threshold)


def sauvola_threshold(gray: np.ndarray, block_size: int, k: float, r: float) -> np.ndarray:
    mean, std = _local_mean_std(gray, block_size)
    r = max(float(r), 1e-5)
    threshold = mean * (1 + k * ((std / r) - 1))
    return _binary_from_threshold(gray, threshold)


def wolf_threshold(gray: np.ndarray, block_size: int, k: float) -> np.ndarray:
    mean, std = _local_mean_std(gray, block_size)
    min_val = float(gray.min())
    max_std = std.max()
    max_std = max(max_std, 1e-5)
    threshold = ((1 - k) * mean) + (k * (min_val + ((std / max_std) * (mean - min_val))))
    return _binary_from_threshold(gray, threshold)


def nick_threshold(gray: np.ndarray, block_size: int, k: float) -> np.ndarray:
    mean, std = _local_mean_std(gray, block_size)
    threshold = mean + k * np.sqrt(std * std + mean * mean)
    return _binary_from_threshold(gray, threshold)


def phansalkar_threshold(gray: np.ndarray, block_size: int, k: float, p: float, q: float) -> np.ndarray:
    mean, std = _local_mean_std(gray, block_size)
    gray_norm = gray.astype(np.float32) / 255.0
    mean_norm = mean / 255.0
    std_norm = std / 255.0
    threshold = mean_norm * (1.0 + p * np.exp(-q * mean_norm) + k * ((std_norm / 0.5) - 1.0))
    threshold = np.clip(threshold * 255.0, 0, 255)
    return _binary_from_threshold(gray, threshold)


def dilate(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    keep_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep_mask[labels == label] = 255
    return keep_mask


def morphological_close(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
