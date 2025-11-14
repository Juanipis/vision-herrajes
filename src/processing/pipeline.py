"""Processing pipeline that applies tunable filters to video frames."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import cv2
import numpy as np

from . import filters


@dataclass
class FilterParameters:
    threshold_method: str = "manual"
    manual_threshold: int = 127
    pre_blur_kernel: int = 0
    block_size: int = 31
    constant_C: float = 0.0
    niblack_k: float = -0.2
    sauvola_k: float = 0.5
    sauvola_r: float = 128.0
    wolf_k: float = 0.5
    nick_k: float = -0.2
    phansalkar_k: float = 0.25
    phansalkar_p: float = 3.0
    phansalkar_q: float = 10.0
    roi_left_pct: float = 0.2
    roi_right_pct: float = 0.2
    trigger_line_pct: float = 0.5
    trigger_band_pct: float = 0.05
    expand_mask: bool = False
    expand_kernel: int = 3
    expand_iterations: int = 1
    use_closing: bool = False
    closing_kernel: int = 0
    closing_iterations: int = 0
    min_component_area: int = 0
    invert_output: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FilterParameters":
        fields = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[arg-type]
        kwargs = {key: value for key, value in data.items() if key in fields}
        return cls(**kwargs)  # type: ignore[arg-type]


class FilterPipeline:
    """Applies configurable thresholding to video frames within a ROI."""

    def __init__(self) -> None:
        self._params = FilterParameters()

    @property
    def parameters(self) -> FilterParameters:
        return self._params

    def set_parameters(self, params: FilterParameters) -> None:
        self._params = params

    def reset_state(self) -> None:
        """Placeholder for compatibility. Thresholding has no state."""
        return

    def apply(self, frame: np.ndarray) -> np.ndarray:
        params = self._params
        gray = filters.to_grayscale(frame)

        height, width = gray.shape
        left = int(width * params.roi_left_pct)
        right = int(width * (1.0 - params.roi_right_pct))
        left = max(0, min(left, width - 1))
        right = max(left + 1, min(width, right))

        roi = gray[:, left:right]
        roi_proc = roi
        if params.pre_blur_kernel > 0:
            roi_proc = filters.gaussian_blur(roi_proc, params.pre_blur_kernel)

        method = params.threshold_method
        if roi.size == 0:
            binary_roi = np.zeros_like(roi, dtype=np.uint8)
        else:
            if method == "manual":
                binary_roi = filters.manual_threshold(roi_proc, params.manual_threshold)
            elif method == "otsu":
                binary_roi = filters.otsu_threshold(roi_proc)
            elif method == "adaptive_mean":
                binary_roi = filters.adaptive_mean_threshold(roi_proc, params.block_size, params.constant_C)
            elif method == "adaptive_gaussian":
                binary_roi = filters.adaptive_gaussian_threshold(roi_proc, params.block_size, params.constant_C)
            elif method == "niblack":
                binary_roi = filters.niblack_threshold(roi_proc, params.block_size, params.niblack_k)
            elif method == "sauvola":
                binary_roi = filters.sauvola_threshold(roi_proc, params.block_size, params.sauvola_k, params.sauvola_r)
            elif method == "wolf":
                binary_roi = filters.wolf_threshold(roi_proc, params.block_size, params.wolf_k)
            elif method == "nick":
                binary_roi = filters.nick_threshold(roi_proc, params.block_size, params.nick_k)
            elif method == "phansalkar":
                binary_roi = filters.phansalkar_threshold(
                    roi_proc,
                    params.block_size,
                    params.phansalkar_k,
                    params.phansalkar_p,
                    params.phansalkar_q,
                )
            else:
                binary_roi = filters.manual_threshold(roi_proc, params.manual_threshold)

        if params.use_closing and params.closing_kernel > 0 and params.closing_iterations > 0:
            binary_roi = filters.morphological_close(binary_roi, params.closing_kernel, params.closing_iterations)

        if params.expand_mask:
            binary_roi = filters.dilate(binary_roi, params.expand_kernel, params.expand_iterations)

        if params.min_component_area > 0:
            binary_roi = filters.remove_small_components(binary_roi, params.min_component_area)

        binary = np.zeros_like(gray, dtype=np.uint8)
        binary[:, left:right] = binary_roi

        if params.invert_output:
            binary = cv2.bitwise_not(binary)

        return binary
