"""GUI for live classification of herraje masks using the trained MLP."""

from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

from vision_defect_detection import load_default_detector, predict_image

from ..features.extractor import FEATURE_NAMES, extract_features_from_mask
from ..processing.pipeline import FilterParameters, FilterPipeline
from ..submodels.size.features import extract_size_features
from ..submodels.size.model import SizeModelBundle, load_bundle as load_size_bundle

RESULTS_DIR = Path("results")
DEFAULT_PRESET_NAME = "PHANSALKAR"


@dataclass
class ModelBundle:
    model: object
    scaler: object
    feature_names: Tuple[str, ...]
    classes_: np.ndarray


@dataclass
class SizeModelOption:
    key: str
    family: str
    quality: str
    path: Path
    bundle: SizeModelBundle
    var: tk.BooleanVar


def resolve_latest_model() -> Optional[Path]:
    latest_dir = RESULTS_DIR / "latest_model" / "model.joblib"
    if latest_dir.exists():
        return latest_dir
    candidates = sorted((RESULTS_DIR).glob("mlp_*"), reverse=True)
    for candidate in candidates:
        model_path = candidate / "model.joblib"
        if model_path.exists():
            return model_path
    return None


def load_model(path: Optional[Path]) -> ModelBundle:
    if path is not None and path.exists():
        model_path = path
    else:
        latest = resolve_latest_model()
        if latest is None:
            raise FileNotFoundError(
                "No trained model found in results/. Run `make train` first or specify --model."
            )
        model_path = latest

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = tuple(bundle.get("features", []))
    classes_ = np.array(model.classes_)
    return ModelBundle(
        model=model, scaler=scaler, feature_names=feature_names, classes_=classes_
    )


def load_preset(
    name: Optional[str] = None, preset_file: Path = Path("config/filter_presets.json")
) -> FilterParameters:
    if preset_file.exists():
        with preset_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            if name and name in data:
                return FilterParameters.from_dict(data[name])
            elif DEFAULT_PRESET_NAME in data:
                return FilterParameters.from_dict(data[DEFAULT_PRESET_NAME])
    return FilterParameters.from_dict(
        {
            "threshold_method": "phansalkar",
            "block_size": 41,
            "phansalkar_k": 0.25,
            "phansalkar_p": 3.0,
            "phansalkar_q": 10.0,
            "roi_left_pct": 0.2,
            "roi_right_pct": 0.2,
            "pre_blur_kernel": 3,
            "use_closing": True,
            "closing_kernel": 5,
            "closing_iterations": 1,
            "expand_mask": True,
            "expand_kernel": 3,
            "expand_iterations": 1,
            "min_component_area": 500,
        }
    )


class VideoClassifierApp(tk.Tk):
    def __init__(self, model_path: Optional[Path], preset_name: Optional[str], size_preset_name: Optional[str]) -> None:
        super().__init__()
        self.title("Herrajes Classifier")
        self.geometry("1400x720")

        self.model_bundle = load_model(model_path)
        params = load_preset(preset_name)
        size_params = load_preset(size_preset_name) if size_preset_name else None
        otsu_params = load_preset("OTSU")

        self._feature_indices = self._compute_feature_indices()

        self.pipeline = FilterPipeline()
        self.pipeline.set_parameters(params)
        self._otsu_pipeline = FilterPipeline()
        self._otsu_pipeline.set_parameters(otsu_params)

        self._size_pipeline: Optional[FilterPipeline] = None
        self._size_preset_name = size_preset_name
        if size_params is not None:
            size_params.roi_left_pct = 0.0
            size_params.roi_right_pct = 0.0
            self._size_pipeline = FilterPipeline()
            self._size_pipeline.set_parameters(size_params)

        self.main_model_var = tk.BooleanVar(value=True)
        self.size_result_var = tk.StringVar(value="")
        self.defect_detector_var = tk.BooleanVar(value=True)
        self.defect_result_var = tk.StringVar(value="")
        self._defect_label_widget: Optional[tk.Label] = None
        self._defect_label_neutral_fg = "#AA8800"
        self._defect_detector = None
        self._defect_detector_lock = threading.Lock()
        self._defect_warmup_started = False
        self._size_model_options: List[SizeModelOption] = []
        self._size_model_lookup: Dict[str, List[SizeModelOption]] = {}
        self._size_models_container: Optional[tk.Widget] = None
        self._load_size_models()
        self._start_defect_warmup()

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[Path] = None
        self._stop_event = threading.Event()
        self._frame_queue: "queue.Queue[Tuple[np.ndarray, np.ndarray, int]]" = queue.Queue(
            maxsize=5
        )

        self._classification_lock = threading.Lock()
        self._last_prediction_time = 0.0
        self._prediction_cooldown = 0.3
        self._last_center_distance = None
        self._approaching = False
        self._last_centroid_x = None
        self._show_contours = False
        self._contour_var = tk.BooleanVar(value=False)
        self._contour_offset = 2
        self._contour_thickness = 2
        self._contour_offset_var = tk.IntVar(value=self._contour_offset)
        self._contour_thickness_var = tk.IntVar(value=self._contour_thickness)
        self._object_present = False
        self._classified_current = False
        self._roi_buffer: deque[np.ndarray] = deque(maxlen=4)
        self._frame_buffer: deque[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=4)
        self._classification_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        self._current_frame_index = 0
        self._current_frame: Optional[np.ndarray] = None
        self._current_processed: Optional[np.ndarray] = None
        self._current_otsu: Optional[np.ndarray] = None

        self._post_setup()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _post_setup(self) -> None:
        """Extension hook for subclasses to prepare state before building widgets."""
        return

    def _build_ui(self) -> None:
        control_bar = tk.Frame(self)
        control_bar.pack(fill="x", padx=10, pady=10)

        self._build_source_controls(control_bar)
        tk.Button(control_bar, text="Play", command=self._start_playback).pack(
            side="left", padx=4
        )
        tk.Button(control_bar, text="Stop", command=self._stop_playback).pack(
            side="left", padx=4
        )

        debug_bar = tk.Frame(self)
        debug_bar.pack(fill="x", padx=10, pady=(0, 6))
        tk.Checkbutton(
            debug_bar,
            text="Show hole contours",
            variable=self._contour_var,
            command=self._toggle_contours,
        ).pack(side="left")

        tk.Label(debug_bar, text="Offset").pack(side="left", padx=(12, 2))
        tk.Scale(
            debug_bar,
            from_=0,
            to=15,
            orient=tk.HORIZONTAL,
            showvalue=True,
            variable=self._contour_offset_var,
            command=lambda _: self._update_contour_params(),
            length=140,
        ).pack(side="left")

        tk.Label(debug_bar, text="Thickness").pack(side="left", padx=(12, 2))
        tk.Scale(
            debug_bar,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            showvalue=True,
            variable=self._contour_thickness_var,
            command=lambda _: self._update_contour_params(),
            length=140,
        ).pack(side="left")

        self.status_var = tk.StringVar(value="Load a video to begin")
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10)

        views = tk.Frame(self)
        views.pack(fill="both", expand=True, padx=10, pady=10)

        self.raw_canvas = VideoCanvas(views, "Raw video")
        self.raw_canvas.pack(side="left", padx=5)

        self.proc_canvas = VideoCanvas(views, "Processed mask")
        self.proc_canvas.pack(side="left", padx=5)

        self.otsu_canvas = VideoCanvas(views, "OTSU mask")
        self.otsu_canvas.pack(side="left", padx=5)

        self._build_model_sidebar(views)

        self.result_var = tk.StringVar(value="")
        result_label = tk.Label(
            self,
            textvariable=self.result_var,
            font=("Helvetica", 36, "bold"),
            fg="#008000",
        )
        result_label.pack(fill="x", pady=10)

        size_label = tk.Label(
            self,
            textvariable=self.size_result_var,
            font=("Helvetica", 24, "bold"),
            fg="#0044AA",
        )
        size_label.pack(fill="x")

        self._defect_label_widget = tk.Label(
            self,
            textvariable=self.defect_result_var,
            font=("Helvetica", 20, "bold"),
            fg=self._defect_label_neutral_fg,
        )
        self._defect_label_widget.pack(fill="x", pady=(4, 0))

        self.warning_var = tk.StringVar(value="")
        warning_label = tk.Label(
            self,
            textvariable=self.warning_var,
            font=("Helvetica", 20, "bold"),
            fg="#FF3333",
        )
        warning_label.pack(fill="x")

        self._update_contour_params()

    def _build_source_controls(self, control_bar: tk.Frame) -> None:
        tk.Button(control_bar, text="Open", command=self._open_video).pack(
            side="left", padx=4
        )

    def _compute_feature_indices(self) -> Tuple[int, ...]:
        if not self.model_bundle.feature_names:
            return tuple(range(len(FEATURE_NAMES)))
        mapping = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
        indices: list[int] = []
        for name in self.model_bundle.feature_names:
            if name not in mapping:
                raise ValueError(f"Feature '{name}' missing in current extractor definition")
            indices.append(mapping[name])
        return tuple(indices)

    def _discover_size_model_paths(self) -> List[Tuple[str, str, Path]]:
        base = RESULTS_DIR / "submodels" / "size"
        discovered: List[Tuple[str, str, Path]] = []
        if not base.exists():
            return discovered
        for family_dir in sorted(base.iterdir()):
            if not family_dir.is_dir():
                continue
            family = family_dir.name
            for quality_dir in sorted(family_dir.iterdir()):
                if not quality_dir.is_dir():
                    continue
                quality = quality_dir.name
                latest = quality_dir / "latest" / "model.joblib"
                if latest.exists():
                    discovered.append((family, quality, latest))
                    continue
                candidates = sorted(quality_dir.glob("mlp_*/model.joblib"), reverse=True)
                if candidates:
                    discovered.append((family, quality, candidates[0]))
        return discovered

    def _load_size_models(self, previous_states: Optional[Dict[str, bool]] = None) -> None:
        options: List[SizeModelOption] = []
        lookup: Dict[str, List[SizeModelOption]] = {}
        for family, quality, path in self._discover_size_model_paths():
            try:
                bundle = load_size_bundle(path)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[WARN] Failed to load size model {path}: {exc}")
                continue
            key = f"{family}:{quality}"
            initial = True
            if previous_states and key in previous_states:
                initial = bool(previous_states[key])
            var = tk.BooleanVar(value=initial)
            option = SizeModelOption(
                key=key,
                family=family,
                quality=quality,
                path=path,
                bundle=bundle,
                var=var,
            )
            options.append(option)
            lookup.setdefault(family, []).append(option)

        for family, opts in lookup.items():
            opts.sort(key=lambda item: (item.quality, item.path.name))

        self._size_model_options = options
        self._size_model_lookup = lookup

    def _build_model_sidebar(self, parent: tk.Widget) -> None:
        self.model_sidebar = tk.LabelFrame(parent, text="Models", padx=8, pady=8)
        self.model_sidebar.pack(side="left", fill="y", padx=5)

        tk.Checkbutton(
            self.model_sidebar,
            text="Main family classifier",
            variable=self.main_model_var,
        ).pack(anchor="w")

        tk.Checkbutton(
            self.model_sidebar,
            text="Defect detector",
            variable=self.defect_detector_var,
            command=self._handle_defect_toggle,
        ).pack(anchor="w")

        self._size_models_container = tk.Frame(self.model_sidebar)
        self._size_models_container.pack(fill="x", pady=(6, 6))
        self._populate_size_model_toggles()

        if self._size_preset_name:
            tk.Label(
                self.model_sidebar,
                text=f"Size preset: {self._size_preset_name}",
                fg="#555",
            ).pack(anchor="w", pady=(4, 0))

        tk.Button(
            self.model_sidebar,
            text="Rescan models",
            command=self._rescan_size_models,
        ).pack(anchor="center", pady=(6, 0))

    def _populate_size_model_toggles(self) -> None:
        if self._size_models_container is None:
            return
        for child in self._size_models_container.winfo_children():
            child.destroy()

        if not self._size_model_options:
            tk.Label(
                self._size_models_container,
                text="No size submodels found",
                fg="#777",
            ).pack(anchor="w")
            return

        tk.Label(
            self._size_models_container,
            text="Size submodels",
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w")

        for option in self._size_model_options:
            label = f"{option.family} [{option.quality}]"
            tk.Checkbutton(
                self._size_models_container,
                text=label,
                variable=option.var,
            ).pack(anchor="w", padx=4)

    def _handle_defect_toggle(self) -> None:
        if self.defect_detector_var.get():
            self._start_defect_warmup()
        else:
            self.defect_result_var.set("")
            if self._defect_label_widget is not None:
                self._defect_label_widget.config(fg=self._defect_label_neutral_fg)

    def _start_defect_warmup(self) -> None:
        if not self.defect_detector_var.get():
            return
        if self._defect_detector is not None:
            return
        if self._defect_warmup_started:
            return
        self._defect_warmup_started = True
        threading.Thread(target=self._warm_up_defect_detector, daemon=True).start()

    def _warm_up_defect_detector(self) -> None:
        detector = self._ensure_defect_detector()
        if detector is None:
            self._defect_warmup_started = False
        else:
            print("[INFO] Defect detector warmed up")

    def _ensure_defect_detector(self):
        if self._defect_detector is not None:
            return self._defect_detector
        with self._defect_detector_lock:
            if self._defect_detector is not None:
                return self._defect_detector
            try:
                detector = load_default_detector()
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[WARN] Unable to load defect detector: {exc}")
                return None
            self._defect_detector = detector
            return detector

    def _rescan_size_models(self) -> None:
        previous = {opt.key: opt.var.get() for opt in self._size_model_options}
        self._load_size_models(previous_states=previous)
        self._populate_size_model_toggles()

    def _open_video(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.hevc *.mpg *.avi"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_playback()
        if self.video_capture is not None:
            self.video_capture.release()
        self.video_path = Path(path)
        self.video_capture = cv2.VideoCapture(str(self.video_path))
        if not self.video_capture.isOpened():
            self.status_var.set(f"Failed to open {path}")
            return
        self.status_var.set(f"Loaded {self.video_path.name}")
        self.result_var.set("")
        self.size_result_var.set("")

    def _start_playback(self) -> None:
        if self.video_capture is None or not self.video_capture.isOpened():
            self.status_var.set("No video loaded")
            return
        self._stop_event.clear()
        threading.Thread(target=self._reader_loop, daemon=True).start()
        self._update_display()

    def _stop_playback(self) -> None:
        self._stop_event.set()
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

    def _reader_loop(self) -> None:
        assert self.video_capture is not None
        cap = self.video_capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.pipeline.reset_state()
        self._otsu_pipeline.reset_state()
        self._roi_buffer.clear()
        self._frame_buffer.clear()
        self._approaching = False
        self._last_center_distance = None
        self._last_centroid_x = None
        self._object_present = False
        self._classified_current = False
        frame_idx = 0
        while not self._stop_event.is_set():
            success, frame = cap.read()
            if not success or frame is None:
                break
            processed = self.pipeline.apply(frame)
            try:
                self._frame_queue.put((frame, processed, frame_idx), timeout=0.05)
            except queue.Full:
                frame_idx += 1
                continue
            frame_idx += 1
        self._stop_event.set()

    def _update_display(self) -> None:
        if self._stop_event.is_set():
            return
        try:
            frame, processed, idx = self._frame_queue.get_nowait()
        except queue.Empty:
            self.after(16, self._update_display)
            return
        self._current_frame_index = idx
        self._current_frame = frame
        self._current_processed = processed
        self._current_otsu = self._compute_otsu_mask(frame)

        self._render_current_frame()
        self._maybe_classify(frame, processed)

        self.after(16, self._update_display)

    def _find_main_contour_with_hole(self, roi: np.ndarray) -> Optional[np.ndarray]:
        contours, hierarchy = cv2.findContours(
            roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours or hierarchy is None:
            return None
        hierarchy = hierarchy[0]
        candidates: List[Tuple[float, int]] = []
        for idx, contour in enumerate(contours):
            first_child = hierarchy[idx][2]
            if first_child >= 0:
                area = cv2.contourArea(contour)
                if area > 0:
                    candidates.append((area, idx))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _, best_idx = candidates[0]
        return contours[best_idx]

    def _maybe_classify(self, frame: np.ndarray, mask: np.ndarray) -> None:
        height, width = mask.shape[:2]
        params = self.pipeline.parameters
        roi_left = int(width * params.roi_left_pct)
        roi_right = int(width * (1.0 - params.roi_right_pct))
        roi = mask[:, roi_left:roi_right]
        roi_binary = (roi > 0).astype(np.uint8) * 255
        roi_frame = frame[:, roi_left:roi_right]

        self._roi_buffer.append(roi_binary.copy())
        self._frame_buffer.append(
            (
                self._current_frame_index,
                frame.copy(),
                mask.copy(),
                roi_binary.copy(),
                roi_frame.copy(),
            )
        )

        contour = self._find_main_contour_with_hole(roi_binary)
        if contour is None:
            self._roi_buffer.clear()
            self._frame_buffer.clear()
            self._approaching = False
            self._last_center_distance = None
            self._last_centroid_x = None
            self._object_present = False
            self._classified_current = False
            return
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return
        centroid_x = (moments["m10"] / moments["m00"]) + roi_left
        previous_centroid_x = self._last_centroid_x
        self._last_centroid_x = centroid_x
        frame_center = width / 2.0
        distance = abs(centroid_x - frame_center)

        if not self._object_present:
            # New object entering the scene: reset buffers, tracking state
            # and clear any previous prediction from the UI.
            self._roi_buffer.clear()
            self._frame_buffer.clear()
            self._object_present = True
            self._classified_current = False
            self._last_center_distance = distance
            self._approaching = False
            self.result_var.set("")
            self.size_result_var.set("")
            self.defect_result_var.set("")
            self.warning_var.set("")
            return

        if self._last_center_distance is None:
            self._last_center_distance = distance
            self._approaching = False
            return

        if distance < self._last_center_distance:
            self._approaching = True
        else:
            if (
                self._approaching
                and not self._classified_current
                and (
                    self._last_prediction_time <= 0.0
                    or (time.time() - self._last_prediction_time)
                    > self._prediction_cooldown
                )
            ):
                center_window = max(width * 0.05, 20)
                if distance <= center_window:
                    self._approaching = False
                    self._submit_classification()
                    self._classified_current = True

        if (
            previous_centroid_x is not None
            and not self._classified_current
            and (
                self._last_prediction_time <= 0.0
                or (time.time() - self._last_prediction_time)
                > self._prediction_cooldown
            )
        ):
            trigger_line_pct = getattr(params, "trigger_line_pct", 0.5)
            trigger_band_pct = getattr(params, "trigger_band_pct", 0.05)
            trigger_line_x = width * trigger_line_pct
            band_half_width = max(width * trigger_band_pct, 10.0)
            left_bound = trigger_line_x - band_half_width
            right_bound = trigger_line_x + band_half_width
            segment_min = min(previous_centroid_x, centroid_x)
            segment_max = max(previous_centroid_x, centroid_x)
            if segment_max >= left_bound and segment_min <= right_bound:
                self._approaching = False
                self._submit_classification()
                self._classified_current = True

        self._last_center_distance = distance

    def _toggle_contours(self) -> None:
        self._show_contours = bool(self._contour_var.get())
        if not self._frame_queue.empty():
            try:
                frame, processed, idx = self._frame_queue.get_nowait()
                self._frame_queue.put((frame, processed, idx))
            except queue.Full:
                pass
        self._render_current_frame()

    def _render_current_frame(self) -> None:
        if self._current_frame is None or self._current_processed is None:
            return
        self.raw_canvas.update_image(self._current_frame)
        if self._show_contours:
            display_processed = self._apply_contour_overlay(self._current_processed)
            self.proc_canvas.update_image(display_processed, is_mask=False)
        else:
            self.proc_canvas.update_image(self._current_processed, is_mask=True)
        if self._current_otsu is not None:
            self.otsu_canvas.update_image(self._current_otsu, is_mask=True)

    def _compute_otsu_mask(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        return self._otsu_pipeline.apply(frame)

    def _update_contour_params(self) -> None:
        self._contour_offset = max(0, int(self._contour_offset_var.get()))
        self._contour_thickness = max(1, int(self._contour_thickness_var.get()))

    def _submit_classification(self) -> None:
        if not self._roi_buffer:
            return
        rois_batch = list(self._roi_buffer)
        frames_batch = list(self._frame_buffer)
        self._last_prediction_time = time.time()
        use_defect_detector = bool(self.defect_detector_var.get())

        future = self._classification_executor.submit(
            self._classify_batch,
            rois_batch,
            frames_batch,
            use_defect_detector,
        )
        self._pending_future = future
        future.add_done_callback(lambda fut: self.after(0, self._handle_classification_result, fut.result()))
        self._roi_buffer.clear()
        self._frame_buffer.clear()

    def _run_size_models(
        self,
        family: str,
        rois_batch: List[np.ndarray],
        frames_batch: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> Optional[Dict[str, object]]:
        candidates = [opt for opt in self._size_model_lookup.get(family, []) if opt.var.get()]
        if not candidates:
            return None
        option = candidates[0]
        bundle = option.bundle

        feature_stack: List[np.ndarray] = []
        avg_probs: Optional[np.ndarray] = None

        use_size_pipeline = self._size_pipeline is not None
        otsu_cache: Dict[int, np.ndarray] = {}
        otsu_params = self._otsu_pipeline.parameters

        for idx, roi in enumerate(rois_batch):
            frame_idx, full_frame, _, _, roi_frame = frames_batch[idx]
            if use_size_pipeline:
                size_mask = self._size_pipeline.apply(roi_frame)
            else:
                if frame_idx not in otsu_cache:
                    otsu_cache[frame_idx] = self._otsu_pipeline.apply(full_frame)
                size_mask_full = otsu_cache.get(frame_idx)
                if size_mask_full is None:
                    size_mask = roi
                else:
                    height, width = size_mask_full.shape[:2]
                    left = int(width * otsu_params.roi_left_pct)
                    right = int(width * (1.0 - otsu_params.roi_right_pct))
                    left = max(0, min(left, width - 1))
                    right = max(left + 1, min(width, right))
                    if right <= left:
                        size_mask = size_mask_full
                    else:
                        size_mask = size_mask_full[:, left:right]

            fv = extract_size_features(size_mask)
            ordered = bundle.transform(fv)
            feature_stack.append(ordered)
            scaled = bundle.scaler.transform(ordered.reshape(1, -1))
            probs = bundle.model.predict_proba(scaled)[0]
            if avg_probs is None:
                avg_probs = probs.copy()
            else:
                avg_probs += probs

        if not feature_stack or avg_probs is None:
            return None

        avg_probs /= len(feature_stack)
        best_idx = int(np.argmax(avg_probs))
        size_label = bundle.size_labels[best_idx]
        confidence = float(avg_probs[best_idx])

        avg_vector = np.mean(np.stack(feature_stack, axis=0), axis=0)
        print("=== Size Features (avg of last 4) ===")
        for name, val in zip(bundle.feature_names, avg_vector.tolist()):
            print(f"{name:>26}: {val:+.6f}")
        print("======================\n")

        return {
            "label": str(size_label),
            "confidence": confidence,
            "probabilities": avg_probs.tolist(),
            "family": family,
            "quality": option.quality,
            "option_key": option.key,
            "model_path": str(option.path),
        }

    def _compute_defect_prediction(
        self,
        use_defect_detector: bool,
        family_label: Optional[str],
        size_result: Optional[Dict[str, object]],
        frames_batch: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> Optional[Dict[str, object]]:
        if not use_defect_detector:
            return None
        if not family_label:
            return {"error": "Family prediction unavailable"}
        size_label = None
        if size_result and size_result.get("label") is not None:
            size_label = str(size_result["label"])
        if not size_label:
            return {"error": "Size prediction unavailable"}
        if not frames_batch:
            return {"error": "No frame data"}
        _, last_frame, *_ = frames_batch[-1]
        if last_frame is None or last_frame.size == 0:
            return {"error": "Empty frame"}
        detector = self._ensure_defect_detector()
        if detector is None:
            return {"error": "Defect detector unavailable"}
        try:
            detection = predict_image(
                last_frame,
                piece_type=str(family_label),
                size=size_label,
                detector=detector,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[WARN] Defect detector failed: {exc}")
            return {"error": str(exc)}
        detection["frame_index"] = frames_batch[-1][0]
        return detection

    def _classify_batch(
        self,
        rois_batch: List[np.ndarray],
        frames_batch: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        use_defect_detector: bool = False,
    ) -> Dict[str, object]:
        touching = self._is_touching_border(frames_batch[-1][2])

        if not self.main_model_var.get():
            return {"label": None, "confidence": None, "touching_border": touching, "size": None}

        features_batch: List[np.ndarray] = []
        for roi in rois_batch:
            fv = extract_features_from_mask(roi)
            values = fv.values
            selected = values[list(self._feature_indices)]
            features_batch.append(selected)

        names = self.model_bundle.feature_names or FEATURE_NAMES
        avg_features = np.mean(np.stack(features_batch, axis=0), axis=0)
        print("\n=== Feature Vector (avg of last 4) ===")
        for name, val in zip(names, avg_features.tolist()):
            print(f"{name:>20}: {val:+.6f}")
        print("======================\n")

        avg_probs: Optional[np.ndarray] = None
        for feats in features_batch:
            feats = np.asarray(feats).reshape(1, -1)
            scaled = self.model_bundle.scaler.transform(feats)
            probs = self.model_bundle.model.predict_proba(scaled)[0]
            if avg_probs is None:
                avg_probs = probs.copy()
            else:
                avg_probs += probs

        assert avg_probs is not None
        avg_probs /= len(features_batch)

        best_idx = int(np.argmax(avg_probs))
        label = str(self.model_bundle.classes_[best_idx])
        confidence = float(avg_probs[best_idx])
        print("Prediction:", label, "Confidence:", f"{confidence:.4f}")
        print("======================\n")

        size_result = self._run_size_models(label, rois_batch, frames_batch)
        defect_result = self._compute_defect_prediction(
            use_defect_detector,
            label,
            size_result,
            frames_batch,
        )

        return {
            "label": label,
            "confidence": confidence,
            "touching_border": touching,
            "size": size_result,
            "defect": defect_result,
        }

    def _is_touching_border(self, mask: np.ndarray) -> bool:
        if mask is None or mask.size == 0:
            return False
        rows, cols = mask.shape[:2]
        if np.any(mask[0, :] > 0) or np.any(mask[rows - 1, :] > 0):
            return True
        if np.any(mask[:, 0] > 0) or np.any(mask[:, cols - 1] > 0):
            return True
        return False

    def _handle_classification_result(self, result: Dict[str, object]) -> None:
        if not result:
            return
        label = result.get("label")
        confidence = result.get("confidence")
        if label:
            label_text = str(label).upper()
            if isinstance(confidence, (float, int)):
                label_text = f"{label_text} ({confidence:.2f})"
            self.result_var.set(label_text)
        else:
            self.result_var.set("")

        size_info = result.get("size") if isinstance(result, dict) else None
        if isinstance(size_info, dict):
            size_label = size_info.get("label")
            size_conf = size_info.get("confidence")
            quality = size_info.get("quality")
            if size_label is not None and size_conf is not None:
                text = f"Size {size_label} ({float(size_conf):.2f})"
                if quality:
                    text += f" [{quality}]"
                self.size_result_var.set(text)
            else:
                self.size_result_var.set("")
        else:
            self.size_result_var.set("")

        self._update_defect_display(result.get("defect"))

        if result.get("touching_border"):
            self.warning_var.set("WARNING: Object touching border")
        else:
            self.warning_var.set("")

    def _update_defect_display(self, defect_info: Optional[Dict[str, object]]) -> None:
        if self._defect_label_widget is None:
            return
        if isinstance(defect_info, dict):
            error_message = defect_info.get("error")
            if error_message:
                self.defect_result_var.set(f"Defect detector: {error_message}")
                self._defect_label_widget.config(fg="#C77C04")
                return
            label = defect_info.get("label")
            confidence = defect_info.get("confidence")
            if label:
                text = f"Defect: {label}"
                if isinstance(confidence, (float, int)):
                    text += f" ({float(confidence):.2f})"
                features = defect_info.get("features")
                if isinstance(features, dict):
                    text += f" · {len(features)} feats"
                normalized_label = str(label).strip().upper()
                color = "#14833B" if normalized_label == "BUENO" else "#C62828"
                self._defect_label_widget.config(fg=color)
                self.defect_result_var.set(text)
                return
        self.defect_result_var.set("")
        self._defect_label_widget.config(fg=self._defect_label_neutral_fg)

    def _on_close(self) -> None:
        self._stop_playback()
        if self.video_capture is not None:
            self.video_capture.release()
        self._classification_executor.shutdown(wait=False)
        self.destroy()

    def _apply_contour_overlay(self, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.size == 0:
            return mask

        if mask.ndim == 2:
            gray = mask.astype(np.uint8)
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            color = mask.copy()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        binary = (gray > 0).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None or len(contours) == 0:
            return color

        hole_mask = np.zeros_like(binary)
        for idx, contour in enumerate(contours):
            parent = hierarchy[0][idx][3]
            if parent >= 0:
                cv2.drawContours(hole_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if not np.any(hole_mask):
            return color

        offset = max(0, self._contour_offset)
        if offset > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (offset * 2 + 1, offset * 2 + 1)
            )
            hole_mask = cv2.erode(hole_mask, kernel, iterations=1)

        hole_contours, _ = cv2.findContours(
            hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if hole_contours:
            cv2.drawContours(
                color, hole_contours, -1, (0, 255, 0), max(1, self._contour_thickness)
            )

        return color


class VideoCanvas(tk.Frame):
    def __init__(
        self, master: tk.Widget, title: str, width: int = 480, height: int = 270
    ) -> None:
        super().__init__(master)
        self._width = width
        self._height = height
        self._aspect_ratio = width / height if height else 1.7777778
        tk.Label(self, text=title, font=("Helvetica", 12, "bold")).pack()
        self._canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            highlightthickness=1,
            highlightbackground="#444",
        )
        self._canvas.pack()
        self._photo: Optional[ImageTk.PhotoImage] = None

    def update_image(self, frame: np.ndarray, is_mask: bool = False) -> None:
        if frame is None or frame.size == 0:
            return
        frame_height, frame_width = frame.shape[:2]
        if frame_height == 0 or frame_width == 0:
            return
        frame_aspect = frame_width / frame_height
        if frame_aspect > self._aspect_ratio:
            target_width = self._width
            target_height = max(1, int(round(self._width / frame_aspect)))
        else:
            target_height = self._height
            target_width = max(1, int(round(self._height * frame_aspect)))
        if frame.ndim == 2:
            image = Image.fromarray(frame)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
        image = image.resize(
            (target_width, target_height), Image.NEAREST if is_mask else Image.BILINEAR
        )
        self._photo = ImageTk.PhotoImage(image=image)
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, self._width, self._height, fill="#000000", outline="")
        self._canvas.create_image(
            self._width // 2, self._height // 2, image=self._photo
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive classification viewer for herrajes"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model.joblib (defaults to latest in results/)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET_NAME,
        help="Preset name in config/filter_presets.json",
    )
    parser.add_argument(
        "--size-preset",
        type=str,
        default=None,
        help="Optional preset to run size submodels (defaults to main preset)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size_preset = args.size_preset or args.preset
    app = VideoClassifierApp(model_path=args.model, preset_name=args.preset, size_preset_name=size_preset)
    app.mainloop()


if __name__ == "__main__":
    main()
