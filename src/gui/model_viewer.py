"""GUI for live classification of herraje masks using the trained MLP."""

from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import argparse

import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

from ..features.extractor import FEATURE_NAMES, extract_features_from_mask
from ..processing.pipeline import FilterParameters, FilterPipeline

RESULTS_DIR = Path("results")
DEFAULT_PRESET_NAME = "PHANSALKAR"


@dataclass
class ModelBundle:
    model: object
    scaler: object
    feature_names: Tuple[str, ...]
    classes_: np.ndarray


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
    def __init__(self, model_path: Optional[Path], preset_name: Optional[str]) -> None:
        super().__init__()
        self.title("Herrajes Classifier")
        self.geometry("1400x720")

        self.model_bundle = load_model(model_path)
        params = load_preset(preset_name)

        self._feature_indices = self._compute_feature_indices()

        self.pipeline = FilterPipeline()
        self.pipeline.set_parameters(params)

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[Path] = None
        self._stop_event = threading.Event()
        self._frame_queue: "queue.Queue[Tuple[np.ndarray, np.ndarray]]" = queue.Queue(
            maxsize=5
        )

        self._classification_lock = threading.Lock()
        self._last_prediction_time = 0.0
        self._prediction_cooldown = 0.5
        self._last_center_distance = None
        self._approaching = False
        self._show_contours = False
        self._contour_var = tk.BooleanVar(value=False)
        self._contour_offset = 2
        self._contour_thickness = 2
        self._contour_offset_var = tk.IntVar(value=self._contour_offset)
        self._contour_thickness_var = tk.IntVar(value=self._contour_thickness)
        self._object_present = False
        self._classified_current = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        control_bar = tk.Frame(self)
        control_bar.pack(fill="x", padx=10, pady=10)

        tk.Button(control_bar, text="Open", command=self._open_video).pack(
            side="left", padx=4
        )
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

        self.result_var = tk.StringVar(value="")
        result_label = tk.Label(
            self,
            textvariable=self.result_var,
            font=("Helvetica", 36, "bold"),
            fg="#008000",
        )
        result_label.pack(fill="x", pady=10)

        self._update_contour_params()

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
        while not self._stop_event.is_set():
            success, frame = cap.read()
            if not success or frame is None:
                break
            processed = self.pipeline.apply(frame)
            try:
                self._frame_queue.put((frame, processed), timeout=0.05)
            except queue.Full:
                continue
        self._stop_event.set()

    def _update_display(self) -> None:
        if self._stop_event.is_set():
            return
        try:
            frame, processed = self._frame_queue.get_nowait()
        except queue.Empty:
            self.after(16, self._update_display)
            return

        self.raw_canvas.update_image(frame)
        if self._show_contours:
            display_processed = self._apply_contour_overlay(processed)
            self.proc_canvas.update_image(display_processed, is_mask=False)
        else:
            self.proc_canvas.update_image(processed, is_mask=True)
        self._maybe_classify(processed)

        self.after(16, self._update_display)

    def _maybe_classify(self, mask: np.ndarray) -> None:
        height, width = mask.shape[:2]
        roi_left = int(width * self.pipeline.parameters.roi_left_pct)
        roi_right = int(width * (1.0 - self.pipeline.parameters.roi_right_pct))
        roi = mask[:, roi_left:roi_right]
        contours, hierarchy = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            self._approaching = False
            self._last_center_distance = None
            self._object_present = False
            self._classified_current = False
            return
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return
        centroid_x = (moments["m10"] / moments["m00"]) + roi_left
        frame_center = width / 2.0
        distance = abs(centroid_x - frame_center)

        if not self._object_present:
            self._object_present = True
            self._classified_current = False
            self._last_center_distance = distance
            self._approaching = False
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
                and (time.time() - self._last_prediction_time)
                > self._prediction_cooldown
            ):
                center_window = max(width * 0.05, 20)
                if distance <= center_window:
                    self._approaching = False
                    self._classify_current_frame(roi)
                    self._classified_current = True
        self._last_center_distance = distance

    def _toggle_contours(self) -> None:
        self._show_contours = bool(self._contour_var.get())
        if not self._frame_queue.empty():
            try:
                frame, processed = self._frame_queue.get_nowait()
                self._frame_queue.put((frame, processed))
            except queue.Full:
                pass
        self._render_current_frame()

    def _update_contour_params(self) -> None:
        self._contour_offset = max(0, int(self._contour_offset_var.get()))
        self._contour_thickness = max(1, int(self._contour_thickness_var.get()))

    def _classify_current_frame(self, roi_mask: np.ndarray) -> None:
        with self._classification_lock:
            self._last_prediction_time = time.time()
            roi_binary = (roi_mask > 0).astype(np.uint8) * 255
            fv = extract_features_from_mask(roi_binary)
            values = fv.values
            if len(self._feature_indices) != len(self.model_bundle.feature_names):
                raise ValueError("Feature dimensionality mismatch between model and extractor")
            selected = values[list(self._feature_indices)]
            features = selected.reshape(1, -1)
            scaled = self.model_bundle.scaler.transform(features)
            probs = self.model_bundle.model.predict_proba(scaled)[0]
            best_idx = int(np.argmax(probs))
            label = self.model_bundle.classes_[best_idx]
            confidence = float(probs[best_idx])
            self.result_var.set(f"{label.upper()} ({confidence:.2f})")

    def _on_close(self) -> None:
        self._stop_playback()
        if self.video_capture is not None:
            self.video_capture.release()
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
        self, master: tk.Widget, title: str, width: int = 640, height: int = 480
    ) -> None:
        super().__init__(master)
        self._width = width
        self._height = height
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
        if frame.ndim == 2:
            image = Image.fromarray(frame)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
        image = image.resize(
            (self._width, self._height), Image.NEAREST if is_mask else Image.BILINEAR
        )
        self._photo = ImageTk.PhotoImage(image=image)
        self._canvas.delete("all")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = VideoClassifierApp(model_path=args.model, preset_name=args.preset)
    app.mainloop()


if __name__ == "__main__":
    main()
