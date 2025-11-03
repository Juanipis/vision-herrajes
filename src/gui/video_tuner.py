"""Tkinter GUI application for tuning video binarisation filters."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import Dict, Optional

import cv2
import numpy as np

from ..io.video_loader import VideoLoader, VideoLoaderError
from ..processing.pipeline import FilterParameters, FilterPipeline
from .widgets import FrameDisplay, ParameterPanel

PRESETS_PATH = Path("config/filter_presets.json")


class VideoTunerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Herrajes Filter Tuner")
        self.geometry("1400x900")

        self.loader = VideoLoader()
        self.pipeline = FilterPipeline()
        self._presets: Dict[str, Dict[str, object]] = {}
        self._active_preset: Optional[str] = None

        self._current_frame_index = 0
        self._playback_job: Optional[str] = None
        self._param_job: Optional[str] = None
        self._pending_params: Optional[FilterParameters] = None
        self._scrubbing = False
        self._ignore_scale_events = False
        self._show_contours = False
        self._contour_var = tk.BooleanVar(value=False)
        self._contour_offset = 2
        self._contour_thickness = 2

        self._build_ui()
        self._load_presets()

    # UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        container = tk.Frame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        controls = tk.Frame(container)
        controls.pack(fill="x", pady=(0, 6))

        tk.Button(controls, text="Open", command=self._open_video_dialog).pack(side="left", padx=2)
        tk.Button(controls, text="Play", command=self.play).pack(side="left", padx=2)
        tk.Button(controls, text="Pause", command=self.pause).pack(side="left", padx=2)
        tk.Button(controls, text="Prev", command=lambda: self.seek_relative(-1)).pack(side="left", padx=2)
        tk.Button(controls, text="Next", command=lambda: self.seek_relative(1)).pack(side="left", padx=2)

        debug_controls = tk.Frame(container)
        debug_controls.pack(fill="x", pady=(0, 6))
        tk.Checkbutton(
            debug_controls,
            text="Show contour overlay",
            variable=self._contour_var,
            command=self._toggle_contours,
        ).pack(side="left")

        tk.Label(debug_controls, text="Offset").pack(side="left", padx=(12, 2))
        self._contour_offset_var = tk.IntVar(value=self._contour_offset)
        tk.Scale(
            debug_controls,
            from_=0,
            to=15,
            orient=tk.HORIZONTAL,
            showvalue=True,
            variable=self._contour_offset_var,
            command=lambda _: self._update_contour_params(),
            length=140,
        ).pack(side="left")

        tk.Label(debug_controls, text="Thickness").pack(side="left", padx=(12, 2))
        self._contour_thickness_var = tk.IntVar(value=self._contour_thickness)
        tk.Scale(
            debug_controls,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            showvalue=True,
            variable=self._contour_thickness_var,
            command=lambda _: self._update_contour_params(),
            length=140,
        ).pack(side="left")

        self._status_header = "No video loaded"
        self.status_var = tk.StringVar(value=self._status_header)
        status_label = tk.Label(container, textvariable=self.status_var, anchor="w")
        status_label.pack(fill="x", side="bottom", pady=(6, 0))

        paned = tk.PanedWindow(container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        paned.pack(fill="both", expand=True)

        frames_container = tk.Frame(paned)
        frames_container.pack(fill="both", expand=True)

        self.original_view = FrameDisplay(frames_container, "Original frame")
        self.original_view.pack(pady=(0, 8))

        self.processed_view = FrameDisplay(frames_container, "Processed mask")
        self.processed_view.pack()

        paned.add(frames_container, stretch="always")

        self.parameter_panel = ParameterPanel(
            paned,
            on_parameters_changed=self._on_parameter_changed,
            on_save_preset=self._on_save_preset,
            on_preset_selected=self._apply_preset_by_name,
        )
        paned.add(self.parameter_panel, stretch="never")

        self.timeline_scale = tk.Scale(
            container,
            orient=tk.HORIZONTAL,
            from_=0,
            to=1,
            showvalue=False,
            command=self._on_slider_command,
            state="disabled",
        )
        self.timeline_scale.pack(fill="x", pady=(6, 0))
        self.timeline_scale.bind("<ButtonPress-1>", self._on_slider_pressed)
        self.timeline_scale.bind("<ButtonRelease-1>", self._on_slider_released)

    # Presets -------------------------------------------------------------

    def _load_presets(self) -> None:
        if PRESETS_PATH.exists():
            try:
                with PRESETS_PATH.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    self._presets = data
            except (json.JSONDecodeError, OSError) as exc:
                messagebox.showwarning("Preset error", f"Failed to read presets: {exc}")
                self._presets = {}
        if "default" in self._presets:
            params = FilterParameters.from_dict(self._presets["default"])
            self.pipeline.set_parameters(params)
            self.parameter_panel.set_parameters(params)
            self._active_preset = "default"
        self.parameter_panel.set_presets(sorted(self._presets.keys()), self._active_preset)

    def _save_presets_to_disk(self) -> None:
        PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PRESETS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(self._presets, handle, indent=2)

    def _on_save_preset(self) -> None:
        current_name = self.parameter_panel.current_preset() or self._active_preset or ""
        name = simpledialog.askstring("Save preset", "Preset name:", initialvalue=current_name, parent=self)
        if not name:
            return
        params = self.parameter_panel.parameters()
        self._presets[name] = params.to_dict()
        self._active_preset = name
        self.parameter_panel.set_presets(sorted(self._presets.keys()), name)
        self._save_presets_to_disk()

    def _apply_preset_by_name(self, name: str) -> None:
        if not name:
            return
        preset = self._presets.get(name)
        if preset is None:
            messagebox.showwarning("Preset missing", f"Preset '{name}' was not found.")
            return
        params = FilterParameters.from_dict(preset)
        self._active_preset = name
        self.pipeline.set_parameters(params)
        self.parameter_panel.set_parameters(params)
        self._render_current_frame()

    # Video interaction ---------------------------------------------------

    def _open_video_dialog(self) -> None:
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.hevc *.mpg *.avi"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            properties = self.loader.open(file_path)
        except VideoLoaderError as exc:
            messagebox.showerror("Video error", str(exc), parent=self)
            return
        self.pipeline.reset_state()
        self._current_frame_index = 0
        self._set_timeline_range(properties.frame_count)
        self._set_timeline_value(0, emit=False)
        self._status_header = (
            f"Loaded: {properties.path.name} | {properties.frame_count} frames @ {properties.fps:.2f} FPS"
        )
        self.status_var.set(self._status_header)
        self._render_current_frame()

    def play(self) -> None:
        if self._playback_job is not None:
            return
        try:
            fps = self.loader.properties.fps
        except VideoLoaderError:
            return
        interval = int(max(1, round(1000.0 / max(1.0, fps))))
        self._playback_job = self.after(interval, self._advance_frame)

    def pause(self) -> None:
        if self._playback_job is not None:
            self.after_cancel(self._playback_job)
            self._playback_job = None

    def seek_relative(self, offset: int) -> None:
        try:
            frame_count = self.loader.properties.frame_count
        except VideoLoaderError:
            return
        new_index = max(0, min(frame_count - 1, self._current_frame_index + offset))
        self._current_frame_index = new_index
        self._set_timeline_value(new_index, emit=False)
        self._render_current_frame()

    def _advance_frame(self) -> None:
        try:
            frame_count = self.loader.properties.frame_count
        except VideoLoaderError:
            self.pause()
            return
        self._current_frame_index += 1
        if self._current_frame_index >= frame_count:
            self._current_frame_index = frame_count - 1
            self.pause()
            return
        self._set_timeline_value(self._current_frame_index, emit=False)
        self._render_current_frame()
        self._playback_job = None
        self.play()

    # Timeline slider ----------------------------------------------------

    def _set_timeline_range(self, frame_count: int) -> None:
        if frame_count <= 0:
            self.timeline_scale.configure(state="disabled", from_=0, to=1)
            return
        self.timeline_scale.configure(state="normal", from_=0, to=max(0, frame_count - 1))

    def _set_timeline_value(self, value: int, emit: bool = True) -> None:
        self._ignore_scale_events = True
        self.timeline_scale.set(value)
        self._ignore_scale_events = False
        if emit:
            self._on_slider_command(str(value))

    def _on_slider_command(self, value: str) -> None:
        if self._ignore_scale_events:
            return
        index = int(float(value))
        self._current_frame_index = index
        if self._scrubbing:
            self._render_current_frame()

    def _on_slider_pressed(self, _event) -> None:
        self._scrubbing = True
        self.pause()

    def _on_slider_released(self, _event) -> None:
        self._scrubbing = False
        self._current_frame_index = int(self.timeline_scale.get())
        self._render_current_frame()

    # Rendering -----------------------------------------------------------

    def _render_current_frame(self) -> None:
        try:
            frame = self.loader.read_frame(self._current_frame_index)
        except VideoLoaderError as exc:
            self.status_var.set(f"Frame error: {exc}")
            return
        params = self.pipeline.parameters
        processed = self.pipeline.apply(frame)
        display_processed = self._apply_contour_overlay(processed) if self._show_contours else processed
        cropped_frame = self._crop_to_roi(frame, params)
        cropped_processed = self._crop_to_roi(display_processed, params)
        self.original_view.update_image(cropped_frame)
        self.processed_view.update_image(cropped_processed)
        try:
            frame_count = self.loader.properties.frame_count
        except VideoLoaderError:
            frame_count = 0
        self.status_var.set(
            f"{self._status_header} | Frame {self._current_frame_index + 1}/{frame_count}"
        )

    # Parameter handling -------------------------------------------------

    def _on_parameter_changed(self, params: FilterParameters) -> None:
        self._pending_params = params
        if self._param_job is not None:
            self.after_cancel(self._param_job)
        self._param_job = self.after(150, self._apply_pending_parameters)

    def _apply_pending_parameters(self) -> None:
        if self._pending_params is None:
            return
        self.pipeline.set_parameters(self._pending_params)
        self._pending_params = None
        self._param_job = None
        self._render_current_frame()

    def _toggle_contours(self) -> None:
        self._show_contours = bool(self._contour_var.get())
        self._render_current_frame()

    def _update_contour_params(self) -> None:
        self._contour_offset = max(0, int(self._contour_offset_var.get()))
        self._contour_thickness = max(1, int(self._contour_thickness_var.get()))
        if self._show_contours:
            self._render_current_frame()

    def _crop_to_roi(self, image: np.ndarray, params: FilterParameters) -> np.ndarray:
        if image is None:
            return image
        height, width = image.shape[:2]
        left = int(width * params.roi_left_pct)
        right = int(width * (1.0 - params.roi_right_pct))
        if right <= left:
            return image
        return image[:, left:right]

    def _apply_contour_overlay(self, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.size == 0:
            return mask
        if mask.ndim == 2:
            gray = mask
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            color = mask.copy()
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        offset = max(0, self._contour_offset)
        if offset > 0:
            kernel_size = offset * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            eroded = cv2.erode(gray, kernel, iterations=1)
            if np.count_nonzero(eroded) > 0:
                gray = eroded
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return color

        hole_mask = np.zeros_like(gray)
        for idx, contour in enumerate(contours):
            parent = hierarchy[0][idx][3]
            if parent >= 0:  # internal contour
                cv2.drawContours(hole_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if not np.any(hole_mask):
            return color

        offset = max(0, self._contour_offset)
        if offset > 0:
            kernel_size = offset * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            hole_mask = cv2.erode(hole_mask, kernel, iterations=1)

        hole_contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hole_contours:
            cv2.drawContours(color, hole_contours, -1, (0, 255, 0), self._contour_thickness)
        return color


def run() -> None:
    app = VideoTunerApp()
    app.mainloop()


if __name__ == "__main__":
    run()
