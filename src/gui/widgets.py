"""Tkinter-based widgets for the video filter tuning GUI."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..processing.pipeline import FilterParameters

if hasattr(Image, "Resampling"):
    _RESAMPLE = Image.Resampling.LANCZOS
else:  # pragma: no cover - Pillow<9 compatibility
    _RESAMPLE = Image.LANCZOS


class FrameDisplay(ttk.Frame):
    """Widget that renders video frames using a Tkinter canvas."""

    def __init__(self, master: tk.Widget, title: str, max_width: int = 480, max_height: int = 320) -> None:
        super().__init__(master)
        self._max_width = max_width
        self._max_height = max_height
        self.pack_propagate(False)
        self.configure(width=max_width + 16, height=max_height + 48)

        title_label = ttk.Label(self, text=title, anchor="center", font=("Helvetica", 12, "bold"))
        title_label.pack(fill="x", pady=(0, 4))

        self._canvas = tk.Canvas(
            self,
            width=max_width,
            height=max_height,
            highlightthickness=1,
            highlightbackground="#444",
        )
        self._canvas.pack()
        self._photo: Optional[ImageTk.PhotoImage] = None

    def update_image(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            self.clear()
            return
        if frame.ndim == 2:
            image = Image.fromarray(frame)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
        image.thumbnail((self._max_width, self._max_height), _RESAMPLE)
        self._photo = ImageTk.PhotoImage(image=image)
        self._canvas.delete("all")
        self._canvas.create_image(self._max_width // 2, self._max_height // 2, image=self._photo)

    def clear(self) -> None:
        self._photo = None
        self._canvas.delete("all")


class SliderField(ttk.Frame):
    """Reusable slider that emits integer or float values to a callback."""

    def __init__(
        self,
        master: tk.Widget,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        default: float,
        integer: bool,
        command: Callable[[float], None],
    ) -> None:
        super().__init__(master)
        if maximum <= minimum:
            raise ValueError("Slider maximum must be greater than minimum")
        if step <= 0:
            raise ValueError("Slider step must be positive")

        self._integer = integer
        self._command = command
        self._suspend = False

        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 2))
        self._label = ttk.Label(header, text=label)
        self._label.pack(side="left")
        self._value_label = ttk.Label(header, text="")
        self._value_label.pack(side="right")

        self._scale = tk.Scale(
            self,
            from_=minimum,
            to=maximum,
            orient=tk.HORIZONTAL,
            resolution=step,
            showvalue=False,
            command=self._on_scale_changed,
        )
        self._scale.pack(fill="x")
        self.set_value(default, emit=False)

    def value(self) -> float:
        raw = float(self._scale.get())
        if self._integer:
            return int(round(raw))
        return raw

    def set_value(self, value: float, emit: bool = True) -> None:
        self._suspend = True
        self._scale.set(value)
        self._update_label(value)
        self._suspend = False
        if emit:
            self._command(self.value())

    def _on_scale_changed(self, value: str) -> None:
        numeric = float(value)
        self._update_label(numeric)
        if self._suspend:
            return
        self._command(self.value())

    def _update_label(self, value: float) -> None:
        if self._integer:
            display = str(int(round(value)))
        else:
            display = f"{value:.2f}"
        self._value_label.configure(text=display)

    def set_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self._scale.configure(state=state)
        foreground = "#1a1a1a" if enabled else "#7a7a7a"
        self._label.configure(foreground=foreground)
        self._value_label.configure(foreground=foreground)


class ParameterPanel(ttk.Frame):
    """Panel that allows configuring binarisation parameters via Tk controls."""

    METHOD_OPTIONS = [
        ("Manual threshold", "manual"),
        ("Otsu (global)", "otsu"),
        ("Adaptive mean", "adaptive_mean"),
        ("Adaptive Gaussian", "adaptive_gaussian"),
        ("Niblack", "niblack"),
        ("Sauvola", "sauvola"),
        ("Wolf", "wolf"),
        ("Nick", "nick"),
        ("Phansalkar", "phansalkar"),
    ]

    def __init__(
        self,
        master: tk.Widget,
        on_parameters_changed: Callable[[FilterParameters], None],
        on_save_preset: Callable[[], None],
        on_preset_selected: Callable[[str], None],
    ) -> None:
        super().__init__(master, padding=12)
        self._on_parameters_changed = on_parameters_changed
        self._on_save_preset = on_save_preset
        self._on_preset_selected = on_preset_selected
        self._updating = False

        self._params = FilterParameters()
        self._sliders: dict[str, SliderField] = {}

        self._method_map = {display: value for display, value in self.METHOD_OPTIONS}
        self._display_map = {value: display for display, value in self.METHOD_OPTIONS}
        self._method_var = tk.StringVar(value=self._params.threshold_method)

        self._build_ui()
        self._update_method_controls()
        self._notify_change()

    # UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        title = ttk.Label(
            self,
            text="Binarisation Parameters",
            anchor="center",
            font=("Helvetica", 13, "bold"),
        )
        title.pack(fill="x", pady=(0, 10))

        preset_row = ttk.Frame(self)
        preset_row.pack(fill="x", pady=(0, 6))
        ttk.Label(preset_row, text="Preset:").pack(side="left")
        self._preset_combo = ttk.Combobox(preset_row, state="readonly")
        self._preset_combo.pack(side="left", fill="x", expand=True, padx=4)
        self._preset_combo.bind("<<ComboboxSelected>>", self._on_preset_combo)

        ttk.Button(self, text="Save preset", command=self._on_save_preset).pack(fill="x", pady=(0, 12))

        method_row = ttk.Frame(self)
        method_row.pack(fill="x", pady=(0, 8))
        ttk.Label(method_row, text="Threshold method:").pack(side="left")
        self._method_combo = ttk.Combobox(method_row, state="readonly")
        self._method_combo.configure(values=list(self._method_map.keys()))
        self._method_combo.pack(side="left", fill="x", expand=True, padx=4)
        display_name = self._display_map.get(self._params.threshold_method, self.METHOD_OPTIONS[0][0])
        self._method_combo.set(display_name)
        self._method_combo.bind("<<ComboboxSelected>>", self._on_method_selected)

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True)

        left_frame = ttk.Frame(columns)
        right_frame = ttk.Frame(columns)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self._sliders["manual_threshold"] = SliderField(
            left_frame,
            label="Manual threshold",
            minimum=0,
            maximum=255,
            step=1,
            default=self._params.manual_threshold,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["manual_threshold"].pack(fill="x", pady=4)

        self._sliders["pre_blur_kernel"] = SliderField(
            left_frame,
            label="Pre-blur kernel",
            minimum=0,
            maximum=31,
            step=2,
            default=self._params.pre_blur_kernel,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["pre_blur_kernel"].pack(fill="x", pady=4)

        self._sliders["block_size"] = SliderField(
            left_frame,
            label="Block size (odd)",
            minimum=3,
            maximum=99,
            step=2,
            default=self._params.block_size,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["block_size"].pack(fill="x", pady=4)

        self._sliders["constant_C"] = SliderField(
            left_frame,
            label="Constant C",
            minimum=-25,
            maximum=25,
            step=0.5,
            default=self._params.constant_C,
            integer=False,
            command=lambda _: self._notify_change(),
        )
        self._sliders["constant_C"].pack(fill="x", pady=4)

        ttk.Label(left_frame, text="ROI (percentage trimmed from sides)").pack(anchor="w", pady=(12, 2))
        self._sliders["roi_left_pct"] = SliderField(
            left_frame,
            label="Left ROI (%)",
            minimum=0,
            maximum=40,
            step=1,
            default=self._params.roi_left_pct * 100.0,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["roi_left_pct"].pack(fill="x", pady=4)

        self._sliders["roi_right_pct"] = SliderField(
            left_frame,
            label="Right ROI (%)",
            minimum=0,
            maximum=40,
            step=1,
            default=self._params.roi_right_pct * 100.0,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["roi_right_pct"].pack(fill="x", pady=4)

        self._invert_var = tk.BooleanVar(value=self._params.invert_output)
        ttk.Checkbutton(
            left_frame,
            text="Invert mask",
            variable=self._invert_var,
            command=self._notify_change,
        ).pack(anchor="w", pady=(12, 0))

        # Right column: advanced method parameters
        self._sliders["niblack_k"] = SliderField(
            right_frame,
            label="Niblack k",
            minimum=-1.0,
            maximum=1.0,
            step=0.01,
            default=self._params.niblack_k,
            integer=False,
            command=lambda _: self._notify_change(),
        )
        self._sliders["niblack_k"].pack(fill="x", pady=4)

        self._sliders["sauvola_k"] = SliderField(
            right_frame,
            label="Sauvola k",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            default=self._params.sauvola_k,
            integer=False,
            command=lambda _: self._notify_change(),
        )
        self._sliders["sauvola_k"].pack(fill="x", pady=4)

        self._sliders["sauvola_r"] = SliderField(
            right_frame,
            label="Sauvola R",
            minimum=1,
            maximum=255,
            step=1,
            default=self._params.sauvola_r,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["sauvola_r"].pack(fill="x", pady=4)

        self._sliders["wolf_k"] = SliderField(
            right_frame,
            label="Wolf k",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            default=self._params.wolf_k,
            integer=False,
            command=lambda _: self._notify_change(),
        )
        self._sliders["wolf_k"].pack(fill="x", pady=4)

        self._sliders["nick_k"] = SliderField(
            right_frame,
            label="Nick k",
            minimum=-1.0,
            maximum=1.0,
            step=0.01,
            default=self._params.nick_k,
            integer=False,
            command=lambda _: self._notify_change(),
        )
        self._sliders["nick_k"].pack(fill="x", pady=4)

        self._sliders["phansalkar_k"] = SliderField(
            right_frame,
            label="Phansalkar k",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            default=self._params.phansalkar_k,
            integer=False,
            command=lambda _: self._notify_change(),
        )
        self._sliders["phansalkar_k"].pack(fill="x", pady=4)

        self._sliders["phansalkar_p"] = SliderField(
            right_frame,
            label="Phansalkar p",
            minimum=0.0,
            maximum=10.0,
            step=0.5,
            default=self._params.phansalkar_p,
            integer=False,
            command=lambda _: self._notify_change(),
        )
        self._sliders["phansalkar_p"].pack(fill="x", pady=4)

        self._sliders["phansalkar_q"] = SliderField(
            right_frame,
            label="Phansalkar q",
            minimum=0.0,
            maximum=20.0,
            step=0.5,
            default=self._params.phansalkar_q,
            integer=False,
            command=lambda _: self._notify_change(),
        )
        self._sliders["phansalkar_q"].pack(fill="x", pady=4)

        ttk.Label(right_frame, text="Post-processing", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(12, 4))

        self._expand_var = tk.BooleanVar(value=self._params.expand_mask)
        ttk.Checkbutton(
            right_frame,
            text="Expand mask",
            variable=self._expand_var,
            command=self._notify_change,
        ).pack(anchor="w")

        self._sliders["expand_kernel"] = SliderField(
            right_frame,
            label="Expansion kernel",
            minimum=1,
            maximum=25,
            step=2,
            default=self._params.expand_kernel,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["expand_kernel"].pack(fill="x", pady=4)

        self._sliders["expand_iterations"] = SliderField(
            right_frame,
            label="Expansion iterations",
            minimum=0,
            maximum=5,
            step=1,
            default=self._params.expand_iterations,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["expand_iterations"].pack(fill="x", pady=4)

        self._sliders["min_component_area"] = SliderField(
            right_frame,
            label="Min component area",
            minimum=0,
            maximum=50000,
            step=100,
            default=self._params.min_component_area,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["min_component_area"].pack(fill="x", pady=4)

        self._closing_var = tk.BooleanVar(value=self._params.use_closing)
        ttk.Checkbutton(
            right_frame,
            text="Morphological closing",
            variable=self._closing_var,
            command=self._notify_change,
        ).pack(anchor="w", pady=(8, 0))

        self._sliders["closing_kernel"] = SliderField(
            right_frame,
            label="Closing kernel",
            minimum=0,
            maximum=25,
            step=2,
            default=self._params.closing_kernel,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["closing_kernel"].pack(fill="x", pady=4)

        self._sliders["closing_iterations"] = SliderField(
            right_frame,
            label="Closing iterations",
            minimum=0,
            maximum=5,
            step=1,
            default=self._params.closing_iterations,
            integer=True,
            command=lambda _: self._notify_change(),
        )
        self._sliders["closing_iterations"].pack(fill="x", pady=4)

        ttk.Frame(self).pack(expand=True, fill="both")

    # Public API ---------------------------------------------------------

    def parameters(self) -> FilterParameters:
        method = self._method_var.get()
        roi_left_pct = self._sliders["roi_left_pct"].value() / 100.0
        roi_right_pct = self._sliders["roi_right_pct"].value() / 100.0
        if roi_left_pct + roi_right_pct >= 0.9:
            roi_right_pct = max(0.0, 0.9 - roi_left_pct)
            self._sliders["roi_right_pct"].set_value(roi_right_pct * 100.0, emit=False)
        params = FilterParameters(
            threshold_method=method,
            manual_threshold=int(self._sliders["manual_threshold"].value()),
            pre_blur_kernel=int(self._sliders["pre_blur_kernel"].value()),
            block_size=int(self._sliders["block_size"].value()),
            constant_C=float(self._sliders["constant_C"].value()),
            niblack_k=float(self._sliders["niblack_k"].value()),
            sauvola_k=float(self._sliders["sauvola_k"].value()),
            sauvola_r=float(self._sliders["sauvola_r"].value()),
            wolf_k=float(self._sliders["wolf_k"].value()),
            nick_k=float(self._sliders["nick_k"].value()),
            phansalkar_k=float(self._sliders["phansalkar_k"].value()),
            phansalkar_p=float(self._sliders["phansalkar_p"].value()),
            phansalkar_q=float(self._sliders["phansalkar_q"].value()),
            roi_left_pct=max(0.0, min(0.9, roi_left_pct)),
            roi_right_pct=max(0.0, min(0.9, roi_right_pct)),
            expand_mask=bool(self._expand_var.get()),
            expand_kernel=int(self._sliders["expand_kernel"].value()),
            expand_iterations=int(self._sliders["expand_iterations"].value()),
            closing_kernel=int(self._sliders["closing_kernel"].value()),
            closing_iterations=int(self._sliders["closing_iterations"].value()),
            min_component_area=int(self._sliders["min_component_area"].value()),
            invert_output=bool(self._invert_var.get()),
            use_closing=bool(self._closing_var.get()),
        )
        self._params = params
        return params

    def set_parameters(self, params: FilterParameters) -> None:
        self._updating = True
        self._params = params
        display_name = self._display_map.get(params.threshold_method, self.METHOD_OPTIONS[0][0])
        self._method_combo.set(display_name)
        self._method_var.set(params.threshold_method)
        self._sliders["manual_threshold"].set_value(params.manual_threshold, emit=False)
        self._sliders["pre_blur_kernel"].set_value(params.pre_blur_kernel, emit=False)
        self._sliders["block_size"].set_value(params.block_size, emit=False)
        self._sliders["constant_C"].set_value(params.constant_C, emit=False)
        self._sliders["roi_left_pct"].set_value(params.roi_left_pct * 100.0, emit=False)
        self._sliders["roi_right_pct"].set_value(params.roi_right_pct * 100.0, emit=False)
        self._invert_var.set(params.invert_output)
        self._sliders["niblack_k"].set_value(params.niblack_k, emit=False)
        self._sliders["sauvola_k"].set_value(params.sauvola_k, emit=False)
        self._sliders["sauvola_r"].set_value(params.sauvola_r, emit=False)
        self._sliders["wolf_k"].set_value(params.wolf_k, emit=False)
        self._sliders["nick_k"].set_value(params.nick_k, emit=False)
        self._sliders["phansalkar_k"].set_value(params.phansalkar_k, emit=False)
        self._sliders["phansalkar_p"].set_value(params.phansalkar_p, emit=False)
        self._sliders["phansalkar_q"].set_value(params.phansalkar_q, emit=False)
        self._expand_var.set(params.expand_mask)
        self._sliders["expand_kernel"].set_value(params.expand_kernel, emit=False)
        self._sliders["expand_iterations"].set_value(params.expand_iterations, emit=False)
        self._sliders["min_component_area"].set_value(params.min_component_area, emit=False)
        self._closing_var.set(params.use_closing)
        self._sliders["closing_kernel"].set_value(params.closing_kernel, emit=False)
        self._sliders["closing_iterations"].set_value(params.closing_iterations, emit=False)
        self._updating = False
        self._update_method_controls()
        self._notify_change()

    def set_presets(self, names: Iterable[str], active: Optional[str]) -> None:
        values = list(names)
        self._preset_combo.configure(values=values)
        if active and active in values:
            self._preset_combo.set(active)
        else:
            self._preset_combo.set("")

    def current_preset(self) -> Optional[str]:
        text = self._preset_combo.get().strip()
        return text or None

    # Internal helpers ---------------------------------------------------

    def _notify_change(self) -> None:
        if self._updating:
            return
        params = self.parameters()
        self._update_method_controls()
        self._on_parameters_changed(params)

    def _on_preset_combo(self, event: tk.Event) -> None:  # type: ignore[override]
        selection = self._preset_combo.get()
        if selection:
            self._on_preset_selected(selection)

    def _on_method_selected(self, event: tk.Event) -> None:  # type: ignore[override]
        display = self._method_combo.get()
        method = self._method_map.get(display, self.METHOD_OPTIONS[0][1])
        self._method_var.set(method)
        self._update_method_controls()
        self._notify_change()

    def _update_method_controls(self) -> None:
        method = self._method_var.get()
        relevant: set[str]
        if method == "manual":
            relevant = {"manual_threshold"}
        elif method in {"adaptive_mean", "adaptive_gaussian"}:
            relevant = {"block_size", "constant_C"}
        elif method == "niblack":
            relevant = {"block_size", "niblack_k"}
        elif method == "sauvola":
            relevant = {"block_size", "sauvola_k", "sauvola_r"}
        elif method == "wolf":
            relevant = {"block_size", "wolf_k"}
        elif method == "nick":
            relevant = {"block_size", "nick_k"}
        elif method == "phansalkar":
            relevant = {"block_size", "phansalkar_k", "phansalkar_p", "phansalkar_q"}
        else:  # otsu and fallback
            relevant = set()

        always_enabled = {
            "roi_left_pct",
            "roi_right_pct",
            "pre_blur_kernel",
            "expand_kernel",
            "expand_iterations",
            "closing_kernel",
            "closing_iterations",
            "min_component_area",
        }

        for name, slider in self._sliders.items():
            if name in always_enabled:
                slider.set_enabled(True)
            else:
                slider.set_enabled(name in relevant)

        expand_enabled = bool(self._expand_var.get())
        self._sliders["expand_kernel"].set_enabled(expand_enabled)
        self._sliders["expand_iterations"].set_enabled(expand_enabled)
        closing_enabled = bool(self._closing_var.get())
        self._sliders["closing_kernel"].set_enabled(closing_enabled)
        self._sliders["closing_iterations"].set_enabled(closing_enabled)
