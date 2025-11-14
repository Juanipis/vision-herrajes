"""Final GUI/front-end for live camera classification with custom layout.

This front-end reuses the existing camera classifier and processing pipeline
but arranges the controls and views to match the requested mockup:

- Camera selector and connect/play/stop controls at the top.
- Live camera view.
- Last prediction frame.
- Output panel with family / size / bueno-malo.
- Mode selector with per-family checkboxes.
- Toggles for size detector and bueno/malo detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk

import numpy as np

from .camera_viewer import CameraClassifierApp
from .model_viewer import DEFAULT_PRESET_NAME, VideoCanvas


class FinalFrontApp(CameraClassifierApp):
    """Camera-based classifier UI matching the requested layout."""

    def __init__(
        self,
        model_path: Optional[Path],
        preset_name: Optional[str],
        size_preset_name: Optional[str],
        *,
        max_probe: int = 8,
        initial_camera_index: Optional[int] = None,
    ) -> None:
        # Layout-specific state prepared before base initialization.
        self.mode_var: tk.StringVar
        self._family_check_vars: Dict[str, tk.BooleanVar] = {}
        self.size_detector_enabled_var: tk.BooleanVar
        self._last_prediction_frame = None
        super().__init__(
            model_path=model_path,
            preset_name=preset_name,
            size_preset_name=size_preset_name,
            max_probe=max_probe,
            initial_camera_index=initial_camera_index,
        )

    # ------------------------------------------------------------------
    # Hooks from VideoClassifierApp/CameraClassifierApp
    # ------------------------------------------------------------------
    def _post_setup(self) -> None:
        """Prepare mode selector and per-family checkboxes."""

        super()._post_setup()

        # Family names come from the main classifier bundle.
        classes = getattr(self, "model_bundle", None)
        family_labels: List[str] = []
        if classes is not None and getattr(classes, "classes_", None) is not None:
            family_labels = sorted({str(label) for label in classes.classes_})

        self.mode_var = tk.StringVar(value="GENERAL")
        self._family_check_vars = {
            label: tk.BooleanVar(value=True) for label in family_labels
        }
        self.size_detector_enabled_var = tk.BooleanVar(value=True)

    def _build_ui(self) -> None:
        """Build the custom layout."""

        # Top control bar: camera selector + connect/play/stop
        control_bar = tk.Frame(self)
        control_bar.pack(fill="x", padx=10, pady=10)

        self._build_source_controls(control_bar)
        tk.Button(control_bar, text="Play", command=self._start_playback).pack(
            side="left", padx=4
        )
        tk.Button(control_bar, text="Stop", command=self._stop_playback).pack(
            side="left", padx=4
        )

        # Status line
        self.status_var = tk.StringVar(
            value="Seleccione una cámara y presione Play"
        )
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(
            fill="x", padx=10
        )

        # Main content area
        main = tk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: live camera view
        left_col = tk.Frame(main)
        left_col.pack(side="left", fill="both", expand=True, padx=5)
        self.raw_canvas = VideoCanvas(
            left_col,
            "Live camera view",
            width=640,
            height=360,
        )
        self.raw_canvas.pack(fill="both", expand=True)

        # Center: last prediction frame + output summary
        center_col = tk.Frame(main)
        center_col.pack(side="left", fill="both", expand=True, padx=5)

        self.last_pred_canvas = VideoCanvas(
            center_col,
            "Last prediction frame",
            width=480,
            height=270,
        )
        self.last_pred_canvas.pack(fill="both", expand=True)

        output_frame = tk.LabelFrame(center_col, text="Output predicción")
        output_frame.pack(fill="x", pady=(10, 0))

        self.result_var = tk.StringVar(value="")
        self.size_result_var = tk.StringVar(value="")
        self.defect_result_var = tk.StringVar(value="")
        self.warning_var = tk.StringVar(value="")

        # Familia
        tk.Label(output_frame, text="Familia:", anchor="w").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        tk.Label(
            output_frame,
            textvariable=self.result_var,
            font=("Helvetica", 14, "bold"),
            fg="#008000",
        ).grid(row=0, column=1, sticky="w", padx=4, pady=2)

        # Tamaño
        tk.Label(output_frame, text="Tamaño:", anchor="w").grid(
            row=1, column=0, sticky="w", padx=4, pady=2
        )
        tk.Label(
            output_frame,
            textvariable=self.size_result_var,
            font=("Helvetica", 12),
            fg="#0044AA",
        ).grid(row=1, column=1, sticky="w", padx=4, pady=2)

        # Bueno/Malo
        tk.Label(output_frame, text="Bueno/Malo:", anchor="w").grid(
            row=2, column=0, sticky="w", padx=4, pady=2
        )
        defect_label = tk.Label(
            output_frame,
            textvariable=self.defect_result_var,
            font=("Helvetica", 12, "bold"),
        )
        defect_label.grid(row=2, column=1, sticky="w", padx=4, pady=2)
        self._defect_label_widget = defect_label

        # Warning line
        tk.Label(
            center_col,
            textvariable=self.warning_var,
            font=("Helvetica", 12, "bold"),
            fg="#FF3333",
        ).pack(fill="x", pady=(8, 0))

        # Right: mode selector + detector toggles
        right_col = tk.Frame(main)
        right_col.pack(side="left", fill="y", padx=5)

        self._build_mode_sidebar(right_col)

        # Initialize contour parameters but keep UI compact; no contour toggles.
        self._update_contour_params()

    # ------------------------------------------------------------------
    # Sidebar / controls
    # ------------------------------------------------------------------
    def _build_mode_sidebar(self, parent: tk.Widget) -> None:
        mode_frame = tk.LabelFrame(
            parent, text="Selector de modo", padx=8, pady=8
        )
        mode_frame.pack(fill="x", pady=(0, 10))

        tk.Label(mode_frame, text="Modo:").pack(anchor="w")
        tk.OptionMenu(
            mode_frame,
            self.mode_var,
            "GENERAL",
            "FILTRAR POR FAMILIAS",
        ).pack(fill="x", pady=(2, 6))

        families_container = tk.Frame(mode_frame)
        families_container.pack(fill="both", expand=True)

        if not self._family_check_vars:
            tk.Label(
                families_container,
                text="Sin familias disponibles",
                fg="#777",
            ).pack(anchor="w")
        else:
            tk.Label(
                families_container,
                text="Familias (checkbox):",
                font=("Helvetica", 10, "bold"),
            ).pack(anchor="w")
            for label, var in sorted(
                self._family_check_vars.items(), key=lambda item: item[0]
            ):
                tk.Checkbutton(
                    families_container,
                    text=str(label),
                    variable=var,
                ).pack(anchor="w")

        detectors_frame = tk.LabelFrame(
            parent, text="Detectores", padx=8, pady=8
        )
        detectors_frame.pack(fill="x")

        tk.Checkbutton(
            detectors_frame,
            text="Activar detector de tamaños",
            variable=self.size_detector_enabled_var,
        ).pack(anchor="w")

        tk.Checkbutton(
            detectors_frame,
            text="Activar detector de buenos/malos",
            variable=self.defect_detector_var,
            command=self._handle_defect_toggle,
        ).pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    # Classification customizations
    # ------------------------------------------------------------------
    def _render_current_frame(self) -> None:
        """Update live view and, if present, last prediction frame."""

        if self._current_frame is None:
            return
        self.raw_canvas.update_image(self._current_frame)

        if self._last_prediction_frame is not None:
            self.last_pred_canvas.update_image(self._last_prediction_frame)

    def _submit_classification(self) -> None:
        """Submit a batch for background classification and remember its frame."""

        if not self._roi_buffer:
            return

        rois_batch = list(self._roi_buffer)
        frames_batch = list(self._frame_buffer)
        # Use the last full frame as "last prediction frame" for display.
        if frames_batch:
            _, last_frame, *_ = frames_batch[-1]
            self._last_prediction_frame = last_frame.copy()

        import time as _time

        self._last_prediction_time = _time.time()
        use_defect_detector = bool(self.defect_detector_var.get())

        future = self._classification_executor.submit(
            self._classify_batch,
            rois_batch,
            frames_batch,
            use_defect_detector,
        )
        self._pending_future = future
        future.add_done_callback(
            lambda fut: self.after(
                0, self._handle_classification_result, fut.result()
            )
        )
        self._roi_buffer.clear()
        self._frame_buffer.clear()

    def _run_size_models(
        self,
        family: str,
        rois_batch: List[np.ndarray],
        frames_batch: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ):
        """Guard size models behind a GUI checkbox."""

        if not self.size_detector_enabled_var.get():
            return None
        return super()._run_size_models(family, rois_batch, frames_batch)  # type: ignore[misc]

    def _classify_batch(
        self,
        rois_batch: List[np.ndarray],
        frames_batch: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        use_defect_detector: bool = False,
    ) -> Dict[str, object]:
        """Apply optional family filtering on top of the base classifier."""

        result = super()._classify_batch(  # type: ignore[misc]
            rois_batch,
            frames_batch,
            use_defect_detector=use_defect_detector,
        )

        label = result.get("label")
        if not label:
            return result

        mode = self.mode_var.get().upper()
        if mode.startswith("GENERAL"):
            return result

        family_label = str(label)
        var = self._family_check_vars.get(family_label)
        if var is not None and not var.get():
            # Mark prediction as filtered-out; keep other diagnostics intact.
            result = dict(result)
            result["filtered_out"] = True
            result["label"] = None
            result["confidence"] = None
        return result

    def _handle_classification_result(self, result: Dict[str, object]) -> None:
        """Forward to base handler then refresh last prediction frame canvas."""

        super()._handle_classification_result(result)  # type: ignore[misc]

        if self._last_prediction_frame is not None:
            self.last_pred_canvas.update_image(self._last_prediction_frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final GUI/front for live camera classification"
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
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Camera index to preselect at launch",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=8,
        help="How many camera indices to probe when refreshing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size_preset = args.size_preset or args.preset
    app = FinalFrontApp(
        model_path=args.model,
        preset_name=args.preset,
        size_preset_name=size_preset,
        max_probe=max(1, args.max_cameras),
        initial_camera_index=args.camera_index,
    )
    app.title("Herrajes Front Final")
    app.geometry("1400x720")
    app.mainloop()


if __name__ == "__main__":
    main()
