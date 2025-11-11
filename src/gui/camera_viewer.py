"""GUI variant for running the classifier directly from a live camera feed."""

from __future__ import annotations

import argparse
import queue
import time
import tkinter as tk
from pathlib import Path
from typing import List, Optional

import cv2

from .model_viewer import DEFAULT_PRESET_NAME, VideoClassifierApp


class CameraClassifierApp(VideoClassifierApp):
    def __init__(
        self,
        model_path: Optional[Path],
        preset_name: Optional[str],
        size_preset_name: Optional[str],
        *,
        max_probe: int = 8,
        initial_camera_index: Optional[int] = None,
    ) -> None:
        self._max_probe = max(1, max_probe)
        self._initial_camera_index = initial_camera_index
        self._camera_indices: List[int] = []
        self._camera_index_var: Optional[tk.StringVar] = None
        self._camera_menu: Optional[tk.OptionMenu] = None
        super().__init__(
            model_path=model_path,
            preset_name=preset_name,
            size_preset_name=size_preset_name,
        )
        if self._initial_camera_index is not None:
            self.after(200, self._auto_connect_initial)

    def _post_setup(self) -> None:
        super()._post_setup()
        discovered = self._discover_cameras()
        if (
            self._initial_camera_index is not None
            and self._initial_camera_index not in discovered
        ):
            discovered.append(self._initial_camera_index)
        self._camera_indices = sorted(set(discovered))
        if self._camera_indices:
            default_value = str(
                self._initial_camera_index
                if self._initial_camera_index in self._camera_indices
                else self._camera_indices[0]
            )
        else:
            default_value = "No cameras"
        self._camera_index_var = tk.StringVar(value=default_value)

    def _build_source_controls(self, control_bar: tk.Frame) -> None:
        tk.Label(control_bar, text="Camera").pack(side="left", padx=(0, 4))
        values = self._camera_display_values()
        # Ensure the string var always has a value present in the menu.
        if self._camera_index_var is not None and self._camera_index_var.get() not in values:
            self._camera_index_var.set(values[0])
        self._camera_menu = tk.OptionMenu(
            control_bar,
            self._camera_index_var,
            *values,
        )
        self._camera_menu.config(width=10)
        self._camera_menu.pack(side="left", padx=4)

        tk.Button(control_bar, text="Refresh", command=self._refresh_cameras).pack(
            side="left", padx=4
        )
        tk.Button(control_bar, text="Connect", command=self._connect_camera).pack(
            side="left", padx=4
        )

    def _camera_display_values(self) -> List[str]:
        if not self._camera_indices:
            return ["No cameras"]
        return [str(idx) for idx in self._camera_indices]

    def _refresh_cameras(self) -> None:
        current_value = self._camera_index_var.get() if self._camera_index_var else ""
        discovered = self._discover_cameras()
        if (
            self._initial_camera_index is not None
            and self._initial_camera_index not in discovered
        ):
            discovered.append(self._initial_camera_index)
        self._camera_indices = sorted(set(discovered))
        values = self._camera_display_values()

        if self._camera_index_var is not None:
            if current_value in values:
                self._camera_index_var.set(current_value)
            elif values:
                self._camera_index_var.set(values[0])

        self._rebuild_camera_menu(values)
        if not self._camera_indices and hasattr(self, "status_var"):
            self.status_var.set("No cameras detected")

    def _rebuild_camera_menu(self, values: List[str]) -> None:
        if self._camera_menu is None or self._camera_index_var is None:
            return
        menu = self._camera_menu["menu"]
        menu.delete(0, "end")
        for value in values:
            menu.add_command(
                label=value,
                command=lambda v=value: self._camera_index_var.set(v),
            )

    def _connect_camera(self) -> bool:
        if self._camera_index_var is None:
            return False
        selection = self._camera_index_var.get()
        try:
            index = int(selection)
        except ValueError:
            self.status_var.set("Select a camera input")
            return False

        self._stop_playback()
        if self.video_capture is not None:
            self.video_capture.release()

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            self.video_capture = None
            self.status_var.set(f"Failed to open camera {index}")
            return False

        self.video_capture = cap
        self.video_path = None
        self.status_var.set(f"Connected to camera {index}")
        self.result_var.set("")
        self.size_result_var.set("")
        return True

    def _start_playback(self) -> None:
        if self.video_capture is None or not self.video_capture.isOpened():
            if not self._connect_camera():
                return
        super()._start_playback()

    def _reader_loop(self) -> None:
        assert self.video_capture is not None
        cap = self.video_capture
        self.pipeline.reset_state()
        self._otsu_pipeline.reset_state()
        frame_idx = 0
        while not self._stop_event.is_set():
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue
            processed = self.pipeline.apply(frame)
            try:
                self._frame_queue.put((frame, processed, frame_idx), timeout=0.05)
            except queue.Full:
                pass
            else:
                frame_idx += 1
        self._stop_event.set()

    def _auto_connect_initial(self) -> None:
        if self._initial_camera_index is None or self._camera_index_var is None:
            return
        target = str(self._initial_camera_index)
        if target not in self._camera_display_values():
            return
        self._camera_index_var.set(target)
        self._connect_camera()

    def _discover_cameras(self) -> List[int]:
        indices: List[int] = []
        for idx in range(self._max_probe):
            cap = cv2.VideoCapture(idx)
            if cap is None:
                continue
            if cap.isOpened():
                indices.append(idx)
            cap.release()
        return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive classification viewer for live camera feeds"
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
    app = CameraClassifierApp(
        model_path=args.model,
        preset_name=args.preset,
        size_preset_name=size_preset,
        max_probe=max(1, args.max_cameras),
        initial_camera_index=args.camera_index,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
