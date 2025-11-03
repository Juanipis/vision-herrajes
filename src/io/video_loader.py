"""Utilities for loading video files frame-by-frame with HEVC support."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

try:  # Optional dependency for HEVC fallback
    import av  # type: ignore
except ImportError:  # pragma: no cover - PyAV may be unavailable during development
    av = None


class VideoLoaderError(RuntimeError):
    """Raised when a video cannot be opened or read."""


@dataclass
class VideoProperties:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


class _BaseBackend:
    def read_frame(self, index: Optional[int] = None):  # pragma: no cover - interface definition
        raise NotImplementedError

    def release(self):  # pragma: no cover - interface definition
        raise NotImplementedError


class _OpenCVBackend(_BaseBackend):
    def __init__(self, capture: cv2.VideoCapture):
        self._capture = capture

    def read_frame(self, index: Optional[int] = None):
        if index is not None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        success, frame = self._capture.read()
        if not success or frame is None:
            raise VideoLoaderError("Could not read frame from video")
        return frame

    def release(self):
        self._capture.release()


class _PyAvBackend(_BaseBackend):
    """Fallback backend using PyAV. Loads frames into memory for random access."""

    def __init__(self, path: Path):
        if av is None:
            raise VideoLoaderError(
                "PyAV is not installed. Install 'av' to enable HEVC fallback decoding."
            )
        container = av.open(str(path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        self._fps = float(stream.average_rate) if stream.average_rate else 30.0
        self._frames = [frame.to_ndarray(format="bgr24") for frame in container.decode(stream)]
        container.close()
        if not self._frames:
            raise VideoLoaderError("No frames were decoded from the video")

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def read_frame(self, index: Optional[int] = None):
        if index is None:
            index = 0
        if index < 0 or index >= len(self._frames):
            raise VideoLoaderError("Frame index out of range for cached video")
        return self._frames[index]

    def release(self):
        self._frames.clear()


class VideoLoader:
    """Video loader that supports HEVC playback for the tuning GUI."""

    def __init__(self) -> None:
        self._backend: Optional[_BaseBackend] = None
        self._properties: Optional[VideoProperties] = None

    @property
    def properties(self) -> VideoProperties:
        if self._properties is None:
            raise VideoLoaderError("No video loaded. Call 'open' first.")
        return self._properties

    def open(self, video_path: str | Path) -> VideoProperties:
        path = Path(video_path).expanduser().resolve()
        if not path.exists():
            raise VideoLoaderError(f"Video not found: {path}")

        self.close()

        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            backend: _BaseBackend = _OpenCVBackend(capture)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            capture.release()
            backend = _PyAvBackend(path)
            if isinstance(backend, _PyAvBackend):
                frame_count = backend.frame_count
                fps = backend.fps
                sample = backend.read_frame(0)
                height, width = sample.shape[:2]
            else:
                frame_count = 0
                fps = 30.0
                sample = backend.read_frame(0)
                height, width = sample.shape[:2]

        if frame_count <= 0:
            frame_count = self._estimate_frame_count(backend)

        self._backend = backend
        self._properties = VideoProperties(path=path, frame_count=frame_count, fps=fps, width=width, height=height)
        return self._properties

    def _estimate_frame_count(self, backend: _BaseBackend) -> int:
        total = 0
        try:
            while True:
                backend.read_frame(total)
                total += 1
        except VideoLoaderError:
            pass
        finally:
            if total:
                try:
                    backend.read_frame(0)
                except VideoLoaderError:
                    pass
        return total

    def read_frame(self, index: Optional[int] = None):
        if self._backend is None:
            raise VideoLoaderError("No video loaded. Call 'open' first.")
        return self._backend.read_frame(index)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.release()
        self._backend = None
        self._properties = None
