"""Command-line entry points for packaged runs."""

from __future__ import annotations

import sys
from typing import Callable, Iterable, List


def _run_with_fixed_args(main_fn: Callable[[], None], fixed_args: Iterable[str]) -> None:
    """Invoke ``main_fn`` forcing a fixed CLI argument list."""

    original_argv: List[str] = sys.argv[:]
    args = list(fixed_args)
    sys.argv = [original_argv[0], *args]
    try:
        main_fn()
    finally:
        sys.argv = original_argv


def run_classify() -> None:
    from .gui.model_viewer import main as model_viewer_main

    _run_with_fixed_args(
        model_viewer_main,
        ["--preset", "PHANSALKAR", "--size-preset", "OTSU"],
    )


def run_classify_camera() -> None:
    from .gui.camera_viewer import main as camera_viewer_main

    _run_with_fixed_args(
        camera_viewer_main,
        ["--preset", "PHANSALKAR", "--size-preset", "OTSU"],
    )


def run_tuner() -> None:
    from .gui.video_tuner import run as tuner_run

    tuner_run()
