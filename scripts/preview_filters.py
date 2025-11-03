"""Generate side-by-side previews of filter outputs for a given video."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.video_loader import VideoLoader, VideoLoaderError
from src.processing.pipeline import FilterParameters, FilterPipeline

PRESETS_PATH = Path("config/filter_presets.json")


def load_preset(name: Optional[str]) -> FilterParameters:
    if not name:
        return FilterParameters()
    if not PRESETS_PATH.exists():
        raise FileNotFoundError("Preset file not found. Run the GUI to create presets first.")
    with PRESETS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if name not in data:
        raise KeyError(f"Preset '{name}' not found in {PRESETS_PATH}")
    return FilterParameters.from_dict(data[name])


def create_preview_frame(original: np.ndarray, mask: np.ndarray, max_width: int) -> np.ndarray:
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    separator = np.full((original.shape[0], 8, 3), 255, dtype=np.uint8)
    combined = np.concatenate([original, separator, mask_rgb], axis=1)

    if combined.shape[1] <= max_width or max_width <= 0:
        return combined

    scale = max_width / combined.shape[1]
    new_size = (int(round(combined.shape[1] * scale)), int(round(combined.shape[0] * scale)))
    resized = cv2.resize(combined, new_size, interpolation=cv2.INTER_AREA)
    return resized


def generate_previews(
    video_path: Path,
    output_dir: Path,
    step: int,
    preset_name: Optional[str],
    max_width: int,
) -> None:
    loader = VideoLoader()
    properties = loader.open(video_path)
    pipeline = FilterPipeline()
    pipeline.set_parameters(load_preset(preset_name))
    pipeline.reset_state()

    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = properties.frame_count or 0
    if total_frames == 0:
        raise RuntimeError("Video reports zero frames; cannot generate previews")

    for frame_index in range(0, total_frames, step):
        try:
            frame = loader.read_frame(frame_index)
        except VideoLoaderError as exc:
            raise RuntimeError(f"Failed to read frame {frame_index}: {exc}") from exc
        processed = pipeline.apply(frame)
        preview = create_preview_frame(frame, processed, max_width=max_width)
        filename = output_dir / f"frame_{frame_index:05d}.png"
        cv2.imwrite(str(filename), preview)
        print(f"Saved {filename}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the video file to preview")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("previews"),
        help="Directory to save preview images (default: ./previews)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=30,
        help="Number of frames to skip between previews (default: 30)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Name of the preset to apply (default: current GUI defaults)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=640,
        help="Maximum width of the combined preview image in pixels (default: 640)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.step <= 0:
        raise ValueError("Step must be a positive integer")
    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    generate_previews(args.video, args.output, args.step, args.preset, args.max_width)


if __name__ == "__main__":
    main()
