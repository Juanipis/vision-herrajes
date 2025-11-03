"""Build a balanced, ROI-cropped dataset of frames using the tuned preset."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "filter_presets.json"
DEFAULT_TRAIN = PROJECT_ROOT / "train"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.video_loader import VideoLoader, VideoLoaderError
from src.processing.pipeline import FilterParameters, FilterPipeline

ROI_MARGIN_RATIO = 0.04
INSIDE_STREAK = 3
MIN_FRAMES_PER_VIDEO = 5


@dataclass(frozen=True)
class VideoSelection:
    path: Path
    copy_index: int

    def alias(self, ordinal: int) -> str:
        suffix = f"copy{self.copy_index:02d}" if self.copy_index else "orig"
        return f"{self.path.stem}_{suffix}_{ordinal:02d}"


@dataclass(frozen=True)
class Task:
    label: str
    selection: VideoSelection
    ordinal: int
    output_root: Path
    preset: Dict[str, object]
    enable_frame_bar: bool = False


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOG_DIR / f"build_dataset_{timestamp}.log"

    logger = logging.getLogger("build_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.info("Logging to %s", logfile)
    return logger


def load_preset(name: str) -> FilterParameters:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Preset file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        presets = json.load(handle)
    if name not in presets:
        raise KeyError(f"Preset '{name}' not present in {CONFIG_PATH}")
    return FilterParameters.from_dict(presets[name])


def collect_videos(train_dir: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for label_dir in sorted(train_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        videos = [p for p in label_dir.iterdir() if p.is_file()]
        if videos:
            mapping[label_dir.name] = sorted(videos)
    if not mapping:
        raise RuntimeError(f"No labelled video folders found in {train_dir}")
    return mapping


def balance_videos(videos: Dict[str, List[Path]], seed: int) -> Dict[str, List[VideoSelection]]:
    rng = random.Random(seed)
    counts = {label: len(paths) for label, paths in videos.items()}
    target = max(counts.values())
    balanced: Dict[str, List[VideoSelection]] = {}

    for label, paths in videos.items():
        selections: List[VideoSelection] = []
        if len(paths) >= target:
            chosen = rng.sample(paths, target) if len(paths) > target else paths
            selections.extend(VideoSelection(path=path, copy_index=0) for path in chosen)
        else:
            copies = 0
            while len(selections) < target:
                path = rng.choice(paths)
                selections.append(VideoSelection(path=path, copy_index=copies))
                copies += 1
        balanced[label] = selections
    return balanced


def mask_inside_roi(mask: np.ndarray, margin_ratio: float) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return False, None
    x, y, w, h = cv2.boundingRect(coords)
    margin_x = int(mask.shape[1] * margin_ratio)
    margin_y = int(mask.shape[0] * margin_ratio)
    inside = (
        x >= margin_x
        and y >= margin_y
        and x + w <= mask.shape[1] - margin_x
        and y + h <= mask.shape[0] - margin_y
    )
    return inside, (x, y, w, h)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def execute_task(task: Task) -> Dict[str, object]:
    params = FilterParameters.from_dict(task.preset)
    pipeline = FilterPipeline()
    pipeline.set_parameters(params)
    pipeline.reset_state()

    loader = VideoLoader()
    try:
        properties = loader.open(task.selection.path)
    except VideoLoaderError as exc:
        return {
            "label": task.label,
            "alias": task.selection.alias(task.ordinal),
            "captured": 0,
            "skipped": 0,
            "error": str(exc),
            "manifest": [],
        }

    left = int(properties.width * params.roi_left_pct)
    right = int(properties.width * (1.0 - params.roi_right_pct))
    top = 0
    bottom = properties.height
    roi_width = right - left
    roi_height = bottom - top
    if roi_width <= 0 or roi_height <= 0:
        loader.close()
        return {
            "label": task.label,
            "alias": task.selection.alias(task.ordinal),
            "captured": 0,
            "skipped": 0,
            "error": "Computed empty ROI",
            "manifest": [],
        }

    alias = task.selection.alias(task.ordinal)
    video_dir = ensure_dir(task.output_root / task.label / alias)
    mask_dir = ensure_dir(video_dir / "mask")
    frame_dir = ensure_dir(video_dir / "frame")

    total_frames = properties.frame_count or 0
    iterable: Iterable[int]
    if total_frames > 0:
        iterable = range(total_frames)
    else:
        iterable = iter(int, 1)  # never ending, will break on read failure

    capture_started = False
    inside_streak = 0
    captured = 0
    skipped = 0
    manifest: List[Dict[str, object]] = []

    bar = tqdm(
        iterable,
        total=total_frames or None,
        desc=f"{task.label}:{alias}",
        leave=False,
        disable=not task.enable_frame_bar,
    )

    for frame_index in bar:
        try:
            frame = loader.read_frame(frame_index)
        except VideoLoaderError:
            break

        processed = pipeline.apply(frame)
        roi_frame = frame[top:bottom, left:right]
        roi_mask = processed[top:bottom, left:right]

        inside, bbox = mask_inside_roi(roi_mask, ROI_MARGIN_RATIO)

        if not capture_started:
            if inside:
                inside_streak += 1
                if inside_streak >= INSIDE_STREAK:
                    capture_started = True
            else:
                inside_streak = 0
                skipped += 1
            continue

        if not inside or bbox is None:
            break

        mask_path = mask_dir / f"frame_{frame_index:05d}.png"
        frame_path = frame_dir / f"frame_{frame_index:05d}.png"
        cv2.imwrite(str(mask_path), roi_mask)
        cv2.imwrite(str(frame_path), roi_frame)

        captured += 1
        manifest.append(
            {
                "label": task.label,
                "video": str(task.selection.path),
                "alias": alias,
                "frame_index": frame_index,
                "frame_path": str(frame_path.relative_to(task.output_root)),
                "mask_path": str(mask_path.relative_to(task.output_root)),
                "bbox": bbox,
                "roi": {
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom,
                },
            }
        )

    loader.close()

    if captured < MIN_FRAMES_PER_VIDEO:
        warning = (
            f"Video {task.selection.path.name} yielded {captured} frames inside ROI (skipped {skipped})"
        )
    else:
        warning = ""

    return {
        "label": task.label,
        "alias": alias,
        "captured": captured,
        "skipped": skipped,
        "warning": warning,
        "manifest": manifest,
    }


def write_manifest(output_root: Path, entries: List[Dict[str, object]]) -> Path:
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    return manifest_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Directory with raw labelled videos")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory to store processed frames")
    parser.add_argument("--preset", type=str, default="BASE1", help="Preset name to apply (default: BASE1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balancing videos")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of parallel workers (default: number of CPU cores)",
    )
    parser.add_argument(
        "--frame-progress",
        action="store_true",
        help="Show per-frame progress bars (only practical with --workers 1)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logger = setup_logging()

    train_dir = args.train.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Train directory: %s", train_dir)
    logger.info("Output directory: %s", output_root)
    logger.info("Workers: %d", args.workers)

    videos = collect_videos(train_dir)
    logger.info("Detected labels: %s", ", ".join(sorted(videos)))
    counts = {label: len(paths) for label, paths in videos.items()}
    logger.info("Video count per label: %s", counts)

    balanced = balance_videos(videos, seed=args.seed)
    balanced_counts = {label: len(sel) for label, sel in balanced.items()}
    logger.info("Balanced selections per label: %s", balanced_counts)

    params = load_preset(args.preset)
    preset_dict = params.to_dict()
    logger.info("Loaded preset '%s'", args.preset)

    tasks: List[Task] = []
    for label, selections in balanced.items():
        for ordinal, selection in enumerate(selections):
            tasks.append(
                Task(
                    label=label,
                    selection=selection,
                    ordinal=ordinal,
                    output_root=output_root,
                    preset=preset_dict,
                    enable_frame_bar=args.frame_progress and args.workers == 1,
                )
            )

    manifest_entries: List[Dict[str, object]] = []

    if args.workers == 1:
        for task in tqdm(tasks, desc="Videos"):
            result = execute_task(task)
            if result.get("warning"):
                logger.warning(result["warning"])  # type: ignore[index]
            if result.get("error"):
                logger.error("Failed %s/%s: %s", task.label, task.selection.path.name, result["error"])
                continue
            logger.info(
                "Processed %s/%s -> %d frames (skipped %d)",
                task.label,
                task.selection.path.name,
                result["captured"],
                result["skipped"],
            )
            manifest_entries.extend(result["manifest"])  # type: ignore[index]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(execute_task, task): task for task in tasks}
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Videos"):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Worker crashed on %s/%s: %s", task.label, task.selection.path.name, exc)
                    continue
                if result.get("warning"):
                    logger.warning(result["warning"])  # type: ignore[index]
                if result.get("error"):
                    logger.error("Failed %s/%s: %s", task.label, task.selection.path.name, result["error"])
                    continue
                logger.info(
                    "Processed %s/%s -> %d frames (skipped %d)",
                    task.label,
                    task.selection.path.name,
                    result["captured"],
                    result["skipped"],
                )
                manifest_entries.extend(result["manifest"])  # type: ignore[index]

    if manifest_entries:
        manifest_path = write_manifest(output_root, manifest_entries)
        logger.info("Manifest written: %s (%d entries)", manifest_path, len(manifest_entries))
    else:
        logger.warning("No frames captured; manifest not written")


if __name__ == "__main__":
    main()
