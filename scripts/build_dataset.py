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
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
    save_frames: bool = True
    png_compression: int = 3
    keep_every: int = 1
    scan_step: int = 10


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
        if not paths:
            raise RuntimeError(f"Label '{label}' has no videos")

        base = [VideoSelection(path=path, copy_index=0) for path in paths]

        if len(paths) >= target:
            if len(paths) > target:
                chosen = rng.sample(base, target)
                balanced[label] = chosen
            else:
                balanced[label] = base
            continue

        copies: Dict[Path, int] = {path: 0 for path in paths}
        selections = base.copy()
        idx = 0
        while len(selections) < target:
            path = paths[idx % len(paths)]
            copies[path] += 1
            selections.append(VideoSelection(path=path, copy_index=copies[path]))
            idx += 1
        rng.shuffle(selections)
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


def _write_pass(
    start_idx: int,
    end_idx: int,
    evaluator: Callable[[int], Optional[Tuple[bool, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]]],
    mask_dir: Path,
    frame_dir: Path,
    task: Task,
    alias: str,
    left: int,
    right: int,
    top: int,
    bottom: int,
    pass_index: int,
    manifest: List[Dict[str, object]],
    cache: Dict[int, Tuple[bool, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]],
) -> int:
    write_params = [cv2.IMWRITE_PNG_COMPRESSION, task.png_compression]
    kept = 0
    for frame_index in range(start_idx, end_idx + 1):
        data = evaluator(frame_index)
        if data is None:
            continue
        inside, roi_frame, roi_mask, bbox = data
        if not inside or bbox is None:
            continue
        if task.keep_every > 1 and (frame_index - start_idx) % task.keep_every != 0:
            continue

        mask_path = mask_dir / f"frame_{frame_index:05d}.png"
        cv2.imwrite(str(mask_path), roi_mask, write_params)

        frame_rel_path: Optional[str] = None
        if task.save_frames:
            frame_path = frame_dir / f"frame_{frame_index:05d}.png"
            cv2.imwrite(str(frame_path), roi_frame, write_params)
            frame_rel_path = str(frame_path.relative_to(task.output_root))

        entry = {
            "label": task.label,
            "video": str(task.selection.path),
            "alias": alias,
            "pass_index": pass_index,
            "frame_index": frame_index,
            "mask_path": str(mask_path.relative_to(task.output_root)),
            "bbox": bbox,
            "roi": {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
            },
        }
        if frame_rel_path is not None:
            entry["frame_path"] = frame_rel_path
        manifest.append(entry)

        cache.pop(frame_index, None)
        kept += 1
    return kept


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
    captured = 0
    skipped = 0
    manifest: List[Dict[str, object]] = []
    cache: Dict[int, Tuple[bool, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]] = {}

    def evaluate(frame_index: int) -> Optional[Tuple[bool, np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]]:
        if frame_index < 0:
            return None
        if total_frames and frame_index >= total_frames:
            return None
        if frame_index in cache:
            return cache[frame_index]
        try:
            frame_local = loader.read_frame(frame_index)
        except VideoLoaderError:
            return None
        processed_local = pipeline.apply(frame_local)
        roi_frame_local = frame_local[top:bottom, left:right]
        roi_mask_local = processed_local[top:bottom, left:right]
        inside_local, bbox_local = mask_inside_roi(roi_mask_local, ROI_MARGIN_RATIO)
        data_local = (inside_local, roi_frame_local, roi_mask_local, bbox_local)
        cache[frame_index] = data_local
        return data_local

    processed_ranges: List[Tuple[int, int]] = []
    pass_index = -1
    scan_step = max(1, task.scan_step)

    if total_frames <= 0:
        frame_index = 0
        bar = tqdm(desc=f"{task.label}:{alias}", leave=False, disable=not task.enable_frame_bar)
        while True:
            data = evaluate(frame_index)
            if data is None:
                break
            inside, *_ = data
            if not inside:
                skipped += 1
                frame_index += scan_step
                continue
            start_idx = frame_index
            while True:
                prev = evaluate(start_idx - 1)
                if prev is None or not prev[0]:
                    break
                start_idx -= 1
            end_idx = frame_index
            while True:
                nxt = evaluate(end_idx + 1)
                if nxt is None or not nxt[0]:
                    break
                end_idx += 1
            if not any(start <= frame_index <= end for start, end in processed_ranges):
                pass_index += 1
                captured += _write_pass(
                    start_idx,
                    end_idx,
                    evaluate,
                    mask_dir,
                    frame_dir,
                    task,
                    alias,
                    left,
                    right,
                    top,
                    bottom,
                    pass_index,
                    manifest,
                    cache,
                )
                processed_ranges.append((start_idx, end_idx))
            frame_index = end_idx + scan_step
        warning = "" if captured >= MIN_FRAMES_PER_VIDEO else f"Video {task.selection.path.name} yielded {captured} frames"
        loader.close()
        return {
            "label": task.label,
            "alias": alias,
            "captured": captured,
            "skipped": skipped,
            "warning": warning,
            "manifest": manifest,
        }

    bar = tqdm(
        range(0, total_frames, scan_step),
        total=(total_frames + scan_step - 1) // scan_step,
        desc=f"{task.label}:{alias}",
        leave=False,
        disable=not task.enable_frame_bar,
    )

    for frame_index in bar:
        data = evaluate(frame_index)
        if data is None:
            continue
        inside, *_ = data
        if not inside:
            skipped += 1
            continue
        if any(start <= frame_index <= end for start, end in processed_ranges):
            continue

        start_idx = frame_index
        while start_idx > 0:
            prev = evaluate(start_idx - 1)
            if prev is None or not prev[0]:
                break
            start_idx -= 1

        end_idx = frame_index
        while end_idx + 1 < total_frames:
            nxt = evaluate(end_idx + 1)
            if nxt is None or not nxt[0]:
                break
            end_idx += 1

        pass_index += 1
        captured += _write_pass(
            start_idx,
            end_idx,
            evaluate,
            mask_dir,
            frame_dir,
            task,
            alias,
            left,
            right,
            top,
            bottom,
            pass_index,
            manifest,
            cache,
        )
        processed_ranges.append((start_idx, end_idx))

    loader.close()

    if captured < MIN_FRAMES_PER_VIDEO:
        warning = f"Video {task.selection.path.name} yielded {captured} frames inside ROI"
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
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Skip exporting ROI frame PNGs (only masks will be written)",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        choices=range(10),
        help="PNG compression level (0-9, lower is faster)",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=1,
        help="Keep only every Nth frame while the object is inside the ROI",
    )
    parser.add_argument(
        "--scan-step",
        type=int,
        default=10,
        help="Initial scan stride in frames when searching for passes",
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
                    save_frames=not args.no_save_frames,
                    png_compression=args.png_compression,
                    keep_every=max(1, args.keep_every),
                    scan_step=max(1, args.scan_step),
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
