"""Batch classification of training videos using the trained MLP and Phansalkar preset."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DEFAULT = PROJECT_ROOT / "train"
RESULTS_DIR = PROJECT_ROOT / "results"
PRESET_FILE = PROJECT_ROOT / "config" / "filter_presets.json"

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import FEATURE_NAMES, extract_features_from_mask  # noqa: E402
from src.processing.pipeline import FilterParameters, FilterPipeline  # noqa: E402


@dataclass
class ModelBundle:
    model: object
    scaler: object
    feature_names: Tuple[str, ...]
    classes_: np.ndarray


def resolve_latest_model() -> Optional[Path]:
    latest = RESULTS_DIR / "latest_model" / "model.joblib"
    if latest.exists():
        return latest
    candidates = sorted(RESULTS_DIR.glob("mlp_*"), reverse=True)
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
            raise FileNotFoundError("No trained model found. Run `make train` or specify --model.")
        model_path = latest

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = tuple(bundle.get("features", []))
    classes_ = np.array(model.classes_)
    return ModelBundle(model=model, scaler=scaler, feature_names=feature_names, classes_=classes_)


def load_preset(name: str) -> FilterParameters:
    if PRESET_FILE.exists():
        with PRESET_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and name in data:
            return FilterParameters.from_dict(data[name])
    raise FileNotFoundError(f"Preset '{name}' not found in {PRESET_FILE}")


def compute_feature_indices(bundle: ModelBundle) -> Tuple[int, ...]:
    if not bundle.feature_names:
        return tuple(range(len(FEATURE_NAMES)))
    mapping = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    indices: List[int] = []
    for name in bundle.feature_names:
        if name not in mapping:
            raise ValueError(f"Feature '{name}' missing in extractor definition")
        indices.append(mapping[name])
    return tuple(indices)


def iter_train_videos(train_dir: Path) -> Iterable[Tuple[str, Path]]:
    for label_dir in sorted(train_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for path in sorted(label_dir.glob("*.mp4")):
            yield label, path


def classify_video(
    label: str,
    video_path: Path,
    bundle: ModelBundle,
    params: FilterParameters,
    feature_indices: Tuple[int, ...],
    center_ratio: float = 0.05,
    capture_dir: Optional[Path] = None,
) -> Dict[str, object]:
    pipeline = FilterPipeline()
    pipeline.set_parameters(params)
    pipeline.reset_state()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "label": label,
            "video": str(video_path),
            "error": "unable to open",
            "predictions": [],
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    left = int(width * params.roi_left_pct)
    right = int(width * (1.0 - params.roi_right_pct))
    center_window = max(int(width * center_ratio), 20)

    last_distance: Optional[float] = None
    approaching = False
    object_present = False
    classified_current = False

    predictions: List[Dict[str, object]] = []

    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break
        processed = pipeline.apply(frame)
        roi_mask = processed[:, left:right]
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            object_present = False
            classified_current = False
            last_distance = None
            approaching = False
            continue

        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centroid_x = (moments["m10"] / moments["m00"]) + left
        frame_center = width / 2.0
        distance = abs(centroid_x - frame_center)

        if not object_present:
            object_present = True
            classified_current = False
            last_distance = distance
            approaching = False
            continue

        if last_distance is None:
            last_distance = distance
            continue

        if distance < last_distance:
            approaching = True
        else:
            if approaching and not classified_current and distance <= center_window:
                roi_binary = (roi_mask > 0).astype(np.uint8) * 255
                fv = extract_features_from_mask(roi_binary)
                values = fv.values
                selected = values[list(feature_indices)]
                features = selected.reshape(1, -1)
                scaled = bundle.scaler.transform(features)
                probs = bundle.model.predict_proba(scaled)[0]
                best_idx = int(np.argmax(probs))
                pred_label = bundle.classes_[best_idx]
                confidence = float(probs[best_idx])
                capture_paths = None
                if capture_dir is not None and pred_label != label:
                    rel_dir = capture_dir / f"{label}_as_{pred_label}" / video_path.stem
                    rel_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = rel_dir / f"raw_{frame_index:06d}.png"
                    mask_path = rel_dir / f"mask_{frame_index:06d}.png"
                    roi_path = rel_dir / f"roi_{frame_index:06d}.png"
                    cv2.imwrite(str(raw_path), frame)
                    cv2.imwrite(str(mask_path), processed)
                    cv2.imwrite(str(roi_path), roi_binary)
                    capture_paths = {
                        "raw": str(raw_path),
                        "mask": str(mask_path),
                        "roi": str(roi_path),
                    }

                predictions.append(
                    {
                        "video": str(video_path),
                        "expected": label,
                        "predicted": str(pred_label),
                        "confidence": confidence,
                        "frame": frame_index,
                        "captures": capture_paths,
                    }
                )
                classified_current = True

        last_distance = distance
        frame_index += 1

    cap.release()
    return {
        "label": label,
        "video": str(video_path),
        "predictions": predictions,
    }


def aggregate_results(results: List[Dict[str, object]], capture_dir: Optional[Path]) -> None:
    rows: List[Tuple[str, str, float, str]] = []
    missing: List[str] = []
    misclassified: List[Dict[str, object]] = []
    for entry in results:
        preds = entry.get("predictions", [])
        if not preds:
            missing.append(entry.get("video", ""))
            continue
        for pred in preds:
            rows.append((pred["expected"], pred["predicted"], pred["confidence"], pred["video"]))
            if pred["expected"] != pred["predicted"]:
                misclassified.append(pred)

    if not rows:
        print("No predictions were generated.")
        return

    y_true = np.array([r[0] for r in rows])
    y_pred = np.array([r[1] for r in rows])
    confidences = np.array([r[2] for r in rows])

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"\nTotal detections: {len(rows)}")
    print(f"Mean confidence: {confidences.mean():.3f}")
    if missing:
        print("\nVideos with no detections:")
        for video in missing:
            print(" -", video)

    if misclassified:
        print("\nMisclassifications:")
        for pred in misclassified:
            captures = pred.get("captures") or {}
            capture_info = ", ".join(f"{k}: {v}" for k, v in captures.items()) if captures else ""
            print(
                f" - {pred['video']} frame {pred['frame']} expected {pred['expected']}"
                f" predicted {pred['predicted']} ({pred['confidence']:.3f})"
                + (f" -> {capture_info}" if capture_info else "")
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch classification of training videos")
    parser.add_argument("--model", type=Path, default=None, help="Path to model.joblib (defaults to latest)")
    parser.add_argument("--preset", type=str, default="PHANSALKAR", help="Preset name from config/filter_presets.json")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DEFAULT, help="Root training directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads")
    parser.add_argument("--capture-dir", type=Path, default=None, help="Directory to save misclassified masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model(args.model)
    params = load_preset(args.preset)
    feature_indices = compute_feature_indices(bundle)

    videos = list(iter_train_videos(args.train_dir))
    if not videos:
        print("No videos found in", args.train_dir)
        return

    results: List[Dict[str, object]] = []
    capture_dir = args.capture_dir
    if capture_dir is not None:
        if not capture_dir.is_absolute():
            capture_dir = PROJECT_ROOT / capture_dir
        capture_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving misclassified frames to {capture_dir}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                classify_video,
                label,
                path,
                bundle,
                params,
                feature_indices,
                capture_dir=capture_dir,
            ): (label, path)
            for label, path in videos
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Videos"):
            result = future.result()
            results.append(result)

    aggregate_results(results, capture_dir)


if __name__ == "__main__":
    main()
