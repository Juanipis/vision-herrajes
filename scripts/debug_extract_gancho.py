"""Quick check to ensure GANCHO_BUENO_T4.mp4 yields features with current pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.json"
TARGET_VIDEO = "GANCHO_BUENO_T4.mp4"


def main() -> None:
    if not MANIFEST.exists():
        print(f"Manifest not found: {MANIFEST}")
        return

    with MANIFEST.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    matches = [entry for entry in data if entry.get("video", "").endswith(TARGET_VIDEO)]

    print(f"Entries for {TARGET_VIDEO}: {len(matches)}")
    for entry in matches[:5]:
        print(json.dumps(entry, indent=2))

    if not matches:
        print("No entries found—video likely produced no frames in dataset.")


if __name__ == "__main__":
    main()
