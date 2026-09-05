"""Output payload formatını doğrular."""

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import RoboligDetector
from src.pipeline.output import frame_result_to_payload


def validate_payload(payload: dict) -> None:
    required_top = {"schema_version", "timestamp", "frame_id", "cubes", "fps"}
    assert required_top.issubset(payload.keys())
    assert isinstance(payload["timestamp"], float)
    assert isinstance(payload["frame_id"], int)
    assert isinstance(payload["cubes"], list)
    assert isinstance(payload["fps"], float)

    for cube in payload["cubes"]:
        assert {"id", "class", "confidence", "bbox", "position_cm", "orientation_deg", "distance_cm"} == set(cube.keys())
        assert {"x1", "y1", "x2", "y2"} == set(cube["bbox"].keys())
        assert {"x", "y", "z"} == set(cube["position_cm"].keys())
        assert {"yaw", "pitch", "roll"} == set(cube["orientation_deg"].keys())


def main() -> int:
    detector = RoboligDetector(config_dir="config/")

    input_dir = Path("data/test_images")
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ) if input_dir.exists() else []

    if not image_paths:
        print("Test görseli bulunamadı.")
        return 1

    processed = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        result = detector.detect(image)
        payload = frame_result_to_payload(result, fps=0.0)
        validate_payload(payload)
        processed += 1
        print(f"OK: {image_path.name}")

    print(json.dumps({"validated_frames": processed}, ensure_ascii=False))
    return 0 if processed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
