"""Test görüntüleri üzerinde RoboligDetector API ile inference."""

import sys
import cv2
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.api import RoboligDetector

INPUT_DIR = Path("data/test_images")
OUTPUT_DIR = Path("data/test_results")
DEVICE = "cuda"
CONFIDENCE = 0.25


def main() -> int:
    detector = RoboligDetector(
        config_dir="config/",
        device=DEVICE,
        confidence=CONFIDENCE,
    )

    image_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ) if INPUT_DIR.exists() else []

    if not image_paths:
        print(f"Test görseli bulunamadı: {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append((image_path.name, "okunamadı"))
            continue

        result, annotated = detector.detect_and_visualize(image)

        out_path = OUTPUT_DIR / f"{image_path.stem}_result{image_path.suffix}"
        cv2.imwrite(str(out_path), annotated)

        summaries = []
        for cube in result.cubes:
            z = f"z={cube.pose.position.z * 100:.0f}cm" if cube.pose else "z=?"
            summaries.append(f"cube:{cube.detection.confidence:.2f} {z}")
        rows.append((image_path.name, " | ".join(summaries) or "tespit_yok"))

    print(f"\n{'Dosya':<30} Sonuçlar")
    print("-" * 70)
    for name, summary in rows:
        print(f"{name:<30} {summary}")
    print(f"\nSonuç klasörü: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
