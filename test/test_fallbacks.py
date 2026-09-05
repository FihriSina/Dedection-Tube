"""Tüm fallback/yedek yöntem testleri."""

import sys
import traceback
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import build_configs
from src.calibration.models import CameraCalibration
from src.detection.classical_detector import ClassicalDetector
from src.detection.fusion import FusionDetector
from src.detection.models import DetectorOutput
from src.pose.pnp import estimate_pose_pnp
from src.pose.geometry import estimate_position_from_bbox
from src.pose.orientation import estimate_orientation_from_bbox
from src.pose.corners import estimate_cube_face_corners
from src.pipeline.output import OutputPublisher
from src.pipeline.assembler import ResultAssembler
from src.utils.types import BoundingBox, DetectionClass

img = cv2.imread("data/test_images/cube_01.jpeg")
if img is None:
    print("cube_01.jpeg bulunamadi!")
    raise SystemExit(1)

bbox = BoundingBox(x1=200, y1=300, x2=500, y2=600)
cube_size = 0.09

default_cal = CameraCalibration(
    camera_matrix=np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32),
    dist_coeffs=np.zeros(5, dtype=np.float32),
    image_width=640,
    image_height=480,
    reprojection_error=0.0
)

results = []


def _run_test(name, func):
    try:
        func()
        results.append((name, "PASS", "OK"))
        print(f"  {name}: PASS")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, "FAIL", str(e)))
        print(f"  {name}: FAIL - {e}")
        print(tb[:500])


def main():
    print("\n=== FALLBACK TESTLERI ===\n")

    # 1. ClassicalDetector tek basina
    print("1. ClassicalDetector tek basina:")
    def t1():
        cd = ClassicalDetector(min_area=100, max_area=50000)
        out = cd.detect(img)
        print(f"    Tespit sayisi: {len(out)}")
    _run_test("classical_detector", t1)

    # 2. Fusion fallback (neural bos)
    print("\n2. Fusion fallback (neural bos):")
    def t2():
        neural = ClassicalDetector(min_area=100, max_area=50000)  # neural yerine dummy
        classical = ClassicalDetector(min_area=100, max_area=50000)
        fusion = FusionDetector(neural_detector=neural, classical_detector=classical, fallback_if_neural_fails=True)
        out = fusion.detect(img)
        print(f"    Fusion tespit: {len(out)}")
    _run_test("fusion_fallback", t2)

    # 3. solvePnP fallback to geometry
    print("\n3. solvePnP fallback (yetersiz nokta):")
    def t3():
        pose = estimate_pose_pnp(img, bbox, cube_size, default_cal)
        if pose:
            print(f"    solvePnP z={pose.position_z:.3f}m")
        else:
            print("    solvePnP None dondu (fallback gerekiyor)")
        geom = estimate_position_from_bbox(bbox, cube_size, default_cal)
        print(f"    geometry z={geom.position_z:.3f}m")
    _run_test("pose_fallback", t3)

    # 4. Orientation
    print("\n4. Orientation fallback:")
    def t4():
        roll, pitch, yaw = estimate_orientation_from_bbox(bbox, img.shape[:2])
        print(f"    Roll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}")
    _run_test("orientation", t4)

    # 5. Corners - net kose olmayan gorsel
    print("\n5. Corners (net kose olmayan):")
    def t5():
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 200
        corners = estimate_cube_face_corners(blank, bbox)
        print(f"    Corners: {corners}")
    _run_test("corners_blank", t5)

    # 6. ResultAssembler - PnP fallback to geometric
    print("\n6. ResultAssembler (PnP -> geometric fallback):")
    def t6():
        assembler = ResultAssembler(
            cube_size_m=cube_size,
            use_pnp=True,
            pnp_fallback_to_geometric=True
        )
        dets = DetectorOutput(
            bboxes=[bbox],
            classes=[DetectionClass.CUBE],
            confidences=[0.95],
            class_names=["cube"]
        )
        result = assembler.assemble(img, dets, default_cal, frame_id=1)
        print(f"    Cubes: {len(result.cubes)}")
        if result.cubes and result.cubes[0].pose:
            print(f"    z={result.cubes[0].pose.position.z:.3f}m")
    _run_test("assembler_fallback", t6)

    # 7. OutputPublisher modlari
    print("\n7. OutputPublisher (serial/socket bagli degil):")
    def t7():
        for mode in ["stdout", "jsonl"]:
            pub = OutputPublisher({"output_mode": mode, "jsonl_path": "output/test.jsonl"})
            print(f"    {mode}: OK")
        pub_ser = OutputPublisher({"output_mode": "serial", "serial_port": "COM99", "serial_baud": 115200})
        pub_sock = OutputPublisher({"output_mode": "socket", "socket_host": "localhost", "socket_port": 99999})
        print("    serial: OK (bagli degil)")
        print("    socket: OK (bagli degil)")
    _run_test("output_modes", t7)

    print("\n" + "=" * 60)
    print("OZET TABLO")
    print("=" * 60)
    print(f"{'Test':<35} {'Durum':<8} {'Notlar'}")
    print("-" * 60)
    for name, status, note in results:
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"{icon:<6} {name:<33} {status:<8} {note}")
    print("-" * 60)
    pass_count = sum(1 for _, s, _ in results if s == "PASS")
    fail_count = len(results) - pass_count
    print(f"Toplam: {len(results)} test | {pass_count} PASS | {fail_count} FAIL")


if __name__ == "__main__":
    main()
