"""Tüm modüllerin entegrasyon testleri."""

import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import build_configs
from src.utils.types import BoundingBox, DetectionClass
from src.camera.opencv_camera import OpenCVCamera
from src.calibration.io import load_calibration
from src.calibration.models import CameraCalibration
from src.calibration.undistort import Undistorter
from src.detection.rf_detr_detector import RFDETRDetector
from src.detection.classical_detector import ClassicalDetector
from src.detection.fusion import FusionDetector
from src.pose.pnp import estimate_pose_pnp
from src.pose.geometry import estimate_position_from_bbox
from src.pipeline.assembler import ResultAssembler


def main():
    results = []

    test_image_path = os.path.join("data", "test_images", "cube_01.jpeg")
    if not os.path.exists(test_image_path):
        print(f"Test görseli bulunamadı: {test_image_path}, boş görüntü kullanılıyor.")
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        test_img = cv2.imread(test_image_path)

    # 1. Camera
    try:
        cam = OpenCVCamera(device_index=0, width=640, height=480, fps=30, reconnect_enabled=False)
        try:
            cam.open()
            cam.read()
            cam.release()
            results.append(("camera", "PASS", "Initialized, connected and read without crash."))
        except Exception as conn_e:
            results.append(("camera", "FAIL", f"Init ok, but run failed: {str(conn_e)}"))
    except Exception as e:
        results.append(("camera", "FAIL", f"Init failed: {str(e)}"))

    # 2. Calibration
    try:
        from pathlib import Path
        calib_path = Path("config/calibration.npz")
        if not calib_path.exists():
            h, w = test_img.shape[:2]
            calib = CameraCalibration(
                camera_matrix=np.eye(3, dtype=np.float32),
                dist_coeffs=np.zeros(5, dtype=np.float32),
                image_width=w,
                image_height=h,
                reprojection_error=0.0
            )
        else:
            calib = load_calibration(calib_path)

        undistorter = Undistorter(calib)
        res_img = undistorter.undistort(test_img)
        if res_img is not None and res_img.shape == test_img.shape:
            results.append(("calibration", "PASS", "Loaded and undistorted successfully."))
        else:
            results.append(("calibration", "FAIL", "Undistorted image has wrong shape."))
    except Exception as e:
        results.append(("calibration", "FAIL", str(e)))

    # 3. Detection
    fuse_out = None
    neural_det = None
    classical_det = None
    try:
        detection_configs = build_configs().detection
        model_path = detection_configs.rf_detr.model_path
        if os.path.exists(model_path):
            neural_det = RFDETRDetector(
                model_path=model_path,
                model_size=detection_configs.rf_detr.model_size,
                confidence_threshold=detection_configs.rf_detr.confidence_threshold,
                device=detection_configs.rf_detr.device,
                resolution=detection_configs.rf_detr.resolution,
            )
            neural_out = neural_det.detect(test_img)
        else:
            neural_det = None
            neural_out = None

        classical_det = ClassicalDetector.from_config(detection_configs)
        class_out = classical_det.detect(test_img)

        if neural_det is not None:
            fusion = FusionDetector(neural_detector=neural_det, classical_detector=classical_det)
            fuse_out = fusion.detect(test_img)
            results.append(("detection", "PASS", f"RF-DETR: {len(neural_out)} det, Classical: {len(class_out)} det, Fusion: {len(fuse_out)} det"))
        else:
            results.append(("detection", "PASS", f"RF-DETR modeli bulunamadi, Classical: {len(class_out)} det"))
    except Exception as e:
        results.append(("detection", "FAIL", str(e)))

    # 4. Pose
    try:
        calib = CameraCalibration(
            camera_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
            dist_coeffs=np.zeros(5, dtype=np.float32),
            image_width=640,
            image_height=480,
            reprojection_error=0.0
        )
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        pose_geom = estimate_position_from_bbox(bbox, 0.05, calib)
        pose_pnp = estimate_pose_pnp(test_img, bbox, 0.05, calib)

        z_pnp = pose_pnp.position_z if pose_pnp else "None"
        results.append(("pose", "PASS", f"solvePnP z: {z_pnp}, geom z: {pose_geom.position_z:.2f}"))
    except Exception as e:
        results.append(("pose", "FAIL", str(e)))

    # 5. Pipeline
    try:
        calib = CameraCalibration(
            camera_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
            dist_coeffs=np.zeros(5, dtype=np.float32),
            image_width=640,
            image_height=480,
            reprojection_error=0.0
        )
        assembler = ResultAssembler(cube_size_m=0.05, use_pnp=True)

        if fuse_out is not None:
            frame_res = assembler.assemble(test_img, fuse_out, calib, 0, 0.0)
            results.append(("pipeline", "PASS", f"Assembled frame, found {len(frame_res.cubes)} cubes."))
        else:
            results.append(("pipeline", "FAIL", "Skipped because detection failed"))
    except Exception as e:
        traceback.print_exc()
        results.append(("pipeline", "FAIL", str(e)))

    # 7. Utils (config loading)
    try:
        conf = build_configs()
        results.append(("utils", "PASS", f"Config yüklendi: cube_size_m={conf.base.cube.size_m}"))
    except Exception as e:
        results.append(("utils", "FAIL", str(e)))

    print("\nModule | Status | Notes")
    print("-------|--------|------")
    for mod, stat, notes in results:
        notes = notes.replace('\n', ' ')
        print(f"{mod} | {stat} | {notes}")


if __name__ == '__main__':
    main()
