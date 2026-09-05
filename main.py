"""Gerçek zamanlı kamera demo."""

import sys
import cv2
from pathlib import Path

from src.api import RoboligDetector
from src.camera.opencv_camera import OpenCVCamera
from src.calibration.io import load_calibration
from src.calibration.undistort import Undistorter
from src.pipeline.runtime import PipelineRuntime
from src.utils.config import build_configs
from src.utils.logging import setup_logger

CONFIG_DIR = Path("config/")


def main() -> int:
    logger = setup_logger("demo", level="INFO")
    configs = build_configs(CONFIG_DIR)

    detector = RoboligDetector(config_dir=CONFIG_DIR, device="cuda")

    calib_path = Path(configs.camera.calibration_file)
    calibration = load_calibration(calib_path)
    undistorter = Undistorter(calibration) if calibration else None

    camera = OpenCVCamera.from_config(configs.camera)
    if not camera.open():
        logger.error("Kamera açılamadı")
        return 1

    runtime = PipelineRuntime(configs=configs)
    logger.info("Demo başladı. Çıkmak için 'q' tuşuna basın.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                break

            if undistorter:
                frame = undistorter.undistort(frame)

            result, annotated = detector.detect_and_visualize(frame)
            fps, _ = runtime.emit(result)

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (annotated.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("TEKNOFEST Robolig AI", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        runtime.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
