"""Runtime yardımcıları ve pose entegrasyonu."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.pipeline.assembler import ResultAssembler
from src.pipeline.output import OutputPublisher
from src.pipeline.visualizer import Visualizer
from src.calibration.io import load_calibration
from src.calibration.models import CameraCalibration
from src.detection.models import DetectorOutput
from src.utils.config import get_project_root
from src.utils.types import FrameAnalysisResult


class FPSCounter:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: list[float] = []

    def update(self) -> float:
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        if len(self.frame_times) < 2:
            return 0.0
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / time_diff

# PipelineRuntime, analiz sonuçlarını yayınlamak için kullanılan bir sınıftır. 
# FPS sayacı ile performansı takip eder ve OutputPublisher aracılığıyla sonuçları yayınlar. 
# Ayrıca, kapanış işlemleri için bir close yöntemi sağlar.
class PipelineRuntime:
    def __init__(
        self,
        configs: "AllConfigs",
        output_mode: Optional[str] = None,
    ):
        self.fps_counter = FPSCounter(window_size=configs.demo.performance.fps_window_size)
        self.output_publisher = OutputPublisher.from_config(configs.output, override_mode=output_mode)

    def emit(self, result: FrameAnalysisResult) -> tuple[float, Dict[str, Any]]:
        fps = self.fps_counter.update()
        payload = self.output_publisher.publish(result=result, fps=fps)
        return fps, payload

    def close(self) -> None:
        self.output_publisher.close()


def build_default_calibration(
    image_shape: tuple[int, ...],
    focal_length_px: float = 800.0,
) -> CameraCalibration:
    height, width = image_shape[:2]
    camera_matrix = np.array(
        [
            [focal_length_px, 0.0, width / 2.0],
            [0.0, focal_length_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=width,
        image_height=height,
        reprojection_error=999.0,
        calibration_date="synthetic",
        num_images=0,
    )

# PoseRuntime, verilen bir görüntü ve tespitler üzerinde küpün pozunu tahmin etmek ve 
# görsel çıktılar oluşturmak için kullanılan bir sınıftır. Yapılandırma dosyalarından 
# gerekli bileşenleri yükler, kalibrasyon bilgilerini yönetir ve analiz sonuçlarını görselleştirir.
class PoseRuntime:
    def __init__(self, config_dir: Optional[Path] = None):
        self.project_root = get_project_root()
        from src.utils.config import build_configs
        configs = build_configs(config_dir)
        self.cube_size_m = configs.base.cube.size_m
        self.focal_length_px = configs.camera.synthetic_focal_length_px
        self.calibration_path = self.project_root / Path(configs.camera.calibration_file)
        self.assembler = ResultAssembler(
            cube_size_m=self.cube_size_m,
            use_pnp=True,
            pnp_fallback_to_geometric=True,
        )
        self.visualizer = Visualizer.from_config(configs.demo)

    def get_calibration(self, image_shape: tuple[int, ...]) -> tuple[CameraCalibration, str]:
        calibration = load_calibration(self.calibration_path)
        if calibration is not None:
            return calibration, f"calib:{self.calibration_path.name}"
        return build_default_calibration(image_shape, self.focal_length_px), "calib:synthetic"

    def analyze_frame(
        self,
        image: np.ndarray,
        detections: DetectorOutput,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> tuple[FrameAnalysisResult, np.ndarray, str]:
        calibration, calibration_source = self.get_calibration(image.shape)
        result = self.assembler.assemble(
            image=image,
            detections=detections,
            calibration=calibration,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        annotated = self.visualizer.draw_results(image, result)
        return result, annotated, calibration_source