"""Üst düzey kütüphane API — RoboligDetector."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from src.utils.config import build_configs
from src.utils.types import FrameAnalysisResult
from src.detection.fusion import FusionDetector
from src.pipeline.assembler import ResultAssembler
from src.pipeline.visualizer import Visualizer
from src.pipeline.runtime import build_default_calibration
from src.calibration.io import load_calibration
from src.tracking import build_tracker_from_config


class RoboligDetector:
    """Küp tespiti ve konumlandırması için üst düzey API.

    Kullanım:
        detector = RoboligDetector(config_dir="config/", device="cpu")
        result = detector.detect(image)
        result, annotated = detector.detect_and_visualize(image)
    """

    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        confidence: Optional[float] = None,
        enable_tracking: Optional[bool] = None,
        tracker_type: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        config_dir = Path(config_dir) if config_dir else Path("config")
        configs = build_configs(config_dir)

        if model_path is not None or device is not None or confidence is not None:
            rf_updates: Dict[str, Any] = {}
            if model_path is not None:
                rf_updates["model_path"] = str(Path(model_path).resolve())
            if device is not None:
                rf_updates["device"] = device
            if confidence is not None:
                rf_updates["confidence_threshold"] = confidence
            new_rf = configs.detection.rf_detr.model_copy(update=rf_updates)
            configs = configs.model_copy(
                update={
                    "detection": configs.detection.model_copy(
                        update={"rf_detr": new_rf}
                    )
                }
            )

        tracking_updates: Dict[str, Any] = {}
        if enable_tracking is not None:
            tracking_updates["enabled"] = enable_tracking
        if tracker_type is not None:
            tracking_updates["tracker_type"] = tracker_type
        if tracking_updates:
            new_tracking = configs.detection.tracking.model_copy(update=tracking_updates)
            configs = configs.model_copy(
                update={"detection": configs.detection.model_copy(update={"tracking": new_tracking})}
            )

        if config_overrides:
            for dotted_key, value in config_overrides.items():
                configs = self._apply_dotted_override(configs, dotted_key, value)

        calib_path = Path(configs.camera.calibration_file)
        self._calibration = load_calibration(calib_path)
        self._synthetic_focal_length_px = configs.camera.synthetic_focal_length_px

        self._detector = FusionDetector.from_config(configs.detection)
        self._tracker = build_tracker_from_config(configs.detection.tracking)
        self._assembler = ResultAssembler(
            cube_size_m=configs.base.cube.size_m,
            use_pnp=True,
            pnp_fallback_to_geometric=True,
        )
        self._visualizer = Visualizer.from_config(configs.demo)
        self._frame_id: int = 0

    @staticmethod
    def _apply_dotted_override(configs: Any, key: str, value: Any) -> Any:
        parts = key.split(".")
        if len(parts) == 1:
            return configs.model_copy(update={parts[0]: value})
        elif len(parts) == 2:
            top, second = parts
            sub = getattr(configs, top)
            return configs.model_copy(update={top: sub.model_copy(update={second: value})})
        elif len(parts) == 3:
            top, second, third = parts
            sub = getattr(configs, top)
            sub2 = getattr(sub, second)
            new_sub = sub.model_copy(update={second: sub2.model_copy(update={third: value})})
            return configs.model_copy(update={top: new_sub})
        return configs

    def detect(self, image: np.ndarray) -> FrameAnalysisResult:
        """Görüntüdeki küpleri tespit eder ve 3D konumlarını tahmin eder."""
        self._frame_id += 1
        calibration = self._calibration or build_default_calibration(
            image.shape, self._synthetic_focal_length_px
        )
        detections = self._detector.detect(image)
        detections = self._tracker.update(image=image, detections=detections)
        return self._assembler.assemble(
            image=image,
            detections=detections,
            calibration=calibration,
            frame_id=self._frame_id,
            timestamp=time.time(),
        )

    def visualize(self, image: np.ndarray, result: FrameAnalysisResult) -> np.ndarray:
        """Tespit sonuçlarını görüntü üzerine çizer."""
        return self._visualizer.draw_results(image, result)

    def detect_and_visualize(
        self, image: np.ndarray
    ) -> tuple[FrameAnalysisResult, np.ndarray]:
        """Tespit eder ve annotated görüntüyü birlikte döndürür."""
        result = self.detect(image)
        return result, self.visualize(image, result)
