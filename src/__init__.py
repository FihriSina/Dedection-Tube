"""TEKNOFEST 2026 Robolig AI — Küp Algılama ve Konumlandırma Sistemi"""

__version__ = "1.0.0"
__author__ = "TEKNOFEST Robolig Team"

from src.api import RoboligDetector
from src.utils.types import (
    FrameAnalysisResult,
    CubeDetectionResult,
    Detection,
    BoundingBox,
    CubePose,
    Position3D,
    Orientation,
    DetectionClass,
)

__all__ = [
    "RoboligDetector",
    "FrameAnalysisResult",
    "CubeDetectionResult",
    "Detection",
    "BoundingBox",
    "CubePose",
    "Position3D",
    "Orientation",
    "DetectionClass",
]
