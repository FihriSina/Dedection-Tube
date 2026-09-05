"""Tüm sistem için ortak veri tipleri ve modeller"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from enum import Enum


class DetectionClass(str, Enum):
    """Algılama sınıfları"""
    CUBE = "cube"


@dataclass
class BoundingBox:
    """2D bounding box"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Detection:
    """Algılama sonucu"""
    bbox: BoundingBox
    class_id: DetectionClass
    confidence: float
    class_name: Optional[str] = None
    track_id: Optional[int] = None  # BoxMOT tarafından atanan kalıcı takip ID'si


@dataclass
class Position3D:
    """3D konum bilgisi"""
    x: float  # metre
    y: float  # metre
    z: float  # metre (derinlik)
    confidence: float  # 0-1 arası güven skoru


@dataclass
class Orientation:
    """Yönelim bilgisi (Euler açıları)"""
    roll: float   # derece
    pitch: float  # derece
    yaw: float    # derece
    confidence: float  # 0-1 arası güven skoru


@dataclass
class CubePose:
    """Küpün tam duruş bilgisi"""
    position: Position3D
    orientation: Orientation
    rvec: Optional[np.ndarray] = None  # Rodrigues rotasyon vektörü
    tvec: Optional[np.ndarray] = None  # Translasyon vektörü


@dataclass
class CubeDetectionResult:
    """Küp algılama sonucu - tüm bilgileri birleştirir"""
    detection: Detection
    pose: Optional[CubePose] = None


@dataclass
class FrameAnalysisResult:
    """Tek frame için tüm analiz sonuçları"""
    cubes: List[CubeDetectionResult]
    timestamp: float
    frame_id: int
